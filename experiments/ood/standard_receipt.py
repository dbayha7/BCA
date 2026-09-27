"""Bind standard-runner process metadata; never accept a scientific checkpoint.

Use explicit paths on the execution host. No directory search, model loading,
event decoding, simulator calls, training, or automatic retry occurs here.
"""

import argparse
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re

MANIFEST_SHA256 = "13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def integer_zero(value):
    return type(value) is int and value == 0


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def exact_json(left, right):
    # Python equality conflates True/1 and False/0; scientific settings do not.
    options = dict(sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.dumps(left, **options) == json.dumps(right, **options)


def bind_receipt(manifest_path, lane_directory, source_directory, run_id):
    """Return an OOD process receipt with all scientific acceptance still pending."""
    try:
        payload = Path(manifest_path).read_bytes()
        if payload.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(payload)
        require(sha(payload) == MANIFEST_SHA256, "Wrong frozen standard-study manifest.")
        manifest = json.loads(payload)
        matches = [x for x in manifest["runs"] if x["row"]["run_id"] == run_id]
        require(len(matches) == 1, "Missing or ambiguous declared run.")
        item = matches[0]
        require(item["algorithm"] in ("td3_bc", "rebrac")
                and item["dataset"] in ("hopper", "walker2d"),
                "Outside the current checkpoint/simulator adapter scope.")
        require(manifest["calibration_weighting"] == "none", "Expected standard no-IW study.")
        lane, source = Path(lane_directory).resolve(), Path(source_directory).resolve()
        run, attempt = lane / "runs" / run_id, lane / "attempts" / run_id
        hashes = {}

        def read(role, path):
            data = path.read_bytes()
            hashes[role] = dict(path=str(path), sha256=sha(data))
            return json.loads(data)

        dispatch = read("dispatch", attempt / "dispatch.json")
        actual = read("actual_exit", attempt / "actual_exit.json")
        command = dispatch["command"]
        require(isinstance(command, list) and all(isinstance(x, str) for x in command)
                and len(command) == 18 and bool(command[0]), "Malformed worker command.")
        require(Path(command[1]).is_absolute() and Path(command[1]).resolve() == source / "train.py",
                "Worker did not name the declared frozen train.py.")
        flags = command[2::2]
        require(len(set(flags)) == len(flags), "Repeated worker option.")
        options = dict(zip(flags, command[3::2]))
        require(set(options) == {"--algorithm", "--dataset", "--method", "--seed",
                                "--output", "--device", "--lock", "--data-dir"},
                "Missing or unknown worker option.")
        require(all(options["--" + key] == str(item[key])
                    for key in ("algorithm", "dataset", "method", "seed")),
                "Dispatch does not match the frozen scientific row.")
        require(options["--device"] == "cuda" and options["--lock"] and options["--data-dir"],
                "Wrong device or absent data/lock argument.")
        require(Path(options["--output"]).is_absolute()
                and Path(options["--output"]).resolve() == run,
                "Dispatch belongs to a different output directory.")
        require(type(dispatch["pid"]) is int and dispatch["pid"] > 0, "Invalid dispatch PID.")
        require(integer_zero(actual["actual_returncode"]) and actual["timeout"] is False
                and actual["interruption_signal"] is None, "Worker did not close successfully.")
        require(actual["started"] == dispatch["started"], "Stale actual worker exit.")
        start, end = (datetime.fromisoformat(actual[k]) for k in ("started", "ended"))
        require(start.utcoffset() is not None and end.utcoffset() is not None and end >= start,
                "Missing timezone or reversed worker times.")
        resolved = read("resolved", run / "resolved.json")
        require(exact_json(resolved, item["row"]), "Resolved row differs from the frozen declaration.")
        reported_source = read("source", run / "source.json")
        require(all(reported_source.get(k) == v for k, v in manifest["source_sha256"].items()),
                "Reported source pins differ from the frozen source.")
        learner = read("learner_exit", run / "exit.json")
        require(learner["completed"] is True and integer_zero(learner["exit_code"]),
                "Learner completion is absent or failed.")
        result = read("result", run / "result.json")
        require(result["completed"] is True and result["run_id"] == run_id
                and type(result["steps_completed"]) is int and result["steps_completed"] == 1000000
                and exact_json(result["expected_counts"], resolved["expected_counts"]),
                "Incomplete or wrong-budget result.")
        checkpoints = result["checkpoints"]
        require([c["step"] for c in checkpoints] == resolved["expected_counts"]["checkpoint_steps"]
                and all(type(c["step"]) is int and valid_hash(c["sha256"])
                        and c["path"] == f"checkpoint_{c['step']}.msgpack" for c in checkpoints),
                "Incomplete or malformed reported checkpoint identities.")
        require(checkpoints[-1]["step"] == 1000000 and valid_hash(result["events_sha256"]),
                "Missing final checkpoint or event identity.")
    except (OSError, KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Incomplete or malformed standard-run metadata: " + str(exc)) from exc
    return dict(schema="ood-process-exit-v1", run_id=run_id, exit_code=0,
        result_sha256=hashes["result"]["sha256"], manifest_sha256=MANIFEST_SHA256,
        supporting_metadata=hashes, started=actual["started"], ended=actual["ended"],
        reported_final_checkpoint_sha256=checkpoints[-1]["sha256"],
        status="pending", process_metadata_checked=True, checkpoint_decoded=False,
        events_verified=False, data_contents_verified=False, source_contents_verified=False,
        ready_for_collection=False,
        remaining="Independently verify all checkpoints, counters, events, source/data contents, "
                  "paired preparation, saved actions and simulator gates before OOD collection.")


def write_receipt(path, receipt):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--lane-directory", required=True, type=Path)
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = bind_receipt(args.manifest, args.lane_directory, args.source_directory, args.run_id)
    write_receipt(args.output, receipt)
    print(json.dumps(dict(run_id=args.run_id, status=receipt["status"], ready_for_collection=False)))


if __name__ == "__main__":
    main()
