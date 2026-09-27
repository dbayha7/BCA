"""Audit the first standard-study TD3+BC Hopper host from saved data, on CPU.

This script has one explicit run identity and never calls a learner, model or
simulator. Run under the recorded Flax/numpy/h5py environment on the execution host.
"""
from pathlib import Path
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import numpy as np
from flax import serialization

RUN_ID = "td3_bc-hopper-host-s202609171"
MANIFEST = "13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a = np.asarray(value)
    h = hashlib.sha256(json.dumps([a.dtype.str, a.shape]).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def audit(root, repo, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    lane = root / "queue-local-v1"
    run = lane / "runs" / RUN_ID
    require(sha(source / "manifest.json") == MANIFEST, "Frozen manifest changed.")
    manifest = read(source / "manifest.json")
    item, = [x for x in manifest["runs"] if x["row"]["run_id"] == RUN_ID]
    row = item["row"]
    for name, wanted in manifest["source_sha256"].items():
        require(sha(source / name) == wanted, "Changed frozen source: " + name)
    sys.path.insert(0, str(repo))
    from experiments.ood.standard_receipt import bind_receipt
    receipt = bind_receipt(source / "manifest.json", lane, source, RUN_ID)
    save(out / "process_exit.json", receipt)
    result = read(run / "result.json")
    queue = read(lane / "queue_status.json")
    closed, = [x for x in queue["completed"] if x["run_id"] == RUN_ID]
    require(closed["result_sha256"] == sha(run / "result.json"), "Queue/result mismatch.")
    actual = read(lane / "attempts" / RUN_ID / "actual_exit.json")
    prep = read(run / "preparation.json")
    require(prep["accepted"] is True and prep["learner_updates"] == 0, "Preparation failed.")
    m = prep["metadata"]
    for name, wanted in manifest["source_sha256"].items():
        require(m["source_files"][name] == wanted, "Preparation source mismatch.")
    require(m["settings"]["args"] == {**row["native_args"], "wandb_project": "unifloral",
            "wandb_team": "flair", "wandb_group": "debug"}, "Native arguments/defaults changed.")
    require(m["settings"]["specification"] == row["specification"], "Method changed.")
    for key, value in row["protocol"].items():
        require(m["settings"]["protocol"][key] == value, "Protocol changed: " + key)
    require(digest(m["settings"]) == m["settings_sha256"], "Settings digest mismatch.")
    cache = Path(m["raw_identity"]["path"])
    require(sha(cache) == row["cache"]["sha256"] == m["raw_identity"]["sha256"],
            "Actual dataset cache mismatch.")
    with h5py.File(cache, "r") as stream:
        raw = {k: stream[k][()] for k in m["raw_array_hashes"]}
    require({k: array_hash(v) for k, v in raw.items()} == m["raw_array_hashes"],
            "Raw array identity mismatch.")
    ids = np.asarray(m["dependency_maps"]["raw_current"], np.int64)
    expected_ids = np.flatnonzero(~raw["timeouts"][:-1].astype(bool))
    require(np.array_equal(ids, expected_ids), "Native conversion row mismatch.")
    converted = dict(observations=raw["observations"][ids].astype(np.float32),
                     next_observations=raw["observations"][ids + 1].astype(np.float32),
                     actions=raw["actions"][ids].astype(np.float32),
                     rewards=raw["rewards"][ids].astype(np.float32),
                     terminals=raw["terminals"][ids].astype(bool))
    require({k: array_hash(v) for k, v in converted.items()} == m["converted_array_hashes"],
            "Converted array identities differ.")
    train_ids = np.asarray(m["training_converted_ids"], np.int64)
    hold_ids = np.asarray(m["heldout_converted_ids"], np.int64)
    require(len(train_ids) == 998895 and len(hold_ids) == 1103
            and not m["reference_converted_ids"], "Unexpected data partition.")
    require(np.array_equal(np.sort(np.r_[train_ids, hold_ids]), np.arange(len(ids))),
            "Training/holdout rows are not a disjoint exhaustive partition.")
    obs = converted["observations"]
    mean, std = obs[train_ids].mean(0), obs[train_ids].std(0) + .001
    require(np.array_equal(mean, np.asarray(m["obs_mean"], np.float32))
            and np.array_equal(std, np.asarray(m["obs_std"], np.float32)),
            "Training-only observation normalization differs.")
    require(m["reward_normalization"] == "disabled", "Unexpected reward transformation.")
    arrays = [(obs - mean) / std, converted["actions"], converted["rewards"],
              (converted["next_observations"] - mean) / std,
              converted["terminals"].astype(np.float32)]
    for key, indices in [("training", train_ids), ("heldout", hold_ids)]:
        require(digest([array_hash(x[indices]) for x in arrays]) == m["run_input_hashes"][key],
                "Reconstructed prepared input identity differs: " + key)
    accepted = read(repo / "docs/validation/standard-bca-preflight.json")
    cell, = [x for x in accepted["data_preparation"]["cells"]
             if (x["host"], x["dataset"]) == ("td3_bc", "hopper")]
    require(m["run_input_hashes"]["training"] == cell["paired_data_fingerprints"][0],
            "Accepted paired training fingerprint differs.")
    del raw, converted, arrays, obs
    # Read counters directly; never initialize an actor or query checkpoint weights.
    sys.path.insert(0, str(source))
    from runtime.validation import checkpoint_counts, counters
    checkpoints = []
    for c in result["checkpoints"]:
        path = run / c["path"]
        require(sha(path) == c["sha256"], "Checkpoint file hash differs.")
        tree = serialization.msgpack_restore(path.read_bytes())
        counts = checkpoint_counts(tree, "td3", c["step"], 2)
        require(counts == c["counters"], "Reported counters differ from saved bytes.")
        state = tree["state"]
        require(all(state[k] is None for k in ["calibrator", "posterior", "residual_scale"]),
                "Plain host contains calibration state.")
        targets = {}
        for name in ["actor_target", "critic_target"]:
            target = counters(state["native"][name])
            require(target["/step"] == c["step"] // 2
                    and all(v == 0 for k, v in target.items() if k.endswith("/count")),
                    "Target counters differ.")
            targets[name] = target
        def finite(value):
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            return value is None or bool(np.isfinite(np.asarray(value)).all())
        require(finite(tree), "Nonfinite checkpoint state.")
        checkpoints.append({**c, "decoded_counters": counts, "target_counters": targets,
                            "training_rng": np.asarray(tree["training_rng"]).tolist()})
        del tree
    require(sha(run / "events.jsonl.gz") == result["events_sha256"], "Journal hash mismatch.")
    kinds = collections.Counter()
    blocks, evaluations = [], []
    previous = 0
    actor_fields = ["q_mean", "lambda", "actor_loss", "bc_loss"]
    with gzip.open(run / "events.jsonl.gz", "rt") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["kind"]
            kinds[kind] += 1
            require(event["execution"] == {"method": "host"}, "Journal method mismatch.")
            if kind == "prepared":
                require(event["step"] == 0 and event["metadata"] == m, "Prepared journal mismatch.")
            elif kind == "accepted_scan":
                step = event["step"]
                require(step == previous + 1000 and event["posterior_snapshot_sha256"] is None,
                        "Scan boundary or posterior mismatch.")
                metrics = event["metrics"]
                require(set(metrics) == set(actor_fields + ["critic_loss", "inputs_valid"]),
                        "Unexpected host metrics.")
                values = {k: np.asarray(v) for k, v in metrics.items()}
                require(all(v.shape == (1000,) and np.isfinite(v).all() for v in values.values())
                        and values["inputs_valid"].all(), "Invalid scan rows.")
                active = np.arange(previous + 1, step + 1) % 2 == 0
                block = dict(step=step, host_rows=1000, actor_rows=int(active.sum()))
                for key in actor_fields + ["critic_loss"]:
                    selected = values[key][active] if key in actor_fields else values[key]
                    if key in actor_fields:
                        require((values[key][~active] == 0).all(), "Unexpected skipped placeholder.")
                    block[key] = dict(mean=float(selected.mean()), min=float(selected.min()),
                                      max=float(selected.max()), zeros=int((selected == 0).sum()))
                blocks.append(block)
                previous = step
            elif kind in ["periodic", "final"]:
                require(event["step"] == previous, "Evaluation precedes its accepted updates.")
                evaluations.append(event)
            elif kind == "completed":
                require(event["step"] == previous == 1000000
                        and event["final_episodes_actual"] == 20,
                        "Invalid completed journal event.")
            else:
                raise ValueError("Unexpected host journal event: " + kind)
    require(kinds == dict(prepared=1, accepted_scan=1000, periodic=200, final=1, completed=1)
            and previous == 1000000 and kind == "completed", "Incomplete journal.")
    require([{k: v for k, v in e.items() if k != "execution"} for e in evaluations]
            == result["evaluations"], "Result/journal evaluation disagreement.")
    for bank, wanted in zip(evaluations, row["protocol"]["evaluation_events"]):
        require(all(bank[k] == wanted[k] for k in ["kind", "step", "episode_seeds"]),
                "Evaluation bank identity differs.")
        require(bank["episode_count"] == len(bank["episodes"]) == len(wanted["episode_seeds"]),
                "Evaluation episode count differs.")
        require(not bank["repeated_episode_seeds"], "Unexpected repeated seeds.")
        transform = bank["score_transform"]
        require(transform == m["evaluation_score_transform"], "Score transform differs.")
        for i, e in enumerate(bank["episodes"]):
            require(e["seed"] == wanted["episode_seeds"][i] and e["episode_index"] == i
                    and 0 < e["length"] <= 1000, "Episode identity/length mismatch.")
            score = 100 * (e["raw_return"] - transform["reference_min"]) / (
                    transform["reference_max"] - transform["reference_min"])
            require(np.isfinite([score, e["normalized_score"]]).all()
                    and abs(score - e["normalized_score"]) < 1e-10, "Score arithmetic differs.")
    final = np.asarray([e["normalized_score"] for e in evaluations[-1]["episodes"]])
    curve = [float(np.mean([e["normalized_score"] for e in bank["episodes"]]))
             for bank in evaluations[:-1]]
    summary = dict(final_mean=float(final.mean()), final_episode_median=float(np.median(final)),
                   final_episode_sd=float(final.std(ddof=1)), final_episode_min=float(final.min()),
                   final_episode_max=float(final.max()), periodic_curve_mean=float(np.mean(curve)),
                   final_checkpoint_periodic_mean=curve[-1], training_seeds=1,
                   training_seed_uncertainty=None, bca_comparison=None)
    evidence_files = ["resolved.json", "source.json", "preparation.json", "result.json",
                      "exit.json", "events.jsonl.gz"]
    audit = dict(schema="standard-td3-hopper-host-audit-v1", accepted=True, run_id=RUN_ID,
                 checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 manifest_sha256=MANIFEST, frozen_source_files_checked=108,
                 scientific_source_commit="6e912b0ec3e3346ba6c628b6eef6fddedc74ffc0",
                 actual_worker_exit=actual, host_updates=1000000, critic_updates=1000000,
                 actor_updates=500000, target_updates=500000, checkpoints=checkpoints,
                 journal_counts=dict(kinds), evaluation_episodes=2020,
                 training_rows=len(train_ids), holdout_rows=len(hold_ids), reference_rows=0,
                 data_cache_sha256=row["cache"]["sha256"], run_input_hashes=m["run_input_hashes"],
                 calibrator_fitting_operations=0, posterior_refreshes=0,
                 accepted_abstained_scale_fits=None, scale=None, radii=None,
                 summary=summary, model_queries_added=0, simulator_steps_added=0,
                 learner_updates_added=0, ood_collection_ready=False,
                 evidence_sha256={n: sha(run / n) for n in evidence_files},
                 metric_aggregation="1,000 host-update blocks; actor fields use one-based even / zero-based odd updates. Skipped placeholders excluded; active zeros retained.",
                 note="One host seed; BCA pair, remaining seeds and OOD acceptance pending.")
    save(out / "evaluations.json", evaluations)
    save(out / "metric_blocks.json", blocks)
    save(out / "audit.json", audit)
    print(json.dumps(audit["summary"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.root, a.repo, a.output)
