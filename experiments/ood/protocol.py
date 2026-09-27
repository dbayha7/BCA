"""Freeze OOD declarations and reject incompatible metadata; never run a model.

Metadata acceptance stays pending until actual checkpoint/event/data decoding
and collection-specific gates pass. Adapter fixtures do not accept training runs.
This module has no execution entrypoint.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import yaml

from runtime.config import resolve

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ood.yaml"
# Canonical scientific contract, not a configurable tuning surface. A deliberate
# future amendment needs a new version and review of the corresponding tests.
CONFIG_SHA256 = "491e5662bfac93bbe58d0318ab133f98c4a93bfedc4ae9f898c59f1c6fb2e35e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
        return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def validate_config(config):
    require(digest(config) == CONFIG_SHA256,
            "OOD settings differ from the reviewed v1 contract; no silent amendments.")


def load_config(path=CONFIG):
    config = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    validate_config(config)
    return config


def training_seeds(rows):
    """All declared training/preparation/refresh/evaluation seed integers."""
    def walk(value, seeded=False):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from walk(child, seeded or "seed" in key)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child, seeded)
        elif seeded and type(value) is int:
            yield value
    return set(walk(rows))


def stream_seeds(manifest):
    for pair in manifest["pairs"]:
        for values in pair["streams"].values():
            yield from values
    for values in manifest["engineering_streams"].values():
        yield from values
    yield from manifest["bootstrap_seeds"]
    yield from manifest["plotting_fixture_seeds"]


def build_manifest():
    config = load_config()
    pins = read_json(ROOT / "configs/sources.json")["source_sha256"]
    for name, expected in pins.items():
        require(file_hash(ROOT / name) == expected, "Training source changed: " + name)
    seeds = yaml.safe_load((ROOT / config["training_seeds_from"]).read_text())["seeds"]
    rows, pairs = [], []
    for host in config["hosts"]:
        for dataset in config["datasets"]:
            for seed in seeds:
                pair_id = f"{host}/{dataset}/s{seed}"
                ids = []
                for method in config["methods"]:
                    row = resolve(ROOT / f"configs/{host}.yaml", method, seed,
                                  "unused_by_these_hosts", dataset)
                    ids.append(row["run_id"])
                    rows.append(dict(run_id=row["run_id"], pair_id=pair_id,
                        status="pending", reason="No checkpoint evidence bound; no directory search performed.",
                        checkpoint_sha256=None, resolved=row, resolved_sha256=digest(row)))
                pairs.append(dict(pair_id=pair_id, host=host, dataset=dataset,
                                  seed=seed, checkpoint_ids=ids))

    forbidden = training_seeds(rows)
    next_seed = config["streams"]["first_seed"]

    def allocate(n):
        nonlocal next_seed
        values = []
        while len(values) < n:
            require(next_seed < 2**32, "Seed allocation exceeds uint32.")
            if next_seed not in forbidden:
                values.append(next_seed)
            next_seed += 1
        return values

    episodes = config["episodes_per_collector"]
    states = len(config["collectors"]) * episodes * len(config["capture_steps"])
    actions = len(config["candidates"]["fixed_slots"]) + (
        len(config["candidates"]["signs"]) * len(config["candidates"]["perturbation_rms"]))
    cal, test = (config["coverage"][k] for k in ("calibration_episodes", "test_episodes"))
    stream_sizes = dict(preparation=2, collection=episodes, candidates=states,
        random_score=states, continuation=states, fresh_calibration=cal, test=test,
        calibration_transition_selection=cal, test_transition_selection=test,
        calibration_target_noise=cal, test_target_noise=test)
    for pair in pairs:
        pair["streams"] = {name: allocate(size) for name, size in stream_sizes.items()}
    cells = [f"{h}/{d}" for h in config["hosts"] for d in config["datasets"]]
    per_pair = dict(collection=len(config["collectors"]) * episodes * max(config["capture_steps"]),
                    outcomes=states * actions * len(config["continuations"]) * config["horizon"],
                    repeat_checks=states * actions * len(config["continuations"]),
                    coverage=(cal + test) * config["coverage"]["maximum_episode_steps"])
    science_total = len(pairs) * sum(per_pair.values())
    engineering = len(cells) * config["resources"]["engineering_transitions_per_cell"]
    local_files = ["README.md", "configs/ood.yaml", "configs/sources.json",
                   "experiments/ood/protocol.py", "experiments/ood/test_protocol.py",
                   "experiments/ood/adapters.py", "experiments/ood/simulator.py",
                   "experiments/ood/oracle.py", "experiments/ood/test_outcomes.py",
                   "experiments/ood/analyze.py", "experiments/ood/report.py",
                   "experiments/ood/test_analysis.py", "experiments/ood/requirements.txt",
                   "docs/superpowers/plans/2026-09-24-ood-experiment.md"]
    manifest = dict(schema="bca-ood-declaration-v1", config=config,
        repository_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        protocol_files_sha256={p: file_hash(ROOT / p) for p in local_files},
        # A documentation-only edit does not invalidate an otherwise identical
        # trained model. Training code/config/dependency pins remain exact;
        # current documentation and OOD code are recorded separately above.
        training_source_sha256={k: v for k, v in pins.items()
                                if k.endswith((".py", ".yaml", ".csv", ".PROVENANCE", ".txt"))},
        checkpoints=rows, pairs=pairs,
        engineering_streams={cell: allocate(1024) for cell in cells},
        bootstrap_seeds=allocate(config["inference"]["bootstrap_draws"]),
        plotting_fixture_seeds=allocate(1),
        budget=dict(per_pair=per_pair, science_total=science_total,
                    engineering_total=engineering, grand_total=science_total + engineering,
                    training_updates_to_execute=0),
        inference_family_size=len(cells) * config["inference"]["primary_contrasts_per_cell"],
        ready_for_collection=False,
        pending_gates=["actual 1M checkpoint and event decoding", "paired preparation verification",
                       "live coverage target including ReBRAC next-action contract",
                       "collection simulator identity and complete restore schema",
                       "pre-outcome candidate/action/step-key hashes",
                       "bounded engineering validation", "explicit execution direction"])
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def validate_manifest(manifest):
    require(isinstance(manifest, dict), "Manifest must be an object.")
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    require(manifest.get("manifest_sha256") == digest(payload), "Manifest digest mismatch.")
    # Recompute from pinned inputs, so merely rehashing a removed row, altered
    # seed, inflated ceiling or invented verified status does not authorize it.
    require(digest(manifest) == digest(build_manifest()),
            "Manifest differs from the current complete declaration or source revision.")


def validate_episode_ids(calibration, test, *, adaptive_feedback=False):
    require(adaptive_feedback is False, "Calibration/test outcomes cannot update frozen objects.")
    require(len(calibration) == 200 and len(test) == 500, "Coverage episode count mismatch.")
    require(all(type(x) is int and 0 <= x < 2**32 for x in calibration + test),
            "Episode IDs must be uint32 integers.")
    require(len(set(calibration + test)) == 700, "Repeated or overlapping coverage episode IDs.")


def validate_request(manifest, request):
    """Validate an intended behavioral query, not permission to execute it."""
    validate_manifest(manifest)
    require(set(request) == {"pair_id", "seed", "continuation", "horizon"}, "Unknown request fields.")
    pairs = {p["pair_id"]: p for p in manifest["pairs"]}
    require(request["pair_id"] in pairs, "Undeclared pair.")
    require(type(request["seed"]) is int and request["seed"] == pairs[request["pair_id"]]["seed"],
            "Undeclared training seed.")
    require(request["continuation"] in manifest["config"]["continuations"], "Undeclared continuation.")
    require(type(request["horizon"]) is int and request["horizon"] == manifest["config"]["horizon"],
            "Changed horizon.")


def validate_usage(manifest, usage):
    """Check a cumulative ledger; persistence/reservation belongs to the collector."""
    validate_manifest(manifest)
    require(set(usage) == {"science", "engineering"}, "Unknown resource categories.")
    pairs = {p["pair_id"] for p in manifest["pairs"]}
    caps = manifest["budget"]["per_pair"]
    total = 0
    for pair, counts in usage["science"].items():
        require(pair in pairs and set(counts) == set(caps), "Unknown pair or accounting category.")
        for name, count in counts.items():
            require(type(count) is int and 0 <= count <= caps[name], "Invalid/exceeded " + name + " budget.")
            total += count
    for cell, count in usage["engineering"].items():
        require(cell in manifest["engineering_streams"], "Unknown engineering cell.")
        require(type(count) is int and 0 <= count <= manifest["config"]["resources"]["engineering_transitions_per_cell"],
                "Invalid/exceeded engineering budget.")
        total += count
    require(total <= manifest["budget"]["grand_total"], "Exceeded grand budget.")
    return total


def validate_evidence(manifest, run_id, artifacts, root):
    """Check hash-bound *reported* metadata from the existing training layout.

    artifacts maps seven roles to path/sha256; paths only locate explicit inputs.
    actual_exit is a separate supervisor receipt (ood-process-exit-v1) bound to
    run_id and result_sha256. The learner's own exit.json is insufficient.
    This does NOT independently decode checkpoint, journal, or cached data.
    """
    validate_manifest(manifest)
    rows = {r["run_id"]: r for r in manifest["checkpoints"]}
    require(run_id in rows, "Undeclared checkpoint/run.")
    r = rows[run_id]["resolved"]
    roles = {"resolved", "source", "preparation", "result", "checkpoint", "events", "actual_exit"}
    require(set(artifacts) == roles, "Missing or unknown evidence artifacts.")
    try:
        paths = {}
        for role, entry in artifacts.items():
            require(set(entry) == {"path", "sha256"}, "Invalid artifact binding.")
            path = Path(root) / entry["path"]
            require(file_hash(path) == entry["sha256"], "Artifact hash mismatch: " + role)
            paths[role] = path
        require(digest(read_json(paths["resolved"])) == digest(r), "Wrong resolved configuration.")
        source = read_json(paths["source"])
        require(all(source.get(k) == v for k, v in manifest["training_source_sha256"].items()),
                "Wrong reported training source.")
        result, actual = read_json(paths["result"]), read_json(paths["actual_exit"])
        require(actual["schema"] == "ood-process-exit-v1" and actual["run_id"] == run_id
                and type(actual["exit_code"]) is int and actual["exit_code"] == 0
                and actual["result_sha256"] == artifacts["result"]["sha256"],
                "Missing, failed or stale actual process exit receipt.")
        require(result["completed"] is True and result["run_id"] == run_id
                and type(result["steps_completed"]) is int and result["steps_completed"] == 1000000
                and digest(result["expected_counts"]) == digest(r["expected_counts"]), "Incomplete/wrong-budget run.")
        require(result["events_sha256"] == artifacts["events"]["sha256"], "Wrong event journal.")
        finals = [c for c in result["checkpoints"]
                  if type(c["step"]) is int and c["step"] == 1000000]
        require(len(finals) == 1, "Missing/ambiguous final checkpoint.")
        final = finals[0]
        require(final["sha256"] == artifacts["checkpoint"]["sha256"]
                and final["path"] == paths["checkpoint"].name, "Wrong final checkpoint.")
        counters = final["counters"]
        require(set(counters) == {"actor", "critic", "accepted_scale_fits"}, "Missing counters.")
        for name, count in (("actor", 500000), ("critic", 1000000)):
            found = counters[name]
            require(isinstance(found, dict) and any(k.endswith("/count") for k in found)
                    and all(type(v) is int and v == count for v in found.values()),
                    "Wrong reported optimizer counters: " + name)
        fits = counters["accepted_scale_fits"]
        require((fits is None if r["method"] == "host" else
                 type(fits) is int and 0 <= fits <= 1000000), "Invalid accepted scale-fit count.")
        prepared = read_json(paths["preparation"])
        require(prepared["accepted"] is True, "Preparation failed.")
        m = prepared["metadata"]
        require(m["raw_identity"]["sha256"] == r["cache"]["sha256"], "Wrong cached data identity.")
        train, held = m["training_converted_ids"], m["heldout_converted_ids"]
        require(train and held and all(type(x) is int and x >= 0 for x in train + held)
                and len(set(train + held)) == len(train + held), "Leaked/repeated/invalid split IDs.")
        require(m["normalization_fit"] == "training_complement"
                and m["normalization_fit_converted_ids"] == train, "Leaked normalization fit.")
        require(len(m["obs_mean"]) > 0 and len(m["obs_mean"]) == len(m["obs_std"])
                and all(type(x) in (int, float) and math.isfinite(x) for x in m["obs_mean"])
                and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in m["obs_std"]),
                "Invalid reported normalization.")
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError("Incomplete or malformed evidence: " + str(exc)) from exc
    return dict(status="pending", metadata_checked=True, checkpoint_decoded=False,
                events_verified=False, data_contents_verified=False, ready_for_collection=False,
                run_id=run_id, artifact_sha256={k: v["sha256"] for k, v in artifacts.items()},
                split_sha256=digest([train, held]),
                normalization_sha256=digest([m["obs_mean"], m["obs_std"]]),
                remaining="Independent adapters must verify checkpoint, events, data and paired preparation.")


def write_manifest(path, manifest):
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New declaration file; refuses overwrite.")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.output:
        write_manifest(args.output, manifest)
    print(json.dumps(dict(manifest_sha256=manifest["manifest_sha256"],
        pairs=len(manifest["pairs"]), checkpoint_requirements=len(manifest["checkpoints"]),
        pending_checkpoints=len(manifest["checkpoints"]), verified_checkpoints=0,
        required_host_updates=1000000, required_actor_updates=500000,
        maximum_future_environment_transitions=manifest["budget"]["grand_total"],
        training_updates_executed=0, simulator_transitions_executed=0,
        ready_for_collection=False), indent=2))


if __name__ == "__main__":
    main()
