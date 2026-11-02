"""Bounded same-process cache/host+device memory check; never runs a study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE / "cache_memory_20261027_v1"
PLAN = HERE / "CACHE_MEMORY_PLAN_20261027.md"
METHODS = ("td3bc", "iql", "cql", "rebrac", "unifloral-unified")
BEHAVIORS = ("random", "feedback_weak", "feedback_strong", "mixed_random_strong")
DATA_SEEDS = (62081, 62082, 62083)
TRAIN_SEEDS = (62091, 62092, 62093)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def gpu_usage():
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], check=True,
                                text=True, capture_output=True, timeout=5).stdout
        used, total = map(int, output.strip().splitlines()[0].split(","))
        return {"used_mib": used, "total_mib": total}
    except Exception as exc:
        return {"unavailable": str(exc)}


def rss():
    # Linux WSL: ru_maxrss is KiB. VmRSS is sampled current resident memory.
    now = None
    with open("/proc/self/status", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                now = int(line.split()[1]) / 1024
    return {"resident_mib": now,
            "process_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def collect():
    from thrusterchain.quality_datasets import generate_behavior_dataset
    from thrusterchain.training_runner import train_transitions
    ROOT.mkdir(exist_ok=False)
    (ROOT / "datasets").mkdir()
    cases = []
    for behavior in BEHAVIORS:
        for index, (seed, train_seed) in enumerate(zip(DATA_SEEDS, TRAIN_SEEDS), 1):
            name = f"{behavior}-r{index}"
            path = ROOT / "datasets" / (name + ".npz")
            started = time.perf_counter()
            dataset = generate_behavior_dataset(path, behavior=behavior, episodes=1000,
                                                seed=seed, provenance="smoke")
            if len(train_transitions(dataset)) != 60000 or dataset.metadata["horizon"] != 100:
                raise RuntimeError("unexpected data split/horizon")
            case = {"id": name, "behavior": behavior, "data_seed": seed,
                    "train_seed": train_seed, "dataset_sha256": dataset.digest,
                    "train_split_sha256": dataset.split_digest("train"),
                    "file_sha256": sha(path), "collection_seconds": time.perf_counter() - started}
            cases.append(case)
            print(json.dumps({"stage": "collected", **case}), flush=True)
    write(ROOT / "collection.json", {"schema": "thrusterchain.cache_probe_data.v1",
          "plan_sha256": sha(PLAN), "probe_sha256": sha(__file__), "cases": cases})


def run():
    import jax
    import study
    from compiled.engine import _PROGRAMS, train_chunk
    from thrusterchain.quality_datasets import load_behavior_dataset
    from thrusterchain.training_protocol import LearnerCheckpoint
    from thrusterchain.training_runner import train_transitions

    coll = json.loads((ROOT / "collection.json").read_text(encoding="utf-8"))
    if coll["plan_sha256"] != sha(PLAN) or coll["probe_sha256"] != sha(__file__):
        raise RuntimeError("frozen plan/probe changed")
    if tuple(study.read_json(HERE.parent / "spec.full.json")["methods"]) != METHODS:
        raise RuntimeError("study method order changed")
    datasets = {}
    for case in coll["cases"]:
        path = ROOT / "datasets" / (case["id"] + ".npz")
        dataset = load_behavior_dataset(path)
        if (sha(path) != case["file_sha256"] or dataset.digest != case["dataset_sha256"]
                or dataset.split_digest("train") != case["train_split_sha256"]
                or len(train_transitions(dataset)) != 60000):
            raise RuntimeError("dataset identity mismatch")
        datasets[case["id"]] = dataset
    if len(datasets) != 12:
        raise RuntimeError("expected exactly twelve smoke datasets")
    source, runtime = study.source_identity(), study.environment()
    manifest = {"schema": "thrusterchain.cache_probe.v1", "plan_sha256": sha(PLAN),
                "probe_sha256": sha(__file__), "collection_sha256": sha(ROOT / "collection.json"),
                "sources": source, "runtime": runtime, "method_order": METHODS,
                "batch_size": 256, "critic_rounds_per_cell": 2, "cases": coll["cases"],
                "initial_host_memory": rss(), "initial_gpu_memory": gpu_usage()}
    write(ROOT / "manifest.json", manifest)
    (ROOT / "cells").mkdir()
    records = []
    for case in coll["cases"]:
        for method in METHODS:
            if source != study.source_identity() or runtime != study.environment():
                raise RuntimeError("frozen source or runtime drift")
            dataset = datasets[case["id"]]
            data_receipt = {"dataset_sha256": case["dataset_sha256"],
                            "train_split_sha256": case["train_split_sha256"]}
            started = time.perf_counter()
            cp, receipt = train_chunk(dataset, method, case["train_seed"], 256,
                                      study.primary_steps(method, 2))
            study.verify_counts(method, cp, 2)
            study.verify_native_receipt(receipt, cp, None, data_receipt,
                                        method, {"batch_size": 256, "chunk_critic_rounds": 2})
            if source != study.source_identity() or runtime != study.environment():
                raise RuntimeError("frozen source or runtime drift after chunk")
            path = ROOT / "cells" / (case["id"] + "_" + method)
            path.mkdir(exist_ok=False)
            cp.save(path / "checkpoint.json")
            loaded = LearnerCheckpoint.load(path / "checkpoint.json")
            if loaded.digest != cp.digest:
                raise RuntimeError("serialized checkpoint mismatch")
            cache = {name: sum(key[0] == name for key in _PROGRAMS) for name in METHODS}
            record = {"case": case["id"], "method": method, "wall_seconds": time.perf_counter() - started,
                      "dataset_sha256": dataset.digest, "checkpoint_sha256": cp.digest,
                      "checkpoint_file_sha256": sha(path / "checkpoint.json"),
                      "optimizer_counts": dict(cp.optimizer_step_counts), "programs_by_method": cache,
                      "host_memory": rss(), "device_memory": gpu_usage(),
                      "jax_allocator_memory": jax.devices()[0].memory_stats()}
            write(path / "result.json", record)
            records.append(record)
            print(json.dumps({"stage": "cell", **record}), flush=True)
    if len(records) != 60:
        raise RuntimeError("expected exactly sixty cells")
    expected = {"td3bc": 1, "iql": 1, "cql": 1, "rebrac": 1, "unifloral-unified": 12}
    if records[-1]["programs_by_method"] != expected:
        raise RuntimeError("unexpected compiled-program cache composition")
    report = {"schema": "thrusterchain.cache_probe_result.v1", "status": "complete",
              "manifest_sha256": sha(ROOT / "manifest.json"), "cells": records,
              "final_host_memory": rss(), "final_device_memory": gpu_usage(),
              "final_programs_by_method": records[-1]["programs_by_method"],
              "caveat": "point-in-time GPU samples can miss peaks; two-round chunks do not measure full study runtime"}
    write(ROOT / "summary.json", report)
    print(json.dumps({"stage": "complete", "cells": len(records),
                      "programs": report["final_programs_by_method"],
                      "host_memory": report["final_host_memory"],
                      "device_memory": report["final_device_memory"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("collect", "run"))
    try:
        args = parser.parse_args()
        {"collect": collect, "run": run}[args.action]()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
