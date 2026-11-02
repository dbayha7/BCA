"""Separate-process full-size bounded resource probe; never starts a study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).parent / "resource_20261027_v1"
METHODS = ("td3bc", "iql", "cql", "rebrac", "unifloral-unified")
PLAN = Path(__file__).with_name("RESOURCE_RESUME_PLAN_20261027.md")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def prepare():
    import study
    from thrusterchain.quality_datasets import load_behavior_dataset
    from thrusterchain.training_runner import train_transitions
    data_receipt = read(ROOT / "data_receipt.json")
    dataset = load_behavior_dataset(ROOT / "data.npz")
    if (dataset.digest != data_receipt["dataset_sha256"]
            or sha(ROOT / "data.npz") != data_receipt["dataset_file_sha256"]
            or dataset.split_digest("train") != data_receipt["train_split_sha256"]
            or len(train_transitions(dataset)) != 60000
            or dataset.metadata["provenance"] != "smoke"
            or dataset.metadata["seed"] != 62077 or dataset.metadata["episodes"] != 1000
            or dataset.metadata["horizon"] != 100 or data_receipt["plan_sha256"] != sha(PLAN)):
        raise RuntimeError("full-size smoke data or frozen plan differs")
    manifest = {"schema": "thrusterchain.resource_probe.v1", "plan_sha256": sha(PLAN),
                "probe_sha256": sha(__file__), "data_receipt": data_receipt,
                "sources": study.source_identity(), "runtime": study.environment(),
                "methods": METHODS, "train_seed": 62078, "batch_size": 256,
                "chunk_critic_rounds": 10000, "chunks_per_method": 2,
                "role": "bounded validation only; no evaluation or benchmark outcomes"}
    write(ROOT / "probe_manifest.json", manifest)
    print(json.dumps({"stage": "prepared", "manifest_sha256": sha(ROOT / "probe_manifest.json")}), flush=True)


def worker(method):
    import jax
    from compiled.engine import train_chunk
    from study import environment, source_identity, primary_steps, verify_counts, verify_native_receipt
    from thrusterchain.quality_datasets import load_behavior_dataset
    from thrusterchain.training_protocol import LearnerCheckpoint
    from thrusterchain.training_runner import _rng

    manifest = read(ROOT / "probe_manifest.json")
    if (manifest["plan_sha256"] != sha(PLAN) or manifest["probe_sha256"] != sha(__file__)
            or manifest["sources"] != source_identity() or manifest["runtime"] != environment()):
        raise RuntimeError("probe source or runtime changed")
    receipt = manifest["data_receipt"]
    dataset = load_behavior_dataset(ROOT / "data.npz")
    if (dataset.digest != receipt["dataset_sha256"] or sha(ROOT / "data.npz") != receipt["dataset_file_sha256"]
            or dataset.split_digest("train") != receipt["train_split_sha256"]):
        raise RuntimeError("smoke data changed")
    dest = ROOT / method
    dest.mkdir(exist_ok=False)
    result = {"method": method, "status": "running", "chunks": [],
              "manifest_sha256": sha(ROOT / "probe_manifest.json"),
              "memory_before": jax.devices()[0].memory_stats()}
    cp = None
    sampler = _rng(62078, "offline_batches")
    spec = {"batch_size": 256, "chunk_critic_rounds": 10000}
    for number in (1, 2):
        if manifest["sources"] != source_identity() or manifest["runtime"] != environment():
            raise RuntimeError("probe source or runtime changed during worker")
        before = cp
        t0 = time.perf_counter()
        cp, native = train_chunk(dataset, method, 62078, 256,
                                 primary_steps(method, 10000), before)
        duration = time.perf_counter() - t0
        verify_native_receipt(native, cp, before, receipt, method, spec)
        verify_counts(method, cp, number * 10000)
        sampler.integers(0, 60000, size=(primary_steps(method, 10000), 256))
        if cp.metadata["runner_state"]["sampler_rng_state"] != sampler.bit_generator.state:
            raise RuntimeError("checkpoint sampler stream does not match independent PCG64 draws")
        path = dest / f"checkpoint_{number * 10000:05}.json"
        cp.save(path)
        restored = LearnerCheckpoint.load(path)
        if restored.digest != cp.digest or restored.to_bytes() != cp.to_bytes():
            raise RuntimeError("serialized checkpoint does not restore exactly")
        verify_native_receipt(native, restored, before, receipt, method, spec)
        cp = restored
        write(dest / f"native_receipt_{number * 10000:05}.json", native)
        row = {"critic_rounds_total": number * 10000, "wall_seconds": duration,
               "checkpoint_sha256": cp.digest, "checkpoint_file_sha256": sha(path),
               "sampler_state": cp.metadata["runner_state"]["sampler_rng_state"],
               "optimizer_counts": dict(cp.optimizer_step_counts),
               "primary_optimizer_updates_this_chunk": native["executed_primary_optimizer_updates"],
               "sampled_transition_draws_this_chunk": native["dataset_draws"],
               "memory_after": jax.devices()[0].memory_stats()}
        result["chunks"].append(row)
        write(dest / f"chunk_{number * 10000:05}.json", row)
        print(json.dumps({"stage": "chunk", "method": method,
                          "critic_rounds": number * 10000, "wall_seconds": duration}), flush=True)
    result["status"] = "complete"
    write(dest / "result.json", result)


def gpu_usage():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=5)
        first = p.stdout.strip().splitlines()[0]
        used, total = [int(x.strip()) for x in first.split(",")]
        return {"used_mib": used, "total_mib": total}
    except Exception as exc:
        return {"unavailable": str(exc)}


def run():
    # Both the manifest and every method output must be absent before launch.
    if (ROOT / "probe_manifest.json").exists() or any((ROOT / m).exists() for m in METHODS):
        raise RuntimeError("probe output already exists; refuse overwrite or retry")
    prep = subprocess.run([sys.executable, "-B", __file__, "--prepare"], check=False)
    if prep.returncode:
        raise RuntimeError(f"probe preparation failed with status {prep.returncode}")
    results = []
    for method in METHODS:
        print(json.dumps({"stage": "starting", "method": method}), flush=True)
        baseline = gpu_usage()
        samples = [baseline]
        with (ROOT / f"{method}.out.log").open("x", encoding="utf-8") as out, \
             (ROOT / f"{method}.err.log").open("x", encoding="utf-8") as err:
            process = subprocess.Popen([sys.executable, "-B", __file__, "--worker", method], stdout=out, stderr=err)
            while process.poll() is None:
                time.sleep(1)
                samples.append(gpu_usage())
        valid = [s["used_mib"] for s in samples if "used_mib" in s]
        result = {"method": method, "exit_status": process.returncode, "gpu_memory_baseline": baseline,
                  "gpu_memory_sampled_peak_mib": max(valid) if valid else None,
                  "gpu_memory_sample_count": len(samples),
                  "status": "complete" if process.returncode == 0 and (ROOT / method / "result.json").is_file() else "failed"}
        write(ROOT / f"{method}.monitor.json", {**result, "samples": samples})
        results.append(result)
        print(json.dumps({"stage": "finished", **result}), flush=True)
    summary = {"schema": "thrusterchain.resource_probe_results.v1",
               "manifest_sha256": sha(ROOT / "probe_manifest.json"), "methods": results,
               "complete": all(r["status"] == "complete" for r in results),
               "caveat": "Whole-device memory is sampled; throughput is bounded, not a full-study ETA."}
    write(ROOT / "summary.json", summary)
    if not summary["complete"]:
        raise RuntimeError("one or more resource methods failed; inspect retained artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--worker", choices=METHODS)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if sum((a.prepare, a.worker is not None, a.run)) != 1:
        ap.error("exactly one of --prepare, --worker, --run is required")
    if a.prepare:
        prepare()
    elif a.worker:
        worker(a.worker)
    else:
        run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
