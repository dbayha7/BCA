"""Preregistered, resumable ThrusterChain study; never edits benchmark sources.

One process owns an exclusive advisory lock. Every training chunk is published
as an atomic directory with checkpoint and receipt; completed data, checkpoints
and evaluations are never overwritten. Evaluation is blocked until *every*
declared training run reaches the registered final checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "experiments/thrusterchain_unifloral_gpu_compat_v1"
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BENCHMARK))
METHODS = ("td3bc", "iql", "cql", "rebrac", "unifloral-unified")
BEHAVIORS = ("random", "feedback_weak", "feedback_strong", "mixed_random_strong")
STARTS = ("recorded", "strong_current", "big_waves", "big_kicks")
SHIFTS = ("baseline", "attenuated_thruster", "bounded_disturbance", "combined")
EXPECTED_BENCHMARK = "72ee69256129c9d48216acc69a505c87575e7f9016b912f14f664dfd09cf4abc"
EXPECTED_UPSTREAM = "14a9bcdbc6d96fad85544a9d98957e98a7e46db13b89d74d54e3c5ce246517c8"
SPEC_KEYS = {"schema", "profile", "cases", "methods", "episodes_per_dataset", "critic_rounds",
             "chunk_critic_rounds", "batch_size", "starts", "shifts", "evaluation_seeds", "expectations"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out: raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out
    def reject(value): raise ValueError("non-finite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=unique, parse_constant=reject)


def write_new(path, data):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(encoded(data))
        stream.flush()
        os.fsync(stream.fileno())


def positive_int(x, name, minimum=1):
    if type(x) is not int or x < minimum: raise ValueError(f"{name} must be an integer >= {minimum}")
    return x


def validate_spec(spec):
    if type(spec) is not dict or set(spec) != SPEC_KEYS or spec["schema"] != "thrusterchain.full_benchmark_spec.v1":
        raise ValueError("unexpected study specification fields or schema")
    if spec["profile"] not in ("full", "smoke"): raise ValueError("unknown profile")
    n = positive_int(spec["episodes_per_dataset"], "episodes_per_dataset", 10)
    rounds = positive_int(spec["critic_rounds"], "critic_rounds")
    chunk = positive_int(spec["chunk_critic_rounds"], "chunk_critic_rounds")
    batch = positive_int(spec["batch_size"], "batch_size")
    if rounds % 2 or chunk % 2 or rounds % chunk:
        raise ValueError("critic rounds and chunk must be even and divide exactly for unified accounting")
    if spec["methods"] != list(METHODS): raise ValueError("method list must match fixed benchmark order")
    if spec["starts"] != list(STARTS) or spec["shifts"] != list(SHIFTS):
        raise ValueError("start and shift cells must match the fixed benchmark order")
    seeds = spec["evaluation_seeds"]
    if type(seeds) is not list or not seeds or any(type(x) is not int or x < 0 or x >= 2**63 for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique 63-bit nonnegative integers")
    if type(spec["expectations"]) is not dict or not spec["expectations"]:
        raise ValueError("predeclared expectations required")
    cases = spec["cases"]
    if type(cases) is not list or not cases: raise ValueError("cases must be a nonempty list")
    ids, behavior_data_seeds = set(), set()
    for case in cases:
        if type(case) is not dict or set(case) != {"id", "behavior", "data_seed", "train_seed"}:
            raise ValueError("case requires id, behavior, data_seed and train_seed")
        ident = case["id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", ident) or ident in ids:
            raise ValueError("case IDs must be unique safe path components")
        if case["behavior"] not in BEHAVIORS: raise ValueError("unknown behavior")
        for key in ("data_seed", "train_seed"):
            positive_int(case[key], key, 0)
            if case[key] >= 2**63: raise ValueError("seeds must fit in 63 bits")
        key = (case["behavior"], case["data_seed"])
        if key in behavior_data_seeds: raise ValueError("dataset seeds must be distinct within each behavior")
        ids.add(ident); behavior_data_seeds.add(key)
    if spec["profile"] == "full":
        if (n != 1000 or rounds != 1_000_000 or chunk != 10_000 or batch != 256 or len(seeds) != 5
                or len(cases) != 12 or any(sum(c["behavior"] == b for c in cases) != 3 for b in BEHAVIORS)):
            raise ValueError("full profile requires the authorized 1000/1M/10000/256, 3 replicates and 5 evaluation seeds")
        for behavior in BEHAVIORS:
            subset = [c["train_seed"] for c in cases if c["behavior"] == behavior]
            if len(set(subset)) != 3: raise ValueError("each behavior needs three distinct training seeds")
        assignments = [{(c["data_seed"], c["train_seed"]) for c in cases if c["behavior"] == b} for b in BEHAVIORS]
        if any(group != assignments[0] for group in assignments[1:]):
            raise ValueError("full-profile dataset and learner seeds must be paired across behaviors")
    return spec


def execution_settings(jax):
    """Freeze arithmetic/compiler settings that can change executed numerics.

    Distinguish an unset environment variable from an explicitly empty one.
    XLA flags, device visibility and vendor math overrides are not reflected by
    the JAX version or device name, but can affect executable selection.
    """
    names = ("XLA_FLAGS", "JAX_PLATFORMS", "JAX_PLATFORM_NAME",
             "JAX_DEFAULT_MATMUL_PRECISION", "JAX_DISABLE_JIT", "JAX_ENABLE_X64",
             "JAX_DEFAULT_PRNG_IMPL", "JAX_THREEFRY_PARTITIONABLE",
             "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE",
             "CUBLAS_WORKSPACE_CONFIG", "XLA_PYTHON_CLIENT_PREALLOCATE",
             "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_COMPILATION_CACHE_DIR")
    return {"environment": {name: os.environ.get(name) for name in names},
            "jax_config": {"disable_jit": bool(jax.config.jax_disable_jit),
                           "enable_x64": bool(jax.config.jax_enable_x64),
                           "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
                           "default_prng_impl": str(jax.config.jax_default_prng_impl),
                           "threefry_partitionable": bool(jax.config.jax_threefry_partitionable)}}


def environment():
    """Capture and enforce validated CUDA runtime; no numerical computation here."""
    import importlib.metadata as meta
    import jax
    import numpy as np
    import scipy
    import gymnasium
    if (os.environ.get("JAX_PLATFORMS") != "cuda" or str(jax.config.jax_default_matmul_precision) != "highest"
            or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false"
            or jax.config.jax_disable_jit or jax.config.jax_enable_x64):
        raise RuntimeError("force CUDA, highest matmul precision, enabled JIT, float32 defaults and disabled JAX preallocation")
    if jax.default_backend() != "gpu" or any(d.platform != "gpu" for d in jax.devices()):
        raise RuntimeError("CUDA GPU backend is required")
    packages = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "scipy", "flax", "optax", "chex",
        "distrax", "tensorflow-probability", "gymnasium", "ml-dtypes", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-nvcc-cu12", "nvidia-cudnn-cu12", "nvidia-cufft-cu12", "nvidia-cusolver-cu12", "nvidia-cusparse-cu12",
        "nvidia-nvjitlink-cu12")
    versions = {name: meta.version(name) for name in packages}
    if versions["jax"] != "0.5.3" or versions["jaxlib"] != "0.5.3" or versions["numpy"] != "1.26.4":
        raise RuntimeError("runtime differs from the bounded-validated JAX/NumPy stack")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True).stdout.strip().splitlines()
    if not gpu: raise RuntimeError("nvidia-smi did not report a GPU")
    return {"versions": versions, "python": sys.version, "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()], "device_kinds": [d.device_kind for d in jax.devices()],
            "gpu_inventory": gpu,
            "precision": str(jax.config.jax_default_matmul_precision),
            "execution_settings": execution_settings(jax),
            "numpy_runtime": np.__version__, "scipy_runtime": scipy.__version__, "gymnasium_runtime": gymnasium.__version__}


def source_identity():
    from run_benchmark import source_manifest
    benchmark = source_manifest()["source_sha256"]
    if benchmark != EXPECTED_BENCHMARK: raise RuntimeError("benchmark differs from the GPU-validated source")
    upstream = REPO / "baselines/unifloral/algorithms/unifloral.py"
    if sha(upstream) != EXPECTED_UPSTREAM: raise RuntimeError("unified upstream source differs")
    files = [REPO / ("algorithms/" + name + ".py") for name in ("td3_bc", "iql", "cql", "rebrac")]
    files += [REPO / "runtime/networks.py", REPO / "runtime/rewards.py", SCRIPT]
    backend = SCRIPT.parent / "compiled" / "engine.py"
    if not backend.is_file(): raise RuntimeError("compiled engine is not installed")
    files.extend(sorted(p for p in backend.parent.rglob("*") if p.is_file() and p.suffix in (".py", ".json")))
    return {"benchmark_source_sha256": benchmark, "unified_upstream_sha256": EXPECTED_UPSTREAM,
            "external_sources": {str(p.relative_to(REPO)): sha(p) for p in files}}


def read_manifest(study):
    """Read the immutable declaration without initializing JAX or acquiring a writer lock."""
    raw = (study / "manifest.json").read_bytes()
    digest = (study / "manifest.sha256").read_text(encoding="ascii").strip()
    if hashlib.sha256(raw).hexdigest() != digest: raise RuntimeError("frozen manifest bytes changed")
    m = read_json(study / "manifest.json")
    validate_spec(m["spec"])
    return m


def load_manifest(study):
    m = read_manifest(study)
    if m["sources"] != source_identity() or m["runtime"] != environment():
        raise RuntimeError("frozen source or runtime changed")
    return m


def prepare(spec_path, study):
    spec = validate_spec(read_json(spec_path))
    identity, runtime = source_identity(), environment()
    study = Path(study)
    study.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "thrusterchain.full_benchmark_manifest.v1", "spec": spec,
                "sources": identity, "runtime": runtime, "strict_offline": True,
                "checkpoint_rule": "final fixed critic-round budget; no simulator selection",
                "scope": "matched critic rounds; optimizer work and sampled rows differ across methods",
                "holdout_status": "seed separation only; prior exposure is not attested"}
    write_new(study / "manifest.json", manifest)
    with (study / "manifest.sha256").open("x", encoding="ascii") as f:
        f.write(sha(study / "manifest.json") + "\n")
    return manifest


@contextlib.contextmanager
def locked(study):
    import fcntl
    lock_path = study / ".study.lock"
    with lock_path.open("a+") as f:
        try: fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("study already has an active process")
        try: yield
        finally: fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def publish_directory(destination):
    """Build a new directory and atomically publish it; never overwrite a result."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise FileExistsError(destination)
    if any(destination.parent.glob(".pending-*")):
        raise RuntimeError("incomplete pending artifact requires inspection before any retry")
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination.parent))
    try:
        yield temporary
        if destination.exists(): raise FileExistsError(destination)
        os.rename(temporary, destination)
    except BaseException:
        # Preserve incomplete evidence rather than deleting or silently retrying.
        raise


def collect(study, spec):
    from thrusterchain.quality_datasets import generate_behavior_dataset, load_behavior_dataset
    for case in spec["cases"]:
        dest = study / "datasets" / case["id"]
        if dest.exists():
            data = load_behavior_dataset(dest / "data.npz")
            receipt = read_json(dest / "receipt.json")
            if (receipt["dataset_sha256"] != data.digest or receipt["file_sha256"] != sha(dest / "data.npz")
                    or receipt["case"] != case): raise RuntimeError("existing dataset identity mismatch")
            validate_case_dataset(data, receipt, case, spec)
            continue
        load_manifest(study)
        with publish_directory(dest) as tmp:
            data = generate_behavior_dataset(tmp / "data.npz", behavior=case["behavior"],
                episodes=spec["episodes_per_dataset"], seed=case["data_seed"], provenance="research")
            receipt = {"case": case, "dataset_sha256": data.digest,
                "file_sha256": sha(tmp / "data.npz"), "train_split_sha256": data.split_digest("train"),
                "episodes": spec["episodes_per_dataset"], "schema": data.metadata["schema"]}
            validate_case_dataset(data, receipt, case, spec)
            write_new(tmp / "receipt.json", receipt)
            load_manifest(study)


def validate_case_dataset(data, receipt, case, spec):
    """Bind the loaded dataset's declared collector configuration to the study.

    The dataset loader validates the internal metadata and arrays; this separate
    check prevents a valid but different behavior or seed from being accepted
    under a frozen case merely because its receipt and byte hashes agree.
    """
    from thrusterchain.quality_datasets import SCHEMA

    metadata = data.metadata
    if (case not in spec["cases"] or not isinstance(receipt, dict)
            or set(receipt) != {"case", "dataset_sha256", "file_sha256", "train_split_sha256", "episodes", "schema"}
            or receipt.get("case") != case or receipt.get("dataset_sha256") != data.digest
            or receipt.get("train_split_sha256") != data.split_digest("train")
            or receipt.get("episodes") != spec["episodes_per_dataset"] or receipt.get("schema") != SCHEMA
            or metadata.get("schema") != SCHEMA or metadata.get("episodes") != spec["episodes_per_dataset"]
            or metadata.get("seed") != case["data_seed"]
            or metadata.get("behavior", {}).get("name") != case["behavior"]
            or metadata.get("horizon") != 100 or metadata.get("gamma") != .98
            or metadata.get("task_semantics") != "finite_horizon"
            or metadata.get("observation_mode") != "time_aware" or metadata.get("start_family") != "recorded"
            or metadata.get("physics") != {"stall": True, "stall_slope": 1.5}
            or metadata.get("provenance") not in (("research",) if spec["profile"] == "full" else ("smoke", "research"))):
        raise RuntimeError("dataset metadata or receipt differs from frozen case and study")


def data_for(study, case):
    from thrusterchain.quality_datasets import load_behavior_dataset
    root = study / "datasets" / case["id"]
    receipt = read_json(root / "receipt.json")
    if receipt["case"] != case or receipt["file_sha256"] != sha(root / "data.npz"):
        raise RuntimeError("dataset receipt or bytes changed")
    data = load_behavior_dataset(root / "data.npz")
    if data.digest != receipt["dataset_sha256"] or data.split_digest("train") != receipt["train_split_sha256"]:
        raise RuntimeError("dataset or train split identity mismatch")
    validate_case_dataset(data, receipt, case, read_manifest(study)["spec"])
    return data


def primary_steps(method, critic_rounds):
    return critic_rounds // 2 if method == "unifloral-unified" else critic_rounds


def verify_counts(method, checkpoint, rounds):
    counts = dict(checkpoint.optimizer_step_counts)
    if counts[checkpoint.primary_optimizer] != primary_steps(method, rounds):
        raise RuntimeError("primary optimizer count disagrees with declared critic rounds")
    if method == "cql":
        if counts.get("critic1") != rounds or counts.get("critic2") != rounds:
            raise RuntimeError("CQL twin-critic counts disagree with declared rounds")
    elif counts.get("critic") != rounds:
        raise RuntimeError("critic optimizer count disagrees with declared rounds")


def checkpoint_progress(study, case, method, spec):
    """Lightweight concurrent snapshot; full checkpoint verification is separate."""
    root = study / "training" / case["id"] / method
    last = 0
    for step in range(spec["chunk_critic_rounds"], spec["critic_rounds"] + 1, spec["chunk_critic_rounds"]):
        path = root / f"round_{step:07d}"
        if not path.exists(): break
        receipt = read_json(path / "receipt.json")
        if (receipt.get("case") != case or receipt.get("method") != method or receipt.get("critic_rounds") != step
                or receipt.get("primary_steps_total") != primary_steps(method, step)
                or not (path / "checkpoint.json").is_file()):
            raise RuntimeError("invalid published checkpoint receipt")
        last = step
    return last


def verify_native_receipt(native, cp, previous, data_receipt, method, spec):
    """Verify the full native receipt and link it to the previous checkpoint."""
    from thrusterchain.training_runner import _hash
    raw = dict(native)
    digest = raw.pop("receipt_sha256", None)
    prev_counts = dict(previous.optimizer_step_counts) if previous is not None else {}
    counts = dict(cp.optimizer_step_counts)
    deltas = {key: value - prev_counts.get(key, 0) for key, value in counts.items()}
    updates = primary_steps(method, spec["chunk_critic_rounds"])
    engine_sha = sha(SCRIPT.parent / "compiled" / "engine.py")
    state = cp.metadata.get("runner_state", {})
    if (digest != _hash(raw) or native.get("schema") != "thrusterchain.compiled_chunk.v1"
            or native.get("method") != method or native.get("engine") != "compiled-native-scan-v1"
            or native.get("engine_sha256") != engine_sha or state.get("engine") != native.get("engine")
            or state.get("engine_sha256") != engine_sha
            or native.get("dataset_sha256") != data_receipt["dataset_sha256"]
            or native.get("split_sha256") != {"train": data_receipt["train_split_sha256"]}
            or native.get("preprocessing_source") != "train_only" or native.get("learner_visible_split") != "train"
            or native.get("adapter_id") != cp.adapter_id or native.get("primary_optimizer") != cp.primary_optimizer
            or native.get("executed_primary_optimizer_updates") != updates
            or native.get("optimizer_step_counts") != counts or native.get("optimizer_step_deltas") != deltas
            or native.get("executed_optimizer_steps_total") != sum(deltas.values())
            or native.get("batch_size") != spec["batch_size"]
            or native.get("dataset_draws") != updates * spec["batch_size"]
            or native.get("environment_interactions") != 0 or native.get("checkpoint_sha256") != cp.digest
            or native.get("resume_from_sha256") != (previous.digest if previous else None)):
        raise RuntimeError("native chunk receipt or checkpoint lineage mismatch")


def checkpoints(study, case, method, spec, *, allow_in_progress=False):
    from thrusterchain.training_protocol import LearnerCheckpoint
    from thrusterchain.__main__ import HOST_ADAPTERS
    root = study / "training" / case["id"] / method
    data_receipt_path = study / "datasets" / case["id"] / "receipt.json"
    found = []
    for step in range(spec["chunk_critic_rounds"], spec["critic_rounds"] + 1, spec["chunk_critic_rounds"]):
        path = root / f"round_{step:07d}"
        if not path.exists():
            if not allow_in_progress and any((root / f"round_{later:07d}").exists() for later in range(step + spec["chunk_critic_rounds"], spec["critic_rounds"] + 1, spec["chunk_critic_rounds"])):
                raise RuntimeError("noncontiguous training checkpoints")
            break
        receipt = read_json(path / "receipt.json")
        cp = LearnerCheckpoint.load(path / "checkpoint.json")
        data_receipt = read_json(data_receipt_path)
        state = cp.metadata.get("runner_state", {})
        if (receipt["checkpoint_sha256"] != cp.digest or receipt["checkpoint_file_sha256"] != sha(path / "checkpoint.json")
                or receipt["case"] != case or receipt["method"] != method or receipt["critic_rounds"] != step
                or receipt["chunk_critic_rounds"] != spec["chunk_critic_rounds"]
                or receipt["primary_steps_total"] != primary_steps(method, step)
                or receipt["optimizer_counts"] != dict(cp.optimizer_step_counts)
                or cp.adapter_id != HOST_ADAPTERS[method][2]
                or cp.optimizer_step_counts[cp.primary_optimizer] != primary_steps(method, step)
                or state.get("dataset_sha256") != data_receipt["dataset_sha256"]
                or state.get("train_split_sha256") != data_receipt["train_split_sha256"]
                or state.get("seed") != case["train_seed"] or state.get("batch_size") != spec["batch_size"]
                or state.get("learning_settings") != {} or state.get("phase") != "offline"
                or state.get("total_runner_updates") != primary_steps(method, step)):
            raise RuntimeError("checkpoint or chunk receipt identity mismatch")
        verify_counts(method, cp, step)
        verify_native_receipt(receipt["native_receipt"], cp, found[-1][1] if found else None,
                              data_receipt, method, spec)
        found.append((step, cp))
    return found


def train_all(study, spec, *, resume):
    from compiled.engine import train_chunk
    data = {case["id"]: data_for(study, case) for case in spec["cases"]}
    verify_data_seeds(data, spec)
    for case in spec["cases"]:
        dataset = data[case["id"]]
        data_receipt = read_json(study / "datasets" / case["id"] / "receipt.json")
        for method in spec["methods"]:
            failure = study / "training" / case["id"] / method / "failure.json"
            if failure.exists():
                # A failed cell is a terminal outcome, not an implicit retry.
                continue
            existing = checkpoints(study, case, method, spec)
            if existing and not resume:
                raise RuntimeError("pre-existing chunks found; explicit --resume required")
            step, checkpoint = existing[-1] if existing else (0, None)
            while step < spec["critic_rounds"]:
                load_manifest(study)
                t0 = time.monotonic()
                chunk = spec["chunk_critic_rounds"]
                try:
                    candidate, native_receipt = train_chunk(dataset, method, case["train_seed"], spec["batch_size"],
                        primary_steps(method, chunk), checkpoint)
                    verify_counts(method, candidate, step + chunk)
                    verify_native_receipt(native_receipt, candidate, checkpoint, data_receipt, method, spec)
                    checkpoint = candidate
                except Exception as exc:
                    # Do not retune/restart a failed cell. Preserve its exception
                    # and continue the other predeclared independent cells.
                    failure.parent.mkdir(parents=True, exist_ok=True)
                    write_new(failure, {"schema": "thrusterchain.study_failure.v1", "case": case,
                        "method": method, "last_completed_critic_rounds": step,
                        "attempted_chunk_critic_rounds": chunk, "exception_type": type(exc).__name__,
                        "message": str(exc), "traceback": traceback.format_exc(),
                        "wall_seconds": time.monotonic() - t0, "terminal": True})
                    print(json.dumps({"stage": "failed", "case": case["id"], "method": method,
                                      "last_completed_critic_rounds": step, "error": str(exc)}), flush=True)
                    break
                step += chunk
                expected = primary_steps(method, step)
                load_manifest(study)
                path = study / "training" / case["id"] / method / f"round_{step:07d}"
                with publish_directory(path) as tmp:
                    checkpoint.save(tmp / "checkpoint.json")
                    write_new(tmp / "receipt.json", {"schema": "thrusterchain.study_chunk.v1", "case": case,
                        "method": method, "critic_rounds": step, "chunk_critic_rounds": chunk,
                        "primary_steps_total": expected, "optimizer_counts": dict(checkpoint.optimizer_step_counts),
                        "checkpoint_sha256": checkpoint.digest, "checkpoint_file_sha256": sha(tmp / "checkpoint.json"),
                        "wall_seconds": time.monotonic() - t0, "native_receipt": native_receipt})
                print(json.dumps({"stage": "training", "case": case["id"], "method": method,
                                  "critic_rounds": step, "target": spec["critic_rounds"]}), flush=True)


def verify_barrier(study, spec):
    for case in spec["cases"]:
        for method in spec["methods"]:
            found = checkpoints(study, case, method, spec)
            failure = study / "training" / case["id"] / method / "failure.json"
            if failure.exists():
                record = read_json(failure)
                if record.get("case") != case or record.get("method") != method or record.get("terminal") is not True:
                    raise RuntimeError("invalid failure record")
                if found and found[-1][0] == spec["critic_rounds"]:
                    raise RuntimeError("training cell cannot be both failed and complete")
            elif not found or found[-1][0] != spec["critic_rounds"]:
                raise RuntimeError(f"evaluation blocked: training incomplete for {case['id']} / {method}")


def verify_data_seeds(data, spec):
    from thrusterchain.shift_scenarios_v1 import _reset_seed
    all_data_seeds = {int(seed) for dataset in data.values() for seed in dataset.arrays["episode_seeds"]}
    # Paired data-seed roots intentionally share reset seeds across behaviors.
    derived = [_reset_seed(seed) for seed in spec["evaluation_seeds"]]
    if len(set(derived)) != len(derived): raise RuntimeError("derived evaluation reset seeds collide")
    if any(seed in all_data_seeds or _reset_seed(seed) in all_data_seeds for seed in spec["evaluation_seeds"]):
        raise RuntimeError("evaluation seeds overlap dataset reset seeds")


def evaluation_record(directory, name):
    path = directory / name
    expected = (directory / (name + ".sha256")).read_text(encoding="utf8").strip()
    if sha(path) != expected:
        raise RuntimeError("evaluation record digest mismatch")
    return read_json(path)


def write_evaluation_record(directory, name, record):
    path = directory / name
    write_new(path, record)
    with (directory / (name + ".sha256")).open("x", encoding="utf8") as handle:
        handle.write(sha(path) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_all(study, spec, *, resume):
    from thrusterchain.__main__ import checkpoint_factory
    from thrusterchain.shift_scenarios_v1 import evaluate_frozen_shifts
    verify_barrier(study, spec)
    data = {case["id"]: data_for(study, case) for case in spec["cases"]}
    verify_data_seeds(data, spec)
    for case in spec["cases"]:
        dataset = data[case["id"]]
        for method in spec["methods"]:
            if (study / "training" / case["id"] / method / "failure.json").exists():
                continue
            dest = study / "evaluation" / case["id"] / method
            if dest.exists():
                if not resume: raise RuntimeError("existing evaluation found; explicit --resume required")
                record_path = dest / ("result.json" if (dest / "result.json").is_file() else "failure.json")
                record = evaluation_record(dest, record_path.name)
                if record["case"] != case or record["method"] != method: raise RuntimeError("evaluation identity mismatch")
                continue
            cp = checkpoints(study, case, method, spec)[-1][1]
            load_manifest(study)
            try:
                learner = checkpoint_factory(cp.adapter_id)()
                learner.restore(cp)
                before = learner.checkpoint().digest
                policy = learner.freeze()
                results = {}
                for start in spec["starts"]:
                    results[start] = evaluate_frozen_shifts(policy, scenarios=spec["shifts"],
                        partitions={"dataset": dataset.arrays["episode_seeds"].tolist(), "selection": [],
                                    "final": spec["evaluation_seeds"]}, purpose="final", start=start,
                        policy_task_semantics="finite_horizon")
                if learner.checkpoint().digest != before: raise RuntimeError("evaluation changed learner state")
            except Exception as exc:
                with publish_directory(dest) as tmp:
                    write_evaluation_record(tmp, "failure.json", {"schema": "thrusterchain.study_evaluation_failure.v1",
                        "case": case, "method": method, "checkpoint_sha256": cp.digest,
                        "exception_type": type(exc).__name__, "message": str(exc),
                        "traceback": traceback.format_exc(), "terminal": True})
                print(json.dumps({"stage": "evaluation_failed", "case": case["id"],
                                  "method": method, "error": str(exc)}), flush=True)
                continue
            load_manifest(study)
            with publish_directory(dest) as tmp:
                write_evaluation_record(tmp, "result.json", {"schema": "thrusterchain.study_evaluation.v1", "case": case,
                    "method": method, "checkpoint_sha256": cp.digest, "dataset_sha256": dataset.digest,
                    "starts": results, "selection_interactions": 0})
            print(json.dumps({"stage": "evaluation", "case": case["id"], "method": method}), flush=True)


def status(study, spec):
    total = len(spec["cases"]) * len(spec["methods"])
    trained, evaluated, failed, pending = 0, 0, [], []
    incomplete, evaluation_failed = [], []
    for case in spec["cases"]:
        for method in spec["methods"]:
            completed = checkpoint_progress(study, case, method, spec)
            if completed == spec["critic_rounds"]: trained += 1
            elif (study / "training" / case["id"] / method / "failure.json").is_file():
                failure = study / "training" / case["id"] / method / "failure.json"
                try:
                    record = read_json(failure)
                except (OSError, ValueError):
                    pending.append({"case": case["id"], "method": method, "stage": "failure_record_being_written"})
                else:
                    if record.get("case") != case or record.get("method") != method or record.get("terminal") is not True:
                        raise RuntimeError("invalid failure record")
                    failed.append({"case": case["id"], "method": method, "record": str(failure)})
            else: incomplete.append({"case": case["id"], "method": method, "stage": "training"})
            evaluation = study / "evaluation" / case["id"] / method
            if (evaluation / "result.json").is_file(): evaluated += 1
            elif (evaluation / "failure.json").is_file():
                evaluation_failed.append({"case": case["id"], "method": method, "record": str(evaluation / "failure.json")})
    return {"profile": spec["profile"], "planned_runs": total, "training_complete": trained,
            "evaluation_complete": evaluated, "evaluation_failed": evaluation_failed,
            "training_failed": failed, "training_incomplete": incomplete,
            "publication_in_progress": pending,
            "snapshot_scope": "read-only receipt snapshot; checkpoint bytes, running-process state and runtime not attested"}


def report(study, spec):
    """Summarize all completed outcomes without dropping failed runs."""
    verify_barrier(study, spec)
    groups = {(method, behavior, start, shift): [] for method in spec["methods"] for behavior in BEHAVIORS
              for start in spec["starts"] for shift in spec["shifts"]}
    failures, episodes, training_costs = [], 0, []
    for case in spec["cases"]:
        dataset = data_for(study, case)
        for method in spec["methods"]:
            train_failure = study / "training" / case["id"] / method / "failure.json"
            completed = checkpoints(study, case, method, spec)
            completed_rounds = completed[-1][0] if completed else 0
            duration = sum(float(read_json(study / "training" / case["id"] / method /
                          f"round_{step:07d}" / "receipt.json")["wall_seconds"]) for step, _ in completed)
            failed_attempt = read_json(train_failure) if train_failure.is_file() else None
            training_costs.append({"case": case["id"], "method": method,
                "status": "failed" if failed_attempt else "completed",
                "critic_rounds_checkpointed": completed_rounds,
                "primary_updates_checkpointed": primary_steps(method, completed_rounds),
                "sampled_transition_draws_checkpointed": primary_steps(method, completed_rounds) * spec["batch_size"],
                "count_semantics": "checkpointed draws include repeats; unique coverage and uncheckpointed failed-attempt draws are not measured",
                "optimizer_counts_checkpointed": dict(completed[-1][1].optimizer_step_counts) if completed else {},
                "checkpointed_chunk_wall_seconds": duration,
                "failed_attempt_wall_seconds": failed_attempt["wall_seconds"] if failed_attempt else None,
                "failed_attempt_updates_and_draws": "unknown" if failed_attempt else "not applicable"})
            if train_failure.is_file():
                failures.append({"case": case["id"], "method": method, "stage": "training",
                                 "detail": failed_attempt})
                continue
            cp = completed[-1][1]
            root = study / "evaluation" / case["id"] / method
            if (root / "failure.json").is_file():
                detail = evaluation_record(root, "failure.json")
                if (detail.get("case") != case or detail.get("method") != method
                        or detail.get("checkpoint_sha256") != cp.digest or detail.get("terminal") is not True):
                    raise RuntimeError("evaluation failure record identity mismatch")
                failures.append({"case": case["id"], "method": method, "stage": "evaluation", "detail": detail})
                continue
            if not (root / "result.json").is_file():
                raise RuntimeError(f"report blocked: evaluation incomplete for {case['id']} / {method}")
            record = evaluation_record(root, "result.json")
            if (record["case"] != case or record["method"] != method or record["checkpoint_sha256"] != cp.digest
                    or record["dataset_sha256"] != dataset.digest or set(record["starts"]) != set(spec["starts"])):
                raise RuntimeError("evaluation record has mismatched run identity")
            for start in spec["starts"]:
                section = record["starts"][start]
                if (set(section["scenarios"]) != set(spec["shifts"])
                        or section["policy_sha256_before"] != section["policy_sha256_after"]):
                    raise RuntimeError("evaluation scenarios or frozen policy identity mismatch")
                for shift in spec["shifts"]:
                    rows = section["scenarios"][shift]["episodes"]
                    if ([r["episode_seed"] for r in rows] != sorted(spec["evaluation_seeds"])
                            or any(r["steps"] != 100 for r in rows)):
                        raise RuntimeError("evaluation episode identities or horizon mismatch")
                    raw, discounted = [float(r["return"]) for r in rows], [float(r["discounted_return"]) for r in rows]
                    if not all(math.isfinite(x) for x in raw + discounted):
                        raise RuntimeError("nonfinite reporting return")
                    episodes += len(rows)
                    groups[(method, case["behavior"], start, shift)].append({"case": case["id"],
                        "data_seed": case["data_seed"], "train_seed": case["train_seed"],
                        "dataset_sha256": dataset.digest, "checkpoint_sha256": cp.digest,
                        "episodes": len(rows), "mean_return": statistics.mean(raw),
                        "mean_discounted_return": statistics.mean(discounted)})
    summaries = []
    for (method, behavior, start, shift), runs in sorted(groups.items()):
        planned = sum(c["behavior"] == behavior for c in spec["cases"])
        raw = [r["mean_return"] for r in runs]
        disc = [r["mean_discounted_return"] for r in runs]
        summaries.append({"method": method, "behavior": behavior, "start": start, "shift": shift,
            "status": "complete" if len(runs) == planned and planned else "incomplete",
            "planned_training_replicates": planned, "completed_training_replicates": len(runs),
            "mean_return": statistics.mean(raw) if raw else None,
            "sample_sd_return_across_runs": statistics.stdev(raw) if len(raw) > 1 else None,
            "mean_discounted_return": statistics.mean(disc) if disc else None,
            "sample_sd_discounted_return_across_runs": statistics.stdev(disc) if len(disc) > 1 else None,
            "per_training_run": runs})
    paired = []
    for method in spec["methods"]:
        if method == "td3bc": continue
        for behavior in BEHAVIORS:
            planned = sum(c["behavior"] == behavior for c in spec["cases"])
            for start in spec["starts"]:
                for shift in spec["shifts"]:
                    control = {r["case"]: r for r in groups[("td3bc", behavior, start, shift)]}
                    candidate = {r["case"]: r for r in groups[(method, behavior, start, shift)]}
                    cases = sorted(set(control) & set(candidate))
                    differences = [candidate[k]["mean_discounted_return"] - control[k]["mean_discounted_return"] for k in cases]
                    paired.append({"method": method, "reference": "td3bc", "behavior": behavior, "start": start,
                        "shift": shift, "planned_pairs": planned, "observed_pairs": len(cases),
                        "status": "complete" if planned and len(cases) == planned else "incomplete",
                        "cases": cases, "discounted_return_differences": differences,
                        "mean_discounted_difference": statistics.mean(differences) if differences else None,
                        "sample_sd_difference": statistics.stdev(differences) if len(differences) > 1 else None})
    h1, h2 = [], []
    for method in spec["methods"]:
        random = {(r["data_seed"], r["train_seed"]): r for r in groups[(method, "random", "recorded", "baseline")]}
        strong = {(r["data_seed"], r["train_seed"]): r for r in groups[(method, "feedback_strong", "recorded", "baseline")]}
        for key in sorted(set(random) & set(strong)):
            h1.append({"method": method, "data_seed": key[0], "train_seed": key[1],
                       "difference": strong[key]["mean_discounted_return"] - random[key]["mean_discounted_return"]})
        for behavior in BEHAVIORS:
            for shift in spec["shifts"]:
                recorded = {r["case"]: r for r in groups[(method, behavior, "recorded", shift)]}
                current = {r["case"]: r for r in groups[(method, behavior, "strong_current", shift)]}
                for key in sorted(set(recorded) & set(current)):
                    h2.append({"method": method, "case": key, "shift": shift,
                               "difference": current[key]["mean_discounted_return"] - recorded[key]["mean_discounted_return"]})
    expected_h1 = len(spec["methods"]) * sum(c["behavior"] == "random" for c in spec["cases"])
    expected_h2 = len(spec["methods"]) * len(spec["cases"]) * len(spec["shifts"])
    mean_h1 = statistics.mean(x["difference"] for x in h1) if h1 else None
    mean_h2 = statistics.mean(x["difference"] for x in h2) if h2 else None
    hypotheses = {"H1": {"planned_pairs": expected_h1, "observed_pairs": len(h1), "differences": h1,
        "mean_strong_minus_random": mean_h1, "expected_direction_met": mean_h1 > 0 if expected_h1 and len(h1) == expected_h1 else None},
        "H2": {"planned_pairs": expected_h2, "observed_pairs": len(h2), "differences": h2,
        "mean_strong_current_minus_recorded": mean_h2, "expected_direction_met": mean_h2 < 0 if len(h2) == expected_h2 else None},
        "H3": {"direction": "none registered"},
        "interpretation": "descriptive contrasts; method and scenario outcomes within a replicate are dependent"}
    destination = study / "report.json"
    result = {"schema": "thrusterchain.full_benchmark_report.v1", "spec": spec,
        "failures": failures, "completed_evaluation_episodes": episodes,
        "completed_evaluation_steps": episodes * 100, "summaries": summaries, "paired_vs_td3bc": paired,
        "hypotheses": hypotheses,
        "training_costs": training_costs,
        "selection_environment_steps": 0,
        "uncertainty": "sample SD across paired dataset/training-seed replicates; no confidence interval",
        "holdout": "previously inspected scenario families; seed separation does not attest exposure",
        "budget_note": "matched critic rounds, not matched compute or dataset draws; unified reuses each sampled batch twice"}
    write_new(destination, result)
    return {"report": str(destination), "failed_cells": len(failures), "evaluation_episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--spec", type=Path, required=True); p.add_argument("--study", type=Path, required=True)
    for name in ("collect", "train", "evaluate", "status", "report"):
        p = sub.add_parser(name); p.add_argument("--study", type=Path, required=True)
        if name in ("train", "evaluate"): p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.spec, args.study)
        print(json.dumps({"status": "prepared", "profile": result["spec"]["profile"], "study": str(args.study)}))
        return
    study = args.study.resolve()
    if args.command == "status":
        manifest = read_manifest(study)
        print(json.dumps(status(study, manifest["spec"])))
        return
    with locked(study):
        manifest = load_manifest(study); spec = manifest["spec"]
        if args.command == "collect": collect(study, spec)
        elif args.command == "train": train_all(study, spec, resume=args.resume)
        elif args.command == "evaluate": evaluate_all(study, spec, resume=args.resume)
        elif args.command == "report": print(json.dumps(report(study, spec)))


if __name__ == "__main__": main()
