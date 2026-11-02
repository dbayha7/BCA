"""One reviewed Path A training job. Importing this file never loads a model.

The authenticated launcher supplies approval, services, output reservation and
the global deadline. Old experiment approvals and checkpoints are not accepted.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNTIME_DIR = REPO / "experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1"
PAYLOAD = REPO / "experiments/thrusterchain_bca/titan_20261026/payload"
EST = timezone(timedelta(hours=-5))
KEYS = ("critic", "actor", "auxiliary", "calibrations")
CAPS = dict(zip(KEYS, (10000, 5000, 10000, 1)))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        value = sys.modules[name]
        require(Path(value.__file__).resolve() == path, "Module path collision: " + name)
        return value
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def load_runtime(binding):
    """Called only inside the authenticated, approved Linux worker, never by tests."""
    require(os.name == "posix", "Manual approved Linux worker required")
    expected_data_receipt = read(HERE / "DATA_RECEIPT.json")
    required = {"JAX_PLATFORMS": "cuda", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                "JAX_ENABLE_X64": "false", "XLA_FLAGS": "--xla_gpu_triton_gemm_any=True",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                "BCA_TRANSFER_BUNDLE_ROOT": str(PAYLOAD / "transfer_bundle")}
    require(not any(k in os.environ and os.environ[k] != v for k, v in required.items()),
            "Conflicting frozen worker environment")
    if "jax" not in sys.modules:
        os.environ.update(required)
    else:
        require(all(os.environ.get(k) == v for k, v in required.items()), "JAX was loaded before frozen worker settings")
    sys.dont_write_bytecode = True
    _module("common", RUNTIME_DIR / "common.py")
    _module("state_exact", RUNTIME_DIR / "state_exact.py")
    runtime = _module("_two_scenario_path_a_runtime", RUNTIME_DIR / "runtime.py")
    native = runtime.load_native()
    runtime.runtime_receipt(native)
    from flax import serialization
    return SimpleNamespace(runtime=runtime, native=native, binding=binding, serialization=serialization,
                           expected_data_receipt=expected_data_receipt)


def _prepare(protocol, condition, seed_index, services, training_review):
    require(condition in ("honeypot", "bottleneck") and type(seed_index) is int and 0 <= seed_index < 3,
            "Unreviewed condition or seed")
    prepared = services.binding.prepare(services.native, condition, protocol=protocol)
    if hasattr(services, "expected_data_receipt"):
        expected = services.expected_data_receipt
        require(isinstance(expected, dict) and isinstance(expected.get("conditions"), dict) and
                isinstance(expected["conditions"].get(condition), dict) and
                isinstance(expected["conditions"][condition].get("metadata"), dict),
                "Missing prepared condition data receipt")
        require(prepared.metadata == expected["conditions"][condition]["metadata"],
                "Runtime data/calibration/normalization binding differs from the prepared exact receipt")
    adapter, state = services.binding.I.initialize(services.runtime, services.native, protocol, seed_index,
        condition, prepared, training_review=training_review)
    return prepared, adapter, state


def _exact_roundtrip(services, adapter, state):
    raw = services.serialization.to_bytes(state)
    diagnostic = {}
    restored = services.runtime.restore_exact(adapter, state, raw, diagnostic=diagnostic)
    expected = services.runtime.tree_digest(state)
    require(services.runtime.tree_digest(restored) == expected, "Exact complete-state roundtrip differs")
    return restored, {"state_digest": expected, "bytes_sha256": digest(raw), "exact_restore": diagnostic}


def _checkpoint_receipt(services, prepared, adapter, state, protocol, condition, seed_index, arm, raw):
    I, R = services.binding.I, services.runtime
    return {"schema": "two_scenario.path_a.full_state.v1", "condition": condition,
        "seed_index": seed_index, "arm": arm, "protocol_sha256": I.digest(protocol),
        "dataset_sha256": prepared.metadata["dataset_sha256"],
        "geometry_sha256": prepared.metadata["geometry_sha256"],
        "prepared_metadata_sha256": I.digest(prepared.metadata),
        "normalization_sha256": I.digest(R.host(adapter.context.normalization)),
        "normalization": R.host(adapter.context.normalization),
        "state_digest": R.tree_digest(state), "checkpoint_sha256": digest(raw),
        "counts": R.host(R.validate(adapter, state)["counts"]),
        "contains": "native actor/critics/targets/optimizers, native phase and held batch, native and auxiliary RNG, auxiliary optimizer, calibration reference and counters"}


def _read_checkpoint(directory, services, prepared, adapter, template, protocol, condition, seed_index, arm):
    directory = Path(directory)
    report, receipt = read(directory / "report.json"), read(directory / "checkpoint.json")
    require(report.get("status") == "completed_training_job" and report.get("execution_counts_known") is True,
            "Checkpoint producer is incomplete")
    require(report.get("condition") == condition and report.get("seed_index") == seed_index and report.get("arm") == arm,
            "Wrong producer condition, seed or arm")
    require(report.get("checkpoint_receipt_sha256") == digest((directory / "checkpoint.json").read_bytes()),
            "Checkpoint metadata differs from its completed report")
    require(receipt.get("arm") == arm and receipt.get("prepared_metadata_sha256") == services.binding.I.digest(prepared.metadata),
            "Checkpoint retention/calibration binding differs")
    require(report.get("counts", {}).get("attempted") == CAPS and report["counts"].get("completed") == CAPS,
            "Producer physical job budget differs")
    raw = (directory / "full_state.msgpack").read_bytes()
    restored, diagnostic = services.binding.I.restore_bound(services.runtime, adapter, template, raw, receipt,
        prepared=prepared, expected_sha256=receipt["checkpoint_sha256"], protocol=protocol, seed_index=seed_index)
    observed = services.runtime.host(services.runtime.validate(adapter, restored)["counts"])
    require(observed == receipt["counts"], "Restored cumulative counters differ from checkpoint metadata")
    expected_critic, expected_cal = (10000, 1) if arm == "warmup" else (20000, 2)
    require(observed["critic"] == expected_critic and observed["refresh_accepted"] == expected_cal,
            "Checkpoint is not the required fixed training boundary")
    return restored, {"report_sha256": digest((directory / "report.json").read_bytes()),
                      "checkpoint_sha256": digest(raw), "restore": diagnostic}


def _branch_start(services, prepared, adapter, state, protocol, condition, seed_index, arm, warmup_dir):
    require(warmup_dir is not None and arm in ("native", "bca"), "A branch requires its own completed warmup")
    warm, lineage = _read_checkpoint(warmup_dir, services, prepared, adapter, state, protocol, condition, seed_index, "warmup")
    pairs = services.binding.I.fork_pair(services.runtime, adapter, warm, protocol, seed_index)
    adapter, state = pairs[arm]
    require(services.runtime.tree_digest(pairs["native"][1]) == services.runtime.tree_digest(pairs["bca"][1]),
            "Paired branch states differ")
    lineage["paired_fork_state_digest"] = services.runtime.tree_digest(state)
    lineage["warmup_state_digest"] = services.runtime.tree_digest(warm)
    return adapter, state, lineage


class PhysicalLedger:
    """Per-job physical work, including an interrupted uncommitted microstep."""
    def __init__(self):
        self.attempted = dict.fromkeys(KEYS, 0)
        self.completed = dict.fromkeys(KEYS, 0)
        self.last_attempted = dict.fromkeys(KEYS, 0)
        self.last_completed = dict.fromkeys(KEYS, 0)

    def begin_step(self):
        self.last_attempted = dict.fromkeys(KEYS, 0)
        self.last_completed = dict.fromkeys(KEYS, 0)

    def dispatch(self, event):
        for key in KEYS:
            source = "refresh" if key == "calibrations" else key
            attempted, completed = int(event["attempted"][source]), int(event["completed"][source])
            require(0 <= self.last_completed[key] <= completed <= attempted and self.last_attempted[key] <= attempted,
                    "Nonmonotone runtime dispatch counts")
            self.attempted[key] += attempted - self.last_attempted[key]
            self.completed[key] += completed - self.last_completed[key]
            self.last_attempted[key], self.last_completed[key] = attempted, completed
            require(self.attempted[key] <= CAPS[key], "Physical job cap exceeded: " + key)

    def receipt(self):
        return {"attempted": dict(self.attempted), "completed": dict(self.completed),
                "execution_counts_known": self.attempted == self.completed}


def run_job(protocol, condition, seed_index, arm, output, warmup_dir=None, deadline=None, *, services, training_review):
    """Exactly 10k additional critic/aux steps per job; branches finish at 20k."""
    require(arm in ("warmup", "native", "bca"), "Unknown training job")
    require(callable(deadline), "Global deadline callback required")
    output = Path(output)
    require(output.is_dir() and not (output / "report.json").exists() and not (output / "TRAINING_START.json").exists(),
            "New launcher-reserved output directory required")
    start, ledger, state, adapter = time.monotonic(), PhysicalLedger(), None, None
    report = {"schema": "two_scenario.path_a.training.v1", "status": "failed_training_job", "condition": condition,
              "seed_index": seed_index, "arm": arm, "started_est": datetime.now(EST).isoformat(),
              "error": None, "simulator_steps": 0, "no_automatic_retry": True, "resume_supported": False,
              "protocol_sha256": services.binding.I.digest(protocol), "training_updates_are_production": True}
    save(output / "TRAINING_START.json", {**report, "physical_caps": CAPS,
         "missing_final_receipt": "Incomplete; persisted counts are a lower bound and physical_caps are upper bounds, never assume zero."})
    try:
        deadline()
        prepared, adapter, state = _prepare(protocol, condition, seed_index, services, training_review)
        report["prepared_metadata"] = prepared.metadata
        report["initial_state_digest"] = services.runtime.tree_digest(state)
        report["runtime"] = services.runtime.host(adapter.context.runtime)
        report["normalization"] = services.runtime.host(adapter.context.normalization)
        state, report["initial_exact_roundtrip"] = _exact_roundtrip(services, adapter, state)
        if arm != "warmup":
            adapter, state, report["lineage"] = _branch_start(services, prepared, adapter, state, protocol,
                condition, seed_index, arm, warmup_dir)
        report["job_start_state_digest"] = services.runtime.tree_digest(state)
        begin = services.runtime.validate(adapter, state)["counts"]
        require(begin["critic"] == (0 if arm == "warmup" else 10000), "Wrong starting update boundary")
        for local_step in range(1, 10001):
            deadline()
            ledger.begin_step()
            def dispatch(event):
                ledger.dispatch(event)
                if event["stage"] == "calibration_refresh":
                    append(output / "progress.jsonl", {"local_step": local_step, "dispatch": event,
                        "physical_counts": ledger.receipt(), "observed_est": datetime.now(EST).isoformat()})
            state, metrics = services.runtime.step(adapter, state, progress=dispatch)
            require(ledger.attempted == ledger.completed, "Completed runtime step has unresolved physical work")
            expected = {"critic": local_step, "actor": local_step // 2, "auxiliary": local_step,
                        "calibrations": int(local_step >= (10000 if arm == "warmup" else 5000))}
            require(ledger.completed == expected, "Physical counts disagree with fixed schedule")
            if local_step == 2:
                state, report["first_two_updates_exact_roundtrip"] = _exact_roundtrip(services, adapter, state)
            if local_step % 100 == 0 or local_step == 1:
                append(output / "progress.jsonl", {"local_step": local_step,
                    "cumulative_counts": services.runtime.host(metrics["counts"]), "physical_counts": ledger.receipt(),
                    "observed_est": datetime.now(EST).isoformat()})
        deadline()
        state, report["final_exact_roundtrip"] = _exact_roundtrip(services, adapter, state)
        raw = services.serialization.to_bytes(state)
        receipt = _checkpoint_receipt(services, prepared, adapter, state, protocol, condition, seed_index, arm, raw)
        require(receipt["counts"]["critic"] == (10000 if arm == "warmup" else 20000), "Final cumulative boundary differs")
        with (output / "full_state.msgpack").open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        save(output / "checkpoint.json", receipt)
        report.update(status="completed_training_job", checkpoint_receipt_sha256=digest((output / "checkpoint.json").read_bytes()),
                      checkpoint_sha256=digest(raw), final_cumulative_counts=receipt["counts"])
    except BaseException as exc:
        report.update(status="failed_training_job", error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    report.update(counts=ledger.receipt(), execution_counts_known=ledger.attempted == ledger.completed,
                  elapsed_seconds=time.monotonic() - start, finished_est=datetime.now(EST).isoformat())
    # A persistence failure propagates: the supervisor must never infer a completed job from checkpoints alone.
    save(output / "report.json", report)
    return report


def restore_final(protocol, condition, seed_index, arm, branch_dir, warmup_dir, *, services, training_review):
    """Load an authenticated final policy and auxiliary reference, with zero updates."""
    require(arm in ("native", "bca"), "Only final native/BCA policies can be evaluated")
    prepared, adapter, state = _prepare(protocol, condition, seed_index, services, training_review)
    adapter, state, _ = _branch_start(services, prepared, adapter, state, protocol, condition, seed_index, arm, warmup_dir)
    state, _ = _read_checkpoint(branch_dir, services, prepared, adapter, state, protocol, condition, seed_index, arm)
    return adapter, state, services.native, services.runtime
