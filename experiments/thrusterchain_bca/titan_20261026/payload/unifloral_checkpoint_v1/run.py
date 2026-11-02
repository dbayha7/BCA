"""Separate correctness gate and bounded UniFloral checkpoint generation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT.parent / "unifloral_td3bc_validation_v1"
STAGE2 = ROOT.parent / "unifloral_bca_pipeline_validation_v1"
EVALUATOR = ROOT.parent / "unifloral_frozen_usefulness_v1"
BUNDLE = Path(r"C:\Users\David Bayha\Downloads\titan_transfer_20261022_v1\titan_transfer_20261022_v1") if os.name == "nt" else Path("/mnt/c/Users/David Bayha/Downloads/titan_transfer_20261022_v1/titan_transfer_20261022_v1")
DATA = BUNDLE / "research_runs/pilot_20261023T115408Z/regular_1280_2026102301.npz"
XLA = "--xla_gpu_triton_gemm_any=True"
sys.dont_write_bytecode = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def identities():
    expected_manifests = {STAGE1: "1e26f713822020275179845f30c8691467951745d6c7c829ff017b7045904516",
                          STAGE2: "6eb91c5bcc6c2edddc1ab93c19424307e12a3c8363bf9063fa4bbe2664180478"}
    result = {}
    for base in (ROOT, STAGE1, STAGE2, EVALUATOR):
        manifest = base / "prepared_files.json"
        digest = sha(manifest)
        if base in expected_manifests and digest != expected_manifests[base]:
            raise ValueError("Previously validated source manifest changed: " + str(base))
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        for name, wanted in files.items():
            target = (base / name).resolve()
            if not target.is_relative_to(base) or sha(target) != wanted:
                raise ValueError("Source identity changed: " + str(target))
        result[base.name] = {"manifest_sha256": digest, "verified_files": len(files)}
    return result


def runtime():
    import jax
    expected = {"jax": "0.6.2", "jaxlib": "0.6.2", "jax-cuda12-plugin": "0.6.2",
                "jax-cuda12-pjrt": "0.6.2", "numpy": "2.5.4", "flax": "0.10.7",
                "optax": "0.2.8", "distrax": "0.1.5", "chex": "0.1.90",
                "tensorflow-probability": "0.25.0", "scipy": "1.18.1"}
    actual = {name: importlib.metadata.version(name) for name in expected}
    value = {"python": sys.version, "executable": sys.executable, "packages": actual,
             "devices": [str(device) for device in jax.devices()],
             "device_kinds": [device.device_kind for device in jax.devices()],
             "backend": jax.default_backend(), "matmul_precision": str(jax.config.jax_default_matmul_precision),
             "jax_disable_jit": bool(jax.config.jax_disable_jit), "jax_enable_x64": bool(jax.config.jax_enable_x64),
             "environment": {key: os.environ.get(key) for key in
                 ("XLA_FLAGS", "JAX_PLATFORMS", "JAX_DEFAULT_MATMUL_PRECISION", "PYTHONUNBUFFERED")}}
    if actual != expected or value["backend"] != "gpu" or value["matmul_precision"] != "highest" or value["jax_disable_jit"] or value["jax_enable_x64"] or os.environ.get("XLA_FLAGS") != XLA:
        raise RuntimeError("Runtime differs from the explicitly validated provisional GPU configuration: " + json.dumps(value))
    return value


def verify_gate(path, sources):
    path = Path(path).resolve()
    raw = path.read_bytes()
    report = json.loads(raw)
    detail_path = path.parent / "checks.json"
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if (report.get("mode") != "checks" or report.get("status") != "pass"
            or report.get("exit_code") != 0 or report.get("identity_before") != sources
            or report.get("identity_after") != sources
            or report.get("checks_sha256") != sha(detail_path) or detail.get("status") != "pass"):
        raise ValueError("A passing exact-source checkpoint correctness gate is required")
    return {"report_sha256": hashlib.sha256(raw).hexdigest(), "checks_sha256": sha(detail_path)}


def train(native, output, source_ids, gate_path, resume):
    import numpy as np
    from data import load_training
    import training
    import checkpoint_io
    import export as inference_export
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    gate = verify_gate(gate_path, source_ids)
    dataset, heldout, heldout_ids, data_info = load_training(native, DATA)
    context, state = training.create(native, dataset, heldout, heldout_ids, config)
    bindings = {"config": config, "sources": source_ids,
        "datasetidentity": {"sha256": data_info["dataset_sha256"], "digest": data_info["digest"],
                            "split_receipt": data_info["split_receipt"]},
        "normalization": {"mean": np.asarray(context.obs_mean).tolist(), "std": np.asarray(context.obs_std).tolist()},
        "native_args": asdict(training.native_args(context)),
        "auxiliary_config": asdict(context.aux_config),
        "auxiliary_rng_schedule": context.auxiliary_rng_schedule,
        "gate_identity": gate, "training_stage_reviewed": True,
        "review_scope": "Explicit user launch of this declared bounded stage after a hash-bound passing correctness gate"}
    write(output / "bindings.json", bindings)
    write(output / "data_split_receipt.json", {key: value for key, value in data_info.items() if key != "scale_bank"})
    if resume:
        state, receipt = checkpoint_io.load(resume, state, bindings)
        training.validate(state)
        write(output / "resume_receipt.json", receipt)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=False)

    def save(label):
        tally = training.counts(state)
        path = checkpoints / (f"critic_{tally['critic']:06d}_" + label)
        receipt = checkpoint_io.save(path, state, bindings, tally)
        print(json.dumps({"event": "checkpoint_committed", "path": str(path), "counts": tally}), flush=True)
        return {"path": str(path), "receipt": receipt}

    last_saved = save("start")
    started = time.monotonic()
    try:
        while training.counts(state)["critic"] < config["critic_budget"]:
            state, metrics = training.step(context, state)
            count = training.counts(state)
            if count["critic"] % 100 == 0 or count["critic"] in config["refresh_at"]:
                entry = {"utc": utc(), "counts": count, "elapsed_seconds": time.monotonic() - started,
                         "metrics": metrics}
                with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
                    stream.flush()
                print(json.dumps({"event": "progress", **entry}, allow_nan=False), flush=True)
            if count["critic"] % 1000 == 0:
                last_saved = save("periodic")
        last_saved = save("terminal")
    except KeyboardInterrupt:
        last_saved = save("interrupt")
        return {"status": "paused_checkpoint_saved", "counts": training.counts(state),
                "latest_checkpoint": last_saved, "elapsed_seconds": time.monotonic() - started}
    except Exception as error:
        if hasattr(error, "last_committed_state"):
            state = error.last_committed_state
        last_saved = save("failure")
        return {"status": "blocked_or_failed", "counts": training.counts(state), "latest_checkpoint": last_saved,
                "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                "failure_diagnostics": getattr(error, "diagnostics", None),
                "failure_stage": getattr(error, "stage", None),
                "attempted_uncommitted_updates": getattr(error, "attempted_counts", None)}
    # Export failure cannot discard or replace the independently committed resume checkpoint.
    try:
        export_receipt = inference_export.export(context, state, data_info, bindings, output / "inference_export")
    except Exception as error:
        return {"status": "training_complete_export_blocked", "counts": training.counts(state),
                "latest_checkpoint": last_saved, "export_error": {"type": type(error).__name__, "message": str(error)}}
    return {"status": "bounded_checkpoint_complete", "counts": training.counts(state),
            "latest_checkpoint": last_saved, "export": export_receipt,
            "elapsed_seconds": time.monotonic() - started,
            "scientific_status": "Checkpoint generation only; no BCA efficacy or selected penalty coefficient"}


def worker(args, output):
    result = {"mode": args.mode, "started_utc": utc(), "status": "pending"}
    try:
        result["identity_before"] = identities()
        result["runtime"] = runtime()
        print(json.dumps({"event": "runtime_verified", **result["runtime"]}), flush=True)
        sys.path.insert(0, str(STAGE1))
        sys.path.insert(0, str(STAGE2))
        from native_loader import load_native
        native = load_native(STAGE1 / "upstream", STAGE1 / "provenance/unifloral.json")
        if args.mode == "checks":
            import checks
            detail = checks.run(native, output, {"sources": result["identity_before"],
                                               "production_config_sha256": sha(ROOT / "config.json")})
            write(output / "checks.json", detail)
            result["checks_sha256"] = sha(output / "checks.json")
            result["status"] = detail["status"]
        else:
            result.update(train(native, output, result["identity_before"], args.gate_report, args.resume))
    except Exception as error:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    try:
        result["identity_after"] = identities()
    except Exception as error:
        result["status"] = "blocked_or_failed"
        result["postcheck_error"] = str(error)
    result["finished_utc"] = utc()
    write(output / "worker.json", result)
    return 0 if result["status"] in ("pass", "bounded_checkpoint_complete", "paused_checkpoint_saved") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoints-test", dest="mode", action="store_const", const="checks")
    mode.add_argument("--train", dest="mode", action="store_const", const="train")
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-outside-upstream-constraints", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.allow_outside_upstream_constraints:
        parser.error("Explicit provisional-runtime acknowledgement required")
    if args.mode == "train" and not args.gate_report:
        parser.error("Training requires --gate-report from a passing correctness run")
    if args.mode != "train" and (args.gate_report or args.resume):
        parser.error("Gate/resume arguments apply only to the separate training stage")
    if args.worker:
        out = args.worker.resolve()
        if out.parent != ROOT or not out.is_dir():
            parser.error("Worker output must be a new local result directory")
        return worker(args, out)
    source_ids = identities()
    if args.mode == "train":
        verify_gate(args.gate_report, source_ids)
    if os.environ.get("XLA_FLAGS", "") not in ("", XLA):
        parser.error("Conflicting XLA_FLAGS; no override attempted")
    out = ROOT / (args.mode + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(exist_ok=False)
    env = dict(os.environ, JAX_PLATFORMS="cuda", JAX_DEFAULT_MATMUL_PRECISION="highest", XLA_FLAGS=XLA,
               XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--checkpoints-test" if args.mode == "checks" else "--train",
               "--allow-outside-upstream-constraints", "--worker", str(out)]
    if args.gate_report:
        command.extend(("--gate-report", str(args.gate_report.resolve())))
    if args.resume:
        command.extend(("--resume", str(args.resume.resolve())))
    print("Starting separate " + args.mode + " stage; output: " + str(out), flush=True)
    started = time.monotonic()
    # Child stdout stays attached for prompt progress and interrupt handling.
    # The child writes immutable receipts and append-only progress itself.
    code, interrupted = None, False
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        code = process.wait(timeout=600 if args.mode == "checks" else None)
    except subprocess.TimeoutExpired:
        # Only the bounded correctness child has a timeout, never a training child.
        process.kill()
        code = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupt received; allowing the child to save its last committed state.", flush=True)
        # Terminal Ctrl+C reaches both processes in their shared foreground group.
        # Do not send a duplicate SIGINT into the child's checkpoint writer.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("Child exit is still unconfirmed; inspect worker.json and COMMITTED markers. No forced training termination was issued.", flush=True)
        finally:
            signal.signal(signal.SIGINT, previous)
    worker_path = out / "worker.json"
    report = json.loads(worker_path.read_text(encoding="utf-8")) if worker_path.exists() else {
        "mode": args.mode, "status": "interrupted_outcome_uncertain" if interrupted else "worker_did_not_complete",
        "identity_before": source_ids}
    report.update(exit_code=code, elapsed_seconds=time.monotonic() - started, interrupted=interrupted)
    write(out / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}), flush=True)
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
