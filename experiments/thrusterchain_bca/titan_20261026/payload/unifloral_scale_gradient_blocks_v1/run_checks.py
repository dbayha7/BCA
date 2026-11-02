"""Pinned eight-block fitting gradient audit; default is a stdlib-only preflight."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RECOVERY = BASE / "unifloral_export_recovery_v2"
RECOVERED = RECOVERY / "recovery_20261023T214750_256386Z"
EXPORT = RECOVERED / "inference_export"
CHECKPOINT = EXPORT / "eligible_checkpoint.json"
CHECKPOINT_SHA = "a54856669e51fd86234c9e9624c62135cd111b0bd7fad07cc1c4fb8e252c899c"
COMMIT_SHA = "c9b58f1c12faeb318cae876b78a725a9ec094fc97a9a1ca1a558817221bf9392"
NATIVE_PROVENANCE_SHA = "7d0c0eccda6730c0b40bf9085cdbc31f565acbd8010db337b01ef5e64cf26a58"
NATIVE_REVISION = "f2dc1278eae18ed3e22c92255119c54885414d17"
NATIVE_SOURCE_SHA = "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee"
PINNED_MANIFESTS = {
    "unifloral_td3bc_validation_v1": "1e26f713822020275179845f30c8691467951745d6c7c829ff017b7045904516",
    "unifloral_bca_pipeline_validation_v1": "6eb91c5bcc6c2edddc1ab93c19424307e12a3c8363bf9063fa4bbe2664180478",
    "unifloral_frozen_usefulness_v1_seedfix": "c21f3ade3dddcb06068168de0a3c509847f4f0d8f4eb6a499d5b3a8b7d851fc2",
    "unifloral_export_recovery_v2": "05caa0b2db6c701fb20199e269c3b30faac04c6d914ff70faaf435e0549478b1",
    "unifloral_scale_gradient_validation_v1": "3216c6df1906c6ee220f9d39d57466c3ecb80559b151dc33de4097f5a4c6cb38",
}
REFERENCE_RELATIVE = "unifloral_scale_gradient_validation_v1/results_20261026T021136_510051Z"
REFERENCE_RESULTS = BASE / REFERENCE_RELATIVE
REFERENCE_REPORT_SHA = "28f00e0c63d9bfe647dde1065b23b485f8e28d63608d40fed81658b9f29757f6"
REFERENCE_CHECKS_SHA = "f634a043ea5158829b09aeae348a6489d00ef637daa982a272dff63c6fb9ba5c"
FIXED_CONTRAST_FACTOR = 31.372587203979492
RECOVERY_EVIDENCE = {
    "report.json": "ab5d8b696b01729542c615fa7f178fcd7ea3e793a9bec3d9cbd36a12e6b750dd",
    "inference_parity.json": "7de3014783569d838f7274334f88d0d732a212372330ecc21d31940c46757dcb",
}
ROLES = {"parameters": "parameters.msgpack", "scale_bank": "scale_bank.npz",
         "training_receipt": "training_receipt.json", "normalization_receipt": "normalization_receipt.json",
         "calibration_receipt": "calibration_receipt.json"}
XLA = "--xla_gpu_triton_gemm_any=True"
TIMEOUT_SECONDS = 300
sys.dont_write_bytecode = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(raw):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), parse_constant=invalid, object_pairs_hook=unique)
    encoded(value)
    return value


def write(path, value):
    raw = encoded(value)  # Validate before creating an exclusive destination.
    with Path(path).open("xb") as stream:
        stream.write(raw)


def checked_file(base, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Expected a relative component filename")
    candidate = base / relative
    target = candidate.resolve()
    if not target.is_relative_to(base.resolve()) or candidate.is_symlink() or not target.is_file():
        raise ValueError("Unsafe or missing pinned file: " + relative)
    return target.read_bytes()


def verify():
    """Only pinned sources/export and the approved passing fitting-audit receipts."""
    sources = {}
    for folder in (ROOT.name, *PINNED_MANIFESTS):
        base = BASE / folder
        raw = checked_file(base, "prepared_files.json")
        digest = sha(raw)
        if folder in PINNED_MANIFESTS and digest != PINNED_MANIFESTS[folder]:
            raise ValueError("Pinned source manifest changed: " + folder)
        manifest = read_json(raw)
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Invalid source manifest: " + folder)
        if folder == ROOT.name and not {"run_checks.py", "checks.py", "protocol.json"}.issubset(files):
            raise ValueError("New source manifest lacks the runner, checks or protocol")
        for name, wanted in files.items():
            if sha(checked_file(base, name)) != wanted:
                raise ValueError("Pinned source changed: " + folder + "/" + name)
        sources[folder] = {"manifest_sha256": digest, "verified_files": len(files)}
    native_base = BASE / "unifloral_td3bc_validation_v1"
    provenance_raw = checked_file(native_base, "provenance/unifloral.json")
    if sha(provenance_raw) != NATIVE_PROVENANCE_SHA:
        raise ValueError("Pinned native provenance manifest changed")
    provenance = read_json(provenance_raw)
    native_files = provenance.get("source_sha256")
    if (provenance.get("revision") != NATIVE_REVISION or not isinstance(native_files, dict)
            or len(native_files) != 40
            or native_files.get("algorithms/td3_bc.py") != NATIVE_SOURCE_SHA):
        raise ValueError("Pinned native revision or 40-file snapshot declaration differs")
    vendor = native_base / "upstream"
    if not vendor.is_dir() or vendor.is_symlink():
        raise ValueError("Missing or unsafe native snapshot directory")
    entries = list(vendor.rglob("*"))
    if (any(entry.is_symlink() for entry in entries)
            or {entry.relative_to(vendor).as_posix() for entry in entries if entry.is_file()} != set(native_files)):
        raise ValueError("Native snapshot has missing, additional or symlinked files")
    for relative, wanted in native_files.items():
        if sha(checked_file(vendor, relative)) != wanted:
            raise ValueError("Pinned native snapshot file changed: " + relative)
    native_snapshot = {"revision": NATIVE_REVISION, "manifest_sha256": NATIVE_PROVENANCE_SHA,
                       "source_sha256": NATIVE_SOURCE_SHA, "verified_files": len(native_files)}
    protocol_raw = checked_file(ROOT, "protocol.json")
    protocol = read_json(protocol_raw)
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object")
    fixed = {"checkpoint_manifest_sha256": CHECKPOINT_SHA, "timeout_seconds": TIMEOUT_SECONDS,
             "fitting_rows": 256, "block_size": 32, "block_starts": list(range(0, 256, 32)),
             "fixed_contrast_factor": FIXED_CONTRAST_FACTOR, "kappa_grid": [0, 0.03, 0.1, 0.3, 1, 3, 10],
             "reference_results": REFERENCE_RELATIVE, "reference_report_sha256": REFERENCE_REPORT_SHA,
             "reference_checks_sha256": REFERENCE_CHECKS_SHA,
             "candidate_actions_generated": 0, "scientific_validation_status": "incomplete_design_review_required",
             "optimizer_updates": 0, "calibrator_updates": 0, "calibration_refreshes": 0,
             "simulator_steps": 0, "development_labels_read": False, "confirmation_data_read": False}
    if any(type(protocol.get(key)) is not type(value) or protocol[key] != value
           for key, value in fixed.items()):
        raise ValueError("Protocol differs from the bounded fitting-only, zero-update scope")
    manifest_raw = checked_file(EXPORT, "eligible_checkpoint.json")
    if sha(manifest_raw) != CHECKPOINT_SHA:
        raise ValueError("The fixed trained export manifest changed")
    manifest = read_json(manifest_raw)
    commit_raw = checked_file(EXPORT, "COMMITTED.json")
    if sha(commit_raw) != COMMIT_SHA:
        raise ValueError("The fixed export commit marker changed")
    commit = read_json(commit_raw)
    expected_names = set(ROLES.values()) | {"eligible_checkpoint.json"}
    if (commit.get("kind") != "inference_export"
            or commit.get("schema") != "unifloral.committed_artifact.v1"
            or set(commit.get("files", {})) != expected_names
            or {entry.name for entry in EXPORT.iterdir()} != expected_names | {"COMMITTED.json"}):
        raise ValueError("Incomplete or modified export file set")
    role_hashes = {}
    for name, expected in commit["files"].items():
        raw = checked_file(EXPORT, name)
        if expected != {"sha256": sha(raw), "bytes": len(raw)}:
            raise ValueError("Committed export component changed: " + name)
        for role, filename in ROLES.items():
            if filename == name:
                role_hashes[role] = sha(raw)
    if manifest.get("files") != {role: {"path": name, "sha256": role_hashes[role]}
                                 for role, name in ROLES.items()}:
        raise ValueError("Export role hashes disagree with the commit marker")
    evidence = {}
    for name, expected in RECOVERY_EVIDENCE.items():
        raw = checked_file(RECOVERED, name)
        if sha(raw) != expected:
            raise ValueError("Passing recovery evidence changed: " + name)
        value = read_json(raw)
        if value.get("status") != "pass":
            raise ValueError("Recovery or inference parity did not pass")
        evidence[name] = sha(raw)
    prior_report_raw = checked_file(REFERENCE_RESULTS, "report.json")
    prior_checks_raw = checked_file(REFERENCE_RESULTS, "checks.json")
    if sha(prior_report_raw) != REFERENCE_REPORT_SHA or sha(prior_checks_raw) != REFERENCE_CHECKS_SHA:
        raise ValueError("The approved fitting-audit reference receipts changed")
    prior_report, prior_checks = read_json(prior_report_raw), read_json(prior_checks_raw)
    rows = prior_checks.get("checks", [])
    if (prior_report.get("status") != "pass" or prior_report.get("exit_code") != 0
            or prior_report.get("runtime_verified") is not True or prior_report.get("zero_work_verified") is not True
            or prior_report.get("check_count") != 11 or prior_report.get("checks_sha256") != REFERENCE_CHECKS_SHA
            or prior_checks.get("status") != "pass" or prior_checks.get("passed") is not True
            or len(rows) != 11 or any(row.get("status") != "pass" for row in rows)):
        raise ValueError("Reference fitting audit did not complete all 11 checks")
    prior_zero = check_zero_work(prior_checks)
    if (prior_report.get("zero_work") != prior_zero or prior_zero.get("development_labels_read") is not False
            or prior_zero.get("confirmation_data_read") is not False):
        raise ValueError("Reference zero-work or fitting-only evidence is inconsistent")
    prior_identity = {"sources": {name: sources[name] for name in PINNED_MANIFESTS},
                      "native_snapshot": native_snapshot,
                      "protocol_sha256": sha(checked_file(BASE / "unifloral_scale_gradient_validation_v1", "protocol.json")),
                      "checkpoint_manifest_sha256": CHECKPOINT_SHA, "checkpoint_commit_sha256": COMMIT_SHA,
                      "checkpoint_role_hashes": role_hashes, "passing_recovery_evidence": evidence}
    if prior_report.get("identity_before") != prior_identity or prior_report.get("identity_after") != prior_identity:
        raise ValueError("Reference audit did not use the exact current pinned sources and checkpoint")
    by_name = {row["name"]: row for row in rows}
    if (len(by_name) != 11 or by_name["detached_fitting_statistics_live_action_route"]["details"]["factor"]
            != FIXED_CONTRAST_FACTOR):
        raise ValueError("Fixed contrast factor differs from the approved live-action diagnostic")
    reference = {"relative_path": REFERENCE_RELATIVE, "report_sha256": REFERENCE_REPORT_SHA,
                 "checks_sha256": REFERENCE_CHECKS_SHA, "verified_checks": len(rows),
                 "fixed_contrast_factor": FIXED_CONTRAST_FACTOR,
                 "block_zero_anchor_sha256": sha(encoded(by_name["actual_candidate_loss_and_gradient_identity"]["details"])),
                 "zero_work_verified": True}
    return {"sources": sources, "native_snapshot": native_snapshot, "protocol_sha256": sha(protocol_raw),
            "checkpoint_manifest_sha256": sha(manifest_raw), "checkpoint_commit_sha256": sha(commit_raw),
            "checkpoint_role_hashes": role_hashes, "passing_recovery_evidence": evidence,
            "passing_fitting_reference": reference}, protocol


def runtime():
    """Worker-only scientific import; exact previously validated provisional stack."""
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
    if (actual != expected or sys.version_info[:3] != (3, 12, 3)
            or sys.executable != "/home/dbayha/bca-gpu/bin/python"
            or value["backend"] != "gpu" or value["devices"] != ["cuda:0"]
            or value["device_kinds"] != ["NVIDIA GeForce RTX 5070 Ti"]
            or value["matmul_precision"] != "highest" or value["jax_disable_jit"] or value["jax_enable_x64"]
            or os.environ.get("XLA_FLAGS") != XLA or os.environ.get("JAX_PLATFORMS") != "cuda"):
        raise RuntimeError("Runtime differs from the fixed provisional GPU configuration: " + json.dumps(value))
    return value


def check_zero_work(detail):
    zero = detail.get("zero_work", {})
    fields = ("optimizer_updates", "calibrator_updates", "calibration_refreshes", "simulator_steps",
              "confirmation_starts", "forbidden_api_calls")
    if (any(type(zero.get(key)) is not int or zero[key] != 0 for key in fields)
            or zero.get("verified") is not True or zero.get("frozen_state_unchanged") is not True
            or not isinstance(zero.get("frozen_state_sha256_before"), str)
            or len(zero["frozen_state_sha256_before"]) != 64
            or zero["frozen_state_sha256_before"] != zero.get("frozen_state_sha256_after")):
        raise ValueError("Checks did not verify zero work and unchanged frozen state")
    return zero


def worker(output, expected_identity):
    result = {"started_utc": utc(), "status": "blocked_or_failed", "runtime_verified": False,
              "zero_work_verified": False, "claim_scope": "Fixed-factor gradients over eight fitting blocks; no efficacy or normalization approval",
              "scientific_validation_status": "incomplete_design_review_required"}
    with (output / "worker_stages.jsonl").open("x", encoding="utf-8") as stages:
        def stage(name):
            stages.write(json.dumps({"utc": utc(), "stage": name}) + "\n")
            stages.flush()

        try:
            stage("source_and_checkpoint_precheck")
            identities, protocol = verify()
            result["identity_before"] = identities
            if sha(encoded(identities)) != expected_identity:
                raise ValueError("Worker identities differ from supervisor preflight")
            stage("runtime_verification")
            result["runtime"] = runtime()
            result["runtime_verified"] = True
            stage("fitting_only_checks")
            spec = importlib.util.spec_from_file_location("_scale_gradient_blocks_checks", ROOT / "checks.py")
            checks = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = checks
            spec.loader.exec_module(checks)
            detail = checks.run(output, CHECKPOINT, protocol)
            write(output / "checks.json", detail)
            result["checks_sha256"] = sha((output / "checks.json").read_bytes())
            result["check_status"] = detail.get("status")
            zero = check_zero_work(detail)
            result["zero_work"] = zero
            result["zero_work_verified"] = True
            rows = detail.get("checks", [])
            if (detail.get("status") != "pass" or detail.get("passed") is not True
                    or not rows or any(row.get("status") != "pass" for row in rows)):
                raise ValueError("One or more bounded scale/gradient checks failed or were skipped")
            result["status"] = "pass"
            result["check_count"] = len(rows)
        except BaseException as error:
            result["status"] = "blocked_or_failed"
            result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        finally:
            stage("source_and_checkpoint_postcheck")
            try:
                result["identity_after"], _ = verify()
                if result.get("identity_before") != result["identity_after"]:
                    raise ValueError("Source or checkpoint identities changed during checks")
            except Exception as error:
                result["status"] = "blocked_or_failed"
                result["postcheck_error"] = str(error)
            result["finished_utc"] = utc()
            write(output / "worker.json", result)
    return 0 if result["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="Only verify fixed source/checkpoint bytes; default")
    modes.add_argument("--execute", action="store_true", help="Run the bounded fitting-only checks")
    parser.add_argument("--allow-outside-upstream-constraints", action="store_true")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-identity-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_output:
        if not args.allow_outside_upstream_constraints or not args.expected_identity_sha256 or args.preflight:
            parser.error("Worker requires provisional-runtime acknowledgement and supervisor identities")
        output = args.worker_output.resolve()
        if output.parent != ROOT or not output.name.startswith("results_") or not output.is_dir():
            parser.error("Worker output must be the new local supervisor result directory")
        return worker(output, args.expected_identity_sha256)
    if args.execute and not args.allow_outside_upstream_constraints:
        parser.error("Execution requires explicit acknowledgement that JAX/NumPy exceed upstream constraints")
    output = ROOT / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    result = {"started_utc": utc(), "mode": "execute" if args.execute else "preflight",
              "status": "blocked_or_failed", "runtime_verified": False, "zero_work_verified": False,
              "timeout_seconds": TIMEOUT_SECONDS, "claim_scope": "Eight fitting blocks with a fixed factor; no efficacy, confirmation or normalization approval",
              "scientific_validation_status": "incomplete_design_review_required"}
    try:
        result["identity_before"], _ = verify()
        if not args.execute:
            result.update(status="preflight_passed", scientific_execution=False)
        else:
            if os.environ.get("XLA_FLAGS", "") not in ("", XLA):
                raise ValueError("Conflicting XLA_FLAGS; no override attempted")
            env = dict(os.environ, JAX_PLATFORMS="cuda", JAX_DEFAULT_MATMUL_PRECISION="highest",
                       XLA_FLAGS=XLA, XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
                       PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                       "--worker-output", str(output), "--allow-outside-upstream-constraints",
                       "--expected-identity-sha256", sha(encoded(result["identity_before"]))]
            print("Starting bounded fitting-only checks; output: " + str(output), flush=True)
            started = time.monotonic()
            with (output / "worker.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    result["exit_code"] = process.wait(timeout=TIMEOUT_SECONDS)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    result["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
                    result["interrupted"] = isinstance(error, KeyboardInterrupt)
                    process.kill()  # Only this newly created, bounded zero-update worker.
                    result["exit_code"] = process.wait()
            result["elapsed_seconds"] = time.monotonic() - started
            if (output / "worker.json").is_file():
                child = read_json((output / "worker.json").read_bytes())
                result["worker_sha256"] = sha((output / "worker.json").read_bytes())
                result["worker_status"] = child.get("status")
                result["runtime_verified"] = child.get("runtime_verified", False)
                result["zero_work_verified"] = child.get("zero_work_verified", False)
                for key in ("runtime", "zero_work", "error", "postcheck_error", "check_count"):
                    if key in child:
                        result[key] = child[key]
                if child.get("checks_sha256"):
                    detail_raw = (output / "checks.json").read_bytes()
                    if sha(detail_raw) != child["checks_sha256"]:
                        raise ValueError("Worker check detail hash mismatch")
                    result["checks_sha256"] = sha(detail_raw)
                if (result["exit_code"] == 0 and child.get("status") == "pass"
                        and child.get("identity_before") == result["identity_before"] == child.get("identity_after")
                        and child.get("runtime_verified") is True and child.get("zero_work_verified") is True
                        and not result.get("timed_out") and not result.get("interrupted")):
                    result["status"] = "pass"
            else:
                result["worker_outcome"] = "No completed worker receipt; inspect worker log and stage journals"
    except BaseException as error:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        try:
            result["identity_after"], _ = verify()
            if result.get("identity_before") != result["identity_after"]:
                raise ValueError("Source or checkpoint identities changed during the supervisor run")
        except Exception as error:
            result["status"] = "blocked_or_failed"
            result["postcheck_error"] = str(error)
        result["finished_utc"] = utc()
        write(output / "report.json", result)
    print(json.dumps({"status": result["status"], "report": str(output / "report.json"),
                      "runtime_verified": result["runtime_verified"],
                      "zero_work_verified": result["zero_work_verified"]}), flush=True)
    return 0 if result["status"] in ("pass", "preflight_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
