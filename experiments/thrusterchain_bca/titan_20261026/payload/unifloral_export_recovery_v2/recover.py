"""Recover the fixed terminal export; zero optimizer updates or simulator steps."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from integrity import content_identity, mapping_order_audit

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "unifloral_checkpoint_v1"
RUN = ORIGINAL / "train_20261023T183248_552877Z"
CHECKPOINT = RUN / "checkpoints/critic_020000_terminal"
GATE = ORIGINAL / "checks_20261023T180300_619035Z/report.json"
EVALUATOR = ROOT.parent / "unifloral_frozen_usefulness_v1_seedfix"
XLA = "--xla_gpu_triton_gemm_any=True"
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def module(name, path):
    if name in sys.modules:
        raise RuntimeError("Private module already loaded: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify():
    """Read/hash only; never update the preserved source or failed report."""
    repair = read(ROOT / "REPAIR_RECEIPT.json")
    for relative, wanted in repair["preserved_files"].items():
        path = (ROOT.parent / relative).resolve()
        if not path.is_relative_to(ROOT.parent) or sha(path) != wanted:
            raise ValueError("Preserved source/evidence changed: " + relative)
    for base in (ROOT, EVALUATOR):
        manifest = read(base / "prepared_files.json")
        for relative, wanted in manifest["files"].items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base) or sha(path) != wanted:
                raise ValueError("Prepared repair source changed: " + str(path))
    original = (ORIGINAL / "export.py").read_bytes()
    fixed = (ROOT / "export_seedfix.py").read_bytes()
    old, new = b"0 <= seed < 2**32", b"0 <= seed < 2**63"
    if original.count(old) != 1 or original.replace(old, new) != fixed:
        raise ValueError("Exporter repair differs from the sole reviewed seed-domain change")
    if sha(EVALUATOR / "prepared_files.json") != repair["evaluator_manifest_sha256"]:
        raise ValueError("Repaired evaluator manifest changed")
    return {"repair_receipt_sha256": sha(ROOT / "REPAIR_RECEIPT.json"),
            "prepared_manifest_sha256": sha(ROOT / "prepared_files.json"),
            "evaluator_manifest_sha256": sha(EVALUATOR / "prepared_files.json"),
            "checkpoint_commit_sha256": sha(CHECKPOINT / "COMMITTED.json")}


def worker(output):
    result = {"status": "blocked_or_failed", "optimizer_updates": 0,
              "simulator_steps": 0, "calibration_refreshes": 0,
              "source_checkpoint": str(CHECKPOINT),
              "scope": "Export recovery and inference parity only; no usefulness or efficacy result"}
    try:
        result["identity_before"] = verify()
        sys.path.insert(0, str(ORIGINAL))
        original = module("_export_recovery_original_runner", ORIGINAL / "run.py")
        sources = original.identities()
        prior = read(RUN / "report.json")
        if (prior["status"] != "training_complete_export_blocked"
                or prior["identity_before"] != sources or prior["identity_after"] != sources):
            raise ValueError("Recovery is bound to the preserved exact-source completed run")
        result["runtime"] = original.runtime()
        print(json.dumps({"event": "runtime_verified", **result["runtime"]}), flush=True)
        gate = original.verify_gate(GATE, sources)
        sys.path.insert(0, str(original.STAGE1))
        sys.path.insert(0, str(original.STAGE2))
        from native_loader import load_native
        from data import load_training
        import training
        import checkpoint_io
        import jax
        import jax.numpy as jnp
        import numpy as np
        from flax import serialization
        native = load_native(original.STAGE1 / "upstream", original.STAGE1 / "provenance/unifloral.json")
        config = read(ORIGINAL / "config.json")
        dataset, heldout, heldout_ids, data_info = load_training(native, original.DATA)
        # Recreate only the matching static template. No training.step or fit/refresh call.
        context, template = training.create(native, dataset, heldout, heldout_ids, config)
        bindings = {"config": config, "sources": sources,
            "datasetidentity": {"sha256": data_info["dataset_sha256"], "digest": data_info["digest"],
                                "split_receipt": data_info["split_receipt"]},
            "normalization": {"mean": np.asarray(context.obs_mean).tolist(), "std": np.asarray(context.obs_std).tolist()},
            "native_args": asdict(training.native_args(context)),
            "auxiliary_config": asdict(context.aux_config),
            "auxiliary_rng_schedule": context.auxiliary_rng_schedule,
            "gate_identity": gate, "training_stage_reviewed": True,
            "review_scope": "Explicit user launch of this declared bounded stage after a hash-bound passing correctness gate"}
        if checkpoint_io._json(bindings) != checkpoint_io._json(read(RUN / "bindings.json")):
            raise ValueError("Reconstructed bindings do not exactly match the completed run")
        state, restore = checkpoint_io.load(CHECKPOINT, template, bindings)
        training.validate(state)
        raw = (CHECKPOINT / "state.msgpack").read_bytes()
        metadata = read(CHECKPOINT / "metadata.json")
        raw_sha = hashlib.sha256(raw).hexdigest()
        if raw_sha != restore["state_sha256"] or raw_sha != metadata["state_sha256"]:
            raise ValueError("Committed checkpoint payload identity changed")
        state_dict = serialization.to_state_dict(state)
        content_before = content_identity(checkpoint_io._describe(state_dict), metadata["state_leaves"])
        # The immutable payload SHA and reencoded runtime SHA identify different
        # byte representations. Exact path/dtype/shape/value checks remain mandatory.
        reencoded = serialization.to_bytes(state)
        before = hashlib.sha256(reencoded).hexdigest()
        decoded = serialization.msgpack_restore(raw)
        content_identity(checkpoint_io._describe(decoded), metadata["state_leaves"])
        decoded_reencoded = serialization.msgpack_restore(reencoded)
        content_identity(checkpoint_io._describe(decoded_reencoded), metadata["state_leaves"])
        order_audit = mapping_order_audit(decoded, decoded_reencoded)
        encoding = {"raw_checkpoint_payload_sha256": raw_sha,
                    "runtime_encoding_sha256_before": before,
                    "wire_bytes_identical": raw == reencoded,
                    **content_before, **order_audit}
        counts_before = training.counts(state)
        result.update(restore=restore, counts_before=counts_before,
                      encoding_identity=encoding)
        write(output / "encoding_identity.json", encoding)
        write(output / "restore_receipt.json", restore)
        print(json.dumps({"event": "exact_checkpoint_content_verified", **content_before,
                          "wire_bytes_identical": encoding["wire_bytes_identical"],
                          "changed_mapping_order_count": order_audit["changed_mapping_order_count"]}), flush=True)
        fixed = module("_export_recovery_seedfix", ROOT / "export_seedfix.py")
        exported = fixed.export(context, state, data_info, bindings, output / "inference_export")
        if exported["full_training_state_sha256"] != before:
            raise ValueError("Export's recorded runtime state encoding differs")
        result["export"] = exported
        write(output / "export_receipt.json", exported)
        print(json.dumps({"event": "inference_export_written", "manifest": exported["manifest"]}), flush=True)
        inference = module("_export_recovery_consumer", EVALUATOR / "inference.py")
        policy, scale_bank, loaded = inference.load(exported["manifest"], read(EVALUATOR / "protocol.json"))
        agent = state.microstep.agent_state
        checks = []

        def exact_tree(label, left, right):
            a, ta = jax.tree_util.tree_flatten(serialization.to_state_dict(left))
            b, tb = jax.tree_util.tree_flatten(serialization.to_state_dict(right))
            if ta != tb or len(a) != len(b):
                raise ValueError("Exported tree differs: " + label)
            for x, y in zip(a, b):
                x, y = np.asarray(x), np.asarray(y)
                if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y):
                    raise ValueError("Exported numeric leaf differs: " + label)
            checks.append({"name": label, "status": "pass", "exact_leaves": len(a)})

        exact_tree("actor_parameters_exact", agent.actor.params, policy.actor_params)
        exact_tree("critic_parameters_exact", agent.dual_q.params, policy.critic_params)
        exact_tree("frozen_reference_exact", state.aux.posterior, policy.reference)
        for key in scale_bank:
            exact_tree("fitting_scale_bank_" + key, data_info["scale_bank"][key], scale_bank[key])
        # First 32 prespecified fitting-only rows; no oracle, test labels or coefficient selection.
        obs = jnp.asarray(scale_bank["observations"][:32])
        recorded_action = jnp.asarray(scale_bank["actions"][:32])
        native_actor = jax.jit(lambda x: agent.actor.apply_fn(agent.actor.params, x).sample(seed=None))
        native_q = jax.jit(lambda x, a: agent.dual_q.apply_fn(agent.dual_q.params, x, a))
        native_width = jax.jit(lambda x, a: context.pipeline.live_width(context.models, state.aux.posterior, x, a))
        action = native_actor(obs)

        def numeric(label, actual, wanted):
            a, b = np.asarray(actual), np.asarray(wanted)
            if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise ValueError("Inference output shapes/finite values differ: " + label)
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
            checks.append({"name": label, "status": "pass", "shape": list(a.shape),
                           "max_abs_error": float(np.max(np.abs(a-b))), "rtol": 2e-5, "atol": 2e-6})

        numeric("actor_output", policy.act(obs), action)
        for label, actions in (("recorded_actions", recorded_action), ("actor_actions", action)):
            numeric("critic_" + label, policy.q(obs, actions), native_q(obs, actions))
            actual, usable, valid = policy.width(obs, actions)
            expected, expected_usable, expected_valid = native_width(obs, actions)
            if not all(bool(x) for x in (usable, valid, expected_usable, expected_valid)):
                raise ValueError("Frozen width is invalid or unusable: " + label)
            numeric("width_" + label, actual, expected)
        policy.assert_unchanged()
        after = hashlib.sha256(serialization.to_bytes(state)).hexdigest()
        content_after = content_identity(checkpoint_io._describe(serialization.to_state_dict(state)), metadata["state_leaves"])
        if after != before or training.counts(state) != counts_before:
            raise ValueError("Recovery/inference changed training state or counters")
        if content_after != content_before or sha(CHECKPOINT / "state.msgpack") != raw_sha:
            raise ValueError("Exact state content or original checkpoint changed")
        result.update(status="pass", checks=checks, counts_after=training.counts(state),
                      content_identity_after=content_after, runtime_encoding_sha256_after=after,
                      training_state_unchanged=True, inference_frozen_state_sha256=loaded["frozen_state_sha256"],
                      fitting_probe_rows=32, frozen_diagnostic_available=True,
                      scientific_validation_status="incomplete_design_review_required")
        write(output / "inference_parity.json", {"status": "pass", "checks": checks,
              "optimizer_updates": 0, "simulator_steps": 0, "training_state_unchanged": True})
        write(output / "RECOVERY_LINK.json", {"status": "pass", "source_checkpoint": str(CHECKPOINT),
              "raw_checkpoint_payload_sha256": raw_sha,
              "canonical_state_leaves_sha256": content_before["canonical_state_leaves_sha256"],
              "export_full_training_state_sha256_is_runtime_encoding": before,
              "export_manifest_sha256": exported["manifest_sha256"], "export_manifest": exported["manifest"],
              "exact_content_match": True, "numeric_leaves": content_before["numeric_leaves"],
              "state_and_counters_unchanged": True, "optimizer_updates": 0, "simulator_steps": 0})
    except Exception as error:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    try:
        result["identity_after"] = verify()
        if result.get("identity_before") != result["identity_after"]:
            raise ValueError("Source/evidence identity changed during recovery")
    except Exception as error:
        result["status"] = "blocked_or_failed"
        result["postcheck_error"] = str(error)
    write(output / "worker.json", result)
    return 0 if result["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-outside-upstream-constraints", action="store_true")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.allow_outside_upstream_constraints:
        parser.error("Explicit provisional-runtime acknowledgement required")
    if args.worker_output:
        out = args.worker_output.resolve()
        if out.parent != ROOT or not out.is_dir():
            parser.error("Worker output must be a new local recovery directory")
        return worker(out)
    identities = verify()
    if os.environ.get("XLA_FLAGS", "") not in ("", XLA):
        parser.error("Conflicting XLA_FLAGS; no override attempted")
    out = ROOT / ("recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(exist_ok=False)
    env = dict(os.environ, JAX_PLATFORMS="cuda", JAX_DEFAULT_MATMUL_PRECISION="highest", XLA_FLAGS=XLA,
               XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--allow-outside-upstream-constraints", "--worker-output", str(out)]
    print("Starting zero-update export recovery; output: " + str(out), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    timed_out = False
    try:
        code = process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()  # Only this bounded zero-update recovery child.
        code = process.wait()
    report = read(out / "worker.json") if (out / "worker.json").exists() else {
        "status": "worker_did_not_complete", "identity_before": identities}
    report.update(exit_code=code, elapsed_seconds=time.monotonic()-started, timed_out=timed_out)
    if code != 0 or timed_out:
        report["status"] = "blocked_or_failed"
    write(out / "report.json", report)
    if report["status"] == "pass":
        # Exact command is saved only after the export has loaded and passed parity.
        manifest = report["export"]["manifest"]
        script = str(EVALUATOR / "run_diagnostic.py")
        next_command = 'wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "' + script + '" --checkpoint-manifest "' + manifest + '" --execute --allow-outside-upstream-constraints'
        (out / "NEXT_FROZEN_DIAGNOSTIC_COMMAND.txt").write_text(next_command + "\n", encoding="utf-8")
        print("Frozen diagnostic command (not executed): " + next_command, flush=True)
    print(json.dumps({"status": report["status"], "report": str(out / "report.json"),
                      "optimizer_updates": 0, "simulator_steps": 0}), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
