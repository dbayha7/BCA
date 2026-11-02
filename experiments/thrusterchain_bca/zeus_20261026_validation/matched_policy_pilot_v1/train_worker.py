"""One bounded, separately reserved continuation; no computation on import.

Only the supervisor calls run. This module never starts a simulator or selects
a checkpoint by outcomes. Failed states are evidence, not resumable jobs.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE.parents[1] / "titan_20261026/payload"
COUNT_KEYS = ("critic", "actor", "actor_target", "critic_target", "outer_steps",
              "auxiliary", "aux_fit_attempts", "refresh_attempts", "refresh_accepted")
EXPECTED_UPDATES = dict(zip(COUNT_KEYS, (2000, 1000, 1000, 1000, 1000, 2000, 2000, 1, 1)))
FIXED = {"schema": "unifloral.matched_policy_pilot.v1", "critic_updates": 2000,
    "actor_updates": 1000, "auxiliary_updates": 2000, "refresh_at": 1000,
    "absolute_critic_start": 20000, "absolute_critic_end": 22000,
    "absolute_refresh": 21000, "worker_timeout_seconds": 600, "progress_every": 20,
    "batch_size": 256, "gamma": .98, "horizon": 100,
    "actor_distance_factor": 19.29619572569095, "actor_distance_coefficient": .1,
    "bca_kappa": 3.0, "alpha": .1, "credibility": .95, "posterior_draws": 1000,
    "local_width_penalty": .005, "local_cal_beta": 20.0, "aux_learning_rate": .001,
    "scale_ema": .99, "full_training_hold": True, "automatic_restart": False,
    "failed_one_sided_variant_excluded": True,
    "checkpoint_state_sha256": "874d65b7f4dcb5ec285a0d4360439c73a787b5e8c567ff7c0138e0c1e22948e8",
    "native_source_sha256": "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee",
    "native_revision": "f2dc1278eae18ed3e22c92255119c54885414d17",
    "dataset_sha256": "4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe(value):
    """Retain nonfinite failure diagnostics without invalid JSON numbers."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")
    return value


def atomic_bytes(path, raw):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "sha256": sha(raw), "bytes": len(raw)}


def atomic_json(path, value):
    return atomic_bytes(path, (json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        result = sys.modules[name]
        require(Path(result.__file__).resolve() == path, "Loaded module path collision")
        return result
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def validate_protocol(protocol, arm, stream_index):
    for key, expected in FIXED.items():
        require(type(protocol.get(key)) is type(expected) and protocol[key] == expected,
                "Fixed worker protocol differs: " + key)
    require([(a["name"], a["kappa"]) for a in protocol["arms"]] ==
            [("native", 0.0), ("bca", 3.0), ("distance", .1)], "Fixed arms differ")
    require(protocol["streams"] == [
        {"index": 0, "native_seed": 2026102681, "aux_seed": 2026102683},
        {"index": 1, "native_seed": 2026102682, "aux_seed": 2026102684}], "Paired stream IDs differ")
    require(arm in ("native", "bca", "distance") and type(stream_index) is int
            and stream_index in (0, 1), "Unknown arm or stream")
    return protocol["streams"][stream_index]


def parameter_leaves(value):
    """Portable content identity over Flax state-dict leaves, including dtype."""
    import numpy as np
    from flax import serialization
    result = []
    def visit(node, path):
        if isinstance(node, dict):
            for key in sorted(node):
                visit(node[key], path + [str(key)])
        elif isinstance(node, (tuple, list)):
            for index, item in enumerate(node):
                visit(item, path + [str(index)])
        else:
            array = np.asarray(node)
            result.append({"path": "/".join(path), "shape": list(array.shape),
                           "dtype": array.dtype.str, "sha256": sha(array.tobytes(order="C"))})
    visit(serialization.to_state_dict(value), [])
    return result


def deltas(initial, final):
    return {key: final[key] - initial[key] for key in COUNT_KEYS}


def reference_summary(reference):
    import numpy as np
    return {key: np.asarray(getattr(reference, key)).item() for key in
            ("threshold", "lambda_hat", "lambda_hpd", "residual_scale", "n_eff", "ready")}


def normalization(context, state):
    import numpy as np
    mean, std = np.asarray(context.obs_mean), np.asarray(context.obs_std)
    require(mean.shape == std.shape == (14,), "Normalization shape differs")
    agent = state.microstep.agent_state
    for train_state in (agent.actor, agent.actor_target, agent.dual_q, agent.dual_q_target):
        model = train_state.apply_fn.__self__
        require(np.array_equal(np.asarray(model.obs_mean), mean)
                and np.array_equal(np.asarray(model.obs_std), std), "Native normalization changed")
    return {"mean": mean.tolist(), "std": std.tolist(), "epsilon": .001,
            "source": "unchanged original native training statistics"}


def run(protocol, output_dir, arm, stream_index):
    """Execute exactly one fixed arm/stream. Supervisor owns process reservation."""
    started = time.monotonic()
    output = Path(output_dir).resolve()
    require(output.is_dir() and not Path(output_dir).is_symlink(), "Output directory must be reserved and real")
    artifacts = ("progress.json", "report.json", "report.sha256", "actor_manifest.json",
                 "final_actor.msgpack", "final_state.msgpack", "partial_state.msgpack",
                 "state_manifest.json", "restore_receipt.json", "refresh_receipt.json")
    require(not any((output / name).exists() for name in artifacts), "Existing worker evidence; refusing duplicate attempt")
    stage, state, adapter, restored = "protocol_validation", None, None, None
    initial, last_metrics, attempted, in_flight = None, None, {}, False
    progress = {"schema": "unifloral.matched_policy.progress.v1", "status": "running",
                "run_id": f"seed{stream_index}_{arm}", "arm": arm, "stream_index": stream_index}
    ledger = {"committed_additional_updates": {key: 0 for key in COUNT_KEYS},
              "attempted_uncommitted_updates": {}, "simulator_transitions": 0,
              "evaluation_episodes": 0, "checkpoint_selection_reads": 0,
              "forced_kill_counts": "Last progress is a committed lower bound; up to 20 further critic/auxiliary dispatches may be unrecorded."}

    def deadline():
        if time.monotonic() - started >= 600:
            raise TimeoutError("Fixed 600 second worker deadline reached")

    def journal(status="running"):
        atomic_json(output / "progress.json", {**progress, "status": status, "stage": stage,
            "ledger": ledger, "elapsed_seconds": time.monotonic() - started,
            "updated_utc": datetime.now(timezone.utc).isoformat()})

    try:
        stream = validate_protocol(protocol, arm, stream_index)
        protocol_raw = (HERE / "protocol.json").read_bytes()
        require(json.loads(protocol_raw) == protocol, "Worker protocol differs from sealed on-disk protocol")
        protocol_sha = sha(protocol_raw)
        progress["protocol_sha256"] = protocol_sha
        journal()
        continuation = module("_matched_continuation", HERE / "continuation.py")
        restore = module("_matched_training_restore", HERE.parent / "bca_one_sided_v1/restore_logged.py")
        stage = "exact_original_checkpoint_restore"
        journal()
        deadline()
        original_protocol = read(PAYLOAD / "unifloral_operator_alignment_v1/protocol.json")
        restored = restore.load(original_protocol)
        from flax import serialization
        import jax
        require(jax.default_backend() == "gpu", "Fixed pilot requires the verified GPU runtime")
        atomic_json(output / "restore_receipt.json", restored.receipt)
        deadline()
        stage = "paired_continuation_initialization"
        adapter, state = continuation.initialize_continuation(restored.context, restored.state, arm,
            stream["native_seed"], stream["aux_seed"], critic_updates=2000, refresh_at=1000)
        initial = adapter.initial_counts
        require(initial["critic"] == 20000 and initial["actor"] == 10000
                and initial["auxiliary"] == 20000 and initial["refresh_accepted"] == 3,
                "Not the declared original complete checkpoint")
        norm = normalization(restored.context, state)
        original_norm = normalization(restored.context, restored.state)
        require(norm == original_norm, "Fork changed normalization")
        original_before = restored.assert_unchanged()
        initial_reference = reference_summary(state.aux.posterior)
        journal()
        for index in range(2000):
            deadline()
            stage = "native_and_shadow_auxiliary_updates"
            in_flight = True
            state, last_metrics = continuation.step(adapter, state)
            in_flight = False
            ledger["committed_additional_updates"] = deltas(initial, last_metrics["counts"])
            if last_metrics["refresh_due"]:
                atomic_json(output / "refresh_receipt.json", {
                    "relative_critic": index + 1, "absolute_critic": last_metrics["counts"]["critic"],
                    "ordering": "Boundary actor and auxiliary fit completed using previous reference; refreshed reference applies to next actor.",
                    "initial_reference": initial_reference, "new_reference": reference_summary(state.aux.posterior),
                    "details": last_metrics["refresh"], "counts": last_metrics["counts"]})
            if (index + 1) % 20 == 0:
                journal()
        deadline()
        stage = "final_integrity_and_artifact_seal"
        final_validation = continuation.validate(adapter, state)
        final = final_validation["counts"]
        require(deltas(initial, final) == EXPECTED_UPDATES and final["phase"] == 0
                and final["critic"] == 22000 and final["last_refresh_critic"] == 21000,
                "Final matched update budget differs")
        require(normalization(restored.context, state) == norm, "Final native normalization changed")
        require(sha((HERE / "protocol.json").read_bytes()) == protocol_sha, "Protocol changed during execution")
        original_after = restored.assert_unchanged()
        require(original_before == original_after, "Original frozen checkpoint identity changed")
        actor_params = state.microstep.agent_state.actor.params
        actor_artifact = atomic_bytes(output / "final_actor.msgpack", serialization.to_bytes(actor_params))
        state_artifact = atomic_bytes(output / "final_state.msgpack", serialization.to_bytes(state))
        manifest = {"schema": "unifloral.matched_policy.actor.v1", "run_id": progress["run_id"],
            "arm": arm, "stream_index": stream_index, "protocol_sha256": protocol_sha,
            "native_revision": FIXED["native_revision"], "native_source_sha256": FIXED["native_source_sha256"],
            "initial_checkpoint_state_sha256": FIXED["checkpoint_state_sha256"],
            "actor_params": actor_artifact, "normalization": norm,
            "architecture": {"class": "DeterministicTanhActor", "source": "pinned standalone UniFloral TD3+BC",
                             "action_dim": 4, "observation_dim": 14},
            "parameter_leaves": parameter_leaves(actor_params), "initial_absolute_counts": initial,
            "final_absolute_counts": final, "additional_updates": deltas(initial, final),
            "phase": final["phase"], "completed": True, "final_state": state_artifact}
        actor_manifest_artifact = atomic_json(output / "actor_manifest.json", manifest)
        state_manifest_artifact = atomic_json(output / "state_manifest.json", {"schema": "unifloral.matched_policy.state.v1",
            "artifact": state_artifact, "protocol_sha256": protocol_sha, "run_id": progress["run_id"],
            "state_leaves": parameter_leaves(state), "counts": final, "rng_fork": adapter.rng_receipt,
            "refresh_schedule": adapter.schedule, "native_and_aux_optimizer_states_preserved_at_fork": True,
            "source_checkpoint_unchanged": original_after,
            "resume_authorized": False})
        deadline()
        stage = "completed"
        report = {"schema": "unifloral.matched_policy.training_result.v1",
            "status": "completed_bounded_policy_training", "run_id": progress["run_id"],
            "arm": arm, "stream_index": stream_index, "protocol_sha256": protocol_sha,
            "execution_counts_known": True, "ledger": ledger, "initial_absolute_counts": initial,
            "final_absolute_counts": final, "additional_updates": deltas(initial, final),
            "rng_fork": adapter.rng_receipt, "initial_reference": initial_reference,
            "final_reference": reference_summary(state.aux.posterior), "checks": final_validation["checks"],
            "original_checkpoint_unchanged": original_after, "actor_manifest": "actor_manifest.json",
            "actor_manifest_sha256": actor_manifest_artifact["sha256"],
            "state_manifest_sha256": state_manifest_artifact["sha256"],
            "last_update_metrics": last_metrics, "elapsed_seconds": time.monotonic() - started,
            "data_scope": "Original 768 logged training episodes and original 128 calibration episodes only.",
            "limitations": ["Exploratory continuation from one checkpoint; two RNG streams are not independent initializations.",
                           "Calibration episodes reused; no selected-action or OOD coverage claim.",
                           "No simulator evaluation or policy advantage conclusion in this worker."],
            "error": None}
        raw = (json.dumps(safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        atomic_bytes(output / "report.json", raw)
        atomic_bytes(output / "report.sha256", (sha(raw) + "\n").encode())
        journal("completed_bounded_policy_training")
        return report
    except Exception as exc:
        committed_known = not in_flight
        if hasattr(exc, "last_committed_state") and hasattr(exc, "attempted_counts"):
            state = exc.last_committed_state
            attempted = dict(exc.attempted_counts)
            stage = exc.stage
            committed_known = True
        ledger["attempted_uncommitted_updates"] = attempted
        if state is not None and adapter is not None:
            try:
                ledger["committed_additional_updates"] = deltas(initial, adapter.training.counts(state))
                from flax import serialization
                atomic_bytes(output / "partial_state.msgpack", serialization.to_bytes(state))
            except Exception as save_error:
                ledger["partial_state_save_error"] = str(save_error)
        # Explicit attempts are known, but a failed dispatch does not prove the
        # number of GPU kernels completed. Preserve both without guessing.
        execution_known = committed_known and not in_flight and not any(attempted.values())
        failure = {"schema": "unifloral.matched_policy.training_result.v1", "status": "failed_bounded_policy_training",
            "run_id": progress["run_id"], "arm": arm, "stream_index": stream_index,
            "stage": stage, "execution_counts_known": execution_known,
            "committed_counts_known": committed_known, "ledger": ledger,
            "error": type(exc).__name__ + ": " + str(exc), "elapsed_seconds": time.monotonic() - started,
            "diagnostics": getattr(exc, "diagnostics", None), "automatic_restart": False}
        if restored is not None:
            try:
                failure["original_checkpoint_unchanged"] = restored.assert_unchanged()
            except Exception as identity_error:
                failure["original_identity_check_error"] = str(identity_error)
        atomic_json(output / "report.json", failure)
        journal("failed_bounded_policy_training")
        return failure
