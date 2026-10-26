"""Numeric evaluator export from the real, terminal 20k checkpoint only.

No checkpoint conversion, weight initialization, optimizer or simulator call.
Reviewed-stage and gate fields record an explicitly authorized stage and matching
receipts. Self-consistent hashes are not independent attestation of their truth.
"""
import io
from pathlib import Path

from checkpoint_io import CheckpointError, _bindings, _json, _normalization, _sha, actual_counts, publish_directory


REVISION = "f2dc1278eae18ed3e22c92255119c54885414d17"
NATIVE_SHA = "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee"
DATA_SHA = "4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226"
DATA_DIGEST = "6b58db68263dff0d7e9b8d6dfa7c1ca5d4127ed9ee6a3b9069add1c565e88331"
PRODUCTION = {"mode": "production", "gamma": 0.98, "batch_size": 256,
    "seed": 2026102302, "aux_seed": 2026102361,
    "refresh_at": [10000, 15000, 20000], "critic_budget": 20000, "kappa_penalty": 0.0}


def _digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def export(context, state, data_info, bindings, path):
    """Create a separate NEW export directory and return its integrity receipt."""
    import numpy as np
    import jax
    from flax import serialization
    from training import validate

    if Path(path).exists() or Path(path).is_symlink():
        raise FileExistsError("Export destination already exists: " + str(path))
    bindings = _bindings(bindings)
    if _json(context.config) != _json(PRODUCTION) or _json(bindings["config"]) != _json(PRODUCTION):
        raise CheckpointError("Only the fixed production 20k native checkpoint can be exported")
    if bindings.get("training_stage_reviewed") is not True:
        raise CheckpointError("Explicit reviewed-stage launch binding is required; not inferred from completion")
    gate = bindings.get("gate_identity", {})
    if (not isinstance(gate, dict) or set(gate) != {"report_sha256", "checks_sha256"}
            or not all(_digest_string(value) for value in gate.values())):
        raise CheckpointError("Export must bind the exact passing gate report and checks")
    source = getattr(context.native, "_source_receipt", {})
    if (source.get("passed") is not True or source.get("revision") != REVISION
            or source.get("source_sha256") != NATIVE_SHA):
        raise CheckpointError("Native source identity is not the verified UniFloral source")
    if (data_info.get("synthetic_fixture") is not False or data_info.get("dataset_sha256") != DATA_SHA
            or data_info.get("digest") != DATA_DIGEST):
        raise CheckpointError("Synthetic or different dataset cannot yield a trained export")
    expected_data = {"sha256": DATA_SHA, "digest": DATA_DIGEST, "split_receipt": data_info["split_receipt"]}
    if _json(bindings["datasetidentity"]) != _json(expected_data):
        raise CheckpointError("Dataset and split differ from checkpoint bindings")
    if float(context.args.gamma) != 0.98 or int(context.args.batch_size) != 256:
        raise CheckpointError("Native task arguments differ from the declared training stage")
    validate(state)
    counts = actual_counts(state)
    if (counts["critic"] != 20000 or counts["actor"] != 10000 or counts["phase"] != 0
            or counts["critic_budget"] != 20000 or counts["auxiliary"] <= 0
            or counts["refresh_accepted"] != 3 or counts["refresh_attempts"] != 3
            or counts["next_refresh_index"] != 3 or counts["last_refresh_critic"] != 20000
            or not counts["reference_ready"]
            or np.asarray(state.refresh_at).tolist() != PRODUCTION["refresh_at"]):
        raise CheckpointError("Export requires the complete 20k boundary and all three accepted refreshes")
    _normalization(state, bindings)
    reference = state.aux.posterior
    if not bool(context.pipeline.reference_valid(reference)) or not bool(reference.ready):
        raise CheckpointError("Frozen calibration reference is invalid or unready")
    reference_scalars = {}
    for key in ("residual_scale", "threshold", "lambda_hat", "lambda_hpd", "n_eff"):
        value = np.asarray(getattr(reference, key))
        if value.shape != () or value.dtype != np.dtype("float32") or not np.isfinite(value):
            raise CheckpointError("Frozen reference must contain finite float32 scalars: " + key)
        reference_scalars[key] = float(value)
    if reference_scalars["n_eff"] <= 0:
        raise CheckpointError("Frozen reference has no positive effective calibration sample size")
    before = _sha(serialization.to_bytes(state))
    mean, std = np.asarray(context.obs_mean), np.asarray(context.obs_std)
    for name, value in (("mean", mean), ("std", std)):
        if (value.shape != (14,) or value.dtype != np.dtype("float32") or not np.all(np.isfinite(value))
                or not np.array_equal(value.astype(np.float64), np.asarray(bindings["normalization"][name], np.float64))):
            raise CheckpointError("Context normalization differs from actual checkpoint normalization")
    native = state.microstep.agent_state
    payload = {"actor_params": native.actor.params, "critic_params": native.dual_q.params,
               "obs_mean": mean, "obs_std": std, "reference": reference._asdict()}
    for name, variables in (("actor", native.actor.params), ("critic", native.dual_q.params),
                            ("calibrator", reference.cal_params)):
        if set(variables) != {"params"}:
            raise CheckpointError("Complete native Flax variables required: " + name)
        leaves = jax.tree_util.tree_leaves(variables)
        if not leaves or any(np.asarray(leaf).dtype != np.dtype("float32")
                             or not np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
            raise CheckpointError("Nonfinite or non-float32 exported parameter: " + name)
    parameters = serialization.msgpack_serialize(
        jax.tree_util.tree_map(np.asarray, serialization.to_state_dict(payload)))
    train_groups, calibration_groups = data_info["train_groups"], data_info["calibration_groups"]
    for groups in (train_groups, calibration_groups):
        if (not isinstance(groups, list) or not groups or len(groups) != len(set(groups))
                or any(type(value) is not int or value < 0 for value in groups)):
            raise CheckpointError("Training/calibration group provenance is invalid")
    if len(calibration_groups) != 128 or set(train_groups) & set(calibration_groups):
        raise CheckpointError("The 128 calibration groups must be disjoint from training")
    used_seeds = data_info["used_episode_seeds"]
    if (not isinstance(used_seeds, list) or len(used_seeds) != len(train_groups) + len(calibration_groups)
            or len(set(used_seeds)) != len(used_seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in used_seeds)
            or set(used_seeds) & (set(range(2026102400, 2026102464)) | set(range(2026102500, 2026102564)))):
        raise CheckpointError("Full episode seed provenance is missing, duplicated or overlaps evaluator reservations")
    bank = data_info["scale_bank"]
    if not isinstance(bank, dict) or set(bank) != {"observations", "actions", "group_ids"}:
        raise CheckpointError("Training-only scale bank has unexpected fields")
    bank = {key: np.asarray(value) for key, value in bank.items()}
    if (bank["observations"].shape != (256, 14) or bank["actions"].shape != (256, 4)
            or bank["group_ids"].shape != (256,)
            or any(bank[key].dtype != np.dtype("float32") for key in ("observations", "actions"))
            or not np.issubdtype(bank["group_ids"].dtype, np.integer)
            or not all(np.all(np.isfinite(value)) for value in bank.values())
            or np.any(np.abs(bank["actions"]) > 1)
            or not set(bank["group_ids"].tolist()).issubset(set(train_groups))):
        raise CheckpointError("Scale bank is not 256 finite training-only observed state/action rows")
    stream = io.BytesIO()
    np.savez_compressed(stream, **bank)
    scale_bytes = stream.getvalue()
    identity = {"host": "unifloral_standalone_td3_bc", "native_revision": REVISION,
        "native_source_sha256": NATIVE_SHA, "task": "ThrusterChain", "gamma": 0.98,
        "horizon": 100, "task_semantics": "finite_horizon", "observation_mode": "time_aware_14d",
        "terminal_bootstrap": False, "reward_scale": 1.0,
        "physics": {"stall": True, "stall_slope": 1.5}, "synthetic_fixture": False,
        "matmul_precision": "highest"}
    selection = {"optimizer_counts": {key: counts[key] for key in ("critic", "actor", "auxiliary")},
        "policy_kind": "frozen_deterministic_current_actor",
        "selection_rule": "prespecified_terminal_checkpoint_no_diagnostic_return_selection",
        "training_stage_reviewed": bindings["training_stage_reviewed"],
        "train_group_ids": train_groups, "calibration_group_ids": calibration_groups,
        "training_and_calibration_episode_seeds": used_seeds, "finite_ready_refreshes": counts["refresh_accepted"]}
    common = {**identity, "parameters_sha256": _sha(parameters)}
    training_receipt = {**common, **selection, "schema": "unifloral.thrusterchain.training_receipt.v1",
        "scale_bank_sha256": _sha(scale_bytes), "full_training_state_sha256": before,
        "all_counters": counts, "bindings": bindings, "bindings_sha256": _sha(_json(bindings)),
        "review_scope": "Explicit user launch of the declared stage after the hash-bound passing gate; consistency records are not independent truth attestation",
        "actor_training_penalty": "Native UniFloral TD3+BC, kappa=0; auxiliary calibration does not establish BCA actor-training benefit"}
    normalization_receipt = {**common, "schema": "unifloral.thrusterchain.normalization_receipt.v1",
        "dtype": "float32", "observation_dimensions": 14, "rule": "raw_native_mean_std_plus_0.001",
        "training_group_ids": train_groups, "obs_mean": mean.tolist(), "obs_std": std.tolist()}
    calibration_receipt = {**common, "schema": "unifloral.thrusterchain.calibration_receipt.v1",
        "training_group_ids": train_groups, "calibration_group_ids": calibration_groups,
        "finite_ready_refreshes": counts["refresh_accepted"], "reference_ready": True,
        "reference_dtype": "float32", **reference_scalars,
        "last_refresh_critic": counts["last_refresh_critic"], "split_receipt": data_info["split_receipt"],
        "interpretation": "Frozen one-step twin-min TD-residual proxy; not a Q1-error coverage guarantee"}
    files = {"parameters.msgpack": parameters, "scale_bank.npz": scale_bytes,
        "training_receipt.json": _json(training_receipt),
        "normalization_receipt.json": _json(normalization_receipt),
        "calibration_receipt.json": _json(calibration_receipt)}
    roles = {"parameters": "parameters.msgpack", "scale_bank": "scale_bank.npz",
        "training_receipt": "training_receipt.json", "normalization_receipt": "normalization_receipt.json",
        "calibration_receipt": "calibration_receipt.json"}
    manifest = {"schema": "unifloral.thrusterchain.inference_export.v1", **identity, **selection,
        "files": {role: {"path": name, "sha256": _sha(files[name])} for role, name in roles.items()},
        "dataset_sha256": DATA_SHA, "dataset_digest": DATA_DIGEST,
        "full_training_state_sha256": before, "training_checkpoint_budget": 20000,
        "scope": "Frozen native-policy diagnostic export only; no million-update or BCA-training-efficacy claim"}
    files["eligible_checkpoint.json"] = _json(manifest)  # Publish this after all required components.
    if _sha(serialization.to_bytes(state)) != before:
        raise CheckpointError("Training state changed during export")
    receipt = publish_directory(path, files, "inference_export")
    return {**receipt, "manifest": str(Path(path).resolve() / "eligible_checkpoint.json"),
        "manifest_sha256": _sha(files["eligible_checkpoint.json"]), "full_training_state_sha256": before,
        "counts": counts, "training_state_unchanged": True,
        "runtime_compatibility": "Inference consumer must still validate and load the exported bytes"}
