"""Inference-only loader for a future authenticated UniFloral export.

No simulator, optimizer, initialization fallback, or checkpoint conversion lives
here. A synthetic/parity/CORL artifact is rejected before scientific imports.
The export format is a proposed contract; no compatible trained export exists.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT.parent / "unifloral_td3bc_validation_v1"
STAGE2 = ROOT.parent / "unifloral_bca_pipeline_validation_v1"
REVISION = "f2dc1278eae18ed3e22c92255119c54885414d17"
NATIVE_SHA = "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee"


class CheckpointNotReady(ValueError):
    pass


def _json_bytes(raw, label):
    def reject_constant(value):
        raise CheckpointNotReady(label + " contains nonfinite JSON: " + value)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointNotReady(label + " contains duplicate JSON key: " + key)
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise CheckpointNotReady(label + " contains an unrepresentable finite JSON number")
        return result

    result = json.loads(raw.decode("utf-8"), parse_constant=reject_constant,
                        parse_float=finite_float, object_pairs_hook=unique_object)
    if not isinstance(result, dict):
        raise CheckpointNotReady(label + " must be a JSON object")
    return result


def _same_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _require_fields(actual, expected, label):
    for key, value in expected.items():
        if key not in actual or not _same_json(actual[key], value):
            raise CheckpointNotReady(f"{label}: incompatible or missing {key}; expected {value!r}")


def _groups(value, label, minimum=1):
    if (not isinstance(value, list) or len(value) < minimum
            or any(type(g) is not int or g < 0 for g in value)
            or len(set(value)) != len(value)):
        raise CheckpointNotReady(label + " must contain distinct nonnegative integer group IDs")
    return value


def read_contract(path, protocol):
    """Stdlib-only consistency gate; hashes and receipts cannot attest truth.

    The second return holds the exact verified bytes, not a promise about a path
    that may change before a separate process reads it. Each process validates
    its own consumed bytes. No training/export producer is implemented here.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise CheckpointNotReady("No eligible trained UniFloral inference export supplied: " + str(path))
    manifest_raw = path.read_bytes()
    manifest = _json_bytes(manifest_raw, "manifest")
    expected = {"schema": "unifloral.thrusterchain.inference_export.v1",
                "host": "unifloral_standalone_td3_bc", "native_revision": REVISION,
                "native_source_sha256": NATIVE_SHA, "task": "ThrusterChain",
                "gamma": 0.98, "horizon": 100, "observation_mode": "time_aware_14d",
                 "task_semantics": "finite_horizon",
                "terminal_bootstrap": False, "reward_scale": 1.0,
                "policy_kind": "frozen_deterministic_current_actor", "synthetic_fixture": False,
                "selection_rule": "prespecified_terminal_checkpoint_no_diagnostic_return_selection",
                "matmul_precision": "highest"}
    _require_fields(manifest, expected, "manifest")
    if not _same_json(manifest.get("physics"), protocol["physics"]):
        raise CheckpointNotReady("Checkpoint physics differs from declared evaluator")
    counts = manifest.get("optimizer_counts", {})
    if not isinstance(counts, dict) or set(counts) != {"critic", "actor", "auxiliary"}:
        raise CheckpointNotReady("Exact critic/actor/auxiliary optimizer counts are required")
    for key in ("critic", "actor", "auxiliary"):
        if type(counts.get(key)) is not int or counts[key] <= 0:
            raise CheckpointNotReady("Real trained optimizer counts required for " + key)
    if counts["critic"] != 2 * counts["actor"]:
        raise CheckpointNotReady("Checkpoint is not a complete native two-critic outer-step boundary")
    if manifest.get("training_stage_reviewed") is not True:
        raise CheckpointNotReady("The separately scoped training stage has not been reviewed")
    refreshes = manifest.get("finite_ready_refreshes")
    if type(refreshes) is not int or refreshes < 1:
        raise CheckpointNotReady("No real finite-ready calibration refresh is recorded")
    train_groups = _groups(manifest.get("train_group_ids"), "train_group_ids")
    calibration_groups = _groups(manifest.get("calibration_group_ids"), "calibration_group_ids", 128)
    if set(train_groups) & set(calibration_groups):
        raise CheckpointNotReady("Calibration groups overlap training")
    used_seeds = manifest.get("training_and_calibration_episode_seeds")
    if (not isinstance(used_seeds, list) or not used_seeds
            or any(type(s) is not int or not 0 <= s < 2**63 for s in used_seeds)
            or len(set(used_seeds)) != len(used_seeds)):
        raise CheckpointNotReady("Full used episode seed provenance is required")
    reserved = set(range(protocol["query_seed"], protocol["query_seed"] + protocol["query_count"]))
    reserved.update(range(protocol["confirmation_seed_start_reserved_not_generated"],
                          protocol["confirmation_seed_start_reserved_not_generated"] + protocol["confirmation_count_reserved_not_generated"]))
    if reserved & set(used_seeds):
        raise CheckpointNotReady("Training/calibration seeds overlap reserved evaluation starts")
    if not isinstance(manifest.get("files"), dict):
        raise CheckpointNotReady("Checkpoint files must be a role-to-component object")
    components = {}
    for role in ("parameters", "scale_bank", "training_receipt", "normalization_receipt", "calibration_receipt"):
        item = manifest.get("files", {}).get(role, {})
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
            raise CheckpointNotReady("Missing checkpoint component declaration: " + role)
        if Path(item["path"]).is_absolute():
            raise CheckpointNotReady("Checkpoint component path must be relative: " + role)
        target = (path.parent / item.get("path", "")).resolve()
        if not target.is_relative_to(path.parent) or not target.is_file():
            raise CheckpointNotReady("Missing or unsafe checkpoint component: " + role)
        raw = target.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != item.get("sha256"):
            raise CheckpointNotReady("Checkpoint component hash mismatch: " + role)
        components[role] = {"path": target, "bytes": raw, "sha256": actual}
        if role.endswith("_receipt"):
            components[role]["json"] = _json_bytes(raw, role)

    common = {key: manifest[key] for key in (
        "host", "native_revision", "native_source_sha256", "task", "gamma", "horizon",
        "task_semantics", "observation_mode", "terminal_bootstrap", "reward_scale",
        "physics", "synthetic_fixture", "matmul_precision")}
    common["parameters_sha256"] = components["parameters"]["sha256"]
    training = components["training_receipt"]["json"]
    _require_fields(training, {**common, "schema": "unifloral.thrusterchain.training_receipt.v1",
        **{key: manifest[key] for key in ("optimizer_counts", "policy_kind", "selection_rule",
            "training_stage_reviewed", "train_group_ids", "calibration_group_ids",
            "training_and_calibration_episode_seeds", "finite_ready_refreshes")},
        "scale_bank_sha256": components["scale_bank"]["sha256"]}, "training_receipt")
    normalization = components["normalization_receipt"]["json"]
    _require_fields(normalization, {**common,
        "schema": "unifloral.thrusterchain.normalization_receipt.v1",
        "dtype": "float32", "observation_dimensions": 14,
        "rule": "raw_native_mean_std_plus_0.001",
        "training_group_ids": train_groups}, "normalization_receipt")
    for key in ("obs_mean", "obs_std"):
        value = normalization.get(key)
        if (not isinstance(value, list) or len(value) != 14
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
            raise CheckpointNotReady("normalization_receipt requires 14 finite values: " + key)
    calibration = components["calibration_receipt"]["json"]
    _require_fields(calibration, {**common,
        "schema": "unifloral.thrusterchain.calibration_receipt.v1",
        "training_group_ids": train_groups, "calibration_group_ids": calibration_groups,
        "finite_ready_refreshes": refreshes, "reference_ready": True,
        "reference_dtype": "float32"}, "calibration_receipt")
    for key in ("residual_scale", "threshold", "lambda_hat", "lambda_hpd", "n_eff"):
        value = calibration.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise CheckpointNotReady("calibration_receipt requires a finite nonnegative " + key)
    if calibration["residual_scale"] <= 0 or calibration["n_eff"] <= 0:
        raise CheckpointNotReady("calibration_receipt requires positive residual_scale and n_eff")
    if calibration["threshold"] != max(calibration["lambda_hat"], calibration["lambda_hpd"]):
        raise CheckpointNotReady("calibration_receipt threshold differs from stored lambda maximum")
    return manifest, components, {"manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "file_hashes": {key: value["sha256"] for key, value in components.items()},
        "validation_scope": "Exact consumed bytes and cross-receipt consistency only; not independent attestation of training, dataset provenance or declaration truth"}


def load(path, protocol):
    manifest, components, receipt = read_contract(path, protocol)
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import serialization
    sys.path.insert(0, str(STAGE1))
    from native_loader import load_native
    sys.path.insert(0, str(STAGE2))
    from bca_bridge import load_pipeline
    native = load_native(STAGE1 / "upstream", STAGE1 / "provenance/unifloral.json")
    bca = load_pipeline(STAGE2)
    # msgpack is numeric structured data, not pickle or executable checkpoint code.
    payload = serialization.msgpack_restore(components["parameters"]["bytes"])
    required = {"actor_params", "critic_params", "obs_mean", "obs_std", "reference"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise CheckpointNotReady("Inference payload does not match the exact numeric schema")
    mean, std = np.asarray(payload["obs_mean"]), np.asarray(payload["obs_std"])
    if (mean.dtype != np.dtype("float32") or std.dtype != np.dtype("float32")
            or mean.shape != (14,) or std.shape != (14,)
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0)):
        raise CheckpointNotReady("Invalid native observation normalization")
    normalization = components["normalization_receipt"]["json"]
    for key, value in (("obs_mean", mean), ("obs_std", std)):
        if not np.array_equal(value.astype(np.float64), np.asarray(normalization[key], np.float64)):
            raise CheckpointNotReady("Normalization payload differs from receipt: " + key)
    actor = native.DeterministicTanhActor(4, jnp.asarray(mean), jnp.asarray(std))
    critic = native.DualQNetwork(jnp.asarray(mean), jnp.asarray(std))
    calibrator = bca.Calibrator(jnp.asarray(mean), jnp.asarray(std), state_dep=True)
    fields = bca.calibration.reference.FrozenReference._fields
    if not isinstance(payload["reference"], dict) or set(payload["reference"]) != set(fields):
        raise CheckpointNotReady("Frozen reference fields differ from actual BCA schema")
    calibration = components["calibration_receipt"]["json"]
    for key in ("residual_scale", "threshold", "lambda_hat", "lambda_hpd", "n_eff"):
        value = np.asarray(payload["reference"][key])
        if (value.shape != () or value.dtype != np.dtype("float32")
                or not np.isfinite(value) or float(value) != calibration[key]):
            raise CheckpointNotReady("Stored frozen reference differs from receipt: " + key)
    ready = np.asarray(payload["reference"]["ready"])
    if ready.shape != () or ready.dtype != np.dtype("bool") or not bool(ready):
        raise CheckpointNotReady("Stored reference ready flag must be scalar Boolean true")
    ref = bca.calibration.reference.FrozenReference(**jax.tree_util.tree_map(jnp.asarray, payload["reference"]))
    if not bool(bca.reference_valid(ref)) or not bool(ref.ready) or not np.isfinite(float(ref.threshold)):
        raise CheckpointNotReady("Invalid BCA reference: unready or nonfinite; evaluation blocked, not zero-penalty success")
    for label, variables in (("actor_params", payload["actor_params"]),
                             ("critic_params", payload["critic_params"]),
                             ("cal_params", payload["reference"]["cal_params"])):
        if not isinstance(variables, dict) or set(variables) != {"params"}:
            raise CheckpointNotReady(label + " must retain the complete native Flax variables dictionary")
        leaves = jax.tree_util.tree_leaves(variables)
        if not leaves or any(np.asarray(leaf).dtype != np.dtype("float32")
                             or not np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
            raise CheckpointNotReady(label + " requires nonempty finite float32 parameter leaves")
    with np.load(io.BytesIO(components["scale_bank"]["bytes"]), allow_pickle=False) as bank:
        if set(bank.files) != {"observations", "actions", "group_ids"}:
            raise CheckpointNotReady("Scale bank requires observations/actions/group_ids only, without labels")
        scale_bank = {key: np.array(bank[key], copy=True) for key in bank.files}
    if scale_bank["observations"].shape != (256, 14) or scale_bank["actions"].shape != (256, 4) or scale_bank["group_ids"].shape != (256,):
        raise CheckpointNotReady("Scale bank shapes differ from the prospective protocol")
    if any(scale_bank[key].dtype != np.dtype("float32") for key in ("observations", "actions")):
        raise CheckpointNotReady("Scale-bank observations and actions must be stored as float32")
    if not np.issubdtype(scale_bank["group_ids"].dtype, np.integer):
        raise CheckpointNotReady("Scale-bank group_ids must use an integer dtype, not float or Boolean")
    bank_groups = set(scale_bank["group_ids"].tolist())
    if not bank_groups.issubset(set(manifest["train_group_ids"])) or bank_groups & set(manifest["calibration_group_ids"]):
        raise CheckpointNotReady("Scale bank is not exclusively training groups")
    if not all(np.all(np.isfinite(value)) for value in scale_bank.values()) or np.any(np.abs(scale_bank["actions"]) > 1):
        raise CheckpointNotReady("Scale bank contains invalid observations or actions")

    class FrozenInference:
        def __init__(self):
            self.actor_params = jax.tree_util.tree_map(jnp.asarray, payload["actor_params"])
            self.critic_params = jax.tree_util.tree_map(jnp.asarray, payload["critic_params"])
            self.reference = ref
            self._actor = jax.jit(lambda obs: actor.apply(self.actor_params, obs).sample(seed=None))
            self._q = jax.jit(lambda obs, act: critic.apply(self.critic_params, obs, act))
            self._width = jax.jit(lambda obs, act: bca.live_width((None, None, calibrator), self.reference, obs, act))
            self._digest = self.digest()

        def digest(self):
            return hashlib.sha256(serialization.to_bytes({"actor": self.actor_params, "critic": self.critic_params,
                "reference": self.reference, "mean": mean, "std": std})).hexdigest()

        def assert_unchanged(self):
            if self.digest() != self._digest:
                raise RuntimeError("Frozen inference state changed")

        def act(self, observations):
            return np.asarray(self._actor(jnp.asarray(observations, jnp.float32)), dtype=np.float32)

        def q(self, observations, actions):
            return np.asarray(self._q(jnp.asarray(observations, jnp.float32), jnp.asarray(actions, jnp.float32)))

        def width(self, observations, actions):
            values = self._width(jnp.asarray(observations, jnp.float32), jnp.asarray(actions, jnp.float32))
            return tuple(np.asarray(value) for value in values)

    policy = FrozenInference()
    # Shape checks do not fit, initialize random weights, or query a simulator.
    obs, action = scale_bank["observations"][:2], scale_bank["actions"][:2]
    predicted_actions, predicted_q = policy.act(obs), policy.q(obs, action)
    if (predicted_actions.shape != (2, 4) or predicted_q.shape != (2, 2)
            or not np.all(np.isfinite(predicted_actions)) or not np.all(np.isfinite(predicted_q))
            or np.any(np.abs(predicted_actions) > 1)):
        raise CheckpointNotReady("Native inference shapes or numeric outputs are incompatible")
    widths, usable, valid = policy.width(obs, action)
    if widths.shape != (2,) or not bool(usable) or not bool(valid):
        raise CheckpointNotReady("Frozen calibrator cannot produce valid usable inference widths")
    policy.assert_unchanged()
    return policy, scale_bank, {**receipt, "manifest": manifest, "frozen_state_sha256": policy._digest}
