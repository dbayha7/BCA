"""Strict-offline behavior cloning, selection receipts and frozen live evaluation."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np

from .datasets import (DatasetError, SPLITS, TrajectoryDataset, canonical_json,
                       content_digest, immutable_array, _ACTIVE_EVALUATION,
                       forbid_during_evaluation)
from .quality_datasets import BehaviorDataset

RECEIPT_SCHEMA = "thrusterchain.selection_receipt.v1"
FAMILIES = ("recorded", "strong_current", "big_waves", "big_kicks")


def _affine_inference(observation, mean, scale, coefficients):
    """Pure inference: fixed normalization, affine transform, clipping and cast."""
    raw = np.asarray(observation)
    if raw.dtype.kind not in "fiu":
        raise ProtocolError("observations must be real numeric values, not bool, complex or strings")
    x = raw.astype(np.float64)
    if x.shape[-1:] != (14,) or not np.all(np.isfinite(x)):
        raise ProtocolError("prediction requires finite (..., 14) observations")
    z = (x - mean) / scale
    return np.clip(z @ coefficients[:-1] + coefficients[-1], -1, 1).astype(np.float32)


_BUILTIN_INFERENCE = _affine_inference
INFERENCE_IMPLEMENTATION_SHA256 = hashlib.sha256(inspect.getsource(_affine_inference).encode("utf8")).hexdigest()


class ProtocolError(ValueError):
    """An operation would violate the declared offline/frozen protocol."""


class FrozenLinearPolicy:
    """Prediction-only, bytes-backed affine policy with fixed train statistics."""
    __slots__ = ("mean", "scale", "coefficients", "ridge")

    def __init__(self, mean, scale, coefficients, ridge):
        shapes = {"mean": (14,), "scale": (14,), "coefficients": (15, 4)}
        for name, value in (("mean", mean), ("scale", scale), ("coefficients", coefficients)):
            raw = np.asarray(value)
            if raw.dtype.kind not in "fiu":
                raise ProtocolError(f"{name} must contain real numeric values")
            a = raw.astype(np.float64)
            if a.shape != shapes[name] or not np.all(np.isfinite(a)):
                raise ProtocolError(f"{name} must have shape {shapes[name]} and finite values")
            object.__setattr__(self, name, immutable_array(a, "<f8"))
        if np.any(self.scale <= 0) or not np.isfinite(ridge) or ridge < 0:
            raise ProtocolError("scale must be positive and ridge finite and nonnegative")
        object.__setattr__(self, "ridge", float(ridge))

    def __setattr__(self, name, value):
        raise AttributeError("policy and normalization are frozen")

    def predict(self, observation):
        return _affine_inference(observation, self.mean, self.scale, self.coefficients)

    __call__ = predict

    @property
    def digest(self):
        return content_digest({"mean": self.mean, "scale": self.scale,
                               "coefficients": self.coefficients},
                              {"kind": "affine_behavior_cloning_v1", "ridge": self.ridge,
                               "action_clip": [-1, 1], "action_dtype": "float32",
                               "inference_implementation_sha256": INFERENCE_IMPLEMENTATION_SHA256,
                               "mutable_buffers": 0, "preprocessing": "fixed_train_mean_scale"})


_BUILTIN_PREDICT = FrozenLinearPolicy.predict


class SelectionReceipt:
    """Tamper-evident protocol record, not a cryptographic proof of conduct."""
    __slots__ = ("_json", "_digest")

    def __init__(self, payload, *, expected_digest=None):
        text = canonical_json(payload)
        digest = hashlib.sha256(text.encode("utf8")).hexdigest()
        if expected_digest is not None and expected_digest != digest:
            raise ProtocolError("selection receipt SHA-256 mismatch")
        value = json.loads(text)
        required = {"schema", "protocol", "dataset_sha256", "split_sha256", "policy_sha256",
                    "physics", "task_semantics", "provenance", "candidate_ridges",
                    "selected_ridge", "tuning_validation_mse", "selection_mse",
                    "selection_episode_mse", "normalization_source", "split_usage",
                    "budget", "dataset_episode_seeds", "limitations"}
        if set(value) != required or value.get("schema") != RECEIPT_SCHEMA:
            raise ProtocolError("invalid selection-receipt schema")
        if value.get("protocol") not in ("strict_offline", "simulator_assisted_selection"):
            raise ProtocolError("unknown selection regime")
        if value.get("normalization_source") != "train_only":
            raise ProtocolError("normalization must be fitted to training observations only")
        if value.get("task_semantics") not in ("finite_horizon", "continuing_time_limit") or value.get("provenance") not in ("smoke", "research"):
            raise ProtocolError("invalid task semantics or provenance in selection receipt")
        physics = value["physics"]
        if (not isinstance(physics, dict) or set(physics) != {"stall", "stall_slope"}
                or not isinstance(physics["stall"], bool)
                or not isinstance(physics["stall_slope"], (int, float))
                or isinstance(physics["stall_slope"], bool) or physics["stall_slope"] not in (1.5, 3.0)):
            raise ProtocolError("invalid frozen physics")
        candidates, scores = value["candidate_ridges"], value["tuning_validation_mse"]
        episode_scores = value["selection_episode_mse"]
        if (not isinstance(candidates, list) or not candidates or
                not isinstance(scores, list) or len(scores) != len(candidates) or
                not isinstance(episode_scores, list) or not episode_scores):
            raise ProtocolError("invalid candidate grid or score arrays")
        for number in candidates + scores + episode_scores + [value["selection_mse"], value["selected_ridge"]]:
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number) or number < 0:
                raise ProtocolError("ridge values and imitation losses must be finite and nonnegative")
        if len(set(candidates)) != len(candidates) or value["selected_ridge"] not in candidates:
            raise ProtocolError("selected ridge is absent from a distinct candidate grid")
        if value["protocol"] == "strict_offline" and value["selected_ridge"] != candidates[int(np.argmin(scores))]:
            raise ProtocolError("selected ridge disagrees with the predeclared offline rule")
        if not np.isclose(value["selection_mse"], np.mean(episode_scores), rtol=1e-13, atol=1e-14):
            raise ProtocolError("selection mean disagrees with per-episode losses")
        budget = value["budget"]
        budget_keys = {"candidate_fits", "unique_training_transitions", "total_training_rows_across_fits",
                       "simulator_selection_episodes", "simulator_selection_steps", "online_parameter_updates",
                       "model_generated_transitions", "prior_development_feedback"}
        if not isinstance(budget, dict) or set(budget) != budget_keys:
            raise ProtocolError("invalid interaction budget")
        for key in budget_keys - {"prior_development_feedback"}:
            if isinstance(budget[key], bool) or not isinstance(budget[key], int) or budget[key] < 0:
                raise ProtocolError("interaction budget counts must be nonnegative integers")
        if (budget["candidate_fits"] != len(candidates) or
                budget["total_training_rows_across_fits"] != budget["candidate_fits"] * budget["unique_training_transitions"]):
            raise ProtocolError("candidate fitting budget is inconsistent")
        if budget["online_parameter_updates"] or (value["protocol"] == "strict_offline" and
                (budget["simulator_selection_episodes"] or budget["simulator_selection_steps"])):
            raise ProtocolError("interaction budget contradicts the offline selection regime")
        seeds = value["dataset_episode_seeds"]
        if not isinstance(seeds, list) or len(seeds) < 10 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds):
            raise ProtocolError("invalid dataset episode seeds")
        if set(value["split_sha256"]) != set(SPLITS):
            raise ProtocolError("receipt must bind all four split hashes")
        for bound_digest in [value["dataset_sha256"], value["policy_sha256"], *value["split_sha256"].values()]:
            if not isinstance(bound_digest, str) or len(bound_digest) != 64 or any(c not in "0123456789abcdef" for c in bound_digest):
                raise ProtocolError("receipt has an invalid SHA-256")
        object.__setattr__(self, "_json", text)
        object.__setattr__(self, "_digest", digest)

    def __setattr__(self, name, value):
        raise AttributeError("selection receipt is immutable")

    @property
    def payload(self):
        return json.loads(self._json)

    @property
    def digest(self):
        return self._digest


def _episode_mse(policy, split):
    error = policy.predict(split["observations"][:, :-1]).astype(np.float64) - split["actions"]
    return np.mean(error ** 2, axis=(1, 2))


def train_behavior_cloning(dataset, *, ridge_values=(0.0, 1e-3, .1, 10.0)):
    """Fit only training rows; choose ridge on tuning_validation, report selection.

    This function never instantiates or calls an environment. The selection score
    is held-out imitation error, not an off-policy estimate of policy return.
    """
    forbid_during_evaluation("train_behavior_cloning")
    if type(dataset) not in (TrajectoryDataset, BehaviorDataset):
        raise ProtocolError("training requires a validated immutable dataset")
    try:
        candidates = tuple(float(x) for x in ridge_values)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("ridge_values must be a nonempty finite nonnegative sequence") from exc
    if not candidates or any(not np.isfinite(x) or x < 0 for x in candidates) or len(set(candidates)) != len(candidates):
        raise ProtocolError("ridge_values must contain distinct finite nonnegative values")
    train = dataset.split("train")
    x = train["observations"][:, :-1].reshape(-1, 14).astype(np.float64)
    y = train["actions"].reshape(-1, 4).astype(np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1., scale)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    tuning = dataset.split("tuning_validation")
    policies, scores = [], []
    for ridge in candidates:
        # Augmented least squares is well-defined even for ridge=0/rank deficiency.
        penalty = np.diag(np.r_[np.full(14, np.sqrt(ridge)), 0.])
        coef = np.linalg.lstsq(np.vstack((design, penalty)),
                               np.vstack((y, np.zeros((15, 4)))), rcond=None)[0]
        policy = FrozenLinearPolicy(mean, scale, coef, ridge)
        policies.append(policy)
        scores.append(float(_episode_mse(policy, tuning).mean()))
    # Grid order is the predeclared deterministic tie breaker.
    winner = int(np.argmin(scores))
    policy = policies[winner]
    selection_scores = _episode_mse(policy, dataset.split("selection"))
    metadata = dataset.metadata
    receipt = SelectionReceipt({
        "schema": RECEIPT_SCHEMA, "protocol": "strict_offline",
        "dataset_sha256": dataset.digest,
        "split_sha256": {name: dataset.split_digest(name) for name in SPLITS},
        "policy_sha256": policy.digest, "physics": metadata["physics"],
        "task_semantics": metadata["task_semantics"], "provenance": metadata["provenance"],
        "candidate_ridges": list(candidates), "selected_ridge": policy.ridge,
        "tuning_validation_mse": scores, "selection_mse": float(selection_scores.mean()),
        "selection_episode_mse": selection_scores.tolist(), "normalization_source": "train_only",
        "split_usage": {"train": "fit normalization and each candidate policy",
                        "tuning_validation": "select ridge by episode-averaged action MSE",
                        "selection": "report selected policy action MSE once; no refit",
                        "final_calibration": "reserved; hashed only, unused by fitting or selection"},
        "budget": {"candidate_fits": len(candidates), "unique_training_transitions": len(x),
                   "total_training_rows_across_fits": len(x) * len(candidates),
                   "simulator_selection_episodes": 0, "simulator_selection_steps": 0,
                   "online_parameter_updates": 0, "model_generated_transitions": 0,
                   "prior_development_feedback": "not attested by this local receipt"},
        "dataset_episode_seeds": dataset.arrays["episode_seeds"].tolist(),
        "limitations": ["Action MSE measures imitation, not expected return.",
                        "Closed-loop occupancy is unknown; requested-action densities do not supply it.",
                        "Receipt hashes reveal edits, not external data access or prior tuning."],
    })
    return policy, receipt


def _verify_frozen(policy, receipt, dataset=None):
    if FrozenLinearPolicy.predict is not _BUILTIN_PREDICT or _affine_inference is not _BUILTIN_INFERENCE:
        raise ProtocolError("frozen inference implementation was replaced")
    if type(policy) is not FrozenLinearPolicy or type(receipt) is not SelectionReceipt:
        raise ProtocolError("evaluation requires the prediction-only frozen policy and a selection receipt")
    if policy.digest != receipt.payload["policy_sha256"]:
        raise ProtocolError("policy/statistics changed after selection")
    if dataset is not None:
        if type(dataset) not in (TrajectoryDataset, BehaviorDataset) or dataset.digest != receipt.payload["dataset_sha256"]:
            raise ProtocolError("calibration dataset differs from the selected dataset")
        for name in SPLITS:
            if dataset.split_digest(name) != receipt.payload["split_sha256"][name]:
                raise ProtocolError(f"{name} content changed after selection")


def freeze_policy(policy, receipt, directory):
    forbid_during_evaluation("freeze_policy")
    _verify_frozen(policy, receipt)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A single exclusive file is the transaction: no partial two-file pair.
    path = directory / "frozen_policy.npz"
    envelope = {"receipt": receipt.payload, "receipt_sha256": receipt.digest, "ridge": policy.ridge}
    with path.open("xb") as handle:
        np.savez_compressed(handle, mean=policy.mean, scale=policy.scale,
                            coefficients=policy.coefficients, manifest=np.asarray(canonical_json(envelope)))
    return path


def load_frozen_policy(directory):
    path = Path(directory)
    if path.is_dir():
        path /= "frozen_policy.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"mean", "scale", "coefficients", "manifest"}:
                raise ProtocolError("invalid frozen-policy members")
            manifest = archive["manifest"]
            if manifest.shape != () or manifest.dtype.kind != "U":
                raise ProtocolError("invalid frozen-policy manifest")
            envelope = json.loads(str(manifest))
            if set(envelope) != {"receipt", "receipt_sha256", "ridge"}:
                raise ProtocolError("invalid frozen-policy envelope")
            policy = FrozenLinearPolicy(archive["mean"], archive["scale"], archive["coefficients"], envelope["ridge"])
        receipt = SelectionReceipt(envelope["receipt"], expected_digest=envelope["receipt_sha256"])
        _verify_frozen(policy, receipt)
        return policy, receipt
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load frozen policy: {exc}") from exc


def calibrate_frozen_policy(dataset, policy, receipt, *, alpha=.1):
    """Fresh episode-level split-conformal *imitation* residual calibration.

    Correlation within trajectories is handled by taking the episode maximum.
    Coverage requires exchangeable future recorder trajectories. It is not a
    rollout-return certificate and supplies no coverage claim under shifts.
    """
    forbid_during_evaluation("calibrate_frozen_policy")
    _verify_frozen(policy, receipt, dataset)
    if not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ProtocolError("alpha must be finite and strictly between zero and one")
    split = dataset.split("final_calibration")
    residual = np.abs(policy.predict(split["observations"][:, :-1]).astype(np.float64) - split["actions"])
    scores = residual.max(axis=(1, 2))
    rank = int(math.ceil((len(scores) + 1) * (1 - alpha)))
    # A small calibration set must produce a vacuous interval, not a falsely
    # finite quantile obtained by clipping the conformal rank.
    radius = None if rank > len(scores) else float(np.sort(scores)[rank - 1])
    _verify_frozen(policy, receipt, dataset)
    balanced_mixture = (type(dataset) is BehaviorDataset and
                        dataset.metadata["behavior"]["name"] == "mixed_random_strong")
    claim = ("descriptive only: fixed-quota mixture does not establish exchangeability with a future independent episode"
             if balanced_mixture else "episode-max imitation residual under exchangeable recorder trajectories only")
    return {"schema": "thrusterchain.imitation_calibration.v1", "receipt_sha256": receipt.digest,
            "policy_sha256": policy.digest, "dataset_sha256": dataset.digest,
            "split": "final_calibration", "alpha": float(alpha), "episodes": len(scores),
            "episode_max_absolute_residuals": scores.tolist(), "order_statistic_rank": rank,
            "radius": radius, "vacuous": radius is None, "policy_updated": False,
            "claim": claim,
            "not_a_return_or_shift_certificate": True}


def evaluate_frozen_policy(policy, receipt, *, episodes_per_family=10, seed=20271022,
                           provenance=None, purpose="final_reporting"):
    """Live, prediction-only rollouts on the declared start families.

    Set purpose='simulator_assisted_tuning' if outputs will inform selection.
    Such a run is explicitly ineligible as strict-offline final evidence.
    """
    _verify_frozen(policy, receipt)
    if isinstance(episodes_per_family, bool) or not isinstance(episodes_per_family, (int, np.integer)) or episodes_per_family < 1:
        raise ProtocolError("episodes_per_family must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ProtocolError("evaluation seed must be a nonnegative integer")
    if purpose not in ("final_reporting", "simulator_assisted_tuning"):
        raise ProtocolError("evaluation purpose must be explicit")
    record = receipt.payload
    provenance = record["provenance"] if provenance is None else provenance
    if provenance not in ("smoke", "research"):
        raise ProtocolError("provenance must be smoke or research")
    if record["provenance"] == "smoke" and provenance != "smoke":
        raise ProtocolError("a smoke dataset cannot be relabeled as research evidence")
    offline_seeds = set(record["dataset_episode_seeds"])
    rng = np.random.default_rng([int(seed), 2701])
    episode_seeds = []
    while len(episode_seeds) < episodes_per_family:
        value = int(rng.integers(0, np.iinfo(np.int32).max))
        if value not in offline_seeds and value not in episode_seeds:
            episode_seeds.append(value)
    from .environment import ThrusterChainEnv
    before = policy.digest
    results = {}
    audit = {"inference_calls": 0, "forbidden_api_calls": 0,
             "scope": "built-in collection, fitting, calibration and persistence APIs",
             "inference_implementation_sha256": INFERENCE_IMPLEMENTATION_SHA256,
             "mutable_policy_buffers": 0, "arbitrary_policy_callbacks": 0}
    if _ACTIVE_EVALUATION.get() is not None:
        raise ProtocolError("nested frozen evaluation is not allowed")
    token = _ACTIVE_EVALUATION.set(audit)
    try:
        for family in FAMILIES:
            mode = "time_aware" if record["task_semantics"] == "finite_horizon" else "physical_padded"
            env = ThrusterChainEnv(**record["physics"], start=family, observation_mode=mode,
                                  task_semantics=record["task_semantics"])
            returns, discounted, stall_fractions = [], [], []
            try:
                for episode_seed in episode_seeds:
                    obs, _ = env.reset(seed=episode_seed)
                    total, disc, stalled = 0., 0., 0
                    for step in range(100):
                        # Use the fixed, source-hashed inference transform, never an
                        # arbitrary user callback with hidden buffers or learning.
                        command = _BUILTIN_INFERENCE(obs, policy.mean, policy.scale, policy.coefficients)
                        audit["inference_calls"] += 1
                        obs, reward, terminated, truncated, info = env.step(command)
                        total += float(reward)
                        disc += .98 ** step * float(reward)
                        stalled += int(np.count_nonzero(info["stalled"]))
                        if bool(terminated or truncated) != (step == 99):
                            raise ProtocolError("live environment ended at an unexpected step")
                        if step == 99 and bool(terminated) != (record["task_semantics"] == "finite_horizon"):
                            raise ProtocolError("live terminal flags disagree with frozen task semantics")
                    returns.append(total)
                    discounted.append(disc)
                    stall_fractions.append(stalled / 400)
            finally:
                env.close()
            results[family] = {
                "episodes": episodes_per_family, "episode_seeds": episode_seeds,
                "returns": returns, "discounted_returns": discounted,
                "mean_return": float(np.mean(returns)), "mean_discounted_return": float(np.mean(discounted)),
                "sample_sd_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else None,
                "stall_fractions": stall_fractions,
                "held_out_from_dataset_generation": family != "recorded",
            }
    finally:
        _ACTIVE_EVALUATION.reset(token)
    if audit["forbidden_api_calls"] != 0:
        raise ProtocolError("forbidden training or persistence attempted during evaluation")
    _verify_frozen(policy, receipt)
    if before != policy.digest:
        raise ProtocolError("policy or normalization changed during live evaluation")
    return {"schema": "thrusterchain.regular_live_evaluation.v2", "provenance": provenance,
            "evidence_status": "smoke_only_not_research_evidence" if provenance == "smoke" else "single_frozen_policy_run",
            "purpose": purpose, "protocol": "simulator_assisted_selection" if purpose == "simulator_assisted_tuning" else record["protocol"],
            # A declaration and a frozen-policy receipt cannot prove that these
            # families/seeds were not inspected during prior development.
            "eligible_strict_offline_final_reporting": (None if purpose == "final_reporting" and record["protocol"] == "strict_offline" else False),
            "eligible_research_evidence": (None if provenance == "research" and purpose == "final_reporting" and record["protocol"] == "strict_offline" else False),
            "eligibility_status": "unverified_prior_exposure" if purpose == "final_reporting" and record["protocol"] == "strict_offline" else "ineligible_declared_tuning",
            "receipt_sha256": receipt.digest, "dataset_sha256": record["dataset_sha256"],
            "policy_sha256_before": before, "policy_sha256_after": policy.digest,
            "parameter_updates": 0, "normalization_updates": 0, "evaluation_seed": int(seed),
            "inference_audit": audit,
            "environment_episodes": episodes_per_family * len(FAMILIES),
            "environment_steps": episodes_per_family * len(FAMILIES) * 100,
            "paired_start_seeds_across_families": True, "families": results,
            "limits": ["Final reporting outcomes must not be used to retune or select this policy.",
                       "Null eligibility means prior exposure and the selection history are not independently attested.",
                       "Family results share random seeds and are not independent across families.",
                       "A single fitted policy does not estimate variation across training seeds."]}
