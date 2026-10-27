"""Tiny iterative affine behavior-cloning adapter for exercising the protocol.

This is a NumPy SGD/momentum example, not BCA, a neural RL implementation, or
evidence of improved control. Each update minimizes mean squared error between
the *unclipped* affine prediction and recorded executed actions. Rewards,
successors and terminal flags never enter this objective. Online replay therefore
imitates executed actions, including any exploration, without reward learning.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from numbers import Real

import numpy as np

from .datasets import assert_training_allowed
from .training_protocol import AdapterCapabilities, LearnerCheckpoint, TransitionBatch


ADAPTER_ID = "affine_sgd_bc_protocol_v1"
_COUNT_KEY = "affine_sgd"
_DEFAULT_SETTINGS = {"learning_rate": 0.01, "momentum": 0.0, "exploration_std": 0.0}
_COMPONENTS = {"model", "optimizer", "target_networks", "rng", "preprocessing"}
_NO_TARGETS = {"applicable": False,
               "reason": "affine_behavior_cloning_has_no_target_networks"}
_TRANSFORM = {"normalization": "fixed_train_mean_scale", "normalization_threshold": 1e-12,
              "constant_feature_scale": 1.0, "coefficient_layout": "14_weights_then_bias",
              "action_clip": [-1.0, 1.0], "action_dtype": "float32"}


class AdapterError(ValueError):
    """An adapter input or saved state violates its declared contract."""


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf8")


def _decode(data, keys):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AdapterError("checkpoint JSON contains duplicate keys")
            result[key] = value
        return result
    try:
        value = json.loads(data.decode("utf8"), object_pairs_hook=unique_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               AdapterError("checkpoint JSON must be finite")))
        if not isinstance(value, dict) or set(value) != set(keys):
            raise AdapterError("unexpected checkpoint component fields")
        if _json_bytes(value) != data:
            raise AdapterError("checkpoint component is not canonical JSON")
        return value
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError("invalid checkpoint JSON") from exc


def _array(value, shape, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu":
            raise AdapterError(f"{name} must contain real numbers, not bool/complex/strings")
        result = raw.astype(np.float64)
    except (TypeError, OverflowError, ValueError) as exc:
        raise AdapterError(f"{name} must contain real numbers") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise AdapterError(f"{name} must have shape {shape} and finite values")
    return result


def _settings(values, base=None):
    if values is None:
        values = {}
    if not isinstance(values, Mapping) or any(key not in _DEFAULT_SETTINGS for key in values):
        raise AdapterError("learning_settings supports only learning_rate, momentum, exploration_std")
    result = dict(_DEFAULT_SETTINGS if base is None else base)
    for key, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise AdapterError(f"{key} must be a real number")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise AdapterError(f"{key} must be finite") from exc
        if not np.isfinite(number):
            raise AdapterError(f"{key} must be finite")
        if ((key == "learning_rate" and number <= 0)
                or (key == "momentum" and not 0 <= number < 1)
                or (key == "exploration_std" and number < 0)):
            raise AdapterError(f"invalid {key}")
        result[key] = number
    return result


def _batch_arrays(batch):
    if not isinstance(batch, TransitionBatch) or len(batch) == 0:
        raise AdapterError("a nonempty TransitionBatch is required")
    observations = _array(batch.observations, (len(batch), 14), "observations")
    actions = _array(batch.actions, (len(batch), 4), "executed actions")
    if np.any(np.abs(actions) > 1):
        raise AdapterError("executed actions must lie in [-1, 1]")
    return observations, actions


def _prediction(observation, mean, scale, coefficients):
    try:
        raw = np.asarray(observation)
    except (TypeError, ValueError) as exc:
        raise AdapterError("observations must be a real (..., 14) array") from exc
    if raw.shape[-1:] != (14,):
        raise AdapterError("observations must have shape (..., 14)")
    x = _array(raw, raw.shape, "observations")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = ((x - mean) / scale) @ coefficients[:-1] + coefficients[-1]
    except FloatingPointError as exc:
        raise AdapterError("affine prediction produced nonfinite values") from exc
    if not np.all(np.isfinite(result)):
        raise AdapterError("affine prediction produced nonfinite values")
    return result


class FrozenAffineSGDBCPolicy:
    """Detached immutable inference state; no optimizer or random exploration."""
    __slots__ = ("_mean_bytes", "_scale_bytes", "_coefficient_bytes", "_state")

    def __init__(self, mean, scale, coefficients):
        for name, value, shape, storage in (("mean", mean, (14,), "_mean_bytes"),
                                            ("scale", scale, (14,), "_scale_bytes"),
                                            ("coefficients", coefficients, (15, 4), "_coefficient_bytes")):
            array = _array(value, shape, name)
            object.__setattr__(self, storage, array.astype("<f8").tobytes())
        if np.any(self.scale <= 0):
            raise AdapterError("normalization scale must be positive")
        object.__setattr__(self, "_state", _json_bytes({
            "schema": "thrusterchain.affine_sgd_bc.frozen.v1", "adapter_id": ADAPTER_ID,
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(), "transform": _TRANSFORM}))

    def __setattr__(self, name, value):
        raise AttributeError("frozen policy and normalization are immutable")

    def __delattr__(self, name):
        raise AttributeError("frozen policy and normalization are immutable")

    @property
    def mean(self):
        # A fresh view also isolates mutable ndarray shape/stride metadata.
        return np.frombuffer(self._mean_bytes, dtype="<f8")

    @property
    def scale(self):
        return np.frombuffer(self._scale_bytes, dtype="<f8")

    @property
    def coefficients(self):
        return np.frombuffer(self._coefficient_bytes, dtype="<f8").reshape(15, 4)

    def act(self, observation):
        return np.clip(_prediction(observation, self.mean, self.scale, self.coefficients),
                       -1, 1).astype(np.float32)

    def state_bytes(self):
        """Complete inference parameters, normalization and transform conventions."""
        return self._state


class AffineSGDBCAdapter:
    """Exactly one actual SGD/momentum step per update, with full safe resume.

    For N examples and four actions, L = sum((X theta - actions)**2)/(4*N),
    gradient = 2*X.T@(X theta-actions)/(4*N), velocity = momentum*velocity
    + gradient, theta = theta-learning_rate*velocity. Initialization fits only
    training-observation normalization and starts weights and velocity at zero.
    This single optimizer is also the declared primary optimizer, so primary
    updates and total physical optimizer steps have the same count here.
    """
    capabilities = AdapterCapabilities(adapter_id=ADAPTER_ID, fit_semantics="optimizer_steps",
                                       supports_online=True, primary_optimizer=_COUNT_KEY)

    def __init__(self):
        self._initialized = False

    def _require_initialized(self):
        if not self._initialized:
            raise AdapterError("adapter must be initialized or restored first")

    @property
    def optimizer_steps(self):
        self._require_initialized()
        return self._steps

    @property
    def optimizer_step_counts(self):
        self._require_initialized()
        return {_COUNT_KEY: self._steps}

    @property
    def learning_settings(self):
        self._require_initialized()
        return dict(self._settings)

    def initialize(self, train_batch, seed, learning_settings=None):
        assert_training_allowed("initialize affine SGD behavior cloning")
        observations, _ = _batch_arrays(train_batch)
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise AdapterError("seed must be a nonnegative integer")
        settings = _settings(learning_settings)
        try:
            with np.errstate(over="raise", invalid="raise"):
                mean, scale = observations.mean(axis=0), observations.std(axis=0)
        except FloatingPointError as exc:
            raise AdapterError("training normalization produced nonfinite values") from exc
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise AdapterError("training normalization produced nonfinite values")
        self._mean = mean
        self._scale = np.where(scale > _TRANSFORM["normalization_threshold"], scale, 1.0)
        self._coefficients = np.zeros((15, 4), dtype=np.float64)
        self._velocity = np.zeros((15, 4), dtype=np.float64)
        self._settings = settings
        self._rng = np.random.Generator(np.random.PCG64(int(seed)))
        self._steps = 0
        self._metadata = {"training_config": {}, "dataset_lineage": {},
                          "primary_optimizer": _COUNT_KEY,
                          "not_applicable_components": ["target_networks"]}
        self._initialized = True

    def configure_learning_settings(self, learning_settings):
        """Apply explicit overrides without resetting parameters, momentum or RNG."""
        assert_training_allowed("configure affine SGD behavior cloning")
        self._require_initialized()
        self._settings = _settings(learning_settings, self._settings)

    def update(self, batch):
        assert_training_allowed("update affine SGD behavior cloning")
        self._require_initialized()
        observations, actions = _batch_arrays(batch)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                normalized = (observations - self._mean) / self._scale
                design = np.column_stack((normalized, np.ones(len(batch))))
                residual = design @ self._coefficients - actions
                loss = float(np.mean(residual ** 2))
                gradient = (2.0 / (4 * len(batch))) * (design.T @ residual)
                velocity = self._settings["momentum"] * self._velocity + gradient
                coefficients = self._coefficients - self._settings["learning_rate"] * velocity
        except FloatingPointError as exc:
            raise AdapterError("SGD update produced nonfinite values; state was not changed") from exc
        if not np.isfinite(loss) or not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(coefficients)):
            raise AdapterError("SGD update produced nonfinite values; state was not changed")
        self._velocity, self._coefficients = velocity, coefficients
        self._steps += 1
        return {"loss": loss, "optimizer_steps_performed": 1, "optimizer_steps": self._steps,
                "optimizer_step_counts": {_COUNT_KEY: self._steps},
                "objective": "unclipped_affine_executed_action_mse"}

    def act(self, observation, explore=False):
        self._require_initialized()
        if not isinstance(explore, bool):
            raise AdapterError("explore must be a bool")
        result = _prediction(observation, self._mean, self._scale, self._coefficients)
        if explore:
            assert_training_allowed("explore with affine SGD behavior cloning")
            # Clone first so a numerical failure does not consume the saved RNG.
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = self._rng.bit_generator.state
            try:
                with np.errstate(over="raise", invalid="raise"):
                    result = result + rng.normal(0.0, self._settings["exploration_std"], result.shape)
            except FloatingPointError as exc:
                raise AdapterError("exploration produced nonfinite values") from exc
            if not np.all(np.isfinite(result)):
                raise AdapterError("exploration produced nonfinite values")
            self._rng = rng
        return np.clip(result, -1, 1).astype(np.float32)

    def _adapter_metadata(self):
        return {"purpose": "protocol_example_only_not_BCA_or_neural_RL",
                "primary_optimizer": _COUNT_KEY,
                "objective": "unclipped_affine_executed_action_mse",
                "online_semantics": "executed_action_imitation_without_reward_learning",
                "learning_settings": dict(self._settings), "transform": _TRANSFORM}

    def checkpoint(self):
        self._require_initialized()
        components = {
            "model": _json_bytes({"schema": "affine_sgd_bc.model.v1",
                                  "coefficients": self._coefficients.tolist()}),
            "optimizer": _json_bytes({"schema": "affine_sgd_bc.optimizer.v1",
                                      "velocity": self._velocity.tolist(),
                                      "learning_settings": self._settings,
                                      "optimizer_steps": self._steps}),
            "target_networks": _json_bytes(_NO_TARGETS),
            "rng": _json_bytes({"schema": "affine_sgd_bc.rng.v1",
                                "state": self._rng.bit_generator.state}),
            "preprocessing": _json_bytes({"schema": "affine_sgd_bc.preprocessing.v1",
                                          "mean": self._mean.tolist(), "scale": self._scale.tolist(),
                                          "normalization_source": "initialize_train_batch_only",
                                          "transform": _TRANSFORM})}
        metadata = json.loads(_json_bytes(self._metadata))
        metadata["adapter"] = self._adapter_metadata()
        return LearnerCheckpoint(adapter_id=ADAPTER_ID, components=components,
                                 optimizer_steps=self._steps,
                                 optimizer_step_counts={_COUNT_KEY: self._steps}, metadata=metadata)

    def restore(self, checkpoint):
        """Validate every component before atomically replacing this learner."""
        assert_training_allowed("restore affine SGD behavior cloning")
        if not isinstance(checkpoint, LearnerCheckpoint) or checkpoint.adapter_id != ADAPTER_ID:
            raise AdapterError("checkpoint adapter identifier mismatch")
        if set(checkpoint.components) != _COMPONENTS:
            raise AdapterError("unexpected checkpoint components")
        model = _decode(checkpoint.components["model"], {"schema", "coefficients"})
        optimizer = _decode(checkpoint.components["optimizer"],
                            {"schema", "velocity", "learning_settings", "optimizer_steps"})
        prep = _decode(checkpoint.components["preprocessing"],
                       {"schema", "mean", "scale", "normalization_source", "transform"})
        rng_data = _decode(checkpoint.components["rng"], {"schema", "state"})
        for value, schema in ((model, "model"), (optimizer, "optimizer"),
                              (prep, "preprocessing"), (rng_data, "rng")):
            if value["schema"] != f"affine_sgd_bc.{schema}.v1":
                raise AdapterError("unsupported checkpoint component schema")
        if checkpoint.components["target_networks"] != _json_bytes(_NO_TARGETS):
            raise AdapterError("this adapter has no target networks")
        if checkpoint.metadata.get("not_applicable_components") != ["target_networks"]:
            raise AdapterError("checkpoint must explicitly declare target networks not applicable")
        if checkpoint.metadata.get("primary_optimizer") != _COUNT_KEY:
            raise AdapterError("checkpoint must declare affine_sgd as its primary optimizer")
        if (prep["normalization_source"] != "initialize_train_batch_only"
                or prep["transform"] != _TRANSFORM):
            raise AdapterError("unsupported checkpoint normalization or transform")
        if (isinstance(optimizer["optimizer_steps"], bool)
                or not isinstance(optimizer["optimizer_steps"], int)
                or optimizer["optimizer_steps"] != checkpoint.optimizer_steps
                or dict(checkpoint.optimizer_step_counts) != {_COUNT_KEY: checkpoint.optimizer_steps}):
            raise AdapterError("inconsistent optimizer step counts")
        candidate = AffineSGDBCAdapter()
        candidate._coefficients = _array(model["coefficients"], (15, 4), "coefficients")
        candidate._velocity = _array(optimizer["velocity"], (15, 4), "velocity")
        candidate._mean = _array(prep["mean"], (14,), "mean")
        candidate._scale = _array(prep["scale"], (14,), "scale")
        if np.any(candidate._scale <= 0):
            raise AdapterError("normalization scale must be positive")
        candidate._settings = _settings(optimizer["learning_settings"])
        candidate._rng = np.random.Generator(np.random.PCG64())
        try:
            candidate._rng.bit_generator.state = rng_data["state"]
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise AdapterError("invalid PCG64 checkpoint RNG state") from exc
        candidate._steps = checkpoint.optimizer_steps
        candidate._metadata = checkpoint.metadata
        candidate._initialized = True
        if candidate.checkpoint().to_bytes() != checkpoint.to_bytes():
            raise AdapterError("checkpoint state or metadata does not round-trip exactly")
        self.__dict__.update(candidate.__dict__)

    def freeze(self):
        self._require_initialized()
        return FrozenAffineSGDBCPolicy(self._mean, self._scale, self._coefficients)
