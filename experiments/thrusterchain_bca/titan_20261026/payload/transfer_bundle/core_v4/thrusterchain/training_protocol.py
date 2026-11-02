"""Explicit learning budgets and a cooperative host-adapter checkpoint contract.

This module imports no live environment. Validation and hashes detect accidental
state changes; they cannot prove that arbitrary host code is pure or complete.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np


DEFAULT_OFFLINE_UPDATES = 1_000_000
MAX_OFFLINE_UPDATES = 3_000_000
DEFAULT_BATCH_SIZE = 256
REQUIRED_CHECKPOINT_COMPONENTS = frozenset(
    {"model", "optimizer", "target_networks", "rng", "preprocessing"}
)
CHECKPOINT_FORMAT = "thrusterchain.learner-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    # bool and NumPy integer scalars deliberately do not satisfy this contract.
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int (not bool or float)")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a Python bool")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _canonical_json(value: Any) -> str:
    """Canonical, finite JSON; no object deserialization or custom encoders."""
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not np.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise TypeError("JSON object keys must be strings")
            for child in item.values():
                check(child)
            return
        if type(item) in (list, tuple):
            for child in item:
                check(child)
            return
        raise TypeError(f"unsupported JSON value type: {type(item).__name__}")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _settings_json(value: Any, name: str) -> str:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an explicit dict")
    return _canonical_json(value)


@dataclass(frozen=True)
class OfflineBudget:
    updates: int = DEFAULT_OFFLINE_UPDATES
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        _integer(self.updates, "updates")
        _integer(self.batch_size, "batch_size", 1)
        if self.updates > MAX_OFFLINE_UPDATES:
            raise ValueError(f"offline updates must be <= {MAX_OFFLINE_UPDATES}")

    def to_dict(self) -> dict[str, Any]:
        return {"updates": self.updates, "batch_size": self.batch_size}


@dataclass(frozen=True, init=False)
class OnlineBudget:
    interactions: int
    updates: int
    offline_fraction: float
    batch_size: int
    _learning_settings_json: str = field(repr=False)

    def __init__(self, interactions: int, updates: int, offline_fraction: float,
                 batch_size: int, learning_settings: dict[str, Any]) -> None:
        _integer(interactions, "interactions")
        _integer(updates, "updates")
        _integer(batch_size, "batch_size", 1)
        if updates > 0 and interactions == 0:
            raise ValueError("positive online updates require positive interactions")
        if type(offline_fraction) not in (int, float):
            raise TypeError("offline_fraction must be a finite Python number")
        if not np.isfinite(offline_fraction) or not 0 <= offline_fraction <= 1:
            raise ValueError("offline_fraction must lie in [0, 1]")
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(self, "updates", updates)
        object.__setattr__(self, "offline_fraction", float(offline_fraction))
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "_learning_settings_json",
                           _settings_json(learning_settings, "learning_settings"))

    @property
    def learning_settings(self) -> dict[str, Any]:
        return json.loads(self._learning_settings_json)

    def to_dict(self) -> dict[str, Any]:
        return {"interactions": self.interactions, "updates": self.updates,
                "offline_fraction": self.offline_fraction,
                "batch_size": self.batch_size,
                "learning_settings": self.learning_settings}


@dataclass(frozen=True)
class EvaluationConfig:
    frozen_policy: bool = True
    episodes: int = 2
    seed: int = 20361022
    online: OnlineBudget | None = None

    def __post_init__(self) -> None:
        _boolean(self.frozen_policy, "frozen_policy")
        _integer(self.episodes, "episodes", 1)
        _integer(self.seed, "seed")
        if self.online is not None and not isinstance(self.online, OnlineBudget):
            raise TypeError("online must be OnlineBudget or None")
        if self.frozen_policy and self.online is not None:
            raise ValueError("frozen evaluation rejects online adaptation settings")
        if not self.frozen_policy and self.online is None:
            raise ValueError("unfrozen evaluation requires an explicit OnlineBudget")

    def to_dict(self) -> dict[str, Any]:
        return {"frozen_policy": self.frozen_policy, "episodes": self.episodes,
                "seed": self.seed,
                "online": None if self.online is None else self.online.to_dict()}


@dataclass(frozen=True)
class AdapterCapabilities:
    adapter_id: str
    fit_semantics: Literal["optimizer_steps", "closed_form"]
    supports_online: bool
    primary_optimizer: str | None = None

    def __post_init__(self) -> None:
        _nonempty_string(self.adapter_id, "adapter_id")
        if self.fit_semantics not in ("optimizer_steps", "closed_form"):
            raise ValueError("fit_semantics must be optimizer_steps or closed_form")
        _boolean(self.supports_online, "supports_online")
        if self.fit_semantics == "optimizer_steps":
            _nonempty_string(self.primary_optimizer, "primary_optimizer")
        elif self.primary_optimizer is not None:
            raise ValueError("closed_form capabilities must declare primary_optimizer=None")

    def to_dict(self) -> dict[str, Any]:
        return {"adapter_id": self.adapter_id, "fit_semantics": self.fit_semantics,
                "supports_online": self.supports_online,
                "primary_optimizer": self.primary_optimizer}


@dataclass(frozen=True, eq=False)
class TransitionBatch:
    """Detached transitions; actions are the actual executed actions.

    Termination disables bootstrapping; time-limit truncation alone does not.
    Arrays use immutable bytes storage, so write flags cannot be re-enabled.
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bootstrap_mask: np.ndarray
    origin: np.ndarray
    group_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays = {name: np.asarray(getattr(self, name)) for name in self.__dataclass_fields__}
        if arrays["observations"].ndim < 1:
            raise ValueError("observations must have a transition axis")
        size = len(arrays["observations"])
        for name, array in arrays.items():
            if array.ndim < 1 or len(array) != size:
                raise ValueError(f"{name} must have the same transition axis")
            if array.dtype.kind in "OV":
                raise TypeError(f"{name} must not use object or structured dtype")
        if arrays["next_observations"].shape != arrays["observations"].shape:
            raise ValueError("next_observations must match observations shape")
        for name in ("rewards", "terminated", "truncated", "bootstrap_mask", "origin", "group_ids"):
            if arrays[name].ndim != 1:
                raise ValueError(f"{name} must have shape (N,)")
        for name in ("observations", "actions", "rewards", "next_observations", "bootstrap_mask"):
            if arrays[name].dtype.kind not in "biuf" or not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"{name} must contain finite real values")
        for name in ("terminated", "truncated"):
            if arrays[name].dtype.kind != "b":
                raise TypeError(f"{name} must be a boolean array")
        if arrays["origin"].dtype.kind not in "iu" or not np.all(np.isin(arrays["origin"], (0, 1))):
            raise ValueError("origin must contain integer 0 (offline) or 1 (online)")
        if arrays["group_ids"].dtype.kind not in "iuSU":
            raise TypeError("group_ids must be integer or fixed-width string identifiers")
        if not np.array_equal(arrays["bootstrap_mask"], ~arrays["terminated"]):
            raise ValueError("bootstrap_mask must equal 1 - terminated; truncation still bootstraps")
        for name, array in arrays.items():
            # Store only immutable data and layout, not a mutable ndarray header.
            object.__setattr__(self, name, (array.tobytes(order="C"), array.dtype, array.shape))

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if (name in object.__getattribute__(self, "__dataclass_fields__")
                and type(value) is tuple and len(value) == 3 and type(value[0]) is bytes):
            storage, dtype, shape = value
            return np.frombuffer(storage, dtype=dtype).reshape(shape)
        return value

    def __len__(self) -> int:
        return len(self.observations)

    def select(self, indices: Any) -> TransitionBatch:
        if not isinstance(indices, slice):
            indices = np.asarray(indices)
            if indices.ndim == 0:
                indices = indices.reshape(1)
            if indices.ndim != 1 or indices.dtype.kind not in "iub":
                raise TypeError("indices must be an integer vector, boolean mask, or slice")
        return TransitionBatch(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})

    def concat(self, *others: TransitionBatch) -> TransitionBatch:
        batches = (self,) + others
        if any(not isinstance(batch, TransitionBatch) for batch in batches):
            raise TypeError("concat accepts TransitionBatch objects")
        return TransitionBatch(**{
            name: np.concatenate([getattr(batch, name) for batch in batches], axis=0)
            for name in self.__dataclass_fields__
        })


@dataclass(frozen=True, init=False)
class LearnerCheckpoint:
    """Safe transport for full cooperative resumption state; never uses pickle."""
    adapter_id: str
    components: Mapping[str, bytes]
    optimizer_steps: int
    optimizer_step_counts: Mapping[str, int]
    _metadata_json: str = field(repr=False)

    def __init__(self, adapter_id: str, components: Mapping[str, bytes],
                 optimizer_steps: int, optimizer_step_counts: Mapping[str, int],
                 metadata: dict[str, Any]) -> None:
        _nonempty_string(adapter_id, "adapter_id")
        _integer(optimizer_steps, "optimizer_steps")
        if not isinstance(components, Mapping) or not isinstance(optimizer_step_counts, Mapping):
            raise TypeError("components and optimizer_step_counts must be mappings")
        copied_components = dict(components)
        missing = REQUIRED_CHECKPOINT_COMPONENTS - copied_components.keys()
        if missing:
            raise ValueError(f"missing checkpoint components: {sorted(missing)}")
        for name, blob in copied_components.items():
            _nonempty_string(name, "component name")
            if type(blob) is not bytes:
                raise TypeError(f"component {name} must be immutable bytes")
        copied_counts = dict(optimizer_step_counts)
        for name, count in copied_counts.items():
            _nonempty_string(name, "optimizer name")
            _integer(count, f"optimizer_step_counts[{name}]")
        if sum(copied_counts.values()) != optimizer_steps:
            raise ValueError("optimizer_step_counts must sum to optimizer_steps")
        metadata_json = _settings_json(metadata, "metadata")
        absent = json.loads(metadata_json).get("not_applicable_components", [])
        if type(absent) is not list or any(type(name) is not str for name in absent):
            raise TypeError("not_applicable_components must be a list of component names")
        if len(set(absent)) != len(absent) or not set(absent) <= copied_components.keys():
            raise ValueError("not_applicable_components must name distinct existing components")
        primary = json.loads(metadata_json).get("primary_optimizer")
        if primary is not None:
            _nonempty_string(primary, "metadata.primary_optimizer")
            if primary not in copied_counts:
                raise ValueError("metadata.primary_optimizer must name an optimizer_step_counts key")
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "components", MappingProxyType(copied_components))
        object.__setattr__(self, "optimizer_steps", optimizer_steps)
        object.__setattr__(self, "optimizer_step_counts", MappingProxyType(copied_counts))
        object.__setattr__(self, "_metadata_json", metadata_json)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a defensive deep copy, suitable for normal JSON serialization."""
        return json.loads(self._metadata_json)

    @property
    def primary_optimizer(self) -> str | None:
        return self.metadata.get("primary_optimizer")

    @property
    def primary_optimizer_steps(self) -> int | None:
        name = self.primary_optimizer
        return None if name is None else self.optimizer_step_counts[name]

    @property
    def component_hashes(self) -> dict[str, str]:
        return {name: hashlib.sha256(blob).hexdigest() for name, blob in self.components.items()}

    def _payload(self) -> dict[str, Any]:
        return {"adapter_id": self.adapter_id,
                "components": {name: base64.b64encode(blob).decode("ascii")
                               for name, blob in self.components.items()},
                "component_hashes": self.component_hashes,
                "optimizer_steps": self.optimizer_steps,
                "optimizer_step_counts": dict(self.optimizer_step_counts),
                "metadata": self.metadata}

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload()).encode("utf-8")).hexdigest()

    def to_bytes(self) -> bytes:
        return _canonical_json({"format": CHECKPOINT_FORMAT,
                                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                                "payload": self._payload(),
                                "sha256": self.digest}).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> LearnerCheckpoint:
        if type(data) is not bytes:
            raise TypeError("checkpoint data must be bytes")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        try:
            document = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
            if type(document) is not dict or set(document) != {"format", "schema_version", "payload", "sha256"}:
                raise ValueError("invalid checkpoint envelope")
            if document["format"] != CHECKPOINT_FORMAT or type(document["schema_version"]) is not int or document["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("unsupported checkpoint format or schema version")
            payload = document["payload"]
            if type(payload) is not dict or set(payload) != {"adapter_id", "components", "component_hashes", "optimizer_steps", "optimizer_step_counts", "metadata"}:
                raise ValueError("invalid checkpoint payload")
            expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
            if type(document["sha256"]) is not str or not hmac.compare_digest(expected, document["sha256"]):
                raise ValueError("checkpoint digest mismatch")
            if type(payload["components"]) is not dict or type(payload["component_hashes"]) is not dict:
                raise ValueError("invalid checkpoint component mappings")
            components = {}
            for name, encoded in payload["components"].items():
                if type(encoded) is not str:
                    raise ValueError("component encoding must be base64 text")
                components[name] = base64.b64decode(encoded, validate=True)
            checkpoint = cls(payload["adapter_id"], components, payload["optimizer_steps"],
                             payload["optimizer_step_counts"], payload["metadata"])
            if checkpoint.component_hashes != payload["component_hashes"]:
                raise ValueError("checkpoint component hash mismatch")
            return checkpoint
        except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid learner checkpoint: {error}") from error

    def save(self, path: str | Path) -> Path:
        """Create a checkpoint exclusively; an existing destination is an error."""
        from .datasets import forbid_during_evaluation
        forbid_during_evaluation("learner checkpoint persistence")
        destination = Path(path)
        with destination.open("xb") as handle:
            handle.write(self.to_bytes())
        return destination

    @classmethod
    def load(cls, path: str | Path) -> LearnerCheckpoint:
        return cls.from_bytes(Path(path).read_bytes())


@runtime_checkable
class FrozenAdapterPolicy(Protocol):
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Infer without learning, exploration, or mutable running statistics."""
        ...

    def state_bytes(self) -> bytes:
        """Include every inference, preprocessing, and action-transform state."""
        ...


@runtime_checkable
class LearnerAdapter(Protocol):
    @property
    def capabilities(self) -> AdapterCapabilities: ...

    @property
    def optimizer_steps(self) -> int:
        """Cheap read-only cumulative count; no checkpoint serialization."""
        ...

    @property
    def optimizer_step_counts(self) -> Mapping[str, int]:
        """Cheap defensive snapshot of the per-optimizer cumulative counts."""
        ...

    def initialize(self, train_batch: TransitionBatch, seed: int,
                   learning_settings: dict[str, Any]) -> None: ...

    def update(self, batch: TransitionBatch) -> dict[str, Any]:
        """Perform exactly one PRIMARY step; separately count scheduled auxiliary steps."""
        ...

    def act(self, observation: np.ndarray, explore: bool) -> np.ndarray: ...

    def checkpoint(self) -> LearnerCheckpoint: ...

    def restore(self, checkpoint: LearnerCheckpoint) -> None: ...

    def freeze(self) -> FrozenAdapterPolicy: ...


@runtime_checkable
class OnlineLearnerAdapter(LearnerAdapter, Protocol):
    """Required extension whenever capabilities.supports_online is True."""

    def configure_learning_settings(self, learning_settings: dict[str, Any]) -> None:
        """Change declared learning settings without resetting training state."""
        ...
