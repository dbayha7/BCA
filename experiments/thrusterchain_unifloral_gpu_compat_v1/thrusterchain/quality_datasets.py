"""Versioned grouped datasets from random and feedback behavior policies.

No behavior density or occupancy ratio is exposed by this schema. The feedback
controller uses the public plant matrices and public float32 observation. Data
generation is fixed-policy collection, separate from learner optimization.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import scipy
from scipy import linalg

from .datasets import (DatasetError, HORIZON, SPLITS, _NUMPY_VERSION, _episode_reset_seed,
                       _source_model_metadata, _split_assignment, _stream_seed_sequence,
                       canonical_json, content_digest, forbid_during_evaluation, immutable_array)
from .environment import Physics, START_LAWS, ThrusterChainEnv, draw_starts, dynamics, matrices

SCHEMA = "thrusterchain.behavior_trajectories.v1"
BEHAVIORS = ("random", "feedback_weak", "feedback_strong", "mixed_random_strong")
COMPONENTS = ("random", "feedback_weak", "feedback_strong")
DTYPES = {"observations": "<f4", "states": "<f8", "actions": "<f4",
          "requested_actions": "<f8", "rewards": "<f8", "terminated": "|b1",
          "truncated": "|b1", "trajectory_ids": "<i8", "group_ids": "<i8",
          "episode_seeds": "<i8", "split_codes": "|u1", "behavior_component": "|u1"}


def _file_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def _feedback_gain(stall_slope):
    # Public rising-thrust relaxation, a hand-designed behavior source. Its
    # realized quality is measured by returns, not assumed from its name.
    physics = Physics(stall=True, stall_slope=stall_slope)
    a, b = matrices(physics)
    q = np.diag([1.] * 10 + [0.] * 3)
    p = linalg.solve_discrete_are(np.sqrt(physics.gamma) * a, np.sqrt(physics.gamma) * b,
                                  q, physics.throttle_cost * np.eye(4))
    k = np.linalg.solve(physics.throttle_cost * np.eye(4) + physics.gamma * b.T @ p @ b,
                        physics.gamma * b.T @ p @ a)
    if k.shape != (4, 13) or not np.all(np.isfinite(k)):
        raise DatasetError("feedback gain is invalid")
    return immutable_array(k, "<f8")


def _components(kind, seed, episodes):
    if kind == "random": return np.zeros(episodes, dtype="|u1")
    if kind == "feedback_weak": return np.ones(episodes, dtype="|u1")
    if kind == "feedback_strong": return np.full(episodes, 2, dtype="|u1")
    if kind != "mixed_random_strong": raise DatasetError("unknown behavior dataset")
    # Exact allocation is deterministic and per-episode; no selection by returns.
    labels = np.r_[np.zeros(episodes // 2, dtype="|u1"),
                   np.full(episodes - episodes // 2, 2, dtype="|u1")]
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2704,)))
    return labels[rng.permutation(episodes)]


def _request(observation, code, rng, gain):
    if code == 0:
        return rng.uniform(-1., 1., size=4)
    if code not in (1, 2):
        raise DatasetError("unknown behavior component")
    multiplier, noise_sd = ((.25, .30) if code == 1 else (1., .10))
    state = np.asarray(observation[:13], np.float64)
    demand = -multiplier * (gain @ state) + noise_sd * rng.standard_normal(4)
    return .8 * np.tanh(demand / .8)


def _expected_metadata(kind, seed, n, physics, semantics, provenance):
    mode = "time_aware" if semantics == "finite_horizon" else "physical_padded"
    _, counts = _split_assignment(n, seed)
    gain = _feedback_gain(physics["stall_slope"])
    return {"schema": SCHEMA, "seed": seed, "episodes": n, "horizon": HORIZON,
            "gamma": .98, "splits": list(SPLITS), "split_counts": dict(zip(SPLITS, counts)),
            "task_semantics": semantics, "observation_mode": mode, "start_family": "recorded",
            "physics": physics, "provenance": provenance,
            "behavior": {"name": kind, "components": list(COMPONENTS),
                         "feedback_weak": {"gain_multiplier": .25, "demand_noise_sd": .30},
                         "feedback_strong": {"gain_multiplier": 1., "demand_noise_sd": .10},
                         "feedback_cap": .8, "feedback_observation": "public_float32_physical_coordinates",
                         "feedback_gain": "discounted_lqr_of_public_rising_thrust_relaxation",
                         "gain_sha256": hashlib.sha256(np.asarray(gain, "<f8").tobytes()).hexdigest(),
                         "mixture": "episode-level seeded balanced random/strong; no return selection"},
            "randomness": {"bit_generator": "PCG64", "seed_sequence_entropy": seed,
                           "reset_stream": "v3 named stream 1701 per trajectory",
                           "action_stream": "v3 named stream 1703 per trajectory",
                           "split_stream": "v3 named stream 1702",
                           "mixture_stream_spawn_key": [2704],
                           "numpy_version": np.__version__, "scipy_version": scipy.__version__},
            "density": {"conditional_action_density": "not_provided",
                        "executed_action_mass": "not_provided", "state_occupancy": "unknown",
                        "joint_state_action_density": "not_provided"},
            "source_model_metadata": _source_model_metadata(physics["stall"], physics["stall_slope"], semantics),
            "collector_sha256": _file_sha(),
            "grouping": "one independently seeded complete episode per group; no row-level split"}


def _validate(arrays, metadata):
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise DatasetError("unknown behavior dataset schema")
    if set(arrays) != set(DTYPES): raise DatasetError("array schema mismatch")
    seed, n = metadata.get("seed"), metadata.get("episodes")
    if type(seed) is not int or seed < 0 or type(n) is not int or n < 10:
        raise DatasetError("invalid episode count or seed")
    physics, semantics, provenance = metadata.get("physics"), metadata.get("task_semantics"), metadata.get("provenance")
    if (not isinstance(physics, dict) or set(physics) != {"stall", "stall_slope"}
            or type(physics["stall"]) is not bool or type(physics["stall_slope"]) is not float
            or physics["stall_slope"] not in (1.5, 3.0)):
        raise DatasetError("invalid physics metadata")
    if semantics not in ("finite_horizon", "continuing_time_limit") or provenance not in ("smoke", "research"):
        raise DatasetError("invalid task semantics or provenance")
    kind = metadata.get("behavior", {}).get("name")
    if kind not in BEHAVIORS: raise DatasetError("unknown behavior metadata")
    expected_meta = _expected_metadata(kind, seed, n, physics, semantics, provenance)
    # Version is provenance, not a demand that the load runtime match it.
    recorded_version = metadata.get("randomness", {}).get("numpy_version")
    if not isinstance(recorded_version, str) or _NUMPY_VERSION.fullmatch(recorded_version) is None:
        raise DatasetError("missing NumPy version provenance")
    expected_meta["randomness"]["numpy_version"] = recorded_version
    recorded_scipy = metadata.get("randomness", {}).get("scipy_version")
    if not isinstance(recorded_scipy, str) or _NUMPY_VERSION.fullmatch(recorded_scipy) is None:
        raise DatasetError("missing SciPy version provenance")
    expected_meta["randomness"]["scipy_version"] = recorded_scipy
    if canonical_json(metadata) != canonical_json(expected_meta):
        raise DatasetError("behavior, collector, model or random-stream metadata mismatch")
    shapes = {"observations": (n, 101, 14), "states": (n, 101, 13),
              "actions": (n, 100, 4), "requested_actions": (n, 100, 4),
              "rewards": (n, 100), "terminated": (n, 100), "truncated": (n, 100),
              "trajectory_ids": (n,), "group_ids": (n,), "episode_seeds": (n,),
              "split_codes": (n,), "behavior_component": (n,)}
    for name, dtype in DTYPES.items():
        a = np.asarray(arrays[name])
        if a.dtype.str != dtype or a.shape != shapes[name] or not np.all(np.isfinite(a)):
            raise DatasetError(f"invalid shape, dtype or values: {name}")
    ids = np.arange(n, dtype=np.int64)
    if not np.array_equal(arrays["trajectory_ids"], ids) or not np.array_equal(arrays["group_ids"], ids):
        raise DatasetError("each ordered trajectory must have its own group")
    expected_seeds = np.asarray([_episode_reset_seed(seed, i) for i in range(n)], dtype=np.int64)
    if not np.array_equal(arrays["episode_seeds"], expected_seeds) or len(np.unique(expected_seeds)) != n:
        raise DatasetError("episode seed linkage or uniqueness failure")
    expected_splits, _ = _split_assignment(n, seed)
    if not np.array_equal(arrays["split_codes"], expected_splits):
        raise DatasetError("trajectory split assignment does not match seeded groups")
    if not np.array_equal(arrays["behavior_component"], _components(kind, seed, n)):
        raise DatasetError("behavior component lineage mismatch")
    if not np.array_equal(arrays["observations"][..., :13], arrays["states"].astype(np.float32)):
        raise DatasetError("observations do not match saved states")
    times = (((100 - np.arange(101)) / 100).astype(np.float32)
             if semantics == "finite_horizon" else np.zeros(101, np.float32))
    if not np.array_equal(arrays["observations"][..., 13], np.broadcast_to(times, (n, 101))):
        raise DatasetError("time coordinate inconsistent with task")
    final = np.zeros((n, 100), bool); final[:, -1] = True
    if not np.array_equal(arrays["terminated"], final if semantics == "finite_horizon" else np.zeros_like(final)):
        raise DatasetError("termination flags inconsistent")
    if not np.array_equal(arrays["truncated"], final if semantics == "continuing_time_limit" else np.zeros_like(final)):
        raise DatasetError("truncation flags inconsistent")
    requests, actions = arrays["requested_actions"], arrays["actions"]
    if np.any(np.abs(requests) > 1.) or not np.array_equal(actions, requests.astype(np.float32)):
        raise DatasetError("executed action mapping or action bounds invalid")
    gain = _feedback_gain(physics["stall_slope"])
    for episode, reset_seed in enumerate(expected_seeds):
        initial = draw_starts(1, np.random.Generator(np.random.PCG64(int(reset_seed))), START_LAWS["recorded"])[0]
        if not np.allclose(arrays["states"][episode, 0], initial, rtol=1e-13, atol=1e-13):
            raise DatasetError("initial state does not match episode seed")
        rng = np.random.Generator(np.random.PCG64(_stream_seed_sequence(seed, "actions", episode)))
        for step in range(HORIZON):
            expected = _request(arrays["observations"][episode, step], int(arrays["behavior_component"][episode]), rng, gain)
            if not np.allclose(requests[episode, step], expected, rtol=1e-13, atol=1e-13):
                raise DatasetError(f"behavior action lineage mismatch at episode {episode}, step {step}")
    predicted, reward = dynamics(arrays["states"][:, :-1], actions, Physics(**physics))
    if (not np.allclose(predicted, arrays["states"][:, 1:], rtol=1e-13, atol=1e-13)
            or not np.allclose(reward, arrays["rewards"], rtol=1e-13, atol=1e-13)):
        raise DatasetError("trajectory replay or reward mismatch")


class BehaviorDataset:
    """Immutable, validated full episodes with distinct data contract."""
    __slots__ = ("_arrays", "_metadata", "_digest")

    def __init__(self, arrays, metadata, *, expected_digest=None):
        copied = {name: immutable_array(a) for name, a in arrays.items()}
        clean = json.loads(canonical_json(metadata))
        _validate(copied, clean)
        digest = content_digest(copied, clean)
        if expected_digest is not None and digest != expected_digest:
            raise DatasetError("behavior dataset digest mismatch")
        object.__setattr__(self, "_arrays", MappingProxyType(copied))
        object.__setattr__(self, "_metadata", canonical_json(clean))
        object.__setattr__(self, "_digest", digest)

    def __setattr__(self, *_): raise AttributeError("BehaviorDataset is immutable")
    @property
    def arrays(self): return self._arrays
    @property
    def metadata(self): return json.loads(self._metadata)
    @property
    def digest(self): return self._digest
    def split(self, name):
        if name not in SPLITS: raise DatasetError("unknown split")
        mask = self._arrays["split_codes"] == SPLITS.index(name)
        return MappingProxyType({k: immutable_array(v[mask]) for k, v in self._arrays.items()})
    def split_digest(self, name):
        return content_digest(self.split(name), {"dataset_sha256": self.digest, "split": name})


def save_behavior_dataset(dataset, path):
    forbid_during_evaluation("save behavior dataset")
    if type(dataset) is not BehaviorDataset: raise DatasetError("validated BehaviorDataset required")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"metadata": dataset.metadata, "sha256": dataset.digest}
    with path.open("xb") as stream:
        np.savez_compressed(stream, **dataset.arrays, manifest=np.asarray(canonical_json(envelope)))
    return path


def load_behavior_dataset(path):
    try:
        with np.load(Path(path), allow_pickle=False) as a:
            if set(a.files) != set(DTYPES) | {"manifest"}: raise DatasetError("unexpected NPZ members")
            manifest = a["manifest"]
            if manifest.shape != () or manifest.dtype.kind != "U": raise DatasetError("invalid manifest")
            envelope = json.loads(str(manifest))
            arrays = {name: a[name] for name in DTYPES}
        if set(envelope) != {"metadata", "sha256"}: raise DatasetError("invalid envelope")
        return BehaviorDataset(arrays, envelope["metadata"], expected_digest=envelope["sha256"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot load behavior dataset: {exc}") from exc


def generate_behavior_dataset(path=None, *, behavior="mixed_random_strong", episodes=40, seed=20261027,
                              stall=True, stall_slope=1.5, task_semantics="finite_horizon", provenance="smoke"):
    forbid_during_evaluation("collect behavior dataset")
    if isinstance(episodes, bool) or not isinstance(episodes, (int, np.integer)) or episodes < 10:
        raise DatasetError("at least ten episodes required")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise DatasetError("seed must be a nonnegative integer")
    if behavior not in BEHAVIORS or task_semantics not in ("finite_horizon", "continuing_time_limit") or provenance not in ("smoke", "research"):
        raise DatasetError("unknown behavior, task semantics or provenance")
    if type(stall) is not bool or isinstance(stall_slope, bool) or stall_slope not in (1.5, 3.0):
        raise DatasetError("invalid physics")
    n, seed = int(episodes), int(seed)
    physics = {"stall": stall, "stall_slope": float(stall_slope)}
    meta = _expected_metadata(behavior, seed, n, physics, task_semantics, provenance)
    codes, _ = _split_assignment(n, seed)
    arrays = {"observations": np.empty((n, 101, 14), "<f4"), "states": np.empty((n, 101, 13), "<f8"),
              "actions": np.empty((n, 100, 4), "<f4"), "requested_actions": np.empty((n, 100, 4), "<f8"),
              "rewards": np.empty((n, 100), "<f8"), "terminated": np.zeros((n, 100), bool),
              "truncated": np.zeros((n, 100), bool), "trajectory_ids": np.arange(n, dtype="<i8"),
              "group_ids": np.arange(n, dtype="<i8"),
              "episode_seeds": np.asarray([_episode_reset_seed(seed, i) for i in range(n)], dtype="<i8"),
              "split_codes": codes, "behavior_component": _components(behavior, seed, n)}
    gain = _feedback_gain(float(stall_slope))
    env = ThrusterChainEnv(stall=stall, stall_slope=float(stall_slope), start="recorded",
                           observation_mode=meta["observation_mode"], task_semantics=task_semantics)
    try:
        for episode, reset_seed in enumerate(arrays["episode_seeds"]):
            rng = np.random.Generator(np.random.PCG64(_stream_seed_sequence(seed, "actions", episode)))
            obs, info = env.reset(seed=int(reset_seed))
            arrays["observations"][episode, 0], arrays["states"][episode, 0] = obs, info["state"]
            for step in range(HORIZON):
                request = _request(obs, int(arrays["behavior_component"][episode]), rng, gain)
                obs, reward, terminated, truncated, info = env.step(request)
                arrays["observations"][episode, step + 1], arrays["states"][episode, step + 1] = obs, info["state"]
                arrays["requested_actions"][episode, step], arrays["actions"][episode, step] = request, info["command"]
                arrays["rewards"][episode, step] = reward
                arrays["terminated"][episode, step], arrays["truncated"][episode, step] = terminated, truncated
    finally:
        env.close()
    dataset = BehaviorDataset(arrays, meta)
    if path is not None: save_behavior_dataset(dataset, path)
    return dataset
