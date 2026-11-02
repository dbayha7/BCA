"""Dataset schema `thrusterchain.knee_edge_trajectories.v1` (protocol 4.3).

The core contract (`datasets.py` schema v2) rejects this design: it requires the recorded start family, a
state-independent box density, an initial current in [0.1, 0.7], a recorded-law lineage replay and a0 > 0.5
regions. This version adds load-dependent density semantics with the truncated CUTOFF density documented, the
knee_mix15 start family, a bit-for-bit lineage replay of the five-uniform recorder, and the world tag of 4.5b.
The old validator rejects data of this schema and this validator rejects the old schema (tested).
"""
from pathlib import Path
from types import MappingProxyType
import json

import numpy as np

import recorder
import worlds
from core import canonical_json, load_datasets, require, sha_file
from knee_region import KNEE, SOURCE_SHA256 as REGION_SHA256, load_band, load_g, past_knee_a0

SCHEMA = "thrusterchain.knee_edge_trajectories.v1"
SPLITS = ("train", "tuning_validation", "selection", "final_calibration")
HORIZON = 100
DTYPES = {
    "observations": "<f4", "states": "<f8", "actions": "<f4", "requested_actions": "<f8", "rewards": "<f8",
    "terminated": "|b1", "truncated": "|b1", "trajectory_ids": "<i8", "group_ids": "<i8", "episode_seeds": "<i8",
    "split_codes": "|u1", "reset_component": "|u1", "behavior_component": "|u1", "load_band": "|u1",
    "behavior_log_density": "<f8",
}
DENSITY_SEMANTICS = {
    "requested_action_measure": "continuous_lebesgue_density",
    "conditional_requested_action_density": "known_exact_load_dependent_mixture",
    "executed_action_measure": "discrete_float32_quantization_mass_not_evaluated",
    "state_occupancy": "unknown_closed_loop_occupancy",
    "joint_state_action_density": "unavailable", "transition_measure": "deterministic_dirac",
    "cutoff_truncated_density": "recorder density restricted to |a0_float32| <= KNEE and renormalized: by 0.506 "
                                "at g >= 0.4 (window mass 0.49 retained + floor 0.016) and by 0.800 at g < 0.4; zero on W",
}


class KneeDatasetError(ValueError):
    pass


def _check(condition, message):
    if not condition:
        raise KneeDatasetError(message)


def generator_metadata(entropy, episodes):
    return {"version": "knee_edge_sampler_v1", "bit_generator": "PCG64", "numpy_version": np.__version__,
            "seed_sequence": {"implementation": "numpy.random.SeedSequence", "entropy": int(entropy), "pool_size": 4},
            "named_streams": {name: {"spawn_key": [key] if name == "split" else [key, "episode"]}
                              for name, key in recorder.STREAM_KEYS.items()},
            "episodes": [{"trajectory_id": e, "reset_seed": recorder.reset_seed(entropy, e),
                          "reset_component": recorder.mixture_component(entropy, e)} for e in range(episodes)]}


def _metadata(entropy, episodes, world, counts, provenance):
    return {"schema": SCHEMA, "seed": int(entropy), "episodes": int(episodes), "splits": list(SPLITS),
            "split_counts": dict(zip(SPLITS, counts)), "observation_mode": "time_aware", "horizon": HORIZON,
            "gamma": .98, "task_semantics": "finite_horizon", "start_family": "knee_mix15",
            "reset_law": recorder.RESET_LAW, "behavior": recorder.BEHAVIOR, "density": DENSITY_SEMANTICS,
            "world": world, "physics": worlds.WORLD_DICTS[world], "provenance": provenance,
            "generator": generator_metadata(entropy, episodes),
            "region": {"module_sha256": REGION_SHA256, "knee": KNEE},
            "recorder_sha256": sha_file(recorder.__file__),
            "grouping": "one independent seeded episode per group; no row-level split"}


def generate(*, episodes, entropy, world, provenance):
    """Collect `episodes` episodes in one world (simulator use: approval A3 for the benchmark seeds)."""
    _check(type(episodes) is int and episodes >= 10, "At least ten episodes")
    _check(provenance in ("smoke", "research"), "Provenance must be smoke or research")
    env = worlds.environment(world)
    codes, counts = recorder.split_assignment(episodes, entropy)
    meta = generator_metadata(entropy, episodes)
    arrays = {name: np.zeros(shape, dtype) for name, (shape, dtype) in {
        "observations": ((episodes, 101, 14), "<f4"), "states": ((episodes, 101, 13), "<f8"),
        "actions": ((episodes, 100, 4), "<f4"), "requested_actions": ((episodes, 100, 4), "<f8"),
        "rewards": ((episodes, 100), "<f8"), "terminated": ((episodes, 100), bool),
        "truncated": ((episodes, 100), bool), "behavior_component": ((episodes, 100), "|u1"),
        "load_band": ((episodes, 100), "|u1"), "behavior_log_density": ((episodes, 100), "<f8")}.items()}
    arrays.update(trajectory_ids=np.arange(episodes, dtype="<i8"), group_ids=np.arange(episodes, dtype="<i8"),
                  episode_seeds=np.asarray([e["reset_seed"] for e in meta["episodes"]], "<i8"),
                  split_codes=codes, reset_component=np.asarray([e["reset_component"] for e in meta["episodes"]], "|u1"))
    try:
        for e in range(episodes):
            draws = recorder.episode_draws(entropy, e)
            obs, info = env.reset(seed=int(arrays["episode_seeds"][e]),
                                  options=recorder.reset_options(arrays["reset_component"][e]))
            arrays["observations"][e, 0], arrays["states"][e, 0] = obs, info["state"]
            for t in range(HORIZON):
                requested, component, logp = recorder.requested_command(obs, draws[t])
                action = requested.astype(np.float32)
                arrays["load_band"][e, t] = load_band(load_g(obs[None]))[0]
                obs, reward, terminated, truncated, info = env.step(action)
                arrays["observations"][e, t + 1], arrays["states"][e, t + 1] = obs, info["state"]
                arrays["actions"][e, t], arrays["requested_actions"][e, t] = action, requested
                arrays["rewards"][e, t], arrays["terminated"][e, t], arrays["truncated"][e, t] = reward, terminated, truncated
                arrays["behavior_component"][e, t], arrays["behavior_log_density"][e, t] = component, logp
    finally:
        env.close()
    return KneeDataset(arrays, _metadata(entropy, episodes, world, counts, provenance))


def _lineage(arrays, entropy, physics_world):
    """Bit-for-bit replay of every seeded draw: reset component, initial state, five uniforms per step, commands."""
    env = worlds.environment(physics_world)
    try:
        for e, seed in enumerate(arrays["episode_seeds"]):
            component = recorder.mixture_component(entropy, e)
            _check(int(arrays["reset_component"][e]) == component, f"Episode {e}: reset component lineage differs")
            obs, info = env.reset(seed=int(seed), options=recorder.reset_options(component))
            _check(np.array_equal(info["state"], arrays["states"][e, 0]), f"Episode {e}: initial state lineage differs")
            draws = recorder.episode_draws(entropy, e)
            for t in range(HORIZON):
                requested, component_t, logp = recorder.requested_command(arrays["observations"][e, t], draws[t])
                _check(np.array_equal(requested, arrays["requested_actions"][e, t])
                       and component_t == int(arrays["behavior_component"][e, t])
                       and logp == float(arrays["behavior_log_density"][e, t]),
                       f"Episode {e} step {t}: recorder lineage differs (command, component or log density)")
    finally:
        env.close()


def validate(arrays, metadata, *, check_dynamics=True):
    _check(isinstance(metadata, dict) and metadata.get("schema") == SCHEMA, "Unknown dataset schema")
    _check(set(arrays) == set(DTYPES), "Array schema mismatch")
    entropy = metadata.get("seed")
    _check(type(entropy) is int and entropy >= 0, "Invalid entropy")
    world = metadata.get("world")
    _check(world in ("stall3", "nostall"), "Collection worlds are stall3 and nostall")
    _check(metadata.get("physics") == worlds.WORLD_DICTS[world], "Physics dict differs from the world tag")
    n = len(arrays["trajectory_ids"])
    _check(n >= 10, "At least ten episodes")
    codes_expected, counts = recorder.split_assignment(n, entropy)
    expected_meta = _metadata(entropy, n, world, counts, metadata.get("provenance"))
    expected_meta["generator"]["numpy_version"] = metadata.get("generator", {}).get("numpy_version")
    _check(metadata.get("provenance") in ("smoke", "research"), "Provenance must be smoke or research")
    sealed_entropy = _sealed_dataset_entropy()
    if metadata["provenance"] == "research":
        _check(entropy == sealed_entropy and n == 1280, "Research data must use the sealed dataset entropy and 1,280 episodes")
    else:
        _check(entropy != sealed_entropy, "Smoke data must never use the sealed dataset entropy")
    _check(canonical_json(metadata) == canonical_json(expected_meta), "Metadata differs from the knee_edge_v1 contract")
    shapes = {"observations": (n, 101, 14), "states": (n, 101, 13), "actions": (n, 100, 4),
              "requested_actions": (n, 100, 4), "rewards": (n, 100), "terminated": (n, 100), "truncated": (n, 100),
              "trajectory_ids": (n,), "group_ids": (n,), "episode_seeds": (n,), "split_codes": (n,),
              "reset_component": (n,), "behavior_component": (n, 100), "load_band": (n, 100),
              "behavior_log_density": (n, 100)}
    for name, dtype in DTYPES.items():
        value = np.asarray(arrays[name])
        _check(value.shape == shapes[name] and value.dtype.str == dtype, f"{name}: shape or dtype")
        _check(bool(np.all(np.isfinite(value))) if value.dtype.kind == "f" else True, f"{name}: nonfinite")
    _check(np.array_equal(arrays["trajectory_ids"], np.arange(n)) and np.array_equal(arrays["group_ids"], np.arange(n)),
           "Trajectory and group ids must be the ordered stream addresses")
    _check(np.array_equal(arrays["split_codes"], codes_expected), "Split codes differ from the split stream")
    _check(np.array_equal(arrays["episode_seeds"], [e["reset_seed"] for e in expected_meta["generator"]["episodes"]]),
           "Reset seeds differ from the named streams")
    obs, states = arrays["observations"], arrays["states"]
    _check(np.array_equal(obs[..., :13], states.astype(np.float32)), "Observations are not float32 states")
    _check(np.array_equal(obs[..., 13], np.broadcast_to(((100 - np.arange(101)) / 100).astype(np.float32), (n, 101))),
           "Remaining-time coordinate differs")
    current, amplitude = states[:, 0, 10], np.hypot(states[:, 0, 11], states[:, 0, 12])
    deep = arrays["reset_component"] == 1
    _check(bool(np.all(np.where(deep, (current >= .9) & (current <= 1.1), (current >= .1) & (current <= .7)))),
           "Initial current outside its mixture component's range")
    _check(bool(np.all((amplitude >= .1 - 1e-14) & (amplitude <= .4 + 1e-14))), "Initial wave amplitude outside [0.1, 0.4]")
    final = np.zeros((n, 100), bool)
    final[:, -1] = True
    _check(np.array_equal(arrays["terminated"], final) and not np.any(arrays["truncated"]),
           "Finite horizon: done only at t = 99, never truncated")
    requests = arrays["requested_actions"]
    _check(bool(np.all(np.abs(requests[..., 0]) <= 1)) and bool(np.all(np.abs(requests[..., 1:]) <= recorder.SIDE_RANGE)),
           "Requested command outside the recorder support")
    _check(np.array_equal(arrays["actions"], requests.astype(np.float32)), "Executed is not float32(requested)")
    _check(bool(np.all(np.abs(arrays["actions"][..., 1:].astype(np.float64)) <= recorder.SIDE_RANGE)),
           "Executed a1..a3 past 0.7")
    bands = load_band(load_g(obs[:, :-1]))
    _check(np.array_equal(arrays["load_band"], bands), "Load bands differ from the observations")
    _check(not np.any((arrays["behavior_component"] == 1) & (bands == 0)), "Window component at low load")
    expected = recorder.log_density(requests, bands)
    _check(bool(np.all(np.isfinite(expected))) and bool(np.allclose(arrays["behavior_log_density"], expected,
                                                                     rtol=0, atol=1e-14)),
           "Logged density differs from the closed form")
    past_knee_a0(arrays["actions"])   # raises on any invalid executed command
    _lineage(arrays, entropy, world)
    if check_dynamics:
        env = worlds.load_environment()
        nxt, reward = env.dynamics(states[:, :-1], arrays["actions"], worlds.physics(world))
        _check(np.array_equal(nxt, states[:, 1:]) and np.array_equal(reward, arrays["rewards"]),
               "Rows do not replay bit for bit under the world's physics")


def _sealed_dataset_entropy():
    import seeds
    return seeds.load()["numpy"]["dataset_entropy"]


class KneeDataset:
    __slots__ = ("_arrays", "_metadata_json", "_digest")

    def __init__(self, arrays, metadata, *, expected_digest=None):
        datasets = load_datasets()
        copied = {name: datasets.immutable_array(value) for name, value in arrays.items()}
        clean = json.loads(canonical_json(metadata))
        validate(copied, clean)
        digest = datasets.content_digest(copied, clean)
        _check(expected_digest is None or digest == expected_digest, "Dataset SHA-256 mismatch")
        object.__setattr__(self, "_arrays", MappingProxyType(copied))
        object.__setattr__(self, "_metadata_json", canonical_json(clean))
        object.__setattr__(self, "_digest", digest)

    def __setattr__(self, name, value):
        raise AttributeError("KneeDataset is immutable")

    arrays = property(lambda self: self._arrays)
    metadata = property(lambda self: json.loads(self._metadata_json))
    digest = property(lambda self: self._digest)


def save(dataset, path):
    _check(type(dataset) is KneeDataset, "save requires a validated KneeDataset")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **dataset.arrays,
                            manifest=np.asarray(canonical_json({"metadata": dataset.metadata, "sha256": dataset.digest})))
    return path


def load(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        _check(set(archive.files) == set(DTYPES) | {"manifest"}, "Unexpected NPZ members (not a knee_edge_v1 dataset)")
        envelope = json.loads(str(archive["manifest"]))
        arrays = {name: archive[name] for name in DTYPES}
    _check(set(envelope) == {"metadata", "sha256"}, "Invalid manifest envelope")
    return KneeDataset(arrays, envelope["metadata"], expected_digest=envelope["sha256"])
