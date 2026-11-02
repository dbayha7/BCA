"""Fixed, grouped offline trajectories with explicit probability measures.

The recorder has a known continuous requested-action density. Float32 execution
quantizes that law; its mass function and the state occupancy are not supplied.
"""
from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

SCHEMA = "thrusterchain.continuous_trajectories.v3"
SPLITS = ("train", "tuning_validation", "selection", "final_calibration")
HORIZON = 100
STREAM_KEYS = MappingProxyType({"reset": 1701, "split": 1702, "actions": 1703})
DTYPES = {
    "observations": "<f4", "states": "<f8", "actions": "<f4",
    "requested_actions": "<f8",
    "rewards": "<f8", "terminated": "|b1", "truncated": "|b1",
    "trajectory_ids": "<i8", "group_ids": "<i8", "episode_seeds": "<i8",
    "split_codes": "|u1", "behavior_component": "|u1",
    "action_region": "|u1", "behavior_log_density": "<f8",
}
REGIONS = {"0": {"condition": "a0 <= 0.5", "volume": 12.0},
           "1": {"condition": "a0 > 0.5", "volume": 4.0}}
DENSITY_SEMANTICS = {
    "requested_action_measure": "continuous_lebesgue_density",
    "conditional_requested_action_density": "known_exact_state_independent_box_mixture",
    "executed_action_measure": "discrete_float32_quantization_mass_not_evaluated",
    "state_occupancy": "unknown_closed_loop_occupancy",
    "joint_state_action_density": "unavailable", "transition_measure": "deterministic_dirac",
}
_ACTIVE_EVALUATION = ContextVar("thrusterchain_frozen_evaluation", default=None)
# NumPy release, prerelease, postrelease, development and local-build provenance.
# Full matching rejects malformed suffixes; no runtime-version equality is implied.
_NUMPY_VERSION = re.compile(
    r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?"
)


def forbid_during_evaluation(operation):
    """Instrument and refuse built-in data collection, fitting and writing APIs."""
    ledger = _ACTIVE_EVALUATION.get()
    if ledger is not None:
        ledger["forbidden_api_calls"] += 1
        raise DatasetError(f"{operation} is disabled during frozen evaluation")


# Public name for fitting APIs in other modules. The check lives inside every
# mutating function, so a callable alias imported before evaluation cannot evade it.
assert_training_allowed = forbid_during_evaluation


@contextmanager
def inference_evaluation_guard(ledger=None):
    """Share a context-local refusal guard with other built-in evaluators.

    Yields a ledger with observed forbidden_api_calls. Nested guards reuse an
    existing ledger when no separate ledger is supplied. This guards package
    entrypoints, not arbitrary external code or operating-system file writes.
    """
    existing = _ACTIVE_EVALUATION.get()
    if existing is not None:
        if ledger is not None and ledger is not existing:
            raise DatasetError("nested evaluation must use its existing audit ledger")
        yield existing
        return
    ledger = {"forbidden_api_calls": 0} if ledger is None else ledger
    if not isinstance(ledger, dict) or ledger.get("forbidden_api_calls") != 0:
        raise DatasetError("evaluation guard needs a fresh audit ledger")
    token = _ACTIVE_EVALUATION.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE_EVALUATION.reset(token)


class DatasetError(ValueError):
    """The artifact does not satisfy the dataset contract."""


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_digest(arrays: Mapping[str, np.ndarray], metadata: dict) -> str:
    """SHA-256 of canonical content, independent of NPZ ZIP timestamps."""
    digest = hashlib.sha256(canonical_json(metadata).encode("utf8"))
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise DatasetError("object arrays are forbidden")
        header = canonical_json([name, value.dtype.str, list(value.shape)])
        digest.update(len(header.encode("utf8")).to_bytes(8, "big"))
        digest.update(header.encode("utf8"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def immutable_array(value, dtype=None):
    """Bytes-backed arrays cannot have WRITEABLE re-enabled by a caller."""
    value = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _stream_seed_sequence(seed, name, episode=None):
    """Address named streams directly so consuming one never advances another."""
    if name not in STREAM_KEYS:
        raise DatasetError(f"unknown random stream {name!r}")
    if name == "split":
        if episode is not None:
            raise DatasetError("the split stream has no episode child")
        key = (STREAM_KEYS[name],)
    else:
        if isinstance(episode, bool) or not isinstance(episode, (int, np.integer)) or episode < 0:
            raise DatasetError("reset and action streams need a nonnegative episode index")
        key = (STREAM_KEYS[name], int(episode))
    return np.random.SeedSequence(int(seed), spawn_key=key, pool_size=4)


def _episode_reset_seed(seed, episode):
    # A portable nonnegative int64 scalar accepted by Gymnasium's reset API.
    word = _stream_seed_sequence(seed, "reset", episode).generate_state(1, dtype=np.uint64)[0]
    return int(word) & ((1 << 63) - 1)


def _split_assignment(episodes, seed):
    n_tune, n_select, n_cal = max(1, episodes // 5), max(1, episodes // 10), max(1, episodes // 10)
    counts = [episodes - n_tune - n_select - n_cal, n_tune, n_select, n_cal]
    codes = np.repeat(np.arange(4, dtype=np.uint8), counts)
    split_rng = np.random.Generator(np.random.PCG64(_stream_seed_sequence(seed, "split")))
    return codes[split_rng.permutation(episodes)], counts


def _generator_metadata(seed, episodes):
    """Complete seed hierarchy, including every episode's address and reset seed."""
    return {
        "version": "continuous_box_sampler_v2", "bit_generator": "PCG64", "numpy_version": np.__version__,
        "seed_sequence": {"implementation": "numpy.random.SeedSequence", "entropy": int(seed),
                          "root_spawn_key": [], "pool_size": 4},
        "named_streams": {
            "reset": {"spawn_key_prefix": [STREAM_KEYS["reset"]], "child": "trajectory_id",
                      "reset_seed_derivation": "generate_state(1, uint64)[0] bitwise-and (2**63 - 1)"},
            "split": {"spawn_key": [STREAM_KEYS["split"]], "operation": "permutation of complete episode groups"},
            "actions": {"spawn_key_prefix": [STREAM_KEYS["actions"]], "child": "trajectory_id",
                        "operation": "fresh PCG64 generator per episode; bounded mixture draws"},
        },
        "episodes": [{"trajectory_id": episode,
                      "reset_spawn_key": [STREAM_KEYS["reset"], episode],
                      "reset_seed": _episode_reset_seed(seed, episode),
                      "action_spawn_key": [STREAM_KEYS["actions"], episode]}
                     for episode in range(episodes)],
        "extension_contract": "existing episode resets and action sequences do not depend on total episode count; split membership may change",
    }


def _source_model_metadata(stall, stall_slope, task_semantics):
    """Bind data to hash-verified plant sources, adapter code and full task config."""
    from .canonical import provenance
    from .environment import Physics, canonical_config
    record = provenance()
    mode = "time_aware" if task_semantics == "finite_horizon" else "physical_padded"
    config = {**canonical_config(), "physics": Physics(stall=stall, stall_slope=stall_slope).as_dict(),
              "start": "recorded", "task_semantics": task_semantics,
              "observation_mode": mode,
              "terminal_bootstrap": task_semantics == "continuing_time_limit"}
    return {
        "canonical_provenance": record,
        "canonical_provenance_sha256": hashlib.sha256(canonical_json(record).encode("utf8")).hexdigest(),
        "adapter_source_sha256": hashlib.sha256(Path(__file__).with_name("environment.py").read_bytes()).hexdigest(),
        "canonical_loader_sha256": hashlib.sha256((Path(__file__).parent / "canonical" / "__init__.py").read_bytes()).hexdigest(),
        "environment_config": config,
    }


def _check_epsilon(epsilon):
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, np.integer, np.floating)) or not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise DatasetError("epsilon must be finite and between zero and one")
    return float(epsilon)


def behavior_log_density(requested_actions, epsilon=.25):
    """Analytic density for pre-quantization commands, not executed-action mass.

    Region a0>.5 has volume 4 and probability epsilon; its complement has
    volume 12 and probability 1-epsilon. Zero support returns negative infinity.
    The function is independent of state and horizon.
    """
    epsilon = _check_epsilon(epsilon)
    raw = np.asarray(requested_actions)
    if raw.dtype.kind not in "fiu":
        raise DatasetError("requested commands must be real numeric values, not bool, complex or strings")
    actions = raw.astype(np.float64)
    if actions.shape[-1:] != (4,) or not np.all(np.isfinite(actions)):
        raise DatasetError("density requires finite (..., 4) requested commands")
    in_cube = np.all(np.abs(actions) <= 1., axis=-1)
    with np.errstate(divide="ignore"):
        rare_log = np.log(epsilon) - np.log(4.)
        common_log = np.log1p(-epsilon) - np.log(12.)
    log_density = np.where(actions[..., 0] > .5, rare_log, common_log)
    return np.where(in_cube, log_density, -np.inf)


def sample_requested_action(rng, epsilon=.25):
    """Directly sample a bounded box; no clipped Gaussian or feedback controller."""
    epsilon = _check_epsilon(epsilon)
    component = int(rng.random() < epsilon)
    action = rng.uniform(-1., 1., size=4)
    if component:
        # Exclude a numerically sampled lower endpoint of the open rare region.
        action[0] = max(float(rng.uniform(.5, 1.)), np.nextafter(.5, 1.))
    else:
        action[0] = rng.uniform(-1., .5)
    return action, component, float(behavior_log_density(action, epsilon))


def _validate_collector_lineage(arrays, seed, epsilon):
    """Replay this schema's seeded draws without collecting simulator trajectories.

    Distinct IDs and seeds alone do not prove that independent episodes were
    recorded. The declared collector also fixes each initial state and requested
    action sequence. Tolerance permits float64 arithmetic differences across
    platforms; the artifact's exact bytes remain covered by its content digest.
    """
    from .environment import START_LAWS, draw_starts
    for episode, reset_seed in enumerate(arrays["episode_seeds"]):
        reset_rng = np.random.Generator(np.random.PCG64(int(reset_seed)))
        expected_start = draw_starts(1, reset_rng, START_LAWS["recorded"])[0]
        if not np.allclose(arrays["states"][episode, 0], expected_start, rtol=1e-13, atol=1e-13):
            raise DatasetError(f"trajectory {episode} disagrees with seeded collector reset lineage")
        action_rng = np.random.Generator(np.random.PCG64(_stream_seed_sequence(seed, "actions", episode)))
        expected_requests = np.asarray([sample_requested_action(action_rng, epsilon)[0]
                                        for _ in range(HORIZON)])
        if not np.allclose(arrays["requested_actions"][episode], expected_requests, rtol=1e-13, atol=1e-13):
            raise DatasetError(f"trajectory {episode} disagrees with seeded collector action lineage")


def _validate(arrays, metadata, *, check_dynamics=True):
    if not isinstance(metadata, dict):
        raise DatasetError("manifest metadata must be a JSON object")
    if set(arrays) != set(DTYPES):
        raise DatasetError(f"array schema mismatch: expected {sorted(DTYPES)}")
    if metadata.get("schema") != SCHEMA:
        raise DatasetError("unknown dataset schema")
    if isinstance(metadata.get("seed"), bool) or not isinstance(metadata.get("seed"), int) or metadata["seed"] < 0:
        raise DatasetError("invalid generator seed")
    if metadata.get("horizon") != HORIZON:
        raise DatasetError("Regular mode requires horizon 100")
    if metadata.get("start_family") != "recorded":
        raise DatasetError("offline data must use recorded conditions; shifted starts are excluded from collection")
    if metadata.get("provenance") not in ("smoke", "research"):
        raise DatasetError("provenance must explicitly be smoke or research")
    semantics = metadata.get("task_semantics")
    if semantics not in ("finite_horizon", "continuing_time_limit"):
        raise DatasetError("unknown task semantics")
    mode = "time_aware" if semantics == "finite_horizon" else "physical_padded"
    if metadata.get("observation_mode") != mode:
        raise DatasetError("observation mode disagrees with task semantics")
    if metadata.get("splits") != list(SPLITS) or metadata.get("gamma") != .98:
        raise DatasetError("split definitions or discount differ from the benchmark")
    if metadata.get("density") != DENSITY_SEMANTICS:
        raise DatasetError("density semantics must distinguish requested density, execution quantization and unknown occupancy")
    epsilon = _check_epsilon(metadata.get("behavior", {}).get("epsilon"))
    if metadata.get("behavior") != {
        "name": "continuous_region_mixture_v1", "epsilon": epsilon,
        "regions": REGIONS, "state_independent": True,
        "execution_transform": "float64 request -> float32 command; already bounded so clipping is inactive",
    }:
        raise DatasetError("unknown recorder density specification")
    physics = metadata.get("physics", {})
    if not isinstance(physics, dict) or set(physics) != {"stall", "stall_slope"} or not isinstance(physics.get("stall"), bool):
        raise DatasetError("physics.stall must be boolean")
    slope = physics.get("stall_slope")
    if isinstance(slope, bool) or not isinstance(slope, (int, float)) or slope not in (1.5, 3.0):
        raise DatasetError("invalid stall slope")
    n = len(arrays["trajectory_ids"])
    if n < 10:
        raise DatasetError("at least ten complete trajectories are required")
    recorded_generator = metadata.get("generator")
    if not isinstance(recorded_generator, dict):
        raise DatasetError("missing or invalid named random-stream hierarchy")
    numpy_version = recorded_generator.get("numpy_version")
    if not isinstance(numpy_version, str) or _NUMPY_VERSION.fullmatch(numpy_version) is None:
        raise DatasetError("missing or malformed generator NumPy version provenance")
    generator = _generator_metadata(metadata["seed"], n)
    # Keep the recorded version in metadata and the digest. Only the runtime
    # version is irrelevant to the structural seed/stream contract.
    generator["numpy_version"] = numpy_version
    # Compare the complete JSON structure with its types: Python equality treats
    # bool/int and integral float/int leaves as equal, even deep inside lists.
    # Canonical serialization preserves those distinctions and ignores key order.
    if canonical_json(recorded_generator) != canonical_json(generator):
        raise DatasetError("missing or invalid named random-stream hierarchy")
    if metadata.get("source_model_metadata") != _source_model_metadata(physics["stall"], float(slope), semantics):
        raise DatasetError("source model provenance or environment revision does not match the current canonical environment")
    expected_shapes = {
        "observations": (n, 101, 14), "states": (n, 101, 13),
        "actions": (n, 100, 4), "requested_actions": (n, 100, 4), "rewards": (n, 100),
        "terminated": (n, 100), "truncated": (n, 100),
        "trajectory_ids": (n,), "group_ids": (n,), "episode_seeds": (n,),
        "split_codes": (n,), "behavior_component": (n, 100),
        "action_region": (n, 100), "behavior_log_density": (n, 100),
    }
    for name, dtype in DTYPES.items():
        a = np.asarray(arrays[name])
        if a.shape != expected_shapes[name] or a.dtype.str != dtype:
            raise DatasetError(f"{name}: expected shape {expected_shapes[name]} and dtype {dtype}")
        if not np.all(np.isfinite(a)):
            raise DatasetError(f"{name} contains nonfinite values")
    ids, groups, codes = arrays["trajectory_ids"], arrays["group_ids"], arrays["split_codes"]
    if len(np.unique(ids)) != n or np.any(ids < 0) or np.any(groups < 0):
        raise DatasetError("trajectory IDs must be unique and group IDs nonnegative")
    if set(codes.tolist()) != set(range(len(SPLITS))):
        raise DatasetError("every named split must be nonempty with no unknown codes")
    counts = {name: int(np.count_nonzero(codes == code)) for code, name in enumerate(SPLITS)}
    if metadata.get("episodes") != n or metadata.get("split_counts") != counts:
        raise DatasetError("manifest episode or split counts disagree with actual trajectories")
    seeds = arrays["episode_seeds"]
    if np.any(seeds < 0):
        raise DatasetError("episode seeds must be nonnegative")
    for label, values in (("group", groups), ("seed", seeds)):
        for value in np.unique(values):
            if len(np.unique(codes[values == value])) != 1:
                raise DatasetError(f"{label} {value} leaks across trajectory splits")
        if len(np.unique(values)) != n:
            raise DatasetError(f"duplicate {label}: this schema requires one independent episode per group and seed")
    if not np.array_equal(ids, np.arange(n, dtype=np.int64)):
        raise DatasetError("trajectory IDs must match the ordered random-stream episode addresses")
    expected_seeds = np.asarray([item["reset_seed"] for item in generator["episodes"]], dtype=np.int64)
    if not np.array_equal(seeds, expected_seeds):
        raise DatasetError("episode reset seeds disagree with the named random-stream hierarchy")
    observations = arrays["observations"]
    if not np.array_equal(observations[..., :13], arrays["states"].astype(np.float32)):
        raise DatasetError("physical observations do not equal recorded float64 states cast to float32")
    times = (((100 - np.arange(101)) / 100).astype(np.float32)
             if semantics == "finite_horizon" else np.zeros(101, dtype=np.float32))
    if not np.array_equal(observations[..., 13], np.broadcast_to(times, (n, 101))):
        raise DatasetError("remaining-time or padding coordinate disagrees with task semantics")
    starts = arrays["states"][:, 0]
    amplitudes = np.linalg.norm(starts[:, 11:13], axis=-1)
    if (np.any(np.linalg.norm(starts[:, :10], axis=-1) > 100) or
            np.any(starts[:, 10] < .1) or np.any(starts[:, 10] > .7) or
            np.any(amplitudes < .1 - 1e-14) or np.any(amplitudes > .4 + 1e-14)):
        raise DatasetError("initial state lies outside recorded-condition support")
    final = np.zeros((n, 100), dtype=bool)
    final[:, -1] = True
    expected_terminated = final if semantics == "finite_horizon" else np.zeros_like(final)
    expected_truncated = np.zeros_like(final) if semantics == "finite_horizon" else final
    if not np.array_equal(arrays["terminated"], expected_terminated):
        raise DatasetError("terminated flags disagree with task semantics")
    if not np.array_equal(arrays["truncated"], expected_truncated):
        raise DatasetError("truncated flags disagree with task semantics")
    requests = arrays["requested_actions"]
    if np.any(np.abs(requests) > 1):
        raise DatasetError("requested command lies outside the bounded sampling cube")
    if not np.array_equal(arrays["actions"], requests.astype(np.float32)):
        raise DatasetError("executed commands disagree with float32 quantization of requests")
    region = (requests[..., 0] > .5).astype(np.uint8)
    if not np.array_equal(arrays["action_region"], region) or not np.array_equal(arrays["behavior_component"], region):
        raise DatasetError("action regions or mixture components disagree with requested commands")
    expected = behavior_log_density(requests, epsilon)
    if not np.all(np.isfinite(expected)) or not np.allclose(arrays["behavior_log_density"], expected, rtol=1e-14, atol=1e-14):
        raise DatasetError("logged density disagrees with exact requested-action mixture density")
    _validate_collector_lineage(arrays, metadata["seed"], epsilon)
    if check_dynamics:
        from .environment import Physics, dynamics
        p = Physics(stall=physics["stall"], stall_slope=float(slope))
        nxt, reward = dynamics(arrays["states"][:, :-1], arrays["actions"], p)
        if not np.allclose(nxt, arrays["states"][:, 1:], rtol=1e-13, atol=1e-13):
            raise DatasetError("trajectories do not replay under their recorded physics")
        if not np.allclose(reward, arrays["rewards"], rtol=1e-13, atol=1e-13):
            raise DatasetError("rewards disagree with canonical dynamics")


class TrajectoryDataset:
    """Validated content with immutable arrays and copy-on-read metadata."""
    __slots__ = ("_arrays", "_metadata_json", "_digest")

    def __init__(self, arrays, metadata, *, expected_digest=None):
        copied = {name: immutable_array(value) for name, value in arrays.items()}
        clean_metadata = json.loads(canonical_json(metadata))
        _validate(copied, clean_metadata)
        digest = content_digest(copied, clean_metadata)
        if expected_digest is not None and digest != expected_digest:
            raise DatasetError("dataset SHA-256 mismatch: content has changed")
        object.__setattr__(self, "_arrays", MappingProxyType(copied))
        object.__setattr__(self, "_metadata_json", canonical_json(clean_metadata))
        object.__setattr__(self, "_digest", digest)

    def __setattr__(self, name, value):
        raise AttributeError("TrajectoryDataset is immutable")

    @property
    def arrays(self):
        return self._arrays

    @property
    def metadata(self):
        return json.loads(self._metadata_json)

    @property
    def digest(self):
        return self._digest

    def split(self, name):
        if name not in SPLITS:
            raise DatasetError(f"unknown split {name!r}; expected one of {SPLITS}")
        mask = self._arrays["split_codes"] == SPLITS.index(name)
        return MappingProxyType({key: immutable_array(value[mask]) for key, value in self._arrays.items()})

    def split_digest(self, name):
        return content_digest(self.split(name), {"dataset_sha256": self.digest, "split": name})


def save_dataset(dataset, path):
    """Write once. Existing artifacts are never overwritten."""
    forbid_during_evaluation("save_dataset")
    if type(dataset) is not TrajectoryDataset:
        raise DatasetError("save_dataset requires a validated TrajectoryDataset")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"metadata": dataset.metadata, "sha256": dataset.digest}
    with path.open("xb") as handle:
        np.savez_compressed(handle, **dataset.arrays, manifest=np.asarray(canonical_json(envelope)))
    return path


def load_dataset(path):
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != set(DTYPES) | {"manifest"}:
                raise DatasetError("unexpected NPZ members")
            manifest = archive["manifest"]
            if manifest.shape != () or manifest.dtype.kind != "U":
                raise DatasetError("manifest must be a scalar Unicode JSON string")
            envelope = json.loads(str(manifest))
            arrays = {name: archive[name] for name in DTYPES}
        if set(envelope) != {"metadata", "sha256"}:
            raise DatasetError("invalid dataset manifest envelope")
        return TrajectoryDataset(arrays, envelope["metadata"], expected_digest=envelope["sha256"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot load dataset: {exc}") from exc


def generate_dataset(path=None, *, episodes=40, seed=20261022, stall=True,
                     stall_slope=1.5, task_semantics="finite_horizon", provenance="smoke", epsilon=.25):
    """Collect once, assign seeded trajectory groups, validate, optionally save.

    Collection is simulator use for a fixed logging policy, not online learner
    training. No learner may recollect data in response to validation outcomes.
    """
    forbid_during_evaluation("generate_dataset")
    if isinstance(episodes, bool) or not isinstance(episodes, (int, np.integer)) or episodes < 10:
        raise DatasetError("episodes must be an integer of at least ten")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise DatasetError("seed must be a nonnegative integer")
    if provenance not in ("smoke", "research"):
        raise DatasetError("provenance must be smoke or research")
    if not isinstance(stall, bool) or isinstance(stall_slope, bool) or not isinstance(stall_slope, (int, float, np.integer, np.floating)) or stall_slope not in (1.5, 3.0):
        raise DatasetError("invalid physics")
    if task_semantics not in ("finite_horizon", "continuing_time_limit"):
        raise DatasetError("unknown task semantics")
    epsilon = _check_epsilon(epsilon)
    from .environment import ThrusterChainEnv
    generator = _generator_metadata(int(seed), int(episodes))
    episode_seeds = np.asarray([item["reset_seed"] for item in generator["episodes"]], dtype="<i8")
    codes, counts = _split_assignment(int(episodes), int(seed))
    arrays = {
        "observations": np.empty((episodes, 101, 14), "<f4"),
        "states": np.empty((episodes, 101, 13), "<f8"),
        "actions": np.empty((episodes, 100, 4), "<f4"),
        "requested_actions": np.empty((episodes, 100, 4), "<f8"),
        "rewards": np.empty((episodes, 100), "<f8"),
        "terminated": np.zeros((episodes, 100), bool),
        "truncated": np.zeros((episodes, 100), bool),
        "trajectory_ids": np.arange(episodes, dtype="<i8"),
        "group_ids": np.arange(episodes, dtype="<i8"),
        "episode_seeds": episode_seeds, "split_codes": codes,
        "behavior_component": np.empty((episodes, 100), "|u1"),
        "action_region": np.empty((episodes, 100), "|u1"),
        "behavior_log_density": np.empty((episodes, 100), "<f8"),
    }
    mode = "time_aware" if task_semantics == "finite_horizon" else "physical_padded"
    env = ThrusterChainEnv(stall=stall, stall_slope=stall_slope, start="recorded",
                          observation_mode=mode, task_semantics=task_semantics)
    try:
        for episode, episode_seed in enumerate(episode_seeds):
            action_rng = np.random.Generator(np.random.PCG64(_stream_seed_sequence(int(seed), "actions", episode)))
            obs, info = env.reset(seed=int(episode_seed))
            arrays["observations"][episode, 0], arrays["states"][episode, 0] = obs, info["state"]
            for step in range(100):
                requested, component, log_density = sample_requested_action(action_rng, epsilon)
                action = requested.astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                arrays["observations"][episode, step + 1] = obs
                arrays["states"][episode, step + 1] = info["state"]
                arrays["actions"][episode, step] = action
                arrays["requested_actions"][episode, step] = requested
                arrays["rewards"][episode, step] = reward
                arrays["terminated"][episode, step] = terminated
                arrays["truncated"][episode, step] = truncated
                arrays["behavior_component"][episode, step] = component
                arrays["action_region"][episode, step] = int(requested[0] > .5)
                arrays["behavior_log_density"][episode, step] = log_density
    finally:
        env.close()
    metadata = {
        "schema": SCHEMA, "seed": int(seed), "episodes": int(episodes),
        "splits": list(SPLITS), "split_counts": dict(zip(SPLITS, counts)),
        "observation_mode": mode, "horizon": 100, "gamma": .98,
        "task_semantics": task_semantics, "start_family": "recorded",
        "physics": {"stall": stall, "stall_slope": float(stall_slope)},
        "provenance": provenance,
        "behavior": {"name": "continuous_region_mixture_v1", "epsilon": epsilon,
                     "regions": REGIONS, "state_independent": True,
                     "execution_transform": "float64 request -> float32 command; already bounded so clipping is inactive"},
        "generator": generator,
        "source_model_metadata": _source_model_metadata(stall, float(stall_slope), task_semantics),
        "density": DENSITY_SEMANTICS,
        "grouping": "one independent seeded episode per group; no row-level split",
    }
    dataset = TrajectoryDataset(arrays, metadata)
    if path is not None:
        save_dataset(dataset, path)
    return dataset
