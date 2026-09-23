"""Td3 runner."""

from runtime import published
from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import inspect
import json
from numbers import Real
from pathlib import Path
import sys
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import algorithms.td3_bc_bca as P

N = P.BASE
from runtime import common as C

DEPENDENCY_CONTRACT = "raw_effective_finite_targets_v1"
DEPENDENCY_LIMIT = "Disjoint direct transition-field dependencies conditional on fixed fitted normalization, model/target/calibrator state and random keys, with accepted finite computations. Masked successors remain literal reads and validity dependencies. Training-fitted objects and adaptive posterior feedback remain causal influences; no total causal independence or exchangeability guarantee."


def _integer(value, name, minimum=0, maximum=2**31 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} requires an explicit integer in [{minimum},{maximum}]"
        )


def _seed(value, name):
    _integer(value, name, maximum=2**32 - 1)


def _real(
    value, name, *, minimum=None, maximum=None, strict_min=False, strict_max=False
):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " requires a finite real scalar")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        f = np.float32(value)
    if not np.isfinite(f):
        raise ValueError(name + " must be representable in float32")
    for x in (value, f):
        if minimum is not None and (x <= minimum if strict_min else x < minimum):
            raise ValueError(name + " is below its valid range")
        if maximum is not None and (x >= maximum if strict_max else x > maximum):
            raise ValueError(name + " is above its valid range")


@dataclass(frozen=True)
class Reservation:
    target_size: int
    seed: int
    max_fraction: float
    dependency_contract: str

    def __post_init__(self):
        _integer(self.target_size, "reservation target", 1)
        _seed(self.seed, "reservation seed")
        _real(
            self.max_fraction,
            "reservation fraction",
            minimum=0,
            maximum=1,
            strict_min=True,
            strict_max=True,
        )
        if self.dependency_contract != DEPENDENCY_CONTRACT:
            raise ValueError(
                "unsupported dependency contract; no fallback is available"
            )


@dataclass(frozen=True)
class Reference:
    size: int
    seed: int

    def __post_init__(self):
        _integer(self.size, "reference size", 1)
        _seed(self.seed, "reference seed")


@dataclass(frozen=True)
class RefreshEvent:
    step: int
    seed: int

    def __post_init__(self):
        _integer(self.step, "refresh step")
        _seed(self.seed, "refresh seed")


@dataclass(frozen=True)
class EvaluationEvent:
    kind: str
    step: int
    episode_seeds: tuple

    def __post_init__(self):
        if self.kind not in ("periodic", "final"):
            raise ValueError("evaluation kind must be periodic/final")
        _integer(self.step, "evaluation step", 1)
        if type(self.episode_seeds) is not tuple or not self.episode_seeds:
            raise ValueError(
                "evaluation requires an explicit nonempty tuple of episode seeds"
            )
        for seed in self.episode_seeds:
            _seed(seed, "episode seed")


@dataclass(frozen=True)
class RunProtocol:
    run_id: str
    seed: int
    num_updates: int
    scan_block_size: int
    eval_interval: int
    eval_workers: int
    eval_periodic_episodes: int
    eval_final_episodes: int
    reservation: Reservation | None
    reference: Reference | None
    refresh_events: tuple
    evaluation_events: tuple

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("explicit run identity required")
        _seed(self.seed, "training seed")
        for name in (
            "num_updates",
            "scan_block_size",
            "eval_interval",
            "eval_workers",
            "eval_periodic_episodes",
        ):
            _integer(getattr(self, name), name, 1)
        _integer(self.eval_final_episodes, "final episode count")
        if self.num_updates % self.eval_interval:
            raise ValueError(
                "native horizon must be divisible by periodic interval; no truncation"
            )
        for value, cls, name in (
            (self.reservation, Reservation, "reservation"),
            (self.reference, Reference, "reference"),
        ):
            if value is not None and type(value) is not cls:
                raise TypeError("typed " + name + " required")
        if (
            type(self.refresh_events) is not tuple
            or type(self.evaluation_events) is not tuple
        ):
            raise TypeError(
                "explicit event tuples required, including empty native refresh tuple"
            )
        if any((type(x) is not RefreshEvent for x in self.refresh_events)):
            raise TypeError("typed refresh events required")
        steps = tuple((x.step for x in self.refresh_events))
        if steps != tuple(sorted(set(steps))) or any(
            (s >= self.num_updates for s in steps)
        ):
            raise ValueError(
                "refresh steps must increase uniquely and precede another update"
            )
        if any((type(x) is not EvaluationEvent for x in self.evaluation_events)):
            raise TypeError("typed evaluation events required")
        keys = tuple(((x.step, x.kind == "final") for x in self.evaluation_events))
        if keys != tuple(sorted(set(keys))):
            raise ValueError(
                "evaluation events must be unique and chronological, periodic before final"
            )
        periodic = [x for x in self.evaluation_events if x.kind == "periodic"]
        if [x.step for x in periodic] != list(
            range(self.eval_interval, self.num_updates + 1, self.eval_interval)
        ):
            raise ValueError(
                "explicit periodic events must match every declared boundary"
            )
        if any((len(x.episode_seeds) != self.eval_periodic_episodes for x in periodic)):
            raise ValueError(
                "periodic episode count differs from its explicit declaration"
            )
        final = [x for x in self.evaluation_events if x.kind == "final"]
        if self.eval_final_episodes:
            if (
                len(final) != 1
                or final[0].step != self.num_updates
                or len(final[0].episode_seeds) != self.eval_final_episodes
            ):
                raise ValueError(
                    "one exact final episode-seed batch at the final step required"
                )
        elif final:
            raise ValueError("zero final count cannot carry final evaluation events")


@dataclass(frozen=True)
class AffinitySpecification:
    mode: str
    bandwidth: float | None
    ess_floor: float | None
    tau_min: float | None
    iterations: int | None

    def config(self):
        if self.mode != "off":
            _real(
                self.bandwidth,
                "native-action-unit affinity bandwidth",
                minimum=0,
                strict_min=True,
            )
            _real(
                self.ess_floor,
                "affinity ESS floor",
                minimum=0,
                maximum=1,
                strict_min=True,
            )
            _real(self.tau_min, "affinity minimum temperature", minimum=0, maximum=1)
            _integer(self.iterations, "affinity iterations", 1)
        return P.AffinityIWConfig(**asdict(self))


@dataclass(frozen=True)
class PosteriorSpecification:
    alpha: float
    credibility: float
    draws: int
    blend: float
    cal_lr: float
    cal_beta: float
    width_penalty: float
    scale_ema: float
    affinity: AffinitySpecification

    def config(self):
        return P.Config(
            "bca",
            self.affinity.config(),
            P.PosteriorConfig(self.alpha, self.credibility, self.draws),
            self.blend,
            self.cal_lr,
            self.cal_beta,
            self.width_penalty,
            self.scale_ema,
        )


@dataclass(frozen=True)
class RunSpecification:
    identity: str
    configuration_class: str
    deviation_reason: str | None
    posterior: PosteriorSpecification | None

    def config(self):
        if self.identity not in ("host", "bca"):
            raise ValueError("Only host and bca are supported.")
        if (
            self.configuration_class != "declared_deviation"
            or not self.deviation_reason
        ):
            raise ValueError("The paired reservation and budget must be declared.")
        if self.identity == "host":
            if self.posterior is not None:
                raise ValueError("Host has no posterior.")
            return P.Config("host")
        if type(self.posterior) is not PosteriorSpecification:
            raise ValueError("Explicit full BCA specification required.")
        return self.posterior.config()


def validate_protocol(args, specification, protocol):
    if (
        type(args) is not N.Args
        or type(specification) is not RunSpecification
        or type(protocol) is not RunProtocol
    ):
        raise TypeError(
            "complete native Args and typed specification/protocol required"
        )
    RunProtocol(**asdict_events(protocol))
    cfg = specification.config()
    for name in (
        "seed",
        "num_updates",
        "eval_interval",
        "eval_workers",
        "eval_final_episodes",
    ):
        if type(getattr(args, name)) is not int or getattr(args, name) != getattr(
            protocol, name
        ):
            raise ValueError("native Args and protocol disagree on " + name)
    for name in ("batch_size", "policy_freq"):
        _integer(getattr(args, name), name, 1)
    for name in ("normalize", "normalize_reward", "allow_off_config", "log"):
        if type(getattr(args, name)) is not bool:
            raise ValueError(name + " must be a boolean")
    for name in ("lr", "alpha"):
        _real(getattr(args, name), name, minimum=0, strict_min=True)
    for name in ("discount", "tau"):
        _real(getattr(args, name), name, minimum=0, maximum=1)
    for name in ("policy_noise", "noise_clip"):
        _real(getattr(args, name), name, minimum=0)
    for name in ("reward_scale", "reward_bias"):
        _real(getattr(args, name), name)
    if args.algorithm != "corl_td3_bc" or args.log:
        raise ValueError(
            "native TD3 identity and explicit event sink (log=False) required"
        )
    if protocol.reservation is None:
        raise ValueError("reserved arm requires an explicit reservation contract")
    if cfg.arm == "bca":
        if protocol.reference is None or not protocol.refresh_events:
            raise ValueError("posterior requires explicit reference and refresh events")
    elif protocol.reference is not None or protocol.refresh_events:
        raise ValueError("native controls cannot carry reference or refresh settings")
    recorded = published.published_config("corl_td3_bc", args.dataset)
    if recorded is None:
        raise ValueError(
            "this runner requires a recorded configuration, even for declared deviations"
        )
    if any((not hasattr(args, name) for name in recorded)):
        raise ValueError("recorded configuration field missing from Args")
    differences = published.diff_against_published("corl_td3_bc", args.dataset, args)
    if specification.configuration_class == "published":
        if args.allow_off_config or differences:
            raise ValueError(
                "published classification requires exact recorded config without escape flag"
            )
    elif not args.allow_off_config:
        raise ValueError("declared deviation requires native allow_off_config=True")
    C.check_reward_transform(
        "corl_td3_bc",
        args.dataset,
        args.reward_transform,
        args.reward_scale,
        args.reward_bias,
    )
    published.check_published_config(
        "corl_td3_bc", args.dataset, args, allow_off_config=args.allow_off_config
    )
    if True and args.normalize_reward:
        raise ValueError("reserved reward normalization has no approved TD3 extension")
    return cfg


def asdict_events(protocol):
    return {name: getattr(protocol, name) for name in protocol.__dataclass_fields__}


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray) or isinstance(value, jax.Array):
        return _json_value(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return {"nonfinite_literal": repr(value)}
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_hash(value):
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise ValueError("object array has no accepted reproducible identity")
    h = hashlib.sha256(json.dumps([a.dtype.str, a.shape]).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _finite(value, label):
    for leaf in jax.tree_util.tree_leaves(value):
        if not np.all(np.isfinite(np.asarray(leaf))):
            raise FloatingPointError(label + " contains a nonfinite value")


def _tree_hash(tree):
    return _digest([_array_hash(x) for x in jax.tree_util.tree_leaves(tree)])


def source_identity():
    from runtime.provenance import source_identity as identify

    return identify()


def score_transform_identity(dataset):
    minimum = C.d4rl.infos.REF_MIN_SCORE[dataset]
    maximum = C.d4rl.infos.REF_MAX_SCORE[dataset]
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("invalid D4RL score reference range")
    return {
        "dataset": dataset,
        "reference_min": float(minimum),
        "reference_max": float(maximum),
        "formula": "100 * (raw_return - reference_min) / (reference_max - reference_min)",
        "source_sha256": _file_hash(Path(C.d4rl.infos.__file__).resolve()),
    }


def _convert(raw, max_episode_steps):
    required = ("observations", "actions", "rewards", "terminals")
    if not isinstance(raw, dict) or any((k not in raw for k in required)):
        raise ValueError("raw D4RL mapping required")
    for value in raw.values():
        _array_hash(value)
    n = len(raw["rewards"])
    if n < 2 or any((len(raw[k]) != n for k in required)):
        raise ValueError("aligned nonempty raw fields required")
    _finite([raw[k] for k in required], "raw core data")
    if (
        np.asarray(raw["observations"]).ndim != 2
        or np.asarray(raw["actions"]).ndim != 2
        or (not np.asarray(raw["observations"]).shape[1])
        or (not np.asarray(raw["actions"]).shape[1])
        or (np.asarray(raw["rewards"]).shape != (n,))
        or (np.asarray(raw["terminals"]).shape != (n,))
        or (not np.all(np.isin(raw["terminals"], (0, 1))))
    ):
        raise ValueError("invalid raw transition shapes or terminal indicators")
    if "timeouts" in raw and (
        np.asarray(raw["timeouts"]).shape != (n,)
        or not np.all(np.isin(raw["timeouts"], (0, 1)))
    ):
        raise ValueError("aligned binary timeouts required")
    _integer(max_episode_steps, "maximum episode steps", 1)
    rows, ids, ep_step, episode = ([], [], 0, 0)
    for i in range(n - 1):
        timeout = (
            bool(raw["timeouts"][i])
            if "timeouts" in raw
            else ep_step == max_episode_steps - 1
        )
        if timeout:
            ep_step, episode = (0, episode + 1)
            continue
        rows.append(i)
        ids.append(episode)
        if bool(raw["terminals"][i]):
            ep_step, episode = (0, episode + 1)
        ep_step += 1
    rows, ids = (np.asarray(rows, np.int64), np.asarray(ids, np.int64))
    converted = C.d4rl.qlearning_dataset(
        SimpleNamespace(_max_episode_steps=max_episode_steps), dataset=raw
    )
    expected = {
        "observations": np.asarray(raw["observations"])[rows].astype(np.float32),
        "next_observations": np.asarray(raw["observations"])[rows + 1].astype(
            np.float32
        ),
        "actions": np.asarray(raw["actions"])[rows].astype(np.float32),
        "rewards": np.asarray(raw["rewards"])[rows].astype(np.float32),
        "terminals": np.asarray(raw["terminals"])[rows].astype(bool),
    }
    if (
        not len(rows)
        or set(converted) != set(expected)
        or any(
            (
                not np.array_equal(converted[k], v) or converted[k].dtype != v.dtype
                for k, v in expected.items()
            )
        )
    ):
        raise ValueError(
            "actual D4RL conversion differs from verified raw identity map"
        )
    return ({k: np.array(v, copy=True) for k, v in converted.items()}, rows, ids)


def dependency_maps(rows, done):
    rows, done = (np.asarray(rows, np.int64), np.asarray(done))
    if (
        rows.ndim != 1
        or not len(rows)
        or done.shape != rows.shape
        or np.any(np.diff(rows) <= 0)
        or (not np.all(np.isin(done, (0, 1))))
    ):
        raise ValueError("ordered raw rows and aligned binary terminal masks required")
    adjacent = np.diff(rows) == 1
    literal = np.cumsum(np.r_[0, ~adjacent], dtype=np.int64)
    effective = np.cumsum(np.r_[0, ~(adjacent & (done[:-1] == 0))], dtype=np.int64)
    return {
        "raw_current": rows.tolist(),
        "raw_next_observation": (rows + 1).tolist(),
        "terminal_mask": done.astype(int).tolist(),
        "literal_components": literal.tolist(),
        "effective_components": effective.tolist(),
    }


def _dependencies(maps, indices, literal=False):
    rows = np.asarray(maps["raw_current"], np.int64)[indices]
    done = np.asarray(maps["terminal_mask"])[indices]
    return np.unique(np.r_[rows, rows + 1 if literal else (rows + 1)[done == 0]])


def _partition(maps, training, heldout, reference):
    n = len(maps["raw_current"])
    for ids, name in (
        (training, "training"),
        (heldout, "heldout"),
        (reference, "reference"),
    ):
        a = np.asarray(ids)
        if (
            a.ndim != 1
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0)
            or np.any(a >= n)
            or np.any(np.diff(a) <= 0)
        ):
            raise ValueError(name + " IDs must be sorted unique converted rows")
    if not len(training) or not np.array_equal(
        np.sort(np.r_[training, heldout]), np.arange(n)
    ):
        raise ValueError(
            "training/heldout identities must partition the converted data exactly"
        )
    if not np.all(np.isin(reference, training)):
        raise ValueError("reference contains a nontraining identity")
    effective_overlap = np.intersect1d(
        _dependencies(maps, training), _dependencies(maps, heldout)
    )
    if effective_overlap.size:
        raise ValueError("effective raw dependencies overlap")
    literal_overlap = np.intersect1d(
        _dependencies(maps, training, True), _dependencies(maps, heldout, True)
    )
    return {
        "effective_overlap": effective_overlap.tolist(),
        "literal_shared_raw_rows": literal_overlap.tolist(),
        "training_effective_raw_rows": _dependencies(maps, training).tolist(),
        "heldout_effective_raw_rows": _dependencies(maps, heldout).tolist(),
        "reference_effective_raw_rows": _dependencies(maps, reference).tolist(),
        "reference_dependency_note": "Recorded full-transition set is conservative: scale-reference predictions read current obs/action only.",
    }


def _settings(args, specification, protocol):
    return _json_value(
        {
            "args": asdict(args),
            "specification": asdict(specification),
            "protocol": asdict(protocol),
        }
    )


def _execution_identity(specification):
    return {"method": specification.identity}


@dataclass(frozen=True)
class PreparedData:
    training: object
    heldout: object
    reference: object
    training_ids: object
    heldout_ids: object
    reference_ids: object
    obs_mean: object
    obs_std: object
    max_action: float
    max_episode_steps: int
    metadata: dict
    metadata_sha256: str


def _prepared_hashes(prepared):
    return {
        name: _tree_hash(getattr(prepared, name))
        for name in (
            "training",
            "heldout",
            "reference",
            "training_ids",
            "heldout_ids",
            "reference_ids",
            "obs_mean",
            "obs_std",
            "max_action",
            "max_episode_steps",
        )
    }


def _raw_identity(identity, dataset):
    if not isinstance(identity, dict) or identity.get("dataset") != dataset:
        raise ValueError(
            "explicit raw identity with matching registered dataset required"
        )
    if identity.get("kind") == "synthetic":
        if (
            set(identity) != {"kind", "dataset", "label"}
            or not isinstance(identity["label"], str)
            or (not identity["label"].strip())
        ):
            raise ValueError(
                "synthetic identity requires only kind/dataset/nonempty label"
            )
    elif identity.get("kind") == "cached_hdf5":
        if set(identity) != {"kind", "dataset", "path", "size_bytes", "sha256"}:
            raise ValueError("cached identity requires exact file provenance fields")
        path = Path(identity["path"])
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.stat().st_size != identity["size_bytes"]
            or (_file_hash(path) != identity["sha256"])
        ):
            raise ValueError("cached raw file no longer matches its declared identity")
    else:
        raise ValueError("raw identity must explicitly be synthetic/cached_hdf5")


def prepare(
    raw, args, specification, protocol, *, max_action, max_episode_steps, raw_identity
):
    cfg = validate_protocol(args, specification, protocol)
    _real(max_action, "uniform symmetric action bound", minimum=0, strict_min=True)
    _raw_identity(raw_identity, args.dataset)
    raw_hashes = {k: _array_hash(raw[k]) for k in sorted(raw)}
    converted, rows, episode_ids = _convert(raw, max_episode_steps)
    converted_hashes = {k: _array_hash(v) for k, v in converted.items()}
    maps = dependency_maps(rows, converted["terminals"])
    n = len(rows)
    training_ids, heldout_ids, reservation = (
        np.arange(n, dtype=np.int64),
        np.empty(0, np.int64),
        None,
    )
    r = protocol.reservation
    data = C.Transition(
        converted["observations"],
        converted["actions"],
        converted["rewards"],
        converted["next_observations"],
        converted["terminals"],
    )
    training_ids, heldout_ids, inherited = P.reserve_pool(
        data,
        r.target_size,
        r.seed,
        max_fraction=r.max_fraction,
        episode_ids=np.asarray(maps["effective_components"]),
    )
    components = np.asarray(maps["effective_components"])
    starts = np.r_[0, np.flatnonzero(np.diff(components)) + 1]
    reservation = {
        "contract": DEPENDENCY_CONTRACT,
        "meaning": "whole contiguous effective raw-dependency components; not inferred independent episodes",
        "inherited_primitive_metadata": inherited,
        "inherited_boundary_string_is_generic": True,
        "component_boundaries": np.r_[starts, n].tolist(),
        "heldout_component_ids": np.unique(components[heldout_ids]).tolist(),
        "training_component_ids": np.unique(components[training_ids]).tolist(),
    }
    reference_ids = np.empty(0, np.int64)
    if cfg.arm == "bca":
        if protocol.reference.size > len(training_ids):
            raise ValueError("reference exceeds training complement")
        local = np.sort(
            np.random.default_rng(protocol.reference.seed).choice(
                len(training_ids), protocol.reference.size, replace=False
            )
        )
        reference_ids = np.asarray(training_ids)[local]
    partition = _partition(maps, training_ids, heldout_ids, reference_ids)
    converted["rewards"] = np.asarray(converted["rewards"], np.float64)
    if args.normalize_reward:
        converted = C.modify_reward(
            converted, args.dataset, max_episode_steps=max_episode_steps
        )
    converted = C.apply_reward_transform(
        converted,
        args.dataset,
        args.reward_transform,
        args.reward_scale,
        args.reward_bias,
        legacy_print=True,
    )
    obs, nxt = (
        np.asarray(converted["observations"], np.float32),
        np.asarray(converted["next_observations"], np.float32),
    )
    if args.normalize:
        mean, std = C.compute_mean_std(obs[training_ids], eps=0.001)
        obs, nxt = (
            C.normalize_states(obs, mean, std),
            C.normalize_states(nxt, mean, std),
        )
    else:
        mean, std = (
            np.zeros(obs.shape[1], np.float32),
            np.ones(obs.shape[1], np.float32),
        )
    all_data = C.Transition(
        jnp.asarray(obs),
        jnp.asarray(converted["actions"], jnp.float32),
        jnp.asarray(converted["rewards"], jnp.float32),
        jnp.asarray(nxt),
        jnp.asarray(converted["terminals"], jnp.float32),
    )
    _finite((all_data, mean, std), "prepared data/statistics")
    if np.any(std <= 0):
        raise ValueError("observation standard deviation must be positive")
    if cfg.iw.mode != "off" and np.any(
        np.abs(np.asarray(all_data.action)) > max_action
    ):
        raise ValueError(
            "affinity arm requires dataset actions inside native bounds; no clipping"
        )
    take = lambda ids: jax.tree_util.tree_map(lambda x: x[ids], all_data)
    training = P.select_training_pool(all_data, cfg.arm, training_ids)
    heldout, reference = (
        take(heldout_ids) if len(heldout_ids) else None,
        take(reference_ids) if len(reference_ids) else None,
    )
    settings = _settings(args, specification, protocol)
    all_seeds = [
        seed for event in protocol.evaluation_events for seed in event.episode_seeds
    ]
    metadata = {
        "schema": "native-td3-prepared-v2",
        "settings": settings,
        "settings_sha256": _digest(settings),
        "execution": _execution_identity(specification),
        "source_files": source_identity(),
        "raw_identity": copy.deepcopy(raw_identity),
        "raw_array_hashes": raw_hashes,
        "raw_dataset_sha256": _digest(raw_hashes),
        "raw_rows": len(raw["rewards"]),
        "converted_rows": n,
        "converted_array_hashes": converted_hashes,
        "dependency_contract": DEPENDENCY_CONTRACT,
        "dependency_maps": maps,
        "dependency_limits": DEPENDENCY_LIMIT,
        "original_terminal_timeout_episode_ids": episode_ids.tolist(),
        "partition": partition,
        "training_converted_ids": np.asarray(training_ids).tolist(),
        "heldout_converted_ids": np.asarray(heldout_ids).tolist(),
        "reference_converted_ids": reference_ids.tolist(),
        "reservation": reservation,
        "normalization_fit_converted_ids": np.asarray(training_ids).tolist(),
        "normalization_fit": "training_complement",
        "obs_mean": mean.tolist(),
        "obs_std": std.tolist(),
        "reward_normalization": (
            "exact_native_modify_reward" if args.normalize_reward else "disabled"
        ),
        "preprocessing_order": [
            "exact native D4RL conversion and raw identity verification",
            "explicit effective-component reservation",
            "native float64 reward handling and registered transform",
            "current training-observation mean/std + 1e-3",
            "same transforms applied to training/heldout/reference/evaluation",
        ],
        "published_config_fields": copy.deepcopy(
            published.published_config("corl_td3_bc", args.dataset)
        ),
        "published_config_differences": _json_value(
            published.diff_against_published("corl_td3_bc", args.dataset, args)
        ),
        "affinity_bandwidth_units": "native action units; detached action-kernel tilt, not a density ratio",
        "posterior_refresh": "frozen arm snapshot, fixed training reference, unweighted heldout Bayesian/conformal posterior",
        "evaluation_seed_convention": "explicit seed per episode; not upstream sequential-reset identity",
        "evaluation_score_transform": score_transform_identity(args.dataset),
        "repeated_episode_seeds": sorted(
            {x for x in all_seeds if all_seeds.count(x) > 1}
        ),
    }
    prepared = PreparedData(
        training,
        heldout,
        reference,
        jnp.asarray(training_ids),
        jnp.asarray(heldout_ids),
        jnp.asarray(reference_ids),
        jnp.asarray(mean),
        jnp.asarray(std),
        float(max_action),
        max_episode_steps,
        metadata,
        "",
    )
    metadata["run_input_hashes"] = _prepared_hashes(prepared)
    return PreparedData(**{**prepared.__dict__, "metadata_sha256": _digest(metadata)})


def load_and_prepare(args, specification, protocol, *, dataset_file):
    validate_protocol(args, specification, protocol)
    path = Path(dataset_file)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.suffix.lower() not in (".h5", ".hdf5")
    ):
        raise ValueError("explicit existing absolute cached HDF5 file required")
    identity = {
        "kind": "cached_hdf5",
        "dataset": args.dataset,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _file_hash(path),
    }
    env = C.gym.make(args.dataset)
    try:
        hi, lo = (np.asarray(env.action_space.high), np.asarray(env.action_space.low))
        if (
            hi.ndim != 1
            or not len(hi)
            or lo.shape != hi.shape
            or (not np.all(np.isfinite(hi)))
            or (not np.all(hi > 0))
            or (not np.all(hi == hi[0]))
            or (not np.array_equal(lo, -hi))
        ):
            raise ValueError("finite uniform symmetric action bounds required")
        raw = env.get_dataset(h5path=str(path.resolve()))
        if (
            np.asarray(raw["observations"]).shape[1:] != env.observation_space.shape
            or np.asarray(raw["actions"]).shape[1:] != env.action_space.shape
        ):
            raise ValueError(
                "raw dimensions disagree with registered environment spaces"
            )
        return prepare(
            raw,
            args,
            specification,
            protocol,
            max_action=float(hi[0]),
            max_episode_steps=env._max_episode_steps,
            raw_identity=identity,
        )
    finally:
        env.close()


def validate_prepared(args, specification, protocol, prepared):
    cfg = validate_protocol(args, specification, protocol)
    if (
        type(prepared) is not PreparedData
        or _digest(prepared.metadata) != prepared.metadata_sha256
    ):
        raise ValueError("complete prepared metadata digest mismatch")
    m = prepared.metadata
    if m.get("schema") != "native-td3-prepared-v2" or m.get(
        "execution"
    ) != _execution_identity(specification):
        raise ValueError("prepared schema or execution identity mismatch")
    if (
        m["settings"] != _settings(args, specification, protocol)
        or _digest(m["settings"]) != m["settings_sha256"]
    ):
        raise ValueError("prepared settings mismatch")
    if m["source_files"] != source_identity() or m[
        "run_input_hashes"
    ] != _prepared_hashes(prepared):
        raise ValueError("prepared source or array identity mismatch")
    _raw_identity(m["raw_identity"], args.dataset)
    maps = dependency_maps(
        m["dependency_maps"]["raw_current"], m["dependency_maps"]["terminal_mask"]
    )
    if maps != m["dependency_maps"] or m["dependency_limits"] != DEPENDENCY_LIMIT:
        raise ValueError("raw dependency map or guarantee mismatch")
    ids = [
        np.asarray(x)
        for x in (prepared.training_ids, prepared.heldout_ids, prepared.reference_ids)
    ]
    if _partition(maps, *ids) != m["partition"]:
        raise ValueError("raw dependency membership mismatch")
    for actual, name, pool in zip(
        ids,
        ("training", "heldout", "reference"),
        (prepared.training, prepared.heldout, prepared.reference),
    ):
        if actual.tolist() != m[name + "_converted_ids"] or len(actual) != (
            len(pool.obs) if pool is not None else 0
        ):
            raise ValueError("prepared identity/array count mismatch")
        if pool is not None and (
            not np.array_equal(pool.done, np.asarray(maps["terminal_mask"])[actual])
        ):
            raise ValueError("prepared terminal mask differs from dependency map")
    if (
        m["normalization_fit_converted_ids"] != ids[0].tolist()
        or not np.array_equal(m["obs_mean"], prepared.obs_mean)
        or (not np.array_equal(m["obs_std"], prepared.obs_std))
    ):
        raise ValueError("normalization fitting identity mismatch")
    if m["dependency_contract"] != DEPENDENCY_CONTRACT:
        raise ValueError("unsupported reserved dependency contract")
    comp = np.asarray(maps["effective_components"])
    boundaries = np.r_[0, np.flatnonzero(np.diff(comp)) + 1, len(comp)]
    lengths = np.diff(boundaries)
    order = np.random.default_rng(protocol.reservation.seed).permutation(len(lengths))
    count = (
        int(
            np.searchsorted(np.cumsum(lengths[order]), protocol.reservation.target_size)
        )
        + 1
    )
    expected = np.sort(
        np.concatenate(
            [np.arange(boundaries[i], boundaries[i + 1]) for i in order[:count]]
        )
    )
    if not np.array_equal(ids[1], expected) or len(
        expected
    ) > protocol.reservation.max_fraction * len(comp):
        raise ValueError("reservation differs from explicit seeded component selection")
    if cfg.arm == "bca":
        local = np.sort(
            np.random.default_rng(protocol.reference.seed).choice(
                len(ids[0]), protocol.reference.size, replace=False
            )
        )
        if not np.array_equal(ids[2], ids[0][local]) or any(
            (
                not np.array_equal(a, b[local])
                for a, b in zip(prepared.reference, prepared.training)
            )
        ):
            raise ValueError(
                "reference arrays/IDs differ from exact seeded training subset"
            )
    _finite(
        (
            prepared.training,
            prepared.heldout,
            prepared.reference,
            prepared.obs_mean,
            prepared.obs_std,
        ),
        "prepared inputs",
    )
    if np.any(np.asarray(prepared.obs_std) <= 0):
        raise ValueError("invalid normalization scale")
    return cfg


class EvaluationFailure(RuntimeError):

    def __init__(self, message, record):
        super().__init__(message)
        self.record = copy.deepcopy(record)


def evaluate_episodes(
    actor, params, prepared, event, *, dataset, max_workers, env_factory=None
):
    if type(event) is not EvaluationEvent:
        raise TypeError("typed evaluation event required")
    if dataset != prepared.metadata["settings"]["args"]["dataset"]:
        raise ValueError("evaluation dataset differs from prepared data identity")
    transform = score_transform_identity(dataset)
    if transform != prepared.metadata["evaluation_score_transform"]:
        raise ValueError("evaluation score reference differs from prepared provenance")
    _integer(max_workers, "maximum evaluation workers", 1)
    _finite(params, "evaluation actor state")
    factory = C.make_eval_env if env_factory is None else env_factory
    records = []
    for start in range(0, len(event.episode_seeds), max_workers):
        seeds = event.episode_seeds[start : start + max_workers]
        workers = len(seeds)
        env = factory(dataset, workers)
        try:
            env.seed(list(seeds))
            obs = np.asarray(env.reset())
            finished, totals = (np.zeros(workers, bool), np.zeros(workers, np.float64))
            lengths = np.zeros(workers, np.int64)
            for _ in range(prepared.max_episode_steps):
                if obs.shape != (workers, prepared.training.obs.shape[1]):
                    raise ValueError("evaluation observation shape mismatch")
                _finite(obs, "evaluation observations")
                normalized = (jnp.asarray(obs) - prepared.obs_mean) / prepared.obs_std
                actions = actor.apply(params, normalized)
                _finite((normalized, actions), "deterministic policy output")
                actions = np.asarray(actions)
                if actions.shape != (
                    workers,
                    prepared.training.action.shape[1],
                ) or np.any(np.abs(actions) > prepared.max_action):
                    raise ValueError(
                        "deterministic action shape/bound violation; no repair"
                    )
                obs, reward, done, _ = env.step(actions)
                obs, reward, done = (
                    np.asarray(obs),
                    np.asarray(reward),
                    np.asarray(done),
                )
                _finite((obs, reward, done), "evaluation environment outputs")
                if (
                    obs.shape != (workers, prepared.training.obs.shape[1])
                    or reward.shape != (workers,)
                    or done.shape != (workers,)
                    or (not np.all(np.isin(done, (0, 1))))
                ):
                    raise ValueError(
                        "evaluation output shape or done-indicator mismatch"
                    )
                totals += reward * ~finished
                lengths += ~finished
                finished |= done.astype(bool)
                if finished.all():
                    break
            if not finished.all():
                raise EvaluationFailure(
                    "declared episodes did not complete within authoritative cap",
                    {
                        "kind": "incomplete_evaluation",
                        "step": event.step,
                        "episode_seeds": list(seeds),
                        "completed_mask": finished.tolist(),
                        "partial_returns_not_scores": totals.tolist(),
                    },
                )
            _finite(totals, "episode returns")
            scores = np.asarray(C.normalized_scores(dataset, totals))
            _finite(scores, "normalized scores")
            if scores.shape != totals.shape:
                raise ValueError("normalized-score shape mismatch")
            records += [
                {
                    "episode_index": start + i,
                    "seed": seed,
                    "batch_start": start,
                    "worker": i,
                    "length": int(lengths[i]),
                    "raw_return": float(totals[i]),
                    "normalized_score": float(scores[i]),
                }
                for i, seed in enumerate(seeds)
            ]
        finally:
            env.close()
    return {
        "kind": event.kind,
        "step": event.step,
        "episodes": records,
        "episode_count": len(records),
        "episode_seeds": list(event.episode_seeds),
        "normalization": "corl_common.normalized_scores, D4RL reference scale ×100",
        "score_transform": transform,
        "repeated_episode_seeds": sorted(
            {x for x in event.episode_seeds if event.episode_seeds.count(x) > 1}
        ),
    }


def _accept(state, metrics):
    P.require_valid(metrics)
    _finite(state.native, "native/target/optimizer state")
    _finite(metrics, "scan/refresh metrics")
    if state.calibrator is not None:
        _finite(
            (state.calibrator, state.residual_scale), "calibrator/optimizer/EMA state"
        )
        if not bool(P.posterior_storage_valid(state.posterior)):
            raise FloatingPointError("invalid posterior storage")


def run_prepared(
    args,
    specification,
    protocol,
    prepared,
    *,
    env_factory=None,
    on_event=None,
    on_checkpoint=None,
):
    cfg = validate_prepared(args, specification, protocol, prepared)
    args, specification, protocol = copy.deepcopy((args, specification, protocol))
    prepared = copy.deepcopy(prepared)
    rng, state, models = P.initialize(
        args,
        cfg,
        prepared.training.obs.shape[1],
        prepared.training.action.shape[1],
        prepared.max_action,
    )
    _accept(state, {})
    carry, completed = ((rng, state, jnp.int32(0)), 0)
    step_fn = P.make_train_step(
        args, cfg, models, prepared.training, prepared.max_action
    )
    records, evaluations = ([], [])

    def emit(record):
        snapshot = copy.deepcopy(
            _json_value(dict(record, execution=_execution_identity(specification)))
        )
        records.append(snapshot)
        if on_event is not None:
            on_event(copy.deepcopy(snapshot))

    def refresh_at(event):
        nonlocal carry
        key = jax.random.fold_in(jax.random.PRNGKey(event.seed), event.step)
        frozen_identity = _tree_hash(carry[1])
        proposal, metrics = P.refresh(
            args,
            cfg,
            models,
            carry[1],
            prepared.reference,
            prepared.heldout,
            key,
            prepared.max_action,
            training_ids=prepared.reference_ids,
            heldout_ids=prepared.heldout_ids,
        )
        _accept(proposal, metrics)
        diagnostics = P.component_diagnostics(cfg, models, proposal, prepared.reference)
        _finite(diagnostics, "component diagnostics")
        if not bool(diagnostics["level_engagement_valid"]):
            raise FloatingPointError("invalid frozen component diagnostic")
        carry = (carry[0], proposal, carry[2])
        emit(
            {
                "kind": "refresh",
                "step": event.step,
                "refresh_seed": event.seed,
                "refresh_key": key,
                "frozen_input_state_sha256": frozen_identity,
                "posterior_snapshot_sha256": _tree_hash(proposal.posterior),
                "reference_sha256": _tree_hash(prepared.reference),
                "heldout_sha256": _tree_hash(prepared.heldout),
                "training_reference_ids": prepared.reference_ids,
                "heldout_ids": prepared.heldout_ids,
                "posterior_weighting": "unweighted",
                "metrics": metrics,
                "component_diagnostics": diagnostics,
                "posterior_ready": proposal.posterior.ready,
                "residual_unit": proposal.posterior.residual_scale,
                "radii": proposal.posterior.radii._asdict(),
            }
        )

    refreshes = {x.step: x for x in protocol.refresh_events}
    evals = {}
    for event in protocol.evaluation_events:
        evals.setdefault(event.step, []).append(event)
    boundaries = sorted(
        set(
            range(
                protocol.scan_block_size, protocol.num_updates, protocol.scan_block_size
            )
        )
        | set(refreshes)
        | set(evals)
        | {protocol.num_updates}
    )
    emit({"kind": "prepared", "step": 0, "metadata": prepared.metadata})
    try:
        if 0 in refreshes:
            refresh_at(refreshes[0])
        for step in boundaries:
            if step == 0:
                continue
            _accept(carry[1], {})
            proposal, metrics = jax.lax.scan(
                step_fn, carry, None, length=step - completed
            )
            _accept(proposal[1], metrics)
            if int(proposal[2]) != step:
                raise RuntimeError(
                    "accepted native iteration differs from declared boundary"
                )
            carry, completed = (proposal, step)
            emit(
                {
                    "kind": "accepted_scan",
                    "step": step,
                    "metrics": metrics,
                    "posterior_snapshot_sha256": (
                        None
                        if carry[1].posterior is None
                        else _tree_hash(carry[1].posterior)
                    ),
                }
            )
            if step in refreshes:
                refresh_at(refreshes[step])
            if on_checkpoint is not None and step in (
                10000,
                50000,
                protocol.num_updates,
            ):
                on_checkpoint(step, copy.deepcopy(carry))
            for event in evals.get(step, ()):
                _accept(carry[1], {})
                record = evaluate_episodes(
                    models[0],
                    carry[1].native.actor.params,
                    prepared,
                    event,
                    dataset=args.dataset,
                    max_workers=protocol.eval_workers,
                    env_factory=env_factory,
                )
                evaluations.append(copy.deepcopy(record))
                emit(record)
        _accept(carry[1], {})
        emit(
            {
                "kind": "completed",
                "step": completed,
                "final_episodes_actual": sum(
                    (r["episode_count"] for r in evaluations if r["kind"] == "final")
                ),
            }
        )
    except Exception as error:
        emit(
            {
                "kind": "failed",
                "last_accepted_step": completed,
                "error_type": type(error).__name__,
                "error": str(error),
                "evaluation_failure": getattr(error, "record", None),
            }
        )
        raise
    return {
        "state": carry[1],
        "training_rng": carry[0],
        "steps_completed": completed,
        "events": records,
        "evaluations": evaluations,
        "metadata": copy.deepcopy(prepared.metadata),
    }
