"""Rebrac runner."""

from runtime import published
import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import algorithms.rebrac_bca as P
import runtime.td3_bc as T
from runtime.td3_bc import (
    _integer,
    _seed,
    _real,
    _json_value,
    _digest,
    _file_hash,
    _array_hash,
    _finite,
    _tree_hash,
    RefreshEvent,
    EvaluationEvent,
    EvaluationFailure,
    score_transform_identity,
    evaluate_episodes,
)

N = P.BASE
from runtime import common as C

DEPENDENCY_CONTRACT = "raw_effective_finite_targets_next_actions_v1"
DEPENDENCY_LIMIT = "Disjoint direct current/successor observation/action and current reward/terminal dependencies conditional on fixed fitted normalization, models, calibrator and keys with accepted finite computations. Terminal-masked successors remain literal reads and validity dependencies. Fitted objects and adaptive feedback remain causal influences; no total causal independence or exchangeability guarantee."
TD3_HELPER_SHA256 = "f4c336cb55374258001adc323c8f991b342b49f378c9fa02320bbcef805068f9"
LOADER_FILE_SHA256 = "35ac35d6fa88227cc46dcb10fba9c74fc44c35b530757fe88951339a0d89775b"
LOADER_SEGMENT_SHA256 = (
    "323386efe1633f189051357c834878c9ad0d4a778c27333be069cc454bec6663"
)


@dataclass(frozen=True)
class Reservation:
    target_size: int
    seed: int
    max_fraction: float
    dependency_contract: str
    rows_per_episode: int  # K calibration rows from each withheld component (calibration/bank.py)

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
        self._check_rows_per_episode()

    def _check_rows_per_episode(self):
        _integer(self.rows_per_episode, "reservation rows per episode", 1)


@dataclass(frozen=True)
class PopulationSplit(Reservation):
    # Population splits ONLY (experiments/wbcp/freeze_rebrac.py): rows_per_episode=None
    # withholds whole components and keeps every row of them, never a WBCP bank
    # (DEPENDENCE.md), so run_prepared refuses it. Reservation, which
    # runtime.config.typed builds, still refuses None.
    def _check_rows_per_episode(self):
        if self.rows_per_episode is not None:
            raise ValueError(
                "a population split keeps every withheld row; rows_per_episode is None"
            )


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
        if self.reservation is not None and type(self.reservation) not in (
            Reservation,
            PopulationSplit,
        ):
            raise TypeError("typed reservation required")
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
    converted = _native_loader()(
        SimpleNamespace(_max_episode_steps=max_episode_steps), dataset=raw
    )
    expected = {
        "observations": np.asarray(raw["observations"])[rows].astype(np.float32),
        "next_observations": np.asarray(raw["observations"])[rows + 1].astype(
            np.float32
        ),
        "actions": np.asarray(raw["actions"])[rows].astype(np.float32),
        "next_actions": np.asarray(raw["actions"])[rows + 1].astype(np.float32),
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
            "actual native ReBRAC next-action conversion differs from verified raw identity map"
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
        "raw_next_action": (rows + 1).tolist(),
        "terminal_mask": done.astype(int).tolist(),
        "literal_components": literal.tolist(),
        "effective_components": effective.tolist(),
    }


def _dependencies(maps, indices, literal=False):
    rows = np.asarray(maps["raw_current"], np.int64)[indices]
    done = np.asarray(maps["terminal_mask"])[indices]
    return np.unique(np.r_[rows, rows + 1 if literal else (rows + 1)[done == 0]])


def _partition(maps, training, withheld, heldout):
    """Training and withheld rows partition the data with disjoint effective dependencies.

    The held-out calibration bank is a subset of the withheld rows, so the no-leak check
    covers every withheld row, calibrated or not.
    """
    n = len(maps["raw_current"])
    for ids, name in ((training, "training"), (withheld, "withheld"), (heldout, "heldout")):
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
        np.sort(np.r_[training, withheld]), np.arange(n)
    ):
        raise ValueError(
            "training/withheld identities must partition the converted data exactly"
        )
    if not np.all(np.isin(heldout, withheld)):
        raise ValueError("the held-out bank must lie inside the withheld rows")
    effective_overlap = np.intersect1d(
        _dependencies(maps, training), _dependencies(maps, withheld)
    )
    if effective_overlap.size:
        raise ValueError("effective raw dependencies overlap")
    literal_overlap = np.intersect1d(
        _dependencies(maps, training, True), _dependencies(maps, withheld, True)
    )
    return {
        "effective_overlap": effective_overlap.tolist(),
        "literal_shared_raw_rows": literal_overlap.tolist(),
        "literal_shared_action_raw_rows": literal_overlap.tolist(),
        "training_effective_raw_rows": _dependencies(maps, training).tolist(),
        "withheld_effective_raw_rows": _dependencies(maps, withheld).tolist(),
        "heldout_effective_raw_rows": _dependencies(maps, heldout).tolist(),
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

    def config(self):
        return P.Config(
            "bca",
            P.WBCPConfig(self.alpha, self.credibility, self.draws),
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
    RunProtocol(**{n: getattr(protocol, n) for n in protocol.__dataclass_fields__})
    cfg = specification.config()
    P.validate_native_args(args)
    if args.log:
        raise ValueError("explicit event sink requires native log=False")
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
            raise ValueError("native Args/protocol mismatch: " + name)
    recorded = published.published_config("corl_rebrac", args.dataset)
    if recorded is None or any((not hasattr(args, k) for k in recorded)):
        raise ValueError("complete recorded native environment configuration required")
    differences = published.diff_against_published("corl_rebrac", args.dataset, args)
    if specification.configuration_class == "published":
        if args.allow_off_config or differences:
            raise ValueError("published class must match recorded native settings")
    elif not args.allow_off_config:
        raise ValueError("declared deviation requires allow_off_config=True")
    if False != (protocol.reservation is None):
        raise ValueError("Both methods require the declared reservation")
    if cfg.is_posterior:
        if not protocol.refresh_events:
            raise ValueError("posterior requires explicit refresh events")
    elif protocol.refresh_events:
        raise ValueError("controls cannot carry refresh fields")
    seeds = [s for e in protocol.evaluation_events for s in e.episode_seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "evaluation episode banks must be disjoint with no repeated seeds"
        )
    fitting = {protocol.seed} | {e.seed for e in protocol.refresh_events}
    fitting |= {protocol.reservation.seed}
    if fitting.intersection(seeds):
        raise ValueError("evaluation seeds overlap training/calibration seeds")
    return cfg


def _native_loader():
    from runtime.next_actions import qlearning_dataset_with_next_actions

    return qlearning_dataset_with_next_actions


def source_identity():
    from runtime.provenance import source_identity as identify

    return identify()


@dataclass(frozen=True)
class PreparedData:
    converted: dict
    training: object
    heldout: object
    training_ids: object
    heldout_ids: object
    obs_mean: object
    obs_std: object
    cal_obs_mean: object
    cal_obs_std: object
    max_action: float
    max_episode_steps: int
    metadata: dict
    metadata_sha256: str


def _settings(args, specification, protocol):
    return _json_value(
        dict(
            args=asdict(args),
            specification=asdict(specification),
            protocol=asdict(protocol),
        )
    )


def _execution(specification):
    return {"method": specification.identity}


def _preprocess(converted, args, training_ids):
    data = dict(converted, rewards=np.asarray(converted["rewards"], np.float64).copy())
    data = C.apply_reward_transform(
        data,
        args.dataset,
        args.reward_transform,
        args.reward_scale,
        args.reward_bias,
        legacy_print=False,
    )
    obs, nxt = (
        np.asarray(data["observations"], np.float32),
        np.asarray(data["next_observations"], np.float32),
    )
    if args.normalize_states:
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
    native = C.TransitionNA(
        jnp.asarray(obs),
        jnp.asarray(data["actions"], jnp.float32),
        jnp.asarray(data["rewards"], jnp.float32),
        jnp.asarray(nxt),
        jnp.asarray(data["terminals"], jnp.float32),
        jnp.asarray(data["next_actions"], jnp.float32),
    )
    if not bool(P.transition_valid(native)) or not np.all(np.isfinite(std) & (std > 0)):
        raise ValueError("finite prepared TransitionNA and native statistics required")
    return (native, jnp.asarray(mean), jnp.asarray(std))


def _prepared_hashes(prepared):
    return {
        name: _tree_hash(getattr(prepared, name))
        for name in prepared.__dataclass_fields__
        if name not in ("metadata", "metadata_sha256")
    }


def _selection(converted, maps, protocol):
    """(training, withheld, heldout, record): training and withheld partition the rows;
    heldout is the WBCP calibration bank, K stratified rows of each withheld component
    (every withheld row for a PopulationSplit, rows_per_episode=None)."""
    r = protocol.reservation
    native = C.TransitionNA(
        converted["observations"],
        converted["actions"],
        converted["rewards"],
        converted["next_observations"],
        converted["terminals"],
        converted["next_actions"],
    )
    train, withheld, hold, inherited = P.reserve_pool(
        native,
        r.target_size,
        r.seed,
        r.rows_per_episode,
        max_fraction=r.max_fraction,
        episode_ids=np.asarray(maps["effective_components"]),
    )
    reservation = dict(
        contract=DEPENDENCY_CONTRACT,
        meaning=(
            "whole withheld effective raw-dependency components, every row kept "
            "(population split for experiments/wbcp/freeze_rebrac.py, not a calibration design); "
            "not inferred independent episodes"
            if r.rows_per_episode is None
            else "K stratified rows from each withheld effective raw-dependency component; "
            "every row of a withheld component is excluded from training; "
            "not inferred independent episodes"
        ),
        inherited_primitive_metadata=inherited,
        inherited_boundary_string_is_generic=True,
    )
    return (train, withheld, hold, reservation)


def prepare(
    raw, args, specification, protocol, *, max_action, max_episode_steps, raw_identity
):
    cfg = validate_protocol(args, specification, protocol)
    _real(max_action, "literal native unit bound", minimum=1, maximum=1)
    _raw_identity(raw_identity, args.dataset)
    converted, rows, episode_ids = _convert(raw, max_episode_steps)
    maps = dependency_maps(rows, converted["terminals"])
    train, withheld, hold, reservation = _selection(converted, maps, protocol)
    if cfg.arm == "bca" and not P.certifiable(len(hold), cfg.posterior):
        raise ValueError("held-out bank is too small for WBCP to certify a finite threshold")
    partition = _partition(maps, train, withheld, hold)
    all_data, mean, std = _preprocess(converted, args, train)
    take = lambda ids: (
        jax.tree_util.tree_map(lambda x: x[ids], all_data) if len(ids) else None
    )
    training = take(train)
    cm, cs = (
        (jnp.mean(training.obs, axis=0), jnp.std(training.obs, axis=0, ddof=0))
        if cfg.is_posterior
        else (None, None)
    )
    _finite((cm, cs), "calibrator-only fixed statistics")
    settings = _settings(args, specification, protocol)
    metadata = dict(
        schema="native-rebrac-prepared-v4",
        settings=settings,
        settings_sha256=_digest(settings),
        execution=_execution(specification),
        source_files=source_identity(),
        raw_identity=copy.deepcopy(raw_identity),
        raw_array_hashes={k: _array_hash(raw[k]) for k in sorted(raw)},
        converted_array_hashes={k: _array_hash(v) for k, v in converted.items()},
        raw_rows=len(raw["rewards"]),
        converted_rows=len(rows),
        dependency_maps=maps,
        original_terminal_timeout_episode_ids=episode_ids.tolist(),
        partition=partition,
        dependency_contract=DEPENDENCY_CONTRACT,
        dependency_limits=DEPENDENCY_LIMIT,
        reservation=reservation,
        training_converted_ids=train.tolist(),
        withheld_converted_ids=withheld.tolist(),
        heldout_converted_ids=hold.tolist(),
        normalization_fit_converted_ids=train.tolist(),
        normalization_fit="training_complement",
        native_normalization_enabled=args.normalize_states,
        reward_preprocessing="float64 then registered ReBRAC transform; no modify_reward",
        cal_normalization=(
            "fixed final-training-observation population mean/std; Calibrator supplies +1e-3"
            if cfg.is_posterior
            else None
        ),
        evaluation_banks=[
            dict(
                kind=e.kind,
                step=e.step,
                episode_seeds=list(e.episode_seeds),
                bank_sha256=_digest(e.episode_seeds),
            )
            for e in protocol.evaluation_events
        ],
        evaluation_score_transform=score_transform_identity(args.dataset),
        published_config_fields=copy.deepcopy(
            published.published_config("corl_rebrac", args.dataset)
        ),
        published_config_differences=_json_value(
            published.diff_against_published("corl_rebrac", args.dataset, args)
        ),
    )
    prepared = PreparedData(
        converted,
        training,
        take(hold),
        train,
        hold,
        mean,
        std,
        cm,
        cs,
        1.0,
        max_episode_steps,
        metadata,
        "",
    )
    metadata["run_input_hashes"] = _prepared_hashes(prepared)
    return PreparedData(**{**prepared.__dict__, "metadata_sha256": _digest(metadata)})


def load_and_prepare(args, specification, protocol, *, dataset_file, cache_registry):
    validate_protocol(args, specification, protocol)
    path = Path(dataset_file)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.suffix.lower() not in (".h5", ".hdf5")
        or (not isinstance(cache_registry, dict))
        or (args.dataset not in cache_registry)
        or (Path(cache_registry[args.dataset]).resolve() != path.resolve())
    ):
        raise ValueError("explicit existing registered absolute HDF5 cache required")
    identity = dict(
        kind="cached_hdf5",
        dataset=args.dataset,
        path=str(path.resolve()),
        size_bytes=path.stat().st_size,
        sha256=_file_hash(path),
    )
    env = C.gym.make(args.dataset)
    try:
        if env.spec.id != args.dataset:
            raise ValueError("registered environment identity mismatch")
        hi, lo = (np.asarray(env.action_space.high), np.asarray(env.action_space.low))
        if (
            hi.ndim != 1
            or not len(hi)
            or lo.shape != hi.shape
            or (not np.all(hi == 1))
            or (not np.all(lo == -1))
        ):
            raise ValueError("native ReBRAC requires literal unit action bounds")
        raw = env.get_dataset(h5path=str(path.resolve()))
        if (
            raw["observations"].shape[1:] != env.observation_space.shape
            or raw["actions"].shape[1:] != env.action_space.shape
        ):
            raise ValueError("cache dimensions disagree with registered spaces")
        return prepare(
            raw,
            args,
            specification,
            protocol,
            max_action=1.0,
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
        raise ValueError("prepared metadata digest mismatch")
    m = prepared.metadata
    if (
        m.get("schema") != "native-rebrac-prepared-v4"
        or m["settings"] != _settings(args, specification, protocol)
        or m["settings_sha256"] != _digest(m["settings"])
        or (m["execution"] != _execution(specification))
        or (m["source_files"] != source_identity())
        or (m["run_input_hashes"] != _prepared_hashes(prepared))
    ):
        raise ValueError("prepared source/settings/input identity mismatch")
    _raw_identity(m["raw_identity"], args.dataset)
    maps = dependency_maps(
        m["dependency_maps"]["raw_current"], prepared.converted["terminals"]
    )
    train, withheld, hold, reservation = _selection(prepared.converted, maps, protocol)
    if (
        maps != m["dependency_maps"]
        or m["dependency_limits"] != DEPENDENCY_LIMIT
        or reservation != m["reservation"]
        or withheld.tolist() != m["withheld_converted_ids"]
        or (_partition(maps, train, withheld, hold) != m["partition"])
    ):
        raise ValueError("prepared next-action dependency partition mismatch")
    expected_contract = DEPENDENCY_CONTRACT
    if (
        m["dependency_contract"] != expected_contract
        or m["normalization_fit_converted_ids"] != train.tolist()
    ):
        raise ValueError("prepared dependency/normalization fit contract mismatch")
    if {k: _array_hash(v) for k, v in prepared.converted.items()} != m[
        "converted_array_hashes"
    ]:
        raise ValueError("converted array identity mismatch")
    all_data, mean, std = _preprocess(prepared.converted, args, train)
    for ids, name in ((train, "training"), (hold, "heldout")):
        expected = (
            jax.tree_util.tree_map(lambda x: x[ids], all_data) if len(ids) else None
        )
        if (
            ids.tolist() != m[name + "_converted_ids"]
            or not np.array_equal(ids, getattr(prepared, name + "_ids"))
            or _tree_hash(expected) != _tree_hash(getattr(prepared, name))
        ):
            raise ValueError("prepared partition arrays/IDs mismatch: " + name)
    if _tree_hash((mean, std)) != _tree_hash((prepared.obs_mean, prepared.obs_std)):
        raise ValueError("native statistics mismatch")
    calstats = (
        (
            jnp.mean(prepared.training.obs, axis=0),
            jnp.std(prepared.training.obs, axis=0, ddof=0),
        )
        if cfg.is_posterior
        else (None, None)
    )
    if _tree_hash(calstats) != _tree_hash(
        (prepared.cal_obs_mean, prepared.cal_obs_std)
    ):
        raise ValueError("calibrator-only fixed statistics mismatch")
    if prepared.max_action != 1.0:
        raise ValueError("native action bound mismatch")
    _integer(prepared.max_episode_steps, "authoritative episode cap", 1)
    return cfg


def _accept(state, metrics, models):
    P.require_valid(metrics)
    _finite((state.native, metrics), "native/optimizer state or metrics")
    if state.posterior is not None and (
        not bool(P.calibration_storage_valid(models, state))
    ):
        raise FloatingPointError("invalid calibration/posterior/statistics storage")


def run_prepared(
    args,
    specification,
    protocol,
    prepared,
    *,
    env_factory=None,
    on_event=None,
    on_checkpoint=None
):
    if isinstance(getattr(protocol, "reservation", None), PopulationSplit):
        raise ValueError("a whole-component population split is not a WBCP bank; it is never run")
    args, specification, protocol, prepared = copy.deepcopy(
        (args, specification, protocol, prepared)
    )
    records, evaluations = ([], [])

    def emit(kind, **values):
        snapshot = copy.deepcopy(
            _json_value(dict(kind=kind, execution=_execution(specification), **values))
        )
        records.append(snapshot)
        if on_event is not None:
            on_event(copy.deepcopy(snapshot))

    emit("phase", phase="validate_prepared", step=0)
    cfg = validate_prepared(args, specification, protocol, prepared)
    emit("phase", phase="initialization", step=0)
    rng, state, models = P.initialize(args, cfg, prepared.training)
    _accept(state, {}, models)
    if _tree_hash((state.cal_obs_mean, state.cal_obs_std)) != _tree_hash(
        (prepared.cal_obs_mean, prepared.cal_obs_std)
    ):
        raise ValueError("initialized calibrator statistics differ from preparation")
    carry, completed = ((rng, state, jnp.int32(0)), 0)
    step_fn = P.make_train_step(args, cfg, models, prepared.training)
    emit(
        "prepared",
        step=0,
        metadata=prepared.metadata,
        metadata_sha256=prepared.metadata_sha256,
    )
    refreshes = {e.step: e for e in protocol.refresh_events}
    evals = {}
    for event in protocol.evaluation_events:
        evals.setdefault(event.step, []).append(event)

    def refresh_at(event):
        nonlocal carry
        emit("phase", phase="refresh", step=completed)
        key = jax.random.fold_in(jax.random.PRNGKey(event.seed), event.step)
        frozen = _tree_hash(carry[1])
        proposed, metrics, diagnostics = P.refresh(
            args,
            cfg,
            models,
            carry[1],
            prepared.heldout,
            key,
            heldout_ids=prepared.heldout_ids,
        )
        _accept(proposed, metrics, models)
        carry = (carry[0], proposed, carry[2])
        emit(
            "refresh",
            step=event.step,
            refresh_seed=event.seed,
            refresh_key=key,
            frozen_input_state_sha256=frozen,
            posterior_snapshot_sha256=_tree_hash(proposed.posterior),
            heldout_sha256=_tree_hash(prepared.heldout),
            posterior_weighting="wbcp_uniform",
            metrics=metrics,
            wbcp=diagnostics,
        )

    if 0 in refreshes:
        refresh_at(refreshes[0])
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
    for boundary in boundaries:
        if boundary == 0:
            continue
        emit("phase", phase="scan", step=completed)
        proposal, metrics = jax.lax.scan(
            step_fn, carry, None, length=boundary - completed
        )
        _accept(proposal[1], metrics, models)
        if int(proposal[2]) != boundary:
            raise RuntimeError("native update count differs from exact boundary")
        carry, completed = (proposal, boundary)
        emit(
            "accepted_scan",
            step=completed,
            metrics=metrics,
            training_rng_sha256=_tree_hash(carry[0]),
            state_sha256=_tree_hash(carry[1]),
        )
        if boundary in refreshes:
            refresh_at(refreshes[boundary])
        if on_checkpoint is not None and boundary in (
            10000,
            50000,
            protocol.num_updates,
        ):
            on_checkpoint(boundary, copy.deepcopy(carry))
        for event in evals.get(boundary, ()):
            emit("phase", phase="evaluation", step=completed)
            _accept(carry[1], {}, models)
            result = evaluate_episodes(
                models[0],
                carry[1].native.actor.params,
                prepared,
                event,
                dataset=args.dataset,
                max_workers=protocol.eval_workers,
                env_factory=env_factory,
            )
            _finite(
                [e["normalized_score"] for e in result["episodes"]], "evaluation scores"
            )
            if result["episode_seeds"] != list(event.episode_seeds) or result[
                "episode_count"
            ] != len(event.episode_seeds):
                raise ValueError("evaluation bank/count mismatch")
            evaluations.append(copy.deepcopy(result))
            emit(result.pop("kind"), **result)
    _accept(carry[1], {}, models)
    if completed != protocol.num_updates or len(evaluations) != len(
        protocol.evaluation_events
    ):
        raise RuntimeError("incomplete scheduled execution")
    emit(
        "completed",
        step=completed,
        final_episodes_actual=sum(
            (r["episode_count"] for r in evaluations if r["kind"] == "final")
        ),
    )
    return dict(
        state=carry[1],
        training_rng=carry[0],
        steps_completed=completed,
        events=records,
        evaluations=evaluations,
        metadata=copy.deepcopy(prepared.metadata),
    )
