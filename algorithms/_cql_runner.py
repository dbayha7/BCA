"""BCA cql runner.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from dataclasses import asdict, dataclass
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import cql as P
from _corl_training_reward_range import normalize_training_rewards

N, C = (P.BASE, P.BASE.C)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + " must be an explicit integer >= " + str(minimum))


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
                "explicit nonempty episode seed tuple required; no scalar expansion"
            )
        for seed in self.episode_seeds:
            _integer(seed, "episode seed")
            if seed > 2**32 - 1:
                raise ValueError("episode seed exceeds uint32 range")
        if len(set(self.episode_seeds)) != len(self.episode_seeds):
            raise ValueError("duplicate episode seeds within evaluation event")


@dataclass(frozen=True)
class RunProtocol:
    run_id: str
    seed: int
    num_updates: int
    scan_block_size: int
    eval_interval: int
    eval_workers: int
    eval_final_episodes: int
    refresh_steps: tuple
    calibration_target_size: int | None
    calibration_seed: int | None
    calibration_max_fraction: float | None
    reference_size: int | None
    reference_seed: int | None
    eval_periodic_episodes: int
    evaluation_events: tuple

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id is required")
        for name in ("seed", "eval_final_episodes"):
            _integer(getattr(self, name), name)
        for name in (
            "num_updates",
            "scan_block_size",
            "eval_interval",
            "eval_workers",
            "eval_periodic_episodes",
        ):
            _integer(getattr(self, name), name, 1)
        if self.num_updates % self.eval_interval:
            raise ValueError(
                "native loop requires num_updates divisible by eval_interval; no silent truncation"
            )
        if not isinstance(self.refresh_steps, tuple):
            raise ValueError(
                "refresh_steps must be an explicit tuple, including () for native controls"
            )
        for step in self.refresh_steps:
            _integer(step, "refresh step")
        if tuple(sorted(set(self.refresh_steps))) != self.refresh_steps:
            raise ValueError("refresh steps must be unique and increasing")
        if self.refresh_steps and self.refresh_steps[-1] >= self.num_updates:
            raise ValueError("refresh must precede a subsequent training update")
        for name in ("calibration_target_size", "reference_size"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name, 1)
        for name in ("calibration_seed", "reference_seed"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name)
        f = self.calibration_max_fraction
        if f is not None and (isinstance(f, bool) or not np.isfinite(f) or (not 0 < f < 1)):
            raise ValueError("calibration_max_fraction must be explicitly in (0,1)")
        if type(self.evaluation_events) is not tuple or any(
            (type(e) is not EvaluationEvent for e in self.evaluation_events)
        ):
            raise TypeError("explicit typed evaluation event tuple required")
        ordering = [(e.step, e.kind == "final") for e in self.evaluation_events]
        if ordering != sorted(set(ordering)):
            raise ValueError(
                "evaluation events must be unique and ordered, periodic before final"
            )
        periodic = [e for e in self.evaluation_events if e.kind == "periodic"]
        if [e.step for e in periodic] != list(
            range(self.eval_interval, self.num_updates + 1, self.eval_interval)
        ):
            raise ValueError("evaluation events must match every periodic boundary")
        if any((len(e.episode_seeds) != self.eval_periodic_episodes for e in periodic)):
            raise ValueError("periodic episode count disagrees with explicit schedule")
        final = [e for e in self.evaluation_events if e.kind == "final"]
        if self.eval_final_episodes:
            if (
                len(final) != 1
                or final[0].step != self.num_updates
                or len(final[0].episode_seeds) != self.eval_final_episodes
            ):
                raise ValueError(
                    "one final event with exact episode count at final step required"
                )
        elif final:
            raise ValueError("zero final count cannot carry final evaluation events")
        all_seeds = [s for e in self.evaluation_events for s in e.episode_seeds]
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError(
                "episode seeds must be disjoint across periodic/final events in an arm"
            )


def validate_protocol(args, config, protocol):
    if not isinstance(protocol, RunProtocol) or not isinstance(config, P.Config):
        raise TypeError("typed protocol and posterior configuration required")
    RunProtocol(**{k: getattr(protocol, k) for k in protocol.__dataclass_fields__})
    for name in ("seed", "num_updates", "eval_interval", "eval_workers", "eval_final_episodes"):
        if getattr(args, name) != getattr(protocol, name):
            raise ValueError("native Args and protocol disagree on " + name)
    split = (
        protocol.calibration_target_size,
        protocol.calibration_seed,
        protocol.calibration_max_fraction,
    )
    if config.arm == "native_full":
        if any((x is not None for x in split)):
            raise ValueError("native_full cannot reserve calibration data")
    elif any((x is None for x in split)):
        raise ValueError("reserved arms require all split inputs")
    reference = (protocol.reference_size, protocol.reference_seed)
    if config.arm == "posterior":
        if not protocol.refresh_steps or any((x is None for x in reference)):
            raise ValueError(
                "posterior requires an explicit refresh schedule and training reference"
            )
    elif protocol.refresh_steps or any((x is not None for x in reference)):
        raise ValueError("native controls cannot carry posterior refresh/reference inputs")
    C.check_reward_transform(
        "corl_cql", args.dataset, args.reward_transform, args.reward_scale, args.reward_bias
    )
    N._published_config.check_published_config(
        "corl_cql", args.dataset, args, allow_off_config=args.allow_off_config
    )
    if (
        C.is_antmaze(args.dataset)
        and args.normalize_reward
        and (args.reward_transform == "cql_scale_bias")
        and (args.cql_reward_scale != 1.0 or args.cql_reward_bias != 0.0)
    ):
        raise ValueError("native antmaze affine reward tail would be applied twice")
    if args.log:
        raise ValueError(
            "this callable runner has no implicit wandb sink; use on_event explicitly"
        )


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
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


def _array_hash(value):
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise ValueError("object arrays cannot provide reproducible raw-data identities")
    h = hashlib.sha256(json.dumps([a.dtype.str, a.shape]).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _finite(value, label):
    for leaf in jax.tree_util.tree_leaves(value):
        a = np.asarray(leaf)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError(label + " contains a nonfinite value")


def _settings(args, config, protocol):
    return _json_value(
        {"args": asdict(args), "attachment": asdict(config), "protocol": asdict(protocol)}
    )


def _prepared_hashes(training, heldout, reference, mean, std, max_action, max_steps):
    pools = {"training": training, "heldout": heldout, "reference": reference}
    return {
        **{
            name: (
                None
                if pool is None
                else {field: _array_hash(x) for field, x in zip(pool._fields, pool)}
            )
            for name, pool in pools.items()
        },
        "obs_mean": _array_hash(mean),
        "obs_std": _array_hash(std),
        "max_action": float(max_action),
        "max_episode_steps": max_steps,
    }


def source_identity():
    from _provenance import source_identity as identify

    return identify()


def score_transform_identity(dataset):
    minimum, maximum = (
        C.d4rl.infos.REF_MIN_SCORE[dataset],
        C.d4rl.infos.REF_MAX_SCORE[dataset],
    )
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("invalid D4RL score reference range")
    return {
        "dataset": dataset,
        "reference_min": float(minimum),
        "reference_max": float(maximum),
        "formula": "100 * (raw_return - reference_min) / (reference_max - reference_min)",
        "reference_source_sha256": hashlib.sha256(
            Path(C.d4rl.infos.__file__).read_bytes()
        ).hexdigest(),
        "score_function_source_sha256": hashlib.sha256(
            Path(inspect.getsourcefile(C.d4rl.get_normalized_score)).read_bytes()
        ).hexdigest(),
    }


def _convert(raw, max_episode_steps):
    required = ("observations", "actions", "rewards", "terminals")
    if any((k not in raw for k in required)):
        raise ValueError("raw D4RL observations/actions/rewards/terminals required")
    n = len(raw["rewards"])
    if n < 2 or any((len(raw[k]) != n for k in required)):
        raise ValueError("raw arrays must be aligned and nonempty")
    _finite([raw[k] for k in required], "raw dataset")
    if (
        np.asarray(raw["observations"]).ndim != 2
        or np.asarray(raw["actions"]).ndim != 2
        or np.asarray(raw["rewards"]).shape != (n,)
        or (np.asarray(raw["terminals"]).shape != (n,))
        or (not np.all(np.isin(raw["terminals"], (0, 1))))
    ):
        raise ValueError("raw transition shapes/terminal indicators are invalid")
    if "timeouts" in raw and (
        np.asarray(raw["timeouts"]).shape != (n,)
        or not np.all(np.isin(raw["timeouts"], (0, 1)))
    ):
        raise ValueError("timeouts must be an aligned binary vector")
    _integer(max_episode_steps, "max_episode_steps", 1)
    rows, ids, ep_step, episode = ([], [], 0, 0)
    for i in range(n - 1):
        timeout = (
            bool(raw["timeouts"][i]) if "timeouts" in raw else ep_step == max_episode_steps - 1
        )
        terminal = bool(raw["terminals"][i])
        if timeout:
            ep_step, episode = (0, episode + 1)
            continue
        rows.append(i)
        ids.append(episode)
        if terminal:
            ep_step, episode = (0, episode + 1)
        ep_step += 1
    rows, ids = (np.asarray(rows, np.int64), np.asarray(ids, np.int64))
    converted = C.d4rl.qlearning_dataset(
        SimpleNamespace(_max_episode_steps=max_episode_steps), dataset=raw
    )
    expected = {
        "observations": np.asarray(raw["observations"])[rows].astype(np.float32),
        "next_observations": np.asarray(raw["observations"])[rows + 1].astype(np.float32),
        "actions": np.asarray(raw["actions"])[rows].astype(np.float32),
        "rewards": np.asarray(raw["rewards"])[rows].astype(np.float32),
        "terminals": np.asarray(raw["terminals"])[rows].astype(bool),
    }
    if not len(rows) or any((not np.array_equal(converted[k], v) for k, v in expected.items())):
        raise ValueError(
            "actual D4RL converter disagrees with raw-row map; refuse unknown ordering"
        )
    return ({k: np.array(v, copy=True) for k, v in converted.items()}, rows, ids)


@dataclass(frozen=True)
class PreparedData:
    training: object
    heldout: object
    reference: object
    obs_mean: object
    obs_std: object
    max_action: float
    max_episode_steps: int
    metadata: dict
    metadata_sha256: str


def prepare(raw, args, config, protocol, *, max_action, max_episode_steps):
    validate_protocol(args, config, protocol)
    if not np.isfinite(max_action) or max_action <= 0:
        raise ValueError("max_action must be finite and positive")
    if config.arm == "posterior" and max_action != 1.0:
        raise ValueError("accepted posterior density requires max_action=1")
    converted, raw_rows, episode_ids = _convert(raw, max_episode_steps)
    n = len(raw_rows)
    if config.arm == "native_full":
        train, cal, reservation = (np.arange(n), np.empty(0, np.int64), None)
    else:
        train, cal, reservation = P.reserve_pool(
            C.Transition(
                converted["observations"],
                converted["actions"],
                converted["rewards"],
                converted["next_observations"],
                converted["terminals"],
            ),
            protocol.calibration_target_size,
            protocol.calibration_seed,
            max_fraction=protocol.calibration_max_fraction,
            episode_ids=episode_ids,
        )
        if set(episode_ids[train]) & set(episode_ids[cal]):
            raise ValueError("training and heldout episode IDs overlap")
    converted["rewards"] = np.asarray(converted["rewards"], np.float64)
    reward_normalization = {"mode": "disabled"}
    if args.normalize_reward:
        if config.arm != "native_full" and any(
            (s in args.dataset for s in ("halfcheetah", "hopper", "walker2d"))
        ):
            converted["rewards"], reward_normalization = normalize_training_rewards(
                converted["rewards"],
                converted["terminals"],
                train,
                max_episode_steps=max_episode_steps,
                reward_scale=args.cql_reward_scale,
                reward_bias=args.cql_reward_bias,
            )
            reward_normalization["fit_raw_indices"] = raw_rows[
                reward_normalization["fit_converted_indices"]
            ].tolist()
        else:
            converted = C.modify_reward(
                converted,
                args.dataset,
                max_episode_steps=max_episode_steps,
                reward_scale=args.cql_reward_scale,
                reward_bias=args.cql_reward_bias,
            )
            reward_normalization = {"mode": "original_native_modify_reward"}
    converted = C.apply_reward_transform(
        converted,
        args.dataset,
        args.reward_transform,
        args.reward_scale,
        args.reward_bias,
        legacy_print=True,
    )
    obs = np.asarray(converted["observations"], np.float32)
    nxt = np.asarray(converted["next_observations"], np.float32)
    if args.normalize:
        mean, std = C.compute_mean_std(
            obs if config.arm == "native_full" else obs[train], eps=0.001
        )
        obs, nxt = (C.normalize_states(obs, mean, std), C.normalize_states(nxt, mean, std))
    else:
        mean, std = (np.zeros(obs.shape[-1], np.float32), np.ones(obs.shape[-1], np.float32))
    all_data = C.Transition(
        jnp.asarray(obs),
        jnp.asarray(converted["actions"], jnp.float32),
        jnp.asarray(converted["rewards"], jnp.float32),
        jnp.asarray(nxt),
        jnp.asarray(converted["terminals"], jnp.float32),
    )
    _finite((all_data, mean, std), "preprocessed data/statistics")
    if not np.all(std > 0):
        raise ValueError("observation standard deviations must be positive")
    training = P.select_training_pool(
        all_data, config.arm, None if config.arm == "native_full" else train
    )
    heldout = jax.tree_util.tree_map(lambda x: x[cal], all_data) if len(cal) else None
    reference, reference_rows = (None, np.empty(0, np.int64))
    if config.arm == "posterior":
        if protocol.reference_size > len(train):
            raise ValueError("training reference cannot exceed training pool")
        local = np.sort(
            np.random.default_rng(protocol.reference_seed).choice(
                len(train), protocol.reference_size, replace=False
            )
        )
        reference = jax.tree_util.tree_map(lambda x: x[local], training)
        reference_rows = train[local]
    hashes = {k: _array_hash(raw[k]) for k in sorted(raw)}
    settings = _settings(args, config, protocol)
    metadata = {
        "schema": "native-cql-prepared-explicit-evaluation-v2",
        "settings": settings,
        "settings_sha256": _digest(settings),
        "source_files": source_identity(),
        "raw_array_hashes": hashes,
        "raw_dataset_sha256": _digest(hashes),
        "raw_rows": int(len(raw["rewards"])),
        "converted_rows": n,
        "conversion_raw_indices": raw_rows.tolist(),
        "converted_episode_ids": episode_ids.tolist(),
        "training_converted_indices": train.tolist(),
        "heldout_converted_indices": cal.tolist(),
        "training_raw_indices": raw_rows[train].tolist(),
        "heldout_raw_indices": raw_rows[cal].tolist(),
        "reference_raw_indices": raw_rows[reference_rows].tolist(),
        "reservation": reservation,
        "boundary_note": "Disjoint transition/episode blocks; retained terminal next_obs can name the next raw episode. This is not a claim of independent observations.",
        "preprocessing_order": [
            "D4RL conversion and verified raw mapping",
            "whole-block reservation if requested",
            "native reward normalization/affine tail if enabled",
            "registered antmaze transform",
            "observation statistics from full data or training complement",
            "apply identical statistics to train/heldout/evaluation",
        ],
        "normalization_fit": (
            "full_data" if config.arm == "native_full" else "training_complement"
        ),
        "reward_normalization": reward_normalization,
        "obs_mean": mean.tolist(),
        "obs_std": std.tolist(),
        "prepared_array_hashes": {
            name: _array_hash(x) for name, x in zip(all_data._fields, all_data)
        },
        "reference_selection": "fixed seeded uniform subset of training identities, sorted in dataset order",
        "historical_vanilla_bca": {
            "required_separate_comparator": True,
            "family": "house cql_bca IW off with retained recipe",
            "executed_by_this_runner": False,
        },
        "published_config_differences": _json_value(
            N._published_config.diff_against_published("corl_cql", args.dataset, args)
        ),
        "max_action": float(max_action),
        "max_episode_steps": max_episode_steps,
    }
    metadata["evaluation_score_transform"] = score_transform_identity(args.dataset)
    metadata["evaluation_events_sha256"] = _digest(
        [asdict(e) for e in protocol.evaluation_events]
    )
    metadata["run_input_hashes"] = _prepared_hashes(
        training, heldout, reference, mean, std, max_action, max_episode_steps
    )
    return PreparedData(
        training,
        heldout,
        reference,
        jnp.asarray(mean),
        jnp.asarray(std),
        float(max_action),
        max_episode_steps,
        metadata,
        _digest(metadata),
    )


def load_and_prepare(args, config, protocol):
    validate_protocol(args, config, protocol)
    env = C.gym.make(args.dataset)
    try:
        hi, lo = (np.asarray(env.action_space.high), np.asarray(env.action_space.low))
        if hi.ndim != 1 or not np.all(hi == hi[0]) or (not np.all(lo == -hi)):
            raise ValueError("native CQL runner requires uniform symmetric action bounds")
        raw = env.get_dataset()
        if (
            np.asarray(raw["observations"]).shape[1:] != env.observation_space.shape
            or np.asarray(raw["actions"]).shape[1:] != env.action_space.shape
        ):
            raise ValueError("dataset dimensions differ from native environment spaces")
        return prepare(
            raw,
            args,
            config,
            protocol,
            max_action=float(hi[0]),
            max_episode_steps=int(env._max_episode_steps),
        )
    finally:
        env.close()


class IncompleteEvaluation(RuntimeError):

    def __init__(self, record):
        super().__init__("episode cap reached without a completion signal")
        self.record = copy.deepcopy(record)


def _evaluation_binding(prepared, event, dataset):
    if type(event) is not EvaluationEvent:
        raise TypeError("typed evaluation event required; no scalar seed expansion")
    if _digest(prepared.metadata) != prepared.metadata_sha256:
        raise ValueError("evaluation prepared metadata digest mismatch")
    m = prepared.metadata
    if (
        _digest(m["settings"]) != m["settings_sha256"]
        or dataset != m["settings"]["args"]["dataset"]
    ):
        raise ValueError("evaluation dataset/settings identity mismatch")
    events = m["settings"]["protocol"]["evaluation_events"]
    if (
        _digest(events) != m["evaluation_events_sha256"]
        or _json_value(asdict(event)) not in events
    ):
        raise ValueError("evaluation event differs from prepared schedule")
    hashes = m["run_input_hashes"]
    if (
        hashes["obs_mean"] != _array_hash(prepared.obs_mean)
        or hashes["obs_std"] != _array_hash(prepared.obs_std)
        or hashes["max_action"] != prepared.max_action
        or (hashes["max_episode_steps"] != prepared.max_episode_steps)
    ):
        raise ValueError("evaluation normalization/bounds/cap identity mismatch")
    transform = score_transform_identity(dataset)
    if transform != m["evaluation_score_transform"]:
        raise ValueError("evaluation score-reference identity mismatch")
    return transform


def evaluate(
    actor, params, prepared, event, *, dataset, max_workers, env_factory=None, env=None
):
    transform = _evaluation_binding(prepared, event, dataset)
    _integer(max_workers, "maximum evaluation workers", 1)
    if max_workers != prepared.metadata["settings"]["protocol"]["eval_workers"]:
        raise ValueError("worker count differs from prepared protocol")
    if env is not None and env_factory is not None:
        raise ValueError("choose supplied environment or owned environment factory")
    if env is not None and (
        len(event.episode_seeds) % max_workers or getattr(env, "num_envs", None) != max_workers
    ):
        raise ValueError(
            "caller environment must match all exact batch sizes; use factory for short batches"
        )
    _finite((params, prepared.obs_mean, prepared.obs_std), "evaluation actor/statistics")
    if np.any(np.asarray(prepared.obs_std) <= 0):
        raise ValueError("evaluation normalization scale must be positive")
    factory, episodes = (C.make_eval_env if env_factory is None else env_factory, [])

    @jax.jit
    def policy(observations):
        normalized = (observations - prepared.obs_mean) / prepared.obs_std
        mean, log_std = jax.vmap(lambda x: actor.apply(params, x))(normalized)
        return (prepared.max_action * jnp.tanh(mean), mean, log_std, normalized)

    for start in range(0, len(event.episode_seeds), max_workers):
        seeds = event.episode_seeds[start : start + max_workers]
        workers = len(seeds)
        current = factory(dataset, workers) if env is None else env
        try:
            if getattr(current, "num_envs", None) != workers:
                raise ValueError("constructed vector environment has wrong batch size")
            current.seed(list(seeds))
            reset = current.reset()
            obs = np.asarray(
                reset[0] if isinstance(reset, tuple) and len(reset) == 2 else reset
            )
            returned, total = (np.zeros(workers, bool), np.zeros(workers))
            lengths = np.zeros(workers, np.int64)
            for _ in range(prepared.max_episode_steps):
                _finite(obs, "evaluation observations")
                if obs.shape != (workers, prepared.training.obs.shape[1]):
                    raise ValueError("evaluation observation shape mismatch")
                action, mean, log_std, normalized = policy(jnp.asarray(obs))
                _finite((action, mean, log_std, normalized), "evaluation policy outputs")
                action = np.asarray(action)
                if action.shape != (workers, prepared.training.action.shape[1]) or np.any(
                    np.abs(action) > prepared.max_action
                ):
                    raise ValueError("evaluation action shape/bounds mismatch")
                result = current.step(action)
                if len(result) == 4:
                    obs, reward, done, _info = result
                    indicators = [np.asarray(done)]
                elif len(result) == 5:
                    obs, reward, terminated, truncated, _info = result
                    indicators = [np.asarray(terminated), np.asarray(truncated)]
                else:
                    raise ValueError("unsupported environment step contract")
                obs, reward = (np.asarray(obs), np.asarray(reward))
                _finite((obs, reward, indicators), "evaluation environment outputs")
                if (
                    obs.shape != (workers, prepared.training.obs.shape[1])
                    or reward.shape != (workers,)
                    or any(
                        (
                            x.shape != (workers,) or not np.all(np.isin(x, (0, 1)))
                            for x in indicators
                        )
                    )
                ):
                    raise ValueError("evaluation output shape or completion indicator mismatch")
                done = np.logical_or.reduce([x.astype(bool) for x in indicators])
                total += reward * ~returned
                lengths += ~returned
                returned |= done
                if returned.all():
                    break
            if not returned.all():
                raise IncompleteEvaluation(
                    {
                        "kind": "incomplete_evaluation",
                        "step": event.step,
                        "episode_seeds": list(seeds),
                        "completed": returned.tolist(),
                        "lengths": lengths.tolist(),
                        "partial_returns_not_scores": total.tolist(),
                    }
                )
            _finite(total, "evaluation returns")
            if score_transform_identity(dataset) != transform:
                raise ValueError("score-reference identity changed during evaluation")
            scores = np.asarray(C.normalized_scores(dataset, total))
            _finite(scores, "evaluation returns/scores")
            if scores.shape != total.shape:
                raise ValueError("normalized-score shape mismatch")
            episodes.extend(
                (
                    {
                        "episode_index": start + i,
                        "seed": seed,
                        "worker": i,
                        "batch_start": start,
                        "length": int(lengths[i]),
                        "completed": True,
                        "return": float(total[i]),
                        "score": float(scores[i]),
                    }
                    for i, seed in enumerate(seeds)
                )
            )
        finally:
            if env is None:
                current.close()
    return {
        "kind": event.kind,
        "step": event.step,
        "episode_count": len(episodes),
        "episode_seeds": list(event.episode_seeds),
        "episodes": episodes,
        "returns": [e["return"] for e in episodes],
        "scores": [e["score"] for e in episodes],
        "event_sha256": _digest(asdict(event)),
        "score_transform": transform,
    }


def _accept(state, metrics):
    P.require_valid(metrics)
    _finite(state.native, "native state/optimizer")
    if state.calibrator is not None:
        _finite((state.calibrator, state.residual_scale), "calibrator state/optimizer/EMA")
    _finite(metrics, "training/refresh metrics")


def run_prepared(
    args,
    config,
    protocol,
    prepared,
    *,
    env=None,
    env_factory=None,
    on_event=None,
    on_checkpoint=None
):
    validate_protocol(args, config, protocol)
    if _digest(prepared.metadata) != prepared.metadata_sha256:
        raise ValueError("prepared provenance metadata differs from its complete digest")
    if _digest(_settings(args, config, protocol)) != prepared.metadata["settings_sha256"]:
        raise ValueError("run settings differ from the prepared data contract")
    if source_identity() != prepared.metadata["source_files"]:
        raise ValueError("current source files differ from the prepared provenance")
    if (
        _prepared_hashes(
            prepared.training,
            prepared.heldout,
            prepared.reference,
            prepared.obs_mean,
            prepared.obs_std,
            prepared.max_action,
            prepared.max_episode_steps,
        )
        != prepared.metadata["run_input_hashes"]
    ):
        raise ValueError("prepared run arrays/statistics differ from their recorded identities")
    if (
        _digest([asdict(e) for e in protocol.evaluation_events])
        != prepared.metadata["evaluation_events_sha256"]
    ):
        raise ValueError("prepared evaluation schedule differs from run events")
    if env is not None and (
        env_factory is not None
        or getattr(env, "num_envs", None) != protocol.eval_workers
        or any(
            (len(e.episode_seeds) % protocol.eval_workers for e in protocol.evaluation_events)
        )
    ):
        raise ValueError("supplied environment cannot execute the exact declared batches")
    args, config, protocol = copy.deepcopy((args, config, protocol))
    prepared = PreparedData(
        **{**prepared.__dict__, "metadata": copy.deepcopy(prepared.metadata)}
    )
    rng, state, models = P.initialize(
        args,
        config,
        prepared.training.obs.shape[1],
        prepared.training.action.shape[1],
        prepared.max_action,
    )
    _accept(state, {})
    step_fn = P.make_train_step(args, config, models, prepared.training, prepared.max_action)
    carry = (rng, state, jnp.int32(0))
    records, evaluations = ([], [])

    def emit(record):
        snapshot = copy.deepcopy(record)
        records.append(snapshot)
        if on_event is not None:
            on_event(copy.deepcopy(snapshot))

    def refresh_at(step):
        nonlocal carry
        refresh_key = jax.random.fold_in(
            jax.random.fold_in(jax.random.PRNGKey(args.seed), 1380271698), step
        )
        updated, metrics = P.refresh(
            args,
            config,
            models,
            carry[1],
            prepared.reference,
            prepared.heldout,
            refresh_key,
            prepared.max_action,
        )
        _accept(updated, metrics)
        carry = (carry[0], updated, carry[2])
        emit(
            {
                "kind": "refresh",
                "step": step,
                "metrics": {k: np.asarray(v).tolist() for k, v in metrics.items()},
            }
        )

    def evaluate_at(event):
        _accept(carry[1], {})
        record = evaluate(
            models[0],
            carry[1].native.actor.params,
            prepared,
            event,
            dataset=args.dataset,
            max_workers=protocol.eval_workers,
            env_factory=env_factory,
            env=env,
        )
        evaluations.append(copy.deepcopy(record))
        emit(record)

    try:
        emit({"kind": "prepared", "step": 0, "metadata": prepared.metadata})
        if 0 in protocol.refresh_steps:
            refresh_at(0)
        boundaries = sorted(
            set(range(protocol.scan_block_size, protocol.num_updates, protocol.scan_block_size))
            | set(
                range(protocol.eval_interval, protocol.num_updates + 1, protocol.eval_interval)
            )
            | set(protocol.refresh_steps)
            | {protocol.num_updates}
        )
        completed = 0
        for step in boundaries:
            if step == 0:
                continue
            proposal, metrics = jax.lax.scan(step_fn, carry, None, length=step - completed)
            _accept(proposal[1], metrics)
            if int(proposal[2]) != step:
                raise RuntimeError("native step counter differs from declared boundary")
            carry, completed = (proposal, step)
            emit(
                {
                    "kind": "accepted_scan",
                    "step": step,
                    "metrics_last": {k: np.asarray(v[-1]).tolist() for k, v in metrics.items()},
                }
            )
            if step in protocol.refresh_steps:
                refresh_at(step)
            if step % protocol.eval_interval == 0:
                evaluate_at(
                    next(
                        (
                            e
                            for e in protocol.evaluation_events
                            if e.kind == "periodic" and e.step == step
                        )
                    )
                )
            if on_checkpoint is not None and step in (10000, 50000, protocol.num_updates):
                on_checkpoint(step, copy.deepcopy(carry))
        _accept(carry[1], {})
        for event in protocol.evaluation_events:
            if event.kind == "final":
                evaluate_at(event)
        _accept(carry[1], {})
        emit(
            {
                "kind": "completed",
                "step": completed,
                "final_episodes_requested": protocol.eval_final_episodes,
                "final_episodes_actual": sum(
                    (e["episode_count"] for e in evaluations if e["kind"] == "final")
                ),
                "scored_final": bool(protocol.eval_final_episodes),
            }
        )
        return {
            "state": carry[1],
            "training_rng": carry[0],
            "steps_completed": completed,
            "evaluations": evaluations,
            "events": records,
            "metadata": copy.deepcopy(prepared.metadata),
        }
    except Exception as error:
        emit(
            {
                "kind": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "evaluation_failure": getattr(error, "record", None),
            }
        )
        raise
