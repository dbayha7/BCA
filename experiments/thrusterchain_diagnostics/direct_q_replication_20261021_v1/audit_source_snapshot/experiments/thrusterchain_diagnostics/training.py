"""Native JAX TD3+BC/BCA training for the new finite-horizon experiment.

This adapter imports the public ``algorithms.td3_bc_bca`` API, including its
initializer, train step and refresh. It does not import historical experiment
approval workflows. ``allow_off_config=True`` declares a new experiment, not a
published benchmark reproduction. The host arm has the exact same reserved
training pool, normalizer, initialization, batches and target-noise stream.

Only a learner-view dictionary is accepted. Evaluator fields, evaluation splits
and oracle information are rejected. Calibration trajectories never train any
network or the normalizer. A retained row remains a complete original transition;
no row shifting, inverse-propensity correction or budget top-ups occur here.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np


SCHEMA = "thrusterchain-native-training-v1"
OBS_DIM, ACTION_DIM = 15, 4
LEARNER_KEYS = frozenset(("episode_id", "time", "observation", "action", "reward",
    "next_observation", "action_probability", "retention_propensity", "terminated",
    "truncated", "bootstrap_mask", "split", "seed", "provenance", "row_id"))
REQUIRED_KEYS = LEARNER_KEYS - {"row_id"}


@dataclass(frozen=True)
class TrainingSettings:
    updates: int = 100_000
    chunk: int = 2_500
    batch_size: int = 256
    horizon: int = 100
    gamma: float = 1.0
    lr: float = 0.0003
    tau: float = 0.005
    alpha: float = 2.5
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    calibration_size: int = 128
    rows_per_episode: int = 1
    reservation_seed: int = 911
    max_reserved_fraction: float = 0.25
    bca_warmup: int = 10_000
    bca_refresh: int = 5_000
    miscoverage: float = 0.1
    credibility: float = 0.95
    draws: int = 1000
    blend: float = 0.5
    cal_lr: float = 0.001
    cal_beta: float = 20.0
    width_penalty: float = 0.005
    scale_ema: float = 0.99

    def __post_init__(self):
        for key in ("updates", "chunk", "batch_size", "horizon", "policy_freq",
                    "calibration_size", "rows_per_episode", "bca_warmup", "bca_refresh", "draws"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(key + " must be a positive integer")
        if not isinstance(self.reservation_seed, int) or self.reservation_seed < 0:
            raise ValueError("reservation_seed must be a nonnegative integer")
        for key in ("gamma", "lr", "tau", "alpha", "policy_noise", "noise_clip",
                    "max_reserved_fraction", "miscoverage", "credibility", "blend",
                    "cal_lr", "cal_beta", "width_penalty", "scale_ema"):
            if not np.isfinite(getattr(self, key)):
                raise ValueError(key + " must be finite")
        if self.horizon != 100 or self.gamma != 1.0:
            raise ValueError("this experiment is frozen to H=100, gamma=1")
        if not 0 < self.max_reserved_fraction < 1:
            raise ValueError("reserved fraction must be in (0,1)")
        if min(self.lr, self.alpha, self.cal_lr, self.cal_beta) <= 0:
            raise ValueError("learning rates and objective coefficients must be positive")
        if not 0 < self.tau <= 1 or min(self.policy_noise, self.noise_clip) < 0:
            raise ValueError("invalid target update configuration")
        if not (0 < self.miscoverage < 1 and 0 < self.credibility < 1 and 0 < self.blend <= 1):
            raise ValueError("invalid posterior configuration")
        if self.width_penalty < 0 or not 0 <= self.scale_ema < 1:
            raise ValueError("invalid scale fitting configuration")


def _stack():
    """Lazy import: portable diagnostics still work without the training stack."""
    import jax
    import jax.numpy as jnp
    import algorithms.td3_bc as base
    import algorithms.td3_bc_bca as posterior
    from calibration.reference import WBCPConfig, certifiable
    from runtime.networks import Transition
    from flax import serialization
    return jax, jnp, base, posterior, WBCPConfig, certifiable, Transition, serialization


def _hash_arrays(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_learner_rows(learner_rows, settings):
    """Validate the finite-horizon contract before constructing any JAX arrays."""
    if not isinstance(learner_rows, dict):
        raise TypeError("pass learner_rows only, never a ContinuousDataset/evaluator object")
    if set(learner_rows) - LEARNER_KEYS:
        raise ValueError("non-learner fields are forbidden: " + str(sorted(set(learner_rows) - LEARNER_KEYS)))
    if REQUIRED_KEYS - set(learner_rows):
        raise ValueError("missing learner fields: " + str(sorted(REQUIRED_KEYS - set(learner_rows))))
    rows = {key: np.asarray(value) for key, value in learner_rows.items()}
    n = len(rows["reward"])
    if n < 2 or any(value.ndim < 1 or len(value) != n for value in rows.values()):
        raise ValueError("all learner fields must have aligned nonempty row axes")
    for key, shape in (("observation", (n, OBS_DIM)), ("next_observation", (n, OBS_DIM)),
                       ("action", (n, ACTION_DIM))):
        if rows[key].shape != shape or not np.isfinite(rows[key]).all():
            raise ValueError(key + " has invalid shape or nonfinite data")
    for key in REQUIRED_KEYS - {"observation", "next_observation", "action", "split", "provenance"}:
        if rows[key].shape != (n,) or not np.isfinite(rows[key]).all():
            raise ValueError(key + " must be a finite row vector")
    if rows["split"].shape != (n,) or not np.all(rows["split"] == "train"):
        raise ValueError("only split='train' may enter training/calibration preparation")
    if rows["provenance"].shape != (n,) or any("fixed" in str(x) for x in rows["provenance"]):
        raise ValueError("training requires trajectory provenance")
    if not np.all(np.abs(rows["action"]) <= 1):
        raise ValueError("actions must lie in [-1,1]^4")
    for key in ("episode_id", "time", "seed"):
        if not np.issubdtype(rows[key].dtype, np.integer) or np.any(rows[key] < 0):
            raise ValueError(key + " must be nonnegative integers")
    ids, times = rows["episode_id"], rows["time"]
    if np.any(ids[1:] < ids[:-1]) or np.any(times >= settings.horizon):
        raise ValueError("episodes must be ordered and times must lie inside the horizon")
    same = ids[1:] == ids[:-1]
    if np.any(same & (times[1:] <= times[:-1])):
        raise ValueError("each episode must have unique increasing retained times")
    obs, nxt = rows["observation"], rows["next_observation"]
    if not np.isin(obs[:, 13], (0, 1)).all() or not np.array_equal(obs[:, 13], nxt[:, 13]):
        raise ValueError("observed regime must be 0/1 and remain fixed in each transition")
    if np.any(same & (obs[1:, 13] != obs[:-1, 13])):
        raise ValueError("regime must remain fixed within each trajectory")
    if not (np.allclose(obs[:, 14], (settings.horizon - times) / settings.horizon, atol=1e-7, rtol=0)
            and np.allclose(nxt[:, 14], (settings.horizon - times - 1) / settings.horizon, atol=1e-7, rtol=0)):
        raise ValueError("observations must include the correct remaining horizon fraction")
    ends = times == settings.horizon - 1
    if (np.any(rows["terminated"] != 0) or not np.array_equal(rows["truncated"].astype(bool), ends)
            or not np.array_equal(rows["bootstrap_mask"], (~ends).astype(float))):
        raise ValueError("horizon truncations require bootstrap_mask=0; all other masks must be 1")
    consecutive = same & (times[1:] == times[:-1] + 1)
    if not np.array_equal(nxt[:-1][consecutive], obs[1:][consecutive]):
        raise ValueError("adjacent retained transitions do not chain exactly")
    for key in ("action_probability", "retention_propensity"):
        if np.any((rows[key] <= 0) | (rows[key] > 1)):
            raise ValueError("retained rows need positive valid " + key)
    return rows


@dataclass
class PreparedTraining:
    train: object
    bank: object
    training_indices: np.ndarray
    withheld_indices: np.ndarray
    calibration_indices: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    manifest: dict
    bca_refusal: str | None


def prepare_training(learner_rows, settings=TrainingSettings(), *, data_metadata=None,
                     calibration_episode_ids=None):
    """Reserve trajectories once; the returned pool is shared by both method arms.

    Reservation failures are explicit, not repaired by new simulation. The host
    can still run on a deterministic reserved complement, while BCA reports its
    refusal. Metadata is optional and never used to construct learner inputs.
    """
    rows = validate_learner_rows(learner_rows, settings)
    if data_metadata is not None:
        if data_metadata.get("budget_topups", 0) != 0:
            raise ValueError("budget top-ups are forbidden")
        if data_metadata.get("split", "train") != "train":
            raise ValueError("metadata must describe the train split")
        objective = data_metadata.get("objective", {})
        if objective and (objective.get("horizon") != settings.horizon or objective.get("gamma") != settings.gamma):
            raise ValueError("metadata objective differs from the frozen objective")
        if objective.get("terminal_bootstrap", False) is not False or objective.get("terminal_reward", 0) != 0:
            raise ValueError("metadata must specify no terminal bootstrap or reward")
        if objective.get("reward", "-(sum(state[:10]**2)+0.3*sum(action**2))") != "-(sum(state[:10]**2)+0.3*sum(action**2))":
            raise ValueError("metadata reward differs from the physical raw reward")
    jax, jnp, _, native, WBCPConfig, certifiable, Transition, _ = _stack()
    n = len(rows["reward"])
    raw = Transition(rows["observation"], rows["action"], rows["reward"],
                     rows["next_observation"], 1.0 - rows["bootstrap_mask"])
    refusal = None
    if calibration_episode_ids is not None:
        reserved = np.asarray(calibration_episode_ids)
        if (reserved.ndim != 1 or not np.issubdtype(reserved.dtype, np.integer) or not len(reserved)
                or len(np.unique(reserved)) != len(reserved) or np.any(reserved < 0)):
            raise ValueError("calibration_episode_ids must be unique nonnegative integers chosen before retention")
        reserved = np.sort(reserved.astype(np.int64))
        withheld_idx = np.flatnonzero(np.isin(rows["episode_id"], reserved)).astype(np.int32)
        train_idx = np.flatnonzero(~np.isin(rows["episode_id"], reserved)).astype(np.int32)
        selected, absent = [], []
        for episode in reserved:
            eligible = np.flatnonzero(rows["episode_id"] == episode)
            if not len(eligible):
                absent.append(int(episode))
                continue
            rng = np.random.default_rng(np.random.SeedSequence([settings.reservation_seed, int(episode), 1128352841]))
            selected.extend(rng.choice(eligible, min(settings.rows_per_episode, len(eligible)), replace=False))
        bank_idx = np.sort(np.asarray(selected, dtype=np.int32))
        if len(bank_idx) > settings.calibration_size:
            raise ValueError("explicit reservation exceeds the frozen calibration-size budget")
        reservation = dict(design="fixed outcome-independent episode IDs, selected before retention; uniform retained row per episode",
            reserved_episode_ids=reserved.tolist(), reserved_episode_sha256=_hash_arrays(dict(ids=reserved)),
            absent_reserved_episode_ids=absent, reserved_blocks=len(reserved), calibration_size=len(bank_idx),
            target_size=settings.calibration_size, rows_per_episode=settings.rows_per_episode,
            seed=settings.reservation_seed, withheld_size=len(withheld_idx), training_size=len(train_idx),
            budget_topups=0)
    else:
        try:
            train_idx, withheld_idx, bank_idx, reservation = native.reserve_pool(
                raw, settings.calibration_size, settings.reservation_seed, settings.rows_per_episode,
                max_fraction=settings.max_reserved_fraction, episode_ids=rows["episode_id"])
        except ValueError as error:
            # This predeclared fallback preserves a baseline even if BCA is infeasible.
            # No BCA fitting runs, and no missing calibration rows are manufactured.
            episodes = np.unique(rows["episode_id"])
            count = min(max(1, int(np.floor(len(episodes) * settings.max_reserved_fraction))), len(episodes) - 1)
            if count < 1:
                raise ValueError("fewer than two retained trajectories; cannot create disjoint pools") from error
            reserved = np.random.default_rng(settings.reservation_seed).permutation(episodes)[:count]
            withheld_idx = np.flatnonzero(np.isin(rows["episode_id"], reserved)).astype(np.int32)
            train_idx = np.flatnonzero(~np.isin(rows["episode_id"], reserved)).astype(np.int32)
            bank_idx = np.empty(0, dtype=np.int32)
            reservation = dict(design="reservation-infeasible baseline-only uniform-episode complement",
                               failure=str(error), reserved_blocks=count, total_blocks=len(episodes))
            refusal = "native calibration reservation infeasible: " + str(error)
    if not len(train_idx):
        raise ValueError("reservation leaves no training rows")
    posterior = WBCPConfig(settings.miscoverage, settings.credibility, settings.draws)
    if refusal is None and not certifiable(len(bank_idx), posterior):
        refusal = "insufficient calibration scores for a numerically certifiable native WBCP threshold"
    if np.intersect1d(rows["episode_id"][train_idx], rows["episode_id"][withheld_idx]).size:
        raise AssertionError("trajectory reservation leaked")
    # Keep the observed regime and remaining fraction in their declared units.
    obs32 = np.asarray(rows["observation"][train_idx], dtype=np.float32)
    mean, std = np.zeros(OBS_DIM, np.float32), np.ones(OBS_DIM, np.float32)
    mean[:13] = obs32[:, :13].mean(axis=0)
    std[:13] = obs32[:, :13].std(axis=0) + np.float32(.001)
    normalized = Transition((np.asarray(raw.obs, np.float32) - mean) / std,
                           np.asarray(raw.action, np.float32), np.asarray(raw.reward, np.float32),
                           (np.asarray(raw.next_obs, np.float32) - mean) / std,
                           np.asarray(raw.done, np.float32))
    take = lambda idx: jax.tree_util.tree_map(lambda x: jnp.asarray(x[idx]), normalized)
    def counts(idx, critical_only=False):
        ob = rows["observation"][idx]
        critical = ((ob[:, 10] >= .8) & np.any(np.abs(rows["action"][idx]) > float(np.float32(.8)), axis=1))
        selected = critical if critical_only else np.ones(len(idx), dtype=bool)
        return {f"regime{regime}_load{load}": int(np.sum((ob[:, 13] == regime) & np.isclose(ob[:, 10], load) & selected))
                for regime in (0, 1) for load in (.3, 1.1)}
    def critical_count(idx):
        return int(np.sum((rows["observation"][idx, 10] >= .8)
                          & np.any(np.abs(rows["action"][idx]) > float(np.float32(.8)), axis=1)))
    manifest = dict(schema="thrusterchain-prepared-training-v1", learner_rows_sha256=_hash_arrays(rows),
        retained_rows=n, training_rows=len(train_idx), withheld_rows=len(withheld_idx), calibration_rows=len(bank_idx),
        training_episodes=int(len(np.unique(rows["episode_id"][train_idx]))),
        withheld_episodes=int(len(np.unique(rows["episode_id"][withheld_idx]))),
        calibration_episodes=int(len(np.unique(rows["episode_id"][bank_idx]))),
        training_context_counts=counts(train_idx), calibration_context_counts=counts(bank_idx),
        training_critical_context_counts=counts(train_idx, True), calibration_critical_context_counts=counts(bank_idx, True),
        training_critical_rows=critical_count(train_idx), calibration_critical_rows=critical_count(bank_idx),
        pool_sha256=_hash_arrays(dict(training=train_idx, withheld=withheld_idx, calibration=bank_idx)),
        normalization_sha256=_hash_arrays(dict(mean=mean, std=std)),
        normalization="float32 train-only physical-coordinate mean and std+0.001; regime/time unchanged",
        reservation=reservation, bca_refusal=refusal, budget_topups=0, gamma=1.0, horizon=100,
        reward_transform="none", bootstrap_rule="done=1-bootstrap_mask, zero bootstrap at H=100",
        calibration_dependence="one row per trajectory" if settings.rows_per_episode == 1 else
            "multiple rows per trajectory; no dataset-specific dependence guarantee established",
        finite_sample_claim="numerical WBCP sufficiency only; no new coverage guarantee asserted",
        native_numerical_certifiability=bool(certifiable(len(bank_idx), posterior)),
        native_uncertified_behavior="valid infinite threshold gives unsupported rows with native multiplier 1",
        settings=asdict(settings))
    manifest = _jsonable(manifest)
    return PreparedTraining(take(train_idx), take(bank_idx), train_idx, withheld_idx, bank_idx,
                            mean, std, manifest, refusal)


def make_args(settings, seed):
    _, _, base, _, _, _, _, _ = _stack()
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return replace(base.Args(), seed=seed, dataset="thrusterchain-diagnostics-finite-h100-observed-regime-v1",
        num_updates=settings.updates, discount=settings.gamma, lr=settings.lr, batch_size=settings.batch_size,
        tau=settings.tau, alpha=settings.alpha, policy_noise=settings.policy_noise, noise_clip=settings.noise_clip,
        policy_freq=settings.policy_freq, reward_transform="none", normalize_reward=False,
        allow_off_config=True, eval_interval=settings.chunk)


def initialize(settings, arm, seed):
    _, jnp, _, native, WBCPConfig, _, _, _ = _stack()
    if arm not in ("td3_bc", "bca"):
        raise ValueError("arm must be 'td3_bc' or 'bca'")
    args = make_args(settings, seed)
    config = native.Config(arm="host") if arm == "td3_bc" else native.Config(
        arm="bca", posterior=WBCPConfig(settings.miscoverage, settings.credibility, settings.draws),
        blend=settings.blend, cal_lr=settings.cal_lr, cal_beta=settings.cal_beta,
        width_penalty=settings.width_penalty, scale_ema=settings.scale_ema)
    rng, state, models = native.initialize(args, config, OBS_DIM, ACTION_DIM, 1.0)
    return (rng, state, jnp.int32(0)), models, args, config


def tree_sha256(tree):
    jax, *_ = _stack()
    return _hash_arrays({str(i): np.asarray(leaf) for i, leaf in enumerate(jax.tree_util.tree_leaves(tree))})


def _source_hashes():
    import inspect
    jax, _, base, native, _, _, _, _ = _stack()
    from calibration import network, reference, dose, wbcp, bank
    from runtime import networks, rewards, published
    modules = (base, native, network, reference, dose, wbcp, bank, networks, rewards, published)
    hashes = {module.__name__: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
              for module in modules}
    hashes[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def _metrics_summary(metrics):
    summary = {}
    for name, value in metrics.items():
        value = np.asarray(value)
        summary[name] = dict(mean=float(value.mean()), minimum=float(value.min()), maximum=float(value.max()))
    return summary


@dataclass
class TrainingResult:
    carry: object
    models: object
    mean: np.ndarray
    std: np.ndarray
    record: dict

    @property
    def state(self):
        return None if self.carry is None else self.carry[1]

    def policy(self, raw_observations):
        if self.carry is None:
            raise ValueError("a refused run has no trained policy")
        _, jnp, *_ = _stack()
        observations = np.asarray(raw_observations, np.float32)
        if observations.shape[-1:] != (OBS_DIM,) or not np.isfinite(observations).all():
            raise ValueError("policy expects finite [...,15] observations")
        return np.asarray(self.models[0].apply(self.state.native.actor.params,
                          jnp.asarray((observations - self.mean) / self.std)))

    def critic(self, raw_observations, actions):
        if self.carry is None:
            raise ValueError("a refused run has no trained critic")
        _, jnp, *_ = _stack()
        observations, actions = np.asarray(raw_observations, np.float32), np.asarray(actions, np.float32)
        if (observations.shape[-1:] != (OBS_DIM,) or actions.shape != observations.shape[:-1] + (ACTION_DIM,)
                or not np.isfinite(observations).all() or not np.isfinite(actions).all()):
            raise ValueError("critic expects aligned finite [...,15] observations and [...,4] actions")
        return np.asarray(self.models[1].apply(self.state.native.critic.params,
                          jnp.asarray((observations - self.mean) / self.std), jnp.asarray(actions)))


def _checkpoint(out_dir, carry, record):
    _, _, _, _, _, _, _, serialization = _stack()
    step = int(np.asarray(carry[2]))
    name = f"checkpoint_{step:09d}.msgpack"
    payload = serialization.to_bytes(carry)
    temporary = out_dir / (name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(out_dir / name)
    record["checkpoint"] = dict(file=name, sha256=hashlib.sha256(payload).hexdigest(), update=step,
                                actor_sha256=tree_sha256(carry[1].native.actor.params))
    for snapshot in record.get("snapshots", []):
        if snapshot["update"] == step:
            snapshot.update(record["checkpoint"])
    record["final_update"] = step
    native = carry[1].native
    record["counters"] = {key: int(np.asarray(getattr(native, key).step))
                          for key in ("actor", "critic", "actor_target", "critic_target")}
    record["counters"]["calibrator"] = 0 if carry[1].calibrator is None else int(np.asarray(carry[1].calibrator.step))
    record["prng_sha256"] = tree_sha256(carry[0])
    _write_json(out_dir / "manifest.json", record)


def _restore(out_dir, template, record):
    _, _, _, _, _, _, _, serialization = _stack()
    checkpoint = record["checkpoint"]
    path = out_dir / checkpoint["file"]
    if path.parent.resolve() != out_dir.resolve():
        raise ValueError("checkpoint must be within its run directory")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != checkpoint["sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")
    carry = serialization.from_bytes(template, payload)
    if int(np.asarray(carry[2])) != checkpoint["update"]:
        raise ValueError("checkpoint counter mismatch")
    return carry


def train(prepared, settings=TrainingSettings(), *, arm, seed, out_dir, resume=True,
          stop_after=None, progress=None):
    """Train to the fixed budget, or a requested resumable intermediate boundary.

    Chunk checkpoints include every optimizer, target, frozen BCA reference,
    PRNG and accepted-update counter. No evaluator is invoked. Invalid native
    transitions fail the row and preserve the last committed checkpoint.
    """
    if prepared.manifest["settings"] != asdict(settings):
        raise ValueError("prepared settings differ from training settings")
    if arm not in ("td3_bc", "bca"):
        raise ValueError("unknown arm")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(schema=SCHEMA, arm=arm, seed=seed, settings=asdict(settings),
                    data=prepared.manifest, source_sha256=_source_hashes())
    record_path = out_dir / "manifest.json"
    if arm == "bca" and prepared.bca_refusal:
        record = dict(identity=identity, status="refused", reason=prepared.bca_refusal, final_update=0,
                      budget_topups=0, checkpoints=[], chunks=[], refreshes=[])
        if record_path.exists():
            old = json.loads(record_path.read_text())
            if old.get("identity") != identity:
                raise ValueError("existing run identity differs")
        _write_json(record_path, record)
        return TrainingResult(None, None, prepared.mean, prepared.std, record)
    jax, jnp, _, native, _, _, _, _ = _stack()
    carry, models, args, config = initialize(settings, arm, seed)
    if record_path.exists():
        if not resume:
            raise FileExistsError("run manifest exists; choose resume=True or a new directory")
        record = json.loads(record_path.read_text())
        if record.get("identity") != identity:
            raise ValueError("resume identity differs: configuration, data, or source changed")
        carry = _restore(out_dir, carry, record)
    else:
        record = dict(identity=identity, args=asdict(args), status="running", final_update=0,
            budget_topups=0, chunks=[], refreshes=[], snapshots=[], training_seconds=0.0,
            runtime=dict(jax=jax.__version__, devices=[str(d) for d in jax.devices()], arithmetic="float32"),
            initial_native_sha256=tree_sha256(carry[1].native),
            initial_prng_sha256=tree_sha256(carry[0]),
            objective="raw finite-horizon undiscounted return; no terminal bootstrap",
            departures=["15D observation with observed regime and remaining time", "gamma=1 and H=100",
                        "physical state normalization fitted only on reserved training complement",
                        "novel environment/configuration explicitly declared allow_off_config",
                        "host/BCA both exclude reserved trajectories"])
        np.savez(out_dir / "normalization.npz", mean=prepared.mean, std=prepared.std)
        np.savez(out_dir / "reservation.npz", training=prepared.training_indices,
                 withheld=prepared.withheld_indices, calibration=prepared.calibration_indices)
        _checkpoint(out_dir, carry, record)
    limit = settings.updates if stop_after is None else int(stop_after)
    done = int(np.asarray(carry[2]))
    if not done <= limit <= settings.updates:
        raise ValueError("stop_after must be between the checkpoint and the full update budget")
    refreshes = set(range(settings.bca_warmup, settings.updates, settings.bca_refresh)) if arm == "bca" else set()
    snapshots = {settings.updates // 2, settings.updates}
    events = sorted({limit} | {x for x in range(settings.chunk, limit + 1, settings.chunk) if x > done}
                    | {x for x in refreshes | snapshots if done < x <= limit})
    step = native.make_train_step(args, config, models, prepared.train, 1.0)
    programs = {}
    session_started = time.perf_counter()
    previous_seconds = float(record.get("training_seconds", 0.0))
    try:
        for target in events:
            if target <= done:
                continue
            length = target - done
            if length not in programs:
                programs[length] = jax.jit(lambda c, length=length: jax.lax.scan(step, c, None, length=length))
            started = time.perf_counter()
            proposed, metrics = programs[length](carry)
            jax.block_until_ready(proposed)
            native.require_valid(metrics)
            if int(np.asarray(proposed[2])) != target:
                raise FloatingPointError("accepted native update counter did not reach requested boundary")
            if target in refreshes:
                # Refresh randomness is a separate deterministic stream; it never
                # consumes or perturbs minibatch/target-noise keys.
                key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), 1380271698), target)
                state, flags, diagnostic = native.refresh(args, config, models, proposed[1],
                    prepared.bank, key, 1.0, heldout_ids=prepared.calibration_indices)
                native.require_valid(flags)
                proposed = (proposed[0], state, proposed[2])
                record["refreshes"].append(dict(update=target, **_jsonable(flags), diagnostics=_jsonable(diagnostic)))
            carry, done = proposed, target
            record["chunks"].append(dict(update=done, updates=length,
                seconds=time.perf_counter() - started, metrics=_metrics_summary(metrics)))
            if done in snapshots:
                record["snapshots"].append(dict(update=done, file=f"checkpoint_{done:09d}.msgpack",
                    role="primary final" if done == settings.updates else "descriptive midpoint"))
            record.update(status="completed" if done == settings.updates else "running",
                          training_seconds=previous_seconds + time.perf_counter() - session_started)
            _checkpoint(out_dir, carry, record)
            if progress:
                progress(dict(arm=arm, seed=seed, update=done, total=settings.updates,
                              chunk_seconds=record["chunks"][-1]["seconds"]))
    except Exception as error:
        # The last checkpoint remains the only committed state. A failed chunk
        # cannot silently advance either counters or random streams.
        record.update(status="failed", failure=type(error).__name__ + ": " + str(error),
                      failed_target=target, training_seconds=previous_seconds + time.perf_counter() - session_started)
        _write_json(record_path, record)
        raise
    if done < settings.updates:
        record["status"] = "paused"
    _write_json(record_path, record)
    return TrainingResult(carry, models, prepared.mean, prepared.std, record)


def load_trained(out_dir, update=None):
    """Load the latest or an authenticated midpoint/final snapshot without data.

    A specifically requested older snapshot has result.record.status='snapshot',
    final_update equal to its own step, and source_status/source_final_update
    identifying the latest manifest from which it was selected.
    """
    out_dir = Path(out_dir)
    record = json.loads((out_dir / "manifest.json").read_text())
    if record.get("status") == "refused":
        raise ValueError("refused run has no policy checkpoint")
    identity = record["identity"]
    if identity["schema"] != SCHEMA or identity["source_sha256"] != _source_hashes():
        raise ValueError("training schema or source fingerprint differs")
    if update is not None and update != record["checkpoint"]["update"]:
        matches = [item for item in record.get("snapshots", []) if item["update"] == update]
        if len(matches) != 1 or "sha256" not in matches[0]:
            raise ValueError("requested update has no authenticated saved snapshot")
        record.update(source_status=record["status"], source_final_update=record["final_update"],
                      status="snapshot", final_update=int(update), checkpoint=dict(matches[0]))
    settings = TrainingSettings(**identity["settings"])
    template, models, _, _ = initialize(settings, identity["arm"], identity["seed"])
    carry = _restore(out_dir, template, record)
    if record["status"] == "snapshot":
        record["counters"] = {key: int(np.asarray(getattr(carry[1].native, key).step))
                              for key in ("actor", "critic", "actor_target", "critic_target")}
        record["counters"]["calibrator"] = 0 if carry[1].calibrator is None else int(np.asarray(carry[1].calibrator.step))
        record["prng_sha256"] = tree_sha256(carry[0])
    with np.load(out_dir / "normalization.npz", allow_pickle=False) as values:
        mean, std = values["mean"], values["std"]
    if _hash_arrays(dict(mean=mean, std=std)) != identity["data"]["normalization_sha256"]:
        raise ValueError("normalization fingerprint differs")
    return TrainingResult(carry, models, mean, std, record)
