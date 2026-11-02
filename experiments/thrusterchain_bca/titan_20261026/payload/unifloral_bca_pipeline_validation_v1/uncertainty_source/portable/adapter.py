"""Real neural TD3+BC/BCA adapter for repaired Regular, version 1.

The operational reference influences training; it is not a posterior over true Q.
No historical model parameters or evaluator transitions are consumed.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import copy
import hashlib
import json
from typing import NamedTuple

from .dependencies import BASE, NATIVE, DEPENDENCY_IDENTITY
from . import kernel
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from calibration.reference import WBCPConfig, certifiable
from thrusterchain.datasets import assert_training_allowed, canonical_json
from thrusterchain.training_protocol import AdapterCapabilities, LearnerCheckpoint, TransitionBatch
from thrusterchain.training_runner import train_transitions

ADAPTER_ID = "thrusterchain_neural_td3bc_direct_v1"
ONLINE_STRATEGY = "frozen_offline_reference_v1"
FIELDS = ("observations", "actions", "rewards", "next_observations", "terminated", "truncated", "bootstrap_mask", "origin", "group_ids")

class NativeBatch(NamedTuple):
    obs: object
    action: object
    next_obs: object
    reward: object
    done: object

@dataclass(frozen=True)
class AdapterConfig:
    method: str = "host"
    gamma: float = .98
    horizon: int = 100
    lr: float = .0003
    alpha: float = 2.5
    tau: float = .005
    policy_noise: float = .2
    noise_clip: float = .5
    policy_freq: int = 2
    kappa: float = 1.
    bc_multiplier: float = 2.
    cal_lr: float = .001
    cal_beta: float = 20.
    width_penalty: float = .005
    scale_ema: float = .99
    miscoverage: float = .1
    credibility: float = .95
    draws: int = 1000
    reference_size: int = 128
    reference_seed: int = 20261022
    warmup_updates: int = 10000
    refresh_interval: int = 5000
    exploration_noise: float = .1

    def __post_init__(self):
        if self.method not in ("host", "bca_direct", "twin_disagreement", "stronger_bc"):
            raise ValueError("unknown neural comparison method")
        if self.gamma != .98 or self.horizon != 100:
            raise ValueError("this adapter is exclusively repaired Regular gamma .98 horizon100")
        for name in ("policy_freq", "draws", "reference_size", "warmup_updates", "refresh_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive Python int")
        if type(self.reference_seed) is not int or self.reference_seed < 0:
            raise ValueError("reference_seed must be nonnegative Python int")
        for name in ("lr", "alpha", "tau", "policy_noise", "noise_clip", "kappa", "bc_multiplier", "cal_lr", "cal_beta", "width_penalty", "scale_ema", "miscoverage", "credibility", "exploration_noise"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite nonnegative")
        if min(self.lr, self.alpha, self.cal_lr, self.cal_beta) <= 0 or self.bc_multiplier <= 1 or not 0 < self.tau <= 1 or not 0 <= self.scale_ema < 1:
            raise ValueError("invalid positive learning parameter or averaging coefficient")
        WBCPConfig(self.miscoverage, self.credibility, self.draws)

def _hash_batch(batch):
    digest = hashlib.sha256()
    for name in FIELDS:
        array = np.asarray(getattr(batch, name))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()

def _validate_batch(batch, *, offline=False):
    if type(batch) is not TransitionBatch or not len(batch):
        raise ValueError("nonempty core TransitionBatch required")
    if batch.observations.shape != (len(batch), 14) or batch.actions.shape != (len(batch), 4):
        raise ValueError("requires 14D physical-plus-time observations and four actions; 15D weights/data incompatible")
    if np.any(np.abs(batch.actions) > 1.):
        raise ValueError("executed actions outside [-1,1]")
    time, next_time = batch.observations[:, 13], batch.next_observations[:, 13]
    if np.any(time <= 0) or np.any(time > 1) or not np.allclose(time - next_time, .01, rtol=0, atol=1e-6):
        raise ValueError("time coordinate must be remaining fraction with one horizon100 transition")
    terminal = np.isclose(next_time, 0., rtol=0, atol=1e-6)
    if np.any(batch.truncated) or not np.array_equal(batch.terminated, terminal) or not np.array_equal(batch.bootstrap_mask, (~terminal).astype(float)):
        raise ValueError("finite horizon termination required; no terminal bootstrap")
    if offline and np.any(batch.origin != 0):
        raise ValueError("online/evaluation transitions cannot enter offline training or reference")

def _args(config, seed, settings):
    return replace(BASE.Args(), seed=seed % (2**32), dataset="thrusterchain-repaired-Regular-14D-h100-gamma098-v1",
        lr=settings["lr"], alpha=settings["alpha"], discount=config.gamma,
        tau=config.tau, policy_noise=config.policy_noise, noise_clip=config.noise_clip,
        policy_freq=config.policy_freq, reward_transform="none", normalize_reward=False,
        allow_off_config=True)

def _kernel_config(config):
    return kernel.Config(arm="bca", method=config.method, kappa=config.kappa,
        bc_multiplier=config.bc_multiplier, posterior=WBCPConfig(config.miscoverage, config.credibility, config.draws),
        blend=1., cal_lr=config.cal_lr, cal_beta=config.cal_beta,
        width_penalty=config.width_penalty, scale_ema=config.scale_ema)

@lru_cache(maxsize=32)
def _compiled(config, lr, alpha, fit_auxiliary):
    args = _args(config, 0, {"lr": lr, "alpha": alpha})
    cfg = _kernel_config(config)
    # Network definitions have no mutable statistics; construction is not fitting.
    actor, critic = BASE.Actor(14, 4, 1.), BASE.DualCritic(14, 4)
    from calibration.network import Calibrator
    cal = Calibrator(jnp.zeros(14), jnp.ones(14), state_dep=True)
    models = (actor, critic, cal)
    return jax.jit(lambda state, batch, it, key: kernel.update(
        args, cfg, models, state, batch, it, key, 1., fit_auxiliary=fit_auxiliary))

@lru_cache(maxsize=1)
def _actor_function():
    actor = BASE.Actor(14, 4, 1.)
    return jax.jit(lambda params, obs: actor.apply(params, obs))

def _pack(value):
    # JIT canonicalizes parameter dictionaries; initialization and restoration can
    # retain a different insertion order. Canonicalize keys to make saved bytes,
    # not merely numerical leaves, identical across exact resume boundaries.
    def ordered(item):
        if isinstance(item, dict):
            return {key: ordered(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            return [ordered(child) for child in item]
        if isinstance(item, tuple):
            return tuple(ordered(child) for child in item)
        return item
    return serialization.msgpack_serialize(ordered(serialization.to_state_dict(value)))

def _unpack(template, data):
    return serialization.from_bytes(template, data)

class FrozenNeuralPolicy:
    """Inference-only detached actor; no reference, critic, replay, RNG or optimizer."""
    __slots__ = ("_params", "_mean", "_std")

    def __init__(self, params, mean, std):
        self._params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)
        self._mean = np.frombuffer(np.asarray(mean, np.float32).tobytes(), dtype=np.float32)
        self._std = np.frombuffer(np.asarray(std, np.float32).tobytes(), dtype=np.float32)
        # Compile before entry to the evaluation guard.
        _actor_function()(self._params, jnp.zeros(14, jnp.float32)).block_until_ready()

    def act(self, observation):
        obs = np.asarray(observation, np.float32)
        if obs.shape != (14,) or not np.all(np.isfinite(obs)):
            raise ValueError("frozen inference requires one finite14D observation")
        return np.asarray(_actor_function()(self._params, jnp.asarray((obs-self._mean)/self._std)), dtype=np.float32).copy()

    def state_bytes(self):
        return _pack({"params": self._params, "mean": self._mean, "std": self._std,
                      "action_transform": "tanh[-1,1]", "observation_dim": 14})

class NeuralTD3BCAdapter:
    capabilities = AdapterCapabilities(ADAPTER_ID, "optimizer_steps", True, "critic")

    def __init__(self, config=AdapterConfig(), *, reference_batch=None, lineage=None):
        if type(config) is not AdapterConfig:
            raise TypeError("AdapterConfig required")
        self.config = config
        self._reference_raw = reference_batch
        self._lineage = copy.deepcopy(lineage)
        self._initialized = False

    @property
    def optimizer_step_counts(self):
        return dict(self._counts) if self._initialized else {"critic": 0, "actor": 0, "auxiliary": 0}

    @property
    def optimizer_steps(self):
        return sum(self.optimizer_step_counts.values())

    def initialize(self, train_batch, seed, learning_settings):
        assert_training_allowed("neural.initialize")
        if self._initialized:
            raise RuntimeError("initialize cannot reset an existing learner; create a fresh factory instance")
        _validate_batch(train_batch, offline=True)
        if self._reference_raw is None or self._lineage is None:
            raise ValueError("use make_factory(dataset, config) to bind explicit offline data roles")
        _validate_batch(self._reference_raw, offline=True)
        if _hash_batch(train_batch) != self._lineage["train_transition_sha256"]:
            raise ValueError("initialize data differs from the factory's train-only lineage")
        if np.intersect1d(train_batch.group_ids, self._reference_raw.group_ids).size:
            raise ValueError("operational reference groups overlap gradient training")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative Python int")
        self.seed = seed
        self.settings = {"lr": self.config.lr, "alpha": self.config.alpha, "exploration_noise": self.config.exploration_noise}
        self.settings.update(self._validate_settings(learning_settings))
        self.args = _args(self.config, seed, self.settings)
        self.kernel_config = _kernel_config(self.config)
        self.rng, self.state, self.models = kernel.initialize(self.args, self.kernel_config, 14, 4)
        self.exploration_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(9301,))))
        self.mean, self.std = np.zeros(14, np.float32), np.ones(14, np.float32)
        physical = np.asarray(train_batch.observations[:, :13], np.float32)
        self.mean[:13] = physical.mean(axis=0)
        self.std[:13] = physical.std(axis=0) + np.float32(.001)
        self.bank = self._native(self._reference_raw)
        self.train_groups = np.unique(train_batch.group_ids)
        self._counts = {"critic": 0, "actor": 0, "auxiliary": 0}
        self.phase = "offline"
        self.refresh_count = 0
        self._metadata = {"algorithm": ADAPTER_ID, "config": asdict(self.config),
                          "seed": seed, "jax_seed_uint32": seed % (2**32),
                          "data_lineage": copy.deepcopy(self._lineage),
                          "dependency_identity": DEPENDENCY_IDENTITY,
                          "primary_optimizer": "critic", "online_strategy": ONLINE_STRATEGY,
                          "historical_weights_loaded": False, "scientific_claim": "none; adapter implementation"}
        self._initialized = True

    def _native(self, batch):
        return NativeBatch(jnp.asarray((np.asarray(batch.observations, np.float32)-self.mean)/self.std),
            jnp.asarray(batch.actions, jnp.float32),
            jnp.asarray((np.asarray(batch.next_observations, np.float32)-self.mean)/self.std),
            jnp.asarray(batch.rewards, jnp.float32), jnp.asarray(batch.terminated, jnp.float32))

    def update(self, batch):
        assert_training_allowed("neural.update")
        if not self._initialized:
            raise RuntimeError("initialize or restore first")
        _validate_batch(batch, offline=self.phase == "offline")
        offline_ids = batch.group_ids[batch.origin == 0]
        if not np.all(np.isin(offline_ids, self.train_groups)):
            raise ValueError("non-Train groups cannot enter optimizer batches")
        if self.phase == "offline" and self._counts["critic"] >= 3_000_000:
            raise ValueError("maximum cumulative offline primary updates reached")
        next_rng, _, noise_key = jax.random.split(self.rng, 3)
        step = self._counts["critic"] + 1
        auxiliary = self.phase == "offline"
        proposed, metrics = _compiled(self.config, self.settings["lr"], self.settings["alpha"], auxiliary)(
            self.state, self._native(batch), jnp.int32(step), noise_key)
        values = {name: np.asarray(value).item() for name, value in metrics.items()}
        kernel.require_valid(values)
        if any(type(v) in (int, float) and not np.isfinite(v) for v in values.values()):
            raise FloatingPointError("nonfinite update rejected before learner state commit")
        values["reference_refreshed"] = False
        if auxiliary and step >= self.config.warmup_updates and (step-self.config.warmup_updates) % self.config.refresh_interval == 0:
            proposed, refresh_values = self._refreshed(proposed, step)
            values.update(refresh_values)
        # A numerical failure in either optimizer work or scheduled calibration
        # leaves every state component and counter at the pre-call snapshot.
        self.state, self.rng = proposed, next_rng
        self._counts["critic"] += 1
        self._counts["actor"] += int(step % self.config.policy_freq == 0)
        self._counts["auxiliary"] += int(auxiliary)
        self.refresh_count += int(values["reference_refreshed"])
        return values

    def _refreshed(self, state, step):
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed % (2**32)), step)
        proposed, flags, diagnostics = kernel.refresh(self.args, self.kernel_config, self.models,
            state, self.bank, key, 1., heldout_ids=np.arange(len(self.bank.obs), dtype=np.int64))
        kernel.require_valid(flags)
        return proposed, {"reference_refreshed": True, "reference_finite": bool(np.isfinite(np.asarray(proposed.posterior.threshold))),
                          "reference_ready": bool(np.asarray(proposed.posterior.ready)), "reference_rows": len(self.bank.obs)}

    def refresh_reference(self):
        assert_training_allowed("neural.refresh_reference")
        if not self._initialized or self.phase != "offline":
            raise RuntimeError("reference refresh supported only in offline training")
        proposed, metrics = self._refreshed(self.state, self._counts["critic"])
        self.state = proposed
        self.refresh_count += 1
        return metrics

    def act(self, observation, explore=False):
        # The live learner is a training API; only freeze() exposes evaluation inference.
        assert_training_allowed("neural.live_act")
        obs = np.asarray(observation, np.float32)
        if obs.shape != (14,) or not np.all(np.isfinite(obs)):
            raise ValueError("requires finite14D observation")
        action = np.asarray(_actor_function()(self.state.native.actor.params,
            jnp.asarray((obs-self.mean)/self.std)), dtype=np.float32)
        if explore:
            action = action + self.exploration_rng.normal(0., self.settings["exploration_noise"], 4)
        return np.clip(action, -1., 1.).astype(np.float32)

    @staticmethod
    def _validate_settings(settings):
        if type(settings) is not dict or set(settings) - {"lr", "alpha", "exploration_noise"}:
            raise ValueError("learning_settings accepts only lr, alpha, exploration_noise")
        for key, value in settings.items():
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0 or (key != "exploration_noise" and value == 0):
                raise ValueError("invalid learning setting " + key)
        return dict(settings)

    def configure_learning_settings(self, learning_settings):
        assert_training_allowed("neural.configure_learning_settings")
        if not self._initialized:
            raise RuntimeError("restore before configuring online adaptation")
        settings = {**self.settings, **self._validate_settings(learning_settings)}
        # Replace only the static optimizer transformation; retain Adam moments/steps.
        tx = BASE.C.torch_adam(settings["lr"])
        native = self.state.native
        native = native._replace(actor=native.actor.replace(tx=tx), critic=native.critic.replace(tx=tx),
                                  actor_target=native.actor_target.replace(tx=tx), critic_target=native.critic_target.replace(tx=tx))
        self.state = self.state._replace(native=native)
        self.settings = settings
        self.args = _args(self.config, self.seed, settings)
        self.phase = "online"

    def freeze(self):
        assert_training_allowed("neural.freeze")
        return FrozenNeuralPolicy(self.state.native.actor.params, self.mean, self.std)

    def _components(self):
        native = self.state.native
        optimizers = {name: {"step": getattr(native, name).step, "opt_state": getattr(native, name).opt_state}
                      for name in ("actor", "critic", "actor_target", "critic_target")}
        optimizers["auxiliary"] = {"step": self.state.calibrator.step, "opt_state": self.state.calibrator.opt_state}
        return {
            "model": _pack({"actor": native.actor.params, "critic": native.critic.params, "auxiliary": self.state.calibrator.params}),
            "optimizer": _pack(optimizers),
            "target_networks": _pack({"actor_target": native.actor_target.params, "critic_target": native.critic_target.params}),
            "rng": _pack({"jax": self.rng, "exploration": canonical_json(self.exploration_rng.bit_generator.state)}),
            "preprocessing": _pack({"mean": self.mean, "std": self.std, "train_groups": self.train_groups}),
            "reference": _pack({"posterior": self.state.posterior, "residual_scale": self.state.residual_scale, "bank": self.bank}),
        }

    def checkpoint(self):
        assert_training_allowed("neural.checkpoint")
        if not self._initialized:
            raise RuntimeError("initialize or restore first")
        metadata = copy.deepcopy(self._metadata)
        metadata.update(learning_settings=dict(self.settings), phase=self.phase, refresh_count=self.refresh_count)
        return LearnerCheckpoint(ADAPTER_ID, self._components(), self.optimizer_steps, self.optimizer_step_counts, metadata)

    def restore(self, checkpoint):
        assert_training_allowed("neural.restore")
        if type(checkpoint) is not LearnerCheckpoint or checkpoint.adapter_id != ADAPTER_ID:
            raise ValueError("only fresh14D v1 adapter checkpoints supported; historical15D checkpoints incompatible")
        meta = checkpoint.metadata
        if meta.get("config") != asdict(self.config) or meta.get("dependency_identity") != DEPENDENCY_IDENTITY:
            raise ValueError("checkpoint configuration/dependency mismatch")
        if self._lineage is not None and meta.get("data_lineage") != self._lineage:
            raise ValueError("checkpoint data-role lineage differs from factory")
        self._lineage = copy.deepcopy(meta["data_lineage"])
        self.seed, self.settings = meta["seed"], self._validate_settings(meta["learning_settings"])
        self.args, self.kernel_config = _args(self.config, self.seed, self.settings), _kernel_config(self.config)
        self.rng, self.state, self.models = kernel.initialize(self.args, self.kernel_config, 14, 4)
        native = self.state.native
        components = checkpoint.components
        model = _unpack({"actor": native.actor.params, "critic": native.critic.params, "auxiliary": self.state.calibrator.params}, components["model"])
        targets = _unpack({"actor_target": native.actor_target.params, "critic_target": native.critic_target.params}, components["target_networks"])
        template = {name: {"step": getattr(native, name).step, "opt_state": getattr(native, name).opt_state} for name in ("actor", "critic", "actor_target", "critic_target")}
        template["auxiliary"] = {"step": self.state.calibrator.step, "opt_state": self.state.calibrator.opt_state}
        optimizers = _unpack(template, components["optimizer"])
        updates = {name: getattr(native, name).replace(params=(model if name in model else targets)[name], **optimizers[name]) for name in ("actor", "critic", "actor_target", "critic_target")}
        native = native._replace(**updates)
        preprocessing = serialization.msgpack_restore(components["preprocessing"])
        self.mean, self.std, self.train_groups = (preprocessing[k] for k in ("mean", "std", "train_groups"))
        # The stored bank is numerical offline data only; msgpack has no executable objects.
        reference = serialization.msgpack_restore(components["reference"])
        self.bank = NativeBatch(*(jnp.asarray(reference["bank"][name]) for name in NativeBatch._fields))
        reference = _unpack({"posterior": self.state.posterior, "residual_scale": self.state.residual_scale, "bank": self.bank}, components["reference"])
        self.state = self.state._replace(native=native, calibrator=self.state.calibrator.replace(params=model["auxiliary"], **optimizers["auxiliary"]), posterior=reference["posterior"], residual_scale=reference["residual_scale"])
        rng = serialization.msgpack_restore(components["rng"])
        self.rng = jnp.asarray(rng["jax"])
        self.exploration_rng = np.random.Generator(np.random.PCG64())
        self.exploration_rng.bit_generator.state = json.loads(rng["exploration"])
        self._counts = dict(checkpoint.optimizer_step_counts)
        expected = {"critic": int(native.critic.step), "actor": int(native.actor.step), "auxiliary": int(self.state.calibrator.step)}
        if self._counts != expected or checkpoint.primary_optimizer != "critic":
            raise ValueError("checkpoint optimizer counters differ from actual optimizer states")
        self.phase, self.refresh_count = meta["phase"], meta["refresh_count"]
        if self.phase not in ("offline", "online"):
            raise ValueError("unsupported checkpoint phase")
        self._metadata = copy.deepcopy(meta)  # Preserve unknown core runner metadata exactly.
        self._initialized = True
        if self.checkpoint().digest != checkpoint.digest:
            raise ValueError("checkpoint restore did not exactly preserve full state")

def make_factory(dataset, config=AdapterConfig()):
    """Bind only Train and an explicitly training-influential operational reference."""
    assert_training_allowed("neural.make_factory")
    if type(config) is not AdapterConfig:
        raise TypeError("AdapterConfig required")
    meta = dataset.metadata
    if meta["gamma"] != .98 or meta["horizon"] != 100 or meta["task_semantics"] != "finite_horizon":
        raise ValueError("requires repaired finite-horizon Regular data")
    train = train_transitions(dataset)
    _validate_batch(train, offline=True)
    split = dataset.split("tuning_validation")
    n = min(config.reference_size, len(split["group_ids"]))
    if not certifiable(n, WBCPConfig(config.miscoverage, config.credibility, config.draws)):
        raise ValueError(f"operational reference has only {n} independent trajectory rows; insufficient for declared numerical WBCP threshold, default requires >=36 (generate >=180 episodes)")
    rng = np.random.default_rng(config.reference_seed)
    episodes = rng.permutation(len(split["group_ids"]))[:n]
    steps = rng.integers(0, 100, n)
    terminal = split["terminated"][episodes, steps]
    reference = TransitionBatch(observations=split["observations"][episodes, steps], actions=split["actions"][episodes, steps],
        rewards=split["rewards"][episodes, steps], next_observations=split["observations"][episodes, steps+1],
        terminated=terminal, truncated=split["truncated"][episodes, steps], bootstrap_mask=(~terminal).astype(float),
        origin=np.zeros(n, np.uint8), group_ids=split["group_ids"][episodes])
    lineage = {"dataset_sha256": dataset.digest, "train_split_sha256": dataset.split_digest("train"),
        "train_transition_sha256": _hash_batch(train), "reference_split_sha256": dataset.split_digest("tuning_validation"),
        "reference_transition_sha256": _hash_batch(reference), "reference_source": "tuning_validation",
        "reference_role": "training-influential operational residual reference; not independent policy validation",
        "reference_groups": reference.group_ids.tolist(), "reference_steps": steps.tolist(), "reference_rows": n,
        "normalization_source": "train physical13 only; time unchanged", "selection_consumed": False, "final_calibration_consumed": False}
    def factory():
        return NeuralTD3BCAdapter(config, reference_batch=reference, lineage=lineage)
    factory.config, factory.lineage = config, copy.deepcopy(lineage)
    return factory
