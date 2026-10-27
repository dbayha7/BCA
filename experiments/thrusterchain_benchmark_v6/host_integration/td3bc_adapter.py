"""Versioned strict-offline adapter for the repository-native TD3+BC host.

Primary steps are critic updates. The host updates its actor and targets after
each second critic step. This adapter exposes no online-training capability.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import flax
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from algorithms import td3_bc as host
import runtime.networks as networks
import runtime.rewards as rewards
from runtime.networks import Transition
from thrusterchain.datasets import assert_training_allowed
from thrusterchain.training_protocol import AdapterCapabilities, LearnerCheckpoint, TransitionBatch


ID = "repository_td3bc_strict_v6_1"


def _source_hashes():
    files = {"host": Path(host.__file__), "runtime_networks": Path(networks.__file__), "runtime_rewards": Path(rewards.__file__),
             "adapter": Path(__file__)}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pack(value):
    """Canonicalize Flax state-dict key order across initialization and jit."""
    def ordered(item):
        if isinstance(item, dict):
            return {key: ordered(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            return [ordered(element) for element in item]
        return item
    return serialization.msgpack_serialize(ordered(serialization.to_state_dict(value)))


def _finite_tree(tree):
    return all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree))


def _initialize(args, obs_dim):
    """Native initialization with a 64-bit root key; preserve 32-bit parity."""
    root = jnp.asarray([args.seed >> 32, args.seed & 0xFFFFFFFF], dtype=jnp.uint32)
    rng = jax.random.fold_in(root, 1414676809)
    rng, ka, kc = jax.random.split(rng, 3)
    actor, critic = host.Actor(obs_dim, 4, 1.0), host.DualCritic(obs_dim, 4)
    obs, action = jnp.zeros(obs_dim), jnp.zeros(4)

    def make(key, network, inputs):
        return TrainState.create(apply_fn=network.apply, params=network.init(key, *inputs),
                                 tx=host.C.torch_adam(args.lr))

    state = host.AgentTrainState(make(ka, actor, [obs]), make(ka, actor, [obs]),
                                 make(kc, critic, [obs, action]), make(kc, critic, [obs, action]))
    return rng, state, (actor, critic)


def _arrays(batch):
    if not isinstance(batch, TransitionBatch) or len(batch) < 1:
        raise ValueError("nonempty transition batch required")
    obs = np.asarray(batch.observations, dtype=np.float32)
    nxt = np.asarray(batch.next_observations, dtype=np.float32)
    action = np.asarray(batch.actions, dtype=np.float32)
    reward = np.asarray(batch.rewards, dtype=np.float32)
    if (obs.ndim != 2 or nxt.shape != obs.shape or action.shape != (len(batch), 4)
            or reward.shape != (len(batch),) or np.any(np.abs(action) > 1)
            or not all(np.all(np.isfinite(x)) for x in (obs, nxt, action, reward))
            or not np.array_equal(np.asarray(batch.bootstrap_mask), ~np.asarray(batch.terminated))):
        raise ValueError("invalid transition shapes, actions, values, or terminal bootstrap")
    return obs, nxt, action, reward


class FrozenTD3BCPolicy:
    """Immutable bytes-backed model and train-only observation transform."""
    __slots__ = ("_params", "_mean", "_scale", "_dimension", "_state", "_variables", "_apply")

    def __init__(self, params, mean, scale):
        object.__setattr__(self, "_params", _pack(params))
        object.__setattr__(self, "_mean", np.asarray(mean, dtype="<f4").tobytes())
        object.__setattr__(self, "_scale", np.asarray(scale, dtype="<f4").tobytes())
        object.__setattr__(self, "_dimension", len(mean))
        model = host.Actor(len(mean), 4, 1.0)
        variables = flax.core.freeze(params)
        apply = jax.jit(model.apply)
        apply(variables, jnp.zeros(len(mean), dtype=jnp.float32)).block_until_ready()
        object.__setattr__(self, "_variables", variables)
        object.__setattr__(self, "_apply", apply)
        object.__setattr__(self, "_state", _json({"adapter": ID, "params_sha256": hashlib.sha256(self._params).hexdigest(),
                         "mean": self._mean.hex(), "scale": self._scale.hex(), "obs_dim": len(mean),
                         "action_transform": "tanh_then_clip_float32"}) + self._params)

    def __setattr__(self, name, value):
        raise AttributeError("frozen policy is immutable")

    def __delattr__(self, name):
        raise AttributeError("frozen policy is immutable")

    def state_bytes(self):
        return self._state

    def act(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (self._dimension,) or not np.all(np.isfinite(x)):
            raise ValueError("invalid observation")
        mean = np.frombuffer(self._mean, dtype="<f4")
        scale = np.frombuffer(self._scale, dtype="<f4")
        action = np.asarray(self._apply(self._variables, jnp.asarray((x - mean) / scale)), dtype=np.float32)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("policy produced invalid actions")
        return np.clip(action, -1, 1)


class TD3BCAdapter:
    capabilities = AdapterCapabilities(adapter_id=ID, fit_semantics="optimizer_steps",
                                       supports_online=False, primary_optimizer="critic")

    def __init__(self):
        self._initialized = False

    def _require(self):
        if not self._initialized:
            raise ValueError("uninitialized TD3+BC adapter")

    @property
    def optimizer_step_counts(self):
        self._require()
        return {"critic": int(self._state.critic.step), "actor": int(self._state.actor.step)}

    @property
    def optimizer_steps(self):
        return sum(self.optimizer_step_counts.values())

    def initialize(self, train_batch, seed, learning_settings=None):
        assert_training_allowed("initialize TD3+BC")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a nonnegative 64-bit integer")
        if learning_settings not in (None, {}):
            raise ValueError("this adapter has a fixed task configuration; overrides are not supported")
        obs, _, _, _ = _arrays(train_batch)
        if np.any(np.asarray(train_batch.origin) != 0):
            raise ValueError("normalization requires offline train observations only")
        self._mean = obs.mean(0)
        self._scale = obs.std(0) + np.float32(.001)
        self._seed = int(seed)
        self._dim = obs.shape[-1]
        self._args = replace(host.Args(), seed=int(seed), dataset="thrusterchain-v6", discount=.98)
        self._rng, self._state, (self._actor, self._critic) = _initialize(self._args, self._dim)
        self._metadata = {"primary_optimizer": "critic", "not_applicable_components": [],
                          "adapter": {"learning_settings": {}, "source_sha256": _source_hashes(),
                                      "args": asdict(self._args), "seed": self._seed, "obs_dim": self._dim,
                                      "numpy": np.__version__, "jax": jax.__version__, "flax": flax.__version__,
                                      "optax": optax.__version__, "reward_transform": "identity",
                                      "seed_mapping": "64-bit high and low words as JAX key data, then native fold-in",
                                      "normalization": "train observations mean; std + 0.001"}}
        self._initialized = True

    def update(self, batch):
        assert_training_allowed("update TD3+BC")
        self._require()
        obs, nxt, action, reward = _arrays(batch)
        if obs.shape[-1] != self._dim or np.any(np.asarray(batch.origin) != 0):
            raise ValueError("strict offline TD3+BC requires matching offline transition batches")
        native = Transition(jnp.asarray((obs - self._mean) / self._scale), jnp.asarray(action),
                            jnp.asarray(reward), jnp.asarray((nxt - self._mean) / self._scale),
                            jnp.asarray(batch.terminated, dtype=jnp.float32))
        next_rng, noise = jax.random.split(self._rng)
        next_step = int(self._state.critic.step) + 1
        state, metrics = host.td3_bc_update(self._args, self._actor.apply, self._critic.apply,
                                           self._state, native, next_step, noise, 1.)
        if not _finite_tree(state) or not _finite_tree(metrics):
            raise FloatingPointError("nonfinite update; previous state and RNG retained")
        self._state, self._rng = state, next_rng
        return {key: float(value) for key, value in metrics.items()}

    def checkpoint(self):
        self._require()
        state = self._state
        components = {
            "model": _pack({"actor": state.actor.params, "critic": state.critic.params}),
            "optimizer": _pack({"actor": state.actor.opt_state, "critic": state.critic.opt_state}),
            "target_networks": _pack({"actor": state.actor_target.params, "critic": state.critic_target.params}),
            "rng": _pack(self._rng),
            "preprocessing": _json({"mean": self._mean.tolist(), "scale": self._scale.tolist()}),
        }
        return LearnerCheckpoint(ID, components, self.optimizer_steps, self.optimizer_step_counts, self._metadata)

    def restore(self, checkpoint):
        assert_training_allowed("restore TD3+BC")
        if not isinstance(checkpoint, LearnerCheckpoint) or checkpoint.adapter_id != ID:
            raise ValueError("checkpoint identity mismatch")
        meta = checkpoint.metadata
        info = meta.get("adapter", {})
        if (info.get("source_sha256") != _source_hashes() or info.get("jax") != jax.__version__
                or info.get("flax") != flax.__version__ or info.get("optax") != optax.__version__
                or info.get("numpy") != np.__version__):
            raise ValueError("source or runtime version differs from checkpoint")
        dim, seed = info.get("obs_dim"), info.get("seed")
        if type(dim) is not int or dim < 1 or type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("invalid checkpoint dimension or seed")
        args = replace(host.Args(), seed=seed, dataset="thrusterchain-v6", discount=.98)
        if (info.get("args") != asdict(args) or info.get("learning_settings") != {}
                or info.get("seed_mapping") != "64-bit high and low words as JAX key data, then native fold-in"):
            raise ValueError("unsupported configuration")
        key, state, (actor, critic) = _initialize(args, dim)
        model = serialization.from_bytes({"actor": state.actor.params, "critic": state.critic.params}, checkpoint.components["model"])
        optimizer = serialization.from_bytes({"actor": state.actor.opt_state, "critic": state.critic.opt_state}, checkpoint.components["optimizer"])
        target = serialization.from_bytes({"actor": state.actor_target.params, "critic": state.critic_target.params}, checkpoint.components["target_networks"])
        rng = serialization.from_bytes(key, checkpoint.components["rng"])
        prep = json.loads(checkpoint.components["preprocessing"].decode())
        mean, scale = np.asarray(prep["mean"], np.float32), np.asarray(prep["scale"], np.float32)
        counts = dict(checkpoint.optimizer_step_counts)
        if (set(counts) != {"critic", "actor"} or counts["actor"] != counts["critic"] // args.policy_freq
                or mean.shape != (dim,) or scale.shape != (dim,) or np.any(scale <= 0)
                or not _finite_tree((model, optimizer, target, rng, mean, scale))):
            raise ValueError("invalid checkpoint counters or state")
        steps, delayed = counts["critic"], counts["actor"]
        state = state._replace(
            actor=state.actor.replace(params=model["actor"], opt_state=optimizer["actor"], step=delayed),
            critic=state.critic.replace(params=model["critic"], opt_state=optimizer["critic"], step=steps),
            actor_target=state.actor_target.replace(params=target["actor"], step=delayed),
            critic_target=state.critic_target.replace(params=target["critic"], step=delayed))
        candidate = TD3BCAdapter()
        candidate._args, candidate._rng, candidate._state = args, rng, state
        candidate._actor, candidate._critic = actor, critic
        candidate._dim, candidate._seed, candidate._mean, candidate._scale = dim, seed, mean, scale
        candidate._metadata, candidate._initialized = meta, True
        if candidate.checkpoint().to_bytes() != checkpoint.to_bytes():
            changed = [name for name in checkpoint.components
                       if candidate.checkpoint().components.get(name) != checkpoint.components[name]]
            raise ValueError(f"checkpoint did not round trip exactly; changed components: {changed}")
        self.__dict__.update(candidate.__dict__)

    def freeze(self):
        self._require()
        return FrozenTD3BCPolicy(self._state.actor.params, self._mean, self._scale)

    def act(self, observation, explore=False):
        if explore:
            raise ValueError("this adapter does not support online exploration")
        return self.freeze().act(observation)
