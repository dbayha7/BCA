"""Strict-offline adapter for this repository's IQL implementation.

The host performs value, critic/target, then actor updates. Its critic target
uses V before the value update; its detached advantage also uses that value.
This adapter preserves that variant rather than claiming canonical equivalence.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import flax
from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms import iql as host
import runtime.networks as networks
import runtime.rewards as rewards
from runtime.networks import Transition
from thrusterchain.datasets import assert_training_allowed
from thrusterchain.training_protocol import AdapterCapabilities, LearnerCheckpoint, TransitionBatch

ID = "repository_iql_strict_v6_1"


def _source_hashes():
    paths = {"host": Path(host.__file__), "runtime_networks": Path(networks.__file__),
             "runtime_rewards": Path(rewards.__file__), "adapter": Path(__file__)}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pack(value):
    def ordered(item):
        if isinstance(item, dict):
            return {key: ordered(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            return [ordered(element) for element in item]
        return item
    return serialization.msgpack_serialize(ordered(serialization.to_state_dict(value)))


def _finite(tree):
    return all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree))


def _batch(batch, mean=None, scale=None):
    if not isinstance(batch, TransitionBatch) or len(batch) < 1:
        raise ValueError("nonempty v5 transition batch required")
    obs = np.asarray(batch.observations, np.float32)
    nxt = np.asarray(batch.next_observations, np.float32)
    action = np.asarray(batch.actions, np.float32)
    reward = np.asarray(batch.rewards, np.float32)
    if (obs.ndim != 2 or nxt.shape != obs.shape or action.shape != (len(batch), 4)
            or reward.shape != (len(batch),) or np.any(np.abs(action) > 1)
            or not _finite((obs, nxt, action, reward)) or np.any(np.asarray(batch.origin) != 0)
            or not np.array_equal(np.asarray(batch.bootstrap_mask), ~np.asarray(batch.terminated))):
        raise ValueError("invalid or non-offline IQL transition batch")
    if mean is not None:
        if obs.shape[1:] != mean.shape:
            raise ValueError("observation dimension differs from fitted normalization")
        obs, nxt = (obs - mean) / scale, (nxt - mean) / scale
    return Transition(jnp.asarray(obs), jnp.asarray(action), jnp.asarray(reward),
                      jnp.asarray(nxt), jnp.asarray(batch.terminated, jnp.float32))


def _initialize(args, obs_dim):
    """Match native initialization for 32-bit seeds while retaining both seed words."""
    root = jnp.asarray([args.seed >> 32, args.seed & 0xffffffff], dtype=jnp.uint32)
    key = jax.random.fold_in(root, 1414676809)
    key, ka, kq, kv = jax.random.split(key, 4)
    dropout = 0.0 if args.actor_dropout is None else float(args.actor_dropout)
    actor = (host.DeterministicPolicy if args.iql_deterministic else host.GaussianPolicy)(obs_dim, 4, 1., dropout)
    q, v = host.TwinQ(obs_dim, 4), host.ValueFunction(obs_dim)
    obs, action = jnp.zeros(obs_dim), jnp.zeros(4)
    def make(k, net, inputs, tx):
        return TrainState.create(apply_fn=net.apply, params=net.init(k, *inputs), tx=tx)
    state = host.AgentTrainState(
        make(ka, actor, [obs], networks.torch_adam(optax.cosine_decay_schedule(args.actor_lr, args.num_updates))),
        make(kq, q, [obs, action], networks.torch_adam(args.qf_lr)),
        make(kq, q, [obs, action], networks.torch_adam(args.qf_lr)),
        make(kv, v, [obs], networks.torch_adam(args.vf_lr)))
    return state, key, actor, q, v


class FrozenIQLPolicy:
    """Deterministic Gaussian-mean inference and immutable preprocessing."""
    __slots__ = ("_state", "_mean", "_scale", "_dim", "_variables", "_apply")

    def __init__(self, actor, params, mean, scale):
        payload = _pack(params)
        object.__setattr__(self, "_mean", np.asarray(mean, dtype="<f4").tobytes())
        object.__setattr__(self, "_scale", np.asarray(scale, dtype="<f4").tobytes())
        object.__setattr__(self, "_dim", len(mean))
        object.__setattr__(self, "_variables", flax.core.freeze(params))
        object.__setattr__(self, "_apply", jax.jit(lambda p, x: actor.apply(p, x, deterministic=True)[0]))
        self._apply(self._variables, jnp.zeros(len(mean), jnp.float32)).block_until_ready()
        object.__setattr__(self, "_state", _json({"adapter": ID, "params_sha256": hashlib.sha256(payload).hexdigest(),
                           "mean": self._mean.hex(), "scale": self._scale.hex(), "obs_dim": len(mean),
                           "evaluation_action": "deterministic tanh mean clipped to [-1,1]"}) + payload)

    def __setattr__(self, name, value):
        raise AttributeError("frozen policy is immutable")

    def __delattr__(self, name):
        raise AttributeError("frozen policy is immutable")

    def state_bytes(self):
        return self._state

    def act(self, observation):
        obs = np.asarray(observation, np.float32)
        if obs.shape != (self._dim,) or not np.all(np.isfinite(obs)):
            raise ValueError("invalid observation")
        mean = np.frombuffer(self._mean, dtype="<f4")
        scale = np.frombuffer(self._scale, dtype="<f4")
        action = np.asarray(self._apply(self._variables, jnp.asarray((obs - mean) / scale)), np.float32)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("invalid policy action")
        return np.clip(action, -1, 1)


class IQLAdapter:
    capabilities = AdapterCapabilities(adapter_id=ID, fit_semantics="optimizer_steps",
                                       supports_online=False, primary_optimizer="critic")

    def __init__(self):
        self._initialized = False

    def _require(self):
        if not self._initialized:
            raise ValueError("uninitialized IQL adapter")

    @property
    def optimizer_step_counts(self):
        self._require()
        return {"critic": int(self._state.qf.step), "actor": int(self._state.actor.step),
                "value": int(self._state.vf.step)}

    @property
    def optimizer_steps(self):
        return sum(self.optimizer_step_counts.values())

    def initialize(self, train_batch, seed, learning_settings=None):
        assert_training_allowed("initialize IQL")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a nonnegative 64-bit integer")
        if learning_settings not in (None, {}):
            raise ValueError("this adapter has a fixed task configuration")
        observations = np.asarray(_batch(train_batch).obs, np.float32)
        self._mean, self._scale = observations.mean(0), observations.std(0) + np.float32(.001)
        self._seed, self._dim = int(seed), observations.shape[-1]
        self._args = replace(host.Args(), seed=int(seed), dataset="thrusterchain-v6", discount=.98)
        self._state, self._rng, self._actor, self._q, self._value = _initialize(self._args, self._dim)
        self._metadata = {"primary_optimizer": "critic", "not_applicable_components": [],
                          "adapter": {"learning_settings": {}, "source_sha256": _source_hashes(),
                              "args": asdict(self._args), "seed": self._seed, "obs_dim": self._dim,
                              "numpy": np.__version__, "jax": jax.__version__, "flax": flax.__version__,
                              "optax": optax.__version__, "reward_transform": "identity",
                              "normalization": "train observations mean; std + 0.001",
                              "seed_mapping": "64-bit high and low words as JAX key data, then native fold-in",
                              "rng_schedule": "runner samples batches separately; adapter splits its key into next key and dropout key",
                              "evaluation_action": "deterministic Gaussian mean clipped to [-1,1]",
                              "update_order": "value, critic, target critic, actor; pre-update value for target and advantage"}}
        self._initialized = True

    def update(self, batch):
        assert_training_allowed("update IQL")
        self._require()
        native = _batch(batch, self._mean, self._scale)
        rng, dropout = jax.random.split(self._rng)
        state, metrics = host.iql_update(self._args, self._actor.apply, self._q.apply,
            self._value.apply, self._state, native, int(self._state.qf.step) + 1, dropout)
        if not _finite((state, rng, metrics)):
            raise FloatingPointError("nonfinite update; previous state and RNG retained")
        self._state, self._rng = state, rng
        return {key: float(value) for key, value in metrics.items()}

    def checkpoint(self):
        self._require()
        state = self._state
        components = {
            "model": _pack({"actor": state.actor.params, "qf": state.qf.params, "vf": state.vf.params}),
            "optimizer": _pack({"actor": state.actor.opt_state, "qf": state.qf.opt_state, "vf": state.vf.opt_state}),
            "target_networks": _pack({"qf": state.qf_target.params}),
            "rng": _pack(self._rng),
            "preprocessing": _json({"mean": self._mean.tolist(), "scale": self._scale.tolist()}),
        }
        return LearnerCheckpoint(ID, components, self.optimizer_steps, self.optimizer_step_counts, self._metadata)

    def restore(self, checkpoint):
        assert_training_allowed("restore IQL")
        if not isinstance(checkpoint, LearnerCheckpoint) or checkpoint.adapter_id != ID:
            raise ValueError("checkpoint identity mismatch")
        metadata = checkpoint.metadata
        info = metadata.get("adapter", {})
        if (info.get("source_sha256") != _source_hashes() or info.get("jax") != jax.__version__
                or info.get("flax") != flax.__version__ or info.get("optax") != optax.__version__
                or info.get("numpy") != np.__version__):
            raise ValueError("source or runtime version differs from checkpoint")
        dim, seed = info.get("obs_dim"), info.get("seed")
        if type(dim) is not int or dim < 1 or type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("invalid checkpoint dimension or seed")
        args = replace(host.Args(), seed=seed, dataset="thrusterchain-v6", discount=.98)
        if info.get("args") != asdict(args) or info.get("learning_settings") != {}:
            raise ValueError("unsupported configuration")
        state, key, actor, q, value = _initialize(args, dim)
        model = serialization.from_bytes({"actor": state.actor.params, "qf": state.qf.params, "vf": state.vf.params}, checkpoint.components["model"])
        optimizer = serialization.from_bytes({"actor": state.actor.opt_state, "qf": state.qf.opt_state, "vf": state.vf.opt_state}, checkpoint.components["optimizer"])
        target = serialization.from_bytes({"qf": state.qf_target.params}, checkpoint.components["target_networks"])
        rng = serialization.from_bytes(key, checkpoint.components["rng"])
        prep = json.loads(checkpoint.components["preprocessing"].decode())
        mean, scale = np.asarray(prep["mean"], np.float32), np.asarray(prep["scale"], np.float32)
        counts = dict(checkpoint.optimizer_step_counts)
        if (set(counts) != {"critic", "actor", "value"}
                or counts["critic"] != counts["actor"] or counts["critic"] != counts["value"]
                or mean.shape != (dim,) or scale.shape != (dim,) or np.any(scale <= 0)
                or not _finite((model, optimizer, target, rng, mean, scale))):
            raise ValueError("invalid checkpoint counters or state")
        step = counts["critic"]
        state = state._replace(
            actor=state.actor.replace(params=model["actor"], opt_state=optimizer["actor"], step=step),
            qf=state.qf.replace(params=model["qf"], opt_state=optimizer["qf"], step=step),
            qf_target=state.qf_target.replace(params=target["qf"], step=step),
            vf=state.vf.replace(params=model["vf"], opt_state=optimizer["vf"], step=step))
        candidate = IQLAdapter()
        candidate._args, candidate._rng, candidate._state = args, rng, state
        candidate._actor, candidate._q, candidate._value = actor, q, value
        candidate._dim, candidate._seed, candidate._mean, candidate._scale = dim, seed, mean, scale
        candidate._metadata, candidate._initialized = metadata, True
        if candidate.checkpoint().to_bytes() != checkpoint.to_bytes():
            raise ValueError("checkpoint did not round trip exactly")
        self.__dict__.update(candidate.__dict__)

    def freeze(self):
        self._require()
        return FrozenIQLPolicy(self._actor, self._state.actor.params, self._mean, self._scale)

    def act(self, observation, explore=False):
        if explore:
            raise ValueError("this adapter does not support online exploration")
        return self.freeze().act(observation)
