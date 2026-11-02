"""Strict-offline adapter for the repository's ReBRAC variant.

Its critic uses logged successor actions on every bootstrapping row. An
incomplete dataset is refused at initialization rather than silently changing
the Bellman target or the sample distribution.
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

from algorithms import rebrac as host
import runtime.networks as networks
import runtime.rewards as rewards
from runtime.networks import TransitionNA
from thrusterchain.datasets import assert_training_allowed
from thrusterchain.training_protocol import AdapterCapabilities, LearnerCheckpoint, TransitionBatch

ID = "repository_rebrac_strict_v5_1"


def _source_hashes():
    files = {"host": Path(host.__file__), "runtime_networks": Path(networks.__file__),
             "runtime_rewards": Path(rewards.__file__), "adapter": Path(__file__)}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


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
    actions = np.asarray(batch.actions, np.float32)
    next_actions = np.asarray(batch.next_actions, np.float32)
    reward = np.asarray(batch.rewards, np.float32)
    if (obs.ndim != 2 or nxt.shape != obs.shape or actions.shape != (len(batch), 4)
            or next_actions.shape != actions.shape or reward.shape != (len(batch),)
            or np.any(np.abs(actions) > 1) or np.any(np.abs(next_actions) > 1)
            or not _finite((obs, nxt, actions, next_actions, reward))
            or np.any(np.asarray(batch.origin) != 0)):
        raise ValueError("invalid or non-offline ReBRAC batch")
    if np.any(~np.asarray(batch.next_action_valid) & (np.asarray(batch.bootstrap_mask) != 0)):
        raise ValueError("ReBRAC requires recorded next actions for all bootstrapped rows")
    if mean is not None:
        if obs.shape[1:] != mean.shape:
            raise ValueError("observation dimension differs from fitted normalization")
        obs, nxt = (obs - mean) / scale, (nxt - mean) / scale
    return TransitionNA(jnp.asarray(obs), jnp.asarray(actions), jnp.asarray(reward),
                        jnp.asarray(nxt), jnp.asarray(batch.terminated, jnp.float32),
                        jnp.asarray(next_actions))


def _initialize(args, sample):
    """Native network/optimizer construction with an injective 64-bit root key.

    The runner supplies up to 63-bit seeds. Writing high and low words directly
    avoids truncating distinct seeds to the same 32-bit host PRNGKey input.
    For 32-bit seeds this is exactly PRNGKey(seed) before the native fold-in.
    """
    root = jnp.asarray([args.seed >> 32, args.seed & 0xFFFFFFFF], dtype=jnp.uint32)
    rng = jax.random.fold_in(root, 1414676809)
    rng, ka, kc = jax.random.split(rng, 3)
    obs, action = sample.obs[:1], sample.action[:1]
    actor = host.DetActor(action.shape[-1], args.hidden_dim, args.actor_ln, args.actor_n_hiddens)
    critic = host.EnsembleCritic(args.hidden_dim, 2, args.critic_ln, args.critic_n_hiddens)
    actor_params, critic_params = actor.init(ka, obs), critic.init(kc, obs, action)
    state = host.AgentTrainState(
        host.ActorTrainState.create(apply_fn=actor.apply, params=actor_params,
                                    target_params=actor_params, tx=networks.torch_adam(args.actor_learning_rate)),
        host.CriticTrainState.create(apply_fn=critic.apply, params=critic_params,
                                     target_params=critic_params, tx=networks.torch_adam(args.critic_learning_rate)))
    return rng, state, (actor, critic)


class FrozenReBRACPolicy:
    __slots__ = ("_state", "_mean", "_scale", "_dim", "_variables", "_apply")

    def __init__(self, actor, params, mean, scale):
        payload = _pack(params)
        object.__setattr__(self, "_mean", np.asarray(mean, dtype="<f4").tobytes())
        object.__setattr__(self, "_scale", np.asarray(scale, dtype="<f4").tobytes())
        object.__setattr__(self, "_dim", len(mean))
        object.__setattr__(self, "_variables", flax.core.freeze(params))
        object.__setattr__(self, "_apply", jax.jit(actor.apply))
        self._apply(self._variables, jnp.zeros(len(mean), jnp.float32)).block_until_ready()
        object.__setattr__(self, "_state", _json({"adapter": ID, "params_sha256": hashlib.sha256(payload).hexdigest(),
                           "mean": self._mean.hex(), "scale": self._scale.hex(), "obs_dim": len(mean),
                           "action_transform": "tanh_then_clip_float32"}) + payload)

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


class ReBRACAdapter:
    capabilities = AdapterCapabilities(adapter_id=ID, fit_semantics="optimizer_steps",
                                       supports_online=False, primary_optimizer="critic")

    def __init__(self):
        self._initialized = False

    def _require(self):
        if not self._initialized:
            raise ValueError("uninitialized ReBRAC adapter")

    @property
    def optimizer_step_counts(self):
        self._require()
        return {"critic": int(self._state.critic.step), "actor": int(self._state.actor.step)}

    @property
    def optimizer_steps(self):
        return sum(self.optimizer_step_counts.values())

    def initialize(self, train_batch, seed, learning_settings=None):
        assert_training_allowed("initialize ReBRAC")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a nonnegative 64-bit integer")
        if learning_settings not in (None, {}):
            raise ValueError("this adapter has a fixed task configuration")
        raw = _batch(train_batch)  # validates all training rows, not just a sampled minibatch
        observations = np.asarray(raw.obs, np.float32)
        self._mean = observations.mean(0)
        self._scale = observations.std(0) + np.float32(.001)
        self._seed, self._dim = int(seed), observations.shape[-1]
        self._args = replace(host.Args(), seed=int(seed), dataset="thrusterchain-v5", gamma=.98)
        sample = _batch(train_batch.select(slice(0, 1)), self._mean, self._scale)
        self._rng, self._state, (self._actor, self._critic) = _initialize(self._args, sample)
        self._metadata = {"primary_optimizer": "critic", "not_applicable_components": [],
                          "adapter": {"learning_settings": {}, "source_sha256": _source_hashes(),
                              "args": asdict(self._args), "seed": self._seed, "obs_dim": self._dim,
                              "numpy": np.__version__, "jax": jax.__version__, "flax": flax.__version__,
                              "optax": optax.__version__, "reward_transform": "identity",
                              "normalization": "train observations mean; std + 0.001",
                              "seed_mapping": "64-bit high and low words as JAX key data, then native fold-in",
                              "update_schedule": "native zero-based it: actor on critic updates 1, 3, ...",
                              "missing_next_action": "refuse bootstrap rows; terminal placeholders inert"}}
        self._initialized = True

    def update(self, batch):
        assert_training_allowed("update ReBRAC")
        self._require()
        native = _batch(batch, self._mean, self._scale)
        # Host's make_train_step supplies it before incrementing: first actor update is at it=0.
        iteration = int(self._state.critic.step)
        state, rng, metrics = host.rebrac_update(self._args, self._actor.apply, self._critic.apply,
                                                self._state, native, iteration, self._rng)
        if not _finite((state, rng, metrics)):
            raise FloatingPointError("nonfinite update; previous state and RNG retained")
        self._state, self._rng = state, rng
        return {key: float(value) for key, value in metrics.items()}

    def checkpoint(self):
        self._require()
        state = self._state
        components = {
            "model": _pack({"actor": state.actor.params, "critic": state.critic.params}),
            "optimizer": _pack({"actor": state.actor.opt_state, "critic": state.critic.opt_state}),
            "target_networks": _pack({"actor": state.actor.target_params, "critic": state.critic.target_params}),
            "rng": _pack(self._rng),
            "preprocessing": _json({"mean": self._mean.tolist(), "scale": self._scale.tolist()}),
        }
        return LearnerCheckpoint(ID, components, self.optimizer_steps, self.optimizer_step_counts, self._metadata)

    def restore(self, checkpoint):
        assert_training_allowed("restore ReBRAC")
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
        args = replace(host.Args(), seed=seed, dataset="thrusterchain-v5", gamma=.98)
        if info.get("args") != asdict(args) or info.get("learning_settings") != {}:
            raise ValueError("unsupported configuration")
        dummy = TransitionNA(jnp.zeros((1, dim)), jnp.zeros((1, 4)), jnp.zeros(1),
                             jnp.zeros((1, dim)), jnp.ones(1), jnp.zeros((1, 4)))
        key, state, (actor, critic) = _initialize(args, dummy)
        model = serialization.from_bytes({"actor": state.actor.params, "critic": state.critic.params}, checkpoint.components["model"])
        optimizer = serialization.from_bytes({"actor": state.actor.opt_state, "critic": state.critic.opt_state}, checkpoint.components["optimizer"])
        target = serialization.from_bytes({"actor": state.actor.target_params, "critic": state.critic.target_params}, checkpoint.components["target_networks"])
        rng = serialization.from_bytes(key, checkpoint.components["rng"])
        prep = json.loads(checkpoint.components["preprocessing"].decode())
        mean, scale = np.asarray(prep["mean"], np.float32), np.asarray(prep["scale"], np.float32)
        counts = dict(checkpoint.optimizer_step_counts)
        expected_actor = (counts.get("critic", -1) + args.policy_freq - 1) // args.policy_freq
        if (set(counts) != {"critic", "actor"} or counts["actor"] != expected_actor
                or mean.shape != (dim,) or scale.shape != (dim,) or np.any(scale <= 0)
                or not _finite((model, optimizer, target, rng, mean, scale))):
            raise ValueError("invalid checkpoint counters or state")
        state = state._replace(
            actor=state.actor.replace(params=model["actor"], opt_state=optimizer["actor"],
                                      target_params=target["actor"], step=counts["actor"]),
            critic=state.critic.replace(params=model["critic"], opt_state=optimizer["critic"],
                                        target_params=target["critic"], step=counts["critic"]))
        candidate = ReBRACAdapter()
        candidate._args, candidate._rng, candidate._state = args, rng, state
        candidate._actor, candidate._critic = actor, critic
        candidate._dim, candidate._seed, candidate._mean, candidate._scale = dim, seed, mean, scale
        candidate._metadata, candidate._initialized = metadata, True
        if candidate.checkpoint().to_bytes() != checkpoint.to_bytes():
            raise ValueError("checkpoint did not round trip exactly")
        self.__dict__.update(candidate.__dict__)

    def freeze(self):
        self._require()
        return FrozenReBRACPolicy(self._actor, self._state.actor.params, self._mean, self._scale)

    def act(self, observation, explore=False):
        if explore:
            raise ValueError("this adapter does not support online exploration")
        return self.freeze().act(observation)
