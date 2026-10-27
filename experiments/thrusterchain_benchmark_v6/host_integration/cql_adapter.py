"""Strict-offline v5 adapter for the repository's continuous CQL variant.

This preserves the repository host's losses, update ordering and default CQL
configuration, with bc_steps=0 to avoid its non-dataset warm-up branch. Reward
is the raw v5 task reward; the target discount is .98. Evaluation uses the
deterministic tanh of the Gaussian mean. It does not implement online updates.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import flax
from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms import cql as host
import runtime.networks as networks
import runtime.rewards as rewards
from runtime.networks import Transition
from thrusterchain.datasets import assert_training_allowed
from thrusterchain.training_protocol import AdapterCapabilities, LearnerCheckpoint
from . import td3bc_adapter as helpers

ID = "repository_cql_strict_v6_1"


def _source_hashes():
    files = {"host": Path(host.__file__), "runtime_networks": Path(networks.__file__),
             "runtime_rewards": Path(rewards.__file__), "shared_helpers": Path(helpers.__file__),
             "adapter": Path(__file__)}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def _args_record(args):
    # The native unbounded clipping configuration uses actual +/- infinity;
    # JSON metadata represents those exact values symbolically, not as NaN.
    return {key: ("+inf" if value > 0 else "-inf") if isinstance(value, float) and math.isinf(value) else value
            for key, value in asdict(args).items()}


def _initialize(args, obs_dim):
    """Replicate the host initializer using both words of the 64-bit seed."""
    root = jnp.asarray([args.seed >> 32, args.seed & 0xFFFFFFFF], dtype=jnp.uint32)
    rng = jax.random.fold_in(root, 1414676809)
    rng, ka, k1, k2 = jax.random.split(rng, 4)
    actor = host.TanhGaussianPolicy(obs_dim, 4, 1., args.orthogonal_init,
                                    args.policy_log_std_multiplier, -1.)
    critic1 = host.FullyConnectedQFunction(obs_dim, 4, args.orthogonal_init, args.q_n_hidden_layers)
    critic2 = host.FullyConnectedQFunction(obs_dim, 4, args.orthogonal_init, 3)
    obs, action = jnp.zeros(obs_dim), jnp.zeros(4)

    def make(key, net, inputs, lr):
        return TrainState.create(apply_fn=net.apply, params=net.init(key, *inputs), tx=host.C.torch_adam(lr))

    def scalar(value, lr):
        return TrainState.create(apply_fn=host.C.identity,
                                 params={"constant": jnp.asarray(value, jnp.float32)}, tx=host.C.torch_adam(lr))

    state = host.AgentTrainState(make(ka, actor, [obs], args.policy_lr),
        make(k1, critic1, [obs, action], args.qf_lr), make(k2, critic2, [obs, action], args.qf_lr),
        make(k1, critic1, [obs, action], args.qf_lr), make(k2, critic2, [obs, action], args.qf_lr),
        scalar(0., args.policy_lr), scalar(1., args.qf_lr))
    return rng, state, (actor, critic1, critic2)


class FrozenCQLPolicy:
    __slots__ = ("_state", "_mean", "_scale", "_dim", "_variables", "_apply")

    def __init__(self, actor, params, mean, scale):
        payload = helpers._pack(params)
        object.__setattr__(self, "_mean", np.asarray(mean, dtype="<f4").tobytes())
        object.__setattr__(self, "_scale", np.asarray(scale, dtype="<f4").tobytes())
        object.__setattr__(self, "_dim", len(mean))
        object.__setattr__(self, "_variables", flax.core.freeze(params))
        object.__setattr__(self, "_apply", jax.jit(actor.apply))
        self._apply(self._variables, jnp.zeros(len(mean), jnp.float32))[0].block_until_ready()
        state = {"adapter": ID, "params_sha256": hashlib.sha256(payload).hexdigest(),
                 "mean": self._mean.hex(), "scale": self._scale.hex(),
                 "observation_dim": len(mean), "mode": "deterministic tanh of Gaussian mean"}
        object.__setattr__(self, "_state", json.dumps(state, sort_keys=True, separators=(",", ":")).encode() + payload)

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
        mu, _ = self._apply(self._variables, jnp.asarray((obs - mean) / scale))
        action = np.asarray(jnp.tanh(mu), np.float32)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("invalid policy action")
        return np.clip(action, -1, 1)


class CQLAdapter:
    capabilities = AdapterCapabilities(adapter_id=ID, fit_semantics="optimizer_steps",
                                       supports_online=False, primary_optimizer="critic1")

    def __init__(self):
        self._initialized = False

    def _require(self):
        if not self._initialized:
            raise ValueError("uninitialized CQL adapter")

    @property
    def optimizer_step_counts(self):
        self._require()
        return {name: int(getattr(self._state, name).step)
                for name in ("critic1", "critic2", "actor", "log_alpha", "log_alpha_prime")}

    @property
    def optimizer_steps(self):
        return sum(self.optimizer_step_counts.values())

    def initialize(self, train_batch, seed, learning_settings=None):
        assert_training_allowed("initialize CQL")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a nonnegative 64-bit integer")
        if learning_settings not in (None, {}):
            raise ValueError("this CQL adapter has fixed settings including bc_steps=0")
        obs, _, _, _ = helpers._arrays(train_batch)
        if np.any(np.asarray(train_batch.origin) != 0):
            raise ValueError("normalization requires offline train observations only")
        self._mean, self._scale = obs.mean(0), obs.std(0) + np.float32(.001)
        self._seed, self._dim = int(seed), obs.shape[-1]
        self._args = replace(host.Args(), seed=int(seed), dataset="thrusterchain-v6", discount=.98, bc_steps=0,
                             cql_reward_scale=1., cql_reward_bias=0.)
        self._rng, self._state, (self._actor, self._critic1, self._critic2) = _initialize(self._args, self._dim)
        self._metadata = {"primary_optimizer": "critic1", "not_applicable_components": [],
            "adapter": {"learning_settings": {}, "source_sha256": _source_hashes(),
            "args": _args_record(self._args), "seed": self._seed, "obs_dim": self._dim,
            "numpy": np.__version__, "jax": jax.__version__, "flax": flax.__version__, "optax": optax.__version__,
            "reward_transform": "identity", "normalization": "train observations mean; std + 0.001",
            "seed_mapping": "64-bit high and low words as JAX key data, then native fold-in",
            "evaluation_action": "deterministic tanh of Gaussian mean", "target_entropy": -4.0,
            "variant": "repository CQL, twin critics, importance sampled conservative gap, no warmup"}}
        self._initialized = True

    def update(self, batch):
        assert_training_allowed("update CQL")
        self._require()
        obs, nxt, action, reward = helpers._arrays(batch)
        if obs.shape[-1] != self._dim or np.any(np.asarray(batch.origin) != 0):
            raise ValueError("strict offline CQL requires matching offline transition batches")
        native = Transition(jnp.asarray((obs - self._mean) / self._scale), jnp.asarray(action),
            jnp.asarray(reward), jnp.asarray((nxt - self._mean) / self._scale),
            jnp.asarray(batch.terminated, dtype=jnp.float32))
        next_rng, update_rng = jax.random.split(self._rng)
        step = int(self._state.critic1.step) + 1
        state, metrics = host.cql_update(self._args, self._actor.apply, self._critic1.apply,
            self._critic2.apply, self._state, native, step, update_rng, 1., -4.)
        if not helpers._finite_tree((state, metrics, next_rng)):
            raise FloatingPointError("nonfinite update; previous state and RNG retained")
        self._state, self._rng = state, next_rng
        return {key: float(value) for key, value in metrics.items()}

    def checkpoint(self):
        self._require()
        names = ("actor", "critic1", "critic2", "log_alpha", "log_alpha_prime")
        state = self._state
        components = {"model": helpers._pack({n: getattr(state, n).params for n in names}),
            "optimizer": helpers._pack({n: getattr(state, n).opt_state for n in names}),
            "target_networks": helpers._pack({n: getattr(state, n + "_target").params for n in ("critic1", "critic2")}),
            "rng": helpers._pack(self._rng),
            "preprocessing": json.dumps({"mean": self._mean.tolist(), "scale": self._scale.tolist()},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()}
        return LearnerCheckpoint(ID, components, self.optimizer_steps, self.optimizer_step_counts, self._metadata)

    def restore(self, checkpoint):
        assert_training_allowed("restore CQL")
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
        args = replace(host.Args(), seed=seed, dataset="thrusterchain-v6", discount=.98, bc_steps=0, cql_reward_scale=1., cql_reward_bias=0.)
        if (info.get("args") != _args_record(args) or info.get("learning_settings") != {}
                or info.get("seed_mapping") != "64-bit high and low words as JAX key data, then native fold-in"):
            raise ValueError("unsupported CQL configuration")
        key, state, (actor, critic1, critic2) = _initialize(args, dim)
        names = ("actor", "critic1", "critic2", "log_alpha", "log_alpha_prime")
        model = serialization.from_bytes({n: getattr(state, n).params for n in names}, checkpoint.components["model"])
        optimizer = serialization.from_bytes({n: getattr(state, n).opt_state for n in names}, checkpoint.components["optimizer"])
        target = serialization.from_bytes({n: getattr(state, n + "_target").params for n in ("critic1", "critic2")}, checkpoint.components["target_networks"])
        rng = serialization.from_bytes(key, checkpoint.components["rng"])
        prep = json.loads(checkpoint.components["preprocessing"].decode())
        mean, scale = np.asarray(prep["mean"], np.float32), np.asarray(prep["scale"], np.float32)
        counts = dict(checkpoint.optimizer_step_counts)
        if (set(counts) != set(names) or len(set(counts.values())) != 1 or mean.shape != (dim,)
                or scale.shape != (dim,) or np.any(scale <= 0)
                or not helpers._finite_tree((model, optimizer, target, rng, mean, scale))):
            raise ValueError("invalid checkpoint counters or state")
        replacements = {n: getattr(state, n).replace(params=model[n], opt_state=optimizer[n], step=counts[n]) for n in names}
        replacements.update({n + "_target": getattr(state, n + "_target").replace(params=target[n], step=counts[n]) for n in ("critic1", "critic2")})
        state = state._replace(**replacements)
        candidate = CQLAdapter()
        candidate._args, candidate._rng, candidate._state = args, rng, state
        candidate._actor, candidate._critic1, candidate._critic2 = actor, critic1, critic2
        candidate._dim, candidate._seed, candidate._mean, candidate._scale = dim, seed, mean, scale
        candidate._metadata, candidate._initialized = meta, True
        if candidate.checkpoint().to_bytes() != checkpoint.to_bytes():
            raise ValueError("checkpoint did not round trip exactly")
        self.__dict__.update(candidate.__dict__)

    def freeze(self):
        self._require()
        return FrozenCQLPolicy(self._actor, self._state.actor.params, self._mean, self._scale)

    def act(self, observation, explore=False):
        if explore:
            raise ValueError("this adapter does not support online exploration")
        return self.freeze().act(observation)
