"""Pinned unified UniFloral TD3+BC composition for the v6 offline runner.

One runner update is one native actor-first outer step: one actor optimizer
step and two critic optimizer steps on the same supplied batch. The runner
samples rows; the upstream inner sampler is replaced only at that boundary.
"""
import ast
from collections import namedtuple
from dataclasses import asdict, dataclass, replace
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import types

import distrax
import flax
import flax.linen as nn
from flax import serialization
from flax.linen.initializers import constant, uniform
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from thrusterchain.datasets import assert_training_allowed
from thrusterchain.training_protocol import AdapterCapabilities, LearnerCheckpoint
from . import td3bc_adapter as helpers

ID = "unifloral_unified_td3bc_outer_v6_1"
REVISION = "f2dc1278eae18ed3e22c92255119c54885414d17"
SOURCE_SHA256 = "14a9bcdbc6d96fad85544a9d98957e98a7e46db13b89d74d54e3c5ce246517c8"
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "baselines/unifloral/algorithms/unifloral.py"
Transition = namedtuple("Transition", "obs action reward next_obs next_action done")


class _JaxCompat:
    """Local alias for removed jax.tree_map; never mutates the JAX module."""
    tree_map = staticmethod(jax.tree_util.tree_map)

    def __getattr__(self, name):
        return getattr(jax, name)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native(external_batch):
    """Compile hash-checked upstream definitions without CLI and D4RL imports.

    No computational definition is rewritten. The only compatibility alias is
    jax.tree_map -> jax.tree_util.tree_map. In external-batch mode the local
    sampler function is replaced by identity after definitions are loaded.
    """
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("pinned unified UniFloral source hash mismatch")
    parsed = ast.parse(raw.decode("utf8"), filename=str(SOURCE))
    names = {"Args", "AgentTrainState", "sym", "SoftQNetwork", "VectorQ",
             "StateValueFunction", "Actor", "EntropyCoef", "create_train_state",
             "sample_from_buffer", "make_train_step"}
    found, nodes = set(), []
    for node in parsed.body:
        name = node.name if isinstance(node, (ast.ClassDef, ast.FunctionDef)) else None
        if isinstance(node, ast.Assign):
            targets = [x.id for x in node.targets if isinstance(x, ast.Name)]
            name = next((x for x in targets if x in names), None)
        if name in names:
            found.add(name)
            nodes.append(node)
    if found != names:
        raise ValueError(f"unexpected pinned computational definitions: {sorted(names ^ found)}")
    module = types.ModuleType("unifloral_unified_v6_external" if external_batch else "unifloral_unified_v6_reference")
    module.__dict__.update(dict(dataclass=dataclass, namedtuple=namedtuple, partial=partial,
        nn=nn, constant=constant, uniform=uniform, TrainState=TrainState, distrax=distrax,
        jax=_JaxCompat(), jnp=jnp, optax=optax, Transition=Transition))
    # Dataclass needs an importable defining module for annotation handling.
    import sys
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), module.__dict__)
    if external_batch:
        module.sample_from_buffer = lambda buffer, batch_size, rng: buffer
    return module


HOST = _native(True)


def _version_tuple(version):
    import re
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        raise ValueError("cannot determine runtime constraint compatibility")
    return tuple(map(int, match.groups()))


def _runtime():
    return {"numpy": np.__version__, "jax": jax.__version__, "flax": flax.__version__,
            "optax": optax.__version__, "distrax": importlib.metadata.version("distrax"),
            "tensorflow_probability": importlib.metadata.version("tensorflow-probability"),
            "upstream_constraints_met": _version_tuple(jax.__version__) < (0, 6) and _version_tuple(np.__version__) < (2, 0),
            "constraint_check_scope": "JAX and NumPy upper bounds only; other upstream requirements are not certified"}


def _source_hashes():
    return {"pinned_unified": _sha(SOURCE), "adapter": _sha(__file__),
            "shared_helpers": _sha(helpers.__file__)}


def _args(seed, batch_size=256):
    # One explicit scalar selection from the pinned unified TD3+BC preset.
    return replace(HOST.Args(), seed=seed, dataset="thrusterchain-v6", algorithm="unifloral_td3bc_v6",
        lr=3e-4, actor_lr=3e-4, batch_size=batch_size, gamma=.98, actor_num_layers=2,
        actor_layer_width=256, actor_ln=False, critic_num_layers=2, critic_layer_width=256,
        critic_ln=False, num_critics=2, deterministic=True, actor_bc_coef=1., actor_q_coef=2.5,
        aggregate_q="first", normalize_q_loss=True, critic_bc_coef=0., diversity_coef=0.,
        num_critic_updates_per_step=2, use_awr=False, use_value_target=False,
        use_entropy_loss=False, use_target_actor=True, dataset_sample_ratio=1.)


def _stats(mean, std):
    o = jnp.asarray(mean, jnp.float32)
    s = jnp.asarray(std, jnp.float32)
    z = jnp.zeros(4, jnp.float32)
    return (Transition(o, z, jnp.float32(0), o, z, jnp.float32(0)),
            Transition(s, z, jnp.float32(0), s, z, jnp.float32(0)))


def _initialize(args, mean, std):
    """Pinned main's actor/critic initialization with explicit 64-bit key data."""
    data_mean, data_std = _stats(mean, std)
    actor = HOST.Actor(args=args, data_mean=data_mean, data_std=data_std, num_actions=4)
    critic = HOST.VectorQ(args=args, data_mean=data_mean, data_std=data_std)
    rng = jnp.asarray([args.seed >> 32, args.seed & 0xffffffff], jnp.uint32)
    rng, actor_key = jax.random.split(rng)
    dummy_obs, dummy_action = jnp.zeros(len(mean)), jnp.zeros(4)
    actor_state = HOST.create_train_state(args, actor_key, actor, [dummy_obs], True)
    actor_target = HOST.create_train_state(args, actor_key, actor, [dummy_obs], True)
    rng, critic_key = jax.random.split(rng)
    critic_state = HOST.create_train_state(args, critic_key, critic, [dummy_obs, dummy_action],
                                            steps=args.num_updates * args.num_critic_updates_per_step)
    critic_target = HOST.create_train_state(args, critic_key, critic, [dummy_obs, dummy_action])
    return rng, HOST.AgentTrainState(actor_state, actor_target, critic_state, critic_target, None, None), actor, critic


def _transition(batch):
    obs, nxt, action, reward = helpers._arrays(batch)
    if np.any(np.asarray(batch.origin) != 0):
        raise ValueError("unified adapter accepts offline data only")
    return Transition(jnp.asarray(obs), jnp.asarray(action), jnp.asarray(reward), jnp.asarray(nxt),
                      jnp.asarray(batch.next_actions, dtype=jnp.float32),
                      jnp.asarray(batch.terminated, dtype=jnp.float32))


def _run_step(module, args, actor, critic, rng, state, batch):
    step = module.make_train_step(args, actor.apply, critic.apply, None, None, batch, None)
    return step((rng, state, None), None)


class FrozenUnifiedPolicy:
    __slots__ = ("_state", "_variables", "_apply", "_dim")

    def __init__(self, actor, params, mean, std):
        payload = helpers._pack(params)
        object.__setattr__(self, "_variables", flax.core.freeze(params))
        object.__setattr__(self, "_apply", jax.jit(lambda p, x: actor.apply(p, x).mean()))
        object.__setattr__(self, "_dim", len(mean))
        self._apply(self._variables, jnp.zeros(len(mean), jnp.float32)).block_until_ready()
        data = {"adapter": ID, "params_sha256": hashlib.sha256(payload).hexdigest(),
                "mean": np.asarray(mean, dtype="<f4").tobytes().hex(),
                "std": np.asarray(std, dtype="<f4").tobytes().hex(),
                "action": "deterministic actor mean"}
        object.__setattr__(self, "_state", json.dumps(data, sort_keys=True, separators=(",", ":")).encode() + payload)

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
        action = np.asarray(self._apply(self._variables, jnp.asarray(obs)), np.float32)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("invalid policy action")
        return np.clip(action, -1, 1)


class UnifiedTD3BCAdapter:
    capabilities = AdapterCapabilities(adapter_id=ID, fit_semantics="optimizer_steps",
                                       supports_online=False, primary_optimizer="actor")

    def __init__(self):
        self._initialized = False
        self._compiled = {}

    def _require(self):
        if not self._initialized:
            raise ValueError("uninitialized unified UniFloral adapter")

    @property
    def optimizer_step_counts(self):
        self._require()
        return {"actor": int(self._state.actor.step), "critic": int(self._state.vec_q.step)}

    @property
    def optimizer_steps(self):
        return sum(self.optimizer_step_counts.values())

    def initialize(self, train_batch, seed, learning_settings=None):
        assert_training_allowed("initialize unified UniFloral")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a nonnegative 64-bit integer")
        settings = {} if learning_settings is None else learning_settings
        if type(settings) is not dict or set(settings) - {"allow_outside_upstream_constraints"} or type(settings.get("allow_outside_upstream_constraints", False)) is not bool:
            raise ValueError("only allow_outside_upstream_constraints boolean is configurable")
        runtime = _runtime()
        if not runtime["upstream_constraints_met"] and not settings.get("allow_outside_upstream_constraints", False):
            raise ValueError("runtime is outside upstream JAX<0.6/NumPy<2 constraints; explicit opt-in required")
        batch = _transition(train_batch)
        mean = jnp.mean(batch.obs, axis=0)
        std = jnp.nan_to_num(jnp.std(batch.obs, axis=0), nan=1.)
        self._mean, self._std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self._seed, self._dim = int(seed), int(batch.obs.shape[-1])
        self._args = _args(self._seed)
        self._rng, self._state, self._actor, self._critic = _initialize(self._args, self._mean, self._std)
        self._metadata = {"primary_optimizer": "actor", "not_applicable_components": [],
            "adapter": {"source_sha256": _source_hashes(), "revision": REVISION, "args": asdict(self._args),
                "seed": self._seed, "obs_dim": self._dim, "runtime": runtime, "learning_settings": settings,
                "composition": "unified TD3+BC: actor first, two critic steps on same batch, then target updates",
                "sampling": "v6 runner samples once; internal sampler boundary is identity; native RNG split retained",
                "batch_size_semantics": "args is the architecture template; actual batch size is the runner budget",
                "normalization": "train observations: native mean and std plus 0.001 in model",
                "reward_transform": "identity", "seed_mapping": "JAX high and low 32-bit key words",
                "compatibility": "unmodified native computational AST; local jax.tree_map alias"}}
        self._initialized = True

    def update(self, batch):
        assert_training_allowed("update unified UniFloral")
        self._require()
        native = _transition(batch)
        if native.obs.shape[1:] != (self._dim,):
            raise ValueError("observation dimension differs from training")
        count = len(batch)
        if count not in self._compiled:
            args = replace(self._args, batch_size=count)
            def run(rng, state, selected):
                return _run_step(HOST, args, self._actor, self._critic, rng, state, selected)
            self._compiled[count] = jax.jit(run)
        (rng, state, _), metrics = self._compiled[count](self._rng, self._state, native)
        if not helpers._finite_tree((rng, state, metrics)):
            raise FloatingPointError("nonfinite update; previous state and RNG retained")
        self._rng, self._state = rng, state
        return {key: float(value) for key, value in metrics.items()}

    def checkpoint(self):
        self._require()
        state = self._state
        components = {"model": helpers._pack({"actor": state.actor.params, "critic": state.vec_q.params}),
            "optimizer": helpers._pack({"actor": state.actor.opt_state, "critic": state.vec_q.opt_state}),
            "target_networks": helpers._pack({"actor": state.actor_target.params, "critic": state.vec_q_target.params}),
            "rng": helpers._pack(self._rng),
            "preprocessing": json.dumps({"mean": self._mean.tolist(), "std": self._std.tolist()},
                                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode()}
        return LearnerCheckpoint(ID, components, self.optimizer_steps, self.optimizer_step_counts, self._metadata)

    def restore(self, checkpoint):
        assert_training_allowed("restore unified UniFloral")
        if not isinstance(checkpoint, LearnerCheckpoint) or checkpoint.adapter_id != ID:
            raise ValueError("checkpoint identity mismatch")
        meta = checkpoint.metadata
        info = meta.get("adapter", {})
        if info.get("source_sha256") != _source_hashes() or info.get("runtime") != _runtime() or info.get("revision") != REVISION:
            raise ValueError("source or runtime version differs from checkpoint")
        settings = info.get("learning_settings")
        if (type(settings) is not dict or set(settings) - {"allow_outside_upstream_constraints"}
                or type(settings.get("allow_outside_upstream_constraints", False)) is not bool
                or (not _runtime()["upstream_constraints_met"] and not settings.get("allow_outside_upstream_constraints", False))):
            raise ValueError("checkpoint runtime opt-in is missing or invalid")
        seed, dim = info.get("seed"), info.get("obs_dim")
        if type(seed) is not int or not 0 <= seed < 2**64 or type(dim) is not int or dim < 1:
            raise ValueError("invalid checkpoint dimension or seed")
        args = _args(seed)
        if info.get("args") != asdict(args):
            raise ValueError("unsupported unified composition")
        prep = json.loads(checkpoint.components["preprocessing"].decode())
        mean, std = np.asarray(prep["mean"], np.float32), np.asarray(prep["std"], np.float32)
        if mean.shape != (dim,) or std.shape != (dim,) or np.any(std < 0) or not helpers._finite_tree((mean, std)):
            raise ValueError("invalid preprocessing state")
        key, state, actor, critic = _initialize(args, mean, std)
        model = serialization.from_bytes({"actor": state.actor.params, "critic": state.vec_q.params}, checkpoint.components["model"])
        optimizer = serialization.from_bytes({"actor": state.actor.opt_state, "critic": state.vec_q.opt_state}, checkpoint.components["optimizer"])
        target = serialization.from_bytes({"actor": state.actor_target.params, "critic": state.vec_q_target.params}, checkpoint.components["target_networks"])
        rng = serialization.from_bytes(key, checkpoint.components["rng"])
        counts = dict(checkpoint.optimizer_step_counts)
        if set(counts) != {"actor", "critic"} or counts["critic"] != 2 * counts["actor"] or not helpers._finite_tree((model, optimizer, target, rng)):
            raise ValueError("invalid unified optimizer counters or state")
        state = state._replace(actor=state.actor.replace(params=model["actor"], opt_state=optimizer["actor"], step=counts["actor"]),
            vec_q=state.vec_q.replace(params=model["critic"], opt_state=optimizer["critic"], step=counts["critic"]),
            actor_target=state.actor_target.replace(params=target["actor"], step=counts["actor"]),
            vec_q_target=state.vec_q_target.replace(params=target["critic"], step=counts["actor"]))
        candidate = UnifiedTD3BCAdapter()
        candidate._args, candidate._rng, candidate._state = args, rng, state
        candidate._actor, candidate._critic = actor, critic
        candidate._dim, candidate._seed, candidate._mean, candidate._std = dim, seed, mean, std
        candidate._metadata, candidate._initialized = meta, True
        if candidate.checkpoint().to_bytes() != checkpoint.to_bytes():
            raise ValueError("checkpoint did not round trip exactly")
        self.__dict__.update(candidate.__dict__)

    def freeze(self):
        self._require()
        return FrozenUnifiedPolicy(self._actor, self._state.actor.params, self._mean, self._std)

    def act(self, observation, explore=False):
        if explore:
            raise ValueError("this adapter does not support online exploration")
        return self.freeze().act(observation)
