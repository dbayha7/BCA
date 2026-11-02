"""Chunked native learner updates with the v6 runner's PCG64 sampling contract.

The adapter's native loss, optimization, model state and random-key schedule are
not rewritten. The same host functions execute within a JAX scan. Pre-sampled
indices use the runner's PCG64 stream; a successful chunk returns a standard
LearnerCheckpoint containing the advanced sampler state. Chunk failures do not
return a checkpoint. Numerical parity with eager execution is tested separately.
"""
import hashlib
import importlib
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from runtime.networks import Transition
from thrusterchain.training_protocol import LearnerCheckpoint, OfflineBudget
from thrusterchain.training_runner import (
    RunnerError, _capabilities, _checkpoint, _counts_delta, _dataset, _hash,
    _identity, _restore, _rng, _seed, _settings, _with_runner_state, train_transitions,
)

ENGINE = "compiled-native-scan-v1"
METHODS = {
    "td3bc": ("host_integration.td3bc_adapter", "TD3BCAdapter"),
    "iql": ("host_integration.iql_adapter", "IQLAdapter"),
    "cql": ("host_integration.cql_adapter", "CQLAdapter"),
    "rebrac": ("host_integration.rebrac_adapter", "ReBRACAdapter"),
    "unifloral-unified": ("host_integration.unifloral_adapter", "UnifiedTD3BCAdapter"),
}
_PROGRAMS = {}


def _engine_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _factory(method):
    if method not in METHODS:
        raise RunnerError("unsupported compiled learner")
    module, cls = METHODS[method]
    return getattr(importlib.import_module(module), cls)


def _native_dataset(model, method, batch):
    if method == "unifloral-unified":
        return importlib.import_module("host_integration.unifloral_adapter")._transition(batch)
    if method == "rebrac":
        return importlib.import_module("host_integration.rebrac_adapter")._batch(batch, model._mean, model._scale)
    # Match the adapters' NumPy float32 preprocessing before transferring to JAX.
    obs = np.asarray(batch.observations, np.float32)
    nxt = np.asarray(batch.next_observations, np.float32)
    action = np.asarray(batch.actions, np.float32)
    reward = np.asarray(batch.rewards, np.float32)
    if (obs.ndim != 2 or nxt.shape != obs.shape or action.shape != (len(batch), 4)
            or reward.shape != (len(batch),) or np.any(np.abs(action) > 1)
            or any(not np.all(np.isfinite(x)) for x in (obs, nxt, action, reward))
            or np.any(np.asarray(batch.origin) != 0)
            or not np.array_equal(batch.bootstrap_mask, ~batch.terminated)):
        raise RunnerError("invalid offline training data for compiled learner")
    return Transition(jnp.asarray((obs - model._mean) / model._scale), jnp.asarray(action),
                      jnp.asarray(reward), jnp.asarray((nxt - model._mean) / model._scale),
                      jnp.asarray(batch.terminated, jnp.float32))


def _program_key(model, method, batch_size):
    config = asdict(model._args)
    # Initialization consumes the seed; none of these native update functions
    # reads args.seed. Dynamic model/RNG values are supplied on every call.
    config["seed"] = 0
    normalization = None
    if method == "unifloral-unified":
        normalization = hashlib.sha256(np.asarray(model._mean, np.float32).tobytes()
                                       + np.asarray(model._std, np.float32).tobytes()).hexdigest()
    return (method, repr(sorted(config.items())), int(model._dim), int(batch_size), normalization,
            jax.default_backend(), str(jax.config.jax_default_matmul_precision))


def _canonical_state(state, template):
    """Reuse equivalent static optimizer objects so chunk resumes share JIT cache.

    Numeric model, target, optimizer and counter leaves all come from `state`.
    Template static functions are used only for identical method/config keys.
    """
    values = {}
    for name in state._fields:
        value, reference = getattr(state, name), getattr(template, name)
        if value is not None and hasattr(value, "tx") and hasattr(value, "apply_fn"):
            value = value.replace(tx=reference.tx, apply_fn=reference.apply_fn)
        values[name] = value
    return state._replace(**values)


def _program(model, method, batch_size):
    key = _program_key(model, method, batch_size)
    if key in _PROGRAMS:
        return _PROGRAMS[key]
    if method == "td3bc":
        host = importlib.import_module("algorithms.td3_bc")
        def update(rng, state, batch):
            rng, noise = jax.random.split(rng)
            state, metrics = host.td3_bc_update(model._args, model._actor.apply, model._critic.apply,
                                                 state, batch, state.critic.step + 1, noise, 1.)
            return rng, state, metrics
    elif method == "iql":
        host = importlib.import_module("algorithms.iql")
        def update(rng, state, batch):
            rng, dropout = jax.random.split(rng)
            state, metrics = host.iql_update(model._args, model._actor.apply, model._q.apply,
                model._value.apply, state, batch, state.qf.step + 1, dropout)
            return rng, state, metrics
    elif method == "cql":
        host = importlib.import_module("algorithms.cql")
        def update(rng, state, batch):
            rng, update_rng = jax.random.split(rng)
            state, metrics = host.cql_update(model._args, model._actor.apply, model._critic1.apply,
                model._critic2.apply, state, batch, state.critic1.step + 1, update_rng, 1., -4.)
            return rng, state, metrics
    elif method == "rebrac":
        host = importlib.import_module("algorithms.rebrac")
        def update(rng, state, batch):
            state, rng, metrics = host.rebrac_update(model._args, model._actor.apply, model._critic.apply,
                                                      state, batch, state.critic.step, rng)
            return rng, state, metrics
    else:
        host = importlib.import_module("host_integration.unifloral_adapter")
        args = replace(model._args, batch_size=int(batch_size))
        def update(rng, state, batch):
            (rng, state, _), metrics = host._run_step(host.HOST, args, model._actor, model._critic, rng, state, batch)
            return rng, state, metrics

    def execute(rng, state, replay, rows):
        def body(carry, input_row):
            rng, state, bad = carry
            selected = jax.tree_util.tree_map(lambda x: x[input_row], replay)
            rng, state, metrics = update(rng, state, selected)
            leaves = jax.tree_util.tree_leaves((rng, state, metrics))
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
            return (rng, state, bad | ~finite), metrics
        return jax.lax.scan(body, (rng, state, jnp.bool_(False)), rows)

    result = (jax.jit(execute), model._state)
    _PROGRAMS[key] = result
    return result


def _scan(model, method, native_dataset, indices):
    fn, template = _program(model, method, int(indices.shape[1]))
    state = _canonical_state(model._state, template)
    (rng, state, bad), metrics = fn(model._rng, state, native_dataset, jnp.asarray(indices, dtype=jnp.int32))
    if bool(np.asarray(bad)):
        raise FloatingPointError("nonfinite state or metrics encountered within compiled chunk")
    if method == "iql":
        # IQL's eager TrainState steps are Python ints; scan necessarily carries
        # int32 device scalars. Restore the host representation on exit, as the
        # native adapter's checkpoint/restore path does. The supported update
        # budget is safely within int32, and numeric counters remain unchanged.
        state = state._replace(**{name: getattr(state, name).replace(step=int(getattr(state, name).step))
                                  for name in ("actor", "qf", "qf_target", "vf")})
    return rng, state, {name: float(np.asarray(value[-1])) for name, value in metrics.items()}


def train_chunk(dataset, method, seed, batch_size, additional_primary_steps, resume_checkpoint=None):
    """Return a v6-compatible checkpoint and explicit matched-budget receipt."""
    from thrusterchain.datasets import assert_training_allowed
    assert_training_allowed("compiled train_chunk")
    budget = OfflineBudget(updates=additional_primary_steps, batch_size=batch_size)
    if budget.updates < 1:
        raise RunnerError("compiled chunks must contain at least one primary update")
    _dataset(dataset)
    seed = _seed(seed)
    settings = _settings({})
    learner = _factory(method)()
    caps = _capabilities(learner)
    identity = _identity(dataset)
    batch = train_transitions(dataset)
    sampler = _rng(seed, "offline_batches")
    previous = 0
    if resume_checkpoint is None:
        init_seed = int(_rng(seed, "learner_initialization").integers(0, 2**63 - 1))
        learner.initialize(batch, seed=init_seed, learning_settings=settings)
        before = _checkpoint(learner)
        if before.optimizer_steps != 0 or any(before.optimizer_step_counts.values()):
            raise RunnerError("initialize must not execute optimizer steps")
    else:
        if not isinstance(resume_checkpoint, LearnerCheckpoint):
            raise RunnerError("resume must be a LearnerCheckpoint")
        state = resume_checkpoint.metadata.get("runner_state", {})
        required = {"phase": "offline", "dataset_sha256": dataset.digest,
                    "train_split_sha256": identity["split_sha256"]["train"], "seed": seed,
                    "batch_size": budget.batch_size, "learning_settings": settings}
        if any(state.get(key) != value for key, value in required.items()):
            raise RunnerError("offline resume requires matching dataset, seed, batch size and settings")
        if state.get("engine") not in (None, ENGINE) or state.get("engine_sha256") not in (None, _engine_sha256()):
            raise RunnerError("compiled engine identity changed across resume")
        if "sampler_rng_state" not in state or "total_runner_updates" not in state:
            raise RunnerError("missing sampler state or update counter")
        sampler.bit_generator.state = state["sampler_rng_state"]
        previous = state["total_runner_updates"]
        if type(previous) is not int or previous < 0 or previous != resume_checkpoint.optimizer_step_counts.get(caps.primary_optimizer):
            raise RunnerError("resume update counter differs from the primary optimizer")
        before = _restore(learner, resume_checkpoint)
    if previous + budget.updates > 3_000_000:
        raise RunnerError("cumulative offline updates cannot exceed 3000000")
    # PCG64 draws exactly as the standard runner, with replacement, in row order.
    rows = sampler.integers(0, len(batch), size=(budget.updates, budget.batch_size))
    native = _native_dataset(learner, method, batch)
    rng, state, metrics = _scan(learner, method, native, rows)
    learner._rng, learner._state = rng, state
    after = _checkpoint(learner)
    deltas = _counts_delta(before, after)
    if deltas[caps.primary_optimizer] != budget.updates:
        raise RunnerError("compiled primary update count mismatch")
    runner_state = {"phase": "offline", "dataset_sha256": dataset.digest,
        "train_split_sha256": identity["split_sha256"]["train"], "seed": seed,
        "batch_size": budget.batch_size, "learning_settings": settings,
        "sampler_rng_state": sampler.bit_generator.state, "total_runner_updates": previous + budget.updates,
        "engine": ENGINE, "engine_sha256": _engine_sha256()}
    checkpoint = _with_runner_state(after, runner_state)
    receipt = {"schema": "thrusterchain.compiled_chunk.v1", **identity,
        "method": method, "engine": ENGINE, "engine_sha256": _engine_sha256(),
        "adapter_id": caps.adapter_id, "primary_optimizer": caps.primary_optimizer,
        "executed_primary_optimizer_updates": budget.updates,
        "executed_optimizer_steps_total": after.optimizer_steps - before.optimizer_steps,
        "optimizer_step_counts": dict(after.optimizer_step_counts), "optimizer_step_deltas": deltas,
        "batch_size": budget.batch_size, "dataset_draws": budget.updates * budget.batch_size,
        "dataset_draw_semantics": "PCG64 with replacement; one sampled batch per primary step",
        "environment_interactions": 0, "last_update_metrics": metrics,
        "checkpoint_sha256": checkpoint.digest,
        "resume_from_sha256": None if resume_checkpoint is None else resume_checkpoint.digest}
    receipt["receipt_sha256"] = _hash(receipt)
    return checkpoint, receipt
