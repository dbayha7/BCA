"""Experimental eager-critic / cached-delayed-actor TD3+BC execution.

This candidate is separate from the existing compiled scan and study runner.
It is not cleared for GPU production use. The update is a structural
transcription of the repository-native TD3+BC objective and update order;
the delayed actor branch functions are created once per learner, and the
minibatch is supplied as a cond operand instead of a freshly closed-over value.
"""
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

from host_integration.td3bc_adapter import TD3BCAdapter, _finite_tree
from runtime.networks import Transition
from thrusterchain.datasets import assert_training_allowed
from thrusterchain.training_protocol import LearnerCheckpoint, OfflineBudget
from thrusterchain.training_runner import (
    RunnerError, _capabilities, _checkpoint, _counts_delta, _dataset, _hash,
    _identity, _restore, _rng, _seed, _settings, _with_runner_state, train_transitions,
)

ENGINE = "candidate-td3bc-cached-branch-v1"


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def dependency_hashes():
    names = ("algorithms.td3_bc", "host_integration.td3bc_adapter", "runtime.networks",
             "runtime.rewards", "thrusterchain.training_runner", "thrusterchain.training_protocol",
             "thrusterchain.datasets")
    return {name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
            for name in names}


def runtime_signature():
    """Bind numerical execution settings, package versions and device identity.

    Keep unset environment variables distinct from explicitly empty values.
    This remains CPU-compatible; a GPU backend requires a readable GPU/driver
    inventory, so an unknown GPU identity cannot silently pass a resume check.
    """
    names = ("XLA_FLAGS", "JAX_PLATFORMS", "JAX_PLATFORM_NAME",
             "JAX_DEFAULT_MATMUL_PRECISION", "JAX_DISABLE_JIT", "JAX_ENABLE_X64",
             "JAX_DEFAULT_PRNG_IMPL", "JAX_THREEFRY_PARTITIONABLE",
             "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE",
             "CUBLAS_WORKSPACE_CONFIG", "XLA_PYTHON_CLIENT_PREALLOCATE",
             "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_COMPILATION_CACHE_DIR")
    packages = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "scipy",
                "flax", "optax", "chex", "distrax", "tensorflow-probability", "gymnasium",
                "ml-dtypes", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvcc-cu12",
                "nvidia-cudnn-cu12", "nvidia-cufft-cu12", "nvidia-cusolver-cu12", "nvidia-cusparse-cu12",
                "nvidia-nvjitlink-cu12")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    backend = jax.default_backend()
    gpu_inventory = None
    if backend == "gpu":
        gpu_inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            check=True, capture_output=True, text=True).stdout.strip().splitlines()
        if not gpu_inventory:
            raise RunnerError("GPU backend requires a recorded GPU/driver inventory")
    return {"backend": backend, "devices": [str(device) for device in jax.devices()],
            "device_kinds": [device.device_kind for device in jax.devices()],
            "gpu_inventory": gpu_inventory, "python": sys.version, "versions": versions,
            "execution_settings": {"environment": {name: os.environ.get(name) for name in names},
                "jax_config": {"disable_jit": bool(jax.config.jax_disable_jit),
                    "enable_x64": bool(jax.config.jax_enable_x64),
                    "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
                    "default_prng_impl": str(jax.config.jax_default_prng_impl),
                    "threefry_partitionable": bool(jax.config.jax_threefry_partitionable)}}}


def _native_dataset(model, batch):
    obs = np.asarray(batch.observations, np.float32)
    nxt = np.asarray(batch.next_observations, np.float32)
    action = np.asarray(batch.actions, np.float32)
    reward = np.asarray(batch.rewards, np.float32)
    if (len(batch) < 1 or obs.ndim != 2 or obs.shape[-1] != model._dim
            or nxt.shape != obs.shape or action.shape != (len(batch), 4)
            or reward.shape != (len(batch),) or np.any(np.abs(action) > 1)
            or any(not np.all(np.isfinite(x)) for x in (obs, nxt, action, reward))
            or np.any(np.asarray(batch.origin) != 0)
            or not np.array_equal(batch.bootstrap_mask, ~batch.terminated)):
        raise RunnerError("invalid offline training data for candidate learner")
    return Transition(jnp.asarray((obs - model._mean) / model._scale), jnp.asarray(action),
                      jnp.asarray(reward), jnp.asarray((nxt - model._mean) / model._scale),
                      jnp.asarray(batch.terminated, jnp.float32))


def make_update(model):
    """Build stable branch closures; preserve native loss and update order."""
    args, actor_apply_fn, critic_apply_fn = model._args, model._actor.apply, model._critic.apply

    def _do_actor_update(operand):
        state, batch = operand

        def _actor_loss_fn(params, critic_params):
            pi = actor_apply_fn(params, batch.obs)
            q = critic_apply_fn(critic_params, batch.obs, pi)[..., 0]
            lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
            bc = jnp.square(pi - batch.action).mean()
            return (-lmbda * q.mean() + bc, (q.mean(), lmbda, bc))

        (a_loss, aux), a_grad = jax.value_and_grad(_actor_loss_fn, has_aux=True)(
            state.actor.params, state.critic.params)
        state = state._replace(actor=state.actor.apply_gradients(grads=a_grad))

        def _soft(src, tgt):
            return tgt.replace(step=tgt.step + 1,
                               params=optax.incremental_update(src.params, tgt.params, args.tau))

        state = state._replace(actor_target=_soft(state.actor, state.actor_target),
                               critic_target=_soft(state.critic, state.critic_target))
        return state, (a_loss, *aux)

    def _skip_actor_update(operand):
        state, _ = operand
        zero = jnp.float32(0.0)
        return state, (zero, zero, zero, zero)

    def update(agent_state, batch, it, rng_noise):
        noise = jax.random.normal(rng_noise, batch.action.shape) * args.policy_noise
        noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
        next_action = actor_apply_fn(agent_state.actor_target.params, batch.next_obs)
        next_action = jnp.clip(next_action + noise, -1., 1.)
        target_q = critic_apply_fn(agent_state.critic_target.params, batch.next_obs, next_action)
        target_q = jnp.min(target_q, axis=-1)
        target = batch.reward + (1.0 - batch.done) * args.discount * target_q
        target = jax.lax.stop_gradient(target)

        def _critic_loss_fn(params):
            q = critic_apply_fn(params, batch.obs, batch.action)
            return jnp.square(q - target[:, None]).mean(axis=0).sum()

        critic_loss, critic_grad = jax.value_and_grad(_critic_loss_fn)(agent_state.critic.params)
        agent_state = agent_state._replace(critic=agent_state.critic.apply_gradients(grads=critic_grad))
        agent_state, (actor_loss, q_mean, lmbda, bc_loss) = jax.lax.cond(
            it % args.policy_freq == 0, _do_actor_update, _skip_actor_update, (agent_state, batch))
        return agent_state, {"critic_loss": critic_loss, "actor_loss": actor_loss,
                             "q_mean": q_mean, "lambda": lmbda, "bc_loss": bc_loss}

    return update


def train_chunk(dataset, seed, batch_size, additional_primary_steps, resume_checkpoint=None):
    """Return a checkpoint and candidate receipt; steps are additional updates."""
    assert_training_allowed("candidate TD3+BC train_chunk")
    budget = OfflineBudget(updates=additional_primary_steps, batch_size=batch_size)
    if budget.updates < 1:
        raise RunnerError("candidate chunks must contain at least one update")
    _dataset(dataset)
    seed = _seed(seed)
    settings = _settings({})
    learner = TD3BCAdapter()
    caps = _capabilities(learner)
    identity = _identity(dataset)
    batch = train_transitions(dataset)
    sampler = _rng(seed, "offline_batches")
    previous = 0
    engine_digest = source_sha256()
    dependencies = dependency_hashes()
    runtime = runtime_signature()
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
                    "batch_size": budget.batch_size, "learning_settings": settings,
                    "engine": ENGINE, "engine_sha256": engine_digest,
                    "source_dependencies_sha256": dependencies, "runtime_signature": runtime}
        if any(state.get(k) != v for k, v in required.items()):
            raise RunnerError("candidate resume identity mismatch")
        if "sampler_rng_state" not in state or "total_runner_updates" not in state:
            raise RunnerError("missing sampler state or update counter")
        sampler.bit_generator.state = state["sampler_rng_state"]
        previous = state["total_runner_updates"]
        if type(previous) is not int or previous < 0 or previous != resume_checkpoint.optimizer_step_counts.get(caps.primary_optimizer):
            raise RunnerError("resume update counter differs from primary optimizer")
        before = _restore(learner, resume_checkpoint)
    if previous + budget.updates > 3_000_000:
        raise RunnerError("cumulative offline updates cannot exceed 3000000")
    native = _native_dataset(learner, batch)
    update = make_update(learner)
    metrics = {}
    for _ in range(budget.updates):
        indices = sampler.integers(0, len(batch), size=budget.batch_size)
        selected = jax.tree_util.tree_map(lambda x: x[indices], native)
        next_rng, noise = jax.random.split(learner._rng)
        next_step = int(learner._state.critic.step) + 1
        state, values = update(learner._state, selected, next_step, noise)
        if not _finite_tree(state) or not _finite_tree(values):
            raise FloatingPointError("nonfinite update; previous learner state and RNG retained")
        learner._state, learner._rng = state, next_rng
        metrics = {key: float(value) for key, value in values.items()}
    if (source_sha256() != engine_digest or dependency_hashes() != dependencies
            or runtime_signature() != runtime):
        raise RunnerError("candidate source or runtime changed during chunk")
    after = _checkpoint(learner)
    deltas = _counts_delta(before, after)
    if deltas[caps.primary_optimizer] != budget.updates:
        raise RunnerError("candidate primary update count mismatch")
    runner_state = {"phase": "offline", "dataset_sha256": dataset.digest,
        "train_split_sha256": identity["split_sha256"]["train"], "seed": seed,
        "batch_size": budget.batch_size, "learning_settings": settings,
        "sampler_rng_state": sampler.bit_generator.state, "total_runner_updates": previous + budget.updates,
        "engine": ENGINE, "engine_sha256": engine_digest,
        "source_dependencies_sha256": dependencies, "runtime_signature": runtime}
    checkpoint = _with_runner_state(after, runner_state)
    receipt = {"schema": "thrusterchain.candidate_td3_chunk.v1", **identity,
        "engine": ENGINE, "engine_sha256": engine_digest, "adapter_id": caps.adapter_id,
        "source_dependencies_sha256": dependencies, "runtime_signature": runtime,
        "primary_optimizer": caps.primary_optimizer, "executed_primary_optimizer_updates": budget.updates,
        "optimizer_step_counts": dict(after.optimizer_step_counts), "optimizer_step_deltas": deltas,
        "batch_size": budget.batch_size, "dataset_draws": budget.updates * budget.batch_size,
        "dataset_draw_semantics": "PCG64 with replacement; one sampled batch per critic update",
        "environment_interactions": 0, "last_update_metrics": metrics,
        "checkpoint_sha256": checkpoint.digest,
        "resume_from_sha256": None if resume_checkpoint is None else resume_checkpoint.digest}
    receipt["receipt_sha256"] = _hash(receipt)
    return checkpoint, receipt
