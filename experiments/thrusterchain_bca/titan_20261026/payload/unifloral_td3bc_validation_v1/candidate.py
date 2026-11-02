"""Isolated, phase-aware candidate for pinned UniFloral standalone TD3+BC.

No networks, optimizers, normalization, or datasets are constructed here. The
caller supplies the unchanged native definitions and initialized TrainStates.
This is a validation candidate, not a production ThrusterChain adapter.
"""
from dataclasses import dataclass
import math
from typing import Any, Callable, NamedTuple


NATIVE_SOURCE_SHA256 = "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee"


class MicrostepState(NamedTuple):
    """Complete continuation state, including the batch held between critics.

    phase 0: sample the next outer step's batch, then run its first critic.
    phase 1: reuse the held batch, run its second critic, actor, and targets.
    The completed batch remains stored at phase 0 but is replaced next call.
    """

    rng: Any
    agent_state: Any
    batch: Any
    phase: Any
    critic_calls: Any
    actor_calls: Any
    actor_target_updates: Any
    critic_target_updates: Any
    outer_steps: Any
    q_loss_first: Any
    reference: Any


@dataclass(frozen=True)
class MicrostepFunctions:
    initial_state: Callable
    step: Callable
    actor_loss: Callable


def make_microsteps(args, actor_apply_fn, q_apply_fn, dataset,
                    width_fn=None, kappa_penalty=0.0):
    """Return pure functions implementing one actual critic update per call.

    ``dataset`` has the native Transition pytree structure. ``width_fn`` is a
    deterministic, batched callable (reference, observations, live_actions)
    returning one scalar width per row. It must not draw RNG or mutate state.
    Its reference pytree is detached at use, while actions remain live.

    The factory's Python-static kappa=0 path does not call width_fn at all and
    retains the native actor loss expression. A positive kappa adds only
    c*kappa*mean(width), with c computed from unpenalized Q1 and detached.
    No extra Q normalization, denominator guard, or precision setting is added.
    """
    import jax
    import jax.numpy as jnp
    import optax

    if args.num_critic_updates_per_step != 2:
        raise ValueError("This candidate requires the native two-critic outer step")
    if args.batch_size <= 0 or len(dataset.obs) <= 0:
        raise ValueError("A nonempty dataset and positive native batch size are required")
    kappa_penalty = float(kappa_penalty)
    if not math.isfinite(kappa_penalty) or kappa_penalty < 0:
        raise ValueError("kappa_penalty must be a finite nonnegative scalar")
    use_penalty = kappa_penalty != 0.0
    if use_penalty and not callable(width_fn):
        raise ValueError("Positive kappa_penalty requires a deterministic width_fn")

    def actor_loss(params, agent_state, batch, reference):
        # Keep this native transition calculation and its reductions unchanged.
        # The extra action output is needed only by the positive-penalty path.
        if not use_penalty:
            def _transition_loss(transition):
                pi = actor_apply_fn(params, transition.obs)
                pi_action = pi.sample(seed=None)
                q = q_apply_fn(agent_state.dual_q.params, transition.obs, pi_action)
                bc_loss = jnp.square(pi_action - transition.action).mean()
                return q[0], bc_loss

            q, bc_loss = jax.vmap(_transition_loss)(batch)
            lambda_ = args.td3_alpha / (jnp.abs(q).mean() + 1e-7)
            lambda_ = jax.lax.stop_gradient(lambda_)
            actor_loss_value = -lambda_ * q.mean() + bc_loss.mean()
            zero = jnp.zeros_like(actor_loss_value)
            return actor_loss_value.mean(), {
                "q_mean": q.mean(),
                "q_scale_coefficient": lambda_.mean(),
                "bc_loss": bc_loss.mean(),
                "width_mean": zero,
                "penalty_mean": zero,
                "actor_penalty_term": zero,
            }

        def _transition_loss(transition):
            pi = actor_apply_fn(params, transition.obs)
            pi_action = pi.sample(seed=None)
            q = q_apply_fn(agent_state.dual_q.params, transition.obs, pi_action)
            bc_loss = jnp.square(pi_action - transition.action).mean()
            return q[0], bc_loss, pi_action

        q, bc_loss, live_actions = jax.vmap(_transition_loss)(batch)
        lambda_ = args.td3_alpha / (jnp.abs(q).mean() + 1e-7)
        lambda_ = jax.lax.stop_gradient(lambda_)
        native_loss = -lambda_ * q.mean() + bc_loss.mean()
        frozen_reference = jax.tree_util.tree_map(jax.lax.stop_gradient, reference)
        widths = width_fn(frozen_reference, batch.obs, live_actions)
        if widths.shape != (batch.obs.shape[0],):
            raise ValueError("width_fn must return exactly one scalar width per row")
        if widths.dtype != q.dtype:
            raise ValueError("width_fn must preserve the native Q floating dtype")
        width_mean = widths.mean()
        penalty_mean = kappa_penalty * width_mean
        actor_penalty_term = lambda_ * penalty_mean
        actor_loss_value = native_loss + actor_penalty_term
        return actor_loss_value.mean(), {
            "q_mean": q.mean(),
            "q_scale_coefficient": lambda_.mean(),
            "bc_loss": bc_loss.mean(),
            "width_mean": width_mean,
            "penalty_mean": penalty_mean,
            "actor_penalty_term": actor_penalty_term,
        }

    def _update_critic(rng, agent_state, batch):
        # Same draws and operations as one iteration of the native critic scan.
        def _compute_target(rng, transition):
            next_obs = transition.next_obs
            next_pi = actor_apply_fn(agent_state.actor_target.params, next_obs)
            rng, rng_action, rng_noise = jax.random.split(rng, 3)
            action = next_pi.sample(seed=rng_action)
            noise = jax.random.normal(rng_noise, shape=action.shape)
            noise *= args.policy_noise
            noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
            action = jnp.clip(action + noise, -1, 1)
            target_q = q_apply_fn(agent_state.dual_q_target.params, next_obs, action)
            next_q_value = (1.0 - transition.done) * jnp.min(target_q)
            return transition.reward + args.gamma * next_q_value

        rng, rng_targets = jax.random.split(rng)
        rng_targets = jax.random.split(rng_targets, args.batch_size)
        targets = jax.vmap(_compute_target)(rng_targets, batch)

        @jax.value_and_grad
        def _q_loss_fn(params):
            q_pred = q_apply_fn(params, batch.obs, batch.action)
            return jnp.square(q_pred - jnp.expand_dims(targets, axis=-1)).mean()

        q_loss, q_grad = _q_loss_fn(agent_state.dual_q.params)
        updated_q_state = agent_state.dual_q.apply_gradients(grads=q_grad)
        agent_state = agent_state._replace(dual_q=updated_q_state)
        return rng, agent_state, q_loss

    def _update_target(state, target_state):
        new_target_params = optax.incremental_update(
            state.params, target_state.params, args.polyak_step_size)
        return target_state.replace(step=target_state.step + 1, params=new_target_params)

    def initial_state(rng, agent_state, reference=None):
        """Wrap a native state at an outer-step boundary without RNG draws.

        Fresh native state is the intended input. A continuation checkpoint must
        restore the full MicrostepState, not call this initializer with only
        network weights. The caller must establish that supplied native steps
        describe a complete outer-step boundary if wrapping a nonfresh state.
        """
        batch = jax.tree_util.tree_map(
            lambda x: jnp.zeros((args.batch_size,) + x.shape[1:], dtype=x.dtype),
            dataset,
        )
        loss_dtype = jnp.result_type(dataset.obs.dtype, dataset.action.dtype,
                                     dataset.reward.dtype)
        return MicrostepState(
            rng=rng,
            agent_state=agent_state,
            batch=batch,
            phase=jnp.asarray(0, dtype=jnp.int32),
            critic_calls=jnp.asarray(agent_state.dual_q.step, dtype=jnp.int32),
            actor_calls=jnp.asarray(agent_state.actor.step, dtype=jnp.int32),
            actor_target_updates=jnp.asarray(agent_state.actor_target.step, dtype=jnp.int32),
            critic_target_updates=jnp.asarray(agent_state.dual_q_target.step, dtype=jnp.int32),
            outer_steps=jnp.asarray(agent_state.actor.step, dtype=jnp.int32),
            q_loss_first=jnp.zeros((), dtype=loss_dtype),
            reference=reference,
        )

    def step(state, unused=None):
        """Pure, lax.scan-compatible microstep; exactly one critic optimizer call."""
        del unused

        def _sample(rng_and_batch):
            rng, _ = rng_and_batch
            rng, rng_batch = jax.random.split(rng)
            batch_indices = jax.random.randint(
                rng_batch, (args.batch_size,), 0, len(dataset.obs))
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)
            return rng, batch

        rng, batch = jax.lax.cond(
            state.phase == 0, _sample, lambda x: x, (state.rng, state.batch))
        rng, agent_state, q_loss = _update_critic(rng, state.agent_state, batch)

        def _finish_outer(agent_state):
            loss_fn = jax.value_and_grad(actor_loss, has_aux=True)
            (value, aux), actor_grad = loss_fn(
                agent_state.actor.params, agent_state, batch, state.reference)
            agent_state = agent_state._replace(
                actor=agent_state.actor.apply_gradients(grads=actor_grad))
            agent_state = agent_state._replace(
                actor_target=_update_target(agent_state.actor, agent_state.actor_target),
                dual_q_target=_update_target(agent_state.dual_q, agent_state.dual_q_target),
            )
            return agent_state, {"actor_loss": value, **aux}

        def _hold_actor(agent_state):
            zero = jnp.zeros_like(q_loss)
            return agent_state, {name: zero for name in (
                "actor_loss", "q_mean", "q_scale_coefficient", "bc_loss",
                "width_mean", "penalty_mean", "actor_penalty_term")}

        completed = state.phase == 1
        agent_state, metrics = jax.lax.cond(
            completed, _finish_outer, _hold_actor, agent_state)
        metrics = {
            **metrics,
            "q_loss": jnp.where(
                completed, jnp.stack([state.q_loss_first, q_loss]).mean(), q_loss),
            "critic_loss_this_call": q_loss,
            "outer_completed": completed,
        }
        next_state = MicrostepState(
            rng=rng,
            agent_state=agent_state,
            batch=batch,
            phase=jnp.where(completed, jnp.int32(0), jnp.int32(1)),
            critic_calls=state.critic_calls + 1,
            actor_calls=state.actor_calls + completed.astype(jnp.int32),
            actor_target_updates=state.actor_target_updates + completed.astype(jnp.int32),
            critic_target_updates=state.critic_target_updates + completed.astype(jnp.int32),
            outer_steps=state.outer_steps + completed.astype(jnp.int32),
            q_loss_first=jnp.where(completed, state.q_loss_first, q_loss),
            reference=state.reference,
        )
        return next_state, metrics

    return MicrostepFunctions(initial_state=initial_state, step=step, actor_loss=actor_loss)
