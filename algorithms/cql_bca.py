"""CQL: full-width Bayesian/conformal modulation of the conservative gap."""

from runtime import published
from dataclasses import dataclass, field
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import algorithms.cql as BASE
from calibration.network import Calibrator, bayesian_bootstrap_weights, soft_coverage
from calibration.posterior import PosteriorConfig
from calibration.reference import (
    initialize_posterior,
    fit_posterior,
    reserve_calibration,
)
from calibration.dose import frozen_level_dose, tree_finite
from calibration.policy_weights import ScaleIWConfig, PolicyLogWeights, fit_weighting


@dataclass(frozen=True)
class Config:
    arm: str = "host"
    iw: ScaleIWConfig = field(default_factory=ScaleIWConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    blend: float = 0.5
    cal_lr: float = 0.001
    cal_beta: float = 20.0
    width_penalty: float = 0.005
    scale_ema: float = 0.99

    def __post_init__(self):
        if (
            self.arm not in ("host", "bca")
            or not isinstance(self.iw, ScaleIWConfig)
            or (not isinstance(self.posterior, PosteriorConfig))
        ):
            raise ValueError("Choose host or bca with typed calibration settings.")
        if self.iw.mode not in (("off", "policy") if self.arm == "bca" else ("off",)):
            raise ValueError(
                "BCA fitting is explicitly unweighted or policy-density; the host has none."
            )
        for name in ("blend", "cal_lr", "cal_beta", "width_penalty", "scale_ema"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(np.float32(value)):
                raise ValueError("Nonfinite calibration setting: " + name)
        if (
            not 0 < self.blend <= 1
            or not 0 <= self.scale_ema < 1
            or self.cal_lr <= 0
            or (self.cal_beta <= 0)
            or (self.width_penalty < 0)
        ):
            raise ValueError("Invalid calibration coefficients.")


class State(NamedTuple):
    native: Any
    calibrator: Any
    residual_scale: Any
    posterior: Any


def select_training_pool(dataset, arm, training_indices=None):
    n = len(dataset.obs)
    if n < 1 or any((len(x) != n for x in dataset)):
        raise ValueError("aligned nonempty native transition arrays required")
    if arm not in ("host", "bca") or training_indices is None:
        raise ValueError("pool/posterior require explicit training indices")
    idx = np.asarray(training_indices)
    if (
        idx.ndim != 1
        or not np.issubdtype(idx.dtype, np.integer)
        or (not 0 < len(idx) < n)
        or (len(np.unique(idx)) != len(idx))
        or np.any(idx < 0)
        or np.any(idx >= n)
    ):
        raise ValueError("training indices must be a unique nonempty strict subset")
    return jax.tree_util.tree_map(lambda x: x[idx], dataset)


def reserve_pool(dataset, target_size, seed, *, max_fraction=0.25, episode_ids=None):
    return reserve_calibration(
        dataset.obs,
        dataset.next_obs,
        dataset.done,
        target_size,
        seed,
        max_fraction=max_fraction,
        episode_ids=episode_ids,
    )


def initialize(args, config, obs_dim, action_dim, max_action=1.0):
    BASE.C.check_reward_transform(
        "corl_cql",
        args.dataset,
        args.reward_transform,
        args.reward_scale,
        args.reward_bias,
    )
    published.check_published_config(
        "corl_cql", args.dataset, args, allow_off_config=args.allow_off_config
    )
    if (
        getattr(args, "cal_iw", False)
        or getattr(args, "acrab_iw_bellman", False)
        or getattr(args, "acrab_cinf", -1.0) >= 0
    ):
        raise ValueError(
            "legacy calibration IW/Bellman/A-Crab are separate interventions"
        )
    if (
        BASE.C.is_antmaze(args.dataset)
        and args.normalize_reward
        and (args.reward_transform == "cql_scale_bias")
        and (args.cql_reward_scale != 1.0 or args.cql_reward_bias != 0.0)
    ):
        raise ValueError("native antmaze affine reward tail would be applied twice")
    if config.arm == "bca" and max_action != 1.0:
        raise ValueError(
            "posterior actor-density convention currently requires max_action=1"
        )
    rng, native, (actor, c1, c2) = BASE.initialize(
        args, obs_dim, action_dim, max_action
    )
    obs, action = jnp.zeros(obs_dim), jnp.zeros(action_dim)
    if config.arm != "bca":
        return (rng, State(native, None, None, None), (actor, c1, c2, None))
    cal = Calibrator(jnp.zeros(obs_dim), jnp.ones(obs_dim), state_dep=True)
    cal_state = TrainState.create(
        apply_fn=cal.apply,
        params=cal.init(jax.random.fold_in(rng, 1128352841), obs[None], action[None]),
        tx=optax.adam(config.cal_lr),
    )
    post = initialize_posterior(cal_state.params, 1, config.posterior.draws)
    return (rng, State(native, cal_state, jnp.asarray(1.0), post), (actor, c1, c2, cal))


def policy_log_weights(actor_apply, params, obs, actions, max_action=1.0):
    if max_action != 1.0:
        raise ValueError("native fitting density currently requires max_action=1")
    if (
        obs.ndim != 2
        or actions.ndim != 2
        or obs.shape[0] != actions.shape[0]
        or (not len(obs))
    ):
        raise ValueError("aligned nonempty observation/action matrices required")
    mean, log_std = actor_apply(
        jax.tree_util.tree_map(jax.lax.stop_gradient, params), obs
    )
    if mean.shape != actions.shape or log_std.shape != actions.shape:
        raise ValueError(
            "native actor must emit one mean/log_std per action coordinate"
        )
    clipped = jnp.clip(actions, -1.0 + 1e-06, 1.0 - 1e-06)
    pre = jnp.arctanh(clipped)
    coordinate_logp = (
        -0.5 * jnp.square((pre - mean) * jnp.exp(-log_std))
        - log_std
        - 0.5 * np.log(2 * np.pi)
    )
    coordinate_logp -= jnp.log1p(-jnp.square(clipped))
    logp = coordinate_logp.sum(axis=-1)
    valid = tree_finite(params) & tree_finite((obs, actions, mean, log_std, logp))
    result = PolicyLogWeights(
        logp, valid, jnp.mean(jnp.any(actions != clipped, axis=-1).astype(jnp.float32))
    )
    return jax.tree_util.tree_map(jax.lax.stop_gradient, result)


def native_target(args, models, native, batch, rng, max_action=1.0):
    actor, c1, c2 = models[:3]
    b, a = batch.action.shape
    shape = (b, args.cql_n_actions, a) if args.cql_max_target_backup else (b, a)
    obs = (
        jnp.broadcast_to(
            batch.next_obs[:, None, :], (b, args.cql_n_actions, batch.obs.shape[-1])
        )
        if args.cql_max_target_backup
        else batch.next_obs
    )
    action, logp = BASE.tanh_gaussian_sample(
        *actor.apply(native.actor.params, obs),
        jax.random.normal(rng, shape),
        max_action
    )
    q = jnp.minimum(
        c1.apply(native.critic1_target.params, batch.next_obs, action),
        c2.apply(native.critic2_target.params, batch.next_obs, action),
    )
    if args.cql_max_target_backup:
        idx = jnp.argmax(q, axis=-1)
        q = jnp.max(q, axis=-1)
        logp = jnp.take_along_axis(logp, idx[:, None], axis=-1).squeeze(-1)
    if args.backup_entropy:
        alpha = (
            jnp.exp(BASE.scalar_value(native.log_alpha)) * args.alpha_multiplier
            if args.use_automatic_entropy_tuning
            else args.alpha_multiplier
        )
        q = q - alpha * logp
    return jax.lax.stop_gradient(batch.reward + (1.0 - batch.done) * args.discount * q)


def fit_scale(args, config, models, state, batch, rng, max_action=1.0):
    target = native_target(
        args,
        models,
        state.native,
        batch,
        jax.random.fold_in(rng, 1179210836),
        max_action,
    )
    q = jax.lax.stop_gradient(
        jnp.minimum(
            models[1].apply(state.native.critic1.params, batch.obs, batch.action),
            models[2].apply(state.native.critic2.params, batch.obs, batch.action),
        )
    )
    prior = jax.lax.stop_gradient(
        bayesian_bootstrap_weights(jax.random.fold_in(rng, 1128352850), len(batch.obs))
    )
    weights, valid, feasible = (prior, jnp.asarray(True), jnp.asarray(True))
    diag = {}
    if config.iw.mode != "off":
        signal = policy_log_weights(
            models[0].apply,
            state.native.actor.params,
            batch.obs,
            batch.action,
            max_action,
        )
        weighted = fit_weighting(
            signal.log_weights, prior, jax.random.fold_in(rng, 1129531735), config.iw
        )
        weights = weighted.product.weights
        valid = signal.inputs_valid & weighted.inputs_valid
        feasible = weighted.fit_feasible
        diag = {
            "iw_ess": weighted.iw.ess_fraction,
            "iw_tau": weighted.iw.tau,
            "iw_product_ess": weighted.product.product_ess_fraction,
            "action_clip_fraction": signal.action_clip_fraction,
        }
    unit = state.residual_scale

    def objective(params):
        eta = models[3].apply(params, batch.obs, batch.action)
        cov = soft_coverage(target / unit, q / unit, eta, config.cal_beta)
        cov_loss = jnp.square(jnp.sum(cov * weights) - (1.0 - config.posterior.alpha))
        width = jnp.sum(jnp.square(eta) * weights)
        return cov_loss + config.width_penalty * width

    loss, grad = jax.value_and_grad(objective)(state.calibrator.params)
    proposed = state.calibrator.apply_gradients(grads=grad)
    proposed_unit = config.scale_ema * unit + (1.0 - config.scale_ema) * jnp.maximum(
        jnp.std(target - q), 1e-06
    )
    valid = (
        valid
        & tree_finite(
            (batch, target, q, prior, unit, loss, grad, proposed, proposed_unit)
        )
        & (unit > 0)
        & (proposed_unit > 0)
    )
    accepted = valid & feasible
    result = jax.lax.cond(
        accepted,
        lambda _: state._replace(calibrator=proposed, residual_scale=proposed_unit),
        lambda _: state,
        operand=None,
    )
    return (
        result,
        dict(
            diag,
            scale_inputs_valid=valid,
            scale_fit_accepted=accepted,
            scale_ess_abstained=valid & ~feasible,
            scale_loss=loss,
        ),
    )


def refresh(
    args, config, models, state, training_reference, heldout, rng, max_action=1.0
):
    if config.arm != "bca" or not len(training_reference.obs) or (not len(heldout.obs)):
        raise ValueError(
            "posterior refresh needs nonempty training and held-out references"
        )
    target = native_target(
        args,
        models,
        state.native,
        heldout,
        jax.random.fold_in(rng, 1213156420),
        max_action,
    )
    q = jnp.minimum(
        models[1].apply(state.native.critic1.params, heldout.obs, heldout.action),
        models[2].apply(state.native.critic2.params, heldout.obs, heldout.action),
    )
    fit_predictions = models[3].apply(
        state.calibrator.params, training_reference.obs, training_reference.action
    )
    cal_predictions = models[3].apply(
        state.calibrator.params, heldout.obs, heldout.action
    )
    post = fit_posterior(
        state.calibrator.params,
        state.residual_scale,
        fit_predictions,
        cal_predictions,
        target - q,
        jax.random.fold_in(rng, 1347375956),
        config.posterior,
        groups=1,
    )
    valid = tree_finite(state.calibrator) & jnp.all(post.radii.inputs_valid)
    result = jax.lax.cond(
        valid, lambda _: state._replace(posterior=post), lambda _: state, operand=None
    )
    return (
        result,
        {
            "posterior_inputs_valid": valid,
            "posterior_supported": jnp.all(post.radii.finite),
        },
    )


def update(args, config, models, state, batch, it, rng, max_action=1.0):
    if config.arm != "bca":
        native, metrics = BASE.cql_update(
            args,
            *(m.apply for m in models[:3]),
            state.native,
            batch,
            it,
            rng,
            max_action,
            -float(batch.action.shape[-1])
        )
        return (state._replace(native=native), metrics)
    fitted, fit_diag = fit_scale(args, config, models, state, batch, rng, max_action)
    frozen_predictions = models[3].apply(
        state.posterior.cal_params, batch.obs, batch.action
    )
    dose = frozen_level_dose(state.posterior, frozen_predictions, "full", config.blend)
    native, metrics = BASE.cql_update(
        args,
        *(m.apply for m in models[:3]),
        state.native,
        batch,
        it,
        rng,
        max_action,
        -float(batch.action.shape[-1]),
        conservative_multiplier=dose.dose
    )
    valid = fit_diag["scale_inputs_valid"] & dose.inputs_valid & tree_finite(native)
    proposed = fitted._replace(native=native)
    result = jax.lax.cond(valid, lambda _: proposed, lambda _: state, operand=None)
    metrics.update(fit_diag)
    metrics.update(
        inputs_valid=valid,
        critic_dose_mean=jnp.mean(dose.dose),
        critic_support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)),
    )
    return (result, metrics)


def make_train_step(args, config, models, dataset, max_action=1.0):
    if not len(dataset.obs):
        raise ValueError("training pool must be nonempty")

    def step(carry, _):
        rng, state, it = carry
        it = it + 1
        rng, kb, ku = jax.random.split(rng, 3)
        idx = jax.random.randint(kb, (args.batch_size,), 0, len(dataset.obs))
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        state, metrics = update(args, config, models, state, batch, it, ku, max_action)
        return ((rng, state, it), metrics)

    return step


def require_valid(metrics):
    for key in ("inputs_valid", "posterior_inputs_valid", "scale_inputs_valid"):
        if key in metrics and (not bool(np.all(np.asarray(metrics[key])))):
            raise FloatingPointError("native CQL posterior invalid transition: " + key)
