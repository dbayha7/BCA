"""TD3+BC: frozen WBCP-width modulation of the actor BC term."""

from runtime import published
from dataclasses import dataclass
from numbers import Real
from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import algorithms.td3_bc as BASE
from calibration.network import Calibrator, bayesian_bootstrap_weights, soft_coverage
from calibration.reference import (
    WBCPConfig,
    initial_reference,
    freeze_reference,
    certifiable,
    reference_valid,
    reserve_calibration,
)
from calibration.dose import frozen_level_dose, tree_finite


def _real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite real scalar")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        v = np.float32(value)
    if not np.isfinite(v) or (positive and v <= 0):
        raise ValueError(name + " must be finite and representable in float32")


@dataclass(frozen=True)
class Config:
    arm: str
    posterior: WBCPConfig | None = None
    blend: float | None = None
    cal_lr: float = 0.001
    cal_beta: float = 20.0
    width_penalty: float = 0.005
    scale_ema: float = 0.99

    def __post_init__(self):
        if self.arm not in ("host", "bca"):
            raise ValueError("Choose the host or bca arm.")
        if self.arm == "bca":
            if type(self.posterior) is not WBCPConfig:
                raise ValueError("BCA requires a typed WBCP threshold configuration.")
            _real(self.blend, "blend", positive=True)
            if self.blend > 1:
                raise ValueError("blend exceeds one")
        elif self.posterior is not None or self.blend is not None:
            raise ValueError("The host cannot carry BCA settings.")
        for name in ("cal_lr", "cal_beta", "width_penalty", "scale_ema"):
            _real(getattr(self, name), name, positive=name in ("cal_lr", "cal_beta"))
        if self.width_penalty < 0 or not 0 <= self.scale_ema < 1:
            raise ValueError("Invalid scale fitting coefficients.")

    @property
    def is_posterior(self):
        return self.arm == "bca"


class State(NamedTuple):
    native: Any
    calibrator: Any
    residual_scale: Any
    posterior: Any  # calibration.reference.FrozenReference on the bca arm


def _ids(indices, n, name):
    idx = np.asarray(indices)
    if (
        idx.ndim != 1
        or len(idx) != n
        or (not np.issubdtype(idx.dtype, np.integer))
        or np.any(idx < 0)
        or (len(np.unique(idx)) != n)
    ):
        raise ValueError(name + " must be unique aligned nonnegative integer row IDs")
    return idx


def select_training_pool(dataset, arm, training_indices=None):
    n = len(dataset.obs)
    if n < 1 or any((len(x) != n for x in dataset)):
        raise ValueError("aligned nonempty native transition arrays required")
    if arm not in ("host", "bca") or training_indices is None:
        raise ValueError(
            "reserved controls/posterior require explicit training indices"
        )
    idx = np.asarray(training_indices)
    if idx.ndim != 1:
        raise ValueError("training indices must be a vector")
    _ids(idx, len(idx), "training indices")
    if not 0 < len(idx) < n or np.any(idx >= n):
        raise ValueError("training pool must be a nonempty strict subset")
    return jax.tree_util.tree_map(lambda x: x[idx], dataset)


def reserve_pool(dataset, target_size, seed, rows_per_episode, *, max_fraction=0.25, episode_ids=None):
    # (training, withheld, calibration, metadata): withheld rows never train; calibration is the WBCP bank.
    return reserve_calibration(
        dataset.obs, dataset.next_obs, dataset.done,
        target_size,
        seed,
        rows_per_episode,
        max_fraction=max_fraction,
        episode_ids=episode_ids,
    )


def initialize(args, config, obs_dim, action_dim, max_action=1.0):
    BASE.C.check_reward_transform(
        "corl_td3_bc",
        args.dataset,
        args.reward_transform,
        args.reward_scale,
        args.reward_bias,
    )
    published.check_published_config(
        "corl_td3_bc", args.dataset, args, allow_off_config=args.allow_off_config
    )
    _real(max_action, "max_action", positive=True)
    rng, native, (actor, critic) = BASE.initialize(
        args, obs_dim, action_dim, max_action
    )
    obs, action = jnp.zeros(obs_dim), jnp.zeros(action_dim)
    if config.arm != "bca":
        return (rng, State(native, None, None, None), (actor, critic, None))
    cal = Calibrator(jnp.zeros(obs_dim), jnp.ones(obs_dim), state_dep=True)
    cal_state = TrainState.create(
        apply_fn=cal.apply,
        params=cal.init(jax.random.fold_in(rng, 1128352841), obs[None], action[None]),
        tx=optax.adam(config.cal_lr),
    )
    posterior = initial_reference(cal_state.params)
    return (
        rng,
        State(native, cal_state, jnp.asarray(1.0), posterior),
        (actor, critic, cal),
    )


def native_target(args, models, native, batch, rng_noise, max_action=1.0):
    noise = jax.random.normal(rng_noise, batch.action.shape) * args.policy_noise
    noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    action = models[0].apply(native.actor_target.params, batch.next_obs)
    action = jnp.clip(action + noise, -max_action, max_action)
    q = (
        models[1]
        .apply(native.critic_target.params, batch.next_obs, action)
        .min(axis=-1)
    )
    return jax.lax.stop_gradient(batch.reward + (1.0 - batch.done) * args.discount * q)


def fit_scale(args, config, models, state, batch, rng, max_action=1.0):
    if config.arm != "bca":
        raise ValueError("only posterior arms fit a scale")
    target = native_target(
        args,
        models,
        state.native,
        batch,
        jax.random.fold_in(rng, 1179210836),
        max_action,
    )
    q = jax.lax.stop_gradient(
        models[1]
        .apply(state.native.critic.params, batch.obs, batch.action)
        .min(axis=-1)
    )
    prior = jax.lax.stop_gradient(
        bayesian_bootstrap_weights(jax.random.fold_in(rng, 1128352850), len(batch.obs))
    )
    unit = state.residual_scale

    def objective(params):
        eta = models[2].apply(params, batch.obs, batch.action)
        if eta.shape != (len(batch.obs),):
            raise ValueError("scale predictor must emit one positive value per row")
        coverage = soft_coverage(target / unit, q / unit, eta, config.cal_beta)
        width = jnp.sum(prior * jnp.square(eta))
        return (
            jnp.square(jnp.sum(prior * coverage) - (1.0 - config.posterior.alpha))
            + config.width_penalty * width,
            eta,
        )

    (loss, eta), grad = jax.value_and_grad(objective, has_aux=True)(
        state.calibrator.params
    )
    proposed = state.calibrator.apply_gradients(grads=grad)
    unit_new = config.scale_ema * unit + (1.0 - config.scale_ema) * jnp.maximum(
        jnp.std(target - q), 1e-06
    )
    valid = (
        tree_finite(
            (
                batch,
                state.native,
                state.calibrator,
                target,
                q,
                prior,
                unit,
                loss,
                grad,
                proposed,
                unit_new,
            )
        )
        & jnp.all(jnp.isfinite(eta) & (eta > 0))
        & jnp.all(prior >= 0)
        & jnp.any(prior > 0)
        & jnp.isclose(prior.sum(), 1.0, rtol=1e-05, atol=1e-06)
        & (unit > 0)
        & (unit_new > 0)
    )
    result = jax.lax.cond(
        valid,
        lambda _: state._replace(calibrator=proposed, residual_scale=unit_new),
        lambda _: state,
        None,
    )
    return (
        result,
        dict(scale_inputs_valid=valid, scale_fit_accepted=valid, scale_loss=loss),
    )


def refresh(args, config, models, state, heldout, rng, max_action=1.0, *, heldout_ids):
    """Freeze the current scale and select the uniform-weight WBCP threshold.

    Runs eagerly (freeze_reference is NumPy): scores are |target - q| / frozen scale
    on the held-out rows. Randomness only folds the refresh key. An invalid refresh
    keeps the previous state and reports posterior_inputs_valid=False.
    """
    if config.arm != "bca" or not len(heldout.obs):
        raise ValueError("posterior refresh requires a nonempty held-out reference")
    _ids(heldout_ids, len(heldout.obs), "held-out IDs")
    target = native_target(
        args,
        models,
        state.native,
        heldout,
        jax.random.fold_in(rng, 1213156420),
        max_action,
    )
    q = (
        models[1]
        .apply(state.native.critic.params, heldout.obs, heldout.action)
        .min(axis=-1)
    )
    cal = models[2].apply(state.calibrator.params, heldout.obs, heldout.action)
    stored = (
        tree_finite((state.native, state.calibrator, heldout, target, q))
        & reference_valid(state.posterior)
        & jnp.all(jnp.isfinite(cal) & (cal > 0))
    )
    reference, frozen, diagnostics = freeze_reference(
        state.calibrator.params,
        state.residual_scale,
        cal,
        target - q,
        jax.random.fold_in(rng, 1347375956),
        config.posterior,
    )
    valid = bool(stored) and frozen
    result = state._replace(posterior=reference) if valid else state
    return (
        result,
        dict(
            posterior_inputs_valid=valid,
            posterior_certified=valid and bool(diagnostics["certified"]),
        ),
        diagnostics,
    )


def bc_readout(models, state, batch, blend):
    predictions = models[2].apply(state.posterior.cal_params, batch.obs, batch.action)
    dose = frozen_level_dose(state.posterior, predictions, blend)
    return dose._replace(
        inputs_valid=dose.inputs_valid & reference_valid(state.posterior)
    )


def consumed_bc_multiplier(config, dose, batch, it, rng):
    return dose


def update(args, config, models, state, batch, it, rng, max_action=1.0):
    fitted, diag, multiplier = (state, {}, None)
    if config.arm == "bca":
        fitted, diag = fit_scale(args, config, models, state, batch, rng, max_action)
        dose = bc_readout(models, state, batch, config.blend)
        multiplier = consumed_bc_multiplier(config, dose.dose, batch, it, rng)
    native, metrics = BASE.td3_bc_update(
        args,
        models[0].apply,
        models[1].apply,
        state.native,
        batch,
        it,
        rng,
        max_action,
        bc_multiplier=multiplier,
    )
    valid = tree_finite((batch, state.native, native, metrics))
    if config.arm == "bca":
        valid = valid & diag["scale_inputs_valid"] & dose.inputs_valid
        metrics.update(
            bc_multiplier_mean=jnp.mean(multiplier),
            bc_support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)),
            posterior_ready=state.posterior.ready,
        )
    result = jax.lax.cond(
        valid, lambda _: fitted._replace(native=native), lambda _: state, None
    )
    metrics.update(diag, inputs_valid=valid)
    return (result, metrics)


def make_train_step(args, config, models, dataset, max_action=1.0):
    if not len(dataset.obs):
        raise ValueError("training pool must be nonempty")

    def step(carry, _):
        rng, state, it = carry
        kr, kb, kn = jax.random.split(rng, 3)
        idx = jax.random.randint(kb, (args.batch_size,), 0, len(dataset.obs))
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        result, metrics = update(
            args, config, models, state, batch, it + 1, kn, max_action
        )
        proposed = (kr, result, it + 1)
        accepted = jax.lax.cond(
            metrics["inputs_valid"], lambda _: proposed, lambda _: carry, None
        )
        return (accepted, metrics)

    return step


def require_valid(metrics):
    for name in ("inputs_valid", "scale_inputs_valid", "posterior_inputs_valid"):
        if name in metrics and (not np.all(np.asarray(metrics[name]))):
            raise FloatingPointError("native TD3 posterior invalid transition: " + name)
