"""ReBRAC: frozen WBCP-width actor BC with unchanged critic BC."""

from runtime import published
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import algorithms.rebrac as BASE
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

KEY_CAL_INIT = 1380141390
KEY_FIT_NOISE = 1380338004
KEY_BOOTSTRAP = 1380077391
KEY_REFRESH_NOISE = 1380470604
KEY_POSTERIOR = 1380994899  # WBCP posterior draws at a refresh


def _real(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
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
    cal_obs_mean: Any
    cal_obs_std: Any


def validate_native_args(args):
    if type(args) is not BASE.Args or args.algorithm != "corl_rebrac":
        raise ValueError("native ReBRAC Args required")
    for f in fields(BASE.Args):
        value, default = (getattr(args, f.name), f.default)
        if isinstance(default, bool) and type(value) is not bool:
            raise ValueError(f.name + " must be boolean")
        if isinstance(default, str) and (not isinstance(value, str)):
            raise ValueError(f.name + " must be a string")
        if isinstance(default, int) and (not isinstance(default, bool)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f.name + " must be an integer")
            if value < (0 if f.name == "seed" else 1):
                raise ValueError(f.name + " is outside its native domain")
        if isinstance(default, float):
            _real(value, f.name)
    if args.seed >= 2**32 or not 0 <= args.gamma <= 1 or (not 0 <= args.tau <= 1):
        raise ValueError("invalid native seed/gamma/tau")
    for name in ("actor_learning_rate", "critic_learning_rate"):
        _real(getattr(args, name), name, positive=True)
    if any(
        (
            getattr(args, n) < 0
            for n in ("actor_bc_coef", "critic_bc_coef", "policy_noise", "noise_clip")
        )
    ):
        raise ValueError("native BC and noise coefficients must be nonnegative")
    BASE.C.check_reward_transform(
        "corl_rebrac",
        args.dataset,
        args.reward_transform,
        args.reward_scale,
        args.reward_bias,
    )
    published.check_published_config(
        "corl_rebrac", args.dataset, args, allow_off_config=args.allow_off_config
    )
    if args.normalize_reward != BASE.C.is_antmaze(args.dataset):
        raise ValueError("native reward normalization must agree with antmaze identity")
    if 1000 % args.policy_freq:
        raise ValueError("policy_freq must divide native 1000-update epoch")


def transition_shape(batch):
    if type(batch) is not BASE.C.TransitionNA:
        raise ValueError("native named TransitionNA with real next_action required")
    n = batch.obs.shape[0] if batch.obs.ndim == 2 else 0
    if (
        not n
        or batch.obs.shape[1] < 1
        or batch.next_obs.shape != batch.obs.shape
        or (batch.action.ndim != 2)
        or (batch.action.shape[0] != n)
        or (batch.action.shape[1] < 1)
        or (batch.next_action.shape != batch.action.shape)
        or (batch.reward.shape != (n,))
        or (batch.done.shape != (n,))
    ):
        raise ValueError("nonempty aligned native transition matrices/vectors required")
    for name in batch._fields:
        dtype = getattr(batch, name).dtype
        if not (
            jnp.issubdtype(dtype, jnp.floating)
            or (name == "done" and dtype == jnp.bool_)
        ):
            raise TypeError(
                "native transition fields must be floating; done may be boolean"
            )


def transition_valid(batch):
    transition_shape(batch)
    return (
        tree_finite(batch)
        & jnp.all(jnp.abs(batch.action) <= 1)
        & jnp.all(jnp.abs(batch.next_action) <= 1)
        & jnp.all((batch.done == 0) | (batch.done == 1))
    )


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
    transition_shape(dataset)
    n = len(dataset.obs)
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
    transition_shape(dataset)
    return reserve_calibration(
        dataset.obs, dataset.next_obs, dataset.done,
        target_size,
        seed,
        rows_per_episode,
        max_fraction=max_fraction,
        episode_ids=episode_ids,
    )


def initialize(args, config, training):
    validate_native_args(args)
    if type(config) is not Config:
        raise ValueError("typed ReBRAC Config required")
    if not bool(np.asarray(transition_valid(training))):
        raise ValueError(
            "initial training pool must be finite with native action/done bounds"
        )
    obs, action = (training.obs[:1], training.action[:1])
    rng, native, (actor, critic) = BASE.initialize(args, training)
    if not config.is_posterior:
        return (rng, State(native, None, None, None, None, None), (actor, critic, None))
    mean, std = (jnp.mean(training.obs, axis=0), jnp.std(training.obs, axis=0, ddof=0))
    if not bool(np.asarray(tree_finite((mean, std)))):
        raise ValueError("calibrator training-pool statistics must be finite")
    cal = Calibrator(mean, std, state_dep=True)
    cal_state = TrainState.create(
        apply_fn=cal.apply,
        params=cal.init(jax.random.fold_in(rng, KEY_CAL_INIT), obs, action),
        tx=optax.adam(config.cal_lr),
    )
    posterior = initial_reference(cal_state.params)
    return (
        rng,
        State(native, cal_state, jnp.asarray(1.0), posterior, mean, std),
        (actor, critic, cal),
    )


def _reference_storage_valid(reference):
    for value in (
        reference.residual_scale,
        reference.threshold,
        reference.lambda_hat,
        reference.lambda_hpd,
        reference.n_eff,
    ):
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("frozen thresholds and unit must be floating")
    return reference_valid(reference)


def calibration_storage_valid(models, state):
    if (
        state.cal_obs_mean.shape != models[2].obs_mean.shape
        or state.cal_obs_std.shape != models[2].obs_std.shape
        or state.residual_scale.shape != ()
    ):
        raise ValueError("fixed calibrator statistics/unit shapes disagree")
    return (
        tree_finite(
            (
                state.calibrator,
                state.residual_scale,
                state.cal_obs_mean,
                state.cal_obs_std,
            )
        )
        & (state.residual_scale > 0)
        & jnp.all(state.cal_obs_std >= 0)
        & jnp.all(state.cal_obs_mean == models[2].obs_mean)
        & jnp.all(state.cal_obs_std == models[2].obs_std)
        & _reference_storage_valid(state.posterior)
    )


def native_target(args, models, native, batch, rng_noise):
    transition_shape(batch)
    action = models[0].apply(native.actor.target_params, batch.next_obs)
    noise = jnp.clip(
        jax.random.normal(rng_noise, action.shape) * args.policy_noise,
        -args.noise_clip,
        args.noise_clip,
    )
    action = jnp.clip(action + noise, -1, 1)
    penalty = ((action - batch.next_action) ** 2).sum(-1)
    q = models[1].apply(native.critic.target_params, batch.next_obs, action).min(0)
    q = q - args.critic_bc_coef * penalty
    return jax.lax.stop_gradient(batch.reward + (1 - batch.done) * args.gamma * q)


def fit_scale(args, config, models, state, batch, rng):
    if not config.is_posterior:
        raise ValueError("only posterior arms fit a scale")
    valid = transition_valid(batch) & calibration_storage_valid(models, state)
    target = native_target(
        args, models, state.native, batch, jax.random.fold_in(rng, KEY_FIT_NOISE)
    )
    q = jax.lax.stop_gradient(
        models[1].apply(state.native.critic.params, batch.obs, batch.action).min(0)
    )
    prior = jax.lax.stop_gradient(
        bayesian_bootstrap_weights(
            jax.random.fold_in(rng, KEY_BOOTSTRAP), len(batch.obs)
        )
    )
    unit = state.residual_scale

    def objective(params):
        eta = models[2].apply(params, batch.obs, batch.action)
        if eta.shape != (len(batch.obs),):
            raise ValueError("scale predictor must emit one positive value per row")
        coverage = soft_coverage(target / unit, q / unit, eta, config.cal_beta)
        width = jnp.sum(prior * jnp.square(eta))
        return (
            jnp.square(jnp.sum(prior * coverage) - (1 - config.posterior.alpha))
            + config.width_penalty * width,
            eta,
        )

    (loss, eta), grad = jax.value_and_grad(objective, has_aux=True)(
        state.calibrator.params
    )
    proposed = state.calibrator.apply_gradients(grads=grad)
    unit_new = config.scale_ema * unit + (1 - config.scale_ema) * jnp.maximum(
        jnp.std(target - q), 1e-06
    )
    valid = (
        valid
        & tree_finite((state.native, target, q, prior, loss, grad, proposed, unit_new))
        & jnp.all(jnp.isfinite(eta) & (eta > 0))
        & (unit_new > 0)
        & jnp.all(prior >= 0)
        & jnp.any(prior > 0)
        & jnp.isclose(prior.sum(), 1.0, rtol=1e-05, atol=1e-06)
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


def refresh(args, config, models, state, heldout, rng, *, heldout_ids):
    """Freeze the current scale and select the uniform-weight WBCP threshold.

    Runs eagerly (freeze_reference is NumPy): scores are |target - q| / frozen scale
    on the held-out rows. Randomness only folds the refresh key. An invalid refresh
    keeps the previous state and reports posterior_inputs_valid=False.
    """
    if not config.is_posterior:
        raise ValueError("only posterior arms refresh")
    valid = transition_valid(heldout)
    _ids(heldout_ids, len(heldout.obs), "heldout IDs")
    target = native_target(
        args, models, state.native, heldout, jax.random.fold_in(rng, KEY_REFRESH_NOISE)
    )
    q = models[1].apply(state.native.critic.params, heldout.obs, heldout.action).min(0)
    cal = models[2].apply(state.calibrator.params, heldout.obs, heldout.action)
    valid = (
        valid
        & tree_finite((state.native, target, q))
        & calibration_storage_valid(models, state)
        & jnp.all(jnp.isfinite(cal) & (cal > 0))
    )
    reference, frozen, diagnostics = freeze_reference(
        state.calibrator.params,
        state.residual_scale,
        cal,
        target - q,
        jax.random.fold_in(rng, KEY_POSTERIOR),
        config.posterior,
    )
    valid = bool(valid & _reference_storage_valid(reference)) and frozen
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
        inputs_valid=dose.inputs_valid & calibration_storage_valid(models, state)
    )


def consumed_bc_multiplier(config, dose, batch, it, rng):
    return dose


def update(args, config, models, state, batch, it, rng):
    valid = transition_valid(batch)
    fitted, diag, multiplier = (state, {}, None)
    if config.is_posterior:
        fitted, diag = fit_scale(args, config, models, state, batch, rng)
        dose = bc_readout(models, state, batch, config.blend)
        multiplier = consumed_bc_multiplier(config, dose.dose, batch, it, rng)
    native, returned_rng, metrics = BASE.rebrac_update(
        args,
        models[0].apply,
        models[1].apply,
        state.native,
        batch,
        it,
        rng,
        actor_bc_multiplier=multiplier,
    )
    valid = valid & tree_finite((state.native, native, metrics))
    if multiplier is not None:
        valid = (
            valid
            & tree_finite(multiplier)
            & jnp.all((multiplier >= 1) & (multiplier <= 2))
        )
        metrics.update(bc_multiplier_mean=jnp.mean(multiplier))
    if config.is_posterior:
        valid = valid & diag["scale_inputs_valid"] & dose.inputs_valid
        metrics.update(
            bc_support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)),
            posterior_ready=state.posterior.ready,
        )
    result, key = jax.lax.cond(
        valid,
        lambda _: (fitted._replace(native=native), returned_rng),
        lambda _: (state, rng),
        None,
    )
    metrics.update(diag, inputs_valid=valid)
    return (result, key, metrics)


def make_train_step(args, config, models, dataset):
    validate_native_args(args)
    transition_shape(dataset)

    def step(carry, _):
        rng, state, it = carry
        rng, batch_key = jax.random.split(rng)
        indices = jax.random.randint(batch_key, (args.batch_size,), 0, len(dataset.obs))
        batch = jax.tree_util.tree_map(lambda x: x[indices], dataset)
        result, rng, metrics = update(args, config, models, state, batch, it, rng)
        accepted = jax.lax.cond(
            metrics["inputs_valid"],
            lambda _: (rng, result, it + 1),
            lambda _: carry,
            None,
        )
        return (accepted, metrics)

    return step


def require_valid(metrics):
    for name in ("inputs_valid", "scale_inputs_valid", "posterior_inputs_valid"):
        if name in metrics and (not np.all(np.asarray(metrics[name]))):
            raise FloatingPointError(
                "native ReBRAC posterior invalid transition: " + name
            )
