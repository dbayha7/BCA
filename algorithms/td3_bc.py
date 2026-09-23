"""TD3+BC: full-width Bayesian/conformal modulation of the actor BC term.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import _native_td3_bc as BASE
from _bca_module import Calibrator, bayesian_bootstrap_weights, soft_coverage
from _bca_posterior import PosteriorConfig
from _bca_iql_adapter import initialize_posterior, fit_posterior, reserve_calibration
from _bca_cql_level import frozen_level_dose, tree_finite, level_component_engagement
from _bca_cql_scale_iw import ScaleIWConfig, fit_weighting


def _real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite real scalar")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        v = np.float32(value)
    if not np.isfinite(v) or (positive and v <= 0):
        raise ValueError(name + " must be finite and representable in float32")


@dataclass(frozen=True)
class AffinityIWConfig:
    mode: str = "off"
    bandwidth: float | None = None
    ess_floor: float | None = None
    tau_min: float | None = None
    iterations: int | None = None

    def __post_init__(self):
        if self.mode not in ("off", "affinity", "affinity_permuted"):
            raise ValueError("fitting mode must be off/affinity/affinity_permuted")
        if self.mode == "off":
            if any(
                (
                    v is not None
                    for v in (self.bandwidth, self.ess_floor, self.tau_min, self.iterations)
                )
            ):
                raise ValueError("off fitting cannot carry unused affinity settings")
        else:
            _real(self.bandwidth, "explicit affinity bandwidth", positive=True)
            if any((v is None for v in (self.ess_floor, self.tau_min, self.iterations))):
                raise ValueError(
                    "enabled affinity requires explicit ESS/temperature/iteration settings"
                )
        self.canonical()

    def canonical(self):
        mode = {"off": "off", "affinity": "policy", "affinity_permuted": "policy_permuted"}[
            self.mode
        ]
        return (
            ScaleIWConfig()
            if mode == "off"
            else ScaleIWConfig(mode, self.ess_floor, self.tau_min, self.iterations)
        )


@dataclass(frozen=True)
class Config:
    arm: str
    width_objective: str = "original"
    iw: AffinityIWConfig = field(default_factory=AffinityIWConfig)
    posterior: PosteriorConfig | None = None
    blend: float | None = None
    cal_lr: float = 0.001
    cal_beta: float = 20.0
    width_penalty: float = 0.005
    scale_ema: float = 0.99
    attachment: str = "bc_anchor_level"
    constant_bc_multiplier: float | None = None
    constant_bc_start_step: int = 0
    permute_consumed_dose: bool = False

    def __post_init__(self):
        if self.arm not in ("native_full", "native_pool", "constant_bc_pool", "posterior"):
            raise ValueError(
                "explicit native_full/native_pool/constant_bc_pool/posterior arm required"
            )
        if type(self.constant_bc_start_step) is not int or self.constant_bc_start_step < 0:
            raise ValueError("constant BC start step must be a nonnegative integer")
        if self.arm != "constant_bc_pool" and self.constant_bc_start_step != 0:
            raise ValueError("only constant BC may declare a start step")
        if type(self.permute_consumed_dose) is not bool:
            raise ValueError("consumed-dose permutation must be an explicit bool")
        if self.permute_consumed_dose and (
            self.arm != "posterior" or self.iw.mode != "affinity"
        ):
            raise ValueError("consumed-dose permutation requires the full aligned-IW posterior")
        if self.arm == "constant_bc_pool":
            _real(self.constant_bc_multiplier, "explicit constant_bc_multiplier")
            if not 1 <= self.constant_bc_multiplier <= 2:
                raise ValueError("constant_bc_multiplier must be in [1,2]")
        elif self.constant_bc_multiplier is not None:
            raise ValueError("constant_bc_multiplier is only valid for constant_bc_pool")
        if self.attachment != "bc_anchor_level" or self.width_objective not in (
            "original",
            "matched",
        ):
            raise ValueError("unsupported TD3 attachment/objective")
        if type(self.iw) is not AffinityIWConfig:
            raise ValueError("typed affinity configuration required")
        if self.iw.mode != "off" and self.width_objective != "matched":
            raise ValueError("affinity fitting requires the matched width objective")
        if self.arm == "posterior":
            if type(self.posterior) is not PosteriorConfig:
                raise ValueError("explicit Bayesian/conformal posterior configuration required")
            _real(self.blend, "explicit blend")
            if not 0 <= self.blend <= 1:
                raise ValueError("blend must be in [0,1]")
        elif (
            self.posterior is not None
            or self.blend is not None
            or self.iw.mode != "off"
            or (self.width_objective != "original")
        ):
            raise ValueError("native controls cannot carry calibration interventions")
        for name in ("cal_lr", "cal_beta", "width_penalty", "scale_ema"):
            _real(getattr(self, name), name, positive=name in ("cal_lr", "cal_beta"))
        if self.width_penalty < 0 or not 0 <= self.scale_ema < 1:
            raise ValueError("invalid width penalty or residual EMA")


class State(NamedTuple):
    native: Any
    calibrator: Any
    residual_scale: Any
    posterior: Any


class AffinitySignal(NamedTuple):
    log_weights: Any
    inputs_valid: Any


def posterior_storage_valid(post):
    if post.ready.shape != () or post.ready.dtype != jnp.bool_:
        raise ValueError("posterior readiness must be a boolean scalar")
    r = post.radii
    if (
        r.radius.shape != (1,)
        or post.residual_scale.shape != ()
        or post.group_edges.shape != (0,)
    ):
        raise ValueError("TD3 requires one global posterior and one scalar residual unit")
    valid = tree_finite((post.cal_params, post.residual_scale)) & (post.residual_scale > 0)
    for value in (r.radius, r.conformal_radius, r.bayesian_radius, r.posterior_quantiles):
        valid = valid & jnp.all(~jnp.isnan(value) & (value >= 0))
    valid = (
        valid
        & tree_finite((r.effective_sample_size, r.supported_count))
        & jnp.all(r.effective_sample_size >= 0)
        & jnp.all(r.supported_count >= 0)
        & jnp.all(r.radius == jnp.maximum(r.conformal_radius, r.bayesian_radius))
    )
    return valid & (
        ~post.ready | jnp.all(r.inputs_valid) & jnp.all(r.finite == jnp.isfinite(r.radius))
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
    n = len(dataset.obs)
    if n < 1 or any((len(x) != n for x in dataset)):
        raise ValueError("aligned nonempty native transition arrays required")
    if arm == "native_full":
        if training_indices is not None:
            raise ValueError("native_full cannot carry a reserved pool")
        return dataset
    if arm not in ("native_pool", "constant_bc_pool", "posterior") or training_indices is None:
        raise ValueError("reserved controls/posterior require explicit training indices")
    idx = np.asarray(training_indices)
    if idx.ndim != 1:
        raise ValueError("training indices must be a vector")
    _ids(idx, len(idx), "training indices")
    if not 0 < len(idx) < n or np.any(idx >= n):
        raise ValueError("training pool must be a nonempty strict subset")
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
        "corl_td3_bc", args.dataset, args.reward_transform, args.reward_scale, args.reward_bias
    )
    BASE._published_config.check_published_config(
        "corl_td3_bc", args.dataset, args, allow_off_config=args.allow_off_config
    )
    _real(max_action, "max_action", positive=True)
    if (
        getattr(args, "cal_iw", False)
        or getattr(args, "acrab_iw_bellman", False)
        or getattr(args, "acrab_cinf", -1.0) >= 0
    ):
        raise ValueError("legacy IW/Bellman/A-Crab are separate interventions")
    rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1414676809)
    rng, ka, kc = jax.random.split(rng, 3)
    actor, critic = (
        BASE.Actor(obs_dim, action_dim, max_action),
        BASE.DualCritic(obs_dim, action_dim),
    )
    obs, action = (jnp.zeros(obs_dim), jnp.zeros(action_dim))

    def make(key, net, inputs):
        return TrainState.create(
            apply_fn=net.apply, params=net.init(key, *inputs), tx=BASE.C.torch_adam(args.lr)
        )

    native = BASE.AgentTrainState(
        make(ka, actor, [obs]),
        make(ka, actor, [obs]),
        make(kc, critic, [obs, action]),
        make(kc, critic, [obs, action]),
    )
    if config.arm != "posterior":
        return (rng, State(native, None, None, None), (actor, critic, None))
    cal = Calibrator(jnp.zeros(obs_dim), jnp.ones(obs_dim), state_dep=True)
    cal_state = TrainState.create(
        apply_fn=cal.apply,
        params=cal.init(jax.random.fold_in(rng, 1128352841), obs[None], action[None]),
        tx=optax.adam(config.cal_lr),
    )
    posterior = initialize_posterior(cal_state.params, 1, config.posterior.draws)
    return (rng, State(native, cal_state, jnp.asarray(1.0), posterior), (actor, critic, cal))


def affinity_log_weights(actor_apply, params, obs, actions, bandwidth, max_action=1.0):
    _real(bandwidth, "explicit affinity bandwidth", positive=True)
    _real(max_action, "max_action", positive=True)
    obs, actions = (jnp.asarray(obs), jnp.asarray(actions))
    if obs.ndim != 2 or actions.ndim != 2 or len(obs) != len(actions) or (not len(obs)):
        raise ValueError("nonempty aligned observation/action matrices required")
    detached = jax.tree_util.tree_map(jax.lax.stop_gradient, (params, obs, actions))
    mu = actor_apply(detached[0], detached[1])
    if mu.shape != actions.shape:
        raise ValueError("deterministic actor must emit one action vector per row")
    logw = -0.5 * jnp.sum(jnp.square((detached[2] - mu) / bandwidth), axis=-1)
    valid = (
        tree_finite((params, obs, actions, mu, logw))
        & jnp.all(jnp.abs(actions) <= max_action)
        & jnp.all(jnp.abs(mu) <= max_action)
    )
    return jax.tree_util.tree_map(jax.lax.stop_gradient, AffinitySignal(logw, valid))


def native_target(args, models, native, batch, rng_noise, max_action=1.0):
    noise = jax.random.normal(rng_noise, batch.action.shape) * args.policy_noise
    noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    action = models[0].apply(native.actor_target.params, batch.next_obs)
    action = jnp.clip(action + noise, -max_action, max_action)
    q = models[1].apply(native.critic_target.params, batch.next_obs, action).min(axis=-1)
    return jax.lax.stop_gradient(batch.reward + (1.0 - batch.done) * args.discount * q)


def fit_scale(args, config, models, state, batch, rng, max_action=1.0):
    if config.arm != "posterior":
        raise ValueError("only posterior arms fit a scale")
    target = native_target(
        args, models, state.native, batch, jax.random.fold_in(rng, 1179210836), max_action
    )
    q = jax.lax.stop_gradient(
        models[1].apply(state.native.critic.params, batch.obs, batch.action).min(axis=-1)
    )
    prior = jax.lax.stop_gradient(
        bayesian_bootstrap_weights(jax.random.fold_in(rng, 1128352850), len(batch.obs))
    )
    weights, valid, feasible = (prior, jnp.asarray(True), jnp.asarray(True))
    diag = {}
    if config.iw.mode != "off":
        signal = affinity_log_weights(
            models[0].apply,
            state.native.actor.params,
            batch.obs,
            batch.action,
            config.iw.bandwidth,
            max_action,
        )
        tilted = fit_weighting(
            signal.log_weights,
            prior,
            jax.random.fold_in(rng, 1413761367),
            config.iw.canonical(),
        )
        weights, valid = (tilted.product.weights, signal.inputs_valid & tilted.inputs_valid)
        feasible = tilted.fit_feasible
        diag = dict(
            iw_ess=tilted.iw.ess_fraction,
            iw_tau=tilted.iw.tau,
            iw_product_ess=tilted.product.product_ess_fraction,
        )
    unit = state.residual_scale

    def objective(params):
        eta = models[2].apply(params, batch.obs, batch.action)
        if eta.shape != (len(batch.obs),):
            raise ValueError("scale predictor must emit one positive value per row")
        coverage = soft_coverage(target / unit, q / unit, eta, config.cal_beta)
        width = (
            jnp.sum(weights * jnp.square(eta))
            if config.width_objective == "matched"
            else jnp.mean(jnp.square(eta))
        )
        return (
            jnp.square(jnp.sum(weights * coverage) - (1.0 - config.posterior.alpha))
            + config.width_penalty * width,
            eta,
        )

    (loss, eta), grad = jax.value_and_grad(objective, has_aux=True)(state.calibrator.params)
    proposed = state.calibrator.apply_gradients(grads=grad)
    unit_new = config.scale_ema * unit + (1.0 - config.scale_ema) * jnp.maximum(
        jnp.std(target - q), 1e-06
    )
    valid = (
        valid
        & tree_finite(
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
    accepted = valid & feasible
    result = jax.lax.cond(
        accepted,
        lambda _: state._replace(calibrator=proposed, residual_scale=unit_new),
        lambda _: state,
        None,
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
    args,
    config,
    models,
    state,
    training_reference,
    heldout,
    rng,
    max_action=1.0,
    *,
    training_ids,
    heldout_ids
):
    if config.arm != "posterior" or not len(training_reference.obs) or (not len(heldout.obs)):
        raise ValueError("posterior refresh requires nonempty training and held-out references")
    train = _ids(training_ids, len(training_reference.obs), "training IDs")
    hold = _ids(heldout_ids, len(heldout.obs), "held-out IDs")
    if np.intersect1d(train, hold).size:
        raise ValueError("training and held-out row identities overlap")
    target = native_target(
        args, models, state.native, heldout, jax.random.fold_in(rng, 1213156420), max_action
    )
    q = models[1].apply(state.native.critic.params, heldout.obs, heldout.action).min(axis=-1)
    fit = models[2].apply(
        state.calibrator.params, training_reference.obs, training_reference.action
    )
    cal = models[2].apply(state.calibrator.params, heldout.obs, heldout.action)
    post = fit_posterior(
        state.calibrator.params,
        state.residual_scale,
        fit,
        cal,
        target - q,
        jax.random.fold_in(rng, 1347375956),
        config.posterior,
        groups=1,
    )
    valid = (
        tree_finite((state.native, state.calibrator, training_reference, heldout, target, q))
        & posterior_storage_valid(state.posterior)
        & jnp.all(jnp.isfinite(fit) & (fit > 0))
        & jnp.all(jnp.isfinite(cal) & (cal > 0))
        & jnp.all(post.radii.inputs_valid)
    )
    result = jax.lax.cond(
        valid, lambda _: state._replace(posterior=post), lambda _: state, None
    )
    return (
        result,
        dict(posterior_inputs_valid=valid, posterior_supported=jnp.all(post.radii.finite)),
    )


def bc_readout(models, state, batch, blend, *, mode="full"):
    predictions = models[2].apply(state.posterior.cal_params, batch.obs, batch.action)
    dose = frozen_level_dose(state.posterior, predictions, mode, blend)
    return dose._replace(
        inputs_valid=dose.inputs_valid & posterior_storage_valid(state.posterior)
    )


def component_diagnostics(config, models, state, batch):
    predictions = models[2].apply(state.posterior.cal_params, batch.obs, batch.action)
    return level_component_engagement(state.posterior, predictions, config.blend)


def consumed_bc_multiplier(config, dose, batch, it, rng):
    if dose is not None:
        return (
            jax.random.permutation(jax.random.fold_in(rng, 1146049349), dose)
            if config.permute_consumed_dose
            else dose
        )
    if config.arm == "constant_bc_pool":
        strength = jnp.where(
            it > config.constant_bc_start_step, config.constant_bc_multiplier, 1.0
        )
        return jnp.full((len(batch.obs),), strength, dtype=batch.action.dtype)
    return None


def update(args, config, models, state, batch, it, rng, max_action=1.0):
    fitted, diag, multiplier = (state, {}, None)
    if config.arm == "posterior":
        fitted, diag = fit_scale(args, config, models, state, batch, rng, max_action)
        dose = bc_readout(models, state, batch, config.blend)
        multiplier = consumed_bc_multiplier(config, dose.dose, batch, it, rng)
    elif config.arm == "constant_bc_pool":
        multiplier = consumed_bc_multiplier(config, None, batch, it, rng)
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
    if config.arm == "posterior":
        valid = valid & diag["scale_inputs_valid"] & dose.inputs_valid
        metrics.update(
            bc_multiplier_mean=jnp.mean(multiplier),
            bc_support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)),
            posterior_ready=state.posterior.ready,
        )
    elif config.arm == "constant_bc_pool":
        metrics.update(bc_multiplier_mean=jnp.mean(multiplier))
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
        result, metrics = update(args, config, models, state, batch, it + 1, kn, max_action)
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
