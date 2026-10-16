"""BCA's arithmetic shared by the four restated Unifloral steps: seeds and keys, settings, the scale networks'
constructors, the scale-fit cores, the width and the dose, and the refresh's calibrate wrapper (plan sections 3.4 and
4.1). Training environment; it also imports in /opt/bca-shift-venv (it needs JAX, flax, optax, numpy and scipy only).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. Nothing here
checks permission: every function is pure arithmetic (most run inside jit). The BCA entry points that call it (the
restated steps' factories and refreshes) call guard.require_bca first.

Imported unchanged from calibration/ (the same objects the ports use):
  network.py      Calibrator, soft_coverage, bayesian_bootstrap_weights
  dose.py         tree_finite, level_critic_dose, frozen_level_dose, CQLLevelDose
  reference.py    WBCPConfig, FrozenReference, initial_reference, freeze_reference, positive_scale, reference_valid,
                  certifiable
  iql_network.py  StateDependentCalibrator, create_cal_train_state, make_cal_loss_fn
  iql_loss.py     _FOLD_BOOTSTRAP (1111704112) and _EPS (1e-6)
  advantage.py    postcap_level_actor_weights, LevelActorWeights (for iw.py and the IQL step)
Restated, with the lines they restate:
  fit_scale_core      algorithms/td3_bc_bca.py:170-231, everything after the target y and the critic value q are known.
                      cql_bca.py:185-224 and rebrac_bca.py:290-337 do the same arithmetic; their validity lists differ
                      (CQL's checks fewer things, ReBRAC's adds its batch and storage checks, which belong to the step);
                      the core keeps TD3+BC's list. Test (c1) feeds each port's own y, q and keys to this core and
                      compares the new scale network, its optimizer state and the unit bit for bit.
  iql_fit_core        calibration/iql_scale.py:36-116 (BootstrapScaleFitter.update) after y and the target-critic heads
                      are known, with bootstrap_masses (iql_scale.py:7-11). Test (c1) compares it with the original.
  make_scale_state    the ports' constructors: Calibrator(mean, std, state_dep=True) with optax.adam(cal_lr) at optax's
                      default epsilon 1e-8 (td3_bc_bca.py:132-137, cql_bca.py:132-137, rebrac_bca.py:217-222); IQL:
                      StateDependentCalibrator(mean, std, 64) through create_cal_train_state, epsilon 1e-5
                      (iql_loss.py:67-82, iql_network.py:44-49). Each gets the training rows' mean and standard
                      deviation as Unifloral's training block computes them (choice 3), not the ports' 0 and 1.
  FOLDS               each port's fold_in constants, which keep their roles (plan section 3.4)
  calibrate_bank      the refresh's freeze_reference call (td3_bc_bca.py:263-270), after the 36-row minimum
                      (reference.certifiable, lines 124-136), returning the bank scores and the posterior's random
                      stream exactly as freeze_reference used them (learners_iw.py:540-552 recomputes them that way)
  apply_refresh       the common tail of the ports' refreshes (td3_bc_bca.py:252-280)
"""

import hashlib
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from calibration.advantage import LevelActorWeights, postcap_level_actor_weights  # noqa: F401  (for iw.py, IQL)
from calibration.dose import CQLLevelDose, frozen_level_dose, level_critic_dose, tree_finite  # noqa: F401
from calibration.iql_loss import _EPS, _FOLD_BOOTSTRAP
from calibration.iql_network import StateDependentCalibrator, create_cal_train_state, make_cal_loss_fn
from calibration.network import Calibrator, bayesian_bootstrap_weights, soft_coverage
from calibration.reference import (  # noqa: F401
    FrozenReference, WBCPConfig, certifiable, freeze_reference, initial_reference, positive_scale, reference_valid)

LEARNERS = ("td3_bc", "cql", "iql", "rebrac")

# ------------------------------------------------------------------------------------------------ seeds and keys

MASTER_SEED = 20261016          # the weighting test's master seed (experiments/shift_iw/world.py:62)
TAG_UNIFLORAL = 31              # Unifloral's Args.seed: learners.TAG_INIT (experiments/shift/learners.py:96)
TAG_FIT = 91                    # BCA's per-step fit keys (plan section 3.4; unused by the weighting test)
TAG_REFRESH = 39                # BCA's refresh keys: learners.TAG_REFRESH (experiments/shift/learners.py:97)
CQL_NOISE_STREAM = 1004         # CQL's scoring noise, default_rng([20261016, 1004, bank]) (choice 7)
BANKS = ("none", "mild", "strong", "recorded", "strong_current", "big_waves", "big_kicks")   # bank index order


class Folds(NamedTuple):
    """A port's fold_in constants (None where the learner has no such key)."""

    scale_init: int
    fit_noise: Any
    bootstrap: int
    refresh_noise: Any
    posterior: int


FOLDS = {
    # td3_bc_bca.py:135 (init), 167 (fit target noise), 176 (bootstrap), 249 (refresh target noise), 268 (posterior)
    "td3_bc": Folds(1128352841, 1179210836, 1128352850, 1213156420, 1347375956),
    # cql_bca.py:135, 182, 192, 244, 259
    "cql": Folds(1128352841, 1179210836, 1128352850, 1213156420, 1347375956),
    # rebrac_bca.py:24-28 (KEY_CAL_INIT, KEY_FIT_NOISE, KEY_BOOTSTRAP, KEY_REFRESH_NOISE, KEY_POSTERIOR)
    "rebrac": Folds(1380141390, 1380338004, 1380077391, 1380470604, 1380994899),
    # iql_imports.py:8 (FOLD_BCA_INIT), no target noise, iql_loss.py:14 (_FOLD_BOOTSTRAP), iql_reference.py:14
    "iql": Folds(1111703881, None, _FOLD_BOOTSTRAP, None, 1347375956),
}


class Seeds(NamedTuple):
    rep: int
    unifloral: int      # Args.seed: default_rng([20261016, rep, 31]).integers(2**31 - 1)
    s91: int            # BCA's fit stream: default_rng([20261016, rep, 91]).integers(2**31 - 1)
    s39: int            # BCA's refresh stream: default_rng([20261016, rep, 39]).integers(2**31 - 1)


def stream_seed(rep, tag):
    """default_rng([20261016, rep, tag]).integers(2**31 - 1), the weighting test's seed rule (learners.py:157)."""
    return int(np.random.default_rng([MASTER_SEED, int(rep), int(tag)]).integers(2 ** 31 - 1))


def seeds(rep):
    """The three seeds of data seed `rep` (plan section 3.4)."""
    return Seeds(int(rep), stream_seed(rep, TAG_UNIFLORAL), stream_seed(rep, TAG_FIT), stream_seed(rep, TAG_REFRESH))


def fit_key_base(s91):
    """PRNGKey(s91): the base of BCA's per-step fit keys (create it after unifloral_import.prepare())."""
    return jax.random.PRNGKey(int(s91))


def fit_key(base, t):
    """BCA's fit key at outer step t (1, 2, ..., num_updates): fold_in(PRNGKey(s91), t). Traceable in t."""
    return jax.random.fold_in(base, t)


def scale_init_key(base, learner):
    """The scale network's start: fold_in(fold_in(PRNGKey(s91), 0), the port's init constant). Step 0 is never a
    fit step."""
    return jax.random.fold_in(jax.random.fold_in(base, 0), FOLDS[learner].scale_init)


def fit_noise_key(key, learner):
    """The fit's target-noise key at a step: fold_in(fit key, the port's constant) (td3_bc_bca.py:167,
    cql_bca.py:182, rebrac_bca.py:288). IQL's target has no noise."""
    c = FOLDS[learner].fit_noise
    if c is None:
        raise ValueError(f"{learner}'s band target has no noise")
    return jax.random.fold_in(key, c)


def refresh_key(s39, t):
    """BCA's refresh key at outer step t: fold_in(PRNGKey(s39), t) (learners.py:1197, 1219)."""
    return jax.random.fold_in(jax.random.PRNGKey(int(s39)), int(t))


def refresh_noise_key(key, learner):
    """The refresh's target-noise key: fold_in(refresh key, the port's constant) (td3_bc_bca.py:249,
    cql_bca.py:244, rebrac_bca.py:352). IQL's target has no noise."""
    c = FOLDS[learner].refresh_noise
    if c is None:
        raise ValueError(f"{learner}'s band target has no noise")
    return jax.random.fold_in(key, c)


def posterior_key(key, learner):
    """The refresh's posterior key: fold_in(refresh key, the port's constant) (td3_bc_bca.py:268, cql_bca.py:259,
    rebrac_bca.py:367, iql_reference.py:193)."""
    return jax.random.fold_in(key, FOLDS[learner].posterior)


def per_row_keys(key, n):
    """One key per row, as Unifloral splits its per-row target keys (td3_bc.py:210-211, rebrac.py:237-238,
    cql.py:272-273): jax.random.split(key, n)."""
    return jax.random.split(key, n)


def refresh_steps(num_updates=100_000, warmup=10_000, every=5_000):
    """Outer steps after which BCA refreshes: after the warm-up, every `every` steps, none at the end
    (learners.refresh_steps, experiments/shift/learners.py:1161-1163): 10,000, 15,000, ..., 95,000 (18)."""
    return list(range(int(warmup), int(num_updates), int(every)))


# ------------------------------------------------------------------------------------------------ settings

@dataclass(frozen=True)
class BCASettings:
    """BCA's settings (the bca: blocks of configs/*.yaml; plan section 4.1). Fixed; nothing is tuned."""

    blend: float = 0.5
    cal_lr: float = 1e-3
    cal_beta: float = 20.0
    width_penalty: float = 0.005
    scale_ema: float = 0.99
    miscoverage: float = 0.1
    credibility: float = 0.95
    draws: int = 1000
    scale_hidden: int = 64          # IQL's StateDependentCalibrator (Calibrator's own default is 64 too)
    balance_weight: float = 0.5     # IQL's fitter only
    decision_gain: float = 1.0      # IQL only (iql_reference.gain_actor_weights refuses anything else)
    min_bank_rows: int = 36         # reference.certifiable at 0.1 / 0.95 / 1,000

    def wbcp(self):
        return WBCPConfig(self.miscoverage, self.credibility, self.draws)


SETTINGS = BCASettings()


@dataclass(frozen=True, eq=False)
class BCASetup:
    """The static BCA configuration a restated step closes over (INTERFACES.md, "Restated steps").
    fit_key_base: PRNGKey(s91); permission: guard.MADE_UP or a gate.Approval, checked by the factory with
    guard.require_bca(permission, dataset)."""

    learner: str
    fit_key_base: Any
    permission: Any
    settings: BCASettings = SETTINGS


@dataclass(frozen=True, eq=False)
class TestOptions:
    """Test-only options of a restated step (tests (a3), (a4), (b2)). unifloral_keys: the fit's target uses
    Unifloral's own key at that point of the step, so y equals Unifloral's own target (a4). dose_override (TD3+BC,
    ReBRAC, CQL) / width_override (IQL): a fixed float32 vector over training rows, read through batch_indices, in place
    of the band's dose / width (a3). With any TestOptions the step's output dict also holds the test outputs."""

    unifloral_keys: bool = False
    dose_override: Any = None
    width_override: Any = None


# ------------------------------------------------------------------------------------------------ BCA state

class BCAState(NamedTuple):
    """BCA's part of a restated step's scan carry (rng, agent_state, bca)."""

    scale: Any          # flax TrainState of the scale network being fit (make_scale_state)
    unit: Any           # float32 scalar, the running unit u (starts at 1.0, td3_bc_bca.py:141)
    reference: Any      # calibration.reference.FrozenReference of the last refresh (initial_reference before it)
    thresholds: Any     # float32 [N_train]: each training row's threshold R_i, read as thresholds[batch_indices];
                        # standard BCA holds one value, IW holds R(t_i); +inf = refused; +inf before the first refresh
    most_caution: Any   # bool scalar: the refusal rule for R_i = +inf (False: standard BCA's; True: IW's)
    step: Any           # int32 scalar: outer steps taken; the fit at the next step uses fit_key(base, step + 1)


class FitResult(NamedTuple):
    scale: Any          # the scale network after the fit (unchanged when not valid)
    unit: Any           # the unit after the fit (unchanged when not valid)
    loss: Any
    eta: Any            # the scale network's prediction at the batch rows, before the update
    masses: Any         # the Bayesian-bootstrap masses
    valid: Any          # the fit's own validity flag


class Calibrated(NamedTuple):
    reference: Any      # the new FrozenReference (or initial_reference when invalid)
    valid: bool
    diagnostics: dict   # freeze_reference's (certified, threshold, lambda_hat, lambda_hpd, sigma_post, n_eff, ...)
    scores: Any         # float64 [n_bank]: |y - q| / (max(eta, 1e-6) u), exactly as freeze_reference computed them
    stream: Any         # uint32 words of the posterior key: np.random.default_rng(stream) is freeze_reference's


class Refreshed(NamedTuple):
    state: Any          # the BCAState after the refresh (unchanged when the refresh is invalid)
    record: dict        # JSON-ready refresh readout
    scores: Any
    stream: Any


def make_scale_network(learner, obs_mean, obs_std, settings=SETTINGS):
    """The scale network as each port builds it (module docstring), given the training rows' mean and std."""
    if learner in ("td3_bc", "cql", "rebrac"):
        return Calibrator(obs_mean, obs_std, state_dep=True)                       # td3_bc_bca.py:132
    if learner == "iql":
        return StateDependentCalibrator(obs_mean, obs_std, settings.scale_hidden)  # iql_loss.py:78-81
    raise ValueError(f"learner must be one of {LEARNERS}")


def make_scale_state(learner, obs_mean, obs_std, key, *, obs_dim=13, action_dim=4, settings=SETTINGS):
    """The scale network's TrainState, initialized with `key` (scale_init_key) on one row of zeros, with each port's
    optimizer: optax.adam(1e-3) (epsilon 1e-8; td3_bc_bca.py:133-137) or, for IQL, create_cal_train_state's
    optax.adam(1e-3, eps=1e-5) (iql_network.py:44-49)."""
    net = make_scale_network(learner, obs_mean, obs_std, settings)
    obs, action = jnp.zeros(obs_dim), jnp.zeros(action_dim)
    if learner == "iql":
        return create_cal_train_state(key, net, [obs[None], action[None]], settings.cal_lr)
    return TrainState.create(apply_fn=net.apply, params=net.init(key, obs[None], action[None]),
                             tx=optax.adam(settings.cal_lr))


def initial_state(scale, n_train, *, most_caution=False):
    """BCA's state before the first refresh: unit 1, the reference not ready (initial_reference: threshold +inf,
    ready False, so every width is 0 and every dose 1), every row's threshold +inf, step 0."""
    return BCAState(scale=scale, unit=jnp.asarray(1.0, jnp.float32), reference=initial_reference(scale.params),
                    thresholds=jnp.full((int(n_train),), jnp.inf, jnp.float32),
                    most_caution=jnp.asarray(bool(most_caution)), step=jnp.asarray(0, jnp.int32))


# ------------------------------------------------------------------------------------------------ fit cores

def fit_scale_core(scale, unit, obs, action, target, q, key, *, bootstrap_fold, settings=SETTINGS):
    """One fit of the scale network on a batch, given the learner's own target y and critic value q (restated from
    algorithms/td3_bc_bca.py:170-231). key: the step's fit key; the Bayesian-bootstrap masses use
    fold_in(key, bootstrap_fold) (td3_bc_bca.py:176; FOLDS[learner].bootstrap). The update is kept only when the fit is
    valid (td3_bc_bca.py:199-227). Returns FitResult."""
    n = obs.shape[0]
    target = jax.lax.stop_gradient(target)
    q = jax.lax.stop_gradient(q)
    prior = jax.lax.stop_gradient(bayesian_bootstrap_weights(jax.random.fold_in(key, bootstrap_fold), n))

    def objective(params):
        eta = scale.apply_fn(params, obs, action)
        if eta.shape != (n,):
            raise ValueError("scale predictor must emit one positive value per row")
        coverage = soft_coverage(target / unit, q / unit, eta, settings.cal_beta)
        width = jnp.sum(prior * jnp.square(eta))
        return (
            jnp.square(jnp.sum(prior * coverage) - (1.0 - settings.miscoverage))
            + settings.width_penalty * width,
            eta,
        )

    (loss, eta), grad = jax.value_and_grad(objective, has_aux=True)(scale.params)
    proposed = scale.apply_gradients(grads=grad)
    unit_new = settings.scale_ema * unit + (1.0 - settings.scale_ema) * jnp.maximum(jnp.std(target - q), 1e-06)
    valid = (
        tree_finite((obs, action, scale, target, q, prior, unit, loss, grad, proposed, unit_new))
        & jnp.all(jnp.isfinite(eta) & (eta > 0))
        & jnp.all(prior >= 0)
        & jnp.any(prior > 0)
        & jnp.isclose(prior.sum(), 1.0, rtol=1e-05, atol=1e-06)
        & (unit > 0)
        & (unit_new > 0)
    )
    new_scale, new_unit = jax.lax.cond(valid, lambda _: (proposed, unit_new), lambda _: (scale, unit), None)
    return FitResult(new_scale, new_unit, loss, eta, prior, valid)


def bootstrap_masses(key, n):
    """One Dirichlet(1, ..., 1) draw per batch from fold_in(key, 1111704112) (calibration/iql_scale.py:7-11)."""
    return jax.lax.stop_gradient(bayesian_bootstrap_weights(jax.random.fold_in(key, _FOLD_BOOTSTRAP), n))


def iql_fit_core(scale, unit, obs, action, target, q_heads, key, *, settings=SETTINGS):
    """IQL's scale fit on a batch (restated from calibration/iql_scale.py:36-116, BootstrapScaleFitter.update, with
    cond_mode "state_action", balance weight 0.5 and masses on coverage and width): the current batch's unit
    std(y - q) + 1e-6 scales y and q; q is the minimum over the target-critic heads q_heads [n, 2] (the copy the step's
    advantage reads, choice 15; iql_targets.py:27-31); key is the step's fit key (the masses fold 1111704112 inside);
    the running unit moves as 0.99 u + 0.01 x the batch unit. Returns FitResult."""
    n = target.shape[0]
    heads = jax.lax.stop_gradient(q_heads)
    q = jax.lax.stop_gradient(jnp.min(heads, axis=-1))
    target = jax.lax.stop_gradient(target)
    loss_fn = make_cal_loss_fn(
        cal_apply_fn=scale.apply_fn, cal_mode="soft", cal_beta=settings.cal_beta, cal_alpha=settings.miscoverage,
        cal_lambda_width=settings.width_penalty, cal_cond_mode="state_action", balance_coverage=True,
        balance_weight=settings.balance_weight, weight_width=True,
    )
    unit_now = jax.lax.stop_gradient(jnp.std(target - q) + _EPS)
    weights = bootstrap_masses(key, n)
    (loss, (eta_mean, cov, cov_var, width)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        scale.params, target / unit_now, q / unit_now, weights, obs, action,
    )
    eta = jax.lax.stop_gradient(jnp.broadcast_to(scale.apply_fn(scale.params, obs, action), target.shape))
    masses_valid = jnp.all(jnp.isfinite(weights) & (weights >= 0)) & jnp.any(weights > 0)
    raw_valid = (
        tree_finite((scale, unit, heads, target, grad, loss, eta_mean, cov, cov_var, width, eta))
        & jnp.all(eta > 0)
        & jnp.isfinite(unit_now)
        & (unit_now > 0)
        & (unit > 0)
        & masses_valid
    )
    cal = jax.lax.cond(raw_valid, lambda _: scale.apply_gradients(grads=grad), lambda _: scale, operand=None)
    proposed_unit = 0.99 * unit + 0.01 * unit_now
    accepted = raw_valid & tree_finite((cal, proposed_unit)) & (proposed_unit > 0)
    new_scale, new_unit = jax.lax.cond(accepted, lambda _: (cal, proposed_unit), lambda _: (scale, unit), None)
    return FitResult(new_scale, new_unit, loss, eta, weights, accepted)


# ------------------------------------------------------------------------------------------------ width and dose

def width(threshold, unit, eta):
    """U = R x (max(eta, 1e-6) x u), in calibration/dose.frozen_level_dose's operand order (dose.py:75-76). R may be one
    number or one per row; R = +inf gives U = +inf."""
    scale = jnp.maximum(eta, 1e-06) * unit
    return threshold * scale


def dose(width_, unit, blend, inputs_valid=True):
    """dose = 1 + blend x U / (U + u) per row, by calibration/dose.level_critic_dose (BCA's own arithmetic, with its
    float32 overflow guard): between 1 and 1 + blend; a row with an infinite width is unsupported and gets 1 (standard
    BCA's refusal rule). Returns calibration.dose.CQLLevelDose (width, dose, support_mask, inputs_valid, has_support)."""
    width_ = jnp.asarray(width_)
    return level_critic_dose(width_, jnp.ones_like(width_, dtype=bool), unit, blend, jnp.asarray(inputs_valid))


def frozen_dose(reference, eta, blend):
    """The ports' dose at one threshold: calibration/dose.frozen_level_dose, unchanged (width 0 and dose 1 while the
    reference is not ready)."""
    return frozen_level_dose(reference, eta, blend)


# ------------------------------------------------------------------------------------------------ refresh

def bank_scores(eta, unit, residuals):
    """The bank scores exactly as reference.freeze_reference computes them (reference.py:164-175): scale =
    max(eta, 1e-6) x u in float32 (positive_scale), then float64; |residual| in float64; score = |residual| / scale."""
    scale = np.asarray(positive_scale(jnp.asarray(eta), jnp.asarray(unit)), np.float64)
    return np.abs(np.asarray(residuals, np.float64)) / scale


def posterior_stream(key):
    """The uint32 words freeze_reference seeds its numpy generator with (reference.py:177)."""
    return np.asarray(key, np.uint32).ravel()


def calibrate_bank(scale_params, unit, eta, residuals, key, settings=SETTINGS):
    """The refresh's threshold: calibration/reference.freeze_reference (wbcp.calibrate with miscoverage 0.1,
    credibility 0.95 and 1,000 draws, uniform weights) on the bank residuals y - q with the frozen scale network's eta
    and unit u. Refuses (ValueError) a bank too small to ever certify (reference.certifiable: 36 rows at these
    settings). R = +inf means "cannot certify". key: the refresh's posterior key. Returns Calibrated."""
    n = int(np.shape(residuals)[0])
    if not certifiable(n, settings.wbcp()):
        raise ValueError(f"a bank of {n} rows cannot certify a threshold (at least {settings.min_bank_rows} needed; "
                         "calibration/reference.certifiable)")
    reference, valid, diagnostics = freeze_reference(scale_params, unit, eta, residuals, key, settings.wbcp())
    return Calibrated(reference, bool(valid), diagnostics, bank_scores(eta, unit, residuals), posterior_stream(key))


def _num(v):
    v = float(v)
    return v if np.isfinite(v) else ("inf" if v > 0 else ("-inf" if v < 0 else "nan"))


def apply_refresh(bca, y, q, eta, key, *, inputs_valid=True, settings=SETTINGS, check_new_reference=False):
    """The common tail of every learner's refresh (td3_bc_bca.py:252-280 after y, q and eta are known): checks the
    stored state (finite scale network, unit, y and q; positive finite eta; a valid previous reference), freezes the
    current scale network and unit with the threshold of calibrate_bank on y - q, and fills every row's threshold with
    it (the standard rule; the runner overwrites thresholds for IW). An invalid refresh keeps the previous state.
    check_new_reference=True also requires the NEW frozen reference to pass reference_valid, as the ReBRAC port does
    (rebrac_bca.py:370); the TD3+BC, CQL and IQL ports check only the previous one, so their refreshes keep the default.
    The record's new_reference_valid says how the new reference fared either way.
    key: the refresh's posterior key (posterior_key). Eager (numpy), between scan chunks. Returns Refreshed."""
    y, q, eta = jnp.asarray(y), jnp.asarray(q), jnp.asarray(eta)
    residuals = y - q
    stored = (bool(inputs_valid) and bool(tree_finite((bca.scale, bca.unit, y, q)))
              and bool(reference_valid(bca.reference)) and bool(jnp.all(jnp.isfinite(eta) & (eta > 0))))
    cal = calibrate_bank(bca.scale.params, bca.unit, eta, residuals, key, settings)
    new_ok = bool(reference_valid(cal.reference))
    valid = stored and cal.valid and (new_ok or not check_new_reference)
    state = bca
    if valid:
        state = bca._replace(reference=cal.reference,
                             thresholds=jnp.full_like(bca.thresholds, cal.reference.threshold))
    d = cal.diagnostics
    record = dict(valid=bool(valid), stored_inputs_valid=bool(stored), new_reference_valid=new_ok,
                  certified=bool(valid and d.get("certified")),
                  threshold=_num(cal.reference.threshold) if valid else None,
                  lambda_hat=_num(d["lambda_hat"]) if "lambda_hat" in d else None,
                  lambda_hpd=_num(d["lambda_hpd"]) if "lambda_hpd" in d else None,
                  n_eff=_num(d["n_eff"]) if "n_eff" in d else None,
                  sigma_post=_num(d["sigma_post"]) if "sigma_post" in d else None,
                  clamp_binds=bool(d["clamp_binds"]) if "clamp_binds" in d else None,
                  unit=_num(bca.unit), bank_rows=int(residuals.shape[0]),
                  scores_sha256=hashlib.sha256(np.ascontiguousarray(cal.scores).tobytes()).hexdigest(),
                  stream=[int(w) for w in cal.stream])
    return Refreshed(state, record, cal.scores, cal.stream)
