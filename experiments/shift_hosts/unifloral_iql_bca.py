"""Unifloral's IQL training step, restated line for line with BCA's additions, for the standard and IW setups of plan
section 3.7 (PLAN_UNIFLORAL.md, 15 October 2026). Restated from baselines/unifloral/algorithms/iql.py:193-268 (commit
f2dc127, pinned in configs/unifloral.json).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. The
repository's IQL + BCA stays the CORL-derived port (algorithms/iql_bca.py; ALGORITHMS.md, Algorithm 1), and Unifloral's
iql.py is imported unchanged and never edited. Unifloral's networks, create_train_state, AgentTrainState, Transition and
Args are not copied: they come from the imported module (unifloral_import.load("iql"); this module re-exports them).

Who uses what (choice 11: three separate runs per seed):
  alone       unifloral_import.make_unifloral_step: Unifloral's own, imported make_train_step, unchanged.
  standard    make_train_step(..., bca=BCASetup(...)), carry (rng, agent_state, bca_state) with most_caution False and
              every row's threshold the refresh's standard threshold.
  IW          the same compiled program, most_caution True and each training row's own threshold R(w_i).
  bca=None    the restated lines alone: the program Unifloral's own make_train_step traces (test (a1)).

What is restated. make_train_step and its inner _train_step, iql.py:193-268. Inside _train_step every line is
Unifloral's, verbatim, except the lines marked "# BCA:". With bca=None none of them runs, so the traced program is
Unifloral's. BCA's additions (plan sections 4.1-4.3; choices 15 and 16; INTERFACES.md section 10):
  carry       (rng, agent_state, bca_state), bca_state = bca_core.BCAState(scale, unit, reference, thresholds,
              most_caution, step). Unifloral's key is split only for the batch, as in iql.py:200, so all three setups
              draw the same batches; BCA never consumes Unifloral's key (plan 3.4).
  fit key     t = bca_state.step + 1 (this outer step's number, 1 ... 100,000); k = fold_in(PRNGKey(s91), t)
              (bca_core.fit_key). The fitter's Bayesian-bootstrap masses fold 1111704112 into k (iql_scale.py:7-11).
  band fit    once per outer step, after the target averaging and q_targets (iql.py:206-221), before the Q and V
              update (plan 4.3): y = q_targets = r + 0.98 (1 - d) V(s') with V before its update, no randomness
              (iql.py:220-221); q = the minimum over the two heads of the just-averaged target critic at (s, a), the
              copy this step's advantage reads (iql.py:218-219; choice 15). IQL's own fitter: bca_core.iql_fit_core,
              restated from calibration/iql_scale.py:36-116 (balance weight 0.5, the batch's own unit, masses on
              coverage and width; scale network StateDependentCalibrator with 64 hidden units, Adam epsilon 1e-5).
  weights     after Unifloral's exp_adv (iql.py:245): the frozen band at the recorded (s, a), no gradient:
              eta = scale.apply_fn(reference.cal_params, s, a), each row's threshold R_b = thresholds[batch_indices]
              (Unifloral's own batch_indices), then iw.per_row_iql_weights: on supported positive-advantage rows
              w_i = 1 + A_i / (A_i + U_i) x (min(e^(3 A_i), 100) - 1) with U_i = R_b x max(eta_i, 1e-6) x u (BCA's
              post-cap shrinkage, calibration/advantage.py:32-70); on every other row, and on every row before the
              first refresh, Unifloral's own weight jnp.exp(adv * args.beta).clip(max=args.exp_adv_clip), so a band
              that is not ready reproduces Unifloral bit for bit (test (a2)). Refused rows (ready, R_b = +inf):
              standard BCA's rule (most_caution False) gives weight 0, the port's unsupported row (ALGORITHMS.md
              procedure D, calibration/iql_reference.py:174-181); IW's rule (most_caution True, choice 16, a new rule)
              gives weight 1 where A > 0 and Unifloral's weight elsewhere. The weights replace exp_adv in the actor
              loss and nothing else changes: the loss stays Unifloral's mean over rows (not a division by the
              surviving weight mass, ALGORITHMS.md procedure D).
  skip rule   no supported row (or invalid weights): the actor keeps its parameters, optimizer state and step count, so
              its cosine schedule does not advance (calibration/iql_state.py:45-79). The proposed update is computed as
              Unifloral computes it and then not kept, which leaves Unifloral's actor computation unchanged when it is
              kept. The port also zeroes unsupported rows' observations and commands before its actor loss
              (iql_state.py:47-48); that is not restated: those rows carry weight exactly 0, and any non-finite row
              would already make Unifloral's own Q and V losses non-finite and the step invalid.
  validity    valid = the fit's own flag & learner_valid, where learner_valid = the weights' inputs_valid & finite
              weights & every leaf of the new agent state and of Unifloral's loss dict finite (the ports' per-step
              flag, td3_bc_bca.py:312-322, plan 4.1); the runner stops on any false flag (stop rule S5). A step that
              skips the actor because no row is supported is valid. What a false flag keeps: BCA's part of the carry
              (scale network, unit, step count) is kept whenever valid is false; the learner's part (Unifloral's key
              and agent state) is kept whenever learner_valid is false (td3_bc_bca.py:340-342, random key included).
              So a step is discarded whole on every failure except one: when only the band fit fails (its own
              arithmetic overflows while every learner value is finite), the fit is rejected (bca_core.iql_fit_core
              keeps the old scale network and unit, as calibration/iql_scale.py:81-91 does) and the flag is false, but
              the learner's own update of that step stands, the actor's included. This differs from the IQL port in
              part: the port never rolls back its Q and V update either (algorithms/iql_bca.py:220-226), but it skips
              that step's actor update (its dependency_valid includes the fit's numerical_valid and is ANDed into
              has_support, iql_bca.py:265-280), while this step keeps the actor update too. The difference exists
              only on a step whose flag is false, and the runner stops the run there (S5), so no reported result
              depends on it. The step's actor_updated output is actor_ok & valid, so on such a step it reads False
              although the carried actor moved. The reason is measured, not chosen: making the learner's new state
              depend on the fit's flag changes how XLA compiles Unifloral's own arithmetic on the processor (about one
              unit in the last place at the first step, 1.7e-5 relative after 60 steps on made-up arrays), which breaks
              test (a2)'s bit-for-bit equality; with the learner's part selected on learner_valid alone, (a2) holds bit
              for bit (tests/test_iql.py). Whenever a learner value is non-finite, learner_valid is false as well, so
              the two selections differ only when the fit's own arithmetic fails.
  outputs     Unifloral's loss dict plus BCA's per-step scalars: inputs_valid, scale_inputs_valid,
              dose_inputs_valid, scale_loss, unit (after the step), posterior_ready, weight_mean, weight_min,
              weight_max, actor_updated, supported_fraction, refused_fraction, support_fraction (for IQL
              support_fraction equals supported_fraction: the share of batch rows with support).
  test mode   test=bca_core.TestOptions(...) (tests only) adds per-row arrays: batch_indices, bca_y, bca_q, bca_heads,
              unifloral_y (q_targets), unifloral_q (v_target after .min(-1)), adv, eta, width, weight,
              unifloral_weight (Unifloral's exp_adv), refused, support_mask, has_support, dosed_loss (the actor loss)
              and dosed_grad (its gradient). width_override (float32 over training rows, read through batch_indices)
              replaces the band's width for test (a3) (weights_at_widths). unifloral_keys has no effect here: IQL's
              band target uses no randomness, so y is Unifloral's own q_targets in every mode. dose_override is
              refused (IQL has no dose).

Also here, for the runner and band_readout.py:
  band_parts(args, apply_fns, agent_state, rows, noise_key=None) -> (y, q), from a stored state: y = r + 0.98 (1 - d)
      V(s') with the stored V; q = min over the two heads of Polyak(stored target, stored critic, 0.005), the copy the
      next outer step's advantage reads (plan 4.3 "Refresh, per learner"; choice 15). No randomness.
  refresh(args, apply_fns, agent_state, bca_state, bank, refresh_key, *, permission) -> bca_core.Refreshed: the guard,
      band_parts on the bank, eta from the live scale network, then bca_core.apply_refresh with the posterior key
      fold_in(refresh key, 1347375956) (calibration/iql_reference.py:14, 193). Eager, between chunks.
  critic_value_parts(agent_state): (dual_q, dual_q_target, value), the parts the three IQL runs of a seed share (plan
      sections 3.7 and 5.4): Unifloral's Q and V updates never read the actor (iql.py:206-242). On the processor all
      three runs share them bit for bit; on the card the standard and IW runs (one compiled program) share them bit for
      bit, while the alone run (another compiled program) drifts from them at rounding level inside lax.scan (test
      (a2)'s card exception; tests/test_iql.py, shared critic).

Imports beyond Unifloral's file: experiments.shift_hosts.bca_core (calibration/advantage.py, dose.py, iql_loss.py,
iql_network.py, network.py, reference.py through it), guard, and iw (per_row_iql_weights, imported when BCA is present).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.shift_hosts import bca_core as BC
from experiments.shift_hosts import guard

LEARNER = "iql"
UNIFLORAL_FILE = "baselines/unifloral/algorithms/iql.py"
RESTATED_LINES = (193, 268)                    # make_train_step, iql.py:193-268
UNIFLORAL_LOSSES = ("value_loss", "q_loss", "actor_loss")
BCA_METRICS = ("inputs_valid", "scale_inputs_valid", "dose_inputs_valid", "scale_loss", "unit", "posterior_ready",
               "weight_mean", "weight_min", "weight_max", "actor_updated", "supported_fraction", "refused_fraction",
               "support_fraction")
_REEXPORTED = ("Args", "AgentTrainState", "Transition", "SoftQNetwork", "DualQNetwork", "StateValueFunction",
               "TanhGaussianActor", "create_train_state")


def unifloral():
    """Unifloral's imported iql.py (unifloral_import.prepare() must have run in this process)."""
    from experiments.shift_hosts import unifloral_import as UI

    return UI.load(LEARNER)


def __getattr__(name):
    """Unifloral's own classes and functions, from the imported module (never copied)."""
    if name in _REEXPORTED:
        return getattr(unifloral(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _iw():
    from experiments.shift_hosts import iw

    return iw


# ------------------------------------------------------------------------------------------------ the restated step

def make_train_step(args, actor_apply_fn, q_apply_fn, value_apply_fn, dataset, *, bca=None, test=None):  # BCA: bca, test
    """Make JIT-compatible agent train step."""
    if bca is not None or test is not None:  # BCA: checks before any BCA code runs (INTERFACES.md section 10)
        _check_bca(args, dataset, bca, test)  # BCA:

    def _train_step(runner_state, _):
        if bca is not None:  # BCA: the carry is (rng, agent_state, bca_state) (INTERFACES.md section 10)
            carry_in, bca_state = tuple(runner_state), runner_state[2]  # BCA:
            runner_state = carry_in[:2]  # BCA:
            _check_carry(bca_state, dataset)  # BCA: (shapes, at trace time)
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)
        if bca is not None:  # BCA: this outer step's number t and BCA's own fit key fold_in(PRNGKey(s91), t)
            t = bca_state.step + 1  # BCA:
            k = BC.fit_key(bca.fit_key_base, t)  # BCA:

        # --- Update Q target network ---
        updated_q_target_params = optax.incremental_update(
            agent_state.dual_q.params,
            agent_state.dual_q_target.params,
            args.polyak_step_size,
        )
        updated_q_target = agent_state.dual_q_target.replace(
            step=agent_state.dual_q_target.step + 1, params=updated_q_target_params
        )
        agent_state = agent_state._replace(dual_q_target=updated_q_target)

        # --- Compute targets ---
        v_target = q_apply_fn(agent_state.dual_q_target.params, batch.obs, batch.action)
        if bca is not None:  # BCA: the two target heads at (s, a), before .min(-1): the copy the advantage reads
            fit_heads = v_target  # BCA: (choice 15)
        v_target = v_target.min(-1)
        next_v_target = value_apply_fn(agent_state.value.params, batch.next_obs)
        q_targets = batch.reward + args.gamma * (1 - batch.done) * next_v_target
        if bca is not None:  # BCA: the band fit, once per outer step, before the Q and V update (plan 4.3)
            fit_y = q_targets  # BCA: y = r + 0.98 (1 - d) V(s'), V before its update, no randomness
            fit = BC.iql_fit_core(  # BCA:
                bca_state.scale, bca_state.unit, batch.obs, batch.action, fit_y, fit_heads, k,  # BCA:
                settings=bca.settings,  # BCA:
            )  # BCA:

        # --- Update Q and value functions ---
        def _q_loss_fn(params):
            # Compute loss for both critics
            q_pred = q_apply_fn(params, batch.obs, batch.action)
            q_loss = jnp.square(q_pred - jnp.expand_dims(q_targets, axis=-1)).mean()
            return q_loss

        @partial(jax.value_and_grad, has_aux=True)
        def _value_loss_fn(params):
            adv = v_target - value_apply_fn(params, batch.obs)
            # Asymmetric L2 loss
            value_loss = jnp.abs(args.iql_tau - (adv < 0.0).astype(float)) * (adv**2)
            return jnp.mean(value_loss), adv

        q_loss, q_grad = jax.value_and_grad(_q_loss_fn)(agent_state.dual_q.params)
        (value_loss, adv), value_grad = _value_loss_fn(agent_state.value.params)
        agent_state = agent_state._replace(
            dual_q=agent_state.dual_q.apply_gradients(grads=q_grad),
            value=agent_state.value.apply_gradients(grads=value_grad),
        )

        # --- Update actor ---
        exp_adv = jnp.exp(adv * args.beta).clip(max=args.exp_adv_clip)
        if bca is not None:  # BCA: BCA's weights replace exp_adv in the actor loss (plan 4.3; choice 16)
            unifloral_exp_adv = exp_adv  # BCA:
            bw, refused, eta, width = _bca_weights(args, bca_state, batch, batch_indices, adv, test)  # BCA:
            exp_adv = jax.lax.stop_gradient(bw.weights)  # BCA:
            actor_ok = bw.has_support & bw.inputs_valid  # BCA: no supported row: no actor step (iql_state.py:79)

        @jax.value_and_grad
        def _actor_loss_function(params):
            def _compute_loss(transition, exp_adv):
                pi = actor_apply_fn(params, transition.obs)
                bc_loss = -pi.log_prob(transition.action)
                return exp_adv * bc_loss.sum()

            actor_loss = jax.vmap(_compute_loss)(batch, exp_adv)
            return actor_loss.mean()

        actor_loss, actor_grad = _actor_loss_function(agent_state.actor.params)
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        if bca is not None:  # BCA: a skipped actor step keeps its parameters, Adam state and step count
            updated_actor = _keep_unless(actor_ok, updated_actor, agent_state.actor)  # BCA: (iql_state.py:72-79)
        agent_state = agent_state._replace(actor=updated_actor)

        loss = {
            "value_loss": value_loss,
            "q_loss": q_loss,
            "actor_loss": actor_loss,
        }
        if bca is not None:  # BCA: the step's validity flag, the new carry and BCA's outputs (INTERFACES.md 10)
            learner_valid = (  # BCA: the learner's side of the ports' per-step flag (td3_bc_bca.py:312-322)
                bw.inputs_valid & jnp.all(jnp.isfinite(bw.weights)) & BC.tree_finite((agent_state, loss))  # BCA:
            )  # BCA:
            valid = fit.valid & learner_valid  # BCA: the step's flag (plan 4.1); the runner stops on False (S5)
            learner = _keep_unless(learner_valid, (rng, agent_state), carry_in[:2])  # BCA: (td3_bc_bca.py:340-342)
            new_bca = _keep_unless(  # BCA: BCA's part is kept only when the whole step is valid
                valid, bca_state._replace(scale=fit.scale, unit=fit.unit, step=t), bca_state)  # BCA:
            carry = (*learner, new_bca)  # BCA:
            outputs = dict(loss, **_metrics(valid, fit, bw, refused, actor_ok, carry[2], bca_state))  # BCA:
            if test is not None:  # BCA: test mode only (tests (a3), (a4), (b2), (b3))
                outputs.update(  # BCA:
                    batch_indices=batch_indices, bca_y=fit_y, bca_q=jnp.min(fit_heads, axis=-1),  # BCA:
                    bca_heads=fit_heads, unifloral_y=q_targets, unifloral_q=v_target, adv=adv, eta=eta,  # BCA:
                    width=width, weight=bw.weights, unifloral_weight=unifloral_exp_adv, refused=refused,  # BCA:
                    support_mask=bw.support_mask, has_support=bw.has_support, dosed_loss=actor_loss,  # BCA:
                    dosed_grad=actor_grad,  # BCA:
                )  # BCA:
            return carry, outputs  # BCA:
        return (rng, agent_state), loss

    return _train_step


# ------------------------------------------------------------------------------------------------ BCA's helpers

def _check_bca(args, dataset, bca, test):
    """The factory's checks, before any BCA code runs (INTERFACES.md section 10): BCA present for test options, the
    setup's learner, guard.require_bca on the training rows (made-up arrays registered with the guard, or the user's
    terminal approval), IQL's fixed decision gain 1 (calibration/iql_reference.py:102-105) and the test options."""
    if bca is None:
        raise ValueError("test options need BCA present (INTERFACES.md section 10, test mode)")
    if not isinstance(bca, BC.BCASetup) or bca.learner != LEARNER:
        raise ValueError(f"this step is IQL's; the BCA setup is for {getattr(bca, 'learner', bca)!r}")
    guard.require_bca(bca.permission, dataset)
    if float(bca.settings.decision_gain) != 1.0:
        raise ValueError("The IQL BCA decision gain is fixed at one (calibration/iql_reference.py:102-105).")
    if test is not None:
        if not isinstance(test, BC.TestOptions):
            raise TypeError("test must be a bca_core.TestOptions")
        if bca.permission is not guard.MADE_UP:                  # as the other three restated steps refuse them
            raise guard.ApprovalError("test options run on made-up arrays only")
        if test.dose_override is not None:
            raise ValueError("IQL has no dose: test (a3) overrides the band's width (width_override)")
        if test.width_override is not None and np.shape(test.width_override) != (len(dataset.obs),):
            raise ValueError("width_override must hold one width per training row")
    _iw()   # the per-row weights (iw.per_row_iql_weights): fail here, not inside the traced step


def _check_carry(bca_state, dataset):
    """BCA's carry at trace time: a BCAState whose thresholds cover every training row (read through batch_indices)."""
    if not isinstance(bca_state, BC.BCAState):
        raise TypeError("the carry's third member must be a bca_core.BCAState")
    if tuple(bca_state.thresholds.shape) != (len(dataset.obs),) or bca_state.thresholds.dtype != jnp.float32:
        raise ValueError("bca_state.thresholds must be float32 with one threshold per training row")


def _keep_unless(flag, proposed, current):
    """proposed when flag is true, else current (the ports' jax.lax.cond(valid, proposed, current),
    td3_bc_bca.py:340-342, calibration/iql_state.py:67)."""
    return jax.lax.cond(flag, lambda _: proposed, lambda _: current, None)


def _bca_weights(args, bca_state, batch, batch_indices, adv, test):
    """BCA's actor weights at the batch rows (plan 4.3; INTERFACES.md section 10, step 4). The frozen band is read at
    the recorded (s, a) with no gradient: eta = the scale network with the reference's frozen parameters (the port's
    predictions, algorithms/iql_bca.py:247-249); each row's threshold through Unifloral's own batch_indices; then
    iw.per_row_iql_weights (module docstring). Test (a3)'s width override replaces the band's width
    (weights_at_widths). Returns (LevelActorWeights, refused, eta, width): refused marks rows with a ready band and
    R = +inf; width is each row's band width as used (0 before the first refresh, +inf where refused)."""
    reference = bca_state.reference
    eta = jax.lax.stop_gradient(bca_state.scale.apply_fn(reference.cal_params, batch.obs, batch.action))
    rows_r = bca_state.thresholds[batch_indices]
    if test is not None and test.width_override is not None:
        width = jnp.asarray(test.width_override, jnp.float32)[batch_indices]
        return weights_at_widths(adv, width, args.beta, args.exp_adv_clip), jnp.zeros(adv.shape, bool), eta, width
    weights, refused = _iw().per_row_iql_weights(reference, eta, rows_r, adv, args.beta, args.exp_adv_clip,
                                                 bca_state.most_caution)
    width = jnp.where(reference.ready, BC.width(rows_r, reference.residual_scale, eta), 0.0)
    return weights, refused, eta, width


def weights_at_widths(adv, width, beta, cap):
    """IQL's BCA weights at given widths, the band ready and every row supported (test (a3)'s width override): BCA's
    post-cap shrinkage w = 1 + A / (A + U) x (min(e^(beta A), cap) - 1) on positive-advantage rows
    (calibration/advantage.py:32-70, postcap_level_actor_weights), Unifloral's own weight jnp.exp(adv * beta).clip(max=cap)
    (iql.py:245) on the others. Returns calibration.advantage.LevelActorWeights."""
    unifloral_weight = jnp.exp(adv * beta).clip(max=cap)
    shrunk = BC.postcap_level_actor_weights(adv, width, jnp.ones(jnp.shape(adv), bool), beta, cap, True)
    weights = jnp.where(adv > 0.0, shrunk.weights, unifloral_weight)
    return shrunk._replace(weights=jax.lax.stop_gradient(weights))


def _metrics(valid, fit, bw, refused, actor_ok, bca_after, bca_before):
    """BCA's per-step scalars (INTERFACES.md section 10, step 8)."""
    support = jnp.mean(bw.support_mask.astype(jnp.float32))
    return dict(
        inputs_valid=valid, scale_inputs_valid=fit.valid, dose_inputs_valid=bw.inputs_valid, scale_loss=fit.loss,
        unit=bca_after.unit, posterior_ready=bca_before.reference.ready, weight_mean=jnp.mean(bw.weights),
        weight_min=jnp.min(bw.weights), weight_max=jnp.max(bw.weights), actor_updated=actor_ok & valid,
        supported_fraction=support, refused_fraction=jnp.mean(refused.astype(jnp.float32)), support_fraction=support)


# ------------------------------------------------------------------------------------------------ band parts, refresh

def band_parts(args, apply_fns, agent_state, rows, noise_key=None):
    """The band's target y and critic value q at `rows` (a Unifloral Transition) from a stored state, float32 [n] each
    (plan 4.3, "Refresh, per learner"; choice 15): q = the minimum over the two heads of Polyak(stored target critic,
    stored critic, args.polyak_step_size), the target copy the next outer step's advantage reads (Unifloral's own
    expressions, iql.py:207-211 and 218-219); y = r + gamma (1 - d) V(s') with the stored V (iql.py:220-221). IQL's
    target has no randomness, so noise_key is ignored. Pure: runs eagerly (as refresh does) or under jit."""
    del noise_key
    _, q_apply_fn, value_apply_fn = apply_fns
    target_params = optax.incremental_update(
        agent_state.dual_q.params, agent_state.dual_q_target.params, args.polyak_step_size)
    q = q_apply_fn(target_params, rows.obs, rows.action).min(-1)
    next_v = value_apply_fn(agent_state.value.params, rows.next_obs)
    y = rows.reward + args.gamma * (1 - rows.done) * next_v
    return y, q


def refresh(args, apply_fns, agent_state, bca_state, bank, refresh_key, *, permission):
    """One refresh between scan chunks (plan 4.1; INTERFACES.md section 10): guard.require_bca on the bank; y and q by
    band_parts from the stored state (no noise key: IQL's target has no randomness); eta from the live scale network,
    which the refresh freezes with the live unit; bca_core.apply_refresh with the posterior key
    fold_in(refresh_key, 1347375956) (calibration/iql_reference.py:14, 193). Eager. Returns bca_core.Refreshed (an
    invalid refresh keeps the previous state; its record says so)."""
    guard.require_bca(permission, bank)
    y, q = band_parts(args, apply_fns, agent_state, bank)
    eta = bca_state.scale.apply_fn(bca_state.scale.params, bank.obs, bank.action)
    return BC.apply_refresh(bca_state, y, q, eta, BC.posterior_key(refresh_key, LEARNER),
                            inputs_valid=BC.tree_finite((agent_state, bank)))


def critic_value_parts(agent_state):
    """(dual_q, dual_q_target, value): what the alone, standard and IW runs of one seed share, since Unifloral's Q and
    V updates never read the actor (iql.py:206-242; plan sections 3.7 and 5.4); bit for bit on the processor, and on
    the card bit for bit between the standard and IW runs (module docstring). Hash it with guard.state_sha256."""
    return agent_state.dual_q, agent_state.dual_q_target, agent_state.value
