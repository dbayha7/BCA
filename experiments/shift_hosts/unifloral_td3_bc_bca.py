"""Unifloral's TD3+BC training step with BCA's additions, plus TD3+BC's band parts and refresh (plan sections 3.1,
3.5, 4.1 and 4.3; choices 13 and 16; PLAN_UNIFLORAL.md). Training environment only (/opt/bca-venv).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. The
repository's TD3+BC with BCA stays the CORL-derived port algorithms/td3_bc_bca.py, and Unifloral's
baselines/unifloral/algorithms/td3_bc.py (revision f2dc127, configs/unifloral.json) is imported unchanged through
unifloral_import and never edited. Its networks, create_train_state, AgentTrainState, Transition and Args are not
restated: callers take them from the imported module (unifloral_import.load("td3_bc"), args_for, make_transition,
init_like_training_block).

Restated, with the lines they restate:
  make_train_step  td3_bc.py:174-279, line for line. Every one of Unifloral's lines appears verbatim, at its own
                   indentation and in its own order, except the two in CHANGED_LINES: line 174, the signature, which
                   gains the keywords bca and test, and line 245, the actor loss, which gains the dose. Every other
                   line is BCA's and carries a "# BCA:" comment. With bca=None and test=None BCA's lines do nothing
                   (_check returns at once, and every other one sits under "if bca is not None" or "if test is not
                   None") and line 245 reads bc_loss.mean(), so the program traced is Unifloral's own (test a1).
                   tests/test_td3_bc.py checks the text against the vendored file.
  _own_target      td3_bc.py:191-208, Unifloral's per-row critic target _compute_target, verbatim, closed over a given
                   agent state: the band's target y in the fit (the state at the start of the step) and in the refresh
                   (the stored state).
  refresh          algorithms/td3_bc_bca.py:234-280, the port's refresh, on Unifloral's state (the band parts below,
                   then bca_core.apply_refresh).
  recompute_bank_scores
                   experiments/shift_iw/learners_iw.py:540-552 (_scores_from_state): a refresh's bank scores and
                   stream recomputed from the stored states, for stop rule S3.
  require_valid    algorithms/td3_bc_bca.py:348-351, the port's check of the per-step validity flags (stop rule S5).

BCA's additions inside the step (plan section 4.3, TD3+BC row; INTERFACES.md section 10):
  carry      (rng, agent_state, bca_state), a tuple; bca_state is bca_core.BCAState. Unifloral's own lines see
             (rng, agent_state) and split the key exactly as Unifloral does. BCA never splits or advances that key, so
             the three setups of a learner draw the same batches.
  fit        once per outer step, right after Unifloral's batch draw and before both critic updates (choice 13):
             y = r + 0.98 x min of the 2 target critics at the noisy target command, computed by _own_target with the
             state at the start of the step and BCA's own per-row keys split(fold_in(fit key, 1179210836), 256), the
             fit key being fold_in(PRNGKey(s91), t) at outer step t = bca_state.step + 1 (bca_core.fit_key,
             fit_noise_key, per_row_keys); q = min(Q1, Q2)(s, a) with the critics at the start of the step; then
             bca_core.fit_scale_core with the port's bootstrap fold 1128352850 (algorithms/td3_bc_bca.py:176).
  dose       from the frozen band at the recorded (s, a), no gradient: eta = the scale network at the reference's
             frozen parameters; each batch row's threshold R_b = bca_state.thresholds[batch_indices], read through
             Unifloral's own batch_indices; iw.per_row_dose gives 1 + 0.5 U / (U + u) with U = R_b u max(eta, 1e-6),
             exactly 1 before the first refresh, and for a refused row (R_b = +inf) 1 under standard BCA's rule or 1.5
             under IW's (bca_state.most_caution; the weighting test's own rule,
             experiments/shift_iw/learners_iw.py:655-683).
  loss       the dose m multiplies each row's cloning error in the actor loss (td3_bc.py:239, 245):
             -lambda x mean Q1 + mean_i[sg(m_i) x bc_i], lambda = 2.5 / (mean |Q1| + 1e-7) from Unifloral's own lines
             243-244. The critic losses are untouched, and the reported bc_loss stays Unifloral's undosed mean
             (line 246).
  validity   the port's per-step flag (algorithms/td3_bc_bca.py:312-322): the batch, the agent state before and after
             the step and Unifloral's loss dict all finite, the fit valid, the dose's inputs valid, the frozen reference
             valid (td3_bc_bca.py:286-288). A step that is not valid is discarded whole, random key included
             (td3_bc_bca.py:340-342); the runner stops on any false flag (stop rule S5; require_valid).
  outputs    Unifloral's loss dict plus, one scalar each per step: inputs_valid, scale_inputs_valid,
             dose_inputs_valid, scale_loss, unit (after the step), posterior_ready, dose_mean, dose_min, dose_max,
             refused_fraction, support_fraction. Test mode (bca_core.TestOptions; made-up arrays only) adds, per batch
             row, batch_indices, bca_y, bca_q, unifloral_y and unifloral_q (the first critic update's own targets and
             the minimum over the critics of its q_pred), dose (as used), width, eta, R_rows and refused, and the
             actor loss holding the dose and its gradient (dosed_loss, dosed_grad).

For the runner (INTERFACES.md section 10):
  step     make_train_step(args, *built.apply_fns, built.dataset, bca=bca_core.BCASetup("td3_bc", fit_key_base(s91),
           permission)), scanned exactly as Unifloral's step, carry (rng, agent_state, bca_state) with
           bca_state = bca_core.initial_state(bca_core.make_scale_state("td3_bc", built.obs_mean, built.obs_std,
           bca_core.scale_init_key(fit_key_base(s91), "td3_bc")), n_train, most_caution=False (standard) or True (IW)).
           The alone setup is unifloral_import.make_unifloral_step(built, args), carry (rng, agent_state).
  refresh  refresh(args, built.apply_fns, agent_state, bca_state, bank, bca_core.refresh_key(s39, t),
           permission=permission) after outer step t, between chunks (eager); bank is the bank rows' Transition
           (unifloral_import.make_transition). It returns bca_core.Refreshed; its scores and stream feed
           iw.threshold_table and stop rule S3, and for IW the runner then sets bca_state.thresholds to
           iw.row_thresholds(...). recompute_bank_scores gives the same scores bit for bit from the stored states
           (same compiled program); a separately compiled recomputation agrees only to float32 rounding.
  checks   require_valid(chunk outputs) after every chunk (S5); every dose_min and dose_max lies in [1, 1.5].
  card     Before the first refresh the BCA step keeps Unifloral's parameters, optimizer states and key bit for bit
           with the alone setup on both devices (test a2). On the card the logged actor_loss and bc_loss may differ by
           a rounding unit (most likely because, with BCA present, the dosed mean and the logged mean are two
           reductions where Unifloral has one, and the card compiles them together), while the parameters and
           optimizer states the gradients drive stay bit for bit (test a2's record).
"""

import dataclasses
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.shift_hosts import bca_core as BC
from experiments.shift_hosts import guard

LEARNER = "td3_bc"
SOURCE = "baselines/unifloral/algorithms/td3_bc.py"
STEP_LINES = (174, 279)             # make_train_step in the vendored file
TARGET_LINES = (191, 208)           # its per-row target, _compute_target
CHANGED_LINES = {                   # Unifloral's lines that the restated step changes (each marked "# BCA:")
    174: "the signature gains the keyword arguments bca and test",
    245: "the actor loss: (bc_loss if bca is None else band.m * bc_loss).mean() in place of bc_loss.mean()",
}
FLAGS = ("inputs_valid", "scale_inputs_valid", "dose_inputs_valid")


def make_train_step(args, actor_apply_fn, q_apply_fn, dataset, *, bca=None, test=None):  # BCA: line 174, + bca, test
    """Make JIT-compatible agent train step."""
    IW = _check(bca, test, dataset)  # BCA: the entry check (guard.require_bca on dataset); iw.py when BCA is present

    def _train_step(runner_state, _):
        if bca is not None:  # BCA: the carry is (rng, agent_state, bca_state); Unifloral's lines see the first two
            runner_state, bca_state = tuple(runner_state[:2]), runner_state[2]  # BCA:
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)
        if bca is not None:  # BCA: the band fit and the dose, once per outer step before both critic updates (_band)
            band = _band(IW, args, actor_apply_fn, q_apply_fn, bca, test, bca_state, agent_state, rng, batch,  # BCA:
                         batch_indices, len(dataset.obs))  # BCA:

        # --- Update critics ---
        def _update_critics(runner_state, _):
            rng, agent_state = runner_state

            def _compute_target(rng, transition):
                next_obs = transition.next_obs

                # --- Sample noised action ---
                next_pi = actor_apply_fn(agent_state.actor_target.params, next_obs)
                rng, rng_action, rng_noise = jax.random.split(rng, 3)
                action = next_pi.sample(seed=rng_action)
                noise = jax.random.normal(rng_noise, shape=action.shape)
                noise *= args.policy_noise
                noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
                action = jnp.clip(action + noise, -1, 1)

                # --- Compute targets ---
                target_q = q_apply_fn(
                    agent_state.dual_q_target.params, next_obs, action
                )
                next_q_value = (1.0 - transition.done) * jnp.min(target_q)
                return transition.reward + args.gamma * next_q_value

            rng, rng_targets = jax.random.split(rng)
            rng_targets = jax.random.split(rng_targets, args.batch_size)
            targets = jax.vmap(_compute_target)(rng_targets, batch)

            # --- Compute critic loss ---
            @jax.value_and_grad
            def _q_loss_fn(params):
                q_pred = q_apply_fn(params, batch.obs, batch.action)
                return jnp.square(q_pred - jnp.expand_dims(targets, axis=-1)).mean()

            q_loss, q_grad = _q_loss_fn(agent_state.dual_q.params)
            if test is not None:  # BCA: test mode: this update's own targets, and q_pred at the params _q_loss_fn got
                seen = (targets, q_apply_fn(agent_state.dual_q.params, batch.obs, batch.action).min(axis=-1))  # BCA:
            updated_q_state = agent_state.dual_q.apply_gradients(grads=q_grad)
            agent_state = agent_state._replace(dual_q=updated_q_state)
            if test is not None:  # BCA: test mode: pass them out through the critic scan
                return (rng, agent_state), (q_loss, seen)  # BCA:
            return (rng, agent_state), q_loss

        # --- Iterate critic update ---
        (rng, agent_state), q_loss = jax.lax.scan(
            _update_critics,
            (rng, agent_state),
            None,
            length=args.num_critic_updates_per_step,
        )
        if test is not None:  # BCA: test mode: split the critic scan's outputs
            q_loss, seen = q_loss  # BCA:

        # --- Update actor ---
        def _actor_loss_function(params):
            def _transition_loss(transition):
                pi = actor_apply_fn(params, transition.obs)
                pi_action = pi.sample(seed=None)
                q = q_apply_fn(agent_state.dual_q.params, transition.obs, pi_action)
                bc_loss = jnp.square(pi_action - transition.action).mean()
                return q[0], bc_loss

            q, bc_loss = jax.vmap(_transition_loss)(batch)
            lambda_ = args.td3_alpha / (jnp.abs(q).mean() + 1e-7)
            lambda_ = jax.lax.stop_gradient(lambda_)
            actor_loss = -lambda_ * q.mean() + (bc_loss if bca is None else band.m * bc_loss).mean()  # BCA: line 245
            return actor_loss.mean(), (q.mean(), lambda_.mean(), bc_loss.mean())

        loss_fn = jax.value_and_grad(_actor_loss_function, has_aux=True)
        (actor_loss, (q_mean, lambda_, bc_loss)), actor_grad = loss_fn(
            agent_state.actor.params
        )
        agent_state = agent_state._replace(
            actor=agent_state.actor.apply_gradients(grads=actor_grad)
        )

        # --- Update target networks ---
        def _update_target(state, target_state):
            new_target_params = optax.incremental_update(
                state.params, target_state.params, args.polyak_step_size
            )
            return target_state.replace(
                step=target_state.step + 1, params=new_target_params
            )

        agent_state = agent_state._replace(
            actor_target=_update_target(agent_state.actor, agent_state.actor_target),
            dual_q_target=_update_target(agent_state.dual_q, agent_state.dual_q_target),
        )

        loss = {
            "actor_loss": actor_loss,
            "q_loss": q_loss.mean(),
            "q_mean": q_mean,
            "lambda": lambda_,
            "bc_loss": bc_loss,
        }
        if bca is not None:  # BCA: the per-step validity flag, BCA's new carry and outputs (_finish)
            return _finish(runner_state, bca_state, rng, agent_state, loss, batch, band, test,  # BCA:
                           seen if test is not None else None, actor_grad)  # BCA:
        return (rng, agent_state), loss

    return _train_step


# ------------------------------------------------------------------------------------------------ BCA's step parts

_Band = namedtuple("_Band", "t batch_indices y q fit eta R dose refused m")


def _check(bca, test, dataset):
    """BCA's entry check, outside jit (INTERFACES.md sections 8 and 10). BCA absent: no test options, nothing else to
    check (the learner alone is Unifloral's own step). BCA present: this learner's setup, and
    guard.require_bca(permission, dataset), so BCA's code runs only on made-up arrays registered with the guard or with
    the user's terminal approval on real data (gate.py). Test options only on made-up arrays and only TD3+BC's
    (dose_override with one dose per training row; IQL's width_override is refused). Returns the iw module (the per-row
    dose) when BCA is present, else None."""
    if bca is None:
        if test is not None:
            raise ValueError("test options need BCA present (bca=bca_core.BCASetup(...))")
        return None
    if not isinstance(bca, BC.BCASetup) or bca.learner != LEARNER:
        raise ValueError(f"this restated step is TD3+BC's: bca must be bca_core.BCASetup({LEARNER!r}, ...)")
    guard.require_bca(bca.permission, dataset)
    if test is not None:
        if not isinstance(test, BC.TestOptions):
            raise ValueError("test must be a bca_core.TestOptions")
        if bca.permission is not guard.MADE_UP:
            raise guard.ApprovalError("test options run on made-up arrays only")
        if test.width_override is not None:
            raise ValueError("width_override is IQL's; TD3+BC takes dose_override")
        if test.dose_override is not None and np.shape(test.dose_override) != (len(dataset.obs),):
            raise ValueError("dose_override must hold one dose per training row")
    from experiments.shift_hosts import iw as IW

    return IW


def _band(IW, args, actor_apply_fn, q_apply_fn, bca, test, bca_state, agent_state, rng, batch, batch_indices, n_train):
    """BCA's part of the step before the critic updates (plan 4.3, TD3+BC row; choice 13; INTERFACES.md section 10,
    items 2-4), traced inside the step. agent_state is the state at the start of the step; rng is Unifloral's key after
    its batch draw, read only in test mode with unifloral_keys and never split into the carry. n_train: the training
    rows; bca_state.thresholds must hold one float32 threshold per row (checked at trace time, as ReBRAC's step does:
    JAX clamps an out-of-range gather index silently, so a wrong-length array would give rows wrong thresholds).
    Returns _Band."""
    if tuple(bca_state.thresholds.shape) != (n_train,) or bca_state.thresholds.dtype != jnp.float32:
        raise ValueError(f"bca_state.thresholds must be float32 with one threshold per training row ({n_train})")
    t = bca_state.step + 1                                                  # this outer step: 1, 2, ...
    k = BC.fit_key(bca.fit_key_base, t)                                     # fold_in(PRNGKey(s91), t)
    if test is not None and test.unifloral_keys:                            # test (a4): Unifloral's own per-row keys
        _, rng_targets = jax.random.split(rng)                              # of its first critic update,
        keys = jax.random.split(rng_targets, args.batch_size)               # td3_bc.py:210-211
    else:                                                                   # BCA's own draw:
        keys = BC.per_row_keys(BC.fit_noise_key(k, LEARNER), args.batch_size)   # split(fold_in(k, 1179210836), 256)
    y, q = _band_values(args, actor_apply_fn, q_apply_fn, agent_state, batch, keys)
    fit = BC.fit_scale_core(bca_state.scale, bca_state.unit, batch.obs, batch.action, y, q, k,
                            bootstrap_fold=BC.FOLDS[LEARNER].bootstrap, settings=bca.settings)
    ref = bca_state.reference                                               # the frozen band of the last refresh
    eta = bca_state.scale.apply_fn(ref.cal_params, batch.obs, batch.action)
    R = bca_state.thresholds[batch_indices]                                 # each batch row's own threshold
    dose, refused = IW.per_row_dose(ref, eta, R, bca.settings.blend, bca_state.most_caution)
    m = dose.dose
    if test is not None and test.dose_override is not None:                # test (a3): a fixed made-up dose per row
        m = jnp.asarray(test.dose_override, jnp.float32)[batch_indices]
    return _Band(t, batch_indices, y, q, fit, eta, R, dose, refused, jax.lax.stop_gradient(m))


def _finish(start, bca_state, rng, agent_state, loss, batch, band, test, seen, actor_grad):
    """BCA's end of the step (INTERFACES.md section 10, items 7-8). The port's per-step validity flag
    (algorithms/td3_bc_bca.py:312-322, with bc_readout's reference check, lines 286-288): the batch, the agent state
    before and after and Unifloral's loss dict finite, the scale fit valid, the dose's inputs valid, the frozen
    reference valid. The new carry (BCA's state with the fitted scale network and unit, step t) is kept only when the
    step is valid; otherwise the old carry is kept whole, random key included (td3_bc_bca.py:340-342). Outputs:
    Unifloral's loss dict, BCA's per-step metrics and, in test mode, the per-row test outputs."""
    rng0, agent_state0 = start
    fit, dose, m = band.fit, band.dose, band.m
    valid = (BC.tree_finite((batch, agent_state0, agent_state, loss)) & fit.valid & dose.inputs_valid
             & BC.reference_valid(bca_state.reference))
    old = (rng0, agent_state0, bca_state)
    new = (rng, agent_state, bca_state._replace(scale=fit.scale, unit=fit.unit, step=band.t))
    carry = jax.lax.cond(valid, lambda _: new, lambda _: old, None)
    out = dict(loss)
    out.update(inputs_valid=valid, scale_inputs_valid=fit.valid, dose_inputs_valid=dose.inputs_valid,
               scale_loss=fit.loss, unit=carry[2].unit, posterior_ready=bca_state.reference.ready,
               dose_mean=jnp.mean(m), dose_min=jnp.min(m), dose_max=jnp.max(m),
               refused_fraction=jnp.mean(band.refused.astype(jnp.float32)),
               support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)))
    if test is not None:
        out.update(batch_indices=band.batch_indices, bca_y=band.y, bca_q=band.q, unifloral_y=seen[0][0],
                   unifloral_q=seen[1][0], dose=m, width=dose.width, eta=band.eta, R_rows=band.R,
                   refused=band.refused, dosed_loss=loss["actor_loss"], dosed_grad=actor_grad)
    return carry, out


# ------------------------------------------------------------------------------------------------ the band's parts

def _own_target(args, actor_apply_fn, q_apply_fn, agent_state):
    """Unifloral's per-row critic target, td3_bc.py:191-208 verbatim, closed over `agent_state` (its actor_target and
    dual_q_target): y = r + gamma (1 - d) min of the 2 target critics at the noisy target command clip(pi_target(s') +
    clip(0.2 N(0, 1), +-0.5), +-1), from one key per row."""

    def _compute_target(rng, transition):
        next_obs = transition.next_obs

        # --- Sample noised action ---
        next_pi = actor_apply_fn(agent_state.actor_target.params, next_obs)
        rng, rng_action, rng_noise = jax.random.split(rng, 3)
        action = next_pi.sample(seed=rng_action)
        noise = jax.random.normal(rng_noise, shape=action.shape)
        noise *= args.policy_noise
        noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
        action = jnp.clip(action + noise, -1, 1)

        # --- Compute targets ---
        target_q = q_apply_fn(
            agent_state.dual_q_target.params, next_obs, action
        )
        next_q_value = (1.0 - transition.done) * jnp.min(target_q)
        return transition.reward + args.gamma * next_q_value

    return _compute_target


def _band_values(args, actor_apply_fn, q_apply_fn, agent_state, rows, keys):
    """The band's target y and critic value q at `rows` (a Unifloral Transition) from `agent_state`, with one key per
    row: y by Unifloral's own per-row target, vmapped over the rows as Unifloral does (td3_bc.py:210-212); q = min(Q1,
    Q2)(s, a) of the online critics, the critic loss's q_pred (td3_bc.py:217) reduced over the two critics."""
    y = jax.vmap(_own_target(args, actor_apply_fn, q_apply_fn, agent_state))(keys, rows)
    q = q_apply_fn(agent_state.dual_q.params, rows.obs, rows.action).min(axis=-1)
    return y, q


def band_parts(args, apply_fns, agent_state, rows, noise_key):
    """TD3+BC's band parts (y, q), float32 [n] each, at `rows` (a Unifloral Transition) from a stored state
    (INTERFACES.md section 10): y = r + 0.98 min of the 2 target critics at the noisy target command, by Unifloral's
    own per-row target with keys bca_core.per_row_keys(noise_key, n); q = min(Q1, Q2)(s, a). apply_fns is
    Built.apply_fns, (actor, q). Traceable; used by refresh() and band_readout.py."""
    actor_apply_fn, q_apply_fn = apply_fns
    return _band_values(args, actor_apply_fn, q_apply_fn, agent_state, rows,
                        BC.per_row_keys(noise_key, rows.obs.shape[0]))


_PROGRAMS = {}


def _refresh_program(args, apply_fns):
    """band_parts and the live scale network's eta at the bank rows as one compiled program, kept per (Args values,
    apply functions) so the 18 refreshes of a run compile once."""

    def run(agent_state, scale, rows, noise_key):
        y, q = band_parts(args, apply_fns, agent_state, rows, noise_key)
        return y, q, scale.apply_fn(scale.params, rows.obs, rows.action)

    try:
        key = (dataclasses.astuple(args), tuple(apply_fns))
        hash(key)
    except TypeError:
        return jax.jit(run)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = jax.jit(run)
    return _PROGRAMS[key]


def refresh(args, apply_fns, agent_state, bca_state, bank, refresh_key, *, permission, settings=BC.SETTINGS):
    """BCA's refresh after an outer step, between scan chunks (eager; plan section 4.1; INTERFACES.md section 10),
    restating the port's refresh (algorithms/td3_bc_bca.py:234-280) on Unifloral's state: guard.require_bca on the bank
    rows; y and q = band_parts at the bank (a Unifloral Transition) from the stored agent state, with the target noise
    of bca_core.refresh_noise_key(refresh_key, "td3_bc") = fold_in(refresh key, 1213156420) (td3_bc_bca.py:249); eta =
    the live scale network, frozen now; then bca_core.apply_refresh with the posterior key fold_in(refresh key,
    1347375956) (td3_bc_bca.py:268): the threshold R on the bank scores |y - q| / (max(eta, 1e-6) u), every row's
    threshold set to R, and the previous state kept when the refresh is not valid. Returns bca_core.Refreshed (state,
    record, scores, stream)."""
    guard.require_bca(permission, bank)
    y, q, eta = _refresh_program(args, apply_fns)(agent_state, bca_state.scale, bank,
                                                  BC.refresh_noise_key(refresh_key, LEARNER))
    return BC.apply_refresh(bca_state, y, q, eta, BC.posterior_key(refresh_key, LEARNER),
                            inputs_valid=bool(BC.tree_finite((agent_state, bank))), settings=settings)


def recompute_bank_scores(args, apply_fns, agent_state, bca_state, bank, refresh_key, *, permission):
    """A refresh's bank scores and posterior stream recomputed from the stored states (the recomputation of plan 4.2
    and stop rule S3; the weighting test's _scores_from_state, experiments/shift_iw/learners_iw.py:540-552): y and q at
    the bank from the stored agent state with the refresh key's target noise, eta from the frozen reference's scale
    parameters, and the scores exactly as reference.freeze_reference computes them (bca_core.bank_scores), with the
    reference's unit. It runs the very compiled program refresh() runs (_refresh_program), so for the agent state and
    refresh key of a refresh and the BCA state that refresh produced, it gives that refresh's scores bit for bit; the
    same arithmetic compiled as another program differs at float32 rounding (tests/test_td3_bc.py, RefreshTest).
    An entry point: guard.require_bca on the bank. Returns (scores float64 [n], stream uint32 words)."""
    guard.require_bca(permission, bank)
    ref = bca_state.reference
    frozen = bca_state.scale.replace(params=ref.cal_params)
    y, q, eta = _refresh_program(args, apply_fns)(agent_state, frozen, bank, BC.refresh_noise_key(refresh_key, LEARNER))
    return BC.bank_scores(eta, ref.residual_scale, jnp.asarray(y) - jnp.asarray(q)), \
        BC.posterior_stream(BC.posterior_key(refresh_key, LEARNER))


def require_valid(metrics):
    """Stop rule S5 on a chunk's outputs, restated from the port's require_valid (algorithms/td3_bc_bca.py:348-351)
    with this step's flag names: raises FloatingPointError naming the first false flag and the outer step (counted
    from 1 within the chunk) where it is false."""
    for name in FLAGS:
        if name in metrics:
            flags = np.asarray(metrics[name]).reshape(-1).astype(bool)
            if not np.all(flags):
                first = int(np.flatnonzero(~flags)[0]) + 1
                raise FloatingPointError(f"Unifloral TD3+BC with BCA: invalid step: {name} is false at step {first} "
                                         "of the chunk")
