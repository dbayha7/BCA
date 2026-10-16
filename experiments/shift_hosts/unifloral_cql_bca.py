"""Unifloral's CQL training step with BCA's additions, plus CQL's band parts and refresh (plan sections 3.1, 3.6, 4.1,
4.3 and choices 14 and 16; INTERFACES.md section 10).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods (the
repository's CQL+BCA stays algorithms/cql_bca.py on the CORL-derived port) and edits nothing under baselines/unifloral/.
Unifloral's networks, create_train_state, AgentTrainState, Transition and Args stay imported, unchanged
(unifloral_import.load("cql"); unifloral() below returns that module). Training environment only (/opt/bca-venv).

What is restated, from baselines/unifloral/algorithms/cql.py at commit f2dc127:
  make_train_step  cql.py:192-325, line for line. Every line of _train_step (cql.py:195-325) is Unifloral's, verbatim
                   and in order; BCA's additions are the lines marked "# BCA:" (test_cql's
                   test_a0_restated_lines_are_unifloral checks this against the vendored file). Two of Unifloral's
                   lines sit one level deeper, inside a "# BCA:" branch, unchanged: cql.py:304 (q_diff, the line the dose
                   line replaces when the dose is active) and cql.py:310 (the critic loss call, which test mode makes
                   with has_aux on the same body). With bca=None and test=None the step is Unifloral's step (test a1).
  band_core        cql.py:223 (alpha = exp(log alpha)), 261-270 (_sample_next_v, its lines verbatim), 272-275 (the
                   target) and 293 (q_pred, then the minimum over the ten critics), for a given state, rows and per-row
                   keys: CQL's band target y and critic value q. The step's fit, the refresh and band_readout.py all use
                   it (band_parts is its compiled form for a stored state).

BCA's additions inside the step (plan section 4.3, CQL row; choice 14):
  carry        (rng, agent_state, bca_state), a tuple; bca_state is bca_core.BCAState. Unifloral's key is split exactly as
               Unifloral splits it, so BCA never changes which batches or commands are drawn.
  band fit     just before Unifloral's critic loss (after the temperature, actor and target updates and Unifloral's own
               target and action draws, cql.py:205-288): t = bca_state.step + 1, k = fit_key(PRNGKey(s91), t);
               (y, q) = band_core on the step's own state at that point (the just-updated actor and temperature, the
               just-averaged target critics, the online critics not yet updated) with per-row keys
               split(fold_in(k, 1179210836), B): y = r + 0.98 [min of the 10 target critics at a' - alpha log pi(a'|s')],
               q = the minimum over the ten online critics at (s, a) (cql.py:293, 310-312); bca_core.fit_scale_core
               with the bootstrap fold 1128352850 (algorithms/cql_bca.py:182, 192). band_core reads a copy of the
               state, batch and keys passed through jax.lax.optimization_barrier: the same values, but the card's
               compiler can no longer merge BCA's reads into Unifloral's critic update, which otherwise changes that
               update by one rounding step on the card (diagnosed in the CQL stage-0 report; the processor is
               unaffected). Test (a4) shows band_core's y and q equal Unifloral's own target and q_pred bit for bit.
  dose         from the frozen band at the recorded (s, a), no gradient: eta = scale network(reference parameters),
               R_b = bca_state.thresholds[batch_indices] (Unifloral's own batch_indices, cql.py:200-203), then
               iw.per_row_dose(reference, eta, R_b, 0.5, bca_state.most_caution): before the first refresh every dose is
               exactly 1; a refused row (R = +inf) gets 1 (standard BCA's rule) or 1.5 (IW's rule, a new rule for CQL,
               choice 16). Then the port's guard (algorithms/cql.py:304-306): a negative or non-finite dose anywhere in
               the batch turns every dose into NaN, and the step is not valid.
  where        cql.py:304, the conservative gap: q_diff = (expand_dims(m, 1) * (expand_dims(q_ood, 1) - q_pred)).mean(),
               so row i's gap is scaled by m_i for all ten critics at once; the soft maximum per row over 40 values
               (cql.py:297-303) and the weight 10 (cql.py:305) are unchanged. On the processor m = 1 gives Unifloral's
               bits (test a2).
  dose_active  make_train_step(..., dose_active=False) builds the step for outer steps before the first refresh, when
               the band is not ready and every dose is exactly 1 (plan 4.1): the conservative gap is Unifloral's own
               line cql.py:304, and a step that finds a ready band is not valid. Plan 4.3 says Unifloral's expressions
               are reproduced exactly at dose 1; on the card the multiply by an exact 1 still changes how the critic's
               gradient is compiled, so only this form keeps the BCA setups equal to the learner alone bit for bit on
               the card until the first refresh (test a2; stop rule S10). Everything else (the fit, the dose read, the
               metrics) is the same in both forms. The default, dose_active=True, is INTERFACES.md's one program.
  validity     the port's per-step flag (algorithms/cql_bca.py:300-307): fit valid & dose inputs valid & the dose guard &
               reference_valid(frozen reference) & tree_finite((new agent state, Unifloral's loss dict)), and with
               dose_active=False also "the band is not ready". When it is false the old carry is kept whole,
               Unifloral's key included (jax.lax.cond, td3_bc_bca.py:340-342); the runner stops on it (stop rule S5).
  metrics      Unifloral's loss dict (cql.py:314-322) plus, per step: inputs_valid, scale_inputs_valid,
               dose_inputs_valid, dose_guard_ok, scale_loss, unit (after the step), posterior_ready, dose_mean, dose_min,
               dose_max (of the dose before the guard), refused_fraction, support_fraction.
  test mode    (test=bca_core.TestOptions(...), made-up arrays only) adds batch_indices, bca_y, bca_q, unifloral_y
               (cql.py:275's target), unifloral_q (the minimum over the ten critics of _q_loss_fn's own q_pred,
               cql.py:293, read through has_aux on Unifloral's own loss body), dose (as used), width, eta, refused,
               dosed_loss (the critic loss), dosed_grad (its gradient) and Unifloral's draws for the conservative term
               (cql_random_actions, pi_actions, pi_next_actions). unifloral_keys=True feeds the fit Unifloral's own
               per-row keys (cql.py:272-273), so bca_y equals unifloral_y (test a4); dose_override replaces the band's
               dose by a fixed vector over training rows read through batch_indices (test a3).

The refresh (eager, between chunks; plan section 4.1, CQL bullet of 4.3): the guard on the bank, then y and q from the
stored actor, temperature and target critics and the stored critics (band_parts) with per-row keys
split(fold_in(refresh key, 1213156420), n), eta from the live scale network, and bca_core.apply_refresh with the
posterior key fold_in(refresh key, 1347375956) (algorithms/cql_bca.py:244, 259).

For the runner (INTERFACES.md section 10):
  step     make_train_step(args, actor_apply_fn, q_apply_fn, alpha_apply_fn, dataset, bca=BC.BCASetup("cql",
           BC.fit_key_base(s91), permission), dose_active=...); scan it 2,500 steps per chunk exactly as Unifloral's
           step, carry (rng, agent_state, bca_state) with bca_state = BC.initial_state(BC.make_scale_state("cql",
           built.obs_mean, built.obs_std, BC.scale_init_key(BC.fit_key_base(s91), "cql")), n_train,
           most_caution=False (standard) or True (IW)). The evaluation split after each chunk acts on the carry's rng
           only. For bit-for-bit pairing on the card: dose_active=False for the chunks up to outer step 10,000, then
           dose_active=True from the first refresh on (two compiled programs per run, shared by standard and IW).
  refresh  refresh(args, built.apply_fns, agent_state, bca_state, bank, BC.refresh_key(s39, t), permission=...) after
           outer steps 10,000, 15,000, ..., 95,000; bank is the prepared bank as Unifloral's Transition
           (unifloral_import.make_transition("cql", rows)); for IW the runner then passes the result to iw.iw_refresh,
           which checks stop rule S3 and sets each training row's threshold.
  compiles The first chunk calls compile more than once (Python-integer step counters and the weak-typed log
           temperature become arrays after the first call; environment report section 3.3, item 4): measured in
           test_cql (Unifloral's own step 3 compiles, this step 2). A fixed cost per run, not a change of any value.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax

from experiments.shift_hosts import bca_core as BC
from experiments.shift_hosts import guard

LEARNER = "cql"
UNIFLORAL_FILE = "baselines/unifloral/algorithms/cql.py"
UNIFLORAL_STEP_LINES = (195, 325)          # _train_step through `return _train_step`, restated verbatim
METRICS = ("inputs_valid", "scale_inputs_valid", "dose_inputs_valid", "dose_guard_ok", "scale_loss", "unit",
           "posterior_ready", "dose_mean", "dose_min", "dose_max", "refused_fraction", "support_fraction")
TEST_OUTPUTS = ("batch_indices", "bca_y", "bca_q", "unifloral_y", "unifloral_q", "dose", "width", "eta", "refused",
                "dosed_loss", "dosed_grad", "cql_random_actions", "pi_actions", "pi_next_actions")


def unifloral():
    """Unifloral's imported cql.py (networks, create_train_state, AgentTrainState, Transition, Args, make_train_step).
    Call unifloral_import.prepare() first."""
    from experiments.shift_hosts import unifloral_import as UI

    return UI.load(LEARNER)


def make_train_step(args, actor_apply_fn, q_apply_fn, alpha_apply_fn, dataset, *, bca=None, test=None,
                    dose_active=True):
    """Make JIT-compatible agent train step.

    Unifloral's make_train_step (cql.py:192-325) restated: with bca=None (and test=None) it is Unifloral's step, carry
    (rng, agent_state); with bca=bca_core.BCASetup("cql", ...) BCA's additions run (module docstring), carry
    (rng, agent_state, bca_state). dose_active=False: the step for outer steps before the first refresh (Unifloral's
    own gap line; not valid with a ready band). test=bca_core.TestOptions(...) (made-up arrays only) adds the test
    outputs."""
    if bca is None and (test is not None or not dose_active):                                   # BCA:
        raise ValueError("test options and dose_active need BCA present")                       # BCA:
    if test is not None and test.dose_override is not None and not dose_active:                 # BCA:
        raise ValueError("a dose override needs the dose-active step")                          # BCA:
    if bca is not None:                                                                         # BCA:
        if bca.learner != LEARNER:                                                              # BCA:
            raise ValueError(f"this restated step is CQL's, not {bca.learner!r}'s")              # BCA:
        guard.require_bca(bca.permission, dataset)                                              # BCA: the entry check
        if test is not None and bca.permission is not guard.MADE_UP:                            # BCA:
            raise guard.ApprovalError("test options run on made-up arrays only")                # BCA:
        from experiments.shift_hosts import iw as IW                                            # BCA: per-row dose
        settings = bca.settings                                                                 # BCA:
        fold = BC.FOLDS[LEARNER]                                                                # BCA:
        override = None                                                                         # BCA:
        if test is not None and test.dose_override is not None:                                 # BCA: test (a3)
            override = jnp.asarray(test.dose_override, jnp.float32)                             # BCA:
            if override.shape != (len(dataset.obs),):                                           # BCA:
                raise ValueError("dose_override must hold one dose per training row")           # BCA:

    def _train_step(runner_state, _):
        if bca is not None:                                                                     # BCA: carry
            start_state = tuple(runner_state)                                                   # BCA: kept whole if not valid
            runner_state, bca_state = start_state[:2], start_state[2]                           # BCA:
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        # --- Update alpha ---
        @jax.value_and_grad
        def _alpha_loss_fn(params, rng):
            def _compute_entropy(rng, transition):
                pi = actor_apply_fn(agent_state.actor.params, transition.obs)
                _, log_pi = pi.sample_and_log_prob(seed=rng)
                return -log_pi.sum()

            log_alpha = alpha_apply_fn(params)
            rng = jax.random.split(rng, args.batch_size)
            entropy = jax.vmap(_compute_entropy)(rng, batch).mean()
            target_entropy = -batch.action.shape[-1]
            return log_alpha * (entropy - target_entropy)

        rng, rng_alpha = jax.random.split(rng)
        alpha_loss, alpha_grad = _alpha_loss_fn(agent_state.alpha.params, rng_alpha)
        updated_alpha = agent_state.alpha.apply_gradients(grads=alpha_grad)
        agent_state = agent_state._replace(alpha=updated_alpha)
        alpha = jnp.exp(alpha_apply_fn(agent_state.alpha.params))

        # --- Update actor ---
        @partial(jax.value_and_grad, has_aux=True)
        def _actor_loss_function(params, rng):
            def _compute_loss(rng, transition):
                pi = actor_apply_fn(params, transition.obs)
                sampled_action, log_pi = pi.sample_and_log_prob(seed=rng)
                log_pi = log_pi.sum()
                q_values = q_apply_fn(
                    agent_state.vec_q.params, transition.obs, sampled_action
                )
                q_min = jnp.min(q_values)
                return -q_min + alpha * log_pi, -log_pi, q_min, q_values.std()

            rng = jax.random.split(rng, args.batch_size)
            loss, entropy, q_min, q_std = jax.vmap(_compute_loss)(rng, batch)
            return loss.mean(), (entropy.mean(), q_min.mean(), q_std.mean())

        rng, rng_actor = jax.random.split(rng)
        (actor_loss, (entropy, q_min, q_std)), actor_grad = _actor_loss_function(
            agent_state.actor.params, rng_actor
        )
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        # --- Update Q target network ---
        updated_q_target_params = optax.incremental_update(
            agent_state.vec_q.params,
            agent_state.vec_q_target.params,
            args.polyak_step_size,
        )
        updated_q_target = agent_state.vec_q_target.replace(
            step=agent_state.vec_q_target.step + 1, params=updated_q_target_params
        )
        agent_state = agent_state._replace(vec_q_target=updated_q_target)

        # --- Compute targets ---
        def _sample_next_v(rng, transition):
            next_pi = actor_apply_fn(agent_state.actor.params, transition.next_obs)
            # Note: Important to use sample_and_log_prob here for numerical stability
            # See https://github.com/deepmind/distrax/issues/7
            next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng)
            # Minimum of the target Q-values
            next_q = q_apply_fn(
                agent_state.vec_q_target.params, transition.next_obs, next_action
            )
            return next_q.min(-1) - alpha * log_next_pi.sum(-1)

        rng, rng_next_v = jax.random.split(rng)
        rng_next_v = jax.random.split(rng_next_v, args.batch_size)
        next_v_target = jax.vmap(_sample_next_v)(rng_next_v, batch)
        target = batch.reward + args.gamma * (1 - batch.done) * next_v_target

        # --- Sample actions for CQL ---
        def _sample_actions(rng, obs):
            pi = actor_apply_fn(agent_state.actor.params, obs)
            return pi.sample(seed=rng)

        rng, rng_pi, rng_next = jax.random.split(rng, 3)
        pi_actions = _sample_actions(rng_pi, batch.obs)
        pi_next_actions = _sample_actions(rng_next, batch.next_obs)
        rng, rng_random = jax.random.split(rng)
        cql_random_actions = jax.random.uniform(
            rng_random, shape=batch.action.shape, minval=-1.0, maxval=1.0
        )

        if bca is not None:                                                                     # BCA: band fit (choice 14)
            if (tuple(bca_state.thresholds.shape) != (len(dataset.obs),)                        # BCA: trace-time check,
                    or bca_state.thresholds.dtype != jnp.float32):                              # BCA: as ReBRAC's step:
                raise ValueError("bca_state.thresholds must be float32 with one threshold per "  # BCA: one per training
                                 "training row")                                                # BCA: row
            t = bca_state.step + 1                                                              # BCA: outer step 1, 2, ...
            k = BC.fit_key(bca.fit_key_base, t)                                                 # BCA: fold_in(PRNGKey(s91), t)
            if test is not None and test.unifloral_keys:                                        # BCA: test (a4) only
                fit_keys = rng_next_v                                                           # BCA: Unifloral's own keys
            else:                                                                               # BCA:
                fit_keys = BC.per_row_keys(BC.fit_noise_key(k, LEARNER), args.batch_size)       # BCA: BCA's own draw
            own_state, own_batch, own_keys = jax.lax.optimization_barrier(                      # BCA: compile hint only:
                (agent_state, batch, fit_keys))                                                 # BCA: same values (module doc)
            bca_y, bca_q = band_core(args.gamma, actor_apply_fn, q_apply_fn, alpha_apply_fn,    # BCA: y and q, cql.py:223,
                                     own_state, own_batch, own_keys)                            # BCA: 261-275, 293
            fit = BC.fit_scale_core(bca_state.scale, bca_state.unit, own_batch.obs, own_batch.action,  # BCA: the fit
                                    bca_y, bca_q, k, bootstrap_fold=fold.bootstrap, settings=settings)  # BCA:
            eta = bca_state.scale.apply_fn(bca_state.reference.cal_params, own_batch.obs, own_batch.action)  # BCA:
            dose, refused = IW.per_row_dose(bca_state.reference, eta, bca_state.thresholds[batch_indices],  # BCA:
                                            settings.blend, bca_state.most_caution)             # BCA: R_b per row
            m = dose.dose if override is None else override[batch_indices]                      # BCA: test (a3) override
            dose_guard_ok = jnp.all(jnp.isfinite(m) & (m >= 0))                                 # BCA: port's guard,
            m_used = jax.lax.stop_gradient(jnp.where(dose_guard_ok, m, jnp.nan))                # BCA: algorithms/cql.py:304-306

        # --- Update critics ---
        @jax.value_and_grad
        def _q_loss_fn(params):
            q_pred = q_apply_fn(params, batch.obs, batch.action)
            critic_loss = jnp.square((q_pred - jnp.expand_dims(target, -1)))
            critic_loss = critic_loss.sum(-1).mean()

            rand_q = q_apply_fn(params, batch.obs, cql_random_actions)
            pi_q = q_apply_fn(params, batch.obs, pi_actions)
            # Note: Source implementation erroneously uses current obs in next_pi_q
            next_pi_q = q_apply_fn(params, batch.next_obs, pi_next_actions)
            all_qs = jnp.concatenate([rand_q, pi_q, next_pi_q, q_pred], axis=1)
            q_ood = jax.scipy.special.logsumexp(all_qs / args.cql_temperature, axis=1)
            q_ood = q_ood * args.cql_temperature
            if bca is None or not dose_active:                                                  # BCA: before 1st refresh
                q_diff = (jnp.expand_dims(q_ood, 1) - q_pred).mean()
            else:                                                                               # BCA: the dose, all 10 critics
                q_diff = (jnp.expand_dims(m_used, 1) * (jnp.expand_dims(q_ood, 1) - q_pred)).mean()  # BCA: plan 4.3
            min_q_loss = q_diff * args.cql_min_q_weight

            critic_loss += min_q_loss.mean()
            if test is not None:                                                                # BCA: test mode only:
                return critic_loss, q_pred                                                      # BCA: q_pred comes out too
            return critic_loss

        if test is None:                                                                        # BCA:
            critic_loss, critic_grad = _q_loss_fn(agent_state.vec_q.params)
        else:                                                                                   # BCA: test mode only: the same
            (critic_loss, q_pred_seen), critic_grad = jax.value_and_grad(                       # BCA: body, differentiated
                _q_loss_fn.__wrapped__, has_aux=True)(agent_state.vec_q.params)                 # BCA: with has_aux
        updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad)
        agent_state = agent_state._replace(vec_q=updated_q)

        loss = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "entropy": entropy,
            "alpha": alpha,
            "q_min": q_min,
            "q_std": q_std,
        }
        if bca is not None:                                                                     # BCA: validity, new carry
            valid = (fit.valid & dose.inputs_valid & dose_guard_ok                              # BCA: cql_bca.py:300-305
                     & BC.reference_valid(bca_state.reference) & BC.tree_finite((agent_state, loss)))  # BCA:
            if not dose_active:                                                                 # BCA: a ready band needs
                valid = valid & ~bca_state.reference.ready                                      # BCA: the dose-active step
            new_state = (rng, agent_state, bca_state._replace(scale=fit.scale, unit=fit.unit, step=t))  # BCA:
            carry = jax.lax.cond(valid, lambda _: new_state, lambda _: start_state, None)       # BCA: old carry whole
            out = dict(loss)                                                                    # BCA: Unifloral's losses,
            out.update(                                                                         # BCA: then BCA's metrics
                inputs_valid=valid, scale_inputs_valid=fit.valid, dose_inputs_valid=dose.inputs_valid,  # BCA:
                dose_guard_ok=dose_guard_ok, scale_loss=fit.loss, unit=carry[2].unit,           # BCA:
                posterior_ready=bca_state.reference.ready, dose_mean=jnp.mean(m), dose_min=jnp.min(m),  # BCA:
                dose_max=jnp.max(m), refused_fraction=jnp.mean(refused.astype(jnp.float32)),     # BCA:
                support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)))               # BCA:
            if test is not None:                                                                # BCA: test outputs
                out.update(                                                                     # BCA:
                    batch_indices=batch_indices, bca_y=bca_y, bca_q=bca_q, unifloral_y=target,  # BCA:
                    unifloral_q=jnp.min(q_pred_seen, axis=-1), dose=m_used, width=dose.width, eta=eta,  # BCA:
                    refused=refused, dosed_loss=critic_loss, dosed_grad=critic_grad,            # BCA:
                    cql_random_actions=cql_random_actions, pi_actions=pi_actions,               # BCA:
                    pi_next_actions=pi_next_actions)                                            # BCA:
            return carry, out                                                                   # BCA:
        return (rng, agent_state), loss

    return _train_step


def band_core(gamma, actor_apply_fn, q_apply_fn, alpha_apply_fn, agent_state, rows, keys):
    """CQL's band target y and critic value q at `rows` (Unifloral's Transition) for `agent_state`, one key per row
    (restated from cql.py:223, 261-275 and 293): y = r + gamma (1 - d) [min of the 10 target critics at a' - alpha
    log pi(a' | s')], a' drawn from the state's actor with the row's key, alpha = exp(the state's log temperature);
    q = the minimum over the 10 online critics at (s, a). Traceable; float32 [n] each."""
    alpha = jnp.exp(alpha_apply_fn(agent_state.alpha.params))                       # cql.py:223

    def _sample_next_v(rng, transition):                                            # cql.py:261-270, verbatim
        next_pi = actor_apply_fn(agent_state.actor.params, transition.next_obs)
        # Note: Important to use sample_and_log_prob here for numerical stability
        # See https://github.com/deepmind/distrax/issues/7
        next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng)
        # Minimum of the target Q-values
        next_q = q_apply_fn(
            agent_state.vec_q_target.params, transition.next_obs, next_action
        )
        return next_q.min(-1) - alpha * log_next_pi.sum(-1)

    next_v_target = jax.vmap(_sample_next_v)(keys, rows)                            # cql.py:274
    y = rows.reward + gamma * (1 - rows.done) * next_v_target                       # cql.py:275
    q = jnp.min(q_apply_fn(agent_state.vec_q.params, rows.obs, rows.action), axis=-1)   # cql.py:293, min of the ten
    return y, q


_band_parts = jax.jit(band_core, static_argnums=(0, 1, 2, 3))


def band_parts(args, apply_fns, agent_state, rows, noise_key):
    """CQL's band target y and critic value q at `rows` (Unifloral's Transition) from a stored state: y = r + gamma
    (1 - d) [min of the 10 stored target critics at a' - alpha log pi(a' | s')], a' drawn from the stored actor with
    per-row keys split(noise_key, n) and alpha the stored temperature; q = min of the 10 stored critics at (s, a)
    (plan section 4.3, CQL refresh bullet; choice 14). float32 [n] each. Used by refresh() and band_readout.py."""
    actor_apply_fn, q_apply_fn, alpha_apply_fn = apply_fns
    keys = BC.per_row_keys(noise_key, int(rows.obs.shape[0]))
    return _band_parts(float(args.gamma), actor_apply_fn, q_apply_fn, alpha_apply_fn, agent_state, rows, keys)


def refresh(args, apply_fns, agent_state, bca_state, bank, refresh_key, *, permission, settings=BC.SETTINGS):
    """BCA's refresh between chunks (eager): the guard on the bank; y and q from the stored state (band_parts) with the
    target-noise key fold_in(refresh_key, 1213156420); eta from the live scale network, which the refresh freezes;
    bca_core.apply_refresh with the posterior key fold_in(refresh_key, 1347375956) (algorithms/cql_bca.py:227-268).
    refresh_key = bca_core.refresh_key(s39, t). Returns bca_core.Refreshed (state, record, scores, stream); for IW the
    runner passes it on to iw.iw_refresh."""
    guard.require_bca(permission, bank)
    y, q = band_parts(args, apply_fns, agent_state, bank, BC.refresh_noise_key(refresh_key, LEARNER))
    eta = bca_state.scale.apply_fn(bca_state.scale.params, bank.obs, bank.action)
    return BC.apply_refresh(bca_state, y, q, eta, BC.posterior_key(refresh_key, LEARNER),
                            inputs_valid=bool(BC.tree_finite((agent_state, bank))), settings=settings)
