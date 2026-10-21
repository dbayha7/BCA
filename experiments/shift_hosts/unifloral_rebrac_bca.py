"""Unifloral's ReBRAC training step with BCA attached: make_train_step of baselines/unifloral/algorithms/rebrac.py:199-309
(commit f2dc127) restated line for line, plus BCA's additions, each marked "# BCA:" (plan sections 3.1, 3.8, 4.1-4.3;
choices 4, 13 and 16). Training environment only (/opt/bca-venv); import it after unifloral_import.prepare().

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. The
repository's ReBRAC+BCA algorithm stays the CORL-derived port algorithms/rebrac_bca.py, and Unifloral's rebrac.py is
imported unchanged through unifloral_import and never edited. Its networks, create_train_state, AgentTrainState,
Transition and Args are used from the imported module (RB.Args etc. below resolve to unifloral_import.load("rebrac")'s
objects); only make_train_step, whose inner losses take no extra argument, is restated.

Restated, with the lines they restate:
  make_train_step   rebrac.py:199-309, every line verbatim in text and order. With bca=None (and test=None) only
                    Unifloral's lines run, so the traced program is the imported one's (test (a1)). BCA's lines are
                    marked "# BCA:"; where BCA replaces a line (the carry, the actor loss with the dose, the return, the
                    test-mode scan), Unifloral's own line stays in place for the BCA-absent path. Seven of Unifloral's
                    lines (REINDENTED_LINES: the critic scan, rebrac.py:255-260, and the actor loss, rebrac.py:274) sit
                    one level (4 spaces) deeper than in rebrac.py, inside BCA's "else:" branches; every other line keeps
                    its own indentation. tests/test_rebrac.py checks the source mechanically, with indentation and blank
                    lines: the lines not marked "# BCA:" are rebrac.py:200-309 exactly, except REINDENTED_LINES, each
                    exactly 4 spaces deeper.
  own_target        rebrac.py:216-235, the body of _compute_target, verbatim, with the agent state as an argument (in
                    the step it is the state at the start of the outer step). It is the band's learning target y in the
                    step's fit, in band_parts and in refresh.
  batch_valid       algorithms/rebrac_bca.py:150-157 (transition_valid without its shape part): finite rows, commands
                    and next commands in [-1, 1], done 0 or 1. check_rows restates the shape part (123-147) for
                    Unifloral's Transition.
  _accept           the port's per-step validity flag and the whole-carry rollback (rebrac_bca.py:394-432, 435-453;
                    td3_bc_bca.py:312-322, 340-342), including the dose range check of rebrac_bca.py:415-417.

BCA present (bca=bca_core.BCASetup("rebrac", PRNGKey(s91), permission)):
  - The factory checks bca.learner, then guard.require_bca(bca.permission, dataset) (made-up arrays or the user's
    terminal approval), and the dataset's shape (check_rows), all outside jit.
  - Carry: (rng, agent_state, bca_state). rng and agent_state are exactly Unifloral's (the key after the network
    split, then the key carried out of each step; AgentTrainState(actor, actor_target, dual_q, dual_q_target)).
    bca_state is bca_core.BCAState(scale, unit, reference, thresholds, most_caution, step); thresholds is float32 over
    all training rows, read as thresholds[batch_indices] with Unifloral's own batch_indices.
  - In order, inside the step (INTERFACES.md section 10):
      1. Unifloral's batch draw, verbatim.
      2. The band fit (plan 4.3; choice 13): once per outer step, before both critic updates. t = step + 1,
         k = fold_in(PRNGKey(s91), t). y = own_target at the step-start state with per-row keys
         split(fold_in(k, 1380338004), batch) (rebrac_bca.py:25, KEY_FIT_NOISE); a' is batch.next_action, Unifloral's
         roll (rebrac.py:335; choice 4). q = min(Q1, Q2)(s, a) of the step-start critic. bca_core.fit_scale_core with
         the bootstrap masses from fold_in(k, 1380077391) (KEY_BOOTSTRAP).
      3. The dose from the frozen band at the recorded (s, a): eta = scale network at reference.cal_params,
         R_b = thresholds[batch_indices], iw.per_row_dose(reference, eta, R_b, blend 0.5, most_caution). Before the
         first refresh every dose is exactly 1. A refused row (R_b = +inf) gets dose 1 with most_caution False (standard
         BCA's rule) and 1 + blend = 1.5 with most_caution True (IW BCA; choice 16, a new rule for ReBRAC).
      4. Unifloral's two critic updates, verbatim (the critic's own cloning pull stays fixed).
      5. The actor loss: actor_loss = -lambda_ * q.mean() + args.actor_bc_coef * (m * bc_loss).mean() with
         m = stop_gradient(dose): Unifloral's order of operations (rebrac.py:274), so m = 1 is Unifloral's value.
      6. Unifloral's actor and target updates, verbatim.
      7. valid = batch_valid & finite (old and new agent state, Unifloral's loss dict) & every dose finite and in [1, 2]
         & the fit's own flag & the dose's own flag & the frozen reference's storage check. The new carry is
         (rng, agent_state, bca_state._replace(scale=fit.scale, unit=fit.unit, step=t)); when not valid,
         jax.lax.cond keeps the old carry whole, random key included. The runner stops on any false flag (stop rule S5).
  - Output: Unifloral's loss dict plus inputs_valid, scale_inputs_valid, dose_inputs_valid, batch_valid,
    dose_in_range, scale_loss, unit (after the step), posterior_ready, dose_mean, dose_min, dose_max, refused_fraction,
    support_fraction (scalars per step).
  - Test mode (test=bca_core.TestOptions(...), made-up arrays only) adds batch_indices, bca_y, bca_q, unifloral_y and
    unifloral_q (the first critic update's targets and the minimum over the two critics of its q_pred, rebrac.py:240,
    245), dose (the multiplier used), band_dose, width, threshold (each batch row's R as read through batch_indices),
    eta (the frozen scale network at the batch rows), refused, dosed_loss (the actor loss) and dosed_grad (its
    gradient). unifloral_keys=True gives the fit Unifloral's own per-row keys of the first critic update
    (rebrac.py:237-238), so bca_y equals unifloral_y (test (a4)); dose_override (float32 over training rows, read
    through batch_indices) replaces the band's dose (test (a3)). width_override is IQL's and is refused here.

band_parts(args, apply_fns, agent_state, rows, noise_key) gives the band's y and q at any rows from a stored state;
refresh(...) is the eager refresh between chunks: guard, y and q from band_parts with fold_in(refresh key, 1380470604)
(KEY_REFRESH_NOISE), eta from the live scale network, bca_core.apply_refresh with the posterior key
fold_in(refresh key, 1380994899) (KEY_POSTERIOR). Bank rows carry the bank rule's a' (plan 3.8; prepared data).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.shift_hosts import bca_core as BC
from experiments.shift_hosts import guard
from experiments.shift_hosts import unifloral_import as UI

LEARNER = "rebrac"
SOURCE = "baselines/unifloral/algorithms/rebrac.py"
RESTATED_LINES = (199, 309)          # make_train_step
REINDENTED_LINES = frozenset(range(255, 261)) | {274}   # Unifloral's lines inside BCA's else: branches (+4 spaces)
TARGET_LINES = (216, 235)            # _compute_target's body (own_target)
DOSE_RANGE = (1.0, 2.0)              # algorithms/rebrac_bca.py:416
TRANSITION_FIELDS = ("obs", "action", "reward", "next_obs", "next_action", "done")       # rebrac.py:68
FLAGS = ("inputs_valid", "scale_inputs_valid", "dose_inputs_valid", "batch_valid", "dose_in_range")  # per-step flags
IMPORTED = ("Args", "Transition", "AgentTrainState", "create_train_state", "DualQNetwork", "SoftQNetwork",
            "DeterministicTanhActor", "sym", "make_train_step_unifloral")


def __getattr__(name):
    """Unifloral's own pieces, imported unchanged (plan section 3.1): RB.Args, RB.Transition, RB.AgentTrainState,
    RB.create_train_state and the networks are unifloral_import.load("rebrac")'s objects; RB.make_train_step_unifloral
    is the imported (unchanged) make_train_step."""
    if name in IMPORTED:
        module = UI.load(LEARNER)
        return module.make_train_step if name == "make_train_step_unifloral" else getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ------------------------------------------------------------------------------------------------ the restated step

def make_train_step(args, actor_apply_fn, q_apply_fn, dataset, *, bca=None, test=None):  # BCA: keyword-only bca, test
    """Make JIT-compatible agent train step."""
    if bca is not None or test is not None:  # BCA: the entry point's checks, outside jit (INTERFACES.md section 10)
        iw, n_train, override = _bca_entry(args, dataset, bca, test)  # BCA:

    def _train_step(runner_state, _):
        if bca is not None:  # BCA: the carry gains BCA's state; Unifloral's own carry is its first two entries
            carry_in, runner_state, bca_state = runner_state, runner_state[:2], runner_state[2]  # BCA:
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        if bca is not None:  # BCA: the band fit before both critic updates (choice 13), then the dose (frozen band)
            t, fit, band_y, band_q = _band_fit(args, (actor_apply_fn, q_apply_fn), bca, bca_state, agent_state,  # BCA:
                                               batch, rng, test)  # BCA:
            dose, refused, m, R_b, eta = _dose(iw, bca, bca_state, batch, batch_indices, n_train, override)  # BCA:
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
                bc_loss = jnp.square(action - transition.next_action).sum()

                # --- Compute targets ---
                target_q = q_apply_fn(
                    agent_state.dual_q_target.params, next_obs, action
                )
                next_q_value = jnp.min(target_q) - args.critic_bc_coef * bc_loss
                next_q_value = (1.0 - transition.done) * next_q_value
                return transition.reward + args.gamma * next_q_value, bc_loss

            rng, rng_targets = jax.random.split(rng)
            rng_targets = jax.random.split(rng_targets, args.batch_size)
            target_fn = jax.vmap(_compute_target)
            targets, bc_loss = target_fn(rng_targets, batch)
            if test is not None:  # BCA: (test mode) this update's targets, and line 245's q_pred at line 249's params
                test_parts = (targets, q_apply_fn(agent_state.dual_q.params, batch.obs, batch.action).min(-1))  # BCA:

            # --- Compute critic loss ---
            @jax.value_and_grad
            def _q_loss_fn(params):
                q_pred = q_apply_fn(params, batch.obs, batch.action)
                q_loss = jnp.square(q_pred - jnp.expand_dims(targets, axis=-1)).sum(-1)
                return q_loss.mean()

            q_loss, q_grad = _q_loss_fn(agent_state.dual_q.params)
            updated_q_state = agent_state.dual_q.apply_gradients(grads=q_grad)
            agent_state = agent_state._replace(dual_q=updated_q_state)
            if test is not None:  # BCA: (test mode) the update also emits test_parts
                return (rng, agent_state), (q_loss, bc_loss) + test_parts  # BCA:
            return (rng, agent_state), (q_loss, bc_loss)

        # --- Iterate critic update ---
        if test is not None:  # BCA: (test mode) the same scan, also collecting each critic update's test_parts
            (rng, agent_state), (q_loss, critic_bc_loss, test_y, test_q) = jax.lax.scan(  # BCA:
                _update_critics, (rng, agent_state), None, length=args.num_critic_updates_per_step  # BCA:
            )  # BCA:
        else:  # BCA: (normal runs, and BCA absent) Unifloral's own lines
            (rng, agent_state), (q_loss, critic_bc_loss) = jax.lax.scan(
                _update_critics,
                (rng, agent_state),
                None,
                length=args.num_critic_updates_per_step,
            )

        # --- Update actor ---
        def _actor_loss_function(params):
            def _transition_loss(transition):
                pi = actor_apply_fn(params, transition.obs)
                pi_action = pi.sample(seed=None)
                q = q_apply_fn(agent_state.dual_q.params, transition.obs, pi_action)
                bc_loss = jnp.square(pi_action - transition.action).sum()
                return q.min(), bc_loss

            q, bc_loss = jax.vmap(_transition_loss)(batch)
            lambda_ = 1.0 / (jnp.abs(q).mean() + 1e-7)
            lambda_ = jax.lax.stop_gradient(lambda_)
            if bca is not None:  # BCA: the dose m = sg(dose) multiplies each row's cloning error (plan 4.3), order kept
                actor_loss = -lambda_ * q.mean() + args.actor_bc_coef * (m * bc_loss).mean()  # BCA:
            else:  # BCA: (BCA absent) Unifloral's own line
                actor_loss = -lambda_ * q.mean() + args.actor_bc_coef * bc_loss.mean()
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
            "critic_bc_loss": critic_bc_loss.mean(),
        }
        if bca is not None:  # BCA: the step's validity flag (plan 4.1); an invalid step keeps the old carry whole
            carry, metrics = _accept(carry_in, rng, agent_state, bca_state, t, fit, dose, refused, m, batch,  # BCA:
                                     loss)  # BCA:
            loss.update(metrics)  # BCA:
            if test is not None:  # BCA: (test mode) INTERFACES.md section 10
                loss.update(dict(batch_indices=batch_indices, bca_y=band_y, bca_q=band_q, unifloral_y=test_y[0],  # BCA:
                                 unifloral_q=test_q[0], dose=m, band_dose=dose.dose, width=dose.width,  # BCA:
                                 threshold=R_b, eta=eta, refused=refused, dosed_loss=actor_loss,  # BCA:
                                 dosed_grad=actor_grad))  # BCA:
            return carry, loss  # BCA:
        return (rng, agent_state), loss

    return _train_step


# ------------------------------------------------------------------------------------------------ the band's parts

def own_target(args, actor_apply_fn, q_apply_fn, agent_state):
    """Unifloral's per-row learning target with the networks of `agent_state`: the body of _compute_target restated
    verbatim from rebrac.py:216-235 (in Unifloral it reads agent_state from the enclosing critic update; here it is an
    argument). For one row and one key: a~' = clip(tanh(actor_target(s')) + clip(0.2 eps, -0.5, 0.5), -1, 1),
    y = r + gamma [(1 - d)(min_k Qbar_k(s', a~') - critic_bc_coef sum_j (a~'_j - a'_j)^2)] with a' = next_action.
    Returns _compute_target(rng, transition) -> (y, critic cloning penalty); map it over rows with jax.vmap, as
    Unifloral does (rebrac.py:239-240)."""

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
        bc_loss = jnp.square(action - transition.next_action).sum()

        # --- Compute targets ---
        target_q = q_apply_fn(
            agent_state.dual_q_target.params, next_obs, action
        )
        next_q_value = jnp.min(target_q) - args.critic_bc_coef * bc_loss
        next_q_value = (1.0 - transition.done) * next_q_value
        return transition.reward + args.gamma * next_q_value, bc_loss

    return _compute_target


def band_parts(args, apply_fns, agent_state, rows, noise_key):
    """The band's target y and critic value q at `rows` (Unifloral's ReBRAC Transition, next_action included) from a
    stored state (plan 4.3 and 4.5; INTERFACES.md section 10): y = own_target mapped over rows with the per-row keys
    jax.random.split(noise_key, n) (bca_core.per_row_keys, the way rebrac.py:237-240 splits its per-row keys);
    q = min(Q1, Q2)(s, a) of the stored online critic (the minimum over the two critics of rebrac.py:245's q_pred).
    apply_fns = (actor_apply_fn, q_apply_fn). Pure; runs eagerly or inside jit. Returns (y, q), float32 [n] each."""
    actor_apply_fn, q_apply_fn = apply_fns
    n = rows.obs.shape[0]
    target_fn = jax.vmap(own_target(args, actor_apply_fn, q_apply_fn, agent_state))
    y, _ = target_fn(BC.per_row_keys(noise_key, n), rows)
    q = q_apply_fn(agent_state.dual_q.params, rows.obs, rows.action).min(-1)
    return y, q


def check_rows(rows, name="rows"):
    """The port's shape and type check of ReBRAC rows (restated from algorithms/rebrac_bca.py:123-147, transition_shape,
    for Unifloral's Transition, rebrac.py:68): Unifloral's Transition with next_action; n > 0 rows; obs and next_obs
    [n, d]; action and next_action [n, k]; reward and done [n]; floating fields (done may be boolean). Outside jit.
    Returns n."""
    if type(rows) is not UI.load(LEARNER).Transition or tuple(rows._fields) != TRANSITION_FIELDS:
        raise ValueError(f"{name}: Unifloral's ReBRAC Transition (rebrac.py:68, with next_action) required")
    n = rows.obs.shape[0] if rows.obs.ndim == 2 else 0
    if (
        not n
        or rows.obs.shape[1] < 1
        or rows.next_obs.shape != rows.obs.shape
        or rows.action.ndim != 2
        or rows.action.shape[0] != n
        or rows.action.shape[1] < 1
        or rows.next_action.shape != rows.action.shape
        or rows.reward.shape != (n,)
        or rows.done.shape != (n,)
    ):
        raise ValueError(f"{name}: nonempty aligned transition matrices and vectors required")
    for field in TRANSITION_FIELDS:
        dtype = getattr(rows, field).dtype
        if not (jnp.issubdtype(dtype, jnp.floating) or (field == "done" and dtype == jnp.bool_)):
            raise TypeError(f"{name}: transition fields must be floating; done may be boolean")
    return int(n)


def batch_valid(batch):
    """The port's validity of a ReBRAC batch (restated from algorithms/rebrac_bca.py:150-157, transition_valid, after
    its shape check, which check_rows does once): every field finite, every command and next command in [-1, 1], done
    0 or 1. Traceable."""
    return (
        BC.tree_finite(batch)
        & jnp.all(jnp.abs(batch.action) <= 1)
        & jnp.all(jnp.abs(batch.next_action) <= 1)
        & jnp.all((batch.done == 0) | (batch.done == 1))
    )


# ------------------------------------------------------------------------------------------------ BCA's additions

def _bca_entry(args, dataset, bca, test):
    """The checks of a BCA entry point, outside jit (INTERFACES.md sections 8 and 10): BCA present for any test option;
    bca.learner is "rebrac"; guard.require_bca(bca.permission, dataset) (made-up arrays or the user's terminal
    approval); test options only on made-up arrays; the dataset's shape (check_rows); dose_override float32, finite,
    one per training row. Returns (the iw module, the number of training rows, the dose override as a JAX array or
    None)."""
    if bca is None:
        raise ValueError("test options need BCA present (INTERFACES.md section 10)")
    if not isinstance(bca, BC.BCASetup) or bca.learner != LEARNER:
        raise ValueError(f"a bca_core.BCASetup for {LEARNER!r} is required")
    guard.require_bca(bca.permission, dataset)
    n_train = check_rows(dataset, "dataset")
    if args.num_critic_updates_per_step < 1:
        raise ValueError("Unifloral's ReBRAC needs at least one critic update per step")
    from experiments.shift_hosts import iw

    override = None
    if test is not None:
        if not isinstance(test, BC.TestOptions):
            raise ValueError("test options must be a bca_core.TestOptions")
        if bca.permission is not guard.MADE_UP:
            raise ValueError("test options run on made-up arrays only (tests/madeup.py)")
        if test.width_override is not None:
            raise ValueError("width_override is IQL's; ReBRAC's test (a3) takes dose_override")
        if test.dose_override is not None:
            d = np.asarray(test.dose_override)
            if d.dtype != np.float32 or d.shape != (n_train,) or not np.all(np.isfinite(d)):
                raise ValueError(f"dose_override must be finite float32 with one value per training row ({n_train})")
            override = jnp.asarray(d)
    return iw, n_train, override


def _band_fit(args, apply_fns, bca, bca_state, agent_state, batch, rng, test):
    """BCA's band fit at outer step t = bca_state.step + 1 (plan 4.3; choice 13): once per outer step, on the step's
    batch, before both critic updates, with the critics at the start of the step. k = fold_in(PRNGKey(s91), t); y and q
    from band_parts at the step-start state with noise key fold_in(k, 1380338004) (rebrac_bca.py:25), or, in test mode
    with unifloral_keys, Unifloral's own rng_targets of the first critic update (rebrac.py:237, from the key the critic
    scan starts with); then bca_core.fit_scale_core with the bootstrap fold 1380077391 (rebrac_bca.py:26).
    Returns (t, FitResult, y, q)."""
    t = bca_state.step + 1
    k = BC.fit_key(bca.fit_key_base, t)
    if test is not None and test.unifloral_keys:
        _, noise_key = jax.random.split(rng)
    else:
        noise_key = BC.fit_noise_key(k, LEARNER)
    y, q = band_parts(args, apply_fns, agent_state, batch, noise_key)
    fit = BC.fit_scale_core(bca_state.scale, bca_state.unit, batch.obs, batch.action, y, q, k,
                            bootstrap_fold=BC.FOLDS[LEARNER].bootstrap, settings=bca.settings)
    return t, fit, y, q


def _dose(iw, bca, bca_state, batch, batch_indices, n_train, override):
    """The dose from the frozen band at the recorded (s, a), no gradient (INTERFACES.md section 10, item 4):
    eta = scale network at reference.cal_params, each batch row's threshold thresholds[batch_indices] (Unifloral's own
    indices), then iw.per_row_dose (one threshold for standard BCA, R(t_i) per row for IW; refused rows by
    most_caution). Test mode's dose_override replaces the multiplier, read through batch_indices.
    Returns (CQLLevelDose, refused rows, m = stop_gradient(the multiplier used), each row's threshold R_b, eta)."""
    if bca_state.thresholds.shape != (n_train,) or bca_state.thresholds.dtype != jnp.float32:
        raise ValueError(f"bca_state.thresholds must be float32 with one value per training row ({n_train})")
    eta = bca_state.scale.apply_fn(bca_state.reference.cal_params, batch.obs, batch.action)
    R_b = bca_state.thresholds[batch_indices]
    dose, refused = iw.per_row_dose(bca_state.reference, eta, R_b, bca.settings.blend, bca_state.most_caution)
    m = dose.dose if override is None else override[batch_indices]
    return dose, refused, jax.lax.stop_gradient(m), R_b, eta


def _accept(carry_in, rng, agent_state, bca_state, t, fit, dose, refused, m, batch, loss):
    """The step's validity flag and the accepted carry (plan 4.1; INTERFACES.md section 10, item 7), carried over from
    the port: batch_valid (rebrac_bca.py:395), finite old and new agent state and Unifloral's loss dict (411), every
    dose finite and in [1, 2] (412-417), the fit's own flag and the dose's own flag (420), and the frozen reference's
    storage check (264, _reference_storage_valid). The proposed carry is (rng, agent_state, bca_state with the fitted
    scale network and unit and step t); jax.lax.cond keeps carry_in whole, random key included, when not valid
    (rebrac_bca.py:425-430, 445-450; td3_bc_bca.py:340-342). Returns (carry, BCA's per-step metrics)."""
    batch_ok = batch_valid(batch)
    dose_in_range = BC.tree_finite(m) & jnp.all((m >= DOSE_RANGE[0]) & (m <= DOSE_RANGE[1]))
    valid = (
        batch_ok
        & BC.tree_finite((carry_in[1], agent_state, loss))
        & dose_in_range
        & fit.valid
        & dose.inputs_valid
        & BC.reference_valid(bca_state.reference)
    )
    proposed = (rng, agent_state, bca_state._replace(scale=fit.scale, unit=fit.unit, step=t))
    carry = jax.lax.cond(valid, lambda _: proposed, lambda _: carry_in, None)
    metrics = dict(
        inputs_valid=valid,
        scale_inputs_valid=fit.valid,
        dose_inputs_valid=dose.inputs_valid,
        batch_valid=batch_ok,
        dose_in_range=dose_in_range,
        scale_loss=fit.loss,
        unit=carry[2].unit,
        posterior_ready=bca_state.reference.ready,
        dose_mean=jnp.mean(m),
        dose_min=jnp.min(m),
        dose_max=jnp.max(m),
        refused_fraction=jnp.mean(refused.astype(jnp.float32)),
        support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)),
    )
    return carry, metrics


# ------------------------------------------------------------------------------------------------ refresh

def refresh(args, apply_fns, agent_state, bca_state, bank, refresh_key, *, permission, settings=BC.SETTINGS):
    """BCA's refresh between chunks (eager; plan 4.1 and 4.3; INTERFACES.md section 10): guard.require_bca(permission,
    bank); y and q at the bank rows from the stored state by band_parts with the noise key
    fold_in(refresh_key, 1380470604) (rebrac_bca.py:27, 352); eta from the live scale network (frozen now, 355);
    bca_core.apply_refresh with the posterior key fold_in(refresh_key, 1380994899) (rebrac_bca.py:28, 367), the
    stored inputs' validity (finite agent state and bank, and batch_valid of the bank: rebrac_bca.py:349, 356-361) and
    the port's check of the NEW frozen reference's storage (rebrac_bca.py:370, valid = valid &
    _reference_storage_valid(reference) and frozen: check_new_reference=True), so a refresh that froze an invalid
    reference fails here (S5, "the refresh failed BCA's own validity checks") rather than one chunk later.
    refresh_key = bca_core.refresh_key(s39, t) after outer step t. bank: Unifloral's Transition of the bank rows with the
    bank rule's next commands (plan 3.8). Returns bca_core.Refreshed(state, record, scores, stream); the state's
    thresholds hold the standard threshold on every training row (the runner overwrites them for IW)."""
    guard.require_bca(permission, bank)
    check_rows(bank, "bank")
    y, q = band_parts(args, apply_fns, agent_state, bank, BC.refresh_noise_key(refresh_key, LEARNER))
    eta = bca_state.scale.apply_fn(bca_state.scale.params, bank.obs, bank.action)
    stored = bool(BC.tree_finite((agent_state, bank)) & batch_valid(bank))
    return BC.apply_refresh(bca_state, y, q, eta, BC.posterior_key(refresh_key, LEARNER), inputs_valid=stored,
                            settings=settings, check_new_reference=True)


def require_valid(metrics):
    """Stop rule S5 on a chunk's stacked outputs, restated from the port's require_valid (algorithms/rebrac_bca.py:
    456-461) with this step's flag names (FLAGS): raises FloatingPointError naming the first false flag and the outer
    step, counted from 1 within the chunk, where it is false."""
    for name in FLAGS:
        if name in metrics:
            flags = np.asarray(metrics[name]).reshape(-1).astype(bool)
            if not np.all(flags):
                first = int(np.flatnonzero(~flags)[0]) + 1
                raise FloatingPointError(f"Unifloral ReBRAC with BCA: invalid step: {name} is false at step {first} "
                                         "of the chunk")
