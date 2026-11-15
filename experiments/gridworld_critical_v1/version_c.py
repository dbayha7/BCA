"""Version C: importance-weighted BCA that acts on the Q values (BCA_IW_PLAN.md).

The shared learner (experiments/gridworld_ood_v1/learner.py) is imported and not edited. This
module adds, for runs whose mode is VERSION_C:

  the weight      w(s, a) = exp(-support score): how much rarer a command is at this cell than among
                  the commands it is compared with. The support score is the learner's classifier,
                  but trained against a different comparison set: commands recorded at other rows
                  as well as uniformly random ones (comparison_commands; the first run of
                  iw_checks.py shows why uniformly random ones alone are not enough).
  the threshold   R(w), the per-step method of experiments/shift_iw (coverage.threshold_table):
                  calibration.wbcp.calibrate on the held-out scores with every held-out row carrying
                  its own weight and the command asked about carrying its own weight as the test
                  mass, on a grid of test masses, every call with the same random stream. +inf
                  means "cannot certify". Between grid points R is interpolated in log mass, so
                  that the policy and the critic have a slope to follow.
  the charge      [R(w) - standard threshold]_+ x scale x unit where the command can be certified;
                  cap x (1 + decades past the last certifiable mass) where it cannot. The cap is
                  half the range of the held-out targets, as before. In the critic's two uses the
                  charge stops at the cap; in the policy's it keeps rising, so the policy always
                  has a slope back toward the data.
  three uses      (1) the critic at random commands that cannot be certified: Q may not stay above
                      the recorded row's target minus KAPPA x charge (pushed down only);
                  (2) the critic's target: the next value is lowered by KAPPA x charge at the next command;
                  (3) the policy's objective: plus c x KAPPA x charge at its own command.

For every other mode the step below is the shared learner's step, line for line; a test checks
that it gives the same numbers to float rounding (about 4e-9; not bit for bit, because the compiler
rounds a few sums differently once the extra terms exist). So only VERSION_C runs are trained with
this step; baselines and Version A stay the shared learner's own.
"""
import numpy as np

import jax
import jax.numpy as jnp
import optax

import learner as L
from calibration import wbcp
from calibration.reference import WBCPConfig, positive_scale

VERSION_C = 100                 # a mode number the shared learner does not use
GRID_POINTS = 40
LOGIT_CLIP = 30.0               # exp(+-30) is representable; the classifier's scores are far inside it
LN10 = L.LN10
IW_FIELDS = ("iw_logm", "iw_R", "iw_logm_cert", "iw_ready")


# ---- host side: the table of thresholds (NumPy) ------------------------------------------------------
def scores_of(eta, residual, unit):
    """A bank's scores, exactly as calibration/reference.freeze_reference computes them."""
    scale = np.asarray(positive_scale(jnp.asarray(eta), jnp.asarray(unit)), np.float64)
    return np.abs(np.asarray(residual, np.float64)) / scale


def weights_of(logit):
    """w = exp(-support score), in float64."""
    return np.exp(-np.clip(np.asarray(logit, np.float64), -LOGIT_CLIP, LOGIT_CLIP))


def mass_grid(weights, points=GRID_POINTS):
    """Log-spaced test masses from well below the lightest held-out row to well past the point
    where nothing can be certified (total held-out weight x alpha / (1 - alpha))."""
    low = np.log(weights.min()) - 2.0
    high = np.log(weights.sum()) + 3.0
    return np.exp(np.linspace(low, high, points))


def threshold_table(scores, seed_words, weights, grid, config=WBCPConfig()):
    """R(t) at every grid mass t: wbcp.calibrate with the held-out weights and test mass t, each
    call with a fresh generator from the same words (experiments/shift_iw/coverage.threshold_table).
    Returns (raw thresholds, running maximum)."""
    words = np.asarray(seed_words, np.uint32).ravel()
    raw = np.array([wbcp.calibrate(np.asarray(scores, np.float64), np.random.default_rng(words),
                                   np.asarray(weights, np.float64), float(t), alpha=config.alpha,
                                   beta=config.credibility, draws=config.draws).threshold for t in grid])
    return raw, np.maximum.accumulate(raw)


def iw_fields(scores, logit, seed_words, standard_threshold):
    """The four reference fields of one run, and a small record for the receipts."""
    weights = weights_of(logit)
    grid = mass_grid(weights)
    raw, used = threshold_table(scores, seed_words, weights, grid)
    finite = np.isfinite(used)
    if not finite.any():        # nothing certifiable at any mass: every command is charged the cap
        last, filled = 0, np.full(len(grid), float(standard_threshold))
        cert = float(np.log(grid[0]) - 1.0)
    else:
        last = int(np.flatnonzero(finite)[-1])
        filled = np.where(finite, used, used[last])
        cert = float(np.log(grid[last]))
    fields = {"iw_logm": np.log(grid).astype(np.float32), "iw_R": filled.astype(np.float32),
              "iw_logm_cert": np.float32(cert), "iw_ready": np.bool_(True)}
    record = {"grid_points": len(grid), "certifiable_points": int(finite.sum()),
              "log_mass_at_the_last_certifiable_point": cert,
              "threshold_at_the_lightest_mass": float(filled[0]),
              "threshold_at_the_last_certifiable_point": float(filled[last]),
              "standard_threshold": float(standard_threshold), "monotone": bool(np.all(raw[1:] >= raw[:-1])),
              "median_log_weight_of_held_out_rows": float(np.median(np.log(weights))),
              "kish_effective_rows": float(weights.sum() ** 2 / np.dot(weights, weights))}
    return fields, record


def blank_iw_fields():
    return {"iw_logm": jnp.linspace(0.0, 1.0, GRID_POINTS, dtype=jnp.float32),
            "iw_R": jnp.zeros((GRID_POINTS,), jnp.float32), "iw_logm_cert": jnp.float32(0.0),
            "iw_ready": jnp.asarray(False)}


def with_blank_iw(states):
    """Add the four fields, not ready, to a stack of states (the leading axis is the runs)."""
    runs = int(states["step"].shape[0])
    blank = blank_iw_fields()
    extra = {name: jnp.broadcast_to(value, (runs,) + value.shape) for name, value in blank.items()}
    return dict(states, reference=dict(states["reference"], **extra))


# ---- the equations (JAX) -------------------------------------------------------------------------
def comparison_commands(key, action):
    """The commands a recorded command is compared with, one per batch row: for half the rows a
    command recorded at ANOTHER row of the batch, for the other half a uniformly random command.

    The shared learner compares with uniformly random commands only. Those are almost never near a
    real move, so its classifier is never shown "a real move that is not recorded at this cell" and
    rates every real move as data-like everywhere (iw_checks.py, first run). The commands recorded
    at other rows put exactly those cases in front of it."""
    k_perm, k_coin, k_uniform = jax.random.split(key, 3)
    borrowed = action[jax.random.permutation(k_perm, action.shape[0])]
    uniform = jax.random.uniform(k_uniform, action.shape, jnp.float32, -1.0, 1.0)
    return jnp.where(jax.random.bernoulli(k_coin, 0.5, (action.shape[0], 1)), borrowed, uniform)


def comparison_rows(key, obs, action):
    """What the classifier compares recorded rows with, one per batch row, in thirds: this row's cell
    with a command recorded at another row; this row's cell with a uniformly random command; a
    uniformly random cell with a command recorded at another row. The last third makes the weight
    large at cells the data never visit, which nothing else can tell the classifier about."""
    k_perm, k_kind, k_uniform, k_cell = jax.random.split(key, 4)
    rows = action.shape[0]
    borrowed = action[jax.random.permutation(k_perm, rows)]
    uniform = jax.random.uniform(k_uniform, action.shape, jnp.float32, -1.0, 1.0)
    kind = jax.random.randint(k_kind, (rows, 1), 0, 3)
    other_cell = jax.nn.one_hot(jax.random.randint(k_cell, (rows,), 0, obs.shape[1]), obs.shape[1], dtype=obs.dtype)
    return jnp.where(kind == 2, other_cell, obs), jnp.where(kind == 1, uniform, borrowed)


def weight_objective(support, obs, action, compared_obs, compared_action):
    """Logistic loss: recorded rows are label 1, comparison rows label 0."""
    return (jnp.mean(jax.nn.softplus(-L.support_logit(support, obs, action)))
            + jnp.mean(jax.nn.softplus(L.support_logit(support, compared_obs, compared_action))))


def log_mass(reference, obs, action):
    return -L.support_logit(jax.lax.stop_gradient(reference["support"]), obs, action)


def cannot_certify(reference, obs, action):
    return jnp.logical_and(reference["iw_ready"], log_mass(reference, obs, action) > reference["iw_logm_cert"])


def charge(cal, reference, obs, action, capped=False):
    """The extra width importance weighting adds at a command. Zero until the table is ready.

    `capped` stops the charge at the cap. The critic's two uses are capped: a larger push would ask
    the critic to hold values far outside anything it is trained on. The policy's use is not, so
    that it always has a slope back toward the data."""
    logm = log_mass(reference, obs, action)
    threshold = jnp.interp(logm, reference["iw_logm"], reference["iw_R"])
    eta = cal.apply(jax.lax.stop_gradient(reference["scale"]), obs, action)
    scale = positive_scale(eta, jax.lax.stop_gradient(reference["unit"]))
    certified = jax.nn.relu(threshold - reference["threshold"]) * scale
    decades = jax.nn.relu(logm - reference["iw_logm_cert"]) / LN10
    beyond = jnp.maximum(certified, reference["width_cap"] * (1.0 + decades))
    value = jnp.where(logm > reference["iw_logm_cert"], beyond, certified)
    if capped:
        value = jnp.minimum(value, reference["width_cap"])
    return jnp.where(reference["iw_ready"], value, jnp.zeros_like(value))


def target_c(cal, reference, actor_t, q1_t, q2_t, next_obs, reward, done, noise):
    """Use 2: the Bellman target with the next value lowered by KAPPA x charge at the next command."""
    next_action = L.smoothed_action(actor_t, next_obs, noise)
    return L.bellman_target(actor_t, q1_t, q2_t, next_obs, reward, done, noise,
                            L.KAPPA * charge(cal, reference, next_obs, next_action, capped=True))


def unsupported_loss(critics, cal, reference, obs, random_action, target):
    """Use 1: at random commands that cannot be certified, each critic may not score above the
    recorded row's target minus KAPPA x charge. A hinge: it only ever pushes a score down."""
    bound = jax.lax.stop_gradient(target - L.KAPPA * charge(cal, reference, obs, random_action, capped=True))
    flagged = jax.lax.stop_gradient(cannot_certify(reference, obs, random_action).astype(jnp.float32))
    return sum(jnp.mean(flagged * jnp.square(jax.nn.relu(L.critic_apply(q, obs, random_action) - bound)))
               for q in critics)


def policy_charge(actor, q1, cal, reference, obs):
    """Use 3: c x KAPPA x mean charge at the policy's own command, c as in the host's objective."""
    command = L.actor_apply(actor, obs)
    q = L.critic_apply(q1, obs, command)
    coefficient = jax.lax.stop_gradient(L.ALPHA / (jnp.mean(jnp.abs(q)) + 1e-7))
    return coefficient * L.KAPPA * jnp.mean(charge(cal, reference, obs, command))


# ---- the update step: the shared learner's, with the three uses added for VERSION_C -------------------
def make_step(observation, n_rows):
    obs_fn, obs_dim = L.OBSERVATIONS[observation]
    cal = L.calibrator(obs_dim)

    def step(state, data, key, mode, t):
        is_c = mode == VERSION_C
        on = is_c.astype(jnp.float32)
        k_batch, k_fit, k_prior, k_host, k_perm = L.step_keys(key, t)
        k_random = jax.random.fold_in(jax.random.fold_in(key, t), L.SUPPORT_KEY_FOLD)
        arm = L.switches(jnp.where(is_c, L.NATIVE, mode))
        idx = jax.random.randint(k_batch, (L.BATCH,), 0, n_rows)
        obs, next_obs = obs_fn(data["s"][idx]), obs_fn(data["s2"][idx])
        action, reward, done = data["a"][idx], data["r"][idx], data["done"][idx]
        targets = (state["actor_t"], state["q1_t"], state["q2_t"])
        reference = state["reference"]
        order = jax.random.permutation(k_perm, L.BATCH)
        blind = jnp.logical_or(arm["shuffled"], arm["blind_base"])
        width_obs = jnp.where(blind, obs[order], obs)
        width_next_obs = jnp.where(blind, next_obs[order], next_obs)
        support_obs = jnp.where(arm["shuffled"], obs[order], obs)
        support_next_obs = jnp.where(arm["shuffled"], next_obs[order], next_obs)

        def target_for(noise):
            shared = L.trained_target(cal, arm, reference, *targets, next_obs, width_next_obs, reward, done, noise,
                                      support_next_obs)
            return jnp.where(is_c, target_c(cal, reference, *targets, next_obs, reward, done, noise), shared)

        # 1. The scale network, on the target this run's critic is trained toward.
        fit_target = target_for(jax.random.normal(k_fit, action.shape))
        fit_q = jnp.minimum(L.critic_apply(state["q1"], obs, action), L.critic_apply(state["q2"], obs, action))
        prior = jax.random.exponential(k_prior, (L.BATCH,))
        prior = prior / prior.sum()
        scale_grad = jax.grad(L.scale_objective)(state["scale"], cal, obs, action, fit_target, fit_q,
                                                 state["unit"], prior)
        updates, scale_opt = L._scale_opt.update(scale_grad, state["scale_opt"], state["scale"])
        scale = optax.apply_updates(state["scale"], updates)
        unit = L.unit_update(state["unit"], fit_target, fit_q)

        # 1b. The support classifier. VERSION_C compares with commands recorded at other rows too,
        #     and shows the critic those same commands in use 1.
        random_action = jax.random.uniform(k_random, action.shape, jnp.float32, -1.0, 1.0)
        rows_obs, rows_action = comparison_rows(jax.random.fold_in(k_random, 2), obs, action)
        support_grad = jax.grad(weight_objective)(state["support"], obs, action, jnp.where(is_c, rows_obs, obs),
                                                  jnp.where(is_c, rows_action, random_action))
        random_action = jnp.where(is_c, comparison_commands(jax.random.fold_in(k_random, 1), action), random_action)
        updates, support_opt = L._support_opt.update(support_grad, state["support_opt"], state["support"])
        support = optax.apply_updates(state["support"], updates)

        # 2. The critics. VERSION_C adds use 1.
        target = jax.lax.stop_gradient(target_for(jax.random.normal(k_host, action.shape)))
        critics = (state["q1"], state["q2"])

        # The shared gradient is computed exactly as the shared learner computes it; Version C's
        # extra gradient is added to it, times 1 for VERSION_C and 0 for every other mode.
        critic_grad = jax.grad(L.critic_loss)(critics, obs, action, target)
        extra_grad = jax.grad(unsupported_loss)(critics, cal, reference, obs, random_action, target)
        critic_grad = jax.tree_util.tree_map(lambda g, e: g + on * e, critic_grad, extra_grad)
        updates, critic_opt = L._host_opt.update(critic_grad, state["critic_opt"], critics)
        q1, q2 = optax.apply_updates(critics, updates)

        # 2b. The proposal network (consumed only by the permission arms of the earlier testbed).
        proposal_grad = jax.grad(L.proposal_objective)(state["proposal"], q1, cal, reference, obs)
        updates, proposal_opt = L._proposal_opt.update(proposal_grad, state["proposal_opt"], state["proposal"])
        proposal = optax.apply_updates(state["proposal"], updates)
        keep = L.permission(proposal, q1, cal, reference, obs, action, arm["gate_no_base"])

        # 3. The policy and the target networks. VERSION_C adds use 3.
        actor_grad = jax.grad(L.actor_objective)(state["actor"], q1, cal, reference, obs, width_obs, action,
                                                 arm["kappa"], arm["support"], arm["one_sided"], arm["no_base"],
                                                 support_obs, keep, arm["gate"])
        extra_grad = jax.grad(policy_charge)(state["actor"], q1, cal, reference, obs)
        actor_grad = jax.tree_util.tree_map(lambda g, e: g + on * e, actor_grad, extra_grad)
        updates, actor_opt_new = L._host_opt.update(actor_grad, state["actor_opt"], state["actor"])
        actor_new = optax.apply_updates(state["actor"], updates)
        due = (t % L.POLICY_DELAY) == 0

        def choose(new, old):
            return jax.tree_util.tree_map(lambda a, b: jnp.where(due, a, b), new, old)

        actor = choose(actor_new, state["actor"])
        return {"actor": actor, "q1": q1, "q2": q2,
                "actor_t": choose(L.polyak(actor_new, state["actor_t"]), state["actor_t"]),
                "q1_t": choose(L.polyak(q1, state["q1_t"]), state["q1_t"]),
                "q2_t": choose(L.polyak(q2, state["q2_t"]), state["q2_t"]),
                "actor_opt": choose(actor_opt_new, state["actor_opt"]), "critic_opt": critic_opt,
                "scale": scale, "scale_opt": scale_opt, "unit": unit,
                "support": support, "support_opt": support_opt,
                "proposal": proposal, "proposal_opt": proposal_opt,
                "reference": reference, "step": jnp.int32(t)}

    return step


def make_weight_fit(observation, n_rows):
    """Train only the support classifier, with Version C's comparison commands, for the update
    numbers in `steps`. Used once before Version C starts from a baseline's saved state, whose
    classifier was trained against uniformly random commands only. Jitted, vmapped over runs."""
    obs_fn, _ = L.OBSERVATIONS[observation]

    def one(carry, t, data, key):
        support, opt = carry
        k_batch, k_commands = jax.random.split(jax.random.fold_in(jax.random.fold_in(key, 3_000_000), t))
        idx = jax.random.randint(k_batch, (L.BATCH,), 0, n_rows)
        obs, action = obs_fn(data["s"][idx]), data["a"][idx]
        compared_obs, compared_action = comparison_rows(k_commands, obs, action)
        grad = jax.grad(weight_objective)(support, obs, action, compared_obs, compared_action)
        updates, opt = L._support_opt.update(grad, opt, support)
        return (optax.apply_updates(support, updates), opt), None

    def fit(state, data, key, steps):
        (support, opt), _ = jax.lax.scan(lambda c, t: one(c, t, data, key), (state["support"], state["support_opt"]),
                                         steps)
        return dict(state, support=support, support_opt=opt, reference=dict(state["reference"], support=support))

    return jax.jit(jax.vmap(fit, in_axes=(0, 0, 0, None)))


def make_segment(observation, n_rows):
    step = make_step(observation, n_rows)

    def segment(state, data, key, mode, steps):
        return jax.lax.scan(lambda s, t: (step(s, data, key, mode, t), None), state, steps)[0]

    return jax.jit(jax.vmap(segment, in_axes=(0, 0, 0, 0, None)))


def make_readouts(observation):
    """Read-outs that need Version C's own target or charge."""
    obs_fn, obs_dim = L.OBSERVATIONS[observation]
    cal = L.calibrator(obs_dim)

    def heldout(state, bank, key, mode):
        """On held-out rows: the scale prediction, the residual against the target this run's critic is
        trained toward, the plain target, and the support score of the recorded command."""
        is_c = mode == VERSION_C
        arm = L.switches(jnp.where(is_c, L.NATIVE, mode))
        obs, next_obs = obs_fn(bank["s"]), obs_fn(bank["s2"])
        targets = (state["actor_t"], state["q1_t"], state["q2_t"])
        noise = jax.random.normal(key, bank["a"].shape)
        order = jax.random.permutation(jax.random.fold_in(key, 1), bank["s"].shape[0])
        width_next_obs = jnp.where(arm["shuffled"], next_obs[order], next_obs)
        plain = L.bellman_target(*targets, next_obs, bank["r"], bank["done"], noise)
        shared = L.trained_target(cal, arm, state["reference"], *targets, next_obs, width_next_obs,
                                  bank["r"], bank["done"], noise)
        trained = jnp.where(is_c, target_c(cal, state["reference"], *targets, next_obs, bank["r"], bank["done"],
                                           noise), shared)
        q = jnp.minimum(L.critic_apply(state["q1"], obs, bank["a"]), L.critic_apply(state["q2"], obs, bank["a"]))
        return (cal.apply(state["scale"], obs, bank["a"]), trained - q, plain,
                L.support_logit(state["support"], obs, bank["a"]))

    def probe(state, cells, corners):
        """At every cell and corner command: the charge, whether it cannot be certified, the log mass."""
        reference = state["reference"]
        obs = jnp.repeat(obs_fn(cells), corners.shape[0], axis=0)
        command = jnp.tile(corners, (cells.shape[0], 1))
        shape = (cells.shape[0], corners.shape[0])
        return (charge(cal, reference, obs, command).reshape(shape),
                cannot_certify(reference, obs, command).reshape(shape),
                log_mass(reference, obs, command).reshape(shape))

    return {"heldout": jax.jit(jax.vmap(heldout, in_axes=(0, 0, 0, 0))),
            "probe": jax.jit(jax.vmap(probe, in_axes=(0, None, None)))}
