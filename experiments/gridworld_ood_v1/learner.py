"""JAX re-implementation of the archived GridWorld TD3+BC host with the direct-width BCA penalty,
plus three candidate changes to the method (PROTOCOL.md, Amendment 3).

Why this exists: the archived host is PyTorch and no PyTorch is installed on this
machine. This file re-derives that host from the archived source and is NOT
bit-identical to it (different random streams, different framework).

Equations and their archived source (under
analysis/gridworld_stress_20261107/archive/td3_gridworld/):
  networks, Adam 3e-4, batch 256, delay 2, tau 0.005   train.py
  Bellman target, critic loss, actor objective           td3bc_source_actor.py
  scale fit, residual unit, width, actor width penalty   bca_direct_width.py
  schedule: fit scale, then host update, then refresh    bca_direct_width.scheduled_step

The scale network, the coverage function and the calibration are NOT re-derived.
They are imported from this repository's calibration/ package, whose files are
byte-identical to the copies the archived port was pinned against.

In the original method the width never enters the critic loss or the Bellman target.
It enters only the actor objective, through d(width)/d(action), with the reference frozen.

Arms. Every arm runs the same code on the same random draws; only these switches differ.

  arm                         width penalty  support-aware  one-sided  in critic target  states shuffled
  baseline                    off
  bca                         on
  shuffled_width              on                                                          yes
  support                     on             yes
  support_shuffled            on             yes                                          yes
  support_onesided            on             yes            yes
  support_onesided_shuffled   on             yes            yes                           yes
  full                        on             yes            yes        yes
  full_shuffled               on             yes            yes        yes                yes
  onesided                    on                            yes
  onesided_shuffled           on                            yes                           yes
  support_only                on             yes, and the calibrated width is dropped (Amendment 4)
  support_only_shuffled       on             the same                                     yes
  support_blind_width         on             yes; the calibrated part alone is read at shuffled states (Amendment 5)

The three changes:
  support-aware  The width is raised where a learned support score says the data hold no
                 similar command at that state. The score is a classifier trained to tell
                 recorded commands from uniformly random ones at the same states.
  one-sided      Only width above a reference level is penalized. The reference is the 90th
                 percentile of the width at held-out recorded commands. Low width is never rewarded.
  critic target  The same excess width is also subtracted from the next-state value. Wherever such
                 an arm forms a one-step target (critic update, scale fit, calibration scores) it
                 forms this penalized one, so the width keeps measuring fit error, not the penalty.
A "shuffled" arm evaluates all of this at a random reordering of the batch's states.

Amendment 4 adds one more switch, used by the last two arms: drop the calibrated width. The width
such an arm uses is only the support-aware raise, cap times the number of decades below the floor,
so it is exactly zero at every supported command.

Amendment 5 adds the last switch: read the calibrated width at shuffled states while the support
score is still read at the true state. That arm is change 1 with the calibrated width blindfolded.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

# NVIDIA GPUs otherwise multiply matrices in a reduced "TensorFloat-32" precision, which put
# the GPU about 3e-4 away from the NumPy reference equations. Ask for true float32.
jax.config.update("jax_default_matmul_precision", "float32")

from calibration.network import Calibrator, soft_coverage  # noqa: E402
from calibration.reference import positive_scale  # noqa: E402

GAMMA, TAU = 0.95, 0.005
HOST_LR, SCALE_LR, SUPPORT_LR = 3e-4, 1e-3, 1e-3
BATCH, HIDDEN = 256, 64
ALPHA = 2.5                         # TD3+BC Q-normalization numerator
POLICY_NOISE, NOISE_CLIP, POLICY_DELAY = 0.2, 0.5, 2
KAPPA = 3.0
COVERAGE_TARGET, COVERAGE_SLOPE, ETA_PENALTY, UNIT_EMA = 0.9, 20.0, 0.005, 0.99
REFRESH_STEPS = (10_000, 15_000, 20_000)
SCALE_INIT_FOLD = 1128352841        # the fold used by the BCA source when it initializes the scale network
SUPPORT_KEY_FOLD = 7919             # separate key for the support classifier's random commands

# Settings of the three changes, fixed before any run (Amendment 3).
SUPPORT_QUANTILE = 0.01             # a command is "unsupported" if it scores below 99% of held-out recorded commands
WIDTH_REFERENCE_QUANTILE = 0.90     # one-sided penalty: only width above this quantile of held-out widths counts
LN10 = float(np.log(10.0))          # the width reaches its cap one decade of density below the support floor

(NATIVE, BCA, SHUFFLED, SUPPORT, SUPPORT_SHUFFLED, SUPPORT_ONESIDED, SUPPORT_ONESIDED_SHUFFLED,
 FULL, FULL_SHUFFLED, ONESIDED, ONESIDED_SHUFFLED, SUPPORT_ONLY, SUPPORT_ONLY_SHUFFLED,
 SUPPORT_BLIND_WIDTH) = range(14)
ARM_NAMES = {NATIVE: "baseline", BCA: "bca", SHUFFLED: "shuffled_width", SUPPORT: "support",
             SUPPORT_SHUFFLED: "support_shuffled", SUPPORT_ONESIDED: "support_onesided",
             SUPPORT_ONESIDED_SHUFFLED: "support_onesided_shuffled", FULL: "full",
             FULL_SHUFFLED: "full_shuffled", ONESIDED: "onesided", ONESIDED_SHUFFLED: "onesided_shuffled",
             SUPPORT_ONLY: "support_only", SUPPORT_ONLY_SHUFFLED: "support_only_shuffled",
             SUPPORT_BLIND_WIDTH: "support_blind_width"}
ARM_NUMBERS = {name: number for number, name in ARM_NAMES.items()}
# Each variant's own state-blind control.
CONTROL_OF = {"bca": "shuffled_width", "support": "support_shuffled",
              "support_onesided": "support_onesided_shuffled", "full": "full_shuffled",
              "onesided": "onesided_shuffled", "support_only": "support_only_shuffled",
              "support_blind_width": "support_shuffled"}     # shuffling everything gives the same control as change 1
_PENALIZED = jnp.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], bool)
_SHUFFLED = jnp.asarray([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], bool)
_SUPPORT = jnp.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], bool)
_ONE_SIDED = jnp.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], bool)
_CRITIC = jnp.asarray([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], bool)
_NO_BASE = jnp.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], bool)
_BLIND_BASE = jnp.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], bool)

_host_opt = optax.adam(HOST_LR)     # optax.adam's eps=1e-8 sits outside the square root, as in PyTorch
_scale_opt = optax.adam(SCALE_LR)
_support_opt = optax.adam(SUPPORT_LR)


def switches(mode):
    """The seven switches of an arm, as arrays so that runs with different arms can share a batch."""
    return {"kappa": jnp.where(_PENALIZED[mode], KAPPA, 0.0), "shuffled": _SHUFFLED[mode],
            "support": _SUPPORT[mode], "one_sided": _ONE_SIDED[mode], "critic": _CRITIC[mode],
            "no_base": _NO_BASE[mode], "blind_base": _BLIND_BASE[mode]}


# ---- observations ----------------------------------------------------------------------------
def one_hot_obs(cells):
    return jax.nn.one_hot(cells, 100, dtype=jnp.float32)


def coordinate_obs(cells):
    return jnp.stack([2 * (cells % 10) / 9 - 1, 2 * (cells // 10) / 9 - 1], axis=-1).astype(jnp.float32)


OBSERVATIONS = {"one_hot": (one_hot_obs, 100), "coordinates": (coordinate_obs, 2)}


def calibrator(obs_dim):
    """The repository's scale network, with the archived port's identity normalization."""
    return Calibrator(jnp.zeros(obs_dim), jnp.full((obs_dim,), 0.999), state_dep=True)


# ---- host networks: two hidden layers of 64, PyTorch's default initialization -----------------
def _init_linear(key, fan_in, fan_out):
    kw, kb = jax.random.split(key)
    bound = 1.0 / np.sqrt(fan_in)
    return {"w": jax.random.uniform(kw, (fan_in, fan_out), jnp.float32, -bound, bound),
            "b": jax.random.uniform(kb, (fan_out,), jnp.float32, -bound, bound)}


def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    return [_init_linear(k, a, b) for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def _mlp(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def actor_apply(params, obs):
    return jnp.tanh(_mlp(params, obs))


def critic_apply(params, obs, action):
    return _mlp(params, jnp.concatenate([obs, action], axis=-1))[..., 0]


def support_logit(params, obs, action):
    """Log-odds that (state, command) is a recorded pair rather than a uniformly random command."""
    return _mlp(params, jnp.concatenate([obs, action], axis=-1))[..., 0]


# ---- the equations, each a pure function of explicit inputs --------------------------------------
def smoothed_action(actor_t, next_obs, noise):
    noise = jnp.clip(noise * POLICY_NOISE, -NOISE_CLIP, NOISE_CLIP)
    return jnp.clip(actor_apply(actor_t, next_obs) + noise, -1.0, 1.0)


def bellman_target(actor_t, q1_t, q2_t, next_obs, reward, done, noise, next_penalty=0.0):
    """y = r + (1 - done) * gamma * (min(Q1', Q2')(s', a') - next_penalty), a' = clip(actor'(s') + clipped noise).

    next_penalty is zero in every arm except "full" and "full_shuffled"."""
    next_action = smoothed_action(actor_t, next_obs, noise)
    next_q = jnp.minimum(critic_apply(q1_t, next_obs, next_action), critic_apply(q2_t, next_obs, next_action))
    return reward + (1.0 - done) * GAMMA * (next_q - next_penalty)


def critic_loss(critics, obs, action, target):
    q1, q2 = critics
    return (jnp.mean(jnp.square(critic_apply(q1, obs, action) - target))
            + jnp.mean(jnp.square(critic_apply(q2, obs, action) - target)))


def width(cal, reference, obs, action):
    """Calibrated one-step residual half-width. Reference parameters are frozen; d/d(action) stays live."""
    eta = cal.apply(jax.lax.stop_gradient(reference["scale"]), obs, action)
    value = (jax.lax.stop_gradient(reference["threshold"])
             * positive_scale(eta, jax.lax.stop_gradient(reference["unit"])))
    return jnp.where(reference["ready"], value, jnp.zeros_like(eta))


def consumed_width(cal, reference, obs, action, use_support=False, drop_base=False, support_obs=None):
    """The width an arm actually uses.

    With use_support, the width is raised toward `width_cap` in proportion to how many decades
    the command's support score lies below `support_floor`, and beyond the cap past one decade.
    A command scoring at or above the floor keeps its original width exactly.
    With drop_base, the calibrated width is replaced by zero before the raise, so only the
    support-aware part is left: cap times the decades below the floor.
    The calibrated width is read at `obs`. The support score is read at `support_obs`, which is
    `obs` unless given; only the "support_blind_width" arm gives the two different states."""
    base = jnp.where(drop_base, 0.0, width(cal, reference, obs, action))
    logit = support_logit(jax.lax.stop_gradient(reference["support"]),
                          obs if support_obs is None else support_obs, action)
    decades = jax.nn.relu(jax.lax.stop_gradient(reference["support_floor"]) - logit) / LN10
    raised = base + decades * jax.nn.relu(jax.lax.stop_gradient(reference["width_cap"]) - base)
    return jnp.where(jnp.logical_and(use_support, reference["ready"]), raised, base)


def penalty(cal, reference, obs, action, use_support=False, one_sided=False, drop_base=False, support_obs=None):
    """What is multiplied by kappa. Two-sided: the width itself. One-sided: only its excess
    over the reference level, so low width is never rewarded."""
    value = consumed_width(cal, reference, obs, action, use_support, drop_base, support_obs)
    level = jnp.where(use_support, reference["width_ref_support"], reference["width_ref"])
    return jnp.where(one_sided, jax.nn.relu(value - jax.lax.stop_gradient(level)), value)


def trained_target(cal, arm, reference, actor_t, q1_t, q2_t, next_obs, width_next_obs, reward, done, noise,
                   support_next_obs=None):
    """The one-step target an arm's critic is trained toward. It is the plain Bellman target in
    every arm except "full" and "full_shuffled", which first subtract kappa times the excess
    width at the next state's smoothed command."""
    next_action = smoothed_action(actor_t, next_obs, noise)
    next_penalty = jnp.where(
        arm["critic"],
        KAPPA * penalty(cal, reference, width_next_obs, next_action, arm["support"], True, arm["no_base"],
                        support_next_obs), 0.0)
    return bellman_target(actor_t, q1_t, q2_t, next_obs, reward, done, noise, next_penalty)


def actor_objective(actor, q1, cal, reference, obs, width_obs, recorded_action, kappa,
                    use_support=False, one_sided=False, drop_base=False, support_obs=None):
    """-c*mean(Q1(s, pi)) + mean((pi - a)^2) + c*kappa*mean(penalty(s_w, pi)), with c detached."""
    policy_action = actor_apply(actor, obs)
    q = critic_apply(q1, obs, policy_action)
    coefficient = jax.lax.stop_gradient(ALPHA / (jnp.mean(jnp.abs(q)) + 1e-7))
    cloning = jnp.mean(jnp.square(policy_action - recorded_action))
    cost = jnp.mean(penalty(cal, reference, width_obs, policy_action, use_support, one_sided, drop_base,
                            support_obs))
    return -coefficient * jnp.mean(q) + cloning + coefficient * kappa * cost


def scale_objective(scale, cal, obs, action, target, q, unit, prior):
    """(weighted soft coverage - 0.9)^2 + 0.005 * weighted eta^2, in residual-unit coordinates."""
    eta = cal.apply(scale, obs, action)
    coverage = soft_coverage(target / unit, q / unit, eta, COVERAGE_SLOPE)
    return (jnp.square(jnp.sum(prior * coverage) - COVERAGE_TARGET)
            + ETA_PENALTY * jnp.sum(prior * jnp.square(eta)))


def support_objective(support, obs, action, random_action):
    """Logistic loss: recorded commands are label 1, uniformly random commands at the same states label 0."""
    return (jnp.mean(jax.nn.softplus(-support_logit(support, obs, action)))
            + jnp.mean(jax.nn.softplus(support_logit(support, obs, random_action))))


def unit_update(unit, target, q):
    return UNIT_EMA * unit + (1.0 - UNIT_EMA) * jnp.maximum(jnp.std(target - q), 1e-6)


def polyak(online, target):
    return jax.tree_util.tree_map(lambda p, t: (1.0 - TAU) * t + TAU * p, online, target)


def step_keys(key, t):
    """Five independent keys for update number t: minibatch, scale-fit noise, bootstrap masses,
    host noise, permutation. Depends only on (run key, t), so matched branches share them."""
    return jax.random.split(jax.random.fold_in(key, t), 5)


def batch_indices(key, t, n_rows):
    return jax.random.randint(step_keys(key, t)[0], (BATCH,), 0, n_rows)


# ---- state ---------------------------------------------------------------------------------------
def init_state(train_seed, obs_dim, action_dim=4):
    """Host networks from PRNGKey(train_seed); scale network exactly as the BCA source initializes it.

    action_dim is 4 for the original testbed. experiments/gridworld_critical_v1 uses 5."""
    ka, k1, k2 = jax.random.split(jax.random.PRNGKey(train_seed), 3)
    actor = init_mlp(ka, (obs_dim, HIDDEN, HIDDEN, action_dim))
    q1 = init_mlp(k1, (obs_dim + action_dim, HIDDEN, HIDDEN, 1))
    q2 = init_mlp(k2, (obs_dim + action_dim, HIDDEN, HIDDEN, 1))
    scale = calibrator(obs_dim).init(
        jax.random.fold_in(jax.random.PRNGKey(700_000 + train_seed), SCALE_INIT_FOLD),
        jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim)))
    support = init_mlp(jax.random.PRNGKey(800_000 + train_seed), (obs_dim + action_dim, HIDDEN, HIDDEN, 1))
    return {"actor": actor, "q1": q1, "q2": q2, "actor_t": actor, "q1_t": q1, "q2_t": q2,
            "actor_opt": _host_opt.init(actor), "critic_opt": _host_opt.init((q1, q2)),
            "scale": scale, "scale_opt": _scale_opt.init(scale), "unit": jnp.float32(1.0),
            "support": support, "support_opt": _support_opt.init(support),
            "reference": blank_reference(scale, support),
            "step": jnp.int32(0)}


def blank_reference(scale, support):
    """Not ready: every width and penalty is zero until the first refresh."""
    return {"scale": scale, "unit": jnp.float32(1.0), "threshold": jnp.float32(0.0), "ready": jnp.asarray(False),
            "support": support, "support_floor": jnp.float32(-1e9), "width_cap": jnp.float32(0.0),
            "width_ref": jnp.float32(0.0), "width_ref_support": jnp.float32(0.0)}


def make_step(observation, n_rows):
    obs_fn, obs_dim = OBSERVATIONS[observation]
    cal = calibrator(obs_dim)

    def step(state, data, key, mode, t):
        k_batch, k_fit, k_prior, k_host, k_perm = step_keys(key, t)
        k_random = jax.random.fold_in(jax.random.fold_in(key, t), SUPPORT_KEY_FOLD)
        arm = switches(mode)
        idx = jax.random.randint(k_batch, (BATCH,), 0, n_rows)
        obs, next_obs = obs_fn(data["s"][idx]), obs_fn(data["s2"][idx])
        action, reward, done = data["a"][idx], data["r"][idx], data["done"][idx]
        targets = (state["actor_t"], state["q1_t"], state["q2_t"])
        reference = state["reference"]
        order = jax.random.permutation(k_perm, BATCH)
        # Where each part of the width is read. A shuffled arm reads both parts at reordered states.
        # The "support_blind_width" arm reorders the states for the calibrated part only.
        blind = jnp.logical_or(arm["shuffled"], arm["blind_base"])
        width_obs = jnp.where(blind, obs[order], obs)
        width_next_obs = jnp.where(blind, next_obs[order], next_obs)
        support_obs = jnp.where(arm["shuffled"], obs[order], obs)
        support_next_obs = jnp.where(arm["shuffled"], next_obs[order], next_obs)

        def target_for(noise):
            return trained_target(cal, arm, reference, *targets, next_obs, width_next_obs, reward, done, noise,
                                  support_next_obs)

        # 1. Fit the scale network on this batch, using the host as it is before its own update.
        fit_target = target_for(jax.random.normal(k_fit, action.shape))
        fit_q = jnp.minimum(critic_apply(state["q1"], obs, action), critic_apply(state["q2"], obs, action))
        prior = jax.random.exponential(k_prior, (BATCH,))
        prior = prior / prior.sum()
        scale_grad = jax.grad(scale_objective)(state["scale"], cal, obs, action, fit_target, fit_q,
                                               state["unit"], prior)
        updates, scale_opt = _scale_opt.update(scale_grad, state["scale_opt"], state["scale"])
        scale = optax.apply_updates(state["scale"], updates)
        unit = unit_update(state["unit"], fit_target, fit_q)

        # 1b. Fit the support classifier. Every arm does this identical work; only some arms consume it.
        random_action = jax.random.uniform(k_random, action.shape, jnp.float32, -1.0, 1.0)
        support_grad = jax.grad(support_objective)(state["support"], obs, action, random_action)
        updates, support_opt = _support_opt.update(support_grad, state["support_opt"], state["support"])
        support = optax.apply_updates(state["support"], updates)

        # 2. Critic update. Only the "full" arms subtract the excess width from the next-state value.
        target = jax.lax.stop_gradient(target_for(jax.random.normal(k_host, action.shape)))
        critics = (state["q1"], state["q2"])
        critic_grad = jax.grad(critic_loss)(critics, obs, action, target)
        updates, critic_opt = _host_opt.update(critic_grad, state["critic_opt"], critics)
        q1, q2 = optax.apply_updates(critics, updates)

        # 3. Actor and target networks, every second update, using the critic just updated
        #    and the reference frozen at the last refresh.
        actor_grad = jax.grad(actor_objective)(state["actor"], q1, cal, reference, obs, width_obs, action,
                                               arm["kappa"], arm["support"], arm["one_sided"], arm["no_base"],
                                               support_obs)
        updates, actor_opt_new = _host_opt.update(actor_grad, state["actor_opt"], state["actor"])
        actor_new = optax.apply_updates(state["actor"], updates)
        due = (t % POLICY_DELAY) == 0

        def choose(new, old):
            return jax.tree_util.tree_map(lambda a, b: jnp.where(due, a, b), new, old)

        actor = choose(actor_new, state["actor"])
        return {"actor": actor, "q1": q1, "q2": q2,
                "actor_t": choose(polyak(actor_new, state["actor_t"]), state["actor_t"]),
                "q1_t": choose(polyak(q1, state["q1_t"]), state["q1_t"]),
                "q2_t": choose(polyak(q2, state["q2_t"]), state["q2_t"]),
                "actor_opt": choose(actor_opt_new, state["actor_opt"]), "critic_opt": critic_opt,
                "scale": scale, "scale_opt": scale_opt, "unit": unit,
                "support": support, "support_opt": support_opt,
                "reference": reference, "step": jnp.int32(t)}

    return step


def make_segment(observation, n_rows):
    """Jitted, vmapped over runs: advance every run through the update numbers in `steps`."""
    step = make_step(observation, n_rows)

    def segment(state, data, key, mode, steps):
        return jax.lax.scan(lambda s, t: (step(s, data, key, mode, t), None), state, steps)[0]

    return jax.jit(jax.vmap(segment, in_axes=(0, 0, 0, 0, None)))


# ---- read-outs used at refreshes and for evaluation ---------------------------------------------------
def make_readouts(observation):
    obs_fn, obs_dim = OBSERVATIONS[observation]
    cal = calibrator(obs_dim)

    def heldout(state, bank, key, mode):
        """On held-out rows: the scale prediction, the residual against the target this arm's critic
        is trained toward, the plain Bellman target, and the support score of the recorded command."""
        arm = switches(mode)
        obs, next_obs = obs_fn(bank["s"]), obs_fn(bank["s2"])
        targets = (state["actor_t"], state["q1_t"], state["q2_t"])
        noise = jax.random.normal(key, bank["a"].shape)
        order = jax.random.permutation(jax.random.fold_in(key, 1), bank["s"].shape[0])
        width_next_obs = jnp.where(arm["shuffled"], next_obs[order], next_obs)
        plain = bellman_target(*targets, next_obs, bank["r"], bank["done"], noise)
        trained = trained_target(cal, arm, state["reference"], *targets, next_obs, width_next_obs,
                                 bank["r"], bank["done"], noise)
        q = jnp.minimum(critic_apply(state["q1"], obs, bank["a"]), critic_apply(state["q2"], obs, bank["a"]))
        return (cal.apply(state["scale"], obs, bank["a"]), trained - q, plain,
                support_logit(state["support"], obs, bank["a"]))

    def audit(state, bank, key, mode):
        """Reporting only: the width this arm consumes at recorded commands, the one-step residual,
        and the frozen support score (to be compared with the frozen floor)."""
        _, residual, _, _ = heldout(state, bank, key, mode)
        obs, arm = obs_fn(bank["s"]), switches(mode)
        return (consumed_width(cal, state["reference"], obs, bank["a"], arm["support"], arm["no_base"]), residual,
                support_logit(state["reference"]["support"], obs, bank["a"]))

    def policy(state, cells):
        return actor_apply(state["actor"], obs_fn(cells))

    def probe(state, cells, commands, mode):
        """For every (cell, command) pair: Q1, Q2, the original width, the width this arm consumes,
        and the support score. Arrays of shape [cells, commands]."""
        obs = jnp.repeat(obs_fn(cells)[:, None, :], commands.shape[0], axis=1)
        act = jnp.repeat(commands[None, :, :], cells.shape[0], axis=0)
        flat_obs, flat_act, shape = obs.reshape(-1, obs_dim), act.reshape(-1, act.shape[-1]), act.shape[:2]
        reference, arm = state["reference"], switches(mode)
        return (critic_apply(state["q1"], obs, act), critic_apply(state["q2"], obs, act),
                width(cal, reference, flat_obs, flat_act).reshape(shape),
                consumed_width(cal, reference, flat_obs, flat_act, arm["support"], arm["no_base"]).reshape(shape),
                support_logit(reference["support"], flat_obs, flat_act).reshape(shape))

    return {"heldout": jax.jit(jax.vmap(heldout, in_axes=(0, 0, 0, 0))),
            "audit": jax.jit(jax.vmap(audit, in_axes=(0, 0, 0, 0))),
            "policy": jax.jit(jax.vmap(policy, in_axes=(0, None))),
            "probe": jax.jit(jax.vmap(probe, in_axes=(0, None, None, 0)))}


# ---- helpers for stacks of runs -------------------------------------------------------------------------
def stack(trees):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)


def take(tree, i):
    return jax.tree_util.tree_map(lambda x: x[i], tree)


def repeat(tree, times):
    """Copy a stack of runs `times` times along the run axis: run order is [copy 0 runs..., copy 1 runs...]."""
    return jax.tree_util.tree_map(lambda x: jnp.concatenate([x] * times, axis=0), tree)
