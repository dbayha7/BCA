"""Exact values by float64 simulation (no learning anywhere in this module).

The dynamics are deterministic, so for any deterministic policy pi (the MLP actors included, with weights cast to
float64) the true action value is one float64 rollout through the same dynamics() as env.step:
  Q^pi(s, a) = r(s, a) + sum_{t=1}^{H-1} 0.98^t r(s_t, pi(s_t)),   H = 1,000 (truncation 0.98^1000 ~ 1.7e-9 relative).
q_value is the numpy form; q_value_jax runs the same arithmetic in JAX under jax.experimental.enable_x64 (never
switched on globally, so learners in the same process keep float32), and action_slope gives grad_a Q^pi by
forward-mode autodiff through that rollout (exact except at the kinks at 80% throttle and at the throttle limits,
which have measure zero).

Per-row quantities (callers pass the float64 of the float32 inputs the critic sees):
  over_preference   D_i = [(Q1(s, pi(s)) - Q1(s, a_i)) - (Q^pi(s, pi(s)) - Q^pi(s, a_i))]_+ : how much more the critic
                    prefers the policy's command over the logged one than the truth does (what extra cloning at row
                    i corrects; the primary harm label).
  caution_benefit   c_i = -grad_a Q^pi(s, pi(s)) . grad_a Q1(s, pi(s)) (first-order effect on the true value of
                    raising row i's cloning weight).
  advantage_error   e_A(s, a) = e(s, a) - e(s, pi(s)) with e = Q1 - Q^pi (the level cancels).
  pull_past_80      o(s) = [Q1(s, c_hi) - Q1(s, c_cap)] - [Q^pi(s, c_hi) - Q^pi(s, c_cap)]: how much more the critic
                    prefers pushing past 80% than the truth does.
  noise_map         the SD of TD3's smoothed target under a perfect critic: 0.98 x the SD, over 16 draws of
                    noise = clip(N(0, 0.2^2), +-0.5), of Q(s', clip(c(s') + noise, -1, 1)), where c is the
                    stall-aware controller, Q its exact action value and s' the row's next state. Fixed per row before
                    training. A control: it carries no information about critic error.
  smoothed_q        TD3's own fixed point: Q of pi with clipped target noise on every action from step 1 on, by
                    Monte Carlo (64 paths, common random numbers), with its SE. Diagnostics and forecasts only.
Returns and points: the undiscounted 100-step return from each bank start is exact; points = 100 (J - J_recording) /
(J_top - J_recording) per situation, J_top the stall-aware controller in the stalling world and the rising-thrust
controller in the no-stall world (anchors()).
Attribution (attribution): exact finite-horizon performance difference along the new policy's own trajectory,
  J_new - J_ref = sum_t [Q_ref^(100-t)(s_t, pi_new(s_t)) - Q_ref^(100-t)(s_t, pi_ref(s_t))],
each term an undiscounted rollout of the reference policy; the terms sum to the return difference within 1e-8.
Closed form (closed_form_check): in the no-stall world a linear policy that never saturates has rollout Q equal to
lq_harness.quadratic_q's quadratic (the check tracks the policy's unclipped output along the whole continuation and
requires it to stay strictly inside [-1, 1]); closed_form_smoothed_q gives TD3's smoothed level there in closed form.
"""

import math
from dataclasses import dataclass

import numpy as np

from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC
from experiments.signal import gapped_lq as GLQ
from experiments.signal import lq_harness as LQ

HORIZON = 1000
NOISE_MAP_HORIZON = 600
TD3_NOISE, TD3_CLIP = 0.2, 0.5
NOISE_MAP_SEED = (TC.DATA_SEED, 11)
SMOOTH_SEED = 21


def _act(policy, x, t):
    act = getattr(policy, "act", None)
    return act(x, t) if act is not None else policy(x)


def _deterministic(policy):
    if isinstance(policy, RC.StreamedRecorder):
        raise TypeError("the true Q is defined for deterministic policies; the recording controller is random")
    return policy


@dataclass
class Run:
    """A rollout: J (undiscounted return per start), Jd (discounted), and with record=True the states (N, L + 1,
    13), the applied (clipped) commands (N, L, 4) and the rewards (N, L)."""

    J: np.ndarray
    Jd: np.ndarray
    states: np.ndarray = None
    commands: np.ndarray = None
    rewards: np.ndarray = None


def rollout(policy, x0, physics=TC.STALL, length=None, record=False, gamma=None):
    """Roll a policy (numpy controller; a recorder.StreamedRecorder is stepped with its streams) from starts x0."""
    L = physics.length if length is None else int(length)
    g = physics.gamma if gamma is None else float(gamma)
    x = np.asarray(x0, np.float64).copy()
    J, Jd, disc = np.zeros(len(x)), np.zeros(len(x)), 1.0
    S, C, R = [x], [], []
    for t in range(L):
        a = np.clip(np.asarray(_act(policy, x, t), np.float64), -1.0, 1.0)
        x2, r = TC.dynamics(x, a, physics)
        J = J + r
        Jd = Jd + disc * r
        disc *= g
        if record:
            S.append(x2), C.append(a), R.append(r)
        x = x2
    if not record:
        return Run(J, Jd)
    return Run(J, Jd, np.stack(S, 1), np.stack(C, 1), np.stack(R, 1))


def q_value(policy, x, a, physics=TC.STALL, horizon=HORIZON, gamma=None):
    """Q^pi(x, a) by one float64 rollout of `horizon` steps (numpy)."""
    _deterministic(policy)
    g = physics.gamma if gamma is None else float(gamma)
    xx, q = TC.dynamics(np.asarray(x, np.float64), np.asarray(a, np.float64), physics)
    disc = 1.0
    for _ in range(1, int(horizon)):
        disc *= g
        xx, rr = TC.dynamics(xx, policy(xx), physics)
        q = q + disc * rr
    return q


def value(policy, x, physics=TC.STALL, horizon=HORIZON, gamma=None):
    return q_value(policy, x, policy(np.asarray(x, np.float64)), physics, horizon, gamma)


def finite_return(policy, x, steps, physics=TC.STALL):
    """Undiscounted return of `steps` steps of a deterministic policy from x (V^(steps)); 0 for steps = 0."""
    _deterministic(policy)
    x = np.asarray(x, np.float64)
    total = np.zeros(x.shape[:-1])
    for _ in range(int(steps)):
        x, r = TC.dynamics(x, policy(x), physics)
        total = total + r
    return total


def bellman_residual(policy, x, a, physics=TC.STALL, horizon=HORIZON):
    """max |Q_H(x, a) - (r(x, a) + gamma Q_{H-1}(x', pi(x')))|: zero up to rounding by construction."""
    x1, r = TC.dynamics(np.asarray(x, np.float64), np.asarray(a, np.float64), physics)
    lhs = q_value(policy, x, a, physics, horizon)
    rhs = r + physics.gamma * q_value(policy, x1, policy(x1), physics, horizon - 1)
    return float(np.abs(lhs - rhs).max())


# ------------------------------------------------------------------------------------------------ JAX

def _jax_q(policy, physics, horizon, gamma):
    import jax
    import jax.numpy as jnp

    def q(x, a):
        x1, r0 = TC.dynamics_jax(x, a, physics)

        def body(carry, _):
            xx, acc, disc = carry
            disc = disc * gamma
            xx2, rr = TC.dynamics_jax(xx, policy.jax(xx), physics)
            return (xx2, acc + disc * rr, disc), None

        (_, acc, _), _ = jax.lax.scan(body, (x1, r0, jnp.asarray(1.0, x.dtype)), None, length=int(horizon) - 1)
        return acc

    return q


def _cached(policy, key, build):
    cache = policy.__dict__.setdefault("_truth_jit", {})
    if key not in cache:
        cache[key] = build()
    return cache[key]


def _precision(x64):
    """The JAX precision context: enable_x64 for float64 (only inside the call), nothing for float32."""
    import contextlib

    from jax.experimental import enable_x64

    return (enable_x64() if x64 else contextlib.nullcontext()), (np.float64 if x64 else np.float32)


def q_value_jax(policy, x, a, physics=TC.STALL, horizon=HORIZON, gamma=None, chunk=None, x64=True):
    """q_value in JAX. x64=True (default): float64, with enable_x64 only inside this call. x64=False: a float32 fast
    path (relative error about 1e-6; for repeated oracle refreshes, never for the exactness checks). chunk: rows per
    call (None: all)."""
    import jax
    import jax.numpy as jnp

    _deterministic(policy)
    g = physics.gamma if gamma is None else float(gamma)
    context, dtype = _precision(x64)
    x, a = np.asarray(x, dtype), np.asarray(a, dtype)
    chunk = chunk or len(x)
    out = []
    with context:
        fn = _cached(policy, ("q", physics, int(horizon), g, bool(x64)),
                     lambda: jax.jit(_jax_q(policy, physics, horizon, g)))
        for i in range(0, len(x), chunk):
            out.append(np.asarray(fn(jnp.asarray(x[i:i + chunk]), jnp.asarray(a[i:i + chunk])), np.float64))
    return np.concatenate(out) if out else np.zeros(0)


def action_slope(policy, x, a, physics=TC.STALL, horizon=HORIZON, gamma=None, chunk=None, x64=True):
    """grad_a Q^pi(x, a) (N, 4) by forward-mode autodiff (one JVP per thruster) through the rollout (float64 unless
    x64=False)."""
    import jax
    import jax.numpy as jnp

    _deterministic(policy)
    g = physics.gamma if gamma is None else float(gamma)
    context, dtype = _precision(x64)
    x, a = np.asarray(x, dtype), np.asarray(a, dtype)
    chunk = chunk or len(x)

    def build():
        q = _jax_q(policy, physics, horizon, g)

        def slopes(xx, aa):
            tangents = jnp.eye(TC.DA, dtype=aa.dtype)[:, None, :] * jnp.ones_like(aa)[None]
            return jax.vmap(lambda tt: jax.jvp(lambda u: q(xx, u), (aa,), (tt,))[1])(tangents).T

        return jax.jit(slopes)

    out = []
    with context:
        fn = _cached(policy, ("slope", physics, int(horizon), g, bool(x64)), build)
        for i in range(0, len(x), chunk):
            out.append(np.asarray(fn(jnp.asarray(x[i:i + chunk]), jnp.asarray(a[i:i + chunk])), np.float64))
    return np.concatenate(out) if out else np.zeros((0, TC.DA))


class MLPPolicy(RC.Controller):
    """The TD3+BC actor (algorithms/td3_bc.Actor: 13-256-256-4, ReLU hidden layers, tanh output times max_action) with
    its weights cast to float64: pi(x) = max_action tanh(W3 relu(W2 relu(W1 z + b1) + b2) + b3), z = (o - mean) / std.
    With round_obs (default) o is the float32 observation the environment would return (float32(x) as float64), so
    the policy sees what the deployed actor sees. params: the flax params ({'params': {'Dense_0': ...}} or the inner
    dict); obs_mean / obs_std: the learner's state normalisation."""

    def __init__(self, params, obs_mean, obs_std, max_action=1.0, round_obs=True, name="MLP actor (float64)"):
        p = params.get("params", params)
        layers = sorted(p, key=lambda k: int(str(k).rsplit("_", 1)[1]))
        if len(layers) != 3:
            raise ValueError("expected the three dense layers of algorithms/td3_bc.Actor")
        self.W = [np.asarray(p[k]["kernel"], np.float64) for k in layers]
        self.b = [np.asarray(p[k]["bias"], np.float64) for k in layers]
        self.mean = np.asarray(obs_mean, np.float64)
        self.std = np.asarray(obs_std, np.float64)
        self.max_action = float(max_action)
        self.round_obs = bool(round_obs)
        self.name = name

    def __call__(self, x):
        o = np.asarray(x, np.float64)
        if self.round_obs:
            o = o.astype(np.float32).astype(np.float64)
        h = (o - self.mean) / self.std
        h = np.maximum(h @ self.W[0] + self.b[0], 0.0)
        h = np.maximum(h @ self.W[1] + self.b[1], 0.0)
        return self.max_action * np.tanh(h @ self.W[2] + self.b[2])

    def jax(self, x):
        import jax.numpy as jnp

        o = x.astype(jnp.float32).astype(x.dtype) if self.round_obs else x
        h = (o - jnp.asarray(self.mean, x.dtype)) / jnp.asarray(self.std, x.dtype)
        h = jnp.maximum(h @ jnp.asarray(self.W[0], x.dtype) + jnp.asarray(self.b[0], x.dtype), 0.0)
        h = jnp.maximum(h @ jnp.asarray(self.W[1], x.dtype) + jnp.asarray(self.b[1], x.dtype), 0.0)
        return self.max_action * jnp.tanh(h @ jnp.asarray(self.W[2], x.dtype) + jnp.asarray(self.b[2], x.dtype))


# ------------------------------------------------------------------------------------------------ per-row labels

def true_preference(policy, states, logged_actions, physics=TC.STALL, horizon=HORIZON, use_jax=False):
    """(Q^pi(s, pi(s)), Q^pi(s, a_logged)) at the rows."""
    s = np.asarray(states, np.float64)
    pi_s = policy(s)
    q = q_value_jax if use_jax else q_value
    return q(policy, s, pi_s, physics, horizon), q(policy, s, np.asarray(logged_actions, np.float64), physics, horizon)


def over_preference(critic_at_policy, critic_at_logged, true_at_policy, true_at_logged):
    """D_i = [(Q1(s, pi(s)) - Q1(s, a)) - (Q^pi(s, pi(s)) - Q^pi(s, a))]_+ (the module docstring)."""
    d = (np.asarray(critic_at_policy) - np.asarray(critic_at_logged)) - (np.asarray(true_at_policy)
                                                                        - np.asarray(true_at_logged))
    return np.maximum(d, 0.0)


def caution_benefit(true_slope_at_policy, critic_slope_at_policy):
    """c_i = -grad_a Q^pi(s, pi(s)) . grad_a Q1(s, pi(s))."""
    return -np.sum(np.asarray(true_slope_at_policy) * np.asarray(critic_slope_at_policy), axis=-1)


def advantage_error(critic_at_action, critic_at_policy, true_at_action, true_at_policy):
    """e_A(s, a) = e(s, a) - e(s, pi(s)), e = Q1 - Q^pi."""
    return (np.asarray(critic_at_action) - np.asarray(true_at_action)) - (np.asarray(critic_at_policy)
                                                                         - np.asarray(true_at_policy))


def pull_past_80(critic_hi, critic_cap, true_hi, true_cap):
    """o(s) = [Q1(s, c_hi) - Q1(s, c_cap)] - [Q^pi(s, c_hi) - Q^pi(s, c_cap)]."""
    return (np.asarray(critic_hi) - np.asarray(critic_cap)) - (np.asarray(true_hi) - np.asarray(true_cap))


def noise_map(states, actions, physics=TC.STALL, policy=None, draws=16, noise_sd=TD3_NOISE, noise_clip=TD3_CLIP,
              horizon=NOISE_MAP_HORIZON, seed=NOISE_MAP_SEED):
    """Per row: the SD of TD3's smoothed target under a perfect critic of the stall-aware controller (docstring)."""
    policy = RC.stall_aware(physics) if policy is None else policy
    x1, _ = TC.dynamics(np.asarray(states, np.float64), np.asarray(actions, np.float64), physics)
    base = policy(x1)
    eps = np.clip(noise_sd * np.random.default_rng(list(seed)).standard_normal((draws,) + base.shape),
                  -noise_clip, noise_clip)
    tq = np.stack([q_value(policy, x1, np.clip(base + eps[i], -1.0, 1.0), physics, horizon) for i in range(draws)])
    return physics.gamma * tq.std(0)


def smoothed_q(policy, x, a, physics=TC.STALL, paths=64, noise_sd=TD3_NOISE, noise_clip=TD3_CLIP,
               horizon=NOISE_MAP_HORIZON, seed=SMOOTH_SEED):
    """TD3's smoothed level: Q of pi with clip(N(0, noise_sd^2), +-noise_clip) added to every action from step 1 on
    (then clipped to [-1, 1]); Monte Carlo over `paths` paths with common random numbers default_rng([seed, path]).
    Returns (mean, SE) per row."""
    _deterministic(policy)
    x, a = np.asarray(x, np.float64), np.asarray(a, np.float64)
    out = []
    for i in range(paths):
        rg = np.random.default_rng([seed, i])
        z, q = TC.dynamics(x, a, physics)
        disc = 1.0
        for _ in range(1, horizon):
            disc *= physics.gamma
            act = np.clip(policy(z) + np.clip(noise_sd * rg.standard_normal(z.shape[:-1] + (TC.DA,)), -noise_clip,
                                              noise_clip), -1.0, 1.0)
            z, rr = TC.dynamics(z, act, physics)
            q = q + disc * rr
        out.append(q)
    out = np.stack(out)
    return out.mean(0), out.std(0) / math.sqrt(paths)


# ------------------------------------------------------------------------------------------------ returns, points

def bank_rollout(policy, situation, physics=TC.STALL, n=TC.BANK_SIZE, record=False):
    return rollout(policy, TC.bank_starts(situation, n), physics, record=record)


def recorder_on_bank(physics=TC.STALL, spec=RC.DEFAULT_SPEC, n=TC.BANK_SIZE):
    """The recording controller with its fixed per-bank streams (recorder.RECORDING_BANK_STREAM)."""
    return RC.StreamedRecorder.on_bank(RC.Recorder(spec, physics), n=n, length=physics.length)


def anchors(physics=TC.STALL, spec=RC.DEFAULT_SPEC, situations=TC.SITUATIONS, n=TC.BANK_SIZE):
    """Per situation: the recording controller's and the top controller's mean return on the bank (0 and 100
    points), with bank standard errors."""
    out = {}
    top = RC.top_anchor(physics)
    for s in situations:
        jo = bank_rollout(recorder_on_bank(physics, spec, n), s, physics, n).J
        jt = bank_rollout(top, s, physics, n).J
        out[s] = dict(recording=float(jo.mean()), top=float(jt.mean()), recording_se=float(jo.std() / math.sqrt(n)),
                      top_se=float(jt.std() / math.sqrt(n)), top_controller=top.name)
    return out


def points(J, anchor):
    """100 (J - J_recording) / (J_top - J_recording); J may be an array (per start) or a mean."""
    return 100.0 * (np.asarray(J, np.float64) - anchor["recording"]) / (anchor["top"] - anchor["recording"])


# ------------------------------------------------------------------------------------------------ attribution

def attribution(policy_new, policy_ref, x0, physics=TC.STALL, length=None):
    """Exact per-step performance-difference terms along policy_new's trajectories (module docstring).

    W[:, t] = V_ref^(L - t)(s_t) (undiscounted, W[:, L] = 0) is computed for every visited state in one batch whose
    rows drop out as their horizon ends; term_t = r_t + W[:, t + 1] - W[:, t]. Returns the terms (N, L), the new
    trajectory (states, commands), J_new, J_ref and the largest |sum_t term_t - (J_new - J_ref)|."""
    _deterministic(policy_new)
    _deterministic(policy_ref)
    L = physics.length if length is None else int(length)
    run = rollout(policy_new, x0, physics, L, record=True)
    N = len(run.J)
    Z = run.states[:, :L].transpose(1, 0, 2).copy()
    acc = np.zeros((L, N))
    for k in range(L):
        active = L - k
        z = Z[:active].reshape(-1, TC.DX)
        z2, r = TC.dynamics(z, policy_ref(z), physics)
        acc[:active] += r.reshape(active, N)
        Z[:active] = z2.reshape(active, N, TC.DX)
    W = np.zeros((N, L + 1))
    W[:, :L] = acc.T
    terms = run.rewards + W[:, 1:] - W[:, :-1]
    J_ref = W[:, 0]
    error = float(np.abs(terms.sum(1) - (run.J - J_ref)).max())
    return dict(terms=terms, states=run.states, commands=run.commands, J_new=run.J, J_ref=J_ref, max_sum_error=error)


# ------------------------------------------------------------------------------------------------ closed form

def linear_system(physics=TC.NO_STALL, action_noise_var=0.0):
    """lq_harness.System of the rising-thrust model (exact in the no-stall world while no command saturates):
    state cost I on the floats, 0 on current and wave; throttle cost 0.3 I; process noise B (var I) B^T."""
    A, B = TC.matrices(physics)
    Qc = np.diag([1.0] * TC.DS + [0.0] * TC.DE)
    noise = float(action_noise_var) * B @ B.T
    return LQ.System(np.array(A), np.array(B), Qc, physics.throttle_cost * np.eye(TC.DA), noise, np.eye(TC.DX),
                     physics.gamma, physics.length)


def closed_form_q(K, x, a, physics=TC.NO_STALL):
    """Q of the linear policy a = -K x at (x, a) from lq_harness.quadratic_q (no noise): z^T H z, z = (x, a)."""
    system = linear_system(physics)
    H, h, _ = LQ.quadratic_q(system, K, LQ.reward_form(system))
    z = np.concatenate([np.asarray(x, np.float64), np.asarray(a, np.float64)], -1)
    return np.einsum("ni,ij,nj->n", z, H, z) + h


def closed_form_smoothed_q(K, x, a, physics=TC.NO_STALL, noise_sd=TD3_NOISE, noise_clip=TD3_CLIP):
    """TD3's smoothed level of a = -K x in closed form (valid while nothing clips): actions from step 1 on carry
    clipped noise of per-thruster variance v (gapped_lq.clipped_noise_variance); with P the noise-free value matrix,
    Q_s(x, a) = z^T H z + gamma c, c = (gamma tr(P B v B^T) - v tr(R)) / (1 - gamma)."""
    v = GLQ.clipped_noise_variance(noise_sd, noise_clip)
    system = linear_system(physics)
    H, h, P = LQ.quadratic_q(system, K, LQ.reward_form(system))
    _, B = TC.matrices(physics)
    g = physics.gamma
    c = (g * v * float(np.trace(P @ B @ B.T)) - v * physics.throttle_cost * TC.DA) / (1 - g)
    z = np.concatenate([np.asarray(x, np.float64), np.asarray(a, np.float64)], -1)
    return np.einsum("ni,ij,nj->n", z, H, z) + h + g * c


CLOSED_FORM_STARTS = dict(current=(0.0, 0.3), wave=(0.0, 0.2), kick=0.2)


def closed_form_check(n=400, seed=1, horizon=2000, law=None):
    """Rollout Q vs the closed form in the no-stall world for the unclipped textbook policy, from small starts
    (CLOSED_FORM_STARTS: current <= 0.3, wave <= 0.2, float SD 0.2, unless `law` is given) and first commands near
    it (clipped to [-1, 1]; the closed form is evaluated at the same clipped command, so clipping it is harmless).

    The check is exact only while the continuation never saturates, so the guard tracks the policy's own unclipped
    output pi(x_t) at every one of the horizon - 1 continuation steps that the rollout Q sums (before dynamics()
    clips it). Returns the largest absolute and relative gaps, max_continuation_command (the largest |pi(x_t)| of
    the continuation), continuation_inside (max_continuation_command < 1: the check is exact), and, reported apart,
    the first commands' largest raw size and the share of rows whose first command was clipped."""
    K = RC.textbook_gain(TC.NO_STALL)
    pol = RC.LinearController(K, None, "textbook gain, unclipped")
    rng = np.random.default_rng(seed)
    xs = TC.draw_starts(n, rng, CLOSED_FORM_STARTS if law is None else law)
    raw = pol(xs) + 0.1 * np.random.default_rng(seed + 1).standard_normal((n, TC.DA))
    a0 = np.clip(raw, -1, 1)
    qroll = q_value(pol, xs, a0, TC.NO_STALL, horizon)
    qcf = closed_form_q(K, xs, a0)
    x, largest = TC.dynamics(xs, a0, TC.NO_STALL)[0], 0.0
    for _ in range(1, int(horizon)):
        u = pol(x)
        largest = max(largest, float(np.abs(u).max()))
        x = TC.dynamics(x, u, TC.NO_STALL)[0]
    gap = np.abs(qroll - qcf)
    return dict(max_abs=float(gap.max()), max_rel=float(gap.max() / np.abs(qcf).mean()),
                max_continuation_command=largest, continuation_inside=bool(largest < 1.0),
                first_command_raw_max=float(np.abs(raw).max()),
                first_command_clipped_share=float(np.mean(np.any(np.abs(raw) > 1.0, axis=1))), rows=n,
                horizon=int(horizon))
