"""Linear-quadratic harness with exact Q: does BCA measure, and usefully apply, the error TD3+BC's actor climbs?

Step 3 of the signal study; expectations, built-in checks and decision rules are pre-registered in
runs/wbcp_signal/expectations.md (Step 3 and Addendum A1). This module builds and smoke-tests the harness;
it draws no conclusions.

MDP. s' = A s + B a + eps, eps ~ N(0, Sigma_eps); reward r = -(s^T Qc s + a^T Rc a); discount gamma; episodes
of length L from s0 ~ N(0, Sigma_0), truncated and bootstrapped (done = 0). Nothing is clipped: TD3's
target-action clip receives max_action = inf, the actor is linear (no tanh), and the gains keep actions
mostly inside [-1, 1] (the share outside is reported), so Q^pi below is exact for every action. For a
per-step reward z^T G z with z = (s, a) and the policy a = -K s (quadratic_q):
    V(s) = s^T P s + h,  P = Pi^T G Pi + gamma M^T P M  (M = A - B K, Pi = [I; -K], discrete Lyapunov),
    h = gamma tr(P Sigma_eps) / (1 - gamma),  Q(s, a) = z^T H z + h,  H = G + gamma [A B]^T P [A B].
With G = -blockdiag(Qc, Rc) this is the specification's Q^pi with P_spec = -P and c = h; J(pi) = tr(P Sigma_0)
+ h. test_lq_harness checks Q^pi and J against Monte Carlo rollouts and the Bellman identity.

Data. Behaviour beta(s) = -K_b s + N(0, sigma_b^2 I): 'good' K_b = K* (discounted LQR optimum, J = -2.30),
'poor' K_b = poor_gain K* (default 0: no feedback, J = -4.43; negative gains destabilise this A). Whole
episodes split into training, held-out calibration (a bank of K rows per episode drawn with
calibration/bank.stratified_bank, BCA's sampler) and evaluation. The evaluated actor pi is linear with gain
K_pi = K_b + pi_offset D (D a fixed unit-Frobenius direction): a perturbed BC of beta. The default
pi_offsets 0.5, 1 and 2 give a mean |pi(s) - beta(s)| of 0.07, 0.14, 0.29 (good) and 0.10, 0.20, 0.41 (poor),
against 0.25 for a logged action; every such closed loop is stable, and at most 1.2% of pi's actions leave
[-1, 1]. Expectation 3 of Step 3 (pi(s) is missed more often than logged actions, 'more so the further pi is
from beta') needs >= 2 offsets or the binned metric (item 3 below): with one offset per behaviour the
distance is confounded with behaviour quality. run() refuses, before any compute, a behaviour or pi_offset
whose closed loop is not discounted-stable (J = -inf).

Critics (frozen; the target critic and target actor equal the critic and pi). Heads
Q_k = Q^pi + kappa_k ||a - beta(s)||^2 + z^T W_k z + w_k, computed as Q^pi plus an error term:
- clean: exact heads, noiseless reward (control);
- noisy_reward: exact heads, observed reward r + N(0, sigma_r^2);
- q1_optimistic: kappa = (kappa, 0), optimism in the head the actor climbs, away from the data. Q^pi's
  action block H_aa has eigenvalues -1.43 to -3.86 over the default offsets and behaviours (-1.55 to -2.5 at
  offset 1), so Q1 stays concave in a while kappa < about 1.4;
- q2_optimistic: kappa = (0, kappa);
- shared_bias: both heads carry b = Q^pi of the shifted reward dr = kappa ||a - beta(s)||^2, i.e. what a
  critic trained on r + dr would converge to. b(s,a) = dr(s,a) + gamma E b(s', pi(s')), so the expected TD
  residual is shifted by -dr(s,a) only: the bootstrapped part gamma E V_b(s') of the bias, which carries the
  future deviation of pi from beta, is invisible to any TD residual. Any bias consistent with a Bellman
  equation for a shifted reward shows in a TD residual only as its reward shift;
- independent_errors (Addendum A1): W_1, W_2 independent symmetric Gaussian matrices of unit Frobenius norm,
  times one scale c set by bisection so that median |Q1 - Q2| = independent_ratio (default 1, as A1) x
  median |t - min(Q1, Q2)| at logged training actions (independent_scale). The ratio rises with c but
  saturates, as the errors come to dominate t - min, at a level that depends on the draw (0.74-1.33 on six
  trial draws). A draw that cannot reach the target by c = 1e3 is replaced by the next seeded draw
  (independent_errors); the attempt count, c and the achieved ratio are reported. At ratio 1 the errors are
  several times the Bellman noise.
An error that is >= 0 by construction is added to Q^pi in float32, so min(Q1, Q2) is bitwise Q^pi in
q1_optimistic and bitwise Q1 in q2_optimistic (built-in checks 3 and 4).

Signals (signal_q). t = native_target (TD3+BC's smoothed target, the repository function). 'min': t - min(Q1,
Q2), BCA today; 'q1': t - Q1; 'maxabs': max(|t - Q1|, |t - Q2|), realised as t - q' with q' the head farther
from t (ties go to Q1). Soft coverage is symmetric in the residual's sign, so this is exactly the max-score
objective, and the residual unit u is std(t - q'). The signal is a traced integer: every case and signal
runs one compiled program, so equal residuals give bitwise-equal fits (built-in check 2).

Per behaviour x case x signal x replicate:
1. sigma: fit_scale_signal, a line-for-line copy of algorithms/td3_bc_bca.fit_scale that differs only in q
   (tested equal to the repository function for min and, with a Q1-only critic, for q1), for fit_steps of
   batch_size rows on the training split. Batches, keys and the calibrator initialisation are shared by
   every case and signal.
2. WBCP (uniform weights, alpha 0.1, beta 0.95, 1,000 draws) on the calibration bank via freeze_reference,
   with td3_bc_bca.refresh's key folding (calibrate_signal; tested equal to refresh for min).
3. Miscoverage on the evaluation split: the share of rows whose residual exceeds threshold x sigma, for the
   signal's own residual, for each head's residual, and for the true Q1 error |Q1 - Q^pi| at logged actions
   and at pi(s) (sigma evaluated at (s, pi(s)); only the oracle can label that action). The true-Q1-error
   misses are also reported by terciles of the per-row distance |pi(s) - beta_mean(s)| on the evaluation
   split (distance_terciles: rank-based, equal sizes, t1 nearest; mean distance per tercile among the setting
   diagnostics): miss_q1err_pi_t{k} and, on the same rows, miss_q1err_logged_t{k}. As pi(s) - beta_mean(s) =
   -pi_offset D s, the terciles are the same rows at every nonzero offset (arbitrary at offset 0, where every
   distance is 0), and they also sort rows by the state's size along D, which residuals and sigma grow with;
   the logged-action miss in the same terciles is the control for that.
4. Dose: td3_bc_bca.bc_readout (frozen_level_dose, blend 0.5) on every training row; mean, SD, 5th/95th
   percentiles, and its correlation with |Q1 - Q^pi| at the logged action and at pi(s).
5. Actor: actor_steps calls of algorithms/td3_bc.td3_bc_update from K_pi (Adam, lr 3e-4, alpha 2.5; the
   critic, target critic and target actor are restored after each call, so only the actor learns), the dose
   as bc_multiplier, on shared training batches. Arms: bca; none (bc_multiplier=None, the host); constant
   (every row at the mean dose); shuffled (the doses permuted across rows by one shared permutation); oracle
   (|Q1 - Q^pi| at the logged action, rescaled to the mean dose; the constant dose when that error is zero).
   Reported: the exact J(pi') - J(pi), the cosine between the update and the exact policy gradient of J,
   and at K_pi the Frobenius norms of the Q-term and BC-term gradients with respect to K (actor_terms, the
   two terms of td3_bc's actor loss). Alignment between Q1 and Q^pi: alignment_K, the cosine of the two
   Q-term K-gradients E[grad_a Q(s, pi(s)) (-s)^T]; alignment_rowwise, the mean per-row cosine of grad_a; and
   alignment_action, the cosine of the mean grad_a as specified. States here are zero-mean and grad_a is
   linear in s, so both means in alignment_action are sampling noise; alignment_K is the informative one.
6. R replicates (data, keys, error matrices); means and standard errors over the finite values, and paired
   BCA-minus-control J differences. results.md marks a mean that excludes values with '(n=..., k nonfinite)'
   (an undefined cosine or correlation, or J of an unstable closed loop; plain 'n/a' when a quantity is
   undefined or not applicable in every replicate, never when one is infinite); results.json is strict JSON
   (allow_nan=False), so a stray non-finite number fails the write instead of emitting 'Infinity'.

TD3's target smoothing biases t: with exact heads, E[t | s, a] - Q^pi(s, a) = gamma tr(H_aa Cov(n)) < 0 for the
clipped target noise n (the linear term vanishes since E n = 0), a constant for every row; about -0.12 here
against a residual SD near 0.2 (mean_residual_own in clean). Error heads add their own smoothing terms.

JAX_PLATFORMS=cpu python experiments/signal/lq_harness.py --output runs/wbcp_signal/lq/<name>
    [--cases ...] [--signals min q1 maxabs] [--behavior good poor] [--replicates R] [--kappa ...]
    [--pi-offset ...]
Writes results.json and results.md into a new directory. Tests: python -m unittest experiments.signal.test_lq_harness
"""

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
from scipy import linalg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

import algorithms.td3_bc as BASE  # noqa: E402
import algorithms.td3_bc_bca as P  # noqa: E402
from calibration import bank  # noqa: E402
from calibration.dose import tree_finite  # noqa: E402
from calibration.network import Calibrator, bayesian_bootstrap_weights, soft_coverage  # noqa: E402
from calibration.reference import WBCPConfig, freeze_reference, initial_reference, positive_scale  # noqa: E402
from runtime.networks import Transition  # noqa: E402

SIGNALS = ("min", "q1", "maxabs")
CASES = ("clean", "noisy_reward", "shared_bias", "q1_optimistic", "q2_optimistic", "independent_errors")
KAPPA_CASES = ("shared_bias", "q1_optimistic", "q2_optimistic")
BEHAVIORS = ("good", "poor")
ARMS = ("bca", "none", "constant", "shuffled", "oracle")
# td3_bc_bca's key folds: calibrator init, fit target noise, fit bootstrap prior, refresh noise, WBCP draws.
KEY_INIT, KEY_FIT_TARGET, KEY_PRIOR, KEY_REFRESH_NOISE, KEY_WBCP = (
    1128352841, 1179210836, 1128352850, 1213156420, 1347375956)
MAX_ACTION = math.inf  # no clipping anywhere, so Q^pi is exact for every action
DIRECTION = np.array([[1.0, -1.0, 0.5], [0.5, 1.0, -1.0]]) / math.sqrt(4.5)  # unit-Frobenius K perturbation
DEFAULT_KAPPAS = (0.5, 1.0, 2.0)
DEFAULT_OFFSETS = (0.5, 1.0, 2.0)  # >= 2 values within a behaviour: Step 3 expectation 3's distance clause
N_TERCILES = 3


# ---------------------------------------------------------------------------------------------------------
# The LQ system and its exact quantities (NumPy, float64)


@dataclass(frozen=True, eq=False)
class System:
    A: np.ndarray
    B: np.ndarray
    Qc: np.ndarray
    Rc: np.ndarray
    noise: np.ndarray  # Sigma_eps
    init: np.ndarray  # Sigma_0
    gamma: float
    length: int  # episode length L

    @property
    def dims(self):
        return self.B.shape  # (ds, da)


def default_system(gamma=0.95, length=50):
    """ds = 3, da = 2; A is stable (spectral radius 0.93), noise std 0.1, s0 std 0.5."""
    A = np.array([[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.1, 0.0, 0.7]])
    B = np.array([[0.5, 0.0], [0.2, 0.4], [0.0, 0.5]])
    return System(A, B, np.eye(3), np.eye(2), 0.01 * np.eye(3), 0.25 * np.eye(3), float(gamma), int(length))


def quadratic_q(system, K, G):
    """Exact (H, h, P) of the policy a = -K s for the per-step reward z^T G z, z = (s, a).

    Q(s, a) = z^T H z + h and V(s) = s^T P s + h (formulas in the module docstring). Raises when
    sqrt(gamma) rho(A - B K) >= 1, where the discounted value is unbounded.
    """
    A, B, gamma = system.A, system.B, system.gamma
    K = np.asarray(K, np.float64)
    M = A - B @ K
    if not math.sqrt(gamma) * np.max(np.abs(np.linalg.eigvals(M))) < 1:
        raise ValueError("the closed loop is not discounted-stable")
    Pi = np.vstack([np.eye(A.shape[0]), -K])
    P = linalg.solve_discrete_lyapunov(math.sqrt(gamma) * M.T, Pi.T @ G @ Pi)
    P = (P + P.T) / 2
    F = np.hstack([A, B])
    H = G + gamma * F.T @ P @ F
    return (H + H.T) / 2, gamma * float(np.trace(P @ system.noise)) / (1 - gamma), P


def reward_form(system):
    return -linalg.block_diag(system.Qc, system.Rc)


def deviation_form(K_b):
    """||a + K_b s||^2 = ||a - beta_mean(s)||^2 as z^T D z."""
    C = np.hstack([K_b, np.eye(K_b.shape[0])])
    return C.T @ C


def exact_q(system, K):
    return quadratic_q(system, K, reward_form(system))


def value(system, K):
    """J(pi) = E_{s0} V(s0) = tr(P Sigma_0) + h; -inf when the closed loop is not discounted-stable."""
    try:
        _, h, P = exact_q(system, K)
    except ValueError:
        return -math.inf
    return float(np.trace(P @ system.init)) + h


def policy_gradient(system, K, step=1e-6):
    """dJ/dK by central differences of the exact J."""
    grad = np.zeros_like(K, np.float64)
    for idx in np.ndindex(K.shape):
        d = np.zeros_like(grad)
        d[idx] = step
        grad[idx] = (value(system, K + d) - value(system, K - d)) / (2 * step)
    return grad


def lqr_gain(system):
    """Discounted LQR optimum: the undiscounted Riccati solution for (sqrt(gamma) A, sqrt(gamma) B)."""
    g = system.gamma
    P = linalg.solve_discrete_are(math.sqrt(g) * system.A, math.sqrt(g) * system.B, system.Qc, system.Rc)
    return np.linalg.solve(system.Rc + g * system.B.T @ P @ system.B, g * system.B.T @ P @ system.A)


def qform(H, h, s, a):
    z = np.concatenate([s, a], axis=-1)
    return np.einsum("...i,ij,...j->...", z, H, z) + h


def reward(system, s, a):
    return -(np.einsum("...i,ij,...j->...", s, system.Qc, s) + np.einsum("...i,ij,...j->...", a, system.Rc, a))


def step(system, s, a, rng):
    return s @ system.A.T + a @ system.B.T + rng.standard_normal(s.shape) @ np.linalg.cholesky(system.noise).T


def simulate(system, K, sigma, episodes, rng):
    """Episode-major rows of `episodes` truncated rollouts of a = -K s + N(0, sigma^2 I)."""
    (ds, da), L = system.dims, system.length
    s = rng.standard_normal((episodes, ds)) @ np.linalg.cholesky(system.init).T
    out = {name: [] for name in ("obs", "action", "reward", "next_obs")}
    for _ in range(L):
        a = -s @ K.T + sigma * rng.standard_normal((episodes, da))
        s2 = step(system, s, a, rng)
        for name, x in zip(out, (s, a, reward(system, s, a), s2)):
            out[name].append(x)
        s = s2
    rows = {name: np.swapaxes(np.stack(x), 0, 1).reshape(episodes * L, *x[0].shape[1:]) for name, x in out.items()}
    rows["episode"] = np.repeat(np.arange(episodes), L)
    return rows


# ---------------------------------------------------------------------------------------------------------
# Critic cases


def case_critic(system, case, K_pi, K_b, kappa=None, forms=None, scale=None):
    """Float64 parameters of the two heads Q_k = Q^pi + kappa_k ||a + K_b s||^2 + z^T W_k z + w_k."""
    ds, da = system.dims
    H, h, _ = exact_q(system, K_pi)
    kap, W, w = np.zeros(2), np.zeros((2, ds + da, ds + da)), np.zeros(2)
    if case in KAPPA_CASES and not (kappa is not None and kappa >= 0):
        raise ValueError(case + " needs a nonnegative kappa")
    if case == "q1_optimistic":
        kap[0] = kappa
    elif case == "q2_optimistic":
        kap[1] = kappa
    elif case == "shared_bias":  # b = Q^pi of the reward kappa ||a + K_b s||^2: the reward part plus its bootstrap
        Hb, hb, _ = quadratic_q(system, K_pi, kappa * deviation_form(K_b))
        kap[:], W[:], w[:] = kappa, Hb - kappa * deviation_form(K_b), hb
    elif case == "independent_errors":
        W[0], W[1] = scale * forms[0], scale * forms[1]
    elif case not in ("clean", "noisy_reward"):
        raise ValueError("unknown case " + repr(case))
    return dict(H=H, h=np.float64(h), Kb=np.asarray(K_b, np.float64), kappa=kap, W=W, w=w)


def head_error(params, s, a):
    """Exact float64 error of both heads, Q_k - Q^pi, shape (..., 2)."""
    z = np.concatenate([s, a], axis=-1)
    dev = np.sum(np.square(a + s @ params["Kb"].T), axis=-1)
    return params["kappa"] * dev[..., None] + np.einsum("...i,kij,...j->...k", z, params["W"], z) + params["w"]


def random_forms(dim, rng):
    """Two independent symmetric Gaussian matrices, each scaled to unit Frobenius norm (equal size)."""
    forms = []
    for _ in range(2):
        X = rng.standard_normal((dim, dim))
        S = (X + X.T) / 2
        forms.append(S / np.linalg.norm(S))
    return np.stack(forms)


def independent_scale(system, rows, K_pi, forms, args, rng, ratio=1.0, c_max=1e3):
    """Scale c with median |Q1 - Q2| = ratio x median |t - min(Q1, Q2)| at logged actions (bisection in log c).

    Float64, with one seeded draw of TD3's clipped target noise and the rows' own rewards. Returns
    (c, achieved ratio); when even c_max falls short, (c_max, its ratio).
    """
    H, h, _ = exact_q(system, K_pi)
    s, a, s2 = rows["obs"], rows["action"], rows["next_obs"]
    n = np.clip(rng.standard_normal(a.shape) * args.policy_noise, -args.noise_clip, args.noise_clip)
    a2 = -s2 @ K_pi.T + n
    z, z2 = np.concatenate([s, a], -1), np.concatenate([s2, a2], -1)
    e, e2 = (np.einsum("ni,kij,nj->nk", x, forms, x) for x in (z, z2))
    q, q2 = qform(H, h, s, a)[:, None], qform(H, h, s2, a2)[:, None]

    def achieved(c):
        t = rows["reward"] + args.discount * np.min(q2 + c * e2, axis=1)
        return float(np.median(np.abs(c * (e[:, 0] - e[:, 1]))) / np.median(np.abs(t - np.min(q + c * e, axis=1))))

    if achieved(c_max) < ratio:
        return c_max, achieved(c_max)
    lo, hi = math.log(1e-4), math.log(c_max)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if achieved(math.exp(mid)) < ratio else (lo, mid)
    return math.exp(hi), achieved(math.exp(hi))


def independent_errors(system, rows, K_pi, args, seed, ratio=1.0, attempts=50):
    """The first seeded pair of forms whose scale reaches the target ratio: (forms, c, ratio, attempt index)."""
    dim = sum(system.dims)
    for attempt in range(attempts):
        forms = random_forms(dim, np.random.default_rng(list(seed) + [attempt, 0]))
        c, achieved = independent_scale(system, rows, K_pi, forms, args,
                                        np.random.default_rng(list(seed) + [attempt, 1]), ratio)
        if achieved >= ratio:
            return forms, c, achieved, attempt
    raise ValueError(f"no draw in {attempts} reached the head-gap ratio {ratio}")


# ---------------------------------------------------------------------------------------------------------
# Models: what td3_bc's update and td3_bc_bca's hooks call as models[0] and models[1]


class LinearActor:
    """pi(s) = -K s: no tanh, no clipping."""

    @staticmethod
    def apply(params, obs):
        return -obs @ params["K"].T


class QuadraticCritic:
    """Both heads: Q^pi plus each head's error term (float32), shape (..., 2).

    Each head runs its own copy of one expression (a batched einsum over heads rounds the two columns
    differently), so heads with equal parameters are bitwise equal.
    """

    @staticmethod
    def apply(params, obs, action):
        z = jnp.concatenate([obs, action], axis=-1)
        quad = lambda M: jnp.einsum("...i,ij,...j->...", z, M, z)
        q = quad(params["H"]) + params["h"]
        dev = jnp.sum(jnp.square(action + obs @ params["Kb"].T), axis=-1)
        return jnp.stack([q + (params["kappa"][k] * dev + quad(params["W"][k]) + params["w"][k]) for k in range(2)],
                         axis=-1)


def signal_q(signal, heads, target):
    """The value q whose residual target - q is the signal's: 0 min(Q1, Q2), 1 Q1, 2 the head farther from target."""
    gap = jnp.abs(target[..., None] - heads)
    far = jnp.where(gap[..., 1] > gap[..., 0], heads[..., 1], heads[..., 0])
    return jnp.where(signal == 0, heads.min(axis=-1), jnp.where(signal == 1, heads[..., 0], far))


def fit_scale_signal(args, config, models, state, batch, rng, signal, max_action=1.0):
    """algorithms/td3_bc_bca.fit_scale with the calibrated value chosen by the traced `signal`.

    Line for line the repository function except q = signal_q(signal, heads, target) in place of
    heads.min(-1); signal 0 reproduces it (test_lq_harness compares both, and signal 1 against fit_scale
    with a Q1-only critic).
    """
    if config.arm != "bca":
        raise ValueError("only posterior arms fit a scale")
    target = P.native_target(args, models, state.native, batch, jax.random.fold_in(rng, KEY_FIT_TARGET),
                             max_action)
    q = jax.lax.stop_gradient(
        signal_q(signal, models[1].apply(state.native.critic.params, batch.obs, batch.action), target))
    prior = jax.lax.stop_gradient(bayesian_bootstrap_weights(jax.random.fold_in(rng, KEY_PRIOR), len(batch.obs)))
    unit = state.residual_scale

    def objective(params):
        eta = models[2].apply(params, batch.obs, batch.action)
        if eta.shape != (len(batch.obs),):
            raise ValueError("scale predictor must emit one positive value per row")
        coverage = soft_coverage(target / unit, q / unit, eta, config.cal_beta)
        width = jnp.sum(prior * jnp.square(eta))
        return (
            jnp.square(jnp.sum(prior * coverage) - (1.0 - config.posterior.alpha))
            + config.width_penalty * width,
            eta,
        )

    (loss, eta), grad = jax.value_and_grad(objective, has_aux=True)(state.calibrator.params)
    proposed = state.calibrator.apply_gradients(grads=grad)
    unit_new = config.scale_ema * unit + (1.0 - config.scale_ema) * jnp.maximum(jnp.std(target - q), 1e-06)
    valid = (
        tree_finite((batch, state.native, state.calibrator, target, q, prior, unit, loss, grad, proposed,
                     unit_new))
        & jnp.all(jnp.isfinite(eta) & (eta > 0))
        & jnp.all(prior >= 0)
        & jnp.any(prior > 0)
        & jnp.isclose(prior.sum(), 1.0, rtol=1e-05, atol=1e-06)
        & (unit > 0)
        & (unit_new > 0)
    )
    result = jax.lax.cond(valid, lambda _: state._replace(calibrator=proposed, residual_scale=unit_new),
                          lambda _: state, None)
    return result, dict(scale_inputs_valid=valid, scale_fit_accepted=valid, scale_loss=loss)


def fit_loop(args, config, models, max_action, state, train, indices, keys, signal):
    """lax.scan of fit_scale_signal over precomputed batch rows and step keys.

    Returns the fitted state and per step (accepted, loss, sum of the batch's native target).
    """
    def body(state, xs):
        rows, key = xs
        batch = jax.tree_util.tree_map(lambda x: x[rows], train)
        state, diag = fit_scale_signal(args, config, models, state, batch, key, signal, max_action)
        target = P.native_target(args, models, state.native, batch, jax.random.fold_in(key, KEY_FIT_TARGET),
                                 max_action)
        return state, (diag["scale_fit_accepted"], diag["scale_loss"], target.sum())

    return jax.lax.scan(body, state, (indices, keys))


def calibrate_signal(args, config, models, state, heldout, rng, signal, max_action=1.0):
    """td3_bc_bca.refresh's eager threshold selection with the signal's residual.

    Same key folds as refresh; returns (reference, diagnostics, target, residual target - q).
    """
    target = P.native_target(args, models, state.native, heldout, jax.random.fold_in(rng, KEY_REFRESH_NOISE),
                             max_action)
    heads = models[1].apply(state.native.critic.params, heldout.obs, heldout.action)
    residual = target - signal_q(jnp.int32(signal), heads, target)
    cal = models[2].apply(state.calibrator.params, heldout.obs, heldout.action)
    reference, frozen, diagnostics = freeze_reference(state.calibrator.params, state.residual_scale, cal,
                                                      residual, jax.random.fold_in(rng, KEY_WBCP),
                                                      config.posterior)
    if not frozen or not diagnostics["certified"]:
        raise FloatingPointError("the calibration bank did not certify a finite threshold")
    return reference, diagnostics, np.asarray(target), np.asarray(residual)


def actor_loop(args, models, max_action, with_dose, native, train, indices, keys, doses):
    """actor-only TD3+BC: td3_bc_update per step, then the frozen critic, target critic and target actor restored.

    `it` = policy_freq x step, so every call updates the actor. with_dose False is the host
    (bc_multiplier=None). Returns the final state and per step (q_mean, lambda, bc_loss).
    """
    def body(state, xs):
        t, rows, key = xs
        batch = jax.tree_util.tree_map(lambda x: x[rows], train)
        new, metrics = BASE.td3_bc_update(args, models[0].apply, models[1].apply, state, batch,
                                          args.policy_freq * t, key, max_action,
                                          bc_multiplier=doses[rows] if with_dose else None)
        new = new._replace(actor_target=native.actor_target, critic=native.critic,
                           critic_target=native.critic_target)
        return new, (metrics["q_mean"], metrics["lambda"], metrics["bc_loss"])

    return jax.lax.scan(body, native, (jnp.arange(len(indices)), indices, keys))


def actor_terms(args, models, critic_params, K, obs, action, dose):
    """K-gradients of the two terms of td3_bc's actor loss: -lambda mean Q1(s, pi(s)) and the (dosed) BC term."""
    def q_term(K):
        q = models[1].apply(critic_params, obs, models[0].apply({"K": K}, obs))[..., 0]
        return -(args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())) * q.mean()

    def bc_term(K):
        sq = jnp.square(models[0].apply({"K": K}, obs) - action)
        return sq.mean() if dose is None else (jax.lax.stop_gradient(dose) * sq.mean(axis=-1)).mean()

    return jax.grad(q_term)(K), jax.grad(bc_term)(K)


def control_doses(dose, error, perm):
    """BCA's per-row doses and the controls with the same mean; also whether the oracle fell back to constant."""
    dose = np.asarray(dose, np.float64)
    mean = dose.mean()
    constant = np.full_like(dose, mean)
    degenerate = not error.mean() > 0
    oracle = constant if degenerate else error * (mean / error.mean())
    arms = dict(bca=dose, constant=constant, shuffled=dose[perm], oracle=oracle)
    return {k: np.asarray(v, np.float32) for k, v in arms.items()}, degenerate


# ---------------------------------------------------------------------------------------------------------
# Harness


@dataclass(frozen=True)
class Settings:
    seed: int = 20261001
    gamma: float = 0.95
    episode_length: int = 50
    train_episodes: int = 200
    cal_episodes: int = 200
    cal_rows_per_episode: int = 5
    eval_episodes: int = 400
    behavior_noise: float = 0.2
    poor_gain: float = 0.0
    reward_noise: float = 0.3
    fit_steps: int = 5000
    actor_steps: int = 500
    batch_size: int = 256
    blend: float = 0.5
    draws: int = 1000
    independent_ratio: float = 1.0


def digest(*arrays):
    h = hashlib.sha256()
    for x in arrays:
        h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()[:16]


def cosine(x, y):
    x, y = np.ravel(x), np.ravel(y)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else None


def correlation(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None


def miss(values, width):
    return float(np.mean(np.asarray(values, np.float64) > width))


def distance_terciles(distance, bins=N_TERCILES):
    """Bin index per row by rank of `distance` (0 nearest): sizes differ by at most 1; ties go by row order."""
    rank = np.argsort(np.argsort(np.asarray(distance), kind="stable"), kind="stable")
    return rank * bins // len(rank)


def binned_miss(values, width, index, name, bins=N_TERCILES):
    """miss() within each bin of `index`, as {name_t1: ..., name_t{bins}: ...} (t1 nearest)."""
    values, width = np.asarray(values, np.float64), np.broadcast_to(width, np.shape(values))
    return {f"{name}_t{k + 1}": miss(values[index == k], width[index == k]) for k in range(bins)}


class Harness:
    """One compiled fit, actor and gradient program per run; `replicate` evaluates one behaviour x seed."""

    def __init__(self, settings, system=None):
        self.cfg = settings
        self.system = system or default_system(settings.gamma, settings.episode_length)
        if self.system.gamma != settings.gamma or self.system.length != settings.episode_length:
            raise ValueError("the system's discount or episode length differs from the settings")
        if self.system.dims[::-1] != DIRECTION.shape:
            raise ValueError("DIRECTION perturbs a (da, ds) = (2, 3) gain")
        self.args = replace(BASE.Args(), discount=settings.gamma, batch_size=settings.batch_size)
        self.config = P.Config(arm="bca", posterior=WBCPConfig(0.1, 0.95, settings.draws), blend=settings.blend)
        ds, da = self.system.dims
        self.models = (LinearActor(), QuadraticCritic(), Calibrator(jnp.zeros(ds), jnp.ones(ds), state_dep=True))
        self.fit = jax.jit(partial(fit_loop, self.args, self.config, self.models, MAX_ACTION))
        self.actor = {flag: jax.jit(partial(actor_loop, self.args, self.models, MAX_ACTION, flag))
                      for flag in (True, False)}
        self.terms = jax.jit(partial(actor_terms, self.args, self.models))
        self.k_opt = lqr_gain(self.system)

    def behavior_gain(self, behavior):
        return self.k_opt * {"good": 1.0, "poor": self.cfg.poor_gain}[behavior]

    def check_stable(self, behaviors, offsets):
        """Raise, before any compute, when a behaviour gain or an evaluated K_pi gives J = -inf."""
        bad = [f"{b} behaviour" for b in behaviors if not math.isfinite(value(self.system, self.behavior_gain(b)))]
        bad += [f"{b} pi_offset={o}" for b in behaviors for o in offsets
                if not math.isfinite(value(self.system, self.behavior_gain(b) + o * DIRECTION))]
        if bad:
            raise ValueError("closed loop not discounted-stable: " + ", ".join(bad))

    def data(self, behavior, replicate):
        """Training, calibration-bank and evaluation rows (float64) from whole behaviour episodes."""
        cfg, b = self.cfg, BEHAVIORS.index(behavior)
        sizes = (cfg.train_episodes, cfg.cal_episodes, cfg.eval_episodes)
        rows = simulate(self.system, self.behavior_gain(behavior), cfg.behavior_noise, sum(sizes),
                        np.random.default_rng([cfg.seed, replicate, b, 0]))
        rows["reward_noise"] = cfg.reward_noise * np.random.default_rng(
            [cfg.seed, replicate, b, 1]).standard_normal(len(rows["reward"]))
        L, edges = self.system.length, np.cumsum((0,) + sizes)
        k = cfg.cal_rows_per_episode
        episodes, offsets = bank.stratified_bank(np.full(sizes[1], L), sizes[1] * k, k,
                                                 np.random.default_rng([cfg.seed, replicate, b, 5]))
        if not np.array_equal(np.sort(episodes), np.arange(sizes[1])):
            raise AssertionError("the bank must take rows from every calibration episode")
        cal = np.sort(np.concatenate([(edges[1] + e) * L + o for e, o in zip(episodes, offsets)]))
        index = dict(train=np.arange(edges[0] * L, edges[1] * L), cal=cal,
                     eval=np.arange(edges[2] * L, edges[3] * L))
        return {split: {name: x[idx] for name, x in rows.items()} for split, idx in index.items()}

    @staticmethod
    def transitions(rows, noisy):
        r = rows["reward"] + rows["reward_noise"] if noisy else rows["reward"]
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        return Transition(f32(rows["obs"]), f32(rows["action"]), f32(r), f32(rows["next_obs"]),
                          jnp.zeros(len(r), jnp.float32))

    def native(self, K_pi, params):
        critic_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), params)
        actor = TrainState.create(apply_fn=LinearActor.apply, params={"K": jnp.asarray(K_pi, jnp.float32)},
                                  tx=BASE.C.torch_adam(self.args.lr))
        critic = TrainState.create(apply_fn=QuadraticCritic.apply, params=critic_params, tx=optax.set_to_zero())
        return BASE.AgentTrainState(actor, actor, critic, critic)

    def settings_grid(self, cases, kappas, offsets):
        return [(case, kappa if case in KAPPA_CASES else None, offset)
                for offset in offsets for case in cases for kappa in (kappas if case in KAPPA_CASES else [None])]

    def replicate(self, behavior, replicate, grid, signals=SIGNALS):
        """Every (case, kappa, pi_offset) in grid x signal for one behaviour and replicate seed.

        Returns (records, setting diagnostics, fingerprints); fingerprints hold digests the built-in checks compare.
        """
        cfg, system, args, models = self.cfg, self.system, self.args, self.models
        b, (ds, da) = BEHAVIORS.index(behavior), system.dims
        data = self.data(behavior, replicate)
        K_b = self.behavior_gain(behavior)
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(cfg.seed), replicate), b)
        fit_keys = jax.vmap(partial(jax.random.fold_in, jax.random.fold_in(key, 1)))(jnp.arange(cfg.fit_steps))
        actor_keys = jax.vmap(partial(jax.random.fold_in, jax.random.fold_in(key, 4)))(jnp.arange(cfg.actor_steps))
        cal_key, eval_key = jax.random.fold_in(key, 2), jax.random.fold_in(key, 3)
        n_train = len(data["train"]["reward"])
        fit_rows = jnp.asarray(np.random.default_rng([cfg.seed, replicate, b, 2]).integers(
            n_train, size=(cfg.fit_steps, cfg.batch_size)), jnp.int32)
        actor_rows = jnp.asarray(np.random.default_rng([cfg.seed, replicate, b, 3]).integers(
            n_train, size=(cfg.actor_steps, cfg.batch_size)), jnp.int32)
        perm = np.random.default_rng([cfg.seed, replicate, b, 4]).permutation(n_train)
        calibrator = models[2]
        zeros = jnp.zeros((1, ds)), jnp.zeros((1, da))
        cal_state = TrainState.create(apply_fn=calibrator.apply,
                                      params=calibrator.init(jax.random.fold_in(key, KEY_INIT), *zeros),
                                      tx=optax.adam(self.config.cal_lr))
        tr, ev = data["train"], data["eval"]
        j_beta, j_opt = value(system, K_b), value(system, self.k_opt)
        records, diagnostics, prints = [], [], []
        for case, kappa, offset in grid:
            K_pi = K_b + offset * DIRECTION
            forms = scale = ratio = attempt = None
            if case == "independent_errors":
                forms, scale, ratio, attempt = independent_errors(system, tr, K_pi, args, (cfg.seed, replicate, b, 6),
                                                                  cfg.independent_ratio)
            params = case_critic(system, case, K_pi, K_b, kappa, forms, scale)
            native = self.native(K_pi, params)
            noisy = case == "noisy_reward"
            train, cal, evals = (self.transitions(data[s], noisy) for s in ("train", "cal", "eval"))
            pi_tr, pi_ev = -tr["obs"] @ K_pi.T, -ev["obs"] @ K_pi.T
            err = dict(train=np.abs(head_error(params, tr["obs"], tr["action"])[:, 0]),
                       train_pi=np.abs(head_error(params, tr["obs"], pi_tr)[:, 0]),
                       eval=np.abs(head_error(params, ev["obs"], ev["action"])[:, 0]),
                       eval_pi=np.abs(head_error(params, ev["obs"], pi_ev)[:, 0]))
            deviation = np.linalg.norm(tr["action"] + tr["obs"] @ K_b.T, axis=1)
            distance_ev = np.linalg.norm(pi_ev + ev["obs"] @ K_b.T, axis=1)  # |pi(s) - beta_mean(s)| per row
            terciles = distance_terciles(distance_ev)
            H, h, _ = exact_q(system, K_pi)
            g_true = 2 * np.concatenate([tr["obs"], pi_tr], 1) @ H[:, ds:]
            z_pi = np.concatenate([tr["obs"], pi_tr], 1)
            g_q1 = g_true + 2 * params["kappa"][0] * (pi_tr + tr["obs"] @ K_b.T) + 2 * z_pi @ params["W"][0][:, ds:]
            row_cos = [cosine(x, y) for x, y in zip(g_q1, g_true)]
            alignment = dict(alignment_K=cosine(g_q1.T @ -tr["obs"], g_true.T @ -tr["obs"]),
                             alignment_rowwise=(float(np.mean([c for c in row_cos if c is not None]))
                                                if any(c is not None for c in row_cos) else None),
                             alignment_action=cosine(g_q1.mean(0), g_true.mean(0)))
            j0, pg = value(system, K_pi), policy_gradient(system, K_pi)
            k0 = jnp.asarray(K_pi, jnp.float32)
            train_actor = lambda flag, doses: np.asarray(
                self.actor[flag](native, train, actor_rows, actor_keys, doses)[0].actor.params["K"], np.float64)
            k_none = train_actor(False, jnp.ones(n_train, jnp.float32))
            heads_ev = np.asarray(models[1].apply(native.critic.params, evals.obs, evals.action), np.float64)
            t_probe = np.asarray(P.native_target(args, models, native, evals,
                                                 jax.random.fold_in(eval_key, KEY_REFRESH_NOISE), MAX_ACTION))
            label = dict(behavior=behavior, case=case, kappa=kappa, pi_offset=offset)
            diagnostics.append(dict(label, replicate=replicate, J_pi=j0, J_beta=j_beta, J_opt=j_opt,
                                    pi_beta_distance=float(distance_ev.mean()),
                                    **{f"pi_beta_distance_t{k + 1}": float(distance_ev[terciles == k].mean())
                                       for k in range(N_TERCILES)},
                                    logged_deviation=float(np.mean(np.linalg.norm(ev["action"] + ev["obs"] @ K_b.T,
                                                                                  axis=1))),
                                    action_outside_logged=float(np.mean(np.any(np.abs(ev["action"]) > 1, axis=1))),
                                    action_outside_pi=float(np.mean(np.any(np.abs(pi_ev) > 1, axis=1))),
                                    q1_error_logged_mean=float(err["eval"].mean()),
                                    q1_error_pi_mean=float(err["eval_pi"].mean()),
                                    head_gap_over_residual=float(np.median(np.abs(heads_ev[:, 0] - heads_ev[:, 1]))
                                                                 / np.median(np.abs(t_probe - heads_ev.min(1)))),
                                    independent_scale=scale, independent_ratio=ratio,
                                    independent_redraws=attempt))
            for signal in signals:
                sid = SIGNALS.index(signal)
                state = P.State(native, cal_state, jnp.asarray(1.0), initial_reference(cal_state.params))
                state, (accepted, loss, target_sums) = self.fit(state, train, fit_rows, fit_keys, jnp.int32(sid))
                reference, diag, t_cal, r_cal = calibrate_signal(args, self.config, models, state, cal, cal_key,
                                                                 sid, MAX_ACTION)
                state = state._replace(posterior=reference)
                thr = float(reference.threshold)
                # Evaluation split: residuals of every head and the true Q1 error, at logged actions and at pi(s).
                t_ev = np.asarray(P.native_target(args, models, native, evals,
                                                  jax.random.fold_in(eval_key, KEY_REFRESH_NOISE), MAX_ACTION))
                heads = models[1].apply(native.critic.params, evals.obs, evals.action)
                own = np.asarray(jnp.asarray(t_ev) - signal_q(jnp.int32(sid), heads, jnp.asarray(t_ev)))
                heads = np.asarray(heads)
                res = dict(min=np.abs(t_ev - heads.min(1)), q1=np.abs(t_ev - heads[:, 0]),
                           q2=np.abs(t_ev - heads[:, 1]))
                res["maxabs"] = np.maximum(res["q1"], res["q2"])
                pi_f32 = models[0].apply(native.actor.params, evals.obs)
                sigma = np.asarray(positive_scale(models[2].apply(reference.cal_params, evals.obs, evals.action),
                                                  reference.residual_scale), np.float64)
                sigma_pi = np.asarray(positive_scale(models[2].apply(reference.cal_params, evals.obs, pi_f32),
                                                     reference.residual_scale), np.float64)
                width, width_pi = thr * sigma, thr * sigma_pi
                # Doses on every training row from BCA's hook, the controls, and the actor runs.
                readout = P.bc_readout(models, state, train, self.config.blend)
                if not bool(readout.inputs_valid):
                    raise FloatingPointError("invalid dose readout")
                dose = np.asarray(readout.dose, np.float64)
                arms, degenerate = control_doses(dose, err["train"], perm)
                finals = {"none": k_none, **{arm: train_actor(True, jnp.asarray(d)) for arm, d in arms.items()}}
                qgrad, _ = self.terms(native.critic.params, k0, train.obs, train.action, None)
                bc_norms = {arm: float(jnp.linalg.norm(self.terms(native.critic.params, k0, train.obs,
                                                                  train.action, None if arm == "none"
                                                                  else jnp.asarray(arms[arm]))[1]))
                            for arm in ARMS}
                metrics = dict(
                    threshold=thr, lambda_hat=diag["lambda_hat"], lambda_hpd=diag["lambda_hpd"],
                    n_cal=diag["scores"], residual_unit=float(reference.residual_scale),
                    fit_accepted=float(np.mean(accepted)), fit_loss_final=float(np.mean(loss[-100:])),
                    width_logged=float(width.mean()), width_pi=float(width_pi.mean()),
                    mean_residual_own=float(own.mean()), median_abs_residual_own=float(np.median(np.abs(own))),
                    miss_own_logged=miss(np.abs(own), width),
                    **{f"miss_resid_{name}_logged": miss(r, width) for name, r in res.items()},
                    miss_q1err_logged=miss(err["eval"], width), miss_q1err_pi=miss(err["eval_pi"], width_pi),
                    **binned_miss(err["eval_pi"], width_pi, terciles, "miss_q1err_pi"),
                    **binned_miss(err["eval"], width, terciles, "miss_q1err_logged"),
                    dose_mean=float(dose.mean()), dose_sd=float(dose.std()),
                    dose_p05=float(np.percentile(dose, 5)), dose_p95=float(np.percentile(dose, 95)),
                    dose_min=float(dose.min()), dose_max=float(dose.max()),
                    dose_corr_q1err_logged=correlation(dose, err["train"]),
                    dose_corr_q1err_pi=correlation(dose, err["train_pi"]),
                    dose_corr_deviation=correlation(dose, deviation),
                    oracle_degenerate=float(degenerate),
                    q_term_grad_norm=float(jnp.linalg.norm(qgrad)), **alignment,
                    **{f"bc_term_grad_norm_{arm}": v for arm, v in bc_norms.items()},
                    **{f"J_change_{arm}": value(system, k) - j0 for arm, k in finals.items()},
                    **{f"update_pg_cosine_{arm}": cosine(k - K_pi, pg) for arm, k in finals.items()},
                )
                records.append(dict(label, signal=signal, replicate=replicate, metrics=metrics))
                prints.append(dict(label, signal=signal, fit_batches=digest(fit_rows), fit_targets=digest(target_sums),
                                   cal_target=digest(t_cal), eval_target=digest(t_ev), residual_cal=digest(r_cal),
                                   residual_eval=digest(own), sigma=digest(sigma, sigma_pi),
                                   threshold=digest(reference.threshold), doses=digest(dose),
                                   actor=digest(*[finals[a] for a in ARMS]), qgrad=digest(qgrad)))
        return records, diagnostics, prints


def builtin_checks(prints):
    """Built-in checks 2-5 (and identical targets and batches) over one behaviour x replicate's fingerprints."""
    by = {}
    for p in prints:
        by.setdefault((p["case"], p["kappa"], p["pi_offset"]), {})[p["signal"]] = p
    same = lambda group, fields: all(len({g[f] for g in group}) == 1 for f in fields)
    out = dict(identical_targets_and_batches=[], check2_equal_heads=[], check3_q2_min_is_q1=[],
               check4_q1_min_is_clean=[], check5_qterm_gradient=[])
    for (case, kappa, offset), sig in by.items():
        tag = f"{case} kappa={kappa} offset={offset}"
        out["identical_targets_and_batches"].append(
            (tag, same(sig.values(), ("fit_batches", "fit_targets", "cal_target", "eval_target"))))
        out["check5_qterm_gradient"].append((tag, same(sig.values(), ("qgrad",))))
        if case in ("clean", "noisy_reward", "shared_bias") and len(sig) > 1:
            out["check2_equal_heads"].append(
                (tag, same(sig.values(), ("residual_eval", "sigma", "threshold", "doses", "actor"))))
        if case == "q2_optimistic" and {"min", "q1"} <= set(sig):
            out["check3_q2_min_is_q1"].append(
                (tag, same([sig["min"], sig["q1"]], ("residual_eval", "sigma", "threshold", "doses", "actor"))))
        if case == "q1_optimistic" and "min" in sig and "min" in by.get(("clean", None, offset), {}):
            ref = by[("clean", None, offset)]["min"]
            out["check4_q1_min_is_clean"].append(
                (tag, same([sig["min"], ref], ("residual_cal", "residual_eval", "fit_targets", "sigma", "threshold"))))
    return out


def summarize(values):
    """Mean and SE over the finite values; the rest (None, or +-inf from an unstable gain) count as nonfinite.

    undefined counts the None among them (a cosine or correlation with a constant input, or a diagnostic
    that does not apply to the case).
    """
    finite = [v is not None and math.isfinite(v) for v in values]
    vals = np.array([v for v, ok in zip(values, finite) if ok], np.float64)
    out = dict(mean=None, se=None, n=int(vals.size), nonfinite=len(values) - int(vals.size),
               undefined=sum(v is None for v in values),
               values=[float(v) if ok else None for v, ok in zip(values, finite)])
    if vals.size:
        out.update(mean=float(vals.mean()), se=float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else None)
    return out


def aggregate(records, diagnostics):
    groups, settings = {}, {}
    for r in records:
        groups.setdefault((r["behavior"], r["case"], r["kappa"], r["pi_offset"], r["signal"]), []).append(r["metrics"])
    for d in diagnostics:
        settings.setdefault((d["behavior"], d["case"], d["kappa"], d["pi_offset"]), []).append(d)
    results = []
    for (behavior, case, kappa, offset, signal), runs in groups.items():
        names = list(runs[0])
        metrics = {name: summarize([m[name] for m in runs]) for name in names}
        for arm in ARMS[1:]:  # paired across replicates
            metrics[f"J_gain_bca_minus_{arm}"] = summarize([m["J_change_bca"] - m[f"J_change_{arm}"] for m in runs])
        results.append(dict(behavior=behavior, case=case, kappa=kappa, pi_offset=offset, signal=signal,
                            replicates=len(runs), metrics=metrics))
    skip = ("behavior", "case", "kappa", "pi_offset", "replicate")
    setting_rows = [dict(behavior=k[0], case=k[1], kappa=k[2], pi_offset=k[3],
                         diagnostics={name: summarize([d[name] for d in ds]) for name in ds[0] if name not in skip})
                    for k, ds in settings.items()]
    return results, setting_rows


def fmt(stat):
    """mean ± SE over the finite values, with '(n=..., k nonfinite)' when values were excluded from them.

    Plain 'n/a' when every value is None (undefined or not applicable); any +-inf or NaN is always flagged.
    """
    k = stat["nonfinite"]
    note = f" (n={stat['n']}, {k} nonfinite)" if k and (stat["n"] or k > stat["undefined"]) else ""
    if stat["mean"] is None:
        return "n/a" + note
    m, se = stat["mean"], stat["se"]
    text = f"{m:.4g}"
    return (text if se is None else f"{text} ± {se:.2g}") + note


def write_markdown(path, payload):
    lines = ["# LQ signal harness", "",
             "Built by experiments/signal/lq_harness.py; expectations: runs/wbcp_signal/expectations.md. "
             "Means ± standard errors over replicates (no SE with one replicate). '(n=..., k nonfinite)' marks a "
             "mean over n replicates that excludes k values: an undefined cosine or correlation (a constant "
             "input), or J of an unstable closed loop; a plain 'n/a' is undefined or not applicable in every "
             "replicate. Miss rates are shares of evaluation rows whose quantity "
             "exceeds threshold x sigma; _t1.._t3 are terciles of |pi(s) - beta_mean(s)| (t1 nearest).", "",
             "## Built-in checks", ""]
    for name, ok in payload["builtin_checks_summary"].items():
        lines.append(f"- {name}: {'pass' if ok else 'FAIL'}" if ok is not None else f"- {name}: not applicable")
    meta = payload["meta"]
    lines += ["", "## Run", "", "```", json.dumps(meta["settings"], indent=1), "```", "",
              f"K* = {np.round(np.array(meta['k_opt']), 4).tolist()}, J(K*) = {meta['J_opt']:.4f}; "
              + "; ".join(f"{b}: J(beta) = {meta['J_beta'][b]:.4f}" for b in meta["J_beta"]), ""]
    for s in payload["settings"]:
        tag = f"{s['behavior']} / {s['case']} / kappa={s['kappa']} / pi_offset={s['pi_offset']}"
        lines += [f"## {tag}", "", "| setting diagnostic | value |", "|---|---|"]
        lines += [f"| {k} | {fmt(v)} |" for k, v in s["diagnostics"].items()]
        rows = [r for r in payload["results"] if (r["behavior"], r["case"], r["kappa"], r["pi_offset"])
                == (s["behavior"], s["case"], s["kappa"], s["pi_offset"])]
        if rows:
            lines += ["", "| metric | " + " | ".join(r["signal"] for r in rows) + " |",
                      "|---|" + "---|" * len(rows)]
            lines += [f"| {name} | " + " | ".join(fmt(r["metrics"][name]) for r in rows) + " |"
                      for name in rows[0]["metrics"]]
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf8")


def run(settings, output, cases=CASES, signals=SIGNALS, behaviors=BEHAVIORS, replicates=5, kappas=DEFAULT_KAPPAS,
        offsets=DEFAULT_OFFSETS, log=print):
    started = time.perf_counter()
    harness = Harness(settings)
    harness.check_stable(behaviors, offsets)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if len(set(offsets)) < 2:
        log("one pi_offset: Step 3 expectation 3's distance clause rests on the tercile metrics alone")
    grid = harness.settings_grid(cases, kappas, offsets)
    records, diagnostics, checks = [], [], {}
    for behavior in behaviors:
        for r in range(replicates):
            t0 = time.perf_counter()
            rec, diag, prints = harness.replicate(behavior, r, grid, signals)
            records += rec
            diagnostics += diag
            checks[f"{behavior}/{r}"] = builtin_checks(prints)
            log(f"{behavior} replicate {r}: {len(rec)} case x signal runs in {time.perf_counter() - t0:.0f}s")
            # each replicate compiles fresh closures; without releasing them the process reaches the kernel's
            # memory-map limit (vm.max_map_count 65,530) after about two replicates. Clearing the compilation
            # caches changes no computed value.
            jax.clear_caches()
            gc.collect()
    summary = {}
    for name in next(iter(checks.values())):
        flags = [ok for c in checks.values() for _, ok in c[name]]
        summary[name] = all(flags) if flags else None
    results, setting_rows = aggregate(records, diagnostics)
    payload = dict(
        meta=dict(settings=asdict(settings), cases=list(cases), signals=list(signals), behaviors=list(behaviors),
                  replicates=replicates, kappas=list(kappas), pi_offsets=list(offsets),
                  system={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in vars(harness.system).items()},
                  k_opt=harness.k_opt.tolist(), J_opt=value(harness.system, harness.k_opt),
                  J_beta={b: value(harness.system, harness.behavior_gain(b)) for b in behaviors},
                  direction=DIRECTION.tolist(), argv=sys.argv, seconds=time.perf_counter() - started,
                  jax=jax.__version__, numpy=np.__version__),
        builtin_checks_summary=summary, builtin_checks=checks, results=results, settings=setting_rows)
    (output / "results.json").write_text(json.dumps(payload, indent=1, default=str, allow_nan=False), encoding="utf8")
    write_markdown(output / "results.md", payload)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", required=True, help="new directory for results.json and results.md")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--signals", nargs="+", choices=SIGNALS, default=list(SIGNALS))
    parser.add_argument("--behavior", nargs="+", choices=BEHAVIORS, default=list(BEHAVIORS))
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--kappa", nargs="+", type=float, default=list(DEFAULT_KAPPAS))
    parser.add_argument("--pi-offset", nargs="+", type=float, default=list(DEFAULT_OFFSETS))
    for name, default in asdict(Settings()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    return parser


def main():
    defaults = Settings()
    ns = build_parser().parse_args()
    settings = Settings(**{name: getattr(ns, name) for name in asdict(defaults)})
    payload = run(settings, ns.output, ns.cases, ns.signals, ns.behavior, ns.replicates, ns.kappa, ns.pi_offset)
    print(json.dumps(payload["builtin_checks_summary"]), f"{payload['meta']['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
