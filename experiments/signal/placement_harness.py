"""Placement harness for step 4: where the Q1-residual signal is applied, at matched realized strength.

Design: experiments/signal/STEP4_DESIGN.md; this module is section 12's items 1-7 (data, critics, weights, the actor
update, the fixed point, strength, Level-1 terms). Expectations: runs/wbcp_signal/step4/expectations_DRAFT.md. The
module builds and tests machinery only: it fixes no evidence set, runs no pilot and never calls lq_harness.value() or
any other J function (score_values.py computes J after the matched knobs are hashed; design section 5, "Blindness").
lq_harness is imported and called, never edited, so block R's anchors keep step 3's code path bit for bit.

System and start. lq_harness.default_system(), K* = lq_harness.lqr_gain, and the S-common start K_0 = 0.5 K* + 0.5 D
(D = lq_harness.DIRECTION). A quality level sets the behaviour gain: expert K*, mixed (half the rows (common states) or
episodes (episodic) at K*, half at 0; population gain K-bar_b = 0.5 K*), medium 0.1338 K*, poor 0.

1. Data. data_common (X-CS and block C): states from K_0 rollouts with behaviour noise sigma_b (train / calibration /
   evaluation episodes as in step 3), then one draw per row of the action noise n ~ N(0, I) and the next-state shock
   eps ~ N(0, Sigma_eps); every level logs a = -K_b s + sigma_b n, r = r(s, a), s' = A s + B a + eps, done = 0. States,
   noise, shocks, reward noise and the mixed mode are drawn without the level in the seed, so they are bitwise identical
   across levels and only K_b differs. With contexts, each episode draws a context uniformly; obs = [s, onehot(c)] and
   next_obs keeps the context (states, noise and shocks are those of the context-free data). data_episodic (X-EP):
   step 3's data() generalised to the four gains (mixed: the mode per episode). Both return K-bar_b, the least-squares
   BC gain K_bc on the training rows (per context too), the per-row gain multiple (mixed: 1 or 0, the mode) and a
   seeded tie-break permutation of the training rows. streams() reproduces step 3's batch rows and keys
   (lq_harness.Harness.replicate) for a stream tag; in X-CS one tag shared by the levels keeps even the batches equal.
2. Critics. case_critic_x: lq_harness.case_critic for the step-3 cases, plus tilt (e1 = (a + K-bar_b s)^T T s, written
   into W[0] = [[sym(K-bar_b^T T), T^T / 2], [T / 2, 0]], so H_aa is unchanged) and cone (e1 = kappa c(s) ||a + K-bar_b
   s||^2 with c(s) = sigmoid(10 (s/|s| . v - c0)), a new parameter field, e1 >= 0 so min(Q1, Q2) is bitwise Q^pi).
   PlacementCritic is QuadraticCritic plus the cone term; it branches in Python on the parameter keys, so a critic
   without a cone runs QuadraticCritic.apply itself (tested bitwise). ContextLinearActor and ContextQuadraticCritic are
   one-hot weighted sums of per-context gains K_c and per-context heads (H_c, h_c = quadratic_q(K_c) plus that
   context's error). head_error_x and context_head_error are the float64 NumPy errors e_k = Q_k - Q^pi.
3. Weights. rank_normal assigns the rank-normal multiset (z_(j) = Phi^-1((j - 1/2)/n) clipped to +-2.5, M = exp(s z)
   / mean exp(s z)) by the global rank of an assignment variable, ties broken by a seeded permutation, never by row
   order. Every assignment indexes one float64 multiset, so the sorted weights of SIG, SHUF, STRAT, ANTI, NUIS, OR1 and
   OR2 are bitwise identical. batch_rank_normal does the same within a batch (pi family in A500). assignments returns
   them per family (L: logged action; pi: the actor's action, at pi_0 here and per step in A500) and placement (STRAT's
   deciles and OR2 are placement-specific).
4. Actor update. placement_step is td3_bc_update's actor part with hooks bc_w (P0/P1), q_w (P2), pen_fn / pen_c (P3),
   exact_fn (EXACT) and pi_eval (weights recomputed at the current actor under stop-gradient). lambda is always alpha /
   mean |Q1(s, pi(s))| from the unweighted, unpenalised Q1 (the lambda-lock). With every hook off it runs td3_bc's own
   expressions, so the compiled actor update matches td3_bc_update bit for bit (tested, one step and the whole loop
   against lq_harness.actor_loop; eager op-by-op calls round differently). The critic update and the soft target
   updates are skipped: the harness critic's optimizer is optax.set_to_zero() and lq_harness.actor_loop restores the
   critic, target critic and target actor after every step, so skipping them leaves the actor bitwise unchanged.
   placement_loop is lax.scan of placement_step with per-step diagnostics (term gradient norms, |dK|_F, Adam's mean
   per-coordinate SNR, Kish n_eff of the batch weights) and the mean bias-corrected Adam second moment; program()
   compiles it once per Spec, vmapped over runs (knob x assignment), with every cell-varying quantity traced.
5. Fixed point (lambda frozen at lambda_0 = alpha / mean_i |Q1(s_i, pi_0(s_i))| over all training rows, weights
   fixed). fp_rows: P_i = Hess_a Q(s_i, 0) / 2 and q_i = grad_a Q(s_i, 0) / 2 by float64 autodiff (exact: every head is
   quadratic in a). fixed_point solves design section 4's 6x6 system [mean(-lambda_0 u_i S_i (x) P_i + v_i / d_a S_i
   (x) I)] vec K = mean(-lambda_0 u_i vec(q_i s_i^T) - v_i / d_a vec(a_i s_i^T)) (column-major vec; the matrix is half
   the loss Hessian, and a minimum eigenvalue <= 0 makes the arm nonfinite). Block C solves it per context with
   lambda_0 global. fixed_point_penalty runs BFGS on the full P3 loss (JAX value_and_grad), warm-started along the c
   ladder, converged when |grad L|_F <= 1e-5 |grad L(K_0)|_F within 500 iterations; it evaluates the loss in float64,
   because a float32 loss stalls about 100x above that criterion (see its docstring). lstd_penalty is block C's
   per-context LSTD penalty critic (15 quadratic monomials of z plus 1; the next-feature expectation is exact under
   TD3's clipped target noise, E n = 0 and E n n^T = v_clip I).
6. Strength. strength = sqrt(tr(delta Sigma_ev delta^T)), delta = K(arm) - K(NONE); match_knob (first crossing on the
   ladder, bisection in log knob, monotone flag, reachability; never extrapolated); frontier (nearest uniform-BC
   point m*); regret_alpha (from J values computed elsewhere).
7. Level 1. terms: the K-gradients of each term of the placement loss on all training rows; level1: rho, the
   projection on span(g_Q,NONE, g_BC,NONE) (m_eff = b / a, tau = |r| / |g|, unidentified when |cos(g_Q, g_BC)| >
   0.98), the Adam-preconditioned increment Delta g / (sqrt(v-bar_NONE) + eps) and the alignment cosines.

Two limits of the design's [derived] items that the tests make explicit (test_placement_harness):
- The cancellation identities of section 4 (ii)-(iv) (expectations Part 0 item 4) are exact only when the error is
  centred at the BC target that the rows actually carry: the (weighted) least-squares gain of the logged actions on the
  training states. With the population gain K-bar_b (the design's error definitions) they hold up to the finite-sample
  gap K_bc - K-bar_b; fixed_point then differs from EXACT by exactly lambda_0 kappa A^-1 vec((K_bc,w - K-bar_b)
  Sigma_w), which the tests check to 1e-8, and which is far above the 1e-6 tolerance on realistic data.
- LSTD equals the closed-form penalty value on an in-class reward only with deterministic transitions: with shocks,
  the sampled next state makes the LSTD fixed point differ from the population value by O(n^-1/2). A constant reward
  gives 1 / (1 - gamma) exactly in either case.

Tests: JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_placement_harness
"""

import math
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402
from jax.experimental import enable_x64  # noqa: E402

import algorithms.td3_bc as BASE  # noqa: E402
import algorithms.td3_bc_bca as P  # noqa: E402
from calibration import bank  # noqa: E402
from calibration.network import Calibrator  # noqa: E402
from calibration.reference import initial_reference  # noqa: E402
from experiments.signal import lq_harness as LQ  # noqa: E402

QUALITIES = ("expert", "mixed", "medium", "poor")
QUALITY_GAIN = dict(expert=1.0, mixed=0.5, medium=0.1338, poor=0.0)  # K-bar_b as a multiple of K*
START = (0.5, 0.5)  # K_0 = 0.5 K* + 0.5 D
RANK_S, RANK_S_ROBUST, RANK_CLIP = 0.83, 0.47, 2.5
CONE_V = np.array([1.0, -1.0, 1.0]) / math.sqrt(3.0)
CONE_SHARPNESS, CONE_MEAN = 10.0, 0.25
TILT_ALIGNMENTS = (0.9, 0.5, 0.0, -0.5)
P0_LADDER = (0.5, 0.75, 1.0, 1.15, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0)
BETA_LADDER = (0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0, 20.0, 30.0)
C_LADDER = (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
TARGET_M = (1.3, 2.0, 4.0)  # S1, S2 (primary), S3 = S(P0(m))
FRONTIER_M = np.geomspace(0.25, 100.0, 200)
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8  # runtime.networks.torch_adam
UNIDENTIFIED_COS = 0.98
FAMILIES, PLACEMENTS = ("L", "pi"), ("P1", "P2")
PI_KINDS = ("sig", "nuis", "or1", "or2_p1", "or2_p2")  # assignment variables at the actor's action
# Seed tags: np.random.default_rng([seed, rep, ...]). Data: [seed, rep, TAG_CS, k] and [seed, rep, TAG_EP, level, k];
# critic draws without the level: [seed, rep, TAG_TILT] and [seed, rep, TAG_INDEPENDENT, attempt, 0 | 1]; per-level
# assignment draws: [seed, rep, level, tag]; streams (step 3's layout): [seed, rep, stream, 2 | 3].
TAG_CS, TAG_EP, TAG_TILT = 90, 91, 50
TAG_INDEPENDENT, TAG_TIE, TAG_SHUF, TAG_STRAT = 6, 9, 10, 30
TAG_BATCH_SHUF, TAG_BATCH_TIE, TAG_BATCH_STRAT = 60, 70, 80


# ---------------------------------------------------------------------------------------------------------
# Settings and the harness object


@dataclass(frozen=True)
class Settings:
    seed: int = 20261002  # blocks X and C (block R keeps step 3's 20261001 through lq_harness)
    gamma: float = 0.95
    episode_length: int = 50
    train_episodes: int = 200
    cal_episodes: int = 200
    cal_rows_per_episode: int = 5
    eval_episodes: int = 400
    behavior_noise: float = 0.2
    reward_noise: float = 0.3
    fit_steps: int = 5000
    actor_steps: int = 500
    batch_size: int = 256
    blend: float = 0.5
    draws: int = 1000
    independent_ratio: float = 1.0
    rank_s: float = RANK_S
    rank_clip: float = RANK_CLIP
    contexts: int = 4

    def lq(self):
        """The lq_harness.Settings with the same fields (its Harness supplies the system, args, config and fit)."""
        return LQ.Settings(seed=self.seed, gamma=self.gamma, episode_length=self.episode_length,
                           train_episodes=self.train_episodes, cal_episodes=self.cal_episodes,
                           cal_rows_per_episode=self.cal_rows_per_episode, eval_episodes=self.eval_episodes,
                           behavior_noise=self.behavior_noise, poor_gain=0.0, reward_noise=self.reward_noise,
                           fit_steps=self.fit_steps, actor_steps=self.actor_steps, batch_size=self.batch_size,
                           blend=self.blend, draws=self.draws, independent_ratio=self.independent_ratio)


class PlacementHarness:
    """The system, start, models and compiled programs of one process; data and cells are built per call."""

    def __init__(self, settings=Settings(), system=None):
        self.cfg = settings
        self.lq = LQ.Harness(settings.lq(), system)
        self.system, self.args, self.config = self.lq.system, self.lq.args, self.lq.config
        ds, da = self.system.dims
        self.k_opt = self.lq.k_opt
        self.K0 = START[0] * self.k_opt + START[1] * LQ.DIRECTION
        C = settings.contexts
        self.models = (LQ.LinearActor(), PlacementCritic(), Calibrator(jnp.zeros(ds), jnp.ones(ds), state_dep=True))
        self.context_models = (ContextLinearActor(), ContextQuadraticCritic(),
                               Calibrator(jnp.zeros(ds + C), jnp.ones(ds + C), state_dep=True))
        self.fit = {flag: jax.jit(partial(LQ.fit_loop, self.args, self.config,
                                          self.context_models if flag else self.models, LQ.MAX_ACTION))
                    for flag in (False, True)}
        self._programs = {}

    def behavior_gain(self, quality):
        """The population behaviour gain K-bar_b of a quality level."""
        return QUALITY_GAIN[quality] * self.k_opt

    def program(self, spec, contexts=False):
        """placement_loop for one Spec, vmapped over the run axis and compiled once per process (cached)."""
        key = (spec, contexts)
        if key not in self._programs:
            models = self.context_models if contexts else self.models
            loop = partial(placement_loop, self.args, models, spec)
            self._programs[key] = jax.jit(jax.vmap(loop, in_axes=(None, None, None, None, None, None, 0)))
        return self._programs[key]


# ---------------------------------------------------------------------------------------------------------
# 1. Data


def seed_rng(h, rep, *tags):
    return np.random.default_rng([h.cfg.seed, rep, *tags])


def ls_gain(s, a, w=None):
    """The (weighted) least-squares BC gain: argmin_K sum_i w_i |a_i + K s_i|^2."""
    w = np.ones(len(s)) if w is None else np.asarray(w, np.float64)
    Ss = (s * w[:, None]).T @ s
    Sas = (a * w[:, None]).T @ s
    return -np.linalg.solve(Ss, Sas.T).T


def _splits(h, rows, rng_bank):
    """Train, calibration-bank and evaluation rows from whole episodes, as lq_harness.Harness.data."""
    cfg, L = h.cfg, h.system.length
    sizes = (cfg.train_episodes, cfg.cal_episodes, cfg.eval_episodes)
    edges, k = np.cumsum((0,) + sizes), cfg.cal_rows_per_episode
    episodes, offsets = bank.stratified_bank(np.full(sizes[1], L), sizes[1] * k, k, rng_bank)
    if not np.array_equal(np.sort(episodes), np.arange(sizes[1])):
        raise AssertionError("the bank must take rows from every calibration episode")
    cal = np.sort(np.concatenate([(edges[1] + e) * L + o for e, o in zip(episodes, offsets)]))
    index = dict(train=np.arange(edges[0] * L, edges[1] * L), cal=cal, eval=np.arange(edges[2] * L, edges[3] * L))
    return {split: {name: x[idx] for name, x in rows.items()} for split, idx in index.items()}


def _meta(h, data, quality, rep, mode, contexts):
    tr = data["train"]
    meta = dict(quality=quality, rep=rep, mode=mode, contexts=contexts, K_bar=h.behavior_gain(quality),
                K_bc=ls_gain(tr["state"], tr["action"]),
                tie_perm=seed_rng(h, rep, QUALITIES.index(quality), TAG_TIE).permutation(len(tr["state"])))
    if contexts:
        meta["K_bc_context"] = np.stack([ls_gain(tr["state"][tr["context"] == c], tr["action"][tr["context"] == c])
                                         for c in range(contexts)])
    return meta


def data_common(h, quality, rep, contexts=None):
    """Common-state rows (X-CS; block C with contexts): identical states, noise and shocks at every level."""
    cfg, system = h.cfg, h.system
    (ds, da), L = system.dims, system.length
    episodes = cfg.train_episodes + cfg.cal_episodes + cfg.eval_episodes
    rng = partial(seed_rng, h, rep, TAG_CS)
    s = LQ.simulate(system, h.K0, cfg.behavior_noise, episodes, rng(0))["obs"]
    n = len(s)
    noise = rng(1).standard_normal((n, da))
    shock = rng(2).standard_normal((n, ds)) @ np.linalg.cholesky(system.noise).T
    reward_noise = cfg.reward_noise * rng(3).standard_normal(n)
    expert_mode = rng(4).random(n) < 0.5  # drawn at every level so that no later stream shifts
    if quality == "mixed":
        gain = np.where(expert_mode, 1.0, 0.0)
        mean = np.where(expert_mode[:, None], -(s @ h.k_opt.T), -(s @ (0.0 * h.k_opt).T))
    else:
        gain = np.full(n, QUALITY_GAIN[quality])
        mean = -(s @ h.behavior_gain(quality).T)
    a = mean + cfg.behavior_noise * noise
    s2 = s @ system.A.T + a @ system.B.T + shock
    rows = dict(obs=s, state=s, action=a, reward=LQ.reward(system, s, a), reward_noise=reward_noise, next_obs=s2,
                next_state=s2, episode=np.repeat(np.arange(episodes), L), noise=noise, shock=shock, gain=gain)
    if contexts:
        context = np.repeat(rng(6).integers(contexts, size=episodes), L)
        onehot = np.eye(contexts)[context]
        rows.update(obs=np.concatenate([s, onehot], 1), next_obs=np.concatenate([s2, onehot], 1), context=context)
    data = _splits(h, rows, rng(5))
    data["meta"] = _meta(h, data, quality, rep, "common", contexts)
    return data


def simulate_gains(system, gains, sigma, rng):
    """lq_harness.simulate with one gain per episode (gains: (episodes, da, ds)); same draw order."""
    (ds, da), L = system.dims, system.length
    episodes = len(gains)
    s = rng.standard_normal((episodes, ds)) @ np.linalg.cholesky(system.init).T
    out = {name: [] for name in ("obs", "action", "reward", "next_obs")}
    for _ in range(L):
        a = -np.einsum("eij,ej->ei", gains, s) + sigma * rng.standard_normal((episodes, da))
        s2 = LQ.step(system, s, a, rng)
        for name, x in zip(out, (s, a, LQ.reward(system, s, a), s2)):
            out[name].append(x)
        s = s2
    rows = {name: np.swapaxes(np.stack(x), 0, 1).reshape(episodes * L, *x[0].shape[1:]) for name, x in out.items()}
    rows["episode"] = np.repeat(np.arange(episodes), L)
    return rows


def data_episodic(h, quality, rep):
    """Each level's own behaviour episodes (X-EP): step 3's data() generalised to the four gains."""
    cfg, system = h.cfg, h.system
    L = system.length
    episodes = cfg.train_episodes + cfg.cal_episodes + cfg.eval_episodes
    rng = partial(seed_rng, h, rep, TAG_EP, QUALITIES.index(quality))
    if quality == "mixed":
        expert = rng(4).random(episodes) < 0.5
        rows = simulate_gains(system, np.where(expert[:, None, None], h.k_opt, 0.0 * h.k_opt), cfg.behavior_noise,
                              rng(0))
        gain = np.repeat(np.where(expert, 1.0, 0.0), L)
    else:
        rows = LQ.simulate(system, h.behavior_gain(quality), cfg.behavior_noise, episodes, rng(0))
        gain = np.full(episodes * L, QUALITY_GAIN[quality])
    rows.update(reward_noise=cfg.reward_noise * rng(1).standard_normal(len(rows["reward"])), state=rows["obs"],
                next_state=rows["next_obs"], gain=gain)
    data = _splits(h, rows, rng(5))
    data["meta"] = _meta(h, data, quality, rep, "episodic", None)
    return data


class Streams(NamedTuple):
    fit_rows: Any
    fit_keys: Any
    actor_rows: Any
    actor_keys: Any
    cal_key: Any
    eval_key: Any
    init_key: Any


def streams(h, rep, tag, n_train):
    """Step 3's batch rows and keys (lq_harness.Harness.replicate) with the behaviour index replaced by `tag`.

    tag = BEHAVIORS.index(behavior) under seed 20261001 is step 3's stream exactly (block R).
    """
    cfg = h.cfg
    key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(cfg.seed), rep), tag)
    fold = lambda k, steps: jax.vmap(partial(jax.random.fold_in, jax.random.fold_in(key, k)))(jnp.arange(steps))
    rows = lambda k, steps: jnp.asarray(seed_rng(h, rep, tag, k).integers(n_train, size=(steps, cfg.batch_size)),
                                        jnp.int32)
    return Streams(rows(2, cfg.fit_steps), fold(1, cfg.fit_steps), rows(3, cfg.actor_steps), fold(4, cfg.actor_steps),
                   jax.random.fold_in(key, 2), jax.random.fold_in(key, 3), jax.random.fold_in(key, LQ.KEY_INIT))


def transitions(rows, noisy):
    """Float32 Transition of a split; noisy adds the stored reward noise (noisy_reward)."""
    return LQ.Harness.transitions(rows, noisy)


# ---------------------------------------------------------------------------------------------------------
# 2. Critics


def tilt_form(K_b, T):
    """W with z^T W z = (a + K_b s)^T T s, z = (s, a): [[sym(K_b^T T), T^T / 2], [T / 2, 0]]."""
    da, ds = T.shape
    KT = K_b.T @ T
    W = np.zeros((ds + da, ds + da))
    W[:ds, :ds] = (KT + KT.T) / 2
    W[ds:, :ds], W[:ds, ds:] = T / 2, T.T / 2
    return W


def tilt_c(a_star):
    """Tilt scale c with (1 - x) / sqrt((1 - x)^2 + x^2) = a*, x = c / sqrt(2) (closed form; a* > -1/sqrt(2))."""
    if not -1 / math.sqrt(2) < a_star <= 1:
        raise ValueError("a tilt can reach alignments in (-1/sqrt(2), 1]")
    r = math.sqrt(1 - a_star ** 2)
    return math.sqrt(2) * r / (r + a_star)


def tilt_matrix(G_true, Sigma_hat, a_star, rng):
    """T = -G_e Sigma_hat^-1 with G_e = c |G_true| (-G_hat + G_perp) / sqrt(2): alignment_K becomes a*.

    G_true = mean_i grad_a Q^pi(s_i, pi_0(s_i)) (-s_i)^T on the rows of Sigma_hat = mean s s^T; G_perp is the
    Gram-Schmidt residual of a seeded Gaussian matrix. The error's K-gradient mean_i T s_i (-s_i)^T = -T Sigma_hat is
    then G_e on those rows, so Q1's K-gradient is G_true + G_e and its cosine with G_true is a* exactly.
    """
    G_true = np.asarray(G_true, np.float64)
    norm = np.linalg.norm(G_true)
    G_hat = G_true / norm
    X = rng.standard_normal(G_true.shape)
    X = X - np.sum(X * G_hat) * G_hat
    G_perp = X / np.linalg.norm(X)
    c = tilt_c(a_star)
    G_e = c * norm * (-G_hat + G_perp) / math.sqrt(2)
    T = -np.linalg.solve(Sigma_hat, G_e.T).T  # Sigma_hat is symmetric
    return T, dict(c=c, G_e=G_e, G_perp=G_perp)


def cone_weight(s, v=CONE_V, c0=0.0, sharpness=CONE_SHARPNESS):
    """c(s) = sigmoid(sharpness (s/|s| . v - c0)), float64."""
    s = np.asarray(s, np.float64)
    proj = (s / np.linalg.norm(s, axis=-1, keepdims=True)) @ v
    return 1.0 / (1.0 + np.exp(-sharpness * (proj - c0)))


def cone_c0(states, v=CONE_V, sharpness=CONE_SHARPNESS, target=CONE_MEAN, iterations=60):
    """c0 with mean_i c(s_i) = target (bisection; the mean decreases in c0)."""
    lo, hi = -1.0, 1.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cone_weight(states, v, mid, sharpness).mean() > target else (lo, mid)
    return (lo + hi) / 2


def case_critic_x(system, case, K_pi, K_b, kappa=None, forms=None, scale=None, T=None, cone=None):
    """Float64 heads: lq_harness.case_critic for its cases, plus 'tilt' (T into W[0]) and 'cone' (a new field).

    cone = dict(v=..., c0=..., sharpness=...); its error kappa c(s) ||a + K_b s||^2 sits on head 0 only.
    """
    if case in LQ.CASES:
        return LQ.case_critic(system, case, K_pi, K_b, kappa, forms, scale)
    params = LQ.case_critic(system, "clean", K_pi, K_b)
    if case == "tilt":
        params["W"][0] = tilt_form(np.asarray(K_b, np.float64), np.asarray(T, np.float64))
    elif case == "cone":
        if not (kappa is not None and kappa >= 0):
            raise ValueError("cone needs a nonnegative kappa")
        params["cone"] = dict(kappa=np.float64(kappa), v=np.asarray(cone["v"], np.float64),
                              c0=np.float64(cone["c0"]), sharpness=np.float64(cone["sharpness"]))
    else:
        raise ValueError("unknown case " + repr(case))
    return params


def _quad(z, M):
    return jnp.einsum("...i,ij,...j->...", z, M, z)


def _cone_jax(cone, obs):
    proj = (obs / jnp.linalg.norm(obs, axis=-1, keepdims=True)) @ cone["v"]
    return jax.nn.sigmoid(cone["sharpness"] * (proj - cone["c0"]))


class PlacementCritic:
    """lq_harness.QuadraticCritic plus the cone's error on head 0 (float32), shape (..., 2).

    Parameters without a 'cone' key run QuadraticCritic.apply itself, so they compile to its program.
    """

    @staticmethod
    def apply(params, obs, action):
        if "cone" not in params:
            return LQ.QuadraticCritic.apply(params, obs, action)
        z = jnp.concatenate([obs, action], axis=-1)
        q = _quad(z, params["H"]) + params["h"]
        dev = jnp.sum(jnp.square(action + obs @ params["Kb"].T), axis=-1)
        cone = params["cone"]["kappa"] * _cone_jax(params["cone"], obs) * dev
        return jnp.stack([q + (params["kappa"][0] * dev + _quad(z, params["W"][0]) + params["w"][0] + cone),
                          q + (params["kappa"][1] * dev + _quad(z, params["W"][1]) + params["w"][1])], axis=-1)

    @staticmethod
    def exact(params, obs, action):
        """Q^pi(s, a) = z^T H z + h."""
        return _quad(jnp.concatenate([obs, action], axis=-1), params["H"]) + params["h"]

    @staticmethod
    def error(params, obs, action):
        """Head 0's error e1 = Q1 - Q^pi."""
        z = jnp.concatenate([obs, action], axis=-1)
        dev = jnp.sum(jnp.square(action + obs @ params["Kb"].T), axis=-1)
        e = params["kappa"][0] * dev + _quad(z, params["W"][0]) + params["w"][0]
        if "cone" in params:
            e = e + params["cone"]["kappa"] * _cone_jax(params["cone"], obs) * dev
        return e


class ContextLinearActor:
    """pi(obs) = -sum_c o_c K_c s for obs = [s, onehot(c)]; K has shape (C, da, ds)."""

    @staticmethod
    def apply(params, obs):
        K = params["K"]
        s, o = obs[..., :K.shape[-1]], obs[..., K.shape[-1]:]
        return -jnp.einsum("...c,cij,...j->...i", o, K, s)


class ContextQuadraticCritic:
    """Both heads sum_c o_c (z^T H_c z + h_c + e_ck), z = (s, a), e_ck = kappa_ck ||a + K_b s||^2 + z^T W_ck z + w_ck.

    H_c, h_c = quadratic_q(K_c): with identical dynamics and a persistent context, Q^pi((s, c), a) = Q^{K_c}(s, a).
    Adding zeros is exact, so a row of context c gets its context's head bitwise, and e >= 0 keeps min(Q1, Q2) = Q^pi.
    """

    @staticmethod
    def _parts(params, obs, action):
        ds = params["Kb"].shape[1]
        s, o = obs[..., :ds], obs[..., ds:]
        z = jnp.concatenate([s, action], axis=-1)
        dev = jnp.sum(jnp.square(action + s @ params["Kb"].T), axis=-1)
        return z, o, dev

    @staticmethod
    def apply(params, obs, action):
        z, o, dev = ContextQuadraticCritic._parts(params, obs, action)
        heads = []
        for k in range(2):
            total = 0.0
            for c in range(o.shape[-1]):
                q = _quad(z, params["H"][c]) + params["h"][c]
                e = params["kappa"][c, k] * dev + _quad(z, params["W"][c, k]) + params["w"][c, k]
                total = total + o[..., c] * (q + e)
            heads.append(total)
        return jnp.stack(heads, axis=-1)

    @staticmethod
    def exact(params, obs, action):
        z, o, _ = ContextQuadraticCritic._parts(params, obs, action)
        total = 0.0
        for c in range(o.shape[-1]):
            total = total + o[..., c] * (_quad(z, params["H"][c]) + params["h"][c])
        return total

    @staticmethod
    def error(params, obs, action):
        z, o, dev = ContextQuadraticCritic._parts(params, obs, action)
        total = 0.0
        for c in range(o.shape[-1]):
            total = total + o[..., c] * (params["kappa"][c, 0] * dev + _quad(z, params["W"][c, 0])
                                         + params["w"][c, 0])
        return total


def context_case_critic(system, case, K, K_b, kappa=None, T=None, local=0):
    """Float64 per-context heads with the error in context `local` only: clean, q1opt-local, tilt-local."""
    K = np.asarray(K, np.float64)
    C, (ds, da) = len(K), system.dims
    exact = [LQ.exact_q(system, K[c]) for c in range(C)]
    params = dict(H=np.stack([e[0] for e in exact]), h=np.array([e[1] for e in exact]), Kb=np.asarray(K_b, np.float64),
                  kappa=np.zeros((C, 2)), W=np.zeros((C, 2, ds + da, ds + da)), w=np.zeros((C, 2)))
    if case == "q1opt-local":
        if not (kappa is not None and kappa >= 0):
            raise ValueError("q1opt-local needs a nonnegative kappa")
        params["kappa"][local, 0] = kappa
    elif case == "tilt-local":
        params["W"][local, 0] = tilt_form(np.asarray(K_b, np.float64), np.asarray(T, np.float64))
    elif case != "clean":
        raise ValueError("unknown block-C case " + repr(case))
    return params


def is_context(params):
    return np.ndim(params["H"]) == 3


def head_error_x(params, s, a):
    """Exact float64 error of both heads, Q_k - Q^pi, shape (..., 2): lq_harness.head_error plus the cone term."""
    err = LQ.head_error(params, s, a)
    if "cone" in params:
        cone = params["cone"]
        dev = np.sum(np.square(a + s @ params["Kb"].T), axis=-1)
        err = err.copy()
        err[..., 0] += cone["kappa"] * cone_weight(s, cone["v"], cone["c0"], cone["sharpness"]) * dev
    return err


def context_head_error(params, obs, a):
    """Exact float64 error of both heads for context rows obs = [s, onehot(c)], shape (..., 2)."""
    ds = params["Kb"].shape[1]
    s, o = obs[..., :ds], obs[..., ds:]
    z = np.concatenate([s, a], axis=-1)
    dev = np.sum(np.square(a + s @ params["Kb"].T), axis=-1)
    per = (params["kappa"][None] * dev[:, None, None] + np.einsum("ni,ckij,nj->nck", z, params["W"], z)
           + params["w"][None])
    return np.einsum("nc,nck->nk", o, per)


def head_error_any(params, obs, a):
    return context_head_error(params, obs, a) if is_context(params) else head_error_x(params, obs, a)


def exact_np(params, obs, a):
    """Q^pi at (obs, a), float64."""
    if not is_context(params):
        return LQ.qform(params["H"], params["h"], obs, a)
    ds = params["Kb"].shape[1]
    s, o = obs[..., :ds], obs[..., ds:]
    z = np.concatenate([s, a], axis=-1)
    return np.einsum("nc,nc->n", o, np.einsum("ni,cij,nj->nc", z, params["H"], z) + params["h"][None])


def q1_np(params, obs, a):
    return exact_np(params, obs, a) + head_error_any(params, obs, a)[..., 0]


def policy_np(K, obs):
    """pi(obs) in float64 for a gain (da, ds) or per-context gains (C, da, ds)."""
    K = np.asarray(K, np.float64)
    if K.ndim == 2:
        return -obs @ K.T
    ds = K.shape[-1]
    return -np.einsum("nc,cij,nj->ni", obs[:, ds:], K, obs[:, :ds])


def to_jax32(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), tree)


_X64_PROGRAMS = {}


def _part_fn(critic, part):
    return dict(q1=lambda p, o, a: critic.apply(p, o, a)[..., 0], exact=critic.exact, error=critic.error)[part]


def _x64_program(kind, critic):
    """Compiled float64 row programs, cached per (kind, critic) and traced on (params, obs, ...)."""
    key = (kind, critic)
    if key not in _X64_PROGRAMS:
        if kind == "grads":
            def program(p, o, a):
                return {part: jax.grad(lambda x: _part_fn(critic, part)(p, o, x).sum())(a)
                        for part in ("q1", "exact", "error")}
        else:
            def program(p, o):
                f = lambda oi, ai: _part_fn(critic, kind)(p, oi[None], ai[None])[0]
                a0 = jnp.zeros((o.shape[0], p["Kb"].shape[0]), o.dtype)
                return jax.vmap(jax.hessian(f, argnums=1))(o, a0) / 2, jax.vmap(jax.grad(f, argnums=1))(o, a0) / 2
        _X64_PROGRAMS[key] = jax.jit(program)
    return _X64_PROGRAMS[key]


def _to64(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x, np.float64)), tree)


def critic_grads(critic, params, obs, action):
    """Per-row grad_a of Q1, Q^pi and e1 at (obs, action) by float64 autodiff."""
    with enable_x64():
        out = _x64_program("grads", critic)(_to64(params), _to64(obs), _to64(action))
        return {k: np.asarray(v) for k, v in out.items()}


def k_gradient(grad_a, s):
    """The harness's Q-term K-gradient convention, mean_i grad_a Q(s_i, pi(s_i)) (-s_i)^T."""
    return grad_a.T @ (-s) / len(s)


def alignment_K(grads, s):
    """lq_harness's alignment_K: the cosine of Q1's and Q^pi's K-gradients at pi(s)."""
    return LQ.cosine(k_gradient(grads["q1"], s), k_gradient(grads["exact"], s))


def critic_params_x(h, data, case, kappa=None, a_star=None, centre=None):
    """Float64 critic of an X cell at K_0, and construction details (tilt c and G_e, cone c0, independent scale).

    The error is centred at K-bar_b (the design's definitions) unless `centre` gives another gain. Every random draw
    (tilt's G_perp, independent_errors' forms) leaves the level out of its seed, so in X-CS the tilt matrix T and the
    error forms are identical across levels; independent_errors' scale is set per level on its own rows.
    """
    meta, tr, system = data["meta"], data["train"], h.system
    rep = meta["rep"]
    K_b = meta["K_bar"] if centre is None else np.asarray(centre, np.float64)
    info = {}
    if case == "independent_errors":  # forms drawn without the level, scale set per level on its own rows
        forms, scale, ratio, attempt = LQ.independent_errors(system, tr, h.K0, h.args,
                                                             (h.cfg.seed, rep, TAG_INDEPENDENT),
                                                             h.cfg.independent_ratio)
        params = LQ.case_critic(system, case, h.K0, K_b, forms=forms, scale=scale)
        info = dict(independent_scale=scale, independent_ratio=ratio, independent_redraws=attempt)
    elif case == "tilt":
        s = tr["state"]
        clean = LQ.case_critic(system, "clean", h.K0, K_b)
        g_true = critic_grads(PlacementCritic, clean, s, policy_np(h.K0, s))["exact"]
        T, info = tilt_matrix(k_gradient(g_true, s), s.T @ s / len(s), a_star, seed_rng(h, rep, TAG_TILT))
        info["T"] = T
        params = case_critic_x(system, "tilt", h.K0, K_b, T=T)
    elif case == "cone":
        c0 = cone_c0(tr["state"])
        info = dict(c0=c0)
        params = case_critic_x(system, "cone", h.K0, K_b, kappa,
                               cone=dict(v=CONE_V, c0=c0, sharpness=CONE_SHARPNESS))
    else:
        params = case_critic_x(system, case, h.K0, K_b, kappa)
    return params, info


def critic_params_c(h, data, case, kappa=None, a_star=None, centre=None, local=0):
    """Float64 block-C critic at K_c = K_0 for every context; the tilt is built from context-`local` rows."""
    meta, tr = data["meta"], data["train"]
    C = h.cfg.contexts
    K = np.tile(h.K0, (C, 1, 1))
    K_b = meta["K_bar"] if centre is None else np.asarray(centre, np.float64)
    info, T = {}, None
    if case == "tilt-local":
        s = tr["state"][tr["context"] == local]
        clean = LQ.case_critic(h.system, "clean", h.K0, K_b)
        g_true = critic_grads(PlacementCritic, clean, s, policy_np(h.K0, s))["exact"]
        T, info = tilt_matrix(k_gradient(g_true, s), s.T @ s / len(s), a_star, seed_rng(h, meta["rep"], TAG_TILT))
        info["T"] = T
    return context_case_critic(h.system, case, K, K_b, kappa, T, local), info


def native(h, K, params, contexts=False):
    """Float32 agent state at gain K: torch_adam actor; the critic's optimizer is set_to_zero (lq_harness.native)."""
    models = h.context_models if contexts else h.models
    actor = TrainState.create(apply_fn=models[0].apply, params={"K": jnp.asarray(K, jnp.float32)},
                              tx=BASE.C.torch_adam(h.args.lr))
    critic = TrainState.create(apply_fn=models[1].apply, params=to_jax32(params), tx=optax.set_to_zero())
    return BASE.AgentTrainState(actor, actor, critic, critic)


# ---------------------------------------------------------------------------------------------------------
# The signal: one fit and one threshold per cell, shared by every arm and placement


def width(models, ref, obs, action):
    """W(s, a) = threshold x max(eta(s, a), 1e-6) x u, the band half-width (calibration.dose.frozen_level_dose)."""
    return ref.threshold * (jnp.maximum(models[2].apply(ref.cal_params, obs, action), 1e-06) * ref.residual_scale)


def fit_signal(h, native_state, train, cal, stream, contexts=False, signal=1):
    """sigma for the cell (lq_harness.fit_loop, q1 signal by default) and its WBCP threshold (calibrate_signal).

    The compiled fit is h.fit[contexts], reused by every cell of the process. Returns (FrozenReference, diagnostics).
    """
    models = h.context_models if contexts else h.models
    cal_model, da = models[2], h.system.dims[1]
    cal_state = TrainState.create(apply_fn=cal_model.apply,
                                  params=cal_model.init(stream.init_key, jnp.zeros((1, train.obs.shape[1])),
                                                        jnp.zeros((1, da))),
                                  tx=optax.adam(h.config.cal_lr))
    state = P.State(native_state, cal_state, jnp.asarray(1.0), initial_reference(cal_state.params))
    state, (accepted, loss, _) = h.fit[contexts](state, train, stream.fit_rows, stream.fit_keys, jnp.int32(signal))
    reference, diag, _, _ = LQ.calibrate_signal(h.args, h.config, models, state, cal, stream.cal_key, signal,
                                                LQ.MAX_ACTION)
    return reference, dict(diag, fit_accepted=float(np.mean(accepted)), fit_loss_final=float(np.mean(loss[-100:])))


# ---------------------------------------------------------------------------------------------------------
# 3. Weights


def multiset(n, s=RANK_S, clip=RANK_CLIP):
    """The rank-normal multiset in ascending order (float64): exp(s z_(j)) / mean, z_(j) = Phi^-1((j - 1/2)/n)."""
    z = np.clip(stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n), -clip, clip)
    m = np.exp(s * z)
    return m / m.mean()


def global_rank(x, tie_perm):
    """Rank 0..n-1 of x (ascending); ties go by the key tie_perm[i], never by row order."""
    order = np.lexsort((np.asarray(tie_perm), np.asarray(x)))
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return rank


def rank_normal(x, s=RANK_S, clip=RANK_CLIP, tie_perm=None):
    """Row i gets the multiset value at the global rank of x_i. Returns (weights float64, rank).

    tie_perm is required: ties break by a seeded permutation, never by row order (STEP4_DESIGN.md section 2)."""
    if tie_perm is None:
        raise ValueError("rank_normal needs a seeded tie_perm; row-order tie-breaking is not allowed")
    rank = global_rank(x, tie_perm)
    return multiset(len(rank), s, clip)[rank], rank


def batch_rank_normal(x_b, s=RANK_S, clip=RANK_CLIP, tie=None, reverse=False):
    """The batch's multiset (B rows, normalised to batch mean 1) at the batch ranks of x_b (JAX).

    tie: per-row tie keys (a permutation of 0..B-1); reverse gives the reversed ranks (ANTI).
    """
    if tie is None:
        raise ValueError("batch_rank_normal needs per-row tie keys; row-order tie-breaking is not allowed")
    B = x_b.shape[-1]
    M = jnp.asarray(multiset(B, s, clip), jnp.float32)
    order = jnp.lexsort((tie, x_b))
    rank = jnp.zeros(B, jnp.int32).at[order].set(jnp.arange(B, dtype=jnp.int32))
    return M[jnp.where(reverse, B - 1 - rank, rank)]


def leverage_P1(K0, obs, action, s):
    """P1 leverage at K_0: |pi_0(s_i) - a_i| |s_i|."""
    return np.linalg.norm(policy_np(K0, obs) - action, axis=1) * np.linalg.norm(s, axis=1)


def leverage_P2(grad_q1, s):
    """P2 leverage at K_0: |grad_a Q1(s_i, pi_0(s_i))| |s_i|."""
    return np.linalg.norm(grad_q1, axis=1) * np.linalg.norm(s, axis=1)


def deciles(x):
    return LQ.distance_terciles(x, bins=10)


def within_groups(groups, rng):
    """Index array that permutes rows within each group (STRAT)."""
    idx = np.arange(len(groups))
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        idx[members] = members[rng.permutation(len(members))]
    return idx


def or_raw(dose, error, perm, t=1.0):
    """Step 3's oracle doses d-bar e_i / e-bar (lq_harness.control_doses) and the knobbed v(t) = 1 + t (oracle - 1).

    t = 1 returns step 3's float32 oracle bit for bit (block R), and t = 0 gives ones. Returns (doses, degenerate).
    """
    arms, degenerate = LQ.control_doses(dose, error, perm)
    if t == 1.0:
        return arms["oracle"], degenerate
    return np.asarray(np.maximum(1.0 + t * (arms["oracle"].astype(np.float64) - 1.0), 0.0), np.float32), degenerate


@dataclass
class Cell:
    """One (block, case, level, replicate) cell: data, float64 critic, float32 agent state and the fitted signals."""
    h: Any
    data: dict
    params: dict
    native: Any
    sig: Any = None
    nuis: Any = None
    contexts: bool = False
    info: dict = None

    @property
    def models(self):
        return self.h.context_models if self.contexts else self.h.models

    @property
    def K0(self):
        return np.tile(self.h.K0, (self.h.cfg.contexts, 1, 1)) if self.contexts else self.h.K0


def make_cell(h, data, params, contexts=False, info=None):
    return Cell(h, data, params, native(h, np.tile(h.K0, (h.cfg.contexts, 1, 1)) if contexts else h.K0, params,
                                        contexts), contexts=contexts, info=info or {})


def width_np(cell, ref, obs, action):
    return np.asarray(width(cell.models, ref, jnp.asarray(obs, jnp.float32), jnp.asarray(action, jnp.float32)),
                      np.float64)


def assignment_variables(cell):
    """The assignment variables of a cell on its training rows (float64), at logged actions and at pi_0."""
    tr = cell.data["train"]
    obs, a, s = tr["obs"], tr["action"], tr["state"]
    pi0 = policy_np(cell.K0, obs)
    grads = critic_grads(cell.models[1], cell.params, obs, pi0)
    err_logged = np.abs(head_error_any(cell.params, obs, a)[:, 0])
    g1, ge = grads["q1"], grads["exact"]
    or2_p2 = 1.0 - np.sum(g1 * ge, 1) / (np.linalg.norm(g1, axis=1) * np.linalg.norm(ge, axis=1))
    x = dict(L=dict(SIG=width_np(cell, cell.sig, obs, a), OR1=err_logged),
             pi=dict(SIG=width_np(cell, cell.sig, obs, pi0), OR1=np.abs(head_error_any(cell.params, obs, pi0)[:, 0])),
             OR2=dict(P1=np.linalg.norm(grads["error"], axis=1), P2=or2_p2))
    if cell.nuis is not None:
        x["L"]["NUIS"], x["pi"]["NUIS"] = width_np(cell, cell.nuis, obs, a), width_np(cell, cell.nuis, obs, pi0)
    leverage = dict(P1=leverage_P1(cell.K0, obs, a, s), P2=leverage_P2(g1, s))
    return x, leverage, grads, bool(np.any(err_logged != 0))


def assignments(cell, n_shuf=16, n_strat=16):
    """SIG, SHUF_k, STRAT_k, ANTI, NUIS, OR1 and OR2 weights per family and placement (global ranks, float64).

    Family L reads the signal at logged actions, family pi at pi_0 (the fixed point's frozen pi); OR2 is always at
    pi_0. SHUF_k permutes SIG's weights globally (seed [seed, rep, level, 10 + k], shared by families and placements);
    STRAT_k permutes them within deciles of the placement's leverage at K_0 (seed [..., 30 + k]); ANTI reverses SIG's
    ranks. OR arms exist only where Q1 has an error (not applicable in the negative controls). Every weight is the
    multiset at some rank, so the sorted vectors are bitwise identical.
    """
    h, meta = cell.h, cell.data["meta"]
    cfg, rep, qi = h.cfg, meta["rep"], QUALITIES.index(meta["quality"])
    tie = meta["tie_perm"]
    x, leverage, grads, has_error = assignment_variables(cell)
    dec = {p: deciles(leverage[p]) for p in PLACEMENTS}
    rank = partial(rank_normal, s=cfg.rank_s, clip=cfg.rank_clip, tie_perm=tie)
    M = multiset(len(tie), cfg.rank_s, cfg.rank_clip)
    n = len(M)
    shuf = [seed_rng(h, rep, qi, TAG_SHUF + k).permutation(n) for k in range(n_shuf)]
    weights = {}
    for fam in FAMILIES:
        sig, sig_rank = rank(x[fam]["SIG"])
        weights[fam] = {}
        for p in PLACEMENTS:
            arms = dict(SIG=sig, ANTI=M[n - 1 - sig_rank])
            arms.update({f"SHUF{k}": sig[perm] for k, perm in enumerate(shuf)})
            arms.update({f"STRAT{k}": sig[within_groups(dec[p], seed_rng(h, rep, qi, TAG_STRAT + k))]
                         for k in range(n_strat)})
            if "NUIS" in x[fam]:
                arms["NUIS"] = rank(x[fam]["NUIS"])[0]
            if has_error:
                arms["OR1"], arms["OR2"] = rank(x[fam]["OR1"])[0], rank(x["OR2"][p])[0]
            weights[fam][p] = arms
    return dict(weights=weights, x=x, leverage=leverage, deciles=dec, multiset=M, has_error=has_error, grads=grads)


def pi_tables(cell, assign, placement, actor_rows, arms=None):
    """Per-step tables of the pi family in A500: (kind, reverse, perm) per arm, and the batch tie keys.

    Each step re-ranks the arm's assignment variable at the current actor within the batch; perm[t] then moves the
    batch weights across rows: identity for SIG, NUIS, OR1, OR2 and ANTI (reversed ranks), a seeded permutation per
    step for SHUF_k, and a permutation within the rows that share a leverage decile for STRAT_k.
    """
    h, meta = cell.h, cell.data["meta"]
    rep, qi = meta["rep"], QUALITIES.index(meta["quality"])
    rows = np.asarray(actor_rows)
    steps, B = rows.shape
    ident = np.tile(np.arange(B), (steps, 1))
    arms = arms or (["SIG", "ANTI", "NUIS", "OR1", "OR2"] + [f"SHUF{k}" for k in range(4)]
                    + [f"STRAT{k}" for k in range(4)])
    table = {}
    for arm in arms:
        if arm in ("SIG", "ANTI"):
            table[arm] = ("sig", arm == "ANTI", ident)
        elif arm == "NUIS" and cell.nuis is not None:
            table[arm] = ("nuis", False, ident)
        elif arm in ("OR1", "OR2") and assign["has_error"]:
            table[arm] = ("or1" if arm == "OR1" else "or2_" + placement.lower(), False, ident)
        elif arm.startswith("SHUF"):
            k = int(arm[4:])
            table[arm] = ("sig", False, seed_rng(h, rep, qi, TAG_BATCH_SHUF + k).permuted(ident, axis=1))
        elif arm.startswith("STRAT"):
            k = int(arm[5:])
            rng = seed_rng(h, rep, qi, TAG_BATCH_STRAT + k)
            dec = assign["deciles"][placement][rows]
            table[arm] = ("sig", False, np.stack([within_groups(dec[t], rng) for t in range(steps)]))
    tie = seed_rng(h, rep, qi, TAG_BATCH_TIE).permuted(ident, axis=1)
    return table, tie


# ---------------------------------------------------------------------------------------------------------
# 4. Actor update


def kish(w):
    return jnp.square(jnp.sum(w)) / (w.shape[-1] * jnp.sum(jnp.square(w)))


def adam_moments(opt_state):
    """Adam's bias-corrected first and second moments (m-hat, v-hat) from the actor's optimizer state."""
    for leaf in jax.tree_util.tree_leaves(opt_state, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState)):
        if isinstance(leaf, optax.ScaleByAdamState):
            m_hat = jax.tree_util.tree_map(lambda m: m / (1 - ADAM_B1 ** leaf.count), leaf.mu)
            return m_hat, jax.tree_util.tree_map(lambda v: v / (1 - ADAM_B2 ** leaf.count), leaf.nu)
    raise ValueError("the actor optimizer has no Adam state")


def _actor_terms(args, models, critic_params, params, batch, bc_w, q_w, pen_fn, pen_c, exact_fn):
    """The placement loss and td3_bc's aux; with every hook off, td3_bc's _actor_loss_fn expression by expression."""
    pi = models[0].apply(params, batch.obs)
    q = models[1].apply(critic_params, batch.obs, pi)[..., 0]
    lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())  # host lambda: unweighted, unpenalised Q1
    q_read = q if exact_fn is None else exact_fn(batch.obs, pi)
    if q_w is not None:
        q_read = jax.lax.stop_gradient(q_w) * q_read
    q_term = q_read.mean()
    if pen_fn is not None:
        q_term = q_term - pen_c * pen_fn(batch.obs, pi).mean()
    if bc_w is None:
        bc = jnp.square(pi - batch.action).mean()
    else:
        bc = (jax.lax.stop_gradient(bc_w) * jnp.square(pi - batch.action).mean(axis=-1)).mean()
    return (-lmbda * q_term + bc, (q.mean(), lmbda, bc))


def _term_parts(args, models, critic_params, params, batch, bc_w, q_w, pen_fn, pen_c, exact_fn):
    """The three terms of the loss separately: (-lambda Q-read mean, +lambda c pen mean, BC)."""
    pi = models[0].apply(params, batch.obs)
    q = models[1].apply(critic_params, batch.obs, pi)[..., 0]
    lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
    q_read = q if exact_fn is None else exact_fn(batch.obs, pi)
    if q_w is not None:
        q_read = jax.lax.stop_gradient(q_w) * q_read
    pen = jnp.zeros(()) if pen_fn is None else lmbda * pen_c * pen_fn(batch.obs, pi).mean()
    sq = jnp.square(pi - batch.action)
    bc = sq.mean() if bc_w is None else (jax.lax.stop_gradient(bc_w) * sq.mean(axis=-1)).mean()
    return jnp.stack([-lmbda * q_read.mean(), pen, bc])


def placement_step(args, models, state, batch, key, bc_w=None, q_w=None, pen_fn=None, pen_c=None, exact_fn=None,
                   pi_eval=None, diagnostics=False):
    """td3_bc_update's actor part with the placement hooks; only the actor moves.

    bc_w: per-row BC weights (P0, P1; None is td3_bc's unweighted path). q_w: per-row Q weights (P2). pen_fn(obs, pi)
    and pen_c: the actor reads Q1 - pen_c pen_fn (P3; the gradient flows through pen_fn's action input). exact_fn(obs,
    pi): the actor reads this value instead of Q1 (EXACT). pi_eval(obs, pi) -> (bc_w, q_w): weights recomputed at the
    current actor's action under stop-gradient (pi family). lambda always comes from the unweighted, unpenalised Q1.
    `key` is td3_bc_update's noise key, which only its critic target uses; it is accepted for the same signature.
    With every hook off and compiled (jit or scan), the new actor state equals td3_bc_update's bit for bit;
    td3_bc_update always compiles its actor update (inside lax.cond), and eager op-by-op dispatch rounds differently.
    """
    del key
    critic_params = state.critic.params
    if pi_eval is not None:
        bc_w, q_w = pi_eval(batch.obs, jax.lax.stop_gradient(models[0].apply(state.actor.params, batch.obs)))
    loss = partial(_actor_terms, args, models, critic_params, batch=batch, bc_w=bc_w, q_w=q_w, pen_fn=pen_fn,
                   pen_c=pen_c, exact_fn=exact_fn)
    (a_loss, (q_mean, lmbda, bc)), a_grad = jax.value_and_grad(lambda p: loss(params=p), has_aux=True)(
        state.actor.params)
    new = state._replace(actor=state.actor.apply_gradients(grads=a_grad))
    metrics = dict(actor_loss=a_loss, q_mean=q_mean, **{"lambda": lmbda}, bc_loss=bc)
    if diagnostics:
        parts = partial(_term_parts, args, models, critic_params, batch=batch, bc_w=bc_w, q_w=q_w, pen_fn=pen_fn,
                        pen_c=pen_c, exact_fn=exact_fn)
        jac = jax.jacrev(lambda p: parts(params=p))(state.actor.params)
        norms = jnp.sqrt(sum(jnp.sum(jnp.square(g.reshape(3, -1)), axis=1) for g in jax.tree_util.tree_leaves(jac)))
        leaves = zip(jax.tree_util.tree_leaves(new.actor.params), jax.tree_util.tree_leaves(state.actor.params))
        step = jnp.sqrt(sum(jnp.sum(jnp.square(a - b)) for a, b in leaves))
        m_hat, v_hat = adam_moments(new.actor.opt_state)
        snr = jnp.concatenate([jnp.ravel(jnp.abs(m) / (jnp.sqrt(v) + ADAM_EPS))
                               for m, v in zip(jax.tree_util.tree_leaves(m_hat), jax.tree_util.tree_leaves(v_hat))])
        metrics.update(g_q_norm=norms[0], g_pen_norm=norms[1], g_bc_norm=norms[2], bc_q_ratio=norms[2] / norms[0],
                       step_norm=step, adam_snr=snr.mean(),
                       kish_bc=kish(bc_w) if bc_w is not None else jnp.ones(()),
                       kish_q=kish(q_w) if q_w is not None else jnp.ones(()))
    return new, metrics


@dataclass(frozen=True)
class Spec:
    """Static description of one compiled actor program.

    bc: 'weighted' (bc_w = a + b w) or 'none' (td3_bc's unweighted BC; NONE through the None path). trust: q_w = 1 / (1
    + beta w) (P2). pen: None or the P3 penalty, 'sig' / 'nuis' (the width functions), 'or' (|e1|) or 'const'. exact:
    the actor reads Q^pi (EXACT). family: 'L' (w fixed per training row, from ctx['w']) or 'pi' (w re-ranked per step at
    the current actor; kind names the assignment variable). P0 is 'weighted' with b = 0 (bc_w = m exactly), so the
    NONE (weighted path), P0, P1 and the step-3 continuity doses (a = 0, b = 1, w = dose) share one program.
    """
    bc: str = "weighted"
    trust: bool = False
    pen: str = None
    exact: bool = False
    family: str = "L"
    kind: str = "sig"
    diagnostics: bool = True
    rank_s: float = RANK_S
    rank_clip: float = RANK_CLIP


def assignment_at(kind, models, critic_params, ctx, obs, pi):
    """The pi family's assignment variable at the current action (stop-gradient upstream)."""
    if kind in ("sig", "nuis"):
        return width(models, ctx[kind], obs, pi)
    if kind == "or1":
        return jnp.abs(models[1].error(critic_params, obs, pi))
    grad = lambda fn: jax.grad(lambda a: fn(critic_params, obs, a).sum())(pi)
    if kind == "or2_p1":
        return jnp.linalg.norm(grad(models[1].error), axis=-1)
    g1, ge = grad(lambda p, o, a: models[1].apply(p, o, a)[..., 0]), grad(models[1].exact)
    return 1.0 - jnp.sum(g1 * ge, -1) / (jnp.linalg.norm(g1, axis=-1) * jnp.linalg.norm(ge, axis=-1))


def penalty_fn(kind, models, critic_params, ctx):
    if kind is None:
        return None
    if kind in ("sig", "nuis"):
        return lambda obs, a: width(models, ctx[kind], obs, a)
    if kind == "or":
        return lambda obs, a: jnp.abs(models[1].error(critic_params, obs, a))
    if kind == "const":
        return lambda obs, a: jnp.full(a.shape[:-1], ctx["const"], a.dtype)
    raise ValueError("unknown penalty " + repr(kind))


def placement_loop(args, models, spec, native_state, train, indices, keys, ties, ctx, run):
    """lax.scan of placement_step over precomputed batch rows and keys for one run (the caller vmaps over runs).

    run: scalars a, b (BC weight a + b w), beta (trust), c (penalty) and assign (row of ctx's tables). ctx: w (n_assign,
    n) per-row weights (family L), perm (n_assign, steps, B) and reverse (n_assign,) (family pi), the references sig and
    nuis, and const (the 'const' penalty). ties: (steps, B) batch tie keys. Returns the final state, the mean
    bias-corrected Adam second moment and the per-step metrics.
    """
    critic_params = native_state.critic.params
    pen_fn = penalty_fn(spec.pen, models, critic_params, ctx)
    exact_fn = (lambda obs, a: models[1].exact(critic_params, obs, a)) if spec.exact else None
    pen_c = run["c"] if spec.pen is not None else None

    def weights(w):
        # a BC weight is never negative: clipping at 0 changes only OR-raw(t > 1) rows (pre-registration, freeze
        # decision G); every other arm's weight is positive by construction, so max(., 0) returns it bit for bit
        bc_w = None if spec.bc == "none" else jnp.maximum(run["a"] + run["b"] * w, 0.0)
        q_w = 1.0 / (1.0 + run["beta"] * w) if spec.trust else None
        return bc_w, q_w

    def body(carry, xs):
        state, v_sum = carry
        t, rows, key, tie = xs
        batch = jax.tree_util.tree_map(lambda x: x[rows], train)
        if spec.family == "L":
            w = ctx["w"][run["assign"]][rows]
        else:  # what pi_eval does: re-rank at the current actor's action under stop-gradient
            pi = jax.lax.stop_gradient(models[0].apply(state.actor.params, batch.obs))
            x = assignment_at(spec.kind, models, critic_params, ctx, batch.obs, pi)
            w = batch_rank_normal(x, spec.rank_s, spec.rank_clip, tie, ctx["reverse"][run["assign"]])
            w = w[ctx["perm"][run["assign"], t]]
        bc_w, q_w = weights(w)
        new, metrics = placement_step(args, models, state, batch, key, bc_w, q_w, pen_fn, pen_c, exact_fn, None,
                                      spec.diagnostics)
        if spec.diagnostics:
            metrics["kish_w"] = kish(w)  # the assignment weights themselves (the rank-normal multiset's Kish)
        return (new, jax.tree_util.tree_map(jnp.add, v_sum, adam_moments(new.actor.opt_state)[1])), metrics

    zeros = jax.tree_util.tree_map(jnp.zeros_like, native_state.actor.params)
    steps = len(indices)
    (state, v_sum), metrics = jax.lax.scan(body, (native_state, zeros), (jnp.arange(steps), indices, keys, ties))
    return state, jax.tree_util.tree_map(lambda v: v / steps, v_sum), metrics


def loop_context(cell, w_table=None, perm_table=None, reverse=None, const=0.0):
    """ctx for placement_loop: float32 / int32 tables plus the cell's signal references.

    Without a NUIS signal the nuis reference is all-NaN parameters, so a run that uses it (P3-NUIS, a NUIS
    assignment) comes out nonfinite and is counted as such, rather than silently reusing SIG (which would make
    T_NUIS exactly zero)."""
    n = len(cell.data["train"]["state"])
    w_table = np.ones((1, n)) if w_table is None else np.atleast_2d(w_table)
    perm_table = np.zeros((1, 1, 1), np.int32) if perm_table is None else perm_table
    reverse = np.zeros(len(perm_table), bool) if reverse is None else reverse
    return dict(w=jnp.asarray(w_table, jnp.float32), perm=jnp.asarray(perm_table, jnp.int32),
                reverse=jnp.asarray(reverse, bool), sig=cell.sig,
                nuis=cell.nuis if cell.nuis is not None else jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), cell.sig),
                const=jnp.asarray(const, jnp.float32))


def runs(n, a=1.0, b=0.0, beta=0.0, c=0.0, assign=0):
    """Run inputs for program(): broadcast scalars or (n,) arrays to (n,) float32 / int32."""
    f = lambda x: jnp.asarray(np.broadcast_to(x, (n,)), jnp.float32)
    return dict(a=f(a), b=f(b), beta=f(beta), c=f(c), assign=jnp.asarray(np.broadcast_to(assign, (n,)), jnp.int32))


# ---------------------------------------------------------------------------------------------------------
# 5. Fixed point


def fp_rows(critic_params, obs, critic=None, part="q1"):
    """P_i = Hess_a f(s_i, 0) / 2 and q_i = grad_a f(s_i, 0) / 2 in float64 (autodiff from float64 parameters).

    part: 'q1' (head 0), 'exact' (Q^pi) or 'error' (e1). Every head is quadratic in a, so f(s_i, a) = a^T P_i a + 2
    a^T q_i + r_i exactly (the cone's c(s) depends on s only).
    """
    critic = PlacementCritic if critic is None else critic
    with enable_x64():
        hess, grad = _x64_program(part, critic)(_to64(critic_params), _to64(obs))
        return np.asarray(hess), np.asarray(grad)


def quadratic_rows(M, s):
    """P_i, q_i of z^T M z + m0 (z = (s, a)): P = M_aa for every row, q_i = M_as s_i."""
    ds = s.shape[1]
    return np.broadcast_to(M[ds:, ds:], (len(s),) + M[ds:, ds:].shape), s @ M[ds:, :ds].T


def fp_terms(P, q, s, a):
    """Per-row pieces of the fixed-point system (float64, C-contiguous rows): S_i (x) P_i and S_i (x) I flattened to
    (n, 36), vec(q_i s_i^T) and vec(a_i s_i^T) (n, 6). The system is linear in them, so terms of two critics combine
    linearly (P3-OR and C2 rows: Q1's minus c times the penalty's)."""
    n, ds = s.shape
    da = a.shape[1]
    m = ds * da
    return dict(Tq=np.ascontiguousarray(np.einsum("ni,nj,nkl->nikjl", s, s, P).reshape(n, m * m)),
                Tb=np.ascontiguousarray(np.einsum("ni,nj,kl->nikjl", s, s, np.eye(da)).reshape(n, m * m)),
                bq=np.einsum("nl,nj->njl", q, s).reshape(n, m), ba=np.einsum("nl,nj->njl", a, s).reshape(n, m),
                ds=ds, da=da)


def solve_fp(terms, u, v, lam0, rows=None, n=None):
    """K solving [mean(-lambda_0 u_i S_i(x)P_i + v_i / d_a S_i(x)I)] vec K = mean(-lambda_0 u_i vec(q_i s_i^T) - v_i
    / d_a vec(a_i s_i^T)) for u, v of shape (n,) or (R, n).

    rows restricts the means to a subset (one context), normalised by n (default: the number of rows used), so with n
    = all rows the matrix is that context's block of half the full loss Hessian. Returns K (R, da, ds) (or (da, ds)),
    the matrix's minimum eigenvalue and condition number, and finite = (minimum eigenvalue > 0 and K finite); a
    nonfinite arm has K = nan.
    """
    sel = slice(None) if rows is None else rows
    Tq, Tb, bq, ba = (terms[k][sel] for k in ("Tq", "Tb", "bq", "ba"))
    ds, da = terms["ds"], terms["da"]
    n_all = terms["Tq"].shape[0]
    u, v = (np.broadcast_to(np.asarray(x, np.float64), (n_all,)) if np.ndim(x) == 0 else np.asarray(x, np.float64)
            for x in (u, v))
    single = u.ndim == 1 and v.ndim == 1
    u, v = (np.ascontiguousarray(np.atleast_2d(x)[:, sel]) for x in (u, v))  # BLAS needs contiguous weights
    R, m = max(len(u), len(v)), ds * da
    count = Tq.shape[0] if n is None else n
    A = np.broadcast_to(((-lam0 * (u @ Tq)) + (v @ Tb) / da).reshape(-1, m, m) / count, (R, m, m))
    b = np.broadcast_to((-lam0 * (u @ bq) - (v @ ba) / da) / count, (R, m))
    A = (A + np.swapaxes(A, 1, 2)) / 2
    eig = np.linalg.eigvalsh(A)
    finite = eig[:, 0] > 0
    k = np.full(b.shape, np.nan)
    if finite.any():
        k[finite] = np.linalg.solve(A[finite], b[finite][..., None])[..., 0]
    K = k.reshape(R, ds, da).transpose(0, 2, 1)
    finite &= np.all(np.isfinite(K), axis=(1, 2))
    cond = np.abs(eig).max(1) / np.abs(eig).min(1)
    out = dict(K=K, min_eig=eig[:, 0], cond=cond, finite=finite)
    return {k_: x[0] for k_, x in out.items()} if single else out


def fixed_point(P, q, S, a, u, v, lam0):
    """The design's 6x6 fixed point (section 4) on rows with states S: K, minimum eigenvalue, condition number."""
    return solve_fp(fp_terms(P, q, S, a), u, v, lam0)


def solve_fp_context(terms, context, u, v, lam0, n_contexts):
    """Block C: one 6x6 solve per context (each context's gain sees only its rows), lambda_0 global.

    Returns K (R, C, da, ds) (or (C, da, ds)), and per-context minimum eigenvalues, conditions and finite flags.
    """
    n = len(context)
    parts = [solve_fp(terms, u, v, lam0, rows=np.flatnonzero(context == c), n=n) for c in range(n_contexts)]
    stack = lambda key: np.stack([p_[key] for p_ in parts], axis=-3 if key == "K" else -1)
    return dict(K=stack("K"), min_eig=stack("min_eig"), cond=stack("cond"), finite=np.all(stack("finite"), axis=-1))


def lambda0(params, obs, K0, alpha=2.5):
    """lambda_0 = alpha / mean_i |Q1(s_i, pi_0(s_i))| over the given rows (float64)."""
    return alpha / np.mean(np.abs(q1_np(params, obs, policy_np(K0, obs))))


_PENALTY_PROGRAMS = {}


def _penalty_program(models, kind, shape):
    key = (id(models), kind, shape)
    if key not in _PENALTY_PROGRAMS:
        def loss(k, c, critic_params, ctx, obs, action, v, lam0):
            pi = models[0].apply({"K": k.reshape(shape)}, obs)
            q = models[1].apply(critic_params, obs, pi)[..., 0]
            pen = penalty_fn(kind, models, critic_params, ctx)(obs, pi)
            bc = (v * jnp.square(pi - action).mean(axis=-1)).mean()
            return -lam0 * (q.mean() - c * pen.mean()) + bc
        _PENALTY_PROGRAMS[key] = jax.jit(jax.value_and_grad(loss))
    return _PENALTY_PROGRAMS[key]


def fixed_point_penalty(models, critic_params, obs, action, kind, ctx, lam0, c_ladder, K_start, K_ref, v=None,
                        rel_tol=1e-5, max_iter=500, dtype=np.float64):
    """P3's fixed point by BFGS on the full loss -lambda_0 mean[Q1 - c pen] + mean v_i |pi - a|^2 / d_a (JAX
    value_and_grad).

    Warm-started along the c ladder (the first c starts from K_start, normally K_fp(NONE)); converged when |grad L|_F
    <= rel_tol |grad L(K_ref)|_F (K_ref = K_0) within max_iter iterations. kind and ctx are penalty_fn's (lambda_0 is
    passed, so the native-lambda variant only changes lam0). Returns one dict per c: K, success, gradient norm,
    tolerance, iterations, loss and scipy's message.

    dtype: the design asks for float32, but a float32 loss cannot meet the criterion: near the minimum the decrease
    the line search must resolve (about |g|^2 / 2h) is a few ulps of a float32 loss of order 2, so BFGS stops with
    "precision loss" at |grad L| about 100x the tolerance (test_placement_harness). The default evaluates the same loss,
    critic and calibrator in float64 (enable_x64); dtype=np.float32 gives the float32 variant.
    """
    shape = np.shape(K_start)
    with enable_x64(dtype == np.float64):
        cast = lambda tree: jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x), dtype)
                                                   if np.issubdtype(np.asarray(x).dtype, np.floating) else x, tree)
        program = _penalty_program(models, kind, shape)
        args = (cast(critic_params), cast(ctx), cast(obs), cast(action),
                cast(np.ones(len(obs)) if v is None else np.asarray(v)), cast(np.float64(lam0)))

        def value_grad(k, c):
            val, grad = program(jnp.asarray(k, dtype), jnp.asarray(c, dtype), *args)
            return float(val), np.asarray(grad, np.float64)

        out, x = [], np.asarray(K_start, np.float64).ravel()
        for c in c_ladder:
            tol = rel_tol * np.linalg.norm(value_grad(np.asarray(K_ref, np.float64).ravel(), c)[1])
            res = optimize.minimize(lambda k: value_grad(k, c), x, jac=True, method="BFGS",
                                    options=dict(gtol=tol, norm=2, maxiter=max_iter))
            loss_, grad = value_grad(res.x, c)
            gn = float(np.linalg.norm(grad))
            out.append(dict(c=c, K=res.x.reshape(shape), success=bool(gn <= tol), grad_norm=gn, tol=float(tol),
                            iterations=int(res.nit), loss=loss_, message=str(res.message)))
            if np.all(np.isfinite(res.x)):
                x = res.x
    return out


def clipped_noise_variance(policy_noise=0.2, noise_clip=0.5):
    """v_clip = E[clip(X, -c, c)^2], X ~ N(0, sigma^2): TD3's target-noise second moment per coordinate."""
    t = noise_clip / policy_noise
    inside = policy_noise ** 2 * ((2 * stats.norm.cdf(t) - 1) - 2 * t * stats.norm.pdf(t))
    return float(inside + noise_clip ** 2 * 2 * stats.norm.sf(t))


def quad_features(z):
    """15 quadratic monomials z_j z_k (j <= k) of z = (s, a) and a constant: shape (n, 16)."""
    j, k = np.triu_indices(z.shape[1])
    return np.concatenate([z[:, j] * z[:, k], np.ones((len(z), 1))], axis=1)


def expected_next_features(s_next, K, v_clip):
    """E[phi(s', pi(s') + n)] over TD3's clipped target noise n (E n = 0, E n n^T = v_clip I), s' as sampled."""
    m = np.concatenate([s_next, -s_next @ np.asarray(K).T], axis=1)
    ds, d = s_next.shape[1], m.shape[1]
    cov = np.zeros((d, d))
    cov[ds:, ds:] = v_clip * np.eye(d - ds)
    j, k = np.triu_indices(d)
    return np.concatenate([m[:, j] * m[:, k] + cov[j, k], np.ones((len(m), 1))], axis=1)


def features_to_form(theta, d):
    """theta (16,) -> (M, m0) with phi(z)^T theta = z^T M z + m0."""
    j, k = np.triu_indices(d)
    M = np.zeros((d, d))
    M[j, k] = theta[:-1]
    return (M + M.T) / 2, float(theta[-1])  # M_jj = theta_jj; M_jk = M_kj = theta_jk / 2 for j < k


def lstd_penalty(s, a, s_next, p, K, gamma, v_clip, context=None):
    """Per-context LSTD fixed point of the reward penalty p under pi (the C2 critic of block C).

    theta_c = (Phi_c^T (Phi_c - gamma Phi-bar'_c))^-1 Phi_c^T p_c with the quadratic features of z = (s, a); K is (da,
    ds) without contexts or (C, da, ds) with them. Returns a list of (M, m0, theta), one per context: Q_p(s, a) = z^T M
    z + m0, quadratic in z, so P3's C2 fixed point stays closed-form.
    """
    K = np.asarray(K, np.float64)
    blocks = [(np.arange(len(s)), K)] if K.ndim == 2 else [(np.flatnonzero(context == c), K[c]) for c in range(len(K))]
    out = []
    for rows, Kc in blocks:
        phi = quad_features(np.concatenate([s[rows], a[rows]], axis=1))
        nxt = expected_next_features(s_next[rows], Kc, v_clip)
        theta = np.linalg.solve(phi.T @ (phi - gamma * nxt), phi.T @ p[rows])
        out.append(features_to_form(theta, s.shape[1] + a.shape[1]) + (theta,))
    return out


# ---------------------------------------------------------------------------------------------------------
# 6. Strength


def second_moment(states, context=None, n_contexts=None):
    """Sigma_ev = mean s s^T; with contexts, (C, ds, ds) blocks (1/n) sum_{i in c} s_i s_i^T (they sum to Sigma_ev)."""
    s = np.asarray(states, np.float64)
    if context is None:
        return s.T @ s / len(s)
    return np.stack([s[context == c].T @ s[context == c] / len(s) for c in range(n_contexts)])


def strength(K_arm, K_none, Sigma_ev):
    """S = sqrt(tr(delta Sigma_ev delta^T)), delta = K_arm - K_none: the RMS action-space movement (batched over
    leading axes; per-context gains (..., C, da, ds) with Sigma_ev (C, ds, ds) sum the contexts' blocks)."""
    d = np.asarray(K_arm, np.float64) - np.asarray(K_none, np.float64)
    S = np.asarray(Sigma_ev, np.float64)
    if S.ndim == 2:
        return np.sqrt(np.einsum("...ij,jk,...ik->...", d, S, d))
    return np.sqrt(np.einsum("...cij,cjk,...cik->...", d, S, d))


def match_knob(ladder, S_of_knob, target, tol=1e-3, max_bisect=40, interpolate=False, S_ladder=None):
    """Knob whose strength matches target: first crossing on the ladder, then bisection in log(knob).

    S is evaluated on the ascending ladder (or taken from S_ladder); the first adjacent pair whose S values straddle
    target is the bracket. interpolate=True first tries the log-linear interpolation in the bracket (A500: one rerun,
    then bisection). A target outside every bracket is unreachable and never extrapolated (a target below S at the
    first knob is too). monotone flags a ladder whose finite S values are not monotone; nonfinite S values (unbounded
    fixed points) never form a bracket. Returns knob, S, reachable, converged (|S / target - 1| <= tol), monotone,
    evaluations and the ladder's S values.
    """
    knobs = np.asarray(ladder, np.float64)
    S = np.array([S_of_knob(k) for k in knobs] if S_ladder is None else S_ladder, np.float64)
    fin = S[np.isfinite(S)]
    monotone = bool(np.all(np.diff(fin) >= 0) or np.all(np.diff(fin) <= 0))
    out = dict(knob=None, S=None, reachable=False, converged=False, monotone=monotone, evaluations=0,
               ladder_S=S.tolist(), bracket=None)
    within = lambda s: abs(s / target - 1) <= tol
    for j in range(len(knobs)):
        if np.isfinite(S[j]) and within(S[j]):
            return dict(out, knob=float(knobs[j]), S=float(S[j]), reachable=True, converged=True, bracket=(j, j))
        if j + 1 < len(knobs) and np.all(np.isfinite(S[j:j + 2])) and (S[j] - target) * (S[j + 1] - target) < 0:
            break
    else:
        return out
    lo, hi, s_lo, s_hi = knobs[j], knobs[j + 1], S[j], S[j + 1]
    best, evaluations = None, 0
    while evaluations < max_bisect + (1 if interpolate else 0):
        if interpolate and evaluations == 0:
            frac = (target - s_lo) / (s_hi - s_lo)
            k = float(np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))))
        else:
            k = float(np.sqrt(lo * hi))
        s = float(S_of_knob(k))
        evaluations += 1
        if np.isfinite(s) and (best is None or abs(s / target - 1) < abs(best[1] / target - 1)):
            best = (k, s)
        if np.isfinite(s) and within(s):
            break
        if not np.isfinite(s) or (s - target) * (s_lo - target) > 0:
            lo, s_lo = k, s if np.isfinite(s) else s_lo
        else:
            hi, s_hi = k, s
    return dict(out, knob=best[0] if best else None, S=best[1] if best else None, reachable=best is not None,
                converged=bool(best is not None and within(best[1])), evaluations=evaluations, bracket=(j, j + 1))


def frontier(K_arm, K_none, K_P0, Sigma_ev, ms=FRONTIER_M):
    """The realized frontier: m* = argmin_m |K(arm) - K(P0(m))|_Sigma over the P0 gains K_P0 (one per m in ms), and
    the outcome-free targeting share |K(arm) - K(P0(m*))|_Sigma / |K(arm) - K(NONE)|_Sigma."""
    d = strength(K_P0, K_arm, Sigma_ev)
    if not np.any(np.isfinite(d)):
        return dict(m_star=None, index=None, distance=None, share=None)
    i = int(np.nanargmin(d))
    disp = float(strength(K_arm, K_none, Sigma_ev))
    return dict(m_star=float(ms[i]), index=i, distance=float(d[i]), share=float(d[i] / disp) if disp > 0 else None)


def regret_alpha(J_arm, J_P0):
    """regret_alpha = max_m G(P0(m)) - G(arm) = max_m J(P0(m)) - J(arm) (G's common J(NONE) cancels); J values are
    computed elsewhere (score_values.py), after the knobs are hashed."""
    finite = np.asarray(J_P0, np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.max() - J_arm) if finite.size and np.isfinite(J_arm) else None


# ---------------------------------------------------------------------------------------------------------
# 7. Level 1


def terms(args, models, critic_params, K, obs, action, bc_w=None, q_w=None, pen_fn=None, pen_c=None, exact_fn=None):
    """K-gradients of the placement loss's terms on all the given rows (full batch), lambda locked from Q1.

    Returns float64 arrays g_Q (of -lambda mean q_w Q-read), g_pen (of +lambda c mean pen), g_BC and their total g, and
    lambda. The loss is placement_step's, evaluated once on every training row at K.
    """
    batch = BASE.C.Transition(jnp.asarray(obs, jnp.float32), jnp.asarray(action, jnp.float32), None, None, None)
    cp = to_jax32(critic_params)
    params = {"K": jnp.asarray(K, jnp.float32)}
    f32 = lambda x: None if x is None else jnp.asarray(x, jnp.float32)
    kwargs = dict(batch=batch, bc_w=f32(bc_w), q_w=f32(q_w), pen_fn=pen_fn, pen_c=pen_c, exact_fn=exact_fn)
    jac = jax.jacrev(lambda p: _term_parts(args, models, cp, p, **kwargs))(params)["K"]
    (_, (_, lmbda, _)), g = jax.value_and_grad(lambda p: _actor_terms(args, models, cp, p, **kwargs), has_aux=True)(
        params)
    as64 = lambda x: np.asarray(x, np.float64)
    return dict(g_Q=as64(jac[0]), g_pen=as64(jac[1]), g_BC=as64(jac[2]), g=as64(g["K"]), lam=float(lmbda))


def level1(t, t_none, t_exact=None, grad_J=None, v_bar=None, pull=None, eps=ADAM_EPS):
    """Level-1 strength and direction metrics of an arm from terms() of the arm, of NONE and (optional) of EXACT.

    rho = |g_BC + g_pen| / |g_Q|; g = a g_Q,NONE + b g_BC,NONE + r by least squares, m_eff = b / a, tau = |r| / |g|
    (identified only when |cos(g_Q,NONE, g_BC,NONE)| <= 0.98); with grad_J (dJ/dK at K_0, computed elsewhere):
    cos(-Delta g, grad J), the first-order efficiency -<grad J, Delta g> / |Delta g| and, with v_bar (the NONE run's
    mean bias-corrected Adam second moment), cos(-Delta g~, grad J) for Delta g~ = Delta g / (sqrt(v_bar) + eps); with
    t_exact: cos(Delta g, g_EXACT - g_NONE); with pull = (K_bc - K_0) Sigma: cos(-Delta g, pull).
    """
    f = lambda x: np.ravel(np.asarray(x, np.float64))
    g, dg = f(t["g"]), f(t["g"]) - f(t_none["g"])
    basis = np.stack([f(t_none["g_Q"]), f(t_none["g_BC"])], axis=1)
    coef = np.linalg.lstsq(basis, g, rcond=None)[0]
    r = g - basis @ coef
    cos_basis = LQ.cosine(basis[:, 0], basis[:, 1])
    out = dict(norm_g_Q=float(np.linalg.norm(f(t["g_Q"]))), norm_g_BC=float(np.linalg.norm(f(t["g_BC"]))),
               norm_g_pen=float(np.linalg.norm(f(t["g_pen"]))), norm_g=float(np.linalg.norm(g)),
               norm_dg=float(np.linalg.norm(dg)),
               rho=float(np.linalg.norm(f(t["g_BC"]) + f(t["g_pen"])) / np.linalg.norm(f(t["g_Q"]))),
               proj_a=float(coef[0]), proj_b=float(coef[1]), m_eff=float(coef[1] / coef[0]),
               tau=float(np.linalg.norm(r) / np.linalg.norm(g)), cos_gq_gbc=cos_basis,
               identified=bool(cos_basis is not None and abs(cos_basis) <= UNIDENTIFIED_COS))
    if grad_J is not None:
        gj = f(grad_J)
        norm = np.linalg.norm(dg)
        out.update(cos_dg_gradJ=LQ.cosine(-dg, gj), efficiency=float(-(gj @ dg) / norm) if norm > 0 else None)
        if v_bar is not None:
            out["cos_dg_adam_gradJ"] = LQ.cosine(-dg / (np.sqrt(f(v_bar)) + eps), gj)
    if t_exact is not None:
        out["cos_correction"] = LQ.cosine(dg, f(t_exact["g"]) - f(t_none["g"]))
    if pull is not None:
        out["cos_pull"] = LQ.cosine(-dg, f(pull))
    return out
