"""Step 4, stage 4B-0 on D4RL: Level-1 placement diagnostics at the frozen TD3+BC pools, without training.

Implements STEP4_DESIGN.md section 11 ("4B-0: no training") for runs/wbcp_signal/step4/expectations_DRAFT.md Part 4.
On a healthy TD3+BC pool (runs/wbcp_frozen/<pool>-s202609171-u100000, its 100k-update checkpoint) with step 2's frozen
q1 scale and threshold (runs/wbcp_signal/frozen/<pool>: frozen_signals.py fit and evaluate), it measures at the frozen
actor, on up to 100k of the pool's training rows, how each placement of the q1 width changes the first gradient of
TD3+BC's actor loss, and compares that change with uniform BC (an alpha retune) and with same-multiset reassignments
of the weights. Nothing is trained, no environment is stepped and no value or return of any policy is computed; the
only returns read are the logged D4RL episode returns. It describes: no forecast, no verdict.

Inputs. frozen_signals.restore_pool rebuilds the pool byte for byte (prepared data, checkpoint, models); the q1 scale
state comes from <signal dir>/q1/scale.msgpack (its sha256 must equal fit.json's), and the run refuses unless the fit's
parent checkpoint sha256 equals the restored one and, on the first 4,096 population rows, the restored sigma, both
critic heads and the policy action reproduce step 2's heads.npz and q1/frozen.npz within 1e-5 relative (float32
forward passes in another chunk size). Signal: W(s, a) = lambda max(eta(s, a), 1e-6) u, eta the q1 scale network, u
its residual unit, lambda the threshold. Default lambda = the population lambda* of step 2's evaluation
(evaluation_*.json, population.lambda_star.q1): the shortest threshold whose population miscoverage is <= alpha, at
which step 2 reported the dose (q1 dose row SD 0.017-0.050 over these pools, the basis of expectation Part 4 item 1).
The bank-mean WBCP thresholds (about 1.07 lambda*) are recorded, and --threshold iid|reservation selects one. Rank
weights do not depend on lambda; it sets W's level only, which enters P3's natural c = 1 and the HOOK dose.

Rows. The pool's training split (prepared.training, the rows the critic and actor were trained on), a seeded subsample
of min(--rows, n) rows (np.random.default_rng([seed, 0]).choice without replacement, sorted). Every Level-1 quantity
is exact on these rows (one full batch).

Loss and gradients. At the checkpoint's online actor pi_theta and online critic head 0 (Q1, the head TD3+BC's actor
climbs), an arm's loss is the build spec's actor loss (STEP4_DESIGN section 12.4) on the full batch:
    L = -lambda_Q mean_i[u_i Q1(s_i, pi(s_i)) - c W(s_i, pi(s_i))] + mean_i v_i mean_a (pi(s_i) - a_i)^2,
lambda_Q = alpha / mean_i |Q1(s_i, pi(s_i))| (alpha = args.alpha = 2.5) computed once on these rows from the
unweighted, unpenalized Q1 (the host lambda lock of section 3; float64 from float32 values). NONE: u = v = 1, c = 0.
Term gradients are with respect to every actor parameter, through one exact linear operator, the actor's vjp with
per-row action cotangents: G(C) = sum_i (d pi_theta(s_i) / d theta)^T C_i, so that
    g_Q[u] = G(-(lambda_Q / N) u_i grad_a Q1(s_i, pi_i)),  g_BC[v] = G((v_i / N)(2 / d_a)(pi_i - a_i)),
    g_pen[c] = G((lambda_Q c / N) grad_a W(s_i, pi_i)),
the action gradients coming from autodiff of the critic and of the scale network at (s_i, pi_i) (rows are
independent, so the gradient of the row sum is the per-row gradient). This is the chain rule; test_d4rl_level1 checks
G against jax.grad of the loss and against the actor gradient of the repository's td3_bc_update. The vjps run in
float32 over padded row chunks; chunk sums are added in float64.

Weights (section 2). Rank-normal multiset for n rows: z_(j) = Phi^-1((j - 1/2) / n) clipped to [-2.5, 2.5], M_(j) =
exp(s z_(j)) / mean exp(s z), s = 0.83 (primary; 0.47's multiset is reported). Row i receives M at the global rank of
its assignment variable x_i, ties broken by one seeded permutation (default_rng([seed, 1])), never by row order.
Family L: x_i = W(s_i, a_i); family pi: x_i = W(s_i, pi(s_i)) at the frozen actor. Assignments, all on the family's
one multiset (bitwise equal sorted vectors, a built-in check): SIG; SHUF_k, k < 8, SIG's weights permuted globally
(default_rng([seed, 10 + k]), the same permutations for both families); STRAT_k, permuted within deciles of the
placement's exact leverage (P1: l_BC, P2: l_Q, both families; default_rng([seed, 30 + k, kind])); ANTI, SIG's ranks
reversed. T-lin (SIG only): W / mean W per family, whose CV, range and Kish efficiency are reported (lambda cancels).

Placements and knobs (section 3). P0(m): v = m. P1 (BC amplification): v = 1 + beta M. P2 (trust, the mirror of P1):
u = 1 / (1 + beta M), v = 1. L and pi denote the family of M. P3 (critic-side lower bound read by the actor): c W with
the gradient through W's action input, its own multiset-free form. Knob ladders as section 3: beta in {0.01, 0.03,
0.1, 0.3, 0.6, 1, 2, 3, 6, 10, 20, 30}, c in {1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}. They were set
for LQ; --beta-ladder / --c-ladder replace them (recorded, with design_ladders false). On D4RL lambda_Q is about
alpha / |Q|, so c W's gradient can need c beyond 100 to reach rho* (a 2,000-row hopper smoke: P3 unreachable at m = 4).

Strength and matching. Without training there is no displacement S; the Level-1 strength is rho = ||g_BC + g_pen|| /
||g_Q|| (section 5; section 11 names it for P3, "c at matched rho, since there is no FP on an MLP", and it is used here
for every placement so that all are read at one strength). rho_host = rho(NONE), and rho(P0(m)) = m rho_host exactly,
so every arm is matched at rho* = m rho_host for m in --levels (1.3, 2, 4: the uniform-BC multipliers of S1, S2, S3;
2 primary): the first crossing of rho* on the knob ladder (knob 0 being NONE), then Illinois regula falsi in log knob
to |rho / rho* - 1| <= 1e-3 (at most 40 steps). A non-monotone ladder is flagged; with no crossing the arm is
unreachable at that level and never extrapolated. P1 and P3 are linear in their knob, so rho is a closed form in three
inner products; P2's u is not, and each of its evaluations is one batched G call. P3 is also reported at its natural
c = 1 (the calibrated lower bound), and the HOOK at its natural dose.

Per arm. knob and rho; Delta g = g - g_NONE (g the arm's total gradient) and its norm; cos(Delta g, Delta g_P0) with
Delta g_P0 = (m - 1) g_BC,NONE, i.e. the cosine with g_BC,NONE at every m; the least-squares projection g = a g_Q,NONE
+ b g_BC,NONE + r, m_eff = b / a, tau = ||r|| / ||g||, flagged unidentified when |cos(g_Q,NONE, g_BC,NONE)| > 0.98.
P2 rescales the Q term: a constant trust u = 1 / (1 + beta) has Delta g = -beta / (1 + beta) g_Q,NONE, not parallel to
Delta g_P0, although it is P0(1 + beta) up to the loss scale (expectations B3). For P2 the scale-free m_eff and tau, and
cos(Delta g, -g_Q,NONE) (the constant-trust increment, reported as cos_dg_const_trust), are the like-for-like reads.
SHUF cloud (expectations Part 4 item 3): per placement and level, the 5th-95th percentile band (np.percentile, linear)
of cos(Delta g, Delta g_P0) over the reachable SHUF arms and whether SIG lies outside it; likewise for STRAT and for
tau, with ANTI's value and the cosine between SIG's increment and the mean SHUF increment.

P3's nulls. STEP4_DESIGN section 1: no same-multiset null exists for P3 in LQ, where a row permutation of the push has
zero expected gradient on zero-mean states. Here P3's SHUF_k / STRAT_k are push permutations, row i receiving
lambda_Q c grad_a W(s_p(i), pi_p(i)) (SHUF: the global permutations above; STRAT: within deciles of l_pen), reported
descriptively; an MLP actor's Jacobian does not average out under them. ANTI is not defined for P3.

HOOK. BCA's dose at the logged actions (algorithms.td3_bc_bca.bc_readout, blend 0.5: 1 + 0.5 lambda eta / (1 + lambda
eta)) as the BC multiplier: rho, cos(Delta g_HOOK, Delta g_P0), m_eff, tau and the dose's distribution.

Leverage. Exact per-row gradient norms by vmap(grad) of each row's loss terms: l_BC = ||grad_theta mean_a (pi(s_i) -
a_i)^2||, l_Q = ||grad_theta lambda_Q Q1(s_i, pi(s_i))||, l_pen = ||grad_theta lambda_Q W(s_i, pi(s_i))||; Spearman
of each SIG weight (and of the dose) with each.

Data quality. Each training row's D4RL episode: raw rows split after every terminal or timeout row, as
runtime.td3_bc._convert counts episodes (checked against the prepared ids); its raw return includes the timeout row's
reward, which the converted rows drop. Normalized return 100 (R - ref_min) / (ref_max - ref_min) with the prepared
D4RL reference. Pearson and Spearman with SIG_L, SIG_pi, W_L, W_pi and the HOOK dose. On a medium-expert dataset (the
raw file is the medium half followed by the expert half; the half with the higher mean episode normalized return is
labelled expert and both means are recorded): the expert half's share of P1L's extra weight beta M, sum_expert M /
sum M (beta cancels), against its row share; the same for P1pi and for the HOOK's extra dose (dose - 1).

Built-in checks (the run refuses when one fails): equal sorted weight vectors within each family (expectation 2.3);
P0(m): m_eff = m and tau = 0 within 1e-9; a constant trust u = 1/2 through G: m_eff = 2 and tau <= 1e-4 (float32 G);
P2's evaluator at knob 0 returns g_Q,NONE within 1e-6 relative (expectation 2.1's identity; equal up to a float32
vjp's batch position).

Outputs (a new directory): level1.json (strict JSON), level1.md, rows.npz (per-row weights, leverage, dose, episode
returns and row ids, for audit). Tests: python -m unittest experiments.signal.test_d4rl_level1

JAX_PLATFORMS=cpu python experiments/signal/d4rl_level1.py --signal-dir runs/wbcp_signal/frozen/<pool> \
    --output runs/wbcp_signal/step4/d4rl_level1/<pool> [--rows 100000] [--levels 1.3 2 4]
<pool> in hopper-medium-v2, walker2d, halfcheetah, maze2d, pen-expert; one process per pool. Cost: about 700 cotangent
passes (two thirds of them P2's matching), 0.38 s each at 100k rows for a halfcheetah-sized actor on the shared 24-core
CPU (measured on random rows), plus about 20 s of restore and 20 s of leverage: about 5-6 minutes per pool.
"""

import argparse
import collections
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

import algorithms.td3_bc_bca as P  # noqa: E402
from calibration.reference import FrozenReference, positive_scale  # noqa: E402

SCHEMA = "wbcp-step4-d4rl-level1-v1"
POOLS = ("hopper-medium-v2", "walker2d", "halfcheetah", "maze2d", "pen-expert")  # runs/wbcp_signal/frozen/<name>
ROWS = 100_000
SEED = 20261004
S_PRIMARY, S_ROBUST, CLIP = 0.83, 0.47, 2.5
LEVELS = (1.3, 2.0, 4.0)
PRIMARY_LEVEL = 2.0
BETA_LADDER = (0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0, 20.0, 30.0)
C_LADDER = (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
N_SHUF = N_STRAT = 8
DECILES = 10
TOL, MAX_STEPS = 1e-3, 40
UNIDENTIFIED = 0.98
BLEND = 0.5
GATE_ROWS, GATE_TOL = 4096, 1e-5
ROW_CHUNK, K_BLOCK, LEVERAGE_CHUNK = 4096, 16, 256
FAMILIES = ("L", "pi")
# placement -> (weight family, leverage kind of its STRAT deciles); P3 has no weights (push permutations, l_pen)
WEIGHTED = {"P1L": ("L", "bc"), "P2L": ("L", "q"), "P1pi": ("pi", "bc"), "P2pi": ("pi", "q")}
PLACEMENTS = ("P1L", "P2L", "P1pi", "P2pi", "P3")
LEVERAGE_KINDS = ("bc", "q", "pen")
THRESHOLDS = ("lambda_star", "iid", "reservation")

Batch = collections.namedtuple("Batch", "obs action")


# ---------------------------------------------------------------------------------------------------------
# Weights: the rank-normal multiset and its assignments (NumPy, float64)


def multiset(n, s=S_PRIMARY, clip=CLIP):
    """Section 2's multiset, ascending: M_(j) = exp(s z_(j)) / mean exp(s z), z_(j) = clip(Phi^-1((j - 1/2) / n))."""
    if n < 1:
        raise ValueError("the multiset needs at least one row")
    z = np.clip(stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n), -clip, clip)
    e = np.exp(s * z)
    return e / e.mean()


def multiset_stats(m):
    """CV, range, Kish n_eff / n and p90 / p10 of a weight vector (the B6 table's columns)."""
    m = np.asarray(m, np.float64)
    p90, p10 = np.percentile(m, [90, 10])
    return dict(n=int(m.size), mean=float(m.mean()), cv=float(m.std() / m.mean()), min=float(m.min()),
                max=float(m.max()), kish=float(m.sum() ** 2 / (m.size * np.sum(m * m))), p90_p10=float(p90 / p10))


def ranks(x, tie_perm):
    """Global rank of every row (0 = smallest x); ties go by tie_perm, a seeded permutation, never by row order."""
    x, tie_perm = np.asarray(x), np.asarray(tie_perm)
    if x.ndim != 1 or tie_perm.shape != x.shape or not np.array_equal(np.sort(tie_perm), np.arange(x.size)):
        raise ValueError("x must be a vector and tie_perm a permutation of its rows")
    if not np.all(np.isfinite(x)):
        raise ValueError("assignment variables must be finite")
    order = np.lexsort((tie_perm, x))
    out = np.empty(x.size, np.int64)
    out[order] = np.arange(x.size)
    return out


def rank_normal(x, s, clip, tie_perm):
    """Rank-normal weights (section 2): row i receives the multiset value at the global rank of x_i."""
    return multiset(len(x), s, clip)[ranks(x, tie_perm)]


def decile_index(leverage, tie_perm, bins=DECILES):
    """Bin of every row by rank of `leverage` (0 lowest); sizes differ by at most one; ties by tie_perm."""
    return ranks(leverage, tie_perm) * bins // len(leverage)


def within_groups_permutation(groups, rng):
    """A permutation p with groups[p] == groups: rows exchange places only inside their group (groups ascending)."""
    groups = np.asarray(groups)
    p = np.arange(groups.size)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        p[idx] = idx[rng.permutation(idx.size)]
    return p


def shuffle_permutations(n, seed, count=N_SHUF):
    """SHUF_k's global permutations, default_rng([seed, 10 + k]); the same for every family and for P3's pushes."""
    return [np.random.default_rng([seed, 10 + k]).permutation(n) for k in range(count)]


def strat_permutations(groups, seed, kind, count=N_STRAT):
    """STRAT_k's within-decile permutations for one leverage kind, default_rng([seed, 30 + k, kind index])."""
    code = LEVERAGE_KINDS.index(kind)
    return [within_groups_permutation(groups, np.random.default_rng([seed, 30 + k, code])) for k in range(count)]


def assignments(x, tie_perm, shuf, strat, s=S_PRIMARY, clip=CLIP):
    """SIG, SHUF_k, STRAT_k and ANTI weights of one placement and family, all on one multiset (row i of a permuted arm
    takes SIG's weight of row p(i)). Returns (dict name -> weights, the multiset ascending)."""
    m = multiset(len(x), s, clip)
    r = ranks(x, tie_perm)
    sig = m[r]
    return dict(SIG=sig, SHUF=[sig[p] for p in shuf], STRAT=[sig[p] for p in strat], ANTI=m[len(x) - 1 - r]), m


def arm_list(assigned):
    """(name, weights) in the order SIG, SHUF_0.., STRAT_0.., ANTI."""
    return ([("SIG", assigned["SIG"])] + [(f"SHUF_{k}", w) for k, w in enumerate(assigned["SHUF"])]
            + [(f"STRAT_{k}", w) for k, w in enumerate(assigned["STRAT"])] + [("ANTI", assigned["ANTI"])])


def same_multiset(arms, m):
    """Expectation 2.3: every arm's sorted weights equal the multiset bitwise."""
    return all(np.array_equal(np.sort(w), m) for _, w in arms)


# ---------------------------------------------------------------------------------------------------------
# Exact Level-1 gradients at one frozen actor (JAX float32 forward and backward passes, float64 sums)


def _rows_kernel(models, blend, params, critic_params, cal_params, unit, threshold, obs, action):
    """Per-row quantities at the frozen actor: pi, Q1 and grad_a Q1 at pi, W and grad_a W at pi, W at the logged action,
    both critic heads at the logged action, eta at both points and BCA's dose (td3_bc_bca.bc_readout)."""
    actor, critic, cal = models
    pi = actor.apply(params, obs)

    def q1(a):
        return critic.apply(critic_params, obs, a)[..., 0]

    def width(a):
        return threshold * positive_scale(cal.apply(cal_params, obs, a), unit)

    reference = FrozenReference(cal_params, unit, threshold, threshold, threshold, jnp.float32(0.0), jnp.asarray(True))
    readout = P.bc_readout(models, P.State(None, None, None, reference), Batch(obs, action), blend)
    return dict(pi=pi, q1_pi=q1(pi), qa=jax.grad(lambda a: q1(a).sum())(pi), w_pi=width(pi),
                wa=jax.grad(lambda a: width(a).sum())(pi), w_logged=width(action),
                eta_logged=cal.apply(cal_params, obs, action), eta_pi=cal.apply(cal_params, obs, pi),
                q_logged=critic.apply(critic_params, obs, action), dose=readout.dose,
                dose_valid=readout.inputs_valid)


def _vjp_kernel(actor, params, obs, cot):
    """(K, P) flat actor gradients sum_i (d pi(s_i) / d theta)^T cot[k, i] for a (K, rows, d_a) cotangent block."""
    _, pullback = jax.vjp(lambda p: actor.apply(p, obs), params)
    return jax.vmap(lambda c: ravel_pytree(pullback(c)[0])[0])(cot)


def _leverage_kernel(models, params, critic_params, cal_params, unit, threshold, lam, obs, action):
    """Per-example gradient norms by vmap(grad) of each row's loss terms: BC, lambda Q1 at pi, lambda W at pi."""
    actor, critic, cal = models

    def bc(p, s, a):
        return jnp.square(actor.apply(p, s[None])[0] - a).mean()

    def q(p, s):
        return lam * critic.apply(critic_params, s[None], actor.apply(p, s[None]))[0, 0]

    def pen(p, s):
        return lam * threshold * positive_scale(cal.apply(cal_params, s[None], actor.apply(p, s[None])), unit)[0]

    def norms(tree):
        return jnp.sqrt(sum(jnp.sum(jnp.square(x).reshape(x.shape[0], -1), axis=1)
                            for x in jax.tree_util.tree_leaves(tree)))

    return (norms(jax.vmap(jax.grad(bc), in_axes=(None, 0, 0))(params, obs, action)),
            norms(jax.vmap(jax.grad(q), in_axes=(None, 0))(params, obs)),
            norms(jax.vmap(jax.grad(pen), in_axes=(None, 0))(params, obs)))


class Level1:
    """Exact Level-1 quantities of TD3+BC's actor loss terms at one frozen actor, on fixed rows.

    models: (actor, critic, calibrator) with .apply as algorithms.td3_bc_bca.initialize builds them; actor_params,
    critic_params, cal_params: the frozen actor, the online critic (head 0 = Q1) and the q1 scale network; unit and
    threshold give W(s, a) = threshold max(eta, 1e-6) unit. Rows are padded to whole chunks; padded rows carry zero
    cotangents and are dropped from every per-row output, so they change no sum.
    """

    def __init__(self, models, actor_params, critic_params, cal_params, unit, threshold, obs, action, alpha,
                 blend=BLEND, row_chunk=ROW_CHUNK, k_block=K_BLOCK, leverage_chunk=LEVERAGE_CHUNK):
        obs, action = np.asarray(obs, np.float32), np.asarray(action, np.float32)
        if obs.ndim != 2 or action.ndim != 2 or len(obs) != len(action) or not len(obs):
            raise ValueError("obs and action must be aligned nonempty matrices")
        self.models = tuple(models)
        self.params, self.critic_params, self.cal_params = actor_params, critic_params, cal_params
        self.unit, self.threshold = jnp.float32(unit), jnp.float32(threshold)
        self.n, self.ds = obs.shape
        self.da = action.shape[1]
        self.alpha, self.blend = float(alpha), float(blend)
        self.row_chunk = int(min(row_chunk, self.n))
        self.leverage_chunk = int(min(leverage_chunk, self.n))
        self.k_block = int(k_block)
        unit_rows = math.lcm(self.row_chunk, self.leverage_chunk)
        padded = -(-self.n // unit_rows) * unit_rows
        self.obs = jnp.asarray(np.concatenate([obs, np.zeros((padded - self.n, self.ds), np.float32)]))
        self.action = jnp.asarray(np.concatenate([action, np.zeros((padded - self.n, self.da), np.float32)]))
        self.action_np = action.astype(np.float64)
        self.size = int(ravel_pytree(actor_params)[0].size)
        self._rows_fn = jax.jit(partial(_rows_kernel, self.models, self.blend))
        self._vjp_fn = jax.jit(partial(_vjp_kernel, self.models[0]))
        self._leverage_fn = jax.jit(partial(_leverage_kernel, self.models))
        self.stats = dict(vjp_calls=0, cotangents=0, vjp_seconds=0.0)
        self.rows = self._per_row()
        if not self.rows.pop("dose_valid"):
            raise FloatingPointError("invalid BCA dose inputs")
        self.lam = self.alpha / float(np.mean(np.abs(self.rows["q1_pi"])))
        n, da = self.n, self.da
        # per-row action cotangents of the three loss terms at unit weight (the 1/N of the batch mean included)
        self.cq = -(self.lam / n) * self.rows["qa"]
        self.cb = (2.0 / (da * n)) * (self.rows["pi"] - self.action_np)
        self.cp = (self.lam / n) * self.rows["wa"]

    def _per_row(self):
        parts = collections.defaultdict(list)
        valid = True
        for r0 in range(0, self.n, self.row_chunk):
            out = self._rows_fn(self.params, self.critic_params, self.cal_params, self.unit, self.threshold,
                                self.obs[r0:r0 + self.row_chunk], self.action[r0:r0 + self.row_chunk])
            take = min(self.row_chunk, self.n - r0)
            for name, value in out.items():
                if name == "dose_valid":
                    valid &= bool(value)  # padded zero rows are valid inputs too
                else:
                    parts[name].append(np.asarray(value, np.float64)[:take])
        rows = {name: np.concatenate(chunks) for name, chunks in parts.items()}
        if not all(np.all(np.isfinite(v)) for v in rows.values()):
            raise FloatingPointError("nonfinite per-row quantities at the frozen actor")
        rows["dose_valid"] = valid
        return rows

    def G(self, build, count):
        """sum_i (d pi_theta(s_i) / d theta)^T C_k[i] for count cotangents C_k (n x d_a each): build(ks, rows), two
        slices, returns the float64 block C[ks][:, rows]. Returns (count, P) float64 flat gradients, ravel_pytree
        order."""
        out = np.zeros((count, self.size))
        started = time.perf_counter()
        for k0 in range(0, count, self.k_block):
            ks = slice(k0, min(k0 + self.k_block, count))
            size = ks.stop - ks.start
            for r0 in range(0, self.n, self.row_chunk):
                r1 = min(r0 + self.row_chunk, self.n)
                block = np.zeros((self.k_block, self.row_chunk, self.da), np.float32)
                block[:size, :r1 - r0] = build(ks, slice(r0, r1))
                grads = self._vjp_fn(self.params, self.obs[r0:r0 + self.row_chunk], jnp.asarray(block))
                out[ks] += np.asarray(grads, np.float64)[:size]
                self.stats["vjp_calls"] += 1
        self.stats["cotangents"] += count
        self.stats["vjp_seconds"] += time.perf_counter() - started
        return out

    def weighted(self, base, weights):
        """G of row-weighted copies of one base cotangent: weights (K, n) -> (K, P)."""
        weights = np.atleast_2d(np.asarray(weights, np.float64))
        return self.G(lambda ks, r: weights[ks, r][:, :, None] * base[None, r], len(weights))

    def stacked(self, bases):
        """G of a list of (n, d_a) cotangents -> (K, P)."""
        return self.G(lambda ks, r: np.stack([bases[k][r] for k in range(ks.start, ks.stop)]), len(bases))

    def permuted(self, base, perms):
        """G of row-permuted copies of one base cotangent (row i receives row p(i)'s cotangent) -> (K, P)."""
        return self.G(lambda ks, r: np.stack([base[perms[k][r]] for k in range(ks.start, ks.stop)]), len(perms))

    def leverage(self):
        """Exact per-row gradient norms of the BC, Q and penalty terms (vmap(grad)); dict kind -> (n,) float64."""
        parts = {kind: [] for kind in LEVERAGE_KINDS}
        lam = jnp.float32(self.lam)
        for r0 in range(0, self.n, self.leverage_chunk):
            out = self._leverage_fn(self.params, self.critic_params, self.cal_params, self.unit, self.threshold, lam,
                                    self.obs[r0:r0 + self.leverage_chunk], self.action[r0:r0 + self.leverage_chunk])
            take = min(self.leverage_chunk, self.n - r0)
            for kind, value in zip(LEVERAGE_KINDS, out):
                parts[kind].append(np.asarray(value, np.float64)[:take])
        return {kind: np.concatenate(chunks) for kind, chunks in parts.items()}


def explicit_loss(models, params, critic_params, cal_params, unit, threshold, obs, action, alpha, bc_w=None, q_w=None,
                  pen_c=0.0):
    """The build spec's actor loss on one full batch (section 12.4): the reference G's assembly is tested against."""
    actor, critic, cal = models
    pi = actor.apply(params, obs)
    q = critic.apply(critic_params, obs, pi)[..., 0]
    lam = alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
    q_read = q if q_w is None else jax.lax.stop_gradient(q_w) * q
    pen = threshold * positive_scale(cal.apply(cal_params, obs, pi), unit)
    q_term = q_read.mean() - pen_c * pen.mean()
    sq = jnp.square(pi - action).mean(axis=-1)
    bc = sq.mean() if bc_w is None else (jax.lax.stop_gradient(bc_w) * sq).mean()
    return -lam * q_term + bc


# ---------------------------------------------------------------------------------------------------------
# Strength matching, projections and records


def match_knobs(ladder, rho_host, ladder_rho, targets, evaluate, tol=TOL, max_steps=MAX_STEPS):
    """Section 5's matching on rho for every (assignment a, target t): the first crossing of rho*_t on the ladder
    (knob 0 = NONE, rho_host), then Illinois regula falsi in log knob (in the knob itself when the bracket starts at 0)
    to |rho / rho* - 1| <= tol, at most max_steps evaluations. ladder_rho: (A, L) rho at the ladder knobs;
    evaluate(a, knobs) -> (rho, payloads) for index and knob arrays, payloads a sequence (or None), one evaluation call
    per step for every unresolved problem. Returns {(a, t): dict(knob, rho, reachable, monotone, converged, steps,
    payload)}."""
    ladder = np.asarray(ladder, np.float64)
    knobs = np.r_[0.0, ladder]
    out, active = {}, []
    for a in range(ladder_rho.shape[0]):
        seq = np.r_[rho_host, ladder_rho[a]]
        monotone = bool(np.all(np.diff(seq) >= 0))
        for t, target in enumerate(targets):
            if not target > rho_host:
                raise ValueError("matching targets must exceed rho_host")
            above = np.flatnonzero(seq >= target)
            base = dict(monotone=monotone, steps=0)
            if not above.size:
                out[a, t] = dict(base, reachable=False, converged=False, knob=None, rho=None, payload=None)
                continue
            j = int(above[0])
            log = knobs[j - 1] > 0
            tx = (lambda k: math.log(k)) if log else (lambda k: k)
            p = dict(a=a, t=t, target=target, log=log, x_lo=tx(knobs[j - 1]), x_hi=tx(knobs[j]),
                     f_lo=math.log(seq[j - 1] / target), f_hi=math.log(seq[j] / target), side=0, base=base, best=None)
            p["x"] = p["x_hi"] if abs(seq[j] / target - 1) <= tol else _falsi(p)
            active.append(p)
    for step in range(max_steps):
        if not active:
            break
        k = np.array([math.exp(p["x"]) if p["log"] else p["x"] for p in active])
        rho, payloads = evaluate(np.array([p["a"] for p in active]), k)
        still = []
        for i, p in enumerate(active):
            f = math.log(rho[i] / p["target"])
            record = dict(knob=float(k[i]), rho=float(rho[i]), payload=None if payloads is None else payloads[i])
            if p["best"] is None or abs(f) < abs(math.log(p["best"]["rho"] / p["target"])):
                p["best"] = record
            if abs(rho[i] / p["target"] - 1) <= tol:
                out[p["a"], p["t"]] = {**p["base"], **record, "reachable": True, "converged": True, "steps": step + 1}
                continue
            if f > 0:  # Illinois: a retained endpoint has its residual halved
                p["x_hi"], p["f_hi"] = p["x"], f
                if p["side"] == 1:
                    p["f_lo"] /= 2
                p["side"] = 1
            else:
                p["x_lo"], p["f_lo"] = p["x"], f
                if p["side"] == -1:
                    p["f_hi"] /= 2
                p["side"] = -1
            p["x"] = _falsi(p)
            still.append(p)
        active = still
    for p in active:
        out[p["a"], p["t"]] = {**p["base"], **p["best"], "reachable": True, "converged": False, "steps": max_steps}
    return out


def _falsi(p):
    """The regula falsi point of a bracket, or its midpoint when that point is not strictly inside."""
    x = p["x_hi"] - p["f_hi"] * (p["x_hi"] - p["x_lo"]) / (p["f_hi"] - p["f_lo"])
    lo, hi = p["x_lo"], p["x_hi"]
    return x if lo < x < hi else (lo + hi) / 2


def cosine(x, y):
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    return float(np.dot(x, y) / (nx * ny)) if nx > 0 and ny > 0 else None


def correlation(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.std() == 0 or y.std() == 0:
        return dict(pearson=None, spearman=None)
    return dict(pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(stats.spearmanr(x, y)[0]))


def spearman(x, y):
    return correlation(x, y)["spearman"]


def projection(g, gq, gb):
    """Least squares g = a gq + b gb + r: (a, b, m_eff = b / a, tau = ||r|| / ||g||)."""
    basis = np.stack([gq, gb], axis=1)
    (a, b), *_ = np.linalg.lstsq(basis, g, rcond=None)
    r = g - basis @ np.array([a, b])
    norm = float(np.linalg.norm(g))
    return dict(a=float(a), b=float(b), m_eff=float(b / a) if a != 0 else None,
                tau=float(np.linalg.norm(r) / norm) if norm > 0 else None)


def arm_record(base, match, g_q, g_bc, g_pen, trust=False):
    """One arm's Level-1 record at its matched knob: strength, increment, uniform-BC cosine and projection."""
    g = g_q + g_bc + g_pen
    dg = g - base["g_none"]
    record = dict(knob=match["knob"], rho=match["rho"], reachable=match["reachable"], monotone=match["monotone"],
                  converged=match["converged"], steps=match["steps"],
                  rho_check=float(np.linalg.norm(g_bc + g_pen) / np.linalg.norm(g_q)),
                  dg_norm=float(np.linalg.norm(dg)), cos_dg_p0=cosine(dg, base["gb"]),
                  g_q_norm=float(np.linalg.norm(g_q)), g_bc_norm=float(np.linalg.norm(g_bc)),
                  g_pen_norm=float(np.linalg.norm(g_pen)), **projection(g, base["gq"], base["gb"]))
    if trust:
        record["cos_dg_const_trust"] = cosine(dg, -base["gq"])
    return record, dg


def unreachable(match):
    return dict(knob=None, rho=None, reachable=False, monotone=match["monotone"], converged=False, steps=0)


def cloud(sig, nulls, metric):
    """The 5th-95th percentile band of `metric` over the reachable null arms, and whether SIG lies outside it."""
    values = [r[metric] for r in nulls if r["reachable"] and r.get(metric) is not None]
    entry = dict(metric=metric, n=len(values), values=values, sig=sig.get(metric) if sig["reachable"] else None)
    if entry["sig"] is None or len(values) < 2:
        return dict(entry, p05=None, p95=None, outside=None, side=None)
    lo, hi = (float(v) for v in np.percentile(values, [5, 95]))
    v = entry["sig"]
    side = "below" if v < lo else "above" if v > hi else None
    return dict(entry, p05=lo, p95=hi, min=float(min(values)), max=float(max(values)), outside=side is not None,
                side=side, nulls_below_sig=int(np.sum(np.asarray(values) < v)))


def level_summary(records, increments, trust=False):
    """SHUF / STRAT clouds for one placement and level, ANTI's values and the SIG-to-mean-SHUF cosine."""
    sig = records["SIG"]
    shuf = [records[f"SHUF_{k}"] for k in range(N_SHUF)]
    strat = [records[f"STRAT_{k}"] for k in range(N_STRAT)]
    anti = records.get("ANTI")
    reachable_shuf = [increments[f"SHUF_{k}"] for k in range(N_SHUF) if f"SHUF_{k}" in increments]
    out = dict(shuf_cos=cloud(sig, shuf, "cos_dg_p0"), strat_cos=cloud(sig, strat, "cos_dg_p0"),
               shuf_tau=cloud(sig, shuf, "tau"), strat_tau=cloud(sig, strat, "tau"),
               anti_cos=anti["cos_dg_p0"] if anti and anti["reachable"] else None,
               anti_tau=anti["tau"] if anti and anti["reachable"] else None,
               cos_sig_shuf_mean=(cosine(increments["SIG"], np.mean(reachable_shuf, axis=0))
                                  if "SIG" in increments and reachable_shuf else None),
               reachable=dict(SIG=sig["reachable"], SHUF=sum(r["reachable"] for r in shuf),
                              STRAT=sum(r["reachable"] for r in strat),
                              ANTI=anti["reachable"] if anti else None),
               nonmonotone_ladders=sum(not r["monotone"] for r in records.values() if r["monotone"] is not None))
    if trust:
        out["shuf_cos_const_trust"] = cloud(sig, shuf, "cos_dg_const_trust")
    return out


# ---------------------------------------------------------------------------------------------------------
# The analysis on one engine (pool-independent; tested on synthetic models)


def analyse(engine, *, levels=LEVELS, s=S_PRIMARY, seed=SEED, beta_ladder=BETA_LADDER, c_ladder=C_LADDER, log=print):
    """Every Level-1 quantity of section 11 on an engine's rows. Returns (payload dict, per-row arrays)."""
    started = time.perf_counter()
    levels = tuple(float(m) for m in levels)
    if not levels or any(not m > 1 for m in levels):
        raise ValueError("levels are uniform-BC multipliers above 1")
    beta_ladder, c_ladder = (tuple(float(k) for k in ladder) for ladder in (beta_ladder, c_ladder))
    for ladder in (beta_ladder, c_ladder):
        if not ladder or ladder[0] <= 0 or any(b <= a for a, b in zip(ladder, ladder[1:])):
            raise ValueError("knob ladders must be positive and strictly increasing")
    rows, n = engine.rows, engine.n
    leverage = engine.leverage()
    leverage_seconds = time.perf_counter() - started
    tie = np.random.default_rng([seed, 1]).permutation(n)
    x = dict(L=rows["w_logged"], pi=rows["w_pi"])
    deciles = {kind: decile_index(leverage[kind], tie) for kind in LEVERAGE_KINDS}
    shuf = shuffle_permutations(n, seed)
    strat = {kind: strat_permutations(deciles[kind], seed, kind) for kind in LEVERAGE_KINDS}
    assigned, checks = {}, {}
    for placement, (family, kind) in WEIGHTED.items():
        weights, m_sorted = assignments(x[family], tie, shuf, strat[kind], s, CLIP)
        assigned[placement] = arm_list(weights)
        checks[f"same_multiset_{placement}"] = same_multiset(assigned[placement], m_sorted)

    # Base vectors through G: Q term (also the P2 evaluator's knob-0 path), BC, penalty per unit c, HOOK increment,
    # and a constant trust u = 1/2 (a derived check: P0(2) up to the loss scale).
    gq = engine.weighted(engine.cq, np.ones((1, n)))[0]
    gb, gp, g_hook = engine.stacked([engine.cb, engine.cp, (rows["dose"] - 1.0)[:, None] * engine.cb])
    g_half = engine.weighted(engine.cq, np.full((1, n), 0.5))[0]
    rho_host = float(np.linalg.norm(gb) / np.linalg.norm(gq))
    targets = [m * rho_host for m in levels]
    base = dict(gq=gq, gb=gb, g_none=gq + gb)
    cos_qb = cosine(gq, gb)
    identified = cos_qb is not None and abs(cos_qb) <= UNIDENTIFIED
    p0 = {f"{m:g}": dict(m=m, rho=float(np.linalg.norm(m * gb) / np.linalg.norm(gq)),
                         cos_dg_p0=cosine((m - 1) * gb, gb), **projection(gq + m * gb, gq, gb)) for m in levels}
    half = projection(g_half + gb, gq, gb)
    checks["p0_projection"] = all(abs(r["m_eff"] - r["m"]) <= 1e-9 * r["m"] and r["tau"] <= 1e-9 for r in p0.values())
    checks["p0_rho"] = all(abs(r["rho"] / (r["m"] * rho_host) - 1) <= 1e-9 for r in p0.values())
    checks["constant_trust_is_p0"] = abs(half["m_eff"] - 2.0) <= 1e-4 * 2.0 and half["tau"] <= 1e-4
    log(f"base: rho_host {rho_host:.4g}, cos(g_Q, g_BC) {cos_qb}, lambda_Q {engine.lam:.4g}")

    placements, timing = {}, dict(leverage_seconds=leverage_seconds)
    knob0 = None
    for placement in PLACEMENTS:
        t0 = time.perf_counter()
        trust = placement.startswith("P2")
        if placement == "P3":
            names = (["SIG"] + [f"SHUF_{k}" for k in range(N_SHUF)] + [f"STRAT_{k}" for k in range(N_STRAT)])
            pushes = engine.permuted(engine.cp, shuf + strat["pen"])
            vectors = np.concatenate([gp[None], pushes])
            ladder = c_ladder
        elif placement.startswith("P1"):
            names = [name for name, _ in assigned[placement]]
            vectors = engine.weighted(engine.cb, np.stack([w for _, w in assigned[placement]]))
            ladder = beta_ladder
        if placement.startswith("P1") or placement == "P3":  # rho linear in the knob: closed form
            bb, gq_norm = float(gb @ gb), float(np.linalg.norm(gq))
            bx, xx = vectors @ gb, np.einsum("kp,kp->k", vectors, vectors)

            def closed(a, k, bx=bx, xx=xx, bb=bb, gq_norm=gq_norm):
                return np.sqrt(np.maximum(bb + 2 * k * bx[a] + k * k * xx[a], 0.0)) / gq_norm, None

            a_all = np.repeat(np.arange(len(names)), len(ladder))
            ladder_rho = closed(a_all, np.tile(ladder, len(names)))[0].reshape(len(names), len(ladder))
            matches = match_knobs(ladder, rho_host, ladder_rho, targets, closed)

            def vectors_at(a, match, vectors=vectors, critic_side=placement == "P3"):
                inc = match["knob"] * vectors[a]
                return (gq, gb, inc) if critic_side else (gq, gb + inc, np.zeros_like(gb))
        else:  # P2: u = 1 / (1 + beta M); every evaluation is a batched G call
            names = [name for name, _ in assigned[placement]]
            M = np.stack([w for _, w in assigned[placement]])
            ladder = beta_ladder
            gb_norm = float(np.linalg.norm(gb))

            def trust_eval(a, k, M=M, gb_norm=gb_norm):
                g = engine.weighted(engine.cq, 1.0 / (1.0 + k[:, None] * M[a]))
                return gb_norm / np.linalg.norm(g, axis=1), list(g)

            if knob0 is None:  # expectation 2.1 at Level 1: knob 0 is NONE through P2's own evaluator
                knob0 = trust_eval(np.array([0]), np.array([0.0]))[1][0]
                checks["knob_zero_is_none"] = bool(np.max(np.abs(knob0 - gq)) <= 1e-6 * np.max(np.abs(gq)))
                checks["knob_zero_max_abs_difference"] = float(np.max(np.abs(knob0 - gq)))
            a_all = np.repeat(np.arange(len(names)), len(ladder))
            ladder_rho = trust_eval(a_all, np.tile(ladder, len(names)))[0].reshape(len(names), len(ladder))
            matches = match_knobs(ladder, rho_host, ladder_rho, targets, trust_eval)

            def vectors_at(a, match):
                return match["payload"], gb, np.zeros_like(gb)

        out = dict(ladder=list(ladder), arms=names, levels={})
        if placement == "P3":
            natural, _ = arm_record(base, dict(knob=1.0, rho=float(np.linalg.norm(gb + gp) / np.linalg.norm(gq)),
                                               reachable=True, monotone=None, converged=None, steps=0), gq, gb, gp)
            out["natural_c1"] = natural
            out["anti"] = "not defined for P3 (push permutations have no rank to reverse)"
            out["nulls"] = ("push permutations: row i receives lambda_Q c grad_a W(s_p(i), pi_p(i)); SHUF global, "
                            "STRAT within deciles of l_pen (descriptive; STEP4_DESIGN section 1)")
        for t, m in enumerate(levels):
            records, increments = {}, {}
            for a, name in enumerate(names):
                match = matches[a, t]
                if not match["reachable"]:
                    records[name] = unreachable(match)
                    continue
                record, dg = arm_record(base, match, *vectors_at(a, match), trust=trust)
                records[name], increments[name] = record, dg
            out["levels"][f"{m:g}"] = dict(m=m, rho_target=targets[t], arms=records,
                                           summary=level_summary(records, increments, trust))
        placements[placement] = out
        timing[f"{placement}_seconds"] = time.perf_counter() - t0
        log(f"{placement}: {time.perf_counter() - t0:.1f}s, cotangents so far {engine.stats['cotangents']}")

    hook, _ = arm_record(base, dict(knob=None, rho=float(np.linalg.norm(gb + g_hook) / np.linalg.norm(gq)),
                                    reachable=True, monotone=None, converged=None, steps=0), gq, gb + g_hook,
                         np.zeros_like(gb))
    dose = rows["dose"]
    hook.update(m_equivalent=hook["rho"] / rho_host, dose=dict(mean=float(dose.mean()), sd=float(dose.std()),
                                                                p05=float(np.percentile(dose, 5)),
                                                                p95=float(np.percentile(dose, 95)),
                                                                min=float(dose.min()), max=float(dose.max())))
    sig = {family: next(w for name, w in assigned["P1" + family] if name == "SIG") for family in FAMILIES}
    spearman_table = {f"SIG_{f}": {kind: spearman(sig[f], leverage[kind]) for kind in LEVERAGE_KINDS} for f in FAMILIES}
    spearman_table["dose"] = {kind: spearman(dose, leverage[kind]) for kind in LEVERAGE_KINDS}
    describe = lambda v: dict(mean=float(v.mean()), sd=float(v.std()),  # noqa: E731
                              **{f"q{int(100 * q)}": float(np.quantile(v, q)) for q in (0.05, 0.5, 0.95)})
    payload = dict(
        rows=n, actor_parameters=engine.size, lambda_q=engine.lam, alpha=engine.alpha,
        threshold=float(engine.threshold), residual_unit=float(engine.unit),
        base=dict(rho_host=rho_host, g_q_norm=float(np.linalg.norm(gq)), g_bc_norm=float(np.linalg.norm(gb)),
                  g_pen_per_c_norm=float(np.linalg.norm(gp)), cos_gq_gbc=cos_qb, identified=identified,
                  unidentified_above=UNIDENTIFIED, cos_gpen_gbc=cosine(gp, gb), cos_gpen_gq=cosine(gp, gq),
                  bc_rms=float(np.sqrt(np.mean(np.square(rows["pi"] - engine.action_np))))),
        levels=list(levels), primary_level=PRIMARY_LEVEL if PRIMARY_LEVEL in levels else None, rho_targets=targets,
        p0=p0, constant_trust_half=half,
        multiset={f"s={s:g}": multiset_stats(multiset(n, s)), f"s={S_ROBUST:g}": multiset_stats(multiset(n, S_ROBUST))},
        t_lin={family: multiset_stats(x[family] / x[family].mean()) for family in FAMILIES},
        signal=dict(W_logged=describe(x["L"]), W_pi=describe(x["pi"]), eta_logged=describe(rows["eta_logged"]),
                    eta_pi=describe(rows["eta_pi"]), spearman_W_logged_W_pi=spearman(x["L"], x["pi"])),
        leverage=dict(summary={kind: describe(v) for kind, v in leverage.items()}, spearman=spearman_table,
                      spearman_bc_q=spearman(leverage["bc"], leverage["q"]),
                      deciles="STRAT deciles: P1 l_BC, P2 l_Q, P3 l_pen (both families)"),
        placements=placements, hook=hook, checks=checks,
        timing=dict(timing, total_seconds=time.perf_counter() - started, **engine.stats))
    arrays = dict(w_logged=x["L"], w_pi=x["pi"], sig_L=sig["L"], sig_pi=sig["pi"], dose=dose,
                  **{f"leverage_{kind}": v for kind, v in leverage.items()}, tie_perm=tie)
    return payload, arrays


# ---------------------------------------------------------------------------------------------------------
# Data quality: logged D4RL episode returns


def raw_episodes(terminals, timeouts):
    """Episode index of every raw row as runtime.td3_bc._convert counts episodes: a new one starts after a terminal or a
    timeout row (a row with both flags counts once)."""
    boundary = np.asarray(terminals, bool) | np.asarray(timeouts, bool)
    return np.r_[0, np.cumsum(boundary[:-1])].astype(np.int64)


def episode_normalized_returns(rewards, terminals, timeouts, reference_min, reference_max):
    """(episode index per raw row, raw return per episode, normalized return per episode): every raw reward counts,
    including the timeout row's, which the converted rows drop."""
    episode = raw_episodes(terminals, timeouts)
    raw = np.bincount(episode, weights=np.asarray(rewards, np.float64))
    return episode, raw, 100.0 * (raw - reference_min) / (reference_max - reference_min)


def expert_half(raw_rows, episode, normalized, rows_raw):
    """Medium-expert halves: raw row < raw_rows // 2 or not; the half with the higher mean episode normalized return
    (over its episodes) is labelled expert. Returns (is_expert per given raw row, record)."""
    cut = raw_rows // 2
    first = np.bincount(episode, minlength=len(normalized))  # rows per episode
    start = np.r_[0, np.cumsum(first)[:-1]]
    second_half = start >= cut
    means = [float(normalized[~second_half].mean()), float(normalized[second_half].mean())]
    expert_second = means[1] > means[0]
    record = dict(cut_raw_row=int(cut), mean_normalized_return_first_half=means[0],
                  mean_normalized_return_second_half=means[1], expert_half="second" if expert_second else "first",
                  cut_is_episode_boundary=bool(episode[cut] != episode[cut - 1]))
    return (np.asarray(rows_raw) >= cut) == expert_second, record


def share(weights, mask):
    weights = np.asarray(weights, np.float64)
    total = weights.sum()
    return float(weights[mask].sum() / total) if total > 0 else None


def data_quality(prepared, rows_converted, payload_arrays, dataset):
    """corr(weight, episode normalized return) on the subsample, and the medium-expert expert-half shares."""
    import h5py

    meta = prepared.metadata
    path = Path(meta["raw_identity"]["path"])
    with h5py.File(path, "r") as f:
        if "timeouts" not in f:
            raise ValueError("episode returns need the raw timeouts")
        rewards, terminals, timeouts = (np.asarray(f[k][()]) for k in ("rewards", "terminals", "timeouts"))
    transform = meta["evaluation_score_transform"]
    episode, raw, normalized = episode_normalized_returns(rewards, terminals, timeouts, transform["reference_min"],
                                                          transform["reference_max"])
    raw_current = np.asarray(meta["dependency_maps"]["raw_current"], np.int64)
    if not np.array_equal(episode[raw_current], np.asarray(meta["original_terminal_timeout_episode_ids"], np.int64)):
        raise ValueError("raw episode segmentation differs from runtime.td3_bc._convert's")
    rows_raw = raw_current[rows_converted]
    row_return = normalized[episode[rows_raw]]
    a = payload_arrays
    weights = dict(SIG_L=a["sig_L"], SIG_pi=a["sig_pi"], W_logged=a["w_logged"], W_pi=a["w_pi"], dose=a["dose"])
    out = dict(raw_file=str(path), reference=transform, episodes_total=int(len(raw)),
               episodes_in_rows=int(np.unique(episode[rows_raw]).size),
               row_return=dict(mean=float(row_return.mean()), min=float(row_return.min()),
                               median=float(np.median(row_return)), max=float(row_return.max())),
               corr_with_episode_normalized_return={k: correlation(w, row_return) for k, w in weights.items()},
               medium_expert=None)
    if "medium-expert" in dataset:
        is_expert, record = expert_half(len(rewards), episode, normalized, rows_raw)
        out["medium_expert"] = dict(record, expert_row_share=float(is_expert.mean()),
                                    p1L_extra_weight_share=share(a["sig_L"], is_expert),
                                    p1pi_extra_weight_share=share(a["sig_pi"], is_expert),
                                    hook_extra_dose_share=share(a["dose"] - 1.0, is_expert),
                                    note="P1's extra weight is beta M: its share is sum_expert M / sum M at every beta")
    return out, dict(episode_normalized_return=row_return, raw_row=rows_raw)


# ---------------------------------------------------------------------------------------------------------
# One pool


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_evaluation(signal_dir):
    found = sorted(Path(signal_dir).glob("evaluation_*.json"))
    if len(found) != 1:
        raise ValueError(f"expected one evaluation_*.json in {signal_dir}, found {len(found)}; pass --evaluation")
    return found[0]


def choose_threshold(evaluation, name):
    """lambda* (population) or a design's bank-mean WBCP threshold from step 2's evaluation; all three recorded."""
    lam_star = float(evaluation["population"]["lambda_star"]["q1"])
    banks = {d["design"]["name"]: d["signals"]["q1"]["threshold_mean"] for d in evaluation["designs"]}
    values = dict(lambda_star=lam_star, **{k: (None if v is None else float(v)) for k, v in banks.items()})
    if name not in values or values[name] is None:
        raise ValueError(f"threshold {name!r} is not available in the evaluation")
    return values[name], values


def reproduction_gate(models, native, cal_params, unit, heldout, signal_dir, rows=GATE_ROWS):
    """Relative max differences of sigma, both heads and the policy action from step 2's artifacts on the first rows."""
    k = min(rows, len(heldout.obs))
    obs, action = heldout.obs[:k], heldout.action[:k]
    with np.load(Path(signal_dir) / "heads.npz", allow_pickle=False) as heads:
        signals = list(heads["signals"])
        ref_sigma, ref_q = heads["sigma_q1"][:k], heads["q"][:k]
        ref_unit = float(heads["unit"][signals.index("q1")])
    with np.load(Path(signal_dir) / "q1" / "frozen.npz", allow_pickle=False) as frozen:
        ref_pi = frozen["policy_action"][:k]
    got = dict(sigma=np.asarray(positive_scale(models[2].apply(cal_params, obs, action), unit)),
               q=np.asarray(models[1].apply(native.critic.params, obs, action)),
               policy_action=np.asarray(models[0].apply(native.actor.params, obs)))
    rel = lambda a, b: float(np.max(np.abs(a.astype(np.float64) - b)) / max(np.max(np.abs(b)), 1e-12))  # noqa: E731
    out = dict(rows=k, sigma=rel(got["sigma"], ref_sigma), q=rel(got["q"], ref_q),
               policy_action=rel(got["policy_action"], ref_pi), unit=abs(float(unit) - ref_unit))
    out["passed"] = max(out["sigma"], out["q"], out["policy_action"]) <= GATE_TOL and out["unit"] == 0.0
    return out


def run(signal_dir, output, *, rows=ROWS, seed=SEED, levels=LEVELS, s=S_PRIMARY, threshold="lambda_star",
        beta_ladder=BETA_LADDER, c_ladder=C_LADDER, pool_dir=None, evaluation=None, data_dir=None, log=print):
    """Level-1 diagnostics of one pool; writes level1.json, level1.md and rows.npz into the new directory `output`."""
    from flax import serialization

    from experiments.signal import frozen_signals as FS

    started = time.perf_counter()
    signal_dir, output = Path(signal_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists")
    if rows < 2:
        raise ValueError("rows must be at least 2")
    fit = json.loads((signal_dir / "fit.json").read_text(encoding="utf8"))
    if "q1" not in fit["signals"]:
        raise ValueError("the signal directory has no q1 fit")
    pool = FS.restore_pool(Path(pool_dir or fit["parent"]["pool"]), data_dir)
    if pool["checkpoint_sha256"] != fit["parent"]["checkpoint"]["sha256"]:
        raise ValueError("the pool's checkpoint differs from the one the q1 scale was fit on")
    scale_path = signal_dir / "q1" / "scale.msgpack"
    if sha256(scale_path) != fit["artifacts"]["q1"]["scale_sha256"]:
        raise ValueError("q1/scale.msgpack differs from the sha256 in fit.json")
    template = {"calibrator": pool["template"].calibrator, "residual_scale": pool["template"].residual_scale}
    scale = serialization.from_bytes(template, scale_path.read_bytes())
    cal_params, unit = scale["calibrator"].params, scale["residual_scale"]
    evaluation_path = Path(evaluation) if evaluation else find_evaluation(signal_dir)
    record = json.loads(evaluation_path.read_text(encoding="utf8"))
    if record["fit_sha256"] != sha256(signal_dir / "fit.json") or record["heads_sha256"] != sha256(
            signal_dir / "heads.npz"):
        raise ValueError("the evaluation was not computed on this fit")
    lam, thresholds = choose_threshold(record, threshold)
    models, native, prepared, args = pool["models"], pool["state"].native, pool["prepared"], pool["args"]
    gate = reproduction_gate(models, native, cal_params, unit, prepared.heldout, signal_dir)
    if not gate["passed"]:
        raise ValueError(f"reproduction gate failed: {gate}")
    n_train = len(prepared.training.obs)
    take = np.sort(np.random.default_rng([seed, 0]).choice(n_train, size=min(rows, n_train), replace=False))
    obs, action = np.asarray(prepared.training.obs)[take], np.asarray(prepared.training.action)[take]
    restored_at = time.perf_counter()
    log(f"{fit['dataset']}: {len(take)} of {n_train} training rows; gate {gate}")
    engine = Level1(models, native.actor.params, native.critic.params, cal_params, unit, lam, obs, action, args.alpha,
                    blend=fit["settings"]["blend"])
    payload, arrays = analyse(engine, levels=levels, s=s, seed=seed, beta_ladder=beta_ladder, c_ladder=c_ladder,
                              log=log)
    jax.clear_caches()  # the per-row and leverage programs are done; release them before the NumPy tail
    gc.collect()
    rows_converted = np.asarray(prepared.training_ids)[take]
    quality, quality_arrays = data_quality(prepared, rows_converted, arrays, fit["dataset"])
    failed = [k for k, v in payload["checks"].items() if v is False]
    if failed:
        raise AssertionError(f"built-in checks failed: {failed}")
    result = dict(
        schema=SCHEMA, dataset=fit["dataset"], dataset_key=fit["parent"]["dataset_key"],
        description="STEP4_DESIGN section 11, stage 4B-0: Level-1 diagnostics, no training, no J; descriptive only",
        inputs=dict(signal_dir=str(signal_dir.resolve()), pool=str(pool["pool_dir"]),
                    checkpoint=dict(step=pool["step"], sha256=pool["checkpoint_sha256"]),
                    fit_sha256=sha256(signal_dir / "fit.json"), scale_sha256=sha256(scale_path),
                    evaluation=str(evaluation_path.resolve()), evaluation_sha256=sha256(evaluation_path),
                    critic_health=record.get("critic_health"),
                    script_sha256=sha256(Path(__file__).resolve())),
        settings=dict(rows_requested=rows, rows=len(take), training_rows=n_train, seed=seed,
                      subsample="np.random.default_rng([seed, 0]).choice(training rows, rows, replace=False), sorted",
                      s=s, clip=CLIP, levels=list(levels), beta_ladder=[float(k) for k in beta_ladder],
                      c_ladder=[float(k) for k in c_ladder],
                      design_ladders=(tuple(map(float, beta_ladder)) == BETA_LADDER
                                      and tuple(map(float, c_ladder)) == C_LADDER),
                      tol=TOL, max_steps=MAX_STEPS, n_shuf=N_SHUF, n_strat=N_STRAT, deciles=DECILES,
                      threshold=threshold, threshold_value=lam, thresholds_available=thresholds,
                      blend=fit["settings"]["blend"], alpha=args.alpha),
        gate=gate, level1=payload, data_quality=quality,
        timing=dict(restore_seconds=restored_at - started, total_seconds=time.perf_counter() - started),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "level1.json").write_text(json.dumps(_json(result), indent=1, allow_nan=False), encoding="utf8")
    write_markdown(output / "level1.md", result)
    with (output / "rows.npz").open("xb") as handle:
        np.savez(handle, training_row=np.asarray(take, np.int64), converted_row=rows_converted.astype(np.int64),
                 **arrays, **quality_arrays)
    return result


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _f(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def write_markdown(path, result):
    """A readable digest of level1.json: base quantities, the per-placement clouds and the HOOK, data quality."""
    p = result["level1"]
    b = p["base"]
    lines = [f"# Step 4B-0 Level-1 diagnostics: {result['dataset']}", "",
             "experiments/signal/d4rl_level1.py (STEP4_DESIGN section 11; expectations_DRAFT Part 4). Descriptive "
             "only: no training, no J, no verdict. Delta g = g(arm) - g(NONE) at the frozen actor on "
             f"{result['settings']['rows']} training rows; arms matched at rho* = m rho_host.", "",
             f"- threshold {result['settings']['threshold']} = {_f(result['settings']['threshold_value'])}, "
             f"unit {_f(p['residual_unit'])}, lambda_Q {_f(p['lambda_q'])}",
             f"- rho_host {_f(b['rho_host'])}; cos(g_Q, g_BC) {_f(b['cos_gq_gbc'])} "
             f"({'identified' if b['identified'] else 'UNIDENTIFIED: m_eff and tau flagged'}); "
             f"cos(g_pen, g_BC) {_f(b['cos_gpen_gbc'])}",
             f"- gate {result['gate']}", f"- checks {p['checks']}", "",
             "## Weights and leverage", "",
             f"- multiset {p['multiset']}", f"- T-lin {p['t_lin']}",
             f"- Spearman(weight, exact per-row gradient norm) {p['leverage']['spearman']}",
             f"- Spearman(l_BC, l_Q) {_f(p['leverage']['spearman_bc_q'])}; "
             f"Spearman(W_logged, W_pi) {_f(p['signal']['spearman_W_logged_W_pi'])}", ""]
    hook = p["hook"]
    lines += ["## HOOK", "", f"cos(Delta g_HOOK, Delta g_P0) {_f(hook['cos_dg_p0'])}; rho/rho_host "
              f"{_f(hook['m_equivalent'])}; m_eff {_f(hook['m_eff'])}; tau {_f(hook['tau'])}; dose {hook['dose']}", ""]
    for name, placement in p["placements"].items():
        lines += [f"## {name}", "", "| m | SIG knob | SIG rho | cos(dg_SIG, dg_P0) | SHUF band p05-p95 | "
                  "outside SHUF | STRAT band | outside STRAT | ANTI cos | SIG a, b | SIG m_eff | SIG tau | "
                  "reachable SIG/SHUF/STRAT/ANTI | non-monotone |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, level in placement["levels"].items():
            sig, summary = level["arms"]["SIG"], level["summary"]
            sc, tc, r = summary["shuf_cos"], summary["strat_cos"], summary["reachable"]
            lines.append(f"| {key} | {_f(sig['knob'])} | {_f(sig['rho'])} | {_f(sig.get('cos_dg_p0'))} | "
                         f"{_f(sc['p05'])} - {_f(sc['p95'])} | {sc['outside']} | {_f(tc['p05'])} - {_f(tc['p95'])} | "
                         f"{tc['outside']} | {_f(summary['anti_cos'])} | {_f(sig.get('a'))}, {_f(sig.get('b'))} | "
                         f"{_f(sig.get('m_eff'))} | {_f(sig.get('tau'))} | "
                         f"{r['SIG']}/{r['SHUF']}/{r['STRAT']}/{r['ANTI']} | {summary['nonmonotone_ladders']} |")
        if "natural_c1" in placement:
            nat = placement["natural_c1"]
            lines.append(f"\nP3 at c = 1: rho {_f(nat['rho'])}, cos(dg, dg_P0) {_f(nat['cos_dg_p0'])}, "
                         f"m_eff {_f(nat['m_eff'])}, tau {_f(nat['tau'])}. Nulls: {placement['nulls']}.")
        if name.startswith("P2"):
            lines.append("\nP2 rescales the Q term: cos(dg, -g_Q) (constant-trust increment) per level: "
                         + "; ".join(f"m={k}: SIG {_f(v['arms']['SIG'].get('cos_dg_const_trust'))}"
                                     for k, v in placement["levels"].items()))
        lines.append("")
    q = result["data_quality"]
    lines += ["## Data quality", "", f"- rows' episode normalized return {q['row_return']}",
              f"- corr(weight, episode normalized return) {q['corr_with_episode_normalized_return']}",
              f"- medium-expert {q['medium_expert']}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--signal-dir", type=Path, required=True, help="runs/wbcp_signal/frozen/<pool> (step 2's fit)")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--rows", type=int, default=ROWS, help="training rows (seeded subsample)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--levels", type=float, nargs="+", default=list(LEVELS),
                        help="uniform-BC multipliers m: rho* = m rho_host")
    parser.add_argument("--s", type=float, default=S_PRIMARY, help="rank-normal spread")
    parser.add_argument("--threshold", choices=THRESHOLDS, default="lambda_star")
    parser.add_argument("--beta-ladder", type=float, nargs="+", default=list(BETA_LADDER),
                        help="P1/P2 knob ladder (default: STEP4_DESIGN section 3's)")
    parser.add_argument("--c-ladder", type=float, nargs="+", default=list(C_LADDER),
                        help="P3 knob ladder (default: STEP4_DESIGN section 3's)")
    parser.add_argument("--pool", type=Path, help="the parent pool (default: fit.json's parent)")
    parser.add_argument("--evaluation", type=Path, help="step 2's evaluation JSON (default: the one evaluation_*.json)")
    parser.add_argument("--data-dir", type=Path, help="cached D4RL HDF5 directory (default ~/.d4rl/datasets)")
    return parser


def main(argv=None):
    opt = build_parser().parse_args(argv)
    if opt.output.exists():
        raise SystemExit("--output already exists")
    result = run(opt.signal_dir, opt.output, rows=opt.rows, seed=opt.seed, levels=opt.levels, s=opt.s,
                 threshold=opt.threshold, beta_ladder=opt.beta_ladder, c_ladder=opt.c_ladder, pool_dir=opt.pool,
                 evaluation=opt.evaluation, data_dir=opt.data_dir)
    print(json.dumps(dict(dataset=result["dataset"], rows=result["settings"]["rows"], gate=result["gate"],
                          checks=result["level1"]["checks"], seconds=result["timing"]["total_seconds"]), default=str))
    return result


if __name__ == "__main__":
    main()
