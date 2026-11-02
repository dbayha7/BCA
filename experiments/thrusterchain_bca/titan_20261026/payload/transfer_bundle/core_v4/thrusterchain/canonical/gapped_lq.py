"""Gapped LQ world (GLQ-10x4x5): exact truth, Stage 0, behaviour data and the R0/R1 log-only checks.

Design: option A of the world audit ("Gapped LQ", scratchpad design_glq.json) with the judge's grafts g1 (decoy
context), g2 (density ladder) and g7 (noisy context code, off by default). This module is the world and its exact
truth only. It trains nothing, fits no band and computes no critic error: glq_train.py trains, glq_acceptance.py runs
the gate stages. lq_harness is imported and called, never edited.

PLANT. s in R^10 is five 2-D oscillators, s = (p1, v1, ..., p5, v5). A = blockdiag_j(rho_j Rot(theta_j)) + C, with
theta = (0.10, 0.20, 0.35, 0.55, 0.80) rad (periods 63, 31, 18, 11, 8 steps), rho_j = rho_top - rho_step (j - 1)
(design: 1.00 and 0.01, radii 1.00-0.96; Stage 0: 0.98 and 0.005, radii 0.98-0.96), and C a seeded Gaussian coupling
scaled to spectral norm 0.02: the first draw np.random.default_rng([coupling_seed, attempt]) with spectral radius
rho(A) <= 1.00 is kept (attempt is recorded). Actuator k = 1..4 pushes the velocities of oscillators k and k+1 with
`gain`, so actuators 1-2 reach oscillators 1-3 only and oscillators 4-5 are reached by actuators 3-4 (and through C).
s' = A s + B_c a + eps_c, eps_c ~ N(0, sd_c^2 I_10). Reward r = -(|s|^2 + R |a|^2), R = action_cost, no reward noise.
Returns depend on (gain, R) only through R / gain^2 (scaling gain by c and R by c^2 scales every gain K by 1 / c);
gain alone sets the action scale. Design: gain 0.3, R 0.5 (R / gain^2 = 5.6); Stage 0: gain 0.085, R = 20 (0.085 /
0.3)^2 = 1.606 (R / gain^2 = 222). gamma 0.98; episodes are truncated at 100 steps (logged done = 0, as lq_harness);
s0 ~ N(0, I_10). Actions are unbounded (TD3's target clip gets max_action = inf, lq_harness.MAX_ACTION), so every
Q^pi below is exact for every action.

CONTEXTS (index: name, B_c, transition noise SD). One context per episode, observed, fixed within the episode:
  0 R0       B                                         0.1
  1 R1       B with actuators 1-2 at 0.6x              0.1
  2 T_wrong  B with columns 3-4 replaced by B_34 W     0.1   W = -I_2 ("negate", default) or Rot(135 deg)
  3 T_right  B                                         0.1
  4 DECOY    B                                         0.3   (graft g1: 3x the transition noise)
Observation = [s (10), code (5)] = 15 numbers; code = onehot(c), or with g7 onehot(c) + code_sd xi_e, xi_e ~ N(0, I_5)
drawn once per episode (code_sd = None, the default, gives the exact one-hot). Context mix (R0, R1, T_wrong, T_right,
DECOY) = (0.30, 0.30, 0.15, 0.15, 0.10); by default every split holds exactly these shares of its episodes, in a
seeded random order ("stratified"; "categorical" draws each episode's context independently, as the design text).

EXACT TRUTH. Per context, lq_harness.System(A, B_c, I_10, R I_4, sd_c^2 I_10, I_10, 0.98, 100) (World.systems), and
for any per-context linear policy a = -K_c s (K of shape (5, 4, 10), ContextLinearActor's layout; a (4, 10) gain is
used in every context):
- V_c(s) = s^T P_c s + h_c and Q_c(s, a) = z^T H_c z + h_c, z = (s, a), from lq_harness.quadratic_q (discrete
  Lyapunov; h_c = gamma tr(P_c Sigma_c) / (1 - gamma)). The infinite-horizon value is what TD3+BC's bootstrapped
  critic targets, so critic errors are e = Q1 - Q^pi with these H_c, h_c (World.q, World.grad_a_q, World.v).
- J_inf,c = tr(P_c) + h_c (lq_harness.value; -inf when sqrt(gamma) rho(A - B_c K_c) >= 1).
- J_c, the headline: the exact 100-step discounted return sum_{t<100} gamma^t E r_t by the forward covariance recursion
  Sigma_{t+1} = M Sigma_t M^T + B Sigma_a B^T + Sigma_eps, M = A - B_c K_c, E r_t = -tr(Sigma_t) - R tr(K Sigma_t K^T
  + Sigma_a) (finite_return; Sigma_a = 0 for a deterministic policy). Always finite. finite_return_backward is the
  independent backward (value-matrix) recursion; the tests check that they agree.
- K*_c = lq_harness.lqr_gain (4 actuators); K2_c = lqr_gain with B_c[:, :2] and R I_2, embedded with zero rows 3-4
  (two_actuator_gain; equal in T_wrong and T_right, whose first two columns agree).
- J_BC,c = J_c(K_beh,c), the clone of the behaviour's mean gain; J_opt,c = J_c(K*_c); NS = (J - J_BC) / (J_opt - J_BC)
  per context and for the mixture J = sum_c w_c J_c (World.score). K* maximises J_inf, not the 100-step J, so NS can
  slightly exceed 1; Stage 0 reports the 100-step gradient at K*.
- Identity (negate): J_T_wrong(K12, D) = J_T_right(K12, -D) for any rows K12 (actuators 1-2) and block D (3-4), for
  J_inf and J, and Q_T_wrong(s, a) = Q_T_right(s, flip(a)) under those policies; with Rot(135 deg), -D becomes W D.
- TD3's fixed point (World.td3_level). TD3's critic target reads Q(s', pi(s') + eps) with eps = clip(N(0, 0.2^2),
  -0.5, 0.5) per actuator, so even a perfect critic converges to Q^{pi + eps}(s, a) = Q^pi_c(s, a) + c_c with the
  constant c_c = gamma var_clip tr(H_c,aa) / (1 - gamma) (var_clip = clipped_noise_variance(0.2, 0.5) = 0.0391; the
  quadratic part H_c and so every action gradient are unchanged). Here c_c is about -13.3, 0.15-0.21 of the median
  |Q^pi| in R0 and R1: level errors are therefore also measured against Q^pi + c_c (glq_train.error_rows' e1_td3), the
  quantity a perfect TD3 critic gets to 0. The min over the two heads adds a further, learned, underestimation.

BEHAVIOUR (no error is planted anywhere). Rich contexts (R0, R1, DECOY): a = -f K*_c s + 0.3 n, n ~ N(0, I_4), f =
behaviour_fraction. Thin contexts (T_wrong, T_right): a_12 = -K2_c s + 0.1 n_12 and a_34 = rung n_34, rung on the
density ladder {0, 0.01, 0.03, 0.1, 0.3} (graft g2; primary rung 0.01: thin, not empty; rung 0 gives a_3 = a_4 = 0
exactly, and then the T_wrong and T_right logs have the same distribution). Null variant (R7 anchor): f = 1.0.

STAGE 0 (outcome-free: closed form, no data, no critic, no actor). Targets, every J being the headline 100-step return:
- S1: every rich context's behaviour gets <= 50% of the way from J(K = 0) to J(K*): (J(f K*) - J(0)) / (J(K*) - J(0)).
- S2: g = J(K*) - J(K2) in a thin context; 0.3 g >= 15% of the total gap sum_c w_c (J_opt,c - J_BC,c).
- S3: K*_T_right used in T_wrong is discounted-stable (sqrt(gamma) rho < 1) and loses >= 2 g against the
  behaviour there: J_Tw(K2) - J_Tw(K*_Tr) >= 2 g. Also reported: the block transplant [K2 rows 1-2; K*_Tr rows 3-4].
- S4: >= 95% of logged actions and of K*'s actions (along their own closed loops, mixture-weighted) have every
  coordinate in [-1, 1] (exact Gaussian box probabilities; per context reported).
- S5 (R0's marks in closed form, behaviour log, exact Gaussian state covariances): participation ratio >= 6.0 of 10
  in every context; transient share >= 30%; integrated autocorrelation time of the leading state component in
  [10, 50] steps; >= 99% of rows with every |s_i| <= 5 (union bound). Definitions in r0_closed_form.
- DECOY: the exact aleatoric target SD (sd of r + gamma V^pi(s') given (s, a), pi = the behaviour clone) has a decoy
  median >= 3x the rich (R0, R1) median, at rows drawn from the exact behaviour state distribution.
- S6 (added here, not in the design's S-list): R6(a)'s mark applied to the ideal transferred policy: |K*_Tr,34 s| >=
  0.3 (one rich noise SD) on >= 30% of thin-context behaviour states. It is necessary for R6(a), which a host whose
  critic transfers perfectly could not pass otherwise (at the design gain 0.3 the share is about 1%).
- Every policy used must be discounted-stable: K = 0, f K*, K*, K2, the expert K*, and both transplants.
Normalised margins: (mark - x) / mark or (x - mark) / mark per target, (x - 0.95) / 0.05 for S4.
Step A (returns structure, at the design gain 0.3): STAGE0_GRID over wrong_map, rho_top, rho_step, action_cost and
behaviour_fraction (noise SDs and the mix stay at the design's values). The sign flip is preferred to the 135-degree
rotation (the design's fallback). Among candidates passing every target but S6, the winner maximises (passes the cheap
targets under each of 4 alternative coupling draws, the worst normalised margin over its targets and the
alternatives', smaller action_cost, larger f). Step B (action scale): gain over STAGE0_GAINS with R x (gain / 0.3)^2,
which leaves S1-S3 unchanged; among gains passing every target including S6 (and every alternative coupling's cheap
targets), the winner maximises the worst margin of the two scale-dependent targets, S4 and S6 (ties: larger gain).
Every candidate's numbers go to stage0.json. The chosen values are STAGE0_SPEC; `stage0` refuses to write if the search
no longer selects it.

DATA (generate). Per seed rep: 1,000 episodes x 100 steps, split by episode into train (500), calibration (200) and
evaluation (300). The calibration bank is calibration/bank.stratified_bank, 5 rows from every one of the 200 held-out
episodes (1,000 rows); evaluation keeps every row. Random streams np.random.default_rng([seed, rep, TAG_DATA, k]):
k = 0 contexts, 1 s0, 2 action noise n (all 4 coordinates, drawn whatever the rung), 3 transition shocks, 5 bank,
6 code noise. The noise streams do not depend on the rung, the behaviour fraction or the context draw, so ladder rungs
and null variants share their random numbers. Rows are float64 (obs, state, action, reward, next_obs, next_state,
noise, shock, context, episode, t); transitions() is the float32 boundary (runtime.networks.Transition, done = 0).

R0/R1 LOG-ONLY CHECKS (log_metrics, on the training split): per context and pooled, the participation ratio of the
state covariance, the transient share, the integrated autocorrelation time of the leading component, the state scale,
the action-covariance eigenvalues and rank, the actuator-3-4 SD in thin rows, and the thin share of training rows.

python experiments/signal/gapped_lq.py stage0 --output runs/glq/stage0
python experiments/signal/gapped_lq.py logcheck --output runs/glq/logcheck --seeds 0 1 2 3 4
Tests: JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_gapped_lq
"""

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy import linalg, stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from calibration import bank  # noqa: E402
from experiments.signal import lq_harness as LQ  # noqa: E402

CONTEXTS = ("R0", "R1", "T_wrong", "T_right", "DECOY")
R0, R1, T_WRONG, T_RIGHT, DECOY = range(5)
RICH, THIN, RICH_MAIN = (R0, R1, DECOY), (T_WRONG, T_RIGHT), (R0, R1)
DS, DA, NC = 10, 4, 5
OBS_DIM = DS + NC
LADDER = (0.0, 0.01, 0.03, 0.1, 0.3)
PRIMARY_RUNG = 0.01
WRONG_MAPS = ("negate", "rotate135")
SPLITS = ("train", "cal", "eval")
TAG_DATA = 7  # np.random.default_rng([seed, rep, TAG_DATA, k])
DATA_SEED = 20261012
MAX_ACTION = LQ.MAX_ACTION  # inf: no clipping anywhere, so Q^pi is exact for every action
TD3_POLICY_NOISE, TD3_NOISE_CLIP = 0.2, 0.5  # td3_bc.Args' target smoothing (World.td3_level)

# Pass marks used by Stage 0 and the log checks (the design's S1-S5, R0 and R1(a)(b) marks).
S1_MAX, S2_MIN, S3_MIN, S4_MIN = 0.5, 0.15, 2.0, 0.95
PR_MIN, TRANSIENT_MIN, IACT_RANGE, SCALE_BOUND, SCALE_SHARE = 6.0, 0.30, (10.0, 50.0), 5.0, 0.99
DECOY_RATIO_MIN = 3.0
S6_SHARE_MIN = 0.30  # R6(a)'s share, applied to K*_T_right's actuator-3-4 action (added in Stage 0)
RICH_EIG_MIN, THIN_EIG_REL, THIN_SD_TOL, THIN_SHARE, THIN_SHARE_TOL = 0.05, 1e-12, 0.05, 0.30, 0.03
IACT_T0, IACT_CUT, IACT_MAX_LAG, IACT_BASELINE_LAGS = 20, 0.05, 60, 24


@dataclass(frozen=True)
class WorldSpec:
    """Every free number of the world. The defaults are the design's values (DESIGN_SPEC)."""

    rho_top: float = 1.00  # rho_j = rho_top - rho_step (j - 1), j = 1..5
    rho_step: float = 0.01
    theta: tuple = (0.10, 0.20, 0.35, 0.55, 0.80)
    coupling_norm: float = 0.02
    coupling_seed: int = 20261011
    max_radius: float = 1.0
    gain: float = 0.3
    action_cost: float = 0.5  # R = action_cost I_4
    gamma: float = 0.98
    length: int = 100
    init_sd: float = 1.0
    noise_sd: float = 0.1
    decoy_noise_sd: float = 0.3
    r1_scale: float = 0.6  # R1: actuators 1-2 at this gain multiple
    wrong_map: str = "negate"
    wrong_gain: float = 1.0  # T_wrong's actuator-3-4 gain multiple (the design's retune lever x 1.5)
    behaviour_fraction: float = 0.4
    rich_noise_sd: float = 0.3
    thin_noise_sd: float = 0.1
    thin_rung: float = PRIMARY_RUNG
    mix: tuple = (0.30, 0.30, 0.15, 0.15, 0.10)


DESIGN_SPEC = WorldSpec()
# Stage 0's selection (stage0_search over STAGE0_GRID; `stage0` checks that the search still returns it).
STAGE0_SPEC = WorldSpec(rho_top=0.98, rho_step=0.005, gain=0.085, action_cost=20.0 * (0.085 / 0.3) ** 2,
                        behaviour_fraction=0.2)
STAGE0_GRID = dict(wrong_map=WRONG_MAPS, rho_top=(1.00, 0.99, 0.98, 0.97), rho_step=(0.01, 0.005),
                   action_cost=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
                   behaviour_fraction=(0.4, 0.3, 0.25, 0.2, 0.175, 0.15))
ALT_COUPLING_SEEDS = (1, 2, 3, 4)  # offsets added to coupling_seed for Stage 0's robustness check
STAGE0_GAINS = (0.3, 0.2, 0.15, 0.12, 0.1, 0.09, 0.085, 0.08, 0.075)  # step B: R scales with gain^2
SCALE_TARGETS = ("S4", "S6")  # the targets the action scale moves; step B maximises their worst margin


def spec_dict(spec):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(spec).items()}


def spec_digest(spec):
    """SHA-256 of the spec's canonical JSON (sorted keys)."""
    return hashlib.sha256(json.dumps(spec_dict(spec), sort_keys=True).encode()).hexdigest()


def array_digest(*arrays):
    h = hashlib.sha256()
    for x in arrays:
        h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------
# The plant


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def wrong_matrix(spec):
    """W of T_wrong's actuator-3-4 columns: exactly -I_2 for 'negate', Rot(135 deg) for 'rotate135'."""
    if spec.wrong_map == "negate":
        W = -np.eye(2)
    elif spec.wrong_map == "rotate135":
        W = rotation(3 * math.pi / 4)
    else:
        raise ValueError("wrong_map must be one of " + ", ".join(WRONG_MAPS))
    return spec.wrong_gain * W


def radii(spec):
    return tuple(spec.rho_top - spec.rho_step * j for j in range(len(spec.theta)))


def oscillators(spec):
    rho = radii(spec)
    return linalg.block_diag(*[r * rotation(t) for r, t in zip(rho, spec.theta)])


def transition_matrix(spec, attempts=1000):
    """A = blockdiag(rho_j Rot(theta_j)) + C: the first seeded coupling of spectral norm coupling_norm with
    rho(A) <= max_radius. Returns (A, attempt)."""
    A0 = oscillators(spec)
    for attempt in range(attempts):
        X = np.random.default_rng([spec.coupling_seed, attempt]).standard_normal((DS, DS))
        A = A0 + spec.coupling_norm * X / np.linalg.norm(X, 2)
        if np.max(np.abs(np.linalg.eigvals(A))) <= spec.max_radius:
            return A, attempt
    raise ValueError(f"no coupling draw in {attempts} keeps the spectral radius <= {spec.max_radius}")


def base_input(gain):
    """Actuator k pushes the velocities (coordinates 2j + 1) of oscillators k and k + 1 with `gain`."""
    B = np.zeros((DS, DA))
    for k in range(DA):
        B[2 * k + 1, k] = gain
        B[2 * (k + 1) + 1, k] = gain
    return B


def input_matrices(spec):
    """(5, 10, 4): B_c of R0, R1, T_wrong, T_right, DECOY."""
    B = base_input(spec.gain)
    r1 = B.copy()
    r1[:, :2] *= spec.r1_scale
    wrong = B.copy()
    wrong[:, 2:] = B[:, 2:] @ wrong_matrix(spec)
    return np.stack([B, r1, wrong, B.copy(), B.copy()])


def two_actuator_gain(system):
    """The discounted LQR optimum using actuators 1-2 only, embedded as a (4, 10) gain with zero rows 3-4."""
    sub = replace(system, B=system.B[:, :2], Rc=system.Rc[:2, :2])
    return np.vstack([LQ.lqr_gain(sub), np.zeros((DA - 2, system.A.shape[0]))])


def clipped_noise_variance(sd, clip):
    """Exact variance of clip(N(0, sd^2), -clip, clip), TD3's target-smoothing noise per actuator (mean 0 by
    symmetry): sd^2 (1 - 2 Q(z) - 2 z phi(z)) + 2 clip^2 Q(z), z = clip / sd, Q the normal tail."""
    sd, clip = float(sd), float(clip)
    if sd == 0:
        return 0.0
    if not math.isfinite(clip):
        return sd * sd
    z = clip / sd
    tail = float(stats.norm.sf(z))
    return float(sd * sd * (1 - 2 * tail - 2 * z * stats.norm.pdf(z)) + 2 * clip * clip * tail)


def discounted_radius(system, K):
    """sqrt(gamma) rho(A - B K): the closed loop is discounted-stable iff this is < 1."""
    return float(math.sqrt(system.gamma) * np.max(np.abs(np.linalg.eigvals(system.A - system.B @ K))))


def finite_return(system, K, action_cov=None, length=None, covariances=False):
    """Exact sum_{t < L} gamma^t E r_t for a = -K s + N(0, action_cov), s0 ~ N(0, Sigma_0) (forward recursion).

    Returns J, or (J, [Sigma_0, ..., Sigma_{L-1}]) with covariances=True. Finite for any K.
    """
    L = system.length if length is None else length
    K = np.asarray(K, np.float64)
    Sa = np.zeros((K.shape[0],) * 2) if action_cov is None else np.asarray(action_cov, np.float64)
    M = system.A - system.B @ K
    G = system.Qc + K.T @ system.Rc @ K
    drive = system.noise + system.B @ Sa @ system.B.T
    const = float(np.trace(system.Rc @ Sa))
    S, J, out = np.array(system.init, np.float64), 0.0, []
    for t in range(L):
        if covariances:
            out.append(S)
        J -= system.gamma ** t * (float(np.trace(G @ S)) + const)
        S = M @ S @ M.T + drive
    return (J, out) if covariances else J


def finite_return_backward(system, K, length=None):
    """The same J by the backward recursion: P_L = 0, P_t = -G_pi + gamma M^T P_{t+1} M, c_t = gamma (tr(P_{t+1}
    Sigma_eps) + c_{t+1}), J = tr(P_0 Sigma_0) + c_0 (deterministic policy)."""
    L = system.length if length is None else length
    K = np.asarray(K, np.float64)
    M = system.A - system.B @ K
    G = system.Qc + K.T @ system.Rc @ K
    P, c = np.zeros_like(system.A), 0.0
    for _ in range(L):
        P, c = -G + system.gamma * M.T @ P @ M, system.gamma * (float(np.trace(P @ system.noise)) + c)
    return float(np.trace(P @ system.init)) + c


# ---------------------------------------------------------------------------------------------------------
# The world and its exact truth


class World:
    """The five per-context LQ systems of one WorldSpec and every exact quantity of per-context linear policies."""

    def __init__(self, spec=STAGE0_SPEC):
        if spec.wrong_map not in WRONG_MAPS:
            raise ValueError("wrong_map must be one of " + ", ".join(WRONG_MAPS))
        if len(spec.mix) != NC or abs(sum(spec.mix) - 1) > 1e-12 or min(spec.mix) < 0:
            raise ValueError("mix must be 5 nonnegative shares summing to 1")
        self.spec = spec
        self.rho = radii(spec)
        self.A, self.coupling_attempt = transition_matrix(spec)
        self.B = input_matrices(spec)
        self.noise_sd = np.array([spec.noise_sd] * 4 + [spec.decoy_noise_sd], np.float64)
        self.mix = np.asarray(spec.mix, np.float64)
        self.systems = tuple(
            LQ.System(self.A, self.B[c], np.eye(DS), spec.action_cost * np.eye(DA), self.noise_sd[c] ** 2 * np.eye(DS),
                      spec.init_sd ** 2 * np.eye(DS), float(spec.gamma), int(spec.length)) for c in range(NC))
        self.k_star = np.stack([LQ.lqr_gain(s) for s in self.systems])
        self.k_two = np.stack([two_actuator_gain(s) for s in self.systems])
        f = spec.behaviour_fraction
        self.k_behaviour = np.stack([f * self.k_star[c] if c in RICH else self.k_two[c] for c in range(NC)])
        self.behaviour_sd = np.stack([np.full(DA, spec.rich_noise_sd) if c in RICH else
                                      np.array([spec.thin_noise_sd] * 2 + [spec.thin_rung] * 2) for c in range(NC)])

    # -- policies --------------------------------------------------------------------------------------------
    @staticmethod
    def per_context(K):
        """(5, 4, 10) float64 from a per-context gain or one (4, 10) gain used in every context."""
        K = np.asarray(K, np.float64)
        if K.shape == (DA, DS):
            return np.repeat(K[None], NC, 0)
        if K.shape != (NC, DA, DS):
            raise ValueError(f"a gain must have shape ({DA}, {DS}) or ({NC}, {DA}, {DS}), not {K.shape}")
        return K

    def action_cov(self, c):
        return np.diag(self.behaviour_sd[c] ** 2)

    def transplant(self, block=False):
        """K*_T_right used in T_wrong (block=True: K2 rows 1-2 with K*_T_right's rows 3-4)."""
        K = self.k_star[T_RIGHT].copy()
        if block:
            K[:2] = self.k_two[T_WRONG][:2]
        return K

    # -- values ----------------------------------------------------------------------------------------------
    def radius(self, K):
        K = self.per_context(K)
        return np.array([discounted_radius(s, K[c]) for c, s in enumerate(self.systems)])

    def quadratic(self, K):
        """Per context (H_c (14, 14), h_c, P_c (10, 10)) of a = -K_c s (lq_harness.quadratic_q); raises if unstable."""
        K = self.per_context(K)
        return [LQ.exact_q(s, K[c]) for c, s in enumerate(self.systems)]

    def j_inf(self, K):
        K = self.per_context(K)
        return np.array([LQ.value(s, K[c]) for c, s in enumerate(self.systems)])

    def j_finite(self, K, behaviour_noise=False):
        """Per-context exact 100-step discounted return (behaviour_noise adds each context's logged action noise)."""
        K = self.per_context(K)
        return np.array([finite_return(s, K[c], self.action_cov(c) if behaviour_noise else None)
                         for c, s in enumerate(self.systems)])

    def j_episodes(self, gains, context, finite=True):
        """Exact J (100-step, or J_inf with finite=False) of per-episode gains K_e in contexts c_e: g7's gain-scheduled
        policy K_theta(code_e) is linear in s within an episode, so each term is exact; average over episodes."""
        fn = finite_return if finite else LQ.value
        return np.array([fn(self.systems[int(c)], np.asarray(K, np.float64)) for K, c in zip(gains, context)])

    @cached_property
    def j_bc(self):
        return self.j_finite(self.k_behaviour)

    @cached_property
    def j_opt(self):
        return self.j_finite(self.k_star)

    def normalised(self, J):
        """NS per context (None where J_opt = J_BC) and of the mixture, from per-context J."""
        J = np.asarray(J, np.float64)
        gap = self.j_opt - self.j_bc
        per = [float((J[c] - self.j_bc[c]) / gap[c]) if gap[c] != 0 else None for c in range(NC)]
        mix_gap = float(self.mix @ gap)
        return per, (float((self.mix @ J - self.mix @ self.j_bc) / mix_gap) if mix_gap != 0 else None)

    def score(self, K):
        """Exact 100-step J per context and mixed, NS per context and mixed, and J_inf per context."""
        J = self.j_finite(K)
        per, mixed = self.normalised(J)
        j_inf = self.j_inf(K)
        return dict(J=J.tolist(), J_mix=float(self.mix @ J), NS=per, NS_mix=mixed,
                    J_inf=[float(x) if math.isfinite(x) else None for x in j_inf],
                    stable=[bool(r < 1) for r in self.radius(K)])

    # -- row-level truth -------------------------------------------------------------------------------------
    def _rows(self, K, context):
        context = np.asarray(context, np.int64)
        return self.quadratic(K), context

    def q(self, K, s, a, context):
        """Exact Q^pi_c(s, a) per row (float64; context per row)."""
        parts, context = self._rows(K, context)
        out = np.empty(len(context))
        for c in range(NC):
            rows = context == c
            if rows.any():
                H, h, _ = parts[c]
                out[rows] = LQ.qform(H, h, s[rows], a[rows])
        return out

    def v(self, K, s, context):
        K = self.per_context(K)
        context = np.asarray(context, np.int64)
        return self.q(K, s, -np.einsum("nij,nj->ni", K[context], s), context)

    def grad_a_q(self, K, s, a, context):
        """Exact grad_a Q^pi_c(s, a) = 2 (H_c z)_a per row, shape (n, 4)."""
        parts, context = self._rows(K, context)
        out = np.empty((len(context), DA))
        for c in range(NC):
            rows = context == c
            if rows.any():
                H = parts[c][0]
                z = np.concatenate([s[rows], a[rows]], 1)
                out[rows] = 2 * z @ H[:, DS:]
        return out

    def td3_level(self, K, policy_noise=TD3_POLICY_NOISE, noise_clip=TD3_NOISE_CLIP):
        """Per context, c_c with Q_td3 = Q^pi_c + c_c: the fixed point of TD3's smoothed bootstrapped target
        r + gamma E Q(s', pi(s') + eps), eps = clip(N(0, policy_noise^2), +-noise_clip) per actuator, is the Q of the
        noisy policy pi + eps from the second step on, = Q^pi_c + gamma var tr(H_c,aa) / (1 - gamma) (module
        docstring). NaN where the closed loop is not discounted-stable."""
        K = self.per_context(K)
        var = clipped_noise_variance(policy_noise, noise_clip)
        out = np.full(NC, np.nan)
        for c, system in enumerate(self.systems):
            try:
                H = LQ.exact_q(system, K[c])[0]
            except ValueError:
                continue
            out[c] = system.gamma * var * float(np.trace(H[DS:, DS:])) / (1 - system.gamma)
        return out

    def policy(self, K, s, context):
        K = self.per_context(K)
        return -np.einsum("nij,nj->ni", K[np.asarray(context, np.int64)], s)

    def aleatoric_sd(self, K, s, a, context):
        """Exact SD of the target r + gamma V^pi(s') given (s, a): gamma sqrt(4 m^T P S P m + 2 tr((P S)^2)),
        m = A s + B_c a, S = Sigma_eps,c (transition noise only; the reward is deterministic)."""
        parts, context = self._rows(K, context)
        out = np.empty(len(context))
        for c in range(NC):
            rows = context == c
            if rows.any():
                P, S = parts[c][2], self.systems[c].noise
                m = s[rows] @ self.A.T + a[rows] @ self.B[c].T
                PSP = P @ S @ P
                var = 4 * np.einsum("ni,ij,nj->n", m, PSP, m) + 2 * np.trace(P @ S @ P @ S)
                out[rows] = self.spec.gamma * np.sqrt(var)
        return out

    def behaviour_density_ratio(self, s, a, context, floor=0.01):
        """The behaviour's conditional action density at a relative to its peak at s (exact Gaussian; SD floored at
        `floor` in never-logged directions), for R1(c)."""
        context = np.asarray(context, np.int64)
        mean = self.policy(self.k_behaviour, s, context)
        sd = np.maximum(self.behaviour_sd[context], floor)
        return np.exp(-0.5 * np.sum(np.square((a - mean) / sd), axis=1))

    def state_covariances(self, K, c, action_cov=None):
        """[Sigma_0 .. Sigma_{L-1}] of the state under a = -K_c s + N(0, action_cov) in context c."""
        return finite_return(self.systems[c], self.per_context(K)[c], action_cov, covariances=True)[1]

    def describe(self):
        return dict(spec=spec_dict(self.spec), spec_sha256=spec_digest(self.spec), rho=list(self.rho),
                    coupling_attempt=self.coupling_attempt,
                    spectral_radius_A=float(np.max(np.abs(np.linalg.eigvals(self.A)))),
                    A=self.A.tolist(), B=self.B.tolist(), noise_sd=self.noise_sd.tolist(),
                    K_star=self.k_star.tolist(), K_two=self.k_two.tolist(), K_behaviour=self.k_behaviour.tolist(),
                    behaviour_sd=self.behaviour_sd.tolist(),
                    matrices_sha256=array_digest(self.A, self.B, self.k_star, self.k_two, self.k_behaviour))


# ---------------------------------------------------------------------------------------------------------
# Closed-form R0 marks (Stage 0's S5) and the exact action-box probabilities (S4)


def participation_ratio(cov):
    ev = np.linalg.eigvalsh(cov)
    return float(ev.sum() ** 2 / np.square(ev).sum())


def settle_step(level):
    """First t with |level_t - stat| <= 0.1 |level_0 - stat|, stat the mean of the last 10 steps (baseline_r0's
    definition, also used on logs)."""
    level = np.asarray(level, np.float64)
    stat = level[-10:].mean()
    return int(np.argmax(np.abs(level - stat) <= 0.1 * abs(level[0] - stat)))


def iact(acf, cut=IACT_CUT):
    """Integrated autocorrelation time 1 + 2 sum_{k=1}^{k*} c_k, summing lags until c_k first drops to <= cut
    (primary), and baseline_r0's 1 + 2 sum of every c_k > cut over lags 1..24 (secondary)."""
    acf = np.asarray(acf, np.float64)
    below = np.flatnonzero(acf[1:] <= cut)
    stop = below[0] + 1 if below.size else len(acf)
    lags = acf[1:IACT_BASELINE_LAGS + 1]
    return float(1 + 2 * acf[1:stop].sum()), float(1 + 2 * lags[lags > cut].sum()), bool(below.size > 0)


def _log_cov_parts(world, c, K=None, action_cov=None):
    K = world.k_behaviour if K is None else K
    action_cov = world.action_cov(c) if action_cov is None else action_cov
    covs = world.state_covariances(K, c, action_cov)
    M = world.A - world.B[c] @ world.per_context(K)[c]
    return np.stack(covs), M


def _acf_closed(covs, M, v, t0=IACT_T0, max_lag=IACT_MAX_LAG):
    """Numerator and denominator of the pooled-over-t autocorrelation of v^T s, t >= t0 (baseline estimator's
    population value): num_k = mean_{t0 <= t < L - k} v^T M^k Sigma_t v."""
    L = len(covs)
    vs = np.einsum("i,tij,j->t", v, covs, v)
    den = vs[t0:].mean()
    num, Mk = [den], np.eye(len(v))
    for k in range(1, max_lag + 1):
        Mk = M @ Mk
        num.append(float(np.mean(np.einsum("i,ij,tjk,k->t", v, Mk, covs[t0:L - k], v))))
    return np.array(num), den


def r0_closed_form(world, K=None, action_sd=None):
    """S5: exact R0 marks of the behaviour log (or of another linear policy with given action SDs)."""
    out, pooled_cov, pooled_level2, parts = {}, 0.0, 0.0, {}
    for c in range(NC):
        Sa = world.action_cov(c) if action_sd is None else np.diag(np.asarray(action_sd[c]) ** 2)
        covs, M = _log_cov_parts(world, c, K, Sa)
        log_cov = covs.mean(0)
        v = np.linalg.eigh(log_cov)[1][:, -1]
        num, den = _acf_closed(covs, M, v)
        tau, tau_base, crossed = iact(num / den)
        level = np.sqrt(np.einsum("tii->t", covs))
        sd = np.sqrt(np.einsum("tii->ti", covs))
        tail = 2 * stats.norm.sf(SCALE_BOUND / sd)
        out[CONTEXTS[c]] = dict(participation_ratio=participation_ratio(log_cov),
                                cov_eigs=np.linalg.eigvalsh(log_cov).tolist(), settle_step=settle_step(level),
                                transient_share=settle_step(level) / len(level), iact=tau, iact_baseline24=tau_base,
                                iact_crossed=crossed, level_first=float(level[0]), level_last10=float(level[-10:].mean()),
                                coord_share_outside_5=float(tail.mean()),
                                row_share_outside_5_union_bound=float(tail.sum(1).mean()),
                                coord_share_outside_3=float((2 * stats.norm.sf(3.0 / sd)).mean()))
        parts[c] = (covs, M)
        pooled_cov = pooled_cov + world.mix[c] * log_cov
        pooled_level2 = pooled_level2 + world.mix[c] * np.einsum("tii->t", covs)
    v = np.linalg.eigh(pooled_cov)[1][:, -1]
    nums = [world.mix[c] * _acf_closed(*parts[c], v)[0] for c in range(NC)]
    acf = np.sum(nums, 0) / np.sum(nums, 0)[0]
    tau, tau_base, crossed = iact(acf)
    level = np.sqrt(pooled_level2)
    bound = float(sum(world.mix[c] * out[CONTEXTS[c]]["row_share_outside_5_union_bound"] for c in range(NC)))
    out["pooled"] = dict(participation_ratio=participation_ratio(pooled_cov), settle_step=settle_step(level),
                         transient_share=settle_step(level) / len(level), iact=tau, iact_baseline24=tau_base,
                         iact_crossed=crossed, row_share_outside_5_union_bound=bound)
    marks = dict(
        participation_ratio=all(out[n]["participation_ratio"] >= PR_MIN for n in CONTEXTS),
        transient_share=out["pooled"]["transient_share"] >= TRANSIENT_MIN,
        iact=IACT_RANGE[0] <= out["pooled"]["iact"] <= IACT_RANGE[1],
        scale=bound <= 1 - SCALE_SHARE)
    return out, marks


def box_probability(cov, seed=0):
    """P(every coordinate of N(0, cov) lies in [-1, 1]); coordinates with zero variance are exactly 0 (inside)."""
    cov = np.asarray(cov, np.float64)
    keep = np.flatnonzero(np.diag(cov) > 0)
    if keep.size == 0:
        return 1.0
    sub = cov[np.ix_(keep, keep)]
    if keep.size == 1:
        return float(1 - 2 * stats.norm.sf(1 / math.sqrt(sub[0, 0])))
    dist = stats.multivariate_normal(np.zeros(keep.size), sub, seed=seed)
    return float(dist.cdf(np.ones(keep.size), lower_limit=-np.ones(keep.size)))


def action_box_shares(world, K, action_cov_fn, seed=0):
    """Per context: the mean over the 100 steps of P(a in [-1, 1]^4) and of the per-coordinate share, for a = -K_c s
    + N(0, action_cov_fn(c)) along its own closed loop from s0 ~ N(0, I)."""
    K = world.per_context(K)
    joint, coord = [], []
    for c in range(NC):
        Sa = action_cov_fn(c)
        covs = world.state_covariances(K, c, Sa)
        acovs = [K[c] @ S @ K[c].T + Sa for S in covs]
        joint.append(float(np.mean([box_probability(C, seed) for C in acovs])))
        sd = np.sqrt(np.maximum(np.stack([np.diag(C) for C in acovs]), 0))
        inside = np.where(sd > 0, 1 - 2 * stats.norm.sf(1 / np.where(sd > 0, sd, 1)), 1.0)
        coord.append(float(inside.mean()))
    return np.array(joint), np.array(coord)


def behaviour_rows(world, n, rng):
    """n rows per context from the exact behaviour state-action distribution (t uniform on 0..L-1, s ~ N(0,
    Sigma_t), a = -K_beh s + N(0, Sigma_a))."""
    out = []
    for c in range(NC):
        covs = world.state_covariances(world.k_behaviour, c, world.action_cov(c))
        t = rng.integers(len(covs), size=n)
        chol = np.stack([np.linalg.cholesky(S + 1e-300 * np.eye(DS)) for S in covs])
        s = np.einsum("nij,nj->ni", chol[t], rng.standard_normal((n, DS)))
        a = -s @ world.k_behaviour[c].T + rng.standard_normal((n, DA)) * world.behaviour_sd[c]
        out.append((s, a, np.full(n, c)))
    s, a, ctx = (np.concatenate(x) for x in zip(*out))
    return s, a, ctx


# ---------------------------------------------------------------------------------------------------------
# Stage 0


def _finite(x):
    return float(x) if math.isfinite(x) else None


def stage0_cheap(spec):
    """S1-S3, S5 (R0 closed form) and the stability of every policy used: everything but sampling and box integrals."""
    w = World(spec)
    zero = np.zeros((NC, DA, DS))
    gains = dict(zero=zero, behaviour=w.k_behaviour, k_star=w.k_star, k_two=w.k_two, expert=w.k_star.copy())
    J = {name: w.j_finite(K) for name, K in gains.items()}
    J_inf = {name: w.j_inf(K) for name, K in gains.items()}
    radius = {name: w.radius(K) for name, K in gains.items()}
    s1 = {CONTEXTS[c]: float((J["behaviour"][c] - J["zero"][c]) / (J["k_star"][c] - J["zero"][c])) for c in RICH}
    s1_inf = {CONTEXTS[c]: float((J_inf["behaviour"][c] - J_inf["zero"][c]) / (J_inf["k_star"][c] - J_inf["zero"][c]))
              for c in RICH}
    gap = J["k_star"] - J["behaviour"]
    total_gap = float(w.mix @ gap)
    g = {CONTEXTS[c]: float(J["k_star"][c] - J["k_two"][c]) for c in THIN}
    thin_w = float(w.mix[T_WRONG] + w.mix[T_RIGHT])
    g_r = g["T_right"]
    sys_w, sys_r = w.systems[T_WRONG], w.systems[T_RIGHT]
    transplants = {}
    for name, block in (("full", False), ("block", True)):
        K = w.transplant(block)
        rad = discounted_radius(sys_w, K)
        j_w, j_r = finite_return(sys_w, K), finite_return(sys_r, K)
        transplants[name] = dict(
            radius_T_wrong=rad, stable=bool(rad < 1), J_T_wrong=j_w, J_T_wrong_inf=_finite(LQ.value(sys_w, K)),
            loss_T_wrong=float(J["k_two"][T_WRONG] - j_w), loss_over_g=float((J["k_two"][T_WRONG] - j_w) / g_r),
            gain_T_right=float(j_r - J["k_two"][T_RIGHT]),
            loss_over_gain=float((J["k_two"][T_WRONG] - j_w) / (j_r - J["k_two"][T_RIGHT])))
    full = transplants["full"]
    stable_all = all(bool(np.all(r < 1)) for r in radius.values()) and all(t["stable"] for t in transplants.values())
    r0, marks = r0_closed_form(w)
    pooled = r0["pooled"]
    min_pr = min(r0[n]["participation_ratio"] for n in CONTEXTS)
    targets = dict(S1=max(s1.values()) <= S1_MAX, S2=thin_w * g_r >= S2_MIN * total_gap,
                   S3=full["stable"] and full["loss_over_g"] >= S3_MIN, stability=stable_all, S5=all(marks.values()))
    margins = dict(
        S1=(S1_MAX - max(s1.values())) / S1_MAX, S2=(thin_w * g_r / total_gap - S2_MIN) / S2_MIN,
        S3=(full["loss_over_g"] - S3_MIN) / S3_MIN if full["stable"] else -math.inf,
        S5_participation=(min_pr - PR_MIN) / PR_MIN,
        S5_transient=(pooled["transient_share"] - TRANSIENT_MIN) / TRANSIENT_MIN,
        S5_iact=min((pooled["iact"] - IACT_RANGE[0]) / IACT_RANGE[0], (IACT_RANGE[1] - pooled["iact"]) / IACT_RANGE[1]),
        S5_scale=((1 - SCALE_SHARE) - pooled["row_share_outside_5_union_bound"]) / (1 - SCALE_SHARE))
    return dict(
        spec=spec_dict(spec), spec_sha256=spec_digest(spec), rho=list(w.rho), coupling_attempt=w.coupling_attempt,
        spectral_radius_A=float(np.max(np.abs(np.linalg.eigvals(w.A)))),
        J={k: v.tolist() for k, v in J.items()}, J_inf={k: [_finite(x) for x in v] for k, v in J_inf.items()},
        radius={k: v.tolist() for k, v in radius.items()},
        S1_share=s1, S1_share_inf=s1_inf, S2=dict(g=g, total_gap=total_gap, gap_per_context=gap.tolist(),
                                                  thin_share_of_gap=thin_w * g_r / total_gap),
        S3=transplants, S5=dict(per_context=r0, marks=marks), targets=targets, margins=margins,
        cheap_pass=all(targets.values()))


def stage0_full(spec, rows=8192, seed=0):
    """stage0_cheap plus S4 (action boxes), the decoy target, S6 and the 100-step gradient at K* and the behaviour."""
    out = stage0_cheap(spec)
    w = World(spec)
    beh_joint, beh_coord = action_box_shares(w, w.k_behaviour, w.action_cov, seed)
    opt_joint, opt_coord = action_box_shares(w, w.k_star, lambda c: np.zeros((DA, DA)), seed)
    out["S4"] = dict(logged_joint=beh_joint.tolist(), logged_coord=beh_coord.tolist(),
                     optimal_joint=opt_joint.tolist(), optimal_coord=opt_coord.tolist(),
                     logged_joint_mix=float(w.mix @ beh_joint), optimal_joint_mix=float(w.mix @ opt_joint))
    s, a, ctx = behaviour_rows(w, rows, np.random.default_rng([seed, 11]))
    sd_bc = w.aleatoric_sd(w.k_behaviour, s, a, ctx)
    sd_opt = w.aleatoric_sd(w.k_star, s, a, ctx)
    rich = np.isin(ctx, RICH_MAIN)
    med = lambda x, m: float(np.median(x[m]))
    out["decoy"] = dict(
        median_sd_bc={CONTEXTS[c]: med(sd_bc, ctx == c) for c in range(NC)},
        ratio_bc=med(sd_bc, ctx == DECOY) / med(sd_bc, rich),
        median_sd_kstar={CONTEXTS[c]: med(sd_opt, ctx == c) for c in range(NC)},
        ratio_kstar=med(sd_opt, ctx == DECOY) / med(sd_opt, rich), rows_per_context=rows)
    thin = np.isin(ctx, THIN)
    off = {name: np.linalg.norm(s[thin] @ w.transplant(block)[2:].T, axis=1)
           for name, block in (("full", False), ("block", True))}
    out["S6"] = dict(share_full=float(np.mean(off["full"] >= spec.rich_noise_sd)),
                     share_block=float(np.mean(off["block"] >= spec.rich_noise_sd)),
                     median_full=float(np.median(off["full"])), threshold=spec.rich_noise_sd,
                     thin_rows=int(thin.sum()))
    grads = {}
    for c in range(NC):
        fd = lambda K: finite_gradient(w.systems[c], K)
        grads[CONTEXTS[c]] = dict(max_abs_grad_J100_at_kstar=float(np.abs(fd(w.k_star[c])).max()),
                                  max_abs_grad_J100_at_behaviour=float(np.abs(fd(w.k_behaviour[c])).max()))
    out["finite_horizon_gradient"] = grads
    s4 = min(out["S4"]["logged_joint_mix"], out["S4"]["optimal_joint_mix"])
    out["targets"].update(S4=s4 >= S4_MIN, decoy=out["decoy"]["ratio_bc"] >= DECOY_RATIO_MIN,
                          S6=out["S6"]["share_full"] >= S6_SHARE_MIN)
    out["margins"].update(S4=(s4 - S4_MIN) / (1 - S4_MIN),
                          decoy=(out["decoy"]["ratio_bc"] - DECOY_RATIO_MIN) / DECOY_RATIO_MIN,
                          S6=(out["S6"]["share_full"] - S6_SHARE_MIN) / S6_SHARE_MIN)
    out["full_pass"] = all(out["targets"].values())
    out["full_pass_step_a"] = all(v for k, v in out["targets"].items() if k != "S6")
    return out


def finite_gradient(system, K, step=1e-6):
    """dJ_100/dK by central differences of finite_return."""
    grad = np.zeros_like(K, np.float64)
    for idx in np.ndindex(K.shape):
        d = np.zeros_like(grad)
        d[idx] = step
        grad[idx] = (finite_return(system, K + d) - finite_return(system, K - d)) / (2 * step)
    return grad


def stage0_candidates(base=DESIGN_SPEC, grid=None):
    grid = STAGE0_GRID if grid is None else grid
    keys = list(grid)
    return [replace(base, **dict(zip(keys, values))) for values in itertools.product(*(grid[k] for k in keys))]


def _summary(result):
    """The compact record of a candidate (strict-JSON friendly)."""
    keep = ("spec", "spec_sha256", "rho", "coupling_attempt", "S1_share", "targets", "margins", "cheap_pass",
            "full_pass", "full_pass_step_a", "S4", "S6")
    out = {k: result[k] for k in keep if k in result}
    out["S2_thin_share_of_gap"] = result["S2"]["thin_share_of_gap"]
    out["S3_loss_over_g"] = result["S3"]["full"]["loss_over_g"]
    out["S3_radius"] = result["S3"]["full"]["radius_T_wrong"]
    out["S5_pooled"] = {k: v for k, v in result["S5"]["per_context"]["pooled"].items()}
    return out


def _robust_score(spec, full, alt_seeds, margin_keys):
    """(passes under every alternative coupling, worst margin over `margin_keys` of the full result and the
    alternatives' cheap margins) and the alternatives' summaries."""
    alts = [stage0_cheap(replace(spec, coupling_seed=spec.coupling_seed + k)) for k in alt_seeds]
    worst = min([full["margins"][k] for k in margin_keys] + [min(a["margins"].values()) for a in alts])
    summary = [dict(offset=k, cheap_pass=a["cheap_pass"], margins=a["margins"], S1=max(a["S1_share"].values()),
                    S2=a["S2"]["thin_share_of_gap"], S3=a["S3"]["full"]["loss_over_g"],
                    transient=a["S5"]["per_context"]["pooled"]["transient_share"]) for k, a in zip(alt_seeds, alts)]
    return all(a["cheap_pass"] for a in alts), worst, summary


def stage0_search(base=DESIGN_SPEC, grid=None, gains=STAGE0_GAINS, alt_seeds=ALT_COUPLING_SEEDS, log=None):
    """The Stage-0 selection (module docstring, STAGE 0), in two steps.

    Step A (returns structure, at base.gain): every candidate of the grid gets the cheap targets; every cheap pass
    gets the full targets except S6 and, if it passes, the cheap targets under each alternative coupling draw
    (coupling_seed + offset). Candidates are taken in WRONG_MAPS order; within the first map with any pass, the
    winner maximises (passes under every alternative coupling, worst normalised margin, -action_cost,
    behaviour_fraction). Step B (action scale): the winner's gain is replaced by each of `gains` with action_cost x
    (gain / base.gain)^2, which leaves every deterministic return unchanged and rescales actions by base.gain / gain;
    among the gains that pass every target including S6, the winner maximises (passes under every alternative
    coupling, worst margin over the scale-dependent targets SCALE_TARGETS = S4 and S6, gain).
    Returns (chosen spec or None, dict(step_a=records, step_b=records))."""
    candidates = stage0_candidates(base, grid)
    results = [stage0_cheap(spec) for spec in candidates]
    records = [dict(stage="cheap", **_summary(r)) for r in results]
    chosen_a = None
    for wm in WRONG_MAPS:
        scored = []
        for i, spec in enumerate(candidates):
            if spec.wrong_map != wm or not results[i]["cheap_pass"]:
                continue
            full = stage0_full(spec)
            records[i] = dict(stage="full", **_summary(full))
            if not full["full_pass_step_a"]:
                continue
            keys = [k for k in full["margins"] if k != "S6"]
            robust, worst, alts = _robust_score(spec, full, alt_seeds, keys)
            records[i].update(coupling_alternatives=alts, robust=robust, worst_margin=worst)
            scored.append((robust, worst, -spec.action_cost, spec.behaviour_fraction, i))
            if log:
                log(f"step A pass {wm} rho_top={spec.rho_top} rho_step={spec.rho_step} R={spec.action_cost} "
                    f"f={spec.behaviour_fraction}: robust={robust} worst margin={worst:.3f}")
        if scored:
            chosen_a = candidates[max(scored)[-1]]
            break
    if chosen_a is None:
        return None, dict(step_a=records, step_b=[])
    step_b, scored = [], []
    for gain in gains:
        spec = replace(chosen_a, gain=gain, action_cost=chosen_a.action_cost * (gain / chosen_a.gain) ** 2)
        full = stage0_full(spec)
        record = dict(stage="full", **_summary(full))
        if full["full_pass"]:
            robust, worst, alts = _robust_score(spec, full, alt_seeds, list(full["margins"]))
            scale = min(full["margins"][k] for k in SCALE_TARGETS)
            record.update(coupling_alternatives=alts, robust=robust, worst_margin=worst, worst_scale_margin=scale)
            scored.append((robust, scale, gain, len(step_b)))
            if log:
                log(f"step B pass gain={gain} R={spec.action_cost:.4f}: robust={robust} worst margin={worst:.3f} "
                    f"worst S4/S6 margin={scale:.3f}")
        step_b.append(record)
    if not scored:
        return None, dict(step_a=records, step_b=step_b)
    best = max(scored)
    chosen = replace(chosen_a, gain=gains[best[-1]],
                     action_cost=chosen_a.action_cost * (gains[best[-1]] / chosen_a.gain) ** 2)
    return chosen, dict(step_a=records, step_b=step_b)


# ---------------------------------------------------------------------------------------------------------
# Data


def context_counts(n_episodes, mix):
    """Exact per-context episode counts: largest-remainder rounding of n mix_c."""
    raw = np.asarray(mix, np.float64) * n_episodes
    counts = np.floor(raw).astype(np.int64)
    rest = n_episodes - counts.sum()
    counts[np.argsort(-(raw - counts), kind="stable")[:rest]] += 1
    return counts


def draw_contexts(world, sizes, rng, mode="stratified"):
    """Per-episode contexts of the three splits (concatenated)."""
    if mode == "stratified":
        return np.concatenate([rng.permutation(np.repeat(np.arange(NC), context_counts(n, world.mix)))
                               for n in sizes])
    if mode == "categorical":
        return rng.choice(NC, size=int(sum(sizes)), p=world.mix)
    raise ValueError("context draw must be 'stratified' or 'categorical'")


def context_code(context, code_sd=None, rng=None):
    """Per-episode code: onehot(c), or onehot(c) + code_sd xi (g7; xi ~ N(0, I_5) per episode)."""
    code = np.eye(NC)[np.asarray(context, np.int64)]
    if code_sd is None:
        return code
    return code + code_sd * rng.standard_normal(code.shape)


def simulate_episodes(world, context, gains, action_sd, s0, noise, shock):
    """Rollouts of a = -K_e s + action_sd_e * n_t in each episode's context; noise (E, L, 4) and shock (E, L, 10)
    are standard normal draws. Returns arrays (E, L, .): state, action, reward, next_state."""
    E, L = noise.shape[:2]
    B = world.B[context]
    sd = world.noise_sd[context][:, None]
    s, out = s0, {k: [] for k in ("state", "action", "reward", "next_state")}
    for t in range(L):
        a = -np.einsum("eij,ej->ei", gains, s) + action_sd * noise[:, t]
        s2 = s @ world.A.T + np.einsum("eij,ej->ei", B, a) + sd * shock[:, t]
        r = -(np.sum(s * s, 1) + world.spec.action_cost * np.sum(a * a, 1))
        for k, x in zip(out, (s, a, r, s2)):
            out[k].append(x)
        s = s2
    return {k: np.stack(v, 1) for k, v in out.items()}


def generate(world, rep, seed=DATA_SEED, sizes=(500, 200, 300), rows_per_episode=5, rung=None, code_sd=None,
             context_draw="stratified", fraction=None):
    """Train, calibration-bank and evaluation rows of one seed (float64), plus meta.

    rung overrides the thin contexts' actuator-3-4 SD (the density ladder), fraction the rich behaviour fraction (the
    null variant uses 1.0); neither changes any random draw. code_sd switches on g7's noisy context code.
    """
    spec = world.spec
    if rung is not None or fraction is not None:
        world = World(replace(spec, thin_rung=spec.thin_rung if rung is None else float(rung),
                              behaviour_fraction=spec.behaviour_fraction if fraction is None else float(fraction)))
    rng = lambda k: np.random.default_rng([seed, rep, TAG_DATA, k])
    E, L = int(sum(sizes)), world.spec.length
    context = draw_contexts(world, sizes, rng(0), context_draw)
    s0 = rng(1).standard_normal((E, DS)) * world.spec.init_sd
    noise = rng(2).standard_normal((E, L, DA))
    shock = rng(3).standard_normal((E, L, DS))
    code = context_code(context, code_sd, rng(6) if code_sd is not None else None)
    ep = simulate_episodes(world, context, world.k_behaviour[context], world.behaviour_sd[context], s0, noise, shock)
    flat = lambda x: x.reshape(E * L, *x.shape[2:])
    codes = np.repeat(code, L, 0)
    rows = dict(state=flat(ep["state"]), action=flat(ep["action"]), reward=flat(ep["reward"]),
                next_state=flat(ep["next_state"]), noise=flat(noise), shock=flat(shock),
                context=np.repeat(context, L), episode=np.repeat(np.arange(E), L), t=np.tile(np.arange(L), E))
    rows["obs"] = np.concatenate([rows["state"], codes], 1)
    rows["next_obs"] = np.concatenate([rows["next_state"], codes], 1)
    edges = np.cumsum((0,) + tuple(sizes))
    episodes, offsets = bank.stratified_bank(np.full(sizes[1], L), sizes[1] * rows_per_episode, rows_per_episode,
                                             rng(5))
    if not np.array_equal(np.sort(episodes), np.arange(sizes[1])):
        raise AssertionError("the bank must take rows from every calibration episode")
    cal = np.sort(np.concatenate([(edges[1] + e) * L + o for e, o in zip(episodes, offsets)]))
    index = dict(train=np.arange(edges[0] * L, edges[1] * L), cal=cal, eval=np.arange(edges[2] * L, edges[3] * L))
    data = {split: {k: x[idx] for k, x in rows.items()} for split, idx in index.items()}
    data["meta"] = dict(rep=rep, seed=seed, sizes=list(sizes), rows_per_episode=rows_per_episode,
                        rung=world.spec.thin_rung, behaviour_fraction=world.spec.behaviour_fraction, code_sd=code_sd,
                        context_draw=context_draw, spec_sha256=spec_digest(world.spec),
                        context_episodes={s: np.bincount(context[edges[i]:edges[i + 1]], minlength=NC).tolist()
                                          for i, s in enumerate(SPLITS)},
                        context_rows={s: np.bincount(data[s]["context"], minlength=NC).tolist() for s in SPLITS})
    return data


def transitions(rows):
    """The float32 boundary: runtime.networks.Transition(obs, action, reward, next_obs, done = 0)."""
    return LQ.Harness.transitions(rows, False)


def rollout_returns(world, gains, context, s0, shock, action_noise=None, horizon=None):
    """Discounted returns sum_{t < horizon} gamma^t r_t of per-episode linear gains (common random numbers: the
    caller passes s0 (E, 10) and shock (E, T, 10) standard normal draws; action_noise (E, T, 4) already scaled)."""
    T = shock.shape[1] if horizon is None else horizon
    B, sd = world.B[context], world.noise_sd[context][:, None]
    s, total = s0, np.zeros(len(s0))
    for t in range(T):
        a = -np.einsum("eij,ej->ei", gains, s)
        if action_noise is not None:
            a = a + action_noise[:, t]
        total += world.spec.gamma ** t * -(np.sum(s * s, 1) + world.spec.action_cost * np.sum(a * a, 1))
        s = s @ world.A.T + np.einsum("eij,ej->ei", B, a) + sd * shock[:, t]
    return total, s


# ---------------------------------------------------------------------------------------------------------
# R0 and R1 log-only metrics


def _r0_rows(states, t, episode, L):
    """R0 marks of one set of rows (whole episodes, rows in episode-major order)."""
    E = len(np.unique(episode))
    s = states.reshape(E, L, DS)
    cov = np.cov(states.T)
    level = np.sqrt((s ** 2).sum(-1)).mean(0)
    rms = np.sqrt((s ** 2).sum(-1).mean(0))
    v = np.linalg.eigh(cov)[1][:, -1]
    x = s[:, IACT_T0:, :] @ v
    x = x - x.mean()
    acf = [1.0] + [float((x[:, :-k] * x[:, k:]).mean() / (x * x).mean()) for k in range(1, IACT_MAX_LAG + 1)]
    tau, tau_base, crossed = iact(acf)
    absmax = np.abs(states).max(1)
    return dict(participation_ratio=participation_ratio(cov), cov_eigs=np.linalg.eigvalsh(cov).tolist(),
                settle_step=settle_step(level), transient_share=settle_step(level) / L,
                settle_step_rms=settle_step(rms), transient_share_rms=settle_step(rms) / L, iact=tau,
                iact_baseline24=tau_base, iact_crossed=crossed, row_share_within_5=float(np.mean(absmax <= 5)),
                row_share_within_3=float(np.mean(absmax <= 3)), episodes=int(E))


def log_metrics(world, rows, rung):
    """R0 (state richness) and R1 (a)(b) (thinness) on one split's rows (whole episodes)."""
    L = world.spec.length
    ctx = rows["context"]
    r0 = {CONTEXTS[c]: _r0_rows(rows["state"][ctx == c], rows["t"][ctx == c], rows["episode"][ctx == c], L)
          for c in range(NC)}
    r0["pooled"] = _r0_rows(rows["state"], rows["t"], rows["episode"], L)
    r1 = {}
    for c in range(NC):
        a = rows["action"][ctx == c]
        eig = np.linalg.eigvalsh(np.cov(a.T))
        rank = int(np.sum(eig > THIN_EIG_REL * eig.max()))
        resid = a - (-rows["state"][ctx == c] @ world.k_behaviour[c].T)
        r1[CONTEXTS[c]] = dict(action_cov_eigs=eig.tolist(), rank=rank,
                               action_resid_sd=resid.std(0).tolist(), rows=int((ctx == c).sum()),
                               actuator34_exact_zero=bool(np.all(a[:, 2:] == 0)))
    thin_share = float(np.isin(ctx, THIN).mean())
    rich_ok = all(min(r1[CONTEXTS[c]]["action_cov_eigs"]) >= RICH_EIG_MIN for c in RICH)
    if rung == 0:
        thin_ok = all(sorted(r1[CONTEXTS[c]]["action_cov_eigs"])[1] <= THIN_EIG_REL * max(
            r1[CONTEXTS[c]]["action_cov_eigs"]) and r1[CONTEXTS[c]]["actuator34_exact_zero"] for c in THIN)
    else:
        thin_ok = all(abs(sd / rung - 1) <= THIN_SD_TOL for c in THIN for sd in r1[CONTEXTS[c]]["action_resid_sd"][2:])
    marks = dict(
        R0_participation_ratio=all(r0[CONTEXTS[c]]["participation_ratio"] >= PR_MIN for c in range(NC)),
        R0_transient_share=r0["pooled"]["transient_share"] >= TRANSIENT_MIN,
        R0_iact=IACT_RANGE[0] <= r0["pooled"]["iact"] <= IACT_RANGE[1],
        R0_scale=r0["pooled"]["row_share_within_5"] >= SCALE_SHARE,
        R1a_rich_eigs=rich_ok, R1a_thin_rung=thin_ok, R1b_thin_share=abs(thin_share - THIN_SHARE) <= THIN_SHARE_TOL)
    return dict(R0=r0, R1=r1, thin_share=thin_share, marks=marks)


def log_check(world, seeds, seed=DATA_SEED, rungs=LADDER, log=None):
    """R0/R1 log-only metrics on the training split for every seed; R0 at the world's rung, R1 at every rung."""
    out = []
    for rep in seeds:
        record = dict(rep=rep)
        for rung in rungs:
            data = generate(world, rep, seed, rung=rung)
            m = log_metrics(World(replace(world.spec, thin_rung=rung)), data["train"], rung)
            m["context_episodes"] = data["meta"]["context_episodes"]
            record[f"rung={rung}"] = m
            if log:
                log(f"seed {rep} rung {rung}: {m['marks']}")
        out.append(record)
    return out


# ---------------------------------------------------------------------------------------------------------
# CLI


def _write(path, payload):
    path = Path(path)
    text = json.dumps(payload, indent=1, allow_nan=False, default=_json_default)
    path.write_text(text, encoding="utf8")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def _json_default(x):
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def _clean(x):
    """-inf / inf / nan -> None, recursively (strict JSON)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def run_stage0(output, log=print):
    started = time.perf_counter()
    chosen, records = stage0_search(log=log)
    if chosen is None:
        raise SystemExit("Stage 0: no candidate in STAGE0_GRID passes every target")
    if chosen != STAGE0_SPEC:
        raise SystemExit(f"Stage 0 selects {chosen}, not STAGE0_SPEC: update STAGE0_SPEC deliberately")
    world = World(chosen)
    design = stage0_full(DESIGN_SPEC)
    payload = dict(
        kind="glq-stage0", chosen=spec_dict(chosen), chosen_sha256=spec_digest(chosen),
        chosen_result=stage0_full(chosen), world=world.describe(), design_defaults_result=design,
        grid={k: list(v) for k, v in STAGE0_GRID.items()},
        selection="step A (gain 0.3): negate before rotate135; within the first map with a pass (every target but "
                  "S6), maximise (passes under every alternative coupling draw, worst normalised margin, smaller "
                  "action_cost, larger behaviour_fraction). Step B: gain over STAGE0_GAINS with R x (gain / 0.3)^2; "
                  "among robust passes of every target, maximise the worst S4/S6 margin (ties: larger gain)",
        alt_coupling_offsets=list(ALT_COUPLING_SEEDS),
        step_a=records["step_a"], step_b=records["step_b"], gains=list(STAGE0_GAINS),
        n_candidates_step_a=len(records["step_a"]), n_cheap_pass_step_a=sum(r["cheap_pass"] for r in records["step_a"]),
        argv=sys.argv, seconds=time.perf_counter() - started, numpy=np.__version__)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    digest = _write(output / "stage0.json", _clean(payload))
    log(f"stage0.json sha256 {digest}")
    return payload


def run_logcheck(output, seeds, log=print):
    started = time.perf_counter()
    world = World(STAGE0_SPEC)
    records = log_check(world, seeds, log=log)
    primary = f"rung={STAGE0_SPEC.thin_rung}"
    summary = {name: [r[primary]["marks"][name] for r in records] for name in records[0][primary]["marks"]}
    ladder = {f"rung={rung}": [r[f"rung={rung}"]["marks"]["R1a_thin_rung"] for r in records] for rung in LADDER}
    payload = dict(kind="glq-logcheck", spec=spec_dict(STAGE0_SPEC), spec_sha256=spec_digest(STAGE0_SPEC),
                   seeds=list(seeds), data_seed=DATA_SEED, primary_rung=STAGE0_SPEC.thin_rung,
                   marks_by_seed=summary, thin_rung_by_seed=ladder, records=records, argv=sys.argv,
                   seconds=time.perf_counter() - started)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    digest = _write(output / "logcheck.json", _clean(payload))
    log(f"logcheck.json sha256 {digest}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p0 = sub.add_parser("stage0", help="closed-form Stage 0 search; writes stage0.json")
    p0.add_argument("--output", required=True)
    p1 = sub.add_parser("logcheck", help="R0/R1 log-only checks; writes logcheck.json")
    p1.add_argument("--output", required=True)
    p1.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ns = parser.parse_args(argv)
    if ns.command == "stage0":
        run_stage0(ns.output)
    else:
        run_logcheck(ns.output, ns.seeds)


if __name__ == "__main__":
    main()
