"""The known shift: current laws and their exact densities, the recording controller, the removal and the exact
weights. No learning and no BCA code anywhere in this module.

The world is v1's ThrusterChain in the stalling world, imported read-only from experiments/shift (physics, dynamics,
env.step, the textbook gain). Only how often each current strength c appears changes between training and test.

Current laws on [0.1, 1.1] (truncated exponentials):
  p_lam(c) = exp(lam c) / M(lam),   M(lam) = (exp(1.1 lam) - exp(0.1 lam)) / lam,   M(0) = 1.
  training law, calibration bank and the "none" test law: lam = -2 (mean c 0.443)
  "mild" test law: lam = 0, uniform (mean c 0.600);   "strong" test law: lam = +2 (mean c 0.757)
Given c, everything else is identical in training and test: float starts 0.5 N(0, I10), wave amplitude U[0.1, 0.4],
wave phase U[0, 2 pi), the recording controller and the physics. So the exact likelihood ratio of a recorded step is
  w(c) = p_test(c) / p_train(c) = (M(-2) / M(lam)) exp((lam + 2) c).
Starts are drawn in thruster_chain.draw_starts's order (floats, current, amplitude, phase); the current is the law's
inverse CDF of one uniform, so under the uniform law the starts equal thruster_chain.draw_starts bit for bit, and two
laws drawn from one seed share every float, wave and noise number (only c differs).

Recording controller (HoldRecorder): v1's normal driving with v1's bad full-throttle command replaced by a good one.
  normal driving: command = 0.8 tanh(u / 0.8), u = -(0.25 K_s s + K_e e) + 0.3 n, n ~ N(0, I4) fresh each step
                  (K the textbook gain, s the floats, e = (c, w1, w2) the water); it never reaches 80%.
  hard water:     thruster 1's hold demand (K_e e)_1 = 0.910 c + 0.822 w1 - 0.320 w2 above 0.65.
  80% hold:       in hard water a fair coin each step picks the hold: thruster 1 at exactly -0.8 (rounded to float32
                  it is -0.800000011920929, the knee, so it does not stall) and thrusters 2-4 at the hold pattern
                  scaled to that point, -0.8 K[:, c] / K[0, c] = (-0.110, -0.468, -0.325), plus 0.03 n.
Commands are rounded to float32 before they reach the world, as env.step does.

The removal (the action gap): v1's densest-15% ball (experiments/truth/gap.find_gap with shift.gap_settings(), on the
float64 of the float32 training-split commands; experiments.shift.shift.define_gap does exactly this). It is cut from
training and calibration-bank rows only. Because the 80% hold repeats exactly, the ball removes almost every hold.

Exact density ratio of a test step to a kept training row:
  Z w(c) for a command outside the ball, +inf inside it, with Z = P_train(command outside the ball).
Z multiplies every bank weight and the test mass alike, so it cancels in wbcp.calibrate; the infinite ratio inside
the ball is a refusal ("cannot certify"), decided before calibrate is called. Z, the in-ball share under each law and
the test mass (the mean of w over test-law steps outside the ball) have no closed form: quadrature() computes them as
one-dimensional integrals over c of per-current curves measured by simulation (common starts at every grid point).
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

import experiments.shift  # noqa: F401  (dependency path and version checks; must come before gymnasium)
from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC
from experiments.truth import gap as GAP

MASTER_SEED = 20261016
C_LO, C_HI = 0.1, 1.1
WAVE = (0.1, 0.4)
KICK = 0.5
LAMBDA_TRAIN = -2.0
TILTS = {"none": -2.0, "mild": 0.0, "strong": 2.0}
TILT_NAMES = ("none", "mild", "strong")
PHYSICS = TC.STALL
GAMMA = PHYSICS.gamma
LENGTH = PHYSICS.length
KNEE32 = np.float32(0.8)  # float32(0.8) = 0.800000011920929 = thruster_chain.KNEE


# ------------------------------------------------------------------------------------------------ current laws

def law_lambda(law):
    """The exponent of a law given by name ('none', 'mild', 'strong', 'train') or as a number."""
    if isinstance(law, str):
        return LAMBDA_TRAIN if law == "train" else TILTS[law]
    return float(law)


def normaliser(lam):
    """M(lam) = integral of exp(lam c) over [0.1, 1.1]; M(0) = 1."""
    lam = float(lam)
    if lam == 0.0:
        return C_HI - C_LO
    return math.exp(lam * C_LO) * math.expm1(lam * (C_HI - C_LO)) / lam


def density(c, law):
    """p_lam(c) on [0.1, 1.1], 0 outside."""
    lam = law_lambda(law)
    c = np.asarray(c, np.float64)
    inside = (c >= C_LO) & (c <= C_HI)
    return np.where(inside, np.exp(lam * c) / normaliser(lam), 0.0)


def cdf(c, law):
    lam = law_lambda(law)
    c = np.clip(np.asarray(c, np.float64), C_LO, C_HI)
    if lam == 0.0:
        return (c - C_LO) / (C_HI - C_LO)
    return np.expm1(lam * (c - C_LO)) / math.expm1(lam * (C_HI - C_LO))


def inverse_cdf(u, law):
    """The law's quantile function. Under the uniform law it is C_LO + (C_HI - C_LO) u, the same arithmetic as
    numpy's Generator.uniform(C_LO, C_HI)."""
    lam = law_lambda(law)
    u = np.asarray(u, np.float64)
    if lam == 0.0:
        return C_LO + (C_HI - C_LO) * u
    c = C_LO + np.log1p(u * math.expm1(lam * (C_HI - C_LO))) / lam
    return np.clip(c, C_LO, C_HI)


def mean_current(law):
    """E[c] in closed form."""
    lam = law_lambda(law)
    if lam == 0.0:
        return 0.5 * (C_LO + C_HI)
    f = lambda c: math.exp(lam * c) * (c / lam - 1.0 / lam ** 2)
    return (f(C_HI) - f(C_LO)) / normaliser(lam)


def ratio(c, tilt):
    """Exact w(c) = p_test(c) / p_train(c) of a tilt ('none', 'mild', 'strong'), closed form:
    (M(-2) / M(lam)) exp((lam + 2) c). It is 1 everywhere under 'none'."""
    lam = law_lambda(tilt)
    c = np.asarray(c, np.float64)
    if lam == LAMBDA_TRAIN:
        return np.ones_like(c)
    return (normaliser(LAMBDA_TRAIN) / normaliser(lam)) * np.exp((lam - LAMBDA_TRAIN) * c)


def ratio_constant(tilt):
    """The constant in front of exp(slope c): M(-2) / M(lam) (spec.json laws.tilts.*.weight_const)."""
    return normaliser(LAMBDA_TRAIN) / normaliser(law_lambda(tilt))


def ratio_slope(tilt):
    """d log w / dc = lam + 2: 0, 2 and 4 (the slope the estimated-weight classifier should find)."""
    return law_lambda(tilt) - LAMBDA_TRAIN


def full_test_mass(tilt):
    """No removal: the test mass E_test[w] = E_train[w^2] = M(-2) M(2 lam + 2) / M(lam)^2 (1, 1.381, 2.841)."""
    lam = law_lambda(tilt)
    return normaliser(LAMBDA_TRAIN) * normaliser(2 * lam - LAMBDA_TRAIN) / normaliser(lam) ** 2


def kish_fraction(tilt):
    """Kish effective size of a bank of independent rows drawn from the training law, as a share of its size:
    1 / E_train[w^2]."""
    return 1.0 / full_test_mass(tilt)


def exact_ratio(c, actions, gap, tilt, Z):
    """The exact density ratio of a test step to a kept training row: Z w(c) for a command outside the ball, +inf
    for a command inside it."""
    w = Z * ratio(c, tilt)
    return np.where(gap.contains(actions), np.inf, w)


# ------------------------------------------------------------------------------------------------ starts

def start_law():
    """The uniform start law on [0.1, 1.1] as a thruster_chain start law (checked against the start set)."""
    return TC.start_law(dict(current=(C_LO, C_HI), wave=WAVE, kick=KICK))


def draw_starts(n, rng, law):
    """n starts of a current law, in thruster_chain.draw_starts's draw order: floats 0.5 N(0, I10) (n x 10), one
    uniform per start mapped to the current by the law's inverse CDF, wave amplitude U[0.1, 0.4], phase U[0, 2 pi)."""
    s = KICK * rng.standard_normal((n, TC.DS))
    c = inverse_cdf(rng.uniform(size=n), law)
    amp = rng.uniform(*WAVE, size=n)
    ph = rng.uniform(0.0, 2 * math.pi, size=n)
    x = np.concatenate([s, c[:, None], (amp * np.cos(ph))[:, None], (amp * np.sin(ph))[:, None]], 1)
    if not np.all(TC.in_start_set(x)):
        raise AssertionError("a start fell outside the environment's start set")
    return x


# ------------------------------------------------------------------------------------------------ recording controller

@dataclass(frozen=True)
class HoldSpec:
    """The recording controller's numbers (module docstring). trigger: the hold demand (K_e e)_1 above which the water
    is hard (0.65). hold_probability: the coin's chance of the hold per hard-water step (0.5). hold_thruster1: the
    hold's thruster-1 command (-0.8: the 80% limit). jitter: SD of the noise on thrusters 2-4 of the hold (0.03).
    Note: experiments/shift/recorder.Recorder reads `trigger` as a push (c + w1) threshold; learners.prepare builds
    one only for a diagnostic, which data.prepare replaces with this controller's own hard-water flag."""

    damping: float = 0.25
    wobble: float = 0.3
    normal_driving_cap: float = 0.8
    hold_probability: float = 0.5
    hold_thruster1: float = -0.8
    jitter: float = 0.03
    trigger: float = 0.65

    def as_dict(self):
        return asdict(self)


HOLD_SPEC = HoldSpec()


class HoldRecorder:
    """The recording controller (module docstring). parts(x, noise, coin) -> (command float64, hard_water, hold):
    noise (..., 4) standard normal (None: no wobble, no jitter), coin (...,) uniform (None: never the hold). Commands
    are float64 here; callers round them to float32 as env.step does. __call__ returns the command only, so
    recorder.StreamedRecorder can drive it with fixed streams on a start bank."""

    def __init__(self, spec=HOLD_SPEC, physics=PHYSICS):
        self.spec = spec
        self.physics = physics
        self.K = np.asarray(RC.textbook_gain(physics), np.float64)
        hold = self.spec.hold_thruster1 * self.K[:, TC.CURRENT] / self.K[0, TC.CURRENT]
        hold[0] = self.spec.hold_thruster1
        self.hold = hold
        self.name = "recording controller (normal driving, 80% hold in hard water)"

    def hold_demand(self, x):
        """(K_e e)_1: what the textbook hold against the water alone asks of thruster 1."""
        _, ex = RC._linear(np.asarray(x, np.float64), self.K)
        return ex[..., 0]

    def hard_water(self, x):
        return self.hold_demand(x) > self.spec.trigger

    def parts(self, x, noise=None, coin=None):
        x = np.asarray(x, np.float64)
        fl, ex = RC._linear(x, self.K)
        u = -(self.spec.damping * fl + ex)
        if noise is not None:
            noise = np.asarray(noise, np.float64)
            u = u + self.spec.wobble * noise
        cap = self.spec.normal_driving_cap
        normal = cap * np.tanh(u / cap)
        hold = np.broadcast_to(self.hold, normal.shape).copy()
        if noise is not None:
            hold[..., 1:] = hold[..., 1:] + self.spec.jitter * noise[..., 1:]
        hold[..., 0] = self.spec.hold_thruster1
        region = ex[..., 0] > self.spec.trigger
        if coin is None:
            press = np.zeros(region.shape, bool)
        else:
            press = region & (np.asarray(coin, np.float64) < self.spec.hold_probability)
        return np.where(press[..., None], hold, normal), region, press

    def __call__(self, x, noise=None, coin=None):
        return self.parts(x, noise, coin)[0]

    def clean(self, x):
        """Normal driving without wobble, never the hold (recorder.hand_only's command)."""
        return self.parts(x)[0]


def round32(a):
    """A command as the world receives it through env.step: float32, then float64."""
    return np.asarray(a, np.float64).astype(np.float32).astype(np.float64)


def simulate(recorder, starts, noise, coin, physics=PHYSICS):
    """Episodes of the recording controller from given starts and streams (noise (E, L, 4), coin (E, L)), by
    thruster_chain.dynamics, the function env.step calls (its sums run in a fixed order, so a batch gives the numbers
    of single rows bit for bit). Returns states (E, L + 1, 13) float64, commands (E, L, 4) float32, rewards (E, L)
    float64, hard_water and hold (E, L)."""
    x = np.asarray(starts, np.float64).copy()
    L = noise.shape[1]
    S, C, R, G, P = [x], [], [], [], []
    for t in range(L):
        cmd, region, press = recorder.parts(x, noise[:, t], coin[:, t])
        a32 = cmd.astype(np.float32)
        x, r = TC.dynamics(x, np.clip(a32.astype(np.float64), -1.0, 1.0), physics)
        S.append(x), C.append(a32), R.append(r), G.append(region), P.append(press)
    st = lambda v: np.stack(v, 1)
    return dict(states=st(S), commands=st(C), rewards=st(R), hard_water=st(G), hold=st(P))


def draw_episodes(n, rng, law, recorder, length=LENGTH):
    """n episodes of a law from one generator, in this order: starts, then noise (n, L, 4), then coin (n, L)."""
    starts = draw_starts(n, rng, law)
    noise = rng.standard_normal((n, length, TC.DA))
    coin = rng.uniform(size=(n, length))
    return simulate(recorder, starts, noise, coin)


# ------------------------------------------------------------------------------------------------ quadrature over c

def simpson_weights(grid):
    grid = np.asarray(grid, np.float64)
    n = len(grid) - 1
    if n < 2 or n % 2:
        raise ValueError("Simpson's rule needs an odd number of at least 3 grid points")
    h = (grid[-1] - grid[0]) / n
    w = np.ones(n + 1)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return w * h / 3.0


def curves(gap, grid_points, per, seed, recorder=None):
    """Per-current curves by simulation: on each of `grid_points` currents spaced evenly over [0.1, 1.1], `per`
    recorder episodes from common base starts (the same floats and waves at every current), the share of each
    episode's steps whose command is inside the ball, is the 80% hold, is both, and is in hard water. Streams:
    default_rng(seed): base starts, then noise and coin per step. Returns the grid and arrays (grid_points, per)."""
    recorder = recorder or HoldRecorder()
    rng = np.random.default_rng(list(seed))
    grid = np.linspace(C_LO, C_HI, int(grid_points))
    base = draw_starts(int(per), rng, "mild")
    starts = np.tile(base, (len(grid), 1))
    starts[:, TC.CURRENT] = np.repeat(grid, per)
    n = len(starts)
    centre, rho = np.asarray(gap.centre, np.float64), float(gap.rho)
    acc = {k: np.zeros(n) for k in ("in_ball", "hold", "in_ball_and_hold", "hard_water")}
    x = starts
    for _ in range(LENGTH):
        noise = rng.standard_normal((n, TC.DA))
        coin = rng.uniform(size=n)
        cmd, region, press = recorder.parts(x, noise, coin)
        a = round32(cmd)
        ing = GAP.in_gap(a, centre, rho)
        acc["in_ball"] += ing
        acc["hold"] += press
        acc["in_ball_and_hold"] += ing & press
        acc["hard_water"] += region
        x, _ = TC.dynamics(x, np.clip(a, -1.0, 1.0))
    return grid, {k: (v / LENGTH).reshape(len(grid), per) for k, v in acc.items()}


def _integral(raw, grid, weight):
    """Per base start j: I_j = sum_i simpson_i weight(c_i) f_ij. Returns the per-start values (the curves share the
    base starts, so their spread counts the correlation across currents)."""
    return (simpson_weights(grid) * weight) @ raw


def _mean_se(v):
    v = np.asarray(v, np.float64)
    return dict(value=float(v.mean()), se=float(v.std(ddof=1) / math.sqrt(len(v))))


def _ratio_se(num, den):
    """Ratio of means with a delta-method standard error."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    r = num.mean() / den.mean()
    se = (num - r * den).std(ddof=1) / math.sqrt(len(num)) / abs(den.mean())
    return dict(value=float(r), se=float(se))


def quadrature(grid, curve):
    """The exact numbers of a ball as integrals over c of the per-current curves (module docstring):
    Z = P_train(command outside the ball); the in-ball share and hold share of steps under each law; the share of
    holds the ball removes under the training law; the gapped test mass E_tilt[w | command outside the ball] per
    tilt; and the closed-form full-data test mass. Each with its standard error over the common base starts."""
    out_ball = 1.0 - curve["in_ball"]
    p_train = density(grid, "train")
    res = dict(Z=_mean_se(_integral(out_ball, grid, p_train)),
               holds_removed_train=_ratio_se(_integral(curve["in_ball_and_hold"], grid, p_train),
                                             _integral(curve["hold"], grid, p_train)),
               in_ball_share={}, hold_share={}, hard_water_share={}, test_mass_gapped={}, test_mass_full={},
               weight_integral_train={})
    for tilt in TILT_NAMES:
        p = density(grid, tilt)
        res["in_ball_share"][tilt] = _mean_se(_integral(curve["in_ball"], grid, p))
        res["hold_share"][tilt] = _mean_se(_integral(curve["hold"], grid, p))
        res["hard_water_share"][tilt] = _mean_se(_integral(curve["hard_water"], grid, p))
        res["test_mass_gapped"][tilt] = _ratio_se(_integral(out_ball, grid, p * ratio(grid, tilt)),
                                                  _integral(out_ball, grid, p))
        res["test_mass_full"][tilt] = full_test_mass(tilt)
        res["weight_integral_train"][tilt] = float(simpson_weights(grid) @ (p_train * ratio(grid, tilt)))
    return res

