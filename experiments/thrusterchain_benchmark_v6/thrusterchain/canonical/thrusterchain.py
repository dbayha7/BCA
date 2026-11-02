"""ThrusterChain: the Gymnasium environment, its one dynamics function, the start laws and the evaluation banks.

The world. Five floats sit in a row in a channel, tied together by springs. A steady current and waves push every
float downstream. Four thrusters sit between neighbouring floats: thruster k pushes the speeds of floats k and k + 1.

State x, 13 float64 numbers:
  s (10)      the positions and speeds of the five floats (coordinates 2j and 2j + 1), the Gapped LQ Stage-0
              oscillator plant imported read-only: gapped_lq.transition_matrix(STAGE0_SPEC) (radii 0.98-0.96,
              angles 0.10/0.20/0.35/0.55/0.80 rad, coupling seed 20261011)
  c (1)       the strength of the current; it never changes during an episode
  (w1, w2)    the wave, a point turning 0.27 rad per step (a 23.3-step cycle), like the hands of a clock

Dynamics (deterministic, no noise anywhere):
  s' = A s + 0.3 B1 thrust(a) + 0.3 v (c + w1),   c' = c,   (w1, w2)' = (w1, w2) turned by 0.27 rad,
  with B1 = gapped_lq.base_input(1.0) and v pushing the speed of every float equally.
Thrust per thruster, after clipping the command a to [-1, 1]: thrust(a) = a while |a| <= 0.8; past 80% the
propeller stalls and the push falls: thrust(a) = sign(a) (0.8 - 1.5 (|a| - 0.8)), so 0.65 at 90% and 0.5 at full
throttle. In the no-stall comparison world thrust(a) = a everywhere. At or below 80% the two worlds are identical.
The 80% point (the knee, KNEE) is the float32 number nearest 0.8, 0.800000011920929: the action space is float32, so
a command of exactly 80% sent through env.step arrives as float32(0.8), and it must sit at the knee, not past it.
(Every value above therefore holds to within 3e-8: thrust is 0.65000003 at 90% and 0.50000003 at full throttle.)
Reward r = -(|s|^2 + 0.3 |a|^2) on the clipped command: energy follows the throttle even while a thruster stalls.

One dynamics function. dynamics(x, a, physics) (numpy) and dynamics_jax (JAX, same arithmetic order) are the only
implementations; ThrusterChainEnv.step, the dataset generator and the truth code all call them. Every matrix product
is written as a fixed-order sum over columns of elementwise operations, so a batch of states gives bit for bit the
same numbers as the same states one at a time.

The environment (gymnasium.Env). Observation: x as float32 in a Box with finite, proven bounds (floats within
+-300, current in [0, 1.5], wave numbers in [-1, 1]). Action: Box(-1, 1, (4,), float32); step casts the command to
float32, then to float64, and clips it, as MuJoCo clamps its control range. Episodes end by truncation only, after
100 steps (terminated is always False). reset(seed, options): the default start draw uses self.np_random;
options={"state": 13 numbers} restarts from an exact state inside the start set (|s| <= 100, 0 <= c <= 1.5, wave
amplitude <= 1); options={"start": {...}} overrides the start law's ranges. Step info: state (float64), t, push (the
water's push c + w1 at the state the command was given), thrust, stalled (|a_k| > 0.8; always False in the no-stall
world; a command of exactly 80% does not stall), command (the clipped float64 command). render_mode="ansi" returns
one line of text.

Registered ids (max_episode_steps = 100). Within a world only the start law differs:
  BCAShift/ThrusterChain-v0                recorded conditions: c ~ U[0.1, 0.7], wave amplitude ~ U[0.1, 0.4],
                                           float start SD 0.5
  BCAShift/ThrusterChain-StrongCurrent-v0  c ~ U[0.7, 1.1] (every start stronger than any recorded current)
  BCAShift/ThrusterChain-BigWaves-v0       wave amplitude ~ U[0.4, 0.7]
  BCAShift/ThrusterChain-BigKicks-v0       float start SD 1.5
  BCAShift/ThrusterChainNoStall-v0 and its -StrongCurrent / -BigWaves / -BigKicks variants: the same four start
  laws in the no-stall comparison world.

Evaluation banks: four fixed sets of 1,000 seeded starts (one per situation), the same for every controller, drawn by
draw_starts from default_rng([20261012, 2, 9]).

Observation bound (observation_bound()): sup_t ||A^t|| = 1.000 and sum_t ||A^t|| = 55.48; the largest input per step
is ||0.3 B1|| |thrust| + ||0.3 v|| (c + amplitude) <= 2.82 (no-stall world, current 1.5, amplitude 1), so
|s_t| <= |s_0| + 156.4 <= 256.4 < 300 from any start in the start set. step() also checks it.
"""

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

import experiments.shift  # noqa: F401  (puts the dependency directory on sys.path and checks the versions)
import gymnasium as gym
from gymnasium import spaces

from experiments.signal import gapped_lq as GLQ

DS, DE, DA = 10, 3, 4
DX = DS + DE
CURRENT, WAVE1, WAVE2 = DS, DS + 1, DS + 2
POSITIONS = slice(0, DS, 2)

FLOAT_BOUND = 300.0          # observation_space bound on every float coordinate (proven: |s_t| <= 256.4)
START_FLOAT_RADIUS = 100.0   # start set: |s_0| <= 100
START_CURRENT = (0.0, 1.5)   # start set: 0 <= c <= 1.5
START_WAVE_AMPLITUDE = 1.0   # start set: wave amplitude <= 1
WAVE_ROUNDING = 1e-9         # the wave's turn keeps its amplitude in exact arithmetic; float64 rounding may lift it
                             # a few ulps above 1, so the start set allows amplitude <= 1 + 1e-9
KNEE = float(np.float32(0.8))  # the 80% point: the float32 number nearest 0.8, so float32(0.8) is at the knee
OBS_LOW = np.array([-FLOAT_BOUND] * DS + [0.0, -1.0, -1.0], np.float32)
OBS_HIGH = np.array([FLOAT_BOUND] * DS + [1.5, 1.0, 1.0], np.float32)

DATA_SEED = 20261012
BANK_SEED = (DATA_SEED, 2, 9)
BANK_SIZE = 1000

START_LAWS = {
    "recorded": dict(current=(0.1, 0.7), wave=(0.1, 0.4), kick=0.5),
    "strong_current": dict(current=(0.7, 1.1), wave=(0.1, 0.4), kick=0.5),
    "big_waves": dict(current=(0.1, 0.7), wave=(0.4, 0.7), kick=0.5),
    "big_kicks": dict(current=(0.1, 0.7), wave=(0.1, 0.4), kick=1.5),
}
SITUATIONS = tuple(START_LAWS)
SITUATION_LABELS = {"recorded": "recorded conditions", "strong_current": "strong current",
                    "big_waves": "big waves", "big_kicks": "big kicks"}
SHIFTED_SITUATIONS = ("strong_current", "big_waves", "big_kicks")
RECORDED_CURRENT_MAX = START_LAWS["recorded"]["current"][1]   # 0.7
RECORDED_WAVE_MAX = START_LAWS["recorded"]["wave"][1]         # 0.4
ID_SUFFIX = {"recorded": "", "strong_current": "-StrongCurrent", "big_waves": "-BigWaves", "big_kicks": "-BigKicks"}
ENTRY_POINT = "experiments.shift.thruster_chain:ThrusterChainEnv"


@dataclass(frozen=True)
class Physics:
    """Every physical number of the world.

    stall: True for the stalling world, False for the no-stall comparison world (thrust keeps rising to 100%).
    stall_slope: how fast the push falls past the knee (1.5: push 0.5 at full throttle; the pre-registered retune
    uses 3.0: push 0.2). knee: the throttle where the stall starts, 80% (KNEE: the float32 number nearest 0.8, so a
    float32 command of exactly 80% is at the knee and does not stall). gain: thruster and water push per unit
    (0.3). throttle_cost: the weight of the squared throttles in the cost (0.3). wave_turn: radians per step (0.27).
    gamma: the discount every learner and the truth use (0.98). length: steps per episode (100).
    """

    stall: bool = True
    stall_slope: float = 1.5
    knee: float = KNEE
    gain: float = 0.3
    throttle_cost: float = 0.3
    wave_turn: float = 0.27
    gamma: float = 0.98
    length: int = 100

    @property
    def thrust_knee(self):
        """Where thrust stops equalling the command: the knee in the stalling world, full throttle otherwise."""
        return self.knee if self.stall else 1.0

    @property
    def thrust_slope(self):
        return self.stall_slope if self.stall else 0.0

    def as_dict(self):
        return asdict(self)


STALL = Physics()
NO_STALL = Physics(stall=False)
WORLDS = {"stall": STALL, "no_stall": NO_STALL}
WORLD_LABELS = {"stall": "stalling world", "no_stall": "no-stall comparison world"}


def world_physics(world, stall_slope=1.5):
    """Physics of a named world ('stall' or 'no_stall'); stall_slope applies to the stalling world only."""
    if world == "stall":
        return Physics(stall=True, stall_slope=float(stall_slope))
    if world == "no_stall":
        return Physics(stall=False, stall_slope=float(stall_slope))
    raise ValueError("world must be 'stall' or 'no_stall'")


# ------------------------------------------------------------------------------------------------ the plant

def push_pattern():
    """v: the water pushes the speed (coordinate 2j + 1) of every float equally."""
    v = np.zeros(DS)
    v[1::2] = 1.0
    return v


@lru_cache(maxsize=None)
def _matrices(gain, wave_turn):
    A10, _ = GLQ.transition_matrix(GLQ.STAGE0_SPEC)
    A = np.zeros((DX, DX))
    A[:DS, :DS] = A10
    v = push_pattern()
    A[:DS, CURRENT] = gain * v
    A[:DS, WAVE1] = gain * v
    A[CURRENT, CURRENT] = 1.0
    c, s = math.cos(wave_turn), math.sin(wave_turn)
    A[WAVE1:, WAVE1:] = [[c, -s], [s, c]]
    B = np.zeros((DX, DA))
    B[:DS] = gain * GLQ.base_input(1.0)
    A.setflags(write=False)
    B.setflags(write=False)
    return A, B


def matrices(physics=STALL):
    """(A, B): x' = A x + B thrust(a). The same in both worlds (only thrust differs). Read-only arrays."""
    return _matrices(float(physics.gain), float(physics.wave_turn))


@lru_cache(maxsize=None)
def _columns(gain, wave_turn):
    A, B = _matrices(gain, wave_turn)
    return tuple(A[:, j].copy() for j in range(DX)), tuple(B[:, k].copy() for k in range(DA))


def thrust(a, physics=STALL):
    """The push each thruster applies for clipped commands a (numpy): a up to the knee, then falling.

    Written as clip(a, -knee, knee) - slope sign(a) max(|a| - knee, 0): the same numbers, bit for bit, as
    sign(a) (min(|a|, knee) - slope max(|a| - knee, 0)), but its JAX form (thrust_jax) has the right derivative 1 at
    a command of exactly 0 (autodiff takes the derivative of sign as 0, which would make the slope 0 there)."""
    a = np.asarray(a, np.float64)
    knee, slope = physics.thrust_knee, physics.thrust_slope
    return np.clip(a, -knee, knee) - slope * np.sign(a) * np.maximum(np.abs(a) - knee, 0.0)


def dynamics(x, a, physics=STALL):
    """One step: (x', r) for states x (..., 13) and commands a (..., 4), float64 (numpy).

    The command is clipped to [-1, 1]; the reward uses the clipped command; x' = sum_j A[:, j] x_j +
    sum_k B[:, k] thrust_k, summed in this fixed order, so the result does not depend on the batch shape."""
    x = np.asarray(x, np.float64)
    a = np.clip(np.asarray(a, np.float64), -1.0, 1.0)
    th = thrust(a, physics)
    acols, bcols = _columns(float(physics.gain), float(physics.wave_turn))
    nxt = x[..., 0:1] * acols[0]
    for j in range(1, DX):
        nxt = nxt + x[..., j:j + 1] * acols[j]
    for k in range(DA):
        nxt = nxt + th[..., k:k + 1] * bcols[k]
    s2 = x[..., 0] * x[..., 0]
    for i in range(1, DS):
        s2 = s2 + x[..., i] * x[..., i]
    a2 = a[..., 0] * a[..., 0]
    for k in range(1, DA):
        a2 = a2 + a[..., k] * a[..., k]
    return nxt, -(s2 + physics.throttle_cost * a2)


def thrust_jax(a, physics=STALL):
    """thrust() in JAX, the same arithmetic order; its autodiff slope is 1 below the knee, 0 included."""
    import jax.numpy as jnp

    knee, slope = physics.thrust_knee, physics.thrust_slope
    return jnp.clip(a, -knee, knee) - slope * jnp.sign(a) * jnp.maximum(jnp.abs(a) - knee, 0.0)


def dynamics_jax(x, a, physics=STALL):
    """dynamics() in JAX with the same arithmetic order. Call it under jax.experimental.enable_x64() for float64."""
    import jax.numpy as jnp

    a = jnp.clip(a, -1.0, 1.0)
    th = thrust_jax(a, physics)
    acols, bcols = _columns(float(physics.gain), float(physics.wave_turn))
    dtype = x.dtype
    nxt = x[..., 0:1] * jnp.asarray(acols[0], dtype)
    for j in range(1, DX):
        nxt = nxt + x[..., j:j + 1] * jnp.asarray(acols[j], dtype)
    for k in range(DA):
        nxt = nxt + th[..., k:k + 1] * jnp.asarray(bcols[k], dtype)
    s2 = x[..., 0] * x[..., 0]
    for i in range(1, DS):
        s2 = s2 + x[..., i] * x[..., i]
    a2 = a[..., 0] * a[..., 0]
    for k in range(1, DA):
        a2 = a2 + a[..., k] * a[..., k]
    return nxt, -(s2 + physics.throttle_cost * a2)


def past_knee(a, physics=STALL):
    """Per row: does any thruster's command go past 80% throttle (|a_k| > the knee)? These are the throttles the
    gapped log no longer shows ('unrecorded throttle'). Uses the knee of the physics even in the no-stall world. A
    command of exactly 80%, in float64 (0.8) or in float32 (0.800000011920929), is not past it."""
    return np.any(np.abs(np.asarray(a, np.float64)) > physics.knee, axis=-1)


def stalled(a, physics=STALL):
    """Per thruster: is it stalling? |a_k| > knee in the stalling world (a command of exactly 80%, float32 or
    float64, is not); never in the no-stall world."""
    a = np.asarray(a, np.float64)
    if not physics.stall:
        return np.zeros(a.shape, bool)
    return np.abs(a) > physics.knee


def push(x):
    """The water's push on the floats at state x: current plus the first wave number (c + w1)."""
    x = np.asarray(x, np.float64)
    return x[..., CURRENT] + x[..., WAVE1]


def wave_amplitude(x):
    x = np.asarray(x, np.float64)
    return np.hypot(x[..., WAVE1], x[..., WAVE2])


def float_radius(x):
    x = np.asarray(x, np.float64)
    return np.sqrt(np.sum(x[..., :DS] ** 2, axis=-1))


# ------------------------------------------------------------------------------------------------ starts

def start_law(base, override=None):
    """The start law `base` (a START_LAWS entry) with an options['start'] override checked against the start set:
    current (lo, hi) within [0, 1.5], wave (lo, hi) amplitudes within [0, 1], kick (the float start SD) >= 0."""
    law = dict(base)
    for key, value in dict(override or {}).items():
        if key not in ("current", "wave", "kick"):
            raise ValueError(f"unknown start-law key {key!r}; use current, wave or kick")
        law[key] = value
    lo, hi = (float(v) for v in law["current"])
    if not START_CURRENT[0] <= lo <= hi <= START_CURRENT[1]:
        raise ValueError("the current range must lie within [0, 1.5]")
    lo, hi = (float(v) for v in law["wave"])
    if not 0.0 <= lo <= hi <= START_WAVE_AMPLITUDE:
        raise ValueError("the wave amplitude range must lie within [0, 1]")
    if not float(law["kick"]) >= 0.0:
        raise ValueError("the float start SD (kick) must be >= 0")
    return dict(current=tuple(float(v) for v in law["current"]), wave=tuple(float(v) for v in law["wave"]),
                kick=float(law["kick"]))


def draw_starts(n, rng, law):
    """n start states from a start law, in this draw order: float positions and speeds kick * N(0, I10) (n x 10),
    then the current U[current], the wave amplitude U[wave] and the wave phase U[0, 2 pi) (n each)."""
    s = law["kick"] * rng.standard_normal((n, DS))
    c = rng.uniform(*law["current"], size=n)
    amp = rng.uniform(*law["wave"], size=n)
    ph = rng.uniform(0.0, 2 * math.pi, size=n)
    return np.concatenate([s, c[:, None], (amp * np.cos(ph))[:, None], (amp * np.sin(ph))[:, None]], 1)


def in_start_set(x):
    """Per row: |s| <= 100, 0 <= c <= 1.5 and wave amplitude <= 1 (and every number finite). The amplitude test
    allows 1 + WAVE_ROUNDING (1e-9): the wave's turn keeps the amplitude constant in exact arithmetic, but float64
    rounding can lift a start of amplitude 1 a few ulps above it, and a state the environment itself produced from
    an allowed start must be accepted again for replay and truth."""
    x = np.asarray(x, np.float64)
    finite = np.all(np.isfinite(x), axis=-1)
    with np.errstate(invalid="ignore"):
        return (finite & (float_radius(x) <= START_FLOAT_RADIUS) & (x[..., CURRENT] >= START_CURRENT[0])
                & (x[..., CURRENT] <= START_CURRENT[1])
                & (wave_amplitude(x) <= START_WAVE_AMPLITUDE + WAVE_ROUNDING))


def bank_starts(situation, n=BANK_SIZE):
    """The fixed evaluation bank of a situation: n starts from default_rng([20261012, 2, 9]) under its start law.
    The four banks share the seed, so they differ only by their start laws (common random numbers)."""
    if situation not in START_LAWS:
        raise ValueError(f"unknown situation {situation!r}; one of {', '.join(SITUATIONS)}")
    return draw_starts(int(n), np.random.default_rng(list(BANK_SEED)), START_LAWS[situation])


def outside_recorded_conditions(x, float_mean, float_sd, float_radius_p99):
    """Per state: is it outside the recorded conditions? current > 0.7, or wave amplitude > 0.4, or the floats'
    standardised radius beyond the training 99th percentile. Returns (outside, by_current, by_wave, by_floats)."""
    x = np.asarray(x, np.float64)
    by_current = x[..., CURRENT] > RECORDED_CURRENT_MAX
    by_wave = wave_amplitude(x) > RECORDED_WAVE_MAX
    z = (x[..., :DS] - np.asarray(float_mean)) / np.asarray(float_sd)
    by_floats = np.sqrt(np.sum(z * z, axis=-1)) > float_radius_p99
    return by_current | by_wave | by_floats, by_current, by_wave, by_floats


def observation_bound(physics_list=(STALL, NO_STALL), terms=5000):
    """The proven float bound: |s_t| <= |s_0| + sum_t ||A^t|| * (largest input per step) over the start set."""
    A, B = matrices(STALL)
    A10 = A[:DS, :DS]
    norms, M = [], np.eye(DS)
    for _ in range(terms):
        norms.append(np.linalg.norm(M, 2))
        M = A10 @ M
    norms = np.asarray(norms)
    b_norm = np.linalg.norm(B[:DS], 2)
    v_norm = np.linalg.norm(A[:DS, CURRENT])
    largest_thrust = max(np.linalg.norm(thrust(np.full(DA, 1.0), p)) if not p.stall
                         else np.linalg.norm(np.full(DA, p.knee)) for p in physics_list)
    per_step = b_norm * largest_thrust + v_norm * (START_CURRENT[1] + START_WAVE_AMPLITUDE + WAVE_ROUNDING)
    growth = float(norms.sum() * per_step)
    return dict(sup_norm=float(norms.max()), sum_norm=float(norms.sum()), input_norm=float(b_norm),
                push_norm=float(v_norm), largest_thrust_norm=float(largest_thrust), per_step_input=float(per_step),
                growth=growth, start_radius=START_FLOAT_RADIUS, bound=START_FLOAT_RADIUS + growth,
                observation_limit=FLOAT_BOUND)


# ------------------------------------------------------------------------------------------------ render

def render_line(x, t, command, physics=STALL):
    """One line of text: time, current, wave, the five float positions and the throttles ('*' = stalling)."""
    if x is None:
        return "ThrusterChain: not started (call reset)"
    floats = " ".join(f"{p:+6.2f}" for p in np.asarray(x)[POSITIONS])
    if command is None:
        throttles = " ".join(["   -  "] * DA)
    else:
        flags = stalled(command, physics)
        throttles = " ".join(f"{c:+5.2f}{'*' if f else ' '}" for c, f in zip(command, flags))
    return (f"t={t:3d} current={x[CURRENT]:.2f} wave={x[WAVE1]:+.2f} floats=[{floats}] "
            f"throttles=[{throttles}]")


# ------------------------------------------------------------------------------------------------ the env

class ThrusterChainEnv(gym.Env):
    """The ThrusterChain Gymnasium environment (module docstring).

    Arguments: stall (True: the stalling world; False: the no-stall comparison world), start (a START_LAWS name: the
    start law of the default reset draw), render_mode (None or "ansi"), stall_slope (1.5; the pre-registered retune
    uses 3.0)."""

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(self, stall=True, start="recorded", render_mode=None, stall_slope=1.5):
        if start not in START_LAWS:
            raise ValueError(f"unknown start law {start!r}; one of {', '.join(SITUATIONS)}")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.physics = Physics(stall=bool(stall), stall_slope=float(stall_slope))
        self.start = start
        self.law = start_law(START_LAWS[start])
        self.render_mode = render_mode
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, (DX,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (DA,), np.float32)
        self._x = None
        self._t = 0
        self._command = None

    def _obs(self):
        return self._x.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        unknown = set(options) - {"state", "start"}
        if unknown:
            raise ValueError(f"unknown reset options {sorted(unknown)}; use 'state' or 'start'")
        if "state" in options and "start" in options:
            raise ValueError("give either options['state'] or options['start'], not both")
        if "state" in options:
            x = np.array(options["state"], dtype=np.float64).reshape(-1)
            if x.shape != (DX,):
                raise ValueError(f"options['state'] must hold {DX} numbers")
            if not bool(in_start_set(x)):
                raise ValueError("options['state'] is outside the start set (|s| <= 100, 0 <= current <= 1.5, "
                                 "wave amplitude <= 1)")
        else:
            x = draw_starts(1, self.np_random, start_law(self.law, options.get("start")))[0]
            if not bool(in_start_set(x)):
                raise ValueError("the start draw fell outside the start set; lower the float start SD")
        self._x, self._t, self._command = x.copy(), 0, None
        return self._obs(), {"state": self._x.copy(), "t": 0}

    def step(self, action):
        if self._x is None:
            raise RuntimeError("call reset() before step()")
        if self._t >= self.physics.length:
            raise RuntimeError(f"the episode ended after {self.physics.length} steps; call reset()")
        a = np.asarray(action, dtype=np.float32).astype(np.float64).reshape(-1)
        if a.shape != (DA,) or not np.all(np.isfinite(a)):
            raise ValueError(f"the action must be {DA} finite numbers")
        a = np.clip(a, -1.0, 1.0)
        x = self._x
        nxt, reward = dynamics(x, a, self.physics)
        if not (np.all(np.isfinite(nxt)) and float_radius(nxt) <= FLOAT_BOUND):
            raise RuntimeError("the float state left the proven bound; this cannot happen from the start set")
        self._x, self._t, self._command = nxt, self._t + 1, a
        info = {"state": nxt.copy(), "t": self._t, "push": float(push(x)), "thrust": thrust(a, self.physics),
                "stalled": stalled(a, self.physics), "command": a.copy()}
        return self._obs(), float(reward), False, self._t >= self.physics.length, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        return render_line(self._x, self._t, self._command, self.physics)


def env_id(world, situation):
    """The registered id of a world ('stall' / 'no_stall') and situation (a START_LAWS name)."""
    if world not in WORLDS or situation not in START_LAWS:
        raise ValueError("unknown world or situation")
    return f"BCAShift/ThrusterChain{'' if world == 'stall' else 'NoStall'}{ID_SUFFIX[situation]}-v0"


ENV_IDS = {env_id(w, s): (w, s) for w in WORLDS for s in SITUATIONS}


def register():
    """Register the 8 BCAShift ids with Gymnasium (idempotent). Called on import."""
    for eid, (world, situation) in ENV_IDS.items():
        if eid not in gym.registry:
            gym.register(id=eid, entry_point=ENTRY_POINT, max_episode_steps=STALL.length,
                         kwargs=dict(stall=world == "stall", start=situation))
    return tuple(ENV_IDS)


register()
