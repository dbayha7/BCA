"""The recording controller (who made the log) and the fixed reference controllers.

Textbook gain K (textbook_gain): the discounted (0.98) LQR gain of the rising-thrust model (thrust = command) on all
13 numbers, with state cost I on the floats and 0 on the current and wave, and throttle cost 0.3 I. K_s = K[:, :10]
acts on the floats; K_e = K[:, 10:] holds against the current and wave. Its current column is (0.910, 0.125, 0.532,
0.370).

The recording controller (Recorder), two modes:
  normal driving (about 87% of steps): u = -(0.25 K_s s + K_e e) + 0.3 n, n ~ N(0, I4), command = 0.8 tanh(u / 0.8).
    It holds fully against the current and waves, damps the floats' swinging only a quarter as hard as K would,
    wobbles with SD 0.3 and never goes past 80% throttle.
  the full-throttle command: in the full-throttle region (the water's push c + w1 > the trigger 0.565) it gives,
    with probability 0.5 per step (a separate uniform stream, the 'coin'), the command clip(P + 0.03 n, -1, 1),
    P = -K_e(current) / |K_e(current)_1| = (-1, -0.137, -0.585, -0.406): the hold pattern scaled so thruster 1 is at
    full throttle. Otherwise it drives normally. In the stalling world this command is a bad choice (full throttle
    pushes 0.5 instead of 0.8); in the no-stall comparison world it is a strong hold. The trigger depends only on
    the water, which no thruster changes, so both worlds share the recording controller and its trigger.
The trigger 0.565 comes from stage0.bisect_trigger (calibration seeds 900-902, a 13% share of full-throttle
commands); stage0 checks that the bisection still returns it.

Reference controllers (fixed; no data):
  recording controller (on a bank): both modes with its own fixed-seed noise and coin streams per bank; 0 points.
  stall-aware controller: K clipped at +-0.8; the 100-point anchor in the stalling world (not proven optimal, so
    scores above 100 are possible).
  rising-thrust controller: K clipped at +-1; it assumes thrust keeps rising, and it is the 100-point anchor of the
    no-stall comparison world.
  normal driving only ('hand only'): the normal mode without wobble; about what cloning the gapped log gives.

Every controller here is elementwise and shape independent: the products with K are fixed-order sums over columns,
so a batch gives bit for bit the numbers of single rows. Controllers have a numpy form (__call__ / act) and a JAX
form (jax) for the exact-truth code.
"""

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy import linalg

from experiments.shift import thruster_chain as TC

TRIGGER = 0.565
RECORDING_BANK_STREAM = (7, 1)  # np.random.default_rng([7, 1]): noise (n, 100, 4) then coin (n, 100), every bank


@dataclass(frozen=True)
class RecorderSpec:
    """The recording controller's numbers.

    damping: the share of the textbook float feedback K_s it applies in normal driving (0.25). wobble: standard
    deviation of the noise added to the throttle demand (0.3). normal_driving_cap: normal-driving commands stay inside
    +-normal_driving_cap via normal_driving_cap * tanh(u / normal_driving_cap) (0.8). full_throttle_probability: the
    chance per step of the full-throttle command inside the full-throttle region (0.5). jitter: standard deviation of
    the noise on the full-throttle command (0.03). trigger: the water's push c + w1 above which the full-throttle
    region starts (0.565). full_throttle_share_target: the share of full-throttle commands the trigger bisection aims
    at (0.13).
    """

    damping: float = 0.25
    wobble: float = 0.3
    normal_driving_cap: float = 0.8
    full_throttle_probability: float = 0.5
    jitter: float = 0.03
    trigger: float = TRIGGER
    full_throttle_share_target: float = 0.13

    def as_dict(self):
        return asdict(self)


DEFAULT_SPEC = RecorderSpec()


@lru_cache(maxsize=None)
def _gain(gain, wave_turn, throttle_cost, gamma):
    A, B = TC._matrices(gain, wave_turn)
    Qc = np.zeros((TC.DX, TC.DX))
    Qc[:TC.DS, :TC.DS] = np.eye(TC.DS)
    g = gamma
    P = linalg.solve_discrete_are(math.sqrt(g) * A, math.sqrt(g) * B, Qc, throttle_cost * np.eye(TC.DA))
    K = np.linalg.solve(throttle_cost * np.eye(TC.DA) + g * B.T @ P @ B, g * B.T @ P @ A)
    K.setflags(write=False)
    return K


def textbook_gain(physics=TC.STALL):
    """K (4 x 13): the discounted LQR gain of the rising-thrust model (the same for both worlds). Read-only."""
    return _gain(float(physics.gain), float(physics.wave_turn), float(physics.throttle_cost), float(physics.gamma))


def full_throttle_pattern(K):
    """P = -K[:, current] / |K[0, current]|: the hold against the current scaled so thruster 1 is at full throttle."""
    K = np.asarray(K, np.float64)
    return -K[:, TC.CURRENT] / abs(K[0, TC.CURRENT])


def _linear(x, M, xp=np):
    """The two parts of M x for M (4 x 13): (x_s @ M_s.T, x_e @ M_e.T), each a fixed-order sum over columns, so the
    result does not depend on the batch shape. Callers combine the parts in one fixed order."""
    cols = [np.asarray(M[:, j], np.float64) for j in range(TC.DX)]
    if xp is not np:
        cols = [xp.asarray(c, x.dtype) for c in cols]
    fl = x[..., 0:1] * cols[0]
    for j in range(1, TC.DS):
        fl = fl + x[..., j:j + 1] * cols[j]
    ex = x[..., TC.DS:TC.DS + 1] * cols[TC.DS]
    for j in range(TC.DS + 1, TC.DX):
        ex = ex + x[..., j:j + 1] * cols[j]
    return fl, ex


class Controller:
    """A deterministic controller: __call__(x) maps states (..., 13) to commands (..., 4) in numpy float64, jax(x)
    does the same in JAX. act(x, t) is the time-indexed form the rollouts call (deterministic controllers ignore t).
    `name` is a plain description."""

    name = "controller"

    def __call__(self, x):
        raise NotImplementedError

    def jax(self, x):
        raise NotImplementedError

    def act(self, x, t):
        return self(x)


class LinearController(Controller):
    """command = clip(-(K_s s + K_e e), -cap, cap): the stall-aware controller (cap 0.8) or the rising-thrust
    controller (cap 1.0); cap=None leaves the command unclipped (used only for the closed-form check)."""

    def __init__(self, K, cap, name):
        self.K = np.asarray(K, np.float64)
        self.cap = None if cap is None else float(cap)
        self.name = name

    def __call__(self, x):
        fl, ex = _linear(np.asarray(x, np.float64), self.K)
        u = -(fl + ex)
        return u if self.cap is None else np.clip(u, -self.cap, self.cap)

    def jax(self, x):
        import jax.numpy as jnp

        fl, ex = _linear(x, self.K, xp=jnp)
        u = -(fl + ex)
        return u if self.cap is None else jnp.clip(u, -self.cap, self.cap)


class HandOnly(Controller):
    """Normal driving without wobble: normal_driving_cap * tanh(-(damping K_s s + K_e e) / normal_driving_cap). It
    never goes past 80%."""

    def __init__(self, K, spec=DEFAULT_SPEC):
        self.K = np.asarray(K, np.float64)
        self.spec = spec
        self.name = "normal driving only (no wobble, never the full-throttle command)"

    def __call__(self, x):
        fl, ex = _linear(np.asarray(x, np.float64), self.K)
        u = -(self.spec.damping * fl + ex)
        return self.spec.normal_driving_cap * np.tanh(u / self.spec.normal_driving_cap)

    def jax(self, x):
        import jax.numpy as jnp

        fl, ex = _linear(x, self.K, xp=jnp)
        u = -(self.spec.damping * fl + ex)
        return self.spec.normal_driving_cap * jnp.tanh(u / self.spec.normal_driving_cap)


class CappedController(Controller):
    """Another controller's commands clipped to +-cap (0.8) at evaluation: the same policy kept out of the
    throttles the gapped log no longer shows (used for the loss from unrecorded throttle)."""

    def __init__(self, base, cap=0.8):
        self.base, self.cap = base, float(cap)
        self.name = f"{base.name}, commands capped at {self.cap:g}"

    def __call__(self, x):
        return np.clip(self.base(x), -self.cap, self.cap)

    def act(self, x, t):
        return np.clip(self.base.act(x, t), -self.cap, self.cap)

    def jax(self, x):
        import jax.numpy as jnp

        return jnp.clip(self.base.jax(x), -self.cap, self.cap)


class Recorder:
    """The recording controller (module docstring). parts(x, noise, coin) -> (command, in_region, full_throttle):
    noise (..., 4) standard normal (None: no wobble and no jitter), coin (...,) uniform in [0, 1) (None: never the
    full-throttle command). All arrays broadcast over leading dimensions."""

    def __init__(self, spec=DEFAULT_SPEC, physics=TC.STALL):
        self.spec = spec
        self.physics = physics
        self.K = textbook_gain(physics)
        self.pattern = full_throttle_pattern(self.K)
        self.name = "recording controller"

    def in_region(self, x):
        """Per state: is the water's push c + w1 above the trigger (the full-throttle region)?"""
        return TC.push(x) > self.spec.trigger

    def parts(self, x, noise=None, coin=None):
        x = np.asarray(x, np.float64)
        fl, ex = _linear(x, self.K)
        u = -(self.spec.damping * fl + ex)
        if noise is not None:
            u = u + self.spec.wobble * np.asarray(noise, np.float64)
        hand = self.spec.normal_driving_cap * np.tanh(u / self.spec.normal_driving_cap)
        full = self.pattern if noise is None else self.pattern + self.spec.jitter * np.asarray(noise, np.float64)
        full = np.clip(full, -1.0, 1.0)
        region = self.in_region(x)
        if coin is None:
            press = np.zeros(region.shape, bool)
        else:
            press = region & (np.asarray(coin, np.float64) < self.spec.full_throttle_probability)
        command = np.where(press[..., None], np.broadcast_to(full, hand.shape), hand)
        return command, region, press

    def __call__(self, x, noise=None, coin=None):
        return self.parts(x, noise, coin)[0]


class StreamedRecorder(Controller):
    """The recording controller on a fixed set of starts with its own fixed noise and coin streams: act(x, t) uses
    noise[:, t] and coin[:, t]. On the evaluation banks the streams are default_rng([7, 1]) (noise (n, 100, 4), then
    coin (n, 100)), the same for every bank."""

    def __init__(self, recorder, noise, coin):
        self.recorder = recorder
        self.noise = np.asarray(noise, np.float64)
        self.coin = np.asarray(coin, np.float64)
        self.name = "recording controller (fixed noise and coin streams)"

    @classmethod
    def on_bank(cls, recorder, n=TC.BANK_SIZE, length=TC.STALL.length, stream=RECORDING_BANK_STREAM):
        rng = np.random.default_rng(list(stream))
        noise = rng.standard_normal((n, length, TC.DA))
        coin = rng.uniform(size=(n, length))
        return cls(recorder, noise, coin)

    def act(self, x, t):
        return self.recorder(x, self.noise[:, t], self.coin[:, t])

    def __call__(self, x):
        raise TypeError("the recording controller is random; call act(x, t) with a step index")


def stall_aware(physics=TC.STALL):
    """K clipped at +-0.8: never past 80% throttle. The stalling world's 100-point anchor."""
    return LinearController(textbook_gain(physics), 0.8, "stall-aware controller (textbook gain, capped at 80%)")


def rising_thrust(physics=TC.STALL):
    """K clipped at +-1: assumes thrust keeps rising. The no-stall comparison world's 100-point anchor."""
    return LinearController(textbook_gain(physics), 1.0, "rising-thrust controller (textbook gain, full range)")


def hand_only(spec=DEFAULT_SPEC, physics=TC.STALL):
    return HandOnly(textbook_gain(physics), spec)


def top_anchor(physics):
    """The 100-point controller of a world: stall-aware in the stalling world, rising-thrust in the no-stall world."""
    return stall_aware(physics) if physics.stall else rising_thrust(physics)
