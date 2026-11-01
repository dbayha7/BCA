"""World certification (protocol 4.5b): every run is in the world it claims.

The canonical Physics(), world_physics() and ThrusterChainEnv all default to stall_slope 1.5. A run at 1.5 by mistake
would pass every mask and count check, so each receipt asserts the full pre-registered physics dict of its world.
The identity uses each world's literal thrust formula, never one read back from `physics`.
"""
import numpy as np

from core import load_environment, require

KNEE = float(np.float32(0.8))
_SHARED = {"knee": KNEE, "gain": 0.3, "throttle_cost": 0.3, "wave_turn": 0.27, "gamma": 0.98, "length": 100}
WORLD_DICTS = {
    "stall3": {"stall": True, "stall_slope": 3.0, **_SHARED},
    "nostall": {"stall": False, "stall_slope": 3.0, **_SHARED},
    "stall15_identity": {"stall": True, "stall_slope": 1.5, **_SHARED},
}
LITERAL_FORMULAS = {"stall3": "T3(a0) = sign(a0)(KNEE - 3(|a0| - KNEE))",
                    "stall15_identity": "T15(a0) = sign(a0)(KNEE - 1.5(|a0| - KNEE))"}


def physics(tag):
    """The canonical Physics of a world tag, asserted equal to its pre-registered dict."""
    require(tag in WORLD_DICTS, "Unknown world tag: " + str(tag))
    env = load_environment()
    if tag == "nostall":
        value = env.canonical.world_physics("no_stall", 3.0)
    else:
        value = env.Physics(stall=True, stall_slope=WORLD_DICTS[tag]["stall_slope"])
    assert_world(value, tag)
    return value


def assert_world(value, tag):
    """Raise unless physics.as_dict() equals the named world's full dict; return the receipt entry."""
    require(tag in WORLD_DICTS, "Unknown world tag: " + str(tag))
    recorded = value.as_dict()
    require(recorded == WORLD_DICTS[tag], f"Physics {recorded} is not world {tag}")
    return {"world": tag, "physics": dict(recorded)}


def environment(tag, start="recorded"):
    """A time-aware finite-horizon canonical environment in the named world, certified."""
    env = load_environment()
    spec = WORLD_DICTS[tag]
    result = env.ThrusterChainEnv(stall=spec["stall"], start=start, stall_slope=spec["stall_slope"],
                                  observation_mode="time_aware", task_semantics="finite_horizon")
    assert_world(result.physics, tag)
    return result


def literal_thrust0(a0, tag):
    """The literal thruster-0 thrust of a stalling world past the knee (identity tests only)."""
    a0 = np.asarray(a0, np.float64)
    slope = {"stall3": 3.0, "stall15_identity": 1.5}[tag]
    return np.sign(a0) * (KNEE - slope * (np.abs(a0) - KNEE))


def identity_shift(a0, tag):
    """L2.1: Q_stall(s, a) = Q_stall(s, a0 <- T(a0)) - 0.3 (a0^2 - T(a0)^2). Returns (T(a0) as float32, offset).

    T(a0) is rounded to float32 because the probe is executed as a float32 command; the 1e-6 relative tolerance of
    the identity absorbs that rounding.
    """
    a0 = np.asarray(a0, np.float64)
    require(bool(np.all(np.abs(a0) > KNEE)), "The identity applies past the knee only")
    image = literal_thrust0(a0, tag).astype(np.float32)
    exact = literal_thrust0(a0, tag)
    return image, -0.3 * (a0 * a0 - exact * exact)


def identity_residual(q_at_a, q_at_image, a0, tag):
    """Residual of the literal identity (must be <= 1e-6 x max(1, |Q|), L2.1) and its receipt: the literal formula
    used and the world's full dict."""
    _, offset = identity_shift(a0, tag)
    q_at_a, q_at_image = np.asarray(q_at_a, np.float64), np.asarray(q_at_image, np.float64)
    return q_at_a - (q_at_image + offset), {"world": tag, "formula": LITERAL_FORMULAS[tag], "physics": WORLD_DICTS[tag]}
