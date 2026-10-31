"""Independent implementation of protocol section 4.2 (knee-edge withheld region, depth and load bands).

Written by the second author from the protocol text alone. It shares no code with knee_region.py and does not
import it; the only outside code it touches is the canonical ThrusterChain `stalled()` function, loaded lazily for
the third (cross-)check.

Definitions
-----------
KNEE      float32(0.8) = 0.800000011920929, bits 0x3F4CCCCD.
W         |a0| > KNEE (strict), a0 = the first of four thruster commands, as the executed float32 value.
          W- is a0 < -KNEE (side -1), W+ is a0 > KNEE (side +1). -0.0 is 0 and not in W.
depth     d = |a0| - KNEE, exact rational arithmetic on the float32 values (fractions.Fraction).
R_deep    d >= 1/10 (the exact rational 1/10, inclusive); R_deep- additionally needs a0 < 0.
load g    float64(obs[..., 10]) + float64(obs[..., 11]) (current plus first wave number). Bands, lower edges
          inclusive: low g < 0.4, mid 0.4 <= g < 1.0, deep g >= 1.0, coded 0 / 1 / 2 (uint8).

Band-edge choice: the edges are the float64 numbers 0.4 and 1.0, so g is compared with Fraction(0.4) (the exact
value of the float64 nearest 0.4, 3602879701896397 / 2**53, slightly above 2/5) and Fraction(1), not with
Fraction(2, 5). Since g is itself a float64, this exact comparison gives the same answer as the float64 comparison
g >= 0.4; a g that equals the float64 0.4 is mid, and the float64 one ulp below it is low.
"""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

KNEE = np.float32(0.8)
KNEE_BITS = 0x3F4CCCCD
ABS_MASK = 0x7FFFFFFF
SIGN_BIT = 0x80000000
KNEE_FRACTION = Fraction(float(KNEE))
DEEP_DEPTH = Fraction(1, 10)
LOAD_EDGE_MID = Fraction(0.4)    # exact value of the float64 0.4 (NOT Fraction(2, 5)); see the module docstring
LOAD_EDGE_DEEP = Fraction(1.0)
LOAD_LOW, LOAD_MID, LOAD_DEEP = 0, 1, 2
CURRENT_INDEX, WAVE1_INDEX = 10, 11
OBS_DIM = 14
STALL_SLOPE = 3.0

if int(KNEE.view(np.uint32)) != KNEE_BITS:
    raise RuntimeError("KNEE is not float32 0x3F4CCCCD")

_HERE = Path(__file__).resolve()
# This file: <repo>/experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v1/region_independent.py
_THRUSTERCHAIN_BCA = _HERE.parents[2]
_REPO_ROOT = _HERE.parents[4]
_CORE = _THRUSTERCHAIN_BCA / "titan_20261026" / "payload" / "transfer_bundle" / "core_v4"
_CANONICAL = None


# ------------------------------------------------------------------------------------------------ validation

def _validate_executed(actions):
    """Return actions as a float32 ndarray of shape (..., 4) or raise ValueError."""
    if not isinstance(actions, (np.ndarray, np.generic)):
        raise ValueError("executed actions must be a numpy float32 array")
    actions = np.asarray(actions)
    if actions.dtype != np.float32:
        raise ValueError(f"executed actions must be float32, got {actions.dtype}")
    if actions.ndim < 1 or actions.shape[-1] != 4:
        raise ValueError(f"actions must have shape (..., 4), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain NaN or inf")
    if np.any(np.abs(actions) > np.float32(1.0)):
        raise ValueError("actions contain a command with |a| > 1")
    return actions


def cast_requested(requested):
    """Requested float64 commands -> the executed float32 commands, by requested.astype(np.float32)."""
    requested = np.asarray(requested)
    if requested.dtype != np.float64:
        raise ValueError(f"requested commands must be float64, got {requested.dtype}")
    if not np.all(np.isfinite(requested)):
        raise ValueError("requested commands contain NaN or inf")
    with np.errstate(over="ignore"):
        executed = requested.astype(np.float32)
    return _validate_executed(executed)


def _a0_bits(actions):
    return np.ascontiguousarray(actions[..., 0]).view(np.uint32)


# ------------------------------------------------------------------------------------------------ region W

def withheld_region(actions):
    """Bit test on executed float32 actions (..., 4): (mask bool (...), side int8 (...): -1 W-, +1 W+, 0 else)."""
    actions = _validate_executed(actions)
    bits = _a0_bits(actions)
    mask = (bits & np.uint32(ABS_MASK)) > np.uint32(KNEE_BITS)
    negative = (bits & np.uint32(SIGN_BIT)) != 0
    side = np.zeros(mask.shape, np.int8)
    side[mask & negative] = -1
    side[mask & ~negative] = 1
    return mask, side


def withheld_region_requested(requested):
    """The same test after casting float64 requested commands with requested.astype(np.float32)."""
    return withheld_region(cast_requested(requested))


# ------------------------------------------------------------------------------------------------ depth

def _exact_map(values, func):
    """Apply func(Fraction of value) to every element of a float array, evaluating each distinct value once."""
    flat = np.asarray(values).reshape(-1)
    unique, inverse = np.unique(flat, return_inverse=True)
    results = [func(Fraction(float(v))) for v in unique]
    return results, inverse.reshape(np.shape(values))


def depth_fraction(a0):
    """Exact depth |a0| - KNEE (a Fraction, or an object array of Fractions) for float32 a0."""
    arr = np.asarray(a0)
    if arr.dtype != np.float32:
        raise ValueError(f"a0 must be float32, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("a0 contains NaN or inf")
    results, inverse = _exact_map(arr, lambda f: abs(f) - KNEE_FRACTION)
    if arr.ndim == 0:
        return results[0]
    lookup = np.empty(len(results), object)
    lookup[:] = results
    return lookup[inverse] if arr.size else np.empty(arr.shape, object)


def _depth_test(actions, test):
    actions = _validate_executed(actions)
    a0 = actions[..., 0]
    results, inverse = _exact_map(a0, lambda f: bool(test(abs(f) - KNEE_FRACTION)))
    lookup = np.array(results, bool)
    return lookup[inverse] if a0.size else np.zeros(a0.shape, bool)


def in_region_by_depth(actions):
    """W membership from the exact depth: d > 0."""
    return _depth_test(actions, lambda d: d > 0)


def in_deep(actions):
    """R_deep membership on either side: exact depth d >= 1/10 (inclusive)."""
    return _depth_test(actions, lambda d: d >= DEEP_DEPTH)


def in_deep_minus(actions):
    """R_deep- membership: a0 < 0 and exact depth d >= 1/10."""
    actions = _validate_executed(actions)
    return in_deep(actions) & (actions[..., 0] < 0)


# ------------------------------------------------------------------------------------------------ load bands

def load_g(obs):
    """g = float64(obs[..., 10]) + float64(obs[..., 11]) for float32 observations (..., 14)."""
    if not isinstance(obs, (np.ndarray, np.generic)):
        raise ValueError("observations must be a numpy float32 array")
    obs = np.asarray(obs)
    if obs.dtype != np.float32:
        raise ValueError(f"observations must be float32, got {obs.dtype}")
    if obs.ndim < 1 or obs.shape[-1] != OBS_DIM:
        raise ValueError(f"observations must have shape (..., {OBS_DIM}), got {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("observations contain NaN or inf")
    return obs[..., CURRENT_INDEX].astype(np.float64) + obs[..., WAVE1_INDEX].astype(np.float64)


def band_of_g(g):
    """Band codes (uint8: 0 low, 1 mid, 2 deep) for float64 loads g, by exact Fraction comparison with the
    float64 edges 0.4 and 1.0 (lower edges inclusive)."""
    g = np.asarray(g)
    if g.dtype != np.float64:
        raise ValueError(f"g must be float64, got {g.dtype}")
    if not np.all(np.isfinite(g)):
        raise ValueError("g contains NaN or inf")

    def band(f):
        if f >= LOAD_EDGE_DEEP:
            return LOAD_DEEP
        if f >= LOAD_EDGE_MID:
            return LOAD_MID
        return LOAD_LOW

    results, inverse = _exact_map(g, band)
    if not g.size:
        return np.zeros(g.shape, np.uint8)
    return np.array(results, np.uint8)[inverse]


def load_band(obs):
    """Band codes (uint8 (...)) of float32 observations (..., 14)."""
    return band_of_g(load_g(obs))


# ------------------------------------------------------------------------------------------------ third check

def _load_canonical():
    global _CANONICAL
    if _CANONICAL is None:
        core = str(_CORE)
        if core not in sys.path:
            sys.path.insert(0, core)
        try:
            import gymnasium  # noqa: F401
        except ImportError:
            root = str(_REPO_ROOT)
            if root not in sys.path:
                sys.path.insert(0, root)
            import experiments.shift  # noqa: F401  (puts gymnasium on the path)
        from thrusterchain.canonical import load_canonical
        _CANONICAL = load_canonical()
    return _CANONICAL


def canonical_stalled_a0(actions):
    """Canonical stalled(actions, Physics(stall=True, stall_slope=3.0))[..., 0]."""
    actions = _validate_executed(actions)
    canonical = _load_canonical()
    physics = canonical.Physics(stall=True, stall_slope=STALL_SLOPE)
    return np.asarray(canonical.stalled(actions, physics), bool)[..., 0]


def three_checks(actions):
    """(bit test mask, exact depth > 0 mask, canonical stalled a0 mask), each bool (...)."""
    actions = _validate_executed(actions)
    bit_mask, _ = withheld_region(actions)
    return bit_mask, in_region_by_depth(actions), canonical_stalled_a0(actions)


def agree(actions):
    """Row-for-row agreement of the three W tests; raise RuntimeError on any disagreement, else return the mask."""
    bit_mask, depth_mask, canonical_mask = three_checks(actions)
    bad = (bit_mask != depth_mask) | (bit_mask != canonical_mask)
    if np.any(bad):
        count = int(np.count_nonzero(bad))
        first = tuple(int(i) for i in np.argwhere(bad)[0])
        raise RuntimeError(f"W tests disagree on {count} row(s); first at index {first}: the study stops")
    return bit_mask
