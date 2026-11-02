"""The shared knee-region module (protocol 4.1). It is the only predicate the loaders, mask builders, normalization,
calibration selection, evaluation counters, gradient audit and analysis use; its source hash goes in every receipt.

W, the withheld region, is |a0| > KNEE (strict) on the executed float32 command, compared in float64.
"""
from pathlib import Path

import numpy as np

from core import load_environment, require, sha_file

KNEE = float(np.float32(0.8))              # 0.800000011920929, bits 0x3F4CCCCD
KNEE_BITS = 0x3F4CCCCD
DEEP_DEPTH = 0.10                          # R_deep- is a0 <= -(KNEE + 0.10), inclusive
LOAD_MID, LOAD_DEEP = 0.4, 1.0             # bands: low g < 0.4 <= mid < 1.0 <= deep (lower edges inclusive)
SOURCE_SHA256 = sha_file(__file__)

# Part 0 slot table: executed float32 values, by their exact bit patterns.
_SLOT_BITS = {
    "s05": 0xBF59999A, "s10": 0xBF666667, "s15": 0xBF733333, "s20": 0xBF800000, "p10": 0x3F666667,
    "edge_knee": 0xBF4CCCCD, "edge_775": 0xBF466666, "edge_750": 0xBF400000, "edge_725": 0xBF39999A,
    "edge_700": 0xBF333333, "mid_m40": 0xBECCCCCD, "zero": 0x00000000, "mid_p40": 0x3ECCCCCD, "knee": 0x3F4CCCCD,
}
SLOTS = {name: float(np.uint32(bits).view(np.float32)) for name, bits in _SLOT_BITS.items()}
S9 = ("edge_knee", "edge_775", "edge_750", "edge_725", "edge_700", "mid_m40", "zero", "mid_p40", "knee")
GATED_PAST_KNEE = ("s05", "s10", "s15", "s20", "p10")


def _executed(actions, kind):
    if kind == "requested":
        raw = np.asarray(actions)
        require(raw.dtype == np.float64, "Requested commands must be float64")
        actions = raw.astype(np.float32)
    else:
        require(kind == "executed", "kind must be 'executed' or 'requested'")
        actions = np.asarray(actions)
        require(actions.dtype == np.float32, "Executed commands must be float32 (float64 passed as executed)")
    require(actions.ndim >= 1 and actions.shape[-1] == 4, "Commands must have shape (..., 4)")
    wide = actions.astype(np.float64)
    require(bool(np.all(np.isfinite(wide))), "Commands must be finite")
    require(bool(np.all(np.abs(wide) <= 1.0)), "Commands must lie in [-1, 1]")
    return actions


def past_knee_a0(actions, kind="executed"):
    """(mask, side): mask is |a0| > KNEE per row; side is -1 (W-), +1 (W+) or 0, as int8. -0.0 counts as 0."""
    a0 = _executed(actions, kind)[..., 0].astype(np.float64)
    mask = np.abs(a0) > KNEE
    side = np.where(mask, np.where(a0 < 0, -1, 1), 0).astype(np.int8)
    return mask, side


def depth(actions, kind="executed"):
    """d = |a0| - KNEE in float64 (negative inside the support)."""
    return np.abs(_executed(actions, kind)[..., 0].astype(np.float64)) - KNEE


def in_deep(actions, kind="executed"):
    """d >= 0.10, inclusive, on either side. R_deep- is in_deep with side -1 (see in_deep_minus)."""
    return depth(actions, kind) >= DEEP_DEPTH


def in_deep_minus(actions, kind="executed"):
    executed = _executed(actions, kind)
    return in_deep(executed) & (executed[..., 0] < 0)


def load_g(observations):
    """Load g = float64(obs10) + float64(obs11): the current plus the first wave number. It never depends on actions."""
    obs = np.asarray(observations)
    require(obs.dtype == np.float32 and obs.ndim >= 1 and obs.shape[-1] in (13, 14),
            "Load needs float32 observations of shape (..., 13 or 14)")
    g = obs[..., 10].astype(np.float64) + obs[..., 11].astype(np.float64)
    require(bool(np.all(np.isfinite(g))), "Load must be finite")
    return g


def load_band(g):
    """0 low (g < 0.4), 1 mid (0.4 <= g < 1.0), 2 deep (g >= 1.0), as uint8; lower edges inclusive."""
    g = np.asarray(g, np.float64)
    require(bool(np.all(np.isfinite(g))), "Load must be finite")
    return np.where(g < LOAD_MID, 0, np.where(g < LOAD_DEEP, 1, 2)).astype(np.uint8)


def receipt():
    return {"module": "knee_region.py", "sha256": SOURCE_SHA256, "knee": KNEE, "knee_bits": hex(KNEE_BITS),
            "deep_depth": DEEP_DEPTH, "load_edges": [LOAD_MID, LOAD_DEEP], "slots": dict(SLOTS)}


def _boundary_fixture():
    knee = np.float32(KNEE)
    values = [knee, np.nextafter(knee, np.float32(1)), np.nextafter(knee, np.float32(0)), np.float32(1),
              np.float32(0), np.float32(-0.0), np.float32(0.5)]
    values += [np.float32(SLOTS[name]) for name in SLOTS]
    a0 = np.asarray(values + [-v for v in values], np.float32)
    actions = np.zeros((len(a0), 4), np.float32)
    actions[:, 0] = a0
    actions[:, 1:] = np.float32(0.7)
    return actions


def _import_time_assertion():
    env = load_environment()
    fixture = _boundary_fixture()
    canonical = env.canonical.stalled(fixture, env.Physics(stall=True, stall_slope=3.0))[..., 0]
    require(np.array_equal(past_knee_a0(fixture)[0], canonical), "Region predicate disagrees with canonical stalled()")
    expected = {"s05": (True, False), "s10": (True, True), "s15": (True, True), "s20": (True, True),
                "p10": (True, False), "edge_knee": (False, False), "edge_700": (False, False)}
    for name, (in_w, in_deep_minus_value) in expected.items():
        row = np.zeros((1, 4), np.float32)
        row[0, 0] = SLOTS[name]
        require(bool(past_knee_a0(row)[0][0]) == in_w and bool(in_deep_minus(row)[0]) == in_deep_minus_value,
                "Slot table disagrees with Part 0: " + name)
    require(float(np.float32(-0.9)) == -0.8999999761581421 and not in_deep_minus(
        np.asarray([[np.float32(-0.9), 0, 0, 0]], np.float32))[0], "float32(-0.9) must not be in R_deep-")


_import_time_assertion()
