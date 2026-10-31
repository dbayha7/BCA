"""Data-level known-answer check (protocol 4.5b).

On stall3 rows in W, thruster 0's thrust is reconstructed from consecutive rows of the dataset's float64 `states`:
B_0 thrust_0 = x' - A x - sum_{k=1..3} B_k a_k (a1..a3 never pass the knee, so their thrust is the command), solved
by the pseudo-inverse of B_0. It must equal the literal T3(a0) to 1e-9 and differ from the slope-1.5 thrust (for
example T3(-1) = -0.2, slope 1.5 gives -0.5). Float32 observations are never used: their rounding would break 1e-9.
"""
import numpy as np

import worlds
from core import require
from knee_region import past_knee_a0

TOLERANCE = 1e-9


def reconstruct_thrust0(states, actions):
    states, actions = np.asarray(states), np.asarray(actions)
    require(states.dtype == np.float64 and states.shape[-1] == 13 and states.shape[-2] == actions.shape[-2] + 1,
            "Need float64 states (..., T+1, 13) and actions (..., T, 4)")
    env = worlds.load_environment()
    A, B = env.matrices(worlds.physics("stall3"))
    x, nxt = states[..., :-1, :], states[..., 1:, :]
    residual = nxt - x @ A.T - actions[..., 1:].astype(np.float64) @ B[:, 1:].T
    b0 = B[:, 0]
    return residual @ b0 / (b0 @ b0)


def identity_rows(states, actions, tag):
    """Row-level literal identity on W rows: the recorded next state equals the step taken with a0 <- T(a0) (inside
    the knee, where both worlds agree). Returns the largest absolute deviation; a slope mix-up breaks it."""
    states, actions = np.asarray(states), np.asarray(actions)
    w = past_knee_a0(actions)[0]
    image = actions.astype(np.float64).copy()
    image[..., 0] = worlds.literal_thrust0(image[..., 0], tag)
    env = worlds.load_environment()
    nxt, _ = env.dynamics(states[..., :-1, :][w], image[w], worlds.physics("nostall"))
    return float(np.max(np.abs(nxt - states[..., 1:, :][w])))


def check(states, actions, world_tag):
    """(ok, report) for the W rows of a dataset claiming `world_tag` = stall3."""
    require(world_tag == "stall3", "The data-level check certifies stall3 data")
    w = past_knee_a0(np.asarray(actions))[0]
    require(bool(np.any(w)), "No W rows to check")
    thrust = reconstruct_thrust0(states, actions)[w]
    a0 = np.asarray(actions)[..., 0].astype(np.float64)[w]
    t3, t15 = worlds.literal_thrust0(a0, "stall3"), worlds.literal_thrust0(a0, "stall15_identity")
    err3 = float(np.max(np.abs(thrust - t3)))
    gap15 = float(np.min(np.abs(thrust - t15)))
    ok = err3 <= TOLERANCE and gap15 > TOLERANCE
    return ok, {"rows": int(np.sum(w)), "max_error_vs_T3": err3, "min_gap_vs_T15": gap15, "formula": worlds.LITERAL_FORMULAS["stall3"]}
