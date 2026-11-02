"""pi_ref: the shared, frozen continuation for policy-evaluation mode (ood_definition_v4 DRAFT, section 3.2). Version 2.

The finite-horizon discounted LQR controller of the canonical ThrusterChain plant, with every command kept inside
the data both arms hold (the bootstrap-support rule). Version 1 (pi_ref_v1_dev.py) clipped every command to +-KNEE.
The v3 review (wf_e1964e57-92d) found that, at the next states of the fitting rows, 78% of its commands set a side
thruster past the recorder's +-0.7, and 36-51% of its high-load a0 lay above -0.6, where the recorder holds almost
no rows. Both arms' critics would then extrapolate just to evaluate pi_ref.

- Model: x' = A x + B u with (A, B) = canonical `matrices(physics)`. These are identical in both worlds, and inside
  +-KNEE the thrust equals the command in both, so pi_ref never stalls.
- Cost per step: the ten float coordinates squared + 0.3 |u|^2 (the canonical reward, negated). The current and wave
  coordinates carry no cost but enter the gains. Discount 0.98, horizon 100, no terminal cost.
- Gains: backward Riccati recursion, P_100 = 0. For t = 99..0: S = R + g B'PB, K_t = g S^-1 B'P A,
  P_t = Q + g A'P (A - B K_t). The raw command is u_t = -K_t x on the float32 observation.
- Support projection (version 2), from the load g of the observation:
  - a1..a3 are clipped to +-SIDE_RANGE (0.7), the recorder's side range;
  - at high load (g >= 0.4), a0 is clipped to [-KNEE, -0.6]. This is the part of the recorder's dense window
    [-1.0, -0.6] that CUTOFF keeps;
  - at low load, a0 is clipped to [-KNEE, KNEE]. The recorder is uniform on [-1, 1] there, and CUTOFF keeps
    |a0| <= KNEE.
  These projections avoid W and respect recorder coordinate ranges. They do not establish empirical joint
  state/action coverage or prevent extrapolation at unsupported states.
- Observation: 14 float32 numbers, the 13 state numbers and remaining time (100 - t) / 100. The time step is recovered
  as t = 100 - round(100 x obs[13]).

No checkpoint, no learned parameter, no randomness. GAINS_SHA256 pins the gains bit for bit (unchanged from v1).
"""
import hashlib

import numpy as np

import worlds
from core import require
from knee_region import load_band, load_g

HORIZON, GAMMA, THROTTLE_COST = 100, 0.98, 0.3
KNEE = worlds.KNEE
SIDE_RANGE = 0.7
HIGH_LOAD_A0 = (-KNEE, -0.6)
GAINS_SHA256 = "7b35523dba2429baba0226069e1e18b6ba5a49be9208ce2490936e91c68be777"


def gains():
    """K (HORIZON, 4, 13) float64."""
    env = worlds.load_environment()
    A, B = env.canonical.matrices(worlds.physics("stall3"))
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    nx, nu = A.shape[0], B.shape[1]
    Q = np.diag([1.0] * 10 + [0.0] * (nx - 10))
    R = THROTTLE_COST * np.eye(nu)
    P = np.zeros((nx, nx))
    K = np.zeros((HORIZON, nu, nx))
    for t in range(HORIZON - 1, -1, -1):
        S = R + GAMMA * B.T @ P @ B
        K[t] = GAMMA * np.linalg.solve(S, B.T @ P @ A)
        P = Q + GAMMA * A.T @ P @ (A - B @ K[t])
        P = 0.5 * (P + P.T)
    require(bool(np.all(np.isfinite(K))), "Non-finite LQR gains")
    return K


_K = None


def _gains():
    global _K
    if _K is None:
        K = gains()
        require(hashlib.sha256(np.ascontiguousarray(K).tobytes()).hexdigest() == GAINS_SHA256,
                "pi_ref gains differ from the pinned GAINS_SHA256")
        _K = K
    return _K


def gains_sha256():
    return hashlib.sha256(np.ascontiguousarray(_gains()).tobytes()).hexdigest()


def raw(obs):
    """The unprojected LQR command (n, 4) float64, for reporting how often the projection binds."""
    obs = np.asarray(obs, np.float32)
    require(obs.ndim == 2 and obs.shape[1] == 14, "pi_ref takes (n, 14) observations")
    t = HORIZON - np.rint(obs[:, 13].astype(np.float64) * HORIZON).astype(np.int64)
    require(bool(np.all((t >= 0) & (t < HORIZON))), "Remaining time outside the horizon")
    return -np.einsum("nij,nj->ni", _gains()[t], obs[:, :13].astype(np.float64))


def act(obs):
    """float32 (n, 14) observations -> float32 (n, 4) commands within the stated coordinate bounds; empirical support is a separate check."""
    obs = np.asarray(obs, np.float32)
    u = raw(obs).astype(np.float32)
    high = load_band(load_g(obs)) >= 1
    lo = np.where(high, np.float32(HIGH_LOAD_A0[0]), np.float32(-KNEE)).astype(np.float32)
    hi = np.where(high, np.float32(HIGH_LOAD_A0[1]), np.float32(KNEE)).astype(np.float32)
    out = np.empty_like(u)
    out[:, 0] = np.clip(u[:, 0], lo, hi)
    out[:, 1:] = np.clip(u[:, 1:], np.float32(-SIDE_RANGE), np.float32(SIDE_RANGE))
    return out
