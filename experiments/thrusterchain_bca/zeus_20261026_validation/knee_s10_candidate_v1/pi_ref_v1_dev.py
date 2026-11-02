"""pi_ref: the shared, frozen continuation for policy-evaluation mode (ood_definition_v4 DRAFT r2, section 3.2).

The finite-horizon discounted LQR controller of the canonical ThrusterChain plant, every command clipped to +-KNEE.
- Model: x' = A x + B u with (A, B) = canonical `matrices(physics)` (identical in both worlds; inside +-KNEE the
  thrust equals the command in both, so pi_ref never stalls and its value is the same law in stall3 and nostall).
- Cost per step: sum of the ten float coordinates squared + 0.3 |u|^2 (the canonical reward, negated); the current
  and wave coordinates carry no cost but enter the gains (feed-forward). Discount 0.98, horizon 100, no terminal cost.
- Gains: backward Riccati recursion, P_100 = 0; for t = 99..0, S = R + g B'PB, K_t = g S^-1 B'P A,
  P_t = Q + g A'P (A - B K_t). Command u_t = -K_t x, then float32, then clipped to +-KNEE (all four commands).
- Observation: 14 float32 numbers, the 13 state numbers and remaining time (100 - t) / 100; t is recovered as
  100 - round(100 x obs[13]). The gains act on the float32 observation, as a deployed actor would.
No checkpoint, no learned parameter, no randomness. `GAINS_SHA256` pins the gains bit for bit.
"""
import hashlib

import numpy as np

import worlds
from core import require

HORIZON, GAMMA, THROTTLE_COST = 100, 0.98, 0.3
KNEE = worlds.KNEE


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
        _K = gains()
    return _K


def gains_sha256():
    return hashlib.sha256(np.ascontiguousarray(_gains()).tobytes()).hexdigest()


def act(obs):
    """float32 (n, 14) observations -> float32 (n, 4) commands, clipped to +-KNEE."""
    obs = np.asarray(obs, np.float32)
    require(obs.ndim == 2 and obs.shape[1] == 14, "pi_ref takes (n, 14) observations")
    t = HORIZON - np.rint(obs[:, 13].astype(np.float64) * HORIZON).astype(np.int64)
    require(bool(np.all((t >= 0) & (t < HORIZON))), "Remaining time outside the horizon")
    K = _gains()[t]                                                     # (n, 4, 13)
    u = -np.einsum("nij,nj->ni", K, obs[:, :13].astype(np.float64))
    return np.clip(u.astype(np.float32), np.float32(-KNEE), np.float32(KNEE))
