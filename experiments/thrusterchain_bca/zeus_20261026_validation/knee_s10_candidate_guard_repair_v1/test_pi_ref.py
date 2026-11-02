"""pi_ref tests (Windows, numpy only): the Riccati gains solve the discounted finite-horizon LQ problem, the rollout of
the unprojected controller returns -x'P_0 x, the time index round-trips through the observation, the gains are pinned,
and every projected command lies inside CUTOFF's data."""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent / "knee_edge_ood_v3"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.dont_write_bytecode = True

import numpy as np

import pi_ref
import worlds
from knee_region import load_band, load_g


def plant():
    env = worlds.load_environment()
    A, B = env.canonical.matrices(worlds.physics("stall3"))
    return env, np.asarray(A, np.float64), np.asarray(B, np.float64)


class Gains(unittest.TestCase):
    def test_pinned(self):
        self.assertEqual(pi_ref.gains_sha256(), pi_ref.GAINS_SHA256)

    def test_rollout_value_equals_riccati_value(self):
        """Unprojected u = -K_t x through the canonical dynamics in the no-stall world: return = -x'P_0 x."""
        env, A, B = plant()
        K = pi_ref.gains()
        Q = np.diag([1.0] * 10 + [0.0] * 3)
        P = np.zeros((13, 13))
        for t in range(99, -1, -1):
            P = Q + 0.3 * K[t].T @ K[t] + 0.98 * (A - B @ K[t]).T @ P @ (A - B @ K[t])
        rng = np.random.default_rng(1)
        x = rng.normal(size=(5, 13)) * 0.05                    # small, so no command reaches the clip in [-1, 1]
        total, xs = np.zeros(5), x.copy()
        for t in range(100):
            u = -(K[t] @ xs.T).T
            self.assertLess(float(np.abs(u).max()), 1.0)
            xs, r = env.dynamics(xs, u, worlds.physics("nostall"))
            total += 0.98 ** t * r
        self.assertTrue(np.allclose(total, -np.einsum("ni,ij,nj->n", x, P, x), rtol=1e-10, atol=1e-12))

    def test_first_gain_is_optimal_against_perturbations(self):
        """Perturbing the first command of the unprojected controller never raises the exact LQ value."""
        env, A, B = plant()
        K = pi_ref.gains()
        x0 = np.random.default_rng(2).normal(size=13) * 0.05

        def value(u0):
            xs, total = x0.copy(), 0.0
            for t in range(100):
                u = u0 if t == 0 else -(K[t] @ xs)
                xs, r = env.dynamics(xs, u, worlds.physics("nostall"))
                total += 0.98 ** t * float(r)
            return total
        best = value(-(K[0] @ x0))
        for d in np.eye(4) * 1e-3:
            self.assertGreaterEqual(best, value(-(K[0] @ x0) + d) - 1e-12)
            self.assertGreaterEqual(best, value(-(K[0] @ x0) - d) - 1e-12)


class Projection(unittest.TestCase):
    def test_time_index_round_trip(self):
        for t in range(100):
            obs = np.zeros((1, 14), np.float32)
            obs[0, 13] = np.float32((100 - t) / 100)
            got = 100 - int(np.rint(np.float64(obs[0, 13]) * 100))
            self.assertEqual(got, t)

    def test_every_command_inside_cutoff_support(self):
        rng = np.random.default_rng(3)
        obs = np.concatenate([rng.normal(size=(4000, 10)) * 3, rng.uniform(0, 1.5, (4000, 1)),
                              rng.uniform(-1, 1, (4000, 2)), rng.integers(1, 101, (4000, 1)) / 100], axis=1).astype(np.float32)
        a = pi_ref.act(obs)
        high = load_band(load_g(obs)) >= 1
        self.assertTrue(np.all(a[:, 0] >= np.float32(-pi_ref.KNEE)) and np.all(a[:, 0] <= np.float32(pi_ref.KNEE)))
        self.assertTrue(np.all(a[high, 0] <= np.float32(-0.6)))
        self.assertTrue(np.all(np.abs(a[:, 1:]) <= np.float32(0.7)))
        self.assertTrue(high.any() and (~high).any())


if __name__ == "__main__":
    unittest.main()
