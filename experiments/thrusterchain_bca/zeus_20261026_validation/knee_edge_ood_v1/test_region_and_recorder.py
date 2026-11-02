"""4.6 tests: predicate boundaries and properties, load-band edges, slot table, recorder. Synthetic arrays only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import numpy as np

import knee_region as R
import recorder
import region_independent as I

KNEE32 = np.float32(R.KNEE)


def rows(a0, side=0.0):
    out = np.zeros((len(a0), 4), np.float32)
    out[:, 0] = np.asarray(a0, np.float32)
    out[:, 1:] = np.float32(side)
    return out


class Predicate(unittest.TestCase):
    def both(self, actions, kind="executed"):
        primary = R.past_knee_a0(actions, kind)[0]
        if kind == "executed":
            self.assertTrue(np.array_equal(primary, I.agree(actions)))
        return primary

    def test_boundaries(self):
        up, down = np.nextafter(KNEE32, np.float32(1)), np.nextafter(KNEE32, np.float32(0))
        cases = [(KNEE32, False), (-KNEE32, False), (up, True), (-up, True), (down, False), (-down, False),
                 (np.float32(1), True), (np.float32(-1), True), (np.float32(0.0), False), (np.float32(-0.0), False)]
        got = self.both(rows([c for c, _ in cases]))
        self.assertEqual(got.tolist(), [e for _, e in cases])
        requested = np.zeros((3, 4))
        requested[:, 0] = [0.8, 0.80000001, 0.8000000119]
        self.assertFalse(np.any(R.past_knee_a0(requested, "requested")[0]))

    def test_raises(self):
        for bad in (rows([np.nan]), rows([np.inf]), rows([0.5], side=1.5), rows([1.5])):
            with self.assertRaises(ValueError):
                R.past_knee_a0(bad)
            with self.assertRaises(ValueError):
                I.past_knee_a0(bad) if hasattr(I, "past_knee_a0") else I.agree(bad)
        with self.assertRaises(ValueError):
            R.past_knee_a0(rows([0.5]).astype(np.float64))

    def test_properties(self):
        sweep = np.unique(np.linspace(0.79, 0.81, 4001).astype(np.float32))
        pos, neg = self.both(rows(sweep)), self.both(rows(-sweep))
        self.assertTrue(np.array_equal(pos, neg))                       # symmetric in sign
        self.assertTrue(np.all(np.diff(pos.astype(int)) >= 0))          # monotone in |a0|
        self.assertTrue(np.array_equal(pos, self.both(rows(sweep, side=-0.69))))   # unaffected by a1..a3
        batch = rows(sweep[:12]).reshape(3, 4, 4)
        self.assertEqual(R.past_knee_a0(batch)[0].shape, (3, 4))
        self.assertTrue(np.array_equal(R.past_knee_a0(batch)[0].reshape(-1), R.past_knee_a0(rows(sweep[:12]))[0]))

    def test_slots_and_deep(self):
        self.assertTrue(R.in_deep_minus(rows([R.SLOTS["s10"]]))[0])
        self.assertFalse(R.in_deep_minus(rows([np.float32(-0.9)]))[0])
        self.assertFalse(R.in_deep_minus(rows([R.SLOTS["s05"]]))[0])
        self.assertTrue(R.in_deep_minus(rows([R.SLOTS["s15"], R.SLOTS["s20"]])).all())
        self.assertFalse(R.in_deep_minus(rows([R.SLOTS["p10"]]))[0])
        self.assertAlmostEqual(R.depth(rows([R.SLOTS["s10"]]))[0], 0.10000002384185791, places=15)

    def test_load_band_edges(self):
        # 1.0 is reachable from float32 pairs: 0.5 + 0.5 lands on it, 1.0 + (-2**-53 as float32 is 0) does not, so
        # use (1.0, -tiny) where tiny is the float32 nearest 2**-53 below zero's neighbour scale.
        def g_of(c, w):
            obs = np.zeros((1, 14), np.float32)
            obs[0, 10], obs[0, 11] = np.float32(c), np.float32(w)
            return R.load_g(obs)
        g = g_of(0.5, 0.5)
        self.assertEqual(g[0], 1.0)
        self.assertEqual(int(R.load_band(g)[0]), 2)
        below = g_of(1.0, -np.float32(2.0 ** -53))
        self.assertEqual(below[0], np.nextafter(1.0, -np.inf))
        self.assertEqual(int(R.load_band(below)[0]), 1)
        # 0.4 cannot be reached exactly by any float32 pair (shown by the independent author's whole-range search),
        # so the edge rule is tested on the float64 values themselves, and on the nearest float32 pairs either side.
        self.assertEqual(R.load_band([0.4, np.nextafter(0.4, -np.inf)]).tolist(), [1, 0])
        c = np.float32(0.25)
        for w in (np.float32(0.15), np.nextafter(np.float32(0.15), np.float32(0)), np.nextafter(np.float32(0.15), np.float32(1))):
            g = g_of(c, w)
            self.assertEqual(int(R.load_band(g)[0]), 1 if g[0] >= 0.4 else 0)


class Recorder(unittest.TestCase):
    def obs_at(self, g_values):
        obs = np.zeros((len(g_values), 14), np.float32)
        obs[:, 10] = np.asarray(g_values, np.float32)
        return obs

    def test_five_uniforms_fixed_order_both_branches(self):
        block = recorder.episode_draws(99, 3)
        rng = recorder.generator(99, "actions", 3)
        self.assertTrue(np.array_equal(block, np.stack([rng.random(5) for _ in range(100)])))
        low, high = self.obs_at([0.2])[0], self.obs_at([0.8])[0]
        for u in block[:20]:                      # every branch reads all five; a1..a3 depend only on w1..w3
            a_low, a_high = recorder.requested_command(low, u)[0], recorder.requested_command(high, u)[0]
            self.assertTrue(np.array_equal(a_low[1:], a_high[1:]))

    def test_one_mixture_uniform_per_episode(self):
        values = [recorder.mixture_component(5, e) for e in range(4000)]
        self.assertTrue(0.13 < np.mean(values) < 0.17)
        self.assertEqual(recorder.mixture_component(5, 17), recorder.mixture_component(5, 17))

    def test_density_and_side_bounds(self):
        rng = np.random.Generator(np.random.PCG64(3))
        for g in (0.2, 0.4, 0.9, 1.05):
            obs = self.obs_at([g])[0]
            for u in rng.random((300, 5)):
                requested, _, logp = recorder.requested_command(obs, u)
                band = 0 if g < 0.4 else 1
                a0 = requested[0]
                p0 = (4.9 * (-0.9 <= a0 <= -0.7) + 0.01) if band else 0.5
                self.assertLessEqual(abs(logp - (np.log(p0) + 3 * np.log(1 / 1.4))), 1e-14)
                executed = requested.astype(np.float32).astype(np.float64)
                self.assertTrue(np.all(np.abs(executed[1:]) <= 0.7))

    def test_frequencies_binomial(self):
        n = 20000
        uniforms = np.random.Generator(np.random.PCG64(8)).random((n, 5))
        high = np.array([recorder.requested_command(self.obs_at([0.8])[0], u)[0][0] for u in uniforms])
        low = np.array([recorder.requested_command(self.obs_at([0.2])[0], u)[0][0] for u in uniforms])
        for value, p in ((np.mean((high >= -0.9) & (high <= -0.7)), 0.98 + 0.02 * 0.1), (np.mean(low < -0.8), 0.1)):
            half = 3.29 * np.sqrt(p * (1 - p) / n)          # 99.9% two-sided normal bound
            self.assertLess(abs(value - p), half)

    def test_twins_identical_requests(self):
        a, b = self.obs_at([0.75]), self.obs_at([0.75])
        b[0, :10] = 3.0                            # different float positions, same load
        for u in np.random.Generator(np.random.PCG64(9)).random((50, 5)):
            self.assertTrue(np.array_equal(recorder.requested_command(a[0], u)[0], recorder.requested_command(b[0], u)[0]))


if __name__ == "__main__":
    unittest.main()
