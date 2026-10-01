"""Exact checks of the Table 2 reproduction: python -m unittest experiments.wbcp.test_reproduce_table2"""

import math
import unittest

import numpy as np

from calibration.wbcp import calibrate
from experiments.wbcp import reproduce_table1 as table1
from experiments.wbcp import reproduce_table2 as table2

POSTERIOR = dict(alpha=0.4, beta=0.95, draws=400)


class CountLossTests(unittest.TestCase):
    def test_one_outcome_per_unit_is_the_miscoverage_calibrator(self):
        # K = 1 makes the exceedance fraction the miscoverage indicator; with sorted input both
        # draw the same exponentials in the same order, so the results agree bit for bit.
        for seed in range(20):
            data = np.random.default_rng(seed)
            n = int(data.integers(1, 60))
            scores = np.sort(data.exponential(size=n))
            weights = data.lognormal(size=n)
            wbar = float(data.lognormal())
            ours = table2.calibrate_count(scores[:, None], np.random.default_rng(seed), weights, wbar, **POSTERIOR)
            theirs = calibrate(scores, np.random.default_rng(seed), weights, wbar, **POSTERIOR)
            self.assertEqual(ours, (theirs.threshold, theirs.lambda_hat, theirs.lambda_hpd))

    def test_posterior_matches_a_direct_count(self):
        # Eq. (6)-(7) evaluated literally: L+ at every grid point for every draw.
        for seed in range(10):
            data = np.random.default_rng(100 + seed)
            n, k = int(data.integers(2, 15)), int(data.integers(1, 5))
            outcomes = data.exponential(size=(n, k))
            weights, wbar = data.lognormal(size=n), float(data.lognormal())
            dep, lam_hat, lam_hpd = table2.calibrate_count(outcomes, np.random.default_rng(seed), weights, wbar,
                                                           **POSTERIOR)
            exponentials = np.random.default_rng(seed).standard_exponential((POSTERIOR["draws"], n + 1))
            masses = exponentials * np.append(weights, wbar)
            v = masses / masses.sum(axis=1, keepdims=True)
            grid = np.sort(outcomes.ravel())
            loss = (outcomes[None, :, :] > grid[:, None, None]).mean(axis=2)  # grid x units
            l_plus = v[:, :n] @ loss.T + v[:, n:]  # draws x grid
            credible = (l_plus <= 0.4).mean(axis=0) >= 0.95
            self.assertEqual(lam_hpd, grid[credible.argmax()] if credible.any() else math.inf)
            empirical = loss @ (weights / weights.sum())
            self.assertEqual(lam_hat, grid[(empirical <= 0.4).argmax()] if (empirical <= 0.4).any() else math.inf)
            self.assertEqual(dep, max(lam_hat, lam_hpd))

    def test_weighted_crc_matches_a_direct_scan(self):
        for seed in range(30):
            data = np.random.default_rng(200 + seed)
            n, k = int(data.integers(1, 20)), int(data.integers(1, 5))
            outcomes = data.exponential(size=(n, k))
            weights, wbar = data.lognormal(size=n), float(data.lognormal(sigma=2))
            grid = np.sort(outcomes.ravel())
            loss = (outcomes[None, :, :] > grid[:, None, None]).mean(axis=2)
            risk = (loss @ weights + wbar) / (weights.sum() + wbar)
            expected = grid[(risk <= 0.4).argmax()] if (risk <= 0.4).any() else math.inf
            self.assertEqual(table2.weighted_crc(outcomes, weights, wbar, 0.4), expected)

    def test_exact_ties_count_as_covered(self):
        # n = 10, K = 4, uniform weights: the empirical risk reaches 0.4 exactly at the 24th outcome.
        outcomes = np.arange(1.0, 41.0).reshape(10, 4)
        _, lam_hat, _ = table2.calibrate_count(outcomes, np.random.default_rng(0), np.ones(10), 1.0, **POSTERIOR)
        self.assertEqual(lam_hat, 24.0)
        self.assertEqual(table2._settle(np.full(40, 0.25), 0.4), 23)

    def test_rcps_rounding(self):
        data = np.random.default_rng(3)
        for n in (10, 37, 250):
            scores = data.exponential(size=n)
            # K = 1: n R_hat is an integer, both roundings are the Table 1 rule.
            j = table1.rcps_index(n, 0.1, 0.05)
            expected = np.sort(scores)[j] if j is not None else math.inf
            for rounding in ("ceil", "floor"):
                self.assertEqual(table2.rcps(scores[:, None], 0.1, 0.05, rounding), expected)
            outcomes = data.exponential(size=(n, 4))
            self.assertLessEqual(table2.rcps(outcomes, 0.4, 0.05, "floor"), table2.rcps(outcomes, 0.4, 0.05, "ceil"))

    def test_rcps_bentkus_count_at_four_outcomes(self):
        # Grid 1..40: at grid value v, 40 - v of the 40 outcomes exceed it, so n R_hat = (40 - v) / 4.
        # Hoeffding alone certifies 2 exceedances (p = 0.036) but not 3 (p = 0.064). At 3 the Bentkus
        # term is e P(Bin(10, 0.4) <= ceil(3/4) = 1) = 0.126 or e P(Bin(10, 0.4) <= floor(3/4) = 0) = 0.016.
        outcomes = np.arange(1.0, 41.0).reshape(10, 4)
        self.assertEqual(table2.rcps(outcomes, 0.4, 0.05, "ceil"), 38.0)
        self.assertEqual(table2.rcps(outcomes, 0.4, 0.05, "floor"), 37.0)

    def test_risk_curve_matches_simulation(self):
        data = np.random.default_rng(4)
        for gamma in (0.0, 1.0, 2.0):
            x = data.normal(gamma, 1.0, 400_000)
            t = data.standard_exponential(x.size) / table2.rate(x, math.log(2), -0.019)
            curve = table2.risk_curve(gamma, math.log(2), -0.019)
            for lam in (0.3, 1.0, 2.5):
                self.assertAlmostEqual(curve(lam), np.mean(t > lam), delta=0.004)


if __name__ == "__main__":
    unittest.main()
