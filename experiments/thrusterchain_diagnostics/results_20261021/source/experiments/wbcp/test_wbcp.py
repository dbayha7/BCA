"""Exact-identity checks for calibration/wbcp.py; NumPy and SciPy only, no RL imports.

python -m unittest experiments.wbcp.test_wbcp
"""
import math
import unittest
from unittest import mock

import numpy as np
from scipy import stats

from calibration import wbcp
from calibration.wbcp import calibrate


def rng(seed=0):
    return np.random.default_rng(seed)


class WBCPTests(unittest.TestCase):
    def test_uniform_weights_are_bq_cp(self):
        # Dir(1,...,1) over n+1 atoms: the first j sorted masses sum to Beta(j, n+1-j),
        # so Pr(crossing <= s_(j)) = Pr(Beta(j, n+1-j) >= 1-alpha) (Snell and Griffiths).
        n, alpha, draws = 9, 0.1, 200_000
        scores = np.arange(1.0, n + 1)
        result = calibrate(scores, rng(1), alpha=alpha, draws=draws)
        for j in range(1, n + 1):
            expected = stats.beta.sf(1 - alpha, j, n + 1 - j)
            observed = np.mean(result.crossings <= scores[j - 1])
            # The 5/draws slack keeps the normal approximation honest for tail probabilities near 1e-8.
            self.assertLess(abs(observed - expected), 5 * math.sqrt(expected * (1 - expected) / draws) + 5 / draws)
        self.assertAlmostEqual(np.mean(np.isinf(result.crossings)), (1 - alpha) ** n, delta=0.004)

    def test_single_weighted_atom_matches_closed_form(self):
        # n=1: U = E1/(E1+E2) is Uniform(0,1) and the draw crosses iff
        # w E1 >= (1-alpha)(w E1 + wbar E2), i.e. with probability alpha w / ((1-alpha) wbar + alpha w).
        alpha, w, wbar, draws = 0.4, 3.0, 1.0, 400_000
        result = calibrate([2.5], rng(2), [w], wbar, alpha=alpha, draws=draws)
        expected = alpha * w / ((1 - alpha) * wbar + alpha * w)
        self.assertAlmostEqual(np.mean(np.isfinite(result.crossings)), expected, delta=0.003)

    def test_exponential_sampler_equals_tilted_dirichlet(self):
        # Theorem 6: V_i w_i / sum_j V_j w_j with V ~ Dir(1,...,1) has the law of w_i E_i / sum_j w_j E_j.
        generator = rng(3)
        n, alpha, draws = 30, 0.1, 100_000
        scores = np.sort(generator.normal(size=n) ** 2)
        weights = generator.lognormal(sigma=1.0, size=n)
        wbar = 1.7
        dirichlet = generator.dirichlet(np.ones(n + 1), size=draws)
        reference = wbcp.crossings(scores, weights, wbar, dirichlet, alpha)
        ours = calibrate(scores, rng(4), weights, wbar, alpha=alpha, draws=draws).crossings
        for level in np.append(scores, np.inf):
            p, q = np.mean(reference <= level), np.mean(ours <= level)
            self.assertLess(abs(p - q), 5 * math.sqrt(max(p * (1 - p), 1e-4) * 2 / draws))

    def test_empirical_selection_excludes_the_test_atom(self):
        # Eq. (1) normalizes over calibration weights only: cutoff 0.7*4 = 2.8 gives s_(3).
        # Counting the test mass 2 would need 0.7*6 = 4.2 > 4 and give +inf.
        self.assertEqual(calibrate([1, 2, 3, 4], rng(), np.ones(4), 2.0, alpha=0.3, draws=10).lambda_hat, 3.0)
        self.assertEqual(calibrate([4, 3, 2, 1], rng(), [7, 1, 1, 1], 1.0, alpha=0.4, draws=10).lambda_hat, 4.0)

    def test_empirical_selection_keeps_exact_ties(self):
        # R_w(lambda) = alpha counts as covered; float rounding used to skip these by one score.
        for n, alpha, weight, expected in ((10, 0.1, 1.0, 9.0), (25, 0.44, 1.0, 14.0), (10, 0.7, 1.0, 3.0),
                                           (200, 0.1, 0.37, 180.0), (5, 0.2, 0.37, 4.0)):
            with self.subTest(n=n, alpha=alpha, weight=weight):
                result = calibrate(np.arange(1.0, n + 1), rng(), np.full(n, weight), weight, alpha=alpha, draws=10)
                self.assertEqual(result.lambda_hat, expected)

    def test_hpd_is_the_smallest_beta_credible_grid_point(self):
        # Algorithm 1 line 6 by direct count, at a beta*M that is not a whole number.
        generator = rng(5)
        scores, weights = generator.exponential(size=5000), generator.lognormal(size=5000)
        result = calibrate(scores, rng(6), weights, 1.3, alpha=0.1, beta=0.95, draws=999)
        grid = np.unique(result.crossings)
        credible = np.array([np.mean(result.crossings <= point) for point in grid]) >= 0.95
        self.assertEqual(result.lambda_hpd, grid[np.argmax(credible)])
        self.assertTrue(np.all(np.diff(result.crossings) >= 0))

    def test_clamp_binds_when_the_posterior_selection_is_lower(self):
        result = calibrate(rng(14).exponential(size=300), rng(15), alpha=0.1, beta=0.2, draws=500)
        self.assertLess(result.lambda_hpd, result.lambda_hat)
        self.assertEqual(result.threshold, result.lambda_hat)

    def test_crossing_counts_equality_as_covered(self):
        # Mass (1, 0.5) plus test mass 0.5: total 2, so covering s_(1) reaches exactly (1-alpha)*total.
        crossing = wbcp.crossings(np.array([1.0, 2.0]), np.ones(2), 1.0, np.array([[1.0, 0.5, 0.5]]), 0.5)
        self.assertEqual(crossing[0], 1.0)

    def test_huge_weights_do_not_overflow(self):
        generator = rng(16)
        scores, weights = generator.exponential(size=1000), generator.lognormal(size=1000)
        base = calibrate(scores, rng(17), weights, 1.5, draws=300)
        huge = calibrate(scores, rng(17), 1e306 * weights, 1.5e306, draws=300)
        self.assertEqual((huge.threshold, huge.lambda_hat), (base.threshold, base.lambda_hat))
        self.assertAlmostEqual(huge.n_eff, base.n_eff)

    def test_common_weight_scale_cancels(self):
        generator = rng(7)
        scores, weights = generator.exponential(size=50), generator.lognormal(size=50)
        base = calibrate(scores, rng(8), weights, 2.0, draws=500)
        scaled = calibrate(scores, rng(8), 1e3 * weights, 2e3, draws=500)
        np.testing.assert_allclose(scaled.crossings, base.crossings)
        self.assertEqual((scaled.threshold, scaled.lambda_hat), (base.threshold, base.lambda_hat))
        self.assertAlmostEqual(scaled.n_eff, base.n_eff)

    def test_input_order_is_irrelevant(self):
        generator = rng(9)
        scores, weights = generator.exponential(size=40), generator.lognormal(size=40)
        permutation = generator.permutation(40)
        base = calibrate(scores, rng(10), weights, 1.0, draws=300)
        permuted = calibrate(scores[permutation], rng(10), weights[permutation], 1.0, draws=300)
        np.testing.assert_array_equal(base.crossings, permuted.crossings)

    def test_chunking_does_not_change_the_posterior(self):
        scores = rng(11).exponential(size=100)
        whole = calibrate(scores, rng(12), draws=257)
        with mock.patch.object(wbcp, "_CHUNK_ELEMENTS", 3 * 101):
            chunked = calibrate(scores, rng(12), draws=257)
        np.testing.assert_array_equal(whole.crossings, chunked.crossings)

    def test_dominant_test_atom_means_nothing_is_certifiable(self):
        result = calibrate(np.arange(1.0, 51), rng(13), np.ones(50), 1e6, draws=200)
        self.assertTrue(math.isinf(result.threshold) and not result.certified)
        self.assertTrue(math.isfinite(result.lambda_hat))
        self.assertTrue(math.isinf(result.sigma_post))

    def test_kish_effective_sample_size(self):
        self.assertAlmostEqual(calibrate([1, 2, 3], rng(), [1, 1, 2], 1.0, draws=5).n_eff, 16 / 6)
        self.assertAlmostEqual(calibrate([1, 2, 3], rng(), draws=5).n_eff, 3.0)

    def test_rejects_invalid_inputs(self):
        for kwargs in (dict(scores=[]), dict(scores=[1.0, np.nan]), dict(scores=[1.0], test_mass=1.0),
                       dict(scores=[1.0], weights=[1.0]), dict(scores=[1.0], weights=[-1.0], test_mass=1.0),
                       dict(scores=[1.0], weights=[0.0], test_mass=1.0), dict(scores=[1.0], alpha=1.0),
                       dict(scores=[1.0], draws=0), dict(scores=[1.0], draws=10.0),
                       dict(scores=[1.0, 2.0], weights=[1.0], test_mass=1.0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calibrate(rng=rng(), **kwargs)
        with self.assertRaises(ValueError):
            calibrate([1.0], 0)


if __name__ == "__main__":
    unittest.main()
