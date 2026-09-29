"""Independent arithmetic and finite-population checks; no RL imports."""
import itertools
import math
import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
from reference import build_reference, density_ratios, normalized_scores


def make(scores, weights=None, alpha=0.1, draws=None):
    n = len(scores)
    return build_reference(scores, np.ones(n) if weights is None else weights,
                           np.ones((8, n)) if draws is None else draws,
                           alpha=alpha, credibility=0.95)


class WeightedConformalTests(unittest.TestCase):
    def test_equal_weights_include_finite_sample_query_atom(self):
        for n in (1, 8, 9, 10, 19, 99):
            result = make(np.arange(1, n + 1)).radii([1])
            rank = math.ceil(Fraction(9, 10) * (n + 1))
            expected = float(rank) if rank <= n else math.inf
            self.assertEqual(result.conformal[0], expected)

    def test_weights_act_on_heldout_threshold(self):
        plain = make([1, 2, 3, 4], alpha=0.4).radii([1])
        shifted = make([1, 2, 3, 4], [1, 1, 1, 7], alpha=0.4).radii([1])
        self.assertEqual(plain.conformal[0], 3)
        self.assertEqual(shifted.conformal[0], 4)

    def test_query_weight_changes_threshold_and_can_force_infinity(self):
        r = make([1, 2, 3, 4], alpha=0.4).radii([0, 2, 100])
        np.testing.assert_array_equal(r.conformal, [3, 4, np.inf])
        self.assertTrue(np.all(np.diff(r.query_mass) > 0))
        self.assertTrue(math.isinf(r.full[-1]))

    def test_exact_boundary_and_tiny_nonzero_query(self):
        r = make([1, 2, 3], [1, 2, 3], alpha=0.5).radii([0, 1e-100])
        np.testing.assert_array_equal(r.conformal, [2, 3])

    def test_ties_zero_weights_and_supported_maximum(self):
        ref = make([1, 2, 2, 999], [0, 1, 1, 0], alpha=0.5)
        r = ref.radii([0, 1])
        np.testing.assert_array_equal(r.conformal, [2, 2])
        self.assertEqual(ref.bayesian, 2)

    def test_common_rescaling_including_query(self):
        expected = make([1, 2, 3, 4], [1, 2, 4, 8], 0.3).radii([1, 4, 100])
        for factor in (2.0 ** -900, 2.0 ** 900):
            actual = make([1, 2, 3, 4], np.array([1, 2, 4, 8]) * factor,
                          0.3).radii(np.array([1, 4, 100]) * factor)
            np.testing.assert_array_equal(actual.conformal, expected.conformal)
            np.testing.assert_allclose(actual.query_mass, expected.query_mass)

    def test_joint_permutation_of_scores_weights_and_bootstrap_columns(self):
        rng = np.random.default_rng(2026092901)
        scores, weights = rng.uniform(size=31), rng.uniform(0.1, 5, size=31)
        draws, perm = rng.exponential(size=(128, 31)), rng.permutation(31)
        a = make(scores, weights, draws=draws).radii([0.1, 1, 40])
        b = make(scores[perm], weights[perm], draws=draws[:, perm]).radii([0.1, 1, 40])
        np.testing.assert_array_equal(a.conformal, b.conformal)
        np.testing.assert_array_equal(a.full, b.full)

    def test_bayesian_tilting_is_separate_and_full_retains_floor(self):
        ref = make([1, 2, 3, 4], [1, 1, 1, 7], alpha=0.4,
                   draws=np.array([[10, 1, 1, 1], [1, 1, 1, 10]]))
        np.testing.assert_array_equal(ref.bootstrap_quantiles, [3, 4])
        self.assertEqual(ref.bayesian, 4)
        result = ref.radii([1, 100])
        np.testing.assert_array_equal(result.full, np.maximum(result.conformal, ref.bayesian))

    def test_density_ratio_requires_denominator_and_overlap(self):
        np.testing.assert_array_equal(density_ratios([0.8, 0.2], [0.2, 0.8]), [0.25, 4])
        np.testing.assert_array_equal(density_ratios([0.5, 0.5], [0, 1]), [0, 2])
        for source, target in (([0], [1]), ([0], [0]), ([-1], [1]), ([1], [np.inf])):
            with self.assertRaises(ValueError):
                density_ratios(source, target)
        with self.assertRaises(FloatingPointError):
            density_ratios([1e-300], [1e300])
        with self.assertRaises(FloatingPointError):
            density_ratios([1e300], [1e-300])

    def test_residual_and_interval_arithmetic(self):
        np.testing.assert_array_equal(normalized_scores([3, 7], [1, 3], [2, 2]), [1, 2])
        ref = make([1, 2, 3, 4], [1, 1, 1, 7], alpha=0.4)
        interval = ref.intervals([10], [2], [1])
        np.testing.assert_array_equal(interval.width, [8])
        np.testing.assert_array_equal(interval.lower, [2])
        np.testing.assert_array_equal(interval.upper, [18])
        infinite = ref.intervals([10], [2], [100])
        self.assertEqual(infinite.lower[0], -math.inf)
        self.assertEqual(infinite.upper[0], math.inf)

    def test_freezes_copies_without_mutating_callers(self):
        s, w = np.array([1., 2.]), np.ones(2)
        ref = make(s, w)
        s[0], w[0] = 999, 999
        self.assertEqual(ref.scores[0], 1)
        self.assertEqual(ref.ratios[0], 1)
        with self.assertRaises(ValueError):
            ref.scores[0] = 4
        with self.assertRaises(FrozenInstanceError):
            ref.bayesian = 0

    def test_invalid_inputs_do_not_become_finite_bands(self):
        for scores, weights in (([], []), ([1], [0]), ([np.nan], [1]),
                                ([-1], [1]), ([1], [-1]), ([1], [np.inf]),
                                ([[1]], [1]), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                make(scores, weights)
        for alpha in (0, 1, -0.1, np.nan):
            with self.assertRaises(ValueError):
                make([1, 2], alpha=alpha)
        for draws in (np.zeros((2, 2)), np.ones((1, 2)), np.ones((2, 3)),
                      np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):
                make([1, 2], draws=draws)
        for query in ([], [np.inf], [np.nan], [-1], [[1]]):
            with self.assertRaises(ValueError):
                make([1, 2]).radii(query)
        for scale in ([0], [-1], [np.inf]):
            with self.assertRaises(ValueError):
                make([1, 2]).intervals([0], scale, [1])

    def test_numerical_overflow_is_distinct_from_statistical_infinity(self):
        with self.assertRaises(FloatingPointError):
            normalized_scores([1e308], [-1e308], [1])
        ref = make([1e308, 1e308], alpha=0.5)
        with self.assertRaises(FloatingPointError):
            ref.intervals([0], [1e308], [1])
        with self.assertRaises(FloatingPointError):
            ref.intervals([1e308], [1], [1])

    def test_extreme_finite_masses_do_not_overflow_bootstrap(self):
        ref = make([1, 2], [1e308, 1e308], alpha=0.5,
                   draws=np.full((8, 2), 1e308))
        self.assertEqual(ref.radii([1e308]).conformal[0], 2)
        self.assertTrue(np.isfinite(ref.bayesian))
        self.assertEqual(ref.ess, 2)

    def test_exhaustive_known_ratio_marginal_coverage(self):
        # P(X=j)=1/6, Q(X=j)=j/21; ratio proportional to j.
        # Y=X, fixed center=0 and scale=1. Enumerate every calibration sample
        # and every independent target query; no empirical sampling uncertainty.
        n, alpha = 4, Fraction(3, 10)
        mass_covered = Fraction(0)
        for sample in itertools.product(range(1, 7), repeat=n):
            result = make(sample, sample, float(alpha)).radii(range(1, 7))
            for j, radius in enumerate(result.conformal, start=1):
                if j <= radius:
                    mass_covered += Fraction(j, 21 * 6 ** n)
        self.assertGreaterEqual(mass_covered, 1 - alpha)
        self.assertLessEqual(mass_covered, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
