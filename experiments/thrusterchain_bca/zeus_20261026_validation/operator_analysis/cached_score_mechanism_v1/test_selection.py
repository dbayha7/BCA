"""Synthetic score/tolerance fixtures; no experiment data is read."""

import unittest

import numpy as np

from run_cached_comparison import select, value_epsilon


class SelectionFixtures(unittest.TestCase):
    def test_unique_best_keeps_original_index(self):
        self.assertEqual(select([0, 1, -1, 3, 2, -4, 0, 0, 0])["index"], 3)

    def test_exact_tie_uses_first_without_dropping_ambiguity(self):
        out = select([0, 4, 4, -1, 0, 0, 0, 0, 0])
        self.assertEqual(out["index"], 1)
        self.assertEqual(out["near_best_indices"], [1, 2])
        self.assertTrue(out["near_tie"])

    def test_near_tie_does_not_replace_first_with_exact_argmax(self):
        out = select([1 - 1e-6, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(out["index"], 0)
        self.assertEqual(out["exact_argmax_index"], 1)

    def test_negative_scores_use_magnitude_of_maximum(self):
        out = select([-1.0000015, -1, -9, -9, -9, -9, -9, -9, -9])
        self.assertEqual(out["index"], 0)
        self.assertTrue(out["near_tie"])

    def test_absolute_tolerance_applies_at_zero(self):
        out = select([-0.5e-6, 0, -1, -1, -1, -1, -1, -1, -1])
        self.assertEqual(out["index"], 0)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            select([np.nan] + [0] * 8)

    def test_return_tolerance_symmetric_and_not_score_tolerance(self):
        a = np.array([-10., 0., 1.])
        b = np.array([1., 2., -100.])
        np.testing.assert_allclose(value_epsilon(a, b), [11e-6, 3e-6, 101e-6], rtol=0, atol=1e-19)
        np.testing.assert_array_equal(value_epsilon(a, b), value_epsilon(b, a))


if __name__ == "__main__":
    unittest.main()
