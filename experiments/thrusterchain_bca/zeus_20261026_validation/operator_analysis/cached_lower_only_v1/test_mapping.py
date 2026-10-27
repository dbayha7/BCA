"""Synthetic lower-only mapping identity fixtures; no audit inputs."""

import unittest
import numpy as np
from run_lower_only import lower_only


class MappingFixtures(unittest.TestCase):
    def test_lower_higher_equal_and_negative_values(self):
        q = np.array([5., 5., 5., -3., -3.])
        y = np.array([2., 7., 5., -5., -1.])
        score, penalty, error = lower_only(q, y)
        np.testing.assert_array_equal(score, [2., 5., 5., -5., -3.])
        np.testing.assert_array_equal(penalty, [3., 0., 0., 2., 0.])
        self.assertEqual(error, 0)

    def test_ideal_penalty_differs_from_unrestricted_upward_target(self):
        score, _, _ = lower_only([0., 2.], [3., 1.])
        np.testing.assert_array_equal(score, [0., 1.])
        self.assertFalse(np.array_equal(score, [3., 1.]))

    def test_roundoff_identity_and_no_score_increase(self):
        q = np.array([1e8, -1e8, 1e-8, -1e-8, np.pi])
        y = np.array([-1e-8, 1e8, -np.pi, np.pi, np.e])
        score, penalty, error = lower_only(q, y)
        self.assertTrue(np.all(score <= q))
        self.assertTrue(np.all(penalty >= 0))
        self.assertLess(error, 1e-6)

    def test_invalid_inputs_stop(self):
        for q, y in (([1.], [1., 2.]), ([np.nan], [0.]), ([0.], [np.inf])):
            with self.assertRaises(ValueError):
                lower_only(q, y)


if __name__ == "__main__":
    unittest.main()
