"""Tests that the cloning weight of Amendment 3 reaches the learner's objective. Needs JAX.

    python -I -B -m unittest discover -s experiments/gridworld_critical_v1 -p "test_alpha.py" -v
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import run_published_alpha as A  # noqa: E402
import learner as L  # noqa: E402


class Alpha(unittest.TestCase):
    def tearDown(self):
        A.set_alpha(A.PUBLISHED_ALPHA)

    def test_the_learner_starts_at_the_published_value(self):
        self.assertEqual(L.ALPHA, A.PUBLISHED_ALPHA)
        self.assertTrue(A.objective_uses_alpha(A.PUBLISHED_ALPHA))

    def test_the_objective_uses_the_value_that_is_set(self):
        for alpha in (1.0, 0.5, 0.25, 0.1):
            A.set_alpha(alpha)
            self.assertEqual(L.ALPHA, alpha)
            self.assertTrue(A.objective_uses_alpha(alpha), alpha)

    def test_the_check_fails_for_a_value_that_is_not_set(self):
        A.set_alpha(0.5)
        self.assertFalse(A.objective_uses_alpha(2.5))
        self.assertFalse(A.objective_uses_alpha(0.25))


if __name__ == "__main__":
    unittest.main()
