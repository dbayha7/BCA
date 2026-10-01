"""Checks of the critic-alignment analysis: python -m unittest experiments.wbcp.test_critic_alignment"""

import unittest

import numpy as np

from experiments.wbcp import critic_alignment as C


def synthetic(n, seed, optimism=0.0):
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    q2 = t + rng.normal(scale=0.5, size=n)
    q1 = q2 + optimism * (rng.random(n) < 0.3)  # Q1 optimistic on 30% of rows
    q = np.column_stack([q1, q2])
    return dict(target=t, q=q, q_pi=q.copy(), sigma=np.ones(n), sigma_pi=np.ones(n), episode=np.arange(n))


class AnalysisTests(unittest.TestCase):
    def test_equal_heads_make_the_scores_identical(self):
        heads = synthetic(5000, 0)
        heads["q"][:, 0] = heads["q"][:, 1]
        out = C.analyze(heads, np.random.default_rng(0), banks=20, n=256, draws=200)
        self.assertEqual(out["population"]["hidden_misses"], 0.0)
        self.assertEqual(out["population"]["reverse"], 0.0)
        self.assertAlmostEqual(out["population"]["q1_miscoverage_at_lambda_min"], 0.1, delta=1e-3)
        by = out["banks"]["by_score"]
        self.assertEqual(by["min"]["fail_q1"], by["min"]["fail_min"])
        self.assertEqual(out["logged"]["q1_above_q2"], 0.0)

    def test_optimistic_q1_is_hidden_by_the_min(self):
        heads = synthetic(20000, 1, optimism=3.0)
        out = C.analyze(heads, np.random.default_rng(1), banks=20, n=256, draws=200)
        self.assertGreater(out["population"]["hidden_misses"], 0.05)
        self.assertGreater(out["population"]["q1_miscoverage_at_lambda_min"], 0.15)
        by = out["banks"]["by_score"]
        self.assertGreater(by["min"]["fail_q1"], 0.5)  # calibrating the min misses the actor's head
        self.assertLessEqual(by["q1"]["fail_q1"], by["min"]["fail_q1"])

    def test_quantile_and_miscoverage_conventions(self):
        s = np.arange(1.0, 11.0)
        self.assertEqual(C.upper_quantile(s, 0.1), 9.0)  # one of ten values (10) lies above 9
        np.testing.assert_allclose(C.miscoverage(s, np.array([9.0, 8.5, 10.0])), [0.1, 0.2, 0.0])


if __name__ == "__main__":
    unittest.main()
