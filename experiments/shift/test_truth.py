"""CPU tests of the exact truth (no learning; the MLP actor is only initialised, never trained; every number is a test
fixture, not a finding).

Checks: the Bellman identity within 1e-9; 1,000 vs 2,000 steps within 1e-6; the no-stall closed form within 1e-10
relative; autodiff slopes match central differences within 1e-6 in the no-stall world; the float64-cast MLP's actions
match the float32 actor's within 1e-5; the JAX rollout matches the numpy rollout within 1e-12 (linear, normal-driving
and MLP policies); the attribution terms sum to the return difference within 1e-8 and vanish for a policy against
itself; TD3's smoothed level by Monte Carlo agrees with its closed form in the no-stall world's linear sub-world; Q is
identical in both worlds for commands and continuations at or below 80%; the per-row formulas give known answers
(a critic equal to the truth has zero over-preference, zero pull past 80% and zero advantage error); the points
anchors give 0 and 100 and the reference controllers' exact bank returns reproduce the design's table.
JAX_PLATFORMS=cpu python -m unittest experiments.shift.test_truth
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402


def small_starts(n, seed):
    """Starts where the textbook policy never saturates (current <= 0.3, wave <= 0.2, float SD 0.2)."""
    return TC.draw_starts(n, np.random.default_rng(seed), dict(current=(0.0, 0.3), wave=(0.0, 0.2), kick=0.2))


class ExactQTest(unittest.TestCase):
    def setUp(self):
        self.stall_aware = RC.stall_aware()
        self.x = TC.bank_starts("recorded", 120)
        self.a = np.clip(self.stall_aware(self.x) + 0.4 * np.random.default_rng(0).standard_normal((120, 4)), -1, 1)

    def test_bellman_and_truncation(self):
        self.assertLessEqual(TR.bellman_residual(self.stall_aware, self.x, self.a), 1e-9)
        q1 = TR.q_value(self.stall_aware, self.x, self.a, horizon=1000)
        q2 = TR.q_value(self.stall_aware, self.x, self.a, horizon=2000)
        self.assertLessEqual(np.abs(q1 - q2).max(), 1e-6)

    def test_closed_form_in_the_no_stall_world(self):
        out = TR.closed_form_check(n=200)
        self.assertTrue(out["continuation_inside"])
        self.assertLess(out["max_continuation_command"], 1.0)
        self.assertLessEqual(out["max_rel"], 1e-10)
        self.assertGreater(out["first_command_clipped_share"], 0.0)   # reported apart; harmless (closed form at the
        self.assertGreater(out["first_command_raw_max"], 1.0)         # same clipped first command)

    def test_closed_form_guard_detects_a_saturating_continuation(self):
        out = TR.closed_form_check(n=30, horizon=300, law=dict(current=(1.0, 1.5), wave=(0.5, 1.0), kick=2.0))
        self.assertFalse(out["continuation_inside"])
        self.assertGreater(out["max_continuation_command"], 1.0)

    def test_jax_matches_numpy(self):
        for pol in (self.stall_aware, RC.hand_only(), RC.rising_thrust()):
            for physics in (TC.STALL, TC.NO_STALL):
                qn = TR.q_value(pol, self.x, self.a, physics)
                qj = TR.q_value_jax(pol, self.x, self.a, physics)
                self.assertLessEqual(np.abs(qn - qj).max(), 1e-12)
        q32 = TR.q_value_jax(self.stall_aware, self.x, self.a, horizon=600, x64=False)
        q64 = TR.q_value_jax(self.stall_aware, self.x, self.a, horizon=600)
        self.assertLessEqual(np.abs(q32 - q64).max() / np.abs(q64).mean(), 1e-4)
        self.assertGreater(np.abs(q32 - q64).max(), 0.0)    # really float32
        import jax.numpy as jnp

        self.assertEqual(jnp.ones(2).dtype, jnp.float32)   # x64 stays off outside the truth calls

    def test_slopes_match_central_differences(self):
        K = RC.textbook_gain()
        pol = RC.LinearController(K, None, "unclipped textbook gain")
        x = small_starts(40, 3)
        a = np.clip(pol(x) + 0.15 * np.random.default_rng(4).standard_normal((40, 4)), -0.9, 0.9)
        g = TR.action_slope(pol, x, a, TC.NO_STALL)
        h = 1e-4
        fd = np.stack([(TR.q_value(pol, x, a + h * np.eye(4)[j], TC.NO_STALL)
                        - TR.q_value(pol, x, a - h * np.eye(4)[j], TC.NO_STALL)) / (2 * h) for j in range(4)], 1)
        self.assertGreater(np.abs(g).max(), 0.05)
        self.assertLessEqual(np.abs(g - fd).max(), 1e-6)

    def test_slope_at_an_exact_zero_command(self):
        """thrust_jax's autodiff slope is 1 at a command of exactly 0 (not 0, as sign()'s derivative would give), so
        action_slope matches central differences at rows whose command or continuation holds an exact 0."""
        pol = self.stall_aware
        x = TC.bank_starts("recorded", 12)
        a = np.clip(pol(x), -0.7, 0.7)
        a[:, 2] = 0.0
        h = 1e-6
        for physics in (TC.STALL, TC.NO_STALL):
            g = TR.action_slope(pol, x, a, physics, horizon=400)
            fd = np.stack([(TR.q_value(pol, x, a + h * np.eye(4)[j], physics, 400)
                            - TR.q_value(pol, x, a - h * np.eye(4)[j], physics, 400)) / (2 * h) for j in range(4)], 1)
            self.assertLessEqual(np.abs(g - fd).max(), 1e-5)
            self.assertGreater(np.abs(g[:, 2]).max(), 0.05)

    def test_worlds_identical_at_or_below_80(self):
        a = np.clip(self.a, -0.8, 0.8)
        q_stall = TR.q_value(self.stall_aware, self.x, a, TC.STALL)
        q_no_stall = TR.q_value(self.stall_aware, self.x, a, TC.NO_STALL)
        self.assertTrue(np.array_equal(q_stall, q_no_stall))
        a32 = np.clip(self.a, -0.8, 0.8).astype(np.float32).astype(np.float64)   # float32(0.8) is the knee itself
        self.assertTrue(np.array_equal(TR.q_value(self.stall_aware, self.x, a32, TC.STALL),
                                       TR.q_value(self.stall_aware, self.x, a32, TC.NO_STALL)))
        hi = a.copy()
        hi[:, 0] = -1.0
        self.assertFalse(np.array_equal(TR.q_value(self.stall_aware, self.x, hi, TC.STALL),
                                        TR.q_value(self.stall_aware, self.x, hi, TC.NO_STALL)))

    def test_random_policy_refused(self):
        streamed = TR.recorder_on_bank(n=120)
        with self.assertRaises(TypeError):
            TR.q_value(streamed, self.x, self.a)


class MLPTest(unittest.TestCase):
    def test_float64_actor_matches_float32_actor(self):
        import jax

        from algorithms import td3_bc as T

        actor = T.Actor(obs_dim=13, action_dim=4, max_action=1.0)
        params = actor.init(jax.random.PRNGKey(0), np.zeros((1, 13), np.float32))
        mean = (0.1 * np.random.default_rng(2).standard_normal(13)).astype(np.float32)
        std = (0.5 + np.abs(np.random.default_rng(3).standard_normal(13))).astype(np.float32)
        mlp = TR.MLPPolicy(params, mean, std)
        x = TC.bank_starts("big_kicks", 400)
        a32 = np.asarray(actor.apply(params, (x.astype(np.float32) - mean) / std))
        self.assertLessEqual(np.abs(mlp(x) - a32).max(), 1e-5)
        xs = x[:60]
        a = np.clip(mlp(xs) + 0.2, -1, 1)
        gap = TR.q_value(mlp, xs, a, horizon=300) - TR.q_value_jax(mlp, xs, a, horizon=300)
        self.assertLessEqual(np.abs(gap).max(), 1e-12)
        g = TR.action_slope(mlp, xs[:8], a[:8], horizon=200)
        self.assertEqual(g.shape, (8, 4))
        self.assertTrue(np.all(np.isfinite(g)))


class AttributionTest(unittest.TestCase):
    def test_terms_sum_to_return_difference(self):
        x0 = TC.bank_starts("strong_current", 150)
        hand, stall_aware = RC.hand_only(), RC.stall_aware()
        out = TR.attribution(hand, stall_aware, x0)
        self.assertLessEqual(out["max_sum_error"], 1e-8)
        j_new, j_ref = TR.rollout(hand, x0).J, TR.rollout(stall_aware, x0).J
        self.assertLessEqual(np.abs(out["terms"].sum(1) - (j_new - j_ref)).max(), 1e-8)
        self.assertEqual(out["terms"].shape, (150, 100))
        same = TR.attribution(stall_aware, stall_aware, x0)
        self.assertLessEqual(np.abs(same["terms"]).max(), 1e-9)


class SmoothedLevelTest(unittest.TestCase):
    def test_monte_carlo_matches_closed_form(self):
        K = RC.textbook_gain()
        pol = RC.LinearController(K, None, "unclipped textbook gain")
        x = small_starts(30, 5)
        a = pol(x)
        mean, se = TR.smoothed_q(pol, x, a, TC.NO_STALL, paths=64, horizon=600)
        cf = TR.closed_form_smoothed_q(K, x, a)
        z = (mean - cf) / se
        self.assertLessEqual(np.abs(z).max(), 4.5)
        self.assertLess(abs(z.mean()), 3.0)
        level_gap = TR.closed_form_smoothed_q(K, x, a) - TR.closed_form_q(K, x, a)
        self.assertTrue(np.all(level_gap < 0))   # target noise lowers the level (makes |Q| larger)


class LabelTest(unittest.TestCase):
    def test_known_answers(self):
        rng = np.random.default_rng(6)
        t_pi, t_a = rng.normal(size=50), rng.normal(size=50)
        self.assertTrue(np.all(TR.over_preference(t_pi, t_a, t_pi, t_a) == 0))
        self.assertTrue(np.all(TR.advantage_error(t_a, t_pi, t_a, t_pi) == 0))
        self.assertTrue(np.all(TR.pull_past_80(t_a, t_pi, t_a, t_pi) == 0))
        bump = np.where(np.arange(50) < 10, 2.0, 0.0)
        d = TR.over_preference(t_pi + bump, t_a, t_pi, t_a)
        self.assertTrue(np.allclose(d, bump))
        self.assertTrue(np.all(TR.over_preference(t_pi - 1.0, t_a, t_pi, t_a) == 0))
        g = rng.normal(size=(50, 4))
        self.assertTrue(np.allclose(TR.caution_benefit(g, g), -np.sum(g * g, 1)))

    def test_noise_map(self):
        x = TC.bank_starts("recorded", 40)
        a = RC.hand_only()(x)
        n1 = TR.noise_map(x, a, horizon=200)
        self.assertTrue(np.array_equal(n1, TR.noise_map(x, a, horizon=200)))
        self.assertTrue(np.all(n1 >= 0))
        self.assertTrue(np.all(TR.noise_map(x, a, horizon=200, draws=1) == 0))


class ReturnsTest(unittest.TestCase):
    def test_anchors_and_design_table(self):
        an = TR.anchors()
        expected = dict(recorded=(-79.8, -26.4), strong_current=(-253.9, -103.8), big_waves=(-106.8, -36.2),
                        big_kicks=(-263.2, -163.5))
        for s, (jo, jt) in expected.items():
            self.assertAlmostEqual(an[s]["recording"], jo, delta=0.05)
            self.assertAlmostEqual(an[s]["top"], jt, delta=0.05)
            self.assertAlmostEqual(float(TR.points(an[s]["recording"], an[s])), 0.0, places=9)
            self.assertAlmostEqual(float(TR.points(an[s]["top"], an[s])), 100.0, places=9)
        hand = {s: float(TR.points(TR.bank_rollout(RC.hand_only(), s).J.mean(), an[s])) for s in TC.SITUATIONS}
        lin = {s: float(TR.points(TR.bank_rollout(RC.rising_thrust(), s).J.mean(), an[s])) for s in TC.SITUATIONS}
        for s, v in dict(recorded=78.4, strong_current=71.3, big_waves=72.5, big_kicks=58.0).items():
            self.assertAlmostEqual(hand[s], v, delta=0.05)
        for s, v in dict(recorded=83.6, strong_current=-54.1, big_waves=63.8, big_kicks=44.2).items():
            self.assertAlmostEqual(lin[s], v, delta=0.05)
        no_stall = TR.anchors(TC.NO_STALL)
        self.assertAlmostEqual(no_stall["recorded"]["recording"], -74.2, delta=0.05)
        self.assertAlmostEqual(no_stall["recorded"]["top"], -25.7, delta=0.05)


if __name__ == "__main__":
    unittest.main()
