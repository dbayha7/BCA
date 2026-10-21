import unittest

import numpy as np

from experiments.thrusterchain_diagnostics.finite import (
    FiniteConfig, FiniteMDP, cap_policy, evaluate_policy, hidden_kernel_certificate,
    occupancy_ratio, occupancies, solve_optimal, support_ratio,
)


class FiniteTests(unittest.TestCase):
    def test_kernel_is_stochastic_and_context_fixed(self):
        mdp = FiniteMDP()
        np.testing.assert_allclose(mdp.P.sum(axis=-1), 1)
        self.assertTrue(np.all(mdp.P >= 0))
        for s in range(mdp.n_states):
            wrong = ((mdp.state_load != mdp.state_load[s])
                     | (mdp.state_regime != mdp.state_regime[s]))
            self.assertTrue(np.all(mdp.P[s, :, wrong] == 0))
        for value in [-10.0, -1.27, 0.0, 0.25, 1.99, 10.0]:
            mass = mdp.quantize(value)
            self.assertAlmostEqual(mass @ mdp.positions, np.clip(value, -2, 2))
            self.assertLessEqual(np.count_nonzero(mass), 2)

    def test_backward_induction_and_forward_return_agree(self):
        mdp = FiniteMDP(FiniteConfig(horizon=5))
        optimum = solve_optimal(mdp)
        evaluated = evaluate_policy(mdp, optimum.policy)
        np.testing.assert_allclose(optimum.V, evaluated.V, atol=1e-13)
        np.testing.assert_allclose(optimum.Q, evaluated.Q, atol=1e-13)
        uniform = evaluate_policy(mdp, np.ones(mdp.n_actions) / mdp.n_actions)
        self.assertTrue(np.all(optimum.V >= uniform.V - 1e-12))
        occupancy = occupancies(mdp, optimum.policy)
        np.testing.assert_allclose(occupancy.state.sum(-1), 1)
        np.testing.assert_allclose(occupancy.state_action.sum((1, 2)), 1)
        forward = sum(mdp.gamma ** t * np.sum(occupancy.state_action[t] * mdp.R)
                      for t in range(mdp.horizon))
        self.assertAlmostEqual(forward, optimum.start_value(mdp.initial_distribution()), places=12)
        np.testing.assert_allclose(optimum.Q[-1], mdp.R)
        np.testing.assert_allclose(optimum.V[-1], 0)

    def test_high_action_effect_reverses_by_observed_regime(self):
        mdp = FiniteMDP()
        optimum = solve_optimal(mdp)
        # At position 1.5 and high load, -1 versus the -knee action.
        p = int(np.argmin(abs(mdp.positions - 1.5)))
        stall = mdp.state_index(p, load=1, regime=0)
        monotone = mdp.state_index(p, load=1, regime=1)
        self.assertLess(optimum.Q[0, stall, 0], optimum.Q[0, stall, 1])
        self.assertGreater(optimum.Q[0, monotone, 0], optimum.Q[0, monotone, 1])
        self.assertTrue(mdp.critical_mask()[stall, 0])
        self.assertTrue(mdp.critical_mask()[monotone, 0])

    def test_coarse_grid_does_not_imply_global_continuous_stall_dominance(self):
        mdp = FiniteMDP()
        # The continuous proof replaces full command by its stalled thrust.
        # That command is not one of this finite companion's five actions.
        replacement = float(mdp.thrust(1.0, regime=0))
        self.assertFalse(np.any(np.isclose(mdp.actions, replacement)))
        chosen = np.argmax(solve_optimal(mdp).policy, axis=-1)
        stall_choices = chosen[:, mdp.state_regime == 0]
        self.assertTrue(np.any(~mdp.common_action_mask[stall_choices]))

    def test_horizon_one_and_policy_validation(self):
        mdp = FiniteMDP(FiniteConfig(horizon=1))
        result = solve_optimal(mdp)
        np.testing.assert_allclose(result.V[0], mdp.R.max(-1))
        with self.assertRaises(ValueError):
            evaluate_policy(mdp, np.zeros(mdp.n_actions))
        with self.assertRaises(ValueError):
            evaluate_policy(mdp, np.full(mdp.n_actions, np.nan))
        with self.assertRaises(ValueError):
            occupancies(mdp, np.ones(mdp.n_actions) / mdp.n_actions, np.zeros(mdp.n_states))

    def test_action_masks_and_matched_caps(self):
        mdp = FiniteMDP()
        constrained = solve_optimal(mdp, mdp.common_action_mask)
        self.assertTrue(np.all(constrained.policy[:, :, ~mdp.common_action_mask] == 0))
        np.testing.assert_allclose(constrained.V, evaluate_policy(mdp, constrained.policy).V)
        raw = np.array([.2, .1, .3, .1, .3])
        capped = cap_policy(mdp, raw)
        np.testing.assert_allclose(capped[0, 0], [0, .3, .3, .4, 0])
        aware = cap_policy(mdp, raw, regime_aware=True)
        np.testing.assert_allclose(aware[0, mdp.state_index(0, regime=0)], capped[0, 0])
        np.testing.assert_allclose(aware[0, mdp.state_index(0, regime=1)], raw)
        with self.assertRaises(ValueError):
            solve_optimal(mdp, np.zeros(mdp.n_actions, dtype=bool))
        with self.assertRaises(ValueError):
            cap_policy(FiniteMDP(hidden_world="stall"), raw, regime_aware=True)

    def test_zero_support_is_not_finite_or_clipped(self):
        result = support_ratio([0, 1, 0, .5], [0, 0, .2, .25])
        np.testing.assert_array_equal(result.unsupported, [False, True, False, False])
        self.assertTrue(np.isnan(result.ratio[0]))
        self.assertTrue(np.isnan(result.ratio[1]))
        np.testing.assert_allclose(result.ratio[2:], [0, 2])
        self.assertEqual(result.target_mass_unsupported, 1)
        mdp = FiniteMDP()
        policy = np.ones(mdp.n_actions) / mdp.n_actions
        mixed = occupancy_ratio(mdp, policy, policy,
                                mdp.initial_distribution("overlap"),
                                mdp.initial_distribution("zero"))
        self.assertGreater(mixed.target_mass_unsupported, 0)
        thin = occupancy_ratio(mdp, policy, policy,
                               mdp.initial_distribution("overlap"),
                               mdp.initial_distribution("thin"))
        self.assertFalse(thin.unsupported.any())
        self.assertGreater(np.nanmax(thin.ratio), 20)

    def test_hidden_world_full_kernel_certificate(self):
        certificate = hidden_kernel_certificate()
        self.assertTrue(certificate["trajectory_laws_equal"])
        self.assertGreater(certificate["probe_kernel_max_difference"], 0)
        stall = FiniteMDP(hidden_world="stall")
        monotone = FiniteMDP(hidden_world="monotone")
        self.assertEqual(stall.observations.shape[1], 2)
        np.testing.assert_array_equal(stall.observations, monotone.observations)
        policy = stall.common_action_mask / stall.common_action_mask.sum()
        np.testing.assert_array_equal(occupancies(stall, policy).state,
                                      occupancies(monotone, policy).state)
        np.testing.assert_array_equal(stall.critical_mask(), monotone.critical_mask())


if __name__ == "__main__":
    unittest.main()
