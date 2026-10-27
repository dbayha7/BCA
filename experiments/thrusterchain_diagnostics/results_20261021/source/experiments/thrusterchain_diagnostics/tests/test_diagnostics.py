import json
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.thrusterchain_diagnostics.diagnostics import (
    MethodOutput, evaluate_method, interval_metrics, prediction_metrics, ranking_metrics,
)
from experiments.thrusterchain_diagnostics.finite import (
    FiniteConfig, FiniteMDP, evaluate_policy, occupancies, solve_optimal,
)


def toy_mdp():
    # Safe earns 1 and stays safe; risky earns 3, then enters an absorbing
    # failure state with zero reward. Horizon two, discount one.
    return SimpleNamespace(
        horizon=2, n_states=2, n_actions=2, gamma=1.,
        P=np.array([[[1., 0.], [0., 1.]], [[0., 1.], [0., 1.]]]),
        R=np.array([[1., 3.], [0., 0.]]),
        initial_distribution=lambda: np.array([1., 0.]),
        critical_mask=lambda: np.array([[False, True], [False, False]]),
    )


def action_policy(action):
    policy = np.zeros((2, 2, 2))
    policy[..., action] = 1
    return policy


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.mdp = toy_mdp()
        self.safe, self.risky = action_policy(0), action_policy(1)
        self.failure = np.array([False, True])

    def test_hand_computed_return_failure_and_regret(self):
        r = evaluate_method(self.mdp, MethodOutput(self.risky),
                            baseline_policy=self.safe, failure_mask=self.failure)
        self.assertEqual(r['returns']['proposed'], 3)
        self.assertEqual(r['returns']['deployed'], 3)
        self.assertEqual(r['returns']['optimal'], 4)
        self.assertEqual(r['returns']['regret'], 1)
        self.assertEqual(r['returns']['harm_vs_baseline'], 0)
        self.assertEqual(r['failure']['any_visit_probability'], 1)
        self.assertAlmostEqual(r['failure']['expected_fraction_of_state_times'], 2 / 3)
        self.assertIsNone(r['errors']['q'])
        self.assertIsNone(r['ranking'])
        self.assertIsNone(r['calibration']['q'])
        json.dumps(r, allow_nan=False)

    def test_perfect_predictions_and_explicit_target(self):
        dp = evaluate_policy(self.mdp, self.safe)
        r = evaluate_method(self.mdp, MethodOutput(
            self.safe, q_values=dp.Q, v_values=dp.V,
            q_intervals=np.stack([dp.Q, dp.Q], axis=-1),
            v_intervals=np.stack([dp.V, dp.V], axis=-1)))
        self.assertEqual(r['errors']['q']['rmse'], 0)
        self.assertEqual(r['errors']['v']['rmse'], 0)
        self.assertEqual(r['ranking']['greedy_action_regret'], 0)
        self.assertEqual(r['calibration']['q']['query_coverage'], 1)
        self.assertEqual(r['calibration']['v']['mean_width'], 0)
        other = evaluate_method(self.mdp, MethodOutput(self.safe, q_values=dp.Q,
                                                       value_target='optimal'))
        self.assertGreater(other['errors']['q']['rmse'], 0)

    def test_biased_predictions_and_interval_misses(self):
        dp = evaluate_policy(self.mdp, self.safe)
        r = evaluate_method(self.mdp, MethodOutput(
            self.safe, q_values=dp.Q + 2, v_values=dp.V[:-1] - 3,
            q_intervals=np.stack([dp.Q + 1, dp.Q + 3], axis=-1)))
        self.assertEqual(r['errors']['q']['bias'], 2)
        self.assertEqual(r['errors']['q']['rmse'], 2)
        self.assertEqual(r['errors']['v']['bias'], -3)
        self.assertEqual(r['calibration']['q']['query_coverage'], 0)
        self.assertEqual(r['calibration']['q']['mean_width'], 2)

    def test_all_refusal_executes_fallback_and_retains_errors(self):
        dp = evaluate_policy(self.mdp, self.risky)
        r = evaluate_method(self.mdp, MethodOutput(
            self.risky, q_values=dp.Q + 2, refusal=np.ones((2, 2))),
            fallback_policy=self.safe, baseline_policy=self.safe, failure_mask=self.failure)
        self.assertEqual(r['returns']['proposed'], 3)
        self.assertEqual(r['returns']['deployed'], 2)
        self.assertEqual(r['failure']['any_visit_probability'], 0)
        self.assertEqual(r['errors']['q']['rmse'], 2)
        self.assertIsNone(r['refusal']['accepted_q_error'])
        self.assertEqual(r['refusal']['decision_refusal_rate'], 1)
        self.assertEqual(r['refusal']['query_acceptance_rate'], 0)
        self.assertEqual(r['refusal']['fallback_return'], 2)
        self.assertAlmostEqual(r['decisions']['refused_one_step_opportunity_vs_fallback'], 1.5)

    def test_all_accept_and_fractional_refusal(self):
        dp = evaluate_policy(self.mdp, self.risky)
        r = evaluate_method(self.mdp, MethodOutput(
            self.risky, q_values=dp.Q, refusal=np.zeros((2, 2))), fallback_policy=self.safe)
        self.assertEqual(r['returns']['deployed'], 3)
        self.assertEqual(r['refusal']['query_acceptance_rate'], 1)
        self.assertEqual(r['refusal']['accepted_q_error']['rmse'], 0)
        mixed = evaluate_method(self.mdp, MethodOutput(self.risky, refusal=np.full((2, 2), .5)),
                                fallback_policy=self.safe, failure_mask=self.failure)
        self.assertAlmostEqual(mixed['returns']['deployed'], 3)
        self.assertAlmostEqual(mixed['failure']['any_visit_probability'], .75)

    def test_unsupported_mass_differs_from_hit_probability(self):
        logging = occupancies(self.mdp, self.safe).state_action
        mixed = .5 * (self.safe + self.risky)
        counts = np.zeros_like(logging, dtype=int)
        counts[0, 0, 0] = 10
        r = evaluate_method(self.mdp, MethodOutput(mixed, q_values=evaluate_policy(self.mdp, mixed).Q),
                            logging_mass=logging, retained_counts=counts)
        support = r['support']['population']
        self.assertAlmostEqual(support['unsupported_deployed_mass'], .625)
        self.assertAlmostEqual(support['unsupported_action_hit_probability'], .75)
        self.assertAlmostEqual(support['occupancy_total_variation'], .625)
        self.assertEqual(r['support']['counts']['retained_transitions'], 10)
        self.assertEqual(r['support']['counts']['critical_retained_transitions'], 0)
        self.assertEqual(r['support']['counts']['critical_unobserved_entries'], 2)
        self.assertEqual(r['errors']['q_supported']['rmse'], 0)
        self.assertEqual(r['errors']['q_unsupported']['rmse'], 0)
        json.dumps(r, allow_nan=False)

    def test_empty_population_logging(self):
        r = evaluate_method(self.mdp, MethodOutput(self.safe), logging_mass=np.zeros((2, 2, 2)))
        support = r['support']['population']
        self.assertEqual(support['unsupported_deployed_mass'], 1)
        self.assertEqual(support['unsupported_action_hit_probability'], 1)
        self.assertIsNone(support['occupancy_total_variation'])
        self.assertIsNone(support['max_ratio_on_supported_entries'])
        json.dumps(r, allow_nan=False)

    def test_initial_failure_and_baseline_harm(self):
        failed = evaluate_method(self.mdp, MethodOutput(self.safe),
                                 initial_distribution=np.array([0., 1.]), failure_mask=self.failure)
        self.assertEqual(failed['failure']['any_visit_probability'], 1)
        harmed = evaluate_method(self.mdp, MethodOutput(self.safe), baseline_policy=self.risky)
        self.assertEqual(harmed['returns']['harm_vs_baseline'], 1)
        self.assertGreater(harmed['decisions']['baseline_continuation_mean_disadvantage'], 0)

    def test_weighted_metrics_and_ranking_ties(self):
        self.assertEqual(prediction_metrics([2, 100], [0, 0], [1, 0])['max_absolute_error'], 2)
        self.assertIsNone(prediction_metrics([2], [0], [0]))
        ranking = ranking_metrics([[0, 0]], [[1, 0]])
        self.assertEqual(ranking['pairwise_disagreement'], .5)
        self.assertEqual(ranking['greedy_action_regret'], .5)
        self.assertIsNone(ranking_metrics([[0, 1]], [[1, 1]])['pairwise_disagreement'])
        self.assertIsNone(interval_metrics([[0, 1]], [0], [0]))

    def test_invalid_inputs_and_no_query_weight_mutation(self):
        with self.assertRaisesRegex(ValueError, 'explicit fallback'):
            evaluate_method(self.mdp, MethodOutput(self.safe, refusal=np.ones((2, 2))))
        with self.assertRaises(ValueError):
            evaluate_method(self.mdp, MethodOutput(self.safe, refusal=np.full((2, 2), 2)),
                            fallback_policy=self.risky)
        with self.assertRaises(ValueError):
            evaluate_method(self.mdp, MethodOutput(self.safe, value_target='unclear'))
        with self.assertRaises(ValueError):
            evaluate_method(self.mdp, MethodOutput(np.zeros_like(self.safe)))
        with self.assertRaises(ValueError):
            interval_metrics([[2, 1]], [0])
        with self.assertRaises(ValueError):
            prediction_metrics([np.nan], [0])
        with self.assertRaises(ValueError):
            evaluate_method(self.mdp, MethodOutput(self.safe), retained_counts=np.full((2, 2, 2), .5))
        weights = np.arange(1., 9.).reshape(2, 2, 2)
        expected = weights.copy()
        evaluate_method(self.mdp, MethodOutput(self.safe), query_weights=weights)
        np.testing.assert_array_equal(weights, expected)

    def test_real_finite_companion_integration(self):
        mdp = FiniteMDP(FiniteConfig(horizon=3, n_positions=5))
        optimal = solve_optimal(mdp)
        mass = occupancies(mdp, optimal.policy).state_action
        r = evaluate_method(mdp, MethodOutput(optimal.policy, q_values=optimal.Q, v_values=optimal.V),
                            baseline_policy=optimal.policy, logging_mass=mass,
                            retained_counts=np.zeros_like(mass, dtype=int))
        self.assertAlmostEqual(r['returns']['regret'], 0)
        self.assertEqual(r['errors']['q']['rmse'], 0)
        self.assertEqual(r['support']['population']['unsupported_deployed_mass'], 0)
        self.assertAlmostEqual(r['support']['population']['occupancy_total_variation'], 0)
        json.dumps(r, allow_nan=False)

    def test_extreme_finite_inputs_preserve_json_contract(self):
        dp = evaluate_policy(self.mdp, self.safe)
        r = evaluate_method(self.mdp, MethodOutput(self.safe, q_values=dp.Q + 2),
                            query_weights=np.full((2, 2, 2), 1e308),
                            logging_mass=np.full((2, 2, 2), 1e308))
        self.assertEqual(r['errors']['q']['rmse'], 2)
        json.dumps(r, allow_nan=False)
        self.assertEqual(ranking_metrics([[0, 0]], [[1e308, -1e308]])['pairwise_disagreement'], .5)
        self.assertEqual(prediction_metrics([2, 1e308], [0, 0], [1, 0])['rmse'], 2)
        with self.assertRaisesRegex(ValueError, 'interval widths'):
            interval_metrics([[-1e308, 1e308]], [0])

    def test_harmful_action_probability_keeps_mixture_cancellation_visible(self):
        # The baseline's equal mixture at the last step has mean reward 2.
        # Safe is harmful relative to that mean even when the proposed mixture
        # is identical to baseline and the policy-distribution disadvantage is 0.
        mixed = .5 * (self.safe + self.risky)
        r = evaluate_method(self.mdp, MethodOutput(mixed), baseline_policy=mixed)
        self.assertEqual(r['decisions']['baseline_continuation_harmful_policy_distribution_rate'], 0)
        self.assertGreater(r['decisions']['baseline_continuation_harmful_action_probability'], 0)


if __name__ == '__main__':
    unittest.main()
