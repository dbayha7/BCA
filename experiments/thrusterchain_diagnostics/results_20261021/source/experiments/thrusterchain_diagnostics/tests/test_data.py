"""Tests of the experiment contracts, not a learner training run."""

import unittest

import numpy as np

from experiments.thrusterchain_diagnostics.data import (
    RetentionSpec, behavior_policy, generate_evaluation_data,
    generate_fixed_state_data, generate_trajectory_data, paired_hidden_control,
    probe_identifiability, rare_probe_control, _sample_with_uniform,
)
from experiments.thrusterchain_diagnostics.finite import FiniteConfig, FiniteMDP


class DataTests(unittest.TestCase):
    def setUp(self):
        self.mdp = FiniteMDP(FiniteConfig(horizon=4))

    def assert_arrays_equal(self, left, right, excluded=()):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            if key not in excluded:
                np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    def test_retention_is_nested_and_does_not_change_complete_physical_logs(self):
        datasets = [generate_trajectory_data(self.mdp, 256, seed=97,
                                             retention=RetentionSpec(mode, 0.20))
                    for mode in ("full", "rare", "zero")]
        full, rare, zero = [dataset.evaluator for dataset in datasets]
        for other in (rare, zero):
            self.assert_arrays_equal(full, other,
                                     excluded=("retention_propensity", "retained"))
        self.assertTrue(np.all(full["retained"]))
        self.assertTrue(np.all(zero["retained"] <= rare["retained"]))
        self.assertTrue(np.all(rare["retained"] <= full["retained"]))
        self.assertGreater(full["critical"].sum(), 0)
        self.assertFalse(np.any(zero["retained"] & zero["critical"]))
        self.assertTrue(np.all(rare["retained"][~rare["critical"]]))
        self.assertGreater(np.count_nonzero(rare["retained"] & rare["critical"]), 0)
        self.assertLess(np.count_nonzero(rare["retained"] & rare["critical"]),
                        np.count_nonzero(rare["critical"]))
        np.testing.assert_array_equal(rare["retention_propensity"],
                                      np.where(rare["critical"], 0.20, 1.0))

    def test_trajectory_rows_are_real_consecutive_dynamics_with_fixed_contexts(self):
        result = generate_trajectory_data(self.mdp, 23, seed=17,
                                          retention=RetentionSpec("zero"))
        rows = result.evaluator
        states = rows["state"].reshape(23, self.mdp.horizon)
        successors = rows["next_state"].reshape(23, self.mdp.horizon)
        np.testing.assert_array_equal(states[:, 1:], successors[:, :-1])
        for context in (self.mdp.state_regime, self.mdp.state_load):
            np.testing.assert_array_equal(context[states],
                                          np.broadcast_to(context[states[:, :1]], states.shape))
            np.testing.assert_array_equal(context[states], context[successors])
        np.testing.assert_array_equal(rows["time"], np.tile(np.arange(self.mdp.horizon), 23))
        self.assertTrue(np.all(self.mdp.P[rows["state"], rows["action"], rows["next_state"]] > 0))
        np.testing.assert_array_equal(rows["reward"], self.mdp.R[rows["state"], rows["action"]])
        self.assertFalse(np.any(rows["terminated"]))
        self.assertEqual(np.count_nonzero(rows["truncated"]), 23)
        self.assertTrue(np.all(np.isnan(rows["state_design_probability"])))

    def test_fixed_state_experiment_has_uniform_action_design_and_separate_episode_ids(self):
        result = generate_fixed_state_data(self.mdp, 3, seed=5)
        rows = result.evaluator
        count = self.mdp.horizon * self.mdp.n_states * self.mdp.n_actions * 3
        self.assertEqual(result.n_complete_rows, count)
        self.assertEqual(len(np.unique(rows["episode_id"])), count)
        counts = np.zeros((self.mdp.horizon, self.mdp.n_states), dtype=int)
        np.add.at(counts, (rows["time"], rows["state"]), 1)
        np.testing.assert_array_equal(counts, np.full_like(counts, self.mdp.n_actions * 3))
        np.testing.assert_array_equal(rows["action_probability"], np.full(count, 1 / self.mdp.n_actions))
        np.testing.assert_array_equal(rows["state_design_probability"],
                                      np.full(count, 1 / (self.mdp.horizon * self.mdp.n_states)))
        self.assertEqual(result.metadata["track"], "fixed_state")

    def test_learner_observation_carries_remaining_time_and_stops_terminal_bootstrap(self):
        datasets = (generate_trajectory_data(self.mdp, 4, seed=15),
                    generate_fixed_state_data(self.mdp, 1, seed=15))
        for dataset in datasets:
            self.assertEqual(dataset.metadata["physical_observation_dimension"], 3)
            self.assertEqual(dataset.metadata["observation_dimension"], 4)
            for rows in (dataset.learner_rows, dataset.evaluator):
                np.testing.assert_array_equal(rows["observation"][:, :-1],
                                              self.mdp.observations[rows["state"]])
                np.testing.assert_array_equal(rows["next_observation"][:, :-1],
                                              self.mdp.observations[rows["next_state"]])
                np.testing.assert_array_equal(rows["observation"][:, -1],
                                              (self.mdp.horizon - rows["time"]) / self.mdp.horizon)
                np.testing.assert_array_equal(rows["next_observation"][:, -1],
                                              (self.mdp.horizon - rows["time"] - 1) / self.mdp.horizon)
                np.testing.assert_array_equal(rows["finite_horizon_done"], rows["truncated"])
                np.testing.assert_array_equal(rows["bootstrap_mask"],
                                              (~rows["truncated"]).astype(float))
                self.assertTrue(np.all(rows["observation"][rows["time"] == 0, -1] == 1.0))
                final = rows["time"] == self.mdp.horizon - 1
                self.assertTrue(np.all(rows["observation"][final, -1] == 1 / self.mdp.horizon))
                self.assertTrue(np.all(rows["next_observation"][final, -1] == 0.0))
                self.assertTrue(np.all(rows["bootstrap_mask"][final] == 0.0))
                self.assertTrue(np.all(rows["bootstrap_mask"][~final] == 1.0))
        one_step = generate_trajectory_data(FiniteMDP(FiniteConfig(horizon=1)), 3)
        self.assertTrue(np.all(one_step.learner_rows["observation"][:, -1] == 1.0))
        self.assertTrue(np.all(one_step.learner_rows["next_observation"][:, -1] == 0.0))
        self.assertTrue(np.all(one_step.learner_rows["finite_horizon_done"]))
        self.assertTrue(np.all(one_step.learner_rows["bootstrap_mask"] == 0.0))

    def test_action_probabilities_are_actual_categorical_masses(self):
        policy = behavior_policy(self.mdp, high_action_mass=0.30)
        result = generate_trajectory_data(self.mdp, 64, policy=policy, seed=22)
        rows = result.evaluator
        np.testing.assert_array_equal(rows["action_probability"],
                                      policy[rows["time"], rows["state"], rows["action"]])
        common_policy = behavior_policy(self.mdp, common_only=True)
        self.assertTrue(np.all(common_policy[..., ~self.mdp.common_action_mask] == 0))
        np.testing.assert_allclose(common_policy.sum(axis=-1), 1.0)
        result = generate_trajectory_data(self.mdp, 64, common_only=True)
        self.assertTrue(np.all(self.mdp.common_action_mask[result.evaluator["action"]]))
        with self.assertRaises(ValueError):
            generate_trajectory_data(self.mdp, 1, policy=policy, common_only=True)

    def test_sampler_never_assigns_roundoff_remainder_to_zero_support(self):
        probabilities = np.array([0.0, 0.5, 0.5 - 1e-13, 0.0])
        self.assertEqual(_sample_with_uniform(probabilities, 0.0), 1)
        self.assertEqual(_sample_with_uniform(probabilities, np.nextafter(1.0, 0.0)), 2)

    def test_holdout_has_independent_namespace_and_unfiltered_evaluation_law(self):
        train = generate_trajectory_data(self.mdp, 64, seed=9,
                                         retention=RetentionSpec("zero"), context_mix="zero")
        evaluation = generate_evaluation_data(self.mdp, 64, seed=9)
        self.assertFalse(np.array_equal(train.evaluator["action"], evaluation.evaluator["action"]))
        self.assertTrue(np.all(evaluation.evaluator["retained"]))
        self.assertTrue(np.all(evaluation.evaluator["split"] == "evaluation"))
        self.assertFalse(np.any(self.mdp.state_load[train.evaluator["state"]] == 1))
        self.assertTrue(np.any(self.mdp.state_load[evaluation.evaluator["state"]] == 1))
        self.assertNotEqual(train.metadata["seed_provenance"]["seed_sequence_entropy"]["start"],
                            evaluation.metadata["seed_provenance"]["seed_sequence_entropy"]["start"])

    def test_learner_export_has_no_oracles_or_aliases_to_evaluator_arrays(self):
        result = generate_fixed_state_data(self.mdp, 1, seed=12,
                                           retention=RetentionSpec("zero"))
        self.assertNotIn("critical", result.learner_rows)
        self.assertNotIn("retained", result.learner_rows)
        self.assertNotIn("transition_probability", result.learner_rows)
        self.assertNotIn("Q", result.learner_rows)
        for key in result.learner_rows:
            self.assertFalse(np.shares_memory(result.learner_rows[key], result.evaluator[key]))
        selected = result.evaluator["retained"]
        for key, values in result.learner_rows.items():
            np.testing.assert_array_equal(values, result.evaluator[key][selected])
        self.assertTrue(np.all(result.learner_rows["retention_propensity"] > 0))

    def test_hidden_worlds_have_equal_full_logs_and_reachable_kernel_certificate(self):
        paired = paired_hidden_control(FiniteConfig(horizon=4), n_episodes=32, seed=6)
        self.assertTrue(all(paired.checks.values()), paired.checks)
        self.assertEqual(paired.stall.metadata, paired.monotone.metadata)
        self.assert_arrays_equal(paired.stall.evaluator, paired.monotone.evaluator)
        self.assert_arrays_equal(paired.stall.learner_rows, paired.monotone.learner_rows)
        for dataset in (paired.stall, paired.monotone):
            self.assertEqual(dataset.learner_rows["observation"].shape[1], 3)
            self.assertEqual(dataset.metadata["physical_observation_dimension"], 2)
            self.assertEqual(dataset.metadata["observation_dimension"], 3)
            public = repr(dataset.metadata) + repr(list(dataset.learner_rows))
            self.assertNotIn("hidden_world", public)
            self.assertNotIn("monotone", public)
            self.assertNotIn("stall", public)
            self.assertTrue(np.all(dataset.learner_rows["split"] == "paired_log"))
        for probes in paired.probes.values():
            self.assertEqual(probes["observation"].shape[1], 3)
            self.assertTrue(np.all(probes["observation"][:, -1] == 1.0))
            self.assertTrue(np.all(probes["next_observation"][:, -1] == 3 / 4))
            self.assertTrue(np.all(probes["bootstrap_mask"] == 1.0))
            self.assertFalse(np.any(probes["finite_horizon_done"]))
            self.assertTrue(np.all(probes["split"] == "paired_probes"))
            self.assertTrue(np.all(probes["retention_propensity"] == 1))
            self.assertTrue(np.all(probes["action_probability"] > 0))
            self.assertTrue(np.all(probes["provenance"] == "finite:paired_probe"))
        np.testing.assert_array_equal(paired.probes["stall"]["state"], paired.probes["monotone"]["state"])
        np.testing.assert_array_equal(paired.probes["stall"]["action"], paired.probes["monotone"]["action"])

    def test_rare_probe_zero_boundary_and_actual_coupled_episodes(self):
        zero = rare_probe_control(FiniteConfig(horizon=4), n_episodes=20, seed=7,
                                  probe_probability=0.0)
        self.assertTrue(zero.report["zero_probe_boundary"])
        self.assertTrue(zero.report["coupled_complete_logs_equal"])
        self.assertTrue(zero.report["safe_kernel_equal"])
        for world in zero.report["per_world"].values():
            self.assertEqual(world["no_differing_kernel_hit_probability_per_episode"], 1.0)
            self.assertEqual(world["at_least_one_differing_kernel_hit_probability"], 0.0)
            self.assertEqual(world["observed_differing_kernel_action_count"], 0)
        rare = rare_probe_control(FiniteConfig(horizon=4), n_episodes=100, seed=7,
                                  probe_probability=0.20)
        self.assertFalse(rare.report["zero_probe_boundary"])
        self.assertTrue(rare.report["learner_metadata_equal"])
        for dataset in (rare.stall, rare.monotone):
            states = dataset.evaluator["state"].reshape(100, 4)
            successors = dataset.evaluator["next_state"].reshape(100, 4)
            np.testing.assert_array_equal(states[:, 1:], successors[:, :-1])
            self.assertNotIn("differing_kernel_action", dataset.learner_rows)
            self.assertNotIn("hidden_world", repr(dataset.metadata))
            self.assertEqual(dataset.learner_rows["observation"].shape[1], 3)
        for world in rare.report["per_world"].values():
            self.assertLess(world["no_differing_kernel_hit_probability_per_episode"], 1.0)
            self.assertGreater(world["at_least_one_differing_kernel_hit_probability"], 0.0)
            self.assertGreater(world["observed_differing_kernel_action_count"], 0)

    def test_probe_probability_dp_matches_one_step_and_availability_increases_with_n(self):
        config = FiniteConfig(horizon=1)
        stall = FiniteMDP(config, hidden_world="stall")
        monotone = FiniteMDP(config, hidden_world="monotone")
        policy = behavior_policy(stall, high_action_mass=0.1)
        different = (np.any(stall.P != monotone.P, axis=-1) | (stall.R != monotone.R))
        expected_no_hit = 1 - np.sum(stall.initial_distribution()[:, None] * policy[0] * different)
        one = probe_identifiability(stall, monotone, policy, n_episodes=1)
        ten = probe_identifiability(stall, monotone, policy, n_episodes=10)
        for name in ("stall", "monotone"):
            self.assertAlmostEqual(one["per_world"][name]["no_differing_kernel_hit_probability_per_episode"],
                                   expected_no_hit)
            self.assertGreater(ten["per_world"][name]["at_least_one_differing_kernel_hit_probability"],
                               one["per_world"][name]["at_least_one_differing_kernel_hit_probability"])
            self.assertAlmostEqual(ten["per_world"][name]["no_differing_kernel_hit_probability_all_episodes"],
                                   expected_no_hit ** 10)
        self.assertEqual(one["per_world"]["stall"], one["per_world"]["monotone"])
        with self.assertRaises(ValueError):
            probe_identifiability(stall, monotone, policy,
                                   initial=stall.initial_distribution() * (1 + 1e-6))

    def test_empty_data_preserves_schema_and_invalid_inputs_raise(self):
        result = generate_trajectory_data(self.mdp, 0)
        self.assertEqual(result.n_rows, 0)
        self.assertEqual(result.learner_rows["observation"].shape, (0, 4))
        for mode in ("unknown",):
            with self.assertRaises(ValueError):
                RetentionSpec(mode)
        for probability in (0.0, 1.0, np.nan):
            with self.assertRaises(ValueError):
                RetentionSpec("rare", probability)
        with self.assertRaises(ValueError):
            generate_trajectory_data(self.mdp, -1)
        with self.assertRaises(ValueError):
            generate_trajectory_data(self.mdp, 1, seed=-1)
        with self.assertRaises(ValueError):
            generate_trajectory_data(self.mdp, 1, split="")
        with self.assertRaises(ValueError):
            generate_fixed_state_data(self.mdp, 1.5)


if __name__ == "__main__":
    unittest.main()
