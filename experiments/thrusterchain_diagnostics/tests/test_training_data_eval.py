"""Small, independent checks for the frozen training data/evaluation contract.

These never train a network or launch the full experiment.  The scalar plant
rollout is the independent reference for vectorized finite-horizon evaluation.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.thrusterchain_diagnostics import continuous as C
from experiments.thrusterchain_diagnostics import training_data as D
from experiments.thrusterchain_diagnostics import training_eval as E


def feedback_policy(observations):
    observations = np.asarray(observations)
    return .93 * np.tanh(observations[:, :4] + .4 * observations[:, 10:11]
                         - .2 * observations[:, 13:14] + observations[:, 14:15])


class TrainingDataTests(unittest.TestCase):
    def make_source(self, seed=41):
        return D.generate_source_data(seed=seed, n_episodes=8,
            reservation_size=2, reservation_seed=911, objective=C.Objective(horizon=4))

    def test_balanced_initial_law_and_shared_reservation(self):
        source = self.make_source()
        rows = source.rows
        initial = rows["time"] == 0
        starts, regimes = rows["observation"][initial, :13], rows["observation"][initial, 13]
        for regime in (0, 1):
            for current in (.3, 1.1):
                self.assertEqual(int(np.sum((regimes == regime) & (starts[:, 10] == current))), 2)
        self.assertTrue(np.all(np.abs(starts[:, :10]) <= .5))
        np.testing.assert_allclose(np.linalg.norm(starts[:, 11:13], axis=1), .2, atol=1e-15)
        reserved = np.asarray(source.calibration_episode_ids)
        self.assertEqual(len(reserved), 2)
        self.assertEqual(len(np.unique(reserved)), 2)
        self.assertTrue(np.isin(reserved, np.arange(8)).all())
        repeated = self.make_source()
        np.testing.assert_array_equal(repeated.calibration_episode_ids, reserved)
        for key in rows:
            np.testing.assert_array_equal(repeated.rows[key], rows[key], err_msg=key)

    def test_exact_categorical_conditional_mass_and_replay(self):
        source = self.make_source()
        rows = source.rows
        atoms = np.array([-1., -C.KNEE, 0., C.KNEE, 1.])
        controller = C.fixed_cap_policy(objective=C.Objective(horizon=4))
        for i, obs in enumerate(rows["observation"]):
            command = np.clip(.6 * (-controller.gains[int(rows["time"][i])] @ obs[:13]),
                              -C.KNEE, C.KNEE)
            preferred = atoms[np.argmin(np.abs(command[:, None] - atoms), axis=1)]
            action = rows["action"][i]
            self.assertTrue(np.isin(action, atoms).all())
            probability = np.prod(np.where(action == preferred, .84, .04))
            self.assertAlmostEqual(float(rows["action_probability"][i]), float(probability), places=14)
            nxt, reward = C.dynamics(obs[:13], action, int(obs[13]))
            np.testing.assert_array_equal(rows["next_observation"][i, :13], nxt)
            self.assertEqual(rows["reward"][i], reward)
            self.assertEqual(rows["next_observation"][i, 13], obs[13])
        np.testing.assert_array_equal(rows["bootstrap_mask"], rows["time"] < 3)
        np.testing.assert_array_equal(rows["truncated"], rows["time"] == 3)
        self.assertFalse(np.any(rows["terminated"]))

    def test_retention_is_nested_and_keeps_original_transition(self):
        source = self.make_source(seed=223)
        source_rows = source.rows
        retained = {name: D.retain_source(source, name, rare_probability=.05)
                    for name in ("full", "rare", "zero")}
        row_ids = {name: set(dataset.learner_rows["row_id"].tolist())
                   for name, dataset in retained.items()}
        self.assertTrue(row_ids["zero"] <= row_ids["rare"] <= row_ids["full"])
        self.assertEqual(len(row_ids["full"]), 32)
        critical = ((source_rows["observation"][:, 10] >= .8)
                    & np.any(np.abs(source_rows["action"]) > C.KNEE, axis=1))
        self.assertTrue(np.any(critical))
        zero_ids = retained["zero"].learner_rows["row_id"]
        self.assertFalse(np.any(critical[zero_ids]))
        for name, dataset in retained.items():
            learner = dataset.learner_rows
            ids = learner["row_id"]
            probability = np.where(critical, {"full": 1., "rare": .05, "zero": 0.}[name], 1.)
            expected_ids = source_rows["row_id"][source_rows["retention_uniform"] < probability]
            np.testing.assert_array_equal(ids, expected_ids)
            for key in ("observation", "action", "reward", "next_observation", "bootstrap_mask"):
                np.testing.assert_array_equal(learner[key], source_rows[key][ids], err_msg=key)
            self.assertTrue(np.all(learner["split"] == "train"))
            self.assertFalse(set(learner) & {"critical", "retained", "retention_uniform",
                                           "state", "next_state", "oracle", "context"})
            np.testing.assert_array_equal(dataset.metadata["calibration_episode_ids"],
                                          source.calibration_episode_ids)

    def test_saved_views_round_trip_and_reject_file_tampering(self):
        source = self.make_source()
        dataset = D.retain_source(source, "rare")
        with tempfile.TemporaryDirectory() as temporary:
            source_path, learner_path = Path(temporary) / "audit", Path(temporary) / "learner"
            D.save_source(source, source_path)
            D.save_dataset(dataset, learner_path)
            restored_source, restored_data = D.load_source(source_path), D.load_dataset(learner_path)
            for key, value in source.rows.items():
                np.testing.assert_array_equal(restored_source.rows[key], value, err_msg=key)
            for key, value in dataset.learner_rows.items():
                np.testing.assert_array_equal(restored_data.learner_rows[key], value, err_msg=key)
            self.assertFalse(restored_data.evaluator)
            np.testing.assert_array_equal(restored_source.calibration_episode_ids,
                                          source.calibration_episode_ids)
            with (learner_path / "learner.npz").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "identity"):
                D.load_dataset(learner_path)


class TrainingEvaluationTests(unittest.TestCase):
    def test_banks_are_reproducible_balanced_and_split_disjoint(self):
        objective = C.Objective(horizon=5)
        bank = E.make_start_bank(split="final", per_context=3, objective=objective)
        repeat = E.make_start_bank(split="final", per_context=3, objective=objective)
        validation = E.make_start_bank(split="validation", per_context=3, objective=objective)
        np.testing.assert_array_equal(bank.states, repeat.states)
        self.assertFalse(np.array_equal(bank.states, validation.states))
        self.assertEqual(bank.states.shape, (12, 13))
        for regime in (0, 1):
            for current in (.3, 1.1):
                self.assertEqual(int(np.sum((bank.regimes == regime)
                                           & (bank.states[:, 10] == current))), 3)
        self.assertTrue(np.all(bank.times == 0))
        self.assertTrue(np.all(np.abs(bank.states[:, :10]) <= .5))
        np.testing.assert_allclose(np.linalg.norm(bank.states[:, 11:], axis=1), .2, atol=1e-15)

    def test_batched_return_matches_scalar_deployed_policy(self):
        objective = C.Objective(horizon=5)
        bank = E.make_start_bank(split="final", per_context=2, objective=objective)
        batched = E.batch_rollout(feedback_policy, bank, float32_deployment=True)
        for i, (state, regime) in enumerate(zip(bank.states, bank.regimes)):
            scalar = C.rollout_policy(state, int(regime),
                lambda obs: feedback_policy(obs[None])[0], objective,
                float32_deployment=True)
            self.assertAlmostEqual(batched["return_value"][i], scalar.return_value, places=11)
            expected_high = np.mean(np.any(np.abs(scalar.actions) > C.KNEE, axis=1))
            self.assertAlmostEqual(batched["high_action_fraction"][i], expected_high)
            expected_position = np.max(np.abs(scalar.states[:, :10:2]))
            self.assertAlmostEqual(batched["max_abs_position"][i], expected_position, places=12)

    def test_batched_q_obeys_forced_first_action_and_remaining_horizon(self):
        objective = C.Objective(horizon=5)
        bank = E.make_start_bank(split="final", per_context=1, objective=objective)
        times = np.array([0, 1, 3, 4])
        actions = np.array([[1.2, -.9, .3, -.1], [-.8, .8, -.3, .2],
                            [.35, -.25, .1, .7], [0., 0., 0., 0.]])
        batched = E.batch_policy_q(feedback_policy, bank.states, bank.regimes,
            times, actions, objective=objective, float32_deployment=True)
        expected = np.array([C.policy_q(state, int(regime), action,
            lambda obs: feedback_policy(obs[None])[0], objective, t=int(t),
            float32_deployment=True)
            for state, regime, t, action in zip(bank.states, bank.regimes, times, actions)])
        np.testing.assert_allclose(batched, expected, rtol=1e-12, atol=1e-12)
        _, terminal_reward = C.dynamics(bank.states[-1], actions[-1], int(bank.regimes[-1]))
        self.assertEqual(batched[-1], terminal_reward)

    def test_scalar_true_critic_has_zero_error_and_correct_action_ranking(self):
        objective = C.Objective(horizon=5)
        bank = E.make_query_bank(objective, per_context=1)
        def exact_scalar_critic(observations, actions):
            values = np.array([C.policy_q(obs[:13], int(obs[13]), action,
                lambda next_obs: feedback_policy(next_obs[None])[0], objective,
                t=int(round(objective.horizon * (1 - obs[-1]))),
                float32_deployment=False) for obs, action in zip(observations, actions)])
            return np.column_stack((values, values))
        summary, arrays = E.evaluate_critic(feedback_policy, exact_scalar_critic, bank,
                                            float32_deployment=False)
        self.assertEqual(arrays["truth"].shape, (12, 6))
        self.assertLess(summary["q_error"]["twin_min"]["rmse"], 1e-11)
        self.assertLess(summary["bellman_residual"]["rmse"], 1e-11)
        self.assertLess(summary["action_ranking"]["mean_true_regret_of_predicted_best"], 1e-11)
        self.assertEqual(summary["action_ranking"]["pairwise_accuracy_excluding_true_ties"], 1.)

    def test_nonfinite_policy_rollouts_are_counted_as_failures(self):
        bank = E.make_start_bank(split="final", per_context=1, objective=C.Objective(horizon=3))
        def invalid_policy(observations):
            actions = np.zeros((len(observations), 4))
            actions[:, 0] = [np.nan, np.inf, -np.inf, np.nan][:len(observations)]
            return actions
        summary, arrays = E.evaluate_policy(invalid_policy, bank)
        self.assertTrue(np.all(arrays["failure"]))
        self.assertFalse(np.any(arrays["completed"]))
        self.assertTrue(np.isnan(arrays["return_value"]).all())
        self.assertIsNone(summary["pooled"]["mean_return"])
        self.assertEqual(summary["pooled"]["failure_rate"], 1.)

    def test_failed_critic_diagnostic_preserves_completed_return_evaluation(self):
        bank = E.make_start_bank(split="final", per_context=1, objective=C.Objective(horizon=3))
        queries = E.make_query_bank(bank.objective, per_context=1)
        def invalid_critic(observations, actions):
            raise ValueError("deliberate diagnostic failure")
        summary, arrays = E.evaluate_policy(feedback_policy, bank, critic=invalid_critic, query_bank=queries)
        self.assertTrue(np.all(arrays["completed"]))
        self.assertIsNotNone(summary["pooled"]["mean_return"])
        self.assertEqual(summary["critic_diagnostics"]["status"], "failed")
        self.assertIn("deliberate diagnostic failure", summary["critic_diagnostics"]["failure"])


if __name__ == "__main__":
    unittest.main()
