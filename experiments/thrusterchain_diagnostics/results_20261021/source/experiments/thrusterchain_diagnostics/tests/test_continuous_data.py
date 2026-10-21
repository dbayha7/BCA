import unittest
import numpy as np

from experiments.thrusterchain_diagnostics.continuous import Objective, dynamics
from experiments.thrusterchain_diagnostics.continuous_data import generate_continuous_data, context_weights


class ContinuousDataTests(unittest.TestCase):
    def test_trajectory_replay_and_end(self):
        data = generate_continuous_data(n_episodes=4, objective=Objective(horizon=4), seed=41)
        rows = data.evaluator
        for i in range(len(rows["reward"])):
            nxt, reward = dynamics(rows["state"][i], rows["action"][i], int(rows["regime"][i]))
            np.testing.assert_array_equal(nxt, rows["next_state"][i])
            self.assertEqual(reward, rows["reward"][i])
            if rows["time"][i] < 3:
                np.testing.assert_array_equal(nxt, rows["state"][i + 1])
        self.assertFalse(rows["terminated"].any())
        np.testing.assert_array_equal(rows["truncated"], rows["time"] == 3)
        np.testing.assert_array_equal(rows["bootstrap_mask"], rows["time"] < 3)
        np.testing.assert_allclose(rows["next_observation"][rows["truncated"], -1], 0)
        np.testing.assert_allclose(rows["action_probability"], 1 / 625)
        self.assertFalse(any(key in data.learner_rows for key in ("state", "next_state", "retained", "critical")))

    def test_nested_masks_and_fixed_state_contract(self):
        states = np.zeros((24, 13)); states[:, 10] = 1.1
        args = dict(track="fixed_state", fixed_states=states, fixed_regimes=np.zeros(24, int),
                    objective=Objective(horizon=4), seed=42)
        full, rare, zero = [generate_continuous_data(retention=r, **args) for r in ("full", "rare", "zero")]
        for key in ("state", "action", "reward", "next_state"):
            np.testing.assert_array_equal(full.evaluator[key], rare.evaluator[key])
            np.testing.assert_array_equal(full.evaluator[key], zero.evaluator[key])
        self.assertTrue(np.all(zero.evaluator["retained"] <= rare.evaluator["retained"]))
        self.assertTrue(np.all(rare.evaluator["retained"] <= full.evaluator["retained"]))
        self.assertGreater(full.evaluator["critical"].sum(), 0)
        self.assertFalse(np.any(zero.evaluator["retained"] & zero.evaluator["critical"]))
        np.testing.assert_array_equal(full.evaluator["state"], states)

    def test_holdout_stream_and_context_support(self):
        train = generate_continuous_data(n_episodes=2, objective=Objective(horizon=2), seed=7)
        holdout = generate_continuous_data(n_episodes=2, objective=Objective(horizon=2), seed=7, split="holdout")
        self.assertFalse(np.array_equal(train.evaluator["state"], holdout.evaluator["state"]))
        self.assertTrue(np.all(context_weights("thin") > 0))
        self.assertTrue(np.all(context_weights("zero")[:, 1] == 0))

    def test_fixed_queries_reject_silent_coercion_or_ignored_fields(self):
        state = np.zeros((1, 13))
        for regimes, times in (([.9], [0]), ([0], [1.9])):
            with self.assertRaises(ValueError):
                generate_continuous_data(track="fixed_state", fixed_states=state,
                                         fixed_regimes=regimes, fixed_times=times)
        with self.assertRaises(ValueError):
            generate_continuous_data(track="fixed_state", fixed_times=[3])
        state[0, 10] = 10
        with self.assertRaises(ValueError):
            generate_continuous_data(track="fixed_state", fixed_states=state, fixed_regimes=[0])


if __name__ == "__main__":
    unittest.main()
