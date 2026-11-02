"""Real simulator checks for the additive v1 shift scenarios."""
import unittest

import numpy as np

from thrusterchain.environment import ThrusterChainEnv, dynamics, matrices
from thrusterchain.shift_scenarios_v1 import (
    SCENARIOS, ShiftScenarioEnv, _reset_seed, evaluate_frozen_shifts,
)


class ConstantPolicy:
    def __init__(self, action=(0., 0., 0., 0.)):
        self.action = np.asarray(action, np.float32)

    def state_bytes(self):
        return self.action.tobytes()

    def act(self, observation):
        if observation.shape != (14,) or observation.dtype != np.float32:
            raise AssertionError("policy must receive the public 14-vector")
        return self.action.copy()


class MutatingPolicy(ConstantPolicy):
    def act(self, observation):
        self.action[0] += .1
        return self.action.copy()


class ShiftScenarioTests(unittest.TestCase):
    def test_baseline_exactly_matches_existing_environment(self):
        shifted = ShiftScenarioEnv("baseline")
        source = ThrusterChainEnv()
        obs, info = shifted.reset(seed=62)
        expected, old_info = source.reset(seed=info["reset_seed"])
        np.testing.assert_array_equal(obs, expected)
        np.testing.assert_array_equal(info["state"], old_info["state"])
        rng = np.random.default_rng(100)
        for index in range(100):
            a = rng.uniform(-1.2, 1.2, 4)
            new = shifted.step(a)
            old = source.step(a)
            np.testing.assert_array_equal(new[0], old[0])
            self.assertEqual(new[1:4], old[1:4])
            np.testing.assert_array_equal(new[4]["state"], old[4]["state"])
            np.testing.assert_array_equal(new[4]["command"], old[4]["command"])
            self.assertEqual(new[2], index == 99)
        with self.assertRaises(RuntimeError):
            shifted.step(np.zeros(4))

    def test_attenuation_changes_only_actuator_mapping_and_reward_costs_command(self):
        env = ShiftScenarioEnv("attenuated_thruster")
        obs, info = env.reset(seed=71)
        before = info["state"].copy()
        command = np.array([.8, -.7, .2, 0.], np.float32)
        obs, reward, terminated, truncated, info = env.step(command)
        applied = command * np.array([.55, 1, 1, 1], np.float32)
        expected_state, _ = dynamics(before, applied.astype(np.float64), env.env.physics)
        _, expected_reward = dynamics(before, command.astype(np.float64), env.env.physics)
        np.testing.assert_array_equal(info["state"], expected_state)
        np.testing.assert_array_equal(info["actuator_command"], applied.astype(np.float64))
        np.testing.assert_array_equal(info["command"], command.astype(np.float64))
        self.assertEqual(reward, expected_reward)
        self.assertEqual((terminated, truncated), (False, False))
        self.assertEqual(obs.shape, (14,))
        self.assertTrue(env.observation_space.contains(obs))
        # Multiplying command before nonlinear thrust is not a uniform loss of
        # physical force above stall: no such claim is made.

    def test_bounded_process_noise_and_reproducible_independent_streams(self):
        first = ShiftScenarioEnv("bounded_disturbance")
        second = ShiftScenarioEnv("bounded_disturbance")
        control = ShiftScenarioEnv("baseline")
        a, ai = first.reset(seed=17)
        b, bi = second.reset(seed=17)
        c, ci = control.reset(seed=17)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(ai["state"], ci["state"])
        self.assertEqual(ai["reset_seed"], _reset_seed(17))
        first_noise = None
        for t in range(100):
            aa = np.array([.1, -.2, .3, -.4], np.float32)
            previous = first.env.state
            a, ra, ta, xa, ia = first.step(aa)
            b, rb, tb, xb, ib = second.step(aa)
            if first_noise is None:
                first_noise = ia["disturbance"].copy()
            predicted, predicted_reward = dynamics(previous, aa.astype(np.float64), first.env.physics)
            predicted[:10] += ia["disturbance"]
            np.testing.assert_array_equal(ia["state"], predicted)
            self.assertEqual(ra, predicted_reward)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(ia["disturbance"], ib["disturbance"])
            self.assertLessEqual(np.max(np.abs(ia["disturbance"])), .02)
            self.assertEqual((ra, ta, xa), (rb, tb, xb))
            self.assertTrue(first.observation_space.contains(a))
        self.assertEqual(ta, True)
        self.assertFalse(xa)
        third = ShiftScenarioEnv("bounded_disturbance")
        third.reset(seed=18)
        _, _, _, _, different = third.step(np.zeros(4))
        self.assertFalse(np.array_equal(different["disturbance"], first_noise))

    def test_repeated_seedless_resets_continue_a_reproducible_episode_stream(self):
        first = ShiftScenarioEnv("bounded_disturbance")
        second = ShiftScenarioEnv("bounded_disturbance")
        first.reset(seed=301)
        second.reset(seed=301)
        for _ in range(3):
            obs1, info1 = first.reset()
            obs2, info2 = second.reset()
            self.assertEqual(info1["episode_seed"], info2["episode_seed"])
            np.testing.assert_array_equal(obs1, obs2)
            np.testing.assert_array_equal(first.step(np.zeros(4))[0], second.step(np.zeros(4))[0])

    def test_combined_transition_matches_declared_action_and_disturbance(self):
        env = ShiftScenarioEnv("combined")
        _, reset = env.reset(seed=531)
        before = reset["state"]
        cmd = np.array([.9, -.2, .3, -.4], np.float32)
        _, observed_reward, _, _, info = env.step(cmd)
        applied = cmd * np.array([.55, 1., 1., 1.], np.float32)
        predicted, _ = dynamics(before, applied.astype(np.float64), env.env.physics)
        predicted[:10] += info["disturbance"]
        _, expected_reward = dynamics(before, cmd.astype(np.float64), env.env.physics)
        np.testing.assert_array_equal(info["state"], predicted)
        self.assertEqual(observed_reward, expected_reward)

    def test_finite_horizon_disturbance_bound_has_large_observation_margin(self):
        # Independent conservative numerical L2 bound: 100 * max_t ||A^t||
        # for the start, plus the sum of operator norms times the largest
        # actuator, current/wave, and bounded per-coordinate noise inputs.
        a, b = matrices()
        np.testing.assert_array_equal(a[:10, 10], a[:10, 11])
        np.testing.assert_array_equal(a[:10, 12], np.zeros(10))
        physical = a[:10, :10]
        powers = [np.linalg.norm(np.linalg.matrix_power(physical, i), 2) for i in range(101)]
        actuator = np.linalg.norm(b[:10], 2) * 2.  # four bounded thrusts
        external = np.linalg.norm(a[:10, 10], 2) * (2.5 + 1e-9)
        disturbance = .02 * np.sqrt(10.)
        bound = 100. * max(powers) + sum(powers[:100]) * (actuator + external + disturbance)
        self.assertLess(bound, 250.)
        self.assertGreater(300. - bound, 50.)

    def test_frozen_evaluation_pairing_seeds_and_purity(self):
        partitions = {"dataset": [1, 2], "selection": [5], "final": [8, 9]}
        result = evaluate_frozen_shifts(ConstantPolicy(), scenarios=list(SCENARIOS), partitions=partitions,
                                        policy_task_semantics="finite_horizon")
        self.assertEqual(result["episodes_total"], 8)
        self.assertEqual(result["environment_steps"], 800)
        self.assertEqual(result["policy_sha256_before"], result["policy_sha256_after"])
        self.assertEqual(result["policy_task_semantics_declared"], "finite_horizon")
        self.assertFalse(result["policy_task_semantics_attested"])
        self.assertEqual(result["inference_audit"]["forbidden_api_calls"], 0)
        for row in result["scenarios"].values():
            self.assertEqual([e["episode_seed"] for e in row["episodes"]], [8, 9])
            self.assertTrue(all(e["steps"] == 100 for e in row["episodes"]))
        resets = [[e["reset_seed"] for e in row["episodes"]] for row in result["scenarios"].values()]
        self.assertTrue(all(value == resets[0] for value in resets))
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_frozen_shifts(ConstantPolicy(), scenarios=["baseline"],
                partitions={"dataset": [1], "selection": [8], "final": [8]},
                policy_task_semantics="finite_horizon")
        with self.assertRaisesRegex(ValueError, "derived evaluation reset seeds"):
            evaluate_frozen_shifts(ConstantPolicy(), scenarios=["baseline"],
                partitions={"dataset": [_reset_seed(8)], "selection": [], "final": [8]},
                policy_task_semantics="finite_horizon")
        with self.assertRaisesRegex(RuntimeError, "changed during action"):
            evaluate_frozen_shifts(MutatingPolicy(), scenarios=["baseline"], partitions=partitions,
                                   policy_task_semantics="finite_horizon")
        with self.assertRaisesRegex(ValueError, "declared finite_horizon policy"):
            evaluate_frozen_shifts(ConstantPolicy(), scenarios=["baseline"], partitions=partitions,
                                   policy_task_semantics="continuing_time_limit")
        with self.assertRaises(TypeError):
            evaluate_frozen_shifts(ConstantPolicy(), scenarios=["baseline"], partitions=partitions)

    def test_real_frozen_trained_policy_evaluation(self):
        from thrusterchain.datasets import generate_dataset
        from thrusterchain.training_protocol import OfflineBudget
        from thrusterchain.training_runner import train_offline
        from thrusterchain.sgd_adapter import AffineSGDBCAdapter

        data = generate_dataset(episodes=10, seed=44, provenance="smoke")
        checkpoint, receipt = train_offline(data, AffineSGDBCAdapter, budget=OfflineBudget(2, 8), seed=46)
        model = AffineSGDBCAdapter()
        model.restore(checkpoint)
        result = evaluate_frozen_shifts(model.freeze(), scenarios=list(SCENARIOS),
            partitions={"dataset": data.arrays["episode_seeds"].tolist(), "selection": [51], "final": [52]},
            policy_task_semantics="finite_horizon")
        self.assertEqual(receipt["environment_interactions"], 0)
        self.assertEqual(result["episodes_total"], 4)
        self.assertEqual(result["environment_steps"], 400)
        self.assertEqual(result["policy_sha256_before"], result["policy_sha256_after"])

    def test_rejected_inputs_do_not_advance_state(self):
        env = ShiftScenarioEnv("combined")
        with self.assertRaises(ValueError): env.reset(seed=True)
        with self.assertRaises(ValueError): env.reset(seed=2**63)
        env.reset(seed=2)
        original = env.env.state
        for bad in ([np.nan] * 4, np.zeros((1, 4)), [1e300] * 4, [True] * 4):
            with self.assertRaises(ValueError): env.step(bad)
            np.testing.assert_array_equal(original, env.env.state)
            self.assertEqual(env.env.t, 0)
        with self.assertRaises(TypeError): SCENARIOS["combined"]["disturbance_bound"] = 2


if __name__ == "__main__":
    unittest.main()
