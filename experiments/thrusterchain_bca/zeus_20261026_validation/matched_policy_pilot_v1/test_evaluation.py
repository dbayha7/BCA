"""Synthetic evaluation fixtures only: no JAX, actual models, resets or simulator."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import evaluate_worker as evaluation


class EvaluationFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="evaluation_fixture_", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "progress.json"
        self.protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())

    def ledger(self, cap=22400, clock=lambda: 0.):
        return evaluation.Ledger(self.path, cap, 600, clock=clock)

    def test_fixed_protocol_and_seed_metadata(self):
        evaluation.validate_protocol(self.protocol)
        receipt = json.loads((Path(__file__).parent / "EVALUATION_SEEDS.json").read_text())
        self.assertEqual(receipt["reset_seeds"], evaluation.SEEDS)
        self.assertEqual(receipt["known_reset_or_reserved_confirmation_collisions"], [])
        changed = copy.deepcopy(self.protocol)
        changed["horizon"] = 101
        with self.assertRaises(ValueError):
            evaluation.validate_protocol(changed)

    def test_raw_observation_time_and_float32_deployment(self):
        states = np.arange(26, dtype=np.float64).reshape(2, 13) / 7
        obs = evaluation.observation(states, 99, 100)
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_array_equal(obs[:, :13], states.astype(np.float32))
        np.testing.assert_array_equal(obs[:, 13], np.full(2, .01, np.float32))
        action = evaluation.deployed_actions(lambda x: np.full((len(x), 4), .123456789, np.float64), obs)
        self.assertEqual(action.dtype, np.float32)
        self.assertEqual(float(action[0, 0]), float(np.float32(.123456789)))
        with self.assertRaises(ValueError):
            evaluation.deployed_actions(lambda x: np.full((len(x), 4), 1.1), obs)

    def test_full_horizon_discount_no_bootstrap_and_own_actions(self):
        ledger = self.ledger()
        # Long arithmetic loops use an in-memory journal to avoid Windows file
        # sharing contention; separate short fixtures check actual atomic files.
        journal = []
        ledger.persist = lambda: journal.append(ledger.data.copy())
        visited = []
        def actor(obs):
            visited.append(obs.copy())
            return np.repeat(obs[:, 13:14], 4, axis=1)
        def dynamics(states, actions):
            self.assertEqual(actions.dtype, np.float64)
            self.assertFalse(journal[-1]["execution_counts_known"])
            return states+1., np.ones(len(states), np.float64)
        result = evaluation.rollout(actor, dynamics, np.zeros((32, 13), np.float64), 100, .98, ledger, "fixture")
        self.assertEqual(len(visited), 100)
        self.assertEqual(result["states"].shape, (32, 101, 13))
        np.testing.assert_array_equal(result["states"][:, -1], np.full((32, 13), 100.))
        expected = 0.
        for t in range(100):
            expected += .98**t
        np.testing.assert_array_equal(result["returns"], np.full(32, expected))
        self.assertEqual(ledger.data["attempted_transition_rows"], 3200)
        self.assertEqual(ledger.data["completed_episodes"], 32)
        self.assertEqual(ledger.data["actor_action_rows"], 3200)
        self.assertEqual(result["actions"][0, 99, 0], np.float32(.01))

    def test_last_reward_is_included(self):
        calls = [0]
        ledger = self.ledger()
        ledger.persist = lambda: None
        def dynamics(states, actions):
            calls[0] += 1
            return states, np.full(len(states), float(calls[0] == 100))
        result = evaluation.rollout(lambda x: np.zeros((len(x), 4)), dynamics,
            np.zeros((1, 13), np.float64), 100, .98, ledger, "fixture")
        self.assertEqual(result["returns"][0], .98**99)

    def test_failure_charge_is_preserved_as_uncertain(self):
        ledger = self.ledger()
        def fails(states, actions):
            saved = json.loads(self.path.read_text())
            self.assertEqual(saved["attempted_transition_rows"], 32)
            self.assertEqual(saved["inflight_transition_rows"], 32)
            raise RuntimeError("synthetic dispatch failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            evaluation.rollout(lambda x: np.zeros((len(x), 4)), fails,
                np.zeros((32, 13), np.float64), 100, .98, ledger, "fixture")
        self.assertFalse(ledger.data["execution_counts_known"])
        self.assertEqual(ledger.data["completed_transition_rows"], 0)
        self.assertEqual(ledger.data["inflight_transition_rows"], 32)
        with self.assertRaises(RuntimeError):
            ledger.before(32, "fixture", 1)

    def test_returned_invalid_transition_count_is_known(self):
        ledger = self.ledger()
        with self.assertRaises(ValueError):
            evaluation.rollout(lambda x: np.zeros((len(x), 4)),
                lambda s, a: (s, np.full(len(s), np.nan)), np.zeros((32, 13), np.float64),
                100, .98, ledger, "fixture")
        self.assertTrue(ledger.data["execution_counts_known"])
        self.assertEqual(ledger.data["completed_transition_rows"], 32)
        self.assertEqual(ledger.data["completed_episodes"], 0)

    def test_cap_stops_before_next_dispatch(self):
        ledger = self.ledger(cap=32)
        calls = []
        def dynamics(s, a):
            calls.append(1)
            return s, np.zeros(len(s))
        with self.assertRaisesRegex(RuntimeError, "cap"):
            evaluation.rollout(lambda x: np.zeros((len(x), 4)), dynamics,
                np.zeros((32, 13), np.float64), 100, .98, ledger, "fixture")
        self.assertEqual(len(calls), 1)
        self.assertEqual(ledger.data["attempted_transition_rows"], 32)

    def test_timeout_stops_before_dispatch(self):
        clock = [0.]
        ledger = self.ledger(clock=lambda: clock[0])
        clock[0] = 600.
        with self.assertRaises(TimeoutError):
            ledger.before(32, "fixture", 0)
        self.assertEqual(ledger.data["attempted_transition_rows"], 0)

    def test_reset_inflight_counts_are_unknown_until_completed(self):
        ledger = self.ledger()
        ledger.before_reset()
        saved = json.loads(self.path.read_text())
        self.assertTrue(saved["reset_inflight"])
        self.assertFalse(saved["execution_counts_known"])
        self.assertEqual(saved["reset_calls_attempted"], 1)
        self.assertEqual(saved["reset_calls_completed"], 0)
        ledger.after_reset()
        self.assertTrue(ledger.data["execution_counts_known"])
        self.assertEqual(ledger.data["reset_calls_completed"], 1)

    def test_numeric_band_all_harms_and_bootstrap(self):
        right = np.full(32, -100.)
        left = right.copy()
        left[0] += 1.
        left[1] -= 2.
        left[2] += 1e-8
        indices = np.tile(np.arange(32), (2000, 1))
        result = evaluation.paired_comparison(left, right, self.protocol, indices)
        self.assertEqual(result["improved_episodes"], 1)
        self.assertEqual(result["harmed_episodes"], 1)
        self.assertEqual(result["numerical_ties"], 30)
        self.assertEqual(result["harm_episode_indices"], [1])
        self.assertFalse(result["positive_screen"])
        self.assertEqual(result["mean_gain_episode_paired_bootstrap_95pct"], [result["mean_gain"]]*2)

    def test_screen_requires_both_controls_in_each_stream(self):
        values = {key: np.full(32, -100.) for key in ["original"]+evaluation.RUN_ORDER}
        values["seed0_bca"] += 3.
        values["seed1_bca"] += 1.
        first = evaluation.decision_report(values, self.protocol)
        self.assertTrue(first["primary_screen_passed"])
        self.assertEqual(first, evaluation.decision_report(values, self.protocol))
        values["seed1_distance"] += 2.
        failed = evaluation.decision_report(values, self.protocol)
        self.assertFalse(failed["primary_screen_passed"])
        self.assertEqual(failed["decision"], "stop_this_pilot_without_tuning")
        self.assertEqual(failed["comparisons_by_continuation_stream"]["1"]["distance"]["harmed_episodes"], 32)

    def test_state_and_action_displacement_are_separate(self):
        std = np.arange(1, 15, dtype=np.float64)
        original_states = np.zeros((32, 101, 13))
        states = original_states.copy()
        states[:, 1:] = std[:13]+.001
        result = evaluation.displacement(np.ones((128, 4)), np.zeros((128, 4)), states, original_states, std)
        self.assertEqual(result["logged_probe_action_mse"], 1.)
        self.assertEqual(result["mean_standardized_physical_state_rms"], 1.)
        # The final three physical dimensions must contribute, and time must not.
        states[:, 1:] = 0.
        states[:, 1:, 12] = std[12]+.001
        result = evaluation.displacement(np.zeros((128, 4)), np.zeros((128, 4)), states, original_states, std)
        self.assertAlmostEqual(result["mean_standardized_physical_state_rms"], 1/np.sqrt(13))
        self.assertEqual(result["logged_probe_action_mse"], 0.)

    def test_actor_manifest_rejects_identity_normalization_and_budget_change(self):
        initial = {"critic": 20000, "actor": 10000, "actor_target": 10000, "critic_target": 10000,
            "outer_steps": 10000, "auxiliary": 20000, "aux_fit_attempts": 20000,
            "refresh_attempts": 3, "refresh_accepted": 3, "phase": 0, "critic_budget": 20000,
            "last_refresh_critic": 20000, "reference_ready": True, "next_refresh_index": 3}
        final = {**initial, **{k: initial[k]+v for k, v in evaluation.ADDITIONAL.items()},
            "critic_budget": 22000, "last_refresh_critic": 21000, "next_refresh_index": 4}
        norm = {"mean": [0.]*14, "std": [1.]*14}
        manifest = {"schema": "unifloral.matched_policy.actor.v1", "run_id": "seed0_bca", "stream_index": 0,
            "arm": "bca", "completed": True, "phase": 0, "protocol_sha256": "fixture",
            "native_revision": self.protocol["native_revision"], "native_source_sha256": self.protocol["native_source_sha256"],
            "initial_checkpoint_state_sha256": self.protocol["checkpoint_state_sha256"],
            "additional_updates": evaluation.ADDITIONAL.copy(), "initial_absolute_counts": initial,
            "final_absolute_counts": final, "normalization": {**norm, "epsilon": .001},
            "architecture": {"class": "DeterministicTanhActor", "source": "pinned standalone UniFloral TD3+BC", "action_dim": 4, "observation_dim": 14}}
        evaluation.validate_actor_manifest(manifest, "seed0_bca", self.protocol, "fixture", norm, initial)
        for field, changed in (("initial_checkpoint_state_sha256", "wrong"), ("protocol_sha256", "wrong"),
                ("normalization", {**norm, "epsilon": .1}), ("final_absolute_counts", {**final, "critic": 22001})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluation.validate_actor_manifest({**manifest, field: changed}, "seed0_bca", self.protocol, "fixture", norm, initial)

    def test_leaf_identity_shape_dtype_and_changes(self):
        value = {"params": {"Dense_0": {"kernel": np.ones((2, 3), np.float32)}}}
        first = evaluation.leaf_descriptions(value)
        self.assertEqual(first[0]["path"], "params/Dense_0/kernel")
        self.assertEqual(first[0]["shape"], [2, 3])
        self.assertEqual(first[0]["dtype"], "<f4")
        value["params"]["Dense_0"]["kernel"][0, 0] = 2
        self.assertNotEqual(first[0]["sha256"], evaluation.leaf_descriptions(value)[0]["sha256"])

    def test_atomic_npz_uses_no_pickle(self):
        path = Path(self.temp.name) / "fixture.npz"
        receipt = evaluation.atomic_npz(path, returns=np.arange(32, dtype=np.float64))
        self.assertEqual(receipt["sha256"], evaluation.sha(path.read_bytes()))
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["returns"], np.arange(32))

    def test_supervisor_report_manifest_seal(self):
        directory = Path(self.temp.name)
        manifest_raw = b'{"fixture":true}\n'
        report = {"status": "completed_bounded_policy_training", "run_id": "seed0_bca",
            "protocol_sha256": "fixture", "execution_counts_known": True,
            "actor_manifest_sha256": evaluation.sha(manifest_raw)}
        report_raw = evaluation.encoded(report)
        supervisor = {"job": "seed0_bca", "completed": True, "exit_code": 0,
            "timed_out": False, "interrupted": False, "report_sha256": evaluation.sha(report_raw)}
        (directory / "actor_manifest.json").write_bytes(manifest_raw)
        (directory / "report.json").write_bytes(report_raw)
        (directory / "SUPERVISOR_RESULT.json").write_bytes(evaluation.encoded(supervisor))
        self.assertEqual(evaluation.training_seal(directory, "seed0_bca", "fixture")[0], manifest_raw)
        (directory / "actor_manifest.json").write_bytes(b'{"fixture":false}\n')
        with self.assertRaisesRegex(ValueError, "seal"):
            evaluation.training_seal(directory, "seed0_bca", "fixture")


if __name__ == "__main__":
    unittest.main()
