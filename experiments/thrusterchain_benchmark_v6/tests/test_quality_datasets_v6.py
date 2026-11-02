"""Checks for episode identity, behavior lineage, replay and runner use."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys
import unittest

import numpy as np

from thrusterchain.datasets import DatasetError, inference_evaluation_guard
from thrusterchain.quality_datasets import (BehaviorDataset, generate_behavior_dataset,
    load_behavior_dataset, save_behavior_dataset)
from thrusterchain.sgd_adapter import AffineSGDBCAdapter
from thrusterchain.regular import (calibrate_frozen_policy, evaluate_frozen_policy,
                                   train_behavior_cloning)
from thrusterchain.training_protocol import OfflineBudget, EvaluationConfig
from thrusterchain.training_runner import run_mode, train_offline, train_transitions
from run_benchmark import export_transitions


class QualityDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {name: generate_behavior_dataset(behavior=name, episodes=10, seed=822)
                    for name in ("random", "feedback_weak", "feedback_strong", "mixed_random_strong")}

    def test_behavior_sources_are_distinct_and_mixture_is_episode_balanced(self):
        random, weak, strong, mixed = (self.data[n] for n in
            ("random", "feedback_weak", "feedback_strong", "mixed_random_strong"))
        self.assertEqual(np.count_nonzero(mixed.arrays["behavior_component"] == 0), 5)
        self.assertEqual(np.count_nonzero(mixed.arrays["behavior_component"] == 2), 5)
        for other in (weak, strong, mixed):
            np.testing.assert_array_equal(random.arrays["states"][:, 0], other.arrays["states"][:, 0])
            self.assertFalse(np.array_equal(random.arrays["actions"], other.arrays["actions"]))
        self.assertFalse(np.array_equal(weak.arrays["actions"], strong.arrays["actions"]))
        self.assertEqual(mixed.metadata["density"]["joint_state_action_density"], "not_provided")
        self.assertEqual(mixed.metadata["behavior"]["feedback_observation"], "public_float32_physical_coordinates")
        for split in mixed.metadata["splits"]:
            self.assertGreater(len(mixed.split(split)["group_ids"]), 0)

    def test_roundtrip_immutability_and_tamper_rejection(self):
        data = self.data["mixed_random_strong"]
        with self.assertRaises(ValueError): data.arrays["actions"].setflags(write=True)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            save_behavior_dataset(data, path)
            loaded = load_behavior_dataset(path)
            self.assertEqual(data.digest, loaded.digest)
            with self.assertRaises(FileExistsError): save_behavior_dataset(data, path)
        changed = {k: np.array(v) for k, v in data.arrays.items()}
        changed["actions"][0, 0, 0] = -changed["actions"][0, 0, 0]
        with self.assertRaises(DatasetError): BehaviorDataset(changed, data.metadata)
        changed = {k: np.array(v) for k, v in data.arrays.items()}
        changed["behavior_component"][0] = 1
        with self.assertRaises(DatasetError): BehaviorDataset(changed, data.metadata)
        with inference_evaluation_guard():
            with self.assertRaises(DatasetError): generate_behavior_dataset(episodes=10)

    def test_train_rows_and_next_actions_are_episode_local(self):
        data = self.data["mixed_random_strong"]
        train = data.split("train")
        batch = train_transitions(data)
        np.testing.assert_array_equal(batch.observations.reshape(-1, 100, 14), train["observations"][:, :-1])
        np.testing.assert_array_equal(batch.next_observations.reshape(-1, 100, 14), train["observations"][:, 1:])
        actual_next = batch.next_actions.reshape(-1, 100, 4)
        np.testing.assert_array_equal(actual_next[:, :-1], train["actions"][:, 1:])
        self.assertFalse(np.any(batch.next_action_valid.reshape(-1, 100)[:, -1]))
        self.assertFalse(np.any(batch.bootstrap_mask.reshape(-1, 100)[:, -1]))
        self.assertEqual(set(batch.group_ids.tolist()), set(train["group_ids"].tolist()))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "transitions.npz"
            info = export_transitions(data, path)
            self.assertEqual(info["dataset_sha256"], data.digest)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(len(archive["observations"]), 1000)
                self.assertFalse(np.any(archive["next_action_valid"].reshape(-1, 100)[:, -1]))
                np.testing.assert_array_equal(archive["next_actions"].reshape(-1, 100, 4)[:, :-1],
                                              data.arrays["actions"][:, 1:])

    def test_continuing_cutoff_has_successor_and_separate_masks(self):
        data = generate_behavior_dataset(behavior="feedback_weak", episodes=10, seed=823,
                                         task_semantics="continuing_time_limit")
        batch = train_transitions(data)
        self.assertTrue(np.all(batch.next_observations[:, -1] == 0))
        self.assertTrue(np.all(batch.bootstrap_mask.reshape(-1, 100)[:, -1] == 1))
        self.assertFalse(np.any(batch.next_action_valid.reshape(-1, 100)[:, -1]))
        self.assertTrue(np.all(batch.truncated.reshape(-1, 100)[:, -1]))

    def test_bounded_real_runner_and_frozen_evaluation(self):
        data = self.data["feedback_strong"]
        checkpoint, receipt = train_offline(data, AffineSGDBCAdapter,
                                           budget=OfflineBudget(2, 16), seed=825)
        self.assertEqual(receipt["dataset_sha256"], data.digest)
        self.assertEqual(receipt["environment_interactions"], 0)
        result = run_mode(data, checkpoint, AffineSGDBCAdapter,
                          EvaluationConfig(frozen_policy=True, episodes=1, seed=826),
                          families=("recorded",))
        self.assertEqual(result["receipt"]["adaptation_updates"], 0)
        self.assertEqual(result["receipt"]["frozen_evaluation_steps"], 100)

    def test_closed_form_bc_and_mixture_calibration_are_scoped(self):
        data = self.data["mixed_random_strong"]
        policy, receipt = train_behavior_cloning(data)
        self.assertEqual(receipt.payload["dataset_sha256"], data.digest)
        calibration = calibrate_frozen_policy(data, policy, receipt)
        self.assertIn("descriptive only", calibration["claim"])
        self.assertTrue(calibration["not_a_return_or_shift_certificate"])
        self.assertTrue(calibration["vacuous"])
        evaluation = evaluate_frozen_policy(policy, receipt, episodes_per_family=1, seed=827)
        self.assertEqual(evaluation["environment_steps"], 400)
        self.assertEqual(evaluation["parameter_updates"], 0)

    def test_explicit_cli_generate_validate_train_and_calibrate(self):
        root = Path(__file__).resolve().parents[1]
        def call(*args, ok=True):
            proc = subprocess.run([sys.executable, "-B", "-m", "thrusterchain", *map(str, args)],
                                  cwd=root, text=True, capture_output=True)
            if ok:
                self.assertEqual(proc.returncode, 0, proc.stderr)
                return json.loads(proc.stdout)
            self.assertNotEqual(proc.returncode, 0)
            return proc.stderr
        with TemporaryDirectory() as tmp:
            data = Path(tmp) / "behavior.npz"
            call("generate-behavior", data, "--episodes", 10, ok=False)
            self.assertFalse(data.exists())
            made = call("generate-behavior", data, "--behavior", "mixed_random_strong", "--episodes", 10, "--seed", 928)
            saved = data.read_bytes()
            self.assertEqual(made["sha256"], call("validate", data)["sha256"])
            frozen = Path(tmp) / "closed"
            self.assertEqual(call("train", data, frozen)["dataset_sha256"], made["sha256"])
            cal = call("calibrate", data, frozen, Path(tmp) / "cal.json")
            self.assertIn("descriptive only", cal["claim"])
            self.assertEqual(call("evaluate", frozen, Path(tmp) / "closed_eval.json", "--episodes", 1)["environment_steps"], 400)
            iterative = Path(tmp) / "iterative"
            call("train", data, iterative, "--adapter", "affine-sgd", "--offline-updates", 2, "--batch-size", 8)
            res = call("evaluate", iterative, Path(tmp) / "iter_eval.json", "--dataset", data, "--episodes", 1)
            self.assertEqual(res["receipt"]["adaptation_updates"], 0)
            call("generate-behavior", data, "--behavior", "random", "--episodes", 10, ok=False)
            self.assertEqual(data.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
