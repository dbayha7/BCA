"""Protocol regressions: leakage, probability measure, tampering and freeze."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thrusterchain.datasets import (
    DatasetError, SPLITS, TrajectoryDataset, behavior_log_density, canonical_json,
    generate_dataset, inference_evaluation_guard, load_dataset, save_dataset,
    sample_requested_action,
)
from thrusterchain.regular import (
    FrozenLinearPolicy, ProtocolError, SelectionReceipt, calibrate_frozen_policy,
    evaluate_frozen_policy, freeze_policy, load_frozen_policy, train_behavior_cloning,
)


class RegularProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(episodes=10, seed=913, provenance="smoke")
        cls.policy, cls.receipt = train_behavior_cloning(cls.dataset)

    def arrays_copy(self):
        return {name: value.copy() for name, value in self.dataset.arrays.items()}

    def test_grouped_splits_full_horizon_and_probability_measure(self):
        seen = set()
        for split_name in SPLITS:
            split = self.dataset.split(split_name)
            groups = set(split["group_ids"].tolist())
            self.assertTrue(groups)
            self.assertFalse(seen.intersection(groups))
            seen.update(groups)
            self.assertFalse(split["terminated"][:, :-1].any())
            self.assertTrue(split["terminated"][:, -1].all())
            self.assertFalse(split["truncated"].any())
            np.testing.assert_array_equal(split["observations"][:, -1, 13], 0)
        np.testing.assert_allclose(self.dataset.arrays["behavior_log_density"], -np.log(16), atol=1e-14)
        self.assertEqual(self.dataset.metadata["density"]["state_occupancy"], "unknown_closed_loop_occupancy")
        self.assertEqual(self.dataset.metadata["density"]["requested_action_measure"], "continuous_lebesgue_density")
        self.assertIn("quantization", self.dataset.metadata["density"]["executed_action_measure"])
        self.assertTrue(np.any(np.abs(self.dataset.arrays["actions"]) > .81))
        np.testing.assert_array_equal(self.dataset.arrays["actions"], self.dataset.arrays["requested_actions"].astype(np.float32))

    def test_region_mixture_normalization_gap_and_quantization_boundary(self):
        points = np.array([[0., 0., 0., 0.], [.75, 0., 0., 0.]])
        for epsilon in (.25, .01, 0.):
            densities = np.exp(behavior_log_density(points, epsilon))
            self.assertAlmostEqual(12 * densities[0] + 4 * densities[1], 1.)
            self.assertAlmostEqual(densities[1], epsilon / 4.)
        self.assertTrue(np.isneginf(behavior_log_density(points, 0.)[1]))
        self.assertTrue(np.isneginf(behavior_log_density(np.array([2., 0., 0., 0.]), .25)))
        gap = generate_dataset(episodes=10, seed=915, epsilon=0.)
        self.assertFalse(gap.arrays["action_region"].any())
        self.assertTrue(np.all(gap.arrays["requested_actions"][..., 0] <= .5))
        boundary = np.array([np.nextafter(.5, 1.), 0., 0., 0.])
        self.assertEqual(boundary.astype(np.float32)[0], .5)
        self.assertAlmostEqual(float(np.exp(behavior_log_density(boundary, .01))), .01 / 4.)
        with self.assertRaises(DatasetError):
            sample_requested_action(np.random.default_rng(1), -1)
        self.assertTrue(np.isfinite(behavior_log_density(points[1], np.nextafter(0., 1.))))
        for malformed in ([True] * 4, ["0"] * 4, np.full(4, 1j)):
            with self.assertRaises(DatasetError):
                behavior_log_density(malformed)

    def test_duplicate_within_split_cannot_inflate_calibration_sample_count(self):
        dataset = generate_dataset(episodes=20, seed=918)
        arrays = {name: value.copy() for name, value in dataset.arrays.items()}
        rows = np.flatnonzero(arrays["split_codes"] == SPLITS.index("final_calibration"))
        for name in arrays:
            if name != "trajectory_ids":
                arrays[name][rows[1]] = arrays[name][rows[0]]
        with self.assertRaisesRegex(DatasetError, "duplicate group"):
            TrajectoryDataset(arrays, dataset.metadata)

    def test_receipt_budget_and_physics_refusals(self):
        payload = self.receipt.payload
        payload["budget"]["simulator_selection_steps"] = 100
        with self.assertRaisesRegex(ProtocolError, "contradicts"):
            SelectionReceipt(payload)
        for update in ({"stall_slope": 2.}, {"observation_mode": "physical"}):
            metadata = self.dataset.metadata
            metadata["physics"].update(update)
            with self.assertRaises(DatasetError):
                TrajectoryDataset(self.dataset.arrays, metadata)

    def test_named_stream_prefixes_survive_extension_and_split_rng_consumption(self):
        from thrusterchain import datasets as module
        extended = generate_dataset(episodes=15, seed=913, provenance="smoke")
        for name in ("episode_seeds", "requested_actions", "actions", "states", "observations"):
            np.testing.assert_array_equal(self.dataset.arrays[name], extended.arrays[name][:10])
        original_assignment = module._split_assignment
        def split_with_extra_draws(episodes, seed):
            _, counts = original_assignment(episodes, seed)
            rng = np.random.Generator(np.random.PCG64(module._stream_seed_sequence(seed, "split")))
            rng.random(1000)
            codes = np.repeat(np.arange(4, dtype=np.uint8), counts)
            return codes[rng.permutation(episodes)], counts
        with patch.object(module, "_split_assignment", split_with_extra_draws):
            changed_split = generate_dataset(episodes=10, seed=913, provenance="smoke")
        self.assertFalse(np.array_equal(self.dataset.arrays["split_codes"], changed_split.arrays["split_codes"]))
        for name in ("episode_seeds", "requested_actions", "actions", "states", "observations"):
            np.testing.assert_array_equal(self.dataset.arrays[name], changed_split.arrays[name])
        generator = self.dataset.metadata["generator"]
        self.assertEqual(set(generator["named_streams"]), {"reset", "split", "actions"})
        self.assertEqual(generator["episodes"][0]["action_spawn_key"], [1703, 0])
        self.assertEqual(generator["episodes"][0]["reset_seed"], self.dataset.arrays["episode_seeds"][0])

    def test_random_stream_and_canonical_revision_tampering_refused(self):
        metadata = self.dataset.metadata
        metadata["generator"]["episodes"][0]["action_spawn_key"][-1] = 99
        with self.assertRaisesRegex(DatasetError, "random-stream hierarchy"):
            TrajectoryDataset(self.dataset.arrays, metadata)
        metadata = self.dataset.metadata
        metadata["source_model_metadata"]["canonical_provenance_sha256"] = "0" * 64
        with self.assertRaisesRegex(DatasetError, "environment revision"):
            TrajectoryDataset(self.dataset.arrays, metadata)
        metadata = self.dataset.metadata
        metadata["source_model_metadata"]["environment_config"]["physics"]["gamma"] = .9
        with self.assertRaisesRegex(DatasetError, "environment revision"):
            TrajectoryDataset(self.dataset.arrays, metadata)
        arrays = self.arrays_copy()
        arrays["episode_seeds"][0] ^= 1
        with self.assertRaisesRegex(DatasetError, "reset seeds disagree"):
            TrajectoryDataset(arrays, self.dataset.metadata)

    def test_explicit_continuing_time_limit_flags(self):
        dataset = generate_dataset(episodes=10, seed=914, task_semantics="continuing_time_limit")
        self.assertFalse(dataset.arrays["terminated"].any())
        self.assertTrue(dataset.arrays["truncated"][:, -1].all())
        self.assertFalse(dataset.arrays["truncated"][:, :-1].any())

    def test_group_leakage_and_nonfinite_refused(self):
        arrays = self.arrays_copy()
        left = np.flatnonzero(arrays["split_codes"] == 0)[0]
        right = np.flatnonzero(arrays["split_codes"] == 1)[0]
        arrays["group_ids"][right] = arrays["group_ids"][left]
        with self.assertRaisesRegex(DatasetError, "leaks"):
            TrajectoryDataset(arrays, self.dataset.metadata)
        arrays = self.arrays_copy()
        arrays["rewards"][0, 0] = np.nan
        with self.assertRaisesRegex(DatasetError, "nonfinite"):
            TrajectoryDataset(arrays, self.dataset.metadata)

    def test_replay_and_time_and_probability_corruption_refused(self):
        for field, index, delta, message in (
            ("rewards", (0, 0), 1., "rewards"),
            ("observations", (0, 20, 13), .1, "remaining-time"),
            ("behavior_log_density", (0, 0), .01, "density"),
        ):
            arrays = self.arrays_copy()
            arrays[field][index] += delta
            with self.subTest(field=field), self.assertRaisesRegex(DatasetError, message):
                TrajectoryDataset(arrays, self.dataset.metadata)

    def test_dataset_immutable_roundtrip_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.npz"
            save_dataset(self.dataset, path)
            restored = load_dataset(path)
            self.assertEqual(restored.digest, self.dataset.digest)
            with self.assertRaises(FileExistsError):
                save_dataset(self.dataset, path)
            with self.assertRaises(ValueError):
                restored.arrays["actions"].setflags(write=True)
            with self.assertRaises(AttributeError):
                restored._digest = "bad"
            metadata = restored.metadata
            metadata["provenance"] = "research"
            self.assertEqual(restored.metadata["provenance"], "smoke")
            # A changed but otherwise valid manifest cannot retain the old hash.
            envelope = {"metadata": metadata, "sha256": self.dataset.digest}
            tampered = Path(directory) / "tampered.npz"
            np.savez_compressed(tampered, **restored.arrays, manifest=np.asarray(canonical_json(envelope)))
            with self.assertRaisesRegex(DatasetError, "SHA-256 mismatch"):
                load_dataset(tampered)

    def test_offline_fitting_never_calls_environment_and_normalizes_train_only(self):
        with patch("thrusterchain.environment.ThrusterChainEnv", side_effect=AssertionError("offline simulator access")):
            policy, receipt = train_behavior_cloning(self.dataset)
        train = self.dataset.split("train")["observations"][:, :-1].reshape(-1, 14).astype(np.float64)
        np.testing.assert_array_equal(policy.mean, train.mean(axis=0))
        self.assertFalse(np.array_equal(policy.mean, self.dataset.arrays["observations"][:, :-1].reshape(-1, 14).mean(axis=0)))
        self.assertEqual(receipt.payload["budget"]["simulator_selection_steps"], 0)
        self.assertEqual(receipt.payload["normalization_source"], "train_only")
        # Training and candidate selection must not inspect calibration samples.
        original_split = TrajectoryDataset.split
        def guarded_split(dataset, name):
            if name == "final_calibration":
                raise AssertionError("calibration was used statistically during fitting")
            return original_split(dataset, name)
        # Hashing is identity accounting rather than fitting; provide known hashes.
        with patch.object(TrajectoryDataset, "split", guarded_split), patch.object(
            TrajectoryDataset, "split_digest", lambda dataset, name: self.receipt.payload["split_sha256"][name]
        ):
            candidate, _ = train_behavior_cloning(self.dataset)
        self.assertEqual(candidate.digest, self.policy.digest)

    def test_invalid_ridge_and_prediction_inputs_refused(self):
        for values in ([], [-1], [np.nan], [np.inf], [1, 1]):
            with self.subTest(values=values), self.assertRaises(ProtocolError):
                train_behavior_cloning(self.dataset, ridge_values=values)
        for obs in (np.zeros(13), np.full(14, np.nan), np.zeros((2, 13)), [True] * 14, ["0"] * 14, np.full(14, 1j)):
            with self.assertRaises(ProtocolError):
                self.policy.predict(obs)

    def test_frozen_roundtrip_receipt_and_parameter_tampering_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = freeze_policy(self.policy, self.receipt, directory)
            policy, receipt = load_frozen_policy(directory)
            self.assertEqual(policy.digest, self.policy.digest)
            self.assertEqual(receipt.digest, self.receipt.digest)
            with self.assertRaises(FileExistsError):
                freeze_policy(policy, receipt, directory)
            with self.assertRaises(ValueError):
                policy.mean.setflags(write=True)
            with self.assertRaises(ValueError):
                policy.scale.setflags(write=True)
            with self.assertRaises(ValueError):
                policy.coefficients.setflags(write=True)
            with self.assertRaises(AttributeError):
                policy.ridge = 0
            with np.load(path, allow_pickle=False) as archive:
                copied = {name: archive[name] for name in archive.files}
            copied["coefficients"] = copied["coefficients"].copy()
            copied["coefficients"][0, 0] += .1
            tampered = Path(directory) / "tampered.npz"
            np.savez_compressed(tampered, **copied)
            with self.assertRaisesRegex(ProtocolError, "changed after selection"):
                load_frozen_policy(tampered)
        payload = self.receipt.payload
        payload["selection_mse"] += 1
        with self.assertRaisesRegex(ProtocolError, "receipt SHA-256 mismatch"):
            SelectionReceipt(payload, expected_digest=self.receipt.digest)

    def test_small_fresh_calibration_is_vacuous_and_cannot_update_policy(self):
        before = self.policy.digest
        report = calibrate_frozen_policy(self.dataset, self.policy, self.receipt, alpha=.1)
        self.assertTrue(report["vacuous"])
        self.assertIsNone(report["radius"])
        self.assertTrue(report["not_a_return_or_shift_certificate"])
        self.assertEqual(self.policy.digest, before)
        with self.assertRaises(ProtocolError):
            calibrate_frozen_policy(self.dataset, self.policy, self.receipt, alpha=0)

    def test_live_frozen_evaluation_reproducible_and_honestly_labeled(self):
        first = evaluate_frozen_policy(self.policy, self.receipt, episodes_per_family=1, seed=42)
        second = evaluate_frozen_policy(self.policy, self.receipt, episodes_per_family=1, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["environment_steps"], 400)
        self.assertEqual(first["parameter_updates"], 0)
        self.assertEqual(first["normalization_updates"], 0)
        self.assertEqual(first["policy_sha256_before"], first["policy_sha256_after"])
        self.assertEqual(first["evidence_status"], "smoke_only_not_research_evidence")
        self.assertFalse(first["eligible_research_evidence"])
        self.assertEqual(first["inference_audit"]["inference_calls"], 400)
        self.assertEqual(first["inference_audit"]["forbidden_api_calls"], 0)
        self.assertEqual(sum(x["held_out_scenario_family"] for x in first["families"].values()), 3)
        for family in first["families"].values():
            self.assertFalse(set(family["episode_seeds"]) & set(self.dataset.arrays["episode_seeds"]))
        with self.assertRaisesRegex(ProtocolError, "relabeled"):
            evaluate_frozen_policy(self.policy, self.receipt, provenance="research")
        tuning = evaluate_frozen_policy(self.policy, self.receipt, episodes_per_family=1, seed=43,
                                        purpose="simulator_assisted_tuning")
        self.assertEqual(tuning["protocol"], "simulator_assisted_selection")
        self.assertFalse(tuning["eligible_strict_offline_final_reporting"])

    def test_evaluator_has_no_training_update_or_persistence_calls(self):
        targets = ["thrusterchain.regular.train_behavior_cloning", "thrusterchain.regular.calibrate_frozen_policy",
                   "thrusterchain.regular.freeze_policy", "thrusterchain.datasets.save_dataset",
                   "thrusterchain.datasets.generate_dataset", "numpy.save", "numpy.savez_compressed"]
        with ExitStack() as stack:
            sentinels = [stack.enter_context(patch(target, side_effect=AssertionError("forbidden evaluation call")))
                         for target in targets]
            update = stack.enter_context(patch.object(FrozenLinearPolicy, "update", create=True,
                                                      side_effect=AssertionError("policy update")))
            result = evaluate_frozen_policy(self.policy, self.receipt, episodes_per_family=1, seed=44)
            for sentinel in sentinels + [update]:
                sentinel.assert_not_called()
        self.assertEqual(result["inference_audit"]["mutable_policy_buffers"], 0)
        self.assertEqual(result["inference_audit"]["arbitrary_policy_callbacks"], 0)

    def test_frozen_evaluation_refuses_write_attempt_and_replaced_transform(self):
        from thrusterchain.environment import ThrusterChainEnv
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forbidden.npz"
            def illegal_step(env, action):
                save_dataset(self.dataset, path)
            with patch.object(ThrusterChainEnv, "step", illegal_step):
                with self.assertRaisesRegex(DatasetError, "disabled during frozen evaluation"):
                    evaluate_frozen_policy(self.policy, self.receipt, episodes_per_family=1)
            self.assertFalse(path.exists())
        with patch.object(FrozenLinearPolicy, "predict", lambda policy, observation: np.zeros(4)):
            with self.assertRaisesRegex(ProtocolError, "implementation was replaced"):
                evaluate_frozen_policy(self.policy, self.receipt, episodes_per_family=1)

    def test_imported_mutation_aliases_cannot_bypass_shared_guard(self):
        # These aliases were imported when this module loaded, before any guard.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forbidden.npz"
            operations = [lambda: train_behavior_cloning(self.dataset),
                          lambda: save_dataset(self.dataset, path),
                          lambda: generate_dataset(episodes=10),
                          lambda: calibrate_frozen_policy(self.dataset, self.policy, self.receipt),
                          lambda: freeze_policy(self.policy, self.receipt, directory)]
            with inference_evaluation_guard() as ledger:
                for operation in operations:
                    with self.assertRaisesRegex(DatasetError, "disabled during frozen evaluation"):
                        operation()
                self.assertEqual(ledger["forbidden_api_calls"], len(operations))
            self.assertFalse(path.exists())
            self.assertFalse((Path(directory) / "frozen_policy.npz").exists())
        with patch("thrusterchain.regular._affine_inference", lambda *args: np.zeros(4)):
            with self.assertRaisesRegex(ProtocolError, "implementation was replaced"):
                evaluate_frozen_policy(self.policy, self.receipt, episodes_per_family=1)

    def test_mutable_or_unbound_policy_refused(self):
        class StatefulPolicy:
            def predict(self, observation):
                return np.zeros(4, dtype=np.float32)
        with self.assertRaisesRegex(ProtocolError, "prediction-only"):
            evaluate_frozen_policy(StatefulPolicy(), self.receipt, episodes_per_family=1)
        other = FrozenLinearPolicy(self.policy.mean, self.policy.scale, self.policy.coefficients + .001, self.policy.ridge)
        with self.assertRaisesRegex(ProtocolError, "changed after selection"):
            evaluate_frozen_policy(other, self.receipt, episodes_per_family=1)

    def test_generation_refuses_invalid_inputs_before_collecting(self):
        for kwargs in ({"episodes": 9}, {"episodes": True}, {"seed": -1},
                       {"provenance": "unspecified"}, {"task_semantics": "ambiguous"}, {"epsilon": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(DatasetError):
                generate_dataset(**kwargs)


if __name__ == "__main__":
    unittest.main()
