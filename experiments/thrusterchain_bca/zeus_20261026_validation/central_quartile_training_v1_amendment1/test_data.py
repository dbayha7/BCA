"""Synthetic data assembly and role-boundary checks; no models or simulation."""
from collections import namedtuple
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import data

NATIVE = SimpleNamespace(Transition=namedtuple("Transition", data.FIELDS))


def fixture():
    nt, nc = 6, 2
    episodes = np.asarray([2, 5, 6, 9, 12, 15, 20, 23], np.int64)
    obs = np.zeros((nt+nc, 101, 14), np.float32)
    obs[:, :, 0] = episodes[:, None]
    obs[:, :, 1] = np.arange(101)[None, :]
    obs[:, :, 10] = np.where(np.arange(101)%20 < 10, .6, .2).astype(np.float32)
    obs[:, :, -1] = (1 - np.arange(101, dtype=np.float64)/100).astype(np.float32)
    actions = np.zeros((nt+nc, 100, 4), np.float32)
    actions[:, :, 0] = np.resize([0., -.5, .5, .9], 100)
    rewards = (episodes[:, None]*1000 + np.arange(100)[None, :]).astype(np.float32)
    ended = np.zeros((nt+nc, 100), bool); ended[:, -1] = True
    source = {"arrays": {"observations": obs, "actions": actions, "rewards": rewards,
                         "terminated": ended, "truncated": np.zeros_like(ended)},
              "train_count": nt, "episode_ids": episodes, "trajectory_ids": episodes.copy(),
              "role_codes": np.r_[np.zeros(nt, np.uint8), np.ones(nc, np.uint8)],
              "dataset_sha256": "synthetic_source_only"}
    return source


class DataTests(unittest.TestCase):
    def setUp(self):
        self.source = fixture()
        self.plan = data._derive(self.source)[0]

    def assembly(self, condition):
        return data._assemble(NATIVE, condition, self.source, self.plan)

    def test_intact_tuples_and_terminal_semantics(self):
        result = self.assembly("hole")
        for i, identity in enumerate(result.source_row_ids):
            episode, t = divmod(int(identity), 100)
            self.assertEqual(result.train.obs[i, 0], episode)
            self.assertEqual(result.train.obs[i, 1], t)
            self.assertEqual(result.train.next_obs[i, 1], t+1)
            self.assertEqual(result.train.reward[i], episode*1000+t)
            self.assertEqual(result.train.done[i], float(t == 99))
        self.assertEqual(result.train.obs.dtype, np.float32)
        self.assertEqual(int(result.train.done.sum()), 6)
        with self.assertRaises(ValueError):
            result.train.action.setflags(write=True)
        self.assertIs(result.receipt, result.metadata)

    def test_common_normalization_is_exact_retained_intersection(self):
        results = {name: self.assembly(name) for name in data.CONDITIONS}
        expected = np.intersect1d(results["hole"].source_row_ids, results["random"].source_row_ids)
        self.assertLess(len(expected), len(results["hole"].source_row_ids))
        self.assertLess(len(expected), len(results["random"].source_row_ids))
        for result in results.values():
            np.testing.assert_array_equal(result.normalization_source_row_ids, expected)
            np.testing.assert_array_equal(result.normalization_fixture.obs, results["full"].normalization_fixture.obs)
            self.assertFalse(np.intersect1d(result.normalization_source_row_ids, result.heldout_ids["row_ids"]).size)
            self.assertTrue(np.all(np.isin(result.normalization_source_row_ids, result.source_row_ids)))
            self.assertFalse(data._geometry().region_masks(result.normalization_fixture.obs,
                                                           result.normalization_fixture.action)["hole"].any())
        # Changing calibration-only physical observations cannot change normalization.
        changed = deepcopy(self.source)
        changed["arrays"]["observations"][6:, :, 0] += 1000
        altered = data._derive(changed)[0]
        self.assertEqual(altered["common_normalization"], self.plan["common_normalization"])

    def test_full_hole_random_counts_state_strata_and_common_calibration(self):
        results = {name: self.assembly(name) for name in data.CONDITIONS}
        self.assertEqual(len(results["full"].train.obs), 600)
        self.assertEqual(len(results["hole"].train.obs), len(results["random"].train.obs))
        self.assertFalse(data._geometry().region_masks(results["hole"].train.obs, results["hole"].train.action)["hole"].any())
        self.assertTrue(data._geometry().region_masks(results["full"].train.obs, results["full"].train.action)["hole"].any())
        for name, result in results.items():
            np.testing.assert_array_equal(result.heldout_ids["row_ids"], results["full"].heldout_ids["row_ids"])
            np.testing.assert_array_equal(result.heldout.action, results["full"].heldout.action)
            self.assertFalse(data._geometry().region_masks(result.heldout.obs, result.heldout.action)["hole"].any())
            self.assertEqual(len(np.unique(result.heldout_ids["episode_ids"])), 2)
            self.assertFalse(np.intersect1d(result.source_row_ids//100, result.heldout_ids["episode_ids"]).size)
        # Exact count matching is episode x ten-step bin x observed load side.
        def strata(result):
            ids = result.source_row_ids
            high = data._geometry().region_masks(result.train.obs, result.train.action)["high"]
            triples = np.stack((ids//100, (ids%100)//10, high.astype(np.int64)), axis=1)
            unique, counts = np.unique(triples, axis=0, return_counts=True)
            return unique, counts
        for left, right in zip(strata(results["hole"]), strata(results["random"])):
            np.testing.assert_array_equal(left, right)

    def test_probe_and_scale_banks_only_retained_training_rows(self):
        for name in data.CONDITIONS:
            result = self.assembly(name)
            record = result.metadata
            probes = np.asarray(record["gradient_probe_source_row_ids"])
            scale = np.asarray(record["scale_bank_source_row_ids"])
            self.assertEqual(len(probes), 6)
            self.assertEqual(len(np.unique(probes//100)), 6)
            self.assertEqual(len(np.unique(scale)), 256)
            self.assertTrue(np.all(np.isin(probes, result.source_row_ids)))
            self.assertTrue(np.all(np.isin(scale, result.source_row_ids)))
            self.assertEqual(record["protected_transition_rows_decoded"], 0)
            self.assertEqual(record["development_probe_transition_rows_decoded"], 0)

    def test_plan_tampering_and_forbidden_roles_rejected(self):
        for field in ("training_source_row_ids_sha256", "gradient_probe_source_row_ids_sha256",
                      "normalization_source_row_ids_sha256", "calibration_source_row_ids_sha256"):
            altered = deepcopy(self.plan)
            altered["conditions"]["hole"][field] = "bad hash"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "DATA_PLAN"):
                data._assemble(NATIVE, "hole", self.source, altered)
        self.source["role_codes"][-1] = 2
        with self.assertRaisesRegex(ValueError, "Protected"):
            data._derive(self.source)

    def test_source_overlap_time_or_terminal_changes_rejected(self):
        cases = []
        source = deepcopy(self.source); source["episode_ids"][-1] = source["episode_ids"][0]; cases.append((source, "identity overlap"))
        source = deepcopy(self.source); source["arrays"]["observations"][0, 50, -1] = 0.; cases.append((source, "remaining-time"))
        source = deepcopy(self.source); source["arrays"]["terminated"][0, 50] = True; cases.append((source, "finite-horizon"))
        for source, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                data._derive(source)

    def test_empty_eligible_calibration_episode_fails_without_backfill(self):
        self.source["arrays"]["observations"][6, :, 10] = .6
        self.source["arrays"]["actions"][6, :, 0] = 0
        with self.assertRaisesRegex(ValueError, "Calibration episode"):
            data._derive(self.source)

    def test_production_decoder_requests_only_896_train_calibration_episodes(self):
        codes = np.r_[np.zeros(768, np.uint8), np.ones(256, np.uint8), np.full(128, 2, np.uint8), np.full(128, 3, np.uint8)]
        p = {"train_indices": np.arange(768), "calibration_indices": np.arange(768, 896),
             "vectors": {"split_codes": codes, "group_ids": np.arange(1280), "trajectory_ids": np.arange(1280)},
             "metadata": {"behavior": {"name": "continuous_region_mixture_v1", "epsilon": .25, "state_independent": True}}}
        calls = []
        fake = SimpleNamespace(provenance=lambda: p, DATA="not_opened", EXPECTED_DATA_SHA="synthetic",
                               EXPECTED_BINDINGS_SHA="synthetic", TRANSITION_NAMES=("observations", "actions", "rewards", "terminated", "truncated"),
                               read_episodes=lambda archive, name, indices, codes: calls.append((name, indices.copy())) or None)
        with patch.object(data, "_module", return_value=fake), patch.object(data.zipfile, "ZipFile", return_value=MagicMock()):
            source = data._read_source()
        self.assertEqual(len(calls), 5)
        for _, indices in calls:
            np.testing.assert_array_equal(indices, np.arange(896))
        self.assertEqual(source["train_count"], 768)
        calls.clear(); codes[895] = 2
        with patch.object(data, "_module", return_value=fake), patch.object(data.zipfile, "ZipFile", return_value=MagicMock()):
            with self.assertRaisesRegex(ValueError, "Protected"):
                data._read_source()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
