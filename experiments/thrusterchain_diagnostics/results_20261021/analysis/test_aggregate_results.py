"""Focused checks of paired inference and cross-source artifact contracts."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from aggregate_results import paired_stats, sample_stats, verify_summary


class PairedInferenceTests(unittest.TestCase):
    def test_known_sample_sd_interval_and_exact_sign_flip(self):
        result = paired_stats(np.zeros(5), np.arange(1., 6.))
        self.assertEqual(result["mean_difference"], 3.)
        self.assertAlmostEqual(result["sample_sd_difference"], np.sqrt(2.5))
        margin = 2.7764451051977987 * np.sqrt(2.5 / 5)
        np.testing.assert_allclose(result["ci95"], [3 - margin, 3 + margin])
        self.assertEqual(result["sign_flip_two_sided_p"], 2 / 32)
        self.assertEqual(result["n"], 5)
        self.assertEqual(result["df"], 4)

    def test_paired_shift_invariance_and_arm_swap(self):
        baseline = np.array([-100., -150., -200., -300., -50.])
        bca = baseline + [1., -2., 3., -4., 5.]
        a, shifted, swapped = paired_stats(baseline, bca), paired_stats(baseline + 1000, bca + 1000), paired_stats(bca, baseline)
        self.assertEqual(a, shifted)
        np.testing.assert_allclose(swapped["ci95"], -np.asarray(a["ci95"])[::-1])
        self.assertEqual(swapped["sign_flip_two_sided_p"], a["sign_flip_two_sided_p"])
        self.assertEqual(paired_stats(baseline, baseline)["sign_flip_two_sided_p"], 1.)

    def test_missing_nonfinite_and_broadcastable_pairs_rejected(self):
        for baseline, bca in (([0.], range(5)), (range(4), range(4)), (range(5), [1, 2, 3, 4, np.nan]), (range(6), range(6))):
            with self.assertRaises(ValueError):
                paired_stats(baseline, bca)
        with self.assertRaises(ValueError):
            sample_stats(np.zeros(400))


class ResultIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(Path(__file__).with_name("aggregate.json").read_text(encoding="utf-8"))

    def test_union_and_pair_arithmetic_preserve_all_seeds(self):
        result = self.result
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["jobs"]), 30)
        self.assertEqual(len({j["job_id"] for j in result["jobs"]}), 30)
        self.assertEqual(sum(j["host"] == "Titan" for j in result["jobs"]), 8)
        self.assertEqual(sum(j["host"] == "Zeus" for j in result["jobs"]), 22)
        for comparison in result["paired_comparisons"]:
            pairs = comparison["replicates"]
            self.assertEqual([p["replicate"] for p in pairs], list(range(1, 6)))
            differences = [p["bca"] - p["td3_bc"] for p in pairs]
            np.testing.assert_allclose(differences, comparison["differences"])
            self.assertAlmostEqual(np.mean(differences), comparison["mean_difference"])

    def test_wrong_bank_or_checkpoint_cannot_enter_summary(self):
        record = self.result["jobs"][0]
        original = record["evaluation"]["final"]
        for field, replacement in (("bank_sha256", "wrong-bank"), ("checkpoint", dict(original["checkpoint"], update=50000))):
            summary = copy.deepcopy(original)
            summary[field] = replacement
            with self.assertRaises(ValueError):
                verify_summary(summary, "final", self.result["banks"], self.result["oracle"], record["job"])

    def test_titan_missing_bytes_are_not_claimed_independently_verified(self):
        for record in self.result["jobs"]:
            if record["host"] == "Titan":
                self.assertFalse(record["source"]["independently_verified_arrays"])
                self.assertFalse(record["source"]["independently_verified_checkpoint"])
                self.assertIsNone(record["refresh_audit"])


if __name__ == "__main__":
    unittest.main()
