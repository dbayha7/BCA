"""Pure JSON/source tests: no model imports, datasets, simulation, fitting or scientific rerun.

The legacy test accepted unversioned arbitrary synthetic attempts. That is the unsafe behavior
being removed; these tests require explicit reviewed producer and complete evidence instead.
"""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amended_analysis as A


class AccountingGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = A.correction_contract()
        cls.attempt = A.REPO / cls.contract["attempt_relative"]
        cls.unit = "warmup_s0_cutoff_stall3"
        cls.report = json.loads((cls.attempt / cls.unit / "report.json").read_text())

    def unknown(self, report, *, unit=None, episodes=1280):
        value = A.corrected_steps(unit or self.unit, report, episodes)
        self.assertIsNone(value["corrected"])
        self.assertIsNone(value["loads"])
        self.assertNotIn("lower bound", value["status"])

    def test_exact_complete_legacy_training_record(self):
        value = A.corrected_steps(self.unit, self.report, 1280)
        self.assertEqual(value["recorded"], 1280)
        self.assertEqual(value["corrected"], 128000)

    def test_already_corrected_count_is_not_corrected_again(self):
        report = deepcopy(self.report)
        report["simulator_steps_exact"] = 128000
        self.unknown(report)

    def test_new_producer_with_old_count_is_unknown(self):
        report = deepcopy(self.report)
        report["producer_sha256"] = "0" * 64
        self.unknown(report)

    def test_incomplete_interrupted_or_missing_status_is_unknown(self):
        for status in (None, "running", "failed", "interrupted"):
            with self.subTest(status=status):
                report = deepcopy(self.report)
                report["status"] = status
                self.unknown(report)

    def test_partial_training_count_is_not_a_lower_bound(self):
        report = deepcopy(self.report)
        report["simulator_steps_exact"] = 123
        self.unknown(report)

    def test_missing_float_and_boolean_count_are_not_exact(self):
        for count in (None, 1280.0, True):
            with self.subTest(count=count):
                report = deepcopy(self.report)
                report["simulator_steps_exact"] = count
                self.unknown(report)

    def test_wrong_episode_count_rejected(self):
        for episodes in (1281, 1280.0, True):
            self.unknown(self.report, episodes=episodes)

    def test_unknown_unit_and_changed_report_content_rejected(self):
        self.unknown(self.report, unit="warmup_s99_cutoff_stall3")
        changed = deepcopy(self.report)
        changed["elapsed_seconds"] += 1
        self.unknown(changed)

    def test_incomplete_evaluation_no_default_two_loads(self):
        unit = "evaluate_s0"
        report = json.loads((self.attempt / unit / "report.json").read_text())
        report.pop("runs")
        self.unknown(report, unit=unit)

    def test_unsupported_attempt_rejected_before_decision(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(A.stage1, "decide") as decide:
            with self.assertRaisesRegex(ValueError, "Unsupported attempt"):
                A.analyse(directory)
            decide.assert_not_called()

    def test_receipt_and_source_mismatch_rejected_before_decision(self):
        original = A.sha_file
        for group in ("receipt_files", "source_files"):
            relative = next(iter(self.contract[group]))
            target = (A.REPO / relative).resolve()
            def changed(path):
                return "0" * 64 if Path(path).resolve() == target else original(path)
            with self.subTest(group=group), patch.object(A, "sha_file", side_effect=changed), patch.object(A.stage1, "decide") as decide:
                with self.assertRaisesRegex(ValueError, "evidence changed or missing"):
                    A.analyse(self.attempt)
                decide.assert_not_called()

    def test_contract_tamper_rejected(self):
        with patch.object(A, "CORRECTION_CONTRACT_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "contract changed"):
                A.corrected_steps(self.unit, self.report, 1280)

    def test_saved_complete_attempt_total_and_decision_preserved_read_only(self):
        before = {relative: A.sha_file(A.REPO / relative) for relative in self.contract["receipt_files"]}
        out = A.analyse(self.attempt, exploratory=True)
        after = {relative: A.sha_file(A.REPO / relative) for relative in self.contract["receipt_files"]}
        self.assertEqual(before, after)
        total = out["R3_accounting"]["totals"]
        self.assertEqual(total["recorded_exact_total"], 6305360)
        self.assertEqual(total["corrected_total"], 10994000)
        self.assertEqual(total["evaluation_units_corrected_total"], 7538000)
        self.assertEqual(total["units_without_counts"], [])
        self.assertEqual(len(out["R3_accounting"]["units"]), 32)
        self.assertEqual(sum(u["loads"] for u in out["R3_accounting"]["units"].values()), 37)
        self.assertEqual(out["original"]["label"], "INCONCLUSIVE")
        self.assertEqual(out["R1_v3_rules"]["label_under_v3"], "INCONCLUSIVE")
        self.assertTrue(all(not v["ok"] for v in out["R1_v3_rules"]["V0f_by_seed"].values()))


if __name__ == "__main__":
    unittest.main()
