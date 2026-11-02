"""Pure fixture regressions: no models, simulator calls, GPU or scientific jobs."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import json
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage1_worker as W
import stage2_worker as W2
import units as U


class IntegrityClosure(unittest.TestCase):
    def close(self, report=None, effect=None, changed=None):
        report = {"status": "completed", "result": {"kept": 4}} if report is None else report
        with mock.patch.object(W.integrity, "changed_files", side_effect=effect, return_value=changed or []):
            return W._close_report(report, SimpleNamespace(steps=7, attempted=8), {"integrity": {"files": {"x": "hash"}}})

    def test_verification_exception_cannot_complete(self):
        for exc in (OSError("unreadable"), ValueError("malformed"), SystemExit("missing"), KeyboardInterrupt()):
            with self.subTest(exception=type(exc).__name__):
                result = self.close(effect=exc)
                self.assertEqual((result["status"], result["unit_label"]), ("failed", "INCOMPLETE"))
                self.assertFalse(result["integrity_after"]["verified"])
                self.assertIn(type(exc).__name__, result["error"])
                self.assertEqual((result["simulator_steps_exact"], result["simulator_steps_attempted"]), (7, 8))
                self.assertEqual(result["result"], {"kept": 4})

    def test_changed_files_cannot_complete(self):
        result = self.close(changed=["changed.py"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("changed.py", result["error"])

    def test_original_failure_is_preserved(self):
        for status in ("failed", "interrupted"):
            result = self.close({"status": status, "error": "original failure", "traceback": "original trace"}, effect=OSError("check failed"))
            self.assertEqual(result["status"], status)
            self.assertEqual(result["error"], "original failure")
            self.assertEqual(result["traceback"], "original trace")
            self.assertIn("check failed", result["integrity_after"]["error"])

    def test_success_requires_successful_check(self):
        result = self.close()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["integrity_after"], {"verified": 1, "changed": []})

    def test_missing_actual_record_fails_closed(self):
        result = W._close_report({"status": "completed"}, None, {})
        self.assertEqual(result["status"], "failed")
        self.assertIn("no integrity record", result["error"])

    def test_stage2_uses_the_same_helper(self):
        self.assertIs(W2.W1._close_report, W._close_report)

    def test_both_worker_entrypoints_persist_failed_integrity_receipt(self):
        # Synthetic dispatcher and counter: exercise real run_job/report serialization,
        # but never enter the simulator, model loader or environment-configuration path.
        for worker in (W, W2):
            with self.subTest(worker=worker.__name__), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "job.json"
                path.write_text(json.dumps({"id": "fixture", "kind": "fixture", "timeout_seconds": 1,
                                            "integrity": {"files": {"x": "hash"}}}), encoding="utf-8")
                counter = mock.MagicMock()
                counter.steps, counter.attempted = 0, 0
                with mock.patch.dict(worker.KINDS, {"fixture": lambda *args: {"fixture": True}}), \
                        mock.patch.object(W, "producer", return_value=None), \
                        mock.patch.object(W.step03, "set_worker_environment"), \
                        mock.patch.object(U, "SimulatorCounter", return_value=counter), \
                        mock.patch.object(W.integrity, "changed_files", side_effect=OSError("unreadable source")):
                    result = worker.run_job(path)
                saved = json.loads((path.parent / "report.json").read_text())
                self.assertEqual(saved, result)
                self.assertEqual((saved["status"], saved["unit_label"]), ("failed", "INCOMPLETE"))
                self.assertEqual((saved["simulator_steps_exact"], saved["simulator_steps_attempted"]), (0, 0))
                self.assertTrue(saved["fixture"])
                self.assertIn("unreadable source", saved["error"])


class TimingAccounting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def unit(self, name="attempt_1/job", supervisor=None, report=None, job=None):
        path = self.root / name
        path.mkdir(parents=True)
        for file, data in (("SUPERVISOR_RESULT.json", supervisor), ("report.json", report), ("job.json", job)):
            if data is not None:
                (path / file).write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_missing_timing_and_cap_are_unknown_at_every_level(self):
        path = self.unit(report={"simulator_steps_exact": 10})
        entry = U._unit_spend(path)
        self.assertIsNone(entry["seconds"])
        self.assertIsNone(entry["upper_bound_seconds"])
        spend = U.attempt_spend(self.root / "attempt_1")
        self.assertEqual(spend["unbounded_seconds_units"], ["job"])
        self.assertIsNone(spend["unknown_seconds_upper_bound"])
        ledger = U.ledger(self.root)
        self.assertEqual(ledger["prior_unbounded_seconds_units"], ["attempt_1/job"])
        self.assertIsNone(ledger["prior_unknown_seconds_upper_bound"])
        total = U.physical(self.root / "attempt_2", ledger)
        self.assertIsNone(total["cumulative_unknown_seconds_upper_bound"])
        self.assertEqual(total["cumulative_unbounded_seconds_units"], 1)
        self.assertEqual(total["cumulative_known_simulator_steps"], 10)

    def test_nominal_timeout_does_not_prove_physical_upper_bound(self):
        for i, cap in enumerate((300, 0)):
            entry = U._unit_spend(self.unit(f"attempt_1/job{i}", job={"timeout_seconds": cap}))
            self.assertEqual(entry["configured_timeout_seconds"], cap)
            self.assertIsNone(entry["upper_bound_seconds"])

    def test_invalid_measurements_and_caps_never_become_zero(self):
        for i, value in enumerate((None, -1, float("nan"), float("inf"), "23", True, {}, [])):
            with self.subTest(value=value):
                entry = U._unit_spend(self.unit(f"attempt_1/job{i}", supervisor={"elapsed_seconds": value}, job={"timeout_seconds": value}))
                self.assertIsNone(entry["seconds"])
                self.assertIsNone(entry["configured_timeout_seconds"])
                self.assertIsNone(entry["upper_bound_seconds"])
                self.assertEqual(entry["timing_errors"], ["supervisor: invalid elapsed_seconds"])

    def test_valid_zero_is_measured_zero(self):
        entry = U._unit_spend(self.unit(supervisor={"elapsed_seconds": 0}))
        self.assertEqual(entry["seconds"], 0)
        self.assertEqual(entry["seconds_source"], "supervisor")
        self.assertNotIn("upper_bound_seconds", entry)

    def test_precedence_and_documented_worker_fallback(self):
        entry = U._unit_spend(self.unit(supervisor={"elapsed_seconds": 4}, report={"elapsed_seconds": 3}))
        self.assertEqual((entry["seconds"], entry["seconds_source"]), (4, "supervisor"))
        entry = U._unit_spend(self.unit("attempt_1/fallback", supervisor={"elapsed_seconds": -1}, report={"elapsed_seconds": 3}))
        self.assertEqual((entry["seconds"], entry["seconds_source"]), (3, "worker"))
        self.assertIn("timing_errors", entry)

    def test_known_and_unknown_do_not_double_count_current_attempt(self):
        self.unit(supervisor={"elapsed_seconds": 2}, report={"simulator_steps_exact": 3})
        self.unit("attempt_2/known", supervisor={"elapsed_seconds": 5}, report={"simulator_steps_exact": 7})
        self.unit("attempt_2/unknown", job={"timeout_seconds": 20}, report={"simulator_steps_exact": 11})
        prior = U.ledger(self.root, current_attempt="attempt_2")
        total = U.physical(self.root / "attempt_2", prior)
        self.assertEqual(total["cumulative_known_seconds"], 7)
        self.assertEqual(total["cumulative_known_simulator_steps"], 21)
        self.assertIsNone(total["cumulative_unknown_seconds_upper_bound"])
        self.assertEqual(total["cumulative_unbounded_seconds_units"], 1)

    def test_legacy_zero_upper_bound_is_not_trusted(self):
        for upper in (None, 0, 300):
            prior = {"prior_unknown_units": ["attempt_1/missing"], "prior_unknown_seconds_upper_bound": upper}
            total = U.physical(self.root / "missing", prior)
            self.assertIsNone(total["cumulative_unknown_seconds_upper_bound"])
            self.assertEqual(total["cumulative_unbounded_seconds_units"], 1)

    def test_no_units_means_no_unknown_work(self):
        total = U.physical(self.root / "missing", U.ledger(self.root))
        self.assertEqual(total["cumulative_unknown_seconds_upper_bound"], 0)
        self.assertEqual(total["cumulative_unbounded_seconds_units"], 0)


if __name__ == "__main__":
    unittest.main()
