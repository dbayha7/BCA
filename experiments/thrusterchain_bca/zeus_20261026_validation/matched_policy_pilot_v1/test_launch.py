"""Synthetic process/reservation fixtures; never imports models or simulates."""
import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
import launch

# Windows venv python.exe is a redirector. Use the actual interpreter for
# process-lifetime fixtures so killing the fixture process closes its log.
FIXTURE_PYTHON = getattr(sys, "_base_executable", sys.executable)


class LauncherFixtures(unittest.TestCase):
    def test_approved_protocol(self):
        p = launch.read(launch.HERE / "protocol.json")
        self.assertIs(launch.validate_protocol(p), p)

    def test_budget_or_arm_changes_rejected(self):
        for key, value in (("critic_updates", 2001), ("total_timeout_seconds", 4201),
                           ("bca_kappa", 1), ("simulator_transition_cap", 22401),
                           ("automatic_restart", True)):
            p = launch.read(launch.HERE / "protocol.json")
            p[key] = value
            with self.assertRaises(ValueError):
                launch.validate_protocol(p)

    def test_reservation_exclusive_and_failed_evidence_preserved(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            root = Path(temp)
            launch.reserve(root, "fixture")
            launch.save_new(root / "run/failed.json", {"status": "synthetic_failed"})
            with self.assertRaises((ValueError, FileExistsError)):
                launch.reserve(root, "retry")
            self.assertEqual(launch.read(root / "ATTEMPT.json")["token"], "fixture")
            self.assertTrue((root / "run/failed.json").exists())

    def test_partial_reservation_without_run_is_not_restarted(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            root = Path(temp)
            launch.save_new(root / "ATTEMPT.json", {"token": "old"})
            with self.assertRaises(FileExistsError):
                launch.reserve(root, "retry")

    def test_atomic_progress_and_readonly_status(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            root = Path(temp)
            self.assertEqual(launch.read_status(root)["status"], "not_started")
            launch.reserve(root, "fixture")
            (root / "run/seed0_native").mkdir()
            launch.atomic_json(root / "PROGRESS.json", {"status": "running", "current_job": "seed0_native"})
            launch.atomic_json(root / "run/seed0_native/progress.json", {"ledger": {"critic_completed": 20}})
            launch.atomic_json(root / "run/seed0_native/progress.json", {"ledger": {"critic_completed": 40}})
            status = launch.read_status(root)
            self.assertEqual(status["worker_progress"]["ledger"]["critic_completed"], 40)
            self.assertIn("uncertain", status["note"])
            launch.save_new(root / "RUN_RESULT.json", {"status": "synthetic_terminal"})
            self.assertEqual(launch.read_status(root)["status"], "synthetic_terminal")

    def test_success_child(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            r = launch.supervise([FIXTURE_PYTHON, "-c", "print('fixture')"], temp, 5, announce=False)
            self.assertEqual(r["exit_code"], 0)
            self.assertFalse(r["timed_out"] or r["interrupted"])

    def test_failure_child(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            r = launch.supervise([FIXTURE_PYTHON, "-c", "raise SystemExit(7)"], temp, 5, announce=False)
            self.assertEqual(r["exit_code"], 7)

    def test_timeout_stops_owned_child(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            r = launch.supervise([FIXTURE_PYTHON, "-c", "import time; time.sleep(3)"], temp, .1, announce=False)
            self.assertTrue(r["timed_out"])
            self.assertNotEqual(r["exit_code"], 0)

    def test_global_deadline_shorter_than_worker_cap(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            r = launch.supervise([FIXTURE_PYTHON, "-c", "import time; time.sleep(3)"], temp, 5,
                                announce=False, deadline=time.monotonic() + .1)
            self.assertTrue(r["timed_out"])
            self.assertLess(r["elapsed_seconds"], 2.)

    def test_interrupt_kills_owned_child(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            child = MagicMock()
            child.wait.side_effect = [KeyboardInterrupt(), -9]
            child.poll.return_value = None
            child.returncode = -9
            with patch.object(launch.subprocess, "Popen", return_value=child), patch.object(launch.os, "name", "nt"):
                r = launch.supervise(["fixture"], temp, 5, announce=False)
            self.assertTrue(r["interrupted"])
            child.kill.assert_called_once()

    def test_final_report_does_not_overwrite(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            path = Path(temp) / "RUN_RESULT.json"
            launch.save_new(path, {"status": "first"})
            with self.assertRaises(FileExistsError):
                launch.save_new(path, {"status": "second"})
            self.assertEqual(launch.read(path)["status"], "first")

    def test_partial_report_read_is_recorded(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            path = Path(temp) / "report.json"
            path.write_text('{"status":', encoding="utf-8")
            report, error = launch.read_report(path)
            self.assertIsNone(report)
            self.assertIn("JSONDecodeError", error)

    def _synthetic_sequence(self, root, fail_at=None, corrupt_eval=False):
        p = launch.read(launch.HERE / "protocol.json")
        (root / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
        env = root / "synthetic_env"
        env.mkdir()
        (env / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
        called = []
        def fake_supervise(command, output, cap, *args, **kwargs):
            job = Path(output).name
            called.append(job)
            failed = job == fail_at
            report = {"status": "synthetic_failure" if failed else
                      ("completed_bounded_policy_evaluation" if job == "evaluation" else "completed_bounded_policy_training"),
                      "ledger": {"synthetic": True}, "execution_counts_known": True}
            if job == "evaluation" and corrupt_eval:
                (Path(output) / "report.json").write_text('{"status":', encoding="utf-8")
            else:
                launch.save_new(Path(output) / "report.json", report)
            return {"exit_code": 1 if failed else 0, "timed_out": False, "interrupted": False,
                    "error": None, "elapsed_seconds": .001}
        with patch.object(launch, "HERE", root), patch.object(launch, "EXPECTED_ENV", str(env)), \
             patch.object(launch.sys, "platform", "linux"), patch.object(launch.sys, "prefix", str(env)), \
             patch.object(launch, "scoped_process_check"), patch.object(launch, "verify_prepared", return_value=(p, {})), \
             patch.object(launch, "supervise", side_effect=fake_supervise), contextlib.redirect_stdout(io.StringIO()):
            code = launch.execute(p, {})
        return code, called, launch.read(root / "RUN_RESULT.json")

    def test_incomplete_job_stops_sequence_and_no_evaluation(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            code, called, receipt = self._synthetic_sequence(Path(temp), fail_at="seed0_bca")
            self.assertEqual(called, ["seed0_native", "seed0_bca"])
            self.assertEqual(code, 1)
            self.assertEqual(receipt["status"], "stopped_after_incomplete_job")
            self.assertFalse(receipt["execution_counts_known"])

    def test_evaluation_only_after_six_successes(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            code, called, receipt = self._synthetic_sequence(Path(temp))
            self.assertEqual(called, launch.RUN_ORDER + ["evaluation"])
            self.assertEqual(code, 0)
            self.assertTrue(receipt["execution_counts_known"])

    def test_corrupt_evaluation_still_saves_terminal_receipt(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE, prefix="fixture_") as temp:
            code, called, receipt = self._synthetic_sequence(Path(temp), corrupt_eval=True)
            self.assertEqual(code, 1)
            self.assertEqual(len(called), 7)
            self.assertIn("JSONDecodeError", receipt["evaluation_report_read_error"])
            self.assertIn("JSONDecodeError", receipt["jobs"][-1]["report_read_error"])
            self.assertFalse(receipt["execution_counts_known"])


if __name__ == "__main__":
    unittest.main()
