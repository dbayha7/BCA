"""Recovery fixtures only: no child process, model or simulator is launched."""
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("path_a_recovery_tested", Path(__file__).with_name("recovery_launch.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def fake_signals():
    return SimpleNamespace(SIGINT=2, SIGTERM=15, SIGHUP=1, SIGKILL=9,
                           getsignal=lambda s: 0, signal=lambda s, value: None)


class RecoveryFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_exact_plan_skips_only_completed_warmup(self):
        remaining = R.jobs()
        self.assertEqual(len(remaining), 18)
        self.assertNotIn(R.REUSED, [j["id"] for j in remaining])
        self.assertEqual(remaining[0]["id"], "honeypot_s0_native")
        self.assertEqual(remaining[-1]["arm"], "evaluation")
        self.assertEqual(sum(j["arm"] == "warmup" for j in remaining), 5)

    def test_final_accounting_preserves_unknown_lost_tail(self):
        accepted = [{"status": "completed_training_job", "counts": {"completed": R.JOB_CAPS}} for _ in range(17)]
        accepted.append({"status": "completed_frozen_evaluation", "physical_counts": {"completed": {"returns": 123}}})
        result = R.accounting(accepted, prior_simulator_steps=192975)
        self.assertEqual(result["cumulative_physical_lower"], dict(critic=184214, actor=92108, auxiliary=184214, calibrations=21))
        self.assertEqual(result["cumulative_physical_upper"], R.CUMULATIVE_CAPS)
        self.assertFalse(result["cumulative_execution_counts_known"])
        self.assertEqual(result["new_verified_completed_training_counts"]["critic"], 170000)
        self.assertEqual(result["cumulative_simulator_transition_upper"], 193098)
        self.assertEqual(Decimal("11490")-Decimal("769.481604245")-Decimal("60")-Decimal("10660"), Decimal("0.518395755"))

    def test_evaluation_interruption_uses_last_valid_lower_and_full_upper(self):
        out = self.root / "eval"
        out.mkdir()
        counts = dict(returns=1000, own_qpi=20, common_reference=30, prefix_parity=50)
        (out / "progress.jsonl").write_text(json.dumps({"completed": counts}) + "\n{truncated", encoding="utf-8")
        result = R.accounting([], {"arm": "evaluation"}, prior_simulator_steps=192975, evaluation_output=out)
        self.assertEqual(result["new_evaluation_transition_bounds"]["lower"], 1100)
        self.assertEqual(result["new_evaluation_transition_bounds"]["upper"], 237600)
        self.assertFalse(result["new_evaluation_transition_bounds"]["execution_counts_known"])
        self.assertIsNotNone(result["new_evaluation_transition_bounds"]["progress_read_error"])
        missing = R.evaluation_bounds([], {"arm": "evaluation"}, self.root / "absent")
        self.assertEqual((missing["lower"], missing["upper"]), (0, 237600))
        self.assertFalse(missing["execution_counts_known"])

    def test_inventory_rejects_added_entry_and_changed_bytes(self):
        source = self.root / "old"
        source.mkdir()
        (source / "a").write_bytes(b"unchanged")
        expected = R.inventory(source)
        (source / "new").mkdir()
        with self.assertRaisesRegex(ValueError, "directory entries changed"):
            R.verify_inventory(source, expected)
        (source / "a").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            R.verify_inventory(source, expected)

    def test_warmup_mirror_is_byte_exact_and_exclusive(self):
        old, out = self.root / "old", self.root / "new"
        warm = old / R.REUSED
        warm.mkdir(parents=True); out.mkdir()
        for name in ("report.json", "checkpoint.json", "full_state.msgpack", "WORKER_EXIT.json"):
            (warm / name).write_bytes((name + " original bytes").encode())
        manifest = {"lineage": {"old_run": R.inventory(old)}}
        (self.root / "RECOVERY_MANIFEST.json").write_text(json.dumps(manifest))
        with patch.object(R, "HERE", self.root), patch.object(R, "OLD", old), patch.object(R, "OUT", out), \
             patch.object(R.L, "REPO", self.root), patch.object(R, "verify_warmup", return_value={}):
            R.copy_warmup()
            for path in warm.iterdir():
                self.assertEqual(path.read_bytes(), (out / R.REUSED / path.name).read_bytes())
            receipt = R.L.read(out / R.REUSED / "REUSED_FROM.json")
            self.assertEqual(receipt["training_updates_in_copy"], 0)
            with self.assertRaises(FileExistsError):
                R.copy_warmup()
            (out / R.REUSED / "checkpoint.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "mirror"):
                R.verify_mirror()

    def test_recorded_pid_guard_is_read_only_and_fails_closed(self):
        kill = Mock(side_effect=ProcessLookupError)
        with patch.object(R, "os", SimpleNamespace(name="posix", kill=kill)):
            R.old_processes_absent()
        self.assertEqual([c.args for c in kill.call_args_list], [(435, 0), (1693, 0)])
        with patch.object(R, "os", SimpleNamespace(name="posix", kill=Mock(return_value=None))):
            with self.assertRaisesRegex(RuntimeError, "possibly reused"):
                R.old_processes_absent()
        with patch.object(R, "os", SimpleNamespace(name="posix", kill=Mock(side_effect=PermissionError))):
            with self.assertRaisesRegex(RuntimeError, "Cannot establish"):
                R.old_processes_absent()

    def test_process_receipt_failure_cleans_up_owned_child(self):
        child = Mock(pid=999)
        child.poll.return_value = None
        child.wait.return_value = -15
        killer = Mock()
        with patch.object(R.subprocess, "Popen", return_value=child) as spawn, \
             patch.object(R.L, "save", side_effect=OSError("receipt full")), \
             patch.object(R, "os", SimpleNamespace(killpg=killer)), patch.object(R, "signal", fake_signals()), \
             patch.object(R.time, "monotonic", return_value=0):
            with self.assertRaisesRegex(OSError, "receipt full"):
                R.run_child({"id": "fixture"}, self.root, 10, None)
        self.assertIn("recovery_launch.py", spawn.call_args.args[0][3])
        killer.assert_called_once_with(999, 15)
        child.wait.assert_called_once_with(timeout=5)

    def test_timeout_escalates_only_owned_child_and_deadline_prevents_spawn(self):
        child = Mock(pid=777)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("fixture", 1), subprocess.TimeoutExpired("fixture", 5), -9]
        killer = Mock()
        with patch.object(R.subprocess, "Popen", return_value=child), patch.object(R.L, "save"), \
             patch.object(R, "os", SimpleNamespace(killpg=killer)), patch.object(R, "signal", fake_signals()), \
             patch.object(R.time, "monotonic", return_value=0):
            with self.assertRaises(subprocess.TimeoutExpired):
                R.run_child({"id": "fixture"}, self.root, 1, None)
        self.assertEqual([call.args for call in killer.call_args_list], [(777, 15), (777, 9)])
        with patch.object(R.subprocess, "Popen") as spawn, patch.object(R.time, "monotonic", return_value=2):
            with self.assertRaisesRegex(ValueError, "before child dispatch"):
                R.run_child({"id": "fixture"}, self.root, 1, None)
            spawn.assert_not_called()

    def _execution_patches(self, *, copy_failure=False, check_failure=False):
        (self.root / "RECOVERY_MANIFEST.json").write_text("{}")
        fake_os = SimpleNamespace(name="posix", getpid=lambda: 1234)
        reports = lambda directory, rc: ({"status": "completed_frozen_evaluation", "physical_counts": {"completed": {"returns": 100}}}
            if directory.name == "evaluation" else {"status": "completed_training_job", "counts": {"completed": R.JOB_CAPS}})
        return [patch("builtins.print"), patch.object(R, "HERE", self.root), patch.object(R, "OUT", self.root / "run_recovery_1"),
            patch.object(R.L, "REPO", self.root), patch.object(R, "os", fake_os), patch.object(R, "signal", fake_signals()),
            patch.object(R, "check", side_effect=ValueError("proof or approval failed") if check_failure else None),
            patch.object(R, "old_processes_absent"), patch.object(R.S, "smoke_identity", return_value={"proof": "fixed"}),
            patch.object(R.S, "smoke_passed", return_value={"physical": {"counts": {"completed": {"all": 3320}}}}),
            patch.object(R, "copy_warmup", side_effect=OSError("mirror failed") if copy_failure else None),
            patch.object(R, "verify_mirror"), patch.object(R, "run_child", return_value=0),
            patch.object(R.L, "accepted_job", side_effect=reports)]

    def test_supervisor_refuses_bad_proof_before_reservation(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            handles = [stack.enter_context(p) for p in self._execution_patches(check_failure=True)]
            with self.assertRaisesRegex(ValueError, "proof or approval"):
                R.execute()
            self.assertFalse((self.root / "run_recovery_1").exists())
            handles[-2].assert_not_called()

    def test_mirror_failure_prevents_any_dispatch_and_preserves_attempt(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            handles = [stack.enter_context(p) for p in self._execution_patches(copy_failure=True)]
            result = R.execute()
            handles[-2].assert_not_called()
        self.assertEqual(result["status"], "incomplete_recovery")
        self.assertTrue((self.root / "run_recovery_1/RESULT.json").exists())
        self.assertEqual(result["new_training_upper_bounds"]["critic"], 0)

    def test_success_runs_only_remaining_jobs_keeps_total_unknown_and_blocks_duplicate(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            handles = [stack.enter_context(p) for p in self._execution_patches()]
            result = R.execute()
            dispatched = [call.args[0]["id"] for call in handles[-2].call_args_list]
            self.assertEqual(dispatched, [job["id"] for job in R.jobs()])
            self.assertNotIn(R.REUSED, dispatched)
            with self.assertRaises(FileExistsError):
                R.execute()
        self.assertEqual(result["status"], "completed_path_a_recovery")
        self.assertTrue(result["new_execution_counts_known"])
        self.assertFalse(result["cumulative_execution_counts_known"])
        self.assertEqual(result["cumulative_physical_upper"], R.CUMULATIVE_CAPS)


if __name__ == "__main__":
    unittest.main()
