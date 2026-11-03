"""Pure Titan orchestration fixtures; no real child, GPU, fitting or plant call."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("path_a_titan_tested", Path(__file__).with_name("titan_launch.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def signals():
    return SimpleNamespace(SIGINT=2, SIGTERM=15, SIGHUP=1, SIGALRM=14, SIGKILL=9, ITIMER_REAL=0,
                           getsignal=lambda s: 0, signal=lambda s, value: None, setitimer=lambda *args: None)


class TitanFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("TITAN_MANIFEST.json", "TITAN_APPROVAL.json", "RECOVERY_MANIFEST.json"):
            (self.root / name).write_text("{}")
        self.identity = {"machine": "Titan", "repository": T.TARGET_REPO, "executable": T.TARGET_PYTHON}

    def test_budget_is_split_not_reset_and_science_plan_unchanged(self):
        a = T.allocation()
        self.assertEqual(a["preparation_and_unmeasured_source_check_allowance_seconds"] +
                         a["validation_cap_seconds"] + a["production_cap_seconds"], 10660)
        self.assertFalse(a["preparation_allowance_is_measured_runtime"])
        self.assertEqual(a["validation_external_timeout_seconds"] + a["kill_grace_seconds"], 120)
        self.assertEqual(a["production_external_timeout_seconds"] + a["kill_grace_seconds"], 10480)
        self.assertEqual(a["production_internal_seconds"], 10465)
        self.assertTrue(all(value == 0 for value in a["validation_new_work"].values()))
        self.assertEqual(len(T.R.jobs()), 18)
        self.assertEqual(T.TARGET_REPO, "/mnt/c/Users/David Bayha/BCA-PathA")

    def test_source_hold_accepts_only_marker_not_real_attempt(self):
        proof = {"status": "held_for_Titan_migration_not_a_run", "execution_started": False}
        (self.root / "TITAN_SOURCE_HOLD.json").write_text(json.dumps(proof))
        hold = self.root / "source_hold"
        with patch.object(T, "HERE", self.root), patch.object(T, "SOURCE_HOLD_DIRECTORY", hold):
            self.assertEqual(T.source_hold(), proof)  # transferred profile, no source directory
            hold.mkdir()
            (hold / "HANDOFF_HOLD.json").write_text(json.dumps(proof))
            self.assertEqual(T.source_hold(), proof)
            (hold / "ATTEMPT.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "real/changed Zeus"):
                T.source_hold()

    def test_target_identity_never_queries_zeus_pids(self):
        fake_os = SimpleNamespace(name="posix", kill=Mock(side_effect=AssertionError("Zeus PID lookup forbidden")))
        repo = SimpleNamespace(resolve=lambda: T.TARGET_REPO)
        with patch.object(T, "os", fake_os), patch.object(T.L, "REPO", repo), \
             patch.object(T.sys, "executable", T.TARGET_PYTHON), patch.object(T.sys, "version_info", (3, 12, 3)), \
             patch.object(T, "SOURCE_HOLD_DIRECTORY", self.root / "absent"):
            result = T.target_identity()
            self.assertEqual(result["machine"], "Titan")
            fake_os.kill.assert_not_called()
        with patch.object(T, "os", SimpleNamespace(name="nt")):
            with self.assertRaisesRegex(ValueError, "Titan checkout"):
                T.target_identity()

    def test_process_local_profile_restores_bindings_on_failure(self):
        old = (T.R.OUT, T.R.check, T.R.old_processes_absent)
        proof = {"fixed": "proof"}
        with patch.object(T, "check", return_value={}), patch.object(T, "validation_proof", return_value=proof), \
             patch.object(T, "target_identity", return_value=self.identity) as target:
            with self.assertRaisesRegex(RuntimeError, "fixture stop"):
                with T.recovery_profile(proof):
                    self.assertEqual(T.R.OUT, T.OUT)
                    T.R.check(approval=True)
                    T.R.old_processes_absent()
                    raise RuntimeError("fixture stop")
            target.assert_called_once()
        self.assertEqual((T.R.OUT, T.R.check, T.R.old_processes_absent), old)

    def _validation_patches(self, passed=True):
        report = {"status": "passed_zero_update_titan_compatibility" if passed else "failed_zero_update_titan_compatibility",
                  "passed": passed, "new_work": T.allocation()["validation_new_work"]}
        compatibility = SimpleNamespace(check_compatibility=Mock(return_value=report))
        loader = lambda name, path: compatibility if name == "_path_a_titan_compatibility" else object()
        return [patch("builtins.print"), patch.object(T, "HERE", self.root),
                patch.object(T, "VALIDATION", self.root / "validation"), patch.object(T, "OUT", self.root / "science"),
                patch.object(T, "check", return_value={}), patch.object(T, "target_identity", return_value=self.identity),
                patch.object(T, "os", SimpleNamespace(getpid=lambda: 1)), patch.object(T, "signal", signals()),
                patch.object(T.S, "_services", return_value=(object(), object(), object(), {})),
                patch.object(T.L, "load", side_effect=loader)], compatibility

    def test_zero_update_validation_then_proof_and_duplicate_refusal(self):
        patches, compatibility = self._validation_patches()
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = T.validate()
            self.assertEqual(result["status"], "passed_titan_validation")
            compatibility.check_compatibility.assert_called_once()
            proof = T.validation_proof()
            self.assertEqual(set(proof), {"ATTEMPT.json", "RESULT.json", "report.json", "progress.jsonl"})
            with self.assertRaises(FileExistsError):
                T.validate()
            report = self.root / "validation/report.json"
            report.write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing, failed or changed"):
                T.validation_proof()

    def test_failed_validation_never_authorizes_science(self):
        patches, _ = self._validation_patches(passed=False)
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = T.validate()
            self.assertEqual(result["status"], "failed_titan_validation")
            with self.assertRaisesRegex(ValueError, "missing, failed or changed"):
                T.validation_proof()
            self.assertFalse((self.root / "science").exists())

    def test_actual_helper_fixture_report_passes_launcher_and_proof(self):
        # Run the actual helper using its pure fake services, then pass its actual
        # returned receipt through the launcher gate. No JAX/model/plant is loaded.
        helper_spec = importlib.util.spec_from_file_location("_titan_helper_contract_fixture",
            Path(__file__).with_name("test_titan_compatibility.py"))
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        report, _ = helper.TitanCompatibilityFixtures().run_fixture(helper.fixture())
        self.assertEqual(report["new_work"], T.allocation()["validation_new_work"])
        self.assertEqual(report["new_work"]["reset_calls"], 0)
        patches, compatibility = self._validation_patches()
        compatibility.check_compatibility.return_value = report
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            self.assertEqual(T.validate()["status"], "passed_titan_validation")
            self.assertIn("report.json", T.validation_proof())

    def test_receipt_failure_cleanup_owns_only_titan_child(self):
        child = Mock(pid=987)
        child.poll.return_value = None
        child.wait.return_value = -15
        killer = Mock()
        with patch.object(T.subprocess, "Popen", return_value=child) as spawn, \
             patch.object(T.L, "save", side_effect=OSError("receipt full")), \
             patch.object(T, "os", SimpleNamespace(killpg=killer)), patch.object(T, "signal", signals()), \
             patch.object(T.time, "monotonic", return_value=0):
            with self.assertRaisesRegex(OSError, "receipt full"):
                T.run_child({"id": "fixture"}, self.root, 10, None)
        self.assertIn("titan_launch.py", spawn.call_args.args[0][3])
        killer.assert_called_once_with(987, 15)

    def _science_patches(self, failed_proof=False):
        reports = lambda directory, rc: ({"status": "completed_frozen_evaluation", "physical_counts": {"completed": {"returns": 10}}}
            if directory.name == "evaluation" else {"status": "completed_training_job", "counts": {"completed": T.R.JOB_CAPS}})
        proof = {"fixed": "validation"}
        return [patch("builtins.print"), patch.object(T, "HERE", self.root), patch.object(T, "OUT", self.root / "science"),
            patch.object(T.L, "REPO", self.root), patch.object(T, "check", return_value={}),
            patch.object(T, "target_identity", return_value=self.identity),
            patch.object(T, "validation_proof", side_effect=ValueError("validation incomplete") if failed_proof else None, return_value=proof),
            patch.object(T.S, "smoke_identity", return_value={"smoke": "fixed"}),
            patch.object(T.S, "smoke_passed", return_value={"physical": {"counts": {"completed": {"all": 3320}}}}),
            patch.object(T, "os", SimpleNamespace(getpid=lambda: 1)), patch.object(T, "signal", signals()),
            patch.object(T.R, "copy_warmup"), patch.object(T.R, "verify_mirror"),
            patch.object(T, "run_child", return_value=0), patch.object(T.L, "accepted_job", side_effect=reports)]

    def test_failed_proof_blocks_science_before_reservation(self):
        with ExitStack() as stack:
            handles = [stack.enter_context(p) for p in self._science_patches(failed_proof=True)]
            with self.assertRaisesRegex(ValueError, "validation incomplete"):
                T.execute()
            handles[-2].assert_not_called()
        self.assertFalse((self.root / "science").exists())

    def test_titan_science_uses_exact_remaining_plan_and_preserves_unknown_tail(self):
        with ExitStack() as stack:
            handles = [stack.enter_context(p) for p in self._science_patches()]
            result = T.execute()
            ids = [call.args[0]["id"] for call in handles[-2].call_args_list]
            self.assertEqual(ids, [job["id"] for job in T.R.jobs()])
            with self.assertRaises(FileExistsError):
                T.execute()
        self.assertEqual(result["status"], "completed_titan_path_a_recovery")
        self.assertTrue(result["new_execution_counts_known"])
        self.assertFalse(result["cumulative_execution_counts_known"])
        self.assertEqual(result["cumulative_physical_upper"], T.R.CUMULATIVE_CAPS)
        self.assertEqual(result["validation_charged_seconds"], 120)
        self.assertEqual(result["preparation_allowance_seconds"], 60)
        self.assertFalse(result["preparation_allowance_is_measured_runtime"])


if __name__ == "__main__":
    unittest.main()
