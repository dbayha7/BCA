"""Pure fixtures; no model imports, subprocesses, WSL, plant calls or fitting."""
from contextlib import nullcontext
import builtins
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

SPEC = importlib.util.spec_from_file_location("retry_under_test", Path(__file__).with_name("titan_retry.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        here, study = self.repo / M.OWN, self.repo / M.PATH_A
        here.mkdir(parents=True)
        study.mkdir(parents=True)
        (here / "titan_retry.py").write_text("fixture only")
        M.save(study / "TITAN_APPROVAL.json", {"original": True})
        first = study / "titan_validation_1"
        first.mkdir()
        M.save(first / "ATTEMPT.json", {"pid": M.PRIOR_PID, "titan_manifest_sha256": M.TITAN_MANIFEST,
                                     "titan_approval_sha256": M.sha(study / "TITAN_APPROVAL.json")})
        (first / "progress.jsonl").write_text(json.dumps({
            "stage": "before_template_initialization", "counts": {"template_initializations_completed": 0}}) + "\n")
        self.identity = {"machine": "Titan", "repository": M.TARGET_REPO, "executable": M.TARGET_PYTHON}
        R = SimpleNamespace(OUT=study / "run_recovery_1", check=Mock(), old_processes_absent=Mock())
        T = SimpleNamespace(R=R, L=SimpleNamespace(), S=SimpleNamespace(),
                            check=Mock(return_value={"protocol": "frozen"}), target_identity=lambda: self.identity)
        self.retry = M.Retry(self.repo, engine=T)
        M.save(here / "RETRY_MANIFEST.json", {"schema": M.SCHEMA, "titan_manifest_sha256": M.TITAN_MANIFEST,
              "allocation": M.allocation(), "files": {(M.OWN / "titan_retry.py").as_posix(): M.sha(here / "titan_retry.py")}})
        M.save(here / "RETRY_APPROVAL.json", M.approval_expected(self.repo))
        M.save(here / "INSTALL_RECEIPT.json", {"schema": M.INSTALL_SCHEMA, "status": "installed_without_overwriting",
              "repository": M.TARGET_REPO, "manifest_sha256": M.sha(here / "RETRY_MANIFEST.json"),
              "installation_elapsed_seconds": 2, "timing_known": True, "path_a_a1_absent": True,
              "first_validation_files": self.retry.first_identity(), "first_validation_pid": M.PRIOR_PID,
              "no_scientific_work": True})
        self.retry.invoked = 100.
        self.addCleanup(patch.stopall)
        patch.object(M, "boot_id", return_value="same-boot").start()
        patch.object(M.time, "monotonic", return_value=100.).start()
        patch.object(M.os, "kill", side_effect=ProcessLookupError).start()
        self.retry.signals = lambda: nullcontext()

    def prepare(self, upper=0):
        return self.retry.prepare(upper, True, True)

    def write_worker_pass(self):
        worker = self.retry.validation / "worker"
        worker.mkdir()
        M.save(worker / "ATTEMPT.json", {"titan_manifest_sha256": M.TITAN_MANIFEST,
              "titan_approval_sha256": M.sha(self.retry.study / "TITAN_APPROVAL.json"), "target_identity": self.identity})
        M.save(worker / "report.json", {"status": "passed_zero_update_titan_compatibility", "passed": True,
                                      "new_work": dict(M.ZERO_WORK)})
        (worker / "progress.jsonl").write_text("{}\n")
        M.save(worker / "RESULT.json", {"status": "passed_titan_validation", "identity_unchanged": True,
              "elapsed_seconds": 1, "report_sha256": M.sha(worker / "report.json"), "target_identity": self.identity})

    def successful_child(self, command, directory, end, stdout, stderr):
        self.assertIn("--validation-worker", command)
        self.assertEqual(end, 205.)
        M.save(directory / "PROCESS.json", {"pid": 99})
        self.write_worker_pass()
        return dict(exit_code=0, signal=None, timed_out=False, interrupted=False, error=None)

    def test_allocation_keeps_cumulative_caps_and_charges(self):
        a = M.allocation()
        self.assertEqual(10660, 60 + 120 + 60 + 120 + a["maximum_production_seconds"])
        self.assertEqual(a["cumulative_optimizer_caps"], dict(critic=190014, actor=95008, auxiliary=190014, calibrations=22))
        self.assertEqual(a["new_validation_work"], dict(critic=0, actor=0, auxiliary=0, calibrations=0, simulator_steps=0, reset_calls=0))

    def test_prepare_requires_operator_confirmation_and_known_bound(self):
        for args in ((None, True, True), (-1, True, True), (0, False, True), (0, True, False)):
            with self.assertRaises(ValueError):
                self.retry.prepare(*args)
        self.assertFalse((self.retry.here / "RETRY_LEDGER.json").exists())

    def test_source_and_installer_hashes_fail_closed(self):
        self.assertEqual(self.retry.source_check(), {"protocol": "frozen"})
        (self.retry.here / "titan_retry.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Retry file differs"):
            self.retry.source_check()

    def test_conflicts_and_old_pid_do_not_dispatch_or_kill(self):
        helper = self.repo / "path_a_a1.py"
        helper.write_text("not executed")
        with self.assertRaisesRegex(ValueError, "path_a_a1"):
            self.retry.source_check()
        helper.unlink()  # Explicit disposable fixture only.
        with patch.object(M.os, "kill", return_value=None) as probe:
            with self.assertRaisesRegex(ValueError, "present"):
                self.retry.prior_inactive()
            probe.assert_called_once_with(34106, 0)
        (self.retry.study / "run_titan_recovery_1").mkdir()
        with self.assertRaisesRegex(ValueError, "production reservation"):
            self.retry.source_check()

    def test_first_attempt_changed_or_advanced_is_rejected(self):
        (self.retry.first / "progress.jsonl").write_text('{"stage":"after_template_initialization"}\n')
        with self.assertRaisesRegex(ValueError, "advanced"):
            self.retry.source_check()

    def test_unrecognized_titan_attempt_blocks_preparation(self):
        (self.retry.study / "titan_validation_3").mkdir()
        with self.assertRaisesRegex(ValueError, "Another Titan validation"):
            self.retry.source_check()

    def test_prepare_exclusive_and_same_boot_waiting_debits_budget(self):
        self.assertEqual(self.prepare(20)["production_cap_seconds"], 10300)
        with self.assertRaises(FileExistsError):
            self.prepare(20)
        ledger = M.read(self.retry.here / "RETRY_LEDGER.json")
        with patch.object(M.time, "monotonic", return_value=300.):
            result = M.remaining(ledger, validation_elapsed=80)
        self.assertEqual(result["preparation_charged_seconds"], 142)
        self.assertEqual(result["production_cap_seconds"], 10218)
        with patch.object(M, "boot_id", return_value="reboot"):
            with self.assertRaisesRegex(ValueError, "Boot changed"):
                M.remaining(ledger)
        with patch.object(M.time, "monotonic", return_value=99):
            with self.assertRaises(ValueError):
                M.remaining(ledger)

    def test_native_crash_leaves_terminal_failure_and_no_pass(self):
        self.prepare()
        outcome = dict(exit_code=-11, signal=11, timed_out=False, interrupted=False, error=None)
        self.retry.child = Mock(return_value=outcome)
        result = self.retry.validate()
        self.assertFalse(result["passed"])
        self.assertEqual(result["child"]["signal"], 11)
        self.assertEqual(M.read(self.retry.validation / "RESULT.json"), result)
        self.assertFalse(self.retry.out.exists())
        with self.assertRaises(ValueError):
            self.retry.proof()

    def test_zero_exit_without_complete_report_never_passes(self):
        self.prepare()
        self.retry.child = Mock(return_value=dict(exit_code=0, signal=None, timed_out=False, interrupted=False, error=None))
        result = self.retry.validate()
        self.assertFalse(result["passed"])
        self.assertIn("FileNotFoundError", result["error"])

    def test_real_helper_shaped_pass_and_duplicate_reservation(self):
        self.prepare()
        self.retry.child = self.successful_child
        result = self.retry.validate()
        self.assertTrue(result["passed"])
        self.assertEqual(set(self.retry.proof()), {"ATTEMPT.json", "RESULT.json", "stdout.log", "stderr.log", "PROCESS.json"})
        with self.assertRaises(FileExistsError):
            self.retry.validate()
        report = self.retry.validation / "worker/report.json"
        value = M.read(report)
        value["new_work"]["reset_calls"] = 1
        report.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.retry.proof()

    def test_process_receipt_write_failure_cleans_only_owned_child(self):
        process = Mock(pid=7)
        process.poll.return_value = None
        process.wait.return_value = -15
        directory = self.repo / "child"
        directory.mkdir()
        with patch.object(M.subprocess, "Popen", return_value=process), \
             patch.object(M, "save", side_effect=OSError("disk full")), \
             patch.object(M.os, "killpg", create=True) as kill:
            result = self.retry.child(["fixture"], directory, 105, io.StringIO(), io.StringIO())
        kill.assert_called_once_with(7, M.signal.SIGTERM)
        self.assertEqual(result["signal"], 15)
        self.assertIn("disk full", result["error"])

    def test_timeout_terminates_owned_child(self):
        process = Mock(pid=8)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 5), -15, -15]
        directory = self.repo / "child"
        directory.mkdir()
        with patch.object(M.subprocess, "Popen", return_value=process), \
             patch.object(M.os, "killpg", create=True) as kill:
            result = self.retry.child(["fixture"], directory, 105, io.StringIO(), io.StringIO())
        self.assertTrue(result["timed_out"])
        kill.assert_called_once_with(8, M.signal.SIGTERM)

    def test_process_local_recovery_bindings_restore_on_exception(self):
        original = self.retry.R.OUT, self.retry.R.check, self.retry.R.old_processes_absent
        with self.assertRaises(RuntimeError):
            with self.retry.recovery_bindings({"proof": "fixture"}):
                self.assertEqual(self.retry.R.OUT, self.retry.out)
                raise RuntimeError("fixture")
        self.assertEqual((self.retry.R.OUT, self.retry.R.check, self.retry.R.old_processes_absent), original)

    def test_production_blocked_before_reservation_without_validation_pass(self):
        self.prepare()
        with self.assertRaises(FileNotFoundError):
            self.retry.execute()
        self.assertFalse(self.retry.out.exists())

    def test_production_budget_enters_external_timeout_without_user_pause(self):
        self.prepare()
        self.retry.child = self.successful_child
        self.retry.validate()
        class ReplacedProcess(BaseException):
            pass
        with patch.object(M.shutil, "which", return_value="/usr/bin/timeout"), \
             patch.object(M.os, "execv", side_effect=ReplacedProcess) as execute:
            with self.assertRaises(ReplacedProcess):
                M.dispatch_production(self.repo)
        command = execute.call_args.args[1]
        self.assertEqual(command[1:4], ["--signal=TERM", "--kill-after=5s", "10295s"])
        self.assertEqual(command[-1], "--production-supervisor")
        self.assertFalse(self.retry.out.exists())

    def test_cli_arms_timeout_before_frozen_launcher_constructor(self):
        class ReplacedProcess(BaseException):
            pass
        with patch.object(M.sys, "argv", ["titan_retry.py", "--execute"]), \
             patch.object(M, "Retry", side_effect=AssertionError("Frozen imports before timeout")) as constructor, \
             patch.object(M, "dispatch_production", side_effect=ReplacedProcess) as dispatch:
            with self.assertRaises(ReplacedProcess):
                M.main()
        constructor.assert_not_called()
        dispatch.assert_called_once()


class ExtractedSourceStartupTests(unittest.TestCase):
    def test_actual_frozen_closure_and_prepare_without_models_or_processes(self):
        archive = Path("C:/Users/David/AppData/Local/Temp/BCA_PathA_Titan_20261102/path_a_titan_recovery.zip")
        if not archive.is_file():
            self.skipTest("Verified local transfer ZIP unavailable; fake tests do not replace this check")
        self.assertEqual(M.sha(archive), "c7f492c7e3d46a8508f8f7df30a975e216230e9e7e6289e2edb53f4793ef64fc")
        workspace = Path(__file__).resolve().parents[2]
        original_import = builtins.__import__
        def safe_import(name, *args, **kwargs):
            if name.split(".")[0] in {"jax", "jaxlib", "flax", "optax", "distrax", "chex",
                                     "tensorflow_probability", "torch", "gymnasium"}:
                raise AssertionError("Source-only check tried a model/runtime import: " + name)
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory(prefix=".retry_source_fixture_", dir=workspace) as temporary:
            repo = Path(temporary).resolve()
            self.assertTrue(repo.is_relative_to(workspace))
            with zipfile.ZipFile(archive) as bundle:
                transfer = json.loads(bundle.read("TRANSFER_MANIFEST.json"))
                members = transfer["files"]
                for relative, expected in members.items():
                    destination = (repo / relative).resolve()
                    self.assertTrue(destination.is_relative_to(repo))
                    raw = bundle.read(relative)
                    self.assertEqual(len(raw), expected["bytes"])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), expected["sha256"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as stream:
                        stream.write(raw)
            here, study = repo / M.OWN, repo / M.PATH_A
            here.mkdir(parents=True)
            (here / "titan_retry.py").write_bytes(Path(M.__file__).read_bytes())
            first = study / "titan_validation_1"
            first.mkdir()
            M.save(first / "ATTEMPT.json", {"pid": M.PRIOR_PID, "titan_manifest_sha256": M.TITAN_MANIFEST,
                                          "titan_approval_sha256": M.sha(study / "TITAN_APPROVAL.json")})
            (first / "progress.jsonl").write_text(json.dumps({"stage": "before_template_initialization",
                "counts": {"template_initializations_completed": 0}}) + "\n")
            M.save(here / "RETRY_MANIFEST.json", {"schema": M.SCHEMA, "titan_manifest_sha256": M.TITAN_MANIFEST,
                   "allocation": M.allocation(), "files": {(M.OWN / "titan_retry.py").as_posix(): M.sha(here / "titan_retry.py")}})
            M.save(here / "RETRY_APPROVAL.json", M.approval_expected(repo))
            M.save(here / "INSTALL_RECEIPT.json", {"schema": M.INSTALL_SCHEMA, "status": "installed_without_overwriting",
                "repository": M.TARGET_REPO, "manifest_sha256": M.sha(here / "RETRY_MANIFEST.json"),
                "installation_elapsed_seconds": 1, "timing_known": True, "path_a_a1_absent": True,
                "first_validation_files": {n: M.sha(first / n) for n in ("ATTEMPT.json", "progress.jsonl")},
                "first_validation_pid": M.PRIOR_PID, "no_scientific_work": True})
            with patch.dict(sys.modules), patch.object(sys, "path", list(sys.path)), \
                 patch.object(builtins, "__import__", side_effect=safe_import), \
                 patch.object(M.subprocess, "Popen", side_effect=AssertionError("No child execution in source test")), \
                 patch.object(M.os, "kill", side_effect=AssertionError("No real PID probe in source test")), \
                 patch.object(M, "boot_id", return_value="fixture-boot"):
                for name in ("launch", "smoke_launch", "recovery_launch", "_path_a_danger_diagnostic"):
                    sys.modules.pop(name, None)
                retry = M.Retry(repo)
                self.assertEqual(len(retry.R.jobs()), 18)
                self.assertEqual(sum(j["arm"] != "evaluation" for j in retry.R.jobs()), 17)
                self.assertNotIn("honeypot_s0_warmup", [j["id"] for j in retry.R.jobs()])
                self.assertEqual(M.sha(study / "run_amendment_1/honeypot_s0_warmup/full_state.msgpack"),
                                 "5422f85ac77d2a51341bb1154d7ba057a623854fd5ec8232b762bf0443bc218f")
                retry.target = lambda: {"target": "explicit fixture, no execution"}
                retry.prior_inactive = lambda: {"pid": M.PRIOR_PID, "absent": "fixture"}
                self.assertEqual(retry.prepare(0, True, True)["status"], "prepared_not_executed")
                self.assertEqual(retry.check(), retry.source_check())
                self.assertFalse(retry.validation.exists())
                self.assertFalse(retry.out.exists())


if __name__ == "__main__":
    unittest.main()
