"""Synthetic reservation/readiness fixtures; never invoke WSL, models or an audit."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


SOURCE = Path(__file__).with_name("run_operator_once.py")
SPEC = importlib.util.spec_from_file_location("operator_once_under_test", SOURCE)
once = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(once)


class OnceFixtures(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix=".operator-fixture-", dir=SOURCE.parent))
        self.addCleanup(self.cleanup_fixture)
        self.audit = self.root / "audit"
        self.audit.mkdir()
        self.python = self.root / "new-venv/bin/python"
        self.model = self.root / "native/td3_bc.py"
        self.model.parent.mkdir()
        self.model.write_text("# synthetic source identity only\n", encoding="utf-8")
        self.package = {"verified_files": 1, "fixture": True}
        self.identity = {"protocol_sha256": "fixture-protocol", "checkpoint": "fixed"}
        self.ready = {
            "status": "ready_for_reserved_bounded_audit",
            "package": self.package,
            "identity": self.identity,
            "runtime": {
                "executable": str(self.python),
                "packages": dict(once.EXPECTED_PACKAGES),
                "backend": "gpu", "devices": ["synthetic-device"],
                "matmul_precision": "highest", "jax_disable_jit": False,
                "jax_enable_x64": False,
                "environment": {"XLA_FLAGS": "--xla_gpu_triton_gemm_any=True"},
            },
            "interpreter": {
                "executable": str(self.python), "prefix": str(self.python.parent.parent),
                "include_system_site_packages": False,
            },
            "model": {
                "module": "unifloral_pinned_td3bc_computational_slice",
                "file": str(self.model), "sha256": once.sha(self.model),
                "native_revision": "f2dc1278eae18ed3e22c92255119c54885414d17",
            },
        }
        self.request = {"fixture": True, "worker_timeout_seconds": 300,
                        "max_transition_steps": 1152}
        self.command = [str(self.python), "-B", "-u", "fixture-launch.py",
                        "--execute", "--allow-outside-upstream-constraints"]

    def cleanup_fixture(self):
        # Verify the exact synthetic target before recursive Windows cleanup.
        if (self.root.resolve().parent != SOURCE.parent.resolve()
                or not self.root.name.startswith(".operator-fixture-") or self.root.is_symlink()):
            raise RuntimeError("Refusing cleanup outside the fixture directory")
        shutil.rmtree(self.root)

    def result_report(self, status=None, counts=True):
        output = self.audit / "results_fixture"
        output.mkdir()
        (output / "report.json").write_text(json.dumps({
            "status": status or once.SUCCESS, "execution_counts_known": counts,
            "ledger": {}, "worker_timeout_seconds": 300,
        }), encoding="utf-8")

    def call(self, launch, process_check=lambda: []):
        with redirect_stdout(io.StringIO()):
            return once.execute_once(self.root, self.audit, self.request,
                                     self.command, process_check, launch)

    def assert_preserved(self, status):
        self.assertEqual(json.loads((self.root / "OPERATOR_ATTEMPT.json").read_text(
            encoding="utf-8")), self.request)
        saved = json.loads((self.root / "OPERATOR_ATTEMPT_RESULT.json").read_text(
            encoding="utf-8"))
        self.assertEqual(saved["status"], status)
        never = mock.Mock(side_effect=AssertionError("must not relaunch"))
        with self.assertRaises((RuntimeError, FileExistsError)):
            self.call(never)
        never.assert_not_called()

    def test_matching_readiness_is_accepted(self):
        once.validate_readiness(self.ready, self.package, self.identity,
                                self.python, self.model)

    def test_readiness_identity_runtime_and_model_mismatches_are_rejected(self):
        changes = [
            (("status",), "blocked_or_failed"),
            (("package",), {}), (("identity",), {}),
            (("runtime", "executable"), "another-python"),
            (("interpreter", "executable"), "another-python"),
            (("interpreter", "prefix"), "another-venv"),
            (("interpreter", "include_system_site_packages"), True),
            (("runtime", "packages", "jax"), "0.7.0"),
            (("runtime", "backend"), "cpu"), (("runtime", "devices"), []),
            (("runtime", "matmul_precision"), "default"),
            (("runtime", "jax_disable_jit"), True),
            (("runtime", "jax_enable_x64"), True),
            (("runtime", "environment", "XLA_FLAGS"), "other"),
            (("model", "module"), "corl"), (("model", "file"), "other.py"),
            (("model", "sha256"), "0" * 64),
            (("model", "native_revision"), "another-revision"),
        ]
        for keys, value in changes:
            with self.subTest(field=".".join(keys)):
                ready = deepcopy(self.ready)
                node = ready
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                with self.assertRaises(ValueError):
                    once.validate_readiness(ready, self.package, self.identity,
                                            self.python, self.model)

    def test_changed_model_bytes_reject_previous_readiness(self):
        self.model.write_text("# changed fixture\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            once.validate_readiness(self.ready, self.package, self.identity,
                                    self.python, self.model)

    def test_previous_result_directory_blocks_before_reservation(self):
        self.result_report()
        never = mock.Mock()
        with self.assertRaises(RuntimeError):
            self.call(never)
        never.assert_not_called()
        self.assertFalse((self.root / "OPERATOR_ATTEMPT.json").exists())

    def test_active_process_blocks_before_reservation(self):
        never = mock.Mock()
        with self.assertRaises(RuntimeError):
            self.call(never, lambda: [12345])
        never.assert_not_called()
        self.assertFalse((self.root / "OPERATOR_ATTEMPT.json").exists())

    def test_partial_reservation_is_not_parsed_overwritten_or_reused(self):
        reservation = self.root / "OPERATOR_ATTEMPT.json"
        reservation.write_text('{"interrupted":', encoding="utf-8")
        never = mock.Mock()
        with self.assertRaises(FileExistsError):
            self.call(never)
        never.assert_not_called()
        self.assertEqual(reservation.read_text(encoding="utf-8"), '{"interrupted":')

    def test_orphan_log_or_terminal_receipt_blocks_without_launch(self):
        for name in ("operator_once.log", "OPERATOR_ATTEMPT_RESULT.json"):
            with self.subTest(name=name):
                path = self.root / name
                path.write_text("preserved evidence", encoding="utf-8")
                never = mock.Mock()
                with self.assertRaises(RuntimeError):
                    self.call(never)
                never.assert_not_called()
                self.assertFalse((self.root / "OPERATOR_ATTEMPT.json").exists())
                path.unlink()

    def test_process_appearing_after_reservation_preserves_stop(self):
        check = mock.Mock(side_effect=[[], [12345]])
        never = mock.Mock()
        result = self.call(never, check)
        never.assert_not_called()
        self.assertFalse(result["launch_dispatched"])
        self.assert_preserved("failed_or_uncertain_preserved")

    def test_success_forwards_command_and_preserves_single_attempt(self):
        def launch(command, log):
            self.assertEqual(command, self.command)
            self.assertFalse(log.closed)
            self.result_report()
            return 0
        result = self.call(launch)
        self.assertEqual(result["status"], once.SUCCESS)
        self.assertTrue(result["execution_counts_known"])
        self.assertEqual(result["worker_timeout_seconds"], 300)
        self.assert_preserved(once.SUCCESS)

    def test_nonzero_exit_does_not_accept_success_report(self):
        def launch(command, log):
            self.result_report()
            return 1
        result = self.call(launch)
        self.assertEqual(result["status"], "audit_failed_preserved")
        self.assert_preserved("audit_failed_preserved")

    def test_zero_exit_does_not_accept_failed_report(self):
        def launch(command, log):
            self.result_report("timed_out", counts=False)
            return 0
        result = self.call(launch)
        self.assertFalse(result["execution_counts_known"])
        self.assert_preserved("audit_failed_preserved")

    def test_missing_report_preserves_unknown_outcome(self):
        result = self.call(lambda command, log: 0)
        self.assertFalse(result["execution_counts_known"])
        self.assert_preserved("failed_or_uncertain_preserved")

    def test_keyboard_interrupt_preserves_reservation_and_terminal_receipt(self):
        def launch(command, log):
            raise KeyboardInterrupt("fixture only")
        result = self.call(launch)
        self.assertTrue(result["launch_dispatched"])
        self.assertFalse(result["execution_counts_known"])
        self.assert_preserved("interrupted_outcome_requires_inspection")

    def test_atomic_reservation_allows_only_one_concurrent_launcher(self):
        barrier = threading.Barrier(2, timeout=5)
        local = threading.local()
        launches = []
        def check():
            local.calls = getattr(local, "calls", 0) + 1
            if local.calls == 1:
                barrier.wait()
            return []
        def launch(command, log):
            launches.append(list(command))
            self.result_report()
            return 0
        def contender():
            try:
                return once.execute_once(self.root, self.audit, self.request,
                                         self.command, check, launch)
            except FileExistsError:
                return "reservation_lost"
        with redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: contender(), range(2)))
        self.assertEqual(len(launches), 1)
        self.assertEqual(results.count("reservation_lost"), 1)
        self.assertEqual([x["status"] for x in results if isinstance(x, dict)], [once.SUCCESS])
        self.assert_preserved(once.SUCCESS)

    def test_main_checks_source_before_reservation_and_uses_frozen_launcher_args(self):
        handoff = self.root / "handoff"
        handoff.mkdir()
        manifest = handoff / "HANDOFF_MANIFEST.json"
        manifest.write_text(json.dumps({"files": [
            {"path": "restore_assets.py", "sha256": "fixture-restorer"},
            {"path": "payload/unifloral_operator_alignment_v1/run_checks.py", "sha256": "fixture-runner"},
        ]}), encoding="utf-8")
        model = handoff / "payload/unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py"
        model.parent.mkdir(parents=True)
        model.write_bytes(self.model.read_bytes())
        self.ready["model"]["file"] = str(model)
        ready_path = self.root / "ready.json"
        ready_path.write_text(json.dumps(self.ready), encoding="utf-8")
        source_checks = []
        def verify_package(path):
            self.assertFalse((self.root / "OPERATOR_ATTEMPT.json").exists())
            source_checks.append("package")
            return self.package
        def verify():
            self.assertFalse((self.root / "OPERATOR_ATTEMPT.json").exists())
            source_checks.append("frozen")
            return self.identity, {"worker_timeout_seconds": 300, "max_transition_steps": 1152}
        modules = {
            "_once_restorer": SimpleNamespace(verify_package=verify_package),
            "_once_frozen_runner": SimpleNamespace(verify=verify, execution_environment=lambda env: env),
            "_once_scoped_process_check": SimpleNamespace(audit_processes=lambda: []),
        }
        def fake_run(command, **kwargs):
            self.assertEqual(source_checks, ["package", "frozen"])
            self.assertEqual(command, [str(self.python), "-B", "-u", str(handoff / "launch.py"),
                                      "--execute", "--allow-outside-upstream-constraints"])
            self.result_report()
            return SimpleNamespace(returncode=0)
        with ExitStack() as stack:
            for name, value in {"ROOT": self.root, "HANDOFF": handoff, "AUDIT": self.audit,
                                "TARGET": self.python.parent.parent, "READY_NAME": ready_path.name,
                                "READY_SHA": once.sha(ready_path), "MANIFEST_SHA": once.sha(manifest)}.items():
                stack.enter_context(mock.patch.object(once, name, value))
            stack.enter_context(mock.patch.object(once, "load_module", side_effect=lambda name, *args: modules[name]))
            stack.enter_context(mock.patch.object(once.sys, "argv", [str(SOURCE), "--execute"]))
            stack.enter_context(mock.patch.object(once.sys, "platform", "linux"))
            stack.enter_context(mock.patch.object(once.sys, "executable", str(self.python)))
            stack.enter_context(mock.patch.object(once.sys, "prefix", str(self.python.parent.parent)))
            stack.enter_context(mock.patch.object(once.subprocess, "run", side_effect=fake_run))
            stack.enter_context(mock.patch.dict(once.os.environ))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(once.main(), 0)
        self.request = json.loads((self.root / "OPERATOR_ATTEMPT.json").read_text(encoding="utf-8"))
        self.assertEqual(self.request["worker_timeout_seconds"], 300)
        self.assertEqual(self.request["max_transition_steps"], 1152)
        self.assertEqual(self.request["readiness_sha256"], once.sha(ready_path))
        self.assert_preserved(once.SUCCESS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
