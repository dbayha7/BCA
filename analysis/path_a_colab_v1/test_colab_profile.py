"""Pure orchestration fixtures: no JAX/model/plant/real child is called."""
from contextlib import ExitStack, nullcontext
import builtins
from datetime import timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

spec = importlib.util.spec_from_file_location("_tested_colab_profile", Path(__file__).with_name("colab_profile.py"))
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


def fake_engine():
    jobs = [{"id": "honeypot_s0_" + a, "arm": a} for a in ("native", "bca")]
    jobs += [{"id": "%s_s%d_%s" % (c, s, a), "arm": a}
             for c, s in (("honeypot", 1), ("honeypot", 2), ("bottleneck", 0), ("bottleneck", 1), ("bottleneck", 2))
             for a in ("warmup", "native", "bca")]
    jobs += [{"id": "evaluation", "arm": "evaluation"}]
    def accounting(accepted, current, **kwargs):
        return {"new_execution_counts_known": current is None and len(accepted) == 18,
                "cumulative_execution_counts_known": False,
                "cumulative_physical_upper": {"critic": 190014, "actor": 95008, "auxiliary": 190014, "calibrations": 22},
                "fixture_accepted": len(accepted), "fixture_incomplete": current,
                "fixture_prior_steps": kwargs["prior_simulator_steps"]}
    R = SimpleNamespace(OUT=Path("original_output"), check=Mock(return_value={}), old_processes_absent=Mock(),
        OLD=Path("old_run"), REUSED="honeypot_s0_warmup", jobs=lambda: jobs, copy_warmup=Mock(), verify_mirror=Mock(),
        worker_main=Mock(return_value={"status": "completed"}), accounting=Mock(side_effect=accounting))
    L = SimpleNamespace(EST=timezone(timedelta(hours=-5)), plain=lambda x: x,
        accepted_job=Mock(return_value={"status": "fixture_accepted"}), load=Mock())
    S = SimpleNamespace(_services=Mock(return_value=(object(), object(), object(), {})),
        smoke_identity=Mock(return_value={"smoke": "fixed"}),
        smoke_passed=Mock(return_value={"physical": {"counts": {"completed": {"smoke": 3320}}}}))
    return SimpleNamespace(R=R, L=L, S=S, check=Mock(return_value={"protocol": "frozen"}))


class ColabFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.engine = fake_engine(); self.p = C.Profile(self.repo, engine=self.engine)
        self.p.directory.mkdir(parents=True); self.p.here.mkdir(parents=True)
        (self.p.directory / "colab_profile.py").write_text("fixture profile source")
        (self.p.here / "RECOVERY_MANIFEST.json").write_text("{}")
        evidence = self.repo / "operator_capture.txt"; evidence.write_text("Explicit fixture source-status capture")
        self.status = {"schema": "path_a.colab_migration.v1", "selected_owner": "Colab",
            "titan_validation": "not_started", "titan_recovery": "not_started", "zeus_hold_preserved": True,
            "other_execution": "none", "additional_scientific_work": 0, "additional_runtime_seconds_known": True,
            "additional_runtime_seconds": 0, "attestation": C.MIGRATION_ATTESTATION, "operator": "Fixture David",
            "checked_at_est": "2026-11-02T09:00:00-05:00", "evidence": [{"path": str(evidence), "sha256": C.sha(evidence)}]}
        self.ledger = {"schema": "path_a.colab_setup.v1", "boot_id": "fixture_boot", "monotonic_started": 100.,
            "wall_started": 1000., "timing_complete": True, "unknown_runtime_seconds": 0,
            "includes_install_upload_extract_and_waits": True}
        self.status_path, self.ledger_path = self.repo / "status.json", self.repo / "ledger.json"
        C.save(self.status_path, self.status); C.save(self.ledger_path, self.ledger)
        self.clock = ExitStack(); self.addCleanup(self.clock.close)
        self.clock.enter_context(patch.object(C, "boot_id", return_value="fixture_boot"))
        self.mono = self.clock.enter_context(patch.object(C.time, "monotonic", return_value=110.))
        self.wall = self.clock.enter_context(patch.object(C.time, "time", return_value=1010.))

    def prepare_authorized(self):
        with patch.object(C, "Profile", return_value=self.p):
            C.prepare(self.repo, self.status_path, self.ledger_path, "/content/path_a_colab_venv_v1/bin/python")
            C.authorize(self.repo, "Fixture David", C.APPROVAL_STATEMENT)

    def test_unknown_started_or_missing_evidence_never_authorizes_migration(self):
        self.assertEqual(C.migration(self.status), self.status)
        for change in ({"titan_validation": "unknown"}, {"titan_recovery": "started"}, {"zeus_hold_preserved": False},
                       {"selected_owner": "Titan"}, {"additional_runtime_seconds_known": False}, {"evidence": []}):
            with self.subTest(change=change), self.assertRaises(ValueError): C.migration({**self.status, **change})
        Path(self.status["evidence"][0]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence differs"): C.migration(self.status)

    def test_budget_debits_setup_pauses_and_full_validation_without_reset(self):
        self.assertEqual(C.budget(self.ledger)["scientific_cap_seconds"], 10470)
        self.mono.return_value = 190.; self.wall.return_value = 1090.
        # 10 s setup + 80 s actual validation, charged as a full120 s reservation.
        self.assertEqual(C.budget(self.ledger, 80.)["scientific_cap_seconds"], 10470)
        self.wall.return_value = 1150.  # A further wall-clock pause is charged.
        self.assertEqual(C.budget(self.ledger, 80.)["scientific_cap_seconds"], 10410)
        with patch.object(C, "boot_id", return_value="different_boot"):
            with self.assertRaisesRegex(ValueError, "restarted"): C.budget(self.ledger)
        with self.assertRaises(ValueError): C.budget({**self.ledger, "unknown_runtime_seconds": None})
        self.mono.return_value = 20000.; self.wall.return_value = 20900.
        with self.assertRaisesRegex(ValueError, "exhausted"): C.budget(self.ledger)

    def test_fresh_approval_required_and_pins_tamper_duplicate_protection(self):
        with patch.object(C, "Profile", return_value=self.p):
            C.prepare(self.repo, self.status_path, self.ledger_path, "/content/venv/bin/python")
            with self.assertRaises(FileNotFoundError): self.p.check(approval=True)
            with self.assertRaisesRegex(ValueError, "Explicit Colab"):
                C.authorize(self.repo, "Fixture David", "old Titan approval")
            C.authorize(self.repo, "Fixture David", C.APPROVAL_STATEMENT)
            self.p.check(approval=True)
            with self.assertRaises(FileExistsError): C.authorize(self.repo, "Fixture David", C.APPROVAL_STATEMENT)
            with self.assertRaisesRegex(ValueError, "reseal/reset/retry"):
                C.prepare(self.repo, self.status_path, self.ledger_path, "/content/venv/bin/python")
        self.ledger_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"): self.p.check(approval=True)

    def test_local_prior_titan_attempt_blocks_even_operator_not_started(self):
        self.prepare_authorized()
        (self.p.here / "titan_validation_1").mkdir()
        with self.assertRaisesRegex(ValueError, "competing/prior"): self.p.check(approval=True)

    def validation_patches(self, passed=True):
        self.prepare_authorized()
        compatibility = SimpleNamespace(EXPECTED_EXECUTABLE="original_titan_python",
            check_compatibility=Mock(return_value={"status": "passed_zero_update_titan_compatibility",
                                                 "passed": passed, "new_work": C.ZERO_WORK}))
        self.engine.L.load.side_effect = lambda name, path: compatibility if name == "_colab_frozen_compatibility" else object()
        identity = {"machine_profile": "Colab", "actual_device": "fixture only"}
        return compatibility, [patch.object(self.p, "identity", return_value=identity),
                               patch.object(self.p, "alarm", side_effect=lambda seconds: nullcontext())]

    def test_zero_update_validation_uses_real_schema_and_restores_path_profile(self):
        compatibility, patches = self.validation_patches()
        old = compatibility.EXPECTED_EXECUTABLE
        with ExitStack() as stack:
            for p in patches: stack.enter_context(p)
            result = self.p.validate()
            self.assertEqual(result["status"], "passed_colab_validation")
            self.assertEqual(compatibility.EXPECTED_EXECUTABLE, old)
            self.assertEqual(result["new_work"], C.ZERO_WORK)
            self.assertEqual(set(self.p.proof()), {"ATTEMPT.json", "RESULT.json", "report.json", "progress.jsonl"})
            with self.assertRaises(FileExistsError): self.p.validate()
        self.engine.S._services.assert_called_once()

    def test_failed_validation_blocks_proof_and_never_dispatches(self):
        compatibility, patches = self.validation_patches(passed=False)
        with ExitStack() as stack:
            for p in patches: stack.enter_context(p)
            self.assertEqual(self.p.validate()["status"], "failed_colab_validation")
            with self.assertRaisesRegex(ValueError, "successful Colab"): self.p.proof()
        self.assertFalse(self.p.output.exists()); self.engine.R.worker_main.assert_not_called()

    def test_scoped_orchestration_bindings_restore_after_exception(self):
        original = self.engine.R.OUT, self.engine.R.check, self.engine.R.old_processes_absent
        with patch.object(self.p, "check", return_value=({}, {})), patch.object(self.p, "proof", return_value={}), \
             patch.object(self.p, "identity", return_value={}) as identity:
            with self.assertRaises(RuntimeError):
                with self.p.recovery_bindings({}):
                    self.engine.R.old_processes_absent(); self.engine.R.check()
                    raise RuntimeError("fixture failure")
            identity.assert_called_once()
        self.assertEqual((self.engine.R.OUT, self.engine.R.check, self.engine.R.old_processes_absent), original)

    def test_child_receipt_failure_reaps_only_owned_child_and_uses_colab_entrypoint(self):
        process = Mock(pid=987); process.poll.return_value = None; process.wait.return_value = -15
        with patch.object(C.subprocess, "Popen", return_value=process) as spawn, \
             patch.object(C, "save", side_effect=OSError("fixture disk full")), \
             patch.object(C, "os", SimpleNamespace(killpg=Mock())) as fake_os:
            with self.assertRaises(OSError): self.p.child({"id": "fixture_job"}, self.repo, 120., None)
            fake_os.killpg.assert_called_once_with(987, C.signal.SIGTERM)
        self.assertIn("colab_profile.py", spawn.call_args.args[0][3])
        process.wait.assert_called_once_with(timeout=5)

    def test_exact_remaining_plan_no_new_warmup_and_unknown_tail_preserved(self):
        compatibility, patches = self.validation_patches()
        with ExitStack() as stack:
            for p in patches: stack.enter_context(p)
            self.p.validate()
            self.mono.return_value = 120.; self.wall.return_value = 1020.  # User pause is charged.
            child = stack.enter_context(patch.object(self.p, "child", return_value=0))
            stack.enter_context(patch("builtins.print"))
            result = self.p.execute()
            self.assertEqual([c.args[0]["id"] for c in child.call_args_list], [j["id"] for j in self.engine.R.jobs()])
            self.assertFalse(any(c.args[0]["id"] == "honeypot_s0_warmup" for c in child.call_args_list))
            self.assertEqual(result["status"], "completed_colab_path_a_recovery")
            self.assertFalse(result["cumulative_execution_counts_known"])
            self.assertEqual(result["fixture_prior_steps"], 192975)
            self.assertEqual(result["science_cap_seconds_after_all_setup_and_waits"], 10460)
            with self.assertRaises(FileExistsError): self.p.execute()

    def test_incomplete_job_keeps_uncertain_accounting_and_does_not_retry(self):
        compatibility, patches = self.validation_patches()
        with ExitStack() as stack:
            for p in patches: stack.enter_context(p)
            self.p.validate()
            child = stack.enter_context(patch.object(self.p, "child", side_effect=TimeoutError("fixture timeout")))
            stack.enter_context(patch("builtins.print"))
            result = self.p.execute()
            self.assertEqual(result["status"], "incomplete_colab_recovery")
            self.assertFalse(result["new_execution_counts_known"])
            self.assertFalse(result["cumulative_execution_counts_known"])
            self.assertEqual(result["fixture_incomplete"]["id"], "honeypot_s0_native")
            child.assert_called_once()


class ExtractedSourceStartup(unittest.TestCase):
    def test_real_extracted_closure_prepare_authorize_and_source_check_only(self):
        archive = Path(os.environ.get("BCA_PATH_A_TEST_ARCHIVE",
            r"C:\Users\David\AppData\Local\Temp\BCA_PathA_Titan_20261102\path_a_titan_recovery.zip"))
        if not archive.is_file(): self.skipTest("Verified handoff ZIP not supplied; set BCA_PATH_A_TEST_ARCHIVE")
        self.assertEqual(C.sha(archive), "c7f492c7e3d46a8508f8f7df30a975e216230e9e7e6289e2edb53f4793ef64fc")
        workspace = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix=".colab_startup_", dir=workspace) as td:
            root = Path(td).resolve()
            self.assertTrue(root.is_relative_to(workspace))  # Cleanup is confined to this new fixture directory.
            with zipfile.ZipFile(archive) as z:
                manifest = json.loads(z.read("TRANSFER_MANIFEST.json"))
                self.assertEqual(len(manifest["files"]), 671)
                for relative, info in manifest["files"].items():
                    destination = (root / relative).resolve()
                    self.assertTrue(destination.is_relative_to(root))
                    raw = z.read(relative)
                    self.assertEqual(len(raw), info["bytes"])
                    self.assertEqual(C.hashlib.sha256(raw).hexdigest(), info["sha256"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as stream: stream.write(raw)
            profile_dir = root / C.PROFILE_DIR; profile_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / "colab_profile.py").write_bytes(Path(C.__file__).read_bytes())
            evidence = root / "fixture_status_capture.txt"; evidence.write_text("Explicit startup-fixture evidence; not real migration approval")
            status = {"schema": "path_a.colab_migration.v1", "selected_owner": "Colab", "titan_validation": "not_started",
                "titan_recovery": "not_started", "zeus_hold_preserved": True, "other_execution": "none",
                "additional_scientific_work": 0, "additional_runtime_seconds_known": True, "additional_runtime_seconds": 0,
                "operator": "STARTUP FIXTURE ONLY", "checked_at_est": "2026-11-02T09:00:00-05:00",
                "attestation": C.MIGRATION_ATTESTATION, "evidence": [{"path": str(evidence), "sha256": C.sha(evidence)}]}
            ledger = {"schema": "path_a.colab_setup.v1", "boot_id": "startup-fixture", "monotonic_started": 100.,
                "wall_started": 1000., "timing_complete": True, "unknown_runtime_seconds": 0,
                "includes_install_upload_extract_and_waits": True}
            status_path, ledger_path = root / "status_fixture.json", root / "ledger_fixture.json"
            C.save(status_path, status); C.save(ledger_path, ledger)
            original_import = builtins.__import__
            blocked = {"jax", "jaxlib", "flax", "optax", "distrax", "chex", "tensorflow_probability", "torch", "gymnasium"}
            def guarded_import(name, *args, **kwargs):
                if name.split(".")[0] in blocked:
                    raise AssertionError("Source-only startup attempted scientific import: " + name)
                return original_import(name, *args, **kwargs)
            with patch.dict(sys.modules), patch.object(sys, "path", list(sys.path)), \
                 patch.object(C, "boot_id", return_value="startup-fixture"), \
                 patch.object(C.time, "monotonic", return_value=110.), patch.object(C.time, "time", return_value=1010.), \
                 patch("builtins.__import__", side_effect=guarded_import), \
                 patch.object(C.subprocess, "Popen", side_effect=AssertionError("No child execution in source fixture")), \
                 patch.object(C.os, "kill", side_effect=AssertionError("No PID inspection in source fixture")):
                for name in ("launch", "smoke_launch", "recovery_launch", "_path_a_danger_diagnostic"):
                    sys.modules.pop(name, None)
                prepared = C.prepare(root, status_path, ledger_path, "/content/path_a_colab_venv_v1/bin/python")
                self.assertEqual(prepared["status"], "prepared_not_authorized_or_executed")
                C.authorize(root, "STARTUP FIXTURE ONLY", C.APPROVAL_STATEMENT)
                profile = C.Profile(root)
                protocol, saved = profile.check(approval=True)
                self.assertEqual(len(profile.R.jobs()), 18)
                self.assertEqual(sum(j["arm"] != "evaluation" for j in profile.R.jobs()), 17)
                self.assertNotIn("honeypot_s0_warmup", [j["id"] for j in profile.R.jobs()])
                self.assertEqual(protocol["kappa"], 3.)
                self.assertEqual(protocol["bc_coefficient"], 1.)
                self.assertEqual(profile.S.smoke_passed()["status"], "passed_runtime_smoke")
                self.assertEqual(profile.R.verify_warmup(profile.R.OLD / profile.R.REUSED)["final_cumulative_counts"]["critic"], 10000)
                self.assertFalse(profile.validation.exists()); self.assertFalse(profile.output.exists())
                self.assertEqual(saved["titan_manifest_sha256"], C.TITAN_MANIFEST_SHA256)


if __name__ == "__main__": unittest.main()
