"""CPU fixtures only: no model, simulator, GPU or worker processes."""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("recovery_launch_tests", HERE / "launch.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
spec = importlib.util.spec_from_file_location("recovery_worker_tests", HERE / "worker.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


class RecoveryTests(unittest.TestCase):
    def test_budget_is_debited_not_restarted(self):
        b = R.budget_record()
        self.assertEqual(b["recovery_process_cap_seconds"] + b["charged_prior_and_wrapper_allowance_seconds"], 10800)
        self.assertAlmostEqual(b["known_prior_seconds"] + b["unmeasured_wrapper_allowance_seconds"], 100)
        self.assertEqual(b["expected_new_simulator_steps"], 76800 + 86400 + 7200)
        self.assertTrue(b["actual_unmeasured_overhead_remains_unknown"])

    def test_prior_raw_uncertainty_preserved(self):
        a = R.prior_assessment()
        self.assertFalse(a["original_raw_accounting_unchanged"]["execution_counts_known"])
        self.assertEqual(a["original_raw_accounting_unchanged"]["counts"]["critic_updates"]["upper"], 10000)
        self.assertEqual(a["source_inferred_prior_training_updates"], 0)

    def test_deadline_limits_child_and_refuses_late_dispatch(self):
        calls = []
        owned = SimpleNamespace(child_environment=lambda: {},
            supervise=lambda *args, **kwargs: calls.append((args, kwargs)) or {"exit_code": 0})
        wrapped = R.bounded_supervisor(owned, 100)
        with patch.object(R.time, "monotonic", return_value=20):
            result = wrapped.supervise(["fixture"], Path("unused"), 1800)
        self.assertEqual(result["recovery_effective_supervisor_cap_seconds"], 70)
        self.assertEqual(calls[0][1]["deadline"], 90)
        with patch.object(R.time, "monotonic", return_value=91), self.assertRaises(ValueError):
            wrapped.supervise(["fixture"], Path("unused"), 1800)
        self.assertEqual(len(calls), 1)

    def test_science_and_pairing_are_original(self):
        old = R.adapted_runner(time.monotonic() + 1000)
        self.assertEqual(old.configuration(), R.read(R.ORIGINAL / "config.json"))
        self.assertEqual(old.HERE, HERE)
        specs = list(old.jobs(HERE / "run"))
        self.assertEqual(len(specs), 12)
        for i in range(3):
            warm, native, bca = specs[3*i:3*i+3]
            self.assertEqual([j["seed_index"] for j in (warm, native, bca)], [i]*3)
            self.assertEqual(native["warmup_dir"], bca["warmup_dir"])
            self.assertEqual(bca["native_branch_dir"], str(HERE / "run" / native["id"]))
        metrics = old.load_module("_minimal_saved_metrics", HERE / "metrics.py")
        self.assertEqual(Path(metrics.__file__).resolve(), R.ORIGINAL / "metrics.py")

    def test_worker_changes_only_consumer_after_original_configuration(self):
        fakeD = SimpleNamespace(load=lambda p: self.fail("Old replay loader called"))
        fakeold = SimpleNamespace(configure=lambda job: (None, None, None, fakeD, None, None, None, None),
            source_roots=lambda: (R.ORIGINAL,), write_new=lambda p, v: v,
            sha=lambda p: "fixture")
        loader = SimpleNamespace(load_verified=lambda path, dataset: ("data", {"simulator_steps": 0}))
        with patch.object(W, "verify_before_import") as verify, \
             patch.object(W, "load_source", side_effect=[fakeold, loader]) as imports:
            worker = W.adapted_worker("fixture_job")
            verify.assert_called_once_with("fixture_job")
            worker.configure({})
            self.assertEqual(fakeD.load("fixture"), "data")
            self.assertEqual(imports.call_args_list[1].args[1], HERE / "immutable_data.py")
        record = worker.write_new("fixture", {})
        self.assertEqual(len(record["immutable_loader_receipts"]), 1)
        self.assertIn(HERE, worker.source_roots())

    def test_failed_integrity_prevents_any_worker_import(self):
        with patch.object(W, "verify_before_import", side_effect=ValueError("fixture corruption")), \
             patch.object(W, "load_source") as imports, self.assertRaises(ValueError):
            W.adapted_worker("fixture_job")
        imports.assert_not_called()

    def test_original_runner_dispatch_and_accounting_with_mock_workers(self):
        old = R.adapted_runner(time.monotonic() + 1000)
        cfg = old.configuration()
        calls = []
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old.HERE = root
            old.check = lambda: (cfg, {"files": {}})
            old.overlap_check = lambda: None
            old.os = SimpleNamespace(name="posix", getpid=lambda: 123)
            (root / "MANIFEST.json").write_text("{}")
            approval = root / "approval.json"
            old.save(approval, {"approval": R.APPROVAL_TEXT, "manifest_sha256": old.sha(root / "MANIFEST.json")})
            def fake_supervise(command, directory, cap, **kwargs):
                job = old.read(directory / "job.json")
                calls.append(job)
                self.assertEqual(Path(command[3]), root / "worker.py")
                self.assertEqual(job["config"], cfg)
                report = {"job": job["id"], "kind": job["kind"], "job_sha256": old.sha(directory / "job.json"),
                          "status": "completed", "execution_counts_known": True,
                          "simulator_steps_exact": 56800 if job["kind"] == "evaluate" else 0}
                if job["kind"] in ("warmup", "branch", "bca_branch"):
                    scale = 1 if job["kind"] == "warmup" else 2
                    report["counts"] = {"critic": scale*10000, "actor": scale*5000,
                                         "auxiliary": scale*10000, "refresh_accepted": scale}
                old.save(directory / "report.json", report)
                return {"exit_code": 0}
            owned = SimpleNamespace(child_environment=lambda: {}, supervise=fake_supervise)
            old.load_module = lambda name, path: R.bounded_supervisor(owned, time.monotonic() + 1000)
            def analysis(path, cfg):
                old.save(path / "ANALYSIS.json", {"fixture": True})
                return {"decision": "fixture_only"}
            old.analyze = analysis
            result = old.execute(approval)
            self.assertEqual(result["status"], "completed_exploratory_comparison")
            self.assertEqual(len(calls), 13)
            self.assertEqual(result["accounting"]["counts"]["critic_updates"]["lower"], 90000)
            self.assertEqual(result["known_simulator_steps"], 170400)
            with self.assertRaises(FileExistsError):
                old.execute(approval)


if __name__ == "__main__":
    unittest.main()
