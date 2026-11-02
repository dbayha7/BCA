"""Saved-file and mocked-dispatch checks only; no model/environment imports."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("continuation_fixture", HERE / "launch.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class ContinuationTests(unittest.TestCase):
    def test_reused_reports_and_twenty_checkpoint_hashes(self):
        verified = C.verify_previous()["completed_jobs"]
        self.assertEqual(len(verified), 6)
        self.assertEqual(sum(len(r["checkpoints"]) for r in verified.values()), 20)

    def test_only_seed_two_restarts_and_all_evaluations_use_correct_paths(self):
        root = HERE / "fixture_run"
        jobs = list(C.jobs(root))
        self.assertEqual([j["id"] for j in jobs], ["warmup_s2", "native_s2", "bca_s2",
                                                   "evaluate_s0", "evaluate_s1", "evaluate_s2"])
        for j in jobs[:3]:
            self.assertEqual(j["seed_index"], 2)
        self.assertEqual(jobs[1]["warmup_dir"], jobs[2]["warmup_dir"])
        for i, j in enumerate(jobs[3:]):
            source = C.PREVIOUS / "run" if i < 2 else root
            self.assertEqual(j["warmup_dir"], str(source / f"warmup_s{i}"))
            self.assertEqual(j["native_branch_dir"], str(source / f"native_s{i}"))
            self.assertEqual(j["bca_branch_dir"], str(source / f"bca_s{i}"))

    def test_time_and_physical_work_not_reset(self):
        b = C.budget_record()
        self.assertEqual(b["charged_prior_seconds"] + b["process_cap_seconds"], 10800)
        self.assertAlmostEqual(b["previous_remaining_seconds"], 7571.494014139)
        self.assertAlmostEqual(b["additional_unmeasured_wrapper_allowance_seconds"], 71.494014139)
        self.assertEqual(b["new_work"]["critic_updates"], 30000)
        self.assertEqual(b["cumulative_physical_bounds"]["critic_updates"], [95000, 100000])
        self.assertEqual(b["proposed_cumulative_caps"]["calibrations"], 10)

    def test_cumulative_ledger_keeps_unknown_tail(self):
        new = {"counts": {k: {"lower": n, "upper": n, "known": True} for k, n in
                           {"critic_updates": 30000, "actor_updates": 15000, "auxiliary_updates": 30000,
                            "calibrations": 3, "simulator_steps": 170400}.items()}}
        total = C.cumulative_work_bounds(new)
        self.assertEqual(total["critic_updates"], {"lower": 95000, "upper": 100000, "known": False})
        self.assertEqual(total["actor_updates"], {"lower": 47500, "upper": 50000, "known": False})
        self.assertEqual(total["simulator_steps"], {"lower": 170400, "upper": 298400, "known": False})

    def test_adapter_uses_original_science_and_shorter_deadline(self):
        adapted = C.adapted_runner()
        self.assertEqual(adapted.HERE, HERE)
        self.assertEqual(adapted.RECOVERY_PROCESS_CAP, 7500)
        self.assertEqual(adapted.DISPATCH_DEADLINE_SECONDS, 7490)
        self.assertEqual(adapted.CHARGED_PRIOR_SECONDS, 3300)
        runner = adapted.adapted_runner(time.monotonic() + 1000)
        self.assertEqual(runner.configuration(), C.read(C.ORIGINAL / "config.json"))
        self.assertEqual(list(runner.jobs(HERE / "fixture")), list(C.jobs(HERE / "fixture")))

    def test_missing_approval_cannot_dispatch(self):
        with patch.object(C, "check", return_value=None), patch.object(C, "adapted_runner") as runner:
            with self.assertRaises(FileNotFoundError):
                C.execute(HERE / "ABSENT_APPROVAL.json")
            runner.assert_not_called()

    def test_old_or_incomplete_approval_rejected(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "approval.json"
            p.write_text(json.dumps({"approval": "old scope", "launch_authorized": True}))
            with self.assertRaises(ValueError):
                C.authorize(p)

    def worker_fixture(self, root):
        spec = importlib.util.spec_from_file_location("continuation_worker_fixture", HERE / "worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        worker.HERE = root
        pins = dict(C.read(C.PREVIOUS / "MANIFEST.json")["files"])
        pins[(C.PREVIOUS / "MANIFEST.json").relative_to(C.REPO).as_posix()] = C.PREVIOUS_MANIFEST_SHA
        manifest = root / "MANIFEST.json"
        manifest.write_text(json.dumps({"files": pins}))
        files = {str(C.REPO / p): h for p, h in pins.items()}
        files[str(manifest)] = C.sha(manifest)
        job = root / "job.json"
        job.write_text(json.dumps({"integrity": {"package_dir": str(C.ORIGINAL), "files": files}}))
        return worker, job

    def test_actual_worker_adapter_chain_constructs_without_scientific_dispatch(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            worker, job = self.worker_fixture(root)
            adapted = worker.adapted_worker(job)
            self.assertEqual(adapted.HERE, C.ORIGINAL)
            self.assertIn(C.PREVIOUS, adapted.source_roots())
            self.assertIn(root, adapted.source_roots())
            self.assertTrue(callable(adapted.configure))
            self.assertTrue(callable(adapted.run_job))
            # Neither configure nor run_job is invoked: only stdlib source imports.

    def test_worker_corrupt_pin_fails_before_importing_previous_worker(self):
        with TemporaryDirectory() as td:
            worker, job = self.worker_fixture(Path(td))
            record = json.loads(job.read_text())
            record["integrity"]["files"][str(C.PREVIOUS / "worker.py")] = "0" * 64
            job.write_text(json.dumps(record))
            with patch.object(worker.importlib.util, "spec_from_file_location") as imported:
                with self.assertRaisesRegex(ValueError, "changed before import"):
                    worker.adapted_worker(job)
                imported.assert_not_called()

    def test_mocked_runner_does_not_retrain_completed_seeds(self):
        previous = C.adapted_runner()
        runner = previous.adapted_runner(time.monotonic() + 1000)
        cfg = runner.configuration()
        calls = []
        with TemporaryDirectory() as td:
            root = Path(td)
            runner.HERE = root
            runner.check = lambda: (cfg, {"files": {}})
            runner.overlap_check = lambda: None
            runner.os = SimpleNamespace(name="posix", getpid=lambda: 123)
            (root / "MANIFEST.json").write_text("{}")
            approval = root / "fixture_approval.json"
            runner.save(approval, {"approval": C.APPROVAL_TEXT, "manifest_sha256": runner.sha(root / "MANIFEST.json")})
            def supervise(command, directory, cap, **kwargs):
                j = runner.read(directory / "job.json")
                calls.append(j)
                r = {"job": j["id"], "kind": j["kind"], "job_sha256": runner.sha(directory / "job.json"),
                     "status": "completed", "execution_counts_known": True,
                     "simulator_steps_exact": 56800 if j["kind"] == "evaluate" else 0}
                if j["kind"] in ("warmup", "branch", "bca_branch"):
                    scale = 1 if j["kind"] == "warmup" else 2
                    r["counts"] = {"critic": scale*10000, "actor": scale*5000,
                                   "auxiliary": scale*10000, "refresh_accepted": scale}
                runner.save(directory / "report.json", r)
                return {"exit_code": 0}
            owned = SimpleNamespace(child_environment=lambda: {}, supervise=supervise)
            runner.load_module = lambda name, path: owned
            def analyze(path, cfg):
                runner.save(path / "ANALYSIS.json", {"fixture": True})
                return {"decision": "fixture_only"}
            runner.analyze = analyze
            result = runner.execute(approval)
            self.assertEqual([j["id"] for j in calls], ["preflight"] + [j["id"] for j in C.jobs(root)])
            self.assertEqual(result["training_updates_completed"], 30000)
            self.assertEqual(result["known_simulator_steps"], 170400)
            with self.assertRaises(FileExistsError):
                runner.execute(approval)


if __name__ == "__main__":
    unittest.main()
