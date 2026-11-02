"""Stdlib-only fixtures; supervisors and workers are mocked, no scientific imports."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("launch.py")
spec = importlib.util.spec_from_file_location("minimal_launch_fixture", SOURCE)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


class AccountingTests(unittest.TestCase):
    def test_job_matrix_and_resource_sum(self):
        cfg = L.configuration()
        jobs = list(L.jobs(Path("fixture")))
        self.assertEqual(len(jobs), 12)
        self.assertEqual(sum(j["kind"] == "warmup" for j in jobs), 3)
        self.assertEqual(sum(j["kind"] in ("branch", "bca_branch") for j in jobs), 6)
        self.assertEqual(sum(j["kind"] == "evaluate" for j in jobs), 3)
        self.assertEqual(9 * 128000 + 3 * 184800, cfg["total_transition_cap"])
        self.assertTrue(all(set(row) == {"returns", "anchors"} for row in cfg["evaluation_seeds"]))

    def test_missing_report_preserves_tail(self):
        a = L.job_accounting("branch", None, 128000)
        self.assertFalse(a["execution_counts_known"])
        self.assertEqual(a["counts"]["simulator_steps"], {"lower": 0, "upper": 128000, "known": False})
        self.assertEqual(a["counts"]["critic_updates"]["upper"], 10000)
        self.assertEqual(a["counts"]["actor_updates"]["upper"], 5000)
        self.assertEqual(a["counts"]["auxiliary_updates"]["upper"], 10000)

    def test_unknown_counter_is_not_exact_zero(self):
        a = L.job_accounting("evaluate", {"execution_counts_known": False, "simulator_steps_exact": 0}, 184800)
        self.assertFalse(a["execution_counts_known"])
        self.assertEqual(a["counts"]["simulator_steps"]["upper"], 184800)

    def test_branch_cumulative_counts_charged_once(self):
        report = {"execution_counts_known": True, "simulator_steps_exact": 128000,
                  "counts": {"critic": 20000, "actor": 10000, "auxiliary": 20000, "refresh_accepted": 2}}
        a = L.job_accounting("bca_branch", report, 128000)
        self.assertTrue(a["execution_counts_known"])
        self.assertEqual(a["counts"]["critic_updates"]["lower"], 10000)
        self.assertEqual(a["counts"]["calibrations"]["lower"], 1)

    def test_invalid_counters_fail_closed(self):
        for steps in (True, -1, 128001, "128000"):
            a = L.job_accounting("branch", {"execution_counts_known": True, "simulator_steps_exact": steps}, 128000)
            self.assertFalse(a["execution_counts_known"])

    def test_aggregate_keeps_uncertain_tail(self):
        done = L.job_accounting("evaluate", {"execution_counts_known": True, "simulator_steps_exact": 184800}, 184800)
        tail = L.job_accounting("branch", None, 128000)
        total = L.aggregate_accounting({"done": {"accounting": done}, "tail": {"accounting": tail}})
        self.assertEqual(total["counts"]["simulator_steps"], {"lower": 184800, "upper": 312800, "known": False})


class ExecuteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.here = Path(self.tmp.name)
        (self.here / "MANIFEST.json").write_text("{}", encoding="utf-8")
        self.approval = self.here / "approval.json"
        self.approval.write_text(json.dumps({"approval": L.APPROVAL_TEXT,
                                            "manifest_sha256": L.sha(self.here / "MANIFEST.json")}), encoding="utf-8")
        self.cfg = L.configuration()
        self.calls = []
        for name, value in (("HERE", self.here), ("os", SimpleNamespace(name="posix", getpid=lambda: 123)),
                            ("overlap_check", lambda: None)):
            context = patch.object(L, name, value)
            context.start()
            self.addCleanup(context.stop)

    def run_mock(self, supervisor=None, checker=None, analysis=None):
        supervisor = supervisor or self.complete_worker
        owned = SimpleNamespace(child_environment=lambda: {}, supervise=supervisor)
        with patch.object(L, "check", checker or (lambda: (self.cfg, {"files": {}}))), \
             patch.object(L, "load_module", return_value=owned), \
             patch.object(L, "analyze", analysis or self.fake_analysis):
            return L.execute(self.approval)

    def complete_worker(self, command, directory, cap, **kwargs):
        self.calls.append((kwargs["job_id"], cap))
        job = L.read(directory / "job.json")
        report = {"job": job["id"], "kind": job["kind"], "job_sha256": L.sha(directory / "job.json"),
                  "status": "completed", "execution_counts_known": True, "simulator_steps_exact": job["transitions"]}
        if job["kind"] in ("warmup", "branch", "bca_branch"):
            scale = 1 if job["kind"] == "warmup" else 2
            report["counts"] = {"critic": 10000 * scale, "actor": 5000 * scale,
                                "auxiliary": 10000 * scale, "refresh_accepted": scale}
        L.save(directory / "report.json", report)
        return {"exit_code": 0, "timed_out": False, "interrupted": False}

    def fake_analysis(self, root, cfg):
        report = {"decision": "fixture_only"}
        L.save(root / "ANALYSIS.json", report)
        return report

    def test_success_and_caps(self):
        result = self.run_mock()
        self.assertEqual(result["status"], "completed_exploratory_comparison")
        self.assertEqual(len(self.calls), 13)
        self.assertEqual(result["accounting"]["counts"]["critic_updates"]["lower"], 90000)
        self.assertEqual(result["accounting"]["counts"]["actor_updates"]["lower"], 45000)
        self.assertEqual(result["accounting"]["counts"]["auxiliary_updates"]["lower"], 90000)
        self.assertEqual(result["accounting"]["counts"]["calibrations"]["lower"], 9)
        self.assertEqual(result["known_simulator_steps"], 1706400)
        self.assertTrue(result["execution_counts_known"])
        self.assertTrue(all(cap <= (120 if name == "preflight" else 600 if name.startswith("evaluate") else 1800)
                            for name, cap in self.calls))

    def test_approval_and_duplicate_reservation(self):
        self.approval.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError): self.run_mock()
        self.assertFalse((self.here / "run").exists())
        self.approval.write_text(json.dumps({"approval": L.APPROVAL_TEXT,
                                            "manifest_sha256": L.sha(self.here / "MANIFEST.json")}), encoding="utf-8")
        (self.here / "run").mkdir()
        with self.assertRaises(FileExistsError): self.run_mock()
        self.assertEqual(self.calls, [])

    def test_missing_worker_report_keeps_unknown_budget(self):
        def worker(command, directory, cap, **kwargs):
            if kwargs["job_id"] == "preflight": return self.complete_worker(command, directory, cap, **kwargs)
            return {"exit_code": -9, "timed_out": True, "interrupted": False}
        result = self.run_mock(worker)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["execution_counts_known"])
        self.assertEqual(result["accounting"]["counts"]["critic_updates"]["upper"], 10000)
        self.assertEqual(result["accounting"]["counts"]["simulator_steps"]["upper"], 128000)
        self.assertFalse((self.here / "run/ANALYSIS.json").exists())

    def test_supervisor_exception_is_recorded_as_unknown(self):
        def worker(command, directory, cap, **kwargs):
            if kwargs["job_id"] == "preflight": return self.complete_worker(command, directory, cap, **kwargs)
            raise KeyboardInterrupt("fixture interruption")
        result = self.run_mock(worker)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["execution_counts_known"])
        self.assertIn("KeyboardInterrupt", result["error"])
        self.assertTrue((self.here / "run/RESULT.json").is_file())

    def test_global_deadline_stops_before_dispatch(self):
        with patch.object(L.time, "monotonic", side_effect=[0, 10801, 10801]):
            result = self.run_mock()
        self.assertEqual(self.calls, [])
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("Global three-hour budget", result["error"])

    def test_integrity_exception_does_not_complete(self):
        count = 0
        def checker():
            nonlocal count
            count += 1
            if count == 15: raise OSError("fixture post-run integrity failure")
            return self.cfg, {"files": {}}
        result = self.run_mock(checker=checker)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("integrity failure", result["error"])
        self.assertFalse((self.here / "run/ANALYSIS.json").exists())

    def test_mismatched_report_rejected(self):
        def worker(command, directory, cap, **kwargs):
            result = self.complete_worker(command, directory, cap, **kwargs)
            report = L.read(directory / "report.json")
            report["job_sha256"] = "wrong"
            (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
            return result
        result = self.run_mock(worker)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("not bound", result["error"])


class AnalysisTests(unittest.TestCase):
    def test_complete_fixture_analysis_and_identity_pairing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(3):
                directory = root / f"evaluate_s{i}"
                directory.mkdir()
                runs = {}
                for name, value in (("native", 1.0), ("bca", 3.0)):
                    runs[name] = {"returns": [value] * 128, "return_ids": list(range(128)),
                                  "q1": [[1., 2., 3., 4.]] * 48, "qpi": [[1., 2., 3., 4.]] * 48,
                                  "width": [[0., 0., 0., 0.]] * 48}
                L.save(directory / "report.json", {"kind": "evaluate", "status": "completed", "seed_index": i, "runs": runs})
            result = L.analyze(root, L.configuration())
            self.assertEqual(result["decision"], "promising_exploratory_screen")
            self.assertEqual(result["across_seeds"]["mean_seed_difference"], 2.0)
            report_path = root / "evaluate_s2/report.json"
            report = L.read(report_path)
            report["runs"]["bca"]["return_ids"].reverse()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError): L.analyze(root, L.configuration())


if __name__ == "__main__":
    unittest.main()
