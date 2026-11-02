"""Standard-library fixtures only: never import JAX, a model, or a simulator."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("minimal_worker_fixture", HERE / "worker.py")
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)


class WorkerTests(unittest.TestCase):
    def test_real_import_and_help_need_only_standard_library(self):
        code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('w',sys.argv[1]);"
                " m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                " assert not ({'numpy','jax','gymnasium','stage1_worker','screen_worker'} & set(sys.modules))")
        subprocess.run([sys.executable, "-I", "-B", "-c", code, str(HERE / "worker.py")],
                       check=True, capture_output=True, text=True)
        result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "worker.py"), "--help"],
                                check=True, capture_output=True, text=True)
        self.assertIn("--job", result.stdout)

    def test_bad_config_fails_before_scientific_imports(self):
        before = set(sys.modules)
        with self.assertRaisesRegex(ValueError, "Unknown job kind"):
            W.configure({"kind": "not_a_job", "config": {}})
        self.assertEqual(set(sys.modules), before)

    def test_actual_dose_helper_captured_default_overridden(self):
        path = W.VALIDATION / "bca_dose_screen_v1/screen_worker.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "patch_runtime")
        namespace = {"KAPPA_A": 63.0}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        helper = SimpleNamespace(patch_runtime=namespace["patch_runtime"], KAPPA_A=63.0)
        runtime = SimpleNamespace(make_adapter=lambda context, state, arm, k, *a, **kw: (arm, k))
        W.force_kappa_three(helper)
        helper.patch_runtime(runtime)
        self.assertEqual(helper.KAPPA_A, 3.0)
        self.assertEqual(runtime.make_adapter(None, None, "bca", 63.0), ("bca", 3.0))
        self.assertEqual(runtime.make_adapter(None, None, "native", 0.0), ("native", 0.0))
        self.assertEqual(runtime.make_adapter(None, None, "distance", 0.5), ("distance", 0.5))
        helper.patch_runtime(runtime)
        self.assertEqual(runtime.make_adapter(None, None, "bca", 63.0), ("bca", 3.0))

    def test_integrity_detects_shadow_file_and_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            extra = root / "shadow.py"
            record = {"package_dir": str(W.HERE), "files": {str(HERE / "worker.py"): W.sha(HERE / "worker.py")}}
            with mock.patch.object(W, "source_roots", return_value=(root,)):
                self.assertEqual(W.verify_integrity(record)["changed"], [])
                extra.write_text("# fixture", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unsealed importable"):
                    W.verify_integrity(record)
                record["files"][str(extra)] = W.sha(extra)
                self.assertEqual(W.verify_integrity(record)["changed"], [])
                extra.unlink()
                with self.assertRaisesRegex(ValueError, "Source integrity failure"):
                    W.verify_integrity(record)

    @staticmethod
    def job(path):
        value = {"id": "fixture", "kind": "preflight", "config": {}, "integrity": {},
                 "transitions": 0, "timeout_seconds": 10}
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_postcheck_exception_cannot_leave_complete_report(self):
        class Counter:
            def __init__(self, limit):
                self.steps = self.attempted = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        fake = SimpleNamespace(Deadline=lambda seconds: lambda: None,
                               preflight=lambda *args: {"kind": "preflight"}, warmup=None, branch=None)
        modules = (fake, None, None, None, None, None, None, SimpleNamespace(SimulatorCounter=Counter))
        with tempfile.TemporaryDirectory() as td:
            job = Path(td) / "job.json"
            self.job(job)
            with mock.patch.object(W, "configure", return_value=modules), mock.patch.object(
                    W, "verify_integrity", side_effect=[{"verified_files": 1}, OSError("fixture read failure")]):
                report = W.run_job(job)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["unit_label"], "INCOMPLETE")
            self.assertIn("fixture read failure", report["postcheck_error"])
            self.assertEqual(report["simulator_steps_exact"], 0)
            self.assertTrue(report["execution_counts_known"])
            self.assertEqual(json.loads((Path(td) / "report.json").read_text())["status"], "failed")

    def test_startup_failure_keeps_unknown_counts_null(self):
        with tempfile.TemporaryDirectory() as td:
            job = Path(td) / "job.json"
            self.job(job)
            with mock.patch.object(W, "verify_integrity", return_value={"verified_files": 1}), mock.patch.object(
                    W, "configure", side_effect=RuntimeError("fixture startup failure")):
                report = W.run_job(job)
            self.assertFalse(report["execution_counts_known"])
            self.assertIsNone(report["transitions"])
            self.assertIsNone(report["simulator_steps_exact"])
            self.assertIsNone(report["simulator_steps_attempted"])

    def test_config_and_own_policy_bindings_are_explicit(self):
        tree = ast.parse((HERE / "worker.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        qcalls = [node for node in calls if isinstance(node.func, ast.Attribute)
                  and node.func.attr == "raw_values_from_states"]
        self.assertEqual(len(qcalls), 1)
        self.assertTrue(any(k.arg == "clip_all" and isinstance(k.value, ast.Constant)
                            and k.value.value is False for k in qcalls[0].keywords))
        source = ast.unparse(tree)
        for key in ("dataset_archive_sha256", "dataset_content_sha256", "dataset_entropy", "recorder_sha256",
                    "data_seeds", "training", "return_ids"):
            self.assertIn(repr(key), source)
        self.assertIn("masks['cutoff']", source)
        dataset_source = (W.VALIDATION / "knee_edge_ood_v3/dataset_knee.py").read_text(encoding="utf-8")
        self.assertIn('"recorder_sha256": sha_file(recorder.__file__)', dataset_source)


if __name__ == "__main__":
    unittest.main()
