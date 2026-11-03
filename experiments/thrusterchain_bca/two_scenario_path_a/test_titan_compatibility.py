"""Fake runtime/plant fixtures only: no JAX, model or simulator execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_titan_compatibility_fixture", HERE / "titan_compatibility.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def digest(value):
    return hashlib.sha256(json.dumps(C._plain(value), sort_keys=True).encode()).hexdigest()


def fixture():
    calls = []
    protocol = {"geometry_recipe": {"fixture": True}}
    normalization = {"mean": [0.] * 14, "std": [1.] * 14, "epsilon": .001}
    metadata = {condition: {"condition": condition, "rows": 256} for condition in ("honeypot", "bottleneck")}
    data_receipt = {"conditions": {c: {"metadata": m} for c, m in metadata.items()}}
    prepared = {c: SimpleNamespace(metadata=m, train=SimpleNamespace(obs=np.zeros((256, 14), np.float32),
        action=np.zeros((256, 4), np.float32)), source_row_ids=np.arange(256, dtype=np.int64)) for c, m in metadata.items()}
    saved_counts = {"critic": 10000, "actor": 5000, "auxiliary": 10000, "refresh_attempts": 1, "refresh_accepted": 1}
    state, template = {"digest": "saved", "counts": saved_counts}, {"digest": "initial", "counts": dict.fromkeys(saved_counts, 0)}
    runtime_receipt = {"executable": C.EXPECTED_EXECUTABLE, "python": "3.12.3 fixture", "backend": "gpu",
                       "devices": ["fixture gpu0"], "device_kinds": ["fixture GPU"], "packages": {"fixture": True}}
    def prepare(native, condition, protocol):
        calls.append(("data", condition)); return prepared[condition]
    binding = SimpleNamespace(prepare=prepare, I=SimpleNamespace(digest=digest))
    runtime = SimpleNamespace(runtime_receipt=lambda native: runtime_receipt,
        host=lambda value: value, validate=lambda adapter, value: {"counts": dict(value["counts"])},
        tree_digest=lambda value: value["digest"])
    services = SimpleNamespace(binding=binding, runtime=runtime, native=object(), expected_data_receipt=data_receipt)
    adapter = SimpleNamespace(context=SimpleNamespace(normalization=normalization))
    def initialize(p, condition, seed, services, review):
        calls.append(("initialize", condition, seed))
        if (condition, seed) != ("honeypot", 0): raise AssertionError("Wrong initialization")
        return prepared[condition], adapter, template
    def restore(directory, s, prep, adapt, templ, p, condition, seed, arm):
        calls.append(("restore", condition, seed, arm))
        if (condition, seed, arm) != ("honeypot", 0, "warmup"): raise AssertionError("Wrong restore")
        return state, {"fixture_exact_restore": True}
    training = SimpleNamespace(_prepare=initialize, _read_checkpoint=restore)
    inference = {"actor": lambda obs: np.zeros((len(obs), 4), np.float32),
                 "q1": lambda obs, act: np.ones(len(obs), np.float32),
                 "width": lambda obs, act: np.full(len(obs), .5, np.float32)}
    def callbacks(r, a, value, identity):
        before = value["digest"]
        def wrap(name):
            def invoke(*args):
                calls.append((name, len(args[0])))
                return inference[name](*args)
            return invoke
        def unchanged():
            if value["digest"] != before: raise ValueError("Changed fixture state")
        return {**{k: wrap(k) for k in inference}, "assert_unchanged": unchanged}
    worker = SimpleNamespace(callbacks=callbacks)
    action = np.array([-.2, .4, -.6, .8], np.float32)
    controller_cases = []
    for bank in range(6):
        for index in range(16):
            x = np.zeros(13, np.float64); x[0] = bank * 16 + index
            controller_cases.append({"id": "%d/common/%d" % (bank, index), "kind": "common", "state": x,
                "time": 29 if bank % 2 == 0 else 25, "expected_actions": np.stack((action, -action, np.zeros_like(action))),
                "source_sha256": "a" * 64})
        controller_cases.append({"id": "%d/prefix" % bank, "kind": "prefix", "state": np.zeros(13, np.float64),
            "time": 0, "expected_actions": action[None, :], "source_sha256": "b" * 64})
    source = {"data_receipt": data_receipt, "checkpoint": {"normalization": normalization,
        "normalization_sha256": digest(normalization), "counts": saved_counts, "state_digest": "saved",
        "checkpoint_sha256": "c" * 64}, "controller_cases": controller_cases, "source_result": {}, "pins": {"fixture": "d" * 64}}
    def make_controller(p, b, source_result):
        calls.append(("controller_init",))
        def controller(x, time):
            calls.append(("controller", x.copy(), time)); return action.copy()
        return controller, {"matrices_calls": 1, "tracking_controller_initializations": 1, "dynamics_calls": 0}
    return SimpleNamespace(**locals())


def frozen_environment():
    return {"JAX_PLATFORMS": "cuda", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_ENABLE_X64": "false",
            "XLA_FLAGS": "--xla_gpu_triton_gemm_any=True", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "BCA_TRANSFER_BUNDLE_ROOT": str(C.CORE.parent)}


class TitanCompatibilityFixtures(unittest.TestCase):
    def run_fixture(self, f, *, deadline=lambda: None, persist=None, controller_factory=None):
        events = []
        def record(event):
            json.dumps(event, allow_nan=False)
            if persist: persist(event)
            events.append(event)
        with patch.object(C, "_source_inputs", return_value=f.source), \
                patch.object(C, "_make_controller", side_effect=controller_factory or f.make_controller), \
                patch.dict(C.os.environ, frozen_environment(), clear=True):
            result = C.check_compatibility(f.protocol, f.binding, f.training, f.services, {}, f.worker,
                deadline, record, warmup_dir=Path("fixture_warmup"))
        return result, events

    def test_exact_restore_one_template_fixed_inference_and_controller_budget(self):
        f = fixture()
        result, events = self.run_fixture(f)
        self.assertEqual(result["status"], "passed_zero_update_titan_compatibility")
        self.assertTrue(result["passed"])
        self.assertEqual(result["new_work"], C.ZERO_WORK)
        counts = result["counts"]
        self.assertEqual(counts["template_initializations_completed"], 1)
        self.assertEqual(counts["checkpoint_restores_completed"], 1)
        self.assertEqual(counts["condition_bindings_completed"], 2)
        self.assertEqual(counts["controller_calls_completed"], 102)
        self.assertEqual(counts["deployed_action_comparisons"], 294)
        for name in ("actor", "q1", "width"):
            self.assertEqual(counts[name + "_batches_completed"], 1)
            self.assertIn((name, 8), f.calls)
        self.assertEqual([c for c in f.calls if c[0] == "initialize"], [("initialize", "honeypot", 0)])
        self.assertEqual([c for c in f.calls if c[0] == "restore"], [("restore", "honeypot", 0, "warmup")])
        self.assertTrue(result["restore"]["saved_updates_are_not_new_work"])
        self.assertEqual(result["restore"]["saved_cumulative_counts"]["critic"], 10000)
        self.assertEqual(result["controller"]["source"]["dynamics_calls"], 0)
        self.assertEqual(len([e for e in events if e["stage"] == "controller_case"]), 102)

    def test_metadata_mismatch_stops_before_model_initialization(self):
        f = fixture()
        f.prepared["bottleneck"].metadata = {"changed": True}
        events = []
        with self.assertRaisesRegex(ValueError, "membership differs|binding differs|membership|Titan condition"):
            self.run_fixture(f, persist=events.append)
        self.assertFalse(any(c[0] == "initialize" for c in f.calls))
        self.assertFalse(events[-1]["passed"])
        self.assertEqual(events[-1]["new_work"], C.ZERO_WORK)

    def test_normalization_change_is_not_tolerated(self):
        f = fixture()
        # Avoid mutating the expected fixture object along with the adapter.
        f.adapter.context.normalization = {**f.normalization, "mean": [1e-12] + [0.] * 13}
        events = []
        with self.assertRaisesRegex(ValueError, "normalization differs exactly"):
            self.run_fixture(f, persist=events.append)
        event = next(e for e in events if e["stage"] == "template_and_normalization")
        self.assertEqual(event["normalization"]["differences"]["mean"]["max_abs_difference"], 1e-12)
        self.assertFalse(any(c[0] == "restore" for c in f.calls))

    def test_bad_restore_counters_fail_before_inference(self):
        f = fixture()
        f.state["counts"] = {**f.saved_counts, "critic": 9999}
        with self.assertRaisesRegex(ValueError, "counters/tree differ"):
            self.run_fixture(f)
        self.assertFalse(any(c[0] == "actor" for c in f.calls))

    def test_nonfinite_inference_is_saved_as_diagnostic_text_and_fails(self):
        f = fixture()
        f.inference["width"] = lambda obs, act: np.full(len(obs), np.nan)
        events = []
        with self.assertRaisesRegex(ValueError, "Invalid finite"):
            self.run_fixture(f, persist=events.append)
        event = next(e for e in events if e["stage"] == "inference_width")
        self.assertEqual(event["values"], ["nan"] * 8)
        self.assertFalse(event["valid"])
        self.assertFalse(any(c[0] == "controller_init" for c in f.calls))

    def test_one_ulp_controller_difference_preserves_all_cases_then_fails(self):
        f = fixture()
        def make(*args):
            action = f.action.copy(); action[0] = np.nextafter(action[0], np.float32(1.))
            return lambda state, time: action.copy(), {"dynamics_calls": 0}
        events = []
        with self.assertRaisesRegex(ValueError, "controller differs"):
            self.run_fixture(f, controller_factory=make, persist=events.append)
        summary = next(e for e in events if e["stage"] == "controller_summary")
        self.assertEqual(summary["calls"], 102)
        self.assertEqual(summary["action_comparisons"], 294)
        self.assertGreater(summary["max_abs_difference"], 0.)
        self.assertEqual(len(summary["mismatched_case_ids"]), 102)
        self.assertFalse(events[-1]["passed"])

    def test_deadline_preserves_partial_counts_without_restore_or_updates(self):
        f = fixture()
        def deadline():
            if any(c[0] == "initialize" for c in f.calls): raise TimeoutError("fixture deadline")
        events = []
        with self.assertRaises(TimeoutError): self.run_fixture(f, deadline=deadline, persist=events.append)
        self.assertEqual(events[-1]["counts"]["template_initializations_completed"], 1)
        self.assertEqual(events[-1]["counts"]["checkpoint_restores_attempted"], 0)
        self.assertEqual(events[-1]["new_work"], C.ZERO_WORK)

    def test_terminal_write_failure_cannot_leave_success(self):
        f = fixture(); events = []
        def fail_completed(event):
            if event["stage"] == "completed": raise OSError("fixture result disk failure")
            events.append(event)
        with self.assertRaises(OSError): self.run_fixture(f, persist=fail_completed)
        self.assertEqual(events[-1]["stage"], "failed")
        self.assertEqual(events[-1]["status"], "failed_zero_update_titan_compatibility")
        self.assertFalse(events[-1]["passed"])

    def test_runtime_path_python_patch_and_no_blas_override(self):
        receipt = fixture().runtime_receipt
        with patch.dict(C.os.environ, frozen_environment(), clear=True):
            self.assertTrue(C._runtime_requirements(receipt)["python_patch_may_differ"])
            with self.assertRaisesRegex(ValueError, "interpreter"):
                C._runtime_requirements({**receipt, "executable": "/usr/bin/python"})
            with patch.dict(C.os.environ, {"OPENBLAS_CORETYPE": "HASWELL"}):
                with self.assertRaisesRegex(ValueError, "override"):
                    C._runtime_requirements(receipt)

    def test_source_bytes_are_exact(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "fixture.json"
            raw = b'{"fixture": true}\n'; path.write_bytes(raw)
            expected = hashlib.sha256(raw).hexdigest()
            self.assertEqual(C._bound(path, expected), {"fixture": True})
            path.write_bytes(raw.replace(b"\n", b"\r\n"))
            with self.assertRaisesRegex(ValueError, "Immutable Titan input"):
                C._bound(path, expected)


if __name__ == "__main__":
    unittest.main()
