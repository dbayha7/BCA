"""Pure fake-runtime tests: no JAX, real model, simulator or scientific updates."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("path_a_train_tested", Path(__file__).with_name("train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class FakeRuntime:
    def __init__(self):
        self.calls = 0
        self.fail = None
        self.restore_calls = 0

    host = staticmethod(lambda value: copy.deepcopy(value))
    tree_digest = staticmethod(lambda value: T.digest(wire(value)))

    def restore_exact(self, adapter, template, raw, diagnostic=None):
        self.restore_calls += 1
        value = json.loads(raw)
        self.validate(adapter, value)
        if diagnostic is not None:
            diagnostic["decoded_exact"] = True
        return value

    def validate(self, adapter, state):
        T.require(state["budget"] == adapter.budget, "fake budget mismatch")
        c = state["critic"]
        T.require(state["actor"] == c // 2 and state["auxiliary"] == c, "fake counts mismatch")
        return {"counts": {"critic": c, "actor": state["actor"], "auxiliary": state["auxiliary"],
                           "refresh_accepted": state["calibrations"], "phase": c % 2,
                           "critic_budget": state["budget"]}}

    def step(self, adapter, state, progress):
        self.calls += 1
        attempt = {"critic": 1, "actor": state["critic"] % 2, "auxiliary": 0, "refresh": 0}
        complete = dict.fromkeys(attempt, 0)
        def notify(stage):
            progress({"stage": stage, "event": "fake_dispatch", "attempted": dict(attempt), "completed": dict(complete)})
        notify("native_microstep")
        if self.fail == "native_inflight":
            raise RuntimeError("fake asynchronous interruption")
        complete["critic"], complete["actor"] = attempt["critic"], attempt["actor"]
        notify("native_microstep")
        if self.fail == "after_native":
            raise RuntimeError("fake auxiliary preparation failed")
        attempt["auxiliary"] = 1; notify("auxiliary_fit")
        complete["auxiliary"] = 1; notify("auxiliary_fit")
        next_c = state["critic"] + 1
        due = next_c in adapter.schedule
        if due:
            attempt["refresh"] = 1; notify("calibration_refresh")
            complete["refresh"] = 1; notify("calibration_refresh")
        result = {**state, "critic": next_c, "actor": next_c // 2, "auxiliary": next_c,
                  "calibrations": state["calibrations"] + int(due)}
        return result, self.validate(adapter, result)


def make_services():
    runtime = FakeRuntime()
    metadata = {"dataset_sha256": "d" * 64, "geometry_sha256": "e" * 64,
                "retention_mask_sha256": "a" * 64, "normalization": "retained fitting only"}
    prepared = SimpleNamespace(metadata=metadata)
    I = SimpleNamespace(digest=lambda value: T.digest(wire(value)))
    def initialize(rt, native, protocol, seed, condition, data, training_review):
        T.require(training_review == {"training_authorized": True, "protocol_sha256": I.digest(protocol),
                 "dataset_sha256": metadata["dataset_sha256"], "geometry_sha256": metadata["geometry_sha256"]}, "approval differs")
        context = SimpleNamespace(normalization={"condition": condition, "rows": 42}, runtime={"fake": True})
        adapter = SimpleNamespace(context=context, budget=10000, schedule=[10000], kappa=0.)
        state = {"critic": 0, "actor": 0, "auxiliary": 0, "calibrations": 0,
                 "budget": 10000, "native_rng": seed, "aux_rng": seed + 10,
                 "optimizer": {"actor": 0, "critic": 0}, "targets": [1, 2], "reference": "fixture"}
        return adapter, state
    def fork_pair(rt, adapter, warm, protocol, seed):
        T.require(warm["critic"] == 10000 and warm["calibrations"] == 1, "wrong warmup")
        state = {**copy.deepcopy(warm), "budget": 20000, "native_rng": seed + 100, "aux_rng": seed + 110}
        return {arm: (SimpleNamespace(context=adapter.context, budget=20000, schedule=[10000, 15000], kappa=kappa),
                      copy.deepcopy(state)) for arm, kappa in (("native", 0.), ("bca", 3.))}
    def restore_bound(rt, adapter, template, raw, receipt, prepared, expected_sha256, protocol, seed_index):
        T.require(T.digest(raw) == expected_sha256 and receipt["protocol_sha256"] == I.digest(protocol), "raw/protocol changed")
        T.require(receipt["condition"] == adapter.context.normalization["condition"] and receipt["seed_index"] == seed_index,
                  "wrong checkpoint condition/seed")
        T.require(receipt["normalization_sha256"] == I.digest(adapter.context.normalization), "wrong normalization")
        state = rt.restore_exact(adapter, template, raw)
        T.require(rt.tree_digest(state) == receipt["state_digest"], "wrong state")
        return state, {"decoded_exact": True}
    I.initialize, I.fork_pair, I.restore_bound = initialize, fork_pair, restore_bound
    binding = SimpleNamespace(I=I, prepare=lambda native, condition, protocol: prepared)
    services = SimpleNamespace(runtime=runtime, native=object(), binding=binding,
                               serialization=SimpleNamespace(to_bytes=wire))
    protocol = {"fixture": True}
    review = {"training_authorized": True, "protocol_sha256": I.digest(protocol),
              "dataset_sha256": metadata["dataset_sha256"], "geometry_sha256": metadata["geometry_sha256"]}
    return services, protocol, review


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.s, self.p, self.review = make_services()

    def run_job(self, arm="warmup", name=None, warmup=None, deadline=lambda: None):
        output = self.root / (name or arm)
        output.mkdir()
        report = T.run_job(self.p, "bottleneck", 0, arm, output, warmup,
                           deadline, services=self.s, training_review=self.review)
        return output, report

    def test_complete_paired_lifecycle_and_zero_update_final_load(self):
        warm, wr = self.run_job()
        native, nr = self.run_job("native", warmup=warm)
        bca, br = self.run_job("bca", warmup=warm)
        for report in (wr, nr, br):
            self.assertEqual(report["status"], "completed_training_job")
            self.assertEqual(report["counts"]["completed"], T.CAPS)
            self.assertTrue(report["execution_counts_known"])
        self.assertEqual(wr["final_cumulative_counts"]["critic"], 10000)
        self.assertEqual(nr["final_cumulative_counts"]["critic"], 20000)
        self.assertEqual(br["final_cumulative_counts"]["refresh_accepted"], 2)
        self.assertEqual(nr["job_start_state_digest"], br["job_start_state_digest"])
        self.assertEqual(nr["initial_state_digest"], wr["initial_state_digest"])
        count = self.s.runtime.calls
        adapter, state, _, _ = T.restore_final(self.p, "bottleneck", 0, "bca", bca, warm,
                                              services=self.s, training_review=self.review)
        self.assertEqual(self.s.runtime.calls, count)
        self.assertEqual(state["critic"], 20000)
        self.assertEqual(adapter.kappa, 3.)

    def test_inflight_update_is_unknown_not_zero(self):
        self.s.runtime.fail = "native_inflight"
        _, report = self.run_job()
        self.assertEqual(report["status"], "failed_training_job")
        self.assertFalse(report["execution_counts_known"])
        self.assertEqual(report["counts"]["attempted"]["critic"], 1)
        self.assertEqual(report["counts"]["completed"]["critic"], 0)

    def test_completed_uncommitted_work_is_preserved(self):
        self.s.runtime.fail = "after_native"
        _, report = self.run_job()
        self.assertTrue(report["execution_counts_known"])
        self.assertEqual(report["counts"]["completed"]["critic"], 1)
        self.assertEqual(report["counts"]["completed"]["auxiliary"], 0)
        self.assertEqual(report["status"], "failed_training_job")

    def test_initialization_timeout_consumes_no_updates_and_is_terminal(self):
        def stop():
            raise TimeoutError("global deadline")
        output, report = self.run_job(deadline=stop)
        self.assertEqual(report["counts"]["attempted"], dict.fromkeys(T.KEYS, 0))
        self.assertIn("global deadline", report["error"])
        with self.assertRaisesRegex(ValueError, "New launcher"):
            T.run_job(self.p, "bottleneck", 0, "warmup", output, deadline=stop,
                       services=self.s, training_review=self.review)

    def test_missing_approval_rejected_before_any_updates(self):
        self.review = None
        _, report = self.run_job()
        self.assertEqual(self.s.runtime.calls, 0)
        self.assertEqual(report["status"], "failed_training_job")

    def test_prepared_data_receipt_mismatch_refused_before_initialization(self):
        metadata = self.s.binding.prepare(None, "bottleneck", protocol=self.p).metadata
        self.s.expected_data_receipt = {"conditions": {"bottleneck": {"metadata": copy.deepcopy(metadata)}}}
        metadata["calibration_row_ids_sha256"] = "changed runtime selection"
        with patch.object(self.s.binding.I, "initialize", wraps=self.s.binding.I.initialize) as initialize:
            _, report = self.run_job()
            initialize.assert_not_called()
        self.assertEqual(self.s.runtime.calls, 0)
        self.assertIn("prepared exact receipt", report["error"])
        self.assertEqual(report["status"], "failed_training_job")

    def test_changed_checkpoint_bytes_refused_before_branch_updates(self):
        warm, _ = self.run_job()
        (warm / "full_state.msgpack").write_bytes(b"changed")
        count = self.s.runtime.calls
        _, report = self.run_job("native", warmup=warm)
        self.assertEqual(self.s.runtime.calls, count)
        self.assertEqual(report["status"], "failed_training_job")

    def test_incomplete_producer_never_forks(self):
        warm, _ = self.run_job()
        report = T.read(warm / "report.json")
        report["status"] = "failed_training_job"
        (warm / "report.json").write_text(json.dumps(report))
        count = self.s.runtime.calls
        _, result = self.run_job("native", warmup=warm)
        self.assertEqual(self.s.runtime.calls, count)
        self.assertIn("producer is incomplete", result["error"])

    def test_changed_mask_metadata_refused_before_branch_updates(self):
        warm, _ = self.run_job()
        self.s.binding.prepare(None, "bottleneck", self.p).metadata["retention_mask_sha256"] = "b" * 64
        _, result = self.run_job("native", warmup=warm)
        self.assertIn("retention/calibration", result["error"])
        self.assertEqual(result["counts"]["attempted"]["critic"], 0)

    def test_report_write_failure_propagates_not_success(self):
        original = T.save
        def fail(path, value):
            if Path(path).name == "report.json":
                raise OSError("disk full")
            return original(path, value)
        self.s.runtime.fail = "after_native"
        with patch.object(T, "save", fail):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_job()
        self.assertFalse((self.root / "warmup/report.json").exists())

    def test_ledger_rejects_duplicate_or_excess_physical_work(self):
        ledger = T.PhysicalLedger()
        ledger.attempted["critic"] = 10000
        ledger.begin_step()
        with self.assertRaisesRegex(ValueError, "cap exceeded"):
            ledger.dispatch({"attempted": {"critic": 1, "actor": 0, "auxiliary": 0, "refresh": 0},
                             "completed": {"critic": 0, "actor": 0, "auxiliary": 0, "refresh": 0}})

    def test_load_runtime_refuses_windows_before_imports(self):
        with patch.object(T.os, "name", "nt"), patch.object(T, "_module") as module:
            with self.assertRaisesRegex(ValueError, "Linux worker"):
                T.load_runtime(self.s.binding)
            module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
