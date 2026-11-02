"""Synthetic supervisor guards only; no GPU, model, or simulator execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_operator_runner_tests", ROOT / "run_checks.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def execution_fixture():
    ledger = {"status": "completed_bounded_operator_execution", "transition_rows_attempted": 1152,
        "transition_rows_completed": 1152, "max_transition_rows": 1152, "full_continuations": 0,
        "current_actor_rows": 128, "current_critic_rows": 1408, "width_rows": 1408,
        "target_actor_rows": 1408, "target_critic_rows": 5632, "max_target_critic_rows": 5632,
        "native_smoothing_draws": 8, "max_native_smoothing_draws": 8,
        "fitting_transition_rows_read": 256, "new_diagnostic_fits": 0,
        "policy_optimizer_updates": 0, "calibrator_updates": 0, "new_candidate_actions": 0,
        "confirmation_files_read": 0, "checkpoint_rng_advanced": False,
        "native_smoothing_draws_are_new_not_original_refresh_replay": True}
    state = {"full_state_unchanged": True, "inference_unchanged": True, "digest": "synthetic"}
    return {"ledger": ledger, "snapshot_before": state, "snapshot_after": deepcopy(state),
            "forward_replays": [{"passed": True, "rtol": 2e-5, "atol": 2e-6} for _ in range(3)]}


class RunnerTests(unittest.TestCase):
    def test_import_does_not_load_science(self):
        before = set(sys.modules)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        self.assertFalse(any(x.split('.')[0] in {"jax", "numpy", "flax", "optax"} for x in set(sys.modules)-before))

    def test_default_preflight_never_dispatches(self):
        with patch.object(r, "verify", return_value=({"synthetic": True}, {})), patch.object(r, "supervise") as launch, patch("builtins.print"):
            self.assertEqual(r.main([]), 0)
            launch.assert_not_called()

    def test_execute_requires_acknowledgement(self):
        with patch.object(r, "verify") as verify, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                r.main(["--execute"])
            verify.assert_not_called()

    def test_hidden_worker_requires_execute(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                r.main(["--worker-output", "ignored"])

    def test_conflicting_xla_rejected(self):
        with self.assertRaises(ValueError):
            r.execution_environment({"XLA_FLAGS": "--different"})

    def test_exact_runtime_environment_set_before_child(self):
        env = r.execution_environment({"BCA_TRANSFER_BUNDLE_ROOT": "portable_bundle"})
        self.assertEqual(env["XLA_FLAGS"], r.XLA)
        self.assertEqual(env["JAX_PLATFORMS"], "cuda")
        self.assertEqual(env["JAX_DEFAULT_MATMUL_PRECISION"], "highest")
        self.assertEqual(env["BCA_TRANSFER_BUNDLE_ROOT"], "portable_bundle")

    def test_complete_execution_gate(self):
        self.assertEqual(r.validate_execution(execution_fixture())["transition_rows_completed"], 1152)

    def test_incomplete_or_excess_dispatch_rejected(self):
        for key, value in (("transition_rows_completed", 1151), ("transition_rows_attempted", 1153),
                           ("target_critic_rows", 5633), ("native_smoothing_draws", 9)):
            fake = execution_fixture(); fake["ledger"][key] = value
            with self.assertRaises(ValueError):
                r.validate_execution(fake)

    def test_training_or_confirmation_work_rejected(self):
        for key in ("policy_optimizer_updates", "calibrator_updates", "confirmation_files_read", "full_continuations"):
            fake = execution_fixture(); fake["ledger"][key] = 1
            with self.assertRaises(ValueError):
                r.validate_execution(fake)

    def test_boolean_work_counter_rejected(self):
        fake = execution_fixture(); fake["ledger"]["calibrator_updates"] = False
        with self.assertRaises(ValueError):
            r.validate_execution(fake)

    def test_state_mutation_rejected(self):
        fake = execution_fixture(); fake["snapshot_after"]["digest"] = "changed"
        with self.assertRaises(ValueError):
            r.validate_execution(fake)

    def test_failed_forward_replay_rejected(self):
        fake = execution_fixture(); fake["forward_replays"][0]["passed"] = False
        with self.assertRaises(ValueError):
            r.validate_execution(fake)

    def test_prelabel_scope_and_training_hold_required(self):
        protocol = {"phase": "synthetic", "claim_scope": "descriptive"}
        result = {"status": "pass", **protocol, "full_training_hold": True, "automatic_training_approval": False,
            "new_policy_training": False, "new_diagnostic_fits": 0, "policy_optimizer_updates": 0,
            "calibrator_updates": 0, "confirmation_files_read": 0, "method_or_strength_selected": False}
        r.validate_analysis(result, protocol)
        with self.assertRaises(ValueError):
            r.validate_analysis({**result, "automatic_training_approval": True}, protocol)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_runner_test_", dir=ROOT)
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_exclusive_outputs(self):
        r.write(self.path / "one.json", {"synthetic": True})
        with self.assertRaises(FileExistsError):
            r.write(self.path / "one.json", {})

    def test_timeout_does_not_claim_zero_work(self):
        report = r.finalize_report(self.path, {}, -9, True, 300.0, {})
        self.assertEqual(report["status"], "timed_out")
        self.assertFalse(report["execution_counts_known"])
        self.assertNotIn("policy_optimizer_updates", report)
        self.assertIn("unknown", report["actual_work_after_last_receipt"])

    def test_exception_partial_ledger_preserved(self):
        r.write(self.path / "worker.json", {"status": "blocked_or_failed", "ledger": {"transition_rows_attempted": 1152}, "execution_counts_known": True})
        report = r.finalize_report(self.path, {}, 1, False, 2, {})
        self.assertEqual(report["ledger"]["transition_rows_attempted"], 1152)
        self.assertTrue(report["execution_counts_known"])

    def test_truncated_worker_receipt_preserves_unknown_work(self):
        (self.path / "worker.json").write_text('{"status":"completed', encoding="utf-8")
        report = r.finalize_report(self.path, {}, -9, True, 300, {})
        self.assertEqual(report["status"], "timed_out")
        self.assertFalse(report["execution_counts_known"])
        self.assertFalse(report["work_budget_verified"])
        self.assertFalse(report["frozen_state_verified"])
        self.assertIn("receipt_parse_error", report)

    def test_incomplete_success_receipt_rejected(self):
        r.write(self.path / "worker.json", {"status": r.SUCCESS})
        self.assertEqual(r.finalize_report(self.path, {}, 0, False, 1, {})["status"], "blocked_or_failed")

    def test_source_postcheck_change_blocks_success(self):
        r.write(self.path / "worker.json", {"status": r.SUCCESS})
        self.assertEqual(r.finalize_report(self.path, {"hash": 1}, 0, False, 1, {"hash": 2})["status"], "blocked_or_failed")

    def test_synthetic_bank_archive_exact_and_exclusive(self):
        import numpy as np
        bank = {"x": np.array([[1, 2], [3, 4]], np.float32), "metadata": [{"group_id": 0}], "valid": np.bool_(True)}
        before = bank["x"].copy()
        receipt = r.save_banks(self.path, bank, {"y": np.array([1.0], np.float64)})
        tree = json.loads((self.path / "raw_banks.json").read_text())
        with np.load(self.path / "raw_banks.npz", allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored[tree["development"]["x"]["npz_key"]], before)
        np.testing.assert_array_equal(bank["x"], before)
        self.assertEqual(receipt["raw_banks.npz"]["sha256"], r.sha((self.path / "raw_banks.npz").read_bytes()))
        with self.assertRaises(FileExistsError):
            r.save_banks(self.path, bank, {})

    def test_object_array_is_not_pickled(self):
        import numpy as np
        with self.assertRaises(ValueError):
            r.save_banks(self.path, {"x": np.array([object()], object)}, {})
        self.assertFalse((self.path / "raw_banks.npz").exists())


if __name__ == "__main__":
    unittest.main()
