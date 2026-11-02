"""Stdlib-only synthetic runner checks; never load real saved outcomes or fit models."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_decision_utility_runner_test", HERE / "run_checks.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def declared_input_allowlist():
    # Parse only the pinned prior reader's literal metadata; no prior function is
    # imported/executed and no saved-data file is opened by these tests.
    raw = (runner.PRIOR / "analyze_saved.py").read_bytes()
    if runner.sha(raw) != runner.PRIOR_READER_SHA256:
        raise AssertionError("Prior reader identity changed")
    tree = ast.parse(raw)
    batches = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                  and any(isinstance(name, ast.Name) and name.id == "BATCHES" for name in node.targets)))
    return {batch["directory"] + "/" + name: digest for batch in batches for name, digest in batch["files"].items()}


def valid_analysis():
    return {"status": "pass", "claim_scope": "fixed_heldout_signed_error_correction_diagnostic_only",
            "phase": "exploratory_saved_development_decision_utility_only", "new_policy_training": False, "new_model_fitted": False, "new_diagnostic_fits": 0,
            "new_simulator_calls": 0, "new_gpu_work": 0, "confirmation_files_read": 0,
            "full_training_hold": True, "automatic_training_approval": False,
            "method_or_strength_selected": False}


def synthetic_predictions():
    batches = []
    for batch_index, batch in enumerate(("development_1", "development_2")):
        rows = [{"group_id": batch_index * 64 + i, "batch": batch, "time": (0, 25, 50, 75)[i % 4],
                 "episode_seed": (2026102400, 2026102700)[batch_index] + i} for i in range(64)]
        batches.append({"name": batch, "queries": rows})
    records = {row["group_id"]: row for batch in batches for row in batch["queries"]}
    folds = runner.expected_fold_assignment(records)

    def predicted(group, fold):
        return {**records[group], "fold": fold, "predictions": {name: [0.0] * 9 for name in runner.PREDICTION_MODELS}}

    def fit(name, heldout):
        return {"split": name, "heldout_group_ids": sorted(heldout),
                "training_group_ids": sorted(set(records) - set(heldout)),
                "models": {name: {} for name in runner.PREDICTION_MODELS}}

    receipt = {"fold_assignment": folds, "fold_assignment_sha256": runner.canonical(folds),
               "cv": {"episodes": [predicted(g, folds[g]) for g in range(128)]},
               "fit_receipts": [fit("cv_fold_" + str(fold), {g for g in records if folds[g] == fold}) for fold in range(8)],
               "cross_batch_transfer": []}
    for train, held in (("development_1", "development_2"), ("development_2", "development_1")):
        groups = {g for g, row in records.items() if row["batch"] == held}
        receipt["fit_receipts"].append(fit("transfer_" + train + "_to_" + held, groups))
        receipt["cross_batch_transfer"].append({"training_batch": train, "heldout_batch": held,
                                               "episodes": [predicted(g, None) for g in sorted(groups)]})
    return batches, receipt


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.batches, self.receipt = synthetic_predictions()

    def validate(self):
        return runner.validate_predictions(self.batches, self.receipt)

    def test_complete_synthetic_oof_and_transfer_boundaries(self):
        result = self.validate()
        self.assertEqual(result["oof_episodes"], 128)
        self.assertEqual(result["training_episodes_per_fold"], 112)
        self.assertTrue(result["all_heldout_episode_ids_excluded_from_training"])

    def test_heldout_episode_leaking_into_training_rejected(self):
        fit = self.receipt["fit_receipts"][0]
        fit["training_group_ids"][0] = fit["heldout_group_ids"][0]
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_or_missing_oof_episode_rejected(self):
        original = deepcopy(self.receipt)
        self.receipt["cv"]["episodes"][1] = deepcopy(self.receipt["cv"]["episodes"][0])
        with self.assertRaises(ValueError):
            self.validate()
        self.receipt = original
        self.receipt["cv"]["episodes"].pop()
        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_fold_rejected_even_after_hash_reseal(self):
        self.receipt["fold_assignment"][0] = (self.receipt["fold_assignment"][0] + 1) % 8
        self.receipt["fold_assignment_sha256"] = runner.canonical(self.receipt["fold_assignment"])
        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_prediction_episode_metadata_rejected(self):
        self.receipt["cv"]["episodes"][0]["episode_seed"] = 2026102500
        with self.assertRaises(ValueError):
            self.validate()

    def test_nonfinite_misshaped_or_boolean_predictions_rejected(self):
        for values in ([0.0] * 8, [float("nan")] * 9, [float("inf")] * 9, [False] * 9):
            self.receipt["cv"]["episodes"][0]["predictions"]["base"] = values
            with self.assertRaises(ValueError):
                self.validate()

    def test_missing_prediction_model_rejected(self):
        del self.receipt["cv"]["episodes"][0]["predictions"]["base_plus_residual_width"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_cross_batch_training_episode_cannot_be_scored_as_heldout(self):
        self.receipt["cross_batch_transfer"][0]["episodes"][0] = {**self.batches[0]["queries"][0], "fold": None,
            "predictions": {name: [0.0] * 9 for name in runner.PREDICTION_MODELS}}
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_transfer_direction_rejected(self):
        self.receipt["cross_batch_transfer"][1] = deepcopy(self.receipt["cross_batch_transfer"][0])
        with self.assertRaises(ValueError):
            self.validate()


class ScopeTests(unittest.TestCase):
    def test_exact_reviewed_ten_input_scope(self):
        allowlist = declared_input_allowlist()
        runner.assert_input_scope(allowlist)
        self.assertEqual(len(allowlist), 10)

    def test_modified_input_hash_missing_file_and_added_confirmation_rejected(self):
        original = declared_input_allowlist()
        changed = dict(original)
        changed[next(iter(changed))] = "0" * 64
        missing = dict(original)
        del missing[next(iter(missing))]
        added = dict(original)
        added["unifloral_locked_confirmation_v1/report.json"] = "0" * 64
        for candidate in (changed, missing, added):
            with self.assertRaises(ValueError):
                runner.assert_input_scope(candidate)

    def test_no_fit_scope_accepted(self):
        runner.validate_analysis(valid_analysis())

    def test_policy_training_confirmation_and_approval_flags_rejected(self):
        changes = {"new_policy_training": True, "new_simulator_calls": 1, "new_gpu_work": 1,
                   "confirmation_files_read": 1, "automatic_training_approval": True,
                   "full_training_hold": False, "method_or_strength_selected": True,
                   "phase": "confirmation", "status": "fail", "new_model_fitted": True, "new_diagnostic_fits": 1}
        for key, value in changes.items():
            with self.subTest(key=key):
                detail = valid_analysis()
                detail[key] = value
                with self.assertRaises(ValueError):
                    runner.validate_analysis(detail)

    def test_boolean_cannot_masquerade_as_zero_work(self):
        detail = valid_analysis()
        detail["new_gpu_work"] = False
        with self.assertRaises(ValueError):
            runner.validate_analysis(detail)

    def test_invalid_postchecks_preserve_but_withhold_analysis(self):
        result = {"status": runner.SUCCESS, "identity_before": {"synthetic": 1},
                  "identity_after": {"synthetic": 2}, "analysis": valid_analysis()}
        runner.finalize(result)
        self.assertFalse(result["diagnostic_result_valid"])
        self.assertEqual(result["status"], "blocked_or_failed")
        self.assertNotIn("analysis", result)
        self.assertFalse(result["unvalidated_analysis"]["new_model_fitted"])

    def test_complete_stable_identity_result_is_valid(self):
        result = {"status": runner.SUCCESS, "identity_before": {"synthetic": 1},
                  "identity_after": {"synthetic": 1}, "analysis": valid_analysis()}
        runner.finalize(result)
        self.assertTrue(result["diagnostic_result_valid"])

    def test_preflight_never_counts_as_completed_model_result(self):
        result = {"status": "preflight_passed", "identity_before": {}, "identity_after": {}}
        runner.finalize(result)
        self.assertFalse(result["diagnostic_result_valid"])
        self.assertEqual(result["status"], "preflight_passed")


class TemporaryWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="decision_runner_synthetic_", dir=HERE)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        if self.root.resolve().parent != HERE.resolve():
            raise AssertionError("Refusing synthetic cleanup outside owned workspace")
        self.temporary.cleanup()

    def test_exclusive_output_cannot_overwrite(self):
        path = self.root / "receipt.json"
        digest = runner.write(path, {"synthetic": True})
        with self.assertRaises(FileExistsError):
            runner.write(path, {"synthetic": False})
        self.assertEqual(runner.sha(path.read_bytes()), digest)

    def test_checked_rejects_parent_and_absolute_paths(self):
        for relative in ("../elsewhere", str(self.root.resolve())):
            with self.assertRaises(ValueError):
                runner.checked(self.root, relative)

    def test_exact_read_module_hash_is_checked_before_execution(self):
        path = self.root / "synthetic_module.py"
        path.write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            runner.load_module("_not_loaded_grouped_test", path, "0" * 64)

    def test_timeout_kills_only_the_new_mock_child_and_preserves_log(self):
        calls = []

        class FakeProcess:
            def wait(self, timeout=None):
                calls.append(("wait", timeout))
                if timeout is not None:
                    raise subprocess.TimeoutExpired(["synthetic_never_launched"], timeout)
                return -1

            def kill(self):
                calls.append(("kill",))

        with patch.object(runner.subprocess, "Popen", return_value=FakeProcess()) as creation:
            result = runner.run_child(["synthetic_never_launched"], self.root, {})
        self.assertEqual(creation.call_count, 1)
        self.assertEqual(calls, [("wait", 60), ("kill",), ("wait", None)])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["exit_code"], -1)
        self.assertTrue((self.root / "checks.log").is_file())

    def test_default_preflight_never_launches_or_loads_outcomes(self):
        identity = {"synthetic_source": 1}
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "verify", return_value=(identity, {})), \
                patch.object(runner, "prior_reader", side_effect=AssertionError("must not load actual data")), \
                patch.object(runner.subprocess, "Popen", side_effect=AssertionError("must not launch")), \
                patch.object(runner.sys, "argv", ["run_checks.py"]), redirect_stdout(io.StringIO()):
            code = runner.main()
        reports = list(self.root.glob("results_*/report.json"))
        self.assertEqual(code, 0)
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_bytes())["status"], "preflight_passed")


class StaticTests(unittest.TestCase):
    def test_no_top_level_numpy_model_or_evaluator_import(self):
        tree = ast.parse((HERE / "run_checks.py").read_bytes())
        permitted = {"argparse", "datetime", "hashlib", "importlib", "json", "os", "pathlib", "platform",
                     "subprocess", "sys", "time", "traceback", "math", "random"}
        for node in tree.body:
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) else [])
            self.assertTrue(all(module.split(".")[0] in permitted for module in modules))

    def test_prior_only_verified_reader_api_is_called(self):
        source = (HERE / "run_checks.py").read_text(encoding="utf-8")
        self.assertIn("previous_identity, _ = reader.verify()", source)
        self.assertIn("batches = prior_reader().load_batches()", source)
        self.assertNotIn("reader.worker(", source)
        self.assertNotIn("reader.metrics_module(", source)
        self.assertNotIn("reader.analyze(", source)
        self.assertIn("result[\"identity_after\"], _ = verify()", source)


if __name__ == "__main__":
    unittest.main()
