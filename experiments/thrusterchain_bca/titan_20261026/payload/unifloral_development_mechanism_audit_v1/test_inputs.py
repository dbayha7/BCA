"""Synthetic input-boundary tests only. No real result or confirmation files read."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_mechanism_input_test", HERE / "analyze_saved.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def seal(prelabel):
    prelabel["prelabel_signature"] = reader.canonical({k: v for k, v in prelabel.items() if k != "prelabel_signature"})


def synthetic_fixture():
    spec = deepcopy(reader.BATCHES[0])
    ledger = {"phase": "development", "status": "complete", "query_count": 64,
              "query_seed": spec["query_seed"], "candidate_seed": spec["candidate_seed"],
              "query_times": [0, 25, 50, 75], "gamma": 0.98, "horizon": 100,
              "confirmation_starts_generated": 0, "learner_optimizer_updates": 0, "calibrator_updates": 0,
              "transition_steps_attempted": 38504, "transition_steps_completed": 38504,
              "prefix_steps_completed": 2400, "verification_steps_completed": 104,
              "continuation_steps_completed": 36000, "continuations_completed": 576,
              "monte_carlo_replications": 0, "deterministic_continuations_per_candidate": 1,
              "conditional_monte_carlo_standard_error": 0.0,
              "frozen_policy_unchanged": True, "candidate_selection_used_simulator_labels": False}
    prelabel = {"schema": "unifloral_frozen_usefulness_scores.v1", "phase": "development",
                "selections_frozen_before_truevalues": True, "protocol_sha256": spec["protocol_canonical_sha256"],
                "queries": [], "methods": {"raw_q": {"family": "raw_q", "available": True, "kappa_penalty": None,
                                                     "scores": [], "selected_indices": [], "selection_details": []}}}
    labels = {"ledger": ledger, "values": [], "conditional_monte_carlo_error": 0}
    arrays = {}
    actions = np.repeat(np.linspace(-0.8, 0.8, 9, dtype=np.float32)[:, None], 4, axis=1)
    q1 = np.arange(9, dtype=np.float64)
    q = np.stack((q1, q1 - 0.1), axis=1)
    for i in range(64):
        t = (0, 25, 50, 75)[i % 4]
        state = np.zeros(13, np.float64)
        observation = np.concatenate((state, [(100 - t) / 100])).astype(np.float32)
        arrays.update({"state_" + str(i): state, "observation_" + str(i): observation,
                       "actions_" + str(i): actions.copy(), "time_" + str(i): np.asarray(t, np.int32),
                       "episode_seed_" + str(i): np.asarray(spec["query_seed"] + i, np.int64),
                       "prefix_actions_" + str(i): np.zeros((t, 4), np.float32)})
        prelabel["queries"].append({"index": i, "episode_group_id": i, "observation": observation.tolist(),
                                    "actions": actions.tolist(), "q": q.tolist(), "u": np.linspace(1, 2, 9).tolist(),
                                    "disagreement": np.abs(q[:, 0] - q[:, 1]).tolist(),
                                    "actor_distance": np.mean((actions.astype(np.float64) - actions[0].astype(np.float64)) ** 2, axis=1).tolist(),
                                    "width_usable": True, "width_valid": True})
        method = prelabel["methods"]["raw_q"]
        method["scores"].append(q1.tolist())
        method["selected_indices"].append(8)
        method["selection_details"].append({"index": 8, "exact_argmax_index": 8, "near_best_indices": [8],
                                             "maximum_score": 8.0, "selected_score": 8.0, "top_two_gap": 1.0,
                                             "near_tie": False, "inconclusive_selection": False})
        labels["values"].append((q1 - 0.5).tolist())
    seal(prelabel)
    return spec, prelabel, labels, arrays


class InputTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.prelabel, self.labels, self.arrays = synthetic_fixture()

    def validate(self):
        return reader.validate_batch(self.spec, self.prelabel, self.labels, self.arrays, group_offset=0)

    def test_valid_aligned_synthetic_inputs(self):
        result = self.validate()
        self.assertEqual(len(result["queries"]), 64)
        self.assertEqual([r["group_id"] for r in result["queries"]], list(range(64)))
        self.assertEqual([sum(r["time"] == t for r in result["queries"]) for t in (0, 25, 50, 75)], [16] * 4)

    def test_confirmation_phase_rejected(self):
        self.prelabel["phase"] = "confirmation"
        seal(self.prelabel)
        with self.assertRaises(ValueError):
            self.validate()

    def test_modified_score_seal_rejected(self):
        self.prelabel["queries"][0]["u"][0] += 1
        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_source_protocol_rejected(self):
        self.prelabel["protocol_sha256"] = "0" * 64
        seal(self.prelabel)
        with self.assertRaises(ValueError):
            self.validate()

    def test_reserved_confirmation_seed_rejected(self):
        self.arrays["episode_seed_0"] = np.asarray(2026102500, np.int64)
        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_time_or_dtype_rejected(self):
        for value in (np.asarray(25, np.int32), np.asarray(0, np.int64)):
            self.arrays["time_0"] = value
            with self.assertRaises(ValueError):
                self.validate()

    def test_observation_state_mismatch_rejected(self):
        self.arrays["state_0"][0] = 1
        with self.assertRaises(ValueError):
            self.validate()

    def test_invalid_uncertainty_flags_rejected(self):
        self.prelabel["queries"][0]["width_valid"] = False
        seal(self.prelabel)
        with self.assertRaises(ValueError):
            self.validate()

    def test_nonfinite_or_misaligned_labels_rejected(self):
        for values in ([float("nan")] * 9, [float("inf")] * 9, [1.0] * 8):
            self.labels["values"][0] = values
            with self.assertRaises(ValueError):
                self.validate()

    def test_bad_raw_q_scores_rejected_even_after_reseal(self):
        self.prelabel["methods"]["raw_q"]["scores"][0][0] += 1
        seal(self.prelabel)
        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_saved_raw_selection_rejected(self):
        self.prelabel["methods"]["raw_q"]["selected_indices"][0] = 0
        seal(self.prelabel)
        with self.assertRaises(ValueError):
            self.validate()

    def test_first_near_best_rule_not_exact_argmax(self):
        self.prelabel["queries"][0]["q"][7][0] = 8 - 2e-6
        q = np.asarray(self.prelabel["queries"][0]["q"])
        self.prelabel["queries"][0]["disagreement"] = np.abs(q[:, 0] - q[:, 1]).tolist()
        method = self.prelabel["methods"]["raw_q"]
        method["scores"][0] = q[:, 0].tolist()
        method["selected_indices"][0] = 7
        method["selection_details"][0] = {"index": 7, "exact_argmax_index": 8, "near_best_indices": [7, 8],
                                           "maximum_score": 8.0, "selected_score": float(q[7, 0]),
                                           "top_two_gap": float(q[8, 0] - q[7, 0]),
                                           "near_tie": True, "inconclusive_selection": True}
        seal(self.prelabel)
        self.assertEqual(self.validate()["queries"][0]["raw_q_index"], 7)

    def test_control_arithmetic_corruption_rejected(self):
        self.prelabel["queries"][0]["actor_distance"][2] += 1
        seal(self.prelabel)
        with self.assertRaises(ValueError):
            self.validate()

    def test_near_best_uses_original_absolute_difference_at_rounding_boundary(self):
        maximum = 0.1
        tolerance = 1e-6 + 1e-6 * abs(maximum)
        q1 = np.linspace(-0.7, maximum, 9, dtype=np.float64)
        q1[7] = maximum - tolerance
        # The threshold-subtraction rewrite would include7; the original
        # subtraction/absolute-value arithmetic correctly excludes this boundary.
        self.assertGreater(abs(q1[7] - maximum), tolerance)
        query = self.prelabel["queries"][0]
        q = np.stack((q1, q1 - 0.1), axis=1)
        query["q"] = q.tolist()
        query["disagreement"] = np.abs(q[:, 0] - q[:, 1]).tolist()
        method = self.prelabel["methods"]["raw_q"]
        method["scores"][0] = q1.tolist()
        method["selected_indices"][0] = 8
        method["selection_details"][0] = {"index": 8, "exact_argmax_index": 8, "near_best_indices": [8],
                                           "maximum_score": maximum, "selected_score": maximum,
                                           "top_two_gap": float(q1[8] - q1[7]),
                                           "near_tie": False, "inconclusive_selection": False}
        seal(self.prelabel)
        self.assertEqual(self.validate()["queries"][0]["raw_q_index"], 8)

    def test_missing_or_extra_npz_arrays_rejected(self):
        self.arrays["unrequested_extra"] = np.zeros(1)
        with self.assertRaises(ValueError):
            self.validate()

    def test_nonzero_update_or_confirmation_ledger_rejected(self):
        for key in ("learner_optimizer_updates", "calibrator_updates", "confirmation_starts_generated"):
            self.labels["ledger"][key] = 1
            with self.assertRaises(ValueError):
                self.validate()
            self.labels["ledger"][key] = 0

    def test_input_allowlist_rejects_before_filesystem_read(self):
        with patch.object(reader, "safe_bytes", side_effect=AssertionError("must not touch files")):
            for path in ("unifloral_locked_confirmation_v1/report.json", "../anything", "arbitrary.json"):
                with self.assertRaises(ValueError):
                    reader.input_bytes(path)

    def test_no_scientific_or_evaluator_imports_at_module_top(self):
        tree = ast.parse((HERE / "analyze_saved.py").read_text(encoding="utf-8"))
        permitted = {"argparse", "datetime", "hashlib", "importlib", "io", "json", "os", "pathlib", "platform",
                     "subprocess", "sys", "time", "traceback"}
        for node in tree.body:
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            self.assertTrue(all(name.split(".")[0] in permitted for name in modules))


if __name__ == "__main__":
    unittest.main()
