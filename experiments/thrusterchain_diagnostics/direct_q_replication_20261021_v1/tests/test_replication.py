"""Six fixed synthetic cases and structural checks; no model or simulator loads.

Prepared as source only. Executing this suite is a separate post-review action.
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "frozen_replication_under_test", Path(__file__).resolve().parents[1] / "replication.py")
replication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replication)


def fixture(q1, truth, width, distance=None, permutation=None, actions=None):
    """One state with 12 indexed commands; duplicates remain separate indices."""
    if actions is None:
        actions = np.zeros((12, 4), dtype=np.float32)
        actions[:, 0] = np.linspace(-1, 1, 12, dtype=np.float32)
    bank = dict(q1=np.asarray(q1, float)[None, :],
                width=np.asarray(width, float)[None, :],
                distance=np.asarray(np.arange(12) if distance is None else distance, float)[None, :],
                permutation=np.asarray(np.arange(11, -1, -1) if permutation is None else permutation, int)[None, :],
                actions=np.asarray(actions, np.float32)[None, :, :],
                width_usable=np.ones((1, 12), dtype=bool))
    scores = replication.score_arrays(bank["q1"], bank["width"], bank["distance"],
                                      bank["width_usable"], bank["permutation"])
    return bank, np.asarray(truth, float)[None, :], scores


class SixFixedSyntheticCases(unittest.TestCase):
    def test_positive_alignment_improves_the_chosen_true_value(self):
        x = np.arange(12, dtype=float)
        bank, truth, scores = fixture(x, -x, 2*x)
        row = replication.state_metrics(bank, truth, scores)[0]
        self.assertTrue(row["valid"])
        self.assertAlmostEqual(row["rho_width"], 1.)
        self.assertAlmostEqual(row["pairwise_rho_width"], 1.)
        self.assertEqual(row["methods"]["raw_q"]["chosen_index"], 11)
        live = row["methods"]["live_width"]
        self.assertEqual(live["chosen_index"], 0)
        self.assertEqual(live["delta_vs_raw"], 11.)
        self.assertEqual(live["concordance"], 1.)
        self.assertEqual(row["methods"]["raw_q"]["concordance"], 0.)

    def test_negative_alignment_harms_the_chosen_true_value(self):
        x = np.arange(12, dtype=float)
        bank, truth, scores = fixture(24+x, 2*x, 4*x)
        row = replication.state_metrics(bank, truth, scores)[0]
        self.assertTrue(row["valid"])
        self.assertAlmostEqual(row["rho_width"], -1.)
        self.assertAlmostEqual(row["pairwise_rho_width"], -1.)
        self.assertEqual(row["methods"]["raw_q"]["chosen_index"], 11)
        live = row["methods"]["live_width"]
        self.assertEqual(live["chosen_index"], 0)
        self.assertEqual(live["delta_vs_raw"], -22.)
        self.assertEqual(live["concordance"], 0.)
        self.assertEqual(row["methods"]["raw_q"]["concordance"], 1.)

    def test_degenerate_correlations_are_undefined_but_state_is_valid(self):
        bank, truth, scores = fixture(np.full(12, 5.), np.full(12, 3.), np.arange(12), np.ones(12))
        row = replication.state_metrics(bank, truth, scores)[0]
        self.assertTrue(row["valid"])
        self.assertEqual(row["invalid_reasons"], [])
        for name in ("width", "distance", "shuffled"):
            self.assertIsNone(row["rho_"+name])
            self.assertIsNone(row["pairwise_rho_"+name])
            self.assertIn("zero_signed_error_centered_norm", row["rho_undefined_reasons"][name])
        self.assertCountEqual(row["rho_undefined_reasons"]["distance"],
                              ["zero_predictor_centered_norm", "zero_signed_error_centered_norm"])
        for method in replication.METHODS:
            result = row["methods"][method]
            self.assertIsNone(result["concordance"])
            self.assertEqual(result["concordance_undefined_reason"], "no_comparable_true_pairs")
            self.assertEqual(result["comparable_pairs"], 0)
            self.assertEqual(result["true_near_ties"], 66)
            self.assertEqual(result["chosen_true_q"], 3.)
            self.assertEqual(result["delta_vs_raw"], 0.)
        self.assertIsNone(replication.pearson(np.ones(3), np.arange(3)))
        self.assertIsNone(replication.pairwise_pearson(np.ones(3), np.arange(3)))

    def test_ties_and_duplicate_commands_keep_index_and_command_counts_separate(self):
        q1 = np.zeros(12); q1[[0, 1, 6]] = 5.; q1[2] = 4.
        width = np.zeros(12); width[[0, 1, 6]] = 2.
        truth = np.zeros(12); truth[[0, 1, 6]] = 2.; truth[2] = 3.
        actions = np.zeros((12, 4), np.float32)
        actions[:, 0] = np.linspace(-1, 1, 12, dtype=np.float32)
        actions[1] = actions[0]; actions[6] = actions[0]
        permutation = np.arange(12); permutation[1], permutation[3] = 3, 1
        bank, labels, scores = fixture(q1, truth, width, permutation=permutation, actions=actions)
        row = replication.state_metrics(bank, labels, scores)[0]
        raw, shuffled = row["methods"]["raw_q"], row["methods"]["shuffled_width"]
        self.assertEqual(raw["chosen_index"], 0, "Exact ties use the first candidate index")
        self.assertEqual(shuffled["chosen_index"], 1)
        self.assertTrue(shuffled["changed_index"])
        self.assertFalse(shuffled["changed_command"])
        self.assertTrue(shuffled["selected_actor_command"])
        self.assertFalse(shuffled["actor_restoration"], "The raw choice already equals the actor command")
        self.assertEqual(shuffled["delta_vs_raw"], 0.)
        pair = replication.concordance(np.array([0., 1e-6, 2.]), np.array([0., 1., 2.]))
        self.assertEqual(pair["pairs"], 3)
        self.assertEqual(pair["predicted_ties_among_comparable"], 1)
        self.assertAlmostEqual(pair["value"], 5/6)
        true_tie = replication.concordance(np.array([0., 1.]), np.array([0., 1e-6]))
        self.assertEqual(true_tie["true_ties"], 1)
        self.assertIsNone(true_tie["value"])

    def test_constant_offset_preserves_the_full_ranking_and_fixed_coefficient(self):
        q1 = np.array([0., 2., 2., -1., 1., 0., 1., -2., .5, 0., 1., -1.])
        width = np.arange(12, dtype=float)/4
        bank, truth, scores = fixture(q1, np.arange(12), width)
        np.testing.assert_array_equal(np.argsort(scores["raw_q"][0], kind="stable"),
                                      np.argsort(scores["constant_offset"][0], kind="stable"))
        np.testing.assert_array_equal(scores["constant_offset"], bank["q1"]-width.mean())
        self.assertEqual(replication.COEFFICIENT, 0.519506430786272)
        np.testing.assert_array_equal(scores["matched_distance"], bank["q1"]-0.519506430786272*bank["distance"])
        np.testing.assert_array_equal(scores["shuffled_width"], bank["q1"]-width[::-1][None, :])
        row = replication.state_metrics(bank, truth, scores)[0]
        constant = row["methods"]["constant_offset"]
        self.assertEqual(constant["chosen_index"], 1)
        self.assertEqual(constant["delta_vs_raw"], 0.)
        self.assertFalse(constant["changed_index"])
        self.assertFalse(constant["changed_command"])

    def test_invalid_rows_are_preserved_and_finite_width_refusal_uses_raw_score(self):
        base, labels, _ = fixture(np.arange(12), -np.arange(12), np.arange(12))
        bank = {key: np.repeat(value, 6, axis=0) for key, value in base.items()}
        truth = np.repeat(labels, 6, axis=0)
        bank["q1"][1, 0] = np.nan
        bank["width"][2, 0] = np.inf
        bank["width"][3, 0] = -1.
        truth[4, 0] = np.nan
        bank["width_usable"][5] = False
        scores = replication.score_arrays(bank["q1"], bank["width"], bank["distance"],
                                          bank["width_usable"], bank["permutation"])
        np.testing.assert_array_equal(replication.valid_state_mask(bank, truth), [True, False, False, False, False, True])
        rows = replication.state_metrics(bank, truth, scores)
        self.assertEqual(len(rows), 6, "Invalid states must not disappear from the report")
        for index, reason in ((1, "nonfinite_q1"), (2, "nonfinite_width"), (3, "negative_width"), (4, "nonfinite_or_missing_truth")):
            self.assertFalse(rows[index]["valid"])
            self.assertIn(reason, rows[index]["invalid_reasons"])
            self.assertEqual(rows[index]["methods"], {})
            self.assertEqual(rows[index]["rho_undefined_reasons"],
                             {name: ["state_invalid"] for name in ("width", "distance", "shuffled")})
        self.assertTrue(np.isnan(scores["raw_q"][1, 0]))
        self.assertTrue(np.isinf(bank["width"][2, 0]))
        self.assertTrue(rows[5]["valid"], "Finite unusable widths are an explicit score refusal, not an invalid numeric state")
        np.testing.assert_array_equal(scores["live_width"][5], scores["raw_q"][5])
        np.testing.assert_array_equal(scores["shuffled_width"][5], scores["raw_q"][5])


class StructuralChecks(unittest.TestCase):
    def test_endpoint_and_pair_specific_masks_keep_informative_width_states(self):
        x = np.arange(12, dtype=float)
        base, labels, _ = fixture(x, -x, 2*x, distance=np.ones(12))
        # Repeated deterministic draw indices isolate endpoint-mask arithmetic;
        # the 10000-draw stratified construction has a separate structural test.
        draws = np.tile(np.arange(32), (3, 1))
        for distance_defined in (0, 16):
            with self.subTest(distance_defined=distance_defined):
                bank = {key: np.repeat(value, 32, axis=0) for key, value in base.items()}
                bank["distance"][:distance_defined] = x
                truth = np.repeat(labels, 32, axis=0)
                scores = replication.score_arrays(bank["q1"], bank["width"], bank["distance"],
                                                  bank["width_usable"], bank["permutation"])
                summary, rows, _ = replication.analyze(bank, truth, scores, draws)
                metrics = summary["metrics"]
                self.assertEqual(summary["pair_common_defined_rho_states"],
                                 {"distance": distance_defined, "shuffled": 32})
                self.assertEqual(metrics["primary_rho_width"]["defined_states"], 32)
                self.assertAlmostEqual(metrics["primary_rho_width"]["mean"], 1.)
                self.assertEqual(metrics["primary_rho_shuffled"]["defined_states"], 32)
                self.assertAlmostEqual(metrics["primary_rho_shuffled"]["mean"], -1.)
                self.assertEqual(metrics["primary_rho_distance"]["defined_states"], distance_defined)
                self.assertEqual(metrics["primary_rho_width_minus_distance"]["defined_states"], distance_defined)
                self.assertEqual(metrics["primary_rho_width_minus_shuffled"]["defined_states"], 32)
                self.assertAlmostEqual(metrics["primary_rho_width_minus_shuffled"]["mean"], 2.)
                undefined = rows[-1]["rho_undefined_reasons"]
                self.assertEqual(undefined["distance"], ["zero_predictor_centered_norm"])
                self.assertEqual(undefined["width"], [])
                self.assertEqual(undefined["shuffled"], [])
                if distance_defined == 0:
                    self.assertIsNone(metrics["primary_rho_distance"]["mean"])
                    self.assertIsNone(metrics["primary_rho_width_minus_distance"]["mean"])
                    self.assertIsNone(metrics["primary_rho_width_minus_distance"]["ci95"])
                else:
                    self.assertAlmostEqual(metrics["primary_rho_distance"]["mean"], 1.)
                    self.assertAlmostEqual(metrics["primary_rho_width_minus_distance"]["mean"], 0.)

    def test_rollout_integrity_flags_invalidate_states_but_physical_failure_does_not(self):
        base, labels, _ = fixture(np.arange(12), -np.arange(12), np.arange(12))
        bank = {key: np.repeat(value, 5, axis=0) for key, value in base.items()}
        truth = np.repeat(labels, 5, axis=0)
        bank["time"] = np.array([0, 50, 99, 0, 99])
        bank["rollout_completed"] = np.ones((5, 12), bool)
        bank["rollout_finite"] = np.ones((5, 12), bool)
        bank["rollout_steps"] = np.repeat((100-bank["time"])[:, None], 12, axis=1)
        bank["rollout_failure"] = np.zeros((5, 12), bool)
        bank["rollout_completed"][1, 0] = False
        bank["rollout_finite"][2, 1] = False
        bank["rollout_steps"][3, 2] -= 1
        bank["rollout_failure"][4, :] = True
        np.testing.assert_array_equal(replication.valid_state_mask(bank, truth), [True, False, False, False, True])
        reasons = replication.invalid_state_reasons(bank, truth)
        self.assertEqual(reasons[0], [])
        self.assertEqual(reasons[4], [], "Crossing the physical failure threshold does not invalidate a complete finite return")
        for index, reason in ((1, "rollout_completed_false"), (2, "rollout_finite_false"), (3, "incomplete_rollout_steps")):
            self.assertEqual(reasons[index], [reason])
        scores = replication.score_arrays(bank["q1"], bank["width"], bank["distance"],
                                          bank["width_usable"], bank["permutation"])
        rows = replication.state_metrics(bank, truth, scores)
        self.assertEqual(len(rows), 5)
        self.assertTrue(rows[4]["valid"])
        self.assertIn("chosen_true_q", rows[4]["methods"]["raw_q"])

    def test_pairwise_identity_on_non_affine_vectors_and_shape_errors(self):
        x, y = np.array([0., 1., 4., 8.]), np.array([2., -1., 3., 0.])
        i, j = np.triu_indices(len(x), 1)
        dx, dy = x[i]-x[j], y[i]-y[j]
        expected = np.dot(dx, dy)/(np.linalg.norm(dx)*np.linalg.norm(dy))
        self.assertAlmostEqual(replication.pearson(x, y), expected, places=14)
        self.assertAlmostEqual(replication.pairwise_pearson(x, y), expected, places=14)
        self.assertAlmostEqual(replication.pearson(x+100, 3*y-20), expected, places=14)
        for function in (replication.pearson, replication.pairwise_pearson):
            with self.assertRaises(ValueError):
                function(np.ones((2, 2)), np.ones((2, 2)))
            with self.assertRaises(ValueError):
                function(np.ones(2), np.ones(3))
            self.assertIsNone(function(np.array([0., np.nan]), np.array([0., 1.])))

    def test_bootstrap_preserves_every_original_context_time_cell(self):
        bank = replication.generate_states()
        self.assertEqual(replication.SEED, 2026102123)
        self.assertEqual(bank["states"].shape, (32, 13))
        self.assertEqual(bank["observation"].dtype, np.float32)
        strata = bank["context"]*100+bank["time"]
        draws = replication.bootstrap_plan(bank["context"], bank["time"])
        self.assertEqual(draws.shape, (10000, 32))
        np.testing.assert_array_equal(strata[draws], np.broadcast_to(strata, draws.shape))
        self.assertEqual([np.count_nonzero(strata == label) for label in np.unique(strata)], [3, 3, 2]*4)
        np.testing.assert_array_equal(draws, replication.bootstrap_plan(bank["context"], bank["time"]))
        record, means, counts = replication.bootstrap_statistic(strata.astype(float), np.ones(32, bool), draws)
        self.assertEqual(record["ci95"], [float(strata.mean())]*2)
        np.testing.assert_array_equal(counts, np.full(10000, 32))
        np.testing.assert_array_equal(means, np.full(10000, strata.mean()))

    def test_undefined_bootstrap_draw_nulls_interval_without_dropping_draws(self):
        draws = np.array([[0, 1], [1, 1], [0, 0]], dtype=int)
        record, means, counts = replication.bootstrap_statistic(np.array([2., np.nan]), np.array([True, True]), draws)
        self.assertEqual(record["planned_states"], 2)
        self.assertEqual(record["defined_states"], 1)
        self.assertEqual(record["undefined_bootstrap_draws"], 1)
        self.assertEqual(record["interval_status"], "unavailable_undefined_draws")
        self.assertIsNone(record["ci95"])
        np.testing.assert_array_equal(counts, [1, 0, 2])
        np.testing.assert_array_equal(means, [2., np.nan, 2.])
        self.assertEqual(record["bootstrap_defined_denominator_min"], 0)
        self.assertEqual(record["bootstrap_defined_denominator_max"], 2)

    def test_scoring_rejects_nonpermutations_and_shape_mismatch(self):
        q = np.zeros((1, 12)); usable = np.ones_like(q, bool)
        with self.assertRaises(ValueError):
            replication.score_arrays(q, q, q, usable, np.zeros((1, 12), int))
        with self.assertRaises(ValueError):
            replication.score_arrays(q, q[:, :11], q, usable, np.arange(12)[None, :])


if __name__ == "__main__":
    unittest.main()
