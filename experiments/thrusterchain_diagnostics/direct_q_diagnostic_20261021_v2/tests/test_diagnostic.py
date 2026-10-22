"""Source-only test suite prepared before the diagnostic freeze.

Default tests exercise pure input/ranking arithmetic and do not load a model.
Set DIRECT_Q_FROZEN_TESTS=1 only after review to enable authenticated replay
fixtures. The integration fixtures use no diagnostic evaluation/adaptation rows.
"""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "diagnostic.py"
SPEC = importlib.util.spec_from_file_location("direct_q_diagnostic_under_test", MODULE_PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)

ATOMS = np.array([-1, -.8, 0, .8, 1], dtype=np.float32)


class NumericalDeploymentTests(unittest.TestCase):
    def test_native_actor_always_receives_one_row_independent_of_outer_batch(self):
        sizes = []
        def policy(observations):
            sizes.append(len(observations))
            return observations[:, :4] + np.float32(len(observations))
        engine = mock.Mock()
        engine.result.policy = policy
        observations = np.arange(45, dtype=np.float64).reshape(3, 15)
        combined = diagnostic.single_row_policy(engine, observations)
        separate = np.concatenate([diagnostic.single_row_policy(engine, row[None]) for row in observations])
        np.testing.assert_array_equal(combined, separate)
        self.assertEqual(sizes, [1]*6)
        self.assertEqual(combined.dtype, np.float32)
        self.assertEqual(diagnostic.single_row_policy(engine, np.empty((0,15))).shape, (0,4))
        for invalid in (np.zeros(15), np.zeros((1,14)), np.full((1,15), np.nan)):
            with self.assertRaises(ValueError):
                diagnostic.single_row_policy(engine, invalid)


class StateLawTests(unittest.TestCase):
    def test_evaluation_balance_times_and_physical_law(self):
        bank = diagnostic.generate_states(8, 1)
        self.assertEqual(bank["states"].shape, (32, 13))
        self.assertEqual(bank["observation"].shape, (32, 15))
        self.assertEqual(bank["states"].dtype, np.float64)
        self.assertEqual(bank["observation"].dtype, np.float32)
        np.testing.assert_array_equal(np.bincount(bank["context"]), [8] * 4)
        for context in range(4):
            selected = bank["context"] == context
            times, counts = np.unique(bank["time"][selected], return_counts=True)
            np.testing.assert_array_equal(times, [0, 50, 99])
            np.testing.assert_array_equal(counts, [3, 3, 2])
            np.testing.assert_array_equal(bank["regimes"][selected], np.full(8, context // 2))
            np.testing.assert_array_equal(bank["states"][selected, 10], np.full(8, .3 if context % 2 == 0 else 1.1))
        self.assertTrue(np.all(np.abs(bank["states"][:, :10]) <= .5))
        np.testing.assert_allclose(np.linalg.norm(bank["states"][:, 11:13], axis=1), .2, atol=1e-15)
        np.testing.assert_array_equal(bank["observation"][:, :13], bank["states"].astype(np.float32))
        np.testing.assert_array_equal(bank["observation"][:, 14], ((100-bank["time"])/100).astype(np.float32))

    def test_stream_replay_and_fresh_adaptation_law(self):
        first, replay = diagnostic.generate_states(8, 1), diagnostic.generate_states(8, 1)
        for key in first:
            np.testing.assert_array_equal(first[key], replay[key])
        adaptation = diagnostic.generate_states(2, 3)
        self.assertEqual(adaptation["states"].shape, (8, 13))
        np.testing.assert_array_equal(adaptation["time"], [0, 50] * 4)
        self.assertFalse(np.array_equal(first["states"][:8], adaptation["states"]))
        for size, stream in ((7, 1), (8, 2), (2, 5)):
            with self.subTest(size=size, stream=stream), self.assertRaises(ValueError):
                diagnostic.generate_states(size, stream)


class CandidateSupportTests(unittest.TestCase):
    def test_exact_categorical_mass_and_actual_float32_support(self):
        actions = np.array([[
            [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0],
            [1, 1, 1, 1], [.2, 0, 0, 0], [.8, 0, 0, 0],
        ]], dtype=np.float32)
        classified = diagnostic.classify_actions(actions, np.zeros((1, 4), np.float32), ATOMS)
        np.testing.assert_allclose(classified["beta"][0], [.84**4, .04*.84**3, .04**2*.84**2, .04**4, 0, .04*.84**3], rtol=1e-14)
        np.testing.assert_array_equal(classified["support_class"][0], ["common_on_grid", "common_on_grid", "rare_on_grid", "rare_on_grid", "unsupported", "common_on_grid"])
        np.testing.assert_array_equal(classified["grid_id"][0], [312, 562, 612, 624, -1, 437])
        adjacent = actions[:, :1].copy()
        adjacent[0, 0, 0] = np.nextafter(ATOMS[3], np.float32(np.inf))
        result = diagnostic.classify_actions(adjacent, np.zeros((1, 4), np.float32), ATOMS)
        self.assertFalse(result["supported"][0, 0], "An adjacent float32 value is not an exact grid atom")
        self.assertEqual(result["beta"][0, 0], 0.)

    def test_candidates_have_six_unique_atoms_and_actual_support_labels(self):
        observations = np.zeros((2, 15), dtype=np.float32)
        actors = np.array([[0, 0, 0, 0], [.21, -.11, .51, -.39]], np.float32)
        modes = np.zeros((2, 4), np.float32)
        bank = diagnostic.build_candidates(observations, actors, modes, ATOMS, diagnostic.rng(2))
        repeated = diagnostic.build_candidates(observations, actors, modes, ATOMS, diagnostic.rng(2))
        self.assertEqual(bank["actions"].shape, (2, 12, 4))
        np.testing.assert_array_equal(bank["actions"], repeated["actions"])
        for row in range(2):
            self.assertEqual(len(np.unique(bank["grid_id"][row, :6])), 6)
            self.assertTrue(np.all(bank["supported"][row, :6]))
            self.assertTrue(np.any(np.all(bank["actions"][row, :6] == modes[row], axis=1)))
        np.testing.assert_array_equal(bank["actions"][:, 6], actors)
        self.assertTrue(bank["supported"][0, 6], "The continuous-actor slot can lie exactly on the grid")
        self.assertFalse(bank["supported"][1, 6])
        self.assertTrue(np.all(np.abs(bank["actions"]) <= 1.))
        expected_distance = np.mean((bank["actions"].astype(float)-modes[:, None].astype(float))**2, axis=-1)
        np.testing.assert_array_equal(bank["distance"], expected_distance)

    def test_nonfinite_actor_is_rejected_before_clipping(self):
        for value in (np.inf, -np.inf, np.nan):
            actor = np.zeros((1, 4), np.float32)
            actor[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostic.build_candidates(np.zeros((1, 15)), actor, np.zeros((1, 4)), ATOMS, diagnostic.rng(2))


class MatchingAndScoreTests(unittest.TestCase):
    def test_coefficient_uses_within_state_centering_without_outcomes(self):
        distance = np.tile(np.arange(12, dtype=float), (8, 1))/11
        width = 3*distance + np.arange(8)[:, None]
        receipt = diagnostic.matching_coefficient(width, distance, np.ones_like(width, bool))
        shifted = diagnostic.matching_coefficient(width+100*np.arange(8)[:, None], distance+np.arange(8)[:, None], np.ones_like(width, bool))
        self.assertEqual(receipt["status"], "matched_without_returns")
        self.assertAlmostEqual(receipt["coefficient"], 3., places=12)
        self.assertAlmostEqual(shifted["coefficient"], 3., places=11)
        self.assertEqual(receipt["truth_labels_used"], 0)
        self.assertEqual(receipt["observations"], 8)
        self.assertEqual(receipt["candidates_per_observation"], 12)

    def test_matching_refuses_instead_of_filtering_or_retuning(self):
        distance = np.tile(np.arange(12, dtype=float), (8, 1))
        usable = np.ones_like(distance, bool)
        invalid = distance.copy(); invalid[0, 0] = np.inf
        partial = usable.copy(); partial[0, 0] = False
        cases = [(invalid, distance, usable), (distance, distance, partial),
                 (distance, np.ones_like(distance), usable), (distance, distance[:, :4], usable)]
        for width, d, ok in cases:
            with self.subTest(width_shape=width.shape, distance_shape=d.shape):
                receipt = diagnostic.matching_coefficient(width, d, ok)
                self.assertEqual(receipt["status"], "refused")
                self.assertIsNone(receipt["coefficient"])

    def test_width_refusal_is_explicit_raw_score_fallback(self):
        bank = dict(q1=np.array([[5., 4., 3., 2.]]),
                    width=np.array([[3., np.inf, np.nan, 2.]]),
                    width_usable=np.array([[True, False, False, False]]),
                    distance=np.array([[1., 2., 3., 4.]]))
        scores = diagnostic.score_arrays(bank, {"coefficient": 2.})
        np.testing.assert_array_equal(scores["live_width"], [[2., 4., 3., 2.]])
        np.testing.assert_array_equal(scores["detached_width"], scores["live_width"])
        np.testing.assert_array_equal(scores["matched_distance"], [[3., 0., -3., -6.]])
        self.assertEqual(diagnostic.first_argmax(scores["constant_offset"][0]), 0)
        refused = diagnostic.score_arrays(bank, {"coefficient": None})
        self.assertTrue(np.isnan(refused["matched_distance"]).all())


class RankingAndAggregationTests(unittest.TestCase):
    def test_first_index_ties_are_independent_of_oracle(self):
        self.assertEqual(diagnostic.first_argmax([4., 4., 1.]), 0)
        self.assertEqual(diagnostic.first_argmax([4., 4., 1.], [False, True, True]), 1)
        with self.assertRaises(ValueError):
            diagnostic.first_argmax([1., np.nan])
        with self.assertRaises(ValueError):
            diagnostic.first_argmax([1., 2.], [False, False])

    def test_concordance_half_credit_and_true_tie_exclusion(self):
        tied = diagnostic.concordance(np.array([2., 2., 0.]), np.array([3., 2., 1.]))
        self.assertAlmostEqual(tied["value"], 5/6)
        self.assertEqual(tied["predicted_ties"], 1)
        self.assertEqual(tied["pairs"], 3)
        oracle_tie = diagnostic.concordance(np.array([1., 0., 2.]), np.array([3., 3., 1.]))
        self.assertEqual(oracle_tie["true_ties"], 1)
        self.assertEqual(oracle_tie["pairs"], 2)
        self.assertEqual(oracle_tie["value"], 0.)
        all_ties = diagnostic.concordance(np.array([1., 2.]), np.array([3., 3.]))
        self.assertIsNone(all_ties["value"])

    def test_pairwise_near_ties_do_not_change_exact_argmax(self):
        predicted = np.array([0., 1e-6])
        near = diagnostic.concordance(predicted, np.array([0., 1.]))
        self.assertEqual(near["value"], .5)
        self.assertEqual(near["predicted_ties"], 1)
        self.assertEqual(near["predicted_ties_among_comparable"], 1)
        self.assertEqual(diagnostic.first_argmax(predicted), 1, "Decision argmax uses exact values, not the pairwise tie tolerance")
        excluded = diagnostic.concordance(np.array([0., 1.]), np.array([0., 1e-6]))
        self.assertEqual(excluded["true_ties"], 1)
        self.assertEqual(excluded["pairs"], 0)
        self.assertIsNone(excluded["value"])
        beyond = diagnostic.concordance(np.array([0., 1.]), np.array([0., np.nextafter(1e-6, np.inf)]))
        self.assertEqual(beyond["pairs"], 1)
        self.assertEqual(beyond["value"], 1.)

    def test_bootstrap_stratifies_context_times_preserves_counts_and_replays(self):
        bank = diagnostic.generate_states(8, 1)
        strata = bank["context"]*100+bank["time"]
        calls = []

        class RecordingRandom:
            def choice(self, ids, size, replace):
                calls.append((np.asarray(ids).copy(), size, replace))
                return np.full(size, ids[0], dtype=int)

        # A value constant within each stratum must give an exact degenerate
        # interval when every bootstrap draw preserves the stratum counts.
        values = strata.astype(float)
        expected = [float(values.mean())]*2
        self.assertEqual(diagnostic.bootstrap_interval(values, strata, RecordingRandom()), expected)
        self.assertEqual(len(calls), 12)
        for (ids, size, replace), label in zip(calls, np.unique(strata)):
            expected_ids = np.flatnonzero(strata == label)
            np.testing.assert_array_equal(ids, expected_ids)
            self.assertEqual(size, (10000, len(expected_ids)))
            self.assertTrue(replace)
        self.assertEqual([len(ids) for ids, _, _ in calls], [3, 3, 2]*4)
        spread = np.arange(32, dtype=float)
        self.assertEqual(diagnostic.bootstrap_interval(spread, strata, diagnostic.rng(5)),
                         diagnostic.bootstrap_interval(spread, strata, diagnostic.rng(5)))

    def test_bootstrap_refuses_any_nonempty_singleton_stratum(self):
        random = mock.Mock()
        self.assertIsNone(diagnostic.bootstrap_interval(np.array([1., 2., 3.]), np.array([0, 0, 50]), random))
        self.assertIsNone(diagnostic.bootstrap_interval(np.array([]), np.array([], dtype=int), random))
        random.choice.assert_not_called()
        # Missing strata are not invented; two states in the only present
        # stratum suffice for this conditional subgroup interval.
        self.assertEqual(diagnostic.bootstrap_interval(np.array([2., 2.]), np.array([99, 99]), diagnostic.rng(5)), [2., 2.])

    def test_aggregation_reports_paired_harm_and_opportunity_arithmetic(self):
        bank = diagnostic.generate_states(8, 1)
        bank["q1"] = np.tile([2., 1.] + [0.]*10, (32, 1))
        bank["width"] = np.tile([3., 0.] + [0.]*10, (32, 1))
        bank["width_usable"] = np.ones((32, 12), bool)
        bank["width_refusal"] = np.zeros((32, 12), bool)
        bank["distance"] = bank["width"].copy()
        bank["support_class"] = np.tile(["common_on_grid"]*4+["rare_on_grid"]*4+["unsupported"]*4, (32, 1))
        truth = np.tile([0., 3.] + [-1.]*10, (32, 1)); truth[:, 6] = -2.
        scores = diagnostic.score_arrays(bank, {"coefficient": 1.})
        # Bootstrap mechanics have a separate test; this case checks exact metrics.
        with mock.patch.object(diagnostic, "bootstrap_interval", side_effect=lambda delta, contexts, random: [float(np.mean(delta))]*2):
            summary, decisions = diagnostic.analyze(bank, truth, scores)
        pooled = next(row for row in summary["groups"] if row["group"] == "pooled" and row["support"] == "all")
        raw, live = pooled["methods"]["raw_q"], pooled["methods"]["live_width"]
        self.assertEqual(pooled["eligible_states"], 32)
        self.assertEqual(pooled["bootstrap_context_time_counts"],
                         {str(context*100+time): count for context in range(4) for time, count in ((0, 3), (50, 3), (99, 2))})
        self.assertEqual(raw["mean_opportunity_loss"], 3.)
        self.assertEqual(live["mean_delta_vs_raw"], 3.)
        self.assertEqual(live["mean_opportunity_loss"], 0.)
        self.assertEqual(live["mean_advantage_vs_original_actor"], 5.)
        self.assertEqual(live["improved_states"], 32)
        self.assertEqual(live["harmed_states"], 0)
        self.assertEqual(pooled["methods"]["constant_offset"]["mean_delta_vs_raw"], 0.)
        self.assertEqual(pooled["methods"]["detached_width"]["mean_delta_vs_raw"], live["mean_delta_vs_raw"])
        live_rows = [row for row in decisions if row["group"] == "pooled" and row["support"] == "all" and row["method"] == "live_width"]
        self.assertEqual(len(live_rows), 32)
        self.assertTrue(all(row["chosen_candidate"] == 1 and row["raw_candidate"] == 0 for row in live_rows))
        # Reverse true values without altering scores: suppression must record harm.
        truth[:, 0], truth[:, 1] = 3., 0.
        with mock.patch.object(diagnostic, "bootstrap_interval", return_value=[-3., -3.]):
            reversed_summary, _ = diagnostic.analyze(bank, truth, scores)
        harmed = next(row for row in reversed_summary["groups"] if row["group"] == "pooled" and row["support"] == "all")["methods"]["live_width"]
        self.assertEqual(harmed["mean_delta_vs_raw"], -3.)
        self.assertEqual(harmed["mean_positive_harm"], 3.)
        self.assertEqual(harmed["mean_opportunity_loss"], 3.)
        self.assertEqual(harmed["harmed_states"], 32)


@unittest.skipUnless(os.environ.get("DIRECT_Q_FROZEN_TESTS") == "1", "Requires explicit post-review frozen-model validation opt-in")
class AuthenticatedFrozenReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = diagnostic.load_engine()
        before = diagnostic.parameter_hashes(cls.engine)
        cls.receipt = diagnostic.validate_truth(cls.engine)
        cls.parameters_unchanged = before == diagnostic.parameter_hashes(cls.engine)

    def test_terminal_99_matches_immediate_reward_with_declared_tolerance(self):
        self.assertEqual(self.receipt["status"], "passed")
        self.assertEqual(self.receipt["terminal99_atol"], 1e-10)
        self.assertLessEqual(self.receipt["terminal99_max_error"], self.receipt["terminal99_atol"])

    def test_policy_action_q_equals_policy_value(self):
        self.assertEqual(self.receipt["status"], "passed")
        self.assertEqual(self.receipt["replay_atol"], 1e-4)
        self.assertEqual(self.receipt["replay_rtol"], 1e-6)
        self.assertTrue(np.isfinite(self.receipt["policy_action_q_equals_v_max_error"]))
        # validate_truth enforces atol + rtol*abs(reference) elementwise.
        # The maximum absolute error alone cannot reproduce that comparison.

    def test_batched_forced_action_q_agrees_with_scalar_replay(self):
        self.assertEqual(self.receipt["status"], "passed")
        self.assertEqual(self.receipt["replay_atol"], 1e-4)
        self.assertEqual(self.receipt["replay_rtol"], 1e-6)
        self.assertTrue(np.isfinite(self.receipt["batch_scalar_max_error"]))
        self.assertEqual(self.receipt["fixtures"], 4)
        self.assertEqual(self.receipt["plant_transitions"], 608)

    def test_frozen_parameters_and_carry_unchanged(self):
        self.assertTrue(self.parameters_unchanged)


if __name__ == "__main__":
    unittest.main()
