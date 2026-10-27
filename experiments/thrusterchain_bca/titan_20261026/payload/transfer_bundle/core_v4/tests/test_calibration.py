import json
import math
import unittest

import numpy as np

from thrusterchain.calibration import (final_terminal_calibration, nine_cell_smoke,
                                      summarize_intervals, terminal_targets,
                                      weighted_conformal_radius, weighted_conformal_quantile,
                                      fit_terminal_bootstrap, final_terminal_lower_calibration,
                                      TerminalLowerCalibration)
from thrusterchain.coverage import CoverageLaw
from thrusterchain.environment import dynamics, ThrusterChainEnv


class CalibrationTests(unittest.TestCase):
    def test_terminal_target_is_reward_only_at_final_finite_decision(self):
        sample = CoverageLaw().sample(40, 7)
        actions = sample.actions*2
        executed = actions.astype(np.float32).astype(np.float64)
        expected = -(np.square(sample.states[:, :10]).sum(axis=1)
                     + .3*np.square(np.clip(executed, -1, 1)).sum(axis=1))
        np.testing.assert_allclose(terminal_targets(sample.states, actions), expected, atol=1e-14)
        np.testing.assert_array_equal(terminal_targets(sample.states, actions), dynamics(sample.states, executed)[1])
        for kw in ({"time_index": 98}, {"horizon": 99}, {"task_semantics": "continuing_time_limit"}):
            with self.assertRaises(ValueError):
                terminal_targets(sample.states, actions, **kw)

    def test_terminal_labels_match_environment_execution_map(self):
        sample = CoverageLaw().sample(10, 77)
        env = ThrusterChainEnv()
        labels = terminal_targets(sample.states, sample.actions)
        for state, action, label in zip(sample.states, sample.actions, labels):
            env.reset(options={"state": state})
            # Replay a designed state at the final decision, not an occupancy claim.
            env._t = 99
            _, reward, terminated, truncated, _ = env.step(action)
            self.assertEqual(label, reward)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
        overflow = sample.actions.copy()
        overflow[0, 0] = 1e300
        with self.assertRaisesRegex(ValueError, "float32"):
            terminal_targets(sample.states, overflow)

    def test_exact_internal_quantile_endpoint(self):
        q = weighted_conformal_quantile([0, 1, 2, 3], [4, 7, 7, 1], 1, alpha=.1)
        self.assertEqual(q.radius, 2)
        # A genuinely positive extra test mass must not be accepted by tolerance.
        q = weighted_conformal_quantile([0, 1, 2, 3], [4, 7, 7, 1], np.nextafter(1., np.inf), alpha=.1)
        self.assertEqual(q.radius, 3)

    def test_equal_weight_rank_and_test_atom(self):
        self.assertEqual(weighted_conformal_radius(np.arange(1, 10), np.ones(9), 1, alpha=.1).radius, 9)
        self.assertTrue(math.isinf(weighted_conformal_radius(np.arange(1, 9), np.ones(8), 1, alpha=.1).radius))
        q = weighted_conformal_radius([1, 2], [1, 3], 1, alpha=.4)
        self.assertEqual(q.radius, 2)
        self.assertAlmostEqual(q.test_atom_mass, .2)
        self.assertTrue(math.isinf(weighted_conformal_radius([1, 2], [1, 3], 100, alpha=.4).radius))

    def test_common_scaling_zero_weights_and_empty_calibration(self):
        a = weighted_conformal_radius([1, 2, 8], [1, 4, 0], 2, alpha=.4)
        b = weighted_conformal_radius([1, 2, 8], [1e250, 4e250, 0], 2e250, alpha=.4)
        self.assertEqual(a.radius, b.radius)
        self.assertAlmostEqual(a.calibration_ess, b.calibration_ess)
        self.assertEqual(a.radius, 2)
        self.assertTrue(math.isinf(weighted_conformal_radius([], [], 1).radius))
        huge_query = weighted_conformal_radius(np.ones(100), np.ones(100), 1e250, min_ess=20)
        self.assertAlmostEqual(huge_query.calibration_ess, 100)
        self.assertTrue(huge_query.ess_gate_passed)
        self.assertTrue(math.isinf(huge_query.radius))
        for weights, test in (([1, np.inf], 1), ([1, -1], 1), ([0, 0], 0), ([1, 1], np.inf)):
            with self.assertRaises(ValueError):
                weighted_conformal_radius([1, 2], weights, test)

    def test_ess_does_not_change_rank_or_imply_support(self):
        q = weighted_conformal_radius([1, 2, 3], [100, 0, 0], 1, alpha=.1, min_ess=2)
        self.assertEqual(q.status, "finite")
        self.assertEqual(q.radius, 1)
        self.assertFalse(q.ess_gate_passed)
        self.assertFalse(q.usable)

    def test_signed_quantile_and_one_sided_zero_clamp(self):
        q = weighted_conformal_quantile([-3, -2, -1], [1, 1, 1], 1, alpha=.5)
        self.assertEqual(q.radius, -2)
        sample = CoverageLaw().sample(40, 55, group_prefix="fit")
        model = fit_terminal_bootstrap(sample.states, sample.actions, sample.group_ids, members=4, seed=7)
        final = CoverageLaw().sample(20, 66, group_prefix="final")
        calibrated = final_terminal_lower_calibration(model, final.states, final.actions, final.group_ids,
                                                      source_law=CoverageLaw(), prior_groups={"fit": sample.group_ids},
                                                      predictor_frozen_before_final=True, target_frozen_before_final=True)
        # Exact synthetic score fixture verifies max(0,q), separately from fit quality.
        fixture = TerminalLowerCalibration(calibrated.base, model, np.full(20, -1.), calibrated.calibration_sigma)
        bound = fixture.bounds(final.states, final.actions)
        np.testing.assert_array_equal(bound["width"], 0)
        np.testing.assert_array_equal(bound["lower"], bound["center"])

    def test_bootstrap_root_weights_are_independent_normalized_and_grouped(self):
        sample = CoverageLaw().sample(30, 5)
        model = fit_terminal_bootstrap(sample.states, sample.actions, sample.group_ids, members=8, seed=19)
        np.testing.assert_allclose(model.root_weights.sum(axis=1), 1)
        self.assertFalse(np.array_equal(model.root_weights[0], model.root_weights[1]))
        self.assertTrue(model.diagnostics(sample.states, sample.actions)["independent_member_draws"])
        doubled = fit_terminal_bootstrap(np.repeat(sample.states, 2, axis=0), np.repeat(sample.actions, 2, axis=0),
                                         np.repeat(sample.group_ids, 2), members=8, seed=19)
        np.testing.assert_allclose(model.coefficients, doubled.coefficients, atol=1e-11)
        collapsed = fit_terminal_bootstrap(sample.states, sample.actions, sample.group_ids, members=8,
                                           root_weights=np.full((8, 30), 1/30), sigma_floor=.01)
        self.assertFalse(collapsed.diagnostics(sample.states, sample.actions)["independent_member_draws"])
        np.testing.assert_allclose(collapsed.predict_distribution(sample.states, sample.actions)[1], .01)

    def test_bootstrap_known_fit_roots_cannot_be_omitted_from_leakage_check(self):
        sample = CoverageLaw().sample(30, 5, group_prefix="fit")
        model = fit_terminal_bootstrap(sample.states, sample.actions, sample.group_ids, members=4)
        with self.assertRaisesRegex(ValueError, "leakage"):
            final_terminal_lower_calibration(model, sample.states, sample.actions, sample.group_ids,
                                              source_law=CoverageLaw(), claim_guarantee=True,
                                              independent_groups=True, predictor_frozen_before_final=True,
                                              target_frozen_before_final=True)

    def test_frozen_calibration_arrays_cannot_be_made_writable(self):
        fit = CoverageLaw().sample(30, 5, group_prefix="fit")
        final = CoverageLaw().sample(20, 6, group_prefix="final")
        model = fit_terminal_bootstrap(fit.states, fit.actions, fit.group_ids, members=4)
        calibrated = final_terminal_lower_calibration(model, final.states, final.actions, final.group_ids,
                                                      source_law=CoverageLaw(), predictor_frozen_before_final=True,
                                                      target_frozen_before_final=True)
        for array in (model.coefficients, model.root_weights, model.root_ids, calibrated.signed_scores,
                      calibrated.calibration_sigma, calibrated.base.scores, calibrated.base.weights,
                      calibrated.base.provenance["calibration_predictions"]):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(TypeError):
            calibrated.base.provenance["predictor"] = None
        with self.assertRaises(TypeError):
            calibrated.base.support["absolute_continuity"] = False
        q = weighted_conformal_radius(np.ones(100), np.ones(100), 1000, min_ess=20)
        self.assertEqual(q.status, "infinite_rank")
        self.assertTrue(q.ess_gate_passed)
        self.assertFalse(q.usable)

    def _calibrate(self, n=20, source=None, **kwargs):
        source = source or CoverageLaw()
        sample = source.sample(n, 2, group_prefix="final")
        predictor = lambda states, actions: np.zeros(len(states))
        defaults = dict(source_law=source, predictor_frozen_before_final=True,
                        target_frozen_before_final=True, independent_groups=True,
                        prior_groups={"fit": ["fit:0"], "selection": ["select:0"], "tuning_calibration": ["tune:0"]})
        defaults.update(kwargs)
        fitted = final_terminal_calibration(predictor, sample.states, sample.actions, sample.group_ids, **defaults)
        return fitted, predictor, sample

    def test_infinite_band_is_vacuous_not_refusal(self):
        fitted, predictor, sample = self._calibrate(n=3)
        intervals = fitted.intervals(predictor, sample.states, sample.actions)
        self.assertTrue(np.isneginf(intervals["lower"]).all())
        self.assertTrue(np.isposinf(intervals["upper"]).all())
        report = summarize_intervals(intervals, terminal_targets(sample.states, sample.actions))
        self.assertEqual(report["infinite_intervals"], 3)
        self.assertEqual(report["support_refusals"], 0)
        self.assertEqual(report["coverage_including_infinite"], 1)
        self.assertEqual(report["usable_intervals"], 0)

    def test_missing_support_is_refusal_even_for_locally_supported_queries(self):
        fitted, predictor, sample = self._calibrate(source=CoverageLaw("absent", "broad"))
        intervals = fitted.intervals(predictor, sample.states, sample.actions)
        self.assertTrue(np.isnan(intervals["lower"]).all())
        report = summarize_intervals(intervals, terminal_targets(sample.states, sample.actions))
        self.assertEqual(report["support_refusals"], len(sample.states))
        self.assertEqual(report["infinite_intervals"], 0)
        self.assertIsNone(report["coverage_including_infinite"])
        outside = sample.states.copy()
        outside[0, 0] = 2.
        outside_intervals = fitted.intervals(predictor, outside, sample.actions)
        self.assertEqual(outside_intervals["status"][0], "outside_target_law")

    def test_fresh_calibration_and_no_group_leakage(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            self._calibrate(prior_groups={"tuning_calibration": ["final:0"]})
        with self.assertRaisesRegex(ValueError, "freeze"):
            self._calibrate(predictor_frozen_before_final=False)
        sample = CoverageLaw().sample(20, 1)
        with self.assertRaisesRegex(ValueError, "one row per independent group"):
            final_terminal_calibration(lambda x, a: np.zeros(len(x)), sample.states, sample.actions,
                                       np.zeros(20), source_law=CoverageLaw(),
                                       predictor_frozen_before_final=True, target_frozen_before_final=True)
        for kw in ({"independent_groups": False}, {"weight_accounting": "estimated"},
                   {"source": CoverageLaw("absent", "broad")}):
            with self.assertRaisesRegex(ValueError, "guarantee"):
                self._calibrate(claim_guarantee=True, **kw)
        fitted, _, _ = self._calibrate(claim_guarantee=True)
        self.assertTrue(fitted.guarantee_claimed)
        self.assertFalse(fitted.provenance["final_calibration_used_for_selection"])

    def test_predictor_cannot_be_replaced_or_mutated_after_calibration(self):
        fitted, predictor, sample = self._calibrate()
        with self.assertRaisesRegex(ValueError, "same frozen predictor"):
            fitted.intervals(lambda x, a: np.ones(len(x)), sample.states, sample.actions)
        parameter = [0.]
        predictor = lambda x, a: np.full(len(x), parameter[0])
        fitted = final_terminal_calibration(predictor, sample.states, sample.actions, sample.group_ids,
                                            source_law=CoverageLaw(), predictor_frozen_before_final=True,
                                            target_frozen_before_final=True)
        parameter[0] = 1.
        with self.assertRaisesRegex(ValueError, "changed"):
            fitted.intervals(predictor, sample.states, sample.actions)

    def test_tiny_nine_cell_smoke_has_honest_statuses(self):
        report = nine_cell_smoke(n_calibration=24, n_evaluation=16, seed=3, repetitions=2, members=4)
        self.assertEqual(len(report["cells"]), 9)
        self.assertIn("not nine trained benchmark", report["kind"])
        self.assertFalse(report["guarantee_claimed"])
        self.assertFalse(report["closed_loop_occupancy_claim"])
        self.assertEqual(sum(c["diagnostic"]["support_refusals"] == 16 for c in report["cells"]), 5)
        self.assertTrue(all(c["fresh_final_after_selection"] for c in report["cells"]))
        for cell in report["cells"]:
            for repetition in cell["repetitions"]:
                self.assertEqual(len(repetition["arms"]), 4)
                self.assertEqual(len(repetition["model"]["coefficients_sha256"]), 64)
                final_raw = repetition["raw"]["final_calibration"]
                np.testing.assert_allclose(final_raw["signed_score"],
                                           (np.array(final_raw["qmean"])-final_raw["true_terminal_q"])/final_raw["sigma"])
                evaluation_raw = repetition["raw"]["evaluation"]
                self.assertEqual(evaluation_raw["time_index"], [99]*16)
                for weighting, bound in evaluation_raw["bounds_by_weighting"].items():
                    coverage = [v for v in bound["covered_when_issued"] if v is not None]
                    summary = repetition["arms"][weighting+"_standard"]["query_coverage"]
                    self.assertEqual(len(coverage), summary["issued_intervals"])
                    self.assertEqual(float(np.mean(coverage)) if coverage else None, summary["coverage_including_infinite"])
                for name, arm in repetition["arms"].items():
                    bank_raw = repetition["raw"]["paired_action_bank"]
                    raw_arm = bank_raw["arms"][name]
                    np.testing.assert_array_equal(raw_arm["selected_reward"],
                                                  np.array(bank_raw["true_reward"])[np.arange(16), raw_arm["selected_index"]])
                    self.assertEqual(np.mean(raw_arm["selected_reward"]), arm["action_selection"]["full_population_mean_reward"])
                    if cell["support"]["status"] == "unsupported":
                        self.assertEqual(arm["query_coverage"]["support_refusals"], 16)
                        self.assertEqual(arm["action_selection"]["fallback_executions"], 16)
                        if name.endswith("most_caution"):
                            self.assertAlmostEqual(arm["action_selection"]["full_population_mean_regret_to_zero"], 0.)
            self.assertEqual(cell["repetition_summary"]["weighted_standard"]["independent_repetitions"], 2)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
