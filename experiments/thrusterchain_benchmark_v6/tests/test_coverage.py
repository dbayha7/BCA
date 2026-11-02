import unittest

import numpy as np

from thrusterchain.coverage import (ACTION_CORE_VOLUME, ACTION_GAP_MASS, ACTION_GAP_VOLUME,
                                   ACTION_VOLUME, ACTION_SPLIT, STATE_SPLIT, STATE_GAP_MASS,
                                   STATE_VOLUME, CoverageLaw,
                                   assert_disjoint_groups, binomial_shortfall_probability,
                                   grouped_split, minimum_equal_weight_calibration, nine_cells,
                                   occupancy_accounting, plan_binomial_count)
from thrusterchain.environment import in_start_set


class CoverageTests(unittest.TestCase):
    def test_all_nine_laws_normalize_by_partition_integration(self):
        states = np.zeros((4, 13))
        states[:, 10] = .4
        states[:, 0] = [-.25, .25, -.25, .25]
        actions = np.zeros((4, 4))
        actions[2:, 0] = .9
        volumes = STATE_VOLUME * np.array([
            (1-STATE_GAP_MASS)*ACTION_CORE_VOLUME, STATE_GAP_MASS*ACTION_CORE_VOLUME,
            (1-STATE_GAP_MASS)*ACTION_GAP_VOLUME, STATE_GAP_MASS*ACTION_GAP_VOLUME])
        for law in nine_cells():
            self.assertAlmostEqual(float(np.dot(law.density(states, actions), volumes)), 1.0)
            self.assertAlmostEqual(sum(law.region_masses.values()), 1.0)
        self.assertEqual(len({law.name for law in nine_cells()}), 9)
        self.assertEqual(ACTION_GAP_VOLUME, 4.)

    def test_broad_is_uniform_and_ratios_integrate_on_covered_support(self):
        query = CoverageLaw().sample(500, 123)
        np.testing.assert_allclose(CoverageLaw().density(query.states, query.actions),
                                   1/(STATE_VOLUME*ACTION_VOLUME))
        np.testing.assert_allclose(CoverageLaw().density_ratio(query.states, query.actions), 1)
        states = np.zeros((4, 13))
        states[:, 10] = .4
        states[:, 0] = [-.25, .25, -.25, .25]
        actions = np.zeros((4, 4))
        actions[2:, 0] = .9
        for law in nine_cells():
            ratios = law.density_ratio(states, actions)
            masses = law.region_masses
            source_mass = np.array([masses[k] for k in ("core_core", "gap_core", "core_gap", "gap_gap")])
            finite = np.isfinite(ratios)
            integrated = np.sum(source_mass[finite]*ratios[finite])
            self.assertAlmostEqual(integrated, 1-law.support_report()["target_unsupported_mass"])
        rare = CoverageLaw("rare", "rare")
        np.testing.assert_allclose(rare.density_ratio(states, actions),
                                   [.375/(.95*.99), .375/(.05*.99), .125/(.95*.01), 250])
        self.assertAlmostEqual(rare.region_masses["gap_gap"], .0005)

    def test_samples_match_declared_regions_and_are_reset_valid(self):
        for law in nine_cells():
            sample = law.sample(5000, 25, group_prefix=law.name)
            self.assertTrue(np.all(in_start_set(sample.states)))
            np.testing.assert_array_equal(sample.state_gap, sample.states[:, 0] > STATE_SPLIT)
            np.testing.assert_array_equal(sample.action_gap, sample.actions[:, 0] > ACTION_SPLIT)
            self.assertEqual(len(np.unique(sample.group_ids)), 5000)
            self.assertTrue(np.all(law.density(sample.states, sample.actions) > 0))
            for realized, expected in ((sample.state_gap.mean(), law.state_gap_mass),
                                       (sample.action_gap.mean(), law.action_gap_mass)):
                self.assertLess(abs(realized-expected), .035)
            repeat = law.sample(5000, 25, group_prefix=law.name)
            np.testing.assert_array_equal(sample.states, repeat.states)
            np.testing.assert_array_equal(sample.actions, repeat.actions)

    def test_absent_is_structural_not_zero_realized_counts(self):
        rare = CoverageLaw("rare", "rare")
        self.assertTrue(rare.support_report()["absolute_continuity"])
        self.assertFalse(CoverageLaw("absent", "broad").support_report()["absolute_continuity"])
        report = CoverageLaw("absent", "absent").support_report()
        self.assertAlmostEqual(report["target_unsupported_mass"], 1-(1-STATE_GAP_MASS)*(1-ACTION_GAP_MASS))
        # An absent source is valid for a target confined to the same support.
        self.assertTrue(CoverageLaw("absent", "absent").support_report(CoverageLaw("absent", "absent"))["absolute_continuity"])
        self.assertEqual(sum(l.support_report()["absolute_continuity"] for l in nine_cells()), 4)
        self.assertEqual(report["target_unsupported_mass"], .625)
        tiny_target = CoverageLaw("rare", "rare", 1e-20, 1e-20)
        tiny_report = CoverageLaw("absent", "absent").support_report(tiny_target)
        self.assertFalse(tiny_report["absolute_continuity"])
        self.assertGreater(tiny_report["target_unsupported_mass"], 0)
        for bad in (1e-200, np.array(.05), np.nan, np.inf):
            with self.assertRaises(ValueError):
                CoverageLaw(state_rare_mass=bad)

    def test_boundaries_outside_domain_and_malformed_inputs(self):
        states = np.zeros((4, 13))
        states[:, 10] = .4
        states[1, 0] = np.nextafter(0., 1.)
        states[3, 0] = 2.
        actions = np.zeros((4, 4))
        actions[:, 0] = .5
        actions[2, 0] = np.nextafter(.5, 1.)
        report = CoverageLaw("absent", "absent").ratio_report(states, actions)
        self.assertEqual(report["status"].tolist(), ["supported_target", "unsupported_target",
                                                    "unsupported_target", "outside_both_laws"])
        self.assertEqual(report["ratio"][3], 0)
        self.assertTrue(np.isposinf(report["ratio"][1:3]).all())
        for bad in (np.nan, np.inf):
            malformed = states.copy()
            malformed[0, 0] = bad
            with self.assertRaises(ValueError):
                CoverageLaw().density(malformed, actions)
        with self.assertRaises(ValueError):
            CoverageLaw().density(states[:, :12], actions)
        with self.assertRaises(ValueError):
            CoverageLaw().density(states.astype(complex), actions)

    def test_group_split_keeps_episodes_intact(self):
        groups = np.repeat(np.arange(40), 7)
        splits = grouped_split(groups, seed=42)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(groups)))
        assert_disjoint_groups({name: groups[rows] for name, rows in splits.items()})
        for rows in splits.values():
            self.assertEqual(len(rows) % 7, 0)
        same = grouped_split(groups, seed=42)
        for key in splits:
            np.testing.assert_array_equal(splits[key], same[key])
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint_groups({"fit": [1, 2], "calibration": [2, 3]})
        with self.assertRaises(ValueError):
            grouped_split([1, 2])
        with self.assertRaises(ValueError):
            grouped_split(np.r_[np.arange(20, dtype=float), np.nan])
        with self.assertRaises(ValueError):
            assert_disjoint_groups({"fit": [[1, 2]]})

    def test_count_plan_is_minimal_and_absence_is_impossible(self):
        plan = plan_binomial_count(9, .01, .05)
        n = plan["required_groups"]
        self.assertLessEqual(binomial_shortfall_probability(n, .01, 9), .05)
        self.assertGreater(binomial_shortfall_probability(n-1, .01, 9), .05)
        self.assertEqual(plan_binomial_count(9, 0)["status"], "impossible_absent_support")
        self.assertEqual(plan_binomial_count(9, 1)["required_groups"], 9)
        self.assertEqual(plan_binomial_count(9, .0001, max_samples=100)["status"], "exceeds_budget")
        self.assertEqual(minimum_equal_weight_calibration(.1), 9)
        self.assertEqual(minimum_equal_weight_calibration(.05), 19)

    def test_known_actions_do_not_imply_occupancy(self):
        result = occupancy_accounting(action_policy_known=True)
        self.assertEqual(result["state_occupancy"], "unknown")
        self.assertEqual(result["joint_occupancy_ratio"], "unknown")
        self.assertFalse(result["exact_weight_guarantee"])
        estimated = occupancy_accounting(action_policy_known=True, state_density_method="estimated")
        self.assertEqual(estimated["joint_occupancy_ratio"], "estimated")
        self.assertFalse(estimated["exact_weight_guarantee"])


if __name__ == "__main__":
    unittest.main()
