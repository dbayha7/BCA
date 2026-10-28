"""Synthetic-only geometry tests. No real data, model or simulator imports."""
import unittest

import numpy as np

import geometry as g


def sample(n):
    obs = np.zeros((n, 14), dtype=np.float32)
    obs[:, 10] = .5
    obs[:, 13] = 1
    actions = np.zeros((n, 4), dtype=np.float32)
    return obs, actions


class GeometryTests(unittest.TestCase):
    def test_executed_boundaries_independent_interval(self):
        values = np.array([-0.8500000238418579, -0.8499999642372131,
                           -.8, -0.7500000596046448, -.75], dtype=np.float64)
        obs, actions = sample(len(values))
        actions[:, 0] = values
        actual = g.region_masks(obs, actions)["in_band"]
        cast = actions[:, 0].astype(np.float64)
        expected = (cast >= -g.KNEE - .05) & (cast <= -g.KNEE + .05)
        np.testing.assert_array_equal(actual, [False, True, True, True, False])
        np.testing.assert_array_equal(actual, expected)

    def test_observed_load_summands_promoted_separately(self):
        obs, actions = sample(1)
        obs[0, 10], obs[0, 11] = np.float32(.3), np.float32(.1)
        actions[0, 0] = -.8
        r = g.region_masks(obs, actions)
        self.assertEqual(r["load"][0], float(obs[0, 10]) + float(obs[0, 11]))
        self.assertNotEqual(r["load"][0], float(np.float32(obs[0, 10] + obs[0, 11])))

    def test_load_rounding_changes_side_but_contract_is_unambiguous(self):
        obs, actions = sample(1)
        obs[0, 10], obs[0, 11] = np.float32(.4), np.float32(-1e-8)
        actions[0, 0] = -.8
        self.assertTrue(float(np.float32(obs[0, 10] + obs[0, 11])) >= .4)
        self.assertFalse(g.region_masks(obs, actions)['high'][0])
        physical = np.zeros((1, 14), np.float64)
        physical[0, 10] = .399999999
        self.assertFalse(physical[0, 10] + physical[0, 11] >= .4)
        self.assertTrue(g.region_masks(physical, actions)['high'][0])

    def test_mathematical_edges_are_cast_before_classification(self):
        obs, _ = sample(2)
        actions = np.zeros((2, 4), np.float64)
        actions[:, 0] = [-g.KNEE-.05, -g.KNEE+.05]
        np.testing.assert_array_equal(g.region_masks(obs, actions)['in_band'], [False, False])

    def test_low_load_retains_same_band(self):
        obs, actions = sample(2)
        obs[1, 10] = .3
        actions[:, 0] = -.8
        r = g.region_masks(obs, actions)
        np.testing.assert_array_equal(r["in_band"], [True, True])
        np.testing.assert_array_equal(r["hole"], [True, False])

    def test_matched_control_strata_and_no_mutation(self):
        obs, actions = sample(40)
        obs[::2, 10] = .3
        actions[::3, 0] = -.8
        before = actions.copy()
        times = np.tile(np.arange(20), 2)
        masks, receipt = g.control_masks(obs, actions, times, time_bin=10,
                                        episode_ids=np.repeat([10, 20], 20), match_within_episode=True)
        self.assertTrue(all(s["hole_retained"] == s["random_retained"] for s in receipt["strata"]))
        self.assertEqual(receipt["hole_retained_forbidden_labels"], 0)
        repeated, _ = g.control_masks(obs, actions, times, time_bin=10,
                                     episode_ids=np.repeat([10, 20], 20), match_within_episode=True)
        np.testing.assert_array_equal(masks["random"], repeated["random"])
        np.testing.assert_array_equal(actions, before)

    def test_population_std_constant_replacement_and_rms(self):
        obs, _ = sample(4)
        obs[:, 0] = [-1, -1, 1, 1]
        metric = g.fit_support_metric(obs)
        self.assertEqual(metric["scale"][0], 1)
        self.assertEqual(metric["scale"][1], 1)
        obs2, act2 = sample(2)
        obs2[:, 0] = [1, 2]
        anchor, _ = sample(1)
        identity = {"mean": np.zeros(13), "scale": np.ones(13)}
        masks = {k: np.ones(2, dtype=bool) for k in g.CONDITIONS}
        report = g.support_report(obs2, act2, np.array([1, 2]), np.array([0, 0]),
                                  anchor, np.array([-1]), np.array([0]), masks, identity)
        self.assertEqual(report["anchors"][0]["neighbor_source_row_ids"], [0])
        self.assertAlmostEqual(report["anchors"][0]["neighbor_distances"][0], 1 / np.sqrt(13))

    def test_whole_episode_exclusion_fixed_neighbors_and_ties(self):
        obs, actions = sample(6)
        actions[:, 0] = [-.8, -.8, -.9, -.7, -.8, 0]
        episodes = np.array([11, 11, 12, 13, 14, 15])
        source = np.array([90, 91, 50, 40, 30, 20])
        masks, _ = g.control_masks(obs, actions, np.zeros(6, dtype=int), episode_ids=episodes)
        anchor = obs[:1]
        report = g.support_report(obs, actions, episodes, np.zeros(6, dtype=int),
                                  anchor, np.array([11]), np.array([0]), masks,
                                  {"mean": np.zeros(13), "scale": np.ones(13)},
                                  source_row_ids=source, row_cap=2)
        row = report["anchors"][0]
        self.assertEqual(row["neighbor_source_row_ids"], [20, 30])
        self.assertEqual(row["counts"]["full"]["episodes"], 2)
        self.assertEqual(row["counts"]["hole"]["rows"], 1)  # No refill from IDs40/50.
        self.assertEqual(row["counts"]["hole"]["hole_rows"], 0)

    def test_cross_load_neighbors_are_reported_not_primary(self):
        obs, actions = sample(3)
        obs[1:, 10] = .39
        obs[0, 10] = .41
        actions[:, 0] = -.8
        masks, _ = g.control_masks(obs, actions, np.zeros(3, dtype=int), episode_ids=np.array([1, 2, 3]))
        report = g.support_report(obs, actions, np.array([1, 2, 3]), np.zeros(3, dtype=int),
                                  obs[:1], np.array([-1]), np.array([0]), masks,
                                  {"mean": np.zeros(13), "scale": np.ones(13)})
        row = report["anchors"][0]
        self.assertEqual(row["counts"]["full"]["rows"], 1)
        self.assertEqual(row["cross_load_counts"]["rows"], 2)
        self.assertEqual(row["cross_load_counts"]["in_band_rows"], 2)

    def test_distinct_episode_counts_do_not_equal_rows(self):
        obs, actions = sample(8)
        actions[:, 0] = -.9
        masks, _ = g.control_masks(obs, actions, np.zeros(8, dtype=int), episode_ids=np.ones(8, dtype=int))
        report = g.support_report(obs, actions, np.ones(8, dtype=int), np.zeros(8, dtype=int),
                                  obs[:1], np.array([-1]), np.array([0]), masks,
                                  {"mean": np.zeros(13), "scale": np.ones(13)})
        counts = report["anchors"][0]["counts"]["full"]
        self.assertEqual(counts["rows"], 8)
        self.assertEqual(counts["episodes"], 1)
        self.assertEqual(counts["left_alternative_episodes"], 1)

    def test_time_bins_do_not_wrap(self):
        obs, actions = sample(3)
        masks = {key: np.ones(3, dtype=bool) for key in g.CONDITIONS}
        report = g.support_report(obs, actions, np.array([1, 2, 3]), np.array([0, 9, 10]),
                                  obs[:1], np.array([-1]), np.array([0]), masks,
                                  {"mean": np.zeros(13), "scale": np.ones(13)})
        self.assertEqual(report["anchors"][0]["counts"]["full"]["rows"], 2)

    def test_calibration_preserves_original_when_eligible_and_replaces_hole(self):
        obs, actions = sample(12)
        ids = np.array([3, 1, 2])
        original_times = np.random.Generator(np.random.PCG64(g.CALIBRATION_SEED)).integers(0, 4, size=3)
        first_original = 4 + original_times[0]
        actions[first_original, 0] = -.8
        p = g.calibration_plan(obs, actions, ids, horizon=4)
        self.assertEqual(p["status"], "eligible_plan_only")
        self.assertEqual(p["calibration_rows_planned"], 3)
        self.assertEqual(p["hole_rows_used"], 0)
        selected = np.array(p["selected_rows"])
        self.assertFalse(np.any(g.region_masks(obs, actions)["hole"][selected]))
        np.testing.assert_array_equal(np.array([3, 1, 2])[selected // 4], [1, 2, 3])
        self.assertNotEqual(p["selected_rows"][0], first_original)
        self.assertEqual(p["selected_rows"][1:], p["original_selected_rows"][1:])
        self.assertEqual(p["original_rows_inside_hole"], 1)
        self.assertEqual(p["replacement_episode_ids"], [1])
        self.assertEqual(p["calibration_updates"], 0)

    def test_empty_calibration_episode_fails_without_backfill(self):
        obs, actions = sample(4)
        actions[:2, 0] = -.8
        p = g.calibration_plan(obs, actions, np.array([1, 2]), horizon=2)
        self.assertEqual(p["status"], "failed_empty_eligible_episode")
        self.assertEqual(p["empty_eligible_episode_ids"], [1])
        self.assertEqual(p["calibration_rows_planned"], 1)

    def test_invalid_inputs_rejected(self):
        obs, actions = sample(1)
        actions[0, 0] = np.nan
        with self.assertRaises(ValueError):
            g.region_masks(obs, actions)
        with self.assertRaises(ValueError):
            g.control_masks(obs, np.zeros((1, 4)), np.array([100]))
        with self.assertRaises(ValueError):
            g.fit_support_metric(obs[:, :13])


if __name__ == "__main__":
    unittest.main()
