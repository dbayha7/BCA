"""Synthetic-only geometry tests for the zero-slab gate. No real data, model or simulator imports.

Region: |a0 - 0| <= 0.1 at observed load >= 0.4. Default synthetic command is 0.5, outside the band and both
adjacent bands. In-band examples use 0.0; adjacent-band examples use -0.2 and +0.2.
"""
import json
from pathlib import Path
import unittest

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
IN_BAND, LEFT, RIGHT, FAR = 0.0, -0.2, 0.2, 0.5


def sample(n):
    obs = np.zeros((n, 14), dtype=np.float32)
    obs[:, 10] = .5
    obs[:, 13] = 1
    actions = np.zeros((n, 4), dtype=np.float32)
    actions[:, 0] = FAR
    return obs, actions


class GeometryTests(unittest.TestCase):
    def test_constants_match_contract_and_stay_below_knee(self):
        c = json.loads((HERE / "contract.json").read_text(encoding="utf-8"))
        self.assertEqual(g.CENTER, c["region"]["center"])
        self.assertEqual(g.HALF_WIDTH, c["region"]["half_width"])
        self.assertEqual(g.LOAD_CUTOFF, c["region"]["high_load"])
        self.assertEqual(g.ROW_CAP, c["support"]["max_original_neighbors"])
        self.assertEqual(g.REFERENCE_QUANTILE, c["support"]["reference_quantile"])
        self.assertEqual(list(g.LOAD_DEPTH_EDGES), c["global_coverage"]["load_depth_edges"])
        self.assertEqual(g.CONTROL_SEED, c["controls"]["random_seed"])
        self.assertEqual(g.CALIBRATION_REPLACEMENT_SEED, c["calibration"]["replacement_seed"])
        self.assertLess(abs(g.CENTER) + 3 * g.HALF_WIDTH, g.KNEE)

    def test_executed_boundaries_independent_interval(self):
        values = np.array([np.float32(-0.1), np.nextafter(np.float32(-0.1), np.float32(0)), 0.0,
                           np.nextafter(np.float32(0.1), np.float32(0)), np.float32(0.1)], dtype=np.float64)
        obs, actions = sample(len(values))
        actions[:, 0] = values
        actual = g.region_masks(obs, actions)["in_band"]
        cast = actions[:, 0].astype(np.float64)
        expected = (cast >= g.CENTER - g.HALF_WIDTH) & (cast <= g.CENTER + g.HALF_WIDTH)
        np.testing.assert_array_equal(actual, [False, True, True, True, False])
        np.testing.assert_array_equal(actual, expected)

    def test_adjacent_bands_and_far_commands(self):
        obs, actions = sample(6)
        actions[:, 0] = [-0.3, -0.2, np.nextafter(np.float32(-0.1), np.float32(-1)), 0.2, 0.3, 0.5]
        r = g.region_masks(obs, actions)
        # float32(-0.3) = -0.30000001192 is just outside [-3h, -h); float32(0.3) = 0.30000001192 just outside (h, 3h].
        np.testing.assert_array_equal(r["left_alternative"], [False, True, True, False, False, False])
        np.testing.assert_array_equal(r["right_alternative"], [False, False, False, True, False, False])
        np.testing.assert_array_equal(r["in_band"], [False] * 6)

    def test_observed_load_summands_promoted_separately(self):
        obs, actions = sample(1)
        obs[0, 10], obs[0, 11] = np.float32(.3), np.float32(.1)
        actions[0, 0] = IN_BAND
        r = g.region_masks(obs, actions)
        self.assertEqual(r["load"][0], float(obs[0, 10]) + float(obs[0, 11]))
        self.assertNotEqual(r["load"][0], float(np.float32(obs[0, 10] + obs[0, 11])))

    def test_load_rounding_changes_side_but_contract_is_unambiguous(self):
        obs, actions = sample(1)
        obs[0, 10], obs[0, 11] = np.float32(.4), np.float32(-1e-8)
        actions[0, 0] = IN_BAND
        self.assertTrue(float(np.float32(obs[0, 10] + obs[0, 11])) >= .4)
        self.assertFalse(g.region_masks(obs, actions)['high'][0])
        physical = np.zeros((1, 14), np.float64)
        physical[0, 10] = .399999999
        self.assertFalse(physical[0, 10] + physical[0, 11] >= .4)
        self.assertTrue(g.region_masks(physical, actions)['high'][0])

    def test_mathematical_edges_are_cast_before_classification(self):
        obs, _ = sample(2)
        actions = np.zeros((2, 4), np.float64)
        actions[:, 0] = [g.CENTER - g.HALF_WIDTH, g.CENTER + g.HALF_WIDTH]
        np.testing.assert_array_equal(g.region_masks(obs, actions)['in_band'], [False, False])

    def test_low_load_retains_same_band(self):
        obs, actions = sample(2)
        obs[1, 10] = .3
        actions[:, 0] = IN_BAND
        r = g.region_masks(obs, actions)
        np.testing.assert_array_equal(r["in_band"], [True, True])
        np.testing.assert_array_equal(r["hole"], [True, False])

    def test_matched_control_strata_and_no_mutation(self):
        obs, actions = sample(40)
        obs[::2, 10] = .3
        actions[::3, 0] = IN_BAND
        before = actions.copy()
        times = np.tile(np.arange(20), 2)
        masks, receipt = g.control_masks(obs, actions, times, time_bin=10,
                                        episode_ids=np.repeat([10, 20], 20), match_within_episode=True)
        self.assertTrue(all(s["hole_retained"] == s["random_retained"] for s in receipt["strata"]))
        self.assertEqual(receipt["hole_retained_forbidden_labels"], 0)
        self.assertEqual(receipt["seed"], 2026102901)
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
                                  anchor, np.array([-1]), np.array([0]), masks, identity, row_cap=1)
        self.assertEqual(report["anchors"][0]["neighbor_source_row_ids"], [0])
        self.assertAlmostEqual(report["anchors"][0]["neighbor_distances"][0], 1 / np.sqrt(13))
        self.assertAlmostEqual(report["anchors"][0]["distance_to_cap_neighbor"], 1 / np.sqrt(13))

    def test_no_absolute_radius_neighbors_are_nearest_rows(self):
        obs, actions = sample(3)
        obs[:, 0] = [5, 6, 7]  # all far from the anchor at 0
        masks = {k: np.ones(3, dtype=bool) for k in g.CONDITIONS}
        report = g.support_report(obs, actions, np.array([1, 2, 3]), np.zeros(3, dtype=int),
                                  sample(1)[0], np.array([-1]), np.array([0]), masks,
                                  {"mean": np.zeros(13), "scale": np.ones(13)}, row_cap=2)
        row = report["anchors"][0]
        self.assertEqual(row["neighbor_source_row_ids"], [0, 1])
        self.assertAlmostEqual(row["distance_to_cap_neighbor"], 6 / np.sqrt(13))
        self.assertNotIn("radius", report)

    def test_whole_episode_exclusion_fixed_neighbors_and_ties(self):
        obs, actions = sample(6)
        actions[:, 0] = [IN_BAND, IN_BAND, LEFT, RIGHT, IN_BAND, FAR]
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
        actions[:, 0] = IN_BAND
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
        actions[:, 0] = LEFT
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

    def test_reference_quantile_uses_training_rows_only_and_excludes_own_episode(self):
        # Eight training episodes; at time 0 each has one row on the high side at position x_e; a second
        # same-bin row at time 1 sits at the same position. Own-episode exclusion means a reference row
        # cannot use its own time-1 twin (distance 0) as a neighbor.
        positions = np.arange(8, dtype=np.float32)
        obs, actions = sample(16)
        obs[:8, 0] = positions
        obs[8:, 0] = positions
        episodes = np.concatenate([np.arange(8), np.arange(8)]).astype(np.int64)
        times = np.concatenate([np.zeros(8, int), np.ones(8, int)])
        z = obs.astype(np.float64)[:, :13]
        high = g.region_masks(obs, actions)["high"]
        thresholds = g.reference_cap_distances(z, episodes, times, high, np.array([0]), row_cap=2, quantile=0.5)
        entry = thresholds[(0, True)]
        self.assertEqual(entry["reference_rows"], 8)
        # Each reference row's 2nd-nearest other-episode row is 1 unit away (both neighbors at +-1, or
        # one neighbor at 1 counted twice from times 0 and 1), i.e. RMS 1/sqrt(13).
        self.assertAlmostEqual(entry["threshold"], 1 / np.sqrt(13))
        self.assertEqual(thresholds[(0, False)]["threshold"], None)  # no low-load reference rows

    def test_locality_pass_and_fail_against_reference(self):
        positions = np.arange(8, dtype=np.float32)
        obs, actions = sample(16)
        obs[:8, 0] = positions
        obs[8:, 0] = positions
        episodes = np.concatenate([np.arange(8), np.arange(8)]).astype(np.int64)
        times = np.concatenate([np.zeros(8, int), np.ones(8, int)])
        masks = {k: np.ones(16, dtype=bool) for k in g.CONDITIONS}
        anchors, _ = sample(2)
        anchors[0, 0] = 3.5    # inside the cloud: 2nd neighbor at 0.5
        anchors[1, 0] = 40.0   # far outside: 2nd neighbor far away
        report = g.support_report(obs, actions, episodes, times, anchors, np.array([-1, -2]), np.array([0, 0]),
                                  masks, {"mean": np.zeros(13), "scale": np.ones(13)}, row_cap=2,
                                  reference_quantile=0.5)
        first, second = report["anchors"]
        self.assertTrue(first["locality_passed"])
        self.assertFalse(second["locality_passed"])
        self.assertEqual(first["reference_cap_distance_threshold"], second["reference_cap_distance_threshold"])
        self.assertEqual(report["reference_quantile"], 0.5)

    def test_missing_reference_threshold_fails_locality(self):
        obs, actions = sample(3)
        masks = {k: np.ones(3, dtype=bool) for k in g.CONDITIONS}
        # Anchor time 5 has no training rows at exactly time 5, so no reference threshold exists.
        report = g.support_report(obs, actions, np.array([1, 2, 3]), np.zeros(3, dtype=int),
                                  obs[:1], np.array([-1]), np.array([5]), masks,
                                  {"mean": np.zeros(13), "scale": np.ones(13)}, row_cap=2)
        row = report["anchors"][0]
        self.assertIsNone(row["reference_cap_distance_threshold"])
        self.assertFalse(row["locality_passed"])

    def test_calibration_preserves_original_when_eligible_and_replaces_hole(self):
        obs, actions = sample(12)
        ids = np.array([3, 1, 2])
        original_times = np.random.Generator(np.random.PCG64(g.CALIBRATION_SEED)).integers(0, 4, size=3)
        first_original = 4 + original_times[0]
        actions[first_original, 0] = IN_BAND
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
        self.assertEqual(p["replacement_seed"], 2026102902)
        self.assertEqual(p["calibration_updates"], 0)

    def test_empty_calibration_episode_fails_without_backfill(self):
        obs, actions = sample(4)
        actions[:2, 0] = IN_BAND
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
