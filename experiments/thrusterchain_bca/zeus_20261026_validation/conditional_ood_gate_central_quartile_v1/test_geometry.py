"""Synthetic arrays only; no dataset, model or simulator dispatch."""
import hashlib
import time
import unittest

import numpy as np

import geometry as g


def fixture():
    n = 80
    obs = np.zeros((n, 14), np.float32)
    obs[:, 0] = np.tile(np.linspace(-1, 1, n // 2), 2)
    obs[n // 2:, 10] = .6
    obs[:, -1] = 1
    actions = np.zeros((n, 4), np.float32)
    actions[:, 0] = np.resize([0., -.5, .5, .9], n)
    episodes = np.arange(n, dtype=np.int64) + 1000
    times = np.zeros(n, np.int64)
    rows = np.arange(n, dtype=np.int64) + 9000
    masks, _ = g.control_masks(obs, actions, times, episode_ids=episodes)
    metric = g.fit_support_metric(obs)
    return obs, actions, episodes, times, rows, masks, metric


def report(data=None, anchors=None, anchor_ids=None, **kwargs):
    obs, actions, episodes, times, rows, masks, metric = fixture() if data is None else data
    if anchors is None:
        anchors = obs[[20, 60]].copy()
    if anchor_ids is None:
        anchor_ids = np.arange(len(anchors), dtype=np.int64) + 10000
    settings = {'row_cap': 8, 'time_bin': 100, 'reference_per_stratum': 8}
    settings.update(kwargs)
    return g.support_report(obs, actions, episodes, times, anchors, anchor_ids,
                            np.zeros(len(anchors), np.int64), masks, metric,
                            source_row_ids=rows, **settings)


class GeometryTests(unittest.TestCase):
    def test_executed_boundaries_and_load_promotion(self):
        commands = [-.75, -.5, np.nextafter(np.float32(-.25), np.float32(-1)),
                    -.25, 0., .25, np.nextafter(np.float32(.25), np.float32(1)), .5, .75]
        obs = np.zeros((len(commands), 14), np.float32)
        obs[:, 10] = .6
        actions = np.zeros((len(commands), 4), np.float32)
        actions[:, 0] = commands
        region = g.region_masks(obs, actions)
        np.testing.assert_array_equal(region['in_band'], [False, False, False, True, True, True, False, False, False])
        np.testing.assert_array_equal(region['left_alternative'], [True, True, True, False, False, False, False, False, False])
        np.testing.assert_array_equal(region['right_alternative'], [False, False, False, False, False, False, True, True, True])
        obs[:, 10] = .2; obs[:, 11] = .2
        self.assertTrue(g.region_masks(obs, actions)['high'].all())
        obs[:, 10] = .1; obs[:, 11] = .1
        lower = g.region_masks(obs, actions)
        self.assertFalse(lower['hole'].any())
        np.testing.assert_array_equal(lower['in_band'], region['in_band'])

    def test_random_control_count_matching_and_calibration_empty_stop(self):
        obs, actions, episodes, times, _, _, _ = fixture()
        # Multiple rows from each episode allow a nontrivial random-removal control.
        episodes = np.repeat(np.arange(8, dtype=np.int64), 10)
        masks, receipt = g.control_masks(obs, actions, times, episode_ids=episodes)
        self.assertTrue(masks['full'].all())
        self.assertEqual(masks['hole'].sum(), masks['random'].sum())
        self.assertTrue(all(x['hole_retained'] == x['random_retained'] for x in receipt['strata']))
        self.assertFalse(np.any(masks['hole'] & g.region_masks(obs, actions)['hole']))
        obs = np.zeros((4, 14), np.float32); obs[:, 10] = .6
        actions = np.zeros((4, 4), np.float32)
        plan = g.calibration_plan(obs, actions, np.asarray([4, 5]), horizon=2)
        self.assertEqual(plan['status'], 'failed_empty_eligible_episode')
        self.assertEqual(plan['empty_eligible_episode_ids'], [4, 5])
        self.assertEqual(plan['calibration_updates'], 0)

    def test_reference_selection_exact_hash_and_distinct_episodes(self):
        data = list(fixture())
        data[2] = np.repeat(np.arange(40, dtype=np.int64), 2)
        result = report(data=tuple(data))
        rows, episodes = data[4], data[2]
        for reference in result['reference_strata']:
            indices = range(40, 80) if reference['high_load'] else range(40)
            indices = sorted(indices, key=lambda j: (
                hashlib.sha256(('central-quartile-reference-v1:' + str(rows[j])).encode('ascii')).digest(), int(rows[j])))
            chosen, seen = [], set()
            for j in indices:
                if episodes[j] not in seen:
                    chosen.append(j); seen.add(episodes[j])
                if len(chosen) == 8:
                    break
            self.assertEqual(reference['reference_source_row_ids'], rows[chosen].tolist())
            self.assertEqual(len(set(reference['reference_episode_ids'])), 8)
            self.assertEqual(reference['threshold'], float(np.quantile(reference['reference_kth_distances'], .95, method='higher')))

    def test_reference_threshold_independent_of_query_actions_and_retention(self):
        data = list(fixture())
        baseline = report(data=tuple(data))
        data[1] = np.full_like(data[1], .7)
        data[5] = {key: np.ones(80, bool) for key in ('full', 'hole', 'random')}
        far = data[0][[0]].copy(); far[0, 0] = 100.
        changed = report(data=tuple(data), anchors=far)
        self.assertEqual(baseline['reference_strata'], changed['reference_strata'])
        self.assertFalse(changed['anchors'][0]['locality_pass'])
        self.assertTrue(baseline['all_reference_strata_available'])

    def test_whole_episode_excluded_for_references_and_query(self):
        data = list(fixture())
        # Each episode contains two nearby states; all records from the query
        # episode must be excluded, not just the exact source row.
        data[2] = np.repeat(np.arange(40, dtype=np.int64), 2)
        data[0][:40, 0] = 0
        data[0][40:, 0] = 0
        data[6] = g.fit_support_metric(data[0])
        result = report(data=tuple(data), anchors=data[0][[0]], anchor_ids=np.asarray([0]))
        selected = result['anchors'][0]['neighbor_source_row_ids']
        self.assertFalse(set(data[4][:2]) & set(selected))
        self.assertEqual(selected, data[4][2:10].tolist())  # Distance ties resolved by source row ID.
        reference = result['reference_strata'][0]
        self.assertEqual(result['distance_accounting']['support_distance_pairs_completed'], 16*38+38)
        self.assertTrue(all(d == 0. for d in reference['reference_kth_distances']))

    def test_neighbors_fixed_before_removal_never_refilled(self):
        data = list(fixture())
        baseline = report(data=tuple(data))
        selected = baseline['anchors'][1]['neighbor_source_row_ids']
        data[5] = {k: v.copy() for k, v in data[5].items()}
        data[5]['hole'][np.isin(data[4], selected)] = False
        changed = report(data=tuple(data))
        self.assertEqual(changed['anchors'][1]['neighbor_source_row_ids'], selected)
        self.assertEqual(changed['anchors'][1]['counts']['full']['rows'], 8)
        self.assertEqual(changed['anchors'][1]['counts']['hole']['rows'], 0)

    def test_insufficient_distinct_episodes_or_neighbor_count_fails_without_fallback(self):
        data = list(fixture())
        data[2][:40] = 1  # Only one low-load episode.
        result = report(data=tuple(data))
        low = result['reference_strata'][0]
        self.assertFalse(low['available'])
        self.assertEqual(low['reference_rows'], 1)
        self.assertIsNone(low['threshold'])
        self.assertFalse(result['anchors'][0]['locality_pass'])
        result = report(row_cap=64)  # Forty rows per side cannot supply 64 neighbors.
        self.assertFalse(result['all_reference_strata_available'])
        self.assertTrue(all(x['threshold'] is None for x in result['reference_strata']))

    def test_metric_uses_training_physical_coordinates_only(self):
        obs = fixture()[0]
        metric = g.fit_support_metric(obs)
        changed = obs.copy(); changed[:, -1] = 100
        other = g.fit_support_metric(changed)
        np.testing.assert_array_equal(metric['scale'], other['scale'])
        np.testing.assert_array_equal(metric['mean'], other['mean'])
        self.assertTrue(np.all(metric['scale'] > 0))
        self.assertEqual(metric['fit_rows'], 80)

    def test_deadline_and_accounting_caps_stop_before_computation(self):
        ledger = {}
        with self.assertRaisesRegex(TimeoutError, 'deadline'):
            report(deadline=time.monotonic()-1, ledger=ledger)
        self.assertEqual(ledger['support_distance_pairs_attempted'], 0)
        ledger = {'support_distance_pairs_attempted': g.MAX_DISTANCE_PAIRS}
        with self.assertRaisesRegex(ValueError, 'distance-pair cap'):
            report(ledger=ledger)
        self.assertEqual(ledger['support_distance_pairs_attempted'], g.MAX_DISTANCE_PAIRS)
        self.assertEqual(ledger['support_reference_queries_attempted'], 0)
        ledger = {'support_reference_queries_attempted': g.MAX_REFERENCE_QUERIES}
        with self.assertRaisesRegex(ValueError, 'query cap'):
            report(ledger=ledger)
        ledger = {}
        calls = []
        result = report(ledger=ledger, progress=lambda x: calls.append(x))
        self.assertEqual(ledger['support_reference_queries_completed'], 16)
        self.assertEqual(ledger['support_anchor_queries_completed'], 2)
        self.assertEqual(ledger['support_distance_pairs_completed'], 16*39+2*40)
        self.assertEqual(ledger['support_distance_pairs_attempted'], ledger['support_distance_pairs_completed'])
        self.assertEqual(len(calls), 18)
        self.assertEqual(result['distance_accounting'], ledger)

    def test_row_permutation_preserves_reference_and_neighbor_identities(self):
        data = list(fixture())
        baseline = report(data=tuple(data))
        anchors = data[0][[20, 60]].copy()
        order = np.arange(79, -1, -1)
        for i in range(5):
            data[i] = data[i][order]
        data[5] = {k: v[order] for k, v in data[5].items()}
        result = report(data=tuple(data), anchors=anchors)
        self.assertEqual(baseline['reference_strata'], result['reference_strata'])
        self.assertEqual(baseline['anchors'], result['anchors'])

    def test_duplicate_rows_and_invalid_parameters_rejected(self):
        data = list(fixture()); data[4][1] = data[4][0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            report(data=tuple(data))
        with self.assertRaisesRegex(ValueError, 'settings'):
            report(reference_quantile=float('nan'))
        with self.assertRaisesRegex(ValueError, 'reference cap'):
            report(time_bin=1, reference_per_stratum=32)


if __name__ == '__main__':
    unittest.main()
