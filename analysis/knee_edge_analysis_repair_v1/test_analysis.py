"""Explicit, synthetic saved-array fixtures only; no experiment imports or execution."""
import struct
import unittest

import numpy as np

import analysis_math as M


EXPECTED_BITS = {
    'edge_knee': 0xBF4CCCCD, 'edge_775': 0xBF466666, 'edge_750': 0xBF400000,
    'edge_725': 0xBF39999A, 'edge_700': 0xBF333333, 'mid_m40': 0xBECCCCCD,
    'zero': 0x00000000, 'mid_p40': 0x3ECCCCCD, 'knee': 0x3F4CCCCD,
    's05': 0xBF59999A, 's10': 0xBF666667, 's15': 0xBF733333,
    's20': 0xBF800000, 'p10': 0x3F666667,
}


def receipt_fixture(world='stall3', clipped=True):
    return {
        'evaluator': 'raw_dynamics_loop', 'gamma': 0.98, 'horizon': 100,
        'bootstrap': 'none after step 99',
        'clip': 'all four commands to +-KNEE' if clipped else
                'none (deployed command, |a| <= 1 asserted)',
        'continuation': {
            'world': world,
            'physics': {'stall': world == 'stall3', 'stall_slope': 3.0,
                        'knee': struct.unpack('>f', bytes.fromhex('3f4ccccd'))[0],
                        'gain': 0.3, 'throttle_cost': 0.3, 'wave_turn': 0.27,
                        'gamma': 0.98, 'length': 100},
        },
    }


def anchors_fixture(kc=0.10, own_gap=0.20):
    """Affine supported values with known past-knee drops, independent row offsets."""
    xs = np.array([struct.unpack('>f', b.to_bytes(4, 'big'))[0]
                   for b in EXPECTED_BITS.values()])
    baseline = 2.0 + 3.0 * xs[None, :] + np.arange(64)[:, None] / 100.0
    clipped = baseline.copy()
    for slot, drop in (('s05', 0.5), ('s10', 0.8), ('s20', 1.2)):
        clipped[:, list(EXPECTED_BITS).index(slot)] -= drop
    q1, own = clipped.copy(), clipped.copy()
    for slot in ('s05', 's10', 's20'):
        j = list(EXPECTED_BITS).index(slot)
        q1[:, j] += kc
        own[:, j] += own_gap
    return {
        'times': [t for t in (0, 25, 50, 75) for _ in range(16)],
        'gated_slots': list(EXPECTED_BITS), 'nostall_slots': list(EXPECTED_BITS),
        'q1': q1.tolist(), 'qpi_clip': clipped.tolist(),
        'qpi_unclipped': own.tolist(), 'qpi_nostall': baseline.tolist(),
        'valid': [True] * 64,
        'receipt_clip': receipt_fixture(),
        'receipt_unclipped': receipt_fixture(clipped=False),
        'receipt_nostall': receipt_fixture('nostall'),
    }


class V0FTests(unittest.TestCase):
    def gate(self, rho=0.30, kc=None):
        if kc is None:
            kc = 0.3 * (rho + 0.15)
        return M.v0f([rho], [kc], [True], [True])

    def test_inclusive_floor_and_bound(self):
        result = self.gate()
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['floor_passed'])
        self.assertTrue(result['bound_passed'])
        self.assertEqual(result['median_rho'], 0.30)

    def test_just_below_floor_is_inconclusive(self):
        self.assertEqual(self.gate(np.nextafter(0.30, -np.inf), -100)['status'], 'INCONCLUSIVE')

    def test_just_above_bound_is_inconclusive(self):
        bound = 0.3 * (0.30 + 0.15)
        result = self.gate(0.30, np.nextafter(bound, np.inf))
        self.assertTrue(result['floor_passed'])
        self.assertFalse(result['bound_passed'])
        self.assertEqual(result['status'], 'INCONCLUSIVE')

    def test_negative_kc_passes_one_sided_rule(self):
        self.assertEqual(self.gate(0.5, -1000)['status'], 'PASS')

    def test_empty_input_is_inconclusive_without_numeric_result(self):
        result = M.v0f([], [], np.array([], bool), np.array([], bool))
        self.assertEqual(result['status'], 'INCONCLUSIVE')
        self.assertEqual(result['n_r_valid'], 0)
        self.assertIsNone(result['median_rho'])
        self.assertIsNone(result['bound'])

    def test_empty_selected_subset_is_inconclusive(self):
        result = M.v0f([0.4, 0.5], [0.0, 0.0], [True, False], [False, True])
        self.assertEqual(result['status'], 'INCONCLUSIVE')
        self.assertEqual(result['n_common'], 1)
        self.assertEqual(result['row_indices'], [])

    def test_common_and_item5_masks_are_intersected(self):
        result = M.v0f([-100, 0.4, -100, 0.6], [100, 0.1, 100, 0.2],
                       [False, True, True, True], [True, True, False, True])
        self.assertEqual(result['n_common'], 3)
        self.assertEqual(result['n_r_valid'], 2)
        self.assertEqual(result['row_indices'], [1, 3])
        self.assertAlmostEqual(result['median_rho'], 0.5)
        self.assertAlmostEqual(result['median_kc'], 0.15)
        self.assertEqual(result['status'], 'PASS')

    def test_malformed_inputs_are_rejected(self):
        cases = [([np.nan], [0], [True], [True]),
                 ([0.4], [np.inf], [True], [True]),
                 ([0.4], [0], [1], [True]),
                 ([0.4], [0, 0], [True], [True])]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(M.InvalidData):
                M.v0f(*values)

    def test_any_seed_failure_blocks_aggregate_but_not_other_seeds(self):
        records = [{'seed': i, 'v0f_random': {'status': 'PASS' if i != 1 else 'INCONCLUSIVE'}}
                   for i in (2, 1, 0)]
        result = M.aggregate(records)
        self.assertEqual(result['v0f_only_status'], 'INCONCLUSIVE')
        self.assertEqual([r['seed'] for r in result['per_seed']], [0, 1, 2])
        self.assertEqual(result['per_seed'][0]['v0f_random']['status'], 'PASS')
        self.assertIsNone(result['protocol_decision'])
        self.assertFalse(result['can_unlock_stage2'])

    def test_all_seed_pass_is_diagnostic_only(self):
        result = M.aggregate([{'seed': i, 'v0f_random': {'status': 'PASS'}} for i in range(3)])
        self.assertEqual(result['v0f_only_status'], 'PASS')
        self.assertIsNone(result['protocol_decision'])
        self.assertFalse(result['can_unlock_stage2'])

    def test_duplicate_or_missing_seeds_rejected(self):
        for seeds in ((0, 1), (0, 0, 2), (0, 1, 3)):
            with self.subTest(seeds=seeds), self.assertRaises(M.InvalidData):
                M.aggregate([{'seed': i, 'v0f_random': {'status': 'PASS'}} for i in seeds])


class PredictorAndSchemaTests(unittest.TestCase):
    def test_exact_float32_bits_and_slot_order(self):
        self.assertEqual(M.SLOTS, tuple(EXPECTED_BITS))
        self.assertEqual(M.SLOT_BITS, EXPECTED_BITS)
        for slot, bits in EXPECTED_BITS.items():
            with self.subTest(slot=slot):
                self.assertEqual(int.from_bytes(struct.pack('>f', M.X[slot]), 'big'), bits)
        self.assertNotEqual(M.X['knee'], 0.8)
        self.assertEqual(M.X['s20'], -1.0)

    def test_actual_shaped_affine_predictor_fixture(self):
        parsed = M.anchors(anchors_fixture())
        result = M.predictor_quantities(parsed)
        for slot, drop in (('s05', 0.5), ('s10', 0.8), ('s20', 1.2)):
            np.testing.assert_allclose(result['rho'][slot], drop, atol=1e-12, rtol=0)
            np.testing.assert_allclose(result['errors'][slot], 0, atol=1e-12, rtol=0)
        self.assertTrue(result['item5_ok'].all())
        self.assertEqual(result['rho']['s10'].shape, (48,))

    def test_item5_uses_nostall_discrepancy_and_combines_slots(self):
        fixture = anchors_fixture()
        fixture['qpi_nostall'][0][M.SLOTS.index('s10')] += 0.3
        result = M.predictor_quantities(M.anchors(fixture))
        self.assertFalse(result['item5_ok'][0])
        self.assertTrue(result['item5_ok'][1:].all())
        self.assertAlmostEqual(result['errors']['s10'][0], 0.3, places=12)
        np.testing.assert_allclose(result['rho']['s10'], 0.8, atol=1e-12, rtol=0)

    def test_quadratic_fixture_checks_edge_line_extrapolation(self):
        fixture = anchors_fixture()
        xs = np.array([M.X[s] for s in M.SLOTS])
        baseline = 2 + xs * xs
        fixture['qpi_clip'] = np.tile(baseline, (64, 1)).tolist()
        fixture['qpi_nostall'] = np.tile(baseline, (64, 1)).tolist()
        j = M.SLOTS.index('s10')
        for row in fixture['qpi_clip']:
            row[j] -= 0.8
        edge_x = xs[:5]
        edge_y = baseline[:5]
        slope = np.sum((edge_x - edge_x.mean()) * (edge_y - edge_y.mean())) / np.sum((edge_x - edge_x.mean()) ** 2)
        line_prediction = edge_y.mean() + slope * (xs[j] - edge_x.mean())
        exact_prediction = baseline[j]
        expected_rho = min(line_prediction, exact_prediction) - (baseline[j] - 0.8)
        expected_error = abs(line_prediction - exact_prediction)
        result = M.predictor_quantities(M.anchors(fixture))
        np.testing.assert_allclose(result['rho']['s10'], expected_rho, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result['errors']['s10'], expected_error, atol=1e-12, rtol=0)

    def test_slots_times_shape_validity_and_finite_values_are_enforced(self):
        bad = []
        fixture = anchors_fixture(); fixture['gated_slots'][0:2] = fixture['gated_slots'][1::-1]; bad.append(fixture)
        fixture = anchors_fixture(); fixture['nostall_slots'][1] = fixture['nostall_slots'][0]; bad.append(fixture)
        fixture = anchors_fixture(); fixture['times'][0] = 25; bad.append(fixture)
        fixture = anchors_fixture(); fixture['q1'].pop(); bad.append(fixture)
        fixture = anchors_fixture(); fixture['valid'][0] = 1; bad.append(fixture)
        fixture = anchors_fixture(); fixture['qpi_unclipped'][0][0] = float('nan'); bad.append(fixture)
        for index, fixture in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(M.InvalidData):
                M.anchors(fixture)

    def test_all_three_receipts_are_required(self):
        for field in ('receipt_clip', 'receipt_unclipped', 'receipt_nostall'):
            fixture = anchors_fixture()
            del fixture[field]
            with self.subTest(field=field), self.assertRaises(M.InvalidData):
                M.anchors(fixture)

    def test_receipt_discount_horizon_clipping_and_world_are_enforced(self):
        mutations = [lambda r: r.update(gamma=0.99), lambda r: r.update(horizon=99),
                     lambda r: r.update(bootstrap='bootstrap'), lambda r: r.update(evaluator='environment_loop'),
                     lambda r: r.update(clip='none'),
                     lambda r: r['continuation'].update(world='stall3'),
                     lambda r: r['continuation']['physics'].update(stall=True),
                     lambda r: r['continuation']['physics'].update(stall_slope=1.5)]
        for index, mutate in enumerate(mutations):
            fixture = anchors_fixture()
            mutate(fixture['receipt_nostall'])
            with self.subTest(index=index), self.assertRaises(M.InvalidData):
                M.anchors(fixture)

    def test_malformed_receipt_mapping_is_rejected(self):
        for value in (None, [], {'continuation': []}):
            with self.subTest(value=value), self.assertRaises(M.InvalidData):
                M.receipt(value, True)
        value = receipt_fixture()
        value['continuation'] = []
        with self.assertRaises(M.InvalidData):
            M.receipt(value, True)


class DecompositionTests(unittest.TestCase):
    def test_signs_with_distinct_reference_offsets(self):
        result = M.decomposition([[5., 8.]], [[1., 3.]], [[4., 8.]])
        np.testing.assert_array_equal(result['kc_clip'], [[0., 1.]])
        np.testing.assert_array_equal(result['kc_own'], [[0., -1.]])
        np.testing.assert_array_equal(result['continuation_gap'], [[0., 2.]])
        self.assertEqual(result['max_identity_residual'], 0)

    def test_medians_are_not_assumed_additive(self):
        own_error = np.array([0., 0., 100.])
        gap = np.array([0., 100., 0.])
        q1 = np.column_stack((np.zeros(3), own_error))
        own = np.zeros((3, 2))
        clipped = np.column_stack((np.zeros(3), -gap))
        result = M.decomposition(q1, clipped, own)
        self.assertEqual(np.median(result['kc_clip'][:, 1]), 100)
        self.assertEqual(np.median(result['kc_own'][:, 1]) + np.median(result['continuation_gap'][:, 1]), 0)
        np.testing.assert_array_equal(result['kc_clip'], result['kc_own'] + result['continuation_gap'])

    def test_paired_seed_summary_preserves_sign_and_identity(self):
        result = M.analyze_seed(0, anchors_fixture(kc=0.5, own_gap=0.3),
                               anchors_fixture(kc=0.1, own_gap=-0.2))
        paired = result['slots']['s10']['cutoff_minus_random']
        self.assertAlmostEqual(paired['kc_clip']['median'], 0.4, places=12)
        self.assertAlmostEqual(paired['kc_own']['median'], -0.1, places=12)
        self.assertAlmostEqual(paired['continuation_gap']['median'], 0.5, places=12)
        self.assertEqual(result['v0f_random']['status'], 'PASS')
        self.assertEqual(result['cutoff_companion']['status'], 'INCONCLUSIVE')
        self.assertEqual(result['n_common'], 48)
        self.assertEqual(len(result['slots']['s10']['rows']), 48)

    def test_common_valid_mask_preserved_and_t75_excluded(self):
        cut, random = anchors_fixture(), anchors_fixture()
        cut['valid'][0] = False
        random['valid'][1] = False
        for i in range(48, 64):
            random['q1'][i][M.SLOTS.index('s10')] = 1e10
        result = M.analyze_seed(1, cut, random)
        self.assertEqual(result['n_common'], 46)
        self.assertEqual(result['v0f_random']['row_indices'], list(range(2, 48)))
        self.assertAlmostEqual(result['v0f_random']['median_kc'], 0.10, places=12)

    def test_invalid_decomposition_shapes_and_values_rejected(self):
        cases = [([[1., 2.]], [[1.]], [[1., 2.]]),
                 ([[1., np.inf]], [[1., 2.]], [[1., 2.]]),
                 ([1., 2.], [1., 2.], [1., 2.])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(M.InvalidData):
                M.decomposition(*args)


if __name__ == '__main__':
    unittest.main()
