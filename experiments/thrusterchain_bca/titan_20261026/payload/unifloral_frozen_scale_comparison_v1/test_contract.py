"""Meaningful selector boundary checks without GPU, models or simulator."""
import copy
import unittest
from score_contract import make_methods, selection, rms, KAPPAS, FAMILIES


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.q = [{'q': [[1., 2.], [3., .5]], 'u': [.2, 1.], 'disagreement': [1., 2.5],
                   'actor_distance': [0., .4], 'actions': [[0.] * 4, [1.] * 4],
                   'width_usable': True, 'width_valid': True}]
        self.p = {'query_count': 1, 'kappa_grid': list(KAPPAS), 'tie_atol': 1e-6, 'tie_rtol': 1e-6,
                  'contrast_factors': {'bca': 4., 'disagreement': 2., 'actor_distance': 5.}}

    def test_exact_selector_count(self):
        self.assertEqual(len(make_methods(self.q, self.p)), 31)

    def test_all_zero_coefficients_equal_raw_q(self):
        methods = make_methods(self.q, self.p)
        for family in FAMILIES:
            self.assertEqual(methods[family + '_kappa_0']['scores'], methods['raw_q']['scores'])
            self.assertEqual(methods[family + '_kappa_0']['selected_indices'], methods['raw_q']['selected_indices'])

    def test_frozen_factor_changes_only_penalty(self):
        m = make_methods(self.q, self.p)
        self.assertEqual(m['bca_kappa_1']['scores'], [[.8, 2.]])
        self.assertEqual(m['contrast_bca_kappa_1']['scores'], [[1 - 4 * .2, -1.]])

    def test_paired_control_formulas(self):
        m = make_methods(self.q, self.p)
        self.assertEqual(m['contrast_disagreement_kappa_1']['scores'], [[-1., -2.]])
        self.assertEqual(m['contrast_actor_distance_kappa_1']['scores'], [[1., 1.]])

    def test_native_actor_and_twin_min_controls(self):
        m = make_methods(self.q, self.p)
        self.assertEqual(m['actor']['selected_indices'], [0])
        self.assertEqual(m['twin_min']['scores'], [[1., .5]])

    def test_inputs_unchanged(self):
        saved = copy.deepcopy((self.q, self.p))
        make_methods(self.q, self.p)
        self.assertEqual((self.q, self.p), saved)

    def test_first_within_tolerance_is_declared(self):
        choice = selection([1., 1.0000005, 0], 1e-6, 1e-6)
        self.assertEqual((choice['index'], choice['exact_argmax_index']), (0, 1))
        self.assertTrue(choice['inconclusive_selection'])

    def test_distinct_winner(self):
        self.assertEqual(selection([-2, 0, -1], 1e-6, 1e-6)['index'], 1)

    def test_zero_q_ratio_unavailable(self):
        self.q[0]['q'] = [[0., 0.], [0., 0.]]
        self.assertIsNone(make_methods(self.q, self.p)['bca_kappa_1']['penalty_rms_to_q1_rms'][0])

    def test_zero_width_degenerate_not_evidence(self):
        self.q[0]['u'] = [0., 0.]
        m = make_methods(self.q, self.p)
        self.assertTrue(m['bca_kappa_1']['penalty_signal_degenerate'])
        self.assertEqual(m['bca_kappa_1']['selected_indices'], m['raw_q']['selected_indices'])
        self.assertFalse(m['bca_kappa_1']['usefulness_claim_allowed'])

    def test_invalid_calibration_rejected(self):
        for field in ('width_usable', 'width_valid'):
            q = copy.deepcopy(self.q); q[0][field] = False
            with self.assertRaises(ValueError): make_methods(q, self.p)

    def test_invalid_factors_rejected(self):
        for value in (0, -1, float('inf'), float('nan')):
            p = copy.deepcopy(self.p); p['contrast_factors']['bca'] = value
            with self.assertRaises(ValueError): make_methods(self.q, p)

    def test_nonfinite_scores_rejected(self):
        for field in ('q', 'u', 'disagreement', 'actor_distance'):
            q = copy.deepcopy(self.q)
            if field == 'q': q[0][field][0][0] = float('nan')
            else: q[0][field][0] = float('nan')
            with self.assertRaises(ValueError): make_methods(q, self.p)

    def test_negative_penalties_rejected(self):
        self.q[0]['u'][0] = -1
        with self.assertRaises(ValueError): make_methods(self.q, self.p)

    def test_misaligned_candidates_rejected(self):
        self.q[0]['u'].append(1.)
        with self.assertRaises(ValueError): make_methods(self.q, self.p)

    def test_wrong_actor_index_rejected(self):
        self.q[0]['actor_distance'][0] = .1
        with self.assertRaises(ValueError): make_methods(self.q, self.p)

    def test_grid_changes_rejected(self):
        self.p['kappa_grid'] = [0, 1]
        with self.assertRaises(ValueError): make_methods(self.q, self.p)

    def test_stable_rms(self):
        self.assertEqual(rms([1e200, -1e200]), 1e200)
        self.assertEqual(rms([0, 0]), 0)


if __name__ == '__main__': unittest.main()
