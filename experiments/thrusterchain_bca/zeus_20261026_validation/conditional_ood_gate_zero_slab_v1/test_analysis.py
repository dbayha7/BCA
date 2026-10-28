import json
from pathlib import Path
import unittest
import numpy as np
import analysis

CONTRACT = json.loads(Path(__file__).with_name('contract.json').read_text())


class AnalysisTests(unittest.TestCase):
    def data(self):
        obs = np.zeros((64, 14), np.float32)
        obs[:32, 10] = .6
        obs[32:, 10] = .2
        banks = np.zeros((64, 6, 4), np.float32)
        banks[:, :, 0] = [.5, -.2, -.09, 0, .09, .2]  # actor slot outside the zero-centered band
        values = np.zeros((64, 6), np.float64)
        values[:8, 3] = 1
        free = {'high_load_action_rows': 200, 'hole_entries': 2, 'episodes_with_hole_entry': 2,
                'returns': np.zeros(16), 'observations': None, 'actions': None, 'episode_seeds': []}
        return obs, banks, values, free

    def test_boundary_and_failure_preservation(self):
        obs, banks, values, free = self.data()
        report = analysis.analyze_values(values, values, banks, obs, free, CONTRACT)
        self.assertTrue(report['passed'])
        self.assertEqual(report['high_load_gap_fraction'], .25)
        free['hole_entries'] = 0
        self.assertFalse(analysis.analyze_values(values, values, banks, obs, free, CONTRACT)['passed'])

    def test_signs_tolerance_and_cells(self):
        obs, banks, values, free = self.data()
        q = values.copy()
        q[:32] += 2
        q[32:] -= 3
        report = analysis.analyze_values(values, q, banks, obs, free, CONTRACT)
        self.assertEqual(report['raw_q_errors_all']['over'], 192)
        self.assertEqual(report['raw_q_errors_all']['under'], 192)
        self.assertEqual(report['raw_q_errors_all']['mae'], 2.5)
        self.assertEqual(sum(x['rows'] for x in report['error_cells'].values()), 384)
        self.assertEqual(report['error_cells']['low_inside_band']['mean_underestimation_magnitude'], 3)

    def test_low_load_gap_cannot_rescue_high_load_gate(self):
        obs, banks, values, free = self.data()
        values[:32] = 0
        values[32:, 3] = 20
        report = analysis.analyze_values(values, values, banks, obs, free, CONTRACT)
        self.assertFalse(report['criteria']['practical_center_flank_gap'])

    def test_numerical_tie_uses_first_candidate(self):
        x = np.zeros((1, 6))
        x[0, 4] = 1e-6
        self.assertEqual(analysis._first_max_with_tolerance(x, CONTRACT).tolist(), [0])
        x[0, 4] = 1
        self.assertEqual(analysis._first_max_with_tolerance(x, CONTRACT).tolist(), [4])

    def test_nonfinite_refused(self):
        obs, banks, values, free = self.data()
        values[0, 0] = np.inf
        with self.assertRaises(ValueError):
            analysis.analyze_values(values, values, banks, obs, free, CONTRACT)


if __name__ == '__main__':
    unittest.main()
