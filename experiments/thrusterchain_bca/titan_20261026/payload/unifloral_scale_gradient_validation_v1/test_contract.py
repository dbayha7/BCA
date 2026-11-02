"""Stdlib tests of the diagnostic contract; no model or scientific imports."""
import math
import unittest
from contract import candidate_scale, contrast_statistics, native_epsilon_response


class ContractChecks(unittest.TestCase):
    q = [[-2.0, 0.0, 2.0], [-4.0, 0.0, 4.0]]
    u = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]

    def test_known_contrasts(self):
        value = contrast_statistics([[1, 3], [2, 4]])
        self.assertAlmostEqual(value["raw_rms"], math.sqrt(7.5))
        self.assertEqual(value["centered_rms"], 1)

    def test_per_state_constant_subtraction(self):
        a = contrast_statistics([[1, 3], [2, 4]])
        b = contrast_statistics([[-99, -97], [502, 504]])
        self.assertEqual(a["centered_rms"], b["centered_rms"])
        self.assertNotEqual(a["raw_rms"], b["raw_rms"])

    def test_ragged_equal_state_weighting(self):
        value = contrast_statistics([[-1, 1], [-2, 0, 2]])
        self.assertAlmostEqual(value["centered_rms"], math.sqrt(11/6))

    def test_coherent_unit_scaling(self):
        base = candidate_scale(self.q, self.u)
        self.assertEqual(base["status"], "available")
        for scale in (0.1, 1, 10):
            value = candidate_scale([[scale*x for x in row] for row in self.q],
                                    [[scale*x for x in row] for row in self.u])
            self.assertAlmostEqual(value["factor"], base["factor"])

    def test_only_q_scale_changes_relative_strength(self):
        base = candidate_scale(self.q, self.u)["factor"]
        value = candidate_scale([[10*x for x in row] for row in self.q], self.u)
        self.assertAlmostEqual(value["factor"], 10*base)

    def test_only_u_scale_changes_relative_strength(self):
        base = candidate_scale(self.q, self.u)["factor"]
        value = candidate_scale(self.q, [[10*x for x in row] for row in self.u])
        self.assertAlmostEqual(value["factor"], base/10)

    def test_tiny_coherent_values_no_absolute_floor(self):
        value = candidate_scale([[1e-200*x for x in row] for row in self.q],
                                [[1e-200*x for x in row] for row in self.u])
        self.assertEqual(value["status"], "available")
        self.assertAlmostEqual(value["factor"], 2)

    def test_zero_u_unavailable(self):
        value = candidate_scale(self.q, [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(value["status"], "unavailable")
        self.assertIsNone(value["factor"])

    def test_constant_u_unavailable(self):
        self.assertEqual(candidate_scale(self.q, [[1, 1, 1], [2, 2, 2]])["status"], "unavailable")

    def test_constant_q_unavailable(self):
        self.assertEqual(candidate_scale([[1, 1, 1], [2, 2, 2]], self.u)["status"], "unavailable")

    def test_zero_q_unavailable(self):
        self.assertEqual(candidate_scale([[0, 0, 0], [0, 0, 0]], self.u)["status"], "unavailable")

    def test_near_constant_u_unavailable(self):
        rows = [[1, 1+1e-9, 1-1e-9]]*2
        self.assertEqual(candidate_scale(self.q, rows)["status"], "unavailable")

    def test_near_constant_q_unavailable(self):
        rows = [[1, 1+1e-9, 1-1e-9]]*2
        self.assertEqual(candidate_scale(rows, self.u)["status"], "unavailable")

    def test_offset_can_change_usability_gate(self):
        self.assertEqual(candidate_scale(self.q, self.u)["status"], "available")
        q = [[x+1e10 for x in row] for row in self.q]
        self.assertEqual(candidate_scale(q, self.u)["status"], "unavailable")

    def test_nonfinite_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                contrast_statistics([[0, value]])

    def test_empty_and_boolean_rejected(self):
        for rows in ([], [[]], [[True, 1]]):
            with self.assertRaises(ValueError):
                contrast_statistics(rows)

    def test_bank_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            candidate_scale([[0, 1]], [[0, 1, 2]])

    def test_invalid_relative_floor_rejected(self):
        for floor in (0, -1, 1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                candidate_scale(self.q, self.u, floor)

    def test_native_zero_q_finite_large_coefficient(self):
        value = native_epsilon_response(0, 0.1)
        self.assertEqual(value["q_scale_coefficient"], 25000000)
        self.assertAlmostEqual(value["value_gradient_multiplier"], 0.1)

    def test_fixed_epsilon_analytic_multiplier(self):
        for mean in (0, 1e-10, 1e-7, 1, 1000):
            for scale in (0.1, 1, 10):
                value = native_epsilon_response(mean, scale)
                measured = scale*value["scaled_q_scale_coefficient"]/value["q_scale_coefficient"]
                self.assertAlmostEqual(measured, value["value_gradient_multiplier"])

    def test_coscaled_epsilon_is_separate_counterfactual(self):
        for mean in (0, 1e-10, 1e-7, 1):
            for scale in (0.1, 10):
                ratio = scale*(2.5/(scale*mean+scale*1e-7))/(2.5/(mean+1e-7))
                self.assertAlmostEqual(ratio, 1)

    def test_invalid_native_coefficient_inputs_rejected(self):
        for mean, scale in ((-1, 1), (1, 0), (1, -1), (float('inf'), 1), (1, float('nan'))):
            with self.assertRaises(ValueError):
                native_epsilon_response(mean, scale)

    def test_constant_subtraction_does_not_change_argmax(self):
        q, u, kappa = [1, 3, 2], [3, 1, 4], 0.3
        a = [x-kappa*y for x, y in zip(q, u)]
        b = [x-kappa*(y-100) for x, y in zip(q, u)]
        self.assertEqual(max(range(3), key=a.__getitem__), max(range(3), key=b.__getitem__))


if __name__ == '__main__':
    unittest.main()
