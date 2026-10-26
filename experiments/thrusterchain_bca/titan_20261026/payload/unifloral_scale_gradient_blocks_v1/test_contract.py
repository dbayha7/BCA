"""No scientific imports: test coverage and aggregation failure modes."""
import unittest
from contract import block_plan, normalized_weights, weighted_mean, ratio_summary


class BlockContractTests(unittest.TestCase):
    def test_exact_coverage(self):
        blocks = block_plan(256, 32, range(0, 256, 32))
        self.assertEqual([i for start, stop in blocks for i in range(start, stop)], list(range(256)))

    def test_omitted_block_rejected(self):
        with self.assertRaises(ValueError):
            block_plan(256, 32, range(32, 256, 32))

    def test_duplicate_block_rejected(self):
        with self.assertRaises(ValueError):
            block_plan(64, 32, [0, 0])

    def test_reordered_block_rejected(self):
        with self.assertRaises(ValueError):
            block_plan(64, 32, [32, 0])

    def test_partial_block_rejected(self):
        with self.assertRaises(ValueError):
            block_plan(65, 32, [0, 32, 64])

    def test_invalid_dimensions_rejected(self):
        for rows, size in [(0, 32), (256, 0), (True, 1), (256.0, 32)]:
            with self.assertRaises(ValueError):
                block_plan(rows, size, [])

    def test_boolean_start_rejected(self):
        with self.assertRaises(ValueError):
            block_plan(32, 32, [False])

    def test_equal_row_weights(self):
        self.assertEqual(normalized_weights([32] * 8), [0.125] * 8)

    def test_unequal_row_weights(self):
        self.assertEqual(normalized_weights([1, 3]), [0.25, 0.75])
        self.assertEqual(weighted_mean([4, 8], [1, 3]), 7)

    def test_invalid_counts_rejected(self):
        for counts in ([], [0], [-1], [True], [1.5]):
            with self.assertRaises(ValueError):
                normalized_weights(counts)

    def test_weighted_gradient_cancellation_retained(self):
        self.assertEqual(weighted_mean([3, -3], [32, 32]), 0)
        self.assertNotEqual(weighted_mean([abs(3), abs(-3)], [32, 32]), 0)

    def test_normalize_after_aggregation_differs(self):
        means, raw_gradients, counts = [1, 9], [1, 1], [32, 32]
        correct = 2.5 / (weighted_mean(means, counts) + 1e-7) * weighted_mean(raw_gradients, counts)
        incorrect = weighted_mean([2.5 / (m + 1e-7) * g for m, g in zip(means, raw_gradients)], counts)
        self.assertAlmostEqual(correct, 0.5, places=7)
        self.assertGreater(abs(correct - incorrect), 0.8)

    def test_nonfinite_and_mismatched_means_rejected(self):
        for values, counts in [([float('nan')], [1]), ([float('inf')], [1]), ([True], [1]), ([1, 2], [1])]:
            with self.assertRaises(ValueError):
                weighted_mean(values, counts)

    def test_missing_ratios_are_not_zero(self):
        result = ratio_summary([None, 0, 2])
        self.assertEqual((result['defined'], result['unavailable'], result['mean']), (2, 1, 1))

    def test_all_missing_ratios(self):
        result = ratio_summary([None, None])
        self.assertIsNone(result['median'])
        self.assertIsNone(result['mean'])
        self.assertEqual(result['unavailable'], 2)

    def test_invalid_ratios_rejected(self):
        for values in [[float('inf')], [float('nan')], [-1], [True]]:
            with self.assertRaises(ValueError):
                ratio_summary(values)


if __name__ == '__main__':
    unittest.main()
