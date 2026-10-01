"""Thinned calibration bank (calibration/bank.py); NumPy + SciPy only.

python -m unittest experiments.wbcp.test_bank
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from calibration.bank import dependence_validated, stratified_bank  # noqa: E402


def max_abs_z(counts, trials, probability):
    return float(np.max(np.abs(counts - trials * probability) / np.sqrt(trials * probability * (1 - probability))))


class StratifiedBankTests(unittest.TestCase):
    def test_episodes_are_distinct_and_rows_spread_one_per_segment(self):
        lengths = np.random.default_rng(0).integers(3, 200, size=80)
        for seed, (target, k) in enumerate([(200, 5), (90, 10), (40, 1), (300, 20)]):
            episodes, offsets = stratified_bank(lengths, target, k, np.random.default_rng(seed))
            self.assertEqual(episodes.size, -(-target // k))
            self.assertEqual(np.unique(episodes).size, episodes.size)
            for episode, rows in zip(episodes, offsets):
                length = lengths[episode]
                self.assertEqual(rows.size, min(k, length))
                self.assertEqual(np.unique(rows).size, rows.size)
                self.assertTrue(rows.min() >= 0 and rows.max() < length)
                if length > k:  # consecutive picks, circularly, are less than two segments apart
                    gaps = np.diff(np.append(rows, rows[0] + length))
                    self.assertLess(gaps.max(), 2 * -(-length // k))

    def test_episode_inclusion_probability_is_exactly_proportional_to_length(self):
        lengths = np.random.default_rng(1).integers(20, 200, size=50)
        target, k, trials = 60, 5, 6000
        counts = np.zeros(lengths.size)
        for seed in range(trials):
            episodes, _ = stratified_bank(lengths, target, k, np.random.default_rng(seed))
            counts[episodes] += 1
        probability = (target // k) * lengths / lengths.sum()
        self.assertLess(max_abs_z(counts, trials, probability), 4.5)

    def test_every_row_is_calibrated_with_the_same_probability(self):
        # K m / N for rows of episodes longer than K (segments of unequal size included);
        # m L / N for the rows of an episode shorter than K, which enter whole.
        lengths = np.array([2, 7, 12, 23, 30])
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        k, target, trials = 3, 6, 30000
        counts = np.zeros(lengths.sum())
        for seed in range(trials):
            episodes, offsets = stratified_bank(lengths, target, k, np.random.default_rng(seed))
            for episode, rows in zip(episodes, offsets):
                counts[starts[episode] + rows] += 1
        m, total = target // k, lengths.sum()
        probability = np.where(np.repeat(lengths, lengths) > k, k * m / total, m * np.repeat(lengths, lengths) / total)
        self.assertLess(max_abs_z(counts, trials, probability), 4.5)

    def test_same_seed_same_bank(self):
        lengths = np.arange(10, 60)
        first = stratified_bank(lengths, 50, 5, np.random.default_rng(3))
        second = stratified_bank(lengths, 50, 5, np.random.default_rng(3))
        np.testing.assert_array_equal(first[0], second[0])
        for a, b in zip(first[1], second[1]):
            np.testing.assert_array_equal(a, b)

    def test_rejects_impossible_or_malformed_requests(self):
        rng = np.random.default_rng(0)
        with self.assertRaisesRegex(ValueError, "longest episode"):  # 25 draws cannot be distinct
            stratified_bank(np.array([100] + [4] * 30), 125, 5, rng)
        for lengths, target, k in (([5, 0, 3], 2, 1), ([5, 6], 2, 0), ([5, 6], 2, True), ([5, 6], 0, 1)):
            with self.assertRaises(ValueError):
                stratified_bank(np.array(lengths), target, k, rng)

    def test_validated_range_is_at_most_ten_rows_from_a_hundred_episodes(self):
        self.assertTrue(dependence_validated(5, 221) and dependence_validated(10, 100))
        self.assertFalse(dependence_validated(11, 500) or dependence_validated(5, 99))


if __name__ == "__main__":
    unittest.main()
