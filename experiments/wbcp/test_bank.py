"""Thinned calibration bank and its dependence evidence lookup (calibration/bank.py); NumPy only.

python -m unittest experiments.wbcp.test_bank
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from calibration import bank  # noqa: E402
from calibration.bank import (CONSISTENT, EXCEEDS, NOT_VALIDATED, dependence_evidence,  # noqa: E402
                              evidence_status, load_evidence, stratified_bank)


def max_abs_z(counts, trials, probability):
    return float(np.max(np.abs(counts - trials * probability) / np.sqrt(trials * probability * (1 - probability))))


def untrimmed_bank(lengths, target_size, k, rng):
    """stratified_bank before the remainder trim (2026-10-01), for the bit-for-bit checks."""
    lengths = np.asarray(lengths, np.int64)
    total, draws = int(lengths.sum()), -(-int(target_size) // int(k))
    order = rng.permutation(lengths.size)
    points = (rng.random() + np.arange(draws)) * (total / draws)
    hits = np.searchsorted(np.cumsum(lengths[order]), points, side="right")
    episodes = order[np.minimum(hits, lengths.size - 1)]
    offsets = []
    for length in lengths[episodes]:
        if length <= k:
            offsets.append(np.arange(length))
        else:
            edges = np.arange(k + 1) * length // k
            picks = edges[:-1] + (rng.random(k) * np.diff(edges)).astype(np.int64)
            offsets.append(np.sort((picks + rng.integers(length)) % length))
    return episodes, offsets


def position_bins(lengths, bins):
    """Each pool row's bin of relative position (t + 1/2) / L inside its episode."""
    position = (np.concatenate([np.arange(length) for length in lengths]) + 0.5) / np.repeat(lengths, lengths)
    return np.minimum((position * bins).astype(np.int64), bins - 1)


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

    def test_a_surplus_is_trimmed_to_exactly_n_rows_from_the_same_draws(self):
        # K does not divide n: the bank keeps n of the K ceil(n / K) drawn rows and every
        # reserved episode, even one left with fewer rows (or none), still counts as reserved.
        long = np.random.default_rng(2).integers(40, 121, size=600)
        mixed = np.random.default_rng(3).integers(3, 121, size=300)  # some episodes shorter than K
        for lengths, target, k in ((long, 1024, 23), (long, 1103, 5), (long, 250, 7), (mixed, 1024, 23)):
            for seed in range(5):
                episodes, offsets = stratified_bank(lengths, target, k, np.random.default_rng(seed))
                old_episodes, old_offsets = untrimmed_bank(lengths, target, k, np.random.default_rng(seed))
                drawn = sum(rows.size for rows in old_offsets)
                np.testing.assert_array_equal(episodes, old_episodes)
                self.assertEqual(sum(rows.size for rows in offsets), min(target, drawn))
                for rows, old in zip(offsets, old_offsets):  # a subset of the same draws, still sorted
                    self.assertTrue(np.all(np.isin(rows, old)) and np.all(np.diff(rows) > 0))

    def test_no_surplus_reproduces_the_untrimmed_bank_bit_for_bit_and_draws_nothing_more(self):
        long = np.random.default_rng(4).integers(40, 121, size=1200)
        human = np.random.default_rng(6).integers(150, 251, size=25)  # pen-human-like: K = 124 < L, n = 2 K
        cases = ((long, 1024, 2), (long, 1025, 5), (long, 1035, 23), (human, 248, 124),
                 (np.full(40, 3), 12, 5))  # K does not divide n, but 3 short episodes give 9 <= 12 rows
        for lengths, target, k in cases:
            for seed in range(5):
                new_rng, old_rng = np.random.default_rng(seed), np.random.default_rng(seed)
                episodes, offsets = stratified_bank(lengths, target, k, new_rng)
                old_episodes, old_offsets = untrimmed_bank(lengths, target, k, old_rng)
                np.testing.assert_array_equal(episodes, old_episodes)
                for rows, old in zip(offsets, old_offsets):
                    np.testing.assert_array_equal(rows, old)
                self.assertEqual(new_rng.bit_generator.state, old_rng.bit_generator.state)

    def test_trimmed_bank_includes_every_within_episode_position_equally(self):
        # Every episode longer than K: each row is calibrated with probability n / N, so the
        # expected count in each position bin is n times the bin's share of pool rows. The
        # untrimmed sampler held 1,035 rows here (each bin about 1% high).
        lengths = np.random.default_rng(5).integers(40, 121, size=300)
        bins, target, k, trials = 4, 1024, 23, 2000
        which, starts = position_bins(lengths, bins), np.concatenate([[0], np.cumsum(lengths)[:-1]])
        counts = np.empty((trials, bins))
        for seed in range(trials):
            episodes, offsets = stratified_bank(lengths, target, k, np.random.default_rng(seed))
            rows = np.concatenate([starts[e] + o for e, o in zip(episodes, offsets)])
            self.assertEqual(rows.size, target)
            counts[seed] = np.bincount(which[rows], minlength=bins)
        expected = target * np.bincount(which, minlength=bins) / lengths.sum()
        z = (counts.mean(0) - expected) / (counts.std(0, ddof=1) / np.sqrt(trials))
        self.assertLess(float(np.max(np.abs(z))), 4.5, (counts.mean(0), expected))
        self.assertAlmostEqual(counts[:, -1].mean() / expected[-1], counts[:, 0].mean() / expected[0], delta=0.01)

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


def entry(**fields):
    base = dict(host="td3_bc", dataset="maze2d-large-v1", rows_per_episode=5, size=1024, sampler="reservation",
                score=bank.MIN_Q, uniform_bca_failure=0.0645, ci95=[0.0571, 0.0726], failures=258, trials=4000)
    return dict(base, **fields)


class DependenceEvidenceTests(unittest.TestCase):
    def test_status_follows_the_lower_bound_of_the_interval(self):
        self.assertEqual(evidence_status([0.0465, 0.0607]), CONSISTENT)
        self.assertEqual(evidence_status([0.05, 0.06]), CONSISTENT)  # the bound may touch the budget
        self.assertEqual(evidence_status([0.0501, 0.06]), EXCEEDS)
        self.assertEqual(evidence_status((0.3179, 0.3473)), EXCEEDS)
        for bad in ([0.06, 0.05], [-0.01, 0.02], [0.5, 1.2]):
            with self.assertRaises(ValueError):
                evidence_status(bad)

    def test_every_host_deploys_a_labeled_score(self):
        self.assertEqual(bank.DEPLOYED_SCORE, dict(td3_bc=bank.MIN_Q, rebrac=bank.MIN_Q, cql=bank.MIN_Q,
                                                   iql=bank.TARGET_MIN_Q))
        self.assertNotEqual(bank.MIN_Q, bank.TARGET_MIN_Q)

    def test_lookup_needs_an_exact_host_dataset_k_n_sampler_and_score_match(self):
        entries = [entry(), entry(rows_per_episode=6, ci95=[0.0465, 0.0607], uniform_bca_failure=0.05325)]
        found = dependence_evidence("td3_bc", "maze2d-large-v1", 5, 1024, score=bank.MIN_Q, entries=entries)
        self.assertEqual(found, dict(entry(), status=EXCEEDS))
        self.assertEqual(dependence_evidence("td3_bc", "maze2d-large-v1", 6, 1024, score=bank.MIN_Q,
                                             entries=entries)["status"], CONSISTENT)
        for query in (("cql", "maze2d-large-v1", 5, 1024), ("td3_bc", "maze2d-umaze-v1", 5, 1024),
                      ("td3_bc", "maze2d-large-v1", 7, 1024), ("td3_bc", "maze2d-large-v1", 5, 8192)):
            missing = dependence_evidence(*query, score=bank.MIN_Q, entries=entries)
            self.assertEqual(missing, dict(status=NOT_VALIDATED, host=query[0], dataset=query[1],
                                           rows_per_episode=query[2], size=query[3], sampler="reservation",
                                           score=bank.MIN_Q))
        self.assertEqual(dependence_evidence("td3_bc", "maze2d-large-v1", 5, 1024, score=bank.MIN_Q,
                                             sampler="stratified", entries=entries)["status"], NOT_VALIDATED)
        found["host"] = "changed"  # the caller gets a copy
        self.assertEqual(entries[0]["host"], "td3_bc")
        with self.assertRaises(TypeError):  # the score is part of every query, never matched by omission
            dependence_evidence("td3_bc", "maze2d-large-v1", 5, 1024, entries=entries)

    def test_evidence_for_another_score_never_stands_in_for_the_deployed_one(self):
        q1 = "Q1 residual / sigma, normalized"
        alone = [entry(score=q1, ci95=[0.04, 0.055])]  # the only run of this design calibrates Q1
        missing = dependence_evidence("td3_bc", "maze2d-large-v1", 5, 1024, score=bank.MIN_Q, entries=alone)
        self.assertEqual((missing["status"], missing["score"]), (NOT_VALIDATED, bank.MIN_Q))
        both = [entry(), entry(score=q1, ci95=[0.04, 0.055])]  # both scores measured: each finds its own
        self.assertEqual(dependence_evidence("td3_bc", "maze2d-large-v1", 5, 1024, score=bank.MIN_Q,
                                             entries=both), dict(entry(), status=EXCEEDS))
        self.assertEqual(dependence_evidence("td3_bc", "maze2d-large-v1", 5, 1024, score=q1, entries=both)["status"],
                         CONSISTENT)
        with self.assertRaisesRegex(ValueError, "2 entries for one design"):
            dependence_evidence("td3_bc", "maze2d-large-v1", 5, 1024, score=bank.MIN_Q, entries=[entry(), entry()])

    def test_registry_file_round_trips_and_is_checked(self):
        entries = [entry(), entry(host="cql", dataset="hopper-medium-v2", rows_per_episode=6)]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "evidence.json")
            with open(path, "w") as handle:
                json.dump(dict(schema=bank.EVIDENCE_SCHEMA, entries=entries), handle)
            loaded = load_evidence(path)
            self.assertEqual(loaded, entries)
            loaded[0]["host"] = "changed"  # copies, not the cached registry
            self.assertEqual(load_evidence(path), entries)
            other = os.path.join(directory, "other.json")
            with open(other, "w") as handle:
                json.dump(dict(schema="something-else", entries=entries), handle)
            with self.assertRaisesRegex(ValueError, "registry"):
                load_evidence(other)

    def test_committed_registry_is_well_formed(self):
        entries = load_evidence()
        self.assertTrue(entries)
        keys = [(e["host"], e["dataset"], e["rows_per_episode"], e["size"], e["sampler"], e["score"]) for e in entries]
        self.assertEqual(len(set(keys)), len(keys))
        for e in entries:
            with self.subTest(source=e["source"]):
                self.assertEqual(e["sampler"], "reservation")
                self.assertAlmostEqual(e["uniform_bca_failure"], e["failures"] / e["trials"], places=12)
                self.assertTrue(e["ci95"][0] <= e["uniform_bca_failure"] <= e["ci95"][1])
                self.assertEqual(len(e["source_sha256"]), 64)
                self.assertIn(e["bank_trim"], (None, bank.REMAINDER_TRIM))
                self.assertIn("with the remainder trim" if e["bank_trim"] else "before the remainder trim", e["note"])
                found = dependence_evidence(e["host"], e["dataset"], e["rows_per_episode"], e["size"], score=e["score"])
                self.assertEqual(found["status"], evidence_status(e["ci95"]))
                self.assertEqual(found["source"], e["source"])


if __name__ == "__main__":
    unittest.main()
