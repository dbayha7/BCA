"""Shared frozen-reference and dose checks (JAX); no host imports.

JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_reference
"""
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from calibration import wbcp
from calibration.dose import frozen_level_dose
from calibration.reference import (WBCPConfig, certifiable, freeze_reference, initial_reference,
                                   reference_valid, reserve_calibration)

PARAMS = {"w": jnp.ones((3, 2))}
KEY = jax.random.fold_in(jax.random.PRNGKey(7), 11)


def frozen(residuals, predictions, unit=0.5, config=WBCPConfig(draws=500)):
    return freeze_reference(PARAMS, jnp.asarray(unit), jnp.asarray(predictions),
                            jnp.asarray(residuals), KEY, config)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(0)
        self.predictions = generator.uniform(0.5, 2.0, 1103).astype(np.float32)
        self.residuals = (generator.normal(size=1103) * self.predictions).astype(np.float32)

    def test_threshold_is_wbcp_on_normalized_scores(self):
        reference, valid, diagnostics = frozen(self.residuals, self.predictions)
        scores = np.abs(self.residuals.astype(np.float64)) / (np.maximum(self.predictions, 1e-6) * 0.5).astype(np.float64)
        expected = wbcp.calibrate(scores, np.random.default_rng(np.asarray(KEY, np.uint32).ravel()),
                                  alpha=0.1, beta=0.95, draws=500)
        self.assertTrue(valid and bool(reference.ready) and diagnostics["certified"])
        self.assertEqual(float(reference.threshold), np.float32(expected.threshold))
        self.assertEqual(float(reference.lambda_hat), np.float32(expected.lambda_hat))
        self.assertEqual(float(reference.lambda_hpd), np.float32(expected.lambda_hpd))
        self.assertTrue(bool(reference_valid(reference)))

    def test_refresh_is_reproducible_from_its_key(self):
        first, _, _ = frozen(self.residuals, self.predictions)
        second, _, _ = frozen(self.residuals, self.predictions)
        self.assertEqual(float(first.threshold), float(second.threshold))

    def test_invalid_inputs_are_rejected_without_a_threshold(self):
        # The last case scores near 1e42: finite in float64, +inf once stored as float32.
        for residuals, predictions, unit in ((np.array([np.nan, 1.0]), np.ones(2), 0.5),
                                             (np.ones(2), np.array([1.0, np.inf]), 0.5),
                                             (np.ones(2), np.ones(2), 0.0),
                                             (np.full(60, 1e30), np.full(60, 1e-6), 1e-6)):
            reference, valid, _ = frozen(residuals, predictions, unit)
            self.assertFalse(valid)
            self.assertFalse(bool(reference.ready))

    def test_initial_reference_leaves_the_native_multiplier(self):
        reference = initial_reference(PARAMS)
        dose = frozen_level_dose(reference, jnp.asarray(self.predictions[:64]), 0.5)
        self.assertTrue(bool(reference_valid(reference)) and bool(dose.inputs_valid))
        np.testing.assert_array_equal(np.asarray(dose.dose), 1.0)

    def test_dose_uses_threshold_times_frozen_scale(self):
        reference, _, _ = frozen(self.residuals, self.predictions)
        eta = jnp.asarray(self.predictions[:64])
        dose = jax.jit(lambda r, p: frozen_level_dose(r, p, 0.5))(reference, eta)
        width = float(reference.threshold) * np.maximum(np.asarray(eta), 1e-6) * float(reference.residual_scale)
        expected = 1.0 + 0.5 * width / (width + float(reference.residual_scale))
        np.testing.assert_allclose(np.asarray(dose.dose), expected, rtol=1e-5)
        self.assertTrue(bool(dose.inputs_valid) and bool(np.all(np.asarray(dose.support_mask))))

    def test_uncertifiable_threshold_falls_back_to_native_rows(self):
        reference, valid, diagnostics = frozen(self.residuals[:10], self.predictions[:10])  # 10 rows: Pr(certify) is essentially 0
        self.assertTrue(valid and not diagnostics["certified"] and math.isinf(float(reference.threshold)))
        dose = frozen_level_dose(reference, jnp.asarray(self.predictions[:8]), 0.5)
        self.assertTrue(bool(dose.inputs_valid))
        np.testing.assert_array_equal(np.asarray(dose.dose), 1.0)
        self.assertFalse(bool(np.any(np.asarray(dose.support_mask))))

    def test_storage_check_is_traceable_and_catches_a_broken_clamp(self):
        reference, _, _ = frozen(self.residuals, self.predictions)
        self.assertTrue(bool(jax.jit(reference_valid)(reference)))
        broken = reference._replace(threshold=reference.lambda_hat * 0.5)
        self.assertFalse(bool(reference_valid(broken)))

    def test_certifiable_bank_is_a_binomial_tail_on_the_test_atom(self):
        config = WBCPConfig()
        self.assertEqual([certifiable(n, config) for n in (0, 35, 36, 1103)], [False, False, True, True])
        self.assertFalse(certifiable(36.0, config))
        # n = 29 passes the expected-crossing rule (1-alpha)^n <= 1-beta, yet many refreshes cannot certify.
        generator = np.random.default_rng(3)
        failures = sum(not wbcp.calibrate(generator.exponential(size=29), generator, draws=1000).certified
                       for _ in range(200))
        self.assertGreater(failures, 20)

    def test_config_rejects_bad_settings(self):
        for kwargs in (dict(alpha=0.0), dict(credibility=1.0), dict(draws=0), dict(draws=1.5)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WBCPConfig(**kwargs)


def episodes_of(lengths):
    return np.repeat(np.arange(len(lengths)), lengths)


def reserve(lengths, target, k, seed=0, max_fraction=0.5):
    ids = episodes_of(lengths)
    obs = np.zeros((ids.size, 2), np.float32)
    return reserve_calibration(obs, obs, np.zeros(ids.size), target, seed, k, max_fraction, episode_ids=ids)


class ReservationTests(unittest.TestCase):
    lengths = np.random.default_rng(4).integers(3, 120, size=300)

    def test_withheld_episodes_leave_training_and_calibration_thins_them(self):
        train, withheld, cal, meta = reserve(self.lengths, 400, 5)
        ids = episodes_of(self.lengths)
        np.testing.assert_array_equal(np.sort(np.r_[train, withheld]), np.arange(ids.size))
        self.assertTrue(set(cal) <= set(withheld))
        self.assertFalse(set(ids[train]) & set(ids[withheld]))  # whole episodes are withheld
        reserved = np.unique(ids[withheld])
        np.testing.assert_array_equal(np.unique(ids[cal]), reserved)
        np.testing.assert_array_equal(np.bincount(ids[cal], minlength=ids.max() + 1)[reserved],
                                      np.minimum(5, self.lengths[reserved]))
        self.assertEqual((meta["reserved_blocks"], meta["rows_per_episode"]), (80, 5))
        self.assertEqual((meta["calibration_size"], meta["withheld_size"]), (cal.size, withheld.size))
        self.assertAlmostEqual(meta["withheld_fraction"], withheld.size / ids.size)
        self.assertFalse(meta["dependence_validated"])  # 80 episodes is below the validated 100

    def test_inferred_boundaries_match_explicit_episode_ids(self):
        # Without IDs, episodes end at a terminal or where next_obs breaks from the next obs.
        ids = episodes_of(self.lengths)
        obs = np.stack([ids, np.arange(ids.size)], axis=1).astype(np.float32)
        nxt = obs.copy()
        nxt[:-1] = obs[1:]
        nxt[np.flatnonzero(np.diff(ids))] += 1000  # break the chain at every episode end
        inferred = reserve_calibration(obs, nxt, np.zeros(ids.size), 400, 0, 5, 0.5)
        explicit = reserve(self.lengths, 400, 5)
        for a, b in zip(inferred[:3], explicit[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(inferred[3]["dependence_validated"] is explicit[3]["dependence_validated"])

    def test_cap_counts_every_withheld_row_and_raises(self):
        _, withheld, cal, _ = reserve(self.lengths, 400, 5)
        self.assertGreater(withheld.size, 5 * cal.size)  # the cap binds on withheld rows, not the bank
        with self.assertRaisesRegex(ValueError, "reserved fraction"):
            reserve(self.lengths, 400, 5, max_fraction=0.5 * withheld.size / self.lengths.sum())

    def test_population_split_mode_keeps_whole_episodes_in_permutation_order(self):
        # rows_per_episode=None is the pre-thinning rule, kept for freeze_scores' population split.
        train, withheld, cal, meta = reserve(self.lengths, 2000, None, seed=911)
        order = np.random.default_rng(911).permutation(self.lengths.size)
        count = int(np.searchsorted(np.cumsum(self.lengths[order]), 2000)) + 1
        expected = np.flatnonzero(np.isin(episodes_of(self.lengths), order[:count]))
        np.testing.assert_array_equal(withheld, expected)
        np.testing.assert_array_equal(cal, expected)
        self.assertEqual(train.size + withheld.size, self.lengths.sum())
        self.assertFalse(meta["dependence_validated"])
        self.assertIsNone(meta["rows_per_episode"])

    def test_same_seed_same_reservation(self):
        first, second = reserve(self.lengths, 300, 10, seed=9), reserve(self.lengths, 300, 10, seed=9)
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(first[3]["calibration_indices_sha256"], second[3]["calibration_indices_sha256"])


if __name__ == "__main__":
    unittest.main()
