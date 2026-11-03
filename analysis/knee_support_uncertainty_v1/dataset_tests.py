"""Synthetic-only checks for saved dataset comparison; no experiment imports."""
import unittest
import numpy as np

import dataset_comparison as D


class DatasetComparisonTests(unittest.TestCase):
    def test_permutation_retains_joint_multiset(self):
        a = {"obs": np.arange(12, dtype=np.float32).reshape(4, 3), "reward": np.arange(4, dtype=np.float32)}
        b = {k: v[[2, 0, 3, 1]] for k, v in a.items()}
        r = D.compare_fields(a, b)
        self.assertFalse(r["rowwise_byte_identical"])
        self.assertTrue(r["multiset_byte_identical"])
        self.assertEqual(r["multiset_common_rows"], 4)

    def test_different_value_and_multiplicity(self):
        a = {"reward": np.array([1, 1, 2], np.float32)}
        b = {"reward": np.array([1, 2, 2], np.float32)}
        r = D.compare_fields(a, b)
        self.assertFalse(r["multiset_byte_identical"])
        self.assertEqual(r["multiset_common_rows"], 2)
        self.assertEqual(r["fields"]["reward"]["different_rows"], 1)

    def test_joint_rows_not_independent_field_marginals(self):
        a = {"obs": np.array([[1], [2]], np.float32), "reward": np.array([3, 4], np.float32)}
        b = {"obs": a["obs"], "reward": a["reward"][::-1]}
        self.assertEqual(D.compare_fields(a, b)["multiset_common_rows"], 0)

    def test_dtype_difference_is_not_exact_identity(self):
        a, b = np.array([1, 2], np.float32), np.array([1, 2], np.float64)
        r = D.compare_array(a, b)
        self.assertTrue(r["equal_values"])
        self.assertFalse(r["byte_identical"])
        self.assertFalse(D.compare_fields({"x": a}, {"x": b})["multiset_byte_identical"])

    def test_shape_and_signed_zero(self):
        self.assertFalse(D.compare_array(np.ones((2, 1)), np.ones((2, 2)))["same_shape"])
        r = D.compare_array(np.array([0.], np.float32), np.array([-0.], np.float32))
        self.assertTrue(r["equal_values"])
        self.assertFalse(r["byte_identical"])

    def test_source_keeps_successor_before_filtering(self):
        a = {"observations": np.arange(2 * 101 * 14, dtype=np.float32).reshape(2, 101, 14),
             "actions": np.zeros((2, 100, 4), np.float32), "rewards": np.zeros((2, 100)),
             "terminated": np.zeros((2, 100), bool)}
        source = np.array([[0, 5], [0, 90], [1, 3]])
        f = D.fields_from_source(a, source)
        np.testing.assert_array_equal(f["next_obs"], a["observations"][source[:, 0], source[:, 1] + 1])
        self.assertFalse(np.array_equal(f["next_obs"][0], f["obs"][1]))
        self.assertEqual(f["done"].dtype, np.dtype(np.float32))

    def test_normalization_uses_saved_stats_and_epsilon(self):
        obs = np.array([[3., 4.]], np.float32)
        norm = {"mean": [1., 2.], "std": [2., 0.], "epsilon": .001}
        actual = D.normalize_obs(obs, norm)
        np.testing.assert_array_equal(actual, (obs - np.array([1., 2.], np.float32)) /
                                      (np.array([2., 0.], np.float32) + np.float32(.001)))
        self.assertTrue(np.isfinite(actual).all())
        with self.assertRaises(ValueError):
            D.normalize_obs(obs, {**norm, "std": [2., -1.]})

    def test_content_hash_binds_metadata_dtype_shape_value(self):
        a = {"x": np.array([1, 2], np.float32)}
        h = D.content_digest(a, {"world": "a"})
        for b, meta in [({"x": a["x"].astype(np.float64)}, {"world": "a"}),
                        ({"x": a["x"].reshape(1, 2)}, {"world": "a"}),
                        ({"x": np.array([1, 3], np.float32)}, {"world": "a"}),
                        (a, {"world": "b"})]:
            self.assertNotEqual(h, D.content_digest(b, meta))

    def test_selector_disjoint_roles_and_exact_knee(self):
        n = 132
        obs = np.zeros((n, 101, 14), np.float32)
        obs[..., 10] = .5
        actions = np.zeros((n, 100, 4), np.float32)
        actions[:, 10, 0] = np.float32(D.KNEE)
        actions[:, 11, 0] = np.nextafter(np.float32(D.KNEE), np.float32(1))
        a = {"observations": obs, "actions": actions, "rewards": np.zeros((n, 100)),
             "terminated": np.zeros((n, 100), bool), "split_codes": np.r_[np.zeros(4, np.uint8), np.ones(128, np.uint8)],
             "group_ids": np.arange(n)}
        seeds = dict(random_control=1, calibration_original=2, calibration_replacement=3, gradient_probes=4, scale_bank=5)
        p = D.reconstruct(a, seeds)
        self.assertEqual(len(p["sources"]["train"]), 396)
        self.assertTrue((p["sources"]["train"][:, 0] < 4).all())
        self.assertTrue((p["sources"]["heldout"][:, 0] >= 4).all())
        self.assertEqual(len(np.unique(p["sources"]["heldout"][:, 0])), 128)
        self.assertFalse((p["sources"]["heldout"][:, 1] == 11).any())


if __name__ == "__main__":
    unittest.main()
