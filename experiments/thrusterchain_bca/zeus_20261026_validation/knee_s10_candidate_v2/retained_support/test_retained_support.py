"""Tests of RS1's pure functions on synthetic data (no dataset, no simulator rollout, no sealed stream).

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/retained_support -p "test_*.py"
"""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND1 = HERE.parent.parent / "knee_s10_candidate_v1"
V3 = HERE.parent.parent / "knee_edge_ood_v3"
for _p in (str(V3), str(CAND1), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                          # noqa: E402

import retained_support as RS                               # noqa: E402  imports step04_math only (no recorder)

if "recorder" in sys.modules:
    raise SystemExit("A recorder module was imported before the widened recorder could replace it")
import recorder_wide                                        # noqa: E402  the candidate's recorder, as g2_formal.py binds it
sys.modules["recorder"] = recorder_wide
import fixtures as F                                        # noqa: E402
import masks as M                                           # noqa: E402
from knee_region import past_knee_a0                        # noqa: E402


def synthetic_pool(n=2000, per_episode=10, seed=5):
    """A 14-coordinate pool with episodes of `per_episode` rows and times 0..per_episode-1; the first half of the
    episodes is deep-load."""
    rng = np.random.Generator(np.random.PCG64(seed))
    pool = rng.normal(0.0, 1.0, (n, 14))
    episodes = np.arange(n) // per_episode
    times = np.arange(n) % per_episode
    deep = episodes < (n // per_episode) // 2
    return pool, episodes, times, deep


class PureFunctions(unittest.TestCase):
    def test_standardize_uses_the_pool_and_guards_constant_coordinates(self):
        pool, _, _, _ = synthetic_pool()
        pool[:, 3] = 7.0
        mean, std = RS.standardize(pool)
        self.assertEqual(mean[3], 7.0)
        self.assertEqual(std[3], 1.0)
        self.assertTrue(np.allclose(mean, pool.mean(axis=0)))

    def test_known_answer_group_queries_are_all_inside(self):
        """Queries equal to the group's own rows: each query's distance (own episode kept) is at most its
        leave-one-episode-out distance, and the 0.99 'higher' quantile of a group under 101 rows is its maximum, so
        every query is inside and the share is exactly 1.0."""
        pool, episodes, times, deep = synthetic_pool()
        mean, std = RS.standardize(pool)
        z = (pool - mean) / std
        thr = RS.thresholds(z, episodes, times, deep, times=(0,), window=0, floor=RS.FLOOR)
        rows = RS.group_rows(times, deep, 0)
        self.assertEqual(thr[0]["rows"], 100)
        self.assertEqual(thr[0]["episodes"], 100)
        self.assertFalse(thr[0]["below_floor"])
        out = RS.shares(z, episodes, thr, z[rows], z[rows], np.zeros(len(rows), np.int64))
        self.assertEqual(out["start_share"], 1.0)
        self.assertEqual(out["anchor_share"], 1.0)
        self.assertEqual(out["anchors_inside"], len(rows))
        self.assertEqual(out["by_t"][0]["share"], 1.0)

    def test_far_query_is_outside(self):
        pool, episodes, times, deep = synthetic_pool()
        mean, std = RS.standardize(pool)
        z = (pool - mean) / std
        thr = RS.thresholds(z, episodes, times, deep, times=(0,), window=0)
        far = np.full((4, 14), 50.0)
        out = RS.shares(z, episodes, thr, far, far, np.zeros(4, np.int64))
        self.assertEqual(out["start_share"], 0.0)
        self.assertEqual(out["anchor_share"], 0.0)
        self.assertTrue(np.all(out["anchor_distances"] > thr[0]["threshold"]))

    def test_floor_marks_a_thin_group_and_labels_inconclusive(self):
        pool, episodes, times, deep = synthetic_pool(n=300, per_episode=3)
        deep = episodes < 10                                   # ten deep episodes: ten rows at t = 0
        mean, std = RS.standardize(pool)
        z = (pool - mean) / std
        thr = RS.thresholds(z, episodes, times, deep, times=(0,), window=0, floor=32)
        self.assertEqual(thr[0]["rows"], 10)
        self.assertTrue(thr[0]["below_floor"])
        self.assertIsNotNone(thr[0]["threshold"])              # still reported
        sh = RS.shares(z, episodes, thr, z[:5], z[:5], np.zeros(5, np.int64))
        below, misses = RS.world_labels(thr, sh)
        self.assertEqual(below, [0])
        head, all_labels = RS.labels({"stall3": {"below_floor": below, "misses": misses}}, digests_ok=True, instrument_ok=True)
        self.assertEqual(head, "INCONCLUSIVE")
        self.assertTrue(any(s.startswith("INCONCLUSIVE") for s in all_labels))

    def test_missing_threshold_admits_nothing(self):
        pool, episodes, times, deep = synthetic_pool(n=300, per_episode=3)
        no_deep = np.zeros(len(pool), bool)
        z = pool
        thr = RS.thresholds(z, episodes, times, no_deep, times=(0,), window=0)
        self.assertEqual(thr[0]["rows"], 0)
        self.assertIsNone(thr[0]["threshold"])
        sh = RS.shares(z, episodes, thr, z[:3], z[:3], np.zeros(3, np.int64))
        self.assertEqual(sh["start_share"], 0.0)
        self.assertEqual(sh["anchor_share"], 0.0)

    def test_secondary_window_widens_the_group(self):
        pool, episodes, times, deep = synthetic_pool()
        z = pool
        narrow = RS.thresholds(z, episodes, times, deep, times=(5,), window=0)
        wide = RS.thresholds(z, episodes, times, deep, times=(5,), window=2)
        self.assertEqual(narrow[5]["rows"], 100)
        self.assertEqual(wide[5]["rows"], 500)
        self.assertEqual(wide[5]["window"], 2)

    def test_label_precedence(self):
        clean = {"stall3": {"below_floor": [], "misses": []}, "nostall": {"below_floor": [], "misses": []}}
        self.assertEqual(RS.labels(clean, digests_ok=True, instrument_ok=True), ("RETAINED SUPPORT HOLDS", ["RETAINED SUPPORT HOLDS"]))
        fails = {"stall3": {"below_floor": [], "misses": ["anchor_share"]}, "nostall": {"below_floor": [], "misses": []}}
        head, all_labels = RS.labels(fails, digests_ok=True, instrument_ok=True)
        self.assertEqual(head, "RETAINED SUPPORT FAILS")
        self.assertIn("stall3 anchor_share", all_labels[0])
        both = {"stall3": {"below_floor": [25], "misses": ["anchor_share"]}, "nostall": {"below_floor": [], "misses": ["start_share"]}}
        head, all_labels = RS.labels(both, digests_ok=True, instrument_ok=True)
        self.assertEqual(head, "INCONCLUSIVE")
        self.assertEqual(len(all_labels), 2)                   # both labels reported
        self.assertEqual(RS.labels(both, digests_ok=False, instrument_ok=True)[0], "INCOMPLETE")
        self.assertEqual(RS.labels(clean, digests_ok=True, instrument_ok=False)[0], "INCOMPLETE")
        head, all_labels = RS.labels(clean, digests_ok=True, instrument_ok=True, error="boom")
        self.assertEqual(head, "INCOMPLETE")
        self.assertIn("boom", all_labels[0])
        for key in RS.ORDER:
            self.assertIn(key, RS.LEADS_TO)


class RetainedPool(unittest.TestCase):
    def test_retained_pool_is_the_fitting_rows_outside_w(self):
        """On a synthetic widened-recorder collection, the retained mask is W's complement within the fitting rows, holds
        no W row, and retains about half the high-load rows and four fifths of the low-load rows."""
        arrays = F.synthetic_arrays(episodes=200, entropy=F.TEST_ENTROPY + 3)
        rows = M.fitting_rows(arrays)
        arm, _ = M.arm_masks(rows["obs"], rows["actions"], rows["times"], rows["episode_ids"], seed=F.TEST_SEEDS["random_control"])
        cutoff = arm["cutoff"]
        past = past_knee_a0(rows["actions"])[0]
        self.assertEqual(int(np.sum(cutoff & past)), 0)
        self.assertTrue(np.array_equal(cutoff, ~past))
        self.assertLess(cutoff.sum(), len(cutoff))
        high, low = rows["bands"] >= 1, rows["bands"] == 0
        if high.any():
            self.assertAlmostEqual(float(cutoff[high].mean()), 0.506, delta=0.06)
        if low.any():
            self.assertAlmostEqual(float(cutoff[low].mean()), 0.800, delta=0.06)
        deep = np.repeat(np.asarray(arrays["reset_component"])[rows["episodes"]] == 1, 100)
        r = np.flatnonzero(cutoff)
        self.assertEqual(len(deep[r]), len(r))
        self.assertTrue(set(np.flatnonzero(cutoff)).issubset(set(range(len(rows["obs"])))))


if __name__ == "__main__":
    unittest.main()
