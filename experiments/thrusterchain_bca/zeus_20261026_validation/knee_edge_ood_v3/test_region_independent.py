"""Unit tests for region_independent.py (protocol section 4.2, independent second implementation).

Run from the repository root:
    python -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v1
        -p test_region_independent.py -v
"""
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import region_independent as R  # noqa: E402

F32 = np.float32
KNEE = F32(0.8)
UP = np.nextafter(KNEE, F32(1))
DOWN = np.nextafter(KNEE, F32(0))


def rows(a0, rest=(0.25, -0.5, 0.75)):
    """(N, 4) float32 actions with the given first commands and fixed a1..a3."""
    a0 = np.asarray(a0, F32).reshape(-1)
    out = np.empty((a0.size, 4), F32)
    out[:, 0] = a0
    out[:, 1:] = np.asarray(rest, F32)
    return out


def f32_from_bits(bits):
    return np.asarray(bits, np.uint32).view(F32)


def float32_band(lo, hi):
    """Every float32 x with bits(float32(lo)) <= bits(x) <= bits(float32(hi)), lo, hi positive."""
    b0 = int(F32(lo).view(np.uint32))
    b1 = int(F32(hi).view(np.uint32))
    return f32_from_bits(np.arange(b0, b1 + 1, dtype=np.uint32))


def find_pair(target, x_start, steps=4000, window=64, required=True):
    """Search float32 (x, y) with float64(x) + float64(y) == target (a float64); x walks down from x_start."""
    x = F32(x_start)
    for _ in range(steps):
        guess = target - np.float64(x)
        y0 = F32(guess)
        y = y0
        for _ in range(window):
            if np.float64(x) + np.float64(y) == target:
                return x, y
            y = np.nextafter(y, F32(np.inf))
        y = y0
        for _ in range(window):
            if np.float64(x) + np.float64(y) == target:
                return x, y
            y = np.nextafter(y, F32(-np.inf))
        x = np.nextafter(x, F32(0))
    if required:
        raise AssertionError(f"no float32 pair sums to {target!r}")
    return None


def obs_with(current, wave, batch=()):
    obs = np.zeros(batch + (14,), F32)
    obs[..., 10] = current
    obs[..., 11] = wave
    obs[..., 13] = F32(0.5)
    return obs


class ConstantsTest(unittest.TestCase):
    def test_knee(self):
        self.assertEqual(int(R.KNEE.view(np.uint32)), 0x3F4CCCCD)
        self.assertEqual(float(R.KNEE), 0.800000011920929)
        self.assertEqual(R.KNEE_FRACTION, Fraction(float(F32(0.8))))
        self.assertEqual(R.LOAD_EDGE_MID, Fraction(0.4))
        self.assertNotEqual(R.LOAD_EDGE_MID, Fraction(2, 5))


class RegionTest(unittest.TestCase):
    def check(self, values, expected_mask, expected_side):
        mask, side = R.withheld_region(rows(values))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(side.dtype, np.int8)
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(side, expected_side)

    def test_edges(self):
        self.check([KNEE, -KNEE], [False, False], [0, 0])
        self.check([UP, -UP], [True, True], [1, -1])
        self.check([DOWN, -DOWN], [False, False], [0, 0])
        self.check([1.0, -1.0], [True, True], [1, -1])

    def test_signed_zero(self):
        values = np.array([0.0, -0.0], F32)
        self.assertEqual(int(values[1].view(np.uint32)), 0x80000000)
        self.check(values, [False, False], [0, 0])

    def test_requested_cast(self):
        requested = np.zeros((3, 4), np.float64)
        requested[:, 0] = [0.8, 0.80000001, 0.8000000119]
        for v in requested[:, 0]:
            self.assertEqual(int(F32(v).view(np.uint32)), 0x3F4CCCCD)
        mask, side = R.withheld_region_requested(requested)
        np.testing.assert_array_equal(mask, [False, False, False])
        np.testing.assert_array_equal(side, [0, 0, 0])
        requested[:, 0] = -requested[:, 0]
        self.assertFalse(R.withheld_region_requested(requested)[0].any())
        # a float64 request just past the float32 knee lands on UP and is in W
        req = np.zeros((1, 4))
        req[0, 0] = float(UP)
        self.assertTrue(R.withheld_region_requested(req)[0][0])

    def test_raises(self):
        good = rows([0.5])
        for bad_value in (np.nan, np.inf, -np.inf):
            for col in range(4):
                bad = good.copy()
                bad[0, col] = bad_value
                with self.assertRaises(ValueError):
                    R.withheld_region(bad)
        for col in range(4):
            bad = good.copy()
            bad[0, col] = np.nextafter(F32(1), F32(2))
            with self.assertRaises(ValueError):
                R.withheld_region(bad)
            bad[0, col] = F32(-1.5)
            with self.assertRaises(ValueError):
                R.withheld_region(bad)
        with self.assertRaises(ValueError):
            R.withheld_region(good.astype(np.float64))  # float64 passed as executed
        with self.assertRaises(ValueError):
            R.withheld_region(good.astype(np.float16))
        with self.assertRaises(ValueError):
            R.withheld_region(np.zeros((2, 3), F32))
        with self.assertRaises(ValueError):
            R.withheld_region([[0.0, 0.0, 0.0, 0.0]])
        req = np.zeros((1, 4))
        for bad_value in (np.nan, np.inf, 1e300):
            req[0, 0] = bad_value
            with self.assertRaises(ValueError):
                R.withheld_region_requested(req)
        req[0, 0] = 1.5
        with self.assertRaises(ValueError):
            R.withheld_region_requested(req)
        with self.assertRaises(ValueError):
            R.withheld_region_requested(good)  # float32 passed as requested

    def test_symmetric_and_monotone(self):
        sweep = float32_band(0.79, 0.81)
        self.assertTrue(np.all(np.diff(sweep.astype(np.float64)) > 0))
        pos_mask, pos_side = R.withheld_region(rows(sweep))
        neg_mask, neg_side = R.withheld_region(rows(-sweep))
        np.testing.assert_array_equal(pos_mask, neg_mask)
        np.testing.assert_array_equal(pos_side, -neg_side)
        self.assertTrue(np.all(np.diff(pos_mask.astype(np.int8)) >= 0))  # monotone in |a0|
        first = int(np.argmax(pos_mask))
        self.assertEqual(sweep[first], UP)
        self.assertEqual(sweep[first - 1], KNEE)

    def test_other_thrusters_ignored(self):
        values = np.array([0.0, 0.5, KNEE, UP, -UP, -1.0, 1.0], F32)
        base, base_side = R.withheld_region(rows(values))
        rng = np.random.default_rng(42)
        for rest in ([0, 0, 0], [1, -1, 1], [-1, -1, -1], [UP, -UP, KNEE]):
            mask, side = R.withheld_region(rows(values, rest))
            np.testing.assert_array_equal(mask, base)
            np.testing.assert_array_equal(side, base_side)
        a = rows(values)
        a[:, 1:] = rng.uniform(-1, 1, size=(values.size, 3)).astype(F32)
        np.testing.assert_array_equal(R.withheld_region(a)[0], base)

    def test_shapes(self):
        rng = np.random.default_rng(7)
        flat = rng.uniform(-1, 1, size=(60, 4)).astype(F32)
        flat[:6, 0] = [KNEE, UP, -UP, -KNEE, 1.0, -0.0]
        mask, side = R.withheld_region(flat)
        self.assertEqual(mask.shape, (60,))
        cube = flat.reshape(3, 20, 4)
        cmask, cside = R.withheld_region(cube)
        self.assertEqual(cmask.shape, (3, 20))
        self.assertEqual(cside.shape, (3, 20))
        np.testing.assert_array_equal(cmask.reshape(-1), mask)
        np.testing.assert_array_equal(cside.reshape(-1), side)
        np.testing.assert_array_equal(mask[:6], [False, True, True, False, True, False])
        # a non-contiguous view works too
        np.testing.assert_array_equal(R.withheld_region(cube[:, ::2])[0], cmask[:, ::2])
        np.testing.assert_array_equal(R.agree(cube), cmask)
        np.testing.assert_array_equal(R.in_deep(cube).shape, (3, 20))


class DepthTest(unittest.TestCase):
    def test_examples(self):
        just_deep = f32_from_bits(0xBF666667)
        just_short = f32_from_bits(0xBF666666)
        self.assertEqual(just_short, F32(-0.9))
        d_deep = R.depth_fraction(just_deep)
        d_short = R.depth_fraction(just_short)
        self.assertIsInstance(d_deep, Fraction)
        self.assertEqual(d_deep, Fraction(float(just_deep)) * -1 - Fraction(float(KNEE)))
        self.assertGreater(d_deep, Fraction(1, 10))
        self.assertLess(d_short, Fraction(1, 10))
        self.assertAlmostEqual(float(d_deep), 0.10000002384185791, places=15)
        self.assertAlmostEqual(float(d_short), 0.09999996423721313, places=15)
        np.testing.assert_array_equal(R.in_deep_minus(rows([just_deep, just_short])), [True, False])
        np.testing.assert_array_equal(R.in_deep(rows([-just_deep, -just_short])), [True, False])
        np.testing.assert_array_equal(R.in_deep_minus(rows([-just_deep])), [False])

    def test_named_points(self):
        s05, s15, s20 = F32(-0.85), F32(-0.95), F32(-1.0)
        a = rows([s05, s15, s20])
        np.testing.assert_array_equal(R.in_deep_minus(a), [False, True, True])
        np.testing.assert_array_equal(R.in_deep(a), [False, True, True])
        np.testing.assert_array_equal(R.withheld_region(a)[0], [True, True, True])
        depths = R.depth_fraction(a[:, 0])
        self.assertEqual(depths.dtype, object)
        self.assertEqual(depths[2], Fraction(1) - Fraction(float(KNEE)))
        self.assertTrue(all(isinstance(d, Fraction) for d in depths))

    def test_depth_sign(self):
        self.assertEqual(R.depth_fraction(KNEE), 0)
        self.assertEqual(R.depth_fraction(-KNEE), 0)
        self.assertGreater(R.depth_fraction(-UP), 0)
        self.assertLess(R.depth_fraction(DOWN), 0)
        self.assertEqual(R.depth_fraction(F32(-0.0)), -Fraction(float(KNEE)))
        with self.assertRaises(ValueError):
            R.depth_fraction(np.float64(0.9))
        with self.assertRaises(ValueError):
            R.depth_fraction(F32(np.nan))

    def test_deep_threshold_is_exact(self):
        # the smallest float32 with exact depth >= 1/10 is 0xBF666667 in magnitude
        sweep = float32_band(0.89, 0.91)
        deep = R.in_deep(rows(sweep))
        first = int(np.argmax(deep))
        self.assertEqual(int(sweep[first].view(np.uint32)), 0x3F666667)
        self.assertTrue(np.all(deep[first:]))
        self.assertFalse(np.any(deep[:first]))


class LoadBandTest(unittest.TestCase):
    def test_edges(self):
        e04 = np.float64(0.4)
        b04 = np.nextafter(e04, 0.0)
        ulp04 = np.spacing(e04)
        self.assertEqual(ulp04, 2.0 ** -54)
        e10 = np.float64(1.0)
        b10 = np.nextafter(e10, 0.0)
        # The float64 0.4 and its predecessor are not reachable as float64(x) + float64(y) for float32 x, y: the
        # nearest reachable sums are 2 ulps below and 6 ulps above (a whole-range search found 0 exact pairs). The
        # search below confirms it near float32(0.4); the band function itself is tested at the exact edges in
        # test_exact_float64_edges, and the nearest reachable float32-pair sums are tested here.
        self.assertIsNone(find_pair(e04, 0.4, steps=2000, required=False))
        self.assertIsNone(find_pair(b04, 0.4, steps=2000, required=False))
        near_below = e04 - 2 * ulp04
        near_above = e04 + 6 * ulp04
        pairs = {
            "0.4 - 2 ulp (nearest reachable below)": (find_pair(near_below, 0.4), near_below, R.LOAD_LOW),
            "0.4 + 6 ulp (nearest reachable above)": (find_pair(near_above, 0.4), near_above, R.LOAD_MID),
            "at 1.0": (find_pair(e10, 0.75), e10, R.LOAD_DEEP),
            "at 1.0 (1 + 0)": ((F32(1.0), F32(0.0)), e10, R.LOAD_DEEP),
            "below 1.0": (find_pair(b10, 1.0), b10, R.LOAD_MID),
        }
        for name, ((x, y), target, code) in pairs.items():
            with self.subTest(name):
                self.assertEqual(x.dtype, F32)
                self.assertEqual(y.dtype, F32)
                obs = obs_with(x, y)
                g = R.load_g(obs)
                self.assertEqual(g.dtype, np.float64)
                self.assertEqual(g, target)                       # the sum is what we claim
                self.assertEqual(Fraction(float(g)), Fraction(float(target)))
                self.assertEqual(R.load_band(obs), code)
                self.assertEqual(R.load_band(obs).dtype, np.uint8)
        self.assertEqual(Fraction(float(b04)), Fraction(0.4) - Fraction(1, 2 ** 54))
        self.assertEqual(Fraction(float(b10)), 1 - Fraction(1, 2 ** 53))

    def test_exact_float64_edges(self):
        g = np.array([np.nextafter(0.4, 0.0), 0.4, np.nextafter(1.0, 0.0), 1.0, 0.0, 2.5])
        np.testing.assert_array_equal(R.band_of_g(g), [0, 1, 1, 2, 0, 2])
        # Fraction(0.4) > 2/5, and the float64 just below 0.4 is below 2/5 too, so the choice of edge never
        # changes a band for a float64 g; the edge used is still the float64 value, as the protocol says
        self.assertGreater(Fraction(0.4), Fraction(2, 5))
        self.assertLess(Fraction(float(np.nextafter(0.4, 0.0))), Fraction(2, 5))
        with self.assertRaises(ValueError):
            R.band_of_g(np.array([0.4], np.float32))

    def test_bands_and_shapes(self):
        obs = obs_with(np.array([[0.0, 0.3, 0.2], [0.9, 1.2, 0.5]], F32),
                       np.array([[0.1, 0.1, 0.8], [-0.2, 0.3, -0.6]], F32), batch=(2, 3))
        bands = R.load_band(obs)
        self.assertEqual(bands.shape, (2, 3))
        np.testing.assert_array_equal(bands, [[0, 1, 2], [1, 2, 0]])

    def test_raises(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            obs = obs_with(0.5, 0.1)
            obs[3] = bad_value
            with self.assertRaises(ValueError):
                R.load_band(obs)
        with self.assertRaises(ValueError):
            R.load_band(obs_with(0.5, 0.1).astype(np.float64))
        with self.assertRaises(ValueError):
            R.load_band(np.zeros(13, F32))
        with self.assertRaises(ValueError):
            R.band_of_g(np.array([np.nan]))


class CanonicalAgreementTest(unittest.TestCase):
    def test_canonical_edges(self):
        a = rows([KNEE, -KNEE, UP, -UP, DOWN, 1.0, -1.0, 0.0])
        np.testing.assert_array_equal(R.canonical_stalled_a0(a),
                                      [False, False, True, True, False, True, True, False])

    def test_all_float32_near_knee(self):
        magnitudes = float32_band(0.79, 0.81)
        values = np.concatenate([magnitudes, -magnitudes])
        rng = np.random.default_rng(20261030)
        actions = np.empty((values.size, 4), F32)
        actions[:, 0] = values
        actions[:, 1:] = rng.uniform(-1, 1, size=(values.size, 3)).astype(F32)
        bit_mask, depth_mask, canonical_mask = R.three_checks(actions)
        disagreements = int(np.count_nonzero((bit_mask != depth_mask) | (bit_mask != canonical_mask)))
        print(f"\n[region_independent] float32 values checked: {values.size}; disagreements: {disagreements}",
              file=sys.stderr)
        self.assertEqual(values.size, 671090)
        self.assertEqual(disagreements, 0)
        mask = R.agree(actions)
        expected_in = 2 * int(np.count_nonzero(magnitudes > KNEE))
        self.assertEqual(int(mask.sum()), expected_in)

    def test_agree_raises_on_disagreement(self):
        a = rows([0.5, UP])
        np.testing.assert_array_equal(R.agree(a), [False, True])
        with mock.patch.object(R, "canonical_stalled_a0", lambda x: np.array([False, False])):
            with self.assertRaises(RuntimeError):
                R.agree(a)
        with mock.patch.object(R, "in_region_by_depth", lambda x: np.array([True, True])):
            with self.assertRaises(RuntimeError):
                R.agree(a)


if __name__ == "__main__":
    unittest.main()
