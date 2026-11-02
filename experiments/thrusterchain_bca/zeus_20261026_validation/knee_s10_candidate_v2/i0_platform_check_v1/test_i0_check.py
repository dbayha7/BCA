"""Tests of the I0 rehearsal's pure decision function (Windows; nothing is simulated)."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import i0_check as C                                          # noqa: E402  (imports nothing from the screen at import time)


class Compare(unittest.TestCase):
    def test_exact_match_passes(self):
        r = C.compare(1.7652868718841468, 1.7652868718841468, 0.0)
        self.assertTrue(r["ok"])
        self.assertEqual(r["rel_diff"], 0.0)

    def test_tolerance_is_relative_to_max_one_and_formal(self):
        self.assertTrue(C.compare(1.0 + 5e-10, 1.0, 0.0)["ok"])
        self.assertFalse(C.compare(1.0 + 2e-9, 1.0, 0.0)["ok"])
        self.assertTrue(C.compare(100.0 + 5e-8, 100.0, 0.0)["ok"])       # 1e-9 relative at 100
        self.assertFalse(C.compare(100.0 + 2e-7, 100.0, 0.0)["ok"])

    def test_h1_failure_fails_even_when_median_matches(self):
        self.assertFalse(C.compare(1.0, 1.0, 2e-9)["ok"])

    def test_matches_g1_unit_rule(self):
        # g1_unit.run: ok = |m - f| <= 1e-9 * max(1, |f|) and h1 <= 1e-9
        for m, f, h in ((0.3, 0.3, 1e-9), (0.3, 0.3 + 1e-9, 0.0), (0.3, 0.3 + 1.0000001e-9, 0.0)):
            expected = abs(m - f) <= 1e-9 * max(1.0, abs(f)) and h <= 1e-9
            self.assertEqual(C.compare(m, f, h)["ok"], expected)


if __name__ == "__main__":
    unittest.main()
