"""Formal G2 tests (Windows; pure functions, no collection, no simulator continuation)."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import d4_bounds as DB
import g2_seeds as GS


def _driver():
    import g2_formal
    return g2_formal


class Seeds(unittest.TestCase):
    def test_names_distinct_and_deterministic(self):
        a, b = GS.build(), GS.build()
        self.assertEqual(a, b)
        values = list(a["numpy"].values())
        self.assertEqual(len(set(values)), len(values))
        self.assertTrue(all(0 < v < 2 ** 63 for v in values))

    def test_bank_schedule_and_gated_items(self):
        ts = [GS.bank_item("g2_e2_anchors", i)["t"] for i in range(128)]
        self.assertEqual([ts.count(t) for t in (0, 25, 50, 75)], [32, 32, 32, 32])
        self.assertEqual(GS.gated_items(), list(range(96)))
        self.assertEqual(GS.bank_item("g2_e2_bank", 5)["current"], (0.9, 1.1))
        self.assertNotIn("t", GS.bank_item("g2_e2_bank", 5))
        with self.assertRaises(ValueError):
            GS.bank_item("g2_e2_bank", 128)

    def test_prefix_uniforms_shape_and_determinism(self):
        u = GS.prefix_uniforms("g2_e2_anchors", 40, 25)
        self.assertEqual(u.shape, (25, 5))
        self.assertTrue(np.array_equal(u, GS.prefix_uniforms("g2_e2_anchors", 40, 25)))
        self.assertFalse(np.array_equal(u, GS.prefix_uniforms("g2_e2_anchors", 41, 25)))

    def test_audit_catches_raw_and_derived_collisions(self):
        record = GS.build()
        clean = GS.audit(record, prior={12345})
        self.assertTrue(clean["clean"])
        raw = GS.audit(record, prior={record["numpy"]["dataset_entropy"]})
        self.assertFalse(raw["clean"]) and self.assertEqual(raw["raw_collisions"], [record["numpy"]["dataset_entropy"]])
        derived = GS.audit(record, prior={GS.numpy_seed("g2_e2_bank")}, episodes=130)   # bank root under key 1901
        self.assertFalse(derived["clean"])
        self.assertTrue(derived["derived_reset_seed_collisions"])

    def test_sealed_file_matches_derivation_when_present(self):
        if GS.SEEDS_PATH.is_file():
            record = GS.load()
            self.assertTrue(record["audit"]["clean"])
            self.assertNotIn(record["numpy"]["dataset_entropy"], GS.DEVELOPMENT_VALUES)


class Bounds(unittest.TestCase):
    def test_density_matches_design_and_bounds_bracket(self):
        stats = DB.simulate(reps=20, seed=1)
        designed = DB.WINDOW_PROBABILITY / DB.WIDTH + (1 - DB.WINDOW_PROBABILITY) / 2
        self.assertAlmostEqual(float(stats["dens_mid"].mean()), designed, delta=0.08)
        self.assertAlmostEqual(float(stats["dens_deep"].mean()), designed, delta=0.2)
        b = DB.bounds(stats)
        for k in ("dens_mid", "dens_deep"):
            self.assertLess(b[k]["lower"], b[k]["mean"])
            self.assertGreater(b[k]["upper"], b[k]["mean"])
            self.assertLessEqual(b[k]["lower"], b[k]["mean"] - 6 * b[k]["sd"])
            self.assertGreaterEqual(b[k]["upper"], b[k]["mean"] + 6 * b[k]["sd"])

    def test_sealed_bounds_file_when_present(self):
        if DB.OUT.is_file():
            record = json.loads(DB.OUT.read_text(encoding="utf-8"))
            self.assertEqual(record["gated"], ["dens_mid", "dens_deep"])
            self.assertLess(record["bounds"]["dens_mid"]["lower"], 2.46)
            self.assertGreater(record["bounds"]["dens_mid"]["upper"], 2.46)


class Rule(unittest.TestCase):
    def d(self, **over):
        base = {"D1": {"ok": True}, "D2": {"ok": True}, "D3": {"ok": True},
                "D4": {"below_lower": False, "above_upper": False}, "B1": {"ok": True}}
        base.update(over)
        return {"stall3": base, "nostall": json.loads(json.dumps(base))}

    def h(self, h1=True, h2=True):
        return {"H1": {"ok": h1}, "H2": {"ok": h2}}

    def test_labels_and_precedence(self):
        G = _driver()
        self.assertEqual(G.labels(self.d(), self.h()), ("PASS", ["PASS"], None))
        self.assertEqual(G.labels(self.d(), self.h(h2=False))[0], "NOT TESTABLE")
        self.assertEqual(G.labels(self.d(D2={"ok": False}), self.h(h2=False))[0], "NOT DATA-OOD")
        head, all_labels, cause = G.labels(self.d(D4={"below_lower": True, "above_upper": False}), self.h(h1=False, h2=False))
        self.assertEqual((head, cause), ("INCONCLUSIVE", "H1"))
        self.assertEqual(len(all_labels), 2)
        head, _, cause = G.labels(self.d(D4={"below_lower": False, "above_upper": True}), self.h())
        self.assertEqual((head, cause), ("INCONCLUSIVE", "D4"))

    def test_d4_two_sided(self):
        G = _driver()
        bounds = {"bounds": {"dens_mid": {"lower": 2.3, "upper": 2.6}, "dens_deep": {"lower": 2.1, "upper": 2.9}, "dens_hi": {"lower": 0, "upper": 9}}}
        rng = np.random.default_rng(0)
        bands = np.repeat([1, 2], [4000, 1000])
        a0 = np.where(rng.random(5000) < 0.246, rng.uniform(-0.95, -0.85, 5000), -0.7)
        d4 = G.d4_check(bands, a0, bounds)
        self.assertTrue(abs(d4["mid"]["density"] - 2.46) < 0.3)
        self.assertFalse(d4["mid"]["below"] or d4["mid"]["above"])
        self.assertFalse(d4["hi"]["gated"])
        tight = {"bounds": {k: {"lower": 2.459, "upper": 2.461} for k in ("dens_mid", "dens_deep", "dens_hi")}}
        d4 = G.d4_check(bands, a0, tight)
        self.assertTrue(d4["mid"]["below"] or d4["mid"]["above"])

    def test_approval_rules(self):
        G = _driver()
        ok = {"approval": G.WORDS + ", attempt 1", "package_manifest_sha256": "abc", "attempt": 1}
        self.assertEqual(G.check_approval(ok, "abc", []), 1)
        for bad in ({**ok, "approval": "I approve something else"}, {**ok, "package_manifest_sha256": "xyz"}, {**ok, "attempt": 2}):
            with self.assertRaises(Exception):
                G.check_approval(bad, "abc", [])
        two = {**ok, "attempt": 2}
        self.assertEqual(G.check_approval(two, "abc", [{"headline": "INCOMPLETE", "manifest_sha256": "abc"}]), 2)
        fixed = {**two, "package_manifest_sha256": "def"}                 # one code-only fix after an H1 fault
        self.assertEqual(G.check_approval(fixed, "def", [{"headline": "INCONCLUSIVE", "inconclusive_cause": "H1", "manifest_sha256": "abc"}]), 2)
        for last in ({"headline": "PASS"}, {"headline": "NOT TESTABLE"}, {"headline": "INCONCLUSIVE", "inconclusive_cause": "D4"}):
            with self.assertRaises(Exception):
                G.check_approval(two, "abc", [{**last, "manifest_sha256": "abc"}])
        with self.assertRaises(Exception):                                   # two code changes
            G.check_approval(two, "ghi", [{"headline": "INCOMPLETE", "manifest_sha256": "abc"}, {"headline": "INCOMPLETE", "manifest_sha256": "def"}])
        with self.assertRaises(Exception):                                   # no attempt 3
            G.check_approval({**ok, "attempt": 3}, "abc", [{"headline": "INCOMPLETE", "manifest_sha256": "abc"}] * 2)

    def test_in_cutoff_support(self):
        G = _driver()
        obs = np.zeros((3, 14), np.float32)
        obs[:, 10] = [1.0, 1.0, 0.1]                                          # load g = obs10 + obs11: high, high, low
        a = np.array([[-0.7, 0.5, 0, 0], [-0.5, 0, 0, 0], [0.75, 0, 0.71, 0]], np.float32)
        self.assertEqual(G.in_cutoff_support(obs, a).tolist(), [True, False, False])

    def test_budget_caps(self):
        G = _driver()
        b = G.Budget(100, 1000)
        b.charge("a", 60)
        with self.assertRaises(G.CapExhausted):
            b.charge("b", 50)
        self.assertEqual(b.used, 60)


if __name__ == "__main__":
    unittest.main()
