"""Formal G2 v2 tests (Windows; pure functions plus two short prefix pairs; no collection)."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g2_formal_v2 as G2                                   # noqa: E402  (imports v1's driver first)
import numpy as np                                          # noqa: E402

import g2_seeds_v2 as GS2                                   # noqa: E402
import prefix_rule as PR                                    # noqa: E402
import stage1_eval as E                                     # noqa: E402
from knee_region import past_knee_a0                        # noqa: E402

G = G2.G
GS1 = G.GS


class Seeds(unittest.TestCase):
    def test_namespace_distinct_from_v1_and_development(self):
        v2 = GS2.build()
        v1 = GS1.build()
        self.assertEqual(GS2.NAMESPACE, "knee_s10_candidate_v2/")
        self.assertEqual(set(v2["numpy"]), set(v1["numpy"]))
        self.assertFalse(set(v2["numpy"].values()) & set(v1["numpy"].values()))
        self.assertFalse(set(v2["numpy"].values()) & set(GS2.DEVELOPMENT_VALUES))
        self.assertEqual(v2["banks"], v1["banks"])
        self.assertEqual(v2["episodes"], 1280)
        self.assertTrue(all(0 < x < 2 ** 63 for x in v2["numpy"].values()))

    def test_bank_items_differ_from_v1(self):
        self.assertNotEqual(GS2.bank_item("g2_e2_anchors", 3)["reset_seed"], GS1.bank_item("g2_e2_anchors", 3)["reset_seed"])
        self.assertFalse(np.array_equal(GS2.prefix_uniforms("g2_e2_anchors", 40, 25), GS1.prefix_uniforms("g2_e2_anchors", 40, 25)))
        self.assertEqual([GS2.bank_item("g2_e2_anchors", i)["t"] for i in (0, 32, 64, 96)], [0, 25, 50, 75])
        self.assertEqual(GS2.gated_items(), list(range(96)))

    def test_audit_catches_v1_formal_values(self):
        record = GS2.build()
        self.assertTrue(GS2.audit(record, prior={12345})["clean"])
        self.assertFalse(GS2.audit(record, prior={record["numpy"]["dataset_entropy"]})["clean"])
        derived = GS2.audit(record, prior={GS2.numpy_seed("g2_e2_bank")}, episodes=130)
        self.assertFalse(derived["clean"])

    def test_sealed_file_matches_derivation_when_present(self):
        if GS2.SEEDS_PATH.is_file():
            record = GS2.load()
            self.assertTrue(record["audit"]["clean"])
            self.assertFalse(set(record["numpy"].values()) & set(GS1.load()["numpy"].values()))


class Bounds(unittest.TestCase):
    def test_bounds_are_v1s_byte_for_byte(self):
        raw = G2.BOUNDS.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), G2.BOUNDS_SHA)
        self.assertEqual(raw, (G2.V1F / "d4_bounds.json").read_bytes())
        record = json.loads(raw)
        self.assertEqual(record["gated"], ["dens_mid", "dens_deep"])


class Approval(unittest.TestCase):
    def test_words_name_v2_and_attempt_rules(self):
        ok = {"approval": G2.WORDS + ", attempt 1", "package_manifest_sha256": "abc", "attempt": 1}
        self.assertEqual(G2.check_approval(ok, "abc", []), 1)
        v1_words = {**ok, "approval": G.WORDS + ", attempt 1"}
        for bad in (v1_words, {**ok, "package_manifest_sha256": "xyz"}, {**ok, "attempt": 2}):
            with self.assertRaises(Exception):
                G2.check_approval(bad, "abc", [])
        two = {**ok, "attempt": 2}
        self.assertEqual(G2.check_approval(two, "abc", [{"headline": "INCOMPLETE", "manifest_sha256": "abc"}]), 2)
        for last in ({"headline": "PASS"}, {"headline": "NOT DATA-OOD"}, {"headline": "NOT TESTABLE"},
                     {"headline": "INCONCLUSIVE", "inconclusive_cause": "D4"}):
            with self.assertRaises(Exception):
                G2.check_approval(two, "abc", [{**last, "manifest_sha256": "abc"}])
        with self.assertRaises(Exception):
            G2.check_approval({**ok, "attempt": 3}, "abc", [{"headline": "INCOMPLETE", "manifest_sha256": "abc"}] * 2)


class Anchors(unittest.TestCase):
    def test_sealed_anchor_identical_in_both_worlds(self):
        item = GS2.gated_items()[40]                                           # t = 25 block
        a = G2.build_anchor(item)
        self.assertEqual(a["t"], 25)
        self.assertTrue(a["both_worlds_identical"])
        self.assertEqual(a["prefix_actions"].shape, (25, 4))
        self.assertFalse(np.any(past_knee_a0(a["prefix_actions"])[0]))
        self.assertTrue(np.array_equal(a["obs"], G2.build_anchor(item)["obs"]))

    def test_driver_prefix_equals_the_development_builder(self):
        stream = 707_070_707
        item = next(i for i in range(64) if E.stream_item("stage1_e2_anchors", i, stream)["t"] == 25)
        spec = E.stream_item("stage1_e2_anchors", item, stream)
        uniforms = np.random.Generator(np.random.PCG64(E._smoke_sequence(stream, "prefix", "stage1_e2_anchors", item))).random((25, 5))
        run = G2.run_prefix_both_worlds(spec["reset_seed"], spec["current"], uniforms)
        dev = PR.build_anchor("stage1_e2_anchors", item, stream)
        self.assertTrue(np.array_equal(run["prefix"], dev["prefix_actions"]))
        self.assertTrue(np.array_equal(run["trace"][-1], dev["obs"]) and np.array_equal(run["state"], dev["state"]))

    def test_start_rows(self):
        rows = G2.start_rows()
        self.assertEqual(len(rows), 128)
        self.assertEqual(rows[0]["prefix_actions"].shape, (0, 4))


class Labels(unittest.TestCase):
    def test_precedence_via_v1_driver(self):
        d = {w: {"D1": {"ok": True}, "D2": {"ok": True}, "D3": {"ok": w == "stall3"},
                 "D4": {"below_lower": False, "above_upper": False}, "B1": {"ok": True}} for w in ("stall3", "nostall")}
        h = {"H1": {"ok": True}, "H2": {"ok": False}}
        self.assertEqual(G.labels(d, h)[0], "NOT DATA-OOD")


class Manifest(unittest.TestCase):
    def test_manifest_matches_when_present(self):
        if G2.MANIFEST.is_file():
            G2.verify_manifest()
            names = set(G2.manifest_files())
            for needed in ("G2_FORMAL_RULE_v2.md", "g2_formal_v2.py", "g2_seeds_v2.json", "d4_bounds.json", "../prefix_rule.py",
                           "../../knee_s10_candidate_v1/formal_g2/g2_formal.py", "../../knee_s10_candidate_v1/pi_ref.py",
                           "../../knee_s10_candidate_v1/recorder_wide.py"):
                self.assertIn(needed, names)


if __name__ == "__main__":
    unittest.main()
