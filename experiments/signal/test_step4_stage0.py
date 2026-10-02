"""Checks of step 4's Stage 0 (step4_stage0.py): blindness, the evidence-set rules, the outputs and their hash.

Blindness: step4_stage0's source (and the runner's and harness's it imports) never imports or reads lq_harness.value /
policy_gradient or a module that computes J (test_run_step4.j_references), and Stage 0 is run here with both J
functions replaced by functions that raise. Tiny settings with seed 7, not the real configuration; no J anywhere.
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_step4_stage0
"""
import ast
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.signal import placement_harness as PH
from experiments.signal import run_step4 as RUN
from experiments.signal import score_values as SV
from experiments.signal import step4_stage0 as S0
from experiments.signal.test_run_step4 import TINY, NoJ, SIGNAL, j_references


class SourceTests(unittest.TestCase):
    def test_stage0_never_references_a_J_function(self):
        self.assertEqual(j_references(SIGNAL / "step4_stage0.py"), [])

    def test_stage0_imports_only_blind_modules(self):
        tree = ast.parse((SIGNAL / "step4_stage0.py").read_text(encoding="utf8"))
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        self.assertEqual(imported & {"lq_harness", "placement_harness", "run_step4"},
                         {"lq_harness", "placement_harness", "run_step4"})
        self.assertFalse(imported & {"score_values", "step4_settings", "scorecard_step4", "value", "policy_gradient"})


class SetRuleTests(unittest.TestCase):
    def test_rules_and_boundaries(self):
        f = S0.evidence_set
        self.assertEqual(f("X-CS", "clean", 1.0, 0.0), "N")
        self.assertEqual(f("C", "clean", 1.0, 0.0), "N")
        self.assertEqual(f("X-CS", "cone", 0.1, 1.0), "P")
        self.assertEqual(f("C", "tilt-local", 0.99, 0.0), "P")
        self.assertEqual(f("R", "q2_optimistic", 1.0, 0.0), "check")
        self.assertEqual(f("X-CS", "tilt", 0.5, 0.01), "M")  # both boundaries inclusive
        self.assertEqual(f("X-CS", "tilt", 0.5, 0.0099), "H")
        self.assertEqual(f("X-CS", "tilt", 0.5000001, 0.02), "W")
        self.assertEqual(f("X-CS", "tilt", 0.9, 0.02), "H")  # W is open at 0.9
        self.assertEqual(f("R", "shared_bias", -0.9, math.inf), "M")  # an unbounded NONE counts as d_crit >= 0.01
        self.assertEqual(f("R", "q1_optimistic", None, 0.5), "H")

    def test_sets_average_over_analysis_replicates_only(self):
        rows = [dict(block="X-CS", cell="independent_errors", case="independent_errors", level="expert", replicate=r,
                     alignment_K=a, d_crit=d, cos_pull_ascent=0.5)
                for r, a, d in ((0, 0.4, 0.02), (1, 0.6, 0.02), (99, -1.0, 9.0))]
        sets = S0.evidence_sets(rows)
        entry = sets["X-CS"]["independent_errors"]["expert"]
        self.assertAlmostEqual(entry["mean_alignment_K"], 0.5)
        self.assertEqual((entry["set"], entry["M_nat"], entry["replicates"]), ("M", True, [0, 1]))
        rows[0]["d_crit"] = math.inf
        self.assertEqual(S0.evidence_sets(rows)["X-CS"]["independent_errors"]["expert"]["unbounded"], 1)
        rows = [dict(r, block="C", cell="q1opt-local_k1", case="q1opt-local") for r in rows]
        self.assertEqual(S0.evidence_sets(rows)["C"]["q1opt-local_k1"]["expert"]["set"], "P")

    def test_context_k_gradient_blocks_sum_to_the_plain_one(self):
        rng = np.random.default_rng(3)
        g, s, c = rng.normal(size=(50, 2)), rng.normal(size=(50, 3)), rng.integers(4, size=50)
        np.testing.assert_allclose(S0.k_gradient(g, s, c, 4).sum(0), PH.k_gradient(g, s), rtol=1e-12, atol=1e-15)


class RunTests(unittest.TestCase):
    """Stage 0 on tiny settings for blocks R, X-CS and C, replicates 0, 1 and the pilot, J disabled."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="stage0_"))
        cls.path = cls.tmp / "stage0.json"
        with NoJ():
            S0.run(cls.path, ("R", "X-CS", "C"), (0, 1, RUN.PILOT_REPLICATE), TINY, log=lambda *a: None)
        cls.payload = json.loads(cls.path.read_text(encoding="utf8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_hash_and_refusal_to_overwrite(self):
        recorded = SV.read_hashes(str(self.path) + ".sha256")
        self.assertEqual(recorded, {"stage0.json": SV.sha256_file(self.path)})
        with self.assertRaises(FileExistsError):
            S0.run(self.path, ("C",), (0,), TINY, log=lambda *a: None)

    def test_checks_pass(self):
        names = {c["name"]: c for c in self.payload["checks"]}
        self.assertEqual(set(names), {"tilt_alignment", "tilt_T_identical_across_levels_xcs",
                                      "xcs_states_noise_shocks_identical_across_levels",
                                      "block_r_alignment_two_computations"})
        for c in names.values():
            self.assertTrue(c["ok"], c)

    def test_rows_cover_every_cell_level_and_replicate(self):
        rows = self.payload["rows"]
        expected = sum(len(RUN.grid(b, r)) * len(RUN.LEVELS[b]) for b in ("R", "X-CS", "C")
                       for r in (0, 1, RUN.PILOT_REPLICATE))
        self.assertEqual(len(rows), expected)
        for r in rows:
            self.assertIsNotNone(r["alignment_K"])
            if r["case"] == "clean":
                self.assertAlmostEqual(r["alignment_K"], 1.0, places=12)
                self.assertEqual(r["d_crit"], 0.0)  # Q1 = Q^pi: EXACT's and NONE's fixed points coincide
            if r["case"] == "cone":
                self.assertAlmostEqual(r["cone_mean"], PH.CONE_MEAN, places=9)
                self.assertTrue(-1 <= r["cone_cos"] <= 1)
            if r["case"] == "q2_optimistic":
                clean = next(x for x in rows if x["block"] == "R" and x["case"] == "clean" and x["level"] == r["level"]
                             and x["replicate"] == r["replicate"] and x["offset"] == r["offset"])
                self.assertEqual((r["alignment_K"], r["d_crit"]), (clean["alignment_K"], clean["d_crit"]))

    def test_sets_follow_the_rules_and_exclude_the_pilot(self):
        rows = self.payload["rows"]
        for block, cells in self.payload["sets"].items():
            for cid, levels in cells.items():
                for level, entry in levels.items():
                    self.assertEqual(entry["replicates"], [0] if entry["case"] == "q2_optimistic" else [0, 1])
                    vals = [r for r in rows if r["block"] == block and r["cell"] == cid and r["level"] == level
                            and r["replicate"] in (0, 1)]
                    d = [math.inf if r["d_crit"] is None else r["d_crit"] for r in vals]
                    self.assertAlmostEqual(entry["mean_alignment_K"], np.mean([r["alignment_K"] for r in vals]))
                    self.assertEqual(entry["set"], S0.evidence_set(block, entry["case"], entry["mean_alignment_K"],
                                                                   float(np.mean(d))))
        xcs = self.payload["sets"]["X-CS"]
        for a in ("0.5", "0", "-0.5"):
            for level in PH.QUALITIES:
                e = xcs[f"tilt_a{a}"][level]
                self.assertLessEqual(e["mean_alignment_K"], 0.5 + 1e-4)
                self.assertEqual(e["set"], "M" if e["mean_d_crit"] >= 0.01 else "H")


if __name__ == "__main__":
    unittest.main()
