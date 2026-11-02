"""amended_analysis.py on a synthetic Stage 1 attempt (review finding: no test covered it). JSON only."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import amended_analysis as AA
import stage1
from test_stage1 import synthetic_eval


class AmendedAnalysis(unittest.TestCase):
    def test_reads_an_attempt_without_changing_it(self):
        with tempfile.TemporaryDirectory() as root:
            attempt = Path(root) / "stage1_run" / "attempt_1"
            dirs = {}
            for i in stage1.SEEDS:
                d = attempt / f"evaluate_s{i}"
                d.mkdir(parents=True)
                body = {**synthetic_eval(i), "simulator_steps_exact": 5_000 + 3 * 1280, "bc_only": {"J": 1.0}}
                (d / "report.json").write_text(json.dumps(body), encoding="utf-8")
                dirs[d.name] = str(d)
            for name, body in (("r0", {"kind": "r0", "seeds": {}, "adequacy_ok": True, "fail_early": False,
                                       "simulator_steps_exact": 900 + 1280}),
                               ("warmup_s0_cutoff_stall3", {"kind": "warmup", "simulator_steps_exact": 1280}),
                               ("branch_s0_cutoff_stall3", {"kind": "branch"})):
                d = attempt / name
                d.mkdir()
                (d / "report.json").write_text(json.dumps(body), encoding="utf-8")
                dirs[name] = str(d)
            (attempt / "report.json").write_text(json.dumps({"label": "PASS", "unit_dirs": dirs, "a3": {}}), encoding="utf-8")
            before = {p: p.read_bytes() for p in attempt.rglob("*.json")}
            out = AA.analyse(attempt)
            self.assertEqual({p: p.read_bytes() for p in attempt.rglob("*.json")}, before)     # read-only
            self.assertEqual(out["original"]["label"], "PASS")
            self.assertIn("V0f", out["R1_v3_rules"]["validity_by_seed"][0])
            units = out["R3_accounting"]["units"]
            self.assertEqual(units["warmup_s0_cutoff_stall3"]["corrected"], 128_000)
            self.assertEqual(units["evaluate_s0"]["loads"], 3)                               # two worlds plus BC-only
            self.assertEqual(units["evaluate_s0"]["corrected"], 5_000 + 3 * 128_000)
            self.assertEqual(units["r0"]["corrected"], 900 + 128_000)
            self.assertIsNone(units["branch_s0_cutoff_stall3"]["corrected"])                # no count: unknown, not zero
            self.assertEqual(out["R3_accounting"]["totals"]["units_without_counts"], ["branch_s0_cutoff_stall3"])
            own = out["R2_own_reference"][0]
            self.assertTrue(all(all(v.values()) for v in own["decomposition_identity_ok"].values()))
            self.assertEqual(own["continuation_gap_contrast"]["s05"]["median_D_C_minus_D_R"], 0.0)
            self.assertIn("sha256", out["sources"]["attempt_report"])


if __name__ == "__main__":
    unittest.main()
