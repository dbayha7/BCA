"""Unauthenticated synthetic attempts cannot inherit a historical counter correction. JSON only.

The former fixture was accepted without producer evidence. It must now be rejected without
changing its files; the pinned completed-input path is covered in test_guard_accounting.py.
"""
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
    def test_rejects_unversioned_attempt_without_changing_it(self):
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
            with self.assertRaisesRegex(ValueError, "Unsupported attempt"):
                AA.analyse(attempt)
            self.assertEqual({p: p.read_bytes() for p in attempt.rglob("*.json")}, before)     # read-only
            for unit, directory in dirs.items():
                body = json.loads((Path(directory) / "report.json").read_text())
                correction = AA.corrected_steps(unit, body, 1280)
                self.assertIsNone(correction["corrected"], unit)   # apparent shape/counts are not provenance
                self.assertIsNone(correction["loads"], unit)       # no guessed dataset-load count
            self.assertEqual({p: p.read_bytes() for p in attempt.rglob("*.json")}, before)


if __name__ == "__main__":
    unittest.main()
