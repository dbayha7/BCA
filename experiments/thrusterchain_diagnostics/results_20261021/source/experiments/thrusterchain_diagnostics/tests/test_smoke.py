import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.thrusterchain_diagnostics.__main__ import run_smoke, HERE


class SmokeTests(unittest.TestCase):
    def test_small_end_to_end_and_no_overwrite(self):
        config = json.loads((HERE / "configs" / "smoke.json").read_text())
        config["finite"]["horizon"] = 3
        config["continuous"]["horizon"] = 4
        config["trajectory_episodes"] = 4
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "new_run"
            report = run_smoke(out, config)
            self.assertTrue((out / "report.json").is_file())
            self.assertEqual(report["budgets"]["optimization_updates"], 0)
            self.assertTrue(all(report["finite"]["hidden_pair"]["coupled_log_checks"].values()))
            self.assertTrue((out / "evaluator" / "finite_holdout.npz").is_file())
            self.assertFalse((out / "learner" / "finite_holdout.npz").exists())
            with np.load(out / "learner" / "hidden_common_log.npz", allow_pickle=False) as rows:
                self.assertNotIn("world", rows.files)
                self.assertNotIn("optimal_Q", rows.files)
            before = (out / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_smoke(out, config)
            self.assertEqual(before, (out / "report.json").read_bytes())
            parsed = json.loads(before)
            self.assertIn("rare_probe", parsed["finite"])


if __name__ == "__main__":
    unittest.main()
