"""Dataset-bound CLI shift evaluation and explicit neural-budget checks."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from thrusterchain.quality_datasets import generate_behavior_dataset


ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run([sys.executable, "-B", "-m", "thrusterchain", *map(str, args)],
                          cwd=ROOT, text=True, capture_output=True)


class ShiftCLITests(unittest.TestCase):
    def test_checkpoint_bound_shift_evaluation_and_refusals(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, other, continuing = (root / "data.npz", root / "other.npz", root / "continuing.npz")
            dataset = generate_behavior_dataset(data, episodes=10, seed=611, behavior="feedback_strong")
            generate_behavior_dataset(other, episodes=10, seed=612, behavior="feedback_strong")
            generate_behavior_dataset(continuing, episodes=10, seed=613, behavior="feedback_strong",
                                      task_semantics="continuing_time_limit")
            trained = root / "model"
            run = cli("train", data, trained, "--adapter", "affine-sgd", "--offline-updates", 2, "--batch-size", 8)
            self.assertEqual(run.returncode, 0, run.stderr)
            out = root / "shift.json"
            run = cli("evaluate-shifts", trained, out, "--dataset", data,
                      "--scenarios", "baseline", "combined", "--final-seeds", 614)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(run.stdout)
            self.assertEqual(result["dataset_sha256"], dataset.digest)
            self.assertEqual(result["environment_steps"], 200)
            self.assertEqual(result["policy_sha256_before"], result["policy_sha256_after"])
            for bad, message in ((other, "checkpoint must be bound"), (continuing, "requires a finite-horizon")):
                path = root / (bad.stem + ".json")
                refused = cli("evaluate-shifts", trained, path, "--dataset", bad,
                              "--scenarios", "baseline", "--final-seeds", 614)
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn(message, refused.stderr)
                self.assertFalse(path.exists())

    def test_neural_adapters_require_explicit_update_budget_before_import(self):
        with TemporaryDirectory() as temp:
            proc = cli("train", Path(temp) / "absent.npz", Path(temp) / "model",
                       "--adapter", "unifloral-unified")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("explicit --offline-updates", proc.stderr)
            self.assertFalse((Path(temp) / "model").exists())


if __name__ == "__main__":
    unittest.main()
