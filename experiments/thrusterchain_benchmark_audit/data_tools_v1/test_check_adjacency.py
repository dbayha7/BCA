import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

from check_adjacency import audit


class AdjacencyCheckTest(unittest.TestCase):
    def test_filtered_step_and_episode_boundary(self):
        # Original episode 0 has steps 0,1,2,3; filtered step 2 is missing.
        # Step 3 ends the episode, then episode 1 starts at observation 50.
        arrays = dict(observations=np.array([[0], [1], [3], [50]]),
                      next_observations=np.array([[1], [2], [4], [51]]),
                      actions=np.zeros((4, 1)), rewards=np.zeros(4),
                      terminals=np.zeros(4), timeouts=np.array([0, 0, 1, 0]))
        result = audit(arrays)
        self.assertEqual(result["interior_successor_mismatches"], 1)
        self.assertEqual(result["first_mismatch_rows"], [1])
        self.assertEqual(result["boundary_rows_excluded"], 1)
        self.assertEqual(result["final_row_without_following_stored_row"], 1)

    def test_equality_is_not_provenance(self):
        arrays = dict(observations=np.array([[0], [1]]), next_observations=np.array([[1], [2]]),
                      actions=np.zeros((2, 1)), rewards=np.zeros((2, 1)),
                      terminals=np.zeros(2), timeouts=np.zeros(2))
        result = audit(arrays)
        self.assertEqual(result["interior_successor_mismatches"], 0)
        self.assertIn("does not establish", result["interpretation"])

    def test_invalid_flags_and_shapes_are_refused(self):
        arrays = dict(observations=np.zeros((2, 1)), next_observations=np.zeros((2, 1)),
                      actions=np.zeros((2, 1)), rewards=np.zeros(2),
                      terminals=np.array([0, 2]), timeouts=np.zeros(2))
        with self.assertRaises(ValueError):
            audit(arrays)
        arrays["terminals"] = np.zeros(2)
        arrays["rewards"] = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            audit(arrays)

    def test_hdf5_cli_reports_mismatch_and_does_not_change_file(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is not installed")
        import hashlib
        arrays = dict(observations=np.array([[0], [3]]), next_observations=np.array([[1], [4]]),
                      actions=np.zeros((2, 1)), rewards=np.zeros(2),
                      terminals=np.zeros(2), timeouts=np.zeros(2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.hdf5"
            with h5py.File(path, "w") as f:
                for name, array in arrays.items():
                    f.create_dataset(name, data=array)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            script = Path(__file__).with_name("check_adjacency.py")
            result = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('"interior_successor_mismatches": 1', result.stdout)
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
if __name__ == "__main__":
    unittest.main()
