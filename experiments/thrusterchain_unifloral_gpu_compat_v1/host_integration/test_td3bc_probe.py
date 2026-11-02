"""Task-specific TD3+BC bridge and numerical checks; no benchmark training."""
import unittest

import numpy as np

from thrusterchain.datasets import generate_dataset
from host_integration.td3bc_probe import prepare, run_probe


class TD3BCProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(episodes=10, seed=991, provenance="smoke")

    def test_task_batch_and_two_native_updates(self):
        receipt = run_probe(self.dataset, [0, 1, 99, 102, 199])
        self.assertEqual(receipt["batch_size"], 5)
        self.assertEqual(receipt["observation_dim"], 14)
        self.assertEqual(receipt["terminal_rows"], 2)
        self.assertEqual(receipt["counts"], {"critic": 2, "actor": 1, "critic_target": 1, "actor_target": 1})
        self.assertEqual(receipt["dataset_digest"], self.dataset.digest)
        for key, value in receipt["metrics"].items():
            self.assertTrue(np.isclose(value, receipt["independent_reductions"][key], rtol=2e-5, atol=2e-5))

    def test_selection_and_train_only_normalization(self):
        batch, mean, scale = prepare(self.dataset, [0, 99])
        train = self.dataset.split("train")
        obs = train["observations"][:, :-1].reshape(-1, 14).astype(np.float32)
        np.testing.assert_array_equal(mean, obs.mean(0))
        np.testing.assert_array_equal(scale, obs.std(0) + np.float32(.001))
        np.testing.assert_allclose(np.asarray(batch.next_obs[1]), (train["observations"][0, 100] - mean) / scale)
        self.assertEqual(float(batch.done[1]), 1.0)
        with self.assertRaises(ValueError):
            prepare(self.dataset, [-1])


if __name__ == "__main__":
    unittest.main()
