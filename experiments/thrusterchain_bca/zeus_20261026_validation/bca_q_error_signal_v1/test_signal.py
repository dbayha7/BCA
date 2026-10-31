"""Q-error signal check v1 tests (Windows; pure functions only, nothing restored or run)."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import numpy as np

import signal_check as SC


class Bins(unittest.TestCase):
    def test_bins_and_contrast(self):
        a0 = np.array([-0.75, -0.72, -0.85, -0.88, -0.95, -0.85, 0.0])
        high = np.array([True, True, True, True, True, False, True])
        masks = SC.bins(a0, high)
        self.assertEqual([int(m.sum()) for m in masks.values()], [2, 2, 1, 1])
        delta = np.array([0.1, 0.3, 1.0, 0.8, 2.0, 5.0, 9.0])
        s = SC.bin_stats(delta, masks)
        self.assertAlmostEqual(s["I_w"], 0.9 - 0.2)
        self.assertEqual(s["deep_high"]["rows"], 1)

    def test_knee_boundary_is_in_support(self):
        a0 = np.array([np.float32(-0.8), np.nextafter(np.float32(-0.8), np.float32(-1))], np.float32)
        masks = SC.bins(a0, np.array([True, True]))
        self.assertEqual((bool(masks["edge_high"][0]), bool(masks["onset_high"][1])), (True, True))


class Decision(unittest.TestCase):
    def stats(self, d_by_seed):
        out = {}
        for i in SC.SEEDS:
            out[f"s{i}_random_stall3"] = {"020000": {"I_w": 0.1 + d_by_seed[i]}}
            out[f"s{i}_random_nostall"] = {"020000": {"I_w": 0.1}}
        return out

    def test_signal_needs_two_of_three(self):
        self.assertEqual(SC.decide(self.stats({0: 0.5, 1: 0.25, 2: 0.0}))[0], "SIGNAL")
        self.assertEqual(SC.decide(self.stats({0: 0.5, 1: 0.1, 2: 0.0}))[0], "NO SIGNAL")
        self.assertEqual(SC.decide(self.stats({0: 0.2, 1: 0.2, 2: -1.0}))[0], "SIGNAL")        # boundary passes

    def test_missing_bin_is_invalid(self):
        s = self.stats({0: 0.5, 1: 0.5, 2: 0.5})
        s["s1_random_stall3"]["020000"]["I_w"] = None
        self.assertEqual(SC.decide(s)[0], "INVALID")


class Delta1(unittest.TestCase):
    def test_one_step_error_formula(self):
        class Policy:
            def q(self, obs, act):
                v = obs[:, 0] + act[:, 0]
                return np.stack([v, v - 1.0], axis=1)

            def act(self, obs):
                return np.full((len(obs), 4), 0.5, np.float32)
        obs = np.array([[1.0] + [0] * 13, [2.0] + [0] * 13], np.float32)
        t = SimpleNamespace(obs=obs, action=np.array([[0.2, 0, 0, 0], [-0.1, 0, 0, 0]], np.float32),
                            reward=np.array([-1.0, -2.0]), next_obs=obs + 1, done=np.array([0.0, 1.0]))
        d = SC.delta1(Policy(), t, 0.98)
        self.assertAlmostEqual(d[0], (1.2) - (-1.0) - 0.98 * (2.0 + 0.5), places=6)          # Q1 head, online actor
        self.assertAlmostEqual(d[1], (1.9) - (-2.0), places=6)                                # done: no bootstrap


if __name__ == "__main__":
    unittest.main()
