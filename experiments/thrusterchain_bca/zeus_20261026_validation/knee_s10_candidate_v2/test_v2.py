"""Tests of the v2 prefix rule (Windows; cheap: no collection, one short prefix pair)."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "knee_s10_candidate_v1" / "formal_g2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g2_formal as G                                       # noqa: E402  replaces `recorder` before any v3 import
import numpy as np                                          # noqa: E402

import prefix_rule as PR                                    # noqa: E402
import recorder_wide                                        # noqa: E402
import stage1_eval as E                                     # noqa: E402
from knee_region import KNEE, past_knee_a0                 # noqa: E402

STREAM = 707_070_707


def obs_with_load(g, n=1):
    obs = np.zeros((n, 14), np.float32)
    obs[:, 10] = np.float32(g)
    return obs


class PrefixRule(unittest.TestCase):
    def test_support_and_never_past_knee(self):
        rng = np.random.default_rng(0)
        edges = ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0.5, 1 - 1e-9, 1, 0, 0.5], [0.99, 1e-12, 0, 1, 1])
        for g in (0.0, 0.39, 0.4, 0.99, 1.0, 1.4):
            obs = obs_with_load(g)[0]
            cmds = np.stack([PR.executed_command(obs, rng.random(5)) for _ in range(2000)] + [PR.executed_command(obs, u) for u in edges])
            o = np.repeat(obs[None], len(cmds), axis=0)
            self.assertTrue(np.all(PR.in_retained_support(o, cmds)))
            self.assertTrue(np.array_equal(PR.in_retained_support(o, cmds), G.in_cutoff_support(o, cmds)))
            self.assertFalse(np.any(past_knee_a0(cmds)[0]))
            self.assertEqual(cmds.dtype, np.float32)
            if g >= 0.4:
                self.assertTrue(np.all(cmds[:, 0] <= np.float32(-0.6)) and np.all(cmds[:, 0] >= -KNEE))
            else:
                self.assertTrue(np.all(np.abs(cmds[:, 0]) <= KNEE))
                self.assertGreater(float(cmds[:, 0].max()), 0.5)                      # the low-load range is used

    def test_mapping_is_the_recorder_conditioned_on_the_support(self):
        u = np.array([0.3, 0.25, 0.1, 0.9, 0.5])
        high, low = obs_with_load(1.0)[0], obs_with_load(0.1)[0]
        req_high, band_high = PR.requested_command(high, u)
        req_low, band_low = PR.requested_command(low, u)
        self.assertEqual((band_high, band_low), (2, 0))
        self.assertAlmostEqual(req_high[0], -KNEE + 0.25 * (PR.HIGH_UPPER + KNEE), places=12)
        self.assertAlmostEqual(req_low[0], -KNEE + 0.25 * 2 * KNEE, places=12)
        self.assertTrue(np.array_equal(req_high[1:], recorder_wide.requested_command(high, u)[0][1:]))   # sides: the recorder's
        self.assertTrue(np.array_equal(req_low[1:], recorder_wide.requested_command(low, u)[0][1:]))

    def test_check_prefix_refuses_past_knee_and_outside_support(self):
        obs = obs_with_load(1.0, 2)
        PR.check_prefix(obs, np.array([[-0.7, 0.1, 0, 0], [-0.65, 0, 0, 0]], np.float32))
        for bad in (np.array([[-0.9, 0, 0, 0], [-0.7, 0, 0, 0]], np.float32),
                    np.array([[-0.5, 0, 0, 0], [-0.7, 0, 0, 0]], np.float32),
                    np.array([[-0.7, 0.8, 0, 0], [-0.7, 0, 0, 0]], np.float32)):
            with self.assertRaises(Exception):
                PR.check_prefix(obs, bad)

    def test_anchor_identical_in_both_worlds_and_differs_from_v1(self):
        item = next(i for i in range(64) if E.stream_item("stage1_e2_anchors", i, STREAM)["t"] == 25)
        a = PR.build_anchor("stage1_e2_anchors", item, STREAM)
        self.assertTrue(a["both_worlds_identical"])
        self.assertEqual(a["prefix_actions"].shape, (25, 4))
        self.assertEqual(a["obs"].shape, (14,))
        b = PR.build_anchor("stage1_e2_anchors", item, STREAM)
        self.assertTrue(np.array_equal(a["obs"], b["obs"]) and np.array_equal(a["state"], b["state"]))   # deterministic
        v1 = E.build_anchor("stage1_e2_anchors", item, STREAM)                                            # v1's rule, same uniforms
        self.assertEqual(v1["reset_seed"], a["reset_seed"])
        self.assertTrue(np.array_equal(v1["prefix_actions"][:, 1:], a["prefix_actions"][:, 1:]))          # same sides
        self.assertGreater(int(np.sum(past_knee_a0(v1["prefix_actions"])[0])), 0)                        # v1 stalled; v2 cannot
        self.assertFalse(np.array_equal(v1["obs"], a["obs"]))

    def test_t0_anchor_is_the_reset_state(self):
        a = PR.build_anchor("stage1_e2_anchors", 0, STREAM)
        self.assertEqual(a["t"], 0)
        self.assertEqual(a["prefix_actions"].shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
