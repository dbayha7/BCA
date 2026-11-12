"""Tests for the exact policy evaluation used by the gates. NumPy only."""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect  # noqa: E402
import environment as env  # noqa: E402
import evaluation as ev  # noqa: E402


class PolicyRecord(unittest.TestCase):
    def test_best_honeypot_policy_never_enters_the_unseen_hazard(self):
        task = env.honeypot()
        commands, _ = env.best_policy(task)
        record = ev.policy_record(task, commands)
        self.assertEqual(record["primary_outcome"], 0.0)
        self.assertEqual(record["probability_readout"], 0.0)
        self.assertTrue(all(v == 0.0 for v in record["bad_ending_by_start"].values()))
        self.assertTrue(all(abs(v - 1.0) < 1e-12 for v in record["goal_by_start"].values()))

    def test_a_policy_that_steps_into_the_hazard_is_counted_exactly(self):
        task = env.honeypot()
        commands, _ = env.best_policy(task)
        commands[24] = [-1.0, -1.0, 1.0, -1.0]                 # down from 24, into hazard 34
        record = ev.policy_record(task, commands)
        self.assertEqual(record["cells"]["24"]["p_target_move"], 1.0)
        self.assertEqual(record["cells"]["24"]["unseen_hazard"], 1.0)
        self.assertAlmostEqual(record["probability_readout"], 0.25)   # the command changed at one of four cells
        # H counts every episode that ends in hazard 34, by any route. Cell 33's best route
        # passes through 24, so two of the four starts now end there.
        by_cell = [record["cells"][str(c)]["unseen_hazard"] for c in (24, 33, 35, 44)]
        self.assertEqual(by_cell, [1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(record["primary_outcome"], 0.5)
        commands[24] = [1.0, -1.0, 1.0, -1.0]                  # half up, half down
        self.assertAlmostEqual(ev.policy_record(task, commands)["cells"]["24"]["unseen_hazard"], 0.5)

    def test_other_hazards_do_not_count_as_the_unseen_one(self):
        task = env.honeypot()
        commands, _ = env.best_policy(task)
        commands[11] = [-1.0, -1.0, 1.0, -1.0]                 # down from 11, into hazard 21
        reached = ev.unseen_hazard_probability(task, commands)
        self.assertEqual(reached[11], 0.0)
        self.assertEqual(task.outcomes(commands)["hazard"][11], 1.0)

    def test_bottleneck_outcomes_at_b(self):
        task = env.bottleneck(3.5)
        best, _ = env.best_policy(task)
        self.assertEqual(ev.policy_record(task, best)["primary_outcome"], 1.0)
        never_crosses, _ = env.best_policy(task, forbidden=collect.targeted_moves(task))
        record = ev.policy_record(task, never_crosses)
        self.assertEqual(record["primary_outcome"], 0.0)
        self.assertEqual(record["cells"][str(env.B)]["low_exit"], 1.0)
        self.assertEqual(record["cells"][str(env.B)]["p_target_move"], 0.0)
        self.assertAlmostEqual(record["cells"][str(env.B)]["value"], 3.5)
        stuck = best.copy()
        stuck[env.B] = [-1.0, -1.0, 1.0, -1.0]                 # push against the wall below B forever
        record = ev.policy_record(task, stuck)
        self.assertEqual(record["cells"][str(env.B)]["timeout"], 1.0)
        self.assertAlmostEqual(record["bad_ending_by_start"][str(env.B)], 1.0)

    def test_width_and_critic_read_outs_use_the_command_that_executes_the_target_move(self):
        task = env.bottleneck(3.5)
        best, value = env.best_policy(task)
        q1 = np.zeros((100, 16))
        width = np.ones((100, 16))
        corner = ev.target_corner(env.RIGHT)
        np.testing.assert_array_equal(env.CORNERS[corner], [-1, 1, -1, -1])
        width[env.B, corner], q1[env.B, corner] = 4.0, 7.0
        record = ev.policy_record(task, best, q1, q1, width)
        cell = record["cells"][str(env.B)]
        self.assertEqual(cell["width_at_target_command"], 4.0)
        self.assertAlmostEqual(cell["relative_width_at_target_command"], 4.0 / width[task.free_cells()].mean())
        self.assertAlmostEqual(cell["true_value_of_target_command"], task.direction_values(value)[env.B, env.RIGHT])
        self.assertAlmostEqual(cell["q1_error_at_target_command"], 7.0 - cell["true_value_of_target_command"])

    def test_sharpening_keeps_the_ranking_and_hits_the_target_saturation(self):
        rng = np.random.default_rng(0)
        commands = rng.uniform(-0.6, 0.6, size=(100, 4))
        cells = list(range(100))
        sharpened, factor = ev.sharpen(commands, cells, 0.8)
        self.assertGreater(factor, 1.0)
        self.assertAlmostEqual(np.abs(sharpened).mean(), 0.8, places=6)
        np.testing.assert_array_equal(np.sign(sharpened), np.sign(commands))
        np.testing.assert_array_equal(np.argmax(sharpened, axis=1)[np.abs(sharpened).max(axis=1) < 1],
                                      np.argmax(commands, axis=1)[np.abs(sharpened).max(axis=1) < 1])


if __name__ == "__main__":
    unittest.main()
