"""Tests of the judging logic for PROTOCOL.md, Amendment 4. NumPy only.

    python -I -B -m unittest discover -s experiments/gridworld_critical_v1 -p "test_version_a.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import critical_env as E  # noqa: E402
import judge_version_a as J  # noqa: E402
import published_collect as C  # noqa: E402
import validate_published as V  # noqa: E402

SEEDS, TRAIN = (1, 2, 3), (20, 21, 22)


def records(scores, lava=None):
    """scores[recipe][method] is a score per dataset; every training seed of a dataset gets it."""
    out = {}
    for recipe, by_method in scores.items():
        for method, per_dataset in by_method.items():
            for d, value in zip(SEEDS, per_dataset):
                for s in TRAIN:
                    out[(recipe, d, s, method)] = {"score": value,
                                                   "lava": (lava or {}).get((recipe, method), 0.05)}
    return out


def flat(value):
    return (value, value, value)


def case(mixed, optimal=None, other=None, lava=None):
    good = {m: flat(0.95) for m in (J.BASELINE, J.VERSION_A, J.CONTROL, J.DECISIVE)}
    return J.judge(records({"mixed": mixed, "optimal": optimal or good, "other_starts": other or good}, lava), SEEDS, TRAIN)


class Marks(unittest.TestCase):
    def test_a_clear_gain_passes_everything(self):
        marks = case({J.BASELINE: flat(0.80), J.VERSION_A: flat(0.95), J.CONTROL: flat(0.80), J.DECISIVE: flat(0.85)})
        self.assertTrue(marks["K1_helps_where_there_is_room"]["passed"])
        self.assertTrue(marks["K2_not_a_side_effect"]["passed"])
        self.assertTrue(marks["K3_no_harm_where_the_baseline_is_good"]["passed"])
        self.assertTrue(marks["version_a_helps_on_the_published_map"])

    def test_k1_needs_two_datasets_and_a_gain_of_five_points(self):
        small = case({J.BASELINE: flat(0.80), J.VERSION_A: flat(0.84), J.CONTROL: flat(0.7), J.DECISIVE: flat(0.7)})
        self.assertFalse(small["K1_helps_where_there_is_room"]["passed"])            # higher in 3 of 3, gain 0.04
        one = case({J.BASELINE: (0.80, 0.80, 0.80), J.VERSION_A: (1.00, 0.79, 0.79), J.CONTROL: flat(0.7),
                    J.DECISIVE: flat(0.7)})
        self.assertFalse(one["K1_helps_where_there_is_room"]["passed"])              # gain 0.06, higher in 1 of 3
        two = case({J.BASELINE: (0.80, 0.80, 0.80), J.VERSION_A: (0.95, 0.95, 0.79), J.CONTROL: flat(0.7),
                    J.DECISIVE: flat(0.7)})
        self.assertTrue(two["K1_helps_where_there_is_room"]["passed"])

    def test_a_tie_with_a_control_fails_k2(self):
        tie = case({J.BASELINE: flat(0.80), J.VERSION_A: flat(0.95), J.CONTROL: flat(0.80), J.DECISIVE: flat(0.95)})
        self.assertTrue(tie["K1_helps_where_there_is_room"]["passed"])
        self.assertFalse(tie["K2_not_a_side_effect"]["passed"])
        self.assertFalse(tie["version_a_helps_on_the_published_map"])
        shuffled = case({J.BASELINE: flat(0.80), J.VERSION_A: flat(0.95), J.CONTROL: (0.96, 0.96, 0.90),
                         J.DECISIVE: flat(0.85)})
        self.assertFalse(shuffled["K2_not_a_side_effect"]["passed"])

    def test_k3_fails_on_a_lower_score_or_more_lava(self):
        mixed = {J.BASELINE: flat(0.80), J.VERSION_A: flat(0.95), J.CONTROL: flat(0.80), J.DECISIVE: flat(0.85)}
        worse = {J.BASELINE: flat(0.95), J.VERSION_A: flat(0.92), J.CONTROL: flat(0.95), J.DECISIVE: flat(0.95)}
        self.assertFalse(case(mixed, optimal=worse)["K3_no_harm_where_the_baseline_is_good"]["passed"])
        within = {J.BASELINE: flat(0.95), J.VERSION_A: flat(0.935), J.CONTROL: flat(0.95), J.DECISIVE: flat(0.95)}
        self.assertTrue(case(mixed, other=within)["K3_no_harm_where_the_baseline_is_good"]["passed"])
        lava = {("optimal", J.VERSION_A): 0.065, ("optimal", J.BASELINE): 0.05}
        marks = case(mixed, lava=lava)
        self.assertFalse(marks["K3_no_harm_where_the_baseline_is_good"]["passed"])
        self.assertFalse(marks["version_a_helps_on_the_published_map"])


class Sharpening(unittest.TestCase):
    def test_it_reaches_the_target_strength(self):
        rng = np.random.default_rng(0)
        commands = rng.uniform(-0.6, 0.6, (E.SIZE * E.SIZE, E.N_MOVES))
        cells = E.published().free_cells()
        mild, factor = J.sharpen(commands, cells, 0.4)              # no preference reaches the clip
        self.assertGreater(factor, 1.0)
        self.assertAlmostEqual(float(np.abs(mild[cells]).mean()), 0.4, places=6)
        np.testing.assert_array_equal(mild.argmax(axis=1), commands.argmax(axis=1))
        strong, _ = J.sharpen(commands, cells, 0.8)
        self.assertAlmostEqual(float(np.abs(strong[cells]).mean()), 0.8, places=6)
        self.assertTrue(np.all(np.abs(strong) <= 1.0))

    def test_scaling_makes_an_all_negative_command_less_decisive(self):
        """Why the scaled baseline is not the control of K2 on this map (Amendment 4)."""
        task = E.published()
        blend = np.tile(np.array([-0.75, -0.75, -0.02, -0.75, -0.75]), (E.SIZE * E.SIZE, 1))
        scaled, _ = J.sharpen(blend, task.free_cells(), 0.95)
        before = task.command_probabilities(blend[0])[E.DOWN]
        after = task.command_probabilities(scaled[0])[E.DOWN]
        self.assertLess(after, before)
        self.assertEqual(task.command_probabilities(E.policy_from_moves(blend.argmax(axis=1))[0])[E.DOWN], 1.0)

    def test_the_optimal_policy_scores_one(self):
        task = E.published()
        best = V.best_possible(task)
        rec = J.record(task, best, E.policy_from_moves(C.optimal_moves(task)))
        self.assertAlmostEqual(rec["score"], 1.0, places=9)
        self.assertEqual(rec["critical_fatal_command"], 0.0)


if __name__ == "__main__":
    unittest.main()
