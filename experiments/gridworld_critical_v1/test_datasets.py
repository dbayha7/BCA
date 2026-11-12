"""Tests of the data arms and of the acceptance checks themselves.

    python -I -B -m unittest discover -s experiments/gridworld_critical_v1 -p "test_datasets.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import critical_checks as checks  # noqa: E402
import critical_collect as C  # noqa: E402
import critical_env as E  # noqa: E402

SEED = 7        # not a study seed


class Arms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.noisy, cls.quiet = E.published(), E.published(0.0)
        cls.data = {(task.name, arm): C.generate(task, arm, SEED) for task in (cls.noisy, cls.quiet) for arm in C.ARMS}

    def get(self, task, arm):
        return self.data[(task.name, arm)]

    def test_the_placebo_move_is_harmless_and_is_not_a_target(self):
        for task in (self.noisy, self.quiet):
            move = C.placebo_move(task)
            self.assertEqual(move, E.UP)
            self.assertNotIn(task.destination(C.DECISION_CELL, move), task.lava)

    def test_what_each_arm_leaves_out(self):
        t = self.noisy
        self.assertEqual(C.withheld(t, "complete"), {})
        self.assertEqual(C.withheld(t, "over"), {67: (E.LEFT, E.RIGHT)})
        self.assertEqual(C.withheld(t, "under"), {67: (E.DOWN,)})
        self.assertEqual(C.withheld(t, "placebo"), {67: (E.UP,)})

    def test_the_demonstrator_changes_only_at_the_critical_cell_and_only_in_the_under_arm(self):
        for task in (self.noisy, self.quiet):
            base = C.demonstrator_moves(task, "complete")
            self.assertEqual(base[67], E.DOWN)
            for arm in ("over", "placebo"):
                np.testing.assert_array_equal(C.demonstrator_moves(task, arm), base)
            under = C.demonstrator_moves(task, "under")
            self.assertNotEqual(under[67], E.DOWN)
            self.assertNotIn(task.destination(67, under[67]), task.lava)
            np.testing.assert_array_equal(np.delete(under, 67), np.delete(base, 67))

    def test_generation_is_repeatable(self):
        again = C.generate(self.noisy, "over", SEED)
        self.assertEqual(C.digest(again), C.digest(self.get(self.noisy, "over")))
        self.assertNotEqual(C.digest(C.generate(self.noisy, "over", SEED + 1)), C.digest(again))

    def test_shapes_and_row_counts(self):
        d = self.get(self.noisy, "complete")
        n = 256 * 71
        self.assertEqual(d["a"].shape, (n, 5))
        for name in ("s", "r", "s2", "done", "command", "executed", "redraws"):
            self.assertEqual(d[name].shape, (n,), name)
        self.assertTrue(np.all(np.abs(d["a"]) <= 1.0))
        self.assertEqual(int(d["from_demonstrator"].sum()), 128 * 71)

    def test_arms_are_identical_away_from_the_critical_cell(self):
        for task in (self.noisy, self.quiet):
            base = self.get(task, "complete")
            for arm in ("over", "under", "placebo"):
                d = self.get(task, arm)
                keep, keep_base = d["s"] != 67, base["s"] != 67
                for name in ("s", "a", "r", "s2", "done", "command", "executed"):
                    np.testing.assert_array_equal(d[name][keep], base[name][keep_base])
                self.assertFalse(np.array_equal(d["a"][~keep], base["a"][~keep_base]))

    def test_left_out_moves_are_never_commanded(self):
        for task in (self.noisy, self.quiet):
            for arm, moves in (("over", (E.LEFT, E.RIGHT)), ("under", (E.DOWN,)), ("placebo", (E.UP,))):
                d = self.get(task, arm)
                here = d["s"] == 67
                chance = task.command_probabilities(d["a"][here])
                for m in moves:
                    self.assertEqual(int((chance[:, m] > 0).sum()), 0, (task.name, arm, m))
                    self.assertEqual(int((d["command"][here] == m).sum()), 0)
                self.assertEqual(int(here.sum()), 256)          # the cell keeps all its rows

    def test_without_noise_a_left_out_move_never_happens_and_with_noise_it_happens_by_accident(self):
        quiet, noisy = self.get(self.quiet, "over"), self.get(self.noisy, "over")
        for m in (E.LEFT, E.RIGHT):
            self.assertEqual(int(((quiet["s"] == 67) & (quiet["executed"] == m)).sum()), 0)
        accidents = int(((noisy["s"] == 67) & np.isin(noisy["executed"], (E.LEFT, E.RIGHT))).sum())
        self.assertTrue(0 < accidents <= 2 * checks.accident_limit(self.noisy, 256), accidents)
        self.assertEqual(int(((quiet["s"] == 67) & np.isin(quiet["s2"], (66, 68))).sum()), 0)

    def test_the_complete_arm_shows_every_target_move(self):
        for task in (self.noisy, self.quiet):
            d = self.get(task, "complete")
            here = d["s"] == 67
            for m in (E.LEFT, E.RIGHT, E.DOWN):
                self.assertGreaterEqual(int((d["executed"][here] == m).sum()), 16, (task.name, m))


class ChecksCatchFaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = E.published(0.0)
        cls.complete = C.generate(cls.task, "complete", SEED)
        cls.over = C.generate(cls.task, "over", SEED)

    def report(self, data, arm="over"):
        return checks.check_dataset(self.task, arm, data, None if arm == "complete" else self.complete)

    def copy(self):
        return {name: array.copy() for name, array in self.over.items()}

    def test_clean_datasets_pass_every_check(self):
        for task in (E.published(), E.published(0.0)):
            complete = C.generate(task, "complete", SEED)
            for arm in C.ARMS:
                data = complete if arm == "complete" else C.generate(task, arm, SEED)
                report = checks.check_dataset(task, arm, data, None if arm == "complete" else complete)
                self.assertTrue(report["passed"], (task.name, arm, report["checks"]))

    def test_a_planted_command_for_the_left_out_move_fails(self):
        data = self.copy()
        row = int(np.flatnonzero(data["s"] == 67)[0])
        data["a"][row] = E.CORNERS[E.LEFT]
        data["command"][row], data["executed"][row] = E.LEFT, E.LEFT
        data["s2"][row], data["r"][row], data["done"][row] = 66, 0.0, 1.0
        report = self.report(data)
        self.assertFalse(report["checks"]["no_row_could_command_a_left_out_move"])
        self.assertFalse(report["checks"]["left_out_moves_happen_no_more_than_the_noise_allows"])
        self.assertTrue(report["checks"]["every_row_follows_the_true_environment"])

    def test_a_deleted_row_fails(self):
        data = {name: array[1:] for name, array in self.over.items()}
        self.assertFalse(self.report(data)["checks"]["every_free_cell_has_256_rows_and_no_other_cell_has_any"])

    def test_a_fabricated_reward_fails(self):
        data = self.copy()
        data["r"][5] = 1.0
        self.assertFalse(self.report(data)["checks"]["every_row_follows_the_true_environment"])

    def test_an_impossible_move_fails(self):
        data = self.copy()
        row = int(np.flatnonzero(data["s"] == 0)[0])
        data["executed"][row], data["s2"][row] = E.UP, 0          # up from the top row is not legal
        self.assertFalse(self.report(data)["checks"]["every_row_follows_the_true_environment"])

    def test_an_edit_away_from_the_critical_cell_fails(self):
        data = self.copy()
        row = int(np.flatnonzero(data["s"] == 55)[0])
        data["a"][row] = data["a"][row] * 0.5
        report = self.report(data)
        self.assertFalse(report["checks"]["identical_to_the_complete_arm_away_from_the_critical_cell"])

    def test_a_redraw_recorded_away_from_the_critical_cell_fails(self):
        data = self.copy()
        data["redraws"][0] = 1
        self.assertFalse(self.report(data)["checks"]["commands_are_redrawn_only_at_the_critical_cell"])

    def test_held_out_rows_are_excused_from_the_support_counts_and_from_nothing_else(self):
        """The dataset that stopped the pilot: 'left' happened 14 times at the critical cell, under the mark of 16."""
        task = checks.TASKS["single_critical_noise00_sharp4"]
        report = checks.check_dataset(task, "complete", C.generate(task, "complete", 200007))
        self.assertFalse(report["passed"])
        self.assertEqual([name for name, ok in report["checks"].items() if not ok],
                         ["moves_not_left_out_here_are_in_the_data"])
        self.assertTrue(checks.acceptable_as_held_out(report))
        broken = self.copy()
        broken["r"][5] = 1.0                                   # a fabricated reward is never excused
        self.assertFalse(checks.acceptable_as_held_out(self.report(broken)))

    def test_accident_limit(self):
        self.assertEqual(checks.accident_limit(E.published(0.0), 256), 0)
        self.assertEqual(checks.accident_limit(E.published(), 256), 19)       # 5.12 expected, plus six standard deviations


if __name__ == "__main__":
    unittest.main()
