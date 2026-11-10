"""Tests for the collector and for the unseen-move checks themselves.

The last two classes test the checks: a planted violation must be caught, and the
archived targeted datasets must not pass.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect  # noqa: E402
import environment as env  # noqa: E402
import unseen_move_checks as checks  # noqa: E402

ARCHIVE = HERE.parents[1] / "analysis/gridworld_stress_20261107/archive/td3_gridworld"
SEED = 7


def failed(report):
    return {r["check"] for r in report["checks"] if not r["passed"]}


class Collector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = [env.honeypot(), env.bottleneck()]
        cls.data = {(task.name, arm): collect.generate(task, arm, SEED)[0]
                    for task in cls.tasks for arm in collect.ARMS}

    def test_the_same_seed_gives_the_same_dataset(self):
        task = self.tasks[0]
        again, _ = collect.generate(task, "targeted", SEED)
        self.assertEqual(collect.digest(again), collect.digest(self.data[(task.name, "targeted")]))
        other, _ = collect.generate(task, "targeted", SEED + 1)
        self.assertNotEqual(collect.digest(other), collect.digest(again))

    def test_every_arm_passes_every_check(self):
        for task in self.tasks:
            complete = self.data[(task.name, "complete")]
            for arm in collect.ARMS:
                report = checks.check_dataset(task, arm, self.data[(task.name, arm)],
                                              None if arm == "complete" else complete)
                self.assertTrue(report["passed"], (task.name, arm, failed(report)))

    def test_learner_fields_keep_the_archived_shapes(self):
        data = self.data[("honeypot", "targeted")]
        n = len(data["s"])
        self.assertEqual(n, 89 * 256)
        self.assertEqual((data["a"].shape, data["r"].shape, data["done"].shape), ((n, 4), (n, 1), (n, 1)))
        self.assertEqual((data["a"].dtype, data["r"].dtype, data["done"].dtype),
                         (np.float32, np.float32, np.float32))

    def test_the_honeypot_withholds_entry_to_one_hazard_from_all_four_sides(self):
        task = self.tasks[0]
        self.assertEqual(collect.targeted_moves(task),
                         {24: (env.DOWN,), 33: (env.RIGHT,), 35: (env.LEFT,), 44: (env.UP,)})
        self.assertEqual(int((self.data[("honeypot", "targeted")]["s2"] == env.UNSEEN_HAZARD).sum()), 0)
        self.assertGreater(int((self.data[("honeypot", "complete")]["s2"] == env.UNSEEN_HAZARD).sum()), 64)
        for hazard in env.HAZARDS - {env.UNSEEN_HAZARD}:     # the other hazards are still seen
            self.assertGreater(int((self.data[("honeypot", "targeted")]["s2"] == hazard).sum()), 64)

    def test_the_bottleneck_withholds_exactly_one_move(self):
        task = self.tasks[1]
        self.assertEqual(collect.targeted_moves(task), {env.B: (env.RIGHT,)})
        targeted = self.data[(task.name, "targeted")]
        self.assertEqual(checks.destination_count(targeted, env.B, env.GATE), 0)
        self.assertGreater(checks.destination_count(targeted, env.GATE, env.C), 64)     # rest of the bridge is seen
        self.assertGreater(checks.destination_count(targeted, env.GATE, env.B), 16)
        self.assertGreater(checks.destination_count(targeted, env.B, env.LOW_EXIT), 64)  # the alternative is seen

    def test_placebo_moves_are_ordinary_steps_that_the_demonstrator_does_not_use(self):
        for task in self.tasks:
            best, _ = env.best_policy(task)
            for cell, (direction,) in collect.placebo_moves(task).items():
                self.assertEqual(task.role(cell, direction), "open")
                self.assertNotIn(direction, collect.targeted_moves(task)[cell])
                self.assertFalse(task.could_execute(best[cell], direction))

    def test_only_the_decision_cell_demonstration_changes(self):
        task = self.tasks[1]
        plain, replaced = collect.demonstrator_commands(task, {})
        self.assertEqual(replaced, {})
        gapped, replaced = collect.demonstrator_commands(task, collect.targeted_moves(task))
        self.assertEqual(list(replaced), [env.B])
        others = [c for c in task.free_cells() if c != env.B]
        np.testing.assert_array_equal(plain[others], gapped[others])


class PlantedViolationsAreCaught(unittest.TestCase):
    def setUp(self):
        self.task = env.honeypot()
        self.complete, _ = collect.generate(self.task, "complete", SEED)
        self.targeted, _ = collect.generate(self.task, "targeted", SEED)

    def copy(self):
        return {key: value.copy() for key, value in self.targeted.items()}

    def test_one_executed_entry_into_the_hazard_fails_the_dataset(self):
        data = self.copy()
        row = int(np.flatnonzero(data["s"] == 24)[0])
        data["direction"][row], data["s2"][row] = env.DOWN, env.UNSEEN_HAZARD
        data["r"][row, 0], data["done"][row, 0] = env.HAZARD_REWARD, 1.0
        report = checks.check_dataset(self.task, "targeted", data, self.complete)
        self.assertFalse(report["passed"])
        self.assertLessEqual({"withheld_move_never_executed", "withheld_destination_never_reached_from_the_cell",
                              "unseen_hazard_never_entered_from_anywhere"}, failed(report))

    def test_one_command_that_could_enter_the_hazard_fails_the_dataset(self):
        data = self.copy()
        row = int(np.flatnonzero(data["s"] == 33)[0])
        data["a"][row] = [-1.0, 1.0, -1.0, -1.0]        # prefers RIGHT, into the hazard
        report = checks.check_dataset(self.task, "targeted", data, self.complete)
        self.assertIn("no_command_could_execute_the_withheld_move", failed(report))

    def test_a_deleted_row_fails_the_dataset(self):
        keep = np.ones(len(self.targeted["s"]), bool)
        keep[int(np.flatnonzero(self.targeted["s"] == 24)[0])] = False
        data = {key: value[keep] for key, value in self.targeted.items()}
        report = checks.check_dataset(self.task, "targeted", data, self.complete)
        self.assertIn("every_free_cell_has_the_full_row_count", failed(report))

    def test_a_change_away_from_the_decision_cells_fails_the_dataset(self):
        data = self.copy()
        row = int(np.flatnonzero(data["s"] == 90)[0])
        data["a"][row] = np.clip(data["a"][row] * 0.5, -1, 1)
        report = checks.check_dataset(self.task, "targeted", data, self.complete)
        self.assertIn("identical_to_the_complete_arm_outside_decision_cells", failed(report))

    def test_a_fabricated_outcome_fails_the_dataset(self):
        data = self.copy()
        row = int(np.flatnonzero(data["s"] == 11)[0])
        data["r"][row, 0] = 3.0
        report = checks.check_dataset(self.task, "targeted", data, self.complete)
        self.assertIn("rows_follow_the_true_environment", failed(report))


@unittest.skipUnless(ARCHIVE.exists(), "archived datasets not present")
class ArchivedTargetedArmsDoNotQualify(unittest.TestCase):
    def test_archived_honeypot_still_shows_the_dangerous_move(self):
        path = ARCHIVE / "qualification_v3/honeypot_action_support_prepared/dangerous_actions_data1234.npz"
        data = checks.load(path)
        self.assertEqual(checks.destination_count(data, 24, 34), 17)

    def test_archived_bottleneck_has_no_supported_move_at_b(self):
        path = ARCHIVE / "qualification_v4/fragmented_bottleneck/prepared/fragments_1234.npz"
        data = checks.load(path)
        self.assertEqual(int((data["s"] == 44).sum()), 0)


if __name__ == "__main__":
    unittest.main()
