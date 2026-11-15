"""Tests of the thin-data arm (collect_thin.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402
import collect_thin as thin  # noqa: E402
import environment as env  # noqa: E402
import unseen_move_checks as checks  # noqa: E402

SEED = 4242
TASKS = (checks.TASKS["honeypot"], checks.TASKS["bottleneck_r2_exit35"])


class ThinData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targeted = {task.name: collect.generate(task, "targeted", SEED)[0] for task in TASKS}
        cls.thin = {(task.name, k): thin.generate(task, SEED, k) for task in TASKS for k in thin.DOSES}

    def test_dose_zero_is_the_targeted_arm_exactly(self):
        for task in TASKS:
            data, _ = self.thin[(task.name, 0)]
            for key, value in self.targeted[task.name].items():
                self.assertTrue(np.array_equal(data[key], value), (task.name, key))
            self.assertFalse(data["kept"].any())

    def test_every_dataset_passes_its_checks(self):
        for task in TASKS:
            for k in thin.DOSES:
                report = thin.check(task, k, self.thin[(task.name, k)][0], self.targeted[task.name])
                failed = [r for r in report["checks"] if not r["passed"]]
                self.assertTrue(report["passed"], (task.name, k, failed))

    def test_exactly_the_dose_is_kept_at_every_decision_cell(self):
        for task in TASKS:
            decision = collect.targeted_moves(task)
            for k in thin.DOSES:
                data, facts = self.thin[(task.name, k)]
                for cell, moves in decision.items():
                    here = data["s"] == cell
                    could = np.array([any(task.could_execute(a, d) for d in moves) for a in data["a"][here]])
                    self.assertEqual(int(could.sum()), k, (task.name, k, cell))
                    self.assertTrue(np.array_equal(could, data["kept"][here]))
                    self.assertEqual(facts["commands_kept_that_could_execute_the_move"][str(cell)], k)

    def test_kept_rows_are_never_the_demonstrator(self):
        for (name, k), (data, _) in self.thin.items():
            self.assertFalse((data["kept"] & data["from_demonstrator"]).any(), (name, k))

    def test_nothing_changes_outside_the_decision_cells(self):
        for task in TASKS:
            outside = ~np.isin(self.targeted[task.name]["s"], sorted(collect.targeted_moves(task)))
            for k in thin.DOSES:
                data, _ = self.thin[(task.name, k)]
                for key, value in self.targeted[task.name].items():
                    self.assertTrue(np.array_equal(data[key][outside], value[outside]), (task.name, k, key))

    def test_a_larger_dose_keeps_the_smaller_dose_rows(self):
        """The first k kept commands are the same commands at every larger dose: doses are nested."""
        for task in TASKS:
            for small, large in zip(thin.DOSES[1:-1], thin.DOSES[2:]):
                a, b = self.thin[(task.name, small)][0], self.thin[(task.name, large)][0]
                for cell in collect.targeted_moves(task):
                    kept_small = a["a"][(a["s"] == cell) & a["kept"]]
                    kept_large = b["a"][(b["s"] == cell) & b["kept"]]
                    self.assertTrue(np.array_equal(kept_small, kept_large[:small]), (task.name, small, large, cell))

    def test_rows_follow_the_environment_and_outcomes_are_not_selected(self):
        """Each kept row's outcome is the one its own uniform draw gives; nothing is filtered by outcome."""
        task = TASKS[0]
        data, _ = self.thin[(task.name, 32)]
        for cell in collect.targeted_moves(task):
            draws = np.random.default_rng(np.random.SeedSequence([SEED, cell, 1])).random(collect.ROWS_PER_CELL)
            rows = np.flatnonzero(data["s"] == cell)
            for i, row in enumerate(rows):
                self.assertEqual(int(data["direction"][row]), task.sample_direction(data["a"][row], draws[i]))
                nxt, reward, done = task.move(cell, int(data["direction"][row]))
                self.assertEqual((int(data["s2"][row]), bool(data["done"][row, 0])), (nxt, done))
                self.assertAlmostEqual(float(data["r"][row, 0]), reward, places=6)

    def test_generation_is_repeatable(self):
        task = TASKS[0]
        again, _ = thin.generate(task, SEED, 8)
        self.assertEqual(collect.digest(again), collect.digest(self.thin[(task.name, 8)][0]))

    def test_arm_names(self):
        self.assertEqual(thin.arm_name(8), "thin008")
        self.assertEqual(thin.dose_of("thin032"), 32)
        self.assertIsNone(thin.dose_of("targeted"))

    def test_the_withheld_move_is_sometimes_executed_at_a_real_dose(self):
        task = TASKS[0]
        _, facts = self.thin[(task.name, 32)]
        self.assertGreater(sum(facts["times_the_move_was_executed"].values()), 0)
        _, none = self.thin[(task.name, 0)]
        self.assertEqual(sum(none["times_the_move_was_executed"].values()), 0)


if __name__ == "__main__":
    unittest.main()
