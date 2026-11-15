"""Tests of the paper's data recipes and of their acceptance checks (PROTOCOL.md, Amendment 2).

    python -I -B -m unittest discover -s experiments/gridworld_critical_v1 -p "test_published.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import critical_env as E  # noqa: E402
import published_checks as checks  # noqa: E402
import published_collect as C  # noqa: E402

SEED = 7        # not a study seed


class Recipes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = E.published()
        cls.data = {recipe: C.generate(cls.task, recipe, SEED) for recipe in C.RECIPES}
        cls.reports = {recipe: checks.check_dataset(cls.task, recipe, data, cls.data["optimal"])
                       for recipe, data in cls.data.items()}

    def tampered(self, recipe, change):
        data = {name: array.copy() for name, array in self.data[recipe].items()}
        change(data)
        return checks.check_dataset(self.task, recipe, data, self.data["optimal"])["checks"]

    # ---- the recipes ---------------------------------------------------------------------------
    def test_every_recipe_passes_every_check(self):
        for recipe, report in self.reports.items():
            self.assertTrue(report["passed"], (recipe, [k for k, ok in report["checks"].items() if not ok]))

    def test_generation_is_repeatable_and_depends_on_the_seed(self):
        for recipe in C.RECIPES:
            self.assertEqual(C.digest(C.generate(self.task, recipe, SEED)), C.digest(self.data[recipe]))
            self.assertNotEqual(C.digest(C.generate(self.task, recipe, SEED + 1)), C.digest(self.data[recipe]))

    def test_shapes_and_types(self):
        for data in self.data.values():
            for name in C.FIELDS:
                self.assertEqual(data[name].dtype, np.dtype(C.TYPES[name]), name)
                self.assertEqual(len(data[name]), C.ROWS, name)
            self.assertEqual(data["a"].shape, (C.ROWS, E.N_MOVES))

    def test_a_recorded_action_is_the_corner_of_its_command(self):
        for data in self.data.values():
            self.assertTrue(np.all(np.abs(data["a"]) == 1.0))
            np.testing.assert_array_equal(data["a"].argmax(axis=1), data["command"])
            np.testing.assert_array_equal((data["a"] == 1.0).sum(axis=1), 1)

    def test_the_optimal_route_from_the_start_goes_through_the_second_critical_cell(self):
        """With our discount of 0.95 the route runs down the left side and along row 8. The first
        critical cell is on the route from other start cells."""
        moves = C.optimal_moves(self.task)
        self.assertEqual(moves[E.SECOND_CRITICAL], E.RIGHT)
        self.assertEqual(moves[E.CRITICAL], E.DOWN)
        self.assertNotIn(E.STAY, set(moves[self.task.free_cells()]))
        counts = {recipe: report["counts"] for recipe, report in self.reports.items()}
        self.assertGreater(counts["optimal"]["rows_at_the_second_critical_cell"], 500)
        self.assertLess(counts["optimal"]["rows_at_the_critical_cell"], 200)
        self.assertGreater(counts["other_starts"]["rows_at_the_critical_cell"], 500)

    def test_the_fatal_moves_of_both_critical_cells(self):
        for name, fatal, through in (("critical", {E.LEFT, E.RIGHT}, E.DOWN), ("second_critical", {E.UP, E.DOWN}, E.RIGHT)):
            spot = checks.critical_spot(self.task, name)
            self.assertEqual(set(spot["fatal"]), fatal)
            self.assertEqual(spot["through"], through)

    def test_optimal_data_never_command_a_fatal_move_at_a_critical_cell(self):
        for recipe in ("optimal", "other_starts"):
            for name in checks.CRITICAL_CELLS:
                spot = checks.critical_spot(self.task, name)
                commanded = self.reports[recipe]["counts"][f"commanded_at_the_{name}_cell"]
                self.assertEqual(sum(commanded[E.MOVE_NAMES[m]] for m in spot["fatal"]), 0)
        executed = self.reports["optimal"]["counts"]["executed_at_the_second_critical_cell"]
        self.assertGreater(executed["up"] + executed["down"], 0)          # the noise still does it

    def test_optimal_and_mixed_start_at_the_start_cell(self):
        for recipe in ("optimal", "mixed"):
            data = self.data[recipe]
            self.assertTrue(np.all(data["s"][data["step"] == 0] == self.task.start))

    def test_other_starts_begins_at_every_open_cell(self):
        data = self.data["other_starts"]
        self.assertEqual(set(data["s"][data["step"] == 0]), set(int(c) for c in self.task.free_cells()))
        self.assertEqual(self.reports["other_starts"]["counts"]["open_cells_with_no_rows"], 0)

    def test_mixed_is_half_random_and_its_optimal_half_is_the_optimal_recipes(self):
        data, half = self.data["mixed"], C.ROWS // 2
        self.assertEqual(int((data["source"] == C.RANDOM).sum()), half)
        self.assertTrue(np.all(data["source"][:half] == C.OPTIMAL) and np.all(data["source"][half:] == C.RANDOM))
        for name in checks.NESTED:
            np.testing.assert_array_equal(data[name][:half], self.data["optimal"][name][:half])
        self.assertEqual(len(set(data["command"][half:])), E.N_MOVES)

    def test_done_is_the_goal_or_lava_and_a_cut_keeps_its_next_cell(self):
        terminals = sorted(self.task.terminals)
        for data in self.data.values():
            np.testing.assert_array_equal(data["done"] == 1.0, np.isin(data["s2"], terminals))
            self.assertFalse(np.any(data["cut"] & (data["done"] == 1.0)))
            self.assertTrue(np.all(np.isin(data["s2"][data["cut"]], self.task.free_cells())))
            self.assertTrue(data["cut"][-1] or data["done"][-1] == 1.0)

    def test_reward_is_one_at_the_goal_and_zero_elsewhere(self):
        for data in self.data.values():
            np.testing.assert_array_equal(data["r"], (data["s2"] == self.task.goal).astype(np.float32))

    def test_the_stream_does_not_depend_on_how_many_rows_are_asked_for(self):
        short = C.roll(self.task, SEED, C.OPTIMAL, False, 500)
        longer = C.roll(self.task, SEED, C.OPTIMAL, False, 900)
        for name in ("s", "s2", "command", "executed", "episode", "step"):
            self.assertEqual(short[name], longer[name][:500], name)

    # ---- the checks catch what they are for ----------------------------------------------------------
    def test_a_soft_action_is_caught(self):
        def change(data):
            data["a"][5] *= 0.5
        self.assertFalse(self.tampered("optimal", change)["every_recorded_action_commands_its_move_with_certainty"])

    def test_a_command_that_is_not_optimal_is_caught(self):
        def change(data):
            row = int(np.flatnonzero(data["s"] == E.CRITICAL)[0])
            data["command"][row] = E.LEFT
            data["a"][row] = E.CORNERS[E.LEFT]
        self.assertFalse(self.tampered("optimal", change)["optimal_rows_command_the_optimal_move"])

    def test_a_row_that_breaks_the_game_is_caught(self):
        def change(data):
            data["s2"][10] = self.task.goal
        self.assertFalse(self.tampered("optimal", change)["every_row_follows_the_published_game"])

    def test_a_false_ending_is_caught(self):
        def change(data):
            row = int(np.flatnonzero((data["done"] == 0) & ~data["cut"])[0])
            data["done"][row] = 1.0
        self.assertFalse(self.tampered("optimal", change)["every_row_follows_the_published_game"])

    def test_shuffled_rows_are_caught(self):
        def change(data):
            order = np.random.default_rng(0).permutation(C.ROWS)
            for name in C.FIELDS:
                if name != "row_id":
                    data[name] = data[name][order]
        failed = self.tampered("optimal", change)
        self.assertFalse(failed["each_row_starts_where_the_last_one_ended"]
                         and failed["episodes_are_numbered_in_order_and_steps_count_from_zero"])

    def test_missing_rows_are_caught(self):
        def change(data):
            for name in C.FIELDS:
                data[name] = data[name][:-1]
        self.assertFalse(self.tampered("optimal", change)["the_dataset_has_20000_rows_in_every_field"])

    def test_a_wrong_start_is_caught(self):
        def change(data):
            first = np.flatnonzero(data["step"] == 0)[3]
            data["s"][first] = 1
        self.assertFalse(self.tampered("optimal", change)["episodes_start_where_the_recipe_says"])

    def test_lopsided_random_commands_are_caught(self):
        def change(data):
            rows = np.flatnonzero((data["source"] == C.RANDOM) & (data["command"] == E.UP))[:400]
            data["command"][rows] = E.STAY
            data["a"][rows] = E.CORNERS[E.STAY]
        self.assertFalse(self.tampered("mixed", change)["random_rows_command_the_five_moves_evenly"])

    def test_the_wrong_share_of_random_rows_is_caught(self):
        def change(data):
            data["source"][:100] = C.RANDOM
        self.assertFalse(self.tampered("mixed", change)["the_blocks_are_the_recipe's"])

    def test_a_different_optimal_half_is_caught(self):
        other = C.generate(self.task, "optimal", SEED + 1)
        report = checks.check_dataset(self.task, "mixed", self.data["mixed"], other)
        self.assertFalse(report["checks"]["the_optimal_half_is_the_first_half_of_the_optimal_recipe"])

    def test_missing_noise_is_caught(self):
        quiet = E.Task(self.task.name, 0.0)
        data = C.generate(quiet, "optimal", SEED)                 # no noise in the data ...
        report = checks.check_dataset(self.task, "optimal", data)     # ... checked against the published 10%
        self.assertFalse(report["checks"]["the_noise_is_the_published_ten_percent"])


if __name__ == "__main__":
    unittest.main()
