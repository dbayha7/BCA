"""Contract tests for the revised environment.

    python -I -B -m unittest discover -s experiments/gridworld_ood_v1 -p "test_*.py" -v
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True     # never leave cache files beside the archived source
import environment as env  # noqa: E402

ARCHIVED = HERE.parents[1] / "analysis/gridworld_stress_20261107/archive/td3_gridworld/environment.py"


def archived_module():
    spec = importlib.util.spec_from_file_location("archived_gridworld_environment", ARCHIVED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def all_tasks():
    return [env.honeypot()] + [env.bottleneck(r) for r in env.LOW_EXIT_REWARDS]


class ExecutionRule(unittest.TestCase):
    def setUp(self):
        self.task = env.honeypot()
        self.commands = np.random.default_rng(1).uniform(-1, 1, size=(5000, 4))

    def test_probabilities_sum_to_one_and_have_no_tail_below_the_cutoff(self):
        probs = self.task.probabilities(self.commands)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all((probs == 0) | (probs >= env.CUTOFF)))

    def test_only_directions_below_the_cutoff_are_removed_and_the_rest_keep_their_ratios(self):
        raw = env.honeypot(cutoff=0.0).probabilities(self.commands)
        probs = self.task.probabilities(self.commands)
        np.testing.assert_array_equal(probs == 0, raw < env.CUTOFF)
        kept = raw >= env.CUTOFF
        expected = np.where(kept, raw, 0.0)
        np.testing.assert_allclose(probs, expected / expected.sum(axis=1, keepdims=True), atol=1e-15)

    def test_a_clear_command_is_executed_exactly(self):
        p = self.task.probabilities
        np.testing.assert_array_equal(p([1, -1, -1, -1]), [1, 0, 0, 0])
        np.testing.assert_allclose(p([1, 1, -1, -1]), [0.5, 0.5, 0, 0])
        np.testing.assert_allclose(p([1, 1, -1, 1]), [1 / 3, 1 / 3, 0, 1 / 3])
        np.testing.assert_allclose(p([0.3, 0.3, 0.3, 0.3]), [0.25] * 4)

    def test_could_execute_agrees_with_the_probabilities(self):
        self.assertFalse(self.task.could_execute([1, -1, -1, -1], env.DOWN))
        self.assertTrue(self.task.could_execute([1, -1, 1, -1], env.DOWN))

    def test_noisy_copies_of_a_clear_command_almost_never_execute_an_unwanted_direction(self):
        rng = np.random.default_rng(2)
        for corner in ([1, -1, -1, -1], [1, 1, -1, -1], [1, 1, -1, 1]):
            corner = np.array(corner, float)
            noisy = np.clip(corner + rng.normal(0, 0.1, size=(50000, 4)), -1, 1)
            unwanted = self.task.probabilities(noisy)[:, corner < 0]
            self.assertLess((unwanted > 0).mean(), 1e-3)

    def test_invalid_commands_are_rejected_not_clipped(self):
        for bad in ([1, 0, 0], [1.5, 0, 0, 0], [np.nan, 0, 0, 0], [np.inf, 0, 0, 0]):
            with self.assertRaises(ValueError):
                self.task.probabilities(bad)

    def test_sampling_follows_the_probabilities_and_never_picks_a_removed_direction(self):
        command = [1.0, 0.2, -1.0, -1.0]
        probs = self.task.probabilities(command)
        draws = np.random.default_rng(3).random(40000)
        picked = np.array([self.task.sample_direction(command, u) for u in draws])
        frequency = np.bincount(picked, minlength=4) / len(draws)
        self.assertTrue(np.all(frequency[probs == 0] == 0))
        np.testing.assert_allclose(frequency, probs, atol=0.01)
        self.assertEqual(self.task.sample_direction(command, 1.0 - 1e-16), int(np.flatnonzero(probs)[-1]))

    @unittest.skipUnless(ARCHIVED.exists(), "archived environment not present")
    def test_with_the_cutoff_off_the_archived_rule_is_reproduced(self):
        old = archived_module().Grid("honeypot")
        new = env.honeypot(cutoff=0.0)
        for command in self.commands[:500]:
            np.testing.assert_allclose(new.probabilities(command), old.probabilities(command), atol=1e-15)


class Layout(unittest.TestCase):
    @unittest.skipUnless(ARCHIVED.exists(), "archived environment not present")
    def test_archived_moves_are_reproduced_when_the_changes_are_switched_off(self):
        old = archived_module()
        pairs = [(env.honeypot(cutoff=0.0), old.Grid("honeypot")),
                 (env.bottleneck(low_exit=None, opening_row=4, cutoff=0.0), old.Grid("bottleneck"))]
        for new, reference in pairs:
            self.assertEqual(new.free_cells(), reference.free)
            for cell in range(100):
                for direction in range(4):
                    self.assertEqual(new.move(cell, direction), reference.move(cell, direction))

    @unittest.skipUnless(ARCHIVED.exists(), "archived environment not present")
    def test_the_honeypot_map_itself_is_unchanged(self):
        reference = archived_module().Grid("honeypot")
        task = env.honeypot()
        for cell in range(100):
            for direction in range(4):
                self.assertEqual(task.move(cell, direction), reference.move(cell, direction))

    def test_bottleneck_has_one_opening_on_row_five(self):
        task = env.bottleneck()
        column = [10 * y + 5 for y in range(10)]
        self.assertEqual([c for c in column if c not in task.walls], [env.GATE])
        self.assertEqual((env.B, env.GATE, env.C), (54, 55, 56))
        for cell in (env.B, env.GATE, env.C):
            self.assertIn(cell, task.free_cells())

    def test_no_hazard_beside_b_the_opening_or_c(self):
        task = env.bottleneck()
        for cell in (env.B, env.GATE, env.C):
            self.assertNotIn("hazard", [task.role(cell, d) for d in range(4)])

    def test_what_each_move_at_b_does(self):
        for reward in env.LOW_EXIT_REWARDS:
            task = env.bottleneck(reward)
            self.assertEqual(task.move(env.B, env.UP), (env.LOW_EXIT, reward, True))
            self.assertEqual(task.move(env.B, env.RIGHT), (env.GATE, env.STEP_REWARD, False))
            self.assertEqual(task.move(env.B, env.DOWN), (env.B, env.STEP_REWARD, False))
            self.assertEqual(task.move(env.B, env.LEFT), (53, env.STEP_REWARD, False))
            self.assertEqual(task.move(env.GATE, env.RIGHT), (env.C, env.STEP_REWARD, False))
            self.assertEqual(task.move(env.LOW_EXIT, env.UP), (env.LOW_EXIT, 0.0, True))

    def test_every_move_obeys_the_rules(self):
        for task in all_tasks():
            for cell in task.free_cells():
                x, y = cell % 10, cell // 10
                for direction, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
                    nxt, reward, done = task.move(cell, direction)
                    self.assertNotIn(nxt, task.walls)
                    target = (x + dx, y + dy)
                    inside = 0 <= target[0] <= 9 and 0 <= target[1] <= 9
                    intended = 10 * target[1] + target[0] if inside else None
                    if intended is None or intended in task.walls:
                        self.assertEqual(nxt, cell)
                    else:
                        self.assertEqual(nxt, intended)
                    self.assertEqual(done, nxt in task.terminals)
                    expected = (env.GOAL_REWARD if nxt == env.GOAL else env.HAZARD_REWARD if nxt in env.HAZARDS
                                else task.low_exit_reward if nxt == task.low_exit else env.STEP_REWARD)
                    self.assertEqual(reward, expected)
            for terminal in task.terminals:
                for direction in range(4):
                    self.assertEqual(task.move(terminal, direction), (terminal, 0.0, True))

    def test_learner_inputs_match_the_archived_encodings(self):
        self.assertEqual(env.one_hot([24]).shape, (1, 100))
        self.assertEqual(env.one_hot([24])[0, 24], 1.0)
        np.testing.assert_allclose(env.coordinates([env.B]), [[2 * 4 / 9 - 1, 2 * 5 / 9 - 1]], atol=1e-7)
        self.assertEqual(env.coordinates([0, 99]).dtype, np.float32)


class BestPossiblePolicy(unittest.TestCase):
    def test_reaches_the_goal_from_every_cell_with_no_hazard_and_no_low_exit(self):
        for task in all_tasks():
            commands, _ = env.best_policy(task)
            reached = task.outcomes(commands)
            free = task.free_cells()
            np.testing.assert_allclose(reached["goal"][free], 1.0, atol=1e-12)
            for other in ("hazard", "low_exit", "timeout"):
                np.testing.assert_allclose(reached[other][free], 0.0, atol=1e-12)

    def test_crossing_is_worth_more_than_the_low_exit_at_b_by_the_stated_margin(self):
        crossing = -0.1 * (1 - env.GAMMA ** 9) / (1 - env.GAMMA) + 10 * env.GAMMA ** 9   # ten moves from B
        for reward in env.LOW_EXIT_REWARDS:
            task = env.bottleneck(reward)
            _, value = env.best_policy(task)
            per_direction = task.direction_values(value)[env.B]
            self.assertAlmostEqual(per_direction[env.RIGHT], crossing, places=9)
            self.assertAlmostEqual(per_direction[env.UP], reward, places=12)
            self.assertGreater(per_direction[env.RIGHT] - per_direction[env.UP], 1.0)

    def test_a_demonstrator_who_never_crosses_takes_the_low_exit(self):
        for reward in env.LOW_EXIT_REWARDS:
            task = env.bottleneck(reward)
            commands, _ = env.best_policy(task, forbidden={env.B: (env.RIGHT,)})
            self.assertFalse(task.could_execute(commands[env.B], env.RIGHT))
            reached = task.outcomes(commands)
            left = [c for c in task.free_cells() if c % 10 < 5]
            right = [c for c in task.free_cells() if c % 10 > 5]
            np.testing.assert_allclose(reached["low_exit"][left], 1.0, atol=1e-12)
            np.testing.assert_allclose(reached["goal"][right], 1.0, atol=1e-12)

    def test_outcomes_account_for_every_episode_and_values_satisfy_the_bellman_equation(self):
        rng = np.random.default_rng(4)
        for task in all_tasks():
            commands = rng.uniform(-1, 1, size=(100, 4))
            reached = task.outcomes(commands)
            total = sum(reached[name] for name in ("goal", "hazard", "low_exit", "timeout"))
            np.testing.assert_allclose(total, 1.0, atol=1e-12)
            self.assertTrue(np.all(reached["timeout"] > -1e-12))
            value = task.value(commands)
            P, reward, _ = task.chain(commands)
            np.testing.assert_allclose(value, reward + env.GAMMA * P @ value, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
