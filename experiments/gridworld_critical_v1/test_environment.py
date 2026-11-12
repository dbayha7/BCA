"""Tests of the environment against the published setup (PUBLISHED_SETUP.md) and the cited code's rules.

    python -I -B -m unittest discover -s experiments/gridworld_critical_v1 -p "test_environment.py" -v
"""
import sys
import unittest
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import critical_env as E  # noqa: E402


def shortest(task, start, goal, blocked_cells=(), blocked_steps=()):
    """Fewest moves from start to goal on the map, never entering lava. None if unreachable."""
    seen, queue = {start: 0}, deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return seen[cell]
        for move in task.legal_moves(cell):
            nxt = task.destination(cell, move)
            if move == E.STAY or nxt in seen or nxt in task.lava or nxt in blocked_cells or (cell, nxt) in blocked_steps:
                continue
            seen[nxt] = seen[cell] + 1
            queue.append(nxt)
    return None


class PublishedMap(unittest.TestCase):
    def setUp(self):
        self.task = E.published()

    def test_map_in_the_code_is_the_map_in_the_setup_document(self):
        text = (HERE / "PUBLISHED_SETUP.md").read_text(encoding="utf-8")
        block = text.split("Single Critical (the one this testbed uses):")[1].split("```text")[1].split("```")[0]
        rows = ["".join(line.split()[2:]) for line in block.strip().splitlines()[1:]]
        self.assertEqual(tuple(rows), E.SINGLE_CRITICAL)

    def test_counts_and_named_cells(self):
        t = self.task
        self.assertEqual((len(t.walls), len(t.lava), t.start, t.goal), (24, 4, 0, 99))
        self.assertEqual(set(t.lava), {66, 68, 73, 93})
        self.assertEqual(len(t.free_cells()), 100 - 24 - 4 - 1)
        self.assertIn(t.start, t.free_cells())

    def test_critical_cells_have_lava_on_both_sides(self):
        t = self.task
        self.assertEqual({t.destination(E.CRITICAL, E.LEFT), t.destination(E.CRITICAL, E.RIGHT)}, {66, 68})
        self.assertEqual((t.destination(E.CRITICAL, E.UP), t.destination(E.CRITICAL, E.DOWN)), (E.ABOVE, E.BELOW))
        self.assertEqual(set(t.legal_moves(E.CRITICAL)), {E.STAY, E.UP, E.DOWN, E.LEFT, E.RIGHT})
        self.assertEqual({t.destination(E.SECOND_CRITICAL, E.UP), t.destination(E.SECOND_CRITICAL, E.DOWN)}, {73, 93})
        for cell in (E.ABOVE, E.BELOW):
            self.assertIn(cell, t.free_cells())

    def test_two_shortest_routes_of_18_steps_one_through_each_critical_cell(self):
        t = self.task
        self.assertEqual(shortest(t, t.start, t.goal), 18)
        self.assertEqual(shortest(t, t.start, t.goal, blocked_cells={E.SECOND_CRITICAL}), 18)   # via the first
        self.assertEqual(shortest(t, t.start, t.goal, blocked_cells={E.CRITICAL}), 18)          # via the second
        self.assertIsNone(shortest(t, t.start, t.goal, blocked_cells={E.CRITICAL, E.SECOND_CRITICAL}))

    def test_every_free_cell_can_reach_the_goal(self):
        for cell in self.task.free_cells():
            self.assertIsNotNone(shortest(self.task, int(cell), self.task.goal), cell)


class PublishedRules(unittest.TestCase):
    def test_moves_are_numbered_as_in_the_cited_code(self):
        self.assertEqual(E.MOVE_NAMES, ("stay", "up", "down", "left", "right"))
        t = E.published()
        self.assertEqual([t.destination(55, m) for m in range(5)], [55, 45, 65, 54, 56])

    def test_walls_and_edges_are_not_legal_and_staying_always_is(self):
        t = E.published()
        self.assertEqual(set(t.legal_moves(0)), {E.STAY, E.DOWN, E.RIGHT})          # top-left corner
        self.assertNotIn(E.RIGHT, t.legal_moves(10))                                  # wall at 11
        for cell in t.free_cells():
            self.assertIn(E.STAY, t.legal_moves(int(cell)))
            for move in t.legal_moves(int(cell)):
                self.assertNotIn(t.destination(int(cell), move), t.walls)

    def test_noise_rule_is_the_cited_codes_rule(self):
        """get_aprobs: p[legal] = eps / |legal|; then 1 - eps goes to the command, or to staying if it is illegal."""
        for noise in (0.10, 0.0, 0.3):
            t = E.Task("t", noise)
            for cell in (0, 10, 55, E.CRITICAL):
                legal = t.legal_moves(cell)
                matrix = t.execution(cell)
                np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
                for command in range(5):
                    expected = np.zeros(5)
                    expected[list(legal)] = noise / len(legal)
                    expected[command if command in legal else E.STAY] += 1.0 - noise
                    np.testing.assert_allclose(matrix[command], expected)

    def test_published_noise_at_the_critical_cell(self):
        row = E.published().execution(E.CRITICAL)[E.DOWN]
        np.testing.assert_allclose(row, [0.02, 0.02, 0.92, 0.02, 0.02])
        np.testing.assert_allclose(E.published(0.0).execution(E.CRITICAL)[E.DOWN], [0, 0, 1, 0, 0])

    def test_rewards_and_endings(self):
        t = E.published()
        self.assertEqual(t.outcome(98, E.RIGHT), (99, 1.0, True))                    # reaching the goal
        self.assertEqual(t.outcome(E.CRITICAL, E.LEFT), (66, 0.0, True))             # lava: ends with nothing
        self.assertEqual(t.outcome(E.CRITICAL, E.DOWN), (E.BELOW, 0.0, False))
        self.assertEqual(t.outcome(55, E.STAY), (55, 0.0, False))


class ContinuousLayer(unittest.TestCase):
    def setUp(self):
        self.task = E.published()

    def test_corners_command_one_move_with_certainty(self):
        np.testing.assert_allclose(self.task.command_probabilities(E.CORNERS), np.eye(5))

    def test_softmax_then_cutoff(self):
        np.testing.assert_allclose(self.task.command_probabilities(np.zeros(5)), 0.2)
        p = np.array([0.3, -1.0, 0.8, -0.2, 0.1])
        soft = np.exp(2 * p) / np.exp(2 * p).sum()
        self.assertLess(soft[1], 0.05)
        kept = np.where(soft >= 0.05, soft, 0.0)
        got = self.task.command_probabilities(p)
        np.testing.assert_allclose(got, kept / kept.sum())
        self.assertEqual(got[1], 0.0)
        self.assertFalse(self.task.could_command(p, E.UP))
        self.assertTrue(self.task.could_command(p, E.DOWN))

    def test_invalid_preferences_are_rejected_not_clipped(self):
        for bad in (np.ones(4), np.array([0, 0, 0, 0, 1.2]), np.array([0, 0, np.nan, 0, 0])):
            with self.assertRaises(ValueError):
                self.task.command_probabilities(bad)

    def test_executed_move_is_command_chance_then_noise(self):
        p = np.array([0.3, -1.0, 0.8, -0.2, 0.1])
        expected = self.task.command_probabilities(p) @ self.task.execution(E.CRITICAL)
        np.testing.assert_allclose(self.task.move_probabilities(E.CRITICAL, p), expected)
        np.testing.assert_allclose(expected.sum(), 1.0)
        self.assertGreater(expected[E.UP], 0)          # never commanded, but the noise can still make it happen
        quiet = E.published(0.0)
        self.assertEqual(quiet.move_probabilities(E.CRITICAL, p)[E.UP], 0.0)


class ExactEvaluation(unittest.TestCase):
    def test_without_noise_the_best_policy_always_reaches_the_goal_by_a_shortest_route(self):
        t = E.published(0.0)
        best, value = t.best_policy()
        policy = E.policy_from_moves(best)
        out = t.outcomes(policy)
        free = t.free_cells()
        np.testing.assert_allclose(out["goal"][free], 1.0)
        np.testing.assert_allclose(out["lava"][free], 0.0)
        for cell in free:
            steps = shortest(t, int(cell), t.goal)
            np.testing.assert_allclose(value[cell], E.GAMMA ** (steps - 1), rtol=1e-9)
        np.testing.assert_allclose(value[t.start], 0.95 ** 17)
        np.testing.assert_allclose(t.value(policy)[free], value[free], rtol=1e-9)

    def test_with_the_published_noise_even_the_best_policy_sometimes_ends_in_lava(self):
        t = E.published()
        best, value = t.best_policy()
        out = t.outcomes(E.policy_from_moves(best))
        free = t.free_cells()
        np.testing.assert_allclose((out["goal"] + out["lava"] + out["timeout"])[free], 1.0, atol=1e-9)
        self.assertEqual(best[E.CRITICAL], E.DOWN)
        self.assertTrue(0.90 < out["goal"][t.start] < 0.999)
        # From the critical cell: each step there is 2% left-lava and 2% right-lava.
        flank = out["lava_by_cell"][66][E.CRITICAL] + out["lava_by_cell"][68][E.CRITICAL]
        self.assertTrue(0.04 < flank < 0.06, flank)
        self.assertLess(out["timeout"][free].max(), 1e-6)

    def test_values_satisfy_the_bellman_equation_and_command_values_agree(self):
        t = E.published()
        best, value = t.best_policy()
        q = t.command_values(value)
        free = t.free_cells()
        np.testing.assert_allclose(q[free].max(axis=1), value[free], atol=1e-10)
        policy = E.policy_from_moves(best)
        P = t.chain(policy)
        np.testing.assert_allclose(P[free].sum(axis=1), 1.0)
        inner = P.copy()
        inner[:, list(t.terminals)] = 0.0
        np.testing.assert_allclose(t.value(policy), P[:, t.goal] + E.GAMMA * inner @ t.value(policy), atol=1e-12)

    def test_forbidding_the_way_through_sends_the_best_policy_around(self):
        t = E.published(0.0)
        best, value = t.best_policy()
        detour, detour_value = t.best_policy(forbidden={E.CRITICAL: (E.DOWN,)})
        self.assertEqual(best[E.CRITICAL], E.DOWN)
        self.assertNotEqual(detour[E.CRITICAL], E.DOWN)
        around = shortest(t, E.CRITICAL, t.goal, blocked_steps={(E.CRITICAL, E.BELOW)})
        self.assertEqual(shortest(t, E.CRITICAL, t.goal), 5)
        self.assertGreater(around, 15)
        np.testing.assert_allclose(value[E.CRITICAL], 0.95 ** 4)
        np.testing.assert_allclose(detour_value[E.CRITICAL], 0.95 ** (around - 1), rtol=1e-9)
        # From the start the detour costs nothing: the other route is just as short.
        np.testing.assert_allclose(detour_value[t.start], value[t.start], rtol=1e-9)

    def test_crossing_read_out(self):
        quiet, noisy = E.published(0.0), E.published()
        step = (E.CRITICAL, E.BELOW)
        best, _ = quiet.best_policy()
        self.assertAlmostEqual(quiet.outcomes(E.policy_from_moves(best), crossing=step)["crossed"][E.CRITICAL], 1.0)
        detour, _ = quiet.best_policy(forbidden={E.CRITICAL: (E.DOWN,)})
        self.assertEqual(quiet.outcomes(E.policy_from_moves(detour), crossing=step)["crossed"][E.CRITICAL], 0.0)
        detour, _ = noisy.best_policy(forbidden={E.CRITICAL: (E.DOWN,)})
        crossed = noisy.outcomes(E.policy_from_moves(detour), crossing=step)["crossed"][E.CRITICAL]
        self.assertTrue(0.0 < crossed < 0.2, crossed)      # only by accident

    def test_a_policy_that_stays_put_times_out(self):
        t = E.published(0.0)
        out = t.outcomes(E.policy_from_moves(np.zeros(100, int)))
        np.testing.assert_allclose(out["timeout"][t.free_cells()], 1.0)
        np.testing.assert_allclose(t.value(E.policy_from_moves(np.zeros(100, int))), 0.0)

    def test_one_hot_observation(self):
        obs = E.one_hot([0, 67])
        self.assertEqual(obs.shape, (2, 100))
        self.assertEqual((obs[0, 0], obs[1, 67], obs.sum()), (1.0, 1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
