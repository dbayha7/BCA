"""Tests of margin_verdict.py on made-up numbers only. No saved run and no dataset of the testbed is opened:
the criteria forbid computing marks M3 to M7 on any real data before the text is committed.

    python -I -B -m unittest discover -s experiments/gridworld_critical_v1 -p "test_margin_verdict.py" -v

The map and its rules are used (they are not data). Critics, margins, row counts and audit rows are invented here.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import critical_env as E  # noqa: E402
import margin_verdict as M  # noqa: E402
import published_collect as C  # noqa: E402

TASK = E.published()
FREE = TASK.free_cells()
BEST = TASK.best_policy()[0]
POLICY = E.policy_from_moves(BEST)
TRUTH = TASK.command_values(TASK.value(POLICY))
FRESH = tuple(range(910001, 910016))          # made-up seeds: no such dataset exists
S1_OK = {"passed": True, "reasons": []}
CELLS, MOVES = E.SIZE * E.SIZE, E.N_MOVES


# ---- made-up blocks -------------------------------------------------------------------------------------------
def made_up_counts(thin_cells=45):
    """Row counts per pair. The best policy's command has 40 rows at every open cell but five. At the first
    `thin_cells` open cells two more commands have 3 and 7 rows; at the next ten a third has 20."""
    counts = np.zeros((CELLS, MOVES), int)
    for k, cell in enumerate(FREE):
        main = int(BEST[cell])
        if k >= len(FREE) - 5:
            counts[cell, main] = 5
            continue
        counts[cell, main] = 40
        if k < thin_cells:
            counts[cell, (main + 1) % MOVES], counts[cell, (main + 2) % MOVES] = 3, 7
        elif k < thin_cells + 10:
            counts[cell, (main + 3) % MOVES] = 20
    return counts


def plain_counts():
    counts = np.zeros((CELLS, MOVES), int)
    counts[FREE, BEST[FREE]] = 40
    return counts


def made_up_audit(counts):
    """1,280 invented audit rows. In each class every tenth row is outside the margin."""
    cls = M.support_classes(counts)
    pairs = np.argwhere(counts > 0)
    picks = pairs[np.arange(M.AUDIT_ROWS) % len(pairs)]
    cell, command = picks[:, 0], picks[:, 1]
    thin, well = cls["thin"][cell, command], cls["well"][cell, command]
    size = np.full(M.AUDIT_ROWS, 0.05)
    for mask in (thin, well, ~thin & ~well):
        size[np.flatnonzero(mask)[9::10]] = 0.2
    residual = size * np.where(np.arange(M.AUDIT_ROWS) % 2, 1.0, -1.0)
    q1 = np.linspace(0.0, 1.0, M.AUDIT_ROWS)
    return {"cell": cell, "command": command, "target": residual + q1, "q1": q1, "q2": q1 + 0.01,
            "residual": residual, "margin": np.full(M.AUDIT_ROWS, 0.09)}


def made_up_run(seed, train_seed, counts, kind="oracle", recipe=M.HALF_RANDOM, audit=None):
    """An invented critic around the best policy's true values. Well recorded commands are valued almost right.
    Thinly recorded ones are valued too low, more so with 3 rows than with 7, except at every second eligible cell,
    where one thinly recorded command that is truly worse is valued above the best. The oracle margin is the size
    of the critic's error; the flat one is the same everywhere."""
    rng = np.random.default_rng([seed, train_seed])
    cls = M.support_classes(counts)
    shape = (CELLS, MOVES)
    error = rng.uniform(0.001, 0.01, shape) * rng.choice([-1.0, 1.0], shape)
    low = np.where(counts == 3, rng.uniform(0.04, 0.05, shape), rng.uniform(0.02, 0.03, shape))
    error = np.where(cls["thin"], -low, error)
    best = np.where(cls["recorded"], TRUTH, -np.inf).max(axis=1)
    for cell in np.flatnonzero(cls["eligible"])[::2]:
        for command in np.flatnonzero(cls["thin"][cell]):
            if best[cell] - TRUTH[cell, command] > 0.06:
                error[cell, command] = best[cell] - TRUTH[cell, command] + 0.1
                break
    q1 = TRUTH + error
    margin = np.abs(error) if kind == "oracle" else np.full(shape, 0.05)
    coverage = 0.9 if audit is None else float((np.abs(audit["residual"]) <= audit["margin"]).mean())
    return {"recipe": recipe, "data_seed": seed, "train_seed": train_seed, "reference_ready": True, "support_floor": 0.0,
            "saved_coverage": coverage, "commands": POLICY.copy(), "q1": q1, "q2": q1 + rng.uniform(-0.01, 0.01, shape),
            "margin": margin, "support": rng.normal(size=shape), "audit": audit}


def made_up_block(seeds=FRESH, kind="oracle", audit=True):
    block = {"label": "made_up", "task": TASK, "candidates": list(seeds), "passing_w0": list(seeds), "runs": [],
             "counts": {}, "executed": {}}
    for seed in seeds:
        counts, plain = made_up_counts(), plain_counts()
        for recipe in (M.HALF_RANDOM, M.BEST_POLICY, M.RANDOM_STARTS):
            block["counts"][(recipe, seed)] = block["executed"][(recipe, seed)] = counts if recipe == M.HALF_RANDOM else plain
        rows = made_up_audit(counts) if audit else None
        for train_seed in (20, 21, 22):
            block["runs"].append(made_up_run(seed, train_seed, counts, kind, audit=rows))
            for recipe in (M.BEST_POLICY, M.RANDOM_STARTS):
                block["runs"].append(made_up_run(seed, train_seed, plain, kind, recipe))
    return block


def runs_of(block, recipe, seed):
    return [run for run in block["runs"] if run["recipe"] == recipe and run["data_seed"] == seed]


# ---- words of section 2 ---------------------------------------------------------------------------------------
class Words(unittest.TestCase):
    def test_the_criteria_s_example_of_holds(self):
        test = M.holds([0.04, 0.03, 0.05, 0.02, 0.04, -0.01, 0.03, 0.02, 0.06])
        self.assertEqual(test["datasets_above"], 8)
        self.assertEqual(test["p"], 2 / 512)
        self.assertTrue(test["holds"])

    def test_a_tie_counts_against(self):
        self.assertFalse(M.holds([0.0] * 9)["holds"])
        self.assertEqual(M.holds([1e-15] * 9)["datasets_above"], 0)
        self.assertEqual(M.holds([0.0] * 9)["p"], 1.0)

    def test_holds_needs_six_of_nine_and_nine_values(self):
        self.assertFalse(M.holds([1.0] * 5 + [-0.001] * 4)["holds"])           # p is small, only five are above
        self.assertTrue(M.holds([1.0] * 6 + [-0.001] * 3)["holds"])
        self.assertFalse(M.holds([1.0] * 8)["holds"])
        self.assertFalse(M.holds([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, -1.0, -1.0, -1.0])["holds"])   # six above, p too large

    def test_tied_values_share_their_average_rank(self):
        self.assertEqual(list(M.average_ranks([10, 20, 20, 30])), [1.0, 2.5, 2.5, 4.0])
        self.assertEqual(list(M.average_ranks([5, 5, 5])), [2.0, 2.0, 2.0])

    def test_rank_correlation(self):
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [10, 20, 30, 400]), 1.0)
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [9, 7, 5, 1]), -1.0)
        self.assertEqual(M.spearman([3, 3, 3, 3], [1, 2, 3, 4]), 0.0)          # no spread scores 0
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), 0.8)

    def test_partial_rank_correlation(self):
        held = np.arange(8.0)
        both = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 3.5])
        self.assertAlmostEqual(M.partial_spearman(both, both, held), 1.0)
        self.assertEqual(M.partial_spearman(held * 2, held ** 3, held), 0.0)   # both only follow the held value
        rng = np.random.default_rng(0)
        value, noise_a, noise_b = rng.normal(size=400), rng.normal(size=400), rng.normal(size=400)
        self.assertGreater(M.spearman(value + 0.3 * noise_a, value + 0.3 * noise_b), 0.7)
        self.assertLess(abs(M.partial_spearman(value + 0.3 * noise_a, value + 0.3 * noise_b, value)), 0.15)

    def test_the_criteria_s_example_of_classes(self):
        counts = np.zeros((CELLS, MOVES), int)
        counts[12] = [3, 0, 31, 6, 0]                                         # stay, up, down, left, right
        counts[13, E.DOWN] = 15                                                # a cell the data barely visit
        cls = M.support_classes(counts)
        self.assertEqual(list(cls["well"][12]), [False, False, True, False, False])
        self.assertEqual(list(cls["thin"][12]), [True, False, False, True, False])
        self.assertEqual(list(cls["never"][12]), [False, True, False, False, True])
        self.assertTrue(cls["visited"][12] and cls["several"][12] and cls["eligible"][12])
        self.assertFalse(cls["visited"][13] or cls["recorded"][13].any() or cls["thin"][13].any() or cls["never"][13].any())
        self.assertEqual(int(M.support_classes(counts, 8)["thin"][12].sum()), 2)
        self.assertEqual(int(M.support_classes(counts, 32)["well"][12].sum()), 0)

    def test_in_range_reads_open_cells_only(self):
        q1 = np.zeros((CELLS, MOVES))
        q1[sorted(TASK.walls)[0]] = 9.0
        self.assertTrue(M.in_range(q1, FREE))
        q1[FREE[3], 2] = -1.5
        self.assertTrue(M.in_range(q1, FREE))
        q1[FREE[3], 2] = -1.5001
        self.assertFalse(M.in_range(q1, FREE))


# ---- M3 to M5 --------------------------------------------------------------------------------------------------
class Tracking(unittest.TestCase):
    def setUp(self):
        self.counts = made_up_counts()
        self.cls = M.support_classes(self.counts)
        rng = np.random.default_rng(1)
        self.size = rng.uniform(0.0, 1.0, (CELLS, MOVES))
        self.truth = rng.uniform(0.0, 1.0, (CELLS, MOVES))

    def test_a_signal_equal_to_the_error_size_scores_one(self):
        got = M.tracking(self.size, self.size, self.truth, self.cls)
        self.assertAlmostEqual(got["M3"], 1.0)
        self.assertAlmostEqual(got["M4"], 1.0)
        self.assertAlmostEqual(got["M5"], 1.0)

    def test_a_signal_that_depends_only_on_the_command_scores_zero(self):
        signal = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), (CELLS, 1))
        self.assertEqual(M.tracking(signal, self.size, self.truth, self.cls), {"M3": 0.0, "M4": 0.0, "M5": 0.0})

    def test_a_reversed_signal_scores_minus_one(self):
        self.assertAlmostEqual(M.tracking(-self.size, self.size, self.truth, self.cls)["M3"], -1.0)

    def test_commands_are_weighted_by_their_cell_counts(self):
        counts = np.zeros((CELLS, MOVES), int)
        counts[FREE[:10], 0] = counts[FREE[:10], 1] = 20                       # two commands at 10 cells
        counts[FREE[:20], 2] = 20                                              # one command at 20 cells
        cls = M.support_classes(counts)
        signal = self.size.copy()
        signal[:, 2] = -self.size[:, 2]
        self.assertAlmostEqual(M.tracking(signal, self.size, self.truth, cls)["M3"], (10 + 10 - 20) / 40)

    def test_fewer_than_three_commands_give_no_value(self):
        counts = np.zeros((CELLS, MOVES), int)
        counts[FREE[:12], 0] = counts[FREE[:12], 1] = 20
        counts[FREE[:9], 2] = 20                                               # nine cells: one short of ten
        got = M.tracking(self.size, self.size, self.truth, M.support_classes(counts))
        self.assertIsNone(got["M3"])
        self.assertIsNone(got["M4"])
        self.assertIsNone(got["M5"])                                           # nothing is thinly recorded

    def test_m5_reads_thinly_recorded_commands_only(self):
        signal = np.where(self.cls["thin"], self.size, -self.size)
        got = M.tracking(signal, self.size, self.truth, self.cls)
        self.assertAlmostEqual(got["M5"], 1.0)
        self.assertLess(got["M3"], 0.9)

    def test_the_worst_tenth_of_cells_is_left_out_rounded_up(self):
        keep = M.without_worst_tenth(self.size, self.cls)
        visited = int(self.cls["visited"].sum())
        self.assertEqual(int((~keep).sum()), int(np.ceil(visited / 10)))
        level = {cell: self.size[cell][self.cls["recorded"][cell]].mean() for cell in np.flatnonzero(self.cls["visited"])}
        self.assertGreaterEqual(min(level[c] for c in np.flatnonzero(~keep)),
                                max(level[c] for c in level if keep[c]))


# ---- M6, M7 ---------------------------------------------------------------------------------------------------
class Decisions(unittest.TestCase):
    """Hand-built cells. Commands: 0 stay, 1 up, 2 down, 3 left, 4 right."""

    def setUp(self):
        self.counts = np.zeros((CELLS, MOVES), int)
        self.truth, self.value, self.charge = (np.zeros((CELLS, MOVES)) for _ in range(3))

    def cell(self, number, rows, truth, value, charge):
        self.counts[number], self.truth[number], self.value[number], self.charge[number] = rows, truth, value, charge

    def read(self, tol=M.TOL):
        return M.decisions(self.value, self.charge, self.truth, M.support_classes(self.counts), tol)

    def test_misled_then_corrected(self):
        self.cell(1, [0, 0, 30, 5, 0], [0, 0, 0.8, 0.5, 0], [0, 0, 0.7, 0.9, 0], [0, 0, 0.0, 0.3, 0])
        got = self.read()
        self.assertEqual((got["eligible"], got["misled"], got["still_wrong"], got["corrected"]), (1, 1, 0, 1))

    def test_misled_and_still_wrong(self):
        self.cell(1, [0, 0, 30, 5, 0], [0, 0, 0.8, 0.5, 0], [0, 0, 0.7, 0.9, 0], [0, 0, 0.0, 0.1, 0])
        got = self.read()
        self.assertEqual((got["misled"], got["still_wrong"], got["corrected"]), (1, 1, 0))
        self.assertEqual(got["moved_to_a_wrong_well_recorded_command"], 0)     # it stayed on the thin command

    def test_moving_from_one_wrong_command_to_another_is_not_a_correction(self):
        self.cell(1, [20, 0, 30, 5, 0], [0.4, 0, 0.8, 0.5, 0], [0.85, 0, 0.7, 0.9, 0], [0.0, 0, 0.0, 0.3, 0])
        got = self.read()
        self.assertEqual((got["misled"], got["still_wrong"], got["corrected"]), (1, 1, 0))
        self.assertEqual(got["moved_to_a_wrong_well_recorded_command"], 1)

    def test_a_right_well_recorded_choice_broken(self):
        self.cell(2, [0, 20, 30, 0, 0], [0, 0.5, 0.8, 0, 0], [0, 0.6, 0.9, 0, 0], [0, 0.0, 0.4, 0, 0])
        got = self.read()
        self.assertEqual((got["eligible"], got["right_well_recorded"], got["broken"]), (0, 1, 1))

    def test_a_wrong_well_recorded_choice_is_in_neither_count(self):
        self.cell(2, [0, 20, 30, 0, 0], [0, 0.5, 0.8, 0, 0], [0, 0.95, 0.9, 0, 0], [0, 0.0, 0.0, 0, 0])
        got = self.read()
        self.assertEqual((got["misled"], got["right_well_recorded"], got["broken"]), (0, 0, 0))

    def test_the_cost_of_caution(self):
        self.cell(3, [0, 0, 30, 5, 0], [0, 0, 0.5, 0.8, 0], [0, 0, 0.6, 0.9, 0], [0, 0, 0.0, 0.4, 0])
        got = self.read()
        self.assertEqual((got["misled"], got["right_thin"], got["right_thin_broken"]), (0, 1, 1))
        self.assertEqual((got["right_well_recorded"], got["broken"]), (0, 0))  # thin right choices are not in M7

    def test_within_the_tolerance_is_right(self):
        self.cell(1, [0, 0, 30, 5, 0], [0, 0, 0.80, 0.76, 0], [0, 0, 0.7, 0.9, 0], [0, 0, 0, 0, 0])
        self.assertEqual(self.read()["misled"], 0)
        self.assertEqual(self.read(0.02)["misled"], 1)

    def test_only_recorded_commands_at_visited_cells_are_choices(self):
        self.cell(1, [0, 0, 30, 5, 0], [0, 0.99, 0.8, 0.78, 0], [0, 5.0, 0.9, 0.7, 0], [0, 0, 0, 0, 0])
        self.cell(4, [0, 0, 9, 5, 0], [0, 0, 0.8, 0.1, 0], [0, 0, 0.1, 0.9, 0], [0, 0, 0, 0, 0])   # 14 rows: not visited
        got = self.read()
        self.assertEqual((got["eligible"], got["misled"], got["right_well_recorded"]), (1, 0, 1))

    def test_the_criteria_s_example_of_what_a_correction_needs(self):
        """The critic values one command 0.20 above the right one. Three margins reverse that only if the margin at
        the wrong command is larger by more than 0.20 / 3."""
        for extra, corrected in ((0.06, 0), (0.07, 1)):
            self.cell(1, [0, 0, 30, 5, 0], [0, 0, 0.8, 0.1, 0], [0, 0, 0.5, 0.7, 0], [0, 0, 3 * 0.059, 3 * (0.059 + extra), 0])
            self.assertEqual(self.read()["corrected"], corrected)

    def test_ties_go_to_the_first_command(self):
        self.cell(1, [0, 20, 30, 0, 0], [0, 0.5, 0.8, 0, 0], [0, 0.9, 0.9, 0, 0], [0, 0, 0, 0, 0])
        cls = M.support_classes(self.counts)
        self.assertEqual(int(M.choose(self.value, cls)[1]), 1)
        self.value[1] = [0, 0.9, 0.9 + 1e-13, 0, 0]                               # equal within 1e-12
        self.assertEqual(int(M.choose(self.value, cls)[1]), 1)
        self.value[1] = [0, 0.9, 0.9 + 1e-9, 0, 0]
        self.assertEqual(int(M.choose(self.value, cls)[1]), 2)

    def test_exactly_at_the_tolerance_is_right(self):
        for best, other in ((0.80, 0.75), (0.35, 0.30), (0.95, 0.90), (0.15, 0.10)):
            self.cell(1, [0, 0, 30, 5, 0], [0, 0, best, other, 0], [0, 0, 0.1, 0.9, 0], [0, 0, 0, 0, 0])
            self.assertEqual(self.read()["misled"], 0, (best, other))
        self.cell(1, [0, 0, 30, 5, 0], [0, 0, 0.80, 0.7499, 0], [0, 0, 0.1, 0.9, 0], [0, 0, 0, 0, 0])
        self.assertEqual(self.read()["misled"], 1)


class Copies(unittest.TestCase):
    def setUp(self):
        self.counts = made_up_counts()
        self.cls = M.support_classes(self.counts)
        self.signal = np.random.default_rng(2).uniform(0.0, 1.0, (CELLS, MOVES))

    def test_the_same_seeds_give_the_same_reorderings(self):
        _, a = M.reorderings(self.cls, (910001, 20), 20)
        _, b = M.reorderings(self.cls, (910001, 20), 20)
        _, c = M.reorderings(self.cls, (910001, 21), 20)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, c)))

    def test_a_copy_keeps_each_command_s_values_and_moves_them_among_its_cells(self):
        cells, orders = M.reorderings(self.cls, (1, 2), 50)
        copies = M.shuffled_copies(self.signal, cells, orders)
        self.assertEqual(copies.shape, (50, CELLS, MOVES))
        for command in range(MOVES):
            there = self.cls["recorded"][:, command]
            self.assertTrue(np.array_equal(copies[:, ~there, command], np.tile(self.signal[~there, command], (50, 1))))
            for r in (0, 17, 49):
                self.assertTrue(np.array_equal(np.sort(copies[r, there, command]), np.sort(self.signal[there, command])))
        self.assertFalse(np.array_equal(copies[0], self.signal))

    def test_a_charge_that_is_the_same_at_every_cell_has_no_shuffled_advantage(self):
        run = made_up_run(910001, 20, self.counts)
        flat = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), (CELLS, 1))
        shuffle = M.reorderings(self.cls, (910001, 20), 30)
        got = M.decision_reading(run["q1"], flat, TRUTH, self.cls, M.TOL, shuffle)
        self.assertEqual(got["still_wrong_after_copy"], float(got["still_wrong"]))

    def test_an_error_sized_charge_beats_its_own_copy(self):
        run = made_up_run(910001, 20, self.counts)
        got = M.decision_reading(run["q1"], M.KAPPA * np.abs(run["q1"] - TRUTH), TRUTH, self.cls, M.TOL,
                                 M.reorderings(self.cls, (910001, 20), 200))
        self.assertGreater(got["misled"], 0)
        self.assertEqual(got["still_wrong"], 0)
        self.assertGreater(got["still_wrong_after_copy"], 0.5 * got["misled"])

    def test_a_rival_is_scaled_to_the_average_of_three_margins(self):
        margin = np.full((CELLS, MOVES), 0.04)
        charge = M.scaled_charge(self.signal, margin, self.cls)
        self.assertAlmostEqual(float(charge[self.cls["recorded"]].mean()), 0.12)
        self.assertEqual(float(np.abs(M.scaled_charge(np.zeros((CELLS, MOVES)), margin, self.cls)).max()), 0.0)


# ---- M1, M2 and the readings of section 8 ----------------------------------------------------------------------
class Coverage(unittest.TestCase):
    def test_coverage_overall_and_by_class(self):
        counts = np.zeros((CELLS, MOVES), int)
        counts[12] = [3, 0, 31, 6, 0]
        counts[13, 2] = 9
        rows = {"cell": np.array([12, 12, 12, 12, 12, 12, 13, 12]), "command": np.array([2, 2, 2, 0, 3, 3, 2, 1]),
                "residual": np.array([0.01, -0.5, 0.02, 0.2, -0.01, 0.03, 0.0, 0.0]), "margin": np.full(8, 0.1)}
        got = M.audit_coverage(rows, M.support_classes(counts))
        self.assertEqual((got["rows"], got["well_rows"], got["thin_rows"], got["rows_in_neither_class"]), (8, 3, 3, 2))
        self.assertAlmostEqual(got["overall"], 6 / 8)
        self.assertAlmostEqual(got["well"], 2 / 3)
        self.assertAlmostEqual(got["thin"], 2 / 3)

    def test_a_class_with_no_rows_has_no_value(self):
        counts = np.zeros((CELLS, MOVES), int)
        counts[12, 2] = 40
        rows = {"cell": np.array([12, 12]), "command": np.array([2, 2]), "residual": np.array([0.0, 1.0]),
                "margin": np.array([0.1, 0.1])}
        got = M.audit_coverage(rows, M.support_classes(counts))
        self.assertIsNone(got["thin"])
        self.assertEqual(got["well"], 0.5)

    def test_the_lower_bound_share_and_the_strict_reading(self):
        counts = np.zeros((CELLS, MOVES), int)
        counts[12] = [0, 0, 30, 5, 0]
        value, margin, truth = (np.zeros((CELLS, MOVES)) for _ in range(3))
        value[12], truth[12], margin[12] = [0, 0, 0.9, 0.9, 0], [0, 0, 0.8, 0.5, 0], [0, 0, 0.05, 0.1, 0]
        got = M.bound_readings(value, margin, truth, M.support_classes(counts))
        self.assertEqual(got["recorded"]["pairs"], 2)
        self.assertEqual(got["well"]["lower_bound"], 1.0)                      # 0.9 - 0.15 is not above 0.8
        self.assertEqual(got["thin"]["lower_bound"], 0.0)                      # 0.9 - 0.30 is above 0.5
        self.assertEqual(got["recorded"]["lower_bound_no_margin"], 0.0)
        self.assertEqual(got["well"]["lower_bound_constant_margin"], 1.0)      # constant 0.075: 0.9 - 0.225
        self.assertEqual(got["thin"]["error_inside_one_margin"], 0.0)

    def test_the_margin_at_never_commanded_commands(self):
        counts = np.zeros((CELLS, MOVES), int)
        counts[12] = [0, 0, 30, 5, 0]
        margin = np.zeros((CELLS, MOVES))
        margin[12] = [0.3, 0.3, 0.1, 0.2, 0.3]
        got = M.never_commanded_reading(margin, M.support_classes(counts))
        self.assertEqual(got["cells"], 1)
        self.assertAlmostEqual(got["margin_at_never_commanded"], 0.3)
        self.assertAlmostEqual(got["margin_at_recorded"], 0.15)


# ---- section 9 ------------------------------------------------------------------------------------------------
class Lava(unittest.TestCase):
    SPOT, MOVE = E.SECOND_CRITICAL, E.UP                                       # cell 83, on the best route: lava above

    def into_lava(self):
        policy = BEST.copy()
        policy[self.SPOT] = self.MOVE
        return E.policy_from_moves(policy)

    def test_a_policy_that_never_commands_a_lava_move_scores_zero(self):
        got = M.lava_reading(TASK, POLICY, plain_counts(), plain_counts())
        self.assertEqual((got["counted"], got["cells_with_few_rows"]), (0.0, 0.0))

    def test_without_noise_the_outcome_is_every_lava_ending(self):
        task = E.published(noise=0.0)
        best = task.best_policy()[0]
        best[self.SPOT] = self.MOVE
        policy = E.policy_from_moves(best)
        counts = np.zeros((CELLS, MOVES), int)
        counts[task.free_cells(), 0] = 40
        got = M.lava_reading(task, policy, counts, counts)
        self.assertAlmostEqual(got["counted"], 100.0 * task.outcomes(policy)["lava"][task.start])
        self.assertGreater(got["counted"], 99.0)
        self.assertAlmostEqual(got["never_executed"], got["counted"])

    def test_with_noise_it_counts_only_endings_after_a_commanded_lava_move(self):
        policy, counts = self.into_lava(), plain_counts()
        counts[self.SPOT] = [40, 0, 0, 0, 0]
        got = M.lava_reading(TASK, policy, counts, counts)
        total = 100.0 * TASK.outcomes(policy)["lava"][TASK.start]
        self.assertGreater(got["counted"], 50.0)
        self.assertLess(got["counted"], total)                                 # noise after safe commands is left out
        executed = counts.copy()
        executed[self.SPOT, self.MOVE] = 2                                     # executed through the noise
        split = M.lava_reading(TASK, policy, counts, executed)
        self.assertAlmostEqual(split["executed_through_noise"], got["counted"])
        self.assertEqual(split["never_executed"], 0.0)

    def test_a_recorded_lava_command_and_a_barely_visited_cell_do_not_count(self):
        policy = self.into_lava()
        counts = plain_counts()
        counts[self.SPOT] = [40, 3, 0, 0, 0]                                   # the lava command is in three rows
        self.assertEqual(M.lava_reading(TASK, policy, counts, counts)["counted"], 0.0)
        counts[self.SPOT] = [15, 0, 0, 0, 0]                                   # fifteen rows: not a visited cell
        got = M.lava_reading(TASK, policy, counts, counts)
        self.assertEqual(got["counted"], 0.0)
        self.assertGreater(got["cells_with_few_rows"], 50.0)

    def test_the_critic_s_ranking_of_never_commanded_lava_moves(self):
        counts = plain_counts()
        q1, margin = np.zeros((CELLS, MOVES)), np.zeros((CELLS, MOVES))
        q1[E.CRITICAL, E.LEFT], margin[E.CRITICAL, E.LEFT] = 0.5, 0.2
        got = M.lava_reading(TASK, POLICY, counts, counts, q1, margin)
        self.assertEqual(got["cells_with_a_never_commanded_lava_move"], 10)         # 12 lava moves at 10 cells
        self.assertEqual(got["critic_rates_one_above_every_recorded_command"], 1)
        self.assertEqual(got["the_same_after_3_margins"], 0)
        q1[E.CRITICAL, E.RIGHT] = 0.4                                          # both lava moves of one cell: one case
        again = M.lava_reading(TASK, POLICY, counts, counts, q1, margin)
        self.assertEqual(again["critic_rates_one_above_every_recorded_command"], 1)
        self.assertEqual(again["the_same_after_3_margins"], 1)


# ---- the marks on nine dataset values ---------------------------------------------------------------------------
class Marks(unittest.TestCase):
    def test_m1(self):
        self.assertTrue(M.mark_coverage([0.9] * 9, M.COVER_MEAN)["passed"])
        self.assertTrue(M.mark_coverage([0.88] * 9, M.COVER_MEAN)["passed"])               # exactly at the bar
        self.assertFalse(M.mark_coverage([0.87] * 9, M.COVER_MEAN)["passed"])
        self.assertFalse(M.mark_coverage([0.99] * 5 + [0.84] * 4, M.COVER_MEAN)["passed"])  # mean passes, five of nine
        self.assertFalse(M.mark_coverage([0.86] * 6 + [0.5] * 3, M.COVER_MEAN)["passed"])   # six of nine, mean fails
        self.assertIsNone(M.mark_coverage([0.9] * 8 + [None], M.COVER_MEAN)["passed"])
        self.assertTrue(M.mark_coverage([0.97] * 9, M.COVER_MEAN)["wider_than_its_label"])
        self.assertTrue(M.mark_coverage([0.97] * 9, M.COVER_MEAN)["passed"])               # wider does not fail
        self.assertFalse(M.mark_coverage([0.95] * 9, M.COVER_MEAN)["wider_than_its_label"])  # exactly 95% is not above

    def test_m2_asks_each_class(self):
        self.assertTrue(M.mark_m2([0.86] * 9, [0.9] * 9)["passed"])
        self.assertFalse(M.mark_m2([0.80] * 9, [0.9] * 9)["passed"])
        self.assertIsNone(M.mark_m2([None] * 9, [0.9] * 9)["passed"])
        self.assertFalse(M.mark_m2([None] * 9, [0.5] * 9)["passed"])                       # a class that fails, fails M2
        self.assertIs(M.mark_m2([0.5] * 9, [None] * 8 + [0.9])["passed"], False)

    def test_m3(self):
        good = [0.4] * 9
        self.assertTrue(M.mark_tracking(good, M.TRACK_BAR, [0.4] * 18 + [-0.1] * 9)["passed"])
        self.assertFalse(M.mark_tracking(good, M.TRACK_BAR, [0.4] * 17 + [-0.1] * 10)["passed"])   # under two thirds
        self.assertFalse(M.mark_tracking(good, M.TRACK_BAR, [0.4] * 17 + [None] * 10)["passed"])   # no value: not above
        self.assertFalse(M.mark_tracking([0.29] * 9, M.TRACK_BAR, [0.29] * 27)["passed"])
        self.assertFalse(M.mark_tracking([2.0] * 5 + [-0.01] * 4, M.TRACK_BAR, [1.0] * 27)["passed"])
        self.assertTrue(M.mark_tracking([0.01] * 9)["passed"])                                      # M4, M5: above 0 holds
        self.assertFalse(M.mark_tracking([0.0] * 9)["passed"])

    def test_m6(self):
        misled, copy = [0.20] * 9, [0.16] * 9
        self.assertTrue(M.mark_m6(misled, [0.10] * 9, copy)["passed"])
        self.assertFalse(M.mark_m6(misled, [0.11] * 9, copy)["passed"])                   # more than half left wrong
        self.assertFalse(M.mark_m6(misled, [0.10] * 9, [0.12] * 9)["passed"])              # more than 3/4 of the copy's
        both = M.mark_m6(misled, [0.20] * 9, copy)
        self.assertFalse(both["passed"] or both["corrects_at_all"])
        only_copy = M.mark_m6(misled, [0.10] * 9, [0.10] * 9)
        self.assertTrue(only_copy["corrects_at_all"] and not only_copy["passed"])
        self.assertIsNone(M.mark_m6(misled, [0.1] * 9, [None] * 9)["passed"])

    def test_m7(self):
        self.assertTrue(M.mark_m7([0.01] * 9, 3, 4)["passed"])
        self.assertFalse(M.mark_m7([0.011] * 9, 3, 4)["passed"])
        self.assertFalse(M.mark_m7([0.0] * 9, 4, 4)["passed"])                             # a tie fails
        zero = M.mark_m7([0.0] * 9, 0, 0)
        self.assertTrue(zero["first_clause"] and not zero["second_clause"] and not zero["passed"])

    def test_s6(self):
        self.assertTrue(M.check_s6([0.05] * 9, [1] * 8 + [0])["passed"])
        self.assertFalse(M.check_s6([0.049] * 9, [1] * 9)["passed"])
        self.assertFalse(M.check_s6([0.5] * 9, [1] * 7 + [0, 0])["passed"])


# ---- guards and verdict -----------------------------------------------------------------------------------------
def marks(fail=(), reference_fail=(), s6=True, no_value=(), m7_first=True, m6_at_all=True):
    """Marks that all pass, with the named ones changed."""
    def one(mark, failed, missing):
        passed = None if missing else not failed
        out = {"passed": passed}
        if mark == "M2":
            out.update(thin={"passed": passed}, well={"passed": True if passed is None else True})
        if mark == "M6":
            out["corrects_at_all"] = m6_at_all
        if mark == "M7":
            out.update(first_clause=None if missing else (m7_first if failed else True), second_clause=not failed)
        return out
    return {"margin": {m: one(m, m in fail, m in no_value) for m in M.MARKS},
            "reference_s5": {m: one(m, m in reference_fail, False) for m in ("M3", "M4", "M5")},
            "reference_s7": {m: one(m, m in reference_fail, False) for m in ("M6", "M7")},
            "S6": {"passed": s6}}


class Verdict(unittest.TestCase):
    def judge(self, s1=True, enough=True, **changes):
        made = marks(**changes)
        return M.judge(made, s1, enough), made

    def test_all_seven_pass(self):
        judged, _ = self.judge()
        self.assertEqual(judged["verdict"], "validated")

    def test_a_pass_stands_whatever_the_reference_margins_do(self):
        judged, _ = self.judge(reference_fail=("M3", "M4", "M5", "M6", "M7"))
        self.assertEqual(judged["verdict"], "validated")

    def test_a_fail_counts_only_if_the_reference_passed_the_same_mark(self):
        judged, _ = self.judge(fail=("M4",))
        self.assertEqual((judged["verdict"], judged["status"]["M4"][0]), ("not validated", "fail"))
        judged, _ = self.judge(fail=("M4",), reference_fail=("M4",))
        self.assertEqual((judged["verdict"], judged["status"]["M4"][0]), ("inconclusive", "failed but not judged"))
        judged, _ = self.judge(fail=("M4",), reference_fail=("M3",))               # another mark's reference: no excuse
        self.assertEqual(judged["verdict"], "not validated")
        judged, _ = self.judge(fail=("M7",), reference_fail=("M6",))
        self.assertEqual(judged["verdict"], "not validated")

    def test_m1_and_m2_have_no_guard(self):
        judged, _ = self.judge(fail=("M1",), reference_fail=("M3", "M4", "M5", "M6", "M7"))
        self.assertEqual(judged["verdict"], "not validated")

    def test_s1_or_too_few_datasets_judge_nothing(self):
        for s1, enough in ((False, True), (True, False)):
            judged, _ = self.judge(s1=s1, enough=enough, fail=("M1", "M3"))
            self.assertEqual(judged["verdict"], "inconclusive")
            self.assertTrue(all(word == "not judged" for word, _ in judged["status"].values()))
        self.assertEqual(self.judge(s1=False, enough=False)[0]["reason"], "S1 failed; fewer than nine valid datasets")

    def test_a_judged_fail_beats_a_mark_that_is_not_judged(self):
        judged, _ = self.judge(fail=("M1",), no_value=("M2",))
        self.assertEqual(judged["verdict"], "not validated")
        judged, _ = self.judge(no_value=("M2",))
        self.assertEqual(judged["verdict"], "inconclusive")
        self.assertIn("M2", judged["reason"])

    def test_when_s6_fails(self):
        judged, _ = self.judge(s6=False)
        self.assertEqual((judged["status"]["M6"][0], judged["status"]["M7"][0], judged["verdict"]),
                         ("not judged", "not judged", "inconclusive"))             # M7 cannot pass
        judged, _ = self.judge(s6=False, fail=("M6",))
        self.assertEqual(judged["verdict"], "inconclusive")                         # M6 is not judged at all
        judged, _ = self.judge(s6=False, fail=("M7",), m7_first=False)
        self.assertEqual((judged["status"]["M7"][0], judged["verdict"]), ("fail", "not validated"))
        judged, _ = self.judge(s6=False, fail=("M7",), m7_first=True)               # only its second clause failed
        self.assertEqual(judged["status"]["M7"][0], "not judged")
        made = marks(s6=False, fail=("M7",), m7_first=False)
        made["reference_s7"]["M7"]["passed"] = False                              # passed its first clause, not the mark
        self.assertEqual(M.judge(made, True, True)["status"]["M7"][0], "failed but not judged")
        self.assertEqual(M.judge(made, True, True)["verdict"], "inconclusive")

    def test_the_fixed_sentences(self):
        judged, made = self.judge(fail=("M3", "M6"))
        lines = M.fixed_sentences(judged, made, 24, [0.6, 0.5], 27)
        self.assertEqual(lines[0], "The margin is a calibrated one-step error bar.")
        self.assertIn("The margin is not shown to be larger where the critic's true error is larger.", lines)
        self.assertIn("Subtracting 3 margins is not shown to correct the critic's wrong choices more than a wrong-cell copy does.", lines)
        self.assertNotIn("The margin is larger where the critic's true error is larger.", lines)   # M3 failed

        judged, made = self.judge(fail=("M2", "M6", "M7"), m6_at_all=False, m7_first=False)
        lines = M.fixed_sentences(judged, made, 27, [], 27)
        self.assertEqual(lines[0], "The margin's 90% coverage does not hold for thinly recorded commands. "
                                   "The method promises it only over all rows.")
        self.assertEqual(lines[1], "The margin is larger where the critic's true error is larger.")
        self.assertEqual(lines[2], "Subtracting 3 margins is not shown to correct the critic's wrong choices at all.")
        self.assertEqual(lines[3], "Subtracting 3 margins breaks too many of the critic's right choices more than 1%.")
        self.assertEqual(len(lines), 4)

        judged, made = self.judge(fail=("M7",), m7_first=True)
        self.assertIn("Subtracting 3 margins breaks too many of the critic's right choices as many as it corrects.",
                      M.fixed_sentences(judged, made, 27, [], 27))

        judged, made = self.judge(s6=False, fail=("M1", "M6"), m6_at_all=False)       # M6's figures failed, S6 failed
        lines = M.fixed_sentences(judged, made, 27, [], 27)
        self.assertEqual(judged["verdict"], "not validated")
        self.assertIn("M6 failed but was not judged: S6 failed: too few misled cases.", lines)
        self.assertFalse(any(line.startswith("Subtracting 3 margins is not shown") for line in lines))

        judged, made = self.judge(fail=("M1", "M5"), reference_fail=("M5",))
        lines = M.fixed_sentences(judged, made, 27, [], 27)
        self.assertEqual(lines[0], "The margin's 90% coverage does not hold overall. The method promises it only over all rows.")
        self.assertTrue(lines[-1].startswith("M5 failed but was not judged"))

    def test_the_validated_sentence(self):
        judged, made = self.judge()
        line = M.fixed_sentences(judged, made, 24, [0.5, 0.6, 0.7], 27)[0]
        self.assertTrue(line.startswith("Validated once, on the standard learner's critic, for well recorded and thinly "
                                        "recorded commands, on half-random data on the published map (a recipe that is our "
                                        "choice on this map), in the 24 of 27 runs whose critic stayed in range. In the other "
                                        "runs the margin covered the one-step error 60.0% of the time."))
        self.assertTrue(line.endswith("No claim for never-commanded commands, for the policy's own softer commands, or "
                                      "for trained behavior."))
        self.assertIn("No run left the range.", M.fixed_sentences(judged, made, 27, [], 27)[0])

    def test_the_out_of_range_sentence(self):
        self.assertEqual(M.out_of_range_sentence([0.5, 0.7]), "In the 2 runs whose critic left the possible range the "
                         "margin covered the one-step error 60.0% of the time (target 90%).")


# ---- a whole made-up block --------------------------------------------------------------------------------------
class Block(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        block = made_up_block()
        runs_of(block, M.HALF_RANDOM, 910002)[0]["q1"][FREE[0], 0] = 2.0          # one run out of range: still valid
        for run in runs_of(block, M.HALF_RANDOM, 910003)[:2]:
            run["q1"][FREE[0], 0] = -2.0                                           # two out of range: S4 fails
        block["counts"][(M.HALF_RANDOM, 910005)] = made_up_counts(10)             # too few thin commands: S2 fails
        runs_of(block, M.BEST_POLICY, 910007)[1]["commands"] = E.policy_from_moves(np.zeros(CELLS, int))   # S3 fails
        cls.block = block
        cls.result = M.evaluate(block, S1_OK)

    def test_the_first_nine_valid_datasets_are_used(self):
        r = self.result
        self.assertEqual(r["mode"], "verdict")
        setup = {row["data_seed"]: row for row in r["setup"]}
        self.assertEqual((setup[910003]["S4"], setup[910005]["S2"], setup[910007]["S3"]), (False, False, False))
        self.assertEqual((setup[910005]["thin_commands"], setup[910005]["cells_with_a_thin_command"]), (20, 10))
        self.assertTrue(setup[910002]["valid"] and setup[910002]["runs_in_range"] == 2)
        self.assertEqual(r["datasets_used"], [910001, 910002, 910004, 910006, 910008, 910009, 910010, 910011, 910012])
        self.assertEqual((r["runs"], r["runs_in_range"]), (27, 26))
        self.assertEqual(len(r["coverage_in_out_of_range_runs_of_the_datasets_used"]), 1)
        self.assertEqual(len(r["coverage_in_out_of_range_runs_of_the_block"]), 3)       # two more on a dropped dataset
        self.assertTrue(r["out_of_range_sentence"].startswith("In the 3 runs whose critic left the possible range"))
        self.assertEqual(r["half_random_runs_in_the_block"], 45)

    def test_an_error_sized_margin_is_validated(self):
        r = self.result
        self.assertEqual({mark: word for mark, (word, _) in r["status"].items()}, {mark: "pass" for mark in M.MARKS})
        self.assertEqual(r["verdict"], "validated")
        self.assertIn("in the 26 of 27 runs whose critic stayed in range", r["sentences"][0])
        own = r["marks"]["margin"]
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in own["M3"]["values"] + own["M4"]["values"] + own["M5"]["values"]))
        self.assertEqual(own["M6"]["mean_still_wrong"], 0.0)
        self.assertGreater(own["M6"]["mean_copy_still_wrong"], 0.0)
        self.assertEqual(own["M7"]["right_to_wrong"], 0)
        self.assertGreater(own["M7"]["corrected"], 0)
        self.assertTrue(r["marks"]["S6"]["passed"])

    def test_dataset_values_use_in_range_runs_only(self):
        dataset = next(d for d in self.result["dataset_values"] if d["data_seed"] == 910002)
        self.assertEqual((dataset["runs"], dataset["runs_in_range"]), (3, 2))
        full = next(d for d in self.result["dataset_values"] if d["data_seed"] == 910001)
        self.assertEqual(dataset["decisions"]["margin"]["eligible"] * 3, full["decisions"]["margin"]["eligible"] * 2)

    def test_a_flat_margin_is_not_validated(self):
        result = M.evaluate(made_up_block(kind="flat"), S1_OK)
        status = {mark: word for mark, (word, _) in result["status"].items()}
        self.assertEqual((status["M1"], status["M2"]), ("pass", "pass"))
        self.assertEqual(result["marks"]["margin"]["M3"]["values"], [0.0] * 9)
        self.assertEqual(result["verdict"], "not validated")
        self.assertEqual((status["M6"], status["M7"]), ("fail", "fail"))           # the error-sized reference passes both
        self.assertEqual(result["sentences"][0], "The margin is a calibrated one-step error bar.")
        self.assertIn("Subtracting 3 margins is not shown to correct the critic's wrong choices at all.", result["sentences"])
        self.assertIn("Subtracting 3 margins breaks too many of the critic's right choices as many as it corrects.",
                      result["sentences"])
        for mark in ("M3", "M4", "M5"):                                            # fails here, counted or not
            self.assertIn(status[mark], ("fail", "failed but not judged"))

    def test_s1_failed_judges_nothing(self):
        result = M.evaluate(self.block, {"passed": False, "reasons": ["made up"]})
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertTrue(all(word == "not judged" for word, _ in result["status"].values()))
        self.assertEqual(result["sentences"], ["Inconclusive: S1 failed."])

    def test_fewer_than_nine_valid_datasets_is_inconclusive(self):
        block = made_up_block()
        for seed in FRESH[:7]:
            block["counts"][(M.HALF_RANDOM, seed)] = made_up_counts(10)
        result = M.evaluate(block, S1_OK)
        self.assertEqual((len(result["valid"]), result["verdict"]), (8, "inconclusive"))
        self.assertIn("fewer than nine valid datasets", result["reason"])

    def test_a_verdict_block_needs_fifteen_candidates_and_s1(self):
        with self.assertRaises(M.Defect):
            M.evaluate(made_up_block(FRESH[:12]), S1_OK)
        with self.assertRaises(M.Defect):
            M.evaluate(self.block, None)

    def test_a_dataset_must_have_three_runs(self):
        block = made_up_block()
        block["runs"].remove(runs_of(block, M.HALF_RANDOM, 910004)[0])
        with self.assertRaises(M.Defect):
            M.evaluate(block, S1_OK)

    def test_seen_data_never_get_a_verdict(self):
        spent = tuple(range(6501, 6516))
        result = M.evaluate(made_up_block(spent, audit=False))                    # made-up numbers under spent seed numbers
        self.assertEqual(result["mode"], "seen")
        self.assertIsNone(result["verdict"])
        self.assertNotIn("sentences", result)
        self.assertNotIn("status", result)
        self.assertEqual(result["standing"]["M3"], ("meets on seen data", ""))
        self.assertEqual(result["standing"]["M2"], ("not read", "the runs saved no audit rows"))
        allowed = {"meets on seen data", "does not meet on seen data", "not read"}
        self.assertTrue(all(word in allowed for word, _ in result["standing"].values()))
        flat = M.evaluate(made_up_block(spent, kind="flat"))
        self.assertEqual(flat["standing"]["M3"], ("does not meet on seen data", ""))
        few = M.evaluate(made_up_block(tuple(range(6401, 6406)), audit=False))
        self.assertTrue(all(standing == ("not read", "fewer than nine valid datasets") for standing in few["standing"].values()))
        self.assertEqual(len(few["dataset_values"]), 5)

    def test_a_block_this_script_already_judged_is_seen(self):
        block = made_up_block()
        block["judged_before"] = True
        result = M.evaluate(block, S1_OK)
        self.assertEqual((result["mode"], result["verdict"]), ("seen", None))

    def test_a_verdict_block_needs_its_audit_rows(self):
        with self.assertRaises(M.Defect):
            M.evaluate(made_up_block(audit=False), S1_OK)

    def test_every_run_out_of_range(self):
        block = made_up_block()
        for run in block["runs"]:
            if run["recipe"] == M.HALF_RANDOM:
                run["q1"][FREE[0], 0] = 3.0
        result = M.evaluate(block, S1_OK)
        self.assertEqual((result["valid"], result["verdict"]), ([], "inconclusive"))
        self.assertTrue(result["out_of_range_sentence"].startswith("In the 45 runs whose critic left the possible range"))
        M.render(result, "made_up", {"criteria": "0" * 64, "verdict_script": "1" * 64})

    def test_both_reference_margins_are_read_on_m3_to_m7(self):
        marks = self.result["marks"]
        self.assertEqual(sorted(marks["reference_s5"]), ["M3", "M4", "M5", "M6", "M7"])
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in marks["reference_s7"]["M3"]["values"]))
        text = M.render(self.result, "made_up", {"criteria": "0" * 64, "verdict_script": "1" * 64})
        for words in ("In neither class", "Share of these broken", "In-range runs that gave no value",
                      "Critic rates a lava move highest"):
            self.assertIn(words, text)

    def test_the_result_can_be_written(self):
        prints = {"criteria": "0" * 64, "verdict_script": "1" * 64}
        for result in (self.result, M.evaluate(made_up_block(tuple(range(6501, 6516)), audit=False)),
                       M.evaluate(made_up_block(kind="flat"), S1_OK)):
            json.dumps(M.jsonable(result), allow_nan=False)
            text = M.render(result, "made_up", prints)
            self.assertFalse(any(chr(code) in text for code in (0x2013, 0x2014)))     # no long dashes
            self.assertNotIn("scor" + "er", text)                                    # the word is critic
        self.assertIn("## Verdict: validated", M.render(self.result, "made_up", prints))

    def test_the_lava_outcome_of_a_policy_that_commands_no_lava_move(self):
        lava = self.result["lava_outcome"]
        self.assertEqual((lava[M.BEST_POLICY]["mean"], lava[M.BEST_POLICY]["L0_room"]), (0.0, False))
        self.assertEqual(len(lava[M.RANDOM_STARTS]["per_dataset"]), 9)


# ---- saved files and fingerprints -------------------------------------------------------------------------------
class Files(unittest.TestCase):
    def write_block(self, root, seed=910001):
        rng = np.random.default_rng(3)
        label, data_label, name = "made", "made_data", TASK.name
        (root / "results" / label).mkdir(parents=True)
        (root / "receipts").mkdir()
        (root / "data" / data_label).mkdir(parents=True)
        entries, digests, runs = [], {}, []
        for recipe in C.RECIPES:
            data = {field: np.zeros(60, C.TYPES[field]) for field in C.FIELDS if field != "a"}
            data["s"] = rng.choice(FREE, 60).astype(np.int32)
            data["command"] = data["executed"] = rng.integers(0, MOVES, 60).astype(np.int8)
            data["a"] = E.CORNERS[data["command"].astype(int)].astype(np.float32)
            file = f"{name}_{recipe}_seed{seed}.npz"
            np.savez_compressed(root / "data" / data_label / file, **data)
            digests[file] = C.digest({field: data[field] for field in C.FIELDS})
            entries.append({"file": file, "array_sha256": digests[file], "checks_passed": True})
            for train_seed in (20, 21, 22):
                runs.append({"recipe": recipe, "data_seed": seed, "train_seed": train_seed, "method": "baseline", "alpha": 1.0,
                             "index": len(runs), "final_step": 25000, "reference_ready": True,
                             "reference": {"support_floor": -1.0}, "audit_bank": {"coverage": 0.9}})
        n = len(runs)
        audit = {field: np.zeros((n, M.AUDIT_ROWS), np.float32) for field in M.AUDIT_FIELDS}
        audit["cell"], audit["command"] = audit["cell"].astype(np.int32) + int(FREE[0]), audit["command"].astype(np.int32)
        audit["margin"][:] = 0.1
        audit["residual"][:, :128] = 0.5                                          # 10% of rows outside: coverage 0.9
        audit["target"] = audit["residual"] + audit["q1"]
        folder = root / "results" / label
        (folder / "START.json").write_text(json.dumps({"alpha": 1.0, "method": "baseline only", "data_label": data_label}))
        (root / "receipts" / f"{data_label}_manifest.json").write_text(json.dumps(
            {"seeds": [seed], "seeds_passing_every_check": [seed], "created_unix": 2e9, "datasets": entries}))
        (folder / f"datasets_{name}.json").write_text(json.dumps(digests))
        (folder / f"baselines_{name}.json").write_text(json.dumps({"runs": runs}))
        arrays = {key: rng.uniform(-0.5, 0.5, (n, CELLS, MOVES)) for key in ("commands", "q1", "q2", "width", "support")}
        np.savez_compressed(folder / f"baselines_{name}.npz", **arrays)
        np.savez_compressed(folder / f"audit_rows_{name}.npz", **audit)
        return label, folder, audit

    def test_a_block_is_loaded_and_checked_against_its_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label, folder, audit = self.write_block(root)
            block = M.load_block(label, root)
            self.assertEqual((len(block["runs"]), block["candidates"]), (9, [910001]))
            self.assertEqual(int(block["counts"][(M.HALF_RANDOM, 910001)].sum()), 60)
            mixed = [run for run in block["runs"] if run["recipe"] == M.HALF_RANDOM]
            self.assertTrue(all(run["audit"] is not None and run["margin"].shape == (CELLS, MOVES) for run in mixed))
            self.assertTrue(all(run["audit"] is None for run in block["runs"] if run["recipe"] != M.HALF_RANDOM))

            audit["residual"][:, :200] = 0.5                                      # no longer the reported coverage
            audit["target"] = audit["residual"] + audit["q1"]
            np.savez_compressed(folder / f"audit_rows_{TASK.name}.npz", **audit)
            with self.assertRaises(M.Defect):
                M.load_block(label, root)

    def test_a_changed_dataset_or_another_learner_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label, folder, _ = self.write_block(root)
            start = json.loads((folder / "START.json").read_text())
            (folder / "START.json").write_text(json.dumps(dict(start, alpha=2.5)))
            with self.assertRaises(M.Defect):
                M.load_block(label, root)
            (folder / "START.json").write_text(json.dumps(start))
            file = root / "data" / "made_data" / f"{TASK.name}_mixed_seed910001.npz"
            with np.load(file) as saved:
                data = {field: saved[field] for field in C.FIELDS}
            data["command"] = (data["command"] + 1) % MOVES
            np.savez_compressed(file, **data)
            with self.assertRaises(M.Defect):
                M.load_block(label, root)

    def test_a_fingerprint_does_not_depend_on_line_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            unix, windows = Path(tmp) / "a.md", Path(tmp) / "b.md"
            unix.write_bytes(b"one\ntwo\n")
            windows.write_bytes(b"one\r\ntwo\r\n")
            self.assertEqual(M.fingerprint(unix), M.fingerprint(windows))
            self.assertNotEqual(M.fingerprint(unix), M.fingerprint_bytes(b"one\ntwo"))

    def test_s1(self):
        saved = (M.FROZEN, M.committed_fingerprint, M.git)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "text").mkdir()
            (root / "results").mkdir()
            frozen = {"criteria": root / "text" / "criteria.md", "verdict_script": root / "text" / "verdict.py"}
            for path in frozen.values():
                path.write_text(path.name)
                (root / "results" / path.name).write_text(path.name)
            prints = {name: M.fingerprint(path) for name, path in frozen.items()}
            record = {"criteria_sha256": prints["criteria"], "verdict_script_sha256": prints["verdict_script"], "commit": "abc12345"}
            try:
                M.FROZEN = frozen
                M.committed_fingerprint = lambda path, commit="HEAD": M.fingerprint(path)
                M.git = lambda *args: b"1000\n"
                good = M.check_s1({M.RECORD_KEY: record, "created_unix": 2000.0}, {M.RECORD_KEY: record}, root / "results")
                self.assertTrue(good["passed"], good["reasons"])
                late = M.check_s1({M.RECORD_KEY: record, "created_unix": 900.0}, {M.RECORD_KEY: record}, root / "results")
                self.assertFalse(late["passed"])                                   # the commit is newer than the receipt
                self.assertFalse(M.check_s1({"created_unix": 2000.0}, {M.RECORD_KEY: record}, root / "results")["passed"])
                self.assertFalse(M.check_s1({M.RECORD_KEY: record, "created_unix": 2000.0}, {}, root / "results")["passed"])
                other = dict(record, criteria_sha256="0" * 64)
                self.assertFalse(M.check_s1({M.RECORD_KEY: other, "created_unix": 2000.0}, {M.RECORD_KEY: record},
                                            root / "results")["passed"])
                (root / "results" / "verdict.py").write_text("changed")
                self.assertFalse(M.check_s1({M.RECORD_KEY: record, "created_unix": 2000.0}, {M.RECORD_KEY: record},
                                            root / "results")["passed"])
                M.git = lambda *args: None                                         # git cannot answer
                (root / "results" / "verdict.py").write_text("verdict.py")
                self.assertFalse(M.check_s1({M.RECORD_KEY: record, "created_unix": 2000.0}, {M.RECORD_KEY: record},
                                            root / "results")["passed"])
            finally:
                M.FROZEN, M.committed_fingerprint, M.git = saved

    def test_the_freeze_covers_the_builder_and_the_run_scripts(self):
        saved = (M.git, M.committed_fingerprint)
        try:
            M.committed_fingerprint = lambda path, commit="HEAD": M.fingerprint(path)
            M.git = lambda *args: b" M experiments/gridworld_critical_v1/run_baselines.py\n?? experiments/gridworld_critical_v1/new.py\n"
            self.assertEqual(M.scripts_not_committed(), ["experiments/gridworld_critical_v1/run_baselines.py",
                                                         "experiments/gridworld_critical_v1/new.py"])
            state = M.freeze_state()
            self.assertTrue(state["criteria"]["committed"] and state["verdict_script"]["committed"])
            self.assertFalse(state["ok"])                                          # both texts committed, a script is not
            M.git = lambda *args: b""
            self.assertTrue(M.freeze_state()["ok"])
            M.git = lambda *args: None                                             # git cannot answer
            self.assertIsNone(M.scripts_not_committed())
            self.assertFalse(M.freeze_state()["ok"])
            with self.assertRaises(M.Defect):
                M.record_for_receipt()
        finally:
            M.git, M.committed_fingerprint = saved

    def test_an_audit_file_without_the_saved_error_and_a_judged_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label, folder, audit = self.write_block(root)
            self.assertFalse(M.load_block(label, root)["judged_before"])
            other = M.verdict_receipt(root, "same_seeds_under_another_label")
            other.write_text(json.dumps({"candidates": [910001, 910002]}))          # the same seed, judged before
            self.assertTrue(M.load_block(label, root)["judged_before"])
            other.write_text(json.dumps({"candidates": [910002]}))
            self.assertFalse(M.load_block(label, root)["judged_before"])
            other.write_text("not json")
            with self.assertRaises(M.Defect):
                M.load_block(label, root)
            other.unlink()
            M.verdict_receipt(root, "made_data").write_text("{}")
            self.assertTrue(M.load_block(label, root)["judged_before"])
            del audit["residual"]
            np.savez_compressed(folder / f"audit_rows_{TASK.name}.npz", **audit)
            with self.assertRaises(M.Defect):
                M.load_block(label, root)

    def test_a_verdict_is_written_once_and_a_seen_reading_never_replaces_it(self):
        prints = {"criteria": "0" * 64, "verdict_script": "1" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "results" / "made"
            folder.mkdir(parents=True)
            (root / "receipts").mkdir()
            verdict = dict(M.evaluate(made_up_block(), S1_OK), fingerprints=prints)
            out = M.write_result(verdict, folder, root, "made", "made_data")
            self.assertEqual(out.name, "margin_verdict.json")
            receipt = json.loads(M.verdict_receipt(root, "made_data").read_text())
            self.assertEqual((receipt["verdict"], receipt["candidates"]), ("validated", list(FRESH)))
            self.assertEqual(M.judged_seeds(root), set(FRESH))
            first = out.read_text()
            with self.assertRaises(M.Defect):                                      # a block is judged once
                M.write_result(verdict, folder, root, "made", "made_data")
            block = made_up_block()
            block["judged_before"] = True                                          # what load_block now reports
            seen = dict(M.evaluate(block, S1_OK), fingerprints=prints)
            again = M.write_result(seen, folder, root, "made", "made_data")
            self.assertEqual(again.name, "margin_seen_reading.json")
            self.assertEqual(out.read_text(), first)                               # the verdict is untouched
            self.assertIn("## Verdict: validated", out.with_suffix(".md").read_text(encoding="utf-8"))
            self.assertIn("Not a verdict", again.with_suffix(".md").read_text(encoding="utf-8"))

    def test_the_freeze_reads_this_folder_and_not_its_subfolders(self):
        self.assertEqual(M.RULE_2_PATHS[0], ":(glob)experiments/gridworld_critical_v1/*.py")

    def test_spent_seeds_are_the_published_gridworld_s(self):
        self.assertEqual(len(M.SPENT_SEEDS), 1 + 3 + 5 + 5 + 15)
        self.assertTrue({7, 6101, 6305, 6405, 6515} <= M.SPENT_SEEDS)
        self.assertFalse(M.SPENT_SEEDS & set(range(6601, 6616)))


if __name__ == "__main__":
    unittest.main()
