"""Tests of the selection rule and the claims of PROTOCOL.md, Amendment 5. NumPy only.

    python -I -B -m unittest discover -s experiments/gridworld_critical_v1 -p "test_confirm.py" -v
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import confirm_version_a_published as K  # noqa: E402
import judge_version_a as J  # noqa: E402

SEEDS, TRAIN = tuple(range(1, 10)), (20, 21, 22)
METHODS = (J.BASELINE, J.VERSION_A, J.CONTROL, J.DECISIVE)


def records(scores, lava=None):
    """scores[recipe][method] is one score per dataset; every training seed of a dataset gets it."""
    out = {}
    for recipe, by_method in scores.items():
        for method, per_dataset in by_method.items():
            for d, value in zip(SEEDS, per_dataset):
                for s in TRAIN:
                    out[(recipe, d, s, method)] = {"score": value, "lava": (lava or {}).get((recipe, method), 0.05)}
    return out


def nine(value):
    return (value,) * 9


def case(mixed=None, optimal=None, other=None, lava=None):
    good = {J.BASELINE: nine(0.80), J.VERSION_A: nine(0.95), J.CONTROL: nine(0.85), J.DECISIVE: nine(0.85)}
    flat = {m: nine(0.98) for m in METHODS}
    table = {"mixed": mixed or good, "optimal": optimal or dict(good, **{J.BASELINE: nine(0.92)}), "other_starts": other or flat}
    return K.judge(records(table, lava), SEEDS, TRAIN)


class ExactTest(unittest.TestCase):
    def test_all_positive_differences_give_the_smallest_p(self):
        self.assertAlmostEqual(K.sign_flip_p([0.1] * 9), 1 / 512)

    def test_symmetric_differences_give_about_one_half(self):
        self.assertGreater(K.sign_flip_p([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.0]), 0.4)

    def test_one_large_loss_can_outweigh_many_small_wins(self):
        self.assertGreater(K.sign_flip_p([0.01] * 8 + [-0.5]), 0.05)

    def test_all_zero_differences_never_pass(self):
        self.assertEqual(K.sign_flip_p([0.0] * 9), 1.0)


class Selection(unittest.TestCase):
    def test_the_first_nine_valid_seeds_in_order(self):
        choice = K.select([6512, 6501, 6503, 6504, 6505, 6506, 6507, 6508, 6509, 6510, 6511])
        self.assertTrue(choice["enough"])
        self.assertEqual(choice["selected"], [6501, 6503, 6504, 6505, 6506, 6507, 6508, 6509, 6510])

    def test_fewer_than_nine_selects_nothing(self):
        choice = K.select(list(range(6501, 6509)))
        self.assertFalse(choice["enough"])
        self.assertEqual(choice["selected"], [])


class Claims(unittest.TestCase):
    def test_clear_gains_confirm(self):
        claims = case()
        for name in ("C1_helps_on_the_half_random_data", "C2_helps_on_optimal_data", "C3_no_harm_where_there_is_no_room",
                     "D1_more_than_decisiveness"):
            self.assertTrue(claims[name]["passed"], name)
        self.assertTrue(claims["version_a_is_confirmed_on_the_published_map"])

    def test_a_comparison_needs_six_datasets_and_the_exact_test(self):
        five = (0.95,) * 5 + (0.70,) * 4                         # higher in 5 of 9
        claims = case(mixed={J.BASELINE: nine(0.80), J.VERSION_A: five, J.CONTROL: nine(0.5), J.DECISIVE: nine(0.5)})
        self.assertFalse(claims["C1_helps_on_the_half_random_data"]["holds"])
        one_collapse = (0.83,) * 8 + (0.05,)                      # higher in 8 of 9, but the mean is lower
        claims = case(mixed={J.BASELINE: nine(0.80), J.VERSION_A: one_collapse, J.CONTROL: nine(0.5), J.DECISIVE: nine(0.5)})
        self.assertEqual(claims["C1_helps_on_the_half_random_data"]["datasets_higher"], 8)
        self.assertFalse(claims["C1_helps_on_the_half_random_data"]["holds"])
        self.assertFalse(claims["version_a_is_confirmed_on_the_published_map"])

    def test_c1_also_needs_the_gain_of_five_points(self):
        claims = case(mixed={J.BASELINE: nine(0.80), J.VERSION_A: nine(0.84), J.CONTROL: nine(0.5), J.DECISIVE: nine(0.5)})
        self.assertTrue(claims["C1_helps_on_the_half_random_data"]["holds"])
        self.assertFalse(claims["C1_helps_on_the_half_random_data"]["passed"])

    def test_a_hair_over_a_control_does_not_hold(self):
        """The first test's K2 passed on differences of 0.003 and 0.006. Here it must not."""
        control = (0.980, 0.975, 0.665, 0.980, 0.975, 0.665, 0.980, 0.975, 0.665)
        version_a = (0.983, 0.972, 0.671, 0.983, 0.972, 0.671, 0.983, 0.972, 0.671)
        claims = case(mixed={J.BASELINE: nine(0.80), J.VERSION_A: version_a, J.CONTROL: control, J.DECISIVE: nine(0.5)})
        self.assertFalse(claims["D2_the_true_cell_matters"]["mixed"]["holds"])
        self.assertFalse(claims["D2_the_true_cell_matters"]["passed"])

    def test_d_claims_do_not_decide_the_confirmation(self):
        tie = {J.BASELINE: nine(0.80), J.VERSION_A: nine(0.95), J.CONTROL: nine(0.95), J.DECISIVE: nine(0.96)}
        claims = case(mixed=tie)
        self.assertFalse(claims["D1_more_than_decisiveness"]["passed"])
        self.assertFalse(claims["D2_the_true_cell_matters"]["passed"])
        self.assertTrue(claims["version_a_is_confirmed_on_the_published_map"])

    def test_harm_on_other_starts_blocks_the_confirmation(self):
        worse = {J.BASELINE: nine(0.98), J.VERSION_A: nine(0.95), J.CONTROL: nine(0.98), J.DECISIVE: nine(0.98)}
        self.assertFalse(case(other=worse)["version_a_is_confirmed_on_the_published_map"])
        lava = {("other_starts", J.VERSION_A): 0.07, ("other_starts", J.BASELINE): 0.05}
        self.assertFalse(case(lava=lava)["C3_no_harm_where_there_is_no_room"]["passed"])


if __name__ == "__main__":
    unittest.main()
