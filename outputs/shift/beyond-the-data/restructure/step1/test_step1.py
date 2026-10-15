"""Unit tests of step 1's pure definitions on synthetic data (no gate data, no JAX, no training).

Run: PYTHONDONTWRITEBYTECODE=1 python test_step1.py   (plain unittest; pytest is not installed here)
Covers: the gap rule (definition, ties, refusal, grouping of held-out rows), the residual's sign, the pairwise
posterior-gap quantile, the certificate statistic and margin, the net across worlds, the outcome shuffle, the
go/no-go, the output guard and the rng tag rule.
"""
import sys

sys.dont_write_bytecode = True

import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1lib as S  # noqa: E402

CFG = dict(alpha=0.1, credibility=0.95, draws=1000, margin=0.10, statistic_quantile=0.05)


def synthetic_throttle(n_bulk=20000, n_tail=40, seed=1, top=0.78, tail_value=1.0):
    r = np.random.default_rng(seed)
    bulk = r.uniform(0.0, top, n_bulk)
    return np.concatenate([bulk, np.full(n_tail, tail_value)]), bulk


class GapRule(unittest.TestCase):
    def test_finds_the_empty_gap_and_the_tail(self):
        t, bulk = synthetic_throttle()
        g = S.gap_rule(t)
        self.assertTrue(g["defined"])
        self.assertEqual(g["t_lo"], float(bulk.max()))
        self.assertEqual(g["t_up"], 1.0)
        self.assertEqual(g["n_tail"], 40)
        np.testing.assert_array_equal(g["tail"], t >= 1.0)

    def test_throttle_index_is_max_abs_over_thrusters(self):
        a = np.array([[0.1, -0.9, 0.2, 0.0], [-1.0, 0.5, 0.0, 0.3], [0.0, 0.0, 0.0, 0.79]], np.float32)
        np.testing.assert_allclose(S.throttle(a), [np.float32(0.9), 1.0, np.float32(0.79)])
        self.assertEqual(S.throttle(a).dtype, np.float64)

    def test_ties_first_largest_gap_and_tied_values(self):
        # dyadic values (exact in binary): bulk up to 0.25, then 0.3125 (gap 0.0625), 0.4375 (gap 0.125),
        # 0.5625 (gap 0.125, a tie): the first of the two equal largest gaps wins
        t = np.concatenate([np.linspace(0.0, 0.25, 3000), [0.3125, 0.4375], [0.5625] * 16])
        g = S.gap_rule(t)
        self.assertEqual(g["t_lo"], 0.3125)
        self.assertEqual(g["t_up"], 0.4375)
        self.assertEqual(g["n_tail"], 17)
        # rows tied at t_up belong to the full-throttle group, rows tied at t_lo to the rest
        t2 = np.concatenate([np.linspace(0.0, 0.75, 3000), [0.78] * 5, [1.0] * 12])
        g2 = S.gap_rule(t2)
        self.assertEqual(g2["t_lo"], 0.78)
        self.assertEqual(g2["n_tail"], 12)
        self.assertFalse(g2["tail"][t2 == 0.78].any())

    def test_not_defined_when_gap_narrow_or_tail_small(self):
        t, _ = synthetic_throttle(n_tail=8)
        self.assertFalse(S.gap_rule(t)["defined"])
        t, _ = synthetic_throttle(top=0.97, tail_value=1.0)
        g = S.gap_rule(t)
        self.assertFalse(g["defined"])
        self.assertEqual(g["n_tail"], 0)
        self.assertFalse(g["tail"].any())

    def test_held_out_rows_are_grouped_by_t_lo(self):
        t, bulk = synthetic_throttle()
        g = S.gap_rule(t)
        held = np.array([0.5, g["t_lo"], g["t_lo"] + 1e-6, 0.9, 1.0])
        np.testing.assert_array_equal(S.full_throttle(held, g["t_lo"]), [False, False, True, True, True])

    def test_reads_commands_only(self):
        # the rule's input is the throttle index alone: any reordering of rows gives the same t_lo / t_up / count
        t, _ = synthetic_throttle()
        g1 = S.gap_rule(t)
        perm = np.random.default_rng(5).permutation(len(t))
        g2 = S.gap_rule(t[perm])
        self.assertEqual((g1["t_lo"], g1["t_up"], g1["n_tail"]), (g2["t_lo"], g2["t_up"], g2["n_tail"]))


class Residual(unittest.TestCase):
    def test_sign_positive_when_critic_overpredicts(self):
        s = S.overestimate(q_sa=1.0, reward=0.2, done=0.0, q_next=0.5, gamma=0.98)
        self.assertAlmostEqual(float(s), 1.0 - (0.2 + 0.98 * 0.5))
        self.assertGreater(float(s), 0)
        self.assertLess(float(S.overestimate(0.1, 0.2, 0.0, 0.5, 0.98)), 0)

    def test_done_stops_bootstrap_and_equals_minus_td_residual(self):
        self.assertAlmostEqual(float(S.overestimate(1.0, 0.2, 1.0, 100.0, 0.98)), 0.8)
        r = np.random.default_rng(2)
        q, rew, qn = r.normal(size=50), r.normal(size=50), r.normal(size=50)
        delta = rew + 0.98 * qn - q
        np.testing.assert_allclose(S.overestimate(q, rew, np.zeros(50), qn, 0.98), -delta)


class Certificate(unittest.TestCase):
    def test_pairwise_quantile_matches_brute_force(self):
        r = np.random.default_rng(3)
        x, y = r.normal(size=60), r.normal(size=45)
        d = np.sort((x[:, None] - y[None, :]).ravel())
        for q in (0.05, 0.5, 0.95):
            k = int(np.ceil(q * len(d))) - 1        # inverted_cdf order statistic
            self.assertEqual(S.pairwise_quantile(x, y, q), d[k])

    def test_pairwise_quantile_infinities(self):
        x = np.array([np.inf, 1.0])
        y = np.array([0.0, np.inf])
        # differences: inf, -inf (inf - inf), 1.0, -inf  -> sorted -inf, -inf, 1, inf
        self.assertEqual(S.pairwise_quantile(x, y, 0.5), -np.inf)
        self.assertEqual(S.pairwise_quantile(x, y, 0.75), 1.0)
        self.assertEqual(S.pairwise_quantile(x, y, 1.0), np.inf)

    def test_statistic_is_the_5pct_quantile_of_the_two_posteriors_gap(self):
        r = np.random.default_rng(4)
        tail, rest = r.normal(0.5, 0.2, 40), r.normal(0.0, 0.1, 800)
        tags = [20261015, 9700, 0, 1]
        c = S.certificate(tail, rest, tags, CFG)
        ct = S.wbcp.calibrate(tail, np.random.default_rng(tags + [1]), alpha=0.1, beta=0.95, draws=1000)
        cr = S.wbcp.calibrate(rest, np.random.default_rng(tags + [2]), alpha=0.1, beta=0.95, draws=1000)
        d = np.sort((ct.crossings[:, None] - cr.crossings[None, :]).ravel())
        self.assertEqual(c["q05"], d[int(np.ceil(0.05 * len(d))) - 1])
        self.assertEqual(c["height"], ct.threshold - cr.threshold)
        self.assertTrue(c["on"])
        self.assertEqual(S.certificate(tail, rest, tags, CFG), c)            # deterministic given the tags

    def test_off_without_a_gap_and_margin_is_strict(self):
        r = np.random.default_rng(6)
        rest = r.normal(0.0, 0.1, 800)
        same = r.normal(0.0, 0.1, 40)
        self.assertFalse(S.certificate(same, rest, [1, 2, 3], CFG)["on"])
        # a small credible gap (+0.08 on narrow noise) is credible at margin 0 but not at 0.10
        small = rest[:400] + 0.08
        c0 = S.certificate(small, rest[400:], [1, 2, 4], dict(CFG, margin=0.0))
        c1 = S.certificate(small, rest[400:], [1, 2, 4], CFG)
        self.assertTrue(c0["on"])
        self.assertFalse(c1["on"])
        self.assertEqual(c0["q05"], c1["q05"])

    def test_empty_group_is_off(self):
        self.assertFalse(S.certificate(np.array([]), np.ones(10), [1, 2, 5], CFG)["on"])

    def test_rank_test_direction(self):
        r = np.random.default_rng(7)
        rest = r.normal(0, 1, 500)
        self.assertLess(S.rank_test(rest[:40] + 2.0, rest)["p"], 1e-6)
        self.assertGreater(S.rank_test(rest[:40] - 2.0, rest)["p"], 0.99)


def toy_points(seeds):
    pts = {"stall": {s: dict(never=-10.0, on=90.0 + s) for s in seeds},
           "no_stall": {s: dict(never=95.0, on=30.0) for s in seeds}}
    return pts


class Net(unittest.TestCase):
    seeds = [0, 1, 2, 3, 4]

    def test_net_definition(self):
        pts = toy_points(self.seeds)
        on = {"stall": {s: True for s in self.seeds}, "no_stall": {s: False for s in self.seeds}}
        n = S.deployment_net(pts, on, self.seeds)
        self.assertAlmostEqual(n["gain_stall"], 102.0)        # mean(90 + s) + 10
        self.assertAlmostEqual(n["loss_no_stall"], 0.0)
        self.assertAlmostEqual(n["net"], 102.0)
        on["no_stall"][3] = True                                 # one false certificate: loss 65 / 5 seeds
        n2 = S.deployment_net(pts, on, self.seeds)
        self.assertAlmostEqual(n2["loss_no_stall"], 13.0)
        self.assertAlmostEqual(n2["net"], 89.0)
        on["stall"][0] = False                                   # a missed stalling seed contributes 0 gain
        self.assertAlmostEqual(S.deployment_net(pts, on, self.seeds)["gain_stall"], (102.0 * 5 - 100.0) / 5)

    def test_go_no_go(self):
        rule = dict(net_min=90.0, shuffle_max_fraction=0.5)
        on = {"stall": {s: True for s in self.seeds}, "no_stall": {s: False for s in self.seeds}}
        self.assertEqual(S.go_no_go(on, 102.0, 10.0, self.seeds, rule)["verdict"], "GO")
        self.assertEqual(S.go_no_go(on, 89.9, 10.0, self.seeds, rule)["verdict"], "NO-GO")
        self.assertEqual(S.go_no_go(on, 102.0, 51.5, self.seeds, rule)["verdict"], "NO-GO")
        self.assertEqual(S.go_no_go(on, 102.0, 51.0, self.seeds, rule)["verdict"], "GO")
        on["no_stall"][2] = True
        v = S.go_no_go(on, 102.0, 10.0, self.seeds, rule)
        self.assertFalse(v["C1"])
        self.assertEqual(v["verdict"], "NO-GO")


class Shuffle(unittest.TestCase):
    def test_permutations_keep_scores_and_are_fixed(self):
        r = np.random.default_rng(8)
        tail, rest = r.normal(1.0, 0.1, 30), r.normal(0.0, 0.1, 300)
        a = S.shuffled_certificates(tail, rest, [1, 9700, 0, 1], [1, 9800, 0, 1], 20, CFG)
        b = S.shuffled_certificates(tail, rest, [1, 9700, 0, 1], [1, 9800, 0, 1], 20, CFG)
        self.assertEqual(a, b)
        pooled = np.concatenate([tail, rest])
        p = pooled[S.rng([1, 9800, 0, 1, 1]).permutation(len(pooled))]
        self.assertEqual(sorted(p), sorted(pooled))
        # the real certificate is on; shuffled outcomes almost never are
        self.assertTrue(S.certificate(tail, rest, [1, 9700, 0, 1], CFG)["on"])
        self.assertLessEqual(np.mean(a["on"]), 0.1)

    def test_shuffle_summary_uses_the_same_permutation_across_seed_worlds(self):
        seeds = [0, 1]
        pts = toy_points(seeds)
        sh = {"stall": {0: [True, False], 1: [True, False]}, "no_stall": {0: [False, True], 1: [False, False]}}
        s = S.shuffle_summary(pts, sh, seeds, net_real=100.0)
        # j = 0: both stall on, no-stall off -> gain mean(100, 101) = 100.5, net 100.5; j = 1: no stall on,
        # no-stall seed 0 on -> loss 65 / 2 = 32.5, net -32.5
        self.assertAlmostEqual(s["mean"], (100.5 - 32.5) / 2)
        self.assertAlmostEqual(s["share_at_least_real"], 0.5)


class ReviewAdditions(unittest.TestCase):
    """Added in review: record validation, the shuffled height, the other reading of condition 3."""

    spec = {"training": {"updates": 100000}}

    def good_record(self):
        return dict(mode="official", prespec_sha256="P", code_manifest_sha256="M", world="stall", seed=3,
                    result=dict(updates=100000), variant_of_run="gap minus survivors",
                    checks=dict(rows=True, no_survivor_in_training=True), cpus=[2],
                    compare_with_gate=dict(args_equal_gate=True, initial_params_equal_gate=True,
                                           file_sha256_equal_gate=False))

    def test_record_problems_accepts_a_good_record(self):
        self.assertEqual(S.record_problems(self.good_record(), self.spec, "P", "M", "stall", 3), [])

    def test_record_problems_rejects_each_deviation(self):
        bad = [("mode", "scratch"), ("prespec_sha256", "X"), ("code_manifest_sha256", "X"), ("world", "no_stall"),
               ("seed", 4), ("result", dict(updates=200)), ("variant_of_run", "gap (no rows removed: reproduction "
                                                                              "check)"),
               ("checks", dict(rows=True, no_survivor_in_training=False)), ("checks", {}), ("cpus", [0, 1]),
               ("compare_with_gate", dict(args_equal_gate=False, initial_params_equal_gate=True)),
               ("compare_with_gate", dict(args_equal_gate=True, initial_params_equal_gate=False)),
               ("compare_with_gate", {})]
        for k, v in bad:
            r = self.good_record()
            r[k] = v
            self.assertTrue(S.record_problems(r, self.spec, "P", "M", "stall", 3), msg=f"{k}={v!r} not caught")

    def test_shuffle_returns_heights_matching_the_certificate(self):
        r = np.random.default_rng(9)
        tail, rest = r.normal(1.0, 0.1, 30), r.normal(0.0, 0.1, 300)
        a = S.shuffled_certificates(tail, rest, [1, 9700, 0, 1], [1, 9800, 0, 1], 3, CFG)
        pooled = np.concatenate([tail, rest])
        p = pooled[S.rng([1, 9800, 0, 1, 2]).permutation(len(pooled))]
        self.assertEqual(a["height"][1], S.certificate(p[:30], p[30:], [1, 9700, 0, 1], CFG)["height"])
        self.assertEqual(len(a["height"]), 3)

    def test_other_reading_of_condition_3_is_information_only(self):
        rule = dict(net_min=90.0, shuffle_max_fraction=0.5)
        seeds = [0, 1, 2, 3, 4]
        on = {"stall": {s: True for s in seeds}, "no_stall": {s: False for s in seeds}}
        v = S.go_no_go(on, 106.0, 50.0, seeds, rule)        # 50 <= 53 (half of NET) but > 45 (half of 90)
        self.assertEqual(v["verdict"], "GO")
        self.assertFalse(v["info_C3_also_at_most_half_of_90"])
        self.assertTrue(S.go_no_go(on, 106.0, 10.0, seeds, rule)["info_C3_also_at_most_half_of_90"])


class Guards(unittest.TestCase):
    def test_guard_out(self):
        with self.assertRaises(SystemExit):
            S.guard_out(S.ROOT / "runs/shift_explore/stage2/x", official=True)
        with self.assertRaises(SystemExit):
            S.guard_out(S.ROOT / "outputs/shift/beyond-the-data/stage2_v7/x", official=False)
        with self.assertRaises(SystemExit):
            S.guard_out(S.ROOT / "runs/shift_explore/restructure/step1/x", official=False)   # scratch: not the repo
        with self.assertRaises(SystemExit):
            S.guard_out("/tmp/elsewhere/x", official=True)
        self.assertEqual(S.guard_out(S.OFFICIAL_CRITICS / "stall/seed0", official=True),
                         (S.OFFICIAL_CRITICS / "stall/seed0").resolve())
        self.assertTrue(str(S.guard_out("/tmp/step1_scratch/x", official=False)).startswith("/tmp/"))

    def test_rng_refuses_trailing_zero(self):
        with self.assertRaises(ValueError):
            S.rng([1, 2, 0])
        self.assertEqual(S.rng([1, 2, 3]).integers(1000), np.random.default_rng([1, 2, 3]).integers(1000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
