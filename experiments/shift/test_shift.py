"""CPU tests of the removal rule, the controls, the pools and the shift card (no learning; every number is a test
fixture, not a finding). One medium log (100 training episodes) keeps it fast.

Checks: find_gap reruns identically; mass in [0.14, 0.16] and rho >= 0.05; the ball is inclusive (the row at exactly
rho is removed); the gapped training rows are the training rows minus the ball rows with next observations untouched;
the calibration bank keeps no gap row and the leak counter finds every gap row of the unmasked split; the evaluation
split is never touched; random removal takes the same count, is seeded and is independent of the commands (KS test
on distance to the centre); at least 99% of the full-throttle commands are removed; m(s) by Monte Carlo agrees with
exact values (0.5 inside the region, 0 outside) within 3 SE, zero shift gives 0 and random removal its rate; the
card's measures give known answers on synthetic trajectories and the loss from unrecorded throttle is exactly 0 for
the stall-aware controller; the pools are seeded, disjoint from what they must avoid and hashed.
JAX_PLATFORMS=cpu python -m unittest experiments.shift.test_shift
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.truth import gap as GAP  # noqa: E402

STATE = {}


def setUpModule():
    log = DS.generate(904, "stall", sizes=(100, 10, 20))
    gap, record = SH.define_gap(log)
    STATE.update(log=log, gap=gap, record=record, recorder=RC.Recorder())


class GapTest(unittest.TestCase):
    def test_gap_rule(self):
        log, gap, record = STATE["log"], STATE["gap"], STATE["record"]
        self.assertIsNotNone(gap, record.get("reason"))
        again, _ = SH.define_gap(log)
        self.assertEqual(again, gap)
        self.assertTrue(0.14 <= gap.mass <= 0.16)
        self.assertGreaterEqual(gap.rho, 0.05)
        self.assertIn(gap.rule, ("base", "tie_block_fallback"))
        self.assertLess(gap.centre[0], -0.6)
        self.assertGreater(gap.competing_ratio, 1.0)
        settings = SH.gap_settings()
        self.assertIsNone(settings["search"]["score_rows"])
        self.assertEqual(settings["q"], 0.15)
        self.assertEqual(settings["band"], [0.14, 0.16])
        train = log.split_rows("train")
        d = GAP.distance(np.asarray(log["actions"][train], np.float64), gap.centre)
        at_rho = np.flatnonzero(d == gap.rho)
        self.assertGreater(len(at_rho), 0)
        self.assertTrue(np.all(gap.contains(log["actions"][train[at_rho]])))
        self.assertAlmostEqual(float(gap.contains(log["actions"][train]).mean()), gap.mass, places=12)

    def test_gap_removes_the_full_throttle_commands(self):
        log, gap = STATE["log"], STATE["gap"]
        train = log.split_rows("train")
        press = log.diagnostic("infos/full_throttle_command")[train]
        ing = gap.contains(log["actions"][train])
        self.assertGreaterEqual(ing[press].mean(), 0.99)
        kept_past = TC.past_knee(np.asarray(log["actions"][train], np.float64))[~ing]
        self.assertLessEqual(kept_past.mean(), 0.002)

    def test_gapped_rows_and_bank(self):
        log, gap = STATE["log"], STATE["gap"]
        rem = SH.make_removal(log, "gap", gap)
        train = log.split_rows("train")
        ing = gap.contains(log["actions"][train])
        self.assertTrue(np.array_equal(rem.kept_train_rows(log), train[~ing]))
        self.assertTrue(np.array_equal(rem.removed_train_rows(log), train[ing]))
        kept = rem.kept_train_rows(log)
        d4 = DS.d4rl_dict(log, kept)
        for k in DS.D4RL_KEYS:
            self.assertTrue(np.array_equal(d4[k], log[k][kept]), k)
        self.assertEqual(SH.leak_count(kept, log, gap), 0)
        self.assertEqual(SH.leak_count(rem.bank_rows(log), log, gap), 0)
        self.assertEqual(SH.leak_count(train, log, gap), int(ing.sum()))
        self.assertEqual(SH.leak_count(log.bank_rows, log, gap), int(rem.bank_removed.sum()))
        self.assertFalse(rem.removed[log.split_rows("evaluation")].any())
        self.assertFalse(rem.removed[log.split_rows("calibration")].any())
        none = SH.make_removal(log, "none")
        self.assertEqual(none.count, 0)
        self.assertTrue(np.array_equal(none.kept_train_rows(log), train))
        summary = rem.summary(log)
        self.assertEqual(summary["removed_training_rows"], int(ing.sum()))
        with self.assertRaises(ValueError):
            SH.make_removal(log, "gap")

    def test_random_removal(self):
        log, gap = STATE["log"], STATE["gap"]
        rem = SH.make_removal(log, "gap", gap)
        rnd = SH.make_removal(log, "random", gap)
        self.assertEqual(rnd.count, rem.count)
        self.assertEqual(int(rnd.bank_removed.sum()), int(rem.bank_removed.sum()))
        self.assertTrue(np.array_equal(SH.make_removal(log, "random", gap).removed, rnd.removed))
        train = log.split_rows("train")
        d = GAP.distance(np.asarray(log["actions"][train], np.float64), gap.centre)
        r = rnd.removed[train]
        self.assertGreater(stats.ks_2samp(d[r], d[~r]).pvalue, 0.001)
        past = TC.past_knee(np.asarray(log["actions"][train], np.float64))
        self.assertGreater(past[~r].sum() / past.sum(), 0.8)


class RemovedShareTest(unittest.TestCase):
    def test_monte_carlo_matches_exact_values(self):
        rec = STATE["recorder"]
        # a ball of radius 0.2 around the full-throttle pattern: every full-throttle command is inside, no
        # normal-driving command can be (it never reaches 80%, the pattern's thruster 1 is at -1), so m = 0.5 inside
        # the region and 0 outside, exactly.
        ball = SH.Gap(centre=tuple(rec.pattern), rho=0.2, mass=0.15, rule="base", q=0.15, competing_ratio=2.0,
                      record_sha256="synthetic")
        removal = SH.Removal("gap", np.zeros(1, bool), np.zeros(0, bool), ball)
        x = np.zeros((6, 13))
        x[:3, TC.CURRENT], x[:3, TC.WAVE1] = 0.6, 0.1
        x[3:, TC.CURRENT], x[3:, TC.WAVE1] = 0.3, 0.1
        x[1, :10] = 0.4
        x[4, :10] = -0.4
        m, se = SH.removed_share(x, removal, rec)
        self.assertTrue(np.all(rec.in_region(x[:3])) and not np.any(rec.in_region(x[3:])))
        se_half = np.sqrt(0.25 / SH.MS_DRAWS)
        self.assertTrue(np.all(np.abs(m[:3] - 0.5) <= 3 * se_half), m)
        self.assertTrue(np.all(m[3:] == 0.0))
        self.assertLessEqual(se.max(), 0.008)
        z, _ = SH.removed_share(x, "none", rec)
        self.assertTrue(np.all(z == 0))
        r, _ = SH.removed_share(x, "random", rec, rate=0.15)
        self.assertTrue(np.all(r == 0.15))
        log = STATE["log"]
        rnd = SH.make_removal(log, "random", STATE["gap"])
        rr, _ = SH.removed_share(x, rnd, rec, rate=rnd.rate(log))
        self.assertTrue(np.allclose(rr, rnd.rate(log)))


class CardTest(unittest.TestCase):
    def setUp(self):
        log, gap = STATE["log"], STATE["gap"]
        self.ref = SH.card_reference(log, SH.make_removal(log, "gap", gap), STATE["recorder"], heldout_rows=500)

    def test_known_answers_on_synthetic_trajectories(self):
        ref, gap = self.ref, STATE["gap"]
        n = 1000
        x = np.zeros((n, 13))
        x[:, :10] = ref.float_mean
        x[:, TC.CURRENT], x[:, TC.WAVE1] = 0.2, 0.0             # inside the recorded conditions, push 0.2
        x[:400, TC.WAVE1] = 0.39                                # push 0.59 > trigger: region for 400 rows
        x[900:, TC.CURRENT] = 0.9                               # 100 rows beyond the recorded current
        a = np.full((n, 4), 0.5)
        a[:250, 0] = 0.9                                        # 250 rows past 80%, all inside recorded conditions
        a[900:950, 0] = -0.95                                   # 50 rows past 80%, outside
        a[600:700] = gap.centre                                 # 100 rows inside the ball at <= 80%
        a[600:700, 0] = -0.79
        a[700:720] = gap.centre
        a[700:720, 0] = -0.82                                   # 20 rows inside the ball and past 80%
        self.assertTrue(np.all(gap.contains(a[600:720])))
        J = np.linspace(-10, -1, 50)
        out = SH.card_measures(x, a, ref, TC.STALL, J=J, J_capped=J + 2.0, anchors=dict(recording=-20.0, top=0.0),
                               ms_states=50, knn_states=200)
        self.assertAlmostEqual(out["unrecorded_throttle_share"], 320 / n)
        self.assertAlmostEqual(out["full_throttle_region_share"], 500 / n)   # 400 + the 100 strong-current rows
        self.assertAlmostEqual(out["outside_recorded_conditions"]["share"], 100 / n)
        self.assertAlmostEqual(out["outside_recorded_conditions"]["by_current"], 100 / n)
        self.assertAlmostEqual(out["inside_removed_ball"]["share"], 120 / n)
        self.assertAlmostEqual(out["inside_removed_ball"]["past_80"], 20 / n)
        self.assertAlmostEqual(out["inside_removed_ball"]["at_or_below_80"], 100 / n)
        self.assertAlmostEqual(out["covered_share"], 270 / 320)
        self.assertAlmostEqual(out["loss_from_unrecorded_throttle"]["J"], 2.0)
        self.assertAlmostEqual(out["points"]["loss_from_unrecorded_throttle"], 10.0)
        self.assertEqual(out["outside_recorded_conditions"]["by_floats"], 0.0)

    def test_stall_aware_controller_loses_nothing_from_unrecorded_throttle(self):
        out = SH.shift_card(RC.stall_aware(), "strong_current", TC.STALL, self.ref, ms_states=100, knn_states=300)
        self.assertEqual(out["unrecorded_throttle_share"], 0.0)
        self.assertEqual(out["loss_from_unrecorded_throttle"]["J"], 0.0)
        self.assertIsNone(out["covered_share"])
        self.assertEqual(out["outside_recorded_conditions"]["by_current"], 1.0)
        lin = SH.shift_card(RC.rising_thrust(), "strong_current", TC.STALL, self.ref, ms_states=100, knn_states=300)
        self.assertGreater(lin["unrecorded_throttle_share"], 0.5)
        self.assertGreater(lin["loss_from_unrecorded_throttle"]["J"], 0.0)


class PoolTest(unittest.TestCase):
    def test_pools(self):
        log, gap, rec = STATE["log"], STATE["gap"], STATE["recorder"]
        p1 = SH.make_pools(log, gap, rec, size=120, ms_draws=512)
        p2 = SH.make_pools(log, gap, rec, size=120, ms_draws=512)
        self.assertEqual(p1["sha256"], p2["sha256"])
        train = set(log.split_rows("train").tolist())
        a = log["actions"]
        self.assertTrue(set(p1["removed"].tolist()) <= train)
        self.assertTrue(np.all(gap.contains(a[p1["removed"]])))
        for name in ("region", "far"):
            self.assertEqual(len(p1[name]), 120)
            self.assertTrue(set(p1[name].tolist()) <= train)
            self.assertFalse(np.any(gap.contains(a[p1[name]])))
        self.assertTrue(np.all(rec.in_region(log["infos/state"][p1["region"]])))
        d = GAP.distance(np.asarray(a[p1["far"]], np.float64), gap.centre)
        self.assertTrue(np.all(d > 2 * gap.rho))
        self.assertTrue(np.all(p1["far_removed_share"] <= 0.01))
        np.testing.assert_allclose(p1["c_cap"], np.clip(rec.pattern, -0.8, 0.8))
        self.assertEqual(p1["c_hi"][0], -1.0)


class StageZeroTest(unittest.TestCase):
    def test_trigger_bisection_and_grid(self):
        from experiments.shift import stage0 as S0

        grid = S0.fallback_grid()
        self.assertEqual(len(grid), 8)
        self.assertEqual(len(set(grid)), 8)
        self.assertEqual(grid[0], (0.13, 0.3, 0.25))
        self.assertEqual(S0.expected_trigger(RC.DEFAULT_SPEC), 0.565)
        self.assertIsNone(S0.expected_trigger(RC.RecorderSpec(full_throttle_share_target=0.12)))
        trigger, history = S0.bisect_trigger()
        self.assertEqual(trigger, RC.TRIGGER)
        self.assertEqual(len(history), 14)
        # the trigger only depends on the water's push, so the stall slope of the retune cannot move it
        self.assertEqual(S0.bisect_trigger(physics=TC.Physics(stall_slope=3.0))[0], RC.TRIGGER)

    def test_forecast_known_answers(self):
        from experiments.shift import stage0 as S0
        from experiments.shift import truth as TR

        lev = S0.dose_cap_leverage(0.58)
        self.assertAlmostEqual(lev["leverage_bend"], 0.088, places=3)
        self.assertAlmostEqual(lev["leverage_edge"], 0.193, places=3)
        self.assertAlmostEqual(lev["leverage_cap2_bend"], 0.142, places=3)
        self.assertAlmostEqual(lev["leverage_bend_max"], 0.101, places=3)
        an = TR.anchors(situations=("strong_current",))["strong_current"]
        hand, rising = RC.hand_only(), RC.rising_thrust()
        harm = {}
        for shape in ("bend", "edge"):
            trusting = S0._Blend(hand, rising, 0.58, shape, S0._const(1.0))
            informed = S0._Blend(hand, rising, 0.58, shape, S0._const(1.0), cap=0.8)
            pts = lambda p: float(TR.points(TR.bank_rollout(p, "strong_current").J.mean(), an))
            harm[shape] = pts(informed) - pts(trusting)
        self.assertAlmostEqual(harm["bend"], 9.5, delta=0.05)
        self.assertAlmostEqual(harm["edge"], 40.4, delta=0.05)


class DensityNoteTest(unittest.TestCase):
    def test_density_note_runs(self):
        note = SH.density_check_note(STATE["log"], STATE["gap"], query_rows=500, gap_rows=300)
        self.assertIn("i_dense", note["passes"])
        self.assertGreater(note["r_over_rho"], 1.0)


if __name__ == "__main__":
    unittest.main()
