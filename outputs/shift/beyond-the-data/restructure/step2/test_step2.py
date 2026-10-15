"""Unit tests of step 2's pure definitions on synthetic data (no gate data, no training).

Run: PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu python test_step2.py   (plain unittest; pytest is not installed)
Covers: the arm table built from a fake step-1 results.json (switches, heights, the forced arm, each world's own
ramp ends, refusals), the ramp and the paired penalty (NumPy and jax.numpy), height 0 gives zero penalty and zero
gradient, NET, the go/no-go G1-G3, the attribution A1-A3 (wordings both ways, per-seed differences, the dose readout,
the dose table, the facts on the frozen arm table), version 6's marks restated, write-once outputs, the output guard,
the seed refusals, the frozen-file checks, the lock checks, the verified imports (exec_verified, unverified_modules,
local_imports, and that every sibling import of the pinned version-6 files is pinned) and PRESPEC.md's tables against
prespec.json (review of 15 Oct 2026).
"""
import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2lib as S  # noqa: E402

RULE = dict(g1_margin=10.0, g2_floor=-3.0, g3_margin=15.0, attribution_margin=5.0, m1_margin=15.0,
            m1_positive_min=4, m2_margin=15.0, m3_floor=-3.0, m3_clone_margin=25.0, m4_floor=-5.0)


def fake_entry(on, height, q90f, q90r, meanf, meanr, p, t_lo=0.786, t_up=0.995):
    return dict(t_lo=t_lo, t_up=t_up,
                certificate=dict(on=on, height=height, plain_gap=q90f - q90r),
                groups=dict(q90_full_throttle=q90f, q90_rest=q90r, mean_full_throttle=meanf, mean_rest=meanr),
                rank_test=dict(p=p))


def fake_step1(seeds=S.SEEDS):
    per = {}
    for s in seeds:
        per[f"stall/seed{s}"] = fake_entry(True, 2.0 + s, 1.2 + 0.1 * s, -0.04, 0.4, -0.15, 1e-10)
        per[f"no_stall/seed{s}"] = fake_entry(False, 0.1, 0.02, -0.03, -0.17, -0.16, 0.3)
    return dict(mode="official", prespec_sha256="p" * 64, go_no_go=dict(verdict="GO"), per_seed_world=per)


class ArmTable(unittest.TestCase):
    def test_switches_heights_and_multipliers(self):
        t = S.arm_table(fake_step1())
        for s in S.SEEDS:
            st, ns = t["stall"][s]["arms"], t["no_stall"][s]["arms"]
            self.assertEqual(st["obca_k1"], dict(switch=True, h=2.0 + s, k=1.0))
            self.assertEqual(st["obca_k2"], dict(switch=True, h=2.0 + s, k=2.0))
            self.assertEqual(st["fixed_gated"], dict(switch=True, h=1.0, k=1.0))
            self.assertAlmostEqual(st["conformal_k1"]["h"], 1.2 + 0.1 * s + 0.04, places=12)
            self.assertTrue(st["meangap_k1"]["switch"])
            self.assertAlmostEqual(st["meangap_k1"]["h"], 0.55, places=12)
            for name in ("obca_k1", "obca_k2", "fixed_gated", "conformal_k1", "meangap_k1"):
                self.assertFalse(ns[name]["switch"])
                self.assertEqual(ns[name]["h"], 0.0)
            # the forced arm: always on, the stalling world's BCA height in BOTH worlds
            self.assertEqual(st["obca_forced"], dict(switch=True, h=2.0 + s, k=1.0))
            self.assertEqual(ns["obca_forced"], dict(switch=True, h=2.0 + s, k=1.0))
            self.assertEqual(t["stall"][s]["t_lo"], 0.786)

    def test_ramp_ends_are_each_worlds_own(self):
        # the two logs are recorded separately, so the gap rule gives each world its own ends; h and k of the forced
        # arm are carried over from the stalling world, the ends are not
        r = fake_step1()
        for s in S.SEEDS:
            r["per_seed_world"][f"no_stall/seed{s}"]["t_lo"] = 0.783
            r["per_seed_world"][f"no_stall/seed{s}"]["t_up"] = 0.993 if s == 4 else 0.995
        t = S.arm_table(r)
        for s in S.SEEDS:
            self.assertEqual((t["stall"][s]["t_lo"], t["stall"][s]["t_up"]), (0.786, 0.995))
            self.assertEqual(t["no_stall"][s]["t_lo"], 0.783)
            self.assertEqual(t["no_stall"][s]["arms"]["obca_forced"], t["stall"][s]["arms"]["obca_k1"])
        self.assertEqual(t["no_stall"][4]["t_up"], 0.993)
        self.assertEqual(t["no_stall"][3]["t_up"], 0.995)

    def test_conformal_margin_is_strict_and_meangap_floor_is_zero(self):
        r = fake_step1()
        r["per_seed_world"]["no_stall/seed0"] = fake_entry(False, 0.1, 0.1, 0.0, -0.2, -0.1, 0.01)
        t = S.arm_table(r)["no_stall"][0]["arms"]
        self.assertFalse(t["conformal_k1"]["switch"])          # a gap of exactly 0.10 is not above 0.10
        self.assertEqual(t["conformal_k1"]["h"], 0.0)
        self.assertTrue(t["meangap_k1"]["switch"])              # p < 0.05 ...
        self.assertEqual(t["meangap_k1"]["h"], 0.0)             # ... but the mean gap is negative: h = max(., 0) = 0
        r["per_seed_world"]["no_stall/seed0"]["groups"]["q90_full_throttle"] = 0.1001
        self.assertTrue(S.arm_table(r)["no_stall"][0]["arms"]["conformal_k1"]["switch"])

    def test_infinite_height_refused_only_when_switched_on(self):
        r = fake_step1()
        r["per_seed_world"]["no_stall/seed3"]["certificate"]["height"] = "inf"   # step 1 stores inf as a string
        t = S.arm_table(r)
        self.assertEqual(t["no_stall"][3]["arms"]["obca_k1"]["h"], 0.0)
        self.assertTrue(math.isinf(t["no_stall"][3]["sources"]["bca_height"]))
        r2 = copy.deepcopy(r)
        r2["per_seed_world"]["no_stall/seed3"]["certificate"]["on"] = True
        with self.assertRaises(SystemExit):
            S.arm_table(r2)
        r3 = fake_step1()
        r3["per_seed_world"]["stall/seed1"]["certificate"]["height"] = "inf"    # also refuses the forced arm
        with self.assertRaises(SystemExit):
            S.arm_table(r3)

    def test_negative_height_refused(self):
        r = fake_step1()
        r["per_seed_world"]["stall/seed2"]["certificate"]["height"] = -0.5
        with self.assertRaises(SystemExit):
            S.arm_table(r)

    def test_table_matches_expected(self):
        t = S.arm_table(fake_step1())
        exp = json.loads(json.dumps(S.jsonable({w: {str(s): r for s, r in t[w].items()} for w in S.WORLDS})))
        self.assertTrue(S.table_matches(t, exp))
        exp["no_stall"]["2"]["arms"]["meangap_k1"]["h"] = 0.01
        self.assertFalse(S.table_matches(t, exp))
        exp = json.loads(json.dumps(S.jsonable({w: {str(s): r for s, r in t[w].items()} for w in S.WORLDS})))
        exp["stall"]["0"]["t_lo"] += 1e-9
        self.assertFalse(S.table_matches(t, exp))

    def test_load_step1_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            good = fake_step1()
            for change, ok in ((None, True), (("mode", "scratch"), False), (("prespec_sha256", "q" * 64), False),
                               (("go_no_go", dict(verdict="NO-GO")), False)):
                r = copy.deepcopy(good)
                if change:
                    r[change[0]] = change[1]
                p = Path(d) / "results.json"
                p.write_text(json.dumps(r))
                want = dict(sha256=S.file_sha256(p), prespec_sha256="p" * 64)
                if ok:
                    self.assertEqual(S.load_step1(p, want)["mode"], "official")
                else:
                    with self.assertRaises(SystemExit):
                        S.load_step1(p, want)
            with self.assertRaises(SystemExit):
                S.load_step1(p, dict(sha256="0" * 64, prespec_sha256="p" * 64))

    def test_num(self):
        self.assertEqual(S.num("inf"), math.inf)
        self.assertEqual(S.num("-inf"), -math.inf)
        self.assertTrue(math.isnan(S.num("nan")))
        self.assertEqual(S.num(1), 1.0)
        with self.assertRaises(ValueError):
            S.num("1.0")


class Penalty(unittest.TestCase):
    T_LO, T_UP = 0.75, 1.0                                     # dyadic: exact arithmetic below

    def test_throttle_and_ramp(self):
        a = np.array([[0.1, -0.9, 0.2, 0.0], [-1.0, 0.5, 0.0, 0.3], [0.0, 0.0, 0.0, 0.5]])
        np.testing.assert_array_equal(S.throttle(a), [0.9, 1.0, 0.5])
        t = np.array([0.0, 0.5, 0.75, 0.875, 1.0, 1.25])
        np.testing.assert_array_equal(S.ramp(t, self.T_LO, self.T_UP), [0.0, 0.0, 0.0, 0.5, 1.0, 1.0])

    def test_paired_one_sided_penalty(self):
        pi = np.array([[0.875, 0, 0, 0], [1.0, 0, 0, 0], [0.5, 0, 0, 0], [0.875, 0, 0, 0]])
        rec = np.array([[0.5, 0, 0, 0], [0.875, 0, 0, 0], [1.0, 0, 0, 0], [0.0, 0.0, -1.0, 0]])
        p = S.ramp_penalty(pi, rec, 2.0, 3.0, self.T_LO, self.T_UP)
        # k [h ramp(t(pi)) - h ramp(t(a_rec))]_+: 3 [2*0.5 - 0]_+, 3 [2*1 - 2*0.5]_+, 3 [0 - 2]_+, 3 [1 - 2]_+
        np.testing.assert_array_equal(p, [3.0, 3.0, 0.0, 0.0])
        np.testing.assert_array_equal(S.outcome_ramp(pi, 2.0, self.T_LO, self.T_UP), [1.0, 2.0, 0.0, 1.0])

    def test_height_zero_gives_zero_penalty(self):
        r = np.random.default_rng(3)
        pi, rec = r.uniform(-1, 1, (500, 4)), r.uniform(-1, 1, (500, 4))
        self.assertEqual(float(np.abs(S.ramp_penalty(pi, rec, 0.0, 2.0, self.T_LO, self.T_UP)).max()), 0.0)

    def test_jax_matches_numpy_and_zero_height_has_zero_gradient(self):
        import jax
        import jax.numpy as jnp
        r = np.random.default_rng(4)
        pi = r.uniform(-1, 1, (256, 4)).astype(np.float32)
        rec = r.uniform(-1, 1, (256, 4)).astype(np.float32)
        lo, up = np.float32(0.786), np.float32(0.995)
        pj = np.asarray(S.ramp_penalty(jnp.asarray(pi), jnp.asarray(rec), jnp.float32(1.7), jnp.float32(1.0),
                                       jnp.float32(lo), jnp.float32(up), xp=jnp))
        pn = S.ramp_penalty(pi.astype(np.float64), rec.astype(np.float64), 1.7, 1.0, float(lo), float(up))
        np.testing.assert_allclose(pj, pn, atol=1e-5)
        self.assertTrue((pn > 0).any())

        def total(p, h):
            return S.ramp_penalty(p, jnp.asarray(rec), h, jnp.float32(2.0), jnp.float32(lo), jnp.float32(up),
                                  xp=jnp).sum()
        g0 = np.asarray(jax.grad(total)(jnp.asarray(pi), jnp.float32(0.0)))
        self.assertEqual(float(np.abs(g0).max()), 0.0)
        self.assertTrue(np.all(np.isfinite(g0)))
        g1 = np.asarray(jax.grad(total)(jnp.asarray(pi), jnp.float32(1.0)))
        self.assertGreater(float(np.abs(g1).max()), 0.0)     # with a height the penalty pushes the command
        # the gradient acts only where the actor's throttle lies inside the ramp and above the recorded one's
        t = np.abs(pi).max(1)
        outside = (t <= lo) | (t >= up)
        self.assertEqual(float(np.abs(g1[outside]).max()), 0.0)


def pts_world(values):
    """values: {arm: (stall strong current per seed, no-stall strong current per seed)} -> the pts structure."""
    pts = {w: {s: {} for s in S.SEEDS} for w in S.WORLDS}
    for arm, (st, ns) in values.items():
        for i, s in enumerate(S.SEEDS):
            pts["stall"][s][arm] = {sit: (st[i] if sit == "strong_current" else 90.0) for sit in S.SITUATIONS}
            pts["no_stall"][s][arm] = {sit: (ns[i] if sit == "strong_current" else 90.0) for sit in S.SITUATIONS}
    return pts


def base_values():
    five = lambda x: [float(x)] * 5   # noqa: E731
    return {"none": (five(-13), five(95)), "clone": (five(50), five(60)),
            "obca_k1": (five(80), five(95)),               # NET 93
            "obca_forced": (five(80), five(35)),           # NET 33
            "standin": (five(85), five(40)),               # NET 43
            "even_pull": (five(-13), five(94)),            # NET -1
            "random_1": (five(-12), five(95)), "random_2": (five(-14), five(95)),
            "conformal_k1": (five(79), five(95)), "fixed_gated": (five(60), five(95)), "meangap_k1": (five(70), five(90))}


class Measures(unittest.TestCase):
    def test_net(self):
        n = S.net(pts_world(base_values()), "obca_k1", S.SEEDS)
        self.assertEqual(n["net"], 93.0)
        self.assertEqual(n["gain_stall"], 93.0)
        self.assertEqual(n["loss_no_stall"], 0.0)
        n = S.net(pts_world(base_values()), "obca_forced", S.SEEDS)
        self.assertEqual((n["net"], n["gain_stall"], n["loss_no_stall"]), (33.0, 93.0, 60.0))
        self.assertEqual(S.net(pts_world(base_values()), "none", S.SEEDS)["net"], 0.0)

    def test_net_needs_every_seed(self):
        pts = pts_world(base_values())
        pts["no_stall"][2]["obca_k1"] = None                      # e.g. a failed arm
        self.assertIsNone(S.net(pts, "obca_k1", S.SEEDS)["net"])
        pts = pts_world(base_values())
        pts["stall"][4] = {}                                      # a stopped seed-world
        self.assertIsNone(S.net(pts, "obca_k1", S.SEEDS)["net"])

    def test_go(self):
        v = S.go_no_go(pts_world(base_values()), S.SEEDS, RULE)
        self.assertEqual((v["G1"], v["G2"], v["G3"], v["verdict"]), (True, True, True, "GO"))
        self.assertEqual(v["G1_detail"]["best_world_blind"], "standin")
        self.assertEqual(v["G3_detail"]["mean"], 93.0)

    def test_g1_margin_boundary(self):
        vals = base_values()
        vals["standin"] = ([83.0] * 5, [95.0 - 0.0] * 5)          # NET 96 -> 93 < 106
        self.assertFalse(S.go_no_go(pts_world(vals), S.SEEDS, RULE)["G1"])
        vals["standin"] = ([83.0] * 5, [12.0] * 5)                # NET 13 -> 93 >= 23
        self.assertTrue(S.go_no_go(pts_world(vals), S.SEEDS, RULE)["G1"])
        vals["standin"] = ([83.0] * 5, [92.0] * 5)                # NET 93: equal is not enough (needs + 10)
        self.assertFalse(S.go_no_go(pts_world(vals), S.SEEDS, RULE)["G1"])
        vals["standin"] = ([83.0] * 5, [82.0] * 5)                # NET 83: 93 >= 93 holds exactly
        self.assertTrue(S.go_no_go(pts_world(vals), S.SEEDS, RULE)["G1"])

    def test_g1_fails_when_a_rival_is_missing(self):
        pts = pts_world(base_values())
        for s in S.SEEDS:
            pts["stall"][s]["random_2"] = None                    # unmatched in the stalling world
        v = S.go_no_go(pts, S.SEEDS, RULE)
        self.assertFalse(v["G1"])
        self.assertEqual(v["verdict"], "NO-GO")

    def test_g2_and_g3(self):
        pts = pts_world(base_values())
        for s in S.SEEDS:
            pts["stall"][s]["obca_k1"]["big_kicks"] = 86.0        # 86 - 90 = -4 < -3
        v = S.go_no_go(pts, S.SEEDS, RULE)
        self.assertFalse(v["G2"])
        self.assertEqual(v["verdict"], "NO-GO")
        for s in S.SEEDS:
            pts["stall"][s]["obca_k1"]["big_kicks"] = 87.0        # -3 holds
        self.assertTrue(S.go_no_go(pts, S.SEEDS, RULE)["G2"])
        vals = base_values()
        vals["even_pull"] = ([66.0] * 5, [94.0] * 5)              # 80 - 66 = 14 < 15
        self.assertFalse(S.go_no_go(pts_world(vals), S.SEEDS, RULE)["G3"])
        vals["even_pull"] = ([65.0] * 5, [94.0] * 5)              # 15 holds
        self.assertTrue(S.go_no_go(pts_world(vals), S.SEEDS, RULE)["G3"])

    def test_attribution(self):
        a = S.attribution(pts_world(base_values()), S.SEEDS, RULE)
        self.assertTrue(a["A1"]["holds"])        # conformal NET 92 >= 93 - 5
        self.assertFalse(a["A2"]["holds"])       # fixed NET 73 < 88
        self.assertFalse(a["A3"]["holds"])       # meangap NET 83 - 5 = 78 < 88
        vals = base_values()
        vals["meangap_k1"] = ([83.0] * 5, [90.0] * 5)             # NET 96 - 5 = 91 >= 88
        self.assertTrue(S.attribution(pts_world(vals), S.SEEDS, RULE)["A3"]["holds"])
        vals["fixed_gated"] = ([75.0] * 5, [95.0] * 5)            # NET 88: boundary holds
        self.assertTrue(S.attribution(pts_world(vals), S.SEEDS, RULE)["A2"]["holds"])
        pts = pts_world(base_values())
        pts["stall"][0]["conformal_k1"] = None
        a = S.attribution(pts, S.SEEDS, RULE)["A1"]
        self.assertIsNone(a["holds"])
        self.assertEqual(a["wording"], S.DOSE_READING["undefined"])
        self.assertEqual((a["seeds_defined"], a["per_seed_difference"][0]), (4, None))

    def test_attribution_wording_both_ways(self):
        # review of 15 Oct 2026: a failed A has a fixed wording too (dose, never credit to a part of BCA)
        a = S.attribution(pts_world(base_values()), S.SEEDS, RULE)
        words = {tag: (h, f_) for tag, _, h, f_ in S.ATTRIBUTION}
        self.assertEqual(a["A1"]["wording"], words["A1"][0])                 # holds
        self.assertEqual(a["A2"]["wording"], words["A2"][1])                 # fails
        self.assertEqual(a["A3"]["wording"], words["A3"][1])
        for tag in ("A1", "A2", "A3"):
            self.assertTrue(a[tag]["meaning_if_fails"].startswith("a larger dose did better"))
            self.assertIn("Nothing is attributed", a[tag]["meaning_if_fails"])
        self.assertFalse(a["calibration_credit"]["credited"])
        self.assertIsNone(a["calibration_credit"]["facts"])                  # no table given
        t = S.arm_table(fake_step1())
        self.assertTrue(S.attribution(pts_world(base_values()), S.SEEDS, RULE, t)["calibration_credit"]["facts"][
            "no_rival_height_reaches_bca"])

    def test_attribution_per_seed(self):
        vals = base_values()
        vals["conformal_k1"] = ([80.0, 80.0, 70.0, 60.0, 80.0], [95.0] * 5)  # NET per seed 93 93 83 73 93
        a = S.attribution(pts_world(vals), S.SEEDS, RULE)["A1"]
        self.assertEqual(a["per_seed_difference"], [0.0, 0.0, -10.0, -20.0, 0.0])
        self.assertEqual((a["holds_in_seeds"], a["seeds_defined"]), (3, 5))
        self.assertEqual(a["net_arm"], 87.0)                                 # 87 < 88: fails on the mean
        self.assertFalse(a["holds"])

    def test_dose_readout(self):
        for k2, reading in (((90.0, 95.0), "rising"),       # NET 103 vs 93: +10
                            ((84.0, 95.0), "plateau"),      # +4
                            ((85.0, 95.0), "plateau"),      # +5: the boundary is a plateau
                            ((86.0, 95.0), "rising"),       # +6
                            ((75.0, 95.0), "plateau"),      # -5
                            ((70.0, 95.0), "falling")):     # -10
            vals = base_values()
            vals["obca_k2"] = ([k2[0]] * 5, [k2[1]] * 5)
            d = S.attribution(pts_world(vals), S.SEEDS, RULE)["dose"]
            self.assertEqual(d["reading"], reading, k2)
            self.assertEqual(d["wording"], S.DOSE_READING[reading])
            self.assertEqual(d["per_seed_difference"], [d["difference"]] * 5)
        d = S.attribution(pts_world(base_values()), S.SEEDS, RULE)["dose"]  # obca_k2 missing
        self.assertEqual((d["reading"], d["difference"]), ("undefined", None))

    def test_dose_table(self):
        t = S.arm_table(fake_step1())
        vals = base_values()
        vals["obca_k2"] = ([84.0] * 5, [95.0] * 5)
        dt = S.dose_table(pts_world(vals), S.SEEDS, t)
        for s in S.SEEDS:
            self.assertEqual(sorted(dt[s]), sorted(S.DOSE_ARMS))
            self.assertEqual(dt[s]["obca_k2"]["dose_stall"], 2.0 * (2.0 + s))
            self.assertEqual(dt[s]["obca_k1"]["dose_stall"], 2.0 + s)
            self.assertEqual(dt[s]["fixed_gated"]["dose_stall"], 1.0)
            self.assertEqual(dt[s]["obca_k1"]["dose_no_stall"], 0.0)
            self.assertEqual(dt[s]["obca_k1"]["gain_stall"], 93.0)
            self.assertEqual(dt[s]["meangap_k1"]["loss_no_stall"], 5.0)
            self.assertEqual(dt[s]["meangap_k1"]["net"], 78.0)
        # prespec.json's table has string seed keys; the same numbers come out
        exp = json.loads(json.dumps(S.jsonable({w: {str(s): r for s, r in t[w].items()} for w in S.WORLDS})))
        self.assertEqual(S.dose_table(pts_world(vals), S.SEEDS, exp), dt)
        pts = pts_world(vals)
        pts["no_stall"][1]["obca_k1"] = None
        self.assertIsNone(S.dose_table(pts, S.SEEDS, t)[1]["obca_k1"]["net"])

    def test_v6_marks_restated(self):
        vals = base_values()
        vals["even_pull"] = ([60.0, 60.0, 60.0, 90.0, 90.0], [94.0] * 5)   # mean 8 ... and only 3 positive
        m = S.analogues(pts_world(vals), S.SEEDS, RULE)
        self.assertEqual(m["M1_even_pull"]["positive"], 3)
        self.assertFalse(m["M1_even_pull"]["passed"])
        self.assertTrue(m["M2_random_surfaces"]["passed"])
        self.assertTrue(m["M4_best_standin"]["passed"])          # 80 - 85 = -5 holds exactly
        self.assertFalse(m["M3_keeps_improvement"]["passed"])    # recorded vs clone 90 - 90 = 0 < 25


class Guards(unittest.TestCase):
    def test_seed_refusals(self):
        for s in S.SEEDS:
            self.assertEqual(S.check_seed(s), s)
        for s in list(range(5, 10)) + list(range(10, 15)):
            with self.assertRaises(SystemExit) as cm:
                S.check_seed(s)
            self.assertIn("reserved", str(cm.exception))
        for s in (-1, 15, 99):
            with self.assertRaises(SystemExit):
                S.check_seed(s)
        with self.assertRaises(SystemExit):
            S.check_world("mixed")

    def test_guard_out(self):
        for rel in ("runs/shift_explore/stage2/x", "runs/shift/x", "experiments/shift/x",
                    "outputs/shift/beyond-the-data/stage2/x", "outputs/shift/beyond-the-data/stage2_v7/x",
                    "outputs/shift/beyond-the-data/restructure/step1/x", "runs/shift_explore/restructure/step1/x",
                    "runs/shift_explore/room/x", "runs/shift_explore/standins/x"):
            for official in (True, False):
                with self.assertRaises(SystemExit):
                    S.guard_out(S.ROOT / rel, official)
        with self.assertRaises(SystemExit):
            S.guard_out(S.OFFICIAL_OUT / "x", official=False)        # scratch must lie outside the repository
        with self.assertRaises(SystemExit):
            S.guard_out("/root/step2_scratch/x", official=True)       # official must lie in the step-2 folder
        self.assertEqual(S.guard_out(S.OFFICIAL_OUT / "policy/seed0/stall", True),
                         (S.OFFICIAL_OUT / "policy/seed0/stall").resolve())
        self.assertTrue(str(S.guard_out("/tmp/step2_test/x", False)).startswith("/tmp/"))

    def test_write_once_and_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a" / "values.json"
            digest = S.write_json_once(p, dict(x=1.0, y=float("inf"), z=np.float32(2.5)), official=False)
            self.assertEqual(S.read_json_checked(p), dict(x=1.0, y="inf", z=2.5))
            self.assertEqual((Path(d) / "a" / "values.json.sha256").read_text().split()[0], digest)
            with self.assertRaises(SystemExit):
                S.write_json_once(p, dict(x=2.0), official=False)
            with self.assertRaises(SystemExit):
                S.write_bytes_once(p, b"x", official=False)
            p.write_text(p.read_text().replace("1.0", "1.5"))
            with self.assertRaises(SystemExit):
                S.read_json_checked(p)
            self.assertIsNone(S.read_json_checked(Path(d) / "missing.json", required=False))
            with self.assertRaises(SystemExit):
                S.read_json_checked(Path(d) / "missing.json")


class Frozen(unittest.TestCase):
    def make(self, d):
        names = ("a.py", "b.sh", "prespec.json")
        for n in names:
            (Path(d) / n).write_text(f"content of {n}\n" if n != "prespec.json" else json.dumps(dict(v=1)))
        (Path(d) / "prespec.json.sha256").write_text(f"{S.file_sha256(Path(d) / 'prespec.json')}  prespec.json\n")
        man = dict(files={n: S.file_sha256(Path(d) / n) for n in names})
        (Path(d) / S.MANIFEST_NAME).write_text(json.dumps(man))
        return names

    def test_frozen_files(self):
        with tempfile.TemporaryDirectory() as d:
            names = self.make(d)
            spec, sha = S.load_prespec(d)
            self.assertEqual(spec, dict(v=1))
            S.verify_manifest(d, names)
            with self.assertRaises(SystemExit):
                S.verify_manifest(d, names + ("c.py",))           # the manifest must list exactly the step-2 files
            (Path(d) / "a.py").write_text("changed\n")
            with self.assertRaises(SystemExit):
                S.verify_manifest(d, names)
        with tempfile.TemporaryDirectory() as d:
            names = self.make(d)
            (Path(d) / "prespec.json").write_text(json.dumps(dict(v=2)))
            with self.assertRaises(SystemExit):
                S.load_prespec(d)
            (Path(d) / "prespec.json.sha256").unlink()
            with self.assertRaises(SystemExit):
                S.load_prespec(d)
            (Path(d) / S.MANIFEST_NAME).unlink()
            with self.assertRaises(SystemExit):
                S.verify_manifest(d, names)

    def test_the_real_folder_is_frozen(self):
        S.load_prespec()
        S.verify_manifest()

    def test_official_outputs_detected(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(S.official_outputs_exist(d), [])
            p = S.seed_dir(d, 0, "stall")
            p.mkdir(parents=True)
            (p / "lock.json").write_text("{}")
            self.assertEqual(len(S.official_outputs_exist(d)), 1)

    def test_lock_problems(self):
        lock = dict(kind="step2-policy-lock", mode="official", prespec_sha256="a", code_manifest_sha256="b", seed=1,
                    world="no_stall", steps=20000, checks=dict(b_w0=dict(passed=True)), cpus=[3])
        self.assertEqual(S.lock_problems(lock, "a", "b", 1, "no_stall", True, 20000), [])
        self.assertTrue(S.lock_problems(dict(lock, mode="scratch"), "a", "b", 1, "no_stall", True, 20000))
        self.assertTrue(S.lock_problems(lock, "x", "b", 1, "no_stall", True, 20000))
        self.assertTrue(S.lock_problems(lock, "a", "b", 2, "no_stall", True, 20000))
        self.assertTrue(S.lock_problems(dict(lock, steps=40), "a", "b", 1, "no_stall", True, 20000))
        self.assertTrue(S.lock_problems(dict(lock, checks=dict(b_w0=dict(passed=False))), "a", "b", 1, "no_stall",
                                        True, 20000))
        self.assertTrue(S.lock_problems(dict(lock, cpus=[0, 1]), "a", "b", 1, "no_stall", True, 20000))
        self.assertEqual(S.lock_problems(dict(lock, mode="scratch", steps=40), "a", "b", 1, "no_stall", False,
                                         20000), [])


class Imports(unittest.TestCase):
    """Review of 15 Oct 2026: every file of a pinned folder runs from verified bytes (weights.py was not)."""

    def test_exec_verified_binds_plain_imports(self):
        with tempfile.TemporaryDirectory() as d:
            tag = f"step2test{os.getpid()}"
            dep, user, other = f"{tag}_dep", f"{tag}_user", f"{tag}_other"
            (Path(d) / f"{dep}.py").write_text("VALUE = 7\n")
            (Path(d) / f"{user}.py").write_text(f"import sys\nsys.path.insert(0, {d!r})\nimport {dep} as D\n")
            (Path(d) / f"{other}.py").write_text("VALUE = 8\n")
            sha = {n: S.file_sha256(Path(d) / f"{n}.py") for n in (dep, user, other)}
            try:
                with self.assertRaises(SystemExit):
                    S.exec_verified(dep, Path(d) / f"{dep}.py", "0" * 64)          # other bytes than pinned
                self.assertNotIn(dep, sys.modules)
                m = S.exec_verified(dep, Path(d) / f"{dep}.py", sha[dep])
                self.assertEqual((m.VALUE, m.__step2_sha256__), (7, sha[dep]))
                self.assertIs(S.exec_verified(dep, Path(d) / f"{dep}.py", sha[dep]), m)
                u = S.exec_verified(user, Path(d) / f"{user}.py", sha[user])
                self.assertIs(u.D, m)                    # a plain `import` binds to the verified module
                self.assertEqual(S.unverified_modules([d]), [])
                importlib.invalidate_caches()
                importlib.import_module(other)           # through the normal import system: no hash check
                bad = S.unverified_modules([d])
                self.assertEqual(len(bad), 1)
                self.assertTrue(bad[0].startswith(other))
                self.assertEqual(S.unverified_modules([Path(d) / "elsewhere"]), [])
                with self.assertRaises(SystemExit):      # already loaded from elsewhere (unverified)
                    S.exec_verified(other, Path(d) / f"{other}.py", sha[other])
            finally:
                for n in (dep, user, other):
                    sys.modules.pop(n, None)
                while d in sys.path:
                    sys.path.remove(d)

    def test_local_imports(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("b", "c", "d", "e"):
                (Path(d) / f"{n}.py").write_text("")
            (Path(d) / "a.py").write_text("import os\nimport b\nfrom c import x\nimport e.sub as z\n\n\ndef f():\n"
                                          "    import d\n    from . import q\n    import numpy\n")
            self.assertEqual(S.local_imports(Path(d) / "a.py", d), ["b", "c", "d", "e"])

    def test_v6_import_closure_is_pinned(self):
        spec, _ = S.load_prespec()
        v6 = spec["inputs"]["v6_code"]
        folder = S.ROOT / v6["dir"]
        pins = v6["sha256"]
        self.assertEqual(sorted(pins), ["band.py", "fresh.py", "policy.py", "stage2_common.py", "weights.py"])
        for name, h in pins.items():
            self.assertEqual(S.file_sha256(folder / name), h, name)
            for dep in S.local_imports(folder / name, folder):
                self.assertIn(f"{dep}.py", pins, f"{name} imports {dep}, which prespec.json does not pin")
        self.assertIn("weights", S.local_imports(folder / "band.py", folder))
        st1 = S.ROOT / spec["inputs"]["step1lib"]["path"]
        self.assertEqual(S.file_sha256(st1), spec["inputs"]["step1lib"]["sha256"])
        self.assertEqual(S.local_imports(st1, st1.parent), [])
        # the pins are version 6's own (its code manifest, read-only)
        man_path = S.ROOT / "runs" / "shift_explore" / "stage2" / "code_manifest.json"
        self.assertEqual(S.file_sha256(man_path), v6["v6_code_manifest_sha256"])
        man = json.loads(man_path.read_text())
        for name, h in pins.items():
            self.assertEqual(man["files"][name], h, name)


class Docs(unittest.TestCase):
    """Review of 15 Oct 2026: the frozen texts agree with the frozen numbers."""

    def setUp(self):
        self.spec, _ = S.load_prespec()
        self.md = (S.HERE / "PRESPEC.md").read_text(encoding="utf8")
        self.exp = self.spec["arm_table_expected"]

    def test_the_two_worlds_have_their_own_ramp_ends(self):
        for s in S.SEEDS:
            self.assertNotEqual(self.exp["stall"][str(s)]["t_lo"], self.exp["no_stall"][str(s)]["t_lo"])
        self.assertNotEqual(self.exp["stall"]["4"]["t_up"], self.exp["no_stall"]["4"]["t_up"])
        self.assertNotIn("recorded commands are the same in both worlds", " ".join(self.md.split()))

    def test_prespec_md_tables_match_prespec_json(self):
        ends = {int(m[0]): m[1:] for m in re.findall(
            r"^\| (\d) \| (0\.\d{4}) \| (0\.\d{4}) \| (0\.\d{4}) \| (0\.\d{4}) \|$", self.md, re.M)}
        self.assertEqual(sorted(ends), list(S.SEEDS))
        for s, got in ends.items():
            st, ns = self.exp["stall"][str(s)], self.exp["no_stall"][str(s)]
            self.assertEqual(got, tuple(f"{x:.4f}" for x in (st["t_lo"], ns["t_lo"], st["t_up"], ns["t_up"])), s)
        stall = {int(m[0]): m[1:] for m in re.findall(
            r"^\| (\d) \| (0\.\d{4}) \| (0\.\d{4}) \| (\d\.\d{3}) \| 1\.0 \| (\d\.\d{3}) \| (\d\.\d{3}) \|$", self.md,
            re.M)}
        self.assertEqual(sorted(stall), list(S.SEEDS))
        for s, got in stall.items():
            e = self.exp["stall"][str(s)]
            want = (f"{e['t_lo']:.4f}", f"{e['t_up']:.4f}") + tuple(
                f"{e['arms'][a]['h']:.3f}" for a in ("obca_k1", "conformal_k1", "meangap_k1"))
            self.assertEqual(got, want, s)

    def test_wordings_match_prespec(self):
        at = self.spec["attribution"]
        for tag, _, holds, fails in S.ATTRIBUTION:
            self.assertEqual((at["wording"][tag]["holds"], at["wording"][tag]["fails"]), (holds, fails), tag)
        for k, v in S.DOSE_READING.items():
            self.assertEqual(at["dose_readout"][k], v, k)
        self.assertEqual(at["calibration_credit"], S.CALIBRATION_CREDIT)

    def test_attribution_facts_on_the_frozen_table(self):
        f = S.attribution_facts(self.exp, S.SEEDS)
        pc = f["per_comparison"]
        self.assertEqual([pc[t]["switch_differs_in"] for t in ("A1", "A2", "A3")], [[], [], ["no_stall/seed2"]])
        self.assertTrue(f["no_rival_height_reaches_bca"])
        r = pc["A1"]["stall_dose_ratio_primary_over_arm"]
        self.assertEqual((round(min(r.values()), 2), round(max(r.values()), 2), max(r, key=r.get)), (1.34, 3.47, 0))
        self.assertEqual(f"{self.exp['no_stall']['2']['arms']['meangap_k1']['h']:.3f}", "0.048")
        for s in S.SEEDS:                               # obca_k2 doubles obca_k1's dose, with the same switch
            a1, a2 = self.exp["stall"][str(s)]["arms"]["obca_k1"], self.exp["stall"][str(s)]["arms"]["obca_k2"]
            self.assertEqual((a2["h"], a2["k"], a2["switch"]), (a1["h"], 2.0, a1["switch"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
