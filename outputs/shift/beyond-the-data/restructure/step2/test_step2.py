"""Unit tests of step 2's pure definitions on synthetic data (no gate data, no training).

Run: PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu python test_step2.py   (plain unittest; pytest is not installed)
Covers: the arm table built from a fake step-1 results.json (switches, heights, the forced arm, refusals), the
ramp and the paired penalty (NumPy and jax.numpy), height 0 gives zero penalty and zero gradient, NET, the go/no-go
G1-G3, the attribution A1-A3, version 6's marks restated, write-once outputs, the output guard, the seed refusals,
the frozen-file checks and the lock checks.
"""
import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
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
        self.assertIsNone(S.attribution(pts, S.SEEDS, RULE)["A1"]["holds"])

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
