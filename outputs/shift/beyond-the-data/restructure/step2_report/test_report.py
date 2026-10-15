"""Unit tests of the step-2 reporting add-on's pure arithmetic and guards, on synthetic numbers (no gate data, no
rollout, no step-2 output).

Run: cd /tmp && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \\
       /root/bcavenv/bin/python /home/user/bca/outputs/shift/beyond-the-data/restructure/step2_report/test_report.py
(plain unittest; pytest is not installed). Covers: the paired standard errors (against a direct loop, shared start
luck, one seed, the exact two-way estimate), the tail statistics, the episode costs and the late peak, regret and
the identity NET = total regret of none - total regret of x, the contrast terms against step2lib's own NET and
paired means, step2_value's lookups, the output and mode guards (official refusal without results.json, scratch
paths outside the repository, write-once) and the banned words.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as R  # noqa: E402

S = R.S


def synthetic_pts(seeds, rng):
    """Step-2-shaped bank means: pts[world][seed][arm][situation]."""
    pts = {w: {s: {} for s in seeds} for w in R.WORLDS}
    for w in R.WORLDS:
        for s in seeds:
            for arm in R.ARMS[w]:
                pts[w][s][arm] = {sit: float(rng.normal(50, 30)) for sit in R.SITUATIONS}
    return pts


class PairedSE(unittest.TestCase):
    def test_against_a_direct_loop(self):
        rng = np.random.default_rng(1)
        k, n = 5, 1000
        E = 3.0 + rng.normal(0, 2, (k, 1)) + rng.normal(0, 5, (1, n)) + rng.normal(0, 1, (k, n))
        out = R.paired_se(E)
        d = [sum(E[s]) / n for s in range(k)]
        ebar = [sum(E[s][i] for s in range(k)) / k for i in range(n)]
        mean_d = sum(d) / k
        mean_e = sum(ebar) / n
        sd_d = math.sqrt(sum((x - mean_d) ** 2 for x in d) / (k - 1))
        sd_e = math.sqrt(sum((x - mean_e) ** 2 for x in ebar) / (n - 1))
        self.assertAlmostEqual(out["value"], float(E.mean()), places=12)
        self.assertAlmostEqual(out["value"], mean_d, places=12)
        self.assertAlmostEqual(out["se_seeds"], sd_d / math.sqrt(k), places=12)
        self.assertAlmostEqual(out["se_starts"], sd_e / math.sqrt(n), places=12)
        self.assertAlmostEqual(out["se"], math.sqrt(out["se_seeds"] ** 2 + out["se_starts"] ** 2), places=12)
        self.assertEqual(len(out["per_seed"]), k)
        self.assertEqual(len(out["se_within_seed"]), k)
        for s in range(k):
            self.assertAlmostEqual(out["se_within_seed"][s], float(np.std(E[s], ddof=1) / math.sqrt(n)), places=12)
        # the exact two-way estimate lies below the conservative sum, and by about the interaction term
        resid = E - E.mean(1, keepdims=True) - E.mean(0, keepdims=True) + E.mean()
        ms = float((resid ** 2).sum() / ((k - 1) * (n - 1)))
        self.assertAlmostEqual(out["se_two_way"] ** 2, out["se"] ** 2 - ms / (k * n), places=12)
        self.assertLessEqual(out["se_two_way"], out["se"])

    def test_shared_start_luck_is_counted_once(self):
        """The same bank in every seed: a difference that depends only on the start has no seed spread, and its
        start-to-start SE is that of one bank (it does not shrink with more seeds)."""
        rng = np.random.default_rng(2)
        b = rng.normal(0, 4, 1000)
        E = np.tile(b, (5, 1))
        out = R.paired_se(E)
        self.assertAlmostEqual(out["se_seeds"], 0.0, places=12)
        self.assertAlmostEqual(out["se_starts"], float(np.std(b, ddof=1) / math.sqrt(1000)), places=12)
        self.assertAlmostEqual(out["se"], out["se_starts"], places=12)

    def test_seed_spread_only(self):
        E = np.repeat(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), 1000, axis=1)
        out = R.paired_se(E)
        self.assertAlmostEqual(out["value"], 3.0)
        self.assertAlmostEqual(out["se_starts"], 0.0, places=12)
        self.assertAlmostEqual(out["se_seeds"], float(np.std([1, 2, 3, 4, 5], ddof=1) / math.sqrt(5)), places=12)

    def test_one_seed(self):
        E = np.random.default_rng(3).normal(0, 1, (1, 1000))
        out = R.paired_se(E)
        self.assertIsNone(out["se_seeds"])
        self.assertIsNone(out["se"])
        self.assertIsNone(out["se_two_way"])
        self.assertAlmostEqual(out["se_starts"], float(np.std(E[0], ddof=1) / math.sqrt(1000)), places=12)

    def test_identical_arms_give_zero(self):
        out = R.paired_se(np.zeros((5, 1000)))
        self.assertEqual((out["value"], out["se"], out["se_starts"], out["se_seeds"]), (0.0, 0.0, 0.0, 0.0))

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            R.paired_se(np.zeros(10))


class Tails(unittest.TestCase):
    def test_p10_and_worst10(self):
        J = -np.arange(1, 1001, dtype=float)
        np.random.default_rng(4).shuffle(J)
        t = R.tail_stats(J)
        self.assertEqual(t["worst10_count"], 100)
        self.assertAlmostEqual(t["worst10_mean"], -950.5)            # mean of -1000 .. -901
        self.assertAlmostEqual(t["p10"], -900.1)                     # numpy's linear interpolation
        self.assertAlmostEqual(t["mean"], -500.5)
        self.assertEqual(t["worst"], -1000.0)

    def test_small_bank(self):
        t = R.tail_stats([-3.0, -1.0, -2.0])
        self.assertEqual(t["worst10_count"], 1)
        self.assertEqual(t["worst10_mean"], -3.0)


class Costs(unittest.TestCase):
    def test_tracking_energy_late_peak(self):
        rng = np.random.default_rng(5)
        N, L = 7, 100
        states = rng.normal(0, 1, (N, L + 1, 13))
        states[:, 0, :10] *= 50.0                                     # a big start: the overall peak
        states[:, 37, :10] = 0.0
        states[:, 37, 0] = 9.0                                        # the late peak at step 37: |s| = 9
        states[:, 100, :10] = 1e3                                     # the final state is never scored
        commands = rng.uniform(-1, 1, (N, L, 4))
        tracking = np.array([sum(float(states[i, t, :10] @ states[i, t, :10]) for t in range(L)) for i in range(N)])
        energy = 0.3 * np.sum(commands ** 2, axis=(1, 2))
        J = -(tracking + energy)
        c = R.episode_costs(states, commands, J, 0.3, L, 10)
        np.testing.assert_allclose(c["tracking"], tracking, rtol=1e-13)
        np.testing.assert_allclose(c["energy"], energy, rtol=1e-13)
        self.assertLess(c["identity_error"], 1e-9)
        radius = np.sqrt(np.sum(states[:, :L, :10] ** 2, axis=2))
        np.testing.assert_allclose(c["late_peak"], radius[:, 10:].max(1))
        self.assertTrue(np.all(c["late_peak"] >= 9.0))
        self.assertTrue(np.all(c["late_peak"] < radius[:, 0]))       # the start is excluded
        self.assertTrue(np.all(c["late_peak"] < 1e3))                # so is the final, unscored state
        # the window starts at step 10 exactly
        st2 = np.zeros((1, L + 1, 13))
        st2[0, 9, 0], st2[0, 10, 0] = 5.0, 2.0
        self.assertEqual(float(R.episode_costs(st2, np.zeros((1, L, 4)), np.zeros(1), 0.3, L, 10)["late_peak"][0]),
                         2.0)


class Regret(unittest.TestCase):
    def setUp(self):
        self.seeds = [0, 1, 2, 3, 4]
        self.pts = synthetic_pts(self.seeds, np.random.default_rng(6))
        self.opt = {w: {sit: 100.0 + 1.0 + i for i, sit in enumerate(R.SITUATIONS)} for w in R.WORLDS}

    def test_regret_is_optimum_minus_points(self):
        reg = R.regret_tables(self.pts, self.opt, self.seeds)
        for w in R.WORLDS:
            for arm in R.ARMS[w]:
                for sit in R.SITUATIONS:
                    per = reg["per_world"][w][arm][sit]["per_seed"]
                    for i, s in enumerate(self.seeds):
                        self.assertAlmostEqual(per[i], self.opt[w][sit] - self.pts[w][s][arm][sit], places=12)
                    self.assertAlmostEqual(reg["per_world"][w][arm][sit]["mean"], float(np.mean(per)), places=12)
        tot = reg["total_strong_current"]["obca_k1"]["per_seed"]
        for i, s in enumerate(self.seeds):
            self.assertAlmostEqual(tot[i], sum(self.opt[w]["strong_current"] - self.pts[w][s]["obca_k1"]
                                               ["strong_current"] for w in R.WORLDS), places=12)
        self.assertNotIn("full_none", reg["total_strong_current"])    # stalling world only
        self.assertAlmostEqual(reg["references"]["stall"]["strong_current"]["reference_100_points"], 2.0)
        self.assertAlmostEqual(reg["references"]["total_strong_current"]["reference_100_points"], 4.0)

    def test_net_equals_the_total_regret_difference(self):
        reg = R.regret_tables(self.pts, self.opt, self.seeds)
        self.assertLess(R.net_identity_error(self.pts, reg, self.seeds), 1e-9)
        for arm in R.ALL_IDS:
            self.assertAlmostEqual(S.net(self.pts, arm, self.seeds)["net"],
                                   reg["total_strong_current"]["none"]["mean"] -
                                   reg["total_strong_current"][arm]["mean"], places=9)

    def test_missing_value_gives_none(self):
        self.pts["no_stall"][3]["even_pull"] = None
        reg = R.regret_tables(self.pts, self.opt, self.seeds)
        self.assertIsNone(reg["per_world"]["no_stall"]["even_pull"]["strong_current"]["per_seed"][3])
        self.assertIsNone(reg["per_world"]["no_stall"]["even_pull"]["strong_current"]["mean"])
        self.assertIsNone(reg["total_strong_current"]["even_pull"]["mean"])


class Contrasts(unittest.TestCase):
    def setUp(self):
        self.seeds = [0, 1, 2, 3, 4]
        rng = np.random.default_rng(7)
        self.per_start = {w: {s: {arm: {sit: rng.normal(50, 30, 1000) for sit in R.SITUATIONS}
                                  for arm in R.ARMS[w]} for s in self.seeds} for w in R.WORLDS}
        self.pts = {w: {s: {arm: {sit: float(self.per_start[w][s][arm][sit].mean()) for sit in R.SITUATIONS}
                            for arm in R.ARMS[w]} for s in self.seeds} for w in R.WORLDS}

    def test_combine_cancels_none(self):
        t = R.combine((1, R.net_terms("obca_k1")), (-1, R.net_terms("even_pull")))
        self.assertEqual(t, {("stall", "obca_k1"): 1.0, ("no_stall", "obca_k1"): 1.0,
                             ("stall", "even_pull"): -1.0, ("no_stall", "even_pull"): -1.0})

    def test_every_contrast_matches_step2lib(self):
        specs = R.contrast_specs()
        names = [c["name"] for c in specs]
        self.assertEqual(len(names), len(set(names)))
        for want in ("NET(obca_k1)", "NET(obca_k1) - NET(standin)", "recorded conditions: obca_k1 - none",
                     "obca_k1 - even_pull (matched)", "A1: NET(conformal_k1) - NET(obca_k1)",
                     "dose: NET(obca_k2) - NET(obca_k1)", "obca_k1 - random_2 (matched)", "obca_k1 - clone",
                     "obca_k1 - standin (matched)", "NET(clone), recorded"):
            self.assertIn(want, names)
        self.assertEqual(sum(1 for c in specs if c["group"].startswith("G1")), len(S.WORLD_BLIND))
        rule = dict(g1_margin=10.0, g2_floor=-3.0, g3_margin=15.0, attribution_margin=5.0, m1_margin=15.0,
                    m1_positive_min=4, m2_margin=15.0, m3_floor=-3.0, m3_clone_margin=25.0, m4_floor=-5.0)
        g = S.go_no_go(self.pts, self.seeds, rule)
        att = S.attribution(self.pts, self.seeds, rule)
        marks = S.analogues(self.pts, self.seeds, rule)
        fake_results = S.jsonable(dict(
            go_no_go=g, attribution=att, v6_marks_restated=marks,
            net_strong_current={a: S.net(self.pts, a, self.seeds) for a in R.ALL_IDS},
            net_recorded_conditions_information_only={a: S.net(self.pts, a, self.seeds, "recorded")
                                                      for a in R.ALL_IDS}))
        for c in specs:
            E = R.contrast_from_starts(self.per_start, c, self.seeds)
            st = R.paired_se(E)
            v_means, _ = R.contrast_value_from_means(self.pts, c, self.seeds)
            v_res = R.step2_value(fake_results, c["ref"])
            self.assertIsNotNone(v_res, c["name"])
            self.assertAlmostEqual(st["value"], v_means, places=9, msg=c["name"])
            self.assertAlmostEqual(v_means, v_res, places=9, msg=c["name"])

    def test_missing_arm_gives_none(self):
        spec = [c for c in R.contrast_specs() if c["name"] == "NET(even_pull)"][0]
        del self.per_start["no_stall"][2]["even_pull"]
        self.assertIsNone(R.contrast_from_starts(self.per_start, spec, self.seeds))
        self.pts["no_stall"][2]["even_pull"] = None
        self.assertIsNone(R.contrast_value_from_means(self.pts, spec, self.seeds)[0])
        self.assertIsNone(R.step2_value(None, spec["ref"]))
        self.assertIsNone(R.step2_value({}, spec["ref"]))


class Guards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.repo = self.t / "repo"
        self.off_root = self.repo / "runs" / "step2"
        self.off_out = self.repo / "runs" / "step2_report"
        self.off_root.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def resolve(self, root, out, allow):
        return R.resolve_paths(root, out, allow, official_root=self.off_root, official_out=self.off_out,
                               repo=self.repo)

    def test_official_refuses_before_results(self):
        with self.assertRaisesRegex(SystemExit, "results.json does not exist"):
            self.resolve(None, None, False)
        with self.assertRaisesRegex(SystemExit, "scratch runs only"):
            self.resolve(None, None, True)
        with self.assertRaisesRegex(SystemExit, "--out is for scratch"):
            self.resolve(None, str(self.t / "x"), False)
        (self.off_root / "results.json").write_text("{}")
        official, root, out = self.resolve(None, None, False)
        self.assertTrue(official)
        self.assertEqual((root, out), (self.off_root, self.off_out))

    def test_scratch_paths(self):
        smoke = self.t / "smoke"
        smoke.mkdir()
        with self.assertRaisesRegex(SystemExit, "outside"):
            self.resolve(str(self.off_root), str(self.t / "o"), True)     # the official folder, in scratch mode
        with self.assertRaisesRegex(SystemExit, "needs --out"):
            self.resolve(str(smoke), None, True)
        with self.assertRaisesRegex(SystemExit, "outside"):
            self.resolve(str(smoke), str(self.repo / "o"), True)
        with self.assertRaisesRegex(SystemExit, "allow-no-results"):
            self.resolve(str(smoke), str(self.t / "o"), False)
        official, root, out = self.resolve(str(smoke), str(self.t / "o"), True)
        self.assertFalse(official)
        (smoke / "results.json").write_text("{}")
        self.resolve(str(smoke), str(self.t / "o"), False)

    def test_write_once(self):
        smoke = self.t / "smoke"
        smoke.mkdir()
        out = self.t / "o"
        R.write_once(out / "report.json", "{}")
        with self.assertRaises(FileExistsError):
            R.write_once(out / "report.json", "{}")
        with self.assertRaisesRegex(SystemExit, "written once"):
            self.resolve(str(smoke), str(out), True)

    def test_real_official_paths(self):
        """Path arithmetic only: these tests never touch step 2's official output folder."""
        self.assertIn(R.REPO, R.OFFICIAL_ROOT.parents)
        self.assertNotIn(R.OFFICIAL_ROOT, R.OFFICIAL_OUT.parents)
        self.assertNotEqual(R.OFFICIAL_OUT, R.OFFICIAL_ROOT)
        self.assertEqual(R.OFFICIAL_OUT.parent, R.OFFICIAL_ROOT.parent)
        self.assertEqual(R.OFFICIAL_ROOT, S.OFFICIAL_OUT)


class Words(unittest.TestCase):
    BANNED = (r"\brisk", r"peak_distance")

    def texts(self):
        yield from R.NOTES
        yield from R.LABELS.values()
        for k, v in R.WORDS:
            yield k
            yield v
        yield R.LABEL

    def test_banned_words(self):
        for t in self.texts():
            for b in self.BANNED:
                self.assertIsNone(re.search(b, t, re.I), f"{b!r} in {t!r}")

    def test_labels_and_notes(self):
        self.assertEqual(set(R.LABELS), set(R.SITUATIONS))
        self.assertIn("entirely outside the training range", R.LABELS["strong_current"])
        self.assertIn("entirely outside the training range", R.LABELS["big_waves"])
        self.assertIn("10-30 steps", R.LABELS["big_kicks"])
        joined = " ".join(R.NOTES)
        for want in ("stall-aware reference", "1,000 starts", "discounted value units", "design effect", "1.0-1.4",
                     "support rule alone", "obca_forced", "one row per episode", "gate's critic"):
            self.assertIn(want, joined)

    def test_stated_optimum(self):
        self.assertEqual(R.STATED_OPTIMUM["stall"]["strong_current"], 109.0)
        self.assertEqual(R.STATED_OPTIMUM["no_stall"]["big_kicks"], 103.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
