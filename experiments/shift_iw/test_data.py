"""CPU tests of data.py, coverage.py and classifier.py. No learning; BCA's calibration (wbcp.calibrate) runs only on
small synthetic arrays here. Every number is a test fixture, not a finding.

A tiny build (24 + 8 + 8 training-law episodes, 8 per test log and sample) keeps it fast; the rules are the same as
for the full build. Checks: deterministic rebuild (same bytes); manifests refuse a tampered file; every row replays
through env.step bit for bit; the gap holds no kept training or bank rows and touches no held-out or test row; the
bank takes 5 rows of every calibration episode; layout (splits, timeouts, next observations); truth columns; the
diagnostic flags are guarded; classifier samples hold observations and commands only; start banks are shared and
differ only in c; learners.prepare takes the log; coverage, refusals, weighting and the classifier on synthetic data.
JAX_PLATFORMS=cpu python -m unittest experiments.shift_iw.test_data
"""

import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401

from experiments.shift import shift as SH
from experiments.shift import thruster_chain as TC
from experiments.shift_iw import classifier as CL
from experiments.shift_iw import coverage as CV
from experiments.shift_iw import data as D
from experiments.shift_iw import world as W

TINY = D.Sizes(train=24, calibration=8, held_out=8, test=8, sample=8, grid_points=5, per_point=20, truth_rows=100,
               noise_map_horizon=50, start_bank=20)
SEED = 3
# Version 4's data at TINY on seed 3, built by version 4's own code (a frozen copy of world.py and data.py taken before
# any version 5 edit; iw/scout/v5_expert_equals_v4.py has the full comparison at the small size on seed 0): per file,
# the SHA-256 of the lines "<array>:<array SHA-256>" over every array version 4 wrote, sorted by name.
V4_TINY_SEED3 = {
    "samples.hdf5": "a4bb2dddadf73870568b8d6193d5e2a7f1f72f2ca29e9a2490c4e366dae1f6f3",
    "test_mild.hdf5": "ac0d9179646f867be0bc96c3e74340a412a9bd74c68eb194a44066ceac972226",
    "test_strong.hdf5": "5ff64d4cae46f9f0b30c1e7cb50aa54840d25c2dbcc499a4492652ccc1a9b8e0",
    "train_law.hdf5": "07bf72873eb130f851feefd167d81a352e4e86068230e30a7998f94586223352",
    "truth.hdf5": "17ce25ba274ce4c09f063e727829f49f23e3666b0e8d650f3544b1817159486a",
}
V4_TINY_SEED3_BALL = dict(rho=0.2724428031993812,
                          record_sha256="4eaa3a242470180c469009f6d7ef020ce12095120364319d6c2d4f3dd96b42cb",
                          Z=0.8450310908535108)
V4_ARRAY_COUNTS = {"samples.hdf5": 12, "test_mild.hdf5": 18, "test_strong.hdf5": 18, "train_law.hdf5": 18,
                   "truth.hdf5": 2}


def combined_digest(manifest, keys):
    lines = "\n".join(f"{k}:{manifest['arrays'][k]['sha256']}" for k in sorted(keys))
    return hashlib.sha256(lines.encode()).hexdigest()


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="shift_iw_test_"))
        quiet = lambda m: None
        D.build_start_banks(D.banks_dir(cls.tmp), TINY.start_bank, log=quiet)
        cls.root = D.data_dir(cls.tmp, "expert")
        cls.world, _ = D.build_seed(SEED, cls.root, TINY, "expert", log=quiet)
        cls.data = D.load_seed(SEED, cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_rebuild_gives_the_same_bytes(self):
        other = Path(tempfile.mkdtemp(prefix="shift_iw_test_"))
        try:
            world, _ = D.build_seed(SEED, D.data_dir(other, "expert"), TINY, "expert", log=lambda m: None)
            self.assertEqual(world["files"], self.world["files"])
            for name in list(world["files"]) + ["world.json"]:
                self.assertEqual(D.sha256_file(D.data_dir(other, "expert") / f"seed{SEED}" / name),
                                 D.sha256_file(self.root / f"seed{SEED}" / name), name)
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_expert_column_is_version_4_bit_for_bit(self):
        """Every array version 4 wrote, in every file, has version 4's SHA-256 (hashes recorded from version 4's own
        code); version 5 adds only the all-False infos/random flag. The ball and Z are version 4's."""
        for name, digest in V4_TINY_SEED3.items():
            m = json.loads((self.root / f"seed{SEED}" / f"{name}.manifest.json").read_text())
            keys = [k for k in m["arrays"] if k != "infos/random"]
            self.assertEqual(len(keys), V4_ARRAY_COUNTS[name], name)
            self.assertEqual(combined_digest(m, keys), digest, name)
            self.assertLessEqual(set(m["arrays"]) - set(keys), {"infos/random"})
        self.assertFalse(self.data.train_log.diagnostic("infos/random").any())
        self.assertEqual(self.world["gap"]["rho"], V4_TINY_SEED3_BALL["rho"])
        self.assertEqual(self.world["gap"]["record_sha256"], V4_TINY_SEED3_BALL["record_sha256"])
        self.assertEqual(self.world["quadrature"]["Z"]["value"], V4_TINY_SEED3_BALL["Z"])
        self.assertEqual(self.world["quadrature"], self.world["quadrature_expert"])

    def test_random_flag_is_a_diagnostic(self):
        log = self.data.train_log
        with self.assertRaises(KeyError):
            log["infos/random"]
        self.assertNotIn("infos/random", list(log))
        self.assertEqual(log.diagnostic("infos/random").dtype, bool)

    def test_manifest_refuses_a_tampered_file(self):
        import h5py

        src = self.root / f"seed{SEED}" / "test_mild.hdf5"
        dst = self.tmp / "tampered.hdf5"
        shutil.copy(src, dst)
        shutil.copy(str(src) + ".manifest.json", str(dst) + ".manifest.json")
        D.read_hdf5(dst)
        with h5py.File(dst, "r+") as f:
            f["rewards"][0] = f["rewards"][0] + 1
        with self.assertRaises(ValueError):
            D.read_hdf5(dst)

    def test_every_row_replays_through_env_step(self):
        for name, r in self.world["replay"].items():
            self.assertEqual(r["mismatches"], 0, name)
        self.assertEqual(self.world["replay"]["train_law"]["rows"], self.data.train_log.n_rows)
        again = D.replay(self.data.tests["strong"])  # from the files as read back
        self.assertEqual(again["mismatches"], 0)
        self.assertEqual(again["rows"], self.data.tests["strong"].n_rows)

    def test_gap_holds_no_training_or_bank_rows(self):
        log, gap = self.data.train_log, self.data.gap
        removal = self.data.removal("gap")
        kept, bank = removal.kept_train_rows(log), removal.bank_rows(log)
        self.assertGreater(len(kept), 0)
        self.assertFalse(gap.contains(log["actions"][kept]).any())
        self.assertFalse(gap.contains(log["actions"][bank]).any())
        self.assertEqual(SH.leak_count(kept, log, gap), 0)
        self.assertFalse(removal.removed[log.split_rows("evaluation")].any())
        self.assertFalse(removal.removed[log.split_rows("calibration")].any())
        np.testing.assert_array_equal(log["truth/in_ball"], gap.contains(log["actions"]))
        for t in self.data.tests.values():
            np.testing.assert_array_equal(t["truth/in_ball"], gap.contains(t["actions"]))
        full = self.data.removal("none")
        self.assertEqual(len(full.kept_train_rows(log)), len(log.split_rows("train")))
        self.assertEqual(self.world["removal"]["leaks_training"], 0)
        self.assertEqual(self.world["removal"]["leaks_bank"], 0)

    def test_calibration_bank(self):
        log = self.data.train_log
        bank = log.bank_rows
        self.assertEqual(len(bank), TINY.calibration * TINY.bank_rows_per_episode)
        self.assertTrue(np.all(log["infos/split"][bank] == 1))
        _, per = np.unique(log["infos/episode"][bank], return_counts=True)
        self.assertEqual(len(per), TINY.calibration)
        self.assertTrue(np.all(per == TINY.bank_rows_per_episode))

    def test_layout(self):
        log = self.data.train_log
        L = W.LENGTH
        self.assertEqual(log.n_rows, sum(TINY.split_sizes) * L)
        self.assertEqual(int((log["infos/split"] == 0).sum()), TINY.train * L)
        self.assertFalse(log["terminals"].any())
        np.testing.assert_array_equal(log["timeouts"], log["infos/t"] == L - 1)
        inside = log["infos/t"][:-1] < L - 1
        np.testing.assert_array_equal(log["next_observations"][:-1][inside], log["observations"][1:][inside])
        np.testing.assert_array_equal(log["observations"], log["infos/state"].astype(np.float32))
        np.testing.assert_array_equal(log["rewards"], log["infos/reward64"].astype(np.float32))
        self.assertEqual(log["actions"].dtype, np.float32)
        for t in self.data.tests.values():
            self.assertTrue(np.all(t["infos/split"] == 2))
            self.assertEqual(len(t.split_rows("train")), 0)

    def test_truth_weights(self):
        for log in [self.data.train_log] + list(self.data.tests.values()):
            for tilt in W.TILT_NAMES:
                np.testing.assert_array_equal(log[f"truth/weight_{tilt}"], W.ratio(log.current, tilt))
            self.assertTrue(np.all(log["truth/weight_none"] == 1.0))

    def test_test_logs_come_from_their_laws(self):
        for tilt, log in self.data.tests.items():
            rng = np.random.default_rng([W.MASTER_SEED, SEED, D.TAGS[f"test_{tilt}"]])
            starts = W.draw_starts(TINY.test, rng, tilt)
            first = log["infos/state"][log["infos/t"] == 0]
            np.testing.assert_array_equal(first, starts)
        rng = np.random.default_rng([W.MASTER_SEED, SEED, D.TAGS["train_law"]])
        first = self.data.train_log["infos/state"][self.data.train_log["infos/t"] == 0]
        np.testing.assert_array_equal(first, W.draw_starts(sum(TINY.split_sizes), rng, "train"))

    def test_diagnostic_flags_are_guarded(self):
        log = self.data.train_log
        with self.assertRaises(KeyError):
            log["infos/hold"]
        self.assertNotIn("infos/hold", list(log))
        before = log.diagnostic_reads
        log.diagnostic("infos/hold")
        self.assertEqual(log.diagnostic_reads, before + 1)
        # the removal rule never reads it
        reads = log.diagnostic_reads
        SH.define_gap(log)
        SH.make_removal(log, "gap", self.data.gap)
        self.assertEqual(log.diagnostic_reads, reads)

    def test_classifier_samples_hold_observations_and_commands_only(self):
        self.assertEqual(set(self.data.samples), set(D.SAMPLE_LAWS))
        for name, s in self.data.samples.items():
            self.assertEqual(set(s), set(D.SAMPLE_FIELDS))
            self.assertEqual(s["observations"].shape, (TINY.sample * W.LENGTH, TC.DX))
            self.assertEqual(s["actions"].dtype, np.float32)
        self.assertFalse(np.array_equal(self.data.samples["train"]["observations"],
                                        self.data.samples["train2"]["observations"]))

    def test_truth_rows(self):
        tr, log = self.data.truth, self.data.train_log
        rows = tr["rows"]
        self.assertEqual(len(rows), TINY.truth_rows)
        self.assertTrue(np.all(np.diff(rows) > 0))
        self.assertTrue(np.all(log["infos/split"][rows] == 2))
        self.assertFalse(log["truth/in_ball"][rows].any())
        self.assertTrue(np.all(np.isfinite(tr["noise_map"]) & (tr["noise_map"] >= 0)))
        self.assertEqual(set(tr), {"rows", "noise_map"})      # only what the spec needs

    def test_world_record(self):
        w = self.world
        q = w["quadrature"]
        self.assertEqual(q["test_mass_gapped"]["none"]["value"], 1.0)
        self.assertEqual(self.data.test_mass("strong", gapped=False), W.full_test_mass("strong"))
        self.assertEqual(self.data.test_mass("mild"), q["test_mass_gapped"]["mild"]["value"])
        self.assertEqual(set(w["D1"]["checks"]), {"holds_removed", "Z", "in_ball_mild", "in_ball_strong"})
        log, rows = self.data.test("none")
        np.testing.assert_array_equal(rows, log.split_rows("evaluation"))
        w = json.loads((self.root / f"seed{SEED}" / "world.json").read_text())
        self.assertEqual((w["seed"], w["quality"], w["ball_source"]), (SEED, "expert", "expert, same seed"))
        self.assertEqual(self.data.quality, "expert")
        self.assertEqual(w["test_in_ball_steps"]["strong"], int(self.data.tests["strong"]["truth/in_ball"].sum()))

    def test_d1_rule(self):
        good = dict(holds_removed_train=dict(value=0.995), Z=dict(value=0.84),
                    in_ball_share=dict(mild=dict(value=0.25), strong=dict(value=0.36)))
        self.assertTrue(D.d1(good)["passes"])
        bad = json.loads(json.dumps(good))
        bad["in_ball_share"]["strong"]["value"] = 0.40
        self.assertFalse(D.d1(bad)["passes"])

    def test_start_banks(self):
        banks, record = D.load_start_banks(D.banks_dir(self.tmp))
        again = D.start_banks(TINY.start_bank)
        other = [j for j in range(TC.DX) if j != TC.CURRENT]
        for t in W.TILT_NAMES:
            np.testing.assert_array_equal(banks[t], again[t])
            np.testing.assert_array_equal(banks[t][:, other], banks["none"][:, other])
            a = record["anchors"][t]
            self.assertLess(a["anchor"]["random"], a["anchor"]["top"])
            rows = a["controllers"]
            self.assertEqual((rows["random"]["points"], rows["stall_aware"]["points"]), (0.0, 100.0))   # exactly
            self.assertEqual(set(rows), {"stall_aware", "zero_command"} | set(W.QUALITIES)
                             | {f"{q} without the hold" for q in W.QUALITIES})
        self.assertLess(banks["none"][:, TC.CURRENT].mean(), banks["strong"][:, TC.CURRENT].mean())

    def test_learners_prepare_takes_the_log(self):
        p = D.prepare(self.data, "gap")
        removal = self.data.removal("gap")
        np.testing.assert_array_equal(p.rows, removal.kept_train_rows(self.data.train_log))
        np.testing.assert_array_equal(p.bank_rows, removal.bank_rows(self.data.train_log))
        np.testing.assert_array_equal(p.region, W.HoldRecorder().hard_water(p.states))
        full = D.prepare(self.data, "none")
        self.assertEqual(full.n, len(self.data.train_log.split_rows("train")))
        np.testing.assert_array_equal(full.obs_mean, p.obs_mean)  # one normalisation rule for both host runs
        kept_obs = self.data.train_log["observations"][p.rows]
        np.testing.assert_allclose(p.obs_mean, kept_obs.mean(0), rtol=1e-5, atol=1e-6)


class QualityBuildTest(unittest.TestCase):
    """Every dataset quality built at the tiny size on one seed (spec version 5)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="shift_iw_quality_"))
        cls.worlds, cls.data = {}, {}
        for q in W.QUALITIES:
            cls.worlds[q], _ = D.build_seed(SEED, D.data_dir(cls.tmp, q), TINY, q, log=lambda m: None)
            cls.data[q] = D.load_seed(SEED, D.data_dir(cls.tmp, q))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_ball_and_the_design_check_are_shared(self):
        """One ball per seed: the expert's (shift.define_gap on version 4's training-law log), and one D1 record,
        identical across the five qualities."""
        gap, _ = SH.define_gap(D.training_law_log(SEED, TINY, "expert"))
        for q, w in self.worlds.items():
            self.assertEqual(w["gap"], gap.as_dict(), q)
            self.assertEqual(w["D1"], self.worlds["expert"]["D1"], q)
            self.assertEqual(w["quadrature_expert"], self.worlds["expert"]["quadrature"], q)
            self.assertEqual((w["quality"], w["ball_source"]), (q, "expert, same seed"))
            self.assertEqual((list(self.data[q].gap.centre), self.data[q].gap.rho), (list(gap.centre), gap.rho))
        self.assertNotEqual(self.worlds["medium"]["quadrature"], self.worlds["expert"]["quadrature"])
        self.assertIsNone(self.worlds["random"]["quadrature"]["holds_removed_train"]["value"])   # no holds at all

    def test_every_quality_replays_through_env_step(self):
        for q, w in self.worlds.items():
            for name, r in w["replay"].items():
                self.assertEqual(r["mismatches"], 0, (q, name))
                self.assertGreater(r["rows"], 0)

    def test_random_shares_follow_the_plan(self):
        for q, d in self.data.items():
            for log in [d.train_log] + list(d.tests.values()):
                self.assertEqual(log.quality, q)
                pick = np.asarray(log.meta["picks"])
                np.testing.assert_array_equal(np.asarray(log.meta["random_shares"]), W.random_share(q, pick))
                rand = log.diagnostic("infos/random")
                hold = log.diagnostic("infos/hold")
                self.assertFalse(np.any(rand & hold))
                eps = np.repeat(np.asarray(log.meta["random_shares"]), W.LENGTH)
                self.assertFalse(np.any(rand[eps == 0.0]))
                self.assertTrue(np.all(rand[eps == 1.0]))
        r = self.worlds["random"]["removal"]
        self.assertEqual((r["random_share_training"], r["hold_share_training"], r["holds_removed_training"]),
                         (1.0, 0.0, None))
        self.assertEqual(self.worlds["expert"]["removal"]["random_share_training"], 0.0)
        for q, w in self.worlds.items():   # a kept row at exactly 80% that is not a hold: none
            self.assertEqual(w["removal"]["kept_at_exactly_80_not_hold"], 0, q)

    def test_the_same_starts_and_water_in_every_quality(self):
        x = {q: d.train_log["infos/state"][d.train_log["infos/t"] == 0] for q, d in self.data.items()}
        for q in W.QUALITIES:
            np.testing.assert_array_equal(x[q], x["expert"])
        c = {q: d.samples["strong"]["observations"][::W.LENGTH, TC.CURRENT] for q, d in self.data.items()}
        for q in W.QUALITIES:
            np.testing.assert_array_equal(c[q], c["expert"])

    def test_an_unknown_quality_is_refused(self):
        with self.assertRaises(ValueError):
            D.build_seed(SEED, self.tmp / "x", TINY, "novice", log=lambda m: None)
        with self.assertRaises(ValueError):
            D.data_dir(self.tmp, "novice")


class AnchorTest(unittest.TestCase):
    def test_the_random_anchor_at_1000_starts(self):
        """The random quality on stream 1002 at the full start banks: -800.29 / -1061.42 / -1351.41 (the design's
        anchor, iw_quality/DESIGN.md section 3; exact simulation)."""
        from experiments.shift import truth as TR

        banks = D.start_banks(1000)
        got = [round(float(TR.rollout(D.quality_on_bank("random", 1000), banks[t], W.PHYSICS).J.mean()), 2)
               for t in W.TILT_NAMES]
        self.assertEqual(got, [-800.29, -1061.42, -1351.41])

    def test_balanced_picks_on_the_start_banks(self):
        *_, pick = D.bank_streams(1000)
        self.assertTrue(np.all(np.bincount(pick, minlength=500) == 2))
        me = W.random_share("medium-expert", pick)
        self.assertEqual(int((me > 0).sum()), 500)


class CoverageTest(unittest.TestCase):
    """Synthetic arrays only (no ThrusterChain data): scores whose spread grows with c, a bank from the training law
    and test steps from the strong law."""

    @staticmethod
    def synthetic(n, law, seed, sd=lambda c: 0.3 + 2.0 * c):
        rng = np.random.default_rng([55, seed])
        c = W.inverse_cdf(rng.uniform(size=n), law)
        return c, np.abs(rng.standard_normal(n) * sd(c))

    def test_unit_weights_reproduce_the_unweighted_threshold(self):
        _, s = self.synthetic(843, "train", 1)
        ok, a, b = CV.unit_check(s, [1, 2, 3])
        self.assertTrue(ok)
        self.assertEqual(a.threshold, b.threshold)

    def test_the_removal_constant_cancels(self):
        c, s = self.synthetic(843, "train", 2)
        w = W.ratio(c, "strong")
        a = CV.calibrate(s, [4], w, 2.5)
        b = CV.calibrate(s, [4], 0.8433 * w, 0.8433 * 2.5)
        self.assertEqual(a.threshold, b.threshold)

    def test_exact_weights_restore_coverage_under_the_shift(self):
        c, s = self.synthetic(843, "train", 3)
        ct, st = self.synthetic(40000, "strong", 4)
        arms = CV.arm_thresholds(s, [5], {"unweighted": None,
                                          "exact": (W.ratio(c, "strong"), W.full_test_mass("strong"))})
        cov = {k: float(np.mean(st <= v.threshold)) for k, v in arms.items()}
        self.assertLess(cov["unweighted"], 0.88)
        self.assertGreater(cov["exact"], 0.89)
        self.assertGreater(arms["exact"].threshold, arms["unweighted"].threshold)
        # the forecast from no-tilt steps: the w(c)-weighted mean of their covered indicators
        c0, s0 = self.synthetic(40000, "train", 6)
        fc = CV.forecast(c0, s0 <= arms["unweighted"].threshold, "strong")
        self.assertLess(abs(fc - cov["unweighted"]), 0.01)

    def test_outcomes_and_refusals(self):
        y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        q = np.array([1.5, 0.0, 1.0, 1.0, 1.2])
        U = np.array([0.5, 0.5, np.inf, 0.1, 0.3])
        refused = CV.refusals(np.array([False, False, False, True, False]), "exact")
        o = CV.outcomes(y, q, U, refused)
        self.assertEqual(o.tolist(), [1, 0, -1, -1, 1])  # a tie is covered; infinite width and the ball refuse
        self.assertFalse(CV.refusals(np.array([True, True]), "unweighted").any())
        s = CV.summarize(o, np.array([0, 0, 1, 1, 2]), boot=200)
        self.assertEqual(s["steps"], 5)
        self.assertAlmostEqual(s["coverage"], 2 / 5)
        self.assertAlmostEqual(s["refused_share"], 2 / 5)
        self.assertAlmostEqual(s["certified_coverage"], 2 / 3)
        g = CV.groups(o, np.arange(5), np.array([False, False, False, True, False]), boot=100)
        self.assertEqual(g["inside"]["steps"], 1)
        self.assertEqual(g["inside"]["refused_share"], 1.0)

    def test_width_rule(self):
        U = CV.width(np.float32(2.0), 0.5, np.array([1e-9, 1.0, 3.0], np.float32))
        self.assertEqual(U.dtype, np.float32)
        np.testing.assert_array_equal(U, np.float32(2.0) * (np.array([1e-6, 1.0, 3.0], np.float32) * np.float32(0.5)))
        self.assertTrue(np.all(np.isinf(CV.width(np.inf, 0.5, np.ones(3, np.float32)))))

    def test_bootstrap_and_seed_intervals(self):
        v = (np.arange(1000) % 7 < 6).astype(float)
        ep = np.repeat(np.arange(100), 10)
        lo, hi = CV.episode_bootstrap(v, ep, boot=500)
        self.assertLess(lo, v.mean())
        self.assertGreater(hi, v.mean())
        self.assertEqual((lo, hi), CV.episode_bootstrap(v, ep, boot=500))
        r = CV.across_seeds([1, 2, 3, 4, 5])
        self.assertAlmostEqual(r["mean"], 3.0)
        self.assertAlmostEqual(r["interval"][0], 3 - 2.7764451 * math.sqrt(2.5 / 5), places=5)
        self.assertAlmostEqual(r["lower_one_sided"], 3 - 2.1318468 * math.sqrt(2.5 / 5), places=5)

    def test_within_decile_correlation_ignores_the_current(self):
        rng = np.random.default_rng(8)
        c = rng.uniform(0.1, 1.1, 20000)
        err = c + 0.2 * rng.standard_normal(20000)
        width_c = c + 0.2 * rng.standard_normal(20000)        # tracks c only
        width_e = err + 0.01 * rng.standard_normal(20000)     # tracks the error itself
        a, b = CV.decile_spearman(width_c, err, c), CV.decile_spearman(width_e, err, c)
        self.assertGreater(a["pooled"], 0.5)
        self.assertLess(abs(a["within_deciles"]), 0.1)
        self.assertGreater(b["within_deciles"], 0.5)

    def test_dose_order(self):
        U = np.random.default_rng(9).gamma(2.0, size=500)
        dose = 1 + 0.5 * U / (U + 1.0)
        m = CV.dose_order(np.append(U, np.inf), np.append(dose, 1.0), unit=1.0)
        self.assertEqual((m["rows"], m["inversions"], m["max_drop"]), (500, 0, 0.0))
        self.assertLess(m["max_closed_form_gap"], 1e-12)
        bad = dose.copy()
        bad[np.argmax(U)] = 1.0
        m = CV.dose_order(U, bad)
        self.assertGreater(m["inversions"], 0)
        self.assertGreater(m["max_drop"], 0.4)

    def test_bcas_float32_dose_falls_by_rounding_only(self):
        """Mark M3's tolerance on 42,500 random widths (about one seed's kept training rows), with BCA's own dose
        (calibration/dose.level_critic_dose under jax.jit, as the readout runs it) on synthetic widths: a strict
        'never falls' rule fails on float32 rounding, the 1e-6 tolerance holds, and the dose matches its closed
        form within 1e-6."""
        import jax
        import jax.numpy as jnp

        from calibration.dose import level_critic_dose

        spec = json.loads(D.SPEC_JSON.read_text())
        tol = next(m for m in spec["pass_marks"] if m["id"] == "M3")["thresholds"]["max_dose_drop"]
        f = jax.jit(lambda w, u: level_critic_dose(w, jnp.ones_like(w, dtype=bool), u, 0.5, jnp.asarray(True)).dose)
        rng = np.random.default_rng([W.MASTER_SEED, 9090])
        strict_failures = 0
        for _ in range(5):
            unit, R = np.float32(rng.uniform(0.05, 3.0)), np.float32(rng.uniform(0.8, 3.0))
            U = CV.width(R, unit, np.float32(np.exp(rng.normal(0.0, 0.7, 42_500))))
            m = CV.dose_order(U, np.asarray(f(jnp.asarray(U), jnp.asarray(unit)), np.float64), unit=unit)
            self.assertEqual(m["rows"], 42_500)
            self.assertLessEqual(m["max_drop"], tol)
            self.assertLessEqual(m["max_closed_form_gap"], 1e-6)
            strict_failures += m["inversions"] > 0
        self.assertGreater(strict_failures, 0)   # why M3 has a tolerance (seeded, so deterministic)


class PerStepTest(unittest.TestCase):
    """Per-step thresholds (spec version 4) on synthetic arrays only: R(t) tabulated on a grid of test masses, each
    step at the smallest grid mass at least its own weight."""

    synthetic = staticmethod(CoverageTest.synthetic)

    def test_mass_grid(self):
        g = CV.mass_grid([0.135, 1.7, 7.39], 1.01)
        self.assertIn(1.0, g.tolist())                                   # exactly 1 when the masses reach it
        self.assertLessEqual(g[0], 0.135)
        self.assertGreaterEqual(g[-1], 7.39)
        np.testing.assert_allclose(g[1:] / g[:-1], 1.01, rtol=1e-12)
        self.assertEqual(CV.mass_grid(np.ones(50), 1.01).tolist(), [1.0])   # no tilt: the grid is [1.0]
        for bad in ([0.0, 1.0], [np.inf], []):
            with self.assertRaises(ValueError):
                CV.mass_grid(bad, 1.01)
        with self.assertRaises(ValueError):
            CV.mass_grid([1.0], 1.0)

    def test_threshold_is_monotone_in_the_test_mass(self):
        """With the same random stream at every test mass, R(t) never falls (the posterior draws are shared); it
        reaches +inf when the test atom is too heavy to certify."""
        c, s = self.synthetic(300, "train", 21)
        w = W.ratio(c, "strong")
        grid = CV.mass_grid([0.05, 5000.0], 1.05)
        tab = CV.threshold_table(s, [21], w, grid)
        self.assertTrue(tab.monotone)
        self.assertFalse(tab.running_max_applied)
        np.testing.assert_array_equal(tab.thresholds, tab.raw)
        self.assertTrue(np.all(tab.raw[1:] >= tab.raw[:-1]))
        self.assertTrue(np.isfinite(tab.raw[0]) and np.isinf(tab.raw[-1]))
        for t in grid[::17]:                                            # each grid point is calibrate itself
            self.assertEqual(CV.calibrate(s, [21], w, float(t)).threshold, tab.raw[np.searchsorted(grid, t)])
        back = CV.table_from_summary(json.loads(json.dumps(tab.summary(1.0))))
        np.testing.assert_array_equal(back.thresholds, tab.thresholds)
        np.testing.assert_array_equal(back.grid, tab.grid)

    def test_running_maximum_enforces_monotone_thresholds(self):
        from types import SimpleNamespace
        from unittest import mock

        fake = iter([1.0, 3.0, 2.0, 4.0])
        with mock.patch.object(CV, "calibrate", lambda *a, **k: SimpleNamespace(threshold=next(fake), lambda_hat=0.5,
                                                                                n_eff=10.0)):
            tab = CV.threshold_table(np.ones(5), [1], np.ones(5), [1.0, 2.0, 3.0, 4.0])
        self.assertFalse(tab.monotone)
        self.assertTrue(tab.running_max_applied)
        self.assertEqual(tab.raw.tolist(), [1.0, 3.0, 2.0, 4.0])
        self.assertEqual(tab.thresholds.tolist(), [1.0, 3.0, 3.0, 4.0])

    def test_lookup_is_conservative(self):
        c, s = self.synthetic(400, "train", 22)
        w = W.ratio(c, "strong")
        masses = W.ratio(self.synthetic(60, "strong", 23)[0], "strong")
        tab = CV.threshold_table(s, [22], w, CV.mass_grid(masses, 1.02))
        R, idx = CV.lookup(tab, masses)
        self.assertTrue(np.all(tab.grid[idx] >= masses) and np.all((idx == 0) | (tab.grid[idx - 1] < masses)))
        exact = np.array([CV.calibrate(s, [22], w, float(m)).threshold for m in masses])
        self.assertTrue(np.all(R >= exact))                              # never below the step's own threshold
        gaps = CV.lookup_gaps(tab, masses, idx)
        self.assertLessEqual(gaps["max_mass_gap"], 0.02 + 1e-12)
        self.assertLessEqual(float(np.max(R / exact - 1.0)), gaps["max_threshold_gap"] + 1e-12)
        with self.assertRaises(ValueError):
            CV.lookup(tab, [tab.grid[-1] * 1.5])                        # beyond the grid: refused

    def test_possibly_rounded_into_refusal_is_an_upper_bound(self):
        """Steps whose looked-up R is +inf while the grid point below is finite are counted; a step on the first
        infinite grid point is counted although its own exact R is already +inf, so the count bounds the steps refused
        only by rounding from above."""
        s = np.abs(np.random.default_rng([W.MASTER_SEED, 27]).standard_normal(300))
        ones = np.ones(300)
        tab = CV.threshold_table(s, [27], ones, CV.mass_grid([1.0, 20.0], 1.05))
        k = int(np.argmax(np.isinf(tab.thresholds)))
        self.assertTrue(k > 0 and np.isfinite(tab.thresholds[k - 1]))   # +inf is reached inside the grid
        lo, hi = tab.grid[k - 1], tab.grid[k]
        masses = np.array([hi, lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)])
        gaps = CV.lookup_gaps(tab, masses)
        self.assertEqual(gaps["possibly_rounded_into_refusal"], 3)
        exact = np.array([CV.calibrate(s, [27], ones, float(m)).threshold for m in masses])
        self.assertTrue(np.isinf(exact[0]))                              # on the infinite grid point: not rounding
        self.assertLessEqual(int(np.sum(np.isfinite(exact))), gaps["possibly_rounded_into_refusal"] - 1)

    def test_per_step_equals_unweighted_under_weights_of_one(self):
        """No tilt: every weight is 1, the grid is [1.0] and every step's threshold is the unweighted one bit for bit,
        so its widths and outcomes are the unweighted ones."""
        c, s = self.synthetic(843, "train", 24)
        ones = W.ratio(c, "none")
        tab = CV.threshold_table(s, [24], ones, CV.mass_grid(ones, 1.01))
        R_u = CV.calibrate(s, [24]).threshold
        self.assertEqual(tab.grid.tolist(), [1.0])
        self.assertEqual(tab.thresholds.tolist(), [R_u])
        ct, st = self.synthetic(2000, "train", 25)
        R, _ = CV.lookup(tab, W.ratio(ct, "none"))
        eta = np.float32(np.exp(np.random.default_rng(26).normal(0, 0.3, 2000)))
        self.assertEqual(CV.width(R, 0.7, eta).tobytes(), CV.width(R_u, 0.7, eta).tobytes())

    def test_per_step_weights_restore_coverage_where_unweighted_falls_short(self):
        """A synthetic tilt where the error grows with c (score spread 0.3 + 2c) and the scale ignores it: under the
        strong tilt, each step at its own threshold R(w(c)) covers about 90% (pooled over banks), the unweighted
        threshold does not."""
        ct, st = self.synthetic(40000, "strong", 30)
        wt = W.ratio(ct, "strong")
        per_step, single, unweighted = [], [], []
        for b in range(4):
            c, s = self.synthetic(843, "train", 31 + b)
            w = W.ratio(c, "strong")
            tab = CV.threshold_table(s, [32, b], w, CV.mass_grid(wt, 1.02))
            R, _ = CV.lookup(tab, wt)
            per_step.append(float(np.mean(st <= R)))
            single.append(float(np.mean(st <= CV.calibrate(s, [32, b], w, W.full_test_mass("strong")).threshold)))
            unweighted.append(float(np.mean(st <= CV.calibrate(s, [32, b]).threshold)))
        self.assertTrue(0.89 <= np.mean(per_step) <= 0.96, per_step)
        self.assertTrue(all(u < 0.88 for u in unweighted), unweighted)
        self.assertGreater(min(p - u for p, u in zip(per_step, unweighted)), 0.03)
        self.assertLess(abs(np.mean(per_step) - np.mean(single)), 0.01)   # the bank weights do the work

    def test_closed_form_dose_and_refusal(self):
        U = np.array([0.0, 1.0, 3.0, np.inf])
        np.testing.assert_allclose(CV.closed_form_dose(U, 1.0), [1.0, 1.25, 1.375, 1.0])          # BCA: refusal 1
        np.testing.assert_allclose(CV.closed_form_dose(U, 1.0, refused_dose=1.5), [1.0, 1.25, 1.375, 1.5])
        s = CV.per_step_summary(np.array([1.0, 2.0, np.inf, 3.0]), unweighted=2.0)
        self.assertEqual((s["refused_share"], s["min"], s["max"], s["ratio_max"]), (0.25, 1.0, 3.0, 1.5))


class ClassifierTest(unittest.TestCase):
    def test_newton_reaches_the_optimum_and_recovers_known_coefficients(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal((60000, 3))
        beta = np.array([0.5, -1.0, 0.0])
        y = rng.uniform(size=60000) < 1 / (1 + np.exp(-(0.2 + x @ beta)))
        clf = CL.fit(x[~y], x[y])
        self.assertTrue(clf.converged)
        self.assertLess(clf.max_score, 1e-6)
        np.testing.assert_allclose(clf.raw_coef, beta, atol=0.05)

    def test_slope_on_the_current_under_the_strong_tilt(self):
        rec = W.HoldRecorder()
        obs = {}
        for law, tag in (("train", 1), ("strong", 2)):
            sim = W.draw_episodes(150, np.random.default_rng([66, tag]), law, rec)
            obs[law] = sim["states"][:, :-1].astype(np.float32).reshape(-1, TC.DX)
        clf = CL.fit(obs["train"], obs["strong"])
        self.assertTrue(clf.converged)
        self.assertLess(abs(clf.slope_on_c - 4.0), 1.5)
        bank = obs["train"][::30]  # rows spread over every episode
        cmp = CL.compare(clf, bank, bank[:, TC.CURRENT], "strong")
        self.assertGreater(cmp["rank_correlation"], 0.9)
        self.assertEqual(cmp["exact_slope"], 4.0)

    def test_estimated_arm_refuses_inside_the_ball(self):
        ball = SH.Gap(centre=(0.0, 0.0, 0.0, 0.0), rho=0.1, mass=0.15, rule="base", q=0.15,
                      competing_ratio=float("nan"), record_sha256="")
        rng = np.random.default_rng(12)
        x0, x1 = rng.standard_normal((2000, 2)), rng.standard_normal((2000, 2)) + [0.5, 0.0]
        clf = CL.fit(x0, x1)
        acts = np.zeros((2000, 4), np.float32)
        acts[1000:] = 0.5
        arm = CL.estimated_arm(clf, x0[:100], x1, acts, ball)
        self.assertAlmostEqual(arm["test_sample_in_ball_share"], 0.5)
        self.assertAlmostEqual(arm["test_mass"], float(clf.weight(x1[1000:]).mean()))
        self.assertEqual(arm["bank_weights"].shape, (100,))

    def test_class_size_correction(self):
        rng = np.random.default_rng(13)
        clf = CL.fit(rng.standard_normal((6000, 2)), rng.standard_normal((2000, 2)))
        w = clf.weight(rng.standard_normal((4000, 2)))
        self.assertLess(abs(np.median(w) - 1.0), 0.1)


if __name__ == "__main__":
    unittest.main()
