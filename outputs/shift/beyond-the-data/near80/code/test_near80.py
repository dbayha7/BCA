"""Unit tests of the near-80 draft code, on synthetic logs only (no recorded seed is generated, loaded or read; no
learner is trained; no BCA band is fitted). The fixed banks are used only through tiny 5-start rollouts of scripted
controllers.

Run (plain unittest; pytest is not installed), from /tmp:
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \
    python /home/user/bca/outputs/shift/beyond-the-data/near80/code/test_near80.py
It takes a few seconds.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80lib as N  # noqa: E402
import facts_near80 as F  # noqa: E402
from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

PATTERN = np.array([-1.0, -0.137, -0.585, -0.406])
SYNTHETIC_REP = 901          # a synthetic tag for the random streams; not a recorded data seed


def synthetic_log(rep=SYNTHETIC_REP, seed=7, sizes=(250, 50, 50), length=20):
    """A small synthetic log with the gate's Log interface: normal driving capped below 80% (dense up to the band),
    a full-throttle cluster (inside the ball), 30 'survivors' near full throttle outside the ball, and a few rows
    exactly on and around the band's edges."""
    rng = np.random.default_rng(seed)
    E = int(sum(sizes))
    n = E * length
    mu = np.array([-0.55, -0.08, -0.35, -0.25])
    a = 0.8 * np.tanh((mu + 0.45 * rng.standard_normal((n, 4))) / 0.8)
    ft = rng.uniform(size=n) < 0.15
    a[ft] = np.clip(PATTERN + 0.03 * rng.standard_normal((int(ft.sum()), 4)), -1, 1)
    train_rows = np.arange(sizes[0] * length)
    surv = rng.choice(train_rows[~ft[train_rows]], 30, replace=False)
    a[surv] = np.array([-0.985, 0.4, 0.3, 0.3]) + 0.004 * rng.standard_normal((30, 4))
    edges = rng.choice(train_rows[~ft[train_rows] & ~np.isin(train_rows, surv)], 6, replace=False)
    edge_vals = [np.float32(0.7), np.nextafter(np.float32(0.7), np.float32(1)), np.float32(-0.9),
                 np.nextafter(np.float32(0.9), np.float32(1)), np.float32(TC.KNEE), np.float32(-0.75)]
    for r, v in zip(edges, edge_vals):
        a[r] = [0.1, 0.1, 0.1, 0.1]
        a[r, 2] = v                      # on thruster 3: inside the band only under scope 'any'
    a32 = a.astype(np.float32)
    x = 0.3 * rng.standard_normal((n, TC.DX))
    x[:, TC.CURRENT] = rng.uniform(0.1, 0.7, n)
    x[:, TC.WAVE1] = rng.uniform(-0.2, 0.3, n)
    x2 = x + 0.01 * rng.standard_normal((n, TC.DX))
    split = np.repeat(np.arange(3, dtype=np.int8), np.asarray(sizes) * length)
    t = np.tile(np.arange(length, dtype=np.int32), E)
    arrays = {
        "observations": x.astype(np.float32), "actions": a32, "rewards": np.zeros(n, np.float32),
        "next_observations": x2.astype(np.float32), "terminals": np.zeros(n, bool), "timeouts": t == length - 1,
        "infos/state": x, "infos/next_state": x2, "infos/episode": np.repeat(np.arange(E, dtype=np.int32), length),
        "infos/t": t, "infos/split": split, "infos/full_throttle_command": ft,
    }
    first = sizes[0]
    offs = np.random.default_rng(seed + 1).integers(0, length, (sizes[1], 5))
    bank = np.sort(np.unique(np.concatenate([(first + e) * length + np.unique(o) for e, o in enumerate(offs)])))
    meta = dict(rep=int(rep), bank_rows=bank.tolist(), sizes=list(sizes), length=length,
                physics=TC.world_physics("stall").as_dict(), recorder=RC.DEFAULT_SPEC.as_dict())
    return DS.Log(arrays, meta), edges, surv


def synthetic_gap():
    return SH.Gap(centre=tuple(float(v) for v in PATTERN), rho=0.3, mass=0.15, rule="base", q=0.15,
                  competing_ratio=2.0, record_sha256="synthetic")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log, cls.edges, cls.surv = synthetic_log()
        cls.gap = synthetic_gap()
        cls.near = N.band_removal(cls.log, cls.gap)
        cls.rnd = N.random_rows_removal(cls.log, cls.gap, cls.near)
        cls.step1 = N.load_step1lib()


class TestBand(Base):
    def test_edges_closed_float64_of_float32(self):
        v = np.zeros((7, 4), np.float32)
        v[:, 0] = [np.float32(0.7), np.nextafter(np.float32(0.7), np.float32(1)), np.float32(0.9),
                   np.nextafter(np.float32(0.9), np.float32(1)), np.float32(TC.KNEE), np.float32(-0.8), 0.5]
        got = N.in_band(v, scope="thruster1")
        # float32(0.7) = 0.69999999 is just below 0.70 (seen side); float32(0.9) = 0.89999998 is inside
        self.assertEqual(got.tolist(), [False, True, True, False, True, True, False])

    def test_scope_and_signs(self):
        v = np.zeros((3, 4), np.float32)
        v[0, 2] = -0.75
        v[1, 0] = -0.75
        v[2, 0] = 0.75
        self.assertEqual(N.in_band(v, scope="any").tolist(), [True, True, True])
        self.assertEqual(N.in_band(v, scope="thruster1").tolist(), [False, True, True])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            N.in_band(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            N.in_band(np.zeros((2, 4)), lo=0.9, hi=0.7)
        with self.assertRaises(ValueError):
            N.in_band(np.zeros((2, 4)), scope="thruster3")

    def test_edge_rows_of_the_synthetic_log(self):
        got = N.in_band(self.log["actions"][self.edges])
        self.assertEqual(got.tolist(), [False, True, True, False, True, True])


class TestRemoval(Base):
    def test_structure(self):
        log, gap, near = self.log, self.gap, self.near
        train, cal, ev = log.split_rows("train"), log.split_rows("calibration"), log.split_rows("evaluation")
        ball, band = gap.contains(log["actions"]), N.in_band(log["actions"])
        self.assertIsInstance(near, SH.Removal)
        self.assertEqual(near.kind, N.KIND_BAND)
        self.assertTrue(np.array_equal(near.removed[train], (ball | band)[train]))
        self.assertFalse(near.removed[cal].any() or near.removed[ev].any())
        self.assertTrue(np.array_equal(near.bank_removed, (ball | band)[log.bank_rows]))
        kept = near.kept_train_rows(log)
        self.assertEqual(int(band[kept].sum()), 0)
        self.assertEqual(int(ball[kept].sum()), 0)
        self.assertGreater(int((band & ~ball)[train].sum()), 100)        # the band adds rows beyond the ball
        self.assertGreater(int(band[ev].sum()), 0)                       # held-out band rows stay

    def test_empty_band_reproduces_the_gate_removal_bit_for_bit(self):
        gate = SH.make_removal(self.log, "gap", self.gap)
        empty = N.band_removal(self.log, self.gap, empty=True)
        self.assertTrue(np.array_equal(empty.removed, gate.removed))
        self.assertTrue(np.array_equal(empty.bank_removed, gate.bank_removed))
        a, b = empty.summary(self.log), gate.summary(self.log)
        self.assertEqual((a["kept_rows_sha256"], a["bank_rows_sha256"]), (b["kept_rows_sha256"], b["bank_rows_sha256"]))

    def test_removed_mask(self):
        m = N.removed_mask(self.log, self.gap)
        self.assertTrue(np.array_equal(m, self.near.removed))
        self.assertEqual(m.dtype, bool)
        self.assertEqual(m.shape, (self.log.n_rows,))

    def test_thruster1_scope_removes_fewer(self):
        t1 = N.band_removal(self.log, self.gap, scope="thruster1")
        self.assertLess(t1.count, self.near.count)
        self.assertTrue(np.all(self.near.removed >= t1.removed))


class TestRandomRows(Base):
    def test_matched_counts_and_source(self):
        log, gap, near, rnd = self.log, self.gap, self.near, self.rnd
        base = SH.make_removal(log, "gap", gap)
        train = log.split_rows("train")
        self.assertEqual(rnd.kind, N.KIND_RANDOM)
        self.assertEqual(int(rnd.removed[train].sum()), int(near.removed[train].sum()))
        self.assertEqual(int(rnd.bank_removed.sum()), int(near.bank_removed.sum()))
        self.assertTrue(np.all(rnd.removed >= base.removed))
        self.assertTrue(np.all(rnd.bank_removed >= base.bank_removed))
        self.assertGreater(int(N.in_band(log["actions"])[rnd.kept_train_rows(log)].sum()), 0)   # band rows remain

    def test_stratified_by_hard_water(self):
        """In each kind of water (calm, hard water), the control removes exactly as many training rows and bank rows
        as the band does, so it keeps the band's share of hard-water rows."""
        log, gap, near, rnd = self.log, self.gap, self.near, self.rnd
        base = SH.make_removal(log, "gap", gap)
        region = RC.Recorder(log.spec, log.physics).in_region(log["infos/state"])
        self.assertTrue(np.array_equal(N.hard_water(log), region))
        band_t, band_b = N.extra_per_stratum(log, near, base, region)
        mine_t, mine_b = N.extra_per_stratum(log, rnd, base, region)
        self.assertEqual(mine_t, band_t)
        self.assertEqual(mine_b, band_b)
        self.assertTrue(band_t[0] > 0 and band_t[1] > 0, band_t)       # the synthetic log exercises both strata
        train = log.split_rows("train")
        kept_share = lambda rm: float(np.mean(~rm.removed[train][region[train]]))
        self.assertEqual(kept_share(rnd), kept_share(near))

    def test_rng_tags(self):
        self.assertEqual(N.rng_tags(3, 0), [TC.DATA_SEED, 3, SH.TAG_RANDOM, N.TAG_NEAR80, 2])
        self.assertEqual(N.rng_tags(3, 1), [TC.DATA_SEED, 3, SH.TAG_RANDOM, N.TAG_NEAR80, 3])
        self.assertEqual(N.rng_tags(3, 0, bank=True), [TC.DATA_SEED, 3, SH.TAG_RANDOM, N.TAG_NEAR80, 4])
        self.assertEqual(N.rng_tags(3, 1, bank=True), [TC.DATA_SEED, 3, SH.TAG_RANDOM, N.TAG_NEAR80, 5])
        for s in (0, 1):
            for bank in (False, True):
                self.assertNotEqual(N.rng_tags(3, s, bank)[-1], 0)          # numpy would drop a trailing 0
        # the training draw in stratum s is exactly default_rng(rng_tags(rep, s)) over that stratum's ball-kept rows
        log, gap, near, rnd = self.log, self.gap, self.near, self.rnd
        base = SH.make_removal(log, "gap", gap)
        region = N.hard_water(log)
        train = log.split_rows("train")
        band_t, _ = N.extra_per_stratum(log, near, base, region)
        for s in (0, 1):
            in_s = region if s == 1 else ~region
            kept_s = train[~base.removed[train] & in_s[train]]
            pick = np.random.default_rng(N.rng_tags(SYNTHETIC_REP, s)).choice(len(kept_s), band_t[s], replace=False)
            got = train[(rnd.removed & ~base.removed)[train] & in_s[train]]
            self.assertTrue(np.array_equal(np.sort(kept_s[pick]), got))

    def test_deterministic_and_rep_dependent(self):
        again = N.random_rows_removal(self.log, self.gap, self.near)
        self.assertTrue(np.array_equal(again.removed, self.rnd.removed))
        other, _, _ = synthetic_log(rep=SYNTHETIC_REP + 1)
        other_rnd = N.random_rows_removal(other, self.gap, N.band_removal(other, self.gap))
        self.assertFalse(np.array_equal(other_rnd.removed, self.rnd.removed))

    def test_refuses_a_removal_without_the_ball(self):
        base = SH.make_removal(self.log, "gap", self.gap)
        rem = self.near.removed.copy()
        rem[np.flatnonzero(base.removed)[0]] = False
        with self.assertRaises(ValueError):
            N.random_rows_removal(self.log, self.gap, SH.Removal(N.KIND_BAND, rem, self.near.bank_removed, self.gap))

    def test_leak_check_of_control(self):
        c = N.leak_check(self.log, self.rnd, self.gap, band_rm=self.near)
        self.assertTrue(c["passed"], c)
        self.assertTrue(c["same_training_count_per_stratum"] and c["same_bank_count_per_stratum"])
        with self.assertRaises(ValueError):
            N.leak_check(self.log, self.rnd, self.gap)

    # negative tests: each tampering of the control must fail its leak check

    def _check(self, removed=None, bank_removed=None):
        rm = SH.Removal(N.KIND_RANDOM, self.rnd.removed.copy() if removed is None else removed,
                        self.rnd.bank_removed.copy() if bank_removed is None else bank_removed, self.gap)
        c = N.leak_check(self.log, rm, self.gap, band_rm=self.near)
        self.assertFalse(c["passed"], c)
        with self.assertRaises(N.LeakError):
            N.assert_no_leak(self.log, rm, self.gap, band_rm=self.near)
        return c

    def test_catches_one_extra_training_row(self):
        train = self.log.split_rows("train")
        rem = self.rnd.removed.copy()
        rem[train[~rem[train]][0]] = True
        c = self._check(removed=rem)
        self.assertFalse(c["same_training_count_as_band"])

    def test_catches_one_missing_training_row(self):
        base = SH.make_removal(self.log, "gap", self.gap)
        rem = self.rnd.removed.copy()
        rem[np.flatnonzero(rem & ~base.removed)[0]] = False
        c = self._check(removed=rem)
        self.assertFalse(c["same_training_count_as_band"])

    def test_catches_bank_count_off_by_one(self):
        br = self.rnd.bank_removed.copy()
        br[np.flatnonzero(~br)[0]] = True
        c = self._check(bank_removed=br)
        self.assertFalse(c["same_bank_count_as_band"])
        base = SH.make_removal(self.log, "gap", self.gap)
        br = self.rnd.bank_removed.copy()
        br[np.flatnonzero(br & ~base.bank_removed)[0]] = False
        c = self._check(bank_removed=br)
        self.assertFalse(c["same_bank_count_as_band"])

    def test_catches_a_removed_evaluation_row(self):
        rem = self.rnd.removed.copy()
        rem[self.log.split_rows("evaluation")[0]] = True
        c = self._check(removed=rem)
        self.assertFalse(c["only_training_rows_removed"])

    def test_catches_a_kept_ball_row(self):
        train = self.log.split_rows("train")
        ball = self.gap.contains(self.log["actions"])
        rem = self.rnd.removed.copy()
        rem[train[ball[train]][0]] = False
        c = self._check(removed=rem)
        self.assertEqual(c["kept_training_rows_in_ball"], 1)
        self.assertFalse(c["ball_rows_all_removed"])

    def test_catches_rows_moved_between_strata(self):
        """Same totals, wrong mix: one extra row moved from hard water to calm."""
        log = self.log
        base = SH.make_removal(log, "gap", self.gap)
        region = N.hard_water(log)
        train = log.split_rows("train")
        rem = self.rnd.removed.copy()
        hard = train[(rem & ~base.removed & region)[train]][0]
        calm = train[(~rem & ~region)[train]][0]
        rem[hard], rem[calm] = False, True
        c = self._check(removed=rem)
        self.assertTrue(c["same_training_count_as_band"])
        self.assertFalse(c["same_training_count_per_stratum"])
        _, band_b = N.extra_per_stratum(log, self.near, base, region)
        bank_region = region[log.bank_rows]
        br = self.rnd.bank_removed.copy()
        hard_b = np.flatnonzero(br & ~base.bank_removed & bank_region)
        calm_b = np.flatnonzero(~br & ~bank_region)
        if band_b[1] > 0 and len(calm_b):
            br[hard_b[0]], br[calm_b[0]] = False, True
            c = self._check(bank_removed=br)
            self.assertTrue(c["same_bank_count_as_band"])
            self.assertFalse(c["same_bank_count_per_stratum"])


class TestTrainingFile(Base):
    def test_matches_the_leak_checked_arrays(self):
        arrays = DS.d4rl_dict(self.log, self.near.kept_train_rows(self.log))
        self.assertTrue(N.training_file_check(arrays, self.log, self.near)["passed"])

    def test_catches_a_changed_or_missing_row(self):
        arrays = DS.d4rl_dict(self.log, self.near.kept_train_rows(self.log))
        bad = dict(arrays, actions=arrays["actions"].copy())
        bad["actions"][0, 0] = np.float32(0.75)                      # a band command slipped into the file
        c = N.training_file_check(bad, self.log, self.near)
        self.assertFalse(c["passed"])
        self.assertFalse(c["actions_sha256_equal"])
        short = {k: v[1:] for k, v in arrays.items()}
        self.assertFalse(N.training_file_check(short, self.log, self.near)["passed"])
        missing = {k: v for k, v in arrays.items() if k != "rewards"}
        self.assertFalse(N.training_file_check(missing, self.log, self.near)["passed"])
        other = DS.d4rl_dict(self.log, self.rnd.kept_train_rows(self.log))
        self.assertFalse(N.training_file_check(other, self.log, self.near)["passed"])


class TestLeakCheck(Base):
    def test_passes(self):
        c = N.assert_no_leak(self.log, self.near, self.gap)
        self.assertTrue(c["passed"])
        self.assertEqual(c["kept_training_rows_in_band"], 0)
        self.assertEqual(c["d4rl_commands_in_band"], 0)

    def _tampered(self, removed=None, bank_removed=None):
        return SH.Removal(N.KIND_BAND, self.near.removed.copy() if removed is None else removed,
                          self.near.bank_removed.copy() if bank_removed is None else bank_removed, self.gap)

    def test_catches_a_kept_band_row(self):
        log, gap = self.log, self.gap
        train = log.split_rows("train")
        ball, band = gap.contains(log["actions"]), N.in_band(log["actions"])
        r = train[(band & ~ball)[train]][0]
        rem = self.near.removed.copy()
        rem[r] = False
        c = N.leak_check(log, self._tampered(removed=rem), gap)
        self.assertFalse(c["passed"])
        self.assertEqual(c["kept_training_rows_in_band"], 1)
        self.assertEqual(c["d4rl_commands_in_band"], 1)
        with self.assertRaises(N.LeakError):
            N.assert_no_leak(log, self._tampered(removed=rem), gap)

    def test_catches_a_kept_bank_band_row(self):
        log, gap = self.log, self.gap
        band_bank = N.in_band(log["actions"])[log.bank_rows] & ~gap.contains(log["actions"])[log.bank_rows]
        i = int(np.flatnonzero(band_bank)[0])
        br = self.near.bank_removed.copy()
        br[i] = False
        c = N.leak_check(log, self._tampered(bank_removed=br), gap)
        self.assertFalse(c["passed"])
        self.assertEqual(c["kept_bank_rows_in_band"], 1)

    def test_catches_a_touched_held_out_row(self):
        rem = self.near.removed.copy()
        rem[self.log.split_rows("evaluation")[0]] = True
        c = N.leak_check(self.log, self._tampered(removed=rem), self.gap)
        self.assertFalse(c["passed"])
        self.assertFalse(c["only_training_rows_removed"])

    def test_catches_an_extra_removed_row(self):
        log, gap = self.log, self.gap
        train = log.split_rows("train")
        rem = self.near.removed.copy()
        rem[train[~rem[train]][0]] = True          # a seen row removed for no reason
        c = N.leak_check(log, self._tampered(removed=rem), gap)
        self.assertFalse(c["passed"])
        self.assertFalse(c["removed_is_exactly_ball_or_band"])

    def test_catches_a_kept_ball_row(self):
        log, gap = self.log, self.gap
        train = log.split_rows("train")
        ball = gap.contains(log["actions"])
        rem = self.near.removed.copy()
        rem[train[ball[train]][0]] = False
        c = N.leak_check(log, self._tampered(removed=rem), gap)
        self.assertFalse(c["passed"])
        self.assertEqual(c["kept_training_rows_in_ball"], 1)


class TestFacts(Base):
    def test_world_card_restates_the_gate(self):
        from experiments.shift import acceptance as AC
        physics, spec = TC.world_physics("stall"), RC.DEFAULT_SPEC
        card, _ = AC.world_marks(self.log, self.gap, spec, physics)
        train = self.log.split_rows("train")
        mine = N.world_card(self.log, self.gap.contains(self.log["actions"])[train], spec, physics)
        for k in ("full_throttle_commands_removed", "kept_past80_share", "identical_rule_share", "region_rows_kept"):
            self.assertEqual(mine[k], card[k], k)

    def test_gap_rule_moves_to_the_band_edge(self):
        acts = self.log["actions"]
        g_near = self.step1.gap_rule(self.step1.throttle(acts[self.near.kept_train_rows(self.log)]))
        g_rnd = self.step1.gap_rule(self.step1.throttle(acts[self.rnd.kept_train_rows(self.log)]))
        self.assertTrue(g_near["defined"])
        self.assertLess(g_near["t_lo"], N.BAND_LO)
        self.assertGreaterEqual(g_near["t_lo"], N.NO_TRAINING["gap_lo_min"])
        self.assertTrue(g_rnd["defined"])
        self.assertGreater(g_rnd["t_lo"], 0.75)           # random rows leave normal driving's top near 0.8

    def test_seed_world_facts_and_checks(self):
        physics, spec = TC.world_physics("stall"), RC.DEFAULT_SPEC
        anchors = F.anchors_for("stall", bank_n=5)
        f = F.seed_world_facts(self.log, self.gap, "stall", physics, spec, self.step1, anchors=anchors, bank_n=5)
        self.assertTrue(f["leak_check_band"]["passed"])
        self.assertTrue(f["leak_check_random"]["passed"])
        self.assertTrue(f["reproduces_gate_removal_with_empty_band"])
        tr = f["training_split"]
        self.assertEqual(f["near80"]["extra"], tr["band_only"])
        self.assertEqual(tr["band_rows"], tr["band_and_ball"] + tr["band_only"])
        self.assertEqual(f["near80"]["kept_rows"], f["ball_alone"]["kept_rows"] - tr["band_only"])
        self.assertEqual(f["random_rows"]["kept_rows"], f["near80"]["kept_rows"])
        self.assertEqual(f["random_rows"]["bank_rows_kept"], f["near80"]["bank_rows_kept"])
        self.assertLess(f["near80"]["max_throttle_index_below_band"], N.BAND_LO)
        self.assertEqual(f["world_card_ball"]["region_rows_kept"], f["gate_world_card_on_ball"]["region_rows_kept"])
        self.assertLessEqual(f["world_card_near80"]["region_rows_kept"], f["world_card_ball"]["region_rows_kept"])
        self.assertIn("data_edge_loss_strong_current", f)
        fixed = F.fixed_bank_facts(anchors, bank_n=5)
        self.assertAlmostEqual(fixed["situations"]["strong_current"]["stall_aware_points"], 100.0, places=6)
        self.assertGreater(fixed["stall_aware_thruster1_band_share_strong_current"], 0.0)
        res = N.no_training_checks([f], fixed)
        names = {r["rule"] for r in res["rules"]}
        self.assertTrue(any(n.startswith("check 1:") for n in names) and any(n.startswith("check 10:") for n in names))
        self.assertEqual(res["not_run"], [])
        json.dumps(res)                                    # serializable

    def test_no_training_thresholds(self):
        physics, spec = TC.world_physics("stall"), RC.DEFAULT_SPEC
        f = F.seed_world_facts(self.log, self.gap, "stall", physics, spec, self.step1)
        rules = lambda ff: {r["rule"]: r["passed"] for r in N.no_training_checks([ff])["rules"]}
        g = json.loads(json.dumps(f))
        g["world_card_near80"]["region_rows_kept"] = 0.3499
        self.assertFalse(rules(g)["check 4: hard-water rows kept (35%, set after seeing seeds 0-4)"])
        g["world_card_near80"]["region_rows_kept"] = 0.35
        self.assertTrue(rules(g)["check 4: hard-water rows kept (35%, set after seeing seeds 0-4)"])
        g["near80"]["bank_rows_kept"] = 499
        self.assertFalse(rules(g)["check 7: the calibration bank keeps enough rows"])
        g["near80"]["gap_rule"]["t_lo"] = 0.786
        self.assertFalse(rules(g)["check 6: BCA's gap rule is defined, lower end at the band's edge"])
        res = N.no_training_checks([f])
        self.assertFalse(res["passed"])                    # rollouts not run: checks 9 and 10 are missing
        self.assertTrue(res["not_run"])


class TestReadoutHelpers(unittest.TestCase):
    def test_hold_and_ramp_start_follow_the_band(self):
        self.assertEqual(N.HOLD, N.BAND_LO)
        self.assertEqual(N.NO_TRAINING["gap_lo_max"], N.BAND_LO)
        self.assertAlmostEqual(N.NO_TRAINING["gap_lo_min"], N.BAND_LO - 0.005, places=12)

    def test_zones(self):
        c = np.zeros((6, 4))
        c[:, 0] = [0.69, 0.70, 0.75, TC.KNEE, 0.8000001, -0.95]
        self.assertEqual(N.throttle_zones(c).tolist(), [0, 1, 1, 1, 2, 2])
        self.assertEqual(N.throttle_zones(np.float32(c)).tolist()[3], 1)    # float32(0.8) is not past the knee

    def test_clip(self):
        c = np.array([[-0.95, 0.75, 0.5, -0.71]])
        self.assertEqual(N.clip_throttle(c, N.HOLD).tolist(), [[-0.7, 0.7, 0.5, -0.7]])
        z = N.throttle_zones(np.float32(N.clip_throttle(c, N.HOLD)))
        self.assertEqual(z.tolist(), [0])                  # held to 70% in float32 is on the seen side

    def test_range_clip_controller(self):
        K = RC.textbook_gain(TC.world_physics("stall"))
        x = TC.bank_starts("strong_current", 5)
        lo, hi = -0.5 * np.ones(4), 0.4 * np.ones(4)
        u = N.RangeClip(K, lo, hi, "test")(x)
        self.assertTrue(np.all(u >= lo) and np.all(u <= hi))


class TestGuards(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(N.check_seeds([0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])
        for s in (5, 9, 10, 14, 15, 900):
            with self.assertRaises(N.Refused):
                N.check_seeds([s])
        with self.assertRaises(N.Refused):
            F.main(["--out", "/nonexistent/x.json", "--seeds", "10"])

    def test_outputs(self):
        for p in (N.ROOT / "outputs" / "x.json", HERE / "x.json", N.ROOT / "runs" / "shift" / "x.json",
                  N.ROOT / "experiments" / "shift" / "x.json", N.ROOT / "runs" / "shift_explore" / "room" / "x.json"):
            with self.assertRaises(N.Refused):
                N.guard_out(p)
        N.guard_out(N.ROOT / "runs" / "shift_explore" / "near80" / "stageA" / "facts.json")   # allowed, not created
        with tempfile.TemporaryDirectory() as d:
            N.guard_out(Path(d) / "facts.json")
            (Path(d) / "exists.json").write_text("{}")
            with self.assertRaises(N.Refused):
                N.guard_out(Path(d) / "exists.json")

    def test_step1lib_is_verified(self):
        mod = N.load_step1lib()
        self.assertTrue(hasattr(mod, "gap_rule") and hasattr(mod, "throttle"))

    def test_nothing_written_into_the_code_folder(self):
        """Only the planned Python files may live here (this draft's three, and the four the final version adds); no
        output, cache or other file."""
        allowed = {"near80lib.py", "facts_near80.py", "test_near80.py",
                   "near80_runner.py", "pooled.py", "import_approval.py", "freeze.py"}
        for p in HERE.iterdir():
            self.assertTrue(p.is_file() and p.suffix == ".py", f"not a Python file: {p.name}")
            self.assertIn(p.name, allowed)
        for name in ("near80lib.py", "facts_near80.py", "test_near80.py"):
            self.assertTrue((HERE / name).is_file(), name)


class TestDraftJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.loads((HERE.parent / "prereg_draft.json").read_text())
        cls.doc = (HERE.parent / "PREREGISTRATION.md").read_text()

    def test_draft_matches_the_code(self):
        d = self.d
        self.assertEqual(d["status"], "draft")
        self.assertEqual(d["band"]["lo"], N.BAND_LO)
        self.assertEqual(d["band"]["hi"], N.BAND_HI)
        self.assertEqual(d["band"]["scope"], N.SCOPE)
        self.assertEqual(d["seeds"]["exploratory"], list(N.EXPLORATORY_SEEDS))
        self.assertEqual(d["seeds"]["confirmatory"], list(N.CONFIRMATORY_SEEDS))
        self.assertEqual(d["kill_rules"]["no_training_checks"]["thresholds"], N.NO_TRAINING)
        self.assertEqual(d["measures"]["hold_level"], N.HOLD)
        tags = d["random_rows_control"]["rng_tags"]
        for s, name in enumerate(N.STRATA):
            self.assertEqual(tags["training"][name], [TC.DATA_SEED, "rep", SH.TAG_RANDOM, N.TAG_NEAR80,
                                                      N.TRAIN_TAG_BASE + s])
            self.assertEqual(tags["bank"][name], [TC.DATA_SEED, "rep", SH.TAG_RANDOM, N.TAG_NEAR80,
                                                  N.BANK_TAG_BASE + s])

    def _strings(self, x, skip_choices=True):
        if isinstance(x, str):
            yield x
        elif isinstance(x, list):
            for v in x:
                yield from self._strings(v, skip_choices)
        elif isinstance(x, dict):
            for k, v in x.items():
                if not (skip_choices and k == "choices"):
                    yield from self._strings(v, skip_choices)

    def test_every_choice_number_names_its_choice(self):
        """Every 'choice N' in the JSON is followed by '(name)', the name of the Nth entry of the choices list."""
        import re
        choices = self.d["choices"]
        for i, c in enumerate(choices):
            self.assertEqual(c["number"], i + 1)
        seen = 0
        for text in self._strings(self.d):
            for m in re.finditer(r"choices? (\d+)", text):
                n = int(m.group(1))
                rest = text[m.end():]
                self.assertTrue(rest.startswith(" ("), f"'choice {n}' without its name in: {text!r}")
                name = rest[2:rest.index(")")]
                self.assertTrue(1 <= n <= len(choices), text)
                self.assertEqual(name, choices[n - 1]["choice"], text)
                seen += 1
        self.assertGreater(seen, 0)

    def test_choices_match_the_document(self):
        """Section 16 of PREREGISTRATION.md lists the same choices, in the same order, with the same names."""
        import re
        sec = self.doc[self.doc.index("## 16."):self.doc.index("## 17.")]
        titles = re.findall(r"^(\d+)\. \*\*([^*]+?)\.?\*\*", sec, flags=re.M)
        self.assertEqual([int(n) for n, _ in titles], list(range(1, len(self.d["choices"]) + 1)))
        for (n, t), c in zip(titles, self.d["choices"]):
            self.assertEqual(t.strip().rstrip(".").lower(), c["choice"].lower(), n)

    def test_document_brackets_are_defined(self):
        """Every [X] bracket in section 14's sentences is defined in its bracket table."""
        import re
        sec = self.doc[self.doc.index("## 14."):self.doc.index("## 15.")]
        table = sec[:sec.index("**Always attached:**")]
        defined = set()
        for row in re.findall(r"^\| ([^|]+) \|", table, flags=re.M):
            defined.update(x.strip() for x in re.split(r",| and ", row) if x.strip())
        used = set(re.findall(r"\[([A-Z][A-Za-z0-9]*)\]", sec[len(table):]))
        self.assertTrue(used <= defined, used - defined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
