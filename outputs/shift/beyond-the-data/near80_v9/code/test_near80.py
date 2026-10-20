"""Unit tests of version 9's data code (the main setting's removal, BCA's pooled held-out groups, their checks, the
outcome-free count, the training-file check and the no-training facts), on synthetic logs only (no recorded seed is
generated, loaded or read; no learner is trained; no BCA certificate is computed on recorded data). The
machine-readable pre-registration and the documents are checked against the code: prereg.json when it exists,
otherwise the draft it is built from.

Run (plain unittest; pytest is not installed), from /tmp:
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \
    python /home/user/bca/outputs/shift/beyond-the-data/near80_v9/code/test_near80.py
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
import near80_constants as K  # noqa: E402
import facts_v9 as F  # noqa: E402
from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

PATTERN = np.array([-1.0, -0.137, -0.585, -0.406])
SYNTHETIC_REP = 901          # a synthetic tag for the random streams; not a recorded data seed
HELD_OUT_PAST = dict(bank=4, calibration_other=3, evaluation=5)


def synthetic_log(rep=SYNTHETIC_REP, seed=7, sizes=(250, 50, 50), length=20):
    """A small synthetic log with the gate's Log interface: normal driving capped below 80%, a full-throttle cluster
    (inside the ball), 30 'survivors' near full throttle outside the ball in the training split, a few rows on and
    around 70%-90% throttles, and HELD_OUT_PAST held-out rows near full throttle outside the ball (calibration-bank,
    other calibration and evaluation rows), so that BCA's pooled full-throttle group holds rows of every source."""
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
    lg = DS.Log(arrays, meta)
    a32 = a32.copy()                     # the Log made its arrays read-only
    held = {}
    for name, split_rows in (("bank", bank), ("calibration_other", np.setdiff1d(lg.split_rows("calibration"), bank)),
                             ("evaluation", lg.split_rows("evaluation"))):
        free = split_rows[~ft[split_rows]]
        held[name] = rng.choice(free, HELD_OUT_PAST[name], replace=False)
        a32[held[name]] = (np.array([-0.985, 0.4, 0.3, 0.3]) + 0.004 * rng.standard_normal((len(held[name]), 4))
                           ).astype(np.float32)
    arrays["actions"] = a32
    return DS.Log(arrays, meta), edges, surv, held


def synthetic_gap():
    return SH.Gap(centre=tuple(float(v) for v in PATTERN), rho=0.3, mass=0.15, rule="base", q=0.15,
                  competing_ratio=2.0, record_sha256="synthetic")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log, cls.edges, cls.surv_planted, cls.held = synthetic_log()
        cls.gap = synthetic_gap()
        cls.removal = SH.make_removal(cls.log, "gap", cls.gap)
        cls.kept = cls.removal.kept_train_rows(cls.log)
        cls.step1 = N.load_step1lib()
        cls.gr = N.gap_rule_of_rows(cls.log, cls.kept, cls.step1, K.GAP_RULE)
        cls.surv = N.survivors_of(cls.kept, cls.gr)
        cls.t_lo = float(cls.gr["t_lo"])
        cls.tail, cls.rest = N.pooled_groups(cls.log, cls.gap, cls.surv, cls.t_lo)

    def brute_heldout(self):
        """H by brute force, row by row (independent of near80lib)."""
        out = []
        split = np.asarray(self.log["infos/split"])
        for r in range(self.log.n_rows):
            if split[r] in (1, 2) and not bool(self.gap.contains(self.log["actions"][r:r + 1])[0]):
                out.append(r)
        return np.asarray(out, np.int64)


class TestRemoval(Base):
    def test_the_main_setting_is_the_gate_removal(self):
        """Version 9's removal is the main test's ball, unchanged: every ball training row out, nothing held out
        touched."""
        log, gap, rm = self.log, self.gap, self.removal
        train, cal, ev = log.split_rows("train"), log.split_rows("calibration"), log.split_rows("evaluation")
        ball = np.asarray(gap.contains(log["actions"]), bool)
        self.assertTrue(np.array_equal(rm.removed[train], ball[train]))
        self.assertFalse(rm.removed[cal].any() or rm.removed[ev].any())
        self.assertTrue(np.array_equal(rm.bank_removed, ball[log.bank_rows]))
        self.assertEqual(int(ball[self.kept].sum()), 0)

    def test_gap_rule_and_survivors(self):
        self.assertTrue(self.gr["defined"])
        self.assertGreaterEqual(len(self.surv), 9)
        self.assertTrue(set(self.surv.tolist()) <= set(self.kept.tolist()))
        self.assertTrue(np.all(N.throttle(self.log["actions"][self.surv]) > self.t_lo))
        planted = set(int(r) for r in self.surv_planted)
        self.assertTrue(planted <= set(self.surv.tolist()))


class TestHeldout(Base):
    def test_heldout_rows_exactly(self):
        h = N.heldout_rows(self.log, self.gap)
        self.assertTrue(np.array_equal(h, self.brute_heldout()))
        self.assertEqual(len(np.intersect1d(h, self.log.split_rows("train"))), 0)
        self.assertFalse(np.asarray(self.gap.contains(self.log["actions"][h]), bool).any())
        bank = np.asarray(self.log.bank_rows)
        kept_bank = bank[~np.asarray(self.gap.contains(self.log["actions"][bank]), bool)]
        self.assertTrue(set(kept_bank.tolist()) <= set(h.tolist()))          # H holds the kept bank
        self.assertGreater(len(h), len(kept_bank))                            # and more

    def test_splits_argument(self):
        cal_only = N.heldout_rows(self.log, self.gap, splits=("calibration",))
        self.assertTrue(set(cal_only.tolist()) < set(N.heldout_rows(self.log, self.gap).tolist()))


class TestGroups(Base):
    def test_exact_row_sets(self):
        """tail = the survivors then every H row past t_lo; rest = every H row at or below t_lo (brute force)."""
        h = self.brute_heldout()
        t = np.abs(self.log["actions"][h].astype(np.float64)).max(axis=1)
        want_tail = np.concatenate([self.surv, h[t > self.t_lo]])
        self.assertTrue(np.array_equal(self.tail, want_tail))
        self.assertTrue(np.array_equal(self.rest, h[t <= self.t_lo]))
        planted = np.concatenate(list(self.held.values()))
        self.assertTrue(set(planted.tolist()) <= set(self.tail.tolist()))

    def test_group_checks_pass(self):
        c = N.group_checks(self.log, self.gap, self.kept, self.surv, self.tail, self.rest, self.t_lo)
        self.assertTrue(c["passed"], c)

    def _check(self, tail=None, rest=None, surv=None):
        return N.group_checks(self.log, self.gap, self.kept, self.surv if surv is None else surv,
                              self.tail if tail is None else tail, self.rest if rest is None else rest, self.t_lo)

    def test_catches_a_ball_row(self):
        ball = np.flatnonzero(np.asarray(self.gap.contains(self.log["actions"]), bool)
                              & (np.asarray(self.log["infos/split"]) == 2))
        c = self._check(rest=np.concatenate([self.rest, ball[:1]]))
        self.assertFalse(c["no_row_inside_the_ball"])
        self.assertFalse(c["passed"])

    def test_catches_a_training_row_that_is_not_a_survivor(self):
        extra = np.setdiff1d(self.kept, self.surv)[:1]
        c = self._check(rest=np.concatenate([self.rest, extra]))
        self.assertFalse(c["training_rows_are_the_survivors_only"])
        self.assertFalse(c["passed"])

    def test_catches_a_missing_or_doubled_row(self):
        self.assertFalse(self._check(rest=self.rest[1:])["groups_are_heldout_plus_survivors"])
        c = self._check(rest=np.concatenate([self.rest, self.tail[-1:]]))
        self.assertFalse(c["groups_disjoint"])

    def test_catches_a_row_on_the_wrong_side_of_t_lo(self):
        tail = self.tail[:-1]
        rest = np.concatenate([self.rest, self.tail[-1:]])
        c = self._check(tail=tail, rest=rest)
        self.assertFalse(c["rest_at_or_below_t_lo"])
        self.assertFalse(c["passed"])

    def test_catches_a_missing_survivor(self):
        c = self._check(surv=self.surv[1:], tail=self.tail[1:])
        self.assertFalse(c["survivors_are_kept_rows_past_t_lo"])

    def test_sources(self):
        s = N.group_sources(self.log, self.tail, self.surv)
        self.assertEqual(s["n"], len(self.tail))
        self.assertEqual(sum(s["by_source"].values()) + s["other"], s["n"])
        self.assertEqual(s["other"], 0)
        self.assertEqual(s["by_source"]["survivors"], len(self.surv))
        for name, k in HELD_OUT_PAST.items():
            self.assertGreaterEqual(s["by_source"][name], k, name)
        self.assertEqual(s["rows_sha256"], N.rows_sha256(self.tail))
        self.assertLessEqual(s["episodes"], s["n"])

    def test_one_per_episode(self):
        keep = N.one_per_episode(self.log, self.tail)
        eps = np.asarray(self.log["infos/episode"])[self.tail[keep]]
        self.assertEqual(len(np.unique(eps)), len(eps))
        self.assertEqual(len(eps), len(np.unique(np.asarray(self.log["infos/episode"])[self.tail])))

    def test_bank_only_groups_are_the_pooled_groups_on_the_bank(self):
        bt, br = N.bank_only_groups(self.log, self.gap, self.surv, self.t_lo)
        bank = set(int(r) for r in self.log.bank_rows)
        self.assertTrue(np.array_equal(bt, np.concatenate([self.surv, [r for r in self.tail[len(self.surv):]
                                                                       if int(r) in bank]]).astype(np.int64)))
        self.assertTrue(np.array_equal(br, np.asarray([r for r in self.rest if int(r) in bank], np.int64)))
        self.assertLess(len(bt), len(self.tail))


class TestCount(Base):
    def test_count_fields(self):
        c = N.outcome_free_count(self.log, self.gap, self.kept, self.step1, K.GAP_RULE)
        self.assertTrue(c["defined"])
        self.assertEqual((c["n_tail"], c["n_rest"]), (len(self.tail), len(self.rest)))
        self.assertEqual(c["n_survivors"], len(self.surv))
        self.assertEqual(c["tail"]["rows_sha256"], N.rows_sha256(self.tail))
        self.assertTrue(c["group_checks"]["passed"])
        self.assertEqual(c["certifiable_bar"], K.CERTIFIABLE_MIN)
        self.assertEqual(c["below_bar"], c["n_tail"] < 36)
        self.assertEqual(c["certifiable"], N.certifiable(c["n_tail"]))

    def test_certifiable_bar(self):
        self.assertTrue(N.certifiable(36))
        self.assertFalse(N.certifiable(35))
        self.assertFalse(N.certifiable(29))

    def test_count_reads_no_outcome(self):
        """Scrambling every reward, next state and state leaves the count unchanged: it reads commands only."""
        c0 = N.outcome_free_count(self.log, self.gap, self.kept, self.step1, K.GAP_RULE)
        arrays = dict(self.log._arrays)                       # every array, the diagnostic flag included
        rng = np.random.default_rng(3)
        bad = dict(arrays)
        for k in ("rewards", "next_observations", "observations", "infos/state", "infos/next_state"):
            bad[k] = rng.permutation(np.asarray(arrays[k]))
        lg = DS.Log(bad, dict(self.log.meta))
        c1 = N.outcome_free_count(lg, self.gap, self.kept, self.step1, K.GAP_RULE)
        self.assertEqual(json.dumps(c0, sort_keys=True, default=str), json.dumps(c1, sort_keys=True, default=str))

    def test_undefined_gap_rule(self):
        few = self.kept[~np.isin(self.kept, self.surv)]
        c = N.outcome_free_count(self.log, self.gap, few, self.step1, K.GAP_RULE)
        self.assertFalse(c["defined"])
        self.assertEqual((c["n_tail"], c["n_rest"]), (0, 0))


class TestTrainingFile(Base):
    def test_matches_the_arrays(self):
        arrays = DS.d4rl_dict(self.log, self.kept)
        self.assertTrue(N.training_file_check(arrays, self.log, self.removal)["passed"])

    def test_catches_a_changed_or_missing_row(self):
        arrays = DS.d4rl_dict(self.log, self.kept)
        bad = dict(arrays, actions=arrays["actions"].copy())
        bad["actions"][0, 0] = np.float32(0.75)
        c = N.training_file_check(bad, self.log, self.removal)
        self.assertFalse(c["passed"])
        self.assertFalse(c["actions_sha256_equal"])
        short = {k: v[1:] for k, v in arrays.items()}
        self.assertFalse(N.training_file_check(short, self.log, self.removal)["passed"])
        missing = {k: v for k, v in arrays.items() if k != "rewards"}
        self.assertFalse(N.training_file_check(missing, self.log, self.removal)["passed"])
        with_held = DS.d4rl_dict(self.log, np.sort(np.concatenate([self.kept, self.rest[:1]])))
        self.assertFalse(N.training_file_check(with_held, self.log, self.removal)["passed"])


class TestFacts(Base):
    def gate(self):
        return dict(ball_leak_checks_passed=True, world_marks=None, passed=True)

    def test_seed_world_facts_and_checks(self):
        f = F.seed_world_facts(self.log, self.gap, "stall", self.step1, self.gate())
        self.assertEqual(f["count"]["n_tail"], len(self.tail))
        self.assertIsNone(f["bank_only_reproduces_step1"])
        mine = dict(n_survivors=f["count"]["n_survivors"], n_tail=f["count"]["bank_only"]["n_tail"],
                    n_rest=f["count"]["bank_only"]["n_rest"], t_lo=f["count"]["t_lo"], t_up=f["count"]["t_up"])
        ok = F.seed_world_facts(self.log, self.gap, "stall", self.step1, self.gate(), dict(mine))
        self.assertTrue(ok["bank_only_reproduces_step1"])
        off = F.seed_world_facts(self.log, self.gap, "stall", self.step1, self.gate(), dict(mine, n_tail=-1))
        self.assertFalse(off["bank_only_reproduces_step1"])
        nt = N.no_training_checks([ok])
        self.assertTrue(nt["passed"], nt)
        self.assertEqual(nt["not_run"], [])
        self.assertEqual(len(nt["rules"]), 4)
        self.assertFalse(N.no_training_checks([off])["passed"])
        self.assertFalse(N.no_training_checks([f])["passed"])                # check 4 not run
        bad = json.loads(json.dumps(ok, default=float))
        bad["count"]["group_checks"]["passed"] = False
        self.assertFalse(N.no_training_checks([bad])["passed"])
        bad = json.loads(json.dumps(ok, default=float))
        bad["gate_checks"]["passed"] = False
        self.assertFalse(N.no_training_checks([bad])["passed"])

    def test_gate_checks_of(self):
        rec = dict(passed=True, checks=dict(gap_rows_in_gapped_training=0, gap_rows_in_gapped_bank=0, ok=True),
                   t1_t5=dict(T1=True, T2=True))
        self.assertTrue(F.gate_checks_of(rec, "stall")["passed"])
        self.assertFalse(F.gate_checks_of(dict(rec, t1_t5=dict(T1=False)), "stall")["passed"])
        self.assertTrue(F.gate_checks_of(dict(rec, t1_t5=dict(T1=False)), "no_stall")["passed"])
        self.assertFalse(F.gate_checks_of(dict(rec, checks=dict(gap_rows_in_gapped_training=1)), "stall")["passed"])
        self.assertEqual(F.GAP_RULE, K.GAP_RULE)


class TestReadoutHelpers(unittest.TestCase):
    def test_zones(self):
        z = N.throttle_zones(np.array([[0.69, 0, 0, 0], [0.7, 0, 0, 0], [0.8, 0, 0, 0], [0, -0.81, 0, 0]]))
        self.assertEqual(z.tolist(), [0, 1, 1, 2])

    def test_clip(self):
        self.assertEqual(N.clip_throttle(np.array([[0.9, -0.95, 0.5, 0.0]]), 0.8).tolist(),
                         [[0.8, -0.8, 0.5, 0.0]])

    def test_throttle(self):
        a = np.array([[0.1, -0.9, 0.2, 0.3]], np.float32)
        self.assertEqual(float(N.throttle(a)[0]), float(np.float32(0.9)))
        with self.assertRaises(ValueError):
            N.throttle(np.zeros((2, 3)))


class TestGuards(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(N.check_seeds([0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])
        for s in (5, 9, 10, 14, 15, 34, 35, 40, 49, 100, 900):
            with self.assertRaises(N.Refused):
                N.check_seeds([s])
        self.assertEqual(N.check_seeds([10, 14], allow_spent=True), [10, 14])
        with self.assertRaises(N.Refused):
            N.check_seeds([35], allow_spent=True)
        with self.assertRaises(N.Refused):
            F.main(["--out", "/nonexistent/x.json", "--seeds", "35"])

    def test_outputs(self):
        for p in (N.ROOT / "outputs" / "x.json", HERE / "x.json", N.ROOT / "runs" / "shift" / "x.json",
                  N.ROOT / "experiments" / "shift" / "x.json", N.ROOT / "runs" / "shift_explore" / "near80" / "x.json"):
            with self.assertRaises(N.Refused):
                N.guard_out(p)
        N.guard_out(N.ROOT / "runs" / "shift_explore" / "near80_v9" / "pre" / "facts.json")   # allowed, not created
        with tempfile.TemporaryDirectory() as d:
            N.guard_out(Path(d) / "facts.json")
            (Path(d) / "exists.json").write_text("{}")
            with self.assertRaises(N.Refused):
                N.guard_out(Path(d) / "exists.json")

    def test_step1lib_is_verified(self):
        mod = N.load_step1lib()
        self.assertTrue(hasattr(mod, "gap_rule") and hasattr(mod, "throttle") and hasattr(mod, "certificate"))

    def test_nothing_written_into_the_code_folder(self):
        """Only the planned Python files may live here (near80_constants.CODE_FILES); no output, cache or other file."""
        allowed = set(K.CODE_FILES)
        for p in HERE.iterdir():
            self.assertTrue(p.is_file() and p.suffix == ".py", f"not a Python file: {p.name}")
            self.assertIn(p.name, allowed)
        for name in K.CODE_FILES:
            self.assertTrue((HERE / name).is_file(), name)


def final_or_draft(root=HERE.parent):
    """('final', prereg.json) when it exists; else ('draft', the draft kept in prereg_history/); else ('draft',
    prereg_draft.json, before freeze.py moves it); else (None, None)."""
    p = root / "prereg.json"
    if p.is_file():
        return "final", json.loads(p.read_text(encoding="utf8"))
    hist = root / "prereg_history"
    kept = sorted(h for h in hist.glob("prereg_draft_*.json") if h.suffix == ".json") if hist.is_dir() else []
    if kept:
        return "draft", json.loads(kept[0].read_text(encoding="utf8"))
    p = root / "prereg_draft.json"
    if p.is_file():
        return "draft", json.loads(p.read_text(encoding="utf8"))
    return None, None


class TestFinalJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.which, cls.d = final_or_draft()
        if cls.d is None:
            raise unittest.SkipTest("neither prereg.json nor the draft it is built from is there")
        cls.doc = (HERE.parent / "PREREGISTRATION.md").read_text(encoding="utf8")
        cls.readme = (HERE.parent / "README.md").read_text(encoding="utf8")

    def test_json_matches_the_code(self):
        d = self.d
        if self.which == "final":
            self.assertEqual(d["kind"], K.PREREG_KIND)
            self.assertEqual(d["status"], K.STATUS_FINAL)
            self.assertEqual(d["status_line"], K.STATUS_LINE_FINAL)
            self.assertIs(d["approvable"], True)
            self.assertEqual(d["missing"], [])
            self.assertEqual(d["arms"]["logical_ids"], {k: list(v) for k, v in K.ARMS.items()})
            self.assertEqual(d["claims"]["block"], json.loads(json.dumps(K.CLAIMS, sort_keys=True)))
        else:
            self.assertEqual(d["status"], "draft")
            self.assertEqual(d["kind"], "near80v9-preregistration-draft")
        self.assertEqual(d["arms"]["main"], list(K.ARMS["main"]))
        self.assertEqual(sorted(d["arms"]["plain_twins"]), sorted(K.PLAIN_TWINS))
        self.assertEqual(d["seeds"]["exploratory"], list(K.EXPLORATORY))
        self.assertEqual(d["seeds"]["pool"], list(K.FRESH_POOL))
        self.assertEqual(d["seeds"]["n_fresh"], K.N_FRESH)
        self.assertEqual(d["situations"], list(K.SITUATIONS))
        c2 = K.CLAIMS["C2"]
        self.assertEqual(d["claims"]["C2_margins"], c2["margins"])
        for k in ("statistic", "alpha", "min_positive_seeds", "multiplicity", "noise_floor_max"):
            self.assertEqual(d["claims"][f"C2_{k}"], c2[k], k)
        self.assertEqual(d["claims"]["decision"], K.CLAIMS["decision"])

    def test_final_choices_constants_and_claims(self):
        if self.which != "final":
            self.skipTest("prereg.json is not written yet")
        d = self.d
        for c in d["choices"]:
            self.assertEqual(c["taken"], c["default"], c["number"])
            self.assertEqual(c["accepted"], K.OWNER_ANSWER["utc"])
        self.assertEqual(d["owner_answer"], K.OWNER_ANSWER)
        self.assertEqual(d["constants"], json.loads(json.dumps(K.as_dict(), sort_keys=True)))
        self.assertEqual(d["claims_word_for_word"]["source"], "PREREGISTRATION.md section 14")

    def test_choices_match_the_document(self):
        """Section 16 of PREREGISTRATION.md lists the same choices, in the same order."""
        import re
        sec = self.doc[self.doc.index("## 16."):self.doc.index("## 17.")]
        rows = re.findall(r"^\| (\d+)\. ([^|]+?) \|", sec, flags=re.M)
        self.assertEqual([int(n) for n, _ in rows], [c["number"] for c in self.d["choices"]])
        for (n, t), c in zip(rows, self.d["choices"]):
            self.assertTrue(t.lower().startswith(c["choice"].lower()), (t, c["choice"]))

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

    def test_documents_status(self):
        """The documents carry one status line, the draft's or the final one, matching near80lib.STATUS while it is
        a draft; a section 14.6 exists."""
        import freeze as FZ
        lines = self.doc.split("\n")
        self.assertEqual(lines[0], FZ.TITLE)
        self.assertTrue(any(line.startswith("### 14.6 ") for line in lines))
        if self.which == "final":
            bold = "**" + K.STATUS_LINE_FINAL + "**"
            for text in (self.doc, self.readme):
                self.assertNotIn("**DRAFT", text)
                self.assertNotIn("this draft", text.lower())
        else:
            bold = "**" + N.STATUS + "**"
        self.assertEqual(lines[2], bold)
        self.assertEqual(self.readme.split("\n")[2], bold)


if __name__ == "__main__":
    unittest.main(verbosity=2)
