"""Unit tests of the near-80 version 9 data code (the pooled groups, the outcome-free count, the training-file check,
the readout helpers, the seed registry), on synthetic logs only (no recorded seed is generated, loaded or read; no
learner is trained). The machine-readable draft and the documents are checked against the code.

Run (plain unittest), from the work folder, with NEAR80_HOST and NEAR80_ROOT set for a host other than the desktop:
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=<repo> \
    python <repo>/outputs/shift/beyond-the-data/near80/code/test_near80.py
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
import facts_near80 as F  # noqa: E402
from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

PATTERN = np.array([-1.0, -0.137, -0.585, -0.406])
SYNTHETIC_REP = 901          # a synthetic tag for the random streams; not a recorded data seed


def synthetic_log(rep=SYNTHETIC_REP, seed=7, sizes=(250, 50, 50), length=20, n_train=30, n_held=8):
    """A small synthetic log with the gate's Log interface: normal driving capped below 80%, a full-throttle cluster
    (inside the ball), survivors near full throttle outside the ball in the training split and in each held-out
    split. Returns (log, training survivors, held-out survivors per split)."""
    rng = np.random.default_rng(seed)
    E = int(sum(sizes))
    n = E * length
    mu = np.array([-0.55, -0.08, -0.35, -0.25])
    a = 0.8 * np.tanh((mu + 0.45 * rng.standard_normal((n, 4))) / 0.8)
    ft = rng.uniform(size=n) < 0.15
    a[ft] = np.clip(PATTERN + 0.03 * rng.standard_normal((int(ft.sum()), 4)), -1, 1)
    bounds = np.cumsum([0] + [s * length for s in sizes])
    survivors = {}
    for i, name in enumerate(("train", "calibration", "evaluation")):
        rows = np.arange(bounds[i], bounds[i + 1])
        k = n_train if i == 0 else n_held
        surv = rng.choice(rows[~ft[rows]], k, replace=False)
        a[surv] = np.array([-0.985, 0.4, 0.3, 0.3]) + 0.004 * rng.standard_normal((k, 4))
        survivors[name] = np.sort(surv)
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
    return DS.Log(arrays, meta), survivors["train"], survivors


def synthetic_gap():
    return SH.Gap(centre=tuple(float(v) for v in PATTERN), rho=0.3, mass=0.15, rule="base", q=0.15,
                  competing_ratio=2.0, record_sha256="synthetic")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log, cls.surv, cls.per_split = synthetic_log()
        cls.gap = synthetic_gap()
        cls.rm = SH.make_removal(cls.log, "gap", cls.gap)
        cls.step1 = N.load_step1lib()
        kept = cls.rm.kept_train_rows(cls.log)
        cls.gr = cls.step1.gap_rule(cls.step1.throttle(cls.log["actions"][kept]), **K.GAP_RULE)


# --------------------------------------------------------------------------------------------- the pooled groups

class TestPooledGroups(Base):
    def test_gap_rule_is_defined_on_the_synthetic_log(self):
        self.assertTrue(self.gr["defined"])
        self.assertGreater(self.gr["n_tail"], 9)

    def test_held_out_rows_are_the_two_splits_outside_the_ball(self):
        held = N.heldout_rows(self.log, self.gap)
        cal, ev = self.log.split_rows("calibration"), self.log.split_rows("evaluation")
        self.assertTrue(set(held) <= set(cal) | set(ev))
        self.assertFalse(self.gap.contains(self.log["actions"][held]).any())
        inside = self.gap.contains(self.log["actions"])
        self.assertEqual(len(held), int((~inside[cal]).sum() + (~inside[ev]).sum()))
        self.assertTrue(np.all(np.diff(held) > 0))

    def test_exact_row_sets(self):
        g = N.pooled_groups(self.log, self.gap, self.rm, self.gr["t_lo"], self.step1)
        t = N.throttle_index(self.log["actions"])
        kept = self.rm.kept_train_rows(self.log)
        want_surv = kept[t[kept] > self.gr["t_lo"]]
        self.assertTrue(np.array_equal(g["survivors"], want_surv))
        self.assertTrue(np.array_equal(g["survivors"], np.sort(self.per_split["train"])))
        held = N.heldout_rows(self.log, self.gap)
        self.assertTrue(np.array_equal(g["heldout_tail"], held[t[held] > self.gr["t_lo"]]))
        self.assertTrue(np.array_equal(g["heldout_rest"], held[t[held] <= self.gr["t_lo"]]))
        self.assertEqual(len(g["heldout_tail"]) + len(g["heldout_rest"]), len(held))
        self.assertEqual(g["n_full_throttle"], len(want_surv) + len(g["heldout_tail"]))
        self.assertEqual(g["n_rest"], len(g["heldout_rest"]))
        # the held-out survivors planted in the two splits are in the full-throttle group
        planted = np.sort(np.concatenate([self.per_split["calibration"], self.per_split["evaluation"]]))
        self.assertTrue(set(planted) <= set(g["heldout_tail"]))

    def test_no_row_from_a_scored_test_set_and_no_training_row_among_the_held_out(self):
        """The pooled rows are recorded rows of the calibration and evaluation splits; the scored tests are the four
        fixed test start sets (rollouts from thruster_chain.bank_starts), which no recorded row enters."""
        g = N.pooled_groups(self.log, self.gap, self.rm, self.gr["t_lo"], self.step1)
        train = set(self.log.split_rows("train"))
        self.assertFalse(train & set(g["heldout_tail"]))
        self.assertFalse(train & set(g["heldout_rest"]))
        self.assertTrue(set(g["survivors"]) <= train)
        starts = TC.bank_starts("strong_current", 5)
        self.assertEqual(np.asarray(starts).shape[-1], TC.DX)      # start states, not log rows

    def test_version8_groups_are_a_subset(self):
        rec = N.pooled_count_record(self.log, self.gap, self.rm, self.step1)
        v8, g = rec["version8_groups"], rec["groups"]
        self.assertLessEqual(v8["n_full_throttle"], g["n_full_throttle"])
        self.assertLessEqual(v8["n_rest"], g["n_rest"])
        bank = self.rm.bank_rows(self.log)
        self.assertTrue(set(bank) <= set(self.log.split_rows("calibration")))

    def test_deterministic_and_hashed(self):
        a = N.pooled_groups(self.log, self.gap, self.rm, self.gr["t_lo"], self.step1)
        b = N.pooled_groups(self.log, self.gap, self.rm, self.gr["t_lo"], self.step1)
        for k in ("survivors_sha256", "heldout_tail_sha256", "heldout_rest_sha256"):
            self.assertEqual(a[k], b[k])
            self.assertEqual(len(a[k]), 64)

    def test_outcome_free(self):
        """The count reads commands and split labels only: changing every reward and state leaves it unchanged."""
        lg2 = DS.Log({k: (v * 0 + 1 if k in ("rewards",) else v) for k, v in self.log._arrays.items()},
                     dict(self.log.meta))
        a = N.pooled_count_record(self.log, self.gap, self.rm, self.step1)
        b = N.pooled_count_record(lg2, self.gap, SH.make_removal(lg2, "gap", self.gap), self.step1)
        self.assertEqual(a["groups"], b["groups"])
        self.assertEqual(a["gap_rule"], b["gap_rule"])

    def test_certification_expectation(self):
        c = N.certification_expectation(29)
        self.assertTrue(c["reaches_min_rows"])
        self.assertGreater(c["expected_finite_share"], 0.95)
        self.assertFalse(N.certification_expectation(28)["reaches_min_rows"])
        self.assertAlmostEqual(N.certification_expectation(19)["expected_finite_share"], 1 - 0.9 ** 19)
        self.assertEqual(N.min_rows_for(), 29)
        self.assertEqual(K.CERT_MIN_ROWS, 29)

    def test_count_record_without_a_gap_rule(self):
        """A log whose ball-kept commands give no gap rule: the record says so and holds no groups."""
        arrays = dict(self.log._arrays)
        a = np.asarray(arrays["actions"]).copy()
        a[:, :] = np.clip(a, -0.5, 0.5)                 # every throttle index equal: no gap
        arrays["actions"] = a.astype(np.float32)
        lg = DS.Log(arrays, dict(self.log.meta))
        rec = N.pooled_count_record(lg, self.gap, SH.make_removal(lg, "gap", self.gap), self.step1)
        self.assertFalse(rec["defined"])
        self.assertIsNone(rec["groups"])


# --------------------------------------------------------------------------------------------- the training file

class TestTrainingFile(Base):
    def test_matches_the_kept_rows(self):
        want = DS.d4rl_dict(self.log, self.rm.kept_train_rows(self.log))
        c = N.training_file_check(want, self.log, self.rm)
        self.assertTrue(c["passed"])

    def test_catches_a_changed_or_missing_row(self):
        want = DS.d4rl_dict(self.log, self.rm.kept_train_rows(self.log))
        bad = dict(want)
        a = np.array(bad["actions"])
        a[0, 0] = np.nextafter(a[0, 0], 1.0)
        bad["actions"] = a
        self.assertFalse(N.training_file_check(bad, self.log, self.rm)["passed"])
        short = {k: v[:-1] for k, v in want.items()}
        self.assertFalse(N.training_file_check(short, self.log, self.rm)["passed"])


# --------------------------------------------------------------------------------------------- readout helpers

class TestReadoutHelpers(unittest.TestCase):
    def test_zones(self):
        z = N.throttle_zones(np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1], [0.81, 0.0, 0.0, 0.0],
                                       [np.float32(0.8), 0, 0, 0]]))
        self.assertEqual(z.tolist(), [0, 1, 2, 1])

    def test_clip(self):
        c = N.clip_throttle(np.array([[0.9, -0.9, 0.5, 0.0]]), 0.7)
        self.assertEqual(c.tolist(), [[0.7, -0.7, 0.5, 0.0]])

    def test_recorded_range(self):
        r = N.recorded_range(np.random.default_rng(1).uniform(-1, 1, (1000, 4)))
        self.assertEqual(len(r), 4)
        self.assertTrue(all(lo < hi for lo, hi in r))

    def test_throttle_index_matches_step1(self):
        s1 = N.load_step1lib()
        a = np.random.default_rng(2).uniform(-1, 1, (50, 4)).astype(np.float32)
        self.assertTrue(np.array_equal(N.throttle_index(a), s1.throttle(a)))


# --------------------------------------------------------------------------------------------- guards and seeds

class TestGuardsAndSeeds(unittest.TestCase):
    def test_facts_seeds(self):
        self.assertEqual(N.check_seeds([0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])
        for bad in ([5], [10], [40], [50], [99]):
            with self.assertRaises(N.Refused):
                N.check_seeds(bad)
        self.assertEqual(N.check_spent_seeds([10, 11, 12, 13, 14]), [10, 11, 12, 13, 14])
        self.assertEqual(N.check_spent_seeds([]), [])
        for bad in ([0], [5], [15], [40], [50], [99]):
            with self.assertRaises(N.Refused):
                N.check_spent_seeds(bad)

    def test_certify_chance_is_exact(self):
        """P(Binomial(1000, 0.9^n) <= 50): 0 up to 24 rows, 0.405 at 28, 0.700 at 29, 0.896 at 30, 0.976 at 31,
        0.9995 at 33 (the preview's table, reproduced from the draws)."""
        for n, want in ((10, 0.0), (24, 0.0001), (28, 0.405), (29, 0.700), (30, 0.896), (31, 0.976), (33, 0.9995),
                        (40, 1.0)):
            self.assertAlmostEqual(K.certify_chance(n), want, places=3, msg=n)
        self.assertEqual(K.certify_chance(0), 0.0)
        c = N.certification_expectation(29)
        self.assertAlmostEqual(c["certify_chance"], 0.700, places=3)
        self.assertEqual(K.c2_positive_needed(10), 8)

    def test_fresh_seeds_are_unused_by_every_registry(self):
        """The fresh seeds lie in the block 40-59, which no registry names: not 0-4, 5-9, 10-14, 15-19, the placement
        test's 20-34, the gate's 100-119 or stage 0's 900-909 (near80_constants.SEED_REGISTRY, from the repository
        and the seed_registry.json of claude/placement-test-design)."""
        used = set(K.USED_SEEDS)
        for s in K.FRESH:
            self.assertNotIn(s, used)
            self.assertIsNone(K.seed_holder(s))
            self.assertTrue(K.FRESH_SEED_BLOCK[0] <= s <= K.FRESH_SEED_BLOCK[1])
        for span in K.SEED_REGISTRY:
            a, b = (int(x) for x in span.split("-"))
            self.assertTrue(set(range(a, b + 1)) <= used)
        self.assertEqual(sorted(K.FRESH), list(K.FRESH))
        self.assertEqual(K.group_of(0), "smoke")
        for s in (5, 9, 10, 14, 15, 19, 20, 25, 30, 34, 100, 119, 900, 909):
            with self.assertRaises(ValueError) as cm:
                K.group_of(s)
            self.assertIn("belongs to", str(cm.exception))
        with self.assertRaises(ValueError):
            K.group_of(60)
        with self.assertRaises(ValueError):
            K.group_of(1)

    def test_outputs(self):
        d = Path(tempfile.mkdtemp(prefix="near80_test_"))
        p = N.guard_out(d / "x.json")
        self.assertEqual(p, (d / "x.json").resolve())
        (d / "y.json").write_text("{}")
        with self.assertRaises(N.Refused):
            N.guard_out(d / "y.json")
        with self.assertRaises(N.Refused):
            N.guard_out(Path(K.ROOT) / "outputs" / "x.json")

    def test_step1lib_is_verified(self):
        s1 = N.load_step1lib()
        self.assertTrue(callable(s1.gap_rule) and callable(s1.certificate))
        self.assertEqual(Path(s1.ROOT), Path(K.ROOT))

    def test_nothing_written_into_the_code_folder(self):
        self.assertFalse((HERE / "__pycache__").exists())


# --------------------------------------------------------------------------------------------- the draft and documents

def final_or_draft(root=HERE.parent):
    """('final', prereg.json of this version) when it exists; else ('draft', the draft kept in prereg_history/);
    else ('draft', prereg_draft.json); else (None, None)."""
    p = root / "prereg.json"
    if p.is_file():
        d = json.loads(p.read_text(encoding="utf8"))
        if d.get("version") == K.VERSION:
            return "final", d
    hist = root / "prereg_history"
    if hist.is_dir():
        for h in sorted(hist.glob("prereg_draft_*.json")):
            if h.suffix == ".json":
                d = json.loads(h.read_text(encoding="utf8"))
                if d.get("version") == K.VERSION:
                    return "draft", d
    p = root / "prereg_draft.json"
    if p.is_file():
        return "draft", json.loads(p.read_text(encoding="utf8"))
    return None, None


class TestDraftJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.which, cls.d = final_or_draft()
        if cls.d is None:
            raise unittest.SkipTest("neither prereg.json nor the draft it is built from is there")
        cls.doc = (HERE.parent / "PREREGISTRATION.md").read_text(encoding="utf8")
        cls.readme = (HERE.parent / "README.md").read_text(encoding="utf8")

    def test_json_matches_the_code(self):
        d = self.d
        self.assertEqual(d["version"], K.VERSION)
        self.assertEqual(d["label"], K.LABEL)
        self.assertEqual(d["seeds"]["fresh"], list(K.FRESH))
        self.assertEqual(d["seeds"]["fresh_block"], list(K.FRESH_SEED_BLOCK))
        self.assertEqual(d["seeds"]["plumbing"], list(K.SMOKE_SEEDS))
        self.assertEqual(d["pooled_groups"]["splits"], list(K.POOLED_SPLITS))
        self.assertEqual(sorted(d["pass_marks"]["c2_twins"]["twins"]), sorted(K.TWIN_ARMS))
        self.assertEqual(d["kill_rules"]["noise_floor"]["rule"].split("<=")[-1].strip(),
                         f"{K.MARKS['noise_floor_max']:g}")
        if self.which == "final":
            self.assertEqual(d["status"], K.STATUS_FINAL)
            self.assertEqual(d["arms"]["logical_ids"], {k: list(v) for k, v in K.ARMS.items()})
            self.assertEqual(d["pass_marks"]["c2_twins"]["net_margin"], K.MARKS_C2["net_margin"])
        else:
            self.assertEqual(d["status"], "draft")
        self.assertEqual(d["pass_marks"]["c2_twins"]["net_margin"], K.MARKS_C2["net_margin"])
        self.assertEqual(d["pass_marks"]["c2_twins"]["positive_share"], K.MARKS_C2["positive_share"])
        self.assertEqual(d["pass_marks"]["c3_mirror"]["net_margin"], K.MARKS_C3["net_margin"])
        self.assertEqual(d["pass_marks"]["min_certified"], K.MIN_CERTIFIED)
        self.assertEqual(d["seeds"]["fresh_count"], len(K.FRESH))

    def test_choices_match_the_document(self):
        import re
        sec = self.doc[self.doc.index("## 16."):self.doc.index("## 17.")]
        titles = re.findall(r"^(\d+)\. \*\*([^*]+?)\.?\*\*", sec, flags=re.M)
        self.assertEqual([int(n) for n, _ in titles], list(range(1, len(self.d["choices"]) + 1)))
        for (n, t), c in zip(titles, self.d["choices"]):
            self.assertEqual(t.strip().rstrip(".").lower(), c["choice"].lower(), n)

    def test_document_brackets_are_defined(self):
        import re
        sec = self.doc[self.doc.index("## 14."):self.doc.index("## 15.")]
        table = sec[:sec.index("**Always attached:**")]
        defined = set()
        for row in re.findall(r"^\| ([^|]+) \|", table, flags=re.M):
            defined.update(x.strip() for x in re.split(r",| and ", row) if x.strip())
        used = set(re.findall(r"\[([A-Z][A-Za-z0-9]*)\]", sec[len(table):]))
        self.assertTrue(used <= defined, used - defined)

    def test_documents_status_lines(self):
        """A draft carries the draft status line and its TODO marks; a final version the final line and none."""
        lines = self.doc.split("\n")
        self.assertTrue(lines[0].startswith("# The main setting on fresh seeds: pre-registration (version 9"))
        self.assertTrue(any(line.startswith("### 14.5 ") for line in lines))
        if self.which == "final":
            bold = "**" + K.STATUS_LINE_FINAL + "**"
            self.assertEqual(lines[2], bold)
            self.assertEqual(self.readme.split("\n")[2], bold)
            self.assertNotIn("TODO(DECISION", self.doc)
            self.assertEqual(K.placeholders_present(), [])
        else:
            bold = "**" + K.STATUS_LINE_DRAFT + "**"
            self.assertEqual(lines[2], bold)
            self.assertEqual(self.readme.split("\n")[2], bold)
            self.assertIn("TODO(FREEZE", self.doc)                 # the facts the freeze fills are the open item
            self.assertNotIn("TODO(OWNER", self.doc)               # the owner answered on 20 October 2026
            self.assertNotIn("TODO(DECISION", self.doc)            # DECISION.md fixed every value
            self.assertEqual(K.placeholders_present(), [])
        self.assertEqual(N.STATUS, K.STATUS_LINE_DRAFT)


class TestFacts(Base):
    def test_switch_calls_and_kill_rule(self):
        """facts_near80.switch_calls_of restates step 2's and policy_phase's switch rules on a certificate entry; the
        kill rule compares BCA's and the mean-difference switch's calls only."""
        entry = dict(certificate=dict(on=True, height=2.5, q05=0.4, finite_share_tail=0.99, tail_certified=True),
                     groups=dict(n_full_throttle=60, n_rest=42000, mean_full_throttle=0.6, mean_rest=0.0,
                                 q90_full_throttle=1.2, q90_rest=-0.04), rank_test=dict(p=1e-6))
        c = F.switch_calls_of(entry)
        self.assertEqual((c["bca"], c["meandiff"], c["conformal"], c["rank"], c["bca_refused"]),
                         (True, True, True, True, False))
        self.assertAlmostEqual(c["plain_gap"], 1.24)
        self.assertAlmostEqual(c["twice_plain_gap"], 2.48)
        refused = dict(entry, certificate=dict(entry["certificate"], height="inf", tail_certified=False))
        self.assertTrue(F.switch_calls_of(refused)["bca_refused"])
        off = dict(entry, certificate=dict(on=False, height=0.1), groups=dict(entry["groups"], mean_full_throttle=0.05,
                                                                            q90_full_throttle=0.05),
                   rank_test=dict(p=0.3))
        o = F.switch_calls_of(off)
        self.assertEqual((o["bca"], o["meandiff"], o["conformal"], o["rank"]), (False, False, False, False))
        kr = F.kill_rule(c, F.switch_calls_of(refused))
        self.assertTrue(kr["passed"])                       # the same calls (on), whatever the heights
        kr = F.kill_rule(o, c)
        self.assertEqual((kr["bca"], kr["meandiff"], kr["passed"]), (False, False, False))
        rows = [dict(seed=10, world="stall", kill_rule=dict(passed=True), pooled=dict(calls=c),
                     recorded=dict(calls=F.switch_calls_of(refused))),
                dict(seed=10, world="no_stall", kill_rule=dict(passed=True), pooled=dict(calls=o),
                     recorded=dict(calls=o))]
        sm = F.spent_summary(rows)
        self.assertEqual((sm["seed_worlds"], sm["kill_rule_passed"], sm["bca_refused_recorded"],
                          sm["bca_refused_pooled"], sm["meandiff_made_bcas_call_pooled"]), (2, True, 1, 0, 2))

    def test_seed_world_counts_record(self):
        rec = F.seed_world_counts(self.log, self.gap, "stall", self.rm, self.step1)
        self.assertEqual(rec["world"], "stall")
        self.assertEqual(rec["rep"], SYNTHETIC_REP)
        self.assertTrue(rec["defined"])
        self.assertEqual(rec["groups"]["n_full_throttle"] + rec["groups"]["n_rest"] - rec["groups"]["n_survivors"],
                         rec["groups"]["n_heldout"])
        s = F.summary([rec, dict(rec, world="no_stall")])
        self.assertEqual(s["stall"]["seed_worlds"], 1)
        self.assertEqual(s["no_stall"]["n_full_throttle_min"], rec["groups"]["n_full_throttle"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
