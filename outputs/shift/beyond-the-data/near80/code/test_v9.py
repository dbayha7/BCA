"""Near-80 (version 9, main setting): the tests DECISION.md asked for by name (BUILD_LIST.md section 7). DRAFT, not
frozen.

  TestPooledGroups        the pooled groups' exact row sets on a synthetic log: every held-out row outside the ball
                          exactly once, no training row other than a survivor, no ball row, and the outcome-free
                          count equal to the sizes the switch scores (with a stand-in critic that takes no gradient
                          step, through learner_phases.switch_scores_pooled)
  TestRefusalReading      the look's two readings of a refusal: C1 substitutes the no-caution actor for BCA's arms
                          and the unmatched rivals; C2 and C3 leave the seed out (the certified seeds)
  TestTwinRules           the sign thresholds 8 / 8 / 7 (n = 10 / 9 / 8) and at most 1; a 4.9 mean is a tie; the
                          lower bound; not read below 8; the noise floor drops both
  TestLeadRivals          the hand-set range limit in the lead's rival set; step 2's own lead reported beside; a
                          refusal lowers BCA's pooled net by its share
  TestSentences           the new anchors extract; the C1 last-sentence rule; the certified sentence; C3's sentences;
                          the refused clause names the seed
  TestConstantsAgree      approve_local's V8_* equal the constants'; TWIN_ARMS, MARKS_C2 / C3, FRESH (the block's first ten); the one
                          owner's answer recorded; CAP_CPU_HOURS 55; the cluster scripts in TOP_FILES and the folder
  TestCertificationChance the exact binomial chance by group size
  TestStubRefuse          the stub run (nothing trained) on the plumbing's seed 0, refusing: the look reads C1 with
                          the substitution (every claim 'invalid': the stand-ins score 0) and writes the refusal
                          sentences; the refusal is recorded per arm
                          (POSIX only; a few minutes)

Everything runs on synthetic data in temporary folders outside the repository; nothing is trained, no recorded seed
is generated, loaded or read.
"""
import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402
import claims_text as CT  # noqa: E402

ROOT = Path(K.ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

P = K.load_local("pooled")


def load_test_module(name):
    spec = importlib.util.spec_from_file_location(f"near80_v9_{name}", str(HERE / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TP = load_test_module("test_pooled")        # make_tree, default_points, override, expect, T, BLOCK, SEEDS
SEEDS = TP.SEEDS
SC = TP.SC
TWIN1, TWIN2, TWIN3 = K.TWIN_ARMS
POSIX = os.name == "posix"
WORK = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()
N80 = HERE.parent                           # the folder this file lives in: the real one, or the freeze's as-frozen copy


def draft_text():
    """The version 9 draft's text: prereg_draft.json at the top of the folder, or, once the freeze has moved it, the
    one kept in prereg_history/ (test_near80.final_or_draft and test_approval.real_draft_bytes read it the same way)."""
    top = N80 / K.DRAFT_NAME
    if top.is_file():
        return top.read_text(encoding="utf8")
    hist = N80 / K.HISTORY_DIR
    for p in sorted(hist.glob(K.DRAFT_HISTORY_NAME.format(prefix="*"))) if hist.is_dir() else []:
        if p.suffix == ".json":
            text = p.read_text(encoding="utf8")
            if json.loads(text).get("version") == K.VERSION:
                return text
    raise FileNotFoundError(f"no version 9 draft at {top} or in {hist}")


def draft_payload():
    return json.loads(draft_text())


def document_is_a_draft():
    """Line 3 of PREREGISTRATION.md is the DRAFT status line (else the folder is finalized or frozen)."""
    return (N80 / "PREREGISTRATION.md").read_text(encoding="utf8").split("\n")[2] == f"**{K.STATUS_LINE_DRAFT}**"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="near80_test_v9_"))
        self.assertFalse(P.inside(self.tmp, K.ROOT))
        self._fresh = mock.patch.object(K, "FRESH", K.TEST_LABELS)      # the synthetic fresh group (no reserved seed)
        self._fresh.start()

    def tearDown(self):
        self._fresh.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def look(self, spec=None, sub="a"):
        spec = dict(spec or {})
        root = TP.make_tree(self.tmp / sub, spec)
        R = P.load_results(root, spec.get("group", "fresh"), "scratch")
        return P.make_look(R, TP.BLOCK)

    def text_of(self, look, sid):
        rows = [r["text"] for r in look["sentences"] if r["id"] == sid]
        self.assertEqual(len(rows), 1, (sid, [r["id"] for r in look["sentences"]]))
        return rows[0]


# --------------------------------------------------------------------------------------------- the pooled groups

class TestPooledGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import near80lib as N
        from experiments.shift import shift as SH
        T80 = load_test_module("test_near80")
        cls.N = N
        cls.log, cls.surv, cls.per_split = T80.synthetic_log()
        cls.gap = T80.synthetic_gap()
        cls.rm = SH.make_removal(cls.log, "gap", cls.gap)
        cls.step1 = N.load_step1lib()
        kept = cls.rm.kept_train_rows(cls.log)
        cls.gr = cls.step1.gap_rule(cls.step1.throttle(cls.log["actions"][kept]), **K.GAP_RULE)
        cls.g = N.pooled_groups(cls.log, cls.gap, cls.rm, cls.gr["t_lo"], cls.step1)

    def test_every_held_out_row_outside_the_ball_exactly_once(self):
        import numpy as np
        held = np.concatenate([self.log.split_rows(s) for s in K.POOLED_SPLITS])
        inside = self.gap.contains(self.log["actions"][held])
        outside = np.sort(held[~inside])
        pooled = np.concatenate([self.g["heldout_tail"], self.g["heldout_rest"]])
        self.assertEqual(len(pooled), len(set(pooled.tolist())))             # exactly once
        self.assertTrue(np.array_equal(np.sort(pooled), outside))              # every one, and no other
        self.assertFalse(self.gap.contains(self.log["actions"][pooled]).any())  # no ball row

    def test_no_training_row_other_than_a_survivor(self):
        import numpy as np
        train = set(self.log.split_rows("train").tolist())
        kept = set(self.rm.kept_train_rows(self.log).tolist())
        self.assertFalse(train & set(self.g["heldout_tail"].tolist()))
        self.assertFalse(train & set(self.g["heldout_rest"].tolist()))
        surv = set(self.g["survivors"].tolist())
        self.assertTrue(surv <= kept)                                          # survivors are kept training rows
        t = self.N.throttle_index(self.log["actions"])
        self.assertTrue(all(t[s] > self.gr["t_lo"] for s in surv))
        self.assertEqual(surv, set(np.sort(self.surv).tolist()))              # the planted ones, exactly

    def test_count_equals_the_sizes_the_switch_scores(self):
        """The outcome-free count of the data phase equals the sizes learner_phases.switch_scores_pooled scores, with
        a critic that took no gradient step (the stub's initial critic): nothing trained."""
        import numpy as np
        import learner_phases as LP
        import stubs
        from experiments.shift import learners as L
        import settingdata as SD

        class SD_:
            log, gap = self.log, self.gap
        sd = SD.SeedView(901, "stall", {}, self.log, self.gap, self.log.spec, self.log.physics)
        learn = L.LearnSettings()
        pg = L.prepare(self.log, self.rm, L.normalization(self.log, self.rm), learn)
        gr = LP.gap_rule_of(pg)
        self.assertTrue(gr["defined"])
        critic = stubs._initial_critic(901, learn)
        tail, rest, detail = LP.switch_scores_pooled(critic, sd, pg, gr, learn.gamma)
        rec = self.N.pooled_count_record(self.log, self.gap, self.rm, self.step1)
        self.assertEqual((rec["groups"]["n_full_throttle"], rec["groups"]["n_rest"]), (len(tail), len(rest)))
        self.assertEqual(len(detail["tail_scores_sorted"]), len(tail))
        self.assertTrue(np.all(np.diff(detail["tail_scores_sorted"]) >= 0))
        self.assertEqual(detail["rest_scores_summary"]["n"], len(rest))
        entry = LP.certificate_entry(tail, rest, 901, "stall", gr)
        self.assertEqual(entry["groups"]["n_full_throttle"], rec["groups"]["n_full_throttle"])
        self.assertIn("certify_chance", rec["certification"])
        # the scores depend on the critic, the count does not
        self.assertEqual(rec["groups"], self.N.pooled_count_record(self.log, self.gap, self.rm, self.step1)["groups"])


# --------------------------------------------------------------------------------------------- the refusal reading

class TestRefusalReading(Base):
    def test_apply_failures_substitutes_none_in_c1(self):
        """BCA refused in the stalling world: there its own arms and the controls matched to its strength (unmatched)
        take the no-caution actor's numbers; in the no-stall world of the same seed BCA's own height is finite, so its
        switched ramp, the ramp at twice its height and the lone copy trained and stand, while the always-on ramp
        (the stalling world's height) and the controls locked from the stalling world's matching have no actor and
        take the no-caution actor's numbers."""
        root = TP.make_tree(self.tmp / "a", dict(refused={SEEDS[2]: ["stall"]}))
        R = P.load_results(root, "fresh", "scratch")
        self.assertEqual(R.certified, [s for s in SEEDS if s != SEEDS[2]])
        raw_s, m_s = R.raw["points"]["main"]["stall"][SEEDS[2]], R.m["points"]["main"]["stall"][SEEDS[2]]
        for a in K.BCA_DEPENDENT + P.MATCHED_RIVALS:
            self.assertIsNone(raw_s[a], a)                                   # the record holds null
            self.assertEqual(m_s[a], raw_s["none"], a)                       # C1 reads the no-caution actor
        self.assertEqual(m_s[TWIN1], raw_s[TWIN1])                           # the twins' numbers stand
        raw_n, m_n = R.raw["points"]["main"]["no_stall"][SEEDS[2]], R.m["points"]["main"]["no_stall"][SEEDS[2]]
        for a in ("obca_k1", "obca_k2", K.NOISE_ARM):
            self.assertIsNotNone(raw_n[a], a)
            self.assertEqual(m_n[a], raw_n[a], a)
        for a in ("obca_forced",) + P.MATCHED_RIVALS:
            self.assertIsNone(raw_n[a], a)
            self.assertEqual(m_n[a], raw_n["none"], a)
        for s in SEEDS:
            if s != SEEDS[2]:
                for w in K.WORLDS:
                    self.assertEqual(R.m["points"]["main"][w][s], R.raw["points"]["main"][w][s])
        self.assertEqual(R.substituted[SEEDS[2]]["stall"], list(K.BCA_DEPENDENT) + list(P.MATCHED_RIVALS))
        self.assertEqual(R.substituted[SEEDS[2]]["no_stall"], ["obca_forced"] + list(P.MATCHED_RIVALS))
        # the twin differences read n - 1 seeds
        d = P.twin_difference(R.m["points"], TWIN1, R.certified)
        self.assertEqual(d["n"], len(SEEDS) - 1)
        self.assertNotIn(SEEDS[2], d["seeds"])

    def test_a_refusal_in_one_world_substitutes_that_world_only_where_null(self):
        """BCA refused in the no-stall world only: its stalling-world arms trained and stand (the gain of 30 there is
        BCA's own), the matching done in the stalling world stands in both worlds, the always-on ramp (the stalling
        world's finite height) trained in both; only the no-stall world's three BCA arms take the no-caution actor's
        numbers, so the seed's net is the stalling gain minus no caution's own loss to itself, 0."""
        root = TP.make_tree(self.tmp / "b", dict(refused={SEEDS[0]: ["no_stall"]}))
        R = P.load_results(root, "fresh", "scratch")
        self.assertIn(SEEDS[0], R.refused)
        self.assertNotIn(SEEDS[0], R.certified)
        self.assertEqual(R.substituted[SEEDS[0]]["no_stall"], ["obca_k1", "obca_k2", K.NOISE_ARM])
        self.assertEqual(R.substituted[SEEDS[0]]["stall"], [])
        raw_s, m_s = R.raw["points"]["main"]["stall"][SEEDS[0]], R.m["points"]["main"]["stall"][SEEDS[0]]
        for a in K.BCA_DEPENDENT + P.MATCHED_RIVALS:
            self.assertIsNotNone(raw_s[a], a)
            self.assertEqual(m_s[a], raw_s[a], a)
        raw_n, m_n = R.raw["points"]["main"]["no_stall"][SEEDS[0]], R.m["points"]["main"]["no_stall"][SEEDS[0]]
        self.assertIsNotNone(raw_n["obca_forced"])
        for a in P.MATCHED_RIVALS:
            self.assertIsNotNone(raw_n[a], a)
        for a in ("obca_k1", "obca_k2", K.NOISE_ARM):
            self.assertIsNone(raw_n[a], a)
            self.assertEqual(m_n[a], raw_n["none"], a)
        nets = P.net_per_seed(R.m["points"], K.PRIMARY, R.seeds)
        self.assertAlmostEqual(nets[0], 30.0)
        texts = [r["text"] for r in P.make_look(R, TP.BLOCK)["sentences"] if r["id"] == "stop.refused"]
        self.assertEqual(len(texts), 1)
        self.assertTrue(texts[0].startswith(f"Seed {SEEDS[0]}, no-stall world"))

    def test_the_refused_seed_is_not_a_failed_seed(self):
        look = self.look(dict(refused={SEEDS[1]: ["stall"]}))
        self.assertEqual(look["failed_checks"], {})
        self.assertEqual(look["outcomes"]["c1"], "pass")
        self.assertEqual(look["marks"]["c2"]["n_certified"], len(SEEDS) - 1)
        self.assertEqual(look["outcomes"]["c2"], "pass")
        self.assertIn("stop.refused", [r["id"] for r in look["sentences"]])


# --------------------------------------------------------------------------------------------- the twin rules

class TestTwinRules(Base):
    def test_thresholds_by_n(self):
        self.assertEqual([K.c2_positive_needed(n) for n in (10, 9, 8)], [8, 8, 7])
        self.assertEqual(K.MARKS_C3["positive_seeds_max"], 1)
        for n, kmin in ((10, 8), (9, 8), (8, 7)):
            d = dict(mean=6.0, positive=kmin)
            self.assertEqual(P.twin_reading(d, n)[0], "ahead")
            self.assertEqual(P.twin_reading(dict(mean=6.0, positive=kmin - 1), n)[0], "tie")
            self.assertEqual(P.twin_reading(dict(mean=-6.0, positive=1), n)[0], "behind")
            self.assertEqual(P.twin_reading(dict(mean=-6.0, positive=2), n)[0], "tie")
        self.assertEqual(P.twin_reading(dict(mean=4.9, positive=10), 10)[0], "tie")
        self.assertEqual(P.twin_reading(dict(mean=5.0, positive=10), 10)[0], "ahead")
        self.assertEqual(P.twin_reading(dict(mean=-4.9, positive=0), 10)[0], "tie")
        self.assertEqual(P.twin_reading(dict(mean=-5.0, positive=0), 10)[0], "behind")
        self.assertIsNone(P.twin_reading(dict(mean=None, positive=0), 10)[0])

    def test_lower_bound(self):
        import statistics
        xs = [12.0, 8.0, 10.0, 14.0, 6.0, 11.0, 9.0, 13.0, 7.0, 10.0]
        lb = P.lower_bound_95(xs)
        self.assertAlmostEqual(lb, statistics.mean(xs) - P.T_95[9] * statistics.stdev(xs) / 10 ** 0.5)
        self.assertLess(lb, statistics.mean(xs))
        self.assertIsNone(P.lower_bound_95([3.0]))
        self.assertIsNone(P.lower_bound_95([None, None]))

    def test_not_read_below_min_certified(self):
        look = self.look(dict(refused={s: ["stall"] for s in SEEDS[:3]}))
        self.assertEqual(look["marks"]["c2"]["n_certified"], len(SEEDS) - 3)
        self.assertEqual(look["outcomes"]["c2"], "not_read")
        look = self.look(dict(refused={s: ["stall"] for s in SEEDS[:2]}), sub="b")
        self.assertEqual(look["marks"]["c2"]["n_certified"], K.MIN_CERTIFIED)
        self.assertEqual(look["outcomes"]["c2"], "pass")
        self.assertEqual(look["marks"]["c2"]["positive_needed"], 7)

    def test_noise_floor_drops_both(self):
        pts = TP.override(TP.default_points, [("stall", K.NOISE_ARM, SC, None, 74.9)])
        look = self.look(dict(points=pts))
        self.assertAlmostEqual(look["validity"]["noise_floor"]["V"], 5.1)
        self.assertEqual((look["outcomes"]["c2"], look["outcomes"]["c3"]), ("dropped", "dropped"))
        pts = TP.override(TP.default_points, [("stall", K.NOISE_ARM, SC, None, 75.0)])
        look = self.look(dict(points=pts), sub="b")
        self.assertAlmostEqual(look["validity"]["noise_floor"]["V"], 5.0)
        self.assertNotEqual(look["outcomes"]["c2"], "dropped")                 # within 5 either way

    def test_c2_side_is_the_lone_copy(self):
        pts = TP.override(TP.default_points, [("stall", K.PRIMARY, SC, None, 84.0)])     # the batch copy moves
        look = self.look(dict(points=pts))
        self.assertAlmostEqual(look["marks"]["c2"]["twins"][TWIN1]["difference"], 12.0)   # unchanged: obca_k1_alone
        self.assertAlmostEqual(look["reported"]["twins"][TWIN1]["against_batch"]["mean"], 17.0)


# --------------------------------------------------------------------------------------------- the lead's rivals

class TestLeadRivals(Base):
    def test_range_hinge_in_the_lead(self):
        self.assertEqual(K.LEAD_RIVALS, ("standin", "even_pull", "random_1", "random_2", "obca_forced", "range_hinge"))
        look = self.look()
        nl = look["marks"]["c1"]["net_lead"]
        self.assertIn("range_hinge", nl["rival_nets"])
        self.assertEqual(nl["step2_lead"]["rivals"], list(K.WORLD_BLIND))
        self.assertNotIn("range_hinge", nl["step2_lead"]["rivals"])
        pts = TP.override(TP.default_points, [("stall", "range_hinge", SC, None, 85.0), ("no_stall", "range_hinge", SC,
                                                                                          None, 90.0)])
        look = self.look(dict(points=pts), sub="b")
        nl = look["marks"]["c1"]["net_lead"]
        self.assertEqual(nl["best_rival"], "range_hinge")
        self.assertAlmostEqual(nl["lead"], 29.0 - 35.0)
        self.assertFalse(nl["passed"])
        self.assertTrue(nl["step2_lead"]["passed"])                           # reported beside, not a mark

    def test_a_refusal_lowers_the_pooled_net_by_its_share(self):
        look = self.look(dict(refused={SEEDS[5]: ["stall", "no_stall"]}))
        nl = look["marks"]["c1"]["net_lead"]
        self.assertAlmostEqual(nl["net"], 29.0 * (len(SEEDS) - 1) / len(SEEDS))
        self.assertAlmostEqual(nl["per_seed"][5], 0.0)
        # the hand-set range limit does not depend on BCA: its net is untouched by the refusal; the matched rivals
        # and the always-on ramp take the no-caution actor's numbers there (a net of 0 on that seed)
        self.assertAlmostEqual(nl["rival_nets"]["range_hinge"], -20.0)


# --------------------------------------------------------------------------------------------- the sentences

class TestSentences(Base):
    def test_new_anchors_extract(self):
        for sid in ("c1.clause.stopped_seed", "c1.last.numbers", "c1.last.stop", "c1.certified", "c2.clause.ahead",
                    "c2.clause.tie", "c2.clause.behind", "c2.last.dose", "c2.last.none", "c2.last.twin",
                    "c2.last.other", "c3.holds", "c3.fails", "c2.not_read"):
            self.assertIn(sid, TP.BLOCK["sentences"])
            self.assertTrue(TP.BLOCK["sentences"][sid]["text"])
        names = set(CT.table_bracket_names(TP.BLOCK))
        for n in ("K", "F", "NC", "DC", "RC", "LC", "EC", "LB1", "LB2", "LB3", "LBT", "KMIN", "A"):
            self.assertIn(n, names)

    def test_c1_last_sentence_rule(self):
        pts = TP.override(TP.default_points, [("stall", "even_pull", SC, None, 70.0)])
        look = self.look(dict(points=pts))
        self.assertTrue(self.text_of(look, "c1.fail").endswith(TP.T["c1.last.numbers"]))
        look = self.look(dict(data_main_fail={SEEDS[1]: ("x", "leak")}), sub="b")
        self.assertTrue(self.text_of(look, "c1.fail").endswith(TP.T["c1.last.stop"]))

    def test_certified_sentence_always_after_c1(self):
        for spec, sub in ((None, "a"), (dict(refused={SEEDS[0]: ["stall"]}), "b"),
                          (dict(data_main_fail={SEEDS[1]: ("x", "leak")}), "c")):
            look = self.look(spec, sub=sub)
            ids = [r["id"] for r in look["sentences"]]
            self.assertEqual(ids[2], "c1.certified", ids)
            self.assertTrue(ids[1].startswith("c1."))

    def test_c3_sentences(self):
        pts = TP.override(TP.default_points, [("stall", TWIN3, SC, None, 95.0), ("no_stall", TWIN3, SC, None, 90.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c3_each"][TWIN3], "holds")
        self.assertEqual(look["claims"]["c3"]["status"], "holds")
        text = self.text_of(look, "c3.holds")
        self.assertIn(K.ARM_PLAIN[TWIN3], text)
        self.assertIn("at most 1 was allowed", text)
        look = self.look(sub="b")
        self.assertEqual(self.text_of(look, "c3.fails"), TP.T["c3.fails"])

    def test_refused_clause_names_the_seed(self):
        pts = TP.override(TP.default_points, [("stall", "obca_forced", SC, None, 70.0), ("no_stall", "obca_forced", SC,
                                                                                          None, 90.0)])
        look = self.look(dict(points=pts, refused={SEEDS[3]: ["stall"]}))
        text = self.text_of(look, "c1.fail")
        self.assertIn(f"on seed {SEEDS[3]} BCA could not set its height from 45 held-out full-throttle rows", text)
        self.assertIn(TP.p_percent(45) + "%", text)
        pts2 = TP.override(pts, [("stall", TWIN1, SC, None, 79.0)])
        look = self.look(dict(points=pts2, refused={SEEDS[3]: ["stall"]}), sub="b")
        self.assertIn(f"seed {SEEDS[3]} was left out of this reading", self.text_of(look, "c2.fails"))


# --------------------------------------------------------------------------------------------- the constants

class TestConstantsAgree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("near80_v9_approve_local", str(HERE.parent / "approve_local_near80.py"))
        cls.AL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.AL)

    def test_approve_local_agrees(self):
        AL = self.AL
        self.assertEqual(AL.V8_HISTORY_FILES, K.V8_HISTORY_FILES)
        self.assertEqual(AL.V8_HISTORY_DIRS, K.V8_HISTORY_DIRS)
        self.assertEqual(AL.V8_FROZEN, K.PRE_APPROVAL_DIR + "/" + K.V8_FROZEN_NAME)
        self.assertEqual(AL.V9_NOTES, K.V9_NOTES)
        self.assertEqual(AL.V9_DECISION_DIR, K.V9_DECISION_DIR)
        self.assertEqual(AL.FROZEN_NAME, K.PRE_APPROVAL_DIR + "/" + K.COUNTS_FROZEN_NAME)
        self.assertEqual(AL.FROZEN_NAME, "pre_approval/pooled_facts_seeds0_4_spent10_14.json")
        self.assertEqual(AL.CHECK_TESTS, K.TEST_FILES)
        self.assertEqual(AL.VERSION, K.VERSION)
        self.assertEqual(AL.BRANCH, K.BRANCH)

    def test_decisions_values(self):
        self.assertEqual(K.FRESH, tuple(range(40, 50)))
        self.assertEqual(K.RESERVED_SEEDS, tuple(range(50, 60)))
        self.assertEqual(K.FRESH_SEED_BLOCK, (40, 59))
        self.assertFalse(set(K.FRESH) & K.USED_SEEDS)
        self.assertEqual(K.SPENT_SEEDS, (10, 11, 12, 13, 14))
        self.assertTrue(set(K.SPENT_SEEDS) <= K.USED_SEEDS)
        self.assertEqual(K.TWIN_ARMS, ("meandiff_q90x1", "meandiff_q90x2", "meandiff_limit"))
        self.assertEqual(K.MARKS_C2, dict(net_margin=5.0, positive_seeds_min=8, positive_share=0.8,
                                          multiplicity="all_three"))
        self.assertEqual(K.MARKS_C3, dict(net_margin=-5.0, positive_seeds_max=1))
        self.assertEqual(K.MIN_CERTIFIED, 8)
        self.assertEqual(K.MARKS["noise_floor_max"], 5.0)
        self.assertEqual(K.LEAD_RIVALS, K.WORLD_BLIND + ("range_hinge",))
        self.assertEqual(K.LONE_CALLS, ("meandiff_k1", "meandiff_q90x1", "meandiff_q90x2", "obca_k1_alone"))
        self.assertEqual(K.NEW_ARMS, ("meandiff_k1", "meandiff_q90x1", "meandiff_q90x2", "meandiff_limit",
                                      "obca_k1_alone"))
        self.assertTrue(set(K.TWIN_ARMS) <= set(K.NEW_ARMS) <= set(K.ARMS["main"]))
        self.assertFalse(set(K.TWIN_ARMS) & set(K.BCA_DEPENDENT))
        self.assertEqual(set(K.ARM_PLAIN), set(K.ARMS["main"]))
        self.assertIn("conformal_k1", K.REPORTED_ONLY)
        self.assertIn("meandiff_k1", K.REPORTED_ONLY)
        self.assertEqual(K.placeholders_present(), [])
        self.assertEqual(K.OWNER_ANSWER["text"], "Accept the defaults and cluster")
        self.assertEqual(K.OWNER_ANSWER["utc"], "2026-10-20T18:29:00Z")
        self.assertEqual(K.OWNER_ANSWER["second"]["text"], "committed from desktop after copying over from cluster")
        self.assertEqual(K.OWNER_ANSWER["second"]["utc"], "2026-10-20T19:20:00Z")
        self.assertEqual(K.OWNER_ANSWER["second"]["choices"], ["9"])
        self.assertEqual(sorted(K.OWNER_ANSWER["taken"]), ["4"])         # choice 9 takes its default (the desktop)
        self.assertEqual(len(K.OWNER_ANSWER["choices"]), 11)
        self.assertIn("committed from the desktop", K.OWNER_ANSWER["choices"][8])
        self.assertEqual(K.PLACEHOLDERS, {})
        self.assertEqual(K.CAP_CPU_HOURS, 55.0)
        self.assertEqual(K.FORECAST_CPU_HOURS_PER_SEED, 2.2)
        self.assertEqual(K.FORECAST_CPU_HOURS, {"plumbing": 2.5, "fresh": 22.0})
        self.assertEqual(K.COUNTS_FROZEN_NAME, "pooled_facts_seeds0_4_spent10_14.json")

    def test_cluster_entry_is_in(self):
        """The owner's answer (20 October 2026): the cluster runs version 9; its two scripts are pinned top-level
        files, tested in stub mode on the cluster (V9_BUILD_NOTES.md); the desktop entry is the fallback."""
        for name in ("run_near80_cluster.sh", "near80_slurm_job.sh"):
            self.assertIn(name, K.TOP_FILES)
            self.assertTrue((HERE.parent / name).is_file(), name)
        self.assertEqual(K.TOP_FILES, ("PREREGISTRATION.md", "README.md", "approve_local_near80.py", "run_near80.sh",
                                       "run_near80_cluster.sh", "near80_slurm_job.sh", "push_runfiles_near80.sh"))
        self.assertIn("cluster", K.HOST_PROFILES)
        self.assertIn("run_near80_cluster.sh", K.HOST_PROFILES["cluster"]["entry"])
        self.assertNotIn("not part of version 9", K.HOST_PROFILES["cluster"]["entry"])
        self.assertIn("test_v9.py", K.CODE_FILES)
        self.assertIn("test_v9.py", K.TEST_FILES)
        draft = draft_payload()
        self.assertEqual(draft["files"]["entries"], ["run_near80_cluster.sh", "near80_slurm_job.sh", "run_near80.sh",
                                                     "push_runfiles_near80.sh"])
        # the entry's job sizes and Slurm settings are the constants' (the scripts cannot import them)
        entry = (HERE.parent / "run_near80_cluster.sh").read_text(encoding="utf8")
        job = (HERE.parent / "near80_slurm_job.sh").read_text(encoding="utf8")
        C = K.CLUSTER
        self.assertIn(f"ACCOUNT={C['account']}\n", entry)
        self.assertIn(f"PARTITION={C['partition']}\n", entry)
        self.assertIn(f"CONSTRAINT={C['constraint']}\n", entry)
        j = C["jobs"]
        self.assertIn(f"plumbing) D_LANES={j['plumbing']['lanes']}; D_MEM={j['plumbing']['mem_gb']}; "
                      f"D_TIME={j['plumbing']['time']};;", entry)
        self.assertIn(f"run)      D_LANES={j['seed']['lanes']}; D_MEM={j['seed']['mem_gb']}; D_TIME={j['seed']['time']};;",
                      entry)
        self.assertIn(f"G_LANES={j['group']['lanes']}; G_MEM={j['group']['mem_gb']}; G_TIME={j['group']['time']}\n", entry)
        self.assertIn(f"PINNED={C['pinned_deps']}\n", job)
        self.assertIn("unshare --map-root-user --mount", job)
        self.assertIn("--group-only --tag group", job)
        self.assertIn('--seeds "$SEED" --tag "s$SEED"', job)
        self.assertEqual(C["scratch"], K.CLUSTER_SCRATCH)

    def test_choices_taken_record_the_cluster(self):
        """near80_prereg.choices_taken on the real draft: the defaults, except choice 4 (the cluster); choice 9 (where
        results are committed from) takes its default, the desktop, by the owner's second answer, whose time it
        carries."""
        NPm = K.load_local("near80_prereg")
        draft = draft_payload()
        taken = NPm.choices_taken(draft["choices"])
        self.assertEqual(len(taken), 11)
        for c, d in zip(taken, draft["choices"]):
            if c["number"] == 4:
                self.assertEqual(c["accepted"], K.OWNER_ANSWER["utc"])
                self.assertIn("cluster", c["taken"])
                self.assertIn("fallback", c["taken"])
                self.assertNotEqual(c["taken"], d["default"])
            elif c["number"] == 9:
                self.assertEqual(c["accepted"], K.OWNER_ANSWER["second"]["utc"])
                self.assertEqual(c["taken"], d["default"])
                self.assertEqual(c["taken"], "the desktop")
                self.assertIn("copied", d["owner_took"])
            else:
                self.assertEqual(c["accepted"], K.OWNER_ANSWER["utc"])
                self.assertEqual(c["taken"], d["default"])
        self.assertIn("after copying over from cluster", draft["choices"][8]["owner_took"])
        with mock.patch.object(K, "OWNER_ANSWER", dict(K.OWNER_ANSWER, taken={"12": "x"})):
            with self.assertRaises(NPm.PreregProblem):
                NPm.choices_taken(draft["choices"])

    def test_seed_job_leaves_the_lock_dependent_phases_to_the_group_job(self):
        R = K.load_local("near80_runner")
        PH = K.load_local("phases")
        f0 = K.SMOKE_SEEDS[0]
        self.assertTrue(R.group_dependent(PH.parse_id("lock.x.x.fresh")))
        self.assertTrue(R.group_dependent(PH.parse_id("look.x.x.fresh")))
        self.assertTrue(R.group_dependent(PH.parse_id(f"evaluate.main.stall.s{f0}")))
        self.assertFalse(R.group_dependent(PH.parse_id(f"policy.main.no_stall.s{f0}")))
        self.assertFalse(R.group_dependent(PH.parse_id(f"data_main.main.stall.s{f0}")))

    def test_documents_name_the_open_items_and_nothing_else(self):
        """While the document is a draft its one open item (the facts the freeze fills) carries a TODO(FREEZE) mark
        and nothing else does; once it is finalized (the freeze's as-frozen copy, or the frozen folder) no mark is
        left anywhere."""
        doc = (HERE.parent / "PREREGISTRATION.md").read_text(encoding="utf8")
        readme = (HERE.parent / "README.md").read_text(encoding="utf8")
        self.assertNotIn("TODO(DECISION", doc)
        self.assertNotIn("TODO(DECISION", readme)
        self.assertNotIn("TODO(OWNER", doc)
        self.assertNotIn("TODO(OWNER", readme)
        self.assertNotIn("TODO(", readme)
        if document_is_a_draft():
            self.assertIn("TODO(FREEZE", doc)
        else:
            self.assertNotIn("TODO(", doc)
        self.assertNotIn("TODO(", draft_text())
        for name in ("run_near80_cluster.sh", "near80_slurm_job.sh"):
            self.assertIn(name, readme)
            self.assertIn(name, doc)
        self.assertNotIn("not part of version 9", doc)
        self.assertNotIn("not part of version 9", readme)
        self.assertIn(K.OWNER_ANSWER["text"], doc)
        self.assertIn(K.OWNER_ANSWER["text"], readme)
        self.assertIn(K.OWNER_ANSWER["second"]["text"], doc)
        self.assertIn(K.OWNER_ANSWER["second"]["text"], readme)
        self.assertNotIn("login node in your own SSH session", doc)      # the second answer: nothing is pushed there
        self.assertNotIn("login node in your own SSH session", readme)


# --------------------------------------------------------------------------------------------- edge cases of the look
# (the independent review of 20 October 2026 wrote the first versions of these; kept here so that the freeze runs them)

class TestEverySeedRefuses(Base):
    def test_all_seeds_refuse_in_the_stalling_world(self):
        """Every seed refused where full throttle stalls: C1 is read from the numbers (every BCA number there is the
        no-caution actor's, and BCA's own no-stall loss stands), C2 and C3 are not read, the noise floor cannot be
        read, ten refusal clauses and ten refusal sentences are written, the certified sentence reads not available."""
        root = TP.make_tree(self.tmp / "a", dict(refused={s: ["stall"] for s in SEEDS}))
        R = P.load_results(root, "fresh", "scratch")
        look = P.make_look(R, TP.BLOCK)
        self.assertEqual(R.certified, [])
        self.assertEqual(look["brackets"]["F"], len(SEEDS))
        self.assertEqual((look["brackets"]["K"], look["brackets"]["NC"]), (0, 0))
        self.assertEqual(look["outcomes"]["c2"], "not_read")
        self.assertEqual(look["outcomes"]["c3"], "not_read")
        self.assertTrue(look["validity"]["noise_floor"]["unmeasured"])
        self.assertIsNone(look["validity"]["noise_floor"]["passed"])
        nl = look["marks"]["c1"]["net_lead"]
        self.assertAlmostEqual(nl["net"], -1.0)                 # stalling gain 0 (no caution), no-stall loss 1 (BCA's)
        self.assertEqual(look["outcomes"]["c1"], "fail")
        text = self.text_of(look, "c1.fail")
        self.assertEqual(text.count("BCA could not set its height"), len(SEEDS))
        self.assertTrue(text.endswith(TP.T["c1.last.numbers"]))   # the lead missed on its numbers
        ids = [r["id"] for r in look["sentences"]]
        self.assertIn("c2.not_read", ids)
        self.assertEqual(ids.count("stop.refused"), len(SEEDS))
        self.assertIn("read not available", self.text_of(look, "c1.certified"))
        md = P.look_md(look, "0" * 64)
        self.assertIn("Certified seeds (C2, C3 and the noise floor): none", md)

    def test_all_seeds_refuse_in_both_worlds(self):
        root = TP.make_tree(self.tmp / "a", dict(refused={s: list(K.WORLDS) for s in SEEDS}))
        R = P.load_results(root, "fresh", "scratch")
        look = P.make_look(R, TP.BLOCK)
        self.assertEqual(look["outcomes"]["c2"], "not_read")
        self.assertEqual([r["id"] for r in look["sentences"]].count("stop.refused"), 2 * len(SEEDS))
        for s in SEEDS:
            for w in K.WORLDS:
                self.assertEqual(set(R.substituted[s][w]), set(K.BCA_DEPENDENT) | set(P.MATCHED_RIVALS))
        self.assertAlmostEqual(look["marks"]["c1"]["net_lead"]["net"], 0.0)


class TestZeroFullThrottleRows(Base):
    def test_look_with_an_empty_group(self):
        look = self.look(dict(counts=lambda s, w: 0 if (s, w) == (SEEDS[0], "stall") else 45))
        cs = look["counts"]
        self.assertEqual((cs["NMIN"], cs["NREACH"]), (0, 2 * len(SEEDS) - 1))
        sw = look["reported"]["switches"]["stall"][str(SEEDS[0])]
        self.assertEqual(sw["n_full_throttle"], 0)
        self.assertEqual(sw["certify_chance"], 0.0)
        self.assertEqual(sw["ranks"], {})                       # no sorted tail: no rank
        self.assertIn("count.reported", [r["id"] for r in look["sentences"]])

    def test_pooled_groups_with_no_full_throttle_row(self):
        import near80lib as N
        from experiments.shift import shift as SH
        T80 = load_test_module("test_near80")
        log, surv, per_split = T80.synthetic_log()
        gap = T80.synthetic_gap()
        rm = SH.make_removal(log, "gap", gap)
        g = N.pooled_groups(log, gap, rm, t_lo=1.5, step1=None)     # above every throttle index
        self.assertEqual((g["n_full_throttle"], g["n_survivors"], g["n_heldout_tail"]), (0, 0, 0))
        self.assertEqual(g["n_rest"], g["n_heldout"])
        self.assertTrue(g["leak_checks"]["passed"])
        c = N.certification_expectation(0)
        self.assertEqual((c["certify_chance"], c["reaches_min_rows"]), (0.0, False))
        self.assertEqual(K.certify_chance(0), 0.0)

    def test_an_undefined_gap_rule_is_check_4_and_a_stop_of_the_arm_table(self):
        """Section 8.1, check 4: an undefined gap rule (or a t_lo outside [0.78, 0.80]) is a plumbing stop of the
        seed-world in the data phase (near80lib.gap_rule_problem, raised by settingdata.data_main), so no switch or
        policy phase of that seed-world runs. Were an undefined entry to reach the arm table all the same, that
        world's row is a recorded stop (never a silent pass) and the other world's own row is unaffected."""
        import near80lib as N
        import learner_phases as LP
        import phases as PHreal
        self.assertIsNone(N.gap_rule_problem(dict(defined=True, t_lo=0.786)))
        self.assertIsNone(N.gap_rule_problem(dict(defined=True, t_lo=0.78)))
        self.assertIsNone(N.gap_rule_problem(dict(defined=True, t_lo=0.80)))
        self.assertIn("check 4", N.gap_rule_problem(dict(defined=False, t_lo=0.95, reason="fewer than 9 rows")))
        self.assertIn("outside", N.gap_rule_problem(dict(defined=True, t_lo=0.775)))
        self.assertIn("outside", N.gap_rule_problem(dict(defined=True, t_lo=0.8001)))
        self.assertIn("outside", N.gap_rule_problem(dict(defined=True, t_lo=None)))
        self.assertEqual(K.T_LO_RANGE, (0.78, 0.80))
        PP = K.load_local("policy_phase")
        S = P.step2()
        S1 = LP.step1lib()
        gr = dict(defined=False, t_lo=0.95, t_up=None, n_tail=0, gap=0.0, reason="fewer than 9 rows")
        und = S1.jsonable(LP.undefined_entry(gr))
        ok = dict(t_lo=0.79, t_up=0.98, certificate=dict(on=True, height=2.5, q05=0.4, R_tail=2.46, R_rest=-0.04),
                  groups=dict(n_full_throttle=45, n_rest=4000, mean_full_throttle=0.5, mean_rest=0.1,
                              q90_full_throttle=1.5, q90_rest=0.3), rank_test=dict(U=1.0, p=0.001))
        s0 = SEEDS[0]
        with self.assertRaises(PHreal.PhaseStop) as cm:
            PP.arm_row_from("main", s0, "stall", {f"stall/seed{s0}": und, f"no_stall/seed{s0}": ok}, S)
        self.assertEqual(cm.exception.reason, PHreal.STOP_REASONS["check"])
        with self.assertRaises(PHreal.PhaseStop):
            PP.arm_row_from("main", s0, "no_stall", {f"stall/seed{s0}": ok, f"no_stall/seed{s0}": und}, S)
        row = PP.arm_row_from("main", s0, "stall", {f"stall/seed{s0}": ok, f"no_stall/seed{s0}": und}, S)
        self.assertEqual(row["refused"], [])                    # the stalling world's row reads its own entry
        row = PP.arm_row_from("main", s0, "stall", {f"stall/seed{s0}": ok, f"no_stall/seed{s0}": ok}, S)
        self.assertEqual(row["refused"], [])


class TestDuplicateRowsAcrossSplits(unittest.TestCase):
    """Identical commands in two held-out splits are two rows (each log row once); an empty split is fine."""

    @classmethod
    def setUpClass(cls):
        T80 = load_test_module("test_near80")
        cls.log, cls.surv, cls.per_split = T80.synthetic_log()
        cls.gap = T80.synthetic_gap()

    def test_identical_commands_in_two_splits_are_two_rows(self):
        import numpy as np
        import near80lib as N
        from experiments.shift import dataset as DS
        from experiments.shift import shift as SH
        log, gap = self.log, self.gap
        cal, ev = log.split_rows("calibration"), log.split_rows("evaluation")
        self.assertEqual(len(cal), len(ev))
        arrays = {k: np.array(v, copy=True) for k, v in log._arrays.items()}
        arrays["actions"][ev] = arrays["actions"][cal]            # the evaluation split repeats the calibration commands
        dup = DS.Log(arrays, dict(log.meta))
        rm = SH.make_removal(dup, "gap", gap)
        kept = rm.kept_train_rows(dup)
        step1 = N.load_step1lib()
        gr = step1.gap_rule(step1.throttle(dup["actions"][kept]), **K.GAP_RULE)
        g = N.pooled_groups(dup, gap, rm, gr["t_lo"], step1)
        held = np.concatenate([g["heldout_tail"], g["heldout_rest"]])
        self.assertEqual(len(held), len(set(held.tolist())))      # each log row once
        outside_cal = int((~gap.contains(dup["actions"][cal])).sum())
        self.assertEqual(g["n_heldout"], 2 * outside_cal)          # both copies counted: distinct rows
        self.assertEqual(int(np.isin(held, cal).sum()), outside_cal)
        self.assertEqual(int(np.isin(held, ev).sum()), outside_cal)
        self.assertFalse(np.isin(held, dup.split_rows("train")).any())
        self.assertTrue(set(g["survivors"].tolist()) <= set(kept.tolist()))
        self.assertTrue(g["leak_checks"]["passed"])

    def test_empty_split_is_fine(self):
        import numpy as np
        import near80lib as N
        from experiments.shift import dataset as DS
        from experiments.shift import shift as SH
        log, gap = self.log, self.gap
        arrays = {k: np.array(v, copy=True) for k, v in log._arrays.items()}
        ev = log.split_rows("evaluation")
        arrays["infos/split"][ev] = 1                                 # every evaluation row relabelled calibration
        one = DS.Log(arrays, dict(log.meta))
        self.assertEqual(len(one.split_rows("evaluation")), 0)
        rm = SH.make_removal(one, "gap", gap)
        g = N.pooled_groups(one, gap, rm, 0.9, None)
        self.assertEqual(g["n_heldout"], int((~gap.contains(one["actions"][one.split_rows("calibration")])).sum()))


class TestDataPhaseStop(Base):
    def test_a_stopped_data_phase_fails_the_seed_with_its_clause(self):
        """A plumbing stop of the data phase (section 8.1, checks 3 and 4: a leak check, an undefined gap rule or a
        t_lo outside its range) leaves no record of that seed-world; the look fails the seed by the stop, says so in
        C1's clause and the stop sentence, and C2 and C3 read the failed-seed sentence."""
        PHm = K.load_local("phases")
        reason = PHm.STOP_REASONS["check"]
        look = self.look(dict(stops=[dict(phase_kind="data_main", setting="main", world="stall", seed=SEEDS[1],
                                          reason=reason)],
                              missing_records=[("stall", SEEDS[1])]))
        self.assertEqual(look["outcomes"]["c1"], "fail")
        self.assertTrue(look["outcomes"]["c1_by_stop"])
        self.assertIn(str(SEEDS[1]), look["failed_checks"])
        self.assertTrue(look["failed_checks"][str(SEEDS[1])].startswith("its main setting's data"))
        self.assertIn(f"seed {SEEDS[1]}'s the main setting's data", self.text_of(look, "c1.fail"))
        self.assertTrue(self.text_of(look, "c1.fail").endswith(TP.T["c1.last.stop"]))
        self.assertEqual(look["outcomes"]["c2"], "fail_failed_seed")
        ids = [r["id"] for r in look["sentences"]]
        self.assertEqual(ids[-1], "stop.phase")
        self.assertIn(f"Seed {SEEDS[1]}, stalling world:", self.text_of(look, "stop.phase"))
        self.assertIn(reason, self.text_of(look, "stop.phase"))
        self.assertEqual(look["marks"]["c2"]["certified"], [s for s in SEEDS if s != SEEDS[1]])


@unittest.skipUnless(POSIX and (K.CODE / "approval_check.py").is_file(), "POSIX only (taskset, fcntl)")
class TestGroupJobBeforeSeedJob(unittest.TestCase):
    """The cluster's group job (--group-only) started BEFORE the seed job: it must wait (start nothing, skip nothing,
    halt nothing) until the seed job's phases are final, then run the lock, the evaluations and the look. Stub mode on
    the plumbing's seed 0, nothing trained; a few minutes."""

    @classmethod
    def setUpClass(cls):
        import threading
        import time
        R = K.load_local("near80_runner")
        import approval_check as AC
        PHm = K.load_local("phases")
        cls.R, cls.PH = R, PHm
        cls.d = Path(tempfile.mkdtemp(prefix="near80_test_v9_group_"))
        fx = cls.d / "fx"
        AC.write_test_fixture(fx)
        R.stub_fixture_memory(fx)
        cls.fx = fx
        cfg = R.stub_cfg(fx, 2, [K.SMOKE_SEEDS[0]])
        env = cfg.phase_env()
        cpus, why = R.lane_cpus(1, cfg.sysfs, given=cfg.cpus)
        assert why is None, why
        cmd = ["env", "-i"] + [f"{k}={v}" for k, v in env.items()] + ["taskset", "-c", str(cpus[0]), cfg.python,
                                                                       str(K.CODE / "stubs.py"), "step2-fixture", str(fx)]
        subprocess.run(cmd, cwd=WORK, check=True, timeout=1800, capture_output=True)
        cls.approval = cfg.approval()
        cls.keys = PHm.plan(spec=cfg.plan)
        cls.lines_g, cls.lines_s = [], []
        cls.codes = {}
        cfg_g = R.stub_cfg(fx, 2, [K.SMOKE_SEEDS[0]])
        cfg_g.group_only, cfg_g.tag = True, "group"

        def group():
            cls.codes["group"] = R.run_locked(cfg_g, cls.keys, cls.approval, out=cls.lines_g.append, signals=False)

        cls.t = threading.Thread(target=group, daemon=True)
        cls.t.start()
        time.sleep(8.0)
        cls.status_early = json.loads((fx / "out" / "status_group.json").read_text(encoding="utf8"))
        cls.lines_early = list(cls.lines_g)
        cfg_s = R.stub_cfg(fx, 2, [K.SMOKE_SEEDS[0]])
        cfg_s.seeds, cfg_s.tag = [K.SMOKE_SEEDS[0]], "s0"
        cls.codes["seed"] = R.run_locked(cfg_s, cls.keys, cls.approval, out=cls.lines_s.append, signals=False)
        cls.t.join(timeout=1200)
        cls.out = fx / "out"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_group_runner_waited_then_finished(self):
        PHm = self.PH
        self.assertFalse(self.t.is_alive(), "the group runner did not finish after the seed job")
        self.assertEqual(self.codes["seed"], 0, "\n".join(self.lines_s[-20:]))
        self.assertEqual(self.codes["group"], 0, "\n".join(self.lines_g[-20:]))
        st = self.status_early
        self.assertTrue(st["group_only"])
        self.assertEqual(st["counts"].get("running", 0), 0)
        self.assertEqual(st["counts"].get("skipped", 0), 0)
        self.assertEqual(st["counts"].get("halted", 0), 0)
        self.assertIsNone(st["halted"])
        self.assertTrue(any("phases of other runners are not finished" in line for line in self.lines_early),
                        self.lines_early[-5:])
        self.assertFalse(any(" start  " in line for line in self.lines_early), self.lines_early)
        started_g = [line for line in self.lines_g if " start  " in line]
        kinds_g = sorted({line.split(" start  ")[1].split(".")[0] for line in started_g})
        self.assertEqual(kinds_g, ["evaluate", "lock", "look"])
        started_s = [line for line in self.lines_s if " start  " in line]
        kinds_s = sorted({line.split(" start  ")[1].split(".")[0] for line in started_s})
        for kind in ("lock", "evaluate", "look"):
            self.assertNotIn(kind, kinds_s)
        self.assertEqual({PHm.phase_id(k): PHm.done(k, self.out) for k in self.keys
                          if PHm.done(k, self.out) != "done"}, {})
        look = json.loads((self.out / K.LAYOUT["look"].format(group="smoke")).read_text(encoding="utf8"))
        self.assertEqual(look["seeds"], [K.SMOKE_SEEDS[0]])


# --------------------------------------------------------------------------------------------- the certification chance

class TestCertificationChance(unittest.TestCase):
    def test_table(self):
        for n, want in ((24, 0.000), (28, 0.405), (29, 0.700), (30, 0.896), (31, 0.976), (33, 0.9995)):
            self.assertAlmostEqual(K.certify_chance(n), want, places=3, msg=n)
        self.assertEqual(K.certify_chance(0), 0.0)
        self.assertLess(K.certify_chance(18), 1e-6)
        self.assertGreater(K.certify_chance(35), 0.99999)
        self.assertAlmostEqual(P.certify_percent(29), 69.99, places=1)
        self.assertIsNone(P.certify_percent(None))


# --------------------------------------------------------------------------------------------- the stub run

@unittest.skipUnless(POSIX and (K.CODE / "approval_check.py").is_file(), "POSIX only (taskset, fcntl)")
class TestStubRefuse(unittest.TestCase):
    """near80_runner.py stub --fixture DIR --seeds 0 --refuse 0: every phase of the plumbing's seed 0 in stub mode
    (nothing trained; the owner's rule: a test runs on seed 0 only), its synthetic log keeping too few full-throttle
    rows for BCA's certificate in both worlds. The stand-ins take no gradient step, so every evaluated point is 0
    and the look's validity check fails by design: C1, C2 and C3 read 'invalid'; the refusal is recorded per arm,
    the substitution is reported, and the refusal sentences are written. The arithmetic of C2 and C3 on the
    certified seeds is tested on synthetic records (TestRefusalReading, TestTwinRules)."""

    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp(prefix="near80_test_v9_stub_"))
        cls.fx = cls.d / "fx"
        env = {k: v for k, v in K.PHASE_ENV.items() if k != "PYTHONPYCACHEPREFIX"}
        for name in ("NEAR80_HOST", "NEAR80_ROOT", "NEAR80_PYTHON", "BCA_SHIFT_DEPS", "NEAR80_STUB_PYTHONPATH"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        lanes = str(min(4, len(os.sched_getaffinity(0))))
        cls.seed = K.SMOKE_SEEDS[0]
        cls.group = "smoke"
        cls.r = subprocess.run([sys.executable, str(HERE / "near80_runner.py"), "stub", "--fixture", str(cls.fx),
                                "--lanes", lanes, "--seeds", str(cls.seed), "--refuse", str(cls.seed)],
                               cwd=WORK, env=env, capture_output=True, text=True, timeout=3600)
        cls.out = cls.fx / "out"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, True)

    def test_every_phase_ran_and_nothing_trained(self):
        import ledger as LG
        import phases as PH
        r = self.r
        self.assertEqual(r.returncode, 0, r.stdout[-4000:] + r.stderr[-4000:])
        keys = PH.plan(spec=f"seeds:{self.seed}")
        not_done = {PH.phase_id(k): PH.done(k, self.out) for k in keys if PH.done(k, self.out) != "done"}
        self.assertEqual(not_done, {})                                      # no stop: the refusal is not a stop
        hits = {}
        for p in (self.out / "ledger").rglob("*.usage.json"):
            for k, v in (LG.read_json_verified(p).get("stub_guard_hits") or {}).items():
                hits[k] = hits.get(k, 0) + v
        self.assertEqual(hits, {})
        self.assertEqual(json.loads((self.fx / "stub.json").read_text())["refuse_seeds"], [self.seed])

    def test_the_refusal_is_recorded_per_arm(self):
        import ledger as LG
        import phases as PH
        bad = self.seed
        sw = LG.read_json_verified(self.out / PH.record_rel(PH.parse_id(f"switch.main.stall.s{bad}")))
        self.assertTrue(sw["defined"])
        self.assertTrue(sw["certificate"]["on"])
        self.assertTrue(sw["refused"], sw["certificate"])
        self.assertLess(sw["groups"]["n_full_throttle"], K.CERT_MIN_ROWS)
        self.assertLess(K.certify_chance(sw["groups"]["n_full_throttle"]), 0.01)
        lock = LG.read_json_verified(self.out / PH.record_rel(PH.parse_id(f"policy.main.stall.s{bad}")))
        self.assertEqual(lock["refused_arms"], sorted(K.BCA_DEPENDENT))
        self.assertTrue(lock["bca_refused"])
        for a in K.BCA_DEPENDENT:
            self.assertIsNone(lock["arm_ids"][a], a)
        for a in K.TWIN_ARMS + ("meandiff_k1", "range_hinge", "fixed_gated"):
            self.assertIsNotNone(lock["arm_ids"][a], a)                      # every other arm trained
        self.assertFalse(lock["matches"]["even_pull"]["matched"])
        self.assertIn("refused", lock["matches"]["even_pull"]["reason"])
        self.assertEqual(lock["ramp_padding"]["arms"], ["pad_refused_" + a for a in ("obca_k1", "obca_k2", "obca_forced")])
        self.assertEqual(lock["ramp_programs"][1], list(K.STEP2_RAMP_ARMS))
        v = LG.read_json_verified(self.out / PH.record_rel(PH.parse_id(f"evaluate.main.stall.s{bad}")))
        self.assertEqual(v["refused_arms"], sorted(K.BCA_DEPENDENT))
        self.assertIsNone(v["arm_ids"][K.PRIMARY])
        self.assertIsNotNone(v["arm_ids"][TWIN1])
        # the no-stall world of the refused seed: obca_forced (the stalling world's height) refused there too
        lock_n = LG.read_json_verified(self.out / PH.record_rel(PH.parse_id(f"policy.main.no_stall.s{bad}")))
        self.assertIn("obca_forced", lock_n["refused_arms"])

    def test_the_look_reads_the_refusal(self):
        import ledger as LG
        bad = self.seed
        look = LG.read_json_verified(self.out / K.LAYOUT["look"].format(group=self.group))
        self.assertEqual(look["kind"], f"near80-v9-look-{self.group}")
        self.assertEqual(look["seeds"], [bad])
        self.assertIn(str(bad), look["refused"])
        self.assertEqual(set(look["refused"][str(bad)]), set(K.WORLDS))
        self.assertEqual(look["failed_checks"], {})
        self.assertEqual(look["brackets"]["F"], 1)
        self.assertEqual(look["brackets"]["K"], 0)
        self.assertEqual(look["brackets"]["NC"], 0)
        self.assertEqual(look["marks"]["c2"]["certified"], [])
        sub = look["reported"]["substituted"][str(bad)]
        self.assertEqual(set(sub["stall"]), set(K.BCA_DEPENDENT) | set(P.MATCHED_RIVALS))
        # C1 read with the substitution: BCA's numbers on the refused seed are the no-caution actor's
        ps = look["per_seed"]
        self.assertAlmostEqual(ps["X"][0], 0.0)
        self.assertAlmostEqual(ps["NET"][0], 0.0)
        # the stand-ins' points are all 0, so the validity check fails and every claim reads 'invalid' (by design
        # of the stub: nothing is trained); the refusal sentences are written all the same
        self.assertEqual(look["outcomes"]["c1"], "invalid")
        self.assertEqual(look["outcomes"]["c2"], "invalid")
        self.assertEqual(look["outcomes"]["c3"], "invalid")
        ids = [r["id"] for r in look["sentences"]]
        self.assertIn("c1.invalid", ids)
        self.assertEqual(ids.count("stop.refused"), 2)
        texts = [r["text"] for r in look["sentences"] if r["id"] == "stop.refused"]
        self.assertTrue(texts[0].startswith(f"Seed {bad}, stalling world: BCA could not set its height"))
        self.assertTrue(texts[1].startswith(f"Seed {bad}, no-stall world: BCA could not set its height"))
        self.assertTrue(all(r["text"].startswith(TP.PREFIX) for r in look["sentences"] if r["id"].startswith("c1.")))
        md = (self.out / K.LAYOUT["look_md"].format(group=self.group)).read_text(encoding="utf8")
        self.assertIn("BCA refused, seed", md)
        self.assertIn("## The switch calls", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
