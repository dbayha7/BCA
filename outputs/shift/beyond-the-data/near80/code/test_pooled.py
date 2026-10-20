"""Near-80 (version 9, main setting): tests of the look (pooled.py) and of section 14's extraction and filling
(claims_text.py).

Every test builds a synthetic result tree in the runner's record schemas with make_tree, in a temporary folder
outside the repository, then checks the exact sentences: each expected sentence is built here from the template text
of the real PREREGISTRATION.md by plain string replacement of the expected bracket strings. Nothing is trained, no
seed's data is generated or read, nothing is written into the repository.

Run: python code/test_pooled.py   (from the work folder, PYTHONPATH=<repo>, NEAR80_HOST and NEAR80_ROOT set)
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402
import claims_text as CT  # noqa: E402

P = K.load_local("pooled")
DOC = (HERE.parent / "PREREGISTRATION.md").read_text(encoding="utf8")
BLOCK = CT.extract_section14(DOC)
T = {k: v["text"] for k, v in BLOCK["sentences"].items()}
PREFIX = BLOCK["smoke_prefix"]
SC = "strong_current"
OTHER_SITS = ("recorded", "big_waves", "big_kicks")
SEEDS = list(K.TEST_LABELS)      # the synthetic fresh group's labels (FRESH is patched to them in Base); never 40-59
NS = str(len(SEEDS))
RANGE = f"{SEEDS[0]}-{SEEDS[-1]}"
TWIN1, TWIN2, TWIN3 = K.TWIN_ARMS
PLAIN = {t: K.ARM_PLAIN[t] for t in K.TWIN_ARMS}


# --------------------------------------------------------------------------------------------- the synthetic tree

def _sha(b):
    return hashlib.sha256(b).hexdigest()


def put(path, body):
    """A record in the runner's format: canonical JSON and its '<name>.sha256' sidecar."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = P.canonical(body).encode("utf8")
    p.write_bytes(data)
    p.with_name(p.name + ".sha256").write_text(f"{_sha(data)}  {p.name}\n", encoding="utf8")
    return _sha(data)


# Stalling world, strong current: BCA's switched ramp 80 (trained alone 79), the twins below it; no-stall world:
# BCA 89, the twins a little lower. Nets across worlds: obca_k1 29, obca_k1_alone 28; the plain 90% gap 16 (d 12),
# twice it 20 (d 8), the gated range limit 0 (d 28): BCA ahead of all three. The best world-blind rival nets 0 (the
# matched even pull, first in order among the ties), so the lead is 29; the hand-set range limit nets -20.
STALL = dict(none=50, full_none=100, clone=40, w0_check=50, obca_k1=80, obca_k2=78, obca_forced=80, fixed_gated=75,
             conformal_k1=70, meangap_k1=65, meandiff_k1=66, meandiff_q90x1=70, meandiff_q90x2=74, meandiff_limit=60,
             obca_k1_alone=79, perfect_band=90, range_hinge=50, even_pull=55, push_less=54, random_1=52, random_2=51,
             standin=60)
NO_STALL = dict(none=90, obca_k1=89, obca_k2=88, obca_forced=60, fixed_gated=85, conformal_k1=86, meangap_k1=86,
                meandiff_k1=86, meandiff_q90x1=86, meandiff_q90x2=86, meandiff_limit=80, obca_k1_alone=89,
                perfect_band=88, range_hinge=70, standin=70, even_pull=85, push_less=84, random_1=88, random_2=89,
                clone=80, w0_check=90)
HEIGHTS = dict(obca_k1=2.5, obca_k2=2.5, obca_forced=2.5, fixed_gated=1.0, conformal_k1=1.2, meangap_k1=0.5,
               meandiff_k1=0.5, meandiff_q90x1=1.2, meandiff_q90x2=1.2, meandiff_limit=1.0, obca_k1_alone=2.5)
KS = dict(obca_k2=2.0, meandiff_q90x2=2.0)


def default_points(setting, world, seed, arm, sit):
    if sit != SC:
        return 71.0 if arm == "obca_k1" else 70.0
    table = NO_STALL if world == "no_stall" else STALL
    return float(table.get(arm, 85 if world == "no_stall" else 50))


def default_past(setting, world, seed, arm, sit):
    return 0.4 if arm == "none" else 0.1


def make_tree(tmp, spec=None):
    """Writes a synthetic result tree under tmp/out and returns its root. spec (all optional):
      group               'fresh' (default) or 'smoke'
      seeds               the seeds written (default: the group's)
      mode, prereg_sha256 the records' mode ('scratch') and fingerprint (None)
      points, past        f(setting, world, seed, arm, situation) -> number
      missing_arms        {(world, seed) | world: [logical ids with no actor]}
      missing_values      [(world, seed)]: no values.json (a stopped or skipped phase)
      missing_records     [(world, seed)]: no record at all of that seed-world (its data phase stopped: write a stop)
      data_main_fail      {seed: (reason, 'leak' | 'marks')}
      refused             {seed: [worlds]}: BCA's certificate on with an infinite height there. As the policy phase
                          does: in a refused world BCA's switched ramp, the ramp at twice its height and the lone
                          copy have no actor; when the STALLING world refused, the always-on ramp (it takes the
                          stalling world's height) has none in either world, and the matched rivals and the
                          stand-in (matched in the stalling world, locked for the other) have none in either world
      switch_on           f(world, seed) -> bool (default: on in the stalling world, off in the other)
      counts              f(seed, world) -> n_full_throttle (default 45)
      stops               [{phase_kind, setting, world, seed, reason}]
      plumbing            the plumbing record's 'reproduction' block (written when given)"""
    spec = dict(spec or {})
    root = Path(tmp) / "out"
    group = spec.get("group", "fresh")
    seeds = spec.get("seeds", list(K.SMOKE_SEEDS if group == "smoke" else K.FRESH))
    mode, psha = spec.get("mode", "scratch"), spec.get("prereg_sha256")
    env = dict(mode=mode, prereg_sha256=psha, written="2026-10-20T00:00:00Z")
    pts_f = spec.get("points", default_points)
    past_f = spec.get("past", default_past)
    missing_arms = spec.get("missing_arms", {})
    missing_values = set(spec.get("missing_values", []))
    missing_records = set(spec.get("missing_records", []))
    refused = spec.get("refused", {})
    switch_on = spec.get("switch_on", lambda w, s: w == "stall")
    counts = spec.get("counts", lambda s, w: 45)
    locks = {}
    for s in seeds:
        for w in K.WORLDS:
            if (w, s) in missing_records:
                continue
            fail = spec.get("data_main_fail", {}).get(s)
            n = counts(s, w)
            put(root / K.LAYOUT["data_main"].format(world=w, seed=s), dict(
                env, kind="near80-data-main", phase_id=K.phase_id("data_main", "main", w, s), seed=s, world=w,
                group=group, source="gate" if s == 0 else "fresh",
                ball_leak_checks_passed=not (fail and fail[1] == "leak"),
                gate_world_marks=None if w != "stall" else {f"T{i}": not (fail and fail[1] == "marks" and i == 2)
                                                             for i in range(1, 6)},
                passed_gate_checks=not fail, seed_failed_reason=fail[0] if fail else None,
                pooled_groups=None if fail else dict(
                    defined=True, groups=dict(n_full_throttle=n, n_rest=4000, n_survivors=30),
                    certification=dict(reaches_min_rows=bool(n >= 29), certify_chance=K.certify_chance(n)),
                    version8_groups=dict(n_full_throttle=max(n - 20, 5), n_rest=700))))
        for w in K.WORLDS:
            if (w, s) in missing_records:
                continue
            ref = s in refused and w in refused[s]
            stall_ref = s in refused and "stall" in refused[s]
            on = bool(switch_on(w, s)) or ref
            n = counts(s, w)
            tail = sorted(0.02 * i for i in range(n))                      # the sorted full-throttle scores
            put(root / K.LAYOUT["switch"].format(setting="main", world=w, seed=s), dict(
                env, kind="near80-switch", phase_id=K.phase_id("switch", "main", w, s), setting="main", world=w,
                seed=s, t_lo=0.79, t_up=0.98, defined=True, n_survivors=30, groups_rule="pooled",
                certificate=dict(on=on, height="inf" if ref else 2.5, q05=0.4 if on else 0.01,
                                 R_tail="inf" if ref else 2.46, R_rest=-0.04,
                                 finite_share_tail=0.85 if ref else 0.99),
                groups=dict(n_full_throttle=n, n_rest=4000, mean_full_throttle=0.5 if on else 0.02, mean_rest=0.0,
                            q90_full_throttle=1.16 if on else 0.05, q90_rest=-0.04),
                rank_test=dict(p=0.01 if on else 0.3), pooled=dict(tail_scores_sorted=tail),
                refused=bool(ref), refusal="BCA's switch is on but its height is not finite" if ref else None))
            if (w, s) in missing_values:
                continue
            gone = set(missing_arms.get((w, s), [])) | set(missing_arms.get(w, []))
            if ref:                                   # this world's height is infinite: BCA's own arms
                gone |= {"obca_k1", "obca_k2", K.NOISE_ARM}
            if stall_ref:                             # the stalling world's height is infinite: the always-on ramp in
                gone |= {"obca_forced"} | set(K.MATCHED) | {"standin"}   # both worlds; nothing to match to
            refused_arms = sorted(a for a in K.BCA_DEPENDENT if a in gone)
            ids = {}
            for a in K.ARMS["main"]:
                ids[a] = None if (a in gone or (w != "stall" and a in K.STALL_ONLY_ARMS)) else a
            table = {a: dict(switch=bool(on), h=(HEIGHTS[a] if on else 0.0), k=KS.get(a, 1.0),
                             refused=bool(a in refused_arms)) for a in HEIGHTS}
            lock_sha = put(root / K.LAYOUT["policy_lock"].format(setting="main", world=w, seed=s), dict(
                env, kind="near80-policy-lock", phase_id=K.phase_id("policy", "main", w, s),
                setting="main", world=w, seed=s, group=group, arm_ids=ids,
                refused_arms=refused_arms, bca_refused=bool(ref),
                strengths={n_: 1.0 for n_ in ids.values() if n_}, matched_names={}, standin=dict(arm="standin"),
                arm_table=dict(t_lo=0.79, t_up=0.98, arms=table)))
            locks[K.phase_id("policy", "main", w, s)] = dict(lock_sha256=lock_sha)
            sits = {}
            for sit in K.SITUATIONS:
                t = {m: {} for m in P.MEASURES}
                for a, name in ids.items():
                    if name is None:
                        continue
                    p = pts_f("main", w, s, a, sit)
                    t["points"][name] = p
                    t["held70_points"][name] = None if p is None else p - 1.0
                    t["capped80_points"][name] = None if p is None else p + 0.5
                    t["unseen_gain"][name] = None if p is None else 1.0
                    t["overshoot_loss"][name] = None if p is None else 0.5
                    t["past80_share"][name] = past_f("main", w, s, a, sit)
                    t["zone_shares"][name] = [0.5, 0.3, 0.2]
                    t["lower_tail_points"][name] = None if p is None else p - 20
                sits[sit] = t
            put(root / K.LAYOUT["values"].format(setting="main", world=w, seed=s), dict(
                env, kind="near80-values", phase_id=K.phase_id("evaluate", "main", w, s), setting="main",
                world=w, seed=s, group=group, source="trained", lock_sha256=lock_sha, arm_ids=ids,
                refused_arms=refused_arms, situations=sits, evaluations={}))
    for st in spec.get("stops", []):
        pid = K.phase_id(st["phase_kind"], st.get("setting", "x"), st.get("world", "x"), st.get("seed"),
                         st.get("group"))
        put(root / K.LAYOUT["phase_stop"].format(phase_id=pid), dict(
            env, kind="near80-phase-stop", phase_id=pid, phase_kind=st["phase_kind"], setting=st.get("setting", "x"),
            world=st.get("world", "x"), seed=st.get("seed"), group=st.get("group") or group, reason=st["reason"]))
    put(root / K.LAYOUT["stage_lock"].format(group=group), dict(
        env, kind="near80-stage-lock", phase_id=K.phase_id("lock", group=group), group=group, locks=locks,
        plan=spec.get("plan", "full")))
    if spec.get("plumbing") is not None:
        put(root / K.LAYOUT["plumbing"], dict(kind="near80-plumbing", passed=False, prereg_sha256=psha,
                                              reproduction=spec["plumbing"], written=env["written"]))
    return root


def override(base, rules):
    """A points function equal to `base` except where a rule (world, arm, situation, seed or None, value) applies."""
    def f(setting, world, seed, arm, sit):
        for w, a, si, sd, v in rules:
            if w == world and a == arm and si == sit and (sd is None or sd == seed):
                return v[seed] if isinstance(v, dict) else v
        return base(setting, world, seed, arm, sit)
    return f


LATER = ("[one clause per missed mark]", "[one clause per twin]", "[last sentence]", "[seeds]")   # filled by the tests


def expect(sid, **br):
    s = T[sid]
    for k, v in br.items():
        s = s.replace(f"[{k}]", v)
    left = s
    for m in LATER:
        left = left.replace(m, "")
    assert "[" not in left, s
    return s


COMMON = dict(SEEDS=RANGE, NSEEDS=NS)
PASS1 = dict(COMMON, S=NS, N=NS, X="30.0", H="50.0", W="1.0", NET="29.0", D="29.0", R="the matched even pull",
             E="25.0", K=NS, F="0")
CERT1 = dict(NC=NS, reading="pass", DC="29.0", RC="the matched even pull", LC="-1.0", EC="25.0")
HOLDS2 = dict(COMMON, NC=NS, KMIN="8", A="20", D1="12.0", K1=NS, D2="8.0", K2=NS, D3="28.0", K3=NS)
COUNTS = dict(COMMON, NMIN="45", NMAX="45", NREACH=str(2 * len(SEEDS)), NSW=str(2 * len(SEEDS)), N8MIN="25",
              N8MAX="25")


def p_percent(n):
    return CT.fmt(100.0 * K.certify_chance(n), "points")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="near80_test_pooled_"))
        self.assertFalse(P.inside(self.tmp, K.ROOT))
        self._fresh = mock.patch.object(K, "FRESH", K.TEST_LABELS)      # the synthetic fresh group (no reserved seed)
        self._fresh.start()

    def tearDown(self):
        self._fresh.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tree(self, spec=None, sub="a"):
        return make_tree(self.tmp / sub, spec)

    def look(self, spec=None, sub="a"):
        spec = dict(spec or {})
        root = self.tree(spec, sub)
        group = spec.get("group", "fresh")
        R = P.load_results(root, group, "scratch")
        return P.make_look(R, BLOCK)

    def texts(self, look):
        return [r["text"] for r in look["sentences"]]

    def ids(self, look):
        return [r["id"] for r in look["sentences"]]

    def text_of(self, look, sid):
        rows = [r["text"] for r in look["sentences"] if r["id"] == sid]
        self.assertEqual(len(rows), 1, (sid, self.ids(look)))
        return rows[0]


# --------------------------------------------------------------------------------------------- extraction

class TestExtraction(Base):
    def test_every_id_once_and_in_order(self):
        self.assertEqual(list(BLOCK["sentences"]), list(CT.SENTENCE_IDS))
        self.assertEqual(len(set(CT.SENTENCE_IDS)), len(CT.SENTENCE_IDS))
        for sid in ("c1.clause.stopped_seed", "c1.last.numbers", "c1.last.stop", "c1.certified", "c2.clause.ahead",
                    "c2.clause.tie", "c2.clause.behind", "c2.last.dose", "c2.last.none", "c2.last.twin",
                    "c2.last.other", "c3.holds", "c3.fails", "c2.not_read"):
            self.assertIn(sid, BLOCK["sentences"])

    def test_templates_have_no_quote_marks_or_blockquote_markers(self):
        for sid, s in BLOCK["sentences"].items():
            self.assertFalse(s["text"].startswith('"'), sid)
            self.assertNotIn("\n", s["text"], sid)
            self.assertNotIn("> ", s["text"], sid)

    def test_brackets_all_defined(self):
        defined = set(CT.table_bracket_names(BLOCK))
        for name in ("S", "N", "X", "H", "W", "NET", "D", "R", "E", "L", "K", "F", "NC", "DC", "RC", "LC", "EC", "D1",
                     "K1", "LB1", "D2", "K2", "LB2", "D3", "K3", "LB3", "DT", "KT", "LBT", "KMIN", "A", "P", "V",
                     "NSEEDS", "SEEDS", "NMIN", "NMAX", "NREACH", "NSW", "N8MIN", "N8MAX"):
            self.assertIn(name, defined)

    def test_section_14_carries_no_todo_mark(self):
        """The margins are fixed; the only marks a draft carries (the freeze's facts) lie outside section 14, and a
        finalized document (the freeze's as-frozen copy, the frozen folder) carries none."""
        self.assertEqual(BLOCK["todo_marks"], [])
        self.assertEqual(CT.todo_marks("x TODO(OWNER: a) y TODO(FREEZE: b)"), ["TODO(OWNER: a)", "TODO(FREEZE: b)"])
        draft = DOC.split("\n")[2] == f"**{K.STATUS_LINE_DRAFT}**"
        if draft:
            self.assertTrue(CT.TODO_MARK.search(DOC))                    # the facts the freeze fills are still open
        else:
            self.assertFalse(CT.TODO_MARK.search(DOC))

    def test_fill_refuses_an_unfilled_bracket(self):
        with self.assertRaises(ValueError):
            CT.fill("a [X] b [Y]", dict(X=1.0))
        self.assertEqual(CT.fill("a [X]% b", dict(X=None)), "a not available b")
        self.assertEqual(CT.fill("a [X]% b", dict(X=12.345)), "a 12.3% b")
        self.assertEqual(CT.fill("[one clause]", {}, specials={"[one clause]": "z"}), "z")

    def test_rounding_and_writing(self):
        self.assertEqual(CT.fmt(-0.04), "0.0")
        self.assertEqual(CT.fmt(-0.06), "-0.1")
        self.assertEqual(CT.fmt(None), CT.NOT_AVAILABLE)
        self.assertEqual(CT.fmt(float("inf")), CT.NOT_AVAILABLE)
        self.assertEqual(CT.fmt(3, "count"), "3")
        self.assertEqual(CT.fmt(0.4, "percent"), "40.0")
        self.assertEqual(CT.join_items(["a", "b", "c"]), "a, b and c")

    def test_check_templates(self):
        CT.check_templates(BLOCK)
        bad = dict(BLOCK, sentences={k: v for k, v in BLOCK["sentences"].items() if k != "c1.pass"})
        with self.assertRaises(ValueError):
            CT.check_templates(bad)


# --------------------------------------------------------------------------------------------- C1

class TestC1(Base):
    def test_all_pass(self):
        look = self.look()
        self.assertEqual(look["outcomes"]["c1"], "pass")
        self.assertEqual(self.ids(look)[:5], ["always_attached", "c1.pass", "c1.certified", "c2.holds", "c3.fails"])
        self.assertEqual(self.texts(look)[1], expect("c1.pass", **PASS1))
        self.assertEqual(self.texts(look)[2], expect("c1.certified", **CERT1))
        self.assertEqual(self.texts(look)[0], CT.fill(BLOCK["always_attached"], COMMON))
        self.assertEqual(look["claims"]["c1"]["status"], "pass")
        self.assertEqual(look["brackets"]["K"], len(SEEDS))
        self.assertEqual(look["brackets"]["F"], 0)

    def test_the_lead_is_over_the_six_rivals_with_step_2s_own_beside(self):
        look = self.look()
        nl = look["marks"]["c1"]["net_lead"]
        self.assertEqual(nl["rivals"], list(K.LEAD_RIVALS))
        self.assertEqual(set(nl["rival_nets"]), set(K.LEAD_RIVALS))
        self.assertAlmostEqual(nl["rival_nets"]["range_hinge"], -20.0)
        self.assertAlmostEqual(nl["net"], 29.0)
        self.assertAlmostEqual(nl["lead"], 29.0)
        self.assertEqual(nl["best_rival"], "even_pull")                 # ties among 0s go to the first in order
        self.assertEqual(nl["step2_lead"]["rivals"], list(K.WORLD_BLIND))
        self.assertAlmostEqual(nl["step2_lead"]["lead"], 29.0)
        S = P.step2()
        self.assertTrue(hasattr(S, "go_no_go") and hasattr(S, "net"))

    def test_the_hand_set_range_limit_can_take_the_lead(self):
        """range_hinge in the rival set: when it nets more than the five, the lead is against it; step 2's own lead
        (its five rivals) is reported beside and may still pass."""
        pts = override(default_points, [("stall", "range_hinge", SC, None, 78.0), ("no_stall", "range_hinge", SC,
                                                                                    None, 89.0)])
        look = self.look(dict(points=pts))
        nl = look["marks"]["c1"]["net_lead"]
        self.assertEqual(nl["best_rival"], "range_hinge")
        self.assertAlmostEqual(nl["lead"], 29.0 - 27.0)
        self.assertFalse(nl["passed"])
        self.assertTrue(nl["step2_lead"]["passed"])
        self.assertEqual(look["outcomes"]["c1"], "fail")
        clause = expect("c1.clause.net", NET="29.0", D="2.0", R="the hand-set range limit")
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 clause)
                         .replace("[last sentence]", T["c1.last.numbers"]))

    def test_fail_net_alone(self):
        pts = override(default_points, [("stall", "obca_forced", SC, None, 79.0), ("no_stall", "obca_forced", SC, None,
                                                                                   88.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c1"], "fail")
        self.assertFalse(look["outcomes"]["c1_by_stop"])
        clause = expect("c1.clause.net", NET="29.0", D="2.0", R="the always-on ramp")
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 clause)
                         .replace("[last sentence]", T["c1.last.numbers"]))
        self.assertEqual(self.ids(look)[2], "c1.certified")                # always after C1 on the fresh seeds

    def test_fail_keeps_and_even_pull(self):
        pts = override(default_points, [("stall", "obca_k1", "big_waves", None, 60.0), ("stall", "even_pull", SC, None,
                                                                                         70.0)])
        look = self.look(dict(points=pts))
        c = [expect("c1.clause.keeps", L="10.0", situation="big waves"), expect("c1.clause.even_pull", E="10.0")]
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 "; ".join(c))
                         .replace("[last sentence]", T["c1.last.numbers"]))

    def test_a_failed_seed_fails_every_mark_with_its_own_clause(self):
        look = self.look(dict(data_main_fail={SEEDS[1]: ("the ball's leak checks failed", "leak")}))
        self.assertEqual(look["outcomes"]["c1"], "fail")
        self.assertTrue(look["outcomes"]["c1_by_stop"])
        self.assertEqual(look["failed_checks"], {str(SEEDS[1]): "the ball's leak checks"})
        clause = expect("c1.clause.failed_seed", s=str(SEEDS[1]), check="the ball's leak checks")
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 clause)
                         .replace("[last sentence]", T["c1.last.stop"]))
        # the failed seed is left out of the validity conditions and of the certified seeds
        self.assertEqual(look["validity"]["claims"]["seeds"], [s for s in SEEDS if s != SEEDS[1]])
        self.assertTrue(look["validity"]["claims"]["passed"])
        self.assertEqual(look["marks"]["c2"]["certified"], [s for s in SEEDS if s != SEEDS[1]])
        # C2 and C3 are written with the failed-seed sentence, never with pooled numbers
        self.assertEqual(look["outcomes"]["c2"], "fail_failed_seed")
        self.assertEqual(look["outcomes"]["c3"], "fail_failed_seed")
        self.assertIn(expect("c2.failed_seed", **COMMON).replace("seed [seeds]", f"seed {SEEDS[1]}"),
                      self.texts(look))
        self.assertNotIn("c3.fails", self.ids(look))

    def test_a_stopped_phase_fails_the_seed_with_the_stop_clause(self):
        look = self.look(dict(stops=[dict(phase_kind="critic", setting="main", world="stall", seed=SEEDS[2],
                                          reason="a learner that fails")],
                              missing_values=[("stall", SEEDS[2]), ("no_stall", SEEDS[2])]))
        self.assertIn(str(SEEDS[2]), look["failed_checks"])
        self.assertEqual(look["failed_checks"][str(SEEDS[2])], "its critic phase (main setting, stalling world)")
        self.assertEqual(look["outcomes"]["c1"], "fail")
        self.assertTrue(look["outcomes"]["c1_by_stop"])
        clause = expect("c1.clause.stopped_seed", s=str(SEEDS[2]), phase="the critic (main setting)",
                        reason="a learner that fails")
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 clause)
                         .replace("[last sentence]", T["c1.last.stop"]))
        self.assertEqual(self.ids(look)[-1], "stop.phase")
        self.assertEqual(self.texts(look)[-1], expect("stop.phase", s=str(SEEDS[2]), world="stalling world",
                                                      phase="the critic (main setting)", reason="a learner that fails"))

    def test_a_stop_makes_every_pooled_number_unavailable(self):
        """Every mark needs every seed (step 2's mean_or_none), so with a stopped seed no mark is missed on its
        numbers: only the stop clause appears, with the stop form of the last sentence, whatever the other seeds'
        numbers say."""
        pts = override(default_points, [("stall", "even_pull", SC, None, 70.0)])
        look = self.look(dict(points=pts, stops=[dict(phase_kind="clone", setting="main", world="no_stall",
                                                     seed=SEEDS[4], reason="x")],
                              missing_values=[("no_stall", SEEDS[4])]))
        self.assertTrue(look["outcomes"]["c1_by_stop"])
        self.assertIsNone(look["marks"]["c1"]["even_pull"]["difference"])
        clause = expect("c1.clause.stopped_seed", s=str(SEEDS[4]), phase="the cloned actor (main setting)", reason="x")
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 clause)
                         .replace("[last sentence]", T["c1.last.stop"]))
        # the marks on the seeds that ran are reported as description: the certified-seed sentence
        self.assertEqual(look["brackets"]["NC"], len(SEEDS) - 1)
        self.assertIn("reading", T["c1.certified"].replace("[reading]", "reading"))
        self.assertEqual(look["outcomes"]["c2"], "fail_failed_seed")

    def test_invalid_by_harm_share_and_both(self):
        pts = override(default_points, [("stall", "full_none", SC, None, 60.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c1"], "invalid")
        self.assertEqual(self.texts(look)[1], expect("c1.invalid", **COMMON, reason=expect("c1.reason.harm", H="10.0")))
        self.assertEqual(look["outcomes"]["c2"], "invalid")
        self.assertEqual(look["outcomes"]["c3"], "invalid")
        self.assertEqual([i for i in self.ids(look) if i.startswith(("c2", "c3"))], [])
        look = self.look(dict(past=lambda *a: 0.1), sub="b")
        self.assertIn(expect("c1.reason.past80", P="10.0"), self.texts(look)[1])
        look = self.look(dict(points=pts, past=lambda *a: 0.1), sub="c")
        both = expect("c1.reason.harm", H="10.0") + "; " + expect("c1.reason.past80", P="10.0")
        self.assertEqual(self.texts(look)[1], expect("c1.invalid", **COMMON, reason=both))

    def test_an_unmatched_rival_fails_c1_only(self):
        look = self.look(dict(missing_arms={("stall", SEEDS[3]): ["even_pull", "standin"]}))
        self.assertEqual(look["c1_only_checks"], {str(SEEDS[3]): P.RIVAL_MISSING_CHECK})
        self.assertEqual(look["failed_checks"], {})
        self.assertEqual(look["outcomes"]["c1"], "fail")
        self.assertEqual(look["outcomes"]["c2"], "pass")          # C2 reads the seed's numbers as usual
        self.assertIn(SEEDS[3], look["marks"]["c2"]["certified"])
        self.assertIn(expect("c1.clause.failed_seed", s=str(SEEDS[3]), check=P.RIVAL_MISSING_CHECK),
                      self.texts(look)[1])

    def test_missing_numbers_with_no_stop_fail_the_seed(self):
        look = self.look(dict(missing_arms={("stall", SEEDS[0]): ["full_none"]}))
        self.assertEqual(look["failed_checks"], {str(SEEDS[0]): P.EVALUATION_MISSING_CHECK})
        self.assertEqual(look["outcomes"]["c1"], "fail")
        look = self.look(dict(missing_arms={("stall", SEEDS[0]): ["range_hinge"]}), sub="b")    # a lead rival
        self.assertEqual(look["failed_checks"], {str(SEEDS[0]): P.EVALUATION_MISSING_CHECK})
        look = self.look(dict(missing_arms={("stall", SEEDS[0]): [TWIN2]}), sub="c")            # a twin
        self.assertEqual(look["failed_checks"], {str(SEEDS[0]): P.EVALUATION_MISSING_CHECK})


# --------------------------------------------------------------------------------------------- the refusal rule

class TestRefusal(Base):
    def test_a_refused_seed_counts_as_no_caution_in_c1_and_is_left_out_of_c2(self):
        look = self.look(dict(refused={SEEDS[1]: ["stall"]}))
        self.assertEqual(look["refused"], {str(SEEDS[1]): {"stall": 45}})
        self.assertEqual(look["failed_checks"], {})                 # not a failed seed
        self.assertEqual(look["c1_only_checks"], {})                # its unmatched rivals are substituted, not missing
        # C1: the refused seed-world's BCA arms and matched rivals took the no-caution actor's numbers
        self.assertEqual(look["outcomes"]["c1"], "pass")
        br = look["brackets"]
        self.assertEqual((br["K"], br["F"], br["NC"]), (len(SEEDS) - 1, 1, len(SEEDS) - 1))
        self.assertAlmostEqual(br["X"], 30.0 * (len(SEEDS) - 1) / len(SEEDS))
        # the no-stall world's height is finite on that seed, so BCA's switched ramp stood there and lost its own 1
        # point to no caution: the seed's net is 0 - 1
        self.assertAlmostEqual(br["NET"], (29.0 * (len(SEEDS) - 1) - 1.0) / len(SEEDS))
        self.assertAlmostEqual(br["E"], 25.0 * (len(SEEDS) - 1) / len(SEEDS))
        sub = look["reported"]["substituted"][str(SEEDS[1])]
        self.assertEqual(set(sub["stall"]), set(K.BCA_DEPENDENT) | set(P.MATCHED_RIVALS))
        self.assertEqual(set(sub["no_stall"]), {"obca_forced"} | set(P.MATCHED_RIVALS))
        # W: BCA's own no-stall loss of 1 on every seed (the refused seed's included); NET 27.0 - 1.0; the best rival
        # nets 0 (the matched even pull, substituted on the refused seed in both worlds), so the lead is 26.0
        self.assertEqual(self.texts(look)[1], expect("c1.pass", **dict(PASS1, X="27.0", W="1.0", NET="26.0", D="26.0",
                                                                      E="22.5", K=str(len(SEEDS) - 1), F="1")))
        # C2 and C3 on the certified seeds (the sign threshold ceil(0.8 x 9) = 8)
        self.assertEqual(look["marks"]["c2"]["certified"], [s for s in SEEDS if s != SEEDS[1]])
        self.assertEqual(look["marks"]["c2"]["positive_needed"], 8)
        self.assertEqual(look["outcomes"]["c2"], "pass")
        self.assertEqual(self.text_of(look, "c2.holds"), expect("c2.holds", **dict(HOLDS2, NC="9", K1="9", K2="9",
                                                                                 K3="9", A="20")))
        # the validity conditions count the refused seed (no caution and the full-log actor do not depend on BCA)
        self.assertEqual(look["validity"]["claims"]["seeds"], SEEDS)
        self.assertTrue(look["validity"]["claims"]["passed"])
        self.assertEqual(look["validity"]["noise_floor"]["seeds"], [s for s in SEEDS if s != SEEDS[1]])
        # the stop sentence names the seed-world with the chance of a finite height
        self.assertIn(expect("stop.refused", s=str(SEEDS[1]), world="stalling world", n="45", p=p_percent(45)),
                      self.texts(look))
        # the reported twin differences over every seed skip the refused one (no BCA number there)
        tw = look["reported"]["twins"][TWIN1]["over_every_seed"]
        self.assertEqual(sum(1 for v in tw["per_seed"] if v is not None), len(SEEDS) - 1)

    def test_a_refusal_lowers_the_lead_by_its_share_and_names_its_clause_when_c1_fails(self):
        pts = override(default_points, [("stall", "obca_forced", SC, None, 70.0), ("no_stall", "obca_forced", SC, None,
                                                                                   90.0)])
        look = self.look(dict(points=pts, refused={SEEDS[0]: ["stall", "no_stall"]}))
        # obca_forced nets 20 on nine seeds and 0 (no caution) on the refused one: 18; BCA 26.1; lead 8.1 < 10
        self.assertEqual(look["outcomes"]["c1"], "fail")
        nl = look["marks"]["c1"]["net_lead"]
        self.assertAlmostEqual(nl["lead"], 26.1 - 18.0)
        text = self.texts(look)[1]
        self.assertIn(expect("c1.clause.net", NET="26.1", D="8.1", R="the always-on ramp"), text)
        self.assertIn(expect("c1.clause.refused_seed", s=str(SEEDS[0]), n="45", p=p_percent(45),
                             world="stalling world and no-stall world"), text)
        self.assertTrue(text.endswith(T["c1.last.numbers"]))
        self.assertEqual(sum(1 for i in self.ids(look) if i == "stop.refused"), 2)

    def test_too_few_certified_seeds_leave_c2_and_c3_unread(self):
        look = self.look(dict(refused={s: ["stall"] for s in SEEDS[:3]}))
        self.assertEqual(look["marks"]["c2"]["n_certified"], len(SEEDS) - 3)
        self.assertTrue(look["marks"]["c2"]["not_read"])
        self.assertEqual(look["outcomes"]["c2"], "not_read")
        self.assertEqual(look["outcomes"]["c3"], "not_read")
        self.assertEqual(self.text_of(look, "c2.not_read"), expect("c2.not_read", NC=str(len(SEEDS) - 3), D1="12.0",
                                                                   D2="8.0", D3="28.0"))
        self.assertNotIn("c3.fails", self.ids(look))
        self.assertIsNone(look["marks"]["c2"]["twins"][TWIN1]["reading"])

    def test_every_seed_refused_leaves_the_noise_floor_unread(self):
        look = self.look(dict(refused={s: ["stall"] for s in SEEDS}))
        nf = look["validity"]["noise_floor"]
        self.assertTrue(nf["unmeasured"])
        self.assertIsNone(nf["passed"])
        self.assertEqual(look["outcomes"]["c2"], "not_read")
        self.assertEqual(look["brackets"]["K"], 0)
        self.assertEqual(look["outcomes"]["c1"], "fail")                 # every BCA number is no caution's
        self.assertIn(expect("c1.certified", NC="0", reading=CT.NOT_AVAILABLE, DC=CT.NOT_AVAILABLE,
                             RC=CT.NOT_AVAILABLE, LC=CT.NOT_AVAILABLE, EC=CT.NOT_AVAILABLE), self.texts(look))


# --------------------------------------------------------------------------------------------- C2 and C3

class TestC2C3(Base):
    def test_holds(self):
        look = self.look()
        self.assertEqual(look["outcomes"]["c2"], "pass")
        self.assertEqual(look["outcomes"]["c3"], "fails")
        self.assertEqual(self.text_of(look, "c2.holds"), expect("c2.holds", **HOLDS2))
        self.assertEqual(self.text_of(look, "c3.fails"), T["c3.fails"])
        for t, v in look["marks"]["c2"]["twins"].items():
            self.assertEqual(v["reading"], "ahead", t)
            self.assertEqual(v["bca"], K.NOISE_ARM)
        tw = look["marks"]["c2"]["twins"]
        self.assertEqual(tw[TWIN3]["difference"], 28.0)
        self.assertEqual(tw[TWIN1]["lower_bound"], 12.0)                  # every seed equal: sd 0
        self.assertEqual(look["claims"]["c2"]["twins"], {TWIN1: "ahead", TWIN2: "ahead", TWIN3: "ahead"})
        self.assertEqual(look["claims"]["c3"]["twins"], {TWIN1: "fails", TWIN2: "fails", TWIN3: "fails"})

    def test_readings_by_margin_and_sign(self):
        per = {s: (95.0 if i == 0 else 70.0) for i, s in enumerate(SEEDS)}       # one seed where twin 1 wins
        pts = override(default_points, [("stall", TWIN1, SC, None, per)])
        look = self.look(dict(points=pts))
        v = look["marks"]["c2"]["twins"][TWIN1]
        self.assertEqual(v["positive"], len(SEEDS) - 1)
        self.assertEqual(v["positive_needed"], 8)
        self.assertEqual(v["reading"], "ahead")                 # mean 9.5, positive 9 of 10
        # a 4.9 mean is a tie whatever the sign count
        pts = override(default_points, [("stall", TWIN1, SC, None, 77.1)])
        look = self.look(dict(points=pts), sub="b")
        v = look["marks"]["c2"]["twins"][TWIN1]
        self.assertAlmostEqual(v["difference"], 4.9)
        self.assertEqual(v["reading"], "tie")
        # positive in 7 of 10 with a large mean is a tie
        per = {s: (90.0 if i < 3 else 20.0) for i, s in enumerate(SEEDS)}
        pts = override(default_points, [("stall", TWIN1, SC, None, per)])
        look = self.look(dict(points=pts), sub="c")
        v = look["marks"]["c2"]["twins"][TWIN1]
        self.assertEqual((v["positive"], v["reading"]), (7, "tie"))
        self.assertGreater(v["difference"], 5.0)

    def test_fails_with_a_tie_and_the_other_last_sentence(self):
        pts = override(default_points, [("stall", TWIN1, SC, None, 79.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c2"], "fail")
        self.assertEqual(look["outcomes"]["c2_each"], {TWIN1: "tie", TWIN2: "ahead", TWIN3: "ahead"})
        clauses = "; ".join([expect("c2.clause.tie", twin=PLAIN[TWIN1], DT="3.0", LBT="3.0", KT=NS, NC=NS),
                             expect("c2.clause.ahead", twin=PLAIN[TWIN2], DT="8.0", KT=NS, NC=NS),
                             expect("c2.clause.ahead", twin=PLAIN[TWIN3], DT="28.0", KT=NS, NC=NS)])
        self.assertEqual(self.text_of(look, "c2.fails"),
                         expect("c2.fails", NC=NS).replace("[one clause per twin]", clauses)
                         .replace("[last sentence]", T["c2.last.other"]))
        self.assertEqual(self.text_of(look, "c3.fails"), T["c3.fails"])

    def test_the_dose_sentence(self):
        pts = override(default_points, [("stall", TWIN2, SC, None, 80.0), ("stall", TWIN3, SC, None, 84.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c2_each"], {TWIN1: "ahead", TWIN2: "tie", TWIN3: "tie"})
        self.assertTrue(self.text_of(look, "c2.fails").endswith(T["c2.last.dose"]))

    def test_all_tied(self):
        pts = override(default_points, [("stall", TWIN1, SC, None, 79.0), ("stall", TWIN2, SC, None, 80.0),
                                        ("stall", TWIN3, SC, None, 84.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(set(look["outcomes"]["c2_each"].values()), {"tie"})
        self.assertTrue(self.text_of(look, "c2.fails").endswith(T["c2.last.none"]))

    def test_a_twin_ahead_holds_c3(self):
        pts = override(default_points, [("stall", TWIN2, SC, None, 90.0), ("no_stall", TWIN2, SC, None, 89.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c2"], "fail")
        self.assertEqual(look["outcomes"]["c3"], "holds")
        self.assertEqual(look["outcomes"]["c3_each"], {TWIN1: "fails", TWIN2: "holds", TWIN3: "fails"})
        v = look["marks"]["c2"]["twins"][TWIN2]
        self.assertEqual((v["reading"], v["positive"]), ("behind", 0))
        self.assertAlmostEqual(v["difference"], -11.0)
        text = self.text_of(look, "c2.fails")
        self.assertIn(expect("c2.clause.behind", twin=PLAIN[TWIN2], DT="-11.0", KT="0", NC=NS), text)
        self.assertTrue(text.endswith(expect("c2.last.twin", twin=PLAIN[TWIN2])))
        self.assertEqual(self.text_of(look, "c3.holds"), expect("c3.holds", NC=NS, twin=PLAIN[TWIN2], DT="-11.0",
                                                                KT="0"))
        self.assertNotIn("c3.fails", self.ids(look))
        self.assertEqual(look["claims"]["c3"]["status"], "holds")
        # positive in 2 of 10 with a mean below -5 is a tie, not behind (at most 1 allowed)
        per = {s: (55.0 if i < 2 else 96.0) for i, s in enumerate(SEEDS)}
        pts = override(default_points, [("stall", TWIN2, SC, None, per)])
        look = self.look(dict(points=pts), sub="b")
        v = look["marks"]["c2"]["twins"][TWIN2]
        self.assertEqual((v["positive"], v["reading"]), (2, "tie"))
        self.assertLess(v["difference"], -5.0)

    def test_refused_seed_clause_in_a_failing_c2(self):
        pts = override(default_points, [("stall", TWIN1, SC, None, 79.0)])
        look = self.look(dict(points=pts, refused={SEEDS[2]: ["no_stall"]}))
        self.assertEqual(look["outcomes"]["c2"], "fail")
        text = self.text_of(look, "c2.fails")
        self.assertIn(expect("c2.clause.refused_seed", s=str(SEEDS[2])), text)
        self.assertIn(f"positive in 9 of 9 seeds", text)

    def test_not_fixed(self):
        with mock.patch.object(K, "MARKS_C2", dict(K.MARKS_C2, net_margin=None)):
            look = self.look()
        self.assertEqual(look["outcomes"]["c2"], "not_fixed")
        self.assertEqual(look["outcomes"]["c3"], "not_fixed")
        self.assertEqual(self.text_of(look, "c2.not_fixed"), T["c2.not_fixed"])
        self.assertIsNone(look["marks"]["c2"]["twins"][TWIN1]["passed"])
        self.assertIn("c2_net_margin", look["marks"]["c2"]["placeholders"])

    def test_dropped_by_the_noise_floor(self):
        pts = override(default_points, [("stall", "obca_k1_alone", SC, None, 70.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c2"], "dropped")
        self.assertEqual(look["outcomes"]["c3"], "dropped")
        self.assertEqual(look["validity"]["noise_floor"]["V"], 10.0)
        self.assertEqual(self.text_of(look, "c2.dropped"), expect("c2.dropped", V="10.0", D1="3.0", D2="-1.0",
                                                                  D3="19.0"))

    def test_not_read_takes_precedence_over_dropped(self):
        pts = override(default_points, [("stall", "obca_k1_alone", SC, None, 70.0)])
        look = self.look(dict(points=pts, refused={s: ["stall"] for s in SEEDS[:3]}))
        self.assertEqual(look["outcomes"]["c2"], "not_read")

    def test_every_seed_failed(self):
        look = self.look(dict(data_main_fail={s: ("x", "leak") for s in SEEDS}))
        self.assertEqual(look["outcomes"]["c2"], "fail_failed_seed")      # before 'not read' (0 certified seeds)
        self.assertIn("c2.every_seed_failed", self.ids(look))
        self.assertTrue(look["validity"]["claims"]["passed"])        # cannot be read; each seed's clause tells it

    def test_a_failed_seed_fails_c2_before_the_noise_floor_is_mentioned(self):
        pts = override(default_points, [("stall", "obca_k1_alone", SC, None, 70.0)])
        look = self.look(dict(points=pts, data_main_fail={SEEDS[3]: ("x", "leak")}))
        self.assertEqual(look["outcomes"]["c2"], "fail_failed_seed")
        self.assertFalse(look["validity"]["noise_floor"]["passed"])

    def test_lower_bound_and_t_table(self):
        self.assertEqual(P.t_one_sided_95(9), 1.833)
        self.assertEqual(P.t_one_sided_95(40), P.T_95[30])
        self.assertIsNone(P.lower_bound_95([1.0]))
        xs = [10.0, 12.0, 8.0, 11.0, 9.0]
        import statistics
        want = statistics.mean(xs) - 2.132 * statistics.stdev(xs) / 5 ** 0.5
        self.assertAlmostEqual(P.lower_bound_95(xs), want)
        self.assertEqual(K.c2_positive_needed(10), 8)
        self.assertEqual(K.c2_positive_needed(9), 8)
        self.assertEqual(K.c2_positive_needed(8), 7)


# --------------------------------------------------------------------------------------------- the reported tables

class TestReported(Base):
    def test_switch_calls_ranks_ladder_and_composites(self):
        look = self.look(dict(refused={SEEDS[1]: ["stall"]}))
        rep = look["reported"]
        calls = rep["switch_calls"]
        self.assertEqual(calls["seed_worlds"], 2 * len(SEEDS))
        self.assertEqual(calls["agree"], dict(meandiff=2 * len(SEEDS), conformal=2 * len(SEEDS), rank=2 * len(SEEDS)))
        self.assertEqual(look["brackets"]["A"], 2 * len(SEEDS))
        sw = rep["switches"]["stall"][str(SEEDS[0])]
        self.assertEqual(sw["ranks"]["n"], 45)
        self.assertEqual(sw["ranks"]["bca_threshold"], 1)                 # 2.46 is above every synthetic score
        self.assertAlmostEqual(sw["heights"]["plain_gap"], 1.2)
        self.assertEqual(sw["certify_chance"], K.certify_chance(45))
        self.assertIsNone(rep["switches"]["stall"][str(SEEDS[1])]["ranks"]["bca_threshold"])   # refused: inf
        ladder = rep["dose_ladder"][str(SEEDS[0])]
        self.assertEqual([r["arm"] for r in ladder], list(P.LADDER))
        self.assertEqual([r["points"] for r in ladder], [65.0, 75.0, 70.0, 80.0, 74.0, 78.0, 50.0])
        self.assertEqual(ladder[3]["height"], 2.5)
        comp = rep["composites"]
        # BCA's switch turning on the range limit: stall on -> range_hinge 50 - 50 = 0; no-stall off -> none: net 0
        self.assertAlmostEqual(comp["bca_switch_range_limit"]["net"], 0.0)
        # refusal means the range limit, read per seed-world: BCA where it set a height (29 on nine seeds; on the
        # refused seed its no-stall world, where it stood: a loss of 1), the range limit where it refused (the
        # stalling world of that seed: 50 - 50 = 0): a net of -1 on the refused seed
        self.assertAlmostEqual(comp["refusal_means_range_limit"]["net"], (29.0 * (len(SEEDS) - 1) - 1.0) / len(SEEDS))
        tw = rep["twins"][TWIN1]
        self.assertEqual(tw["against_batch"]["bca"], K.PRIMARY)
        self.assertAlmostEqual(tw["against_batch"]["mean"], 13.0)
        self.assertEqual(rep["certified"], [s for s in SEEDS if s != SEEDS[1]])
        md = P.look_md(look, "0" * 64)
        for head in ("## The switch calls", "## The dose ladder", "## Reported composites", "rank of BCA's threshold"):
            self.assertIn(head, md)


# --------------------------------------------------------------------------------------------- the count, the smoke

class TestCountAndSmoke(Base):
    def test_count_sentence_always(self):
        look = self.look(dict(counts=lambda s, w: 20 if (s, w) == (SEEDS[0], "stall") else 45))
        c = look["counts"]
        self.assertEqual((c["NMIN"], c["NMAX"], c["NREACH"], c["NSW"]), (20, 45, 2 * len(SEEDS) - 1, 2 * len(SEEDS)))
        want = expect("count.reported", **dict(COUNTS, NMIN="20", NREACH=str(2 * len(SEEDS) - 1), N8MIN="5"))
        self.assertIn(want, self.texts(look))

    def test_smoke_prefix_and_substitution(self):
        look = self.look(dict(group="smoke"))
        self.assertEqual(look["group"], "smoke")
        self.assertEqual(look["claims"]["c1"]["status"], P.SMOKE_STATUS)
        self.assertEqual(look["claims"]["c1"]["smoke_outcome"], "pass")
        t1 = self.texts(look)[0]
        self.assertTrue(t1.startswith(PREFIX + " "))
        self.assertIn("the plumbing's seed 0", t1)
        self.assertNotIn("fresh seeds", t1)
        self.assertNotIn("always_attached", self.ids(look))
        self.assertNotIn("c1.certified", self.ids(look))
        self.assertEqual(look["seeds"], [0])
        self.assertEqual(look["outcomes"]["c2"], "not_read")                  # one seed: fewer than 8
        self.assertTrue(self.text_of(look, "c2.not_read").startswith(PREFIX + " "))


# --------------------------------------------------------------------------------------------- files, modes, report

class TestFilesAndModes(Base):
    def test_run_look_writes_once_and_the_fields_the_runner_reads(self):
        root = self.tree()
        prereg = dict(constants=P.canonical_constants())
        look = P.run_look("fresh", root, "scratch", prereg)
        body, digest = P.read_json_verified(root / "looks" / "fresh.json")
        self.assertEqual(body["kind"], "near80-v9-look-fresh")
        self.assertEqual((body["group"], body["prereg_sha256"]), ("fresh", None))
        self.assertEqual(body["sentences"], look["sentences"])
        self.assertEqual(set(body["claims"]), {"c1", "c2", "c3"})
        md = (root / "looks" / "fresh.md").read_text(encoding="utf8")
        self.assertIn(digest, md)
        self.assertIn(look["sentences"][0]["text"], md)
        self.assertIn(f"data/main/stall/seed{SEEDS[0]}/data.json", body["inputs"])
        with self.assertRaises(P.LookRefused):
            P.run_look("fresh", root, "scratch", prereg)
        with self.assertRaises(P.LookRefused):
            P.run_look("fresh", root, "scratch", dict(constants={"ALPHA": 1}))
        never = K.ROOT / "runs" / "shift_explore" / "near80_no_such_folder"
        with self.assertRaises(P.LookRefused):
            P.run_look("fresh", never, "scratch", prereg)
        self.assertFalse(never.exists())

    def test_official_refuses_a_scratch_record_and_another_fingerprint(self):
        good = "a" * 64
        root = self.tree(dict(mode="official", prereg_sha256=good))
        with mock.patch.object(K, "OUT", root):
            R = P.load_results(root, "fresh", "official", prereg_sha256=good)
            self.assertEqual(R.mode, "official")
            with self.assertRaises(P.LookRefused) as e:
                P.load_results(root, "fresh", "official", prereg_sha256="b" * 64)
            self.assertIn("another pre-registration", str(e.exception))
        rel = K.LAYOUT["values"].format(setting="main", world="stall", seed=SEEDS[2])
        body, _ = P.read_json_verified(root / rel)
        put(root / rel, dict(body, mode="scratch"))
        with mock.patch.object(K, "OUT", root):
            with self.assertRaises(P.LookRefused) as e:
                P.load_results(root, "fresh", "official", prereg_sha256=good)
            self.assertIn("not an official record", str(e.exception))
        with self.assertRaises(P.LookRefused):
            P.load_results(root, "fresh", "official", prereg_sha256=good)   # not the official folder

    def test_tampered_and_mismatched_records_are_refused(self):
        root = self.tree()
        rel = K.LAYOUT["values"].format(setting="main", world="stall", seed=SEEDS[1])
        (root / rel).write_text((root / rel).read_text().replace("80.0", "81.0"))
        with self.assertRaises(P.LookRefused) as e:
            P.load_results(root, "fresh", "scratch")
        self.assertIn("sidecar", str(e.exception))
        root = self.tree(sub="b")
        body, _ = P.read_json_verified(root / rel)
        put(root / rel, dict(body, lock_sha256="0" * 64))
        with self.assertRaises(P.LookRefused) as e:
            P.load_results(root, "fresh", "scratch")
        self.assertIn("not the evaluation of the lock", str(e.exception))

    def test_seeds_from_the_plan_and_used_seeds_refused(self):
        root = self.tree(dict(seeds=SEEDS[:2], plan="seeds:" + ",".join(str(s) for s in SEEDS[:2])))
        R = P.load_results(root, "fresh", "scratch")
        self.assertEqual(R.seeds, SEEDS[:2])
        self.assertEqual(P.resolve_seeds("smoke", None, "smoke", "scratch"), [0])
        self.assertEqual(R.pts["main"]["stall"][SEEDS[0]]["obca_k1"][SC], 80.0)
        for bad in ([5], [10, SEEDS[0]], [20], [50], [60]):
            with self.assertRaises(P.LookRefused):
                P.resolve_seeds("fresh", bad, None, "scratch")
        with self.assertRaises(P.LookRefused):
            P.resolve_seeds("fresh", SEEDS[:2], None, "official")

    def test_halt_report(self):
        root = self.tree(dict(refused={SEEDS[0]: ["no_stall"]},
                              stops=[dict(phase_kind="policy", setting="main", world="stall", seed=SEEDS[2],
                                          reason="the height-0 check failed")],
                              plumbing={"stall": {"obca_k1": dict(passed=False, plain="step 2's switched-ramp actor"),
                                                  "critic": dict(passed=True, plain="the main test's critic")}}))
        rep = P.halt_report(root, mode="scratch")
        self.assertEqual(rep["stop"], "stop.reproduction")
        self.assertEqual(rep["sentences"][0]["text"], T["stop.reproduction"].replace(
            "[the main test's removal / critic / cloned actor / no-caution actor, or step 2's plumbing-check / "
            "switched-ramp / hand-set range-limit actor]", "step 2's switched-ramp actor"))
        self.assertEqual([r["id"] for r in rep["sentences"]], ["stop.reproduction", "stop.refused", "stop.phase"])
        self.assertEqual(rep["refused"], [(SEEDS[0], "no_stall", 45)])
        self.assertEqual(rep["sentences"][1]["text"], expect("stop.refused", s=str(SEEDS[0]), world="no-stall world",
                                                             n="45", p=p_percent(45)))

    def test_main_cli_refuses_official_looks(self):
        self.assertEqual(P.main(["fresh", "--out-root", str(self.tmp), "--mode", "official"]), K.EXIT_REFUSED)

    def test_nothing_written_into_the_code_folder(self):
        self.assertFalse((HERE / "__pycache__").exists())
        for name in ("pooled.py", "claims_text.py", "test_pooled.py"):
            self.assertIn(name, K.CODE_FILES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
