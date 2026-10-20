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
FIXED = dict(net_margin=5.0, positive_seeds_min=4, multiplicity="all_three")
SEEDS = list(K.FRESH)
NS = str(len(SEEDS))
RANGE = f"{SEEDS[0]}-{SEEDS[-1]}"


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


# Stalling world, strong current: BCA's switched ramp 80, the twins below it; no-stall world: BCA 89, twins a bit
# lower (so BCA's net beats each twin's by a clear margin).
STALL = dict(none=50, full_none=100, clone=40, w0_check=50, obca_k1=80, obca_k2=78, obca_forced=80, fixed_gated=75,
             conformal_k1=70, meangap_k1=65, meandiff_k1=66, meandiff_limit=60, obca_k1_alone=79, perfect_band=90,
             range_hinge=50, even_pull=55, push_less=54, random_1=52, random_2=51, standin=60)
NO_STALL = dict(none=90, obca_k1=89, obca_k2=88, obca_forced=60, fixed_gated=85, conformal_k1=86, meangap_k1=86,
                meandiff_k1=86, meandiff_limit=80, obca_k1_alone=89, perfect_band=88, range_hinge=70, standin=70,
                even_pull=85, push_less=84, random_1=88, random_2=89, clone=80, w0_check=90)


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
      data_main_fail      {seed: (reason, 'leak' | 'marks')}
      refused             {seed: [worlds]}: BCA's certificate on with an infinite height there
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
    refused = spec.get("refused", {})
    switch_on = spec.get("switch_on", lambda w, s: w == "stall")
    counts = spec.get("counts", lambda s, w: 45)
    locks = {}
    for s in seeds:
        for w in K.WORLDS:
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
                    certification=dict(reaches_min_rows=bool(n >= 29)),
                    version8_groups=dict(n_full_throttle=max(n - 20, 5), n_rest=700))))
        for w in K.WORLDS:
            ref = s in refused and w in refused[s]
            on = bool(switch_on(w, s)) or ref
            put(root / K.LAYOUT["switch"].format(setting="main", world=w, seed=s), dict(
                env, kind="near80-switch", phase_id=K.phase_id("switch", "main", w, s), setting="main", world=w,
                seed=s, t_lo=0.79, t_up=0.98, defined=True, n_survivors=30, groups_rule="pooled",
                certificate=dict(on=on, height="inf" if ref else 2.5, q05=0.4, finite_share_tail=0.85 if ref else 0.99),
                groups=dict(n_full_throttle=counts(s, w), n_rest=4000), rank_test=dict(p=0.01),
                refused=bool(ref), refusal="BCA's switch is on but its height is not finite" if ref else None))
            if (w, s) in missing_values:
                continue
            gone = set(missing_arms.get((w, s), [])) | set(missing_arms.get(w, []))
            if s in refused:
                gone |= set(K.BCA_DEPENDENT) | set(K.MATCHED) | {"standin"}
            ids = {}
            for a in K.ARMS["main"]:
                ids[a] = None if (a in gone or (w != "stall" and a in K.STALL_ONLY_ARMS)) else a
            lock_sha = put(root / K.LAYOUT["policy_lock"].format(setting="main", world=w, seed=s), dict(
                env, kind="near80-policy-lock", phase_id=K.phase_id("policy", "main", w, s),
                setting="main", world=w, seed=s, group=group, arm_ids=ids,
                refused_arms=sorted(K.BCA_DEPENDENT) if s in refused else [], bca_refused=s in refused,
                strengths={n_: 1.0 for n_ in ids.values() if n_}, matched_names={}, standin=dict(arm="standin"),
                arm_table=dict(t_lo=0.79, t_up=0.98, arms={"obca_k1": dict(switch=True, h=2.5, k=1.0)})))
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
                refused_arms=sorted(K.BCA_DEPENDENT) if s in refused else [], situations=sits, evaluations={}))
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


LATER = ("[one clause per missed mark]", "[one clause per twin not beaten]", "[seeds]")   # filled by the tests


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
             E="25.0")
# nets: BCA 80-50 - (90-89) = 29; conformal 70-50 - (90-86) = 16; meandiff ramp 66-50 - 4 = 12; limit 60-50-10 = 0
HOLDS2 = dict(COMMON, D1="13.0", K1=NS, D2="17.0", K2=NS, D3="29.0", K3=NS)
COUNTS = dict(COMMON, NMIN="45", NMAX="45", NREACH=str(2 * len(SEEDS)), NSW=str(2 * len(SEEDS)), N8MIN="25",
              N8MAX="25")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="near80_test_pooled_"))
        self.assertFalse(P.inside(self.tmp, K.ROOT))
        self._fixed = mock.patch.object(K, "MARKS_C2", dict(FIXED))
        self._fixed.start()

    def tearDown(self):
        self._fixed.stop()
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


# --------------------------------------------------------------------------------------------- extraction

class TestExtraction(Base):
    def test_every_id_once_and_in_order(self):
        self.assertEqual(list(BLOCK["sentences"]), list(CT.SENTENCE_IDS))
        self.assertEqual(len(set(CT.SENTENCE_IDS)), len(CT.SENTENCE_IDS))

    def test_templates_have_no_quote_marks_or_blockquote_markers(self):
        for sid, s in BLOCK["sentences"].items():
            self.assertFalse(s["text"].startswith('"'), sid)
            self.assertNotIn("\n", s["text"], sid)
            self.assertNotIn("> ", s["text"], sid)

    def test_brackets_all_defined(self):
        defined = set(CT.table_bracket_names(BLOCK))
        for name in ("S", "N", "X", "H", "W", "NET", "D", "R", "E", "L", "D1", "K1", "D2", "K2", "D3", "K3", "DT", "KT",
                     "P", "V", "NSEEDS", "SEEDS", "NMIN", "NMAX", "NREACH", "NSW", "N8MIN", "N8MAX"):
            self.assertIn(name, defined)

    def test_todo_marks_are_reported(self):
        self.assertEqual(BLOCK["todo_marks"], ["TODO(DECISION.md: the margin)", "TODO(DECISION.md: the seed count)"])
        self.assertEqual(CT.todo_marks("x TODO(DECISION.md: a) y"), ["TODO(DECISION.md: a)"])

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
        self.assertEqual(self.ids(look)[:3], ["always_attached", "c1.pass", "c2.holds"])
        self.assertEqual(self.texts(look)[1], expect("c1.pass", **PASS1))
        self.assertEqual(self.texts(look)[0], CT.fill(BLOCK["always_attached"], COMMON))
        self.assertEqual(look["claims"]["c1"]["status"], "pass")

    def test_net_uses_step2s_own_functions(self):
        look = self.look()
        S = P.step2()
        self.assertTrue(hasattr(S, "go_no_go") and hasattr(S, "net"))
        self.assertAlmostEqual(look["marks"]["c1"]["net_lead"]["net"], 29.0)

    def test_fail_net_alone(self):
        pts = override(default_points, [("stall", "obca_forced", SC, None, 79.0), ("no_stall", "obca_forced", SC, None,
                                                                                   88.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c1"], "fail")
        clause = expect("c1.clause.net", NET="29.0", D="2.0", R="the always-on ramp")
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 clause))

    def test_fail_keeps_and_even_pull(self):
        pts = override(default_points, [("stall", "obca_k1", "big_waves", None, 60.0), ("stall", "even_pull", SC, None,
                                                                                         70.0)])
        look = self.look(dict(points=pts))
        c = [expect("c1.clause.keeps", L="10.0", situation="big waves"), expect("c1.clause.even_pull", E="10.0")]
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 "; ".join(c)))

    def test_a_failed_seed_fails_every_mark_with_its_own_clause(self):
        look = self.look(dict(data_main_fail={SEEDS[1]: ("the ball's leak checks failed", "leak")}))
        self.assertEqual(look["outcomes"]["c1"], "fail")
        self.assertEqual(look["failed_checks"], {str(SEEDS[1]): "the ball's leak checks"})
        clause = expect("c1.clause.failed_seed", s=str(SEEDS[1]), check="the ball's leak checks")
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 clause))
        # the failed seed is left out of the validity conditions
        self.assertEqual(look["validity"]["claims"]["seeds"], [s for s in SEEDS if s != SEEDS[1]])
        self.assertTrue(look["validity"]["claims"]["passed"])
        # C2 is written with the failed-seed sentence, never with pooled numbers
        self.assertEqual(look["outcomes"]["c2"], "fail_failed_seed")
        self.assertIn(expect("c2.failed_seed", **COMMON).replace("seed [seeds]", f"seed {SEEDS[1]}"),
                      self.texts(look))

    def test_a_stopped_phase_fails_the_seed(self):
        look = self.look(dict(stops=[dict(phase_kind="critic", setting="main", world="stall", seed=SEEDS[2],
                                          reason="a learner that fails")],
                              missing_values=[("stall", SEEDS[2]), ("no_stall", SEEDS[2])]))
        self.assertIn(str(SEEDS[2]), look["failed_checks"])
        self.assertEqual(look["failed_checks"][str(SEEDS[2])], "its critic phase (main setting, stalling world)")
        self.assertEqual(self.ids(look)[-1], "stop.phase")
        self.assertEqual(self.texts(look)[-1], expect("stop.phase", s=str(SEEDS[2]), world="stalling world",
                                                      phase="the critic (main setting)", reason="a learner that fails"))

    def test_invalid_by_harm_share_and_both(self):
        pts = override(default_points, [("stall", "full_none", SC, None, 60.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c1"], "invalid")
        self.assertEqual(self.texts(look)[1], expect("c1.invalid", **COMMON, reason=expect("c1.reason.harm", H="10.0")))
        self.assertEqual(look["outcomes"]["c2"], "invalid")
        self.assertEqual([i for i in self.ids(look) if i.startswith("c2")], [])
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
        self.assertIn(expect("c1.clause.failed_seed", s=str(SEEDS[3]), check=P.RIVAL_MISSING_CHECK),
                      self.texts(look)[1])

    def test_missing_numbers_with_no_stop_fail_the_seed(self):
        look = self.look(dict(missing_arms={("stall", SEEDS[0]): ["full_none"]}))
        self.assertEqual(look["failed_checks"], {str(SEEDS[0]): P.EVALUATION_MISSING_CHECK})
        self.assertEqual(look["outcomes"]["c1"], "fail")


# --------------------------------------------------------------------------------------------- the refusal rule

class TestRefusal(Base):
    def test_a_refused_seed_fails_the_bca_marks_only(self):
        look = self.look(dict(refused={SEEDS[1]: ["stall"]}))
        self.assertEqual(look["refused"], {str(SEEDS[1]): {"stall": 45}})
        self.assertEqual(look["failed_checks"], {})                 # not a failed seed
        self.assertEqual(look["outcomes"]["c1"], "fail")
        self.assertEqual(look["outcomes"]["c2"], "fail")
        clause = expect("c1.clause.refused_seed", s=str(SEEDS[1]), n="45", world="stalling world")
        self.assertEqual(self.texts(look)[1], expect("c1.fail", **COMMON).replace("[one clause per missed mark]",
                                                                                 clause))
        c2 = expect("c2.clause.refused_seed", s=str(SEEDS[1]), n="45", world="stalling world")
        self.assertEqual(self.texts(look)[2], expect("c2.fails", **COMMON).replace("[one clause per twin not beaten]",
                                                                                  c2))
        # the validity conditions count the refused seed (no caution and the full-log actor do not depend on BCA)
        self.assertEqual(look["validity"]["claims"]["seeds"], SEEDS)
        self.assertTrue(look["validity"]["claims"]["passed"])
        # the twins' numbers on the other seeds are reported
        tw = look["reported"]["twins"]["conformal_k1"]
        self.assertEqual(sum(1 for v in tw["per_seed"] if v is not None), len(SEEDS) - 1)
        self.assertIn(expect("stop.refused", s=str(SEEDS[1]), world="stalling world", n="45"), self.texts(look))
        self.assertEqual(look["validity"]["noise_floor"]["seeds"], [s for s in SEEDS if s != SEEDS[1]])

    def test_refused_in_both_worlds(self):
        look = self.look(dict(refused={SEEDS[4]: ["stall", "no_stall"]}))
        self.assertIn("stalling world and no-stall world", self.texts(look)[1])
        self.assertEqual(sum(1 for i in self.ids(look) if i == "stop.refused"), 2)

    def test_every_seed_refused_leaves_the_noise_floor_unread(self):
        look = self.look(dict(refused={s: ["stall"] for s in SEEDS}))
        nf = look["validity"]["noise_floor"]
        self.assertTrue(nf["unmeasured"])
        self.assertIsNone(nf["passed"])
        self.assertEqual(look["outcomes"]["c2"], "fail")


# --------------------------------------------------------------------------------------------- C2

class TestC2(Base):
    def test_holds(self):
        look = self.look()
        self.assertEqual(look["outcomes"]["c2"], "pass")
        self.assertEqual(self.texts(look)[2], expect("c2.holds", **HOLDS2))
        for t, v in look["marks"]["c2"]["twins"].items():
            self.assertTrue(v["passed"], t)
        self.assertEqual(look["marks"]["c2"]["twins"]["meandiff_limit"]["difference"], 29.0)

    def test_fails_one_twin(self):
        pts = override(default_points, [("stall", "conformal_k1", SC, None, 79.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c2"], "fail")
        clause = expect("c2.clause.twin", twin="the split-conformal switch", DT="4.0", KT=NS, NSEEDS=NS)
        self.assertEqual(self.texts(look)[2], expect("c2.fails", **COMMON).replace("[one clause per twin not beaten]",
                                                                                  clause))

    def test_positive_seeds_needed(self):
        per = {s: (95.0 if i == 0 else 20.0) for i, s in enumerate(SEEDS)}     # one seed where the twin wins
        pts = override(default_points, [("stall", "meandiff_k1", SC, None, per)])
        look = self.look(dict(points=pts))
        tw = look["marks"]["c2"]["twins"]["meandiff_k1"]
        self.assertEqual(tw["positive"], len(SEEDS) - 1)
        self.assertEqual(tw["passed"], len(SEEDS) - 1 >= FIXED["positive_seeds_min"] and tw["difference"] >= 5.0)

    def test_each_separately(self):
        pts = override(default_points, [("stall", "conformal_k1", SC, None, 79.0)])
        with mock.patch.object(K, "MARKS_C2", dict(FIXED, multiplicity="each_separately")):
            look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c2"], "each")
        self.assertEqual(look["outcomes"]["c2_each"], dict(conformal_k1="fail", meandiff_k1="pass",
                                                            meandiff_limit="pass"))
        ids = self.ids(look)
        self.assertEqual(ids[2:5], ["c2.each.fails", "c2.each.holds", "c2.each.holds"])
        self.assertEqual(self.texts(look)[2], expect("c2.each.fails", **COMMON, twin="the split-conformal switch",
                                                     DT="4.0", KT=NS))
        self.assertEqual(look["claims"]["c2"]["twins"], look["outcomes"]["c2_each"])

    def test_not_fixed(self):
        with mock.patch.object(K, "MARKS_C2", dict(net_margin=None, positive_seeds_min=None, multiplicity=None)):
            look = self.look()
        self.assertEqual(look["outcomes"]["c2"], "not_fixed")
        self.assertEqual(self.texts(look)[2], expect("c2.not_fixed", **COMMON))
        self.assertIsNone(look["marks"]["c2"]["twins"]["conformal_k1"]["passed"])
        self.assertIn("c2_net_margin", look["marks"]["c2"]["placeholders"])

    def test_dropped_by_the_noise_floor(self):
        pts = override(default_points, [("stall", "obca_k1_alone", SC, None, 70.0)])
        look = self.look(dict(points=pts))
        self.assertEqual(look["outcomes"]["c2"], "dropped")
        self.assertEqual(look["validity"]["noise_floor"]["V"], 10.0)
        self.assertEqual(self.texts(look)[2], expect("c2.dropped", **COMMON, V="10.0"))

    def test_every_seed_failed(self):
        look = self.look(dict(data_main_fail={s: ("x", "leak") for s in SEEDS}))
        self.assertEqual(look["outcomes"]["c2"], "fail_failed_seed")
        self.assertIn("c2.every_seed_failed", self.ids(look))
        self.assertTrue(look["validity"]["claims"]["passed"])        # cannot be read; each seed's clause tells it


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
        self.assertEqual(look["seeds"], [0])


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
        for bad in ([5], [10, SEEDS[0]], [20], [60]):
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

    def test_main_cli_refuses_official_looks(self):
        self.assertEqual(P.main(["fresh", "--out-root", str(self.tmp), "--mode", "official"]), K.EXIT_REFUSED)

    def test_nothing_written_into_the_code_folder(self):
        self.assertFalse((HERE / "__pycache__").exists())
        for name in ("pooled.py", "claims_text.py", "test_pooled.py"):
            self.assertIn(name, K.CODE_FILES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
