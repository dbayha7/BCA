"""Near-80 (version 9, DRAFT): tests of the looks (pooled.py) and of section 14's extraction and filling (claims_text.py).

Every test builds a synthetic result tree in the runner's record schemas with make_tree, in a temporary folder outside
the repository, then checks the exact sentences: each expected sentence is built here from the template text of the
real PREREGISTRATION.md by plain string replacement of the expected bracket strings. C2's arithmetic (the paired
differences, the one-sided paired t-test, the t-interval, the multiplicity levels) is checked against a computation
written here. Nothing is trained, no seed's data is generated or read, nothing is written into the repository.

Run: /tmp/claude-0/bcavenv/bin/python3 code/test_pooled.py   (from /tmp, PYTHONPATH=/home/user/bca)
"""
import sys

sys.dont_write_bytecode = True

import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402
import claims_text as CT  # noqa: E402

P = K.load_local("pooled")
DOC = (HERE.parent / "PREREGISTRATION.md").read_text(encoding="utf8")
BLOCK = CT.extract_section14(DOC)
T = {k: v["text"] for k, v in BLOCK["sentences"].items()}
PREFIX = BLOCK["exploratory_prefix"]
SC = "strong_current"
SEEDS = list(K.CONFIRMATORY)


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


STALL = dict(none=50, full_none=100, clone=40, w0_check=50, obca_k1=80, obca_k2=82, obca_forced=80, fixed_gated=75,
             conformal_k1=70, meangap_k1=65, meandiff_k1=66, meandiff_limit=55, perfect_band=90, range_hinge=55,
             even_pull=55, push_less=54, random_1=52, random_2=51, standin=60)
NO_STALL = dict(none=90, obca_k1=89, standin=70, even_pull=85, random_1=88, random_2=89, obca_forced=60, clone=80,
                w0_check=90, conformal_k1=89, meandiff_k1=88, meandiff_limit=80, obca_k2=88, range_hinge=84)
JITTER = {s: [0.0, 2.0, -1.0, 3.0, 1.0][i % 5] for i, s in enumerate(SEEDS + list(K.EXPLORATORY))}


def default_points(world, seed, arm, sit):
    if sit != SC:
        return 71.0 if arm == "obca_k1" else 70.0
    if world == "no_stall":
        return float(NO_STALL.get(arm, 85))
    v = float(STALL[arm])
    return v + JITTER[seed] if arm == "obca_k1" else v


def net_of(points, arm, seed):
    """Step 2's per-seed net across worlds, written here: the gain over no caution where full throttle stalls minus
    the loss to no caution where it helps (strong current)."""
    gain = points("stall", seed, arm, SC) - points("stall", seed, "none", SC)
    loss = points("no_stall", seed, "none", SC) - points("no_stall", seed, arm, SC)
    return gain - loss


def make_tree(tmp, spec=None):
    """Writes a synthetic result tree under tmp/out and returns its root. spec (all optional):
      group               'confirmatory' (default) or 'exploratory'
      seeds               the seeds written (default: the group's; exploratory: seed 0)
      mode                the records' mode ('scratch')
      points              f(world, seed, arm, situation) -> number (default_points)
      past                f(world, seed, arm, situation) -> share of steps past 80% (no caution 0.4, others 0.1)
      missing_arms        {(world, seed) | world: [logical ids with no actor]}
      missing_values      [(world, seed)]: no values.json
      data_main_fail      {seed: (reason, 'leak' | 'marks')}
      refused             {(world, seed): n_tail}: BCA's switch on with an infinite height there
      switch_on           f(world, seed) -> bool (default: on in the stalling world)
      counts              f(world, seed) -> pooled full-throttle rows (default 120)
      stops               [{phase_kind, world, seed, reason}]
      plumbing            the plumbing record's 'reproduction' block (written when given)"""
    spec = dict(spec or {})
    root = Path(tmp) / "out"
    group = spec.get("group", "confirmatory")
    seeds = spec.get("seeds", SEEDS if group == "confirmatory" else [0])
    mode = spec.get("mode", "scratch")
    env = dict(mode=mode, prereg_sha256=None, written="2026-10-20T00:00:00Z")
    pts = spec.get("points", default_points)
    past = spec.get("past", lambda w, s_, a, sit: 0.4 if a == "none" else 0.1)
    missing_arms = spec.get("missing_arms", {})
    missing_values = set(spec.get("missing_values", []))
    switch_on = spec.get("switch_on", lambda w, s: w == "stall")
    counts = spec.get("counts", lambda w, s: 120)
    refused = spec.get("refused", {})
    locks = {}
    for s in seeds:
        for w in K.WORLDS:
            fail = spec.get("data_main_fail", {}).get(s)
            n_tail = counts(w, s)
            put(root / K.LAYOUT["data_main"].format(world=w, seed=s), dict(
                env, kind="near80v9-data-main", phase_id=K.phase_id("data_main", "main", w, s), seed=s, world=w,
                group=group, ball_leak_checks_passed=not (fail and fail[1] == "leak"),
                gate_world_marks=None if w != "stall" else {f"T{i}": not (fail and fail[1] == "marks" and i == 2)
                                                             for i in range(1, 6)},
                seed_failed_reason=fail[0] if fail else None,
                group_count=dict(defined=True, n_tail=n_tail, n_rest=900, below_bar=n_tail < 36,
                                 certifiable=n_tail >= 36)))
            ref = refused.get((w, s))
            put(root / K.LAYOUT["switch"].format(setting="main", world=w, seed=s), dict(
                env, kind="near80v9-switch", phase_id=K.phase_id("switch", "main", w, s), setting="main", world=w,
                seed=s, t_lo=0.7, t_up=0.95, defined=True, n_survivors=12,
                certificate=dict(on=True if ref is not None else bool(switch_on(w, s)),
                                 height="inf" if ref is not None else 2.5, q05=0.4),
                groups=dict(n_full_throttle=ref if ref is not None else n_tail, n_rest=900),
                rank_test=dict(p=0.01)))
            if (w, s) in missing_values:
                continue
            gone = set(missing_arms.get((w, s), [])) | set(missing_arms.get(w, []))
            ids = {a: None if (a in gone or (w != "stall" and a in K.STALL_ONLY_ARMS)) else a for a in K.ARMS["main"]}
            lock_sha = put(root / K.LAYOUT["policy_lock"].format(setting="main", world=w, seed=s), dict(
                env, kind="near80v9-policy-lock", phase_id=K.phase_id("policy", "main", w, s), setting="main",
                world=w, seed=s, group=group, arm_ids=ids, strengths={n: 1.0 for n in ids.values() if n},
                matched_names={}, standin=dict(arm="standin"),
                arm_table=dict(t_lo=0.7, t_up=0.95, arms={"obca_k1": dict(switch=True, h=2.5, k=1.0)})))
            locks[K.phase_id("policy", "main", w, s)] = dict(lock_sha256=lock_sha)
            sits = {}
            for sit in K.SITUATIONS:
                t = {m: {} for m in P.MEASURES}
                for a, name in ids.items():
                    if name is None:
                        continue
                    p = pts(w, s, a, sit)
                    t["points"][name] = p
                    t["held70_points"][name] = None if p is None else p - 1.0
                    t["capped80_points"][name] = None if p is None else p + 0.5
                    t["unseen_gain"][name] = 1.0
                    t["overshoot_loss"][name] = 0.5
                    t["past80_share"][name] = past(w, s, a, sit)
                    t["zone_shares"][name] = [0.5, 0.3, 0.2]
                    t["lower_tail_points"][name] = None if p is None else p - 20
                sits[sit] = t
            put(root / K.LAYOUT["values"].format(setting="main", world=w, seed=s), dict(
                env, kind="near80v9-values", phase_id=K.phase_id("evaluate", "main", w, s), setting="main", world=w,
                seed=s, group=group, source="trained", lock_sha256=lock_sha, arm_ids=ids, situations=sits,
                evaluations={}))
    for st in spec.get("stops", []):
        pid = K.phase_id(st["phase_kind"], st.get("setting", "main"), st.get("world", "x"), st.get("seed"),
                         st.get("group"))
        put(root / K.LAYOUT["phase_stop"].format(phase_id=pid), dict(
            env, kind="near80v9-phase-stop", phase_id=pid, phase_kind=st["phase_kind"],
            setting=st.get("setting", "main"), world=st.get("world", "x"), seed=st.get("seed"),
            group=st.get("group") or group, reason=st["reason"]))
    put(root / K.LAYOUT["stage_lock"].format(group=group), dict(
        env, kind="near80v9-stage-lock", phase_id=K.phase_id("lock", group=group), group=group, locks=locks,
        plan=spec.get("plan", "full" if group == "confirmatory" else "smoke")))
    if spec.get("plumbing") is not None:
        put(root / K.LAYOUT["plumbing"], dict(kind="near80v9-plumbing", passed=False, prereg_sha256=None,
                                              reproduction=spec["plumbing"], written=env["written"]))
    return root


def override(base, rules):
    """A points function equal to `base` except where a rule (world, arm, situation, seed or None, value) applies
    (value may be a per-seed dict)."""
    def f(world, seed, arm, sit):
        for w, a, si, sd, v in rules:
            if w == world and a == arm and si == sit and (sd is None or sd == seed):
                return v[seed] if isinstance(v, dict) else v
        return base(world, seed, arm, sit)
    return f


def expect(sid, **br):
    s = T[sid]
    for k, v in br.items():
        s = s.replace(f"[{k}]", v)
    assert "[" not in s, s
    return s


@contextlib.contextmanager
def claims(**c2):
    """near80_constants.CLAIMS with C2's open values (and 'decision') filled for one test; restored after."""
    saved = copy.deepcopy(K.CLAIMS)
    try:
        decision = c2.pop("decision", K.CLAIMS["decision"])
        K.CLAIMS["C2"].update(c2)
        K.CLAIMS["decision"] = decision
        yield K.CLAIMS
    finally:
        K.CLAIMS.clear()
        K.CLAIMS.update(saved)


FILLED = dict(margins={"conformal_k1": 5.0, "meandiff_k1": 5.0, "meandiff_limit": 5.0}, statistic="mean",
              alpha=0.05, min_positive_seeds=None, multiplicity="all", noise_floor_max=None)

MEAN_NET = sum(net_of(default_points, "obca_k1", s) for s in SEEDS) / len(SEEDS)


def fmt(x):
    return CT.fmt(x)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="near80v9_test_pooled_"))
        self.assertFalse(str(self.tmp).startswith("/home/user/bca"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tree(self, spec=None, sub="a"):
        return make_tree(self.tmp / sub, spec)

    def look(self, spec=None, sub="a"):
        spec = dict(spec or {})
        root = self.tree(spec, sub)
        group = spec.get("group", "confirmatory")
        R = P.load_results(root, group, "scratch", seeds=spec.get("seeds"))
        return P.make_look(R, BLOCK, group)

    def texts(self, look):
        return [r["text"] for r in look["sentences"]]

    def ids(self, look):
        return [r["id"] for r in look["sentences"]]

    def sentence(self, look, sid):
        hits = [r["text"] for r in look["sentences"] if r["id"] == sid]
        self.assertEqual(len(hits), 1, (sid, self.ids(look)))
        return hits[0]


# --------------------------------------------------------------------------------------------- extraction

class TestExtraction(Base):
    def test_every_id_once_and_in_order(self):
        self.assertEqual(list(BLOCK["sentences"]), list(CT.SENTENCE_IDS))
        self.assertEqual(len(set(CT.SENTENCE_IDS)), len(CT.SENTENCE_IDS))

    def test_templates_have_no_quote_marks_or_blockquote_markers(self):
        for sid, text in T.items():
            self.assertNotIn('"', text, sid)
            self.assertNotIn("> ", text, sid)

    def test_brackets_all_defined(self):
        defined = set(CT.table_bracket_names(BLOCK))
        for sid, text in T.items():
            used = set(CT.CAPITAL_BRACKET.findall(text))
            self.assertTrue(used <= defined, (sid, used - defined))

    def test_filling_every_template_leaves_no_bracket(self):
        values = {n: 1.0 for n in CT.table_bracket_names(BLOCK)}
        values.update(seeds="35", s="35", check="a check", situation="big waves", world="stalling world",
                      phase="the critic", reason="a reason", value="1", threshold="2", T="the twin", R="a rival",
                      RESULT="met", MULT="every comparison", RULE="C2 holds", SEEDS="35-39")
        specials = {"[clauses]": "x", "[comparisons]": "y", P.SEEDS_WORDS: "seed 35"}
        for sid, text in T.items():
            if sid == "stop.reproduction":
                out = P.reproduction_sentence(text, ["the main test's critic"])
            else:
                out = CT.fill(text, values, specials)
            self.assertNotIn("[", out, sid)

    def test_fill_rules_are_the_14_6_bullets(self):
        sec = DOC[DOC.index("### 14.6"):DOC.index("## 15.")]
        self.assertEqual(len(BLOCK["fill_rules"]), sum(1 for line in sec.split("\n") if line.startswith("- ")))

    def test_refuses_a_missing_or_doubled_sentence_and_an_undefined_bracket(self):
        with self.assertRaises(ValueError):
            CT.extract_section14(DOC.replace("**If BCA refused:**", "**If BCA declined:**"))
        with self.assertRaises(ValueError):
            CT.extract_section14(DOC.replace("**Not shown:**", "**Shown:**"))
        with self.assertRaises(ValueError):
            CT.extract_section14(DOC.replace("held [GMIN] to [GMAX] rows", "held [GMIN] to [GZZ] rows"))
        with self.assertRaises(ValueError):
            CT.extract_section14(DOC.replace("### 14.6 How the code", "### 14.7 How the code"))

    def test_fill_refuses_an_unfilled_bracket(self):
        with self.assertRaises(ValueError):
            CT.fill("a [NET] b [D]", dict(NET=1.0))

    def test_rounding_and_writing(self):
        self.assertEqual(CT.fmt(-0.04), "0.0")
        self.assertEqual(CT.fmt(2.25), "2.2")
        self.assertEqual(CT.fmt(None), "not available")
        self.assertEqual(CT.fmt(float("inf")), "not available")
        self.assertEqual(CT.fmt(3.0, "count"), "3")
        self.assertEqual(CT.fmt(0.012345, "value"), "0.01235")
        self.assertEqual(CT.fmt(0.95, "percent"), "95.0")
        self.assertEqual(CT.fill("[X]%", dict(X=None)), "not available")


# --------------------------------------------------------------------------------------------- C1 and the count

class TestC1(Base):
    def test_all_pass_and_the_open_claims_block(self):
        lk = self.look()
        n = str(len(SEEDS))
        seeds = f"{SEEDS[0]}-{SEEDS[-1]}"
        self.assertEqual(self.ids(lk), ["always_attached", "count.always", "refusal.none", "c1.holds", "c2.scale"])
        self.assertEqual(lk["outcomes"], {"C1": "pass", "C2": "open"})
        self.assertEqual(lk["decision"]["status"], "open")
        self.assertEqual(self.sentence(lk, "count.always"), expect("count.always", GMIN="120", GMAX="120",
                                                                    NSW=str(2 * len(SEEDS)), GB="0"))
        self.assertEqual(self.sentence(lk, "refusal.none"), expect("refusal.none", NS=n))
        x = sum(default_points("stall", s, "obca_k1", SC) - 50 for s in SEEDS) / len(SEEDS)
        self.assertEqual(self.sentence(lk, "c1.holds"), expect(
            "c1.holds", SEEDS=seeds, S=n, N=n, NS=n, X=fmt(x), H="50.0", W="1.0", NET=fmt(MEAN_NET),
            D=fmt(MEAN_NET), R="the matched even pull", E=fmt(x - 5)))
        self.assertEqual(lk["sentences"][0]["text"], CT.fill(BLOCK["always_attached"], dict(NS=n, SEEDS=seeds)))

    def test_net_uses_step2s_own_functions(self):
        lk = self.look()
        S = P.step2()
        self.assertAlmostEqual(lk["marks"]["C1"]["net_lead"]["net"], MEAN_NET)
        self.assertIsNotNone(S.go_no_go)

    def test_fail_net_alone(self):
        pts = override(default_points, [("stall", "even_pull", SC, None, 76.0)])
        lk = self.look(dict(points=pts))
        self.assertEqual(lk["outcomes"]["C1"], "fail")
        best = MEAN_NET - (76 - 50 - 5)
        self.assertEqual(self.sentence(lk, "c1.fails"), expect(
            "c1.fails", SEEDS=f"{SEEDS[0]}-{SEEDS[-1]}", clauses=CT.fill(T["c1.clause.net"], dict(
                NET=MEAN_NET, D=best, R="the matched even pull")) + "; " + CT.fill(T["c1.clause.even_pull"], dict(
                    E=sum(default_points("stall", s, "obca_k1", SC) - 76 for s in SEEDS) / len(SEEDS)))))

    def test_fail_keeps_each_situation(self):
        pts = override(default_points, [("stall", "obca_k1", "big_waves", None, 60.0)])
        lk = self.look(dict(points=pts))
        text = self.sentence(lk, "c1.fails")
        self.assertIn(CT.fill(T["c1.clause.keeps"], dict(L=10.0, situation="big waves")), text)

    def test_a_failed_seed_is_told_by_its_own_clause(self):
        s = SEEDS[1]
        lk = self.look(dict(data_main_fail={s: ("the ball's leak checks failed", "leak")}))
        self.assertEqual(lk["outcomes"]["C1"], "fail")
        self.assertIn(CT.fill(T["c1.clause.failed_seed"], dict(s=str(s), check="the ball's leak checks")),
                      self.sentence(lk, "c1.fails"))
        self.assertIn(str(s), lk["failed_seeds"]["C2"])

    def test_a_missing_rival_fails_c1_only(self):
        s = SEEDS[2]
        lk = self.look(dict(missing_arms={("stall", s): ["random_2"]}))
        self.assertEqual(lk["failed_checks"]["C1"], {str(s): P.RIVAL_MISSING_CHECK})
        self.assertEqual(lk["failed_checks"]["C2"], {})

    def test_a_missing_c2_number_fails_c2_only(self):
        s = SEEDS[3]
        lk = self.look(dict(missing_arms={("no_stall", s): ["meandiff_limit"]}))
        self.assertEqual(lk["failed_checks"]["C2"], {str(s): P.EVALUATION_MISSING_CHECK})
        self.assertEqual(lk["failed_checks"]["C1"], {})

    def test_count_bracket_reads_the_data_records(self):
        lk = self.look(dict(counts=lambda w, s: 30 if (w, s) == ("stall", SEEDS[0]) else 200))
        self.assertEqual(self.sentence(lk, "count.always"), expect("count.always", GMIN="30", GMAX="200",
                                                                    NSW=str(2 * len(SEEDS)), GB="1"))


# --------------------------------------------------------------------------------------------- validity

class TestValidity(Base):
    def test_holds_by_default(self):
        lk = self.look()
        v = lk["marks"]["validity"]
        self.assertTrue(v["passed"])
        self.assertAlmostEqual(v["H"], 50.0)
        self.assertAlmostEqual(v["P"], 0.4)
        self.assertEqual(K.CLAIMS["validity"]["harm_min"], 30.0)
        self.assertEqual(K.CLAIMS["validity"]["past80_min"], 0.25)

    def _invalid(self, spec, reasons):
        with claims(**dict(FILLED, decision="C2")):
            lk = self.look(spec)
        self.assertEqual(lk["outcomes"], {"C1": "invalid", "C2": "invalid"})
        self.assertEqual(lk["decision"]["status"], "no conclusion")
        self.assertEqual(self.ids(lk), ["always_attached", "count.always", "refusal.none", "validity.fails",
                                        "c2.scale"])
        self.assertEqual(self.sentence(lk, "validity.fails"), CT.fill(
            T["validity.fails"], dict(SEEDS=f"{SEEDS[0]}-{SEEDS[-1]}"), specials={"[reason]": "; ".join(reasons)}))

    def test_too_little_harm_by_h_by_p_and_both(self):
        low_h = override(default_points, [("stall", "full_none", SC, None, 70.0)])
        low_p = lambda w, s, a, sit: 0.1
        rh = CT.fill(T["validity.reason.harm"], dict(H=20.0))
        rp = CT.fill(T["validity.reason.share"], dict(P="10.0"))
        self._invalid(dict(points=low_h), [rh])
        self._invalid(dict(past=low_p), [rp])
        self._invalid(dict(points=low_h, past=low_p), [rh, rp])

    def test_a_failed_seed_never_makes_validity_fail(self):
        s = SEEDS[0]
        pts = override(default_points, [("stall", "full_none", SC, s, 50.0)])
        lk = self.look(dict(points=pts, data_main_fail={s: ("the ball's leak checks failed", "leak")}))
        self.assertTrue(lk["marks"]["validity"]["passed"])
        lk = self.look(dict(refused={("stall", t): 19 for t in SEEDS}), sub="b")
        self.assertTrue(lk["marks"]["validity"]["passed"])                  # every seed failed: not read

    def test_a_missing_validity_number_fails_the_seed_for_c1(self):
        s = SEEDS[2]
        lk = self.look(dict(missing_arms={("stall", s): ["full_none"]}))
        self.assertEqual(lk["failed_checks"]["C1"], {str(s): P.EVALUATION_MISSING_CHECK})


# --------------------------------------------------------------------------------------------- the refusal rule

class TestRefusal(Base):
    def test_a_refusal_fails_both_claims_on_that_seed_and_says_so(self):
        s = SEEDS[0]
        with claims(**FILLED):
            lk = self.look(dict(refused={("stall", s): 19}))
        n = str(len(SEEDS))
        self.assertEqual(lk["refused"], {str(s): {"stall": dict(n_tail=19, q05=0.4)}})
        self.assertEqual(self.sentence(lk, "refusal.some"), CT.fill(
            T["refusal.some"], dict(RF="1", NS=n, seeds=str(s)), specials={P.SEEDS_WORDS: f"seed {s}"}))
        check = P.refused_check(19, "stall")
        self.assertEqual(lk["failed_checks"]["C1"], {str(s): check})
        self.assertEqual(lk["failed_checks"]["C2"], {str(s): check})
        self.assertIn(CT.fill(T["c1.clause.failed_seed"], dict(s=str(s), check=check)), self.sentence(lk, "c1.fails"))
        self.assertEqual(lk["outcomes"]["C2"], "fail_failed_seed")
        self.assertEqual(self.sentence(lk, "c2.failed_seed"), CT.fill(
            T["c2.failed_seed"], dict(seeds=str(s)), specials={P.SEEDS_WORDS: f"seed {s}"}))

    def test_a_refused_seed_keeps_its_other_arms_in_the_reported_tables(self):
        s = SEEDS[0]
        lk = self.look(dict(refused={("stall", s): 19}, missing_arms={("stall", s): list(K.BCA_HEIGHT_ARMS)
                                                                      + list(K.BCA_MATCHED_ARMS)}))
        rep = lk["reported"]["measures"]["points"]["stall"]
        self.assertIsNotNone(rep["meandiff_k1"][SC]["mean"])
        self.assertIsNotNone(rep["conformal_k1"][SC]["mean"])
        self.assertTrue(lk["reported"]["switches"]["stall"][str(s)]["refused"])
        self.assertEqual(set(lk["failed_seeds"]["C1"]), {str(s)})

    def test_a_switched_off_infinite_height_is_not_a_refusal(self):
        self.assertFalse(P.refused_cert(dict(on=False, height="inf")))
        self.assertTrue(P.refused_cert(dict(on=True, height="inf")))
        self.assertTrue(P.refused_cert(dict(on=True, height=None)))
        self.assertFalse(P.refused_cert(dict(on=True, height=2.0)))

    def test_every_seed_refused(self):
        with claims(**FILLED):
            lk = self.look(dict(refused={("stall", s): 18 for s in SEEDS}))
        self.assertEqual(lk["outcomes"], {"C1": "fail", "C2": "fail_failed_seed"})
        self.assertIn("c2.every_seed_failed", self.ids(lk))


# --------------------------------------------------------------------------------------------- C2

def diffs_of(points, twin):
    return [net_of(points, "obca_k1", s) - net_of(points, twin, s) for s in SEEDS]


class TestC2(Base):
    def test_open_while_any_value_is_todo(self):
        lk = self.look()
        c2 = lk["marks"]["C2"]
        self.assertTrue(c2["open"])
        self.assertIsNone(c2["passed"])
        for twin in K.CLAIMS["C2"]["comparators"]:
            self.assertEqual(c2["comparisons"][twin]["per_seed"], diffs_of(default_points, twin))
        self.assertNotIn("c2.holds", self.ids(lk))
        self.assertNotIn("c2.fails", self.ids(lk))

    def test_mean_statistic_holds_word_for_word(self):
        with claims(**FILLED):
            lk = self.look()
        self.assertEqual(lk["outcomes"]["C2"], "pass")
        parts = []
        for twin in K.CLAIMS["C2"]["comparators"]:
            d = diffs_of(default_points, twin)
            parts.append(CT.fill(T["c2.comparison.mean"], dict(
                T=K.ARM_PLAIN[twin], DIFF=sum(d) / len(d), MARGIN=5.0, KPOS=str(sum(1 for v in d if v > 0)),
                NS=str(len(SEEDS)), RESULT="met")))
        self.assertEqual(self.sentence(lk, "c2.holds"), CT.fill(T["c2.holds"], dict(
            SEEDS=f"{SEEDS[0]}-{SEEDS[-1]}", MULT=P.MULT_WORDS["all"]), specials={"[comparisons]": "; ".join(parts)}))

    def test_mean_statistic_fails_one_comparison(self):
        f = dict(FILLED, margins={"conformal_k1": 50.0, "meandiff_k1": 5.0, "meandiff_limit": 5.0})
        with claims(**f):
            lk = self.look()
        c = lk["marks"]["C2"]["comparisons"]
        self.assertEqual([c[t]["passed"] for t in ("conformal_k1", "meandiff_k1", "meandiff_limit")],
                         [False, True, True])
        self.assertEqual(lk["outcomes"]["C2"], "fail")
        self.assertIn("missed", self.sentence(lk, "c2.fails"))

    def test_paired_t_against_a_hand_computation_and_the_levels(self):
        from scipy import stats
        for mult, want_levels in (("all", [0.05] * 3), ("bonferroni", [0.05 / 3] * 3)):
            with claims(**dict(FILLED, statistic="paired_t", multiplicity=mult)):
                lk = self.look(sub=mult)
            for twin, lv in zip(("conformal_k1", "meandiff_k1", "meandiff_limit"), want_levels):
                c = lk["marks"]["C2"]["comparisons"][twin]
                d = diffs_of(default_points, twin)
                n, m = len(d), sum(d) / len(d)
                sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
                p = float(stats.t.sf((m - 5.0) / (sd / math.sqrt(n)), df=n - 1)) if sd > 0 else (0.0 if m > 5 else 1)
                self.assertAlmostEqual(c["p"], p, places=12)
                self.assertAlmostEqual(c["level"], lv)
                self.assertEqual(c["passed"], p < lv)

    def test_holm_steps_down_in_order_of_p(self):
        self.assertEqual(P.t_pvalue([1.0, 1.0], 0.0), 0.0)
        self.assertEqual(P.t_pvalue([1.0, 1.0], 2.0), 1.0)
        self.assertIsNone(P.t_pvalue([1.0], 0.0))
        self.assertIsNone(P.t_pvalue([1.0, None], 0.0))
        with claims(**dict(FILLED, statistic="paired_t", multiplicity="holm")):
            lk = self.look()
        comps = lk["marks"]["C2"]["comparisons"]
        order = sorted(comps, key=lambda t: comps[t]["p"])
        self.assertEqual([comps[t]["level"] for t in order], [0.05 / 3, 0.05 / 2, 0.05])

    def test_interval_shown_excluded_unresolved(self):
        from scipy import stats
        margins = {"conformal_k1": 5.0, "meandiff_k1": 200.0, "meandiff_limit": 35.5}
        with claims(**dict(FILLED, statistic="interval", multiplicity="bonferroni", margins=margins)):
            lk = self.look()
        comps = lk["marks"]["C2"]["comparisons"]
        for twin in comps:
            d = diffs_of(default_points, twin)
            n, m = len(d), sum(d) / len(d)
            sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
            half = float(stats.t.ppf(1 - 0.05 / 3 / 2, df=n - 1)) * sd / math.sqrt(n)
            lo, hi = comps[twin]["interval"]
            self.assertAlmostEqual(lo, m - half, places=12)
            self.assertAlmostEqual(hi, m + half, places=12)
        self.assertEqual(comps["conformal_k1"]["interval_result"], "shown")
        self.assertEqual(comps["meandiff_k1"]["interval_result"], "excluded")
        self.assertEqual(comps["meandiff_limit"]["interval_result"], "unresolved")
        self.assertEqual(lk["outcomes"]["C2"], "fail")
        text = self.sentence(lk, "c2.fails")
        self.assertIn("98.3% interval", text)
        self.assertIn(": excluded", text)
        with claims(**dict(FILLED, statistic="interval", multiplicity="holm")):
            with self.assertRaises(P.LookRefused):
                self.look(sub="holm")

    def test_min_positive_seeds(self):
        pts = override(default_points, [("stall", "conformal_k1", SC, SEEDS[0], 200.0)])
        with claims(**dict(FILLED, min_positive_seeds=len(SEEDS))):
            lk = self.look(dict(points=pts))
        c = lk["marks"]["C2"]["comparisons"]["conformal_k1"]
        self.assertEqual(c["positive"], len(SEEDS) - 1)
        self.assertFalse(c["passed"])

    def test_noise_floor_drops_c2(self):
        with claims(**dict(FILLED, noise_floor_max=0.01)):
            lk = self.look()
        self.assertEqual(lk["outcomes"]["C2"], "dropped")
        c2 = lk["marks"]["C2"]
        self.assertEqual(self.sentence(lk, "c2.dropped"), CT.fill(T["c2.dropped"], dict(
            T=K.ARM_PLAIN[c2["se_max_twin"]], V=c2["se_max"], NF=0.01)))

    def test_scale_sentence(self):
        lk = self.look()
        nets = {a: sum(net_of(default_points, a, s) for s in SEEDS) / len(SEEDS)
                for a in ("obca_k1", "conformal_k1", "meandiff_k1", "meandiff_limit", "range_hinge", "obca_k2")}
        self.assertEqual(self.sentence(lk, "c2.scale"), CT.fill(T["c2.scale"], dict(
            NETRAW=nets["obca_k1"], N1=nets["conformal_k1"], N2=nets["meandiff_k1"], N3=nets["meandiff_limit"],
            NZ=nets["range_hinge"], NK2=nets["obca_k2"])))

    def test_the_decision_rule(self):
        with claims(**dict(FILLED, decision="C2")):
            lk = self.look()
        self.assertEqual(lk["decision"]["status"], "shown")
        self.assertEqual(self.sentence(lk, "decision.shown"), expect("decision.shown", RULE="C2 holds"))
        pts = override(default_points, [("stall", "even_pull", SC, None, 79.0)])
        with claims(**dict(FILLED, decision="C1_and_C2")):
            lk = self.look(dict(points=pts), sub="b")
        self.assertEqual(lk["outcomes"]["C1"], "fail")
        self.assertEqual(self.sentence(lk, "decision.not_shown"), expect("decision.not_shown",
                                                                          RULE="C1 and C2 both hold"))
        with claims(**dict(FILLED, decision="something else")):
            with self.assertRaises(P.LookRefused):
                self.look(sub="c")


# --------------------------------------------------------------------------------------------- stops and the halt report

class TestStops(Base):
    def test_plumbing_prefix_and_substitution(self):
        lk = self.look(dict(group="exploratory"))
        self.assertNotIn("always_attached", self.ids(lk))
        for r in lk["sentences"]:
            self.assertTrue(r["text"].startswith(PREFIX + " "), r["id"])
        self.assertNotIn("fresh seeds", self.sentence(lk, "c1.holds"))
        self.assertEqual(lk["claims"]["C1"]["status"], P.EXPLORATORY_STATUS)

    def test_a_phase_stop_fails_the_seed_and_adds_its_sentence(self):
        s = SEEDS[4]
        lk = self.look(dict(stops=[dict(phase_kind="critic", world="stall", seed=s, reason="a NaN in training")],
                            missing_values=[("stall", s)]))
        self.assertIn(str(s), lk["failed_seeds"]["C1"])
        self.assertEqual(lk["failed_checks"]["C1"][str(s)], "its critic phase (stalling world)")
        self.assertEqual(self.sentence(lk, "stop.phase"), CT.fill(T["stop.phase"], dict(
            s=str(s), world="stalling world", phase="the critic", reason="a NaN in training")))

    def test_no_training_first_failed_rule(self):
        rules = [dict(rule="check 1: x", passed=True, value=True, threshold="all pass"),
                 dict(rule="check 3: BCA's two groups hold held-out rows and the survivors only", passed=False,
                      value=False, threshold="all pass")]
        self.assertEqual(P.halt_sentence_no_training(rules, BLOCK), CT.fill(T["stop.no_training"], dict(
            check="check 3, BCA's two groups hold held-out rows and the survivors only", value="failed",
            threshold="all pass")))
        with self.assertRaises(ValueError):
            P.halt_sentence_no_training(rules[:1], BLOCK)

    def test_halt_report_reproduction_one_two_three(self):
        for k, items in enumerate((["critic"], ["critic", "obca_k1"], ["removal", "critic", "range_hinge"])):
            repro = {name: dict(passed=name not in items) for name in P.REPRO_ORDER}
            root = self.tree(dict(plumbing=repro), sub=f"r{k}")
            rep = P.halt_report(root, templates=BLOCK, mode="scratch")
            self.assertEqual(rep["stop"], "stop.reproduction")
            names = [P.REPRO_PLAIN[n] for n in items]
            self.assertEqual(rep["sentences"][0]["text"], P.reproduction_sentence(T["stop.reproduction"], names))
            self.assertIn(CT.join_items(names), rep["sentences"][0]["text"])

    def test_halt_report_with_no_stop(self):
        rep = P.halt_report(self.tree(), templates=BLOCK, mode="scratch")
        self.assertEqual(rep["sentences"], [])
        self.assertEqual(rep["note"], "No stop applies now.")


# --------------------------------------------------------------------------------------------- files and modes

class TestFilesAndModes(Base):
    def prereg(self):
        return dict(constants=P.canonical_constants(), claims_word_for_word=BLOCK)

    def test_run_look_writes_once(self):
        root = self.tree()
        lk = P.run_look("confirmatory", root, "scratch", self.prereg(), templates=BLOCK)
        self.assertTrue((root / K.LAYOUT["look"].format(group="confirmatory")).is_file())
        self.assertTrue((root / K.LAYOUT["look_md"].format(group="confirmatory")).is_file())
        body, _ = P.read_json_verified(root / K.LAYOUT["look"].format(group="confirmatory"))
        self.assertEqual(body["sentences"], lk["sentences"])
        self.assertEqual(body["claims_block"], json.loads(P.canonical(K.CLAIMS)))
        with self.assertRaises(P.LookRefused):
            P.run_look("confirmatory", root, "scratch", self.prereg(), templates=BLOCK)

    def test_run_look_refuses_other_constants_and_the_repository(self):
        root = self.tree()
        bad = self.prereg()
        bad["constants"] = dict(bad["constants"], N_FRESH=5)
        with self.assertRaises(P.LookRefused):
            P.run_look("confirmatory", root, "scratch", bad, templates=BLOCK)
        with self.assertRaises(P.LookRefused):
            P.run_look("confirmatory", K.OUT, "scratch", self.prereg(), templates=BLOCK)

    def test_the_official_look_refuses_while_the_claims_block_is_open(self):
        self.assertTrue(K.todo_paths())
        with self.assertRaises(P.LookRefused):
            P.run_look("confirmatory", K.OUT, "official", self.prereg(), templates=BLOCK)

    def test_tampered_and_mismatched_records_are_refused(self):
        root = self.tree()
        p = root / K.LAYOUT["values"].format(setting="main", world="stall", seed=SEEDS[0])
        p.write_bytes(p.read_bytes().replace(b'"points"', b'"pointz"', 1))
        with self.assertRaises(P.LookRefused):
            P.load_results(root, "confirmatory", "scratch")
        root = self.tree(dict(mode="stub"), sub="b")
        with self.assertRaises(P.LookRefused):
            P.load_results(root, "confirmatory", "scratch")

    def test_seeds_and_reserved_seeds(self):
        self.assertEqual(P.resolve_seeds("confirmatory", None, "full", "scratch"), SEEDS)
        self.assertEqual(P.resolve_seeds("exploratory", None, "smoke", "scratch"), [0])
        for s in (5, 10, 14, 15, 34, 100, 900):
            with self.assertRaises(P.LookRefused):
                P.resolve_seeds("confirmatory", [s], None, "scratch")
        with self.assertRaises(P.LookRefused):
            P.resolve_seeds("confirmatory", SEEDS[:2], None, "official")
        with self.assertRaises(P.LookRefused):
            P.load_results(self.tree(), "exploratory", "official")

    def test_nothing_written_into_the_code_folder(self):
        for p in HERE.iterdir():
            self.assertIn(p.name, K.CODE_FILES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
