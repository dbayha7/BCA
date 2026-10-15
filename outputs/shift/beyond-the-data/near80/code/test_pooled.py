"""Near-80 (version 8): tests of the looks (pooled.py) and of section 14's extraction and filling (claims_text.py).

Every test builds a synthetic result tree in the runner's record schemas (build spec, section 5) with make_tree, in a
temporary folder outside the repository, then checks the exact sentences: each expected sentence is built here from
the template text of the real PREREGISTRATION.md by plain string replacement of the expected bracket strings.
Nothing is trained, no seed's data is generated or read, nothing is written into the repository.

Run: /tmp/claude-0/bcavenv/bin/python3 code/test_pooled.py   (from /tmp, PYTHONPATH=/home/user/bca)
"""
import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
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
PREFIX = BLOCK["exploratory_prefix"]
SC = "strong_current"
OTHER_SITS = ("recorded", "big_waves", "big_kicks")


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


NEAR_STALL = dict(none=50, full_none=100, clone=40, w0_check=50, obca_k1=80, obca_k2=78, obca_forced=80,
                  fixed_gated=75, conformal_k1=70, meangap_k1=65, meandiff_k1=66, knee_ramp=79, perfect_band=90,
                  range_hinge=50, knee_limit=55, even_pull=55, push_less=54, random_1=52, random_2=51, standin=60,
                  version6_bca=60)
NO_STALL = dict(none=90, obca_k1=89, standin=70, even_pull=85, random_1=88, random_2=89, obca_forced=60, clone=80,
                w0_check=90)
RR_STALL = dict(none=52, clone=41, w0_check=52, obca_k1=82, range_hinge=60)
MAIN_STALL = dict(NEAR_STALL, obca_k1=83, range_hinge=61, none=52, full_none=102)


def default_points(setting, world, seed, arm, sit):
    if sit != SC:
        return 71.0 if arm == "obca_k1" else 70.0
    if world == "no_stall":
        return float(NO_STALL.get(arm, 85))
    table = {"near80": NEAR_STALL, "random_rows": RR_STALL, "main": MAIN_STALL}[setting]
    return float(table[arm])


def default_gain(setting, world, seed, arm, sit):
    return dict(obca_k1=8.0, none=10.0, range_hinge=0.5, version6_bca=3.0).get(arm, 1.0)


def default_over(setting, world, seed, arm, sit):
    return dict(obca_k1=2.0, none=6.0, range_hinge=0.0, version6_bca=1.0).get(arm, 0.0)


def default_past(setting, world, seed, arm, sit):
    return 0.4 if arm == "none" else 0.1


def group_of(seed):
    return "exploratory" if seed in K.EXPLORATORY else "confirmatory"


def make_tree(tmp, spec=None):
    """Writes a synthetic result tree under tmp/out and returns its root. spec (all optional):
      group               'confirmatory' (default) or 'exploratory'
      seeds               the seeds written (default: the group's)
      mode, prereg_sha256 the records' mode ('scratch') and fingerprint (None)
      points, gain, over, past   f(setting, world, seed, arm, situation) -> number (defaults above)
      settings            the settings that ran (default all three)
      missing_arms        {(setting, world, seed) | (setting, world): [logical ids with no actor]}
      missing_values      [(setting, world, seed)]: no values.json (a stopped or skipped phase)
      data_main_fail      {seed: (reason, 'leak' | 'marks')}
      data_band_fail      {seed: (world, rule name)}
      check10_fail        [seeds]
      switch_on           f(setting, world, seed) -> bool (default: on in the stalling world, off in the other)
      stops               [{phase_kind, setting, world, seed, reason}]
      q_points            f(seed) -> points (default 70)
      exploratory         dict for write_exploratory (confirmatory group only; default allowed, nothing dropped)
      plumbing            the plumbing record's 'reproduction' block (written when given)
      halts               {(seed, world): [rules]} data_band records of seeds 0-4 that halt the run"""
    spec = dict(spec or {})
    root = Path(tmp) / "out"
    group = spec.get("group", "confirmatory")
    seeds = spec.get("seeds", list(K.EXPLORATORY if group == "exploratory" else K.CONFIRMATORY))
    mode, psha = spec.get("mode", "scratch"), spec.get("prereg_sha256")
    env = dict(mode=mode, prereg_sha256=psha, written="2026-10-15T00:00:00Z")
    pts_f = spec.get("points", default_points)
    gain_f, over_f, past_f = (spec.get("gain", default_gain), spec.get("over", default_over),
                              spec.get("past", default_past))
    settings = spec.get("settings", K.SETTINGS)
    missing_arms = spec.get("missing_arms", {})
    missing_values = set(spec.get("missing_values", []))
    switch_on = spec.get("switch_on", lambda st, w, s: w == "stall")
    q_f = spec.get("q_points", lambda s: 70.0)
    locks = {}
    step2_saved = {}
    for s in seeds:
        for w in K.WORLDS:
            fail = spec.get("data_main_fail", {}).get(s)
            put(root / K.LAYOUT["data_main"].format(world=w, seed=s), dict(
                env, kind="near80-data-main", phase_id=K.phase_id("data_main", "main", w, s), seed=s, world=w,
                group=group_of(s), source="gate" if s < 5 else "fresh",
                ball_leak_checks_passed=not (fail and fail[1] == "leak"),
                gate_world_marks=None if w != "stall" else {f"T{i}": not (fail and fail[1] == "marks" and i == 2)
                                                             for i in range(1, 6)},
                passed_gate_checks=not fail, seed_failed_reason=fail[0] if fail else None))
            bfail = spec.get("data_band_fail", {}).get(s)
            rules = [dict(rule="check 2: the evaluation split keeps band rows", scope=f"{w}/seed{s}", value=800,
                          threshold=">= 300", passed=True),
                     dict(rule="check 7: the calibration bank keeps enough rows", scope=f"{w}/seed{s}", value=810,
                          threshold=">= 500", passed=True)]
            if bfail and bfail[0] == w:
                rules.append(dict(rule=bfail[1], scope=f"{w}/seed{s}", value=0.071, threshold="0.01-0.06",
                                  passed=False))
            halt = (spec.get("halts") or {}).get((s, w))
            if halt:
                rules = halt
            c10 = None
            if w == "stall":
                c10 = dict(value=18.0, passed=s not in spec.get("check10_fail", []))
                rules.append(dict(rule="check 10: the band costs a rule that reads only the data", scope=f"stall/seed{s}",
                                  value=18.0 if c10["passed"] else 4.0, threshold=">= 10.0", passed=c10["passed"]))
            first = next((r for r in rules if not r["passed"]), None)
            bad18 = bool(bfail and bfail[0] == w) or bool(halt)
            put(root / K.LAYOUT["data_band"].format(world=w, seed=s), dict(
                env, kind="near80-data-band", phase_id=K.phase_id("data_band", "band", w, s), seed=s, world=w,
                group=group_of(s), rules=rules, passed_checks_1_8=not bad18,
                first_failed_check=None if first is None else {k: first[k] for k in ("rule", "value", "threshold")},
                check10=c10, halts_run=bool(halt),
                data_edge_points=None if w != "stall" else dict(main={SC: 88.0}, near80={SC: q_f(s)},
                                                                random_rows={SC: 87.0}),
                seed_failed_reason=f"no-training check failed: {bfail[1]}" if (bfail and bfail[0] == w) else None))
        for setting in settings:
            for w in K.WORLDS:
                if setting in ("near80", "random_rows") or s in K.CONFIRMATORY:
                    put(root / K.LAYOUT["switch"].format(setting=setting, world=w, seed=s), dict(
                        env, kind="near80-switch", phase_id=K.phase_id("switch", setting, w, s), setting=setting,
                        world=w, seed=s, t_lo=0.7, t_up=0.95, defined=True, n_survivors=12,
                        certificate=dict(on=bool(switch_on(setting, w, s)), height=2.5, q05=0.4),
                        groups=dict(n_full_throttle=40, n_rest=700), rank_test=dict(p=0.01)))
                if (setting, w, s) in missing_values:
                    continue
                gone = set(missing_arms.get((setting, w, s), [])) | set(missing_arms.get((setting, w), []))
                ids = {}
                for a in K.ARMS[setting]:
                    ids[a] = None if (a in gone or (w != "stall" and a in K.STALL_ONLY_ARMS)) else a
                saved = setting == "main" and s in K.EXPLORATORY
                if saved:
                    lock_sha = _sha(f"step2 lock {w} {s}".encode())
                    step2_saved[f"{w}/seed{s}"] = dict(lock_sha256=lock_sha)
                else:
                    lock_sha = put(root / K.LAYOUT["policy_lock"].format(setting=setting, world=w, seed=s), dict(
                        env, kind="near80-policy-lock", phase_id=K.phase_id("policy", setting, w, s),
                        setting=setting, world=w, seed=s, group=group_of(s), arm_ids=ids,
                        strengths={n: 1.0 for n in ids.values() if n}, matched_names={}, standin=dict(arm="standin"),
                        arm_table=dict(t_lo=0.7, t_up=0.95, arms={"obca_k1": dict(switch=True, h=2.5, k=1.0)})))
                    locks[K.phase_id("policy", setting, w, s)] = dict(lock_sha256=lock_sha)
                sits = {}
                for sit in K.SITUATIONS:
                    t = {m: {} for m in P.MEASURES}
                    for a, name in ids.items():
                        if name is None:
                            continue
                        p, g, o = pts_f(setting, w, s, a, sit), gain_f(setting, w, s, a, sit), over_f(setting, w, s,
                                                                                                         a, sit)
                        t["points"][name] = p
                        t["held70_points"][name] = None if p is None or g is None else p - g
                        t["capped80_points"][name] = None if p is None or o is None else p + o
                        t["unseen_gain"][name] = g
                        t["overshoot_loss"][name] = o
                        t["past80_share"][name] = past_f(setting, w, s, a, sit)
                        t["zone_shares"][name] = [0.5, 0.3, 0.2]
                        t["lower_tail_points"][name] = None if p is None else p - 20
                    sits[sit] = t
                put(root / K.LAYOUT["values"].format(setting=setting, world=w, seed=s), dict(
                    env, kind="near80-values", phase_id=K.phase_id("evaluate", setting, w, s), setting=setting,
                    world=w, seed=s, group=group_of(s), source="step2_saved" if saved else "trained",
                    lock_sha256=lock_sha, arm_ids=ids, situations=sits, evaluations={}))
    for st in spec.get("stops", []):
        pid = K.phase_id(st["phase_kind"], st.get("setting", "x"), st.get("world", "x"), st.get("seed"),
                         st.get("group"))
        put(root / K.LAYOUT["phase_stop"].format(phase_id=pid), dict(
            env, kind="near80-phase-stop", phase_id=pid, phase_kind=st["phase_kind"], setting=st.get("setting", "x"),
            world=st.get("world", "x"), seed=st.get("seed"),
            group=st.get("group") or (group_of(st["seed"]) if st.get("seed") is not None else group),
            reason=st["reason"]))
    put(root / K.LAYOUT["stage_lock"].format(group=group), dict(
        env, kind="near80-stage-lock", phase_id=K.phase_id("lock", group=group), group=group, locks=locks,
        plan=spec.get("plan", "full"), step2_saved=step2_saved or None))
    if group == "exploratory":
        put(root / K.LAYOUT["fixed"], dict(env, kind="near80-fixed-checks", phase_id=K.phase_id(
            "fixed_checks", group="exploratory"), rules=[], passed=True, halts_run=False))
    if group == "confirmatory" and spec.get("exploratory", {}) is not None:
        write_exploratory(root, mode=mode, prereg_sha256=psha, **spec.get("exploratory", {}))
    if spec.get("plumbing") is not None:
        put(root / K.LAYOUT["plumbing"], dict(kind="near80-plumbing", passed=False, prereg_sha256=psha,
                                              reproduction=spec["plumbing"], written=env["written"]))
    return root


def write_exploratory(root, mode="scratch", prereg_sha256=None, allowed=True, dropped=False, V=1.0,
                      arm="the hand-set range limit", H=50.0, Pshare=0.4, B=10.0):
    harm_ok, share_ok = H is not None and H >= 30, Pshare is not None and Pshare >= 0.25
    return put(Path(root) / K.LAYOUT["look"].format(group="exploratory"), dict(
        kind="near80-look-exploratory", mode=mode, prereg_sha256=prereg_sha256, group="exploratory",
        phase_id=K.phase_id("look_exploratory", group="exploratory"), fresh_near80_allowed=allowed,
        claim3_dropped=dropped, brackets=dict(V=V, arm=arm),
        stops=dict(harm=dict(H=H, P=Pshare, harm_passed=harm_ok, share_passed=share_ok, passed=harm_ok and share_ok),
                   rival_unaffected=dict(B=B, passed=B is not None and B >= 5),
                   noise_floor=dict(run=True, passed=not dropped, V=V, arm=arm)),
        written="2026-10-15T00:00:00Z"))


def override(base, rules):
    """A points function equal to `base` except where a rule (setting, world, arm, situation, seed or None, value)
    applies (value may be a per-seed dict)."""
    def f(setting, world, seed, arm, sit):
        for st, w, a, si, sd, v in rules:
            if st == setting and w == world and a == arm and si == sit and (sd is None or sd == seed):
                return v[seed] if isinstance(v, dict) else v
        return base(setting, world, seed, arm, sit)
    return f


def expect(sid, **br):
    s = T[sid]
    for k, v in br.items():
        s = s.replace(f"[{k}]", v)
    assert "[" not in s, s
    return s


def explo(text):
    return f"{PREFIX} {text}"


def explo_t(sid):
    text = T[sid]
    if "seeds 0-4" not in text:
        text = text.replace("fresh seeds 10-14", "seeds 0-4").replace("fresh seeds", "seeds")
    return text


PASS1 = dict(S="5", N="5", X="30.0", H="50.0", W="1.0", NET="29.0", D="29.0", R="the matched even pull", E="25.0")
SCALE = dict(G0="10.0", O0="6.0", GZ="0.5", OZ="0.0", GV="3.0", OV="1.0", Q="70.0")
HOLDS3 = dict(A="2.0", B="10.0", C="8.0", K="5")
HOLDS4 = dict(NET="30.0", D="32.0", R="the matched even pull")
BOTH = dict(dB="-3.0", dR="-11.0", d0="-2.0")
LAST_PASSED = T["claim1.last.claim4_passed"]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="near80_test_pooled_"))
        self.assertFalse(str(self.tmp).startswith("/home/user/bca"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tree(self, spec=None, sub="a"):
        return make_tree(self.tmp / sub, spec)

    def look(self, spec=None, sub="a", defaults=None):
        spec = dict(spec or {})
        root = self.tree(spec, sub)
        group = spec.get("group", "confirmatory")
        R = P.load_results(root, group, "scratch", defaults=defaults)
        if group == "exploratory":
            return P.exploratory_look(R, BLOCK)
        return P.confirmatory_look(R, R.exploratory, BLOCK)

    def texts(self, look):
        return [r["text"] for r in look["sentences"]]

    def ids(self, look):
        return [r["id"] for r in look["sentences"]]


# --------------------------------------------------------------------------------------------- extraction

class TestExtraction(Base):
    def test_every_id_once_and_in_order(self):
        self.assertEqual(list(BLOCK["sentences"]), list(CT.SENTENCE_IDS))
        self.assertEqual(len(CT.SENTENCE_IDS), len(set(CT.SENTENCE_IDS)))
        self.assertEqual(BLOCK["source"], "PREREGISTRATION.md section 14")
        self.assertTrue(BLOCK["intro"].startswith("The code fills in the brackets."))
        self.assertEqual(BLOCK["exploratory_prefix"], "Exploratory, seeds 0-4, designed after seeing them; not a claim:")
        self.assertTrue(BLOCK["always_attached"].startswith("One simulated task, two worlds, five fresh seeds"))

    def test_templates_have_no_quote_marks_or_blockquote_markers(self):
        for sid, text in T.items():
            self.assertFalse(text.startswith('"') or text.endswith('"'), sid)
            self.assertNotIn("> ", text, sid)
            self.assertNotIn("  ", text, sid)

    def test_brackets_all_defined(self):
        names = set(CT.table_bracket_names(BLOCK))
        self.assertEqual(names, {"S", "N", "X", "H", "W", "NET", "D", "R", "E", "L", "G", "O", "G0", "O0", "GZ",
                                 "OZ", "GV", "OV", "Q", "A", "B", "C", "K", "M", "P", "V", "dB", "dR", "d0"})
        for sid, text in T.items():
            for b in re.findall(r"\[([A-Z][A-Za-z0-9]*)\]", text):
                self.assertIn(b, names, sid)

    def test_filling_every_template_leaves_no_bracket(self):
        values = {n: 1.25 for n in CT.table_bracket_names(BLOCK)}
        values.update(s="12", check="check 3", reason="why", arm="the hand-set range limit", value="0.07",
                      threshold="at least 1", world="stalling world", phase="the critic", situation="big waves",
                      what="the harm", seeds="2")
        specials = {"[one clause per missed mark]": "a clause", "[last sentence]": "Last.",
                    "[clauses as in claim 1]": "a clause"}
        for sid, text in T.items():
            if sid == "stop.harm":
                out = P.harm_stop_sentence(text, 10.0, 0.1)
            elif sid == "stop.reproduction":
                out = P.reproduction_sentence(text, ["A", "B"])
            else:
                out = CT.fill(text, values, specials)
            self.assertNotIn("[", out, sid)
            self.assertNotIn("]", out, sid)
        self.assertNotIn("[", CT.fill(BLOCK["always_attached"], {}))

    def test_fill_rules_are_the_14_6_bullets(self):
        sec = DOC[DOC.index("### 14.6"):]
        sec = sec[:sec.index("\n---")]
        bullets = [" ".join(x.split()) for x in re.split(r"\n- ", "\n" + sec.split("\n", 2)[2].strip())
                   if x.strip()]
        self.assertEqual(BLOCK["fill_rules"], bullets)
        self.assertEqual(len(BLOCK["fill_rules"]), 15)

    def test_refuses_a_missing_or_doubled_sentence_and_an_undefined_bracket(self):
        with self.assertRaises(ValueError):
            CT.extract_section14(DOC.replace("**Any mark fails:**", "**Some mark fails:**"))
        sec = DOC.index("### 14.4")
        doubled = DOC[:sec] + '**Holds:**\n> "again."\n\n' + DOC[sec:]
        with self.assertRaises(ValueError):
            CT.extract_section14(doubled)
        with self.assertRaises(ValueError):
            CT.extract_section14(DOC.replace("would have added only [O].", "would have added only [OO]."))
        with self.assertRaises(ValueError):
            CT.extract_section14(DOC.replace("### 14.6 How the code fills in the sentences", "### 14.7 Other"))

    def test_fill_refuses_an_unfilled_bracket(self):
        with self.assertRaises(ValueError):
            CT.fill("a [G] and [O]", dict(G=1.0))
        with self.assertRaises(ValueError):
            CT.fill("a [one clause] b", {})
        self.assertEqual(CT.fill("[P]% and [P]", dict(P=None)), "not available and not available")
        self.assertEqual(CT.fill("x [V] y", dict(V="[0.695, 0.70]")), "x [0.695, 0.70] y")

    def test_rounding_and_writing(self):
        f = CT.fmt
        self.assertEqual(f(-0.04), "0.0")
        self.assertEqual(f(2.25), "2.2")
        self.assertEqual(f(None), "not available")
        self.assertEqual(f(float("nan")), "not available")
        self.assertEqual(f(-3.0), "-3.0")
        self.assertEqual(f(12.345), "12.3")
        self.assertEqual(f(0.2504, "percent"), "25.0")
        self.assertEqual(f(3, "count"), "3")
        self.assertEqual(f(0.69123, "value"), "0.6912")
        self.assertEqual(f(1072, "value"), "1072")
        self.assertEqual(CT.join_items(["A"]), "A")
        self.assertEqual(CT.join_items(["A", "B"]), "A and B")
        self.assertEqual(CT.join_items(["A", "B", "C"]), "A, B and C")


# --------------------------------------------------------------------------------------------- claim 1

class TestClaim1(Base):
    def test_all_pass_confirmatory(self):
        lk = self.look()
        self.assertEqual(self.ids(lk), ["always_attached", "claim1.pass", "claim2.pass_pass", "claim2.scale",
                                        "claim3.holds", "claim4.holds", "claim4.both_settings"])
        self.assertEqual(self.texts(lk), [BLOCK["always_attached"], expect("claim1.pass", **PASS1),
                                          expect("claim2.pass_pass", G="8.0", O="2.0"), expect("claim2.scale", **SCALE),
                                          expect("claim3.holds", **HOLDS3), expect("claim4.holds", **HOLDS4),
                                          expect("claim4.both_settings", **BOTH)])
        self.assertEqual({c: v["status"] for c, v in lk["claims"].items()},
                         dict(claim1="pass", claim2="pass", claim3="pass", claim4="pass"))
        self.assertEqual(lk["claims"]["claim1"]["sentence_ids"], ["claim1.pass"])
        self.assertEqual(lk["claims"]["claim2"]["sentence_ids"], ["claim2.pass_pass", "claim2.scale"])

    def test_net_uses_step2s_own_functions(self):
        root = self.tree()
        R = P.load_results(root, "confirmatory", "scratch")
        S = P.step2()
        self.assertEqual(P.brackets(R)["NET"], S.net(R.pts["near80"], "obca_k1", R.seeds)["net"])
        self.assertEqual(S.net(R.pts["near80"], "obca_k1", R.seeds)["net"], 29.0)

    def fail_texts(self, clauses, last=LAST_PASSED):
        return T["claim1.fail"].replace("[one clause per missed mark]", "; ".join(clauses)).replace(
            "[last sentence]", last)

    def test_fail_net_alone(self):
        pts = override(default_points, [("near80", "no_stall", "obca_k1", SC, None, 69.0)])
        lk = self.look(dict(points=pts))
        net = expect("claim1.clause.net", NET="9.0", D="9.0", R="the matched even pull")
        self.assertEqual(lk["sentences"][1], dict(id="claim1.fail", text=self.fail_texts([net])))
        self.assertEqual(lk["claims"]["claim1"]["status"], "fail")

    def test_fail_keeps_each_situation_alone(self):
        for sit, plain in (("recorded", "recorded conditions"), ("big_waves", "big waves"),
                           ("big_kicks", "big kicks")):
            pts = override(default_points, [("near80", "stall", "obca_k1", sit, None, 66.0)])
            lk = self.look(dict(points=pts), sub=sit)
            clause = expect("claim1.clause.keeps", L="4.0", situation=plain)
            self.assertEqual(lk["sentences"][1]["text"], self.fail_texts([clause]), sit)

    def test_fail_even_pull_alone(self):
        pts = override(default_points, [("near80", "stall", "even_pull", SC, None, 70.0)])
        lk = self.look(dict(points=pts))
        self.assertEqual(lk["sentences"][1]["text"],
                         self.fail_texts([expect("claim1.clause.even_pull", E="10.0")]))

    def test_fail_a_failed_seed_alone(self):
        spec = dict(missing_values=[("random_rows", "stall", 12), ("random_rows", "no_stall", 12)],
                    stops=[dict(phase_kind="policy", setting="random_rows", world="stall", seed=12,
                                reason="a learner that fails (a non-finite network)")])
        lk = self.look(spec)
        clause = expect("claim1.clause.failed_seed", s="12",
                        check="its actor training phase (random-rows setting, stalling world)")
        self.assertEqual(lk["sentences"][1], dict(id="claim1.fail", text=self.fail_texts([clause])))
        self.assertEqual(lk["failed_seeds"]["claims_1_3"]["12"],
                         "the actor training stopped (a learner that fails (a non-finite network))")
        self.assertEqual(lk["failed_seeds"]["claim_4"], {})
        self.assertEqual(lk["claims"]["claim4"]["status"], "pass")
        self.assertEqual(lk["sentences"][-1], dict(id="stop.phase", text=expect(
            "stop.phase", s="12", world="stalling world", phase="the actor training (random-rows setting)",
            reason="a learner that fails (a non-finite network)")))
        self.assertIn(dict(id="claim2.fail_fail", text=expect("claim2.fail_fail", G="not available",
                                                              O="not available")), lk["sentences"])

    def test_several_clauses_in_order(self):
        pts = override(default_points, [("near80", "no_stall", "obca_k1", SC, None, 69.0),
                                        ("near80", "stall", "obca_k1", "big_kicks", None, 60.0),
                                        ("near80", "stall", "obca_k1", "recorded", None, 65.5),
                                        ("near80", "stall", "even_pull", SC, None, 70.0)])
        lk = self.look(dict(points=pts))
        cl = [expect("claim1.clause.net", NET="9.0", D="-6.0", R="the matched even pull"),
              expect("claim1.clause.keeps", L="4.5", situation="recorded conditions"),
              expect("claim1.clause.keeps", L="10.0", situation="big kicks"),
              expect("claim1.clause.even_pull", E="10.0")]
        self.assertEqual(lk["sentences"][1]["text"], self.fail_texts(cl))

    def test_missing_arm_writes_not_available(self):
        lk = self.look(dict(missing_arms={("near80", "stall", 11): ["even_pull"]}))
        cl = [expect("claim1.clause.net", NET="29.0", D="not available", R="not available"),
              expect("claim1.clause.even_pull", E="not available")]
        self.assertEqual(lk["sentences"][1]["text"], self.fail_texts(cl))

    def test_each_last_sentence(self):
        pts = override(default_points, [("near80", "no_stall", "obca_k1", SC, None, 69.0),
                                        ("main", "stall", "even_pull", SC, None, 75.0)])
        lk = self.look(dict(points=pts), sub="failed")
        net = expect("claim1.clause.net", NET="9.0", D="9.0", R="the matched even pull")
        self.assertEqual(lk["sentences"][1]["text"],
                         self.fail_texts([net], T["claim1.last.claim4_failed"]))
        pts = override(default_points, [("near80", "no_stall", "obca_k1", SC, None, 69.0)])
        lk = self.look(dict(points=pts, settings=("near80", "random_rows")), sub="notrun",
                       defaults=dict(main_on_fresh_seeds=False))
        self.assertEqual(lk["sentences"][1]["text"],
                         self.fail_texts([net], T["claim1.last.claim4_not_run"]))
        self.assertEqual(lk["claims"]["claim4"]["status"], "not run")
        self.assertNotIn("claim4.both_settings", self.ids(lk))

    def test_invalid_by_harm_share_and_both(self):
        harm = expect("claim1.reason.harm", H="20.0")
        share = expect("claim1.reason.past80", P="20.0")
        cases = (("h", dict(points=override(default_points, [("near80", "stall", "full_none", SC, None, 70.0)])),
                  harm),
                 ("p", dict(past=lambda st, w, s, a, si: 0.2 if a == "none" else 0.1), share),
                 ("hp", dict(points=override(default_points, [("near80", "stall", "full_none", SC, None, 70.0)]),
                             past=lambda st, w, s, a, si: 0.2 if a == "none" else 0.1), harm + "; " + share))
        for sub, spec, reason in cases:
            lk = self.look(spec, sub=sub)
            self.assertEqual(lk["sentences"][1], dict(id="claim1.invalid", text=expect("claim1.invalid",
                                                                                       reason=reason)), sub)
            self.assertEqual(lk["claims"]["claim1"]["status"], "invalid")
            self.assertEqual(lk["claims"]["claim2"]["sentence_ids"], ["claim2.scale"])
            self.assertEqual(lk["claims"]["claim3"]["sentence_ids"], [])


# --------------------------------------------------------------------------------------------- claims 2 and 3

class TestClaims2And3(Base):
    def test_four_claim2_sentences(self):
        cases = (("pass_pass", 8.0, 2.0, "8.0", "2.0"), ("fail_pass", 3.0, 2.0, "3.0", "2.0"),
                 ("pass_fail", 8.0, 7.0, "8.0", "7.0"), ("fail_fail", 3.0, 7.0, "3.0", "7.0"))
        for key, g, o, gs, os_ in cases:
            lk = self.look(dict(gain=lambda st, w, s, a, si, g=g: g if a == "obca_k1" else default_gain(st, w, s, a,
                                                                                                         si),
                                over=lambda st, w, s, a, si, o=o: o if a == "obca_k1" else default_over(st, w, s, a,
                                                                                                         si)),
                           sub=key)
            self.assertEqual(lk["sentences"][2], dict(id=f"claim2.{key}", text=expect(f"claim2.{key}", G=gs, O=os_)))

    def test_gain_needs_four_positive_seeds(self):
        per = {10: 10.0, 11: 10.0, 12: 10.0, 13: 0.0, 14: 0.0}
        lk = self.look(dict(gain=lambda st, w, s, a, si: per[s] if a == "obca_k1" else default_gain(st, w, s, a, si)))
        self.assertEqual(lk["marks"]["claim2"]["gain"]["positive"], 3)
        self.assertEqual(lk["sentences"][2]["id"], "claim2.fail_pass")
        self.assertEqual(lk["sentences"][2]["text"], expect("claim2.fail_pass", G="6.0"))

    def test_check10_in_two_seeds(self):
        lk = self.look(dict(check10_fail=[10, 13]))
        self.assertEqual(self.ids(lk)[2:5], ["claim2.check10", "claim2.scale", "claim4.holds"])
        self.assertEqual(lk["sentences"][2]["text"], expect("claim2.check10", M="2"))
        self.assertEqual(lk["claims"]["claim3"]["status"], "invalid")
        lk = self.look(dict(check10_fail=[10]), sub="one")
        self.assertIn("claim2.pass_pass", self.ids(lk))
        self.assertIn("claim3.holds", self.ids(lk))

    def test_scale_with_version6_missing(self):
        lk = self.look(dict(missing_arms={("near80", "stall", 12): ["version6_bca"]}))
        self.assertIn(dict(id="claim2.scale", text=expect("claim2.scale", **dict(SCALE, GV="not available",
                                                                                 OV="not available"))),
                      lk["sentences"])

    def test_claim3_holds_fails_invalid_dropped(self):
        lk = self.look(sub="holds")
        self.assertIn(expect("claim3.holds", **HOLDS3), self.texts(lk))
        pts = override(default_points, [("random_rows", "stall", "range_hinge", SC, None, 56.0)])
        lk = self.look(dict(points=pts), sub="c")
        self.assertIn(dict(id="claim3.fails", text=expect("claim3.fails", A="2.0", B="6.0", C="4.0", K="5")),
                      lk["sentences"])
        per = {10: 70.0, 11: 70.0, 12: 70.0, 13: 50.0, 14: 50.0}
        pts = override(default_points, [("random_rows", "stall", "range_hinge", SC, None, per)])
        lk = self.look(dict(points=pts), sub="k")
        self.assertIn(dict(id="claim3.fails", text=expect("claim3.fails", A="2.0", B="12.0", C="10.0", K="3")),
                      lk["sentences"])
        pts = override(default_points, [("random_rows", "stall", "range_hinge", SC, None, 53.0)])
        lk = self.look(dict(points=pts), sub="inv")
        self.assertIn(dict(id="claim3.invalid", text=expect("claim3.invalid", B="3.0")), lk["sentences"])
        self.assertEqual(lk["claims"]["claim3"]["status"], "invalid")
        lk = self.look(dict(exploratory=dict(dropped=True, V=6.5, arm="BCA's switched ramp")), sub="drop")
        self.assertIn(dict(id="claim3.dropped", text=expect("claim3.dropped", arm="BCA's switched ramp", V="6.5")),
                      lk["sentences"])
        self.assertEqual(lk["claims"]["claim3"]["status"], "dropped")


# --------------------------------------------------------------------------------------------- claim 4

class TestClaim4(Base):
    def test_holds_fails_and_both_settings(self):
        lk = self.look(sub="holds")
        self.assertIn(expect("claim4.holds", **HOLDS4), self.texts(lk))
        self.assertEqual(self.ids(lk)[-1], "claim4.both_settings")
        pts = override(default_points, [("main", "stall", "obca_k1", "big_waves", None, 65.0)])
        lk = self.look(dict(points=pts), sub="fails")
        self.assertIn(dict(id="claim4.fails", text=T["claim4.fails"].replace(
            "[clauses as in claim 1]", expect("claim1.clause.keeps", L="5.0", situation="big waves"))),
            lk["sentences"])
        self.assertEqual(lk["sentences"][1]["text"], expect("claim1.pass", **PASS1))

    def test_both_settings_only_when_both_ran(self):
        lk = self.look(dict(settings=("main",), exploratory=dict(allowed=False, H=20.0)))
        self.assertEqual(self.ids(lk), ["always_attached", "claim4.holds", "stop.harm", "stop.claim4_note"])
        self.assertEqual({c: v["status"] for c, v in lk["claims"].items()},
                         dict(claim1="not run", claim2="not run", claim3="not run", claim4="pass"))

    def test_a_band_failure_does_not_fail_claim4_a_main_failure_fails_both(self):
        spec = dict(data_band_fail={13: ("stall", "check 3: the band is the planned size")},
                    missing_values=[(st, w, 13) for st in ("near80", "random_rows") for w in K.WORLDS])
        lk = self.look(spec, sub="band")
        self.assertEqual(lk["failed_seeds"]["claim_4"], {})
        self.assertIn("13", lk["failed_seeds"]["claims_1_3"])
        self.assertEqual(lk["failed_checks"]["claims_1_3"]["13"], "check 3, the extra rows as a share of the ball-kept "
                                                                  "rows")
        self.assertEqual(lk["claims"]["claim4"]["status"], "pass")
        # the failed seed is told by its own clause (section 14.6); it never makes claim 1's validity fail
        self.assertEqual(lk["claims"]["claim1"]["status"], "fail")
        clause = expect("claim1.clause.failed_seed", s="13", check="check 3, the extra rows as a share of the "
                                                                   "ball-kept rows")
        self.assertEqual(clause, "seed 13 failed check 3, the extra rows as a share of the ball-kept rows and, by the "
                                 "pre-set rule, counts as failing every mark")
        self.assertEqual(lk["sentences"][1], dict(id="claim1.fail", text=T["claim1.fail"].replace(
            "[one clause per missed mark]", clause).replace("[last sentence]", LAST_PASSED)))
        self.assertEqual(lk["validity"]["claims_1_3"]["seeds"], [10, 11, 12, 14])
        self.assertEqual(lk["validity"]["claims_1_3"]["harm"]["value"], 50.0)
        spec = dict(data_main_fail={13: ("the gate's data checks failed on this seed", "marks")},
                    missing_values=[(st, w, 13) for st in K.SETTINGS for w in K.WORLDS])
        lk = self.look(spec, sub="main")
        self.assertIn("13", lk["failed_seeds"]["claim_4"])
        self.assertIn("13", lk["failed_seeds"]["claims_1_3"])
        clause = expect("claim1.clause.failed_seed", s="13", check="the gate's five world checks on the ball alone")
        self.assertIn(dict(id="claim4.fails", text=T["claim4.fails"].replace("[clauses as in claim 1]", clause)),
                      lk["sentences"])


    def test_a_near80_stalling_world_policy_stop_fails_the_seed_not_the_validity(self):
        """A stopped near-80 policy phase in the stalling world (its no-stall phase and evaluations are skipped): the
        seed fails every mark of claims 1-3 and claim 1 names it, as in the no-stall world; claim 4 is untouched."""
        why = "a learner that fails (a non-finite network)"
        for world in K.WORLDS:
            missing = [("near80", w, 12) for w in K.WORLDS] if world == "stall" else [("near80", "no_stall", 12)]
            lk = self.look(dict(missing_values=missing, stops=[dict(phase_kind="policy", setting="near80",
                                                                    world=world, seed=12, reason=why)]), sub=world)
            clause = expect("claim1.clause.failed_seed", s="12", check="its actor training phase (near-80 setting, "
                                                                       f"{K.WORLD_PLAIN[world]})")
            self.assertEqual(lk["claims"]["claim1"]["status"], "fail", world)
            self.assertEqual(lk["sentences"][1], dict(id="claim1.fail", text=T["claim1.fail"].replace(
                "[one clause per missed mark]", clause).replace("[last sentence]", LAST_PASSED)), world)
            self.assertEqual(lk["claims"]["claim4"]["status"], "pass")
            self.assertNotIn("claim1.invalid", self.ids(lk))

    def test_two_gate_failed_seeds_are_not_check10_failures(self):
        """Seeds whose main data failed have no near-80 data and no check-10 record: M leaves them out, so the
        check-10 sentence is not written and claim 1 names both seeds."""
        spec = dict(data_main_fail={11: ("the gate's data checks failed on this seed", "marks"),
                                    13: ("the gate's data checks failed on this seed", "leak")},
                    missing_values=[(st, w, s) for st in K.SETTINGS for w in K.WORLDS for s in (11, 13)])
        root = self.tree(spec)
        for s in (11, 13):
            for w in K.WORLDS:
                p = root / K.LAYOUT["data_band"].format(world=w, seed=s)
                p.unlink()
                p.with_name(p.name + ".sha256").unlink()
        R = P.load_results(root, "confirmatory", "scratch")
        lk = P.confirmatory_look(R, R.exploratory, BLOCK)
        self.assertEqual(lk["validity"]["claims_2_3"]["check10"]["M"], 0)
        self.assertTrue(lk["validity"]["claims_2_3"]["passed"])
        self.assertNotIn("claim2.check10", self.ids(lk))
        clauses = [expect("claim1.clause.failed_seed", s="11", check="the gate's five world checks on the ball alone"),
                   expect("claim1.clause.failed_seed", s="13", check="the ball's leak checks")]
        self.assertEqual(lk["sentences"][1], dict(id="claim1.fail", text=T["claim1.fail"].replace(
            "[one clause per missed mark]", "; ".join(clauses)).replace("[last sentence]",
                                                                         T["claim1.last.claim4_failed"])))
        # one real check-10 failure and one gate-failed seed: M is 1, the condition holds
        spec = dict(spec, check10_fail=[12])
        root = self.tree(spec, sub="b")
        for w in K.WORLDS:
            p = root / K.LAYOUT["data_band"].format(world=w, seed=11)
            p.unlink()
            p.with_name(p.name + ".sha256").unlink()
        R = P.load_results(root, "confirmatory", "scratch")
        v = P.validity(R)
        self.assertEqual(v["claims_2_3"]["check10"]["M"], 1)
        self.assertTrue(v["claims_2_3"]["passed"])

    def test_a_stopped_lock_fails_every_seed_of_its_group(self):
        missing = [(st, w, s) for st in K.SETTINGS for w in K.WORLDS for s in K.CONFIRMATORY]
        lk = self.look(dict(missing_values=missing, stops=[dict(phase_kind="lock", group="confirmatory",
                                                                reason="a saved input whose fingerprint does not "
                                                                       "match")]))
        clause = expect("claim1.clause.failed_seed", s="10", check="its lock phase")
        self.assertEqual(lk["claims"]["claim1"]["status"], "fail")
        self.assertTrue(lk["sentences"][1]["text"].startswith(T["claim1.fail"].split("[")[0] + clause))
        self.assertEqual(lk["claims"]["claim4"]["status"], "fail")
        self.assertNotIn("claim1.invalid", self.ids(lk))

    def test_every_seed_failed(self):
        missing = [(st, w, s) for st in ("near80", "random_rows") for w in K.WORLDS for s in K.CONFIRMATORY]
        lk = self.look(dict(missing_values=missing, data_band_fail={s: ("stall", "check 3: the band is the planned "
                                                                             "size") for s in K.CONFIRMATORY}))
        self.assertEqual(lk["claims"]["claim1"]["status"], "fail")
        self.assertEqual(lk["sentences"][1]["text"].count("by the pre-set rule, counts as failing every mark"), 5)


# --------------------------------------------------------------------------------------------- stops

class TestStops(Base):
    def test_exploratory_all_pass_prefix_and_substitutions(self):
        lk = self.look(dict(group="exploratory"))
        self.assertTrue(lk["fresh_near80_allowed"])
        self.assertFalse(lk["claim3_dropped"])
        self.assertNotIn("always_attached", self.ids(lk))
        pass1 = explo_t("claim1.pass")
        for k, v in dict(PASS1, NET="29.0").items():
            pass1 = pass1.replace(f"[{k}]", v)
        self.assertTrue(pass1.startswith("On seeds 0-4, with every recorded command"))
        self.assertEqual(lk["sentences"][0]["text"], explo(pass1))
        both = explo_t("claim4.both_settings")
        self.assertIn("against the main setting on the same seeds:", both)
        for k, v in BOTH.items():
            both = both.replace(f"[{k}]", v)
        self.assertEqual(lk["sentences"][-1]["text"], explo(both))
        for r in lk["sentences"]:
            self.assertTrue(r["text"].startswith(PREFIX), r["id"])
            self.assertNotIn("fresh seeds", r["text"].replace(PREFIX, ""))
        for c, v in lk["claims"].items():
            self.assertEqual(v["status"], "exploratory, not a claim")
            self.assertEqual(v["claim"], "")
        self.assertEqual(lk["claims"]["claim1"]["exploratory_outcome"], "pass")
        self.assertEqual(lk["kind"], "near80-look-exploratory")

    def harm_case(self, spec, sub):
        lk = self.look(dict(spec, group="exploratory"), sub=sub)
        self.assertFalse(lk["fresh_near80_allowed"])
        self.assertEqual(self.ids(lk)[-2:], ["stop.harm", "stop.claim4_note"])
        self.assertEqual(lk["sentences"][-1]["text"], T["stop.claim4_note"])
        return lk["sentences"][-2]["text"]

    def test_harm_by_h_by_p_and_both(self):
        h_pts = override(default_points, [("near80", "stall", "full_none", SC, None, 70.0)])
        low = (lambda st, w, s, a, si: 0.2 if a == "none" else 0.1)
        t = T["stop.harm"]
        i = t.index(" [or: ")
        j = t.index("]. There is")
        first_head, second = t[:i], t[i + len(" [or: "):j]
        tail = t[j + 1:]
        by_h = (first_head + tail).replace("[H]", "20.0")
        start = t.index("lost only [H]")
        by_p = (t[:start] + second + tail).replace("[P]", "20.0")
        both = (first_head + " and " + second + tail).replace("[H]", "20.0").replace("[P]", "20.0")
        self.assertEqual(self.harm_case(dict(points=h_pts), "h"), by_h)
        self.assertEqual(self.harm_case(dict(past=low), "p"), by_p)
        self.assertEqual(self.harm_case(dict(points=h_pts, past=low), "hp"), both)
        self.assertTrue(by_h.endswith("(at least 30 needed). There is too little harm to test. Nothing on seeds 10-14 "
                                      "was generated for the near-80 and random-rows settings."))
        self.assertIn("the controller trained without caution went past 80% on only 20.0% of steps (at least 25% "
                      "needed). There", by_p)

    def test_a_stop_rule_that_cannot_be_read(self):
        """A near-80 stalling-world policy stop on seed 2 leaves the harm and the band's cost unmeasured on seed 2:
        the near-80 settings stop before the fresh seeds with section 14.5's own sentence, never with a "too little"
        sentence about a number that was never measured."""
        why = "a learner that fails (a non-finite network)"
        spec = dict(group="exploratory", missing_values=[("near80", w, 2) for w in K.WORLDS],
                    stops=[dict(phase_kind="policy", setting="near80", world="stall", seed=2, reason=why)])
        lk = self.look(spec)
        self.assertFalse(lk["fresh_near80_allowed"])
        self.assertEqual(lk["stops"]["unmeasured"]["seeds"], [2])
        text = expect("stop.unmeasured", what="the harm and the band's cost to the hand-set range limit", seeds="2")
        self.assertTrue(text.startswith("Stopped before the fresh seeds: on exploratory seeds 0-4, the harm and the "
                                        "band's cost to the hand-set range limit could not be measured on seed 2,"))
        self.assertEqual(self.ids(lk)[-3:], ["stop.unmeasured", "stop.claim4_note", "stop.phase"])
        self.assertEqual(lk["sentences"][-3]["text"], text)
        self.assertNotIn("stop.harm", self.ids(lk))
        self.assertEqual(lk["claims"]["claim1"]["exploratory_outcome"], "fail")
        root = self.tree(spec, sub="report")
        P.run_look("exploratory", root, "scratch", dict(constants=P.canonical_constants()))
        rep = P.halt_report(root, templates=BLOCK, mode="scratch")
        self.assertEqual(rep["stop"], "stop.unmeasured")
        self.assertEqual([r["id"] for r in rep["sentences"]], ["stop.unmeasured", "stop.claim4_note", "stop.phase"])
        self.assertEqual(rep["sentences"][0]["text"], text)
        # the harm alone (the random-rows control intact): only "the harm" is named
        pts = override(default_points, [("near80", "stall", "full_none", SC, 2, None)])
        lk = self.look(dict(group="exploratory", points=pts), sub="h")
        self.assertEqual(lk["sentences"][-2]["text"], expect("stop.unmeasured", what="the harm", seeds="2"))

    def test_rival_unaffected(self):
        pts = override(default_points, [("random_rows", "stall", "range_hinge", SC, None, 53.0)])
        lk = self.look(dict(group="exploratory", points=pts))
        self.assertFalse(lk["fresh_near80_allowed"])
        self.assertEqual(lk["sentences"][-2:], [dict(id="stop.rival", text=expect("stop.rival", B="3.0")),
                                                dict(id="stop.claim4_note", text=T["stop.claim4_note"])])
        self.assertEqual(lk["stops"]["rival_unaffected"]["B"], 3.0)

    def test_noise_floor_drops_claim3(self):
        pts = override(default_points, [("main", "stall", "obca_k1", SC, None, 75.0)])
        lk = self.look(dict(group="exploratory", points=pts))
        self.assertTrue(lk["fresh_near80_allowed"])
        self.assertTrue(lk["claim3_dropped"])
        self.assertEqual(lk["brackets"]["V"], 7.0)
        self.assertEqual(lk["brackets"]["arm"], "BCA's switched ramp")
        self.assertIn(dict(id="claim3.dropped", text=explo(expect("claim3.dropped", arm="BCA's switched ramp",
                                                                  V="7.0"))), lk["sentences"])
        self.assertEqual(lk["claims"]["claim3"]["exploratory_outcome"], "dropped")
        self.assertEqual(lk["stops"]["noise_floor"]["differences"]["none"]["mean"], 0.0)

    def test_noise_floor_tie_names_the_range_limit(self):
        lk = self.look(dict(group="exploratory"))
        self.assertEqual((lk["brackets"]["V"], lk["brackets"]["arm"]), (1.0, "the hand-set range limit"))

    def test_without_the_random_rows_control(self):
        root = self.tree(dict(group="exploratory"))
        R = P.load_results(root, "exploratory", "scratch", defaults=dict(random_rows_control=False))
        st = P.stop_rules(R)
        self.assertFalse(st["noise_floor"]["run"])
        self.assertFalse(st["claim3_dropped"])
        self.assertEqual(st["rival_unaffected"]["B"], 11.0)       # main 61 minus near-80 50
        self.assertTrue(st["fresh_near80_allowed"])

    def test_no_training_first_failed_rule(self):
        rules = [dict(rule="check 2: the evaluation split keeps band rows", scope="stall/seed0", value=800,
                      threshold=">= 300", passed=True),
                 dict(rule="check 7: the calibration bank keeps enough rows", scope="stall/seed0", value=480,
                      threshold=">= 500", passed=False),
                 dict(rule="check 3: the band is the planned size", scope="stall/seed2", value=0.0712,
                      threshold="0.01-0.06", passed=False)]
        self.assertEqual(P.halt_sentence_no_training(rules, BLOCK), expect(
            "stop.no_training", check="check 3, the extra rows as a share of the ball-kept rows", value="7.12%",
            threshold="1% to 6%"))
        gate = [dict(rule="check 4: kept rows past 80%", value=0.0031, threshold="<= 0.002", passed=False)]
        self.assertEqual(P.halt_sentence_no_training(gate, BLOCK), expect(
            "stop.no_training", check="check 4, the share of kept rows past 80% in the near-80 setting",
            value="0.0031", threshold="the gate's mark, at most 0.002"))
        gap = [dict(rule="check 6: BCA's gap rule is defined, lower end at the band's edge", value=None,
                    threshold="defined, t_lo in [0.695, 0.7]", passed=False)]
        self.assertIn("(not defined; needed defined, lower end in [0.695, 0.70])",
                      P.halt_sentence_no_training(gap, BLOCK))
        with self.assertRaises(ValueError):
            P.halt_sentence_no_training(rules[:1], BLOCK)

    def test_halt_report_no_training_comes_first(self):
        bad = [dict(rule="check 5: the band was a real part of driving where it matters", scope="stall/seed1",
                    value=0.08, threshold=">= 0.1", passed=False)]
        root = self.tree(dict(group="exploratory", halts={(1, "stall"): bad},
                              plumbing={"stall": {"critic": dict(passed=False, plain="the main test's critic")}}))
        rep = P.halt_report(root, templates=BLOCK, mode="scratch")
        self.assertEqual(rep["stop"], "stop.no_training")
        self.assertEqual(rep["sentences"], [dict(id="stop.no_training", text=expect(
            "stop.no_training", check="check 5, the band rows among the ball-kept hard-water rows", value="8%",
            threshold="at least 10%"))])

    def test_halt_report_reproduction_one_two_three(self):
        cases = (({"critic": dict(passed=False, plain="the main test's critic"), "clone": dict(passed=True)},
                  "the main test's critic"),
                 ({"stall": {"obca_k1": dict(passed=False, plain="step 2's switched-ramp actor")},
                   "no_stall": {"critic": dict(passed=False, plain="the main test's critic"),
                                "obca_k1": dict(passed=False, plain="step 2's switched-ramp actor")}},
                  "the main test's critic and step 2's switched-ramp actor"),
                 ({"range_hinge": dict(passed=False), "removal": dict(passed=False), "critic": dict(passed=False)},
                  "the main test's removal, the main test's critic and step 2's hand-set range-limit actor"))
        t = T["stop.reproduction"]
        lst = re.search(r"\[[^\]]* / [^\]]*\]", t).group(0)
        for n, (repro, names) in enumerate(cases):
            root = self.tree(dict(group="exploratory", plumbing=repro), sub=f"r{n}")
            rep = P.halt_report(root, templates=BLOCK, mode="scratch")
            self.assertEqual(rep["sentences"], [dict(id="stop.reproduction", text=t.replace(lst, names))])

    def test_halt_report_harm_rival_and_phase_lines(self):
        root = self.tree(dict(stops=[dict(phase_kind="critic", setting="main", world="no_stall", seed=11,
                                          reason="a learner that fails (a non-finite network)"),
                                     dict(phase_kind="lock", group="exploratory",
                                          reason="a saved input whose fingerprint does not match")],
                              exploratory=dict(allowed=False, H=50.0, Pshare=0.4, B=2.0)))
        rep = P.halt_report(root, templates=BLOCK, mode="scratch")
        self.assertEqual([r["id"] for r in rep["sentences"]], ["stop.rival", "stop.claim4_note", "stop.phase",
                                                               "stop.phase"])
        self.assertEqual(rep["sentences"][0]["text"], expect("stop.rival", B="2.0"))
        self.assertEqual(rep["sentences"][2]["text"], expect("stop.phase", s="0-4", world="both worlds",
                                                             phase="the lock",
                                                             reason="a saved input whose fingerprint does not match"))
        self.assertTrue(rep["sentences"][2]["text"].startswith("Seed 0-4, both worlds: the lock stopped"))
        self.assertEqual(rep["sentences"][3]["text"], expect(
            "stop.phase", s="11", world="no-stall world", phase="the critic (main setting)",
            reason="a learner that fails (a non-finite network)"))
        root = self.tree(dict(exploratory=None), sub="none")
        rep = P.halt_report(root, templates=BLOCK, mode="scratch")
        self.assertEqual((rep["stop"], rep["sentences"]), (None, []))


# --------------------------------------------------------------------------------------------- files and modes

class TestFilesAndModes(Base):
    def test_run_look_writes_once_and_the_fields_the_runner_reads(self):
        root = self.tree(dict(group="exploratory"))
        prereg = dict(constants=P.canonical_constants())
        look = P.run_look("exploratory", root, "scratch", prereg)
        body, digest = P.read_json_verified(root / "looks" / "exploratory.json")
        self.assertEqual(body["kind"], "near80-look-exploratory")
        self.assertEqual((body["group"], body["prereg_sha256"], body["fresh_near80_allowed"]),
                         ("exploratory", None, True))
        self.assertEqual(body["sentences"], look["sentences"])
        md = (root / "looks" / "exploratory.md").read_text()
        self.assertIn(digest, md)
        self.assertIn(look["sentences"][0]["text"], md)
        self.assertIn("data/main/stall/seed0/data.json", body["inputs"])
        with self.assertRaises(P.LookRefused):
            P.run_look("exploratory", root, "scratch", prereg)
        with self.assertRaises(P.LookRefused):
            P.run_look("confirmatory", root, "scratch", dict(constants={"ALPHA": 1}))
        with self.assertRaises(P.LookRefused):
            P.run_look("exploratory", K.N80 / "no_such_folder", "scratch", prereg)
        self.assertFalse((K.N80 / "no_such_folder").exists())

    def test_explore_then_confirm_end_to_end(self):
        root = self.tree(dict(group="exploratory"))
        make_tree(self.tmp / "a", dict(exploratory=None))
        prereg = dict(constants=P.canonical_constants())
        P.run_look("exploratory", root, "scratch", prereg)
        look = P.run_look("confirmatory", root, "scratch", prereg)
        self.assertEqual(self.ids(look), ["always_attached", "claim1.pass", "claim2.pass_pass", "claim2.scale",
                                          "claim3.holds", "claim4.holds", "claim4.both_settings"])
        self.assertEqual(look["exploratory_look_sha256"], P.file_sha256(root / "looks" / "exploratory.json"))
        self.assertTrue((root / "looks" / "confirmatory.md").is_file())

    def test_official_refuses_a_scratch_record_and_another_fingerprint(self):
        good = "a" * 64
        root = self.tree(dict(mode="official", prereg_sha256=good, group="exploratory"))
        with mock.patch.object(K, "OUT", root):
            R = P.load_results(root, "exploratory", "official", prereg_sha256=good)
            self.assertEqual(R.mode, "official")
            with self.assertRaises(P.LookRefused) as e:
                P.load_results(root, "exploratory", "official", prereg_sha256="b" * 64)
            self.assertIn("another pre-registration", str(e.exception))
        rel = K.LAYOUT["values"].format(setting="near80", world="stall", seed=3)
        body, _ = P.read_json_verified(root / rel)
        put(root / rel, dict(body, mode="scratch"))
        with mock.patch.object(K, "OUT", root):
            with self.assertRaises(P.LookRefused) as e:
                P.load_results(root, "exploratory", "official", prereg_sha256=good)
            self.assertIn("not an official record", str(e.exception))
        with self.assertRaises(P.LookRefused):
            P.load_results(root, "exploratory", "official", prereg_sha256=good)   # not the official folder

    def test_official_run_look_with_a_prereg_fixture(self):
        n80 = self.tmp / "n80"
        n80.mkdir()
        root = self.tmp / "a" / "out"
        with mock.patch.object(K, "OUT", root), mock.patch.object(K, "N80", n80):
            prereg = dict(kind="near80-preregistration", constants=P.canonical_constants(),
                          claims_word_for_word=BLOCK)
            digest = put(n80 / "prereg.json", prereg)
            self.tree(dict(mode="official", prereg_sha256=digest, group="exploratory"))
            look = P.run_look("exploratory", root, "official", json.loads(P.canonical(prereg)))
            with self.assertRaises(P.LookRefused):
                P.run_look("confirmatory", root, "official", dict(prereg, kind="other"))
        self.assertEqual((look["mode"], look["prereg_sha256"]), ("official", digest))
        body, _ = P.read_json_verified(root / "looks" / "exploratory.json")
        self.assertEqual(body["prereg_sha256"], digest)

    def test_tampered_and_mismatched_records_are_refused(self):
        root = self.tree()
        rel = K.LAYOUT["values"].format(setting="near80", world="stall", seed=12)
        (root / rel).write_text((root / rel).read_text().replace("80.0", "81.0"))
        with self.assertRaises(P.LookRefused) as e:
            P.load_results(root, "confirmatory", "scratch")
        self.assertIn("sidecar", str(e.exception))
        root = self.tree(sub="b")
        rel = K.LAYOUT["values"].format(setting="near80", world="stall", seed=12)
        body, _ = P.read_json_verified(root / rel)
        put(root / rel, dict(body, lock_sha256="0" * 64))
        with self.assertRaises(P.LookRefused) as e:
            P.load_results(root, "confirmatory", "scratch")
        self.assertIn("not the evaluation of the lock", str(e.exception))
        root = self.tree(sub="c")
        body, _ = P.read_json_verified(root / rel)
        put(root / rel, dict(body, mode="stub"))
        with self.assertRaises(P.LookRefused):
            P.load_results(root, "confirmatory", "scratch")

    def test_seeds_from_the_plan_and_reserved_seeds_refused(self):
        root = self.tree(dict(seeds=[10, 11], plan="seeds:0,10,11"))
        R = P.load_results(root, "confirmatory", "scratch")
        self.assertEqual(R.seeds, [10, 11])
        self.assertEqual(P.resolve_seeds("exploratory", None, "smoke", "scratch"), [0])
        self.assertEqual(R.pts["near80"]["stall"][10]["obca_k1"][SC], 80.0)
        self.assertEqual(R.unseen_gain["near80"]["stall"][11]["none"][SC], 10.0)
        for bad in ([5], [7, 10], [20]):
            with self.assertRaises(P.LookRefused):
                P.resolve_seeds("confirmatory", bad, None, "scratch")
        with self.assertRaises(P.LookRefused):
            P.resolve_seeds("confirmatory", [10, 11], None, "official")

    def test_confirmatory_needs_the_exploratory_look(self):
        root = self.tree(dict(exploratory=None))
        with self.assertRaises(P.LookRefused):
            P.load_results(root, "confirmatory", "scratch")

    def test_main_cli_refuses_official_looks(self):
        self.assertEqual(P.main(["exploratory", "--out-root", str(self.tmp), "--mode", "official"]), K.EXIT_REFUSED)

    def test_nothing_written_into_the_code_folder(self):
        self.assertFalse((HERE / "__pycache__").exists())
        for name in ("pooled.py", "claims_text.py", "test_pooled.py"):
            self.assertIn(name, K.CODE_FILES)
            text = (HERE / name).read_text(encoding="utf8")
            self.assertNotIn("DRAFT", text.split("\n")[0])
            self.assertTrue("sys.dont_write_bytecode = True" in text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
