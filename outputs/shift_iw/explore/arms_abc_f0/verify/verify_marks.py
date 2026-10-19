#!/usr/bin/env python3
"""Independent recomputation of every mark of arms A, C and F0 (and the information rows asked for) from the per-seed
raw scores, following PASSMARKS_arms_ABC_and_F_control.md sections 3.2, 4.2-4.3, 6.2-6.3 and 7.4.

Verification only. Reads the evaluation records, F0's records, A2's evaluate_seed<k>.json, KILLS.json, results.json and
RESULTS.md; writes only verify/verified.json (and VERIFIED_NUMBERS.md is written by make_md.py from it). Does not import
or call score.py or any project code. Pure Python statistics (no numpy), so the arithmetic is independent of score.py.

Run: PYTHONDONTWRITEBYTECODE=1 python3 -I verify_marks.py
"""
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

SP = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad")
ARMS = SP / "routes2" / "arms"
OUT = ARMS / "verify"
RUNS = SP / "iw_rerun" / "repo" / "runs" / "shift_iw_v5_explore"
ROOT = RUNS / "arms_abc_f0"
EVAL = ROOT / "eval"
CH = RUNS / "channel" / "train_f1"
PASS = ARMS / "PASSMARKS_arms_ABC_and_F_control.md"

QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
P1_Q = ("medium-replay", "medium", "medium-expert")          # section 4.3 P1
WEAK = ("random", "medium-replay", "medium")                  # section 6.3 B1
F0_Q = ("medium-expert", "expert")                            # section 7
SEEDS = (0, 1, 2, 3, 4)
LAWS = ("none", "mild", "strong")
T4 = 2.132
PLAIN_A = ("plain_1s", "plain_2s", "plain_4s")
PLAIN_C = ("wmedian", "wmean", "sign", "wttest", "wbernstein")
TOL = 1e-9

problems = []        # integrity problems with raw records
notes = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sidecar_ok(p):
    s = Path(str(p) + ".sha256")
    if not s.exists():
        return None
    want = s.read_text().split()[0]
    return want == sha(p)


# ------------------------------------------------------------------ frozen file, kills
DIGEST = sha(PASS)
if not sidecar_ok(PASS):
    problems.append("frozen pass-mark file does not match its sidecar")
KPATH = ARMS / "KILLS.json"
if not sidecar_ok(KPATH):
    problems.append("KILLS.json does not match its sidecar")
KSHA = sha(KPATH)
K = json.loads(KPATH.read_text())


def usable(arm, q, k):
    v = (K.get(arm) or {}).get(q) or {}
    return bool(v.get("go_on")) and int(k) in [int(x) for x in v.get("seeds_usable", [])]


# ------------------------------------------------------------------ raw readers
_cache = {}


def eval_record(arm, q, k):
    """The evaluation record of arm A or C (score.py eval's output), checked: sidecar, status, quality and seed match
    the path (pairing by seed), the frozen file's digest, the current KILLS.json, and the arm record it scored is the
    current one."""
    key = (arm, q, k)
    if key in _cache:
        return _cache[key]
    p = EVAL / arm / q / f"seed{k}" / "record.json"
    if not p.exists():
        problems.append(f"missing {p}")
        _cache[key] = (p, None)
        return _cache[key]
    r = json.loads(p.read_text())
    bad = []
    if sidecar_ok(p) is not True:
        bad.append("sidecar mismatch")
    if r.get("status") != "finished":
        bad.append(f"status {r.get('status')}")
    if r.get("quality") != q or int(r.get("seed")) != int(k):
        bad.append(f"quality/seed fields {r.get('quality')}/{r.get('seed')} do not match the path")
    if r.get("passmarks_sha256") != DIGEST:
        bad.append("another passmarks digest")
    if DIGEST not in (r.get("label") or ""):
        bad.append("label without the frozen digest")
    if r.get("kills_sha256") != KSHA:
        bad.append("kills_sha256 is not the current KILLS.json")
    arm_rec = ROOT / arm / q / f"seed{k}" / "record.json"
    if r.get("arm_record_sha256") != sha(arm_rec):
        bad.append("arm record changed since it was scored")
    if bad:
        problems.append(f"{p}: {'; '.join(bad)}")
    _cache[key] = (p, r if not bad else r)
    return _cache[key]


def pts_eval(arm, q, k, policy, law="strong"):
    """(points, source path) or (None, path) when missing/non-finite/not usable."""
    p, r = eval_record(arm, q, k)
    if r is None or not usable(arm, q, k):
        return None, p
    e = ((r.get("evaluations") or {}).get(policy) or {}).get(law)
    if e is None or e.get("finite") is not True or not math.isfinite(e.get("points", float("nan"))) \
            or e.get("starts") != 1000:
        return None, p
    return float(e["points"]), p


def f0_record(q, k):
    key = ("F0rec", q, k)
    if key in _cache:
        return _cache[key]
    p = ROOT / "F0" / q / "F0_no_band" / f"seed{k}" / "record.json"
    r = json.loads(p.read_text()) if p.exists() else None
    bad = []
    if r is None:
        bad.append("missing")
    else:
        if sidecar_ok(p) is not True:
            bad.append("sidecar mismatch")
        if r.get("status") != "finished":
            bad.append(f"status {r.get('status')}")
        if r.get("quality") != q or int(r.get("seed")) != int(k):
            bad.append("quality/seed fields do not match the path")
        if r.get("passmarks_sha256") != DIGEST:
            bad.append("another passmarks digest")
        a2rec = CH / q / "A2_cmd_strong" / f"seed{k}" / "record.json"
        if r.get("a2_record") != str(a2rec):
            bad.append(f"a2_record points at {r.get('a2_record')}")
        if r.get("a2_record_sha256") != sha(a2rec):
            bad.append("A2 record changed since F0 read it")
        kf3 = (r.get("checks") or {}).get("KF3") or {}
        if kf3.get("clone_sha256") is None or kf3.get("clone_sha256") != kf3.get("a2_clone_sha256"):
            bad.append("KF3 clone digests differ or are missing")
        a2 = json.loads(a2rec.read_text())
        if r.get("rows_sha256") != a2.get("rows_sha256"):
            bad.append("F0's row stream digest differs from A2's")
    if bad:
        problems.append(f"{p}: {'; '.join(bad)}")
    _cache[key] = (p, r)
    return _cache[key]


def a2_eval(q, k):
    key = ("A2", q, k)
    if key in _cache:
        return _cache[key]
    p = CH / q / f"evaluate_seed{k}.json"
    rec = CH / q / "A2_cmd_strong" / f"seed{k}" / "record.json"
    b = json.loads(p.read_text()) if p.exists() else None
    bad = []
    if b is None:
        bad.append("missing")
    else:
        if b.get("quality") != q or int(b.get("seed")) != int(k):
            bad.append("quality/seed fields do not match the path")
        if os.stat(p).st_mtime < os.stat(rec).st_mtime:
            bad.append("evaluation older than A2's record")
    if bad:
        problems.append(f"{p}: {'; '.join(bad)}")
    _cache[key] = (p, b)
    return _cache[key]


def pts_a2(q, k, law):
    p, b = a2_eval(q, k)
    if b is None:
        return None, p
    e = b["evaluations"]["A2_cmd_strong"][law]
    if e.get("finite") is not True or not math.isfinite(e["points"]):
        return None, p
    return float(e["points"]), p


def pts_f0(q, k, law):
    p, r = f0_record(q, k)
    if r is None or r.get("status") != "finished":
        return None, p
    e = r["evaluation"][law]
    if e.get("finite") is not True or not math.isfinite(e["points"]):
        return None, p
    return float(e["points"]), p


# ------------------------------------------------------------------ statistics (section 3.2)

def summarize(per, raw_a, raw_b, src_a, src_b):
    out = dict(per_seed=per, arm_points=raw_a, comparator_points=raw_b, source_arm=src_a, source_comparator=src_b)
    if any(v is None for v in per) or len(per) != 5:
        out.update(mean=None, positives=None, negatives=None, ties=None, lower_bound=None,
                   lower_bound_population_sd=None, sd_sample=None, sd_population=None, missing=True)
        return out
    mean = math.fsum(per) / 5.0
    sd_s = statistics.stdev(per)          # sample SD (n - 1): what a t with 4 df implies
    sd_p = statistics.pstdev(per)         # population SD (n), shown only for contrast
    out.update(mean=mean, positives=sum(1 for v in per if v > 0), negatives=sum(1 for v in per if v < 0),
               ties=sum(1 for v in per if v == 0), sd_sample=sd_s, sd_population=sd_p,
               lower_bound=mean - T4 * sd_s / math.sqrt(5), lower_bound_population_sd=mean - T4 * sd_p / math.sqrt(5),
               missing=False)
    return out


def diff_eval(arm, q, a, b, law="strong"):
    per, ra, rb, sa, sb = [], [], [], [], []
    for k in SEEDS:
        x, pa = pts_eval(arm, q, k, a, law)
        y, pb = pts_eval(arm, q, k, b, law)
        ra.append(x)
        rb.append(y)
        sa.append(str(pa))
        sb.append(str(pb))
        per.append(None if x is None or y is None else x - y)
    s = summarize(per, ra, rb, sa, sb)
    s.update(arm_policy=a, comparator_policy=b, law=law, quality=q)
    return s


def mean5(arm, q, policy, law="strong"):
    v = [pts_eval(arm, q, k, policy, law)[0] for k in SEEDS]
    return None if any(x is None for x in v) else math.fsum(v) / 5.0


def at_least(s, bar, positives=None):
    if s["mean"] is None:
        return False
    ok = s["mean"] >= bar                          # inclusive, full precision
    if positives is not None:
        ok = ok and s["positives"] >= positives    # a tie (d == 0) is not positive
    return bool(ok)


# ------------------------------------------------------------------ arm A (sections 4.2-4.3)
A = dict(P1={}, P2={}, P3={}, P4={}, info={})
passing_A = []
for q in P1_Q:
    s = diff_eval("A", q, "A1", "none")
    s["meets"] = at_least(s, 2.0, 4)
    A["P1"][q] = s
    if s["meets"]:
        passing_A.append(q)
P1 = len(passing_A) >= 2
P2_fail = []
for q in QUALITIES:
    for law in LAWS:
        s = diff_eval("A", q, "A1", "none", law)
        s["meets"] = at_least(s, -1.0)
        A["P2"][f"{q}/{law}"] = s
        if not s["meets"]:
            P2_fail.append((q, law, s["mean"]))
P2 = not P2_fail
P3 = bool(passing_A)
P4 = bool(passing_A)
rivals_tied_or_won = []
for q in passing_A:
    s = diff_eval("A", q, "A1", "untargeted")
    s["meets"] = at_least(s, 1.5)
    A["P3"][q] = s
    P3 = P3 and s["meets"]
    plain_means = {p: mean5("A", q, p) for p in PLAIN_A}
    cand = [p for p in PLAIN_A if plain_means[p] is not None]
    best = None
    if cand:
        top = max(plain_means[p] for p in cand)
        tied = [p for p in cand if plain_means[p] == top]
        best = tied[0]
        if len(tied) > 1:
            notes.append(f"A P4 {q}: plain margins tie for best: {tied}")
    row = dict(plain_means=plain_means, best_plain=best)
    for name, rival in (("margin0", "margin0"), ("best_plain", best), ("matched_share", "matched_share")):
        if rival is None:
            s = summarize([None] * 5, [None] * 5, [None] * 5, [], [])
        else:
            s = diff_eval("A", q, "A1", rival)
        s["rival"] = rival
        s["meets"] = at_least(s, 1.0)
        row[name] = s
        if not s["meets"]:
            rivals_tied_or_won.append(f"{rival} on {q}")
    # every plain margin, for transparency (only the best one counts)
    row["all_plain"] = {p: diff_eval("A", q, "A1", p) for p in PLAIN_A}
    A["P4"][q] = row
    P4 = P4 and all(row[n]["meets"] for n in ("margin0", "best_plain", "matched_share"))
for q in QUALITIES:
    A["info"][q] = {"even_x1.33-none": diff_eval("A", q, "even_x1.33", "none"),
                    "A1c-none": diff_eval("A", q, "A1c", "none")}
A.update(passing=passing_A, met=dict(P1=P1, P2=P2, P3=P3, P4=P4), GO=bool(P1 and P2 and P3 and P4),
         rivals_tied_or_won=rivals_tied_or_won, harm=P2_fail)

# ------------------------------------------------------------------ arm C (sections 6.2-6.3)
Cm = dict(B1={}, B2={}, B3={}, B4={}, B5={}, info={})
passing_C = []
for q in WEAK:
    s = diff_eval("C", q, "rho0.5+bca", "rho1")
    s["meets"] = at_least(s, 3.0, 4)
    Cm["B1"][q] = s
    if s["meets"]:
        passing_C.append(q)
B1 = len(passing_C) >= 2
B5_fail = []
for q in QUALITIES:
    for law in LAWS:
        s = diff_eval("C", q, "rho0.5+bca", "rho1", law)
        s["meets"] = at_least(s, -1.0)
        Cm["B5"][f"{q}/{law}"] = s
        if not s["meets"]:
            B5_fail.append((q, law, s["mean"]))
B5 = not B5_fail
# B2-B4: values computed for every passing quality; whether they are scored is a kill-rule outcome (section 6.4)
B2v = B3v = B4v = bool(passing_C)
for q in passing_C:
    s = diff_eval("C", q, "rho0.5+bca", "rho0.5")
    s["meets"] = at_least(s, 1.5)
    Cm["B2"][q] = s
    B2v = B2v and s["meets"]
    row = {}
    for name, bar in (("range_limit", 1.5), ("shuffled", 1.5), ("matched_global", 1.0)):
        s = diff_eval("C", q, "rho0.5+bca", f"rho0.5+{name}")
        s["meets"] = at_least(s, bar)
        row[name] = s
    Cm["B3"][q] = row
    B3v = B3v and all(v["meets"] for v in row.values())
    means = {t: mean5("C", q, f"rho0.5+{t}") for t in PLAIN_C}
    cand = [t for t in PLAIN_C if means[t] is not None]
    best = None
    if cand:
        top = max(means[t] for t in cand)
        tied = [t for t in cand if means[t] == top]
        best = tied[0]
        if len(tied) > 1:
            notes.append(f"C B4 {q}: plain tests tie for best: {tied}")
    s = diff_eval("C", q, "rho0.5+bca", f"rho0.5+{best}") if best else summarize([None] * 5, [None] * 5,
                                                                                 [None] * 5, [], [])
    s.update(best_plain_test=best, plain_means=means, meets=at_least(s, 1.0))
    Cm["B4"][q] = s
    B4v = B4v and s["meets"]
for q in QUALITIES:
    Cm["info"][q] = {"rho0.5-rho1": diff_eval("C", q, "rho0.5", "rho1"),
                     "rho0.5+knee_clip-rho1": diff_eval("C", q, "rho0.5+knee_clip", "rho1"),
                     "rho1+bca_built_for_rho1-rho1": diff_eval("C", q, "rho1+bca_built_for_rho1", "rho1"),
                     "rho0.5+bca-rho1": diff_eval("C", q, "rho0.5+bca", "rho1")}

# the guard-void condition (section 6.4), read from the records as a side check (B2-B4 are scored only if it is not void)
screen = json.loads((ARMS / "SCREEN_seed0.json").read_text())
b_k1 = {q: (screen["B"].get(q) or {}).get("K1") for q in QUALITIES}
b_k1_fires = [q for q, v in b_k1.items() if (v or {}).get("fires")]
own_void = []
for q in QUALITIES:
    for k in SEEDS:
        r = json.loads((ROOT / "C" / q / f"seed{k}" / "record.json").read_text())
        t1 = [x for x in r["details"]["rules"]["bca"] if x["thruster"] == 1][0]
        rung7 = [g for g in t1["rungs"] if g["rung"] == 7]
        tries7 = rung7[0]["tries"] if rung7 else 0
        if r["maps"]["bca"][0] >= 0.85 - 1e-7 and tries7 >= 30:
            own_void.append(f"{q} seed {k} (t1 {r['maps']['bca'][0]:.2f}, {tries7} tries in (0.80, 0.85])")
guard_void = bool(b_k1_fires) or bool(own_void)
Cm.update(passing=passing_C, guard=dict(B_K1_fires_on=b_k1_fires, B_K1_screen=b_k1, own_map_void=own_void,
                                        void=guard_void),
          values_if_scored=dict(B2=B2v, B3=B3v, B4=B4v),
          met=dict(B1=B1, B2=None if guard_void else B2v, B3=None if guard_void else B3v,
                   B4=None if guard_void else B4v, B5=B5),
          GO=bool(B1 and B5 and (not guard_void) and B2v and B3v), harm=B5_fail)

# ------------------------------------------------------------------ F0 (section 7.4)
F = {}
for q in F0_Q:
    row = {}
    for law in LAWS:
        per, ra, rb, sa, sb = [], [], [], [], []
        for k in SEEDS:
            x, pa = pts_a2(q, k, law)
            y, pb = pts_f0(q, k, law)
            ra.append(x)
            rb.append(y)
            sa.append(str(pa))
            sb.append(str(pb))
            per.append(None if x is None or y is None else x - y)
        s = summarize(per, ra, rb, sa, sb)
        s.update(arm_policy="A2_cmd_strong", comparator_policy="F0_no_band", law=law, quality=q)
        row[law] = s
    st = row["strong"]
    row["credit"] = at_least(st, 1.0, 4)
    row["negative"] = st["mean"] is not None and st["mean"] <= -1.0 and st["negatives"] >= 4
    F[q] = row
credit = [q for q in F0_Q if F[q]["credit"]]
if len(credit) == 2:
    f_word = "BCA's band earned part of result F's gain"
elif len(credit) == 1:
    f_word = f"Result F's gain comes partly from BCA's band (on {credit[0]})"
else:
    f_word = "Result F's gain comes from refusing credit past the data, not from BCA's band."

# A2 seed 0 is the result F rerun's: its evaluate_seed0.json must equal scores_seed0.json's f1_A2_cmd_strong
rf = json.loads((SP / "resultF" / "results" / "scores_seed0.json").read_text())
a2_seed0_same = {}
for q in F0_Q:
    b = a2_eval(q, 0)[1]
    a2_seed0_same[q] = all(b["evaluations"]["A2_cmd_strong"][law]["points"] ==
                           rf["scores"][q]["f1_A2_cmd_strong"][law]["points"] for law in LAWS)

# ------------------------------------------------------------------ arm B: nothing of B enters a mark
res = json.loads((ARMS / "results.json").read_text())
if sidecar_ok(ARMS / "results.json") is not True:
    problems.append("results.json does not match its sidecar")
Bcheck = dict(
    kills_go_on=K["B"].get("go_on"), kills_stopped_by=K["B"].get("stopped_by"),
    eval_B_dir_exists=(EVAL / "B").exists(),
    results_inputs_with_B=[p for p in res["inputs_sha256"] if "/eval/B/" in p or "/arms_abc_f0/B/" in p],
    results_B_numbers=[f"{m}:{k}" for m in ("G2",) for k, v in res["B"]["G2"].items()
                       if any(x is not None for x in v["per_seed"])] +
                      ([f"G1"] if any(x is not None for x in res["B"]["G1"]["per_seed"]) else []) +
                      [f"G3:{k}" for k, v in res["B"]["G3"].items() if any(x is not None for x in v["per_seed"])],
    results_B_met=res["B"]["met"], results_B_GO=res["B"]["GO"],
    G4_best=res["B"]["G4"]["best_plain_test"], G4_total=res["B"]["G4"]["total"],
    B_host_map_in_C_marks=False,   # set below by scanning the C marks' policies
)
c_policies = set()
for grp in ("B1", "B2", "B4"):
    for v in Cm[grp].values():
        c_policies |= {v.get("arm_policy"), v.get("comparator_policy")}
for v in Cm["B5"].values():
    c_policies |= {v.get("arm_policy"), v.get("comparator_policy")}
for row in Cm["B3"].values():
    for v in row.values():
        c_policies |= {v.get("arm_policy"), v.get("comparator_policy")}
Bcheck["B_host_map_in_C_marks"] = any("B_host_map" in (p or "") for p in c_policies)
Bcheck["C_mark_policies"] = sorted(p for p in c_policies if p)

# ------------------------------------------------------------------ comparison with results.json and RESULTS.md
mismatch = []        # (where, field, mine, theirs, size)


def cmp_num(where, field, mine, theirs):
    if mine is None and theirs is None:
        return
    if (mine is None) != (theirs is None):
        mismatch.append((where, field, mine, theirs, "missing on one side"))
        return
    if isinstance(mine, bool) or isinstance(theirs, bool) or isinstance(mine, int) and isinstance(theirs, int):
        if mine != theirs:
            mismatch.append((where, field, mine, theirs, "differs"))
        return
    d = abs(float(mine) - float(theirs))
    if d > TOL:
        mismatch.append((where, field, mine, theirs, d))


def cmp_summary(where, mine, theirs):
    if theirs is None:
        mismatch.append((where, "row", "present", None, "absent in results.json"))
        return {}
    for i, (a, b) in enumerate(zip(mine["per_seed"], theirs["per_seed"])):
        cmp_num(where, f"d_{i}", a, b)
    cmp_num(where, "mean", mine["mean"], theirs.get("mean"))
    cmp_num(where, "positives", mine["positives"], theirs.get("positives"))
    cmp_num(where, "lower_bound", mine["lower_bound"], theirs.get("lower_bound"))
    if "meets" in mine and "meets" in theirs:
        cmp_num(where, "meets", bool(mine["meets"]), bool(theirs["meets"]))
    dev = max([abs(a - b) for a, b in zip(mine["per_seed"], theirs["per_seed"]) if a is not None and b is not None]
              + ([abs(mine["mean"] - theirs["mean"])] if mine["mean"] is not None and theirs.get("mean") is not None
                 else []) + ([abs(mine["lower_bound"] - theirs["lower_bound"])]
                             if mine["lower_bound"] is not None and theirs.get("lower_bound") is not None else []),
              default=None)
    return dict(max_abs_dev=dev)


# RESULTS.md rows
md = (ARMS / "RESULTS.md").read_text().splitlines()
md_rows = {}
for line in md:
    if line.startswith("| ") and not line.startswith("|---"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        md_rows.setdefault(cells[0], cells)


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def cmp_md(label, mine, kind="mark"):
    """Compare the 2-decimal rendering. kind 'mark': | name | mean | pos | lb | per seed | bar | met |;
    'info': | name | mean | pos | per seed |."""
    cells = md_rows.get(label)
    if cells is None:
        mismatch.append((f"RESULTS.md '{label}'", "row", "expected", None, "row absent"))
        return None
    want_seed = ", ".join(f2(v) for v in mine["per_seed"])
    pos = "-" if mine["positives"] is None else str(mine["positives"])
    if kind == "mark":
        got = dict(mean=cells[1], pos=cells[2], lb=cells[3], seeds=cells[4], met=cells[6])
        exp = dict(mean=f2(mine["mean"]), pos=pos, lb=f2(mine["lower_bound"]), seeds=want_seed,
                   met="yes" if mine.get("meets") else "no")
    else:
        got = dict(mean=cells[1], pos=cells[2], seeds=cells[3])
        exp = dict(mean=f2(mine["mean"]), pos=pos, seeds=want_seed)
    for k in exp:
        if exp[k] != got[k]:
            mismatch.append((f"RESULTS.md '{label}'", k, exp[k], got[k], "2-decimal rendering differs"))
    return got


rows = []   # (section, label, mine, results.json summary, RESULTS.md cells, kind, bar)
RA, RC, RF = res["A"], res["C"], res["F0"]
for q, s in A["P1"].items():
    rows.append(("A", f"P1 A1 - none, {q}", s, RA["P1"].get(q), "mark", ">= +2.0, 4 of 5"))
for k_, s in A["P2"].items():
    rows.append(("A", f"P2 A1 - none, {k_}", s, RA["P2"].get(k_), "mark", ">= -1.0"))
for q, s in A["P3"].items():
    rows.append(("A", f"P3 A1 - untargeted, {q}", s, RA["P3"].get(q), "mark", ">= +1.5"))
for q, v in A["P4"].items():
    for n in ("margin0", "best_plain", "matched_share"):
        theirs = (RA["P4"].get(q) or {}).get(n)
        if theirs is not None and theirs.get("rival") != v[n]["rival"]:
            mismatch.append((f"A P4 {q} {n}", "rival", v[n]["rival"], theirs.get("rival"), "different rival"))
        rows.append(("A", f"P4 A1 - {v[n]['rival']}, {q}", v[n], theirs, "mark", ">= +1.0"))
for q, v in Cm["B1"].items():
    rows.append(("C", f"B1 (rho 0.5 + BCA) - rho 1, {q}", v, RC["B1"].get(q), "mark", ">= +3.0, 4 of 5"))
for k_, v in Cm["B5"].items():
    rows.append(("C", f"B5 (rho 0.5 + BCA) - rho 1, {k_}", v, RC["B5"].get(k_), "mark", ">= -1.0"))
for q, v in Cm["B2"].items():
    rows.append(("C", f"B2 (rho 0.5 + BCA) - rho 0.5, {q}", v, RC["B2"].get(q), "mark", ">= +1.5 (not scored)"))
for q, row in Cm["B3"].items():
    for n, v in row.items():
        rows.append(("C", f"B3 BCA - {n}, {q}", v, (RC["B3"].get(q) or {}).get(n), "mark",
                     (">= +1.0" if n == "matched_global" else ">= +1.5") + " (not scored)"))
for q, v in Cm["B4"].items():
    theirs = RC["B4"].get(q)
    if theirs is not None and theirs.get("best_plain_test") != v["best_plain_test"]:
        mismatch.append((f"C B4 {q}", "best_plain_test", v["best_plain_test"], theirs.get("best_plain_test"),
                         "different best plain test"))
    rows.append(("C", f"B4 BCA - {v['best_plain_test']}, {q}", v, theirs, "mark", ">= +1.0 (not scored)"))
for q in F0_Q:
    mine = dict(F[q]["strong"], meets=F[q]["credit"])
    theirs = dict(RF["per_quality"][q]["strong"], meets=RF["per_quality"][q]["credit"])
    rows.append(("F0", f"A2 - F0, {q} (strong)", mine, theirs, "mark", ">= +1.0, 4 of 5"))
# information rows
for q in QUALITIES:
    for n in ("even_x1.33-none", "A1c-none"):
        rows.append(("A-info", f"{n}, {q}", A["info"][q][n], RA["info"][q].get(n), "info", "information"))
for q in QUALITIES:
    for n in ("rho0.5-rho1", "rho0.5+knee_clip-rho1", "rho1+bca_built_for_rho1-rho1"):
        rows.append(("C-info", f"{n}, {q}", Cm["info"][q][n], RC["info"][q].get(n), "info", "information"))
    # rho0.5+bca - rho1 strong is B5's strong row; RESULTS.md has it only as that mark row
    rows.append(("C-info", f"rho0.5+bca-rho1, {q} (= B5 strong)", Cm["info"][q]["rho0.5+bca-rho1"],
                 RC["B5"].get(f"{q}/strong"), "info-b5", "information"))
for q in F0_Q:
    for law in ("mild", "none"):
        rows.append(("F0-info", f"A2 - F0, {q} ({law})", F[q][law], RF["per_quality"][q][law], "info",
                     "information"))

table = []
for sec, label, mine, theirs, kind, bar in rows:
    dev = cmp_summary(f"{sec} {label}", mine, theirs) if theirs is not None else {}
    if theirs is None:
        mismatch.append((f"{sec} {label}", "row", "present", None, "absent in results.json"))
    if kind == "info-b5":
        md_label = f"B5 (rho 0.5 + BCA) - rho 1, {label.split(', ')[1].split(' ')[0]}/strong"
        cells = md_rows.get(md_label)
        got = dict(mean=cells[1], pos=cells[2], seeds=cells[4]) if cells else None
        exp = dict(mean=f2(mine["mean"]), pos=str(mine["positives"]),
                   seeds=", ".join(f2(v) for v in mine["per_seed"]))
        if got != exp:
            mismatch.append((f"RESULTS.md '{md_label}'", "row", exp, got, "2-decimal rendering differs"))
    else:
        got = cmp_md(label, mine, "mark" if kind == "mark" else "info")
    table.append(dict(section=sec, label=label, bar=bar, mine=mine, results_json=theirs, results_md=got,
                      max_abs_dev=dev.get("max_abs_dev")))

# marks-level comparison
cmp_num("A met P1", "met", A["met"]["P1"], RA["met"]["P1"])
cmp_num("A met P2", "met", A["met"]["P2"], RA["met"]["P2"])
cmp_num("A met P3", "met", A["met"]["P3"], RA["met"]["P3"])
cmp_num("A met P4", "met", A["met"]["P4"], RA["met"]["P4"])
cmp_num("A GO", "GO", A["GO"], RA["GO"])
if A["passing"] != RA["passing"]:
    mismatch.append(("A passing", "qualities", A["passing"], RA["passing"], "differs"))
for m in ("B1", "B5"):
    cmp_num(f"C met {m}", "met", Cm["met"][m], RC["met"][m])
for m in ("B2", "B3", "B4"):
    if (Cm["met"][m] is None) != (RC["met"][m] is None) or (Cm["met"][m] is not None and Cm["met"][m] != RC["met"][m]):
        mismatch.append((f"C met {m}", "met", Cm["met"][m], RC["met"][m], "differs"))
cmp_num("C GO", "GO", Cm["GO"], RC["GO"])
if Cm["passing"] != RC["passing"]:
    mismatch.append(("C passing", "qualities", Cm["passing"], RC["passing"], "differs"))
if credit != RF["credit"]:
    mismatch.append(("F0 credit", "qualities", credit, RF["credit"], "differs"))
f_md = [l for l in md if l.startswith("> ") and "Result F" in l]
if not f_md or f_word not in f_md[0]:
    mismatch.append(("F0 wording", "line", f_word, f_md, "differs"))
a_word_md = [l for l in md if l.startswith("> Filtering helps") or l.startswith("> Exploratory: the credible") or
             l.startswith("> The credible-target pull did not") or l.startswith("> The gain does not come")]
if A["met"]["P1"] and A["met"]["P3"] and not A["met"]["P4"]:
    a_word = ("Filtering helps; the band adds nothing over a plain margin. Tied or won: "
              + ", ".join(A["rivals_tied_or_won"]) + ".")
else:
    a_word = None
if a_word is not None and (not a_word_md or a_word_md[0][2:] != a_word):
    mismatch.append(("A wording", "line", a_word, a_word_md, "differs"))

out = dict(frozen_sha256=DIGEST, kills_sha256=KSHA, results_json_sha256=sha(ARMS / "results.json"),
           results_md_sha256=sha(ARMS / "RESULTS.md"), A=A, C=Cm, F0=dict(per_quality=F, credit=credit, wording=f_word),
           a2_seed0_equals_resultF_scores=a2_seed0_same, B=Bcheck, problems=problems, notes=notes,
           mismatches=[dict(where=w, field=f, mine=m, theirs=t, size=s) for w, f, m, t, s in mismatch],
           table=table, a_wording=a_word, sd_code="score.summary: numpy std(ddof=1), the sample SD",
           tolerance=TOL)
(OUT / "verified.json").write_text(json.dumps(out, indent=1, default=str))
print(f"problems: {len(problems)}; mismatches: {len(mismatch)}; notes: {notes}")
for p in problems:
    print("PROBLEM", p)
for m in mismatch:
    print("MISMATCH", m)
print("A met", A["met"], "passing", A["passing"], "GO", A["GO"])
print("C met", Cm["met"], "passing", Cm["passing"], "guard void", Cm["guard"]["void"], Cm["guard"]["B_K1_fires_on"])
print("F0 credit", credit, f_word)
print("A2 seed 0 equals result F scores_seed0:", a2_seed0_same)
print("B:", {k: v for k, v in Bcheck.items() if k != "C_mark_policies"})
devs = [t["max_abs_dev"] for t in table if t["max_abs_dev"] is not None]
print("rows compared:", len(table), "largest |mine - results.json|:", max(devs) if devs else None)
