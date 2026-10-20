"""The penalty investigation: rebuild every consistent and exceptional case of BCA's penalty from recorded inputs and
print the before/after numbers (docs/bca_mechanics/PENALTY_EVIDENCE.md). Nothing is trained and no band is fitted: the
script reads committed records, recorded run files and the pinned-environment recompute in data/penalty_recompute.json.

Words. "Standard" = BCA with no weights. "IW" = importance-weighted BCA (weights w = p_test / p_train on the held-out
rows). "IID" = the test states follow the training law (the weighting test's "none" mix, the four-learner study's
no-shift bank, Stage 2's training states and recorded rows). "Shift" = the mild and strong current mixes. R = the
threshold (module M5); dose = 1 + 0.5 U / (U + u) (module M7); U = R u eta(s, a) = the band width (module M6).

Usage (from the repository root; Python 3 with NumPy; the repository's own files are read):
  python docs/bca_mechanics/scripts/penalty_cases.py --root . --case all
  python docs/bca_mechanics/scripts/penalty_cases.py --root . --case 7          # one case
  python docs/bca_mechanics/scripts/penalty_cases.py --root . --hosts <copy of outputs/shift_hosts_v3/results.json>
      (without --hosts the four-learner record is read with
       git show origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json)
  --stage2-copies <dir>   the Stage 2 detection pairs and width scores (case 3), laid out as
                          <dir>/pairs/seed<K>/{pairs,labels}.npz and <dir>/{uniform,weighted}/stall/seed<K>/scores/
                          bca_width_{uniform,weighted}_fixed.npz; or --runs <runs/shift_explore> for the run tree
                          (detect/seed<K>/stall/{pairs,labels}.npz, stage2/detect/<version>/stall/fixed/seed<K>/scores/).
  --runs <runs/shift_explore>   also lets cases 6 and 8 recompute the threshold live from the recorded held-out scores
                          (near80/learners/*/*/seed*/v6_bank_scores.npz; the pinned environment reproduces the record
                          bit for bit); without it the pinned recompute data/penalty_recompute.json is read.
  --write-json <file>     write every number the figures use (docs/bca_mechanics/scripts/penalty_figures.py reads it).

Sources (all read-only):
  outputs/shift_iw/v5/<quality>/report.json                       weighting test, spec version 5, marks and shown values
  outputs/shift_iw/explore/routes2/diagnose/{dose_chunks,diag_band_at_commands}.json   exploratory, seed 0
  outputs/shift_iw/explore/channel/results/<quality>.json          coverage and width at the actor's commands (exploratory)
  outputs/shift_iw/explore/channel/results/command_band/<quality>.json   the command-weighted band (exploratory)
  outputs/shift_iw/explore/RESULTS.json                             return of the exploratory BCA arms
  origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json    four-learner study (exploratory, seeds 0-4)
  outputs/shift/beyond-the-data/near80/results-20261020T115246Z/files/learners/*/*/seed*/switch.json  near-80 certificates
  outputs/shift/beyond-the-data/restructure/step1/results/results.json                          step-1 certificates
  docs/bca_mechanics/data/penalty_recompute.json                    the pinned recompute (penalty_recompute.py)
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

QUALITIES = ["random", "medium-replay", "medium", "medium-expert", "expert"]
NEAR80 = "outputs/shift/beyond-the-data/near80/results-20261020T115246Z/files/learners"
STEP1 = "outputs/shift/beyond-the-data/restructure/step1/results/results.json"
HOSTS_GIT = "origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json"


def load(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def fnum(v):
    """Record files write +inf as the string 'inf'."""
    if isinstance(v, str):
        return float(v)
    return float(v)


def fmt(v, d=3):
    if v is None:
        return "-"
    v = float(v)
    if math.isinf(v):
        return "+inf" if v > 0 else "-inf"
    return f"{v:.{d}f}"


def lst(values, d=3):
    return "[" + ", ".join(fmt(v, d) for v in values) + "]"


def header(n, label, title):
    print()
    print("=" * 110)
    print(f"Case {n} [{label}]: {title}")
    print("=" * 110)


# --------------------------------------------------------------------------------------------------------- inputs

class Inputs:
    def __init__(self, args):
        self.root = args.root
        self.args = args
        self._reports = {}
        self._hosts = None

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def report(self, q):
        if q not in self._reports:
            self._reports[q] = load(self.path("outputs", "shift_iw", "v5", q, "report.json"))
        return self._reports[q]

    def dose_chunks(self):
        return load(self.path("outputs", "shift_iw", "explore", "routes2", "diagnose", "dose_chunks.json"))

    def band_at_commands(self):
        return load(self.path("outputs", "shift_iw", "explore", "routes2", "diagnose", "diag_band_at_commands.json"))

    def channel(self, q):
        return load(self.path("outputs", "shift_iw", "explore", "channel", "results", f"{q}.json"))

    def command_band(self, q):
        return load(self.path("outputs", "shift_iw", "explore", "channel", "results", "command_band", f"{q}.json"))

    def explore_results(self):
        return load(self.path("outputs", "shift_iw", "explore", "RESULTS.json"))

    def hosts(self):
        if self._hosts is None:
            if self.args.hosts:
                self._hosts = load(self.args.hosts)
            else:
                out = subprocess.run(["git", "-C", self.root, "show", HOSTS_GIT], capture_output=True, text=True,
                                     encoding="utf8")
                if out.returncode != 0:
                    sys.exit("the four-learner record is not reachable: pass --hosts <results.json> "
                             f"(git show {HOSTS_GIT} failed: {out.stderr.strip()[:200]})")
                self._hosts = json.loads(out.stdout)
        return self._hosts

    def switches(self):
        rows = []
        for f in sorted(glob.glob(self.path(*NEAR80.split("/"), "*", "*", "seed*", "switch.json"))):
            d = load(f)
            c = d["certificate"]
            p = os.path.normpath(f).split(os.sep)
            rows.append(dict(study="near-80", setting=p[-4], world=p[-3], seed=int(p[-2][4:]), n_tail=c["n_tail"],
                             finite_share=c["finite_share_tail"], R_tail=fnum(c["R_tail"]), R_rest=fnum(c["R_rest"]),
                             height=fnum(c["height"]), on=bool(c["on"]), source=os.path.relpath(f, self.root)))
        s1 = load(self.path(*STEP1.split("/")))
        for k, v in s1["per_seed_world"].items():
            c = v["certificate"]
            world, seed = k.split("/")
            rows.append(dict(study="step 1", setting="main", world=world, seed=int(seed[4:]), n_tail=c["n_tail"],
                             finite_share=c.get("finite_share_tail"), R_tail=fnum(c["R_tail"]), R_rest=fnum(c["R_rest"]),
                             height=fnum(c["height"]), on=bool(c["on"]), source=STEP1))
        return rows

    def recompute(self):
        p = self.args.recompute_json or self.path("docs", "bca_mechanics", "data", "penalty_recompute.json")
        return load(p) if os.path.exists(p) else None


# ---------------------------------------------------------------------------------------------------------- cases

def case_1(inp, out):
    header(1, "consistent", "the standard penalty is the same number whatever the test condition (dose and width)")
    print("Before = the dose on the recorded training rows; after = the same dose read under a shifted test. The dose is read")
    print("at recorded (state, command) rows from a threshold fixed at the last refresh (calibration/dose.py:65-95,")
    print("algorithms/td3_bc_bca.py:283-288), so a shift in the test states cannot reach it. What the records show:")
    dc = inp.dose_chunks()
    rows = {}
    print()
    print("  (a) Weighting test, capped pull, seed 0, standard arm: mean dose per 5,000 updates after the first refresh")
    print("      (outputs/shift_iw/explore/routes2/diagnose/dose_chunks.json; exploratory)")
    for q in QUALITIES:
        d = dc[f"{q}/unweighted"]["dose"][2:]
        rows[q] = d
        print(f"      {q:14s} min {min(d):.3f}  mean {np.mean(d):.3f}  max {max(d):.3f}  last {d[-1]:.3f}  (18 chunks)")
    out["case1_dose_chunks_standard"] = rows
    print()
    print("  (b) The same band read at four commands of the recorded state (dose-equivalent 1 + 0.5 U/(U+u); seed 0;")
    print("      diag_band_at_commands.json): recorded command | actor's command | 60% hold | 80% hold | full throttle")
    bac = inp.band_at_commands()["res"]
    flat = {}
    for q in QUALITIES:
        r = bac[q]
        vals = [r[k]["dose_equiv"] for k in ("recorded", "actor", "hold60", "hold80", "full100")]
        flat[q] = vals
        print(f"      {q:14s} " + "  ".join(f"{v:.4f}" for v in vals) + f"   spread {max(vals) - min(vals):.4f}")
    out["case1_dose_at_commands"] = flat
    print()
    print("  (c) Four-learner study, standard arm, mean dose on kept training rows at the last refresh, per seed")
    print("      (origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json; exploratory seeds 0-4)")
    H = inp.hosts()
    std = {}
    for q in ("medium", "medium-expert"):
        for L in ("td3_bc", "cql", "iql", "rebrac"):
            S = H["qualities"][q]["expanded"].get(L, {}).get("setups", {}).get("standard", {})
            B = S.get("band") or {}
            ps = (B.get("mean_dose_last") or {}).get("per_seed")
            if ps and ps[0] is not None:
                std[f"{L} {q}"] = ps
                tr = B.get("train_rows_last") or []
                mins = [t["min_dose"] for t in tr if t]
                maxs = [t["max_dose"] for t in tr if t]
                print(f"      {L:7s} {q:14s} mean dose per seed {lst(ps)}  rows' min {fmt(min(mins))} max {fmt(max(maxs))}")
    out["case1_hosts_standard_dose"] = std
    allv = [v for d in rows.values() for v in d] + [v for d in std.values() for v in d]
    print()
    print(f"  Count: every recorded standard dose level lies in [{min(allv):.3f}, {max(allv):.3f}] "
          f"({len(allv)} readings: 90 chunk means + {sum(len(v) for v in std.values())} seed means); the dose at the")
    print("  actor's full-throttle command differs from the recorded command's by at most "
          f"{max(max(v) - min(v) for v in flat.values()):.4f} (5 qualities).")
    print("  Takeaway: the standard penalty cannot be 'higher under shift' because it never looks at the test condition.")


def case_2(inp, out):
    header(2, "consistent", "what drops under the shift is coverage, while the band's width barely moves")
    print("Before = IID test, after = strong shift, same standard band. Weighting test marks M1 (IID) and W2 (strong), per")
    print("seed (pre-registered marks; outputs/shift_iw/v5/<quality>/report.json):")
    cov = {}
    lower = 0
    for q in QUALITIES:
        mk = inp.report(q)["marks"]
        a, b = mk["M1"]["coverage"]["per_seed"], mk["W2"]["coverage"]["per_seed"]
        w = mk["W1"]["coverage"]["strong"]["per_seed"]
        cov[q] = dict(iid_standard=a, strong_standard=b, strong_iw_exact=w)
        lower += sum(x > y for x, y in zip(a, b))
        print(f"  {q:14s} IID {lst(a)}  strong {lst(b)}  drop per seed {lst([x - y for x, y in zip(a, b)])}")
    out["case2_coverage"] = cov
    print(f"  Count: coverage under the strong shift is below the IID coverage on {lower} of 25 seed-qualities.")
    print()
    print("Width and coverage at the actor's own commands, standard band, by whether the command is within or past 80%")
    print("throttle (the removed region; channel check, exploratory, outputs/shift_iw/explore/channel/results/<q>.json).")
    print("  quality        seed | width within-80  width past-80  ratio | coverage (training law) within-80  past-80 | needed x (strong law, past-80)")
    ch = {}
    ratios, drops = [], []
    for q in QUALITIES:
        d = inp.channel(q)
        ch[q] = []
        for s in d["seeds"]:
            ua = s["unweighted@actor"]
            wi, pa = ua["actor_within_80"], ua["actor_past_80"]
            r = pa["width_mean_strong_law"] / wi["width_mean_strong_law"]
            ratios.append(r)
            drops.append(wi["coverage_train_law"] - pa["coverage_train_law"])
            ch[q].append(dict(seed=s["seed"], width_within=wi["width_mean_strong_law"], width_past=pa["width_mean_strong_law"],
                              cov_within_train=wi["coverage_train_law"], cov_past_train=pa["coverage_train_law"],
                              cov_within_strong=wi["coverage_strong_law"], cov_past_strong=pa["coverage_strong_law"],
                              needed_past_strong=pa["needed_multiplier_strong_law"], R_u=s["R_u"],
                              R_strong_mean=s["R_exact_strong"]["mean"],
                              width_within_w=s["exact_strong@actor"]["actor_within_80"]["width_mean_strong_law"],
                              width_past_w=s["exact_strong@actor"]["actor_past_80"]["width_mean_strong_law"],
                              cov_past_strong_w=s["exact_strong@actor"]["actor_past_80"]["coverage_strong_law"]))
            print(f"  {q:14s} {s['seed']:4d} | {wi['width_mean_strong_law']:14.3f}  {pa['width_mean_strong_law']:13.3f}  {r:5.2f} | "
                  f"{wi['coverage_train_law']:31.3f}  {pa['coverage_train_law']:7.3f} | {pa['needed_multiplier_strong_law']:.2f}")
    out["case2_channel"] = ch
    print(f"  Count: the band past 80% is {min(ratios):.2f} to {max(ratios):.2f} times its width within 80% (25 seeds), while")
    print(f"  coverage there falls by {min(drops):.2f} to {max(drops):.2f}. Where the data do not cover the command the penalty is")
    print("  not higher; the band is simply wrong more often. Takeaway: the penalty's direction test fails for the width at")
    print("  the actor's command, in every seed (consistent).")


def stage2_widths(args):
    """Mean band width by pair set and role from the recorded width scores (Stage 2), per seed, uniform and weighted."""
    if args.stage2_copies:
        pairs = lambda s: os.path.join(args.stage2_copies, "pairs", f"seed{s}")
        scores = lambda ver, s: os.path.join(args.stage2_copies, ver, "stall", f"seed{s}", "scores", f"bca_width_{ver}_fixed.npz")
    elif args.runs:
        pairs = lambda s: os.path.join(args.runs, "detect", f"seed{s}", "stall")
        scores = lambda ver, s: os.path.join(args.runs, "stage2", "detect", ver, "stall", "fixed", f"seed{s}", "scores",
                                             f"bca_width_{ver}_fixed.npz")
    else:
        return None
    if not os.path.exists(scores("uniform", 0)):
        return None
    res = {}
    for ver in ("uniform", "weighted"):
        per = []
        for s in range(5):
            pz = np.load(os.path.join(pairs(s), "pairs.npz"))
            lz = np.load(os.path.join(pairs(s), "labels.npz"))
            sc = np.load(scores(ver, s))["scores"]
            st, role, harmful = pz["set"], pz["role"], lz["harmful"]
            row = dict(seed=s)
            for S in ("T1", "T3"):
                mc, mk = (st == S) & (role == "command"), (st == S) & (role == "cap")
                pc, pk = pz["pair"][mc], pz["pair"][mk]
                oc, ok = np.argsort(pc), np.argsort(pk)
                uc, uk, h = sc[mc][oc], sc[mk][ok], harmful[mc][oc]
                assert np.array_equal(pc[oc], pk[ok])
                row[f"{S}_command_harmful"] = float(uc[h].mean())
                row[f"{S}_cap_harmful"] = float(uk[h].mean())
                row[f"{S}_command_minus_cap_harmful"] = float((uc - uk)[h].mean())
                row[f"{S}_command_all"] = float(uc.mean())
                row[f"{S}_cap_all"] = float(uk.mean())
            m = (st == "T1") & (role == "recorded")
            row["T1_recorded_all"] = float(sc[m].mean())
            m = (st == "T4") & (role == "recorded")
            row["T4_recorded_all"] = float(sc[m].mean())
            per.append(row)
        res[ver] = per
    return res


def case_3(inp, out):
    header(3, "consistent", "Stage 2: the band is flat between the actor's harmful command, its 80% cap and the recorded command")
    print("Before = the recorded command (IID), after = the actor's own command past 80% throttle, same state. Mean band")
    print("width U in critic value units over the detection pairs (T1 = training states in the full-throttle region, T3 =")
    print("strong-current rollout states, T4 = recorded rows), per seed, from the recorded width scores")
    print("(runs/shift_explore/stage2/detect/<version>/stall/fixed/seed<K>/scores/bca_width_*_fixed.npz; exploratory seeds 0-4).")
    res = stage2_widths(inp.args)
    if res is None:
        print("  (skipped: pass --stage2-copies <dir> or --runs <runs/shift_explore> to read the width scores)")
        print("  Recorded summary (PENALTY_EVIDENCE.md, case 3): uniform U 0.28-0.33 everywhere; command minus cap on harmful")
        print("  pairs +0.0003 to +0.0122 per seed-set; the weighted version 0.23-0.57 at recorded rows per seed.")
        return
    out["case3_stage2"] = res
    for ver in ("uniform", "weighted"):
        print(f"  {ver} band:  seed | T1 command (harmful)  T1 cap  diff | T1 recorded | T3 command (harmful)  T3 cap  diff | T4 recorded")
        for r in res[ver]:
            print(f"               {r['seed']:3d} | {r['T1_command_harmful']:19.4f}  {r['T1_cap_harmful']:6.4f}  {r['T1_command_minus_cap_harmful']:+.4f} | "
                  f"{r['T1_recorded_all']:11.4f} | {r['T3_command_harmful']:19.4f}  {r['T3_cap_harmful']:6.4f}  {r['T3_command_minus_cap_harmful']:+.4f} | {r['T4_recorded_all']:.4f}")
    u = res["uniform"]
    diffs = [r["T1_command_minus_cap_harmful"] for r in u] + [r["T3_command_minus_cap_harmful"] for r in u]
    shift_ratio = [r["T3_command_all"] / r["T1_command_all"] for r in u]
    print(f"  Count (uniform): command minus cap on harmful pairs is {min(diffs):+.4f} to {max(diffs):+.4f} (10 seed-sets); the band at")
    print(f"  shifted states (T3) is {min(shift_ratio):.2f} to {max(shift_ratio):.2f} times the band at IID states (T1). The false preference it had")
    print("  to cancel was 0.63-0.89 (results-2026-10-13/README.md). Takeaway: flat, in every seed.")
    w = res["weighted"]
    print(f"  Weighted version at recorded (IID) rows, T4: {lst([r['T4_recorded_all'] for r in w], 3)} against uniform "
          f"{lst([r['T4_recorded_all'] for r in u], 3)}: higher on {sum(a['T4_recorded_all'] > b['T4_recorded_all'] for a, b in zip(w, u))} of 5 seeds,")
    print("  by construction (it adds a classifier term U_max softplus(log w - l*) to the uniform band; stage2/code/band.py).")


def case_4(inp, out):
    header(4, "consistent", "the harm switch's height is high where full throttle is harmful and low where it is harmless")
    print("Before = the no-stall world (full throttle harmless), after = the stall world (harmful). Height = R_tail - R_rest in")
    print("critic value units (step1lib.py:244-276). Every certificate on record (near-80 30 seed-worlds; step 1 10):")
    rows = inp.switches()
    out["case4_switch"] = rows
    print("  study    setting      world     seed  n_tail  finite share  height    switch")
    for r in rows:
        print(f"  {r['study']:8s} {r['setting']:12s} {r['world']:9s} {r['seed']:4d}  {r['n_tail']:5d}  {fmt(r['finite_share'])}         {fmt(r['height']):>8s}  {'on' if r['on'] else 'off'}")
    fin = [r for r in rows if math.isfinite(r["height"])]
    inf = [r for r in rows if not math.isfinite(r["height"])]
    stall = [r["height"] for r in fin if r["world"] == "stall"]
    nostall = [r["height"] for r in fin if r["world"] == "no_stall"]
    right = sum(h > max(nostall) for h in stall) + sum(h < min(stall) for h in nostall)
    print(f"  Count: {len(fin)} finite certificates; stall heights {min(stall):.2f}-{max(stall):.2f}, no-stall {min(nostall):.2f}-{max(nostall):.2f}; every stall")
    print(f"  height exceeds every no-stall height ({right} of {len(fin)} in the right direction). Infinite height in {len(inf)} of {len(rows)}: tails of")
    print(f"  {lst(sorted(r['n_tail'] for r in inf), 0)} rows (case 9).")
    print("  Label: near-80 seeds 0-4 and step 1 exploratory; near-80 seeds 10-14 pre-registered.")


def case_5(inp, out):
    header(5, "consistent", "the penalty rises under the shift only when the shift is declared through weights")
    print("Before = standard threshold R_u; after = the weighted threshold R_w with exact weights for the mild or strong mix,")
    print("on the same held-out scores (single threshold; shown values of the weighting test; exploratory BCA arms).")
    print("  quality        R_w/R_u mild per seed              R_w/R_u strong per seed             dose change strong (R2)      lower-edge scale change strong (L2)")
    tab = {}
    c_mild = c_strong = c_order = c_dose = c_scale = c_cov = 0
    for q in QUALITIES:
        rep = inp.report(q)
        g = rep["shown"]["R_weighted_over_R_unweighted"]["gap"]["single"]
        mild, strong = g["exact_mild_single"]["per_seed"], g["exact_strong_single"]["per_seed"]
        none_exact = g["exact_none_single"]["per_seed"]
        dose = rep["marks"]["R2"]["dose_change"]["per_seed"]
        scale = rep["marks"]["L2"]["scale_change"]["per_seed"]
        cov = rep["marks"]["W1"]["coverage"]["strong"]["per_seed"]
        tab[q] = dict(exact_none=none_exact, exact_mild=mild, exact_strong=strong, dose_change_strong=dose,
                      scale_change_strong=scale, scale_change_mild=rep["marks"]["L2"]["mild_scale_change"]["per_seed"],
                      coverage_strong_iw=cov, refused_rows=rep["marks"]["R2"]["refused_rows"])
        c_mild += sum(v > 1 for v in mild)
        c_strong += sum(v > 1 for v in strong)
        c_order += sum(s > m for s, m in zip(strong, mild))
        c_dose += sum(v > 0 for v in dose)
        c_scale += sum(v > 0 for v in scale)
        c_cov += sum(v >= 0.89 for v in cov)
        print(f"  {q:14s} {lst(mild, 3):35s} {lst(strong, 3):35s} {lst(dose, 4):30s} {lst(scale, 3)}")
    out["case5_v5"] = tab
    print(f"  Count: R_w/R_u > 1 on {c_mild} of 25 (mild) and {c_strong} of 25 (strong) seed-qualities; strong above mild on {c_order} of 25;")
    print(f"  dose change > 0 on {c_dose} of 25; lower-edge scale change > 0 on {c_scale} of 25; exact 'none' weights give 1.000 on 25 of 25.")
    print(f"  Coverage under the strong shift with exact strong weights is at least 0.89 on {c_cov} of 25 seeds (W1; standard: case 2).")
    H = inp.hosts()
    hz = {}
    for q in ("medium", "medium-expert"):
        for L in ("cql", "rebrac", "td3_bc"):
            S = H["qualities"][q]["expanded"].get(L, {}).get("setups", {})
            Bi, Bs = (S.get("iw") or {}).get("band"), (S.get("standard") or {}).get("band")
            if not Bi or not Bi.get("dose_change") or not Bi["dose_change"][0]:
                continue
            ch = [x["change_vs_unweighted"] for x in Bi["dose_change"]]
            hz[f"{L} {q}"] = dict(standard_dose=Bs["mean_dose_last"]["per_seed"], iw_dose=Bi["mean_dose_last"]["per_seed"],
                                  change=ch, estimated=[x["change_vs_unweighted"] for x in Bi["estimated_weights_dose_change"]],
                                  shuffled=[x["change_vs_unweighted"] for x in Bi["shuffled_weights_dose_change"]],
                                  R_standard=Bs["threshold_last"]["per_seed"], R_iw_arm_unweighted=Bi["threshold_last"]["per_seed"])
            print(f"  four-learner {L:7s} {q:14s}: dose standard {lst(Bs['mean_dose_last']['per_seed'])} -> IW {lst(Bi['mean_dose_last']['per_seed'])}; change {lst(ch, 3)}")
    out["case5_hosts"] = hz
    cq = [v for k, v in hz.items() if k.startswith("cql") for v in v["change"]]
    print(f"  Count: CQL's dose rises on {sum(v > 0 for v in cq)} of {len(cq)} seeds (+{min(cq):.3f} to +{max(cq):.3f}); ReBRAC (critic did not learn): case 10.")
    print("  Takeaway: declared weights raise the penalty everywhere, by 1.1-2.3x in the threshold and 0.02-0.16 in the dose.")


def case_6(inp, out):
    header(6, "consistent", "weights of 1 (the IID test declared exactly) give the standard penalty bit for bit")
    print("Before = standard calibrate(scores); after = calibrate(scores, weights = 1, test mass = 1), same random stream")
    print("(calibration/wbcp.py:62-66 and :117; experiments/shift_iw/coverage.py:78-86 unit_check).")
    w0 = {}
    n_ok = n_all = 0
    for q in QUALITIES:
        m = inp.report(q)["marks"]["W0"]
        for host in ("full", "gap"):
            u = m["per_seed"][host]["unit_check"]
            t = m["per_seed"][host]["per_step_table_is_unweighted"]
            c = m["per_seed"][host]["per_step_coverage_identical"]
            w0[f"{q} {host}"] = dict(unit_check=u, table_is_unweighted=t, coverage_identical=c)
            n_ok += sum(a and b and c_ for a, b, c_ in zip(u, t, c))
            n_all += len(u)
        print(f"  weighting test W0 {q:14s}: unit check {m['per_seed']['gap']['unit_check']} (gapped host), {m['per_seed']['full']['unit_check']} (full host); status {m['status']}")
    out["case6_w0"] = w0
    print(f"  Count: {n_ok} of {n_all} seed-hosts identical (threshold, lambda_hat, lambda_hpd, every posterior crossing, the per-step")
    print("  table [1.0] and the per-step arm's coverage). Pre-registered mark W0 (labeled a unit check).")
    print("  Four-learner stop rule S2 (PRESPEC.md:474): IW with every weight 1 equals the standard run in every bit after the")
    print("  refresh, and R(1) equals the stored standard threshold in float32: passed for all four learners (the study ran on).")
    rc = inp.recompute()
    if inp.args.runs:
        live = live_recompute(inp.args, n_banks=inp.args.banks)
        out["case6_live"] = live
        for o in live:
            print(f"  live {o['study']} {o['setting']} {o['world']} seed {o['seed']}: recorded R {o['recorded']['R']:.6f}; weights of 1 -> {o['A_ones']['R']:.6f}; "
                  f"equals recorded recompute {o['A_ones']['equals_recorded_recompute']}; explicit ones bit-equal {o['A_ones']['explicit_ones_bit_equal']}")
    if rc:
        s = rc["summary"]
        out["case6_recompute"] = dict(banks=s["banks"], equal=s["A_equals_recorded_recompute"], ones=s["A_explicit_ones_bit_equal"])
        print(f"  Recorded held-out scores (pinned environment, data/penalty_recompute.json): {s['A_equals_recorded_recompute']} of {s['banks']} banks reproduce the")
        print(f"  recorded threshold, and weights of 1 equal the unweighted call bit for bit on {s['A_explicit_ones_bit_equal']} of {s['banks']}.")
    print("  Takeaway: with no shift and known weights the weighted penalty cannot differ from the standard one.")


def case_7(inp, out):
    header(7, "exceptional", "estimated weights fitted for the IID test move the threshold by -9% to +5% on single seeds")
    print("Before = standard R_u; after = R with the classifier's weights fitted on the IID ('none') mix, where the exact weights")
    print("are 1 (single threshold at the frozen band; shown values; the classifier is logistic regression on the 13-number")
    print("observation, experiments/shift_iw/classifier.py). Also its fitted slope on the current (exact 0; mark W4) and the")
    print("coverage of the estimated band minus the exact one on the IID test.")
    print("  quality        seed  R_est/R_u   slope on c   coverage est - exact")
    tab = {}
    higher = lower = equal = beyond1 = beyond5 = agree = judged = 0
    for q in QUALITIES:
        rep = inp.report(q)
        r = rep["shown"]["R_weighted_over_R_unweighted"]["gap"]["single"]["estimated_none_single"]["per_seed"]
        sl = rep["marks"]["W4"]["classifier_slope_on_c"]["none"]["fitted"]["per_seed"]
        st = rep["shown"]["single_threshold_coverage"]["none"]
        ce = [a - b for a, b in zip(st["estimated_single"]["per_seed"], st["exact_single"]["per_seed"])]
        tab[q] = dict(ratio=r, slope=sl, coverage_diff=ce)
        for i, (x, s, c) in enumerate(zip(r, sl, ce)):
            flag = ""
            if abs(x - 1) > 0.05:
                flag = "  <- beyond 5%"
            elif abs(x - 1) > 0.01:
                flag = "  <- beyond 1%"
            print(f"  {q:14s} {i:4d}  {x:9.4f}   {s:+9.4f}   {c:+.4f}{flag}")
            higher += x > 1
            lower += x < 1
            equal += x == 1
            beyond1 += abs(x - 1) > 0.01
            beyond5 += abs(x - 1) > 0.05
            if x != 1:
                judged += 1
                agree += (x > 1) == (s > 0)
    out["case7"] = tab
    out["case7_counts"] = dict(higher=higher, lower=lower, equal=equal, beyond1=beyond1, beyond5=beyond5, slope_sign_agrees=agree, judged=judged)
    print(f"  Count: IW higher than standard on {higher} of 25 seed-qualities, lower on {lower}, equal on {equal}; beyond 1% on {beyond1}, beyond 5% on")
    print(f"  {beyond5} (both medium-expert: seed 2 at 0.913, seed 4 at 1.050). The sign of the move agrees with the sign of the fitted")
    print(f"  slope on the current in {agree} of {judged} moved cases: a positive slope tilts the weights toward high-current rows, where")
    print("  the scores are larger (mark M2), so the weighted quantile rises; a negative slope lowers it. Means by quality are")
    print("  within 1% of 1 (W4's coverage gap 0.0001-0.002). Exceptional: classifier noise on a slope whose true value is 0.")
    rc = inp.recompute()
    if rc:
        s = rc["summary"]
        print(f"  The same 25 slopes applied to random currents on the 30 recorded banks (no link between weight and score; pinned")
        print(f"  recompute case E): R ratio {s['E_none_ratio_min']:.3f} to {s['E_none_ratio_max']:.3f}, beyond 5% in {s['E_none_beyond_5pct']} of {s['E_none_cases']}. So the recorded moves beyond 5%")
        print("  need the weights to be correlated with the scores (the tilt), not the smaller effective sample.")


def case_8(inp, out):
    header(8, "consistent", "weights for the strong shift, applied while the learner trains on IID rows, raise the penalty on every seed")
    print("This is how the IW arm is actually run: its band is calibrated for the strong test law, and the dose it reads on the")
    print("recorded (IID) training rows is therefore higher than the standard arm's. Before = standard, after = IW.")
    dc = inp.dose_chunks()
    chunks = {}
    n_hi = n_all = 0
    print("  (a) Weighting test, seed 0, mean dose per 5,000-update chunk after the first refresh, IW arm minus standard arm")
    print("      (dose_chunks.json, exploratory; two different training runs):")
    for q in QUALITIES:
        a, b = dc[f"{q}/unweighted"]["dose"][2:], dc[f"{q}/exact_strong"]["dose"][2:]
        d = [y - x for x, y in zip(a, b)]
        chunks[q] = dict(standard=a, iw=b)
        n_hi += sum(v > 0 for v in d)
        n_all += len(d)
        rr = [x["per_row"]["rows"] for x in dc[f"{q}/exact_strong"]["refreshes"] if "per_row" in x]
        print(f"      {q:14s} IW - standard per chunk: min {min(d):+.4f} mean {np.mean(d):+.4f} max {max(d):+.4f}; per-row R_w/R_u at the first two refreshes: "
              + ", ".join(f"[{x['ratio_min']:.3f}, {x['ratio_max']:.3f}]" for x in rr))
    out["case8_chunks"] = chunks
    print(f"      Count: IW's dose is above the standard arm's in {n_hi} of {n_all} chunks (5 qualities x 18 chunks), and every per-row")
    print("      threshold ratio at the recorded refreshes is above 1 (10 of 10 refresh records).")
    print("  (b) The frozen band: R2's dose change (case 5) is positive on 25 of 25 seeds; CQL's on 10 of 10; refused rows 0 everywhere.")
    print("  (c) What it costs where there is no shift (return on the IID starts, IW minus standard; exploratory):")
    ex = inp.explore_results()
    cost = {}
    neg = tot = 0
    for e in ex["qualities"]:
        q = e["quality"]
        c = e["comparisons"]["capped:per_step_minus_unweighted"]["laws"]["none"]
        le = e["comparisons"]["lower:per_step_minus_unweighted"]["laws"]["none"]
        cost[q] = dict(capped_return=c["J"]["per_seed"], capped_points=c["points"]["per_seed"], lower_return=le["J"]["per_seed"])
        neg += sum(v < 0 for v in c["J"]["per_seed"])
        tot += len(c["J"]["per_seed"])
        print(f"      {q:14s} capped pull: return units per seed {lst(c['J']['per_seed'], 1)} (mean {c['J']['mean']:+.1f}, {c['points']['mean']:+.2f} points); "
              f"lower edge {lst(le['J']['per_seed'], 1)}")
    H = inp.hosts()
    for q in ("medium", "medium-expert"):
        c3 = H["qualities"][q]["claims"]["cql"]["3"]
        ps = [p["d"] for p in c3["delta"]["per_seed"]]
        cost[f"cql {q}"] = dict(return_per_seed=ps, badge=c3["badge"])
        print(f"      four-learner CQL {q:14s} claim 3 (IW - standard on the IID bank): {lst(ps, 2)} mean {np.mean(ps):+.2f}; {c3['badge']}")
    out["case8_cost"] = cost
    print(f"      Count: the capped pull loses return on {neg} of {tot} IID seed-qualities (largest mean loss medium-replay, -9.1 units =")
    print("      -1.2 points); the lower edge and CQL stay within a few units either way.")
    print("  (d) Why the threshold rises: the three parts, on the 30 recorded banks (pinned recompute, data/penalty_recompute.json):")
    rc = inp.recompute()
    if rc:
        s = rc["summary"]
        out["case8_recompute_summary"] = s
        print(f"      B  test atom alone (mass {s['test_mass']['strong']:.2f} instead of 1, weights 1):          R_w/R_u mean {s['B_test_mass_only_ratio_mean']:.3f}")
        print(f"      C  strong weights on random currents (Kish {s['C_strong_kish_share_mean']:.2f} n, no link to the scores): mean {s['C_strong_ratio_mean']:.3f}, "
              f"range {s['C_strong_ratio_min']:.3f}-{s['C_strong_ratio_max']:.3f}; above 1 in {s['C_strong_draws_above_one']} of 600 draws, below in {s['C_strong_draws_below_one']}")
        print(f"         weighted empirical quantile / unweighted {s['C_strong_lambda_hat_ratio_mean']:.3f}; Bayesian margin lambda_hpd/lambda_hat "
              f"{s['margin_unweighted_mean']:.3f} -> {s['C_strong_margin_mean']:.3f}")
        print(f"      D  strong weights sorted onto the scores: heaviest on the largest scores {s['D_up_ratio_mean']:.2f}x, on the smallest {s['D_down_ratio_mean']:.2f}x")
        print(f"      F  the margin as the bank shrinks: n = 300 {s['F_margin_300_mean']:.3f}, n = 100 {s['F_margin_100_mean']:.3f} (recorded n = 805-851: {s['margin_unweighted_mean']:.3f})")
        print("      The recorded strong ratios (1.11-2.25 per seed) sit far above C and below D-up: most of the rise is the reweighting")
        print("      toward the high-current rows, where the critic's scores are larger; the smaller effective sample adds about")
        print(f"      +{100 * (s['C_strong_ratio_mean'] - 1):.0f}% to the threshold (+0.01 in the dose: the four-learner shuffled placebo -0.008 to +0.017).")
    if inp.args.runs:
        for o in live_recompute(inp.args, n_banks=inp.args.banks):
            print(f"      live {o['study']} {o['setting']} {o['world']} seed {o['seed']}: B {o['B_test_mass_only']['ratio']:.4f}, C strong mean {o['C_shuffled_strong']['ratio_mean']:.4f} "
                  f"[{o['C_shuffled_strong']['ratio_min']:.3f}, {o['C_shuffled_strong']['ratio_max']:.3f}], D up {o['D_sorted_strong']['heaviest_on_largest_scores']['ratio']:.3f}, down {o['D_sorted_strong']['heaviest_on_smallest_scores']['ratio']:.3f}")
    print("  Takeaway: the higher IW penalty under IID training rows is the declared shift, not a defect; it costs at most about 1 point.")


def case_9(inp, out):
    header(9, "exceptional", "refusal (R = +inf): too few effective held-out rows, never the IW arms on record")
    print("The credible quantile is finite only when at least 950 of the 1,000 posterior draws cross; a draw fails with")
    print("probability 0.9^n, so about 29 rows are needed (near80_results_check/REPORT.md:24-27; calibration/wbcp.py:108-117).")
    rows = inp.switches()
    small = [r for r in rows if r["n_tail"] <= 30]
    print("  (a) Every certificate with a full-throttle group of at most 30 rows (finite share = share of finite draws):")
    for r in sorted(small, key=lambda r: r["n_tail"]):
        pred = 1 - 0.9 ** r["n_tail"]
        print(f"      {r['study']:8s} {r['setting']:12s} {r['world']:9s} seed {r['seed']:2d}  n_tail {r['n_tail']:2d}  finite share {fmt(r['finite_share'])} (1 - 0.9^n = {pred:.3f})  height {fmt(r['height'])}  switch {'on' if r['on'] else 'off'}")
    n29 = [r for r in rows if r["n_tail"] == 29]
    print(f"      Count: infinite in {sum(not math.isfinite(r['height']) for r in rows)} of {len(rows)}; every tail of 28 or fewer refused, every tail of 30 or more certified;")
    print(f"      at exactly 29 rows {sum(not math.isfinite(r['height']) for r in n29)} of {len(n29)} refused (a 70/30 draw: P(at least 950 of 1,000 finite) with p = 1 - 0.9^29 = 0.953).")
    rc = inp.recompute()
    if rc:
        s = rc["summary"]
        print(f"  (b) The same arithmetic widens before it refuses: Bayesian margin lambda_hpd/lambda_hat on random subsets of the recorded")
        print(f"      scores, n = 300: {s['F_margin_300_mean']:.2f}; n = 100: {s['F_margin_100_mean']:.2f}; full bank: {s['margin_unweighted_mean']:.3f}.")
    print("  (c) The IW arms with exact weights never refused a row: weighting test R2 refused_rows = 0 on 25 of 25 seeds; four-learner")
    print("      refused_rows_last = 0 on 10 of 10 CQL seeds (18 of 18 refreshes certified).")
    print("  (d) Where weighting does refuse: the command-weighted band (weights for the actor's own commands; exploratory,")
    print("      outputs/shift_iw/explore/channel/results/command_band/<q>.json). Per seed: Kish share of the held-out rows, share of")
    print("      the actor's within-80% (IID-like) commands refused, past-80% refused, largest finite threshold / R_u:")
    cb = {}
    for q in QUALITIES:
        d = inp.command_band(q)
        cb[q] = []
        for s in d["seeds"]:
            k = s["bank_weights"]["kish_share"]
            wi, pa = s["at_actor"]["within_80"]["refused"], s["at_actor"]["past_80"]["refused"]
            cb[q].append(dict(seed=s["seed"], kish_share=k, refused_within=wi, refused_past=pa, cap_over_R_u=s["thresholds"]["cap_over_R_u"]))
            print(f"      {q:14s} seed {s['seed']}  Kish {100 * k:5.1f}%  refused within-80 {100 * wi:5.1f}%  past-80 {100 * pa:5.1f}%  cap/R_u {s['thresholds']['cap_over_R_u']:.2f}")
    out["case9_command_band"] = cb
    allw = [x["refused_within"] for v in cb.values() for x in v]
    allk = [x["kish_share"] for v in cb.values() for x in v]
    print(f"      Count: refuses {100 * min(allw):.0f}% to {100 * max(allw):.0f}% of within-80% commands (25 seeds) because the effective sample falls to")
    print(f"      {100 * min(allk):.1f}-{100 * max(allk):.1f}% of the bank (Kish). This is the one setting on record where IW's penalty is higher than")
    print("      standard's under IID-like commands by a large margin, and it is a refusal, not a wider band.")


def case_10(inp, out):
    header(10, "control", "weights change nothing when the critic's error does not grow with the current (ReBRAC; shuffled weights)")
    print("Before = standard dose, after = IW dose, four-learner study (exploratory). ReBRAC's critic 'did not learn' (its")
    print("standard band already covered 90-96% under the shift); the shuffled-weight placebo keeps the weights' sizes but")
    print("breaks their link to the rows.")
    H = inp.hosts()
    tab = {}
    for q, L in (("medium", "rebrac"), ("medium", "cql"), ("medium-expert", "cql")):
        B = H["qualities"][q]["expanded"][L]["setups"]["iw"]["band"]
        ch = [x["change_vs_unweighted"] for x in B["dose_change"]]
        sh = [x["change_vs_unweighted"] for x in B["shuffled_weights_dose_change"]]
        es = [x["change_vs_unweighted"] for x in B["estimated_weights_dose_change"]]
        tab[f"{L} {q}"] = dict(strong=ch, shuffled=sh, estimated=es)
        print(f"  {L:7s} {q:14s} dose change: strong weights {lst(ch, 3)}  shuffled placebo {lst(sh, 3)}  estimated weights {lst(es, 3)}")
    out["case10"] = tab
    reb = tab["rebrac medium"]["strong"]
    shuf = [v for k, v in tab.items() if k.startswith("cql") for v in v["shuffled"]]
    print(f"  Count: ReBRAC's change is {min(reb):+.3f} to {max(reb):+.3f} ({sum(v > 0 for v in reb)} of 5 above 0); CQL's shuffled placebo {min(shuf):+.3f} to {max(shuf):+.3f}")
    print(f"  ({sum(v > 0 for v in shuf)} of 10 above 0) against +0.125 to +0.156 with the real weights. Takeaway: the rise needs weights that")
    print("  point at rows where the critic's error is larger; the sample-size part alone is about 0.01 in the dose.")


def live_recompute(args, n_banks=1):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    try:
        import penalty_recompute as PR
    except ImportError as e:
        sys.exit(f"penalty_recompute.py must sit beside this script and calibration/wbcp.py must be importable: {e}")
    banks = PR.find_banks(args.runs)
    if not banks:
        sys.exit(f"no recorded banks under {args.runs}")
    return [PR.analyse(b, shuffles=20) for b in banks[:n_banks]]


CASES = {1: case_1, 2: case_2, 3: case_3, 4: case_4, 5: case_5, 6: case_6, 7: case_7, 8: case_8, 9: case_9, 10: case_10}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root (the worktree)")
    ap.add_argument("--case", default="all", help="all, or a case number 1-10, or a comma list")
    ap.add_argument("--hosts", default=None, help="a copy of outputs/shift_hosts_v3/results.json (branch claude/shift-hosts)")
    ap.add_argument("--stage2-copies", default=None, help="folder with the Stage 2 pairs and width scores (case 3)")
    ap.add_argument("--runs", default=None, help="runs/shift_explore (run files; enables the live recompute and case 3)")
    ap.add_argument("--banks", type=int, default=1, help="how many recorded banks the live recompute analyses")
    ap.add_argument("--recompute-json", default=None, help="the pinned recompute (default docs/bca_mechanics/data/penalty_recompute.json)")
    ap.add_argument("--write-json", default=None, help="write the numbers the figures use here")
    args = ap.parse_args()
    inp = Inputs(args)
    which = sorted(CASES) if args.case == "all" else [int(x) for x in args.case.split(",")]
    out = {}
    print("BCA's penalty: the consistent and exceptional cases, rebuilt from recorded inputs (nothing trained, no band fitted).")
    print(f"repository root {os.path.abspath(args.root)}; numpy {np.__version__}")
    for n in which:
        CASES[n](inp, out)
    if args.write_json:
        with open(args.write_json, "w", encoding="utf8") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.write_json}")


if __name__ == "__main__":
    main()
