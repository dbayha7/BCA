"""EXPLORATORY: arms A, B, C and F0 -- scoring (frozen pass marks, section 8 step 6), only after every kill rule has
been checked on all five seeds (KILLS.json complete).

  python -B score.py eval --arm {A,B,C,F} --quality Q --seed K
      Exact float64 100-step returns (points beside them) of every surviving actor and ceiling on the three start
      banks' 1,000 fixed starts (learners_iw.evaluate_actor's arithmetic; a ceiling is clip(pi(x), -t_k, t_k) inside
      the policy), and the main test's four situations for the primary arms (A1, none, host + BCA, host, rho 0.5 + BCA,
      rho 1, A2, F0) as /scratchpad/sit4/situations.py computes them (information only). Writes
      <sandbox>/repo/runs/shift_iw_v5_explore/arms_abc_f0/eval/<arm>/<Q>/seed<K>/record.json (+ .sha256); an arm or
      seed a kill rule stopped gets a 'skipped' record (stopped returns never enter a mark).
  python -B score.py marks
      Every mark of sections 4-7 from the evaluation records, F0's records and A2's evaluations; writes
      routes2/arms/results.json (+ .sha256) and RESULTS.md with the wording fixed in the frozen file, every stop line
      ("Stopped by <rule> on <quality, seed>: <reason>."), and after the marks the information outputs (section 4.3,
      5.3, 6.3, 7.4 and the main test's four situations for the primary arms). Refuses until every evaluation record
      the scheduler runs exists; records the SHA-256 of every file it read (rt.py done: current only while unchanged
      and while KILLS.json is the one it read). K3 counts the kill checks' seeds and the seeds whose returns came out
      non-finite together (two stop B, or C). An arm stopped as a whole gets no outcome line, only its stop lines.
Refuses when the frozen file differs from its sidecar, and refuses records without the label or with another
passmarks_sha256 (common.arm_record).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402

LAWS = ("none", "mild", "strong")
T4 = 2.132                       # one-sided 95% t quantile, 4 degrees of freedom
PLAIN_A = ("plain_1s", "plain_2s", "plain_4s")
PLAIN_B = ("wmedian", "wmean", "sign", "wttest", "wbernstein")


def eval_path(arm, q, k):
    return C.out_root() / "eval" / arm / q / f"seed{int(k)}" / "record.json"


def kills(digest):
    K = C.arm_record(C.docs() / "KILLS.json", digest)
    if K.get("complete") is not True:
        raise C.Refused(f"refused: the kill checks are not complete ({K.get('missing')}); no return is computed")
    return K


# ------------------------------------------------------------------------------------------------ eval

def eval_job(arm, quality, seed):
    from experiments.shift import learners as LR
    digest = C.require_passmarks()
    K = kills(digest)
    started = time.perf_counter()
    rec = C.base_record(f"eval_{arm}", quality, seed, digest)
    rec["kills_sha256"] = C.sha256_file(C.docs() / "KILLS.json")
    out = eval_path(arm, quality, seed)

    def skip(reason):
        rec.update(status="skipped", stopped_by=reason, reason=f"stopped by {reason}: no return enters a mark",
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        C.write_record(out, rec)
        C.log(f"eval {arm} {quality} seed {seed}: skipped ({reason})")
        return 0

    # a stopped arm, quality or seed is skipped before any input is loaded (its host may have failed for good)
    if arm == "A":
        ka = K["A"].get(quality) or {}
        if not ka.get("go_on") or seed not in ka.get("seeds_usable", []):
            return skip(ka.get("stopped_by") or f"seed {seed} not usable")
    elif arm == "B":
        kb = K["B"]
        brec = C.arm_record(C.arm_dir("B", quality, seed) / "record.json")
        if not kb.get("go_on") or brec["status"] != "finished" or (brec["kills"].get("K3") or {}).get("fires"):
            return skip(", ".join(kb.get("stopped_by") or []) or brec.get("stopped_by") or "K3")
    elif arm == "C":
        kc = K["C"].get(quality) or {}
        if not kc.get("go_on") or seed not in kc.get("seeds_usable", []):
            return skip(kc.get("stopped_by") or f"seed {seed} not usable")
    elif arm == "F":
        kf = K["F0"].get(quality) or {}
        f0 = C.read_record(C.arm_dir("F0", quality, seed) / "record.json")
        if f0 is None or f0.get("status") != "finished":
            return skip((f0 or {}).get("stopped_by") or kf.get("stopped_by") or "no finished F0 record")
    else:
        raise C.Refused(f"refused: unknown arm {arm!r}")

    S = C.Seed(quality, seed)
    sb, sa = C.situation_banks()
    host_pol = C.actor_policy(S.host.actor.params, S.prep, name=f"{quality} seed {seed} host")
    ev, sit = {}, {}
    if arm == "A":
        import arm_a
        arec, actors = arm_a.load_actors(quality, seed)
        for name in arm_a.ARMS:
            ev[name] = C.evaluate_policy(C.actor_policy(actors[name], S.prep), S.banks, S.anchors)
        for name in ("A1", "none"):
            sit[name] = C.evaluate_situations(C.actor_policy(actors[name], S.prep), sb, sa)
        ev["host"] = C.evaluate_policy(host_pol, S.banks, S.anchors)
        rec["arm_record_sha256"] = C.sha256_file(C.arm_dir("A", quality, seed) / "record.json")
    elif arm == "B":
        ev["host"] = C.evaluate_policy(host_pol, S.banks, S.anchors)
        for name, t in brec["maps"].items():
            ev[f"host+{name}"] = C.evaluate_policy(C.ceiling_policy(host_pol, t), S.banks, S.anchors)
        sit["host"] = C.evaluate_situations(host_pol, sb, sa)
        sit["host+bca"] = C.evaluate_situations(C.ceiling_policy(host_pol, brec["maps"]["bca"]), sb, sa)
        rec["arm_record_sha256"] = C.sha256_file(C.arm_dir("B", quality, seed) / "record.json")
    elif arm == "C":
        import arm_c
        crec, actors = arm_c.load_actors(quality, seed)
        p1 = C.actor_policy(actors["rho1"], S.prep, name="rho 1")
        p05 = C.actor_policy(actors["rho0.5"], S.prep, name="rho 0.5")
        ev["rho1"] = C.evaluate_policy(p1, S.banks, S.anchors)
        ev["rho0.5"] = C.evaluate_policy(p05, S.banks, S.anchors)
        for name in ("bca",) + PLAIN_B + ("range_limit", "shuffled", "matched_global", "knee_clip"):
            ev[f"rho0.5+{name}"] = C.evaluate_policy(C.ceiling_policy(p05, crec["maps"][name]), S.banks, S.anchors)
        ev["rho1+bca_built_for_rho1"] = C.evaluate_policy(C.ceiling_policy(p1, crec["rho1_bca_map"]), S.banks,
                                                          S.anchors)
        b = C.read_record(C.arm_dir("B", quality, seed) / "record.json")
        if b is not None and b.get("status") == "finished":
            ev["rho0.5+B_host_map"] = C.evaluate_policy(C.ceiling_policy(p05, b["maps"]["bca"]), S.banks, S.anchors)
        ev["host"] = C.evaluate_policy(host_pol, S.banks, S.anchors)
        sit["rho0.5+bca"] = C.evaluate_situations(C.ceiling_policy(p05, crec["maps"]["bca"]), sb, sa)
        sit["rho1"] = C.evaluate_situations(p1, sb, sa)
        rec["arm_record_sha256"] = C.sha256_file(C.arm_dir("C", quality, seed) / "record.json")
    else:
        import arm_f0
        a2_dir = arm_f0.F1.arm_dir(quality, "A2_cmd_strong", seed)
        a2 = json.loads((a2_dir / "record.json").read_text())
        a2p = LR.load_tree(a2_dir / "actor.msgpack", LR.actor_template(), a2["actor_sha256"])
        f0p = LR.load_tree(C.arm_dir("F0", quality, seed) / "actor.msgpack", LR.actor_template(), f0["actor_sha256"])
        sit["A2"] = C.evaluate_situations(C.actor_policy(a2p, S.prep), sb, sa)
        sit["F0"] = C.evaluate_situations(C.actor_policy(f0p, S.prep), sb, sa)
    rec.update(status="finished", evaluations=ev, situations=sit, seconds=time.perf_counter() - started,
               situation_anchors={s: v["anchor"] for s, v in sa.items()},
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    C.write_record(out, rec)
    C.log(f"eval {arm} {quality} seed {seed}: {len(ev)} policies on 3 banks, {len(sit)} in the four situations; "
          f"{rec['seconds']:.0f}s -> {out}")
    return 0


# ------------------------------------------------------------------------------------------------ marks

def summary(d):
    """A mark's numbers from per-seed differences d_k (None where missing): mean (None if any is missing), positives,
    and the one-sided 95% lower bound (reported, never deciding)."""
    vals = [None if v is None else float(v) for v in d]
    out = dict(per_seed=vals, missing=sum(1 for v in vals if v is None))
    if out["missing"] or len(vals) != 5:
        out.update(mean=None, positives=None, lower_bound=None)
        return out
    a = np.asarray(vals)
    sd = float(a.std(ddof=1))
    out.update(mean=float(a.mean()), positives=int(np.sum(a > 0)), negatives=int(np.sum(a < 0)),
               lower_bound=float(a.mean() - T4 * sd / math.sqrt(5)), sd=sd)
    return out


def at_least(s, bar, positives=None):
    if s["mean"] is None:
        return False
    ok = s["mean"] >= bar
    if positives is not None:
        ok = ok and s["positives"] >= positives
    return bool(ok)


class Points:
    """Per (arm, quality, seed): {policy: {law: points}} from the evaluation records."""

    def __init__(self, digest, K):
        self.digest = digest
        self.K = K
        self.cache = {}
        self.refused = []         # evaluation records that exist but fail the sidecar, the label or the digest
        self.inputs = {}          # every evaluation record read, with its SHA-256 (results.json's freshness)

    def allowed(self, arm, q, k):
        """Only arms, qualities and seeds no kill rule stopped enter a mark (a stopped arm's returns never do)."""
        if arm == "B":
            return bool(self.K["B"].get("go_on"))
        v = (self.K.get(arm) or {}).get(q) or {}
        return bool(v.get("go_on")) and int(k) in [int(x) for x in v.get("seeds_usable", [])]

    def get(self, arm, q, k):
        key = (arm, q, k)
        if not self.allowed(arm, q, k):
            return None
        if key not in self.cache:
            p = eval_path(arm, q, k)
            try:
                r = C.arm_record(p, self.digest)
                self.inputs[str(p)] = C.sha256_file(p)
                self.cache[key] = r if r.get("status") == "finished" else None
            except C.Refused as error:
                if p.exists():
                    self.refused.append(f"{p}: {error}")
                self.cache[key] = None
        return self.cache[key]

    def nonfinite(self, arm, policy):
        """(quality, seed) pairs whose evaluation of the policy holds a non-finite return on any bank."""
        out = []
        for q in C.QUALITIES:
            for k in C.SEEDS:
                r = self.get(arm, q, k)
                e = ((r or {}).get("evaluations") or {}).get(policy)
                if e is not None and not all((e.get(law) or {}).get("finite", False) for law in LAWS):
                    out.append(f"{q} seed {k}")
        return out

    def pts(self, arm, q, k, policy, law="strong"):
        r = self.get(arm, q, k)
        if r is None:
            return None
        e = (r.get("evaluations") or {}).get(policy)
        if e is None or not (e.get(law) or {}).get("finite", False):
            return None
        return e[law]["points"]

    def diff(self, arm, q, a, b, law="strong"):
        out = []
        for k in C.SEEDS:
            x, y = self.pts(arm, q, k, a, law), self.pts(arm, q, k, b, law)
            out.append(None if x is None or y is None else x - y)
        return summary(out)

    def mean(self, arm, q, policy, law="strong"):
        v = [self.pts(arm, q, k, policy, law) for k in C.SEEDS]
        return None if any(x is None for x in v) else float(np.mean(v))

    def situation(self, arm, q, k, policy, sit):
        r = self.get(arm, q, k)
        v = (((r or {}).get("situations") or {}).get(policy) or {}).get(sit) or {}
        return v.get("points") if v.get("finite") else None


def harm_lines(harm):
    """P2/G2/B5 not met: the harm line for each measured fall below -1.0; a missing number (a stopped seed or quality)
    is covered by the stop lines and makes the mark not met."""
    out = [f"It lowered return on {q}, {law} bank by {-m:.2f} points." for q, law, m in harm if m is not None]
    missing = sorted({q for q, _, m in harm if m is None})
    if missing:
        out.append(f"No-harm mark not met: numbers are missing on {', '.join(missing)} (stopped or not run).")
    if harm:
        out.append("There is no GO.")
    return out


def whole_arm_stopped(name):
    """The line when every quality an arm's marks need was stopped: none of its returns enters a mark (section 3.3),
    so no outcome line is chosen from marks that were never measured."""
    return f"No mark of arm {name} is met: the arm was stopped, so none of its returns enters a mark."


def marks_A(P, K):
    out = dict(P1={}, P2={}, P3={}, P4={}, info={})
    passing = []
    for q in C.P1_QUALITIES:
        s = P.diff("A", q, "A1", "none")
        out["P1"][q] = dict(s, meets=at_least(s, 2.0, 4))
        if out["P1"][q]["meets"]:
            passing.append(q)
    P1 = len(passing) >= 2
    harm = []
    for q in C.QUALITIES:
        for law in LAWS:
            s = P.diff("A", q, "A1", "none", law)
            out["P2"][f"{q}/{law}"] = dict(s, meets=at_least(s, -1.0))
            if not out["P2"][f"{q}/{law}"]["meets"]:
                harm.append((q, law, s["mean"]))
    P2 = not harm
    P3 = True
    P4 = True
    rivals_won = []
    for q in passing:
        s = P.diff("A", q, "A1", "untargeted")
        out["P3"][q] = dict(s, meets=at_least(s, 1.5))
        P3 &= out["P3"][q]["meets"]
        means = {p: P.mean("A", q, p) for p in PLAIN_A}
        best = max((p for p in PLAIN_A if means[p] is not None), key=lambda p: means[p], default=None)
        row = {}
        for name, rival in (("margin0", "margin0"), ("best_plain", best), ("matched_share", "matched_share")):
            s = P.diff("A", q, "A1", rival) if rival else summary([None] * 5)
            row[name] = dict(s, rival=rival, meets=at_least(s, 1.0))
            if not row[name]["meets"]:
                rivals_won.append(f"{rival or 'a plain margin'} on {q}")
        row["plain_means"] = means
        out["P4"][q] = row
        P4 &= all(row[n]["meets"] for n in ("margin0", "best_plain", "matched_share"))
    if not passing:
        P3 = P4 = False
    for q in C.QUALITIES:
        out["info"][q] = {f"{a}-{b}": P.diff("A", q, a, b) for a, b in (("A1", "shuffled_band"), ("A1_IW", "A1"),
                                                                        ("A1c", "none"), ("even_x1.33", "none"),
                                                                        ("host", "none"))}
    go = P1 and P2 and P3 and P4
    import kills as KL
    lines = KL.stop_lines(K, "A")
    if not any((K["A"].get(q) or {}).get("go_on") for q in C.QUALITIES):
        lines.append(whole_arm_stopped("A"))
    # the case's line from P1, P3 and P4; a failed P2 adds its harm line and removes the GO (section 4.4)
    elif not P1:
        lines.append("The credible-target pull did not raise return over TD3+BC.")
    elif P3 and P4:
        lines.append(f"Exploratory: the credible-target pull raised return over TD3+BC on {', '.join(passing)}, and "
                     "BCA's band beat margin 0, every plain margin and a matched-share margin.")
    elif P3:
        lines.append(f"Filtering helps; the band adds nothing over a plain margin. Tied or won: {', '.join(rivals_won)}.")
    else:
        lines.append("The gain does not come from targeting: releasing as many random rows did as well.")
    if any((K["A"].get(q) or {}).get("go_on") for q in C.QUALITIES):
        lines += harm_lines(harm)
    out.update(passing=passing, met=dict(P1=P1, P2=P2, P3=P3, P4=P4), GO=go, wording=lines)
    return out


def marks_B(P, K):
    out = dict(info={})
    s = P.diff("B", "medium-expert", "host+bca", "host")
    G1 = at_least(s, 2.0, 4)
    out["G1"] = dict(s, meets=G1)
    harm = []
    out["G2"] = {}
    for q in C.QUALITIES:
        for law in LAWS:
            s = P.diff("B", q, "host+bca", "host", law)
            out["G2"][f"{q}/{law}"] = dict(s, meets=at_least(s, -1.0))
            if not out["G2"][f"{q}/{law}"]["meets"]:
                harm.append((q, law, s["mean"]))
    G2 = not harm
    out["G3"] = {}
    for name, bar in (("range_limit", 1.5), ("shuffled", 1.5), ("matched_global", 1.0)):
        s = P.diff("B", "medium-expert", "host+bca", f"host+{name}")
        out["G3"][name] = dict(s, meets=at_least(s, bar))
    G3 = all(v["meets"] for v in out["G3"].values())
    sums = {}
    for t in PLAIN_B:
        m = [P.mean("B", q, f"host+{t}") for q in C.QUALITIES]
        sums[t] = None if any(v is None for v in m) else float(sum(m))
    best = max((t for t in PLAIN_B if sums[t] is not None), key=lambda t: sums[t], default=None)
    per_q, total = {}, 0.0
    for q in C.QUALITIES:
        s = P.diff("B", q, "host+bca", f"host+{best}") if best else summary([None] * 5)
        per_q[q] = s
        total = None if (total is None or s["mean"] is None) else total + s["mean"]
    G4 = best is not None and total is not None and total >= 2.0 and all(v["mean"] >= -1.0 for v in per_q.values())
    out["G4"] = dict(best_plain_test=best, sums=sums, per_quality=per_q, total=total, meets=bool(G4))
    out["info"]["G1_expert"] = P.diff("B", "expert", "host+bca", "host")
    for q in C.QUALITIES:
        out["info"][q] = {f"bca-{m}": P.diff("B", q, "host+bca", f"host+{m}") for m in
                          ("knee_clip", "bca_unweighted", "bca_exact", "bca_shifted")}
    # K3 (section 5.5): a non-finite gain, weight, actor or return stops that seed; two such seeds stop B. The seeds
    # counted in KILLS.json (records, every quality, seed 0 included) and the seeds whose returns came out non-finite
    # are counted together.
    rec_k3 = list((K["B"].get("rules") or {}).get("K3", {}).get("where") or K["B"].get("k3_seeds") or [])
    ret_k3 = sorted(set(P.nonfinite("B", "host+bca")) | set(P.nonfinite("B", "host")))
    all_k3 = sorted(set(rec_k3) | set(ret_k3))
    stopped_k3 = len(all_k3) >= 2 and "K3" not in (K["B"].get("stopped_by") or [])
    out["K3"] = dict(record_seeds=rec_k3, nonfinite_return_seeds=ret_k3, seeds=all_k3, stops_B=len(all_k3) >= 2)
    import kills as KL
    lines = KL.stop_lines(K, "B")
    if stopped_k3:
        lines.append(f"Stopped by K3 on {'; '.join(w.replace(' seed ', ', seed ') for w in all_k3)}: non-finite "
                     "values (a gain, weight, actor or return) on two or more seeds stop B.")
    if not K["B"].get("go_on") or stopped_k3:
        G1 = G2 = G3 = G4 = False
        lines.append(whole_arm_stopped("B"))
    # the case's line from G1, G3 and G4; a failed G2 adds its harm line and removes the GO (section 5.4)
    elif not G1:
        lines.append("The certified ceiling did not raise return over TD3+BC on medium-expert.")
    elif G3 and G4:
        lines.append("Exploratory: BCA's certified throttle ceiling raised TD3+BC's return on medium-expert, and BCA's "
                     "certificate beat the best plain test.")
    elif G3:
        lines.append(f"Reading held-out outcomes raised return on medium-expert; BCA's certificate did no better than "
                     f"the {best}. In short: outcome reading helped, not BCA.")
    else:
        did = [n for n, v in out["G3"].items() if not v["meets"]]
        lines.append(f"A ceiling raised return, but not by reading outcomes: {', '.join(did)} did as well.")
    if K["B"].get("go_on") and not stopped_k3:
        lines += harm_lines(harm)
    go = bool(G1 and G2 and G3)
    out.update(met=dict(G1=bool(G1), G2=bool(G2), G3=bool(G3), G4=bool(G4)), GO=go, wording=lines)
    return out


def marks_C(P, K):
    out = dict(B1={}, B2={}, B3={}, B4={}, B5={}, info={})
    scored = K["C_rules"]["B2_B4_scored"]
    passing = []
    for q in C.WEAK:
        s = P.diff("C", q, "rho0.5+bca", "rho1")
        out["B1"][q] = dict(s, meets=at_least(s, 3.0, 4))
        if out["B1"][q]["meets"]:
            passing.append(q)
    B1 = len(passing) >= 2
    harm = []
    for q in C.QUALITIES:
        for law in LAWS:
            s = P.diff("C", q, "rho0.5+bca", "rho1", law)
            out["B5"][f"{q}/{law}"] = dict(s, meets=at_least(s, -1.0))
            if not out["B5"][f"{q}/{law}"]["meets"]:
                harm.append((q, law, s["mean"]))
    B5 = not harm
    B2 = B3 = B4 = bool(passing) and scored
    best_of = {}
    for q in passing:
        s = P.diff("C", q, "rho0.5+bca", "rho0.5")
        out["B2"][q] = dict(s, meets=at_least(s, 1.5))
        B2 &= out["B2"][q]["meets"]
        row = {}
        for name, bar in (("range_limit", 1.5), ("shuffled", 1.5), ("matched_global", 1.0)):
            s = P.diff("C", q, "rho0.5+bca", f"rho0.5+{name}")
            row[name] = dict(s, meets=at_least(s, bar))
        out["B3"][q] = row
        B3 &= all(v["meets"] for v in row.values())
        means = {t: P.mean("C", q, f"rho0.5+{t}") for t in PLAIN_B}
        best = max((t for t in PLAIN_B if means[t] is not None), key=lambda t: means[t], default=None)
        best_of[q] = best
        s = P.diff("C", q, "rho0.5+bca", f"rho0.5+{best}") if best else summary([None] * 5)
        out["B4"][q] = dict(s, best_plain_test=best, plain_means=means, meets=at_least(s, 1.0))
        B4 &= out["B4"][q]["meets"]
    if not scored:
        out["B2_B4_not_scored"] = ("KB0 failed or the guard is void (KILLS.json C_rules); B2, B3 and B4 are not "
                                   "scored")
    # information (section 6.3): the release alone, the knee clip, a BCA map built for rho 1 and B's host-built map,
    # each against rho 1 and each against the co-trained host
    for q in C.QUALITIES:
        pairs = [("rho0.5", "rho1"), ("rho0.5+knee_clip", "rho1"), ("rho1+bca_built_for_rho1", "rho1"),
                 ("rho0.5+B_host_map", "rho1"), ("rho0.5", "host"), ("rho0.5+knee_clip", "host"),
                 ("rho1+bca_built_for_rho1", "host"), ("rho0.5+B_host_map", "host"), ("rho0.5+bca", "host"),
                 ("rho1", "host")]
        out["info"][q] = {f"{a}-{b}": P.diff("C", q, a, b) for a, b in pairs}
    # K3 as in B: record seeds (KILLS.json) and seeds with non-finite returns of the arm and its comparator together
    rec_k3 = list((K["C_rules"].get("K3") or {}).get("where") or [])
    ret_k3 = sorted(set(P.nonfinite("C", "rho0.5+bca")) | set(P.nonfinite("C", "rho1")))
    all_k3 = sorted(set(rec_k3) | set(ret_k3))
    stopped_k3 = len(all_k3) >= 2 and not K["C_rules"].get("stopped_by")
    out["K3"] = dict(record_seeds=rec_k3, nonfinite_return_seeds=ret_k3, seeds=all_k3, stops_C=len(all_k3) >= 2)
    import kills as KL
    lines = KL.stop_lines(K, "C")
    if stopped_k3:
        lines.append(f"Stopped by K3 on {'; '.join(w.replace(' seed ', ', seed ') for w in all_k3)}: non-finite "
                     "values (a gain, weight, actor or return) on two or more seeds stop C.")
    whole = stopped_k3 or bool(K["C_rules"].get("stopped_by")) or not any(
        (K["C"].get(q) or {}).get("go_on") for q in C.QUALITIES)
    if whole:
        B1 = B2 = B3 = B4 = B5 = False
        lines.append(whole_arm_stopped("C"))
    else:
        if not K["C_rules"]["KB0"]["holds"]:
            lines.append("The guard had nothing to guard: the bolder actor rarely went past 80%.")
        # the case's line from B1-B4 (B1 alone when KB0 failed or the guard is void); a failed B5 adds its harm line
        # and removes the GO (section 6.5)
        if not B1:
            lines.append("Halving the pull did not raise return on the weak qualities, guarded or not.")
        elif not scored:
            lines.append(f"Halving the pull, guarded by BCA's ceiling, met B1 on {', '.join(passing)}; B2-B4 are not "
                         "scored (KB0 failed or the guard is void).")
        elif not (B2 and B3):
            lines.append("Less pull helps; BCA adds nothing.")
        elif B4:
            lines.append(f"Exploratory: with BCA's certificate as its guard, a TD3+BC with half the pull raised return "
                         f"on {', '.join(passing)}.")
        else:
            lines.append(f"A held-out-outcome guard makes a bolder TD3+BC safe; BCA's certificate did no better than "
                         f"{', '.join(f'the {best_of[q]} on {q}' for q in passing)}.")
        lines += harm_lines(harm)
    go = bool(B1 and B2 and B3 and B5)
    out.update(passing=passing, met=dict(B1=bool(B1), B2=bool(B2) if scored else None,
                                         B3=bool(B3) if scored else None, B4=bool(B4) if scored else None,
                                         B5=bool(B5)), GO=go, wording=lines)
    return out


def marks_F(K, inputs):
    """F0's mark (section 7.4) per quality. A quality is decided only when no KF check (or failed input) stopped one of
    its seeds and all five A2 - F0 strong-test differences exist; an undecided quality's wording is "not decided:
    <reason>" (section 7.3) and no conclusion is drawn about it. The outcome line is chosen from the decided qualities
    only."""
    import kills as KL
    out = {}
    for q in C.F0_QUALITIES:
        kf = K["F0"].get(q) or {}
        per = {law: [] for law in LAWS}
        gaps = []
        for k in C.SEEDS:
            try:
                p = C.arm_dir("F0", q, k) / "record.json"
                f0 = C.arm_record(p)
                inputs[str(p)] = C.sha256_file(p)
            except C.Refused:
                f0 = None
            path, a2 = KL.a2_evaluation(q, k)
            if a2 is not None:
                inputs[str(path)] = C.sha256_file(path)
            ok = f0 is not None and f0.get("status") == "finished" and a2 is not None
            if f0 is None or f0.get("status") != "finished":
                gaps.append(f"F0 of seed {k} is {'missing' if f0 is None else 'stopped'}")
            elif a2 is None:
                gaps.append(f"A2's evaluation of seed {k} is missing")
            for law in LAWS:
                v = a2[law] - f0["evaluation"][law]["points"] if ok else None
                per[law].append(v if v is not None and math.isfinite(v) else None)
        s = summary(per["strong"])
        decided = bool(kf.get("decided")) and s["mean"] is not None
        reason = kf.get("note") or ("; ".join(gaps) if gaps else None) or "a strong-test difference is not finite"
        row = dict(strong=s, mild=summary(per["mild"]), none=summary(per["none"]), decided=decided,
                   credit=decided and at_least(s, 1.0, 4),
                   negative=decided and s["mean"] <= -1.0 and s.get("negatives", 0) >= 4,
                   note=None if decided else reason)
        out[q] = row
    decided = [q for q in C.F0_QUALITIES if out[q]["decided"]]
    credit = [q for q in decided if out[q]["credit"]]
    lines = [f"{q}: not decided: {out[q]['note']}." for q in C.F0_QUALITIES if not out[q]["decided"]]
    if len(decided) == 2:
        if len(credit) == 2:
            lines.append(f"BCA's band earned part of result F's gain: A2 beat the same no-credit rule without the band "
                         f"by {out['medium-expert']['strong']['mean']:.2f} points on medium-expert and "
                         f"{out['expert']['strong']['mean']:.2f} on expert.")
        elif len(credit) == 1:
            other = [q for q in decided if q not in credit][0]
            lines.append(f"Result F's gain comes partly from BCA's band (on {credit[0]}); on {other} it comes from "
                         "refusing credit past the data, not from BCA's band.")
        else:
            lines.append("Result F's gain comes from refusing credit past the data, not from BCA's band.")
    elif len(decided) == 1:          # the other quality is not decided: only the decided quality's part is worded
        q = decided[0]
        if credit:
            lines.append(f"Result F's gain comes partly from BCA's band (on {q}).")
        else:
            lines.append(f"On {q}, result F's gain comes from refusing credit past the data, not from BCA's band.")
    for q in decided:
        if out[q]["negative"]:
            lines.append(f"On {q}, the band's finite part lowered return by {-out[q]['strong']['mean']:.2f} points.")
    return dict(per_quality=out, decided=decided, credit=credit, wording=lines)


def info_A_records(K, inputs):
    """Information (section 4.3) from arm A's records: s, the band's unit, mean U / s, and A1's release shares per
    refresh among all kept rows, recorder rows and random rows, with the releases the guard blocked."""
    out = {}
    for q in C.QUALITIES:
        for k in C.SEEDS:
            p = C.arm_dir("A", q, k) / "record.json"
            try:
                r = C.arm_record(p)
            except C.Refused:
                continue
            inputs[str(p)] = C.sha256_file(p)
            inp = r.get("inputs") or {}
            ref = [dict(step=h.get("step"), **{f: (h.get("arms") or {}).get("A1", {}).get(f) for f in
                                               ("released_share", "released_share_recorder_rows",
                                                "released_share_random_rows", "guard_blocked",
                                                "guard_blocked_recorder_rows", "guard_blocked_random_rows")})
                   for h in r.get("history") or []]
            out.setdefault(q, {})[str(k)] = dict(status=r.get("status"), stopped_by=r.get("stopped_by"),
                                                 s=inp.get("s"), unit=inp.get("unit"),
                                                 U_mean_over_s=inp.get("U_mean_over_s"), A1_per_refresh=ref)
    return out


SITUATION_ARMS = (("A", "A1", "none"), ("B", "host+bca", "host"), ("C", "rho0.5+bca", "rho1"), ("F", "A2", "F0"))


def situations_info(P, K, inputs):
    """Information only, after the marks (section 8 step 6): the main test's four situations for the primary arms
    (A1, none, host + BCA, host, rho 0.5 + BCA, rho 1, A2, F0): 5-seed mean points of each and the paired difference,
    per quality and situation, over the seeds that entered the marks (F: every F0 seed that finished)."""
    out = {}
    for arm, a, b in SITUATION_ARMS:
        for q in (C.F0_QUALITIES if arm == "F" else C.QUALITIES):
            row = {}
            for sname in C.SITUATIONS:
                xa, xb = [], []
                for k in C.SEEDS:
                    if arm == "F":
                        r = None
                        pth = eval_path("F", q, k)
                        try:
                            r = C.arm_record(pth)
                            inputs[str(pth)] = C.sha256_file(pth)
                        except C.Refused:
                            pass
                        sit = ((r or {}).get("situations") or {}) if (r or {}).get("status") == "finished" else {}
                        va = (sit.get(a) or {}).get(sname) or {}
                        vb = (sit.get(b) or {}).get(sname) or {}
                        xa.append(va.get("points") if va.get("finite") else None)
                        xb.append(vb.get("points") if vb.get("finite") else None)
                    else:
                        xa.append(P.situation(arm, q, k, a, sname))
                        xb.append(P.situation(arm, q, k, b, sname))
                d = [None if x is None or y is None else x - y for x, y in zip(xa, xb)]
                row[sname] = {a: xa, b: xb, f"{a}-{b}": summary(d),
                              f"{a}_mean": None if None in xa else float(np.mean(xa)),
                              f"{b}_mean": None if None in xb else float(np.mean(xb))}
            out[f"{arm}/{q}"] = row
    return out


def expected_evals(K):
    """The evaluation records the scheduler runs (rt.py jobs_continue): every seed of every quality the screen let
    through, per arm. Scoring refuses until each exists and verifies (a missing number from an unfinished job is not a
    stopped seed)."""
    sv = K.get("survivors") or {}
    seeds = (0,) if K.get("smoke_test") else C.SEEDS
    return [(arm, q, k) for arm, key in (("A", "A"), ("B", "B"), ("C", "C"), ("F", "F0"))
            for q in sv.get(key, []) for k in seeds]


def marks_job():
    digest = C.require_passmarks()
    K = kills(digest)
    kpath = C.docs() / "KILLS.json"
    missing = []
    for arm, q, k in expected_evals(K):
        try:
            C.arm_record(eval_path(arm, q, k), digest)
        except C.Refused:
            missing.append(f"eval {arm} {q} seed {k}")
    if missing:
        raise C.Refused(f"refused: evaluation records are missing or unverified ({'; '.join(missing)}); no mark is "
                        "computed until every evaluation job has finished")
    P = Points(digest, K)
    inputs = {str(kpath): C.sha256_file(kpath)}
    res = dict(kind="arms-abc-f0-results", label=C.label(digest), exploratory=True, passmarks_sha256=digest,
               made_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), kills_sha256=inputs[str(kpath)],
               A=marks_A(P, K), B=marks_B(P, K), C=marks_C(P, K), F0=marks_F(K, inputs),
               confirmatory="none: seeds 0-4 are seen seeds; a confirmatory claim needs fresh seeds 15-19 and its own "
                            "pre-registration")
    res["info"] = dict(A_records=info_A_records(K, inputs), situations=situations_info(P, K, inputs))
    res["refused_records"] = P.refused
    inputs.update(P.inputs)
    res["inputs_sha256"] = inputs
    C.write_record(C.docs() / "results.json", res)
    C.write_atomic(C.docs() / "RESULTS.md", results_md(res))
    C.log(f"marks: A GO {res['A']['GO']}, B GO {res['B']['GO']}, C GO {res['C']['GO']}, F0 credit "
          f"{res['F0']['credit']} -> {C.docs() / 'RESULTS.md'}")
    return 0


def results_md(r):
    def f(x):
        return "-" if x is None else f"{x:.2f}"

    def row(name, s, bar):
        pos = "-" if s.get("positives") is None else s["positives"]
        return (f"| {name} | {f(s.get('mean'))} | {pos} | {f(s.get('lower_bound'))} | "
                f"{', '.join(f(v) for v in s['per_seed'])} | {bar} | {'yes' if s.get('meets') else 'no'} |")

    def info_row(name, s):
        pos = "-" if s.get("positives") is None else s["positives"]
        return f"| {name} | {f(s.get('mean'))} | {pos} | {', '.join(f(v) for v in s['per_seed'])} |"
    L = ["# Arms A, B, C and F0: results (EXPLORATORY)", "", f"**EXPLORATORY.** {r['label']}.", "",
         f"Made {r['made_at']}. {r['confirmatory']}.", "",
         "Differences are in points on the named bank (strong unless named), 5-seed means; the lower bound (mean - "
         "2.132 SD / sqrt 5) is reported and decides nothing. A missing number (a stopped seed, a failed actor, a "
         "refused record) makes the mark not met.", ""]
    if r.get("refused_records"):
        L += ["**Records refused** (no label, another pass-mark file, or not matching their checksum): "
              + "; ".join(r["refused_records"]), ""]
    head = "| Mark | Mean | Seeds positive | Lower bound | Per seed | Bar | Met |\n|---|---|---|---|---|---|---|"
    ihead = "| Difference (information only) | Mean | Seeds positive | Per seed |\n|---|---|---|---|"
    A = r["A"]
    L += ["## Arm A: credible-target pull", "", *[f"> {x}" for x in A["wording"]], "", head]
    for q, s in A["P1"].items():
        L.append(row(f"P1 A1 - none, {q}", s, ">= +2.0, 4 of 5"))
    for k, s in A["P2"].items():
        L.append(row(f"P2 A1 - none, {k}", s, ">= -1.0"))
    for q, s in A["P3"].items():
        L.append(row(f"P3 A1 - untargeted, {q}", s, ">= +1.5"))
    for q, v in A["P4"].items():
        for n in ("margin0", "best_plain", "matched_share"):
            L.append(row(f"P4 A1 - {v[n].get('rival')}, {q}", v[n], ">= +1.0"))
    L += ["", f"Marks met: {A['met']}; GO: {A['GO']}.", ""]
    B = r["B"]
    L += ["## Arm B: certified throttle ceiling", "", *[f"> {x}" for x in B["wording"]], "", head,
          row("G1 (host + BCA) - host, medium-expert", B["G1"], ">= +2.0, 4 of 5")]
    for k, s in B["G2"].items():
        L.append(row(f"G2 (host + BCA) - host, {k}", s, ">= -1.0"))
    for n, s in B["G3"].items():
        L.append(row(f"G3 BCA - {n}, medium-expert", s, ">= +1.5" if n != "matched_global" else ">= +1.0"))
    L += ["", f"G4: best plain test {B['G4']['best_plain_test']}, summed difference {f(B['G4']['total'])} (bar +2.0, "
              f"no quality below -1.0): {'met' if B['G4']['meets'] else 'not met'}.",
          f"Marks met: {B['met']}; GO: {B['GO']}.", ""]
    Cm = r["C"]
    L += ["## Arm C: guarded release", "", *[f"> {x}" for x in Cm["wording"]], "", head]
    for q, s in Cm["B1"].items():
        L.append(row(f"B1 (rho 0.5 + BCA) - rho 1, {q}", s, ">= +3.0, 4 of 5"))
    for k, s in Cm["B5"].items():
        L.append(row(f"B5 (rho 0.5 + BCA) - rho 1, {k}", s, ">= -1.0"))
    for q, s in Cm["B2"].items():
        L.append(row(f"B2 (rho 0.5 + BCA) - rho 0.5, {q}", s, ">= +1.5"))
    for q, v in Cm["B3"].items():
        for n, s in v.items():
            L.append(row(f"B3 BCA - {n}, {q}", s, ">= +1.5" if n != "matched_global" else ">= +1.0"))
    for q, s in Cm["B4"].items():
        L.append(row(f"B4 BCA - {s.get('best_plain_test')}, {q}", s, ">= +1.0"))
    L += ["", f"Marks met: {Cm['met']}; GO: {Cm['GO']}.", ""]
    F = r["F0"]
    L += ["## F0: result F's no-band control", "", *[f"> {x}" for x in F["wording"]], "", head]
    for q, v in F["per_quality"].items():
        L.append(row(f"A2 - F0, {q} (strong)", dict(v["strong"], meets=v["credit"]), ">= +1.0, 4 of 5"))
    L.append("")
    # information only, after the marks
    L += ["## Information only (after the marks; decides nothing)", ""]
    L += ["### Arm A", "", ihead]
    for q, d in A["info"].items():
        for n, s in d.items():
            L.append(info_row(f"{n}, {q}", s))
    L += ["", "### Arm B", "", ihead, info_row("G1's value on expert: (host + BCA) - host", B["info"]["G1_expert"])]
    for q in (x for x in B["info"] if x != "G1_expert"):
        for n, s in B["info"][q].items():
            L.append(info_row(f"{n}, {q}", s))
    L += ["", "### Arm C (against rho 1 and against the co-trained host)", "", ihead]
    for q, d in Cm["info"].items():
        for n, s in d.items():
            L.append(info_row(f"{n}, {q}", s))
    L += ["", "### F0 on the mild and none banks", "", ihead]
    for q, v in F["per_quality"].items():
        for law in ("mild", "none"):
            L.append(info_row(f"A2 - F0, {q} ({law})", v[law]))
    L += ["", "### The main test's four situations (primary arms; points, 5-seed means)", "",
          "| Arm, quality | Situation | Arm | Comparator | Difference (mean) | Seeds positive |", "|---|---|---|---|---|---|"]
    for key, row_ in r["info"]["situations"].items():
        for sname, v in row_.items():
            names = [n for n in v if not n.endswith("_mean") and "-" not in n]
            dkey = [n for n in v if "-" in n and not n.endswith("_mean")][0]
            d = v[dkey]
            L.append(f"| {key} ({dkey}) | {sname} | {f(v[names[0] + '_mean'])} | {f(v[names[1] + '_mean'])} | "
                     f"{f(d.get('mean'))} | {'-' if d.get('positives') is None else d['positives']} |")
    L += ["", "Arm A's release shares per refresh, s, the band's unit and mean U / s are in results.json "
              "(info.A_records).", ""]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description="EXPLORATORY scoring of arms A, B, C and F0")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--arm", required=True, choices=("A", "B", "C", "F"))
    e.add_argument("--quality", required=True, choices=C.QUALITIES)
    e.add_argument("--seed", type=int, required=True, choices=C.SEEDS)
    sub.add_parser("marks")
    a = p.parse_args(argv)
    try:
        if a.cmd == "eval":
            return eval_job(a.arm, a.quality, a.seed)
        return marks_job()
    except C.Refused as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
