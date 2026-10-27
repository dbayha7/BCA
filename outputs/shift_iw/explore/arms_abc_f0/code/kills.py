"""EXPLORATORY: arms A, B, C and F0 -- every kill rule across the five seeds, before any final return is computed
(frozen pass marks, sections 3.3, 4.5, 5.5, 6.4, 7.3 and 8 step 5).

Reads the screen (SCREEN_seed0.json) and every arm record of seeds 0-4 (label and frozen digest checked), and A2's
evaluation of every seed whose F0 finished, and writes routes2/arms/KILLS.json (+ .sha256) and KILLS.md:
  A   K0 (binding on seeds 1-4: a stopped seed is never replaced), K1 (any seed stops the quality), K3 (that seed),
      K2 across the seeds that reached 8,000 steps (A1 - none, strong test, -3.0 or lower on at least 3 of 5 stops the
      quality); a quality the screen stopped keeps the screen's rule, seed 0 and reason
  B   K0 and K4 on the 5-seed means (medium-expert), K1 and K2 over every seed-quality pair, K3 over every seed (the
      screen's seed 0 included; two seeds stop B). B's maps of seeds 1-4 are read for all five qualities when B survives
      or when C survives and B's certificate was not voided at the screen (b_map_qualities)
  C   KB1 (any seed stops rho 0.5 on the quality), K3 (two seeds stop C, as in B), KB0 on the 5-seed means (fails:
      B2-B4 are not scored), the guard-void rule (B's K1 or K2 over every seed and quality, or C's own map's K1
      condition in any seed and quality)
  F0  KF1-KF4 per seed and inputs that failed for good (a failed seed: the quality is "not decided: <reason>")
Every stop is kept with its rule, seed and reason (stop_lines gives the frozen wording "Stopped by <rule> on <quality,
seed>: <reason>."). KILLS.json is complete only when every expected record exists; it records the SHA-256 of every file
it read, so rt.py done counts it as done only while those files are unchanged. Nothing here computes a return.

  python -B kills.py
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402


def b_map_qualities(screen):
    """The qualities whose arm-B maps of seeds 1-4 are built: all five when B survives the screen (its K1 and K2 count
    over every seed and quality, section 5.5), or when C survives and B's certificate was not already voided at the
    screen (C's guard is void if B's K1 or K2 fires, section 6.4; those rules read every seed's map, so they are built
    even when B itself was stopped at the screen by K0, K3 or K4). rt.py jobs_continue uses the same rule."""
    sv = screen["survivors"]
    if sv["B"] or (sv["C"] and not screen["C_rules"]["guard_void"]["by_B"]):
        return list(C.QUALITIES)
    return []


def a2_evaluation(q, k):
    """A2's evaluation of a seed (result F's unchanged script): (path, A2's points per bank) when it exists, parses,
    holds A2 and is newer than A2's record; else (path, None)."""
    p = C.f1_out() / q / f"evaluate_seed{int(k)}.json"
    rec = C.f1_out() / q / "A2_cmd_strong" / f"seed{int(k)}" / "record.json"
    try:
        body = json.loads(p.read_text())
        ev = body["evaluations"]["A2_cmd_strong"]
        pts = {law: float(ev[law]["points"]) for law in ("none", "mild", "strong")}
        if body.get("quality") != q or int(body.get("seed")) != int(k) or p.stat().st_mtime < rec.stat().st_mtime:
            return p, None
        return p, pts
    except (OSError, ValueError, KeyError, TypeError):
        return p, None


def build(digest):
    screen_path = C.docs() / "SCREEN_seed0.json"
    screen = C.arm_record(screen_path, digest)
    if screen.get("complete") is not True:
        raise C.Refused("refused: no complete, verified seed-0 screen")
    sv = screen["survivors"]
    inputs = {str(screen_path): C.sha256_file(screen_path)}
    out = dict(kind="arms-abc-f0-kills", label=C.label(digest), exploratory=True, passmarks_sha256=digest,
               made_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), screen_sha256=inputs[str(screen_path)], survivors=sv,
               A={}, B={}, C={}, F0={}, missing=[], inputs_sha256=inputs)

    def records(arm, qualities, seeds=C.SEEDS):
        got, missing = {}, []
        for q in qualities:
            for k in seeds:
                p = C.arm_dir(arm, q, k) / "record.json"
                try:
                    got[(q, k)] = C.arm_record(p)
                    inputs[str(p)] = C.sha256_file(p)
                except C.Refused:
                    missing.append(f"{arm} {q} seed {k}")
        return got, missing

    def screen_stop(arm, q):
        """A quality the screen did not pass on: its rule, seed 0 and reason, as the screen recorded them."""
        v = screen[arm].get(q) or {}
        if v.get("status") == "blocked":
            return dict(checked=True, go_on=False, at="screen", stopped_by="D1", seed=None,
                        reason=v.get("reason") or "D1 differs from the committed report or the report's D1 did not "
                                                  "pass: band fits on this quality wait for the owner (section 8 "
                                                  "step 2)")
        rule = v.get("stopped_by") or ("K3" if (v.get("K3") or {}).get("fires") else None)
        reason = v.get("reason") or "; ".join((v.get("K3") or {}).get("reasons") or []) or None
        return dict(checked=True, go_on=False, at="screen", stopped_by=rule or v.get("status") or "screen", seed=0,
                    reason=reason)

    def k3_of(r):
        return r.get("stopped_by") == "K3" or bool(((r.get("kills") or {}).get("K3") or {}).get("fires"))

    # ---------------------------------------------------------------- A
    recs, miss = records("A", sv["A"])
    out["missing"] += miss
    for q in C.QUALITIES:
        if q not in sv["A"]:
            out["A"][q] = screen_stop("A", q)
            out["A"][q]["stops"] = [dict(rule=out["A"][q]["stopped_by"], seeds=[0] if out["A"][q]["seed"] == 0
                                         else [], reason=out["A"][q]["reason"], at="screen")]
            continue
        per, stops, stopped_quality = {}, [], None
        for k in C.SEEDS:
            r = recs.get((q, k))
            if r is None:
                per[k] = dict(status="missing")
                continue
            kk = r.get("kills") or {}
            per[k] = dict(status=r["status"], stopped_by=r.get("stopped_by"), reason=r.get("reason"),
                          not_run=bool(r.get("not_run")), K0=kk.get("K0"), K1=kk.get("K1"),
                          K2=(kk.get("K2_value") or {}).get("d"))
            if r.get("stopped_by") and not r.get("not_run"):    # a rule that fired on this seed
                stops.append(dict(rule=r["stopped_by"], seeds=[k], reason=r.get("reason")))
                if r["stopped_by"] == "K1":
                    stopped_quality = f"K1 on seed {k}"
        reached = {k: v["K2"] for k, v in per.items() if v.get("K2") is not None}
        bad = [k for k, d in reached.items() if d <= -3.0]
        complete = all(per[k]["status"] != "missing" for k in C.SEEDS)
        k2 = dict(values=reached, seeds_reached_8000=len(reached), at_or_below_minus_3=len(bad), fires=len(bad) >= 3,
                  rule="A1 - none on the strong test from the 8,000-step actors, -3.0 or lower on at least 3 of 5 "
                       "seeds")
        if k2["fires"]:
            stops.append(dict(rule="K2", seeds=bad, reason=f"A1 - none on the strong test from the 8,000-step actors "
                                                           f"was -3.0 points or lower on {len(bad)} of 5 seeds"))
        stopped_by = "K1" if stopped_quality else ("K2" if k2["fires"] else None)
        out["A"][q] = dict(checked=complete, per_seed=per, K2=k2, stopped_by=stopped_by, K1=stopped_quality,
                           stops=stops, seeds_usable=[k for k, v in per.items() if v["status"] == "finished"],
                           go_on=complete and stopped_by is None)
    # ---------------------------------------------------------------- B
    bq = b_map_qualities(screen)
    recs, miss = records("B", bq)
    out["missing"] += miss
    room, k4 = {}, {}
    k1, k2, k3, k3_why = [], [], [], {}
    for (q, k), r in sorted(recs.items(), key=lambda x: (C.QUALITIES.index(x[0][0]), x[0][1])):
        kk = r.get("kills") or {}
        if (kk.get("K1") or {}).get("fires"):
            k1.append(f"{q} seed {k}")
        if kk.get("K2_over_refusal"):
            k2.append(f"{q} seed {k}")
        if k3_of(r):
            k3.append(f"{q} seed {k}")
            k3_why[f"{q} seed {k}"] = r.get("reason") or "non-finite values: " + "; ".join(
                (kk.get("K3") or {}).get("reasons") or [])
        if q == "medium-expert" and r.get("status") == "finished":
            room[k] = (kk.get("K0_room") or {}).get("room")
            k4[k] = kk.get("K4_clip_share_thruster1")
    for q in C.QUALITIES:                 # seed-0 K3 of a quality whose seeds 1-4 were not built
        sb = screen["B"].get(q) or {}
        if q not in bq and (sb.get("K3") or {}).get("fires"):
            k3.append(f"{q} seed 0")
            k3_why[f"{q} seed 0"] = sb.get("reason") or "non-finite values: " + "; ".join(sb["K3"].get("reasons") or [])
    room_mean = float(np.mean(list(room.values()))) if len(room) == 5 and None not in room.values() else None
    k4_mean = float(np.mean(list(k4.values()))) if len(k4) == 5 and None not in k4.values() else None
    b_rules = dict(K0=dict(mean=room_mean, seeds=len(room), fires=room_mean is not None and room_mean < 1.0),
                   K1=dict(where=k1, fires=bool(k1)), K2=dict(pairs=k2, fires=len(k2) >= 2),
                   K3=dict(where=k3, fires=len(k3) >= 2), K4=dict(mean=k4_mean, seeds=len(k4),
                                                                   fires=k4_mean is not None and k4_mean < 0.02))
    screen_b = screen.get("B_stopped_by") or []
    stopped = list(screen_b) + [n for n, v in b_rules.items() if v["fires"] and n not in screen_b]
    stops = []
    for n in screen_b:
        v = screen["B_rules"][n]
        stops.append(dict(rule=n, at="screen", where=v.get("where") or v.get("pairs") or ["medium-expert seed 0"],
                          value=v.get("value"), reason=v.get("rule")))
    for n in [n for n in stopped if n not in screen_b]:
        v = b_rules[n]
        stops.append(dict(rule=n, at="seeds 0-4", where=v.get("where") or v.get("pairs") or
                          ["medium-expert, 5-seed mean"], value=v.get("mean"),
                          reason=dict(K0="(host + knee clip) - host on the strong test, medium-expert, 5-seed mean, "
                                         "below +1.0",
                                      K1="BCA's thruster-1 ceiling 0.85 or higher with at least 30 tries in rung "
                                         "(0.80, 0.85]",
                                      K2="BCA's thruster-1 ceiling 0.60 or lower in 2 or more seed-quality pairs",
                                      K3="non-finite gain, weight, actor or return on two seeds",
                                      K4="BCA's map clips under 2% of the host's thruster-1 commands (strong starts, "
                                         "medium-expert, 5-seed mean)")[n]))
    if "K3" not in stopped:              # one K3 seed stops that seed and quality only (section 5.5)
        stops += [dict(rule="K3", at="seed", where=[w], value=None, reason=k3_why.get(w) or "non-finite values")
                  for w in k3]
    full = bool(sv["B"]) and not miss
    usable = {q: [k for k in C.SEEDS if (q, k) in recs and recs[(q, k)]["status"] == "finished"
                  and not k3_of(recs[(q, k)])] for q in sv["B"]}
    out["B"] = dict(checked=(not miss) if bq else True, built_for=bq, rules=b_rules, stopped_by=stopped, stops=stops,
                    go_on=full and not stopped, k3_seeds=k3, seeds_usable=usable, screen_stopped_by=screen_b)
    # ---------------------------------------------------------------- C
    recs, miss = records("C", sv["C"])
    out["missing"] += miss
    shares, own_void = {}, []
    c_k3 = [w for w in screen["C_rules"].get("K3", {}).get("where", []) if w.split(" seed")[0] not in sv["C"]]
    for (q, k), r in sorted(recs.items(), key=lambda x: (C.QUALITIES.index(x[0][0]), x[0][1])):
        if k3_of(r):
            c_k3.append(f"{q} seed {k}")
    c_stopped = ["K3"] if len(c_k3) >= 2 else []
    for q in C.QUALITIES:
        if q not in sv["C"]:
            out["C"][q] = screen_stop("C", q)
            if out["C"][q]["stopped_by"] == "finished" and screen["C_rules"].get("K3", {}).get("fires"):
                out["C"][q].update(stopped_by="K3", reason="C stops: non-finite values on two seeds at the screen "
                                                           f"({', '.join(screen['C_rules']['K3']['where'])})")
            out["C"][q]["stops"] = [dict(rule=out["C"][q]["stopped_by"], seeds=[0], reason=out["C"][q]["reason"],
                                         at="screen")]
            continue
        per, stop, stops = {}, None, []
        for k in C.SEEDS:
            r = recs.get((q, k))
            if r is None:
                per[k] = dict(status="missing")
                continue
            kk = r.get("kills") or {}
            per[k] = dict(status=r["status"], stopped_by=r.get("stopped_by"), reason=r.get("reason"),
                          KB1=kk.get("KB1"), KB0=kk.get("KB0_seed"), K3=kk.get("K3"))
            if r.get("stopped_by"):
                stops.append(dict(rule=r["stopped_by"], seeds=[k], reason=r.get("reason")))
            elif k3_of(r):
                stops.append(dict(rule="K3", seeds=[k], reason="non-finite values: " +
                                  "; ".join(kk["K3"].get("reasons") or [])))
            if r.get("stopped_by") == "KB1":
                stop = f"KB1 on seed {k}"
            if (kk.get("guard_void_own_map") or {}).get("fires"):
                own_void.append(f"{q} seed {k}")
            if kk.get("KB0_seed") is not None:
                shares.setdefault(q, []).append((kk["KB0_seed"]["share_rho05"], kk["KB0_seed"]["share_rho1"]))
        complete = all(per[k]["status"] != "missing" for k in C.SEEDS)
        out["C"][q] = dict(checked=complete, per_seed=per, stopped_by="KB1" if stop else None, KB1=stop, stops=stops,
                           seeds_usable=[k for k, v in per.items() if v["status"] == "finished"
                                         and not (v.get("K3") or {}).get("fires")],
                           go_on=complete and not stop and not c_stopped)
    kb0 = {}
    for q in C.WEAK:
        v = shares.get(q)
        if v and len(v) == 5:
            m05, m1 = float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))
            kb0[q] = dict(mean_share_rho05=m05, mean_share_rho1=m1, passes=bool(m05 >= 0.02 and m05 >= 2.0 * m1))
        else:
            kb0[q] = dict(passes=False, note="fewer than five seeds")
    kb0_holds = sum(1 for v in kb0.values() if v["passes"]) >= 2 and screen["C_rules"]["KB0"]["holds"]
    void_b = bool(b_rules["K1"]["fires"] or b_rules["K2"]["fires"] or screen["C_rules"]["guard_void"]["by_B"])
    out["C_rules"] = dict(K3=dict(where=c_k3, fires=bool(c_stopped),
                                  rule="two seeds with a non-finite gain, weight, actor or return stop C (as in B)"),
                          stopped_by=c_stopped,
                          KB0=dict(per_quality=kb0, holds=kb0_holds, seed0_holds=screen["C_rules"]["KB0"]["holds"]),
                          guard_void=dict(by_B=void_b, by_B_rules=dict(K1=b_rules["K1"], K2=b_rules["K2"]),
                                          B_maps_built_for=bq, by_own_map=own_void, void=bool(void_b or own_void)),
                          B2_B4_scored=bool(kb0_holds and not (void_b or own_void)))
    # ---------------------------------------------------------------- F0
    recs, miss = records("F0", sv["F0"])
    out["missing"] += miss
    for q in C.F0_QUALITIES:
        if q not in sv["F0"]:
            st = screen_stop("F0", q)
            st.update(decided=False, failed_seeds=[0],
                      note=f"stopped by {st['stopped_by']} on seed 0: {st['reason'] or 'no reason recorded'}")
            out["F0"][q] = st
            continue
        per, failed = {}, []
        for k in C.SEEDS:
            r = recs.get((q, k))
            if r is None:
                per[k] = dict(status="missing")
                continue
            per[k] = dict(status=r["status"], stopped_by=r.get("stopped_by"), reason=r.get("reason"))
            if r.get("stopped_by"):
                failed.append(k)
            elif r["status"] == "finished":           # the mark needs A2's evaluation of the same seed
                path, pts = a2_evaluation(q, k)
                if pts is None:
                    out["missing"].append(f"A2's evaluation of {q} seed {k} ({path})")
                    per[k]["a2_evaluation"] = "missing"
                else:
                    inputs[str(path)] = C.sha256_file(path)
                    per[k]["a2_evaluation"] = str(path)
        complete = all(v["status"] != "missing" and v.get("a2_evaluation") != "missing" for v in per.values())
        note = None
        if failed:
            note = "; ".join(f"seed {k} not run: {per[k]['reason']}" if per[k]["stopped_by"] == "input" else
                             f"stopped by {per[k]['stopped_by']} on seed {k}: {per[k]['reason']}" for k in failed)
        out["F0"][q] = dict(checked=complete, per_seed=per, failed_seeds=failed, decided=complete and not failed,
                            note=note)
    out["complete"] = not out["missing"]
    return out


def _where(v):
    seeds = v.get("seeds")
    if seeds is None or len(seeds) == 0:
        return ""
    return f", seed {seeds[0]}" if len(seeds) == 1 else f", seeds {', '.join(str(x) for x in seeds)}"


def stop_lines(K, arm):
    """Every kill rule that fired for an arm, in the frozen form "Stopped by <rule> on <quality, seed>: <reason>."
    (section 4.4; used for B and C as well). Seeds that were not run after a quality-wide stop are not listed as
    stops. A quality held back by the D1 gate and a seed whose input failed for good are worded as not run."""
    out = []
    if arm == "B":
        for st in K["B"].get("stops") or []:
            where = "; ".join(w.replace(" seed ", ", seed ") for w in st.get("where") or [])
            val = "" if st.get("value") is None else f" (value {st['value']:.4g})"
            out.append(f"Stopped by {st['rule']} on {where}: {st['reason']}{val}.")
        return out
    if arm == "C" and (K.get("C_rules") or {}).get("stopped_by"):
        out.append(f"Stopped by K3 on {'; '.join(w.replace(' seed ', ', seed ') for w in K['C_rules']['K3']['where'])}: "
                   "non-finite values on two seeds stop C.")
    for q, v in (K.get(arm) or {}).items():
        for st in v.get("stops") or []:
            if st["rule"] == "D1":
                out.append(f"Not run on {q}: {st['reason']}.")
            elif st["rule"] == "input":
                out.append(f"Not run on {q}{_where(st)}: {st['reason']}.")
            else:
                out.append(f"Stopped by {st['rule']} on {q}{_where(st)}: {st.get('reason') or 'no reason recorded'}.")
    return out


def markdown(K):
    L = ["# Kill checks across seeds 0-4 (EXPLORATORY)", "",
         f"**EXPLORATORY.** {K['label']}. Seeds 0-4 are seen seeds.", "",
         f"Made {K['made_at']}. Complete: {'yes' if K['complete'] else 'no: ' + '; '.join(K['missing'])}.", "",
         "## Arm A", "", "| Quality | Stopped by | K2 values (A1 - none at 8,000 steps) | Seeds usable |", "|---|---|---|---|"]
    for q, v in K["A"].items():
        vals = ", ".join(f"{k}: {d:.2f}" for k, d in (v.get("K2") or {}).get("values", {}).items())
        L.append(f"| {q} | {v.get('stopped_by') or '-'} | {vals or '-'} | {v.get('seeds_usable', '-')} |")
    L += ["", "Every rule that fired:", ""] + [f"- {line}" for line in stop_lines(K, "A")] if stop_lines(K, "A") else []
    b = K["B"]
    L += ["", "## Arm B", "", f"Stopped by: {', '.join(b['stopped_by']) or 'nothing'}.", ""]
    L += [f"- {line}" for line in stop_lines(K, "B")] + [""]
    for n, v in b["rules"].items():
        L.append(f"- **{n}:** {json.dumps(v)}")
    L += ["", "## Arm C", "", "| Quality | Stopped by | Seeds usable |", "|---|---|---|"]
    for q, v in K["C"].items():
        L.append(f"| {q} | {v.get('stopped_by') or '-'} | {v.get('seeds_usable', '-')} |")
    cr = K["C_rules"]
    L += ["", f"- **K3 (two seeds stop C):** {json.dumps(cr['K3'])}.",
          f"- **KB0 on the 5-seed means:** {json.dumps(cr['KB0']['per_quality'])}; holds: {cr['KB0']['holds']}.",
          f"- **Guard void:** {json.dumps(cr['guard_void'])}. B2-B4 scored: {cr['B2_B4_scored']}.", "",
          "## F0", "", "| Quality | Decided | Note |", "|---|---|---|"]
    for q, v in K["F0"].items():
        L.append(f"| {q} | {v.get('decided', False)} | {v.get('note') or '-'} |")
    L += ["", "## Arm C: every rule that fired", ""] + ([f"- {line}" for line in stop_lines(K, "C")] or ["- none"])
    L.append("")
    return "\n".join(L)


def smoke_stub(digest):
    """Only under ARMS_SMOKE (run_arms.sh smoke): a stand-in kill record that lets score.py's evaluation and mark code
    run on the smoke test's records (medium-expert for A, B and F0; random for C, seed 0). Never a real result."""
    if C.smoke() is None:
        raise C.Refused("refused: the stand-in kill record exists only for the smoke test (ARMS_SMOKE)")
    go = dict(go_on=True, seeds_usable=[0], checked=True, stopped_by=None, stops=[])
    stop = dict(go_on=False, seeds_usable=[], checked=True, stopped_by="smoke test: not run", stops=[])
    sv = dict(A=["medium-expert"], B=["medium-expert"], C=["random"], F0=["medium-expert"])
    return dict(kind="arms-abc-f0-kills", label=C.label(digest), exploratory=True, passmarks_sha256=digest,
                smoke_test="SMOKE TEST: a stand-in, not results", complete=True, missing=[], survivors=sv,
                inputs_sha256={},
                A={q: (go if q == "medium-expert" else stop) for q in C.QUALITIES},
                B=dict(go_on=True, stopped_by=[], stops=[], rules=dict(K3=dict(where=[])),
                       seeds_usable={"medium-expert": [0]}),
                C={q: (go if q == "random" else stop) for q in C.QUALITIES},
                C_rules=dict(K3=dict(where=[], fires=False), stopped_by=[], KB0=dict(holds=False),
                             guard_void=dict(void=False), B2_B4_scored=False),
                F0={q: dict(decided=False, note="smoke test") for q in C.F0_QUALITIES})


def main():
    digest = C.require_passmarks()
    if "--smoke-stub" in sys.argv[1:]:
        C.write_record(C.docs() / "KILLS.json", smoke_stub(digest))
        C.log(f"kills: smoke-test stand-in -> {C.docs() / 'KILLS.json'}")
        return 0
    K = build(digest)
    C.write_record(C.docs() / "KILLS.json", K)
    C.write_atomic(C.docs() / "KILLS.md", markdown(K))
    C.log(f"kills: complete {K['complete']} -> {C.docs() / 'KILLS.md'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.Refused as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
