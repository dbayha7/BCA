"""EXPLORATORY: arms A, B, C and F0 -- the seed-0 screen report (frozen pass marks, section 8 step 4).

Reads every seed-0 record of this run (arm records must carry the label and this frozen file's digest), the D1 checks
of the three qualities whose data seeds 1-4 this run built, result F's seed-0 A2 evaluation and the rerun's no-caution
actor's evaluation, applies the kill rules that the screen can check, and writes routes2/arms/SCREEN_seed0.json
(+ .sha256) and SCREEN_seed0.md (SCREEN.md, the name the frozen file uses, links to it). It lists every kill quantity
and which arms and qualities go on to seeds 1-4. No final return of A, B or C is shown or computed (the screen has
none); F0's seed-0 score is shown, because no later rule guards it.

  python -B screen.py
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import common as C  # noqa: E402

SEED = 0


def pts(ev, law="strong"):
    return None if ev is None else (ev.get(law) or {}).get("points")


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x):
    return "-" if x is None else f"{100.0 * float(x):.1f}%"


def build(digest):
    out = dict(kind="arms-abc-f0-screen", label=C.label(digest), exploratory=True, passmarks_sha256=digest,
               seed=SEED, made_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), data={}, A={}, B={}, C={}, F0={},
               missing=[], inputs_sha256={})
    missing = out["missing"]

    def rec_of(arm, q):
        p = C.arm_dir(arm, q, SEED) / "record.json"
        try:
            r = C.arm_record(p)
            out["inputs_sha256"][str(p)] = C.sha256_file(p)
            return r
        except C.Refused:
            return None

    # data seeds 1-4 and D1 for the three qualities built here
    for q in ("random", "medium-replay", "medium"):
        d1 = None
        p = C.state() / f"d1_{q}.json"
        if p.is_file():
            d1 = json.loads(p.read_text())
            out["inputs_sha256"][str(p)] = C.sha256_file(p)
        out["data"][q] = dict(d1_identical=None if d1 is None else d1.get("all_identical"),
                              report_D1=None if d1 is None else d1.get("sandbox_report_D1_status"),
                              band_fits_allowed=None if d1 is None else d1.get("allows_band"))

    # arm A
    for q in C.QUALITIES:
        r = rec_of("A", q)
        blocked = q in out["data"] and out["data"][q]["band_fits_allowed"] is False
        if r is None:
            out["A"][q] = dict(status="blocked" if blocked else "missing",
                               reason="D1 differs or the report's D1 did not pass: band fits wait for the owner"
                               if blocked else "no record")
            if not blocked:
                missing.append(f"A {q}")
            continue
        k = r.get("kills") or {}
        h0 = (r.get("history") or [{}])[0].get("arms", {}).get("A1", {}) if r.get("history") else {}
        out["A"][q] = dict(status=r["status"], stopped_by=r.get("stopped_by"), reason=r.get("reason"),
                           K0=k.get("K0"), K1=k.get("K1"), K2_seed0=k.get("K2_value"),
                           K3=r.get("stopped_by") == "K3",
                           s=(r.get("inputs") or {}).get("s"), unit=(r.get("inputs") or {}).get("unit"),
                           U_mean_over_s=(r.get("inputs") or {}).get("U_mean_over_s"),
                           iw_table_matches_band_record=((r.get("inputs") or {}).get("iw") or {}).get(
                               "matches_band_record_table"),
                           refresh0_A1=h0)
        if r["status"] == "finished" and k.get("K2_value") is None:
            missing.append(f"A {q}: K2's seed-0 value")

    # arm B
    k2_pairs = []
    k3_count = 0
    for q in C.QUALITIES:
        r = rec_of("B", q)
        if r is None:
            out["B"][q] = dict(status="missing")
            missing.append(f"B {q}")
            continue
        k = r.get("kills") or {}
        rules = ((r.get("details") or {}).get("rules") or {}).get("bca")
        out["B"][q] = dict(status=r["status"], stopped_by=r.get("stopped_by"), reason=r.get("reason"),
                           bca_map=(r.get("maps") or {}).get("bca"), maps=r.get("maps"),
                           K1=k.get("K1"), over_refusal=k.get("K2_over_refusal"), K3=k.get("K3"),
                           K4_share=k.get("K4_clip_share_thruster1"), K0=k.get("K0_room"),
                           tries=(r.get("tries") or {}).get("rows"),
                           thruster1_tries_per_rung=[x["tries"] for x in rules[0]["rungs"]] if rules else None)
        if k.get("K2_over_refusal"):
            k2_pairs.append(f"{q} seed 0")
        if (k.get("K3") or {}).get("fires"):
            k3_count += 1
    me = out["B"].get("medium-expert", {})
    k3_where = [f"{q} seed 0" for q, v in out["B"].items() if (v.get("K3") or {}).get("fires")]
    b_rules = dict(
        K0=dict(value=(me.get("K0") or {}).get("room"), fires=None if me.get("K0") is None else
                bool(me["K0"]["room"] < 1.0), rule="(host + knee clip) - host, strong test, medium-expert, below +1.0"),
        K1=dict(fires=any(bool((v.get("K1") or {}).get("fires")) for v in out["B"].values()),
                where=[q for q, v in out["B"].items() if (v.get("K1") or {}).get("fires")],
                rule="BCA's thruster-1 ceiling 0.85 or higher with at least 30 tries in rung (0.80, 0.85]"),
        K2=dict(pairs=k2_pairs, fires=len(k2_pairs) >= 2,
                rule="BCA's thruster-1 ceiling 0.60 or lower in 2 or more seed-quality pairs"),
        K3=dict(count=k3_count, where=k3_where, fires=k3_count >= 2,
                rule="two seeds with a non-finite gain, weight, actor or return"),
        K4=dict(value=me.get("K4_share"), fires=None if me.get("K4_share") is None else bool(me["K4_share"] < 0.02),
                rule="BCA's map clips under 2% of the host's thruster-1 commands (strong starts, medium-expert)"))
    out["B_rules"] = b_rules
    b_stopped = [n for n, v in b_rules.items() if v.get("fires")]

    # arm C
    kb0_pass = []
    for q in C.QUALITIES:
        r = rec_of("C", q)
        if r is None:
            out["C"][q] = dict(status="missing")
            missing.append(f"C {q}")
            continue
        k = r.get("kills") or {}
        out["C"][q] = dict(status=r["status"], stopped_by=r.get("stopped_by"), reason=r.get("reason"),
                           KB1=k.get("KB1"), K3=k.get("K3") or (dict(fires=True) if r.get("stopped_by") == "K3"
                                                                 else None),
                           KB0_seed=k.get("KB0_seed"), guard_void_own_map=k.get("guard_void_own_map"),
                           guard_map=(r.get("maps") or {}).get("bca"))
        if q in C.WEAK and (k.get("KB0_seed") or {}).get("passes"):
            kb0_pass.append(q)
    c_k3 = [f"{q} seed 0" for q, v in out["C"].items() if (v.get("K3") or {}).get("fires")]
    out["C_rules"] = dict(
        K3=dict(where=c_k3, count=len(c_k3), fires=len(c_k3) >= 2,
                rule="two seeds with a non-finite gain, weight, actor or return stop C (as in B)"),
        KB0=dict(passing=kb0_pass, holds=len(kb0_pass) >= 2,
                 rule="rho 0.5's share of thruster-1 commands past 80% at least 2% and at least twice rho 1's, on at "
                      "least 2 of random, medium-replay, medium (seed 0); if it fails, B2-B4 are not scored"),
        guard_void=dict(by_B=bool(b_rules["K1"]["fires"] or b_rules["K2"]["fires"]),
                        by_own_map=[q for q, v in out["C"].items() if (v.get("guard_void_own_map") or {}).get("fires")]))

    # F0
    for q in C.F0_QUALITIES:
        r = rec_of("F0", q)
        a2_ev = None
        p = C.f1_out() / q / f"evaluate_seed{SEED}.json"
        if p.is_file():
            a2_ev = (json.loads(p.read_text()).get("evaluations") or {}).get("A2_cmd_strong")
            out["inputs_sha256"][str(p)] = C.sha256_file(p)
        le = C.read_record(C.v5_root() / q / "evaluate" / f"seed{SEED}" / "room_lower_edge.json")
        le_ev = ((le or {}).get("evaluations") or {}).get("le_none")
        if r is None:
            out["F0"][q] = dict(status="missing")
            missing.append(f"F0 {q}")
            continue
        f0_ev = r.get("evaluation")
        out["F0"][q] = dict(status=r["status"], stopped_by=r.get("stopped_by"), reason=r.get("reason"),
                            checks=r.get("checks"), F0=f0_ev, A2=a2_ev, le_none=le_ev,
                            A2_minus_F0_strong=None if f0_ev is None or a2_ev is None else
                            pts(a2_ev) - pts(f0_ev),
                            F0_minus_le_none_strong=None if f0_ev is None or le_ev is None else
                            pts(f0_ev) - pts(le_ev),
                            A2_minus_le_none_strong=None if a2_ev is None or le_ev is None else
                            pts(a2_ev) - pts(le_ev),
                            no_credit_share_per_chunk=[c.get("no_credit_share") for c in r.get("chunks", [])],
                            a2_no_credit_share_per_chunk=r.get("a2_no_credit_share_per_chunk"))

    # survivors (section 3.3, 4.5, 5.5, 6.4, 7.3)
    sv = dict(A=[q for q, v in out["A"].items() if v.get("status") == "finished"],
              B=[] if b_stopped else [q for q, v in out["B"].items() if v.get("status") == "finished"
                                      and not (v.get("K3") or {}).get("fires")],
              C=[] if out["C_rules"]["K3"]["fires"] else
              [q for q, v in out["C"].items() if v.get("status") == "finished" and not (v.get("K3") or {}).get("fires")],
              F0=[q for q, v in out["F0"].items() if v.get("status") == "finished"])
    out["survivors"] = sv
    out["B_stopped_by"] = b_stopped
    out["complete"] = not missing
    return out


def markdown(s):
    L = []
    w = L.append
    w("# Seed-0 screen: arms A, B, C and F0 (EXPLORATORY)")
    w("")
    w(f"**EXPLORATORY.** {s['label']}. Data seed 0 is a seen seed. Nothing here is a confirmatory result.")
    w("")
    w(f"Made {s['made_at']}. Pass marks and kill rules: `routes2/arms/PASSMARKS_arms_ABC_and_F_control.md` "
      f"(SHA-256 `{s['passmarks_sha256']}`), checked before this report was written.")
    w("")
    w("This screen computes only what the kill rules read, plus arm B's room check. **No final return of arm A, B or "
      "C is shown or computed.** F0's seed-0 score is shown, because no later rule guards it.")
    w("")
    w(f"**Complete:** {'yes' if s['complete'] else 'no: ' + '; '.join(s['missing'])}.")
    w("")
    sv = s["survivors"]
    w("## What goes on to seeds 1-4")
    w("")
    w(f"- **Arm A** (credible-target pull): {', '.join(sv['A']) or 'none'}.")
    w(f"- **Arm B** (certified throttle ceiling): {', '.join(sv['B']) or 'none'}"
      + (f" (stopped by {', '.join(s['B_stopped_by'])})" if s["B_stopped_by"] else "") + ".")
    w(f"- **Arm C** (guarded release): {', '.join(sv['C']) or 'none'}.")
    w(f"- **F0** (result F's no-band control) and result F's A2: {', '.join(sv['F0']) or 'none'}.")
    w("")
    w("## Data for random, medium-replay and medium (seeds 1-4)")
    w("")
    w("| Quality | D1 identical to the committed report (seeds 0-4) | Report says D1 | Band fits allowed |")
    w("|---|---|---|---|")
    for q, v in s["data"].items():
        w(f"| {q} | {fmt(v['d1_identical'])} | {v['report_D1'] or '-'} | {fmt(v['band_fits_allowed'])} |")
    w("")
    w("## Arm A: kill checks")
    w("")
    w("K0 is information only on seed 0 (change 1). K1 and K3 bind on every seed. K2 is judged once all five seeds have "
      "reached 8,000 steps; seed 0's value is recorded here.")
    w("")
    w("| Quality | Status | K0: recorder rows released (at most 10%) | K0: random rows released (at least 15%) | "
      "K0 would pass | K1: rows released at 4,000 (above 60% fires) | K1: mean D / mean U (above 3 fires) | "
      "K2 seed 0: A1 - none at 8,000 (points) | s | mean U / s |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for q, v in s["A"].items():
        k0, k1 = v.get("K0") or {}, v.get("K1") or {}
        ratio = None
        if k1.get("U_mean"):
            ratio = k1["D_mean"] / k1["U_mean"]
        st = v.get("status") + (f" ({v.get('stopped_by')})" if v.get("stopped_by") else "")
        w(f"| {q} | {st} | {pct(k0.get('recorder_share'))} | {pct(k0.get('random_share'))} | "
          f"{fmt(k0.get('passes')) if k0.get('applies') else 'not checked'} | {pct(k1.get('released_share'))} | "
          f"{fmt(ratio)} | {fmt((v.get('K2_seed0') or {}).get('d'))} | {fmt(v.get('s'), 3)} | "
          f"{fmt(v.get('U_mean_over_s'), 3)} |")
    w("")
    w("## Arm B: kill checks")
    w("")
    br = s["B_rules"]
    w(f"- **K0, room** ({br['K0']['rule']}): {fmt(br['K0']['value'])} points; fires: {fmt(br['K0']['fires'])}.")
    w(f"- **K1** ({br['K1']['rule']}): fires: {fmt(br['K1']['fires'])} {br['K1']['where'] or ''}.")
    w(f"- **K2** ({br['K2']['rule']}): pairs so far: {br['K2']['pairs'] or 'none'}; fires: {fmt(br['K2']['fires'])}.")
    w(f"- **K3** ({br['K3']['rule']}): {br['K3']['count']}; fires: {fmt(br['K3']['fires'])}.")
    w(f"- **K4** ({br['K4']['rule']}): {pct(br['K4']['value'])}; fires: {fmt(br['K4']['fires'])}.")
    w("")
    w("| Quality | BCA ceilings (thrusters 1-4) | Thruster-1 tries per rung (0.50-0.55 ... 0.95-1.00) | Share of the host's "
      "thruster-1 commands the map clips |")
    w("|---|---|---|---|")
    for q, v in s["B"].items():
        if v.get("status") != "finished":
            w(f"| {q} | {v.get('status')}{' (' + str(v.get('stopped_by')) + ': ' + str(v.get('reason')) + ')' if v.get('stopped_by') else ''} | | |")
            continue
        w(f"| {q} | {', '.join(fmt(t) for t in v['bca_map'] or [])} | "
          f"{', '.join(str(t) for t in v['thruster1_tries_per_rung'] or [])} | {pct(v['K4_share'])} |")
    w("")
    w("## Arm C: kill checks")
    w("")
    cr = s["C_rules"]
    w(f"- **K3** ({cr['K3']['rule']}): {cr['K3']['where'] or 'none'}; fires: {fmt(cr['K3']['fires'])}.")
    w(f"- **KB0** ({cr['KB0']['rule']}): passing on {cr['KB0']['passing'] or 'none'}; holds: "
      f"{fmt(cr['KB0']['holds'])}.")
    w(f"- **Guard void:** by B's K1 or K2: {fmt(cr['guard_void']['by_B'])}; by C's own map: "
      f"{cr['guard_void']['by_own_map'] or 'none'}.")
    w("")
    w("| Quality | Status | KB1: G(rho 1) | KB1: G(rho 0.5) | KB1 bar | KB1 fires | rho 0.5 past 80% | rho 1 past 80% | "
      "Guard ceilings (thrusters 1-4) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for q, v in s["C"].items():
        kb1, kb0 = v.get("KB1") or {}, v.get("KB0_seed") or {}
        st = v.get("status") + (f" ({v.get('stopped_by')})" if v.get("stopped_by") else "")
        w(f"| {q} | {st} | {fmt(kb1.get('G_rho1'), 3)} | {fmt(kb1.get('G_rho05'), 3)} | {fmt(kb1.get('bar'), 3)} | "
          f"{fmt(kb1.get('fires'))} | {pct(kb0.get('share_rho05'))} | {pct(kb0.get('share_rho1'))} | "
          f"{', '.join(fmt(t) for t in v.get('guard_map') or [])} |")
    w("")
    w("## F0 (result F's no-band control): checks and seed-0 score")
    w("")
    w("No cross-machine bar is used. A2 and F0 are compared in this sandbox only. The mark needs all five seeds, so "
      "the seed-0 difference below decides nothing.")
    w("")
    w("| Quality | Status | KF1 disagreement | KF2 (a) / (b) | KF3 same clone | F0 strong points | A2 strong points | "
      "A2 - F0 | F0 - no extra caution | A2 - no extra caution |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for q, v in s["F0"].items():
        ch = v.get("checks") or {}
        kf2 = ch.get("KF2") or {}
        st = v.get("status") + (f" ({v.get('stopped_by')})" if v.get("stopped_by") else "")
        w(f"| {q} | {st} | {fmt((ch.get('KF1') or {}).get('disagreement'), 4)} | "
          f"{fmt((kf2.get('a') or {}).get('passes'))} / {fmt((kf2.get('b') or {}).get('passes'))} | "
          f"{fmt((ch.get('KF3') or {}).get('passes'))} | {fmt(pts(v.get('F0')))} | {fmt(pts(v.get('A2')))} | "
          f"{fmt(v.get('A2_minus_F0_strong'))} | {fmt(v.get('F0_minus_le_none_strong'))} | "
          f"{fmt(v.get('A2_minus_le_none_strong'))} |")
    w("")
    w("Seeds 1-4 start only with `bash run_arms.sh continue`, which refuses unless this screen is complete and verified.")
    w("")
    return "\n".join(L)


def main(argv=None):
    digest = C.require_passmarks()
    s = build(digest)
    C.write_record(C.docs() / "SCREEN_seed0.json", s)
    C.write_atomic(C.docs() / "SCREEN_seed0.md", markdown(s))
    link = C.docs() / "SCREEN.md"
    if not link.exists():
        os.symlink("SCREEN_seed0.md", link)
    C.log(f"screen: complete {s['complete']}; survivors {s['survivors']} -> {C.docs() / 'SCREEN_seed0.md'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.Refused as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
