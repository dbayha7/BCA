"""Step 4 pilot report: the items of Addendum A part 2 (runs/wbcp_signal/step4/expectations.md, addendum slot A).

Reads the pilot's run_step4.py directories (replicate 99; OUTPUT/<block>/<level>/rep99), Stage 0's stage0.json and the
pilot_power.json files that score_values.py --pilot-power wrote. It never computes J. Every J-based number (the power
check, J* and J(K_0)) is read from pilot_power.json, which holds J only for the power check's whitelisted rows: K_0, NONE,
EXACT, the P0 ladder and the matched OR1 / OR2 / SHUF rows of the weighting placements, in block C's localized cells
and X-CS's cone cells.

Items (expectations Part 0 unless noted):
- hashes: every pilot directory's HASHES.sha256 is verified before anything is read;
- G0: every built-in check of matched_knobs.json, tallied by name and kind (a check of kind 'eps' fails G0 only above
  100x its tolerance, as the scorecard reads it);
- item 3: the cone's anisotropy cosine and |corr(sigma, |s|^2)|, and its c0, from stage0.json;
- item 5: the fixed-point Hessian's definiteness (NONE row) per cell, and the share of nonfinite fixed points among
  every kept FP row of blocks X-CS, X-EP and C;
- item 6: J*, J(K_0) (pilot_power.json) and lambda_0 in clean X-CS (matched_knobs.json);
- item 7: the empirical cos(BC pull, ascent) in X-CS (stage0.json, clean cells) against B5's closed-form values, and
  item 2.11's re-fixed cell list;
- item 8: the power check, C = G(OR2) - G(SHUF) (SHUF averaged over its permutations) and I = G(OR1) - G(SHUF), at FP
  and expert, per cell and pooled over block C's two localized cases, for the main pilot and any retune roots;
- item 9: the outcome-free targeting share of SIG-P1L at matched S2;
- Part 2 items 9-10: reachability of S1-S3 per placement and regime, below-floor targets, the S1 fallback rule, and
  S(SIG)/S(SHUF) at beta = 1 for P2L;
- timing: seconds per process and per regime, and a projection for the main run.

python experiments/signal/pilot_report.py --pilot runs/wbcp_signal/step4/pilot \
    --stage0 runs/wbcp_signal/step4/stage0.json [--retune ROOT ...] --output runs/wbcp_signal/step4/pilot_report
Writes OUTPUT.json and OUTPUT.md.
"""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

PLACEMENTS = ("P1L", "P2L", "P1pi", "P2pi", "P3")
LEVELS = ("S1", "S2", "S3")
POWER_PLACEMENTS = ("P1L", "P2L", "P1pi", "P2pi")
LOCALIZED = ("q1opt-local", "tilt-local")
B5_COS = {"expert": 0.99, "mixed": 0.53, "medium": -0.11, "poor": -0.24}  # expectations B5, X-CS closed form
TILT_A = (0.9, 0.5, 0.0, -0.5)
X_BLOCKS = ("X-CS", "X-EP", "C")
SUBSET_REPLICATES, ANALYSIS_REPLICATES = 5, 10  # run_step4: subsets on replicates 0-4 of 0-9


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def processes(root):
    """Every process directory under root, its hashes verified first."""
    out = []
    for knobs in sorted(Path(root).glob("*/*/rep*/matched_knobs.json")):
        d = knobs.parent
        recorded = {}
        for line in (d / "HASHES.sha256").read_text().splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                recorded[name.strip().lstrip("*")] = digest
        for name in ("matched_knobs.json", "arrays.npz"):
            if recorded.get(name) != sha256(d / name):
                raise SystemExit(f"{d}: {name} differs from HASHES.sha256; nothing is read")
        out.append(dict(dir=d, payload=json.loads(knobs.read_text(encoding="utf-8")), arrays=np.load(d / "arrays.npz")))
    return out


def stage0_sets(stage0):
    """(block, cell, level) -> evidence set."""
    return {(b, cell, lev): rec["set"] for b, cells in stage0["sets"].items() for cell, levels in cells.items()
            for lev, rec in levels.items()}


def checks_tally(procs):
    tally = {}
    for p in procs:
        for rec in p["payload"]["cells"].values():
            for c in rec["checks"]:
                t = tally.setdefault(c["name"], dict(kind=c["kind"], items=set(), n=0, failed=0, worst=0.0))
                t["items"].add(c.get("item"))
                t["n"] += 1
                ok = c["ok"]
                if c["kind"] == "eps" and not ok and c.get("value") is not None and c.get("tol"):
                    ok = c["value"] <= 100 * c["tol"]  # the scorecard's G0 reading of an 'eps' check
                t["failed"] += 0 if ok else 1
                if c.get("value") is not None and c.get("tol"):
                    t["worst"] = max(t["worst"], float(c["value"]) / float(c["tol"]))
    return {k: dict(v, items=sorted(i for i in v["items"] if i)) for k, v in sorted(tally.items())}


def reachability(procs, sets):
    """Per (regime, placement, level): cells reached / unreachable / below floor, for SIG; the S1 fallback rule."""
    counts = defaultdict(lambda: dict(reached=0, unreachable=0, below_floor=0))
    m_cells = defaultdict(lambda: dict(n=0, s2_missing=0))
    floor_cells = dict(n=0, below=defaultdict(int), by_block=defaultdict(lambda: defaultdict(int)),
                       n_by_block=defaultdict(int), x_cells=defaultdict(list))
    for p in procs:
        meta = p["payload"]["meta"]
        for cid, rec in p["payload"]["cells"].items():
            for reg, regime in rec["regimes"].items():
                idx = regime["index"]
                if reg == "FP":
                    floor_cells["n"] += 1
                    floor_cells["n_by_block"][meta["block"]] += 1
                    usable = idx["targets"].get("usable", {})
                    for lev in LEVELS:
                        if usable.get(lev) is None:
                            floor_cells["below"][lev] += 1
                            floor_cells["by_block"][meta["block"]][lev] += 1
                            if meta["block"] in ("X-CS", "X-EP"):
                                floor_cells["x_cells"][lev].append(f"{meta['block']}/{meta['level']}/{cid}")
                for pl in PLACEMENTS:
                    m = idx["matched"].get(f"{pl}:SIG")
                    if m is None:
                        continue
                    for lev in LEVELS:
                        r = m.get(lev)
                        if r is None:
                            continue
                        key = (meta["block"], reg, pl, lev)
                        if r.get("no_target"):
                            counts[key]["below_floor"] += 1
                        elif r.get("reachable") and r.get("row") is not None:
                            counts[key]["reached"] += 1
                        else:
                            counts[key]["unreachable"] += 1
                    if meta["block"] == "X-CS" and sets.get(("X-CS", cid, meta["level"])) == "M":
                        r = m.get("S2") or {}
                        mc = m_cells[(reg, pl)]
                        mc["n"] += 1
                        mc["s2_missing"] += int(bool(r.get("no_target")) or not r.get("reachable"))
    fallback = {f"{reg}/{pl}": dict(v, fallback=v["n"] > 0 and v["s2_missing"] > v["n"] / 2)
                for (reg, pl), v in sorted(m_cells.items())}
    table = {"/".join(k): v for k, v in sorted(counts.items())}
    floor = dict(n=floor_cells["n"], below=dict(floor_cells["below"]), n_by_block=dict(floor_cells["n_by_block"]),
                 by_block={b: dict(v) for b, v in floor_cells["by_block"].items()},
                 x_cells={k: v for k, v in floor_cells["x_cells"].items()})
    return dict(sig=table, fallback=fallback, floor=floor)


def overall_reach(sig, blocks=None):
    """Share of SIG cell-regimes reaching each level, pooled over regimes (and blocks, or only `blocks`)."""
    agg = defaultdict(lambda: dict(reached=0, unreachable=0, below_floor=0))
    for key, v in sig.items():
        block, reg, pl, lev = key.split("/")
        if blocks is not None and block not in blocks:
            continue
        for k in v:
            agg[(pl, lev)][k] += v[k]
    out = {}
    for (pl, lev), v in sorted(agg.items()):
        targeted = v["reached"] + v["unreachable"]
        out[f"{pl}/{lev}"] = dict(v, share_reached=v["reached"] / targeted if targeted else None)
    return out


def fp_health(procs):
    rows, r_not_pd = [], []
    finite = dict(n=0, nonfinite=0)
    for p in procs:
        meta = p["payload"]["meta"]
        if meta["block"] not in X_BLOCKS:
            for cid, rec in p["payload"]["cells"].items():
                fp = p["arrays"][f"D/{cid}/FP/fp"]
                none_row = rec["regimes"]["FP"]["index"]["NONE"]
                if not fp[none_row, 0] > 0 or fp[none_row, 2] == 0:
                    r_not_pd.append(dict(level=meta["level"], cell=cid, none_min_eig=float(fp[none_row, 0])))
            continue
        for cid, rec in p["payload"]["cells"].items():
            fp = p["arrays"][f"D/{cid}/FP/fp"]  # min_eig, cond, finite, bfgs_success per kept FP row
            none_row = rec["regimes"]["FP"]["index"]["NONE"]
            rows.append(dict(block=meta["block"], level=meta["level"], cell=cid,
                             none_min_eig=float(fp[none_row, 0]), pd=bool(fp[none_row, 0] > 0)))
            fin = fp[:, 2]
            fin = fin[~np.isnan(fin)]
            finite["n"] += int(fin.size)
            finite["nonfinite"] += int(np.sum(fin == 0))
    return dict(cells=rows, not_pd=[r for r in rows if not r["pd"]], block_r_not_pd=r_not_pd, nonfinite=finite,
                nonfinite_share=finite["nonfinite"] / finite["n"] if finite["n"] else None)


def targeting_share(procs):
    out = []
    for p in procs:
        meta = p["payload"]["meta"]
        for cid, rec in p["payload"]["cells"].items():
            if not (meta["block"] == "C" and rec["case"] in LOCALIZED) and not (
                    meta["block"] in ("X-CS", "X-EP") and rec["case"] == "q1_optimistic"):
                continue
            m = rec["regimes"]["FP"]["index"]["matched"].get("P1L:SIG", {}).get("S2") or {}
            row = m.get("row") if m.get("reachable") and not m.get("no_target") else None
            share = None if row is None else float(p["arrays"][f"D/{cid}/FP/frontier"][row, 1])
            out.append(dict(block=meta["block"], level=meta["level"], cell=cid, case=rec["case"], share=share))
    return out


def s_ratio(procs):
    """Part 2 item 10: S(SIG)/mean S(SHUF) at beta = 1 for the trust placements (P2L, P2pi), per regime: over block X's
    non-control cells (X-CS and X-EP without clean, noisy_reward and the cone, the forecast's cells) and, for reference,
    over every non-negative-control cell of every block."""
    out = defaultdict(list)
    for p in procs:
        block = p["payload"]["meta"]["block"]
        for cid, rec in p["payload"]["cells"].items():
            if rec["case"] in ("clean", "noisy_reward"):
                continue
            scope = ["all"] + (["X"] if block in ("X-CS", "X-EP") and rec["case"] != "cone" else [])
            for reg in ("FP", "A500"):
                regime = rec["regimes"].get(reg)
                if regime is None:
                    continue
                S = p["arrays"][f"D/{cid}/{reg}/S"]
                for pl in ("P2L", "P2pi"):
                    rows = defaultdict(list)
                    for i, (arm, knob, kind) in enumerate(regime["labels"]):
                        if kind == "ladder" and knob == 1.0 and arm.startswith(f"{pl}:") and (
                                arm == f"{pl}:SIG" or "SHUF" in arm):
                            rows["SIG" if arm == f"{pl}:SIG" else "SHUF"].append(float(S[i]))
                    if rows["SIG"] and rows["SHUF"] and statistics.mean(rows["SHUF"]) > 0:
                        for sc in scope:
                            out[f"{sc}/{pl}/{reg}"].append(rows["SIG"][0] / statistics.mean(rows["SHUF"]))
    return {k: dict(n=len(v), in_range=sum(1.0 <= x <= 1.4 for x in v), share=sum(1.0 <= x <= 1.4 for x in v) / len(v),
                    median=statistics.median(v)) for k, v in sorted(out.items()) if v}


def stage0_items(stage0):
    rows = stage0["rows"]
    cone = defaultdict(lambda: dict(cos=[], corr=[], c0=[]))
    pull = defaultdict(list)
    redraws = defaultdict(dict)
    for r in rows:
        if r["block"] in ("X-CS", "X-EP") and r["case"] == "cone":
            key = (r["block"], r["level"], r.get("kappa"))
            cone[key]["cos"].append(r.get("cone_cos"))
            cone[key]["corr"].append(r.get("cone_corr_s2"))
            cone[key]["c0"].append(r.get("cone_c0"))
        if r["block"] == "X-CS" and r["case"] == "independent_errors":
            redraws[r["replicate"]][r["level"]] = r.get("independent_redraws")
        if r["block"] == "X-CS" and r["case"] == "clean" and r.get("cos_pull_ascent") is not None:
            pull[r["level"]].append(r["cos_pull_ascent"])
    cone_out = {}
    for (blk, lev, kap), v in sorted(cone.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or 0)):
        vals = {k: [x for x in xs if x is not None] for k, xs in v.items()}
        entry = {k: dict(mean=statistics.mean(xs), min=min(xs), max=max(xs)) for k, xs in vals.items() if xs}
        if vals["cos"]:
            entry["cos_outside_band"] = sum(not 0.88 <= x <= 0.96 for x in vals["cos"])
            entry["n"] = len(vals["cos"])
        cone_out[f"{blk}/{lev}/kappa{kap:g}"] = entry
    cos_emp = {lev: statistics.mean(v) for lev, v in pull.items()}
    item7 = {lev: dict(empirical=cos_emp.get(lev), closed_form=B5_COS[lev],
                       within=None if lev not in cos_emp else abs(cos_emp[lev] - B5_COS[lev]) <= 0.05) for lev in B5_COS}
    refixed = []
    for lev in B5_COS:
        if lev not in cos_emp:
            continue
        for a in TILT_A:
            diff = cos_emp[lev] - a
            if abs(diff) >= 0.3:
                refixed.append(dict(level=lev, a_star=a, cos_bc=cos_emp[lev], sign="+" if diff > 0 else "-"))
    differ = sorted(rep for rep, levels in redraws.items() if len(set(levels.values())) > 1)
    amendment_l = dict(by_replicate={str(k): v for k, v in sorted(redraws.items())}, differing_replicates=differ)
    return dict(cone=cone_out, item7=item7, item2_11_cells=refixed, amendment_l=amendment_l)


def power(root, label):
    """Per pilot_power.json: C and I per cell, placement and S_k at FP; pooled over block C's localized cases."""
    cells = []
    for path in sorted(Path(root).glob("*/*/rep*/pilot_power.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        block, level = path.parts[-4], path.parts[-3]
        for cid, rec in d["cells"].items():
            fp = rec["regimes"]["FP"]
            entry = dict(root=label, block=block, level=level, cell=cid, case=rec["case"], kappa=rec.get("kappa"),
                         J_opt=rec["J_opt"], J0=rec["J0"], gap=rec["gap"], G_exact=fp.get("EXACT", {}).get("G"),
                         G_p0_max=max((v["G"] for k, v in fp.items() if k.startswith("P0:") and v["G"] is not None),
                                      default=None),
                         nonfinite_rows=sorted(k for k, v in fp.items() if v.get("G") is None), contrasts={})
            for pl in POWER_PLACEMENTS:
                for lev in LEVELS:
                    shuf_all = [v["G"] for k, v in fp.items() if k.startswith(f"{pl}:SHUF") and k.endswith(f"@{lev}")]
                    shuf = [g for g in shuf_all if g is not None]  # a nonfinite row is dropped and counted below
                    or1, or2 = fp.get(f"{pl}:OR1@{lev}", {}).get("G"), fp.get(f"{pl}:OR2@{lev}", {}).get("G")
                    if not shuf:
                        continue
                    s = statistics.mean(shuf)
                    entry["contrasts"][f"{pl}@{lev}"] = dict(
                        shuf=s, n_shuf=len(shuf), n_shuf_lost=len(shuf_all) - len(shuf),
                        C=None if or2 is None else or2 - s, I=None if or1 is None else or1 - s)
            cells.append(entry)
    pooled = {}
    for data in ("expert", "poor"):
        for pl in POWER_PLACEMENTS:
            for lev in LEVELS:
                vals = [c["contrasts"].get(f"{pl}@{lev}", {}).get("C") for c in cells
                        if c["block"] == "C" and c["level"] == data and c["case"] in LOCALIZED]
                vals = [v for v in vals if v is not None]
                if vals:
                    pooled[f"{data}/{pl}@{lev}"] = dict(C=statistics.mean(vals), n=len(vals))
    return dict(cells=cells, pooled_block_c=pooled)


def timing(procs):
    per_process, per_regime = [], defaultdict(float)
    projection = 0.0
    for p in procs:
        meta = p["payload"]["meta"]
        sub = 0.0
        for rec in p["payload"]["cells"].values():
            for reg, sec in rec["timing"].items():
                if reg != "total":
                    per_regime[reg] += sec
                if reg in ("A5000", "FB"):
                    sub += sec
        total = float(meta["seconds"])
        nosub = total - sub if meta.get("subsets") else total
        per_process.append(dict(block=meta["block"], level=meta["level"], seconds=total, subset_seconds=sub,
                                no_subset_seconds=nosub))
        projection += SUBSET_REPLICATES * total + (ANALYSIS_REPLICATES - SUBSET_REPLICATES) * nosub
    reduced = sum(SUBSET_REPLICATES * r["seconds"] + (ANALYSIS_REPLICATES - SUBSET_REPLICATES) * r["no_subset_seconds"]
                  if r["block"] in ("X-CS", "C") else ANALYSIS_REPLICATES * r["no_subset_seconds"] for r in per_process)
    return dict(per_process=per_process, per_regime=dict(per_regime),
                pilot_process_hours=sum(r["seconds"] for r in per_process) / 3600,
                main_run_process_hours=projection / 3600, reduced_plan_upper_bound_process_hours=reduced / 3600)


def retune_summary(root):
    """A retune root's checks and its cells' NONE fixed-point definiteness (not part of the pilot's G0)."""
    cells = []
    for p in processes(root):
        for cid, rec in p["payload"]["cells"].items():
            fp = p["arrays"][f"D/{cid}/FP/fp"]
            index = rec["regimes"]["FP"]["index"]
            eig = fp[:, 0]
            cells.append(dict(cell=cid, none_min_eig=float(fp[index["NONE"], 0]), exact_min_eig=float(fp[index["EXACT"], 0]),
                              rows_not_pd=int(np.sum(~(eig > 0))), rows=int(eig.size),
                              failed_checks=[c["name"] for c in rec["checks"] if not c["ok"]]))
    return dict(root=str(root), cells=cells)


def lam0_clean(procs):
    return {p["payload"]["meta"]["level"]: p["payload"]["cells"]["clean"]["lam0"] for p in procs
            if p["payload"]["meta"]["block"] == "X-CS" and "clean" in p["payload"]["cells"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--stage0", required=True)
    ap.add_argument("--retune", nargs="*", default=[], help="roots of pre-registered retune pilots (power check only)")
    ap.add_argument("--output", required=True, help="path stem for .json and .md")
    a = ap.parse_args(argv)
    stage0 = json.loads(Path(a.stage0).read_text(encoding="utf-8"))
    procs = processes(a.pilot)
    sets = stage0_sets(stage0)
    reach = reachability(procs, sets)
    report = dict(
        inputs=dict(pilot=a.pilot, stage0=a.stage0, stage0_sha256=sha256(a.stage0), processes=len(procs),
                    retune=a.retune),
        checks=checks_tally(procs),
        stage0=stage0_items(stage0),
        fp=fp_health(procs),
        lambda0_clean_xcs=lam0_clean(procs),
        targeting_share=targeting_share(procs),
        reachability=dict(reach, overall=overall_reach(reach["sig"]),
                          block_x=overall_reach(reach["sig"], blocks=("X-CS", "X-EP"))),
        s_ratio_beta1=s_ratio(procs),
        power=[power(a.pilot, "pilot")] + [power(r, Path(r).name) for r in a.retune],
        retune=[retune_summary(r) for r in a.retune],
        timing=timing(procs),
    )
    out = Path(a.output)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    out.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(f"wrote {out.with_suffix('.json')} and .md; sha256 {sha256(out.with_suffix('.json'))}")


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(r):
    L = ["# Step 4 pilot report (replicate 99; outcome-free except the power check)", ""]
    failed = {k: v for k, v in r["checks"].items() if v["failed"]}
    L += [f"**G0 checks:** {sum(v['n'] for v in r['checks'].values())} checks of {len(r['checks'])} kinds; "
          f"{sum(v['failed'] for v in r['checks'].values())} failed" + (f": {sorted(failed)}" if failed else "."), ""]
    L += ["## Power check (FP, C = G(OR2) − mean G(SHUF), I = G(OR1) − mean G(SHUF); units of the gap)", ""]
    for pw in r["power"]:
        L += [f"### {pw['cells'][0]['root'] if pw['cells'] else '(none)'}", "",
              "| Block | Level | Cell | G(EXACT) | max G(P0) | P1L C@S2 | P2L C@S2 | P1L I@S2 | P2L I@S2 |", "|---|---|---|---|---|---|---|---|---|"]
        for c in pw["cells"]:
            g = lambda pl, k: c["contrasts"].get(f"{pl}@S2", {}).get(k)
            L.append(f"| {c['block']} | {c['level']} | {c['cell']} | {fmt(c['G_exact'])} | {fmt(c['G_p0_max'])} | "
                     f"{fmt(g('P1L', 'C'))} | {fmt(g('P2L', 'C'))} | {fmt(g('P1L', 'I'))} | {fmt(g('P2L', 'I'))} |")
        L += ["", "Pooled over block C's localized cells: " + ", ".join(
            f"{k} {fmt(v['C'])} (n={v['n']})" for k, v in pw["pooled_block_c"].items()), ""]
    st = r["stage0"]
    L += ["## Stage 0 items", "", "Item 7, cos(BC pull, ascent) in X-CS (clean): " + "; ".join(
        f"{lev} {fmt(v['empirical'], 3)} vs {v['closed_form']} ({'within' if v['within'] else 'outside'} ±0.05)"
        for lev, v in st["item7"].items()), "",
        "Item 2.11 re-fixed cells (|cos_BC − a*| ≥ 0.3): " + ", ".join(
            f"{c['level']} a*={c['a_star']:g} ({c['sign']})" for c in st["item2_11_cells"]) + f" — {len(st['item2_11_cells'])} cells", "",
        "Item 3, cone: " + "; ".join(f"{k}: cos {fmt(v['cos']['mean'], 3)} [{fmt(v['cos']['min'], 3)}, {fmt(v['cos']['max'], 3)}], "
                                      f"|corr| max {fmt(max(abs(v['corr']['min']), abs(v['corr']['max'])), 3)}, c0 {fmt(v['c0']['mean'], 3)}"
                                      for k, v in st["cone"].items()), ""]
    fp = r["fp"]
    L += ["## Fixed points (item 5)", "", f"NONE's fixed-point Hessian not PD in {len(fp['not_pd'])} of {len(fp['cells'])} X/C cells"
          + (": " + ", ".join(f"{c['block']}/{c['level']}/{c['cell']}" for c in fp["not_pd"]) if fp["not_pd"] else "") + ".",
          f"Nonfinite fixed points: {fp['nonfinite']['nonfinite']} of {fp['nonfinite']['n']} kept FP rows ({fmt(fp['nonfinite_share'], 3)}).", ""]
    L += ["## Targeting share of SIG-P1L at S2 (item 9)", "", "| Block | Level | Cell | Share |", "|---|---|---|---|"]
    L += [f"| {t['block']} | {t['level']} | {t['cell']} | {fmt(t['share'], 3)} |" for t in r["targeting_share"]] + [""]
    L += ["## Reachability (SIG), pooled over blocks and regimes", "", "| Placement/level | Reached | Unreachable | Below floor | Share reached |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['reached']} | {v['unreachable']} | {v['below_floor']} | {fmt(v['share_reached'], 3)} |"
          for k, v in r["reachability"]["overall"].items()]
    fl = r["reachability"]["floor"]
    L += ["", f"FP cells with S* below the floor: " + ", ".join(f"{k} {v} of {fl['n']}" for k, v in fl["below"].items())
          + "; by block " + json.dumps(fl["by_block"]) + " of " + json.dumps(fl["n_by_block"]) + ".",
          "Block X cells below the floor: " + json.dumps(fl["x_cells"]) + ".",
          "S1 fallback (X-CS M, S2 missing in > 50%): " + ", ".join(f"{k} {v['s2_missing']}/{v['n']}{' FALLBACK' if v['fallback'] else ''}"
                                                                    for k, v in r["reachability"]["fallback"].items()), ""]
    sr = r["s_ratio_beta1"]
    L += ["S(SIG)/S(SHUF) at β = 1 (Part 2 item 10; X = block X non-control cells): " + "; ".join(
        f"{k}: {v['in_range']} of {v['n']} in [1.0, 1.4] ({fmt(v['share'], 2)}), median {fmt(v['median'], 3)}" for k, v in sr.items()), ""]
    t = r["timing"]
    L += ["## Timing", "", f"Pilot: {fmt(t['pilot_process_hours'], 2)} process-hours. Main-run projection (replicates 0-9, "
          f"subsets on 0-4): {fmt(t['main_run_process_hours'], 1)} process-hours; reduced plan at most "
          f"{fmt(t['reduced_plan_upper_bound_process_hours'], 1)}.", "",
          "| Block | Level | Seconds | Subset seconds |", "|---|---|---|---|"]
    L += [f"| {p['block']} | {p['level']} | {p['seconds']:.0f} | {p['subset_seconds']:.0f} |" for p in t["per_process"]]
    L += ["", "Per regime (seconds, summed over cells): " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(t["per_regime"].items())), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
