"""Amended analysis of an existing Stage 1 attempt under the v3 rules (PROTOCOL_V3.md R1, R2, R3). Read-only on the
attempt: it opens JSON reports only (never a checkpoint or dataset), changes nothing in the attempt's folder, and
writes one separately labelled file under knee_edge_ood_v3/amended_analyses/. The attempt's original decision
record stays the record; this file is an amended analysis beside it, named as such.

What it computes from the attempt's evaluate and r0 reports:
- R1: every Stage 1 rule under v3's stage1_math, which adds V0(f); the v3 label the same inputs would have received,
  next to the original label and the original per-seed rules;
- R2: KC against the unclipped own-continuation reference, the per-anchor term D = KC_clip - KC_own with its identity
  check, and the K1a contrast's uncancelled term D_C - D_R;
- R3: the exact simulator-step accounting corrected for v2's leading-dimension undercount: every KneeDataset
  construction replays episodes x 100 transitions and v2 counted it as `episodes`; the number of constructions per
  job kind comes from the worker's code path (warmup, branch, bc_only, r0: one world; evaluate: one per world it
  evaluated), checked against the recorded count where the kind uses no other simulator path.

  /c/Users/David/dev/BCA/.venv/Scripts/python.exe -B amended_analysis.py <attempt directory> [--exploratory]
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

import stage1
import stage1_math as SM
import units as U
from core import require, sha_file
from stage1_worker import sanitize

OUT = HERE / "amended_analyses"
HORIZON = 100
TRAINING_KINDS = ("warmup", "branch", "bc_only")
LOADS_BY_KIND = {"warmup": 1, "branch": 1, "bc_only": 1, "r0": 1}          # KneeDataset constructions per job kind
SCHEMA = "knee_edge_ood_v3.amended_analysis.v1"


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")), sha_file(path)


def unit_reports(attempt_dir, report):
    """Every unit report of the attempt (reused units are read where the report's unit_dirs point)."""
    out = {}
    for unit, path in (report.get("unit_dirs") or {}).items():
        p = Path(path) / "report.json"
        if p.is_file():
            out[unit] = read(p)
    for p in sorted(Path(attempt_dir).iterdir()):                      # units the attempt ran but did not link
        if p.is_dir() and p.name not in out and (p / "report.json").is_file():
            out[p.name] = read(p / "report.json")
    return out


def corrected_steps(unit, rep, episodes):
    """R3: recorded exact steps plus (episodes x HORIZON - episodes) per dataset construction the job kind made."""
    kind = rep.get("kind") or unit.split("_")[0]
    recorded = rep.get("simulator_steps_exact")
    if recorded is None:
        return {"kind": kind, "recorded": None, "loads": None, "corrected": None, "status": "unknown: no exact count recorded"}
    per_load = episodes * HORIZON - episodes
    if kind in TRAINING_KINDS:
        loads, status = LOADS_BY_KIND[kind], "derived from the code path (training kinds use no other simulator path)"
        if recorded != loads * episodes:
            status = f"recorded {recorded} is not {loads} x {episodes}: the code path assumption does not hold; corrected is a lower bound"
    elif kind == "r0":
        loads, status = LOADS_BY_KIND[kind], "one stall3 construction (stage1_worker.r0 load_world)"
    elif kind == "evaluate":
        worlds_seen = {w for name in (rep.get("runs") or {}) for w in ("stall3", "nostall") if name.endswith(w)}
        loads, status = len(worlds_seen) or 2, f"one construction per evaluated world ({sorted(worlds_seen) or ['stall3', 'nostall']})"
    else:
        return {"kind": kind, "recorded": recorded, "loads": 0, "corrected": recorded, "status": "no dataset construction in this kind"}
    return {"kind": kind, "recorded": recorded, "loads": loads, "per_load_undercount": per_load,
            "corrected": int(recorded) + loads * per_load, "status": status}


def analyse(attempt_dir, *, exploratory=False):
    attempt_dir = Path(attempt_dir).resolve()
    report, report_sha = read(attempt_dir / "report.json")
    sources = {"attempt_report": {"path": str(attempt_dir / "report.json"), "sha256": report_sha}}
    units = unit_reports(attempt_dir, report)
    sources["unit_reports"] = {u: {"sha256": sha} for u, (_, sha) in units.items()}
    collection_dir = Path(report.get("a3", {}).get("collection_dir", ""))
    episodes = 1280
    if (collection_dir / "collection.json").is_file():
        collection, sha = read(collection_dir / "collection.json")
        sources["collection"] = {"path": str(collection_dir / "collection.json"), "sha256": sha}
        episodes = int(next(iter(collection.get("datasets", {}).values()), {}).get("episodes", episodes))

    # R3: accounting.
    accounting = {u: corrected_steps(u, rep, episodes) for u, (rep, _) in units.items()}
    known = [a for a in accounting.values() if a["corrected"] is not None]
    accounting_total = {"recorded_exact_total": sum(int(a["recorded"]) for a in known),
                        "corrected_total": sum(int(a["corrected"]) for a in known),
                        "units_without_counts": [u for u, a in accounting.items() if a["corrected"] is None],
                        "evaluation_cap": stage1.EVAL_TRANSITIONS,
                        "evaluation_units_corrected_total": sum(int(a["corrected"]) for u, a in accounting.items()
                                                                if a["corrected"] is not None and a["kind"] in ("r0", "evaluate")),
                        "episodes": episodes, "horizon": HORIZON}
    out = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).isoformat(), "attempt": str(attempt_dir),
           "status": "AMENDED ANALYSIS under PROTOCOL_V3.md; the attempt's original decision record is unchanged",
           "exploratory": bool(exploratory or "EXPLORATORY" in report),
           "original": {"label": report.get("label"), "protocol_label": report.get("protocol_label", report.get("label")),
                        "exploratory_label": report.get("exploratory_label"), "reasons": report.get("reasons"),
                        "reading": report.get("reading")},
           "sources": sources, "R3_accounting": {"units": accounting, "totals": accounting_total}}
    evaluations = {i: units[f"evaluate_s{i}"][0] for i in stage1.SEEDS if f"evaluate_s{i}" in units}
    if len(evaluations) != 3 or "r0" not in units:
        out["R1_R2"] = {"status": "not computed: the attempt has no complete evaluate_s0..2 and r0 reports"}
        return out
    r0 = units["r0"][0]
    training = {u: rep for u, (rep, _) in units.items() if u.startswith(("warmup_", "branch_"))}
    decision = stage1.decide(evaluations, r0, training)                # v3 math: V0(f) and the R2 block included
    out["R1_v3_rules"] = {"label_under_v3": decision["label"], "reasons_under_v3": decision["reasons"],
                          "reading_under_v3": decision["reading"],
                          "V0f_by_seed": {s: d["V0f"] for s, d in decision["per_seed"].items()},
                          "validity_by_seed": {s: {k: d[k]["ok"] for k in SM.VALIDITY} for s, d in decision["per_seed"].items()},
                          "criteria_by_seed": {s: {k: d[k]["ok"] for k in SM.CRITERIA} for s, d in decision["per_seed"].items()},
                          "note": "the v3 label is what these inputs receive under v3's rules; it is not the attempt's label"}
    out["R2_own_reference"] = {s: decision["reported"][s]["v3_own_reference"] for s in decision["reported"]}
    out["per_seed_v3"] = decision["per_seed"]
    return out


def write(result):
    OUT.mkdir(exist_ok=True)
    name = Path(result["attempt"]).parent.name + "__" + Path(result["attempt"]).name
    path = OUT / f"{name}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(sanitize(result), stream, indent=2, sort_keys=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attempt")
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()
    result = analyse(args.attempt, exploratory=args.exploratory)
    path = write(result)
    print(json.dumps({"written": str(path), "original_label": result["original"]["label"],
                      "label_under_v3": result.get("R1_v3_rules", {}).get("label_under_v3"),
                      "corrected_total_steps": result["R3_accounting"]["totals"]["corrected_total"]}, indent=2))
