"""Amended analysis of the pinned completed v2 Stage 1 attempt under the v3 rules (R1, R2, R3). Read-only on the
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
  evaluated plus one more stall3 construction for the BC-only summary when the report carries one), checked against
  the recorded count where the kind uses no other simulator path.

  /c/Users/David/dev/BCA/.venv/Scripts/python.exe -B amended_analysis.py <attempt directory> [--exploratory]
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
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
from core import REPO, require, sha_file
from stage1_worker import sanitize

OUT = HERE / "amended_analyses"
HORIZON = 100
TRAINING_KINDS = ("warmup", "branch", "bc_only")
LOADS_BY_KIND = {"warmup": 1, "branch": 1, "bc_only": 1, "r0": 1}          # KneeDataset constructions per job kind
SCHEMA = "knee_edge_ood_v3_guard_repair_v1.amended_analysis.v2"
CORRECTION_CONTRACT = HERE / "KNOWN_V2_CORRECTION.json"
CORRECTION_CONTRACT_SHA256 = "904e794aba1056b9c03ed875f9dba20a5ca9faacfd2f005e0d00bc61c6b74f96"


def correction_contract():
    require(sha_file(CORRECTION_CONTRACT) == CORRECTION_CONTRACT_SHA256,
            "Historical correction contract changed; review required")
    return json.loads(CORRECTION_CONTRACT.read_text(encoding="utf-8"))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def authenticate_attempt(attempt_dir):
    """Allow only the pinned, completed legacy attempt, never infer a producer from a count.

    Receipts and the saved producer source closure must still match. This establishes byte
    identity with the reviewed historical evidence; it is not independent runtime attestation.
    """
    contract = correction_contract()
    require(Path(attempt_dir).resolve() == (REPO / contract["attempt_relative"]).resolve(),
            "Unsupported attempt: historical counter correction is restricted to the pinned completed v2 attempt")
    for group in ("source_files", "receipt_files"):
        for relative, expected in contract[group].items():
            p = (REPO / relative).resolve()
            require(p.is_relative_to(REPO.resolve()) and p.is_file() and sha_file(p) == expected,
                    f"Historical correction evidence changed or missing: {relative}")
    expected_reports = {u["report_relative"] for u in contract["units"].values()}
    actual_reports = {p.relative_to(REPO).as_posix() for p in Path(attempt_dir).glob("*/report.json")}
    require(actual_reports == expected_reports, "Historical attempt unit set changed")
    return contract


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")), sha_file(path)


def unit_reports(attempt_dir, report):
    """Only pinned local report paths; a report cannot redirect reads to an arbitrary producer."""
    contract = authenticate_attempt(attempt_dir)
    return {unit: read(REPO / item["report_relative"]) for unit, item in contract["units"].items()}


def corrected_steps(unit, rep, episodes):
    """Apply the legacy correction once, only to a byte-identified reviewed legacy report.

    An unknown, partial, failed, edited or already corrected report is not a lower bound.
    Its corrected count is unknown here. The original counter remains available separately.
    """
    kind = rep.get("kind")
    recorded = rep.get("simulator_steps_exact")
    unknown = {"kind": kind, "recorded": recorded, "loads": None, "corrected": None,
               "status": "unknown: report is not an authenticated completed legacy counter record"}
    contract = correction_contract()
    approved = contract["units"].get(unit)
    if (not approved or type(episodes) is not int or episodes != contract["episodes"]
            or rep.get("status") != "completed" or type(recorded) is not int
            or recorded != approved["recorded"] or kind != approved["kind"]
            or rep.get("producer_sha256") != approved["producer_sha256"]):
        return unknown
    try:
        if canonical_sha(rep) != approved["canonical_report_sha256"]:
            return unknown
    except (TypeError, ValueError):
        return unknown
    loads = approved["dataset_loads"]
    per_load = episodes * contract["horizon"] - episodes
    return {"kind": kind, "recorded": recorded, "loads": loads, "per_load_undercount": per_load,
            "corrected": recorded + loads * per_load,
            "status": "exact reconstruction for pinned complete v2 report; not applicable to new producers",
            "correction_contract_sha256": CORRECTION_CONTRACT_SHA256}


def analyse(attempt_dir, *, exploratory=False):
    attempt_dir = Path(attempt_dir).resolve()
    contract = authenticate_attempt(attempt_dir)
    report, report_sha = read(attempt_dir / "report.json")
    require(report_sha == contract["receipt_files"][(attempt_dir / "report.json").relative_to(REPO).as_posix()],
            "Historical attempt report changed after authentication")
    sources = {"attempt_report": {"path": str(attempt_dir / "report.json"), "sha256": report_sha}}
    units = unit_reports(attempt_dir, report)
    sources["unit_reports"] = {u: {"sha256": sha} for u, (_, sha) in units.items()}
    episodes = contract["episodes"]
    sources["correction_contract"] = {"path": str(CORRECTION_CONTRACT), "sha256": CORRECTION_CONTRACT_SHA256}

    # R3: accounting.
    accounting = {u: corrected_steps(u, rep, episodes) for u, (rep, _) in units.items()}
    require(all(a["corrected"] is not None for a in accounting.values()),
            "Incomplete correction evidence; neither an exact total nor a scientific decision is available")
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
