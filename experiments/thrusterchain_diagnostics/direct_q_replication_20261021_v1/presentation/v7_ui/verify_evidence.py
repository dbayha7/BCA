"""Audit V7's saved-data mapping and actual pure JS arithmetic without a browser.

Requires Python 3.10+ and Node. Reads frozen CSVs, V7 data/component files and the
built HTML. Only evidence_check.json is written. No model/simulator code imports.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "results"
HTML = HERE.parent / "ThrusterChain_diagnostics_results_v7_interactive.html"
METHODS = ["raw_q", "live_width", "matched_distance", "shuffled_width", "constant_offset"]
FROZEN = {
    "state_records.csv": "595ba89e1089f0c5d28a9d67b1f0dc6299c20fa6e48f8c6405381d0f78d889dc",
    "candidate_records.csv": "2b429c4ea8f59e646b0404e1a0aca8b448f6f1a8950de5fa12d12c214d27d574",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), "Nonfinite " + label)
    require(abs(actual - expected) < 1e-10, f"Mismatch {label}: {actual} versus {expected}")


def main():
    paths = [RESULTS / name for name in FROZEN] + [HERE / "data.json", HERE / "explorer.js", HERE / "explorer.html", HTML]
    before = {str(p): digest(p) for p in paths}
    for name, expected in FROZEN.items():
        require(digest(RESULTS / name) == expected, "Frozen input changed: " + name)
    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
    require(data["source_sha256"] == FROZEN, "Source provenance mismatch")
    html = HTML.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="xe-data">(.*?)</script>', html, re.S)
    require(match is not None and json.loads(match.group(1)) == data, "Embedded data differs from data.json")
    interaction = re.search(r'<script id="xe-interaction">\s*(.*?)\s*</script>', html, re.S)
    require(interaction is not None and interaction.group(1) == (HERE / "explorer.js").read_text(encoding="utf-8").strip(), "Built JavaScript differs from source")
    with (RESULTS / "state_records.csv").open(newline="", encoding="utf-8") as handle:
        states = list(csv.DictReader(handle))
    with (RESULTS / "candidate_records.csv").open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    require([s["id"] for s in data["states"]] == list(range(32)), "State indices changed")
    copied_fields = 0
    concordances = 0
    choices = 0
    for state in data["states"]:
        source = states[state["id"]]
        for key, value in state.items():
            if key in ("methods", "candidates", "id"):
                continue
            require(value == convert(source[key]), f"State field changed: {state['id']} {key}")
            copied_fields += 1
        require([c["id"] for c in state["candidates"]] == list(range(12)), "Candidate indices changed")
        for method, values in state["methods"].items():
            require(method in METHODS, "Unknown method")
            for key, value in values.items():
                require(value == convert(source[method + "__" + key]), f"Method field changed: {state['id']} {method} {key}")
                copied_fields += 1
        for candidate in state["candidates"]:
            original = candidates[state["id"] * 12 + candidate["id"]]
            require(int(original["state_id"]) == state["id"] and int(original["candidate_id"]) == candidate["id"], "CSV identity mismatch")
            for key, value in candidate.items():
                if key == "id":
                    continue
                if key == "scores":
                    for method, score in value.items():
                        require(score == float(original["score_" + method]), "Frozen score changed")
                        copied_fields += 1
                elif key == "action":
                    require(value == [float(original[f"action_{j}"]) for j in range(4)], "Action changed")
                    copied_fields += 4
                else:
                    require(value == convert(original[key]), f"Candidate field changed: {state['id']} {candidate['id']} {key}")
                    copied_fields += 1
            close(candidate["signed_q1_error"], candidate["q1"] - candidate["true_policy_q"], "Signed error")
        cs = state["candidates"]
        for method in METHODS:
            values = state["methods"][method]
            winner = max(range(12), key=lambda j: cs[j]["scores"][method])
            require(winner == values["chosen_index"], "Saved winner is not first exact argmax")
            close(cs[winner]["true_policy_q"], values["chosen_true_q"], "Winner return")
            raw_winner = int(state["methods"]["raw_q"]["chosen_index"])
            close(values["delta_vs_raw"], cs[winner]["true_policy_q"] - cs[raw_winner]["true_policy_q"], "Outcome sign")
            credit = []
            for a in range(12):
                for b in range(a + 1, 12):
                    truth_gap = cs[a]["true_policy_q"] - cs[b]["true_policy_q"]
                    score_gap = cs[a]["scores"][method] - cs[b]["scores"][method]
                    if abs(truth_gap) > 1e-6:
                        credit.append(.5 if abs(score_gap) <= 1e-6 else float((truth_gap > 0) == (score_gap > 0)))
            close(statistics.mean(credit), values["concordance"], "Concordance")
            require(len(credit) == values["comparable_pairs"], "Pair denominator changed")
            concordances += 1
            choices += 1
        close(statistics.correlation([c["width"] for c in cs], [c["signed_q1_error"] for c in cs]), state["rho_width"], "Width/signed-error correlation")

    # Extract and execute the actual pure functions. No DOM/browser shim is used.
    node_code = r"""
const fs = require('fs');
const source = fs.readFileSync(process.argv[1], 'utf8');
const data = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const start=source.indexOf('  const mean = '), end=source.indexOf('  const initial=');
if(start<0||end<=start)throw Error('Pure function boundary missing');
const block=source.slice(start,end);
const api=new Function('data',block+'\nreturn {filterStates,summarize,sortedCandidates,csvFor,outcome};')(data);
const methods=['raw_q','live_width','matched_distance','shuffled_width','constant_offset'];
const contexts=['all','stall_low','stall_high','monotone_low','monotone_high'];
const times=['all','0','50','99'];
const before=JSON.stringify(data);
const groups=[];
for(const context of contexts)for(const time of times){
 const subset=api.filterStates(context,time);
 for(const method of methods)groups.push({context,time,method,ids:subset.map(s=>s.id),summary:api.summarize(subset,method)});
}
const sorts=[],exports=[];
for(const state of data.states)for(const method of methods){
 for(const sort of ['id','score','truth','width'])sorts.push({state:state.id,method,sort,ids:api.sortedCandidates(state,method,sort).map(c=>c.id)});
 exports.push({state:state.id,method,csv:api.csvFor(state,method)});
}
if(JSON.stringify(data)!==before)throw Error('Pure functions mutated the saved data');
process.stdout.write(JSON.stringify({groups,sorts,exports,outcomes:[-1,0,1].map(api.outcome),inputsUnchanged:true}));
"""
    run = subprocess.run(["node", "-e", node_code, str(HERE / "explorer.js"), str(HERE / "data.json")], text=True, encoding="utf-8", capture_output=True, check=True)
    actual = json.loads(run.stdout)
    require(actual["outcomes"] == ["loss", "same", "gain"], "Outcome labels reversed")
    filter_results = []
    for group in actual["groups"]:
        selected = [r for r in states if r["valid"] == "True" and (group["context"] == "all" or r["context"] == group["context"]) and (group["time"] == "all" or r["time"] == group["time"])]
        require(group["ids"] == [int(r["state_id"]) for r in selected], "Context/time filter changed identity")
        method, summary = group["method"], group["summary"]
        deltas = [float(r[method + "__delta_vs_raw"]) for r in selected]
        concordance_deltas = [float(r[method + "__concordance"]) - float(r["raw_q__concordance"]) for r in selected]
        close(summary["gain"], statistics.mean(deltas), "Filtered equal-state mean")
        close(summary["concordanceChange"], statistics.mean(concordance_deltas), "Filtered concordance mean")
        require(summary["n"] == summary["rankingN"] == len(selected), "Filtered denominator changed")
        require(summary["improved"] == sum(d > 0 for d in deltas), "Improved count")
        require(summary["harmed"] == sum(d < 0 for d in deltas), "Harmed count")
        require(summary["unchanged"] == sum(d == 0 for d in deltas), "Unchanged count")
        require(summary["changedCommands"] == sum(r[method + "__changed_command"] == "True" for r in selected), "Command-change count")
        filter_results.append({k: group[k] for k in ("context", "time", "method", "summary")})
    for item in actual["sorts"]:
        cs = data["states"][item["state"]]["candidates"]
        key = {"id": lambda c: (c["id"],), "score": lambda c: (-c["scores"][item["method"]], c["id"]),
               "truth": lambda c: (-c["true_policy_q"], c["id"]), "width": lambda c: (c["width"], c["id"])}[item["sort"]]
        require(item["ids"] == [c["id"] for c in sorted(cs, key=key)], "Presentation sort mismatch")
    for item in actual["exports"]:
        rows = list(csv.DictReader(io.StringIO(item["csv"])))
        state = data["states"][item["state"]]
        require(len(rows) == 12, "Export lost candidates")
        for candidate, row in zip(state["candidates"], rows):
            require(int(row["candidate_id"]) == candidate["id"] and int(row["state_id"]) == state["id"], "Export identity mismatch")
            require(row["method"] == item["method"] and row["support"] == candidate["support_class"], "Export label mismatch")
            expected = {"raw_q1": candidate["q1"], "score": candidate["scores"][item["method"]], "measured_qpi": candidate["true_policy_q"],
                        "signed_q1_error": candidate["signed_q1_error"], "width": candidate["width"], "distance": candidate["distance"]}
            expected["penalty"] = expected["raw_q1"] - expected["score"]
            expected.update({f"action_{j}": value for j, value in enumerate(candidate["action"])})
            for key, value in expected.items():
                require(float(row[key]) == value, "Export changed full precision: " + key)
    after = {str(p): digest(p) for p in paths}
    require(before == after, "Inputs changed during audit; rerun on stable files")
    receipt = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Exact saved CSV-to-JSON/HTML mapping plus independent arithmetic and actual pure JavaScript filter/sort/export execution; no browser.",
               "copied_scalar_fields_verified": copied_fields, "candidate_scores_verified": 1920,
               "state_choices_recomputed": choices, "state_concordances_recomputed": concordances,
               "within_state_width_correlations_recomputed": 32, "filter_method_combinations_executed": len(actual["groups"]),
               "sort_combinations_executed": len(actual["sorts"]), "csv_exports_executed": len(actual["exports"]),
               "embedded_json_matches_data_file": True, "embedded_js_matches_source": True,
               "source_data_unchanged": True, "pure_js_data_unchanged": actual["inputsUnchanged"],
               "new_model_calls": 0, "new_simulator_transitions": 0, "bootstrap_recomputed": False,
               "input_sha256": before, "verifier_sha256": digest(Path(__file__)), "filter_results": filter_results,
               "limitations": ["No browser rendering or DOM event behavior was tested.", "This script verifies arithmetic and data, not every prose claim."]}
    output = HERE / "evidence_check.json"
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "receipt": str(output), "copied_scalar_fields": copied_fields,
                      "filter_method_combinations": len(actual["groups"]), "sort_combinations": len(actual["sorts"]),
                      "csv_exports": len(actual["exports"]), "saved_inputs_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
