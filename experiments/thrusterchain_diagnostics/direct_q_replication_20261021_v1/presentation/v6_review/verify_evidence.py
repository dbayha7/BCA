"""Independently check the frozen evidence used in the V6 reading revision.

Run with Python 3.10+; only the standard library is used. The script reads saved
CSV/JSON files, hashes all fresh result artifacts, and writes this directory's
evidence_validation.json. It does not import experiment code, load model files,
run a simulator, regenerate bootstrap draws, or change experiment data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
FRESH_ROOT = HERE.parent.parent
FIRST_ROOT = FRESH_ROOT.parent / "direct_q_diagnostic_20261021_v2"
METHODS = (
    "raw_q", "live_width", "matched_distance", "shuffled_width", "constant_offset"
)
ATOL = 1e-10
FROZEN_FRESH_HASHES = {
    "bootstrap_statistics.npz": "9011ec1d2fab4f1b6ff87586624ba23792e31d606d4d36c440af90729e1ca744",
    "candidate_records.csv": "2b429c4ea8f59e646b0404e1a0aca8b448f6f1a8950de5fa12d12c214d27d574",
    "execution_receipt.json": "432ca63a737ea60d8412680dac1a39f2d1f906e82e20c839270794e76ed9457f",
    "metric_summary.csv": "5255b9737f43375d57f90f3bd026f517634c669d28da1d8f6f52b1663504aaf7",
    "paired_results.json": "6e815ac9fe1e86b08619a45812635fe9ea677abbd4c6cb77508617c398bd157b",
    "raw_results.npz": "f31718e6c1261cdd784d6d009cc7494499467f5881f4ddf4931257c7667920bb",
    "state_records.csv": "595ba89e1089f0c5d28a9d67b1f0dc6299c20fa6e48f8c6405381d0f78d889dc",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), "Nonfinite: " + label)
    require(abs(actual - expected) <= ATOL, f"Mismatch {label}: {actual} != {expected}")


def concordance(scores, truth):
    credits = []
    near_ties = 0
    for a in range(len(truth)):
        for b in range(a + 1, len(truth)):
            true_gap = truth[a] - truth[b]
            if abs(true_gap) <= 1e-6:
                near_ties += 1
                continue
            score_gap = scores[a] - scores[b]
            credits.append(0.5 if abs(score_gap) <= 1e-6 else float((true_gap > 0) == (score_gap > 0)))
    require(bool(credits), "This frozen bank should have defined concordance")
    return statistics.mean(credits), len(credits), near_ties


def fresh_arithmetic():
    result_root = FRESH_ROOT / "results"
    candidates = read_csv(result_root / "candidate_records.csv")
    states = read_csv(result_root / "state_records.csv")
    summary = {r["metric"]: r for r in read_csv(result_root / "metric_summary.csv")}
    require(len(candidates) == 384 and len(states) == 32, "Unexpected frozen-bank dimensions")
    require([int(s["state_id"]) for s in states] == list(range(32)), "State identity/order changed")
    rows, pair_counts, true_ties = [], [], []
    for state_id in range(32):
        batch = [r for r in candidates if int(r["state_id"]) == state_id]
        require([int(r["candidate_id"]) for r in batch] == list(range(12)), "Candidate identity/order changed")
        require(states[state_id]["valid"] == "True", "Unexpected invalid frozen state")
        truth = [float(r["true_policy_q"]) for r in batch]
        error = [float(r["q1"]) - value for r, value in zip(batch, truth)]
        metrics = {}
        for name, field in (("width", "width"), ("distance", "distance"), ("shuffled", "shuffled_width")):
            rho = statistics.correlation([float(r[field]) for r in batch], error)
            metrics["primary_rho_" + name] = rho
            close(rho, float(states[state_id]["rho_" + name]), f"state {state_id} rho {name}")
        for control in ("distance", "shuffled"):
            metrics["primary_rho_width_minus_" + control] = metrics["primary_rho_width"] - metrics["primary_rho_" + control]
        choices = {}
        for method in METHODS:
            scores = [float(r["score_" + method]) for r in batch]
            # Python's max preserves the first index on an exact tie.
            choice = max(range(12), key=scores.__getitem__)
            choices[method] = choice
            value, pairs, ties = concordance(scores, truth)
            metrics["concordance_" + method] = value
            metrics["secondary_chosen_true_q_" + method] = truth[choice]
            require(choice == int(states[state_id][method + "__chosen_index"]), f"Choice mismatch {state_id} {method}")
            close(value, float(states[state_id][method + "__concordance"]), f"concordance {state_id} {method}")
            require(pairs == int(states[state_id][method + "__comparable_pairs"]), "Pair count mismatch")
            if method == "raw_q":
                pair_counts.append(pairs)
                true_ties.append(ties)
        for method in METHODS:
            metrics["primary_concordance_change_" + method] = metrics["concordance_" + method] - metrics["concordance_raw_q"]
            metrics["secondary_delta_vs_raw_" + method] = truth[choices[method]] - truth[choices["raw_q"]]
            metrics["secondary_advantage_vs_actor_" + method] = truth[choices[method]] - truth[6]
        metrics["secondary_original_actor_true_q"] = truth[6]
        for control in ("matched_distance", "shuffled_width"):
            metrics["secondary_selected_q_live_width_minus_" + control] = truth[choices["live_width"]] - truth[choices[control]]
        rows.append(metrics)

    require(len(rows[0]) == 33, "Expected 33 nonredundant aggregate metrics")
    calculated = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
    errors = {}
    for key, actual in calculated.items():
        expected = float(summary[key]["mean"])
        close(actual, expected, key)
        errors[key] = abs(actual - expected)
        require(summary[key]["defined_states"] == "32", "Unexpected summary eligibility")

    effects = {}
    for method in METHODS:
        deltas = [r["secondary_delta_vs_raw_" + method] for r in rows]
        effects[method] = {
            "improved": sum(d > 0 for d in deltas),
            "harmed": sum(d < 0 for d in deltas),
            "unchanged": sum(d == 0 for d in deltas),
            "changed_state_effects": [{"state_id": i, "delta": d} for i, d in enumerate(deltas) if d != 0],
        }
    worked_candidates = [r for r in candidates if r["state_id"] == "24" and r["candidate_id"] in ("0", "7")]
    worked = {
        r["candidate_id"]: {"support_class": r["support_class"], **{
            field: float(r[field]) for field in ("q1", "width", "score_live_width", "true_policy_q")
        }} for r in worked_candidates
    }
    worked["raw_gap_0_minus_7"] = worked["0"]["q1"] - worked["7"]["q1"]
    worked["width_gap_0_minus_7"] = worked["0"]["width"] - worked["7"]["width"]
    worked["return_change_7_minus_0"] = worked["7"]["true_policy_q"] - worked["0"]["true_policy_q"]
    return {
        "states": len(states), "candidate_records": len(candidates),
        "recomputed_state_correlations": 96, "recomputed_state_concordances": 160,
        "recomputed_selected_choices": 160, "recomputed_aggregate_metrics": 33,
        "maximum_saved_mean_difference": max(errors.values()),
        "aggregate_means": calculated, "choice_effects": effects,
        "width_error_positive_states": sum(r["primary_rho_width"] > 0 for r in rows),
        "width_error_negative_states": sum(r["primary_rho_width"] < 0 for r in rows),
        "comparable_pairs": sum(pair_counts), "true_near_ties": sum(true_ties),
        "simulator_transitions_from_saved_rows": sum(int(r["steps"]) for r in candidates),
        "failure_flags_from_saved_rows": sum(r["failure_event"] == "True" for r in candidates),
        "state_24_worked_example": worked,
        "intervals": {key: json.loads(summary[key]["ci95"]) for key in calculated},
        "interval_note": "Read from authenticated saved summaries; not independently regenerated by this script.",
    }


def first_bank_choices():
    candidates = read_csv(FIRST_ROOT / "results" / "candidate_records.csv")
    decisions = read_csv(FIRST_ROOT / "results" / "paired_decisions.csv")
    lookup = {(int(r["state_id"]), r["method"]): r for r in decisions if r["group"] == "pooled" and r["support"] == "all"}
    summary = {}
    for method in ("raw_q", "live_width", "matched_distance", "constant_offset", "detached_width"):
        deltas = []
        for state_id in range(32):
            batch = [r for r in candidates if int(r["state_id"]) == state_id]
            require([int(r["candidate_id"]) for r in batch] == list(range(12)), "First-bank candidate order changed")
            choice = max(range(12), key=lambda j: float(batch[j]["score_" + method]))
            raw = max(range(12), key=lambda j: float(batch[j]["score_raw_q"]))
            delta = float(batch[choice]["true_policy_q"]) - float(batch[raw]["true_policy_q"])
            record = lookup[(state_id, method)]
            require(int(record["chosen_candidate"]) == choice, "First-bank choice mismatch")
            close(delta, float(record["delta_vs_raw"]), "First-bank return contrast")
            deltas.append(delta)
        summary[method] = {
            "mean_delta_vs_raw": statistics.mean(deltas),
            "improved": sum(d > 0 for d in deltas),
            "harmed": sum(d < 0 for d in deltas),
            "unchanged": sum(d == 0 for d in deltas),
        }
    return {"recomputed_selected_choices": 160, "methods": summary}


def main():
    result_root = FRESH_ROOT / "results"
    status_path = result_root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    require(status["status"] == "complete", "Frozen experiment did not complete")
    require(status["output_sha256"] == FROZEN_FRESH_HASHES, "Frozen result manifest changed")
    inputs = [result_root / name for name in FROZEN_FRESH_HASHES]
    inputs += [status_path, FIRST_ROOT / "results" / "candidate_records.csv", FIRST_ROOT / "results" / "paired_decisions.csv"]
    before = {str(p.relative_to(FRESH_ROOT.parent)): digest(p) for p in inputs}
    for name, expected in FROZEN_FRESH_HASHES.items():
        require(digest(result_root / name) == expected, "Frozen result hash mismatch: " + name)
    fresh = fresh_arithmetic()
    first = first_bank_choices()
    after = {str(p.relative_to(FRESH_ROOT.parent)): digest(p) for p in inputs}
    require(before == after, "Evidence input changed during verification")
    html = HERE.parent / "ThrusterChain_diagnostics_results_v6_reviewed.html"
    receipt = {
        "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent standard-library arithmetic from saved CSV records; frozen fresh output hashes authenticated.",
        "new_model_calls": 0, "new_simulator_transitions": 0, "new_training_updates": 0,
        "bootstrap_recomputed": False, "saved_inputs_unchanged_during_check": True,
        "input_sha256": before, "verifier_sha256": digest(Path(__file__)),
        "presentation_snapshot": {"path": str(html), "sha256": digest(html), "note": "Snapshot identity only; numeric script does not certify rendered HTML or prose."},
        "fresh_bank": fresh, "first_bank": first,
    }
    output = HERE / "evidence_validation.json"
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(output), "fresh_metrics": 33,
                      "maximum_saved_mean_difference": fresh["maximum_saved_mean_difference"],
                      "unchanged_saved_inputs": True, "bootstrap_recomputed": False}, indent=2))


if __name__ == "__main__":
    main()
