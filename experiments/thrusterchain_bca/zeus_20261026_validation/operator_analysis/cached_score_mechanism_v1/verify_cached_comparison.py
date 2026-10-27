"""Verify fixed saved comparison outputs; never invoke the comparison executable.

Only stdlib and NumPy are imported. This checks the already frozen four scores
and outcomes; it adds no selector, sweep, fit, model call or scientific run.
The separate INDEPENDENT_REVIEW.json is created exclusively after all checks.
"""

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PROTOCOL_SHA = "287c0b0dd050fa87d20ec17a36004f976d3434b7a6f617278e0b228a911f116c"
COMPARISON_SHA = "8e0d61ccb2b06ac7357b8ed8db39e27d5f3d9273a49f0085a172e0d4ee0c411a"
METHODS = ("q1", "twin_min", "actor", "mean_native_target")
ARTIFACTS = (
    "protocol.json", "FREEZE.json", "ATTEMPT.json", "SELECTIONS.json",
    "SELECTIONS_SEALED.json", "STATE_OUTCOMES.json", "all_prespecified_strata.csv",
    "report.json", "run_cached_comparison.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    output = HERE / "INDEPENDENT_REVIEW.json"
    require(not output.exists(), "Independent review already exists; do not overwrite it")
    artifact_hashes = {name: digest(HERE / name) for name in ARTIFACTS}
    require(artifact_hashes["protocol.json"] == PROTOCOL_SHA, "Protocol identity mismatch")
    require(artifact_hashes["run_cached_comparison.py"] == COMPARISON_SHA, "Executable identity mismatch")
    protocol, freeze, attempt = read("protocol.json"), read("FREEZE.json"), read("ATTEMPT.json")
    selection_file, seal = read("SELECTIONS.json"), read("SELECTIONS_SEALED.json")
    outcome_file, report = read("STATE_OUTCOMES.json"), read("report.json")
    for record in (freeze, attempt, selection_file, report):
        require(record["protocol_sha256"] == PROTOCOL_SHA, "Recorded protocol hash mismatch")
    for record in (attempt, report):
        require(record["script_sha256"] == COMPARISON_SHA, "Recorded executable hash mismatch")
    require(freeze["new_selector_outcomes_computed"] is False, "Invalid freeze declaration")
    require(seal["saved_value_array_decoded"] is False and seal["outcomes_computed"] is False, "Invalid seal declaration")
    require(seal["selections_sha256"] == artifact_hashes["SELECTIONS.json"], "Sealed selections changed")
    dates = [datetime.fromisoformat(record["utc"]) for record in (freeze, attempt, seal, report)]
    require(dates == sorted(dates), "Freeze/reservation/seal/completion chronology mismatch")
    for name, expected in report["outputs"].items():
        require(artifact_hashes[name] == expected, "Output artifact hash mismatch: " + name)
    require(report["inputs"] == protocol["inputs"], "Input manifest mismatch")
    results = ROOT / protocol["input_results_directory"]
    input_hashes = {name: digest(results / name) for name in protocol["inputs"]}
    require(input_hashes == protocol["inputs"], "Frozen input hash mismatch")
    require(report["ledger"] == protocol["budget"], "Budget ledger mismatch")
    require(report["status"] == "completed_cached_mechanism_comparison", "Comparison did not complete")
    require(report["full_training_hold"] is True and report["pilot_executed"] is False, "Execution scope flags mismatch")
    require(report["scientific_superiority_established"] is False and report["deployable_policy_established"] is False, "Claim flags mismatch")

    counters = {"array_descriptors": 0, "numeric_comparisons": 0}

    def number(actual, expected):
        require(math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=2e-12), "Numerical result mismatch")
        counters["numeric_comparisons"] += 1

    def compare(actual, expected):
        if isinstance(expected, dict):
            require(set(actual) == set(expected), "Object keys mismatch")
            for key, value in expected.items():
                compare(actual[key], value)
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected), "List shape mismatch")
            for value, reference in zip(actual, expected):
                compare(value, reference)
        elif expected is None or isinstance(expected, (str, bool)):
            require(actual == expected and type(actual) is type(expected), "Categorical result mismatch")
        elif isinstance(expected, int):
            require(type(actual) is int and actual == expected, "Integer result mismatch")
        else:
            number(actual, expected)

    raw = json.loads((results / "raw_banks.json").read_text(encoding="utf-8"))["development"]
    metadata = raw["metadata"]
    require([(m["group_id"], m["candidate_id"]) for m in metadata] == [(g, c) for g in range(128) for c in range(9)], "Candidate ordering mismatch")
    require(raw["smoothing_seeds"] == protocol["smoothing_seeds"], "Draw identity mismatch")
    with np.load(results / "raw_banks.npz", allow_pickle=False) as archive:
        def array(name, dtype, shape):
            desc = raw[name]
            value = archive[desc["npz_key"]]
            require(value.dtype.str == desc["dtype"] == dtype, "Array dtype mismatch")
            require(list(value.shape) == desc["shape"] == shape, "Array shape mismatch")
            require(hashlib.sha256(value.tobytes()).hexdigest() == desc["array_bytes_sha256"], "Array bytes mismatch")
            require(np.all(np.isfinite(value)), "Nonfinite saved array")
            counters["array_descriptors"] += 1
            return value

        q = array("q_current", "<f4", [1152, 2]).astype(np.float64).reshape(128, 9, 2)
        target = array("native_y", "<f4", [4, 1152]).astype(np.float64).mean(axis=0).reshape(128, 9)
        actions = array("actions", "<f4", [1152, 4]).reshape(128, 9, 4)
        actor = array("saved_actor_actions", "<f4", [128, 4])
        require(np.array_equal(actions[:, 0, :], actor), "Actor candidate zero mismatch")
        values = array("saved_values", "<f8", [1152]).reshape(128, 9)
    require(report["verified_array_descriptors"] == ["q_current", "native_y", "actions", "saved_actor_actions", "saved_values"], "Executed descriptor list mismatch")

    def choose(scores):
        scores = [float(value) for value in scores]
        maximum = max(scores)
        near = [i for i, value in enumerate(scores) if abs(value - maximum) <= 1e-6 + 1e-6 * abs(maximum)]
        ordered = sorted(scores)
        return {
            "index": near[0], "exact_argmax_index": scores.index(maximum),
            "near_best_indices": near, "near_tie": len(near) > 1,
            "maximum_score": maximum, "selected_score": scores[near[0]],
            "top_two_gap": ordered[-1] - ordered[-2], "scores": scores,
        }

    selections, expected_outcomes = [], []
    for group in range(128):
        meta = {key: metadata[9 * group][key] for key in ("group_id", "batch", "time", "episode_seed")}
        selection = {**meta, "methods": {
            "q1": choose(q[group, :, 0]),
            "twin_min": choose([min(pair) for pair in q[group]]),
            "actor": {"index": 0, "near_tie": False, "selection_rule": "frozen_actor_candidate_0", "scores": None},
            "mean_native_target": choose(target[group]),
        }}
        compare(selection_file["states"][group], selection)
        selections.append(selection)
        selected = {method: float(values[group, selection["methods"][method]["index"]]) for method in METHODS}
        best = float(max(values[group]))
        methods = {}
        for method in METHODS:
            chosen = selection["methods"][method]
            gain = selected[method] - selected["q1"]
            epsilon = 1e-6 + 1e-6 * max(abs(selected[method]), abs(selected["q1"]))
            methods[method] = {
                "selected_index": chosen["index"], "value": selected[method],
                "regret": best - selected[method], "gain_vs_q1": gain,
                "gain_vs_actor": selected[method] - selected["actor"],
                "epsilon_vs_q1": epsilon,
                "switched_vs_q1": chosen["index"] != selection["methods"]["q1"]["index"],
                "classification_vs_q1": "beneficial" if gain > epsilon else "harmful" if gain < -epsilon else "numerically_inconclusive",
                "score_near_tie": chosen["near_tie"],
            }
        outcome = {**meta, "candidate_values": values[group].tolist(), "best_candidate_value": best, "methods": methods}
        compare(outcome_file["states"][group], outcome)
        expected_outcomes.append(outcome)
    require(len(selection_file["states"]) == len(outcome_file["states"]) == 128, "Saved state count mismatch")

    def average(items):
        return math.fsum(items) / len(items)

    def in_stratum(row, name):
        if name == "overall":
            return True
        if "/" in name:
            batch, time = name.split("/")
            return row["batch"] == batch and row["time"] == int(time[4:])
        if name.startswith("time"):
            return row["time"] == int(name[4:])
        return row["batch"] == name

    summaries = []
    for name in protocol["strata"]:
        rows = [row for row in expected_outcomes if in_stratum(row, name)]
        n = len(rows)
        require(n == (128 if name == "overall" else 16 if "/" in name else 32 if name.startswith("time") else 64), "Stratum size mismatch")
        for method in METHODS:
            entries = [row["methods"][method] for row in rows]
            switched = [entry for entry in entries if entry["switched_vs_q1"]]
            harmful = [entry for entry in switched if entry["classification_vs_q1"] == "harmful"]
            beneficial = [entry for entry in switched if entry["classification_vs_q1"] == "beneficial"]
            inconclusive = [entry for entry in switched if entry["classification_vs_q1"] == "numerically_inconclusive"]
            require(len(harmful) + len(beneficial) + len(inconclusive) == len(switched), "Switch accounting mismatch")
            summaries.append({
                "stratum": name, "method": method, "states": n,
                "mean_selected_value": average([entry["value"] for entry in entries]),
                "mean_gain_vs_q1": average([entry["gain_vs_q1"] for entry in entries]),
                "mean_gain_vs_actor": average([entry["gain_vs_actor"] for entry in entries]),
                "mean_regret": average([entry["regret"] for entry in entries]),
                "mean_epsilon_vs_q1": average([entry["epsilon_vs_q1"] for entry in entries]),
                "switches_vs_q1": len(switched), "switch_rate_vs_q1": len(switched) / n,
                "harmful_switches": len(harmful), "beneficial_switches": len(beneficial),
                "inconclusive_switches": len(inconclusive), "harm_rate_all_states": len(harmful) / n,
                "harm_rate_among_switches": len(harmful) / len(switched) if switched else None,
                "mean_loss_on_harmful_switches": -average([entry["gain_vs_q1"] for entry in harmful]) if harmful else None,
                "mean_gain_on_beneficial_switches": average([entry["gain_vs_q1"] for entry in beneficial]) if beneficial else None,
                "minimum_gain_vs_q1": min(entry["gain_vs_q1"] for entry in entries),
                "maximum_gain_vs_q1": max(entry["gain_vs_q1"] for entry in entries),
                "score_near_tie_states": sum(entry["score_near_tie"] for entry in entries),
            })
    compare(report["strata"], summaries)
    with (HERE / "all_prespecified_strata.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(summaries) == 60, "Summary row count mismatch")
    for actual, expected in zip(csv_rows, summaries):
        require(set(actual) == set(expected), "CSV columns mismatch")
        for key, value in expected.items():
            if value is None:
                require(actual[key] == "", "CSV null mismatch")
            elif isinstance(value, str):
                require(actual[key] == value, "CSV label mismatch")
            elif isinstance(value, int):
                require(int(actual[key]) == value, "CSV count mismatch")
            else:
                number(actual[key], value)

    screen = [{
        "batch": row["stratum"], "mean_gain": row["mean_gain_vs_q1"],
        "mean_numerical_allowance": row["mean_epsilon_vs_q1"],
        "pass": row["mean_gain_vs_q1"] > row["mean_epsilon_vs_q1"],
    } for row in summaries if row["method"] == "mean_native_target" and row["stratum"] in ("development_1", "development_2")]
    compare(report["primary_screen"], screen)
    require(len(screen) == 2, "Primary screen count mismatch")
    expected_decision = "PASS_EXPLORATORY_SCREEN_PROPOSE_ONLY" if all(row["pass"] for row in screen) else "STOP_OPERATOR_REPLACEMENT_BRANCH"
    require(report["decision"] == expected_decision, "Primary decision mismatch")
    require(artifact_hashes == {name: digest(HERE / name) for name in ARTIFACTS}, "Comparison artifacts changed during verification")
    require(input_hashes == {name: digest(results / name) for name in protocol["inputs"]}, "Raw inputs changed during verification")
    review = {
        "schema": "unifloral.cached_score_mechanism.independent_review.v1",
        "status": "pass", "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent verification of the existing fixed four-selector saved result only",
        "verifier_sha256": digest(Path(__file__)), "artifact_sha256": artifact_hashes,
        "input_sha256": input_hashes, "hashes_unchanged_after_review": True,
        "comparison_counts": {**counters, "states": 128, "method_state_records": 512, "stratum_method_records": 60, "primary_batch_screens": 2},
        "arithmetic_tolerance": {"rtol": 2e-12, "atol": 2e-12},
        "primary_screen": screen, "decision": expected_decision,
        "overall_verified_rows": [row for row in summaries if row["stratum"] == "overall"],
        "runtime": {"python": sys.version, "numpy": np.__version__, "executable": sys.executable},
        "command_from_repository_root": ".\\.venv\\Scripts\\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/operator_analysis/cached_score_mechanism_v1/verify_cached_comparison.py",
        "limitations": [
            "The original comparison executable was not invoked; no new selector, sweep, outcome filter, model forward, simulator call, fit, confirmation read or pilot was performed.",
            "Recomputing the frozen scores and outcomes is a verification of the existing result, not an additional scientific comparison execution.",
            "The mean target is a privileged evaluator-derived four-draw mechanism control, not an exact expected operator or deployable policy.",
            "Both-batch numerical screen passage is exploratory and does not establish significance, causal mechanism, deployable predictor accuracy or training efficacy.",
            "All results reuse previously examined development episodes under one frozen checkpoint; all 15 strata and harmful switches are retained.",
            "Stored chronology and sealed hashes agree with the reviewed source ordering; this review does not independently observe the original process internals.",
        ],
        "full_training_hold": True, "pilot_executed": False,
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(review, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "pass", "comparison_counts": review["comparison_counts"], "decision": expected_decision, "review": str(output), "review_sha256": digest(output)}, indent=2))


if __name__ == "__main__":
    main()
