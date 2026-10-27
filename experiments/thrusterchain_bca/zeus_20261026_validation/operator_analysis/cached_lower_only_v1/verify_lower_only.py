"""Independently verify the single saved lower-only diagnostic, without rerunning it."""

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "cached_score_mechanism_v1"
PROTOCOL_SHA = "2299b78c55b71cb2e4fabd9ffaebddace06dfbfddc68ba7a85470856638ba5ea"
SCRIPT_SHA = "91b1ec50e703567aeb8c3db5500dd237b8fccf4655c90285075a51d57c9effbb"
AMENDMENT_SHA = "810c6126d150bfbe00c8bcf99b55642b2cf62a0360e223365a6a4a89c5e73af4"
ARTIFACTS = (
    "protocol.json", "FREEZE.json", "SCOPE_AMENDMENT.json", "ATTEMPT.json",
    "SELECTIONS.json", "SELECTIONS_SEALED.json", "STATE_OUTCOMES.json",
    "all_prespecified_strata.csv", "report.json", "run_lower_only.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    output = HERE / "INDEPENDENT_REVIEW.json"
    require(not output.exists(), "Independent review exists; do not overwrite")
    hashes = {name: digest(HERE / name) for name in ARTIFACTS}
    require(hashes["protocol.json"] == PROTOCOL_SHA, "Protocol hash mismatch")
    require(hashes["run_lower_only.py"] == SCRIPT_SHA, "Script hash mismatch")
    require(hashes["SCOPE_AMENDMENT.json"] == AMENDMENT_SHA, "Scope amendment hash mismatch")
    protocol, freeze = read(HERE, "protocol.json"), read(HERE, "FREEZE.json")
    amendment, attempt = read(HERE, "SCOPE_AMENDMENT.json"), read(HERE, "ATTEMPT.json")
    selections, seal = read(HERE, "SELECTIONS.json"), read(HERE, "SELECTIONS_SEALED.json")
    outcomes, report = read(HERE, "STATE_OUTCOMES.json"), read(HERE, "report.json")
    for record in (freeze, amendment, attempt, selections, report):
        require(record["protocol_sha256"] == PROTOCOL_SHA, "Recorded protocol mismatch")
    for record in (attempt, report):
        require(record["script_sha256"] == SCRIPT_SHA, "Recorded script mismatch")
    require(freeze["lower_only_scores_and_outcomes_computed"] is False, "Freeze declaration mismatch")
    require(amendment["lower_only_scores_and_outcomes_computed"] is False, "Amendment declaration mismatch")
    require(seal["previous_saved_values_read"] is False and seal["lower_only_outcomes_computed"] is False, "Selection seal declaration mismatch")
    require(seal["selections_sha256"] == hashes["SELECTIONS.json"], "Sealed selections changed")
    dates = [datetime.fromisoformat(record["utc"]) for record in (freeze, amendment, attempt, seal, report)]
    require(dates == sorted(dates), "Freeze/amendment/attempt/seal/result chronology mismatch")
    previous_hashes = {name: digest(PREVIOUS / name) for name in protocol["inputs"]}
    require(previous_hashes == protocol["inputs"] == report["inputs"], "Input chain hash mismatch")
    for name, expected in report["outputs"].items():
        require(hashes[name] == expected, "Output hash mismatch: " + name)
    previous_selections = read(PREVIOUS, "SELECTIONS.json")["states"]
    previous_outcomes = read(PREVIOUS, "STATE_OUTCOMES.json")["states"]
    previous_report = read(PREVIOUS, "report.json")
    require(report["previous_controls_reused_unchanged"] == previous_report["strata"], "Previous controls changed")
    require(report["ledger"] == protocol["budget"], "Ledger mismatch")
    require(report["status"] == "completed_cached_lower_only_feasibility", "Incomplete comparison")
    for flag, expected in (("original_BCA_unchanged", True), ("pilot_executed", False), ("full_training_hold", True), ("scientific_superiority_established", False), ("deployable_penalty_established", False)):
        require(report[flag] is expected, "Scope flag mismatch: " + flag)

    numeric_comparisons = 0

    def number(actual, expected):
        nonlocal numeric_comparisons
        require(math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=2e-12), "Numeric mismatch")
        numeric_comparisons += 1

    def compare(actual, expected):
        if isinstance(expected, dict):
            require(set(actual) == set(expected), "Object keys mismatch")
            for key, value in expected.items():
                compare(actual[key], value)
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected), "List length mismatch")
            for value, reference in zip(actual, expected):
                compare(value, reference)
        elif expected is None or isinstance(expected, (str, bool, int)):
            require(type(actual) is type(expected) and actual == expected, "Categorical/count mismatch")
        else:
            number(actual, expected)

    q = np.asarray([s["methods"]["q1"]["scores"] for s in previous_selections], dtype=np.float64)
    y = np.asarray([s["methods"]["mean_native_target"]["scores"] for s in previous_selections], dtype=np.float64)
    require(q.shape == y.shape == (128, 9), "Sealed score shape mismatch")
    require(np.all(np.isfinite(q)) and np.all(np.isfinite(y)), "Nonfinite scores")
    score = np.minimum(q, y)
    penalty = np.maximum(q - y, 0)
    difference = np.abs(score - (q - penalty))
    allowance = 64 * np.finfo(np.float64).eps * (1 + np.abs(q) + np.abs(y) + np.abs(score))
    require(np.all(score <= q) and np.all(penalty >= 0) and np.all(difference <= allowance), "Mapping identity failure")
    compare(report["mapping_identity"], {"candidate_rows": 1152, "max_abs_difference": float(difference.max()), "all_scores_no_greater_than_q1": True, "coefficient": 1})

    expected_outcomes = []
    for group, prior in enumerate(previous_selections):
        meta = {key: prior[key] for key in ("group_id", "batch", "time", "episode_seed")}
        require(meta["group_id"] == group, "Group ordering mismatch")
        require(all(previous_outcomes[group][key] == value for key, value in meta.items()), "Prior outcome identity mismatch")
        vector = score[group].tolist()
        maximum = max(vector)
        near = [i for i, value in enumerate(vector) if abs(value - maximum) <= 1e-6 + 1e-6 * abs(maximum)]
        ordered = sorted(vector)
        index = near[0]
        choice = {
            "index": index, "exact_argmax_index": vector.index(maximum),
            "near_best_indices": near, "near_tie": len(near) > 1,
            "maximum_score": maximum, "selected_score": vector[index],
            "top_two_gap": ordered[-1] - ordered[-2], "scores": vector,
            "positive_q_excess": penalty[group].tolist(),
        }
        compare(selections["states"][group], {**meta, "methods": {"lower_only": choice}})
        values = previous_outcomes[group]["candidate_values"]
        require(len(values) == 9 and all(math.isfinite(v) for v in values), "Invalid saved values")
        baseline_index = prior["methods"]["q1"]["index"]
        value, baseline = values[index], values[baseline_index]
        number(baseline, previous_outcomes[group]["methods"]["q1"]["value"])
        gain = value - baseline
        epsilon = 1e-6 + 1e-6 * max(abs(value), abs(baseline))
        method = {
            "selected_index": index, "value": value, "regret": max(values) - value,
            "gain_vs_q1": gain, "gain_vs_actor": value - values[0],
            "epsilon_vs_q1": epsilon, "switched_vs_q1": index != baseline_index,
            "classification_vs_q1": "beneficial" if gain > epsilon else "harmful" if gain < -epsilon else "numerically_inconclusive",
            "score_near_tie": choice["near_tie"],
        }
        outcome = {**meta, "methods": {"lower_only": method}}
        compare(outcomes["states"][group], outcome)
        expected_outcomes.append(outcome)
    require(len(selections["states"]) == len(outcomes["states"]) == len(expected_outcomes) == 128, "State count mismatch")

    def average(values):
        return math.fsum(values) / len(values)

    def in_stratum(row, name):
        if name == "overall":
            return True
        if "/" in name:
            batch, time = name.split("/")
            return row["batch"] == batch and row["time"] == int(time[4:])
        return row["time"] == int(name[4:]) if name.startswith("time") else row["batch"] == name

    names = ["overall", "development_1", "development_2"] + [f"time{t}" for t in (0, 25, 50, 75)] + [f"{b}/time{t}" for b in ("development_1", "development_2") for t in (0, 25, 50, 75)]
    summaries = []
    for name in names:
        rows = [row for row in expected_outcomes if in_stratum(row, name)]
        entries = [row["methods"]["lower_only"] for row in rows]
        n = len(entries)
        require(n == (128 if name == "overall" else 16 if "/" in name else 32 if name.startswith("time") else 64), "Stratum size mismatch")
        switched = [e for e in entries if e["switched_vs_q1"]]
        harms = [e for e in switched if e["classification_vs_q1"] == "harmful"]
        helps = [e for e in switched if e["classification_vs_q1"] == "beneficial"]
        inconclusive = [e for e in switched if e["classification_vs_q1"] == "numerically_inconclusive"]
        require(len(harms) + len(helps) + len(inconclusive) == len(switched), "Switch accounting mismatch")
        summaries.append({
            "stratum": name, "method": "lower_only", "states": n,
            "mean_selected_value": average([e["value"] for e in entries]),
            "mean_gain_vs_q1": average([e["gain_vs_q1"] for e in entries]),
            "mean_gain_vs_actor": average([e["gain_vs_actor"] for e in entries]),
            "mean_regret": average([e["regret"] for e in entries]),
            "mean_epsilon_vs_q1": average([e["epsilon_vs_q1"] for e in entries]),
            "switches_vs_q1": len(switched), "switch_rate_vs_q1": len(switched) / n,
            "harmful_switches": len(harms), "beneficial_switches": len(helps), "inconclusive_switches": len(inconclusive),
            "harm_rate_all_states": len(harms) / n,
            "harm_rate_among_switches": len(harms) / len(switched) if switched else None,
            "mean_loss_on_harmful_switches": -average([e["gain_vs_q1"] for e in harms]) if harms else None,
            "mean_gain_on_beneficial_switches": average([e["gain_vs_q1"] for e in helps]) if helps else None,
            "minimum_gain_vs_q1": min(e["gain_vs_q1"] for e in entries),
            "maximum_gain_vs_q1": max(e["gain_vs_q1"] for e in entries),
            "score_near_tie_states": sum(e["score_near_tie"] for e in entries),
        })
    compare(report["strata"], summaries)
    with (HERE / "all_prespecified_strata.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(summaries) == 15, "Summary row count mismatch")
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
    screen = [{"batch": row["stratum"], "mean_gain": row["mean_gain_vs_q1"], "mean_numerical_allowance": row["mean_epsilon_vs_q1"], "pass": row["mean_gain_vs_q1"] > row["mean_epsilon_vs_q1"]} for row in summaries if row["stratum"] in ("development_1", "development_2")]
    compare(report["primary_screen"], screen)
    expected_decision = "PASS_EXPLORATORY_LOWER_ONLY_SCREEN_PROPOSE_LOGGED_DATA_PILOT_ONLY" if all(row["pass"] for row in screen) else "STOP_THIS_NONNEGATIVE_PENALTY_PROPOSAL"
    require(report["decision"] == expected_decision, "Frozen screen decision mismatch")
    require(hashes == {name: digest(HERE / name) for name in ARTIFACTS}, "Artifacts changed during verification")
    require(previous_hashes == {name: digest(PREVIOUS / name) for name in protocol["inputs"]}, "Inputs changed during verification")
    review = {
        "schema": "unifloral.cached_lower_only.independent_review.v1", "status": "pass",
        "utc": datetime.now(timezone.utc).isoformat(), "verifier_sha256": digest(Path(__file__)),
        "artifact_sha256": hashes, "input_sha256": previous_hashes, "hashes_unchanged_after_review": True,
        "comparison_counts": {"numeric_comparisons": numeric_comparisons, "candidate_mapping_rows": 1152, "new_selector_state_records": 128, "stratum_rows": 15, "unchanged_prior_control_rows": 60, "batch_screens": 2},
        "arithmetic_tolerance": {"rtol": 2e-12, "atol": 2e-12},
        "mapping_identity": report["mapping_identity"], "primary_screen": screen, "overall_verified": summaries[0],
        "frozen_historical_decision": expected_decision,
        "scope_amendment_verified_before_execution": True,
        "effective_continuation": "Diagnostic only; propose a minimal BCA-consistent revision. No supervised positive/signed correction replacement, evaluator-label fitting, implementation or pilot.",
        "runtime": {"python": sys.version, "numpy": np.__version__, "executable": sys.executable},
        "command_from_repository_root": ".\\.venv\\Scripts\\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/operator_analysis/cached_lower_only_v1/verify_lower_only.py",
        "limitations": [
            "This verifies the existing lower-only selector only; original execution, previous comparison, helper main functions and experiment runtime were not invoked.",
            "No new selector, metric family, fit, model forward, simulation, GPU call, confirmation read or pilot occurred.",
            "The ideal positive-excess score uses privileged evaluator-derived targets and is neither learned nor calibrated BCA.",
            "Passing two numerical direction screens on reused development states does not establish statistical superiority, calibration validity, actor-training benefit or deployability.",
            "The pre-execution scope amendment supersedes the historical pass string's positive-predictor pilot proposal while preserving every numerical rule.",
        ],
        "original_BCA_unchanged": True, "pilot_executed": False, "full_training_hold": True,
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(review, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "pass", "comparison_counts": review["comparison_counts"], "review": str(output), "review_sha256": digest(output)}, indent=2))


if __name__ == "__main__":
    main()
