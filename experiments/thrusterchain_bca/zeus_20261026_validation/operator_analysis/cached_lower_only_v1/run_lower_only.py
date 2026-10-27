"""One lower-only feasibility check on already sealed cached scores/values."""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "cached_score_mechanism_v1"
EXPECTED_PROTOCOL = "2299b78c55b71cb2e4fabd9ffaebddace06dfbfddc68ba7a85470856638ba5ea"


def load_helpers():
    import hashlib
    path = PREVIOUS / "run_cached_comparison.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "8e0d61ccb2b06ac7357b8ed8db39e27d5f3d9273a49f0085a172e0d4ee0c411a":
        raise ValueError("Verified helper source changed")
    spec = importlib.util.spec_from_file_location("verified_comparison_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Its main() guard is false; no data execution.
    return module


def lower_only(q1, target):
    q1 = np.asarray(q1, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if q1.shape != target.shape or not np.all(np.isfinite(q1)) or not np.all(np.isfinite(target)):
        raise ValueError("Invalid mapping inputs")
    penalty = np.maximum(q1 - target, 0)
    score = np.minimum(q1, target)
    subtraction_score = q1 - penalty
    allowed = 64 * np.finfo(np.float64).eps * (1 + np.abs(q1) + np.abs(target) + np.abs(score))
    if not np.all(score <= q1) or not np.all(penalty >= 0) or not np.all(np.abs(score - subtraction_score) <= allowed):
        raise ValueError("Lower-only mapping identity failed")
    return score, penalty, float(np.max(np.abs(score - subtraction_score)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True, action="store_true")
    parser.parse_args()
    h = load_helpers()
    h.require(h.sha(HERE / "protocol.json") == EXPECTED_PROTOCOL, "Protocol changed")
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    freeze = json.loads((HERE / "FREEZE.json").read_text(encoding="utf-8"))
    h.require(freeze["protocol_sha256"] == EXPECTED_PROTOCOL and not freeze["lower_only_scores_and_outcomes_computed"], "Freeze invalid")
    for name, digest in protocol["inputs"].items():
        h.require(h.sha(PREVIOUS / name) == digest, f"Input changed: {name}")
    outputs = ("ATTEMPT.json", "SELECTIONS.json", "SELECTIONS_SEALED.json", "STATE_OUTCOMES.json", "all_prespecified_strata.csv", "report.json", "FAILED.json")
    h.require(not any((HERE / name).exists() for name in outputs), "Existing attempt evidence; do not rerun")
    h.save_new(HERE / "ATTEMPT.json", {"utc": h.now(), "protocol_sha256": EXPECTED_PROTOCOL, "script_sha256": h.sha(Path(__file__)), "status": "reserved_once"})
    try:
        previous_selections = json.loads((PREVIOUS / "SELECTIONS.json").read_text(encoding="utf-8"))["states"]
        h.require([s["group_id"] for s in previous_selections] == list(range(128)), "State ordering mismatch")
        q1 = np.array([s["methods"]["q1"]["scores"] for s in previous_selections], dtype=np.float64)
        target = np.array([s["methods"]["mean_native_target"]["scores"] for s in previous_selections], dtype=np.float64)
        h.require(q1.shape == target.shape == (128, 9), "Score bank shape mismatch")
        scores, penalties, mapping_error = lower_only(q1, target)
        selections = []
        metadata = []
        for g, previous in enumerate(previous_selections):
            meta = {k: previous[k] for k in ("group_id", "batch", "time", "episode_seed")}
            metadata.append(meta)
            selections.append({**meta, "methods": {"lower_only": {
                **h.select(scores[g]), "scores": scores[g].tolist(), "positive_q_excess": penalties[g].tolist(),
            }}})
        h.save_new(HERE / "SELECTIONS.json", {"protocol_sha256": EXPECTED_PROTOCOL, "states": selections})
        h.save_new(HERE / "SELECTIONS_SEALED.json", {"utc": h.now(), "selections_sha256": h.sha(HERE / "SELECTIONS.json"), "previous_saved_values_read": False,
                    "lower_only_outcomes_computed": False, "prior_unrestricted_outcomes_known": True})

        prior_outcomes = json.loads((PREVIOUS / "STATE_OUTCOMES.json").read_text(encoding="utf-8"))["states"]
        h.require(len(prior_outcomes) == 128, "Missing prior value rows")
        for a, b in zip(prior_outcomes, metadata):
            h.require(all(a[k] == b[k] for k in b), "Value/state identity mismatch")
        values = np.array([s["candidate_values"] for s in prior_outcomes], dtype=np.float64)
        h.require(values.shape == (128, 9) and np.all(np.isfinite(values)), "Invalid saved V")
        selected = np.array([s["methods"]["lower_only"]["index"] for s in selections])
        raw_indices = np.array([s["methods"]["q1"]["index"] for s in previous_selections])
        selected_v = values[np.arange(128), selected]
        raw_v = values[np.arange(128), raw_indices]
        actor_v = values[:, 0]
        best_v = values.max(axis=1)
        h.require(np.array_equal(raw_v, np.array([s["methods"]["q1"]["value"] for s in prior_outcomes])), "Prior Q1 value mismatch")
        states = []
        for g, meta in enumerate(metadata):
            delta = float(selected_v[g] - raw_v[g])
            epsilon = float(h.value_epsilon(selected_v[g], raw_v[g]))
            states.append({**meta, "methods": {"lower_only": {
                "selected_index": int(selected[g]), "value": float(selected_v[g]),
                "regret": float(best_v[g] - selected_v[g]), "gain_vs_q1": delta,
                "gain_vs_actor": float(selected_v[g] - actor_v[g]), "epsilon_vs_q1": epsilon,
                "switched_vs_q1": bool(selected[g] != raw_indices[g]),
                "classification_vs_q1": "beneficial" if delta > epsilon else "harmful" if delta < -epsilon else "numerically_inconclusive",
                "score_near_tie": selections[g]["methods"]["lower_only"]["near_tie"],
            }}})
        summaries = []
        for name, indices in h.strata(metadata):
            h.require(len(indices) == (128 if name == "overall" else 16 if "/" in name else 64 if name.startswith("development_") else 32), "Stratum size mismatch")
            v = selected_v[indices]
            delta = v - raw_v[indices]
            epsilon = h.value_epsilon(v, raw_v[indices])
            switch = selected[indices] != raw_indices[indices]
            harm = (delta < -epsilon) & switch
            help_ = (delta > epsilon) & switch
            inconclusive = (np.abs(delta) <= epsilon) & switch
            n_switch = int(switch.sum())
            summaries.append({
                "stratum": name, "method": "lower_only", "states": len(indices),
                "mean_selected_value": float(v.mean()), "mean_gain_vs_q1": float(delta.mean()),
                "mean_gain_vs_actor": float((v - actor_v[indices]).mean()),
                "mean_regret": float((best_v[indices] - v).mean()), "mean_epsilon_vs_q1": float(epsilon.mean()),
                "switches_vs_q1": n_switch, "switch_rate_vs_q1": float(switch.mean()),
                "harmful_switches": int(harm.sum()), "beneficial_switches": int(help_.sum()), "inconclusive_switches": int(inconclusive.sum()),
                "harm_rate_all_states": float(harm.mean()), "harm_rate_among_switches": float(harm.sum() / n_switch) if n_switch else None,
                "mean_loss_on_harmful_switches": float(-delta[harm].mean()) if harm.any() else None,
                "mean_gain_on_beneficial_switches": float(delta[help_].mean()) if help_.any() else None,
                "minimum_gain_vs_q1": float(delta.min()), "maximum_gain_vs_q1": float(delta.max()),
                "score_near_tie_states": sum(selections[i]["methods"]["lower_only"]["near_tie"] for i in indices),
            })
        screen = [{"batch": r["stratum"], "mean_gain": r["mean_gain_vs_q1"], "mean_numerical_allowance": r["mean_epsilon_vs_q1"],
                   "pass": r["mean_gain_vs_q1"] > r["mean_epsilon_vs_q1"]} for r in summaries if r["stratum"] in ("development_1", "development_2")]
        h.require(len(screen) == 2, "Missing batch screen")
        passed = all(s["pass"] for s in screen)
        for name, digest in protocol["inputs"].items():
            h.require(h.sha(PREVIOUS / name) == digest, f"Input mutated: {name}")
        h.save_new(HERE / "STATE_OUTCOMES.json", {"states": states})
        with (HERE / "all_prespecified_strata.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        prior_report = json.loads((PREVIOUS / "report.json").read_text(encoding="utf-8"))
        report = {
            "status": "completed_cached_lower_only_feasibility", "utc": h.now(), "protocol_sha256": EXPECTED_PROTOCOL,
            "script_sha256": h.sha(Path(__file__)), "inputs": protocol["inputs"], "input_hashes_unchanged": True,
            "outputs": {name: h.sha(HERE / name) for name in ("SELECTIONS.json", "SELECTIONS_SEALED.json", "STATE_OUTCOMES.json", "all_prespecified_strata.csv")},
            "mapping_identity": {"candidate_rows": 1152, "max_abs_difference": mapping_error, "all_scores_no_greater_than_q1": True, "coefficient": 1},
            "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "scope": "CPU cached-array arithmetic"},
            "ledger": protocol["budget"], "primary_screen": screen,
            "decision": "PASS_EXPLORATORY_LOWER_ONLY_SCREEN_PROPOSE_LOGGED_DATA_PILOT_ONLY" if passed else "STOP_THIS_NONNEGATIVE_PENALTY_PROPOSAL",
            "strata": summaries, "previous_controls_reused_unchanged": prior_report["strata"],
            "scientific_superiority_established": False, "deployable_penalty_established": False,
            "original_BCA_unchanged": True, "pilot_executed": False, "full_training_hold": True,
            "limitations": ["Privileged evaluator-derived candidate targets; no learned or logged-data predictor tested.",
                            "Reused development banks, finite four-draw mean and numerical screen; no confidence or OOD claim.",
                            "Original calibrated BCA is unchanged; this positive Q-excess mapping carries no calibration guarantee."]}
        h.save_new(HERE / "report.json", report)
        print(json.dumps({"status": report["status"], "decision": report["decision"], "primary_screen": screen,
                          "overall": summaries[0], "report": str(HERE / "report.json"), "new_transitions": 0, "pilot_executed": False}))
    except BaseException as exc:
        if not (HERE / "FAILED.json").exists():
            h.save_new(HERE / "FAILED.json", {"utc": h.now(), "error": repr(exc), "attempt_preserved": True})
        raise


if __name__ == "__main__":
    main()
