"""One reserved, frozen comparison using cached development values only.

No model, simulator or experiment-runtime imports. Numeric selection records are
sealed before the saved continuation-value array is decoded. Existing attempts
are never overwritten or automatically retried.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
EXPECTED_PROTOCOL = "287c0b0dd050fa87d20ec17a36004f976d3434b7a6f617278e0b228a911f116c"
METHODS = ("q1", "twin_min", "actor", "mean_native_target")
ATOL = RTOL = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def select(scores):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.shape == (9,) and np.all(np.isfinite(scores)), "Invalid scores")
    best = float(np.max(scores))
    near = np.flatnonzero(np.abs(scores - best) <= ATOL + RTOL * abs(best))
    idx = int(near[0])
    ordered = np.sort(scores)
    return {
        "index": idx,
        "exact_argmax_index": int(np.argmax(scores)),
        "near_best_indices": near.tolist(),
        "near_tie": len(near) > 1,
        "maximum_score": best,
        "selected_score": float(scores[idx]),
        "top_two_gap": float(ordered[-1] - ordered[-2]),
    }


def value_epsilon(method, control):
    return ATOL + RTOL * np.maximum(np.abs(method), np.abs(control))


def strata(metadata):
    all_rows = np.arange(128)
    yield "overall", all_rows
    for batch in ("development_1", "development_2"):
        yield batch, np.array([i for i, m in enumerate(metadata) if m["batch"] == batch])
    for time in (0, 25, 50, 75):
        yield f"time{time}", np.array([i for i, m in enumerate(metadata) if m["time"] == time])
    for batch in ("development_1", "development_2"):
        for time in (0, 25, 50, 75):
            yield f"{batch}/time{time}", np.array([
                i for i, m in enumerate(metadata) if m["batch"] == batch and m["time"] == time
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    protocol_path = HERE / "protocol.json"
    require(sha(protocol_path) == EXPECTED_PROTOCOL, "Frozen protocol changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    freeze = json.loads((HERE / "FREEZE.json").read_text(encoding="utf-8"))
    require(freeze["protocol_sha256"] == EXPECTED_PROTOCOL, "Freeze mismatch")
    require(not freeze["new_selector_outcomes_computed"], "Invalid pre-outcome freeze")
    results = ROOT / protocol["input_results_directory"]
    for name, digest in protocol["inputs"].items():
        require(sha(results / name) == digest, f"Input hash mismatch: {name}")
    require(not any((HERE / name).exists() for name in (
        "ATTEMPT.json", "SELECTIONS.json", "SELECTIONS_SEALED.json", "STATE_OUTCOMES.json",
        "all_prespecified_strata.csv", "report.json", "FAILED.json"
    )), "Existing attempt evidence: inspect it; do not rerun")
    save_new(HERE / "ATTEMPT.json", {
        "utc": now(), "protocol_sha256": EXPECTED_PROTOCOL,
        "script_sha256": sha(Path(__file__)), "status": "reserved_once",
        "scope": "cached development arithmetic only",
    })

    try:
        raw = json.loads((results / "raw_banks.json").read_text(encoding="utf-8"))["development"]
        metadata = raw["metadata"]
        require([(m["group_id"], m["candidate_id"]) for m in metadata] == [
            (g, c) for g in range(128) for c in range(9)
        ], "Changed state/candidate ordering")
        state_metadata = [metadata[g * 9] for g in range(128)]
        require(len({(m["batch"], m["episode_seed"]) for m in state_metadata}) == 128,
                "Unexpected episode identities")
        require(raw["smoothing_seeds"] == protocol["smoothing_seeds"], "Changed noise draws")
        for g in range(128):
            for m in metadata[g * 9:(g + 1) * 9]:
                require(all(m[k] == state_metadata[g][k] for k in ("group_id", "batch", "time", "episode_seed")),
                        "Inconsistent state metadata")

        verified_arrays = []
        with np.load(results / "raw_banks.npz", allow_pickle=False) as archive:
            def decode(name):
                desc = raw[name]
                value = archive[desc["npz_key"]]
                require(value.dtype.str == desc["dtype"] and list(value.shape) == desc["shape"],
                        f"Array schema mismatch: {name}")
                require(hashlib.sha256(value.tobytes()).hexdigest() == desc["array_bytes_sha256"],
                        f"Array hash mismatch: {name}")
                require(np.all(np.isfinite(value)), f"Nonfinite array: {name}")
                verified_arrays.append(name)
                return value

            q = decode("q_current").astype(np.float64).reshape(128, 9, 2)
            targets = decode("native_y").astype(np.float64)
            require(targets.shape == (4, 1152), "Target bank shape mismatch")
            mean_target = targets.mean(axis=0).reshape(128, 9)
            actions = decode("actions").reshape(128, 9, 4)
            actors = decode("saved_actor_actions")
            require(np.array_equal(actions[:, 0, :], actors), "Actor candidate identity mismatch")
            scores = {"q1": q[:, :, 0], "twin_min": q.min(axis=2), "mean_native_target": mean_target}
            selections = []
            for g, meta in enumerate(state_metadata):
                entry = {k: meta[k] for k in ("group_id", "batch", "time", "episode_seed")}
                entry["methods"] = {}
                for method in METHODS:
                    if method == "actor":
                        entry["methods"][method] = {
                            "index": 0, "near_tie": False, "selection_rule": "frozen_actor_candidate_0",
                            "scores": None,
                        }
                    else:
                        entry["methods"][method] = {**select(scores[method][g]), "scores": scores[method][g].tolist()}
                selections.append(entry)
            save_new(HERE / "SELECTIONS.json", {"protocol_sha256": EXPECTED_PROTOCOL, "states": selections})
            save_new(HERE / "SELECTIONS_SEALED.json", {
                "utc": now(), "selections_sha256": sha(HERE / "SELECTIONS.json"),
                "saved_value_array_decoded": False, "outcomes_computed": False,
                "note": "Prior operator diagnostics were known; this is not a fresh blind experiment.",
            })
            values = decode("saved_values").astype(np.float64).reshape(128, 9)

        selected_indices = {
            method: np.array([s["methods"][method]["index"] for s in selections]) for method in METHODS
        }
        selected_values = {
            method: values[np.arange(128), indices] for method, indices in selected_indices.items()
        }
        best_values = values.max(axis=1)
        state_outcomes = []
        for g, s in enumerate(selections):
            out = {k: s[k] for k in ("group_id", "batch", "time", "episode_seed")}
            out.update({"candidate_values": values[g].tolist(), "best_candidate_value": float(best_values[g]), "methods": {}})
            for method in METHODS:
                value = selected_values[method][g]
                raw_value = selected_values["q1"][g]
                delta = float(value - raw_value)
                epsilon = float(value_epsilon(value, raw_value))
                switched = bool(selected_indices[method][g] != selected_indices["q1"][g])
                out["methods"][method] = {
                    "selected_index": int(selected_indices[method][g]),
                    "value": float(value), "regret": float(best_values[g] - value),
                    "gain_vs_q1": delta, "gain_vs_actor": float(value - selected_values["actor"][g]),
                    "epsilon_vs_q1": epsilon, "switched_vs_q1": switched,
                    "classification_vs_q1": "beneficial" if delta > epsilon else "harmful" if delta < -epsilon else "numerically_inconclusive",
                    "score_near_tie": s["methods"][method]["near_tie"],
                }
            state_outcomes.append(out)

        summaries = []
        for name, indices in strata(state_metadata):
            require(len(indices) == (128 if name == "overall" else 16 if "/" in name else 64 if name.startswith("development_") else 32),
                    f"Stratum size mismatch: {name}")
            for method in METHODS:
                method_values = selected_values[method][indices]
                baseline = selected_values["q1"][indices]
                delta = method_values - baseline
                epsilon = value_epsilon(method_values, baseline)
                switch = selected_indices[method][indices] != selected_indices["q1"][indices]
                harmful = (delta < -epsilon) & switch
                beneficial = (delta > epsilon) & switch
                inconclusive = (np.abs(delta) <= epsilon) & switch
                n_switch = int(switch.sum())
                summaries.append({
                    "stratum": name, "method": method, "states": len(indices),
                    "mean_selected_value": float(method_values.mean()),
                    "mean_gain_vs_q1": float(delta.mean()),
                    "mean_gain_vs_actor": float((method_values - selected_values["actor"][indices]).mean()),
                    "mean_regret": float((best_values[indices] - method_values).mean()),
                    "mean_epsilon_vs_q1": float(epsilon.mean()),
                    "switches_vs_q1": n_switch, "switch_rate_vs_q1": float(switch.mean()),
                    "harmful_switches": int(harmful.sum()), "beneficial_switches": int(beneficial.sum()),
                    "inconclusive_switches": int(inconclusive.sum()),
                    "harm_rate_all_states": float(harmful.mean()),
                    "harm_rate_among_switches": float(harmful.sum() / n_switch) if n_switch else None,
                    "mean_loss_on_harmful_switches": float(-delta[harmful].mean()) if harmful.any() else None,
                    "mean_gain_on_beneficial_switches": float(delta[beneficial].mean()) if beneficial.any() else None,
                    "minimum_gain_vs_q1": float(delta.min()), "maximum_gain_vs_q1": float(delta.max()),
                    "score_near_tie_states": sum(selections[i]["methods"][method]["near_tie"] for i in indices),
                })
        batch_screen = [r for r in summaries if r["method"] == "mean_native_target" and r["stratum"] in ("development_1", "development_2")]
        screen = [{"batch": r["stratum"], "mean_gain": r["mean_gain_vs_q1"], "mean_numerical_allowance": r["mean_epsilon_vs_q1"],
                   "pass": r["mean_gain_vs_q1"] > r["mean_epsilon_vs_q1"]} for r in batch_screen]
        passed = all(s["pass"] for s in screen)
        require(len(screen) == 2, "Missing primary batch screen")
        for name, digest in protocol["inputs"].items():
            require(sha(results / name) == digest, f"Input changed during execution: {name}")
        save_new(HERE / "STATE_OUTCOMES.json", {"states": state_outcomes})
        with (HERE / "all_prespecified_strata.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        report = {
            "status": "completed_cached_mechanism_comparison", "utc": now(),
            "protocol_sha256": EXPECTED_PROTOCOL, "script_sha256": sha(Path(__file__)),
            "inputs": protocol["inputs"], "input_hashes_unchanged": True,
            "outputs": {name: sha(HERE / name) for name in ("SELECTIONS.json", "SELECTIONS_SEALED.json", "STATE_OUTCOMES.json", "all_prespecified_strata.csv")},
            "verified_array_descriptors": verified_arrays,
            "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "purpose": "CPU arithmetic on saved arrays only"},
            "ledger": protocol["budget"], "primary_screen": screen,
            "decision": "PASS_EXPLORATORY_SCREEN_PROPOSE_ONLY" if passed else "STOP_OPERATOR_REPLACEMENT_BRANCH",
            "scientific_superiority_established": False, "deployable_policy_established": False,
            "full_training_hold": True, "pilot_executed": False, "strata": summaries,
            "limitations": [
                "Exploratory reused development banks; no fresh confirmation or training-seed replication.",
                "Mean native target uses evaluator-derived reward and next state; privileged mechanism control only.",
                "Original Q1-kappa*U penalty remains unchanged and is not evaluated at a new coefficient here.",
                "Numerical tie screen is not statistical uncertainty, a confidence bound, or superiority evidence.",
                "Actor control is candidate0, not a newly tuned actor-distance penalty.",
                "Four declared target draws are a finite noise bank, not exact expected-target integration.",
            ],
        }
        save_new(HERE / "report.json", report)
        print(json.dumps({"status": report["status"], "decision": report["decision"], "primary_screen": screen,
                          "report": str(HERE / "report.json"), "new_transitions": 0, "pilot_executed": False}))
    except BaseException as exc:
        if not (HERE / "FAILED.json").exists():
            save_new(HERE / "FAILED.json", {"utc": now(), "error": repr(exc), "attempt_preserved": True})
        raise


if __name__ == "__main__":
    main()
