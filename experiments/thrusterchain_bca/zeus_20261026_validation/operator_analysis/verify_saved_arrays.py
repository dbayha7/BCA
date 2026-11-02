r"""Independently verify the completed audit using saved numeric arrays only.

This script performs no model/runtime imports, simulator calls, training, new
decision selection, or scientific rerun. It reproduces the arithmetic review
already performed on 2026-10-26 and writes independent_review.json beside itself.
Run from the repository root with the existing native numerical environment:

    .\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/operator_analysis/verify_saved_arrays.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
RESULTS = (
    SCRIPT.parents[2]
    / "titan_20261026/payload/unifloral_operator_alignment_v1"
    / "results_20261026T221700_762125Z"
)
RECEIPT = SCRIPT.with_name("independent_review.json")
INPUT_NAMES = (
    "raw_banks.json", "raw_banks.npz", "checks.json", "execution.json", "report.json"
)
NUMERIC_RTOL = 2e-12
NUMERIC_ATOL = 2e-12


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity():
    return {
        name: {"bytes": (RESULTS / name).stat().st_size, "sha256": sha256(RESULTS / name)}
        for name in INPUT_NAMES
    }


def require(condition, message):
    # Explicit checks remain active even if an interpreter enables optimization.
    if not condition:
        raise ValueError(message)


def main():
    before = input_identity()
    raw_json = json.loads((RESULTS / "raw_banks.json").read_text(encoding="utf-8"))
    checks = json.loads((RESULTS / "checks.json").read_text(encoding="utf-8"))["analysis"]
    counts = {"array_descriptors": 0, "numeric_comparisons": 0}

    with np.load(RESULTS / "raw_banks.npz", allow_pickle=False) as archive:
        def decode(value):
            if isinstance(value, dict) and "npz_key" in value:
                array = archive[value["npz_key"]]
                require(array.dtype.str == value["dtype"], "Array dtype mismatch")
                require(list(array.shape) == value["shape"], "Array shape mismatch")
                require(
                    hashlib.sha256(array.tobytes()).hexdigest() == value["array_bytes_sha256"],
                    "Array bytes hash mismatch",
                )
                counts["array_descriptors"] += 1
                return array
            if isinstance(value, dict):
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        raw = decode(raw_json)

    def equal(actual, expected):
        require(
            np.allclose(actual, expected, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL),
            "Saved numerical result mismatch",
        )
        counts["numeric_comparisons"] += 1

    def correlation(x, y):
        if np.ptp(x) == 0:
            return None, "constant_predictor"
        if np.ptp(y) == 0:
            return None, "constant_outcome"
        # Independent implementation: NumPy correlation, not the audit helper.
        return float(np.corrcoef(x, y)[0, 1]), None

    def magnitude(x):
        return {
            "mean": float(np.mean(x)), "median": float(np.median(x)),
            "min": float(np.min(x)), "max": float(np.max(x)),
            "rms": float(np.sqrt(np.mean(x * x))),
        }

    def check_correlation(x, y, reported):
        value, reason = correlation(x, y)
        require(reported["defined"] == (value is not None), "Correlation definition mismatch")
        require(reported["reason"] == reason, "Undefined correlation reason mismatch")
        require(reported["rows"] == len(x), "Correlation row count mismatch")
        if value is not None:
            equal(value, reported["value"])
        return value, reason

    def check_spread(values, reported):
        equal(values, reported["per_draw"])
        equal(np.mean(values), reported["mean"])
        equal(min(values), reported["min"])
        equal(max(values), reported["max"])
        require(reported["defined_draws"] == len(values), "Draw count mismatch")

    summary = {}
    for bank_name in ("development", "fitting"):
        bank = raw[bank_name]
        out = checks[bank_name]
        n = len(bank["u"])
        q = bank["q_current"].astype("float64")
        u = bank["u"].astype("float64")
        y = bank["native_y"].astype("float64")
        residual = q.min(1)[None, :] - y
        require(y.shape == (4, n) and q.shape == (n, 2), "Q/target shapes mismatch")
        require(len(bank["metadata"]) == n, "Metadata length mismatch")
        require(
            np.all(np.isfinite(q)) and np.all(np.isfinite(u)) and np.all(np.isfinite(y)),
            "Nonfinite Q/radius/target",
        )
        require(np.all(u >= 0), "Negative radius")
        require(bool(np.all(bank["u_valid"])) and bool(np.all(bank["u_usable"])), "Invalid radius")
        expected_seeds = (
            [2026103041, 2026103042, 2026103043, 2026103044]
            if bank_name == "development"
            else [2026103051, 2026103052, 2026103053, 2026103054]
        )
        require(bank["smoothing_seeds"] == expected_seeds, "Declared seeds mismatch")
        equal(np.clip(bank["raw_noise"], -.5, .5), bank["clipped_noise"])
        equal(
            np.clip(bank["target_actor_next"][None, :, :] + bank["clipped_noise"], -1, 1),
            bank["next_actions"],
        )
        target = (
            bank["rewards"][None, :]
            + (np.float32(1) - bank["done"][None, :])
            * np.float32(.98) * bank["target_q"].min(2)
        )
        require(np.array_equal(target, bank["native_y"]), "Float32 target arithmetic mismatch")
        require(np.all(np.isin(bank["done"], [0, 1])), "Nonbinary terminal mask")
        summary[bank_name] = {
            "pooled_u_absr_eachdraw": [correlation(u, np.abs(row))[0] for row in residual],
            "U": magnitude(u),
            "absR_eachdraw": [magnitude(np.abs(row)) for row in residual],
            "R_eachdraw": [magnitude(row) for row in residual],
            "containment_n_eachdraw": [int(np.sum(np.abs(row) <= u)) for row in residual],
            "terminal_rows": int(np.sum(bank["done"])),
            "smoothing_y_std_across4draws_mean": float(np.std(y, axis=0).mean()),
            "smoothing_y_std_across4draws_median": float(np.median(np.std(y, axis=0))),
            "smoothing_y_std_across4draws_max": float(np.std(y, axis=0).max()),
            "mean_absR_over_draws_then_rows": float(np.abs(residual).mean()),
            "mean_abs_meanR_over_draws_then_rows": float(np.abs(residual.mean(0)).mean()),
        }

        if bank_name == "fitting":
            groups = {}
            for index, meta in enumerate(bank["metadata"]):
                groups.setdefault(meta["group_id"], []).append(index)
            require(len(groups) == out["unique_episode_groups"] == 221, "Fitting group count mismatch")
            equal(n, out["rows"])
            for draw, reported in enumerate(out["draws"]):
                require(reported["draw"] == draw, "Draw index mismatch")
                require(reported["noise_seed"] == expected_seeds[draw], "Draw seed mismatch")
                check_correlation(u, np.abs(residual[draw]), reported["pearson_u_abs_residual"])
                contained = np.abs(residual[draw]) <= u
                equal(contained, reported["radius_containment"])
                equal(residual[draw], reported["residual"])
                equal(y[draw], reported["native_y"])
                equal(contained.mean(), reported["row_weighted_radius_containment"])
                equal(
                    np.mean([contained[indices].mean() for indices in groups.values()]),
                    reported["equal_episode_group_radius_containment"],
                )
                for group in reported["groups"]:
                    indices = groups[group["group_id"]]
                    require(group["rows"] == len(indices), "Fitting group size mismatch")
                    equal(contained[indices].mean(), group["containment_fraction"])
                for key, values in (
                    ("U_magnitude", u),
                    ("absolute_residual_magnitude", np.abs(residual[draw])),
                    ("signed_residual_magnitude", residual[draw]),
                ):
                    for statistic, value in magnitude(values).items():
                        equal(value, reported[key][statistic])
            for key, values in (
                ("pearson_u_abs_residual", summary[bank_name]["pooled_u_absr_eachdraw"]),
                ("row_containment", [d["row_weighted_radius_containment"] for d in out["draws"]]),
                ("equal_group_containment", [d["equal_episode_group_radius_containment"] for d in out["draws"]]),
            ):
                check_spread(values, out["draw_summary"][key])
            summary[bank_name]["episode_group_sizes"] = {
                str(size): sum(len(indices) == size for indices in groups.values())
                for size in sorted({len(indices) for indices in groups.values()})
            }
            continue

        v = bank["saved_values"].astype("float64")
        error = q[:, 0] - v
        head_gap = q[:, 0] - q.min(1)
        target_gap = y - v[None, :]
        difference = np.abs(error[None, :] - (residual + head_gap[None, :] + target_gap))
        allowed = 64 * np.finfo("float64").eps * (
            1 + np.abs(error[None, :]) + np.abs(residual)
            + np.abs(head_gap[None, :]) + np.abs(target_gap)
        )
        require(np.all(difference <= allowed), "Accounting identity roundoff limit failed")
        equal(difference.max(), out["accounting_identity"]["max_abs_difference"])
        equal(np.max(difference / allowed), out["accounting_identity"]["max_fraction_of_allowed_roundoff"])
        summary[bank_name].update({
            "E": magnitude(error), "H": magnitude(head_gap),
            "G_eachdraw": [magnitude(row) for row in target_gap],
            "negative_E_rows": int(np.sum(error < 0)),
            "zero_H_rows": int(np.sum(head_gap == 0)),
            "accounting_pairs": int(difference.size),
            "accounting_max_error": float(difference.max()),
        })
        require(
            [(m["group_id"], m["candidate_id"]) for m in bank["metadata"]]
            == [(group, candidate) for group in range(128) for candidate in range(9)],
            "Candidate row ordering mismatch",
        )
        require(
            len({(m["batch"], m["episode_seed"]) for m in bank["metadata"][::9]}) == 128,
            "Development episode identities mismatch",
        )
        require(
            np.array_equal(bank["physics_rewards"].astype(np.float32), bank["rewards"]),
            "Physics reward float32 cast mismatch",
        )
        for draw, reported in enumerate(out["draws"]):
            require(len(reported["states"]) == 128, "State count mismatch")
            require(reported["noise_seed"] == expected_seeds[draw], "Draw seed mismatch")
            states = []
            for group, state in enumerate(reported["states"]):
                indices = slice(9 * group, 9 * group + 9)
                meta = bank["metadata"][9 * group]
                for key in ("group_id", "batch", "time", "episode_seed"):
                    require(state[key] == meta[key], "State identity mismatch")
                E, H = error[indices], head_gap[indices]
                R, G, U = residual[draw, indices], target_gap[draw, indices], u[indices]
                relationships = {
                    "u_abs_residual": (U, np.abs(R)), "u_signed_error": (U, E),
                    "u_positive_error": (U, np.maximum(E, 0)), "u_absolute_error": (U, np.abs(E)),
                    "abs_residual_signed_error": (np.abs(R), E),
                    "abs_residual_positive_error": (np.abs(R), np.maximum(E, 0)),
                    "R_with_E": (R, E), "H_with_E": (H, E), "G_with_E": (G, E),
                }
                computed = {
                    key: check_correlation(x, z, state["relationships"][key])
                    for key, (x, z) in relationships.items()
                }
                terms = {"E": E, "R": R, "H": H, "G": G}
                rms = {key: float(np.std(values)) for key, values in terms.items()}
                for key, values in terms.items():
                    equal(values, state["terms"][key])
                    equal(rms[key], state["centered_action_rms"][key])
                equal(U, state["radius"])
                equal(y[draw, indices], state["native_y"])
                equal(np.abs(R) <= U, state["radius_containment"])
                equal(np.mean(np.abs(R) <= U), state["radius_containment_fraction"])
                states.append({
                    "meta": meta, "corr": computed, "rms": rms,
                    "containment": float(np.mean(np.abs(R) <= U)),
                })
            strata = [(states, reported["overall"])]
            for batch, stratum in reported["by_batch"].items():
                strata.append(([s for s in states if s["meta"]["batch"] == batch], stratum))
            for time, stratum in reported["by_time"].items():
                strata.append(([s for s in states if s["meta"]["time"] == int(time)], stratum))
            for batch, times in reported["by_batch_time"].items():
                for time, stratum in times.items():
                    selected = [
                        s for s in states
                        if s["meta"]["batch"] == batch and s["meta"]["time"] == int(time)
                    ]
                    require(len(selected) == 16, "Batch/time cell count mismatch")
                    strata.append((selected, stratum))
            require(len(strata) == 15, "Stratum count mismatch")
            for selected, stratum in strata:
                require(len(selected) == stratum["states"], "Stratum state count mismatch")
                for key in relationships:
                    values = [s["corr"][key][0] for s in selected if s["corr"][key][0] is not None]
                    reasons = {}
                    for state in selected:
                        reason = state["corr"][key][1]
                        if reason:
                            reasons[reason] = reasons.get(reason, 0) + 1
                    comparison = stratum["relationships"][key]
                    require(len(values) == comparison["defined_states"], "Defined state count mismatch")
                    require(len(selected) - len(values) == comparison["undefined_states"], "Undefined state count mismatch")
                    require(reasons == comparison["undefined_reasons"], "Undefined reasons mismatch")
                    if values:
                        equal(np.mean(values), comparison["equal_state_mean"])
                    else:
                        require(comparison["equal_state_mean"] is None, "Undefined mean mismatch")
                for key in terms:
                    equal(np.mean([s["rms"][key] for s in selected]), stratum["mean_state_centered_action_rms"][key])
                equal(np.mean([s["containment"] for s in selected]), stratum["equal_state_radius_containment"])
        for key in out["draw_summary"]["relationships"]:
            check_spread(
                [d["overall"]["relationships"][key]["equal_state_mean"] for d in out["draws"]],
                out["draw_summary"]["relationships"][key],
            )
        for key in out["draw_summary"]["centered_action_rms"]:
            check_spread(
                [d["overall"]["mean_state_centered_action_rms"][key] for d in out["draws"]],
                out["draw_summary"]["centered_action_rms"][key],
            )
        check_spread(
            [d["overall"]["equal_state_radius_containment"] for d in out["draws"]],
            out["draw_summary"]["radius_containment"],
        )

    require(before == input_identity(), "Input artifacts changed during arithmetic verification")
    for bank_summary in summary.values():
        bank_summary["pooled_u_absr_mean_over4draws"] = float(np.mean(bank_summary["pooled_u_absr_eachdraw"]))
    receipt = {
        "schema": "unifloral.operator_alignment.independent_saved_arithmetic.v1",
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved-array arithmetic verification and labelled posthoc descriptive summaries only",
        "results_directory": str(RESULTS),
        "script_sha256": sha256(SCRIPT),
        "inputs": before,
        "input_hashes_unchanged_after_review": True,
        "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__},
        "command_from_repository_root": ".\\.venv\\Scripts\\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/operator_analysis/verify_saved_arrays.py",
        "comparison_counts": {
            **counts, "development_state_draws": 512,
            "development_stratum_draws": 60, "fitting_draws": 4,
            "fitting_episode_groups": 221, "accounting_row_draw_pairs": 4608,
        },
        "comparison_tolerances": {
            "numeric_rtol": NUMERIC_RTOL, "numeric_atol": NUMERIC_ATOL,
            "accounting": "64*float64_epsilon*(1+abs(E)+abs(R)+abs(H)+abs(G))",
            "native_float32_target": "Exact array equality for the stored source-order expression",
        },
        "supplementary_posthoc_saved_array_descriptives": summary,
        "limitations": [
            "This review does not replay neural networks, dynamics, checkpoint restoration or seeded noise generation. Parent provenance checks authenticate the executed sources and checkpoint.",
            "No model/runtime imports, simulation, training, new action selection, penalty sweep, confirmation access or policy comparison occurred in this script.",
            "Fitting pooled correlations and development within-state correlations use different aggregations; their difference is not a direct generalization-drop estimate. Supplementary development pooled correlations are labelled posthoc.",
            "Four smoothing draws are one finite declared noise bank, not an exact expected operator or fourfold increase in independent episodes. Reported across-draw standard deviations use ddof=0.",
            "Mean absolute residual and absolute mean residual are distinct quantities; both are explicitly labelled.",
            "Fitting-bank containment is descriptive, not held-out calibration or empirical evidence of a conformal coverage guarantee.",
            "E=R+H+G is algebraic; correlations and centered RMS do not identify causal contributions or establish decision utility.",
            "This receipt verifies the stored numerical analysis and does not authorize training or further evaluator calls.",
        ],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"], "receipt": str(RECEIPT),
        "comparison_counts": receipt["comparison_counts"],
        "script_sha256": receipt["script_sha256"],
        "receipt_sha256": sha256(RECEIPT),
    }, indent=2))


if __name__ == "__main__":
    main()
