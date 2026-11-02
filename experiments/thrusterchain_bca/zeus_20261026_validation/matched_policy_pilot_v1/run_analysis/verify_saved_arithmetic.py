"""Independent saved-data arithmetic only: no model imports or simulator calls."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
RUN = PILOT / "run" / "evaluation"
checks = 0

def check(value, label):
    global checks
    if not value:
        raise AssertionError(label)
    checks += 1

def equal(actual, expected, label):
    check(np.array_equal(np.asarray(actual), np.asarray(expected)), label)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

protocol = json.loads((PILOT / "protocol.json").read_text())
report = json.loads((RUN / "report.json").read_text())
check(digest(PILOT / "protocol.json") == report["protocol_sha256"], "protocol hash")
arrays, returns = {}, {}
for name, receipt in report["policy_results"].items():
    path = RUN / receipt["path"]
    check(path.stat().st_size == receipt["bytes"], name + " bytes")
    check(digest(path) == receipt["sha256"], name + " hash")
    with np.load(path, allow_pickle=False) as data:
        arrays[name] = {key: data[key] for key in data.files}
    a = arrays[name]
    check(a["rewards"].shape == (32, 100), name + " reward shape")
    check(a["states"].shape == (32, 101, 13), name + " state shape")
    check(a["actions"].shape == (32, 100, 4), name + " action shape")
    check(a["rewards"].dtype == np.float64, name + " reward dtype")
    check(a["states"].dtype == np.float64, name + " state dtype")
    check(a["actions"].dtype == np.float32, name + " deployed action dtype")
    check(all(np.isfinite(value).all() for value in a.values()), name + " finite")
    check(np.max(np.abs(a["actions"])) <= 1.0, name + " bounded actions")
    equal(a["reset_seeds"], protocol["evaluation_seeds"], name + " reset seeds")
    total = np.zeros(32, dtype=np.float64)
    for t in range(100):
        total += protocol["gamma"] ** t * a["rewards"][:, t]
    equal(total, a["returns"], name + " reward to return exact")
    equal(total, report["returns_by_policy"][name], name + " report returns exact")
    equal(total.mean(), report["mean_discounted_returns"][name], name + " mean exact")
    returns[name] = total

starts = arrays["original"]["states"][:, 0]
for name, a in arrays.items():
    equal(a["states"][:, 0], starts, name + " common start")
    z = (a["states"][:, 1:] - arrays["original"]["states"][:, 1:]) / (np.asarray(report["normalization"]["std"][:13]) + .001)
    rms = np.sqrt(np.mean(z*z, axis=2))
    equal(rms.mean(), report["action_and_state_displacement"][name]["mean_standardized_physical_state_rms"], name + " trajectory shift")
    equal(rms.mean(axis=1), report["action_and_state_displacement"][name]["standardized_physical_state_rms_by_episode"], name + " trajectory shifts by episode")

rng = np.random.Generator(np.random.PCG64(protocol["evaluation_bootstrap_seed"]))
indices = rng.integers(0, 32, size=(protocol["evaluation_bootstrap_draws"], 32))
check(hashlib.sha256(indices.astype(np.int64).tobytes(order="C")).hexdigest() == report["bootstrap"]["indices_sha256"], "bootstrap indices")
rows = []
for stream in (0, 1):
    for control in ("native", "distance", "original"):
        left = returns[f"seed{stream}_bca"]
        right = returns["original" if control == "original" else f"seed{stream}_{control}"]
        difference = left - right
        band = protocol["return_band_atol"] + protocol["return_band_rtol"] * np.maximum(abs(left), abs(right))
        interval = np.quantile(difference[indices].mean(axis=1), [.025, .975], method="linear")
        expected = report["comparisons_by_continuation_stream"][str(stream)][control]
        computed = {
            "paired_differences": difference.tolist(),
            "mean_gain": float(difference.mean()),
            "mean_numerical_band": float(band.mean()),
            "positive_screen": bool(difference.mean() > band.mean()),
            "improved_episodes": int(np.count_nonzero(difference > band)),
            "harmed_episodes": int(np.count_nonzero(difference < -band)),
            "numerical_ties": int(np.count_nonzero(abs(difference) <= band)),
            "harm_episode_indices": np.flatnonzero(difference < -band).tolist(),
            "mean_gain_episode_paired_bootstrap_95pct": interval.tolist(),
        }
        for key, value in computed.items():
            equal(value, expected[key], f"stream{stream} {control} {key}")
        rows.append({"stream": stream, "control": control, **{k: v for k, v in computed.items() if k not in ("paired_differences", "harm_episode_indices")}})

passed = all(row["positive_screen"] for row in rows if row["control"] != "original")
check(passed == report["primary_screen_passed"], "screen")
check(report["decision"] == ("promising_exploratory_screen" if passed else "stop_this_pilot_without_tuning"), "decision")
check(sum(a["rewards"].size for a in arrays.values()) == report["counts"]["completed_transition_rows"] == 22400, "transition count")
check(sum(a["rewards"].shape[0] for a in arrays.values()) == report["counts"]["completed_episodes"] == 224, "episode count")
output = {
    "status": "passed_independent_saved_arithmetic_review",
    "checks": checks,
    "protocol_sha256": digest(PILOT / "protocol.json"),
    "evaluation_report_sha256": digest(RUN / "report.json"),
    "verification_scope": "Only NumPy calculations on already saved rewards, trajectories, returns and protocol; zero model imports, simulator calls, fitting, calibration or confirmation access.",
    "mean_discounted_returns": {name: float(value.mean()) for name, value in returns.items()},
    "paired_comparisons": rows,
    "primary_screen_passed": passed,
    "equal_stream_means_descriptive": {
        arm: float(np.mean([returns[f"seed{s}_{arm}"].mean() for s in (0, 1)]))
        for arm in ("native", "bca", "distance")
    },
    "limitations": [
        "Two continuation streams share a single trained checkpoint and the same 32 nominal reset seeds; these are not 64 independent starts or independent training initialization replications.",
        "Episode bootstrap intervals condition on these fixed trained actors; they do not establish training-seed generalization or OOD benefit.",
        "Current original BCA kappa3 passed a numerical-direction exploratory screen after development selection and adaptive reuse of calibration episodes; no fresh confirmatory guarantee.",
        "Logged action displacement cannot be independently recomputed without the actor forward outputs, which are not persisted; trajectory displacement was independently verified."
    ],
}
out_path = HERE / "INDEPENDENT_ARITHMETIC_REVIEW.json"
out_path.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
