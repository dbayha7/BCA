"""One reset only; never invoke the full dataset loader or a learned model.

Run manually in the existing WSL environment. This is a diagnostic, not a
dataset repair, a validation pass, or authorization to restart training.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import struct
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
VALIDATION = HERE.parents[1]
PACKAGE = HERE.parent / "minimal_comparison"
MANIFEST_SHA = "400490f6f2c3a5ab702e80733fcf28a34a1308f7d06df6df476bc8235e593c1d"
EPISODE = 16
EST = timezone(timedelta(hours=-5))
SOFT_LIMIT_SECONDS = 110
COORDINATES = [name for j in range(5) for name in (f"position_{j}", f"speed_{j}")] + [
    "current", "wave_1", "wave_2"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bits(value):
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def ulps(a, b):
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    def ordered(value):
        raw = bits(value)
        return (1 << 63) - (raw & ((1 << 63) - 1)) if raw >> 63 else (1 << 63) + raw
    return abs(ordered(a) - ordered(b))


def describe_values(expected, actual, np):
    """Preserve dtype/shape; never cast before the original array_equal check."""
    result = {"expected_dtype": expected.dtype.str, "actual_dtype": actual.dtype.str,
              "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
              "original_array_equal": bool(np.array_equal(actual, expected)),
              "same_dtype": expected.dtype == actual.dtype,
              "same_shape": expected.shape == actual.shape,
              "byte_equal": expected.dtype == actual.dtype and expected.shape == actual.shape
                            and expected.tobytes() == actual.tobytes()}
    if expected.shape != (13,) or actual.shape != (13,):
        return result
    rows = []
    for i, (e, a) in enumerate(zip(expected, actual)):
        e, a = float(e), float(a)
        finite = math.isfinite(e) and math.isfinite(a)
        rows.append({"coordinate": i, "name": COORDINATES[i],
                     "expected": e if math.isfinite(e) else str(e),
                     "actual": a if math.isfinite(a) else str(a),
                     "expected_hex": e.hex(), "actual_hex": a.hex(),
                     "expected_float64_bits": f"0x{bits(e):016x}",
                     "actual_float64_bits": f"0x{bits(a):016x}",
                     "actual_minus_expected": a - e if finite else None,
                     "absolute_difference": abs(a - e) if finite else None,
                     "ulp_distance": ulps(e, a), "numerically_equal": e == a})
    result.update(coordinates=rows, differing_coordinates=[r["coordinate"] for r in rows
                                                           if not r["numerically_equal"]])
    return result


def verify_seal():
    require(sha(PACKAGE / "MANIFEST.json") == MANIFEST_SHA, "Frozen manifest changed")
    manifest = json.loads((PACKAGE / "MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (REPO / relative).resolve()
        require(path.is_relative_to(REPO), "Sealed path escapes repository")
        require(sha(path) == expected, "Sealed file changed: " + relative)
    return manifest


def one_reset(env, seed, options, counts):
    try:
        counts["reset_calls_attempted"] += 1
        observation, info = env.reset(seed=seed, options=options)
        counts["reset_calls_completed"] += 1
        return observation, info
    finally:
        env.close()


def forbidden_step(*args, **kwargs):
    raise RuntimeError("Simulator steps are forbidden in this reset-only diagnostic")


def execute(report):
    manifest = verify_seal()
    config = json.loads((PACKAGE / "config.json").read_text())
    dataset_path = (REPO / config["dataset_relative"]).resolve()
    require(dataset_path.is_relative_to(REPO), "Dataset path escapes repository")
    archive_sha = sha(dataset_path)
    require(archive_sha == config["dataset_archive_sha256"], "Dataset archive changed")
    report["identity"] = {"manifest_sha256": MANIFEST_SHA, "verified_files": len(manifest["files"]),
                          "dataset_path": str(dataset_path), "dataset_archive_sha256": archive_sha,
                          "script_sha256": sha(__file__)}
    report["prior_attempt"] = {"path": str(PACKAGE / "run"), "files": {
        name: sha(PACKAGE / name) for name in ("LAUNCH_APPROVAL.json", "run/ATTEMPT.json",
                                              "run/RESULT.json", "run/warmup_s0/report.json")}}
    prior = json.loads((PACKAGE / "run/RESULT.json").read_text())
    report["prior_attempt"]["elapsed_seconds"] = prior["elapsed_seconds"]
    report["prior_attempt"]["accounting_preserved"] = prior["accounting"]

    import numpy as np
    import gymnasium
    # Read the stored row directly. D.load would validate all episodes and step dynamics.
    with np.load(dataset_path, allow_pickle=False) as archive:
        envelope = json.loads(str(archive["manifest"]))
        metadata = envelope["metadata"]
        expected = archive["states"][EPISODE, 0].copy()
        expected_obs = archive["observations"][EPISODE, 0].copy()
        seed = int(archive["episode_seeds"][EPISODE])
        component = int(archive["reset_component"][EPISODE])
    require(envelope["sha256"] == config["dataset_content_sha256"], "Stored content digest differs")
    require(metadata["seed"] == config["dataset_entropy"] and metadata["world"] == "stall3",
            "Wrong dataset identity")
    report["runtime"] = {"executable": sys.executable, "python": sys.version,
                         "platform": platform.platform(), "machine": platform.machine(),
                         "numpy": np.__version__, "numpy_path": np.__file__,
                         "numpy_build": np.show_config(mode="dicts"),
                         "gymnasium": importlib.metadata.version("gymnasium"),
                         "gymnasium_path": gymnasium.__file__,
                         "dataset_generator": metadata["generator"]["numpy_version"]}
    v3 = VALIDATION / "knee_edge_ood_v3"
    candidate1 = VALIDATION / "knee_s10_candidate_v1"
    for path in reversed((v3, candidate1)):
        sys.path.insert(0, str(path))
    import recorder_wide as recorder
    import worlds
    import core
    environment = core.load_environment()
    for module, expected_path in ((recorder, candidate1 / "recorder_wide.py"),
                                  (worlds, v3 / "worlds.py"), (core, v3 / "core.py"),
                                  (environment, core.CORE_ROOT / "thrusterchain/environment.py")):
        require(Path(module.__file__).resolve() == expected_path.resolve(), "Unexpected module path")
    require(metadata["recorder_sha256"] == sha(recorder.__file__), "Recorder identity differs")
    require(seed == recorder.reset_seed(metadata["seed"], EPISODE), "Reset seed lineage differs")
    require(component == recorder.mixture_component(metadata["seed"], EPISODE),
            "Reset component lineage differs")
    options = recorder.reset_options(component)
    report["inputs"] = {"episode_index_zero_based": EPISODE, "dataset_entropy": metadata["seed"],
                        "reset_seed": seed, "reset_component": component, "reset_options": options,
                        "dataset_reset_law": metadata["reset_law"], "world": metadata["world"],
                        "state_shape": list(expected.shape), "state_dtype": expected.dtype.str,
                        "stored_generator_episode": metadata["generator"]["episodes"][EPISODE]}
    # Process-local fail-closed guards only; reset and its RNG are unmodified.
    environment.ThrusterChainEnv.step = forbidden_step
    environment.canonical.ThrusterChainEnv.step = forbidden_step
    environment.dynamics = forbidden_step
    environment.canonical.dynamics = forbidden_step
    env = worlds.environment("stall3")
    report["inputs"]["effective_start_law"] = environment.canonical.start_law(env.law, options.get("start"))
    observation, info = one_reset(env, seed, options, report["counts"])
    report["comparison"] = describe_values(expected, info["state"], np)
    report["observation"] = {"original_array_equal": bool(np.array_equal(observation, expected_obs)),
                             "expected": expected_obs.tolist(), "actual": observation.tolist(),
                             "dtype": observation.dtype.str, "shape": list(observation.shape),
                             "time": info["t"]}
    report["rng"] = {"bit_generator": type(env.np_random.bit_generator).__name__,
                     "state_after_reset": env.np_random.bit_generator.state}
    report["loaded_sources"] = {name: {"path": str(Path(module.__file__).resolve()),
                                      "sha256": sha(module.__file__)}
                                for name, module in list(sys.modules.items())
                                if getattr(module, "__file__", None)
                                and Path(module.__file__).resolve().is_relative_to(REPO)}
    require(not any(name.split(".")[0] in {"jax", "jaxlib", "torch", "tensorflow"}
                    for name in sys.modules), "Unexpected model/accelerator import")
    verify_seal()
    require(sha(dataset_path) == archive_sha, "Dataset changed during diagnostic")
    require(all(sha(PACKAGE / name) == digest for name, digest in report["prior_attempt"]["files"].items()),
            "Prior attempt evidence changed during diagnostic")
    report["integrity_after"] = "unchanged"
    report["status"] = "diagnostic_complete_equal" if report["comparison"]["original_array_equal"] else "diagnostic_complete_mismatch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-episode-16", action="store_true", required=True)
    parser.parse_args()
    require(sys.platform == "linux", "Execute only manually in the established WSL Python")
    started = time.monotonic()
    stamp = datetime.now(EST)
    output = HERE / ("receipt_" + stamp.strftime("%Y%m%dT%H%M%S_%f_EST_") + str(os.getpid()) + ".json")
    report = {"status": "started", "started_est": stamp.isoformat(), "training_authorized": False,
              "episode": EPISODE, "soft_limit_seconds": SOFT_LIMIT_SECONDS,
              "counts": {"reset_calls_attempted": 0, "reset_calls_completed": 0,
                         "simulator_steps": 0, "training_updates": 0, "calibrations": 0},
              "error": None}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    def timeout_handler(signum, frame):
        raise TimeoutError("Reset diagnostic exceeded its 110-second soft limit")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, SOFT_LIMIT_SECONDS)
    try:
        execute(report)
    except BaseException as error:
        report.update(status="diagnostic_failed_or_interrupted", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_est"] = datetime.now(EST).isoformat()
        if "prior_attempt" in report:
            total = report["prior_attempt"]["elapsed_seconds"] + report["elapsed_seconds"]
            report["cumulative_active_seconds_including_failed_training_attempt"] = total
            report["three_hour_allowance_less_known_active_seconds"] = max(0., 10800. - total)
        report["allowance_note"] = "Accounting only; no deadline reset or training retry authorized. Prior unknown update bounds are preserved."
        temp = output.with_suffix(".tmp")
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        os.replace(temp, output)
    print(json.dumps({"status": report["status"], "receipt": str(output), "counts": report["counts"],
                      "elapsed_seconds": report["elapsed_seconds"], "error": report["error"]}), flush=True)
    return int(report["error"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
