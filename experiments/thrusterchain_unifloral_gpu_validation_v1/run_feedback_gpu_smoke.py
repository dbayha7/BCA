"""Run the predeclared small feedback-data/shift integration on GPU."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--cpu-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    sys.path.insert(0, str(root.parent.parent))
    sys.path.insert(0, str(root))
    import jax
    import jaxlib
    from run_benchmark import source_manifest
    from thrusterchain import __main__ as cli
    from thrusterchain.quality_datasets import load_behavior_dataset
    from thrusterchain.shift_scenarios_v1 import SCENARIOS
    from thrusterchain.training_protocol import LearnerCheckpoint
    from host_integration.unifloral_adapter import _runtime

    reference = json.loads((args.cpu_reference / "reference.json").read_text(encoding="utf8"))
    runtime = _runtime()
    runtime["jaxlib"] = jaxlib.__version__
    source = source_manifest()
    if (reference.get("status") != "bounded_tests_passed" or reference.get("mode") != "cpu"
            or reference.get("runtime") != runtime or not runtime["upstream_constraints_met"]
            or reference.get("source_sha256") != source["source_sha256"]):
        raise ValueError("CPU reference source, runtime, or upper-bound checks do not match")
    if (jax.default_backend() != "gpu" or not jax.devices()
            or any(device.platform != "gpu" for device in jax.devices())
            or str(jax.config.jax_default_matmul_precision) != "highest"
            or os.environ.get("JAX_PLATFORMS") != "cuda"):
        raise RuntimeError("GPU backend, forced CUDA and highest precision are required")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    def invoke(*argv):
        previous = sys.argv
        sys.argv = ["thrusterchain", *map(str, argv)]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                cli.main()
            return json.loads(capture.getvalue())
        finally:
            sys.argv = previous

    dataset_path, trained = output / "feedback.npz", output / "unified"
    made = invoke("generate-behavior", dataset_path, "--behavior", "feedback_strong",
                  "--episodes", 10, "--seed", 62001)
    verified = invoke("validate", dataset_path)
    dataset = load_behavior_dataset(dataset_path)
    if made["sha256"] != dataset.digest or verified["sha256"] != dataset.digest:
        raise AssertionError("dataset failed independent save/load identity")
    train = invoke("train", dataset_path, trained, "--adapter", "unifloral-unified",
                   "--offline-updates", 2, "--batch-size", 8, "--seed", 62003)
    checkpoint = LearnerCheckpoint.load(trained / "checkpoint.json")
    if (dict(checkpoint.optimizer_step_counts) != {"actor": 2, "critic": 4}
            or train["environment_interactions"] != 0
            or checkpoint.metadata["adapter"]["learning_settings"].get("allow_outside_upstream_constraints", False)):
        raise AssertionError("unexpected optimizer, data or runtime-override counts")
    result = invoke("evaluate-shifts", trained, output / "shift_results.json", "--dataset", dataset_path,
                    "--scenarios", *SCENARIOS, "--final-seeds", 62004)
    if (result["episodes_total"] != 4 or result["environment_steps"] != 400
            or result["policy_sha256_before"] != result["policy_sha256_after"]
            or any(e["steps"] != 100 for row in result["scenarios"].values() for e in row["episodes"])):
        raise AssertionError("shift rollout or frozen policy mismatch")
    if source_manifest()["source_sha256"] != source["source_sha256"]:
        raise AssertionError("benchmark source changed during smoke")
    receipt = {"schema": "thrusterchain.feedback_gpu_smoke.v1", "status": "pass",
        "scope": "bounded wiring only; not performance or an unexposed holdout",
        "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
        "precision": str(jax.config.jax_default_matmul_precision), "runtime": runtime,
        "source_sha256": source["source_sha256"], "script_sha256": digest(__file__),
        "plan_sha256": digest(Path(__file__).with_name("END_TO_END_PLAN.md")),
        "dataset_sha256": dataset.digest, "checkpoint_sha256": checkpoint.digest,
        "optimizer_counts": dict(checkpoint.optimizer_step_counts),
        "training_environment_interactions": train["environment_interactions"], "evaluation": result}
    with (output / "receipt.json").open("x", encoding="utf8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "pass", "backend": receipt["backend"],
                      "optimizer_counts": receipt["optimizer_counts"],
                      "evaluation_episodes": result["episodes_total"], "output": str(output)}))


if __name__ == "__main__":
    main()
