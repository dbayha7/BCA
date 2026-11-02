"""Execute the predeclared small v6 integration check; never overwrite output."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--optional-site-packages", type=Path,
                        help="append a site-packages directory after the active runtime")
    args = parser.parse_args()
    if args.optional_site_packages is not None:
        if not args.optional_site_packages.is_dir():
            raise ValueError("optional site-packages directory does not exist")
        sys.path.append(str(args.optional_site_packages))
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from run_benchmark import source_manifest
    from thrusterchain import __main__ as cli
    from thrusterchain.quality_datasets import load_behavior_dataset
    from thrusterchain.shift_scenarios_v1 import SCENARIOS
    from thrusterchain.training_protocol import LearnerCheckpoint
    from host_integration.unifloral_adapter import _runtime

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source = source_manifest()
    (output / "source_before.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf8")
    dataset_path, trained = output / "behavior.npz", output / "unified"

    def invoke(*argv):
        old = sys.argv
        sys.argv = ["thrusterchain", *map(str, argv)]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                cli.main()
            return json.loads(stream.getvalue())
        finally:
            sys.argv = old

    made = invoke("generate-behavior", dataset_path, "--behavior", "feedback_strong",
                  "--episodes", 10, "--seed", 62001)
    validated = invoke("validate", dataset_path)
    data = load_behavior_dataset(dataset_path)
    if made["sha256"] != data.digest or validated["sha256"] != data.digest:
        raise AssertionError("behavior dataset digest mismatch")
    settings = json.dumps({"allow_outside_upstream_constraints": True})
    training = invoke("train", dataset_path, trained, "--adapter", "unifloral-unified",
                      "--offline-updates", 2, "--batch-size", 8, "--seed", 62003,
                      "--learning-settings", settings)
    checkpoint = LearnerCheckpoint.load(trained / "checkpoint.json")
    if dict(checkpoint.optimizer_step_counts) != {"actor": 2, "critic": 4}:
        raise AssertionError("unified actor/critic update count mismatch")
    if training["environment_interactions"] != 0:
        raise AssertionError("offline training used environment interactions")
    result = invoke("evaluate-shifts", trained, output / "shift_results.json",
                    "--dataset", dataset_path, "--scenarios", *SCENARIOS,
                    "--final-seeds", 62004)
    if (result["episodes_total"] != 4 or result["environment_steps"] != 400
            or result["policy_sha256_before"] != result["policy_sha256_after"]
            or any(episode["steps"] != 100 for entry in result["scenarios"].values() for episode in entry["episodes"])):
        raise AssertionError("shift evaluation or frozen-policy invariant failed")
    if source_manifest()["source_sha256"] != source["source_sha256"]:
        raise AssertionError("source changed during the bounded smoke")
    receipt = {"schema": "thrusterchain.v6_bounded_smoke.v1", "status": "pass", "profile": "smoke_only",
        "source_sha256": source["source_sha256"], "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_sha256": data.digest, "dataset_behavior": "feedback_strong", "dataset_episodes": 10,
        "checkpoint_sha256": checkpoint.digest, "optimizer_counts": dict(checkpoint.optimizer_step_counts),
        "training_environment_interactions": 0, "evaluation": result, "runtime": _runtime(),
        "limitations": ["One unified composition and one fixed policy only; no performance comparison.",
                        "The final label and seed separation do not attest an unexposed holdout."]}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"status": "pass", "output": str(output), "source_sha256": source["source_sha256"],
                      "dataset_sha256": data.digest, "optimizer_counts": receipt["optimizer_counts"],
                      "scenario_episodes": result["episodes_total"]}))


if __name__ == "__main__":
    main()
