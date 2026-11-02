"""Bounded cross-device elementwise post-update check with fixed tolerances."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ATOL = 1e-4
RTOL = 0.005
ROWS = [0, 1, 99, 102, 199]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--cpu-reference", required=True, type=Path,
                        help="successful bounded harness CPU reference directory")
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-tree", type=Path, help="elementwise CPU output directory; required for GPU")
    args = parser.parse_args()
    if (args.mode == "gpu") != (args.cpu_tree is not None):
        raise ValueError("GPU requires --cpu-tree; CPU must not receive one")
    root = args.benchmark_root.resolve()
    repo = root.parent.parent
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(root))
    import numpy as np
    import jax
    import jaxlib
    from run_benchmark import source_manifest
    from thrusterchain.datasets import generate_dataset
    from thrusterchain.training_runner import train_transitions
    from thrusterchain.training_protocol import LearnerCheckpoint
    from host_integration.unifloral_adapter import UnifiedTD3BCAdapter, _runtime

    ref = json.loads((args.cpu_reference / "reference.json").read_text(encoding="utf8"))
    initial = args.cpu_reference / "initial_checkpoint.json"
    if (ref.get("mode") != "cpu" or ref.get("backend") != "cpu" or ref.get("debug_only")
            or ref.get("status") != "bounded_tests_passed" or ref.get("unified_tests_passed") != 4
            or digest(initial) != ref.get("initial_checkpoint_file_sha256")):
        raise ValueError("expected an intact successful CPU reference and matching checkpoint")
    mode = "gpu" if args.mode == "gpu" else "cpu"
    if jax.default_backend() != mode or not jax.devices() or any(d.platform != mode for d in jax.devices()):
        raise RuntimeError("unexpected backend or device; refusing fallback")
    precision = str(jax.config.jax_default_matmul_precision)
    runtime = _runtime()
    runtime["jaxlib"] = jaxlib.__version__
    if (precision != "highest" or runtime != ref["runtime"] or not runtime["upstream_constraints_met"]
            or source_manifest()["source_sha256"] != ref["source_sha256"]):
        raise ValueError("source, runtime, constraints or precision differs from CPU reference")
    data = generate_dataset(episodes=10, seed=1097, provenance="smoke")
    if data.digest != ref["dataset_sha256"]:
        raise ValueError("dataset digest differs from CPU reference")
    batch = train_transitions(data).select(ROWS)
    model = UnifiedTD3BCAdapter()
    model.restore(LearnerCheckpoint.from_bytes(initial.read_bytes()))
    metrics = model.update(batch)
    if model.optimizer_step_counts != {"actor": 1, "critic": 2}:
        raise AssertionError("unexpected unified optimizer counts")
    state = model._state
    groups = {"actor_params": state.actor.params, "critic_params": state.vec_q.params,
              "actor_target": state.actor_target.params, "critic_target": state.vec_q_target.params,
              "actor_optimizer": state.actor.opt_state, "critic_optimizer": state.vec_q.opt_state,
              "rng": model._rng}
    arrays, structure = {}, {}
    for group, tree in groups.items():
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        structure[group] = str(treedef)
        for index, leaf in enumerate(leaves):
            array = np.asarray(leaf)
            if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
                raise AssertionError(f"nonfinite or nonnumeric leaf {group}[{index}]")
            arrays[f"{group}__{index:04d}"] = array
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    identity = {"schema": "thrusterchain.unified_elementwise_device_check.v1",
                "mode": args.mode, "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
                "source_sha256": ref["source_sha256"], "dataset_sha256": data.digest,
                "initial_checkpoint_file_sha256": digest(initial), "runtime": runtime, "precision": precision,
                "script_sha256": digest(__file__), "plan_sha256": digest(Path(__file__).with_name("ELEMENTWISE_PLAN.md")),
                "rows": ROWS, "optimizer_counts": model.optimizer_step_counts,
                "tolerance": {"atol": ATOL, "rtol": RTOL}, "structure": structure,
                "leaf_count": len(arrays), "metrics": metrics,
                "scope": "one update, same checkpoint and batch; not training-trajectory equivalence"}
    if args.mode == "cpu":
        path = output / "post_update.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        identity.update(status="cpu_reference", arrays_sha256=digest(path))
        write_json(output / "reference.json", identity)
        print(json.dumps({"status": identity["status"], "leaves": len(arrays), "output": str(output)}))
        return
    cpu = json.loads((args.cpu_tree / "reference.json").read_text(encoding="utf8"))
    path = args.cpu_tree / "post_update.npz"
    for key in ("source_sha256", "dataset_sha256", "initial_checkpoint_file_sha256", "runtime", "precision", "rows", "optimizer_counts", "tolerance", "structure", "script_sha256", "plan_sha256"):
        if cpu.get(key) != identity[key]:
            raise ValueError(f"elementwise CPU reference differs in {key}")
    if cpu.get("status") != "cpu_reference" or cpu.get("backend") != "cpu" or cpu.get("arrays_sha256") != digest(path):
        raise ValueError("elementwise CPU artifact is missing or changed")
    comparisons = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise ValueError("elementwise leaf names differ")
        for name, observed in arrays.items():
            expected = archive[name]
            if observed.shape != expected.shape or observed.dtype != expected.dtype or not np.all(np.isfinite(expected)):
                raise AssertionError(f"shape, dtype or finiteness mismatch: {name}")
            if observed.dtype.kind in "biu":
                if not np.array_equal(observed, expected):
                    raise AssertionError(f"integer or PRNG mismatch: {name}")
                maximum = 0.0
            else:
                if not np.allclose(observed, expected, atol=ATOL, rtol=RTOL):
                    raise AssertionError(f"floating-point mismatch: {name}")
                maximum = float(np.max(np.abs(observed - expected))) if observed.size else 0.0
            group = name.rsplit("__", 1)[0]
            entry = comparisons.setdefault(group, {"leaves": 0, "max_absolute_difference": 0.0})
            entry["leaves"] += 1
            entry["max_absolute_difference"] = max(entry["max_absolute_difference"], maximum)
    identity.update(status="bounded_elementwise_passed", comparisons=comparisons,
                    cpu_tree_arrays_sha256=cpu["arrays_sha256"])
    write_json(output / "receipt.json", identity)
    print(json.dumps({"status": identity["status"], "leaves": len(arrays),
                      "comparisons": comparisons, "output": str(output)}))


if __name__ == "__main__":
    main()
