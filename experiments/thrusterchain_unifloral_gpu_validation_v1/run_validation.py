"""Bounded v6 unified CPU reference / matched-checkpoint GPU validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

EXPECTED_SOURCE = "e2f4e56c0cafee494145c292781ac4b9fd4761fa1463c9ebead68e6418d9a7e5"
EXPECTED_UPSTREAM = "14a9bcdbc6d96fad85544a9d98957e98a7e46db13b89d74d54e3c5ce246517c8"
COMPAT_DELTAS = {
    "pyproject.toml": "d4ec261ac27d6b3ddd461523b9647ddba39de413e8a3ee426a5f6ab068ef0596",
    "tests/test_wheel_notices.py": "992a37f1afacfeb24bd6f7aa54c9bb85842125d20086a8b33df3deab34210079",
    "requirements-tested.txt": "a7459084eac39a8eda4e55e3250e093aa71ab0cc817bc502b96744f7226897f4",
}
BASE_REQUIREMENTS = "980634da6598dd2eb924b738c7d583e8669a08047bb2221dff2af9232f0313bc"
FD_H = (0.001, 0.002)
FD_ATOL, FD_RTOL = 0.001, 0.03
COMPARE_ATOL, COMPARE_RTOL = 0.0001, 0.005
GRAD_ATOL, GRAD_RTOL = 0.001, 0.03


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path,
                        help="untouched v6 snapshot used for byte-for-byte comparison")
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-reference", type=Path)
    parser.add_argument("--allow-incompatible-for-debug", action="store_true")
    args = parser.parse_args()
    if args.mode == "gpu" and args.cpu_reference is None:
        raise ValueError("GPU mode requires a CPU reference directory")
    if args.mode == "cpu" and args.cpu_reference is not None:
        raise ValueError("CPU mode does not consume a CPU reference")
    root = args.benchmark_root.resolve()
    baseline = args.baseline_root.resolve()
    repo = root.parent.parent
    if not (root / "thrusterchain").is_dir() or not (repo / "baselines/unifloral/algorithms/unifloral.py").is_file():
        raise ValueError("benchmark must retain the BCA/experiments/v6 layout")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(root))
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jaxlib
    from run_benchmark import source_manifest
    from thrusterchain.datasets import generate_dataset
    from thrusterchain.training_runner import train_transitions
    from thrusterchain.training_protocol import LearnerCheckpoint
    from host_integration.unifloral_adapter import UnifiedTD3BCAdapter, _runtime

    source = source_manifest()
    original = json.loads((baseline / "CURRENT_SOURCE_MANIFEST.json").read_text(encoding="utf8"))
    original_files = original.get("files", {})
    encoded = json.dumps(original_files, sort_keys=True, separators=(",", ":")).encode()
    if (original.get("source_sha256") != EXPECTED_SOURCE or hashlib.sha256(encoded).hexdigest() != EXPECTED_SOURCE
            or any(not (baseline / name).is_file() or sha(baseline / name) != digest
                   for name, digest in original_files.items()) or sha(baseline / "requirements-tested.txt") != BASE_REQUIREMENTS):
        raise ValueError("baseline does not match the validated v6 source snapshot")
    if set(source["files"]) != set(original_files):
        raise ValueError("compatibility copy changes the source-file inventory")
    for name, digest in original_files.items():
        expected = COMPAT_DELTAS.get(name, digest)
        if source["files"][name] != expected:
            raise ValueError(f"unexpected compatibility-copy change: {name}")
    if sha(root / "requirements-tested.txt") != COMPAT_DELTAS["requirements-tested.txt"]:
        raise ValueError("unexpected compatibility requirements file")
    upstream = repo / "baselines/unifloral/algorithms/unifloral.py"
    if sha(upstream) != EXPECTED_UPSTREAM:
        raise ValueError("pinned unified UniFloral source mismatch")
    expected_backend = "gpu" if args.mode == "gpu" else "cpu"
    if jax.default_backend() != expected_backend:
        raise RuntimeError(f"expected {expected_backend} backend, got {jax.default_backend()}")
    devices = jax.devices()
    if not devices or any(device.platform != expected_backend for device in devices):
        raise RuntimeError("unexpected JAX device platform")
    precision = str(jax.config.jax_default_matmul_precision)
    if precision != "highest":
        raise RuntimeError("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    runtime = _runtime()
    runtime["jaxlib"] = jaxlib.__version__
    if not runtime["upstream_constraints_met"] and not args.allow_incompatible_for_debug:
        raise RuntimeError("JAX or NumPy outside upstream bounds; debug requires explicit opt-in")
    settings = {"allow_outside_upstream_constraints": True} if args.allow_incompatible_for_debug else {}
    dataset = generate_dataset(episodes=10, seed=1097, provenance="smoke")
    batch_all = train_transitions(dataset)
    batch = batch_all.select([0, 1, 99, 102, 199])
    if not np.array_equal(batch.terminated, [False, False, True, False, True]):
        raise AssertionError("fixture no longer has its expected terminal rows")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    reference = None
    if args.mode == "cpu":
        model = UnifiedTD3BCAdapter()
        model.initialize(batch_all, seed=13, learning_settings=settings)
        initial_bytes = model.checkpoint().to_bytes()
        with (output / "initial_checkpoint.json").open("xb") as stream:
            stream.write(initial_bytes)
    else:
        refdir = args.cpu_reference
        reference = json.loads((refdir / "reference.json").read_text(encoding="utf8"))
        if (reference["source_sha256"] != source["source_sha256"] or reference["dataset_sha256"] != dataset.digest
                or reference["runtime"] != runtime or reference["precision"] != precision
                or reference["debug_only"] != args.allow_incompatible_for_debug):
            raise ValueError("CPU reference source, dataset, runtime, precision or debug mode mismatch")
        initial_bytes = (refdir / "initial_checkpoint.json").read_bytes()
        if hashlib.sha256(initial_bytes).hexdigest() != reference["initial_checkpoint_file_sha256"]:
            raise ValueError("CPU checkpoint bytes do not match their reference")
        model = UnifiedTD3BCAdapter()
        model.restore(LearnerCheckpoint.from_bytes(initial_bytes))

    # Exercise the unchanged v6 numerical, timing, resume and frozen-eval tests.
    suite = unittest.defaultTestLoader.loadTestsFromName("host_integration.test_unifloral_adapter")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 4 or result.skipped:
        raise AssertionError("the four v6 unified integration tests did not pass")

    obs = jnp.asarray(batch.observations[0], jnp.float32)
    action = model._actor.apply(model._state.actor.params, obs).mean()
    direction = jnp.asarray([1., 0., 0., 0.], jnp.float32)
    def first_critic(a):
        return model._critic.apply(model._state.vec_q.params, obs[None], a[None])[0, 0]
    value = first_critic(action)
    grad = jax.grad(first_critic)(action)
    directional = jnp.vdot(grad, direction)
    fd = [(first_critic(action + h * direction) - first_critic(action - h * direction)) / (2 * h) for h in FD_H]
    grad_values = np.asarray(grad, dtype=np.float64)
    value, directional, fd = float(value), float(directional), [float(x) for x in fd]
    if not np.all(np.isfinite([value, directional, *fd, *grad_values])):
        raise AssertionError("nonfinite action-gradient fixture")
    for approximation in fd:
        if not np.isclose(approximation, directional, atol=FD_ATOL, rtol=FD_RTOL):
            raise AssertionError(f"action-gradient finite difference mismatch: autodiff={directional}, fd={approximation}")

    before_actor = jax.tree_util.tree_leaves(model._state.actor.params)
    before_critic = jax.tree_util.tree_leaves(model._state.vec_q.params)
    metrics = model.update(batch)
    if model.optimizer_step_counts != {"actor": 1, "critic": 2}:
        raise AssertionError("native actor/critic cadence changed")
    def norm_delta(before, after):
        terms = [np.sum(np.square(np.asarray(y, np.float64) - np.asarray(x, np.float64)))
                 for x, y in zip(before, jax.tree_util.tree_leaves(after))]
        return float(np.sqrt(sum(terms)))
    measured = {"critic_first_q": value, "action_gradient": grad_values.tolist(),
                "directional_derivative": directional, "finite_difference": fd,
                "actor_update_norm": norm_delta(before_actor, model._state.actor.params),
                "critic_update_norm": norm_delta(before_critic, model._state.vec_q.params),
                "metrics": metrics}
    if not all(np.isfinite(float(value)) for value in metrics.values()):
        raise AssertionError("nonfinite native update metrics")
    if reference is not None:
        expected = reference["measured"]
        for key in ("critic_first_q", "actor_update_norm", "critic_update_norm"):
            if not np.isclose(measured[key], expected[key], atol=COMPARE_ATOL, rtol=COMPARE_RTOL):
                raise AssertionError(f"CPU/GPU mismatch in {key}")
        if set(measured["metrics"]) != set(expected["metrics"]):
            raise AssertionError("CPU/GPU metric names differ")
        for key in measured["metrics"]:
            if not np.isclose(measured["metrics"][key], expected["metrics"][key], atol=COMPARE_ATOL, rtol=COMPARE_RTOL):
                raise AssertionError(f"CPU/GPU mismatch in metric {key}")
        for key in ("action_gradient", "directional_derivative", "finite_difference"):
            if not np.allclose(measured[key], expected[key], atol=GRAD_ATOL, rtol=GRAD_RTOL):
                raise AssertionError(f"CPU/GPU gradient mismatch in {key}")

    receipt = {"schema": "thrusterchain.unified_bounded_device_validation.v1", "mode": args.mode,
        "status": "debug_only" if args.allow_incompatible_for_debug else "bounded_tests_passed",
        "debug_only": args.allow_incompatible_for_debug, "source_sha256": source["source_sha256"],
        "baseline_source_sha256": EXPECTED_SOURCE, "allowed_compatibility_deltas": COMPAT_DELTAS,
        "pinned_source_sha256": sha(upstream), "harness_sha256": sha(__file__),
        "dataset_sha256": dataset.digest, "runtime": runtime, "backend": jax.default_backend(),
        "devices": [str(device) for device in devices], "precision": precision,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"), "initial_checkpoint_file_sha256": hashlib.sha256(initial_bytes).hexdigest(),
        "fixture_rows": [0, 1, 99, 102, 199], "unified_tests_passed": result.testsRun,
        "additional_actor_updates": 1, "additional_critic_updates": 2,
        "finite_difference": {"steps": list(FD_H), "atol": FD_ATOL, "rtol": FD_RTOL},
        "comparison": {"atol": COMPARE_ATOL, "rtol": COMPARE_RTOL,
                       "gradient_atol": GRAD_ATOL, "gradient_rtol": GRAD_RTOL},
        "measured": measured,
        "scope": "bounded computational checks only; no full training or scientific performance claim"}
    json_write(output / ("reference.json" if args.mode == "cpu" else "receipt.json"), receipt)
    print(json.dumps({"status": receipt["status"], "mode": args.mode, "backend": receipt["backend"],
                      "output": str(output), "unified_tests_passed": result.testsRun}))


if __name__ == "__main__":
    main()
