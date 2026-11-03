"""Zero-update Titan compatibility proof, called only by its approved worker.

The caller authenticates the full transfer/recovery closure and creates frozen
services through smoke_launch._services(protocol). This helper neither launches
nor configures a runtime. No model step, fit, refresh, dynamics, reset or rollout
is called. All writes use the caller's append-only evidence callback.
"""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V2 = REPO / "experiments/thrusterchain_bca/two_scenario_v2"
CORE = REPO / "experiments/thrusterchain_bca/titan_20261026/payload/transfer_bundle/core_v4"
EXPECTED_EXECUTABLE = "/home/dbayha/bca-gpu/bin/python"
PINS = {
    "protocol.json": "7baf2d5a15e8d204a19ce47f9b17c2421256fe50a042cd61a514bfbd9f0d05a8",
    "DATA_RECEIPT.json": "27d4bac1b028b063adfb8e2a1a263af0ea45be781f42a0397325a66beb8fbd99",
    "smoke_attempt_1/ARTIFACTS.json": "92c54c13e3a9986a352f9de427f849d09e74d1e5dee09e7458e8a14167220c9a",
    "run_amendment_1/honeypot_s0_warmup/checkpoint.json": "8425bf05bb9ba679d0c632115ea9eccebc129a8800ccf183c1a9907445a98291",
    "run_amendment_1/honeypot_s0_warmup/report.json": "67997f5b6fd8e22dad1d94d5075350e6ee5de1e23ba6ea05726f4bebba47e02b",
    "run_amendment_1/honeypot_s0_warmup/full_state.msgpack": "5422f85ac77d2a51341bb1154d7ba057a623854fd5ec8232b762bf0443bc218f",
}
ZERO_WORK = {"critic": 0, "actor": 0, "auxiliary": 0, "calibrations": 0, "simulator_steps": 0, "reset_calls": 0}


def require(value, message):
    if not value: raise ValueError(message)


def _plain(value):
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, np.generic): return _plain(value.item())
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"
    return value


def _bound(path, expected, *, json_value=True):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Immutable Titan input differs: " + str(path))
    return json.loads(raw) if json_value else None


def _source_inputs(protocol, warmup_dir, deadline):
    require(Path(warmup_dir).resolve() == (HERE / "run_amendment_1/honeypot_s0_warmup").resolve(), "Wrong transferred warmup directory")
    data = {}
    for relative, digest in PINS.items():
        deadline()
        data[relative] = _bound(HERE / relative, digest, json_value=relative.endswith(".json"))
    require(protocol == data["protocol.json"], "Scientific protocol differs from completed warmup")
    closure = data["smoke_attempt_1/ARTIFACTS.json"]
    cases = []
    for seed in range(3):
        for condition, t in (("honeypot", 29), ("bottleneck", 25)):
            prefix = "physical/smoke_%s_seed%d" % (condition, seed)
            coverage_name = prefix + "_coverage.json"
            deadline()
            coverage = _bound(HERE / "smoke_attempt_1" / coverage_name, closure[coverage_name])
            states, actions, times = (np.asarray(coverage[k]) for k in ("states", "actions", "times"))
            require(states.shape == (16, 13) and actions.shape == (16, 3, 4) and times.shape == (16,) and
                    np.all(times == t) and coverage["passed"] and coverage["common_edges"] == 48,
                    "Incomplete fixed Zeus common-action bank")
            for index in range(16):
                cases.append({"id": "%s/%d/common/%d" % (condition, seed, index), "kind": "common",
                    "state": states[index].astype(np.float64), "time": t,
                    "expected_actions": actions[index].astype(np.float32), "source_sha256": closure[coverage_name]})
            anchor_name = prefix + "_anchor_00.json"
            deadline()
            anchor = _bound(HERE / "smoke_attempt_1" / anchor_name, closure[anchor_name])
            row = anchor["trace"][0]
            require(row["time"] == 0 and anchor["last_prefix_step_exact_replay"], "Invalid saved prefix sample")
            cases.append({"id": "%s/%d/prefix/0" % (condition, seed), "kind": "prefix",
                "state": np.asarray(row["state"], np.float64), "time": 0,
                "expected_actions": np.asarray(row["action"], np.float32)[None, :], "source_sha256": closure[anchor_name]})
    require(len(cases) == 102 and sum(len(c["expected_actions"]) for c in cases) == 294, "Controller bank changed")
    original = _bound(V2 / "setup_attempt_1/RESULT.json",
                      "1e9ba131c35b3e4cafbb8815e3f6338ba6001cc35e1f3e05d2be7518f89941f1")
    return {"data_receipt": data["DATA_RECEIPT.json"],
            "checkpoint": data["run_amendment_1/honeypot_s0_warmup/checkpoint.json"],
            "source_result": original, "controller_cases": cases, "pins": dict(PINS)}


def _runtime_requirements(receipt):
    require(receipt["executable"] == EXPECTED_EXECUTABLE, "Use the reviewed Titan interpreter path exactly")
    require(str(receipt["python"]).split()[0].split(".")[:2] == ["3", "12"], "Expected Python3.12; patch is recorded, not pinned")
    require(receipt["backend"] == "gpu" and receipt["devices"] and receipt["device_kinds"], "No actual GPU device receipt")
    required = {"JAX_PLATFORMS": "cuda", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_ENABLE_X64": "false",
                "XLA_FLAGS": "--xla_gpu_triton_gemm_any=True", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                "BCA_TRANSFER_BUNDLE_ROOT": str(CORE.parent)}
    require(all(os.environ.get(k) == v for k, v in required.items()), "Frozen service environment differs")
    require(not os.environ.get("OPENBLAS_CORETYPE"), "OPENBLAS_CORETYPE override is not part of this compatibility profile")
    return {**required, "OPENBLAS_CORETYPE": os.environ.get("OPENBLAS_CORETYPE"), "python_patch_may_differ": True}


def _normalization_comparison(expected, actual):
    differences = {}
    for key in ("mean", "std"):
        a, b = np.asarray(expected[key], np.float64), np.asarray(actual[key], np.float64)
        differences[key] = {"expected_shape": list(a.shape), "actual_shape": list(b.shape),
            "max_abs_difference": float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else None,
            "array_exact": bool(a.shape == b.shape and np.array_equal(a, b))}
    return differences


def _make_controller(protocol, binding, source_result):
    """Only matrices/controller algebra; no call or reference to a plant step."""
    sys.path.insert(0, str(CORE))
    canonical = importlib.import_module("thrusterchain.environment")
    require(Path(canonical.__file__).resolve() == CORE / "thrusterchain/environment.py", "Wrong canonical source")
    physics = canonical.Physics(stall=True, stall_slope=3.)
    require(physics.as_dict() == source_result["physics"] and
            canonical.provenance() == source_result["runtime"]["canonical"], "Controller physics/source differs")
    A, B = canonical.matrices(physics)
    return binding.D.T.tracking_controller(A, B, protocol["geometry_recipe"]), {
        "physics": physics.as_dict(), "canonical": canonical.provenance(), "matrices_calls": 1,
        "tracking_controller_initializations": 1, "dynamics_calls": 0}


def check_compatibility(protocol, binding, training, services, review, worker, deadline, persist, *, warmup_dir):
    """Run inside the launcher's120-second reservation; persist(event) is durable.

    Services must already have passed frozen package/GPU/JIT/source checks via
    S._services. One fresh Honeypotseed0 template is initialized only to restore
    the authenticated completed10k state; no fresh parameters train or replace it.
    """
    counts = {"condition_bindings_attempted": 0, "condition_bindings_completed": 0,
              "template_initializations_attempted": 0, "template_initializations_completed": 0,
              "checkpoint_restores_attempted": 0, "checkpoint_restores_completed": 0,
              "actor_batches_attempted": 0, "actor_batches_completed": 0,
              "q1_batches_attempted": 0, "q1_batches_completed": 0,
              "width_batches_attempted": 0, "width_batches_completed": 0,
              "inference_rows_per_batch": 8, "controller_initializations_attempted": 0,
              "controller_initializations_completed": 0, "controller_calls_attempted": 0,
              "controller_calls_completed": 0, "deployed_action_comparisons": 0}
    result = {"status": "failed_zero_update_titan_compatibility", "passed": False,
              "new_work": dict(ZERO_WORK), "counts": counts, "checks": {}, "error": None}
    def emit(stage, value):
        persist(_plain({"stage": stage, **value, "new_work": ZERO_WORK, "counts": dict(counts)}))
    try:
        deadline()
        source = _source_inputs(protocol, warmup_dir, deadline)
        emit("authenticated_transferred_inputs", {"source_pins": source["pins"], "controller_cases": 102})
        deadline()
        runtime = services.runtime.runtime_receipt(services.native)
        result["runtime"], result["environment"] = runtime, _runtime_requirements(runtime)
        emit("runtime", {"runtime": runtime, "environment": result["environment"]})
        require(services.expected_data_receipt == source["data_receipt"], "Services loaded a different DATA_RECEIPT")
        require(services.binding is binding, "Different binding object supplied by services")
        binding_checks = {}
        for condition in ("honeypot", "bottleneck"):
            deadline(); counts["condition_bindings_attempted"] += 1
            emit("before_data_binding", {"condition": condition})
            prepared = binding.prepare(services.native, condition, protocol=protocol)
            counts["condition_bindings_completed"] += 1
            expected = source["data_receipt"]["conditions"][condition]["metadata"]
            exact = prepared.metadata == expected
            binding_checks[condition] = {"exact": exact, "expected_digest": binding.I.digest(expected),
                                         "actual_digest": binding.I.digest(prepared.metadata)}
            emit("data_binding_" + condition, binding_checks[condition])
            require(exact, "Titan condition data/normalization/calibration membership differs: " + condition)
        result["checks"]["both_condition_metadata_exact"] = True
        deadline(); counts["template_initializations_attempted"] += 1
        emit("before_template_initialization", {"condition": "honeypot", "seed_index": 0})
        prepared, adapter, template = training._prepare(protocol, "honeypot", 0, services,
            {**review, "scope": "zero_update_titan_compatibility"})
        counts["template_initializations_completed"] += 1
        initial_counts = services.runtime.host(services.runtime.validate(adapter, template)["counts"])
        require(all(initial_counts[k] == 0 for k in ("critic", "actor", "auxiliary", "refresh_attempts", "refresh_accepted")),
                "Template initialization unexpectedly performed training/calibration")
        expected_norm = source["checkpoint"]["normalization"]
        actual_norm = services.runtime.host(adapter.context.normalization)
        norm_exact = binding.I.digest(actual_norm) == source["checkpoint"]["normalization_sha256"]
        norm_evidence = {"exact": norm_exact, "expected": expected_norm, "observed": actual_norm,
                        "differences": _normalization_comparison(expected_norm, actual_norm)}
        emit("template_and_normalization", {"initial_counts": initial_counts, "normalization": norm_evidence})
        require(norm_exact, "Titan normalization differs exactly; no tolerance or numerical workaround is permitted")
        deadline(); counts["checkpoint_restores_attempted"] += 1
        emit("before_checkpoint_restore", {"checkpoint_sha256": source["checkpoint"]["checkpoint_sha256"]})
        state, lineage = training._read_checkpoint(warmup_dir, services, prepared, adapter, template, protocol,
                                                    "honeypot", 0, "warmup")
        counts["checkpoint_restores_completed"] += 1
        restored_counts = services.runtime.host(services.runtime.validate(adapter, state)["counts"])
        before = services.runtime.tree_digest(state)
        result["restore"] = {"lineage": lineage, "normalization_exact": True, "state_digest": before,
                             "saved_cumulative_counts": restored_counts, "saved_updates_are_not_new_work": True}
        emit("restored_checkpoint", result["restore"])
        require(restored_counts == source["checkpoint"]["counts"] and before == source["checkpoint"]["state_digest"],
                "Restored full-state counters/tree differ")
        result["checks"]["exact_full_state_and_normalization_restore"] = True
        policy = worker.callbacks(services.runtime, adapter, state, {"compatibility_only": True,
            "checkpoint_sha256": source["checkpoint"]["checkpoint_sha256"]})
        obs = np.asarray(prepared.train.obs[:8], np.float32).copy()
        act = np.asarray(prepared.train.action[:8], np.float32).copy()
        require(obs.shape == (8, 14) and act.shape == (8, 4), "Fixed8-row retained inference fixture unavailable")
        inference = {"source_row_ids": np.asarray(prepared.source_row_ids[:8]).tolist(), "observations": obs, "logged_actions": act}
        for name, args, shape in (("actor", (obs,), (8, 4)), ("q1", (obs, act), (8,)), ("width", (obs, act), (8,))):
            deadline(); counts[name + "_batches_attempted"] += 1
            emit("before_inference_" + name, {"rows": 8})
            value = np.asarray(policy[name](*args))
            counts[name + "_batches_completed"] += 1
            valid = value.shape == shape and np.isfinite(value).all()
            if name == "actor": valid = valid and value.dtype == np.float32 and np.all(np.abs(value) <= 1)
            if name == "width": valid = valid and np.all(value >= 0)
            inference[name] = value
            emit("inference_" + name, {"values": value, "valid": bool(valid)})
            require(valid, "Invalid finite retained-data inference: " + name)
        policy["assert_unchanged"]()
        require(services.runtime.tree_digest(state) == before and
                services.runtime.host(services.runtime.validate(adapter, state)["counts"]) == restored_counts,
                "Compatibility inference changed saved state")
        result["inference"] = _plain(inference)
        result["checks"]["finite8row_actor_q1_width_and_state_unchanged"] = True
        deadline(); counts["controller_initializations_attempted"] += 1
        emit("before_controller_initialization", {"matrices_only": True, "dynamics_calls": 0})
        controller, controller_source = _make_controller(protocol, binding, source["source_result"])
        counts["controller_initializations_completed"] += 1
        emit("controller_source", controller_source)
        maximum, mismatches = 0., []
        for case in source["controller_cases"]:
            deadline(); counts["controller_calls_attempted"] += 1
            action = np.asarray(controller(case["state"].copy(), int(case["time"])))
            counts["controller_calls_completed"] += 1
            require(action.shape == (4,) and action.dtype == np.float32 and np.isfinite(action).all() and np.all(np.abs(action) <= 1),
                    "Controller did not return a finite deployed float32 action")
            actual = np.stack((action, -action, np.zeros_like(action))) if case["kind"] == "common" else action[None, :]
            expected = case["expected_actions"]
            require(actual.shape == expected.shape, "Cached controller comparison shape differs")
            delta = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            exact = bool(np.array_equal(actual, expected))
            counts["deployed_action_comparisons"] += len(actual)
            maximum = max(maximum, delta)
            if not exact: mismatches.append(case["id"])
            emit("controller_case", {"id": case["id"], "kind": case["kind"], "time": case["time"],
                 "state": case["state"], "source_sha256": case["source_sha256"],
                 "expected_actions": expected, "observed_actions": actual, "exact": exact, "max_abs_difference": delta})
        result["controller"] = {"source": controller_source, "calls": counts["controller_calls_completed"],
                                "action_comparisons": counts["deployed_action_comparisons"],
                                "mismatched_case_ids": mismatches, "max_abs_difference": maximum,
                                "criterion": "exact deployed float32 equality; no tolerance or BLAS override"}
        emit("controller_summary", result["controller"])
        require(counts["controller_calls_completed"] == 102 and counts["deployed_action_comparisons"] == 294 and not mismatches,
                "Titan controller differs from the frozen Zeus action bank; no workaround applied")
        result["checks"]["exact_saved_controller_actions"] = True
        deadline()
        result.update(status="passed_zero_update_titan_compatibility", passed=True)
        emit("completed", result)
        return _plain(result)
    except BaseException as exc:
        result.update(status="failed_zero_update_titan_compatibility", passed=False,
                      error=type(exc).__name__ + ": " + str(exc))
        try: emit("failed", result)
        except BaseException as write_error:
            print(json.dumps(_plain({**result, "persistence_error": str(write_error)}), allow_nan=False), flush=True)
        raise
