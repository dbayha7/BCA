"""User-run synthetic checkpoint correctness gate; no scientific imports here.

This checks a boundary interruption after critic 1, including the pending batch
and independent auxiliary state. It does not inject an OS signal or claim that
the synthetic fixture is an eligible trained checkpoint. Resume equality is
exact. Native two-critic versus split-microstep float parity uses the previously
fixed rtol=2e-5, atol=2e-6, with integer counters and RNG always exact.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import traceback


RTOL = 2e-5
ATOL = 2e-6
TRAIN_SEED = 2026102362
HELDOUT_SEED = 2026102363
TINY_CONFIG = {
    "mode": "tiny", "gamma": 0.98, "batch_size": 8,
    "seed": 2026102302, "aux_seed": 2026102361,
    "refresh_at": [2, 4, 6], "critic_budget": 6, "kappa_penalty": 0.0,
}
CAP = {"critic": 17, "actor": 9, "auxiliary": 11, "refresh": 6}


def run(native, output, bindings):
    """Return a strict-JSON report; called only by the explicit GPU worker."""
    import numpy as np
    from flax import serialization
    import checkpoint_io
    import training

    jax, jnp = native.jax, native.jnp
    output = Path(output).resolve()
    if not output.is_dir():
        raise ValueError("The worker must create a new output directory first")
    results, state_log, metrics_log = [], {}, {}
    cache = {}
    ledger = {
        "caps": dict(CAP),
        "dispatches": {"main_microsteps": 0, "replay_microsteps": 0, "native_outer_steps": 0},
        "reserved_dispatch_upper_bounds": dict.fromkeys(CAP, 0),
        "verified_returned_operations": dict.fromkeys(CAP, 0),
        "failed_dispatches": [],
        "counts_meaning": "Includes replay and independent native baseline; initialization has no optimizer updates.",
    }

    def record(name, fn, requires=()):
        missing = [key for key in requires if key not in cache]
        if missing:
            results.append({"name": name, "status": "skip", "reason": "Prerequisite failed: " + ", ".join(missing)})
            return
        try:
            detail = fn()
            results.append({"name": name, "status": "pass", "details": detail})
        except Exception as exc:
            results.append({"name": name, "status": "fail", "error_type": type(exc).__name__,
                            "error": str(exc), "traceback": traceback.format_exc(limit=8)})

    def assert_true(condition, message):
        if not bool(condition):
            raise AssertionError(message)

    def leaves(value):
        return jax.tree_util.tree_flatten(serialization.to_state_dict(value))

    def compare(left, right, *, exact, label):
        aa, ta = leaves(left)
        bb, tb = leaves(right)
        assert_true(ta == tb and len(aa) == len(bb), label + ": different serialized tree structure")
        maximum_error = 0.0
        for index, (a, b) in enumerate(zip(aa, bb)):
            a, b = np.asarray(a), np.asarray(b)
            assert_true(a.shape == b.shape and a.dtype == b.dtype,
                        f"{label}: leaf {index} shape/dtype mismatch")
            if a.dtype.kind in "fc":
                assert_true(not np.any(np.isnan(a)) and not np.any(np.isnan(b)), label + ": NaN leaf")
                if exact:
                    assert_true(np.array_equal(a, b), f"{label}: leaf {index} differs exactly")
                else:
                    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=False,
                                               err_msg=f"{label}: leaf {index}")
                finite = np.isfinite(a) & np.isfinite(b)
                if np.any(finite):
                    maximum_error = max(maximum_error, float(np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)))))
            else:
                assert_true(np.array_equal(a, b), f"{label}: nonfloating leaf {index} differs")
        return {"leaves": len(aa), "maximum_absolute_float_difference": maximum_error,
                "comparison": "exact array equality" if exact else "float tolerance; nonfloating exact",
                "rtol": 0.0 if exact else RTOL, "atol": 0.0 if exact else ATOL}

    def fingerprint(value):
        aa, tree = leaves(value)
        digest = hashlib.sha256(str(tree).encode("utf-8"))
        for a in aa:
            a = np.asarray(a)
            digest.update(json.dumps({"shape": list(a.shape), "dtype": str(a.dtype)}, sort_keys=True).encode())
            digest.update(np.ascontiguousarray(a).tobytes())
        return digest.hexdigest()

    def checkpoint_hashes(path):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(path.iterdir()) if p.is_file()}

    def expected_counts(critic):
        refreshes = sum(value <= critic for value in TINY_CONFIG["refresh_at"])
        return {"critic": critic, "actor": critic // 2, "actor_target": critic // 2,
                "critic_target": critic // 2, "outer_steps": critic // 2, "phase": critic % 2,
                "auxiliary": critic, "aux_fit_attempts": critic,
                "refresh_attempts": refreshes, "refresh_accepted": refreshes,
                "last_refresh_critic": max((x for x in (0, 2, 4, 6) if x <= critic)),
                "next_refresh_index": refreshes, "reference_ready": critic >= 2, "critic_budget": 6}

    def reserve(work, dispatch_name):
        for key in CAP:
            assert_true(ledger["reserved_dispatch_upper_bounds"][key] + work[key] <= CAP[key],
                        "Declared operation cap would be exceeded: " + key)
        for key in CAP:
            ledger["reserved_dispatch_upper_bounds"][key] += work[key]
        ledger["dispatches"][dispatch_name] += 1

    def call_step(state, branch):
        before = training.counts(state)
        expected = {"critic": 1, "actor": int(before["phase"] == 1), "auxiliary": 1,
                    "refresh": int(before["critic"] + 1 in TINY_CONFIG["refresh_at"])}
        reserve(expected, branch + "_microsteps")
        try:
            updated, metrics = training.step(cache["ctx"], state)
        except BaseException as exc:
            attempted = getattr(exc, "attempted_counts", None)
            ledger["failed_dispatches"].append({"branch": branch,
                "stage": getattr(exc, "stage", "unknown"), "type": type(exc).__name__,
                "reported_attempted_counts": attempted,
                "actual_completion": "not established; reserved bounds are not completed updates"})
            if hasattr(exc, "last_committed_state"):
                compare(exc.last_committed_state, state, exact=True, label="failed-step last committed snapshot")
            raise
        after = training.counts(updated)
        increments = {"critic": after["critic"] - before["critic"],
                      "actor": after["actor"] - before["actor"],
                      "auxiliary": after["auxiliary"] - before["auxiliary"],
                      "refresh": after["refresh_accepted"] - before["refresh_accepted"]}
        for key in CAP:
            ledger["verified_returned_operations"][key] += increments[key]
        assert_true(increments == expected, "Committed operation counts differ from dispatched phase")
        return updated, metrics

    def fixture(rows, seed):
        generator = np.random.default_rng(seed)
        obs = generator.normal(0.0, 0.7, (rows, 14)).astype(np.float32)
        action = generator.uniform(-0.9, 0.9, (rows, 4)).astype(np.float32)
        reward = generator.uniform(-3.0, -0.1, rows).astype(np.float32)
        next_obs = generator.normal(0.0, 0.7, (rows, 14)).astype(np.float32)
        done = ((np.arange(rows) % 7) == 0).astype(np.float32)
        return native.Transition(*(jnp.asarray(x) for x in (obs, action, reward, next_obs, done)))

    def initialize():
        train, heldout = fixture(32, TRAIN_SEED), fixture(128, HELDOUT_SEED)
        train_ids = np.arange(32, dtype=np.int64)
        heldout_ids = np.arange(1000, 1128, dtype=np.int64)
        assert_true(not np.intersect1d(train_ids, heldout_ids).size, "Synthetic episode groups overlap")
        ctx, initial = training.create(native, train, heldout,
            {"row_ids": heldout_ids, "episode_ids": heldout_ids}, deepcopy(TINY_CONFIG))
        assert_true(training.validate(initial)["passed"], "Initial state failed validation")
        assert_true(training.counts(initial) == expected_counts(0), "Unexpected initial counters")
        bound = deepcopy(bindings)
        bound.update({"config": deepcopy(TINY_CONFIG), "datasetidentity": {
            "kind": "synthetic_checkpoint_gate_only", "train_rows": 32, "heldout_rows": 128,
            "observation_dimension": 14, "action_dimension": 4,
            "train_seed": TRAIN_SEED, "heldout_seed": HELDOUT_SEED,
            "train_sha256": fingerprint(train), "heldout_sha256": fingerprint(heldout),
            "train_episode_ids": train_ids.tolist(), "heldout_episode_ids": heldout_ids.tolist(),
            "heldout_row_ids": heldout_ids.tolist(), "environment_rollouts": 0},
            "normalization": {"mean": np.asarray(ctx.obs_mean).tolist(), "std": np.asarray(ctx.obs_std).tolist()},
            "training_stage_reviewed": False, "eligible_for_usefulness": False,
            "purpose": "Synthetic boundary-interruption and checkpoint correctness gate"})
        json.dumps(bound, sort_keys=True, allow_nan=False)
        cache.update(ctx=ctx, initial=initial, bindings=bound,
                     initial_fingerprint=fingerprint(initial), train=train, heldout=heldout)
        state_log[0] = initial
        return {"effective_native_args": asdict(training.native_args(ctx)),
                "fixture": bound["datasetidentity"], "normalization": bound["normalization"],
                "initial_state_sha256": cache["initial_fingerprint"], "initial_counts": training.counts(initial),
                "synthetic_only": True, "normalization_formula": "(observation - mean) / (std + 0.001)"}

    record("synthetic_initialization_and_disjoint_fixture_bindings", initialize)

    for number in range(1, 7):
        def main_step(number=number):
            prior = state_log[number - 1]
            state, metrics = call_step(prior, "main")
            state_log[number], metrics_log[number] = state, metrics
            cache["main_" + str(number)] = state
            assert_true(training.validate(state)["passed"], "Committed state failed validation")
            assert_true(training.counts(state) == expected_counts(number), "Unexpected main trajectory counters")
            assert_true(metrics["counts"] == expected_counts(number), "Metrics counters disagree")
            assert_true(metrics["refresh_due"] == (number in (2, 4, 6)), "Refresh schedule differs")
            assert_true(metrics["committed"] is True and metrics["kappa_penalty"] == 0.0, "Native control altered")
            if number % 2 == 0:
                compare(prior.microstep.batch, state.microstep.batch, exact=True,
                        label="Native batch reuse across paired critics")
            return {"counts": training.counts(state), "state_sha256": fingerprint(state),
                    "refresh_due": metrics["refresh_due"],
                    "reference_threshold": float(np.asarray(state.aux.posterior.threshold)) if number >= 2 else "unready_positive_infinity"}

        record("uninterrupted_critic_" + str(number), main_step,
               requires=("ctx",) if number == 1 else ("main_" + str(number - 1),))

        if number == 1:
            def save_odd_snapshot():
                state = state_log[1]
                assert_true(training.counts(state)["phase"] == 1, "Must checkpoint between paired critics")
                path = output / "synthetic_odd_critic_000001"
                receipt = checkpoint_io.save(path, state, cache["bindings"], training.counts(state))
                cache.update(checkpoint=path, checkpoint_hashes=checkpoint_hashes(path),
                             checkpoint_state_sha256=fingerprint(state), checkpoint_receipt=receipt)
                return {"path": str(path), "receipt": receipt, "counts": training.counts(state),
                        "pending_batch_sha256": fingerprint(state.microstep.batch),
                        "state_sha256": fingerprint(state),
                        "interrupt_scope": "Boundary interruption modeled after a committed call; no OS signal injected",
                        "training_stage_reviewed": False, "eligible_for_usefulness": False}
            record("save_complete_odd_critic_interruption_snapshot", save_odd_snapshot, requires=("main_1",))

        if number == 3:
            def ready_reference_roundtrip():
                state = state_log[3]
                assert_true(training.validate(state)["passed"], "Finite-ready phase-one snapshot invalid")
                assert_true(training.counts(state)["reference_ready"], "Reference should be ready after refresh at critic 2")
                assert_true(training.counts(state)["last_refresh_critic"] == 2, "Wrong frozen-reference age")
                assert_true(int(np.asarray(state.aux.calibrator.step)) == 3, "Live calibrator must retain third fit")
                compare(state.aux.posterior, state_log[2].aux.posterior, exact=True,
                        label="Frozen reference retained across third auxiliary fit")
                path = output / "synthetic_ready_critic_000003"
                receipt = checkpoint_io.save(path, state, cache["bindings"], training.counts(state))
                before = checkpoint_hashes(path)
                restored, restored_receipt = checkpoint_io.load(path, cache["initial"], cache["bindings"])
                comparison = compare(restored, state, exact=True, label="Finite-ready full checkpoint roundtrip")
                assert_true(training.validate(restored)["passed"], "Restored finite-ready state invalid")
                assert_true(checkpoint_hashes(path) == before, "Ready-reference load altered the saved checkpoint")
                return {"path": str(path), "counts": training.counts(restored), "comparison": comparison,
                        "receipt": receipt, "restored_receipt": restored_receipt,
                        "frozen_reference_from_critic": 2, "live_auxiliary_fit_count": 3,
                        "training_stage_reviewed": False, "eligible_for_usefulness": False,
                        "additional_optimizer_updates": 0}
            record("finite_ready_reference_roundtrip_with_newer_live_auxiliary_state", ready_reference_roundtrip,
                   requires=("main_3",))

    def restore_snapshot():
        restored, receipt = checkpoint_io.load(cache["checkpoint"], cache["initial"], cache["bindings"])
        comparison = compare(restored, state_log[1], exact=True, label="Restored odd-critic snapshot")
        assert_true(fingerprint(state_log[1]) == cache["checkpoint_state_sha256"], "Continued main run mutated saved boundary state")
        assert_true(checkpoint_hashes(cache["checkpoint"]) == cache["checkpoint_hashes"], "Main run modified checkpoint")
        assert_true(training.validate(restored)["passed"], "Restored state failed validation")
        cache["replay_1"] = restored
        return {"comparison": comparison, "counts": training.counts(restored), "receipt": receipt,
                "includes": ["native optimizer and targets", "native RNG", "pending batch and first critic loss",
                             "auxiliary optimizer and RNG", "frozen reference", "schedule and counters"]}
    record("restore_boundary_snapshot_exactly", restore_snapshot, requires=("checkpoint",))

    for number in range(2, 7):
        def replay_step(number=number):
            state, metrics = call_step(cache["replay_" + str(number - 1)], "replay")
            cache["replay_" + str(number)] = state
            state_comparison = compare(state, state_log[number], exact=True, label="Resume full state at critic " + str(number))
            metric_comparison = compare(metrics, metrics_log[number], exact=True, label="Resume full metrics at critic " + str(number))
            assert_true(training.validate(state)["passed"], "Resumed state failed validation")
            return {"counts": training.counts(state), "state": state_comparison, "metrics": metric_comparison,
                    "state_sha256": fingerprint(state)}
        record("exact_replay_critic_" + str(number), replay_step,
               requires=("replay_" + str(number - 1), "main_" + str(number)))

    def baseline_initialize():
        # Initialization is independently reproduced, without an optimizer call.
        loader = training._module("native_loader", training.STAGE1 / "native_loader.py")
        rng, agent = loader.initialize(native, training.native_args(cache["ctx"]), cache["train"], TINY_CONFIG["seed"])
        compare(agent, cache["initial"].microstep.agent_state, exact=True, label="Native initialization")
        compare(rng, cache["initial"].microstep.rng, exact=True, label="Native initial RNG")
        cache["baseline_0"] = (rng, agent)
        return {"native_initialization_exact": True, "auxiliary_work_does_not_consume_native_rng": "tested below"}
    record("independent_native_baseline_initialization", baseline_initialize, requires=("ctx",))

    for outer in range(1, 4):
        def baseline_step(outer=outer):
            reserve({"critic": 2, "actor": 1, "auxiliary": 0, "refresh": 0}, "native_outer_steps")
            before_rng, before_agent = cache["baseline_" + str(outer - 1)]
            try:
                pair, metrics = cache["ctx"].native_step((before_rng, before_agent), None)
                pair, metrics = jax.tree_util.tree_map(
                    lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, (pair, metrics))
            except BaseException as exc:
                ledger["failed_dispatches"].append({"branch": "native_baseline", "stage": "native_outer_step",
                    "type": type(exc).__name__, "reported_attempted_counts": None,
                    "actual_completion": "not established; at most two critics and one actor dispatched"})
                raise
            rng, agent = pair
            verified = {"critic": int(np.asarray(agent.dual_q.step)) - int(np.asarray(before_agent.dual_q.step)),
                        "actor": int(np.asarray(agent.actor.step)) - int(np.asarray(before_agent.actor.step)),
                        "auxiliary": 0, "refresh": 0}
            for key in CAP:
                ledger["verified_returned_operations"][key] += verified[key]
            assert_true(verified == {"critic": 2, "actor": 1, "auxiliary": 0, "refresh": 0}, "Native step count mismatch")
            cache["baseline_" + str(outer)] = pair
            target = state_log[2 * outer].microstep
            state_comparison = compare(agent, target.agent_state, exact=False, label="Native baseline full agent")
            compare(rng, target.rng, exact=True, label="Native baseline RNG")
            mapped = {"actor_loss": metrics["actor_loss"], "q_loss": metrics["q_loss"],
                      "q_mean": metrics["q_mean"], "q_scale_coefficient": metrics["lambda"], "bc_loss": metrics["bc_loss"]}
            # Scheduler metrics are JSON-host scalars. Cast them back to the
            # corresponding native dtype solely for the existing float check.
            candidate_metrics = {name: np.asarray(metrics_log[2 * outer]["native"][name], dtype=np.asarray(value).dtype)
                                 for name, value in mapped.items()}
            metric_comparison = compare(mapped, candidate_metrics, exact=False, label="Native baseline shared metrics")
            return {"critic": 2 * outer, "actor": outer, "full_native_agent": state_comparison,
                    "native_rng_exact": True, "shared_metrics": metric_comparison,
                    "auxiliary_updates_on_candidate_only": 2 * outer}
        record("native_baseline_outer_" + str(outer), baseline_step,
               requires=("baseline_" + str(outer - 1), "main_" + str(2 * outer)))

    def rejects_wrong_binding():
        wrong = deepcopy(cache["bindings"])
        wrong["config"]["gamma"] = 0.97
        try:
            checkpoint_io.load(cache["checkpoint"], cache["initial"], wrong)
        except checkpoint_io.CheckpointError as exc:
            assert_true(checkpoint_hashes(cache["checkpoint"]) == cache["checkpoint_hashes"], "Binding rejection modified checkpoint")
            return {"rejected": True, "error": str(exc), "changed_binding": "config.gamma", "valid_checkpoint_unchanged": True}
        raise AssertionError("Checkpoint load accepted a different bound gamma")
    record("wrong_scientific_binding_rejected", rejects_wrong_binding, requires=("checkpoint",))

    def rejects_truncated_payload():
        path = output / "negative_truncated_checkpoint_copy"
        shutil.copytree(cache["checkpoint"], path)
        payload_path = path / "state.msgpack"
        original = payload_path.read_bytes()
        assert_true(len(original) > 1, "Unexpected empty checkpoint payload")
        payload_path.write_bytes(original[:-1])
        try:
            checkpoint_io.load(path, cache["initial"], cache["bindings"])
        except checkpoint_io.CheckpointError as exc:
            assert_true(checkpoint_hashes(cache["checkpoint"]) == cache["checkpoint_hashes"], "Truncation probe modified original")
            return {"rejected": True, "error": str(exc), "negative_copy": str(path), "valid_checkpoint_unchanged": True}
        raise AssertionError("Checkpoint load accepted a truncated payload")
    record("truncated_payload_copy_rejected", rejects_truncated_payload, requires=("checkpoint",))

    def rejects_overwrite():
        try:
            checkpoint_io.save(cache["checkpoint"], state_log[1], cache["bindings"], training.counts(state_log[1]))
        except (checkpoint_io.CheckpointError, FileExistsError) as exc:
            assert_true(checkpoint_hashes(cache["checkpoint"]) == cache["checkpoint_hashes"], "Duplicate save modified original")
            return {"rejected": True, "error_type": type(exc).__name__, "valid_checkpoint_unchanged": True}
        raise AssertionError("Save overwrote an existing checkpoint directory")
    record("exclusive_checkpoint_save_rejects_overwrite", rejects_overwrite, requires=("checkpoint",))

    def final_integrity():
        restored, _ = checkpoint_io.load(cache["checkpoint"], cache["initial"], cache["bindings"])
        compare(restored, state_log[1], exact=True, label="Original checkpoint after negative probes")
        assert_true(checkpoint_hashes(cache["checkpoint"]) == cache["checkpoint_hashes"], "Original artifact changed")
        assert_true(fingerprint(cache["initial"]) == cache["initial_fingerprint"], "Initial immutable snapshot changed")
        assert_true(cache["bindings"]["training_stage_reviewed"] is False and cache["bindings"]["eligible_for_usefulness"] is False,
                    "Synthetic checkpoint eligibility guard changed")
        return {"original_checkpoint_usable": True, "initial_and_boundary_snapshots_unchanged": True,
                "synthetic_checkpoint_ineligible_for_usefulness": True}
    record("original_checkpoint_survives_all_negative_probes", final_integrity, requires=("checkpoint",))

    def complete_budget():
        assert_true(ledger["verified_returned_operations"] == CAP, "Gate did not complete all declared operations")
        assert_true(ledger["reserved_dispatch_upper_bounds"] == CAP and not ledger["failed_dispatches"], "Unexpected dispatched work")
        assert_true(training.counts(cache["main_6"]) == expected_counts(6), "Main final count mismatch")
        assert_true(training.counts(cache["replay_6"]) == expected_counts(6), "Resume final count mismatch")
        assert_true(cache["main_6"].microstep.reference is None, "Auxiliary reference entered native control")
        return {"operations": dict(CAP), "optimizer_updates_total_including_auxiliary": 37,
                "primary_critic_plus_actor_optimizer_updates": 26,
                "main": {"critic": 6, "actor": 3, "auxiliary": 6, "refresh": 3},
                "resume_replay": {"critic": 5, "actor": 3, "auxiliary": 5, "refresh": 3},
                "independent_native_baseline": {"critic": 6, "actor": 3, "auxiliary": 0, "refresh": 0},
                "environment_rollouts": 0, "final_reference_ready_and_finite": True}
    record("exact_completed_work_ledger_and_scope", complete_budget,
           requires=("main_6", "replay_6", "baseline_3"))

    passed = all(item["status"] == "pass" for item in results)
    report = {
        "schema": "unifloral.synthetic_checkpoint_gate.v1", "status": "pass" if passed else "fail",
        "passed": passed, "checks": results,
        "check_counts": {status: sum(x["status"] == status for x in results) for status in ("pass", "fail", "skip")},
        "counts_ledger": ledger, "scientific_config": deepcopy(TINY_CONFIG),
        "tolerance": {"resume_state_and_metrics": "exact array equality including nonfinite warmup sentinels; no NaNs",
                      "native_baseline_float_rtol": RTOL, "native_baseline_float_atol": ATOL,
                      "counters_and_rng": "exact"},
        "environment_rollouts": 0, "training_stage_reviewed": False, "eligible_for_usefulness": False,
        "claims": ["Synthetic continuation and native-control correctness only",
                   "No BCA efficacy, useful-training checkpoint, production refresh cadence, or OS-signal delivery claim",
                   "Exact comparisons cover serialized dynamic state; source/config/statistics bind static templates"],
    }
    # Fail here rather than let a serializer silently publish NaN/Infinity.
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report
