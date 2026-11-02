"""Bounded actual-BCA composition checks; invoked only by the user-run harness.

No scientific imports occur at module import time. No primary critic or actor
optimizer is called. Synthetic fixtures carry no environment or efficacy claim.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import traceback


RTOL = 2e-5
ATOL = 2e-6
KAPPAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SCALES = (0.1, 1.0, 10.0)
MAX_AUXILIARY_ATTEMPTS = 4
MAX_REFRESH_CALLS = 3


def run(native, output_dir, pipeline_root):
    """Exercise actual auxiliary fitting/refresh plus zero-update actor gradients.

    Maximum work: fit1, fit2, replay-fit2, invalid-fit attempt; main refresh,
    replay refresh, controlled head-mismatch refresh. All native snapshots stay
    fixed. Early manual refreshes are test-only, not a training schedule choice.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from native_loader import initialize
    from candidate import make_microsteps
    from bca_bridge import load_pipeline

    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise ValueError("Harness must provide an existing new output directory")
    checks, context = [], {}
    ledger = {"auxiliary_attempts": 0, "auxiliary_returned_step_increments": 0,
              "refresh_calls": 0, "refresh_inputs_accepted": 0,
              "primary_critic_updates": 0, "actor_updates": 0,
              "environment_rollouts": 0}

    def safe(value):
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [safe(v) for v in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, (float, np.floating)):
            number = float(value)
            if math.isnan(number):
                return "NaN"
            if math.isinf(number):
                return "+Infinity" if number > 0 else "-Infinity"
            return number
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        return safe(np.asarray(value).tolist())

    def atomic_receipt(name, value):
        target = output_dir / name
        temporary = output_dir / (name + ".pending")
        if target.exists():
            raise FileExistsError("Check receipts are immutable: " + str(target))
        payload = json.dumps(safe(value), indent=2, allow_nan=False) + "\n"
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    def record(name, fn, *, category="pipeline", requires=True):
        entry = {"name": name, "category": category}
        if not requires:
            entry.update(status="skipped", reason="Prerequisite check failed")
        else:
            try:
                entry.update(status="pass", details=safe(fn()))
            except Exception as error:
                entry.update(status="fail", error_type=type(error).__name__,
                             error=str(error), traceback=traceback.format_exc())
        entry["work_ledger_after"] = dict(ledger)
        checks.append(entry)
        atomic_receipt(f"pipeline_check_{len(checks):02d}_{name}.json", entry)
        print(json.dumps({"check": name, "status": entry["status"]}), flush=True)
        return entry["status"] == "pass"

    def fingerprint(tree):
        return hashlib.sha256(serialization.to_bytes(tree)).hexdigest()

    def scalar(value):
        result = float(np.asarray(value))
        if not math.isfinite(result):
            raise AssertionError("Expected a finite scalar")
        return result

    def integer(value):
        return int(np.asarray(value))

    def truth(value):
        return bool(np.asarray(value))

    def compare(left, right, *, exact=False, allow_infinity=False):
        leaves_a, shape_a = jax.tree_util.tree_flatten(serialization.to_state_dict(left))
        leaves_b, shape_b = jax.tree_util.tree_flatten(serialization.to_state_dict(right))
        if shape_a != shape_b:
            raise AssertionError("Serialized pytree structures differ")
        largest = 0.0
        for left_leaf, right_leaf in zip(leaves_a, leaves_b):
            a, b = np.asarray(left_leaf), np.asarray(right_leaf)
            integer_leaves = a.dtype.kind in "biu" and b.dtype.kind in "biu"
            if a.shape != b.shape or (a.dtype != b.dtype and not integer_leaves):
                raise AssertionError("State leaf shape or dtype differs")
            if np.any(np.isnan(a)) or np.any(np.isnan(b)):
                raise AssertionError("NaN in compared state")
            if not allow_infinity and (not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
                raise AssertionError("Nonfinite compared state")
            if allow_infinity:
                np.testing.assert_array_equal(np.isposinf(a), np.isposinf(b))
                np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
            if exact or integer_leaves:
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
            finite = np.isfinite(a) & np.isfinite(b)
            if np.any(finite):
                largest = max(largest, float(np.max(np.abs(
                    a[finite].astype(np.float64) - b[finite].astype(np.float64)))))
        return largest

    def norm(tree):
        squared = sum(float(np.sum(np.asarray(a, np.float64) ** 2))
                      for a in jax.tree_util.tree_leaves(tree))
        if not math.isfinite(squared):
            raise AssertionError("Nonfinite gradient norm")
        return math.sqrt(squared)

    def nonzero_norm(tree, name):
        value = norm(tree)
        if value <= 1e-8:
            raise AssertionError(name + " is vacuous or below the fixed 1e-8 floor")
        return value

    def tree_sub(a, b):
        return jax.tree_util.tree_map(lambda x, y: x - y, a, b)

    def tree_scale(factor, tree):
        return jax.tree_util.tree_map(lambda x: factor * x, tree)

    def verify_native(view):
        agent = context["native_agent"]
        for view_name, native_name in (("actor", "actor"), ("actor_target", "actor_target"),
                                       ("critic", "dual_q"), ("critic_target", "dual_q_target")):
            compare(getattr(view, view_name), getattr(agent, native_name), exact=True)
            if integer(getattr(view, view_name).step) != 0:
                raise AssertionError("A native optimizer or target counter changed")

    def fit(carry, batch):
        if ledger["auxiliary_attempts"] >= MAX_AUXILIARY_ATTEMPTS:
            raise AssertionError("Auxiliary attempt ceiling reached before dispatch")
        ledger["auxiliary_attempts"] += 1
        next_rng, key = jax.random.split(carry["fit_rng"])
        before = carry["state"]
        proposed, flags = context["pipeline"].fit_scale(
            context["aux_args"], context["config"], context["models"],
            before, batch, key, 1.0)
        jax.block_until_ready((proposed, flags))
        delta = integer(proposed.calibrator.step) - integer(before.calibrator.step)
        ledger["auxiliary_returned_step_increments"] += delta
        accepted = truth(flags["scale_inputs_valid"]) and truth(flags["scale_fit_accepted"])
        if delta != int(accepted):
            raise AssertionError("Returned auxiliary TrainState step disagrees with acceptance")
        verify_native(proposed.native)
        # The test transaction commits only the source function's accepted result.
        # This does not define a new production training policy.
        if not accepted:
            compare(proposed, before, exact=True, allow_infinity=True)
            return carry, flags
        return {**carry, "state": proposed, "fit_rng": next_rng,
                "fit_calls": carry["fit_calls"] + jnp.int32(1)}, flags

    def refresh(carry, heldout=None, models=None, heldout_ids=None):
        if ledger["refresh_calls"] >= MAX_REFRESH_CALLS:
            raise AssertionError("Refresh ceiling reached before dispatch")
        ledger["refresh_calls"] += 1
        bank = context["heldout"] if heldout is None else heldout
        selected_models = context["models"] if models is None else models
        ids = context["heldout_ids"] if heldout_ids is None else heldout_ids
        next_rng, key = jax.random.split(carry["refresh_rng"])
        proposed, flags, diagnostics = context["pipeline"].refresh(
            context["aux_args"], context["config"], selected_models, carry["state"],
            bank, key, 1.0, heldout_ids=ids)
        accepted = truth(flags["posterior_inputs_valid"])
        ledger["refresh_inputs_accepted"] += int(accepted)
        verify_native(proposed.native)
        if not accepted:
            compare(proposed, carry["state"], exact=True, allow_infinity=True)
            return carry, flags, diagnostics
        return {**carry, "state": proposed, "refresh_rng": next_rng,
                "refresh_calls": carry["refresh_calls"] + jnp.int32(1)}, flags, diagnostics

    def initialization():
        pipeline = load_pipeline(pipeline_root)
        seed, fixture_seed, auxiliary_seed = 174, 2026102307, 2026102308
        args = replace(native.Args(), seed=seed, dataset="synthetic_bca_pipeline_only",
                       num_updates=0, batch_size=8, eval_final_episodes=0)
        if args.gamma != 0.99 or args.num_critic_updates_per_step != 2:
            raise AssertionError("Native gamma or critic cadence changed")
        gen = np.random.default_rng(fixture_seed)

        def transitions(n):
            return native.Transition(
                jnp.asarray(gen.normal(size=(n, 14)).astype(np.float32)),
                jnp.asarray(gen.uniform(-0.9, 0.9, size=(n, 4)).astype(np.float32)),
                jnp.asarray(gen.uniform(-4.0, -0.1, size=n).astype(np.float32)),
                jnp.asarray(gen.normal(size=(n, 14)).astype(np.float32)),
                jnp.asarray((np.arange(n) % 7 == 0).astype(np.float32)),
            )

        dataset, heldout = transitions(32), transitions(128)
        train_ids = np.arange(32, dtype=np.int64)
        heldout_ids = np.arange(1000, 1128, dtype=np.int64)
        if np.intersect1d(train_ids, heldout_ids).size:
            raise AssertionError("Synthetic training and reference IDs overlap")
        native_rng, agent = initialize(native, args, dataset, seed)
        models, state, config, aux_args = pipeline.initialize_pipeline(
            agent, dataset, args, auxiliary_seed)
        context.update(pipeline=pipeline, args=args, native_agent=agent, native_rng=native_rng,
                       native_rng_sha256=fingerprint(native_rng), native_sha256=fingerprint(agent),
                       dataset=dataset, heldout=heldout, heldout_ids=heldout_ids,
                       fixed_batch=jax.tree_util.tree_map(lambda x: x[:8], dataset),
                       models=models, config=config, aux_args=aux_args,
                       fixture_seed=fixture_seed, auxiliary_seed=auxiliary_seed)
        verify_native(state.native)
        if integer(state.calibrator.step) != 0:
            raise AssertionError("Auxiliary initialization performed an optimizer update")
        if (config.posterior.alpha, config.posterior.credibility, config.posterior.draws) != (0.1, 0.95, 1000):
            raise AssertionError("Production calibration defaults changed")
        if (config.cal_lr, config.cal_beta, config.width_penalty, config.scale_ema) != (0.001, 20.0, 0.005, 0.99):
            raise AssertionError("Production auxiliary fitting defaults changed")
        carry = {"state": state,
                 "fit_rng": jax.random.fold_in(jax.random.PRNGKey(auxiliary_seed), 7001),
                 "refresh_rng": jax.random.fold_in(jax.random.PRNGKey(auxiliary_seed), 7002),
                 "fit_calls": jnp.int32(0), "refresh_calls": jnp.int32(0),
                 "native_rng": native_rng}
        context["initial"] = carry
        return {"native_snapshot_sha256": context["native_sha256"],
                "native_rng_sha256": context["native_rng_sha256"],
                "training_sha256": fingerprint(dataset), "heldout_sha256": fingerprint(heldout),
                "training_rows": 32, "heldout_rows": 128, "batch_size": 8,
                "heldout_ids_sha256": hashlib.sha256(heldout_ids.tobytes()).hexdigest(),
                "seeds": {"native": seed, "fixture": fixture_seed, "auxiliary": auxiliary_seed},
                "preprocessing": "Declared fixture binding: raw observations; Calibrator uses native train-only mean/std and divides by std+0.001. Not old portable preprocessing parity.",
                "config": asdict(config), "native_args": asdict(args),
                "source_receipt": pipeline.receipt,
                "data_scope": "Independent synthetic row banks; no environment/trajectory/coverage claim"}

    initialized = record("initialize_fixed_native_and_actual_auxiliary", initialization)

    def first_fit():
        carry, flags = fit(context["initial"], context["fixed_batch"])
        if not truth(flags["scale_fit_accepted"]):
            raise AssertionError("First valid auxiliary fit rejected")
        context["after_one"] = carry
        return {"flags": flags, "auxiliary_step": integer(carry["state"].calibrator.step),
                "residual_scale": scalar(carry["state"].residual_scale)}

    first_ok = record("actual_auxiliary_fit_one", first_fit, requires=initialized)

    def second_fit():
        carry, flags = fit(context["after_one"], context["fixed_batch"])
        if not truth(flags["scale_fit_accepted"]):
            raise AssertionError("Second valid auxiliary fit rejected")
        context["after_two"] = carry
        return {"flags": flags, "auxiliary_step": integer(carry["state"].calibrator.step),
                "residual_scale": scalar(carry["state"].residual_scale)}

    second_ok = record("actual_auxiliary_fit_two", second_fit, requires=first_ok)

    def main_refresh():
        carry, flags, diagnostics = refresh(context["after_two"])
        ref = carry["state"].posterior
        if not truth(flags["posterior_inputs_valid"]) or not truth(flags["posterior_certified"]):
            raise AssertionError("Predetermined 128-row reference did not certify a finite threshold")
        if not truth(ref.ready) or not math.isfinite(scalar(ref.threshold)):
            raise AssertionError("Expected a ready finite reference")
        context["ready"] = carry
        return {"flags": flags, "diagnostics": diagnostics,
                "reference_sha256": fingerprint(ref),
                "schedule": "Manual refresh after two auxiliary fits: test-only; production warmup 10000 and interval 5000 are not exercised or changed"}

    ready_ok = record("manual_actual_reference_refresh", main_refresh, requires=second_ok)

    def checkpoint_resume():
        saved = context["after_one"]
        blob = serialization.to_bytes(saved)
        path = output_dir / "auxiliary_after_000001.msgpack"
        with path.open("xb") as stream:
            stream.write(blob)
        restored = serialization.from_bytes(saved, path.read_bytes())
        compare(restored, saved, exact=True, allow_infinity=True)
        replay, flags = fit(restored, context["fixed_batch"])
        if not truth(flags["scale_fit_accepted"]):
            raise AssertionError("Replay fit rejected")
        compare(replay, context["after_two"], exact=True, allow_infinity=True)
        replay, refresh_flags, diagnostics = refresh(replay)
        if not truth(refresh_flags["posterior_certified"]):
            raise AssertionError("Replay refresh uncertified")
        compare(replay, context["ready"], exact=True)
        context["replayed"] = replay
        return {"checkpoint": path.name, "checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
                "checkpoint_bytes": len(blob), "restored_full_state_exact": True,
                "fit_and_refresh_replay_exact": True, "diagnostics": diagnostics,
                "includes": ["auxiliary parameters and Adam state", "frozen reference", "residual unit",
                             "fit RNG", "refresh RNG", "native RNG", "native snapshot", "counters"]}

    record("auxiliary_serialization_and_exact_replay", checkpoint_resume,
           category="resume", requires=ready_ok)

    def invalid_fit():
        before = context["ready"]
        batch = context["fixed_batch"]._replace(
            reward=context["fixed_batch"].reward.at[0].set(jnp.nan))
        after, flags = fit(before, batch)
        if truth(flags["scale_inputs_valid"]) or truth(flags["scale_fit_accepted"]):
            raise AssertionError("Invalid auxiliary input was accepted")
        compare(after, before, exact=True)
        return {"flags": flags, "state_rng_and_counters_unchanged_exactly": True,
                "scope": "Existing auxiliary rejection plus explicit test transaction; no production policy change"}

    record("invalid_auxiliary_fit_preserves_snapshot", invalid_fit,
           category="rejection", requires=ready_ok)

    def width_fn(ref, obs, actions):
        width, _, _ = context["pipeline"].live_width(context["models"], ref, obs, actions)
        return width

    def functions(kappa, *, actor_apply=None, q_apply=None, width=width_fn):
        agent = context["native_agent"]
        return make_microsteps(context["args"], actor_apply or agent.actor.apply_fn,
                               q_apply or agent.dual_q.apply_fn, context["dataset"],
                               width_fn=width, kappa_penalty=kappa)

    def actor_actions(params):
        agent, batch = context["native_agent"], context["fixed_batch"]
        return jax.vmap(lambda obs: agent.actor.apply_fn(params, obs).sample(seed=None))(batch.obs)

    def raw_q(params):
        agent, batch = context["native_agent"], context["fixed_batch"]
        actions = actor_actions(params)
        return jax.vmap(lambda obs, action: agent.dual_q.apply_fn(
            agent.dual_q.params, obs, action)[0])(batch.obs, actions)

    def actual_width_and_zero():
        agent, batch = context["native_agent"], context["fixed_batch"]
        ref = context["ready"]["state"].posterior
        widths, usable, valid = context["pipeline"].live_width(
            context["models"], ref, batch.obs, actor_actions(agent.actor.params))
        if not truth(usable) or not truth(valid) or not np.all(np.asarray(widths) > 0):
            raise AssertionError("Real finite-reference width is not positive and usable")

        def native_actor_loss(params):
            def one(transition):
                action = agent.actor.apply_fn(params, transition.obs).sample(seed=None)
                q = agent.dual_q.apply_fn(agent.dual_q.params, transition.obs, action)
                return q[0], jnp.square(action - transition.action).mean()
            q, bc = jax.vmap(one)(batch)
            c = jax.lax.stop_gradient(context["args"].td3_alpha / (jnp.abs(q).mean() + 1e-7))
            return (-c * q.mean() + bc.mean()).mean()

        fns = functions(0.0)
        (loss, aux), grad = jax.value_and_grad(lambda p: fns.actor_loss(
            p, agent, batch, ref), has_aux=True)(agent.actor.params)
        native_loss, native_grad = jax.value_and_grad(native_actor_loss)(agent.actor.params)

        def forbidden_width(*unused):
            raise AssertionError("Zero kappa called the width function")
        bypass = functions(0.0, width=forbidden_width).actor_loss(
            agent.actor.params, agent, batch, ref)
        compare((loss, aux), bypass, exact=True)
        return {"width_mean": scalar(widths.mean()), "width_min": scalar(widths.min()),
                "native_loss_error": compare(loss, native_loss),
                "native_actor_gradient_max_abs_error": compare(grad, native_grad),
                "native_actor_gradient_norm": nonzero_norm(grad, "Native actor gradient"),
                "q_scale_coefficient": scalar(aux["q_scale_coefficient"]),
                "zero_kappa_width_callback_bypassed": True}

    width_ok = record("actual_width_and_zero_kappa_native_equivalence", actual_width_and_zero,
                      category="gradient", requires=ready_ok)

    def gradient_identity(kappa):
        agent, batch, args = context["native_agent"], context["fixed_batch"], context["args"]
        ref = context["ready"]["state"].posterior
        f0, fk = functions(0.0), functions(kappa)
        (l0, a0), g0 = jax.value_and_grad(lambda p: f0.actor_loss(
            p, agent, batch, ref), has_aux=True)(agent.actor.params)
        (lk, ak), gk = jax.value_and_grad(lambda p: fk.actor_loss(
            p, agent, batch, ref), has_aux=True)(agent.actor.params)
        mean_width = lambda p: width_fn(ref, batch.obs, actor_actions(p)).mean()
        u, gu = jax.value_and_grad(mean_width)(agent.actor.params)
        c = a0["q_scale_coefficient"]
        independent_c = args.td3_alpha / (jnp.abs(raw_q(agent.actor.params)).mean() + 1e-7)
        compare(c, independent_c)
        compare(ak["q_scale_coefficient"], c)
        delta = tree_sub(gk, g0)
        expected = tree_scale(c * kappa, gu)
        q_gradient = jax.grad(lambda p: -jax.lax.stop_gradient(c) * raw_q(p).mean())(agent.actor.params)
        q_norm = nonzero_norm(q_gradient, "Actor Q gradient")
        base_norm = nonzero_norm(g0, "Baseline actor gradient")
        u_norm = nonzero_norm(gu, "Actual width actor gradient")

        def probe_actor(params, obs):
            return SimpleNamespace(sample=lambda seed=None: params)
        action = jnp.asarray([-0.45, -0.15, 0.15, 0.45], jnp.float32)
        p0, pk = functions(0.0, actor_apply=probe_actor), functions(kappa, actor_apply=probe_actor)
        (al0, aa0), ag0 = jax.value_and_grad(lambda a: p0.actor_loss(
            a, agent, batch, ref), has_aux=True)(action)
        (alk, aak), agk = jax.value_and_grad(lambda a: pk.actor_loss(
            a, agent, batch, ref), has_aux=True)(action)
        action_width = lambda a: width_fn(ref, batch.obs,
                                         jnp.broadcast_to(a, batch.action.shape)).mean()
        action_gu = jax.grad(action_width)(action)
        action_u_norm = nonzero_norm(action_gu, "Actual width action gradient")
        ac = aa0["q_scale_coefficient"]
        aq = lambda a: jax.vmap(lambda obs: agent.dual_q.apply_fn(
            agent.dual_q.params, obs, a)[0])(batch.obs).mean()
        action_q_grad = jax.grad(lambda a: -jax.lax.stop_gradient(ac) * aq(a))(action)
        action_q_norm = nonzero_norm(action_q_grad, "Action Q gradient")
        action_delta = agk - ag0
        return {"kappa_penalty": kappa, "q_scale_coefficient": scalar(c),
                "width_mean": scalar(u), "actor_gradient_identity_max_abs_error": compare(delta, expected),
                "loss_identity_error": compare(lk - l0, c * kappa * u),
                "width_actor_gradient_norm": u_norm, "actor_penalty_gradient_norm": norm(delta),
                "actor_q_gradient_norm": q_norm, "actor_baseline_gradient_norm": base_norm,
                "actor_penalty_to_q_gradient_ratio": norm(delta) / q_norm,
                "actor_penalty_to_baseline_gradient_ratio": norm(delta) / base_norm,
                "width_action_gradient_norm": action_u_norm,
                "action_penalty_gradient_norm": norm(action_delta),
                "action_q_gradient_norm": action_q_norm,
                "action_penalty_to_q_gradient_ratio": norm(action_delta) / action_q_norm,
                "action_gradient_identity_max_abs_error": compare(action_delta, ac * kappa * action_gu),
                "action_loss_identity_error": compare(alk - al0, ac * kappa * action_width(action)),
                "action_probe": "One shared predetermined 4D action across eight rows",
                "interpretation": "Gradient dosage diagnostic, not a selected kappa or performance result"}

    for kappa in KAPPAS:
        record("actual_width_gradient_kappa_" + format(kappa, "g"),
               lambda kappa=kappa: gradient_identity(kappa), category="gradient", requires=width_ok)

    def frozen_reference():
        agent, batch = context["native_agent"], context["fixed_batch"]
        ref = context["ready"]["state"].posterior
        # The discrete ready flag is held fixed; every differentiable reference
        # field is included in the differentiation argument.
        floating = {name: getattr(ref, name) for name in ref._fields if name != "ready"}
        fns = functions(1.0)
        grad = jax.grad(lambda values: fns.actor_loss(
            agent.actor.params, agent, batch, ref._replace(**values))[0])(floating)
        for leaf in jax.tree_util.tree_leaves(grad):
            value = np.asarray(leaf)
            np.testing.assert_array_equal(value, np.zeros_like(value))
        unblocked = jax.grad(lambda params: context["models"][2].apply(
            params, batch.obs, actor_actions(agent.actor.params)).mean())(ref.cal_params)
        return {"actual_actor_reference_gradient_norm": norm(grad),
                "all_differentiable_reference_fields": list(floating),
                "unblocked_calibrator_parameter_gradient_norm": nonzero_norm(
                    unblocked, "Unblocked calibrator gradient"),
                "frozen_reference_sha256": fingerprint(ref), "actor_steps": 0}

    record("actual_reference_gradient_freeze", frozen_reference,
           category="gradient", requires=width_ok)

    def reference_states():
        agent, batch = context["native_agent"], context["fixed_batch"]
        pipeline, ref = context["pipeline"], context["ready"]["state"].posterior
        action = actor_actions(agent.actor.params)
        unready = pipeline.initial_reference(ref.cal_params)
        infinite = ref._replace(threshold=jnp.float32(jnp.inf), lambda_hpd=jnp.float32(jnp.inf),
                                ready=jnp.bool_(True))
        rows = []
        for label, reference in (("not_ready", unready), ("ready_infinite", infinite)):
            width, usable, valid = pipeline.live_width(context["models"], reference, batch.obs, action)
            if not truth(pipeline.reference_valid(reference)) or not truth(valid) or truth(usable):
                raise AssertionError("Legacy reference behavior changed")
            np.testing.assert_array_equal(np.asarray(width), np.zeros_like(np.asarray(width)))
            f0, f1 = functions(0.0), functions(1.0)
            (l0, _), g0 = jax.value_and_grad(lambda p: f0.actor_loss(
                p, agent, batch, reference), has_aux=True)(agent.actor.params)
            (l1, _), g1 = jax.value_and_grad(lambda p: f1.actor_loss(
                p, agent, batch, reference), has_aux=True)(agent.actor.params)
            compare(l0, l1)
            compare(g0, g1)
            would_reject = truth(reference.ready) and not np.isfinite(np.asarray(reference.threshold)).item()
            rows.append({"case": label, "ready": truth(reference.ready), "threshold": reference.threshold,
                         "legacy_storage_valid": True, "legacy_width_valid": True,
                         "legacy_width_usable": False, "legacy_width_mean": scalar(width.mean()),
                         "legacy_positive_kappa_actor_equals_native": True,
                         "proposed_ready_but_nonfinite_rejection": would_reject})
        if not rows[1]["proposed_ready_but_nonfinite_rejection"]:
            raise AssertionError("Proposed rejection diagnostic failed to identify infinite ready reference")
        return {"cases": rows, "design_change_approved": False,
                "scope": "Exposes existing zero-penalty fallback and a separate proposed rejection predicate; no consumption policy was changed"}

    record("legacy_infinite_ready_fallback_and_proposed_rejection", reference_states,
           category="scientific_gate", requires=width_ok)

    def head_mismatch():
        batch = context["heldout"]._replace(
            reward=jnp.zeros_like(context["heldout"].reward),
            done=jnp.ones_like(context["heldout"].done))

        def controlled_heads(params, obs, action):
            return jnp.broadcast_to(jnp.asarray([10.0, 0.0], jnp.float32),
                                    obs.shape[:-1] + (2,))
        models = (context["models"][0], SimpleNamespace(apply=controlled_heads), context["models"][2])
        result, flags, diagnostics = refresh(context["ready"], heldout=batch, models=models)
        if not truth(flags["posterior_certified"]):
            raise AssertionError("Zero-residual 128-row contrast fixture unexpectedly uncertified")
        ref = result["state"].posterior
        compare(ref.threshold, jnp.float32(0), exact=True)
        widths, usable, valid = context["pipeline"].live_width(models, ref, batch.obs, batch.action)
        if not truth(usable) or not truth(valid):
            raise AssertionError("Zero-radius contrast reference is not usable")
        compare(widths, jnp.zeros_like(widths), exact=True)
        return {"fixture": "Explicit controlled terminal predictions: reward=0, Q1=10, Q2=0; not measured model behavior",
                "target": 0.0, "absolute_twin_min_td_residual": 0.0,
                "absolute_q1_td_residual": 10.0, "actual_calibrated_threshold": scalar(ref.threshold),
                "actual_width_mean": scalar(widths.mean()), "diagnostics": diagnostics,
                "conclusion": "A certified twin-min TD-residual width does not establish a Q1-error bound; residual target choice remains unresolved"}

    record("twin_min_residual_is_not_q1_error", head_mismatch,
           category="scientific_gate", requires=ready_ok)

    def scale_checks():
        agent, batch, args = context["native_agent"], context["fixed_batch"], context["args"]
        pipeline = context["pipeline"]
        auxiliary_state = context["ready"]["state"]
        reference = auxiliary_state.posterior
        action = actor_actions(agent.actor.params)
        base_width = width_fn(reference, batch.obs, action)
        raw = raw_q(agent.actor.params)
        denominator = jnp.abs(raw).mean()
        if scalar(denominator) <= 0:
            raise AssertionError("Scale fixture has zero raw-Q denominator")
        base_fn, zero_fn = functions(1.0), functions(0.0)
        (base_loss, base_aux), base_grad = jax.value_and_grad(lambda p: base_fn.actor_loss(
            p, agent, batch, reference), has_aux=True)(agent.actor.params)
        (zero_loss, _), zero_grad = jax.value_and_grad(lambda p: zero_fn.actor_loss(
            p, agent, batch, reference), has_aux=True)(agent.actor.params)
        base_delta = tree_sub(base_grad, zero_grad)
        nonzero_norm(base_delta, "Reference penalty gradient")
        # Same dedicated diagnostic key for every scale; neither saved RNG
        # stream is advanced. These are auxiliary target evaluations only.
        target_key = jax.random.fold_in(jax.random.PRNGKey(context["auxiliary_seed"]), 7003)
        target = pipeline.native_target(context["aux_args"], context["models"],
                                        auxiliary_state.native, batch, target_key, 1.0)
        min_q = context["models"][1].apply(
            auxiliary_state.native.critic.params, batch.obs, batch.action).min(axis=-1)
        residual = target - min_q
        unit = auxiliary_state.residual_scale
        eta = context["models"][2].apply(auxiliary_state.calibrator.params, batch.obs, batch.action)
        score = jnp.abs(residual) / pipeline.positive_scale(eta, unit)
        coverage = pipeline.soft_coverage(target / unit, min_q / unit, eta, context["config"].cal_beta)
        ema = context["config"].scale_ema
        base_next_unit = ema * unit + (1.0 - ema) * jnp.maximum(jnp.std(residual), 1e-6)
        rows = []
        for scale in SCALES:
            scaled_reference = reference._replace(residual_scale=reference.residual_scale * scale)
            scaled_q = lambda params, obs, act, scale=scale: scale * agent.dual_q.apply_fn(params, obs, act)
            fn = functions(1.0, q_apply=scaled_q)
            fn_zero = functions(0.0, q_apply=scaled_q)
            (loss, aux), grad = jax.value_and_grad(lambda p: fn.actor_loss(
                p, agent, batch, scaled_reference), has_aux=True)(agent.actor.params)
            (loss_zero, aux_zero), grad_zero = jax.value_and_grad(lambda p: fn_zero.actor_loss(
                p, agent, batch, scaled_reference), has_aux=True)(agent.actor.params)
            expected_c = args.td3_alpha / (scale * denominator + 1e-7)
            # The 1e-7 native epsilon prevents exact scale invariance, even when
            # Q and uncertainty units scale coherently. No new normalizer used.
            exact_epsilon_ratio = scale * (denominator + 1e-7) / (scale * denominator + 1e-7)
            scaled_width = width_fn(scaled_reference, batch.obs, action)
            compare(scaled_width, scale * base_width)
            compare(aux["q_scale_coefficient"], expected_c)
            compare(aux["q_scale_coefficient"], aux_zero["q_scale_coefficient"])
            compare(aux["bc_loss"], base_aux["bc_loss"])
            delta = tree_sub(grad, grad_zero)
            scaled_batch = batch._replace(reward=scale * batch.reward)
            scaled_models = (context["models"][0], SimpleNamespace(apply=scaled_q), context["models"][2])
            scaled_target = pipeline.native_target(context["aux_args"], scaled_models,
                                                   auxiliary_state.native, scaled_batch, target_key, 1.0)
            scaled_min_q = scaled_models[1].apply(
                auxiliary_state.native.critic.params, batch.obs, batch.action).min(axis=-1)
            scaled_residual = scaled_target - scaled_min_q
            scaled_unit = scale * unit
            scaled_score = jnp.abs(scaled_residual) / pipeline.positive_scale(eta, scaled_unit)
            scaled_coverage = pipeline.soft_coverage(
                scaled_target / scaled_unit, scaled_min_q / scaled_unit, eta, context["config"].cal_beta)
            scaled_next_unit = ema * scaled_unit + (1.0 - ema) * jnp.maximum(jnp.std(scaled_residual), 1e-6)
            rows.append({"scale": scale, "width_ratio_expected": scale,
                         "width_mean": scalar(scaled_width.mean()),
                         "q_scale_coefficient": scalar(aux["q_scale_coefficient"]),
                         "native_epsilon": 1e-7, "mean_abs_unscaled_q1": scalar(denominator),
                         "penalty_ratio_including_native_epsilon": scalar(exact_epsilon_ratio),
                         "penalty_loss_error": compare(loss - loss_zero,
                             exact_epsilon_ratio * (base_loss - zero_loss)),
                         "penalty_gradient_error": compare(delta, tree_scale(exact_epsilon_ratio, base_delta)),
                         "penalty_gradient_norm": norm(delta),
                         "actual_aux_target_scale_error": compare(scaled_target, scale * target),
                         "actual_min_q_residual_scale_error": compare(scaled_residual, scale * residual),
                         "dimensionless_score_invariance_error": compare(scaled_score, score),
                         "actual_soft_coverage_invariance_error": compare(scaled_coverage, coverage),
                         "ema_unit": scalar(scaled_next_unit),
                         "ema_unit_minus_scaled_baseline": scalar(scaled_next_unit - scale * base_next_unit),
                         "ema_std_floor_active": truth(jnp.std(scaled_residual) < 1e-6)})
        analytic = [{"mean_abs_q": d, "scale": s,
                     "q_scale_coefficient": args.td3_alpha / (s * d + 1e-7),
                     "penalty_ratio": s * (d + 1e-7) / (s * d + 1e-7)}
                    for d in (0.0, 1e-9, 1e-7, 1.0) for s in SCALES]
        floor_examples = [{"residual_std": std, "scale": s,
                           "unscaled_floor_term": max(std, 1e-6),
                           "scaled_floor_term": max(s * std, 1e-6),
                           "coherent_expected_floor_term": s * max(std, 1e-6)}
                          for std in (0.0, 1e-8, 1e-6, 1e-3) for s in SCALES]
        return {"fixed_state_results": rows, "analytic_epsilon_examples": analytic,
                "analytic_ema_std_floor_examples": floor_examples,
                "native_zero_q_coefficient": args.td3_alpha / 1e-7,
                "zero_and_near_zero_q_behavior": "Native epsilon yields finite coefficient amplification; no new rejection or denominator normalization",
                "initial_residual_unit": scalar(context["initial"]["state"].residual_scale),
                "scale_caveats": ["Initial residual unit stays 1 unless explicitly transformed.",
                                  "The fixed 1e-6 residual-std floor can break coherent EMA scaling.",
                                  "The native 1e-7 Q denominator epsilon breaks exact penalty scale invariance."],
                "scope": "Fixed-state Q outputs, diagnostic rewards and residual unit scaled coherently; actual auxiliary target/coverage functions evaluated without fitting, refresh or optimizer updates",
                "normalization_decision": "Unresolved; fixed-state identities do not establish cross-environment robustness"}

    record("coherent_q_and_width_scale_with_native_epsilon", scale_checks,
           category="scale", requires=width_ok)

    def final_integrity():
        if fingerprint(context["native_agent"]) != context["native_sha256"]:
            raise AssertionError("Original native agent snapshot changed")
        if fingerprint(context["native_rng"]) != context["native_rng_sha256"]:
            raise AssertionError("Original native RNG changed")
        for key in ("initial", "after_one", "after_two", "ready", "replayed"):
            if key in context:
                verify_native(context[key]["state"].native)
                compare(context[key]["native_rng"], context["native_rng"], exact=True)
        if ledger["auxiliary_attempts"] > MAX_AUXILIARY_ATTEMPTS or ledger["refresh_calls"] > MAX_REFRESH_CALLS:
            raise AssertionError("Declared work budget exceeded")
        if any(ledger[key] != 0 for key in ("primary_critic_updates", "actor_updates", "environment_rollouts")):
            raise AssertionError("Prohibited primary work recorded")
        return {"native_state_and_rng_unchanged_exactly": True, "ledger": dict(ledger),
                "no_candidate_step_or_native_train_step_called": True}

    record("native_integrity_and_bounded_work", final_integrity,
           category="integrity", requires=initialized)
    passed = all(check["status"] == "pass" for check in checks)
    result = {"schema": "unifloral_bca_pipeline_validation.v1", "status": "pass" if passed else "fail",
              "passed": passed, "checks": checks, "check_count": len(checks), "work_ledger": ledger,
              "maximum_auxiliary_attempts": MAX_AUXILIARY_ATTEMPTS,
              "maximum_refresh_calls": MAX_REFRESH_CALLS,
              "tolerances": {"rtol": RTOL, "atol": ATOL, "replay_native_state_rng_reference_gradients": "exact",
                             "nonvacuous_gradient_floor": 1e-8},
              "legacy_behavior_exposure_is_not_design_approval": True,
              "limitations": ["No primary critic, actor, target updates or environment rollouts.",
                              "Frozen native snapshots; no production auxiliary scheduling chosen.",
                              "Manual early refreshes are test-only; production warmup/interval retained as recorded context.",
                              "Synthetic banks do not validate environmental dynamics or statistical coverage.",
                              "Twin-min TD residual versus Q1 error and ready-infinite rejection remain scientific decisions.",
                              "No efficacy, kappa selection, reward-scale robustness or full-budget readiness claim."]}
    atomic_receipt("pipeline_checks_summary.json", result)
    return safe(result)
