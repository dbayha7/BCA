"""Bounded UniFloral TD3+BC parity checks, executed only by the user-run harness.

No scientific modules are imported at module import time.  The width below is a
frozen, synthetic, action-dependent fixture, NOT an implementation of BCA
calibration.  The native architecture and gamma=.99 are preserved.  These are
implementation checks, not a ThrusterChain experiment or efficacy evidence.
"""

from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace


RTOL = 2e-5
ATOL = 2e-6
KAPPAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MAX_CRITIC_UPDATES = 31
MAX_ACTOR_UPDATES = 16
EXPECTED_CHECK_COUNT = 19


def run(native, output_dir):
    """Return JSON-safe checks and exact attempted/verified update accounting.

    Full successful budget: 12 native + 12 zero-kappa candidate + 4 positive-
    kappa candidate + 3 resumed critic calls = 31; actor calls = 6+6+2+2 = 16.
    Resume starts from the already-computed first odd microstep and compares
    against the already-computed fourth microstep; no extra baseline is run.
    Gradient-only checks perform no optimizer updates or environment rollouts.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from native_loader import initialize
    from candidate import make_microsteps

    output_dir = Path(output_dir)
    checks, context = [], {}
    ledger = {"attempted": {"critic": 0, "actor": 0},
              "verified": {"critic": 0, "actor": 0}, "branches": {}}
    seed, fixture_seed = 174, 191
    args = replace(native.Args(), seed=seed, dataset="synthetic_14d_4a_validation_only",
                   num_updates=6, batch_size=8, eval_final_episodes=0)
    gen = np.random.default_rng(fixture_seed)
    observations = gen.normal(size=(32, 14)).astype(np.float32)
    actions = gen.uniform(-0.9, 0.9, size=(32, 4)).astype(np.float32)
    rewards = gen.uniform(-4.0, -0.1, size=32).astype(np.float32)
    next_observations = gen.normal(size=(32, 14)).astype(np.float32)
    done = (np.arange(32) % 7 == 0).astype(np.float32)
    dataset = native.Transition(*(jnp.asarray(a) for a in
                                 (observations, actions, rewards, next_observations, done)))
    fixed_batch = jax.tree_util.tree_map(lambda a: a[:8], dataset)
    reference = {"base": jnp.float32(0.4),
                 "weights": jnp.asarray([0.3, 0.6, 0.9, 1.2], jnp.float32)}

    def synthetic_width(ref, obs, action):
        displacement = action + 0.2 * jnp.tanh(obs[..., :4])
        return ref["base"] + jnp.sum(ref["weights"] * displacement ** 2, axis=-1)

    def scalar(value):
        result = float(np.asarray(value))
        if not np.isfinite(result):
            raise AssertionError("nonfinite scalar")
        return result

    def integer(value):
        return int(np.asarray(value))

    def arrays(tree):
        return serialization.to_state_dict(tree)

    def fingerprint(tree):
        return hashlib.sha256(serialization.to_bytes(tree)).hexdigest()

    def compare(left, right, *, exact=False):
        leaves_l, shape_l = jax.tree_util.tree_flatten(arrays(left))
        leaves_r, shape_r = jax.tree_util.tree_flatten(arrays(right))
        if shape_l != shape_r:
            raise AssertionError("serialized pytree structure mismatch")
        largest = 0.0
        for left_leaf, right_leaf in zip(leaves_l, leaves_r):
            a, b = np.asarray(left_leaf), np.asarray(right_leaf)
            integer_leaves = a.dtype.kind in "biu" and b.dtype.kind in "biu"
            if a.shape != b.shape or (a.dtype != b.dtype and not integer_leaves):
                raise AssertionError("state leaf shape/dtype mismatch")
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise AssertionError("nonfinite state/gradient leaf")
            if exact or integer_leaves:
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
            if a.size:
                largest = max(largest, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
        return largest

    def norm(tree):
        total = sum(float(np.sum(np.asarray(a, np.float64) ** 2))
                    for a in jax.tree_util.tree_leaves(tree))
        if not np.isfinite(total):
            raise AssertionError("nonfinite gradient norm")
        return float(np.sqrt(total))

    def counter_values(state):
        return {name: integer(getattr(state, name)) for name in
                ("phase", "critic_calls", "actor_calls", "actor_target_updates",
                 "critic_target_updates", "outer_steps")}

    def native_counts(agent):
        return {"critic": integer(agent.dual_q.step), "actor": integer(agent.actor.step),
                "actor_target": integer(agent.actor_target.step),
                "critic_target": integer(agent.dual_q_target.step)}

    def verify_counts(state, critic, actor):
        observed = counter_values(state)
        expected = {"phase": critic % 2, "critic_calls": critic, "actor_calls": actor,
                    "actor_target_updates": actor, "critic_target_updates": actor,
                    "outer_steps": actor}
        if observed != expected:
            raise AssertionError(f"candidate counters: {observed} != {expected}")
        if native_counts(state.agent_state) != {"critic": critic, "actor": actor,
                                               "actor_target": actor, "critic_target": actor}:
            raise AssertionError("candidate counters disagree with TrainState steps")
        return observed

    def record(name, fn, *, category="parity", requires=True):
        if not requires:
            checks.append({"name": name, "category": category, "status": "skipped",
                           "reason": "prerequisite check failed"})
            return False
        try:
            details = fn()
            checks.append({"name": name, "category": category, "status": "pass", "details": details})
            return True
        except Exception as error:
            checks.append({"name": name, "category": category, "status": "fail",
                           "error_type": type(error).__name__, "error": str(error)})
            return False

    def dispatch(branch, fn, before, *, native_step=False):
        before_agent = before[1] if native_step else before.agent_state
        expected_critic = 2 if native_step else 1
        expected_actor = 1 if native_step else int(integer(before.phase) == 1)
        if ledger["attempted"]["critic"] + expected_critic > MAX_CRITIC_UPDATES:
            raise AssertionError("critic execution budget exceeded before dispatch")
        if ledger["attempted"]["actor"] + expected_actor > MAX_ACTOR_UPDATES:
            raise AssertionError("actor execution budget exceeded before dispatch")
        item = ledger["branches"].setdefault(branch, {
            "attempted": {"critic": 0, "actor": 0}, "verified": {"critic": 0, "actor": 0}})
        for key, amount in (("critic", expected_critic), ("actor", expected_actor)):
            ledger["attempted"][key] += amount
            item["attempted"][key] += amount
        after, metrics = fn(before, None)
        jax.block_until_ready((after, metrics))
        after_agent = after[1] if native_step else after.agent_state
        old, new = native_counts(before_agent), native_counts(after_agent)
        delta = {name: new[name] - old[name] for name in old}
        for key in ("critic", "actor"):
            ledger["verified"][key] += delta[key]
            item["verified"][key] += delta[key]
        if delta != {"critic": expected_critic, "actor": expected_actor,
                     "actor_target": expected_actor, "critic_target": expected_actor}:
            raise AssertionError(f"unexpected actual TrainState update deltas: {delta}")
        return after, metrics

    def initialization():
        if args.gamma != 0.99 or args.num_critic_updates_per_step != 2:
            raise AssertionError("native gamma or scheduler default changed")
        native_rng, native_agent = initialize(native, args, dataset, seed)
        candidate_rng, candidate_agent = initialize(native, args, dataset, seed)
        compare(native_rng, candidate_rng, exact=True)
        compare(native_agent, candidate_agent, exact=True)
        fns = make_microsteps(args, candidate_agent.actor.apply_fn,
                              candidate_agent.dual_q.apply_fn, dataset,
                              width_fn=synthetic_width, kappa_penalty=0.0)
        initial = fns.initial_state(candidate_rng, candidate_agent, reference=reference)
        verify_counts(initial, 0, 0)
        native_step = jax.jit(native.make_train_step(
            args, native_agent.actor.apply_fn, native_agent.dual_q.apply_fn, dataset))
        context.update(initial=initial, candidate=initial, fns=fns,
                       candidate_step=jax.jit(fns.step), native=(native_rng, native_agent),
                       native_step=native_step)
        return {"native_initial_sha256": fingerprint(native_agent),
                "candidate_initial_sha256": fingerprint(candidate_agent),
                "microstep_initial_sha256": fingerprint(initial),
                "rng": np.asarray(native_rng).tolist(), "counts": counter_values(initial),
                "comparison": "exact initial numerical state and RNG; no optimizer updates"}

    initial_ok = record("native_and_candidate_initialization", initialization)

    def outer_parity(outer):
        before = context["candidate"]
        expected_rng, sample_key = jax.random.split(before.rng)
        indices = jax.random.randint(sample_key, (args.batch_size,), 0, len(dataset.obs))
        expected_batch = jax.tree_util.tree_map(lambda x: x[indices], dataset)
        expected_rng, _ = jax.random.split(expected_rng)
        half, half_metrics = dispatch("candidate_zero_kappa", context["candidate_step"], before)
        verify_counts(half, 2 * outer - 1, outer - 1)
        compare(half.rng, expected_rng, exact=True)
        compare(half.batch, expected_batch, exact=True)
        for name in ("actor", "actor_target", "dual_q_target"):
            compare(getattr(half.agent_state, name), getattr(before.agent_state, name), exact=True)
        if bool(np.asarray(half_metrics["outer_completed"])):
            raise AssertionError("first critic call must not complete an outer step")
        if outer == 1:
            context["midpoint"] = half
        expected_rng, _ = jax.random.split(expected_rng)
        candidate_after, metrics = dispatch("candidate_zero_kappa", context["candidate_step"], half)
        verify_counts(candidate_after, 2 * outer, outer)
        compare(candidate_after.rng, expected_rng, exact=True)
        compare(candidate_after.batch, half.batch, exact=True)
        if not bool(np.asarray(metrics["outer_completed"])):
            raise AssertionError("second critic call must complete an outer step")
        native_after, native_metrics = dispatch("native", context["native_step"],
                                                context["native"], native_step=True)
        compare(candidate_after.rng, native_after[0], exact=True)
        error = compare(candidate_after.agent_state, native_after[1])
        metric_errors = {}
        for key, value in native_metrics.items():
            candidate_key = "q_scale_coefficient" if key == "lambda" else key
            metric_errors[candidate_key] = compare(metrics[candidate_key], value)
        compare(candidate_after.reference, reference, exact=True)
        context["native"], context["candidate"] = native_after, candidate_after
        if outer == 1:
            context["first_boundary"] = candidate_after
        if outer == 2:
            context["resume_expected"] = candidate_after
        return {"outer_step": outer, "counts": counter_values(candidate_after),
                "all_parameters_optimizer_states_and_targets_max_abs_error": error,
                "native_metric_max_abs_errors": metric_errors,
                "rng_exact": True, "batch_reused_exactly": True,
                "batch_indices": np.asarray(indices).tolist(),
                "native_rng": np.asarray(native_after[0]).tolist()}

    parity_ok = initial_ok
    for outer in range(1, 7):
        parity_ok = record(f"native_outer_step_{outer}_vs_two_microsteps",
                           lambda outer=outer: outer_parity(outer), requires=parity_ok)

    def midpoint_resume():
        midpoint = context["midpoint"]
        blob = serialization.to_bytes(midpoint)
        checkpoint_path = output_dir / "microstep_after_000001.msgpack"
        with checkpoint_path.open("xb") as stream:
            stream.write(blob)
        restored = serialization.from_bytes(midpoint, checkpoint_path.read_bytes())
        compare(restored, midpoint, exact=True)
        verify_counts(restored, 1, 0)
        for _ in range(3):
            restored, _ = dispatch("resume_replay", context["candidate_step"], restored)
        expected = context["resume_expected"]
        verify_counts(restored, 4, 2)
        compare(restored.rng, expected.rng, exact=True)
        compare(restored.batch, expected.batch, exact=True)
        compare(restored.reference, expected.reference, exact=True)
        error = compare(restored, expected)
        return {"checkpoint": checkpoint_path.name, "checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
                "checkpoint_bytes": len(blob), "saved_critic_calls": 1, "saved_phase": 1,
                "saved_batch_sha256": fingerprint(midpoint.batch),
                "saved_reference_sha256": fingerprint(midpoint.reference),
                "replay_critic_calls": 3, "replay_actor_calls": 2,
                "full_resumed_state_max_abs_error": error, "rng_and_counters_exact": True}

    record("mid_schedule_checkpoint_serialization_and_resume", midpoint_resume,
           category="resume", requires=parity_ok)

    def positive_updates():
        initial = context["initial"]
        fns = make_microsteps(args, initial.agent_state.actor.apply_fn,
                              initial.agent_state.dual_q.apply_fn, dataset,
                              width_fn=synthetic_width, kappa_penalty=1.0)
        step = jax.jit(fns.step)
        current = fns.initial_state(initial.rng, initial.agent_state, reference=reference)
        boundaries = []
        for number in range(1, 5):
            current, metrics = dispatch("candidate_positive_kappa", step, current)
            verify_counts(current, number, number // 2)
            compare(current.reference, reference, exact=True)
            if number % 2 == 0:
                if scalar(metrics["width_mean"]) <= 0 or scalar(metrics["penalty_mean"]) <= 0:
                    raise AssertionError("positive synthetic width was not consumed")
                boundaries.append({"critic_calls": number, "actor_calls": number // 2,
                                   "width_mean": scalar(metrics["width_mean"]),
                                   "penalty_mean": scalar(metrics["penalty_mean"]),
                                   "actor_loss": scalar(metrics["actor_loss"])})
                if number == 2 and "first_boundary" in context:
                    compare(current.agent_state.dual_q, context["first_boundary"].agent_state.dual_q)
                    compare(current.rng, context["first_boundary"].rng, exact=True)
        return {"kappa_penalty": 1.0, "boundaries": boundaries,
                "reference_frozen_exactly": True, "synthetic_width_only": True}

    record("positive_kappa_four_critic_calls", positive_updates, requires=initial_ok)

    def loss_functions(kappa, actor_apply=None, width_fn=synthetic_width):
        initial = context["initial"].agent_state
        return make_microsteps(args, actor_apply or initial.actor.apply_fn,
                               initial.dual_q.apply_fn, dataset,
                               width_fn=width_fn, kappa_penalty=kappa)

    def gradient_identity(kappa):
        agent = context["initial"].agent_state
        fns0, fns_k = loss_functions(0.0), loss_functions(kappa)
        fn0 = lambda p: fns0.actor_loss(p, agent, fixed_batch, reference)
        fnk = lambda p: fns_k.actor_loss(p, agent, fixed_batch, reference)
        (loss0, aux0), grad0 = jax.value_and_grad(fn0, has_aux=True)(agent.actor.params)
        (lossk, auxk), gradk = jax.value_and_grad(fnk, has_aux=True)(agent.actor.params)

        def mean_width(p):
            action = agent.actor.apply_fn(p, fixed_batch.obs).sample(seed=None)
            return synthetic_width(reference, fixed_batch.obs, action).mean()

        width_value = mean_width(agent.actor.params)
        width_grad = jax.grad(mean_width)(agent.actor.params)
        if norm(width_grad) <= 1e-6:
            raise AssertionError("synthetic width actor gradient is vacuous")
        action = agent.actor.apply_fn(agent.actor.params, fixed_batch.obs).sample(seed=None)
        raw_q1 = agent.dual_q.apply_fn(agent.dual_q.params, fixed_batch.obs, action)[..., 0]
        independent_c = args.td3_alpha / (jnp.abs(raw_q1).mean() + 1e-7)
        c = aux0["q_scale_coefficient"]
        compare(c, independent_c)
        compare(auxk["q_scale_coefficient"], c)
        delta = jax.tree_util.tree_map(lambda a, b: a - b, gradk, grad0)
        expected = jax.tree_util.tree_map(lambda g: c * kappa * g, width_grad)
        error = compare(delta, expected)
        loss_error = compare(lossk - loss0, c * kappa * width_value)
        return {"kappa_penalty": kappa, "q_scale_coefficient": scalar(c),
                "width_actor_gradient_norm": norm(width_grad),
                "actor_gradient_delta_norm": norm(delta), "identity_max_abs_error": error,
                "loss_identity_abs_error": loss_error,
                "identity": "grad(L_k)-grad(L_0)=q_scale_coefficient*kappa*grad(mean(U))",
                "reference": "frozen synthetic width; no BCA calibrator integrated"}

    for kappa in KAPPAS:
        record("actor_gradient_identity_kappa_" + format(kappa, "g"),
               lambda kappa=kappa: gradient_identity(kappa), category="gradient", requires=initial_ok)

    def frozen_reference():
        agent = context["initial"].agent_state
        fns = loss_functions(1.0)
        grad = jax.grad(lambda ref: fns.actor_loss(
            agent.actor.params, agent, fixed_batch, ref)[0])(reference)
        for leaf in jax.tree_util.tree_leaves(grad):
            array = np.asarray(leaf)
            np.testing.assert_array_equal(array, np.zeros_like(array))
        actions = agent.actor.apply_fn(agent.actor.params, fixed_batch.obs).sample(seed=None)
        unblocked = jax.grad(lambda ref: synthetic_width(ref, fixed_batch.obs, actions).mean())(reference)
        if norm(unblocked) <= 1e-6:
            raise AssertionError("reference blocking fixture is vacuous")
        return {"actual_actor_loss_reference_gradient_norm": norm(grad),
                "unblocked_fixture_reference_gradient_norm": norm(unblocked),
                "assertion": "actual candidate blocks the complete reference tree exactly"}

    record("synthetic_reference_gradient_is_frozen", frozen_reference,
           category="gradient", requires=initial_ok)

    def live_action():
        agent = context["initial"].agent_state
        def probe_actor(p, obs):
            return SimpleNamespace(sample=lambda seed=None: p)
        f0, f1 = loss_functions(0.0, probe_actor), loss_functions(1.0, probe_actor)
        action = jnp.asarray([-0.5, -0.2, 0.1, 0.4], jnp.float32)
        g0 = jax.grad(lambda a: f0.actor_loss(a, agent, fixed_batch, reference)[0])(action)
        g1 = jax.grad(lambda a: f1.actor_loss(a, agent, fixed_batch, reference)[0])(action)
        c = f0.actor_loss(action, agent, fixed_batch, reference)[1]["q_scale_coefficient"]
        gu = jax.grad(lambda a: synthetic_width(
            reference, fixed_batch.obs, jnp.broadcast_to(a, fixed_batch.action.shape)).mean())(action)
        if norm(gu) <= 1e-6 or norm(g1 - g0) <= 1e-6:
            raise AssertionError("actual candidate lost the live action gradient")
        return {"action_gradient_delta_norm": norm(g1 - g0), "width_action_gradient_norm": norm(gu),
                "identity_max_abs_error": compare(g1 - g0, c * gu),
                "probe": "one shared 4D action vector across the fixed 8-row bank"}

    record("actual_candidate_preserves_live_width_action_gradient", live_action,
           category="gradient", requires=initial_ok)

    def zero_kappa_skips_width():
        agent = context["initial"].agent_state
        def forbidden_width(*unused):
            raise AssertionError("zero-kappa branch called width fixture")
        fns = loss_functions(0.0, width_fn=forbidden_width)
        expected = loss_functions(0.0).actor_loss(agent.actor.params, agent, fixed_batch, reference)
        actual = fns.actor_loss(agent.actor.params, agent, fixed_batch, reference)
        compare(actual, expected)
        return {"width_callback_called": False, "scope": "actual zero-kappa actor-loss branch"}

    record("zero_kappa_bypasses_synthetic_width_callback", zero_kappa_skips_width,
           category="gradient", requires=initial_ok)

    if len(checks) != EXPECTED_CHECK_COUNT:
        raise AssertionError("declared check count differs from registrations")
    return {"schema": "unifloral_td3bc_validation.parity.v1",
            "passed": all(item["status"] == "pass" for item in checks),
            "checks": checks, "check_count": len(checks),
            "update_counts": {**ledger, "maximum_critic": MAX_CRITIC_UPDATES,
                              "maximum_actor": MAX_ACTOR_UPDATES,
                              "maximum_optimizer_total": MAX_CRITIC_UPDATES + MAX_ACTOR_UPDATES,
                              "environment_rollouts": 0,
                              "accounting": "verified deltas read from returned TrainState step counters; failed dispatch may leave attempted above verified"},
            "effective_args": asdict(args), "fixture_seed": fixture_seed,
            "fixture_sha256": fingerprint(dataset), "synthetic_reference_sha256": fingerprint(reference),
            "fixture": {"rows": 32, "observation_dimensions": 14, "action_dimensions": 4,
                        "batch_size": 8, "terminal_rows": int(done.sum()),
                        "nonterminal_rows": int((1 - done).sum()), "is_environment_dataset": False},
            "source": getattr(native, "_source_receipt", None),
            "tolerances": {"float_rtol": RTOL, "float_atol": ATOL,
                           "rng_counters_batch_reuse_reference_blocking": "exact"},
            "precision": str(jax.config.jax_default_matmul_precision),
            "limitations": ["No real BCA residual fitting or calibration integrated.",
                            "No efficacy, reward-scale robustness, full-budget readiness, or ThrusterChain task claim.",
                            "Floating-state parity is tolerance-based, not a bitwise floating-point claim."]}
