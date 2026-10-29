"""Small analysis-only diagnostic. NumPy/JAX imports follow runtime configuration."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SEED = 2026102122
METHODS = ("raw_q", "live_width", "matched_distance", "constant_offset", "detached_width")
CONTEXTS = ("stall_low", "stall_high", "monotone_low", "monotone_high")
EXPECTED_CHECKPOINT = "4088278230fc413c509e27e189daa342468cfc4fb6d2479e3e8c29e149dac3d8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def rng(stream):
    import numpy as np
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, stream])))


def load_engine():
    """Load actual engine first, then its authenticated historical module closure."""
    path = WORKSPACE / "action_only_20261021/engine.py"
    original = json.loads((WORKSPACE / "action_only_20261021/freeze.json").read_text())
    require(digest(path) == original["source_sha256"]["engine.py"], "existing frozen engine source changed")
    spec = importlib.util.spec_from_file_location("direct_q_frozen_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    runtime = module.configure_runtime((8, 9, 10, 11))
    engine = module.FrozenScoreEngine.load(bundle_root=original["bundle_root"],
        canonical_run_dir=original["canonical_run_dir"], state_seed=2026102101)
    require(engine.metadata["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT, "checkpoint identity differs")
    require(engine.settings.miscoverage == .1 and engine.settings.credibility == .95
            and engine.result.record["identity"]["data"]["calibration_rows"] == 128
            and engine.reference_update == 95000, "frozen calibration specification differs")
    from experiments.thrusterchain_diagnostics import training_eval
    relative = Path(training_eval.__file__).resolve().relative_to(Path(engine.metadata["source_closure"])).as_posix()
    require(digest(training_eval.__file__) == engine.metadata["source_sha256"][relative], "rollout module escaped verified closure")
    engine.eval_module = training_eval
    engine.direct_runtime = runtime
    engine.direct_policy = lambda observations: single_row_policy(engine, observations)
    return engine


def single_row_policy(engine, observations):
    """V2 deployment contract: unchanged native float32 actor, one row per call.

    Batch1 is used regardless of candidate group size; no JIT/precision change.
    This intentionally differs from V1's batch-size-dependent numerical policy.
    """
    import numpy as np
    observations = np.asarray(observations, np.float32)
    require(observations.ndim == 2 and observations.shape[1] == 15
            and np.isfinite(observations).all(), "policy requires finite [N,15]")
    if not len(observations):
        return np.empty((0, 4), np.float32)
    return np.concatenate([engine.result.policy(observations[i:i+1])
                           for i in range(len(observations))], axis=0)


def parameter_hashes(engine):
    state = engine.result.state
    values = {name: engine.T.tree_sha256(getattr(state.native, name).params)
              for name in ("actor", "critic", "actor_target", "critic_target")}
    values.update(carry=engine.T.tree_sha256(engine.result.carry),
                  live_calibrator=engine.T.tree_sha256(state.calibrator),
                  frozen_reference=engine.T.tree_sha256(state.posterior),
                  frozen_calibrator=engine.T.tree_sha256(state.posterior.cal_params))
    return values


def generate_states(per_context, stream):
    import numpy as np
    require(per_context in (2, 8) and stream in (1, 3), "only frozen evaluation/adaptation state sizes permitted")
    random = rng(stream)
    context = np.repeat(np.arange(4, dtype=np.int64), per_context)
    time = np.tile(np.resize(np.array([0, 50, 99], dtype=np.int64), per_context), 4)
    states = np.empty((len(context), 13), np.float64)
    states[:, :10] = random.uniform(-.5, .5, (len(context), 10))
    states[:, 10] = np.where(context % 2 == 0, .3, 1.1)
    phase = random.uniform(-np.pi, np.pi, len(context))
    states[:, 11:] = .2 * np.column_stack((np.cos(phase), np.sin(phase)))
    regimes = context // 2
    observation = np.column_stack((states, regimes, (100-time)/100)).astype(np.float32)
    return dict(states=states, regimes=regimes, context=context, time=time, observation=observation)


def classify_actions(actions, mode, atoms):
    import numpy as np
    actions, mode, atoms = np.asarray(actions, np.float32), np.asarray(mode, np.float32), np.asarray(atoms, np.float32)
    equal = actions[..., None] == atoms
    supported = np.all(np.any(equal, axis=-1), axis=-1)
    indices = np.argmax(equal, axis=-1)
    grid_id = np.sum(indices * np.array([125, 25, 5, 1]), axis=-1)
    beta = np.prod(np.where(actions == mode[:, None, :], .84, .04), axis=-1)
    beta = np.where(supported, beta, 0.)
    return dict(supported=supported, grid_id=np.where(supported, grid_id, -1), beta=beta,
                support_class=np.where(beta >= .01, "common_on_grid", np.where(beta > 0, "rare_on_grid", "unsupported")))


def build_candidates(observations, actor_actions, behavior_mode, atoms, random):
    import numpy as np
    atoms = np.asarray(atoms, np.float32)
    actor = np.asarray(actor_actions, np.float32)
    require(np.isfinite(actor).all(), "nonfinite actor action before clipping")
    actor = np.clip(actor, -1, 1)
    mode = np.asarray(behavior_mode, np.float32)
    require(actor.shape == mode.shape == (len(observations), 4) and np.isfinite(actor).all(), "invalid actor/mode actions")
    grid_indices = np.indices((5, 5, 5, 5)).reshape(4, -1).T
    grid = atoms[grid_indices]
    multipliers = np.array([125, 25, 5, 1])
    projected = np.argmin(np.abs(actor[..., None].astype(float) - atoms.astype(float)), axis=-1) @ multipliers
    mode_ids = np.argmin(np.abs(mode[..., None].astype(float) - atoms.astype(float)), axis=-1) @ multipliers
    actions = np.empty((len(actor), 12, 4), np.float32)
    for i in range(len(actor)):
        ids = list(dict.fromkeys((int(projected[i]), int(mode_ids[i]))))
        while len(ids) < 6:
            proposed = int(random.integers(0, 625))
            if proposed not in ids:
                ids.append(proposed)
        actions[i, :6] = grid[ids]
    actions[:, 6] = actor
    actions[:, 7:] = random.uniform(-1., 1., (len(actor), 5, 4)).astype(np.float32)
    return dict(actions=actions, behavior_mode=mode, actor_actions=actor,
                distance=np.mean((actions.astype(float)-mode[:, None, :].astype(float))**2, axis=-1),
                **classify_actions(actions, mode, atoms))


def read_width_and_q(engine, bank):
    import numpy as np
    import jax.numpy as jnp
    from calibration.dose import frozen_level_dose
    obs = np.repeat(bank["observation"], 12, axis=0)
    actions = bank["actions"].reshape(-1, 4)
    heads = np.asarray(engine.result.critic(obs, actions), np.float64)
    normalized = (obs-engine.result.mean)/engine.result.std
    eta_jax = engine.result.models[2].apply(engine.result.state.posterior.cal_params,
                                           jnp.asarray(normalized), jnp.asarray(actions))
    reference = engine.result.state.posterior
    native = frozen_level_dose(reference, eta_jax, .5)
    eta = np.asarray(eta_jax, np.float64)
    width = np.asarray(native.width, np.float64)
    formula = float(reference.threshold) * float(reference.residual_scale) * np.maximum(eta, 1e-6)
    require(np.allclose(width, formula, atol=2e-6, rtol=2e-6, equal_nan=True), "native width and R*unit*max(eta,1e-6) disagree")
    usable = np.asarray(native.support_mask, bool) & bool(reference.ready) & bool(native.inputs_valid)
    require(heads.shape == (len(obs), 2) and np.isfinite(heads).all(), "nonfinite/invalid current critic heads")
    shape = (len(bank["states"]), 12)
    return dict(q1=heads[:, 0].reshape(shape), q2=heads[:, 1].reshape(shape), eta=eta.reshape(shape),
                width=width.reshape(shape), width_formula=formula.reshape(shape), width_usable=usable.reshape(shape),
                width_refusal=(~usable | ~np.isfinite(width)).reshape(shape))


def matching_coefficient(width, distance, usable):
    import numpy as np
    width, distance, usable = np.asarray(width), np.asarray(distance), np.asarray(usable)
    if not (width.shape == distance.shape == usable.shape and width.ndim == 2 and np.isfinite(width).all() and usable.all()):
        return dict(status="refused", coefficient=None, reason="unusable/nonfinite adaptation widths")
    u = float(np.sqrt(np.mean((width-width.mean(axis=1, keepdims=True))**2)))
    d = float(np.sqrt(np.mean((distance-distance.mean(axis=1, keepdims=True))**2)))
    if not np.isfinite(d) or d <= 0 or not np.isfinite(u):
        return dict(status="refused", coefficient=None, reason="zero/nonfinite centered-distance RMS")
    return dict(status="matched_without_returns", coefficient=u/d, width_centered_rms=u, distance_centered_rms=d,
                observations=8, candidates_per_observation=12, truth_labels_used=0)


def score_arrays(bank, coefficient):
    import numpy as np
    raw = bank["q1"].astype(float)
    usable = bank["width_usable"] & np.isfinite(bank["width"])
    penalized = raw - np.where(usable, bank["width"], 0.)
    count = usable.sum(axis=1, keepdims=True)
    offset = np.divide(np.where(usable, bank["width"], 0.).sum(axis=1, keepdims=True), count,
                       out=np.zeros_like(count, dtype=float), where=count > 0)
    matched = raw-coefficient["coefficient"]*bank["distance"] if coefficient["coefficient"] is not None else np.full_like(raw, np.nan)
    result = dict(raw_q=raw, live_width=penalized, matched_distance=matched,
                  constant_offset=raw-offset, detached_width=penalized.copy())
    # A state-constant subtraction is an algebraic ranking control. Detect any
    # floating-point rank perturbation explicitly rather than favoring the oracle.
    require(np.array_equal(np.argmax(raw, axis=1), np.argmax(result["constant_offset"], axis=1)), "constant-offset rounding perturbed ranking")
    return result


def bank_inputs(engine, per_context, state_stream, action_stream):
    bank = generate_states(per_context, state_stream)
    indices, _ = engine.D.behavior_parameters(bank["observation"], engine.C.Objective())
    mode = engine.atoms[indices].astype("float32")
    candidates = build_candidates(bank["observation"], engine.direct_policy(bank["observation"]), mode, engine.atoms, rng(action_stream))
    bank.update(candidates)
    bank.update(read_width_and_q(engine, bank))
    return bank


def truth_returns(engine, bank):
    import numpy as np
    states = np.repeat(bank["states"], 12, axis=0)
    regimes, times = np.repeat(bank["regimes"], 12), np.repeat(bank["time"], 12)
    truth = engine.eval_module.batch_policy_q(engine.direct_policy, states, regimes, times,
        bank["actions"].reshape(-1, 4), objective=engine.C.Objective(), float32_deployment=True)
    return truth.reshape(len(bank["states"]), 12)


def truth_rollouts(engine, bank):
    """Same authenticated kernel as batch_policy_q, retaining failure metadata."""
    import numpy as np
    arrays = engine.eval_module._simulate(engine.direct_policy, np.repeat(bank["states"], 12, axis=0),
        np.repeat(bank["regimes"], 12), np.repeat(bank["time"], 12), engine.C.Objective(), True,
        forced_actions=bank["actions"].reshape(-1, 4))
    return {key: value.reshape((len(bank["states"]), 12) + value.shape[1:]) for key, value in arrays.items()}


def validate_truth(engine, progress=None):
    """Four deterministic fixtures; total608 plant transitions, no data selection."""
    import numpy as np
    c = engine.C
    states = np.zeros((4, 13)); states[:, 10] = [.3, 1.1, .3, 1.1]; states[:, 11] = .2
    regimes, times = np.array([0, 0, 1, 1]), np.array([0, 50, 99, 99])
    actions = np.array([[.3, -.4, .8, -.2], [1, -1, .5, -.5], [0, 0, 0, 0], [1, -.8, .2, -.4]], np.float32)
    completed = 0
    def phase_done(phase):
        nonlocal completed
        completed += 152
        if progress is not None:
            progress(phase, completed)
    scalar_policy = lambda obs: engine.direct_policy(obs[None, :])[0]
    batched = engine.eval_module.batch_policy_q(engine.direct_policy, states, regimes, times, actions,
                                               objective=c.Objective(), float32_deployment=True)
    phase_done("batch_forced_q")
    scalar = np.array([c.policy_q(s, int(g), a, scalar_policy, c.Objective(), t=int(t), float32_deployment=True)
                       for s, g, t, a in zip(states, regimes, times, actions)])
    phase_done("scalar_forced_q")
    np.testing.assert_allclose(batched, scalar, atol=1e-4, rtol=1e-6)
    obs = np.column_stack((states, regimes, (100-times)/100)).astype(np.float32)
    policy_action = engine.direct_policy(obs)
    action_q = engine.eval_module.batch_policy_q(engine.direct_policy, states, regimes, times, policy_action,
                                                objective=c.Objective(), float32_deployment=True)
    phase_done("batch_policy_action_q")
    values = np.array([c.policy_value(s, int(g), scalar_policy, c.Objective(), t=int(t), float32_deployment=True)
                       for s, g, t in zip(states, regimes, times)])
    phase_done("scalar_policy_v")
    np.testing.assert_allclose(action_q, values, atol=1e-4, rtol=1e-6)
    terminal_errors = []
    for i in (2, 3):
        # Independent declared reward expression; no extra plant transition.
        immediate = -(np.sum(states[i, :10]**2) + .3*np.sum(np.clip(actions[i].astype(float), -1, 1)**2))
        terminal_errors.append(abs(float(batched[i]-immediate)))
        require(abs(batched[i]-immediate) <= 1e-10, "terminal99 must equal immediate reward within1e-10")
    return dict(status="passed", fixtures=4, plant_transitions=608, batch_scalar_max_error=float(np.max(np.abs(batched-scalar))),
                policy_action_q_equals_v_max_error=float(np.max(np.abs(action_q-values))), terminal99_atol=1e-10,
                terminal99_max_error=max(terminal_errors),
                replay_atol=1e-4, replay_rtol=1e-6)


def first_argmax(values, eligible=None):
    import numpy as np
    values = np.asarray(values)
    ids = np.arange(len(values)) if eligible is None else np.flatnonzero(eligible)
    require(len(ids) > 0 and np.isfinite(values[ids]).all(), "nonfinite/empty ranking")
    return int(ids[np.argmax(values[ids])])


def concordance(predicted, truth):
    import numpy as np
    i, j = np.triu_indices(len(truth), 1)
    td, pd = truth[i]-truth[j], predicted[i]-predicted[j]
    t, p = np.where(np.abs(td) <= 1e-6, 0, np.sign(td)), np.where(np.abs(pd) <= 1e-6, 0, np.sign(pd))
    valid = t != 0
    return dict(value=float(np.mean(np.where(p[valid] == 0, .5, p[valid] == t[valid]))) if valid.any() else None,
                true_ties=int(np.sum(~valid)), predicted_ties=int(np.sum(p == 0)), pairs=int(valid.sum()),
                predicted_ties_among_comparable=int(np.sum((p == 0) & valid)))


def bootstrap_interval(deltas, contexts, random):
    import numpy as np
    deltas, contexts = np.asarray(deltas), np.asarray(contexts)
    ids = [np.flatnonzero(contexts == c) for c in np.unique(contexts)]
    if not ids or any(len(ix) < 2 for ix in ids):
        return None
    draws = np.concatenate([random.choice(ix, size=(10000, len(ix)), replace=True) for ix in ids if len(ix)], axis=1)
    means = deltas[draws].mean(axis=1)
    return np.quantile(means, [.025, .975], method="linear").tolist()


def analyze(bank, truth, scores):
    import numpy as np
    from scipy.stats import spearmanr
    require(truth.shape == (32, 12) and np.isfinite(truth).all(), "all384 truth rows required; failed rows must not be dropped")
    contexts, times = bank["context"], bank["time"]
    masks = [("pooled", np.ones(32, bool))]
    masks += [(CONTEXTS[c], contexts == c) for c in range(4)]
    masks += [(f"regime_{g}", contexts//2 == g) for g in (0, 1)]
    masks += [(f"load_{load}", contexts % 2 == k) for k, load in enumerate((.3, 1.1))]
    masks += [(f"time_{t}", times == t) for t in (0, 50, 99)]
    groups, decisions, correlations = [], [], []
    random = rng(5)
    for group, states in masks:
        for support in ("all", "common_on_grid", "rare_on_grid", "unsupported"):
            eligible = np.ones((32, 12), bool) if support == "all" else bank["support_class"] == support
            state_ids = np.flatnonzero(states & (eligible.sum(axis=1) >= 2))
            record = dict(group=group, support=support, eligible_states=len(state_ids),
                          excluded_fewer_than_two_actions=int(np.sum(states & (eligible.sum(axis=1) < 2))),
                          bootstrap_context_time_counts={str(int(c)): int(np.sum(contexts[state_ids]*100+times[state_ids] == c))
                              for c in np.unique(contexts[state_ids]*100+times[state_ids])}, methods={})
            for method in METHODS:
                values, ranks, advantages, loss, cs = [], [], [], [], []
                if not np.isfinite(scores[method][state_ids][eligible[state_ids]]).all():
                    record["methods"][method] = dict(status="refused", reason="matching coefficient unavailable")
                    continue
                for i in state_ids:
                    ids = np.flatnonzero(eligible[i]); base = first_argmax(scores["raw_q"][i], eligible[i])
                    choice = first_argmax(scores[method][i], eligible[i])
                    delta = float(truth[i, choice]-truth[i, base])
                    opportunity = float(truth[i, ids].max()-truth[i, choice])
                    con = concordance(scores[method][i, ids], truth[i, ids])
                    values.append(delta); advantages.append(float(truth[i, choice]-truth[i, 6])); loss.append(opportunity); cs.append(con)
                    if con["value"] is not None: ranks.append(con["value"])
                    decisions.append(dict(group=group, support=support, state_id=int(i), method=method,
                        chosen_candidate=choice, raw_candidate=base, chosen_true_q=float(truth[i, choice]),
                        raw_true_q=float(truth[i, base]), delta_vs_raw=delta, advantage_vs_original_actor=advantages[-1],
                        opportunity_loss=opportunity, harm_vs_raw=max(-delta, 0.), width_refused_at_choice=bool(bank["width_refusal"][i, choice])))
                if not values:
                    record["methods"][method] = dict(status="unavailable", reason="no state has two eligible candidates")
                    continue
                values = np.asarray(values)
                record["methods"][method] = dict(status="evaluated", mean_delta_vs_raw=float(values.mean()),
                    paired_state_bootstrap_ci95=bootstrap_interval(values, contexts[state_ids]*100+times[state_ids], random),
                    improved_states=int(np.sum(values > 0)), harmed_states=int(np.sum(values < 0)), unchanged_states=int(np.sum(values == 0)),
                    mean_positive_harm=float(np.mean(np.maximum(-values, 0))), mean_opportunity_loss=float(np.mean(loss)),
                    mean_advantage_vs_original_actor=float(np.mean(advantages)),
                    mean_ranking_concordance=float(np.mean(ranks)) if ranks else None,
                    valid_concordance_states=len(ranks), comparable_pairs=sum(c["pairs"] for c in cs),
                    true_ties=sum(c["true_ties"] for c in cs), predicted_ties=sum(c["predicted_ties"] for c in cs),
                    predicted_ties_among_comparable=sum(c["predicted_ties_among_comparable"] for c in cs),
                    bootstrap_refusal_reason="fewer than two eligible states in a nonempty context-time stratum" if any(v < 2 for v in record["bootstrap_context_time_counts"].values()) else None)
            groups.append(record)
    for i in range(32):
        for support in ("all", "common_on_grid", "rare_on_grid", "unsupported"):
            ix = np.ones(12, bool) if support == "all" else bank["support_class"][i] == support
            width, error = bank["width"][i, ix], np.maximum(bank["q1"][i, ix]-truth[i, ix], 0.)
            valid = len(width) >= 2 and np.isfinite(width).all() and np.ptp(width) > 0 and np.ptp(error) > 0
            correlations.append(dict(state_id=i, context=int(contexts[i]), time=int(times[i]), support=support,
                candidates=int(ix.sum()), spearman=float(spearmanr(width, error).statistic) if valid else None,
                undefined_reason=None if valid else "insufficient, constant ranks, or nonfinite width"))
    return dict(groups=groups, within_state_width_positive_error_spearman=correlations), decisions


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader(); writer.writerows(rows)
