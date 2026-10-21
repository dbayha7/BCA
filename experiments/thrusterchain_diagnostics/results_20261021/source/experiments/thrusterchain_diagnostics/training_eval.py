"""Frozen, independent finite-start evaluation for the continuous training core.

Every policy consumes batches of raw 15-coordinate observations and returns
continuous commands. Learned wrappers perform their own training-only transform.
The plant runs float64 after an explicit float32 deployment boundary. Known-start
deterministic returns/Q values are simulated exactly to the finite horizon; bank
averages are empirical estimates, never exact continuous population expectations.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .continuous import KNEE, Objective, FiniteHorizonQP, dynamics, fixed_cap_policy, regime_cap_policy
from .training_data import array_hash


EVALUATION_NAMESPACE = 2026102002
SPLIT_SEEDS = {"validation": 72001, "final": 82001, "query": 92001}
CONTEXT_NAMES = ("stall_low", "stall_high", "monotone_low", "monotone_high")
FAILURE_POSITION_LIMIT = 3.0


@dataclass
class StartBank:
    states: np.ndarray
    regimes: np.ndarray
    contexts: np.ndarray
    times: np.ndarray
    metadata: dict

    @property
    def objective(self):
        obj = self.metadata["objective"]
        return Objective(horizon=obj["horizon"], gamma=obj["gamma"])

    @property
    def arrays(self):
        return dict(states=self.states, regimes=self.regimes, contexts=self.contexts, times=self.times)


def make_start_bank(*, split="final", per_context=None, seed=None, objective=Objective()):
    if split not in SPLIT_SEEDS:
        raise ValueError("unknown evaluation split")
    per_context = ({"validation": 20, "final": 100, "query": 4}[split]
                   if per_context is None else per_context)
    if isinstance(per_context, bool) or not isinstance(per_context, int) or per_context < 1:
        raise ValueError("per_context must be a positive integer")
    seed = SPLIT_SEEDS[split] if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    stream = [seed, EVALUATION_NAMESPACE, list(SPLIT_SEEDS).index(split)]
    rng = np.random.default_rng(np.random.SeedSequence(stream))
    contexts = np.repeat(np.arange(4, dtype=np.int64), per_context)
    states = np.empty((len(contexts), 13))
    states[:, :10] = rng.uniform(-.5, .5, (len(contexts), 10))
    states[:, 10] = np.where(contexts % 2 == 0, .3, 1.1)
    phase = rng.uniform(-np.pi, np.pi, len(contexts))
    states[:, 11:] = .2 * np.column_stack((np.cos(phase), np.sin(phase)))
    arrays = dict(states=states, regimes=contexts // 2, contexts=contexts,
                  times=np.zeros(len(contexts), dtype=np.int64))
    metadata = dict(schema="thrusterchain-training-start-bank-v1", split=split, seed=seed,
        seed_sequence=stream, per_context=per_context, rows=len(contexts),
        objective=objective.as_dict(), context_order=list(CONTEXT_NAMES),
        start_law="independent U[-.5,.5]^10; fixed current .3/1.1; wave amplitude .2, uniform phase",
        purpose="evaluation only; never normalization, fitting, calibration, selection or restart",
        failure_rule="any abs(position state[0:10:2])>3 at pre/post state; event only, episode continues to H",
        deployment="float32 observations and commands; clipped [-1,1]; float64 actual continuous.dynamics",
        averaging="equal context empirical mean over frozen independent starts; not analytic population expectation",
        arrays_sha256=array_hash(arrays))
    return StartBank(**arrays, metadata=metadata)


def make_evaluation_banks(objective=Objective()):
    return {split: make_start_bank(split=split, objective=objective) for split in ("validation", "final")}


def make_query_bank(objective=Objective(), *, per_context=4):
    """Independent start-law physical states at times 0, H//2 and H-1.

    This is an explicitly designed query law, not on-policy visitation. The
    physical state is repeated across times to separate remaining-horizon effects.
    """
    base = make_start_bank(split="query", per_context=per_context, objective=objective)
    times = np.unique([0, objective.horizon // 2, objective.horizon - 1])
    arrays = dict(states=np.repeat(base.states, len(times), axis=0),
                  regimes=np.repeat(base.regimes, len(times)),
                  contexts=np.repeat(base.contexts, len(times)), times=np.tile(times, len(base.states)))
    metadata = dict(base.metadata, schema="thrusterchain-training-query-bank-v1", rows=len(arrays["states"]),
                    query_law="start-law physical states crossed with fixed times; not policy occupancy",
                    query_times=times.tolist(), arrays_sha256=array_hash(arrays))
    return StartBank(**arrays, metadata=metadata)


def save_bank(bank, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "bank.npz", **bank.arrays)
    metadata = dict(bank.metadata, file="bank.npz",
                    file_sha256=hashlib.sha256((directory / "bank.npz").read_bytes()).hexdigest())
    (directory / "bank.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_bank(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "bank.json").read_text(encoding="utf-8"))
    path = directory / metadata["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["file_sha256"]:
        raise ValueError("evaluation bank file identity differs")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if array_hash(arrays) != metadata["arrays_sha256"]:
        raise ValueError("evaluation bank array identity differs")
    return StartBank(**arrays, metadata=metadata)


def _observation_batch(states, regimes, times, objective, float32_deployment):
    obs = np.column_stack((states, regimes, (objective.horizon - np.asarray(times)) / objective.horizon))
    return obs.astype(np.float32 if float32_deployment else np.float64)


def _deploy_actions(actions, n, float32_deployment):
    actions = np.asarray(actions)
    if actions.shape != (n, 4):
        raise ValueError("batch policy must return shape [N,4]")
    if float32_deployment:
        actions = actions.astype(np.float32)
    actions = actions.astype(np.float64)
    # Clipping +/-inf would conceal a failed policy as an admissible command.
    return np.where(np.isfinite(actions), np.clip(actions, -1., 1.), np.nan)


def batched_control(kind="fixed_cap", objective=Objective()):
    """Vectorized, untrained named controls with the same observation interface."""
    if kind not in ("fixed_cap", "regime_cap"):
        raise ValueError("unknown control")
    controller = fixed_cap_policy(KNEE, objective) if kind == "fixed_cap" else regime_cap_policy(objective)
    def policy(observations):
        observations = np.asarray(observations)
        times = np.rint(objective.horizon * (1 - observations[:, -1])).astype(int)
        action = -np.einsum("nad,nd->na", controller.gains[times], observations[:, :13])
        caps = np.full(len(action), KNEE) if kind == "fixed_cap" else np.where(observations[:, 13] == 0, KNEE, 1.)
        return np.clip(action, -caps[:, None], caps[:, None])
    return policy


def _simulate(policy, states, regimes, times, objective, float32_deployment, forced_actions=None):
    states, regimes, times = np.asarray(states, np.float64).copy(), np.asarray(regimes), np.asarray(times)
    n = len(states)
    if (states.shape != (n, 13) or regimes.shape != (n,) or times.shape != (n,)
            or not np.isfinite(states).all() or not np.isin(regimes, (0, 1)).all()
            or not np.issubdtype(times.dtype, np.integer) or np.any((times < 0) | (times > objective.horizon))):
        raise ValueError("invalid known-state rollout bank")
    finite, totals = np.ones(n, bool), np.zeros(n)
    max_position = np.max(np.abs(states[:, :10:2]), axis=1)
    max_action, high_steps, critical_steps, steps = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, int)
    if forced_actions is not None:
        forced_actions = _deploy_actions(forced_actions, n, float32_deployment)
        if np.any(times == objective.horizon):
            raise ValueError("forced Q action is undefined at terminal time")
    for t in range(int(times.min(initial=objective.horizon)), objective.horizon):
        selected = np.flatnonzero(finite & (times <= t))
        if not len(selected):
            continue
        observations = _observation_batch(states[selected], regimes[selected], np.full(len(selected), t),
                                          objective, float32_deployment)
        if forced_actions is None:
            actions = _deploy_actions(policy(observations), len(selected), float32_deployment)
        else:
            is_first = times[selected] == t
            actions = np.empty((len(selected), 4))
            actions[is_first] = forced_actions[selected[is_first]]
            if np.any(~is_first):
                actions[~is_first] = _deploy_actions(policy(observations[~is_first]), int(np.sum(~is_first)), float32_deployment)
        good = np.isfinite(actions).all(axis=1)
        finite[selected[~good]] = False
        selected, actions = selected[good], actions[good]
        if not len(selected):
            continue
        max_action[selected] = np.maximum(max_action[selected], np.max(np.abs(actions), axis=1))
        high = np.any(np.abs(actions) > KNEE, axis=1)
        high_steps[selected] += high
        critical_steps[selected] += high & (states[selected, 10] >= .8)
        nxt, reward = np.empty((len(selected), 13)), np.empty(len(selected))
        for regime in (0, 1):
            mask = regimes[selected] == regime
            if np.any(mask):
                nxt[mask], reward[mask] = dynamics(states[selected[mask]], actions[mask], regime)
        good = np.isfinite(nxt).all(axis=1) & np.isfinite(reward)
        finite[selected[~good]] = False
        selected, nxt, reward = selected[good], nxt[good], reward[good]
        totals[selected] += objective.gamma ** (t - times[selected]) * reward
        states[selected] = nxt
        max_position[selected] = np.maximum(max_position[selected], np.max(np.abs(nxt[:, :10:2]), axis=1))
        steps[selected] += 1
    totals[~finite] = np.nan
    completed = finite & (steps == objective.horizon - times)
    return dict(return_value=totals, max_abs_position=max_position, max_abs_action=max_action,
                high_action_fraction=np.divide(high_steps, steps, out=np.zeros(n), where=steps > 0),
                critical_fraction=np.divide(critical_steps, steps, out=np.zeros(n), where=steps > 0),
                high_action_steps=high_steps.astype(int), critical_steps=critical_steps.astype(int),
                steps=steps, finite=finite, completed=completed,
                failure=(max_position > FAILURE_POSITION_LIMIT) | ~completed)


def batch_rollout(policy, bank, *, float32_deployment=True):
    return _simulate(policy, bank.states, bank.regimes, bank.times, bank.objective, float32_deployment)


def batch_policy_q(policy, states, regimes, times, actions, *, objective=Objective(), float32_deployment=True):
    """One forced command and exact deterministic continuation until H."""
    return _simulate(policy, states, regimes, times, objective, float32_deployment,
                     forced_actions=actions)["return_value"]


def _finite_summary(values):
    values = np.asarray(values)
    return float(np.mean(values)) if np.isfinite(values).all() else None


def evaluate_policy(policy, bank, *, critic=None, optimum=None, query_bank=None, float32_deployment=True):
    """Return JSON-compatible summary and raw arrays; never selects a checkpoint."""
    arrays = batch_rollout(policy, bank, float32_deployment=float32_deployment)
    arrays.update(context=bank.contexts.copy(), regime=bank.regimes.copy(), start_time=bank.times.copy())
    if optimum is not None:
        if optimum.get("bank_sha256") != bank.metadata["arrays_sha256"]:
            raise ValueError("oracle and policy start banks differ")
        lower, upper = np.asarray(optimum["return_lower"]), np.asarray(optimum["return_upper"])
        if lower.shape != arrays["return_value"].shape or upper.shape != lower.shape:
            raise ValueError("oracle row count differs")
        arrays.update(regret_lower=lower - arrays["return_value"], regret_upper=upper - arrays["return_value"])
    def summarize(mask):
        return dict(starts=int(np.sum(mask)), mean_return=_finite_summary(arrays["return_value"][mask]),
                    return_std=float(np.std(arrays["return_value"][mask], ddof=1)) if np.sum(mask) > 1 and np.isfinite(arrays["return_value"][mask]).all() else None,
                    failure_rate=float(np.mean(arrays["failure"][mask])),
                    completed=int(np.sum(arrays["completed"][mask])),
                    high_action_fraction=float(np.mean(arrays["high_action_fraction"][mask])),
                    critical_fraction=float(np.mean(arrays["critical_fraction"][mask])),
                    **({key: _finite_summary(arrays[key][mask]) for key in ("regret_lower", "regret_upper")} if optimum is not None else {}))
    summary = dict(schema="thrusterchain-training-policy-evaluation-v1", split=bank.metadata["split"],
        bank_sha256=bank.metadata["arrays_sha256"], objective=bank.objective.as_dict(),
        deployment=bank.metadata["deployment"] if float32_deployment else "float64 observations/actions/plant",
        failure_rule=bank.metadata["failure_rule"], pooled=summarize(np.ones(len(bank.states), bool)),
        strata={name: summarize(bank.contexts == k) for k, name in enumerate(CONTEXT_NAMES)},
        continuous_optimum_status="numerical feasible/lower and Frank-Wolfe upper return bounds" if optimum is not None else "not computed",
        inference="finite shared bank sample estimates; replication unit is trained seed, not rollout",
        selection="none; terminal checkpoint primary, midpoint descriptive")
    if critic is not None:
        if query_bank is None:
            raise ValueError("critic evaluation requires an explicit frozen independent query bank")
        try:
            q_summary, q_arrays = evaluate_critic(policy, critic, query_bank, float32_deployment=float32_deployment)
        except Exception as error:
            # Preserve the independently completed return evaluation and expose
            # failed diagnostics; a missing Q result is never a passing result.
            summary["critic_diagnostics"] = dict(status="failed", failure=type(error).__name__ + ": " + str(error))
        else:
            summary["critic_diagnostics"] = dict(status="completed", **q_summary)
            arrays.update({"q_" + key: value for key, value in q_arrays.items()})
    return summary, arrays


def evaluate_critic(policy, critic, bank, *, float32_deployment=True):
    """Critic error/ranking and Bellman residual are distinct measured targets."""
    objective = bank.objective
    obs = _observation_batch(bank.states, bank.regimes, bank.times, objective, float32_deployment)
    actor_actions = _deploy_actions(policy(obs), len(obs), float32_deployment)
    fixed = np.array([np.zeros(4), np.full(4, -KNEE), np.full(4, KNEE), np.full(4, -1.), np.full(4, 1.)])
    actions = np.concatenate((np.broadcast_to(fixed, (len(obs), 5, 4)), actor_actions[:, None, :]), axis=1)
    actions = _deploy_actions(actions.reshape(-1, 4), len(obs) * 6, float32_deployment)
    states, regimes, times = np.repeat(bank.states, 6, axis=0), np.repeat(bank.regimes, 6), np.repeat(bank.times, 6)
    repeated_obs = np.repeat(obs, 6, axis=0)
    truth = batch_policy_q(policy, states, regimes, times, actions, objective=objective,
                           float32_deployment=float32_deployment)
    predictions = np.asarray(critic(repeated_obs, actions), np.float64)
    if predictions.shape != (len(truth), 2) or not np.isfinite(predictions).all() or not np.isfinite(truth).all():
        raise ValueError("critic must return finite [N,2] twin predictions and policy Q must be finite")
    minimum = predictions.min(axis=1)
    # Bellman residual uses the same deployed continuation but learned min-Q.
    nxt, rewards = np.empty_like(states), np.empty(len(states))
    for regime in (0, 1):
        selected = regimes == regime
        nxt[selected], rewards[selected] = dynamics(states[selected], actions[selected], regime)
    next_obs = _observation_batch(nxt, regimes, times + 1, objective, float32_deployment)
    continuation = np.zeros(len(states))
    active = times + 1 < objective.horizon
    if np.any(active):
        next_actions = _deploy_actions(policy(next_obs[active]), int(np.sum(active)), float32_deployment)
        next_q = np.asarray(critic(next_obs[active], next_actions), np.float64)
        if next_q.shape != (int(np.sum(active)), 2) or not np.isfinite(next_q).all():
            raise ValueError("invalid continuation critic predictions")
        continuation[active] = next_q.min(axis=1)
    residual = rewards + objective.gamma * continuation - minimum
    ranked_truth, ranked_prediction = truth.reshape(-1, 6), minimum.reshape(-1, 6)
    chosen = np.argmax(ranked_prediction, axis=1)
    ranking_regret = ranked_truth.max(axis=1) - ranked_truth[np.arange(len(obs)), chosen]
    error = np.column_stack((predictions, minimum)) - truth[:, None]
    scores = {name: dict(rmse=float(np.sqrt(np.mean(error[:, i] ** 2))),
                        mean_error=float(np.mean(error[:, i])), mae=float(np.mean(np.abs(error[:, i]))))
              for i, name in enumerate(("critic_1", "critic_2", "twin_min"))}
    pair_i, pair_j = np.triu_indices(6, 1)
    true_sign = np.sign(ranked_truth[:, pair_i] - ranked_truth[:, pair_j])
    pred_sign = np.sign(ranked_prediction[:, pair_i] - ranked_prediction[:, pair_j])
    non_ties = np.abs(ranked_truth[:, pair_i] - ranked_truth[:, pair_j]) > 1e-9
    accuracy = float(np.mean(pred_sign[non_ties] == true_sign[non_ties])) if np.any(non_ties) else None
    summary = dict(bank_sha256=bank.metadata["arrays_sha256"], query_states=len(obs), candidates=6,
        candidate_names=["zero", "all_negative_knee", "all_positive_knee", "all_negative_one", "all_positive_one", "deployed_actor"],
        query_law=bank.metadata.get("query_law", "specified fixed start bank"),
        target="finite Q_t^pi: forced command, deterministic deployed policy thereafter, no terminal bootstrap",
        q_error=scores, action_ranking=dict(mean_true_regret_of_predicted_best=float(np.mean(ranking_regret)),
                                          predicted_best_tie_rule="first candidate index (numpy.argmax)",
                                          pairwise_accuracy_excluding_true_ties=accuracy),
        bellman_residual=dict(mae=float(np.mean(np.abs(residual))), rmse=float(np.sqrt(np.mean(residual ** 2)))),
        coverage="point Q error and one-step Bellman residual are separate; no residual or long-horizon interval coverage certificate asserted")
    arrays = dict(truth=truth.reshape(-1, 6), predictions=predictions.reshape(-1, 6, 2),
                  bellman_residual=residual.reshape(-1, 6), ranking_regret=ranking_regret,
                  actions=actions.reshape(-1, 6, 4), context=bank.contexts.copy(), time=bank.times.copy())
    return summary, arrays


def solve_start_optima(bank, *, progress=None):
    """Optional expensive continuous QPs: solve once per shared bank, not per job.

    The stall dominance argument and convex certificate come from continuous.py.
    Bounds are floating point certificates, not outward-rounded interval proofs.
    """
    results = []
    problems = {}
    for index, (state, regime, t) in enumerate(zip(bank.states, bank.regimes, bank.times)):
        problem = problems.setdefault(int(t), None)
        if problem is None:
            problems[int(t)] = problem = FiniteHorizonQP(bank.objective, t=int(t))
        solution = problem.solve(state, int(regime))
        results.append(solution.as_dict())
        if progress is not None:
            progress(dict(completed=index + 1, total=len(bank.states), solution=results[-1]))
    return dict(schema="thrusterchain-shared-continuous-optima-v1", bank_sha256=bank.metadata["arrays_sha256"],
                return_lower=[row["return_interval"][0] for row in results],
                return_upper=[row["return_interval"][1] for row in results], solutions=results,
                scope="fixed shared starts only; not exact continuous population optimum")


def save_evaluation(summary, arrays, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "per_start.npz", **arrays)
    summary = dict(summary, arrays_sha256=array_hash(arrays), file="per_start.npz",
                   file_sha256=hashlib.sha256((directory / "per_start.npz").read_bytes()).hexdigest())
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary
