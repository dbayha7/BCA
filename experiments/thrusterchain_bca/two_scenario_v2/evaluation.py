"""Explicit true-terminal rollouts and own-policy Q references; no implicit simulator imports."""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rollout(task, starts, times, forward, *, first_actions=None, max_transitions):
    """Run the supplied task only when explicitly called by the approved evaluator.

    The caller wires its physical counter into task.plant_step. These counts are
    logical attempted/completed transitions, not a second charge to that meter.
    Exact state/time is passed to every step. Terminated rows never step again.
    """
    import numpy as np
    states = np.asarray(starts)
    times = np.asarray(times)
    require(states.dtype == np.float64 and states.ndim == 2 and states.shape[1] == 13 and
            np.isfinite(states).all(), "Exact finite float64[N,13] start states required")
    n = len(states)
    require(n > 0 and times.dtype.kind in "iu" and times.shape == (n,) and np.all((times >= 0) & (times < 100)),
            "Actual integer time in [0,100) required")
    require(type(max_transitions) is int and max_transitions >= 0, "An explicit transition cap is required")
    initial_times = times.copy()
    x, t = states.copy(), times.copy()
    done = np.zeros(n, bool)
    values = np.zeros(n, np.float64)
    attempts = completed = actor_calls = 0
    traces = [[] for _ in range(n)]
    if first_actions is not None:
        first_actions = np.asarray(first_actions)
        require(first_actions.shape == (n, 4) and first_actions.dtype == np.float32 and
                np.isfinite(first_actions).all() and np.all(np.abs(first_actions) <= 1), "Invalid forced complete actions")
    try:
        while not done.all():
            active = np.flatnonzero(~done)
            obs = np.stack([task.observe(x[i], int(t[i])) for i in active])
            if first_actions is not None and np.array_equal(t[active], initial_times[active]):
                actions = first_actions[active]
            else:
                actions = np.asarray(forward(obs))
                actor_calls += 1
            require(actions.shape == (len(active), 4) and actions.dtype == np.float32 and
                    np.isfinite(actions).all() and np.all(np.abs(actions) <= 1), "Policy must return deployed float32 actions in [-1,1]")
            for local, i in enumerate(active):
                require(attempts < max_transitions, "Evaluation transition cap exhausted before step")
                attempts += 1
                nxt, reward, terminal, info = task.step(x[i].copy(), int(t[i]), actions[local].copy())
                completed += 1
                nxt = np.asarray(nxt)
                require(nxt.shape == (13,) and nxt.dtype == np.float64 and np.isfinite(nxt).all() and
                        np.isfinite(reward) and isinstance(terminal, (bool, np.bool_)), "Invalid task transition")
                offset = int(t[i] - initial_times[i])
                values[i] += .98 ** offset * float(reward)
                traces[i].append({"state": x[i].copy(), "t": int(t[i]), "action": actions[local].copy(),
                                  "reward": float(reward), "terminal": bool(terminal), "info": info})
                x[i], t[i], done[i] = nxt, t[i] + 1, terminal
                require(t[i] <= 100 and (t[i] < 100 or done[i]), "Task failed to terminate at horizon")
    except BaseException as exc:
        # Preserve attempted work even if the injected simulator/policy raised.
        exc.evaluation_counts = {"transitions_attempted": attempts, "transitions_completed": completed,
                                 "actor_calls": actor_calls, "complete_episodes": int(done.sum())}
        raise
    return {"returns": values, "traces": traces, "final_states": x, "final_times": t,
            "counts": {"transitions_attempted": attempts, "transitions_completed": completed,
                       "actor_calls": actor_calls, "complete_episodes": n},
            "reference": "actual first action then the supplied deployed policy; gamma .98; no terminal bootstrap"}


def own_policy_q(task, states, times, actions, forward, *, max_transitions):
    import numpy as np
    actions = np.asarray(actions)
    require(actions.ndim == 3 and actions.shape[0] == len(states) and actions.shape[2] == 4 and
            actions.dtype == np.float32, "Complete action bank must be float32[N,K,4]")
    n, k = actions.shape[:2]
    result = rollout(task, np.repeat(states, k, axis=0), np.repeat(times, k), forward,
                     first_actions=actions.reshape(n*k, 4), max_transitions=max_transitions)
    return result["returns"].reshape(n, k), result["counts"]


def score_diagnostics(q1, width, qpi, *, error_tolerance, tie_tolerance, common_slots=3):
    """One model/reference at a time. Scores are not substituted for critic predictions."""
    import numpy as np
    q, u, truth = (np.asarray(v, np.float64) for v in (q1, width, qpi))
    require(q.ndim == 2 and q.shape == u.shape == truth.shape and q.shape[0] > 0 and
            all(np.isfinite(v).all() for v in (q, u, truth)) and np.all(u >= 0), "Invalid aligned Q/width/reference arrays")
    require(0 <= error_tolerance and 0 <= tie_tolerance and 1 <= common_slots <= q.shape[1], "Invalid fixed diagnostic tolerances")
    score = q - 3. * u
    def errors(e):
        over, under = e > error_tolerance, e < -error_tolerance
        return {"n": int(e.size), "over": int(over.sum()), "under": int(under.sum()),
                "near_zero": int((~(over | under)).sum()), "mae": float(np.abs(e).mean()),
                "mean_signed_error": float(e.mean()),
                "over_magnitude_sum": float(e[over].sum()), "under_magnitude_sum": float(-e[under].sum())}
    def choose(a):
        return np.asarray([int(np.flatnonzero(row >= row.max() - tie_tolerance)[0]) for row in a])
    a, b = choose(q[:, :common_slots]), choose(score[:, :common_slots])
    idx = np.arange(len(q))
    gain = truth[idx, b] - truth[idx, a]
    raw_error, score_error = q - truth, score - truth
    delta = np.abs(score_error) - np.abs(raw_error)
    return {"raw_critic": errors(raw_error), "adjusted_actor_score": errors(score_error),
        "absolute_error_improved": int((delta < -error_tolerance).sum()),
        "absolute_error_worsened": int((delta > error_tolerance).sum()),
        "common_action_ranking": {"n_states": len(q), "raw_indices": a.tolist(), "adjusted_indices": b.tolist(),
            "beneficial_switches": int((gain > error_tolerance).sum()), "harmful_switches": int((gain < -error_tolerance).sum()),
            "near_tie_changes": int(((a != b) & (np.abs(gain) <= error_tolerance)).sum()),
            "mean_value_gain": float(gain.mean()),
            "raw_mean_regret": float((truth[:, :common_slots].max(axis=1) - truth[idx, a]).mean()),
            "adjusted_mean_regret": float((truth[:, :common_slots].max(axis=1) - truth[idx, b]).mean())},
        "own_actor_slot": "excluded from common-action ranking; included in all signed-error counts if supplied",
        "scope": "within this critic's own-policy Qpi; cross-method targets can differ"}


def paired_returns(native, bca, native_ids, bca_ids):
    import numpy as np
    a, b = np.asarray(native, np.float64), np.asarray(bca, np.float64)
    require(a.ndim == 1 and a.shape == b.shape and len(a) > 0 and np.isfinite(a).all() and np.isfinite(b).all(), "Invalid returns")
    require(list(native_ids) == list(bca_ids) and len(native_ids) == len(a) and len(set(native_ids)) == len(a),
            "Return starts must have identical unique paired identities")
    delta = b-a
    return {"episodes": len(a), "native_mean": float(a.mean()), "bca_mean": float(b.mean()),
            "paired_mean_difference": float(delta.mean()), "paired_differences": delta.tolist(),
            "unit": "one trained seed; episodes are nested, not independent training replications"}
