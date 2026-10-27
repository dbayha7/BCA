"""Prospective frozen-selector scores and development-only usefulness metrics.

Pure NumPy functions: no file access, scientific source imports, model updates or
simulator calls. score_candidates never receives true values. analyze never
receives a policy and cannot change the prelabel selections. Bootstrap intervals
describe variation across independent episode groups conditional on one frozen
checkpoint; deterministic simulator values have zero Monte Carlo sampling error.
"""
import hashlib
import json

import numpy as np


KAPPAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SCHEMA = "unifloral_frozen_usefulness_scores.v1"


class InvalidCalibration(ValueError):
    """Unready, unusable or numerically invalid calibration blocks scoring."""


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("Report contains an unsupported or nonfinite value")


def _digest(value):
    raw = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _protocol(protocol):
    if protocol.get("phase") != "development":
        raise ValueError("This diagnostic is restricted to the development phase")
    if tuple(protocol["kappa_grid"]) != KAPPAS:
        raise ValueError("Unexpected prospective kappa grid")
    if protocol["query_count"] != 64 or protocol["bootstrap_draws"] != 2000:
        raise ValueError("Expected 64 independent episode queries and 2000 bootstrap draws")
    if protocol["bootstrap_seed"] != 2026102343:
        raise ValueError("Unexpected prospective bootstrap seed")
    if protocol.get("bootstrap_scheme") != "episode_groups_stratified_by_fixed_query_time":
        raise ValueError("Expected the prospective fixed-time stratified episode bootstrap")
    if list(protocol["query_times"]) != [0, 25, 50, 75]:
        raise ValueError("Unexpected fixed query-time strata")
    if protocol["tie_atol"] != 1e-6 or protocol["tie_rtol"] != 1e-6:
        raise ValueError("Numerical tie tolerances differ from the frozen protocol")
    groups = protocol["episode_group_ids"]
    if list(groups) != list(range(64)):
        raise ValueError("Expected one query from each of 64 distinct episode groups")
    return _plain(protocol)


def _finite(value, name, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(name + " has an unexpected shape: " + str(result.shape))
    if not np.all(np.isfinite(result)):
        raise ValueError(name + " must be finite")
    return result


def _rms(values):
    values = _finite(values, "RMS input")
    maximum = float(np.max(np.abs(values)))
    if maximum == 0.0:
        return 0.0
    result = maximum * float(np.sqrt(np.mean(np.square(values / maximum))))
    if not np.isfinite(result):
        raise ValueError("Nonfinite RMS")
    return result


def _ratio(numerator, denominator):
    # Zero denominators are represented as unavailable, never repaired by epsilon.
    return None if denominator == 0.0 else float(numerator / denominator)


def _width(policy, obs, actions, location):
    width, usable, valid = policy.width(obs, actions)
    usable_array, valid_array = np.asarray(usable), np.asarray(valid)
    if usable_array.dtype.kind != "b" or valid_array.dtype.kind != "b":
        raise InvalidCalibration(location + ": width usability/validity flags must be boolean")
    if usable_array.size == 0 or valid_array.size == 0 or not np.all(usable_array) or not np.all(valid_array):
        raise InvalidCalibration(location + ": calibration is unready, unusable or invalid")
    try:
        width = _finite(width, "width", (len(actions),))
    except ValueError as error:
        raise InvalidCalibration(location + ": " + str(error)) from error
    if np.any(width < 0.0):
        raise InvalidCalibration(location + ": width is negative")
    return width


def _values(policy, obs, actions, location, actor):
    obs = _finite(obs, "observation", (14,))
    actions = _finite(actions, "actions")
    if actions.ndim != 2 or actions.shape[1] != 4 or len(actions) == 0:
        raise ValueError("Expected a nonempty action matrix with four columns")
    if np.any(np.abs(actions) > 1.0):
        raise ValueError("Action outside the declared [-1,1] bounds")
    actor = _finite(actor, "frozen actor action", (4,))
    # Native critic and Calibrator concatenate observations/actions directly;
    # they require matching batch rank instead of relying on broadcasting.
    observations = np.broadcast_to(obs, (len(actions), 14))
    q = _finite(policy.q(observations, actions), "critic values", (len(actions), 2))
    width = _width(policy, observations, actions, location)
    disagreement = _finite(np.abs(q[:, 0] - q[:, 1]), "absolute twin disagreement")
    distance = _finite(np.mean(np.square(actions - actor[None, :]), axis=1), "actor distance")
    return actor, q, width, disagreement, distance


def _selection(scores, atol, rtol):
    scores = _finite(scores, "selection scores")
    best = float(np.max(scores))
    close = np.flatnonzero(np.abs(scores - best) <= atol + rtol * abs(best))
    selected = int(close[0])
    ordered = np.sort(scores)
    return {"index": selected, "exact_argmax_index": int(np.argmax(scores)),
            "near_tie": len(close) > 1, "inconclusive_selection": len(close) > 1,
            "near_best_indices": close.tolist(), "maximum_score": best,
            "selected_score": float(scores[selected]),
            "top_two_gap": None if len(scores) < 2 else float(ordered[-1] - ordered[-2])}


def score_candidates(policy, observations_list, banks, scale_bank, protocol):
    """Freeze every selection without simulator returns or final-test labels.

    scale_bank contains observations (256,14) and aligned actions (256,4) from
    fitting data only. The caller must establish split membership/provenance.
    Candidate banks contain at most nine unique actions, frozen actor first.
    """
    protocol = _protocol(protocol)
    if len(observations_list) != 64 or len(banks) != 64:
        raise ValueError("Expected exactly 64 query observations and candidate banks")
    fit_obs = _finite(scale_bank["observations"], "scale-bank observations", (256, 14))
    fit_actions = _finite(scale_bank["actions"], "scale-bank actions", (256, 4))
    if "role" in scale_bank and scale_bank["role"] != "fitting":
        raise ValueError("Penalty matching may use fitting data only")
    fit_actors = _finite(policy.act(fit_obs), "fitting-bank actor actions", (256, 4))
    fit_u, fit_d, fit_a = [], [], []
    for i, (obs, action, actor) in enumerate(zip(fit_obs, fit_actions, fit_actors)):
        _, _, u, d, a = _values(policy, obs, action[None, :], "scale bank row " + str(i), actor)
        fit_u.append(u[0]); fit_d.append(d[0]); fit_a.append(a[0])
    u_rms, d_rms, a_rms = map(_rms, (fit_u, fit_d, fit_a))
    factors = {"disagreement": None if u_rms == 0.0 or d_rms == 0.0 else u_rms / d_rms,
               "actor_distance": None if u_rms == 0.0 or a_rms == 0.0 else u_rms / a_rms}
    for factor in factors.values():
        if factor is not None and not np.isfinite(factor):
            raise ValueError("Nonfinite RMS matching coefficient")

    query_observations = _finite(observations_list, "query observations", (64, 14))
    # Match candidate_bank's exact (64,14) actor batch. Single-row Dense calls
    # can round differently; retain exact actor-first checks without relaxing
    # their tolerance or substituting another initial action.
    query_actors = _finite(policy.act(query_observations), "query actor actions", (64, 4))
    queries = []
    for i, (obs, actions, actor) in enumerate(zip(query_observations, banks, query_actors)):
        obs, actions = _finite(obs, "query observation", (14,)), _finite(actions, "candidate bank")
        if actions.ndim != 2 or actions.shape[1] != 4 or not 1 <= len(actions) <= 9:
            raise ValueError("Expected one to nine four-dimensional candidate actions")
        if len(np.unique(actions, axis=0)) != len(actions):
            raise ValueError("Candidate actions must already be deduplicated")
        actor, q, u, d, a = _values(policy, obs, actions, "query " + str(i), actor)
        if not np.array_equal(actions[0], actor):
            raise ValueError("Candidate index zero must equal the same frozen actor action exactly")
        queries.append({"index": i, "episode_group_id": protocol["episode_group_ids"][i],
                        "observation": obs.tolist(), "actions": actions.tolist(),
                        "q": q.tolist(), "u": u.tolist(), "disagreement": d.tolist(),
                        "actor_distance": a.tolist(), "width_usable": True, "width_valid": True,
                        "q1_rms": _rms(q[:, 0]), "u_rms": _rms(u),
                        "u_rms_to_q1_rms": _ratio(_rms(u), _rms(q[:, 0]))})

    methods = {}

    def add_method(name, family, kappa=None, factor=1.0, unavailable_reason=None):
        entry = {"family": family, "kappa_penalty": kappa, "available": unavailable_reason is None,
                 "unavailable_reason": unavailable_reason, "rms_match_factor": factor,
                 "actual_u_scale_zero": u_rms == 0.0,
                 "penalty_signal_degenerate": u_rms == 0.0 if family == "bca" else False,
                 "usefulness_claim_allowed": False,
                 "selected_indices": [], "selection_details": [], "scores": [],
                 "penalty_rms": [], "penalty_rms_to_q1_rms": []}
        if unavailable_reason is None:
            for query in queries:
                q = np.asarray(query["q"])
                penalty = np.zeros(len(q), dtype=np.float64)
                if family == "twin_min":
                    scores = q.min(axis=1)
                elif family == "actor":
                    scores = -np.asarray(query["actor_distance"])
                else:
                    if family == "bca":
                        penalty = kappa * np.asarray(query["u"])
                    elif family in ("raw_disagreement", "matched_disagreement"):
                        penalty = kappa * factor * np.asarray(query["disagreement"])
                    elif family == "matched_actor_distance":
                        penalty = kappa * factor * np.asarray(query["actor_distance"])
                    scores = q[:, 0] - penalty
                choice = _selection(scores, protocol["tie_atol"], protocol["tie_rtol"])
                if family == "actor":
                    choice = {"index": 0, "exact_argmax_index": 0, "near_tie": False,
                              "inconclusive_selection": False, "near_best_indices": [0],
                              "maximum_score": 0.0, "selected_score": 0.0, "top_two_gap": None,
                              "rule": "fixed actor index; no score-based selection"}
                entry["selected_indices"].append(choice["index"])
                entry["selection_details"].append(choice)
                entry["scores"].append(_finite(scores, "method scores").tolist())
                entry["penalty_rms"].append(_rms(penalty))
                entry["penalty_rms_to_q1_rms"].append(_ratio(_rms(penalty), query["q1_rms"]))
        methods[name] = entry

    add_method("raw_q", "raw_q")
    add_method("twin_min", "twin_min")
    add_method("actor", "actor")
    for kappa in KAPPAS:
        tag = format(kappa, "g")
        add_method("bca_kappa_" + tag, "bca", kappa)
        add_method("raw_disagreement_kappa_" + tag, "raw_disagreement", kappa)
        for key, family in (("disagreement", "matched_disagreement"),
                            ("actor_distance", "matched_actor_distance")):
            reason = None
            if u_rms == 0.0:
                reason = "Actual U RMS is zero; strength matching is unavailable"
            elif factors[key] is None:
                reason = "Control RMS is zero; strength matching is unavailable"
            add_method(family + "_kappa_" + tag, family, kappa, factors[key], reason)

    result = {"schema": SCHEMA, "phase": "development", "protocol_sha256": _digest(protocol),
              "selection_rule": "First index within atol+rtol*abs(max_score); near ties flagged inconclusive",
              "disagreement_definition": "abs(Q1-Q2), without a one-half factor",
              "actor_distance_definition": "mean over four coordinates of (a-frozen_actor(obs))**2",
              "scale_bank": {"rows": 256, "role": "fitting", "u_rms": u_rms,
                             "disagreement_rms": d_rms, "actor_distance_rms": a_rms,
                             "matching_factors": factors, "bca_strength_matching_available": u_rms > 0.0,
                             "membership_verification": "Caller supplies authenticated fitting-only membership",
                             "bank_sha256": _digest({"observations": fit_obs, "actions": fit_actions}),
                             "raw_disagreement_is_unmatched": True},
              "queries": queries, "methods": methods,
              "candidate_table_sha256": _digest([{k: q[k] for k in ("observation", "actions")} for q in queries]),
              "selections_frozen_before_truevalues": True,
              "scientific_scope": "Frozen candidate reranking; not actor training or an efficacy claim"}
    result = _plain(result)
    result["prelabel_signature"] = _digest(result)
    return result


def _correlation(x, y):
    x, y = _finite(x, "correlation predictor"), _finite(y, "correlation outcome")
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Correlation requires aligned vectors")
    if len(x) < 2:
        return {"defined": False, "reason": "fewer than two candidates", "n": len(x), "value": None}
    # Detect exact constants before centering: a rounded mean of repeated
    # identical floats can otherwise leave a spurious nonzero residual.
    if np.all(x == x[0]) or np.all(y == y[0]):
        return {"defined": False, "reason": "constant predictor or outcome", "n": len(x), "value": None}
    x, y = x - x.mean(), y - y.mean()
    sx, sy = _rms(x), _rms(y)
    if sx == 0.0 or sy == 0.0:
        return {"defined": False, "reason": "constant predictor or outcome", "n": len(x), "value": None}
    value = float(np.mean((x / sx) * (y / sy)))
    if not np.isfinite(value):
        raise ValueError("Nonfinite correlation")
    return {"defined": True, "reason": None, "n": len(x), "value": float(np.clip(value, -1.0, 1.0))}


def _ranks(values):
    values = _finite(values, "rank input")
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranked


def _relationships(x, y):
    return {"pearson": _correlation(x, y), "spearman": _correlation(_ranks(x), _ranks(y))}


def analyze(prelabeldata, truevalues, protocol):
    """Analyze fixed selections; truevalues are deterministic Q^frozen_actor.

    Every method/kappa is reported. No best method or best kappa is chosen.
    Candidate-bank oracle indices are retrospective diagnostics only.
    """
    protocol = _protocol(protocol)
    sealed = {k: v for k, v in prelabeldata.items() if k != "prelabel_signature"}
    if prelabeldata.get("schema") != SCHEMA or _digest(sealed) != prelabeldata.get("prelabel_signature"):
        raise ValueError("Prelabel scores or frozen selections changed")
    if prelabeldata["protocol_sha256"] != _digest(protocol):
        raise ValueError("Protocol changed after score selection")
    queries, methods = prelabeldata["queries"], prelabeldata["methods"]
    if len(queries) != 64 or len(truevalues) != 64:
        raise ValueError("Expected deterministic value vectors for all 64 query groups")
    values = [_finite(v, "deterministic candidate values", (len(q["actions"]),))
              for q, v in zip(queries, truevalues)]
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    bootstrap_indices = np.empty((protocol["bootstrap_draws"], 64), dtype=np.int64)
    strata = []
    for stratum, query_time in enumerate(protocol["query_times"]):
        indices = np.arange(stratum, 64, 4, dtype=np.int64)
        draws = rng.integers(0, len(indices), size=(protocol["bootstrap_draws"], len(indices)))
        bootstrap_indices[:, indices] = indices[draws]
        strata.append({"query_time": query_time, "episode_group_ids": indices.tolist(),
                       "groups_per_draw": len(indices), "sampling": "with replacement within stratum"})

    def interval(per_episode):
        per_episode = _finite(per_episode, "per-episode metric", (64,))
        replicates = per_episode[bootstrap_indices].mean(axis=1)
        lower, upper = np.quantile(replicates, [0.025, 0.975], method="linear")
        return {"mean": float(per_episode.mean()), "pointwise_ci95": [float(lower), float(upper)],
                "pointwise_interval_contains_zero": bool(lower <= 0.0 <= upper)}

    def selected_values(method):
        indices = method["selected_indices"]
        if len(indices) != 64:
            raise ValueError("Incomplete frozen selections")
        if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(v)
               for index, v in zip(indices, values)):
            raise ValueError("Invalid frozen candidate index")
        return np.asarray([v[index] for v, index in zip(values, indices)])

    raw_values, actor_values = selected_values(methods["raw_q"]), selected_values(methods["actor"])
    best_values = np.asarray([v.max() for v in values])
    oracle = [_selection(v, protocol["tie_atol"], protocol["tie_rtol"]) for v in values]
    reports = {}
    for name, method in methods.items():
        entry = {"family": method["family"], "kappa_penalty": method["kappa_penalty"],
                 "available": method["available"], "unavailable_reason": method["unavailable_reason"],
                 "penalty_signal_degenerate": method["penalty_signal_degenerate"],
                 "usefulness_claim_allowed": False}
        if method["available"]:
            chosen = selected_values(method)
            raw_gain, actor_gain, regret = chosen - raw_values, chosen - actor_values, best_values - chosen
            tolerance = protocol["tie_atol"] + protocol["tie_rtol"] * np.abs(raw_values)
            tie_flags = [item["near_tie"] for item in method["selection_details"]]
            entry.update(selected_indices=method["selected_indices"], selected_values=chosen.tolist(),
                         gain_vs_raw_q=interval(raw_gain), gain_vs_actor=interval(actor_gain),
                         candidate_bank_regret=interval(regret),
                         per_episode_gain_vs_raw_q=raw_gain.tolist(),
                         per_episode_gain_vs_actor=actor_gain.tolist(), per_episode_regret=regret.tolist(),
                         score_near_tie_flags=tie_flags, score_near_tie_count=int(sum(tie_flags)),
                         gain_vs_raw_q_numerically_inconclusive=(np.abs(raw_gain) <= tolerance).tolist(),
                         improved_vs_raw_q_count=int(np.sum(raw_gain > tolerance)),
                         harmed_vs_raw_q_count=int(np.sum(raw_gain < -tolerance)),
                         penalty_rms=method["penalty_rms"],
                         penalty_rms_to_q1_rms=method["penalty_rms_to_q1_rms"])
        reports[name] = entry

    errors, per_state, pooled = [], [], {}
    predictor_names = ("u", "disagreement", "actor_distance")
    outcome_names = ("signed_q1_error", "positive_q1_optimism", "absolute_q1_error")
    for query, v in zip(queries, values):
        signed = np.asarray(query["q"])[:, 0] - v
        outcomes = {"signed_q1_error": signed, "positive_q1_optimism": np.maximum(signed, 0.0),
                    "absolute_q1_error": np.abs(signed)}
        errors.append({key: item.tolist() for key, item in outcomes.items()})
        per_state.append({"episode_group_id": query["episode_group_id"],
                          "relationships": {p: {o: _relationships(query[p], outcomes[o]) for o in outcome_names}
                                            for p in predictor_names}})
    within_summary = {}
    for predictor in predictor_names:
        pooled[predictor], within_summary[predictor] = {}, {}
        x = np.concatenate([q[predictor] for q in queries])
        for outcome in outcome_names:
            y = np.concatenate([e[outcome] for e in errors])
            pooled[predictor][outcome] = _relationships(x, y)
            summary = {}
            for statistic in ("pearson", "spearman"):
                entries = [row["relationships"][predictor][outcome][statistic] for row in per_state]
                defined = [entry["value"] for entry in entries if entry["defined"]]
                summary[statistic] = {"defined_states": len(defined), "undefined_states": 64 - len(defined),
                                      "mean_over_defined_states": None if not defined else float(np.mean(defined)),
                                      "scope": "Descriptive within-state associations; no significance test"}
            within_summary[predictor][outcome] = summary

    mean_signed = np.asarray([np.mean(e["signed_q1_error"]) for e in errors])
    mean_optimism = np.asarray([np.mean(e["positive_q1_optimism"]) for e in errors])
    mean_absolute = np.asarray([np.mean(e["absolute_q1_error"]) for e in errors])
    result = {"schema": "unifloral_frozen_usefulness_analysis.v1", "phase": "development",
              "prelabel_signature": prelabeldata["prelabel_signature"],
              "truevalues_sha256": _digest(values), "protocol_sha256": _digest(protocol),
              "methods": reports, "scale_bank": prelabeldata["scale_bank"],
              "q1_errors": {"per_query_candidate_errors": errors, "mean_signed_error": interval(mean_signed),
                            "mean_positive_optimism": interval(mean_optimism), "mean_absolute_error": interval(mean_absolute)},
              "relationships": {"pooled": pooled, "within_state": per_state, "within_state_summary": within_summary,
                                "pooled_relationships_confounded_by_state": True,
                                "correlations_are_not_causal_or_calibration_guarantees": True},
              "candidate_bank_oracle": {"values": best_values.tolist(), "near_tie_details": oracle,
                                        "use": "Retrospective candidate-bank regret only; never a deployed selector"},
              "bootstrap": {"draws": 2000, "seed": protocol["bootstrap_seed"], "independent_episode_groups": 64,
                            "index_sha256": _digest(bootstrap_indices), "resampling_unit": "one episode/state, all actions together",
                            "scheme": protocol["bootstrap_scheme"], "strata": strata,
                            "intervals": "Pointwise percentile95%; not simultaneous or multiple-comparison corrected",
                            "uncertainty": "Episode variation under the fixed 16-per-time design, conditional on one frozen checkpoint; zero simulator Monte Carlo error"},
              "best_kappa_selected": False, "confirmation_data_accessed": False,
              "limitations": ["Development diagnostic; no efficacy or confirmation claim.",
                              "ActualU=0 yields degenerate BCA scores, not positive usefulness evidence.",
                              "RMS matching fixes fitting-bank magnitude, not information content or gradient strength.",
                              "Regret is restricted to the supplied candidate banks, not globally optimal control.",
                              "Frozen discrete reranking does not establish trained actor improvement.",
                              "Current-policy values may differ from the critic's smoothed lagged target-policy estimand.",
                              "Fitting/evaluation provenance, deterministic simulator and checkpoint authenticity are caller gates."]}
    return _plain(result)
