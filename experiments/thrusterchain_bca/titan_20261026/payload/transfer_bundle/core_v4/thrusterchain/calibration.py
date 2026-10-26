"""Narrow split-conformal diagnostic for the final finite-objective decision.

At t=99 of the canonical 100-step finite objective, Q_99(x,a)=r(x,a).
This module makes no multi-step Q, neural posterior, policy-selection coverage,
or closed-loop occupancy claim. Support failure, infinite conformal rank, and
low effective sample size have separate representations.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

from .coverage import (CoverageLaw, _arrays, _integer, _group_ids, assert_disjoint_groups,
                       grouped_split, nine_cells, plan_binomial_count,
                       minimum_equal_weight_calibration)
from .environment import Physics, dynamics


def _immutable_array(value):
    """Bytes-backed arrays cannot have WRITEABLE re-enabled by a caller."""
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("frozen arrays cannot contain Python objects")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header+b"\n"+array.tobytes(order="C")).hexdigest()


def _json_raw(value):
    """Strict JSON arrays: absent/NaN=null, infinities are explicit strings."""
    if isinstance(value, np.ndarray):
        return _json_raw(value.tolist())
    if isinstance(value, np.generic):
        return _json_raw(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_raw(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_raw(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


def terminal_targets(states, actions, *, time_index=99, horizon=100,
                     task_semantics="finite_horizon"):
    """Exact final-decision Q labels from the canonical shared reward function.

    Raw canonical episodes return truncated=True at step 100. The finite-task
    adapter maps that boundary to termination, with zero continuation. A raw
    truncation in a continuing objective is NOT a terminal Q label.
    """
    states, actions = _arrays(states, actions)
    if task_semantics != "finite_horizon" or horizon != 100:
        raise ValueError("terminal Q labels require the canonical 100-step finite_horizon objective")
    times = np.asarray(time_index)
    if times.shape not in ((), (len(states),)) or not np.all(times == 99):
        raise ValueError("terminal Q labels require t=99, with one decision remaining")
    # The covariate is the continuously sampled requested command. Its response
    # includes the environment's float32 deployment map before canonical clipping.
    # Thus exact weights concern requested commands, not quantized action masses.
    with np.errstate(over="ignore", invalid="ignore"):
        executed = actions.astype(np.float32).astype(np.float64)
    if not np.isfinite(executed).all():
        raise ValueError("actions must remain finite at the environment float32 boundary")
    return np.asarray(dynamics(states, executed, Physics())[1], dtype=float)


@dataclass(frozen=True)
class ConformalRadius:
    radius: float
    status: str
    calibration_ess: float
    test_atom_mass: float
    ess_gate_passed: bool
    calibration_count: int

    @property
    def usable(self):
        return self.status == "finite" and self.ess_gate_passed


def weighted_conformal_quantile(scores, calibration_weights, test_weight, *, alpha=0.1, min_ess=0.0):
    """Quantile of sum_i p_i delta_score_i + p_test delta_(+infinity).

    Exact target/source weights are normalized jointly with the test weight.
    Finite calibration rank cannot be manufactured by dropping the test atom.
    ESS is an optional numerical usability gate and never changes the quantile.
    """
    scores, weights = np.asarray(scores, float), np.asarray(calibration_weights, float)
    if scores.ndim != 1 or weights.shape != scores.shape:
        raise ValueError("scores and calibration_weights must be matching vectors")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite; signed one-sided scores are allowed")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("calibration weights must be finite and nonnegative; do not hide missing support")
    if not np.isfinite(test_weight) or test_weight < 0:
        raise ValueError("test weight must be finite and nonnegative; check support first")
    if not np.isfinite(alpha) or not 0 < alpha < 1 or not np.isfinite(min_ess) or min_ess < 0:
        raise ValueError("alpha must be in (0,1); min_ess must be finite and nonnegative")
    original_weights = weights
    maximum = max(float(weights.max()) if len(weights) else 0.0, float(test_weight))
    if maximum == 0:
        raise ValueError("the combined calibration and test mass must be positive")
    calibration_max = float(weights.max()) if len(weights) else 0.0
    # ESS describes calibration weights only. Scaling it by a much larger test
    # weight can underflow squared masses and spuriously report ESS zero.
    ess_weights = weights/calibration_max if calibration_max else weights
    ess_mass = math.fsum(ess_weights.tolist())
    ess_square_mass = math.fsum((ess_weights*ess_weights).tolist())
    ess = ess_mass**2/ess_square_mass if ess_square_mass else 0.0
    # Scaling all masses together prevents overflow and preserves the quantile.
    weights, query_weight = weights/maximum, float(test_weight)/maximum
    calibration_mass = math.fsum(weights.tolist())
    total_mass = calibration_mass + query_weight
    threshold = (1.0-float(alpha))*total_mass
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.longdouble)
    # Force the final cumulative sum to the accurately summed total. A tiny
    # summation error must not turn the exact endpoint into a spurious infinity.
    if len(cumulative):
        cumulative[-1] = calibration_mass
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    # Resolve numerical rank boundaries exactly. A loose tolerance must not
    # decide acceptance: it only invokes an exact rational comparison on the
    # original float weights and the decimal alpha supplied by the caller.
    nearby = [j for j in (index-1, index) if 0 <= j < len(cumulative)]
    rounding_window = 8*np.finfo(float).eps*max(1., total_mass)*max(1, len(scores))
    if any(abs(float(cumulative[j])-threshold) <= rounding_window for j in nearby):
        exact_weights = [Fraction.from_float(float(w)) for w in original_weights]
        exact_total = sum(exact_weights, Fraction()) + Fraction.from_float(float(test_weight))
        exact_threshold = (1-Fraction(str(float(alpha))))*exact_total
        accumulated = Fraction()
        index = len(scores)
        for j, sorted_index in enumerate(order):
            accumulated += exact_weights[sorted_index]
            if accumulated >= exact_threshold:
                index = j
                break
    radius = float(scores[order[index]]) if index < len(scores) else math.inf
    return ConformalRadius(radius, "finite" if math.isfinite(radius) else "infinite_rank",
                           float(ess), query_weight/total_mass, ess >= min_ess, len(scores))


def weighted_conformal_radius(scores, calibration_weights, test_weight, *, alpha=0.1, min_ess=0.0):
    """Two-sided absolute-score wrapper; use quantile() for signed scores."""
    if np.any(np.asarray(scores) < 0):
        raise ValueError("absolute-residual scores must be nonnegative")
    return weighted_conformal_quantile(scores, calibration_weights, test_weight, alpha=alpha, min_ess=min_ess)


def _prediction(predictor, states, actions):
    prediction = np.asarray(predictor(states, actions), float)
    if prediction.shape != (len(states),) or not np.isfinite(prediction).all():
        raise ValueError("predictor must return one finite scalar per query")
    return prediction


@dataclass(frozen=True)
class TerminalCalibration:
    scores: np.ndarray
    weights: np.ndarray
    source_law: CoverageLaw
    target_law: CoverageLaw
    alpha: float
    min_ess: float
    support: dict
    guarantee_claimed: bool
    provenance: dict

    def __post_init__(self):
        object.__setattr__(self, "scores", _immutable_array(self.scores))
        object.__setattr__(self, "weights", _immutable_array(self.weights))
        object.__setattr__(self, "support", MappingProxyType(dict(self.support)))
        provenance = dict(self.provenance)
        for key in ("calibration_states", "calibration_actions", "calibration_predictions", "group_ids"):
            provenance[key] = _immutable_array(provenance[key])
        provenance["prior_group_counts"] = MappingProxyType(dict(provenance["prior_group_counts"]))
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    def intervals(self, predictor, states, actions):
        """Construct per-query intervals; support refusal is not an infinite band.

        Passing a different predictor from the one calibrated is forbidden by
        the protocol; the fingerprint checks predictions on calibration rows.
        A mutable predictor must remain frozen for the entire evaluation.
        """
        states, actions = _arrays(states, actions)
        if predictor is not self.provenance["predictor"]:
            raise ValueError("use the same frozen predictor that was calibrated")
        check = _prediction(predictor, self.provenance["calibration_states"], self.provenance["calibration_actions"])
        if not np.array_equal(check, self.provenance["calibration_predictions"]):
            raise ValueError("predictor changed after final calibration")
        center = _prediction(predictor, states, actions)
        n = len(states)
        lower, upper = np.full(n, np.nan), np.full(n, np.nan)
        radius, ess, atom = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        statuses, usable, ess_passed = [], np.zeros(n, bool), np.zeros(n, bool)
        query_weights = self.source_law.density_ratio(states, actions, self.target_law)
        target_density = self.target_law.density(states, actions)
        for i, weight in enumerate(query_weights):
            if target_density[i] <= 0:
                statuses.append("outside_target_law")
                continue
            if not self.support["absolute_continuity"]:
                statuses.append("support_refused")
                continue
            q = weighted_conformal_radius(self.scores, self.weights, weight, alpha=self.alpha, min_ess=self.min_ess)
            radius[i], ess[i], atom[i] = q.radius, q.calibration_ess, q.test_atom_mass
            lower[i], upper[i] = center[i]-q.radius, center[i]+q.radius
            statuses.append(q.status)
            usable[i], ess_passed[i] = q.usable, q.ess_gate_passed
        return {"center": center, "lower": lower, "upper": upper, "radius": radius,
                "status": np.array(statuses), "usable": usable, "ess_gate_passed": ess_passed,
                "calibration_ess": ess, "test_atom_mass": atom,
                "guarantee_claimed": self.guarantee_claimed}


def final_terminal_calibration(predictor, states, actions, group_ids, *, source_law,
                               target_law=None, prior_groups=None, alpha=0.1, min_ess=0.0,
                               predictor_frozen_before_final=False, target_frozen_before_final=False,
                               independent_groups=False, claim_guarantee=False,
                               weight_accounting="exact", task_semantics="finite_horizon", time_index=99):
    """Calibrate only after predictor/target selection, with fresh independent units.

    ``prior_groups`` records fitting, selection, and tuning-calibration groups.
    Each final row must come from a distinct independent group. Disjoint IDs
    cannot prove independence; it is an explicit data-generation precondition.
    Estimated ratios cannot activate the finite-sample guarantee flag.
    """
    states, actions = _arrays(states, actions)
    groups = _group_ids(group_ids)
    if groups.shape != (len(states),) or len(states) == 0:
        raise ValueError("one group ID is required for each nonempty calibration row")
    if len(np.unique(groups)) != len(groups):
        raise ValueError("calibration needs one row per independent group; dependent trajectory rows are not IID")
    prior_groups = dict(prior_groups or {})
    if "final_calibration" in prior_groups:
        raise ValueError("final_calibration is reserved for the fresh calibration groups")
    assert_disjoint_groups({**prior_groups, "final_calibration": groups})
    if not predictor_frozen_before_final or not target_frozen_before_final:
        raise ValueError("freeze predictor and target selection before drawing/using final calibration")
    if weight_accounting not in ("exact", "estimated"):
        raise ValueError("weight_accounting must be exact or estimated")
    target_law = target_law or CoverageLaw()
    support = source_law.support_report(target_law)
    if claim_guarantee and (not independent_groups or weight_accounting != "exact" or not support["absolute_continuity"]):
        raise ValueError("guarantee requires independent groups, exact ratios, and target absolute continuity")
    # Known design laws provide exact ratios. The 'estimated' marker only
    # suppresses a guarantee; it is not an estimator implementation.
    if np.any(source_law.density(states, actions) <= 0):
        raise ValueError("a calibration query lies outside the declared source law")
    labels = terminal_targets(states, actions, task_semantics=task_semantics, time_index=time_index)
    prediction = _prediction(predictor, states, actions)
    weights = source_law.density_ratio(states, actions, target_law)
    # Validate alpha/ESS even when global support refusal will prevent intervals.
    weighted_conformal_radius(np.abs(labels-prediction), weights, 1.0, alpha=alpha, min_ess=min_ess)
    provenance = {"predictor": predictor, "calibration_states": states.copy(),
                  "calibration_actions": actions.copy(), "calibration_predictions": prediction.copy(),
                  "group_ids": groups.copy(), "prior_group_counts": {k: len(np.unique(v)) for k, v in prior_groups.items()},
                  "independent_groups_asserted": bool(independent_groups), "weight_accounting": weight_accounting,
                  "final_calibration_used_for_selection": False, "target": "Q_99=r; finite horizon100; continuation0",
                  "neural_Q_or_posterior_claim": False}
    return TerminalCalibration(np.abs(labels-prediction), weights, source_law, target_law,
                               float(alpha), float(min_ess), support, bool(claim_guarantee), provenance)


def summarize_intervals(intervals, truth):
    """Keep vacuous coverage in coverage, but report it and usable coverage apart."""
    truth = np.asarray(truth, float)
    if truth.shape != intervals["center"].shape or not np.isfinite(truth).all():
        raise ValueError("truth must be a matching finite vector")
    status = intervals["status"]
    issued = (status == "finite") | (status == "infinite_rank")
    infinite = status == "infinite_rank"
    covered = issued & (truth >= intervals["lower"]) & (truth <= intervals["upper"])
    finite_width = 2*intervals["radius"][status == "finite"]
    usable = intervals["usable"]
    return {"queries": len(truth), "issued_intervals": int(issued.sum()),
            "support_refusals": int(np.sum(status == "support_refused")),
            "outside_target_queries": int(np.sum(status == "outside_target_law")),
            "infinite_intervals": int(infinite.sum()),
            "ess_gate_failures_among_issued": int(np.sum(issued & ~intervals["ess_gate_passed"])),
            "coverage_including_infinite": float(covered[issued].mean()) if issued.any() else None,
            "usable_interval_coverage": float(covered[usable].mean()) if usable.any() else None,
            "usable_intervals": int(usable.sum()),
            "mean_finite_width": float(finite_width.mean()) if len(finite_width) else None,
            "empirical_only": not intervals["guarantee_claimed"]}


def _features(states, actions):
    # Deliberately misspecified, low-dimensional terminal surrogate. This is
    # neither the benchmark actor/critic nor a Bayesian posterior.
    return np.column_stack([np.ones(len(states)), np.abs(states[:, :10]).sum(axis=1),
                            np.abs(actions).sum(axis=1), states[:, 10]])


def two_sided_nine_cell_smoke(*, n_calibration=128, n_evaluation=64, seed=20261022, alpha=0.1):
    """Tiny terminal mechanism check, explicitly not nine trained benchmarks."""
    n_calibration, n_evaluation = (_integer(n_calibration, "n_calibration", 1),
                                  _integer(n_evaluation, "n_evaluation", 1))
    rng, target, cells = np.random.default_rng(seed), CoverageLaw(), []
    for law in nine_cells():
        development = law.sample(128, rng, group_prefix=law.name+":development")
        splits = grouped_split(development.group_ids,
                               fractions={"fit": .5, "selection": .25, "tuning_calibration": .25}, seed=seed)
        labels = terminal_targets(development.states, development.actions)
        feature = _features(development.states, development.actions)
        fit, selection, tuning = (splits[name] for name in ("fit", "selection", "tuning_calibration"))
        candidates, losses = [], []
        for ridge in (0.01, 0.1, 1.0):
            coefficient = np.linalg.solve(feature[fit].T @ feature[fit] + ridge*np.eye(feature.shape[1]),
                                          feature[fit].T @ labels[fit])
            candidates.append(coefficient)
            losses.append(float(np.mean(np.abs(feature[selection] @ coefficient-labels[selection]))))
        coefficient = candidates[int(np.argmin(losses))].copy()
        def predictor(x, a, coefficient=coefficient):
            return _features(x, a) @ coefficient
        tuning_scores = np.abs(predictor(development.states[tuning], development.actions[tuning])-labels[tuning])
        # Fresh final calibration is generated only after the selected predictor
        # is frozen. Tuning calibration is reported, never reused as final.
        calibration = law.sample(n_calibration, rng, group_prefix=law.name+":fresh_final")
        fitted = final_terminal_calibration(predictor, calibration.states, calibration.actions,
                                            calibration.group_ids, source_law=law, target_law=target,
                                            prior_groups={k: development.group_ids[v] for k, v in splits.items()},
                                            alpha=alpha, min_ess=20, predictor_frozen_before_final=True,
                                            target_frozen_before_final=True, independent_groups=True,
                                            claim_guarantee=False)
        evaluation = target.sample(n_evaluation, rng, group_prefix=law.name+":evaluation")
        assert_disjoint_groups({"development": development.group_ids,
                                "final_calibration": calibration.group_ids, "evaluation": evaluation.group_ids})
        intervals = fitted.intervals(predictor, evaluation.states, evaluation.actions)
        summaries = summarize_intervals(intervals, terminal_targets(evaluation.states, evaluation.actions))
        actual_regions = {"core_core": int(np.sum(~calibration.state_gap & ~calibration.action_gap)),
                          "gap_core": int(np.sum(calibration.state_gap & ~calibration.action_gap)),
                          "core_gap": int(np.sum(~calibration.state_gap & calibration.action_gap)),
                          "gap_gap": int(np.sum(calibration.state_gap & calibration.action_gap))}
        cells.append({"cell": law.name, "law": law.metadata(), "support": dict(fitted.support),
                      "n_final_calibration": n_calibration, "realized_final_region_counts": actual_regions,
                      "development_group_counts": {k: len(v) for k, v in splits.items()},
                      "fresh_final_after_selection": True, "selection_absolute_error": min(losses),
                      "tuning_mean_absolute_error": float(tuning_scores.mean()),
                      "joint_gap_count_plan": plan_binomial_count(minimum_equal_weight_calibration(alpha),
                                                                   law.region_masses["gap_gap"]),
                      "diagnostic": summaries})
    return {"schema": "thrusterchain-terminal-nine-cell-smoke-v1", "seed": seed, "alpha": alpha,
            "kind": "tiny controlled diagnostic smoke; not nine trained benchmark results",
            "target": "terminal Q_99=r for finite objective horizon100 with no continuation",
            "density_measure": "continuous float64 physical states and commands",
            "model": "small misspecified terminal linear surrogate; no neural Q or posterior claim",
            "guarantee_claimed": False, "closed_loop_occupancy_claim": False, "cells": cells}


@dataclass(frozen=True)
class TerminalBootstrapModel:
    """Root-weighted terminal regression ensemble; not a Bayesian Q posterior."""
    coefficients: np.ndarray
    root_weights: np.ndarray
    root_ids: np.ndarray
    ridge: float
    sigma_floor: float
    draws_generated_independently: bool

    def __post_init__(self):
        for name in ("coefficients", "root_weights", "root_ids"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))

    def predict_distribution(self, states, actions):
        states, actions = _arrays(states, actions)
        members = _features(states, actions) @ self.coefficients.T
        raw_sigma = members.std(axis=1, ddof=1)
        return members.mean(axis=1), np.maximum(raw_sigma, self.sigma_floor), raw_sigma

    def __call__(self, states, actions):
        return self.predict_distribution(states, actions)[0]

    def diagnostics(self, states, actions):
        _, _, raw = self.predict_distribution(states, actions)
        return {"members": len(self.coefficients), "fit_roots": len(self.root_ids),
                "root_ids": self.root_ids.tolist(), "dirichlet_root_weights": self.root_weights.tolist(),
                "independent_member_draws": self.draws_generated_independently,
                "weight_origin": "generated independent Dirichlet draws" if self.draws_generated_independently else "caller supplied; independence unverified",
                "ridge": self.ridge, "sigma_floor": self.sigma_floor,
                "coefficients": self.coefficients.tolist(),
                "coefficients_sha256": _array_sha256(self.coefficients),
                "coefficients_dtype": self.coefficients.dtype.str,
                "coefficients_shape": list(self.coefficients.shape),
                "coefficient_diversity": float(np.max(np.ptp(self.coefficients, axis=0))),
                "raw_sigma_min": float(raw.min()), "raw_sigma_mean": float(raw.mean()),
                "scale_floor_fraction": float(np.mean(raw < self.sigma_floor)),
                "collapsed": bool(np.all(raw <= np.finfo(float).eps)),
                "interpretation": "terminal Bayesian-bootstrap diagnostic; ensemble spread is not a calibrated posterior"}


def fit_terminal_bootstrap(states, actions, group_ids, *, members=16, ridge=.01,
                           sigma_floor=.001, seed=0, root_weights=None):
    """Fit B terminal reward estimators using independent Dirichlet root weights.

    A root shared by several rows retains one total root weight, divided across
    its rows. No selection, tuning-calibration, or final-calibration labels may
    be included in these fitting inputs.
    """
    states, actions = _arrays(states, actions)
    members = _integer(members, "members", 2)
    groups = _group_ids(group_ids)
    if groups.shape != (len(states),) or len(states) == 0:
        raise ValueError("one fit root ID is required per nonempty row")
    if not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(sigma_floor) or sigma_floor <= 0:
        raise ValueError("ridge and sigma_floor must be finite and positive")
    roots, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = (np.random.default_rng(seed).dirichlet(np.ones(len(roots)), size=members)
               if root_weights is None else np.asarray(root_weights, float).copy())
    if (weights.shape != (members, len(roots)) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.allclose(weights.sum(axis=1), 1)):
        raise ValueError("root_weights must be positive normalized Dirichlet-draw arrays (members, roots)")
    features, labels = _features(states, actions), terminal_targets(states, actions)
    coefficients = []
    for draw in weights:
        per_row = len(roots)*draw[inverse]/counts[inverse]
        coefficients.append(np.linalg.solve(features.T @ (per_row[:, None]*features) + ridge*np.eye(features.shape[1]),
                                            features.T @ (per_row*labels)))
    coefficients = np.asarray(coefficients)
    for array in (coefficients, weights, roots):
        array.setflags(write=False)
    return TerminalBootstrapModel(coefficients, weights, roots, float(ridge), float(sigma_floor), root_weights is None)


@dataclass(frozen=True)
class TerminalLowerCalibration:
    base: TerminalCalibration
    model: TerminalBootstrapModel
    signed_scores: np.ndarray
    calibration_sigma: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "signed_scores", _immutable_array(self.signed_scores))
        object.__setattr__(self, "calibration_sigma", _immutable_array(self.calibration_sigma))

    def bounds(self, states, actions, *, weighting="weighted"):
        if weighting not in ("weighted", "unweighted"):
            raise ValueError("weighting must be weighted or unweighted")
        states, actions = _arrays(states, actions)
        center, sigma, raw_sigma = self.model.predict_distribution(states, actions)
        check_mean, check_sigma, _ = self.model.predict_distribution(
            self.base.provenance["calibration_states"], self.base.provenance["calibration_actions"])
        if (not np.array_equal(check_mean, self.base.provenance["calibration_predictions"])
                or not np.array_equal(check_sigma, self.calibration_sigma)):
            raise ValueError("model or scale changed after final calibration")
        n = len(states)
        lower, width, quantile = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        ess, atom = np.full(n, np.nan), np.full(n, np.nan)
        status, ess_passed, usable = [], np.zeros(n, bool), np.zeros(n, bool)
        query_weights = self.base.source_law.density_ratio(states, actions, self.base.target_law)
        target_density = self.base.target_law.density(states, actions)
        weights = self.base.weights if weighting == "weighted" else np.ones(len(self.signed_scores))
        for i, query_weight in enumerate(query_weights):
            if target_density[i] <= 0:
                status.append("outside_target_law")
                continue
            if not self.base.support["absolute_continuity"]:
                status.append("support_refused")
                continue
            q = weighted_conformal_quantile(self.signed_scores, weights,
                                            query_weight if weighting == "weighted" else 1.,
                                            alpha=self.base.alpha, min_ess=self.base.min_ess)
            quantile[i] = q.radius
            width[i] = max(0., q.radius)*sigma[i]
            lower[i] = center[i]-width[i]
            ess[i], atom[i] = q.calibration_ess, q.test_atom_mass
            status.append(q.status)
            ess_passed[i], usable[i] = q.ess_gate_passed, q.usable
        return {"center": center, "sigma": sigma, "raw_sigma": raw_sigma, "lower": lower,
                "width": width, "quantile": quantile, "status": np.array(status),
                "calibration_ess": ess, "test_atom_mass": atom, "ess_gate_passed": ess_passed,
                "usable": usable, "weighting": weighting,
                "guarantee_claimed": self.base.guarantee_claimed and weighting == "weighted"}


def final_terminal_lower_calibration(model, states, actions, group_ids, **kwargs):
    """Fresh calibration of z=(Qhat-r)/max(sigma,epsilon) for lower Q bounds."""
    if not isinstance(model, TerminalBootstrapModel):
        raise ValueError("this diagnostic requires the named terminal bootstrap model")
    # The model already knows its fitting roots. Callers cannot bypass leakage
    # checks by omitting the otherwise explicit prior_groups provenance.
    prior = dict(kwargs.pop("prior_groups", None) or {})
    fit = _group_ids(prior.get("fit", np.array([], dtype=model.root_ids.dtype)), allow_empty=True)
    prior["fit"] = np.unique(np.concatenate([fit, model.root_ids]))
    kwargs["prior_groups"] = prior
    base = final_terminal_calibration(model, states, actions, group_ids, **kwargs)
    center, sigma, _ = model.predict_distribution(states, actions)
    scores = (center-terminal_targets(states, actions))/sigma
    scores.setflags(write=False)
    sigma.setflags(write=False)
    return TerminalLowerCalibration(base, model, scores, sigma)


def summarize_lower_bounds(bounds, truth):
    truth = np.asarray(truth, float)
    if truth.shape != bounds["center"].shape or not np.isfinite(truth).all():
        raise ValueError("truth must be a matching finite vector")
    status = bounds["status"]
    issued = (status == "finite") | (status == "infinite_rank")
    finite = status == "finite"
    covered = issued & (truth >= bounds["lower"])
    usable = bounds["usable"]
    return {"queries": len(truth), "issued_intervals": int(issued.sum()),
            "support_refusals": int(np.sum(status == "support_refused")),
            "outside_target_queries": int(np.sum(status == "outside_target_law")),
            "infinite_intervals": int(np.sum(status == "infinite_rank")),
            "ess_gate_failures_among_issued": int(np.sum(issued & ~bounds["ess_gate_passed"])),
            "coverage_including_infinite": float(covered[issued].mean()) if issued.any() else None,
            "usable_interval_coverage": float(covered[usable].mean()) if usable.any() else None,
            "usable_intervals": int(usable.sum()),
            "mean_finite_width": float(bounds["width"][finite].mean()) if finite.any() else None,
            "empirical_only": not bounds["guarantee_claimed"]}


def _paired_terminal_action_arms(calibrator, states, rng, *, return_raw=False):
    """Four real action-selection arms, with separately retained bound statuses.

    If every bank candidate has a usable bound, choose the largest lower bound.
    Otherwise standard uses the host Qhat argmax; most_caution executes zero.
    Zero is a deliberately privileged, known terminal optimum in this task.
    """
    n, k = len(states), 8
    bank = rng.uniform(-1., 1., (n, k, 4))
    bank[:, 0] = 0.
    bank[:, 1] = 1.
    bank[:, 2] = -1.
    flat_states, flat_actions = np.repeat(states, k, axis=0), bank.reshape(-1, 4)
    reward = terminal_targets(flat_states, flat_actions).reshape(n, k)
    all_arms, accepted, raw_bounds = {}, {}, {}
    for weighting in ("unweighted", "weighted"):
        bound = calibrator.bounds(flat_states, flat_actions, weighting=weighting)
        mean = bound["center"].reshape(n, k)
        lower = bound["lower"].reshape(n, k)
        valid = bound["usable"].reshape(n, k).all(axis=1)
        raw_bounds[weighting] = _bound_record(bound, reward.reshape(-1), shape=(n, k))
        # Invalid rows do not enter the lower-bound argmax (NaNs remain visible
        # in the status arrays, never converted into plausible finite widths).
        calibrated_choice = np.zeros(n, dtype=int)
        calibrated_choice[valid] = np.argmax(lower[valid], axis=1)
        for fallback in ("standard", "most_caution"):
            fallback_choice = np.argmax(mean, axis=1) if fallback == "standard" else np.zeros(n, int)
            choice = np.where(valid, calibrated_choice, fallback_choice)
            chosen_reward = reward[np.arange(n), choice]
            name = f"{weighting}_{fallback}"
            accepted[name] = valid
            all_arms[name] = {"chosen_reward": chosen_reward, "selected_index": choice,
                              "accepted": valid,
                              "candidate_support_refusals": int(np.sum(bound["status"] == "support_refused")),
                              "candidate_infinite_bounds": int(np.sum(bound["status"] == "infinite_rank")),
                              "candidate_ess_gate_failures": int(np.sum(~bound["ess_gate_passed"] &
                                                                           (bound["status"] != "support_refused")))}
    common = np.logical_and.reduce(list(accepted.values()))
    result = {}
    raw_arms = {}
    for name, values in all_arms.items():
        valid, rewards = values["accepted"], values["chosen_reward"]
        result[name] = {"queries": n, "invalid_state_events": int(np.sum(~valid)),
                        "fallback_executions": int(np.sum(~valid)),
                        "full_population_mean_reward": float(rewards.mean()),
                        "accepted_count": int(valid.sum()),
                        "accepted_mean_reward": float(rewards[valid].mean()) if valid.any() else None,
                        "common_accepted_count": int(common.sum()),
                        "common_accepted_mean_reward": float(rewards[common].mean()) if common.any() else None,
                        "full_population_mean_regret_to_zero": float(np.mean(reward[:, 0]-rewards)),
                        "selected_action_bank_indices": values["selected_index"].tolist(),
                        "candidate_support_refusals": values["candidate_support_refusals"],
                        "candidate_infinite_bounds": values["candidate_infinite_bounds"],
                        "candidate_ess_gate_failures": values["candidate_ess_gate_failures"]}
        choice = values["selected_index"]
        raw_arms[name] = {"selected_index": choice.tolist(),
                          "selected_actions": bank[np.arange(n), choice].tolist(),
                          "selected_reward": rewards.tolist(), "accepted": valid.tolist(),
                          "fallback_executed": (~valid).tolist(), "common_accepted": common.tolist(),
                          "regret_to_zero": (reward[:, 0]-rewards).tolist()}
    if not return_raw:
        return result
    ratios = calibrator.base.source_law.ratio_report(flat_states, flat_actions, calibrator.base.target_law)
    raw = {"state_rows_from_evaluation": list(range(n)), "time_index": 99,
           "actions": bank.tolist(), "true_reward": reward.tolist(),
           "executed_actions": np.clip(bank.astype(np.float32).astype(np.float64), -1., 1.).tolist(),
           "exact_target_source_weights": _json_raw(ratios["ratio"].reshape(n, k)),
           "density_ratio_status": ratios["status"].reshape(n, k).tolist(),
           "state_gap": (states[:, 0] > 0).tolist(), "action_gap": (bank[:, :, 0] > .5).tolist(),
           "bounds_by_weighting": raw_bounds, "arms": raw_arms}
    return result, raw


def _bound_record(bounds, truth, shape=None):
    """Raw arrays sufficient to recompute coverage, status and usability counts."""
    truth = np.asarray(truth)
    issued = np.isin(bounds["status"], ["finite", "infinite_rank"])
    covered = truth >= bounds["lower"]
    record = {}
    for key in ("center", "sigma", "raw_sigma", "lower", "width", "quantile", "status",
                "calibration_ess", "test_atom_mass", "ess_gate_passed", "usable"):
        values = bounds[key].reshape(shape) if shape is not None else bounds[key]
        record[key] = _json_raw(values)
    record["issued"] = _json_raw(issued.reshape(shape) if shape is not None else issued)
    cover_values = np.where(issued, covered.astype(float), np.nan)
    # Coverage is 1/0 for issued bounds and null when no bound was issued.
    record["covered_when_issued"] = _json_raw(cover_values.reshape(shape) if shape is not None else cover_values)
    return record


def _query_record(queries, source, target):
    ratios = source.ratio_report(queries.states, queries.actions, target)
    return {"states": queries.states.tolist(), "actions": queries.actions.tolist(),
            "executed_actions": np.clip(queries.actions.astype(np.float32).astype(np.float64), -1., 1.).tolist(),
            "group_ids": queries.group_ids.tolist(), "time_index": [99]*len(queries.states),
            "state_gap": queries.state_gap.tolist(), "action_gap": queries.action_gap.tolist(),
            "region": [f"{'gap' if s else 'core'}_{'gap' if a else 'core'}"
                       for s, a in zip(queries.state_gap, queries.action_gap)],
            "true_terminal_q": terminal_targets(queries.states, queries.actions).tolist(),
            "source_density": ratios["source_density"].tolist(),
            "target_density": ratios["target_density"].tolist(),
            "exact_target_source_weights": _json_raw(ratios["ratio"]),
            "density_ratio_status": ratios["status"].tolist()}


def nine_cell_smoke(*, n_calibration=128, n_evaluation=64, seed=20261022, alpha=.1,
                   repetitions=3, members=16, min_ess=20.):
    """Repeated small terminal mechanism experiments; no trained benchmark claim."""
    n_calibration = _integer(n_calibration, "n_calibration", 1)
    n_evaluation = _integer(n_evaluation, "n_evaluation", 1)
    repetitions = _integer(repetitions, "repetitions", 1)
    target, cells = CoverageLaw(), []
    for cell_index, law in enumerate(nine_cells()):
        runs = []
        for repetition in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([seed, cell_index, repetition]))
            prefix = f"{law.name}:repeat{repetition}"
            development = law.sample(128, rng, group_prefix=prefix+":development")
            split = grouped_split(development.group_ids,
                                  fractions={"fit": .5, "selection": .25, "tuning_calibration": .25},
                                  seed=seed+repetition)
            fit, selection, tuning = (split[k] for k in ("fit", "selection", "tuning_calibration"))
            candidates, losses, candidate_seeds, selection_predictions = [], [], [], []
            for ridge in (.01, .1, 1.):
                fit_seed = int(rng.integers(0, 2**32))
                model = fit_terminal_bootstrap(development.states[fit], development.actions[fit],
                                               development.group_ids[fit], members=members, ridge=ridge,
                                               seed=fit_seed)
                candidates.append(model)
                candidate_seeds.append(fit_seed)
                prediction = model(development.states[selection], development.actions[selection])
                selection_predictions.append(prediction.tolist())
                truth = terminal_targets(development.states[selection], development.actions[selection])
                losses.append(float(np.mean(np.abs(prediction-truth))))
            model = candidates[int(np.argmin(losses))]
            tuning_mean, tuning_sigma, _ = model.predict_distribution(development.states[tuning], development.actions[tuning])
            tuning_z = (tuning_mean-terminal_targets(development.states[tuning], development.actions[tuning]))/tuning_sigma
            # Neither these tuning scores nor selection roots are used in final calibration.
            final = law.sample(n_calibration, rng, group_prefix=prefix+":fresh_final")
            calibrated = final_terminal_lower_calibration(
                model, final.states, final.actions, final.group_ids, source_law=law, target_law=target,
                prior_groups={k: development.group_ids[v] for k, v in split.items()},
                alpha=alpha, min_ess=min_ess, predictor_frozen_before_final=True,
                target_frozen_before_final=True, independent_groups=True, claim_guarantee=False)
            evaluation = target.sample(n_evaluation, rng, group_prefix=prefix+":evaluation")
            assert_disjoint_groups({"development": development.group_ids, "final": final.group_ids,
                                    "evaluation": evaluation.group_ids})
            query_truth = terminal_targets(evaluation.states, evaluation.actions)
            query_bounds = {name: calibrated.bounds(evaluation.states, evaluation.actions, weighting=name)
                            for name in ("unweighted", "weighted")}
            query_coverage = {name: summarize_lower_bounds(bound, query_truth) for name, bound in query_bounds.items()}
            action_arms, action_raw = _paired_terminal_action_arms(calibrated, evaluation.states, rng, return_raw=True)
            action_raw["group_ids"] = evaluation.group_ids.tolist()
            arms = {name: {"query_coverage": query_coverage[name.split("_")[0]], "action_selection": values}
                    for name, values in action_arms.items()}
            final_raw = _query_record(final, law, target)
            final_mean, final_sigma, final_raw_sigma = model.predict_distribution(final.states, final.actions)
            final_raw.update(qmean=final_mean.tolist(), sigma=final_sigma.tolist(), raw_sigma=final_raw_sigma.tolist(),
                             signed_score=calibrated.signed_scores.tolist(),
                             unweighted_calibration_weights=[1.]*n_calibration,
                             weighted_calibration_weights=calibrated.base.weights.tolist())
            evaluation_raw = _query_record(evaluation, law, target)
            evaluation_raw["bounds_by_weighting"] = {name: _bound_record(bound, query_truth)
                                                      for name, bound in query_bounds.items()}
            runs.append({"repetition": repetition, "seed_namespace": [seed, cell_index, repetition],
                         "selection_mae": min(losses), "tuning_score_mean": float(tuning_z.mean()),
                         "fresh_final_after_selection": True,
                         "development_group_counts": {k: len(v) for k, v in split.items()},
                         "development_group_ids": {k: development.group_ids[v].tolist() for k, v in split.items()},
                         "candidate_ridges": [.01, .1, 1.], "candidate_selection_mae": losses,
                         "candidate_fit_seeds": candidate_seeds,
                         "model": model.diagnostics(evaluation.states, evaluation.actions), "arms": arms,
                         "raw": {"development": _query_record(development, law, target),
                                 "development_role_indices": {k: v.tolist() for k, v in split.items()},
                                 "candidate_selection_predictions": selection_predictions,
                                 "tuning_signed_scores": tuning_z.tolist(),
                                 "final_calibration": final_raw, "evaluation": evaluation_raw,
                                 "paired_action_bank": action_raw},
                         "realized_final_region_counts": {
                             "core_core": int(np.sum(~final.state_gap & ~final.action_gap)),
                             "gap_core": int(np.sum(final.state_gap & ~final.action_gap)),
                             "core_gap": int(np.sum(~final.state_gap & final.action_gap)),
                             "gap_gap": int(np.sum(final.state_gap & final.action_gap))}})
        summary = {}
        for arm in runs[0]["arms"]:
            coverage = [run["arms"][arm]["query_coverage"]["coverage_including_infinite"] for run in runs]
            returns = np.array([run["arms"][arm]["action_selection"]["full_population_mean_reward"] for run in runs])
            measured = np.array([value for value in coverage if value is not None])
            summary[arm] = {"independent_repetitions": repetitions,
                            "mean_query_coverage": float(measured.mean()) if len(measured) else None,
                            "coverage_between_repetition_sd": float(measured.std(ddof=1)) if len(measured)>1 else None,
                            "coverage_mean_mc_se": float(measured.std(ddof=1)/np.sqrt(len(measured))) if len(measured)>1 else None,
                            "full_population_mean_reward": float(returns.mean()),
                            "reward_between_repetition_sd": float(returns.std(ddof=1)) if repetitions>1 else None,
                            "uncertainty_note": "between independent repetitions; tiny smoke, no precision or superiority claim"}
        cells.append({"cell": law.name, "law": law.metadata(), "support": law.support_report(target),
                      "n_final_calibration": n_calibration, "fresh_final_after_selection": True,
                      "diagnostic": runs[0]["arms"]["weighted_standard"]["query_coverage"],
                      "joint_gap_count_plan": plan_binomial_count(minimum_equal_weight_calibration(alpha),
                                                                   law.region_masses["gap_gap"]),
                      "repetition_summary": summary, "repetitions": runs})
    return {"schema": "thrusterchain-terminal-nine-cell-smoke-v3", "seed": seed, "alpha": alpha,
            "kind": "tiny controlled diagnostic smoke; not nine trained benchmark results",
            "target": "Q_99=r for finite horizon100, continuation0; fixed broad/broad query law",
            "score": "z=(Qhat-r)/max(sigma,epsilon); b=max(0,q)*sigma; lower=Qhat-b",
            "model": "terminal Bayesian-bootstrap linear estimators with independent Dirichlet fit-root weights",
            "sigma_interpretation": "ensemble spread with floor, not neural Q posterior uncertainty",
            "density_measure": "continuous float64 state13/requested-command4; fixed time99 counting-measure atom",
            "response_action_map": "requested action -> float32 -> float64 -> canonical clipping and dynamics reward",
            "four_arms": ["unweighted_standard", "unweighted_most_caution", "weighted_standard", "weighted_most_caution"],
            "fallback_preregistration": {"accepted": "all 8 action-bank candidates usable; argmax lower bound",
                                          "standard": "host Qhat argmax when any bank candidate is invalid",
                                          "most_caution": "zero command when any bank candidate is invalid",
                                          "privilege": "zero is a known terminal optimum; no policy-learning superiority implication"},
            "coverage_scope": "IID broad queries only; action-bank selected actions have no adaptive coverage guarantee",
            "raw_record_encoding": {"absent_or_nan": None, "positive_infinity": "Infinity", "negative_infinity": "-Infinity",
                                    "covered_when_issued": "1/0 for issued bounds, null for refused/outside-target queries"},
            "statistical_recipe": {
                "features": "[1, sum(abs(state[:10])), sum(abs(requested_action)), state[10]]",
                "bootstrap": "B independent Dirichlet(ones(number_of_fit_roots)); row_weight=number_of_fit_roots*root_weight/rows_in_root",
                "fit": "coefficient=solve(F.T@(row_weight[:,None]*F)+ridge*I, F.T@(row_weight*terminal_reward))",
                "selection": "minimum selection-root mean absolute error among ridges [.01,.1,1]; first argmin tie",
                "scale": "sample standard deviation across B predictions, ddof=1; max(std,.001)",
                "final_scores": "signed (Qmean-true_terminal_Q)/sigma; final roots fresh after selection",
                "quantile": "smallest score with cumulative calibration weight >= (1-alpha)*(sum calibration weights+test weight); otherwise +infinity",
                "unweighted_arm_weights": "unit weights for every calibration score and test atom; support checks still enforced",
                "near_rank_endpoint": "exact rational comparison using original float weights and decimal alpha; no tolerance acceptance",
                "lower_bound": "Qmean-max(0,quantile)*sigma",
                "calibration_ess": "(sum calibration weights)^2/sum(calibration weights^2); independent of test atom",
                "min_ess": min_ess,
                "query_coverage": "mean(true_terminal_Q>=lower) over issued bounds including infinite; refused/outside bounds excluded with separate counts",
                "uncertainty": "sample SD(ddof=1) of repetition coverage; MC_SE=SD/sqrt(number_of_repetitions_with_issued_bounds)",
                "return": "mean chosen canonical terminal reward over all evaluation states including fallback; accepted/common masks separately reported",
                "coefficient_hash": "sha256(JSON({dtype: array.dtype.str, shape:list(array.shape)},sort_keys=True,separators=(',',':')).ascii + newline + coefficient C-order bytes)"},
            "guarantee_claimed": False, "closed_loop_occupancy_claim": False,
            "final_calibration_reused_across_fixed_arms": True,
            "no_arm_selected_using_final_calibration": True, "cells": cells}
