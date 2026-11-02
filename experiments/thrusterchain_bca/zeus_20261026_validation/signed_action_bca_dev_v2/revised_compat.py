"""Cross-validated action compatibility, for frozen-checkpoint diagnostics.

This model estimates a local behavior-action density, not support or value.
Only retained training episodes enter its fit, folds, or distance threshold.
The caller supplies exactly one probe per episode; all statistics and neighbor
rows for a probe are fitted without its entire episode fold. The final bank is
refit on all training episodes after selecting a scalar bandwidth multiplier.
"""
from dataclasses import dataclass, replace
import hashlib

import numpy as np
import compatibility as original


@dataclass(frozen=True)
class Reference:
    base: original.Reference
    radius_p95: float
    multiplier: float


def _cost_numpy(actions, neighbors, bandwidth):
    """Unnormalized negative log-mixture, matching the JAX scoring cost."""
    z = (actions[:, None, :] - neighbors) / bandwidth[:, None, :]
    logw = -.5 * np.sum(z * z, axis=2)
    peak = logw.max(axis=1)
    return -(peak + np.log(np.exp(logw - peak[:, None]).mean(axis=1)))


def _likelihoods(query_actions, cross, candidates, floor):
    scores = []
    for multiplier in candidates:
        widths = np.maximum(cross['bandwidths'] * multiplier, floor)
        loss = _cost_numpy(query_actions.astype(np.float64), cross['actions'].astype(np.float64), widths)
        nll = loss + np.log(widths).sum(axis=1) + 2.0 * np.log(2 * np.pi)
        original._require(np.isfinite(nll).all(), 'nonfinite cross-validated density')
        scores.append(float(nll.mean()))
    return scores


def fit_crossvalidated(observations, actions, episode_ids, timesteps, query_indices, *,
                       row_ids=None, folds=5, seed=20261029, k=64, time_bin=10,
                       multipliers=(0.125, 0.25, 0.5, 1.0), min_action_bandwidth=0.05):
    """Select the multiplier by out-of-fold logged-action likelihood.

    For a disjoint evaluation state, all cross-fitted probe likelihoods select
    the final multiplier. For each training probe's outer fold, a nested inner
    CV selects its multiplier using only episodes outside that outer fold; this
    avoids using the probe's action in its own gradient-strength matching. Each
    fold's state/action statistics and neighbor rows exclude that fold. Density
    includes the Gaussian bandwidth normalizer. The p95 kth-neighbor radius is
    a training-only heuristic, not an OOD certificate.
    """
    x = np.asarray(observations)
    a = np.asarray(actions)
    e = np.asarray(episode_ids)
    ix = np.asarray(query_indices)
    original._require(ix.ndim == 1 and ix.dtype.kind in 'iu' and len(ix) > 0
                      and len(np.unique(ix)) == len(ix), 'invalid distinct query indices')
    original._require(e.ndim == 1 and len(e) == len(x) and a.shape == (len(x), 4),
                      'invalid training arrays')
    original._require(np.all((ix >= 0) & (ix < len(x))), 'query index outside training rows')
    original._require(len(np.unique(e[ix])) == len(ix), 'use exactly one query per episode')
    candidates = np.asarray(multipliers, dtype=np.float64)
    original._require(candidates.ndim == 1 and len(candidates) > 0
                      and np.isfinite(candidates).all() and np.all(candidates > 0)
                      and len(np.unique(candidates)) == len(candidates), 'invalid bandwidth grid')
    original._require(np.isfinite(min_action_bandwidth) and min_action_bandwidth > 0,
                      'invalid bandwidth floor')
    original._require(type(folds) is int and folds >= 3, 'nested CV requires at least three folds')
    cross = original.crossfit_query(x, a, e, timesteps, ix, row_ids=row_ids, folds=folds,
                                    seed=seed, k=k, time_bin=time_bin,
                                    min_action_bandwidth=min_action_bandwidth)
    # Both the fold-specific std and the selected scaled width have the floor.
    # This only differs from scaling raw std for degenerate coordinates.
    scores = _likelihoods(a[ix], cross, candidates, min_action_bandwidth)
    # Stable grid-order tie breaking is part of the deterministic protocol.
    winner = int(np.argmin(scores))
    multiplier = float(candidates[winner])
    outer_scores, outer_multiplier, inner_probe_counts, outer_radius = {}, {}, {}, {}
    cross['reliable'] = np.empty(len(ix), dtype=bool)
    for f in np.unique(cross['query_folds']):
        # Select on a completely disjoint set of episodes, including all
        # neighbor rows and normalization used by the inner selection.
        mapping = original.episode_folds(e, folds=folds, seed=seed)
        all_folds = np.asarray([mapping[int(v)] for v in e], dtype=np.int64)
        allowed = all_folds != f
        allowed_rows = np.flatnonzero(allowed)
        inverse = np.full(len(x), -1, dtype=np.int64)
        inverse[allowed_rows] = np.arange(len(allowed_rows))
        subquery = inverse[ix[cross['query_folds'] != f]]
        original._require(np.all(subquery >= 0), 'inner CV includes outer-fold episodes')
        sub_row_ids = None if row_ids is None else np.asarray(row_ids)[allowed]
        inner = original.crossfit_query(x[allowed], a[allowed], e[allowed], np.asarray(timesteps)[allowed],
                                        subquery, row_ids=sub_row_ids, folds=folds - 1,
                                        seed=seed + int(f) + 1, k=k, time_bin=time_bin,
                                        min_action_bandwidth=min_action_bandwidth)
        candidate_nll = _likelihoods(a[allowed][subquery], inner, candidates, min_action_bandwidth)
        chosen = float(candidates[int(np.argmin(candidate_nll))])
        positions = cross['query_folds'] == f
        cross['bandwidths'][positions] = np.maximum(cross['bandwidths'][positions] * chosen,
                                                    min_action_bandwidth)
        inner_radius = float(np.quantile(inner['distances'][:, -1], .95, method='higher'))
        cross['reliable'][positions] = cross['distances'][positions, -1] <= inner_radius
        outer_scores[str(int(f))] = candidate_nll
        outer_multiplier[str(int(f))] = chosen
        inner_probe_counts[str(int(f))] = int(len(subquery))
        outer_radius[str(int(f))] = inner_radius
    radius = float(np.quantile(cross['distances'][:, -1], .95, method='higher'))
    original._require(np.isfinite(radius) and radius >= 0, 'invalid distance threshold')
    base = original.fit_reference(x, a, e, timesteps, row_ids=row_ids, k=k, time_bin=time_bin,
                                  min_action_bandwidth=min_action_bandwidth)
    bandwidth = original._readonly(np.maximum(base.action_bandwidth * multiplier,
                                              min_action_bandwidth), '<f8')
    digest = hashlib.sha256((base.fingerprint_sha256 + '|cv-v2|'
                             + repr(multiplier) + '|' + repr(radius)).encode()).hexdigest()
    base = replace(base, action_bandwidth=bandwidth, fingerprint_sha256=digest)
    receipt = {'schema': 'revised_local_compatibility_v2',
               'grid_multipliers': candidates.tolist(), 'mean_negative_log_likelihood': scores,
               'selected_multiplier': multiplier, 'action_bandwidth': bandwidth.tolist(),
               'min_action_bandwidth': float(min_action_bandwidth),
               'state_radius_p95': radius, 'state_radius_definition': 'cross-fit kth-neighbor distance',
               'threshold_order': '95th percentile, higher interpolation',
               'crossfit_probe_count': int(len(ix)), 'folds': int(folds), 'fold_seed': int(seed),
               'inner_folds': int(folds - 1), 'outer_fold_selected_multipliers': outer_multiplier,
               'outer_fold_inner_mean_negative_log_likelihood': outer_scores,
               'outer_fold_inner_probe_counts': inner_probe_counts,
               'outer_fold_inner_radius_p95': outer_radius,
               'k': int(k), 'time_bin': int(time_bin),
               'reference_sha256': digest,
               'fold_references': cross['reference_sha256_by_fold'],
               'note': 'training-only empirical heuristic; not support or OOD certification'}
    return Reference(base, radius, multiplier), cross, receipt


def query_neighbors(reference, observations, timesteps, *, exclude_episode_ids=None):
    original._require(isinstance(reference, Reference), 'expected revised Reference')
    found = original.query_neighbors(reference.base, observations, timesteps,
                                     exclude_episode_ids=exclude_episode_ids)
    found['reliable'] = found['distances'][:, -1] <= reference.radius_p95
    found['state_radius_p95'] = reference.radius_p95
    return found


compatibility_cost_jax = original.compatibility_cost_jax
