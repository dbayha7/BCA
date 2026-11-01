"""Training-only local action compatibility for the development diagnostic.

This is a finite-data KDE-like cost, *not* a support classifier or a calibrated
bound. Neighbor selection uses observed physical state and a supplied, explicit
episode time. It never sees Q labels, simulator references or the hole mask.

For online actor updates, select neighbors outside the transformed actor step
and carry the resulting [B,K,4] array alongside the same observations. The
action cost below is JIT-compatible and differentiates only the live action;
its neighbor set and bandwidth are frozen within that actor step.
"""
from dataclasses import dataclass
import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer_vector(x, n, name):
    a = np.asarray(x)
    _require(a.shape == (n,) and a.dtype.kind in 'iu', f'{name} must be an integer vector of shape ({n},)')
    return a.astype(np.int64, copy=False)


def _obs(x, name):
    a = np.asarray(x)
    _require(a.ndim == 2 and a.shape[1] == 14 and a.dtype.kind == 'f' and np.isfinite(a).all(),
             f'{name} must be finite floating [N,14] raw observations')
    return a.astype(np.float64, copy=False)


def _times(x, n, time_bin):
    a = _integer_vector(x, n, 'timesteps')
    _require(np.all((a >= 0) & (a < 100)), 'timesteps must be in [0,100)')
    return a // time_bin


def _check_time_coordinate(observations, timesteps):
    t = np.asarray(timesteps, dtype=np.float64)
    _require(np.all(np.abs(observations[:, -1] - (1.0 - t / 100.0)) <= 1e-6),
             'raw remaining-time observation does not match the supplied timestep')


@dataclass(frozen=True)
class Reference:
    """Frozen retained-training bank; arrays must be treated as read-only."""
    states: np.ndarray
    actions: np.ndarray
    episodes: np.ndarray
    time_bins: np.ndarray
    row_ids: np.ndarray
    state_mean: np.ndarray
    state_scale: np.ndarray
    action_bandwidth: np.ndarray
    k: int
    time_bin: int
    fingerprint_sha256: str


def _readonly(a, dtype):
    x = np.ascontiguousarray(a, dtype=dtype)
    # A bytes-backed array cannot be made writable by a consumer.
    return np.frombuffer(x.tobytes(), dtype=x.dtype).reshape(x.shape)


def fit_reference(observations, actions, episode_ids, timesteps, *, row_ids=None,
                  k=64, time_bin=10, min_action_bandwidth=0.05):
    """Fit from retained *training* rows only; no calibration/evaluation rows.

    The first 13 raw physical coordinates are standardized using the supplied
    rows. The final observation coordinate is time and is not double-counted;
    neighbors must share the supplied time bin. Action bandwidth is the
    training-row population standard deviation in each action coordinate,
    floored at the prespecified ``min_action_bandwidth`` in action units.
    This symmetric Gaussian mixture has no fitted labels or simulator outcomes.
    """
    x = _obs(observations, 'training observations')
    n = len(x)
    a = np.asarray(actions)
    _require(n > 0 and a.shape == (n, 4) and a.dtype.kind == 'f' and np.isfinite(a).all(),
             'training actions must be finite floating [N,4]')
    _require(np.all(np.abs(a) <= 1.0 + 1e-6), 'training action outside declared [-1,1] coordinates')
    ep = _integer_vector(episode_ids, n, 'episode IDs')
    rid = np.arange(n, dtype=np.int64) if row_ids is None else _integer_vector(row_ids, n, 'row IDs')
    _require(len(np.unique(rid)) == n, 'row IDs must be unique')
    _require(type(k) is int and k > 0 and type(time_bin) is int and time_bin > 0
             and 100 % time_bin == 0, 'invalid K or time-bin width')
    _require(np.isfinite(min_action_bandwidth) and min_action_bandwidth > 0,
             'action bandwidth floor must be positive and finite')
    bins = _times(timesteps, n, time_bin)
    _check_time_coordinate(x, timesteps)
    physical = x[:, :13]
    mean = physical.mean(axis=0, dtype=np.float64)
    std = physical.std(axis=0, dtype=np.float64, ddof=0)
    scale = np.where(std < 1e-12, 1.0, std)
    bandwidth = np.maximum(a.astype(np.float64).std(axis=0, ddof=0), min_action_bandwidth)
    z = (physical - mean) / scale
    items = [_readonly(z, '<f8'), _readonly(a, '<f4'), _readonly(ep, '<i8'),
             _readonly(bins, '<i8'), _readonly(rid, '<i8'), _readonly(mean, '<f8'),
             _readonly(scale, '<f8'), _readonly(bandwidth, '<f8')]
    digest = hashlib.sha256()
    digest.update(f'compatibility-v1|K={k}|bin={time_bin}|floor={min_action_bandwidth:.17g}|'.encode())
    for item in items:
        digest.update(str(item.shape).encode()); digest.update(item.tobytes())
    return Reference(*items, k, time_bin, digest.hexdigest())


def query_neighbors(reference, observations, timesteps, *, exclude_episode_ids=None):
    """Return frozen nearest-state actions and audit identities for each query.

    If queries come from reference episodes, pass their episode IDs to exclude
    their *entire* episodes. For truly held-out episodes use a cross-fitted
    reference or a disjoint reference. Ties use ascending original row ID.
    A time bin with fewer than K eligible rows fails; no replacement/backfill.
    Returns [B,K,4] actions, [B,K] source row IDs, and [B,K] distances.
    """
    _require(isinstance(reference, Reference), 'expected fitted compatibility Reference')
    q = _obs(observations, 'query observations')
    bins = _times(timesteps, len(q), reference.time_bin)
    _check_time_coordinate(q, timesteps)
    exclude = None if exclude_episode_ids is None else _integer_vector(exclude_episode_ids, len(q), 'excluded episode IDs')
    z = (q[:, :13] - reference.state_mean) / reference.state_scale
    selected = np.empty((len(q), reference.k), dtype=np.int64)
    dist = np.empty((len(q), reference.k), dtype=np.float64)
    for i in range(len(q)):
        eligible = reference.time_bins == bins[i]
        if exclude is not None:
            eligible &= reference.episodes != exclude[i]
        indices = np.flatnonzero(eligible)
        _require(len(indices) >= reference.k, f'query {i}: fewer than K eligible retained rows')
        diff = reference.states[indices] - z[i]
        d2 = np.einsum('ij,ij->i', diff, diff)
        _require(np.isfinite(d2).all(), f'query {i}: nonfinite state distance')
        order = np.lexsort((reference.row_ids[indices], d2))[:reference.k]
        selected[i] = indices[order]
        dist[i] = np.sqrt(d2[order])
    return {'actions': reference.actions[selected], 'row_ids': reference.row_ids[selected],
            'episode_ids': reference.episodes[selected], 'distances': dist,
            'reference_sha256': reference.fingerprint_sha256}


def episode_folds(episode_ids, *, folds=5, seed=20261029):
    """Deterministically assign whole unique episodes to balanced folds."""
    ep = np.asarray(episode_ids)
    _require(ep.ndim == 1 and ep.dtype.kind in 'iu', 'episode IDs must be an integer vector')
    ids = np.unique(ep.astype(np.int64, copy=False))
    _require(type(folds) is int and 2 <= folds <= len(ids), 'invalid number of episode folds')
    _require(type(seed) is int and seed >= 0, 'invalid fold seed')
    ids = ids.copy()
    np.random.Generator(np.random.PCG64(seed)).shuffle(ids)
    return {int(episode): int(i % folds) for i, episode in enumerate(ids)}


def crossfit_query(observations, actions, episode_ids, timesteps, query_indices, *,
                   row_ids=None, folds=5, seed=20261029, k=64, time_bin=10,
                   min_action_bandwidth=0.05):
    """Out-of-fold queries; reference statistics and neighbor rows exclude fold.

    Query indices refer to supplied retained training rows. This is strict
    episode-fold cross-fitting: bandwidth and normalization are fitted anew on
    every reference fold, not on the query episodes. The caller may query only
    a bounded subset of rows. The returned per-query bandwidth can therefore
    differ by fold; see ``compatibility_cost_jax``.
    """
    x = _obs(observations, 'training observations')
    n = len(x)
    a = np.asarray(actions)
    _require(a.shape == (n, 4), 'training actions must have shape [N,4]')
    ep = _integer_vector(episode_ids, n, 'episode IDs')
    t = _integer_vector(timesteps, n, 'timesteps')
    rid = np.arange(n, dtype=np.int64) if row_ids is None else _integer_vector(row_ids, n, 'row IDs')
    ix = np.asarray(query_indices)
    _require(ix.ndim == 1 and ix.dtype.kind in 'iu' and np.all((ix >= 0) & (ix < n)), 'invalid query indices')
    mapping = episode_folds(ep, folds=folds, seed=seed)
    row_fold = np.array([mapping[int(v)] for v in ep], dtype=np.int64)
    result_actions = np.empty((len(ix), k, 4), dtype=np.float32)
    result_ids = np.empty((len(ix), k), dtype=np.int64)
    result_episodes = np.empty((len(ix), k), dtype=np.int64)
    result_distances = np.empty((len(ix), k), dtype=np.float64)
    bandwidths = np.empty((len(ix), 4), dtype=np.float64)
    digests = {}
    for f in np.unique(row_fold[ix]):
        train = row_fold != f
        positions = np.flatnonzero(row_fold[ix] == f)
        reference = fit_reference(x[train], a[train], ep[train], t[train], row_ids=rid[train],
                                  k=k, time_bin=time_bin, min_action_bandwidth=min_action_bandwidth)
        found = query_neighbors(reference, x[ix[positions]], t[ix[positions]])
        result_actions[positions], result_ids[positions] = found['actions'], found['row_ids']
        result_episodes[positions], result_distances[positions] = found['episode_ids'], found['distances']
        bandwidths[positions] = reference.action_bandwidth
        digests[str(int(f))] = reference.fingerprint_sha256
        _require(not np.isin(found['episode_ids'], ep[row_fold == f]).any(), 'cross-fit episode leakage')
    return {'actions': result_actions, 'row_ids': result_ids, 'episode_ids': result_episodes,
            'distances': result_distances, 'bandwidths': bandwidths,
            'query_folds': row_fold[ix], 'reference_sha256_by_fold': digests}


def compatibility_cost_jax(actions, neighbor_actions, bandwidth):
    """Return nonnegative KDE action costs [B], preserving live action gradient.

    Actions [B,4], neighbors [B,K,4], and bandwidth [4] or [B,4] must be in
    matching action coordinates. Neighbor actions and bandwidth are always
    stop-gradient. Use prevalidated arrays from ``fit_reference`` / queries.
    The function is safe to call inside JAX ``jit`` or from an actor loss.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    _require(actions.ndim == 2 and actions.shape[1] == 4, 'actions must be [B,4]')
    _require(neighbor_actions.ndim == 3 and neighbor_actions.shape[0] == actions.shape[0]
             and neighbor_actions.shape[1] > 0 and neighbor_actions.shape[2] == 4, 'neighbors must be [B,K,4]')
    _require(bandwidth.shape == (4,) or bandwidth.shape == actions.shape, 'bandwidth must be [4] or [B,4]')
    neighbors = jax.lax.stop_gradient(jnp.asarray(neighbor_actions, dtype=actions.dtype))
    widths = jax.lax.stop_gradient(jnp.asarray(bandwidth, dtype=actions.dtype))
    if widths.ndim == 1:
        widths = jnp.broadcast_to(widths, actions.shape)
    squared = jnp.sum(jnp.square((actions[:, None, :] - neighbors) / widths[:, None, :]), axis=-1)
    return -(logsumexp(-0.5 * squared, axis=1) - jnp.log(jnp.asarray(neighbors.shape[1], dtype=actions.dtype)))


def jax_reference(reference):
    """Prepare frozen full-bank arrays for JAX state-to-neighbor selection.

    Rows are ordered by unique original row ID to make equal-distance ties
    deterministic (JAX top_k chooses the lower array index). State distance
    uses float32 on the learner runtime; CPU ``query_neighbors`` uses float64,
    so their neighbor membership can differ for nearly tied distances. This
    adapter is intended for held-out observations or for a query batch whose
    complete episodes were excluded when its reference was fitted.
    """
    _require(isinstance(reference, Reference), 'expected fitted compatibility Reference')
    limits = np.iinfo(np.int32)
    _require(np.all((reference.episodes >= limits.min) & (reference.episodes <= limits.max)),
             'episode IDs cannot be represented exactly on the JAX int32 runtime')
    _require(np.all((reference.row_ids >= limits.min) & (reference.row_ids <= limits.max)),
             'row IDs cannot be represented exactly on the JAX int32 runtime')
    for b in range(100 // reference.time_bin):
        _require(np.count_nonzero(reference.time_bins == b) >= reference.k,
                 f'time bin {b}: fewer than K eligible retained rows')
    import jax.numpy as jnp
    order = np.argsort(reference.row_ids, kind='stable')
    return {'states': jnp.asarray(reference.states[order], dtype=jnp.float32),
            'actions': jnp.asarray(reference.actions[order], dtype=jnp.float32),
            'episode_ids': jnp.asarray(reference.episodes[order], dtype=jnp.int32),
            'row_ids': jnp.asarray(reference.row_ids[order], dtype=jnp.int32),
            'time_bins': jnp.asarray(reference.time_bins[order], dtype=jnp.int32),
            'mean': jnp.asarray(reference.state_mean, dtype=jnp.float32),
            'scale': jnp.asarray(reference.state_scale, dtype=jnp.float32),
            'bandwidth': jnp.asarray(reference.action_bandwidth, dtype=jnp.float32)}


def compatibility_from_observations_jax(reference, observations, actions, *, k=64,
                                        time_bin=10, exclude_episode_ids=None,
                                        return_neighbors=False):
    """JIT-compatible compatibility cost from raw learner observations.

    The last observation coordinate is the environment's remaining-time value
    ``1 - t/100``; the first 13 are physical coordinates. Reference arrays and
    state-based neighbor selection are stop-gradient. Per-row episode IDs may be
    supplied for whole-episode neighbor exclusion; full episode-fold cross-fit
    additionally requires fitting the reference without that fold. If a row has
    invalid time or fewer than K candidates its returned cost is NaN, causing
    downstream finiteness checks to fail rather than silently backfill.
    """
    import jax
    import jax.numpy as jnp

    _require(type(k) is int and k > 0 and type(time_bin) is int and time_bin > 0
             and 100 % time_bin == 0, 'invalid K or time-bin width')
    _require(observations.ndim == 2 and observations.shape[1] == 14, 'observations must be [B,14]')
    _require(actions.ndim == 2 and actions.shape == (observations.shape[0], 4), 'actions must be [B,4]')
    _require(reference['states'].ndim == 2 and reference['states'].shape[1] == 13
             and reference['actions'].shape == (reference['states'].shape[0], 4)
             and k <= reference['states'].shape[0], 'invalid frozen reference shapes')
    if exclude_episode_ids is not None:
        _require(exclude_episode_ids.shape == (observations.shape[0],), 'exclude IDs must be [B]')
    ref = jax.tree_util.tree_map(jax.lax.stop_gradient, reference)
    obs = jax.lax.stop_gradient(jnp.asarray(observations, dtype=jnp.float32))
    step_float = (1.0 - obs[:, -1]) * 100.0
    steps = jnp.rint(step_float).astype(jnp.int32)
    time_valid = (jnp.isfinite(step_float) & (steps >= 0) & (steps < 100)
                  & (jnp.abs(step_float - steps) <= 1e-3))
    z = (obs[:, :13] - ref['mean']) / ref['scale']
    excluded = (jnp.zeros((len(obs),), dtype=jnp.int32) if exclude_episode_ids is None
                else jax.lax.stop_gradient(jnp.asarray(exclude_episode_ids, dtype=jnp.int32)))
    do_exclude = exclude_episode_ids is not None

    def select(args):
        query, step, ex = args
        delta = ref['states'] - query[None, :]
        d2 = jnp.sum(jnp.square(delta), axis=1)
        eligible = ref['time_bins'] == step // time_bin
        if do_exclude:
            eligible = eligible & (ref['episode_ids'] != ex)
        eligible = eligible & jnp.isfinite(d2)
        _, indices = jax.lax.top_k(jnp.where(eligible, -d2, -jnp.inf), k)
        return ref['actions'][indices], ref['row_ids'][indices], jnp.sqrt(d2[indices]), jnp.sum(eligible) >= k

    # lax.map keeps the working distance matrix O(N), instead of O(B*N*13).
    neighbors, row_ids, distances, enough = jax.lax.map(select, (z, steps, excluded))
    costs = compatibility_cost_jax(actions, neighbors, ref['bandwidth'])
    valid = time_valid & enough & jnp.all(jnp.isfinite(obs), axis=1)
    costs = jnp.where(valid, costs, jnp.asarray(jnp.nan, dtype=costs.dtype))
    if return_neighbors:
        return costs, row_ids, distances, valid
    return costs
