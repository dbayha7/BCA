"""Frozen diagnostic score controls with training-probe-only matching.

The caller supplies already frozen models, thresholds, actor actions and
episode-excluding compatibility references. This module never fits a model,
uses a simulator, or reads evaluation labels. Scaling a one-sided calibrated
penalty is an experimental score transformation, not a coverage statement.
"""
import jax
import jax.numpy as jnp
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _first(result):
    return result[0] if isinstance(result, (tuple, list)) else result


def _rms(gradient):
    value = np.asarray(gradient, dtype=np.float64)
    _require(value.ndim == 2 and value.shape[1] == 4 and np.isfinite(value).all(),
             'action gradient must be finite and have shape (N,4)')
    return float(np.sqrt(np.mean(np.sum(value ** 2, axis=1))))


def match_controls(observations, actor_actions, q_fn, width_fn, signed_fn,
                   cost_fn, neighbors, bandwidths, guard, *, max_coefficient=30.):
    """Fit frozen scalar coefficients on actor actions at permitted probes.

    ``q_fn(obs,a)`` returns a Q1 vector. ``width_fn`` returns U, or
    (U, usable, valid); ``signed_fn`` returns B or (B, mu, sigma).
    ``cost_fn(a,neighbors,bandwidths)`` returns local D. The caller must
    pre-bind episode-excluded neighbors and any nested-fold bandwidth and
    state-distance gate. Gradients are only with respect to action, and the
    state gate, neighbors, model parameters and calibrated thresholds are fixed.
    """
    obs = jnp.asarray(observations, jnp.float32)
    actions = jnp.asarray(actor_actions, jnp.float32)
    n = len(obs)
    reference = jnp.asarray(neighbors, jnp.float32)
    bandwidth = jnp.asarray(bandwidths, jnp.float32)
    gate_raw = np.asarray(guard)
    _require(n > 0 and obs.ndim == 2 and actions.shape == (n, 4), 'invalid probe shapes')
    _require(reference.ndim == 3 and reference.shape[0] == n and reference.shape[2] == 4,
             'neighbor shape must be (N,K,4)')
    _require(bandwidth.shape in ((4,), (n, 4)), 'bandwidth must be (4,) or (N,4)')
    _require(gate_raw.shape == (n,) and gate_raw.dtype.kind == 'b', 'guard must be a boolean vector')
    _require(all(np.isfinite(np.asarray(v)).all() for v in (obs, actions, reference, bandwidth)),
             'nonfinite probe input')
    _require(np.all(np.asarray(bandwidth) > 0), 'action bandwidth must be positive')
    _require(np.isfinite(max_coefficient) and max_coefficient > 0, 'invalid coefficient cap')
    gate = jnp.asarray(gate_raw, jnp.float32)
    q_values = np.asarray(_first(q_fn(obs, actions)), np.float64)
    width_result = width_fn(obs, actions)
    if isinstance(width_result, (tuple, list)) and len(width_result) == 3:
        _require(np.asarray(width_result[1]).all() and np.asarray(width_result[2]).all(),
                 'symmetric width reference unusable or invalid')
    widths = np.asarray(_first(width_result), np.float64)
    bounds = np.asarray(_first(signed_fn(obs, actions)), np.float64)
    costs = np.asarray(cost_fn(actions, reference, bandwidth), np.float64)
    for label, value in [('q', q_values), ('width', widths), ('signed', bounds), ('cost', costs)]:
        _require(value.shape == (n,) and np.isfinite(value).all(), 'invalid probe ' + label)
    _require(np.all(widths >= 0) and np.all(bounds >= 0), 'penalties must be nonnegative')
    _require(np.all(costs >= -1e-6), 'local kernel cost must be nonnegative')
    qg = jax.grad(lambda a: jnp.sum(_first(q_fn(obs, a))))(actions)
    wg = jax.grad(lambda a: jnp.sum(_first(width_fn(obs, a))))(actions)
    bg = jax.grad(lambda a: jnp.sum(_first(signed_fn(obs, a))))(actions)
    dg = jax.grad(lambda a: jnp.sum(cost_fn(a, reference, bandwidth) * gate))(actions)
    qr, wr, br, dr = map(_rms, (qg, wg, bg, dg))
    signed_mean, sym_mean = float(bounds.mean()), float(3 * widths.mean())
    _require(all(np.isfinite(v) and v > 1e-6 for v in (qr, wr, br, dr, signed_mean, sym_mean)),
             'matching requires finite nonzero Q, width, signed, and gated cost signals')
    eta = .25 * qr / dr
    mean_factor = sym_mean / signed_mean
    gradient_factor = 3 * wr / br
    hybrid_kappa = _rms(3 * np.asarray(wg) + eta * np.asarray(dg)) / wr
    for name, value in [('eta', eta), ('lambda_mean', mean_factor),
                        ('lambda_gradient', gradient_factor), ('hybrid_matched_kappa', hybrid_kappa)]:
        _require(np.isfinite(value) and 0 < value <= max_coefficient, name + ' outside (0, cap]')
    return {'lambda_mean': mean_factor, 'lambda_gradient': gradient_factor,
            'eta': eta, 'hybrid_matched_kappa': hybrid_kappa,
            'q_action_gradient_rms': qr, 'width_action_gradient_rms': wr,
            'signed_action_gradient_rms': br, 'guarded_cost_action_gradient_rms': dr,
            'symmetric_penalty_mean': sym_mean, 'signed_penalty_mean': signed_mean,
            'guarded_probe_count': int(gate_raw.sum()), 'probe_count': n,
            'max_coefficient': float(max_coefficient),
            'matching_data_role': 'training probes; caller must supply fold-excluding compatibility references',
            'interpretation': 'scaled score controls; no conformal or true-Q coverage claim'}


def apply_scores(raw_q1, width, signed_bound, compatibility_cost, guard, matching):
    """Calculate identical-action score variants with frozen scalar controls."""
    q, u, b, d = (np.asarray(x, np.float64) for x in
                   (raw_q1, width, signed_bound, compatibility_cost))
    g = np.asarray(guard)
    _require(q.size > 0 and all(x.shape == q.shape and np.isfinite(x).all() for x in (u, b, d)),
             'scores need aligned finite per-action vectors')
    _require(np.isfinite(q).all() and g.shape == q.shape and g.dtype.kind == 'b',
             'invalid Q or guard vector')
    _require(np.all(u >= 0) and np.all(b >= 0) and np.all(d >= -1e-6), 'invalid penalty')
    keys = ('lambda_mean', 'lambda_gradient', 'eta', 'hybrid_matched_kappa')
    _require(all(k in matching and np.isfinite(matching[k]) and matching[k] > 0 for k in keys),
             'invalid or incomplete matching receipt')
    dg = np.where(g, d, 0.)
    eta = matching['eta']
    scores = {'raw_q1': q, 'q1_aligned': q - 3 * u, 'signed_bayes': q - b,
              'signed_mean_matched': q - matching['lambda_mean'] * b,
              'signed_gradient_matched': q - matching['lambda_gradient'] * b,
              'revised_compatibility': q - eta * dg,
              'sym_compat_guarded': q - 3 * u - eta * dg,
              'sym_compat_unguarded': q - 3 * u - eta * d,
              'sym_hybrid_strength_matched': q - matching['hybrid_matched_kappa'] * u,
              'signed_plus_compat': q - b - eta * dg}
    _require(all(np.isfinite(x).all() for x in scores.values()), 'nonfinite score')
    return scores
