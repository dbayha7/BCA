"""Outcome-blind v2 preparation. Support-near/distant are empirical proxies.

Only numeric arrays are accepted. No critic, width, return, policy or simulator
argument exists. Unit native bounds require separate live-runtime acceptance.
"""
import hashlib
import json
import numpy as np

NAMESPACE = 'bca-action-support-robustness-v2/2026-09-28'
SCALES = (.01, .025, .05, .10, .20, .40, .80, 1.60)


def _matrix(value):
    a = np.asarray(value)
    if (a.dtype.kind not in 'fi' or a.ndim != 2 or not 0 < a.shape[0] <= 32768
            or not 0 < a.shape[1] <= 64 or not np.isfinite(a).all()):
        raise ValueError('Bounded finite numeric matrix required.')
    return a


def thresholds(validation_distances):
    d = np.asarray(validation_distances)
    if (d.ndim != 1 or not 2 <= len(d) <= 100_000 or d.dtype.kind not in 'fi'
            or not np.isfinite(d).all() or (d < 0).any()):
        raise ValueError('Invalid training-only calibration distances.')
    q95, q99 = np.quantile(d.astype(np.float64), [.95, .99], method='linear')
    if not 0 < q95 < q99:
        raise ValueError('Degenerate support boundaries; no quota-driven correction.')
    return float(q95), float(q99)


def band(distance, q95, q99):
    if not np.isfinite([distance, q95, q99]).all() or not 0 < q95 < q99 or distance < 0:
        raise ValueError('Invalid support band input.')
    return 'near' if distance <= q95 else 'moderate' if distance <= q99 else 'strong'


def applied_unit(sent):
    a = np.asarray(sent)
    if a.dtype != np.float32 or not np.isfinite(a).all() or (abs(a) > 1).any():
        raise ValueError('Finite float32 sent action within unit bounds required.')
    lo, hi = np.full_like(a, -1), np.full_like(a, 1)
    return np.clip(lo+(a+np.float32(1))*np.float32(.5)*(hi-lo), lo, hi)


def distances(applied, neighboring_actions):
    a, neighbors = _matrix(applied), _matrix(neighboring_actions)
    if a.shape[1] != neighbors.shape[1] or len(neighbors) > 32 or (abs(a) > 1).any() or (abs(neighbors) > 1).any():
        raise ValueError('Expected <=32 state-neighbor native actions.')
    return np.sqrt(np.square(a[:, None, :].astype(np.float64)-neighbors[None, :, :]).mean(axis=2)).min(axis=1)


def stream_seed(host, environment, training_seed, purpose, index):
    if (host not in ('td3_bc', 'rebrac') or environment not in ('hopper', 'walker2d')
            or type(training_seed) is not int or training_seed not in range(202609171, 202609176)
            or purpose not in ('reset', 'candidate', 'continuation', 'random_score')
            or type(index) is not int or not 0 <= index < (64 if purpose == 'reset' else 256)):
        raise ValueError('Invalid fixed protocol stream identifier.')
    raw = json.dumps([NAMESPACE, host, environment, training_seed, purpose, index], separators=(',', ':')).encode()
    original = int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big')
    if (host, environment, training_seed, purpose, index) == ('td3_bc', 'walker2d', 202609171, 'candidate', 54):
        if original != 404735174:
            raise ValueError('Original approved stream identity changed.')
        return 404735181
    return original


def proposal_pool(anchor, seed):
    anchor = np.asarray(anchor)
    if anchor.ndim != 1 or not 0 < len(anchor) <= 64 or not np.isfinite(anchor).all() or (abs(anchor) > 1).any():
        raise ValueError('Invalid anchor.')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid candidate seed.')
    rng = np.random.Generator(np.random.PCG64(seed))
    uniform = rng.uniform(-1, 1, (4096, len(anchor)))
    direction = rng.normal(size=(4096, len(anchor)))
    rms = np.sqrt(np.square(direction).mean(axis=1))
    if (rms == 0).any() or not np.isfinite(rms).all():
        raise ValueError('Invalid direction; no silent resampling.')
    scale = np.asarray(SCALES)[np.arange(4096) % len(SCALES)]
    local = anchor.astype(np.float64)+direction/rms[:, None]*scale[:, None]
    proposed = np.empty((8192, len(anchor)), np.float64)
    proposed[::2], proposed[1::2] = uniform, local
    sent = np.clip(proposed, -1, 1).astype(np.float32)
    return dict(proposed=proposed, sent=sent, applied=applied_unit(sent), seed=seed)


def select_bands(anchor_sent, pool, neighboring_actions, q95, q99):
    anchor_sent = np.asarray(anchor_sent)
    if anchor_sent.ndim != 1:
        raise ValueError('Anchor must be one action.')
    anchor = applied_unit(anchor_sent)
    proposed, sent, applied = (pool[k] for k in ('proposed', 'sent', 'applied'))
    if (_matrix(proposed).shape != (8192, len(anchor)) or sent.shape != proposed.shape
            or applied.shape != proposed.shape or proposed.dtype != np.float64
            or not np.array_equal(sent, np.clip(proposed, -1, 1).astype(np.float32))
            or not np.array_equal(applied, applied_unit(sent))):
        raise ValueError('Pool coordinates disagree.')
    anchor_distance = float(distances(anchor[None], neighboring_actions)[0])
    if band(anchor_distance, q95, q99) != 'near':
        raise ValueError('Nearest-recorded anchor is not support-near after native transform.')
    ds = distances(applied, neighboring_actions)
    selected = {name: [] for name in ('near', 'moderate', 'strong')}
    selected['near'].append(dict(proposal_index=-1, sent=anchor_sent.tolist(), applied=anchor.tolist(), distance=anchor_distance, anchor=True))
    seen = {tuple(float(x) for x in anchor)}
    duplicate_proposals = 0
    for i, action in enumerate(applied):
        key = tuple(float(x) for x in action)
        if key in seen:
            duplicate_proposals += 1
            continue
        seen.add(key)
        label = band(float(ds[i]), q95, q99)
        if len(selected[label]) < 4:
            selected[label].append(dict(proposal_index=i, sent=sent[i].tolist(), applied=action.tolist(), distance=float(ds[i]), anchor=False))
    missing = {name: 4-len(rows) for name, rows in selected.items()}
    return dict(selected=selected, missing=missing, complete=not any(missing.values()),
                duplicate_proposals=duplicate_proposals, pool_size=len(applied),
                empirical_support_only=True, outcome_access=False)


def attach_policy_slots(selection, host_sent, bca_sent):
    """Preserve native choices and exact aliases; do not simulate aliases twice."""
    rows = [dict(row, role=name) for name in ('near', 'moderate', 'strong') for row in selection['selected'][name]]
    for name, action in [('host_policy', host_sent), ('bca_policy', bca_sent)]:
        action = np.asarray(action)
        if action.ndim != 1 or len(action) != len(rows[0]['applied']):
            raise ValueError('Policy action shape mismatch.')
        applied = applied_unit(action)
        rows.append(dict(role=name, sent=action.tolist(), applied=applied.tolist()))
    for i, row in enumerate(rows):
        row['alias_of'] = next((j for j in range(i) if row['applied'] == rows[j]['applied']), i)
    return rows

# Approved prospective exception, not automatic collision repair.
AMENDMENT_SHA256 = '188395c83b3558c99a8671146128c1e58ee189fa0e19f1715cca08ee7fe4359b'
