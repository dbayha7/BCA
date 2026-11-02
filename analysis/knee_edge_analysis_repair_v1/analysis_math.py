"""Isolated, array-only exploratory analysis. No imports from experiment code."""
import struct

import numpy as np


class InvalidData(ValueError):
    pass


SLOT_BITS = {
    "edge_knee": 0xBF4CCCCD, "edge_775": 0xBF466666, "edge_750": 0xBF400000,
    "edge_725": 0xBF39999A, "edge_700": 0xBF333333, "mid_m40": 0xBECCCCCD,
    "zero": 0, "mid_p40": 0x3ECCCCCD, "knee": 0x3F4CCCCD,
    "s05": 0xBF59999A, "s10": 0xBF666667, "s15": 0xBF733333,
    "s20": 0xBF800000, "p10": 0x3F666667,
}
SLOTS = tuple(SLOT_BITS)
S9 = SLOTS[:9]
X = {k: struct.unpack('>f', v.to_bytes(4, 'big'))[0] for k, v in SLOT_BITS.items()}
TIMES = [t for t in (0, 25, 50, 75) for _ in range(16)]
ITEM5_TOL = {"s05": 0.10, "s10": 0.15, "s20": 0.30}
IDENTITY_ATOL, IDENTITY_RTOL = 1e-10, 1e-12


def require(ok, message):
    if not ok:
        raise InvalidData(message)


def vector(value, name, size=None):
    a = np.asarray(value)
    require(a.dtype.kind in 'iuf' and a.ndim == 1, f'{name}: numeric vector required')
    a = a.astype(np.float64)
    require(np.isfinite(a).all(), f'{name}: nonfinite values')
    require(size is None or len(a) == size, f'{name}: wrong length')
    return a


def matrix(value, name):
    a = np.asarray(value)
    require(a.dtype.kind in 'iuf' and a.shape == (64, 14), f'{name}: numeric (64,14) required')
    a = a.astype(np.float64)
    require(np.isfinite(a).all(), f'{name}: nonfinite values')
    return a


def bools(value, name, size):
    a = np.asarray(value)
    require(a.dtype.kind == 'b' and a.shape == (size,), f'{name}: {size} booleans required')
    return a


def receipt(value, clipped, world_name='stall3'):
    require(isinstance(value, dict), 'Missing Q-pi receipt')
    require(value.get('evaluator') == 'raw_dynamics_loop', 'Wrong Q-pi evaluator')
    require(value.get('gamma') == 0.98 and value.get('horizon') == 100, 'Wrong discount/horizon')
    require(value.get('bootstrap') == 'none after step 99', 'Wrong termination/bootstrap')
    expected_clip = 'all four commands to +-KNEE' if clipped else 'none (deployed command, |a| <= 1 asserted)'
    require(value.get('clip') == expected_clip, 'Wrong continuation policy')
    world = value.get('continuation', {})
    expected = {'stall': world_name == 'stall3', 'stall_slope': 3.0, 'knee': X['knee'], 'gain': 0.3,
                'throttle_cost': 0.3, 'wave_turn': 0.27, 'gamma': 0.98, 'length': 100}
    require(isinstance(world, dict) and world.get('world') == world_name
            and world.get('physics') == expected, 'Wrong continuation physics')


def anchors(value):
    """Validate the actual saved Stage 1 20k E2 layout; never reorder/filter by outcomes."""
    require(isinstance(value, dict), 'Missing anchors_e2')
    require(value.get('gated_slots') == list(SLOTS), 'Wrong/duplicated/reordered slots')
    require(value.get('nostall_slots') == list(SLOTS), 'Wrong no-stall slots')
    times = vector(value.get('times'), 'times', 64)
    require(np.array_equal(times, TIMES), 'Wrong anchor times/order')
    result = {k: matrix(value.get(k), k) for k in ('q1', 'qpi_clip', 'qpi_unclipped', 'qpi_nostall')}
    result['valid'] = bools(value.get('valid'), 'valid', 64)
    receipt(value.get('receipt_clip'), True)
    receipt(value.get('receipt_unclipped'), False)
    receipt(value.get('receipt_nostall'), True, 'nostall')
    return result


def predictor_quantities(a):
    """Reconstruct the existing three deterministic polynomial summaries from cached values.

    This is not an auxiliary-model fit: no transitions, model parameters or calibration
    are loaded. Float32 slot bits and NumPy least-squares order match step04_math.py.
    """
    q = a['qpi_clip'][:48]
    ns = a['qpi_nostall'][:48]
    support = q[:, :9]
    coefficients = []
    for cols, degree in ((list(range(9)), 2), (list(range(5)), 1), (list(range(9)), 3)):
        xs = np.asarray([X[S9[i]] for i in cols], np.float64)
        design = np.stack([xs ** power for power in range(degree + 1)], axis=1)
        coef = np.linalg.lstsq(design, support[:, cols].T, rcond=None)[0].T
        coefficients.append(coef)
    rho, errors = {}, {}
    for slot in ITEM5_TOL:
        pred = np.stack([sum(coef[:, power] * X[slot] ** power for power in range(coef.shape[1]))
                         for coef in coefficients])
        j = SLOTS.index(slot)
        rho[slot] = np.min(pred - q[:, j], axis=0)
        errors[slot] = np.max(np.abs(pred - ns[:, j]), axis=0)
    eligible = np.logical_and.reduce([errors[k] <= tol for k, tol in ITEM5_TOL.items()])
    return {'rho': rho, 'errors': errors, 'item5_ok': eligible}


def v0f(rho, kc, common_valid, random_item5):
    """Exactly the one-sided written V0(f); V0(c)'s sample floor is a separate rule."""
    rho = vector(rho, 'rho')
    kc = vector(kc, 'kc', len(rho))
    common = bools(common_valid, 'common_valid', len(rho))
    item5 = bools(random_item5, 'random_item5', len(rho))
    selected = common & item5
    out = {'n_gated': len(rho), 'n_common': int(common.sum()), 'n_r_valid': int(selected.sum()),
           'row_indices': np.flatnonzero(selected).tolist(), 'median_rho': None, 'median_kc': None,
           'bound': None, 'floor_passed': False, 'bound_passed': False, 'status': 'INCONCLUSIVE'}
    if not selected.any():
        out['reason'] = 'No R-valid anchors; cannot establish V0(f)'
        return out
    r, k = float(np.median(rho[selected])), float(np.median(kc[selected]))
    bound = 0.3 * (r + 0.15)
    floor_ok, bound_ok = r >= 0.30, k <= bound
    out.update(median_rho=r, median_kc=k, bound=bound, floor_passed=floor_ok, bound_passed=bound_ok,
               status='PASS' if floor_ok and bound_ok else 'INCONCLUSIVE')
    return out


def decomposition(q1, clipped, own, edge=0):
    """Per-anchor identity; no assumption that medians or cross-arm gaps cancel."""
    arrays = [np.asarray(a, np.float64) for a in (q1, clipped, own)]
    require(all(a.ndim == 2 and a.shape == arrays[0].shape and np.isfinite(a).all() for a in arrays),
            'Decomposition requires same finite matrix shapes')
    require(arrays[0].shape[1] > edge >= 0, 'Invalid edge index')
    dq, dc, du = [a - a[:, edge:edge + 1] for a in arrays]
    kc_clip, kc_own, gap = dq - dc, dq - du, du - dc
    residual = kc_clip - (kc_own + gap)
    require(np.allclose(kc_clip, kc_own + gap, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL),
            'Per-anchor decomposition identity failed')
    return {'kc_clip': kc_clip, 'kc_own': kc_own, 'continuation_gap': gap,
            'max_identity_residual': float(np.max(np.abs(residual))) if residual.size else 0.0}


def summary(values, mask):
    a = values[mask]
    return {'n': int(len(a)), 'median': float(np.median(a)) if len(a) else None,
            'mean': float(np.mean(a)) if len(a) else None}


def analyze_seed(seed, cutoff, random):
    require(type(seed) is int and seed in (0, 1, 2), 'Unexpected seed')
    arms = {'cutoff': anchors(cutoff), 'random': anchors(random)}
    common = arms['cutoff']['valid'][:48] & arms['random']['valid'][:48]
    quantities = {name: predictor_quantities(a) for name, a in arms.items()}
    dec = {name: decomposition(a['q1'][:48], a['qpi_clip'][:48], a['qpi_unclipped'][:48])
           for name, a in arms.items()}
    s10 = SLOTS.index('s10')
    gate = v0f(quantities['random']['rho']['s10'], dec['random']['kc_clip'][:, s10],
              common, quantities['random']['item5_ok'])
    companion = v0f(quantities['cutoff']['rho']['s10'], dec['cutoff']['kc_clip'][:, s10],
                   common, quantities['cutoff']['item5_ok'])
    companion['meaning'] = 'Report-only CUTOFF companion; not an added criterion'
    output = {'seed': seed, 'v0f_random': gate, 'cutoff_companion': companion,
              'n_common': int(common.sum()), 'slots': {}, 'predictor_rows': {}}
    for arm in arms:
        output['predictor_rows'][arm] = {'rho_s10': quantities[arm]['rho']['s10'].tolist(),
                                        'item5_ok': quantities[arm]['item5_ok'].tolist(),
                                        'item5_error': {k: v.tolist() for k, v in quantities[arm]['errors'].items()}}
    for slot in ('s05', 's10', 's20'):
        j = SLOTS.index(slot)
        paired = {key: dec['cutoff'][key][:, j] - dec['random'][key][:, j]
                  for key in ('kc_clip', 'kc_own', 'continuation_gap')}
        require(np.allclose(paired['kc_clip'], paired['kc_own'] + paired['continuation_gap'],
                            atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL), 'Paired decomposition identity failed')
        output['slots'][slot] = {
            'arms': {arm: {key: summary(dec[arm][key][:, j], common)
                           for key in ('kc_clip', 'kc_own', 'continuation_gap')} for arm in arms},
            'cutoff_minus_random': {k: summary(v, common) for k, v in paired.items()},
            'rows': [{'index': i, 'time': TIMES[i], 'common_valid': bool(common[i]),
                      'cutoff': {k: float(dec['cutoff'][k][i, j]) for k in paired},
                      'random': {k: float(dec['random'][k][i, j]) for k in paired},
                      'cutoff_minus_random': {k: float(v[i]) for k, v in paired.items()}} for i in range(48)],
        }
    return output


def aggregate(results):
    require(len(results) == 3 and sorted(x['seed'] for x in results) == [0, 1, 2], 'Exactly three distinct seeds required')
    passed = all(x['v0f_random']['status'] == 'PASS' for x in results)
    return {'v0f_all_seeds_pass': passed,
            'v0f_only_status': 'PASS' if passed else 'INCONCLUSIVE',
            'protocol_decision': None, 'can_unlock_stage2': False,
            'scope': 'Amended exploratory diagnostic only; no original label is replaced',
            'per_seed': sorted(results, key=lambda x: x['seed'])}
