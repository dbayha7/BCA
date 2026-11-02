"""Fixed decision arithmetic only. No files, models, training or simulation."""
import numpy as np
import geometry


def _ratio(a, b):
    return float(a / b) if b else None


def analyze_geometry(train_obs, train_actions, row_episode_ids, times, masks,
                     control_receipt, metric, support, calplan, anchor_obs, *, contract):
    g, s = contract['global_coverage'], contract['support']
    region = geometry.region_masks(train_obs, train_actions)
    hole, high, band = region['hole'], region['high'], region['in_band']
    counts = {
        'original_hole_rows': int(hole.sum()),
        'original_hole_episodes': int(len(np.unique(row_episode_ids[hole]))),
        'hole_rows_by_ten_step_bin': [int(np.sum(hole & (times // 10 == j))) for j in range(10)],
        'low_load_band_rows': int(np.sum(~high & band)),
        'hole_rows_by_load_depth': [int(np.sum(hole & (region['load'] >= lo) & (region['load'] < hi)))
                                  for lo, hi in zip(geometry.LOAD_DEPTH_EDGES,
                                                    geometry.LOAD_DEPTH_EDGES[1:] + (np.inf,))],
        'retained_rows': {k: int(v.sum()) for k, v in masks.items()},
        'random_retained_hole_rows': int(np.sum(masks['random'] & hole)),
        'hole_retained_forbidden_rows': int(np.sum(masks['hole'] & hole)),
    }
    z = np.asarray(train_obs, np.float32).astype(np.float64)[:, :13]
    delta = (z[masks['hole']].mean(axis=0) - z[masks['random']].mean(axis=0)) / metric['scale']
    if not np.all(np.isfinite(delta)):
        raise ValueError('Invalid state-balance denominator or empty condition')
    local = []
    for item in support['anchors']:
        full, kept = item['counts']['full'], item['counts']['hole']
        passed = (item['locality_passed']
                  and full['rows'] >= s['min_rows'] and full['episodes'] >= s['min_episodes']
                  and kept['rows'] >= s['min_rows'] and kept['episodes'] >= s['min_episodes']
                  and full['in_band_rows'] >= s['min_original_band_rows']
                  and full['in_band_episodes'] >= s['min_original_band_episodes']
                  and all(kept[side + '_alternative_rows'] >= s['min_each_adjacent_rows']
                          and kept[side + '_alternative_episodes'] >= s['min_each_adjacent_episodes']
                          for side in ('left', 'right')))
        local.append({'anchor_index': item['anchor_index'], 'high_load': item['high_load'],
                      'locality_passed': bool(item['locality_passed']), 'passed': bool(passed)})
    by_load = {}
    for name, flag in [('high', True), ('low', False)]:
        selected = [v for v in local if v['high_load'] == flag]
        passing = sum(v['passed'] for v in selected)
        by_load[name] = {'anchors': len(selected), 'passing': passing, 'fraction': _ratio(passing, len(selected))}
    criteria = {
        'original_hole_rows': counts['original_hole_rows'] >= g['min_original_hole_rows'],
        'original_hole_episodes': counts['original_hole_episodes'] >= g['min_original_hole_episodes'],
        'time_bin_coverage': min(counts['hole_rows_by_ten_step_bin']) >= g['min_hole_rows_each_ten_step_bin'],
        'low_load_retained_band': counts['low_load_band_rows'] >= g['min_low_load_band_rows'],
        'load_depth_coverage': min(counts['hole_rows_by_load_depth']) >= g['min_hole_rows_each_load_depth'],
        'zero_forbidden_hole_replay_labels': counts['hole_retained_forbidden_rows'] == 0,
        'zero_forbidden_shared_calibration_labels': calplan['hole_rows_used'] == 0,
        'all_calibration_episodes_eligible': not calplan['empty_eligible_episode_ids'],
        'exact_random_control_counts': all(x['hole_retained'] == x['random_retained'] for x in control_receipt['strata']),
        'state_mean_balance': float(np.max(np.abs(delta))) <= g['max_hole_random_state_mean_difference_in_original_sd'],
    }
    for name, x in by_load.items():
        criteria[name + '_anchors'] = x['anchors'] >= s['min_anchors_each_load_side']
        criteria[name + '_local_support'] = x['fraction'] is not None and x['fraction'] >= s['min_passing_fraction_each_load_side']
    return {'passed': all(criteria.values()), 'criteria': criteria, 'counts': counts,
            'local_passes': local, 'support_by_load': by_load,
            'hole_random_standardized_mean_difference': delta.tolist(),
            'max_abs_state_mean_difference': float(np.max(np.abs(delta))),
            'conditional_claim': 'Constructed marginal-a0 hole with measured local state neighborhoods; not certified joint4D action support',
            'missing_label_causal_effect': 'not tested; no condition-specific models fitted'}


def _tolerance(values, contract):
    t = contract['decision']['q_error_tolerance']
    return t['absolute'] + t['relative_to_abs_reference'] * np.abs(values)


def _errors(q, v, contract):
    error = q - v
    tau = _tolerance(v, contract)
    over, under = error > tau, error < -tau
    def mean_or_none(x):
        return float(np.mean(x)) if len(x) else None
    return {'rows': int(len(error)), 'over': int(over.sum()), 'under': int(under.sum()),
            'near_zero': int((~(over | under)).sum()),
            'mae': mean_or_none(np.abs(error)), 'mean_signed_error': mean_or_none(error),
            'mean_overestimation_magnitude': mean_or_none(error[over]),
            'mean_underestimation_magnitude': mean_or_none(-error[under]),
            'min_signed_error': float(error.min()) if len(error) else None,
            'max_signed_error': float(error.max()) if len(error) else None}


def _first_max_with_tolerance(rows, contract):
    maximum = rows.max(axis=1, keepdims=True)
    tied = (maximum - rows) <= np.maximum(_tolerance(maximum, contract), _tolerance(rows, contract))
    return np.argmax(tied, axis=1)


def analyze_values(values, q1, banks, anchor_obs, free_result, contract):
    values, q1, banks, obs = map(np.asarray, (values, q1, banks, anchor_obs))
    if values.shape != (64, 6) or q1.shape != values.shape or banks.shape != (64, 6, 4):
        raise ValueError('Frozen six-candidate/64-anchor bank changed')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(q1)):
        raise ValueError('Nonfinite reference or critic score')
    high = geometry.region_masks(obs, banks[:, 0])['high']
    gap = np.maximum(np.abs(values[:, 3] - values[:, 1]), np.abs(values[:, 3] - values[:, 5]))
    within = np.ptp(values[:, [2, 3, 4]], axis=1)
    passing = gap >= contract['decision']['practical_center_flank_gap']
    n_high = int(high.sum())
    fraction = _ratio(int(np.sum(high & passing)), n_high)
    free_den = free_result['high_load_action_rows']
    free_fraction = _ratio(free_result['hole_entries'], free_den)
    criteria = {
        'enough_high_load_anchors': n_high >= contract['support']['min_anchors_each_load_side'],
        'practical_center_flank_gap': fraction is not None and fraction >= contract['decision']['min_high_load_gap_fraction'],
        'free_high_load_steps': free_den >= contract['decision']['min_free_high_load_steps'],
        'free_high_load_entry_fraction': free_fraction is not None and free_fraction >= contract['decision']['min_free_high_load_entry_fraction'],
        'free_entry_episodes': free_result['episodes_with_hole_entry'] >= contract['decision']['min_free_entry_episodes'],
    }
    region = geometry.region_masks(np.repeat(obs, 6, axis=0), banks.reshape(-1, 4))
    cells = {}
    for load_name, load_mask in [('high', region['high']), ('low', ~region['high'])]:
        for band_name, band_mask in [('inside_band', region['in_band']), ('outside_band', ~region['in_band'])]:
            mask = load_mask & band_mask
            cells[load_name + '_' + band_name] = _errors(q1.ravel()[mask], values.ravel()[mask], contract)
    selected = _first_max_with_tolerance(q1, contract)
    selected_v = values[np.arange(64), selected]
    regret = values.max(axis=1) - selected_v
    gain = selected_v - values[:, 0]
    tau = np.maximum(_tolerance(selected_v, contract), _tolerance(values[:, 0], contract))
    errors, error_tau = q1-values, _tolerance(values, contract)
    per_probe = [{'anchor_index': i, 'candidate': j, 'high_load': bool(high[i]),
                  'in_band': bool(region['in_band'].reshape(64, 6)[i,j]),
                  'q1': float(q1[i,j]), 'qpi': float(values[i,j]), 'signed_error': float(errors[i,j]),
                  'absolute_error': float(abs(errors[i,j])), 'tolerance': float(error_tau[i,j]),
                  'error_class': 'over' if errors[i,j] > error_tau[i,j] else 'under' if errors[i,j] < -error_tau[i,j] else 'near_zero'}
                 for i in range(64) for j in range(6)]
    return {'passed': all(criteria.values()), 'criteria': criteria,
            'high_load_anchors': n_high, 'high_load_gap_fraction': fraction,
            'center_flank_gap_per_anchor': gap.tolist(), 'within_band_spread_per_anchor': within.tolist(),
            'free_actor': {k: v for k, v in free_result.items() if k not in ('observations', 'actions', 'returns', 'episode_seeds')},
            'free_high_load_entry_fraction': free_fraction,
            'free_actor_returns': np.asarray(free_result['returns']).tolist(),
            'forced_center_actions_in_band': int(region['in_band'].reshape(64, 6)[:, 3].sum()),
            'actor_at_logged_anchor_hole_entries': int(region['hole'].reshape(64, 6)[:, 0].sum()),
            'per_probe_errors': per_probe,
            'error_cells': cells, 'raw_q_errors_all': _errors(q1.ravel(), values.ravel(), contract),
            'raw_q_decisions': {'selected_candidate': selected.tolist(), 'mean_regret': float(regret.mean()),
                'regret_per_anchor': regret.tolist(), 'gain_over_actor_per_anchor': gain.tolist(),
                'mean_gain_over_actor': float(gain.mean()), 'improved': int((gain > tau).sum()),
                'worsened': int((gain < -tau).sum()), 'tied': int((np.abs(gain) <= tau).sum()),
                'harmful_switches': int(np.sum((selected != 0) & (gain < -tau))),
                'practical_regret_rows': int(np.sum(regret >= contract['decision']['regret_practical_threshold']))},
            'interpretation': 'Same broad-trained native critic and its actor only. No BCA comparison or missing-label causal estimate.'}
