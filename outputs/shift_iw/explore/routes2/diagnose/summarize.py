"""EXPLORATORY: gathers the diagnosis numbers (seed-0 reads in this folder, plus the 5-seed records in the repository) into
diagnosis_numbers.json. Reads only; no training."""
import json
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
REPO = Path('/home/user/bca')
rows = json.load(open(H / 'diag_rows.json'))['by_quality']
rolls = json.load(open(H / 'diag_rollouts.json'))['res']
band = json.load(open(H / 'diag_band_at_commands.json'))['res']
chunks = json.load(open(H / 'dose_chunks.json'))
lc = json.load(open(H / 'lc_data.json'))['cotrain']
Q = ['random', 'medium-replay', 'medium', 'medium-expert', 'expert']
SITS = ['recorded', 'big_kicks', 'big_waves', 'strong_current']
out = dict(label='EXPLORATORY: seed 0 of the cloud rerun unless marked 5 seeds; no training, no band fit', qualities={})
for q in Q:
    rep = json.load(open(REPO / 'outputs/shift_iw/v5' / q / 'report.json'))['marks']
    r1, m2 = rep['R1'], rep['M2']
    h = np.array(lc['arms']['host_gap'][q]['pts']['strong']); b = np.array(lc['arms']['capped_unweighted'][q]['pts']['strong'])
    r = rows[q]
    out['qualities'][q] = dict(
        random_share_of_kept_rows=r['random_share'],
        dose_mean_seed0=r['dose']['mean'], dose_sd_seed0=r['dose']['sd'], dose_p5_p95_seed0=[r['dose']['pct']['5'], r['dose']['pct']['95']],
        eta_p5_p95_seed0=[r['eta']['pct']['5'], r['eta']['pct']['95']], equivalent_alpha_seed0=r['dose']['equivalent_alpha'],
        dose_random_rows=r['dose_by']['random']['dose_mean'], dose_recorder_rows=r['dose_by']['recorder']['dose_mean'],
        dose_iw_end_seed0=chunks[f'{q}/exact_strong']['dose'][-1], dose_unweighted_end_seed0=chunks[f'{q}/unweighted']['dose'][-1],
        shrink_k_on_training_states_seed0=r['commands']['shrink_bca_vs_host'],
        t1_hard_water_host_vs_bca_seed0=[r['commands']['t1_host_hard'], r['commands']['t1_bca_hard']],
        situations_seed0={s: dict(points_host=rolls[f'{q}|host_gap|{s}']['points'], points_bca=rolls[f'{q}|bca_unweighted|{s}']['points'],
                                  dJ=rolls[f'{q}|bca_unweighted|{s}']['J'] - rolls[f'{q}|host_gap|{s}']['J'],
                                  d_drift_cost=-(rolls[f'{q}|bca_unweighted|{s}']['drift'] - rolls[f'{q}|host_gap|{s}']['drift']),
                                  d_energy_cost=-(rolls[f'{q}|bca_unweighted|{s}']['energy'] - rolls[f'{q}|host_gap|{s}']['energy']),
                                  shrink_k_on_host_states=rolls[f'{q}|bca_unweighted|{s}']['on_host_states']['k'],
                                  past80_host=rolls[f'{q}|host_gap|{s}']['past80_any'], past80_bca=rolls[f'{q}|bca_unweighted|{s}']['past80_any'])
                        for s in SITS},
        band_full100_vs_hold80_seed0=band[q]['full100_vs_hold80'],
        host_q_full100_minus_hold80_seed0=band[q]['full100']['q_host'] - band[q]['hold80']['q_host'],
        curves_5seeds=dict(max_abs_gap_before_first_refresh=float(np.abs(b[:, :2] - h[:, :2]).max()),
                           gap_100k_strong_points=(b[:, -1] - h[:, -1]).tolist()),
        room_check_5seeds=dict(placed_minus_none=r1['placed_minus_none']['mean'],
                               placed_minus_matched_even=(r1['placed_minus_matched_even'] or {}).get('mean'),
                               matched_even_found=r1['matched_even_found_per_seed']),
        band_tracking_5seeds=dict(width_vs_exact_error_within_deciles=m2['within_deciles']['mean'],
                                  width_vs_noise_map_within_deciles=m2['noise_map_within_deciles']['mean']))
(H / 'diagnosis_numbers.json').write_text(json.dumps(out, indent=1))
for q, v in out['qualities'].items():
    print(q, json.dumps({k: v[k] for k in ['room_check_5seeds', 'band_tracking_5seeds', 'host_q_full100_minus_hold80_seed0']}, default=lambda x: round(x, 3)))
