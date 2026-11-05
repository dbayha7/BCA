"""Open questions 1 (load-band part), 2 and 3 of the confirmation v1 attempt 3 readout, from saved records only.

Readout: experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1/CF_READOUT_attempt3_20261102.md, section 4
Inputs: confirmation_v1/cf_run/attempt_3 evaluation reports, and the formal G2 v2 CUTOFF datasets
(formal_g2/formal_g2_run/attempt_1/knee_s10_v2_<world>.npz; archives 5f4933fc… stall3, 125131d7… no-stall, the ones the
runs used), read only to look up each probe row's load band by its [episode, t] index.
Usage: python analysis/confirmation_v1_kappa_profile/analysis_saved.py [attempt_3 dir] [dataset dir] [out json]
Reads saved files with numpy only; trains, evaluates and simulates nothing.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                    'experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2')
A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CAND, 'confirmation_v1/cf_run/attempt_3')
G = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CAND, 'formal_g2/formal_g2_run/attempt_1')
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, 'analysis_saved.json')
WORLDS, SEEDS, DOSES = ('stall3', 'nostall'), ('0', '1', '2'), ('21', '63', '189')
BANDS = ('low', 'mid', 'deep')


def rep(unit):
    return json.load(open(os.path.join(A, unit, 'report.json')))


def unit(world, arm, s):
    return f'eval_native_s{s}_{world}' if arm == 'native' else f'eval_bca_s{s}_k{arm}_{world}'


out = {'q1_band_exposure': {}, 'q2_w_value_by_band': {}, 'q2_per_start': {}, 'q3_native': {}}
for world in WORLDS:
    z = np.load(os.path.join(G, f'knee_s10_v2_{world}.npz'), allow_pickle=False)
    # each probe row is [episode, t], zero-based: its logged command must be the dataset's action there, exactly
    r0 = rep(unit(world, 'native', '0'))
    rows0 = np.asarray(r0['data']['rows'])
    assert np.array_equal(z['actions'][rows0[:, 0], rows0[:, 1]].astype(np.float64), np.asarray(r0['data']['logged'], np.float64))
    # ---- Q1 (band part): W exposure among executed steps, by load band (mid, deep), native and each dose
    q1 = {}
    for arm in ('native',) + DOSES:
        q1[arm] = {}
        for s in SEEDS:
            r = rep(unit(world, arm, s))['rates']
            q1[arm][s] = {b: {'steps': r['by_band'][b]['steps'], 'w': r['by_band'][b]['w_any_depth']} for b in ('mid', 'deep')}
    out['q1_band_exposure'][world] = q1
    # ---- Q2a: exact value of the withheld commands against the logged one, by the probe row's load band
    q2 = {}
    for arm in ('native',) + DOSES:
        q2[arm] = {}
        for s in SEEDS:
            r = rep(unit(world, arm, s))
            rows = np.asarray(r['data']['rows'])
            band = z['load_band'][rows[:, 0], rows[:, 1]]
            qpi = np.asarray(r['data']['qpi'], np.float64)           # (192, 4): logged, s05, s10, s20
            d = qpi[:, 1:] - qpi[:, [0]]
            q2[arm][s] = {}
            for bi, b in enumerate(BANDS):
                m = band == bi
                q2[arm][s][b] = {'n': int(m.sum()),
                                 'mean_dQ': [float(x) for x in d[m].mean(axis=0)] if m.any() else None,
                                 'share_w_better': [float(x) for x in (d[m] > 0).mean(axis=0)] if m.any() else None}
    out['q2_w_value_by_band'][world] = q2
    # ---- Q2b: per-start return change, BCA minus native, paired by start (128 starts)
    q2b = {}
    for k in DOSES:
        q2b[k] = {}
        for s in SEEDS:
            dj = np.asarray(rep(unit(world, k, s))['returns'], np.float64) - np.asarray(rep(unit(world, 'native', s))['returns'], np.float64)
            loss = -dj[dj < 0]
            worst = np.sort(dj)[: max(1, len(dj) // 10)]
            q2b[k][s] = {'mean': float(dj.mean()), 'median': float(np.median(dj)), 'share_worse': float((dj < 0).mean()),
                         'share_better': float((dj > 0).mean()), 'min': float(dj.min()), 'max': float(dj.max()),
                         'worst10pct_share_of_total_loss': float(-worst.sum() / loss.sum()) if loss.sum() > 0 else None}
    out['q2_per_start'][world] = q2b
    # ---- Q3: how near the knee each native actor sits
    q3 = {}
    for s in SEEDS:
        r = rep(unit(world, 'native', s))
        a0 = np.abs(np.asarray(r['own']['actions'], np.float64)[:, 0])
        e_own = np.asarray(r['own']['q1'], np.float64) - np.asarray(r['own']['qpi'], np.float64)
        q3[s] = {'exposure': r['rates']['w_any_depth'], 'a123_past_knee': r['rates']['a123_past_knee'],
                 'a123_in_0p7_knee': r['rates']['a123_in_0p7_knee'],
                 'own_a0_share_past_0p8': float((a0 > 0.8).mean()), 'own_a0_share_0p7_to_0p8': float(((a0 > 0.7) & (a0 <= 0.8)).mean()),
                 'own_a0_median': float(np.median(a0)), 'e_own_mean': float(e_own.mean()), 'J': r['J']}
    out['q3_native'][world] = q3

json.dump(out, open(OUT, 'w'), indent=1)
print('written', OUT)
