"""EXPLORATORY diagnosis (seed 0 of the cloud rerun, no training): the capped pull's mean dose (BC multiplier), the
TD3+BC lambda, the critic's Q at the actor's command and the BC loss, per 5,000-update chunk, read from each run's
record.json. Also the host's chunks where present."""
import json, sys
from pathlib import Path
RUNS = Path('/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun/repo/runs')
out = {}
for q in ['random', 'medium-replay', 'medium', 'medium-expert', 'expert']:
    for arm in ['unweighted', 'exact_strong']:
        p = RUNS / 'shift_iw_v5_explore' / q / 'bca' / arm / 'seed0' / 'record.json'
        r = json.load(open(p))['result']
        ch = r['chunks']
        thr = [x.get('unweighted_threshold') for x in r['refreshes']]
        row = dict(update=[c['update'] for c in ch], dose=[c['bc_multiplier_mean'] for c in ch],
                   lam=[c['lambda'] for c in ch], q=[c['q_mean'] for c in ch], bc=[c['bc_loss'] for c in ch],
                   thr=thr, refreshes=r['refreshes'][:2])
        out[f'{q}/{arm}'] = row
        sel = [1, 2, 3, 5, 9, 14, 19]
        print(f"{q:14s} {arm:13s} dose " + ' '.join(f"{row['update'][i]//1000}k:{row['dose'][i]:.3f}" for i in sel))
        print(f"{'':28s} bc   " + ' '.join(f"{row['bc'][i]:.3f}" for i in sel), '| lambda end', round(row['lam'][-1], 4), 'q end', round(row['q'][-1], 2))
        print(f"{'':28s} thr  " + ' '.join(f"{t:.2f}" for t in thr[:6]), '...', f"{thr[-1]:.2f}")
json.dump(out, open(Path(__file__).with_name('dose_chunks.json'), 'w'), indent=1)
