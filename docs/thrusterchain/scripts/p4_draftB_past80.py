"""Read-only: share of weighting-test (version 5) training steps with thruster 1 more than 80% upstream, per quality
and data seed, for the full log and the log after the 80%-hold removal. Run with /opt/bca-shift-venv python.
Prints the structure first so the reading can be checked."""
import json, sys
import numpy as np
import h5py

ROOT = '/root/bca-shift/runs/shift_iw_v5'
QUALS = ['random', 'medium-replay', 'medium', 'medium-expert', 'expert']


def walk(f):
    out = []
    f.visititems(lambda n, o: out.append((n, getattr(o, 'shape', None), getattr(o, 'dtype', None))))
    return out


if '--peek' in sys.argv:
    with h5py.File(f'{ROOT}/medium/data/seed0/samples.hdf5', 'r') as f:
        for x in walk(f):
            print(x)
        print(dict(f.attrs))
    w = json.load(open(f'{ROOT}/medium/data/seed0/world.json'))
    print(json.dumps(w['gap'])[:1500])
    print(json.dumps(w['removal'])[:800])
    print(json.dumps(w['hold_command'])[:300])
    sys.exit()

res = {}
for q in QUALS:
    for s in range(5):
        d = f'{ROOT}/{q}/data/seed{s}'
        with h5py.File(d + '/train_law.hdf5', 'r') as f:
            split = np.asarray(f['infos/split'])
            if q == QUALS[0] and s == 0:
                print('split values', np.unique(split, return_counts=True))
            tr = split == 0
            a = np.asarray(f['actions'], dtype=np.float64)[tr]
            inball = np.asarray(f['truth/in_ball'])[tr]
            rnd = np.asarray(f['infos/random'])[tr]
        n = len(a)
        a1 = a[:, 0]
        past = a1 < -0.8 - 1e-6   # upstream is negative
        if past.any():
            assert rnd[past].all(), 'a past-80 command that is not random'
        keep = ~inball
        res[(q, s)] = (n, past.mean(), keep.sum(), past[keep].mean() if keep.any() else float('nan'))
        print(q, s, 'rows', n, 'past80 full %.4f' % past.mean(), 'kept', keep.sum(), 'past80 kept %.4f' % past[keep].mean())
for q in QUALS:
    f = [res[(q, s)][1] for s in range(5)]
    k = [res[(q, s)][3] for s in range(5)]
    print('%-14s full mean %.4f (%.4f..%.4f)   kept mean %.4f (%.4f..%.4f)' % (q, np.mean(f), min(f), max(f), np.mean(k), min(k), max(k)))
