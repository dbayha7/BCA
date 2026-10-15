"""Read-only: print the v5 numbers the weighting-test text uses, per quality."""
import json
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5"
def g(d, *ks):
    for k in ks:
        if d is None: return None
        d = d.get(k) if isinstance(d, dict) else None
    return d
def mn(x):
    if isinstance(x, dict) and 'mean' in x: return round(x['mean'], 4), (round(x['lower_one_sided'], 4) if x.get('lower_one_sided') is not None else None)
    return x
for q in QS:
    r = json.load(open(f"{ROOT}/{q}/report/report.json"))
    m = r['marks']
    print('=====', q, {k: v['status'] for k, v in sorted(m.items())})
    for key in ['R1', 'R2', 'L1', 'L2', 'W3', 'D1', 'M3', 'W4']:
        x = m[key]
        def walk(p, v, depth=0):
            if isinstance(v, dict) and 'mean' in v and 'n' in v:
                print('  ', p, mn(v)); return
            if isinstance(v, dict) and depth < 4:
                for k2, v2 in v.items():
                    if k2 in ('per_seed', 'interval', 'history', 'ladder_S', 'ladder_knobs', 'match'): continue
                    walk(p + '.' + k2, v2, depth + 1)
            elif not isinstance(v, (dict, list)):
                print('  ', p, v)
        walk(key, x)
