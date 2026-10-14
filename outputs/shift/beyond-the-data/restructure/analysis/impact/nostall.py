import json, numpy as np
R='/home/user/bca/runs/shift_explore/stage2/policy'
v0=json.load(open(f'{R}/seed0/no_stall/values.json')); l0=json.load(open(f'{R}/seed0/no_stall/lock.json'))
print('values keys',v0.keys()); print('lock keys',l0.keys())
print('eval arms',list(v0['evaluations'].keys()))
acc={}
for s in range(5):
    v=json.load(open(f'{R}/seed{s}/no_stall/values.json')); l=json.load(open(f'{R}/seed{s}/no_stall/lock.json'))
    for n,e in v['evaluations'].items():
        key=n.split('@')[0]
        for sit in ('strong_current','recorded'):
            acc.setdefault((key,sit),[]).append((e[sit]['points'],e[sit]['capped_points'],e[sit]['loss_from_unrecorded_throttle'],e[sit]['unrecorded_throttle_share'],l['strengths'].get(n,np.nan),l['region_strengths'].get(n,np.nan)))
for sit in ('strong_current','recorded'):
    print('==',sit)
    for k in sorted(set(a for a,b in acc if b==sit)):
        a=np.array(acc[(k,sit)]); m=np.nanmean(a,0)
        print(f"{k:28s} pts {m[0]:6.1f} capped {m[1]:6.1f} loss_unrec {m[2]:6.1f} share {m[3]:.3f} S {m[4]:.3f} SR {m[5]:.3f} per-seed pts {np.round(a[:,0],1).tolist()}")
print(json.dumps(l0.get('bca_penalty_shape'),indent=0)[:1500])
for s in range(5):
    l=json.load(open(f'{R}/seed{s}/no_stall/lock.json'))
    a=l['arms'].get('bca_k1')
    if a: print(s,'kW',np.round(np.array(a['chunk_metrics'])[:,4],4).tolist(), 'cls loss', [round(r['loss_last100'],3) for r in a['refits']])
