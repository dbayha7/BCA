import json, numpy as np, sys
R='/home/user/bca/runs/shift_explore/stage2/policy'
arms_core=['none','full_none','clone','bca_k1','bca_k2','bca_unpaired','bca_shuffled','perfect_band','range_hinge']
rows={}
for s in range(5):
    v=json.load(open(f'{R}/seed{s}/values.json')); l=json.load(open(f'{R}/seed{s}/lock.json'))
    st=l['standin']['arm']; mn=l['matched_names']
    names=arms_core+[st]+[mn[f] for f in ('even_pull','push_less','random_1','random_2')]
    keys=arms_core+['hinge_matched','even_pull','push_less','random_1','random_2']
    for k,n in zip(keys,names):
        sc=v['situations']['strong_current']; rc=v['situations']['recorded']
        rows.setdefault(k,[]).append(dict(
            sc=sc['points'][n], rec=rc['points'][n],
            p80_sc=sc['past_80_share_per_thruster'][n][0], p80_rec=rc['past_80_share_per_thruster'][n][0],
            unrec_sc=sc['unrecorded_throttle_share'][n],
            S=l['strengths'].get(n,np.nan), SR=l['region_strengths'].get(n,np.nan)))
def m(k,f): return np.mean([r[f] for r in rows[k]])
print(f"{'arm':14s} {'SC pts':>7s} {'rec':>6s} {'p80 SC t1':>9s} {'p80rec':>7s} {'unrecSC':>8s} {'S':>6s} {'S_reg':>6s}   per-seed SC")
for k in rows:
    print(f"{k:14s} {m(k,'sc'):7.1f} {m(k,'rec'):6.1f} {m(k,'p80_sc'):9.3f} {m(k,'p80_rec'):7.4f} {m(k,'unrec_sc'):8.3f} {m(k,'S'):6.3f} {m(k,'SR'):6.3f}  ", [round(r['sc'],1) for r in rows[k]])
print()
for k in ['bca_k1','bca_k2','bca_unpaired','bca_shuffled','perfect_band','range_hinge','hinge_matched','even_pull']:
    print(k,'S per seed',[round(r['S'],4) for r in rows[k]],'SR',[round(r['SR'],4) for r in rows[k]],'p80',[round(r['p80_sc'],3) for r in rows[k]])
json.dump(rows,open('/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact/rows.json','w'))
