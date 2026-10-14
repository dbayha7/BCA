import json, numpy as np
R='/home/user/bca/runs/shift_explore/stage2/policy'
out={}
for s in range(5):
    v=json.load(open(f'{R}/seed{s}/values.json')); l=json.load(open(f'{R}/seed{s}/lock.json'))
    names=dict(none='none',full_none='full_none',clone='clone',bca_k1='bca_k1',bca_k2='bca_k2',bca_unpaired='bca_unpaired',
               bca_shuffled='bca_shuffled',perfect_band='perfect_band',range_hinge='range_hinge',hinge_matched=l['standin']['arm'],
               even_pull=l['matched_names']['even_pull'])
    for k,n in names.items():
        for sit in ('strong_current','recorded','big_waves','big_kicks'):
            e=v['evaluations'][n][sit]
            out.setdefault((k,sit),[]).append((e['points'],e['capped_points'],e['loss_from_unrecorded_throttle'],e['unrecorded_throttle_share']))
for sit in ('strong_current','recorded','big_kicks'):
    print('==',sit)
    print(f"{'arm':14s} {'pts':>6s} {'capped':>7s} {'loss':>6s} {'share':>6s} {'loss/share':>10s}")
    for k in ['none','full_none','clone','bca_k1','bca_k2','bca_unpaired','bca_shuffled','perfect_band','range_hinge','hinge_matched','even_pull']:
        a=np.array(out[(k,sit)])
        m=a.mean(0)
        print(f"{k:14s} {m[0]:6.1f} {m[1]:7.1f} {m[2]:6.1f} {m[3]:6.3f} {np.mean(a[:,2]/np.maximum(a[:,3],1e-9)):10.1f}  capped/seed {np.round(a[:,1],1).tolist()}")
