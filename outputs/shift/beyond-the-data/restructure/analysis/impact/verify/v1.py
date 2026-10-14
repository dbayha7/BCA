import json, numpy as np
B='/home/user/bca/runs/shift_explore/stage2/policy/seed%d/'
rows={}
for K in range(5):
    v=json.load(open(B%K+'values.json'))
    L=json.load(open(B%K+'lock.json'))
    s=v['situations']['strong_current']
    names=list(s['points'].keys())
    st=[n for n in names if n.startswith('standin')][0]
    print('seed',K,'standin name',st, 'matched_names', v.get('matched_names'), 'standin', str(v.get('standin'))[:300])
    for n in names:
        key=n
        if n.startswith('standin'): key='matched_hinge'
        elif n.startswith('even_pull'): key='even_pull'
        elif n.startswith('random_1'): key='random_1'
        elif n.startswith('random_2'): key='random_2'
        elif n.startswith('push_less'): key='push_less'
        e=v['evaluations'][n]['strong_current']
        r=rows.setdefault(key,{'points':[],'cap':[],'loss':[],'ush':[],'p80':[],'sev':[], 'pts_e':[]})
        r['points'].append(s['points'][n]); r['cap'].append(e['capped_points']); r['loss'].append(e['loss_from_unrecorded_throttle'])
        r['ush'].append(e['unrecorded_throttle_share']); r['p80'].append(s['past_80_share_per_thruster'][n][0])
        r['sev'].append(e['loss_from_unrecorded_throttle']/e['unrecorded_throttle_share'])
        r['pts_e'].append(e['points'])
print()
for k,r in rows.items():
    print(f"{k:14s} pts {np.mean(r['points']):7.2f} [{', '.join('%.1f'%x for x in r['points'])}] cap {np.mean(r['cap']):6.1f} loss {np.mean(r['loss']):6.1f} ush {np.mean(r['ush']):.3f} p80 {np.mean(r['p80']):.3f} sev {np.mean(r['sev']):6.1f} sev_ratio {np.mean(r['loss'])/np.mean(r['ush']):6.1f}  pts_e {np.mean(r['pts_e']):.2f}")
json.dump(rows,open('/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact/verify/rows_v1.json','w'))
