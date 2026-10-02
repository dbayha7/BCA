import json, numpy as np
d = json.load(open(r'C:\Users\David\dev\BCA\runs\wbcp_signal\lq\results.json'))
print("behaviour case kappa offset : alignment_K mean ; J_opt-J_pi gap ; J_change_none ; oracle-const")
sett = {(s['behavior'], s['case'], s['kappa'], s['pi_offset']): s['diagnostics'] for s in d['settings']}
for r in d['results']:
    if r['signal'] != 'q1' or r['case'] in ('clean','noisy_reward','q2_optimistic'): continue
    m = r['metrics']; dg = sett[(r['behavior'], r['case'], r['kappa'], r['pi_offset'])]
    gap = dg['J_opt']['mean'] - dg['J_pi']['mean']
    oc = np.mean(np.array(m['J_change_oracle']['values']) - np.array(m['J_change_constant']['values']))
    print(r['behavior'], r['case'], r['kappa'], r['pi_offset'], f"{m['alignment_K']['mean']:.3f}", f"{gap:.3f}", f"{m['J_change_none']['mean']:.3f}", f"{oc:+.4f}")
