# AFTER-THE-FACT (post-result) arithmetic on harm_dose results.json; not part of the kill rule.
# Reproduces the verifier's margin sweep so the route log can cite a file.
import json, sys
d = json.load(open('/home/user/bca/runs/shift_explore/routes/harm_dose/results.json'))
p = d['per_seed_sigma']; seeds = sorted(p.keys()); sig = d['sigmas']; interior = d['interior_sigmas']
def v(s, g): return p[s][g]['v']
always7 = sum(sum(v(s, str(g) if str(g) in p[s] else g) for s in seeds)/len(seeds) for g in map(str, interior))
out = {'label': 'after-the-fact sweep, exploratory, seeds 0-4; not decision-relevant', 'always_clip_sum7': always7, 'sweep': {}}
for m in [0.0, 0.02, 0.04, 0.06, 0.067, 0.08, 0.10, 0.115, 0.12, 0.15, 0.20]:
    row = {}
    for name, key in [('bca_q05', 'q05'), ('plain_gap', 'plain_gap')]:
        tot = 0.0
        for g in map(str, interior):
            tot += sum((1.0 if float(p[s][g]['cert'][key]) > m else 0.0) * v(s, g) for s in seeds) / len(seeds)
        row[name + '_sum7'] = tot
        row[name + '_minus_best_constant'] = tot - max(always7, 0.0)
    out['sweep'][str(m)] = row
print(json.dumps(out, indent=1))
json.dump(out, open('margin_sweep.json', 'w'), indent=1)
