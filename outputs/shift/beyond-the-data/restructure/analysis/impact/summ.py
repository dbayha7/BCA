import json, numpy as np
D='/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact'
R=[json.load(open(f'{D}/actor_readout_seed{s}.json')) for s in range(5)]
V=[json.load(open(f'/home/user/bca/runs/shift_explore/stage2/policy/seed{s}/values.json')) for s in range(5)]
arms=list(R[0]['arms'].keys())
def m(f):
    xs=[f(r) for r in R]; xs=[x for x in xs if x is not None]; return np.mean(xs)
print('c_grid',R[0]['c_grid'])
print(f"{'arm':14s} | reg p80  depth  | p>0.8 p80 | SC p80  depth  >=.95 | sanity SC p80 values.json")
for a in arms:
    reg=m(lambda r:r['arms'][a]['region']['past80']); regd=m(lambda r:r['arms'][a]['region']['depth_given_past80'])
    hi=m(lambda r:r['arms'][a]['push_gt_0p8']['past80'])
    sc=m(lambda r:r['arms'][a]['strong_current_rollout']['past80']); scd=m(lambda r:r['arms'][a]['strong_current_rollout']['depth_given_past80'])
    sc95=m(lambda r:r['arms'][a]['strong_current_rollout']['share_ge_0p95'])
    vn=np.mean([V[s]['situations']['strong_current']['past_80_share_per_thruster'][R[s]['arms'][a]['arm']][0] for s in range(5)])
    print(f"{a:14s} | {reg:.3f}  {regd:.3f} | {hi:.3f}    | {sc:.3f}  {scd:.3f}  {sc95:.3f} | {vn:.3f}")
print()
print('sweep rest_amp0.25 throttle1 mean by current; and past80 share by current')
for a in arms:
    t=np.mean([r['arms'][a]['sweep']['rest_amp0.25']['throttle1_mean'] for r in R],0)
    p=np.mean([r['arms'][a]['sweep']['trainfloats_amp0.25']['past80'] for r in R],0)
    print(f"{a:14s} thr1 {np.round(t,2).tolist()}  | trainfloats p80 {np.round(p,2).tolist()}")
print()
print('per seed SC >=0.95 share, region past80')
for a in ['bca_k1','bca_k2','perfect_band','hinge_matched','none']:
    print(a,[round(r['arms'][a]['strong_current_rollout']['share_ge_0p95'],3) for r in R],[round(r['arms'][a]['region']['past80'],3) for r in R])
