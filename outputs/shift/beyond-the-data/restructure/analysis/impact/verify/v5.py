import json, numpy as np
KNEE=float(np.float32(0.8))
DET="/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
S2D="/home/user/bca/runs/shift_explore/stage2/detect/{v}/stall/fixed/seed{k}"
sp=lambda x: np.logaddexp(0,x)
for k in range(5):
    P=np.load(DET.format(k=k)+"/pairs.npz")
    uw=np.load(S2D.format(v='weighted',k=k)+"/scores/bca_width_weighted_fixed.npz")['scores'].astype(float)
    uu=np.load(S2D.format(v='uniform',k=k)+"/scores/bca_width_uniform_fixed.npz")['scores'].astype(float)
    lw=np.load(DET.format(k=k)+"/scores/classifier_ratio.npz")['scores'].astype(float)
    b=json.load(open(S2D.format(v='weighted',k=k)+"/band.json")); wf=b['weighted_fit']; R=b['uniform']['R']
    t1=np.interp(lw,np.array(wf['log_t']),np.array(wf['R_c']))/R*uu; t2=wf['U_max']*sp(lw-wf['l_star'])
    e=np.abs(t1+t2-uw)
    sets=P['set']; t3a=(sets=='T3')&(P['t3_source']=='actor')&(P['role']=='command')
    print(k,'share err>0.05 all %.3f t3a %.3f'%((e>0.05).mean(),(e[t3a]>0.05).mean()), 'R_c range', min(wf['R_c']),max(wf['R_c']), 'log_t range', min(wf['log_t']), max(wf['log_t']))
