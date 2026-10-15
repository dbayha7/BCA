import json, numpy as np
from scipy.stats import rankdata
KNEE=float(np.float32(0.8))
DET="/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
S2D="/home/user/bca/runs/shift_explore/stage2/detect/{v}/stall/fixed/seed{k}"
def auc(p,n):
    r=rankdata(np.concatenate([p,n])); return (r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))
for k in range(5):
    P=np.load(DET.format(k=k)+"/pairs.npz")
    cmd=np.asarray(P['commands'],np.float64); p80=np.any(np.abs(cmd)>KNEE,1)
    t3a=(P['set']=='T3')&(P['t3_source']=='actor')&(P['role']=='command')
    c=np.asarray(P['states'],np.float64)[:,10]
    S={'w':np.load(S2D.format(v='weighted',k=k)+"/scores/bca_width_weighted_fixed.npz")['scores'].astype(float),
       'u':np.load(S2D.format(v='uniform',k=k)+"/scores/bca_width_uniform_fixed.npz")['scores'].astype(float),
       'lw':np.load(DET.format(k=k)+"/scores/classifier_ratio.npz")['scores'].astype(float),
       'dist':np.load(DET.format(k=k)+"/scores/distance.npz")['scores'].astype(float)}
    out={}
    for n,s in S.items():
        out[n]=[round(auc(s[t3a&p80&(c>=a)&(c<b)], s[t3a&~p80&(c>=a)&(c<b)]),2) for a,b in [(0.7,0.85),(0.85,1.0),(1.0,1.11)]]
    print(k,out)
