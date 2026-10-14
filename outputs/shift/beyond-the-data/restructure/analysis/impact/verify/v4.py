import json, numpy as np
from scipy.stats import rankdata, spearmanr
KNEE=float(np.float32(0.8))
DET="/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
S2D="/home/user/bca/runs/shift_explore/stage2/detect/{v}/stall/fixed/seed{k}"
def auc(p,n):
    r=rankdata(np.concatenate([p,n])); return (r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))
res={}
for k in range(5):
    P=np.load(DET.format(k=k)+"/pairs.npz")
    sets,role,src=P['set'],P['role'],P['t3_source']
    cmd=np.asarray(P['commands'],np.float64); p80=np.any(np.abs(cmd)>KNEE,1)
    t3a=(sets=='T3')&(src=='actor')&(role=='command'); t4=sets=='T4'
    w=np.load(S2D.format(v='weighted',k=k)+"/scores/bca_width_weighted_fixed.npz")['scores'].astype(float)
    u=np.load(S2D.format(v='uniform',k=k)+"/scores/bca_width_uniform_fixed.npz")['scores'].astype(float)
    cr=np.load(DET.format(k=k)+"/scores/classifier_ratio.npz")['scores'].astype(float)
    di=np.load(DET.format(k=k)+"/scores/distance.npz")['scores'].astype(float)
    tw=json.load(open(S2D.format(v='weighted',k=k)+"/tau.json"))['tau']
    fl=~(w<=tw)
    step=P['t3_step']
    r=dict(n=int(t3a.sum()),p80=p80[t3a].mean(),rec=fl[t3a&p80].mean(),ff=fl[t3a&~p80].mean(),t4=fl[t4].mean(),
           auc_w=auc(w[t3a&p80],w[t3a&~p80]),auc_u=auc(u[t3a&p80],u[t3a&~p80]),auc_c=auc(cr[t3a&p80],cr[t3a&~p80]),auc_d=auc(di[t3a&p80],di[t3a&~p80]),
           rho_w_c=spearmanr(w[t3a],cr[t3a]).correlation, nonfinite=int((~np.isfinite(w)).sum()), nonfin_u=int((~np.isfinite(u)).sum()))
    for a,b in [(0,5),(5,15),(15,30),(30,60),(60,100)]:
        m=t3a&(step>=a)&(step<b)
        r[f's{a}-{b-1}']=(round(p80[m].mean(),2), round(fl[m&p80].mean(),2), round(fl[m&~p80].mean(),2) if (m&~p80).any() else None, int(m.sum()))
    # classifier at its own t4 95% level
    thc=np.quantile(cr[t4],0.95); flc=cr>thc
    r['cr_rec']=flc[t3a&p80].mean(); r['cr_ff']=flc[t3a&~p80].mean()
    # weighted band thresholds for recall 0.9
    thr=np.quantile(w[t3a&p80],0.10); r['w_ff_at_rec90']=(w[t3a&~p80]>thr).mean(); r['w_t4_at_rec90']=(w[t4]>thr).mean()
    res[k]=r
    print(k,{a:(round(b,3) if isinstance(b,float) else b) for a,b in r.items()})
for key in ['rec','ff','auc_w','auc_u','auc_c','auc_d','rho_w_c','cr_rec','cr_ff']:
    print(key, round(np.mean([res[k][key] for k in range(5)]),3))
