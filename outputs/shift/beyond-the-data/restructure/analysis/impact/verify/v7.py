import json, numpy as np
from scipy.stats import rankdata
KNEE=float(np.float32(0.8))
DET="/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
S2D="/home/user/bca/runs/shift_explore/stage2/detect/{v}/stall/fixed/seed{k}"
bins=[(0.7,0.8),(0.8,0.9),(0.9,1.0),(1.0,1.1),(1.1,1.3)]
agg={}
for k in range(5):
    P=np.load(DET.format(k=k)+"/pairs.npz")
    cmd=np.asarray(P['commands'],np.float64); p80=np.any(np.abs(cmd)>KNEE,1)
    t3a=(P['set']=='T3')&(P['t3_source']=='actor')&(P['role']=='command')
    t4=P['set']=='T4'
    st=np.asarray(P['states'],np.float64); c=st[:,10]
    w=np.load(S2D.format(v='weighted',k=k)+"/scores/bca_width_weighted_fixed.npz")['scores'].astype(float)
    u=np.load(S2D.format(v='uniform',k=k)+"/scores/bca_width_uniform_fixed.npz")['scores'].astype(float)
    lw=np.load(DET.format(k=k)+"/scores/classifier_ratio.npz")['scores'].astype(float)
    tw=json.load(open(S2D.format(v='weighted',k=k)+"/tau.json"))['tau']
    fl=w>tw
    row=[]
    for a,b in bins:
        m=t3a&(c>=a)&(c<b)
        row.append((int(m.sum()), round(p80[m].mean(),2) if m.any() else None, round(fl[m&p80].mean(),2) if (m&p80).any() else None, round(fl[m&~p80].mean(),2) if (m&~p80).any() else None, round(float(np.median(lw[m&p80])),2) if (m&p80).any() else None))
    print(k,'T4 current range %.2f-%.2f'%(c[t4].min(),c[t4].max()),'T3a current %.2f-%.2f'%(c[t3a].min(),c[t3a].max()))
    for bb,r in zip(bins,row): print('   c in',bb,'n,p80,recall,ff,median logit(p80)',r)
    # correlation of logit with current among p80 T3a
    m=t3a&p80
    print('   spearman(current, logit | p80) %.2f ; (current, W | p80) %.2f'%(np.corrcoef(rankdata(c[m]),rankdata(lw[m]))[0,1], np.corrcoef(rankdata(c[m]),rankdata(w[m]))[0,1]))
