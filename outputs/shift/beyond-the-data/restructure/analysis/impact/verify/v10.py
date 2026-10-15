import json, numpy as np
KNEE=float(np.float32(0.8))
DET="/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
S2D="/home/user/bca/runs/shift_explore/stage2/detect/{v}/stall/fixed/seed{k}"
for k in range(5):
    P=np.load(DET.format(k=k)+"/pairs.npz")
    cmd=np.asarray(P['commands'],np.float64); p80=np.any(np.abs(cmd)>KNEE,1)
    t3a=(P['set']=='T3')&(P['t3_source']=='actor')&(P['role']=='command'); t4=P['set']=='T4'
    c=np.asarray(P['states'],np.float64)[:,10]
    w=np.load(S2D.format(v='weighted',k=k)+"/scores/bca_width_weighted_fixed.npz")['scores'].astype(float)
    tw=json.load(open(S2D.format(v='weighted',k=k)+"/tau.json"))['tau']
    row=[]
    for name,thr in [('tau95',tw),('t4q99',np.quantile(w[t4],0.99))]:
        fl=w>thr
        row.append((name, round(thr,3), 'rec %.2f ff %.2f fire %.2f prec %.2f'%(fl[t3a&p80].mean(), fl[t3a&~p80].mean(), fl[t3a].mean(), (fl&t3a&p80).sum()/max((fl&t3a).sum(),1)),
                    'rec c>=1.0: %.2f'%fl[t3a&p80&(c>=1.0)].mean()))
    print(k,row)
