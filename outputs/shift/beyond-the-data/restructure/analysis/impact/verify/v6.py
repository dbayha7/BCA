import json, numpy as np
KNEE=float(np.float32(0.8))
DET="/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
S2D="/home/user/bca/runs/shift_explore/stage2/detect/{v}/stall/fixed/seed{k}"
for k in range(5):
    P=np.load(DET.format(k=k)+"/pairs.npz")
    cmd=np.asarray(P['commands'],np.float64); p80=np.any(np.abs(cmd)>KNEE,1)
    t3a=(P['set']=='T3')&(P['t3_source']=='actor')&(P['role']=='command')
    w=np.load(S2D.format(v='weighted',k=k)+"/scores/bca_width_weighted_fixed.npz")['scores'].astype(float)
    tw=json.load(open(S2D.format(v='weighted',k=k)+"/tau.json"))['tau']
    miss=t3a&p80&(w<=tw); step=P['t3_step']
    tot=miss.sum()
    sh={f'{a}-{b-1}':round(float((miss&(step>=a)&(step<b)).sum()/tot),2) for a,b in [(0,5),(5,15),(15,30),(30,60),(60,100)]}
    # magnitude of missed vs caught commands
    mx=np.abs(cmd).max(1)
    print(k,'misses',int(tot),'share by step',sh, 'max|a| missed med %.3f caught med %.3f'%(np.median(mx[miss]),np.median(mx[t3a&p80&(w>tw)])))
    # state current coordinate
    st=np.asarray(P['states'],np.float64) if 'states' in P.files else None
    if st is not None and k==0: print('  states shape',st.shape)
    if st is not None:
        c=st[:,10]
        print('   current missed med %.3f caught med %.3f'%(np.median(c[miss]),np.median(c[t3a&p80&(w>tw)])))
print(np.load(DET.format(k=0)+"/pairs.npz").files)
