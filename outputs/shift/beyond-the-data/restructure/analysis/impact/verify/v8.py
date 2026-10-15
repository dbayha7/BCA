import json, numpy as np
from scipy.stats import rankdata
DET="/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
D1="/home/user/bca/runs/shift_explore/deploy/seed{k}_stall_gapped.json"
S2="/home/user/bca/runs/shift_explore/stage2/deploy/stall/seed{k}/values.json"
allc=None
for k in range(5):
    P=np.load(DET.format(k=k)+"/pairs.npz")
    t3a=(P['set']=='T3')&(P['t3_source']=='actor')
    st=np.asarray(P['states'],np.float64); start=P['t3_start'][t3a]; c=st[t3a,10]
    cur={}
    for s,cc in zip(start,c): cur.setdefault(int(s),set()).add(round(float(cc),9))
    assert all(len(v)==1 for v in cur.values())
    cur={s:list(v)[0] for s,v in cur.items()}
    if allc is None: allc=cur
    else:
        common=set(cur)&set(allc); assert all(abs(cur[s]-allc[s])<1e-9 for s in common)
        allc.update(cur)
    s2=json.load(open(S2.format(k=k)))['results']['none']['strong_current']
    d1=json.load(open(D1.format(k=k)))['situations']['strong_current']['rules']
    bca=np.array(s2['bca']['per_start_points']); rnd=np.array(s2['random:bca']['per_start_points']); perf=np.array(d1['unrec:clip99']['per_start_points']); ratio=np.array(d1['ratio:clip99']['per_start_points']); rr=np.array(d1['random:ratio:clip99']['per_start_points'])
    idx=np.array(sorted(cur)); cc=np.array([cur[i] for i in idx])
    sh=(perf-bca)[idx]; dr=(bca-rnd)[idx]
    out=[]
    for a,b in [(0.7,0.8),(0.8,0.9),(0.9,1.0),(1.0,1.11)]:
        m=(cc>=a)&(cc<b); out.append((int(m.sum()), round(dr[m].mean(),1), round(sh[m].mean(),1), round(((ratio-rr)[idx])[m].mean(),1)))
    print(k,'n starts w/ current',len(idx),'spearman(current, shortfall vs perfect) %.2f'%np.corrcoef(rankdata(cc),rankdata(sh))[0,1], 'bins (n, bca-random, perfect-bca, ratio-rratio):',out)
print('starts with known current across seeds:',len(allc))
print('--- contributions to pooled M5 by current bin, and perfect-minus-random(bca rate) room')
tot={}
for k in range(5):
    s2=json.load(open(S2.format(k=k)))['results']['none']['strong_current']
    d1=json.load(open(D1.format(k=k)))['situations']['strong_current']['rules']
    bca=np.array(s2['bca']['per_start_points']); rnd=np.array(s2['random:bca']['per_start_points']); perf=np.array(d1['unrec:clip99']['per_start_points'])
    cc=np.array([allc[i] for i in range(200)])
    for a,b in [(0.7,0.9),(0.9,1.0),(1.0,1.11)]:
        m=(cc>=a)&(cc<b)
        tot.setdefault((a,b),[]).append(((bca-rnd)[m].sum()/200, (perf-rnd)[m].sum()/200, m.sum()))
for kk,v in tot.items():
    print(kk,'n/seed',v[0][2],'contrib to M5 per seed',[round(x[0],1) for x in v],'pooled %.2f'%np.mean([x[0] for x in v]),'| perfect-random contrib pooled %.2f'%np.mean([x[1] for x in v]))
