import json, numpy as np
D1="/home/user/bca/runs/shift_explore/deploy/seed{k}_stall_gapped.json"
S2="/home/user/bca/runs/shift_explore/stage2/deploy/stall/seed{k}/values.json"
rows=[]
for k in range(5):
    d=json.load(open(D1.format(k=k)))['situations']['strong_current']['rules']
    nv=d['none:None']['points']
    for r,v in d.items():
        if r.startswith('random:') and r.endswith('clip99'):
            rows.append((r,k,v['rate'],v['fire_share'],v['points']-nv))
    rows.append(('always',k,1.0,1.0,d['always:clip99']['points']-nv))
    s=json.load(open(S2.format(k=k)))['results']['none']['strong_current']
    nv2=s['none']['points']
    rows.append(('s2random:bca',k,s['random:bca']['rate'],s['random:bca']['fire_share'],s['random:bca']['points']-nv2))
    rows.append(('s2random:sa95',k,s['random:sa95']['rate'],s['random:sa95']['fire_share'],s['random:sa95']['points']-nv2))
names=sorted(set(r[0] for r in rows))
for n in names:
    q=[r for r in rows if r[0]==n]
    print(f"{n:28s} rate {np.mean([r[2] for r in q]):.3f} gain {np.mean([r[4] for r in q]):6.1f}  per-seed {[round(r[4],1) for r in q]}")
# linear fit of gain on rate
X=np.array([[1,r[3]] for r in rows]); y=np.array([r[4] for r in rows])
c,*_=np.linalg.lstsq(X,y,rcond=None); print('linear gain~rate',c)
print('--- per-point sorted by rate')
pts=sorted([(r[3],r[4],r[0],r[1]) for r in rows])
for b in [(0.2,0.3),(0.3,0.45),(0.5,0.62),(0.62,0.7),(0.7,0.8),(0.99,1.01)]:
    q=[p for p in pts if b[0]<=p[0]<b[1]]
    if q: print(b, len(q), 'rate %.3f gain %.1f'%(np.mean([p[0] for p in q]),np.mean([p[1] for p in q])))
