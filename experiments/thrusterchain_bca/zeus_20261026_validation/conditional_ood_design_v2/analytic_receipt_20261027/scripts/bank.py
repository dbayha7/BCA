import json, sys
import numpy as np
r=json.load(open(sys.argv[1]))
k=float(np.float32(0.8))
st=r['development_banks']['states']
g=np.array([s['state'][10]+s['state'][11] for s in st])
t=np.array([s['time'] for s in st])
actor_a0=np.array([s['actions'][0][0] for s in st])
print('n states',len(st),' share g>=0.4: %.3f'%np.mean(g>=0.4), ' g range %.3f..%.3f'%(g.min(),g.max()))
print('actor a0: min %.3f mean %.3f max %.3f ; share a0<=-0.7: %.3f ; share in band h=0.05: %.3f'%(actor_a0.min(),actor_a0.mean(),actor_a0.max(),np.mean(actor_a0<=-0.7),np.mean(np.abs(actor_a0+k)<=0.05)))
for gl,gh in ((-1,0.4),(0.4,0.7),(0.7,2)):
    m=(g>=gl)&(g<gh)
    if m.sum(): print('  g in [%.1f,%.1f): n=%d actor a0 mean %.3f min %.3f max %.3f'%(gl,gh,m.sum(),actor_a0[m].mean(),actor_a0[m].min(),actor_a0[m].max()))
# correlation of actor a0 with g and with time
print('corr(actor a0, g) = %.3f ; corr(actor a0, t) = %.3f'%(np.corrcoef(actor_a0,g)[0,1],np.corrcoef(actor_a0,t)[0,1]))
# candidates in band
rows=[]
for s in st:
    gg=s['state'][10]+s['state'][11]
    for j,a in enumerate(s['actions']):
        if abs(a[0]+k)<=0.05:
            rows.append((s['batch'],s['group_id'],s['time'],round(gg,3),j,round(a[0],4),round(s['q1'][j]-s['v'][j],2), round(s['u'][j],2)))
print('candidates with |a0+k|<=0.05 (batch, group, t, g, cand, a0, q1-v, u):', len(rows))
for x in rows: print('  ',x)
# where is the best candidate (by v) relative to a0 at high load
best=[]
for s in st:
    gg=s['state'][10]+s['state'][11]
    v=np.array(s['v']); a0s=np.array([a[0] for a in s['actions']])
    best.append((gg, a0s[np.argmax(v)], s['time']))
best=np.array(best)
for gl,gh in ((-1,0.4),(0.4,0.7),(0.7,2)):
    m=(best[:,0]>=gl)&(best[:,0]<gh)
    print('g in [%.1f,%.1f): mean a0 of best-v candidate %.3f (n=%d)'%(gl,gh,best[m,1].mean(),m.sum()))
