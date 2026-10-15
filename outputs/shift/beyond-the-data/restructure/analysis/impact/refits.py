import json, numpy as np
R='/home/user/bca/runs/shift_explore/stage2/policy'
for s in range(5):
    l=json.load(open(f'{R}/seed{s}/lock.json'))
    print('=== seed',s, 'shape', {k:(round(v,3) if isinstance(v,float) else v) for k,v in l['bca_penalty_shape'].items() if k!='mean_W'})
    print('   mean_W', {k:round(v,3) for k,v in l['bca_penalty_shape']['mean_W'].items()})
    for arm in ['bca_k1','bca_k2','bca_unpaired','bca_shuffled','perfect_band']:
        a=l['arms'][arm]
        cm=np.array(a['chunk_metrics'])
        line=f"  {arm:13s} q_mean {np.round(cm[:,1],2).tolist()} lam {np.round(cm[:,2],3).tolist()} bc {np.round(cm[:,3],3).tolist()} kW {np.round(cm[:,4],4).tolist()}"
        print(line)
        if a['refits']:
            for c,r in enumerate(a['refits']):
                Rc=np.array(r['R_c']); R_=r['R']
                ninf=sum(x is None for x in R_)
                print(f"     chunk{c}: loss {r['loss_first100']:.3f}->{r['loss_last100']:.3f} Umax {r['U_max']:.3f} l* {r['l_star']:.2f} bank logw q {[round(r['bank_logw_quantiles'][q],3) for q in ('5','50','95','99')]} Rc[min,max] {Rc.min():.2f},{Rc.max():.2f} n_inf {ninf} audit {r['audit']['covered_share']:.3f}")
