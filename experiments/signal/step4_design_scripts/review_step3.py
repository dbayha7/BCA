import json, numpy as np
d = json.load(open(r'C:\Users\David\dev\BCA\runs\wbcp_signal\lq\results.json'))
rows = []
for r in d['results']:
    m = r['metrics']
    v = (lambda mm: (lambda k: np.array([x if x is not None else np.nan for x in mm[k]['values']], float)))(m)
    rows.append(dict(b=r['behavior'], c=r['case'], k=r['kappa'], o=r['pi_offset'], s=r['signal'], v=v, m=m))
# 1. ratio of bc-term norms oracle/constant and bca/constant for q1
print("case behaviour : oracle/const BC-norm ratio range ; bca/const range ; qterm/bc_none (rho_host) range")
from collections import defaultdict
g = defaultdict(list)
for x in rows:
    if x['s'] != 'q1': continue
    v = x['v']
    ro = v('bc_term_grad_norm_oracle')/v('bc_term_grad_norm_constant')
    rb = v('bc_term_grad_norm_bca')/v('bc_term_grad_norm_constant')
    rho = v('bc_term_grad_norm_none')/v('q_term_grad_norm')
    g[(x['c'], x['b'])].append((ro, rb, rho))
for key, lst in sorted(g.items()):
    ro = np.concatenate([a for a,_,_ in lst]); rb = np.concatenate([b for _,b,_ in lst]); rho = np.concatenate([c for _,_,c in lst])
    print(key, f"{np.nanmin(ro):.3f}-{np.nanmax(ro):.3f}", f"{np.nanmin(rb):.4f}-{np.nanmax(rb):.4f}", f"{np.nanmin(rho):.3f}-{np.nanmax(rho):.3f} med {np.nanmedian(rho):.3f}")
# 2. strength-only model and its sign mechanics, all signals
pred, obs, se, sgn_cn = [], [], [], []
for x in rows:
    v = x['v']
    jn, jc, jb = v('J_change_none'), v('J_change_constant'), v('J_change_bca')
    dbar = v('dose_mean')
    r = v('bc_term_grad_norm_bca')/v('bc_term_grad_norm_constant')
    p = (jc - jn) * (r - 1) * dbar/(dbar - 1)
    o = jb - jc
    pred.append(np.nanmean(p)); obs.append(np.nanmean(o)); se.append(np.nanstd(o, ddof=1)/np.sqrt(np.sum(np.isfinite(o))))
    sgn_cn.append(np.nanmean(jc - jn))
pred, obs, se, sgn_cn = map(np.array, (pred, obs, se, sgn_cn))
print("n cells", len(obs), "corr(pred, obs)", np.corrcoef(pred, obs)[0,1])
out = np.abs(obs) > 2*se
print("outside 2SE", out.sum(), "sign agree pred", np.sum(np.sign(pred[out]) == np.sign(obs[out])),
      "sign agree with sign(const-none)", np.sum(np.sign(sgn_cn[out]) == np.sign(obs[out])))
# spearman to discount scale
from scipy.stats import spearmanr
print("spearman", spearmanr(pred, obs).correlation)
# drop the top-5 |obs|
idx = np.argsort(-np.abs(obs))[5:]
print("corr without 5 largest |obs|", np.corrcoef(pred[idx], obs[idx])[0,1])
# 3. update_pg_cosine none vs constant: how different are directions
c_none, c_const = [], []
for x in rows:
    if x['s']!='q1': continue
    c_none.append(np.nanmean(x['v']('update_pg_cosine_none'))); c_const.append(np.nanmean(x['v']('update_pg_cosine_constant')))
print("update_pg_cosine none median", np.median(c_none), "const median", np.median(c_const))
