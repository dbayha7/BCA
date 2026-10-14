"""READ-ONLY attribution check for the calibration design's world gate (numpy/scipy on saved residual dumps; no fit,
no training). Does BCA's WBCP posterior add anything over plain two-sample tests on the same tail-vs-bulk residuals?
Tail = rows with max|a| >= t_up (calibration/e4_tailband.json, an outcome-free rule). Score B = -delta (overestimate).
Tests (one-sided, 'tail overestimated more than bulk'):
  mw_p     : Mann-Whitney U p-value, tail vs bulk
  t_p      : Welch t-test p-value on the means
  boot_q05 : plain (non-Bayesian) bootstrap 5% bound of [tail q90 - bulk q90], 2000 resamples of the tail
  wbcp_q05 : the design's own number (e11_credible_gap.json B.post_q05_gap), for comparison
Also the same on the BANK rows only (out-of-sample for the critic), pooled tail rows across seeds per world."""
import json
import numpy as np
from scipy import stats

CAL = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/judge_builder"
e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{CAL}/e4_tailband.json"))}
e11 = json.load(open(f"{CAL}/e11_credible_gap.json"))
res = {}
for w in ("stall", "no_stall"):
    for s in range(5):
        E = e4[(s, w)]
        d = np.load(f"{CAL}/data_seed{s}_{w}.npz")
        dl = np.load(f"{CAL}/delta_seed{s}_{w}.npz")["train_delta"]
        sc = -dl
        tail = np.abs(d["train_act"]).max(1) >= E["t_up"]
        a, b = sc[tail], sc[~tail]
        mw = stats.mannwhitneyu(a, b, alternative="greater").pvalue
        tt = stats.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue
        rng = np.random.default_rng([20261014, s, len(w)])
        bq = np.quantile(b, 0.9)
        boots = np.array([np.quantile(rng.choice(a, len(a)), 0.9) - bq for _ in range(2000)])
        res[f"{s}_{w}"] = dict(n_tail=int(tail.sum()), mean_shift=float(a.mean() - b.mean()),
                               mw_p=float(mw), t_p=float(tt), boot_q05=float(np.quantile(boots, 0.05)),
                               plain_q90_gap=float(np.quantile(a, 0.9) - bq),
                               wbcp_q05_valueunits=e11[f"{s}_{w}"]["B"]["post_q05_gap"])
for k, v in res.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
json.dump(res, open(f"{OUT}/gate_ablation.json", "w"), indent=1)
