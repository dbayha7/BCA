"""READ-ONLY numpy. Is the stall visible in the one-step errors of the few past-80% rows the gapped log keeps?
Per seed and world: delta (signed) and |delta| at kept training rows by max-throttle stratum; stall vs no-stall."""
import json
import numpy as np
from scipy.stats import mannwhitneyu

OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
K = np.float32(0.8)
STRATA = [("<0.5", 0, 0.5), ("0.5-0.7", 0.5, 0.7), ("0.7-0.8", 0.7, 0.8000001), ("past80", 0.8000001, 9)]
res = {}
pool = {w: {s[0]: [] for s in STRATA} for w in ("stall", "no_stall")}
for seed in range(5):
    r = {}
    for world in ("stall", "no_stall"):
        d = np.load(f"{OUT}/data_seed{seed}_{world}.npz")
        z = np.load(f"{OUT}/delta_seed{seed}_{world}.npz")
        dl = z["train_delta"]
        thr = np.abs(d["train_act"]).max(1)
        o = {}
        for nm, lo, hi in STRATA:
            m = (thr >= lo) & (thr < hi) if nm != "past80" else thr > K
            o[nm] = dict(n=int(m.sum()), mean_delta=float(dl[m].mean()), q90_abs=float(np.quantile(np.abs(dl[m]), .9)),
                         median_abs=float(np.median(np.abs(dl[m]))), max_abs=float(np.abs(dl[m]).max()))
            pool[world][nm].append(np.abs(dl[m]))
        r[world] = o
    for nm, _, _ in STRATA:
        a = np.abs(np.load(f"{OUT}/delta_seed{seed}_stall.npz")["train_delta"][np.abs(np.load(f"{OUT}/data_seed{seed}_stall.npz")["train_act"]).max(1) > K]) if nm == "past80" else None
    ds_ = np.load(f"{OUT}/delta_seed{seed}_stall.npz")["train_delta"]; dn_ = np.load(f"{OUT}/delta_seed{seed}_no_stall.npz")["train_delta"]
    ts = np.abs(np.load(f"{OUT}/data_seed{seed}_stall.npz")["train_act"]).max(1) > K
    tn = np.abs(np.load(f"{OUT}/data_seed{seed}_no_stall.npz")["train_act"]).max(1) > K
    u = mannwhitneyu(np.abs(ds_[ts]), np.abs(dn_[tn]), alternative="greater")
    r["past80_abs_auc_stall_gt_nostall"] = float(u.statistic / (ts.sum() * tn.sum()))
    r["past80_abs_mwu_p"] = float(u.pvalue)
    r["past80_q90abs_ratio"] = r["stall"]["past80"]["q90_abs"] / r["no_stall"]["past80"]["q90_abs"]
    r["inrange_q90abs_ratio"] = r["stall"]["<0.5"]["q90_abs"] / r["no_stall"]["<0.5"]["q90_abs"]
    res[seed] = r
    print(seed, {w: {k: (v["n"], round(v["mean_delta"], 3), round(v["q90_abs"], 3)) for k, v in r[w].items()} for w in ("stall", "no_stall")})
    print("   past80 AUC(|d| stall > no_stall)", round(r["past80_abs_auc_stall_gt_nostall"], 3), "p", f"{r['past80_abs_mwu_p']:.2g}",
          "q90 ratio past80", round(r["past80_q90abs_ratio"], 2), "in-range", round(r["inrange_q90abs_ratio"], 2))
pooled = {}
for nm, _, _ in STRATA:
    s, n = np.concatenate(pool["stall"][nm]), np.concatenate(pool["no_stall"][nm])
    pooled[nm] = dict(n_stall=int(len(s)), n_no_stall=int(len(n)), q90_stall=float(np.quantile(s, .9)),
                      q90_no_stall=float(np.quantile(n, .9)), median_stall=float(np.median(s)), median_no_stall=float(np.median(n)))
res["pooled_abs_delta"] = pooled
print(json.dumps(pooled, indent=0))
json.dump(res, open(f"{OUT}/e3b_trace.json", "w"), indent=1)
