"""READ-ONLY numpy. Out-of-sample corroboration: the calibration bank's few full-throttle rows (t >= the tail's lower
end from e4; never seen by the critic) -- their overestimate -delta minus the bulk bank median, by world."""
import json
import numpy as np
from scipy.stats import mannwhitneyu
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{OUT}/e4_tailband.json"))}
res, pool = {}, {"stall": [], "no_stall": []}
for w in ("stall", "no_stall"):
    for s in range(5):
        d = np.load(f"{OUT}/data_seed{s}_{w}.npz")
        bd = np.load(f"{OUT}/delta_seed{s}_{w}.npz")["bank_delta"]
        t = np.abs(d["bank_act"]).max(1)
        tail = t >= e4[(s, w)]["t_up"]
        x = -bd[tail] - np.median(-bd[~tail])
        res[f"{s}_{w}"] = dict(n=int(tail.sum()), excess_overestimate=[round(float(v), 3) for v in x])
        pool[w] += list(x)
u = mannwhitneyu(pool["stall"], pool["no_stall"], alternative="greater")
res["pooled"] = dict(stall=sorted(round(v, 3) for v in pool["stall"]), no_stall=sorted(round(v, 3) for v in pool["no_stall"]),
                     auc=float(u.statistic / (len(pool["stall"]) * len(pool["no_stall"]))), p_one_sided=float(u.pvalue))
print(json.dumps(res, indent=0))
json.dump(res, open(f"{OUT}/e8_bank_tail.json", "w"), indent=1)
