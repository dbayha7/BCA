"""READ-ONLY numpy on IMP/vw/vw.json and IMP/extra.json: whose is the 'eta frozen' ranking gain (0.903 vs 0.901)?
Pooled stall-world means over seeds 0-4 of the T3 (strong-current) ranking; index 1 of each rank_* list is T3."""
import json
import numpy as np
IMP = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
vw = [r for r in json.load(open(f"{IMP}/vw/vw.json")) if r["world"] == "stall"]
ex = [r for r in json.load(open(f"{IMP}/extra.json")) if r["world"] == "stall"]
o = {}
for k in ("weighted", "uniform", "term1", "term2", "lw", "cr", "eta", "cf_T4", "cf_bank"):
    o[k] = dict(T12=float(np.mean([r[f"rank_{k}"][0] for r in vw])), T3=float(np.mean([r[f"rank_{k}"][1] for r in vw])),
                T3_per_seed=[round(r[f"rank_{k}"][1], 4) for r in vw])
o["extra_rank_T3_eta_frozen_mean"] = float(np.mean([r["rank_T3_eta_frozen"] for r in ex]))
o["extra_rank_T3_weighted_mean"] = float(np.mean([r["rank_T3_weighted"] for r in ex]))
o["cf_equals_lw_every_seed"] = all(r["rank_cf_T4"] == r["rank_lw"] and r["rank_cf_bank"] == r["rank_lw"] for r in vw)
o["lw_vs_cr_spearman"] = [round(r["lw_vs_cr_spearman"], 4) for r in vw]
json.dump(o, open(f"{OUT}/e1_attrib.json", "w"), indent=1)
print(json.dumps(o, indent=0))
