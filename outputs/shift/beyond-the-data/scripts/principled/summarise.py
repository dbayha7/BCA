"""EXPLORATORY: pool the stand-in readouts (readout, learned_ratio[ _floor*], classifier_ratio) over seeds 0-4 into
runs/shift_explore/principled/summary.json (mean, min, max over seeds). No BCA code."""
import json
import numpy as np
from pathlib import Path

D = Path("/home/user/bca/runs/shift_explore/principled")


def pool(files, keyfn):
    R = [json.loads((D / f).read_text()) for f in files]
    out = {}
    pools = R[0]["pools"].keys()
    for p in pools:
        out[p] = {}
        for k in R[0]["pools"][p]:
            v = [r["pools"][p].get(k) for r in R]
            if all(isinstance(x, (int, float)) and x is not None for x in v):
                a = np.asarray(v, float)
                out[p][k] = dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()), per_seed=[float(x) for x in a])
    return out


S = dict(label="EXPLORATORY: stand-in signals only (oracle one-step error, exact recorder likelihood ratio, generic "
               "mixture-density ratio, classifier ratio, twin-head disagreement); no BCA band, calibrator or dose; "
               "seeds 0-4, base world, saved gate artifacts",
         exact_ratio_and_oracle_error=pool([f"readout_seed{s}.json" for s in range(5)], None),
         classifier_ratio=pool([f"classifier_ratio_seed{s}.json" for s in range(5)], None))
for tag in ("", "_floor0.05", "_floor0.1"):
    S["mixture_density_ratio" + (tag or "_nofloor")] = pool([f"learned_ratio_seed{s}{tag}.json" for s in range(5)], None)
S["classifier_bank_kish_ess"] = {k: [json.loads((D / f"classifier_ratio_seed{s}.json").read_text())["boundaries"][k]
                                     ["classifier"]["kish_ess"] for s in range(5)] for k in ("gapped", "full", "clone")}
S["exact_bank_kish_ess"] = {k: [json.loads((D / f"readout_seed{s}.json").read_text())["boundaries"][k]["kish_ess"]
                                for s in range(5)] for k in ("gapped", "full", "clone", "td3bc_gapped")}
(D / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True))
print("wrote", D / "summary.json")
