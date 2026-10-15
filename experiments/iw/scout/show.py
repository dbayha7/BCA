import json, sys
import numpy as np
for name in sys.argv[1:]:
    r = json.load(open(f"/root/bca-shift/runs/shift_iw/scout/{name}.json"))
    print("=====", name, r["recorder"])
    for row in r["training_logs"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    if "Z" not in r:
        continue
    print("Z", r["Z"])
    g = np.array(r["grid"])
    cv = r["curves"]
    print("c      special exact80 in_gap  region  SAlim_rec SAlim_own  J_rec   J_clean  J_sa  J_v1hand")
    for i in range(0, len(g), 4):
        print(f"{g[i]:.3f}  " + "  ".join(f"{cv[k]['mean'][i]:7.3f}" for k in ("special", "exact80", "in_gap", "region", "sa_limit_on_rec", "sa_limit_own", "J_rec", "J_clean", "J_sa", "J_v1_hand")))
    for t, row in r["tilts_table"].items():
        print(t, {k: ((round(v["mean"], 4), round(v["se"], 4)) if isinstance(v, dict) and "mean" in v else (round(v, 3) if isinstance(v, float) else v)) for k, v in row.items()})
    print("direct", r["direct_check_strong"])
    for gk in ("goodness_train", "goodness_strong"):
        gd = r.get(gk)
        if gd is None:
            print(gk, None); continue
        f = lambda d: f"{d['mean']:+.3f}+-{d['se']:.3f} (pos {d['share_positive']:.2f}, n {d['n']})"
        print(gk, "states", gd["states"], "mean c", round(gd["current_mean"], 3), "SA at limit", round(gd["stall_aware_at_limit"], 3))
        print("   special-normal", f(gd["special_minus_normal"]), " SA-normal", f(gd["stall_aware_minus_normal"]),
              " v1full-normal", f(gd["v1_full_throttle_minus_normal"]), " special-SA", f(gd["special_minus_stall_aware"]))
        for b, bd in gd["by_hold_demand"].items():
            print("   hold demand", b, "special-normal", f(bd["special_minus_normal"]), "special-SA", f(bd["special_minus_stall_aware"]), "SA at limit", round(bd["stall_aware_at_limit"], 3))
