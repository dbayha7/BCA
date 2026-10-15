import json, sys
import numpy as np
for name in sys.argv[1:]:
    r = json.load(open(f"/root/bca-shift/runs/shift_iw/scout/{name}.json"))
    g = np.array(r["grid"]); cv = r["curves"]
    print(name)
    print(" c     hold  in_gap  ordinary_in_gap  SA_limit(rec states)  SA_limit(own)  J_rec  J_normal  J_SA")
    for k in range(0, 41, 4):
        f = lambda key: cv[key]["mean"][k]
        print(f" {g[k]:.1f}  {f('exact80'):.3f}  {f('in_gap'):.3f}  {f('in_gap') - f('in_gap_and_special'):.3f}  {f('sa_limit_on_rec'):.3f}  {f('sa_limit_own'):.3f}  {f('J_rec'):7.1f} {f('J_clean'):7.1f} {f('J_sa'):7.1f}")
