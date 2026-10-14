import sys
sys.dont_write_bytecode = True
import json, math
import numpy as np
d = np.load("/home/user/bca/runs/shift_explore/routes/small_evidence/draws.npz")
r = json.load(open("/home/user/bca/runs/shift_explore/routes/small_evidence/results.json"))
g = {int(k): v for k, v in r["gain_stall"].items()}; l = {int(k): v for k, v in r["loss_no_stall"].items()}
rules = {"B": lambda w, s, n: d[f"{w}_{s}_n{n}_q05"] > 0.10, "C": lambda w, s, n: d[f"{w}_{s}_n{n}_plain_gap"] > 0.10,
         "R": lambda w, s, n: d[f"{w}_{s}_n{n}_p"] < 0.05, "B0": lambda w, s, n: d[f"{w}_{s}_n{n}_q05"] > 0.0}
N = (12, 20, 30, 45)
for X in ("C", "R"):
    per_seed = []
    var = 0.0
    for s in range(5):
        c = 0.0
        for n in N:
            a = g[s] * (rules["B"]("stall", s, n).astype(float) - rules[X]("stall", s, n).astype(float))
            b = -l[s] * (rules["B"]("no_stall", s, n).astype(float) - rules[X]("no_stall", s, n).astype(float))
            c += (a.mean() + b.mean()) / len(N)
        per_seed.append(c / 5)
    print(X, "contribution per seed to D:", [round(x, 2) for x in per_seed], "sum", round(sum(per_seed), 3))
    # D without seed 1, rescaled to a 4-seed mean
    print(X, "D over seeds 0,2,3,4 only:", round(sum(per_seed[i] for i in (0, 2, 3, 4)) * 5 / 4, 3))
# decomposition of B - R
for n in N:
    print(n, "B-C", round(r["nets"]["B"][str(n)] - r["nets"]["C"][str(n)], 2), "C-R", round(r["nets"]["C"][str(n)] - r["nets"]["R"][str(n)], 2),
          "B-B0", round(r["nets"]["B"][str(n)] - r["nets"]["B0"][str(n)], 2), "B0-R", round(r["nets"]["B0"][str(n)] - r["nets"]["R"][str(n)], 2))
print("mean C-R over deciding n", np.mean([r["nets"]["C"][str(n)] - r["nets"]["R"][str(n)] for n in N]))
print("mean B-B0", np.mean([r["nets"]["B"][str(n)] - r["nets"]["B0"][str(n)] for n in N]))
print("stored se", r["decision"]["D_C"]["mc_se"], r["decision"]["D_R"]["mc_se"])
