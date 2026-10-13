"""Reviewer: the numbers the viewer shows from the thousandths-encoded sweeps against the same numbers from the
7-digit file. Read only; prints only."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

Q7 = json.loads(Path("/root/bca-shift/runs/qviewer/precise/q_data.json").read_text())
QV = json.loads(Path("/mnt/c/Users/David/dev/bca-shift-work/qviewer/qv_data.json").read_text())
F7 = {f: i for i, f in enumerate(Q7["episode_fields"])}
FV = {f: i for i, f in enumerate(QV["fields"])}


def fx(v, d):
    return str(Decimal(repr(abs(float(v)))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)) if True else None


def fv(v):
    return fx(v, 1 if abs(v) >= 100 else 2)


cnt = dict(cost=0, cost_states=0, cost_zero_hat_ne_best=0, sweep_q=0, sweep_p=0, sweep_diff=0, rel_q=0, rel_p=0, cells=0)
worst_cost = (0, None)
zero_examples = []
for e, e7 in zip(QV["eps"], Q7["episodes"]):
    for t in range(100):
        qa, pa = e[FV["q_actor"]][t], e[FV["qpi_actor"]][t]
        rq = np.cumsum(e[FV["q_sweep"]][t]) / 1000.0
        rp = np.cumsum(e[FV["qpi_sweep"]][t]) / 1000.0
        q7, p7 = np.asarray(e7[F7["q_sweep"]][t]), np.asarray(e7[F7["qpi_sweep"]][t])
        h, b = e[FV["a_hat"]][t], e[FV["a_best_true"]][t]
        shown, true = rp[b] - rp[h], p7[b] - p7[h]
        cnt["cost_states"] += 1
        if fx(shown, 3) != fx(true, 3):
            cnt["cost"] += 1
            if abs(shown - true) > worst_cost[0]:
                worst_cost = (abs(shown - true), (e[:3], t, fx(shown, 3), round(true, 6)))
        if h != b and fx(shown, 3) == "0.000":
            cnt["cost_zero_hat_ne_best"] += 1
            if len(zero_examples) < 5:
                zero_examples.append((e[:3], t, h, b, round(true, 6)))
        for k in range(41):
            cnt["cells"] += 1
            q, p = qa + rq[k], pa + rp[k]
            cnt["sweep_q"] += fv(q) != fv(q7[k])
            cnt["sweep_p"] += fv(p) != fv(p7[k])
            cnt["sweep_diff"] += fx(q - p, 2) != fx(q7[k] - p7[k], 2)
            cnt["rel_q"] += fx(rq[k], 3) != fx(q7[k] - qa, 3)
            cnt["rel_p"] += fx(rp[k], 3) != fx(p7[k] - pa, 3)
print(json.dumps(cnt))
print("worst cost difference", worst_cost)
print("'would lose 0.000' with a-hat != true best:", zero_examples)

# the findings' range of the critic's mean change 80% -> 100% in the strong mix, per row
for x in QV["ctx"]:
    if x["situation"] == "strong":
        print(x["quality"], x["critic"], "critic change", x["critic_change_80_to_100_mean"], "truth change",
              x["truth_change_80_to_100_mean"], "critic higher share", x["critic_higher_at_100_than_80_share"])
