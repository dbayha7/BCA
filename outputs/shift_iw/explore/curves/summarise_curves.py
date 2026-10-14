"""What the learning curves show, in numbers: for every quality x test x run, the mean over seeds at the first and last
checkpoint, the best checkpoint and when it comes, the drop from the best to the end, and the spread over the last
quarter of training. Read only; reads the packed data of build_lc_data.py.

    python summarise_curves.py LC_DATA.json
"""
import json
import sys

D = json.loads(open(sys.argv[1], encoding="utf-8").read())


def mean_curve(runs):
    runs = [r for r in runs if r]
    if not runs:
        return None
    return [sum(r[i] for r in runs) / len(runs) for i in range(len(runs[0]))], len(runs)


rows = []
for group in ("actor", "cotrain"):
    steps = D[group]["steps"]
    for arm, per_q in D[group]["arms"].items():
        for q, a in per_q.items():
            if not a:
                continue
            for law in D["laws"]:
                mc = mean_curve(a["pts"][law])
                if not mc:
                    continue
                c, n = mc
                best = max(range(len(c)), key=lambda i: c[i])
                q4 = c[3 * len(c) // 4:]
                rows.append(dict(group=group, arm=arm, q=q, law=law, seeds=n, start=c[0], end=c[-1], best=c[best],
                                 best_step=steps[best], drop=c[best] - c[-1], last_quarter_range=max(q4) - min(q4),
                                 at_quarter=c[len(c) // 4]))
print(f"{'group':8s} {'run':18s} {'quality':14s} {'test':6s} n  start   end  best@step       drop  q4-range  at-25%")
for r in rows:
    print(f"{r['group']:8s} {r['arm']:18s} {r['q']:14s} {r['law']:6s} {r['seeds']}  {r['start']:5.1f} {r['end']:5.1f} "
          f"{r['best']:5.1f}@{r['best_step']:<7d} {r['drop']:5.2f}  {r['last_quarter_range']:6.2f}  {r['at_quarter']:5.1f}")
json.dump(rows, open(sys.argv[1].replace(".json", "_summary.json"), "w"), indent=1)
