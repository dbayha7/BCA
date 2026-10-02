"""Post-hoc re-reading of step 3: a dose with the same mean is not a dose with the same strength.

Step 3 compared BCA's dose and an oracle dose against a constant dose with the same mean. Here that is checked
against the BC term's actual gradient norm at the starting actor (lq_harness metric bc_term_grad_norm_<arm>), and a
strength-only model is fitted:
  BCA - constant  ~=  (J_constant - J_none) * (|g_BC, arm| / |g_BC, constant| - 1) * d / (d - 1),
where d is the mean dose. That is what a constant dose of BCA's realized BC strength would add, if J responds
linearly to BC strength between the no-BCA arm and the constant arm. The script is post hoc (written after step 3
was scored; DEPENDENCE-style label "not pre-registered") and reads only runs/wbcp_signal/lq/results.json.

python experiments/signal/strength_reanalysis.py [--results runs/wbcp_signal/lq/results.json] [--output runs/wbcp_signal/strength_reanalysis.json]
"""

import argparse
import collections
import json
import math
import statistics


def per_replicate(r, name):
    return r["metrics"][name]["values"]


def corr(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="runs/wbcp_signal/lq/results.json")
    parser.add_argument("--output", default="runs/wbcp_signal/strength_reanalysis.json")
    args = parser.parse_args(argv)
    with open(args.results, encoding="utf-8") as handle:
        res = json.load(handle)

    ratios = collections.defaultdict(lambda: {"oracle": [], "bca": []})
    model, observed = [], []
    for r in res["results"]:
        const = per_replicate(r, "bc_term_grad_norm_constant")
        for arm, name in (("oracle", "bc_term_grad_norm_oracle"), ("bca", "bc_term_grad_norm_bca")):
            for c, a in zip(const, per_replicate(r, name)):
                if c and math.isfinite(c) and a is not None and math.isfinite(a):
                    if r["signal"] == "q1":
                        ratios[(r["case"], r["behavior"])][arm].append(a / c)
        # strength-only model for BCA - constant, on cell means
        jc, jn = r["metrics"]["J_change_constant"]["mean"], r["metrics"]["J_change_none"]["mean"]
        d = r["metrics"]["dose_mean"]["mean"]
        ratio = statistics.mean(a / c for c, a in zip(const, per_replicate(r, "bc_term_grad_norm_bca")) if c)
        if d > 1:
            model.append((jc - jn) * (ratio - 1) * d / (d - 1))
            observed.append(r["metrics"]["J_gain_bca_minus_constant"]["mean"])

    table = {f"{case}/{beh}": dict(
        oracle_over_constant=dict(min=min(v["oracle"]), median=statistics.median(v["oracle"]), max=max(v["oracle"])),
        bca_over_constant=dict(min=min(v["bca"]), median=statistics.median(v["bca"]), max=max(v["bca"])))
        for (case, beh), v in sorted(ratios.items())}
    out = dict(
        note="post hoc; BC-term gradient norm at the starting actor relative to the constant dose with the same mean (q1 signal)",
        bc_gradient_ratio=table,
        strength_only_model=dict(cells=len(model), pearson=corr(model, observed),
                                 description="BCA - constant against (J_const - J_none)(ratio - 1) d/(d - 1), all 216 cells"),
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    for key, v in table.items():
        o, b = v["oracle_over_constant"], v["bca_over_constant"]
        print(f"{key:28s} oracle/constant {o['min']:.2f}-{o['max']:.2f} (median {o['median']:.2f})   "
              f"BCA/constant {b['min']:.3f}-{b['max']:.3f}")
    print(f"strength-only model vs observed BCA - constant: Pearson {out['strength_only_model']['pearson']:.2f} "
          f"over {len(model)} cells")


if __name__ == "__main__":
    main()
