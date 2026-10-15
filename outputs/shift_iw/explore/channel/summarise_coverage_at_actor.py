"""Table of channel/coverage_at_actor.py's results: means over seeds (min to max) per quality. Read only."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("/root/bca-shift/runs/shift_iw_v5_explore/channel")
QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")


def get(r, *path):
    for p in path:
        r = r.get(p) if isinstance(r, dict) else None
    return r


def fmt(vals, pct=False, nd=1):
    v = np.array([x for x in vals if x is not None], float)
    if not len(v):
        return "n/a"
    k = 100.0 if pct else 1.0
    return f"{k * v.mean():.{nd}f} ({k * v.min():.{nd}f} to {k * v.max():.{nd}f})"


rows = []
for q in QUALITIES:
    seeds = json.loads((OUT / f"{q}.json").read_text())["seeds"]
    col = lambda *p: [get(s, *p) for s in seeds]
    rows.append((q, {
        "actor past 80%, share of rows (strong law)": fmt(col("actor_past_80_share_strong_law"), True),
        "actor past 80%, share of hard-water rows": fmt(col("actor_past_80_share_hard_water"), True),
        "coverage at recorded, train law, unweighted": fmt(col("unweighted@recorded", "all", "coverage_train_law"), True),
        "coverage at recorded, strong law, unweighted": fmt(col("unweighted@recorded", "all", "coverage_strong_law"), True),
        "coverage at recorded, strong law, exact": fmt(col("exact_strong@recorded", "all", "coverage_strong_law"), True),
        "coverage at actor, strong law, unweighted": fmt(col("unweighted@actor", "all", "coverage_strong_law"), True),
        "coverage at actor, strong law, exact": fmt(col("exact_strong@actor", "all", "coverage_strong_law"), True),
        "  within 80%: unweighted": fmt(col("unweighted@actor", "actor_within_80", "coverage_strong_law"), True),
        "  within 80%: exact": fmt(col("exact_strong@actor", "actor_within_80", "coverage_strong_law"), True),
        "  past 80%: unweighted": fmt(col("unweighted@actor", "actor_past_80", "coverage_strong_law"), True),
        "  past 80%: exact": fmt(col("exact_strong@actor", "actor_past_80", "coverage_strong_law"), True),
        "needed multiplier at actor past 80% (unweighted)": fmt(col("unweighted@actor", "actor_past_80",
                                                                    "needed_multiplier_strong_law"), nd=2),
        "needed multiplier at actor within 80% (unweighted)": fmt(col("unweighted@actor", "actor_within_80",
                                                                      "needed_multiplier_strong_law"), nd=2),
        "signed error y - q at actor past 80%": fmt(col("unweighted@actor", "actor_past_80",
                                                        "signed_error_mean_strong_law"), nd=3),
        "width U at actor past 80% (unweighted)": fmt(col("unweighted@actor", "actor_past_80",
                                                          "width_mean_strong_law"), nd=3),
        "signed error y - q at actor within 80%": fmt(col("unweighted@actor", "actor_within_80",
                                                          "signed_error_mean_strong_law"), nd=3),
        "width U at actor within 80% (unweighted)": fmt(col("unweighted@actor", "actor_within_80",
                                                            "width_mean_strong_law"), nd=3),
        "exact-strong R / R_u (mean)": fmt([get(s, "R_exact_strong", "mean") / s["R_u"] for s in seeds], nd=2),
        "sanity: reward / next-obs max diff": f"{max(col('sanity', 'reward_max_abs_diff')):.2g} / "
                                              f"{max(col('sanity', 'next_obs_max_abs_diff')):.2g}",
    }))
for q, d in rows:
    print("==", q)
    for k, v in d.items():
        print(f"  {k:52s} {v}")
