"""EXPLORATORY: pooled throttle-sweep curves (critic and truth, Q(v) - Q(0.8), stall-aware command base) for the
anatomy note. Prints a table; writes runs/shift_explore/anatomy/curves.json."""
import json
from pathlib import Path

import numpy as np

OUT = Path("/home/user/bca/runs/shift_explore/anatomy")
runs = [json.loads(p.read_text()) for p in sorted(OUT.glob("critic_beliefs_seed*.json"))]
v = runs[0]["v_grid"]
res = {}
for s in ("train_region", "train_other", "strong_visit_stall_aware", "strong_visit_gapped_actor", "probe_c0.5",
          "probe_c0.9"):
    row = {}
    for base in ("controller", "pattern"):
        t = np.mean([r["sets"][s][f"{base}|gapped"]["curve_true"] for r in runs], 0)
        row[f"{base}|truth"] = t.tolist()
        line = [f"{s:26s} {base:10s} truth  " + " ".join(f"{x:+.2f}" for x in t)]
        for c in ("gapped", "full", "random"):
            q = np.mean([r["sets"][s][f"{base}|{c}"]["curve_critic"] for r in runs], 0)
            row[f"{base}|{c}"] = q.tolist()
            line.append(f"{'':37s} {c:6s} " + " ".join(f"{x:+.2f}" for x in q))
        if base == "controller":
            print("\n".join(line))
    res[s] = row
print("throttle grid:", v, "seeds:", len(runs))
(OUT / "curves.json").write_text(json.dumps(dict(
    label="EXPLORATORY: mean over states and gate seeds of Q(thruster 1 at throttle v) - Q(at 80%), critic and exact "
          "truth (Q of the stall-aware controller); not a gated output", v_grid=v, seeds=len(runs), curves=res),
    indent=1))
