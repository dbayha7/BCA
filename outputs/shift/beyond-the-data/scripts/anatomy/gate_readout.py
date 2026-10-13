"""EXPLORATORY readout (no computation beyond means): from the gate's saved evaluations, each actor's points, its
points when the same actor is capped at 80% at evaluation, and its share of steps past 80%, per situation, means over
gate seeds 0-4. Reads runs/shift/... read-only; writes runs/shift_explore/anatomy/gate_readout.json."""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/bca/runs/shift/68d9e8779e148b7b")
OUT = Path("/home/user/bca/runs/shift_explore/anatomy")
SITS = ("recorded", "strong_current", "big_waves", "big_kicks")


def main():
    res = {}
    for stage, actors in (("harm_return", ("clone", "full", "random", "gapped", "td3bc_gapped", "host_swap_full",
                                           "host_swap_gapped", "no_stall_full", "no_stall_gapped")),
                          ("room", ("none", "harm_placed", "strength_matched_even", "amount_matched", "cap2",
                                    "even_x1.5", "even_x2", "even_x3"))):
        for a in actors:
            row = {}
            for s in SITS:
                v = {m: [] for m in ("points", "capped_points", "unrecorded_throttle_share",
                                     "loss_from_unrecorded_throttle")}
                for rep in range(5):
                    d = json.loads((ROOT / stage / "base" / "alpha5" / f"seed{rep}" / "values.json").read_text())
                    ev = d["evaluations"].get(a)
                    if ev is None:
                        continue
                    for m in v:
                        v[m].append(ev[s][m])
                row[s] = {m: (float(np.mean(x)) if x else None) for m, x in v.items()}
                row[s]["n"] = len(v["points"])
            res[f"{stage}:{a}"] = row
            r = row["strong_current"]
            print(f"{stage:11s} {a:22s} strong current: points {r['points']:+7.1f} capped at 80% {r['capped_points']:+7.1f}"
                  f" past80 share {r['unrecorded_throttle_share']:.2f} | recorded {row['recorded']['points']:+6.1f} "
                  f"(capped {row['recorded']['capped_points']:+6.1f}) big waves {row['big_waves']['points']:+6.1f} "
                  f"(capped {row['big_waves']['capped_points']:+6.1f}) big kicks {row['big_kicks']['points']:+6.1f} "
                  f"(capped {row['big_kicks']['capped_points']:+6.1f})")
    (OUT / "gate_readout.json").write_text(json.dumps(dict(
        label="EXPLORATORY readout of saved gate evaluations (means over gate seeds 0-4); not a gated output",
        note="no_stall_* actors are scored in the no-stall comparison world (its own anchors)", actors=res),
        indent=1, sort_keys=True))
    print(f"wrote {OUT / 'gate_readout.json'}")


if __name__ == "__main__":
    main()
