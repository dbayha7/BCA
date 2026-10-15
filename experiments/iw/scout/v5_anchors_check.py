"""Check of version 5's start-bank anchors and reference rows against the design (iw_quality/DESIGN.md section 3 and
iw_quality/out/one_controller.json). Exact simulation only: no learning, no BCA.

data.anchors on the shared start banks (1,000 starts per law): the random quality on stream 1002 must give
-800.29 / -1061.42 / -1351.41 (the design's random anchor), the stall-aware controller about -35.19 / -54.66 / -80.61,
and every quality's points with and without the hold must match one_controller.json.
  python iw/scout/v5_anchors_check.py OUT_JSON"""

import json
import sys
import time
from pathlib import Path

import experiments.shift  # noqa: F401
from experiments.shift_iw import data as D

DESIGN = Path(__file__).resolve().parents[2] / "iw_quality" / "out" / "one_controller.json"


def main():
    t0 = time.perf_counter()
    rows = D.anchors(D.start_banks(1000))
    design = json.loads(DESIGN.read_text())["points"] if DESIGN.is_file() else None
    out = dict(anchors={law: v["anchor"] for law, v in rows.items()}, rows=rows, compare={})
    for law, v in rows.items():
        c = {}
        for name, row in v["controllers"].items():
            key = {"stall_aware": "stall-aware", "zero_command": "zero command"}.get(name, name.replace(
                " without the hold", ", no hold"))
            d = (design or {}).get(law, {}).get(key)
            c[name] = dict(J=round(row["mean"], 2), points=round(row["points"], 1),
                           design_J=None if d is None else round(d["J"], 2),
                           design_points=None if d is None else round(d["points"], 1),
                           abs_J_difference=None if d is None else abs(row["mean"] - d["J"]))
        out["compare"][law] = c
    out["seconds"] = time.perf_counter() - t0
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    for law, c in out["compare"].items():
        print(law)
        for name, r in c.items():
            print(f"  {name:32s} J {r['J']:10.2f} (design {r['design_J']})  points {r['points']:6.1f} "
                  f"(design {r['design_points']})")
    print(f"{out['seconds']:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
