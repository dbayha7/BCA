"""Progress of the curve jobs: files written, and every reproduction flag that is False."""
import json
from pathlib import Path
ROOT = Path("/root/bca-shift/runs/shift_iw_v5_explore/curves")
n, bad, new = 0, [], 0
for f in sorted(ROOT.glob("*/seed*/*.json")):
    n += 1
    for arm, c in json.loads(f.read_text())["curves"].items():
        m = c["final_matches_saved"]
        if m is False:
            bad.append(f"{f.relative_to(ROOT)}:{arm}")
        elif m is None:
            new += 1
print(f"{n} of 210 curve files; {len(bad)} reruns that do not end at the saved actor; {new} new arms")
for b in bad:
    print("  NOT REPRODUCED:", b)
