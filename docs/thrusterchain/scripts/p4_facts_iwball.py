"""Read the weighting test's removed ball (centre, radius) per expert seed (read only). Prints JSON."""
import json

out = {}
for s in range(5):
    d = json.load(open(f"/root/bca-shift/runs/shift_iw_v5/expert/data/seed{s}/world.json"))
    hits = {}

    def walk(x, p=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, p + "/" + k)
        elif any(t in p.lower() for t in ("centre", "center", "rho", "radius", "hold", "pattern", "ball")):
            if not isinstance(x, list) or len(x) <= 13:
                hits[p] = x

    walk(d)
    out[s] = hits
print(json.dumps(out, indent=1)[:6000])
