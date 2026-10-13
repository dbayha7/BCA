"""CPU 2: dump each v5 quality's marks (read only), compact: means, lower bounds, statuses, notes."""
import json
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5"
def compact(x, depth=0):
    if isinstance(x, dict):
        if "mean" in x and "n" in x:
            ps = x.get("per_seed")
            return {"mean": round(x["mean"], 4), "lo": None if x.get("lower_one_sided") is None else round(x["lower_one_sided"], 4),
                    "seeds": [round(v, 4) if isinstance(v, (int, float)) else v for v in ps] if ps else None}
        return {k: compact(v, depth + 1) for k, v in x.items() if k not in ("history",)}
    if isinstance(x, list):
        if len(x) > 12:
            return f"<list {len(x)}>"
        return [compact(v, depth + 1) for v in x]
    if isinstance(x, float):
        return round(x, 5)
    return x
for q in QS:
    r = json.load(open(f"{ROOT}/{q}/report/report.json"))
    print("=" * 30, q)
    print("top keys:", list(r.keys()))
    for k in ["D1", "M1", "M2", "M3", "W0", "W1", "W2", "W3", "W4", "R1", "R2", "I1", "L1", "L2", "L3"]:
        print("--", k, json.dumps(compact(r["marks"][k]))[:3000])
