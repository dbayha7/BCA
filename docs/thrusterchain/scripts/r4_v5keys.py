"""Read-only: show the structure of marks L1, W3 (strong, medium-expert), M1 per_seed, W2 per_seed, D1."""
import json
ROOT = "/root/bca-shift/runs/shift_iw_v5"
def short(x, d=0):
    if isinstance(x, dict):
        return {k: short(v, d + 1) for k, v in x.items()} if d < 3 else '{...}'
    if isinstance(x, list):
        return [short(v, d + 1) for v in x[:6]] + (['...'] if len(x) > 6 else [])
    return x
for q in ["medium-expert", "expert", "random"]:
    r = json.load(open(f"{ROOT}/{q}/report/report.json"))
    m = r["marks"]
    print("=====", q)
    print("L1", json.dumps(short({k: v for k, v in m["L1"].items() if k not in ("oracle_minus_none", "oracle_minus_matched_even")}), default=str)[:1500])
    print("D1", json.dumps(short(m["D1"]), default=str)[:1200])
    print("M1 per_seed", json.dumps(m["M1"].get("coverage", {}).get("per_seed"), default=str)[:300])
    print("W2 per_seed", json.dumps(m["W2"].get("coverage", {}).get("per_seed"), default=str)[:300])
s = json.load(open(f"{ROOT}/summary/summary.json"))
print("summary keys", list(s.keys())[:30])
print(json.dumps(short(s), default=str)[:2500])
