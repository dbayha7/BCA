"""Read-only: summary.json pieces named by the pass-3 findings (data points, host differences)."""
import json
S = json.load(open("/root/bca-shift/runs/shift_iw_v5/summary/summary.json"))
def find(x, pat, pre=""):
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            p = pre + "." + str(k)
            if pat in str(k).lower(): out.append((p, v))
            out += find(v, pat, p)
    elif isinstance(x, list):
        for i, v in enumerate(x[:40]):
            out += find(v, pat, pre + f"[{i}]")
    return out
print(type(S["qualities"]), (list(S["qualities"].keys()) if isinstance(S["qualities"], dict) else len(S["qualities"])))
for pat in ("data_points", "host", "differ", "identical"):
    for p, v in find(S, pat)[:12]:
        print(p, json.dumps(v)[:350])
print("---- rows.random keys")
R = S["rows"]["random"]
for k, v in R.items():
    print(k, json.dumps(v)[:260])
