"""Read-only: version-5 numbers named by the pass-3 findings."""
import json, math
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5"
def find(x, pat, pre=""):
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            p = pre + "." + str(k)
            if pat in str(k).lower(): out.append((p, v))
            out += find(v, pat, p)
    elif isinstance(x, list):
        for i, v in enumerate(x[:30]):
            out += find(v, pat, pre + f"[{i}]")
    return out
for q in QS:
    r = json.load(open(f"{ROOT}/{q}/report/report.json"))
    e = r["shown"]["exact_minus_unweighted_coverage"]
    st = e["per_tilt"]["strong"]["mean"] if "per_tilt" in e else None
    w4 = r["marks"]["W4"]
    print(q, "exact-unw strong %.5f" % st, "W4 strong %.5f none %.5f" % (w4["strong_estimated_minus_exact"], w4["none_estimated_minus_unweighted"]))
    for p, v in find(r, "shape") + find(r, "proxy"):
        s = json.dumps(v)
        print("   ", p, s[:400])
