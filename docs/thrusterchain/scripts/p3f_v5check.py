"""Read-only: check the version-5 numbers that the findings of pass 3 name (reports and summary are only read)."""
import json, math
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5"
def keys(x, pre="", depth=0, out=None):
    out = [] if out is None else out
    if isinstance(x, dict) and depth < 4:
        for k, v in x.items():
            out.append(pre + "." + k + (" =" + json.dumps(v)[:90] if not isinstance(v, (dict, list)) else ""))
            keys(v, pre + "." + k, depth + 1, out)
    return out
r = json.load(open(f"{ROOT}/medium-expert/report/report.json"))
print("\n".join(k for k in keys(r["marks"]["W4"], "W4")))
print("\n".join(k for k in keys(r["marks"]["W3"], "W3")[:60]))
print([k for k in r["shown"].keys()])
le = r["shown"].get("lower_edge")
print("\n".join(keys(le, "lower_edge")[:120]))
