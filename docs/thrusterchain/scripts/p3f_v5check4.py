"""Read-only: the capped-pull predicted return change (R2) and average dose change per quality."""
import json
for q in ["random", "medium-replay", "medium", "medium-expert", "expert"]:
    r = json.load(open(f"/root/bca-shift/runs/shift_iw_v5/{q}/report/report.json"))
    R2 = r["marks"]["R2"]
    keys = {k: v for k, v in R2.items() if not isinstance(v, (dict, list))}
    print(q, json.dumps(keys)[:400])
