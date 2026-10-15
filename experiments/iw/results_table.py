"""Print a compact cross-quality table of the version-5 results from each quality's report.json (read only)."""
import json

QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5"


def mean(x):
    if isinstance(x, dict) and "mean" in x:
        return x["mean"]
    return x


def lo(x):
    return x.get("lower_one_sided") if isinstance(x, dict) else None


def fmt(v, pct=False, d=2):
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{100 * v:.1f}%" if pct else f"{v:.{d}f}"
    return str(v)[:40]


def pick(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


rows = []
for q in QS:
    r = json.load(open(f"{ROOT}/{q}/report/report.json"))
    m, s = r["marks"], r["shown"]
    w1 = m["W1"]["coverage"]
    row = dict(
        q=q,
        M1=mean(m["M1"]["coverage"]),
        M2=mean(m["M2"].get("within_deciles")),
        W2_unw_strong=mean(m["W2"]["coverage"]),
        W1=w1,
        W1_single=m["W1"].get("single_threshold_coverage"),
        W3=m["W3"]["status"],
        W3_inside=m["W3"].get("inside_versus_outside_judged"),
        W4=m["W4"].get("strong_estimated_minus_exact"),
        R1=m["R1"]["placed_minus_none"], R1_pts=m["R1"].get("placed_minus_none_points"),
        R1_even=m["R1"].get("placed_minus_matched_even"),
        R2=m["R2"].get("predicted_return"), R2_dose=mean(m["R2"].get("dose_change")),
        L1=m["L1"]["oracle_minus_none"], L1_pts=m["L1"].get("oracle_minus_none_points"),
        L1_even=m["L1"].get("oracle_minus_matched_even"),
        L2=m["L2"].get("predicted_return"), L2_scale=mean(m["L2"].get("scale_change")),
        L2_slope=mean(m["L2"].get("return_per_unit_k")),
        status={k: v.get("status") for k, v in m.items()},
        plain_pts=s.get("plain_td3bc_points"),
        room_pts=s.get("room_points"),
        gap_cost=s.get("full_minus_gapped_plain_td3bc_strong"),
    )
    rows.append(row)

print(json.dumps(rows, indent=1, default=str)[:20000])
