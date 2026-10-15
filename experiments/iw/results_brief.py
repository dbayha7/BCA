"""One line per quality and measure: means (and one-sided lower bounds where they exist) from each report.json."""
import json

QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5"


def m_(x):
    return x["mean"] if isinstance(x, dict) and "mean" in x else x


def lo(x):
    return x.get("lower_one_sided") if isinstance(x, dict) else None


def f(x, d=3):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))


def walk_means(name, x, out, depth=0):
    """Flatten nested summaries to name -> mean (lower)."""
    if isinstance(x, dict) and "mean" in x and "n" in x:
        out.append((name, x["mean"], x.get("lower_one_sided")))
    elif isinstance(x, dict) and depth < 3:
        for k, v in x.items():
            if k in ("per_seed", "interval", "history", "match", "ladder_S", "ladder_knobs"):
                continue
            walk_means(f"{name}.{k}", v, out, depth + 1)


for q in QS:
    r = json.load(open(f"{ROOT}/{q}/report/report.json"))
    print(f"===== {q}: status " + " ".join(f"{k}={v['status']}" for k, v in sorted(r["marks"].items())))
    for key in ["M1", "M2", "W1", "W2", "W3", "W4", "R1", "R2", "L1", "L2"]:
        out = []
        walk_means(key, r["marks"][key], out)
        for name, mean, low in out:
            print(f"  {name:70s} {f(mean):>10s}  lower {f(low)}")
    out = []
    for key in ["plain_td3bc_points", "room_points", "full_minus_gapped_plain_td3bc_strong", "lower_edge"]:
        walk_means("shown." + key, r["shown"].get(key), out)
    for name, mean, low in out:
        print(f"  {name:70s} {f(mean):>10s}  lower {f(low)}")
