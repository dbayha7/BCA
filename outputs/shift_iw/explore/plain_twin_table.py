"""Print the plain-versus-BCA coverage comparison from runs/shift_iw_v5_explore/plain_twin/<quality>.json."""
import json

QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5_explore/plain_twin"


def m(x):
    return None if x is None else x["mean"]


def pct(x):
    return "  -  " if x is None else f"{100 * x:5.1f}"


S = {q: json.load(open(f"{ROOT}/{q}.json"))["summary"] for q in QS}
for kind in ["gap", "full"]:
    print(f"=== host: {kind}   coverage % (plain twin | BCA unweighted | BCA weighted [refused])")
    for part in ["outside", "inside"]:
        print(f"-- {part} the removed region")
        for tilt in ["none", "mild", "strong"]:
            line = f"{tilt:6s}"
            for q in QS:
                s = S[q][kind][tilt][part]
                if not s:
                    line += f" | {q[:6]:6s}  n/a"
                    continue
                line += (f" | {q[:6]:6s} {pct(m(s['plain']))} {pct(m(s['unweighted_bca']))} "
                         f"{pct(m(s.get('weighted_bca')))}[{pct(m(s.get('weighted_bca_refused')))}]")
            print(line)
    print("-- widths")
    for q in QS:
        s = S[q][kind]
        o, si, so = s["none"]["outside"], s["strong"]["inside"], s["strong"]["outside"]
        ratio = m(o["plain_median_width"]) / m(o["bca_unweighted_median_width"])
        io = (m(si["plain_median_width"]) / m(so["plain_median_width"])) if si else None
        print(f"   {q:14s} plain/BCA median width {ratio:.3f}  plain inside/outside (strong) "
              f"{'-' if io is None else f'{io:.2f}'}  median |error| {m(o['median_abs_error']):.4f}  "
              f"plain median width {m(o['plain_median_width']):.4f}  BCA {m(o['bca_unweighted_median_width']):.4f}")
    # lower bounds for the strong shift, plain outside
    print("-- plain coverage, one-sided 95% lower and upper spread (outside, strong):",
          {q: (round(S[q][kind]["strong"]["outside"]["plain"]["mean"], 4),
               [round(v, 4) for v in S[q][kind]["strong"]["outside"]["plain"]["per_seed"]]) for q in QS})
