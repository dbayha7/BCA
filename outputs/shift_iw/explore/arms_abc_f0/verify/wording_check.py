"""Information-only differences for WORDING.md (EXPLORATORY; decides nothing).
Reads arm C's evaluation records with plain json; no project code."""
import json, math, statistics, sys
SP = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad"
R = SP + "/iw_rerun/repo/runs/shift_iw_v5_explore/arms_abc_f0/eval/C"
def pts(q, k, pol, bank="strong"):
    r = json.load(open(f"{R}/{q}/seed{k}/record.json"))
    e = r["evaluations"][pol][bank]
    assert e.get("finite", True) and e.get("starts", 1000) == 1000, (q, k, pol)
    return e["points"]
def row(q, a, b, bank="strong"):
    d = [pts(q, k, a, bank) - pts(q, k, b, bank) for k in range(5)]
    m = math.fsum(d) / 5
    lb = m - 2.132 * statistics.stdev(d) / math.sqrt(5)
    return dict(quality=q, bank=bank, arm=a, comparator=b, mean=m, pos=sum(x > 0 for x in d), lb=lb, d=d)
out = []
plain = ["sign", "wmedian", "wmean", "wttest", "wbernstein"]
for q in ["random", "medium-replay", "medium", "medium-expert", "expert"]:
    means = {p: math.fsum(pts(q, k, "rho0.5+" + p) for k in range(5)) / 5 for p in plain}
    best = max(means, key=means.get)
    out.append(dict(quality=q, plain_means=means, best_plain=best,
                    bca_mean=math.fsum(pts(q, k, "rho0.5+bca") for k in range(5)) / 5,
                    knee_mean=math.fsum(pts(q, k, "rho0.5+knee_clip") for k in range(5)) / 5))
    for b in ["rho0.5+knee_clip", "rho0.5+" + best, "rho0.5+range_limit", "rho0.5+shuffled", "rho0.5+matched_global", "rho0.5"]:
        out.append(row(q, "rho0.5+bca", b))
json.dump(out, open(SP + "/routes2/arms/verify/wording_check.json", "w"), indent=1)
for o in out:
    if "plain_means" in o:
        print(f"\n{o['quality']}: best plain {o['best_plain']} {o['plain_means'][o['best_plain']]:.4f}; bca {o['bca_mean']:.4f}; knee {o['knee_mean']:.4f}; plain {', '.join(f'{p} {v:.4f}' for p, v in o['plain_means'].items())}")
    else:
        print(f"  {o['arm']} - {o['comparator']}: mean {o['mean']:+.4f}, pos {o['pos']}, lb {o['lb']:+.4f}, d {', '.join(f'{x:+.4f}' for x in o['d'])}")
