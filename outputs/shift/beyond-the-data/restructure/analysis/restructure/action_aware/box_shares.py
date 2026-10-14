"""READ-ONLY: share of T4 held-out recorded rows and of T3a strong-current commands outside the recorded box (room
lock.json:shared.recorded_range), and pooled existing deployment rules (runs/shift_explore/deploy). No fitting."""
import json
import numpy as np
X = "/home/user/bca/runs/shift_explore"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/action_aware/box_shares.json"
res = {}
for w in ("stall", "no_stall"):
    rows = []
    rules = {}
    for k in range(5):
        f = f"{X}/room/seed{k}/lock.json" if w == "stall" else f"{X}/room/seed{k}/no_stall/lock.json"
        rr = json.load(open(f))["shared"]["recorded_range"]
        lo, hi = np.array(rr["lo"]), np.array(rr["hi"])
        P = np.load(f"{X}/detect/seed{k}/{w}/pairs.npz")
        c = np.asarray(P["commands"], np.float64)
        out = ((c < lo) | (c > hi)).any(1)
        p80 = (np.abs(c) > np.float32(0.8)).any(1)
        t4 = P["set"] == "T4"
        t3a = (P["set"] == "T3") & (P["t3_source"] == "actor") & (P["role"] == "command")
        rows.append(dict(seed=k, T4_outside_box=float(out[t4].mean()), T3a_outside_box=float(out[t3a].mean()),
                         T3a_past80=float(p80[t3a].mean())))
        d = json.load(open(f"{X}/deploy/seed{k}_{w}_gapped.json"))["situations"]
        for sit in ("strong_current", "recorded"):
            for name, v in d[sit]["rules"].items():
                rules.setdefault(sit, {}).setdefault(name, []).append((v["points"], v.get("fire_share")))
    pooled = {sit: {n: dict(points=float(np.mean([x[0] for x in v])),
                            fire=(float(np.mean([x[1] for x in v])) if v[0][1] is not None else None))
                    for n, v in r.items()} for sit, r in rules.items()}
    for sit in pooled:
        for n in list(pooled[sit]):
            if not n.startswith("random:") and f"random:{n}" in pooled[sit]:
                pooled[sit][n]["minus_random"] = pooled[sit][n]["points"] - pooled[sit][f"random:{n}"]["points"]
    res[w] = dict(per_seed=rows, deploy_pooled=pooled)
json.dump(res, open(OUT, "w"), indent=1)
for w in res:
    print(w, [(r["seed"], round(r["T4_outside_box"], 3), round(r["T3a_outside_box"], 3), round(r["T3a_past80"], 3)) for r in res[w]["per_seed"]])
    for n in ("none:None", "unrec:clip99", "always:clip99", "heads:clip99"):
        print("  ", n, {s: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[w]["deploy_pooled"][s][n].items()} for s in ("strong_current", "recorded")})
