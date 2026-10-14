"""Read-only post-processing of capsweep_seed*.json: where the BCA-placed box sits relative to the symmetric-cap
frontier, per seed and world (equivalent symmetric cap by linear interpolation of strong-current points)."""
import json
import numpy as np
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/consumer"
res = []
for k in range(5):
    d = json.load(open(f"{OUT}/capsweep_seed{k}.json"))
    caps = d["caps"]
    row = dict(rep=k)
    for w in ("stall", "no_stall"):
        for a in ("none", "bca_k1"):
            sc = d["worlds"][w]["actors"][a]["strong_current"]
            y = np.array([sc[f"cap_{c:g}"]["points"] for c in caps])
            box = sc["bca_box"]["points"]
            # points fall (stall) or rise (no_stall) monotonically with c past 0.8; interpolate on c >= 0.8
            cc, yy = np.array(caps[1:]), y[1:]
            order = np.argsort(yy)
            ceq = float(np.interp(box, yy[order], cc[order]))
            row[f"{w}_{a}_box_points"] = box
            row[f"{w}_{a}_equiv_cap"] = ceq
        row[f"{w}_box_th1_neg"] = -d["worlds"][w]["boxes"]["bca_box"][0][0]
        row[f"{w}_box"] = d["worlds"][w]["boxes"]["bca_box"]
    res.append(row)
    print(k, "box th1- stall %.3f no_stall %.3f | equiv cap (none) stall %.3f no_stall %.3f | (bca_k1) stall %.3f no_stall %.3f" % (
        row["stall_box_th1_neg"], row["no_stall_box_th1_neg"], row["stall_none_equiv_cap"], row["no_stall_none_equiv_cap"],
        row["stall_bca_k1_equiv_cap"], row["no_stall_bca_k1_equiv_cap"]))
    print("   boxes stall", np.round(row["stall_box"], 3).tolist(), "no_stall", np.round(row["no_stall_box"], 3).tolist())
json.dump(res, open(f"{OUT}/box_frontier.json", "w"), indent=1)
