"""READ-ONLY numpy. (1) The perfect band's size at the same T1 harmful pairs (its score is already paired against
the reference command), the yardstick for how large a k = 1 penalty must be (it won back 94 points at k = 1).
(2) A lower-edge selection proxy at T3a command pairs (strong-current states the no-caution actor visits): the lower
edge prefers the 80% cap when U(s, a) - U(s, cap) > Q1(s, a) - Q1(s, cap) (critic_pref from labels.npz, the frozen
gapped critic's own preference). Share of pairs where it picks the cap, by world: a world-selective band should pick
the cap in the stalling world and keep the command in the no-stall world.
(3) Seed 3 no-stall under BCA's refusal ('cannot certify' -> maximum caution: R_tail := the largest training score)."""
import json
import numpy as np
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
DET = "/home/user/bca/runs/shift_explore/detect"
res = {"pb_size": {}, "le_select": {}}
for world in ("stall", "no_stall"):
    for seed in range(5):
        pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
        lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
        W = np.load(f"{OUT}/e4_widths_seed{seed}_{world}.npz")
        st, ro, pr = pz["set"], pz["role"], pz["pair"]

        def idx(s, r):
            rows = np.flatnonzero((st == s) & (ro == r))
            return rows[np.argsort(pr[rows])]
        ic, ip = idx("T1", "command"), idx("T1", "cap")
        H, past, over, cp = lab["harmful"], lab["unsupported"], lab["over_pref"], lab["critic_pref"]
        if H[ic].any():
            pb = W["perfect_band"]
            sel = H[ic]
            res["pb_size"][f"{seed}"] = dict(size_median=float(np.median(((pb[ic] - pb[ip]) / over[ic])[sel])),
                                            pb_cmd_median=float(np.median(pb[ic][sel])),
                                            over_pref_median=float(np.median(over[ic][sel])))
        c3, p3 = idx("T3", "command"), idx("T3", "cap")
        a_src = pz["t3_source"][c3] == "actor"
        m = a_src & past[c3]
        o = {}
        for nm in ("tail_B", "tail_A", "weighted_bca", "classifier_ratio", "range_hinge"):
            U = W[nm]
            if nm in ("classifier_ratio", "range_hinge"):
                continue
            pick_cap = (U[c3] - U[p3]) > cp[c3]
            o[nm] = dict(pick_cap_share_past80=float(np.mean(pick_cap[m])),
                         pick_cap_share_harmful=float(np.mean(pick_cap[m & H[c3]])) if (m & H[c3]).any() else None)
        res["le_select"][f"{seed}_{world}"] = o
        print(seed, world, {k: {kk: round(vv, 3) for kk, vv in v.items() if vv is not None} for k, v in o.items()})
print(json.dumps(res["pb_size"], indent=0))
# (3) seed 3 no-stall refusal reading
d = np.load(f"{OUT}/delta_seed3_no_stall.npz")["train_delta"]
e4 = json.load(open(f"{OUT}/e4_tailband.json"))
r = [x for x in e4 if x["seed"] == 3 and x["world"] == "no_stall"][0]
unit = r["unit"]
sB = -d / unit
res["seed3_no_stall_refusal"] = dict(B_ramp_value_if_max_caution=float((sB.max() - r["R_B"]["bulk"]) * unit),
                                     B_ramp_value_if_Rc_plain=float((r["R_B"]["tail_plain"] - r["R_B"]["bulk"]) * unit),
                                     n_tail=r["n_tail"])
print(res["seed3_no_stall_refusal"])
for w in ("stall", "no_stall"):
    for nm in ("tail_B", "tail_A", "weighted_bca"):
        v = [res["le_select"][f"{s}_{w}"][nm]["pick_cap_share_past80"] for s in range(5)]
        print(w, nm, "pick-cap share at past-80% T3a commands, mean", round(float(np.mean(v)), 3), [round(x, 3) for x in v])
json.dump(res, open(f"{OUT}/e5_pb_le.json", "w"), indent=1)
