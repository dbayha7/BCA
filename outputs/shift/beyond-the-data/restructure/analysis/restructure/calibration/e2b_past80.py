"""READ-ONLY. Past-80% rows that survive in the gapped log: counts in training and bank, and BCA's bank residuals
there vs elsewhere, per world (is there any trace of the stall a calibration could use?)."""
import json
import numpy as np
ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
K = np.float32(0.8)
res = {}
for seed in range(5):
    for world in ("stall", "no_stall"):
        d = np.load(f"{OUT}/data_seed{seed}_{world}.npz")
        bs = np.load(f"{ST2}/uniform/{world}/fixed/seed{seed}/bank_scores.npz")
        unit = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))["uniform"]["unit"]
        r = bs["scores"] * np.maximum(bs["eta"], 1e-6) * unit
        pb = np.abs(d["bank_act"]).max(1) > K
        pt = np.abs(d["train_act"]).max(1) > K
        cur_t = d["train_obs"][:, 10] * d["obs_std"][10] + d["obs_mean"][10]
        res[f"{seed}_{world}"] = dict(train_past80=int(pt.sum()), bank_past80=int(pb.sum()),
                                      train_past80_current_q=[float(np.quantile(cur_t[pt], q)) for q in (.1, .5, .9)] if pt.any() else None,
                                      bank_resid_past80=[round(float(x), 3) for x in np.sort(r[pb])],
                                      bank_resid_q90_rest=float(np.quantile(r[~pb], .9)),
                                      bank_resid_share_above_rest_q90=float(np.mean(r[pb] > np.quantile(r[~pb], .9))) if pb.any() else None)
        print(seed, world, res[f"{seed}_{world}"])
json.dump(res, open(f"{OUT}/e2b_past80.json", "w"), indent=1)
