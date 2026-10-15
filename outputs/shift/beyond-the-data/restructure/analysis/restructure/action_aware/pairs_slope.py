"""READ-ONLY. The candidate extrapolation-slope term at the locked pairs (seeds 0-4, both worlds), plus a rough conformal
multiplier for the two-head spread. No fitting.

  U_x(s, a) = sum_k g_k(s, a) e_k(a),  e_k = [a_k - hi_k]_+ + [lo_k - a_k]_+ (room lock.json:shared.recorded_range,
  the hinge's own 0.5%/99.5% bounds), g_k = |dQ1/da_k| of the gapped critic at the projection of a onto [lo, hi].
  dU = U_x(cmd) - U_x(cap). Compared with the label critic's preference (critic_pref; flipping needs k dU >= it) and
  over-preference; harm-vs-novelty AUROC among past-80% commands; wrong flips at past-80% commands that are truly
  better by > 0.05 (the cost side, mostly the no-stall world).
Conformal multiplier for E2 (inferred, not exact): R_E ~ q90 of bank |y - q| (Stage 2 uniform bank_scores.npz:
scores * max(eta, 1e-6) * unit) / median two-head std at T4 held-out recorded rows (bank states are not saved).
"""
import json
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/action_aware")
from shape_sweep import load_critic, idx, bounds, GRADQ1, ST2, DET, OUT  # noqa: E402
from ens_analyse import auroc, qs, med  # noqa: E402

KS = (1, 1.5, 2)


def one(seed, world):
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    data = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))
    mean, std = np.asarray(data["data"]["obs_mean"], np.float32), np.asarray(data["data"]["obs_std"], np.float32)
    lo, hi = bounds(world, seed)
    p = load_critic(world, "stall_aware_gap", seed)
    obs = jnp.asarray(((np.asarray(pz["states"], np.float32) - mean) / std).astype(np.float32))
    cmd = np.asarray(pz["commands"], np.float64)
    cap = np.clip(cmd, -0.8, 0.8)

    def ux(a):
        proj = np.clip(a, lo, hi).astype(np.float32)
        g = np.abs(np.asarray(GRADQ1(p, obs, jnp.asarray(proj)), np.float64))
        e = np.maximum(a - hi, 0) + np.maximum(lo - a, 0)
        return (g * e).sum(1)

    uc, up = ux(cmd), ux(cap)
    harm, past = lab["harmful"].astype(bool), lab["unsupported"].astype(bool)
    over, cpref, tpref = lab["over_pref"], lab["critic_pref"], lab["true_pref"]
    out = dict(seed=seed, world=world)
    for st in ("T1", "T3"):
        ic = idx(pz, st, "command")
        sel = harm[ic]
        pick = sel if sel.any() else past[ic]
        ds = (uc - up)[ic]
        d = dict(sel="harmful" if sel.any() else "past80", n=int(pick.sum()),
                 critic_pref_q50=med(cpref[ic][pick]), over_pref_q50=med(over[ic][pick]),
                 dU_q10_q50_q90=qs(ds[pick]), dU_over_critic_pref_q50=med(ds[pick] / np.maximum(cpref[ic][pick], 1e-9)),
                 dU_over_over_pref_q50=med(ds[pick] / over[ic][pick]),
                 U_cap_q50=med(up[ic][pick]))
        for k in KS:
            d[f"flip_share_k{k:g}"] = float(np.mean(cpref[ic][pick] - k * ds[pick] <= 0))
        good = past[ic] & (tpref[ic] > 0.05)
        if good.any():
            d["good_n"] = int(good.sum())
            d["good_true_pref_q50"] = med(tpref[ic][good])
            for k in KS:
                d[f"good_wrong_flip_share_k{k:g}"] = float(np.mean(cpref[ic][good] - k * ds[good] <= 0))
        pp = past[ic]
        if harm[ic][pp].any() and (~harm[ic][pp]).any():
            d["auroc_harm_among_past80_dU"] = auroc(ds[pp], harm[ic][pp])
        d["dU_past80_all_q50"] = med(ds[pp])
        out[st] = d
    # rough conformal multiplier for the two-head spread (inferred)
    z = np.load(f"{OUT}/q_seed{seed}_{world}.npz")
    t4 = idx(pz, "T4", "recorded")
    s2 = z["cmd__stall_aware_gap"].std(1, ddof=1)
    bs = np.load(f"{ST2}/uniform/{world}/fixed/seed{seed}/bank_scores.npz")
    resid = bs["scores"] * np.maximum(bs["eta"], 1e-6) * float(data["uniform"]["unit"])
    out["E2_conformal_multiplier_estimate"] = float(np.quantile(resid, 0.9) / np.median(s2[t4]))
    out["bank_abs_resid_q90"] = float(np.quantile(resid, 0.9))
    return out


if __name__ == "__main__":
    res = [one(s, w) for w in ("stall", "no_stall") for s in range(5)]
    json.dump(res, open(f"{OUT}/pairs_slope.json", "w"), indent=1)
    for r in res:
        for st in ("T1", "T3"):
            d = r[st]
            print(r["world"], r["seed"], st, d["sel"], d["n"], "cpref", round(d["critic_pref_q50"], 3), "dU q50",
                  round(d["dU_q10_q50_q90"][1], 3), "dU/cpref", round(d["dU_over_critic_pref_q50"], 2),
                  "flip k1/1.5/2", [round(d[f"flip_share_k{k:g}"], 2) for k in KS],
                  "good n", d.get("good_n"), "wrongflip k1/1.5/2", [round(d.get(f"good_wrong_flip_share_k{k:g}", np.nan), 2) for k in KS],
                  "auroc", d.get("auroc_harm_among_past80_dU"))
        print("   E2 conformal multiplier est", round(r["E2_conformal_multiplier_estimate"], 2), "bank |r| q90",
              round(r["bank_abs_resid_q90"], 3))
