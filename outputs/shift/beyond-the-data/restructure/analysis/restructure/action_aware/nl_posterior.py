"""READ-ONLY. Last-layer Bayesian posterior ("neural-linear") of the SAVED gapped critic, evaluated at the locked pairs.

No gradient step and no learned parameter: the critic is frozen; its penultimate ReLU features phi_h(s, a) (256 + a
bias 1, per head h) are read at the gapped log's kept training rows (prepared('gap').train, loaded read-only through
the gate's verified loaders: runs/shift_explore/detect/code/common.context -> room harness SeedContext ->
acceptance.load_seed), and the Gaussian linear-model posterior of the last layer is formed in closed form:

    Sigma_h = sigma^2 (Phi_h' Phi_h + lam I)^-1,   s_NL,h(x) = sqrt(phi_h(x)' Sigma_h phi_h(x)),   s_NL = mean_h s_NL,h

sigma = u_f, the Stage 2 uniform band's unit (EMA std of the TD residual; band.json:uniform.unit). lam = c * mean
diagonal of Phi'Phi, c in (1e-6, 1e-4, 1e-2) (prior strength sensitivity; not tuned on outcomes).
Also the leverage ratio h(x) / median h(recorded), h = phi' (Phi'Phi + lam I)^-1 phi (scale-free).
"""
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, "/home/user/bca")
sys.path.insert(0, "/home/user/bca/runs/shift_explore/detect/code")
import common as DC  # noqa: E402  (read-only loaders; writes nothing unless its save functions are called)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import serialization  # noqa: E402
import algorithms.td3_bc as BASE  # noqa: E402
from ens_analyse import idx, auroc, qs, med  # noqa: E402

G = "/home/user/bca/runs/shift/68d9e8779e148b7b/critics/base"
ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
DET = "/home/user/bca/runs/shift_explore/detect"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/action_aware"
DX, DA = 13, 4
CRITIC = BASE.DualCritic(DX, DA)
TEMPLATE = CRITIC.init(jax.random.PRNGKey(0), jnp.zeros(DX), jnp.zeros(DA))
CS = (1e-6, 1e-4, 1e-2)
KS = (1, 2, 5, 10)


def load_gap_critic(world, seed):
    d = f"{G}/{world}/stall_aware_gap/seed{seed}"
    rec = json.load(open(f"{d}/record.json"))
    blob = open(f"{d}/critic.msgpack", "rb").read()
    assert hashlib.sha256(blob).hexdigest() == rec["result"]["file_sha256"]
    return serialization.from_bytes(TEMPLATE, blob)


def feats(p, obs, act):
    v = p["params"]["VmapCritic_0"]
    x = np.concatenate([np.asarray(obs, np.float64), np.asarray(act, np.float64)], 1)
    out, q = [], []
    for h in range(2):
        h1 = np.maximum(x @ np.asarray(v["Dense_0"]["kernel"][h], np.float64) + np.asarray(v["Dense_0"]["bias"][h]), 0)
        h2 = np.maximum(h1 @ np.asarray(v["Dense_1"]["kernel"][h], np.float64) + np.asarray(v["Dense_1"]["bias"][h]), 0)
        q.append(h2 @ np.asarray(v["Dense_2"]["kernel"][h], np.float64)[:, 0] + float(v["Dense_2"]["bias"][h][0]))
        out.append(np.concatenate([h2, np.ones((len(x), 1))], 1))
    return out, np.stack(q, 1)


def one(seed, world):
    ctx = DC.context(seed, worlds=(world,))
    pg = ctx.prepared(world)
    p = load_gap_critic(world, seed)
    tr_obs, tr_act = np.asarray(pg.train.obs, np.float32), np.asarray(pg.train.action, np.float32)
    bu = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))
    sigma = float(bu["uniform"]["unit"])
    mean, std = np.asarray(pg.obs_mean, np.float32), np.asarray(pg.obs_std, np.float32)
    assert np.allclose(mean, np.asarray(bu["data"]["obs_mean"], np.float32)) and \
        np.allclose(std, np.asarray(bu["data"]["obs_std"], np.float32))
    Ftr, _ = feats(p, tr_obs, tr_act)
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    obs = ((np.asarray(pz["states"], np.float32) - mean) / std).astype(np.float32)
    cmd = np.asarray(pz["commands"], np.float32)
    cap = np.clip(cmd.astype(np.float64), -0.8, 0.8).astype(np.float32)
    Fc, qc = feats(p, obs, cmd)
    Fp, _ = feats(p, obs, cap)
    chk = float(np.max(np.abs(qc[:, 0] - lab["q1_command"])))
    harm, past = lab["harmful"].astype(bool), lab["unsupported"].astype(bool)
    over, cpref, tpref = lab["over_pref"], lab["critic_pref"], lab["true_pref"]
    out = dict(seed=seed, world=world, sigma_unit=sigma, n_train=int(len(tr_obs)), q1_recompute_max_abs=chk)
    for c in CS:
        sc, sp, lc, lp = np.zeros(len(cmd)), np.zeros(len(cmd)), np.zeros(len(cmd)), np.zeros(len(cmd))
        for h in range(2):
            Gm = Ftr[h].T @ Ftr[h]
            lam = c * float(np.mean(np.diag(Gm)))
            Ginv = np.linalg.inv(Gm + lam * np.eye(Gm.shape[0]))
            lev_c = np.einsum("ij,jk,ik->i", Fc[h], Ginv, Fc[h])
            lev_p = np.einsum("ij,jk,ik->i", Fp[h], Ginv, Fp[h])
            sc += 0.5 * sigma * np.sqrt(np.maximum(lev_c, 0))
            sp += 0.5 * sigma * np.sqrt(np.maximum(lev_p, 0))
            lc += 0.5 * lev_c
            lp += 0.5 * lev_p
        t4 = idx(pz, "T4", "recorded")
        e = dict(s_T4_q10_q50_q90=qs(sc[t4]), lev_T4_median=med(lc[t4]))
        for st in ("T1", "T3"):
            ic, ip = idx(pz, st, "command"), idx(pz, st, "cap")
            sel = harm[ic]
            pick = sel if sel.any() else past[ic]
            ds = sc[ic] - sp[ic]
            d = dict(sel="harmful" if sel.any() else "past80", n=int(pick.sum()), over_pref_median=med(over[ic][pick]),
                     s_cmd_q10_q50_q90=qs(sc[ic][pick]), s_cap_q10_q50_q90=qs(sp[ic][pick]),
                     ds_q10_q50_q90=qs(ds[pick]), k_need_median=med(over[ic][pick] / np.maximum(ds[pick], 1e-9)),
                     lev_ratio_cmd_over_T4_median=med(lc[ic][pick] / med(lc[t4])),
                     lev_ratio_cap_over_T4_median=med(lp[ic][pick] / med(lc[t4])))
            for k in KS:
                d[f"flip_share_k{k}"] = float(np.mean(cpref[ic][pick] - k * ds[pick] <= 0))
            if st == "T1":
                ir = idx(pz, st, "recorded")
                d["s_rec_q50"] = med(sc[ir][pick])
            good = past[ic] & (tpref[ic] > 0.05)
            if good.any():
                d["good_n"] = int(good.sum())
                d["good_ds_q50"] = med(ds[good])
                for k in KS:
                    d[f"good_wrong_flip_share_k{k}"] = float(np.mean(cpref[ic][good] - k * ds[good] <= 0))
            pp = past[ic]
            if harm[ic][pp].any() and (~harm[ic][pp]).any():
                d["auroc_harm_among_past80_ds"] = auroc(ds[pp], harm[ic][pp])
            d["ds_past80_all_q50"] = med(ds[pp])
            e[st] = d
        out[f"c{c:g}"] = e
    return out


if __name__ == "__main__":
    res = []
    for world in ("stall", "no_stall"):
        for seed in range(5):
            r = one(seed, world)
            res.append(r)
            json.dump(res, open(f"{OUT}/nl_posterior.json", "w"), indent=1)
            print(world, seed, "sigma", round(r["sigma_unit"], 3), "chk", r["q1_recompute_max_abs"], flush=True)
            for c in CS:
                e = r[f"c{c:g}"]
                for st in ("T1", "T3"):
                    d = e[st]
                    print(f"   c={c:g} {st} {d['sel']} ds50={d['ds_q10_q50_q90'][1]:.4f} s_cmd50={d['s_cmd_q10_q50_q90'][1]:.4f}"
                          f" s_cap50={d['s_cap_q10_q50_q90'][1]:.4f} kneed={d['k_need_median']:.0f}"
                          f" levratio cmd/cap={d['lev_ratio_cmd_over_T4_median']:.1f}/{d['lev_ratio_cap_over_T4_median']:.1f}"
                          f" flip k1/5/10={d['flip_share_k1']:.2f}/{d['flip_share_k5']:.2f}/{d['flip_share_k10']:.2f}"
                          f" auroc={d.get('auroc_harm_among_past80_ds')}", flush=True)
                print(f"   c={c:g} T4 s q10/50/90={[round(v, 4) for v in e['s_T4_q10_q50_q90']]}", flush=True)
