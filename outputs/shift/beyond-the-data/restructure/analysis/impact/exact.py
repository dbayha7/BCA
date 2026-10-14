"""READ-ONLY exact decomposition of the saved weighted BCA band (Stage 2 detection, seeds 0-4) into its two terms.

Re-evaluates the SAVED frozen networks (eta_f and the classifier) stored in each weighted band.msgpack at the saved
pair sets. No training, no fitting, no new band: forward passes of two 2-layer networks on 24,500 pairs per seed and
world. Verifies term1 + term2 equals the saved weighted scores, then writes per-pair terms to this folder only.

  U~ = R_c(w) * u_f * max(eta_f, 1e-6)  [term1]  +  U_max * softplus(log w - l*)  [term2]
"""
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization

sys.path.insert(0, "/home/user/bca")
from calibration.network import Calibrator  # noqa: E402

ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
DET = "/home/user/bca/runs/shift_explore/detect"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact"


class ClassifierNet(nn.Module):  # copy of stage2/code/weights.py ClassifierNet
    hidden: int = 64

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs / (1.0 + 0.001), action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)


CAL = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True)
CLS = ClassifierNet()


def run(seed, world):
    rec = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))
    mean = np.asarray(rec["data"]["obs_mean"], np.float32)
    std = np.asarray(rec["data"]["obs_std"], np.float32)
    b = serialization.msgpack_restore(open(f"{ST2}/weighted/{world}/fixed/seed{seed}/band.msgpack", "rb").read())
    bu = serialization.msgpack_restore(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.msgpack", "rb").read())
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    obs = (np.asarray(pz["states"], np.float32) - mean) / std
    act = np.asarray(pz["commands"], np.float32)
    eta = np.asarray(jax.jit(CAL.apply)(b["cal_params"], jnp.asarray(obs), jnp.asarray(act)), np.float64)
    eta_u = np.asarray(jax.jit(CAL.apply)(bu["cal_params"], jnp.asarray(obs), jnp.asarray(act)), np.float64)
    lw = np.asarray(jax.jit(CLS.apply)(b["cls_params"], jnp.asarray(obs), jnp.asarray(act)), np.float64)
    unit, U_max, l_star = float(b["unit"]), float(b["U_max"]), float(b["l_star"])
    log_t, R_tab = np.asarray(b["log_t"], np.float64), np.asarray(b["R_tab"], np.float64)
    Rw = np.interp(lw, log_t, R_tab)
    t1 = Rw * unit * np.maximum(eta, 1e-6)
    t2 = U_max * np.logaddexp(0.0, lw - l_star)
    Uw = np.load(f"{ST2}/weighted/{world}/fixed/seed{seed}/scores/bca_width_weighted_fixed.npz")["scores"]
    Uu = np.load(f"{ST2}/uniform/{world}/fixed/seed{seed}/scores/bca_width_uniform_fixed.npz")["scores"]
    lw_rehearsal = np.load(f"{DET}/seed{seed}/{world}/scores/classifier_ratio.npz")["scores"]
    Uu_re = float(bu["R"]) * float(bu["unit"]) * np.maximum(eta_u, 1e-6)
    chk = dict(seed=seed, world=world,
               weighted_recon_max_abs_err=float(np.max(np.abs(t1 + t2 - Uw))),
               weighted_recon_max_rel_err=float(np.max(np.abs(t1 + t2 - Uw) / np.maximum(Uw, 1e-9))),
               uniform_recon_max_abs_err=float(np.max(np.abs(Uu_re - Uu))),
               eta_same_as_uniform_fit_max_abs=float(np.max(np.abs(eta - eta_u))),
               lw_vs_rehearsal_classifier_max_abs=float(np.max(np.abs(lw - lw_rehearsal))),
               lw_vs_rehearsal_classifier_median_abs=float(np.median(np.abs(lw - lw_rehearsal))),
               lw_vs_rehearsal_spearman=float(__import__("scipy.stats").stats.spearmanr(lw, lw_rehearsal).correlation),
               unit=unit, U_max=U_max, l_star=l_star, R_uniform=float(bu["R"]),
               R_tab_min=float(R_tab.min()), R_tab_max=float(R_tab.max()))
    np.savez(f"{OUT}/terms_seed{seed}_{world}.npz", eta=eta, lw=lw, Rw=Rw, term1=t1, term2=t2, Uw=Uw, Uu=Uu)
    return chk


if __name__ == "__main__":
    out = [run(s, w) for w in ("stall", "no_stall") for s in range(5)]
    json.dump(out, open(f"{OUT}/exact_checks.json", "w"), indent=1)
    for c in out:
        print(json.dumps(c))
