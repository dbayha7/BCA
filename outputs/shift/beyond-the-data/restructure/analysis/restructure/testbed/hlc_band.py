"""READ-ONLY evidence E2 (Angle D). A held-out, locally weighted conformal band ("HLC", a numpy STAND-IN for a
restructured BCA band; no BCA module is called, nothing is fitted) read at the trace test bed's pair sets, across the
evidence sweep (shares 0-10%), both worlds, seeds 0-2.

Inputs (all saved): heldout_seed<k>_<world>.npz (E1: training-form one-step errors of saved critics at recorded rows),
the trace test bed's signals.npz (pair sets T1, T1c, T3a and T4 rows), and the Stage 2 weighted band's frozen
classifier (cls_params in runs/shift_explore/stage2/detect/weighted/<world>/fixed/seed<k>/band.msgpack, fitted on the
gapped log against the gapped critic's actor) used only as a forward function.

Score (one-sided, overrating): e_j = -delta_j at held-out calibration rows j (a critic that overrates leaves negative
delta). Band at a query x = (s, a): the weighted conformal quantile of e with a test atom (WBCP, Lou and Luo 2026,
as calibration/wbcp.py computes it, re-implemented here in numpy): R = max(lambda_hat, lambda_hpd), alpha 0.1,
beta 0.95, 200 Bayesian-bootstrap draws (wbcp uses 1,000); +inf (refusal) when the test atom alone keeps the posterior
from crossing. Two weightings, both fixed before any number below was read:
  cls     BCA's own weights: w_j = exp(logit_cls(s_j, a_j)), test mass t = exp(logit_cls(x)) (the frozen classifier).
  kern    a Gaussian kernel on standardised (state, command), test mass 1, lambda_hat only; bandwidth h = median over
          T4 queries of the distance to the 10th-nearest calibration row (no outcome used).
Configurations (calibration rows are held out from the critic whose residuals they carry):
  A   BCA's split as built, on the trace log: critic = the trace critic (gap at share 0); calibration = the gate bank
      plus the trace bank rows (1-14 past-80% rows).
  B   calibrate instead of train: critic = the gate's gapped critic; calibration = gate bank + the share's added
      training rows (33-660 past-80% rows, never seen by the gapped critic).
  C   cross-fit proxy: critic = the trace critic; calibration = gate bank + as many past-80% candidates NOT added at
      that share as were added (held out for the trace critic), drawn by default_rng([20261014, seed, 601, share]).
  POP population limit: critic = gap; calibration = gate bank + all ~6,600 past-80% candidates.
Queries: B and POP read share 0's sets (the gapped critic's actor); A and C read the share's own sets (the trace
critic's actor). T1c (check 1's full-throttle pattern at T1 states) is the same command in both worlds.
Flag level tau: the split-conformal 95% level of U over the 5,000 T4 rows at their recorded commands.
"""
import json
import math
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
from scipy.spatial import cKDTree

OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/testbed"
TRACE = "/home/user/bca/runs/shift_explore/trace"
ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
SHARES = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1)
SHARE_DIR = {0.0: "share_0pct", 0.005: "share_0.5pct", 0.01: "share_1pct", 0.02: "share_2pct", 0.03: "share_3pct",
             0.05: "share_5pct", 0.1: "share_10pct"}
ALPHA, BETA, DRAWS = 0.1, 0.95, 200
SETS = ("T1c", "T1", "T3a")
KNEE = 0.8


class ClassifierNet(nn.Module):  # copy of stage2/code/weights.py ClassifierNet (forward only)
    hidden: int = 64

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs / (1.0 + 0.001), action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)


CLS = ClassifierNet()
CLS_APPLY = jax.jit(CLS.apply)


def past80(a):
    return (np.abs(np.asarray(a, np.float32)) > np.float32(KNEE)).any(1)


def conformal_level(scores, level=0.95):
    s = np.sort(np.asarray(scores, np.float64))
    k = int(math.ceil((len(s) + 1) * level))
    return float(s[k - 1]) if 1 <= k <= len(s) else math.inf


def wbcp_table(scores, weights, log_t, rng, draws=DRAWS):
    """R(t) = max(lambda_hat, lambda_hpd) on a grid of test masses t = exp(log_t) (wbcp.calibrate's rule)."""
    order = np.argsort(scores, kind="stable")
    s, w = np.asarray(scores, np.float64)[order], np.asarray(weights, np.float64)[order]
    scale = w.max()
    w, T = w / scale, np.exp(np.asarray(log_t, np.float64)) / scale
    cum, W = np.cumsum(w), w.sum()
    E = rng.standard_exponential((draws, len(s) + 1))
    mass = E[:, :-1] * w
    cm, tb = np.cumsum(mass, 1), mass.sum(1)
    lam_hat, lam_hpd = [], []
    for t in T:
        need = (1 - ALPHA) * (W + t)
        k = int(np.searchsorted(cum, need * (1 - 1e-12)))
        lam_hat.append(s[k] if k < len(s) else np.inf)
        cov = cm >= ((1 - ALPHA) * (tb + E[:, -1] * t))[:, None]
        cross = np.where(cov.any(1), s[cov.argmax(1)], np.inf)
        cross.sort()
        lam_hpd.append(cross[int(math.ceil(BETA * draws)) - 1])
    return np.maximum(np.asarray(lam_hat), np.asarray(lam_hpd))


def read_table(log_t, R, lq):
    """Conservative step read: R at the first grid point at or above the query's log t (R is non-decreasing in t)."""
    i = np.clip(np.searchsorted(log_t, lq), 0, len(log_t) - 1)
    out = R[i]
    out[lq > log_t[-1]] = np.inf
    return out


def kernel_lambda_hat(Zc, e, Zq, h, chunk=None):
    """Weighted split-conformal lambda_hat with Gaussian kernel weights and test mass K(0) = 1."""
    chunk = chunk or max(1, int(3e6 // (len(Zc) * Zc.shape[1])))
    order = np.argsort(e, kind="stable")
    Zc, e = Zc[order], np.asarray(e, np.float64)[order]
    out = np.empty(len(Zq))
    for i in range(0, len(Zq), chunk):
        d2 = ((Zq[i:i + chunk, None, :] - Zc[None, :, :]) ** 2).sum(-1)
        w = np.exp(-0.5 * d2 / h ** 2)
        cum = np.cumsum(w, 1)
        need = (1 - ALPHA) * (cum[:, -1] + 1.0)
        ok = cum >= need[:, None]
        out[i:i + chunk] = np.where(ok.any(1), e[ok.argmax(1)], np.inf)
    return out


def summary(Ucmd, Ucap, past, tau):
    p = past
    if p.sum() == 0:
        return None
    fin = np.isfinite(Ucmd[p])
    gap = Ucmd[p] - Ucap[p]
    return dict(n_past80=int(p.sum()), refusal=float((~fin).mean()), flag=float((Ucmd[p] > tau).mean()),
                flag_cap=float((Ucap[p] > tau).mean()),
                median_U=float(np.median(Ucmd[p])), median_U_cap=float(np.median(Ucap[p])),
                widening_share=float((Ucmd[p] > Ucap[p]).mean()),
                median_gap=float(np.median(np.where(np.isnan(gap), 0.0, gap))))


def run(rep):
    t0 = time.time()
    res = {}
    for world in ("stall", "no_stall"):
        z = np.load(f"{OUT}/heldout_seed{rep}_{world}.npz")
        rows, X, Aall = z["rows"], np.asarray(z["states"], np.float64), np.asarray(z["actions"], np.float32)
        pos = lambda r: np.searchsorted(rows, np.asarray(r, np.int64))  # noqa: E731
        rec = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{rep}/band.json"))
        mean = np.asarray(rec["data"]["obs_mean"], np.float32)
        std = np.asarray(rec["data"]["obs_std"], np.float32)
        b = serialization.msgpack_restore(open(f"{ST2}/weighted/{world}/fixed/seed{rep}/band.msgpack", "rb").read())
        lw_fn = lambda x, a: np.asarray(CLS_APPLY(b["cls_params"], jnp.asarray(((np.asarray(x, np.float32) - mean) / std)), jnp.asarray(np.asarray(a, np.float32))), np.float64)  # noqa: E731
        lw_all = lw_fn(X, Aall)
        kept = z["pos_kept"]
        Zall = np.concatenate([X, Aall.astype(np.float64)], 1)
        mu, sd = Zall[kept].mean(0), Zall[kept].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Zall = (Zall - mu) / sd
        cand_rows = rows[z["pos_cand"]]
        res[world] = {}
        for config in ("A", "B", "C", "POP"):
            res[world][config] = {}
            for sh in (SHARES if config != "POP" else (1.0,)):
                key = f"{sh:g}"
                if config == "A":
                    crit = "gap" if sh == 0 else f"trace_{key}"
                    extra = pos(z[f"addedbank_{key}"]) if sh > 0 else np.zeros(0, np.int64)
                elif config == "B":
                    crit = "gap"
                    extra = pos(z[f"added_{key}"]) if sh > 0 else np.zeros(0, np.int64)
                elif config == "C":
                    crit = "gap" if sh == 0 else f"trace_{key}"
                    if sh > 0:
                        added = z[f"added_{key}"]
                        rest = np.setdiff1d(cand_rows, added)
                        pick = np.random.default_rng([20261014, rep, 601, int(round(sh * 1000))]).choice(
                            len(rest), len(added), replace=False)
                        extra = pos(np.sort(rest[pick]))
                    else:
                        extra = np.zeros(0, np.int64)
                else:
                    crit, extra = "gap", z["pos_cand"]
                cal = np.concatenate([z["pos_bank"], extra])
                e = -z[f"delta_{crit}"][cal]
                sig = np.load(f"{TRACE}/{world}/seed{rep}/{SHARE_DIR[0.0 if config in ('B', 'POP') else sh]}/signals.npz")
                t4 = pos(sig["T4_rows"])
                q = {"T4": (X[t4], Aall[t4], None)}
                for S in SETS:
                    c = np.asarray(sig[f"{S}|command"], np.float32)
                    q[S] = (np.asarray(sig[f"{S}|states"], np.float64), c,
                            np.clip(c.astype(np.float64), -0.8, 0.8).astype(np.float32))
                # cls weighting
                lq = {S: (lw_fn(v[0], v[1]), None if v[2] is None else lw_fn(v[0], v[2])) for S, v in q.items()}
                lo = min(lw_all[cal].min(), min(x[0].min() for x in lq.values())) - 1.0
                hi = max(lw_all[cal].max(), max(x[0].max() for x in lq.values())) + 1.0
                log_t = np.linspace(lo, hi, 96)
                Rt = wbcp_table(e, np.exp(lw_all[cal] - lw_all[cal].max()) * 1.0, log_t - lw_all[cal].max(),
                                np.random.default_rng([20261014, rep, 602]))
                Ucls = {S: (read_table(log_t, Rt, lq[S][0]), None if lq[S][1] is None else read_table(log_t, Rt, lq[S][1]))
                        for S in q}
                # kernel weighting
                Zc = Zall[cal]
                ztf = lambda x, a: (np.concatenate([x, np.asarray(a, np.float64)], 1) - mu) / sd  # noqa: E731
                tree = cKDTree(Zc)
                h = float(np.median(tree.query(ztf(*q["T4"][:2]), k=10)[0][:, -1]))
                Uk = {S: (kernel_lambda_hat(Zc, e, ztf(v[0], v[1]), h),
                          None if v[2] is None else kernel_lambda_hat(Zc, e, ztf(v[0], v[2]), h)) for S, v in q.items()}
                out = dict(critic=crit, cal_rows=int(len(cal)), cal_past80=int(past80(Aall[cal]).sum()),
                           kern_h=h, cal_score_bank_median=float(np.median(e[:len(z['pos_bank'])])),
                           cal_score_extra_median=float(np.median(e[len(z['pos_bank']):])) if len(extra) else None)
                for wname, U in (("cls", Ucls), ("kern", Uk)):
                    tau = conformal_level(U["T4"][0])
                    out[wname] = dict(tau=tau)
                    for S in SETS:
                        out[wname][S] = summary(U[S][0], U[S][1], past80(q[S][1]), tau)
                res[world][config][key] = out
            print(rep, world, config, f"{time.time() - t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    reps = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    for rep in reps:
        r = run(rep)
        json.dump(r, open(f"{OUT}/hlc_seed{rep}.json", "w"), indent=1, default=float)
