"""S0 VARIANT (share 0, the gate's own gapped log): calibration = the gate bank plus the 34-59 SURVIVING past-80%
training rows, with the gapped critic's IN-SAMPLE residuals there (a conservative, read-only stand-in for the held-out
residuals a cross-fitted critic would give). Everything else as below.
READ-ONLY evidence E5 (Angle D). POST HOC: chosen after E3/E4 showed that BCA's own (classifier) weights and a
joint (state, command) kernel give no paired gap between the actor's command and its clipped version at the same
state, so the lower-edge comparison never fires once evidence exists. This variant localises the held-out conformal
quantile on the COMMAND only ("akern"):
    w_j(a) = exp(-|(a_j - a) / sd_a|^2 / (2 h^2)),  test mass 1,  lambda_hat at alpha 0.1 (one-sided score -delta_j),
sd_a the per-thruster sd of the kept training commands, h = the median over the 5,000 T4 rows of the distance (in
that space) from the row's recorded command to the 10th-nearest calibration command (no outcome is used).
Calibration rows: config B (the gapped critic's held-out residuals at the gate bank plus the share-p trace rows) and
POP (all past-80% candidates). Readouts, seeds 0-2, both worlds:
  (i)  T1c pairs: coverage of the true overrating -delta*(s, a) (exact, as truth_cover.py), median U, paired gap
       U(s, cmd) - U(s, cap) against the true paired overrating;
  (ii) the deployment check of deploy_lcb.py with U = akern (strong current and recorded conditions, 200 starts).
No training, no fit, no BCA module (the calibrator is not used). Writes only to this folder.
"""
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/user/bca/runs/shift_explore/trace/code")
sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/testbed")
import tracebed as T  # noqa: E402
import hlc_band as H  # noqa: E402
import truth_cover as TCV  # noqa: E402

L, TC, TR = T.L, T.TC, T.TR
OUT = H.OUT
SHARES = (0.0, 0.005, 0.02, 0.1)
SITS = ("strong_current", "recorded")
N_STARTS = 200


class CommandBand:
    def __init__(self, cal_a, e, sd_a, t4_a):
        order = np.argsort(e, kind="stable")
        self.A = np.asarray(cal_a, np.float64)[order] / sd_a
        self.e = np.asarray(e, np.float64)[order]
        self.sd_a = sd_a
        self.h = float(np.median(cKDTree(self.A).query(np.asarray(t4_a, np.float64) / sd_a, k=10)[0][:, -1]))

    def __call__(self, a, chunk=256):
        q = np.asarray(a, np.float64) / self.sd_a
        out = np.empty(len(q))
        for i in range(0, len(q), chunk):
            d2 = ((q[i:i + chunk, None, :] - self.A[None]) ** 2).sum(-1)
            w = np.exp(-0.5 * d2 / self.h ** 2)
            cum = np.cumsum(w, 1)
            ok = cum >= ((1 - H.ALPHA) * (cum[:, -1] + 1.0))[:, None]
            out[i:i + chunk] = np.where(ok.any(1), self.e[ok.argmax(1)], np.inf)
        return out


def run(rep):
    t0 = time.time()
    g = T.GateSeed(rep)
    acts, _, _ = g.h2()
    res = {}
    for wi, world in enumerate(T.WORLDS):
        sd = g.sd[world]
        pg = g.gapped_prepared(world)
        crit = g.critic(world, "gap")[0]
        name = "gapped" if world == "stall" else "no_stall_gapped"
        actor = L.mlp_policy(acts[name], pg, name)
        kept_a = np.asarray(pg.actions, np.float64)
        lo99, hi99 = np.quantile(kept_a, 0.005, axis=0), np.quantile(kept_a, 0.995, axis=0)
        sd_a = kept_a.std(0)
        mean, std = pg.obs_mean.astype(np.float64), pg.obs_std.astype(np.float64)
        z = np.load(f"{OUT}/heldout_seed{rep}_{world}.npz")
        rows, Aall = z["rows"], np.asarray(z["actions"], np.float64)
        pos = lambda r: np.searchsorted(rows, np.asarray(r, np.int64))  # noqa: E731
        sig0 = np.load(f"{H.TRACE}/{world}/seed{rep}/share_0pct/signals.npz")
        t4a = Aall[pos(sig0["T4_rows"])]
        xs = np.asarray(sig0["T1c|states"], np.float64)
        cmd = np.asarray(sig0["T1c|command"], np.float32)
        cap = np.clip(cmd.astype(np.float64), -0.8, 0.8).astype(np.float32)
        rec = np.asarray(sig0["T1|recorded"], np.float32)
        d_cmd = TCV.delta_star(crit, sd.physics, mean, std, xs, cmd, rep)
        d_cap = TCV.delta_star(crit, sd.physics, mean, std, xs, cap, rep)
        d_rec = TCV.delta_star(crit, sd.physics, mean, std, xs, rec, rep)
        bands = {}
        for key in ("akern_S0", "akern_S0_noisy"):
            if key == "akern_S0":
                extra = z["pos_surv"]
            else:
                extra = z["pos_surv"]
            cal = np.concatenate([z["pos_bank"], extra])
            if key == "akern_S0_noisy":
                continue
            bands[key] = CommandBand(Aall[cal], -z["delta_gap"][cal], sd_a, t4a)
        res[world] = dict(actor=name, truth=dict(overrating_cmd=float(np.median(-d_cmd)),
                                                 paired_cap=float(np.median(-(d_cmd - d_cap))),
                                                 paired_rec=float(np.median(-(d_cmd - d_rec)))),
                          bands={}, situations={})
        for key, band in bands.items():
            Uc, Uk, Ur = band(cmd), band(cap), band(rec)
            gk = np.nan_to_num(Uc - Uk, nan=0.0)
            gr = np.nan_to_num(Uc - Ur, nan=0.0)
            res[world]["bands"][key] = dict(
                h=band.h, coverage=float((-d_cmd <= Uc).mean()), median_U=float(np.median(Uc)),
                refusal=float((~np.isfinite(Uc)).mean()), median_U_cap=float(np.median(Uk)),
                median_U_rec=float(np.median(Ur)),
                median_gap_cap=float(np.median(gk)), median_gap_rec=float(np.median(gr)),
                covers_paired_cap=float((-(d_cmd - d_cap) <= gk).mean()),
                covers_paired_rec=float((-(d_cmd - d_rec) <= gr).mean()))
        an = g.anchors(world)
        qf = lambda x, a: L.critic_rows(crit, ((x - mean) / std).astype(np.float32), np.asarray(a, np.float32))[0].min(1)  # noqa: E731
        for si, sit in enumerate(SITS):
            x0 = TC.bank_starts(sit)[:N_STARTS]
            out = {}
            for key, band in list(bands.items()) + [("none", None)]:
                x = np.asarray(x0, np.float64).copy()
                n, Tn = len(x), sd.physics.length
                J = np.zeros(n)
                fired = np.zeros((n, Tn), bool)
                past = np.zeros((n, Tn), bool)
                for t in range(Tn):
                    a = np.clip(actor(x), -1.0, 1.0)
                    f = np.clip(a, lo99, hi99)
                    moved = np.any(a != f, 1)
                    fire = np.zeros(n, bool)
                    if band is not None and moved.any():
                        xm, am, fm = x[moved], a[moved], f[moved]
                        Ua, Uf = band(am), band(fm)
                        la, lf_ = qf(xm, am) - Ua, qf(xm, fm) - Uf
                        fire[moved] = ~np.isfinite(Ua) | (np.isfinite(Uf) & (la < lf_))
                    past[:, t] = TC.past_knee(a, sd.physics)
                    a = np.where(fire[:, None], f, a)
                    fired[:, t] = fire
                    x, r = TC.dynamics(x, a, sd.physics)
                    J = J + r
                pts = TR.points(J, an[sit])
                fp = fired & past
                out[key] = dict(points=float(pts.mean()), fire=float(fired.mean()), past80=float(past.mean()),
                                recall=float(fp.sum() / max(past.sum(), 1)),
                                precision=float(fp.sum() / max(fired.sum(), 1)))
            res[world]["situations"][sit] = out
            print(rep, world, sit, " ".join(f"{k}:{v['points']:.1f}/{v['fire']:.2f}" for k, v in out.items()),
                  f"{time.time() - t0:.0f}s", flush=True)
        print(rep, world, json.dumps({k: (round(v["coverage"], 2), round(v["median_U"], 2), round(v["median_gap_cap"], 2),
                                          round(v["median_gap_rec"], 2)) for k, v in res[world]["bands"].items()}),
              json.dumps(res[world]["truth"]), flush=True)
    return res


if __name__ == "__main__":
    rep = int(sys.argv[1])
    json.dump(run(rep), open(f"{OUT}/akern_s0_seed{rep}.json", "w"), indent=1, default=float)
