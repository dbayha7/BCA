"""READ-ONLY evidence E3 (Angle D): coverage of the TRUE one-step overrating at harmful pairs.

At the trace test bed's T1c pairs (check 1's full-throttle pattern at 1,500 full-throttle-region training states,
the same command in both worlds) and their 80% caps, compute the exact one-step error of a saved critic through the
simulator, in the critic's training form (target averaged over 32 fixed draws of the training noise):
    delta*(s, a) = r(s, a) + 0.98 mean_d min_h Q_h(s', clip(c(s') + eps_d)) - min_h Q_h(s, a),
s' and r from experiments.shift.thruster_chain.dynamics_jax (float64), c the stall-aware controller.
Then ask whether each band COVERS the truth there:
  BCA uniform (saved Stage 2 band, share 0 only): |delta*| <= U = R u eta_f(s, a)   (BCA's own two-sided claim)
  BCA weighted (saved Stage 2 band, share 0 only): |delta*| <= U~ (band.py's width, reconstructed as in IMP/exact.py)
  HLC stand-in (hlc_band.py functions), configs B, C and POP: -delta* <= U (one-sided overrating claim)
No training or fitting; forward passes of saved networks and the simulator only. Writes only to this folder.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/bca/runs/shift_explore/trace/code")
sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/testbed")
import tracebed as T  # noqa: E402
import hlc_band as H  # noqa: E402

L, TC, RC = T.L, T.TC, T.RC
import jax.numpy as jnp  # noqa: E402
from jax.experimental import enable_x64  # noqa: E402
from flax import serialization  # noqa: E402
from calibration.network import Calibrator  # noqa: E402  (forward pass of the SAVED eta_f only, as IMP/exact.py)

OUT = H.OUT
CAL = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True)


def delta_star(critic, physics, mean, std, x, a, rep):
    x = np.asarray(x, np.float64)
    a = np.asarray(a, np.float64)
    with enable_x64():
        x1, r = TC.dynamics_jax(jnp.asarray(x), jnp.asarray(a), physics)
        c1 = RC.stall_aware(physics).jax(x1)
        x1, r, c1 = np.asarray(x1), np.asarray(r), np.asarray(c1)
    o = ((x - mean) / std).astype(np.float32)
    o1 = ((x1 - mean) / std).astype(np.float32)
    q0 = L.critic_rows(critic, o, a.astype(np.float32))[0].min(1)
    rng = np.random.default_rng([20261014, rep, 603])
    acc = np.zeros(len(x))
    for _ in range(32):
        eps = np.clip(0.2 * rng.standard_normal(c1.shape), -0.5, 0.5)
        acc += L.critic_rows(critic, o1, np.clip(c1 + eps, -1, 1).astype(np.float32))[0].min(1)
    return r + 0.98 * acc / 32 - q0


def hlc_at(z, lw_all, Zall, mu, sd, lw_fn, crit, extra, xs, acts, t4x, t4a, rep):
    cal = np.concatenate([z["pos_bank"], extra])
    e = -z[f"delta_{crit}"][cal]
    lq = [lw_fn(x, a) for x, a in zip(xs, acts)]
    lo = min(lw_all[cal].min(), min(v.min() for v in lq)) - 1.0
    hi = max(lw_all[cal].max(), max(v.max() for v in lq)) + 1.0
    log_t = np.linspace(lo, hi, 96)
    m = lw_all[cal].max()
    Rt = H.wbcp_table(e, np.exp(lw_all[cal] - m), log_t - m, np.random.default_rng([20261014, rep, 602]))
    Ucls = [H.read_table(log_t, Rt, v) for v in lq]
    Zc = Zall[cal]
    ztf = lambda x, a: (np.concatenate([x, np.asarray(a, np.float64)], 1) - mu) / sd  # noqa: E731
    h = float(np.median(H.cKDTree(Zc).query(ztf(t4x, t4a), k=10)[0][:, -1]))
    Uk = [H.kernel_lambda_hat(Zc, e, ztf(x, a), h) for x, a in zip(xs, acts)]
    return Ucls, Uk


def cover(U, truth):
    U, truth = np.asarray(U), np.asarray(truth)
    return dict(coverage=float((truth <= U).mean()), median_truth=float(np.median(truth)),
                median_U=float(np.median(U)), refusal=float((~np.isfinite(U)).mean()))


def run(rep):
    t0 = time.time()
    g = T.GateSeed(rep)
    res = {}
    for world in T.WORLDS:
        sd_ = g.sd[world]
        prep = g.gapped_prepared(world)
        mean, std = prep.obs_mean.astype(np.float64), prep.obs_std.astype(np.float64)
        z = np.load(f"{OUT}/heldout_seed{rep}_{world}.npz")
        rows, X, Aall = z["rows"], np.asarray(z["states"], np.float64), np.asarray(z["actions"], np.float32)
        pos = lambda r: np.searchsorted(rows, np.asarray(r, np.int64))  # noqa: E731
        b = serialization.msgpack_restore(open(f"{H.ST2}/weighted/{world}/fixed/seed{rep}/band.msgpack", "rb").read())
        bu = serialization.msgpack_restore(open(f"{H.ST2}/uniform/{world}/fixed/seed{rep}/band.msgpack", "rb").read())
        m32, s32 = prep.obs_mean, prep.obs_std
        lw_fn = lambda x, a: np.asarray(H.CLS_APPLY(b["cls_params"], jnp.asarray(((np.asarray(x, np.float32) - m32) / s32)), jnp.asarray(np.asarray(a, np.float32))), np.float64)  # noqa: E731
        lw_all = lw_fn(X, Aall)
        Zall = np.concatenate([X, Aall.astype(np.float64)], 1)
        mu, sdv = Zall[z["pos_kept"]].mean(0), Zall[z["pos_kept"]].std(0)
        sdv = np.where(sdv > 0, sdv, 1.0)
        Zall = (Zall - mu) / sdv
        sig0 = np.load(f"{H.TRACE}/{world}/seed{rep}/share_0pct/signals.npz")
        t4 = pos(sig0["T4_rows"])
        xs = np.asarray(sig0["T1c|states"], np.float64)
        cmd = np.asarray(sig0["T1c|command"], np.float32)
        cap = np.clip(cmd.astype(np.float64), -0.8, 0.8).astype(np.float32)
        out = {}
        # saved BCA bands at share 0 (gapped critic)
        crit_gap = g.critic(world, "gap")[0]
        d_cmd = delta_star(crit_gap, sd_.physics, mean, std, xs, cmd, rep)
        d_cap = delta_star(crit_gap, sd_.physics, mean, std, xs, cap, rep)
        o32 = ((xs - mean) / std).astype(np.float32)
        eta_u = np.asarray(CAL.apply(bu["cal_params"], jnp.asarray(o32), jnp.asarray(cmd)), np.float64)
        Uu = float(bu["R"]) * float(bu["unit"]) * np.maximum(eta_u, 1e-6)
        eta_w = np.asarray(CAL.apply(b["cal_params"], jnp.asarray(o32), jnp.asarray(cmd)), np.float64)
        lw = lw_fn(xs, cmd)
        Rw = np.interp(lw, np.asarray(b["log_t"], np.float64), np.asarray(b["R_tab"], np.float64))
        Uw = Rw * float(b["unit"]) * np.maximum(eta_w, 1e-6) + float(b["U_max"]) * np.logaddexp(0.0, lw - float(b["l_star"]))
        out["truth_gap"] = dict(overrating_cmd=float(np.median(-d_cmd)), overrating_cap=float(np.median(-d_cap)),
                                paired_overrating=float(np.median(-(d_cmd - d_cap))))
        out["bca_uniform_two_sided"] = cover(Uu, np.abs(d_cmd))
        out["bca_weighted_two_sided"] = cover(Uw, np.abs(d_cmd))
        out["bca_uniform_one_sided"] = cover(Uu, -d_cmd)
        out["bca_weighted_one_sided"] = cover(Uw, -d_cmd)
        # HLC stand-in
        cand_rows = rows[z["pos_cand"]]
        for config in ("B", "C", "POP"):
            for sh in ((1.0,) if config == "POP" else (0.005, 0.02, 0.05, 0.1)):
                key = f"{sh:g}"
                if config == "B":
                    crit, extra, cp = "gap", pos(z[f"added_{key}"]), crit_gap
                    dc, dk = d_cmd, d_cap
                elif config == "POP":
                    crit, extra, cp = "gap", z["pos_cand"], crit_gap
                    dc, dk = d_cmd, d_cap
                else:
                    crit = f"trace_{key}"
                    added = z[f"added_{key}"]
                    rest = np.setdiff1d(cand_rows, added)
                    pick = np.random.default_rng([20261014, rep, 601, int(round(sh * 1000))]).choice(
                        len(rest), len(added), replace=False)
                    extra = pos(np.sort(rest[pick]))
                    cp = T.L.load_tree(T.job_dir(world, rep, sh) / "critic.msgpack", L.critic_template(),
                                       json.load(open(T.job_dir(world, rep, sh) / "lock.json"))["trace_critic_file_sha256"])
                    dc = delta_star(cp, sd_.physics, mean, std, xs, cmd, rep)
                    dk = delta_star(cp, sd_.physics, mean, std, xs, cap, rep)
                (Uc_cmd, Uc_cap), (Uk_cmd, Uk_cap) = hlc_at(z, lw_all, Zall, mu, sdv, lw_fn, crit, extra,
                                                            [xs, xs], [cmd, cap], X[t4], Aall[t4], rep)
                gap_true = -(dc - dk)
                out[f"{config}_{key}"] = dict(
                    critic=crit, truth_overrating_cmd=float(np.median(-dc)), truth_paired=float(np.median(gap_true)),
                    cls=cover(Uc_cmd, -dc), kern=cover(Uk_cmd, -dc),
                    cls_paired=dict(median_gap=float(np.median(np.nan_to_num(Uc_cmd - Uc_cap, nan=0.0))),
                                    covers_paired=float((gap_true <= np.nan_to_num(Uc_cmd - Uc_cap, nan=0.0)).mean())),
                    kern_paired=dict(median_gap=float(np.median(np.nan_to_num(Uk_cmd - Uk_cap, nan=0.0))),
                                     covers_paired=float((gap_true <= np.nan_to_num(Uk_cmd - Uk_cap, nan=0.0)).mean())))
        res[world] = out
        print(rep, world, f"{time.time() - t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    reps = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    for rep in reps:
        json.dump(run(rep), open(f"{OUT}/truth_cover_seed{rep}.json", "w"), indent=1, default=float)
