"""READ-ONLY evidence E4 (Angle D): the deployment check as a calibrated LOWER-EDGE comparison, across the evidence
sweep and both worlds. No training and no fit: exact 100-step rollouts of SAVED actors (check 2's gapped critic's
actor of each world), forward passes of the SAVED gapped critic, the SAVED Stage 2 bands (eta_f, classifier, tables)
and the HLC stand-in tables (hlc_band.wbcp_table over held-out residuals of the saved gapped critic, E1).

Rule (fixed before any rollout): at every step the actor proposes a = clip(pi(x), -1, 1); the fallback is
f = clip(a, lo99, hi99), the recorded range (the fallback diagnostic's clip99: 0.5% and 99.5% quantiles of the gapped
training commands). The check replaces a by f when the actor's command has the lower certified value:
    fire  iff  U(x, a) = +inf (cannot certify: strongest caution)  or  minQ(x, a) - U(x, a) < minQ(x, f) - U(x, f)
(a +inf U at f alone never fires). U is one of:
  bca_uniform    the saved Stage 2 uniform band  R u eta_f(x, a)                       (BCA as built, share 0)
  bca_weighted   the saved Stage 2 weighted band (band.py's width, as IMP/exact.py)     (BCA as built, share 0)
  hlc_B_<p>      the HLC stand-in, config B: the gapped critic's held-out residuals at the gate bank plus the
                 share-p trace rows (never seen by that critic), BCA's own classifier weights, WBCP R(t)
  hlc_POP        the same with every past-80% candidate row held out (population limit)
Baselines: none; always (clip every step: the recorded-range limit); random at the rate of selected hlc rules (coin
default_rng([20261014, seed, 700, world, situation, rule])).
Starts: the first 200 starts of each 1,000-start bank (the fallback diagnostic's subset), so 'none' and 'always' are
checked against runs/shift_explore/fallback/seed<k>_<world>_<actor>.json.
"""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/bca/runs/shift_explore/trace/code")
sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/testbed")
import tracebed as T  # noqa: E402
import hlc_band as H  # noqa: E402

L, TC, TR = T.L, T.TC, T.TR
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import serialization  # noqa: E402
from calibration.network import Calibrator  # noqa: E402  (forward pass of the SAVED eta_f only)

OUT = H.OUT
CAL = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True)
CAL_APPLY = jax.jit(CAL.apply)
N_STARTS = 200
HLC_SHARES = (0.0, 0.005, 0.02, 0.1)
RANDOM_FOR = ("hlc_B_0.02",)
SITS = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("strong_current", "recorded")


def run(rep):
    t0 = time.time()
    g = T.GateSeed(rep)
    acts, rec2, _ = g.h2()
    res = {}
    for wi, world in enumerate(T.WORLDS):
        sd = g.sd[world]
        pg = g.gapped_prepared(world)
        crit = g.critic(world, "gap")[0]
        name = "gapped" if world == "stall" else "no_stall_gapped"
        actor = L.mlp_policy(acts[name], pg, name)
        kept_a = np.asarray(pg.actions, np.float64)
        lo99, hi99 = np.quantile(kept_a, 0.005, axis=0), np.quantile(kept_a, 0.995, axis=0)
        m32, s32 = pg.obs_mean, pg.obs_std
        norm = lambda x: jnp.asarray(((np.asarray(x, np.float64) - m32) / s32).astype(np.float32))  # noqa: E731
        b = serialization.msgpack_restore(open(f"{H.ST2}/weighted/{world}/fixed/seed{rep}/band.msgpack", "rb").read())
        bu = serialization.msgpack_restore(open(f"{H.ST2}/uniform/{world}/fixed/seed{rep}/band.msgpack", "rb").read())
        lwf = lambda o, a: np.asarray(H.CLS_APPLY(b["cls_params"], o, jnp.asarray(np.asarray(a, np.float32))), np.float64)  # noqa: E731
        etaf = lambda p, o, a: np.asarray(CAL_APPLY(p, o, jnp.asarray(np.asarray(a, np.float32))), np.float64)  # noqa: E731
        qf = lambda o, a: L.critic_rows(crit, np.asarray(o), np.asarray(a, np.float32))[0].min(1)  # noqa: E731
        # HLC tables (config B and POP)
        z = np.load(f"{OUT}/heldout_seed{rep}_{world}.npz")
        rows, X, Aall = z["rows"], np.asarray(z["states"], np.float64), np.asarray(z["actions"], np.float32)
        pos = lambda r: np.searchsorted(rows, np.asarray(r, np.int64))  # noqa: E731
        lw_all = lwf(norm(X), Aall)
        tables = {}
        for sh in HLC_SHARES + ("POP",):
            if sh == "POP":
                extra, key = z["pos_cand"], "hlc_POP"
            else:
                extra = pos(z[f"added_{sh:g}"]) if sh > 0 else np.zeros(0, np.int64)
                key = f"hlc_B_{sh:g}"
            cal = np.concatenate([z["pos_bank"], extra])
            e = -z["delta_gap"][cal]
            lo = lw_all[cal].min() - 5.0
            log_t = np.linspace(lo, 40.0, 240)
            mx = lw_all[cal].max()
            Rt = H.wbcp_table(e, np.exp(lw_all[cal] - mx), log_t - mx, np.random.default_rng([20261014, rep, 602]))
            tables[key] = (log_t, Rt, int(len(extra)))

        def width(kind, o, a):
            if kind == "bca_uniform":
                return float(bu["R"]) * float(bu["unit"]) * np.maximum(etaf(bu["cal_params"], o, a), 1e-6)
            lw = lwf(o, a)
            if kind == "bca_weighted":
                Rw = np.interp(lw, np.asarray(b["log_t"], np.float64), np.asarray(b["R_tab"], np.float64))
                return (Rw * float(b["unit"]) * np.maximum(etaf(b["cal_params"], o, a), 1e-6)
                        + float(b["U_max"]) * np.logaddexp(0.0, lw - float(b["l_star"])))
            log_t, Rt, _ = tables[kind]
            return H.read_table(log_t, Rt, lw)

        an = g.anchors(world)
        res[world] = dict(actor=name, lo99=lo99.tolist(), hi99=hi99.tolist(),
                          hlc_extra_rows={k: v[2] for k, v in tables.items()}, situations={})
        for si, sit in enumerate(TC.SITUATIONS):
            if sit not in SITS:
                continue
            x0 = TC.bank_starts(sit)[:N_STARTS]
            out = {}

            def rollout(kind, rate=None, coin=None):
                x = np.asarray(x0, np.float64).copy()
                n, Tn = len(x), sd.physics.length
                J = np.zeros(n)
                fired = np.zeros((n, Tn), bool)
                past = np.zeros((n, Tn), bool)
                rng = np.random.default_rng(coin) if coin is not None else None
                for t in range(Tn):
                    a = np.clip(actor(x), -1.0, 1.0)
                    f = np.clip(a, lo99, hi99)
                    if kind == "none":
                        fire = np.zeros(n, bool)
                    elif kind == "always":
                        fire = np.ones(n, bool)
                    elif kind == "random":
                        fire = rng.uniform(size=n) < rate
                    else:
                        o = norm(x)
                        Ua, Uf = width(kind, o, a), width(kind, o, f)
                        la, lf_ = qf(o, a) - Ua, qf(o, f) - Uf
                        fire = ~np.isfinite(Ua) | (np.isfinite(Uf) & (la < lf_))
                        fire &= np.any(a != f, 1)
                    pa = TC.past_knee(a, sd.physics)
                    a = np.where(fire[:, None], f, a)
                    fired[:, t], past[:, t] = fire, pa
                    x, r = TC.dynamics(x, a, sd.physics)
                    J = J + r
                pts = TR.points(J, an[sit])
                fp = fired & past
                return dict(points=float(pts.mean()), fire=float(fired.mean()), past80=float(past.mean()),
                            recall=float(fp.sum() / max(past.sum(), 1)), precision=float(fp.sum() / max(fired.sum(), 1)),
                            per_start=np.round(pts, 4).tolist())

            for kind in ("none", "always", "bca_uniform", "bca_weighted") + tuple(tables):
                out[kind] = rollout(kind)
            for kind in RANDOM_FOR:
                out[f"random@{kind}"] = rollout("random", out[kind]["fire"],
                                                [20261014, rep, 700, wi, si, sum(map(ord, kind))])
            res[world]["situations"][sit] = out
            print(rep, world, sit, " ".join(f"{k}:{v['points']:.1f}/{v['fire']:.2f}" for k, v in out.items()),
                  f"{time.time() - t0:.0f}s", flush=True)
        # check against the fallback diagnostic's saved numbers on the same 200 starts
        fb = json.load(open(f"/home/user/bca/runs/shift_explore/fallback/seed{rep}_{world}_{name}.json"))
        res[world]["check_vs_fallback"] = {
            s: dict(none=res[world]["situations"][s]["none"]["points"] - fb["situations"][s]["rules"]["none:None"]["points"],
                    always=res[world]["situations"][s]["always"]["points"]
                    - fb["situations"][s]["rules"]["always:clip99"]["points"])
            for s in SITS if s in fb["situations"]}
    return res


if __name__ == "__main__":
    rep = int(sys.argv[1])
    json.dump(run(rep), open(f"{OUT}/deploy_lcb_seed{rep}.json", "w"), indent=1, default=float)
