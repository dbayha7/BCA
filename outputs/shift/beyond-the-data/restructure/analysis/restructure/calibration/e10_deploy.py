"""READ-ONLY exact rollouts of the SAVED Stage 2 'none' actor (no caution) with deployment rules built from the
throttle-stratified band (e4). No training, no fitting. Loaders and the clip fallback are the consumer study's
(restructure/consumer/common.py: recorded range = 0.5% / 99.5% quantiles of the gapped training commands; first 200
starts of each situation's bank; TC.dynamics; TR.points with the world's anchors), as stage2/code/deploy_check.

Rules per step, a = clip(pi(s), -1, 1), a_clip = clip(a, lo, hi):
  never            a                                                      (plumbing vs saved values.json)
  tau_raw          Stage 2's check as run (weighted band, its tau)        (plumbing vs saved values.json)
  tailA_tau        U_A(s, a) > tau_A -> a_clip; U_A = R_A(t(a)) u_f max(eta_f(s, a), 1e-6), tau_A = the 95% level
                   of U_A on the locked T4 held-out recorded rows (e4 widths)  [Stage 2's trigger, new band]
  tailA_le         lower-edge choice between a and a_clip: a_clip if min_h Q(s, a_clip) - U_A(s, a_clip) >
                   min_h Q(s, a) - U_A(s, a)  (the paired band compared in value units; frozen gapped critic)
  tailB_le         the same with U_B = R_B(t(a)) u_f (one-sided score)
  random:<rule>    a_clip with probability = <rule>'s firing share (M5's control)
'Cannot certify' tail (R_tail = +inf; seed 3 no-stall): maximum caution, every command with t(a) > t_lo is refused
(a_clip). Usage: python e10_deploy.py <seed> [n_starts]"""
import json
import math
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/consumer")
import common as C  # noqa: E402  (read-only loaders of the consumer study)

TC, TR, L = C.TC, C.TR, C.L
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
DET = "/home/user/bca/runs/shift_explore/detect"
SITS = ("strong_current", "recorded")
RULES = ("never", "tau_raw", "tailA_tau", "random:tailA_tau", "tailA_le", "random:tailA_le", "tailB_le",
         "random:tailB_le")


def make_R(E, form):
    Rb, Rt = E[f"R_{form}"]["bulk"], E[f"R_{form}"]["tail"]
    t_lo, t_hi = E["t_lo"], E["t_hi"]
    refuse = not (t_hi is not None and np.isfinite(Rt))

    def R(t):
        if refuse:
            return np.where(t > t_lo, np.inf, Rb)
        f = np.clip((t - t_lo) / (t_hi - t_lo), 0, 1)
        return Rb + f * (Rt - Rb)
    return R, refuse


def run_rule(rule, actor, band, crit, UA, UB, tauA, physics, anchor, x0, lo, hi, rate=None, coin=None):
    x = np.asarray(x0, np.float64).copy()
    n, T = len(x), physics.length
    J = np.zeros(n)
    fired = np.zeros((n, T), bool)
    past_prop = np.zeros((n, T), bool)
    rng = np.random.default_rng(coin) if coin is not None else None
    for t in range(T):
        a_act = np.clip(actor(x), -1.0, 1.0)
        a_clip = np.clip(a_act, lo, hi)
        a = a_act.copy()
        on = band.norm(x)
        if rule == "never":
            fire = np.zeros(n, bool)
        elif rule == "tau_raw":
            fire = ~(band.width(on, a_act) <= band.tau)
        elif rule == "tailA_tau":
            fire = ~(UA(on, a_act) <= tauA)
        elif rule in ("tailA_le", "tailB_le"):
            U = UA if rule == "tailA_le" else UB
            ua, uc = U(on, a_act), U(on, a_clip)
            le_a = np.where(np.isfinite(ua), C.qmin(crit, on, a_act) - np.where(np.isfinite(ua), ua, 0), -np.inf)
            le_c = C.qmin(crit, on, a_clip) - uc
            fire = (le_c > le_a) & np.any(a_clip != a_act, 1)
        elif rule.startswith("random:"):
            fire = rng.uniform(size=n) < rate
        else:
            raise ValueError(rule)
        a[fire] = a_clip[fire]
        fired[:, t] = fire
        past_prop[:, t] = TC.past_knee(a_act, physics)
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    pts = TR.points(J, anchor)
    fp = fired & past_prop
    return dict(points=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)), fire_share=float(fired.mean()),
                past80_proposed=float(past_prop.mean()), recall_past80=float(fp.sum() / max(past_prop.sum(), 1)),
                flag_rate_not_past80=float((fired & ~past_prop).sum() / max((~past_prop).sum(), 1)), rate=rate,
                per_start_points=np.round(pts, 4).tolist())


def main(rep, n):
    t0 = time.perf_counter()
    e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{OUT}/e4_tailband.json"))}
    out = dict(rep=rep, n_starts=n, worlds={})
    for wi, world in enumerate(("stall", "no_stall")):
        E = e4[(rep, world)]
        sd, pg = C.seed_data(world, rep)
        physics = pg.physics
        band = C.Band(world, rep)                     # weighted band: plumbing rule tau_raw
        bu = C.Band(world, rep, version="uniform")    # eta_f and u_f of the uniform fit
        crit = C.critic(sd)
        lo, hi = C.recorded_range(pg)
        RA, refA = make_R(E, "A")
        RB, refB = make_R(E, "B")
        unit = E["unit"]
        assert abs(unit - bu.unit) < 1e-6

        def UA(on, a):
            t = np.abs(np.asarray(a, np.float64)).max(1)
            return RA(t) * unit * np.maximum(bu.eta(on, a), 1e-6)

        def UB(on, a):
            t = np.abs(np.asarray(a, np.float64)).max(1)
            return RB(t) * unit
        pz = np.load(f"{DET}/seed{rep}/{world}/pairs.npz")
        Wd = np.load(f"{OUT}/e4_widths_seed{rep}_{world}.npz")
        t4 = pz["set"] == "T4"
        tauA = float(np.quantile(Wd["tail_A"][t4], 0.95))
        acts, lock, vals = C.policy_actors(world, rep)
        pol = L.mlp_policy(acts["none"], pg, "none")
        an = C.anchors(world)
        saved = json.loads((C.R2 / "deploy" / world / f"seed{rep}" / "values.json").read_text())
        W = dict(tauA=tauA, refuse_A=refA, refuse_B=refB, results={})
        for si, s in enumerate(SITS):
            x0 = TC.bank_starts(s, TC.BANK_SIZE)[:n]
            r = {}
            for rule in RULES:
                kw = {}
                if rule.startswith("random:"):
                    kw = dict(rate=r[rule.split(":", 1)[1]]["fire_share"], coin=[20261014, rep, wi, si, 991])
                r[rule] = run_rule(rule, pol, band, crit, UA, UB, tauA, physics, an[s], x0, lo, hi, **kw)
            sv = saved["results"]["none"][s]
            r["plumbing"] = dict(never_diff=r["never"]["points"] - sv["none"]["points"],
                                 tau_raw_diff=r["tau_raw"]["points"] - sv["bca"]["points"])
            for k in ("tailA_tau", "tailA_le", "tailB_le"):
                r[f"{k}_minus_random"] = r[k]["points"] - r[f"random:{k}"]["points"]
                r[f"{k}_minus_never"] = r[k]["points"] - r["never"]["points"]
            W["results"][s] = r
            print(f"seed {rep} {world:8s} {s:14s} " + " ".join(
                f"{k}={v['points']:.1f}/{v['fire_share']:.2f}" for k, v in r.items() if isinstance(v, dict) and "points" in v)
                + f" | plumbing never {r['plumbing']['never_diff']:+.1e} tau {r['plumbing']['tau_raw_diff']:+.1e}"
                f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        out["worlds"][world] = W
    out["seconds"] = time.perf_counter() - t0
    with open(f"{OUT}/e10_deploy_seed{rep}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200)
