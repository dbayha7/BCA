"""READ-ONLY attribution rollouts for deploy.py (same starts, arithmetic, clip fallback). Separates the two changes in
refuse_clamped (read on the training-state box; trigger = BCA's refusal instead of U~ > tau):

  tau_clamped              Stage 2's trigger U~(s~, a) > tau, band read at the clamped state s~
  refuse_raw               log w(s, a) >= l*, band read at the raw state
  uniform_tau_clamped      the UNIFORM band (BCA as written, no weights): U(s~, a) > its own tau
  each with random switching (clip) at its own firing share.
Usage: python deploy_extra.py <seed> [n]
"""
import json
import math
import sys
import time

import numpy as np

import common as C

TC, TR, L = C.TC, C.TR, C.L
SITS = ("strong_current", "recorded", "big_waves", "big_kicks")


def rollout(actor, physics, anchor, x0, lo, hi, trig, rate=None, coin=None):
    x = np.asarray(x0, np.float64).copy()
    n, T = len(x), physics.length
    J = np.zeros(n)
    fired = np.zeros((n, T), bool)
    pp = np.zeros((n, T), bool)
    rng = np.random.default_rng(coin) if coin is not None else None
    for t in range(T):
        a_act = np.clip(actor(x), -1.0, 1.0)
        fire = trig(x, a_act) if trig is not None else rng.uniform(size=n) < rate
        a = a_act.copy()
        a[fire] = np.clip(a_act, lo, hi)[fire]
        fired[:, t] = fire
        pp[:, t] = TC.past_knee(a_act, physics)
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    pts = TR.points(J, anchor)
    fp = fired & pp
    return dict(points=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)), fire_share=float(fired.mean()),
                recall_past80=float(fp.sum() / max(pp.sum(), 1)),
                flag_rate_not_past80=float((fired & ~pp).sum() / max((~pp).sum(), 1)))


def main(rep, n):
    t0 = time.perf_counter()
    out = dict(rep=rep, n_starts=n, worlds={})
    for wi, world in enumerate(("stall", "no_stall")):
        sd, pg = C.seed_data(world, rep)
        bw = C.Band(world, rep)
        bu = C.Band(world, rep, "uniform")
        tau_u = float(json.loads((C.R2 / "detect" / "uniform" / world / "fixed" / f"seed{rep}" / "tau.json")
                                 .read_text())["tau"]) if (C.R2 / "detect" / "uniform" / world / "fixed" /
                                                           f"seed{rep}" / "tau.json").exists() else None
        lo, hi = C.recorded_range(pg)
        blo, bhi = C.state_box(pg)
        acts, _, _ = C.policy_actors(world, rep)
        an = C.anchors(world)
        trigs = {
            "tau_clamped": lambda x, a: ~(bw.width(bw.norm(np.clip(x, blo, bhi)), a) <= bw.tau),
            "refuse_raw": lambda x, a: bw.logw(bw.norm(x), a) >= bw.l_star,
        }
        if tau_u is not None:
            trigs["uniform_tau_clamped"] = lambda x, a: ~(bu.width(bu.norm(np.clip(x, blo, bhi)), a) <= tau_u)
        W = dict(tau_uniform=tau_u, actors={})
        for ai, name in enumerate(("none", "bca_k1")):
            pol = L.mlp_policy(acts[name], pg, name)
            W["actors"][name] = {}
            for si, s in enumerate(SITS):
                x0 = TC.bank_starts(s, TC.BANK_SIZE)[:n]
                r = {}
                for ti, (k, f) in enumerate(trigs.items()):
                    r[k] = rollout(pol, pg.physics, an[s], x0, lo, hi, f)
                    r["random:" + k] = rollout(pol, pg.physics, an[s], x0, lo, hi, None, rate=r[k]["fire_share"],
                                               coin=[20261014, rep, wi, ai, si, 888, ti])
                W["actors"][name][s] = r
                print(f"seed {rep} {world:8s} {name:7s} {s:14s} " + " ".join(
                    f"{k}={v['points']:.1f}/{v['fire_share']:.2f}" for k, v in r.items()) +
                      f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        out["worlds"][world] = W
    C.dump(out, f"deploy_extra_seed{rep}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200)
