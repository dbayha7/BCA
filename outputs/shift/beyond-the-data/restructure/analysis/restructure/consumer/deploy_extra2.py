"""READ-ONLY data-only rivals of refuse_clamped with the SAME saved classifier, read on the same support (clamped
state), clip fallback, same starts and arithmetic (deploy_extra.rollout):

  plain_refuse_clamped   log w(s~, a) >= closed-form refusal limit of PLAIN weighted split conformal on the band's
                         threshold half (no BCA posterior: the ratio_refuse stand-in's rule)
  ratio95_clamped        log w(s~, a) > its own split-conformal 95% level on the 5,000 held-out recorded pairs (T4 of
                         the detection pair set): the classifier-ratio stand-in's rule
  each with random switching (clip) at its own firing share.
What is left for BCA in refuse_clamped is the difference to these rivals. Usage: python deploy_extra2.py <seed> [n]
"""
import math
import sys
import time

import numpy as np

import common as C
from deploy_extra import rollout

TC, TR, L = C.TC, C.TR, C.L
SITS = ("strong_current", "recorded", "big_waves", "big_kicks")


def main(rep, n):
    t0 = time.perf_counter()
    out = dict(rep=rep, n_starts=n, worlds={})
    for wi, world in enumerate(("stall", "no_stall")):
        sd, pg = C.seed_data(world, rep)
        bw = C.Band(world, rep)
        l_plain = C.plain_wcp_limit(bw, pg, rep)
        P = np.load(C.DET / f"seed{rep}" / world / "pairs.npz")
        t4 = P["set"] == "T4"
        lw4 = np.sort(bw.logw(bw.norm(P["states"][t4]), P["commands"][t4]))
        q95 = float(lw4[math.ceil((len(lw4) + 1) * 0.95) - 1])
        lo, hi = C.recorded_range(pg)
        blo, bhi = C.state_box(pg)
        acts, _, _ = C.policy_actors(world, rep)
        an = C.anchors(world)
        trigs = {
            "plain_refuse_clamped": lambda x, a: bw.logw(bw.norm(np.clip(x, blo, bhi)), a) >= l_plain,
            "ratio95_clamped": lambda x, a: bw.logw(bw.norm(np.clip(x, blo, bhi)), a) > q95,
        }
        W = dict(l_star=bw.l_star, l_plain=l_plain, ratio_q95=q95, actors={})
        for ai, name in enumerate(("none", "bca_k1")):
            pol = L.mlp_policy(acts[name], pg, name)
            W["actors"][name] = {}
            for si, s in enumerate(SITS):
                x0 = TC.bank_starts(s, TC.BANK_SIZE)[:n]
                r = {}
                for ti, (k, f) in enumerate(trigs.items()):
                    r[k] = rollout(pol, pg.physics, an[s], x0, lo, hi, f)
                    r["random:" + k] = rollout(pol, pg.physics, an[s], x0, lo, hi, None, rate=r[k]["fire_share"],
                                               coin=[20261014, rep, wi, ai, si, 999, ti])
                W["actors"][name][s] = r
                print(f"seed {rep} {world:8s} {name:7s} {s:14s} l*={bw.l_star:.2f} plain={l_plain:.2f} q95={q95:.2f} "
                      + " ".join(f"{k}={v['points']:.1f}/{v['fire_share']:.2f}" for k, v in r.items()) +
                      f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        out["worlds"][world] = W
    C.dump(out, f"deploy_extra2_seed{rep}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200)
