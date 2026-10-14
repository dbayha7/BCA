"""READ-ONLY. The 'hinge in disguise' frontier: state-independent caps on the SAVED actors, by exact rollouts.

Same starts and arithmetic as deploy.py. Rules:
  cap_c       clip every thruster to [-c, c], c in CAPS (a blanket cap; c = 0.8 is the knee)
  bca_box     per thruster and sign, clip to the MEDIAN of BCA's refusal boundary over the full-throttle-region
              training states (boundary.json, T1, band read at the raw training state): a constant box whose edges
              BCA's own refusal rule placed (the fallback variant of the design)
  plain_box   the same with the plain weighted-conformal refusal limit's boundary medians (attribution)
If a constant cap traces a frontier (stall gain vs no-stall cost) that the state-dependent projection (deploy.py,
project_clamped) does not beat, the projection's gain is a hinge in disguise. Usage: python capsweep.py <seed> [n]
"""
import json
import sys
import time

import numpy as np

import common as C

TC, TR, L = C.TC, C.TR, C.L
CAPS = (0.75, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.95)
SITS = ("strong_current", "recorded", "big_waves", "big_kicks")


def box_from_boundary(bnd, key):
    lo, hi = np.zeros(C.TC.DA), np.zeros(C.TC.DA)
    for j in range(C.TC.DA):
        lo[j] = -min(bnd[f"th{j + 1}-"][key]["q"]["50"], 1.0)
        hi[j] = min(bnd[f"th{j + 1}+"][key]["q"]["50"], 1.0)
    return lo, hi


def run(actor, physics, anchor, x0, lo, hi):
    x = np.asarray(x0, np.float64).copy()
    J = np.zeros(len(x))
    past = 0.0
    for _ in range(physics.length):
        a_act = np.clip(actor(x), -1.0, 1.0)
        a = np.clip(a_act, lo, hi)
        past += TC.past_knee(a, physics).mean()
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    return dict(points=float(TR.points(J, anchor).mean()), past80_applied=past / physics.length)


def main(rep, n):
    t0 = time.perf_counter()
    bnd_all = json.loads((C.OUT / "boundary.json").read_text())
    out = dict(rep=rep, n_starts=n, caps=list(CAPS), worlds={})
    for world in ("stall", "no_stall"):
        sd, pg = C.seed_data(world, rep)
        bnd = [r for r in bnd_all if r["rep"] == rep and r["world"] == world][0]["sets"]["T1"]["boundaries"]
        boxes = {f"cap_{c:g}": (-np.full(4, c), np.full(4, c)) for c in CAPS}
        boxes["bca_box"] = box_from_boundary(bnd, "refuse_raw")
        boxes["plain_box"] = box_from_boundary(bnd, "plainwcp_raw")
        acts, _, _ = C.policy_actors(world, rep)
        an = C.anchors(world)
        W = dict(boxes={k: [v[0].tolist(), v[1].tolist()] for k, v in boxes.items()}, actors={})
        for name in ("none", "bca_k1"):
            pol = L.mlp_policy(acts[name], pg, name)
            W["actors"][name] = {}
            for s in SITS:
                x0 = TC.bank_starts(s, TC.BANK_SIZE)[:n]
                W["actors"][name][s] = {k: run(pol, pg.physics, an[s], x0, lo, hi) for k, (lo, hi) in boxes.items()}
                print(f"seed {rep} {world:8s} {name:7s} {s:14s} " + " ".join(
                    f"{k}={v['points']:.1f}" for k, v in W["actors"][name][s].items()) +
                      f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        out["worlds"][world] = W
    C.dump(out, f"capsweep_seed{rep}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200)
