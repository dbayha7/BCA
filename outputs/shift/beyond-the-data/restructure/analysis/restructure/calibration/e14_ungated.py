"""READ-ONLY rollouts. Attribution control for e12: the same deployment rule with the gate FORCED ON in the no-stall
world (a data-only 'refuse past the bulk's throttle edge t_lo' check, no BCA residual), saved 'none' actor, strong
current and recorded. The difference to e12's gated rule (gate off there, cost 0) is what BCA's residual certificate
is worth in the world where the missing commands help. In the stalling world the gate is on in every seed, so the
ungated rule equals e12's gated rule there."""
import json
import math
import sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/consumer")
import common as C  # noqa: E402
TC, TR, L = C.TC, C.TR, C.L
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"


def run(actor, physics, anchor, x0, lo, hi, t_lo, on):
    x = np.asarray(x0, np.float64).copy()
    J = np.zeros(len(x))
    fired = []
    for t in range(physics.length):
        a_act = np.clip(actor(x), -1.0, 1.0)
        fire = (np.abs(a_act).max(1) > t_lo) if on else np.zeros(len(x), bool)
        a = np.where(fire[:, None], np.clip(a_act, lo, hi), a_act)
        fired.append(fire.mean())
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    pts = TR.points(J, anchor)
    return float(pts.mean()), float(np.mean(fired))


e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{OUT}/e4_tailband.json"))}
res = {}
for rep in range(5):
    sd, pg = C.seed_data("no_stall", rep)
    lo, hi = C.recorded_range(pg)
    acts, _, _ = C.policy_actors("no_stall", rep)
    pol = L.mlp_policy(acts["none"], pg, "none")
    an = C.anchors("no_stall")
    for s in ("strong_current", "recorded"):
        x0 = TC.bank_starts(s, TC.BANK_SIZE)[:200]
        never = run(pol, pg.physics, an[s], x0, lo, hi, e4[(rep, "no_stall")]["t_lo"], False)
        forced = run(pol, pg.physics, an[s], x0, lo, hi, e4[(rep, "no_stall")]["t_lo"], True)
        res[f"{rep}_{s}"] = dict(never=never[0], ungated=forced[0], ungated_fire=forced[1], cost=never[0] - forced[0])
        print(rep, s, res[f"{rep}_{s}"], flush=True)
for s in ("strong_current", "recorded"):
    print(s, "mean ungated cost", round(float(np.mean([res[f"{r}_{s}"]["cost"] for r in range(5)])), 1),
          [round(res[f"{r}_{s}"]["cost"], 1) for r in range(5)])
json.dump(res, open(f"{OUT}/e14_ungated.json", "w"), indent=1)
