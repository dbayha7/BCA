"""READ-ONLY evidence E7 (Angle D): the recorded-range limit across the evidence sweep. At each trace share, the
recorded range is recomputed from that share's kept training commands (0.5% and 99.5% quantiles, read from the trace
bed's signals.json readout.stand_ins.recorded_range); the check-2 gapped critic's actor of each world is clipped to it
at every step (the blanket limit, the deployment analogue of the range hinge). Same 200 starts as the fallback
diagnostic. Shows that the hinge's only evidence knob is a COUNT of kept commands, identical in both worlds.
No training, no fit, no BCA module. Writes only to this folder.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/bca/runs/shift_explore/trace/code")
import tracebed as T  # noqa: E402

L, TC, TR = T.L, T.TC, T.TR
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/testbed"
SITS = ("strong_current", "recorded")


def run(rep):
    t0 = time.time()
    g = T.GateSeed(rep)
    acts, _, _ = g.h2()
    res = {}
    for world in T.WORLDS:
        sd = g.sd[world]
        pg = g.gapped_prepared(world)
        name = "gapped" if world == "stall" else "no_stall_gapped"
        actor = L.mlp_policy(acts[name], pg, name)
        an = g.anchors(world)
        res[world] = {}
        for sh in T.SHARES:
            rr = json.load(open(T.job_dir(world, rep, sh) / "signals.json"))["readout"]["stand_ins"]["recorded_range"]
            lo, hi = np.asarray(rr["lo"]), np.asarray(rr["hi"])
            res[world][f"{sh:g}"] = dict(lo=lo.tolist(), hi=hi.tolist())
            for sit in SITS:
                x = np.asarray(TC.bank_starts(sit)[:200], np.float64).copy()
                J = np.zeros(len(x))
                for _ in range(sd.physics.length):
                    a = np.clip(np.clip(actor(x), -1.0, 1.0), lo, hi)
                    x, r = TC.dynamics(x, a, sd.physics)
                    J = J + r
                res[world][f"{sh:g}"][sit] = float(TR.points(J, an[sit]).mean())
            print(rep, world, sh, {k: round(v, 1) for k, v in res[world][f"{sh:g}"].items() if k in SITS},
                  f"{time.time() - t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    rep = int(sys.argv[1])
    json.dump(run(rep), open(f"{OUT}/hinge_sweep_seed{rep}.json", "w"), indent=1)
