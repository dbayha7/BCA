"""Route 'runtime_monitor', phase 2 (per seed): the run-time monitor's looks after every episode.

EXPLORATORY, designed after seeing seeds 0-4. Seeds 0-4 only. Reads phase 1's arrays (scores of every executed step
of the unguarded 'never' rollout, the step's group, per-episode points without and with the clip). Before the latch
every executed step is a 'never' step, so the monitor's evidence at the look after episode e is exactly the scores
of episodes 1..e of the 'never' rollout; after the latch the episodes run with the clip ('on' points).
Rules: scratchpad/routes/runtime_monitor/KILL_RULE.md (fixed before any result).
Writes only runs/shift_explore/routes/runtime_monitor/seed<k>/looks.json.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/restructure/step1")
import step1lib as S1  # noqa: E402

OUT = Path("/home/user/bca/runs/shift_explore/routes/runtime_monitor")
WI = {"S": 1, "N": 2, "M3": 3, "M4": 4}
LOGW = {"S": "stall", "N": "no_stall", "M3": "no_stall", "M4": "stall"}
E = 200


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def compose(pn, po, latch):
    """Monitor points: episodes 1..latch without the clip, latch+1..200 with it (latch None: never on)."""
    m = pn.copy()
    if latch is not None:
        m[latch:] = po[latch:]
    return float(m.mean())


def run(seed, key, sit, z, cfg, full):
    wi = WI[key]
    sc, tl = z[f"{key}_{sit}_score"], z[f"{key}_{sit}_tail"]
    pn, po = z[f"{key}_{sit}_pts_never"], z[f"{key}_{sit}_pts_on"]
    bank_rest = z[f"bank_rest_{LOGW[key]}"]
    latch = dict(real=None, permuted=None, mann_whitney=None, offline_bank_rest=None, fleet=None)
    trace = dict(real=[], permuted=[], mann_whitney=[], offline_bank_rest=[], fleet=[])
    cache = {}
    t0 = time.process_time()
    for e in range(1, E + 1):
        tail = sc[:e][tl[:e]]
        rest = sc[:e][~tl[:e]]
        tags = [20261015, 9930, seed, wi, e]
        if latch["real"] is None or (e % 10 == 0 and latch["fleet"] is None):
            c = S1.certificate(tail, rest, tags, cfg)
            cache[e] = c
            if latch["real"] is None:
                trace["real"].append([e, len(tail), len(rest), c["q05"], bool(c["on"])])
                if c["on"]:
                    latch["real"] = e
            if e % 10 == 0 and latch["fleet"] is None:
                trace["fleet"].append([e, c["q05"], bool(c["on"])])
                if c["on"]:
                    latch["fleet"] = e
        if full and latch["permuted"] is None:
            pooled = np.concatenate([tail, rest])
            pm = pooled[S1.rng([20261015, 9940, seed, wi, e]).permutation(len(pooled))]
            c = S1.certificate(pm[:len(tail)], pm[len(tail):], tags, cfg)
            trace["permuted"].append([e, c["q05"], bool(c["on"])])
            if c["on"]:
                latch["permuted"] = e
        if latch["mann_whitney"] is None:
            r = S1.rank_test(tail, rest)
            trace["mann_whitney"].append([e, r["p"]])
            if r["p"] is not None and r["p"] < 0.05:
                latch["mann_whitney"] = e
        if full and latch["offline_bank_rest"] is None:
            c = S1.certificate(tail, bank_rest, [20261015, 9950, seed, wi, e], cfg)
            trace["offline_bank_rest"].append([e, c["q05"], bool(c["on"])])
            if c["on"]:
                latch["offline_bank_rest"] = e
        if all(v is not None for k, v in latch.items() if full or k in ("real", "mann_whitney", "fleet")):
            break
    # points: a look after episode 200 changes no episode
    pts = {k: compose(pn, po, v if (v is not None and v < E) else None) for k, v in latch.items()}
    pts["never"] = float(pn.mean())
    pts["clip_always"] = float(po.mean())
    first = cache.get(1)
    res = dict(world=key, situation=sit, world_index=wi, latch_after_episode=latch, points=pts,
               n_looks=dict(real=len(trace["real"]), permuted=len(trace["permuted"]),
                            offline_bank_rest=len(trace["offline_bank_rest"])),
               first_look=None if first is None else dict(q05=first["q05"], n_tail=first["n_tail"],
                                                          n_rest=first["n_rest"], on=first["on"]),
               max_q05_real=max([t[3] for t in trace["real"] if t[3] is not None], default=None),
               trace=trace, cpu_seconds=time.process_time() - t0)
    if latch["real"] is not None:
        c = cache[latch["real"]]
        res["at_switch"] = dict(q05=c["q05"], q50=c["q50"], height=c["height"], n_tail=c["n_tail"],
                                n_rest=c["n_rest"], mean_tail=c["mean_tail"], mean_rest=c["mean_rest"])
    log(f"seed {seed} {key}/{sit}: latch {latch}; points {pts}; max q05 {res['max_q05_real']}; "
        f"cpu {res['cpu_seconds']:.0f}s")
    return res


def main(seed):
    t0 = time.process_time()
    spec, _ = S1.load_prespec()
    cfg = spec["certificate"]
    d = OUT / f"seed{seed}"
    if (d / "looks.json").exists():
        raise SystemExit("refused: looks.json exists")
    z = dict(np.load(d / "arrays.npz"))
    out = dict(seed=seed, label="EXPLORATORY route test (runtime_monitor), designed after seeing seeds 0-4.",
               runs={})
    for key in ("S", "N", "M3", "M4"):
        out["runs"][f"{key}/strong_current"] = run(seed, key, "strong_current", z, cfg, True)
    out["runs"]["N/recorded"] = run(seed, "N", "recorded", z, cfg, False)
    out["cpu_seconds"] = time.process_time() - t0
    p = (d / "looks.json").resolve()
    assert S1.inside(p, OUT)
    p.write_text(json.dumps(S1.jsonable(out)))
    log(f"seed {seed} done, cpu {out['cpu_seconds']:.0f}s")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(int(s))
