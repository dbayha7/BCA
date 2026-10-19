"""EXPLORATORY diagnosis, seed 0 of the cloud rerun (no training, no band): roll the three seed-0 actors (plain TD3+BC,
BCA's capped pull unweighted, BCA IW) from the main test's 1,000 fixed starts in each of the four situations (as
sit4/situations.py does) and split each return into its two costs: drift (sum of |s|^2, the floats' distance from rest)
and energy (0.3 sum of |a|^2). Also: command sizes per thruster, the share of commands past 80%, and, on the plain
TD3+BC actor's own visited states, how the BCA actors' commands compare (scalar shrink k = <pi_b, pi_h> / <pi_h, pi_h>).
Writes diag_rollouts.json beside this file. Run from <iw_rerun>/repo with
PYTHONPATH=<iw_rerun>/repo:<iw_rerun>/explore:<iw_rerun>/driver."""
import json, math, time
from pathlib import Path
import numpy as np

import experiments.shift  # noqa: F401
from experiments.shift import thruster_chain as TC
from experiments.shift import truth as TR
from experiments.shift_iw import data as D
from experiments.shift_iw import run as R
from experiments.shift_iw import world as W
import jobs as J

OUT = Path(__file__).resolve().parent / "diag_rollouts.json"
SITS = ("recorded", "big_kicks", "big_waves", "strong_current")
t0 = time.time()
banks = {s: TC.bank_starts(s) for s in SITS}
anchors = D.anchors(banks)
spec = R.spec_digest(J.ctx_of("expert"))
res = {}
for q in J.QUALITIES:
    ctx = J.ctx_of(q)
    data = R.load_data(ctx, J.SEED, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    pols = {n: TR.MLPPolicy(a, prep.obs_mean, prep.obs_std, name=n) for n, (a, c, f) in J.learners_of(ctx, spec).items()}
    for s in SITS:
        x0 = banks[s]
        runs = {n: TR.rollout(p, x0, W.PHYSICS, record=True) for n, p in pols.items()}
        hs = runs["host_gap"].states[:, :100].reshape(-1, TC.DX)
        ph = np.clip(pols["host_gap"](hs), -1, 1)
        for n, run in runs.items():
            S, C = run.states[:, :100], run.commands
            drift = float(np.sum(S[..., :TC.DS] ** 2, axis=(1, 2)).mean())
            energy = float(0.3 * np.sum(C ** 2, axis=(1, 2)).mean())
            body = dict(J=float(run.J.mean()), points=float(W.points(run.J.mean(), anchors[s]["anchor"])),
                        drift=drift, energy=energy,
                        abs_cmd=np.abs(C).mean(axis=(0, 1)).tolist(),
                        past80_t1=float(np.mean(np.abs(C[..., 0]) > TC.KNEE)),
                        past80_any=float(np.mean(np.any(np.abs(C) > TC.KNEE, axis=-1))),
                        float_size=float(np.sqrt(np.sum(S[..., :TC.DS] ** 2, -1)).mean()))
            if n != "host_gap":
                pb = np.clip(pols[n](hs), -1, 1)
                body["on_host_states"] = dict(k=float(np.sum(pb * ph) / np.sum(ph * ph)),
                                              cos=float(np.sum(pb * ph) / math.sqrt(np.sum(pb * pb) * np.sum(ph * ph))),
                                              abs_t1_host=float(np.abs(ph[:, 0]).mean()), abs_t1_bca=float(np.abs(pb[:, 0]).mean()),
                                              norm_host=float(np.linalg.norm(ph, axis=1).mean()), norm_bca=float(np.linalg.norm(pb, axis=1).mean()))
            res[f"{q}|{n}|{s}"] = body
        h, b = res[f"{q}|host_gap|{s}"], res[f"{q}|bca_unweighted|{s}"]
        print(f"{q:14s} {s:15s} pts h {h['points']:.1f} b {b['points']:.1f} | dJ {b['J']-h['J']:+.1f} = drift {-(b['drift']-h['drift']):+.1f} + energy {-(b['energy']-h['energy']):+.1f}"
              f" | |a1| h {h['abs_cmd'][0]:.3f} b {b['abs_cmd'][0]:.3f} | past80 h {h['past80_any']:.3f} b {b['past80_any']:.3f} | k {b['on_host_states']['k']:.3f}"
              f" | {time.time()-t0:.0f}s", flush=True)
OUT.write_text(json.dumps(dict(kind="capped-pull-diagnosis-rollouts", label="EXPLORATORY: seed 0 of the cloud rerun; no training, no band", res=res), indent=1))
print("written", OUT)
