"""EXPLORATORY feasibility count, seed 0 of the cloud rerun. Reads NO outcome (no reward, no next state, no critic
value), fits no band, trains nothing, runs no arm. It counts, per data quality:
  (a) held-out tries a gate could read: rows of the training-law log's calibration split (episodes 500-699) whose
      command lies outside the seed's ball, by thruster k, current band (thirds of [0.1, 1.1]) and throttle bin of |a_k|
      (0.5-0.6, 0.6-0.7, 0.7-0.8 incl. the knee, 0.8-0.9, 0.9-1.0); and the Kish effective size of each cell under the
      exact strong-mix weight w(c) (world.ratio);
  (b) where the plain TD3+BC host's own commands fall on its strong-start rollouts (the rollout the host's evaluation
      already ran; only the commands are binned).
Run from <iw_rerun>/repo with PYTHONPATH=<iw_rerun>/repo:<iw_rerun>/explore:<iw_rerun>/driver."""
import json
import sys
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401
from experiments.shift import learners as LR
from experiments.shift import truth as TR
from experiments.shift_iw import data as D
from experiments.shift_iw import run as R
from experiments.shift_iw import world as W
import jobs as J

OUT = Path(__file__).resolve().parent / "evidence_counts.json"
KNEE = float(np.float32(0.8))
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, KNEE), (KNEE, 0.9), (0.9, 1.0 + 1e-9)]
BANDS = [(0.1, 0.1 + 1 / 3), (0.1 + 1 / 3, 0.1 + 2 / 3), (0.1 + 2 / 3, 1.1 + 1e-9)]


def bin_of(x):
    """0: [0.5, 0.6), 1: [0.6, 0.7), 2: [0.7, knee], 3: (knee, 0.9], 4: (0.9, 1.0]; -1 below 0.5."""
    out = np.full(x.shape, -1)
    out[(x >= 0.5) & (x < 0.6)] = 0
    out[(x >= 0.6) & (x < 0.7)] = 1
    out[(x >= 0.7) & (x <= KNEE)] = 2
    out[(x > KNEE) & (x <= 0.9)] = 3
    out[x > 0.9] = 4
    return out


def band_of(c):
    out = np.full(c.shape, -1)
    for j, (lo, hi) in enumerate(BANDS):
        out[(c >= lo) & (c < hi)] = j
    return out


ctx0 = J.ctx_of("expert")
spec = R.spec_digest(ctx0)
banks, _ = D.load_start_banks(D.banks_dir(J.V5))
res = {}
for q in J.QUALITIES:
    ctx = J.ctx_of(q)
    data = R.load_data(ctx, J.SEED, spec)
    log = data.train_log
    gap = data.removal("gap").gap
    rows = log.split_rows("calibration")
    a = np.asarray(log["actions"][rows], np.float32).astype(np.float64)
    keep = ~gap.contains(a)
    a, c = a[keep], np.asarray(log.current[rows])[keep]
    w = W.ratio(c, "strong")
    band = band_of(c)
    cells = {}
    for k in range(4):
        b = bin_of(np.abs(a[:, k]))
        for j in range(3):
            for i in range(len(BINS)):
                m = (b == i) & (band == j)
                n = int(m.sum())
                kish = float(w[m].sum() ** 2 / (w[m] ** 2).sum()) if n else 0.0
                cells[f"t{k + 1}|band{j}|bin{i}"] = dict(n=n, kish_strong=round(kish, 1))
    # (b) the host's own commands on its strong-start rollout
    prep = D.prepare(data, "gap", ctx.cfg)
    host = R.load_host_state(ctx, "gap", J.SEED, spec)
    pol = LR.mlp_policy(host.actor.params, prep)
    run = TR.rollout(pol, banks["strong"], W.PHYSICS, record=True)
    cmd = run.commands.reshape(-1, 4)
    cs = run.states[:, :-1, 10].reshape(-1)
    hb = band_of(cs)
    share = {}
    for k in range(4):
        b = bin_of(np.abs(cmd[:, k]))
        for j in range(3):
            for i in range(len(BINS)):
                share[f"t{k + 1}|band{j}|bin{i}"] = round(float(((b == i) & (hb == j)).mean()), 4)
    share["band_share"] = [round(float((hb == j).mean()), 3) for j in range(3)]
    res[q] = dict(calibration_rows_outside_ball=int(keep.sum()), cells=cells, host_strong_command_share=share)
    t1 = {kk: v for kk, v in cells.items() if kk.startswith("t1|")}
    print(q, "rows", int(keep.sum()), "t1 n/kish", {kk[3:]: (v["n"], v["kish_strong"]) for kk, v in t1.items()},
          flush=True)
    print("   host strong share t1", {kk[3:]: v for kk, v in share.items() if kk.startswith("t1|")},
          "bands", share["band_share"], flush=True)
OUT.write_text(json.dumps(dict(label="EXPLORATORY seed 0; counts only: no outcome, no band, no training", res=res),
                          indent=1))
