"""Probe: load one host with the package's loaders, time the sweep, look at magnitudes. Writes nothing."""
import time

import numpy as np

import experiments.shift  # noqa: F401  (before gymnasium)
from experiments.shift import learners as LR
from experiments.shift import truth as TR
from experiments.shift_iw import data as D
from experiments.shift_iw import run as RUN
from experiments.shift_iw import world as W
from pathlib import Path

ROOT = Path("/root/bca-shift/runs/shift_iw_v5")
t0 = time.perf_counter()
ctx = RUN.Context(root=ROOT, quality="expert", log=lambda m: None)
spec = RUN.spec_digest(ctx)
print("spec", spec)
data = RUN.load_data(ctx, 0, spec)
mean, std = D.normalization(data)
print("load data", time.perf_counter() - t0)
banks, brec = D.load_start_banks(ctx.banks_root)
for law in ("none", "mild", "strong"):
    c = banks[law][:10, 10]
    print(law, "c[0:10]", np.round(c, 3), "u", np.round(W.cdf(c, law), 3), "IQR", W.inverse_cdf(np.array([0.25, 0.75]), law))
rec = RUN.require_host(ctx, "gap", 0, spec)
state = RUN.load_host_state(ctx, "gap", 0, spec)
snap = rec["result"]["snapshots"][-1]
print("actor sha ok", LR.tree_sha256(state.actor.params) == snap["actor_sha256"],
      "critic sha ok", LR.tree_sha256(state.critic.params) == snap["critic_sha256"])
print("norm ok", np.array_equal(np.asarray(rec["result"]["data"]["obs_mean"], np.float32), mean),
      np.array_equal(np.asarray(rec["result"]["data"]["obs_std"], np.float32), std))
pol = TR.MLPPolicy(state.actor.params, mean, std)
for law in ("none", "strong"):
    x0 = banks[law][0:1]
    run = TR.rollout(pol, x0, W.PHYSICS, length=100, record=True)
    S, C, R = run.states[0, :100], run.commands[0], run.rewards[0]
    print(law, "J", run.J, "points", W.points(run.J, brec["anchors"][law]["anchor"]))
    print(" cmd a1 range", C[:, 0].min(), C[:, 0].max(), "past80 steps", int(np.sum(np.abs(C[:, 0]) > 0.8)))
    sweep = -np.arange(41) / 40.0
    sweep = sweep.astype(np.float32).astype(np.float64)
    A = np.repeat(C[:, None, :], 41, 1)
    A[:, :, 0] = sweep[None]
    Ss = np.repeat(S[:, None, :], 41, 1).reshape(-1, 13)
    obs = (np.asarray(Ss, np.float32) - mean) / std
    t1 = time.perf_counter()
    q = LR.critic_rows(state.critic.params, obs, A.reshape(-1, 4).astype(np.float32))[0].min(1).reshape(100, 41)
    t2 = time.perf_counter()
    qpi = TR.q_value_jax(pol, Ss, A.reshape(-1, 4), W.PHYSICS, horizon=600).reshape(100, 41)
    t3 = time.perf_counter()
    print(" critic s", round(t2 - t1, 2), "qpi s", round(t3 - t2, 2))
    print(" q level", q.mean(), "q spread per state median", np.median(q.max(1) - q.min(1)))
    print(" qpi level", qpi.mean(), "qpi spread per state median", np.median(qpi.max(1) - qpi.min(1)))
    i80 = 32
    print(" critic rise past 80 (q[-1]-q[-0.8]) mean", (q[:, 40] - q[:, i80]).mean(), "share >0", np.mean(q[:, 40] > q[:, i80]))
    print(" true rise past 80 mean", (qpi[:, 40] - qpi[:, i80]).mean(), "share >0", np.mean(qpi[:, 40] > qpi[:, i80]))
    print(" a-hat idx", np.bincount(q.argmax(1), minlength=41))
    print(" best true idx", np.bincount(qpi.argmax(1), minlength=41))
print("total", time.perf_counter() - t0)
