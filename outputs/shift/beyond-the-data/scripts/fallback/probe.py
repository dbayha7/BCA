"""EXPLORATORY probe (no training, no BCA): load one gate seed, look at the recorded command ranges and time kNN."""
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import experiments.shift.acceptance as A
import experiments.shift.learners as L
import experiments.shift.thruster_chain as TC

rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
cfg = prereg.config
outcome = A.gate_outcome(prereg)
print("outcome", {k: v for k, v in outcome.items() if k in ("variant", "alpha", "verdict")})
variant = outcome["variant"]
st = A.load_seed(prereg.runs, cfg, variant, "stall", rep)
pg = st.prepared("gap", cfg.learn)
log = st.log
kept = st.removals["gap"].kept_train_rows(log)
a = np.asarray(log["actions"][kept], np.float64)
print("kept rows", len(kept), "bank rows", len(st.removals["gap"].bank_rows(log)))
print("min", a.min(0).round(3), "max", a.max(0).round(3))
for q in (0.001, 0.005, 0.01):
    print("q", q, np.quantile(a, q, axis=0).round(3), np.quantile(a, 1 - q, axis=0).round(3))
print("past 80 share", TC.past_knee(a).mean(), "count", TC.past_knee(a).sum())
print("per thruster past 80", (np.abs(a) > TC.KNEE).sum(0))
x = log["infos/state"][kept]
print("state mean", x.mean(0).round(3), "sd", x.std(0).round(3))
mu, sd = x.mean(0), x.std(0)
t0 = time.perf_counter()
tree = cKDTree((x - mu) / sd)
print("tree build", time.perf_counter() - t0)
bank = st.removals["gap"].bank_rows(log)
xb = log["infos/state"][bank]
t0 = time.perf_counter()
d = tree.query((xb[:1000] - mu) / sd, k=10)[0][:, -1]
print("state knn 1000 queries", time.perf_counter() - t0)
xa = np.concatenate([x, a], 1)
mu2, sd2 = xa.mean(0), xa.std(0)
tree2 = cKDTree((xa - mu2) / sd2)
qa = np.concatenate([xb, np.asarray(log["actions"][bank], np.float64)], 1)
t0 = time.perf_counter()
d2 = tree2.query((qa[:1000] - mu2) / sd2, k=10)[0][:, -1]
print("state-action knn 1000 queries", time.perf_counter() - t0)
print("bank state dist quantiles", np.quantile(d, [0.5, 0.9, 0.95, 0.99]).round(3))
print("bank sa dist quantiles", np.quantile(d2, [0.5, 0.9, 0.95, 0.99]).round(3))
print("obs norm mean", pg.obs_mean.round(3), "std", pg.obs_std.round(3))
