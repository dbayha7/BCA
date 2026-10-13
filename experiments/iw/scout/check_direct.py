"""Direct-sampling check of the curve integrals: many episodes drawn straight from a tilt, streamed."""
import json, math, sys
import numpy as np
sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/iw/scout")
import scout as S
from experiments.truth import gap as GAP
from experiments.shift import thruster_chain as TC

name, lam, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
r = json.load(open(f"{S.OUT}/{name}.json"))
centre, rho = np.array(r["training_logs"][0]["centre"]), r["training_logs"][0]["rho"]
rec = S.SPECS[name]()
rng = np.random.default_rng([S.SEED, 99, int(1000 * lam) % 997])
c = S.inv_cdf(rng.uniform(size=n), lam)
x = S.draw_starts(n, rng, c)
acc = {k: np.zeros(n) for k in ("in_gap", "exact80", "sa_limit", "region")}
for t in range(S.L):
    cmd, region, special = rec.parts(x, rng.standard_normal((n, 4)), rng.uniform(size=n))
    a = cmd.astype(np.float32).astype(np.float64)
    acc["in_gap"] += GAP.in_gap(a, centre, rho)
    acc["exact80"] += S.exact80(a.astype(np.float32))
    acc["sa_limit"] += np.abs(S.SA(x)[:, 0]) >= 0.8
    acc["region"] += region
    x, _ = TC.dynamics(x, a)
print(name, "lam", lam, "episodes", n, "mean c", round(c.mean(), 4), "expected", round(float(S.simpson_weights(np.array(r["grid"])) @ (np.array(r["grid"]) * S.pdf(np.array(r["grid"]), lam))), 4))
for k, v in acc.items():
    v = v / S.L
    print(f"  {k}: {v.mean():.4f} +- {v.std(ddof=1) / math.sqrt(n):.4f}")
