"""Made-up arrays only (no ThrusterChain data): cost of one A-BCA two-posterior decision and one tail gate.
wbcp.py copied by `git show origin/main:calibration/wbcp.py` (unchanged)."""
import sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import wbcp
rs = np.random.default_rng(1)
rest = rs.standard_normal(850); grp = rs.standard_normal(50) + 0.3
def decision(seed):
    a = wbcp.calibrate(grp, np.random.default_rng([seed, 1]))
    b = wbcp.calibrate(rest, np.random.default_rng([seed, 2]))
    d = a.crossings[:, None] - b.crossings[None, :]
    return np.quantile(d.ravel(), 0.05)
decision(0)
t = time.perf_counter(); [decision(s) for s in range(1, 41)]; dt = (time.perf_counter() - t) / 40
t2 = time.perf_counter(); [wbcp.calibrate(rs.standard_normal(100), np.random.default_rng([s, 3])) for s in range(200)]; dt2 = (time.perf_counter() - t2) / 200
print(f"A-BCA decision (850 rest, 50 group, 1000 draws, 1e6 pairwise): {dt*1000:.1f} ms; tail gate n=100: {dt2*1000:.2f} ms")
