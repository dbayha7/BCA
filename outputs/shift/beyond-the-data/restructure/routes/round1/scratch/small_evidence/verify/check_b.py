import sys
sys.dont_write_bytecode = True
import json, time
import numpy as np
sys.path.insert(0, "/home/user/bca"); sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/restructure/step1")
import step1lib as S
SC = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/small_evidence/scores.npz"
DR = "/home/user/bca/runs/shift_explore/routes/small_evidence/draws.npz"
z = np.load(SC); d = np.load(DR)
spec, _ = S.load_prespec(); cfg = spec["certificate"]
t0 = time.process_time()
rs = np.random.default_rng(123)
maxdiff = 0.0; flips = 0; cnt = 0; fresh_flips = 0
for w in ("stall", "no_stall"):
    wt = S.WORLD_TAG[w]
    for s in range(5):
        pool = np.asarray(z[f"{w}_{s}_pool"], float); rest = np.asarray(z[f"{w}_{s}_rest"], float)
        cr = S.posterior(rest, [20261015, 9931, s, wt, 2], cfg)
        for n in (8, 12, 30):
            idx_all = d[f"{w}_{s}_n{n}_idx"]; q05s = d[f"{w}_{s}_n{n}_q05"]
            for j in rs.choice(400, 4, replace=False):
                tail = pool[idx_all[j]]
                ct = S.posterior(tail, [20261015, 9931, s, wt, n, int(j) + 1, 1], cfg)
                q = S.pairwise_quantile(ct.crossings, cr.crossings, 0.05)
                maxdiff = max(maxdiff, abs(q - q05s[j]) if np.isfinite(q) else 0.0)
                flips += (q > 0.10) != (q05s[j] > 0.10); cnt += 1
                # the full step1lib.certificate with fresh tags (both posteriors new): does on/off change?
                c = S.certificate(tail, rest, [20261015, 7777, s, wt, n, int(j) + 1], cfg)
                fresh_flips += c["on"] != (q05s[j] > 0.10)
print("checked", cnt, "max |q05 diff|", maxdiff, "on/off flips", flips, "fresh-tag certificate flips", fresh_flips,
      "cpu", time.process_time() - t0)
