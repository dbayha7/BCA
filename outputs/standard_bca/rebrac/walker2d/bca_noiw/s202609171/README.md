# ReBRAC Walker2d: first accepted standard host/BCA pair

Seed 202609171, native losses, Bayesian and conformal BCA, no importance weighting.
Both saved runs passed independent CPU audits in fixed host-then-BCA order, each
once, followed by a separate standard-library export review. These are recorded
training results, with one of five declared paired training seeds accepted.

| Normalized score | Host | BCA | BCA minus host |
|---|---:|---:|---:|
| Final 20-episode mean | 77.330779 | 82.311281 | +4.980503 |
| Mean of 200 periodic-bank means | 74.516979 | 75.461936 | +0.944957 |
| Final episode median | 86.797668 | 82.850190 | N/A |
| Final episode sample SD | 24.507212 | 5.488982 | N/A |

BCA is ahead in 7/20 paired final episodes, with
0 ties. Its higher mean coexists with a lower median
and fewer episode wins. All original episodes are retained. This single-seed
contrast does not establish a reliable BCA benefit or better handling of OOD
actions. Episode spread is reset variation, not training-seed uncertainty.
There is no five-seed or independent-episode confidence interval.

Each run completed 1M critic/host and 500k actor updates. Worker actual exits were
0: host 2026-09-27T22:43:17.148690+00:00, BCA 2026-09-27T23:44:25.590340+00:00.
All 108 frozen source files, raw cache and converted data, disjoint training/
holdout partitions, preparation and checkpoint identities, finite native and
embedded target states, all 2M metric rows, exact phase/event order, six total
checkpoints, 402 evaluation banks and 4,040 raw-to-normalized episode scores were
checked. Actor metrics use zero-based even/one-based odd updates; skipped
placeholders are excluded and active zeros retained. Target parameters are
saved within native states; there is no independently saved target-update counter.

The shared pool has 300,549 training and 1,149 held-out rows; BCA also uses a
seeded 1,024-row training reference. Native state normalization is disabled;
Walker observations and rewards remain raw. The registered ReBRAC x100 reward
branch applies to AntMaze only. Actor/critic BC coefficients are 0.05/0.01.

BCA has 1M accepted scale fits, zero abstentions and 198 posterior refreshes.
Every recorded refresh key, radius, reference/holdout identity and support/ESS
1,149 passed. Only existing training keys were recomputed on CPU. Final effective/
Bayesian radius is 2.4717495441436768, conformal radius
2.083059787750244. Bayesian is larger at all
198 refreshes; the recorded frozen-reference floor-dose diagnostic changes at
0/198 refreshes. These are diagnostics, not causal or OOD evidence. The conformal
component remains present.

Evidence limits: checkpoint calibrator mean/std arrays match preparation hashes
and frozen training-only preparation source; GPU population-reduction arithmetic
was not replayed. Full posterior hashes were reconstructed at three BCA saved
checkpoints and matched preceding refreshes. ReBRAC does not archive per-scan
posterior hashes or every refresh's residual-unit contents/explicit row IDs;
none were invented. Full journal state hashes were retained without claiming
reconstruction across serialization field orders. Fresh ReBRAC targets requiring
unavailable recorded next actions remain unavailable.

The host audit exited 0 at 21:37:57.773074 UTC and BCA at 21:38:04.973236 UTC on
September 28; independent export review exited 0 at 21:39:51.289525 UTC. Fourteen
new synthetic guard tests passed without skips. No preimplementation red failure
was captured. No closed audit/test was rerun, and no model query, simulator call,
learner update, GPU request or new v2 stream/ledger/lock/worker was added. BCA stderr
retains optional CUDA-plugin discovery CUDA_ERROR_NO_DEVICE while CPU key checks
and the audit completed 0 with JAX_PLATFORMS=cpu and CUDA_VISIBLE_DEVICES empty.
Two read-only path-inspection shell failures are preserved; neither ran an audit.

The exact tested/executed source bytes, derivative diffs, inspection, tests,
dispatch, actual exits, host binding and independent review are in evidence.json.gz,
indexed by package.json. In particular the host derivative retains mixed line
endings inherited from its parent and is packaged as exact bytes; no normalized
copy is represented as the executed source. Direct JSON exports are in verified-v1.
No weights, SQLite or NPZ are published. OOD execution and global readiness remain
false; the one-stream amendment remains pending explicit direction.
