# ReBRAC Walker2d: third declared training seed accepted

Seed 202609173 retains native losses, Bayesian and conformal BCA, and no importance
weighting. The fixed host then BCA saved-input CPU audits passed once each, followed
by independent standard-library arithmetic and export review. ReBRAC Walker2d and
Hopper now each have three of five paired training seeds accepted. All five remain
required for the planned inference.

| Normalized score | Host | BCA | BCA minus host |
|---|---:|---:|---:|
| Final 20-episode mean | 45.288932187 | 78.928872114 | +33.639939927 |
| Mean of 200 periodic-bank means | 73.372049506 | 75.045082164 | +1.673032658 |
| Final episode median | 27.228566999 | 87.059176245 | N/A |
| Final episode sample SD | 39.119928304 | 15.016595113 | N/A |

BCA is ahead in 13/20 paired final episodes, with
0 ties. Both the final mean and average across training
evaluations are higher in this seed. The final host scores vary widely across
episodes, reflected in the mean, median and sample SD above. These are recorded
performance observations, not a failed worker or a reason to replace a seed.

The first Walker2d final difference was +4.9805, the second -2.9879, and this third
one is +33.6399. All declared seeds and original episodes are retained. This is
a descriptive training comparison; it does not establish a reliable general
benefit or better OOD-action handling. No training-seed confidence interval is
claimed. Episode SD describes reset variation, not uncertainty across training
seeds. The direct OOD-action study has separate pending execution gates.

Each run completed 1,000,000 critic/host updates and 500,000 actor updates. Original
worker exits were host 0 at 2026-09-29T01:22:13.303368+00:00 and BCA 0 at
2026-09-29T02:24:02.649411+00:00. Both audits checked all 108 original cluster
source files against the unchanged frozen manifest, actual raw/cache/conversion
identities, disjoint training/holdout data and preparation bindings. All six saved
checkpoint file hashes and finite native/embedded-target states, two million
metric rows and exact phase/update/evaluation order passed. Independent review
checked 402 evaluation banks, 4,040 raw-to-normalized episode scores and 2,000
metric blocks. Actor rows retain ReBRAC's zero-based even / one-based odd schedule;
skipped placeholders are excluded and active zeros retained. Embedded targets are
finite; no independently saved target update counter is available.

The shared pool contains 300,549 training and 1,149 held-out rows. BCA's seeded
1,024-row training reference passed exact membership/order checks. Native
normalization is disabled, with 17 observation and 6 action dimensions and
actor/critic BC coefficients 0.05/0.01. Walker2d rewards remain raw; the registered
x100 branch applies only to AntMaze. BCA has one million accepted scale fits,
zero abstentions and 198 posterior refreshes. Existing training refresh keys,
radius and data identities passed; no new OOD key was generated.

Final effective/Bayesian radius is 2.3787527084350586;
conformal radius is 2.1421213150024414.
Bayesian exceeds conformal at 198/198
refreshes; the frozen-reference floor-dose diagnostic changes at
0/198. These are diagnostics,
not causal evidence or OOD outcomes.

Calibrator mean/std arrays match fixed training-only preparation hashes; GPU
population-reduction arithmetic was not replayed. Posterior hashes were rebuilt
at three saved checkpoints and linked to preceding refreshes. Per-scan posterior
hashes and every refresh's residual-unit contents/explicit row IDs were not saved
and are not invented. Journal state hashes are retained without reconstruction
across serialization field order. Fresh ReBRAC targets needing unavailable
recorded next actions remain unavailable; actor predictions cannot replace them.

Host audit actual 0 at 2026-09-29T02:25:44.435572+00:00, BCA actual 0 at
2026-09-29T02:25:51.039770+00:00; independent review actual 0 at
2026-09-29T02:26:03.952748+00:00. Sixteen new synthetic guard tests
passed with no skips; no preimplementation red was captured. New separately named
auditors change only the fixed seed, exact preparation/result pins and docstring
from the closed second-Walker auditors. Reversing those changes restores exact
parent bytes, including inherited mixed host line endings. No closed audit or
test suite was rerun. Optional CUDA plugin discovery CUDA_ERROR_NO_DEVICE remains
in BCA stderr; the audit selected CPU and requested no GPU. Both audit exits are
0, with no scientific retry.

The local mirror matches 107/108 training manifest files: only the previously
reviewed advisor .gitattributes append differs. Both cluster audits independently
checked all 108 original files. This is publication metadata context, not a
scientific source or runtime exception. All prior closed v2 components, failed
native probes, original plan, approved one-stream amendment, latest filelock
conditional source graph and prior accepted training artifacts remain unchanged.
The failed key producer remains actual 1 and its 40 saved tables stay held. No
native import probe, model query, simulator/physics, learner update, new OOD key,
scientific resource ledger, archive or shared lease ran.

The initial bounded snapshot at 2026-09-29T02:22:08.278036+00:00 had 61 closures and the
fixed BCA partner still active at 975k. No pair inspection or audit ran then. The
separate snapshot at 2026-09-29T02:24:54.700841+00:00 confirmed both declared runs closed
successfully before inspection and audit. It records job 27045057 on str-gpu13,
matching controller PID 763309 and current worker
PID 870231, with 62/140 closures: 32 ReBRAC and 30 IQL;
28 first-seed, 28 second-seed and six third-seed runs. All saved closure receipts
were successful. The current worker was iql-walker2d-host-s202609173; no
numeric update step was present in its bounded recent-event summary. Current
worker/controller actual exits remained pending. After accepting this pair, 50
cluster closures await independent training audits. Overall 14/280 physical runs
(14/315 actors), seven training pairs, are accepted. Queues and holds are unchanged.

Exact executed sources, derivation diffs, raw outputs, actual-exit receipts and
independent review are packaged in evidence.json.gz and indexed by package.json.
Direct JSON exports are in verified-v1. No weights, SQLite, NPZ, native binaries
or key arrays are published. This accepts saved training evidence only. The real
native import bridge, live stdlib/source-alias and hook-state bindings, explicit
15-field worker-owned scratch contract, storage/ownership/physical engineering
and pair-wide OOD precommit/science gates remain pending.
