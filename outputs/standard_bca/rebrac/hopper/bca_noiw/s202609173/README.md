# ReBRAC Hopper: third declared training seed accepted

Seed 202609173 retains native losses, Bayesian and conformal BCA, and no importance
weighting. Separate host then BCA CPU audits passed once each, followed by an
independent standard-library review of the saved exports. ReBRAC Hopper now has
three of five paired training seeds accepted. All five remain required for the
planned inference.

| Normalized score | Host | BCA | BCA minus host |
|---|---:|---:|---:|
| Final 20-episode mean | 102.607410288 | 102.350302422 | -0.257107866 |
| Mean of 200 periodic-bank means | 88.835074065 | 92.861491198 | +4.026417132 |
| Final episode median | 102.643041861 | 102.290114237 | N/A |
| Final episode sample SD | 0.362520128 | 0.260030116 | N/A |

BCA is ahead in 5/20 paired final episodes, with
0 ties. Its final mean is slightly lower, while its average
across training evaluations is higher. The first two Hopper seeds had small
positive final differences; this third seed has a negative final difference.
Every original episode and all three declared seeds are retained. This is a
descriptive training comparison: it does not establish a reliable general
benefit, a training-seed confidence interval, or better OOD-action handling.
Episode SD describes reset variation, not uncertainty across trained seeds.

Each run completed 1,000,000 critic/host updates and 500,000 actor updates. Original
worker exits were host 0 at 2026-09-28T23:04:30.972295+00:00 and BCA 0 at
2026-09-29T00:05:21.189472+00:00. Each audit verified all 108 original cluster
source files against the unchanged frozen manifest, actual raw/cache/conversion
identities, disjoint training/holdout data and preparation bindings. Across the
pair, all six saved checkpoint file hashes and finite native/embedded-target
states, two million metric rows and exact phase/update/evaluation order passed.
The independent review checked 402 evaluation banks, 4,040 raw-to-normalized
episode scores and 2,000 metric blocks. Actor rows retain ReBRAC's zero-based even
/ one-based odd schedule; skipped placeholders are excluded and active zeros
retained. Embedded targets are finite; there is no independently saved target
update counter.

The shared pool contains 998,895 training and 1,103 held-out rows. BCA's seeded
1,024-row training reference passed exact membership/order checks. Native
normalization is disabled, with 11 observation and 3 action dimensions and
actor/critic BC coefficients 0.01/0.01. Hopper rewards remain raw; the registered
x100 branch applies only to AntMaze. BCA has one million accepted scale fits,
zero abstentions and 198 posterior refreshes. Existing training refresh keys,
radius and data identities passed; no new OOD key was generated.

Final effective/Bayesian radius is 1.0559669733047485;
conformal radius is 0.9846489429473877.
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

Host audit actual 0 at 2026-09-29T01:20:13.624597+00:00, BCA actual 0 at
2026-09-29T01:20:22.785463+00:00; independent review actual 0 at
2026-09-29T01:20:50.126072+00:00. Sixteen new synthetic guard tests
passed with no skips; no preimplementation red was captured. The separately named
auditors change only the fixed seed, exact preparation/result pins and docstring
from the closed second-Hopper auditors. Reversing those declared changes restores
the exact parent bytes, including inherited mixed host line endings. No closed
audit or test suite was rerun. Optional CUDA plugin discovery CUDA_ERROR_NO_DEVICE
remains in BCA stderr; the audit selected CPU and requested no GPU. Both audit
actual exits are 0, with no scientific retry.

The local mirror still matches 107/108 training manifest files: only the previously
reviewed advisor .gitattributes append differs. Both cluster audits independently
checked all 108 original files. This is publication metadata context, not a
scientific source or runtime exception. All prior closed v2 components, failed
native probes, original scientific plan, approved one-stream amendment and prior
accepted training artifacts remain unchanged. The failed key producer remains
actual 1 and its 40 saved tables stay held. No native import, model query,
simulator/physics, learner update, new v2 key, resource ledger or shared lease ran.

The bounded cluster snapshot at 2026-09-29T01:18:05.364407+00:00 records job 27045057 on
str-gpu13, matching original controller/current worker identities, and 60/140
closures: 30 ReBRAC and 30 IQL, comprising 28 first-seed, 28 second-seed and four
third-seed runs. All saved worker and learner closure receipts were successful.
The current worker was rebrac-walker2d-host-s202609173 at 925k updates;
controller/worker actual exits were still pending. After this pair's acceptance,
50 cluster closures await independent training audits. Twelve of 280 physical
runs (12/315 actors), six training pairs, are accepted overall. ReBRAC Hopper has
3/5 and Walker2d 2/5 paired seeds accepted. Queues and existing holds are unchanged.

Exact executed sources, derivation diffs, raw outputs, actual-exit receipts and
independent review are packaged in evidence.json.gz and indexed by package.json.
Direct JSON exports are in verified-v1. No weights, SQLite, NPZ, native binaries
or key arrays are published. This accepts saved training evidence only. The real
native import bridge, storage, ownership, physical engineering and independent
pair-wide OOD precommit/science gates remain pending.
