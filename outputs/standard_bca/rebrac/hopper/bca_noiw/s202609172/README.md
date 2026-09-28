# ReBRAC Hopper: second declared training seed accepted

Seed202609172, native losses, Bayesian and conformal BCA, no importance weighting.
Both saved runs passed independent CPU audits once in fixed host-then-BCA order,
followed by a separate standard-library export review. This artifact describes
one seed. Alongside the closed first-seed pair, two of five declared ReBRAC Hopper
training pairs are now accepted; the full five-seed comparison remains pending.

| Normalized score | Host | BCA | BCA minus host |
|---|---:|---:|---:|
| Final 20-episode mean | 101.956213 | 102.038035 | +0.081822 |
| Mean of 200 periodic-bank means | 90.968232 | 89.197225 | -1.771007 |
| Final episode median | 101.973315 | 102.045266 | N/A |
| Final episode sample SD | 0.271292 | 0.240404 | N/A |

BCA is ahead in 10/20 paired final episodes, with
0 ties. Final performance is nearly tied; its average
score across periodic training evaluations is lower. All original episodes are
retained. These descriptive results do not establish a reliable BCA benefit or
better handling of OOD actions. Episode spread is reset variation, not uncertainty
across training seeds. No training-seed interval or pooled-episode inference is supplied.

Each run completed1M critic/host and500k actor updates. Actual worker exits were0:
host 2026-09-28T09:43:19.496147+00:00, BCA 2026-09-28T10:42:43.610179+00:00.
All108 frozen source files, raw cache and converted data, disjoint training/
holdout partitions, preparation and checkpoint identities, finite native and
embedded target states, all2M metric rows, exact phase/event order, six total
checkpoints,402 evaluation banks and4,040 raw-to-normalized episode scores passed.
Actor metrics use zero-based even/one-based odd updates; skipped placeholders are
excluded and active zeros retained. Embedded targets are finite; there is no
independently saved target-update counter.

The shared pool has998,895 training and1,103 held-out rows. BCA's seeded1,024-row
training reference was checked for exact membership/order. Native normalization
is disabled;11 observation and3 action dimensions, raw Hopper rewards and
actor/critic BC coefficients0.01/0.01 are retained. The registered ReBRAC x100
reward branch applies only to AntMaze.

BCA has1M accepted scale fits, zero abstentions and198 posterior refreshes.
Every existing training refresh key, radius, reference/holdout identity and
support/ESS1,103 passed; no new v2 key was derived. Final effective/Bayesian radius
is 1.258787751197815, conformal radius
1.1105952262878418. Bayesian is larger at
198/198 refreshes; the frozen-reference
floor-dose diagnostic changes at 0/198.
These are diagnostics, not causal or OOD evidence; conformal remains present.

Evidence limits: checkpoint calibrator mean/std arrays match preparation hashes
and frozen training-only source; GPU population-reduction arithmetic was not
replayed. Full posterior hashes were reconstructed at three saved BCA checkpoints
and linked to preceding refreshes. Per-scan posterior hashes and every refresh's
residual-unit contents/explicit row IDs were not saved and are not invented.
Journal full-state hashes are retained without claiming reconstruction across
serialization field order. Fresh ReBRAC targets needing unavailable recorded next
actions remain unavailable; actor predictions do not replace them.

The host audit exited0 at 2026-09-28T22:32:16.936446+00:00, BCA at
2026-09-28T22:32:26.633911+00:00; independent review exited0 at
2026-09-28T22:32:46.489624+00:00.14 new synthetic guard tests passed,
no skips; no preimplementation red captured. No closed audit/test was rerun.
No learner/model/simulator call, GPU request, new OOD stream, ledger or lock was
added. BCA stderr preserves optional CUDA plugin discovery CUDA_ERROR_NO_DEVICE;
the CPU audit and existing-training-key checks completed0. Initial nonexistent
status/helper path lookups were read-only; no audit ran in those lookups.

Exact executed source bytes, derivative diffs, inspection, dispatch, actual exits,
host binding and independent review are in evidence.json.gz, indexed by
package.json. The host source retains inherited mixed line endings; no normalized
copy is represented as executed. Direct JSON exports are in verified-v1. No
weights, SQLite or NPZ are published. Eight physical training runs/four pairs are
accepted across the study. OOD execution/global readiness remain false, and the
one-stream amendment is still pending explicit direction.
