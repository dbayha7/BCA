# ReBRAC Walker2d: second declared training seed accepted

Seed202609172, native losses, Bayesian and conformal BCA, no importance weighting.
Saved host then BCA audits ran once each on CPU, followed by separate standard-library
export review. This artifact describes one training seed; Walker2d now has two of
five paired seeds accepted. The full five-seed inference remains pending.

| Normalized score | Host | BCA | BCA minus host |
|---|---:|---:|---:|
| Final 20-episode mean | 87.807234 | 84.819327 | -2.987906 |
| Mean of 200 periodic-bank means | 74.215885 | 77.346488 | +3.130603 |
| Final episode median | 89.188968 | 86.104875 | N/A |
| Final episode sample SD | 5.272649 | 6.535583 | N/A |

BCA is ahead in 3/20 paired final episodes, with
0 ties. Its final mean is lower, while its average
across periodic training evaluations is higher. All original episodes are retained.
The previously accepted first Walker seed had a positive final mean difference;
the two seeds therefore differ in direction. No training-seed interval, reliable
general benefit or OOD-action conclusion follows. Episode spread measures reset
variation, not uncertainty across independently trained seeds.

Each run completed 1M critic/host and 500k actor updates. Original worker exits:
host0 at 2026-09-28T11:58:55.638079+00:00, BCA0 at 2026-09-28T13:00:36.795703+00:00.
The audits checked all108 frozen source files, raw/cache/conversion and disjoint
training/holdout data, preparation identities, six total checkpoint file hashes,
finite native and embedded target states, all2M metric rows and exact phase/event
order. Independent review covered402 evaluation banks/4,040 raw-to-normalized
episode scores and2,000 metric blocks. ReBRAC actor rows retain zero-based even /
one-based odd scheduling, skipped placeholders excluded and active zeros retained.
There is no independently saved target-update counter.

The shared pool has300,549 training and1,149 held-out rows; BCA's seeded1,024-row
training reference passed exact membership/order checks. Native normalization is
disabled, with17 observation/6 action dimensions and actor/critic BC coefficients
0.05/0.01. Walker rewards remain raw; the registered x100 branch is AntMaze-only.
BCA has1M accepted scale fits, zero abstentions and198 posterior refreshes.
All existing training refresh keys, radius and data identities passed; no new OOD
key was generated. Final effective/Bayesian radius is
2.3200933933258057, conformal radius
2.129730224609375. Bayesian exceeds conformal
at 198/198 refreshes; frozen-reference
floor-dose changes at 0/198.
These are diagnostics, not causal evidence or OOD outcomes.

Calibrator mean/std arrays match fixed training-only preparation hashes; GPU
population-reduction arithmetic was not replayed. Posterior hashes were rebuilt
at3 saved checkpoints and linked to preceding refreshes. Per-scan posterior hashes
and every refresh's residual-unit contents/explicit row IDs were not saved and
are not invented. Full journal state hashes are retained without claiming a
reconstruction across serialization field order. Fresh ReBRAC targets requiring
unavailable recorded next actions remain unavailable; actor predictions cannot
replace them.

Host audit actual0 at 2026-09-28T22:53:27.322221+00:00, BCA actual0 at
2026-09-28T22:53:34.189075+00:00; independent review actual0 at
2026-09-28T22:53:50.622137+00:00.16 new synthetic guard tests passed,
no skips and no preimplementation red captured. Only new separately versioned
auditors ran; no closed audit or suite was rerun. CPU audit stderr retains optional
CUDA plugin discovery CUDA_ERROR_NO_DEVICE, with no GPU request or scientific retry.
Read-only path lookup mistakes are preserved separately, with no audit execution
in those failures. No learner/model/simulator, new v2 key, ledger or shared lock
was used.

Publication context: the first export stopped before package/repository changes,
actual1, because another chat's advisor-brief commit appended five lines to the
local .gitattributes file included in the frozen manifest. A separate stdlib
review verified the unchanged original prefix and exact advisor-only path rules.
The local mirror matches107/108 manifest files; both actual cluster audits checked
all108 original source files. A separately versioned exporter accepts only that
exact publication-metadata delta, preserving the original failure and every
scientific source pin. This does not authorize a runtime-source exception or
change the original manifest, protocol or numerical implementation.


The exact executed source bytes, derivation diffs, raw audit outputs/exits and
independent review are packaged in evidence.json.gz and indexed by package.json.
The host source retains inherited mixed line endings. Direct JSON exports are in
verified-v1; no weights, SQLite or NPZ are published. Ten physical runs/five
training pairs are accepted overall. Both ReBRAC Hopper and Walker2d have2/5
paired seeds accepted. All5 remain required in each cell.

The one-stream amendment is explicitly approved and belongs to the other chat's
separate implementation. Its new files are preserved. This audit does not accept
actual v2 stream derivation, runtime/storage/ownership, engineering or OOD science.
The inherited review flag v2_stream_hold_unchanged records unchanged execution
readiness; it does not retract or question the recorded amendment approval.
