# Standard BCA: execution status

## September 29, 12:35 UTC: first TD3+BC Walker training pair accepted

Seed202609171 host/BCA passed saved-data audits and independent arithmetic review,
all actualexit0. The paired training inventory now has8 accepted pairs and16
successful runs; the recovered CQL Hopper host with original exit1 remains separate.
Final normalized means are77.246103 host and82.896940 BCA (+5.650837); periodic
curve means are73.795470 host and71.579274 BCA (-2.216197). BCA wins8/20 paired
final resets. One paired training seed supports a descriptive contrast, not a
replicated benefit or training-seed uncertainty estimate.

The raw data, dependency split, training-only normalization,108 execution-source
pins, three checkpoints per method,402 evaluation banks/4040 episodes and198 BCA
refreshes passed. No learner/model/simulator was run by the review. OOD checkpoint
queries and native simulator gates remain pending for this new pair. All five
existing OOD executions continue; no v2 OOD comparison is complete.

The recovery training queue has5 completed/1 active/131 unstarted rows at12:34.
CQL Walker BCA171 completed actualexit0 at12:18:51; its independent training review
is pending. TD3 HalfCheetah host171 is the current worker. The local queue stays
delegated, the original ReBRAC/IQL queue is unchanged, and the two-GPU cap holds.

[Independent training review](validation/standard-td3-walker-first-pair.json),
[training and OOD evidence readout](../outputs/ood/robustness-v2/prefix-td3-walker-acceptance-v1/README.md).

## September 29, 09:11 UTC: CQL+BCA training is running

Recoveryjob27068529 passed all14 data-cell checks (7TD3 reused from pinned
acceptances, 7CQL newly checked) and all4 GPU fixtures with actualexit0. The
live CQLHopperBCA worker's command, process start ticks and Slurm allocation
were independently matched. Its journal console has completed the original
5k/10k/15k/20k ten-episode evaluation banks; training is at least20k/1M.
The queue has1active,136notstarted and0closed recovery rows. Both original
completedTD3runs and recoveredCQLhost are excluded. ExistingReBRAC/IQL and
CPUOODjobs continue; no new OOD pair is claimed by this training progress.

[Actual checks and live-worker evidence](validation/standard-cql-cluster-recovery-execution.json).

## September 29, 09:05 UTC: separate CQL data-wrapper correction dispatched

The initial recovery clusterjob27068516 stopped before any learner/model/fixture
execution. Its data wrapper called a TD3/ReBRAC-only helper on CQL after passing
all seven TD3 cells. All original bytes and actual exit1 remain preserved.
Job27068529 now owns the unchanged137-row manifest in a separatev2 root. It
reuses the seven pinned TD3 acceptances and applies CQL's own exact preparation
contracts and original CQL fingerprint to the remaining seven cells. All
scientific sources, settings and numerical gates remain unchanged.

[Execution correction and actual-attempt records](../experiments/standard_bca/cql_validation_recovery/v2/README.md).
The saved CQL host acceptance below remains valid; this was a new preflight
wrapper failure, not another failed training run. Training starts only after
the v2 data/GPU gates pass.

## September 29, 08:56 UTC: CQL result recovered; untouched queue delegated

The CQL validation bug is fixed. Its episode fields are `score`/`return`, while
the old common validator expected `normalized_score`/`raw_return`. The first
CQL host had already finished 1M updates and all evaluations when it failed.
The saved-only audit and independent review now accept its result without
retraining: final normalized mean62.9825183, 201banks/2020episodes and all three
checkpoints verified. Original worker/controller exit1 remains; this is one
recovered host result, not an additional OOD pair or evidence of BCA benefit.
Nine execution/regression tests passed; no scientific gate changed.

Job27068516 owns the frozen137 untouched CQL/TD3 rows on a second A100, starting
with CQLHopperBCA seed202609171 after mandatory data/GPU gates. Existing
ReBRAC/IQLjob27045057 remains on the other allocated A100. The local CPU OOD
worker and its shared lock remain undisturbed. A durable local delegation claim
prevents duplicate execution. Submission is not training completion; follow
the new job's actual gate/controller/worker receipts.

The first recovered CQL host ran on RTX5070Ti; its future BCA counterpart runs
on A100. Preserve this mixed-hardware qualification in comparisons. Four later
CQL seeds will use A100 for both methods; no existing seed is replaced.

[Recovery readout](../outputs/standard_bca/cql/hopper/host/s202609171/recovery-v1/README.md),
[independent review](validation/standard-cql-recovery-independent-review.json),
[continuation declaration](../experiments/standard_bca/cql_validation_recovery/README.md).

## September 29, 02:29 UTC: third ReBRAC Walker2d training pair accepted

Seed202609173 host/BCA saved training evidence passed fixed-order CPU audits and
independent arithmetic/export review, all actual0. Fourteen of280 physical runs
(14/315 actors), seven training pairs, are accepted. Both ReBRAC Hopper and
Walker2d now have3/5 paired training seeds; all5 remain required.

Final normalized host 45.2889322, BCA 78.9288721,
difference +33.6399399; periodic-curve host 73.3720495,
BCA 75.0450822, difference +1.6730327.
BCA wins 13/20 final paired episodes, zero ties. The host
final scores vary widely (episode SD 39.1199, BCA
15.0166). Prior Walker final differences were
+4.9805 and -2.9879; this third contrast is +33.6399. All episodes and declared
seeds remain included. No training-seed interval, reliable general benefit or
OOD-action conclusion is claimed.

The02:24:54UTC bounded cluster snapshot confirms job27045057/str-gpu13,
controller763309 and worker870231 command/start/group identities.62/140 closures
(32ReBRAC/30IQL;28 first,28 second,6 third seed) all have worker0/learner completed.
After this acceptance,50 cluster closures await independent audits. The current
worker is iql-walker2d-host-s202609173, with no numeric step in the bounded
recent-event summary; current worker/controller actual exits remain pending.
Queues and holds unchanged. The initial02:22:08UTC snapshot with61 closures and
BCA975k is retained. Both fixed runs closed before inspection and audit.

[Pair readout](../outputs/standard_bca/rebrac/walker2d/bca_noiw/s202609173/README.md),
[independent validation](validation/standard-third-rebrac-walker-pair.json).

## September 29, 01:23 UTC: third ReBRAC Hopper training pair accepted

Seed202609173 host/BCA saved training evidence passed separate CPU audits in fixed
order and independent export/arithmetic review, all actual0. Twelve of280 physical
runs (12/315 actors), six training pairs, are accepted. ReBRAC Hopper has3/5 and
Walker2d2/5 paired training seeds accepted; all5 remain required.

Final normalized host 102.6074103, BCA 102.3503024,
difference -0.2571079; periodic-curve host 88.8350741,
BCA 92.8614912, difference +4.0264171.
BCA wins 5/20 final paired episodes, zero ties. Its slightly
lower final mean coexists with higher average across training evaluations. This
third final difference is negative after two small positive Hopper differences.
All episodes/seeds are retained; no training-seed interval or reliable general
benefit/OOD conclusion is claimed.

Bounded cluster snapshot01:18:05UTC: job27045057 on str-gpu13; controller763309 and
worker863913 command/start/group identities match.60/140 closures (30ReBRAC/30IQL;
28 first-seed,28 second-seed,4 third-seed), all saved worker0/learner completed.
After this acceptance,50 cluster closures await audits. The current third-seed
Walker host was925k; actual worker/controller exits remained pending. Queues and
holds unchanged. No model/physics/learner/new OOD-key execution was added.

[Pair readout](../outputs/standard_bca/rebrac/hopper/bca_noiw/s202609173/README.md),
[independent validation](validation/standard-third-rebrac-hopper-pair.json).

## September 28, 22:56 UTC: second ReBRAC Walker2d pair accepted

Host and BCA seed202609172 saved evidence passed separate CPU audits once each in
fixed order and independent export/arithmetic review, all actual0. Ten of280
physical runs (10/315 actors), five training pairs, are now accepted. ReBRAC
Hopper and Walker2d each have two of five paired seeds accepted; all5 are required.

Final normalized host 87.8072335, BCA 84.8193274,
difference -2.9879062; periodic-curve host 74.2158849,
BCA 77.3464882, difference +3.1306033.
BCA wins 3/20 final paired episodes, zero ties.
Lower final mean coexists with higher average across training evaluations.
Unlike the first Walker seed, the final difference is negative. Neither seed
alone nor these two descriptive contrasts establishes reliable general benefit
or better OOD handling.

Bounded cluster observation22:51:51UTC: job27045057 str-gpu13, controller/worker
identities match,56 closures (28ReBRAC/28IQL;28 each first/second seed), all saved
worker and learner completion receipts successful. With this pair accepted,
48 cluster closures await full audits. Current worker was770k, exits pending.
Queues unchanged. No new model/physics/learner/OOD key execution.

[Pair readout](../outputs/standard_bca/rebrac/walker2d/bca_noiw/s202609172/README.md),
[independent validation](validation/standard-second-rebrac-walker-pair.json).

## September 28, 22:42 UTC: stream approval observed during publication review

David explicitly approved the one-seed correction in the active chat
"Analyze BCA algorithm performance (3)". That chat is implementing the amendment.
This supersedes the pending-approval statements in earlier status entries and in
the second Hopper pair's saved audit package. The training audit evidence and
results remain unchanged. This monitor has not accepted the other chat's new
amendment validation, and runtime and scientific execution gates remain pending.
See [the authorization context record](validation/standard-second-rebrac-hopper-pair-context-update.json).

## September28,22:33UTC: second ReBRAC Hopper seed independently accepted

Saved host and BCA evidence for seed202609172 passed separate CPU audits, each
once in fixed order, and independent export/arithmetic review, all actual0.
Eight of280 physical training runs (8/315 actors) and four pairs are now accepted.
ReBRAC Hopper has two of five paired seeds accepted; full five-seed inference
remains pending. Existing first-seed Hopper/Walker results are unchanged.

Final normalized host101.9562131/BCA102.0380352,
delta+0.0818220; periodic-curve host90.9682323/BCA89.1972255,
delta-1.7710068. BCA wins10/20 final paired episodes,
zero ties. Final performance is nearly tied and the average across training
evaluations is lower. This is descriptive training evidence, not an OOD benefit.

Latest bounded cluster probe22:28:55UTC: job27045057 str-gpu13, original
controller/current-worker identity matched,56/140 successful closures
(28ReBRAC/28IQL),28 closures each for the first two seeds. After acceptance,
50 cluster closures still await independent audits. Current exits remain pending;
queues are unchanged. No new model/physics/training/stream execution occurred.

[Pair evidence and limits](../outputs/standard_bca/rebrac/hopper/bca_noiw/s202609172/README.md),
[independent validation](validation/standard-second-rebrac-hopper-pair.json).


## September 28,21:40 UTC: first ReBRAC Walker2d pair accepted

Independent saved-input audits of seed202609171 host and BCA each exited0, in
fixed order, followed by an independent arithmetic/export review0. Accepted
physical runs now total6/280 (6/315 actors), forming three first-seed pairs:
TD3+BC Hopper, ReBRAC Hopper and ReBRAC Walker2d. All five paired seeds remain required.

Walker2d final normalized means: host77.3307787, BCA82.3112812, delta+4.9805026.
Periodic-curve means: host74.5169790, BCA75.4619361, delta+0.9449571. BCA wins7/20
paired final episodes, with zero ties. Its higher mean coexists with a lower
median; no episode was excluded. This single-seed result supplies neither a
reliable general benefit nor evidence of improved OOD-action handling.

All saved data/source/counters/evaluation/metric/checkpoint evidence passed under
the documented limits. No learner/model/simulator call or new OOD stream was run.
Latest live probe21:32:54UTC: job27045057 str-gpu13, controller/current-worker
identity matched,53/140 successful cluster closures (27ReBRAC/26IQL). After this
pair acceptance49 cluster closures still await independent audits. Queues unchanged.

[Pair readout and limits](../outputs/standard_bca/rebrac/walker2d/bca_noiw/s202609171/README.md),
[independent validation](validation/standard-first-rebrac-walker-pair.json).


## September 28,20:40 UTC: first ReBRAC Hopper pair independently accepted

The saved ReBRAC Hopper BCA run at seed202609171 passed its CPU audit and a
separate export review, both actualexit0. Accepted physical runs now total4/280:
the TD3+BC Hopper and ReBRAC Hopper host/BCA pairs, one seed each. All five paired
training seeds remain required for each cell.

Final normalized mean: host102.0239558, BCA102.1122394, delta+0.0882836.
Periodic-curve mean: host92.5215207, BCA93.0327780, delta+0.5112573.
BCA wins10/20 paired final episodes. This small single-seed difference does not
establish a reliable benefit or improved OOD handling. No training-seed interval.

All1M updates/metric rows,500k actor updates,1M accepted scale fits,198 refreshes,
three checkpoints, declared evaluation banks and exact source/data identities
were checked. The closed host audit was reused. No model query, simulator step,
learner update, GPU request or training retry was added. OOD gates remain separate.

Latest live snapshot20:28:20UTC: job27045057 on str-gpu13, matching original
controller/current worker identities,50/140 cluster closures (26 ReBRAC/24 IQL),
all worker/learner success receipts. With the newly accepted pair,48 cluster
closures still need independent audits. Active queues and local CQL hold unchanged.

- [Accepted pair and evidence limits](../outputs/standard_bca/rebrac/hopper/bca_noiw/s202609171/README.md)
- [Independent validation](validation/standard-first-rebrac-pair.json)


## September 28, 19:02 UTC: VPN restored; cluster training continued

Live compute-node inspection confirms the original controller and worker still
match their saved command, start time and process group. The cluster has **45/140
training runs with successful worker and learner exit receipts: 23 ReBRAC and
22 IQL**. All seven datasets have host and BCA run closures for the first seed;
17 further closures are from the second seed. The active run is ReBRAC Pen-human
BCA, seed 202609172, at 950,000 updates in this snapshot. No restart, duplicate
submission, new model query or simulator call was needed for this check.

These are execution completions, **not 45 independently accepted scientific
results**. Accepted training results remain three physical runs: the TD3+BC Hopper
host/BCA pair and the ReBRAC Hopper host. The other 44 cluster closures need
independent source/data, counter, checkpoint and evaluation-bank audits. The next
cluster pair to verify is ReBRAC Hopper BCA against its already accepted host.
The local CQL queue remains stopped on its preserved validation error.

The first TD3+BC Hopper OOD action comparison is separately complete and reported;
all other OOD comparisons and the full five-seed study remain incomplete.

- [Live identity and all 45 actual-exit receipts](validation/standard-cluster-vpn-return.json)
- [Current OOD readout status](OOD_ACTION_COLLECTION_STATUS.md)

## Earlier training observations (historical timestamps)

The September 27 study compares each plain host with BCA **without an importance
tilt**, retaining Bayesian bootstrap masses and both radius components. Each
run has 1M host updates. The frozen matrix contains 280 runs / 315 actor
trajectories across four hosts, seven datasets and five seeds.

| Execution lane | Hosts | Status |
|---|---|---|
| Local RTX 5070 Ti | TD3+BC, CQL | **Stopped at 21:56 UTC.** CQL Hopper host reached final evaluation then failed schema validation; worker/controller actual exits 1. First TD3+BC pair remains verified. No retry. |
| Cluster A100 | ReBRAC, IQL | Job 27045057 continues; ReBRAC HalfCheetah BCA at 485k with matching live identities on September 28 at 01:44 UTC. Eight closure receipts await independent audits; accepted results remain unchanged. |

Three of 280 declared physical runs (three of 315 actor trajectories) are independently verified complete. The first TD3+BC Hopper host final mean is **65.2157311**; its periodic-curve mean is **57.1475157**. See the [complete host readout and plots](../outputs/standard_bca/td3_bc/hopper/host/s202609171/README.md) and [audit receipts](validation/standard-first-td3-host.json). Its first BCA pair is now verified; four further paired training seeds remain pending for this cell.

The first ReBRAC Hopper host final mean is **102.0239558**; its periodic-curve mean is **92.5215207**. See its [host readout and plots](../outputs/standard_bca/rebrac/hopper/host/s202609171/README.md) and [audit receipts](validation/standard-first-rebrac-host.json). Its paired BCA comparison and four further host seeds remain pending. The ReBRAC pair remains pending in this readout.

The first TD3+BC Hopper BCA final mean is **65.9275509** (host **65.2157311**, delta **+0.7118198**); its curve mean is **55.6709296** (host **57.1475157**, delta **-1.4765860**). BCA wins 11/20 paired final episodes. See the [first paired readout and plots](../outputs/standard_bca/td3_bc/hopper/bca_noiw/s202609171/README.md) and [validation receipts](validation/standard-first-td3-pair.json). One paired training seed does not supply five-seed uncertainty or establish robust benefit, causality or OOD ranking. Initial unit-warmup numerical differences remain unresolved; no replay.

A checkpoint below 1M is progress, not completion. All original outcomes remain included.

## Accepted preparation

- 280 typed configurations and 47 exact integration-map snippets verified.
- Eight host/BCA CPU numerical comparisons match preserved implementations
  exactly, including BCA radius refreshes and decoded checkpoint counters.
- All 28 host/dataset preparations match the saved data identities; paired
  host/BCA data agree. IQL Pen reservation feasibility passes.
- Four no-IW known-answer tests, three harmless launcher lifecycle tests,
  and 39 OOD protocol/arithmetic/figure regression tests pass.
- IQL host and shared-Q/V BCA execution/checkpoint fixtures pass eight mock
  updates each, with zero simulator outcomes.
- Local TD3+BC/CQL GPU fixtures pass. All four plain host files and all 40
  bundled Unifloral files remain unchanged.

## Preserved cluster preflight attempts

1. Job 27044796: missing `GL/osmesa.h`; exit 1 before learner/simulator work.
2. Job 27044823: headers fixed; installed `patchelf` absent from PATH; exit 1
   before learner/simulator work.
3. Job 27044828: older JAX plugin appeared through inherited packages; typed
   validation exited by signal 11 (Slurm wrapper 245). The exact crash cause
   is unresolved. No learner/simulator phase was reached.

4. Job 27045031: **all five phases passed**, actual wrapper exit 0: typed
   protocols, no-IW/lifecycle tests, seven explicit simulator test steps, and
   ReBRAC/IQL GPU fixtures. This supplies engineering validation, not OOD results.

The initial isolated runtime inherited the old environment's packages. A second,
fully isolated runtime now passes with the repository's pinned versions and
MJRL commit `3871d93763d3b49c4741e6daeaebbc605fe140dc`. Gym, D4RL and MJRL
Python source files are byte-identical across the local and cluster environments.
The original runtime and all failed attempts remain intact. Graphics installation
finished and its files were verified, but its first shell wrapper had a quoting
error and exited 1; its original JSON must not be read as proof of installer exit.
These are environment/setup attempts, not scientific repetitions.

## Reproduction and next gates

Scientific source commit: `6e912b0ec3e3346ba6c628b6eef6fddedc74ffc0`.
All 108 frozen source files match that commit. Frozen manifest SHA256:
`13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe`.
See [manifest and runner](../experiments/standard_bca/README.md) and
[validation evidence](validation/standard-bca-preflight.json).

Cluster runtime, simulator identity and GPU checks passed. Its mandatory
14-cell data gate passed in job 27045057 before the cluster training controller started. Both methods matched the accepted local fingerprints in every cell. Live command/start/group identities were checked on the compute node; a login-node PID lookup is insufficient. See [data-gate and live-process evidence](validation/standard-bca-live-gate.json). Real OOD collection follows
verified 1M checkpoints and adapter checks. TD3+BC/ReBRAC Hopper/Walker adapters
have prior engineering evidence; IQL/CQL and remaining environments still need
their own gates. ReBRAC's missing recorded-next-action coverage target remains
unavailable unless a valid target contract is established.

Importance weighting is a later study. Its recipe will be chosen using separate,
predeclared development evidence rather than these final test outcomes. Old 1M
and 100k results retain their original identities; the halted old queue and two
old behavioral recoveries remain untouched. At most two GPUs total are used.

The compressed scientific manifest and implementation are published. See the
[cluster attempt receipts](validation/standard-bca-cluster-preflight.json) for
the original failures, separately documented corrections and actual exits.

The frozen `configs/sources.json` records its creation-time preflight state.
The completed 28-cell acceptance is in the newer validation receipt linked above;
the frozen snapshot is deliberately preserved.

## OOD process-receipt preparation

`experiments/ood/standard_receipt.py` binds the standard runner's separate actual
worker exit to its dispatch, declared row and result. It rejects failed, interrupted,
stale and mismatched metadata, and refuses to overwrite a receipt. Its current
scope is TD3+BC/ReBRAC Hopper/Walker. This bridge performs no model query or
simulator step and leaves scientific acceptance pending: checkpoints, journals,
source/data contents, paired preparation and simulator parity still need independent
verification. See [synthetic receipt tests](validation/standard-ood-receipts.json).

The original login-node process observation and direct compute-node SSH timeout
are retained in the monitoring snapshot. Process identity was subsequently verified
with a read-only Slurm step in the existing allocation, requesting no GPU. Neither
inspection attempt changed scientific execution.

## Local CQL validation failure: September 27, 21:56 UTC

CQL's episode records use `score` and `return`; the shared `verify_events` validator
requires `normalized_score` and `raw_return`. After the final 1M evaluation, this
raised `KeyError: normalized_score`. Separate worker/controller actual exits are 1,
with no timeout or interruption. The journal and 10k/50k/1M checkpoint files remain;
`result.json` and learner `exit.json` were never written. This is not an accepted
completion, and checkpoint counters have not been independently decoded here.

[Actual receipts and source diagnosis](validation/standard-cql-host-validation-failure.json)
retain the original failure. No frozen file was changed, controller restarted or
scientific retry launched. The cluster lane continues unchanged. A separately
checked correction/handling is needed before local recovery; do not silently
relabel or replace this attempt. OOD action preparation remains the priority.

## September 28 OOD interruption

The separate CPU action-harm worker was interrupted across a Windows restart, with 143/512 panels in its last durable progress and missing actual-exit receipts. Partial outcomes are not accepted results. No retry or training-queue change was made. See [the OOD interruption receipt](validation/ood-outcome-interruption.json).
