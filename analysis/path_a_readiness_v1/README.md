# Path A: current readiness and fixed scope

This is the current unpinned readiness document. It supersedes status and launch instructions in the historical sealed Path A README, amendment, recovery and Titan handoff documents; it does not replace their scientific protocol, approvals, receipts or source hashes. The original studies and failed attempts remain intact.

**Decision: the frozen Path A implementation and its narrow scientific questions are supported by saved setup/runtime evidence and focused CPU checks. Production recovery is not ready to dispatch until the existing Titan operational prerequisites below are satisfied. No new scientific gate, dose change or collection is needed for this already approved comparison.** A completed technical check is not evidence that BCA improves policies.

## What is already established

| Component | Verified evidence | Boundary of the claim |
|---|---|---|
| Fixed task | Existing v2 ThrusterChain plant, 13-state snapshots plus remaining time in observations, horizon 100, discount 0.98. Reward is `-1 - 0.01 ||a||^2`, +100 for goal entry, -100 for danger entry. Danger crossings take priority; goal, danger and finite horizon are terminal. | Regions are in benchmark phase space, not metres or an empirical chaotic obstacle. Physics/reward are identical across collection and evaluation. |
| Meaningful danger choice | All 15 saved same-state cases replayed the fatal edge exactly and reached the goal under the alternative controller continuation. Safe-minus-fatal returns range 118.6586888357–120.0428924937. The diagnostic used 613 actual transitions and was completed November 1, 2026, 11:12:54 p.m. EST. | Path A's hazard is reached by five opposing actions from B at t25. Its local probe is t29 after four opposing actions, where one more action can enter danger. This does **not** establish the separately proposed one-action-from-B Path B geometry or that a free learned actor will visit the approach. |
| Honeypot exclusion | The immutable final loader retains 55,649 fitting tuples from 768 parents. It excludes source, next-state and chord-crossing edges touching danger. | The absence of danger labels does not guarantee learned optimism. Forced-action Q error and free-policy exposure must be measured. Honeypot intentionally retains successful safe routes. |
| Missing connector | The final Bottleneck loader uses the closed projected B-to-C slab `[0.2, 0.8]`, including jumps/crossings in either direction. Actual consumed fitting tuples: 53,669, from 768 parents; zero intact successful fitting trajectories; 716 genuine upstream/downstream goal-fragment pairs. | This is unsupported-connector generalization. Missing labels cannot guarantee underestimation or successful stitching. No fabricated transition connects the fragments; no unique physical route is asserted. The slab can change the state/time mixture. |
| Retained full-state endpoints | Bottleneck fitting support at B/t25: 93 distinct parents; at C/t30: 167. Saved checks use physical-state, current and wave tolerances at the exact time. | This is the specified local support evidence, not a global density or action-only shift guarantee. |
| Tuple and role integrity | Original row IDs, times, next observations and terminal flags are preserved. Deleted gaps have distinct fragment IDs; `next_action` is not consumed. Calibration uses one retained original tuple from each of 128 distinct calibration parents, disjoint from fitting. Normalization and auxiliary fitting use only condition-retained fitting rows. | No conditional omitted-region or selected-action calibration guarantee is claimed. The construction data have already been inspected as development. |
| Matched method | Two conditions, native UniFloral TD3+BC versus original BCA `Q1 - 3 U`, three independent initialization seeds. Each condition/seed has a 10k native warmup and matched 10k native/BCA continuations to 20k cumulative updates. BC coefficient remains 1; native alpha remains 2.5. | BCA directly changes the actor score with a live action gradient. Native critic update/targets remain unchanged in form. A downward penalty cannot directly raise a fixed underestimated Q. No Bayesian-uniqueness claim or matched simple-penalty attribution is available from these two arms. |
| Evaluation | 128 shared return starts per final policy; 16 full-state/time anchors per condition/seed. Shared candidates are controller, opposite controller and zero; an own-action fourth slot is excluded from shared-action ranks. Own-policy forced-first-action Q uses the same deployed actor thereafter; the common-controller panel is separately labeled. Early terminal/horizon and full-state restoration are explicit. | Returns are primary. Raw Q error, adjusted-score error, ranking benefit/harm and exposure remain separate. Error tolerance is 1 return unit, tie tolerance 1e-10. No bank replacement, outcome filtering or guaranteed error sign. |
| Actual Zeus technical runtime | Saved smoke PASS at November 2, 2026, 3:43:54 a.m. EST. Real disposable native/BCA updates, genuine calibration, exact restored actor-bearing next step, production evaluator and early-terminal paths passed. All six fixed probe banks contained retained and omitted first edges. | This was a real GPU smoke, not merely fixtures. It used Bottleneck numerical updates and checked both data bindings/banks. It is not a Titan compatibility pass, a 20k learning/convergence check or efficacy evidence. |

The smoke bank counts (omitted / retained out of 48 common first edges) were Honeypot 16/32 for every seed and Bottleneck 37/11, 40/8, 38/10. These are forced-probe coverage counts; actual learned-policy exposure remains an outcome.

## Actual execution state

- Completed and reusable: `honeypot_s0_warmup`, 10,000 critic / 5,000 actor / 10,000 auxiliary updates and one calibration. Full-state SHA-256 `5422f85ac77d2a51341bb1154d7ba057a623854fd5ec8232b762bf0443bc218f`.
- Original production run interrupted at November 2, 2026, 4:57:31 a.m. EST after 769.481604245 seconds. The unfinished native branch last recorded 4,200 critic/auxiliary and 2,100 actor updates, without a checkpoint; its unobserved tail remains bounded, not zero.
- Zeus `run_recovery_1` contains a migration hold, not an executed recovery reservation. There is no `ATTEMPT.json` or `RESULT.json` there. Do not remove the hold or rerun an old command.
- The live Zeus original source seal currently has one known mismatch: `two_scenario_v2/SETUP_APPROVAL.json` changed only line endings. The isolated Titan bundle contains the authenticated frozen CRLF bytes; the live file remains untouched. The strict pin check must not be weakened or silently resealed.
- Titan validation 1 started but has no terminal compatibility receipt. User-provided evidence showed GPU/data-binding success followed by `before_template_initialization`; PID 34106 was absent at the later check. Its failure cause is unknown. This review has no direct Titan access and cannot establish newer Titan state.
- No completed Path A BCA policy or final scientific evaluation is present in accessible Zeus evidence. The completed knee/stall confirmation is a separate study and does not satisfy this workload.

## The only remaining launch components

| Item | Owner / action | Pass evidence | Current status |
|---|---|---|---|
| Reconcile possible Titan-only `path_a_a1.py` | David: run the existing read-only project search; if present, supply source/hash for review without executing it. Review agent: reconcile any conflicting helper. | Verified absence/not used, or an explicit reviewed reconciliation; no competing Path A attempt. | **Unresolved prerequisite.** David says no Path A training ran on Titan; helper presence remains unknown. This is conflict prevention, not a new research test. |
| Use the corrected existing transport | David: install the already delivered retry ZIP version 1 in the existing isolated `BCA-PathA` tree, following its manifest-bound installer. | ZIP SHA and payload pins match; installer receipt binds the original dataset/checkpoint/source closure and the first validation files. | **Prepared, target installation unverified.** The original 671-file transfer was user-verified; do not recollect data or copy a fresh warmup. |
| Complete the approved supervised zero-update compatibility retry | David: run the existing `--prepare` and one `--validate` command. Review terminal result. | Authenticated `titan_validation_2/RESULT.json` pass, same runtime/profile and exact checkpoint/source/data checks, no unexpected work. | **Pending.** CPU checks and validation 1 do not substitute for this. Preserve all stdout/stderr/exit/timeout evidence if it fails; no automatic retry. |
| Dispatch the unchanged recovery on one target | David: only after the authenticated pass, run the existing conditional `--execute`. | Exclusive new attempt, same approved lineage, positive remaining allowance, Zeus hold retained. | **Conditional approval already exists; not dispatched by this review.** No distributed lock is claimed. |

The exact current installation, read-only helper check and commands are in [retry README](../path_a_titan_validation_retry_v1/README.md) and [RUN_COMMANDS.md](../path_a_titan_validation_retry_v1/RUN_COMMANDS.md). These links are authoritative for execution. Do not follow the historical Zeus or first Titan launcher commands. A copied command without the current receipt is insufficient.

Corrected retry ZIP: SHA-256 `a7015234eac28e1564596fa2e43c65c2df21a1c6e10951a3a2593cd51367ef73`, Library item `libfile_b77e8ea94f788191888f0849dbdc9cc6`, version 1. This document does not install or execute it.

## Remaining workload and unchanged limits

After compatibility passes, reuse the saved warmup and execute only 17 remaining training jobs plus the frozen evaluation. These produce the originally planned 12 final policies; native and BCA each add 10k updates from their common 10k fork. There is no mid-job checkpoint for the interrupted native segment, so that branch restarts from its warmup, not from 14,200.

- New scientific work: at most 170,000 critic / 85,000 actor / 170,000 auxiliary updates and 17 calibrations.
- Conservative cumulative physical upper limits, including smoke and the old lost tail: 190,014 critic / 95,008 actor / 190,014 auxiliary and 22 calibrations. The old unknown tail remains an interval even if recovery completes.
- Evaluation: 1,536 return episodes; 768 own-policy action references and 288 common-controller references; 237,600 reserved new simulator transitions (233,376 tight bound). Cumulative planned maximum: 430,575 of the original 498,400.
- The original four-hour allowance is **not reset**. From the 10,660-second recovery remainder, previous preparation 60 seconds and both validation reservations of 120 seconds are charged. Production is at most 10,360 seconds minus current preparation execution, charged at least 60 seconds: initially at most **10,300 seconds (2h 51m 40s)**. Larger measured/bounded execution reduces this further. Idle human waiting is not charged. The runner computes and enforces the actual remaining allowance; it is a cap, not an ETA.

Path B's new one-action-from-B geometry/recollection, Colab migration and the knee/stall study are not prerequisites for Path A and must not share or reset this reservation. No Path B collection budget is approved. The previously delivered Colab source directory is absent in this checkout and has no established target runtime pass; do not use it to bypass Titan's recorded attempt.

## After launch: what will answer the research questions

Completion requires all 12 final authenticated policies and the frozen evaluation, including unfavorable or neutral outcomes. Report each of the three paired native/BCA return differences per condition, goal/danger/horizon counts, actual omitted-edge exposure, raw Q over/under/near-zero errors and MAE, adjusted-score crossings and harmful/beneficial choices. Separate own-policy and common-controller values. A missing connector may be learned, ignored, or already valued correctly; absence of BCA benefit does not establish that all offline stitching is impossible.

There is no implementation change required by this audit of the narrow approved study. Broader claims remain unavailable: guaranteed optimism/underestimation, correction of underestimated fixed Q by a downward penalty, action-only causal isolation, uniqueness of the Bayesian contribution, or a literal one-action hazard at B. These are scope boundaries, not extra launch gates.

## Checks actually performed in this review

Fresh CPU tests: **9 binding tests passed in 4.295 seconds; 12 evaluator tests passed in 3.733 seconds**. The binding suite used the actual immutable saved arrays and a namedtuple transition adapter. It independently recomputed zero intact Bottleneck successes, 716 genuine pairs, endpoint support and exact retained tuples. The evaluator suite injected fake dynamics and fake policies, covering full state/time, early terminals, no anchor replacement, common/own-action separation, signed harms, paired identities, failure persistence and counts. No actual plant, model, GPU, optimizer, calibration or WSL execution occurred.

Historical preparation evidence is separate: 67 smoke-preparation tests, 78 recovery-preparation tests, 98 relocated Titan source/fixture tests plus seven unpacker tests, and 33 corrected retry tests. Those receipts are not relabeled as fresh tests. The saved real Zeus smoke and warmup are actual runtime evidence. The original sealed data loader still emits an unclosed-reader ResourceWarning in the binding test; data/pin checks pass, and no historical source was changed to suppress it.

Skills actually read and applied:

- `.agents/skills/offline-rl-experiment-design/SKILL.md`: paired units, fixed scope/seeds/budget, separate technical checks, action diagnostics and trained-policy efficacy, no new confirmation claim.
- `.agents/skills/bca-code-and-calibration-audit/SKILL.md`: actor-score placement, fixed BC, raw-Q versus adjusted-score semantics and calibration/coverage limits.
- `.agents/skills/thrusterchain-benchmark-validation/SKILL.md`: full-state/time, danger/slab edge masks, retained support, value-gap evidence, no guaranteed error direction.
- `C:/Users/David/.codex/skills/offline-rl-data-audit/SKILL.md` and its `references/data-contract.md`: original successor tuples and scalar fields, source episode/time linkage across cuts, genuine finite-horizon termination, no shifted-row next actions, and train-only normalization. The existing project binding tests were used on this nonstandard parent archive; no generic loader conversion or dataset rewrite was introduced.

No repository `AGENTS.md` was found in the workspace search. Internal builder/validator coordination is evidenced in this task. No external Claude agent was contacted or claimed aligned.

Source hashes, exact evidence paths, this review's observed state and check results are recorded in [VERIFICATION.json](VERIFICATION.json). Root review may append an independent-review finding to that unpinned evidence file; all sealed inputs remain unchanged.
