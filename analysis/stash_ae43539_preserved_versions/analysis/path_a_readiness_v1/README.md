# Path A: current readiness and remaining work

## Colab option added November 2, 2026, 12:42 p.m. EST

The [single Colab notebook](../path_a_colab_v1/Path_A_Colab.ipynb) now packages the same frozen continuation; it is an **optional transport, pending target validation**, not an observed running study. See its [validation report](../path_a_colab_v1/VALIDATION.md). It embeds all source and requires the existing 112.35 MB ZIP for immutable data/checkpoints. No new experiment was run.

A fresh saved-file observation at 12:42 p.m. EST still finds only the Zeus migration hold and no local Titan validation/recovery receipt. Actual Titan state is unknown. Before selecting Colab, the operator must establish that Titan validation and recovery never started and retain the Zeus hold. The notebook refuses unknown or already-consumed attempts and requires an explicit Colab migration selection. Installation, transfer and waits reduce the same remaining allowance. Use **one** target, never both.

The Titan checklist below remains its existing route. Colab instead uses the notebook's persistent ledger, isolated exact Python 3.12 package lock, zero-update compatibility and conditional continuation. Colab GPU compatibility and actual execution have not been verified. The historical documentation verification below predates this option; the new notebook report records its separate checks.

Observed on November 2, 2026, at 11:25 a.m. EST. This is a documentation-only status overlay. It supersedes stale launch/status prose, not frozen science, source, approvals, results or budgets. All times here use EST (UTC−05:00).

**The implementation and setup are prepared. The missing evidence is delivery to Titan and a successful Titan compatibility receipt. No further benchmark, method, dataset, or control-arm development is required before the existing check.**

## Actionable launch checklist

| Item | Owner and action | Existing evidence | Completion criterion | Current state |
|---|---|---|---|---|
| 1. Deliver the exact files | David/Titan operator: copy the ZIP and unpacker, verify and unpack into `C:\Users\David Bayha\BCA-PathA`; run the existing source check | 112,349,548-byte ZIP; all 671 payload files clean-extracted and verified on Zeus; all 203 historical pins match in the bundle | Titan unpacker reports all files verified with zero overwrites, followed by `source_checked_titan_not_executed` | **Target state unknown.** No delivery receipt is accessible here. No new files need to be built. |
| 2. Verify Titan compatibility | David/Titan operator: run `titan_launch.py --validate` once | User-reported Python/package metadata matches; real update/restore/evaluation smoke passed on Zeus | Authenticated `titan_validation_1/RESULT.json` says `passed_titan_validation`; exact restore/inference/controller comparisons pass with zero new optimizer/calibration/simulator work | **Required target proof pending.** CPU tests and Zeus smoke do not substitute for it. |
| 3. Start the approved recovery once | David/Titan operator, after items 1–2: use the conditional command in the runbook | Manifest-bound Titan approval exists; Zeus has an explicit migration hold; target launcher reserves output exclusively | One `run_titan_recovery_1/ATTEMPT.json` tied to the same manifest and passed validation, followed by accepted job receipts | **Dispatch not observed.** Inspect any existing target reservation instead of retrying. Do not run a second clone. |

These are existing operational steps, not three new scientific gates. The source check is already part of the prepared flow. Target disk/write access and current target process state are unobserved from Zeus; the operator must not infer delivery, free resources or absence of another attempt from this document. The migration hold is not a distributed lock. No agent is authorized to stop another study to make room.

**Approval is already present for this exact conditional recovery.** The assistant's WSL route remains denied, so the operator executes the commands. A failed check or interrupted reservation must be reported and preserved, not deleted or automatically retried. No new approval is needed merely to perform the already approved steps.

## What is already done

| Component | Evidence and limit |
|---|---|
| Fixed Honeypot opportunity | Same saved state/time, safe versus fatal first action, then the same fixed controller: 15/15 safe goals; return advantage 118.6587–120.0429. This establishes an action consequence, not learned optimism or free-policy avoidance. |
| Missing-connector data | Corrected final Bottleneck input has zero intact successful fitting trajectories and 716 genuine upstream/downstream fragment pairs; endpoint support 93/167 parents. Original successors, times and terminals are retained; deleted gaps create no artificial joins. This does not guarantee that an offline learner can infer the connector. |
| Method and evaluator | Pinned UniFloral standalone TD3+BC host; original actor score `Q1−3U`; unchanged BC. Zeus's real native/BCA update, calibration, exact restore, early-termination and evaluation smoke passed. |
| Portable implementation | 98 relocated Path A CPU tests, 7 unpacker fixtures, independent source review and real clean extraction passed. No new tests or scientific operations were run for this documentation update. |
| Durable scientific progress | Honeypot seed-0 warmup completed 10,000 critic / 5,000 actor / 10,000 auxiliary updates and one calibration, at 4:53:28 a.m. EST. Its checkpoint is in the bundle. |
| Interruption preserved | Production stopped at 4:57:31 a.m. EST. Native continuation last saved 4,200 local critic/auxiliary and 2,100 actor updates; no continuation checkpoint. The final interrupted tail remains unknown within preserved bounds. |

No new Path A validation or recovery directory is present in the accessible Zeus checkout. This does **not** prove Titan is idle. The latest separate confirmation attempt has saved training progress on Zeus; it is not Path A and is not a prerequisite. No process was started, stopped or queried through WSL in this review.

## Fixed scope and budget

- Two conditions: **Honeypot** and **Bottleneck**. Two arms: native UniFloral and original BCA at **κ=3**. Three paired initialization seeds; 12 final policies.
- BC coefficient remains **1.0**, native TD3 coefficient **2.5**. BCA changes the actor score, not the critic loss or Bellman target directly. A downward score penalty cannot directly raise a fixed underestimated Q.
- Each condition/seed has its own 10k native warmup; native and BCA branch from the same complete state and each add 10k critic updates. Final policies therefore have **20k cumulative**, not 20k additional, critic updates.
- Reuse the completed warmup. Remaining work is **17 training jobs plus one unchanged evaluation**, totaling 170,000 new critic / 85,000 actor / 170,000 auxiliary updates and 17 calibrations.
- Remaining allowance is **10,660 seconds = 60 reserved preparation/source-check-overhead + 120 compatibility + 10,480 recovery**. This is a hard cap, not an ETA or a renewed four-hour allowance. The completed Zeus warmup took 486.017 seconds; Titan's comparable throughput is unknown.
- Conservative cumulative physical caps remain **190,014 critic / 95,008 actor / 190,014 auxiliary updates and 22 calibrations**, including smoke and the interrupted tail bound.
- Final evaluation: **1,536 return episodes**, 12 policies, 6 paired condition/seed cells; at most **237,600 new simulator transitions**, cumulative maximum **430,575 of 498,400**. Compatibility adds no simulator transitions, optimizer steps or calibrations.
- Keep the same data, normalization, seeds, checkpoint rules, masks, reward/physics, probes, error tolerance **1 return unit**, and numerical tie tolerance **1e−10**. No tuning from outcomes.
- r5/knee-edge, κ=21/63/189 confirmation, Path B, head sweeps, BC strengthening, new signed-correction methods and extra control arms are **separate work, not dependencies**. Do not import their approvals, runtime limits or results.

## Work after launch: answering the research questions

| Question | Required saved result | What must not be claimed prematurely |
|---|---|---|
| Does BCA improve the learned policy? | All paired return differences per seed and condition; final success/danger/connector behavior where recorded; variability across the three trained seeds | A completed process, lower score, one seed or technical smoke is not efficacy. Report negative and inconclusive outcomes. |
| Is harmful overestimation reduced? | Raw Q1 minus forced-first-action own-policy Qπ, adjusted score minus the same reference, signed over/under/near-zero counts, magnitudes and MAE on the frozen probes | Entering danger alone does not prove optimism. Lowering Q1−3U does not change the raw critic prediction. |
| Does caution suppress a useful missing connector? | Bottleneck return/exposure, underestimation and worsening counts, harmful versus beneficial action switches and regret | Missing-connector generalization is not guaranteed stitching. No direct two-sided Q correction is implemented. |
| Is action choice improved? | Common complete-action comparisons using the common-controller panel, plus the distinct own-policy operational panel | Different own-policy Qπ references must not be treated as the same ranking truth; forced probes do not establish free-policy exposure. |
| Is the comparison complete and interpretable? | All 17 new training jobs accepted, inherited warmup authenticated, evaluation reports 12 policies / 6 paired cells / 1,536 episodes, terminal accounting and identities verified | No partial-result winner selection or automatic retry. Three trained seeds are the replication units; episodes/anchors are nested. |

This is an exploratory comparison on a development-informed construction with fixed evaluation seeds. It does not establish action-only causal attribution, conditional OOD calibration coverage, Bayesian uniqueness, or general superiority from this two-arm design. Those are limits of the claim, not prerequisites requiring another study before this run.

## Documentation authority and index

| Document | Role |
|---|---|
| [This checklist](README.md) | Canonical current scope, missing steps and postlaunch objectives |
| [Runbook](RUNBOOK.md) | One ordered operator path; exact existing transfer/check/launch commands |
| [Agent handoff and skill record](AGENT_HANDOFF.md) | Delivered internal alignment, undelivered Claude handoff, skill files actually read |
| [Documentation verification](VERIFICATION.json) | Snapshot, hashes, reviewer outcome, unchanged seals/ZIP and changed-file inventory |
| [Frozen Path A protocol](../../experiments/thrusterchain_bca/two_scenario_path_a/protocol.json) | Original scientific specification; historical `training_authorized:false` and initial time fields remain untouched |
| [Frozen smoke amendment](../../experiments/thrusterchain_bca/two_scenario_path_a/AMENDMENT.md) | Historical real-update smoke scope and approval lineage |
| [Frozen recovery](../../experiments/thrusterchain_bca/two_scenario_path_a/RECOVERY.md) | Preserved interrupted-attempt lineage; its old Zeus command is superseded operationally by the Titan handoff |
| [Frozen Titan handoff](../../experiments/thrusterchain_bca/two_scenario_path_a/TITAN_HANDOFF.md) | Current sealed machine profile and conditional runtime authority |

Pinned historical documents are deliberately not rewritten. The manifest/approval chain, not this status overlay, controls execution. The transfer ZIP remains unchanged; this status update is not silently inserted into it.
