# OOD progress audit and proposed repair plan

Prepared October 30, 2026, EST (UTC-05:00). Read-only scientific/code review; only this audit directory was written. The exact closing observation, repository state, file hashes and saved-progress snapshot are in `SNAPSHOT.json`. This is a repair proposal, not an execution approval or an amendment to any experiment.

**Closing observation: 10:05:32 p.m. EST.** HEAD remains `29ccfe50`; tracked source files are clean. Five exploratory warmups have completed worker and supervisor receipts. The sixth, `warmup_s1_random_stall3`, saved 4,500 critic / 2,250 actor / 4,500 auxiliary updates at 10:05:31 p.m. EST. Aggregate observed training is at least 54,500 critic / 27,250 actor / 54,500 auxiliary updates, with five completed calibrations. No overall terminal report exists. This is a point-in-time observation, not a promise of continuing monitoring.

Eight existing package manifests were checked against their listed files with zero mismatches. The audit snapshot hashes 197 source/config/evidence files; a passing listed-file check does not resolve the unlisted-dependency defect in R4.

## Decision and compact checklist

**Do not promote the current work to a successful gated OOD study.** Preserve both inconclusive gates and the separately identified exploratory run. Do not edit the source serving that live attempt, restart it, or change its thresholds. This audit did not terminate or otherwise control it.

- [x] Verify latest local commits, approvals and terminal gate receipts.
- [x] Recompute v1 A2 shares from its nine saved cell records: 7/16, 4/16, 2/16; INCONCLUSIVE remains correct.
- [x] Identify v2 A3's completed INCONCLUSIVE result and the separate exploratory Stage 1 attempt.
- [x] Reconcile prior findings against the revised implementation; acknowledge real fixes.
- [x] Trace remaining source, accounting and scientific-definition defects without running experiment code.
- [ ] R1: implement the written V0(f) check before any claim of v2 protocol compliance.
- [ ] R2: correct the clipped/own-policy explanation and add the existing-data decomposition; obtain a decision before changing a primary estimand.
- [ ] R3: repair multidimensional transition counting and attach an immutable accounting correction to prior receipts.
- [ ] R4: close and authenticate the transitive dependency seal.
- [ ] R5: make timeout/interruption finalization single-owner and preserve known/unknown accounting.
- [ ] R6: distinguish logical work from cumulative physical spend across retries.
- [ ] R7: bind the exploratory wrapper itself to its approval; verify historical smoke provenance without reading blinded outcomes.
- [ ] Review and test repairs in a separate version after the active attempt is terminal. No new budget or launch is approved by this document.

## What changed since the earlier 7:28 p.m. EST snapshot

The branch is `codex/thrusterchain-bca-handoff-20261026`. Initial audit HEAD was `29ccfe50d3d55447077853aecbfc566b3d83f209`; the closing value is recorded in the snapshot.

| Commit | Time, October 30 EST | What it records |
|---|---|---|
| `a286fee5ac14aba6eaa7478258f342a9b6eec3e7` | 8:20:02 p.m. | Preserved v1 A2 result; committed revised v2 protocol, A3/A4/A5 manifests, counter/watchdog/integrity changes. David author, Claude Fable 5.1 coauthor. |
| `29ccfe50d3d55447077853aecbfc566b3d83f209` | 9:00:03 p.m. | A3 v2 attempt-1 approval under David's Git identity. |

These are local commits. The separately supplied remote review, observed at 10:00 p.m. EST, found no published commit newer than `0e833441`; local origin-tracking likewise remained there. This audit did not fetch, merge, push or independently re-query GitHub. A Git author or approval file alone is not proof of who physically typed it.

### Current experiment inventory

| Experiment/stage | Verified evidence and interpretation |
|---|---|
| Central-quartile original study | Completed 144 contexts; historical Q-accuracy pass, primary policy/rescue failures. Its conclusions are unchanged by later development. |
| Knee-edge v1 A2 | Nine old native 20k cells, 288,000 transitions, zero optimizer updates. Runtime/reproduction passed; seed 2 had 2/16 qualifying probes against a 25% minimum. Final INCONCLUSIVE, October 30 6:58:23 p.m. EST. No rerun justified. |
| Knee-edge v2 A3 | Approved package hash `f681abb063d4f0d107e25a2bc9bfcc04b29e7eeed5130045b38b79c5777d0548`. Started 9:00:22 p.m.; finished 9:02:53 p.m. EST. INCONCLUSIVE: L2 item 5 passed 44/48, 42/48, 42/48 anchors; seeds 1 and 2 miss the 90% rule. This is benchmark/predictor-validity evidence, not a BCA efficacy test. |
| Formal v2 A4/A5 | Manifests now exist; corresponding approval files were absent at inspection. A3's INCONCLUSIVE result prevents formal progression. |
| Exploratory Stage 1 | `EXPLORATORY_APPROVAL.json` explicitly permits an outside-protocol Stage 1 run after A3 stopped, with no gated claim. An untracked wrapper writes `stage1_exploratory_run`, not `stage1_run`. It cannot satisfy A5's prerequisite. At 10:00 p.m. EST four warmups had completed and the fifth was at 8,000 critic/4,000 actor/8,000 auxiliary updates. Closing progress is in `SNAPSHOT.json`; saved-file activity is not a process-liveness check. |
| GPU smoke | Revised protocol says an earlier smoke ran, and quotes an 11,853-second Stage 1 projection. Accessible package documents do not identify a verifiable blinded receipt path/hash. A4 manifest says a smoke from v2 is still required. Do not assume it ran under this version or count it as a cheap unit test. |

The v2 change is explicitly post hoc: it carries v1's inconclusive proxy test forward, makes reach descriptive, adds a RANDOM-arm validity rule, and changes resource safeguards. Preserve this lineage; do not describe v2 as the original preregistered design. Its new collection law, slope-3 stall physics and global action cutoff also differ materially from the earlier central-quartile study.

### Other local development work

The five directories below remain separate from knee-edge progression. Their current local files are untracked; Git does not establish authorship or conversational authorization. Their reports disclose development reuse and negative/inconclusive outcomes.

| Package | Saved screen | Auxiliary updates | Simulator transitions |
|---|---|---:|---:|
| `action_sensitive_bca_dev_v1` | Failed | 24,000 | 0 |
| `signed_action_bca_dev_v2` | Inconclusive | 12,000 | 197,704 |
| `signed_action_bca_dev_v3` | Failed | 0 | 377,704 |
| `common_action_bca_dev_v4` | Inconclusive | 0 | 364,904 |
| `capped_width_bca_dev_v5` | Failed: 0/3 native and 1/3 BCA checkpoints qualify | 48,000 | 192,104 |
| Total | No successful successor screen | 84,000 | 1,132,416 |

All five report zero actor/critic optimizer updates. Auxiliary fits and calibration/reconstruction work are real resource use, not policy training. These totals were checked in the preceding source/evidence audit; this update preserves that assessment and hashes their current reports. The capped relative-width reranker can raise some action scores and is not unchanged calibrated BCA. Keep these negative results; do not promote them or delete them merely because they are untracked.

## Repair register

Paths below use `K = experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/`. Exact file bytes are recorded in the snapshot; line numbers describe this snapshot, not later edits. Most K sources were committed in `a286fee5`; `stage1_exploratory.py` and its approval/run artifacts are uncommitted additions.

### R1 — High: a written validity gate is not implemented (new confirmed defect)

**Evidence:** `K/PROTOCOL_V2.md:80` adds V0(f), but `K/stage1_math.py:144` and `:178` enumerate only V0(a)-V0(e). No V0(f) code path exists in the inspected Python sources. Thus a result can pass the implementation without satisfying the sealed written rule.

**Minimal patch:** implement the already specified rule on R-valid anchors: median rho at s10 at least 0.30, and one-sided median KC at s10 no greater than `0.3 * (median rho + 0.15)`. Add it to per-seed output, overall validity, label reasons and reporting. Do not select a new threshold. Keep any current attempt's original decision record; any later recomputation must be a separately labeled amended analysis.

**Acceptance:** boundary equality passes; values just outside either bound fail; empty/nonfinite R-valid data cannot pass; failure in any seed produces INCONCLUSIVE even when every other criterion passes; report contains denominators and the exact bound. Inspect all decision readers, not just the helper.

**Dependencies/decision:** coordinate with R2 because KC's reference matters. Implementing the existing written rule needs no new scientific bar, but execution/reanalysis of frozen results requires explicit lineage and review. Do not patch the active run.

### R2 — High: policy-reference confounding remains; the new cancellation claim is false

**Evidence:** `K/stage1.py:246` uses clipped Q-pi for primary KC. `K/stage1_eval.py:230` and `:232` already compute both clipped and unclipped references. `K/PROTOCOL_V2.md:131` acknowledges the mismatch but claims cancellation between CUTOFF and RANDOM. Those actors differ; their continuation-value gaps need not match.

For each anchor and candidate versus the knee-edge action, define delta as the within-state action difference. Then:

`KC_clip = KC_own + D`, where `D = delta(Q_own) - delta(Q_clip)`.

Consequently the CUTOFF-minus-RANDOM comparison contains `D_C - D_R`. Removing a common critic offset does not remove that term. Compute this identity per anchor before aggregation: medians are not additive. Native TD3 target smoothing/lag also remains distinct from deterministic deployed-policy Q-pi.

**Minimal patch:** preserve clipped physics diagnostics, add `kc_own`, `continuation_gap_contrast` and their paired decomposition from the already computed arrays, with separate finite-value checks. Correct the explanatory claim. No additional simulator references are needed for this decomposition.

**Acceptance:** a perfect own-policy critic has zero own KC even with nonzero clipped KC; unequal arm gaps survive the arm contrast; identical gaps cancel; numerical per-anchor reconstruction passes a frozen tolerance; missing/nonfinite companion arrays cannot certify accuracy. Retain signed benefit/harm and denominators, not only average magnitude.

**Decision required:** retain the clipped primary with a narrower clipped-reference claim, or approve own-policy accuracy as a revised primary. The latter requires reviewing associated relative-gap/reachability gates before resealing. Do not silently swap estimands after observing outcomes.

### R3 — High before further capped execution: the new counter undercounts episode batches

**Evidence:** `K/units.py:279` uses `x.shape[0]`. `K/dataset_knee.py:188` validates dynamics on `(1280,100,13)` states: 128,000 transitions, not 1,280. Current exploratory warmup receipts record 1,280 validation steps, confirming the affected path.

**A3 reconciliation:** recorded 2,587,760 plus `2 * (128000 - 1280) = 253440` omitted validation steps gives **2,841,200**, which is **158,800 below the newly approved 3,000,000 cap**. This is corrected source-derived accounting, not a new measurement. The old 2.5M breach finding no longer applies to this approved attempt. Its INCONCLUSIVE scientific label is unchanged. The manifest's measured-step field repeats the undercount and needs an appended correction, not historical rewriting.

**Minimal patch:** count all leading state dimensions (`product(shape[:-1])`), asserting the state-coordinate dimension and either explicitly supporting broadcast semantics or rejecting them. Preserve attempted versus completed counts if the underlying call raises. Keep environment-step and raw-dynamics counting disjoint.

**Acceptance:** fixtures for `(13,)`, `(N,13)`, `(E,T,13)`, empty batches and broadcast cases; reject before a call would exceed the cap; no double-count through `env.step`; independently reconcile a tiny dataset's collection, replay, prefix and continuation totals. Add a source-linked accounting correction for A3 and, after termination, the exploratory run. Never replace unknown work with zero.

**Decision:** no new scientific threshold is needed. Correct accounting does not authorize additional compute or changes to the active attempt.

### R4 — High before future launch: integrity improved but dependency closure remains incomplete

**Status:** the previous listed-file live-edit gap is substantially improved. `K/stage1.py` captures an integrity record once; workers verify it before package imports. This catches later changes to listed files. Do not repeat the old claim that no worker check exists.

**Remaining evidence:** `K/units.py:232` includes package files and only `step03.RERUN1_SOURCES`. That list still omits rerun1 `design.py` used by `runtime.py:332` to compute the distance coefficient; the dynamically imported `matched_policy_pilot_v1/launch.py` supervisor; and environment/payload dependencies reached through bindings. The added check authenticates a runtime snapshot, not the complete approved dependency closure. It also cannot eliminate all check-then-load races by itself.

**Minimal patch:** enumerate the scientific/execution dependency closure and seal it under the reviewed manifest. Derive worker records from those expected hashes, not an unrestricted collection of whatever files happen to exist. Verify before dispatch, before loading and after work; an immutable run-local code snapshot is preferable while other agents edit the checkout. Include the actual launcher/wrapper in the approved identity.

**Acceptance:** mutate a local module, `design.py`, environment wrapper, payload adapter and supervisor independently and require rejection before scientific work. Unchanged sources start normally. Simulate edits between jobs and check both producer identity and terminal status. No claim is made that these omitted files actually changed during prior runs.

### R5 — Medium: watchdog added, but terminal accounting/races remain

**Status:** A3 now has a one-hour watchdog (`K/step04.py:311`, `:380`), substantially addressing the former cooperative-only cap. The completed 151-second A3 run was unaffected by timeout handling.

**Remaining evidence:** interruption writes its receipt before `finally` attaches the counter; the watchdog has no final counter snapshot. Main-thread and watchdog finalization need one owner. Older development runners still use cooperative limits; some can write successful results before a final deadline check raises, creating contradictory result/failure artifacts.

**Minimal patch:** one terminal-state owner, capture counts/elapsed/knownness before every terminal write, stop/join the watchdog on normal completion, and preserve partial/uncertain work on termination. For future development runs, use the reviewed supervising timeout rather than relabeling cooperative checks as hard caps.

**Acceptance:** deterministic fixtures for interruption before/after a counted call, timeout, write failure and simultaneous finish/timeout. Exactly one authoritative terminal state; no success plus failure pair; incomplete work never reported as zero by default. No changes to finished receipts beyond a separate linked correction.

### R6 — Medium: failed-report fallback improved; cumulative retry spend is still incomplete

**Status:** Stage 2 now conservatively reserves the remaining allowance if its worker report is missing. Prefix/precheck work is instrumented, subject to R3. These are real improvements.

**Remaining evidence:** `K/stage1.py:165`, `K/stage2.py:152` initialize fresh attempt counters; `K/units.py:193` charges successful reused work or the current reproduction but omits some previous failed and duplicate physical work. This may be a logical per-attempt ledger, but it is not cumulative physical spend.

**Minimal patch:** separate logical target counts, current-attempt physical counts, prior known physical counts, and prior unknown upper bounds. Preserve every attempted unit in lineage, including nonreproducing retries. State whether an approval applies per attempt or cumulatively; never infer a renewed budget from a new directory.

**Acceptance:** two-attempt fixture with successful reuse, failed partial work and a repeated reproduction; every physical unit counted once, logical work not duplicated, unknown tail carried explicitly. A3/Stage1/Stage2 totals must reconcile with independent per-call accounting.

### R7 — High scope/provenance requirement: exploratory wrapper and smoke are separate execution paths

**Evidence:** untracked `K/stage1_exploratory.py:69` replaces Stage 1's gate, approval path, output path and finalization. This is disclosed, not hidden: the saved approval explicitly says A3 ended inconclusive and the run supports no protocol claim. It names the A4 manifest, which does not include this wrapper. The attempt records the wrapper hash, but approval is not bound to that hash. Presence of the file does not independently establish human authorship.

The optional `stage1_smoke.py` is 20k native updates plus 10k BC updates and two 1,280-episode datasets, not a cheap unit test. The protocol says an earlier smoke ran; no verifiable blinded receipt path/hash was available in the inspected repository. Do not read its blinded worker outcomes to resolve this bookkeeping issue.

**Minimal patch for future execution:** give each wrapper an immutable manifest with exact code, scope, output identity and budget; approval must name that manifest. Record smoke approval, version, counts and a blinded receipt. Keep its scientific outcomes blinded if that is the stated design. Prevent exploratory results from satisfying A5 prerequisites.

**Acceptance:** mutation of the wrapper invalidates approval; ordinary A4 approval cannot authorize exploratory execution; exploratory output cannot unlock A5; smoke requires an explicit bounded scope and records every resource category. Existing attempts remain immutable and explicitly exploratory.

**Decision:** verify authorization through trusted user history; do not infer unauthorized execution from missing Git history. Any new run needs its own reviewed scope. No automatic stop, restart or termination is requested by this audit.

## What to keep, fix, archive or revert

- **Keep unchanged:** original central-quartile evidence; v1 A2 and v2 A3 receipts/labels; current exploratory sources, checkpoints and receipts; negative development results. Retain independent region and dual Q-pi implementations: they check different failure modes and are justified validation code.
- **Fix in a new version:** R1-R7, with explicit source lineage and acceptance tests. Preserve the existing full-state optimizer/target/RNG/calibration restore checks, retained-row data bindings, episode-disjoint calibration and intersection normalization.
- **Archive only after termination and approval:** inactive experiments may be indexed as evidence archives; active output paths must not be moved. Do not confuse archival of results with deactivating runnable code.
- **Revert:** no blanket revert is justified. No experiment code was reverted. A targeted removal or retirement of a launcher requires a separate reviewed change and must preserve its evidence and reproducibility.

## Safe sequence after review

1. Let the existing owner manage the active exploratory attempt; obtain its terminal receipt before any source correction or archival. This document grants no authority to stop/restart it.
2. Preserve a source/receipt snapshot and reconcile actual, estimated and unknown resource use. Add append-only corrections to R3/R6 accounting.
3. Decide R2's scientific claim and the role, if any, of the stopped knee-edge design. Neither INCONCLUSIVE gate becomes PASS because training exists.
4. Implement the agreed repairs in a new package with the existing science unchanged unless an amendment is explicitly approved. Complete the small synthetic fixtures described above before a runtime test.
5. Verify the complete seal and a bounded target-runtime startup/restore check through an allowed route. The assistant-side WSL denial remains; do not route around it. GPU smoke is separately budgeted, not automatic.
6. Review the repaired package, exact compute ceilings, data exposure and approval lineage. Only then provide a manual launch command for an explicitly approved next step. **No command or new budget is supplied here.**

## Coverage, provenance and limitations

The review covered the recent local Claude-coauthored knee-edge implementation commits; A1 core numerical/data modules; A2/A3 launch and decision paths; v1/v2 training supervisors, workers, evaluator and reuse machinery; the exploratory wrapper; and the five local BCA development packages and directly used dependencies. Focused independent reviewers challenged protocol adherence, numerical/data paths, training integrity and resource accounting. All worked read-only.

This is not an exhaustive audit of every historical branch, every test body, every plot or all repository Python files. No production or synthetic experiment code, models, tests, simulations or WSL commands were executed in this update. Saved test-count claims (including the commit's 169 tests) are documentary evidence, not a newly witnessed test run. Read-only standard-library parsing, hashing and arithmetic were used. No remote sharing occurred.

Current files can change concurrently. `SNAPSHOT.json` records exact bytes for the inspected source/config/approval files and selected evidence, plus saved active progress read at closing. It is not an atomic snapshot of an entire running study. Line references and recommendations must be rechecked if their recorded hashes change. `ARTIFACT_MANIFEST.json` authenticates this audit's own deliverables; it does not authorize any experiment.
