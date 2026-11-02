# Untracked packages: reconciliation note

Prepared 2026-10-30 (evening, local time) on branch `codex/thrusterchain-bca-handoff-20261026`, folder `experiments/thrusterchain_bca/zeus_20261026_validation/`.

This is a read-only audit. Nothing was deleted, moved, renamed, committed, edited or run. This note is a new, uncommitted file. Times are file-system modified times in local time (EDT, UTC-4) unless a receipt gives UTC; the two agree wherever both exist (for example 19:10:31Z in a receipt is 15:10:31 on the file).

## 1. Bottom line

1. **The auditor's "five packages" are almost certainly these five**, a hash-linked chain: `action_sensitive_bca_dev_v1` → `signed_action_bca_dev_v2` → `signed_action_bca_dev_v3` → `common_action_bca_dev_v4` → `capped_width_bca_dev_v5`. Their own ledgers sum to **84,000 auxiliary updates** and **1,132,416 simulator transitions**, which reproduces the audit's "about 84,000" and "1.13M" exactly (section 5.4). The brief named only the first and the last; the middle three are also entirely untracked, so this note covers all five.
2. **No approval file exists for any of the five.** Only v1's protocol says in its own text that "the user authorized" it, and its seal says authorization is "external to this receipt." v2 to v5 say nothing either way. Who launched them is not recorded. They ran on CPU inside the WSL interpreter, which the committed READMEs say the assistant cannot reach (section 7).
3. **All five finished and are labeled: three failed, two inconclusive.** None was ever committed on any branch (not in any stash either), and no tracked file anywhere cites any of them. They cite each other, so they should be kept or removed as one unit.
4. **The other untracked items are a different kind of thing.** The three runtime-validation folders under `central_quartile_training_v1/`, attempt 3 under `central_quartile_training_v1_amendment1/`, and the 672 run-output files under `central_quartile_training_v1_authorized1/run/` are **cited by SHA-256 in committed files**. I verified 48 of 48 pins in the committed `AMENDMENT_LINEAGE.json` against the files on disk, and 672 of 672 run files have their SHA-256 cited in committed files (for example the `RECOVERY.json` files and the per-job reports). Deleting any of them would break a committed verification chain.
5. **Recommendation:** commit-as-record for everything in section 6 (it repairs the committed chain), and commit-as-record for the five development packages as one unit once you say whether they were authorized. Do not delete any of the 11 items; none is a byte-for-byte duplicate of committed content. Details and the decisions only you can make are in section 8.

## 2. Scope, and one correction to the brief

Named in the brief and covered: `action_sensitive_bca_dev_v1/`, `capped_width_bca_dev_v5/`, the four central-quartile validation folders, and the untracked `central_quartile_training_v1_authorized1/run/` outputs.

Not named but covered, because v5's source seal pins their result hashes and they have zero tracked files: `signed_action_bca_dev_v2/`, `signed_action_bca_dev_v3/`, `common_action_bca_dev_v4/`.

Seen but not reviewed (section 9): `knee_edge_ood_v1` and `knee_edge_ood_v2` (active study; v2 is being written right now), and the partly untracked `central_quartile_training_v1_continuation1` and `_evaluation_rerun1`.

Naming trap: the "v2" in your memory note (`conditional_ood_gate_v2`, superseded, never frozen) is unrelated to `signed_action_bca_dev_v2`.

## 3. How this was checked, and what was not

- Read: each package's protocol, report, seals, preflights, terminal results and progress logs; tracked neighbors (`AMENDMENT_LINEAGE.json`, `AMENDMENT_RECONCILIATION.md`, `AUTHORIZATION.json`, `RUN_AUTHORIZATION.json`, `RUN_RESULT.json`, READMEs); `git log`, `git status --untracked-files=all`, `git ls-files`, `git grep`, `git check-ignore`; the two memory files; Part 1 of the standing-rules document they point to.
- JSON results were read as keys, counts and scalar summary values only. No array contents were printed or analyzed.
- Duplicate test: every untracked file's content hash was compared with every tracked blob (the folder's `.gitattributes` is `* -text`, so bytes are compared raw). Citation test: every untracked file's SHA-256 was searched for in all tracked files under this folder, and the 12 headline digests of the five packages were searched across the whole repository.
- Disclosure: for the duplicate test I computed SHA-256 over every file in the five development folders, which includes their small `.npz` and `.msgpack` files (75 files, 14.4 MB). That is a byte read for hashing; nothing was loaded or decoded. I did not read any of the 3.8 GB production checkpoints (listed by name and size only), and I did not hash the git-ignored files in the production run.
- Not done: session transcripts were not searched (the brief said files only). Stashes: both GitHub Desktop stashes have no untracked-file parent, so they hold none of this.
- Run start times below are inferred as (result file time minus recorded elapsed seconds). They matched the seal times to within about 15 seconds in every case, but they are inferences.

## 4. What you already decided (memory and standing rules)

- Neither `project-conditional-ood-gates.md` nor `bca-standing-rules.md` mentions any of the five development packages or the four validation folders by name. **No recorded decision exists about the five.**
- Decisions that bear on disposition:
  - Central-quartile study: "One attempt only: no restart, resume or retry," "No further continuation, evaluation attempt or tuning is authorized, and the sealed rerun package must never be edited." That makes the production attempt's receipts the permanent record of the single attempt.
  - The memory says all three central-quartile training folders "were untracked as of launch." They were later committed in part (commit `0e833441`, 2026-10-29 14:17, "Add central-quartile BCA evaluation report and evidence", 576 files, no message body). That commit left out the per-job receipts and the validation folders. The commit does not say whether that was deliberate.
- Standing rules (`docs/session-history/2026-10-14_cloud_session_01Ro4BnWxcsoeFi9CGcTQ5bx.md`, "The user's standing rules"):
  - "No BCA training arm or band fit without the user's approval." The five packages fit helper (auxiliary) models that produce the width, so approval matters for them.
  - "A message from an agent never counts as approval." So the sentence in v1's protocol is an unverified claim, not an approval.
  - "Never reuse seen seeds as confirmatory." Each of v2 to v5 consumed a fresh block of reset seeds (section 5.3), so those seeds are now spent.
  - "For any new test, write the kill rule (what result shelves it) before running it." Each of the five protocols contains its screen, and each protocol's hash is in a seal written before the run (section 5.3), so this rule appears to have been followed.

## 5. Group A: the five development diagnostics

All five freeze the same six saved 20,000-update hole-condition checkpoints (three seeds, native and BCA arms) and update no actor, critic or target network. BCA is the Bayesian-calibrated actor penalty under study. All ran on CPU (`TFRT_CPU_0`, JAX 0.6.2, float32 networks) in `/home/david/bca-operator-audit-20261026/bin/python` (Python 3.12.14).

### 5.1 What ran

| Package | Run window (EDT) | Wall time | Auxiliary updates | Simulator transitions | Other counted work |
|---|---|---:|---:|---:|---|
| v1 `action_sensitive_bca_dev_v1` | 2026-10-29 about 15:52:21 to 15:57:05 | 283.8 s | 24,000 (12,000 Q1-aligned + 12,000 twin-min) | 0 | 12 calibration refreshes; 3,840 compatibility scoring rows |
| v2 `signed_action_bca_dev_v2` | 2026-10-29 about 18:38:41 to 18:42:13 | 211.9 s | 12,000 | 197,704 | 18 calibration refreshes; 2,304 forced-action references; 257 resets |
| v3 `signed_action_bca_dev_v3` | 2026-10-30 about 10:02:07 to 10:04:23 | 135.3 s | 0 | 377,704 | 18 refreshes; 5,760 references; 257 resets |
| v4 `common_action_bca_dev_v4` | 2026-10-30 about 12:02:55 to 12:05:40 | 164.1 s | 0 | 364,904 | 18 replayed reconstruction refreshes; 5,760 references; 129 resets |
| v5 `capped_width_bca_dev_v5`, paired refits | 2026-10-30 about 15:28:39 to 15:39:31 | 651.2 s | 48,000 | 0 | 24 calibration refreshes; 23,040 width rows |
| v5, reranker evaluation | 2026-10-30 about 15:35:28 to 15:43:16 | 467.6 s | 0 | 192,104 | 960 full episodes; 960 first-action references; 1,152,000 candidate-score rows; 33 resets |

The two v5 components overlapped from about 15:35 to 15:39 (inferred from file times). Every package's completed transitions equal its declared cap exactly (197,704; 377,704; 364,904; 192,104), and every ledger shows zero actor, critic and target updates.

### 5.2 Approval, outcome, duplicates, recommendation

| Package | Approval evidence | Outcome label | Duplicates a tracked file? | Recommendation |
|---|---|---|---|---|
| v1 | None as a file. Protocol line 3 asserts it (quoted below). | `completed_development_diagnostic`, screen **failed** | No (0 of 23 files match any tracked blob) | Commit-as-record, with v2 to v5 as one unit |
| v2 | None. Text is silent. | `completed_frozen_checkpoint_diagnostic`, screen **inconclusive** | No tracked match; internal plot-pass repeats (5.3) | Same |
| v3 | None. Text is silent. | `completed_expanded_frozen_diagnostic`, screen **failed** | No tracked match; `test_fresh_data.py` is a copy of v2's | Same |
| v4 | None. Text says only "No user-authorized training study is included." | `completed_common_action_continuation_diagnostic`, screen **inconclusive** | No | Same |
| v5 | None. Text is silent. | `completed_paired_auxiliary_refits` and `completed_frozen_controller_evaluation`, screen **failed** | No tracked match; `plots_v1` and `plots_v2` share 6 identical files | Same |

### 5.3 Per-package notes

**v1, `action_sensitive_bca_dev_v1/`** (23 git-visible files, 17.5 MB; 12 ignored `.msgpack` files, 1.06 MB)
- Purpose: a bounded mechanism screen of two modifications on the saved checkpoints: (1) refit the width helper against the critic head the actor actually uses (online Q1) instead of the twin minimum, and (2) add a training-only "action compatibility" penalty built from the nearest retained training states.
- Authorization text, quoted from `DIAGNOSTIC_PROTOCOL.md` line 3: "The user authorized implementing the two modifications and a bounded diagnostic on saved checkpoints, but not a full training study." `DEV_SEAL.json`: "Byte integrity only; user authorization is external to this receipt".
- Outcome, quoted from `DIAGNOSTIC_REPORT.md`: "completed; the preregistered exploratory mechanism screen failed in both contexts." The Q1-aligned width had lower error than the matched twin-min fit in 3 of 12 entries and higher in 9.
- Sequence: code and tests 15:38 to 15:45, protocol 15:49:34, seal 15:52:06, run, report 15:59:02. A 4.5-second CPU neighbor-parity check at 15:43:55 used 0 updates and 0 transitions.
- Depends on committed material: its pinned reference report hash `55cae667…` is the committed `central_quartile_training_v1_evaluation_rerun1/run/evaluation/report.json` (verified on disk), and it pins 38 source files including production checkpoints.

**v2, `signed_action_bca_dev_v2/`** (50 files, 27.1 MB; 9 ignored files, 1.34 MB)
- Purpose: test a "signed" one-sided width (predicts online Q1 minus the one-step target) plus a revised compatibility penalty, using 64 fresh calibration episodes and 64 separate check episodes.
- Outcome (`REPORT.md`): "completed; the prespecified action-selection screen was inconclusive in both contexts." Opportunity failed: ordinary banks had 0 to 3 high-load raw-Q1 mistakes, strong-current 2 to 7, against a required 8. The combined score was also less accurate than the symmetric width in all 12 comparisons.
- Seeds consumed: ordinary anchors 2026121000 to 2026121031, free starts 2026121100 to 2026121131; strong anchors 2026121200 to 2026121231, free starts 2026121300 to 2026121331; prefix action seeds 2026121401 and 2026121402; shuffle stream 2026121510.
- Unrecorded work: its report says the preflight "separately used six discarded one-step auxiliary test updates," but no receipt file exists for them.
- Internal repeats: three plot passes (`plots`, `plots_revision_01`, `plots_revision_02`); the three `paired_metrics.csv` files are identical, and three PNGs repeat between revisions 01 and 02. `plots/` and `plots_revision_01/` have identical `FIGURE_GUIDE.md`.
- Seal pins v1's result and seal by hash (verified).

**v3, `signed_action_bca_dev_v3/`** (35 files, 14.4 MB; 3 ignored files, 1.04 MB)
- Purpose: an expanded 15-action bank on new fresh state pools (16 natural plus 16 "ambiguity-selected" states per context per time block) with signed-scale control multipliers, asking whether the guarded hybrid beats its controls. It reuses v2's fitted helpers, so it fits nothing.
- Outcome (`REPORT.md`): "the prespecified guarded-hybrid screen failed in both contexts and both state strata." In strong current on natural states the hybrid qualified for 2 of 3 BCA seeds and 0 of 3 native seeds; two per group were required.
- Seeds consumed: ordinary resets 2026122000 to 2026122063, strong 2026122100 to 2026122163, prefix 2026122200 and 2026122201, behavior root 2026122300, tuple selection 2026122301.
- Seal pins v2's result and seal by hash (verified). Preflights show 0 updates and 0 transitions.

**v4, `common_action_bca_dev_v4/`** (31 files, 13.8 MB; 6 ignored files, 0.52 MB)
- Purpose: hold states and candidate actions common across the native and BCA critics and apply both saved actors as the continuation, to separate critic effects from continuation effects.
- Outcome (`REPORT.md`): "the prespecified own-visited action-selection screen was inconclusive in ordinary and strong current" (each seed and arm had fewer than the required four non-tied changed actions).
- Seal pins v3's result and seal by hash (verified). Its new reset seeds and prefix streams (128 and 2) are recorded in `RESULT.json` `seed_audit`; I did not extract the numeric block.

**v5, `capped_width_bca_dev_v5/`** (63 files, 2.7 MB; 45 ignored `.msgpack` and `.npz` files, 10.4 MB, including `CAP_PROBES.npz`, whose hash `777a425c…` is pinned in the reranker seal)
- Purpose: (1) measure how much the width contrast varies across four paired refits per checkpoint; (2) test whether clipping the action-dependent width ("capped width") helps five fixed controllers, evaluated as full online episodes from 16 fresh starts per context.
- Outcome (`REPORT.md`): "both bounded components completed, but capped width failed its prespecified strong-current screen." Strong-current qualifying seeds were 0 of 3 native and 1 of 3 BCA; no seed lost more than 1 unit against its actor under ordinary current. `COMPARISON_AUDIT.json` is `incomplete` (50 null metadata fields), which concerns the comparison manifest, not the run.
- Seeds consumed: ordinary 2026130000 to 2026130015, strong 2026130100 to 2026130115, parity reset 2026102803; its seed audit checked 2,261 prior seeds.
- Seals pin v4's result and seal, and v3's result, by hash (verified). Post-run parity check: 192 of 192 episodes exactly equal.
- Sequence: protocol 15:23:50, refit seal 15:27:30, reranker seal 15:34:14, report 15:52:18.

### 5.4 Totals, with sources

| Quantity | v1 | v2 | v3 | v4 | v5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Auxiliary updates | 24,000 | 12,000 | 0 | 0 | 48,000 | **84,000** |
| Simulator transitions | 0 | 197,704 | 377,704 | 364,904 | 192,104 | **1,132,416** |
| Recorded wall time (s) | 283.8 | 211.9 | 135.3 | 164.1 | 1,118.8 | **1,914.0 (31.9 min)** |

Sources: the `ledger` and `elapsed_seconds` fields of
- `action_sensitive_bca_dev_v1/diagnostic_run_01/RESULT.json`
- `signed_action_bca_dev_v2/diagnostic_run_01/RESULT.json`
- `signed_action_bca_dev_v3/diagnostic_run_01/RESULT.json`
- `common_action_bca_dev_v4/diagnostic_run_01/RESULT.json`
- `capped_width_bca_dev_v5/paired_refits_run_01/RESULT.json` (651.2 s; 48,000 updates; 0 transitions)
- `capped_width_bca_dev_v5/reranker_run_01/RERANKER_RESULT.json` (467.6 s; 192,104 transitions; 0 updates)

The last `complete` line of each `progress.jsonl` agrees on the counts (v1: 24,000 updates; v5 refits: 48,000). Its elapsed seconds differ from `RESULT.json` by under a second (v1: 284.5 against 283.8), and I used the `RESULT.json` values.

Not in these totals:
- v2's six discarded one-step preflight updates (report text only, no receipt), so the auxiliary-update total is 84,000 as recorded and 84,006 if that sentence is taken at face value.
- Preflights and checks that recorded 0 updates and 0 transitions: v1 neighbor parity (4.5 s), v3 `PREFLIGHT_REAL_MODELS_FINAL.json`, v4 `PREFLIGHT_REAL.json`, v5 `RERANKER_PREFLIGHT.json` (172.6 s). Unit-test runs on Windows are not recorded anywhere.
- Zero actor, critic and target updates in all five.
- The authorized production study (section 6) is a different category and is not part of the audit's 84,000 or 1.13M.

### 5.5 Why the five should be kept or removed together

- **Results chain:** v2's seal names v1's `RESULT.json` and `DEV_SEAL.json` hashes; v3's names v2's; v4's names v3's; v5's names v3's and v4's. All hash matches verified on disk.
- **Code chain:** v3's, v4's and v5's preflights and runners pin source files in earlier packages by absolute path (for example v1 `q1_bca.py`, v2 `signed_bca.py` and `revised_compat.py`, v3 `runner.py` and `score_controls.py`). Moving or renaming any package breaks path-keyed verification of the later ones.
- **Spent seeds:** the seed blocks above are recorded only inside these packages. Removing them erases the record of which seeds are no longer usable as confirmatory.
- **Nothing tracked depends on them:** no tracked file names them or cites any of their 12 headline digests. This is why the audit sees them as orphans, and why leaving them uncommitted breaks nothing.
- **Cost of keeping:** 202 git-visible files, 75.5 MB (the four largest are `RESULT.json` files of 13.1, 11.5, 6.8 and 6.8 MB), plus 14.4 MB of git-ignored `.npz` and `.msgpack`.

## 6. Group B: central-quartile validation folders and production-run outputs

These were produced under the committed central-quartile packages. Their purpose and authorization are recorded in tracked files.

### 6.1 Summary

| Item | Untracked files | What it is | When (EDT) | Outcome | Cited by committed files? | Recommendation |
|---|---:|---|---|---|---|---|
| `central_quartile_training_v1/runtime_validation_v1_archived_interrupted_20261028T193817Z/` | 4 (+1 empty ignored `worker.log`) | Attempt 1 of the runtime validation, interrupted by you | 2026-10-28 15:10:11 to 15:10:31; archived 15:38:17 | `user_interrupted_terminal_receipt_missing` | Yes, all 5 pinned in `AMENDMENT_LINEAGE.json` | Commit-as-record |
| `central_quartile_training_v1/runtime_validation_v1/` | 6 (+1 empty ignored) | Attempt 2 output | 16:05:46 to 16:06:48 | `failed_synthetic_runtime_validation` | Yes, all 7 pinned | Commit-as-record |
| `central_quartile_training_v1/validation_retry_20261028_v1/` | 8 | Wrapper that ran attempt 2 (code, tests, review, result) | Prepared 15:41 to 15:43; run 16:05:46 | Prepared and ran attempt 2 | Yes, all 8 pinned | Commit-as-record |
| `central_quartile_training_v1_amendment1/runtime_validation_v1/` | 6 (+1 empty ignored) | Attempt 3 output | 17:30:15 to 17:31:33 | `passed_synthetic_runtime_validation` | 5 of 6 pinned (`progress.json` is not) | Commit-as-record |
| `central_quartile_training_v1_authorized1/run/` | 672 (22.1 MB) + 3.83 GB ignored | Per-job receipts and checkpoint manifests of the one authorized production attempt | 2026-10-28 18:31:39 to 23:05:40 | `stopped_after_incomplete_job` | Yes, 672 of 672 | Commit-as-record (files); back up the ignored `.msgpack` |

### 6.2 Notes

**Attempt 1 (archived).** Reserved 15:10:11, worker entered 15:10:25, last durable progress 15:10:31 at stage `synthetic_initialization` with zero updates and zero transitions; you interrupted it, and no terminal receipt was written. Final counts are unknown (`prior_final_counts_known: false`); elapsed lower bound is 6.33 s for the worker and 19.76 s for the supervisor. The four files were moved into the archive folder at 15:38:17 with hashes verified before and after (`ARCHIVE_RECEIPT.json`). Authorization is by standing README instruction, not an approval file: `central_quartile_training_v1/README.md` line 22, "The user may manually run the following after the package is sealed:", and `protocol.json` says `execution_authorized: false` for production.

**Attempt 2 (`runtime_validation_v1/` and `validation_retry_20261028_v1/`).** Reserved 16:05:46, worker 57.31 s (62.32 s with postcheck), exit code 1, status `failed_synthetic_runtime_validation`; the committed lineage records the error as "ValueError: Full-state restore not byte exact." Synthetic work only: 8 critic, 4 actor, 4 auxiliary updates and 1 calibration; 0 transitions; 0 production rows. `validation_retry_20261028_v1/RESULT.json` and `runtime_validation_v1/RESULT.json` are byte-identical (`df10c62b…`) and both are pinned under their own paths, so keep both. The retry wrapper's own `RETRY_PREPARED.json` says `production_execution_authorized: false`; its README gives the manual command for you to run; its 15 CPU fixture tests passed in 0.653 s.

**Attempt 3 (amendment1).** Reserved 17:30:15, worker 73.20 s (78.02 s with postcheck), exit 0, `passed_synthetic_runtime_validation`. Synthetic work: 23 critic, 10 actor, 19 auxiliary updates, 4 calibrations, 10 resets and 3 simulator transitions in the Gymnasium checker; read-only source initialization of 4,480 episodes (448,000 tuples) with 0 optimizer updates. Authorization, quoted from the tracked `AMENDMENT_LINEAGE.json`: "User 2026-10-28T20:50:41UTC approved scoped reconciliation and exact serialization repair; no execution authorization". `RESULT.json` is byte-identical to the tracked `VALIDATION_RESULT_3.json` (`e7d3f03d…`). `report.json` (`490cc4e6…`) is cited by the tracked `VALIDATION_RESULT_3.json`, `central_quartile_training_v1_authorized1/AUTHORIZATION.json` and `PREPARED.json`.

**Production run outputs (`authorized1/run/`).**
- Authorization, quoted from the tracked `AUTHORIZATION.json`: question "Do you approve the 12-hour maximum for the three-seed training study?", response "Yes", scope "One prepared three-seed central-quartile study; 36 calibrations; no automatic retry, expansion or WSL access override", time 2026-10-28T21:49:53+00:00, `approved_attempts: 1`, `resume_authorized: false`. It cites the attempt-3 result `e7d3f03d…` by hash.
- What ran: attempt reserved 22:31:39Z (18:31:39 EDT); stopped about 23:05 EDT. `RUN_RESULT.json`: status `stopped_after_incomplete_job`, `execution_counts_known: false`, `all_declared_jobs_completed: false`, supervisor elapsed 16,451.8 s (4 h 34 min). 32 jobs completed (9 warmups, 23 branches). Job 33, `seed2_hole/bca`, ended with exit -9 and "KeyboardInterrupt: Supervisor signal 2".
- Counts, from the tracked per-job `report.json` files (completed jobs only): 160,000 actor, 320,000 critic and 320,000 auxiliary updates. The interrupted job's last durable snapshot adds 3,300 actor, 6,600 critic and 6,600 auxiliary updates, giving a lower bound of 163,300 / 326,600 / 326,600. Simulator transitions: 0 (no job report shows a nonzero value; the evaluation did not run in this attempt). These totals are a lower bound for the same reason the run's own receipt says counts are unknown.
- Untracked content: 33 each of `DISPATCH`, `WORKER_ENTRY`, `data_receipt`, `matching_receipt`, `progress`, `refresh_receipt`, `runtime_receipt`; 32 `actor_manifest`, `state_manifest` and `unique_replay_source_row_ids.i64`; 9 `untrained_actor_manifest`; 335 checkpoint `manifest.json`; 1 `PAIRED_INITIALIZATION.json`. Total 672 files, 640 JSON plus 32 `.i64` (binary row-id lists; not opened).
- Citing files include the tracked `continuation1/RECOVERY.json`, `evaluation_rerun1/RECOVERY.json`, `evaluation_rerun1/RERUN_INPUTS.json`, `evaluation_rerun1/run/evaluation/report.json`, and the job `report.json` files.
- Duplicates: the 33 `runtime_receipt.json` files are byte-identical to one tracked receipt from the 2026-10-27 support-shift pilot (same runtime fingerprint); that is not a duplicate of the run. Otherwise unique.
- Ignored checkpoints: `.gitignore` lines 11 and 12 ignore `*.msgpack` and `*.npz`. Measured today: 3.83 GB, of which 335 `full_state.msgpack` files are 3,467.5 MB. Every one of the 335 checkpoint folders has its `full_state.msgpack`, so nothing appears missing; your memory note's "about 4.2 GB" was an earlier estimate that I did not reconcile.

## 7. Gaps and unknowns

| # | Gap | Status |
|---|---|---|
| G1 | No approval file for any of v1 to v5. | Confirmed. `find -iname '*APPROVAL*'` over this whole folder returns only `knee_edge_ood_v1/A2_APPROVAL.json`, and a search of the five folders for names containing approv, authoriz, consent or permission finds nothing. |
| G2 | Who created and who launched v1 to v5. | **Unknown from files.** No author or launcher field exists. All five ran in the WSL interpreter; the committed READMEs say "Assistant-side WSL access remains denied," so the launcher was you or another process with WSL access. The validation attempts and the production run are the opposite case: the READMEs, receipts and memory place the launch in your own terminal. |
| G3 | Whether v1's sentence "The user authorized…" reflects a real approval. | **Unknown.** It is text inside the package; under your rule it is not an approval. |
| G4 | Whether v2 to v5 were authorized. | **Unknown.** Their text neither claims nor denies it. |
| G5 | v2's six discarded preflight updates. | Report sentence only; no receipt. |
| G6 | Why `0e833441` left out the per-job receipts and validation folders. | Unknown; the commit has no body. |
| G7 | Session transcripts. | Not searched. The linked sessions `thrusterchain-researc-3` and `thrusterchain-research-i` are the likeliest places to find the approval messages for v1 to v5, if any exist. |
| G8 | v4's numeric reset-seed block. | Present in `RESULT.json` `seed_audit`; not extracted. |

Committing does not retroactively create authorization. A commit made by you would record custody and a date; it would not show that you approved the runs.

## 8. Recommendations and decisions for you

**Recommendation by item**

| Item | Action | Reason |
|---|---|---|
| Section 6 items (24 small files + 672 run files, about 22 MB) | **Commit-as-record** | A committed lineage already depends on them by hash; a clean clone currently cannot verify its own pins (20 of the 48 lineage pins point to untracked files). Committing adds files; it edits nothing sealed. |
| Git-ignored production `.msgpack` (3.83 GB) | **Keep, and back up outside git** | Too large for normal commits; the committed checkpoint manifests pin each one by hash. |
| The five development packages (202 git-visible files, 75.5 MB) | **Commit-as-record as one unit, after you answer D1** | Unique content (no tracked duplicate), a verified hash chain, spent-seed record, and negative results your shelving rules want kept. Keeping the path names unchanged preserves path-keyed hashes. |
| If you did not authorize some of the five | **Archive instead of commit** | Keep them together, unmoved internally, with a cover note stating they were unauthorized, rather than placing them in the main history. Still do not delete: they are the evidence. |
| Superseded plot passes (v2 `plots` and `plots_revision_01`; v5 `plots_v1`) | Optional trim only | Repeats of later passes, each with a `PLOT_RECEIPT.json`; savings are small. I do not recommend trimming. |
| Delete | **Not recommended for any item** | None of the 11 items is a byte-for-byte duplicate of committed content, so deletion destroys unique evidence. |

**Decisions only you can make**

- **D1.** Were v1 to v5 authorized, not authorized, or do you not remember? Nothing in the files can answer this.
- **D2.** Commit section 6 first (it repairs the committed chain regardless of D1)?
- **D3.** For the 14.4 MB of ignored `.npz` and `.msgpack` in the five packages (including `CAP_PROBES.npz` and the fitted helper models): force-add them (`git add -f`) so the commit is self-contained, or leave them local?
- **D4.** Should I search the two linked session transcripts for the approval messages (G7)?

**Do not do**

- Do not move or rename any of the 11 folders: committed and cross-package hashes are keyed by path.
- Do not delete the two empty `worker.log` files: the lineage pins their hash (`e3b0c442…`, the empty-file hash). They are git-ignored by `*.log`.
- Do not edit `AMENDMENT_LINEAGE.json`, any `PREPARED.json`, or any seal to "fix" the untracked state.

## 9. Other untracked items seen (not reviewed)

- `knee_edge_ood_v1/` (26 untracked): later-stage code (`stage1*.py`, `stage2*.py`, A4 and A5 READMEs) and `step03_run/attempt_1/` outputs, the A2 run after your 19:55 approval commit `8e79ab9b`. Part of the active knee-edge study.
- `knee_edge_ood_v2/` (51 untracked; newest file 20:50 local): being written right now by another session; 33 of its files are byte-identical to tracked `knee_edge_ood_v1` files. Leave alone.
- `central_quartile_training_v1_continuation1/` (233 untracked: evaluation outputs and `seed2_hole`, `seed2_random` job receipts) and `_evaluation_rerun1/` (1 untracked: `run/PAIRED_INITIALIZATION.json`): all 234 hash-cited by committed files, and 149 are byte-identical to tracked files (the examples I looked at are rerun1 files; consistent with the memory note that rerun1 reproduced the earlier contexts byte for byte). The same commit-as-record logic as section 6 applies; this note did not review them.
- Repo-level untracked folders: `.agents/`, `.claude/worktrees/`, `analysis/central_quartile_review/qa_pdf_render/`, `analysis/ood_region_activity/`, and the nested `bca/` repository (kept out of commits per the earlier note).

## 10. Evidence index

Headline digests (SHA-256, computed on disk today):

| File | Digest |
|---|---|
| v1 `diagnostic_run_01/RESULT.json` | `2ee1121f0610a33312cc2e5341cf5cfebb570ffcb5e336b97993d8647127591f` |
| v1 `DEV_SEAL.json` | `5a8e0cd82acf507e18d0f1e1f1620a480cfc5f03f574286c136380bff1819a02` |
| v2 `diagnostic_run_01/RESULT.json` | `db19fb35637df3417c8c9fd9a8abf0001c3965499be0433e5dd7f35744109200` |
| v2 `SOURCE_SEAL.json` | `988c7695639f48782357ef9fff84c3aedd49cde8a4259779cd688aa63a7e3632` |
| v3 `diagnostic_run_01/RESULT.json` | `3b607c1c810d75477ccc927f23000d2649d5f7585ad65838cc07745e67698387` |
| v3 `SOURCE_SEAL.json` | `8e78f537b6015a22c8c198c4e9d07b3032022b39c314f600faa554ef56e0e5e2` |
| v4 `diagnostic_run_01/RESULT.json` | `fce3ac4458e9a7a02c605a80db09d61effcbf3f491824c0261846533b63426b2` |
| v4 `SOURCE_SEAL.json` | `b39132a81cd2283e2b27ebbe65784553c1223bcd55911dfa9b0f77903ae798f8` |
| v5 `paired_refits_run_01/RESULT.json` | `a6c7c052fb84770d8a7059a76309def35a00a71ccfa47d5e930f182cf08f3dd2` |
| v5 `reranker_run_01/RERANKER_RESULT.json` | `6a971cbc76833077156a673fa7b936a155cb186b4aeb6f392332608e5c7e77ac` |
| Committed `evaluation_rerun1/run/evaluation/report.json` (pinned by v1 and v2) | `55cae667d369abb1912101161e0468f7f88ce889295cdd21ef51531c27762be7` |
| Attempt-3 `RESULT.json` = tracked `VALIDATION_RESULT_3.json` | `e7d3f03d5770d17ea5109d1c39c2ae55927782aa1d7dbccb39577f3d46c6b48f` |
| Attempt-2 `RESULT.json` (both copies) | `df10c62b6900ec678a62cf84cb6840f2c92641284808c34b05d8b71cd3e3a133` |

Git facts: last commit touching any central-quartile folder is `0e833441` (2026-10-29 14:17, David Bayha); the dev packages began 15:38 the same day; the five commits since touch only the knee-edge study. `git log --all` for the five dev paths returns nothing.

Glossary: **BCA**, the Bayesian-calibrated actor penalty being studied. **Auxiliary update**, one optimizer step on the small helper network that predicts the width. **Simulator transition**, one environment step. **Calibration refresh**, one recomputation of the width threshold from calibration episodes (a replay when no new data is collected). **Context**, one of the two simulator conditions, ordinary current or strong current. **Screen**, a pass-or-fail rule written before the run. **Hash-pinned**, a committed file records another file's SHA-256 fingerprint, so any change or loss is detectable. **Git-ignored**, matched by `.gitignore`, so neither `git status` nor `git add` sees it.
