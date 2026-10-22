# Audit record — 2026-10-21

The initial update read sources and wrote planning documents only. After new execution authorization, the bounded mechanical stage ran; mechanics/REPORT.md records it. No new training was run.

## Live local state

Repository C:\Users\David\dev\BCA:
- HEAD 30c5c68cbcfab7339ed4b69c109513e7acbb121b, branch claude/bca-mechanics.
- Untracked .claude/worktrees/, bca/, experiments/thrusterchain_diagnostics/.
- No tracked modifications reported by git status --short.
- Nested bca checkout: HEAD 0ebe03684b38b123d32461aceec2970e2df56ba1, branch claude/main-bca-connection-x4o46s; status clean.
- No applicable AGENTS.md or CLAUDE.md found in repository/workspace; a sibling project's instruction file was not applied.
- No pull, fetch, checkout, reset, merge or push was performed.
- Remote-study source/status paths are absent from the live root checkout. Completion is verified from pinned remote records, not inferred from local directory contents.

The following current source blobs match remote commit 9fe4f2acb93fe52326ba274af5d71f0fe193db65 and their file SHA256 values match the preserved source closure used by the controlled experiment:

| Path | Git blob | SHA256 |
|---|---|---|
| algorithms/td3_bc.py | c0dfe870061b96853d4fb45e2adc0b399dd429b6 | bf95a54dcb7402cca6533bcb0cb7c5ed5a383ef2e71a8c118e24b67310300bb8 |
| algorithms/td3_bc_bca.py | dbad8069bcfbf5ec49fb028606677750cc5520aa | 3f8452df67b1ac51c78b67c36a6f6269c849aa9746f5e15df771d553fce0536e |
| calibration/dose.py | e36c4d3e9e3ea58fe0c725bd8a105faf9fe23659 | 6d2fab35cb772952103adeeb6e9ca5f0a17040c0bc05217e5704c278ad617470 |

This match is scoped to these three inspected files, not the whole checkout.

## Remote v4 reconciliation

Independent read-only reviewer verified claude/shift-hosts resolves to 9fe4f2acb93fe52326ba274af5d71f0fe193db65, commit “shift studies,” timestamp 2026-10-21T14:06:46Z.

- 118 records = 20 stage-1 checks + 98 stage-2 runs.
- 98 stage-2 = 56 medium-expert +42 medium.
- 416 lanes =20 stage-1 precision +232 medium-expert +164 medium.
- Parsed committed export:98 unique supplement IDs;98/98 planned;zero excluded/unscored/halted entries.
- Existing v3 values were preserved;41 reproduction comparisons matched and77 lacked a v3 comparator. Those77 are not failed reproductions.
- V4 retains precision failures and is exploratory. Float64-adapter versus float32-deployment differences remain.

Pinned sources:
- [RESULTS_SUMMARY.md:3–10](https://github.com/dbayha7/BCA/blob/9fe4f2acb93fe52326ba274af5d71f0fe193db65/outputs/shift_hosts_v4/RESULTS_SUMMARY.md#L3-L10)
- [Lane rules, stages.py:978–1038](https://github.com/dbayha7/BCA/blob/9fe4f2acb93fe52326ba274af5d71f0fe193db65/experiments/shift_hosts/stages.py#L978-L1038)
- [V1 October16 halt, PRESPEC:913–932](https://github.com/dbayha7/BCA/blob/9fe4f2acb93fe52326ba274af5d71f0fe193db65/experiments/shift_hosts/PRESPEC.md#L913-L932)
- [V3 completed permitted scope, PRESPEC:1412–1440](https://github.com/dbayha7/BCA/blob/9fe4f2acb93fe52326ba274af5d71f0fe193db65/experiments/shift_hosts/PRESPEC.md#L1412-L1440)
- [V4 supplement, PRESPEC:1627–1660](https://github.com/dbayha7/BCA/blob/9fe4f2acb93fe52326ba274af5d71f0fe193db65/experiments/shift_hosts/PRESPEC.md#L1627-L1660)
- [Completion verification](https://github.com/dbayha7/BCA/blob/9fe4f2acb93fe52326ba274af5d71f0fe193db65/outputs/shift_hosts_v4/operations/COMPLETION_VERIFICATION_20261020.json)

Independent Option C reviewer verified the design/implementation distinction and gradient contract documented in SOURCE_AND_FIXTURES.md. Neither reviewer ran experiments or modified code.

## Library delivery attempt

Specific user consent was received for these two existing artifacts. Before retry, Library read confirmed the guide was still version1; exact-title ZIP search returned no result. No prior upload process or successful receipt existed. The earlier automatic approval denial was before process creation.

One authorized retry used the official freshly retrieved unmodified Library helper and one ordered batch. The shell process started, then exited1 with:

    library upload failed: prepare_uploads failed

The helper failed in the preparation stage before transfer/finalization. Its stdout receipt file is empty. The helper did not expose a more specific cause; do not label this as a confirmed authorization denial or transport diagnosis. No alternate upload route or further retry was attempted.

Afterward, Library read again confirmed:
- Existing guide ID libfile_a781cf619b88819189c8571ff32e611c
- Version1, filename ThrusterChain_diagnostics_results_v2.html
- Backing file file_00000000381c81f4b1d0c0e862f0d0dc
- Size11,358,393 bytes
- New ZIP exact-title search still returned no results. Search alone is not proof of absence; the pre-transfer helper failure and unchanged guide establish no completed delivery in this attempt.

The new files remain verified locally:
- presentation/ThrusterChain_diagnostics_results_v3_delivery.html
  - 11,470,144 bytes
  - SHA256 9e3ddabd4bfd6a1d3f3bf7f801c6b6f3310ab9843e722e2726212f563f86fe38
- release/ThrusterChain_action_only_evidence_20261021.zip
  - 7,395,837 bytes
  - SHA256 5fbb2ce2b553931195b9642854d07904c2c6a0a5e0e05462fe9d4bcba92d0dbb

Paths above are relative to C:\Users\David\Documents\Codex\2026-10-20\task\action_only_20261021, with verified copies under the project's action_only_20261021 directory. The existing Library guide was not replaced; no new successful file ID/version/hash receipt is available.

Prior browser QA remains unperformed because file-URL navigation was denied by browser policy and workaround use was explicitly prohibited. No browser workaround was attempted. Static HTML and the prior scientific tests remain recorded in TEST_EVIDENCE.md; they were not rerun for this document-only update.

## Stewardship

User confirmed approval of the device/AI workflow; this is not independently audited. No new permission, service, external publication or institutional ownership claim was added. Existing evidence and source records remain. The new documents are versioned by directory and hashed after writing; they are not a pre-result protocol freeze.
