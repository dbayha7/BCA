# Path A: one supervised Titan validation retry

Accounting revision 2 corrects idle-time charging before installation. The previous prepared ZIP, source, approval and manifest remain byte-for-byte preserved under `archive/before_idle_accounting_fix_20261102`; they are historical and must not be installed. The same Library item is updated, not duplicated.

This additive operational amendment preserves the frozen Path A study and the incomplete first Titan validation. It is not a new experiment, a numerical tolerance change, or an automatic retry. It implements David's approval in message `Sentinel_974cfbc5e7f08191af8f7739e30adf85`.

## Current evidence and unresolved prerequisite

The first Titan validation verified GPU identity and both dataset bindings, then last recorded `before_template_initialization` on November 2, 2026, at approximately 2:43:55 p.m. EST. PID 34106 was absent in the user's later check. No terminal receipt or model-restore pass exists. The journal's clock-adjustment message does not establish why the process stopped. No optimizer update, calibration, reset or simulator operation is present in the unchanged validation path; unreported template/inference progress remains unknown.

**Before installing or running this patch, resolve the Titan-only `path_a_a1.py` helper.** It was not found in the accessible Zeus checkout. That does not prove absence on Titan. If it exists or was used, supply its source/hash for reconciliation. Do not execute it to discover what it does. The installer requires explicit confirmation that no competing helper is present or used, and checks the isolated project for that filename. This is not a global search or distributed lock.

All other Path A targets must remain stopped. Preserve `titan_validation_1` exactly. Do not reuse the old `titan_launch.py --validate` or `--execute` commands. No production run is allowed merely because this patch installs or its CPU tests pass.

## What changes

- A separate supervisor launches the original zero-update compatibility validator as a child, captures stdout/stderr and records its exit status, signal, timeout and monotonic elapsed time.
- Attempt 2 is exclusively reserved. Failures and partial output are preserved; no automatic retry or deletion occurs.
- Production requires an authenticated attempt-2 pass, unchanged original source/data/checkpoint/numerical pins, and the same target/runtime. The original 17 remaining jobs and evaluation are reused.
- New files live only in `analysis/path_a_titan_validation_retry_v1`. No frozen original source, approval, result or checkpoint is replaced.

The parent can record a child crash or kill. Loss of the entire WSL instance or computer can still prevent a final receipt; absence of a receipt remains inconclusive and never permits an automatic rerun.

## Unchanged study and limits

Two conditions (Honeypot and missing-connector Bottleneck), native UniFloral versus original actor score `Q1 - 3 U`, unchanged BC, three paired seeds. The completed Honeypot seed-0 warmup is reused. No Path B or knee/stall confirmation work is included. Runtime success is not evidence of BCA efficacy.

The original 10,660-second remainder is not reset: previous preparation 60 seconds, validation 1 charged 120 seconds, this retry charged another 120 seconds. Production is at most **10,360 seconds minus new preparation execution**. New preparation is charged at least 60 seconds (a conservative reservation, not measured runtime), or the greater sum of measured installation/prepare/dispatch process wall time, explicit bounds on unrecorded execution overhead, and reserved receipt/exit overhead. Therefore the initial maximum remains **10,300 seconds**; greater actual or bounded execution costs reduce it further.

**Idle human time between commands does not consume the allowance.** Waiting for a response, reading results, or pausing before the next command is not GPU or execution usage. Monotonic clocks measure individual process durations only; no cross-command elapsed interval is charged. The prepare and dispatch phases each reserve five additional seconds for receipt/exit overhead, explicitly identified as bounds rather than measurements. Production's own prechecks and work are already inside its independently enforced process cap. The operator's unrecorded-overhead bound must concern execution only, excluding idle time; unknown execution overhead is not silently zero.

The retry's total process cap is 120 seconds including termination grace. It performs zero new optimizer/calibration/simulator/reset operations. Scientific work remains within cumulative caps of 190,014 critic / 95,008 actor / 190,014 auxiliary updates and 22 calibrations; evaluation is unchanged, capped at 237,600 new simulator transitions. The total study transition ceiling remains 498,400.

The nested, unchanged compatibility worker can still print the original profile's `remaining_science_seconds: 10480`. That historical field is **not** the recovery allowance. Only the new outer supervisor's reduced budget is authoritative, and it is recomputed immediately before production dispatch. Its `--execute` command starts an independent GNU timeout with the current allowance; do not wrap or replace it with the old fixed-duration launcher.

## Manual installation and validation

These commands are for Titan, **only after the helper conflict is resolved and no other Path A attempt is running**. Download and expand the small retry ZIP into a fresh temporary directory. Verify its SHA-256 against the delivered value before running code. The existing isolated dataset/checkpoint bundle must already be at `C:\Users\David Bayha\BCA-PathA`; this patch does not contain or regenerate data.

Read-only helper check in the known project locations (absence elsewhere remains an operator assertion):

```powershell
$roots = @('C:\Users\David Bayha\BCA-PathA', 'C:\Users\David Bayha\BCA')
foreach ($root in $roots) {
  if (Test-Path -LiteralPath $root) {
    Get-ChildItem -LiteralPath $root -Filter path_a_a1.py -File -Recurse |
      ForEach-Object { $_.FullName; Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256; Get-Content -LiteralPath $_.FullName }
  }
}
```

After confirming no helper was used or exists, set `$patch` to the fresh expanded patch folder and install using Titan's already observed WSL Python (no new installation):

```powershell
$patch = 'C:\Users\David Bayha\Downloads\Path_A_Titan_Validation_Retry_v1'
$manifest = (Get-FileHash -LiteralPath "$patch\RETRY_MANIFEST.json" -Algorithm SHA256).Hash.ToLowerInvariant()
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B '/mnt/c/Users/David Bayha/Downloads/Path_A_Titan_Validation_Retry_v1/install_retry.py' --repository '/mnt/c/Users/David Bayha/BCA-PathA' --manifest-sha256 $manifest --confirm-helper-absent-not-used
```

Use `/home/dbayha/bca-gpu/bin/python` in Ubuntu for the retry; package/numerical requirements are unchanged. Preparation and validation commands, followed by a conditional production command, are defined in `RUN_COMMANDS.md`. Do not run any production command without an authenticated pass and positive remaining allowance. The source and command tests are CPU preparation evidence only, not a Titan runtime pass.

## Scope and skills

Repository skills actually read: `.agents/skills/offline-rl-experiment-design/SKILL.md` and `.agents/skills/bca-code-and-calibration-audit/SKILL.md`. Applied checks: immutable method/data identity, paired design unchanged, no extra scientific operations, consumed and unknown work retained, failure versus scientific outcome distinguished. Independent review covers the actual added code. Separate external Claude agents have not been contacted or claimed aligned.

## Preserved evidence

Original Titan manifest: `32718856c6b4714ea1a4004554d61b80223d502e3517b5563c372a97047a15ed`.
Original approval: `4fc24a5bb025d54372cdd14736cd24894b4053516e4393130bb4e8712dd3a156`.
The new manifest binds only the additive patch; the new approval records the already granted single-retry authorization. Installation pins both first-attempt files. Final verification/results and exact new hashes are supplied with the patch.
