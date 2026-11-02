# I0 platform rehearsal (run this in WSL before approving attempt 3)

**What it is for.** The native screen's first real unit, g1, opens with the instrument check I0: the exact median
rho(s10) over the 96 gated anchors, recomputed in the run's own environment, must equal the formal G2 report's value
(computed on Windows) to 1e-9 relative, and the H1 identity must hold. Attempts 1 and 2 showed the recorded Linux
environment does not reproduce a Windows reset bit for bit (one ULP, episode 16). If that difference reaches rho, I0
fails, g1 reads INCONCLUSIVE, and attempt 3, the last one the rule allows, is spent before any training.

This script runs, outside any attempt, every exact Windows-versus-Linux comparison the screen makes before training:
the g1 unit's instrument path (I0), the amendment A1 dataset loader for both archives, and the data-plan
re-derivation every training and evaluation job performs first (`stage1_worker.load_world`, which refuses if the plan
rebuilt in the job differs from the plan `prepare.py` recorded on Windows). It consumes no attempt, trains nothing,
opens no checkpoint, and writes only its own receipt here. The sealed package is untouched (its manifest does not
cover this folder).

**Windows result (2026-11-01, `receipt_20261101T191553Z_windows.json`):** I0 would pass; every recomputed statistic
equals the formal report exactly (relative difference 0.0); both archives load through the A1 loader with the formal
digests; both worlds' data plans re-derive equal for both arms; 201,600 exact simulator steps; under a minute.

## Run in WSL (PowerShell, from `C:\Users\David\dev\BCA`), about a minute, CPU only

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/i0_platform_check_v1/i0_check.py"
```

It prints a JSON summary and writes `receipt_<utc>_linux.json`. Paste the summary.

## What each outcome means

| Verdict | Meaning | Then |
|---|---|---|
| `I0 WOULD PASS on this platform` with `max_rel_diff_any_reported_statistic` 0.0 | Linux reproduces the Windows exact values bit for bit at the anchors | approve attempt 3 and launch (native_screen/README.md) |
| `I0 WOULD PASS` with a small nonzero difference (under 1e-9) | platform drift exists but stays inside I0's tolerance | approve and launch; the receipt records the drift |
| `I0 WOULD FAIL on this platform` | the one-ULP reset difference reaches rho, or H1 breaks on Linux | do not approve attempt 3; I0 needs an amendment (for example, a Linux-computed reference recorded under its own rule) before the last attempt is spent |
| `DATA-PLAN RE-DERIVATION DIFFERS` | the plan rebuilt on Linux differs from the Windows record | do not approve attempt 3; the first warmup job would refuse; `prepare.py` would have to be rerun in WSL and the package re-sealed |

## Files

| File | What it is |
|---|---|
| `i0_check.py` | the rehearsal (read-only; imports the sealed screen's own modules) |
| `test_i0_check.py` | tests of the decision function (Windows, 4 tests) |
| `receipt_*_windows.json`, `receipt_*_linux.json` | the receipts, one per run |
