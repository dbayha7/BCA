# Run the zero-slab replacement gate on Zeus

Status: **superseded, never frozen; do not launch.** See `SUPERSEDED.md`. The central-quartile gate (`../conditional_ood_gate_central_quartile_v1/`) is the replacement that was approved, run and passed. This folder screened the first thruster's commands within 0.1 of
zero at load at least 0.4. It is kept as a record only.

## What changed from the shelved gate

1. **The band moved to zero thrust.** It sits where the saved actor operates and leaves the behavior-cloning target
   unchanged.
2. **The band is wider.** The half-width is 0.1, from the options memo's value model.
3. **The support rule changed.** The fixed RMS radius of 0.5 could not pass in 13 dimensions. It is replaced by the 64
   nearest neighbors and a training-only reference quantile.
4. **The anchors are new.** They are the 64 development episodes not used by attempt 2.

Every other threshold, the budget and the evaluator code paths are unchanged from attempt 2. A worker startup check
refuses to run if the geometry constants differ from the frozen contract.

## Launch

From PowerShell on Zeus, run this once:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/conditional_ood_gate_zero_slab_v1/launch.py --execute
```

For a metadata check that reserves nothing, use `--preflight` instead of `--execute`. That check does not test CUDA or
model execution.

**Limits.**
- **Ceiling:** 900 seconds and 28,104 simulator transitions.
- **Training:** zero updates of any kind.

**Where results go.**
- **Progress:** printed while running and saved to `run/progress.json`.
- **Accounting:** `run/RESULT.json`.
- **Decisions and measurements:** `run/report.json`.

An interruption keeps the reservation and all evidence. Do not delete `run/` to repeat an attempt.

## How it was checked before freezing

- **CPU tests.** 44 tests pass on Windows with synthetic arrays and toy dynamics. They cover the new region
  boundaries in float32, the candidate bank, the free-rollout hole test, the reference quantile, the locality rule and
  the contract-constant check.
- **Synthetic timing.** On arrays of the real size (76,800 rows, 64 anchors), the support computation took about 2
  seconds.
- **Synthetic pass rates.** In-distribution synthetic anchors passed locality about 95% to 100% of the time, and the
  band-count criteria about 97%. These are design checks, not measurements of the real data.
- **What stays unchecked.** CUDA, model inference and the real-data phases run only inside the capped worker.
