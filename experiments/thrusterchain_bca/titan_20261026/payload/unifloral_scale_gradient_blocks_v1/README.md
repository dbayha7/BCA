# Exhaustive fitting-block gradient audit

Extends the passing first-32-row diagnostic to all 256 stored fitting rows of the same frozen native UniFloral checkpoint. The first block is recomputed as a reproducibility anchor; the remaining seven blocks extend coverage. No training, fitting, calibration refresh, simulator rollout, candidate-bank generation, development return-label access or confirmation-data access occurs.

The fixed penalty grid is {0, .03, .1, .3, 1, 3, 10}. The hypothetical contrast factor is frozen at 31.372587203979492, from the earlier passing gradient-route result. It is neither recalculated per block nor installed in production. Native Q normalization and BC remain unchanged.

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_scale_gradient_blocks_v1/run_checks.py" --execute --allow-outside-upstream-constraints
```

The existing verified GPU/highest-precision runtime is required. The worker has a 300-second limit and writes to a new timestamped result directory. The launcher defaults to file/provenance preflight without `--execute`; it never installs packages. WSL execution remains manual because the earlier tool-access denial is binding.

Reports separate per-block gradient ratios, per-action ratios, directional cancellation, and the derived full-256-row objective. The latter averages raw component gradient trees before applying the global native coefficient; averaging normalized block gradients would be a different objective.

The local preparation tests verify row coverage, weighting, missing-value handling, and aggregation failure modes with the standard library only. GPU checks remain pending until the command completes. A pass validates coverage and implementation identities; block heterogeneity is descriptive and cannot select a coefficient or prove BCA efficacy.

After these results are inspected, the next scientific step is a fixed development comparison of action-selection usefulness against raw Q and simple controls. Any choice must be locked before untouched confirmation. This launcher does not perform that later stage or resume full training.
