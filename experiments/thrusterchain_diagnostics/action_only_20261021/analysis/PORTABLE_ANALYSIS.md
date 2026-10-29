# Reaggregate the saved action-only evidence

The compact evidence bundle contains the 200 saved replication JSON records, their original full-analysis hashes, the frozen protocol and state-bank identities, and analysis source. It intentionally omits the draw-level NPZ files and trained checkpoint binaries.

From the experiment folder, run:

```text
python analysis/aggregate_results.py --records-only --output-dir replay_analysis
```

This command requires SciPy for the Student-t quantiles. It verifies each saved JSON against the original full-analysis manifest, validates the sample budgets and numerator/denominator accounting, and rebuilds paired tables and conditional intervals. It does not execute the model, draw new scores, recalibrate thresholds, or train a policy.

The new receipt explicitly states `raw_checksums_checked: false`. JSON reaggregation is not independent rollout or raw-array reproduction. The original `analysis/analysis_validation.json` records the complete local validation that did check every raw NPZ checksum; this receipt is preserved. Records-only mode refuses to overwrite the original analysis directory.

On the original machine, where all raw NPZ files remain available, a full check can instead write to another output folder:

```text
python analysis/aggregate_results.py --output-dir replay_full_analysis
```

The analysis tests are independent of model execution:

```text
cd analysis
python -m unittest -v test_analysis
```

The original full aggregate and validation receipt from before the portable-mode addition are retained in `analysis/history/initial_full_analysis/`. The final full analysis was regenerated using the final script, and all mixture-level numerical results were checked to be identical to that earlier aggregate. The portable-mode replay was also checked against the final full analysis, with no numerical differences.
