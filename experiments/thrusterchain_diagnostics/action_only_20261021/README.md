# Controlled action-only calibration evaluation

This isolated post-hoc diagnostic holds one trained ThrusterChain model, its scale network, the state/time/context law, and its stochastic one-step target kernel fixed. It changes only current-action probabilities. There is no new actor, critic, or scale training.

The complete frozen 30-job study is unchanged. Its data and training protocol are not retroactively relabeled as this diagnostic's source law.

## What is compared

Fresh calibration rows are drawn from uniform sampling over 64 fixed states, followed by the original product categorical behavior law over 625 commands. Fresh evaluation rows use the same state law and the action mixture `q = (1-lambda) beta + lambda delta_grid_actor` for lambda 0, 0.5, and 1. Both methods receive exactly the same scored calibration and evaluation rows.

The unweighted method uses weights and query mass equal to one. The weighted method uses exact categorical ratios `q/beta` and the actual query's ratio as its test mass. This is the explicitly identified **per-query Bayesian WBCP variant**, not a weighted-conformal coverage theorem and not a claim about every historical importance-weighted BCA implementation.

Calibration uses the original native NumPy procedure from the preserved, authenticated training source closure. A 100,000-update predictor and target generate all fresh scores. The scale network and residual unit are frozen at their stored 95,000-update refresh. No old residual score bank or old threshold is reused.

## Frozen budget and interpretation

- 200 independent calibration replications, each with 128 rows and 1,000 posterior draws.
- 4,096 evaluation queries per mixture and replication, paired between methods.
- Nominal 90% coverage concerns the named stochastic **one-step target residual**, not true long-horizon policy-value error.
- Every replication is retained. Infinite bands, empty positive-weight banks, and numerical failures are separate outcomes.
- Infinite bands are vacuous coverage and use the native neutral multiplier 1. They do not reject an action or certify support.
- Uncertainty intervals use independent replication-level values and paired differences, conditional on this fixed model and state bank. Queries are not counted as independent experimental replications.
- Exact grid support does not imply adequate finite calibration information. Deployed off-grid actor commands receive no finite ratio or coverage claim.

`protocol.json` records the candidate protocol; `freeze.json` is the timestamped lock of that protocol, implementation hashes, preflight evidence, and all replication IDs. The protocol was specified after the earlier training results and before this diagnostic's results. It is not a prospective preregistration of the original study.

## Reproduction

On Zeus, from this directory with the original read-only assets still available:

```powershell
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B run_experiment.py run
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B analysis/aggregate_results.py
```

`run` verifies the existing freeze and reuses only matching completed replication files with verified raw checksums. It never overwrites the freeze or a completed replication. A live execution lock blocks duplicate launch. Do not remove a lock without checking its recorded process. Failed replication records remain as failure evidence.

The original execution directory is `C:\Users\David\Documents\Codex\2026-10-20\task\action_only_20261021`. If using a copied project evidence directory, preserve the frozen external source locations explicitly:

```powershell
$actionOnlyFreeze = Get-Content .\freeze.json -Raw | ConvertFrom-Json
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B run_experiment.py run --bundle $actionOnlyFreeze.bundle_root --run $actionOnlyFreeze.canonical_run_dir
```

The genuine-checkpoint tests use the original execution-directory layout. The portable evidence bundle supports records-only reaggregation without that layout; see `analysis/PORTABLE_ANALYSIS.md`. It is not a self-contained model checkpoint archive.

Initial preparation used `run_experiment.py prepare`, followed by independent review and `run_experiment.py freeze`. These commands intentionally refuse to replace an existing freeze. Alternative locations can be supplied with `--bundle` and `--run` only when making a separately identified reproduction; canonical frozen root identities remain fixed.

The analysis-only delivery bundle includes implementation, protocol, source maps, every paired replication record, machine-readable summaries, figures, and the fixed state/law bank. It excludes historical full datasets and model checkpoint binaries. Recomputing model scores needs the exact authenticated checkpoint, normalizer, original source audit, and preserved training source closure identified in `engine_identity.json`. Recomputing aggregate statistics only needs the supplied records; raw-checksum verification requires the local raw replication arrays or the bundle's recorded verification receipt.

## Advisor verification sequence

1. Authenticate the model and fixed state bank; verify horizon, time observation, terminal mask, and native target parity.
2. Enumerate the source and target categorical laws. Check normalization, support, and the expectation-one ratio identity.
3. Generate fresh paired banks using separate state, action, label-noise, and posterior random streams.
4. Run actual native weighted quantile and posterior crossings, preserving zero weights and query mass. Unit weights must reproduce the uniform procedure.
5. Convert thresholds into bands and a matched neutral-fallback multiplier. Finite usability is distinct from support certification.
6. Measure coverage, finite width, refusal, information loss, and residual alignment on independent queries. Report consistent and exceptional cases.
7. Interpret this conditional calibration diagnostic separately from policy benefit, occupancy robustness, true-Q coverage, and validation across training seeds.

See `SOURCE_MAP_ENGINE.md`, `source_map_calibration.json`, the independent review files, and the new guide section for actual code-line mappings and labeled pseudocode. The original checkout changed during the task; the preserved source closure was used and authenticated rather than assuming the current checkout was identical.
