# Weighted conformal development reference

This implements actual target/source weighting of the **held-out conformal threshold**, including the new query's weight at infinity. It is separate from existing scale-fitting IW and is not connected to the four training hosts yet.

Read [the design and remaining integration gates](../../docs/WEIGHTED_CONFORMAL_DESIGN.md). Existing host code, scientific manifests, default configurations and results are preserved. This directory imports no hosts, loads no checkpoints, trains no models and steps no simulator.

## Run the mathematical checks

Python with NumPy is enough for the unit checks:

```bash
python experiments/weighted_conformal/test_reference.py
```

The predeclared synthetic validation also needs SciPy and PyYAML, already in the BCA environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/weighted_conformal/validate.py \
  --output outputs/weighted_conformal/new-validation
```

The output directory must not exist. Every YAML setting is commented. Source hashes and the protocol are saved before outcomes; actual test exit, trial-level inputs, known ratios, RNG identities, intervals and coverage counts are retained. `synthetic_panels.json.gz` is ordinary gzip JSON, with `null` radii explicitly representing positive infinity. Bootstrap draws are reproducible from the pinned PCG64 seed/stream/trial and NumPy version. Each trial uses independent calibration data and one independent target query.

## API

```python
ratios = density_ratios(source_density_at_calibration, target_density_at_calibration)
scores = normalized_scores(targets, frozen_centers, frozen_positive_scales)
reference = build_reference(scores, ratios, supplied_exponential_draws,
                            alpha=0.1, credibility=0.95)
query_ratios = density_ratios(source_density_at_queries, target_density_at_queries)
intervals = reference.intervals(query_centers, query_scales, query_ratios)
```

Density ratios for calibration and query points must share one common scale. Never normalize the two arrays separately. Never substitute policy density alone, advantage, or affinity without establishing the corresponding source/target law. Passing arrays does not establish support or ratio accuracy globally; the function can reject only violations visible in its inputs.

The conformal floor uses exact rational mass comparisons for the supplied float64 weights and decimal alpha. The Bayesian component uses explicitly supplied exponential bootstrap draws tilted by calibration ratios on observed scores. Their maximum preserves the conformal floor, including infinity. It does not certify the Bayesian posterior risk claim or correct estimated ratios, dependent trajectories, or adaptive bank reuse.

This is a CPU correctness reference, not a JIT training implementation. Successful synthetic coverage does not establish OOD action safety, true-Q accuracy, policy improvement, or completion of the intended IW-BCA training method.
