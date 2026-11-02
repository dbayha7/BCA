# Saved held-out decision-utility diagnostic

This stage uses saved out-of-fold predictions without any new fitting. It asks
whether the added width information improves candidate decisions compared with
the same diagnostic correction without width. All 128 development episodes
were previously examined; this is exploratory reuse, not new confirmation.

The mapping is fixed at `score = Q1 - predicted centered signed error`, with
coefficient one. The prediction target was `Q1 - V` in original return units.
An exact centered prediction therefore yields V plus an action-independent
constant. Predictions are not clipped, recentered, rescaled or selected by
outcomes. Scores use the original stable first-near-maximum tie rule, without
using returns to break ties. The two corrected scores are diagnostic signed
corrections, not calibrated uncertainty bounds or the original BCA algorithm.

All scoring predictions came from fits excluding the episode being scored,
with all candidate actions grouped. Exact saved prediction receipts, fold
membership, candidate ordering and original source/input hashes are verified.
Cross-batch transfer predictions are reported separately with the same mapping.
The primary comparison is corrected-width minus corrected-base selected value.
Raw Q, original BCA kappa 3 and scaled actor-distance kappa 0.1 are fixed secondary
descriptive references. The correction regressions used other development
episodes' simulator labels; these references did not. This is not a fair claim
of deployable algorithm superiority over those original methods.

Values are saved deterministic candidate-first continuations under the same
frozen current actor, gamma .98 and the remaining finite horizon of 100. Conditional
Monte Carlo sampling noise is absent, but the numerical values are not certified
exact oracle values. The critic's smoothed lagged-policy target can differ.

The primary mean retains every episode and every actual paired value difference.
Different selected indices define disagreement events. Conditional gain on those
events is reported separately from the all-episode mean, with their product
identity checked. Score ambiguities are reported, never excluded after seeing
outcomes. Regret is limited to the fixed candidate bank.

The 2,000-replicate primary paired bootstrap resamples episodes within the eight
batch/time cells. Its nominal interval conditions on saved fitted predictions
and omits refitting, prior analytical selection and training-seed uncertainty.
No secondary confidence intervals, coefficient sweep or method selection occur.

The runner performs only CPU arithmetic with a 60-second child timeout and one
BLAS thread. Supervisor source hashing/postchecks are outside that time limit.
Every result uses a new exclusive directory. No new prediction inference, model
fit, actor/critic/calibrator update, simulation, GPU work or confirmation access
is allowed. The original artifacts and inconclusive confirmation stay intact.

The assistant can execute this permitted local calculation; no WSL command is
needed. For later local reproduction after the synthetic preparation checks:

```powershell
& 'C:\Users\David Bayha\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B 'C:\Users\David Bayha\Documents\Codex\2026-10-22\task\unifloral_development_decision_utility_v1\run_checks.py' --execute
```

A positive result would support further development of how width informs
decisions, not immediate production use. A negative result would separate
predictive association from useful decisions. Neither result rehabilitates the
earlier failed signed propagated-error candidate or establishes Q-error coverage.
Any production redesign requires a separately named, validated candidate while
preserving native BCA and its negative findings. Full training remains on hold.
