# Saved-development mechanism audit

This is an exploratory analysis of two already observed development batches.
It reads no confirmation files, generates no new states or returns, runs no GPU
inference, and performs no optimizer/calibration updates. The completed locked
confirmation remains inconclusive and is not reanalyzed or expanded here.

Definitions were frozen in `protocol.json` and `DEFINITIONS_FROZEN.json` before
this audit loaded further detailed development outcomes. A subsequent methodology
review added Q-margin adjustment and decision-event decomposition through
`PROTOCOL_AMENDMENT_01.json`, also frozen before detailed input loading in
`AMENDMENT_FROZEN.json`. Both versions are preserved. This timing does not make
the audit independent confirmation: aggregate findings were already known.

## Questions

1. Does larger residual width identify larger **signed Q overestimation** among
   actions at the same state? Signed error, absolute error, and positive-part
   optimism are separate outcomes.
2. Does width retain a within-state association after linear adjustment for
   actor distance, and after adjustment for both actor distance and Q margins?
3. When raw Q chooses a suboptimal candidate, does width penalize that choice
   more than the better candidate? The corresponding actor-distance contrast is
   reported alongside it.
4. Do the already fixed BCA and actor-distance selectors actually choose
   different actions? How often, and with what conditional versus unconditional
   continuation-value gain?
5. Is the candidate bank informative enough to expose mistakes, and what does
   development-only variance imply for illustrative precision planning?

The score settings remain exactly raw Q, raw BCA kappa 3, and the actor-distance
control with kappa 0.1 and fixed factor 19.29619572569095. They are reconstructed
only to describe event frequency and variance. There is no new strength sweep,
new fitted selector, scale refit, or winner selection.

## Correct interpretation

For each candidate, the target is its saved discounted continuation return:
take that action first, then follow the same frozen deterministic actor for the
remaining finite horizon. These are deterministic simulator values, so there is
no conditional Monte Carlo sampling noise. They are not exact real arithmetic
or certified oracle values. The task uses gamma .98, horizon 100, the original
reward/action scaling, full simulator state, and no terminal bootstrap.

The critic may target a smoothed lagged policy rather than the current frozen
actor. Therefore `Q1 - saved continuation value` measures discrepancy against
this declared target, not a uniquely attributable optimization error. Were the
saved inputs stochastic single-rollout returns, this audit would reject them;
their absolute errors could otherwise mostly measure return noise.

Each episode/state receives equal weight. Candidate actions and action pairs
within a state are dependent. Undefined correlations are reported explicitly,
including constant signals and rank-deficient/near-zero partial residuals. The
common-defined subset is reported when comparing marginal and partial results.
Partial correlations remove only the stated linear effects; they are neither
causal tests nor proof of all information beyond the controls. No predictive
model is trained. Any future learned diagnostic would require validation by
held-out episode/state groups.

The raw-Q-versus-better-candidate comparison is deliberately conditioned on a
mistake. Its error ordering is partly constrained by construction. A favorable
width contrast only means the penalty reduces that particular score gap; it
does not establish calibrated Q error, correction at the current coefficient,
or improvement over all feasible actions. Near-tied best candidates are
reported separately, and the representative is never chosen using width.

For fixed BCA versus actor-distance, the unconditional gain equals the frequency
of differing decisions multiplied by the mean gain on those differing decisions.
The latter must not be presented as the overall benefit. Same-action cases have
zero paired gain by construction and remain in the unconditional denominator.

The endpoint-relative numerical tie bands are fixed before analysis. Their
classification is not exactly invariant to arbitrary offsets or unit changes;
correlation algebra and numerical tie classifications are distinct checks.

All outputs are descriptive and exploratory. There are no new significance
tests or confidence claims. Illustrative sample counts use development paired
variance and normal approximations for fixed effect targets 0.05, 0.1, and 0.2
discounted-return units. These are not chosen practical thresholds, power
guarantees, or recommendations to collect more data. Sparse decision changes and
one shared checkpoint limit extrapolation. A practical effect threshold and a
new independent plan are prerequisites for any future sample-size commitment.

## Execution

`analyze_saved.py --preflight` verifies only the exact allowed input files,
definition receipts, schemas, and metadata. `analyze_saved.py --execute` runs the
same read-only saved-data calculation in a CPU child with a 60-second timeout and
writes a new exclusive output directory. It does not launch WSL, a simulator,
or a training runner. The assistant can perform this permitted local analysis;
the user does not need to run another GPU command.

The authoritative results are in the newly created `results_.../report.json`
and analysis artifact. Inputs and the fixed checkpoint are never overwritten.
Preparation tests use synthetic fixtures. See `PREPARATION_RECEIPT.json` for
the actual test count and source hashes.

Full training remains on hold. This audit can identify a mechanism problem or
an information/precision limitation; it cannot establish BCA efficacy. Any
revised method must be developed without confirmation rows and evaluated under
a new fixed plan on genuinely fresh data.
