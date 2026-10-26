# Controlled coverage and terminal calibration

These modules implement a controlled terminal diagnostic. They do not turn the
nine cells into nine trained offline-RL benchmark results. Trajectory datasets,
controller training, and closed-loop evaluation remain separate benchmark work.
The smoke uses small terminal reward regressors and explicitly claims no neural
Q posterior, adaptive-policy coverage, or known closed-loop state occupancy.

## Exact query law

The physical state has 13 coordinates. Its containing box is
`[-.5,.5]^10 x [.1,1.1] x [-.2,.2]^2`. Every point is reset-valid: the float
radius is at most `sqrt(10)/2`, current lies in the canonical range, and wave
amplitude is at most `sqrt(.08)`. This is a directly sampled box, without a
nonlinear coordinate transformation or an unaccounted Jacobian.

The state gap is `C={x[0]>0}`, equivalently normalized first coordinate `z1>0`.
The action box is `[-1,1]^4`; its gap is `B={a[0]>.5}`. Equality belongs to the
lower region. Outer box endpoints are included. This rectangular diagnostic gap
is distinct from the physical propeller-stall region.

| Level | State gap probability | Action gap probability |
|---|---:|---:|
| broad | .5 | .25 |
| rare | .05 | .01 |
| absent | 0 | 0 |

State and action draws are independent, producing all nine combinations. The
target is fixed broad/broad. Conditional densities are uniform in each region.
The state box volume is `.16`; its two region volumes are `.08`. The action
region volumes are `12` and `4`. For state-gap probability `p_s` and action-gap
probability `p_a`, the four source masses and target masses are:

| Region | Source probability | Target probability |
|---|---:|---:|
| lower state, lower action | `(1-p_s)(1-p_a)` | .375 |
| state gap, lower action | `p_s(1-p_a)` | .375 |
| lower state, action gap | `(1-p_s)p_a` | .125 |
| joint gap | `p_s p_a` | .125 |

The rare/rare joint probability is `.0005`; its joint-gap density ratio is
`.125/.0005=250`. Tests integrate the piecewise density and the weighted source
mass analytically over these four rectangles. They also test endpoints, invalid
shapes, NaN/Inf, reset validity, and reproducible sampling.

`CoverageLaw.density_ratio` returns target/source without weight clipping.
`ratio_report` distinguishes `unsupported_target` (positive target, zero source,
ratio `+inf`) from `outside_both_laws` (both zero, represented by ratio zero with
its separate flag). An absent state gap leaves target mass `.5` unsupported;
an absent action gap leaves `.25`; both absent leave `.625`. A tiny positive
missing target mass still means unsupported. Configurable rare masses must be
scalar and at least `sqrt(float64.tiny)` so positive joint densities remain
representable; malformed and unrepresentably tiny configurations are rejected.

The density measure is physical-state13 and continuous-requested-command4 Lebesgue measure,
with a counting-measure atom at the fixed terminal time `t=99`. Numerical sampling
uses float64. These density formulas must not be silently reused as probability
masses for a quantized float32 deployment law. The benchmark environment's
`step` casts requested commands to float32. The diagnostic applies that same
float32 execution map before calling the shared dynamics. Its response is the
actual environment reward, while its covariate and density remain the continuous
requested command. A deterministic quantized response does not require an
executed-action density.

A known behavior action policy does not identify the closed-loop state
occupancy. `occupancy_accounting` reports unknown or estimated occupancy and
never grants an exact joint-weight claim from a known action policy alone.

## Groups, sample counts, and freshness

`grouped_split` assigns whole independent roots to fit, selection, tuning
calibration, final calibration, and evaluation roles. It never splits an episode
by transition. Numeric IDs must be finite; all IDs must form one-dimensional
vectors. Disjoint groups prevent leakage, but do not make dependent rows inside
a group IID. Final calibration therefore requires one terminal query per
independent root. The bootstrap gives each fit root one Dirichlet weight and
divides that root's weight among its rows.

The smoke fits candidate models on fit roots, chooses the ridge parameter using
selection roots, and records a separate tuning-calibration score diagnostic.
Only then does it generate fresh final-calibration roots. Final target queries
use another independent draw. Final-calibration labels select neither models nor
arms. The same final calibration is reused for the fixed, preregistered four-arm
comparison; no arm is selected from these final results.

`plan_binomial_count(k,p,delta)` finds the smallest independent-root count `N`
for which `P(Binomial(N,p)<k)<=delta`, or reports impossible absent support or an
exceeded sample budget. For example, obtaining at least nine rare/rare joint-gap
roots with 95% probability requires **28,867 roots**, not merely the expected
count calculation `9/.0005`. This count plan is not a weighted-rank or ESS
guarantee. The default smoke has only 128 final-calibration roots per cell and
repetition, deliberately exposing sparse-count failures.

## Terminal labels and one-sided bounds

The shared nonlinear canonical plant remains unchanged. At `t=99` of its
100-step finite objective, continuation is zero and
`Q_99(x,a)=r(x,a)=-(sum(x[:10]**2)+.3*sum(clip(float64(float32(a)),-1,1)**2))`.
`terminal_targets` calls the canonical `dynamics` reward instead of implementing
another plant, after the environment's action cast. Requests that overflow
float32 are rejected just as they are by the environment. It rejects earlier
times, another horizon, and the continuing
time-limit objective. The raw source environment ends with `truncated=True`;
the finite-task adapter maps this boundary to termination. A truncation in a
continuing objective does not justify terminal-Q labels.

The named **terminal Bayesian-bootstrap diagnostic** fits a small ensemble of
terminal reward estimators using independent `Dirichlet(1,...,1)` draws over fit
roots. It saves the draws, root IDs, member diversity, raw prediction spread,
scale-floor frequency, and collapse status. The model uses deliberately
misspecified absolute-value features and ridge regression. Its spread is a
scale heuristic, not a Bayesian posterior over neural or long-horizon Q.

For frozen ensemble mean `Qhat` and `sigma=max(member_std,epsilon)`, the final
scores are signed: `z_i=(Qhat_i-r_i)/sigma_i`. Given exact density ratios `w_i`
and query weight `w_*`, the quantile distribution is
`sum_i w_i/(sum_j w_j+w_*) delta(z_i) + w_*/(sum_j w_j+w_*) delta(+infinity)`.
The lower bound is `Qhat-max(0,q)*sigma`. The zero clamp conservatively enlarges
the prediction set when the signed quantile is negative. The target-weighted
test atom must remain in the denominator and at infinity. This construction is
based on the weighted conformal covariate-shift result in
[Tibshirani et al., NeurIPS 2019](https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf).
Its marginal coverage prerequisites include independent source calibration,
an independent target query, unchanged conditional labels, exact ratios,
absolute continuity, and a score function fixed before final calibration.

The low-level `weighted_conformal_quantile` supports signed scores. The additional
two-sided utility `weighted_conformal_radius` requires nonnegative scores, and
`final_terminal_calibration` builds absolute-residual intervals. Both preserve
the test atom. Exact toy tests verify rank boundaries, signed scores, ties/zero
weights, common weight scaling, and the infinite outcome. At a numerical rank
boundary, an exact rational comparison using the original float weights and
decimal alpha resolves the endpoint. A floating-point tolerance only triggers
that exact calculation; it never grants acceptance. This prevents both
conservative endpoint skips and permissive tolerance errors.

Three different outcomes remain separate:

1. **Support refusal:** the specified target is not absolutely continuous with
   respect to the source. No valid bound is issued; missing bounds are `NaN`,
   status `support_refused`, and coverage is unreported. Global target refusal
   also applies to locally supported points in an unsupported cell.
2. **Infinite rank:** source support exists but the `(1-alpha)` quantile reaches
   the test atom. The lower bound is `-inf` (or two-sided interval is all real
   numbers). Its coverage is vacuous and counted explicitly, not dropped.
3. **ESS gate:** `(sum w)^2/sum(w^2)` is an optional usability diagnostic. A gate
   failure never changes the quantile and is neither proof of absent support
   nor a coverage theorem. The default smoke threshold is 20.

Estimated-weight annotations cannot enable the guarantee flag. The controlled
smoke always sets `guarantee_claimed=False`: it is an empirical mechanism check.
The generic final-calibration API additionally rejects shared prior/final
groups and requires explicit frozen-model, frozen-target, and independence
conditions before any requested guarantee flag can be activated.
The terminal bootstrap factory automatically includes its known fitting-root
IDs in the leakage check, even if a caller omits prior provenance. Frozen model
and calibration arrays are backed by immutable bytes, so callers cannot
re-enable NumPy writes; protocol mappings are read-only.

## Four arms and repeated smoke

`nine_cell_smoke()` returns a strict-JSON-compatible dictionary and writes no
files. Its current schema is `thrusterchain-terminal-nine-cell-smoke-v3`.
Defaults are three independent repetitions, 16 bootstrap members,
128 final-calibration queries, and 64 target evaluation queries per cell.
It executes `unweighted_standard`, `unweighted_most_caution`,
`weighted_standard`, and `weighted_most_caution`.

Coverage is evaluated on independent broad/broad state-action queries. A separate
paired action-bank demo shares each target state and eight candidate actions
across the four arms. The bank includes zero, all +1, all -1, and five random
commands. If every candidate has a finite bound and passes the ESS gate, the arm
selects the maximum lower bound. Otherwise the preregistered standard fallback
executes the host `Qhat` action-bank argmax, while most-caution executes zero.
No fallback creates a missing bound or turns an infinite bound finite.

**Zero is the known terminal optimum for this reward.** It is a deliberately
privileged task-structural fallback. Its performance cannot demonstrate policy
learning superiority or usefulness for a nonterminal controller. The action-bank
law differs from the broad continuous target law; selected actions have no
adaptive-coverage guarantee.

The archived `results/smoke_v2/diagnostic_results.json` makes this limitation
concrete: all **6,912 arm-level choices** selected the zero command and had
exactly zero recorded and independently recomputed terminal regret. The count
is `9 cells * 3 repetitions * 64 target states * 4 arms`; each arm contributes
1,728 choices. There are only 1,728 shared state cases, not 6,912 independent
states. The same zero choice occurs under both accepted decisions and fallback
events, and all four arms have identical selected rewards on each shared state.
The [claim-verification receipt](../audits/revision_v2_claims/claims_receipt.json)
pins the raw file hash and records counts for every cell, repetition, and arm.

Consequently this run is a **mechanics, coverage, and refusal diagnostic**. It
cannot show an uncertainty-guided decision benefit, a BCA advantage, or improved
control from calibrated uncertainty. Its exact target remains terminal
`Q_99=r`, horizon 100, zero continuation, with fixed broad/broad IID coverage
queries and the separately specified paired action bank. The continuous
requested-action density, float32 execution map, independence and support
assumptions above are unchanged. No new reward, action bank, target, or tuned
replacement experiment is substituted to produce a favorable comparison.
Any nontrivial decision experiment must be separately specified as future work,
including its objective, time/horizon, candidate law, information budget,
selection protocol, and applicable coverage assumptions before evaluation.

Reports retain support refusals, infinite bounds, ESS failures, fallback-event
counts, full-population rewards, accepted-only rewards, and a common-accepted
comparison across all four arms. Unweighted query coverage in shifted cells is
only empirical. Infinite bounds remain in issued-bound coverage, with their
count reported separately. Unsupported cells report null coverage.

Repeated-run summaries report between-repetition standard deviation and Monte
Carlo standard error for mean query coverage. They do not pool all queries into
a binomial confidence interval: queries within a repetition share a fitted
model and calibration sample. Three tiny repetitions are not a precise coverage
study, and these outputs are not nine trained benchmark results.

Every repetition retains raw development, final-calibration, target-query, and
paired-action-bank records. These include requested and executed actions,
physical states, group IDs, role indices, time 99, region membership, true
terminal reward/Q, exact source/target densities and weights, signed scores,
means/scales, lower bounds, quantiles, atom masses, status and usable masks,
selected actions/rewards, fallback events, and common-accepted masks. Model
coefficients and their SHA-256 fingerprints are saved alongside root-weight
draws. Candidate seeds and selection predictions permit selection checks.
The returned `statistical_recipe` defines the fit, quantile, selection, coverage,
reward, and between-repetition uncertainty calculations.

Raw `null` means an absent/NaN value; `"Infinity"` and `"-Infinity"` preserve
infinite bounds or ratios in strict JSON. `covered_when_issued` is 1 or 0 for an
issued bound and null for a refused/outside-target query. Status arrays, rather
than a finite-only filter, determine coverage denominators. Raw records allow
all reported proportions and reward means to be recomputed without rerunning
model fitting.

## Verification

Run the focused tests with `python -m unittest discover -s tests -p test_coverage.py`
and `python -m unittest discover -s tests -p test_calibration.py` in an environment
with the package's NumPy/Gymnasium dependencies available. The tests cover
normalization, exact ratios and missing support, grouped leakage, minimal count
planning, terminal objective semantics, test atoms, infinite intervals, ESS,
freshness, signed lower bounds, independent root weights, duplicate-root
invariance, predictor mutation, strict JSON output, and all four executed arms.
