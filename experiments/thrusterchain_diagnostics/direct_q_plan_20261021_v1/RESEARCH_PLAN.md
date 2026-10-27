# Direct action-value penalty research plan v1

Status: later pilot proposed, not run. The isolated simple-objective mechanical stage was subsequently authorized and completed; see mechanics/REPORT.md. Created after inspecting existing results. Preserve all earlier numerical evidence.

## Objective and meaning

BCA should supply an adaptive conservative penalty for candidate actions. The immediate research question is whether this penalty changes the action-value score in a useful direction when actions are outside the observed data distribution.

In TD3+BC, the actor maps observations directly to actions at deployment. The critic guides actor training; it is not queried by the deployed actor at every decision. Subtracting uncertainty from the actor's training score changes the effective objective, not the stored critic predictions. If the aim is to change the critic's raw predictions themselves, that requires a separate critic-loss or target intervention.

A one-step residual band is not a confidence interval for the full future return. Treat its width as a proposed action penalty whose utility must be measured, not assumed.

| Placement | What changes | What must retain a gradient |
|---|---|---|
| Actor score Q_safe(s,a)=Q1(s,a)-kappa U(s,a) | Actor learning objective; critic fitting stays unchanged | Q and U with respect to actor action; model parameters may be frozen |
| Critic-loss regularizer | Fitted critic parameters | The regularizer with respect to critic parameters; a detached additive scalar has no effect |
| Bootstrap-target subtraction | Future regression labels and ultimately the critic's fixed point | The target may correctly be detached and still change critic regression labels |
| Current native BCA | Weight on squared distance from logged actions | BC distance with respect to actor action; detached multiplier still has an effect |

## What already exists

Native BCA reads its frozen uncertainty scale at the logged action, maps width to a bounded detached multiplier, and multiplies the BC loss. It does not subtract width from Q1. The current actor denominator uses raw Q1, and its actor loss is -c mean(Q1)+BC.

The proposed Option C at commit 743eba669e368bdd9241738d849a7194b6d648b3 is explicitly design-only. Its penalty is the positive difference between candidate and logged-action widths when usable, and a positive Q advantage over the logged action when refused. It uses a separate noisy band-query action, raw unsmoothed Q1 for actor value, and BC coefficient one. A throughput benchmark in that folder changes both action smoothing and normalization and is not faithful implementation evidence.

Remote shift-hosts v4 is complete: 118 records and 416 evaluation lanes. Those counts are not 416 independent training replicates. Its TD3+BC/ReBRAC cloning multipliers, IQL likelihood weights, and CQL critic penalty are different placements. Its completion does not establish direct candidate-action Q-penalty benefit. Earlier precision caveats and exploratory status remain.

## Gate 1: mechanism before outcomes

Implement only in a new isolated directory. First run the deterministic fixtures in SOURCE_AND_FIXTURES.md:

- Baseline identity, zero coefficient, fully detached penalty, constant-offset placebo.
- Frozen critic/calibrator parameters with live action derivatives.
- Correct action query, raw-Q normalization, separate random streams.
- Invalid/infinite width handling before arithmetic, support/refusal bookkeeping, finite gradients.
- For Option C separately: paired hinge and refusal cancellation, clipping and threshold lookup boundaries.

No rollouts or training are needed to diagnose an actor-gradient no-op. A material failure stops the affected candidate, records the failing input, and leaves unrelated controls intact. A repaired candidate gets a new version; original failures remain.

## Gate 2: smallest proposed direct-Q pilot

This is a conditional mechanism pilot, not a new trained-seed efficacy study. Use the already selected full-retention BCA replicate 01 terminal 100k checkpoint. Its SHA256 is 4088278230fc413c509e27e189daa342468cfc4fb6d2479e3e8c29e149dac3d8. Keep its actor, critic, target networks, calibrator reference, preprocessing, remaining-time observation, horizon 100 and discount 1.0. Import the authenticated preserved source closure, not a changed live checkout.

Start with the simpler actor-score candidate below. This is a deliberate new candidate, not Option C:

    a = actor_theta(s)
    c = alpha_host / stop_gradient(mean(abs(Q1(s,a))))
    U = frozen_R * frozen_unit * max(eta_frozen(s,a), 1e-6)
    L = -c * mean(Q1(s,a) - kappa * U) + mean((a-a_logged)^2)

Use kappa=1 and alpha_host=2.5; hold them fixed. Use the host's raw-Q normalizer unchanged. Stop if it is zero/nonfinite; do not silently add a stabilizer. Freeze R, unit, and calibrator parameters while keeping the action-input derivative of eta. This width is a residual-derived heuristic at actor actions. It is not a certified Q lower bound.

The preserved reference is the historical 95k scale/reference used by the previous frozen-model study. Record its exact identity and whether it is finite before applying the candidate. If the scalar reference is unusable, report the learned candidate as refused and run valid controls; do not substitute a new favorable reference. Never evaluate zero times infinity. Distinguish a finite numerical width from statistical support.

Proposed design (freeze only after Gate 1 and before outcome evaluation):

- Use the existing 64-row state bank, SHA256 2001f5444bb004063dce61e8f05d65defac70ecd2a05ac43385cb0cacc0bb63e. Validate the full hash.
- Within each of the four regime/load contexts, allocate four rows to adaptation and twelve to evaluation by seeded row IDs, independent of outcomes. Total: 16 adaptation states and 48 evaluation states. Preserve and report the actual remaining-time counts. This is a split of an already studied bank, not a fresh confirmatory population.
- Create eight paired adaptation batches, each with four independently sampled categorical behavior anchors per adaptation state: 64 rows per batch. Use the original known beta probabilities only for drawing anchors; do not claim ratios for continuous actor actions.
- All five arms start each probe from the identical original actor and optimizer state. Make exactly one actor update per arm per batch on disposable copies, then reset. Freeze the critic and target networks. Maximum 40 isolated actor steps; no sequential learning, no critic updates, no overwriting canonical checkpoints.
- Draw source/anchor randomness from NumPy PCG64 SeedSequence([2026102121, probe_id, stream_id]); split stream 1, anchors stream 2. Record all arrays and their hashes before outcome scoring.
- Evaluate the original actor and all 40 probe actors from the same 48 held-out designed states. At most 196,800 simulator transitions using the loose upper bound 41*48*100. Remaining horizons are respected. No environment-driven policy selection.
- Runtime must be estimated from a small fixed fixture before execution. This is a proposed cap, not an authorized launched run.

Arms:

| Arm | Actor-score penalty | Purpose |
|---|---|---|
| Host | 0 | Baseline |
| Direct width | kappa U(s,a) | Candidate adaptive action-value penalty |
| Matched constant strength | k_const D(s,a), D=mean_action((a-a_logged)^2) | Constant coefficient on an action-varying caution template |
| Detached width | kappa stop_gradient(U(s,a)) | Detect whether an apparent benefit comes from an unintended path |
| Constant additive offset | kappa u_const | Placebo; no intended actor gradient under raw-Q normalization |

Set k_const once using only the adaptation states and first paired anchor batch, before any oracle outcome: match the RMS norm of the action-gradient contribution c*kappa*grad_a U to c*k_const*grad_a D. If the denominator is zero/nonfinite, record the control as unmatched; do not retune it using outcomes. Set u_const to the mean finite adaptation width, or explicitly refuse the placebo if undefined. Keep these values fixed for all probes. This matches per-row action-gradient RMS, not the aggregate actor-parameter gradient or realized update; repeated-state anchor terms can cancel. Record parameter-gradient norms and actual action displacement too. Different directions and BC-template geometry remain a limitation.

Primary evidence: exact gradient/control identities; change in raw Q1 and penalized score; change in actor action; actual return difference from paired states; increase/decrease in return separated by stall/monotone and load; numerical optimal-regret bounds using the existing per-regime QP reference. Record all rows, including failure/refusal. QP bounds are numerical certificates conditional on the model and start state, not population means.

For the original actor and the five arms from probe0 only, at all48 held-out states, compare Q1 to the actual finite-horizon value of taking that action then following the original frozen policy. This fixed subset adds at most6*48*100=28,800 transitions, giving a total loose cap of225,600. Keep this Q^pi target distinct from Q*, and from the native minimum-of-two-head residual predictor. Action rankings must use the same continuation target across actions. Report overestimation, ranking errors, and decision harm/opportunity loss separately.

The eight probes share one trained model and a finite bank. Do not treat them as eight independent trained seeds or use 48 correlated rollout rows as the unit of method-level uncertainty. Report all paired probe values and conditional descriptive variation; no claim of general policy improvement.

## Why Option C is separate

Implement its analytic fixtures for comparison, but do not include it silently as the simple-width arm. Option C requires:

- U(query)-U(logged) positive-part penalty.
- Query clip(actor+N(0,0.2^2), -1,1), without inner noise clipping.
- Unsmooth Q1 in the actor score; raw-Q denominator with the draft's explicit 1e-7.
- Refusal penalty [Q1(actor)-Q1(logged)]+.
- k=1, ordinary BC coefficient one.
- Defined treatment of query-mass threshold lookup and action derivatives.

The known categorical source in the completed action-only experiment supports 625 grid commands. All 64 deployed actor commands were off-grid. Continuous noisy Option C queries are also not covered by that discrete law. Do not assign them finite exact categorical ratios, floor denominators, or call projection a certificate for the deployed action.

Therefore the first learned direct-width pilot is explicitly heuristic outside support. A support-strict Option C extension would route unsupported queries to its declared refusal branch; it cannot be sold as a finite-band action-shift demonstration. Establish a separate continuous-source/target law, including boundary atoms if clipping is used, before a future weighted coverage claim. No such new data-law experiment is launched here.

## Decision after the small pilot

Advance to a separately specified matched training study only if the gradient contract passes and the penalty's direction has useful evidence across both harmful stall actions and beneficial high-load monotone actions. A penalty that suppresses every unfamiliar action has not demonstrated the intended discrimination.

Use three decisions: advance, inconclusive, redesign. Failure to beat a constant-strength control weakens the claim for adaptive calibration. A no-op on the explicitly action-varying analytic fixture is an implementation problem. A correct learned graph can still have zero effect because eta is flat or floored, R=0, the actor saturates, or batch gradients cancel; inspect action and parameter gradients separately before classifying that outcome. Accurate residual coverage without helpful ranking is a limitation of the penalty signal. A useful gradient does not by itself establish calibrated uncertainty or policy benefit.

A later training comparison must use multiple matched trained seeds, all coverage levels, fixed budgets, validation-only selection, complete failures/refusals, and a frozen manifest. None of that training is part of this planning turn.

## Separate nominal-coverage sensitivity plan

Do not use the name alpha for three different controls:

| Setting | Proposed values | Meaning |
|---|---|---|
| Nominal residual coverage c_nom | 0.80, 0.90, 0.95, 0.99 | Intended one-step residual coverage |
| Miscoverage epsilon=1-c_nom | 0.20, 0.10, 0.05, 0.01 | Native calibration alpha |
| Bayesian credibility beta_cred | 0.95, fixed | Posterior risk credibility |
| TD3+BC alpha_host | 2.5, fixed | Raw-Q/BC objective scaling |
| Direct penalty kappa | 1, fixed for first pilot | Penalty strength |

Proposed analysis: reuse all 200 paired fresh-law calibration replications, n=128, M=1000 posterior draws, and the same 4096 evaluation queries per mixture from the completed controlled experiment. Keep the frozen model, residual label kernel, state law, score scale, all random streams, exact categorical ratios, and fallback rules identical. Report every level and every replication. This is a post-hoc paired sensitivity analysis on already observed samples, not new confirmatory evidence. New independent confirmation would need separately frozen seeds and scope.

Compare native uniform Bayesian calibration, exact categorical IW Bayesian calibration, and ordinary split conformal on the identical normalized score banks. Ordinary split conformal uses k=ceil((n+1)*c_nom), with infinity if k>n. At n=128 the four ranks are 104,117,123,128. Its standard marginal guarantee requires exchangeable calibration and test scores: applicable to the matched lambda=0 law, not automatically to shifted lambda=.5 or1. Those shifted unweighted results remain empirical stress checks. See [Angelopoulos and Bates](https://arxiv.org/html/2107.07511v6).

Explain expected conservatism before results: with equal unit weights and query mass one, the posterior infinity-atom mass has Beta(1,n) law. The probability a posterior crossing is finite is 1-(1-epsilon)^n. For n=128 and epsilon=.01 this is about .72375, below credibility .95. A 99% nominal band can therefore commonly be infinite even under IID sampling; that is not automatically a formula bug. The analogous minimum n for this probability to reach .95 is 14,29,59,299 at the four requested levels. Finite M introduces Monte Carlo variability. This is an analytical prediction, not an executed sensitivity result, and it does not apply unchanged to unequal weights.

Report coverage including infinity; finite-covered/all and finite-conditional coverage with denominators; widths; refusals/fallback; positive-weight hits; ESS, tails and query mass; context/time strata. Use independent calibration replication for paired uncertainty. Separate zero target-action hits from finite-sample posterior vacuity and implementation failure.

Do not choose a nominal level by maximizing test return or minimizing width on these reported samples. A future selection rule must state its coverage/utility objective and use separate validation data, retaining the final test set.

## Evidence and boundaries

Preserve the original 30-job run, 200-replication action-only experiment, raw arrays, source closure, historical failures, and all hashes. This plan adds documents only. Future implementation must record source, dependency versions before execution, configuration, seeds, all rejects, analysis commands and a manifest. Device/cloud workflow approval was user-confirmed, not independently audited. No institutional ownership or publication authority is inferred.

The initial audit was document-only. Following the16:56 execution request, the simple-objective mechanical tests and read-only actual-checkpoint gradient audit were run, as separately recorded in mechanics/REPORT.md. No learned-policy training, upload workaround, Git pull/merge/push, or unrelated V9 modification occurred. A newly authorized frozen-policy ranking diagnostic is separate from the proposed40-step actor pilot above; its own manifest governs execution.
