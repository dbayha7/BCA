# Which saved signals distinguish good from bad unsupported actions?

Completed knee/stall study, not Path A. This is a retrospective follow-up using 24 saved models and 13,824 withheld-versus-logged pairs. Three training seeds are independent replications; actions and states are nested and reused. No model inference, simulator calls, training or recalibration ran.

## 1. Native Q gets the relative decision wrong in stall

For each state, g is the true own-policy return advantage of a withheld action over the logged action; m is the Q1 difference for that same pair. Relative error is e = m - g. A shared additive Q bias cancels. Harm means g < 0; useful means g > 0. These labels do not depend on the sign of either action's absolute Q error.

Native metric | Stall: seeds 0 / 1 / 2 | No-stall: seeds 0 / 1 / 2
Harmful pairs raw Q prefers | 170 / 168 / 172 | 26 / 19 / 30
Useful pairs raw Q misses | 115 / 116 / 122 | 37 / 38 / 34
Relative Q1-gap MAE | 2.242 / 2.079 / 2.056 | 0.332 / 0.342 / 0.336

Each model has 576 pairs: 192 states x three withheld alternatives. Native stall falsely prefers 510 of 1,212 harmful pairs and misses 353 of 516 useful pairs across the three models. No-stall falsely prefers 75 of 1,119 and misses 109 of 609. These counts differ from the previous bank-argmax counts because each bank contributes three pair comparisons.

Among the 510 harmful stall preferences, 117 have no positive absolute Q1 error. No-stall has 38 such cases out of 75. Conversely, 203 of the 353 useful stall actions that Q misses have positive absolute Q error. Detecting absolute overestimation alone therefore cannot identify every ranking mistake.

Native stall harmful false preferences have mean true advantage -3.747 but predicted advantage +0.725. Missed useful pairs have mean true advantage +0.707 but predicted advantage -0.973. The limitation is not only a common downward or upward offset.

# Native ranking evidence

See figure 1.

# What U captures, and what it was trained to predict

## 2. Relative U reveals context that absolute U hid

The table uses pooled harm AUROC across state-relative pairs within each model: higher signal predicts a lower-valued withheld action. This is descriptive discrimination, not a calibrated decision threshold. No AUROC direction was flipped.

Risk signal (native) | Stall: seeds 0 / 1 / 2 | No-stall: seeds 0 / 1 / 2
Negative Q1 margin | 0.470 / 0.442 / 0.428 | 0.957 / 0.942 / 0.941
U minus logged U | 0.474 / 0.489 / 0.485 | 0.813 / 0.813 / 0.751
Distance beyond cutoff | 0.533 / 0.544 / 0.525 | 0.505 / 0.515 / 0.510
Q1 - twin-min contrast | 0.481 / 0.535 / 0.485 | 0.445 / 0.522 / 0.510

In no-stall, delta U contains useful cross-state information (AUROC 0.751-0.813 for native), refining the earlier near-chance result for absolute U. Subtracting logged U removes a large state-level offset. Within a bank the subtraction is constant and cannot change candidate ordering. Only 2/6/4 native no-stall banks have both harm labels, so within-state harm discrimination remains weakly sampled.

In stall, neither raw margin nor delta U reliably separates harmful from useful pairs. For native harmful false preferences mean delta U is only 0.00934, versus 0.04225 for missed useful pairs. A larger global multiplier does not supply the missing sign. These are associations, not a proof of why training produced the policy.

## 3. The actual residual objective is a different estimand

The local scale minimizes squared error between a Bayesian-weighted minibatch average of soft coverage and 0.9, plus a width regularizer. Its residual is a one-step, smoothed-target twin-min TD target minus current twin-min Q. Calibration uses an absolute residual divided by scale. Neither target is the signed full-return pair error m - g.

The actor uses Q1 - kappa U with unchanged BC strength and native critic loss. Q1-to-twin-min mismatch is real, but changing the saved margin to twin-min barely changes native MAE: stall 2.12546 -> 2.12064; no-stall 0.33645 -> 0.34421. It is not a sufficient explanation of the observed ranking problem.

# The penalty helps some decisions and harms others

See figure 2.

# Interpretation, sensitivity and the smallest next test

## 4. Preserve the benefit while locating the cost

At kappa 189, stall corrects 68/192/68 harmful pair preferences, suppresses 32/54/33 useful preferences, and creates 42/6/27 new harmful preferences. No-stall corrects 10/14/42, suppresses 72/159/126, and creates 22/12/18. A positive U may still have a negative difference from logged U, so a downward score can increase relative preference for some actions.

Relative score MAE increases after adjustment for every BCA model. That does not erase improved argmax regret or policy returns: a conservative score is not an unbiased Q estimate. The actual kappa-189 stall return gains remain +11.245/+121.479/+9.760, with no-stall changes -41.421/-98.321/-20.525. Candidate pairs cannot establish the cause of those policy effects.

## 5. Numerical near-ties do not explain the broad pattern

Exact original labels are preserved. Sensitivity excludes |g| <= 0.000657327361 and <= 0.001314654721, anchored to the largest saved I1 discrepancy and twice that value. The larger level marks 20/13,824 pairs near-tied, at most three per model. This is an empirical stress scale, not a proven error bound; the two original I1 failures and formal INCONCLUSIVE result remain unchanged.

## 6. One next measurement, before choosing an estimator change

Saved candidates contain Q1, Q2, U and own-policy Qpi, but no aligned signed TD targets, pointwise residuals, eta components or candidate gradients. Existing fitting-probe gradient files have different actions and no matched candidate oracle values. These cannot prove whether information was lost in the residual proxy, scale fit, calibration or penalty mapping.

The smallest justified prospective test is a residual-alignment measurement on the existing native checkpoints and these same banks, not another training matrix. Use authenticated 15k states to examine the reference used by final U; 20k residuals answer a different current-model question. Up to 1,152 unique new one-step withheld successors would cover two worlds x 192 states x three alternatives; logged successors can be reused. Six models require 4,608 target-row evaluations per checkpoint level. No long rollouts or optimizer updates are needed. This would require separate execution approval and remain development analysis.

Do not select a new algorithm from these labels. The evidence argues against simply increasing kappa, switching Q1 to twin-min, or subtracting a logged-state constant as a cure. First determine whether signed action-relative residuals contain the missing distinction. BC and Path A remain unchanged.

# Evidence boundaries and reproducibility

## What is available when deciding?

Given a state and candidate actions, Q1, Q2 and U are model outputs. Margins and delta U also require a comparator; this analysis has the saved logged action, which a deployed policy at a new state does not automatically possess. The cutoff label is benchmark knowledge. It is not a universal support estimator.

Reward and successor state arrive after acting. One-step TD targets also need target-policy/critic queries and a specified smoothing-noise realization. Full-return Qpi, true gap g, relative error e and harm labels are evaluator-only. They were not fed into learning or calibration.

## What is and is not paired

Within a world, complete candidate inputs are shared across models, but each model has its own continuation policy. Between worlds, identical row IDs do not mean identical physical states: the preceding analysis verified different retained states, rewards, successors and normalization. World identification is not the same problem as choosing the best action within one world.

## Source and numerical provenance

Source review binds the actual actor, bridge, scale, calibration and evaluator code to recorded job integrity pins. All 24 complete original evaluation reports and previous analysis outputs are hash checked before and after analysis. The independent review is saved in SOURCE_REVIEW.md. Raw per-pair values and all seed/strength/sensitivity statistics are included; no fitted classifier, predictor search or new threshold claim is used.

Reproduce from the authorized BCA checkout with the existing NumPy/matplotlib environment: python -B analysis/knee_decision_signal_v1/analyze.py --output-dir analysis/knee_decision_signal_v1/output/new_run. Then run build_report.py --input <new_run> --output <new_report_directory>. Both output directories must be new. Original saved data/reports are required; the ZIP contains analysis code and outputs, not large original datasets or checkpoints.

Original CF_RESULT SHA256: b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197. The report is an exploratory explanation, not a replacement of the frozen verdict, a conditional-coverage guarantee, a Bayesian-superiority claim, or a finding about Honeypot/Bottleneck.
