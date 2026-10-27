# Continuous ThrusterChain: completed core comparison

All 30 planned jobs reached 100,000 critic updates and 50,000 actor updates. Terminal results show no pooled return advantage for BCA in any retention condition. Both known-model, untrained capped-LQR controls outperform every learned policy on the shared final bank.

The primary endpoint is raw undiscounted return over H=100, averaged equally across four contexts and 400 common starts. Larger (less negative) is better. Each mean and SD below uses five independent training replicates.

| Retention | TD3+BC mean ± SD | BCA mean ± SD | Paired BCA − TD3+BC | Exploratory 95% paired-t CI | Positive pairs |
|---|---:|---:|---:|---:|---:|
| full | -142.444 ± 12.699 | -146.301 ± 19.173 | -3.857 | [-14.302, 6.588] | 3/5 |
| rare | -182.931 ± 2.347 | -186.628 ± 4.302 | -3.697 | [-8.447, 1.052] | 1/5 |
| zero | -181.739 ± 2.331 | -187.736 ± 3.278 | -5.998 | [-8.597, -3.399] | 0/5 |

Intervals use paired seed differences, df=4; they are exploratory, unadjusted for multiple comparisons, approximate, and conditional on this fixed evaluation bank. Zero retention has five negative differences, but four pairs compare different hosts. Its interval excluding zero does not isolate a BCA causal effect. Rare retention is entirely Zeus and its pooled interval includes zero.

## Every terminal seed result

| Retention | Replicate | TD3+BC return | BCA return | Difference | Hosts TD3 / BCA |
|---|---:|---:|---:|---:|---|
| full | 1 | -163.643362 | -180.564246 | -16.920884 | Zeus / Zeus |
| full | 2 | -141.518526 | -138.314636 | +3.203891 | Zeus / Titan |
| full | 3 | -139.189343 | -138.797727 | +0.391616 | Zeus / Titan |
| full | 4 | -138.368618 | -136.715924 | +1.652694 | Zeus / Titan |
| full | 5 | -129.497707 | -137.109991 | -7.612284 | Zeus / Titan |
| rare | 1 | -180.156955 | -186.088830 | -5.931875 | Zeus / Zeus |
| rare | 2 | -184.993799 | -187.966452 | -2.972653 | Zeus / Zeus |
| rare | 3 | -182.297099 | -179.641801 | +2.655299 | Zeus / Zeus |
| rare | 4 | -185.699260 | -191.135656 | -5.436396 | Zeus / Zeus |
| rare | 5 | -181.509263 | -188.309086 | -6.799823 | Zeus / Zeus |
| zero | 1 | -180.337145 | -184.429139 | -4.091994 | Zeus / Zeus |
| zero | 2 | -185.109428 | -190.721116 | -5.611689 | Titan / Zeus |
| zero | 3 | -178.970924 | -184.459647 | -5.488723 | Titan / Zeus |
| zero | 4 | -181.670596 | -191.255206 | -9.584610 | Titan / Zeus |
| zero | 5 | -182.604529 | -187.816787 | -5.212259 | Titan / Zeus |

## Contexts, controls, and regret

| Retention | Stall low Δ | Stall high Δ | Monotone low Δ | Monotone high Δ |
|---|---:|---:|---:|---:|
| full | -2.313 | -1.072 | -0.910 | -11.133 |
| rare | -2.800 | +8.922 | -3.706 | -17.204 |
| zero | -1.592 | -6.504 | -1.625 | -14.270 |

BCA's rare-retention stall/high-load mean improves, while monotone/high-load return worsens in all three retention settings. This is a descriptive context pattern, not a selected endpoint or a multiplicity-adjusted discovery.

- Known-model `fixed_cap`: return -100.197230, mean fixed-start regret bounds [35.104456, 35.104456].
- Known-model `regime_cap`: return -75.008281, mean fixed-start regret bounds [9.915506, 9.915506].
- Shared 400-start QP optimum mean return bounds: [-65.092774280972, -65.092774280967]. Maximum numerical Frank-Wolfe gap: 1.27e-11.

These controls use known dynamics and finite-horizon LQR gains, so they are not data-matched learned benchmarks. Full retention learned mean regret is about 77–81; rare/zero mean regret about 117–123. All per-seed/context bounds and failure frequencies are in the CSVs and JSON. The failure event is any |position|>3, with episodes still run to H; it is not an episode termination rule or a general safety criterion.

## Value error and empirical calibration

| Retention | Arm | True Q RMSE | Deterministic Bellman RMSE | Ranking regret | Failure frequency | Residual coverage |
|---|---|---:|---:|---:|---:|---:|
| full | td3_bc | 31.076 | 6.012 | 0.831 | 0.000% | absent |
| full | bca | 29.682 | 5.408 | 0.857 | 0.000% | 0.656 |
| rare | td3_bc | 118.724 | 5.926 | 1.019 | 0.250% | absent |
| rare | bca | 111.772 | 6.604 | 1.141 | 0.050% | 0.589 |
| zero | td3_bc | 125.392 | 6.479 | 0.829 | 0.300% | absent |
| zero | bca | 121.530 | 7.199 | 0.842 | 0.200% | 0.610 |

The table averages five per-seed diagnostics; it does not pool their errors before computing RMSE. True Q is a forced command followed by the deployed policy to H, not optimal Q. Small one-step Bellman residuals coexist with much larger long-horizon Q error, especially after critical-row deletion. Ranking regret compares the critic-selected command against the best true Q among six candidates.

Each method's true Q follows its own policy and its sixth candidate is its own actor command, so cross-method Q-error differences do not isolate critic quality under one shared target. The 288 query pairs are 48 designed states crossed with six candidates, not 288 independent draws.

Coverage targets a different residual: the native noisy TD3 target-network target minus current twin-min Q, compared with the frozen checkpoint band (last refresh 95k). Every BCA run has empirical query coverage below nominal 0.9; this does not establish any long-horizon Q coverage. The designed query law differs from the behavior calibration law, so this is not by itself a refutation of a conformal theorem. The pooled 128-row calibration bank is adaptively reused. Rare critical counts are 1, 3, 0, 0, 1; zero counts are all 0. Numerical readiness is not critical-region certification. All BCA final band refusal rates are zero and native usability is one; no executed actions are refused, and these values are not statistical support guarantees.

## Provenance and limits

The analysis combines 22 locally verified Zeus results with eight results in the user-supplied Titan report. The report SHA256 is `1f65b23ab78d06a16390941a9b146b6ec80e7cde6d44e0e790e92f64b15ca2ee`. Local source/preparation/checkpoint/evaluation hashes and summary arithmetic were verified. Titan job, bank, checkpoint and paired initialization/PRNG metadata were checked against available records; Titan raw arrays and checkpoints were not independently accessed or rerun. Both hosts trained on CPU. The reported fixture tolerance check does not remove host/OS/runtime confounding.

The 50k validation results are descriptive and use a separate 80-start bank. No checkpoint selection, tuning, retraining, or modifications to frozen records occurred. Per-context matched Q summaries are unavailable for the eight supplied jobs; they are not imputed. Full context return/coverage summaries remain available.

Reproduce against the complete local run with `python aggregate_results.py --run-dir <run-directory> --titan-report ../inputs/titan_results.json --output .`. A portable evidence bundle can use `--source-root <source-closure-directory> --evidence-only`; training archives and checkpoint bytes may be omitted only in that explicit mode, and every omission is recorded rather than claimed verified. The run folder keeps its original name `core_20261020T2250Z`. This script uses NumPy/SciPy only. `aggregate.json` preserves all summaries and source identities; CSVs expose every seed/context, paired differences, method statistics, calibration counts, and controls. Analytical next steps are to inspect the reported context/Q/calibration failures and design any future study separately; this completed run does not establish a BCA benefit.
