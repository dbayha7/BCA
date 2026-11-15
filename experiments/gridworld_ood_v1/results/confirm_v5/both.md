# Version A in both environments

Rule: PROTOCOL.md Amendment 10.

## Honeypot

Datasets: 9. A gate passes with at least 6.

C0: 9 valid datasets used of 12 candidates (dropped, baseline not competent: none; valid and not needed: [7510, 7511, 7512]), G1 9 of 9 datasets: pass

| R1 gate | Datasets passing | Verdict |
|---|---|---|
| G2 width responds | 9 of 9 | pass |
| G3 beats both controls | 5 of 9 | FAIL |
| G4 specific to the gap | 9 of 9 | pass |
| G5 no wider harm | 8 of 9 | pass |

R1: FAIL

| R2 control | Mean difference in H | Version A lower / tie / control lower | p | Verdict |
|---|---|---|---|---|
| shuffled control | +5.42 points | 6 / 3 / 0 | 0.0156 | pass |
| sharpened baseline | +3.12 points | 7 / 2 / 0 | 0.0078 | pass |

R2: pass

| Dataset | Baseline | Version A | Shuffled control | Sharpened baseline |
|---|---|---|---|---|
| 7501 | 10.04% | 0.52% | 8.98% | 6.19% |
| 7502 | 2.06% | 0.00% | 14.02% | 0.00% |
| 7503 | 3.01% | 0.00% | 0.00% | 0.00% |
| 7504 | 13.56% | 0.00% | 8.33% | 7.38% |
| 7505 | 6.47% | 0.00% | 9.34% | 1.47% |
| 7506 | 2.49% | 0.00% | 0.00% | 0.55% |
| 7507 | 8.73% | 0.00% | 1.05% | 3.78% |
| 7508 | 10.67% | 0.00% | 7.61% | 8.31% |
| 7509 | 2.78% | 0.00% | 0.00% | 0.91% |

R1 fails and R2 passes: Version A has fewer hazard endings than both controls on average, and the per-dataset gates did not pass.

## Bottleneck

### Low exit pays 4.5

C0: 9 valid datasets used of 12 candidates (dropped: none), G1 9 of 9: pass

| Claim | Result | Verdict |
|---|---|---|
| U1 cautious | baseline minus Version A +1.40 points; lower / tie / higher 4 / 5 / 0; p 0.0625 | FAIL |
| U2 no wider harm | G5 passes in 9 of 9 (mark 6) | pass |
| U3 brings the bridge back | Version A minus baseline -1.40 points; p 1.0000 | FAIL |

Other gates, datasets passing of 9: {'G2': 9, 'G3': 1, 'G4': 9}

| Dataset | Baseline | Version A | Sharpened baseline |
|---|---|---|---|
| 7501 | 4.54% | 0.00% | 2.01% |
| 7502 | 0.00% | 0.00% | 0.00% |
| 7503 | 0.00% | 0.00% | 0.00% |
| 7504 | 6.19% | 2.82% | 2.95% |
| 7505 | 0.00% | 0.00% | 0.00% |
| 7506 | 2.00% | 0.00% | 0.00% |
| 7507 | 2.72% | 0.00% | 0.00% |
| 7508 | 0.00% | 0.00% | 0.00% |
| 7509 | 0.00% | 0.00% | 0.00% |

### Low exit pays 3.5

C0: 9 valid datasets used of 12 candidates (dropped: none), G1 9 of 9: pass

| Claim | Result | Verdict |
|---|---|---|
| U1 cautious | baseline minus Version A +5.50 points; lower / tie / higher 6 / 3 / 0; p 0.0156 | pass |
| U2 no wider harm | G5 passes in 7 of 9 (mark 6) | pass |
| U3 brings the bridge back | Version A minus baseline -5.50 points; p 1.0000 | FAIL |

Other gates, datasets passing of 9: {'G2': 9, 'G3': 1, 'G4': 9}

| Dataset | Baseline | Version A | Sharpened baseline |
|---|---|---|---|
| 7501 | 12.48% | 2.53% | 9.18% |
| 7502 | 0.00% | 0.00% | 0.00% |
| 7503 | 2.47% | 0.00% | 0.00% |
| 7504 | 15.84% | 0.00% | 13.07% |
| 7505 | 5.34% | 0.00% | 2.26% |
| 7506 | 2.49% | 0.00% | 1.90% |
| 7507 | 13.37% | 0.00% | 12.35% |
| 7508 | 0.00% | 0.00% | 0.00% |
| 7509 | 0.00% | 0.00% | 0.00% |

### Low exit pays 2.5

C0: 9 valid datasets used of 12 candidates (dropped: none), G1 9 of 9: pass

| Claim | Result | Verdict |
|---|---|---|
| U1 cautious | baseline minus Version A +8.83 points; lower / tie / higher 8 / 1 / 0; p 0.0039 | pass |
| U2 no wider harm | G5 passes in 5 of 9 (mark 6) | FAIL |
| U3 brings the bridge back | Version A minus baseline -8.83 points; p 1.0000 | FAIL |

Other gates, datasets passing of 9: {'G2': 9, 'G3': 3, 'G4': 9}

| Dataset | Baseline | Version A | Sharpened baseline |
|---|---|---|---|
| 7501 | 16.82% | 0.00% | 12.85% |
| 7502 | 25.89% | 0.00% | 30.41% |
| 7503 | 2.06% | 0.00% | 0.00% |
| 7504 | 12.81% | 0.00% | 7.59% |
| 7505 | 12.88% | 0.00% | 5.27% |
| 7506 | 2.73% | 0.00% | 2.36% |
| 7507 | 4.08% | 0.00% | 4.05% |
| 7508 | 2.18% | 0.00% | 0.00% |
| 7509 | 0.00% | 0.00% | 0.00% |

U3 fails and U1 is not shown at every exit value: Version A does not relieve the underestimation. U2 fails at one or more exit values: reported with its harm.
