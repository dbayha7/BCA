# Step 1 results: GO

EXPLORATORY: restructured BCA, step 1 (the decisive test). Designed after seeing seeds 0-4; the specification was written before any step-1 critic was trained. Outside the gate; changes no pre-registered rule and no v6/v7 file.

Mode: official. results.json SHA-256 e513fd196b49fd1f6c1e76b6d24d2b3e45d1aef93071540d38cf7a69dde4fd55. prespec.json 669e836cb9dfc92def9282393eb3e2015cfdd901bd2e722f6c44c7f3154ac06e.

## Conditions (fixed in prespec.json before any step-1 critic was trained)

- C1 certificate on in 5 of 5 stalling seeds: True; off in 5 of 5 no-stall seeds: True
- C2 net across worlds 106.1 >= 90: True (stalling gain 106.1, no-stall loss 0.0)
- C3 mean shuffled net 0.1 <= half of 106.1: True (95th percentile 0.0; 200 permutations); information only, under the other reading of 'half of that' (at most 45, half of the 90-point bar): True
- C4 plain rank test reported beside the certificate: below

## Per seed and world (held-out overestimates, value units)

| world | seed | n full | n rest | 5% gap | certificate | height | MW p | shuffled on-rate |
|---|---|---|---|---|---|---|---|---|
| stall | 0 | 35 | 850 | +1.048 | on | +6.019 | 2.3e-07 | 0.000 |
| stall | 1 | 47 | 846 | +1.005 | on | +2.420 | 3e-26 | 0.000 |
| stall | 2 | 59 | 839 | +0.838 | on | +2.065 | 2e-22 | 0.000 |
| stall | 3 | 43 | 850 | +0.779 | on | +1.605 | 1.4e-17 | 0.000 |
| stall | 4 | 47 | 825 | +0.852 | on | +1.796 | 7.1e-21 | 0.005 |
| no_stall | 0 | 38 | 849 | +0.021 | off | +0.083 | 0.21 | 0.000 |
| no_stall | 1 | 36 | 843 | +0.002 | off | +0.151 | 0.24 | 0.000 |
| no_stall | 2 | 52 | 838 | +0.010 | off | +0.200 | 0.032 | 0.000 |
| no_stall | 3 | 29 | 849 | +0.020 | off | inf | 0.18 | 0.000 |
| no_stall | 4 | 60 | 827 | -0.047 | off | +0.038 | 0.42 | 0.000 |

Stalling world, strong current: check 91.1 points against random switching at the check's own rate 68.9. Recorded conditions net 7.1.
Information only: forced-on net 44.4; rank-test-switched net 93.6.

## Meaning

BCA's own residual posterior, fed held-out residuals, separates the worlds and carries the cross-world deployment gain on seeds 0-4. This is still exploratory (designed after seeing seeds 0-4); it licenses asking for step 2 (the policy pilot), not a claim. If the rank test gives the same yes/no, the claim stays 'BCA's residual score reads outcomes', not 'the Bayesian aggregation adds value'.
