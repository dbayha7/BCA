# Result F, seed 0, rerun in the cloud sandbox: comparison with the desktop

**EXPLORATORY: outside the version-5 pre-registration; result F's arms retrained for data seed 0 in the cloud sandbox (iw_rerun) with result F's own code unchanged; evaluation only.**

**Overall: CONSISTENT WITH RESULT F: A2 beats No extra caution on the strong starts on both qualities, both started from the same cloned actor, and A2's and No extra caution's returns lie within the desktop run's own spread**

- medium-expert: A2 - No extra caution on the strong starts +18.49 return units (+1.46 points); desktop seed 0 +23.87, seeds 0-4 [23.388397057917103, 29.574472055612148]; inside the desktop's range: False
- expert: A2 - No extra caution on the strong starts +79.90 return units (+6.29 points); desktop seed 0 +80.94, seeds 0-4 [65.5683230374341, 80.94427629785413]; inside the desktop's range: True

Return J: the exact 100-step return averaged over the 1,000 fixed starts of a start bank (closer to 0 is better); points: 0 = the random quality, 100 = the stall-aware controller. Tolerance: 3 x the SD of the desktop's own J over its learning-curve checkpoints at 16,000-20,000 steps (the same run, seed 0).

## medium-expert

| Learner | Starts | Desktop J (points) | Sandbox J (points) | Gap | Tolerance | Verdict |
|---|---|---|---|---|---|---|
| Fixed fallback, exact weights (result F) | strong | -152.44 (94.35) | -148.15 (94.69) | +4.28 | 15.1 | within the desktop run's own spread |
| Fixed fallback, exact weights (result F) | none | -61.09 (96.62) | -60.19 (96.73) | +0.90 | 6.5 | within the desktop run's own spread |
| Fixed fallback, training mix of currents | strong | -152.01 (94.38) | -146.62 (94.81) | +5.38 | 14.5 | within the desktop run's own spread |
| Fixed fallback, training mix of currents | none | -61.04 (96.62) | -59.73 (96.79) | +1.31 | 6.1 | within the desktop run's own spread |
| Untargeted control of the fixed fallback | strong | -176.49 (92.46) | -164.11 (93.43) | +12.38 | 13.3 | within the desktop run's own spread |
| Untargeted control of the fixed fallback | none | -66.48 (95.91) | -63.25 (96.33) | +3.23 | 5.7 | within the desktop run's own spread |
| Estimated weights | strong | -177.27 (92.39) | -169.47 (93.01) | +7.80 | 12.7 | within the desktop run's own spread |
| Estimated weights | none | -66.66 (95.89) | -64.12 (96.22) | +2.54 | 5.7 | within the desktop run's own spread |
| Untargeted control of the estimated weights | strong | -175.40 (92.54) | -165.39 (93.33) | +10.00 | 11.6 | within the desktop run's own spread |
| Untargeted control of the estimated weights | none | -65.72 (96.01) | -63.45 (96.31) | +2.27 | 5.0 | within the desktop run's own spread |
| No extra caution | strong | -176.31 (92.47) | -166.65 (93.23) | +9.66 | 10.6 | within the desktop run's own spread |
| No extra caution | none | -65.62 (96.02) | -62.96 (96.37) | +2.66 | 4.6 | within the desktop run's own spread |
| Perfect lower edge | strong | -107.96 (97.85) | -107.01 (97.92) | +0.95 | 6.6 | within the desktop run's own spread |
| Perfect lower edge | none | -51.52 (97.87) | -50.57 (97.99) | +0.95 | 3.7 | within the desktop run's own spread |
| Version 5's lower edge, unweighted | strong | -176.49 (92.46) | -165.86 (93.29) | +10.63 | 11.6 | within the desktop run's own spread |
| Version 5's lower edge, unweighted | none | -65.50 (96.04) | -62.62 (96.42) | +2.88 | 4.7 | within the desktop run's own spread |

| Paired difference | Starts | Sandbox | Desktop seed 0 | Desktop seeds 0-4 | Inside the desktop's range |
|---|---|---|---|---|---|
| A2 - No extra caution | strong | +18.49 (+1.46 points) | +23.87 | +23.9, +23.4, +25.5, +24.0, +29.6 | False |
| A2 - No extra caution | none | +2.77 (+0.36 points) | +4.53 | +4.5, +3.9, +4.3, +5.2, +5.6 | False |
| A2 - version 5's unweighted lower edge | strong | +17.70 (+1.39 points) | +24.05 | +24.1, +23.0, +26.2, +22.8, +28.8 | False |
| A2 - version 5's unweighted lower edge | none | +2.43 (+0.32 points) | +4.41 | +4.4, +3.4, +4.3, +4.7, +5.4 | False |
| A2 - C2 (untargeted control) | strong | +15.95 (+1.26 points) | +24.05 | +24.1, +23.0, +22.2, +23.1, +27.0 | False |
| A2 - C2 (untargeted control) | none | +3.06 (+0.40 points) | +5.39 | +5.4, +4.6, +4.3, +5.2, +5.6 | False |
| A2 - B2 (training mix) | strong | -1.53 (-0.12 points) | -0.43 | -0.4, -0.1, +0.1, +0.1, -0.8 | False |
| A2 - B2 (training mix) | none | -0.46 (-0.06 points) | -0.05 | -0.1, -0.1, -0.1, -0.2, -0.1 | False |
| Perfect lower edge - No extra caution | strong | +59.64 (+4.69 points) | +68.34 | +68.3, +68.8, +75.4, +76.5, +78.5 | False |
| Perfect lower edge - No extra caution | none | +12.39 (+1.62 points) | +14.10 | +14.1, +14.3, +15.5, +16.4, +15.7 | False |
| A3 - No extra caution | strong | -2.82 (-0.22 points) | -0.96 | -1.0, +1.9, +2.2, +2.8, +3.2 | False |
| A3 - No extra caution | none | -1.16 (-0.15 points) | -1.04 | -1.0, +0.0, -0.1, +0.4, +0.2 | False |

- Same cloned actor (A2 and A3 against No extra caution, the clone SHA-256 in their records): **True** (identical). Fixed fallback, exact weights (result F) f87093cad7a4; Fixed fallback, training mix of currents f87093cad7a4; Untargeted control of the fixed fallback f87093cad7a4; Estimated weights f87093cad7a4; Untargeted control of the estimated weights f87093cad7a4; No extra caution f87093cad7a4; Perfect lower edge f87093cad7a4; Version 5's lower edge, unweighted f87093cad7a4.
- Bit checks against the desktop: actor-only row stream identical: True; data D1 of seeds 0-4 identical: True; cloned actor identical: False (information only: the desktop trained the channel arms unpinned with OMP_NUM_THREADS=2, this run pins every job to one CPU with one thread, so the clone's last bits may differ).
- Result F's checks before training (sandbox): {'check1_max_param_diff': 0.0, 'disagree_no_credit': 0.0, 'disagree_capped': 0.0, 'passed': True}.
- 'No value credit' share per chunk, A2: sandbox [0.0623, 0.077, 0.0777, 0.0786, 0.0782], desktop [0.0607, 0.0711, 0.073, 0.0758, 0.0698].
- Scores equal the scripts' own evaluate records: {'f1_A2_cmd_strong': True, 'f1_B2_cmd_train': True, 'f1_C2_untargeted': True, 'est_A3_estimated': True, 'est_C3_untargeted': True, 'le_none': True, 'le_oracle': True, 'le_bca_unweighted': True}.
- Version 5's lower edge, unweighted: the v5 lower-edge record's pre_registered_report is copied from the sandbox's own PARTIAL report (run.py report on data seeds 0-4 only: D1 pass, every other mark pending or not run), not from the committed version-5 report; it is not a verdict on this row (report SHA-256 8caddd13ad3b; the sandbox's report: True; the committed report: False).

| Learner | recorded | big_kicks | big_waves | strong_current |
|---|---|---|---|---|
| Fixed fallback, exact weights (result F) | 97.6 | 91.6 | 95.6 | 94.2 |
| Fixed fallback, training mix of currents | 97.6 | 91.7 | 95.7 | 94.4 |
| Untargeted control of the fixed fallback | 97.8 | 91.2 | 96.3 | 92.7 |
| Estimated weights | 97.8 | 91.7 | 96.0 | 92.2 |
| Untargeted control of the estimated weights | 97.8 | 91.3 | 96.3 | 92.6 |
| No extra caution | 98.0 | 91.7 | 96.3 | 92.4 |
| Perfect lower edge | 98.0 | 91.8 | 96.7 | 97.9 |
| Version 5's lower edge, unweighted | 98.0 | 91.7 | 96.4 | 92.5 |

## expert

| Learner | Starts | Desktop J (points) | Sandbox J (points) | Gap | Tolerance | Verdict |
|---|---|---|---|---|---|---|
| Fixed fallback, exact weights (result F) | strong | -116.40 (97.18) | -118.29 (97.04) | -1.89 | 13.0 | within the desktop run's own spread |
| Fixed fallback, exact weights (result F) | none | -49.45 (98.14) | -50.92 (97.94) | -1.47 | 1.7 | within the desktop run's own spread |
| Fixed fallback, training mix of currents | strong | -118.82 (96.99) | -122.87 (96.68) | -4.05 | 12.0 | within the desktop run's own spread |
| Fixed fallback, training mix of currents | none | -49.93 (98.07) | -51.78 (97.83) | -1.84 | 1.3 | OUTSIDE the desktop run's own spread |
| Untargeted control of the fixed fallback | strong | -197.29 (90.82) | -199.13 (90.67) | -1.83 | 7.4 | within the desktop run's own spread |
| Untargeted control of the fixed fallback | none | -66.60 (95.89) | -67.51 (95.78) | -0.91 | 1.2 | within the desktop run's own spread |
| Estimated weights | strong | -165.37 (93.33) | -172.27 (92.79) | -6.89 | 14.9 | within the desktop run's own spread |
| Estimated weights | none | -59.36 (96.84) | -61.75 (96.53) | -2.40 | 2.1 | OUTSIDE the desktop run's own spread |
| Untargeted control of the estimated weights | strong | -197.88 (90.77) | -199.06 (90.68) | -1.17 | 7.9 | within the desktop run's own spread |
| Untargeted control of the estimated weights | none | -66.63 (95.89) | -67.40 (95.79) | -0.77 | 1.4 | within the desktop run's own spread |
| No extra caution | strong | -197.35 (90.81) | -198.19 (90.75) | -0.84 | 9.6 | within the desktop run's own spread |
| No extra caution | none | -66.24 (95.94) | -66.67 (95.89) | -0.43 | 1.9 | within the desktop run's own spread |
| Perfect lower edge | strong | -87.78 (99.44) | -91.79 (99.12) | -4.01 | 5.0 | within the desktop run's own spread |
| Perfect lower edge | none | -42.16 (99.09) | -43.34 (98.94) | -1.17 | 0.9 | OUTSIDE the desktop run's own spread |
| Version 5's lower edge, unweighted | strong | -196.79 (90.86) | -198.35 (90.74) | -1.56 | 10.0 | within the desktop run's own spread |
| Version 5's lower edge, unweighted | none | -66.01 (95.97) | -66.64 (95.89) | -0.63 | 2.1 | within the desktop run's own spread |

| Paired difference | Starts | Sandbox | Desktop seed 0 | Desktop seeds 0-4 | Inside the desktop's range |
|---|---|---|---|---|---|
| A2 - No extra caution | strong | +79.90 (+6.29 points) | +80.94 | +80.9, +65.6, +79.1, +70.9, +73.9 | True |
| A2 - No extra caution | none | +15.75 (+2.06 points) | +16.79 | +16.8, +13.9, +16.3, +14.8, +16.2 | True |
| A2 - version 5's unweighted lower edge | strong | +80.05 (+6.30 points) | +80.39 | +80.4, +65.6, +80.0, +71.0, +74.3 | True |
| A2 - version 5's unweighted lower edge | none | +15.72 (+2.05 points) | +16.56 | +16.6, +13.9, +16.4, +14.8, +16.3 | True |
| A2 - C2 (untargeted control) | strong | +80.83 (+6.36 points) | +80.89 | +80.9, +63.6, +77.5, +70.5, +74.7 | True |
| A2 - C2 (untargeted control) | none | +16.59 (+2.17 points) | +17.15 | +17.1, +13.9, +16.1, +15.2, +16.8 | True |
| A2 - B2 (training mix) | strong | +4.58 (+0.36 points) | +2.41 | +2.4, +0.9, +0.3, -1.1, +0.1 | False |
| A2 - B2 (training mix) | none | +0.85 (+0.11 points) | +0.48 | +0.5, +0.2, +0.1, -0.2, +0.1 | False |
| Perfect lower edge - No extra caution | strong | +106.41 (+8.37 points) | +109.57 | +109.6, +97.5, +110.7, +102.3, +118.0 | True |
| Perfect lower edge - No extra caution | none | +23.34 (+3.05 points) | +24.08 | +24.1, +21.1, +23.6, +21.5, +26.1 | True |
| A3 - No extra caution | strong | +25.93 (+2.04 points) | +31.97 | +32.0, +25.3, +34.1, +26.7, +28.3 | True |
| A3 - No extra caution | none | +4.92 (+0.64 points) | +6.89 | +6.9, +5.5, +7.6, +6.0, +6.8 | False |

- Same cloned actor (A2 and A3 against No extra caution, the clone SHA-256 in their records): **True** (identical). Fixed fallback, exact weights (result F) 73e03006c0f6; Fixed fallback, training mix of currents 73e03006c0f6; Untargeted control of the fixed fallback 73e03006c0f6; Estimated weights 73e03006c0f6; Untargeted control of the estimated weights 73e03006c0f6; No extra caution 73e03006c0f6; Perfect lower edge 73e03006c0f6; Version 5's lower edge, unweighted 73e03006c0f6.
- Bit checks against the desktop: actor-only row stream identical: True; data D1 of seeds 0-4 identical: True; cloned actor identical: False (information only: the desktop trained the channel arms unpinned with OMP_NUM_THREADS=2, this run pins every job to one CPU with one thread, so the clone's last bits may differ).
- Result F's checks before training (sandbox): {'check1_max_param_diff': 0.0, 'disagree_no_credit': 0.0, 'disagree_capped': 0.0, 'passed': True}.
- 'No value credit' share per chunk, A2: sandbox [0.1239, 0.1446, 0.1562, 0.1483, 0.1403], desktop [0.1282, 0.1427, 0.1509, 0.1488, 0.1446].
- Scores equal the scripts' own evaluate records: {'f1_A2_cmd_strong': True, 'f1_B2_cmd_train': True, 'f1_C2_untargeted': True, 'est_A3_estimated': True, 'est_C3_untargeted': True, 'le_none': True, 'le_oracle': True, 'le_bca_unweighted': True}.
- Version 5's lower edge, unweighted: the v5 lower-edge record's pre_registered_report is copied from the sandbox's own PARTIAL report (run.py report on data seeds 0-4 only: D1 pass, every other mark pending or not run), not from the committed version-5 report; it is not a verdict on this row (report SHA-256 b3739da8f00a; the sandbox's report: True; the committed report: False).

| Learner | recorded | big_kicks | big_waves | strong_current |
|---|---|---|---|---|
| Fixed fallback, exact weights (result F) | 98.3 | 94.4 | 97.6 | 96.9 |
| Fixed fallback, training mix of currents | 98.3 | 94.3 | 97.6 | 96.5 |
| Untargeted control of the fixed fallback | 98.4 | 93.9 | 97.3 | 89.2 |
| Estimated weights | 98.2 | 94.2 | 97.6 | 91.9 |
| Untargeted control of the estimated weights | 98.5 | 94.0 | 97.4 | 89.2 |
| No extra caution | 98.6 | 94.3 | 97.5 | 89.3 |
| Perfect lower edge | 98.9 | 95.5 | 98.3 | 99.2 |
| Version 5's lower edge, unweighted | 98.6 | 94.3 | 97.5 | 89.3 |

