# Result F, seed 0, rerun in the cloud sandbox: comparison with the desktop

**SMOKE TEST (the TD3+BC host actor stands in for every learner; not results). EXPLORATORY: outside the version-5 pre-registration; result F's arms retrained for data seed 0 in the cloud sandbox (iw_rerun) with result F's own code unchanged; evaluation only.**

**Overall: RESULT F'S GAIN NOT REPRODUCED in the sandbox on seed 0 (see the paired table) [partial scoring: core rows missing: medium-expert f1_B2_cmd_train, f1_C2_untargeted, le_oracle; expert f1_B2_cmd_train, f1_C2_untargeted, le_oracle]**

**Partial scoring** (`eval_resultF.py run --allow-missing`): core rows missing: medium-expert f1_B2_cmd_train, f1_C2_untargeted, le_oracle; expert f1_B2_cmd_train, f1_C2_untargeted, le_oracle. The missing rows are listed at the end.

- medium-expert: A2 - No extra caution on the strong starts +0.00 return units (+0.00 points); desktop seed 0 +23.87, seeds 0-4 [23.388397057917103, 29.574472055612148]; inside the desktop's range: False
- expert: A2 - No extra caution on the strong starts +0.00 return units (+0.00 points); desktop seed 0 +80.94, seeds 0-4 [65.5683230374341, 80.94427629785413]; inside the desktop's range: False

Return J: the exact 100-step return averaged over the 1,000 fixed starts of a start bank (closer to 0 is better); points: 0 = the random quality, 100 = the stall-aware controller. Tolerance: 3 x the SD of the desktop's own J over its learning-curve checkpoints at 16,000-20,000 steps (the same run, seed 0).

## medium-expert

| Learner | Starts | Desktop J (points) | Sandbox J (points) | Gap | Tolerance | Verdict |
|---|---|---|---|---|---|---|
| Fixed fallback, exact weights (result F) | strong | -152.44 (94.35) | -177.71 (92.36) | -25.28 | 15.1 | OUTSIDE the desktop run's own spread |
| Fixed fallback, exact weights (result F) | none | -61.09 (96.62) | -64.12 (96.22) | -3.03 | 6.5 | within the desktop run's own spread |
| No extra caution | strong | -176.31 (92.47) | -177.71 (92.36) | -1.41 | 10.6 | within the desktop run's own spread |
| No extra caution | none | -65.62 (96.02) | -64.12 (96.22) | +1.50 | 4.6 | within the desktop run's own spread |

| Paired difference | Starts | Sandbox | Desktop seed 0 | Desktop seeds 0-4 | Inside the desktop's range |
|---|---|---|---|---|---|
| A2 - No extra caution | strong | +0.00 (+0.00 points) | +23.87 | +23.9, +23.4, +25.5, +24.0, +29.6 | False |
| A2 - No extra caution | none | +0.00 (+0.00 points) | +4.53 | +4.5, +3.9, +4.3, +5.2, +5.6 | False |
| A2 - version 5's unweighted lower edge | - | missing: le_bca_unweighted | | | |
| A2 - C2 (untargeted control) | - | missing: f1_C2_untargeted | | | |
| A2 - B2 (training mix) | - | missing: f1_B2_cmd_train | | | |
| Perfect lower edge - No extra caution | - | missing: le_oracle | | | |
| A3 - No extra caution | - | missing: est_A3_estimated | | | |

- Same cloned actor (A2 against No extra caution, the clone SHA-256 in their records): **True** (identical). Fixed fallback, exact weights (result F) SMOKE: one s; No extra caution SMOKE: one s.
- Bit checks against the desktop: actor-only row stream identical: False; data D1 of seeds 0-4 identical: None; cloned actor identical: False (information only: the desktop trained the channel arms unpinned with OMP_NUM_THREADS=2, this run pins every job to one CPU with one thread, so the clone's last bits may differ).
- Result F's checks before training (sandbox): None.
- 'No value credit' share per chunk, A2: sandbox [], desktop [0.0607, 0.0711, 0.073, 0.0758, 0.0698].
- Scores equal the scripts' own evaluate records: {'f1_A2_cmd_strong': None, 'le_none': None}.

| Learner | recorded | big_kicks | big_waves | strong_current |
|---|---|---|---|---|
| Fixed fallback, exact weights (result F) | 98.3 | 92.0 | 96.2 | 91.3 |
| No extra caution | 98.3 | 92.0 | 96.2 | 91.3 |

## expert

| Learner | Starts | Desktop J (points) | Sandbox J (points) | Gap | Tolerance | Verdict |
|---|---|---|---|---|---|---|
| Fixed fallback, exact weights (result F) | strong | -116.40 (97.18) | -204.43 (90.26) | -88.02 | 13.0 | OUTSIDE the desktop run's own spread |
| Fixed fallback, exact weights (result F) | none | -49.45 (98.14) | -68.87 (95.60) | -19.41 | 1.7 | OUTSIDE the desktop run's own spread |
| No extra caution | strong | -197.35 (90.81) | -204.43 (90.26) | -7.08 | 9.6 | within the desktop run's own spread |
| No extra caution | none | -66.24 (95.94) | -68.87 (95.60) | -2.62 | 1.9 | OUTSIDE the desktop run's own spread |

| Paired difference | Starts | Sandbox | Desktop seed 0 | Desktop seeds 0-4 | Inside the desktop's range |
|---|---|---|---|---|---|
| A2 - No extra caution | strong | +0.00 (+0.00 points) | +80.94 | +80.9, +65.6, +79.1, +70.9, +73.9 | False |
| A2 - No extra caution | none | +0.00 (+0.00 points) | +16.79 | +16.8, +13.9, +16.3, +14.8, +16.2 | False |
| A2 - version 5's unweighted lower edge | - | missing: le_bca_unweighted | | | |
| A2 - C2 (untargeted control) | - | missing: f1_C2_untargeted | | | |
| A2 - B2 (training mix) | - | missing: f1_B2_cmd_train | | | |
| Perfect lower edge - No extra caution | - | missing: le_oracle | | | |
| A3 - No extra caution | - | missing: est_A3_estimated | | | |

- Same cloned actor (A2 against No extra caution, the clone SHA-256 in their records): **True** (identical). Fixed fallback, exact weights (result F) SMOKE: one s; No extra caution SMOKE: one s.
- Bit checks against the desktop: actor-only row stream identical: False; data D1 of seeds 0-4 identical: None; cloned actor identical: False (information only: the desktop trained the channel arms unpinned with OMP_NUM_THREADS=2, this run pins every job to one CPU with one thread, so the clone's last bits may differ).
- Result F's checks before training (sandbox): None.
- 'No value credit' share per chunk, A2: sandbox [], desktop [0.1282, 0.1427, 0.1509, 0.1488, 0.1446].
- Scores equal the scripts' own evaluate records: {'f1_A2_cmd_strong': None, 'le_none': None}.

| Learner | recorded | big_kicks | big_waves | strong_current |
|---|---|---|---|---|
| Fixed fallback, exact weights (result F) | 98.4 | 93.5 | 97.0 | 88.8 |
| No extra caution | 98.4 | 93.5 | 97.0 | 88.8 |

## Missing learners

- medium-expert f1_B2_cmd_train: smoke test
- medium-expert f1_C2_untargeted: smoke test
- medium-expert le_oracle: smoke test
- medium-expert le_bca_unweighted: smoke test
- medium-expert est_A3_estimated: smoke test
- medium-expert est_C3_untargeted: smoke test
- expert f1_B2_cmd_train: smoke test
- expert f1_C2_untargeted: smoke test
- expert le_oracle: smoke test
- expert le_bca_unweighted: smoke test
- expert est_A3_estimated: smoke test
- expert est_C3_untargeted: smoke test
