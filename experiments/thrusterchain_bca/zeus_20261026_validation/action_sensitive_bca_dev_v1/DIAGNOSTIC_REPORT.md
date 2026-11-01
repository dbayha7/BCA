# Q1-aligned BCA and action-compatibility: bounded diagnostic

**Result:** completed; the preregistered exploratory mechanism screen failed in both contexts. Q1 alignment did not consistently reduce true-Q MAE relative to the matched fresh twin-min fit. The compatibility penalty changed bank choices, but it did not consistently outperform both the compatibility-only and strength-matched Q1-width controls. This diagnostic did not train a policy and cannot establish policy-return effects.

## What ran

Six saved 20k hole-condition checkpoints (three seeds, native and BCA-trained), each evaluated on the 32 cached ordinary and 32 cached strong-current anchors. Each bank contains the policy action and five fixed actions; the reference is that checkpoint's own policy continuation. All comparisons on one checkpoint share states, candidate actions, critic and reference values. The 128 cached visited rows per context were also scored.

For each checkpoint, a fresh Q1-centered scale and a fresh twin-min-centered control started with identical parameters and used the same 2,000 minibatches, random keys, 64 calibration rows, and 64 check rows. Both use the unchanged smoothed target-actor/twin-min Bellman target. The compatibility coefficient was matched on 768 training-probe states to 25% of the RMS Q1 action gradient at the saved 20k actor; it is not a 10k-fork coefficient. The stronger width-only control was matched to the combined action-gradient RMS on those same probes.

The previously used 128 calibration episodes and cached simulator Q references are development-consumed. The 64-row split provides a descriptive one-step TD check, not fresh coverage of true Q or shifted policy actions. No simulator Q or Qπ label was used in fitting, calibration, compatibility or gradient matching.

## Matched Q1 versus twin-min fitting

The table gives high-load fixed-band MAE; lower is better. Ordinary has 21 anchor families and 63 correlated action rows per checkpoint, strong current has 32 and 96. The 12 checkpoint/context entries are not 12 independent training replications.

| Context | Saved policy | Fresh twin-min MAE, seeds 0 / 1 / 2 | Q1-aligned MAE, seeds 0 / 1 / 2 |
|---|---|---|---|
| Ordinary | Native | 20.07 / 21.99 / 24.89 | 19.75 / 22.15 / 25.07 |
| Ordinary | BCA | 19.74 / 22.42 / 24.99 | 19.63 / 23.18 / 25.35 |
| Strong current | Native | 22.43 / 16.33 / 30.07 | 22.35 / 17.05 / 30.56 |
| Strong current | BCA | 21.49 / 16.85 / 31.40 | 21.67 / 17.61 / 31.57 |

Q1-aligned MAE was lower in 3 of 12 entries and higher in 9. The earlier historical BCA width is retained separately in the raw output, but it was trained for longer against evolving critics and used a different calibration schedule; it is not a controlled comparison of critic-head alignment.

On the 64 previously used TD-check episodes, the aligned model contained the absolute Q1-centered one-step residual in 60–62 rows, while the matched twin-min width contained that same Q1-centered residual in 59–63 rows. These counts do not establish population, true-Q, off-support, or shifted-action coverage.

## Action choices

For the three **BCA-trained hole policies**, the combined score changed the raw-Q1 bank choice in 4 / 3 / 1 of 32 ordinary anchors and 3 / 3 / 1 of 32 strong-current anchors. Ordinary changed choices yielded 3 / 2 / 0 improvements and 1 / 1 / 1 harms across seeds; strong current yielded 2 / 0 / 0 improvements and 1 / 2 / 1 harms. Improvement or harm is measured against the same critic's raw-choice Qπ, with ±0.1 tolerance. Remaining changed choices are near-tied.

| Context | Saved policy | Raw bank regret | Combined regret | Strength-matched Q1-width regret | Compatibility-only regret |
|---|---|---|---|---|---|
| Ordinary | Native | 0.277 | 0.223 | 0.231 | 0.292 |
| Ordinary | BCA | 0.245 | 0.240 | 0.204 | 0.280 |
| Strong current | Native | 0.280 | 0.297 | 0.271 | 0.314 |
| Strong current | BCA | 0.192 | 0.227 | 0.216 | 0.224 |

Values are equal-seed means of mean regret to the best action in the **six-action bank**, in discounted-return units. The combined score did not consistently exceed the controls. In particular, strong-current BCA seed 1 regret increased from 0.191 raw to 0.257 combined; seed 2 increased from 0.248 to 0.308. These are forced-first-action diagnostics, not whole-policy return or global-optimum regret.

The frozen screen required at least two qualifying seeds in both saved-policy groups in one context. Ordinary had 2/3 qualifying native seeds but 0/3 BCA seeds; strong current had 1/3 native and 0/3 BCA. Neither context passed. The full per-seed results, all-action and stratified error changes, visited-state errors, per-probe scores, gradients and calibration values are in [RESULT.json](diagnostic_run_01/RESULT.json).

## Execution and implementation boundaries

- The fresh Q1-centered fitting and calibration paths are in [q1_bca.py](q1_bca.py); [compatibility.py](compatibility.py) implements the frozen nearest-state, four-action KDE-like cost; [actor_penalty.py](actor_penalty.py) binds both terms into the pinned actor-loss factory without changing critic targets.
- The diagnostic completed 24,000 auxiliary updates and 12 calibrations. It performed **zero actor, critic or target optimizer updates and zero simulator transitions**. No full training study ran.
- CPU execution used the authenticated saved GPU normalization. Across all contexts, the largest CPU-versus-saved-GPU scaled parity discrepancy for actor/Q1/original-width checks was 2.24×10⁻⁶, below the declared 10⁻⁵ threshold. A bounded real-input check found identical neighbor IDs for 64 evaluation anchors and 64 cross-fitted training probes; that does not establish parity for every state or GPU execution.
- The compatibility metric is an action-density heuristic, not a verified local support boundary. It can penalize useful unseen actions. The original frozen learner's batch API does not itself carry episode IDs, so training-time use must explicitly bind an episode-excluding or fold-specific neighbor reference. This diagnostic used cross-fitted training probes and disjoint saved evaluation anchors; it is not a completed full-training integration.
- The method was tested on three seed initializations, one fixed dataset/removal mask, and previously inspected simulator references. The [protocol](DIAGNOSTIC_PROTOCOL.md), [integrity seal](DEV_SEAL.json), [terminal result](diagnostic_run_01/RESULT.json), per-case files and 12 fitted auxiliary artifacts preserve the exact scope and provenance.
