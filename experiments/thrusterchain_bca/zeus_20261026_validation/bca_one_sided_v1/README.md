# BCA one-sided auxiliary validation

This is one bounded, exploratory adaptation of BCA's uncertainty target. It preserves the local positive scale architecture and Bayesian/conformal calibration; it is not a supervised return-error correction. User execution is authorized. No real fitting, operational calibration or model restoration was performed during preparation.

The earlier cached ideal lower-only penalty improved saved continuation values in both development batches. This run tests whether a BCA scale learned from **logged transitions**, with separate calibration, can preserve useful action choices. The ideal result does not establish that this learned bound will work.

## Fixed change

Keep the exact transferred standalone UniFloral checkpoint, actor/current and target critics, target actor, BC term, normalization, discount, masks and native smoothing law. Original BCA remains unchanged. Copy the existing auxiliary parameters into a separate 64-by-64 positive scale network with a fresh Adam optimizer; freeze the original residual unit for this short pilot.

For each logged row, compute the native smoothed target y using its recorded reward and next observation. Fit the one-sided event `y >= Q1 - unit*eta`:

```text
coverage = sigmoid(20 * (y/unit - Q1/unit + eta))
loss = (sum(w*coverage) - 0.9)^2 + 0.005*sum(w*eta^2)
```

Each update uses normalized exponential Bayesian-bootstrap weights w. Q1, y, unit and w are detached; only copied eta parameters update. The float32 operation order above is fixed. The architecture retains `softplus(log_eta)+0.01`; public scale uses `unit*max(eta,1e-6)`. Negative Q1 excess is covered by the one-sided event; calibration clips its nonconformity score at zero without inventing a positive residual floor.

After fitting, freeze eta and calibrate scores `max(Q1-y,0)/(unit*max(eta,1e-6))` on separate logged episodes. Retain the pinned Bayesian/conformal calibration mechanism, alpha 0.1, credibility 0.95 and 1,000 posterior draws. This posterior concerns residual-score miscoverage risk, not Q-network weights or true return uncertainty. The actor/candidate score remains `Q1 - kappa*U`, with fixed kappa 1 and `U = threshold*unit*eta`.

The current task makes no actor update. Synthetic gradient tests verify that a later actor integration can freeze scale parameters while preserving gradients through the action input. Infinite calibration causes refusal; zero threshold is valid but degenerate, with no claimed penalty benefit.

## Data and budget

The pinned logged dataset is `regular_1280_2026102301.npz`, SHA-256 `4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226`.

| Role | Rows | Episodes | Use |
|---|---:|---:|---|
| Exact original fitting bank | 256 | 221 | Auxiliary scale adaptation |
| Exact original calibration bank | 128 | 128 | One posterior calibration |
| Remaining tuning-validation episodes | 128 | 128 | Local validation, one fixed row per episode |

All three banks are episode-disjoint. The loader excludes reserved dataset selection/final-calibration transition records and completed confirmation outputs. The restoration template reads the permitted original 76,800 training rows to reproduce its original normalization and structure, then exactly restores the checkpoint; those rows are not additional adaptation samples. Fitting/calibration never use evaluator candidate transitions or saved continuation values.

The calibration bank was used by earlier original-BCA refreshes. This is disclosed adaptive reuse, not fresh confirmation. The new local validation bank is separate from the current fit/calibration banks; prior global inspection history is not claimed absent. All development evaluation remains exploratory.

- Exactly **128 auxiliary updates**, full batch **256**, Adam learning rate **0.001**.
- One root seed **2026102671**, one checkpoint; one fixed native-law smoothing draw per logged bank.
- Exactly **one** posterior calibration, **1,000** posterior draws.
- **300-second worker wall-time cap**, including restoration and compilation. No automatic extension or retry.
- Logged forwards: 512 current-critic rows, 512 target-actor rows, 512 target-critic rows. Auxiliary work: 32,768 gradient rows, 896 diagnostic rows, 1,152 development scale rows. Template-validation work is separate and disclosed.
- **Zero** actor, critic or target optimizer updates; zero simulator transitions, new rollouts, new candidates or confirmation reads.

The fit budget is 0.64% of the original 20,000 auxiliary updates. Warm-starting the same architecture makes this a small adaptation screen, not a convergence study or hyperparameter search. Labels are fixed during the fit; there is no outcome-based early stopping.

## Readouts and decision

Report pre/post local objective, directional coverage and widths on fitting and logged validation data; then calibrated coverage, widths, zero-threshold behavior and refusal. Calibration-set coverage is descriptive; marginal calibration does not certify adaptively selected actions, OOD actions, or true continuation value V. Revalidate dependence/exchangeability or justified shift weighting, score freezing, data separation, smoothing/terminal laws, adaptive reuse and posterior approximation before any coverage claim.

Only after fitting/calibration are sealed, score the existing 128-by-nine development bank. Seal all choices before reading saved V. Report every batch/time stratum, paired value gains, regret, harmful switches and numerical ambiguities.

Fixed controls are original BCA at kappa 1, Q1, the frozen actor, actor-distance `Q1 - 0.1*19.29619572569095*mean((a-actor)^2)`, one-sided ordinary-quantile and Bayesian thresholds, raw local scale, and effective-strength-matched controls. The actor-distance strength is retained from the prior reviewed control. Matching global `kappa*threshold` on the same local scale must give the same ranking; global threshold changes alone are not new action information. Undefined zero-threshold matching is reported, never repaired by division or tuning.

The exploratory utility screen requires a finite nonzero Bayesian threshold and mean paired saved-V gain above the mean numerical tie band versus **both Q1 and original BCA, in both development batches**. Actor-distance comparisons are mandatory descriptive readouts. A failed screen stops automatic follow-up; it neither proves BCA impossible nor authorizes a larger run. No completed confirmation data or coefficient sweep is allowed. Full training remains paused.

## Prepared checks and manual execution

Preparation passed **16 mathematical/gradient fixtures, 8 data/leakage fixtures and 5 launcher fixtures**. These include one-sided signs, stopped model/target gradients, live action gradients through frozen scale parameters, JIT/vmap/scan, positive scale floors, zero/infinite calibration, scaling, dataset leakage rejection, exclusive reservation, timeout and interruption cleanup. Operational GPU restoration and Flax integration remain runtime checks performed before real updates; they were not tested by launching the real experiment here.

Run once manually in PowerShell, from any directory:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/bca_one_sided_v1/launch.py --execute
```

This uses the existing isolated Ubuntu environment and frozen GPU package versions. The previously disclosed outside-upstream JAX/NumPy runtime exception remains; no package or environment modification is performed.

The launcher verifies the prepared file hashes, checks this user's scoped audit processes, atomically creates `ATTEMPT.json`, and launches one capped worker. The original checkpoint and sources are checked before/after. The new auxiliary parameters and reference are saved separately under `run/`. Timeout, failure or interruption preserves evidence and prohibits an automatic retry.

Terminal supervisor evidence is `RUN_RESULT.json`. Scientific outputs are `run/report.json`, `run/stages.jsonl`, `run/worker.log`, logged-label and calibration arrays, separate auxiliary parameters, and the sealed development selections/results. If interrupted before a complete report, counts are explicitly unknown rather than fabricated. Do not remove the reservation or rerun to repair an uncertain outcome.

The assistant cannot invoke WSL because that access remains denied. This manual command is the remaining execution step, not an additional permission request. No commits or pushes were made.
