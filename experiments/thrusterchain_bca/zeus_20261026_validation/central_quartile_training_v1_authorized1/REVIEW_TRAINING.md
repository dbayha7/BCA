# Inherited original-package review

This review body describes central_quartile_training_v1 before its failed target validation. It is retained for provenance, not as amendment verification. Read AMENDMENT_REVIEW.md, AMENDMENT_RECONCILIATION.md and TEST_RESULTS.json for current status. Training remains held.

# Training-path review

This review covers the unexecuted `central_quartile_training_v1` package. It does not alter or reopen the completed matched-policy pilot or any design-gate reservation. Production execution remains disabled. Actual target GPU initialization and the bounded runtime checks remain unverified here.

## Confirmed source properties

- `data.py` reads only the original 768 training and 128 calibration episodes through the role-safe reader. It selects intact tuples, keeps source identities, and rejects calibration/replay overlap. The saved plan has 76,800 full rows and 67,108 rows in each removal condition.
- Common normalization is fitted on 60,924 training observations retained by both hole and random controls. The identical raw-observation fixture is passed to both pinned native and auxiliary initializers; their statistics are not silently changed after initialization. Condition-specific retained replay, not the fixture, enters updates.
- All conditions share the same 128 non-hole calibration rows. Thirteen original selections are replaced uniformly within the same episode from eligible non-hole rows. Empty eligible episodes stop the study. This is outcome-independent selection, not fresh independent calibration: episodes are development-consumed and reused adaptively after their thresholds affect training. There is no conditional hole-coverage guarantee.
- Each condition initializes fresh parameters, optimizer states and RNG before its own 10k warmup. The branches restore their own complete condition/seed warmup, then fold the declared branch RNG streams identically across arms. No broad trained checkpoint is reused for the hole arm.
- Native, BCA and distance branches carry matched auxiliary fitting and one 15k recalibration. Those computations have no actor-gradient path in the native arm. This controls shadow-work accounting but makes the native baseline costlier than ordinary native TD3+BC.
- The pinned actor objective is `−lambda*mean(Q1) + mean(BC) + lambda*kappa*mean(width)`, with `lambda=stop_gradient(2.5/(mean(abs(Q1))+1e-7))`. The BC term is unchanged. Uncertainty parameters/reference are stopped; action gradients remain live. The critic target retains the native smoothed target actor and twin-min operator.
- The distance penalty is tied to the frozen 10k actor. Its coefficient is selected once by RMS action-gradient matching on retained logged probes. This is not matching on-policy gradient strength, actor-parameter gradient direction or Bayesian contribution. At the fork actor its action gradient is exactly zero.

## Findings and resolutions

The paired-initialization gate originally checked fresh parameter hashes and optimizer counters while excluding normalization. That would have accepted a semantically different, hash-valid normalization receipt. `launch.verify_warmup_pairs` now authenticates supervisor/report/actor/data receipts and compares common normalization, its row/observation provenance, initial optimizer hashes and cell identities. The five receipt-integration tests include validly resealed but wrong statistics, wrong row provenance, wrong optimizer state and a copied-cell report. All are rejected.

Symmetric retained action tails have conditional population mean zero under the state-independent recorder. Squared behavior cloning therefore has an attraction toward the central hole even without a critic-driven preference. Finite logged probes need not have exactly zero sample mean. Free actor exposure alone cannot identify critic exploitation.

`gradient_audit.py` adds a fixed training-only decomposition at 10k warmup and 15k/20k branches. It uses the existing 768 retained probes in three 256-row batches, preserving the actor objective's batch-specific stopped lambda. It reports Q, BC and applied-penalty actor-parameter gradients, norms and cosines, plus per-row action-space gradients and inward/outward signs. A prospective κ=3 width gradient in native/distance arms is explicitly shadow information, not their applied penalty. At calibration boundaries it describes the frozen objective the next actor update would use; the preceding actor update used the old reference. No optimizer, calibration, RNG or simulator update is dispatched by this diagnostic.

The diagnostic performs four logical forward/autodiff row-dispatches per unique probe, plus a frozen-anchor actor forward in distance arms. Across 63 snapshots this is 48,384 distinct probe evaluations, 193,536 base logical dispatch-row uses, and 13,824 extra distance-anchor rows. These are logical batch operations, not an estimate of internal autodiff primitives or FLOPs. The caller must enforce the 63-snapshot limit and retain attempted/completed diagnostic counts on failure.

## Residual confounds and limits

The random control matches source episode × ten-step time bin × observed-load side. It does not exactly match every state, reward, next state or terminal label. The saved retained terminal-row counts are full 768, hole 653, random 663. This must be disclosed; changing the random mask now would be a new design choice, not a bug fix. The protocol correctly avoids calling this an isolated action-only causal experiment.

Three fresh initialization seeds remain an exploratory sample. One dataset and one random-removal realization do not estimate dataset/mask variability. Calibration reuse and fixed strength selection remain development choices. The current actor gradients diagnose the frozen sampled objective; they do not prove the optimizer or subsequent trajectories follow those directions.

Full-state serialization is exercised with toy states, including optimizer momentum, RNG, odd phase/held minibatch and a refresh boundary. This does not replace an actual pinned-model target-runtime restore check. Checkpoint identity verification is also separate from proof that a target GPU numerical path matches CPU fixtures.

## Accounting and tests

Nine 10k warmups plus twenty-seven 10k branches imply 360,000 critic updates, 180,000 actor updates, 360,000 auxiliary updates and 36 calibrations. There are 369 immutable 1k-boundary full-state snapshots including nine initial states, plus 36 duplicated final-state exports. At the previous serialized state size of 10,350,823 bytes, full states alone occupy approximately 4.19 GB (3.90 GiB), before actor copies, gradients and evaluation artifacts. At least 6 GiB free workspace headroom is a planning estimate, not a measured new-run requirement.

Command run with the existing Windows CPU environment:

`C:/Users/David/dev/BCA/.venv/Scripts/python.exe -B -m unittest test_gradient_audit test_training_integration -v`

**12 tests passed in 10.589 seconds.** These use synthetic arrays, toy actor/critic functions, the pinned actor-loss code and synthetic receipt files. They perform no production model load, dataset decode, simulator transition or WSL/GPU call. The gradient tests also verify sum-of-components against the pinned total actor gradient, zero fork-distance gradient, immutable state, source-ID rejection and prospective diagnostic failure accounting.

Remaining integration responsibilities: connect the diagnostic to the worker with immutable receipts and live progress; freeze and verify all dependency hashes; run the combined permitted CPU suite; then perform the separately capped manual target-runtime validation. Production budget approval remains separate and cannot be inferred from these tests.

