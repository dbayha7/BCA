# V6 bounded validation — 2026-10-26

## Source and preservation

The v6 code/test/contract manifest contains 78 files with aggregate SHA-256
`e2f4e56c0cafee494145c292781ac4b9fd4761fa1463c9ebead68e6418d9a7e5`.
`../CURRENT_SOURCE_MANIFEST.json` matches the source snapshot recorded before
the integrated smoke. The `run_bounded_smoke.py` hash is separately recorded in
the smoke receipt, since the manifest's source scope does not include validation
scripts. All 71 v5 and all 52 v4 manifest entries matched their original source
hashes after this work. The pinned unified source also retained its recorded
`14a9bcdbc6d96fad85544a9d98957e98a7e46db13b89d74d54e3c5ce246517c8` hash.
These checks do not verify every historical result or untracked asset.

## Tests and integrated smoke

- On Windows with the repository Python 3.10 interpreter, `validate_v6.py`
  ran 241 tests: 178 passed and 63 were explicitly skipped. The skipped tests
  are historical-v4 artifact classes requiring missing saved results and the
  original v4 loader; they are not v6 passes. The new tests cover all behavior sources, deterministic
  replay and group linkage, save/load and tampering, CLI dataset dispatch,
  terminal/timeout/next-action masks, all new shifts, checkpoint-bound CLI
  evaluation and refusal of a continuing-task checkpoint for finite scenarios.
- A separate read-only CLI dispatch check resolved all five neural adapter
  names and saved-checkpoint identities to their matching factories.
- On WSL Ubuntu-22.04 CPU, 25 bounded host tests passed across the four native
  offline hosts and the pinned unified UniFloral composition. Unified tests
  include fixed-batch comparison to the pinned internal sampler for full model,
  optimizer, target and RNG state; independent actor-loss and critic-target
  calculations; terminal and timeout semantics; the actor-first one-actor/two-
  critic schedule; exact checkpoint resumption; and frozen evaluation.
- The predeclared [plan](PLAN.md) was executed via the actual CLI paths. A new
  ten-episode `feedback_strong` smoke dataset was generated and validated;
  unified training used two primary actor updates and four critic updates, with
  no training-time simulator interactions. The saved checkpoint was reloaded
  and evaluated for one 100-step episode in each of the four new scenarios.
  All four completed and the frozen policy state hash stayed unchanged. The
  [machine-readable receipt](unified_smoke_02/receipt.json) records dataset,
  checkpoint, scenario, source and script identities. This is a wiring check,
  not a comparison of policy returns or evidence of generalization.

The earlier `unified_smoke_01` receipt remains an intermediate validation of a
different source snapshot. It preceded the v6-specific standalone-adapter
identity and dataset-label changes; it is not presented as evidence for the
final snapshot. The bounded smoke was repeated with the same predeclared limits.

The unified test runtime was Python 3.10.12, JAX 0.6.2, NumPy 2.2.6,
Flax 0.10.7, Optax 0.2.8, Distrax 0.1.5 and TensorFlow Probability 0.25.0.
Actual Distrax was imported from a second existing site-packages directory
appended after the active environment. JAX and NumPy are outside the pinned
upstream bounds; an explicit opt-in was recorded. The compatibility flag checks
only those two bounds, not the complete dependency stack. No GPU validation or
research-scale performance test was performed.

## Limits

Unified support is one deterministic TD3+BC-style composition, not arbitrary
unified configurations. Native-update parity is conditional on a matched batch;
the runner's sampling stream is different from the upstream sampler. New
scenarios are finite-horizon synthetic shifts; their identity is latent and
they do not establish real-world transfer. ReBRAC cannot use bootstrapped rows
without an observed next action. Mixture calibration is descriptive, and
declared seeds do not attest an unseen holdout. No full experiment, tuning sweep,
or scientific performance comparison was run.
