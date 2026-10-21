# Independent distribution and action-only plan review

Reviewed on Zeus, 2026-10-21. **Approved for delivery: the v2 source-to-target explanation and proposed action-only diagnostic match the inspected implementation and preserve the earlier results. No material finding remains in this clarification.** This is approval of the description and plan, not evidence that the proposed diagnostic has been executed or that its coverage guarantees hold.

The signed-off v1 HTML and `results_20261021/review.md` were not modified. No training, calibration rerun, policy rollout, Titan access, or new performance experiment was performed during this review.

## Source-to-target labels

Independently traced the actual continuous training path, rather than the separate generic uniform-action generator:

- `training_data.py:60-80,110-117` defines five atoms per actuator and a product categorical behavior law over 625 commands. The nearest knee-clipped 0.6-LQR atom has coordinate mass 0.84; each other atom has mass 0.04. These are masses, not continuous densities. `continuous.py:31` defines the knee as the float32 representation of 0.8.
- `continuous_data.py:32-34` and `training_data.py:171-191` make retention depend on **high current AND any above-knee action**. Inclusion is 1/0.05/0 for that intersection in full/rare/zero and 1 otherwise. Deletion preserves the already-generated successor states. It does not regenerate trajectories under a different policy.
- `training_data.py:95-108` reserves 128 whole episodes before retention from a balanced 1,024-episode source. `training.py:223-247` independently seeds a uniform retained-row selection within each reserved episode. Conditional on realized trajectories and masks, the selection law weights episodes equally and rows within an episode by its inverse retained count. It is not automatically the pooled-row behavior-times-retention law. All reserved episodes in this run contribute one row, but their realized context counts are not 32 each.
- `training_eval.py:45-95,267-275` constructs 16 reset-law physical states, repeats them at times 0/50/99, and evaluates six candidate commands at each of the 48 observations. The later time labels do not mean the physical states were evolved to those times. The five fixed commands already change the conditional action law; the policy-specific sixth command can be off the behavior grid. The 288 pairs are a designed grid, not 288 independent probability draws or an actor-occupancy sample.
- `training.py:491-520`, `algorithms/td3_bc_bca.py:146-156,234-270`, and `training_calibration.py:70-91,102-117` confirm that the reference is last refreshed after 95,000 accepted updates, while the terminal residual uses the online and target networks at 100,000. The frozen scale, unit and threshold are tested against a potentially changed model residual. Physical dynamics/rewards are unchanged; independently drawn evaluation noise uses the same noise law.

Consequently, the v2 label **joint state/time and action mismatch, plus model-residual drift** is accurate. Full retention does not make this comparison IID. The coverage graphic does not separately identify state-only, action-only, or predictor-age effects. Its 288-pair query bank is correctly distinguished from the separate 400-start return bank. The description also retains the adaptive-bank and categorical-support limitations.

## Proposed action-only diagnostic

The final plan fixes the available full-retention BCA replicate 01 at 100k, its online/target networks, normalizer, frozen scale/unit, and native conditional label-noise mechanism. It recomputes calibration scores against the same fixed predictor and target kernel; it does not compare old 95k calibration scores with 100k scores as an action-only change.

A frozen 64-row state/time bank defines a finite law mu. The proposed 128 calibration draws sample state IDs independently from mu, then draw actions from the actual categorical behavior and label noise independently. Evaluation uses independent draws from the same mu, shared across the compared methods. Conditional on the bank and frozen functions, this supplies a common source-law sampling design. It does not turn the 64 historical episodes into a fresh population holdout or establish training-seed generalization.

The target mixtures at lambda = 0, 0.5, and 1 have exact masses `(1-lambda)*b + lambda*delta_h`, where h is the fixed actor projected onto the same grid. Their ratios are exactly q/b because the state law and conditional label kernel are unchanged. The projected target is explicitly distinguished from the unrounded deployed actor. The plan correctly separates structural off-grid support failure from a finite calibration sample missing a supported grid atom, preserves true zero weights, requires refusal for an empty positive-weight bank, and reports ESS/refusal without treating them as coverage certificates.

Both weighting arms share calibration scores, query draws, posterior streams, and the native neutral refusal policy. Clipped next-action noise belongs to the common label kernel and is not assigned an action-density ratio. The optional retention extension explicitly defines a **new conditional action sampler** at fixed mu, not the law of the old equal-episode retained bank. No occupancy-ratio claim is made. Finite action enumeration, label-noise Monte Carlo error, conditional inference, and any later episode-population inference are properly separated. The plan remains post-hoc and unexecuted.

## Final artifact verification

All 13 v2 build-manifest source hashes match the current files. Independently checked that all 194 v1 table blocks, all 30 script blocks, and all 545 SVG blocks are preserved exactly. V2 adds two tables and one diagram. All 3,796 prior IDs remain; the 3,800 total IDs are unique, and all 768 internal links resolve. Thus the previously reviewed numerical tables remain unchanged. The embedded distribution audit and final action-only plan match their declared versions.

| Artifact | SHA256 |
|---|---|
| `results_20261021/analysis/DISTRIBUTION_SHIFT_AUDIT.md` | `70433ca851d62f35fbaa5df5b42e52929aa34b51fe02969b914fba50c73963d4` |
| `results_20261021/module_fixtures/ACTION_ONLY_POSTHOC_PLAN.md` | `ae54d8f259ae70abcdc663f3a255753c6232695303e45118ed1b28f07a8f1e0c` |
| `results_20261021/presentation/ThrusterChain_diagnostics_results_v2.html` | `cbb9e49f0324547443a5b3c2f587438b65f11ea240193d55754add643badba7f` |
| Preserved v1 HTML | `524f64418fab2c0d708078daada37704be8f76c2ad3ff203e85ddfab1398a34d` |
| Preserved v1 review | `b7a4a0532ccefbf6f14d4cae917878d5de16fe633699bb02f5da20e02ca9beef` |

V2 HTML size: 11,358,393 bytes. Its build manifest records 19 JavaScript syntax checks and 24 filter mock-DOM combinations performed by the guide author. Live browser rendering/navigation remain unverified. The earlier historical clipped-command-ratio defect remains disclosed; this clarification neither repairs nor validates that branch. No unresolved blocker remains to delivering this v2 explanation and plan.
