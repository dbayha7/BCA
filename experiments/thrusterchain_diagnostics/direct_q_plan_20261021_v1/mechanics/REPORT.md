# Direct-Q mechanical stage — verified result

The proposed simple Q-minus-width objective has the intended gradient path. This does not establish that the width is accurate, calibrated at actor actions, or beneficial to a learned policy.

18/18 deterministic tests passed, zero failures/errors/skips. Runtime1.485seconds on Zeus CPU. The tests include zero-penalty equivalence, live/detached gradients, raw-Q scaling, finite differences with a frozen normalizer, parameter ownership, same-state scalar SGD identity, shapes, overflow/refusal, a width floor, and examples where the same ranking change helps or harms.

At theta=.4 in the analytic fixture:
- Host and fully detached penalty gradient: -.2416666666666667.
- Proposed live-width gradient:1.8416666666666668.
- Frozen-normalizer finite difference:1.8416666666523088.
The scalar SGD fixture is a unit-test calculation, not a trained-policy update.

## Actual native calibrator audit

Checkpoint: original full-retention BCA replicate01 terminal100k, SHA2564088278230fc413c509e27e189daa342468cfc4fb6d2479e3e8c29e149dac3d8. Native frozen reference:95k. The audit loaded the authenticated preserved source closure. It did not refresh calibration, update optimizers, train, or roll out a policy.

| Quantity | Observed |
|---|---:|
| Fixed previously studied states |64|
| Proposed raw-width range |1.34153593–3.99818754|
| States with nonzero proposed width action derivative |64/64|
| Native exported width action-gradient norm |0|
| Proposed width action-gradient norm |2.48940289|
| Host actor-parameter gradient norm |.551228457|
| Live-width actor-parameter gradient norm |.554201384|
| Norm of live-minus-host actor gradient |.0166921241|
| Norm of detached-minus-host actor gradient |0 exactly|
| Off-grid actor actions |64/64|

The norm of the gradient difference is not the difference of the two gradient norms.

The native frozen_level_dose helper explicitly detaches every returned value. The proposed candidate instead recomputes R*unit*max(eta(s,a),floor) with the SAME frozen network/reference values and retains the derivative through action. Forward width values matched native output within declared float32 arithmetic tolerance (rtol2e-6,atol1e-5). This is a proposed differentiation change; it is not evidence that current native additive width already affects the actor gradient. Existing native BCA still affects the BC term through its detached multiplier.

With a scalar frozen R*unit, all action-dependent geometry comes from eta. This is not per-action IW threshold adaptation.

Directional finite differences of the SUM OF WIDTHS with respect to action used all three predeclared step sizes:
- .003: relative error .00579%.
- .001: relative error .06165%.
- .0003: relative error .01257%.
All plus/minus perturbations stayed inside[-1,1]. This checks the width's action derivative, not a full actor-parameter-loss finite difference. Synthetic tests cover the latter scalar loss structure with the normalizer frozen.

The frozen model/reference/state hash was unchanged. The current actual-checkpoint receipt is checkpoint_evidence_v2/results.json, SHA25682325a829fb4f6bc3da01ba4aef900307445cd46787410fc1a9bb5bb830ba113. The initial receipt and exact executed audit source are retained in checkpoint_evidence. V2 added action-bound checks, selected-bank identity and engine/source hashes; numerical results were unchanged. The original18-test receipt includes an earlier audit-script hash because that script was subsequently improved; the test runner did NOT execute the checkpoint audit.

All64 actor actions were outside the625-command categorical logger's support. These finite extrapolated widths and derivatives are not a support certificate, true-Q interval, or overlap-valid IWBCA result.

## Source-to-stage map

| Stage | Implemented source | Evidence |
|---|---|---|
| M1 freeze parameters, preserve action differentiation |objective.py:24; audit_checkpoint_width.py raw_width/eta definitions |synthetic parameter gradients zero; actual model unchanged |
| M2 validate shapes, consumed widths and derived arithmetic |objective.py:29 |invalid,zero,overflow cases |
| M3 raw-Q stopped normalizer |objective.py:57 |mixed-sign Q fixture and normalizer finite-difference countercheck |
| M4 proposed live/detached/constant/zero score |objective.py:62 |gradient and ranking controls |
| M5 actor loss with unchanged ordinary BC |objective.py:78 |analytic/same-state scalar-update checks |
| M6 actual frozen-width numerical/gradient comparison |audit_checkpoint_width.py |checkpoint_evidence_v2/results.json and per_state_gradients.npz |

Independent scientific/code review found and corrected a derivative-label typo, rollout-budget omission, overbroad no-op diagnosis, matching-strength wording and arithmetic-preflight gaps. A second read-only reviewer checked the final code and receipts, including source hashes and perturbation bounds. No duplicate reviewer execution occurred.

Unrun: Option C's paired/noisy/refusal objective; actual host critic/target-update isolation; a real actor optimizer step; multi-step learning; policy benefit; empirical calibration sensitivity. The frozen ranking diagnostic is a separate next stage with independently seeded evaluator-only samples and its own freeze.

## Reproduce the mechanical tests

From this directory, using the existing project environment:

    C:\Users\David\dev\BCA\.venv\Scripts\python.exe -B -m unittest discover -s . -p test_objective.py -v

The receipt-writing runner refuses to overwrite evidence. To rerun the actual checkpoint audit in the ORIGINAL WORKSPACE COPY, preserving earlier receipts:

    C:\Users\David\dev\BCA\.venv\Scripts\python.exe -B audit_checkpoint_width.py --evidence-name checkpoint_evidence_replay

Choose a new evidence directory name. The audit uses the adjacent preserved source release and canonical checkpoint; the project copy is a source/evidence deliverable and is not a self-contained checkpoint archive. No dependency install is required.

The initialization reports there is no published reference configuration for this new ThrusterChain cell. This audit does not claim to reproduce a published benchmark score.
