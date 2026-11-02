# Separately scoped checkpoint-generation proposal

Status: proposed only. No training, data generation, installation or exporter execution is authorized by this document. The full experiment remains paused. The frozen evaluator currently lacks an authentic trained UniFloral ThrusterChain checkpoint.

## Recommended first bounded stage

Prepare and review a one-seed native UniFloral TD3+BC host with kappa_penalty=0 and the actual auxiliary width-fitting/calibration pipeline. This tests whether the proxy is informative on a trained baseline before changing actor learning with a nonzero penalty. It is not a four-arm comparison or an efficacy experiment.

Proposed terminal budget: **20,000 actual primary critic optimizer updates**, corresponding to **10,000 native outer steps and actor updates**. Preserve two critic updates on the same native minibatch, then the actor and target updates, native RNG order, initialization, observation normalization, Adam epsilon 1e-5, reductions and Q normalization denominator epsilon 1e-7. Task-specific overrides are gamma 0.98, horizon 100, time-aware 14D observations, raw rewards and zero terminal bootstrap; this is not a published benchmark reproduction.

The current legacy auxiliary warmup/refresh plan suggests a first reference at 10,000 critic updates, then refreshes at 15,000 and 20,000. Those are proposed actual-critic-update milestones, not outer-loop counts. Freeze the prespecified terminal checkpoint at 20,000 irrespective of diagnostic returns. No evaluator oracle, final-evaluation reward or checkpoint ranking enters learner/calibrator selection.

This budget is an operational starting point that crosses warmup and subsequent refreshes, not a statistical minimum or a guarantee of a useful policy. Before proceeding, the combined scheduler must have a bounded parity test showing the exact native RNG, repeated minibatch and actor-delay semantics. The existing synthetic microstep and frozen-pipeline fixtures do not yet constitute that production trainer or its checkpoint exporter.

## Required evidence within that budget

- Verified native source/config and data identities; exact actual critic, actor and auxiliary optimizer counters, including attempts/rejections separately.
- Genuine auxiliary fitting on training groups, finite-ready calibration on at least 128 distinct held-out episode groups, and a subsequent real refresh after further learning.
- Explicit invalid-reference status instead of silent zero-penalty success. Production handling is a design decision to review; the evaluator's rejection does not patch the old learner.
- Frozen terminal actor/critic/reference, exact native normalization, real group/seed provenance and a 256-row fitting-only scale bank; an export tested against the original native inference outputs.
- Resume evidence retaining phase, minibatch, RNG, optimizers, auxiliary state and calibration banks. The evaluator-only export is not a resumable training checkpoint.

If these milestones fail within the cap, stop and report insufficient diagnostic readiness. Do not automatically extend training, pick a favorable earlier checkpoint or interpret failure as evidence for or against BCA efficacy. No universal update count guarantees readiness, and no throughput or duration estimate has been measured for this proposed stage.

## Dataset option, pending independent verification

The existing 1,280-episode recorded dataset is algorithm-independent and may be reusable after checking its original provenance, groups and complete per-episode seeds. This does not permit reusing CORL policy weights. Expected dataset digest: `6b58db68263dff0d7e9b8d6dfa7c1ca5d4127ed9ee6a3b9069add1c565e88331`; file SHA256: `4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226`.

Location: the transfer bundle's `research_runs/pilot_20261023T115408Z/regular_1280_2026102301.npz`. Previously recorded split counts are train 768, tuning_validation 256, selection 128 and final_calibration 128. An explicitly declared subset of at least 128 distinct tuning-validation episodes could supply the training-influential reference calibration, separate from fitting. Selection and final-calibration groups remain untouched. This reuse and split assignment have not been performed here.

If the source data do not preserve enough exact episode-seed/group provenance for the export contract, report that gap before changing the protocol or generating replacements. Development seeds 2026102400 through 2026102463 and reserved confirmation seeds 2026102500 through 2026102563 are excluded from training and calibration.

The next decision is approval of this bounded checkpoint stage and its real scheduler/export implementation. The evaluator can then be runtime-tested on the resulting reviewed artifact. No full-budget restart is implied.
