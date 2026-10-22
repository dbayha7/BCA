# Direct action-value penalty: next-step plan

The goal is to make uncertainty reduce the value assigned to an unfamiliar action, so it can change which action the policy learns to take.

The current native BCA implementation instead increases the strength of behavior cloning. That is a useful separate experiment, but it does not establish the direct action-value idea. The completed 30-job study and action-only calibration study remain unchanged.

The next step is small:

1. Prove the penalty changes the intended action gradient, using deterministic fixtures.
2. If those pass, test a simple Q-minus-width objective on disposable copies of one frozen checkpoint, with a baseline, a matched constant-strength penalty, and gradient controls.
3. Only then specify a learned-policy comparison.

Update after the user's execution request: the isolated mechanical stage is complete. All18 tests passed. A separate real-checkpoint audit shows that the proposed live width changes the actor gradient while the native exported width is detached. See [mechanics/REPORT.md](mechanics/REPORT.md). No training occurred. A separate bounded frozen-policy ranking diagnostic is being prepared before any actor-update pilot.

Option C is a different, paired proposal with its own query action and refusal rule; it is not another name for simple Q-minus-width. The current action-only grid also does not certify the continuous actor's off-grid actions.

Coverage at 80%, 90%, 95%, and 99% is a separate sensitivity question. Higher requested coverage may widen bands or cause refusal; it should not be selected just because it produces a desirable result. Bayesian credibility, nominal coverage, and the TD3+BC objective coefficient are separate settings.

Read [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the proposed pilot and decision rules, [SOURCE_AND_FIXTURES.md](SOURCE_AND_FIXTURES.md) for source-mapped pseudocode and expected gradient checks, and [AUDIT_RECORD.md](AUDIT_RECORD.md) for versions and delivery status.

Status: mechanical stage executed; later pilot and coverage sensitivity remain proposed, 2026-10-21. This plan is not a retrospective preregistration. Each future diagnostic protocol must be reviewed and frozen before execution.
