# Method-change design notes (Track B), 2026-10-31

These notes record what the design panel (`wf_bdf1de0d-a7e`, with the user's rule "Make sure we are acting on q values"
in every design, judging and checking prompt) produced, and what its checkers found. Nothing here is built.

## Why BCA as built cannot act where the critic is wrong (investigation, with code evidence)

U = R x u x eta(s, a). R and u are scalars, so all action leverage comes from eta, which is near-flat in the action
(|dU/da0| about 3% of |dQ1/da0|; moving from the logged to the actor's action changes U by about 1-2% of its level).
The causes, strongest first:
1. **The target.** eta is fit to the size of the twin-min one-step residual, on the very rows the larger critic just
   fit, so what remains is mostly noise with no action shape. A critic that over-values past the knee is consistent
   with itself one step at a time.
2. **No within-state contrast.** eta is fit only at logged actions, one per state, so it never compares two actions
   at the same state.
3. **The objective.** The coverage loss with a width penalty constrains only the batch-average coverage.
4. **Calibration.** R is a single number, so it adds no resolution.
5. **Minor causes.** Twin-min versus Q1, input scaling, and capacity are minor or not causes.

Raising kappa only scales a slope with no consistent sign off support.

## The recommended variant (BCA-PE) and what its checkers found

- **Proposal.** Fit a signed head m(s, a) to Q1's propagated Bellman error, with label
  delta1 = Q1(s,a) - r - gamma Q1(s', pi(s')), target delta1 + gamma m_bar(s', pi(s')), and U = m + c. The actor
  maximizes Q1 - kappa U.
- **What the checkers found** (both rejected it as written):
  - **The full propagated E is not identifiable from the data.** Under the actor's unclipped continuation, E is about
    200 in the stall world, built along off-data paths. The head bootstraps through its own extrapolation at
    (s', pi(s')), so it inherits the critic's problem.
  - **c is a constant** and drops out of every ranking. "Calibrated lower bound" was unsupported. An anchor-coverage
    clause is needed, or U needs an action-dependent spread (bootstrap heads).
  - **The decision rule could FAIL a perfect estimator.** The clipped and unclipped references disagree on no-stall
    false flips. One reference must be fixed per clause, and the rule must be checked on hypothetical estimators
    first: an oracle must PASS, raw Q1 must FAIL, and a uniform action ramp must FAIL.
  - **Leverage readouts on the actor's past-knee rows mostly measure depth.** The decision should use onset rows only,
    and correlations should be controlled for depth.
  - **No test of placement across states within a world.** Even caution could tie it. A within-world Spearman of
    kappa x dU(s10) with KC(s10) is needed, and the method must beat a matched even-caution ramp.
  - **The stall target is almost all per-state level.** Reparameterize m = b(s) + A(s, a) with A(s, pi(s)) = 0, and
    judge only A.
  - **Compute was underestimated 5-10x.** Precompute delta1 and fit with a jitted scan, vmapped over refits.
  - **Several "by import" claims were wrong.** New modules are needed.

## Order of work this leads to

1. **The signal check (this package).** Does delta1 carry the kink at RANDOM's labelled onset rows?
   - NO SIGNAL: stop the Q-error-fitting line on this bed.
2. **On SIGNAL, a revised variant.**
   - The head: one-step signed head A(s, a) primary, which is identifiable, with the propagated head reported. It is
     fit to Q1's error under the data-supported continuation.
   - The rule: calibration checked by anchor coverage; one reference per clause; onset-row leverage; within-world
     placement; a matched even-caution ramp; and the hypothetical-estimator check before sealing.
   - Its frozen-checkpoint screen comes first. Branch training comes only after a PASS.
