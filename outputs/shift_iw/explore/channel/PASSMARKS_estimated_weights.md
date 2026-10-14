# Pass marks: the return test with estimated instead of exact weights (FROZEN)

Written and frozen 2026-10-14T04:09Z, before the code for this test was written. The SHA-256 is in
`PASSMARKS_estimated_weights.md.sha256`. EXPLORATORY: outside the version-5 pre-registration.

**The owner's go (chat, 2026-10-14):** "Proceed with all three", in answer to "I can commit this (and push, if you say
so), test weights estimated from data instead of exact ones, or add the result to the ThrusterChain page."

## Question

Does the return gain of `RESULTS_fallback_rerun.md` survive when nothing about the recording process or the current
laws is known, and the weights must be estimated from data?

## What is estimated, and how

The exact weight of a (row, command) was w(c) × π_h(a | s) × Z(s) / β(a | s). Both unknown parts are replaced by
estimates. The actor's spread π_h stays as it was (0.2, clipped at 0.5): it is our own choice, not a property of the
data.

- **The current-law part:** version 5's own estimated weight, ŵ(observation) from `classifier.py`. This is a logistic
  regression on the observation, separating training-law steps from strong-law steps (the same samples and fit as
  version 5).
- **The command part:** a classifier on (observation, command) at the kept training rows' states.
  - Class 0 is the recorded command. Class 1 is the actor's command plus its spread, with a fresh draw at every
    step. The classes are equal in size, so the fitted odds estimate π_h(a | s) / (β(a | s) Z(s)) directly.
  - It is a 2 × 128 ReLU network with Adam at 1e-3, 3,000 steps, and batches of 512 per class. The stream is
    default_rng([20261016, seed, 4245, refresh]).
  - It is refitted from scratch at every refresh, every 4,000 training steps, because the actor changes.
- **Nothing exact is used:** not the recording controller's formula, its random share, the hard-water flag, Z, w(c)
  or the 80% knee. The removed ball is still refused, because the analyst made the removal and knows it, as in
  version 5's estimated arms.

## The refusal rule

An estimate is never infinite, so the rule "the data lack the command" becomes one that uses no knobs:

- **No value credit:** the command is inside the removed ball, or its estimated command odds are larger than the
  largest estimated odds of any recorded command among the kept training rows. In words, it looks more like the
  actor's commands than any recorded command does.
- **Capped threshold (version 5's rule (a)):** the command is not in either group above, but its test mass is past
  the onset of refusal.
- **Certified:** every other command keeps the band penalty.

The bank weights are ŵ(observation) × the estimated command odds at each bank row's recorded command. Everything else
is the rerun's: the table, U(s, a_i), the protocol, the seeds and both qualities.

## Arms

- **A3:** estimated weights, strong law, the rule above.
- **C3:** "no value credit" on random rows, as many per chunk as A3 gives it.

That is 20 runs on CPU. The references are already trained: version 5's unweighted lower edge, A2 (exact weights),
the no-caution actor and the exact oracle.

## Pass marks

Every mark is checked on both qualities, medium-expert and expert.

### Check without training, at the no-caution actor (as `command_band_check.py`)

- **V1 (the estimated weights are valid):** at draws from the test distribution, covered or refused is at least 88%
  under the strong law, as a mean over seeds. "Strong law" here means truth rows reweighted by the exact w(c), for
  scoring only.
- **V2 (the estimated support agrees with the exact one):** two conditions, as means over seeds:
  - The "no value credit" decision at the actor's own commands agrees with the exact rule's on at least 90% of rows.
  - Of the past-80% commands the exact rule gives no credit, the estimated rule also gives no credit to at least 80%.

### Training (paired by seed; mean and one-sided 95% lower bound, return units)

- **X1:** A3 minus version 5's unweighted lower edge, on the strong starts, has a mean of at least 1.8 and a lower
  bound above 0.
- **X2:** A3 minus C3, on the strong starts, has a mean of at least 1.8 and a lower bound above 0.
- **X3:** A3 minus version 5's unweighted lower edge, on the no-shift starts, has a mean of at least −1.8.
- **X4:** A3 keeps at least half of A2's gain: (A3 − unweighted) ≥ 0.5 × (A2 − unweighted), on the strong starts.

Training runs whatever V1 and V2 show (it is exploratory and cheap). Both are reported.

## Predictions

- **Expert:** the data never go past 80%, so the classifier should find that support easily. V1 and V2 pass, X1–X4
  pass, and A3 keeps 60–100% of A2's gain (A2: +74.2).
- **Medium-expert:** the exact "no credit" set is small and depends on the episode type, which the classifier cannot
  see, so agreement will be lower. My chance that V2 passes is 50%. X1 passes with about 60% chance. The gain lands
  between 0% and 120% of A2's (A2: +25.0).
