# Pass marks: rerun with the refusal fallback fixed (FROZEN)

Written and frozen 2026-10-14T03:40Z, before the code for this rerun was written. The SHA-256 is in
`PASSMARKS_fallback_rerun.md.sha256`. EXPLORATORY: outside the version-5 pre-registration.

**The owner's go (chat, 2026-10-14):** the question was "Expert data show the return gain; medium-expert loses
because refusals there make the controller copy random recorded commands. What next?" The answer was "Fix the
fallback, then rerun (Recommended)". That option read: "Apply 'no value credit' only where the data truly lack the
command (exact ratio infinite: no recorded command like it, or the removed hold); keep version 5's capped threshold
where the bank is merely too small. No new tuning knob. Pass marks first; expect expert to keep most of its gain and
medium-expert to stop losing."

## The one change from the first return test (`PASSMARKS_training_test.md`)

A refused command now gets one of two treatments.

- **The data lack the command:** the exact ratio is infinite, because the command is inside the removed ball or the
  recording process's density is 0 at the row's state and episode. It keeps "no value credit over the recorded
  command", W = [Q1(s, a) − Q1(s, a_i)]_+.
- **The bank is merely too small:** the ratio is finite but its test mass is past the onset of refusal. It gets
  version 5's rule (a): the table's largest finite threshold, W = [R_cap · unit · scale(s, a) − U(s, a_i)]_+.

Everything else is as in the first test: the weights, the actor's spread (0.2), certified commands, U(s, a_i), the
protocol, seeds 0–4, medium-expert and expert, and the refresh every 4,000 steps.

## Arms

- **A2, command-weighted under the strong law,** with the fixed fallback.
- **B2:** A2 with w(c) = 1 (the training law).
- **C2, untargeted control:** "no value credit" on random rows, as many per chunk as A2 gives "no value credit" at
  the same chunk and seed. There is no band penalty otherwise.

Thirty runs, CPU only, about 150 s each.

## Pass marks

All comparisons are paired by seed: the mean, and the one-sided 95% lower bound (t, 4 degrees of freedom), in return
units, against the same references as before.

- **U1:** A2 minus version 5's unweighted lower edge, on the strong starts, has a mean of at least 1.8 and a lower
  bound above 0, on both qualities.
- **U2:** A2 minus C2, on the strong starts, has a mean of at least 1.8 and a lower bound above 0, on both qualities.
- **U3:** A2 minus version 5's unweighted lower edge, on the no-shift starts, has a mean of at least −1.8, on both
  qualities.
- **U4 (the fix does its job):**
  - On medium-expert, A2 minus the first test's A, on the strong starts, has a mean of at least +50: the loss is
    mostly gone.
  - On expert, A2 minus version 5's unweighted lower edge, on the strong starts, has a mean of at least +39.3: at
    least half of the first test's +78.6 is kept.
- **Shown:** A2 minus B2; the shares of each refusal type during training.

## Predictions

- **Expert:** +50 to +85 return units against version 5's unweighted lower edge on the strong starts. Every
  past-80% command at an expert state has density 0, so "no value credit" still applies there.
- **Medium-expert:** −20 to +40. At random-share episodes, past-80% commands now get the capped threshold, which is
  weak (the shape check failed with it), so the controller may still push past 80% there. My chance that U1 passes on
  medium-expert is about 35%. My chance that U4 passes there is about 75%.
- **A2 minus B2:** within ±10 on both qualities.

## Checks before training

- With the weights switched off, the step still equals version 5's unweighted step: the first test's check, rerun.
- At the no-caution actor's commands, the shares of refusal types in the graph equal the numpy ones.
