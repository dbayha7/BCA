# Results: the return test with estimated weights

EXPLORATORY: outside the version-5 pre-registration.

- **Pass marks:** `PASSMARKS_estimated_weights.md`, frozen 2026-10-14T04:09Z, SHA-256 `4560769a…`, before the code
  was written.
- **Owner's go:** "Proceed with all three".
- **Code:** `command_le_train_est.py`, scored by `score_estimated.py`.
- **Runs:** 20, CPU only, all finite.
- **Outputs:** `runs/shift_iw_v5_explore/channel/train_est/`.

## What was estimated

Nothing exact about the recording process or the current laws was used.

- **The current-law part:** version 5's own estimate, a logistic regression on the observation.
- **The command part:** a classifier on (observation, command) that separates recorded commands from the actor's
  command plus its spread. It is refitted at every refresh.
- **The refusal rule:**
  - "No value credit" where the command is inside the removed ball, or where its estimated odds exceed those of every
    recorded command.
  - The capped threshold where the bank is too small.

## Results

| | medium-expert | expert |
|---|---|---|
| **V1** draws from the test distribution: covered or refused | 96.9% | 97.5% |
| **V2** "no credit" decision agrees with the exact one | 96.7% | 97.9% |
| **V2** of the exact rule's past-80% "no credit", also estimated | **5.0%** (1.4–11.1%) | 82.7% (75.1–87.6%) |
| **X1** A3 − version 5's unweighted, strong starts | +1.5 (+0.2) | **+29.4** (+25.8), 2.3 points |
| **X2** A3 − untargeted control C3, strong starts | +0.8 (−0.9) | **+28.5** (+24.8) |
| **X3** A3 − version 5's unweighted, no-shift starts | −0.3 | +6.5 |
| **X4** share of A2's (exact weights) gain kept (bar 0.5) | 0.06 | 0.40 |
| Share of the exact oracle's gain | 0.02 | 0.27 |

## Scored against the pass marks

- **Expert:** V1, V2, X1, X2 and X3 met; X4 missed.
  - Estimated weights still raise return under the strong shift on every seed, by +25.3 to +35.0 return units.
  - The untargeted control gains nothing.
  - They keep 40% of the exact weights' gain, short of half.
- **Medium-expert:** V1 and X3 met; V2, X1, X2 and X4 missed. The gain, +1.5, is under the 1.8 bar.
- **Predictions:**
  - Expert, 60–100% of A2's gain: missed, at 40%.
  - Medium-expert, V2 at a 50% chance: it failed.
  - Medium-expert, X1 at a 60% chance: it failed.
  - Medium-expert, 0–120% of A2's gain: met at the bottom, 6%.

## Why

- **The estimated band is valid (V1 on both).** Where it certifies, it covers.
- **The return gain depends on finding where the data lack the actor's command.**
  - On expert data, no recorded command goes past 80%, so the classifier finds most of that region (83%).
  - On medium-expert, half the episodes contain random commands past 80% at states like the expert episodes'. The
    exact rule refuses past-80% commands only at the expert episodes' states, using the episode's random share, which
    the classifier cannot see. So the estimated rule refuses almost none (5%), gives them the weak capped threshold,
    and the actor keeps pushing past 80%.
- **On expert, the estimated version keeps 40% of the gain, not more.** It misses 17% of the past-80% cases, and the
  actor moves into exactly those commands during training.

## What it means

- With exact weights, the return gain is shown on both qualities.
- With estimated weights, it holds on expert data (+2.3 points) and vanishes on medium-expert.
- The hard part is estimating where the data lack a command: the support. Estimating the size of the weights is not
  the problem, since coverage held.
