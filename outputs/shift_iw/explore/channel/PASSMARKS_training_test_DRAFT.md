# Pass marks, DRAFT for the owner's approval: return test of the command-weighted lower edge

Drafted 2026-10-14, after `RESULTS_command_weighted_band.md` and before any code for this test was written or run.
It is frozen (timestamped, with its SHA-256 recorded) only after the owner approves it. EXPLORATORY: outside the
version-5 pre-registration.

## Question

Does weighting BCA's band for the test distribution, meaning the actor's own commands under the strong current law,
raise the learned controller's return under the shift through the lower edge?

## Setup

- **Qualities:** medium-expert and expert, the only two where the lower edge has room even with exact information.
  Seeds 0–4.
- **Protocol:** the version-5 lower-edge arms', unchanged.
  - Actor-only training against the frozen gapped critic, starting from the cloned actor.
  - 20,000 steps in chunks of 4,000.
  - Batch rows from default_rng([20261016, seed, 35]); Adam 3e-4; batch 256; α 2.5; k = 1.
  - BCA's frozen band: the scale network, unit, bank scores, stream and posterior.
- **New arms:**
  - **A, command-weighted (strong law):** the weights of `command_band_check.py`, w(c) × π_h × Z / β, with the spread
    centred on the current actor's command.
    - A refused command gets no value credit over the row's recorded command, so W = [Q1(s, a) − Q1(s, a_i)]_+.
    - A certified command keeps W = [U(s, a) − U(s, a_i)]_+.
    - The bank weights, the table of R and U(s, a_i) are recomputed from the current actor at the start of every
      4,000-step chunk.
  - **B, command-weighted (training law):** A with w(c) = 1. It knows the actor's commands but not the test law, and
    shows what knowing the test current law adds.
  - **C, untargeted control:** "no credit" on randomly chosen rows, as many as A refuses at the same chunk and seed,
    and no band penalty otherwise. It has the same amount of caution as A, placed without regard to the command.
- **References, already trained and not rerun:** the no-caution actor, version 5's unweighted and exact-strong lower
  edges, and the exact oracle lower edge.
- **Return:** the exact 100-step return over the fixed start banks (strong, mild and no-shift starts, 1,000 each),
  as in version 5's evaluation.
- **Compute:** 30 runs, CPU only, an estimated 1.5–3 CPU-hours (about 20–40 minutes on 8–10 cores).

## Pass marks

All comparisons are paired by seed: the mean, and the one-sided 95% lower bound (t with 4 degrees of freedom).
1.8 return units is version 5's size bar.

- **T1 (weighting raises return under the shift):** A minus version 5's unweighted lower edge, on the strong starts,
  has a mean of at least 1.8 and a lower bound above 0, on both qualities.
- **T2 (from where the caution is placed, not how much):** A minus C, on the strong starts, has a mean of at least
  1.8 and a lower bound above 0, on both qualities.
- **T3 (little cost without the shift):** A minus version 5's unweighted lower edge, on the no-shift starts, has a
  mean of at least −1.8, on both qualities.
- **Shown, not marked:**
  - A minus B on each start bank;
  - A's gain as a share of the oracle's gain;
  - the refusal shares during training;
  - the share of past-80% commands on the strong starts.

## Predictions

- **Expert:** A minus unweighted, on the strong starts, lies between +30 and +110 return units. The oracle's room is
  +107.6.
- **Medium-expert:** a positive but smaller gain, between 0 and +60. There is a real chance of a loss, because 48% of
  the actor's within-80% commands are refused and become cloning-only rows, and half of the recorded commands there
  are random. My chance that T1 passes here is 50–60%.
- **A minus B:** within ±10 return units on both. Refusal past 80% comes from the command part of the weight. The
  current law changes only how many within-80% commands are refused.
- **A minus C on expert:** at least +20.

## Checks before training

- With the command weights replaced by version 5's unweighted threshold and no refusals, arm A's step must equal
  version 5's unweighted lower-edge step.
- The in-graph recorder density must match the numpy one used in `command_band_check.py`.
- At chunk 0, the refusal shares must match `command_band_check.py`'s for the same actor.
