# Results: return test rerun with the refusal fallback fixed

EXPLORATORY: outside the version-5 pre-registration.

- **Pass marks:** `PASSMARKS_fallback_rerun.md`, frozen 2026-10-14T03:40Z, SHA-256 `8b8f7a27…`, before the code was
  written.
- **Owner's go:** "Fix the fallback, then rerun (Recommended)".
- **Runs:** 30 runs on CPU (2026-10-14, 03:43–03:50Z by the records), all finite.
- **Code:** `command_le_train_f1.py`, scored by `score_fallback_rerun.py`.
- **Outputs:** `runs/shift_iw_v5_explore/channel/train_f1/`.

## The change

A refused command gets "no value credit over the recorded command" only where the exact ratio is infinite: the
command is inside the removed hold, or the recording process never gives it at that state and episode. Where the ratio
is finite but the bank is too small to certify it, the command gets version 5's capped threshold instead. Nothing else
changed from the first return test.

## Checks before training (passed)

- **Unchanged step:** with the weights switched off, the step equals version 5's unweighted step bit for bit. The
  largest parameter difference after 300 steps is 0.0, on both qualities.
- **Refusal types:** in the graph they equal the numpy ones on every row.
  - **Expert:** 17.7% of the no-caution actor's commands get no credit (every past-80% command, and 7.9% within
    80%), and 2.7% get the capped threshold.
  - **Medium-expert:** 6.4% get no credit, including 17.5% of past-80% commands. 26.6% get the capped threshold,
    including 82.5% of past-80% commands, which come from random-share episodes, where past-80% commands do occur in
    the data.

## Results: return units, mean over seeds, with the one-sided 95% lower bound

| | medium-expert | expert |
|---|---|---|
| **U1** A2 − version 5's unweighted lower edge, strong starts | **+25.0** (+22.6), 2.0 points | **+74.2** (+68.3), 5.8 points |
| … per seed | +24.1, +23.0, +26.2, +22.8, +28.8 | +80.4, +65.6, +80.0, +71.0, +74.3 |
| **U2** A2 − untargeted control C2, strong starts | **+23.9** (+22.1) | **+73.4** (+67.1) |
| **U3** A2 − version 5's unweighted, no-shift starts | **+4.5** (+3.8) | **+15.6** (+14.4) |
| **U4** medium-expert: A2 − the first test's A, strong | **+149.8** (+117.5) | |
| **U4** expert: A2 − version 5's unweighted, strong (bar 39.3) | | **+74.2** |
| Shown: A2 − B2 (what knowing the test current law adds), strong | −0.2 (−0.6) | +0.5 (−0.7) |
| Shown: C2 − version 5's unweighted, strong | +1.1 (−0.6) | +0.8 (−0.5) |
| Shown: A2's gain as a share of the exact oracle's, strong | 0.34 | 0.69 |

Mean points (0 = uniform random commands, 100 = the stall-aware controller), strong / mild / no-shift starts:

| Controller | medium-expert | expert |
|---|---|---|
| No-caution actor | 92.1 / 94.2 / 96.1 | 90.7 / 93.4 / 95.9 |
| Version 5 lower edge, unweighted | 92.1 / 94.2 / 96.1 | 90.6 / 93.4 / 95.9 |
| First test's A (no credit for every refusal) | 82.3 / 86.9 / 91.4 | 96.8 / 97.5 / 98.0 |
| **A2: command-weighted, fixed fallback** | **94.1 / 95.5 / 96.7** | **96.5 / 97.3 / 98.0** |
| B2: the same, training law | 94.1 / 95.5 / 96.7 | 96.4 / 97.2 / 97.9 |
| C2: untargeted, same amount of no credit | 92.2 / 94.2 / 96.0 | 90.7 / 93.4 / 95.9 |
| Exact oracle lower edge | 97.9 / 98.0 / 98.0 | 99.1 / 99.0 / 99.0 |

## Scored against the pass marks: U1–U4 met on both qualities

- **U1:** weighting raises return under the strong shift on every seed of both qualities: +2.0 points on
  medium-expert and +5.8 on expert.
- **U2:** the same amount of "no value credit" placed at random gains nothing (+1.1 and +0.8). The gain comes from
  where the weighted band refuses.
- **U3:** there is no cost without the shift; return rises there too (+4.5 and +15.6).
- **U4:** the fix removed medium-expert's loss (+149.8 over the first test). Expert kept 94% of its first-test gain
  (−4.3 return units against the first test's A).
- **Predictions:**
  - Expert, +50 to +85: met (+74.2).
  - Medium-expert, −20 to +40: met (+25.0). I had given U1 there about a 35% chance; it passed.
  - A2 − B2 within ±10: met (−0.2 and +0.5).

## What it means

- **Importance weighting with the known test distribution raises return** on both qualities where the lower edge has
  room: 34% (medium-expert) and 69% (expert) of the gain exact information gives.
  - **The test distribution:** where the band is read, the actor's own commands under the test current law. Both its
    parts are known exactly in this world.
  - **The weights:** the exact density ratios.
  - **The channel:** the lower edge, a penalty on the value of the actor's own command.
  - **The refusal rule:** where the data contain no such command, the actor gets no value credit over the recorded
    command.
- **The return gain comes from the command part of the weight.** The current-law part adds nothing to return
  (A2 − B2: −0.2 and +0.5), although it is what restores coverage at recorded commands under the shift (version 5).
- **What the weighting does here is a calibrated check for support.** It refuses where the recorded commands never go
  (past 80% at expert states, and the removed hold), and certifies elsewhere. That is the exact-ratio rule of weighted
  conformal prediction, not a hand-set limit: no knee or threshold of the world was given to it.
- **Limits:**
  - One environment, with its own data.
  - Only actor-only training against a frozen critic, as in version 5's lower-edge arms.
  - Exact weights only. Weights estimated from data are not tested yet.
  - The command spread (0.2) is BCA's own target noise and was not tuned.
  - The three lower qualities have no room for the lower edge even with exact information, and were not run.
