# Results: return test of the command-weighted lower edge

EXPLORATORY: outside the version-5 pre-registration.

- **Pass marks:** `PASSMARKS_training_test.md`, frozen 2026-10-14T02:58Z with the owner's approval ("Approve as
  drafted (Recommended)"), SHA-256 `204f53ff…`, before this code was written.
- **Runs:** trained 23:03–23:11 local time (2026-10-14 03:03–03:11Z): 30 runs, all finite, about 150 s each, CPU only.
  Code: `command_le_train.py`. Scored by `score_training_test.py`.
- **Post hoc diagnosis:** `why_medium_expert.py`, labelled as such.
- **Outputs:** `runs/shift_iw_v5_explore/channel/train/`.

## Checks before training (all passed)

- **Unchanged step:** with the command weights replaced by version 5's unweighted threshold and no refusals, the new
  arm's step equals version 5's unweighted lower-edge step bit for bit. The largest parameter difference after 300
  steps is 0.0, on expert and medium-expert seed 0.
- **Density:** the in-graph recorder density matches the numpy one. Finiteness agrees on all 4,000 test points; the
  99th-percentile difference is 3e-5 and the largest 0.012.
- **Refusals:** at the no-caution actor's commands, the in-graph refusals equal the numpy ones on every row.

## Results: return units (mean over seeds, with the one-sided 95% lower bound)

Negative return; higher is better. 1.8 return units is the size bar.

| | medium-expert | expert |
|---|---|---|
| **T1** A − version 5's unweighted lower edge, strong starts | **−124.8** (−157.8) | **+78.6** (+72.8) |
| **T2** A − untargeted control C, strong starts | **−135.0** (−168.2) | **+77.5** (+71.2) |
| **T3** A − version 5's unweighted lower edge, no-shift starts | **−35.6** (−44.4) | **+16.3** (+15.0) |
| Shown: A − B (what knowing the test current law adds), strong | −15.1 (−30.9) | +1.0 (−0.3) |
| Shown: A − B, no-shift | −2.9 | +0.3 |
| Shown: C − version 5's unweighted, strong | +10.2 (+6.9) | +1.1 (−0.7) |
| Shown: the exact oracle − no-caution actor, strong (the room) | +73.5 | +107.6 |
| Shown: A's gain as a share of the oracle's, strong | −1.69 | **0.73** |

Mean points (0 = uniform random commands, 100 = the stall-aware controller), strong / mild / no-shift starts:

| Controller | medium-expert | expert |
|---|---|---|
| No-caution actor | 92.1 / 94.2 / 96.1 | 90.7 / 93.4 / 95.9 |
| Version 5 lower edge, unweighted | 92.1 / 94.2 / 96.1 | 90.6 / 93.4 / 95.9 |
| Version 5 lower edge, current-law weighted | 92.2 / 94.3 / 96.1 | 90.6 / 93.3 / 95.9 |
| **A: command-weighted, strong law** | **82.3 / 86.9 / 91.4** | **96.8 / 97.5 / 98.0** |
| B: command-weighted, training law | 83.5 / 87.8 / 91.8 | 96.8 / 97.4 / 98.0 |
| C: untargeted, same amount | 92.9 / 94.6 / 95.9 | 90.7 / 93.4 / 95.9 |
| Exact oracle lower edge | 97.9 / 98.0 / 98.0 | 99.1 / 99.0 / 99.0 |

## Scored against the pass marks

- **Expert: T1, T2 and T3 all met.**
  - Under the strong shift, A beats version 5's unweighted lower edge by 78.6 return units (6.2 points), with every
    seed between +70.2 and +84.5. It recovers 73% of the exact oracle's gain.
  - The untargeted control, with the same amount of "no credit", gains nothing (+1.1), so the gain comes from where
    the band refuses.
  - Without the shift, A gains 16.3 rather than costing anything.
- **Medium-expert: T1, T2 and T3 all missed.** A loses 124.8 return units (9.8 points) under the strong shift and 35.6
  without it, with every seed negative.
- **Predictions:**
  - Expert, +30 to +110: met (+78.6).
  - Expert, A − C at least +20: met (+77.5).
  - Medium-expert, 0 to +60 with "a real chance of a loss": the loss happened, and it was far larger than the range.
  - A − B within ±10: met on expert (+1.0). Missed on medium-expert, where it was −15.1 and B did better.

## Why medium-expert lost (post hoc, `why_medium_expert.py`, final actor of arm A)

- **Where refusals fall:** they concentrate in the strongest third of currents, where 62–68% of the actor's commands
  are refused, against 11–17% in the weakest third.
  - They also concentrate on episodes with random commands: 47–56% are refused there, against 13–19% on the expert
    half.
  - 59–65% of the refused rows have a random recorded command, while 36–39% of all rows do.
- **What a refusal does:** under "no value credit", a refused row only clones its recorded command. In strong
  currents the actor therefore learns to copy commands that are random most of the time, and it loses control exactly
  where the strong shift puts it.
- **Why the strong law made it worse:** weighting by the strong law raises the test mass in strong currents, which
  refuses even more there. That is why A did worse than B (−15.1).
- **Why expert gained:** refusals also concentrate in strong currents there (36–41% in the top third), but every
  recorded command is the expert's, so cloning there is good. The refusals past 80% are 100%, and they remove the
  stall.

## What it means

- **A return gain from importance weighting, shown on expert data.**
  - Weighting BCA's band for the distribution where the lower edge reads it, the actor's own commands, makes it honest
    there.
  - Turning its refusals into "no value credit" raises return under the strong shift by 6.2 points, 73% of what exact
    information gives. It costs nothing without the shift.
  - An untargeted control of the same size gains nothing.
- **The gain comes from the command part of the known test distribution.** Knowing the test current law adds nothing
  to return (expert: +1.0, with a lower bound below 0), although it is what fixed coverage at recorded commands in
  version 5.
- **The method fails on mixed data.** Where half the recorded commands are random, the same refusals turn into
  copying random commands in the states the shift emphasises. The refusal rule's fallback (the recorded command) is
  only as good as the data.
