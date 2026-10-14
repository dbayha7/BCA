# Expectations, written before running: the new arms on the three lower qualities

Written 2026-10-14T08:46Z, before any of these runs. The owner's go was "Full set, ~3 h CPU (Recommended)", for
learning curves on all 5 qualities × 3 test conditions. EXPLORATORY and shown only: there are no pass marks. The
exact oracle has no room on these qualities.

The arms are the fixed-fallback set (A2 command-weighted under the strong law, B2 under the training law, C2 its
untargeted control) and the estimated-weight set (A3, C3), on random, medium-replay and medium, seeds 0–4. They use
the same code and settings as on the top two qualities.

## Expected

- On these three qualities every episode has a random share of at least 0.65, so the recording process can give any
  command and the exact ratio is never infinite outside the removed ball.
  - A2 and B2 give "no value credit" only inside the ball.
  - Everything else the band refuses gets version 5's capped threshold: a mild penalty.
- **Return:** A2 − version 5's unweighted lower edge, on the strong starts, lies within ±10 return units on all three
  qualities. The same holds for A3. The untargeted controls lie within ±5.
- **Learning curves:** these arms track the no-caution actor's curve closely, as version 5's lower edge does.
