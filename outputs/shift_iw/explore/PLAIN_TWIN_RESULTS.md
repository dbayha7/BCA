# Plain TD3+BC coverage with its two critics' disagreement: results

EXPLORATORY: outside the pre-registered version-5 claims. Expectations were written first, in PLAIN_TWIN_PREREG.md.
Data: runs/shift_iw_v5_explore/plain_twin/<quality>.json. Made by plain_twin_coverage.py and printed by
plain_twin_table.py. Five seeds; means across seeds. Every row passed the check that min(Q1, Q2) reproduces the saved
critic value.

## Host trained with the 80% hold removed (the experiment's host)

Coverage of the one-step target, %. Plain band: q ± |Q1 − Q2|. "BCA" is unweighted BCA; "weighted" is per-step,
exact-weighted BCA.

| Quality | No shift: plain / BCA / weighted | Strong shift: plain / BCA / weighted | Removed-hold steps, strong: plain / BCA / weighted |
|---|---|---|---|
| random | 45.6 / 91.8 / 91.8 | 44.7 / 86.7 / 92.9 | 43.9 / 87.4 / refuses 100% |
| medium-replay | 44.9 / 91.2 / 91.2 | 43.9 / 85.0 / 91.0 | 42.6 / 84.8 / refuses 100% |
| medium | 46.0 / 91.7 / 91.7 | 45.3 / 83.7 / 92.8 | 44.0 / 81.9 / refuses 100% |
| medium-expert | 46.7 / 92.5 / 92.5 | 48.1 / 83.6 / 94.6 | 49.8 / 90.1 / refuses 100% |
| expert | 38.7 / 91.4 / 91.4 | 38.6 / 83.3 / 91.9 | 38.2 / 76.9 / refuses 100% |

Widths:

- The plain band's median width is 0.22–0.32 times unweighted BCA's.
- The plain band is about as wide inside the removed region as outside: the median ratio is 0.80–1.23 under the strong
  shift.

The full-data host gives the same picture: plain coverage is 35.6–46.5%.

## Expectations

1. **Below 50% with no shift: met.** It was 38.7–46.7%.
2. **Drops or stays flat under the strong shift: met on four qualities, roughly met on the fifth.**
   - It stayed within 1 point on four qualities.
   - On medium-expert it rose by 1.4 points, from 46.7 to 48.1. The likely reason: the two critics disagree more
     where the current is strong, which is where the shifted test puts its weight, so the plain band widens there
     along with the error. Its coverage of about 45% sits in the middle of the error distribution, so the shift moves
     it little. BCA's 90% coverage depends on the tail, which the shift does move.
3. **No warning inside the removed region: met.** The width ratio is 0.80–1.23, below 2.
4. **Plain band under half of BCA's width: met.** It is 0.22–0.32 times.

## Plain reading

- **Plain TD3+BC's own uncertainty covers its errors less than half the time,** with or without shift. Its two critics
  learn from the same data, so they agree more closely than their real error.
- **It also gives no sign that a command is missing from the data.** Its band is no wider at the removed 80% hold.
- **Unweighted BCA reaches the 90% target with no shift,** but falls to 83–87% under the strong shift.
- **Weighted BCA keeps 91–95% under the strong shift,** and refuses every step that uses the removed command.
