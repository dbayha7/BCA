# BCA dose screen v1: readout (2026-10-31)

This readout covers the exploratory dose screen in `../bca_dose_screen_v1/`. It used development data.

- **Verdict.** The verdict comes from the frozen rule (`PROTOCOL.md`, `screen_math.py`).
- **Everything else is descriptive.** It was analyzed after the result, on fixed networks, and it is not a verdict.
- **Checking.** Two rounds of independent checkers reproduced every number (workflows `wf_c71a0249-59d` and
  `wf_058dd92d-4c2`).

## Verdict (frozen rule)

**CUTOFF: MIXED. RANDOM: MIXED.** The protocol says: "report it, and make no claim."

- **Run.** All 12 BCA branches and 3 evaluations completed at kappa_A = 63, and every pairing check passed. The run
  took 5,808 s and 8,475,600 exact simulator steps. There were no dose failures and no cap was reached.
- **Recomputation.** `screen_math.decide`, run on Windows from the saved JSON reports, gives a decision identical to
  `SCREEN_RESULT.json`.
- **CUTOFF.** Delta_entry(stall3) was +0.058, -0.153, +0.001, so there was no unanimous drop.
  - HARM needs 2 seeds. Only s0 meets its entry clause (+0.058 against +0.05), so the arm is one seed short of HARM.
  - NO EFFECT fails on s1.
- **RANDOM.** Delta_entry(stall3) was -0.043, -0.172, -0.033, so there was no unanimous drop.
  - HARM is met in no seed.
  - NO EFFECT fails on s1.

## The paired behavior (BCA minus native, at the 20k readout)

| Seed-arm | dEntry stall | dEntry no-stall | Placement | dJ stall | dJ no-stall | q_rank_gain_s10 |
|---|---|---|---|---|---|---|
| CUTOFF s0 | +0.058 | -0.020 | +0.078 | -16.6 | +2.0 | 0.143 |
| CUTOFF s1 | -0.153 | -0.154 | +0.002 | -18.0 | +0.6 | 0.209 |
| CUTOFF s2 | +0.001 | -0.336 | +0.336 | -20.2 | +0.7 | 0.098 |
| RANDOM s0 | -0.043 | -0.480 | +0.438 | +10.0 | -0.7 | 0.220 |
| RANDOM s1 | -0.172 | -0.324 | +0.152 | +3.3 | -8.7 | 0.341 |
| RANDOM s2 | -0.033 | -0.075 | +0.043 | -35.3 | +3.4 | 0.093 |

- **Placement.** Placement is >= 0 in all six, so entry was never cut more where it hurts than where it is allowed.
  The return-based Sel readout was positive in RANDOM s0 and s1 (+10.6, +12.0) and negative in the other four.
- **Returns depend on the checkpoint.**
  - At 20k, CUTOFF stall-world return was lower with BCA in all three seeds. At 15k the same differences were -9.7,
    -19.5 and +26.9.
  - The protocol's "8.1" is one native run's largest move from 15k to 20k. It is not a noise band for a paired
    difference, and BCA runs at kappa 63 moved up to 44.7 within a run.
  - HARM does not use dJ(stall3), so these numbers are descriptive only.
- **Near misses.** RANDOM s0's dJ(stall3) of +9.97 is just under the +10 bar. RANDOM s1's dJ(no-stall) of -8.7 is past
  the -5 bar.

## Q values: does the penalized Q rank s10 correctly where Q1 is wrong?

These readouts use the 48 gated E2 anchors, comparing the s10 slot with edge_knee against exact Q^pi under the run's
own clipped continuation. The penalized Q is Q1 - 63 x U, which is the value the BCA actor maximizes.

**Stall world** (41-44 harmful anchors per run, where s10 is truly worse):

- **Raw Q1.** It ranks s10 wrongly at 36-40 of them. Across anchors, Q1's contrast runs opposite to the truth
  (Spearman -0.80 to -0.93): the critic likes s10 most where s10 is worst.
- **Fixes.** The penalty fixes 4-15 anchors and breaks 0-1, and the penalized Q is still wrong at 55-88%.
  - q_rank_gain_s10 reaches 0.20 in CUTOFF s1, RANDOM s0 and RANDOM s1. No arm reaches it in every seed.
  - The bar equals 9 anchors, and two of the three passes sit at exactly 9.
- **Size relative to what the ranking needs.** On wrongly ranked anchors, the median 63 x dU / C_raw is 0.04, 0.33,
  0.29 and 0.62 in the four runs where it points the right way. In s2 of both arms it is -0.32 and -0.56, the wrong
  sign.
- **Size relative to the critic's error.** It offsets at most about 15% of the median KC (1.13-2.02). In s2 of both arms
  it adds about 6% to the error.
- **Sign.** dU is > 0 at only 27-91% of harmful anchors. It is <= 0 at half or more of the wrongly ranked anchors in
  CUTOFF s0 (20/40), CUTOFF s2 (30/40) and RANDOM s2 (24/38), so no kappa flips the median wrong anchor in those
  three runs. In the other three runs, the kappa that flips it is about 100-220.
- **Placement across anchors.**
  - U's rise past the knee is never larger where the critic's error is larger. The raw Spearman(dU, KC) is negative in
    all six runs (-0.15 to -0.50, clear in five).
  - KC is nearly collinear with true harm and with a state proxy, the native actor's a0. With that proxy controlled, the
    correlation is -0.31 to +0.14. So the data cannot say whether U ignores the error or follows a state feature that
    runs against it.
- **Shuffled U does as well.** Shuffling dU across anchors within the stall run gives the same or a higher gain in all
  six (share of shuffles >= BCA: 0.65-0.96, from 200k shuffles). Where U sits across anchors adds nothing beyond its
  size. The median dU is lower on harmful than on non-harmful anchors in all six stall runs, and higher on them in all
  six no-stall runs, so this is not U locating the critic's errors.
- **BCA's own Q1 is a weaker baseline.** BCA's raw Q1 ranks worse than native's in all six runs. Measured against
  native raw Q1, the penalized gain is 0.12, 0.22, 0.16, 0.29, 0.03 and 0.00.
- **The share metric is fragile.**
  - The share can rise while the median error worsens (s2 of both arms).
  - RANDOM s0 passed although its harmful-anchor median KC was unchanged (1.67 to 1.68).

**No-stall world** (where entry past the knee is allowed):

- The critic's error is small there (median KC -0.02 to 0.17). The penalty contrast is 1.6-3.9 times larger
  (median |63 dU| 0.23-0.52), so it distorts values where the critic is roughly right.
- The direction is mixed. The penalty favors s10 in 4 of 6 runs, and the false caution is concentrated in RANDOM s0 and
  s1 (19/41 and 13/41 non-harmful anchors newly ranked s10-worse).

## BCA's U against even caution, at matched cost (fixed networks, anchor ranking only)

This is the comparison the study's question names: does caution placed by uncertainty beat caution spread evenly?

- **What is measured.** The gain is the net number of harmful stall anchors that the penalized Q newly ranks
  correctly. The cost is the number of false flips: non-harmful no-stall anchors that it newly ranks s10-worse.
- **The two penalties.** BCA's is kappa x dU. Even caution is a constant penalty c, in Q units.
- **The knob.** Each knob is shared by the stall and no-stall runs, because a deployed knob cannot know the world.
- **Exact frontier.** Both frontiers were enumerated exactly at every sign-change point (`frontier.py`), and an
  independent checker reproduced them.

**One knob for all six seed-arms (how kappa_A is used):**

| False flips allowed (of 220) | 0 | 5 | 11 | 22 | 33 | 43 | 66 | 88 |
|---|---|---|---|---|---|---|---|---|
| BCA best gain (of 254) | 1 | 9 | 19 | 32 | 38 | 48 | 79 | 110 |
| Even caution best gain | 1 | 9 | 28 | 43 | 55 | 74 | 99 | 123 |

- **Even caution is never behind.** It is ahead at 79 of the 81 budgets from 0 to 80 false flips and tied at 0 and
  5.
- **At kappa_A's own cost.** At kappa 63, BCA corrects 47 at 43 false flips. Even caution at the same cost corrects 74.
- **The s05 slot.** It shows the same picture: even caution is ahead at all 81 budgets.

**A separate knob per seed-arm** (an oracle tuning that lets kappa absorb each network's U scale, which favors BCA).
The table gives BCA minus even caution, in harmful stall anchors, at 0 to 8 false flips:

| Seed-arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| CUTOFF s0 | +4 | +1 | +4 | +2 | -3 | -2 | -1 | -1 | -2 |
| CUTOFF s1 | +1 | +3 | +9 | +12 | +13 | +12 | +8 | +13 | +14 |
| CUTOFF s2 | -3 | -2 | -3 | -1 | 0 | -1 | 0 | -1 | -5 |
| RANDOM s0 | -1 | -6 | -6 | -6 | -8 | -9 | -12 | -13 | -14 |
| RANDOM s1 | 0 | 0 | 0 | 0 | -1 | -1 | +3 | +5 | +5 |
| RANDOM s2 | -1 | -2 | -2 | -4 | 0 | +1 | -2 | -2 | -2 |

With a separate knob per seed-arm the outcome varies:

- **BCA clearly better.** CUTOFF s1.
- **BCA clearly worse.** RANDOM s0.
- **Small and unstable.** The other four are within a few anchors, with signs that depend on the budget. Several best
  points need knob windows less than 1% wide.

Where BCA does help, the source differs by seed-arm:

- **RANDOM s1.** A per-run constant using the signed median dU reproduces the gain, because signed dU is lower in the
  no-stall run.
- **CUTOFF s1.** About half the lead survives shuffling dU within both runs. The rest depends on dU being lower on the
  non-harmful no-stall anchors.
- **The size of dU is not lower in the no-stall run.** Median |dU| is higher there in 5 of 6 pairs.

**Bottom line (descriptive).** On these networks, BCA's U does not place caution where the critic is wrong any
better than an even penalty does. With one knob, as BCA is actually used, the even penalty is never behind. It
corrects more of the critic's wrong rankings at 79 of the 81 no-stall false-flip budgets from 0 to 80, and ties at 0
and 5. (Corrected 2026-10-31: the first version said "at every level".)

## What this does not show

- **No claim.** Both arms are MIXED.
- **Development data.** The forks are development forks with seen outcomes. kappa_A was chosen after the leverage
  values were printed, and the anchors had already been used. Every analysis above was written after the result.
- **Fixed networks.** The kappa rescoring, shuffle and even-caution comparisons hold the trained networks fixed and
  score anchor rankings. They do not predict what a training run at another kappa, or with an even penalty, would do,
  and they say nothing about returns.
- **Mechanism untested.** The results are consistent with `../bca_q_error_signal_v1/METHOD_DESIGN_NOTES.md` (eta is fit
  to the size of one-step residuals at logged actions and has no action shape), but this screen does not test that
  mechanism.
- **No fresh seeds.** The fresh-seed higher-dose screen in `../EXPERIMENT_DIRECTION_20261031.md` was not run.

## Lessons for the next protocol (a future rule, never a re-score of this screen)

1. **Matched nulls.** Judge a rank-share gain against matched nulls (shuffled U, a uniform ramp, sign-flipped U), not
   against 0. Here a uniform ramp of BCA's typical size clears 0.20 in all six stall runs.
2. **More clauses.** Pair the share with two further clauses: penalized |KC| must fall below raw |KC|, and no-stall false
   caution must stay within a limit. The no-stall world has 27-41 non-harmful anchors, against 4-7 in the stall world.
3. **Counts and resolution.** Report fixed and broken counts. One anchor is about 2.3 points of share.
4. **Native baseline.** Also compare against native raw Q1, because a BCA run's own Q1 can be the weaker baseline.
5. **Return bars.** These need a spread for the paired difference, not one native run's drift. Every return statement
   should name its checkpoint.

## Next

- **The Q-error signal check** (`../bca_q_error_signal_v1/`) runs under its pre-written rule. It is the separate
  method-change track, and this screen neither licenses nor blocks it.
- **Commit first.** Commit its RULE.md and signal_check.py before running it, so that "written first" can be checked.
- **Scope.** The signal check decides on RANDOM only. A SIGNAL would not show that a Q-error U helps in CUTOFF, the
  no-label case where data-only rules are the natural competitor.

## Files

| File | What it is |
|---|---|
| `dose_readout.py` -> `dose_readout.json` | verdict recomputation and the per-anchor Q breakdown |
| `even_caution.py` -> `even_caution.json` | BCA's U against shuffled U, sign-flipped U and an even ramp of matched typical size |
| `frontier.py` -> `frontier.json` | the matched-cost comparison above |

All three are read-only on the saved JSON reports. Run each with
`.venv\Scripts\python.exe -B <file> > <output>.json`, with PYTHONDONTWRITEBYTECODE=1.
