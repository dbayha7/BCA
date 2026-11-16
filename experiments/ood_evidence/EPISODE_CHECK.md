# Episode check: do played episodes match the exact formulas?

Written 2026-11-15, before any episode was played. Sections 1 to 6 are the plan and the pass marks.
Section 7 is filled in after the run and changes nothing above it.

Jump to: [1 Why](#1-why) · [2 What is compared](#2-what-is-compared) · [3 Policies and starts](#3-policies-and-starts) ·
[4 Pass marks](#4-pass-marks) · [5 What the check can and cannot catch](#5-what-the-check-can-and-cannot-catch) ·
[6 Files and commands](#6-files-and-commands) · [7 Result](#7-result)

## 1 Why

Every outcome reported for the Honeypot, the Bottleneck and the published gridworld (chance of
reaching the goal, of ending in a hazard or lava, of running out of time, and the value of a policy)
comes from an exact formula: matrix arithmetic on the chain a policy induces (`chain`, `outcomes`,
`value` in each environment file). No episode has ever been played move by move, and no test
compares the formula with played episodes. If the formula miscounted the 200-move limit, paid a
reward twice at an ending, or disagreed with the one-move rule, every reported number would share
the error.

This check plays episodes with the one-move rules only and compares the counts with the formulas.
It trains nothing, fits nothing and reads no BCA result. It is a test of the environments.

## 2 What is compared

**The episode sampler** (`episode_sampler.py`) starts at a cell, looks up the policy's command for
that cell, makes one move, and repeats until the episode ends or 200 moves have been made.

- Honeypot and Bottleneck: one move = the environment's own `sample_direction` then `move`.
- Published gridworld: the environment file has no function that draws a move, so the sampler
  follows the rule in the file's own words, in two stages. First the commanded move is drawn from
  `command_probabilities`. Then, with chance `noise`, the executed move is drawn evenly from the
  legal moves (the commanded one included); otherwise the commanded move is executed, or the agent
  stays if that move is not legal. The result of the move is the environment's `outcome`.
  This path does not use the execution table that `chain` is built from, so it also tests that table.

**The exact numbers** come from the environment files, unchanged:

| Quantity | Honeypot, Bottleneck | Published gridworld |
|---|---|---|
| Chance of each ending within 200 moves | `outcomes`: goal, hazard, low exit, time-out | `outcomes`: goal, lava, each lava cell, time-out |
| Chance of ending in the withheld hazard (cell 34) | `evaluation.unseen_hazard_probability` (Honeypot only) | not applicable |
| Value of the policy from a start cell | `value` | `value` |

## 3 Policies and starts

Fixed here, before the run. Random seed of the check: 2026111501.

**Designed policies, 200,000 episodes per policy and start.**

| Environment | Policies | Start cells |
|---|---|---|
| Honeypot | best policy; the demonstrator of the withheld-move data; the best policy changed to step into hazard 34 from the four cells beside it; the best policy at half strength; random commands (one draw per cell, seed 2026111502) | 24, 33, 35, 44, 90 |
| Bottleneck revision 2, low exit paying 4.5, 3.5 and 2.5 | the same five, with the withheld move being the step right at cell 54 | 54, 56, 90 |
| Published gridworld, noise 10% | best policy; the best policy changed to command "up" (into lava) at cell 83; the best policy at half strength; random preferences (seed 2026111502) | 0, 57, 83 |
| Published gridworld, noise 0% and 30% | best policy; the best policy at half strength | 0, 57, 83 |

"At half strength" means every command multiplied by 0.5, which makes every move possible at every
cell. With noise 0% and the best policy nothing is random, so every episode must be the same path.

**Trained policies, 40,000 episodes per policy and start.** These are the saved policies behind the
latest reported numbers, so the formula is checked on exactly the policies it was used on.

| Environment | Saved policies | Count |
|---|---|---|
| Honeypot | `gridworld_ood_v1/results/confirm_v5`, withheld-move data, datasets 7501 to 7509, training seeds 70 to 72: standard learner and BCA policy guard | 54 |
| Bottleneck revision 2 | the same block, each of the three low-exit settings | 162 |
| Published gridworld | `gridworld_critical_v1/results/confirm_version_a`, best-policy data, datasets 6501 to 6509, training seeds 20 to 22: standard learner and BCA policy guard | 54 |

Start cells as in the table above.

## 4 Pass marks

The check passes only if all four marks hold for every environment. One failure fails the check.
The check is run once with the seed above; a failure is investigated and reported, not rerun with
another seed.

| Mark | What must hold | Allowed |
|---|---|---|
| **E1 Same draws, same episode** | For every policy and start, 200 episodes played twice from the same random draws, once one step at a time with the environment's own functions and once with the fast batch sampler, are identical in every cell visited, every reward and every ending. | 0 differences |
| **E2 No impossible step** | In the full sample, every step from one cell to the next is a step the environment's one-move rule allows with a chance above zero under that policy; nothing happens after an ending; no episode is longer than 200 moves. | 0 violations |
| **E3 Endings match** | For every policy, start and ending, the count of played episodes with that ending agrees with the formula's chance. Judged by the exact two-sided binomial test. With K comparisons in all, no comparison may have a p-value below 0.001 / K. An ending the formula gives chance 0 must never occur; one it gives chance 1 must always occur. | family-wise false alarm chance at most 0.1% |
| **E4 Value matches** | For every policy and start, the average discounted return of the played episodes is within 5 standard errors plus 0.001 of the formula's value. | as stated |

Notes on the marks.

- The 0.001 in E4 covers the only intended difference: the formula's value runs forever, a played
  episode stops at 200 moves. With discount 0.95 the part after move 200 is at most
  0.95^200 x 1 / 0.05 = 0.0007 in the Honeypot and Bottleneck and 0.00004 in the published gridworld.
- K is counted by the script and printed. It is expected to be about 4,600.
- Reported but not judged: the smallest p-value, and the share of p-values under 0.05 (about 5% or
  less is expected when the formulas are right).

## 5 What the check can and cannot catch

It can catch (design figures, from the binomial spread at these sample sizes):

- For trained policies (40,000 episodes): an ending whose real chance differs from the formula's by
  2 percentage points or more, more than 99 times in 100; by 1 point when the chance is under 10%
  or over 90%, more than 9 times in 10.
- For designed policies (200,000 episodes): a difference of 1 percentage point or more, more than
  99 times in 100.
- Any ending that the formula calls impossible but that happens, and the reverse.
- A value wrong by more than about 0.02 for typical policies (the exact figure depends on the spread
  of returns and is printed).

It cannot catch:

- An error shared by the one-move rule and the formula (for example a wrong map). The map and the
  one-move rule have their own tests; the published map's transcription from the paper's figure is
  a separate open item.
- Differences smaller than the figures above.
- Anything about a learner, a critic or BCA. No learner code is imported.

## 6 Files and commands

| File | Role |
|---|---|
| `episode_sampler.py` | the sampler: one-step-at-a-time version and fast batch version |
| `episode_check.py` | runs E1 to E4, prints a table, writes `episode_check_result.json` |
| `episode_check_result.json` | written once by the run: every comparison, the marks, the verdict |

```
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/ood_evidence/episode_check.py
```

A wiring test with a different seed and 2,000 episodes (`--smoke`) may be run first to time the
script. Its numbers are not judged and are not written to the result file.

## 7 Result

Written after the run. Nothing in sections 1 to 6 was changed.

**Verdict: PASS.** Run once on 2026-11-15 with the seed and sample sizes of section 3 (129 seconds).
A wiring test (`--smoke`, another seed, 2,000 episodes) was run first to time the script; it was not
judged and wrote nothing.

| Environment | Policy and start pairs | E1 episodes that differ | E2 impossible steps | E3 failures | E3 smallest p | Largest gap, points | E4 failures | Largest value gap |
|---|---|---|---|---|---|---|---|---|
| Honeypot | 295 | 0 | 0 | 0 | 0.0135 | 0.554 | 0 | 0.0505 |
| Bottleneck, low exit 4.5 | 177 | 0 | 0 | 0 | 0.0864 | 0.239 | 0 | 0.0156 |
| Bottleneck, low exit 3.5 | 177 | 0 | 0 | 0 | 0.0034 | 0.358 | 0 | 0.0143 |
| Bottleneck, low exit 2.5 | 177 | 0 | 0 | 0 | 0.0067 | 0.428 | 0 | 0.0168 |
| Published gridworld, noise 10% | 174 | 0 | 0 | 0 | 0.0078 | 0.312 | 0 | 0.0026 |
| Published gridworld, noise 0% | 6 | 0 | 0 | 0 | 0.0216 | 0.244 | 0 | 0.0013 |
| Published gridworld, noise 30% | 6 | 0 | 0 | 0 | 0.0025 | 0.292 | 0 | 0.0014 |

- Ending comparisons: K = 4,606. A comparison would fail below p = 2.17e-07. The smallest p-value
  seen is 0.0025, about ten thousand times above that line.
- "Largest gap, points" is the largest difference, in percentage points, between the share of played
  episodes with an ending and the formula's chance. It is 0.55 points at most, at 40,000 episodes.
- Share of p-values under 0.05: 1.1%. It is below 5% because many comparisons have a chance of
  exactly 0 or 1, where the count must match exactly and the p-value is 1.
- 1,012 policy and start pairs in all; about 56 million episodes played.

What this establishes: for these policies and start cells, the exact formulas for the endings and
for the value agree with episodes played by the one-move rules, in all three environments, within
the limits of section 5. The reported outcomes of the trained policies in the two confirmation
blocks therefore do not rest on an error in the formulas.

What it does not establish: anything in the "cannot catch" list of section 5.

Files: `episode_check_result.json` (every comparison), `episode_check_output.txt` (the printed table).
