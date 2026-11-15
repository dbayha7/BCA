# BCA on the published gridworld: what works so far

Prepared November 12 2026 for the research meeting of November 13. A summary of the results in this folder; the slides are the deck "BCA on a Published Gridworld". Every number is exact for the saved policies. Details and limits: [VALIDATION_PUBLISHED.md](VALIDATION_PUBLISHED.md), [RESULTS_VERSION_A.md](RESULTS_VERSION_A.md), [BCA_IW_PLAN.md](BCA_IW_PLAN.md), [RESULTS_VERSION_C.md](RESULTS_VERSION_C.md).

Quick navigation: [The environment](#1-the-environment) · [The setup](#2-the-setup) · [What works](#3-what-works) · [The trials at a glance](#4-the-trials-at-a-glance) · [How firm each result is](#5-how-firm-each-result-is) · [Next](#6-next) · [Words used in the deck](#7-words-used-in-the-deck) · [Presenting the deck](#8-presenting-the-deck)

## 1 The environment

- **Source:** Kumar, Hong, Singh and Levine, "When Should We Prefer Offline Reinforcement Learning Over Behavioral Cloning?", ICLR 2022 (arXiv 2204.05618). The gridworld is from Fu, Kumar, Soh and Levine, ICML 2019.
- **The map:** the paper's "Single Critical" gridworld, 10 by 10, read cell by cell from its Figure 7. Start top left, goal bottom right, walls, four lava cells.
- **The rules:** five moves (stay, up, down, left, right). Each step has a 10% chance of being replaced by a random move. Reward 1 at the goal. Lava ends the run with nothing.
- **Critical cells:** two cells with lava on two sides, where one wrong move is fatal. The best route from the start passes through one of them.
- **The best possible policy** reaches the goal 95.1% of the time.

## 2 The setup

**Taken from the paper unchanged:** the map, the moves, the noise, the reward, the three data recipes and the score.

| Data recipe | What it is |
|---|---|
| Optimal | Runs of the best policy from the start |
| Other starts | The same, from random open cells |
| Half-random | Half best-policy rows, half random moves |

**The score:** the chance of reaching the goal from the start, as a share of the best policy's. 1.00 is perfect.

**Ours, and why:** the learner is TD3+BC with continuous actions, because BCA is built on it; and the settings the paper leaves unstated (20,000 rows per dataset, discount 0.95, 200 steps).

**How each result was produced:** pass marks written before the run; three training seeds per dataset; fresh datasets for every confirmation.

## 3 What works

**1. A plain baseline that works on the published map.**

- With the published setting (alpha 2.5) the baseline scored 0.75 on optimal data, while plainly copying the data scores 1.00.
- The cause was traced: the policy drifted toward commands no row contains, because the critic guessed high there (199 of 199 wrong choices).
- One setting of TD3+BC was changed (alpha 1.0), by a rule written first. On fresh datasets all 15 baselines on optimal data reach the 0.90 mark (mean 0.945).

**2. Version A: BCA's penalty in the policy. Confirmed on optimal data.**

- On nine untouched datasets the score on optimal data goes from 0.953 to 0.990. Higher in 9 of 9 datasets, exact test p = 0.002.
- Episodes ending in lava fall from 8.6% to 5.8%; the best possible is 4.9%.
- No harm where the baseline was already strong (0.986 to 0.987).
- Reading the penalty at the right cell beats a shuffled control on all three recipes.

**3. Importance-weighted BCA that corrects the Q values (Versions C, C2, C3, C4).**

- It gives every command a weight, runs BCA's weighted calibration with those weights, and pushes down the Q value of any command that cannot be certified.
- Checked before any training: with equal weights it gives standard BCA back bit for bit (27 of 27 runs); at both critical cells the fatal moves are flagged and the right move is certified (9 of 9 runs).
- **The critic itself is corrected.** At the first critical cell the critic rates a fatal move above the right one in 0 of 9 runs. For the baseline and for Version A it is 9 of 9.
- **Best scores so far on every recipe** (Version C3): 0.990 on optimal data, 0.990 on other-starts data, 0.956 on the half-random data.
- **Steadier:** on the half-random data every Version C3 run scores 0.888 or more. Version A has a collapsed run there (lowest 0.084).
- **The critic's values are back on the true scale** (Version C4): its error at recorded commands falls from -3.25 to -0.02 on optimal data and from -14.3 to -0.11 on the half-random data.

## 4 The trials at a glance

Three study datasets, nine runs per entry.

| Policy | Optimal | Other starts | Half-random |
|---|---:|---:|---:|
| Baseline (TD3+BC, alpha 1.0) | 0.942 | 0.986 | 0.801 |
| Standard BCA (run November 13, [RESULTS_STANDARD_BCA.md](RESULTS_STANDARD_BCA.md)) | 0.943 | 0.986 | 0.817 |
| Version A | 0.989 | 0.987 | 0.875 |
| Version C | 0.990 | 0.990 | 0.527 |
| Version C2 | 0.990 | 0.990 | 0.924 |
| Version C3 | 0.990 | 0.990 | 0.956 |
| Version C4 | 0.990 | 0.990 | 0.865 |

Version C's dip on the half-random data was one setting feeding on itself across refreshes. C2 fixed it. C3 removed a second term that added up over an episode. C4 scores the critic's target without TD3's noise: the critic's drift is gone, and eight of its nine half-random runs score 0.861 or more (six beat C3), but one run collapses (0.045) and pulls the average down.

## 5 How firm each result is

| Result | Evidence |
|---|---|
| A plain baseline that works | Validated on fresh datasets: 15 of 15 runs at the mark |
| Version A helps on optimal data | Confirmed on nine untouched datasets, p = 0.002 |
| The critic's ranking is corrected | Three tests on three datasets: 0 of 9 runs wrong, down from 9 of 9 |
| Best scores on the half-random data | First tests: 0.956 (Version C3) against the baseline's 0.801. Ahead of Version A on average (0.875), but in only 1 of 3 datasets. Not yet confirmed |
| The critic's values on the true scale | One test (Version C4): error at recorded commands from -3.25 to -0.02. One run of nine collapsed on the half-random data |

Also true, and worth saying if asked: Version A's gain on the half-random data was not confirmed (0.885 to 0.909, p = 0.31). No importance-weighted version has yet passed the mark against Version A on that recipe. Version C3's critic is right in its ranking but sat a few units below the true values; Version C4 removed that drift, but why one of its runs collapses is not established. On the half-random data the critic's ranking at the first critical cell is still wrong in 3 of 9 Version C4 runs (Version C3: 9 of 9).

## 6 Next

1. **Make the fix hold in every run.** Measured in an instrumented rerun: the critic's target scored a noisy next command about 0.5 lower than the un-noised one at every step, and that added up. Version C4 scores the target without the noise and the drift is gone. One of its nine runs on the half-random data collapsed; finding out why comes first.
2. **Confirm on untouched datasets**, the importance-weighted version against Version A and the baseline, with the claims written first.
3. **Calibrate the width to returns.** BCA's width measures one-step error today; the data are whole episodes, so it can be calibrated to the return that actually followed.

## 7 Words used in the deck

The slides avoid technical names. This table connects them to the records.

| In the deck | In the records |
|---|---|
| The critic (called "the scorer" in deck versions before November 13, 09:35) | The critic (the Q function) |
| The standard learner | The baseline: TD3+BC with alpha 1.0 |
| "One setting" of the standard learner | alpha: 2.5 as published, 1.0 here |
| Best-policy data | `optimal` |
| Random-starts data | `other_starts` |
| Half-random data, "the hardest data" | `mixed` |
| Standard BCA | The shared learner's plain `bca` arm: the calibrated width charged in the policy's choice |
| The margin of error | BCA's calibrated width |
| BCA policy guard | Version A |
| BCA critic correction | The importance-weighted line: Versions C, C2, C3, C4 |
| Critic correction: first build | Version C |
| + charge limit held fixed | Version C2 (the cap is set once) |
| + no charge on the look-ahead ("best build") | Version C3 (no charge in the critic's target) |
| + no jitter on the look-ahead ("latest build") | Version C4 (the target is scored without TD3's noise) |
| The charge limit | The cap on the charge |
| The look-ahead | The critic's bootstrap target at the next command |
| BCA "cannot vouch for" a move | "Cannot certify": the weighted calibration returns no finite threshold |
| The jitter on the next move | TD3's target policy smoothing noise |
| "Under 1 chance in 500 of luck" | Exact sign-flip test, p = 0.002 |
| The scorer's error at recorded moves | The critic's error at recorded commands (section 7e of RESULTS_VERSION_C.md) |

## 8 Presenting the deck

The slot is about 15 minutes (David, November 12 2026). The deck is built for that: 13 main slides and 2 backup slides after the close.

Each speaker note has up to three parts. The first part is the script to read aloud. "IF THERE IS TIME" is extra detail to add when ahead of the clock. "IF SOMEONE ASKS" holds answers to likely questions. Times below are for the read-aloud part only, at 140 words a minute.

On the morning of November 13 David said: "dont get too ahead with the versions we need to focus on the setup validation" and "we are introducing a lot of different versions without introducing them". The talk was reordered: the setup validation and standard BCA are in the main talk, each method is defined before its numbers, and the detail of the later versions is in backup.

| Slide | Title | Minutes | Extra in the notes |
|---:|---|---:|---|
| 1 | Helping a learner avoid moves it has no evidence for | 0.4 | |
| 2 | Can a learner trained only on recordings avoid moves nobody recorded? | 1.0 | |
| 3 | A published test, used as published | 0.6 | If there is time |
| 4 | The paper's grid, with two critical cells | 1.2 | If there is time |
| 5 | Three kinds of recordings, one score | 0.8 | If there is time |
| 6 | How the learner works, and where it goes wrong | 1.3 | If there is time |
| 7 | Does a standard learner work on this setup? (the validation table) | 1.4 | If there is time. Asks: why not simply copy the recordings |
| 8 | What standard BCA does | 1.0 | If there is time |
| 9 | Standard BCA: no harm, and a small gain | 1.3 | If there is time |
| 10 | What each addition is, and why we chose it (slide `define`) | 1.4 | If there is time. Asks: what else was tried |
| 11 | Two additions, and what each one scores | 0.6 | If there is time |
| 12 | BCA policy guard: what the policy maximizes (the math; slide `guard-math`) | 1.9 | If there is time. Asks: why standard BCA alone changes so little |
| 13 | Three steps from here | 0.8 | Asks: why one run failed |
| 14 | Three things to remember | 0.8 | Points to the backup slides |
| | **Main path** | **14.7** | |
| 15 | Backup: BCA policy guard: holding the policy back | 1.4 | Asks: would any penalty do the same; which part does the work |
| 16 | Backup: BCA critic correction: fixing the critic itself | 1.2 | |
| 17 | Backup: Score on the hardest data, attempt by attempt | 1.5 | Asks: why averages and not middle values |
| 18 | Backup: The critic stops preferring a step into lava | 0.9 | |
| 19 | Backup: BCA matches or beats the standard learner (chart) | 0.9 | |
| 20 | Backup: What is shown so far, and how firmly | 2.3 | |

At 14.7 minutes the main path has no slack in a 15-minute slot. The slide to move to backup first is slide 12 (the math, 1.9 minutes): slide 10 already defines the policy guard in words.

Slide 10 was added at David's request ("need another slide to define policy guard and scorer correction", "and why did we choose these versions"). David then said: "the scorer is the critic right? use critic instead of scorer if thats the case". The deck now says "critic" everywhere and the second addition is called "BCA critic correction".

Slide 11 was added on November 13 by a forked session, after David asked there: "i need to see the policy guard math on a slide to make it clear for everyone" and "W, s and m need to be defined on the slide". It shows what the policy maximizes for the standard learner, standard BCA and the policy guard, defines every symbol, and works one example at the first critical cell (random-starts data, from `results/version_a_published`). Its numbers are that session's and are not among the 85 checked by `check_meeting_numbers.py`.

**Every number checked.** On November 13 2026 the numbers on the slides were recomputed from the saved result files with `check_meeting_numbers.py` (reads only, trains nothing): 85 numbers, 0 mismatches, including the validation table and standard BCA. From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_critical_v1/check_meeting_numbers.py
```

Three figures on the slides are taken from the written records and are not recomputed by that script: "about 940 recorded steps" at the first critical cell (942, [VALIDATION_PUBLISHED.md](VALIDATION_PUBLISHED.md) section 4), "199 of 199 wrong choices" (the same file, section 8), and the pre-training check "at both critical cells, in 9 of 9 runs" ([BCA_IW_PLAN.md](BCA_IW_PLAN.md) section 4).

**The hardest question to expect** is answered in the notes of slide 7: plainly copying the recordings scores 1.00 on this map (0.997 on the half-random data), which no learner here beats. The script's answer: copying cannot do better than what was recorded, the paper's own question is when a value-based learner should be preferred, and this map is used here only as a clean published case of a standard learner overrating moves nobody recorded.
