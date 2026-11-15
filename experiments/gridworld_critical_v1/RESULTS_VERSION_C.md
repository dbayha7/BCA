# Version C on the published map: first test

Prepared November 12 2026 for David. Version C is importance-weighted BCA that acts on the Q values ([BCA_IW_PLAN.md](BCA_IW_PLAN.md)). The pass marks are Amendment 6 of [PROTOCOL.md](PROTOCOL.md), written before any Version C policy was trained. Run once. **A first test, not a confirmation.**

Quick navigation: [At a glance](#the-whole-version-c-line-at-a-glance) · [Verdict](#1-verdict) · [What was run](#2-what-was-run) · [Scores](#3-scores) · [The marks](#4-the-marks) · [What worked](#5-what-worked) · [What failed, and why](#6-what-failed-and-why) · [Next step](#7-next-step-yours-to-decide) · [Version C2](#7b-version-c2-the-cap-set-once) · [Version C3](#7c-version-c3-no-charge-in-the-critics-target) · [The instrumented run](#7d-the-instrumented-run-what-still-lowers-the-values) · [Version C4](#7e-version-c4-the-target-scored-without-the-noise) · [Limits](#8-limits) · [Files](#9-files-and-commands)

## The whole Version C line at a glance

Added November 12 2026, after Version C4. Means of nine runs on the three study datasets (seeds 6401 to 6403). Sections 1 to 7 below are the first version; each later version has its own section.

| Version | The one change | Optimal | Other starts | Half-random | Marks V1 to V4 | Section |
|---|---|---:|---:|---:|---|---|
| Baseline | | 0.942 | 0.986 | 0.801 | | |
| Version A | | 0.989 | 0.987 | 0.875 | | |
| Version C | First importance-weighted version | 0.990 | 0.990 | 0.527 | V1, V3 pass. V2, V4 fail | 1 to 7 |
| Version C2 | The cap is set once | 0.990 | 0.990 | 0.924 | V1 to V3 pass. V4 fails | 7b |
| Version C3 | No charge in the critic's target | 0.990 | 0.990 | **0.956** | V1 to V3 pass. V4 fails | 7c |
| Version C4 | The target is scored without TD3's noise | 0.990 | 0.990 | 0.865 | V1 to V3 pass. V4 fails | 7e |

- **Every version corrects the critic's ranking** at the first critical cell on optimal and random-starts data (0 of 9 runs wrong; baseline and Version A 9 of 9).
- **No version passes V4**, "better than Version A on the half-random data". C2 and C3 are ahead on average but in only 1 of 3 datasets. C4 is ahead in 2 of 3 datasets but 0.010 behind on average, because one run of nine collapsed.
- **Best average on the half-random data: Version C3.**
- **Only version whose critic is on the true scale at recorded commands: Version C4** (error -0.02, -0.05 and -0.11, against -3.25, -2.55 and -14.3 for Version C3).
- All of it is development on the same three datasets. Nothing here is confirmed on new datasets.

## 1 Verdict

**Version C does not pass its first test. Two marks pass and two fail.**

- **It does what it was built for where the gap in the data is real.** At the first critical cell the critic no longer rates a fatal move above the right one: 0 of 9 runs, against 9 of 9 for the baseline and for Version A. No earlier version changed that. Its scores on optimal and random-starts data are the best so far (0.990 on both).
- **It fails badly on the half-random data**: 0.527, against 0.801 for the baseline and 0.875 for Version A.
- **The cause is found and it is one setting**, the cap. It fed on itself during training (section 6).

## 2 What was run

- The three study datasets (seeds 6401, 6402, 6403), three recipes, training seeds 20, 21, 22: 27 Version C policies on the GPU, every calibration and every importance-weighted table usable.
- Each run starts from the baseline's saved state after 10,000 updates, trains its weight classifier 10,000 updates, then trains to 25,000 with refreshes after 15,000 and 20,000.
- Compared with the baseline and Version A on the same datasets and training seeds.

## 3 Scores

Means of nine runs. The score is the paper's: the chance of reaching the goal from the start, as a share of the best possible policy's.

| Recipe | Baseline | Version A | **Version C** | Version C's lowest run | Version C runs below 0.50 |
|---|---:|---:|---:|---:|---|
| Optimal-policy runs | 0.942 | 0.989 | **0.990** | 0.985 | 0 of 9 |
| Same, from random start cells | 0.986 | 0.987 | **0.990** | 0.987 | 0 of 9 |
| Half optimal, half random moves | 0.801 | 0.875 | 0.527 | 0.014 | 4 of 9 |

On the half-random data, per dataset: 0.927, 0.039, 0.615.

## 4 The marks

| Mark | Result | Verdict |
|---|---|---|
| **V1** The Q values are corrected | At the first critical cell the critic rates a fatal move above the way through in 0 of 9 runs on optimal data and 0 of 9 on random-starts data. Baseline and Version A: 9 of 9 on both | **PASS** |
| **V2** Better than the baseline | Optimal: +0.048, higher in 3 of 3 datasets. Half-random: -0.274, higher in 1 of 3 | **FAIL** (half-random) |
| **V3** As good as Version A where Version A is strong | Optimal: +0.002, lava endings 0.1 points lower. Random-starts: +0.003, lava 0.3 points lower | **PASS** |
| **V4** Better than Version A where it is weak | Half-random: -0.348, higher in 0 of 3 datasets | **FAIL** |

My predictions: V1 passes (right). V2 passes on optimal (right) and is unsure on half-random (it failed). V3 is the mark most at risk (wrong: it passed). V4 is a coin flip (it failed, clearly).

## 5 What worked

- **The critic's ranking at the critical cell is fixed.** This is the first version in which the critic itself stops preferring a fatal move there.
- **The importance weights flag the right commands.** At the first critical cell the fatal moves cannot be certified in 8 of 9 runs on optimal data and 9 of 9 on random-starts data. At the second critical cell: 9 of 9 on both.
- **The policy never commands a fatal move** at either critical cell on optimal or random-starts data.
- **The scores there are the highest and the steadiest so far**: no run below 0.985.

## 6 What failed, and why

**The cap ran away.** The cap is the charge for a command that cannot be certified. I defined it as half the range of the critic's own targets on held-out rows, recomputed at every refresh. Version C pushes Q values down by up to three times the cap. That widens the range of the targets, so the next refresh sets a larger cap, which pushes further.

| | Cap at update 10,000 | at 15,000 | at 20,000 |
|---|---:|---:|---:|
| Optimal data | 0.50 | 4.39 | 30.3 |
| Random-starts data | 0.50 | 3.74 | 30.6 |
| Half-random data | 0.77 | 6.31 | 86.1 |

What followed from that:

- **The critic's scores left the possible range in 27 of 27 runs.** No score can truly be above 1 or below 0. At the fatal moves Version C's critic ends at about -360 to -1,500. The ranking is right, the size is far wrong.
- **On optimal and random-starts data the policy survives**, because only uncertified commands are pushed down and the recorded command stays on top.
- **On the half-random data it does not.** Thinly recorded commands there sit close to the "cannot certify" line. With the cap at 86, the right move at the critical cells is charged about 67, and the policy commands a fatal move 22% of the time at the first critical cell.

In Version A the cap only shaped the policy's penalty and never touched the critic, so it could not feed on itself. I did not see this loop before the run. The unit tests check each equation on its own, not what three refreshes do together.

## 7 Next step, yours to decide

**My recommendation: fix the cap and test again as a new version.** Fix the cap once, at update 10,000, when the run is still the plain baseline, and never recompute it. That is one line in the runner. It would be a new version (C2) with its own marks written first, because this test's result stands as a fail.

Whether the half-random data then work is not certain. The soft spot found in the checks remains: commands missing by chance there are treated as nearly uncertifiable.

## 7b Version C2: the cap set once

Run November 12 2026 under Amendment 7 of [PROTOCOL.md](PROTOCOL.md), written before any Version C2 policy was trained. David said: "Yes, fix and test". **Development**: the fix was chosen after Version C failed on these same three datasets. Run once.

**The one change.** The cap is set at update 10,000, while the run is still the plain baseline, and never recomputed. Everything else is Version C. 27 runs on the GPU, every calibration and table usable.

**Verdict: three of four marks pass. Version C2 does not pass, on V4.**

| Recipe | Baseline | Version A | Version C | **Version C2** | C2's lowest run | C2 runs below 0.50 |
|---|---:|---:|---:|---:|---:|---|
| Optimal-policy runs | 0.942 | 0.989 | 0.990 | **0.990** | 0.981 | 0 of 9 |
| Same, from random start cells | 0.986 | 0.987 | 0.990 | **0.990** | 0.987 | 0 of 9 |
| Half optimal, half random moves | 0.801 | 0.875 | 0.527 | **0.924** | 0.842 | 0 of 9 |

| Mark | Result | Verdict |
|---|---|---|
| **V1** The Q values are corrected | The critic rates a fatal move above the way through in 0 of 9 runs on optimal data and 0 of 9 on random-starts data. Baseline and Version A: 9 of 9 | **PASS** |
| **V2** Better than the baseline | Optimal: +0.048, 3 of 3 datasets. Half-random: +0.123, 3 of 3 datasets | **PASS** |
| **V3** As good as Version A where Version A is strong | Optimal: +0.001. Random-starts: +0.003, lava endings 0.3 points lower | **PASS** |
| **V4** Better than Version A where it is weak | Half-random: +0.048 on average, but higher in only 1 of 3 datasets | **FAIL** |

**What Version C2 does.**

- **The fix worked.** The half-random score goes from 0.527 to 0.924, and no run collapses.
- **It is the best version so far on the half-random data on average**, 0.924 against 0.875 for Version A and 0.801 for the baseline, and the only one with no run below 0.84 there.
- **It keeps Version C's two gains**: the critic's ranking at the critical cell is right in every run, and both strong recipes score 0.990.

**Why V4 still fails.** Per dataset on the half-random data, Version C2 scores 0.948, 0.908, 0.914. Version A scores 0.983, 0.972, 0.671. Version A is higher in the two datasets where none of its runs collapsed; Version C2 is higher in the one where a Version A run did. So Version C2 is steadier, and Version A is better when it works. Version C2's lava endings there are 12.1%, against 6.3% for Version A.

**The critic's values are still off scale, and I think I know why.** The cap now stays put (0.50 at every refresh), but the critic's scores still drift: at the fatal moves they end near -10 to -25, and the largest score in a run is 14 to 122. No true value is outside 0 to 1.

| | Cap used | Half the range of the critic's targets at update 10,000 | at 15,000 | at 20,000 |
|---|---:|---:|---:|---:|
| Optimal data | 0.50 | 0.50 | 4.39 | 7.55 |
| Half-random data | 0.77 | 0.77 | 6.31 | 13.85 |

My reading, not yet measured: the second use, lowering the next step's value by the charge. The next command it charges is the policy's command plus the small noise TD3 adds to it. The data hold exact moves, so a command with noise on it is no longer an exact recorded move, and it gets charged at almost every step. A charge of up to 1.5 at every step adds up over an episode to about 30, which is the size of the drift. The ranking survives because the shift is roughly the same for every command, which is why the scores stay high. This was the first risk listed in the plan.

**The next step I would take.** Drop the second use, or apply it to the policy's command without the added noise. The first use, the direct correction, is what fixes the ranking; the second is what makes the values drift. That would be a new version with its own marks.

My predictions: the cap no longer moves (right). V1 and V3 pass (right). A clear improvement on the half-random data (right: 0.527 to 0.924). Whether V2 and V4 hold there I did not know: V2 passes, V4 fails.

Result files: `results/version_c2_published/` (`version_c.md`, `version_c.json`, `version_c_records.json`), run by `run_version_c2_published.py`, judged by `judge_version_c.py --method version_c2`.

## 7c Version C3: no charge in the critic's target

Run November 12 2026 under Amendment 8 of [PROTOCOL.md](PROTOCOL.md), written before any Version C3 policy was trained. David said: "Yes, test it". **Development**: a fourth look at the same three datasets. Run once. 27 runs on the GPU, every calibration and table usable.

**The one change from Version C2.** The critic is trained toward the plain target again. The direct correction of the Q values and the charge in the policy's choice stay; the cap stays set once.

**Verdict: three of four marks pass again. Version C3 does not pass, on V4. It is the best version so far on every recipe on average.**

| Recipe | Baseline | Version A | Version C2 | **Version C3** | C3's lowest run | C3 runs below 0.50 |
|---|---:|---:|---:|---:|---:|---|
| Optimal-policy runs | 0.942 | 0.989 | 0.990 | **0.990** | 0.983 | 0 of 9 |
| Same, from random start cells | 0.986 | 0.987 | 0.990 | **0.990** | 0.985 | 0 of 9 |
| Half optimal, half random moves | 0.801 | 0.875 | 0.924 | **0.956** | 0.888 | 0 of 9 |

| Mark | Result | Verdict |
|---|---|---|
| **V1** The Q values are corrected | 0 of 9 runs on optimal data, 0 of 9 on random-starts data (baseline and Version A: 9 of 9) | **PASS** |
| **V2** Better than the baseline | Optimal: +0.048. Half-random: +0.155. 3 of 3 datasets each | **PASS** |
| **V3** As good as Version A where Version A is strong | Optimal: +0.001. Random-starts: +0.003 | **PASS** |
| **V4** Better than Version A where it is weak | Half-random: +0.081 on average, but higher in only 1 of 3 datasets | **FAIL** |

**What Version C3 does.**

- **Half-random data: 0.956, up from 0.924**, with lava endings down from 12.1% to 9.0%. Every run scores 0.888 or more.
- **It beats the baseline on the half-random data by 0.155** and keeps 0.990 on the other two recipes.
- **The critic's ranking at the critical cell is right in every run** on optimal and random-starts data.

**Why V4 still fails.** Per dataset on the half-random data: Version C3 0.968, 0.957, 0.944; Version A 0.983, 0.972, 0.671. Version A is ahead by 0.015 in the two datasets where none of its runs collapsed. Version C3 is ahead by 0.273 in the one where a Version A run did.

**The critic's values: better, still off scale.** My prediction was that they would stay within a few units of the true scale. They are about half of Version C2's, which is not that.

| | Version C | Version C2 | Version C3 |
|---|---:|---:|---:|
| Critic's score at a fatal move, optimal data (true value about 0.04) | -362 | -10 | -5.5 |
| Largest score in a run, optimal data (true maximum 1.0) | 195 to 808 | 14 to 31 | 9 to 13 |
| Largest score in a run, half-random data | 514 to 8,787 | 28 to 122 | 15 to 63 |

So dropping the charge from the target removed about half of the drift. My explanation was half right. Something still lowers the values step after step, and I have not measured what. My reading: the direct correction pushes down commands near the ones the critic's own target looks at, because TD3 adds noise to the next command before scoring it.

**On the half-random data the drift costs something.** The critic there rates a fatal move above the right one at the first critical cell in 9 of 9 runs, and the right move is charged as much as a fatal one (about 7). The policy commands a fatal move 1.2% of the time there, against 0.0% for Version A. Thin cells are where an inflated scale hurts.

**The next step I would take: measure before changing anything again.** Two fixes each removed about half of the problem, and both were found by guessing at the cause after the fact. One instrumented run would record, at each refresh, how often the commands the target looks at are flagged, what they are charged, and how far the critic's score at recorded commands is from the truth. That says what the third fix should be.

My predictions: V1, V3 and V2 on optimal pass (right). At least 0.924 on the half-random data with fewer lava endings (right: 0.956 and 9.0%). The critic's scores within a few units of scale (wrong: 9 to 63).

Result files: `results/version_c3_published/`, run by `run_version_c3_published.py`, judged by `judge_version_c.py --method version_c3`.

## 7d The instrumented run: what still lowers the values

Run November 12 2026 under Amendment 9 of [PROTOCOL.md](PROTOCOL.md). A measurement with no pass marks and no new method. David said: "Yes, measure it". Version C3 was trained again with extra read-outs at each refresh. **It reproduced Version C3's policies exactly** (largest difference in any preference: 0).

Means over nine runs, on held-out rows, at update 20,000:

| Read-out | Optimal | Other starts | Half-random |
|---|---:|---:|---:|
| Next commands flagged "cannot certify", with TD3's noise | 5.9% | 11.1% | 15.4% |
| The same, without the noise | 0.1% | 0.9% | 13.1% |
| Critic's score at the next command, with noise minus without | -0.49 | -0.53 | -1.23 |
| Critic's error at recorded commands | -3.25 | -2.55 | -14.3 |
| The same at update 10,000, before Version C3 starts | -0.11 | -0.12 | -0.36 |
| Comparison commands flagged | 75% | 78% | 52% |
| Critic's score at flagged comparison commands | -7.9 | -8.1 | -26.1 |

**What it shows.**

- **The drift is at the recorded commands themselves.** The critic's error there goes from about -0.1 to about -3 on optimal data, and to -14 on the half-random data. Every value sinks together.
- **The leak is through TD3's noise.** The command the target scores, with the noise on it, scores about half a unit lower than the same command without it. That loss is taken at every step. With a discount of 0.95 it adds up to about ten units.
- **It is not mainly outright flagging.** With the noise the next command is flagged only 6% to 15% of the time. The critic has simply become steep around each recorded command, because nearly everything else is pushed down.
- **On the half-random data the policy's own next command is flagged 13% of the time even without noise.** That is the thin-cell soft spot from the checks, and a second thing to fix there.

**This is the first outcome listed in Amendment 9.** The fix it points to: score the critic's target at the un-noised command for this version. Nothing has been changed in response.

## 7e Version C4: the target scored without the noise

Run November 12 2026 under Amendment 10 of [PROTOCOL.md](PROTOCOL.md), written before any Version C4 policy was trained. David said: "Yes, test it". **Development**: a fifth look at the same three datasets. Run once. 27 runs on the GPU, every calibration and table usable. The first smoke test was interrupted during training and left only a start receipt (`results/smoke_version_c4/`); it was rerun under a new label (`results/smoke_version_c4_b/`) and passed.

**The one change from Version C3.** The critic's target scores the next command as the policy gives it, with none of TD3's added noise. Everything else is Version C3.

**Verdict: three of four marks pass. Version C4 does not pass, on V4. The drift is gone. On the half-random data one run of nine collapses, and that run costs the average.**

| Recipe | Baseline | Version A | Version C3 | **Version C4** | C4's lowest run | C4 runs below 0.50 |
|---|---:|---:|---:|---:|---:|---|
| Optimal-policy runs | 0.942 | 0.989 | 0.990 | **0.990** | 0.985 | 0 of 9 |
| Same, from random start cells | 0.986 | 0.987 | 0.990 | **0.990** | 0.986 | 0 of 9 |
| Half optimal, half random moves | 0.801 | 0.875 | 0.956 | **0.865** | 0.045 | 1 of 9 |

| Mark | Result | Verdict |
|---|---|---|
| **V1** The Q values are corrected | 0 of 9 runs on optimal data, 0 of 9 on random-starts data (baseline and Version A: 9 of 9) | **PASS** |
| **V2** Better than the baseline | Optimal: +0.048, 3 of 3 datasets. Half-random: +0.064, 2 of 3 datasets | **PASS** |
| **V3** As good as Version A where Version A is strong | Optimal: +0.001. Random-starts: +0.003 | **PASS** |
| **V4** Better than Version A where it is weak | Half-random: -0.010 on average, higher in 2 of 3 datasets | **FAIL** |

**What Version C4 fixes: the critic's values at recorded commands are back on the true scale.** Means over nine runs at update 20,000, from the same read-outs as section 7d:

| Read-out | Version C3: optimal | other starts | half-random | **Version C4**: optimal | other starts | half-random |
|---|---:|---:|---:|---:|---:|---:|
| Critic's error at recorded commands | -3.25 | -2.55 | -14.3 | **-0.02** | **-0.05** | **-0.11** |
| Residual unit | 0.40 | 0.46 | 1.12 | 0.05 | 0.08 | 0.11 |
| Critic's score at flagged comparison commands | -7.9 | -8.1 | -26.1 | -4.7 | -4.8 | -10.6 |
| Critic's score at the next command, with noise minus without | -0.49 | -0.53 | -1.23 | -0.50 | -0.55 | -1.20 |

- **The drift is gone.** The error at recorded commands is within 0.05 of the truth on optimal and random-starts data and within 0.11 on the half-random data. That was the purpose of the change.
- The critic is still steep around each recorded command (the last row is unchanged). The target simply no longer reads it.
- **The critic's ranking is right on optimal and random-starts data** in 9 of 9 runs each. On the half-random data it is wrong at the first critical cell in 3 of 9 runs. Version C3: 9 of 9. Baseline: 5 of 9. Version A: 0 of 9.
- **The critic's scores still leave the possible range** in 9 of 9 runs on optimal and on random-starts data and 7 of 9 on the half-random data. Unsupported commands are pushed to about -5, which the method intends. But the largest score in a run is 4 to 9 on optimal data and 6 to 10 on random-starts data (Version C3: 9 to 13 and 11 to 15), where no true value is above 1. I have not looked at which commands those are.

**The half-random data, run by run.**

| Dataset, training seed | Baseline | Version C3 | Version C4 |
|---|---:|---:|---:|
| 6401, 20 | 0.586 | 0.986 | 0.996 |
| 6401, 21 | 0.987 | 0.939 | 0.861 |
| 6401, 22 | 0.739 | 0.978 | **0.045** |
| 6402, 20 | 0.973 | 1.000 | 1.002 |
| 6402, 21 | 0.915 | 0.915 | 0.977 |
| 6402, 22 | 0.750 | 0.956 | 0.986 |
| 6403, 20 | 0.608 | 0.998 | 0.981 |
| 6403, 21 | 0.942 | 0.888 | 0.958 |
| 6403, 22 | 0.707 | 0.945 | 0.979 |

- **Version C4 is higher than Version C3 in 6 of 9 runs**, and its middle run scores 0.979 against 0.956. Eight of nine runs score 0.861 or more.
- **One run collapses** (dataset 6401, training seed 22: 0.045). Per dataset: Version C4 0.634, 0.988, 0.972; Version A 0.983, 0.972, 0.671; Version C3 0.968, 0.957, 0.944. So Version C4 is ahead of Version A in two datasets and far behind in the one with the collapsed run.
- Lava endings on the half-random data: 16.2%, against 9.0% for Version C3 and 6.3% for Version A. Most of that is the collapsed run (86%).

**The collapsed run, read from the saved policy.**

- The policy commands "down" at 86% of open cells. From the start it reaches the bottom edge at the cell left of a lava cell (cell 92), keeps commanding "down" into the edge, and the 10% random moves push it into the lava (cell 93) in 85% of episodes.
- **The policy and the critic have come apart there.** One cell earlier on the route (cell 82) the critic rates "right", the correct move, 3.7 above "down". The policy commands "down", the least-charged move at that cell (charge 4.7, against 8.1 for "right").
- The policy commands no fatal move at either critical cell. The failure is on the way to them.
- **Charges on the half-random data are large for every move.** The move the policy commands is charged 7.1 on average there, against 1.1 on optimal and 0.2 on random-starts data. At the first critical cell the right move is charged 7.0 and a fatal one 6.3. This is the thin-cell soft spot of the checks, and removing the drift did not touch it.
- I have not established why this run collapses and the other eight do not. The same run under Version C3 scored 0.978.

**My reading, not tested.** The charge is capped in the critic's two uses and not in the policy's choice ([BCA_IW_PLAN.md](BCA_IW_PLAN.md), section 4.1). On the half-random data nearly every move at a thin cell is charged well above the cap of about 1.05. So the policy is steered by differences between charges that are all beyond the cap, and at some cells those differences outweigh the critic's ranking. In Version C3 the critic's values were inflated about tenfold, which may have hidden this.

**The next step I would take** (a new version with its own marks, yours to decide): cap the charge in the policy's choice at the same cap as in the critic. Moves that BCA cannot vouch for would then all be charged the same, and the corrected critic would decide between them. It is one line in a new wrapper file. It would be a sixth look at these three datasets, so whatever it shows, the real test is still confirmation on new datasets.

My predictions: the critic's error at recorded commands within half a unit on optimal and random-starts data (right: -0.02 and -0.05). V1, V3 and V2 pass (right). At least 0.956 on the half-random data (wrong: 0.865, one collapsed run). The drift there may shrink less (wrong: it shrank from -14.3 to -0.11). V4: I did not know (it fails).

Result files: `results/version_c4_published/` (with `drift.json`), run by `run_version_c4_published.py`, judged by `judge_version_c.py --method version_c4`, read-outs printed by `summarize_drift.py --label version_c4_published`.

## 8 Limits

- First test: three datasets, three training seeds each.
- The marks follow two earlier looks at these datasets.
- Version C's update step matches the shared learner's to float rounding only.
- The weight classifier's comparison set uses random cells, which suits a map of 100 cells and is not a general recipe.
- The width still measures one-step error.

## 9 Files and commands

| File | What it is |
|---|---|
| `BCA_IW_PLAN.md` | The version, the four checks and what they found |
| `PROTOCOL.md`, Amendment 6 | The marks, the decision rule and the predictions, written before any Version C policy was trained |
| `version_c.py`, `test_version_c.py` | The method and its 14 tests |
| `iw_checks.py`, `results/iw_checks_published_validation_2*.json` | The four checks, before and after the classifier was corrected |
| `run_version_c_published.py`, `results/version_c_published/` | The run: policies, read-outs, charges, refresh records, run receipts |
| `judge_version_c.py` | Computes every number here: `version_c.md`, `version_c.json`, `version_c_records.json` |
| `results/smoke_version_c/` | The smoke test on dataset seed 7 |
| `run_version_c4_published.py`, `results/version_c4_published/` | Version C4: policies, charges, refresh records, `drift.json`, run receipts; judged with `--method version_c4` |
| `results/smoke_version_c4_b/` | Version C4's smoke test (`results/smoke_version_c4/` is the interrupted first attempt: a start receipt only) |

To re-derive every number without a GPU, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_critical_v1/judge_version_c.py --label version_c_published --version-a version_a_published
```
