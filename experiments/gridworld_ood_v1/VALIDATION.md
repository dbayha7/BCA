# Verification and validation of the two environments

Prepared November 9 2026 for David. This file covers the environments only. It says nothing about how BCA does; that is in [RESULTS.md](RESULTS.md) and was written after this.

Quick navigation: [Verdict](#1-verdict) · [What the two words mean here](#2-what-the-two-words-mean-here) · [Verification](#3-verification-is-it-built-right) · [Validation](#4-validation-is-it-the-right-test) · [What was fixed](#5-what-was-fixed-and-what-was-not) · [Not validated](#6-what-is-not-validated)

## 1 Verdict

| Environment | Network input | Verified | Validated | Used for BCA |
|---|---|---|---|---|
| Honeypot | one-hot cell | Yes | Yes | Yes |
| Bottleneck revision 2, exit 4.5 / 3.5 / 2.5 | one-hot cell | Yes | Yes, all three | Yes |
| Bottleneck revision 2, exit 4.5 / 3.5 / 2.5 | two coordinates | Yes | **No** (8, 8 and 7 of 9 runs) | No |
| Bottleneck revision 1 | either | Yes | **No**, superseded | No |

Everything here uses development seeds. "Validated" means these environments passed the checks set out in [PROTOCOL.md](PROTOCOL.md) on those seeds. It is not a claim about untouched data.

## 2 What the two words mean here

- **Verified** means the environment and its data are built the way the design says: the moves, rewards and execution rule behave as written, and the withheld move is absent from the data.
- **Validated** means the environment can actually test the question: with complete data the plain baseline is competent, and withholding the move changes what the baseline does, in the predicted direction. Both are judged with baseline policies only. No BCA or control policy is involved.

## 3 Verification: is it built right?

| Check | Result |
|---|---|
| Environment contract tests: execution rule, every move at every cell, layout, best possible policy, parity with the archived environment when the changes are switched off | 27 of 27 pass |
| Of those, tests specific to Bottleneck revision 2: it differs from revision 1 only in hazard 58 and the cost of a blocked move; no hazard on the opening's row; stepping right from C never meets a hazard; the best route from B passes no hazard; crossing is still worth 5.563 | 8 of 8 pass |
| Collector and unseen-move checks, including planted violations that must be caught and the archived datasets that must fail | 14 of 14 pass |
| Unseen-move checks on the Honeypot datasets | 9 of 9 datasets pass |
| Unseen-move checks on the Bottleneck revision 2 datasets | 27 of 27 datasets pass |
| Learner tests: each equation against an independent NumPy version, gradient ownership, update order, matching of arms | 21 of 21 pass, on CPU and on the GPU |
| Exact-evaluation tests | 6 of 6 pass |
| The Honeypot is unchanged by the Bottleneck fix | All 36 earlier datasets still pass under the changed code, with unchanged content hashes |

The withheld move in the data, counted per decision cell:

| | Complete arm | Targeted arm | Placebo arm |
|---|---:|---:|---:|
| Honeypot: entering hazard 34 | 27 to 41 times | **0** | 38 to 53 times |
| Bottleneck revision 2: stepping right at B | 158 to 163 times | **0** | 166 to 171 times |

Every decision cell keeps its full 256 rows in every arm, and the arms are byte-identical outside the decision cells.

## 4 Validation: is it the right test?

### 4.1 Honeypot

Datasets 5101 to 5103, three training seeds each, baseline only.

| Question | Evidence | Verdict |
|---|---|---|
| Is the baseline competent with complete data? | 9 of 9 runs reach the goal at least 90% of the time with value at least 3 from each of cells 24, 33, 35, 44. Worst cell in any run: 91.1% goal, value 4.00 | Pass |
| Does withholding the move mislead the critic? | Error on the hazard-entering command: +0.5 with complete data, **+10.0** with the move withheld. The critic's margin for the best move over the fatal one falls from 9.9 to 0.6, and is negative on average at the two cells where the hazard lies ahead | Yes |
| Does it change what the baseline does, in the predicted direction? | Executed probability of entering hazard 34, mean of the four cells: 0.36% complete, 0.37% placebo, **4.29%** targeted. Higher in the targeted arm in 3 of 3 datasets, by more than the complete-versus-placebo spread | Pass |
| Does the harmless placebo gap leave the baseline alone? | Placebo matches complete (0.37% against 0.36%) | Yes |
| Do policies stall? | 0 stalled cells and 0% time-outs in all 9 complete-data baselines | No |

### 4.2 Bottleneck revision 2, one-hot input

Datasets 5201 to 5203, three training seeds each, baseline only.

| Question | Exit 4.5 | Exit 3.5 | Exit 2.5 | Verdict |
|---|---|---|---|---|
| Competent with complete data: from C, goal at least 90% and value at least 3; from B, hazard plus time-out at most 10% | 9 of 9 | 9 of 9 | 9 of 9 | Pass |
| Goal rate from C, complete data | 99.9% | 100% | 100% | |
| Hazard or time-out from B, complete data | 0% | 0% | 0% | |
| Step right at B: complete / placebo / **targeted** | 64% / 76% / **5%** | 78% / 81% / **7%** | 79% / 83% / **11%** | |
| Targeted lower than placebo by more than the spread | 3 of 3 datasets | 3 of 3 | 3 of 3 | Pass |
| From B in the targeted arm: goal / low exit | 5% / 95% | 8% / 92% | 13% / 87% | |
| Critic's error on the crossing command: complete / targeted | -0.14 / +0.31 | -0.25 / -0.71 | -0.19 / -1.78 | |
| Bad endings over all 80 start cells, complete data | 0.5% | 0.6% | 0.6% | |

- **The placebo gap behaves.** Withholding a harmless move at B leaves crossing at or slightly above the complete-data level.
- **The critic undervalues the unseen crossing at the two lower exit values**, by 0.7 and 1.8. At exit 4.5 it does not. So "a missing link makes the bridge look worse than it is" holds at exits 3.5 and 2.5.
- **Most of the baseline's reluctance is imitation.** With the crossing withheld, the demonstrator at B takes the low exit, and the baseline follows it.

### 4.3 Bottleneck revision 2, coordinate input: not validated

| | Exit 4.5 | Exit 3.5 | Exit 2.5 |
|---|---|---|---|
| Runs passing the competence gate | 8 of 9 | 8 of 9 | 7 of 9 |

The failing runs reach the goal every time from B and C with no hazard or time-out, but their value at C is 1.35 to 1.51 against a required 3. They waste steps into walls, and each of those now costs 1.0. The fix removed the fatal fault (hazard 58 took 65 to 75% of episodes from C in revision 1; it is 0% now), but not this network's sloppiness. The gate was not moved, so no BCA or control policy was trained with coordinate input.

## 5 What was fixed, and what was not

| Item | Status |
|---|---|
| Hazard 58 on the opening's row | Fixed in revision 2: removed from the Bottleneck |
| A blocked move cost the same as a step, so stalling looked almost as good as the best move | Fixed in revision 2: a blocked move costs 1.0 in the Bottleneck. The gap to the best move went from 0.23 to 0.60 up to 1.13 to 1.50 |
| The competence gate at B demanded a 90% goal rate although a low exit sits beside B | Corrected in Amendment 1, before any study data were used |
| Honeypot | Nothing needed fixing; unchanged |
| Rewards now differ between the tasks | A blocked move costs 0.1 in the Honeypot and 1.0 in the Bottleneck. Left as is, because the Honeypot showed no stalls and its results had already been seen |

Both Bottleneck fixes were chosen from baseline runs on complete data, checked first on a dataset seed outside every study, and then validated on fresh datasets. 1.0 was the only blocked-move cost tried.

## 6 What is not validated

- **The far start A.** The A-start criterion (goal at least 90%, value at least 1.5) was reported, not gated. It passes in 6 of 9 Honeypot baselines and 0 of 27 one-hot Bottleneck baselines. In the Bottleneck that is mostly the low exit being taken some of the time, which is a legitimate ending but not the goal. No claim about the full journey from A should be made.
- **Coordinate input in the Bottleneck.** See 4.3.
- **Anything beyond these seeds.** Three datasets with three training seeds each, all development.
- **The learner against the archived PyTorch host.** It is a re-implementation, checked equation by equation, not bit for bit.
