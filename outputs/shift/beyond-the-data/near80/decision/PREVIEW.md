# Near-80 version 9, preview: BCA's two groups pooled from every held-out row outside the ball

**Analysis only, in the pinned environment.** Nothing official was run, nothing was approved, no actor or critic was
trained, nothing was written under any `runs/` folder or in any clone. Every file of this preview is under
`scratchpad\v9preview\`. The numbers come from version 8's own run files and version 8's own code, recomputed in the
project's pinned environment (WSL Ubuntu-24.04, `/tmp/claude-0/bcavenv`: Python 3.10.20, numpy 2.2.6, scipy 1.15.3,
jax 0.6.2, flax 0.10.7, `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=1`), the environment version 8 ran in. A first pass of
the same computation in a Windows copy of the pins (written earlier today, when the WSL watcher was down) is kept as a
cross-check: it agrees to the bit on every score and every count (section 7). Each number's source is in
`preview.json` (`inputs` for the code files, `seed_worlds[*].sources` for the records read, `seed_worlds[*].checks`
for what was verified about each).

## 0. The short answer

- **Pooling ends the refusal on every spent seed-world.** Under version 9's rule the full-throttle group has 35-135
  rows on the 20 seed-worlds (version 8: 18-60) and the rest 41,476-42,997 (version 8: 825-878). BCA's tail
  threshold is finite on all 20 (version 8: `+inf` on 6, of which 3, seeds 10, 12 and 13 in the stalling world,
  stopped the actor phase). The smallest pooled group, seed 10 stalling, has 35 rows, where the exact refusal chance is
  2.5e-6; it falls about tenfold per further row.
- **No call of BCA's switch changes** on the five fresh seeds that have a second critic (stalling world on, no-stall
  world off, 10 of 10, as recorded), and the mean-difference switch (margin 0.10) makes BCA's call in all 10 under
  pooling, as it did in all 20 recorded unpooled seed-worlds. The split-conformal switch does not change either (it
  stays wrongly on in the no-stall world of seeds 10 and 13). The rank test changes once, seed 11 no-stall, off to on
  (p 0.16 to 0.017): with 41,818 rest rows it calls a shift of AUC 0.58 significant, a wrong call it did not make before.
- **BCA's heights shrink.** Where version 8's height was finite and the switch on (seeds 11 and 14, stalling world)
  the pooled height is 0.73 and 0.49 of the recorded one (2.289 to 1.671; 4.595 to 2.256). Where it was infinite the
  pooled heights are 2.775 (seed 10), 1.851 (seed 12) and 1.845 (seed 13). The stalling-world ramp heights on seeds
  10-14 are now 1.67-2.78 (step 1's seeds 0-4, unpooled: 1.61-6.02). The height is still an order statistic: the
  largest pooled tail score with 35-38 rows, the second with 46-58, the third with 63-64, the fourth with 76, the
  sixth with 109, minus the rest's threshold, which moves from about -0.02/-0.03 to about -0.04 with 42,000 rest rows.
- **The check passed exactly.** In the pinned environment version 8's recorded unpooled certificate, groups and rank
  test were recomputed from the regenerated data and the recorded second critic on all 10 fresh seed-worlds with
  zero difference in every number (the rank-test p-value included); on all 20 seed-worlds every array of the
  regenerated log equals the data record's hash, the removal's kept-row and bank hashes equal the record's, the gap
  rule's ends, both group sizes, the recorded finite share of draws and the certified flag are reproduced exactly.
- **What this preview cannot give: the pooled heights and switch calls of seeds 0-4.** Step 1's second critics for
  seeds 0-4 are on no branch and not on the desktop: step 1 ran on another machine (`machine: "vm"` in its results)
  and its critics folder `runs/shift_explore/restructure/step1/critics` does not exist in the study clone (the only
  `critics` folder there is the gate's). The earlier Windows pass said they were on the desktop; that was wrong.
  Their pooled group sizes, rest sizes and refusal decisions are exact without a critic (section 5). DECISION.md's
  choice 11 (retrain them, default no) is the only way to get those heights.

## 1. What was asked

For the spent seeds 0-4 and 10-14, main setting (the densest-15% ball removed), both worlds: the size of BCA's
full-throttle group and of its rest under version 9's pooling rule; BCA's threshold `R_tail` and ramp height under
pooling, with BCA's own calibrate code, 1,000 draws and the same rng tags; the decisions of BCA's switch, the
split-conformal switch, the rank-test switch and the mean-difference switch under pooling; the split-conformal height
under pooling; the comparison with version 8's recorded unpooled values; and, first, a recomputation of version 8's
recorded unpooled values as the check that the pipeline is right, exactly in the pinned environment.

## 2. The two rules

- **Version 8 (recorded)** (`near80/code/learner_phases.py`, `switch_scores`): full-throttle group = the gap rule's
  survivors (kept training rows with throttle index `t > t_lo`) plus the calibration-bank rows outside the ball with
  `t > t_lo`; rest = the bank rows outside the ball with `t <= t_lo`. The bank is 1,000 rows (5 per calibration
  episode), 827-881 of them outside the ball.
- **Version 9 (the refusal fix)**: full-throttle group = the survivors plus every calibration-split row (20,000) and
  every evaluation-split row (30,000) outside the removal ball with `t > t_lo`; rest = every such row with `t <= t_lo`.
  `t_lo` is the gap rule's, read from the kept training rows' commands exactly as before. The training rows, the
  second critic, the certificate (`step1lib.certificate` -> `calibration/wbcp.calibrate`, alpha 0.1, credibility 0.95,
  1,000 draws, margin 0.10, tags `[20261015, 9700, seed, world tag] + [1 | 2]`), the rank test and the arm rules
  (`step2lib.arm_table`: conformal margin 0.10, rank p 0.05; `policy_phase.arm_row_from`: mean-difference margin
  0.10) are unchanged. Rows inside the ball are left out of both groups, as the bank's in-ball rows were in version
  8 (`shift.make_removal`: `bank_removed = in_ball[bank]`). The held-out rows inside the ball are many (6,981-8,447
  per seed-world) and are the removed commands themselves; they stay out. This is the same definition as
  `near80lib.pooled_groups` in the version 9 draft on `claude/near80-v9` (survivors, then the held-out calibration and
  evaluation rows outside the ball, split at `t_lo`).

Scores are step 1's one-sided overestimates of the second critic (trained on the kept training rows minus the
survivors) with the stall-aware controller at the next state and gamma 0.98 (`learner_phases.overestimates`,
restated line for line in `preview_compute_pinned.py`). The mean-difference switch "turning on the hand-set range
limit" is the same switch as the mean-difference switch with the ramp (one call per seed-world, a different arm
behind it), so it has no separate number here.

## 3. Environment and inputs

- **The pinned environment, through the WSL watcher.** The job ran as `env -i ... PYTHONPATH=/home/user/bca
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/v9preview/pycache JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
  /tmp/claude-0/bcavenv/bin/python -I /tmp/v9preview/preview_compute_pinned.py 0 1 2 3 4 10 11 12 13 14`
  (`run_near80.sh`'s flags), 194 s for the 20 seed-worlds, output under `/tmp/v9preview` only. The study clone was
  read and not written: `find /home/user/bca -newer <start marker> -type f` found nothing afterwards and `git status`
  was unchanged. (`preview.json` -> `environment`.)
- **The study clone** `/home/user/bca`, branch `claude/main-bca-connection-x4o46s`, commit `67209d2` (an ancestor of
  `52914e4`; the two commits between them add results files only, and the near-80 code, `experiments`,
  `calibration` and the restructure folders are identical). Code read, with SHA-256 checked in the script:
  `near80_constants.py` `9a94e675…` (equal to the file committed at `52914e4`), `learner_phases.py` `e99f8aca…`,
  `step1lib.py` `727aba0f…` (executed from bytes checked against step 1's `code_manifest.json`, as
  `near80lib.load_step1lib` does), `calibration/wbcp.py` `fdd36b42…`, `step2lib.py` `8a2ea838…`,
  `experiments/shift/learners.py` `82b07382…`, `dataset.py` `3e0193bd…`; step 1's `results/results.json` `e513fd19…`
  gives the recorded unpooled values of seeds 0-4. (`preview.json` -> `inputs`.)
- **Version 8's run files**: the study clone's own `runs/shift_explore/near80` (the originals the branch
  `origin/claude/near80-run-files`, commit `7f0fe12`, was pushed from). Read: `data/main/<world>/seed<k>/data.json`
  (all ten seeds), and on seeds 10-14 `learners/main/<world>/seed<k>/critic_second.json` + `critic_second.msgpack`
  and `switch.json`. Every record's `.sha256` sidecar was checked, every `critic_second.msgpack` was checked against
  its record's `file_sha256` and its parameter SHA-256 recomputed (equal on all 10), and all 40 files read were
  checked against the branch's `RUNFILES_SHA256SUMS`: 40 equal, 0 unequal (`summary.branch_copy_checks`;
  `seed_worlds[*].sources.*.equals_branch_copy`).
- **Regeneration**: each log with `dataset.generate(seed, world, the gate's recorder, (500, 200, 300), slope 1.5)`.
  In the pinned environment **every** array of every log, `infos/state` and `infos/next_state` included, hashes equal
  to the record's `log_sha256` on all 20 seed-worlds (`seed_worlds[*].checks.log_all_arrays`; the Windows pass had
  last-bit differences in the float64 state arrays on all 20 and in `actions`/`observations` on seed 11, section 7).
  The removal's kept rows and bank rows hash equal to the record's on all 20; on seeds 10-14 `learners.prepare`'s
  rows, train, bank and normalization hashes equal the record's `prepared_sha256` (20 of 20 hashes). Seeds 0-4 have no
  such record (the gate's loader checks the gate's data there).

## 4. The check: version 8's recorded unpooled values recomputed

| seed | world | t_lo, t_up equal | n_tail, n_rest equal | finite share of draws equal | certified flag equal | certificate, groups, rank test recomputed from the recorded critic |
|---|---|---|---|---|---|---|
| 0-4 | both | yes (10 of 10) | yes (10 of 10) | yes (10 of 10, to the third decimal as recorded) | yes (10 of 10) | not possible: step 1's second critics are on no machine reachable (section 0) |
| 10-14 | both | yes (10 of 10) | yes (10 of 10) | yes | yes | every number equal, zero difference: R_tail, R_rest, height, q05, lambda_hat, both means, both 90% quantiles, rank U and p; the same four switch calls (`seed_worlds[*].v8.recomputed_match`) |

The recorded unpooled values of seeds 0-4 (step 1's results, main setting): stalling world on with heights 6.019, 2.420,
2.065, 1.605, 1.796 (R_tail 5.986, 2.394, 2.039, 1.578, 1.776; n_tail 35, 47, 59, 43, 47); no-stall world off with q05
0.021, 0.002, 0.010, 0.020, -0.047 (n_tail 38, 36, 52, 29, 60; seed 3's tail threshold is `+inf` at 29 rows, with the
switch off it did not matter). On the 20 recorded unpooled seed-worlds (`recorded_calls.txt`, from the recorded
entries) BCA's switch was right in 20 of 20 (on in the stalling world, off in the no-stall world), the mean-difference
switch agreed with it in 20 of 20, the split-conformal switch in 18 (wrongly on: no-stall seeds 10 and 13), the rank
test in 18 (wrongly on: no-stall seeds 2 and 14).

## 5. Refusal depends on the group size and the draws, never on the scores

In `wbcp.calibrate`, a draw's crossing is `+inf` when its test atom alone exceeds alpha of the mass
(`E_{n+1} > alpha/(1-alpha) * (sum of n exponentials)`, probability exactly `0.9^n`), and the deployed threshold is the
950th smallest of the 1,000 crossings. So the tail threshold is `+inf` exactly when more than 50 draws are infinite,
which depends only on `n_tail` and the rng stream (tags `[20261015, 9700, seed, world tag, 1]`), not on any score.
Running BCA's own calibrate on placeholder scores `0..n-1` with each seed-world's tags therefore gives the refusal
decision and the finite share of draws exactly; on every recorded seed-world the finite share so rebuilt equals the
recorded `finite_share_tail` and the certified flag equals the recorded `tail_certified` (20 of 20). The exact refusal
probability by group size, P(Binomial(1000, 0.9^n) > 50) (`summary.refusal_probability_by_group_size`):

| n_tail | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P(one draw infinite) = 0.9^n | 0.072 | 0.065 | 0.058 | 0.052 | 0.047 | 0.042 | 0.038 | 0.034 | 0.031 | 0.028 | 0.025 | 0.023 | 0.020 | 0.015 |
| P(BCA refuses) | 0.997 | 0.969 | 0.85 | 0.60 | 0.30 | 0.10 | 0.025 | 0.0040 | 4.6e-4 | 3.9e-5 | 2.5e-6 | 1.2e-7 | 4.6e-9 | 8.0e-14 |

(At 24 rows and below the refusal is certain to four decimals; the full table 15-60 is in `preview_tables.md`.) When
the threshold is finite it is an order statistic of the tail scores; the draws alone say which one, and the scores
confirm it on the 10 seed-worlds that have them (`seed_worlds[*].v9.R_tail_rank_from_top`, 0 = the largest): the
largest with 35-38 rows (seeds 10 both worlds, 13 stalling), the second with 46-58 (12 both, 14 stalling), the third
with 63-64 (11 and 14 no-stall), the fourth with 76 (13 no-stall), the sixth with 109 (11 stalling); the plain 90%
quantile `lambda_hat` (the split-conformal switch's tail point) is the 4th to 11th from the top. BCA's height is that
order statistic minus the rest's threshold.

## 6. Results

### 6.1 Group sizes (outcome-free; exact on all 20 seed-worlds)

`v8 n_tail` = survivors + bank full-throttle rows; `v9 n_tail` = survivors + calibration-split + evaluation-split
full-throttle rows outside the ball (the bank rows are part of the calibration split). "draws finite" is the refusal
decision from the draws alone (section 5); on seeds 10-14 the scores confirm it. (`seed_worlds[*].v8`, `.v9`,
`.v9.refusal_from_draws`.)

| seed | world | t_lo | v8 n_tail (surv+bank) | v8 n_rest | v9 n_tail (surv+cal+eval) | v9 n_rest | v8 draws finite | v9 draws finite |
|---|---|---|---|---|---|---|---|---|
| 0 | stall | 0.7860 | 35 (34+1) | 850 | 70 (34+18+18) | 42089 | yes | yes |
| 0 | no_stall | 0.7863 | 38 (37+1) | 849 | 82 (37+18+27) | 42124 | yes | yes |
| 1 | stall | 0.7868 | 47 (45+2) | 846 | 106 (45+30+31) | 42027 | yes | yes |
| 1 | no_stall | 0.7843 | 36 (34+2) | 843 | 84 (34+26+24) | 42026 | yes | yes |
| 2 | stall | 0.7914 | 59 (59+0) | 839 | 121 (59+24+38) | 42447 | yes | yes |
| 2 | no_stall | 0.7829 | 52 (52+0) | 838 | 105 (52+21+32) | 42469 | yes | yes |
| 3 | stall | 0.7910 | 43 (42+1) | 850 | 100 (42+22+36) | 42176 | yes | yes |
| 3 | no_stall | 0.7840 | 29 (28+1) | 849 | 66 (28+15+23) | 42210 | no (+inf) | yes |
| 4 | stall | 0.7871 | 47 (45+2) | 825 | 110 (45+30+35) | 41505 | yes | yes |
| 4 | no_stall | 0.7854 | 60 (58+2) | 827 | 135 (58+27+50) | 41476 | yes | yes |
| 10 | stall | 0.7910 | 19 (18+1) | 876 | 35 (18+6+11) | 42997 | no (+inf) | yes |
| 10 | no_stall | 0.7893 | 18 (15+3) | 878 | 37 (15+9+13) | 42997 | no (+inf) | yes |
| 11 | stall | 0.7863 | 53 (51+2) | 836 | 109 (51+23+35) | 41788 | yes | yes |
| 11 | no_stall | 0.7845 | 32 (30+2) | 836 | 63 (30+18+15) | 41818 | yes | yes |
| 12 | stall | 0.7894 | 22 (21+1) | 872 | 46 (21+10+15) | 42547 | no (+inf) | yes |
| 12 | no_stall | 0.7894 | 21 (20+1) | 873 | 47 (20+11+16) | 42565 | no (+inf) | yes |
| 13 | stall | 0.7890 | 18 (18+0) | 857 | 38 (18+5+15) | 42762 | no (+inf) | yes |
| 13 | no_stall | 0.7873 | 36 (36+0) | 853 | 76 (36+11+29) | 42734 | yes | yes |
| 14 | stall | 0.7867 | 30 (30+0) | 850 | 58 (30+13+15) | 41899 | yes | yes |
| 14 | no_stall | 0.7823 | 31 (31+0) | 851 | 64 (31+13+20) | 41922 | yes | yes |

The pooled group is 1.8 to 2.3 times version 8's: the ball's leak is the same small share of every split (seed 10
stalling: 18 of 50,000 training rows, 6 of 20,000 calibration rows, 11 of 30,000 evaluation rows). Seeds 10, 12 and
13 leak least, which is why version 8 met the refusal on those three fresh seeds and on no exploratory seed. For a
fresh seed of this generator the pooled group should be expected in the 35-135 range seen here; 35 is the smallest
seen and already past the certain-finite line (33-36).

### 6.2 BCA's threshold and height (seeds 10-14: recorded second critic; seeds 0-4: no critic available)

`rank` = which tail score R_tail is, from the top (1 = the largest), version 8 -> version 9.
(`seed_worlds[*].v8.{R_tail,R_rest,height}` recorded; `.v9.{R_tail,R_rest,height,q05,R_tail_rank_from_top}`.)

| seed | world | v8 BCA | v8 R_tail | v8 R_rest | v8 height | v9 BCA | v9 R_tail | v9 R_rest | v9 height | v9 q05 | height v9/v8 | rank v8 -> v9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | stall | on | +inf | -0.022 | +inf | on | 2.736 | -0.040 | 2.775 | 1.247 | inf -> finite | - -> 1 of 35 |
| 10 | no_stall | off | +inf | -0.034 | +inf | off | 0.177 | -0.042 | 0.219 | 0.017 | inf -> finite | - -> 1 of 37 |
| 11 | stall | on | 2.262 | -0.027 | 2.289 | on | 1.630 | -0.041 | 1.671 | 0.908 | 0.73 | 2 of 53 -> 6 of 109 |
| 11 | no_stall | off | 0.107 | -0.027 | 0.134 | off | 0.086 | -0.036 | 0.122 | 0.014 | 0.91 | 1 of 32 -> 3 of 63 |
| 12 | stall | on | +inf | -0.034 | +inf | on | 1.813 | -0.038 | 1.851 | 1.165 | inf -> finite | - -> 2 of 46 |
| 12 | no_stall | off | +inf | -0.030 | +inf | off | 0.041 | -0.037 | 0.079 | -0.001 | inf -> finite | - -> 2 of 47 |
| 13 | stall | on | +inf | -0.025 | +inf | on | 1.810 | -0.036 | 1.845 | 0.646 | inf -> finite | - -> 1 of 38 |
| 13 | no_stall | off | 0.223 | -0.018 | 0.241 | off | 0.131 | -0.035 | 0.165 | 0.062 | 0.69 | 1 of 36 -> 4 of 76 |
| 14 | stall | on | 4.564 | -0.031 | 4.595 | on | 2.220 | -0.036 | 2.256 | 0.906 | 0.49 | 1 of 30 -> 2 of 58 |
| 14 | no_stall | off | 0.752 | -0.023 | 0.775 | off | 0.159 | -0.033 | 0.192 | 0.011 | 0.25 | 1 of 31 -> 3 of 64 |

Seeds 0-4 (both worlds): no second critic is available (section 0), so `R_tail`, the height and the four switch calls
under pooling were not computed; their recorded unpooled values are in section 4 and `preview.json`.

Why the heights shrink: the pooled tail threshold is a lower order statistic of a larger group and the pooled rest
threshold, now the 950th-smallest-draw crossing of about 42,000 scores, sits a little lower (-0.033 to -0.042
against -0.018 to -0.034). Both move the height down; in the stalling world the switch's statistic `q05` stays far
above the margin (0.65-1.25), in the no-stall world it stays near zero (-0.001 to 0.062).

### 6.3 Decisions under pooling (seeds 10-14)

(`seed_worlds[*].v8.decisions` recorded, `.v9.decisions`; the split-conformal height is its gap when on,
`.v9.split_conformal_height`.)

| seed | world | BCA v8 -> v9 | split-conformal v8 -> v9 (gap = its height) | rank test v8 -> v9 (p) | mean-difference v8 -> v9 (gap) | arm table refuses v8 -> v9 |
|---|---|---|---|---|---|---|
| 10 | stall | on -> on | on -> on (2.708 -> 2.480) | on -> on (5.6e-06 -> 7.2e-15) | on -> on (0.706 -> 0.779) | obca_k1 -> none |
| 10 | no_stall | off -> off | on -> on (0.194 -> 0.121) | off -> off (0.16 -> 0.14) | off -> off (0.053 -> 0.028) | none -> none |
| 11 | stall | on -> on | on -> on (0.892 -> 1.200) | on -> on (4.2e-15 -> 3.6e-36) | on -> on (0.466 -> 0.507) | none -> none |
| 11 | no_stall | off -> off | off -> off (0.055 -> 0.065) | **off -> on** (0.16 -> 0.017) | off -> off (0.024 -> 0.034) | none -> none |
| 12 | stall | on -> on | on -> on (1.290 -> 1.639) | on -> on (9.5e-10 -> 2.4e-20) | on -> on (0.589 -> 0.645) | obca_k1 -> none |
| 12 | no_stall | off -> off | off -> off (0.014 -> 0.008) | off -> off (0.40 -> 0.35) | off -> off (-0.005 -> 0.008) | none -> none |
| 13 | stall | on -> on | on -> on (1.520 -> 1.408) | on -> on (5.7e-07 -> 1.0e-10) | on -> on (0.583 -> 0.396) | obca_k1 -> none |
| 13 | no_stall | off -> off | on -> on (0.112 -> 0.116) | off -> off (0.058 -> 0.19) | off -> off (0.032 -> 0.012) | none -> none |
| 14 | stall | on -> on | on -> on (0.881 -> 0.987) | on -> on (2.8e-11 -> 3.3e-22) | on -> on (0.615 -> 0.594) | none -> none |
| 14 | no_stall | off -> off | off -> off (0.038 -> 0.044) | on -> on (0.032 -> 0.0017) | off -> off (0.066 -> 0.054) | none -> none |

- BCA's switch: 10 of 10 calls unchanged and right. The mean-difference switch (margin 0.10): 10 of 10 unchanged and
  equal to BCA's; its heights 0.40-0.78 in the stalling world.
- The split-conformal switch (90% quantile gap > 0.10): 10 of 10 unchanged; it remains on in the no-stall world of
  seeds 10 and 13 (gaps 0.121 and 0.116), two wrong calls pooling does not cure. Its pooled heights in the stalling
  world: 2.480, 1.200, 1.639, 1.408, 0.987 (version 8: 2.708, 0.892, 1.290, 1.520, 0.881); in the no-stall world
  where on: 0.121 (seed 10), 0.116 (seed 13). Against BCA's pooled heights (2.775, 1.671, 1.851, 1.845, 2.256) the
  plain gap is below BCA's on all five stalling seeds and twice it (4.96, 2.40, 3.28, 2.82, 1.97) is above BCA's on
  seeds 10-13 and below on seed 14.
- The rank test (p < 0.05): one change, seed 11 no-stall, off to on. With 41,818 rest rows the one-sided
  Mann-Whitney test has the power to call a tiny shift (AUC 0.58) significant; a switch that turns the ramp on in the
  no-stall world is a wrong call. Seed 14 no-stall was already (wrongly) on in version 8 and stays on (p 0.0017). This
  is the one decision pooling changes, and it changes for the worse; the rank-test switch is a reported-only arm.
- The arm table: version 8's three refusals (obca_k1 on with an infinite height on seeds 10, 12 and 13, stalling
  world) are gone; no arm is refused on any of the 10 seed-worlds.

### 6.4 Aggregates (the task's three questions)

- **How often BCA would still refuse:** 0 of 20 seed-worlds (exact, from the draws; the scores agree on the 10 that
  have them). Version 8: tail threshold `+inf` on 6 of 20 (seeds 3 no-stall, 10 both, 12 both, 13 stalling), of which
  3 stopped the actor phase (10, 12, 13 stalling; the no-stall ones and seed 3 had the switch off, so arm_table did
  not refuse). The smallest pooled group is 35 rows (seed 10 stalling; refusal probability 2.5e-6); the next are 37,
  38, 46, 47, 58.
- **How much its heights shrink:** stalling world, switch on, finite before and after: seed 11 2.289 -> 1.671 (x0.73),
  seed 14 4.595 -> 2.256 (x0.49); infinite before: seed 10 -> 2.775, seed 12 -> 1.851, seed 13 -> 1.845. No-stall world
  (switch off, height unused): 0.134 -> 0.122, 0.241 -> 0.165, 0.775 -> 0.192, and +inf -> 0.219, 0.079. The pooled
  stalling-world heights span 1.67-2.78 against step 1's 1.61-6.02 on seeds 0-4 and version 8's 2.29 and 4.60 on the
  two fresh seeds that had one. Twice BCA's height (`obca_k2`) would be 3.3-5.6.
- **Whether any decision changes:** none of BCA's, none of the mean-difference switch's, none of the split-conformal
  switch's; one of the rank test's (seed 11 no-stall, off to on, wrong). Seeds 0-4 could not be read (section 0); on
  the unpooled records all four switches agreed with BCA there except the rank test on no-stall seed 2 (p 0.032).

## 7. The cross-check: the Windows pass against the pinned run

The first pass (`preview_compute.py`, `preview_raw_all.json`, `scores/`) ran the same computation in a Windows venv
of the project's pins (Python 3.10.20, numpy 2.2.6, scipy 1.15.3, jax 0.6.2, flax 0.10.7). `compare_runs.py`
compared it with the pinned run seed-world by seed-world (`compare_windows_vs_pinned.json`;
`preview.json` -> `cross_check_windows_run`):

- counts, `t_lo`/`t_up` and the draws-only refusal results: equal on 20 of 20;
- the score arrays (`tail8`, `rest8`, `tail9`, `rest9`) and the row indices of the pooled groups: bit-identical on
  10 of 10 scored seed-worlds;
- every certificate and group number (R_tail, R_rest, height, q05, q50, lambda_hat, finite shares, means, 90%
  quantiles, rank U): identical on 10 of 10; the only differences are two Mann-Whitney p-values in the far tail
  (3.2e-27 on the unpooled seed 14 stalling, 3.0e-36 on the pooled seed 12 stalling), where the platform's normal
  tail differs; no decision depends on them;
- the regenerated logs: on Windows the float64 `infos/state` and `infos/next_state` differed from the record in the
  last bit on all 20 seed-worlds, and on seed 11 `actions`, `observations` and `next_observations` differed too; in
  the pinned environment every array equals the record on all 20. Since the pooled scores of seed 11 are identical
  between the two runs, the Windows divergence lay among rows that enter neither group (the ball-removed training rows
  or rows inside the ball); the pinned run settles it either way.

So the Windows pass was right in every number it reported; its two statements that needed the pinned environment
(seed 11's pooled numbers, and the exactness of the rank-test p on seed 14) are now settled, and its claim that
step 1's seeds 0-4 critics were "on the desktop" is withdrawn (section 0).

## 8. What this preview is and is not

1. **Exploratory, on spent seeds.** Every seed here has been seen; the numbers license the rule (pooling ends the
   refusal without changing a call of BCA's or the mean-difference switch) and say what to expect; they are no claim.
   The version 9 run retrains BCA's second critic on fresh seeds, so the pooled heights of a fresh seed are new
   draws; what carries over is the mechanism (an order statistic of a 35-135-row group minus a rest threshold of about
   -0.04) and the group sizes' range.
2. **Seeds 0-4's pooled heights and calls are not available anywhere reachable.** Step 1's critics never left the
   machine step 1 ran on (`machine: "vm"`); the desktop has no `runs/shift_explore/restructure/step1` folder and no
   branch holds the files. The counts, rest sizes and refusal decisions of seeds 0-4 are exact without them (section
   5). A retrain (DECISION.md choice 11, default no) would be a preview substitute, not step 1's files; the earlier
   Windows attempt at one (`retrain_critic.py`, seed 10 stalling) reproduced the recorded survivors and training rows
   but was stopped before the 100,000 updates finished, and no critic file was written.
3. **The theory of section 5** (refusal probability by group size) is exact given numpy's `standard_exponential`
   stream; the stream was verified against every recorded finite share (20 of 20), so it is not an open point.
4. **Nothing here is a frozen fact.** DECISION.md section 8.1 (check 5) asks the frozen code to recompute the pooled
   switch on seeds 10-14 from version 8's recorded second critics before approval; the numbers it should reproduce are
   sections 6.2 and 6.3 (BCA on 5 of 5 stalling, off 5 of 5 no-stall; the mean-difference switch the same).

## 9. Files

- `preview.json`: every number of this preview with its source (environment, input files and SHA-256s, rules, per
  seed-world checks, record paths and hashes, counts, recorded values, recomputed values, pooled values, the sorted
  pooled tail scores, decisions, refusal probabilities, the cross-check summary). Non-finite numbers are the strings
  `"inf"`, `"-inf"`, `"nan"`.
- `preview_raw_pinned.json`: the raw per seed-world output of the pinned run (what `preview.json` is built from);
  `preview_pinned_all.log`: its console output.
- `scores_pinned/scores_seed<k>_<world>.npz` (seeds 10-14): the tail and rest score arrays, unpooled and pooled,
  with the log row indices of the pooled groups, the survivors and the kept bank rows, from the pinned run.
- `preview_compute_pinned.py`: the computation (Linux paths; the copy that ran is at `/tmp/v9preview` in the WSL
  distro); `preview_write_pinned.py`: the consolidation; `compare_runs.py` and `compare_windows_vs_pinned.json`: the
  cross-check; `recorded_calls.py` / `recorded_calls.txt`: the four switches' calls on the 20 recorded unpooled
  entries; `preview_tables.md`: the tables of section 6 as generated, plus the full refusal table (n = 15-60).
- The Windows pass, kept: `preview_compute.py`, `preview_write.py`, `preview_raw_all.json` (= `preview_raw.json`),
  `scores/`, `preview_all.log`, `preview_seed10.log`, `smoke_regen.py`, `retrain_critic.py`,
  `retrain_seed10_stall.log`.
- Not here, read-only: the run-files worktree `scratchpad\runfiles\` (commit `7f0fe12`, whose `RUNFILES_SHA256SUMS`
  the pinned run checked the study clone's files against), the study clone `/home/user/bca`, the pinned venv
  `/tmp/claude-0/bcavenv`.
