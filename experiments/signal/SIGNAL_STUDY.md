# Signal study, steps 1-3: does BCA measure the critic error that matters to the actor, and apply it usefully?

**Question** (the user's plan, 2026-10-01): before touching ensembles or penalty placement, establish
- whether BCA's calibration signal measures the error the actor actually uses, and
- whether the current BC-multiplier hook turns that measurement into a better actor update.

The actor stayed fixed throughout: TD3+BC's objective, which climbs Q1, with BCA's current BC-multiplier hook.

**Pre-registration.** `runs/wbcp_signal/expectations.md`, written before any step 1-3 run.
- The original text was written 2026-10-01T20:51:26Z (SHA-256 `ec302bec…`; the text before Addendum A still
  hashes to this).
- Addendum A added a sixth LQ case and dose outputs for step 2.
- Addendum B relabelled item A1.3.
- The final file's SHA-256 is `53c13f11…`.

The advisor page mirrors it: https://claude.ai/artifact/YVdVpznLwXY6P9Xuc9YJHn. Every expectation is quoted below
with its pre-registered label:
- **[derived]**: a miss means a bug;
- **[replication]**: an earlier measurement repeated under the new design;
- **[forecast]**: a judgment; a miss is a finding.

**How verdicts were made.**
- The step 2 items and step 3 items checks, E1, E2, E3, E5, E6 and A1.2 were scored mechanically by
  `experiments/signal/scorecard.py`. For E3, E5 and E6 the figures quoted in the text (48 of 106; 121/60/34/1;
  E6's second clause) are hand computations over all cells. `scorecard.md`'s own E3 count (17 of 108) uses only
  the largest κ and includes cells where both miss rates are zero. All of them give the same verdicts. It was last modified at 22:35:36Z, before every evaluation file (22:37:14Z
  on) and before the LQ results (23:08Z); its SHA-256 is in `code_provenance.txt`.
  - A range counts as met when the 95% interval of the measured rate overlaps it.
  - "Within two standard errors" uses the paired difference's own standard error.
- Every other verdict (steps 1a-1c, E4, E7, A1.1, A1.3, A1.4), every qualifier such as "partly", and every
  decision-rule reading is a hand verdict, labelled as such.
- Healthy pools are hopper, walker2d, halfcheetah, maze2d and pen-expert. pen-cloned and pen-human are reported
  but excluded from verdicts (critic not converged; DEPENDENCE.md, Change 12).
- Explanations of misses are my readings unless they say they were tested.

**Code.** The base is main at `fdba570`. The uncommitted changes are snapshotted in `runs/wbcp_signal/`, with
SHA-256 values in `code_provenance.txt`.
- Snapshot 1 ran step 2.
- Snapshot 2 ran step 3. It differs only by `jax.clear_caches()` after each LQ replicate. That changes no computed
  value. On a reduced grid run twice, once with the clear and once with it disabled, replicate 1 is computed after
  a clear in the first run and without one in the second. All 2,565 of its values are bit-identical
  (`clear_caches_check.txt`; result files in `evidence_runs/`).
- 266 tests pass, each module run in its own process (`tests_per_module.log`). In one process, JAX's compiled code
  reaches the kernel's memory-map limit (vm.max_map_count, 65,530). That limit also crashed the first LQ run
  (`lq_failed_max_map_count.log`), which is what led to the cache clear.

## Summary

1. **The baseline is locked down, with one exception.**
   - Both samplers now trim the remainder at random. Banks are exactly n rows, with flat inclusion, unless reserved
     episodes are shorter than K. Then a bank is smaller: walker2d K = 23 banks hold 1,003-1,024 rows.
   - `dependence_validated` is replaced by measured evidence.
   - All 26 pools verify. Each dataset's withheld population is identical across the hosts that have a pool for it
     (four, or three for pen-expert and pen-human).
   - The configuration fingerprint is unchanged (`1dd66a57…`).
2. **BCA's current signal does not measure the error the actor uses.**
   - On real D4RL critics, a band calibrated for min(Q1, Q2) fails for Q1's residual in 10-82% of banks on all
     five healthy pools.
   - A band calibrated for Q1 is valid for Q1 (4.0-5.1% of banks fail) and still covers the min residual (at most
     2.9%), at 2-10% more width.
3. **The current hook passes little of that measurement to the actor.**
   - In the linear-quadratic task, BCA's dose tracks the true Q1 error. The correlation averages 0.4-0.8 per case
     for the q1 signal; cells range from 0.14 to 0.92.
   - But the dose stays below 1.5 and varies by only about 0.02 from row to row.
   - Against a constant dose at the same mean, BCA's change in true value differs by a median 6% of its effect
     against no BCA. BCA is worse than both the constant and shuffled controls by more than two paired standard
     errors in 121 of 216 cells, better in 60 and within in 34.
   - A dose built from the true error, at the same mean, moved J much more in one case (independent_errors:
     +0.14 with good data, −1.01 with poor, on average). **Post hoc, that is mostly strength, not targeting.** At
     the same mean dose, its BC gradient was 2.3-3.1× the constant dose's (good data). A strength-only model also
     explains BCA's own departures from the constant dose (correlation 0.85 over 216 cells). Matching the mean dose
     did not match strength, so step 3 does not show that targeting moves J (see "Post hoc: same mean, different
     strength").
4. **One limit holds for any residual signal.** A critic bias that both heads share and that satisfies the
   Bellman equation is invisible. Residual coverage stays near 91% in every setting. The true error is missed on
   a share that grows with κ and the offset: 0-1% at the mildest settings, 45-100% once the bias is large (for
   example 86.6% at κ = 2 and offset 0.5 with good data; 100% at κ ≥ 1 and offset 2).
5. **Decision rules: none applies exactly as written. Here is how each was read.**
   - Q1 becomes the measurement signal for step 4. This is a judgment: rule 1's precondition was only partly met.
   - maxabs is dropped. This is a judgment: it is never meaningfully better, which is not the rule's "never better".
   - Rules 2 and 3 are each partly triggered. Step 4 therefore does both:
     - it compares where the signal is applied, using a dose with real dynamic range, against constant-dose and
       shuffled controls, crossed with good and poor behaviour data;
     - it tests whether any of BCA's wins over the controls survive on D4RL. BCA beats both controls in 60 LQ
       cells, but 15 of them have no Q1 error to target, so whether any of the wins is targeting is open.

## Step 1a. Sampler remainder fix

**What changed.**
- `d4rl_benchmark.draw_calibration` (random and spaced per-episode designs) cut ⌈n/K⌉·K drawn rows to n by
  dropping the last episode's final segments. It now keeps a uniformly random n of them.
- `calibration/bank.stratified_bank` (BCA's own sampler) never trimmed. It now removes any surplus uniformly at
  random. Every reserved episode still leaves training.
- Neither draws any extra randomness when K divides n.

**Why.** Every later comparison draws its banks with these samplers. A failure rate is unbiased only if every
pool row has the same chance of entering a bank.
- Before the fix, late timesteps were under-sampled: late/early inclusion was 0.979 at n = 1,024, K = 23.
- BCA's banks were off-size: 1,025-1,035 rows at n = 1,024 for K ≤ 23, and as few as 1,002 on walker2d K = 23,
  where short episodes contribute fewer rows. walker2d's K = 67 banks held 7,888-8,241 rows at n = 8,192.

**Why it makes sense.** A uniformly random subset of the drawn rows keeps every drawn row's expected count equal.

**Expected.**
1. Designs where K divides n reproduce bit for bit. **[derived]**
2. Every bank has exactly n rows, and inclusion is flat by position. **[derived]**
3. Rerunning any design already reported changes its failure rate by less than 0.7 points, and no verdict in
   DEPENDENCE.md flips. **[forecast]**

**Result.**
- **Item 1.**
  - halfcheetah K = 2 at n = 1,024 and pen-human K = 124 at n = 248 reproduce every failure count exactly.
  - Independent-row and whole-episode designs never reach the trim, so they are unchanged by construction.
  - Unit tests check bit-identity when K divides n.
- **Item 2.**
  - On a synthetic pool (n = 1,024, K = 23), the trimmed samplers give exactly 1,024 rows. Late/early inclusion
    is 0.9998 for the benchmark sampler and 0.9997 for BCA's.
  - On real data, BCA's sampler still gives fewer than n rows when reserved episodes are shorter than K. Those
    contribute all their rows, as the docstring states. walker2d K = 23 banks hold 1,003-1,024 rows (mean
    1,023.8) after the fix.
- **Item 3.** 14 Change 12 designs were rerun at their original settings and seeds (`runs/wbcp_signal/step1_reruns`).
  - Every arm except W-CRC (uniform BCA, both WBCP arms, RCPS) moves by at most 0.53 points in every rerun.
  - The W-CRC baseline moves more: +1.78 points (pen-human K = 5, normalized) and +1.02 (pen-expert K = 10, raw).
  - Surplus designs not rerun, among others:
    - walker2d K = 23, which has the largest surplus;
    - pen-cloned K = 5;
    - pen-human K = 10 at n = 248;
    - the three K = 5 shift designs;
    - CQL hopper K = 6;
    - TD3+BC hopper K = 5 at n = 1,103;
    - every 8,192-row bank.
  - No DEPENDENCE.md verdict flips.
  - Under step 1b's evidence rule, three arm-level statuses cross the 5% line, all from "exceeds" to "consistent"
    and all by 2-6 banks in 4,000:
    - pen-expert K = 5, raw score, uniform BCA: 5.78% [5.07, 6.54] → 5.62% [4.93, 6.38];
    - pen-expert K = 5, normalized, WBCP with exact weights: 5.73% [5.03, …] → 5.67% [4.98, …];
    - maze2d K = 10, raw score, RCPS: 5.75% [5.05, …] → 5.65% [4.95, …].

    The registry holds uniform BCA with the normalized score, whose status does not change: pen-expert K = 5
    goes 5.45% → 5.42%, which stays "consistent".

**Met?** Partly (hand verdict).
- Item 1 holds.
- Item 2 holds only when every reserved episode is longer than K.
- Item 3 holds for the uniform BCA and WBCP arms on the designs rerun. It does not hold for W-CRC, and it is
  untested for the largest-surplus designs.

## Step 1b. Evidence-based dependence label

**What changed.** `dependence_validated` was true whenever K ≤ 10 and the bank spans at least 100 episodes, a
rule that rested on hopper's rho alone. It is replaced by a lookup into `calibration/dependence_evidence.json`,
generated by `experiments/wbcp/dependence_evidence.py` from the measured runs.
- The lookup is keyed by host, dataset, K, n, sampler and calibrated score.
- Its status is "consistent with the 5% budget" when the 95% interval's lower bound is at most 5%, "exceeds"
  otherwise, and "not validated" when no run matches.
- `train.py` writes it next to `resolved.json`. It is kept out of the resolved row, so regenerating the evidence
  never changes a run's identity: the configuration fingerprint stays `1dd66a57…`.

**Why.** The old label marked banks as validated that measured above budget, for example maze2d K = 5 at 6.5%.
Keying by host and score also makes explicit that validation carries over neither to another host nor to a new
signal, such as step 2's Q1.

**Expected.** The registry reproduces the pre-registered table exactly. **[derived]**

**Result: met** (hand verdict, mechanically checked).
- All 13 pre-registered rows match in value and status. Two examples:
  - maze2d K = 5 at 6.5% [5.7, 7.3]: "exceeds", where the old rule said validated;
  - CQL hopper K = 6 at 5.1% [4.5, 5.9]: "consistent".
- The registry holds two more entries beyond the table: TD3+BC hopper K = 5 at n = 1,103, and K = 18 at
  n = 8,192.
- `--check` confirms the file regenerates byte for byte.
- All entries predate the trim and use the min(Q1, Q2) score; each says so.

## Step 1c. Pool hash verification

**What changed.** A new tool, `experiments/wbcp/verify_pools.py`, recomputes for every pool:
- the score file and checkpoint hashes;
- the cached D4RL file's hash;
- the withheld split, re-derived from the dataset and split seed and compared row for row;
- a population hash for comparison across hosts.

**Why.** Step 2 reuses the TD3+BC checkpoints as frozen critics, and every comparison across hosts assumes the
same withheld episodes.

**Expected.**
1. Every score file, checkpoint and dataset hash matches. **[forecast]**
2. Every split reproduces exactly, and each dataset's population hash is identical across hosts. **[replication]**
3. Diverged pools pass, because divergence is a training outcome. **[derived]**

**Result: met** (hand verdict).
- 26 pools verified, 0 failed (`runs/wbcp_signal/verify_pools`).
- Each dataset's population hash is identical across all the hosts that have a pool for it. pen-expert and
  pen-human have no IQL pool yet.
- With re-preparation off, 22 of the 26 pools have 1-12 fields reported as "not re-derivable without
  re-preparation". The pre-registration anticipated such fields only in the oldest pools (hopper, September 30).
  They are a consequence of running without re-preparation, not a mismatch.
- The IQL pools are marked superseded: they predate the IQL target-head fix and will be retrained.

## Step 2. Three signals on frozen D4RL critics

**What changed.** For each of the seven TD3+BC pools, the actor, critics and target heads were frozen. Three
signals were compared:
- **min:** t − min(Q1, Q2), BCA today;
- **q1:** t − Q1;
- **maxabs:** max(|t − Q1|, |t − Q2|).

Each signal got a fresh σ(s, a) and residual unit u, fit with BCA's own objective for the parent's 100,000
updates, on the same batches with the same seeds and the same native target. Every update was accepted, for every
signal on every pool. Each signal was then calibrated with WBCP on the same 4,000 banks per design, and every
residual was scored under every band.

**Why.** This is the measurement half of the question, on real critics, with the signal definition as the only
difference.

**Result** (independent-row banks, n = 1,024; uniform-weight WBCP; % of banks failing):

| Pool | Own failure, min / q1 / maxabs | **Q1's residual under the min band** | Min's residual under the q1 band | Both residuals under maxabs | Width maxabs/q1, q1/min | Head gap ÷ residual | Dose mean ± row SD (q1) |
|---|---|---|---|---|---|---|---|
| hopper | 4.8 / 4.6 / 5.1 | **65.3** | 0.1 | 0.0 | 1.155, 1.075 | 0.93 | 1.296 ± 0.024 |
| walker2d | 4.0 / 4.3 / 4.9 | **82.4** | 0.0 | 0.0 | 1.154, 1.103 | 1.03 | 1.327 ± 0.029 |
| halfcheetah | 4.6 / 4.0 / 4.6 | **15.6** | 1.4 | 0.0 | 1.092, 1.027 | 0.52 | 1.298 ± 0.025 |
| maze2d | 4.3 / 4.4 / 4.4 | **27.4** | 2.1 | 0.0 | 1.190, 1.047 | 1.15 | 1.262 ± 0.050 |
| pen-expert | 5.1 / 5.1 / 4.8 | **10.5** | 2.9 | 0.0 | 1.118, 1.020 | 0.64 | 1.312 ± 0.017 |
| pen-cloned ⚠ | 4.2 / 4.1 / 4.4 | 41.5 | | | | | |
| pen-human ⚠ | 4.5 / 4.0 / 4.9 | 0.1 | | | | | |

Configured TD3+BC banks, own failure min / q1 / maxabs:
- hopper: 5.1 / 5.0 / 4.9
- walker2d: 15.6 / 15.1 / 14.3
- halfcheetah: 5.2 / 4.6 / 5.0
- maze2d: 7.0 / 6.6 / 7.0
- pen-expert: 5.8 / 5.0 / 6.2

**Met?** 6 of 8 items (scored mechanically).

| # | Expectation | Result | Met? |
|---|---|---|---|
| 1a | With independent rows, each band fails its own score in 3.5-6.5% of banks **[derived]** | 4.0-5.1% | yes |
| 1b | On the configured bank, own failure follows dependence: about 5% (hopper, halfcheetah), 5-7% (maze2d, pen-expert), 15-20% (walker2d) **[replication]** | as listed above | yes |
| 2 | The min band fails for the Q1 residual in more than 8% of banks on at least 4 of 5 pools; the ranking stays walker2d > hopper > halfcheetah > maze2d ≈ pen-expert; each pool within ±10 points of the alignment study **[replication]** | more than 8% on 5 of 5. maze2d ranks above halfcheetah, and maze2d ≈ pen-expert fails (27.4 against 10.5). hopper (+27.9 points), walker2d (+12.2) and maze2d (+16.7) are outside ±10 | **no** (magnitude and ranking) |
| 3 | The q1 band fails for the min residual in at most 3% of banks **[replication, low confidence]** | 0.0-2.9% | yes |
| 4a | maxabs's failure for the Q1 and min residuals cannot exceed its own failure **[derived]** | 0.0% against 4.4-5.1% | yes |
| 4b | ... and is expected to be at most 1% **[replication]** | 0.0% | yes |
| 5 | The maxabs band is 3-15% wider than q1's; the q1 and min bands are within ±10% of each other **[forecast]** | maxabs/q1 1.09-1.19; q1/min 1.02-1.10 | **no**: hopper (1.155), walker2d (1.154) and maze2d (1.190) are above 1.15; walker2d's q1/min is 1.103 |
| 6 | Mean dose 1.25-1.33, row SD at most 0.06, signal means within 0.03 **[derived from B1; forecast for the numbers]** | 1.26-1.33; SD up to 0.053; signal means within 0.006 | yes |

**Why the misses (readings, not tested).**
- **Item 2.** The alignment study used each pool's own σ, fit online while the critic moved. Here σ is refit
  from scratch to the frozen final critic. My reading is that the refit min band fits the min residual more
  tightly and leaves more of Q1's excess uncovered. No output compares the two bands directly, and the
  pre-registration had reasoned the opposite ("a σ refit … changes the band's shape, not that blindness").
  Either way the main claim got stronger: the min band misses Q1's error on every healthy pool.
- **Item 5.** maxabs's own σ widens where the heads disagree. maze2d has both the largest head disagreement (gap ÷
  residual 1.15) and the widest maxabs band (1.19). Across the five pools the two rank similarly but not
  identically. The misses are 0.3-4 points beyond the forecast range.

**What it means.**
- Swapping the signal changes what the band certifies: min does not cover Q1's error, and q1 does.
- It hardly changes what the actor receives: the signals' mean doses differ by at most 0.006.

## Step 3. Linear-quadratic harness with exact Q

**What changed.** The harness is `experiments/signal/lq_harness.py`. It uses a continuous-action MDP with linear
dynamics, quadratic reward and linear policies, so Q^π, J(π) and every gradient are exact.
- **Critic heads:** exact Q^π plus a controlled error, in six cases: q1_optimistic, q2_optimistic, shared_bias,
  noisy_reward, clean and independent_errors (Addendum A1).
- **Error sizes:** κ ∈ {0.5, 1, 2}.
- **Distance of π from β:** three offsets, {0.5, 1, 2}.
- **Grid:** good and poor behaviour data, 5 replicates. Each replicate draws its own data; within a replicate,
  all arms and controls share the data.
- **Per cell:** σ fit and calibration with BCA's own code, coverage of the true Q1 error at logged actions and at
  π(s) (from the oracle), and TD3+BC actor updates with BCA's dose against four controls: no BCA, a constant dose
  at the same mean, shuffled doses, and an oracle dose built from the true Q1 error at the logged action.

*Change in true value J after the actor update* (q1 signal, mean over κ and offsets, 5 replicates each; "BCA −
constant" is the mean paired difference):

| Case | Behaviour | No BCA | BCA | Constant dose | Shuffled dose | Oracle dose | BCA − constant |
|---|---|---|---|---|---|---|---|
| clean | good | 1.0340 | 1.0332 | 1.0334 | 1.0334 | 1.0334 | −0.0002 |
| clean | poor | 3.7349 | 3.6354 | 3.6405 | 3.6404 | 3.6405 | −0.0051 |
| noisy_reward | good | 1.0340 | 1.0333 | 1.0334 | 1.0334 | 1.0334 | −0.0001 |
| noisy_reward | poor | 3.7349 | 3.6356 | 3.6400 | 3.6399 | 3.6400 | −0.0044 |
| q1_optimistic | good | 0.8962 | 0.9336 | 0.9333 | 0.9333 | 0.9310 | +0.0003 |
| q1_optimistic | poor | 3.8151 | 3.7448 | 3.7487 | 3.7486 | 3.7175 | −0.0038 |
| q2_optimistic | good | 1.0340 | 1.0332 | 1.0334 | 1.0334 | 1.0334 | −0.0002 |
| q2_optimistic | poor | 3.7349 | 3.6354 | 3.6405 | 3.6404 | 3.6405 | −0.0051 |
| shared_bias | good | 0.1607 | 0.1737 | 0.1732 | 0.1731 | 0.1815 | +0.0005 |
| shared_bias | poor | 3.5993 | 3.6548 | 3.6549 | 3.6549 | 3.6136 | −0.0001 |
| independent_errors | good | 0.2758 | 0.2996 | 0.2983 | 0.2983 | 0.4423 | +0.0014 |
| independent_errors | poor | 3.4897 | 3.3856 | 3.4012 | 3.4009 | 2.3899 | −0.0156 |

Where the true Q1 error is zero (clean, noisy_reward, q2_optimistic), the oracle dose is undefined. It falls
back to the constant dose.

**BCA against its controls, cell by cell** (216 cells: 6 cases × 3 signals × 2 behaviours × κ and offset
settings):
- **Worse than both controls by more than 2 paired SE in 121 cells; better in 60; within 2 SE of both in 34;
  mixed in 1.**
- The 60 better cells are in shared_bias (poor 15, good 6), q1_optimistic (poor 12, good 6), q2_optimistic poor
  (9), independent_errors good (6), and clean and noisy_reward poor (3 each).
- The largest gains over the constant dose that clear 2 SE are +0.0176 (shared_bias, poor data, κ = 2, offset 2;
  z = 5.9) and +0.0061 (shared_bias, good data, κ = 2, offset 0.5; z = 23). The largest in q1_optimistic is +0.0039
  (good data, κ = 2, offset 1; z = 8.0). A +0.0056 in independent_errors (good data, offset 2) does not clear 2 SE
  (z = 1.5).
- 15 of the 60 cells where BCA beats both controls are in clean, noisy_reward and q2_optimistic. There Q1 is
  exact, so there is nothing to target. All 15 are poor data at offset 0.5, the same slice as E6's exception, so
  they are wins from varying the dose, not from targeting Q1.
- **Size:** |BCA − constant| is a median 6% of |BCA − none| (90th percentile 30%). The median |BCA − none| is
  0.0022, and the case means reach about 0.1 with poor data.

**Oracle dose against constant dose** (mean, with range across cells):

| Case | Good data | Poor data |
|---|---|---|
| independent_errors | +0.144 (+0.031 to +0.279) | −1.008 (−3.03 to −0.002) |
| q1_optimistic | −0.002 (−0.059 to +0.129) | −0.031 (−0.104 to +0.005) |
| shared_bias | +0.008 (−0.064 to +0.126) | −0.041 (−0.247 to +0.056) |

Coverage and doses:
- **q1_optimistic, logged actions.** Every band covers its own residual about 91% of the time. The min band misses
  the true Q1 error more as κ grows (good data, offset 0.5: 0.0%, 1.0%, 10.7%). The q1 and maxabs bands miss it
  on at most 0.2% of rows.
- **independent_errors, logged actions.** True-Q1-error miss: min 2.0-3.9%, q1 0.8-1.4%, maxabs 0.4-0.6%.
- **shared_bias.** Every band covers its own residual about 91% of the time. The true Q1 error at logged
  actions (q1 band) is missed on:

  | Behaviour, κ | offset 0.5 | offset 1 | offset 2 |
  |---|---|---|---|
  | good, 0.5 | 0.0% | 0.3% | 100% |
  | good, 1 | 1.1% | 100% | 100% |
  | good, 2 | 86.6% | 100% | 100% |
  | poor, 0.5 | 0.0% | 0.0% | 97.7% |
  | poor, 1 | 0.0% | 45.0% | 100% |
  | poor, 2 | 1.0% | 100% | 100% |
- **Dose.**
  - Cell-mean doses are 1.29-1.35; case means are 1.30-1.32. Each replicate's maximum dose averages at most 1.42
    across replicates, and the largest single dose is 1.448.
  - Correlation of the dose with the true Q1 error averages 0.4-0.8 per case for q1 and maxabs. For min it is
    0.15-0.65, except in shared_bias, where min equals q1 (0.73-0.80). q1 cells range from 0.14 to 0.92.

**Met?**

| # | Expectation | Result | Met? |
|---|---|---|---|
| check 1 | Q^π matches Monte Carlo rollouts and satisfies the Bellman identity **[derived]** | covered by `test_lq_harness` (passes; `tests_per_module.log`), not by the run's own summary | yes (by tests) |
| checks 2-5 | Equal heads ⇒ identical signals; min ≡ q1 in q2_optimistic; min ≡ clean in q1_optimistic; same Q-term gradient **[derived]** | all hold in every replicate, with identical targets and batches across signals | yes |
| E1 | The mean dose is 1.2-1.4 in every case, including exact critics, and never reaches 1.5; with exact critics BCA still applies about 1.3× BC **[derived]** | 1.29-1.35; maximum single dose 1.448 | yes |
| E2 | q1_optimistic: each signal covers its own residual about 90% **[derived]**; the min band misses the true Q1 error on a share that grows with κ **[derived]**; q1 and maxabs miss it on at most about 20%, probably 10% **[derived bound; forecast for 10%]** | own about 91%; min 0.0 → 1.0 → 10.7%; q1 and maxabs at most 0.2% | yes |
| E3 | At π(s), every signal misses the true Q1 error more often than at logged actions, and more so the further π is from β **[forecast]** | the main clause holds in only 48 of 106 non-trivial cells. The distance clause holds: the miss at π(s) does not fall with offset in 34 of 38 series (min band, independent_errors, good: 2.7 → 3.7 → 7.9%) | **no** (main clause); distance clause yes |
| E4 | q2_optimistic: min and q1 ignore Q2's error; maxabs raises the dose where ‖a − β‖ is large **[derived]**; with poor data, maxabs's gain in J is no larger than min's or q1's **[forecast]** | corr(dose, ‖a − β‖): maxabs 0.84 against 0.37 (good data), 0.42 against 0.15 (poor); poor-data J 3.6352 against 3.6354 | yes (hand verdict) |
| E5 | BCA's change in J is within 2 SE of the constant and shuffled controls in every case, while the gap to no BCA is larger. The oracle dose also lands close to the constant dose **[forecast, moderate confidence]** | within 2 SE of both in 34 of 216 cells, worse in 121, better in 60; gap to no BCA larger in 215 of 216. Oracle against constant: −3.03 to +0.28 | **no**, on both clauses |
| E6 | Exact critics: J improves at least as much without BCA as with it, and the gap is larger with poor data **[forecast]** | first clause: 30 of 36 cells. All 6 exceptions form one slice (poor data, offset 0.5, both cases, all signals), where BCA is ahead by +0.0023 (z ≈ 18). Second clause: holds at offsets 1-2, reverses at 0.5 | **no**: small but systematic exception |
| E7 | shared_bias: the three signals are identical; residual coverage stays near nominal while coverage of the true error can fail badly **[derived]** | identical; about 91% residual coverage throughout, while the true error is missed on up to 100% of rows as κ and the offset grow (table above) | yes (hand verdict) |
| A1.1 | independent_errors: the signals differ; each covers its own residual about 90% **[derived]** | they differ; own miss 8.4-9.1% | yes (hand verdict) |
| A1.2 | The min band misses the true Q1 error more often than the q1 band **[forecast]** | min 2.0-3.9% against q1 0.8-1.4% | yes |
| A1.3 | The maxabs band covers both heads' residuals at least as often as the q1 band, and it is wider **[derived, up to calibration noise]** | Q1 and Q2 residual miss: maxabs 6.2-6.9% and 5.0-5.9%, against q1 8.6-9.1% and 9.2-10.4%. Width 0.92-4.56 against 0.82-4.10 | yes (hand verdict) |
| A1.4 | Targeting against strength, as E5 **[forecast]** | independent_errors: BCA − constant +0.0014 with good data, −0.016 with poor | **no** (as E5) |

**Why the misses (readings, not tested).**
- **E5.** The paired comparisons are very precise. Each replicate's arms share data, so differences of
  0.00001-0.018 in J reach a median |z| of 7 (maximum 35) among the cells outside 2 SE.
  - In size, BCA's departure from the constant dose is small next to its effect against no BCA: median 6%.
  - In sign, it is more often harmful than helpful: 121 cells worse, 60 better.
  - The oracle clause failed outright. A dose built from the true error is far from constant in effect, by far
    the most in independent_errors (+0.14 good, −1.01 poor on average). Individual cells elsewhere also reach
    −0.247 (shared_bias, poor) and +0.129 (q1_optimistic, good), an order of magnitude beyond BCA's own departures
    (at most 0.018). Post hoc, the oracle's BC gradient was 2-3× the constant dose's in independent_errors at the
    same mean, so much of this is strength, not targeting (see "Post hoc: same mean, different strength").
- **E3.** The forecast assumed π(s) is always further from the data than logged actions. My reading is that the
  error terms grow with ‖a − β(s)‖, and that at small offsets π(s) sits closer to β than noisy logged actions
  do. This is only partly consistent with the data: with poor data at offset 2, π is further from β than logged
  actions (0.41 against 0.25), yet it is covered as well or better in 11 of 15 cells. The explanation is
  untested.
- **E6.** With poor data at the smallest offset, the extra BC helps slightly (+0.0023, very consistently).
  Otherwise, as forecast, extra BC slows improvement when the critic is exact.

**What it means (readings).**
- **The signal is there; the hook dampens it.** The dose tracks where Q1 is wrong. But the hook
  1 + 0.5·λη / (1 + λη) caps it below 1.5. On D4RL, the doses at λ\* (1.25-1.32) imply λη of about 1-2, where
  the curve is flat, so row-to-row variation is small. That this cap is why targeting is weak is plausible, but
  untested.
- **The large oracle effects are mostly strength.** They appear in one case (independent_errors: +0.14 good,
  −1.01 poor), where the oracle's BC gradient was 2-3× the constant dose's at the same mean. In q1_optimistic and
  shared_bias, where the oracle's strength is within 1.0-1.3× of the constant's, it is about neutral with good
  data and harmful with poor. So these results do not show that targeting moves J (post hoc, below).
- **Behaviour quality sets the sign.** More BC where Q1 is wrong pulls toward the logged actions, which helps
  only when they are good. That is the cross the user asked step 4 to test.

## Post hoc: same mean, different strength (not pre-registered)

Found while designing step 4, after steps 1-3 were scored and audited. Script:
`experiments/signal/strength_reanalysis.py` (SHA-256 `8e54b358…`). Output: `runs/wbcp_signal/strength_reanalysis.json`.

Step 3 matched the constant control to each arm's mean dose. The BC term's actual gradient norm at the starting
actor shows that this did not match strength (q1 signal, per replicate, relative to the constant dose):

| Case | Oracle / constant | BCA / constant |
|---|---|---|
| independent_errors, good | 2.25-3.06 (median 2.61) | 1.005-1.017 |
| independent_errors, poor | 1.07-2.82 (median 2.26) | 1.012-1.034 |
| shared_bias, good / poor | 1.05-1.26 / 1.16-1.29 | 1.002-1.024 |
| q1_optimistic, good / poor | 0.87-1.01 / 0.98-1.11 | 1.000-1.021 |
| clean, noisy_reward, q2_optimistic | 1.00 (oracle = constant) | 1.000-1.024 |

**Why.** Rows with a large true Q1 error (oracle) or a wide band (BCA) tend to be rows with large per-row gradient
leverage (large ‖s‖ or ‖π(s) − a‖). Weighting them up enlarges the BC gradient at the same mean weight.

**A strength-only model** predicts BCA − constant from BCA's realized BC strength alone:
(J_constant − J_none) × (‖g_BC, BCA‖ / ‖g_BC, constant‖ − 1) × d / (d − 1), with d the mean dose. Across all 216
cells it correlates 0.85 with the observed BCA − constant.

**What it changes.**
- The oracle's large effects in independent_errors cannot be credited to targeting: it was a 2-3× stronger dose.
- BCA's own departures from the constant dose, its 60 wins and 121 losses, are mostly its slightly higher BC
  strength (1-3%), not targeting. This makes the reading of decision rules 2 and 3 more one-sided: the evidence
  for any targeting in step 3 is weaker than stated above.
- Step 4 must therefore match **realized strength** (the update's actual size), not the mean dose. Its design
  does (`runs/wbcp_signal/step4/`).

## Decision rules (fixed in advance): what each says, and how it was read

1. **Rule:** if step-2 items 1 and 2 hold for q1, q1 becomes the measurement signal for step 4.
   - Item 1 holds. Item 2 holds in its main clause (more than 8% on 5 of 5) but misses on magnitude and ranking.
   - **Reading (judgment):** the precondition is partly met, in the direction that strengthens the case.
     Adopted.
2. **Rule:** if BCA matches the constant dose (E5), the hook applies the signal as uniform extra BC, and step 4
   must compare where it is applied, with a dose of real dynamic range, against constant and shuffled controls.
   - E5 did not hold as written.
   - **Reading (post hoc):** BCA's departure from the constant dose is small (median 6% of its effect) and
     mostly harmful, which is close to uniform extra BC in practice. The size criterion was not pre-registered.
     Partly triggered.
3. **Rule:** if BCA beats both controls by more than 2 SE, targeting is real in LQ, and step 4 tests whether it
   survives on D4RL.
   - **As written, triggered in 60 of 216 cells** (it is beaten in 121).
   - **Reading (post hoc):** partly triggered. The gains are small (at most +0.018).
     - 39 of the 60 are in shared_bias and q1_optimistic (either behaviour).
     - 15 are in cases with no Q1 error (clean, noisy_reward, q2_optimistic, all poor data at offset 0.5).
     - So whether any of it is targeting is open. Step 4 should test whether it survives on D4RL.
4. **Rule:** if coverage at the actor's own actions fails (E3), the logged/proposed gap is the main hole, and
   step 5 moves ahead of step 6.
   - E3 missed in the opposite direction: π(s) was often covered better.
   - Coverage at π(s) does fail in shared_bias, which no residual can detect, and it worsens with distance from
     β.
   - **Reading (judgment):** not triggered as a reordering. Step 5 still has to test proposed actions on real
     tasks, which D4RL cannot do.
5. **Rule:** if maxabs is never better on actor outcomes and is always wider, drop it.
   - maxabs's J exceeds both q1's and min's in 10 of 72 cells, by at most 4×10⁻⁵. It beats one of them by more
     than 2 paired SE in 8 comparisons, at most +0.0012 (z ≈ 3.8). It is always wider (D4RL 1.09-1.19× q1).
   - **Reading (judgment):** never meaningfully better, rather than never better. Dropped.

## Limits

- **The LQ errors are constructed.** Real critic error may not grow with ‖a − β‖, and the oracle dose's effects
  depend on the error model. The oracle's large effects come from one case.
- **D4RL checks only logged actions.** Coverage at π(s) on real tasks needs fresh transitions or simulator labels
  (step 5).
- **σ is refit to frozen critics.** In BCA it is fit online while the critic moves, which may be why step 2's
  min-band misses run higher than the alignment study's (untested).
- **Two TD3+BC critics never converged.** pen-cloned and pen-human are excluded from verdicts.
- **Step 1b's evidence is provisional.** It predates the trim and uses the min score. It will be regenerated
  once the post-trim runs and Q1 runs exist.
- **Not every surplus design was rerun.** Among others, walker2d K = 23, pen-cloned K = 5, pen-human K = 10,
  the K = 5 shift designs, CQL hopper K = 6, TD3+BC hopper K = 5 at n = 1,103 and every 8,192-row bank were not
  rerun after the trim.

## Reproduce

```
bash runs/wbcp_signal/run_all.sh            # steps 3, 2, 1, in that order (CPU)
python experiments/signal/scorecard.py      # runs/wbcp_signal/scorecard.md
```
