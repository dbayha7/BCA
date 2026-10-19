
---

## 7. Marks, decision rules, kill rules and what each outcome would mean (fixed now)

All bars are in points (strong current) unless stated. "LB > 0" is the one-sided 95% bound over seeds of section 8.1.
"Holm p <= 0.05" is the Holm-adjusted one-sided p within the family named (section 8.3). Where a mark compares with
"the pooled-best" of several policies, that policy is fixed first by its pooled mean, and the per-seed differences are
then taken against it.

### 7.1 Kill and stop rules (read first, in this order)

**Stop rules** (each stops its phase; a stopped phase is never rerun or replaced; every claim that needs it fails):
- S1 fingerprints: Unifloral files against `configs/unifloral.json`; the frozen prespec and the code manifest (every
  placement module, every `experiments/shift_hosts` module it imports, `baselines/unifloral/algorithms/bc.py`,
  `configs/unifloral.json` and the hosts' `EXTERNAL_CODE` paths, HOSTS `gate.py:56-65`) against your approval.
- S2 seeds: the allow-list of section 3.4.
- S3 reproduction: C-k0 must equal the host bit for bit (restated step with k = 0; tolerance 1e-5 per parameter at
  most), on the first seed-world of each stage; the first host of each stage is trained twice and must match bit for
  bit. C-ORACLE is exempt (x64 step).
- S4 float32 check (the hosts' S7): open loop within 1e-7; closed loop within 0.05 points per start (declared now,
  before any data, for the reason HV2 PRESPEC.md:11-14 and 888-915 give).
- S5 non-finite: any non-finite network, score, threshold or return.
- S6 equal-weight identity: the exact tail gate equals the exact binomial test on 100% of decisions.
- S7 attribution identity: `|sum_t term_t - (J_new - J_ref)| <= 1e-8`.
- S8 runner: bit equality with gymnasium `env.step` on the spot-check starts, and attribution's `J_new` equal to the
  online J there.
- S9 leaks: as section 3.2.
- S10 budget: before a stage starts, its projected cost is checked against its cap (section 9.2); during the stage it
  stops between whole seeds only if measured spend would pass the cap.
- S11 bytecode: no `__pycache__` folder or `.pyc` file in the package or in `experiments/`, `experiments/shift`,
  `experiments/shift_iw`, `experiments/shift_hosts`, `calibration` or `baselines/unifloral/algorithms`, checked before
  and after every process (the hosts' gate refuses such files, HOSTS `gate.py:56-65, 124-134`).

**Kill rules** (judged on the pooled seeds of the stage):
- **K1 host learned**: the host's recorded-conditions points are at least the clone's + 10 in S and in N. Otherwise "did
  not learn": no claim is judged on that host.
- **K2 room for A and B**: `V_S >= +20` and `V_N <= -10` (strong current, pooled). Otherwise A's and B's return claims
  read "no room" and are not judged.
- **K3 room for C (= C-R0)**: the largest of `P_S(oracle) - P_S(host)` over C-ORACLE (co-trained), C-ORACLE-S1 and
  C-ORACLE-V (actor-only) is at least +20, strong current, pooled. Otherwise C's return claims are not judged and C's
  matched controls are not trained; C-BCA's coverage claims are still judged.
- **K4 margin**: the margin rule finds a crossing in at least 3 of 5 pilot seeds. Otherwise no confirmatory stage for A
  and B without a new version.

### 7.2 Option A marks

| Claim | Pass iff |
|---|---|
| **A-R1** | (i) `NET(A-BCA, clip) - NET(W*)`, W* the pooled-best member of W: mean >= +10 and LB > 0; (ii) `NET(A-BCA) - NET(A-RAND)` with the exact expectation `NET_k(A-RAND) = r [P_k,S(clip) - P_k,S(host)] + r [P_k,N(clip) - P_k,N(host)]`, r = A-BCA's on-rate pooled over the stage's S and N seed-worlds: mean >= +10 and LB > 0; (iii) in S, recorded / big waves / big kicks each >= host - 3 (pooled mean, step 2's G2 form) |
| **A-V1** | the boundary false-switch count pooled over every S and N seed-world of the stage (R = 1,000 each) has a one-sided Clopper-Pearson 95% upper bound <= 0.065, and no seed-world has more than 73 switches in 1,000 |
| A-K1 (machinery) | per seed, `E-NET(A-BCA) - E-NET(rival*)`, rival* the pooled-best of A-SC, A-OS, A-BOOT, A-BB, A-MEAN, A-HL, A-RANK and A-EXC, each at its own margin: mean >= +5 and Holm p <= 0.05 in family {A-K1, A-K4}; and A-BCA leads rival* at least at 3 of the 4 margin multiples 0, 0.5, 1 and 2 |
| A-K4 (machinery) | per seed, `NET_mixed(A-KERN) - NET_mixed(rival*)`, rival* the pooled-best of world-level A-BCA, A-CELL, per-cell A-SC, A-KQ, A-KMEAN, clip always and the pilot's best cap: mean >= +5 and Holm p <= 0.05 in family {A-K1, A-K4}; and under A-KPERM A-KERN keeps less than half of that lead (mean over the 200 permutations) |
| A-K2 (outcome) | per seed, `E-NET_lad(A-BCA) - E-NET_lad(R*)`, R* the pooled-better of clipping always and never clipping: mean >= +10 and Holm p <= 0.05 in family {A-K2, A-K3}; confirmatory seeds only (on the pilot m is fitted on the same ladder) |
| A-K3 (outcome) | per seed, realized `NET(A-BCA, held-out) - NET(A-INS)`: mean >= +5 and Holm p <= 0.05 in family {A-K2, A-K3} |
| Shuffle sanity (outcome verdict) | A-SHUF's on-rate in S is at most 0.05 (pooled over seeds and the 200 permutations) while A-BCA's real on-rate in S is at least 0.95 |

Reported, decides nothing: the gated hinge in place of the clip, `NET(A-BCA, gated hinge at m_hinge) - NET(gated hinge
always)`; `NET(A-BCA) - NET(clip always)` against draft 1's +20; A-V2's items (n-matched coverage, natural
false-switch rate, equal-shapes null, rivals' size table); one realized A-RAND coin; a Bayesian-bootstrap E-NET for
A-K3 (resampling the fixed survivors, labeled resampling, not replication); decision accuracy (10 or 20 seed-worlds);
the margin and evidence-size sweeps; the ladder; M3 and M4 with the online monitor; the clone fallback; A-KERN-CAP and
the 0.80 cap; A-ORACLE-REGION's regret.

**Rationale.** A-R1's bar is the best world-blind rule, as step 2's G1 was ("the best world-blind rival + 10", CONN
`step2_results/results.md:9`). With the record's values a switch that false-alarms in 1 of 5 no-stall seeds fails it
(per-seed differences against the hinge always 29.6 four times and -32.2 once: mean 17.2, LB -9.1), and a perfect
switch passes whenever the seed sd of its difference stays below 31 (REV `revise_arith.out` section 9). A-RAND is the
placement control of step 1 (PRESPEC.md:201-202); with one realized coin per seed-world a perfect switch failed it about
21% of the time (`review_stats/marks_arith.out` line 1), so its exact expectation is used (about 83.9 per seed with
record values, REV section 10). A-V1 is the error the switch controls, at its hardest point; by the shift identity its
count is also the coverage of the true gap by L_gap, which rests only on stretched theory for the two-posterior gap
(PLACE rank, O8), so it is measured. Its mark pools independent replicates: a rule at exactly 5% passes with
probability 1.000 (pooled) and 0.994 (guard) over 10 seed-worlds, a rule at 6% about 0.66, at 6.5% about 0.05 (REV
section 1); draft 1's "every seed-world at 0.073" failed an exact 5% rule 60% of the time over 10 seed-worlds
(`review_cov/arith.out`). The +5 credit bar is round 1's (`ROUND1.md:107`); expected NET over replicates removes the
luck of one evidence draw (`ROUND1.md:101-107`). A-K2 is round 1's middle-strength rule (+23.0 over clipping always
against a bar of 10, `ROUND1.md:74-96`); plain split conformal came within 2.4 points of BCA there, so its credit goes to
outcome reading, not to BCA.

**What the outcomes mean (wording fixed for both outcomes; with K2 passed).**

| A-R1 | A-V1 | Meaning |
|---|---|---|
| pass | pass | The harm switch works on a Unifloral host: it beats every world-blind rule and random switching, and at the boundary it fires in at most 5% of evidence draws (L_gap covers the true gap). Credit is read from the outcome and machinery verdicts. |
| pass | fail | Return earned on these seeds, but at the boundary the switch fires more than 5% of the time: L_gap is not a valid 95% bound on the gap; the return rests on the evidence lying far from the margin. If the n-matched coverage passes: "the posterior bounds the test atom's gap, not the margin's". |
| fail, dominated | any | A world-blind rule does as well or better: the switch is not needed here. |
| fail, not dominated | any | Not distinguishable from the best world-blind rule on these seeds. |

A-K1 passing: BCA's posterior beats every plain test at its own break-even margin (credit to the Bayesian step);
failing: plain tests at their own margins match it (expected; the credit stays with outcome reading and the margin).
A-K4 passing (with A-KPERM): credit to localization by weights; failing: credit to the action (a plain cap) or the
world-level switch. A-K2 passing: the switch follows the harm as it grows (outcome reading, shared by plain tests);
failing: on the ladder it does no better than a fixed rule. A-K3 failing: held-out rows are not needed (in-sample
survivors suffice, as round 1 saw, `ROUND1.md:39`).

### 7.3 Option B marks

`alpha_B = 0.1`, level 0.95, the score s, `tau = lambda_R + m` offline (decoupled rest) and `tau_on = lambda_R,on +
m_on` online, the exact tail gate.

| Claim | Pass iff |
|---|---|
| **B-R1** (B-GROUP, offline) | safety: in S, for every situation, `P(p_B) - P(F)` has LB >= -3; and release: `NET(p_B) - NET(F always)`: mean >= +10 and LB > 0 |
| **B-R2** (B-GROUP, staged, batch at N) | safety: in S, the episodes after the decision have LB >= -3 against F (the Seldonian promise is about the policy returned, not about probing); release: on the cumulative points of the whole H = 1,000-episode deployment, reference and probe episodes included, `NET - NET(F always)`: mean >= +10 and LB > 0. The probing cost in S (F always minus the cumulative points) is reported beside it |
| **B-V1** | at the boundary null, separately for B-GROUP offline and B-GROUP staged: the error count pooled over every S and N seed-world of the stage (R = 1,000 each) has a one-sided Clopper-Pearson 95% upper bound <= 0.065, and no seed-world has more than 73 errors in 1,000 |
| **B-V2** | per world with uses (offline map and staged map separately): the two-stage bootstrap 95% upper bound of the seed mean of `HR_k` is <= alpha_B + 0.02; it must pass in at least one world; a world with fewer than 3 seeds having uses reads "vacuous" |
| B-K1 (outcome) | B-GROUP offline in S: per seed, the one-sided permutation p-value of the real harm-event count among the tries against its 200 shuffles (same tau) is <= 0.05 in at least 4 of 5 seeds (8 of 10) |
| B-K2, B-K3 (machinery) | module W only (section 10) |

Reported beside B-R1 and B-R2: the best world-blind rule's NET and A's switched clip on the same seeds; B-RUNG offline
and staged; the shipped gate and every other gate; the looks and the e-process; the twin outcome; the realized mean
variant; the clone fallback; refusal prices; B's safety readouts (section 6.4); the certification rates with real and
shuffled scores (both over the 1,000 replicate try sets); arm B's gain g against B-K1's statistic; the confusion table
of A's and B's decisions on the same evidence.

**Rationale.** B's default is the fallback, so its safety mark is non-inferiority to F in the world where deviation
harms. It uses a bound (LB >= -3), not a mean, because safety is B's own promise; A's and C's "keeps the improvement"
is a side condition copied from step 2's G2 (pooled mean >= -3, step 2 PRESPEC.md:223-224). When B refuses everywhere,
`p_B = F`, every difference is 0, sd = 0 and LB = 0: safety passes. The release mark is the release of F where
deviation helps (+10 over clipping always, a sixth of the no-stall clip cost). In the stalling world 100 probe episodes
of 1,000 cost about a tenth of the value of caution, which is why B-R2's safety is judged after the decision and its
release on the whole deployment. B-V1 is the promise, judged where it can fail: with a fixed threshold and independent
tries the exact gate's size near the boundary is only 2.3-3.4% at 36-100 tries (`review_cov/arith.out`); what can push
it past 5% is the clustering of tries in episodes and a threshold computed from data, which only replicates near the
boundary show. B-V2 transfers the certificate's own event to the deployed map; the share of negative attribution terms
is a readout, because one negative term has no fixed relation to `s > tau` (in N a full-throttle step can lose a little
to the energy cost `0.3 |a|^2`, `thruster_chain.py:22`, while the map gains overall). The +0.02 allowance covers the
bootstrap's resolution at the expected number of uses. B-K1's draft-1 form (a certification-rate difference) could fail
even when the score carries the harm, since at 50 tries the exact gate certifies only 28% of the time at a harm rate of
0.05 (`review_cov/arith.out`); the permutation test does not depend on certifying, and with no information it passes
with probability 3e-5 (REV section 8). Release power is set in section 8.5.

**What the outcomes mean.**

| B-R | B-V1 | B-V2 | Meaning |
|---|---|---|---|
| pass | pass | pass | A safety certificate per deviation works here: it keeps the fallback where deviation harms and releases it where it helps. Credit to the placement only (equal weights make BCA the exact binomial test). |
| safety pass, release fails, marked "underpowered by design" (8.5) | pass | any | Safe; it refuses because the evidence is too small for the no-stall harm rate: evidence size, not the method. |
| safety pass, release fails, not underpowered | pass | pass or vacuous | Safe but refuses where it should release. |
| release passes, but dominated by W or by A's switched clip | any | any | It releases F, but a world-blind rule or the harm switch does as well: the certificate is not needed. |
| safety fails | any | fail | The certificate lets harm through: the score does not carry the harm at deployment. |
| pass | pass | fail | Promise kept on the tries, broken in deployment: the try law does not represent the deployed map. |
| any | fail | any | The tail gate's promise breaks near the boundary on clustered tries with a threshold computed from data. |

B-R2 passing and B-R1 failing: online tries rescue the per-deviation certificate at the probing cost reported. B-K1
failing: the score's tries are not worse than shuffled ones; the score does not carry the harm.

### 7.4 Option C marks

| Claim | Pass iff |
|---|---|
| C-R0 (kill K3) | the largest `P_S(oracle) - P_S(host)` over C-ORACLE, C-ORACLE-S1 and C-ORACLE-V is >= +20, pooled, strong current |
| **C-R1** | per seed, `NET(C*) - NET(P*)`, C* the pooled-best of C-BCA and C-BCA x A, P* the pooled-best of CLIP x A, HINGE1 x A and C-HINGE x A: mean >= +5 and LB > 0; and in S, recorded / big waves / big kicks of C* each >= host - 3 (pooled) |
| **C-V1** | `COR` (mean over the 7 declared refreshes) >= 0.90 as the seed mean with LB >= 0.88, in S and in N |
| **C-V2** | `RS <= 0.50` and `CAC >= 0.85` (seed means, soft refusals counted as refusals), overall and in the near-knee hard-water zone, in S and in N |
| **C-V3** | `F3`, pooled over S and N, has a one-sided 95% upper bound <= 0.073 (bootstrap over (seed-world, bank) units, 2,000 resamples) |
| C-K1 (machinery) | `NET(C-BCA) - NET(C-F0_m)`: mean >= +5 and Holm p <= 0.05 in family {C-K1, C-K2, C-K3, C-K4} |
| C-K2 (machinery) | `NET(C-BCA) - NET(C-RATIO_m)`: the same |
| C-K3 (machinery) | `NET(C-BCA) - NET(C-SC_m)`: the same |
| C-K4 (machinery) | `NET(C-BCA) - NET(C-SUP)`: the same |
| C-K6 (outcome) | `NET(C-BCA) - NET(C-SHUF_m)`: mean >= +5 and one-sided p <= 0.05 |

Reported: C-R1b, `NET(C-BCA) - NET(C-HINGE)` (draft 1's primary; mean >= +5 and LB > 0 shown against it); every k = 1
version; C-EVEN and C-RAND; the three oracles; C-I1 to C-I3; each `COR_j`, staleness and the final-refresh band; RS and
CAC past 80%, inside and outside the ball; `F3(C-SC) - F3(C-BCA)`; the actual bank's COR per seed-world.

**Rationale.** The record puts C's bars at the switched clip (106.1) and the switched hinge (107.4), not at the host
(PLACE `red.challenges[6]`): within the stalling world a no-signal clip beat the exact-error oracle (93.1 against 79.7,
stage 1 `RESULTS.md:44, 55`), and the lower edge lost to a matched range hinge by 45.6 (results-2026-10-14
`README.md:25`). So C-R1 compares C's best variant with the best plain action a switch can deploy; a C that beats only
the always-on matched hinge while far below A's switched clip is not a useful placement. C-V1 is the band's own promise
where the band acts: the certificate is about the critic at the refresh (PLACE `theory.findings[5]`), and nothing uses a
band refreshed on the final networks. C-V2 blocks meeting the promise by refusing, overall and where a finite band is
possible (48-75% of draws were refused before, PLACE `red.challenges[11]`); at RS = 0.5 the promise alone implies only
CAC >= 0.80 (`review_cov/arith.out`), so CAC is a usefulness bar. C-V3 is the per-bank promise that the Bayesian step
adds (DEPENDENCE.md:10-12; PLACE `theory.findings[9]`); over 2,000 units a rule at exactly 5% passes with probability
0.996 and a rule at 8% with 0.002 (REV section 2). The credit claims compare at matched strength, because step 2 could
not tell "Bayesian" from "larger" (CONN `step2_results/results.md:34, 38`). C-I1 is expected to fail (eta's coefficient
of variation 0.065-0.111 against a ratio spanning orders of magnitude, PLACE `red.challenges[11]`).

**What the outcomes mean.**

| C-R1 | C-V1 | C-V2 | C-V3 | Meaning |
|---|---|---|---|---|
| pass | pass | pass | pass | C beats every plain action a switch can deploy, and its band keeps its promise where it is used, bank by bank, without hiding behind refusal. |
| fail, dominated | any | any | any | A's switched clip or hinge does better: once the switch is right, the lower edge is not needed (expected). |
| fail, not dominated | any | any | any | Not distinguishable from the switched plain actions on these seeds. |
| any | pass | fail | any | The promise is met by refusing; where a finite band is possible it is refused or too wide to use. |
| any | fail | any | any | The band does not keep its promise at the place of use on a Unifloral critic. |
| any | pass | any | fail | The average promise holds but not bank by bank: the per-bank promise fails under these weights and this dependence. |

C-K1 failing: the gain is refusal (no value credit past the data), not the band (as F - F0 = +0.14/+0.08 before, CONN
`arms_abc_f0/RESULTS.md:122-123`); on the main recorder this is expected past 80% by construction. C-K2 failing: eta
carries nothing (wording also by C-I1). C-K3 failing: at equal strength the Bayesian threshold shapes the actor no
better than split conformal. C-K4 failing: the band acts as a support rule. C-K6 failing: the band's shape does not
depend on outcomes.

### 7.5 Across options: verdicts, domination and the credit map

**Three verdicts per option** (for B, separately offline and staged; for C, separately on the main recorder and in
module W):

1. **Placement works**: every primary claim passes (A: A-R1, A-V1; B offline: B-R1, B-V1 offline, B-V2 of the offline
   map; B staged: B-R2, B-V1 staged, B-V2 of the staged map; C: C-R1, C-V1, C-V2, C-V3), its kill rules passed, and the
   option is not dominated.
2. **Outcome reading earns its place**: A: at least one of A-K2, A-K3 passes in its Holm family and the shuffle sanity
   holds; B: B-K1; C: C-K6. The verdict says that this credit is shared by every plain test that reads the same score.
3. **BCA's own machinery earns credit**: at least one machinery claim passes in its Holm family: A {A-K1, A-K4}; C on the
   main recorder {C-K1, C-K2, C-K3, C-K4}; module W {C-K1W, C-K2W, C-K3W, C-K4W, C-K5} and {B-K2, B-K3}. The verdict
   names the family and the part.

**Dominated** (computed for every option): the best plain alternative available to the same deployment (A: the
pooled-best of W; B: the pooled-best of W and A's switched clip; C: the pooled-best of CLIP x A, HINGE1 x A, C-HINGE x A
and W) beats the option's pooled-best variant by at least 5 NET points with LB > 0 over the per-seed differences. A
dominated option cannot read "placement works". At a true difference of 0 this reading arises at most about 4% of the
time at a seed sd of 8.8 (REV section 11).

"At least one placement works" uses Holm's correction over the three options' primary return claims.

A and B read the same score with the same margin; they differ in the default. Where the evidence is thin, A leaves the
host alone and B falls back. The paired table of their decisions per seed-world is reported, with the returns of each
disagreement.

**Credit map: which comparison separates which part of BCA from which plain rule.**

| BCA's part | Separated from | By | Verdict |
|---|---|---|---|
| The residual score as an outcome reader | rules that read no outcome; shuffled outcomes | A-R1 (against W and random switching), A-K2, the shuffle sanity, B-K1, C-K6 | outcome (shared with plain tests) |
| Reading held-out rows | in-sample survivors | A-K3 | outcome |
| The Bayesian step (WBCP's posterior and test atom) | split conformal, order statistics, bootstrap, means, rank test, exact binomial, each at its own margin | A-K1, C-K3, C-K3W; for B none: with equal weights it is the exact binomial test (stop rule S6); C-V3 (per-bank promise, reported credit shared with Wilks' tolerance limits) | machinery |
| eta (the scale network) | the same band with eta = 1, matched | C-K2, C-K2W (wording also by C-I1) | machinery |
| The band's finite width | refusal alone, matched | C-K1, C-K1W | machinery |
| The band beyond support | a support-only penalty and a range hinge of the same strength | C-K4, C-K4W, C-R1b | machinery |
| The weights | unweighted posteriors and localizers, rejection sampling, weighted split conformal, weighted means | A-K4 (localizing weights, mixed worlds); B-K2, B-K3, C-K5 (importance weights, module W) | machinery |
| The margin (not BCA's, but carried the record) | the same rules at margin 0 | reported: A-BCA0 and the margin sweeps | none |

---

## 8. Statistics

### 8.1 Pairing and seeds

- For each claim's difference D (between two policies, or two rules), per seed: the mean over the bank's 1,000 starts
  (paired by start), with its paired standard error.
- Across seeds: the mean and sd (n - 1) of the per-seed values; `LB = mean - t(0.95, S - 1) x sd / sqrt(S)`
  (t = 2.1318 for S = 5, 1.8331 for S = 10). One-sided p = `P(T_{S-1} >= mean / (sd / sqrt(S)))`; when sd = 0, p = 0 if
  the mean is above 0, else 1 (HOSTS PRESPEC.md:540-542).
- **Nothing is shown until a stage is complete.** Until every seed of a stage is complete, only the integrity checks
  (S1-S11) and the spend are shown; no per-seed value, pooled value or bound. (Draft 1 showed per-seed values as they
  arrived; together with a stop between whole seeds that would have let early seeds steer a stop.)

### 8.2 Coverage and error rates

Per seed: episode bootstrap for step-level rates; Clopper-Pearson for replicate rates. Replicate rates that are judged
(A-V1, B-V1, C-V3) pool independent replicates over seed-worlds with a per-unit guard (section 7). Across seeds: seed
mean with the t bound, plus the count of seeds below the bar.

### 8.3 Multiplicity

- Within an option, a verdict needs all its primary claims: no correction (all must pass).
- Across the three options: each verdict at one-sided 5%; "at least one placement works" uses Holm's correction over
  the three primary return claims.
- **Families** (Holm within each, as the hosts apply it: sort the one-sided p-values ascending, ties by claim id; the
  i-th smallest passes if it and every smaller one are at most `0.05 / (k - i + 1)`; the adjusted p is the running
  maximum of `min(1, (k - i + 1) p)`, HOSTS PRESPEC.md:543-548): A machinery {A-K1, A-K4}; A outcome {A-K2, A-K3}; C
  machinery on the main recorder {C-K1, C-K2, C-K3, C-K4}; module W C {C-K1W, C-K2W, C-K3W, C-K4W, C-K5}; module W B
  {B-K2, B-K3}. C-K6 and B-K1 are alone. LB is also reported at the Holm level. Each family keeps the chance of any false
  pass at 5%; the verdicts name their family, so no correction is made across families.
- Fixed-rule checks without a p-value: the shuffle sanity, domination, the per-unit guards.
- Information items: no correction; pointers, never evidence.

### 8.4 Power and the size rule

- **Planning seed spreads, per claim, from the nearest record** (sd of per-seed differences):
  - A-R1: NET seed sd about 9.4 (step 2: seed SE 4.20, CONN `addon_report/report.md:143`); a perfect switch's lead over
    the record's hinge always is about 29.6, so power is about 1 at 5 seeds unless a false switch occurs.
  - A-K1, A-K3: round 1's small-evidence leads (BCA over split conformal 1.78, over the rank test 4.29, half of that from
    one seed, `ROUND1.md:101-113`): skewed by rare false alarms; expected fail.
  - A-K2: round 1's ladder lead +23.0, +15.7 to +33.1 when any one seed is dropped, one seed at -17.2 (`ROUND1.md:74-96`):
    seed sd about 15-20, so at Holm's first step for two claims (t = 2.776 at 5 seeds) the mean must reach about 19-25.
  - C-K1: 0.1-1.9 (A2 - F0 per seed 0.04-0.33 return units, CONN `arms_abc_f0/RESULTS.md:122-123`; step 2's A1 1.9,
    `addon_report/report.md:185`). C-K2, C-K3: 1.9-6.4 (`report.md:185-187`). C-K4, C-K6, C-R1, C-R1b: 8.8
    (`report.md:161`).
  - At Holm's first step for four claims (level 0.0125, t = 3.495 at 5 seeds), the mean must reach 3.0 at sd 1.9, 8.6 at
    5.5 and 13.8 at 8.8 (`review_stats/marks_arith.out` line 5). With a five-claim family at 5 seeds, a true 10 at sd 5.5
    passes 64% of the time, at 10 seeds 99% (REV section 6).
  - The switch's error rate from seed-worlds alone is weak (10 of 10 correct bounds it at 74%), which is why A-V1 and
    B-V1 use 1,000 replicates per seed-world.
- **Size rule (fixed now).** After the pilot, for every primary and credit claim, compute its power at its own bar (the
  smallest effect of interest) and its own level (Holm's first-step level for a family claim), using the upper 80%
  confidence limit of the pilot's seed sd. Confirm on 5 seeds (25-29); add 30-34 if any primary return claim has power
  below 0.8 at 5 seeds and at least 0.8 at 10. Any claim still below 0.8 at the chosen size is labeled "underpowered"
  in the confirmatory registration, before any confirmatory datum exists. The decision is written to
  `size_decision.json`.

### 8.5 B's release power (a pre-registered discriminability step on the pilot)

Release in N needs N's true harm rate `pi_N` well below alpha_B, while the margin rule places m at break-even. With the
exact gate, release passes iff at least 3 of 5 seeds certify (per-seed values about 61.8 or 0). Pass probabilities at 5
seeds (REV `revise_arith.out` section 4):

| Tries per seed | pi_N = 0.02 | 0.03 | 0.05 | 0.07 |
|---|---|---|---|---|
| 50 | 0.88 | 0.60 | 0.14 | 0.02 |
| 70 | 0.97 | 0.76 | 0.18 | 0.02 |
| 100 | 1.00 | 0.95 | 0.38 | 0.03 |
| 200 | 1.00 | 1.00 | 0.94 | 0.24 |
| 300 | 1.00 | 1.00 | 1.00 | 0.61 |

Per-seed certification of at least 0.9 needs 76 tries at pi_N = 0.02, 103 at 0.03, 154 at 0.04 and 239 at 0.05.

The step, fixed now: on each pilot seed-world in N, compute `pi_N` from the truth population (offline and staged).
Offline: if `P(release passes)` at the pilot's try counts and median `pi_N` is below 0.8, B-R1's release is marked
"underpowered by design" in the confirmatory registration. Staged: N is the smallest of 100, 150, 200 and 300 whose
per-seed certification is at least 0.9 at the pilot's median `pi_N`; if none, N = 300 and B-R2's release is marked
"underpowered by design". The probing cost in S grows with N (about a tenth of the value of caution per 100 probe
episodes) and is reported.

---

## 9. Stages and compute

### 9.1 Training runs per seed (100,000 outer steps each unless stated)

| Runs | S | N | MW+ | MW- | Cost class |
|---|---|---|---|---|---|
| Host (Unifloral TD3+BC) | 1 | 1 | 1 | 1 | host |
| Clone (Unifloral BC) | 1 | 1 | | | light |
| A's gated hinge (k = 1) | 1 | 1 | | | host |
| C-BCA | 1 | 1 | | | with BCA refreshes |
| k = 1 references C-SC, C-RATIO, C-F0, C-SHUF | 4 | 4 | | | with BCA refreshes |
| Matched C-F0_m, C-SC_m, C-RATIO_m, C-SHUF_m: 3-rung ladder + matched run + about 1.5 rounds in S; locked in N | about 22 | 4 | | | with BCA refreshes |
| Matched C-HINGE, C-EVEN, C-SUP | about 16.5 | 3 | | | C-family |
| C-RAND1, C-RAND2: matched on the pilot; locked coefficient on confirmation | about 11 (pilot) or 2 | 2 | | | C-family |
| C-ORACLE (co-trained, x64 step) | 1 | 1 | | | oracle |
| Actor-only, 20,000 steps: C-I2 (C-BCA, C-F0, C-HINGE), C-ORACLE-S1, C-ORACLE-V | 5 | 5 | | | actor-only |
| Once per stage: C-k0 and the second training of the first host (S3) | 2 | | | | |
| **Total per seed** | | | | | **about 88.5 (pilot), 79.5 (confirmatory)** |

Extra matching rounds beyond the expected 1.5 (at most 5 per control) are covered by a 20% allowance. With K3 failed,
the matched controls are not trained (section 9.3).

### 9.2 Compute per stage

Measured speeds: on this container (4 CPUs, no card), Unifloral's TD3+BC step restated on made-up arrays runs 13.8-14.5
ms per outer step with four runs side by side (about 23.5 minutes per 100,000 steps), 16.3-16.8 ms with a BCA-like
per-step fit (about 27.5 minutes), and 14.0-14.1 ms with the lower edge (`bench/conc_*.out`); that benchmark left out
C's real per-step work (the recorder density, the threshold-table lookup, eta and Q1 at the recorded command). On the
laptop's card the hosts measured 0.58 ms alone and 1.26 ms with BCA, and 3.5-5.2 s per threshold table on a 900-row
bank (HOSTS PRESPEC.md:601-604): on the card BCA's work more than doubles the step, while on this CPU it adds 18%. Stage
1's card runs implied about 0.5 minutes of fixed cost per run (HV2 PRESPEC.md:880-915). An evaluation of one policy on
one bank with its capped twin takes about 2 s; one online episode 11-17 ms; `Q^pi` for 1,000 pairs about 9 s
(`bench/rollout_*.out`); one A-BCA decision about 35 ms (`review_feas/bench_wbcp.out`).

**Cost rule (fixed now).** Stage 0 benchmarks every C penalty kind on the card with its full per-step work and the 18
refresh tables. Every C-family run is costed at the larger of its measured rate and 1.26 ms per outer step, plus 0.5
minutes of compilation, plus the tables for runs with BCA refreshes. Card-hours are card wall time (HOSTS `budget.py:8-14`).

| Stage | What | Laptop card for training, CPU for evaluation | CPU only (needs a built training environment, D11) |
|---|---|---|---|
| 0, code tests on made-up data | unit and reproduction tests, the two-environment tests, smoke runs of 2,000 steps, the card benchmark, freeze | about 3-5 card-hours and 20-25 CPU-hours, booked separately (the hosts' smaller stage 0 took 3.8-4.5 and 15-19, HOSTS PRESPEC.md:574-586) | about 20-25 CPU-hours |
| Pilot, main, seeds 0-4 (exploratory) | about 88.5 runs x 5; margins; replicates; online episodes; coverage | about 20.9 card-hours (25.1 with 20%) + about 42 CPU-hours | about 170 (204) + 42 CPU-hours |
| Confirmatory, main, per 5 seeds | the same, frozen, after approval | about 18.9 (22.7) card-hours + about 42 CPU-hours | about 152 (183) + 42 CPU-hours |
| Module W, per 5 seeds (section 10) | about 51.5 runs per seed | about 14.1 (17.0) card-hours + about 15 CPU-hours | about 112 (134) + 15 CPU-hours |
| Trimmed main (decision D14) | C-RATIO_m and C-SHUF_m reported at k = 1 instead of matched | pilot 16.8 (20.1), confirmatory 14.8 (17.8) card-hours | 140 (168), 123 (147) CPU-hours |

(All from REV `budget_arith.out`.) Draft 1's 6-8 card-hours per 5 seeds charged C's runs at the bare host rate, left out
the refresh tables, and had 29 fewer runs per seed; the review's re-costing of draft 1's own run list already gave
7.9-11.3 (`review_feas/card_arith.out`).

Evaluation per seed (CPU, about 8-9 hours): replicate held-out blocks (about 1 hour); A's replicates, shuffles, sweeps
and the size table (about 0.7); the ladder decisions (about 1.4); staged deployments with their replicate probe sets
(about 0.7); online scoring (about 0.4); C-V1 on the refresh bands (about 0.5); C-V3's replicate banks (about 3.2: 200
banks x 7 refreshes x 2 worlds x 2 bands at about 4 s per table); truths and attribution (about 0.5).

**Budget caps (proposed for your approval):** card route: pilot main 26 card-hours + 50 CPU-hours; confirmatory main 24
card-hours + 50 CPU-hours per 5 seeds; module W 18 card-hours + 18 CPU-hours per 5 seeds; stage 0 booked separately.
CPU route: 260, 240 and 155 CPU-hours. Stop rule S10 checks the projected cost against the cap before a stage starts.

**Where and when.** Card runs go one at a time: about 42 card-hours per stage with module W. The laptop card is shared
with the hosts' version 2 restart (HV2 PRESPEC.md:11-16): the placement pilot runs after the hosts' stages, or under a
time-share you set, and laptop evaluation never overlaps card training (keep heavy processor work off the laptop while
card runs go, HOSTS PRESPEC.md:598-604), or evaluation runs in the cloud scoring environment. On the CPU route a 25-29
hour job cannot run as one background command here (two-hour limit), so the driver is resumable per whole run with a
crash-safe ledger (HOSTS `budget.py:21-30`), under an environment signature like the hosts' S9: Unifloral's Dockerfile,
Python 3.10, JAX 0.6.2, numpy 1.26.4, d4rl at its pinned revision, mujoco-py with the MuJoCo 2.1 binaries, plus the
scoring environment with gymnasium 1.0.0.

### 9.3 Order

1. **Stage 0** on made-up data; `freeze_pilot.py`; your approval of stage 0's summary and of the pilot.
2. **Pilot:**
   1. data and counts (commands only: `t_lo`, group sizes, B's conditioned tries, rung tries, cell counts);
   2. hosts, clones and gated hinges (S and N first, then MW+ and MW-); lock;
   3. replicate blocks (scoring environment), then every offline score (one hand-off round trip);
   4. the margin rules (they read the ladder's returns): m, m_f, m_hinge, m_on. The pilot's A and B decisions are
      therefore not blind; they are written to `decisions.json` right after the margins, before any other return;
   5. K1, K2, K4;
   6. C-ORACLE and the actor-only oracles for every seed; K3 pooled;
   7. C's arms, matching, lock (matched controls only if K3 passed);
   8. all online episodes, staged deployments, truths and ladder readouts;
   9. the pilot report, then the declared slots: the size decision, B's discriminability step, the best plain caps, the
      C-RAND coefficients, the budget.
3. **Confirmatory registration**: `freeze_confirm.py` with the filled slots; your approval at your terminal, before any
   confirmatory datum exists.
4. **Confirmatory seeds:** data and counts; hosts, clones and gated hinges, lock; replicate blocks and offline scores;
   `decisions.json` with the frozen margins, **before any return of any policy is read**; then K1 and K2; C-ORACLE and
   the actor-only oracles for every seed, K3 pooled (C-BCA is trained in any case, since its coverage claims stand
   without K3; matched controls only if K3 passes); C's arms, matching, lock; online episodes, staged deployments,
   truths and the ladder; verdicts only when every seed is complete.

### 9.4 Freezes and changes

- `freeze_pilot.py` writes `prespec_pilot.json` and the code manifest, and refuses to run again once any pilot output
  exists (the pattern of step 1, MAIN `outputs/shift/beyond-the-data/restructure/step1/PRESPEC.md:5-8`).
- `freeze_confirm.py` runs only once the pilot report exists, hash-chains to `prespec_pilot.json`, and may fill only the
  declared slots: m, the m_f, m_hinge, m_on; the size decision; staged B's N and the "underpowered by design" labels;
  the best plain cap (W and A-K4); C-RAND's coefficients; the budget caps.
- Any other change after the pilot goes into `CHANGES.md` with its reason and the pilot result that prompted it, and
  needs its own approval fingerprint (the pattern of HOSTS `gate.py:7-22`). Results pages flag every claim whose mark
  changed after the pilot.
