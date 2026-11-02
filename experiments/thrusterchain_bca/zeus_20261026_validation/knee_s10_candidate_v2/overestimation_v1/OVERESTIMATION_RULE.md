# OOD-action overestimation comparison, native versus BCA, on the knee_s10_v2 CUTOFF data (written before any number exists)

Written 2026-11-01, after the native screen ended INCONCLUSIVE at G1 (`../native_screen/SCREEN_READOUT_20261101.md`):
the host critic cannot resolve the 2-unit value departure at the knee even with exact labels, so no gate built on that
departure can be read on this host. This package asks the question the offline RL literature asks instead, with the
pieces already built and sealed. It is exploratory and supports no protocol claim. One sealed package, one approval,
one run.

## The question

On a fixed offline dataset with a withheld action region, does the BCA actor (Unifloral TD3+BC host, actor score
Q1 − kappa·U, kappa = 63) choose actions on which the critic is less wrong than the native actor's, without losing
return? This is the extrapolation-error question of Fujimoto, Meger and Precup (2019) and the Q-versus-rollout
comparison of Kumar et al. (2019, 2020), An et al. (2021) and Bai et al. (2022): "OOD" means outside the behavior
data's action support, "truth" means the value of the learned policy itself, and the claim is less overestimation on
the policy's own actions with return not worse. Here the truth is exact (float64 restore of the state, the policy's
own deployed continuation, no Monte Carlo), which is stricter than the literature's rollout estimate.

| Claim | What is measured | Reference |
|---|---|---|
| The TD critic over-values withheld actions | critic error e = Q1 − Q^pi at dataset states, on the logged (in-support) action and on three withheld a0 magnitudes | Q^pi: the frozen actor's own unclipped continuation from the exact state, gamma 0.98, to step 99 |
| BCA's actor picks actions the critic is less wrong about | e on the actor's own action at 48 visited states of its own rollouts, native versus BCA, paired per seed | same |
| BCA does not lose return | mean discounted return over 128 common starts, paired per seed | the rollouts themselves |
| The width flags the withheld region | U at the same dataset probes, logged versus withheld, both arms | reported |

## Decisions fixed here

| Decision | Value | Why |
|---|---|---|
| Host | Unifloral TD3+BC, pinned revision f2dc1278, through `../native_screen/screen_worker.py` (Stage 1 warmup, branch, dose-screen BCA branch) | the user's rule: Unifloral base only |
| Data | the formal G2 v2 stall3 dataset by digest c2a6d616…, CUTOFF arm (fitting rows with \|a0\| ≤ KNEE = float32(0.8); W = \|a0\| > KNEE withheld), read by the authenticated loader (native screen change 7) | the one dataset whose W region is known exactly |
| Worlds | stall3, the honeypot (W is a trap: thrust stalls past the knee), and no-stall, the placebo (amendment A2: the same CUTOFF data by digest e3549869…, W is harmless) | a penalty that keeps the actor out of W only where W is a trap is calibrated; one that keeps it out in both worlds and loses return in the harmless one over-corrects (the underestimation failure) |
| Dose | kappa = 63, one dose | the dose screen's leverage point; configured 3 showed none |
| Seeds | the native screen's three training rows and data-plan seeds (`../native_screen/screen_seeds.json`), never used for training (the screen never trained); two new seeds of this package for the visited states and the dataset probe rows (`oe_seeds.json`, audited) | fresh, audited, no confirmation seed touched |
| Training | 10k native warmup, then native branch 10k→20k and BCA branch 10k→20k from the same fork, paired (same minibatch stream, same warmup) | Stage 1's protocol, unchanged |
| Policies read | the 20k checkpoint of each branch | the host's update budget |
| Starts | the formal G2 v2's 128 E2 starts (`bank_states("stage1_e2_bank")`, native screen change 4) | sealed, common to both arms |
| Visited states | Stage 1's rule: 16 episodes at each of t = 0, 25, 50 (`stage1_eval.visited_states`, this package's seed) | 48 own-action probes per policy |
| Dataset probes | 192 CUTOFF-retained fitting rows drawn without replacement by this package's seed; at each, the logged action and three withheld variants: a0 replaced by sign(a0)·{s05, s10, s20} = {0.85, 0.9, 1.0} (float32 slot values), a1..a3 logged | in-support reference and W probes at the same states |
| Reference | `qpi.raw_values_from_states(..., clip_all=False)`: first action forced, then the frozen actor's own deployed command to step 99, exact dynamics, gamma 0.98, no bootstrap | the policy's own value, as deployed |

## Units, in order (fail-fast)

In stall3 (ids unsuffixed), then in no-stall (ids suffixed `_nostall`; amendment A2):

1. `warmup_s{i}`, i = 0, 1, 2 (kind `warmup`, cutoff; GPU, ≤ 1,800 s each)
2. `branch_s{i}` (kind `branch`, native 10k→20k)
3. `bca_branch_s{i}` (kind `bca_branch`, kappa 63, paired with `branch_s{i}`)
4. `eval_native_s{i}`, `eval_bca_s{i}` (kind `oe_evaluate`: rollouts, own-action probes, dataset probes in that world;
   ≤ 900 s and 200,000 exact transitions each; expected about 94,400: 12,800 rollout steps, ≤ 4,800 own-action
   continuation steps, ≤ 76,800 dataset-probe continuation steps)

then the decision (`oe_math.decide`), written to `decision.json`; `OE_RESULT.json` always written.

Eighteen training jobs (about 8 minutes each on this machine), twelve evaluations (about 20 s each). The no-stall
units use the same seed rows, the same 128 starts, the same visited-state rule and the same probe-row seed; the
no-stall dataset is the stall3 dataset's twin (same reset seeds and recorder draws, different physics), so its
retained rows differ slightly and the 192 probe rows are drawn from them by the same seed.

## Reads, per unit

- J: mean discounted return over the 128 starts.
- EXPO: share of the 12,800 executed commands with \|a0\| > KNEE (`stage1_eval.step_rates` `w_any_depth`).
- e_own: mean over the 48 visited states of Q1(s, pi(s)) − Q^pi(s, pi(s)); also the share with Q1 > Q^pi.
- e_data: mean over the 192 rows of Q1 − Q^pi at the logged action. e_W: the mean over the 192 × 3 withheld probes,
  and per magnitude.
- U_data, U_W: the calibrated width at the same probes (both arms; the native arm trains the same auxiliary).
- For the BCA arm: p_data, p_W = mean of (Q1 − kappa·U) − Q^pi, the penalized score's error.
- I1 instrument: at the 16 visited states with t = 0, Q^pi(s, pi(s)) equals the rollout's own return from that
  start to 1e-6 relative (the same deterministic policy and dynamics; amendment A1, was 1e-9). I2: pi(s) at every
  visited state equals the command the rollout executed there to 1e-6 in every component (amendment A1, was bit for
  bit). Both values are recorded. Every quantity finite.

## Paired differences and readings

Per seed i and world, BCA minus native: dJ_i, de_own_i, dEXPO_i. Validity first (INCOMPLETE > INCONCLUSIVE > NOT
REACHED, the last on stall3's native exposure), then the reading is the pair "honeypot / placebo": the stall3
labels below, and the no-stall labels of amendment A2. Every label computed is reported.

| Label | When |
|---|---|
| INCOMPLETE | a unit failed, a cap bound, or the integrity check refused |
| INCONCLUSIVE | I1 or I2 fails in any unit, or any non-finite read |
| NOT REACHED | native EXPO < 0.01 in every seed: the native actor never leaves the support, so on-policy over-valuation of W cannot matter here; the dataset-probe reads are still reported |
| PROMISING | de_own_i ≤ −0.5 in every seed and dJ_i ≥ −2.0 in every seed |
| HARMFUL | dJ_i ≤ −2.0 in two or more seeds |
| NO EFFECT | \|de_own_i\| < 0.5 and \|dJ_i\| < 2.0 in every seed |
| MIXED | anything else |

No-stall (the placebo), per seed dJ_i and dEXPO_i:

| Label | When |
|---|---|
| PLACEBO WITHOUT STAKE | native EXPO < 0.01 in every seed: the native actor does not use W where it is harmless, so avoiding W costs nothing and over-correction cannot show |
| OVER-CORRECTION | dEXPO_i ≤ −0.05 in two or more seeds and dJ_i ≤ −2.0 in two or more seeds: BCA leaves the harmless region and pays for it (the underestimation failure of a sign-blind penalty) |
| CALIBRATED | dJ_i ≥ −2.0 in every seed: no return loss where W is harmless |
| PLACEBO MIXED | anything else |

The 0.5, 2.0 and 0.05 are fixed here before any number exists. They are modest against G1's in-support resolution band
(1.84 return units for one critic's contrast error); paired arms from one fork share the warmup, the minibatch
stream and most of the critic, so their differences are expected to be less noisy than that band, but how much less
is unknown and three seeds cannot tell. A PROMISING reading is a reason to design a confirmation with fresh seeds;
it is not a result. Reported without a bar, as the literature's plot: native e_W − e_data per seed (does the TD
critic over-value withheld actions at all, relative to in-support ones?) and U_W − U_data (does the width flag them?).

## Kill rules and caps

Training: 1,800 s per job, 3.5 h in all (eighteen jobs at about 8 minutes; a reused unit is charged its original
time). Evaluation: 900 s and 200,000 exact transitions per job, 1.5 h and 1,500,000 in all (the `SimulatorCounter` counts every dynamics call; the authenticated loader performs none). A
failed job ends the attempt INCOMPLETE. No bar, dose, slot or seed changes after the approval.

## Attempts

One attempt. After INCOMPLETE, a second attempt under a new approval naming attempt 2 reuses every unit an earlier
attempt completed (same id, kind, seed, arm; the evaluations always rerun) and runs the rest. Amendment A1: after
INCONCLUSIVE too, only when every evaluation of the previous attempt recorded its I1 identity within 1e-6 relative
(the named instrument cause below); the supervisor checks the saved reports. No attempt 3. `--decide` recomputes
the decision from the saved reports.

Amendment A2 runs in its own folder, `oe_run_a2/`, with its own attempt count and its own approval sentence,
because the honeypot comparison under A1 is decided (`oe_run/attempt_2`, PROMISING) and a decided comparison takes
no further attempt. A2's supervisor reuses, by path, the completed stall3 training units of `oe_run/attempt_1` and
`oe_run/attempt_2` (same id, kind, seed, arm, world) and reruns every evaluation.

## What the user's approval states

`OE_APPROVAL.json`, written at the user's terminal, must contain these sentences, the manifest's sha256 and the attempt:

1. "I approve the OOD-action overestimation comparison on the knee_s10_v2 CUTOFF data, Unifloral TD3+BC host, kappa 63, with the no-stall placebo world (amendment A2), attempt {n}" (the A1 run's sentence ended "kappa 63, attempt {n}")
2. "the BCA arm at kappa 63 is approved"
3. "the comparison is exploratory and supports no protocol claim"

## Integrity

`OE_MANIFEST.json` hashes this package and every file of the native screen's manifest (its package, collection
record, the formal G2 package and record, the prefix rule, v1's pi_ref and recorder, the g1 modules, r4), and pins
knee_edge_ood_v3's A4 and exploratory manifests. Every job's record is the v3 dependency closure plus those files
plus this package; the worker refuses on any change before and after its work, and on any importable file added to
this folder or the native screen's.

## What this can and cannot say

It can say whether, on this data and host at this dose, the BCA actor's own actions carry less critic error than
the native actor's and whether return moved, in three paired seeds; whether the TD critic over-values withheld
actions at dataset states and whether U flags them; and (A2) whether the penalty's avoidance of W is specific to
the world where W is a trap or is paid for where W is harmless. It cannot say anything about the knee departure (G1 showed the
critic cannot resolve it), about distribution shift in states, about Bayesian uniqueness, or about a protocol-level
claim; a null here is "no effect at this dose and budget", not "BCA cannot work".

## Amendments

| id | Date | What changed, and why | What did not change |
|---|---|---|---|
| A2 | 2026-11-01 21:02 EST, before attempt 2 ran (the user: the experiment is to use the honeypot-edge and underestimation stress-test setup) | The placebo world: the same comparison in the no-stall world, where the withheld region is harmless, as the underestimation stress test; the labels OVER-CORRECTION, CALIBRATED, PLACEBO WITHOUT STAKE, PLACEBO MIXED; the reading becomes the pair "honeypot / placebo". The honeypot half is attempt 1's design unchanged. | the data (the no-stall twin is the formal G2 v2's, by digest), the host, the dose, the seeds, every stall3 read, bar and label, the caps per job. Attempt 1's nine stall3 training units are reused. |
| A1 | 2026-11-01, after attempt 1 (every unit completed; reading INCONCLUSIVE by the instrument checks alone) | I1's tolerance 1e-9 to 1e-6 relative; I2 from bit equality to 1e-6 per component; both values recorded; attempt 2 allowed after this INCONCLUSIVE. Cause: the actor is a float32 JAX network on the GPU, and its output differs in the last float32 place between a batch of 128 (the rollout) and a batch of 48 (the probes) or the shrinking active batch of the continuation; the identity held to 1.1e-7 to 2.4e-7 relative in all six units. The 1e-9 bar was set on the CPU float64 identity (`test_oe.Measure`) and never saw a GPU actor. The probe is self-consistent (Q1 and the reference use the same recomputed action); only the comparison against the rollout's command is affected. | the data, the host, the dose, the seeds, the units, every read, every bar, every reading. Attempt 1's nine training units are reused as they are. |

## Ledger entry

Candidate knee_s10_v2 data, overestimation comparison v1 at kappa 63: the reading, the three paired dJ and de_own,
native EXPO, native e_W − e_data and U_W − U_data, GPU hours and exact transitions.
