# Confirmation v1, attempt 3 (2026-11-02, 19:08 to 21:49 UTC, Zeus): what the baseline does, what BCA changes, what it cost

**Reading: INCONCLUSIVE**, by the instrument check alone. Every one of the 54 units completed (13 stall3 training
units reused from attempt 2, 17 trained here, 24 evaluations); 9,061 s of training charged against the 13 h cap;
636 s and 1,313,952 exact transitions of evaluation. Record: `cf_run/attempt_3/` (`CF_RESULT.json`, `decision.json`,
one folder per unit), manifest 6a524bf8…, approval d7033a81. Attempts are closed (amendment A3: no attempt 4), and
the rule's own condition for a repeat after INCONCLUSIVE (every evaluation's I1 within 1e-6) is not met, so this
record is the confirmation package's final one.

The labels the rule computes regardless of validity are on record beside the reading: **C1 NOT CONFIRMED**,
**C2 OVER-CORRECTION**. They are not a reading. They are what the fresh seeds showed.

This readout follows the reporting standard set on 2026-11-02: the baseline first, then what changed, which
component of the algorithm was involved, what the evidence explains, what stays a hypothesis, and the smallest test
that would settle each one. Nothing below was run beyond the sealed package; every number is from the saved reports.

## 1. The instrument miss

I1 checks, at the 16 visited states with t = 0, that Q^pi(s, pi(s)) equals the rollout's own return from that start,
to 1e-6 relative. Two of 24 evaluations missed it; I2 (the executed command equals pi(s) to 1e-6 per component)
held everywhere.

| Unit | I1 max relative error | I2 max abs |
|---|---|---|
| eval_native_s2_stall3 | 1.86e-6 | 3.0e-7 |
| eval_bca_s0_k21_stall3 | 1.11e-6 | 6.6e-7 |
| the other 22 | 2.5e-8 to 3.9e-7 | 1.2e-7 to 5.1e-7 |

**What is established:** the two misses are 1.1 and 1.9 times the bar; the identity is not broken at any scale a
return read could see (the returns are hundreds of units; 2e-6 relative is about 7e-4 of one unit). The reads of
these two units are therefore numerically sound, and the labels computed from them are not in doubt on this account.

**What is a hypothesis:** the exploratory package's amendment A1 attributed its own I1 misses (at the 1e-9 bar) to
the float32 JAX actor on the GPU giving a last-place-different command for a batch of 128 (the rollout) and a batch of
48 (the probes). That diagnosis was never verified by a controlled comparison; it was a reading of where the
discrepancy sat. The same cause is plausible here (both misses are in the stall3 world, whose rollouts are long and
whose returns are large, so a last-place command difference compounds further), but it is not established.

**Smallest test (H1):** on the two failing units' saved checkpoints, recompute the I1 identity twice, with no
training: once on the CPU in float64, and once on the GPU with the probe batch padded to the rollout's 128. If the
discrepancy falls below 1e-9 on the CPU and below the bar with the matched batch, the cause is established. It is
an evaluation-only unit of a few seconds, user-run (it reads checkpoints), under its own one-line approval. If the
CPU recomputation still misses, the cause is elsewhere (the dynamics step, not the actor) and the instrument needs a
different design, not a wider bar.

## 2. The baseline first: the native TD3+BC actor at 20k on the CUTOFF data

Three fresh training seeds; the same CUTOFF datasets as the exploratory run (stall3 c2a6d616…, no-stall e3549869…),
which withhold the past-knee region W (|a0| > 0.8) from the training rows. Per seed 0 / 1 / 2:

| world | J (128 starts) | W exposure | e_own | e_data | e_W | U_own | U_data | U_W |
|---|---|---|---|---|---|---|---|---|
| no-stall | −64.5 / −61.4 / −62.6 | 0.80 / 0.83 / 0.80 | +8.2 / +6.5 / +3.7 | +2.4 / +2.3 / +1.0 | +2.4 / +2.2 / +0.9 | 2.27 / 2.63 / 2.39 | 2.27 / 2.64 / 2.45 | 2.29 / 2.67 / 2.48 |
| stall3 | −364.6 / −368.6 / −365.2 | 0.68 / 0.76 / 0.70 | +188.9 / +187.1 / +189.2 | +17.3 / +20.7 / +18.5 | +18.3 / +21.5 / +19.2 | 3.30 / 3.38 / 2.79 | 2.66 / 2.57 / 2.29 | 2.68 / 2.62 / 2.32 |

(e_own: Q1 minus the exact continuation return at the actor's own actions on the visited states; e_data and e_W: the
same at the 192 probe rows, at the logged action and with a0 replaced by the W magnitudes 0.85 / 0.9 / 1.0; U: BCA's
width at the same points, read from the same critics for the comparison.)

**What the baseline does well.** In the no-stall world it is a good controller: it uses W for 80 to 83 percent of
its steps and returns about −62, and its critic is close to honest (own-action over-valuation +4 to +8; at the probe
rows +1 to +2, and no larger at W than at the logged action: e_W − e_data is −0.1 to 0.0). TD3+BC's two parts,
the Bellman-fitted critic and the actor that maximizes Q1 under a behavior-cloning pull, do what they are built to do
when the region the actor walks into behaves as the data suggest.

**Where it falls short, and which part.** In the stall3 world the same actor does the same thing (W exposure 0.68 to
0.76; the policies are trained on data that never enter W, so nothing in training differs by world) and the return
is 300 units worse. The critic over-values its own actions by +187 to +189 and the probe rows by +17 to +21 (+18 to
+22 at W). The component is the critic's extrapolation across the knee: the Bellman fit on retained rows constrains
Q only where rows exist, and past 0.8 it continues the smooth world's trend. The actor's gradient ascent on Q1 then
walks through the knee, and the BC term does not stop it, because the logged actions reach the knee and the pull is a
distance in action space, not a support check. This is the honeypot: the data cannot tell the critic that W stalls,
and TD3+BC has no component whose job is to notice that W is outside what the data can speak for.

**What the baseline is lacking, stated exactly.** Not "the baseline cannot handle OOD". It lacks any term that
distinguishes an action the data support from one they do not, so its decision at the knee is set entirely by the
critic's extrapolation, which is right in one world and wrong by 300 units in the other. Whether anything offline can
do better is the design check in section 6.

## 3. What BCA changes, by component

BCA's actor maximizes Q1 − kappa·U, U the uncertainty width. Three doses on the same forks (10k warmup shared with
the native branch, same minibatch stream). Paired per seed, BCA minus native (dJ return, dEXPO W exposure, de_own
own-action over-valuation):

| world | kappa | dJ | dEXPO | de_own | BCA exposure |
|---|---|---|---|---|---|
| stall3 | 21 | +7.4 / −8.5 / +19.7 | −0.05 / −0.18 / −0.14 | −13 / −5 / −28 | 0.63 / 0.58 / 0.56 |
| stall3 | 63 | +3.9 / +42.5 / −27.0 | −0.12 / −0.49 / −0.27 | −16 / −52 / −13 | 0.57 / 0.26 / 0.44 |
| stall3 | 189 | +11.2 / +121.5 / +9.8 | −0.17 / −0.72 / −0.33 | −31 / −110 / −63 | 0.52 / 0.04 / 0.38 |
| no-stall | 21 | +5.9 / −4.1 / −1.0 | −0.07 / −0.15 / −0.30 | −3 / +3 / +3 | 0.73 / 0.68 / 0.50 |
| no-stall | 63 | −0.7 / −39.6 / −9.9 | −0.13 / −0.72 / −0.61 | −1 / +23 / +10 | 0.67 / 0.11 / 0.19 |
| no-stall | 189 | −41.4 / −98.3 / −20.5 | −0.24 / −0.70 / −0.58 | +15 / +41 / +15 | 0.56 / 0.13 / 0.22 |

**The width U (component: uncertainty estimator).** U does not separate W from the data. At the probe rows, U at the
W magnitudes exceeds U at the logged action by 0.02 to 0.05 in both worlds (U_W − U_data), against a level of 2.3
to 3.4. U is a width of the critic's TD residuals on the retained rows, and the retained rows are the same kind of
rows in both worlds, so this is what the construction predicts: **U is world-blind**, and here it is also nearly
support-blind at the probe rows. This is established by the reads, in both runs now.

**The penalty kappa·U (component: penalty mapping).** At kappa 189 the penalized score at the probe rows sits about
550 below Q1 (p_data), a near-constant offset: the level of U does not change which action wins. What moves the actor
is U's gradient along the actor's own trajectory, where U is higher than at the probe rows (U_own 2.8 to 3.4 against
U_W 2.3 to 2.7 in stall3). So the pull acts on the actions the actor actually visits, not through a judgment of W as
such.

**The actor's decision (component: actor update).** The pull is in the same direction and of similar size in both
worlds at every dose (dEXPO at kappa 63: −0.12 / −0.49 / −0.27 against −0.13 / −0.72 / −0.61), and it generally
grows with kappa, though not monotonically in every seed (`profile.exposure_monotone` false: no-stall seed 1 is
0.68, 0.11, 0.13 and seed 2 is 0.50, 0.19, 0.22 across the doses). The actor leaves W because kappa·U's gradient
points away from where it was going, whatever W is worth.

**The return (the outcome).** The sign of dJ is set by the world, not by BCA. Where W stalls, leaving it earns return
in 7 of 9 seed-doses; where W is good, leaving it costs return in 8 of 9, by up to 98 units. The two stall3 losses
(seed 1 at kappa 21, seed 2 at kappa 63) and the exposure non-monotonicity say that the action BCA retreats to
matters as much as leaving W, and U says nothing about that action's quality either.

## 4. What the evidence explains, and what it does not

**Explained (established by the paired reads).**

1. The exploratory PROMISING reading at kappa 63 did not replicate on fresh seeds: stall3 dJ +3.9 / +42.5 / −27.0
   against +11.9 / +41.3 / +22.8 before. One of three fresh seeds lost 27 units.
2. The exploratory placebo cost, −0.3 / +3.4 / −6.7, was the small end of a wide distribution. On fresh seeds the same
   dose cost −0.7 / −39.6 / −9.9. The cost tracks how far exposure falls: the seeds whose exposure collapsed (0.83
   to 0.11, 0.80 to 0.19) paid 40 and 10 units; the seed whose exposure barely moved (0.80 to 0.67) paid 0.7.
3. Kappa 189 raised the stall3 return in all three seeds (+11.2 / +121.5 / +9.8). This is an observed finding at a
   dose the rule did not confirm, and it comes with no-stall costs of −41.4 / −98.3 / −20.5 on the same seeds. The
   gain and the cost are the same mechanism read in two worlds.
4. The ordering of effects follows from U being world-blind: the pull out of W is a property of BCA alone, and the
   bed decides whether that pull is worth having. BCA, as built, cannot be selective between these two worlds,
   because the only thing that differs between them is outside the data it reads.

**Not explained (hypotheses, each with its smallest test).**

- **H2, what drives the pull.** The pull out of W is driven by U's dependence on the states and actions the actor
  visits (U_own > U_W), not by U flagging W as unsupported at dataset states (U_W − U_data ≈ +0.03). Test, no
  training: on the saved per-row U arrays of the 24 reports, correlate U with a plain support-distance score (the
  retained-support cutoff's own distance, or a k-nearest-neighbour distance to the retained rows) at the probe rows
  and at the visited states. If U is uncorrelated with support distance at the probe rows but rises along the
  trajectory, H2 holds, and the penalty is acting as a trajectory regularizer rather than a support penalty. This
  reads reports only.
- **H3, why fresh seeds collapsed exposure where the exploratory seeds did not.** At kappa 63 in no-stall, fresh
  seeds 1 and 2 reached exposure 0.11 and 0.19; the exploratory seeds stayed at 0.53 to 0.73. U_W is 0.0 to 0.7
  higher on the fresh seeds (2.3 to 2.7 against 1.9 to 2.3), which at kappa 63 is up to 44 units of penalty.
  Hypothesis: the collapse is set by the seed's U level, so the effective dose varies by seed at fixed kappa. Test,
  no training: regress dEXPO on kappa·U_own across the 18 BCA units; if the fit is tight across worlds and seeds,
  the dose should be set in units of kappa·U, not kappa.
- **H4, whether the retreat action's quality explains the two stall3 losses.** Seed 2 at kappa 63 left W (0.70 to
  0.44) and lost 27 units; at kappa 189 it left further (0.38) and gained 10. Hypothesis: the retreat landed in a
  region the critic values worse (e_own fell only 13 at kappa 63 against 63 at kappa 189). Test, no training: at
  seed 2's visited states, compare the exact continuation return of the native command, the kappa-63 command and the
  kappa-189 command; the reports hold the first, and the other two are one evaluation unit each.
- **H1**, section 1.

## 5. The dose profile, reported as the rule asks

`exposure_monotone` false, `honeypot_gain_monotone` false (stall3 dJ by dose: seed 0 +7.4, +3.9, +11.2; seed 1
−8.5, +42.5, +121.5; seed 2 +19.7, −27.0, +9.8). The best-looking dose on this run, kappa 189 in stall3, is not a
confirmed dose, and in the placebo world it is the most expensive. No dose on this run gives the gain without the
cost.

## 6. The design check this forces

If the two worlds' retained data are indistinguishable (same rows in support, same dynamics everywhere the rows
reach), then no score computed from those data alone can know which world it is in, and any offline-only penalty
must pay in one world what it earns in the other. Whether the two CUTOFF datasets are indistinguishable in this sense
has not been checked; they differ by digest, and the prefix rule only guarantees the anchors agree. This is the
first thing to settle before any further BCA design on this bed, because it decides whether selectivity is a
property BCA can have here at all, or whether the right claim is the trade-off in section 4 stated honestly. Test,
no training: compare the two datasets row by row (states, actions, rewards, next states) on their retained rows; if
every row agrees to float32, the worlds are indistinguishable offline and the honeypot measures a trade, not a
detection.

## 7. What stays open after this package

The confirmation package is closed, not the question. What is on record: the baseline's failure is the critic's
extrapolation across the knee, exactly and by 300 units; BCA's pull out of W is real, dose-dependent and
world-blind; its return effect is a trade whose sign the bed sets. What comes next is sections 4 and 6, in the order
H1, the dataset check, H2, H3, H4: four reads of saved reports and two evaluation-only units, before any training
run is designed. Each new run needs its own rule and the user's approval, as before.

## Ledger entry

Confirmation v1, attempt 3 (Zeus): INCONCLUSIVE (I1 at 1.86e-6 and 1.11e-6 on two of 24 evaluations, bar 1e-6);
computed labels C1 NOT CONFIRMED (stall3 kappa 63 dJ +3.9 / +42.5 / −27.0), C2 OVER-CORRECTION (no-stall kappa 63 dJ
−0.7 / −39.6 / −9.9, dEXPO −0.13 / −0.72 / −0.61); profile not monotone in either flag; kappa 189 stall3 dJ +11.2 /
+121.5 / +9.8 against no-stall −41.4 / −98.3 / −20.5; 9,061 s training, 636 s and 1,313,952 transitions of
evaluation. Attempts closed. Readout: this file.
