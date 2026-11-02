# Overestimation comparison, attempt 1 (2026-11-01, 18:39 to 20:00 EST): readout

**Reading: INCONCLUSIVE, by the instrument tolerances alone.** Every unit completed: nine training jobs (4,475 s GPU,
about 8 minutes each) and six evaluations (143 s, 335,448 exact transitions). The computed reading, reported as the
rule requires, is PROMISING in every seed, by wide margins. The headline is INCONCLUSIVE because I1 and I2 failed
in all six evaluation units, and the rule's precedence puts an instrument miss above any reading.

Record: `oe_run/attempt_1/` (manifest 9834a461…, the user's approval naming attempt 1).

## Why the instrument checks failed

I1 asked that the own-action value at a t = 0 visited state equal the rollout's return from that start to 1e-9
relative. It held to 1.1e-7 to 2.4e-7 relative in all six units. I2 asked that pi(s) recomputed at the 48 probe
states equal the rollout's executed command bit for bit; it did not. Both have the same cause: the actor is a float32
JAX network on the GPU, and its output differs in the last float32 place between a batch of 128 (the rollout), a
batch of 48 (the probes) and the shrinking active batch of the exact continuation. The 1e-9 bar was calibrated on the
CPU float64 identity with a batch-invariant numpy actor (`test_oe.Measure`) and never saw a GPU actor. The probe
itself is self-consistent: Q1 and the reference use the same recomputed action. Nothing in the reads is affected at
any digit that matters (the reads below are differences of tens of units). Amendment A1 sets I1 to 1e-6 relative
and I2 to 1e-6 per component, records both values, and allows attempt 2 after this INCONCLUSIVE; the nine training
units are reused and the six evaluations rerun (about three minutes).

## The numbers (20k policies, stall3, CUTOFF data)

| seed | arm | J (128 starts) | W exposure | e_own (48 visited) | e_data (logged) | e_W (withheld) | U_data | U_W | U_own |
|---|---|---|---|---|---|---|---|---|---|
| 0 | native | −363.8 | 0.696 | +186.4 | +24.8 | +25.2 | 2.55 | 2.57 | 3.07 |
| 0 | BCA | −351.8 | 0.700 | +185.8 | +20.6 | +21.4 | 2.65 | 2.68 | 3.20 |
| 1 | native | −358.8 | 0.684 | +178.8 | +23.5 | +24.4 | 2.19 | 2.21 | 2.71 |
| 1 | BCA | −317.5 | 0.623 | +162.6 | +19.7 | +20.4 | 2.26 | 2.27 | 2.72 |
| 2 | native | −359.3 | 0.672 | +183.2 | +23.7 | +24.6 | 2.49 | 2.51 | 2.95 |
| 2 | BCA | −336.5 | 0.433 | +154.3 | +19.9 | +22.0 | 2.30 | 2.32 | 2.70 |

Paired, BCA minus native:

| seed | dJ | de_own | dEXPO | de_W |
|---|---|---|---|---|
| 0 | +11.9 | −0.6 | +0.004 | −3.8 |
| 1 | +41.3 | −16.2 | −0.062 | −4.0 |
| 2 | +22.8 | −28.9 | −0.239 | −2.6 |

Bars: PROMISING needs de_own ≤ −0.5 and dJ ≥ −2.0 in every seed. Both hold in every seed. No HARMFUL, no NO EFFECT.

## What the numbers say

1. **The native TD3+BC actor exploits the critic's extrapolation error, hard.** At 20k updates its median own
   action has |a0| = 0.993, on the negative side (R_deep− rate 0.65, W+ 0.0): it saturates thruster 0 at −1, deep
   in the withheld region, for 68 to 70 percent of all steps. At those visited states the critic values its own
   action at a median of −80 while the exact value is −305: over-valuation of about 180 return units, on every one
   of the 48 probes in every unit (share over-valued 1.00). This is the Fujimoto 2019 picture, at the magnitude the
   literature reports, on this host and data.
2. **The error is carried by state shift more than by the action alone.** At dataset states the critic is close on
   logged actions (median Q1 −26 against exact −27; mean error +24 is skew from a tail) and only slightly worse on
   withheld actions at the same states (e_W − e_data = +0.4 to +0.9). The 180-unit error appears where the policy has
   driven the state away from the data. The user's earlier caution that state shift would confound an action-only
   story is borne out: on this bed the two cannot be separated, and the on-policy error is the state-shifted one.
3. **BCA at kappa 63 helps, in all three seeds, with return as the outcome.** +12, +41 and +23 return units on a
   base of about −360. Own-action over-valuation falls in every seed (barely in seed 0, by 16 and 29 in seeds 1 and
   2); withheld exposure falls in two seeds (0.68 to 0.62; 0.67 to 0.43) and is flat in seed 0. The policies remain
   poor in absolute terms: pi_ref's values on dataset states are around −20 to −60, so both arms are far from the
   behavior quality. "Less bad" is the honest description.
4. **The width does not flag the withheld action.** U at dataset states is 2.2 to 2.65 whether the action is logged
   or withheld (difference 0.01 to 0.03), and about 0.5 higher at the policy's own visited states. kappa·U is
   therefore close to a constant offset of about 160 in the actor's score (p_data about −135: the penalized score
   under-values everything by that much). Whatever BCA did here, it did not do it by recognizing withheld actions
   through U. The small state-dependence of U (higher off-data) and the penalty's gradient through the action are
   the candidates; this package cannot tell them apart.

## What this does not say

Three seeds; one dose; one dataset; one host budget; policies far below the behavior quality; an exploratory
package with modest pre-registered bars and no confirmation seeds. It supports no protocol claim, says nothing about
the knee departure (G1 showed the critic cannot resolve it), and does not establish the mechanism. A PROMISING
reading on attempt 2 is a reason to design a confirmation with fresh seeds and the mechanism question (U's
state-dependence versus the penalty gradient) made explicit; it is not a result.

## Ledger entry

Overestimation comparison v1, attempt 1: INCONCLUSIVE (I1 1.1e-7 to 2.4e-7 against 1e-9; I2 not bit-equal; GPU
float32 batch effect); computed PROMISING in every seed (dJ +11.9, +41.3, +22.8; de_own −0.6, −16.2, −28.9); native
W exposure 0.67 to 0.70, own-action over-valuation about +180; e_W − e_data +0.4 to +0.9; U_W − U_data +0.01 to
+0.03. GPU 1.24 h; 335,448 exact transitions.

## Post-hoc diagnostic (2026-11-01 20:35 EST; not pre-registered, computed from the saved attempt-1 reports)

Prompted by an outside review's "dose selectivity" question (does the uncertainty concentrate where the critic
over-values?), per unit over the 768 dataset probes (192 rows × 4 slots):

| unit | U given over-valued | U given under-valued | ratio | corr(e, U) | corr(\|e\|, U) | under-valued share | mean error when under |
|---|---|---|---|---|---|---|---|
| native s0 | 2.593 | 2.508 | 1.03 | +0.42 | +0.42 | 0.32 | −3.6 |
| native s1 | 2.250 | 2.130 | 1.06 | +0.49 | +0.47 | 0.40 | −4.3 |
| native s2 | 2.516 | 2.494 | 1.01 | +0.26 | +0.26 | 0.36 | −4.0 |
| BCA s0 | 2.714 | 2.614 | 1.04 | +0.41 | +0.41 | 0.45 | −4.1 |
| BCA s1 | 2.320 | 2.205 | 1.05 | +0.45 | +0.43 | 0.46 | −4.0 |
| BCA s2 | 2.322 | 2.288 | 1.02 | +0.31 | +0.32 | 0.32 | −3.4 |

Reading, which refines point 4 above: U does carry the error's magnitude (correlation 0.26 to 0.49 with the error,
the same with its absolute value) but not its sign (the ratio of U on over- to under-valued probes is 1.01 to 1.06).
The width is a symmetric scale of the critic's miss, as its calibration to |target − Q| implies, so the penalty
kappa·U falls on under-valued probes (about a third of dataset probes, mean miss about −4) as it does on over-valued
ones. The earlier sentence "the width does not flag the withheld action" stands as written: U_W − U_data is 0.01 to
0.03. This table says, in addition, that U is informative about how wrong the critic is and uninformative about
which way. Not pre-registered; reported only.

## Attempt 2 under amendment A1 (2026-11-01 20:37 to 20:40 EST, manifest 64d331e4…): PROMISING

The user ran attempt 2 with the amended tolerances. The nine training units were reused from attempt 1; the six
evaluations reran (142 s, 335,448 exact transitions). I1 held exactly (0.0 relative) and I2 to 1.8e-7 to 5.7e-7 per
component against 1e-6. Every paired read equals attempt 1's to the printed digits (dJ +11.94, +41.30, +22.79;
de_own −0.61, −16.23, −28.91). **The honeypot comparison is decided: PROMISING in every seed**, with the caveats of
"What this does not say" above. Record: `oe_run/attempt_2/`. The no-stall placebo world (amendment A2) runs as its
own comparison in `oe_run_a2/`.
