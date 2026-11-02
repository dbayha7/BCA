# OOD definition v4, revision 5 (DRAFT r5, not approved): the native screen, redesigned after the audit

Written 2026-11-01 after `DESIGN_AUDIT_r4_20261101.md`. This file revises `REVISION_r4.md`; r3 (`DRAFT_DEFINITION.md`)
and r4 stand except where this file changes them. r5 changes no sealed G2 check, so candidate knee_s10_v2's formal
G2 PASS stands. Nothing here is approved or run; the user adopts r5 in their own approval words, candidate by
candidate.

## 1. What r5 changes, and why (the audit's defects D1 to D5, D7, D8)

| # | r4 | r5 | Why |
|---|---|---|---|
| D1 | G1 decides at an absolute bar (0.15 of the exact contrast) at the host's TD budget of 20,000 updates, and a miss stops the screen | G1 is a **reported calibration unit**. It fits to a loss plateau (not a borrowed budget), reads the departure at s10 **relative to the fit's own in-support error** at the matched distance inside the knee, and never stops the screen | At 20k the supervised regression is 10 to 50 times coarser than the TD critic in support (audit E2); it measured convergence, not representation |
| D2 | Twin learnability (V0(f), L1n, BAL) is validity: a miss ends the screen INCONCLUSIVE with no BCA reading | V0(f), L1n, BAL and O2 are **findings about withholding**, reported with their labels, never a stop | The BCA question needs the critic to be wrong (O1) and the actor to go there (R1), not the labelled twin to have learned the kink; on record, the twin captures about 15% of it |
| D3 | B3 and B3s (paired return, +2.0 in every seed) decide | **Returns are reported** with their checkpoint and never decide | Paired returns on record spread -35 to +10 across seeds and 45 between checkpoints; +2.0 is under the noise floor |
| D4 | B2 (ranking gain) is judged against zero | B2 must also **beat two matched nulls** computed from the saved U: the same U shuffled across anchors, and a constant penalty of U's median size | Both nulls cleared +0.20 on the dose screen's networks; without them B2 cannot tell placement from strength, which is the study's question |
| D5 | Every read must hold in every seed; MIXED is a reading | Anchor-level reads (O1, B2 and its nulls, U1's ranking part) **pool the 3 x 96 anchors** with the seed as stratum; rate reads (R1, R2, placement) pool over seeds; **unanimity only for B1** (entry, the one behavioral go read); **MIXED is removed** | With nine unanimous reads on three seeds, "no reading" was the modal outcome |
| D7 | Tests on Windows only; the package sealed untested in its environment | Sealing **requires a recorded WSL dry run**: every job kind once on smoke data from a test entropy, plus the instrument checks on the real anchors; its receipt is hashed into the manifest | Three attempts died on faults a dry run shows in minutes |
| D8 | An attempt is spent when the supervisor starts | **Stage 0** (preflight, instrument unit, G1) is exempt; an attempt is counted from the first training job | A 7-second loader fault cost the same as a 6-hour run |

Everything else in r4 stands: the question, the three corrections of section 1, the host, BCA as built, kappa 63,
the pairing by shared random numbers, the data by digest, RS1 as recorded, the seeds and the reserved confirmation
namespace, the bounded claims of section 6.

## 2. Preconditions (r4 section 3, revised)

1. **G2 PASS** (exact; unchanged).
2. **RS1 recorded**, with the user's decision on the primary miss in the approval words (unchanged).
3. **The WSL dry run passed** before the seal (new): preflight; the instrument unit I0 on the real anchors (the
   median exact rho(s10) equals the formal report to 1e-9 relative, H1 holds, both archives load by digest, both
   worlds' data plans re-derive equal); and one job of every training and evaluation kind on smoke data from a test
   entropy, structure only. Its receipt is in the package and hashed by the manifest.
4. **The frozen protocol, budget and approval words reviewed by the user** (unchanged).

G1 is no longer a precondition. It runs in stage 0 and is reported (section 4).

## 3. Stages, attempts and caps

- **Stage 0 (exempt from the attempt count):** preflight (GPU environment receipt), i0 (CPU instrument unit), g1
  (CPU calibration unit). A failure or an instrument miss in stage 0 ends the run INCONCLUSIVE (instrument) and is
  recorded; it spends no attempt. After three stage-0 failures the package must be re-sealed.
- **An attempt begins with the first training job** (the first warmup). One attempt; after INCOMPLETE, one
  re-attempt under a new approval naming attempt 2, reusing every completed unit (same id, kind, seed, arm, world).
  No attempt 3.
- **Caps (unchanged):** 1,800 s per training job; 10 h of training; 2 h and 36,000,000 exact transitions of
  evaluation; 1 h and 9,000,000 exact steps for g1; 600 s for the preflight and for i0.

## 4. Reads and readings

Definitions as in r4 section 5: KC against the arm's own clipped continuation; rho_own exact; the penalized contrast
of the BCA run is dQ1 - kappa dU. "Pooled" means over the common valid anchors of all three seeds (seed as stratum),
or the mean over seeds for a rate.

**Instrument (stage 0; a miss is INCONCLUSIVE (instrument), no attempt spent).**

| id | Read | Bar |
|---|---|---|
| I0 | median exact rho(s10) over the 96 anchors recomputed in the run's environment; H1 identity; both archives by digest; data plans re-derived | formal value to 1e-9 relative; H1 <= 1e-9; equal |

**Validity (a miss is INCONCLUSIVE; no BCA reading).**

| id | Read | Bar |
|---|---|---|
| V0(a) | adequacy J_E1(20k) - J_E1(untrained), CUTOFF and RANDOM, stall3 | >= +50 in every seed |
| V0(c) | anchors valid under both arms | >= 32 in every seed |
| V0(e) | a1..a3 past the knee on E2 steps | <= 0.005 in every seed |
| PLC | pooled no-stall-world median of (KC_C - KC_R)(s10) | within +-0.30 |

**Bed (stops, in order).**

| id | Read | Bar | Miss reads |
|---|---|---|---|
| R1 | CUTOFF R_deep- entry on E2 (stall3) minus its BC-only actor's, pooled over seeds | >= 0.25 | STOP: not reached (no BCA unit trains) |
| R2 | K3a, the return gained by clipping the CUTOFF actor's commands at the knee, pooled over seeds | >= 2.0 | STOP: no harm |
| O1 | pooled median KC_C(s10) >= 0.5 and >= 0.5 x pooled median rho_own | both | NO-GO: nothing to repair |

**BCA (the go/no-go; each paired by seed with the native branch of the same fork).**

| id | Read | Bar |
|---|---|---|
| B1 | entry(BCA-C) - entry(HOST-C) on E2, stall3 | <= -0.10 in every seed |
| B2 | pooled harmful-anchor ranking gain of the penalized value over raw Q1 (q_rank_gain_s10) | >= +0.25 |
| B2-null | the **net gain** (harmful-anchor gain minus the benign-anchor under-ranking change) of the real U against the same statistic for (a) dU shuffled across each seed's anchors, 2,000 shuffles (seed 20261101), median; (b) a constant penalty kappa x median\|dU\| per seed. Net, because a large enough even penalty flips every harmful anchor and every benign one; counting the benign cost is what makes the nulls matched. | real net >= each null's net + 0.05 |
| U1 | pooled under-ranking change on benign anchors (penalized ranks s10 below the knee where it is truly better, minus raw); placement = mean over seeds of [entry(BCA-C) - entry(HOST-C)](stall3) - [the same](no-stall) | <= +0.25; <= +0.05 |

**Reported, never deciding.** Returns J(BCA-C) - J(HOST-C) in both worlds and their difference, with the checkpoint;
V0(f); L1n (against G1's q* when G1 gives one); BAL (against G1's band); O2 (pooled median (KC_C - KC_R)(s10) >= 0.5
and larger than its no-stall value); U0; the twin's entry and KC; sign-flipped U's gain; G1's label, converged update
count, in-support band and departure capture.

**Readings, in precedence.**

| Reading | When |
|---|---|
| INCOMPLETE | a job failed, a cap bound, or the integrity check failed (stage 0 failures are INCONCLUSIVE (instrument), not INCOMPLETE) |
| INCONCLUSIVE (instrument) | I0 misses, or a stage-0 unit fails |
| INCONCLUSIVE | a validity read misses |
| STOP: not reached | R1 misses |
| STOP: no harm | R2 misses |
| NO-GO: nothing to repair | O1 misses |
| NO-GO: no consistent effect at this dose and budget | B1 misses in any seed, or B2 < +0.25 |
| NO-GO: strength without placement | B1 and B2 hold, B2 does not beat a null |
| NO-GO: over-correction | B1, B2 and the nulls hold, U1 misses |
| GO | all of B1, B2, B2-null and U1 hold |

GO is permission to design the confirmation stage (r4 section 6), not a claim. A NO-GO reads exactly its sentence.

## 5. G1 as a calibration unit (D1)

Fit as in `G1_RULE.md` (the host's critic class, exact labels, Adam 3e-4, batch 256, three seeds, both worlds), but
**to convergence**: checkpoints every 20,000 updates to 200,000; the converged checkpoint is the first at which the
mean training loss fell by less than 5% over the preceding 20,000 updates, else 200,000 (reported as not converged).
Read at that checkpoint, on the stall3 fit, over valid anchors (rho > 0):

- **in-support band** = the 90th percentile of \|KC\| at the four edge slots inside the knee (edge_775 to edge_700);
- **departure capture** = 1 - median KC(s10) / median rho;
- **label REPRESENTABLE** when median KC(s10) <= band + 0.15 in every seed (the fit's error at s10 is no worse than
  its own error at the matched distance inside the knee), else NOT REPRESENTABLE;
- q* and the resolution band at edge_700, as in G1_RULE.md, for the reported L1n and BAL.

No label of G1 stops the screen. Its reading is a calibration of what this function class can hold when told the
truth, placed beside the TD critic's reads.

## 6. What stays and what this does not show

Unchanged from r4 section 6. The screen still has no even-caution training arm and no data-only arm; the B2 nulls
are anchor-level stand-ins for even caution on fixed networks, not a trained control. GO leads to the confirmation
stage with those arms.

## Change log

- **r5 (2026-11-01):** D1 to D5, D7 and D8 from the design audit; G1 demoted to a reported calibration unit; the
  instrument unit i0 added to stage 0; stage 0 exempt from the attempt count; the WSL dry run required before the
  seal; MIXED removed; B2 nulls added; returns reported only; pooled anchor-level reads.
