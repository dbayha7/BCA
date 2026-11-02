# Design audit of the r4 native screen (2026-11-01, after attempt 3)

Scope: `REVISION_r4.md` sections 3 to 7, `knee_s10_candidate_v2/native_screen/SCREEN_RULE.md`, `screen_math.py`,
`knee_s10_candidate_v2/g1/G1_RULE.md`. Question asked of every precondition, read and bar: (a) can this host, at this
budget, resolve what the bar asks, and (b) does the read answer the study's question or something else. Evidence is
what is already on disk: the attempt-3 G1 record, the knee-edge exploratory Stage 1 native evaluations (same host,
same anchor design, 20k), and the dose screen. No new run.

## Evidence

**E1. The native TD critic's contrast error by slot, 20k, knee-edge bed (12 runs, median KC against edge_knee,
own clipped continuation).** Inside the knee the error is small near the edge and grows with distance from it, in
both worlds: edge_700 (0.1 inside) -0.06 to +0.17; mid_m40 -0.34 to +0.68; zero -0.11 to +1.49; knee (+0.8) +0.65 to
+4.65. Just outside, at s10 (0.1 past the knee): no-stall world +0.03 to +0.11; stall world +1.24 to +1.61, in both
arms, against rho about 1.5 to 1.9. So the host resolves a 0.1 step in a0 to about 0.05 to 0.15 where the truth is
smooth, and misses the kink by about 85% of its size when it is there, labels or not.

**E2. G1's supervised fit at the same 20k (attempt 3).** Its error at edge_700 is +0.78 to +0.87, at zero +4.2 to
+5.2, at knee +6.5 to +7.3: ten to fifty times the TD critic's error at the same in-support slots. The MSE regression
to exact returns is far less converged at 20,000 updates than the TD critic is. In the no-stall world the same fit
is within +-0.05 everywhere, so the stall world's labels are the hard function for this regression.

## Defects, each with its fix

**D1. G1 is not an instrument of the host's representation; it is an under-converged regression read at an absolute
bar.** Its bar (within 0.15 of the exact contrast at s10, at 20k updates) is set in absolute Q units and read at a
budget borrowed from TD by analogy. E2 shows the analogy fails: the regression is an order of magnitude coarser than
the host's critic in support, so G1 fails everywhere, kink or no kink. The 60k diagnostic improving in every slot
confirms convergence, not representation, is what it measures. Consequence: attempt 3 was spent, and q* (0.11) and
the band (1.84) it would have handed the screen are meaningless.
*Fix.* G1 must judge the departure relative to the fit's own in-support error at the matched slot (does the fitted
critic capture the kink at s10 beyond its error at edge_700?), and it must be fit to convergence (a loss plateau
criterion, with the update count reported), not to a TD budget. Its per-anchor share bar follows from the same
relative reading. Until then G1 cannot gate anything.

**D2. Twin learnability is a validity condition for the BCA reading, and the evidence says it will fail.** r4 makes
V0(f), L1n and BAL validity: any miss ends the screen INCONCLUSIVE with no BCA reading. E1 shows the labelled RANDOM
arm on the knee-edge bed captured about 15% of the kink (KC_R(s10) 1.24 to 1.35 against a V0(f) bound of about 0.55).
The new recorder has deeper labels (28,822 W rows), so the twin may do better, but nothing on record says so. As
written, the most likely outcome of 6 to 8 GPU hours is INCONCLUSIVE on V0(f) or L1n, with the BCA units never run.
And the question "does BCA act where the critic is wrong" does not need the labelled twin to have learned the kink: it
needs the critic to be wrong (O1) and the actor to go there (R1). Twin learnability answers a different question, the
withholding attribution (O2).
*Fix.* Validity for the BCA reads is V0(a), V0(c), V0(e), the placebo PLC and the instrument checks only. V0(f), L1n,
BAL and O2 become reported findings about withholding, with their own labels, never a stop. The BCA units train
whenever R1 and O1 hold.

**D3. The return bars are below the paired noise floor.** B3 and B3s ask for +2.0 in every seed. The dose screen's
paired BCA-minus-native E2 returns moved by -35 to +10 across seeds and by up to 45 between the 15k and 20k checkpoints
of one run. A +2.0 bar on three seeds under that spread is a coin flip per seed, so the B3 consensus will be MIXED,
which r4 defines as "no reading". The go/no-go then never reads.
*Fix.* Returns are reported with their checkpoint, not decided on. The deciding BCA reads are the two with
within-seed power: B1 (entry rate over 128 episodes) and B2 (ranking over the 96 anchors). If a return read must
decide, its bar is set from the paired spread on record (about 20), not from the within-run drift the earlier protocol
quoted.

**D4. B2 has no null.** The dose screen showed that U shuffled across anchors, and an even ramp of U's typical size,
both clear a +0.20 ranking-gain bar on the same networks. B2 at +0.25 against zero therefore cannot tell placement from
strength, which is the study's question. The null costs nothing: U at the anchors is saved.
*Fix.* B2 decides only if the penalized ranking gain beats, in every seed, the gain of the same U shuffled across
anchors (median of 2,000 shuffles) and of a constant penalty of the same median size. Both nulls are computed from the
saved U by the frozen reads. Sign-flipped U is reported.

**D5. Unanimity over three seeds on nine reads makes "no reading" the modal outcome.** Every validity read and every
deciding read must hold in all three seeds; one seed off on one read gives INCONCLUSIVE or MIXED. With the spreads on
record that is the expected result, and r4 sends MIXED to "a design question for the confirmation stage", which is a
rerun by another name.
*Fix.* Pool where pooling is valid: anchor-level reads (O1, B2 and its nulls) pool the 3 x 96 anchors with the seed as
a stratum and decide on the pooled statistic, reporting per seed. Unanimity stays only for the one go decision (B1
and B2 together). MIXED is removed as a reading; a read either holds on its pre-registered statistic or it does not.

**D6. The reads at s10 are resolvable; keep them.** E1 shows the host's error 0.1 past the knee is 0.03 to 0.11 where
the truth is smooth, so O1 (0.5 and half of rho) and the s10 contrasts are well inside the host's resolution. The
placebo PLC at +-0.30 is also resolvable (no-stall contrasts differ by under 0.1 between arms). No change.

**D7. The package was never run end to end in the environment that runs it.** Attempts 1 to 3 died on an
environment variable, a loader that regenerates data bit for bit across platforms, and a bar never checked against
in-support error. All three would have shown in one dry run of the sealed package on tiny data in WSL before the seal.
Tests ran on Windows only.
*Fix.* Sealing requires a recorded WSL dry run: every unit kind once on smoke data, the instrument checks on the real
anchors (as `i0_platform_check_v1` now does), exempt from the attempt count like the preflight. The seal records its
receipt. An attempt is spent only after the dry run passed.

**D8. Attempt accounting charges setup faults to the science.** The rule allows two (then three) attempts and
counts an attempt the moment the supervisor starts, so a 7-second loader failure costs the same as a 6-hour run.
*Fix.* Units before training (preflight, data load, instrument checks) are stage 0 and exempt; an attempt is counted
from the first training job. The cap on attempts stays.

## What stays

The bed (formal G2 PASS, exact rho 1.765 at s10), the datasets by digest, RS1 as recorded with the user's decision,
pi_ref v2, the recorder, the A1 loader, the dose kappa 63 (chosen by rule, with kappa 3 shown to have no leverage),
the pairing of BCA and native branches by shared random numbers, the three corrections in r4 section 1, and the
bounded claims in r4 section 6. The study question is unchanged.

## What this leads to

One revision, r5, carrying D1 to D5, D7 and D8, and one re-sealed native screen package under it, dry-run in WSL
before the seal, approved once, run once. No new gate and no panel. The G1 unit stays in the package but as a
reported calibration unit under its corrected reading, not as a stop.
