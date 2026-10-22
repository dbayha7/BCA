# BCA, checked one component at a time: the one-page brief for the meeting

Read aloud in about five minutes. Every number here is in `index.html`, in a module page under `modules/`, or in
`PENALTY_EVIDENCE.md`, which name the script or record behind it. Nothing was trained for this check.

## The story, in order

BCA puts an error band around an offline critic and turns the band into caution for the actor. I took it apart into
eleven components in the order data flow through them, and for each one I wrote down what "working" means as a
checkable statement, labeled every line of its pseudocode with its purpose and its place in the code, tested it on
synthetic data where the truth is known, and recomputed it from the recorded inputs of our ThrusterChain studies in
the pinned environment, so that the recorded numbers had to come back exactly.

The first six components build the band. The residual (M1) is the critic's own one-step error at a recorded row; the
error-size network and the unit (M2) predict how large that error should be; the scores (M3) divide the one by the
other; the weights (M4) say how much each held-out row counts; the threshold (M5) is the published weighted Bayesian
conformal rule with its refusal; the band (M6) is the threshold times the predicted size. Every one of these
reproduces its recorded numbers bit for bit: 10 of 10 bands in M1, 25 of 25 in M3, 30 of 30 thresholds in M5, and
with weights of 1 the weighted band is the standard band to the last bit in every study.

The other five components are the ways the band has been used. Three of them read the band's width: the dose on the
behavior-cloning term (M7), a toy width penalty on critic values (M8) and the lower edge (M9). Two read held-out
outcomes with the same posterior: the harm switch (M10) and the ramp it places (M11).

## The verdict per module

- M1 residual: works.
- M2 error-size network and unit: works with a caveat (it is nearly a constant on ThrusterChain).
- M3 scores: works.
- M4 weights: works with a caveat (every weight on record comes from a shift law we know).
- M5 threshold with refusal: works (refusal is pure arithmetic of the group size).
- M6 band: works with a caveat (a band on the one-step target, on average over rows, as wrong as the critic where
  the data stop).
- M7 dose: works with a caveat (exact, but nearly a constant 1.33; saturates).
- M8 width penalty on critic values: does not work (toy only; worse than refusing thin data).
- M9 lower edge: does not work (penalty 0.1-0.5% of the critic's over-optimism; the actor does not move).
- M10 harm switch: works with a caveat (right in 40 of 40 seed-worlds; the height is undefined below about 29
  full-throttle rows).
- M11 ramp height: works with a caveat (plumbed correctly; large enough, not shown calibrated; 3 of 5 fresh seeds
  stopped on an undefined height).

Overall: works with a caveat. The arithmetic is right everywhere; the band describes the recorded rows and cannot see
the critic's error where the data stop; the one use that works reads outcomes, not width.

## The penalty answer (the two questions from last time)

Is the penalty low under IID and higher under a shift? For standard BCA, no, and consistently so: the penalty is the
same number under IID and shifted tests because it is read on the training rows from a threshold fixed at the last
refresh. The dose is 1.29 to 1.39 on every host and benchmark on record, and the band at the actor's unseen commands is
only 1.10 to 1.29 times its width at seen commands. What drops under a shift is coverage: 0.89-0.94 to 0.80-0.87 per
seed, on 25 of 25 seeds. The one penalty that points the right way is the harm switch's height, high where full
throttle stalls and low where it does not on 32 of 32 finite cases, because it reads held-out outcomes.

Is the weighting right, and why can IW give a higher penalty under IID? The weighting is mechanically correct: with the
IID test declared exactly (weights of 1), IW is standard BCA to the last bit in two studies and on 30 recomputed banks.
With weights for a declared shift IW's penalty is higher on every seed, by construction: 1.1 to 2.3 times in the
threshold, 0.02 to 0.16 in the dose. On the banks' own recorded currents the rise is 1.47 times at the weighting test's
effective sample size, of which 1.38 comes from pointing the band at the high-current rows, where the critic is wrong
by more, and about 7 to 9% from the smaller effective sample. IW's penalty is higher under IID without a declared shift
in exactly two cases, both bounded: classifier-estimated weights add about plus or minus 5% of noise (14 seeds up, 9
down, 2 beyond 5%), and the threshold refuses when the effective number of held-out rows falls below about 29 (8 of 40
harm-switch certificates, predictable from the group size alone). Neither is a bug in BCA's arithmetic.

## Three takeaways

1. The mechanics are right. Every stored score, threshold, width and certificate reproduces from the saved networks,
   and weights of 1 give the standard band exactly.
2. The band describes the recorded rows. It covers 90-94% of fresh rows from the recorded situations, but it is a band
   on the critic's one-step target, it holds on average (0.94 at low current, 0.87 at high), and in the stalling world it
   covers only half of the removed commands; the width is flat across commands, so the dose and the lower edge have no
   leverage.
3. The outcome-reading use works and the height is the open problem. The switch was right in 40 of 40 seed-worlds; the
   height it hands to the ramp is infinite for groups under about 29 rows, which stopped the pre-registered
   confirmation on 3 of 5 fresh seeds, and the records show the height is large enough, not that it is calibrated.

## Open questions for the advisor

1. The height. When the switch is on but the full-throttle group is too small to certify, should a pre-registered
   fallback height be allowed (the plain 90%-quantile gap did nearly as well on seeds 0-4; a fixed height of 1.0 was
   already an arm), or is "on but uncertified" the honest stop?
2. The band's meaning. Since the band is on the one-step target and the width is flat across commands, is there a
   version of the error-size network, or a different residual, that could carry the exact-error signal the oracle
   lower edge needs (it wins back 94 of 99 points with the exact error), or should the width-reading uses be dropped?
3. The weights. Every weight on record comes from a known shift law; where the shift is the actor's own commands the
   estimated weights collapse to a handful of effective rows and the band refuses. Is refusal the right answer there,
   or is there a weighting we have not tried?
