# Signed and locally compatible BCA: frozen-checkpoint diagnostic

This prospective diagnostic fits only auxiliary residual models on the retained
training tuples. Six authenticated 20k hole-condition checkpoints (three seeds,
native and BCA) remain frozen; no actor, critic or target update is permitted.
It is an exploratory mechanism screen, not a policy-performance experiment.

Use the old fresh Q1 and twin-min auxiliary fits as fixed model parameters, but
recalibrate them with the same new 64 calibration episodes as the new signed
model. Fit one signed model per checkpoint for 2000 Adam steps, batch 256; use
the exact saved Q1 normalizer and original one-step smoothed target twin-min.
The signed residual is Q1-target. One row per fresh eligible episode; 64
episodes are calibration and 64 separate episodes are checks. Do not use the
old study's protected transition records. Freeze model, calibration, neighbor
selection, and coefficient matching before any new Q-pi reference is computed.

The compatible-action estimator uses 5 episode-disjoint folds and 768 retained
training probes, one per training episode, K=64, 10-step time strata, bandwidth
grid (.125,.25,.5,1)*training action standard deviations and .05 minimum. Select
by cross-validated logged-action Gaussian mixture likelihood (including the
bandwidth normalizer); tie break by listed order. Matching probes use nested
inner folds excluding their entire outer fold for bandwidth selection and the
radius threshold. State radius for fresh evaluation is the higher
95th percentile of cross-fitted 64th-neighbor distances, a heuristic. Outside
this radius compatibility contributes zero; this is not a support certificate.
Eta matches compatibility action-gradient RMS to .25 times Q1 RMS on the
cross-fitted probes. A symmetric Q1 width-only control matches the full combined
signed-plus-compatibility action-gradient RMS on the same probes. Guard masks
are fixed during action-gradient calculations. Positive finite gradients and
coefficient <=30 are required; there is no adaptive retuning.

At evaluation compare raw Q1, the original saved BCA (old calibration; descriptive
control), fresh-calibrated Q1 and twin-min symmetric BCA, original
compatibility-only, revised compatibility-only, signed Bayesian BCA, signed
split-conformal BCA, signed-plus-revised compatibility, strength-matched Q1,
and a deterministic shuffled-neighbor compatibility control. Scores use
Q1-3*width for symmetric BCA, Q1-signed_bound for signed, and Q1-eta*cost
for compatibility. The shuffled neighbor permutation is seeded independent of
evaluation outcomes. No score is a new trained actor.

Use 32 new anchor families/context at times 0,25,50,75 (8 each), and 32 new
free actor episodes/context, shared across the six checkpoints. Frozen reset
seed blocks are ordinary anchors 2026121000-2026121031 and free starts
2026121100-2026121131; strong anchors 2026121200-2026121231 and free starts
2026121300-2026121331; uniform-prefix action seeds 2026121401,2026121402.
Calibration-collection seeds and collision checks are specified in
fresh_data.py, whose exact hash is sealed before collection. The independent
shuffle stream is PCG64 seed 2026121510, and its permutation and unchanged
neighbor-set count are recorded. Every q-pi value
is a forced candidate action followed by that checkpoint's own frozen actor,
gamma .98, finite horizon 100. Slots are actor, and five fixed complete vectors
with a0=-.5,-.25,0,.25,.5 and other action dimensions zero.

Budget: 12800 calibration transitions; 2400 anchor-prefix transitions; 104
parity transitions; 144000 continuation transitions; 38400 free-policy
transitions: exactly 197704 simulator transitions, max 256 bank/calibration
resets plus parity if the adapter charges one, 12000 auxiliary updates, 18
calibration refreshes, 2304 forced-action references, 384 free episodes,
zero actor/critic/target updates. Two-hour deadline, no automatic retry.

Prospective exploratory gate: within a context at least 8 high-load anchors
and at least 8 raw-Q1 high-load choices with regret >.1. Each qualifying seed
must have combined high-band MAE at least 1 lower than symmetric Q1 BCA; mean
high-load bank regret at least .1 lower than each of signed-only, revised
compatibility-only and matched width; and harmed choice switches <= improved
switches using .1 tolerance. At least 2/3 seeds must qualify in each of the
native and BCA checkpoint groups for a context to pass. Failure of opportunity
is inconclusive, not success; report all measurements and caveats. These
thresholds are engineering screening criteria and use the three training seeds
as the replication units. No new simulator labels are used for tuning.

Calibration assesses one-step errors on fresh retained behavior rows only;
it is not a Q-pi or shifted-action coverage guarantee. Shared states, shared
calibration episodes and six candidate actions are paired observations, not
independent training seeds. This is a development diagnostic after earlier
exploration; it cannot establish out-of-sample policy improvement.
