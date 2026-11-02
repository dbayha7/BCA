# Formal G2, candidate knee_s10_v2, attempt 1: readout

Run by the user at their terminal, 2026-11-01 08:39 UTC (03:39 EST), approval commit `1672e956`, manifest
`e7be6e218b77623dd5c4c67791dbb1f000f3c4269f7bf30911d8ab34806de2be`, integrity re-check after the run clean. Record:
`formal_g2_run/attempt_1/report.json` and `collection.json` (datasets git-ignored; digests in the report:
stall3 `c2a6d616...`, no-stall `e3549869...`, 1,280 episodes each on the sealed v2 entropy). This readout reads the
report against `G2_FORMAL_RULE_v2.md` as written before the run. It adds no number the report does not hold; the one
derived bound in the last section is the rule's own arithmetic.

## Headline

**PASS**: every check met its bar in both worlds, P1 included. By the rule's table, PASS means the candidate is
testable under OOD definition v4 r3. It says nothing about what any learner does. There is no retry after PASS, and
none is needed.

## Every check

| Check | stall3 | no-stall | bar | result |
|---|---|---|---|---|
| P1 prefix commands inside the retained support, none past the knee; anchors identical in both worlds | 128 of 128; 0 past knee | same states, bit for bit | exact | pass |
| D1 CUTOFF holds no W row; predicates agree; CUTOFF is W's complement | exact | exact | exact | pass |
| D2 retained high-load a0 at or past s10; nearest distance | 0 rows; 0.1000 | 0 rows; 0.1000 | 0 rows; >= 0.0999 | pass |
| D3 starts inside their group's 0.99 quantile | 126 of 128 (0.984) | 127 of 128 (0.992) | >= 0.95 | pass |
| D3 gated anchors inside their group's 0.99 quantile | 96 of 96 (1.000) | 93 of 96 (0.969) | >= 0.90 (87 of 96) | pass |
| D4 label density, mid band | 2.451 | 2.451 | [2.328, 2.587] | inside |
| D4 label density, deep band | 2.440 | 2.440 | [2.114, 2.813] | inside |
| B1 pi_ref inside CUTOFF support at next states (76,032 rows) | 1.000 | 1.000 | >= 0.99 | pass |
| H1 max relative \|Q_stall - Q_nostall\| on S9 and anchors | 0.0 | | <= 1e-9 | pass (exact) |
| H2 median exact rho(s10), all 96 gated anchors | 1.765 | | >= 0.779 | pass |

Budget: 492,896 exact simulator steps of 1,000,000; 42.9 s of 1,800. Collection 2 x 128,000; anchor prefixes in both
worlds 9,600; continuations 2 x 100,800; projection readout 96; returns 25,600.

## What changed from v1, and what the numbers say about it

v1 failed D3 for anchors in the no-stall world (84 of 96) because its anchors were reached by replaying the widened
recorder's commands in stall3, and half of the high-load draws stall. v2 changed one thing: the prefix draws a0 inside
CUTOFF's retained support, so no prefix command stalls, and each anchor is one physical state for both worlds (P1
confirmed it for all 128 anchors, bit for bit). The D3 anchor shares moved from 95 and 84 of 96 to 96 and 93 of 96.
The three no-stall misses are inside the bar with room (87 is the bar). The t = 0 groups are reset states as before;
the per-time thresholds (stall3 4.484, 4.010, 3.254; no-stall 4.406, 3.685, 3.110 at t = 0, 25, 50; 126 deep-load
reference rows each) are now within a few percent of each other between worlds, as expected when the fitting data
differ only in executed thrust and the anchors do not.

H2 was the risk named before the run, since the in-support prefix moves the anchor states and the development value
(2.198) was seen data. On fresh anchors the median exact gap is 1.765, above v1's 1.536 and more than twice the bar.

## Reported, not claims

- Gap distribution over the 96 anchors: quantiles -0.925 (10%), 0.051 (25%), 1.765 (50%), 3.505 (75%), 4.881 (90%).
  About a quarter of the anchors carry little or no gap at s10, and a tenth a negative one. The gate scores the pooled
  median, as the rule says; the per-anchor spread is what the later gates will have to resolve.
- By anchor time: 0.835 at t = 0, 3.595 at t = 25, 0.465 at t = 50. The t = 50 median alone is below the pooled bar
  of 0.779. That is a reported split, not a decision, and the scoring set is all 96 by the user's approval.
- Where pi_ref's unprojected first continuation command exceeds the knee (72% of anchors) the median gap is 2.812;
  elsewhere -0.693. The gap at s10 lives mostly where pi_ref wants more thrust than the support allows.
- Other depths: median rho 0.836 at s05, 3.797 at s20, -4.238 at p10.
- Predictors against exact values: median rho_hat(s10) 1.739; item 5 shares 0.792 (max form) and 0.854 (min form);
  item 6 sign control exact share negative 0.979 (P1-P3 0.969).
- pi_ref's mean return from the 128 starts in stall3 is -71.8; the zero policy's is -716.3.
- The data: 76,800 fitting rows, 28,822 in W, 47,978 retained; high-load share 0.596, deep share 0.087; the D4
  densities agree to every digit between worlds because both executed the same commands.

## A note on the D3 reference population

A concurrent note (`analysis/knee_guard_repair_v1/`, written before v1's result) argues D3 should measure support
against CUTOFF-retained rows rather than all fitting rows. This run follows its sealed rule (all fitting rows). Under
this recorder a0 depends only on the load band and the sealed uniforms, so within a band retention is a coin flip
independent of the state; the retained rows have the same per-band state distribution and about half the density at
mid and deep load, and D3's threshold is calibrated on the same pool the queries use. The variant would be a noisier
reading of the same question, and at 126 deep-load reference rows per time step its retained groups (about 64) sit at
its own 64-row floor. It is not adopted and did not decide anything here.

## What this leads to

By the rule the user approved, PASS leads to the next gates, each with its own rule written first, its own sealed
package and the user's own approval, and each run on exactly these two datasets by digest:

1. **G1, the representation positive control**: the 2 x 256 raw-action critic fit by supervised MSE to exact
   Q^{pi_ref} labels on the twin's (s, a) rows, no TD. It must meet L1 at s10 and it fixes q* and the in-support
   resolution band.
2. **The variance pilot**: screen-only seeds of both arms in policy-evaluation mode, used only to estimate SD_theta.
   Then H2 is evaluated in full: R_req = (0.5 + 0.3 x 0.15 + 1.28 x SD_theta) / 0.7. With this run's median 1.765,
   the full H2 holds while SD_theta <= (1.765 x 0.7 - 0.545) / 1.28 = 0.539 (the rule's arithmetic).
3. **G3, learnability**: policy-evaluation mode, twin only, both worlds, screen seeds; L1 in every seed.

The ledger entry is written (`../../ood_definition_v4/LEDGER.md`). No claim about BCA follows from this run.
