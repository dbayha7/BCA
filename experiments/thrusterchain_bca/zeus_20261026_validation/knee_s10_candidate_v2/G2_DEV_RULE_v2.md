# Candidate knee_s10_v2: development G2 rule (written before the screen ran)

Written 2026-11-01, after the formal G2 of candidate knee_s10_v1 read NOT DATA-OOD (attempt 1, record
`../knee_s10_candidate_v1/formal_g2/formal_g2_run/attempt_1/report.json`, readout `G2_FORMAL_READOUT_20261101.md`),
and before any v2 number exists. The user asked for v2 ("Yes, build v2", 01:39 EDT).

## What v2 is: one change

Candidate knee_s10_v2 keeps everything of knee_s10_v1 except the anchor prefix rule: the same sigma (s10), the same
widened recorder (`../knee_s10_candidate_v1/recorder_wide.py`), the same pi_ref v2 (`../knee_s10_candidate_v1/pi_ref.py`,
gains 7b35523d...), the same definition (v4 r3), the same checks and bars.

v1 built each anchor by running the recorder's own commands for t steps in stall3. Half of the recorder's high-load
window draws are past the knee (|a0| > 0.8), where stall3 cuts thrust to 0.8 - 3(|a0| - 0.8), so after 25 or 50
steps the anchor state carried the stall's effect. The no-stall episodes got full thrust on the same commands and
seldom visit those states: 12 of 96 gated anchors fell outside the no-stall data (formal D3, 84 of 96 against 87),
while 95 of 96 were inside the stall3 data. The development screen had already shown the asymmetry (no-stall 44 of 48,
stall3 48 of 48) without failing.

v2's prefix (`prefix_rule.py`) maps every prefix draw onto CUTOFF's retained support for the load band of the state
the command is given in, using the same five uniforms per step the recorder would use:

| load band of the state | v1 (recorder) | v2 (conditioned on the retained support) |
|---|---|---|
| high (g >= 0.4), u < 0.98 | a0 = -1.0 + 0.4 v, on [-1.0, -0.6] | a0 = -0.8 + 0.2 v, on [-0.8, -0.6] |
| high, u >= 0.98 | a0 = -1 + 2 v, on [-1, 1] | a0 = -0.8 + 0.2 v, on [-0.8, -0.6] |
| low (g < 0.4) | a0 = -1 + 2 v, on [-1, 1] | a0 = -0.8 + 1.6 v, on [-0.8, 0.8] |
| sides a1..a3 | -0.7 + 1.4 w_k | unchanged |

No prefix command is past the knee, so the stall never acts during a prefix and the prefix trajectory is the same
in stall3 and nostall: each anchor is one physical state for both worlds' D3 checks. The anchor's a0 at the anchor
itself (s10, past the knee) is unchanged; so are the continuation (pi_ref v2) and the exact Q loop.

## Status: a development screen, never confirmatory

- Test entropy 424263 and test-only stream 707,070,707, the same as v1's development screens (their outcomes are
  known). Both are in the formal seed audit's development set. The two datasets are therefore byte-identical to
  v1's development data, so D1, D2, D4 and B1 replicate v1's development values exactly, and the only quantities
  that can move are the anchor-side ones: D3 anchors, H1, H2 and the reported anchor readouts.
- 64 anchors from the development bank `stage1_e2_anchors` (16 per time in 0 / 25 / 50 / 75; the 48 at t in
  {0, 25, 50} are gated), the same reset seeds and prefix uniforms as v1's development screen: the comparison with
  `../knee_s10_candidate_v1/g2_development_result_r3.json` is paired on anchors.
- 128 starts from `stage1_e2_bank` under the same stream.
- Exact arithmetic only (raw float64 loop, pi_ref v2). No training, no checkpoint, no GPU. Run by Claude on Windows,
  as v1's development screens were; it touches no sealed file.
- Caps: 1,000,000 exact simulator steps and 1,800 s, charged before each unit as in the formal driver.

The measurements, D4 bounds and label code are v1's sealed formal driver (`../knee_s10_candidate_v1/formal_g2/g2_formal.py`,
imported, not copied; its preflight and execute are not called) and its sealed two-sided bounds `d4_bounds.json`,
which are a property of the unchanged recorder at 1,280 episodes.

## Checks and bars

| id | Check | Bar | Fails to |
|---|---|---|---|
| P1 | every prefix command inside CUTOFF's retained support for its load band, none past the knee; the prefix replayed in nostall gives the same observation and state as in stall3, bit for bit | exact | INCOMPLETE (the rule was not implemented) |
| D1 | `masks.arm_masks` CUTOFF holds 0 W rows; the two independent W predicates agree row for row | exact | NOT DATA-OOD |
| D2 | no retained high-load a0 <= s10; distance from s10 to the nearest retained high-load a0 | 0 rows; >= 0.0999 | NOT DATA-OOD |
| D3 | L1.5 as amended: the 128 starts and the 48 gated anchors against the 0.99 quantile of their own group | starts >= 95%, anchors >= 90% (44 of 48) | NOT DATA-OOD |
| D4 | twin label density at a0 in [-0.95, -0.85], per high-load band, against v1's sealed two-sided bounds | mid in [2.328, 2.587], deep in [2.114, 2.813] | below: NOT TESTABLE; above: INCONCLUSIVE |
| B1 | share of fitting-row next states where pi_ref's command is inside CUTOFF's retained support | >= 0.99 | NOT TESTABLE |
| H1 | max over S9 slots and gated anchors of \|Q_stall - Q_nostall\| / max(1, \|Q\|), exact, under pi_ref | <= 1e-9 | INCONCLUSIVE |
| H2 | median over the 48 gated anchors of the exact rho(s10) = Q_nostall(s10) - Q_stall(s10) | >= (0.5 + 0.3 x 0.15) / 0.7 = 0.7786 | NOT TESTABLE |

Headline precedence: INCOMPLETE, INCONCLUSIVE, NOT DATA-OOD, NOT TESTABLE, PASS. Every label computed is reported.

## What each label leads to

| Label | Leads to |
|---|---|
| PASS | Seal a formal G2 for v2 for the user's approval: a sibling of v1's formal package with the v2 prefix rule, a new seed namespace (`knee_s10_candidate_v2/`), fresh entropy and streams audited against every integer on record, 96 gated anchors, the same two-sided D4 bounds (recorder unchanged, pinned by hash), the same words of approval naming v2. Nothing in this screen is evidence for that run. |
| NOT DATA-OOD | This prefix rule stops at this recorder and sigma; the ledger records which check. A different prefix rule, recorder or sigma is a new candidate. |
| NOT TESTABLE | The same. If H2 is the cause, it says the value gap at s10 lives in states only a stalled prefix produces, which are outside the no-stall data by construction: a finding about this candidate family, recorded, not a reason to move a bar or a scoring set. |
| INCONCLUSIVE | H1: an instrument fault, fix and rerun once. D4 above its bound cannot occur here (the data are v1's development data, already inside the bounds). |
| INCOMPLETE | Fix the instrument and rerun; the attempt is listed. |

## Stated before the run

- v1's formal run put the exact gap at s10 at a median of 0.761 over the t = 0 anchors (reset states, no prefix),
  4.195 at t = 25 and 0.902 at t = 50 (bar 0.7786). v2's prefixed anchors will be nearer to reset-like states than
  v1's stalled ones. H2 is therefore open; the screen decides it, and the rule above says what a miss means.
- The development screen's PASS admits nothing. It only lets the formal G2 be sealed.

## Reported only, never used to decide

D3 anchor share by anchor time; rho(s10) by anchor time and its quantiles; rho split by whether pi_ref's a0 is
projected at the first continuation state; the paired comparison with v1's development r3 result (D3 anchor shares,
H2, medians); the predictors and item 6 (as in r3); pi_ref's and the zero policy's mean return from the starts.
