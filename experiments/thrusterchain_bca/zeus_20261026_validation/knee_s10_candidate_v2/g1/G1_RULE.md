# Candidate knee_s10_v2: the G1 rule, the representation positive control (written before any G1 number exists)

This is gate G1 of OOD definition v4 (`../../ood_definition_v4/DRAFT_DEFINITION.md` section 5, kept by
`../../ood_definition_v4/REVISION_r4.md` section 3) for candidate knee_s10_v2, after its formal G2 PASS
(`../formal_g2/G2_FORMAL_READOUT_20261101.md`). Written 2026-11-01. **It runs as the `g1` unit of the native screen
(`../native_screen/SCREEN_RULE.md`), under the screen's approval, manifest, caps and attempt rules; it has no seal,
approval or attempt of its own.** Its headline is read by the screen's validity (a label other than PASS ends the
screen INCONCLUSIVE before anything trains), and its q* and resolution band are read by the screen's L1n and BAL.

## The question

Can the host's critic, as a function class at the host's settings, hold the value departure at the scored slot when
it is given exact labels? If not, no learning run on this host can be read as "withholding caused the miss", and no
BCA run can be read as "BCA repaired it". G1 is a positive control on the representation. It trains nothing by
temporal difference and asks nothing about any learner's behavior.

## What is fit, and on what

- **Rows.** Every fitting row (split code 0) of each formal G2 dataset, by digest: stall3 c2a6d616... and no-stall
  e3549869..., 76,800 rows each (the twin's rows: CUTOFF's retained rows plus the W rows).
- **Labels.** Exact Q^{pi_ref}(s_t, a_t) in the row's own world: the row's reward for its executed command from its
  exact float64 state, then the continuation pi_ref version 2 (`../../knee_s10_candidate_v1/pi_ref.py`, gains
  pinned) from the next state to the horizon, discount 0.98, no bootstrap after step 99; the raw float64 loop
  `qpi.raw_values_from_states`, with the same clip (all four commands to +-KNEE, a no-op for pi_ref v2) the formal
  G2 used. No fitted or bootstrapped quantity enters a label.
- **The critic.** Unifloral's `SoftQNetwork` reproduced line for line (upstream `algorithms/td3_bc.py`, sha256
  6f5ab6f9...): the observation normalized by (mean, SD + 1e-3), concatenated with the raw action, two Dense(256)
  layers with ReLU, one Dense(1); flax's default initialization. One head per fit seed. Inputs: the 14 observation
  numbers and the 4 executed commands, as the host sees them.
- **The normalizer.** CUTOFF's retained rows' observation mean and SD in that world (the shared normalizer of r3
  section 3.3), computed from the same CUTOFF mask the formal G2's D1 used.
- **The fit.** Mean-squared error between the head and the label; Adam, learning rate 3e-4, epsilon 1e-5; batch 256
  drawn uniformly with replacement; 20,000 updates (the host's update budget, where the gate reads) continued to
  60,000 (reported only, the under-training diagnostic); three fit seeds (`g1_seeds.json`, namespace
  `knee_s10_candidate_v2/g1/`, audited against every integer on record), each seeding the initialization and the
  minibatch stream. Both worlds are fit; the stall3 fit decides, the no-stall fit is the smooth placebo.

## Where it is read

The 96 gated anchors of the formal G2, rebuilt from the sealed v2 seeds by the same prefix rule (P1 holds for every
one), at the 14 gated slots (S9 and s05, s10, s15, s20, p10; a1..a3 = 0, `probes.build`). The exact values at those
(state, slot) pairs under pi_ref are recomputed in both worlds exactly as the formal G2 computed them.

- **I0 instrument check.** The recomputed median exact rho(s10) over the 96 anchors must equal the formal report's
  value (`../formal_g2/formal_g2_run/attempt_1/report.json`, H.H2.median_exact_rho_s10) to 1e-9 relative, and the
  recomputed H1 identity must hold (<= 1e-9). A miss is INCONCLUSIVE (an instrument fault), never a result.
- **KC.** For a fit and a slot, KC_j(slot) = [Q1(s_j, slot) - Q1(s_j, edge_knee)] - [Q^{pi_ref}(s_j, slot) -
  Q^{pi_ref}(s_j, edge_knee)], per anchor, exact reference, in the fit's own world (`stage1_math.kc`).
- **rho_j** = Q^{pi_ref}_nostall(s_j, s10) - Q^{pi_ref}_stall(s_j, s10), exact (the G2 quantity).
- **Valid for L1:** rho_j > 0, a positive departure at s10. The definition's "valid anchors" is read that way here:
  "learns the departure" presupposes one. Anchors with rho_j <= 0 are reported under the plain resolution band
  |KC_j(s10)| <= tol, and never enter the share. The valid count is reported and is the same for every seed
  (rho is exact and seed-free).

## Checks and bars (stall3 fit, at 20,000 updates, tol = 0.15, c = 0.3)

| id | Check | Bar | On a miss |
|---|---|---|---|
| I0 | instrument: recomputed median rho(s10) equals the formal report's; H1 holds | 1e-9 relative | INCONCLUSIVE |
| F0 | every fit's loss finite at every checkpoint; every Q1 at the anchors finite | exact | INCONCLUSIVE |
| L1-median | -0.15 <= median_j KC_j(s10) <= 0.3 median_j rho_j + 0.15 over valid anchors | every fit seed | NOT REPRESENTABLE |
| L1-share | per valid anchor, -0.15 <= KC_j(s10) <= 0.3 rho_j + 0.15; the pass share pooled over the three seeds | >= 0.70 pooled, and >= 0.60 in every seed | NOT REPRESENTABLE |

The share floor is the smallest positive-control share at which q* (below) stays near 0.6 or above with three screen
seeds, so that the screen's learnability read asks the labelled arm to capture the departure at most anchors rather
than at a chance-level fraction. r1's 70% was a value taken from development runs and withdrawn; this 0.70 is a
floor on a positive control with exact labels, chosen before any G1 number exists, and the screen's bar is q*, not
this floor.

## What a PASS fixes (written into the report, then used by the native screen)

- **q\*** = the largest k / n_valid such that [P(Binomial(n_valid, p_hat) >= k)]^3 >= 0.90, with p_hat the pooled
  L1-share. It is the largest share that a critic with G1's per-anchor pass probability clears in every one of three
  screen seeds with probability at least 0.9 (r3 section 4, "Allowances"). The binomial tail is computed exactly in
  log space; no library distribution is used.
- **The in-support resolution band** = the 90th percentile, over valid anchors and the three seeds, of
  |KC_j(edge_700)|, the contrast error at the slot 0.1 inside the knee (the matched in-support distance to s10).
  The screen's balance read BAL compares the arms' in-support disagreement with this band.

## Labels

Precedence: INCOMPLETE > INCONCLUSIVE > NOT REPRESENTABLE > PASS. Every label computed is reported.

| Label | When | Leads to |
|---|---|---|
| INCOMPLETE | the run raised, a cap bound, or the integrity check failed | a second attempt with at most one code-only fix, under a new approval naming attempt 2 |
| INCONCLUSIVE | I0 or F0 fails | fix the named instrument cause; attempt 2 under a new approval; no third attempt |
| NOT REPRESENTABLE | L1-median or L1-share fails | stop this candidate on this host configuration: change the feature or the host settings (r3 section 5), never the data, the slot or the tolerance; a changed setting is a new candidate in the ledger |
| PASS | every check passes | q* and the resolution band are fixed; the native screen may be sealed (r4 section 4) once RS1 holds |

No label here says anything about withholding, over-valuation or BCA. G1 says only whether this critic can hold
this departure when told the truth.

## Reported only, never used to decide

- The no-stall fit at 20,000 and 60,000: the share of all 96 anchors with |KC_j(s10)| <= 0.15 and the KC quantiles
  (the smooth placebo: whether the tolerance is achievable at all at this slot).
- The stall3 fit at 60,000: the same L1 reads as at 20,000 (the under-training diagnostic).
- The training mean-squared error at 5,000, 10,000, 20,000, 40,000 and 60,000 updates, per fit, and the mean-squared
  error against the exact values at the anchors' 14 slots.
- KC quantiles at s05, s10, s15, s20, p10 and at every S9 slot (the resolution profile inside and outside the
  support), stall3 fit; the share of anchors with KC_j(s10) <= -0.15 (under-capture) and >= 0.3 rho_j + 0.15
  (over-capture, the critic over-values the departure even with labels).
- The anchors with rho_j <= 0: their count and their plain resolution share.
- The label statistics: quantiles of the exact Q labels per world, the share of W rows, the wall clock and exact
  simulator steps of every unit.

## Attempts

The screen's: one attempt; after INCOMPLETE a second attempt reuses a completed g1 unit. There is no retry of G1
after NOT REPRESENTABLE or PASS. A rerun after INCONCLUSIVE (an instrument fault) is a rerun of the whole screen's
attempt with the fault fixed, under a new approval.

## Caps

9,000,000 exact simulator steps (expected about 7.8 million: two worlds x 76,800 rows x about 50.5 remaining steps
for the labels, plus 2 x 96 x 14 x about 75 for the anchor values) and 3,600 s of wall clock for the unit. The
fits run on the CPU JAX runtime (JAX_PLATFORMS=cpu, set by the worker before any JAX import; float32); no GPU and no
checkpoint of any learner is read or written by this unit.

## What the user's approval states

The screen's approval (`../native_screen/SCREEN_RULE.md`) covers this unit; it adopts r4 for the candidate. The two
readings fixed here, the valid-anchor reading of L1 (rho_j > 0) and the 0.70 / 0.60 positive-control floor, are part
of the sealed rule the approval names.

## Integrity

The screen's manifest hashes this rule and the g1 modules (`g1_math.py`, `g1_labels.py`, `g1_fit.py`,
`../native_screen/g1_unit.py`), the formal G2 package the anchors are rebuilt with, the prefix rule, v1's pi_ref and
recorder, the definition and r4; the worker refuses on any change before and after the unit. Both datasets must
equal their digests in the formal report. The labels and the fitted parameters are saved in the unit folder
(git-ignored) with their digests in the report.

## Ledger entry

Candidate knee_s10_v2, sigma s10, G1 under r4: this attempt's headline, q*, the resolution band, the pooled share
and the median KC at s10, with the fit budget at which they were read.
