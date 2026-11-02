# OOD definition v4 for the ThrusterChain BCA study (DRAFT r3, not approved)

> **r3 (2026-10-31, after the knee_s10 development screens and review `wf_e1964e57-92d`)** changes four things. Every
> other part of this r2 text stands.
> 1. **pi_ref must stay inside both arms' data (new condition D5, bootstrap support).** Policy-evaluation critics
>    bootstrap through pi_ref's commands, so every one of them must lie in CUTOFF's retained support for the state's
>    load band. The first pi_ref clipped all four commands to +-KNEE. It met this at only 15-17% of next states:
>    its side thrusters went past the recorder's +-0.7, and its high-load a0 went above the window. pi_ref version 2
>    projects onto the support. It meets D5 at 100%, and its return is unchanged (-74.9 against -73.9).
> 2. **"Hidden" is exact (H1).** Under a fixed pi_ref, Q_stall and Q_nostall are both exact, and theta already uses the
>    no-stall world as the smooth counterfactual. So H1 is the L2.2 identity: stall equals no-stall at every supported
>    slot, to 1e-9 relative. H2 uses the exact gap rho_j = Q_nostall(sigma) - Q_stall(sigma). The fitted predictors
>    P1-P3 (rho_hat, item 5) and the sign control (item 6) are reported, never gated. Item 6 is reported as the max
>    over P1 and P3, with P2 at p10 shown separately as outside its fitting range, and with the exact sign share.
>    This is a change of estimator. It was prompted by two development item-5 misses: v2 step 0.4, and the first
>    knee_s10 screen at 43 of 48. Both stay in the ledger. The coherent gated form, |min_k P_k - Q_nostall|, also
>    gives 43 of 48, so the change rescues no recorded label.
> 3. **L1 and H2 use the same band.** L1 per anchor is -tol <= KC_L,j <= c(rho_j + tol), the V0(f) form. H2 is
>    R_req = (0.5 + c tol + z SD)/(1 - c).
> 4. **Formal G2 hygiene.** D4 needs two-sided six-SD Monte Carlo bounds computed for the candidate's own recorder.
>    96 gated anchors are used, not 48. Every label states what it leads to. A miss caused by a known estimator limit
>    allows no rerun. A reach forecast (no training) comes before any G5 compute.
>
> **An open scoring decision raised by r3.** The exact gap depends on the state. In the knee_s10 development screen it
> is +3.40 where pi_ref's unconstrained a0 is beyond -KNEE at the first continuation state (77% of anchors), and -0.41
> where it is not (there, the stall helps). The user decides whether the scored anchor set is all anchors (the median
> then carries the mix) or only the anchors where the continuation's own command exceeds the knee. The second is a
> state-conditioned relevance rule defined by pi_ref alone, never by a critic.

Written 2026-10-31. r1 came from a 13-agent diagnosis and design panel (workflow `wf_eccee84a-5c9`), whose synthesis
three adversarial checkers rejected; r1 took those corrections. Three further checkers (workflow `wf_35690ad8-e6b`)
rejected r1; this r2 takes every one of their corrections (the change log is at the end). Nothing here is approved,
sealed or run. It changes no sealed protocol; it applies to future candidates only, through an explicit amendment the
user writes. Numbers marked *development* come from `development_evidence.py` / `.json` in this folder (the v2
exploratory Stage 1 reports, outcomes known); they may shape a rule's form and never serve as a frozen threshold.

## 1. What was wrong with the old definition

The old definition (`knee_edge_ood_design_v1/DRAFT_PROTOCOL.md` Part 2, lines 107-252) decided "OOD" from what the
data lacked (L1: no CUTOFF rows; L2: a smooth fit would be wrong there) and then measured harm (L3). **It never
required that the same learner, given labels in the region, gets the value right**, so a CUTOFF-minus-RANDOM
difference could not be attributed to withholding. Two further defects: **the arm contrast carried a continuation
confound** (each arm scored against its own actor's continuation: PROTOCOL_V3.md R2), and **no feasibility
arithmetic** checked that the planned effect could reach its bar even with a perfect labelled control.

Development evidence on the knee edge (native host, 20k updates, each arm under its own clipped continuation):
- Resolution was not the problem: in-support action-contrast error at matched distance has a median of 0.05-0.09,
  and rho at s05 is 4.7-6.8 times its spread. In the no-stall world the critics follow the smooth value past the knee
  (to within 0.01-0.06 at s05; 0.03-0.11 at s10).
- The labelled twin did not learn the kink: RANDOM, with 9,530 high-load labels at a0 in [-0.9, -0.8) at 2.1x uniform
  density, captured 2-9% of the true drop (median per anchor), with a past-knee slope of the wrong sign; 4-13% of its
  anchors meet the L1 rule below.
- The withholding contrast was therefore capped near zero: theta = median_j[(KC_C - KC_R)_stall - (KC_C - KC_R)_nostall]
  = 0.090 / 0.031 / 0.007 at s05.
- The fold test G0.5 had predicted this for labelled slope-1.5 critics (median F 0.66 / 1.16 / 1.00); v2 removed it.
- The central quartile was an interior gap on a near-linear slope (interpolable); the stall band was bracketed by
  labelled bands and never reached.

## 2. Vocabulary: what a region IS versus what a learner DOES

| Term | Meaning | Established by |
|---|---|---|
| **Data-OOD** | W is absent from every stream the withheld arm touches, on one side of the retained actions, with action shift and not state shift at the scored states. | Construction, exact computation |
| **Testable** | Data-OOD, the labelled twin holds labels at the scored depth and load, and the feasibility arithmetic (H2) shows the effect could clear its bar if the twin learns. | Exact computation, before any training |
| **Label-dependent OOD for H** | Testable, and the labelled twin trained by H learns the hidden departure. | Learnability screen |
| **Exploited under PE(H, pi_ref)** (finding) | In policy-evaluation mode with the shared continuation pi_ref, withholding the labels makes the critic over-value the action contrast in W by the pre-registered margin. | Withholding gate |
| **Reached and harmful** (finding) | The native host's actor enters W because of Q (not BC) and entering costs return because of the hidden structure. | Native stage |

A valid-instrument miss at the withholding gate is the finding **"under PE(H, pi_ref), withholding does not produce
over-valuation of the action contrast here"**, never "not OOD" (as DRAFT_PROTOCOL.md:117 already said of L3). Findings
are always named with the mode they were measured in.

## 3. Measurement design

**3.1 Policy-evaluation (PE) mode for learnability and withholding.** Both arms' critics are trained with the
Unifloral TD3+BC critic loss with two declared departures from the native host:
- the target actor is one common, frozen, deterministic continuation pi_ref, clipped to +-KNEE on all four commands,
  and the actor is not updated;
- **target-policy smoothing is off (policy_noise = 0).** With smoothing on, pi_ref's target actions spill into W
  (pi_ref sits at -KNEE at high load and the noise SD is 0.2), so the critics would learn the value of a noisy policy,
  not Q^{pi_ref}, and CUTOFF's in-support values, the -KNEE reference included, would absorb its unlabelled W
  extrapolation. Smoothing is a host property; its effect is measured in the native stage, not assumed away.

Then both critics target Q^{pi_ref} exactly, which the raw float64 loop computes, and rho under pi_ref is an exact
pre-training quantity.

**3.2 pi_ref is a committed, hashed module before G0.** It states its linear model source, weights, the mapping from
the 14-number observation (13 state plus remaining time) to four commands, the float32 cast and the +-KNEE clip. The
clipped LQR behind the earlier 0.61 estimate (DRAFT_PROTOCOL.md:203) is a scratch script, not an artifact, and the
canonical core offers only a constrained planner; pi_ref must be written and reviewed. The runtime hook is a
parameter-ignoring `actor_apply_fn` wrapper (zero actor gradients, learner math unchanged) plus a new, tested path in
the rerun1 adapter and its update-counter checks.

**3.3 Arm contract: labels are the only difference.**
- **Twin = every CUTOFF row plus the W rows** (a superset), so in-support rows are identical.
- **One shared normalizer:** both arms use CUTOFF's observation mean and SD, frozen and recorded (Unifloral computes
  them from the training set and builds them into the critic, td3_bc.py:310, 71, 105).
- **Matched in-support exposure:** the twin's update budget is scaled by N_twin / N_cutoff, so each in-support row gets
  the same expected number of minibatch visits (uniform sampling, td3_bc.py:182-184).
- **Balance check (gated):** at -KNEE and at the matched in-support slot, the per-anchor |Q1_C - Q1_L| contrast error
  difference stays within the in-support resolution band from G1 (section 5).

## 4. Conditions, in the order they are checked

| id | Condition | Kind |
|---|---|---|
| D1 | CUTOFF holds 0 rows in W in every stream: fit, calibration, shadow, probe/scale, normalization. Three predicate implementations agree row for row. | Construction |
| D2 | One-sided: at every scored state the scored query lies outside the retained a0 range (no bracketed gap). | Construction |
| D3 | Action shift, not state shift, at the scored states (the L1.5 rule with amendment 1). | Exact computation |
| D4 | The twin holds labels at sigma's depth AND load band (the scored anchors' band), inside sealed six-SD bounds (the L1.4 method). | Exact computation |
| H1 | Hidden: rho(sigma) = min_k [P_k(sigma) - Q_stall(sigma)] under pi_ref, with P1-P3 as in L2.3, and item 5's estimator validity under pi_ref. A raw center-to-flank gap is never used. | Exact computation |
| H2 | Feasible: median rho(sigma) under pi_ref >= R_req(sigma) = (0.5 + c_s tol_s + z SD_theta) / (1 - c_s), z = 1.28. The SD-free part, (0.5 + c tol)/(1 - c), is checked at G2 before any compute: 0.90 at s05, 0.78 at s10, 0.70 at s20. | Exact computation (SD-free at G2; full after the variance pilot) |
| L1 | Twin learns the departure at sigma itself, per anchor and two-sided: -tol_s <= KC_L,j <= c_s rho_j + tol_s at a share of valid anchors >= q*, and the median inside the same band. | Learnability screen |
| E1 | Withholding effect: theta = median_j[(KC_C - KC_L)_stall - (KC_C - KC_L)_nostall] >= 0.5, under PE(H, pi_ref). | Withholding gate |
| E2 | Carried by CUTOFF over-valuation at the same anchors: among anchors with did_j >= 0.5, at least half have KC_C,j >= 0.5; theta is reported split into its CUTOFF and twin parts. | Withholding gate |
| N1 | Native agreement: in the native stage, the level contrast [Q1 - Q^pi](s, a) at visited states with a in W, CUTOFF minus twin, has the same sign as theta in every seed. | Native stage |
| R1 | Reach driven by Q: CUTOFF's entry rate minus a BC-only actor's rate on CUTOFF's own data >= 0.25 (sealed K2 bar, DRAFT_PROTOCOL.md:690); the twin's excess <= 0.10 (sealed V0(b) r_deep_max, DRAFT_PROTOCOL.md:686). | Native stage |
| R2 | Harm from the hidden structure: K3a, K3b, K3c >= +2.0 (sealed). | Native stage |

**Allowances.** c_s comes from the sealed softplus rounding rule (PROTOCOL_V2.md change 3) at sigma's depth and
w = 0.05, rounded up as v1 and v2 did: 0.4 at s05 (V0(b)), 0.3 at s10 (V0(f)), 0.2 at s20 (new; the rule gives
0.17). tol_s is item 5's tolerance: 0.10 / 0.15 / 0.30. **The share q\*, and any slope bar, are calibrated on G1** (the
supervised positive control): q\* is the largest share that a critic with G1's per-anchor error distribution passes in
every screen seed with probability >= 0.9 at the pre-registered anchor count. The r1 values (70%, slope 0.85) were taken
from development runs and are withdrawn.

**One scored slot per candidate**, named before any computation. Other slots are reported, never used to admit.

## 5. Gates and decision rules

Every gate reports every label it computes; the headline follows INCOMPLETE > INCONCLUSIVE > FAIL > PASS, and no
lower-precedence result is hidden.

| Gate | Rule | On failure |
|---|---|---|
| **G0 instrument** | The screen code reproduces the v3 records (V0(b), V0(f), development_evidence.json's per-anchor theta) to 1e-9; on a synthetic critic with a known kink it reports KC within 1e-6; pi_ref's module hash matches. | INSTRUMENT INVALID |
| **G2 exact** (no training) | D1-D4, H1, and H2's SD-free part under pi_ref. | NOT DATA-OOD, NOT TESTABLE, or INCONCLUSIVE (estimator) |
| **G1 representation positive control** | The 2x256 raw-action critic fit by supervised MSE to exact Q^{pi_ref} on the twin's (s, a) rows (no TD) meets L1 at sigma. It also fixes q\* and the in-support resolution band. | NOT REPRESENTABLE: change the feature or the host settings, not the data |
| **Variance pilot** | Screen-only seeds of both arms in PE mode; their outputs are used ONLY to estimate SD_theta (and SD of the twin's median KC), never for any admission decision, and are never reused. H2 is then evaluated in full. | NOT TESTABLE at this power |
| **G3 learnability** | PE mode, twin only, both worlds, screen seeds. L1 in every seed. | NOT LABEL-DEPENDENT FOR H: stop before any CUTOFF gate seed |
| **G4 withholding** | PE mode, both arms, both worlds, fresh gate seeds, held-out anchors (a named stream disjoint from development and screen anchors). Primary rule below; E2; L1 re-checked on the twin in every gate seed (a miss is INCONCLUSIVE, not FAIL); balance check (section 3.3); placebo equivalence: the pooled no-stall median contrast lies within +-tol_s with a pre-registered 90% interval. | Valid-instrument E1 miss: finding; else INCONCLUSIVE |
| **G5 native** | Native host (smoothing on), both arms, one BC-only actor per arm. N1, R1, R2. | Findings: not exploited natively / not reached / not harmful |
| **G6 ledger** | Every candidate screened, its sigma, mode and outcome, listed with any later efficacy claim. | PROTOCOL VIOLATION if omitted |

**Primary decision rule for G4 (pre-registered with both error rates).** Let theta_1..theta_n be the gate seeds'
per-anchor medians. PASS when the one-sided 90% lower bound, mean - t(0.90, n-1) x s / sqrt(n), with s the gate
seeds' own sample SD, is >= 0.5. Before any gate seed runs, n is chosen from the pilot's SD_theta so that (i) the
false-pass rate at theta = 0.5 is <= 0.10 (by construction of the bound) and (ii) the power at the reachable
alternative theta_alt = rho(sigma)(1 - c_s) - c_s tol_s, computed under pi_ref, is >= 0.8. If no n <= 6 achieves (ii),
the candidate is NOT TESTABLE at this power and no gate seed runs. (r1's "every seed" rule and its arithmetic were
wrong: P(every seed passes) falls as n grows, so it cannot be used to choose n.)

**Retries and search.** At most one rerun after INCONCLUSIVE, only for a named instrument cause, with fresh seeds and
anchors. Across candidates, the G4 false-pass budget alpha = 0.10 is spent over the candidates the ledger lists
(Bonferroni over the planned list, written before the first G4). The G4 claim is limited to "over-valuation of the
action contrast under PE(H, pi_ref)"; a native over-valuation claim also needs N1.

**Threshold hygiene.** Rules are frozen here. Values (q\*, the resolution band, SD_theta) come from G1 and the pilot,
never from the knee-edge development runs and never from a gate seed. Screen, pilot and gate seeds are never reused
for BCA efficacy.

## 6a. Candidate knee_s10_v1 (r3)

Folder `../knee_s10_candidate_v1/`. The recorder's high-load window is widened to [-1.0, -0.6], and sigma = s10.
pi_ref is the support-projected LQR. Both screens below are development screens on test entropy 424263 and stream
707,070,707; they are never confirmatory.
- **Under G2_RULE.md (pi_ref v1, item 5 gated): INCONCLUSIVE**, item 5 at 43 of 48, needing 44. This is on record.
- **Under G2_RULE_r3.md (pi_ref v2, exact H1/H2, D5 gated): PASS.**
  - D1-D3, D4 (2.46 / 2.51 against 2.00 / 1.82) and D5 (100%) pass.
  - The identity residual is 0.0.
  - Median exact rho(s10) is 2.32 against R_req 0.78. Its quartiles are 0.42 / 2.32 / 4.46, and the 10th percentile
    is -0.48.
- **Next, each step needing the user's approval:**
  1. adopt r3;
  2. decide the scoring set;
  3. seal the formal G2 (new entropy, new streams, 96 anchors, two-sided D4 bounds, committed pi_ref);
  4. run G1, the supervised representation check;
  5. run the variance pilot;
  6. run G3.

## 6. What this means for the knee edge (knee_edge_v1)

**The knee cannot host the test on its existing data, by exact arithmetic, before any compute.**
- **s05 fails H2's SD-free part.** A twin at the edge of L1 leaves theta <= rho(1 - 0.4) - 0.04, so H2 needs median
  rho(s05) >= 0.90. The clipped-LQR estimate under a shared continuation is about 0.61 (DRAFT_PROTOCOL.md:203); the
  learners' own continuations give 0.60-0.65 (development); only the old full-data actors' continuation reaches
  0.87-1.21 (step 0.4 `l2.*.item1_median`), two of three seeds still below 0.90 at the median.
- **s10 and s20 pass H2's SD-free part** (rho about 1.3 and 2.9-3.2 under the learners' continuations, against 0.78
  and 0.70) **but fail D4**: the twin has about 19 rows at depth >= 0.1 at high load (DRAFT_PROTOCOL.md:147), and every
  gated E2 anchor is high load.
- So a knee candidate needs **new data with deep high-load labels for the twin**: a re-collection (for example a high-load
  recorder window reaching a0 = -1.0), which is a new candidate in the ledger with its own sealed bounds, scored at
  s10, under pi_ref.
- Its sealed rules are untouched: V0(b), V0(d) (item 6 included) and V0(f) stay gated for knee_edge_v1 on this host
  (PROTOCOL_V2.md:149-151). Formally, step 0.4 is INCONCLUSIVE (item 5, seeds 1 and 2). The development evidence that the
  native twin captured 2-9% of the drop is a prediction, not a rule, and does not transfer to PE mode with
  smoothing off.

The two older conditions: the stall band fails D2 (bracketed) and R1 (never reached); the central quartile fails D2
and H1 (interior, near-linear, rho <= 0).

## 7. Known and unknown

- **Known:** the native 2x256 Unifloral critic, at 20k TD updates with smoothing on, did not learn a 0.6-1.4 kink past
  the knee even with 9,530 onset labels, while resolving smooth contrasts of that size.
- **Unknown:** whether that is representation (G1 answers it), target-smoothing spill into W (PE mode with smoothing
  off answers it), or under-training. Whether native exploitation (N1) appears where PE mode shows none: the native
  feedback loop (actor drift into W, bootstraps querying W) is absent in PE mode, so a PE miss is a finding about PE
  mode only.

## 8. Decisions for you

1. **Adopt v4** for future candidates (a written amendment; v1-v3 sealed rules unchanged).
2. **PE mode with policy_noise = 0** for learnability and withholding, and the **superset twin with a shared normalizer
   and scaled budget**: approve the engineering (pi_ref module, actor-frozen runtime path, tests).
3. **pi_ref:** a committed clipped-LQR module (recommended: analytic, no checkpoint, identical across arms) or one
   frozen learned actor.
4. **The next candidate:** the knee scored at s10 needs a re-collection with deep high-load labels. Alternatives: a
   different feature with a larger, smoother deviation, or host settings (Route B). Each goes through G2 first, which
   costs no training.
5. **Then G1** for that candidate (supervised fit on exact Q^{pi_ref} labels): about 76,800 rows times about 50
   remaining steps, roughly 3.8M batched raw-loop steps plus exact state replay of every row, and one small supervised
   fit. It needs your approval as compute.

## Change log

- **r2 (after `wf_35690ad8-e6b`):**
  - policy_noise = 0 in PE mode (target smoothing spilled pi_ref's targets into W);
  - shared frozen normalizer, budget scaled to the row ratio, gated balance check;
  - H2 corrected to include the twin's allowed slack (c tol) and made an SD-free exact check at G2;
  - the knee excluded at s05 by H2 and at s10/s20 by D4;
  - SD_theta from a CUTOFF-inclusive variance pilot used only for variance;
  - power rule replaced by a lower-bound rule with both error rates;
  - L1 two-sided at sigma with item 5's additive tolerance;
  - q* and slope bars calibrated on G1 instead of development runs;
  - E2 per anchor on E1's anchors;
  - native agreement check N1;
  - findings named by mode;
  - one rerun only, and Bonferroni over the candidate list;
  - pi_ref must be committed;
  - development SD ranges corrected (own clipped continuation, s20 included: clipped 0.04 / 0.07 / 0.24, own
    0.22 / 0.26 / 0.44 at s05 / s10 / s20);
  - the no-stall tracking figure qualified by slot.
- **r1 (after `wf_eccee84a-5c9`):**
  - vocabulary split;
  - PE mode;
  - superset twin;
  - E2 against a pessimistic twin;
  - learnability at sigma;
  - BC-only per arm;
  - softplus allowances;
  - sealed rules untouched;
  - one sigma per candidate;
  - ledger.
