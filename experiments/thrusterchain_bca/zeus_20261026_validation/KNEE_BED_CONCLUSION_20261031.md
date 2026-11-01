# The knee bed: what we found (2026-10-31)

The user stopped BCA work on this bed in a message at 19:50 EDT on 2026-10-31, after the three readouts below.

- **Status of the evidence.** Everything here is exploratory, on development data, and supports no protocol claim.
- **Scope of the stop.** It covers this bed only, not the BCA concept or other beds.

**Shelving summary (standing format)**

- **The idea:** place caution where the critic is wrong, using BCA's calibrated uncertainty.
- **What we tested (and the kill rule):** three exploratory readouts, each with its own rule written before it ran
  (sections 3 and 8). No kill rule was written for the bed as a whole; the stop is the user's decision after these
  readouts.
- **What happened:**
  - leverage: NO LEVERAGE;
  - dose screen: MIXED in both arms, so no claim;
  - Q-error signal check: NO SIGNAL.
- **Why:** section 4. Its reasons are descriptive and were not tested.

## Terms

### The setup

- **BCA.** The method under study: a penalty on the actor built from calibrated uncertainty about the critic's Q
  values.
  - The actor maximizes Q1 - kappa x U.
  - U = R x eta(s, a) x u. eta is a small network fit to the size of the critic's one-step (twin-min) residuals on
    logged training rows, R is one conformal quantile, and u is a running scale.
  - The configured dose is kappa = 3.
- **Host.** Unifloral's TD3+BC learner. Only Unifloral's learners were used; CORL, another offline RL code base, was
  not.
- **Q1.** The critic head the actor maximizes.
- **a0.** The first thruster's command, from -1 to 1.
- **High load.** Load g >= 0.4.
- **The bed.**
  - **Worlds.** A ThrusterChain world whose first thruster stalls past a knee: for |a0| > 0.8 it delivers less
    thrust, with slope 3. An exact twin of that world has no stall.
  - **Data arms.** CUTOFF has no training rows past the knee. RANDOM has labeled high-load rows just past it, with a0
    between -0.9 and -0.8.
- **OOD gates.** The bed's own pre-registered checks of whether withholding data past the knee makes the critic
  over-value it. OOD means "out of distribution".

### How results are measured

- **Anchors.** 48 fixed evaluation states.
  - At each anchor we compare the action 0.10 past the knee (s10, a0 = -0.9) with the knee edge (a0 = -0.8).
  - The comparison uses Q1 and the exact value Q^pi of each action when the actor then continues with its commands
    clipped to the knee. This is the clipped-continuation reference. Under the actor's own, unclipped continuation
    the values differ a lot.
  - **Harmful anchor:** one where s10 is truly worse under that reference.
  - **"Ranked correctly":** Q1, or the penalized Q1 - kappa x U, also says s10 is worse.
- **Seed-arm.** One training seed in one data arm.
- **Even caution.** Caution that is the same everywhere. In the anchor analysis below it is a constant penalty c on
  s10 relative to the knee edge, the same at every anchor: an "even ramp" in a0. As a training arm it would be a
  lower TD3+BC alpha. Neither was trained.
- **Data-only rule.** A penalty based only on where the data is.

## 1. The question

"Whether placing caution where a learned critic is wrong, based on calibrated uncertainty, beats spreading caution
evenly or simple data-only rules." This is the study's question, from Part 1 of the 2026-10-14 session history
(section "The project").

## 2. What the bed showed (numbers only: its gates were INCONCLUSIVE)

The bed's pre-registered OOD gates never passed:

- knee_edge_ood_v1, step 0.3: INCONCLUSIVE.
- knee_edge_ood_v2, step 0.4: INCONCLUSIVE.
- The exploratory Stage 1 run outside the protocol: INCONCLUSIVE, because V0(b), V0(d) and V0(e) failed in every
  seed (and V0(f) in v3's re-reading).

Under the v2 protocol, criteria from such a run are recorded but get no reading. The numbers below are therefore
shown as numbers only. All of them come from that run, under the native host:

- **Entry.** The actor went past the knee on 64-70% of high-load evaluation steps.
- **Return.** In the stall world it earned about 285-303 less return than the same actor with its commands clipped at
  the knee.
- **Anchors.** s10 was truly worse at 40-43 of 48 anchors, and raw Q1 ranked it correctly at only 7-21% of them.
- **Labels and the error at -0.85.** At -0.85, inside RANDOM's labeled band, RANDOM's critic over-valued by about as
  much as CUTOFF's (median 0.59-0.64 against 0.57-0.75, on the common anchors).
  - This is the V0(b) miss.
  - The protocol says such a miss cannot tell a critic that fails to carry the kink to high load from a correctly
    labeled critic that rounds it off.
  - At s10, RANDOM has almost no high-load labels: 28 rows lie deeper than -0.9.

So the bed has a costly critic error for a method to target. Whether caution placed exactly where the critic is wrong
would beat even caution here was never run. That is the program's "room" test: an oracle-placed penalty against even
caution and a data-only rule, at matched no-stall cost.

## 3. What the readouts show for this bed (no claim)

**No readout shows that caution placed by BCA's calibrated uncertainty beats even caution.** Data-only rules were not
tested, because no data-only arm ever ran.

| Readout (rule written first) | Verdict | What it says |
|---|---|---|
| Leverage at kappa 3 (`bca_leverage_readout_v1/`) | **NO LEVERAGE** | See below. |
| Dose screen at kappa 63 (`bca_dose_screen_v1/`, readout `bca_dose_screen_v1_readout/`) | **CUTOFF MIXED, RANDOM MIXED** (no claim) | See below. |
| Q-error signal check (`bca_q_error_signal_v1/`) | **NO SIGNAL** | See below. |

- **Leverage.**
  - **What was measured.** BCA's push was measured in shadow on logged training rows. At kappa 3 it would reverse
    1.5-3.1% of the actor's outward pushes past the knee, and its pull on a0 is 3-12% of the critic's.
  - **What the rule concludes.** BCA as built cannot steer the actor at its configured dose.
  - **Limits.** The analysis is linear in the gradient, uses training states rather than the anchors, and trained no
    BCA actor at kappa 3.
- **Dose screen.**
  - BCA never cut entry past the knee more where it hurts (stall) than where it is allowed (no-stall).
  - The penalized Q still ranked s10 wrongly at 55-88% of harmful anchors.
- **Q-error signal check.**
  - **What it measured.** D is the stall-specific part of Q1's one-step error on RANDOM's labeled training rows just
    past the knee: the onset-minus-edge contrast in the stall world, minus the same in the no-stall twin. It was
    0.172, 0.207 and 0.172.
  - **The verdict.** D reached the 0.20 bar in one seed, and the rule needed two. So by its rule, the track that fits
    U to the critic's Q error is shelved on this bed.
  - **The signal was there but small.** It was positive in all three seeds (no-stall control 0.003-0.021).

The table's "rule written first" can be checked by commit for the dose screen (3e05614e) and the signal check
(f79c4eb6). For the leverage readout, it rests on file times and the rule hash in its result.

**A descriptive comparison with even caution.** This was analyzed after the dose screen's result, on its fixed
BCA-trained networks, at the level of anchor rankings.

- **Shuffling U across anchors does as well as U itself.** Its placement adds nothing beyond its size.
- **With one knob for all seed-arms**, as kappa is actually used, an even penalty never did worse.
  - It did better at 79 of the 81 budgets from 0 to 80 no-stall false flips, and tied at 0 and 5. A false flip is a
    non-harmful no-stall anchor newly ranked s10-worse.
  - At BCA's own cost (43 of 220 such anchors), even caution newly ranked a net 74 of the 254 harmful stall anchors
    correctly, and BCA 47.
- **With a separate knob per seed-arm**, which favors BCA, the outcome is mixed. BCA was clearly ahead in one of six
  (CUTOFF s1), clearly behind in one (RANDOM s0), and within a few anchors in four.

## 4. Likely reasons (descriptive, analyzed after the results, not tested)

None of these was tested as a mechanism. They are readings of the records.

1. **U has little shape across actions.**
   - On the central-quartile bed (a different region, |a0| within 0.25 of zero), U varied by 0.5-0.8% across a0
     within a state and by 9-12.5% across states, so it acted as a per-state offset.
   - On this bed, the matching evidence is the leverage readout: BCA's pull on a0 is 3-12% of the critic's on logged
     rows.
   - The actor only feels U's slope in the action, so at kappa 3 that slope would reverse only 1.5-3.1% of its
     outward pushes.
2. **A higher dose adds strength, not information.**
   - At kappa 63 the penalty's action gradient was larger than native's on 74-90% of probe rows at the 10k fork, and
     entry past the knee moved by up to 0.48.
   - But the sign of U's slope varies by run, and shuffling U across anchors does as well.
   - Across anchors, U's rise past the knee is not larger where the critic's error is larger. The rank correlation is
     -0.15 to -0.50, but -0.31 to +0.14 once a state proxy is controlled, so the reason is unclear.
3. **The critic's error at -0.85 is local, and it shows only weakly in the one-step errors on training rows.**
   - In the stall world, -0.85 (the s05 slot) and -0.65 deliver the same thrust, so they share a next state.
   - Q1 gets that pair wrong by 0.53-0.67, which is about 0.8-1.1 times its over-valuation of -0.85 over the knee
     edge (post hoc, `bca_q_error_signal_v1/post_hoc/twin_split.py`). So at -0.85 the error is in the immediate step.
   - At s10 (-0.9) the over-valuation is about twice as large (native median 1.24-1.61), and this check does not
     split it.
   - On RANDOM's labeled training rows, D is only about 0.2, about a third of the error at -0.85. The simplest reading
     is that the critic fits those rows. Another reading, also untested, is that the anchor states lie away from the
     labeled rows' states.
   - If the first reading holds, eta would see only the small in-sample part, because it is fit to the size of
     twin-min residuals on training rows. The check measured Q1's signed error, not eta's target, so this is an
     inference.

## 5. What this does not show

- **No claim.** No result is confirmatory. The three seeds share one dataset and one anchor set, and every readout
  used development forks with known outcomes.
- **One bed, one host, one algorithm.** IQL, CQL and ReBRAC were not tested. The general claim is unsupported here,
  not refuted.
- **The even-caution comparison is an anchor-ranking analysis on fixed networks.** It says nothing about what an
  actor trained with an even penalty would do, or about returns.
- **No data-only arm and no even-caution training arm ever ran.** What a data-only rule would have done here is an
  expectation, not a run:
  - In CUTOFF, which has no rows past the knee, it would penalize every past-knee action.
  - At the onset in RANDOM, where 9,530 rows exist, it would see nothing and so act about like native. That is the
    "labeled but wrong" case the question is about (`EXPERIMENT_DIRECTION_20261031.md` section 5).
- **Two readings untested.**
  - Whether held-out (cross-fitted) residuals, meaning residuals computed on rows the critic did not train on, would
    see the anchor error.
  - Whether the anchor states simply lie away from the labeled rows' states.

  Testing either on this bed would be a rescue designed after seeing D.

## 6. What stops with this decision

Nothing is deleted, and every record stays as it is.

| Package | State |
|---|---|
| `knee_edge_ood_v3/` (seven repairs, R1-R7) | Committed (cdebf428, fc12abf0), never approved for any run; its GPU smoke never ran. It carries v2's step 0.4 record (INCONCLUSIVE), so its later steps could not be requested without an approved amendment. Its re-reading of the exploratory Stage 1 (`amended_analyses/`, still INCONCLUSIVE) stays on record. |
| `knee_edge_ood_v2/` A4 (Stage 1) and A5 (the sealed Stage 2) | Already blocked by step 0.4 (INCONCLUSIVE) before this decision. A4 never ran under the protocol; an exploratory Stage 1 ran outside it (INCONCLUSIVE). A5 never ran. It had no even-caution or data-only arm, and its "distance" arm is a proximal control. |
| `knee_edge_ood_v1/` untracked A4-A5 drafts | Never run; v1 ended at step 0.3 (INCONCLUSIVE). |
| `KNEE_EDGE_V2_EXPLORATORY_STAGE1_READOUT_PLAN.md` | Uncommitted; its open decisions (its section 8) stay undecided. |
| `ood_definition_v4/` (DRAFT r3, never approved) | Paused; uncommitted. It is a study-wide draft, so this stop pauses only its use on the knee. |
| `knee_s10_candidate_v1/` | Paused; uncommitted. Its development gate (G2) was INCONCLUSIVE under the first rule and PASS under the third revision (r3), which was written after that result and reused the same test stream. It was never run formally. |
| The leverage rule's follow-ups (U at the evaluation anchors, the room screen), the fresh-seed higher-dose screen, and a Stage 2 with even-caution and data-only arms (`EXPERIMENT_DIRECTION_20261031.md` section 7) | Never written or never run. |
| The Q-error variant (`bca_q_error_signal_v1/METHOD_DESIGN_NOTES.md`) | Already shelved by its own rule (8faf9e09); never built. |

## 7. Lessons for any next bed

1. **Check for room first.** Before building anything, show that an oracle-placed penalty beats even caution and a
   true data-only rule at matched no-stall cost. An oracle-placed penalty is one placed exactly where the critic is
   known to be wrong.
2. **Measure the identifiable error where it matters.** That means at the evaluation states, or on held-out rows, not
   only in-sample.
   - Include a positive control: a case built so that the check must fire.
3. **Judge placement against matched nulls**, not against zero. Matched nulls are penalties of the same size with no
   information in their placement: U shuffled across states, an even ramp, and U with its sign flipped.
   - Check every rule on hypothetical estimators first: an oracle must pass, and raw Q1 and an even ramp must fail.
4. **Include the comparison arms the question names**: even caution and a data-only rule, at matched cost.
5. **Use fresh seeds with their own data.**
   - Return bars need a spread for the paired difference.
   - Every return statement should name its checkpoint.

## 8. Records

| What | Where | Committed? |
|---|---|---|
| Leverage readout | `bca_leverage_readout_v1/` (RULE.md, leverage.py, leverage_result.json) | yes (72eca0ca, after its run) |
| Direction note | `EXPERIMENT_DIRECTION_20261031.md` | yes (72eca0ca) |
| Dose screen package and approval | `bca_dose_screen_v1/` | yes (3e05614e) |
| Dose screen run records, including SCREEN_RESULT.json | `bca_dose_screen_v1/screen_run/` (checkpoints git-ignored) | yes (72eca0ca) |
| Dose screen readout | `bca_dose_screen_v1_readout/` | yes (5bd83a23) |
| Signal check, rule first; its result and shelving note | `bca_q_error_signal_v1/` | yes (f79c4eb6; 8faf9e09) |
| Bed and gates | `knee_edge_ood_v2/`: the package and the v1 step 0.3 record (a286fee5), step 0.4 and the exploratory Stage 1 (03cb0e6e); `knee_edge_ood_v3/` (cdebf428, fc12abf0) | yes |
| This note | `KNEE_BED_CONCLUSION_20261031.md` | yes (db2e2c36) |

The leverage rule was committed only after its run. Its "written first" rests on file times (RULE.md 09:53:22, leverage.py 09:53:59,
result 09:54:33 EDT) and on the rule hash recorded in its result, not on a prior commit.
