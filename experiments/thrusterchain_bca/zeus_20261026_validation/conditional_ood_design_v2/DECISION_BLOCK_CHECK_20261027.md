# Three-critic check of DECISION_BLOCK.md (conditional OOD design, v2)

**Status:** this was a document check only. No training, simulation, array decoding or protected-split reads took place. The form was not edited, and every user blank is still empty.

**Method.** Three critics read the form, each through one lens:
- **A, fidelity:** does the form match its sources?
- **B, decision logic:** are the rules complete and unambiguous?
- **C, provenance:** are the sources, receipts and approvals sound?

Each critic's findings then went to its own adversarial verifier. The verifier re-read the cited lines, re-ran the stated arithmetic and gave each finding a verdict and a severity. This report merges the duplicates and uses the verified severities. Where two verifiers rated the same issue differently, the higher rating is used and the split is noted.

Counts: 59 findings raised (A 21, B 19, C 19). 43 confirmed, 16 partly confirmed, 0 refuted, 0 unverified. After merging there are 7 blocking groups, 18 major groups and 14 minor items. I spot-checked the three most severe merged groups (sign, outcome overlap, freeze points) against the form and the sources myself. All three hold.

Terms used below:
- **BCA:** Bayesian Conformal Aggregation. Here it means an actor penalty of kappa × U added to Unifloral TD3+BC, with kappa = 3. U is BCA's uncertainty width.
- **Native:** unmodified TD3+BC.
- **q:** the critic's raw value for a command.
- **z:** the adjusted score, q − kappa × U.
- **v:** the measured return reference for that command.
- **MAE:** mean absolute error, the average of |score − v|.
- **Band:** first-thruster commands within h = 0.05 of the knee −k.
- **Hole:** the band at high load (g ≥ 0.4), withheld from training.
- **Gate:** a section-7 exposure check, read on native and distance-control outputs before any BCA result is opened.

## Verdict

No. The form cannot be frozen as written. Its main decision rule has three faults:
- The statistic has no fixed sign.
- The Pass, Fail and Inconclusive outcomes overlap.
- One branch, the "readiness check", is cited but never defined.

Freeze points also contradict each other. Section 10.7 lets S and the section 9 counts change after they are frozen. Several rules cannot be evaluated where they are placed: the support kill rule (stage 4 needs probes that exist only at stage 7), seed-failure handling, and the S ≥ 5 test. Separately, the form never mentions the v3 audit addendum committed alongside it. That addendum rejects this band as the next efficacy candidate and disputes several of the form's references.

Before any approval, the decision rule and the stage and freeze structure must be fixed, and the user must record whether this band proceeds at all.

## Blocking findings

| Ids | Form item | Problem | Correction |
|---|---|---|---|
| A2 = B1 | 3.1, 3.5, 3.8 | 3.5 calls the statistic a "Decrease in mean absolute error" but writes it as BCA's error "minus" native's error (DECISION_BLOCK.md:102). 3.1 says "BCA minus native" (:98). Read literally, the value is positive when BCA is *worse*, and Pass needs "the decrease exceeds 3.6" (:107). Example: MAE_BCA = 50 and MAE_native = 40 give +10, a Pass when BCA is worse. The review's wording has the same gap (REVIEW_20261027.md:50). | Write one per-seed formula with an explicit sign that matches "decrease", say what a positive value means, and use that symbol in 3.1, 3.5, 3.6 and 3.8. |
| A1 = B2 = C6 | 3.8 vs 7.6 vs section 8 step 3 | The outcomes are not ordered and can overlap. Fail (:109) needs every gate to pass but says nothing about 3.9. Inconclusive (:110) applies when any gate or 3.9 fails. Pass names no gate at all. So a run with 3.9 failed, all gates passed and the pass rule unmet is both Fail and Inconclusive. 7.6 says a 7.5 failure leaves only the return claim untested and "Forced-probe accuracy may still be reported" (:202-203). The primary *is* forced-probe accuracy. Step 3 (:210) uses a third wording. With 7.5 failed, the primary can reach Pass but never Fail, which favors BCA. That case is likely: v3 AUDIT_ADDENDUM.md:13 reports 0 of 59 high-load actor states in the band. | Rewrite 3.8 as an ordered, exhaustive tree: Inconclusive first, then Pass, otherwise Fail. Add a user choice for whether 7.5 applies to the primary. Make 3.8, 7.6 and step 3 name the same set of gates. |
| B3 = A10 | 3.8, 3.9, section 8 step 1 | Line 110 and step 1 (:208) refer to "readiness" checks in 3.9. 3.9 (:112-114) defines only a learning check. No readiness criterion, threshold or outcome exists anywhere. Step 1 lists "refusal and invalid counts", but no rule maps them to an outcome. v2 treats an unavailable BCA reference as a refusal or unready state (DRAFT_PROTOCOL.md:68) and requires a shared finite subset (:144). Verifiers split: A10 major, B3 blocking. | Add a readiness item with user blanks: what counts as ready, the maximum refused or invalid share per arm, whether the primary uses the shared finite subset, and which outcome each failure produces. Or delete "readiness" from 3.8 and section 8. |
| A3 = B6 = C4 | Rule 3 (:10), stage table (:41, :43, :46), headers of sections 9 and 10 (:215, :231), 10.7 (:255), 3.7/9.1 | Section 9 is approved at stage 1 but headed "fixed in stage 3". Stage 3 freezes "Every remaining blank", yet section 10 is headed "fixed in stage 6". 10.7 lets a stage-6 projection "reduce S or the counts in section 9". That breaks line 10 ("After a stage has run, its blanks never change"). It would happen after the stage-4 coverage counts are seen, and it can switch 3.8 between its S ≤ 4 and S ≥ 5 branches. S also has two blanks (3.7 and 9.1). This does not conflict with section 8's rule, which starts at stage 8. Verifiers split: C4 major, A3 and B6 blocking. | Give each section one freeze stage, and use it in the header, the stage table and section 12. Either move the budget sheet before S is frozen, or say that an over-ceiling projection requires a new labeled version under line 10. Keep one blank for S. |
| B4 | 3.8, 7.1-7.5, 10.7 | No rule covers failed seeds, or gates that fail in only some seeds. Pass is read per seed ("in every seed", :107). 10.7 stops timed-out jobs with no retry (:256). The existing matching code stops on degenerate norms (support_shift_pilot_v1/design.py:91-92). With one seed missing, "every seed" is undefined, and the handling could be chosen after outcomes are seen. | Add user blanks for: what counts as a seed failure; its fixed effect on 3.8 (for example, a non-pass seed, or Inconclusive); and whether each gate is read per seed or pooled. State that S never shrinks after training starts. |
| B5 | 5.9-5.11, section 8 | Section 5 is "checked in stage 4" (:135), and stage 4 decodes arrays only, with no model or simulator (:44). Yet 5.9 and 5.11 count "probe neighborhoods" and "in-band high-load probes" (:160, :163-165). Probe banks are simulator outputs (v2:107; 10.5 counts "state-bank generation"). v2:90 runs the stage-4 audit on a separate training anchor set. Section 8 has no support-labeling step. The form also never says whether support-unknown probes stay in the primary stratum. | Split the stage-4 anchor-set audit from probe-level support labeling. Add the labeling as an explicit section 8 step before step 2. Add a user choice for whether support-unknown probes are excluded from the primary. |
| B7 | 3.8, S ≥ 5 | The form does not say what the sign test or paired t tests (D > 0, or D > the 3.6 value), and gives no tie rule. The minimum one-sided sign-test p is 0.5^S: 0.03125 at S = 5. So any alpha at or below that makes Pass impossible at S = 5. At alpha 0.05, S = 5, 6 and 7 all need every seed to pass, so a smaller count blank has no effect. The count can also exceed S. If 9.9 is "shared", seeds are not independent (v2:148). | Name the tested quantity, the null hypothesis and a tie rule. Print the minimum achievable p for each S next to the blank. Say that an unreachable p or a count above S is invalid. Tie the test to the 9.9 choice. |

## Major findings

| Ids | Form item | Problem | Correction |
|---|---|---|---|
| A8 = C1 | Header (:13), stage 7 (:47) | The form names only the review and v2 as sources. v3/AUDIT_ADDENDUM.md, added in the same commit 850120f, says "reject the present band as the next policy-efficacy pilot candidate; keep launch blocked" (:3) and "do not spend the proposed policy-training budget on this candidate" (:67). Stage 7 still offers training on this band. | Add a status note citing the addendum by path and hash. Say it is an agent audit, not a user decision, and that the repo copy is an older snapshot to re-read. Add a section 12 row where the user records whether this candidate proceeds. |
| A9 = B8 = C15 | 3.1, 3.3, 3.5 | 3.5 compares BCA's adjusted z with native's raw q (:102). v2 keeps the same-critic q-versus-z panel apart from the trained-critic panel (v2:125-126). The review proposed absolute Q error only as "One candidate primary" (REVIEW:50). Where the critic overestimates, with error e > 0 and 0 < kappa × U < 2e, the subtraction alone lowers BCA's error, with no training effect. The form also omits v2's warning that each arm's v is not "one common target". Items 3.1, 3.3 and 3.5 come pre-written and offer only an optional "Alternative". | Turn the score choice into a user choice among equally listed options (q vs q, z vs z, z vs q) with no default. Label z vs q as combining a training effect with the subtraction effect. Add v2's common-target caveat. |
| A7 = C7 | Lines 12-13 and all reference columns | Line 12 says every analytic reference is a closed-form result from the recorder and start law. That is false for: 5.2 (a Gaussian approximation, REVIEW:41-43, :184); 7.1 (a hand model, REVIEW:57-60); 7.3 and 7.5 (development measurements); and 5.3-5.5 (v2 proposals). Two references, 4.4's 192 and 4.6's 25% and 46%, appear in neither source. They are correct, but the form's author derived them. No reference gives file:line. | Add a "source and type" label to each reference. Rewrite line 12. Apply "not a recommendation" to every reference. |
| C8 | 5.2, Option A (:139-147) | The 0.67-0.77 radius figure has no saved receipt. v3:12 calls it unverified and says "Do not adopt 0.7-0.75". Printing it beside the radius blank steers the user toward a radius that makes the support gate pass. Option A lacks v3's whole-episode exclusion for anchors (v3:12, :65). Its fixed same-load-side rule is disputed in REVIEW:68-69. | Label the figure as an unverified approximation and cite v3:12, or remove it. Add an item on whether the anchor's own episode is excluded. Note the dispute about the same-side rule. |
| C9 = B10 (7.1 part) | 7.1, 9.6 | The "1 to 1.5 units" reference is an unreceipted hand-model average across bands. 7.1 thresholds a per-state best-minus-worst spread between candidate families, which also include the far set and the actor command. v3:11 says no full-return bound is verified, and v3:23 says the saved bank has "zero same-state candidate pairs holding all three other action coordinates fixed". v2:84 requires the other three action coordinates to be held fixed, but 9.6 does not restate this. | Define the 7.1 spread exactly: which families, and whether it uses family means or single candidates. Relabel the reference. Restate v2:84's fixed-other-actions rule at 7.1 and 9.6. |
| A6 = C10 = B10 (7.3 part) | 7.3 (:195) | The reference "+19 to +82" is a signed per-command range, not an MAE, so it is in the wrong unit for an MAE threshold. v3:11 and v3:21 report signed −6.5421 to +81.9592 and MAE 47.48894 for 25 candidates from a broad-data, penalty-off checkpoint. v3's EVIDENCE.json is not in the folder, but both endpoint values do appear in the operator-alignment checks.json (:7337, :9852). | Give the reference as an MAE with its checkpoint, condition and row count. Show the review figure and the v3 figure side by side, and say they disagree. |
| A12 = B10 (7.5 part) | 7.1-7.5 | Only 7.4 names its data conditions. No gate names its checkpoint. "Native or distance-control actor" in 7.5 has no quantifier: either arm, both arms, or pooled. The pilot used "any" (support_shift_pilot_v1/evaluate_worker.py:79). | For each gate, add the data condition, the checkpoint and the arm quantifier, as fixed text or as user blanks. |
| A5 = C5 | 7.5 entry test (:197) | `abs(a0 + k) <= h + 1e-6` counts float32 −0.75 and −0.85000002 as in the band. v2:47 and the form's own 2.2 fixtures (:86) treat both as outside. The pilot added its 1e-6 in the stricter direction (evaluate_worker.py:36); 7.5 adds it in the looser direction. The numerical effect is tiny, about 17 float32 values per edge. Verifiers split: A5 minor, C5 major. | Use exactly v2's band test (≤ h). If a tolerance is kept, state its direction and the reason, and add the edge values as fixtures. |
| A4 | Stage table vs section 12 | Sections 1 and 2 have no blanks (:62, :82), yet they are listed as "Blanks approved". Stage 4 approves sections 4 and 5, which stage 3 already froze. Stage 7 lists sections 3, 7, 8, 9 and 10 (:47), but row 7 of section 12 lists only "run authorization" (:287). Section 7 is headed "read first in stage 8", but section 8 reads 3.9 first. | Separate "blanks frozen" from "permission to run". List only sections that contain blanks. Make the two stage-7 rows match. Fix the section 7 header. |
| C3 | Stages 4 and 5 (:44-45), line 5 | Stages 4 and 5 list approvals only for sections already frozen at stage 3. Only stage 7 names a separate run authorization. Line 5 says the form authorizes no decoding. So nothing names what authorizes array decoding at stages 4 and 5. (Their own section 12 rows and line 9 do still require an approval.) | Give stages 4 and 5 an explicit decoding authorization that names the arrays and episode roles, recorded in its own row. |
| B9 | 3.3, 3.5, 9.6 | It is unclear whether actor-command rows that land in the band count toward the primary. Those commands differ between arms. v2:146 makes actor rows a separate family. The form does not say whether the primary is one stratum or an equal-weight pool over the load-depth, depth and time sub-strata. v2:146 calls an empty stratum a design failure, but 3.8 gives that case no outcome. | State that the primary uses only the forced in-band candidates, or add a user choice. Name the pooling and weights. Give the 3.8 outcome for an empty stratum. |
| B11 | 3.9 | "In every seed and condition" does not say whether "condition" means the data conditions or the current conditions. Strong current never enters training (v2:132). No checkpoint or episode set is given. The distance control has no learning check, but it feeds gate 7.5. | Name the conditions, the checkpoint and the episodes. Say whether the distance arm gets its own check. |
| B14 | 3.2, 6.5 | If the hole-minus-random interaction is chosen as the primary, an imbalance above 6.4 is only "reported as an imbalanced control" (:184). Its effect on 3.8 is not stated. | State the 3.8 consequence, or add a user choice for it. |
| A13 | 6.1-6.3 | v2:82 requires random-removal count matching inside the calibration episodes. The form does not cover this, and does not say which episodes 6.3's stream covers. Answering "yes" to 6.2 would silently depart from v2:82. | Add an item for random-removal calibration rows, or label 6.2 "yes" as a departure from v2. |
| A14 = C11 | Glossary "v" (:25), stage 2, 10.5 | Reset always sets t = 0 (thrusterchain.py:434). The form states no continuation contract and no parity check at an interior time, both of which REVIEW:101-102 and v3:50-57 require. 10.5 omits the parity transitions that v2:154 counts. | Add the continuation contract as fixed text. Add a required interior-time parity check in a stage that runs the simulator, not stage 2. Add parity transitions to 10.5. |
| C12 | 4.5, section 11, 9.4 | The low-load in-band control cell appears only as a row-count minimum. Section 11 does not report it or the high-load out-of-band cell. v3:35-44 requires a four-cell recorded-current table and says the cross-load contrast is descriptive. | Add the four-cell table to section 11, split 9.4 counts by load side, and add v3's caveat. |
| C13 | Lines 50-51, section 12 | The form cites only ATTEMPT_HOLD.json, a 03:54Z snapshot. Later records show more: USER_PROCESS_CHECK.json says a fresh process check is still required; the retry-2 ARCHIVE.json records "user_canceled_after_seed0_warmups" and a request to move execution to Titan; README.md:1 says execution is moving to Titan. No section 12 row records how the hold is resolved. | Cite all of these records and give the current status in plain words. Add a hold-resolution row that names the machine. |
| A11 = C14 | Line 229, 1.1 | "Standing rules reserve seeds 10 to 14" cites no source. That reservation belongs to a different project's seed namespace (an earlier session's notes, not a file in this repo). The near-80 pre-registration has since claimed and used those seeds (near80/PREREGISTRATION.md:25; near80_results_check/checks.json:150). This design's seeds are large integers such as 2026102362, so the yes/no question rests on a stale premise. 1.1's "every local receipt" names no directories. | Replace the line with a sourced factual note, or remove it. Define the ledger's scope by explicit paths. |

## Minor findings

- **A15 = B19:** 4.6's blank is a row count, but its reference gives shares. The expected rows are about 559, 483 and 877 per stratum (start-law Monte Carlo × 1,920).
- **A16:** 5.2 does not say that v2 proposes 0.5. It also omits that the 0.4/0.6 sensitivity pair does not reach the review's range (REVIEW:45-46), and v2's tie-break by source row ID.
- **A17 = C17:** 7.5's "nearest 0.32 away" is the distance from the band center. It is 0.267 from the band edge. The high-load denominator should be 0 of 59 (v3:13), not 0 of 128. The reference is a single saved penalty-off actor.
- **A18 = C19:** the runtime estimate names its source but not its 12× linear scaling. It gives 4.25-4.72 h for two seeds, against the review's 4.3-5.0 h. The RUN_RESULT.json path is not cited.
- **A19:** 2.2 lists four edge values without saying which are inside the hole. "float32 observation sum" should say float64 sum of the float32 coordinates (v2:45).
- **A20 = C18:** "family" has two undefined senses (candidate family and test family). Far set, RMS, RNG address, readiness, fork, warmup, snapshot, Zeus and Titan are undefined. BCA is never spelled out. tau and eta are used (:262) before the table that gives them. The hole definition omits "executed". (U *is* defined, at :27.)
- **A21 = C16:** 5.7 and 5.8 show only expected counts. The review asks for the chance of an empty band: 0.0375 per band at n = 64, and 0.109 (n = 64) or 0.484 (n = 32) that at least one of three is empty (REVIEW:86-89).
- **B12:** 3.10's correction option names no test family. An alternative primary entered in 3.1, 3.3 or 3.5 would keep 3.6's unit and 3.8's rule unchanged. (Line 94 already stops secondaries from deciding efficacy.)
- **B13:** no mechanism enforces the BCA seal in stage-7 outputs: no separate files, hashes or frozen analysis script. This is a procedural gap only, because the rules freeze at stage 3.
- **B15:** the distance-control strength match has no item naming its probe set per data condition (design.py:89-93; v2:62).
- **B16:** 3.4 "other" allows 0 or 10k, where BCA's actor has had no BCA update (v2:130). It also allows checkpoints after the budgeted 20k.
- **B17:** S has two blanks approved at different stages. The 9.3 label "per trained seed" conflicts with the 9.9 option "shared".
- **B18:** 2.1 does not say whether an implementation bug may be fixed, or who classifies a disagreement between the two mask implementations.
- **C2:** section 12 has no typed approval statement or explicit file hash, as the user's earlier approvals had. Rows 2-7 lack `[ ____ ]` markers. "Approval commit identity" is undefined. (The Commit column does pin the content.)

## Refuted or withdrawn

No finding was refuted outright. These sub-claims were withdrawn or corrected on verification:
- **A3:** the 10.7 change does not break section 8's rule, which starts only at stage 8.
- **A5:** downgraded to minor, because the effect is about 17 float32 values per edge.
- **A9:** v2's larger-tau rule governs the secondary tie classification, not the MAE primary.
- **A11:** "only other hit" was wrong. Several outputs/shift files mention seeds 10-14, all from another study.
- **A13:** 6.3 does not say "training". It is ambiguous, not training-only.
- **A14:** the parity check does not belong in hand-built-only stage 2. Actor-command candidates are already in 10.5.
- **A18:** the form does name its source (two Zeus warmups).
- **A20:** U is defined in the glossary.
- **B6:** line 213 is a weaker, redundant prohibition, not a permission to change blanks.
- **B8:** the form does label each arm's v as its own. Only v2's common-target caveat is missing.
- **B12:** no contradiction with section 11. Line 94 already settles that secondaries do not decide.
- **B13:** downgraded, because the rules freeze at stage 3.
- **C1:** downgraded to major, because stage 7 still needs a separate authorization.
- **C2:** the git commit does bind the content. Only the typed statement and the markers are missing.
- **C3:** the stage-3 freeze alone does not authorize decoding, because stages 4 and 5 have their own approval rows.
- **C5:** downgraded to major, because the numerical effect is negligible.
- **C9:** the fixed-other-actions rule exists in v2:84. It is just not restated in the form.
- **C10:** −6.542… is also present in checks.json. Only the MAE figure lacks a saved match.
- **B10:** split across the 7.1, 7.3 and 7.5 groups above.

## What checked out

- **Section 10 formulas** match v2:152-154: 120,000 × S critic and auxiliary updates, 60,000 × S actor updates, 21 × S refreshes and 33 × S unique snapshots. C = 3, A = 3, the 10k warmup, the 20k end and checkpoints {0, 10k, 11k, 15k, 20k} also match.
- **Analytic references** 4.2 (1,920), 4.3 (about 606, 79%), 4.4 (192), 4.5 (1,920), 4.6 (29/25/46%), 5.7 (3.2), 5.8 (6.4) and 6.1 (3.2 of 128) reproduce under the documented recorder and start law.
- **Float32 values:** k = float32(0.8) (thrusterchain.py:79), g = c + w1, and the four 2.2 edge values are correct.
- **Citations:** the "v2 line 146" pointer is correct. The episode roles match v2:35 and data.py:43-46. The v1, v2 and AUDIT hashes match their receipts.
- **Rules:** the host is Unifloral TD3+BC only. No blank is filled. The form authorizes nothing. There is no automatic retry, radius relaxation, region search or redraw. The S ≤ 4 / S ≥ 5 split covers every S. Nothing is called confirmatory. Strong current is outside support by definition (v2:105).
- **Spelling** is American throughout.

## Fit with the v3 addendum

The addendum in the repo is an older snapshot. A newer version is being edited elsewhere, so this comparison holds only as of this commit.

**Where the form agrees with v3:**
- It does not call the band value-flat.
- It forbids relaxing the radius or searching over regions.
- It treats strong current as outside support.
- It adds a row minimum for the low-load in-band control cell (4.5).

**Where the form conflicts with v3 or leaves it out:**
- It never cites v3, and it offers stage 7 for a band that v3 rejects as the next efficacy candidate (A8 = C1).
- It repeats the 7.3 reference that v3 recomputed differently (A6 = C10).
- It repeats the 5.2 radius figure that v3 calls unverified (C8).
- It repeats the 7.1 flatness figure, which v3 says is not bounded (C9).
- It lacks whole-episode anchor exclusion (C8, A16).
- It lacks the fixed-other-actions rule at 9.6 (C9).
- It lacks the interior-time continuation parity check (A14 = C11).
- It lacks the four-cell reported table (C12).
- It uses the 0/128 denominator instead of 0/59 (A17 = C17).

v3's own recomputations cite an EVIDENCE.json that is absent at this commit. Its signed-error endpoints match saved JSON, but its MAE does not.

## Open questions no critic covered

1. **Scope of 4.1.** 4.1 forbids hole labels in "any fitting, calibration or matching bank". The full-data condition contains hole rows by design (v2:78). Does 4.1 apply only to the hole and random-removal conditions?
2. **Naming the three data conditions.** The form never lists the three conditions (full, hole, random-removal) in one place. They appear only by implication in 6.2 and 7.4.
3. **Candidate sampling.** How are the in-band, adjacent-band and far-set candidates in 9.6 sampled (grid or random, with what RNG address)? Where does the far set lie? Neither is defined.
4. **Practical size under the interaction primary.** Does the 3.6 practical size keep its meaning if 3.2 selects the hole-minus-random interaction, which is a difference of differences?
5. **The 3.9 baseline.** What is native's "untrained actor": the checkpoint-0 actor, or a fresh initialization? And on which return episodes is it measured?
6. **Seeds versus machine.** 1.1 allows seeds to freeze while the ledger is labeled "Titan collisions unverified", yet 10.6 allows execution on Titan. Must Titan manifests be imported before 9.2 can freeze if Titan is chosen?
7. **Gate failure and later stages.** When a stage-4 or stage-5 kill rule fires, does the form say plainly that stages 6-8 do not run, or only that the region is "shelved"?
8. **Re-check after v3 is updated.** No step re-checks the form after the newer v3 addendum lands, and there is no labeled-version rule for that case.

## Proposed edits to the form

None of these fills a blank. Where an edit adds a choice, the new choice is a user blank.

1. **Status note.** Add a status line naming `conditional_ood_design_v3/AUDIT_ADDENDUM.md` (path, commit and hash; older snapshot, re-read before freezing) and its rejection of this band. Add a section 12 row: "Does this candidate proceed? `[ yes / no ]`".
2. **Line 12.** Rewrite it so references are not all called closed-form. Add a source-and-type label with file:line to every reference cell, and apply "not a recommendation" to all of them.
3. **The statistic (3.1, 3.5).** Replace the wording with a single per-seed formula whose sign matches "decrease". State what a positive value means, and use that symbol in 3.6 and 3.8.
4. **Score choice in 3.5.** Replace the pre-written choice with a user choice among q vs q, z vs z, and z vs q, with no default. Add a note that z vs q combines the subtraction with the training effect, and add v2:126's "not one common target" caveat.
5. **3.8 as a decision tree.** Rewrite it as an ordered tree: (a) Inconclusive if readiness, learning or a named gate fails; (b) Pass if the pass rule is met; (c) otherwise Fail. Add a user choice for whether 7.5 applies to the primary. Make 7.6 and section 8 step 3 cite the same gate list.
6. **S ≥ 5 test.** Name the tested quantity, the null hypothesis and a tie rule. Print the minimum achievable p for each S (0.5^S for the sign test). Say that an unreachable p or a count above S is invalid. Link the test to 9.9.
7. **Readiness item.** Add a readiness item to 3.9 with user blanks: what counts as ready, the maximum refused or invalid share per arm, use of the shared finite subset, and the outcome for each failure. Or delete "readiness" from 3.8 and section 8.
8. **Seed failures.** Add an item that defines a seed failure and its fixed 3.8 consequence, as a user choice. State that S never shrinks after training starts.
9. **One freeze stage per section.** Give each section one freeze stage and use it in the headers, the stage table and section 12. Change 10.7 so that an over-ceiling projection requires a new labeled version. Remove one of the two S blanks and point to the other.
10. **Approval tables.** Separate "blanks frozen" from "permission to run". Add explicit decoding authorizations for stages 4 and 5 and a hold-resolution row. Make the stage-7 rows match. Mark every approval cell as `[ ____ ]`, and add a typed statement with the SHA-256 of the form and of v2.
11. **Support labeling.** Split section 5 into a stage-4 anchor-set audit and a probe-level labeling step, added to section 8 before step 2. Add a user choice for whether support-unknown probes stay in the primary.
12. **Gate scope.** For each of 7.1-7.5, add the data condition, the checkpoint, the arm quantifier and per-seed or pooled reading, as fixed text or user blanks. Define the 7.1 spread exactly. Replace 7.5's `<= h + 1e-6` with v2:47's test.
13. **References in 7.1, 7.3 and 7.5.** In 7.3, give an MAE-unit reference showing the review and v3 figures side by side. In 7.1, label the figure an unreceipted hand-model average. In 7.5, write "0.32 from the band center (0.27 from the edge); 0 of 59 high-load states; single saved penalty-off actor".
14. **Option A (5.2).** Label 0.67-0.77 as an unverified approximation (v3:12), or remove it. Add "v2 proposes 0.5", the tie-break by source row ID, a whole-anchor-episode exclusion item, and a note that the same-side rule is disputed. Add empty-band probabilities beside 5.7 and 5.8.
15. **Primary stratum (3.3).** State whether actor-command rows and which sub-strata enter the primary, with the v2:146 weights. Give the 3.8 outcome for an empty stratum and for an imbalance above 6.4 under the interaction primary.
16. **Learning check (3.9).** Name the conditions, the checkpoint and the episode set, and say whether the distance arm has its own check.
17. **Continuation contract.** Add the contract and an interior-time parity check in a simulator stage. Add parity transitions to 10.5.
18. **Section 11.** Add the four-cell recorded-current table with v3's descriptive caveat. Split 9.4 by load side. Add an item for random-removal calibration rows (v2:82).
19. **Seeds 10-14.** Replace line 229 with a sourced note on where the reservation came from and its later use by near-80. Define the 1.1 ledger scope by paths.
20. **Zeus hold.** Update the citation on lines 50-51 to list USER_PROCESS_CHECK.json, the retry-2 ARCHIVE.json, README.md:1 and titan_continuation_20261027/PROPOSAL.md.
21. **Wording fixes.**
    - Write "executed" in the hole and band definitions.
    - Label each 2.2 edge value as inside or outside.
    - Reword the float32-sum case.
    - Define the two senses of "family" and the other missing terms; spell out Bayesian Conformal Aggregation.
    - Move the tau and eta table before section 11's list.
    - Give 4.6's reference in rows.
    - State the 12× scaling and cite the RUN_RESULT.json path.
    - Limit or annotate 3.4's "other" option.
    - Reword 9.3 so it fits both 9.9 options.
    - Say who classifies a 2.1 disagreement.
    - Name the test family for 3.10.
    - Name the sealing mechanism and the distance-match probe set.
