# Revision log: DECISION.md after three reviews (20 October 2026)

Every finding of the three reviews (statistics S1-S11, fairness F1-F14, feasibility E1-E12) was re-verified against the
records on `B:\` (commit `52914e4` and the uncommitted build on branch `claude/near80-v9`), the study clone inside WSL
(read only; the watcher was alive), and the recomputation outputs beside this file (`review_stats_calc.out`, `c2_rule.out`,
`power_v9.out`, the new `c3_rule.out` and `ledger_v8_main.py`). Verdicts: **accepted**, **accepted with a change of
remedy**, **superseded** (already true when re-checked), or **rejected** (with the reason). The text before the
revision is `DECISION_v1_before_reviews.md`; `revise_decision.py` holds every replacement.

## Statistics review

| # | finding | verdict | verification and what changed |
|---|---|---|---|
| S1 | "two or more refusals: C1 fails by the lead arithmetic" is an expectation | **accepted** | `review_stats_calc.out`: resampled lead with ten seeds 26.3 / 18.8 / 8.4 / -2.1 at 0-3 refusals, P(lead >= 10 given 2) = 0.25. DECISION 2.3, 9.2 and the section-3 table now say C1 is read from the marks only, the refusal count is an expectation, two refusals are "expected to fail (about 1 chance in 4 of passing)". |
| S2 | a plumbing stop read two ways | **accepted** (reading fixed as: the marks fail by the pre-set rule, the verdict is "no reading") | 2.6, 8.1, 9, 14.1 (a stop clause and a two-form last sentence), 14.4 and the table's first row agree: a stop fails every mark that needs it (version 8's anti-rerun rule kept), the sentence says a stop decided it, the marks on the seeds that ran are description, and the verdict about BCA is "no reading". |
| S3 | "T ahead of BCA" labelled description only but used as "L2 contradicted" | **accepted** (pre-registered as C3) | A mirrored claim C3 per twin: mean d <= -5 and d positive in at most 1 of n certified seeds (coin 0.011 / 0.020 / 0.035 for n = 10 / 9 / 8); family false-alarm under a true tie 0.001 / 0.014 / 0.023 / 0.028 at spreads 5 / 7.5 / 10 / 15 (`c3_rule.out`, shared BCA side). The table's "L2 contradicted" row now requires C3. |
| S4 | "tie" read as "interchangeable" / "did as well" | **accepted** | Every tie is "was not shown to beat T by 5 points (mean, one-sided 95% lower bound, positive in k of n)"; the last sentences say "No difference of 5 points or more ... was shown" (2.6, section 3, 14.2). |
| S5 | L2 at the recorded effect is undecidable by this rule at any seed count | **accepted** | `c2_rule.out` n = 20: -4.5 reads "twin ahead" 0.33, tie 0.67; -3.0 tie 0.96. 2.4 "why not twenty" rewritten: the likely reading is "tie"; deciding a 3-4-point edge needs a smaller margin (a different claim), not more seeds. |
| S6 | bracketing holds on 4 of 5 preview seeds | **accepted** | PREVIEW 6.2: seed 14 BCA 2.256 against 2 x 0.987 = 1.974. Section 1 item 3 and 2.5 say "four of the five"; seed 14 named. |
| S7 | sign slip on the doubled ramp for seeds 11 / 14 | **accepted** | `review_stats_calc.out`: k2 minus k1 is -1.2 (seed 11) and +1.7 (seed 14). Section 1 L3 corrected. |
| S8 | range limit below the always-on ramp only in the main setting | **accepted** | near-80: range_hinge 54.7 against obca_forced 49.1. 2.5 says "on every recorded main-setting set" and names the near-80 exception. |
| S9 | C2's null rate quoted at sd 5 only | **accepted** | `c3_rule.out`: zero edge reads "BCA ahead" 0.001 (sd 5), 0.028 (sd 10), 0.044 (sd 15). Quoted at both spreads in 2.4, 2.6, 9.3. |
| S10 | per-seed refusal chance stated as 1 in 30 while factor 2.0 gives 1 in 23 | **accepted** | Factor 2.0 used throughout (1 in 23), with the 1.8-2.3 range given wherever a ten-seed chance appears (2.2, 9.2, 17.2). |
| S11 | "within every seed points rise with height" holds only up to BCA's height and not in near-80 | **accepted** | `review_stats_calc.out` dose ladders: seed 11 and 4 of 5 near-80 seeds break it beyond BCA; near-80 seeds 0 and 3 before it. Section 1 L3 reworded ("up to about BCA's, then flat; the near-80 setting is not monotone"). |

## Fairness review

| # | finding | verdict | verification and what changed |
|---|---|---|---|
| F1 | decision table keyed on the refusal count, not the marks; two cells missing | **accepted** | The table is re-keyed on (C1 pass / fail) x (0-1 / 2+ refusals) x (C2 / C3 / neither), with the two missing cells (C1 passes with 2+ refusals; C1 fails with 0-1 on a mark read from the numbers), the failed mark named, and the certified-seed marks as description. The deployment sentence stays as an extra sentence for 2+ refusals. |
| F2 | a single refusal decides C2 (a refused seed enters d at about -100) | **accepted, option (a)** | C2 and C3 are read on the certified seeds (at least 8), with sign thresholds ceil(0.8 n) for C2 (8 / 8 / 7) and at most 1 for C3; the refusal is charged in C1 and named in C2's sentence (`c2.clause.refused_seed` filled). Not outcome selection: the refusal depends on group size and BCA's draws, written before any arm trains. |
| F3 | "L2 contradicted" from a description-only reading over three twins without multiplicity | **accepted** | As S3: C3 pre-registered with the stricter sign count and the family rate stated. |
| F4 | tie wording; the conjunction's power unstated | **accepted** | As S4; 2.6 states the conjunction's power from `c3_rule.out`: three true 5-point edges 0.23 (sd 5) / 0.09 (sd 10); three 10-point edges 0.99 / 0.56. |
| F5 | twin 1 (split conformal) differs from BCA in its switch as well as its height | **accepted, first remedy** | PREVIEW 6.3: conformal wrongly on in no-stall seeds 10 and 13 under pooling. Twin 1 becomes `meandiff_q90x1` (the mean-difference switch with the plain 90% gap); `conformal_k1` stays reported with its false alarms counted. Cost: one more lone ramp program per seed-world (about 63 s / 40 s). |
| F6 | bracketing fails on seed 14 | **accepted** | As S6. |
| F7 | sign slip 11 / 14 | **accepted** | As S7. |
| F8 | range limit vs always-on ramp in near-80 | **accepted** | As S8. |
| F9 | "C1 unchanged in its marks" beside a change to the rival set | **accepted** | Version 8's PREREGISTRATION.md:389 lists the hand-set range limit as "reported; used in claim 3". 2.5, 2.6 and 9.2 now say "marks unchanged; the world-blind rival set gains the hand-set range limit (15.6)"; 15.6 adds that the lead is computed in version 9's look with step 2's lead reported beside it (step 2's `go_no_go` is pinned at `step2lib.py:54, 440-451`). |
| F10 | the noise floor does not remove a batch-vs-alone offset below 5 points from C2's differences | **accepted** | `policy_phase.ramp_layout` (243-251): twins train alone, `obca_k1` in the six-arm batch. C2 and C3 read `obca_k1_alone` as BCA's side; the noise floor stays as the validity check; the differences against `obca_k1` are reported (2.5, 2.6, 9.1, 15.9). |
| F11 | choice 2 offers two readings of a refusal; the document discusses three | **accepted** | Choice 2 lists the hard-clip / range-limit fallback as a third option labelled "a method change", with the stated ground for rejecting it (93.1 against 88.6; would net about 107 and reward refusing). |
| F12 | every seed is scored on the same four fixed start banks | **accepted as an owner's choice, default no** | `thruster_chain.bank_starts` (331-336): one stream `[20261012, 2, 9]` for every seed. Added as choice 10 (a fresh bank per seed, stream `[20261012, 2, 9, 1000 + seed]`, description only; about 1.5 min per seed-world plus anchors and code in `evaluate_phase.py`). Default no because the banks are symmetric across arms and pre-registered since version 6, and the owner asked for speed; the reviewer's stream `[20261012, k, 9]` is not used because its shape coincides with the log generator's 3-element streams. |
| F13 | the pre-approval kill rule is a seen-seed iteration on BCA's switch only | **accepted, with E2's correction** | The kill rule now runs on the spent seeds 10-14 from version 8's recorded second critics (nothing trained), on BCA's and the mean-difference switch's calls; the twins' calls and heights are pinned beside; the number of redrafts is recorded in section 0 (2.7, appendix 0, 4, 8.1, 17.9). |
| F14 | row 7 cites hosts v3's +6.54 without the same record's harm flag | **accepted** | `hosts_v3_results.json` qualities/medium-expert/claims/td3_bc/5: "Flagged as harm: standard BCA F2 on recorded +6.84 percentage points, lower bound +5.76; big waves +6.60; big kicks +2.22". Carried in the table's C1-fails row. |

## Feasibility review

| # | finding | verdict | verification and what changed |
|---|---|---|---|
| E1 | version-8 unit tests do not import against the rewritten modules; `test_v9.py` missing | **accepted in part (state moved on)** | Re-checked at 11:48 local: `test_near80.py` and `test_pooled.py` were retargeted after the review read them (11:39, 11:42) and `test_pooled.py` / `test_approval.py` import at module level in the scratch venv; `test_approval.py` still carries 13 lines of version 8's names (1662-1949) and `test_runner.py` 10+ (89-94, 150-186, 277, 566-567, 1151-1215, 1316); `test_v9.py` does not exist although `CODE_FILES` and `TEST_FILES` list it (constants 162-167), so the freeze would refuse. BUILD_LIST section 7 lists the lines and the new file. |
| E2 | the pooled switch on seeds 0-4 before approval cannot be done: step 1's second critics are not on the desktop | **accepted, remedy changed** | WSL clone: `runs/shift_explore/restructure/` holds `step2` and `step2_report` only; `runs/shift_explore/near80/learners/main/` holds seeds 10-14 only; step 1's results.json says machine "vm". The pre-approval check now reads the spent seeds 10-14 with version 8's recorded second critics (no training); retraining seeds 0-4's ten second critics (about 1 CPU-hour, exploratory) is choice 11, default no. |
| E3 | `meandiff_q90x2`, the no-caution reading and `range_hinge` in C1's rival set are not in the build | **accepted** | Confirmed: `TWIN_ARMS` at constants 239, `arm_row_from` 212-235, `ramp_layout` 243-251, `apply_failures` nulls BCA arms (pooled.py 323-336), `_claim1_numbers` uses step 2's pinned `go_no_go` (643-652; `WORLD_BLIND` hard-coded at step2lib.py:54). BUILD_LIST sections 2-4 give the changes line by line (now two new twins). |
| E4 | DECISION's appendix 14 uses labels the extractor does not accept | **accepted** | `claims_text._ANCHORS` (34-62) read. Appendix 14 rewritten in the extractor's labels, with the new sentences under new labels; BUILD_LIST section 5 lists the anchors to add. |
| E5 | folder and branch: the build supersedes in place; DECISION's default was `near80_v9/`; the owner's steps named files that do not exist | **accepted, default changed to in place** | Constants 95-98, `freeze.supersede_previous` (205-228), `approve_local` 70-81, `test_approval.py:338`. Choice 7's default is now in place (version 8's chain checkable at `52914e4`); 4.3 names `near80/approve_local_near80.py`, `approval_near80_<16>.json` and the branch `claude/main-bca-connection-x4o46s`; `claude/near80-v9` is to be merged into it before the freeze (the WSL clone is on that branch at `67209d2`, two commits behind origin's `52914e4`; `git pull --ff-only` works). |
| E6 | pre-freeze placeholders are wider than 4.2 step 1 listed | **accepted** | Counted: PREREGISTRATION.md 14 marks and `**DRAFT` on line 3, README.md 1 mark, prereg_draft.json 5 marks, constants `PLACEHOLDERS` 31-38, `OWNER_ANSWER` 122, `FRESH` 198, `MARKS_C2` 274, every module's first docstring line says DRAFT. 4.2 step 1 and BUILD_LIST section 1 list all of them. |
| E7 | the build's pre-registration states the wall time at about twice the ledger figure; DECISION's per-program cost was high and its cap off | **accepted** | `ledger_v8_main.py` on `results-20261020T115246Z/files/ledger`: seeds 11 and 14 cost 1.533 and 1.508 CPU-hours; policy.main.stall.s11 2,538 s, no-stall about 560-582 s; a lone ramp program about 63 s (lock `seconds`). DECISION: about 1.65 CPU-hours per seed, 18 in all, 5-5.5 h of wall in one sitting; the forecast constant stays 2.2 per seed and the cap is stated as 55 (2.5 x 22). BUILD_LIST corrects PREREGISTRATION.md section 13 and prereg_draft.json's compute block. |
| E8 | the results step keeps version 8's `.gitignore` gap | **accepted** | `.gitignore` line 7 `data/`; `near80_runner.py:1511-1514` prints a plain `git add`. BUILD_LIST: print `git add -f`; the owner's step 4 says `-f`. |
| E9 | parts of 2.6 / 2.7 are not computed (tie / behind readings, lower bound, switch table, ranks, ladder; the certification chance) | **accepted** | `pooled.marks` (782-816) reads "BCA ahead" only; `near80lib.certification_expectation` (165-176) records 1 - 0.9^n. BUILD_LIST section 4 lists each addition (the ranks need the sorted tail scores stored in the switch record). |
| E10 | cluster entry scripts drafted, untested, pinned | **accepted** | `TOP_FILES` (constants 160-161) lists both; neither has run. They leave `TOP_FILES` and the folder for version 9; section 12.2 of the pre-registration says so. |
| E11 | `run_near80.sh` and `push_runfiles_near80.sh` are version 8's unchanged files | **superseded** | Both were rewritten at 11:36 (after the review): the header says version 9, `version7-ended` is gone, the run-files commit message says "version 9" (line 55). No change needed. |
| E12 | the freeze must run inside WSL; `__pycache__` folders exist under the folder | **accepted** | `freeze.py:195-196` refuses them; `B:\...\near80\__pycache__` and `near80\code\__pycache__` exist (this step wrote nothing under `B:\`, so they are listed in BUILD_LIST for deletion). The freeze runs through the WSL watcher in the pinned environment after the branch lands. |

## Rejected or narrowed

- **F12's stream `[20261012, k, 9]`**: narrowed (see F12); the choice is kept with a 4-element stream.
- **E11**: superseded by the build's own later edit; nothing to change.
- No finding was rejected outright. Two were accepted with a different remedy than the reviewer's first suggestion:
  S2 (a stop fails the marks by the pre-set rule rather than giving "no reading" for them, so that no stop can be
  rerun into a pass; the verdict about BCA is still "no reading") and E2 (the kill rule moves to the spent seeds
  10-14, which need no training, instead of either dropping it or retraining seeds 0-4's critics).

## Changes not asked by a reviewer

- Twin 1 replaced (F5) implies four lone programs per seed-world and a fixed call order; 2.5, 5.3 and 4.1 say so.
- `V9_BUILD_NOTES.md` is named by constants (157), `approve_local` (86), the pre-registration (657) and the README (39)
  but does not exist in the folder; BUILD_LIST asks for it or for the references to go.
- The WSL clone is two commits behind origin (`67209d2` against `52914e4`), not one as the feasibility review said;
  `git pull --ff-only` still works.
