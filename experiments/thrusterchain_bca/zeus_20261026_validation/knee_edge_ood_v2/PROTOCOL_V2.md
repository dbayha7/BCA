# knee_edge_ood_v2: what changes from v1, and why

**Base.** v2 is `knee_edge_ood_v1/PROTOCOL.md` (revision 13) with `PROTOCOL_AMENDMENT_1.md`, except for the changes
below. Every rule, bar, bank, seed, world, budget and label not named here is v1's, word for word. v1 stays sealed in its
own folder and is reported as it ended.

## Why v2 exists (a post-hoc protocol, disclosed)

- **v1 ended at step 0.3.** A2 ran once (attempt 1, approved and committed by the user; report sha256
  `6c15575541f86ba024949655b987d27bd5f9288d242a428de0df92fedc504215`) and ended **INCONCLUSIVE**:
  "under 25% of steep ordinary probes have F <= 0.5 in seed 2".
  - Full-data old critics, 16 steep ordinary probes per seed: median F 0.66 / 1.16 / 1.00; share with F <= 0.5
    0.44 / 0.25 / 0.125 (seeds 0 / 1 / 2).
  - The known-answer reproduction passed (scaled error 0.0 in all 9 cells) and the past-knee recount passed.
  - The reach-forecast values exceed v1's bars in every cell (probe 0.625-0.6875 against 0.35; visited 0.64-0.90
    against 0.25), but v1's label never evaluated them, because the G0.5 stop comes first.
  - The onset contrast, v1's early warning for V0(b) and its only planned measurement of KC's noise floor, never ran:
    it runs only after a gated pass. KC noise on real critics is therefore still unmeasured.
  - An independent audit (code path, recomputed numbers, protocol reading, and an adversarial reviewer) found the
    computation correct and the label robust: seed 2 has 2 of 16 probes at F <= 0.5, and the next value is 0.528.
- **Under v1 that stops the line.** v1's table says "Stop; the region is not declared non-OOD". 5.0 says "No outcome may
  change the recorder, world, arms, seeds or thresholds" and "After an INCONCLUSIVE: the step stops. It cannot be
  re-attempted under the same design." v2 is what those rules forbid *within* v1. It is legitimate only as a new,
  disclosed, post-hoc protocol, with v1 reported as INCONCLUSIVE at step 0.3. v2 does not rerun step 0.3 and does not
  change its label.
- **The reason for v2.** G0.5 tested a *proxy*: old critics trained in the `stall15_identity` world, where full
  throttle folds thrust back to 0.5 (T15(-1) = -0.5). The experiment's critics are trained in `stall3`, where full
  throttle folds thrust to 0.2 (T3(-1) = -0.2, L2.1: Q(-1) = Q(-0.2) - 0.288), so the value gap at the fold is about
  twice as large. v2 removes the proxy gate and puts a knee check on the real slope-3 critics in Stage 1 (change 3).
- **This is post hoc.** The change was chosen after seeing the A2 result, to continue a line that v1 stopped. The one new
  threshold (V0(f)) is derived from v1's own pre-registered rounding rule, not from A2's numbers. No v1 Stage 1 or
  Stage 2 rule is loosened. A reader should weigh any v2 PASS with that in mind.
- **Correction recorded with the disclosure.** The old "hole" critics withheld the central quartile (|a0| <= 0.25),
  not the knee. All nine old critics had past-knee labels (about 10% of high-load rows; hole 118 of 1,140). A2 shows
  that labelled slope-1.5 critics extrapolate through the fold (Q1(-1) > Q1(-0.5) at 116 of 145 steep probes over all
  three arms). It does not compare critics with and without knee data.

## What was seen, and what is untouched

- **Seen:** v1's A2 report, `gated.json` and the nine cell files, including the reported (not gated) hole, random and
  strong-current F values. These are old critics on old data; they are development data for v2.
- **Untouched:** no step 0.4, Stage 1 or Stage 2 stream, bank, anchor, start or training seed has been drawn or
  evaluated. A3 never ran. The A4 GPU smoke test used test-only data, streams and seeds, checked distinct from every
  sealed seed (`stage1_smoke.py` `smoke_training_seeds`), trained no RANDOM run, and its worker reports, which hold
  values on replicate data, were never read; only the blinded `SMOKE_RESULT.json` (structure, timings, checks) was.
- **Smoke provenance.** The smoke ran from the v1 folder on 2026-10-30 at 22:36Z, at the user's terminal, with no
  receipt inside the repository (its output is outside it by design). Change 6 below edits `stage1_worker.py`, so that
  smoke no longer certifies v2's worker: **the smoke is rerun from the v2 folder before A4 is approved**, and its
  blinded `SMOKE_RESULT.json` is copied into `knee_edge_ood_v2/` and committed as the receipt. The v1 smoke stands as
  the pilot that exercised the training and evaluation code and the projection of the training time.
- **External audit, 2026-10-30 about 20:30 EDT.** A reviewer outside this work audited the uncommitted packages and
  found: the reference mixes critic error with a different policy's value (change 7); step 0.4's transition budget
  omitted replays and its hour was not a hard cap; source-integrity checks let workers load live files; a precheck
  charged no transitions and a missing report charged zero; the smoke is a pilot without a receipt; and v2 changes a
  stopping rule post hoc. Changes 6 and 7 answer the code points; the rest is disclosed above.
- **Seeds.** v2 reuses v1's sealed seeds (`seeds.py`, derived from `sha256("knee_edge_ood_v1/" + name)`), unchanged, so
  the A1 manifest stays byte-identical. They were fixed before A2 and none has been drawn.

## The changes

1. **Step 0.3 is carried, not rerun.** v2's A3 (`step04.a2_outcome`) reads v1's single step 0.3 report from
   `knee_edge_ood_v1/step03_run/attempt_1/report.json`, pinned by its sha256, and requires:
   - the report was produced under the committed A2 package;
   - its label is PASS, or INCONCLUSIVE with **every** reason being G0.5's share rule;
   - the known-answer reproduction and past-knee recount passed;
   - the A3 approval carries the acknowledgement below.
   The reach forecasts are recorded, not used for admission: using the slope-1.5 proxy only where it passed would be
   selective, and Stage 1's R0 tests reach on the real critics. v2's copies of `step03.py`, `MANIFEST.json` and
   `A2_MANIFEST.json` are hash anchors only; v2 refuses to run if a `step03_run/` or `A2_APPROVAL.json` exists in it.
2. **Written predictions.**
   - At A3: "I acknowledge that knee_edge_ood_v1 step 0.3 ended INCONCLUSIVE on G0.5 and the written prediction:
     Stage 1 V0(b) or V0(d) is likely to end INCONCLUSIVE".
   - At A4, replacing v1's: "I acknowledge the written predictions: Stage 1 V0(d) may end INCONCLUSIVE, and after
     the v1 step 0.3 INCONCLUSIVE, Stage 1 V0(b) or V0(d) is likely to end INCONCLUSIVE". V0(f) is part of the
     same prediction: a RANDOM critic that rounds the knee misses V0(b) at s05 or V0(f) at s10.
   - **Why "likely".** All nine labelled slope-1.5 critics extrapolated nearly linearly through the fold, with
     probe-to-probe F spread of about 2.1-2.8. If slope-3 RANDOM critics round the knee the same way, they miss V0(b)'s
     or V0(f)'s allowance, or eat K1a's contrast term in V0(d)(i). The deeper slope-3 fold carries more loss weight and
     may be learned; that is the reason the proxy may not transfer.
3. **New Stage 1 validity item V0(f) (INCONCLUSIVE): RANDOM folds at K1c's slot.**
   - **Why.** V0(b) tests RANDOM at the onset s05, which guards K1a, a CUTOFF-minus-RANDOM contrast. K1c is absolute,
     at s10, and has no RANDOM term, so it is the criterion G0.5's confound ("approximation failure, not withholding")
     threatens. Without V0(f), removing G0.5 would replace it with nothing.
   - **Rule**, per seed, on the R-valid anchors (predictor-valid under RANDOM's own clipped continuation), using
     quantities Stage 1 already computes:
     - resolvability floor: median rho_hat_R(s10) >= 0.30 (twice item 5's 0.15 tolerance at s10, as V0(b)'s floor is
       twice its s05 tolerance);
     - one-sided bound: median KC_s10(RANDOM) <= 0.3 x (median rho_hat_R(s10) + 0.15).
   - **Where 0.3 comes from (v1 Part 2, the softplus rounding rule, not A2).** A correct critic that rounds the slope-3
     kink with softplus width w has KC/rho = w[d/w + ln 2 - ln(1 + e^(d/w))]/d at depth d past the knee. At s05
     (d = 0.05) this gives v1's 0.25 / 0.31 / 0.35 / 0.38 for w = 0.02-0.05, which set V0(b)'s 0.4. At s10 (d = 0.10)
     and w = 0.05 it gives 0.283; V0(f) rounds it up to 0.3. The +0.15 is item 5's s10 tolerance, used at its upper
     end as V0(b) uses 0.10.
   - **What it guarantees.** With the floor, 0.3(rho + 0.15) < 0.5 rho whenever rho > 0.225, so a RANDOM critic that
     passes V0(f) could not itself satisfy K1c's relative clause. A CUTOFF K1c pass then exceeds what the labelled
     control does at the same slot.
   - **One-sided** because the confound is over-valuation; a pessimistic RANDOM critic is reported, not penalized.
   - A miss in any seed makes Stage 1 INCONCLUSIVE, as for every V0 item; all criterion outcomes are still recorded.
   - **Disclosed choice:** gating at s10 rather than at -1 (s20) was decided after A2. It rests on s10 being K1c's
     slot and the edge of RANDOM's dense high-load labels ([-0.9, -0.8]); RANDOM's labels past -0.9 at high load are
     sparse, so a gate at -1 would test cross-load transfer, not the knee.
4. **Reported, not gated.**
   - Per seed and arm: median KC_s20 (at -1.0, bits 0xBF800000) and the share of common gated anchors with
     KC_s20 <= 0.5, next to v1's G0.5 numbers: the slope-3 analogue of G0.5.
   - V0(f)'s quantities for CUTOFF too, for comparison.
5. **Identity.** Approval words name `knee_edge_ood_v2`. v2's A3, A4 and A5 manifests are new and pin each other as in
   v1 (A4 pins v2's A3 manifest, A5 pins v2's A4 manifest). v1's A1 and A2 manifests are carried byte-identical.

6. **Caps and integrity (after the external audit).**
   - **Exact simulator counting.** `units.SimulatorCounter` wraps both paths the package uses to move the simulator:
     `env.step` on environments made through `worlds.environment`, and the canonical `dynamics()` function the raw
     float64 loops, the lineage replay and the thrust check call directly (counted per state row). So every
     transition of a process is counted: collection, the lineage replay, anchor rebuilds by prefix replay,
     continuations on both evaluators, the cross-check and free rollouts. The step that would cross the
     cap raises before it runs. Each report records the exact count (`transitions`) and the former formula charge
     (`transitions_prospective`). In Stage 1 and Stage 2 an evaluation job's exact count is what the stage cap is
     charged, and a job without a report charges its whole allowance. Stage 2's precheck, which rebuilds 64 anchors
     (2,400 steps) and charged nothing, is now counted.
   - **Step 0.4 cap.** v1 wrote "2.5M transitions or 1 hour" and charged about 2.1M by formula; counted exactly, the
     step takes about 2,587,760 steps (measured in the Windows dry run of the same code on a 1,280-episode
     test collection). v2's cap is 3,000,000 simulator steps, the exact count plus margin, or 1 hour. A
     watchdog thread writes the attempt INCOMPLETE and ends the process at 3,600 s whatever it is doing;
     `Budget.check_time`, which only runs between charges, stays as the early check.
   - **Worker-side integrity.** The supervisor writes into every `job.json` the sha256 of every package source,
     manifest, seed, protocol and approval file and of rerun1's reused sources (`units.integrity_record`). Before a
     worker imports any package module it re-hashes the same paths (`integrity.py`, standard library only) and
     refuses to run on any difference, so a file changed after the supervisor's check is never loaded. The one file
     the check cannot vouch for is `integrity.py` itself; its hash is in the manifests and in the record. The
     Unifloral sources are covered as before: `td3_bc.py` by the preflight source receipt and `bc.py` by the BC-only
     job's hash pin.
7. **Reference definition, stated in the claim scope.** KC is measured against the exact Q^pi under the actor's own
   continuation **clipped to ±KNEE on all four commands** (v1 Part 2), so the reference never stalls and a one-step
   knee measurement is not mixed with downstream stall behavior. The critic, however, estimates the value of the
   actor's unclipped continuation. The resulting bias in KC is the difference of two continuation-value gaps at
   next states that differ only through one thrust change, and the same bias enters CUTOFF and RANDOM alike, so it
   cancels in K1a's contrast; it does not cancel in K1c, which is absolute. The unclipped Q^pi is computed and
   reported at every anchor (v1 Stage 1, "exact own-continuation Q^pi, clipped as primary and unclipped as a
   report"), so the size of the bias is on record. v2 keeps the clipped reference as primary; a reader of a K1c
   result should read the unclipped companion with it.

## Claim scope

- **v2 PASS (Stage 1):** at slope 3, the native Unifloral TD3+BC critic over-values the knee-edge region beyond what a
  labelled control does at the onset (K1a) and at s10 (K1c with V0(f)), its actor goes there (K2), and that costs return
  because of the hidden knee (K3). It permits only the request for Stage 2. It says nothing about slope-1.5 critics,
  which A2 showed do not learn the fold.
- **v2 INCONCLUSIVE at V0(b), V0(d) or V0(f):** the second independent sign, now on the real critics, that this host
  cannot separate approximation error from withholding at the knee. For the OOD line of work that means the knee-edge
  region is not usable as a decision-relevant OOD test with this host and network; the next step is a host change
  (Route B), not another threshold. No later version may remove, weaken or re-threshold V0(b), V0(d) or V0(f) on this
  host.
- **v2 FAIL:** read with v1's readings 1-6, unchanged.

## Attempts (5.0)

v1's step 0.3 attempt is final and is not counted again. v2's A3, A4 and A5 start their own attempt counts under v1's
5.0 rules, unchanged. v2's approval files live only in `knee_edge_ood_v2/`, and their words name v2, so a v1 approval
cannot authorize a v2 step.

## Budget

Unchanged from v1. The A4 GPU smoke projected Stage 1 training at 11,853 s against the 18,000 s cap.
