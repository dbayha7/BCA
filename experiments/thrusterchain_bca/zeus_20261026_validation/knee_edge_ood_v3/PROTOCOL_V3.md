# knee_edge_ood_v3: repair protocol on top of PROTOCOL_V2.md (a disclosed engineering version; no new science)

Written 2026-10-31 (EDT) from the read-only audit `analysis/ood_progress_audit_20261030_v1/AUDIT_AND_REPAIR_PLAN.md`
(October 30, 10:05 p.m. EST snapshot). It applies on top of `PROTOCOL.md` r13, `PROTOCOL_AMENDMENT_1.md` and
`PROTOCOL_V2.md`, which it does not edit. Every v2 file is carried byte for byte except where a change is listed
below. **This package is not approved for any run. No approval file exists in it. It launches nothing.**

## Why v3 exists, and what it is not

v2's step 0.4 (A3) ended INCONCLUSIVE on L2 item 5 (seeds 1 and 2 at 42 of 48 predictor-valid anchors, bar 44), so
the gated line stopped there under 5.0, and the user then launched an exploratory Stage 1 run outside the protocol
(`knee_edge_ood_v2/stage1_exploratory_run`, disclosed in its approval and in every report it writes). The audit found
seven code and accounting defects in v2 that would weaken any later run, gated or exploratory. v3 fixes them in a
separate folder so that the live v2 attempt, whose workers hash v2's files before every job, is untouched.

v3 is not a new design and does not reopen either stop:
- knee_edge_ood_v1 step 0.3 (A2) stays INCONCLUSIVE on G0.5; v2 step 0.4 (A3) stays INCONCLUSIVE on item 5. Neither
  label changes because training exists, and no threshold, seed, world, arm, recorder or decision rule changes here.
- **Gating status.** `stage1.a3_outcome` still requires a step 0.4 PASS, and the only step 0.4 record is v2's
  INCONCLUSIVE one (carried, pinned; see change 8). So v3 cannot request A4 as it stands. Running Stage 1 under v3
  would need either (a) an explicitly approved amendment that admits the v2 step 0.4 record with its item 5 miss,
  written and reviewed before any outcome of the exploratory run is read, or (b) the exploratory wrapper, now
  sealed (change 7), which supports no claim. This document does not choose; it records the choice as open.
- Step 0.4 is not re-run under the same design (5.0). `step04.execute` in v3 refuses a research-provenance run
  (change 8); the Windows dry run on smoke data stays available for tests and counting.

## Changes (the audit's R1-R7, in its numbering)

1. **R1, V0(f) implemented** (`stage1_math.py`). PROTOCOL_V2.md change 3 wrote the rule and v2's code never computed
   it, so a v2 Stage 1 result could pass the implementation while failing the sealed written rule. v3 adds, per seed on
   the R-valid anchors at s10: median rho_hat_R(s10) >= 0.30 and median KC_s10(RANDOM) <= 0.3 x (median rho_hat_R(s10)
   + 0.15), one-sided, exactly as written; the bound, the floor and the denominator (n R-valid) are in every per-seed
   record; empty or non-finite inputs fail; a failure in any seed is INCONCLUSIVE (label precedence unchanged). The
   validity list is `VALIDITY = (V0a, ..., V0f)`. No threshold was chosen here: 0.30 and 0.3 are v2's, with their
   derivation in PROTOCOL_V2.md change 3.
2. **R2, reference definition corrected; the cancellation claim withdrawn** (`stage1_math.py`, `stage1.py`).
   PROTOCOL_V2.md change 7 says the clipped-reference bias "enters CUTOFF and RANDOM alike, so it cancels in K1a's
   contrast". That is false as a general claim: per anchor, KC_clip = KC_own + D with D = dQ^pi_unclipped - dQ^pi_clip,
   the within-state difference of the two continuation-value gaps, and D depends on the actor, so the CUTOFF-minus-
   RANDOM contrast carries D_C - D_R, which cancels only where the two actors' gaps agree. v3 computes KC_own, D and
   D_C - D_R per anchor from the arrays Stage 1 already stores (`qpi_unclipped`), checks the identity per anchor to
   1e-9 (a missing or non-finite companion shows as `identity_ok` False, never as accuracy), and reports them in
   `reported[seed]["v3_own_reference"]`: KC_own medians, the K1a contrast under the own reference, K1c's absolute
   clause under the own reference, and `continuation_gap_contrast` (median, signed shares, denominators,
   `cancels_exactly`). **The primary estimand is unchanged: KC against the clipped own continuation.** Whether the
   own-policy reference should become primary is a scientific decision the user has not made; v3 keeps the clipped
   primary and narrows the claim: K1a's contrast is read with D_C - D_R beside it, K1c with KC_own beside it.
   Medians are not additive, so the decomposition is per anchor before any median.
3. **R3, exact counting over every leading dimension** (`units.SimulatorCounter`). v2 counted `states.shape[0]` per
   raw `dynamics()` call. `dataset_knee.validate` replays the whole dataset in one call on `(episodes, 100, 13)`
   states, so each KneeDataset construction counted 1,280 instead of 128,000. v3 counts the product of the broadcast
   leading dimensions of states (..., 13) and commands (..., 4): a 1-D state is 1, `(N, 13)` is N, `(E, T, 13)` is E x T,
   `(1, 13)` with `(N, 4)` is N, incompatible leading shapes are rejected, and the state and command widths are
   asserted. The counter now keeps `attempted` (reserved before the call; the cap is checked on it) apart from
   `steps` (completed after the call), so a simulator that raises leaves attempted > steps on record. The only 3-D
   call in the package is the validation replay; the raw Q^pi loops, the free-episode loops and the thrust check pass
   2-D arrays and were counted correctly.
   **Accounting correction for v2 A3 attempt 1** (append-only; v2's sealed record is not edited): the collection
   built two 1,280-episode datasets (stall3 and its twin), each validated once; recorded exact 2,587,760 + 2 x
   (128,000 - 1,280) = **2,841,200**, under the approved 3,000,000 cap by 158,800. The label (INCONCLUSIVE, item 5)
   does not depend on this. The lineage replay (`_lineage`) uses `env.reset` and the recorder only and takes no
   simulator step. The Windows dry run on the same code path now measures 2,841,200 exactly (test_step04).
   **For the v2 exploratory run**, each warmup, branch and BC-only job constructs one dataset (recorded 1,280 ->
   corrected 128,000), r0 one, and each evaluate job one per world; `amended_analysis.py` writes the correction per
   unit after the run is terminal, from the reports only.
4. **R4, dependency closure** (`units.dependency_closure`, `integrity.py`, both workers). v2's record covered the
   package folder and rerun1's pinned sources. It omitted rerun1 `design.py` (loaded by path in `runtime.match` for
   the distance coefficient) and `matched_policy_pilot_v1/launch.py` (the supervisor every dispatch runs under). v3
   lists them in `EXTERNAL_DEPENDENCIES`, hashes them into every job record, refuses to start when any is missing,
   and the worker re-hashes the whole record after its work (`integrity.changed_files`; `integrity_after` in every
   report), so an edit during a job is recorded and fails the job. Still outside the record, by design and as before:
   the canonical core authenticates its own sources on load; Unifloral `td3_bc.py` is pinned by the preflight source
   receipt and `bc.py` by the BC job's hash. A check-then-load window between the worker's first verification and
   its imports remains; an immutable per-run code snapshot would close it and is not done here.
5. **R5, one terminal owner** (`units.Terminal`; `step04.py`, `stage1.py`, `stage2.py`). Every attempt's terminal
   write goes through one owner: the first of the main thread, an interruption handler or the watchdog to finish
   writes, later callers are no-ops, and the owner snapshots the exact and attempted step counts and the elapsed time
   into the report before the write. v2's watchdog wrote without the counter and its interruption path wrote before
   the `finally` that attached the count; a late timer could also have overwritten a finished report.
6. **R6, cumulative physical spend** (`units.ledger`; every Stage 1 and Stage 2 attempt report carries `ledger`). The
   per-attempt `caps` stay the logical ledger of the attempt. The new ledger sums every prior attempt's unit
   directories: known seconds and exact steps from supervisor results and worker reports; a unit with neither is
   listed as UNKNOWN with its cap as an upper bound, never zero. Approvals remain per attempt, as 5.0 states; the
   ledger makes the cumulative cost visible, it does not renew any budget.
7. **R7, the exploratory wrapper is sealed and bound** (`stage1_exploratory.py`, `EXPLORATORY_MANIFEST.json`;
   `stage2.stage1_outcome`). The wrapper's manifest hashes the wrapper and the Stage 1 files it drives; the approval
   must carry `exploratory_manifest_sha256` equal to the manifest's sha256; a changed file stops the run before
   dispatch; an A4 approval cannot start it (different words and file), and Stage 2 rejects any Stage 1 report that
   carries an EXPLORATORY block or no protocol label, in addition to reading a different folder. The smoke receipt
   (`stage1_smoke.py`) now records its scope, the package identity it exercised, per-job exact and attempted steps,
   and totals of every resource category; it stays blinded to outcome values.
8. **Carried step 0.4 record and the step 0.4 guard** (`stage1.py`, `stage1_exploratory.py`, `step04.py`). As v2
   carried v1's step 0.3 record, v3 reads v2's single step 0.4 attempt where it is (`knee_edge_ood_v2/step04_run`)
   and pins its report (sha256 `f36b1a71fa4b4e11ca4f1903a8593d925f89f0dcb8c90c80c59cd4abc8ed3884`, label
   INCONCLUSIVE). The gate still requires PASS (see "Gating status"). `step04.execute` refuses research provenance in
   v3; smoke provenance (the Windows dry run) is allowed for tests.

## Amended analysis of existing attempts (R1-R3 applied to a record, without changing it)

`amended_analysis.py <attempt dir>` reads an existing Stage 1 attempt's JSON reports (never a checkpoint or dataset),
recomputes every rule under v3's math (so V0(f) and the R2 block appear), corrects the step accounting per unit, and
writes one file under `amended_analyses/` with the sha256 of every report it read and the original label beside the
v3 label. It is the route for the v2 exploratory run once terminal, and for any future attempt whose record predates
a rule. The original decision record stays the record.

## Claim scope

Unchanged from PROTOCOL_V2.md: no later version may remove, weaken or re-threshold V0(b), V0(d) or V0(f). v3 adds:
the primary KC reference is the clipped own continuation, and every K1a or K1c statement is read with its own-
reference companion and D_C - D_R on record. An exploratory Stage 1 outcome, from v2 or v3, is not a protocol label and
supports no claim about the knee-edge OOD hypothesis.

## Tests

`test_v3_repairs.py` (R1 boundaries and label precedence, R2 identity and contrast, R3 shapes, cap-on-attempted and
raise behaviour, R4 closure and tamper detection, R5 single write and watchdog, R6 ledger, R7 binding and A5
rejection) plus the carried v2 suite with three fixture updates (the synthetic anchors carry `qpi_unclipped`; the
watchdog test uses the terminal owner; the step 0.4 dry run asserts 2,841,200 exact and attempted steps). Run from
the repository root on Windows:
`PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v3 -p "test_*.py"`.

## What this package does not do

It does not run anything, approve anything, fetch, push, delete or move evidence, or touch `knee_edge_ood_v1`,
`knee_edge_ood_v2`, their run folders or their manifests. The five unauthorized development packages listed in
`UNTRACKED_PACKAGES_RECONCILIATION.md` are outside it. The GPU smoke of v3 has not been run; A4_MANIFEST.json still
says it is required before any approval.
