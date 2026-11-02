# Candidate knee_s10_v2: the native screen rule (written before any screen number exists)

The one package of OOD definition v4 r4 (`../../ood_definition_v4/REVISION_r4.md`, sections 3 to 7) for candidate
knee_s10_v2, after its formal G2 PASS (`../formal_g2/G2_FORMAL_READOUT_20261101.md`) and the retained-support check
RS1 (`../retained_support/RS1_READOUT.md`). Written 2026-11-01 after the user chose one sealed package, one approval,
one run. **Exploratory, go/no-go.** It supports no protocol claim; its seeds are never confirmatory.

## The question

Does BCA as built, at the dose fixed below, change the learning actor's decisions where the critic's value is
unsupported, beyond a native baseline paired with it by seed and random numbers, on a bed where the past-knee action
is worth less than the smooth world says? Three corrections from the user are part of the reading (r4 section 1):
G2 established a cross-world value difference, not learned-critic over-estimation, so over-valuation is a read here
(O1, O2), not a premise; a three-seed null does not identify BCA as the cause; and frozen-actor evaluation cannot
measure this actor-only BCA, which is why the screen is native.

## Decisions fixed here

| Decision | Value | Why |
|---|---|---|
| Host | the pinned Unifloral TD3+BC (`MODEL_IDENTITY.md`: revision f2dc1278, upstream `algorithms/td3_bc.py` sha256 6f5ab6f9..., loaded by `payload/unifloral_td3bc_validation_v1/native_loader.py`), native settings | the user's rule: Unifloral base, never the CORL port |
| BCA | the uncertainty-only bridge (`payload/unifloral_bca_pipeline_validation_v1/bca_bridge.py`): the actor maximizes Q1 - kappa U | BCA acts on Q values |
| Dose | kappa = 63 (the dose screen's kappa_A), one dose | kappa 3 showed no leverage (1.5 to 3.1% of outward pushes reversed); 63 is the dose at which BCA's pull is dominant where its direction is right; a second dose is a second screen |
| Labelled control | Stage 1's count-matched RANDOM arm (`masks.arm_masks`, seed = the formal G2 v2's random_control) | no adapter change; the confound is stated: CUTOFF and RANDOM differ in which in-support rows they hold, so O2 compares arms that differ in-support as well as in W; BAL measures that difference |
| Data | the formal G2 v2 datasets by digest: stall3 c2a6d616..., no-stall e3549869... | the rows every gate of this candidate runs on |
| Seeds | `screen_seeds.json` (namespace `knee_s10_candidate_v2/native_screen/`, audited); `knee_s10_candidate_v2/confirmation/` reserved, never drawn | screen seeds are never confirmatory |
| Scored slot | s10 (a0 = -0.9), tol 0.15, c 0.3 | the candidate's one slot (r3 section 4) |
| Update budget | 10,000 warmup + 10,000 branch = 20,000 per run; BC-only 10,000 | Stage 1's settled convention |

## What is reused and what is changed

knee_edge_ood_v3's Stage 1 pipeline (`stage1_worker.py`, `stage1_eval.py`, `stage1_math.py`, `units.py`,
`integrity.py`) runs unchanged on disk. `screen_bind.py` makes seven declared in-process changes: the widened recorder
(`../../knee_s10_candidate_v1/recorder_wide.py`) replaces the knee-edge recorder before any import; research data
validate against the formal G2 v2 collection entropy; the data plan draws from this screen's seeds for the two
datasets by digest; research training seeds are this screen's three rows; the E2 start bank is the formal G2 v2's
128 starts; the E2 anchors are the formal G2 v2's 96 gated anchors (built by the v2 prefix rule, identical in both
worlds; P1 re-checked for each). The E1 bank, the E1 anchors, the R0 banks and the visited-state sample stay
knee_edge_ood_v3's sealed ones (states and selections, reported or validity only). The BCA branch and its evaluation
are the dose screen's (`../../bca_dose_screen_v1/screen_worker.py`), at kappa 63, with U recorded at the anchors.
`collection/collection.json` (written by `prepare.py`, CPU) holds the datasets by digest and the Stage 1 data plan
of both arms in both worlds; every job re-derives the plan and refuses on any difference. The seventh change
(amendment A1, below): the two archives are read by `screen_data.load`, which authenticates the archive bytes, the
canonical content digest and the formal G2's sealed run record (its manifest, PASS report, collection and all 44
pinned source files) by sha256 and returns the recorded bytes as the pipeline's `KneeDataset`, instead of
`dataset_knee.load`'s regeneration of every episode by reset and replay. No tolerance is introduced anywhere; the
loader accepts exactly the two recorded archives.

## Units, in order (fail-fast)

| # | Unit | Count | Kind | Stop rule after it |
|---|---|---|---|---|
| 0 | preflight | 1 | GPU environment receipt | a failure is exempt (no attempt used) |
| 1 | g1 | 1 | CPU: G1_RULE.md (`../g1/`), the supervised positive control on exact Q^{pi_ref} labels, both worlds, three fit seeds; fixes q* and the resolution band | NOT REPRESENTABLE or INCONCLUSIVE: the screen ends INCONCLUSIVE (representation); nothing trains |
| 2 | warmup_s{i}_{arm}_{world} | 12 | native, 10k | |
| 3 | r0 | 1 | adequacy (+50) and reach at 10k | adequacy fails: INCONCLUSIVE; CUTOFF reach < 0.05 in every seed: STOP: not reached |
| 4 | branch_s{i}_{arm}_{world} | 12 | native, 10k to 20k | |
| 5 | bc_only_s{i} | 3 | BC-only actor on CUTOFF (stall3) | |
| 6 | evaluate_s{i} | 3 | every Stage 1 measurement of the four native runs and the BC-only actor | R1 fails in every seed: STOP: not reached; no BCA unit trains |
| 7 | bca_branch_s{i}_{arm}_{world} | 12 | BCA at kappa 63 from the same fork, same branch seeds and random-number stream as the native branch; pairing checked | |
| 8 | evaluate_bca_s{i} | 3 | the same measurements of the four BCA runs, with U at the anchors | |
| 9 | decision | | `screen_math.decide` | |

## Reads and bars

Every read is reported per seed. The bars are r4 section 5's, restated with the values fixed at review:

| id | Read | Bar |
|---|---|---|
| V0(a) | J_E1(20k) - J_E1(untrained), CUTOFF and RANDOM, stall3 | >= +50 |
| V0(c) | anchors valid under both arms | >= 32 |
| V0(e) | a1..a3 past the knee on E2 steps (CUTOFF, RANDOM, CUTOFF in the no-stall world) | <= 0.005 |
| V0(f) | median KC_R(s10) <= 0.3 (median rho_own + 0.15), median rho_own >= 0.30 | as sealed |
| L1n | per anchor -0.15 <= KC_R <= 0.3 rho_own + 0.15 on anchors with rho_own > 0: share >= q* (G1) and median inside the band | q* from G1 |
| BAL | median \|KC_C - KC_R\| at edge_700 | <= the resolution band from G1 |
| PLC | pooled no-stall-world median of (KC_C - KC_R)(s10) | within +-0.30 |
| O1 | median KC_C(s10) >= 0.5 and >= 0.5 x median rho_own | every seed |
| O2 | median (KC_C - KC_R)(s10) >= 0.5 and larger than its no-stall-world value | every seed |
| U0 | shares of anchors with KC <= -0.15 per arm and world, KC quantiles, the twin's under-capture share | reported |
| R1 | CUTOFF R_deep- entry rate on E2 minus its BC-only actor's | >= 0.25 every seed; RANDOM's excess <= 0.10 reported |
| R2 | K3a, K3b, K3c | >= +2.0 every seed |
| B1 | entry(BCA-C) - entry(HOST-C), stall3 | <= -0.10 every seed |
| B2 | q_rank_gain_s10: on anchors where s10 is truly worse, the share ranked correctly by Q1 - 63 U minus by raw Q1 | >= +0.25 every seed |
| B3 | J(BCA-C) - J(HOST-C), E2, stall3 | >= +2.0 every seed |
| B3s | B3 minus the same difference in the no-stall world | >= +2.0 every seed |
| U1 | J(BCA-C) - J(HOST-C) in the no-stall world; the penalized value's under-ranking share minus raw Q1's on anchors where s10 is truly better | >= -2.0 and <= +0.25, every seed |

Readings, in precedence: INCOMPLETE, INCONCLUSIVE (G1 not PASS, or any validity read not holding in every seed, or
PLC), STOP: not reached (R0 or R1 fails in every seed), NO-GO: nothing to repair (R1 holds, O1 fails in every seed),
NO-GO: no effect at this dose and budget (O1 and R1 hold, a B read fails in every seed), NO-GO: over-correction (the
B reads hold, U1 does not hold in every seed), MIXED (seeds disagree on a deciding read), GO. What each leads to is
r4 section 5's table. A NO-GO reads "no pre-registered effect at this dose and budget on this bed", never "BCA does
not work". GO is permission to design the confirmation stage (r4 section 6), not a claim.

## Kill rules and caps

Every job runs under matched_policy_pilot_v1's supervise() in its own process group and is killed at its cap. Caps:
1,800 s per training job; 10 h of training in all (39 training jobs); 2 h and 36,000,000 exact simulator transitions
of evaluation (r0, the six evaluations); 1 h and 9,000,000 exact steps for g1; 600 s for the preflight. A non-finite
loss, Q or U fails the job. A failed job ends the attempt INCOMPLETE. No bar, dose, slot or seed changes after the
approval; a change is a new screen with its own rule.

## Attempts

One attempt by default. After INCOMPLETE only, a second attempt under a new approval naming attempt 2 reuses every
unit an earlier attempt completed (same id, kind, seed, arm and world; its SUPERVISOR_RESULT.json says completed) and
runs the rest. No attempt 3, with one exception written after the fact (amendment A1): when attempt 2 ended
INCOMPLETE because the dataset loader's reset regeneration failed ("initial state lineage differs"), a third attempt
under a new approval naming attempt 3 runs the same way; the supervisor checks the failed unit's recorded error.
Never a fourth. `--decide` recomputes the decision from the saved reports (JSON only).

## What the user's approval states

`SCREEN_APPROVAL.json`, written at the user's terminal and committed under their identity (README.md has the
command), must contain these sentences, with the manifest's sha256 and the attempt number:

1. "I approve the native screen for candidate knee_s10_v2 under OOD definition v4 r4 at kappa 63, attempt 1"
2. "I adopt definition v4 r4 for this candidate"
3. "RS1: the scored anchors are retained-supported; I overrule the start-share primary miss (121 of 128 against 122,
   a one-draw threshold); the secondary's agreement with D3 is on record"
4. "the BCA arm at kappa 63 is approved"
5. "the screen is exploratory and supports no protocol claim"

Sentence 3 is the user's own decision on RS1 (its rule left that decision to the definition and the user); the
primary miss stays in the ledger as recorded. Sentence 4 is the standing rule for any BCA training arm.

## Integrity

`SCREEN_MANIFEST.json` (written by `seal.py`) hashes this package, the collection record, the formal G2 package and
its attempt-1 record, the prefix rule, v1's pi_ref and recorder, the g1 modules and rule, the definition and r4, and
RS1's record; it pins knee_edge_ood_v3's A4 and exploratory manifests. Every job's record (`integrity`) is the v3
dependency closure plus those files; the worker refuses on any change before and after its work. The datasets are
checked by digest on every load (archive bytes and canonical content, `screen_data.py`).

## Amendments

| id | Date | What changed, and why | What did not change |
|---|---|---|---|
| A1 | 2026-11-01, after attempts 1 and 2 (both INCOMPLETE before any training) | Change 7, the authenticated non-regenerating dataset loader (`screen_data.py`), and attempt 3 after the named loader fault only. Cause: the formal G2 collected and validated both datasets in the Windows venv; the recorded Linux environment's reset of episode 16 reproduces `wave_1` one ULP off (`../reset_lineage_diagnostic_v1/receipt_20261101T092712_649674_EST_424.json`), so `dataset_knee.load`'s exact replay cannot pass there. Attempt 1's fix (the g1 unit's CPU environment, `screen_worker.run_job`) is the one code-only fix the rule allowed. | the question, the dose, the control, the data (same bytes, same digests), the seeds, the units, the reads, the bars, the readings, the caps. The I0 instrument check still compares the Linux-recomputed exact values with the Windows-computed formal report at 1e-9 relative; if platform arithmetic breaks it, the screen ends INCONCLUSIVE and says so. |

## Ledger entry

Candidate knee_s10_v2, sigma s10, native screen under r4 at kappa 63: the reading, every consensus, G1's q* and
band, the GPU hours and exact transitions spent.
