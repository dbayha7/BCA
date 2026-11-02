# Candidate knee_s10_v2: the native screen rule under r5 (written before any screen number exists)

The one package of OOD definition v4 r5 (`../../ood_definition_v4/REVISION_r5.md`) for candidate knee_s10_v2, after
its formal G2 PASS and RS1. It replaces `../native_screen/SCREEN_RULE.md` (r4), whose three attempts ended before any
training (see `../../ood_definition_v4/LEDGER.md`); the r4 package stays on disk, sealed and untouched, and this
package imports its unchanged modules by path. **Exploratory, go/no-go.** It supports no protocol claim; its seeds
are never confirmatory.

## The question

Does BCA as built, at kappa 63, change the learning actor's decisions where the critic's value is unsupported,
beyond a native baseline paired with it by seed and random numbers, on a bed where the past-knee action is worth less
than the smooth world says? r4 section 1's three corrections are part of the reading.

## Decisions fixed here (unchanged from r4 unless marked)

| Decision | Value |
|---|---|
| Host | the pinned Unifloral TD3+BC, native settings (`MODEL_IDENTITY.md`) |
| BCA | the uncertainty-only bridge; the actor maximizes Q1 - kappa U |
| Dose | kappa = 63, one dose |
| Labelled control | Stage 1's count-matched RANDOM arm (confound stated: the arms differ in which in-support rows they hold) |
| Data | the formal G2 v2 datasets by digest (stall3 c2a6d616..., no-stall e3549869...), read by the amendment A1 loader (`../native_screen/screen_data.py`) |
| Seeds | `../native_screen/screen_seeds.json` (unspent: no r4 attempt trained); `knee_s10_candidate_v2/confirmation/` reserved |
| Scored slot | s10, tol 0.15, c 0.3 |
| Update budget | 10,000 warmup + 10,000 branch per run; BC-only 10,000 |
| **New (r5)** | G1 reported, not deciding; stage 0 exempt from attempts; the WSL dry run before the seal; B2 nulls; returns reported; pooled anchor reads; MIXED removed |

## What is reused and what is changed

knee_edge_ood_v3's Stage 1 pipeline runs unchanged on disk. `../native_screen/screen_bind.py` makes the seven declared
in-process changes (recorder, dataset entropy, data-plan seeds, training seeds, E2 bank, E2 anchors, kappa, the A1
loader). This package adds: `screen_worker_r5.py` (the same job kinds plus `i0`, and `g1` under section 5's reading;
the A1 loader is installed for research jobs only, so smoke jobs read their own test collection), `i0_unit.py`,
`g1_unit_r5.py`, `screen_math_r5.py` (the reads below), `screen_r5.py` (the supervisor with stage 0 and the dry run),
`dry_run.py`.

## Units, in order (fail-fast)

| Stage | # | Unit | Count | Kind |
|---|---|---|---|---|
| 0 | 0 | preflight | 1 | GPU environment receipt |
| 0 | 1 | i0 | 1 | CPU instrument: I0 on the 96 anchors, both archives by digest, data plans re-derived |
| 0 | 2 | g1 | 1 | CPU calibration (section 5), reported |
| 1 | 3 | warmup_s{i}_{arm}_{world} | 12 | native, 10k; **the attempt is counted when the first starts** |
| 1 | 4 | r0 | 1 | adequacy (+50) and reach at 10k |
| 1 | 5 | branch_s{i}_{arm}_{world} | 12 | native, 10k to 20k |
| 1 | 6 | bc_only_s{i} | 3 | BC-only actor on CUTOFF (stall3) |
| 1 | 7 | evaluate_s{i} | 3 | every Stage 1 measurement of the native runs and the BC-only actor |
| 1 | 8 | bca_branch_s{i}_{arm}_{world} | 12 | BCA at kappa 63 from the same fork, pairing checked; only if R1, R2 and O1 hold |
| 1 | 9 | evaluate_bca_s{i} | 3 | the same measurements of the BCA runs, with U at the anchors |
| | 10 | decision | | `screen_math_r5.decide` |

## Reads and bars (r5 section 4, restated)

Instrument: I0 (stage 0; a miss is INCONCLUSIVE (instrument), no attempt spent).

Validity (a miss is INCONCLUSIVE): V0(a) >= +50 every seed; V0(c) >= 32 every seed; V0(e) <= 0.005 every seed;
PLC pooled within +-0.30.

Bed: R1 pooled mean entry excess over BC-only >= 0.25, else STOP: not reached; R2 pooled mean K3a >= 2.0, else STOP:
no harm; O1 pooled median KC_C(s10) >= 0.5 and >= 0.5 x pooled median rho_own, else NO-GO: nothing to repair. R1, R2
and O1 are read after the native evaluations; no BCA unit trains unless all three hold.

BCA: B1 <= -0.10 in every seed; B2 pooled harmful-anchor gain >= +0.25; B2's net gain (harmful gain minus the benign
under-ranking change) >= the shuffled-U null's median net + 0.05 and >= the constant-penalty null's net + 0.05 (2,000
shuffles, seed 20261101); U1 pooled under-ranking change <= +0.25 and placement <= +0.05.

Reported, never deciding: returns in both worlds with the checkpoint; V0(f); L1n (q* from G1 when given); BAL (G1's
band); O2; U0; the twin's reads; sign-flipped U; G1's label, converged updates, band and capture.

Readings in precedence: INCOMPLETE; INCONCLUSIVE (instrument); INCONCLUSIVE; STOP: not reached; STOP: no harm;
NO-GO: nothing to repair; NO-GO: no consistent effect at this dose and budget; NO-GO: strength without placement;
NO-GO: over-correction; GO.

## Kill rules and caps

Every job runs under matched_policy_pilot_v1's supervise() in its own process group and is killed at its cap: 1,800 s
per training job; 14 h of training (amendment A4; was 10 h); 2 h and 36,000,000 exact transitions of evaluation; 1 h and 9,000,000 exact
steps for g1; 600 s each for the preflight and i0. A non-finite loss, Q or U fails the job. A failed training or
evaluation job ends the attempt INCOMPLETE. No bar, dose, slot or seed changes after the approval.

## Stages and attempts (r5 section 3)

Stage 0 is exempt: a failure or an I0 miss ends the run INCONCLUSIVE (instrument), is recorded under
`screen_run/run_<utc>/`, and spends no attempt; after three such runs the package must be re-sealed. An attempt is
counted when the first warmup job starts (`TRAINING_STARTED.json`). One attempt; after INCOMPLETE, attempt 2 under a
new approval reuses every completed unit of any earlier run; no attempt 3 (amendment A5: attempt 3 after attempt 2
was killed from outside; no attempt 4). `--decide` recomputes the decision from
saved reports.

## The dry run (precondition 3)

`screen_r5.py --dry-run` runs in the recorded WSL environment before the seal: preflight; i0 on the real anchors; a
smoke collection from the test entropy; then one job of each kind, warmup, r0, branch, bc_only, evaluate,
bca_branch, evaluate_bca, on seed 0, CUTOFF, stall3, with smoke-only reductions, under the same caps. Output goes
outside the repository; the structure-only receipt (completed, errors, timings, exact steps; no outcome values) is
copied to `dry_run/DRY_RUN_RECEIPT.json` and hashed by the manifest. `seal_r5.py` refuses without a completed
receipt whose recorded hashes of the worker, bind, data and math modules equal the files being sealed.

## What the user's approval states

`SCREEN_APPROVAL.json`, written at the user's terminal and committed under their identity (README.md), must contain
these sentences, with the manifest's sha256 and the attempt number:

1. "I approve the native screen for candidate knee_s10_v2 under OOD definition v4 r5 at kappa 63, attempt N"
2. "I adopt definition v4 r5 for this candidate"
3. "RS1: the scored anchors are retained-supported; I overrule the start-share primary miss (121 of 128 against 122,
   a one-draw threshold); the secondary's agreement with D3 is on record"
4. "the BCA arm at kappa 63 is approved"
5. "the screen is exploratory and supports no protocol claim"

## Amendments

| # | When | Change | Unchanged |
|---|---|---|---|
| A2 | 2026-11-01, after attempt 1 on Zeus (run `run_20261102T034946310438Z`: stage 0 passed, two warmups completed, the user interrupted the third to move the run to Titan; reading INCOMPLETE; attempt counted) | **Binding 8, device kinds.** knee_edge_ood_v3 pins one recorded GPU kind (Zeus's "NVIDIA GeForce RTX 5070 Ti Laptop GPU") in its preflight and in every training bind. The worker now sets the recorded kind to the one allowed kind the machine has (`device_kinds.py`: Zeus's, or Titan's "NVIDIA GeForce RTX 5070 Ti", the payload's own machine), and every GPU report records it. **Same-device reuse.** A completed GPU unit of an earlier run is reused only if its report carries this run's device kinds; CPU units (g1) are reused as before, preflight and i0 never. So attempt 2 on Titan starts its own warmups and never continues a Zeus checkpoint (which may not be in the clone at all: `*.msgpack` is git-ignored). `TRAINING_STARTED.json` records the device kinds. **Dry run on the running machine.** The worker changed, so the dry-run receipt must be made again on the machine that runs attempt 2 before the re-seal; the receipt records the device kind and the seal requires it to be an allowed one. | the question, the dose, the control, the data (same bytes, same digests), the seeds, the units, every read, every bar, every reading, the caps, the two-attempt limit, the approval words (attempt 2). |
| A3 | 2026-11-02, after Titan's first dry run (i0 failed in 6 s: "pi_ref gains differ from the pinned GAINS_SHA256") | **Environment, not code.** `knee_s10_candidate_v1/pi_ref.py` recomputes its LQR gains (a 100-step Riccati recursion in float64 through OpenBLAS) at first use and pins their bytes by sha256, computed on Zeus. Titan's CPU selects different OpenBLAS kernels and reproduces the recursion one last bit off, so the pin refuses. On Titan every invocation of this package sets `OPENBLAS_CORETYPE=Haswell`, under which the recursion reproduces Zeus's bytes exactly and the pin passes (Titan's second dry run, 2026-11-02 ~09:40 UTC). The pin itself is the verification: it runs in stage 0 (i0, then g1) before any attempt is counted, so a missing or wrong setting costs nothing. A code change (loading the pinned gains from a stored artifact) was written, committed and reverted unused (`d84a38b5`, `96e16267`), because the environment route changes no file on either machine. | every file of the package and of pi_ref. |
| A4 | 2026-11-02, after Titan's third dry run completed (receipt: status completed, Linux, real dispatch, device kind "NVIDIA GeForce RTX 5070 Ti") | **Total training cap 10 h -> 14 h.** Titan trains about 1.8 times slower per job than Zeus (smoke warmup 889 s, branch 952 s, BCA branch 792 s, against Zeus's 512, 480, 493 s), so the dry run projects 31,674 s of training for the run against the 36,000 s cap: 12% headroom on the last allowed attempt. The cap is a kill rule for a runaway run, not a read; 14 h (50,400 s) restores about 60% headroom. Set by the user before the seal, from the dry run's structure-only timings; no outcome value was seen. `screen_r5.py` is not among the dry-run-hashed modules, so the receipt stands. | the per-job cap (1,800 s; jobs take about 950 s), the evaluation and g1 caps, and everything else. |
| A5 | 2026-11-02, after attempt 2 on Titan (run `run_20261102T115816759197Z`, started 11:58 UTC under manifest 904542fd…, approval 7e5887b1) was killed at 12:30:37 UTC by a forced Windows restart (Windows Update finishing Windows 11 26H2; User32 event 1074 at 12:30:05 UTC, "Operating System: Service pack (Planned)"), 32 minutes in: stage 0 passed, two warmups completed (794 s, 743 s), the third stood at 3,500 of 10,000 steps; no evaluation, read or outcome value existed. The user's decision. | **Attempt limit 2 -> 3** (`MAX_ATTEMPTS`). **A killed attempt reads INCOMPLETE.** Every ending the supervisor sees writes `SCREEN_RESULT.json`; a power-off cannot, so `check_approval` reads a counted attempt with no readable `SCREEN_RESULT.json` as INCOMPLETE (`recorded_reading`), which permits the next attempt under a new approval. Attempt 3 reuses completed units under the existing rules (A2: GPU units only on the same device kind, so attempt 2's two Titan warmups; g1 from attempt 1). No attempt 4. Windows updates on Titan are paused until 2026-11-30 before the run. `screen_r5.py` is not among the dry-run-hashed modules, so the dry-run receipt stands; stage 0 re-verifies the environment after the OS update before anything is counted. | the 14 h training cap and every other cap, the question, the dose, the control, the data, the seeds, the units, every read, bar and reading, the approval words (attempt 3). |

## Integrity

`SCREEN_MANIFEST.json` (written by `seal_r5.py`) hashes this package, the r4 package's modules it imports, the
collection record, the formal G2 package and its attempt-1 record, the prefix rule, v1's pi_ref and recorder, the g1
modules, the definition (r3, r4, r5), RS1's record and the dry-run receipt; it pins knee_edge_ood_v3's A4 and
exploratory manifests. Every job's integrity record is the v3 closure plus those files; the worker refuses on any
change before and after its work. The datasets are checked by digest on every load.

## Ledger entry

Candidate knee_s10_v2, sigma s10, native screen under r5 at kappa 63: the reading, every read (pooled and per
seed), the nulls, G1's calibration, the GPU hours and exact transitions spent, and the dry-run receipt.
