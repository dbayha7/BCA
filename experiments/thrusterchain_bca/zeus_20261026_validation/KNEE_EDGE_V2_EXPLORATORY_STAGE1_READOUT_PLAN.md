# Exploratory Stage 1 of knee_edge_ood_v2: monitor, audit and read-out plan

**Status of this document:** uncommitted, written read-only and blinded on 2026-10-30 (EDT) while the run was still in its warmup phase. At 23:12 EDT the run folder still had no `r0/` directory, so no Stage 1 outcome existed when this was written. The file's own modification time and, once you commit it, the commit time are the proof that the reading rules below were fixed before any result was opened.

**Where it lives:** `experiments/thrusterchain_bca/zeus_20261026_validation/` (top level), deliberately *outside* `knee_edge_ood_v2/`. Every worker re-hashes the package folder's `.py`, `.json` and `.md` files before it starts (finding C4 below), so adding a file inside that folder could affect the run.

**Prepared by:** a Claude side session with three read-only specialist agents (code audit, ledger/timing monitor, read-out plan), each followed by a skeptic agent that tried to refute its claims. All three skeptics reported no blinding breach. Nothing was edited, moved, signaled, launched, committed or run.

## A. Bottom line (plain language)

1. **The run is healthy and nothing in the package changed after launch.** At 23:03 EDT, 5 of 31 units were done (all 5 warmups completed, exit code 0, no timeouts) and a sixth was running. At 23:12 a seventh warmup folder exists. Every `job.json` records 70 file hashes; all 70 matched the files on disk in all 7 jobs checked, and the 10 files in `A4_MANIFEST.json` match their manifest hashes.
2. **The biggest finding: the V0(f) check from `PROTOCOL_V2.md` is not implemented in the shipped code.** `PROTOCOL_V2.md:80-101` says a miss on V0(f) makes Stage 1 INCONCLUSIVE, but `stage1_math.py` implements only V0(a) to V0(e). The label this run computes can therefore be PASS or FAIL where the v2 rules would say INCONCLUSIVE. The plan below fixes a by-hand V0(f) calculation now, before any result exists. **You need to confirm it (Section D, item 1).**
3. **The label is weakly marked.** The computed outcome is stored under the plain key `label` in `report.json` and `decision.json`, and `decision.json` carries no EXPLORATORY marker at all. Only the attempt-level `report.json` has `exploratory_label` and `protocol_label: null`. Cite `exploratory_label` and the EXPLORATORY block, never `decision.json` or `label` alone.
4. **Whether the run reaches 02:00-02:40 EDT depends on R0.** If R0 (at about 23:59 to 00:05 EDT) fails adequacy or shows FAIL-early, the run ends right there and the 12 branches, 3 BC-only jobs and 3 evaluations never run. Otherwise the working estimate is 02:00 to 02:40 EDT on Oct 31, with evaluation time the main unknown (ceiling about 04:20).
5. **A restart is not a simple retry** (C5, C6). Do not close the terminal that launched it.

## B. Monitor snapshot (23:03 EDT; structure and timings only)

| Item | Value |
|---|---|
| Launch | 22:20:35 EDT (`ATTEMPT.json` `started_utc` 02:20:35Z) |
| Done at 23:03 | preflight (4.0 s) + 5 of 12 warmups; warmup 6 (`warmup_s1_random_stall3`) running |
| Warmup durations | 491.7, 471.2, 505.1, 507.6, 481.3 s; mean 491.4 s (stall3 about 481 s, nostall about 506 s, small samples) |
| Caps | per job 1,800 s (slowest warmup 28%); training 18,000 s (13.6% charged); evaluation 3,600 s and 18,000,000 transitions (0 charged) |
| Supervisor overhead between jobs | under 0.1 s |
| Errors | none in the console log (case-insensitive search for error, warn, traceback, exception, nan, killed, timeout, fail); all `worker.log` files empty |
| Disk | about 10.9 MB per warmup; 134 GB free on C: |
| Machine | on AC power; AC sleep and hibernate timeouts set to never; no reboot since Oct 27 |

**Projection (inference, not a record).** Twelve warmups end near 23:59 EDT (range 23:55 to 00:02). The protocol quotes 11,853 s of training from a smoke run of the *v1* folder (`PROTOCOL_V2.md:44-50`; no receipt exists in the repository, and it does not certify the v2 worker). The projection is defined as 12 warmups + 12 branches + 3 BC-only jobs, so dividing by 24 instead of 27 was wrong. **Whether the run is "on pace" with that smoke figure is therefore unestablished.** The 5 h training cap would bind only if branches averaged roughly 0.9 to 1.0 thousand seconds each. Training end if continuous and branches match warmups: about 01:38 EDT, plus evaluation time.

**Correction to the monitor's own text:** it said "6 of 31 done, 24 pending". The unit list has 31 units *without* preflight, so the correct statement at 23:03 was 5 of 31 done, 1 running, 25 pending.

## C. Code and integrity audit

| # | Finding | Severity | What it means |
|---|---|---|---|
| C1 | The five monkeypatches in `stage1_exploratory.py` (`RUN`, `APPROVAL`, `APPROVAL_WORDS`, `a3_outcome`, `_finish`) take effect, because `stage1.execute` looks each up as a module global at call time. `stage1.py` is byte-identical to the sealed A4 file. The run writes only under `stage1_exploratory_run/`; `stage1_run/` does not exist. | info | The run is what the driver says it is. The real differences from A4 are exactly the four the driver lists. The driver has no test and is outside `A4_MANIFEST.json`. |
| C2 | The computed label stays under the plain key `label`; `decision.json` also holds `per_seed`, `reported` and `failed_unit`, and has no disclosure. `_finish_exploratory` has not run yet, so it is unexercised. | medium | Easy to quote a label without its caveat. |
| C3 | Stage 2 cannot be requested on this run, but only because of the folder name. `stage2.stage1_outcome` checks label == PASS, the manifest hash and the A5 precondition, and never looks at the EXPLORATORY block. This run's report carries the real A4 manifest hash. | medium | Do not rename, copy or symlink `stage1_exploratory_run` to `stage1_run`. A5 would still need a user-written `A5_APPROVAL.json`. A suggested hardening (reject any report with an EXPLORATORY key) is **not applied**. |
| C4 | Editing any existing `.py`, `.json` or `.md` file in the package folder (or deleting or renaming one) makes the next worker refuse to start; that failed unit is a counted failure. New files are safe for the live run. | medium | Leave the folder alone while the run is live. |
| C5 | A hard kill (SIGTERM, closing the terminal, `wsl --terminate`, power loss, out-of-memory kill) leaves no `report.json`. The code then refuses a new attempt (`attempts.py:62-63`). The run is attached to a console through `tee`, so closing that terminal may kill it (inferred, not proven). | medium | Keep the terminal open and the laptop awake and plugged in. The lid-close action could not be read. |
| C6 | A graceful stop (Ctrl-C, worker crash or timeout, exhausted cap) writes an INCOMPLETE report and counts as an attempt under 5.0 unless it happens in preflight. A restart would need a new approval naming attempt 2, and would re-run every sealed unit and require it to reproduce bit for bit (the integrity record changes when the approval file changes). Whether GPU training is bit-reproducible is unknown. | medium | Plan on no restart (also the stance in Section 6 below). |
| C7 | `EXPLORATORY_APPROVAL.json` satisfies the approval check (manifest hash, words, attempt 1, prediction acknowledgement). It is uncommitted, and no readable file shows who wrote it; its modification time (22:20:09) is 26 seconds before launch. | low | Consistent with your own description; stated here only as a record gap. |
| C8 | Untracked in git and relied on by the run: `stage1_exploratory.py`, `EXPLORATORY_APPROVAL.json`, `AUDIT_NOTES_20261030.md`, `stage1_exploratory_run/`, and **`step04_run/`** (the step 0.4 evidence the driver reads; `ATTEMPT.json` records only its `collection_sha256`). | low | "Committed before any result is opened" does not cover these unless you commit them too (Section D, item 5). |
| C9 | No v2 GPU smoke receipt exists in the package, although `A4_README.md` asks for one before an A4 approval. | low | Not blocking an exploratory run; relevant if anyone later cites this run as A4-like. |
| C10 | The integrity hashes cover the package and rerun1 sources, not the supervisor module `matched_policy_pilot_v1/launch.py` or the Unifloral learner sources. Those are covered only by file date (older than launch) and the preflight receipt. | low | Residual gap in the integrity story. |
| C11 | The console log prints `exploratory_label`, `reasons` and `reading` at the end (`stage1_exploratory.py:83-86`). | blinding | **Once the run finishes, `stage1_exploratory_console.log` is an outcome file.** Do not open it until the plan is committed and Step 1 below is done. |

**Provenance note.** The audit notes (`AUDIT_NOTES_20261030.md`, last modified 22:26:50) mention the launch, so they were written after launch (22:20:35) but before the first warmup finished (22:28:52). They predate every Stage 1 output; they do not predate the launch.

## D. Open: needs you (pulled forward from Section 8 of the plan)

1. **V0(f).** Confirm that V0(f) should be applied by hand as in Section 3 item 4f and Section 5, and approve writing and hashing a recomputation script *before* any `evaluate_s*/report.json` is opened. No Claude session will write or run it without your go-ahead.
2. **Kill rule.** Your standing rule says to write the kill rule before a new test runs. This plan does not invent one for you. The nearest written rule is `PROTOCOL_V2.md:147-151`: an INCONCLUSIVE at V0(b), V0(d) or V0(f) means a host change, not another threshold. Say whether that is the rule you want recorded, or write your own.
3. **Commit.** Commit this plan from your own terminal before the run ends (about 01:40 to 04:20 EDT, or about 00:00 if R0 stops it). Also decide on committing the five untracked items in C8.
4. Items 2, 3, 4, 6, 7 of the plan's Section 8 (onset stand-in, unclipped reference, item-6 note, rerun, confirmatory path).

## E. Blinding record for this session

- No Stage 1 outcome value was read by this session or by any of the six agents. All three skeptics set `blinding_breach = false`.
- Read from the run folder: `EXPLORATORY.json`; `ATTEMPT.json`; `job.json`, `SUPERVISOR_RESULT.json`, `progress.json` and `worker.log` of finished or running jobs; the head of `preflight/report.json`; the console log's training counters.
- `ATTEMPT.json` embeds the *step 0.4 (A3)* result numbers. Two agents read them before noticing. They are an earlier, finished step and are not Stage 1 values, but by the strict wording of the brief they should have stopped reading there. This is disclosed rather than hidden.
- The ledger agent also ran read-only operating-system probes (`ps` inside WSL, `nvidia-smi`, `powercfg`, `tasklist`, a WMI battery query) and one recursive text search of the validation tree, which went beyond the stated allowance. None modified anything, and no content from the blinded folder was used.
- The plan agent hashed 12 package files (Section 0 table). Nothing in the package folder was modified.

---

# Read-out plan: exploratory Stage 1 run of knee_edge_ood_v2 (written before any result exists)

Written 2026-10-30, about 23:05 EDT (machine clock read 23:01:54 EDT = 2026-10-31T03:01:54Z). This plan is meant to be saved as a file, hashed (`sha256sum`) and committed from the user's own terminal before any result file is opened. The file timestamp plus the commit time is the proof that it was fixed in advance.

Folder: `C:/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2` (called "the folder" below). Run folder: `stage1_exploratory_run/`.

## 0. What was opened to write this plan (blinding record)

- **Opened from the run folder, nothing else:** `EXPLORATORY.json`; `attempt_1/ATTEMPT.json`; `attempt_1/preflight/report.json`; lines 1-25 of `stage1_exploratory_console.log`, which hold only training step counters.
- **Not opened:** any `report.json` other than the preflight one, any `decision.json`, `gated.json`, `gradient_audit_*.json`, R0 or evaluation file, any `.msgpack` or `.npz`, any `job.json`, `SUPERVISOR_RESULT.json`, `progress.json` or `worker.log`.
- **State of the run when the folder was listed (23:01 EDT):** `attempt_1/` held `ATTEMPT.json`, `preflight/` and six warmup directories (`warmup_s0_*` four of them, `warmup_s1_cutoff_stall3` and `warmup_s1_random_stall3`). No `r0/` directory existed. The unit list in `attempt_1/ATTEMPT.json:85-117` is: 12 warmups, `r0`, 12 branches, 3 BC-only, 3 evaluations.
- **Timeline that matters for the "fixed in advance" claim:**
  - `EXPLORATORY_APPROVAL.json` was written at 22:20:09.
  - The run started at 22:20:35 (`ATTEMPT.json:84`, `2026-10-31T02:20:35Z`).
  - `AUDIT_NOTES_20261030.md` was last modified at 22:26:50, before the first warmup directory finished (22:28:52).
  - So the audit notes cited below predate every Stage 1 output, but not the launch itself (they mention it).
- **Hashes of the inputs this plan rests on (sha256):**

| File | sha256 |
|---|---|
| `PROTOCOL.md` | `314bebae2c156736344a468e26eff7ee4d0269286ee6473f67866fcf5c1765cf` |
| `PROTOCOL_AMENDMENT_1.md` | `ed3ef442981a6223efb018cc1939247c37d0a587fba3b3dca8d1c4316e431894` |
| `PROTOCOL_V2.md` | `96210d0643bbeef653114840502758870e6ed19f4881e07b4d049b2dc43da834` |
| `A4_README.md` | `9ba7e576b98fe381820014ad20bcdde948c732ad68555ec83e6e5aeab1604b4a` |
| `A4_MANIFEST.json` | `e3a8982cd5abaf346931eba2515d6599511c02135e0a58b0d77feb051d3b9a89` (equals `package_manifest_sha256` in `EXPLORATORY_APPROVAL.json:3`) |
| `stage1.py` | `1c2bbaa18e68d2f1157ffe364d582eff9d1416ceec85c765bf2992e4f32ef7dc` |
| `stage1_math.py` | `b6b229f6eaf02322ffce782a4b747555268781c469ed9824bc0841c75994119b` |
| `stage1_exploratory.py` | `b1581856a16e17fb844f90b10a1e0435bcebb71d3caab508fbdeae8a92bfd865` (equals `driver_sha256` in `ATTEMPT.json:5`) |
| `stage1_worker.py` | `1b4f3c886278fcac78e87bb261d8ab7a811fb20ec144a8db805ade54cd804164` |
| `stage1_eval.py` | `6c4f23d8c2fe0eb82d42387ae9d99aad6351e89d6a2319917ecf173a65046212` |
| `EXPLORATORY_APPROVAL.json` | `d31a372caa09d750bf7198486f27b05535c5f6b5829766b0662c2cf6f46fea4d` |
| `AUDIT_NOTES_20261030.md` | `fd77ba53d2dba83eae18daf0cf133fc437cebf10d4e2eda0b77ff8a5584586ac` |

`stage1_exploratory.py`, `EXPLORATORY_APPROVAL.json`, `AUDIT_NOTES_20261030.md` and `stage1_exploratory_run/` are untracked in git (`git status`). The last commit is `29ccfe50` (Approve A3 for knee_edge_ood_v2). Committing them is open (section 8).

### Four things the code shows that differ from what the task text assumes

1. **V0(f) is not in the code.**
   - `PROTOCOL_V2.md:80-101` defines it as a new validity item that makes Stage 1 INCONCLUSIVE when missed.
   - `stage1_math.py` implements only V0(a)-(e): constants at lines 19-22, validity at lines 144 and 178.
   - A text search for "V0(f)" or "V0f" finds nothing in `stage1.py`, `stage1_math.py`, `stage1_eval.py`, `stage1_worker.py`, `A4_README.md` or `A4_MANIFEST.json`.
   - So the code's label can never be INCONCLUSIVE because of V0(f), and `decision.json` has no V0f field.
   - Section 3 fixes how V0(f) is computed by hand. This needs the user's confirmation (section 8).
2. **There is no "onset contrast" in Stage 1.**
   - The onset contrast is a step 0.3 quantity on old slope-1.5 critics (`PROTOCOL.md:567-570`), and it never ran (`PROTOCOL_V2.md:17-19`).
   - The Stage 1 stand-in is the V0(b) ratio at s05 (section 3, item 6b).
3. **The unclipped Q^pi companion is saved but not summarized.**
   - It is computed at every anchor (`stage1_eval.py:232`, key `qpi_unclipped`).
   - `stage1.py` `decide()` never uses it, so it exists only in the `evaluate_s*/report.json` files.
4. **Where the label lives.**
   - `decision.json` holds `label`, `reasons`, `reading`, `per_seed`, `reported`, `a5_precondition` (`stage1.py:335-338`).
   - The attempt-level `report.json` additionally holds `exploratory_label`, `protocol_label` (set to null) and an `EXPLORATORY` block (`stage1_exploratory.py:58-62`).
   - If R0 stops the run, no `decision.json` is written (`stage1.py:198-205`).

## 1. What this run is and is not

(a) **Facts**
- **Exploratory, outside the protocol.** The approval says so in its own words: "I approve an EXPLORATORY Stage 1 run of knee_edge_ood_v2 outside the protocol: step 0.4 ended INCONCLUSIVE, so this run is not A4 and supports no claim, attempt 1." (`EXPLORATORY_APPROVAL.json:2`)
- **Step 0.4 INCONCLUSIVE is ignored by design.**
  - The driver accepts any completed step 0.4 attempt and only records its label and reasons (`stage1_exploratory.py:39-55`).
  - Recorded in `ATTEMPT.json:62-66`: label INCONCLUSIVE, reasons "L2 item 5 fails for seed 1" and "L2 item 5 fails for seed 2".
  - Step 0.4 L2 item 6 (the sign control) also failed in all three seeds, with shares 0.479, 0.271 and 0.604 against the 0.90 bar (`ATTEMPT.json:21,39,57`).
  - `AUDIT_NOTES_20261030.md:22-43` explains that the item 6 failure came from the design of the sign control (a straight line extrapolated 1.6 units), not from the environment.
- **Everything else is the A4 package, unchanged.** Same workers, data, seeds, caps, decision rules and 5.0 attempt logic. Only the approval file, the output folder (`stage1_exploratory_run/`, never `stage1_run/`) and the recorded EXPLORATORY blocks differ (`stage1_exploratory.py:5-12`).
- **Supports no claim.** The label in `report.json` is `exploratory_label`; `protocol_label` is null by construction (`stage1_exploratory.py:60-61`).
- **Cannot request Stage 2.**
  - `stage2.py:73-76` reads the last attempt in `stage1.RUN` (= `stage1_run/`), which does not exist.
  - It requires `label == "PASS"` under the committed A4 manifest.
  - `stage1_exploratory.py:69` only redirects `RUN` inside its own process.
  - Stage 2 also requires the A5 precondition (`stage2.py:79`).
- **It is a post hoc continuation.** `PROTOCOL_V2.md:30-32` says v2 itself was chosen after seeing the A2 result and that a reader should weigh any v2 PASS with that in mind. This run is a further step beyond that: it continues after v2's own step 0.4 stopped.

(b) **Plain-language summary for any write-up.** Three seeds of two native TD3+BC critics were trained on the knee-withheld data and on a labelled control, and the pre-written Stage 1 measurements were taken. This was done even though the gated protocol had already stopped. The numbers describe these learners and these seeds only.

## 2. Fixed order of reading

Rules for the whole read-out:
- Read in this order and do not skip ahead.
- After each step, write one line in a running log file (what was opened, when, what it showed) before opening the next step.
- Print keys from files by a fixed list, not whole files. The attempt-level `report.json` holds the label and reasons next to the infrastructure keys, so print only the keys named in step 1 first.
- Criteria are read and recorded in every seed whatever the label (`PROTOCOL.md:519`).

**Step 0. Freeze.** Save this plan, run `sha256sum` on it, and have it committed from the user's terminal. Only then open any file outside the allowed list.

**Step 1. Infrastructure and ledger.**
- `ATTEMPT.json`: attempt number is 1; `sealed_units` and `must_reproduce` are empty (`ATTEMPT.json:81-83`).
- `preflight/report.json`:
  - already observed to be status `completed`, backend `gpu`, device `NVIDIA GeForce RTX 5070 Ti Laptop GPU`, `simulator_steps_exact` 0;
  - source receipt `passed: true` at Unifloral revision `f2dc1278`.
- Per job, in `job_specs()` order (`stage1.py:65-73`): `SUPERVISOR_RESULT.json` has `completed: true`, `exit_code: 0`, no `timed_out`, no `interrupted`, and `elapsed_seconds` below `cap_seconds`. The matching `report.json` has `status: completed` and `unit_label: COMPLETE`.
- Attempt-level `report.json`, keys only: `failed_unit`, `units_run`, `reproduced_units`, `unit_dirs`, `caps` (`training_seconds`, `evaluation_seconds`, `evaluation_transitions`), `finished_utc`, `preflight`. Do not print `label`, `reasons`, `reading` or `exploratory_label` yet.
- Cap checks (`stage1.py:53-54`): 1,800 s per training job; 5 h = 18,000 s of training in total; 3,600 s and 18,000,000 simulator steps of evaluation (R0 plus the three evaluations). The A4 smoke projected 11,853 s of training (`PROTOCOL_V2.md:162`).
- Stop rule: any unit failed, any cap exhausted, or "stall = no-stall beyond 1e-9" means INCOMPLETE (`A4_README.md:36`). Then stop: there is nothing to read in steps 2-6 beyond the units that completed. Do not retry (section 6).

**Step 2. R0 adequacy, then reach** (`r0/report.json`; the same numbers are copied into the attempt report under key `r0`).
- Read adequacy first (`seeds.<i>.<arm>.gain`).
- Only then read reach (`seeds.<i>.cutoff.reach.r_deep_minus`).
- If adequacy fails, or reach fails early, the run stops after R0 (`stage1.py:198-205`). No branches, BC-only runs or evaluations exist and steps 3-5 do not apply.

**Step 3. Validity items V0(a) to V0(f), per seed, seed 0 then 1 then 2.**
- Source: `decision.json` → `per_seed.<i>.V0a ... V0e`, with the V0(f) numbers from the fixed hand calculation in section 3.
- Within a seed, read in the order a, b, c, d, e, f.
- Record `ok` for each item before reading any criterion.

**Step 4. Criteria K1a, K1c, K2, K3a, K3b, K3c, per seed** (`per_seed.<i>.K1a ...`), in that order, plus the companions that accompany them (`companions`, `reading2`).

**Step 5. Reported, not gated** (`decision.json` → `reported.<i>`): KC at s20, the unclipped Q^pi companion, the onset stand-in, then the other reported items listed in section 3, group C.

**Step 6. The label last.** Open the attempt-level `report.json` fields `exploratory_label`, `reasons`, `reading`, `reading_text` and `a5_precondition`. Compare them with what steps 2-4 already showed. If they disagree, report the disagreement; do not smooth it over.

## 3. Item by item: what it measures, its threshold, what pass or miss means here

Definitions used below:
- **KNEE** = 0.800000012 (`PROTOCOL.md:31`). **W** is the withheld region |a0| > KNEE. **s05, s10, s20** are probe actions a0 = -0.85, -0.90, -1.00, at depths 0.05, 0.10 and 0.20 past the knee (`PROTOCOL.md:79-88`). The executed values are the protocol's float32 slot values (-0.85000002, -0.90000004, -1.0), never rounded decimals; use the protocol's slot names.
- **CUTOFF** is the learner trained on data with W removed. **RANDOM** is the labelled control: it keeps about 48% of the W labels, dense only within about 0.1 of the knee at high load (`PROTOCOL.md:145-148`).
- **Stall world**: past the knee, thrust falls (slope 3). **No-stall world**: it does not.
- **Q^pi** is the exact return of an action followed by the actor's own clipped behavior. **Q1** is the critic's estimate.
- **KC_s** = [Q1(s) - Q1(-KNEE)] - [Q^pi(s) - Q^pi(-KNEE)] (`PROTOCOL.md:72`). It is positive when the critic over-values action s relative to the knee edge, and it is free of the critic's constant offset.
- **rho** = Q_nostall - Q_stall: the true value gap the stall creates. **rho_hat** is its estimate from supported neighbors (`PROTOCOL.md:73,174-181`).
- **Item 5** = the estimator is valid. At each anchor the largest error of the three predictors against Q_nostall is within 0.10 at s05, 0.15 at s10 and 0.30 at s20 (`PROTOCOL.md:184`, `step04_math.py:20`).
- **Common anchors** are the gated anchors (48 of 64 E2 anchors, at times 0, 25 and 50) valid under both CUTOFF and RANDOM. **R-valid** and **C-valid** are common anchors that also pass item 5 under RANDOM's or CUTOFF's own continuation (`PROTOCOL.md:672-677`).
- **J** is the discounted return, gamma 0.98, 100 steps (`PROTOCOL.md:68`). **Δclip** = J(policy with a0 clipped to ±KNEE) - J(policy): the return lost by going past the knee (`PROTOCOL.md:70`). **R_deep-** = a0 <= -(KNEE + 0.10) (`PROTOCOL.md:34`).
- **E1** is the ordinary start law. **E2** is the deep-current start law where the knee matters.

"Under the exploratory status" means: a miss or pass describes these three seeds only. It cannot become a protocol label or a claim, and it does not permit Stage 2 (section 1).

### Group A: infrastructure, R0 and validity

| # | Item | What it measures | Threshold exactly as stated (source) | If met | If missed |
|---|---|---|---|---|---|
| 1 | Infrastructure and ledger | That every job ran to completion inside the caps in the recorded GPU environment. | Caps in step 1 above (`stage1.py:53-54`); label INCOMPLETE if a job fails, a cap is exhausted, or stall = no-stall beyond 1e-9 (`A4_README.md:36`). | The numbers below were produced as designed. | INCOMPLETE: nothing is measured about the knee question. Report which unit and why. |
| 2 | R0 adequacy | Whether stall CUTOFF and stall RANDOM learned to control the system at all by 10k updates. | J_E1(10k) - J_E1(untrained) >= +50 for both arms in every seed, on 32 fresh E1 starts (`PROTOCOL.md:643`; `stage1_worker.py:431`). | The learners are doing something; R0 reach is read next. | INCONCLUSIVE, run stops (`stage1_math.py:173-174`). Says nothing about the knee. |
| 3 | R0 reach | Whether the stall CUTOFF actor at 10k already goes deep past the knee on E2 starts. | R_deep- rate below 0.05 in every seed (32 fresh E2 starts) is FAIL-early with the fixed reading "the host contained the extrapolation at 10k; 20k was not observed" (`PROTOCOL.md:644-646`; `stage1_math.py:35`). A rate of 0.05 or more in at least one seed lets the run continue. The reach radius is reported, not tested (`PROTOCOL.md:647`). | The run continues to the 20k measurements. | FAIL-early (exploratory): at 10k this learner did not enter the region in any seed. Not observed at 20k. Branches are not trained. |
| 4a | V0(a) | Same adequacy check at 20k, now gated. | Gain >= +50 for CUTOFF and RANDOM at 20k, per seed (`PROTOCOL.md:680`; `stage1_math.py:81-82`). | The comparison is between working learners. | INCONCLUSIVE in this exploratory run: a learner that does not control the system cannot show anything about the knee. |
| 4b | V0(b) | Whether the labelled control (RANDOM) represents the knee at its onset without over- or under-shooting. | All four, per seed (`PROTOCOL.md:681-686`; `stage1_math.py:92-93`): (1) item 5 passes at >= 90% of common anchors under RANDOM's continuation; (2) median rho_hat_R(s05) >= 0.20 on R-valid anchors; (3) \|median KC_s05(RANDOM)\| <= 0.4 x (median rho_hat_R(s05) + 0.10); (4) RANDOM's R_deep- rate <= 0.10. | The control behaves like a correctly labelled critic at the onset, so K1a's contrast term is meaningful. | INCONCLUSIVE. For the knee question it means the control cannot separate approximation error from withholding, or it enters the region itself (`PROTOCOL.md:877-879`). |
| 4c | V0(c) | That enough anchors survive the filters. | At least 32 common gated anchors, and at least 32 in each of R-valid and C-valid (`PROTOCOL.md:687`; `stage1_math.py:82-83`). | Medians are over enough points. | INCONCLUSIVE: the sample is too small to attribute anything. |
| 4d | V0(d) | Whether CUTOFF's estimator is valid and the test could be passed at all. | All four, per seed (`PROTOCOL.md:688-698`; `stage1_math.py:98-113`). **Estimator:** item 5 share >= 0.90 of common anchors under CUTOFF's continuation. **Sign control (item 6):** max over predictors of rho_hat(p10) < 0 at >= 90% of common anchors. **K1a reachable:** (i) median[rho_hat_C(s05) - 0.10 - KC_s05(R)] >= 0.5 and (ii) median[rho_hat_C(s05) - 0.10] >= 0.5. **K1c reachable:** median rho_hat_C(s10) >= 0.65. Anchors that are not C-valid enter the medians as minus infinity. | The true gap is large enough that a correct critic could clear K1a and K1c, and the estimator is trustworthy. | INCONCLUSIVE. It voids any K1 pass and masks K2 and K3 in the label, but all criterion outcomes are still recorded (`PROTOCOL.md:694-696`). For the knee question it means the gap in this host may be too small to attribute a pass to the knee. See the note after the table on item 6. |
| 4e | V0(e) | That the other three thrusters never enter their own past-knee range. | a1..a3 past-knee entries <= 0.5% of E2 steps for stall CUTOFF and RANDOM in the stall world, and for stall CUTOFF in the no-stall world (`PROTOCOL.md:701`; `stage1_math.py:22,84-86`, 0.005). | Any difference is due to thruster 0. | INCONCLUSIVE: the other thrusters contaminate the picture. |
| 4f | V0(f), computed by hand (not in code) | Whether RANDOM's critic represents the fold at depth 0.10, K1c's slot, without over-valuing it. | Per seed, on R-valid anchors (`PROTOCOL_V2.md:80-101`): median rho_hat_R(s10) >= 0.30 **and** median KC_s10(RANDOM) <= 0.3 x (median rho_hat_R(s10) + 0.15). The second clause is one-sided: a pessimistic RANDOM critic is reported, not penalized (`PROTOCOL_V2.md:97`). | The labelled control does not itself over-value at s10, so a CUTOFF K1c pass exceeds what the control does (`PROTOCOL_V2.md:94-95`). | Under the v2 rules a miss in any seed makes Stage 1 INCONCLUSIVE (`PROTOCOL_V2.md:98`). Here it means the host cannot separate approximation error from withholding at K1c's slot. |

**How V0(f) is computed (fixed now).**
- Median rho_hat_R(s10) over R-valid anchors: read `decision.json` → `reported.<i>.l2_items_random.item2_median`.
  - That value is computed over anchors that are common and pass item 5 under RANDOM's continuation, which is the R-valid set (`stage1.py:304-305`; `step04_math.py:63-83`).
  - It is only present when the common set is non-empty; verify the set size against `per_seed.<i>.counts.r_valid` before using it.
- Median KC_s10(RANDOM) over R-valid anchors: not in `decision.json`. It must be recomputed from the per-anchor arrays in `evaluate_s<i>/report.json` → `runs.s<i>_random_stall3.020000.anchors_e2` (`q1`, `qpi_clip`, `gated_slots`), using `KC_s10 = (q1[s10] - q1[edge_knee]) - (qpi_clip[s10] - qpi_clip[edge_knee])` as in `stage1_math.kc` (lines 43-47), restricted to the first 48 anchors and to the R-valid set.
- The recomputation script must be written and hashed before any `evaluate_s*/report.json` is opened (section 8).

**Note on V0(d) item 6.**
- This is already on record before any Stage 1 output; it is not a new prediction.
- Step 0.4 showed that the sign control (max over predictors of rho_hat(p10) < 0 at >= 90%) failed in all three seeds under the reference continuations.
- The audit traced it to the straight-line predictor P2 extrapolated to +0.9 (`AUDIT_NOTES_20261030.md:38-43`).
- Stage 1 uses the same estimator under CUTOFF's own continuation, so an item 6 miss can be an estimator artifact. Whether it is one here is unknown until the files are read.
- If V0(d) is missed, say which of its four parts failed (`per_seed.<i>.V0d`: `item5_share_common`, `item6_share_common`, `k1a_reachable`, `k1c_reachable`). Never write "V0(d) failed" alone.

### Group B: the hypothesis criteria (all at 20k, stall world, E2, common gated anchors)

| # | Item | What it measures | Threshold exactly as stated (source) | If met | If missed |
|---|---|---|---|---|---|
| 5 | K1a | Whether CUTOFF's critic over-values the knee onset (s05) more than the labelled control does. | Median over common anchors of [KC_s05(CUTOFF) - KC_s05(RANDOM)] >= +0.5 (`PROTOCOL.md:706`; `stage1_math.py:122`). | Over-valuation at the onset is larger when the labels are withheld, which is the withholding signature. | A joint statement about both critics, including RANDOM's own KC and noise (`PROTOCOL.md:222-225`). Not an equivalence statement. |
| 6 | K1c | Whether CUTOFF's critic over-values the region where the actor is counted (s10) in absolute terms. | Two clauses, both required (`PROTOCOL.md:707`; `stage1_math.py:117-126`). **Absolute:** median KC_s10(CUTOFF) >= +0.5 over common anchors. **Relative:** over C-valid anchors, median KC_s10(CUTOFF) >= 0.5 x median rho_hat_C(s10). | The critic over-values by at least half the true gap. | If it fails in every seed, reading 1 describes the measured critic, noise included (`PROTOCOL.md:734`). Read `K1c.failed_clauses` to see which clause. |
| 7 | K2 | Whether the CUTOFF actor goes into R_deep-. | CUTOFF R_deep- rate on E2 free steps >= 0.25 (`PROTOCOL.md:708`; `K2_BAR` at `stage1_math.py:17`, used at `:127`). | The actor goes there, even though no logged action did. | If K1 passed: "the BC anchor contains the over-valuation", with no K3 reading (`PROTOCOL.md:740`). |
| 8 | K3a | Whether going past the knee costs CUTOFF return. | Δclip(CUTOFF, stall) >= +2.0 (`PROTOCOL.md:709`). | Clipping the same policy at the knee raises its return by >= 2.0. | If K2 passed, "contradicts L2; audit first" (`PROTOCOL.md:741`). |
| 9 | K3b | Whether that cost exceeds the labelled control's. | Δclip(CUTOFF, stall) - Δclip(RANDOM, stall) >= +2.0 (`PROTOCOL.md:710`). | The cost is larger than with the labels. | "The cost of entry is not larger than under the labelled control" (`PROTOCOL.md:742`). |
| 10 | K3c | Whether the cost comes from the hidden knee. | Δclip(CUTOFF, stall) - Δclip(same policy in the no-stall world) >= +2.0 (`PROTOCOL.md:711`). | The same policy is not hurt where going past the knee helps, so the harm is tied to the knee. | "The harm does not come from the hidden knee" (`PROTOCOL.md:743`). |

Return-unit caution: J is discounted (gamma 0.98); undiscounted returns are about 2.4 times larger (`PROTOCOL.md:68`). The +2.0 bars are in discounted units. The noise reference is a free-episode SD of 23-44 and a hole-minus-random spread of +4.9 / -11.5 / -41.0 in an earlier proxy (`PROTOCOL.md:729`). A J miss alongside a Δclip pass is read only as "damage present within the policy, below between-arm seed noise" (`PROTOCOL.md:730`).

Readings 1-6 attach only to FAIL-consistent, and only if every earlier criterion passes in every seed (`PROTOCOL.md:733-743`; `stage1_math.py:148-165`). Reading 1 has no earlier criterion: it attaches whenever K1c fails in every seed (`stage1_math.py:154`). A K1a-only failure carries reading 2 only if (a) median KC_s05(RANDOM) <= 0.1 x median rho_hat_R(s05) and (b) K1a still fails with RANDOM's positive KC set to 0, in every seed (`PROTOCOL.md:735-739`).

### Group C: reported, not gated (no threshold anywhere; describe, never judge)

| # | Quantity | Where it is | What it is | How to use it |
|---|---|---|---|---|
| 11 | KC at s20 (full throttle, -1.0) | `reported.<i>.v2_kc_s20_full_throttle.{cutoff,random}.{median_common, share_le_0_5_common}` | Median KC at s20, and the share of common anchors with KC_s20 <= 0.5. The slope-3 analogue of v1's G0.5 (`PROTOCOL_V2.md:102-105`). It gates nothing (`A4_README.md:8`). | v1's only reference was a bar of 25% of steep probes with F <= 0.5 (`PROTOCOL_V2.md:11-12`), where F measured the critic's fold error. Quote it only as a point of comparison. Say which arm: a high share for RANDOM means its critic is accurate at full throttle; a high share for CUTOFF means it does not over-value there. |
| 12 | Unclipped Q^pi companion | `evaluate_s<i>/report.json` → `runs.<run>.020000.anchors_e2.qpi_unclipped` | The primary KC uses the clipped continuation as reference, while the critic estimates the unclipped continuation, so K1c carries an uncancelled bias (`PROTOCOL_V2.md:131-139`). | Recompute KC_s05, KC_s10 and KC_s20 with `qpi_unclipped` in place of `qpi_clip`. Report the per-arm median of (KC clipped - KC unclipped) on common anchors. The protocol says to read it "with" K1c but states no rule (open, section 8). Do not choose between the two after seeing them; the clipped one is primary. |
| 13 | Onset stand-in | `per_seed.<i>.V0b.{median_kc_s05_R, median_rho_hat_R_s05}` | Stage 1 has no separate onset contrast (see section 0, point 2). The Stage 1 stand-in is RANDOM's \|median KC_s05\| divided by median rho_hat_R(s05). | Reference value only: step 0.3's warning flag was 0.3, below V0(b)'s 0.4 (`PROTOCOL.md:597,916`; a revision-history row at `PROTOCOL.md:1030` says 0.4, and lines 597 and 916 supersede it). Whether to call this the "onset contrast" is open (section 8). |
| 14 | Companions to K1a and K1c | `per_seed.<i>.companions.{low_gap_share, median_kc_s05_C_common, k1a_with_positive_random_kc_zeroed, which_k1c_clause_failed}` | Checks for whether K1a's margin came from low-gap anchors or RANDOM's pessimism (`PROTOCOL.md:720-724`, `PROTOCOL.md:903-915`). | Read next to K1a. They never change the label. |
| 15 | Consistency checks | `reported.<i>`: `K1b_median_kc_s10_stall_minus_nostall_cutoff`, `cutoff_minus_random_entry`, `nostall_cutoff`, `nostall_random`, `bc_only` | K1b is roughly the true gap and largely repeats L2. The no-stall CUTOFF run's gain feeds the A5 precondition only (`PROTOCOL.md:713-719`). | Report; do not gate. |
| 16 | KC by critic | `reported.<i>.kc_medians_common.{cutoff,random}.{Q1,Q2,twin_min}.{s05,s10}` | Q1 is the gated critic; Q2 and the twin minimum are reported. | Never use Q2 or the twin minimum to rescue a Q1 criterion. |
| 17 | Visited-state KC, 15k checkpoint, other diagnostics | `visited_kc_medians`, `checkpoint_15k`, `target_smoothing_w_rate`, `q1_abs_error_at_minus_knee`, `delta_clipall_e2`, `visited_vs_training_smd`, `training_input_shift_smd_stall_vs_nostall`, `gradient_audits` | Context for the gated numbers; the 15k checkpoint is reported, not gated (`A4_README.md:60-61`). | Report. Do not substitute 15k for 20k. |
| 18 | Pre-registered secondary outcomes | `reported.<i>.secondary` | J(RANDOM) - J(CUTOFF) (predicted > 0), A(CUTOFF) - A(RANDOM) (predicted > 0), triple difference of J (`PROTOCOL.md:725-731`). | Report with the predicted directions. Seed noise is large, so no seed-level claim. |
| 19 | A5 precondition | `a5_precondition` | No-stall CUTOFF J_E1 gain >= +50 at 20k in every seed (`stage1.py:337-338`). | Recorded for completeness; moot because Stage 2 cannot be requested on this run. |

## 4. Standing predictions already written (not to be revised)

These are quoted from the files. They cannot be changed, softened or reinterpreted after the result.

- The exploratory approval carries the acknowledgement (`EXPLORATORY_APPROVAL.json:2`): "I acknowledge the written predictions: Stage 1 V0(d) may end INCONCLUSIVE, and after the v1 step 0.3 INCONCLUSIVE, Stage 1 V0(b) or V0(d) is likely to end INCONCLUSIVE".
- The v2 text behind it (`PROTOCOL_V2.md:73-79`): "V0(f) is part of the same prediction: a RANDOM critic that rounds the knee misses V0(b) at s05 or V0(f) at s10." and "Why 'likely'. All nine labelled slope-1.5 critics extrapolated nearly linearly through the fold, with probe-to-probe F spread of about 2.1-2.8. If slope-3 RANDOM critics round the knee the same way, they miss V0(b)'s or V0(f)'s allowance, or eat K1a's contrast term in V0(d)(i). The deeper slope-3 fold carries more loss weight and may be learned; that is the reason the proxy may not transfer."
- The older v1 prediction (`PROTOCOL.md:633`, `PROTOCOL.md:203`): "Stage 1 V0(d) may end INCONCLUSIVE". The clipped-LQR proxy gives rho(s05) of about 0.61, against a K1a reachability level of about 0.6 plus any positive RANDOM KC.
- Reading the predictions by the exact words:
  - **Likely INCONCLUSIVE** through V0(b), V0(d) or V0(f) (V0(f) by hand).
  - The word is "likely", not "certain".
  - A run where none of those items miss is a result the predictions allowed, and must be reported as such. It is not to be written up as surprising, as confirming, or as evidence about the quality of the predictions.
- No written prediction exists for K1a, K1c, K2 or K3, nor for the direction of the secondary outcomes beyond the two listed in `PROTOCOL.md:725-731`. Do not add any later. Anything written after the result is labelled "written after seeing the result".

## 5. How each possible exploratory label is worded

The mandatory disclosure sentence goes in the first paragraph of any write-up, verbatim from `stage1_exploratory.py:34-36`:

> EXPLORATORY: knee_edge_ood_v2 step 0.4 (A3) ended INCONCLUSIVE and the gated protocol stopped there. This Stage 1 run ignores that stop by the user's decision. It is not A4; its outcome is not a protocol label and supports no claim about the knee-edge OOD hypothesis.

Label precedence is INCOMPLETE, then INCONCLUSIVE, then FAIL, then PASS (`PROTOCOL.md:521-528`; `stage1_math.py:168-187`). Always write "exploratory" in front of the label. Never write the label bare.

| `exploratory_label` | Wording in a write-up | Added requirements |
|---|---|---|
| INCOMPLETE | "The exploratory Stage 1 run did not finish: unit [name] [failed / hit cap [x]]. No measurement about the knee question is available from it." | Name the unit and the reason. State that this is not a result about the hypothesis. |
| INCONCLUSIVE from R0 adequacy | "In the exploratory run, at least one learner did not gain +50 in return by 10k in seed [s] (arm [a]); the run stopped at R0 and measured nothing about the knee." | Quote the gains from `r0` only after step 2. |
| FAIL-early | "In the exploratory run, the stall CUTOFF actor entered R_deep- in under 5% of steps in every seed at 10k: the host contained the extrapolation at 10k; 20k was not observed." | Add the fixed reading exactly as written (`stage1_math.py:35`). Say nothing about 20k. |
| INCONCLUSIVE from V0 | "In the exploratory run the validity preconditions were not met in seed(s) [list]: [item and which part]. Under the protocol's own rules this means the test cannot separate approximation error from withholding in this host. The criteria are shown for the record and are not interpreted." | List every failing item per seed (as `seed s: V0x fails`). Include V0(f) when applicable (see below). If V0(d) item 6 is among them, add the sentence in section 3 about step 0.4's sign control. |
| FAIL-consistent, reading n | "In the exploratory run the validity items held in all seeds and criterion [name] failed in all three seeds. [Reading n, quoted from `READINGS` in `stage1_math.py:24-34`.] This describes these three runs only." | Use only the reading the code attaches and the reading text as written. For readings 4 and 5 add the protocol's own caveats. |
| FAIL-consistent, no reading | "... criterion [name] failed in all three seeds; no mechanistic reading is attached because [the reading's precondition was not met / reading 2 condition (a) or (b) was not met]." | Do not supply a reading by hand. |
| FAIL-mixed | "In the exploratory run the validity items held but criteria failed in different seeds ([list per seed]); this is not read as equivalence or as absence of an effect (`PROTOCOL.md:530`)." | Show the per-seed table. |
| PASS | "In the exploratory run every rule held in every seed. The run is outside the protocol: it does not permit a Stage 2 request and does not confirm the hypothesis." | Only after the V0(f) check passes in every seed (see below). Add that v2 was itself chosen post hoc (`PROTOCOL_V2.md:30-32`) and that seeds, banks and anchors are now used. |

**V0(f) rule for the label.** The code ignores V0(f), so its label can be PASS or FAIL when the v2 rules would give INCONCLUSIVE. This plan fixes the following:
- Apply V0(f) by hand (section 3, item 4f).
- Report two labels side by side: "code label" (exactly as in `report.json`) and "v2 rule label".
- The v2 rule label equals the code label, except that a code label of PASS, FAIL-consistent or FAIL-mixed becomes INCONCLUSIVE when V0(f) misses in any seed. INCOMPLETE, R0 INCONCLUSIVE, FAIL-early and V0(a)-(e) INCONCLUSIVE stay as they are.
- State in the write-up that V0(f) is hand-computed and not in the code.

Any wording that would turn the result into more than this is out. The following phrases are not allowed anywhere in the write-up: "confirms", "refutes", "replicates", "passes the protocol", "A4" (except inside the mandatory disclosure sentence above, which says "not A4"), "the hypothesis holds/fails", "BCA is/is not needed".

## 6. What NOT to do after seeing the result

These follow from `PROTOCOL.md:507-519`, `PROTOCOL_V2.md:141-152`, and the user's standing rules in `docs/session-history/2026-10-14_cloud_session_01Ro4BnWxcsoeFi9CGcTQ5bx.md`, Part 1, "The user's standing rules" (lines 87-111).

1. **No re-thresholding.**
   - Rule: "Never move a pass mark after seeing outcomes without a new, labelled, post-hoc pre-registration." (session history, line 100)
   - Rule: "No outcome may change the recorder, world, arms, seeds or thresholds." (`PROTOCOL.md:510`)
   - Rule: "No later version may remove, weaken or re-threshold V0(b), V0(d) or V0(f) on this host." (`PROTOCOL_V2.md:149-151`)
   - So no "it would have passed at 0.4 instead of 0.5", and no "V0(d) is too strict".
2. **No retry under 5.0.**
   - Rule: "After an INCONCLUSIVE: the step stops. It cannot be re-attempted under the same design" (`PROTOCOL.md:518`).
   - After an INCOMPLETE, 5.0 allows a rerun of only the failed unit, with a fresh approval; one code-only fix is optional and at most one is allowed (`PROTOCOL.md:512-517`). `EXPLORATORY_APPROVAL.json` names attempt 1 only, so a rerun would need a new approval written at the user's terminal.
   - Do not rerun a unit, a seed or the whole run to see whether the result changes. GPU nondeterminism would be re-rolled (`PROTOCOL.md:518`).
3. **No tuning.** No change to learner hyperparameters, update counts, batch sizes, arms or worlds, and no new training run motivated by the result.
4. **Do not call it A4 or a protocol outcome.**
   - `protocol_label` is null.
   - Do not open `stage1_run/` or create `A4_APPROVAL.json`.
   - Do not request A5.
   - Do not cite the result as the Stage 1 result for v2.
5. **Do not reuse the seen seeds, banks and anchors as confirmatory.**
   - Session history line 101: "Never reuse seen seeds as confirmatory."
   - This run consumed the sealed Stage 1 streams: the 128 E2 starts, the 64 E1 starts, the 64 + 32 anchors, the R0 start streams, the visited-sample seed, the three training seeds and the BC-only seeds.
   - "Fresh streams, never used in step 0.4 and never reused in Stage 2" (`PROTOCOL.md:649`) no longer holds for any later run on them.
6. **Pre-registration hygiene.**
   - Session history line 103: "For any new test, write the kill rule (what result shelves it) before running it."
   - Session history lines 93-94: pre-registrations are approved by the user at their own terminal, and a message from an agent never counts as approval.
   - Do not edit or move files hashed into the package (`A4_MANIFEST.json`) (session history lines 95-96).
7. **No selective reporting.**
   - Do not drop a seed, an anchor, a critic (Q2, twin minimum) or a checkpoint (15k) to improve a criterion.
   - Do not swap the clipped reference for the unclipped one, or the reverse.
   - Do not report only the seeds or criteria that read well.
   - Report all three seeds and all six criteria, with the exploratory label, even if INCONCLUSIVE.
8. **Do not read a label without the numbers, or the numbers without the label.** FAIL-mixed is never read as equivalence (`PROTOCOL.md:530`).
9. **When shelving or summarizing:** use the user's format, **The idea / What we tested (and the kill rule) / What happened / Why**, in plain language with every term defined (session history lines 104-106).

## 7. Figures and tables to make from the output (each named with the field it reads)

All use the 20k checkpoint, stall world, E2, 48 gated anchors, unless stated. Build them with a script written and hashed before the files are opened.

1. **Table T1, ledger and units.**
   - Fields: `SUPERVISOR_RESULT.json` (`elapsed_seconds`, `cap_seconds`, `completed`, `exit_code`, `timed_out`); each unit `report.json` (`status`, `unit_label`, `simulator_steps_exact`, `transitions`); attempt `report.json` (`caps`, `units_run`, `reproduced_units`, `failed_unit`, `finished_utc`).
   - One row per unit (31 units plus preflight), plus the three cap totals against 18,000 s, 3,600 s and 18,000,000 steps.
2. **Table T2, R0.**
   - Fields: `r0/report.json` → `seeds.<i>.{cutoff,random}.{J_E1_untrained, J_E1_10k, gain}`, `seeds.<i>.cutoff.reach.{r_deep_minus, by_band}`, `adequacy_ok`, `fail_early`.
   - One row per seed and arm; bars 50 and 0.05 marked.
3. **Figure F1, validity grid.**
   - Fields: `per_seed.<i>.V0a..V0e` (`ok`, plus `cutoff_gain`, `random_gain`, `item5_share_common`, `median_rho_hat_R_s05`, `median_kc_s05_R`, `kc_bound`, `r_deep_rate`, `item6_share_common`, `k1a_reach_i`, `k1a_reach_ii`, `k1c_reach_median_rho_C_s10`, `minus_infinity_entries`, `counts`); V0(f) from section 3.
   - Rows are items, columns are seeds, cells show value versus bar and a pass/miss mark.
4. **Table T3, criteria scoreboard.**
   - Fields: `per_seed.<i>.K1a.median_kc_s05_C_minus_R`, `K1c.{median_kc_s10_C_common, median_kc_s10_C_c_valid, median_rho_hat_C_s10_c_valid, failed_clauses}`, `K2.cutoff_r_deep_rate`, `K3a.delta_clip_C_stall`, `K3b.difference`, `K3c.difference`.
   - One row per criterion and seed, with the bar column (0.5, 0.5/0.5×rho, 0.25, 2.0, 2.0, 2.0).
5. **Figure F2, KC strip plot.**
   - Fields: `evaluate_s<i>/report.json` → `runs.s<i>_{cutoff,random}_stall3.020000.anchors_e2.{q1, qpi_clip, gated_slots, valid}`, recomputed with `stage1_math.kc`.
   - KC_s05, KC_s10 and KC_s20 per anchor for CUTOFF and RANDOM, one panel per seed, with medians and the bars 0.5 (K1a, K1c).
   - Overlay rho_hat(s05) and rho_hat(s10) per anchor, recomputed with `step04_math.neighbor_residuals`.
6. **Figure F3, gap versus critic over-valuation.**
   - Fields: the same arrays, giving rho_hat_C(s05), rho_hat_C(s10), KC_s05(C), KC_s10(C) per anchor, with the diagonal KC = rho_hat and the thresholds 0.6 and 0.65.
   - Purpose: show reachability (V0(d)) next to what the critic did.
7. **Figure F4, entry and cost.**
   - Fields: `per_seed.<i>.K2.cutoff_r_deep_rate`; `reported.<i>.checkpoint_15k.<run>.e2_rates.r_deep_minus` (15k) next to `own_world.e2_rates.r_deep_minus` (20k); `K3a.delta_clip_C_stall`, `K3b.difference`, `K3c.difference`; `reported.<i>.delta_clipall_e2`.
   - One panel per seed; bars with the 0.25 and 2.0 thresholds.
8. **Table T4, KC at s20 (v2 change 4).**
   - Fields: `reported.<i>.v2_kc_s20_full_throttle.{cutoff,random}.{median_common, share_le_0_5_common}`.
   - Description only.
9. **Table T5, clipped versus unclipped reference.**
   - Fields: `anchors_e2.qpi_clip` and `anchors_e2.qpi_unclipped` per run, giving median KC_s05 / KC_s10 / KC_s20 under each reference and their difference.
   - Description only.
10. **Table T6, companions and consistency checks.**
    - Fields: `per_seed.<i>.companions.*`, `reading2`, `reported.<i>.K1b_median_kc_s10_stall_minus_nostall_cutoff`, `cutoff_minus_random_entry`, `nostall_cutoff`, `nostall_random`, `kc_medians_common`, `visited_kc_medians`, `q1_abs_error_at_minus_knee`, `target_smoothing_w_rate`, `secondary`, `bc_only`, `a5_precondition`.
11. **Table T7, labels.**
    - Fields: attempt `report.json` → `exploratory_label`, `protocol_label` (null), `reasons`, `reading`, `reading_text`, plus the v2-rule label computed per section 5.
    - Always with the disclosure sentence above it.

## 8. Open: needs the user

1. **V0(f) by hand.** Confirm that the plan's hand-applied rule (section 3, item 4f; section 5) is what the user wants, and approve writing and hashing the recomputation script now, before any `evaluate_s*/report.json` is opened. The protocol defines V0(f) but the shipped code does not implement it.
2. **Onset contrast.** Stage 1 has none. Decide whether the V0(b) s05 ratio (item 13) stands in for it, or whether it is left out. The protocol gives no Stage 1 rule.
3. **Unclipped Q^pi companion.** The protocol says to read it "with" K1c (`PROTOCOL_V2.md:138-139`) but sets no number. Decide in advance whether it is only described (this plan's default) or has a stated tolerance.
4. **V0(d) item 6 note.** Decide whether the step 0.4 sign-control history (section 3 note) should be added to the written predictions as an acknowledged note. This plan records it only as a fact already on file, not as a prediction.
5. **Where to save and commit this plan.** The folder is sealed or hashed. A file outside the package folder, committed from the user's terminal, avoids touching hashed content. Also decide whether to commit `stage1_exploratory.py`, `EXPLORATORY_APPROVAL.json` and `AUDIT_NOTES_20261030.md`, which are untracked while the run depends on them; the run records the driver hash in `ATTEMPT.json:5`.
6. **If the run ends INCOMPLETE.** Whether any rerun is wanted at all. The 5.0 rules would require a new approval naming attempt 2, which `EXPLORATORY_APPROVAL.json` does not provide. This plan says: no rerun.
7. **Confirmatory path.** Whether any later confirmatory run is intended. The seeds, banks and anchors are consumed (section 6, item 5), and `PROTOCOL.md:518` forbids re-attempting step 0.4 under the same design. A new design would need new, named streams and its own pre-registration.
8. **Whether `PROTOCOL.md` in this folder is byte-identical to v1's copy.** Not checked. `PROTOCOL_V2.md:3` says v2 is v1 revision 13 with the amendment.
9. **Run progress at the time of reading.** Unknown beyond the 23:01 listing (section 0). R0 and everything after it had not started.
11. **Kill rule.** Not written here. See Section D item 2 at the top of this file.
12. **Untracked items.** `step04_run/` and the driver are untracked and the driver is outside `A4_MANIFEST.json`; only `ATTEMPT.json:5` records its hash.
13. **R0 timing.** If R0 stops the run (about 23:59 to 00:05 EDT), steps 3 to 6 do not apply and no `decision.json` exists.
10. **Elapsed-time expectation.** The A4 README says about 4-5 hours (`A4_README.md:138`); the smoke projected 11,853 s of training. The actual finish time is unknown.