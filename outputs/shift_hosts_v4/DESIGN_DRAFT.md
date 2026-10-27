# Unifloral hosts, version 4 (S7 supplement): design draft, not frozen

Status: a draft written before any version-3 result was read. Version 3's driver finished on 20 October 2026 at
11:03:13 UTC with no halt. Its results.json was built by version 3's own code under version 3's approval
(runs/shift_hosts/results_v3.json, SHA-256 73c3f12e07fac5d026381a7e2c50b01fc681442e3109c5a131e4d69d1d6ba48b) and had
not been opened when this file was committed. Nothing in this draft trains; version 4 runs only after its own freeze,
stage-0 suite and the owner's terminal approval.

## What version 3's S7 checks stopped (from driver/state.json "s7_stopped"; no returns)

- medium-expert: ReBRAC (learner; stage 1, rebrac-standard and rebrac-iw_ones), TD3+BC's IW setup (stage-2 seed-0
  100k-step actor), IQL (learner; iql-alone seed-0 100k-step actor).
- medium: TD3+BC (learner; td3_bc-alone seed 0), IQL (learner; iql-alone seed 0).
- Version 3's driver/state.json SHA-256 e577472ffc68a18079c1cd4ae2f778315d506b6db47a23dad0e5c295fc21df11; ledger.json
  25ec5762ab14b1e9e47cf38346cb231908003a904618af385632d4c0a5ca7fbf.

## The owner's instruction and decisions (chat, 20 October 2026)

- 06:39 UTC: "dont stop experiments, let them run in full so we can work from there towards solution"; 06:42 UTC:
  "same with IW BCA".
- D1, approach: "Supplement after v3" (version 3 finishes untouched and is judged once by its own code; version 4
  trains and scores every scope S7 stopped, on both qualities, shown and never judged).
- D2, budget: "Its own ledger" (the supplement's own ledger under section 12's per-quality cap numbers; the
  study-wide total across all versions is disclosed).
- D3, S7 coverage in the supplement: "Yes, plus 50k-step actors" (S7 lanes on the final 100k-step actor of every
  supplement run at seeds 0-4, and on the 50k-step actors as well; processor time only).
- D4, S5 on a BCA setup inside the supplement: "Halt until traced" (unchanged rule; --accept-s5 exists).
- Unchanged: S7's limits (open loop 1e-7, closed loop 0.05 points on every start), every claim, mark and bar;
  version 3's stopped scopes stay "not judged" with p = 1; the S7 fix, if any, is a separate later version with its
  own kill rule.

## The design (multi-agent design workflow wf_2b24777e-1f6: three designs, three judges, synthesis)

The design below predates the owner's D3 answer; D3's 50k-step addition is a small change to the S7 lane labels.

1) RECOMMENDATION

Build version 4 as a post-hoc "S7 supplement". It starts only after v3's driver has exited. Five pieces:

- **v3 stays untouched and is judged once.** v3 is judged by its own code, under its own approval, before any v4 file reaches /root/bca-hosts. The output goes to runs/shift_hosts/results_v3.json, a git-ignored folder. A copy with its sidecar is committed at outputs/shift_hosts_v3/results.json.
- **Own root, own ledger, stage 1 rerun.** v4 writes only under runs/shift_hosts/supplement_v4/, with its own ledger. It reruns stage 1 (20 runs, about 0.24 card-h). That gives fresh S2, S4, S8, S9, S10 and S6 checks under v4's fingerprint, a real S6 decision, and a reproduction reference.
- **Every stopped scope is trained and scored in full.** The list is taken mechanically from v3's final state.json "s7_stopped" on both qualities:
  - a learner scope: sham_alone (choice-19 learners only), alone, standard and iw at seeds 0-4, and room x5. That is 21 runs for TD3+BC or IQL and 20 for CQL or ReBRAC.
  - an IW scope: iw at seeds 0-4, plus that learner's alone and standard at seeds 0-4 as comparators under the same fingerprint. That is 15 runs.
- **A failed S7 check is recorded and nothing is dropped.** This is done by deleting code in stages.py and adding one frozen run-list filter, not by adding a second driver mode.
- **Judged results cannot move.** claims.py, export.py, runner.py, evaluate.py, band_readout.py, gate.py and every training and scoring file stay byte-identical to v3's approved manifest, and a stage-0 test enforces this. So no judged badge, p-value, Holm step, mark or bar can change. The final outputs/shift_hosts/results.json is v3's file plus added keys only: qualities.<q>.s7_shown, s7_supplement and one appended caveat. A canonical-JSON equality check refuses to write anything else.

S7's limits stay as they are (open loop 1e-7, closed loop 0.05 points on every start). v3's stopped scopes stay "not judged" with p = 1.

Why this design: all three judges ranked it first. It is the only option where no judged number passes through new code, a second analysis or mixed fingerprints. It is also the cheapest, at about 5-6 card-h central. A halt inside the supplement touches only shown numbers.

Changes from the winning draft, each checked against the code:

- **Stage 1 reruns** instead of seeding S6 from v3. This removes the skip-stage-1 path, the "S6s" check key (export.driver_rule keeps only keys whose prefix equals the rule, export.py:918) and the inherited-check disclosure.
- **Deletions instead of a second mode** (engineering judge).
- **v3's entries are not pre-seeded** into the supplement's state.json. Pre-seeding would let the dedupe at stages.py:574 swallow a v4 failure on the same run id. It would also let the exit-3 cross-check at stages.py:1069-1070 pass on the carried entry.
- **IW-scope comparators are retrained under v4.** So no shown number pairs two fingerprints, and no cross-version S4 code is needed.
- **budget.py does not change.** The supplement gets its own ledger under section 12's cap numbers (decision D2).
- **S7 verdicts at seeds 1-4 are kept out of state["checks"].** export._s7_check_key (export.py:891-899) parses only the "S7/<l>/<s>" and "S7/<q>/seed0/<l>/<s>" shapes, so any other failed key would make integrity S7 read "failed". export.own_checks already reads every lanes/s7-step*/s7.json (export.py:528) and treats a failure as covered once s7_stopped has an entry for it.
- **chattr +i, not chmod a-w,** to protect v3's folders. The driver runs as root, and root ignores mode bits.
- **The new tests go into the existing test_stages.py and test_claims.py**, so run_stage0.sh and test_all_stage0.MODULES do not change.
- **The shown block puts no field beside a bar.** No no-shift mean next to the Claim-0 bar, no coverage next to 4a's [89%, 96%], no delta next to 1.8.

2) EXACT CODE CHANGES (anchors are v3 at HEAD 6f9ce2b)

experiments/shift_hosts/stages.py

- **Module docstring, 35-38 and 54-62.** Add a version-4 paragraph:
  - S7 records a failed check in state.json s7_stopped and drops nothing.
  - On real data the driver trains only the frozen supplement plan (prespec.json "supplement") under runs/shift_hosts/supplement_v4.
  - Stage 1 always runs whole.
  - Keep v3's text as history.
- **Imports, line 94 onward.** Add `from experiments.shift_hosts import supplement as S`.
- **Config, 396-416.** Add `plan: dict = None` ({quality: {learner: [setups]}}; None means the full grid), `s7_all_seeds: bool = False` and `supplement: dict = None` (the frozen block, for provenance).
- **config_record, 423-428.** Add `plan` (as given) and `s7_all_seeds`. A resume with another plan is then refused.
- **Driver.__init__, 469-473.**
  - The new-state dict gains `s7_results=[]`, and `supplement=` {plan_sha256, v3 fingerprint, the v3 source SHA-256 values} when cfg.supplement is set.
  - Add `self.state.setdefault("s7_results", [])`.
  - Never copy v3's s7_stopped entries into the state.
  - Keep `counts["cards_dropped"]` and `counts["lanes_dropped"]` (always 0) so the status schema stays the same.
- **status, 535-549.** Add `s7_results=len(self.state["s7_results"])` and `supplement=self.state.get("supplement")`.
- **Delete Driver.s7_stopped (558-563) and Driver._drop_stopped (584-604).**
- **New `in_plan(quality, learner, setup)`.** It returns `self.cfg.plan is None or setup in self.cfg.plan.get(quality, {}).get(learner, ())`.
- **s7_failed, 565-582.**
  - Keep 570-577: the scope lookup, the halt when a setup has no scope, and the once-only append. The dedupe key stays the five fields.
  - Add `version=4` to the appended dict only, not to the key.
  - Delete line 581, `self._drop_stopped()`.
  - New log line: `S7 (version 4): {label}: the float32 check failed (see its s7.json); recorded; nothing is dropped; {learner|IW setup} on {quality} is shown, never judged`.
  - Return None.
- **_card_done.** Delete 867-869, so every completed run reaches lanes_for (870-872).
- **lanes_for, 947.** The condition becomes `job.stage == "stage1" or ((seed0 or cfg.s7_all_seeds) and job.setup in STAGE2_SETUPS)`.
- **_verdict, 1019-1028.** At the start of the s7 branch, after `body` verifies, upsert into `state["s7_results"]` keyed by lane label, so a resume does not duplicate it:
  - label, run, quality, learner, setup, seed (parsed from the run id), step (from the folder name);
  - s7_passed, open_loop_passed = body["passed"];
  - closed_loop = {k: {max_points, mean_points, max_return, passed}} for each "readout/bank" key of body["closed_loop"] (evaluate.closed_loop_check, evaluate.py:607-609);
  - file, sha256, at_utc.
  - Save the state. The rest of the branch is unchanged.
- **_lane_step_done.** Delete 1084-1089. Keep 1064-1077 as they are: exit 3 at the last step reads the verdict; an exit status that disagrees with s7.json halts; exit 3 at an earlier step halts.
- **stage1_checks.** The logic is unchanged; 1135-1138 still record through s7_failed. Change the text at 1163-1166 to "S7 failed (recorded; shown, never judged; version 4) on medium-expert: ...".
- **plan_units, 1221-1253.**
  - `live = lambda l, s: self.in_plan(quality, l, s)` (1228).
  - `lanes["s7"] = 1` when `k == 0 or cfg.s7_all_seeds`; the viewer stays at k == 0 only.
  - Update the docstring.
- **unit_jobs, 1255-1268.** Remove the `every` parameter; `live` = in_plan. Update the callers: line 1360, and test_stages lines 357, 373 and 434.
- **stage2.** Change only the text at 1286-1291: "nothing to run: this unit has no supplement run (version 4)".
- **seed_checks.** Delete 1319-1320. Keep the len(r) < 2 skip (1323-1328) as a guard, with a note naming version 4's plan.
- **room_checks.** Delete 1340-1341. Room runs exist only for learner scopes, whose alone runs are in the same root (1342).
- **s7_seed0_checks, 1353-1386.**
  - Iterate `self.unit_jobs(quality, "seed0")`.
  - Delete `stopped` (1362), the "never trained" branch (1366-1369) and the "before its lane ran" branch (1377-1379).
  - A failure calls s7_failed (recorded, no halt).
  - A missing or unverified record or s7.json halts (S7 or error).
  - Check keys stay "S7/<q>/seed0/<l>/<s>". Never write S7 check keys for seeds 1-4.
- **run(), 1441-1447.** Finish line: "finished: every unit of every stage ran and no stop rule halted; S7 failures recorded under version 4 (shown, never judged): <learner[ IW] on quality>, ...".
- **write_stop, 1460-1465.** Add a STOP.txt line with the count and runs of s7_stopped entries.
- **frozen_config, 1509-1537.**
  - Also read `pre["supplement"]` (missing means refused).
  - `plan = S.validate(pre["supplement"], learners, seeds, choice19)`.
  - Add root=S.ROOT, plan, s7_all_seeds and supplement to `out`.
- **check_real_args, 1546.**
  - Compare --root with `S.ROOT`, not RN.RUNS.
  - Refuse `--supplement-plan` and `--s7-all-seeds`.
- **config_from, 1582-1584 and 1614-1618.**
  - The real default root is S.ROOT.
  - A dry run refuses RN.RUNS, S.ROOT and anything inside S.ROOT.
  - Dry: plan from `--supplement-plan PLAN.json` (or None) and `--s7-all-seeds`.
  - Real: plan and s7_all_seeds from `frozen`.
- **add_run_args, 1626-1645.** Add the two dry-run-only options; fix the --root help text.
- **preflight, 1648-1671.**
  - The ledger is at `S.ROOT / "driver" / "ledger.json"`, so it is absent on the first start and both stage-0 hours (0/0) are required.
  - Add `S.check_sources(frozen["supplement"])`. It re-hashes v3's runs/shift_hosts/driver/state.json, driver/ledger.json, runs/shift_hosts/results_v3.json and outputs/shift_hosts_v3/results.json against prespec.json, and checks that approval_fe9fb77b0851c75b.json (with its sidecar) is committed in HEAD.
  - Add `S.code_unchanged(gate.code_manifest())`. It must be empty, or the start is refused.
- **main, 1772.** The real default root for status and request-stop is S.ROOT.
- **Unchanged:** S7_SCOPE (claims.py asserts it equal), S7_INJECT, lane_main, lane_permission, s7_result, card_job, _s7_lane, check_s2/s4/s8/s10, the env functions, command_run and command_start.

experiments/shift_hosts/supplement.py (new)

At module level it imports only the standard library, gate and handoff. claims and export are imported inside the functions, so the training environment can import it.

- **Constants:**
  - `ROOT_REL = "runs/shift_hosts/supplement_v4"`; `ROOT = gate.REPO / ROOT_REL`.
  - `LABEL = "S7 failed: trained and scored in full, shown, not judged"`.
  - `CAVEAT`, the version-4 post-hoc caveat.
  - `V3 = dict(fingerprint=<full 64>, approval_file="approval_fe9fb77b0851c75b.json", freeze_commit="bafd86f", approval_commit="6f9ce2b")`.
  - `CHANGEABLE` = {experiments/shift_hosts/stages.py, supplement.py (added), PRESPEC.md, INTERFACES.md, tests/test_stages.py, tests/test_claims.py}.
- **`scopes(entries)`** returns {quality: {learner: "learner"|"iw"}} by calling claims.s7_scopes for each quality. A learner scope outranks an IW scope.
- **`plan(scopes, learners, choice19)`** returns {q: {L: setups}}:
  - learner scope: ["alone", "standard", "iw", "room"], plus "sham_alone" if choice19[L];
  - IW scope: ["alone", "standard", "iw"].
- **`run_ids(plan, seeds)`** returns {q: [run ids]} in plan_units/unit_jobs order: sham, seed0..seed4 (learner order, then STAGE2_SETUPS), then room (seed, then learner).
- **`validate(block, learners, seeds, choice19)`** checks:
  - root == ROOT_REL;
  - kind == "s7-supplement";
  - the scopes recomputed from `block["source"]["s7_stopped_entries"]` equal `block["scopes"]`;
  - `run_ids(plan(...))` equals `block["runs"]`;
  - every source SHA-256 is present;
  - s7_all_seeds is a bool.
  - It returns the plan, or raises gate.ApprovalError.
- **`code_unchanged(manifest_now)`** loads the code_manifest from approval_fe9fb77b0851c75b.json and returns the list of differences outside CHANGEABLE, counting added or removed entries; supplement.py is the only file allowed to be added.
- **`check_sources(block)`** is described under preflight above.
- **`read_carried(path, sha256)`** runs handoff.read_json and requires the file's SHA-256 to equal the frozen value.
- **`reproduction(v4_root, v3_root, run_ids)`** (R1, read-only, never raises). For each run id with a record in both roots:
  - v3's sidecar must verify and its code.fingerprint must equal V3's;
  - compare record["hashes"] (all of initial, key_carry, params_10000 and final, runner.py:811 and 1020), and `refreshes` with timing keys removed;
  - compare each snapshot actor's `actor_tree_sha256` from its actor manifest (runner.py:997);
  - for stage 1 and shared seed-0 lanes, also compare the s7.json closed_loop max_points and mean_points;
  - result per run: equal / differs (fields) / missing.
- **`shown(q, runs, aux, plan_q, v3_q, r1, s7_results)`** builds the block below from the unfiltered supplement runs:
  - `label`, and the `caveat`.
  - `scopes`: v3's scope and entries from the carried qualities.<q>.s7_stopped.scopes, plus v4's own s7_stopped entries for that learner.
  - `setups`: export._setup_block(q, L, s, runs, aux) for each shown setup (learner scope: alone, standard, iw, room, sham_alone; IW scope: iw only), plus per-run provenance {run_id, record_sha256, trained_under: "version 4", v3_record_sha256 or null, r1}.
  - `comparisons`: claims.paired on none, mild and strong for J and points:
    - learner scope: standard-alone, iw-standard, iw-alone, room-alone, and sham_alone-alone (seed 0 only);
    - IW scope: iw-standard and iw-alone.
    - The summaries are cut to per_seed, n, missing, complete, mean and sd. se, t, lower, upper and p_one_sided are dropped.
  - `table`: rows {learner, setup, status: LABEL, badge: "not judged", holm_adjusted_p: null, strong and none per-seed values with mean and sd, v3_excluded: the matching v3 s7_stopped.excluded entries}.
  - `writeup`: markdown in the user's format (The idea / What we tested / What happened / Why). It carries "S7, with version 2's post-hoc limit of 0.05 points (PRESPEC.md section 19); trained and scored under version 4's post-hoc supplement (section 21); shown, never judged".
  - Never: no bar, mark, window, badge other than "not judged", Holm entry or p-value.
- **`build(permission, *, frozen=None, root=ROOT, carried_path, carried_sha256, v3_root=gate.REPO/"runs/shift_hosts", anchors=None)`:**
  - E.resolve; drv = E.read_driver(root, fz); col = E.collect(root, fz, C.QUALITIES, frozenset(drv["accepted_s5"])); E.check_score_anchors; integ = E.integrity(col, fz, drv, checked).
  - v3 = read_carried(...); results = deepcopy(v3).
  - results["caveats"].append(CAVEAT).
  - results["carried_from"] = {version: 3, path, sha256, fingerprint}.
  - qualities.<q>.s7_shown for each quality, with {"note": "no scope stopped by S7 in this quality"} where the plan is empty.
  - results["s7_supplement"] = {built (v4 fingerprint, permission, code SHA-256), plan, runs, excluded, unscored, halted_runs, integrity, s7_results (from state.json), reproduction, totals (card and processor seconds), ledger against caps, study-wide totals (v1+v2+v3 ledger + v4) against section 12's caps}.
  - Then `assert_judged_unchanged(results, v3)`: delete the added keys, pop the last caveat, and require json.dumps(sort_keys=True) equality with v3. Raise otherwise.
- **`main()`:** `python -m experiments.shift_hosts.supplement export [--out outputs/shift_hosts/results.json]`. It runs gate.require_run_approval, then build, then E.write. export.write already allows OUT (export.py:1543).

Files that do not change (byte-identical to approval_fe9fb77b0851c75b.json's manifest, enforced by test): claims.py, export.py (CAVEATS untouched, so v3's s7_stopped.rule keeps v3's text), runner.py, evaluate.py, policies.py, actors_f32.py, handoff.py, scoring_cache.py, band_readout.py, bca_core.py, iw.py, the four unifloral_*_bca.py files, unifloral_import.py, prepare.py, data_copy.py, guard.py, gate.py, budget.py, approve_hosts.py and every gate.EXTERNAL_CODE file.

experiments/shift_hosts/INTERFACES.md

- 734-767 (state and status): s7_stopped entries gain `version`; new `s7_results` and `supplement`; under v4 nothing is dropped and the dropped counts stay 0. The key name s7_stopped is kept on purpose.
- 844: version-4 row.
- 855-867: the merged results.json (carried_from, qualities.<q>.s7_shown, s7_supplement, caveats +1); every v3 key unchanged.
- New subsection: the supplement root layout.

Staging (outside the package)

- Copy scratchpad wf8/pkg to wf9/pkg.
- Copy run_stage0.sh (STAGING, line 42) and sync.sh (SRC) and point both at wf9. run_stage0.sh:144 calls $STAGING/sync.sh, so running anything against wf8 while v3 runs would rsync into WSL and halt v3 on S1.
- sync.sh strips CR only from *.py, so .md and .json files must be written with LF.

3) prespec.json AND PRESPEC.md

prespec.json

- `version` 4; `title` "Unifloral hosts on ThrusterChain: pre-specification version 4 (post hoc)"; `drafted` "2026-10-20"; `status` "frozen" at freeze.
- **`supersedes`:**
  - version 3, freeze_commit bafd86f, approval_commit 6f9ce2b, the full fingerprint, prespec_sha256, approved_at_utc, started_utc, finished_utc;
  - records: "kept in place under runs/shift_hosts/{stage1,stage2,driver}, never moved or written (chattr +i); version 4 reads them only for the R1 hash comparison; they never count as version-4 results";
  - results {path outputs/shift_hosts_v3/results.json, sha256, built_utc, built_under: "version 3's approval and code"};
  - previous: the current v2 block, unchanged.
- **`approval_summary`:** four new first lines, with the v2 and v3 lines kept after them:
  1. The rule. Every stage-2 run of every scope v3's S7 stopped is trained and scored in full, shown and never judged. A failed S7 check is recorded and drops nothing. S7's limits are unchanged. No judged badge, p-value, Holm step, mark or bar can move, because v3's results.json is carried byte for byte.
  2. The user's words verbatim and what they override.
  3. Runs per quality, the projection, and the cap booking (D2).
  4. What had been read before the freeze, with times.
- **New block `supplement`:**
  - kind "s7-supplement"; root "runs/shift_hosts/supplement_v4"; label (LABEL);
  - rule: "train and score in full every stage-2 run of every scope version 3's S7 stopped (learner scope: sham_alone for choice-19 learners, alone, standard and IW at seeds 0-4, room x5; IW scope: IW at seeds 0-4 plus that learner's alone and standard at seeds 0-4 as same-fingerprint comparators); stage 1 runs whole; nothing is judged";
  - source: {state_json {path, sha256}, s7_stopped_entries (verbatim), ledger_json {path, sha256}, results_v3 {path, sha256}, results_v3_committed {path, sha256}, approval_file};
  - scopes; runs {quality: [run ids]}; runs_total; already_trained_by_v3 (the R1 list);
  - s7_all_seeds (D3); s7_on_failure: "recorded in state.json s7_stopped (version 4); nothing dropped; its lanes go on";
  - checks: every v3 rule halts as approved except that S7 records; R1 is reported and never halts;
  - report: {path outputs/shift_hosts/results.json, judged_keys: "every key of results_v3.json, byte-equal by canonical JSON", shown_keys, shown_statistics: "per-seed values, 5-seed mean and sd, paired differences with their sd; no t, p-value, bound, badge, Holm entry, bar or kill rule"};
  - kill_rule: "none: the shown numbers decide nothing; any use toward an S7 fix is a new pre-registered test with its own kill rule, confirmed on fresh seeds 10-14";
  - code_unchanged: the CHANGEABLE list.
- **`stop_rules.S7`:**
  - `rule`, `open_loop_max_abs` 1e-7 and `closed_loop_max_points` 0.05 stay word for word.
  - `on`: add "version 4: also every supplement run's final actor at seeds 1-4 (alone, standard, IW) when supplement.s7_all_seeds".
  - `on_failure`: add "version 4: recorded in state.json s7_stopped with its scope; no run or lane is dropped; the scope stays not judged with p = 1 (version 3's judgment carried); a result that does not verify halts".
  - `on_failure_changed`: add "version 4, post hoc (PRESPEC.md section 21; P4)".
- `stop_rules.meaning`: the S7 clause.
- `claims.holm_membership` rule 2: append "; its runs enter no claim (version 4: trained, scored and shown, labeled, section 21)".
- `stage_order.stage2_supplement`; `driver_command.version4_start` and `version4_why`; `driver_command.not_used_on_real_data` adds `--supplement-plan` and `--s7-all-seeds`; `caps.covers` gets a v4 note if D2 is separate booking (the numbers stay the same).
- `budget_recost.version3` (actual per quality, from v3's ledger) and `budget_recost.version4` (projection central and high, booking).
- **`post_hoc_changes` P4,** with P3's fields:
  - id, version 4, plan "6.1 (S7) and 6.2", post_hoc true, what, code;
  - why: the failure table;
  - decided: the verbatim quote, the date, UTC time, "in chat", what had been shown, whether a question with options was asked and which option was marked recommended;
  - overrides: 9.1, 20.4 and 20.6's "never trains / never scored or used"; 20.6's "No run of a stopped scope enters any claim, mean or statistic", which becomes "enters no claim; its means are shown in section 21's supplement, labeled"; and the training half of 19 October's "the stop stands";
  - still_holds: "no later version relaxes S7 again" (limits unchanged); not judged with p = 1; 20 October's "no dont give up we need a result"; "nothing is retried", in the sense that no v4 outcome can reverse a v3 stop or change a judged key;
  - chosen: after the S7 failures were seen, with the rule drafted and committed before any return was read;
  - returns_read_before; claims_affected "none"; unchanged; expected.

PRESPEC.md

- **Line 1:** "... pre-specification version 4 (post hoc)".
- **New "Version 4 (post hoc)" paragraph above line 3.** It states:
  - drafted 20 October 2026, while v3 ran, after v3's S7 failures;
  - what changes: S7-failed scopes are trained and scored in full in a supplement, shown and never judged;
  - it was chosen after the failures were seen;
  - S7's limits are unchanged and no judged scope moves;
  - v3 at bafd86f / 6f9ce2b / fe9fb77b0851c75b;
  - the sections changed.
- **"How this document binds" (43-45):** add section 21.
- **One-line amendments in place, each marked "(version 4, section 21)":**
  - 8 (415-417): stage 2b;
  - 8 (426-428): results.json = v3's file carried verbatim plus a supplement built only from v4 records;
  - 8 (438): the frozen root runs/shift_hosts/supplement_v4;
  - 9.1 (457-460);
  - 9.2's S7 row (480): one appended sentence; the limits stay word for word;
  - 10 (526-529);
  - 12: cap booking per D2;
  - 17 (811-813);
  - 18 (822): "for version 4, section 21.9 replaces steps 1-3";
  - pointers at 20.4 (1283-1285) and 20.6 (1300-1302, 1313-1314);
  - "Words used here": add "Shown" and "Supplement".
- **New section 21, "Version 4: the S7 supplement (post hoc)":**
  - 21.1 What v3 ran: commits, approval time, start time, pid 30764, root, the outcomes of stages 1, 2a and 2b, spend per quality from v3's ledger.
  - 21.2 The S7 failures. A table of run id, actor step, bank, largest and mean per-start points, s7.json SHA-256 and scope:
    - stage1 rebrac-standard 0.13227 on big kicks, and rebrac-iw_ones with the same bits;
    - td3_bc-iw seed 0 at 100k, 0.282 on big kicks;
    - iql-alone seed 0 at 100k, 0.122 on strong current;
    - CQL passing at about 4e-5;
    - medium's failures, filled in;
    - the S7 checks v3 never ran (for example iql standard and iw at seed 0, if they were dropped);
    - the runs each scope left out, split into never trained / trained but unscored / scored but hidden (from results_v3 s7_stopped.excluded).
  - 21.3 Your decision:
    - the verbatim words "dont stop experiments, let them run in full so we can work from there towards solution", 20 October 2026, UTC time, in chat, what you had been shown, and whether a question with options was asked;
    - the main session's reading, stated as a reading: v3 untouched; every run S7 dropped is trained and scored; shown, never judged; a fix comes later;
    - why v3 itself could not stop dropping: any code edit halts it through S1;
    - the terminal approval is the decision that counts;
    - what it overrides and what still holds (as in P4).
  - 21.4 What had and had not been seen:
    - seen: S7 differences, driver logs and status, the budget;
    - not seen: any return, points level or mean of a learned policy (to be confirmed by the main session; scores files exist on disk, unread);
    - give the draft-commit time, the time results_v3.json was built, and the time it was first read, and say the rule did not change after;
    - it cannot favor a claim;
    - the shown numbers carry no kill rule.
  - 21.5 The supplement, exactly: rule, scopes, run list, R1 list, lanes, stage 1 rerun, S7 recorded, every other rule halts, the root, the ledger.
  - 21.6 The report: carried keys, shown keys, statistics shown and withheld. The caveat: shown scores are float64-adapter returns of actors whose float32 check failed (seed 0) or ran only under s7_all_seeds (seeds 1-4). They may differ from the learner's own float32 policy by more than 0.05 points per start, which is 0.635 return units on the strong mix (table at 983-991).
  - 21.7 Code changed, and code byte-identical.
  - 21.8 What to expect, stated now:
    - retrains repeat bit for bit and the known failures repeat;
    - iql standard and iw seed-0 S7 measured for the first time;
    - ReBRAC's medium-expert 100k actors likely fail;
    - R1 equal everywhere.
  - 21.9 The restart, in order.
  - 21.10 Budget: booking, projection, and the study-wide total against section 12.
  - 21.11 Where the post-hoc label appears: the title, header, section 21, the prespec fields and P4, the caveat, every s7_shown block and every supplement write-up. The study stays exploratory. Seeds 0-4 are seen, so any S7 fix must be confirmed on seeds 10-14.

4) TESTS (inside test_stages.py and test_claims.py)

Rewrites in test_stages.py:

- **test_s7_scopes (302-362) → test_s7_failure_drops_nothing.** Driver with a plan:
  - _verdict on an iql IW failure and exit 3 at the last step on td3_bc standard: no halt;
  - card_queue and lane_queue unchanged; cards_dropped and lanes_dropped both 0;
  - entries recorded once, with version 4;
  - plan_units and unit_jobs list exactly the plan's runs; the other quality's plan is unchanged;
  - exit 3 at an earlier step still halts.
- **test_s7_scope_branches (364-425):**
  - s7_seed0_checks records a failure once and halts on a run with a record but no s7.json, and on an s7.json that does not verify;
  - a viewer lane of a failed scope launches its next step;
  - exit 3 with an s7.json that says it passed halts;
  - a unit with no plan run is skipped with the v4 note.
- **test_s7_stopped_while_running (427-452):** a run whose scope failed while it trained gets its lanes; room_checks checks every room job.
- **test_simulated_s7_stops_its_scope_only (784-836) → test_supplement_dry_run.**
  - Command: `--dry-run --learners td3_bc iql --seeds 0 1 --supplement-plan <iql learner scope + td3_bc IW scope> --s7-all-seeds --inject stage1/...td3_bc-standard=s7fail stage2/...seed0/iql-alone=s7fail`.
  - Expect: exit 0 and no STOP.json; stage 1 whole (10 records); stage-2 records exactly equal supplement.run_ids(plan).
  - The s7_stopped entries carry version 4.
  - S7 lanes and s7_results exist for every seed-1 final actor.
  - No state check key matches "S7/*/seed[1-9]/*".
  - The log contains "S7 (version 4)".
  - A resume retrains nothing.
- **Unchanged:** test_failed_s7_never_passes_on_resume (264-300).
- **test_modes (504) and test_real_options_are_refused (532-546):**
  - a dry run refuses S.ROOT and paths inside it;
  - check_real_args refuses --root other than S.ROOT, --supplement-plan and --s7-all-seeds;
  - config_from's real root == S.ROOT;
  - frozen_config refuses a prespec whose supplement run list differs from plan(scopes).

New tests in test_claims.py:

- **CodeUnchangedTest:** supplement.code_unchanged(gate.code_manifest()) == [] on the real package; a one-byte change to export.py appears in the result. This also proves claims.py and export.py are byte-identical to v3.
- **SupplementPlanTest:**
  - counts: learner scope 21 for TD3+BC and IQL, 20 for CQL and ReBRAC; IW scope 15; a learner scope outranks an IW scope; per quality;
  - run_ids order equals a dry Driver's plan_units/unit_jobs labels;
  - scopes() agrees with C.s7_scopes;
  - validate refuses a wrong run list, a wrong root, an unknown learner or scope, or a missing source SHA-256.
- **SupplementBuildTest (made-up):**
  - carry a made-up "v3" file built by E.build on test_stages' plain dry run, and build over test_supplement_dry_run's root;
  - every v3 key is byte-equal; tampering with one judged key raises; a wrong carried SHA-256 or sidecar is refused;
  - s7_shown has no t, p_one_sided, lower, upper, bar, mark or badge other than "not judged", and every row carries LABEL;
  - integrity S7 reads "outcome", including seed-1 S7 failures covered by v4 entries;
  - S1 passes, and one v3-fingerprint record dropped into the supplement root fails S1.
- **ReproductionTest:** equal on identical records; differs on a changed hashes.final; ignores identity, code and file SHA-256 differences; never raises.
- **InvarianceTest:** for the existing S7 scenarios (test_learner_scope, test_iw_scope), C.judge and C.phone_table output is byte-identical with and without extra scored runs for the stopped scopes. This proves the judging code ignores shown runs.

Finally, the full stage-0 suite (wf9 copy of run_stage0.sh) on the frozen v4 bytes with the card idle, before any v4 approval exists.

5) RUN ORDER AFTER v3 FINISHES

Now, while v3 runs. Windows only and kept light: no JAX, no tests, nothing under /root/bca-hosts or wf8.

- Write the code, tests and docs in wf9/pkg and in a separate worktree on branch claude/shift-hosts-v4-draft from 6f9ce2b.
- Leave slots for: v3's scopes and run list, the SHA-256 values, times, spend and caps.
- Commit and push the draft now. Its timestamp proves the rule came before any return was read.
- Put D1-D4 to the user in chat.

T0: status.json says "finished", driver.log has "finished: ...", the lock is free, there is no STOP.json and the card is idle. This is expected early-to-mid afternoon UTC. If v3 halts instead, stop and report: the supplement covers S7 scopes only.

1. **T0 to T0+0:20.** With v3's bytes still in place:
   - `cd /root/bca-hosts && env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /opt/bca-shift-venv/bin/python -m experiments.shift_hosts.export --out runs/shift_hosts/results_v3.json`. It prints only the path and digest; do not read the file.
   - Copy it with a fresh sidecar to outputs/shift_hosts_v3/results.json and commit it in WSL (GIT_CONFIG_GLOBAL=/root/.bca-gitconfig).
   - Record build time and SHA-256 values: results_v3, driver/state.json, driver/ledger.json.
   - `chattr -R +i runs/shift_hosts/{stage1,stage2,driver}` and check with lsattr. If chattr is unavailable, rely on preflight's SHA-256 re-checks.
2. **T0+0:20 to 0:50.** Fill the slots from v3's final s7_stopped (state.json is not returns), the s7.json differences and the ledger.
   - On Windows, merge, not rebase, the results commit into the draft branch, so the pushed draft commit keeps its timestamp. Then fast-forward /root/bca-hosts.
   - Check that staging equals WSL byte for byte, with LF endings and no __pycache__.
3. **T0+0:50 to 1:50.** The stage-0 suite on v4's bytes (about 55 min, card idle and cool; S0).
4. **About T0+2:00.** Pre-commit check, then the freeze commit "Unifloral hosts: freeze pre-specification version 4 (fingerprint <first 16>)", pushed.
5. **The user approves at the terminal** (section 18 step 1, approve_hosts.py), then commits and pushes. No agent does this.
6. **Pre-start checks:** gate, S0, graphics driver 610.62 (S9), no other card user, preflight's source checks.
7. **The user starts it, as root:** `env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /opt/bca-venv/bin/python -m experiments.shift_hosts.stages start --stage0-card-hours 0 --stage0-processor-hours 0`. The root comes from the code and there is no --resume-after-stop. Keep the Ubuntu window open.
8. **After the last lane:** `python -m experiments.shift_hosts.supplement export` (scoring environment) writes outputs/shift_hosts/results.json. It refuses unless v3's keys are byte-equal. Commit the file.
9. **The fix step is a later version** with its own kill rule. Its input is s7_supplement.s7_results.

Compute: concurrency 2 with slowdown 2.13. Card-h ≈ 1.065 x solo wall; processor-h ≈ 2 x card-h plus lanes at budget.DEFAULT_LANE_SECONDS. Central is 1.0-1.23 x the model; high uses 5 x alone ms for BCA.

| Part | Runs | Card-h central | Card-h high | Processor-h central | Processor-h high |
|---|---|---|---|---|---|
| Stage 1 (v3 measured) | 20 | 0.24 | 0.24 | 0.33 | 0.33 |
| ReBRAC learner, medium-expert | 20 | 1.6-2.0 | 3.1 | | |
| IQL learner, medium-expert | 21 | 0.9-1.1 | 1.2 | | |
| TD3+BC IW scope incl. comparators, medium-expert | 15 | 0.8-1.0 | 1.4 | | |
| Medium-expert subtotal | 56 | 3.3-4.1 | 5.6 | 10.7-12.3 (incl. 3.8 lanes) | about 16 |
| Medium, expected (ReBRAC learner, PRESPEC 20.7) | 20 | 1.6-2.0 | 3.1 | 4.5-5.3 | about 7.7 |
| Expected total | 96 | 5.1-6.3 | about 9.0 | 15-17.6 | about 24 |

- **Wall time** at concurrency 2: about 1.1 x card-h, plus about 0.3 h of lane tail per quality. That is 6.3-7.6 h central and about 10.6 h high.
- **Before the start:** about 2 h plus the user's approval time.
- **s7_all_seeds** costs about 0.4 processor-h per scope (12 lanes x 120 s).
- **Worst case on medium** (all four learners stop, 82 runs): plus about 5.1 central / 9.0 high card-h and 12 / 20 processor-h. The high end passes medium's 16 processor-h cap, so S6 or S6b could stop medium between units.
- **Against section 12's numbers on the supplement's own ledger (D2):** medium-expert about 4.4 / 12.3 of 14 / 21 central; medium about 2.0 / 5.3 of 12 / 16. Study-wide totals (v1+v2 0.49 / 0.71, plus v3's actual ledger, plus v4) are disclosed.

6) RISKS AND THE USER'S OPEN DECISIONS

Risks:

- **Order.** If v4 bytes land in /root/bca-hosts before step 1, v3's approval no longer verifies and its results.json cannot be built without restoring v3's exact tree.
- **Size is unknown until v3 finishes.** Medium's S7 outcome sets the size: 56 to 138 runs.
- **Reading risk.** Shown numbers are float64-adapter scores of actors whose float32 check failed or was not run, and they invite selection after the fact. They are labeled, kept in a separate block, carry no bar, and need a kill rule before any use. The supplement and the user's page must keep them out of the 12-row table.
- **R1 or S9 drift.** If retrains differ from v3's records, the premise that "this is what v3 would have produced" fails. R1 reports it per run; S9 should halt first.
- **S5 on a BCA setup inside the supplement halts it** until it is traced (D4).
- **Seeds 1-4 S7 lanes run on TD3+BC's alone and standard comparators**, which are bit-identical to v3's judged runs. They can reveal tail failures in a judged scope at seeds 1-4. These would be reported as a caveat on TD3+BC's claims; no judged key changes, because S7's pre-registered coverage is seed 0 only.
- **Committed results go to the remote.** The results_v3 copy, once pushed, puts v3's results on the remote before anyone reads them.
- **chattr +i must be removed** (chattr -R -i) before any later archive or move of v3's folders.

Decisions for the user:

- **D1. Confirm the reading in chat before the freeze** (a question with options; recommended: this supplement). The reading: v3 runs to the end untouched; afterwards a v4 supplement trains and scores every scope S7 stopped; shown, never judged; the S7 fix is a separate later version.
- **D2. Budget booking.** Recommended: the supplement gets its own ledger under section 12's cap numbers per quality, and the study-wide total is disclosed. No code change, but it changes what the caps cover. The alternative is to count v1-v3's spend in the same caps, with a real risk of an S6 or S6b stop on medium's 16 processor-h.
- **D3. S7 lanes on the seed 1-4 final actors of every supplement run.** Recommended yes: about 0.4 processor-h per scope. This is the data for the fix: how the tails spread across seeds. Optional and not included unless asked: S7 lanes on the 50k-step actors as well (a small _s7_lane label change, processor time only).
- **D4. S5 on a BCA setup inside the supplement.** Recommended: keep halting until it is traced (--accept-s5 exists). The alternative, record and continue, is a further post-hoc stop-rule change that would go into P4.
