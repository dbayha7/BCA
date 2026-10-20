# Version 9 (the deciding test): build notes (20 October 2026)

Status: **DRAFT, not frozen, awaiting the owner.** Nothing was frozen, approved, imported or faked. Nothing was trained
and no official phase ran. The draft package is `outputs/shift/beyond-the-data/near80_v9/` on branch
`claude/near80-v9`, built from `origin/claude/main-bca-connection-x4o46s` at 52914e4. It is committed and not pushed.

Paths below are relative to `outputs/shift/beyond-the-data/near80_v9/` (written `V9/`). Version 8's folder
`near80/` is unchanged; this file is the only file added to it.

---

## 1. What was built, item by item (file:line)

### 1.0 Identity, layout, seeds

The package is a copy of version 8's `near80/code/` with every name moved to version 9:
- the folder `near80_v9`;
- the outputs `runs/shift_explore/near80_v9/`;
- the kinds `near80v9-*`;
- `approval_near80v9_<16>.json` and `import_near80v9_<16>.json`;
- the run-files branch `claude/near80-v9-run-files`;
- the scripts `run_near80_v9.sh`, `push_runfiles_near80_v9.sh` and `approve_local_near80_v9.py`.

Version 6's band code (`v6bca.py`) and the near-80 facts script are not carried. `facts_v9.py` replaces the facts script.

| change | where |
|---|---|
| The open-value marker `TODO = "TODO(DECISION.md)"`; `todo_paths()` lists every place it is still left (a value `None` counts as filled) | `V9/code/near80_constants.py:20`, `:351` |
| Identity, version 8 paths (`V8_DIR`, `V8_R8`, `V8_RESULTS`), the owner's answer left open | `near80_constants.py:25-55` |
| The seed registry: every block any study of the main generator named, checked against every branch, `origin/claude/placement-test-design`'s `placement_test/seed_registry.json` among them | `near80_constants.py:100-118` |
| The fresh pool **35-49** (DECISION.md 2.3, process.md 2.2); `N_FRESH` open (5, 10 or 15); `CONFIRMATORY` = the first `N_FRESH`. The whole pool stands in while N_FRESH is open, and the freeze refuses | `near80_constants.py:122-128` |
| Reserved and refused in every mode: 5-9, 10-14 (spent), 15-34, 100-119, 900-909 and the unused rest of 35-49 | `near80_constants.py:119-121`, `:320` (`refusal_of`) |
| The facts script reads seeds 0-4 only (10-14 with `--spent`) and never the pool | `near80lib.py:71` (`check_seeds`) |
| Main setting only, no band, no random rows, no version 6 band, no readouts | `near80_constants.py:177` (`ARMS`), `PHASE_KINDS`, `LAYOUT` (`:276`); `phases.py:146-200` |
| Seeds 0-4 carry no phase of the official plan. Seed 0 is the plumbing's smoke; the official plan is the confirmatory group only | `phases.py:154` (`table_keys`), `:477` (`admit`) |
| Version 7 gating replaced by version 8 gating. Version 8 has finished: its `RESULTS.json` is in place | `near80_runner.py:84`, `:133`, `:907`, `:937`; `run_near80_v9.sh` (`version8-ended`) |
| Version 8's amendment 1, root cause fixed: the wrapper writes its run sizes under `run_sizes`, never over a data record's `sizes` | `phase_main.py:290` |
| Version 8's results pinned for the approval chain | `approval_check.py:67` |

### 1.1 The refusal fix, part 1: pooled groups (`refusal.md` section 8, option 1)

| change | where |
|---|---|
| H = every calibration-split and evaluation-split row outside the ball (`gap.contains` on the stored command) | `near80lib.py:158` (`heldout_rows`) |
| tail = the survivors, then the H rows past t_lo; rest = the H rows at or below t_lo | `near80lib.py:175` (`pooled_groups`) |
| Version 8's bank-only groups, kept for the reproduction and the reported comparison | `near80lib.py:184` |
| By source (survivors, bank, other calibration, evaluation), distinct episodes, and one row per episode (reported) | `near80lib.py:209`, `:218` |
| The group checks: no ball row; no training row but the survivors; the survivors are exactly the kept rows past t_lo; every other row held out; disjoint; equal to H plus the survivors; each row on its side of t_lo | `near80lib.py:232` |
| The switch scores the two pooled groups **once** with the second critic: the survivors from the prepared training transitions, H through the gate's `_transition`. One certificate entry, so one pair of groups, feeds every switch arm | `learner_phases.py:204` (`pooled_rows`), `:213` (`switch_scores`), `:230` (`certificate_entry`) |
| The switch refuses unless its held-out rows hash to the data record's `heldout_rows_sha256` and its group sizes equal the outcome-free count. It records sources, checks, the count, `bca_refused`, and the reported-only bank-only and one-per-episode certificates | `learner_phases.py:265` (`switch`) |
| The second critic still trains on the kept rows minus the survivors. No H row is in any training file | `learner_phases.py:107`, `:119` |
| No scored number reads the groups: scores are rollouts from `thruster_chain.bank_starts` (`evaluate_phase.actor_values`). The reproduction records the pooled certificate on seed 0 as reported only | `evaluate_phase.py:145`, `:321` |

### 1.2 The refusal fix, part 2: the outcome-free count before any training

| change | where |
|---|---|
| The count of the pooled groups from commands alone, beside BCA's certifiable bar (36, `calibration.reference.certifiable`) | `near80lib.py:262`, `:269` |
| The data phase writes `group_count` and `heldout_rows_sha256` into every data record before the training file. Every training phase needs the data phase | `settingdata.py:351` (`group_count`), `:366` (`data_main`) |
| Reported in every look as the first sentence, never a stop | `pooled.py:810` (`count_numbers`); `PREREGISTRATION.md` 14.1 |
| The frozen no-training checks on seeds 0-4: the gate's checks, the gap rule, the group checks, and the bank-only groups reproducing step 1's recorded sizes | `near80lib.py:315`; `facts_v9.py:67`, `:123`; `freeze.py:55-56` |

### 1.3 The refusal fix, part 3: a refusal fails only BCA's arms

| change | where |
|---|---|
| Step 2's arm-table rule restated. A switched-on arm with a non-finite (or negative) height is marked `refused` instead of raising. When step 2's `arm_table` does not raise, the restated table must equal it | `policy_phase.py:116` (`restated_table`), `:160`, `:171` (`arm_row_from`) |
| A refused arm keeps its place in step 2's program as a height-0 pad (`refused_pad_<arm>`), checked equal to no caution and never saved or scored. Every other arm keeps its bits | `policy_phase.py:203-230` |
| Refused: `obca_k1` and `obca_k2` where BCA refuses; `obca_forced` in both worlds when the stalling world refuses. Without a strength to match, the matched rivals (`standin`, `even_pull`, `push_less`, `random_1`, `random_2`) are unmatched. Every other arm is trained and scored | `near80_constants.py:175-176`; `policy_phase.py:291` (`train_world`), `:465` (the `refusal` record) |
| The looks: a refused seed fails every mark of C1 and C2, with the refusal's own check; never a whole-seed stop | `pooled.py:410-420`, `:444` (`load_results`) |

### 1.4 Arms

The arm list is version 8's claim-4 arms plus the two trained plain twins (`near80_constants.py:169`, `:177`).

| twin | rule | where |
|---|---|---|
| `meandiff_k1` | switch on when mean(tail) - mean(rest) > 0.10; the ramp at that height, k = 1; a program of its own | `policy_phase.py:195` |
| `meandiff_limit` | the same switch turning on the hand-set range limit; switched off it is the no-caution actor. Trained as a hinge program when on, and reported against `range_hinge` | `policy_phase.py:196`, `:364` |

### 1.5 The claims block

The block is `near80_constants.py:209-235` (`CLAIMS`), with `DECISION_RULES`.

| part | what it holds | where |
|---|---|---|
| C1 | version 8's claim 4 marks, unchanged (net lead 10, keeps -3, even pull 15), through `step2lib.go_no_go` | `pooled.py:617` |
| C2 | BCA's switched ramp against each comparator (`conformal_k1`, `meandiff_k1`, `meandiff_limit`), paired by seed on the net across worlds. Margins, statistic, alpha, seeds-ahead rule, multiplicity and noise floor are open (TODO). Statistics: `mean`, `paired_t` (one-sided), `interval` (two-sided t-interval: shown, excluded or unresolved). Multiplicity: `all`, `holm`, `bonferroni`. While any value is open, C2 is computed but "open", and no sentence is written | `near80_constants.py:198-226`; `pooled.py:665`, `:678`, `:696` |
| Decision rule | `C2` or `C1_and_C2`, open | `pooled.py:773` |
| Validity (version 8's rule carried to the main setting) | harm at least 30, past-80 share at least 25%, over the seeds that did not fail. When it fails, neither claim is read and the decision is "no conclusion" | `near80_constants.py:230`; `pooled.py:104`, `:593` |

Other parts of the looks:
- the sentences of section 14, the bracket table, `[comparisons]` and `[reason]`: `pooled.py:940-1010`, `claims_text.py:33`;
- the official look refuses while any value is open: `pooled.py:1153`.

### 1.6 The process files

| file | change | where |
|---|---|---|
| `freeze.py` | Refuses while any marker is left: `todo_paths()` plus the marker text in `near80_constants.py`, `prereg_draft.json`, `PREREGISTRATION.md` and `README.md`. Version 9's title; the facts are `facts_v9.py` | `freeze.py:52-57`, `:135` (`todo_problems`), `:342`, `:397` |
| `near80_prereg.py` | Version 9's content keys; the claims block copied into `content['claims']['block']`; the summary shows C1, C2 and the decision rule | `near80_prereg.py:27`, `:71`, `:138` |
| `approve_local_near80_v9.py` | Its summary reads version 9's prereg: the confirmatory seeds and forecast from the constants | `approve_local_near80_v9.py:431` |
| `PREREGISTRATION.md`, `README.md`, `prereg_draft.json` | New, DRAFT; section 14 matches `claims_text._ANCHORS` | the files themselves |
| `stubs.py` | Stub seed 36 has 10 survivors, so BCA refuses there | `stubs.py:53-55` |

---

## 2. Tests and results

The five test files ran, each alone, from /tmp, with the phases' environment (as `freeze.py` step 6 runs them). They
ran inside a private mount namespace in which this worktree is `/home/user/bca`; the real file system was not changed.

| file | tests | result | time |
|---|---|---|---|
| `test_near80.py` (rewritten for version 9's data code) | 34 | OK, 1 skipped (needs `prereg.json`) | 2 s |
| `test_approval.py` (adapted; the freeze tests run on a finalized copy) | 106 | OK | 21 s |
| `test_runner.py` (adapted; the end-to-end stub includes a refusing seed) | 83 | OK | 202 s |
| `test_pooled.py` (rewritten for C1, C2, validity and the refusal) | 45 | OK | 8 s |
| `test_v9.py` (new) | 31 | OK | 3 s |

What `test_v9.py` covers:
- **Seed reservation:** the pool is 35-49 and named nowhere else; the confirmatory seeds are the first N_FRESH; every
  other block is refused; the official plan reads only confirmatory seeds; the facts script never reads the pool.
- **Pooled groups:**
  - exact row sets, checked against a brute-force rebuild;
  - no training row but the survivors;
  - no group row in any training file;
  - no scored test reads a group row: the scoring functions read no split or bank rows and start from `TC.bank_starts`;
  - the same two groups for every switch arm: one entry, from one (tail, rest) pair, feeds BCA, split conformal, the
    rank test and the mean difference.
- **The outcome-free count:**
  - every training phase waits for the data phase;
  - the count does not change when rewards and next states are scrambled;
  - the switch must find the counted sizes;
  - a small count is reported, never a stop.
- **The refusal rule:**
  - step 2's arm table raises where the restated table marks;
  - refused arms and BCA's refusal are recorded, and the twins are kept;
  - without a refusal the table equals step 2's;
  - a refused place is padded in step 2's program;
  - only BCA's arms fail;
  - the stub's refusing seed lies in every N_FRESH choice.
- **The new arms:** the arm list, the 0.10 mean-difference switch, the programs.
- **The claims block:** C1 equals version 8's claim-4 marks; C2's shape; `todo_paths` lists exactly the open values;
  the draft holds the same open values.
- **The freeze refuses while a value is open:**
  - it names every open value;
  - `write` refuses and writes nothing;
  - a filled folder has no open-value problem;
  - `check` lists the open values.

`test_pooled.py` adds:
- C2's arithmetic against a hand computation: the mean, the one-sided paired t p-values, Bonferroni and Holm levels,
  the t-interval and its three outcomes;
- the noise floor, the decision rule, the refusal, the validity rule;
- word-for-word sentences.

---

## 3. The plumbing dry run (stub mode; nothing trained)

`near80_runner.py stub` ran on synthetic fixtures from `approval_check.write_test_fixture`. Every learner was a
stand-in that takes no gradient step, and the guards counted zero entries.

| run | result | phases | wall |
|---|---|---|---|
| seed 0 (the plumbing group, every phase kind) | exit 0 | 19 done | 51 s |
| fresh seeds 35 and 36 (36 is the refusing seed) | exit 0 | 36 done | 70 s |

On seed 36 the record shows the refusal path:
- **The data record:** pooled tail 10, below 36.
- **The switch:** on, height infinite, `bca_refused`.
- **The policy lock:**
  - refused `obca_k1`, `obca_k2` and `obca_forced`;
  - the matched rivals unmatched;
  - the program `[w0_check] [refused_pad_obca_k1, refused_pad_obca_k2, refused_pad_obca_forced, fixed_gated,
    conformal_k1, meangap_k1] [meandiff_k1]`;
  - both twins trained.
- **The evaluation:** it scored `none`, `clone`, `full_none`, `w0_check`, `fixed_gated`, `conformal_k1`,
  `meangap_k1`, `meandiff_k1`, `meandiff_limit`, `perfect_band` and `range_hinge`.
- **The confirmatory look:**
  - the count ("held 10 to 60 rows over 4 seed-worlds; 2 were below the 36 rows");
  - "BCA refused on 1 of 2 seeds (seed 36)";
  - C1 failed by seed 36's refusal check and by seed 35's matching (the stub's stand-in is not run);
  - C2 open.

`freeze.py check` on the real folder refuses as it should. It lists 64 problems: the draft status lines and
docstrings, and every open value in the constants, the draft JSON and the documents.

Not run here: `facts_v9.py` on seeds 0-4. It needs the gate's verified logs under `runs/shift/`, which are on the
desktop only. `freeze.py write` runs it with the frozen code.

Environment used for the tests:
- `runs/shift_explore/room/` and `runs/shift_explore/stage2/prereg.json` (+ sidecar) were extracted from
  `origin/claude/stage2-run-files` into a scratch copy mounted read-only;
- `runs/shift/preregistration.json` the same way.

The policy code and the freeze's pins read these. The repository and `/home/user/bca/runs/` were not touched.

---

## 4. Desktop time (from version 8's ledger, `near80/results-20261020T115246Z/files/ledger/`)

The basis, per fresh seed with both worlds' actors trained (seeds 11 and 14):
- 1.51-1.54 CPU-hours in all;
- stalling-world policy 41-42 minutes, no-stall policy 9-10 minutes;
- the three critics about 5.7 minutes each;
- reference actors 3.4 and 2.2 minutes;
- version 9's two plain-twin programs add about 1.6 minutes per world.

The forecast is 1.6 CPU-hours per fresh seed. The plumbing is about 1.2-2.5 CPU-hours (version 8's took 1.23). Under
version 7's memory rule at most 2 policy phases run at once, so the policy phases pace the run at about 26 minutes per
seed, plus about 35 minutes of start and tail.

| fresh seeds | CPU-hours (run + plumbing) | wall time of the run | plus the plumbing |
|---|---|---|---|
| 5 (35-39) | about 8 + 1.2-2.5 | about 2.8-3 h | about 1-1.5 h |
| 10 (35-44) | about 16 + 1.2-2.5 | about 4.9-5.4 h | about 1-1.5 h |
| 15 (35-49) | about 24 + 1.2-2.5 | about 7.1-8 h | about 1-1.5 h |

The budget cap is 2 x (2.5 + 1.6 N): 26, 42 or 58 CPU-hours.

---

## 5. What DECISION.md must fill before the freeze

`freeze.py` refuses while any of these holds the marker. `K.todo_paths()` lists them, and the documents' marker
lines are listed too.

| placeholder | where | DECISION.md's proposal |
|---|---|---|
| `N_FRESH` | `near80_constants.py:124`; `PREREGISTRATION.md` sections 0, 10, 16; `prereg_draft.json` `seeds.n_fresh` | 15 (35-49) |
| `CLAIMS.C2.margins` (`conformal_k1`, `meandiff_k1`, `meandiff_limit`) | `near80_constants.py:217`; PREREGISTRATION 9.2 | bar B = 5 points (for the contrasts it keeps) |
| `CLAIMS.C2.statistic` | `:218` | `interval` (two-sided 95% t-interval; shown, excluded, unresolved) |
| `CLAIMS.C2.alpha` | `:219` | 0.05 per contrast (with `multiplicity` `all`), or 0.10 with `bonferroni` over two contrasts |
| `CLAIMS.C2.min_positive_seeds` | `:220` | none (`None`) |
| `CLAIMS.C2.multiplicity` | `:221` | Bonferroni over the contrasts, at 2.5% one-sided each |
| `CLAIMS.C2.noise_floor_max` | `:222` | none (`None`) |
| `CLAIMS.decision` | `:227`; PREREGISTRATION 9.4 | the R0-R7 table of DECISION.md section 3. The nearest built rule is `C1_and_C2`, read with validity |
| `OWNER_ANSWER.text`, `.utc`, `.where` | `near80_constants.py:52` | the owner's own words, time and place |
| choice 4, lanes | PREREGISTRATION 16 | 10 |

After the fill, the session also:
- turns the documents' status lines to FINAL;
- removes "this draft";
- removes "DRAFT" from the first docstring lines;
- reruns the five test files and `freeze.py check`.

---

## 6. Where DECISION.md's design differs from this build

These were not built, and are left for the owner to choose.

1. **Two C2 contrasts, not three.** DECISION.md keeps C2a (against `conformal_k1`) and C2b (against `meandiff_limit`).
   `meandiff_k1` is reported only.
   - To adopt: set `CLAIMS.C2.comparators = ("conformal_k1", "meandiff_limit")`. The code and section 14 read the
     list.
2. **Missing data makes a contrast "unresolved".** DECISION.md says so. This build fails C2 with its "If a seed
   failed" sentence instead. Both refuse to credit BCA; only the wording differs.
3. **The top-up.**
   - DECISION.md: if a pooled group is below 36 rows, add blocks of 100 extra episodes, up to 10, read from commands
     only.
   - This build: no top-up. The count is reported, and a refusal fails BCA's arms only.
4. **The rest group.**
   - DECISION.md keeps it as version 8's (the kept bank rows at or below t_lo).
   - This build pools it as the build specification says (every held-out row at or below t_lo).
   - Either is a one-line change in `near80lib.pooled_groups`; no test reads the rest group's source except
     `group_sources`.
5. **`plain_limit` computed, not trained.**
   - DECISION.md computes it from the trained `range_hinge` and `none` actors.
   - This build trains `meandiff_limit` as its own hinge program and records its parameter difference from
     `range_hinge` (`same_as_range_hinge`). It deploys the same rule, so the computed form is a subset.
6. **Split conformal at the textbook rank.**
   - DECISION.md names the textbook rank.
   - This build keeps step 2's `conformal_k1` (q90 difference > 0.10), unchanged.
7. **Dose arms.** DECISION.md's reported-only arms `plain_h25` and `plain_h50` (fixed heights 2.5 and 5.0) are not
   built.
8. **The preview** (`refusal.md` Step P, process.md 4.5: scoring pooled rows with the saved second critics of seeds 0-4
   and 10-14) is not built. Nothing here reads a saved learner.
9. **Validity** is built as version 8's rule in the main setting (harm 30, past-80 share 25%), matching DECISION.md
   2.5.
10. **File names.**
    - process.md suggests `approval_v9_<16>.json`.
    - This build keeps the package pattern: `approval_near80v9_<16>.json`.

Small things seen in the stub look:
- the plumbing look writes "On seeds 0" for one seed;
- the "for scale" nets read "not available" when a seed refused, since BCA's own numbers are missing there.
