# Usable repair handoff

This handoff supersedes the initial README's preparation status. The repaired package is now usable directly for source verification, CPU regression checks and saved-data analysis. No manual copying is required. It is **not** a scientific launcher.

From `C:\Users\David\dev\BCA`, the verified Windows command is:

```powershell
.\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v3_guard_repair_v1/launch_repaired.py check
```

Replace `check` with:

| Command | Action |
|---|---|
| `test` | Run the selected CPU regressions in separate processes. Final result: **95 passed**, 81 + 14. |
| `analyse-known-v2` | Authenticate and reanalyse the one pinned completed v2 attempt; print JSON. No new correction is accepted for a different producer. |
| `spend PATH` | Read the saved attempt's physical-accounting records with the repaired unknown-time handling; print JSON. |
| `support BUNDLE.json` | Read a hashed fitting-only NPZ export and run retained-only support math; no dataset constructor, reset, dynamics replay, model or learner calls. |

There is no `execute`, training, calibration or collection command. Invalid inputs are refused. Results go to stdout; the entrypoint does not modify historical receipts. The historical training workers, their approvals and reservations are not redirected to this package. A future scientific runner must use this versioned package with a new approved contract, never an old reservation. Missing A4/A5/EXPLORATORY seals remain intentional barriers.

The command verifies `REPAIR_PACKAGE.json`, its source-snapshot lineage, all listed files, absence of unlisted Python files/importable package folders, and actual imported module locations. `check` resolves `units`, Stage 1/2 workers and the historical analyser to `knee_edge_ood_v3_guard_repair_v1`; retained support resolves to the sibling `knee_s10_candidate_guard_repair_v1`. Its repository/payload paths preserve the existing directory depth.

## Concurrent formal runner — outstanding integration boundary

During final review, commit **`ee704bf0c7a99c74d01efbae9120e248a579634d`** added `knee_s10_candidate_v1/formal_g2/`. Its `g2_formal.py:219` still calls L1.5 on **all fitting rows**, despite computing `cutoff` at line 215. The new runner is therefore **not repaired by this handoff**, and its “ready to run” commit title does not establish retained-state support.

At inspection, neither `G2_FORMAL_APPROVAL.json` nor `formal_g2_run/` existed. This is absence of saved approval/start artifacts at that observation, not a global process check. The new formal source, rule and manifest were left unchanged. Integrating a changed reference population into that newly sealed 96-anchor study requires an explicitly new protocol/package; it must not be silently substituted into its seal or conflated with the proposed 48-anchor development support check. This is the exact remaining adoption boundary for that scientific runner. The verified analysis/verification CLI above remains usable now.

## Why the repair was isolated

**No active process was verified.** The saved dose-screen receipt is COMPLETE, with 12 branches and three evaluations, ending October 31 at **12:58:20 p.m. EST**. Filesystem records are not a global process-liveness check, and the existing WSL denial was respected.

The concrete isolation reasons were preservation of hash-pinned historical sources and concurrent work. The initial source snapshot was HEAD `af30c346d35b2fd13a3c0f8b3b9d1ae49962f582`; concurrent HEAD `d703134637d607e57862256cd70268311b7429e6` changes the knee-bed conclusion note only. All 52 original snapshotted source/fixture files remain byte-identical. No original source, result, approval or live output was overwritten. There is no blocker to the new analysis/verification entrypoint. Scientific execution is outside this repair authorization.

`HANDOFF_TEST_RECEIPT_FINAL.json` and `HANDOFF_REGRESSION_FINAL.log` record the actual 95-test command and successful results. Startup tests run the CLI from an unrelated working directory, exercise the saved-v2 and accounting paths, and process synthetic saved support arrays. The shared worker finalizer is exercised through both worker entrypoints with mocked jobs. Review additionally caught and fixed inconsistent time coordinates, snapshot verification and import shadowing. These checks do not claim GPU/runtime training validation.

## Saved support input format

`support` takes a JSON envelope with schema `retained_support_fitting_export_v1`, `data_kind` (`development` or `synthetic_fixture`), `world` (`stall3` or `nostall`), `arrays_file` (adjacent NPZ filename) and `arrays_sha256`.

The NPZ has exactly:

- `obs`: n × 14 observations, with consistent remaining-time coordinate;
- `actions`: n × 4 finite executed float32 actions;
- `episode_ids`, `times`, `deep_episode`, `row_roles`: n aligned entries; roles must all be fitting/code 0;
- `start_obs`: 128 × 14, all at time 0;
- `anchor_obs`: 48 × 14, and `anchor_times`: 16 each at 0, 25 and 50.

The retention mask is recomputed using the existing CUTOFF predicate, not accepted from the input. The output reports hashes, denominators and the original support bars. Byte checks do not independently authenticate a producer's collection or role declarations; the output explicitly states that limitation and never grants formal admission. A real collector must bind exports to its raw datasets and sealed source/role manifest. Fixture results are labeled synthetic.

## Support-only proposal, fixed before new outcomes

`PROPOSED_SUPPORT_ONLY_CHECK_v2.json` supersedes the original proposal. Its **reset count corrects a bookkeeping mistake**; the transition count and decision thresholds are unchanged. Nothing has run under this proposal.

Existing source identities, rules and summary evidence can be reused. Existing summaries establish the reported 65/70/60 retained reference counts, but lack the raw states/anchors required to recompute distances. The old narrow-recorder knee datasets and trained checkpoints are not substitutes for the widened recorder's missing raw arrays. If an authenticated original export is found, its pure reanalysis costs zero transitions; the known 60-row group still cannot satisfy a 64-row requirement. No global filesystem absence is claimed.

| Item | Fixed proposed specification |
|---|---|
| Data | Existing widened recorder and physics; 2,048 episodes per world, paired stall/no-stall worlds |
| Collection entropy | **424264** |
| Query stream | **707070708** |
| Split per world | **1,231 fitting / 409 tuning / 204 selection / 204 final-calibration episodes** under existing split stream 1802; whole-episode assignment |
| Decision inputs | Fitting rows only. Other roles are generated/validated for dataset identity but are excluded from support normalization, reference groups and decision math. No calibration is performed. |
| Query banks | 128 `stage1_e2_bank` starts, item IDs 0–127; 48 `stage1_e2_anchors`, IDs 0–47, 16 each at t=0,25,50; exact states/time/prefixes saved |
| CUTOFF/reference metric | CUTOFF-retained observations only, including normalization; 64th-nearest-row standardized Euclidean distance; reference queries exclude their own episode; same-time deep-reference 0.99 quantile |
| Required evidence | At least 64 reference rows in each time group in both worlds; report distinct episode counts too |
| Pass bars | At least **95%** of starts and **90%** of anchors in each world |
| Maximum work | **820,400 transitions; 8,368 resets; 600 seconds globally** |
| Other work | Zero learned-model calls, actor/critic/auxiliary updates, calibration and Q-pi continuation evaluations |

Why 820,400: two worlds × 2,048 episodes × 100 steps = **409,600 collection transitions**. Each world's constructor performs one full dynamics-validation replay, adding **409,600**. Forty-eight anchor prefixes add 16 × (0+25+50) = **1,200**. Start resets add no transitions. Lineage validation resets the dataset episodes a second time, so resets are 2 worlds × 2 passes × 2,048 + 48 anchors + 128 starts = **8,368**.

This is a small bounded proposal retaining the existing generator and role-split convention, not a claim of the mathematical minimum possible cost. A custom fitting-only collector could have a different budget but would change that convention. Save the already validated datasets directly; saving does not replay dynamics. **Do not call `dataset_knee.load()` or reconstruct `KneeDataset` afterward** within this cap, because that would add another 409,600 transitions. Read the fitting export with the new pure CLI instead.

The sample size addresses inadequate reference counts, not BCA performance: the recorder implies P(deep and retained)=0.0759. For 1,231 fitting episodes the expected count is 93.43 per time group; the nominal probability of fewer than 64 is 0.0003503, with conservative six-group union bound 0.002102. These calculations do not predict whether distances pass.

Scoped identity checks found 2,224 distinct proposed reset seeds (2,048 dataset + 176 query), no internal collision and no overlap with 6,080 known prior reset seeds in the reviewed knee families. The raw values also do not collide with sealed named seeds. This is a scoped known-family check, not an exhaustive repository-history guarantee; record the final collision receipt when sealing a scientific executable. The governing `seeds.json` SHA256 is `0dd0b38e3b6d48148c2bbba73e1a2f1b95e3b542807343cfefdf9c5d4765e471`.

Failure handling is fixed: malformed/unaligned data, nonfinite values, fewer than 64 eligible reference rows, a cap hit or incomplete lineage means **INCONCLUSIVE** and stop. Adequate evidence below either share bar means **FAIL** and stop. No sample top-up, entropy replacement, region/radius change, threshold relaxation or automatic retry follows a failure. A pass is retained-state typicality evidence only; it does not prove unchanged state distributions, empirical joint action support, complete G2 validity or BCA efficacy.

Persist both complete raw dataset NPZs, exact role/episode identities, recorder/dynamics/source hashes, fitting-only exports, CUTOFF masks/row IDs, query states/time/prefixes, reference group counts and thresholds, and supervisor receipts with attempted/completed transitions and resets. This prevents the previous loss of recomputable input arrays.

The design was revised after consumed development evidence. Fresh streams make the next samples independent **development evidence**, not untouched confirmation. Scientific adoption, its immutable contract and a new bounded reservation remain pending. No simulation, training, calibration, commit or push occurred in this handoff.
