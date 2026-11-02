# RS1: state support against the retained rows, candidate knee_s10_v2 (rule written before any number)

Written 2026-11-01, after the formal G2 of candidate knee_s10_v2 read PASS (attempt 1, readout
`../formal_g2/G2_FORMAL_READOUT_20261101.md`) and before any RS1 number existed.

## What RS1 asks, and why

D3 of the formal G2 measured state support against **all** fitting rows, which is its sealed rule
(`../formal_g2/G2_FORMAL_RULE_v2.md`, D3). CUTOFF's critic trains on the **retained** rows only (every fitting row
outside W), so a concurrent note (`../../knee_s10_candidate_guard_repair_v1/RETAINED_SUPPORT_AMENDMENT.md`) argued the
reference population should be the retained rows. RS1 asks D3's question against that population.

RS1 is a precondition for the native package of OOD definition v4 r4 and is recorded in the ledger. It is **not** a
re-run of G2 and it changes no G2 label.

**Expected by construction.** Under the widened recorder (`../../knee_s10_candidate_v1/recorder_wide.py`) the
command a0 depends only on the state's load band and the sealed uniforms, never on the rest of the state. So within
a load band, whether a row is retained is a coin flip independent of its state (retention 0.506 at high load, 0.800
at low load), the retained rows have the same per-band state distribution as all fitting rows at about half the
density at high load, and RS1 should agree with D3 up to noise. RS1 verifies that empirically instead of assuming it.

## Data

The two datasets of the formal G2's attempt 1, `../formal_g2/formal_g2_run/attempt_1/` (`knee_s10_v2_stall3.npz`,
`knee_s10_v2_nostall.npz`, git-ignored), loaded with `dataset_knee.load` and required to equal the digests in that
folder's `collection.json`:

- stall3 `c2a6d61665a539ad8740d191424f90aeea9973566f3c6325226d928161fee533`
- no-stall `e3549869af13b367cd726d008935b3eb466e38c005d1d5b7856faae3397e572a`

Before loading, the dataset validator's sealed entropy is bound to this candidate's sealed collection entropy
(`g2_seeds_v2.json`), exactly as the formal driver did, because the validator otherwise compares with
knee_edge_ood_v3's sealed entropy.

## Definition

Per world:

1. **Rows.** The fitting rows (`masks.fitting_rows`). The retained mask is `masks.arm_masks(...)["cutoff"]` with the
   sealed `random_control` seed of `g2_seeds_v2.json`, the same call D1 made. A row is deep-load when its episode's
   reset component is 1, as D3 used.
2. **Queries.** The 128 start states and the 96 gated anchors (t in {0, 25, 50}), rebuilt from the sealed v2 seeds
   exactly as the formal driver built them (`g2_formal_v2.start_rows`, `g2_formal_v2.build_anchor`; the anchor prefix
   is identical in both worlds, P1).
3. **Distance.** The 14 observation coordinates are standardized by the **retained** rows' mean and SD (an SD below
   1e-12 becomes 1). The distance of a query is its 64th-nearest-neighbor Euclidean distance to the retained rows
   (`step04_math.knn_distance`, the sealed function). A group row excludes its own episode; a bank query has no
   episode.
4. **Threshold at time t in {0, 25, 50}.** The group is the retained rows that are deep-load and at exactly time t.
   The threshold is the 0.99 quantile (method "higher") of the group's leave-one-episode-out distances to the retained
   rows. The group's row count and distinct episode count are reported.
5. **Floor.** A group must hold at least **32 rows**; below that the world's label is INCONCLUSIVE. The sealed D3 used
   a floor of 64 on groups of about 126 rows. Retention halves the deep group (about 126 x 0.506 = 64 rows expected per
   time), so a 64-row floor would make RS1 undecidable by construction at deep load, which is not information about
   support. The 32-row floor keeps the 0.99 "higher" quantile meaningful (it is the group's maximum for groups under
   101 rows, as it already was for the sealed check's groups of 126 rows only at the second-highest value).
6. **Shares.** The share of the 128 starts inside the t = 0 threshold, and the share of the 96 anchors inside their own
   time's threshold.
7. **Bars, unchanged from D3.** Starts >= 0.95; anchors >= 0.90 (87 of 96). Per world.

## Reported, never deciding

- (a) A **secondary** threshold per time from the widened group, retained and deep-load with |time - t| <= 2 (the
  remaining-time coordinate differs by at most 0.02 within it), with its shares. It has about five times the rows and
  shows what the primary's small groups cost.
- (b) The **instrument check**: the all-fitting-rows D3 numbers recomputed with the sealed `step04_math.l15` on the same
  queries. They must equal the formal report's counts (`../formal_g2/formal_g2_run/attempt_1/report.json`,
  `D.<world>.D3`); if they differ, RS1 is INCOMPLETE, because then it did not rebuild the formal queries or rows.
- (c) Per anchor, the ratio of its retained-pool distance to its all-rows distance (median, min, max), and the ratio of
  the primary thresholds to the sealed thresholds by time. Half the density should raise both by a similar factor.
- (d) Anchor shares by time.
- (e) Group sizes and distinct episodes, primary and secondary.

## Labels

| Label | When | Leads to |
|---|---|---|
| INCOMPLETE | a dataset digest differs, the instrument check differs from the formal report, or the run raised | Fix the instrument and run again; no number from the run is read. |
| INCONCLUSIVE | a primary group is below the 32-row floor in some world | The retained population is too thin at that time to decide; report it. |
| RETAINED SUPPORT FAILS | a world misses the start bar or the anchor bar | The r4 native package does not run on this candidate. The finding goes to the ledger; what a retained-support miss means for the candidate is the definition's decision, not RS1's. |
| RETAINED SUPPORT HOLDS | both worlds meet both bars | The r4 native package may be built on this candidate. |

Precedence: INCOMPLETE > INCONCLUSIVE > FAILS > HOLDS. Every label computed is reported.

## Caps

Exact simulator steps only for rebuilding the anchors (the 96 prefixes in both worlds, about 4,800 steps) and the
starts (128 resets, no steps); 20,000 steps and 600 s wall clock. No training.

## Integrity

`rs1_result.json` (always written; INCOMPLETE with the traceback if anything raises) records the sha256 of
`retained_support.py` and this rule, both dataset digests, the sha256 of `g2_seeds_v2.json`, and the formal report's
manifest hash. No sealed file is read for writing.
