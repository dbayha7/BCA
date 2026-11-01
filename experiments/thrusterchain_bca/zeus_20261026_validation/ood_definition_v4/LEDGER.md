# G6 ledger: every candidate screened under OOD definition v4

Required by DRAFT_DEFINITION.md (G6): every candidate screened, its sigma, mode and outcome, with any later efficacy
claim. Development screens run on development entropy with outcomes known; they are seen data and never confirmatory.
A formal G2 runs once on fresh sealed streams under a rule written first.

| Candidate | sigma | mode | Development (seen) | Formal G2 | Efficacy claim |
|---|---|---|---|---|---|
| knee_s10_v1 (`knee_s10_candidate_v1/`) | s10 | exact, pi_ref v2 (LQR, support-projected) | G2 v1 INCONCLUSIVE (item 5 at 43 of 48); G2 r3 PASS (development entropy) | attempt 1, 2026-11-01 05:19 UTC, approval `5742605a`, manifest 0ba3e4b5...: **NOT DATA-OOD** (no-stall D3 anchors 84 of 96, bar 87; all else passed; H2 median rho(s10) 1.536 vs 0.779). Readout `formal_g2/G2_FORMAL_READOUT_20261101.md`. | none; candidate stopped by its rule |
| knee_s10_v2 (`knee_s10_candidate_v2/`: v1 with the in-support anchor prefix, `prefix_rule.py`) | s10 | exact, pi_ref v2 (unchanged) | development G2 v2, 2026-11-01 05:49 UTC, rule `G2_DEV_RULE_v2.md` (sha 4e9242e4...), v1's development entropy and stream, 48 gated anchors: **PASS** (D3 anchors 48 of 48 in both worlds, v1 had 44 of 48 in no-stall; H1 0.0; H2 median rho(s10) 2.198 vs 0.779; P1 every prefix identical in both worlds). Record `g2_development_result_v2.json`. | sealed `4853c29f` (manifest e7be6e21..., new seed namespace, v1's D4 bounds byte for byte); attempt 1, 2026-11-01 08:39 UTC, approval `1672e956`: **PASS** (P1 128 of 128 anchors identical in both worlds; D3 anchors 96 of 96 stall3 and 93 of 96 no-stall, starts 126 and 127 of 128; D4 2.451 mid, 2.440 deep, inside; B1 1.000; H1 0.0; H2 median rho(s10) 1.765 vs 0.779; 492,896 steps, 42.9 s). Readout `formal_g2/G2_FORMAL_READOUT_20261101.md`; datasets c2a6d616... (stall3), e3549869... (no-stall). | none; G2 is exact arithmetic about the data and pi_ref. Next, each with its own rule and approval: G1, the variance pilot, G3 |

## Under definition v4 r4 (DRAFT, `REVISION_r4.md`, 2026-11-01, not approved)

- **RS1 retained-state support, knee_s10_v2, on the formal G2 attempt-1 datasets** (rule written first,
  `../knee_s10_candidate_v2/retained_support/RS1_RULE.md`; record `rs1_result.json`; readout `RS1_READOUT.md`):
  **RETAINED SUPPORT FAILS by the primary**, on the start share in both worlds: starts 121 of 128 (0.9453) against
  the 0.95 bar (122) in stall3 and no-stall; anchors 93 of 96 (stall3) and 91 of 96 (no-stall) against 87. The t = 0
  reference group holds 61 retained deep-load rows, so its 0.99 "higher" quantile is the group's single largest
  distance, and that one draw set the start bar. The pre-registered secondary (same retained pool, time window +-2,
  168 to 314 rows per group) gives 126/128, 96/96 and 127/128, 93/96, the sealed D3's counts exactly. Instrument
  check: the sealed all-rows D3 recomputed on the rebuilt queries equals the formal report. The primary was not rerun
  and is not relabeled here; what the miss means for the native screen is the user's decision, to be written here.
- **The native screen for knee_s10_v2 (the one package the user chose, 2026-11-01 06:38 EST):**
  `../knee_s10_candidate_v2/native_screen/` (rule `SCREEN_RULE.md`, written first). One dose, kappa 63; Stage 1's
  count-matched RANDOM as the labelled control (confound stated); the formal G2 v2 datasets by digest; fresh screen
  seeds (`screen_seeds.json`, audited); G1 folded in as the `g1` unit (`../knee_s10_candidate_v2/g1/G1_RULE.md`),
  reported and read by the screen's validity. Units: preflight, g1, 12 warmups, r0, 12 native branches, 3 BC-only,
  3 evaluations, the R1 stop, 12 BCA branches, 3 BCA evaluations, the decision. Caps: 1,800 s per training job, 10 h
  training, 2 h and 36M transitions evaluation, 1 h and 9M steps for g1. 22 tests pass; sealed; awaiting the user's
  review of the frozen protocol and budget and their own approval (`SCREEN_APPROVAL.json`); nothing has trained.
  The approval's RS1 sentence is the user's decision on the primary miss above.
- **Native screen, attempts 1 and 2 (2026-11-01, user-run in WSL):** both INCOMPLETE at the `g1` unit before any
  training. Attempt 1 (manifest 6cbf5b2d…, approved by the user): the Stage 1 environment setter refused the g1 unit's CPU
  platform ("Conflicting environment values"); one code-only fix in `screen_worker.run_job` (the g1 kind sets the
  recorded environment with `JAX_PLATFORMS=cpu` itself), re-sealed as manifest 584b30a1…. Attempt 2 (approved
  naming attempt 2, preflight reused): `dataset_knee.load` failed its exact reset replay, "Episode 16: initial
  state lineage differs", in 7.0 s, for the stall3 dataset. Cause on record (`../knee_s10_candidate_v2/
  reset_lineage_diagnostic_v1/receipt_20261101T092712_649674_EST_424.json`, written for the user's separate minimal
  comparison, which failed at the same check the same morning): the formal G2 datasets were collected and validated
  in the Windows venv; the recorded Linux environment's reset of episode 16 reproduces `wave_1` one ULP off
  (0x3fc11b368736b87b against ...7c), the float32 observation equal. Not a data fault and not a code fault of the
  screen: the v3 loader's regeneration is not bit-portable across the two platforms. The rule allows no attempt 3;
  what follows (a versioned amendment with an authenticated, non-regenerating loader, or stopping) is the user's
  decision, to be written here. GPU time spent by the screen: none.
- **Native screen amendment A1 (2026-11-01 13:48 EST, the user's decision: "amend"):** change 7 in
  `screen_bind.py`, the authenticated non-regenerating loader `screen_data.py` (archive bytes, canonical content and
  the formal G2's sealed run record all checked by sha256; the two recorded archives only; no tolerance; no reset or
  replay), and attempt 3 permitted after the named loader fault only (`screen.py` checks the failed unit's recorded
  error; never a fourth). Nothing else changed: dose, control, data bytes and digests, seeds, units, reads, bars,
  readings, caps. 26 tests pass. Re-sealed; attempt 3 awaits the user's approval naming it, and the GPU, which the
  user's separate minimal-comparison recovery run holds as of 13:45.
