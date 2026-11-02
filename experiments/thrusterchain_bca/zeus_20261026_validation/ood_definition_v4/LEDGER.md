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
- **Native screen, attempt 3 (2026-11-01 14:01-14:05 EST, user-run, manifest e2d09d22…): INCONCLUSIVE, G1 NOT
  REPRESENTABLE** (`../knee_s10_candidate_v2/native_screen/SCREEN_READOUT_20261101.md`; record
  `screen_run/attempt_3/`). Instrument checks exact (I0 median rho 1.7652868718841468 to the last digit, H1 0.0,
  F0 ok, upstream match). Stall3 fit at 20,000 updates: median KC(s10) over the 72 valid anchors 1.559 / 1.403 /
  1.437 against the band −0.15 ≤ KC ≤ 0.980 (0.3 × median rho 2.767 + 0.15); L1 shares 0.111 / 0.222 / 0.236,
  pooled 0.190 against 0.70; over-capture (the critic values s10 too high) 0.68 to 0.76; q* 0.111; in-support
  resolution band 1.843 (median 0.812). Smooth placebo: median KC(s10) ≈ 0, 55 to 64 percent within 0.15. 60k
  diagnostic: median KC 0.92 to 1.10, pooled share 0.389. Profile: the critic reproduces about half the drop at s05,
  s10, s15 and s20 alike. Under section 5: stop this candidate on this host configuration; a changed host setting
  or feature is a new candidate; data, slot and tolerance may not change. Nothing trained; GPU hours 0; 7,968,000
  exact CPU steps. G2 PASS and RS1 stand as recorded. The screen's approval is spent; no attempt 4 exists.
- **I0 platform rehearsal (2026-11-01 14:15 EST, `../knee_s10_candidate_v2/i0_platform_check_v1/`, attempt-free,
  read-only):** before attempt 3 is approved, the screen's exact Windows-versus-Linux comparisons are rehearsed
  outside any attempt: the g1 unit's I0 (median exact rho(s10) over the 96 anchors against the formal report at 1e-9
  relative, and H1), the amendment A1 loader for both archives, and the data-plan re-derivation every job performs.
  Windows: all equal exactly (relative difference 0.0; 201,600 exact steps). The WSL receipt decides whether attempt 3
  is approved or I0 is amended first; it is the user's to run.
- **Native screen attempt 3 (2026-11-01 14:01-14:05 EST, user-run, approval naming attempt 3, manifest e2d09d22…):**
  the A1 loader read both archives; I0 passed exactly on Linux (median exact rho(s10) 1.7652868718841468, H1 0.0);
  **G1 read NOT REPRESENTABLE** (stall3 fit at 20,000 updates: median KC(s10) 1.559 / 1.403 / 1.437 against the
  0.98 band; share 0.111 / 0.222 / 0.236 against 0.60, pooled 0.190 against 0.70; 72 valid anchors; 60k diagnostic:
  median 0.94 / 0.91 / 1.10, share 0.39 / 0.43 / 0.35; the no-stall placebo fits to within +-0.05 at every slot).
  The screen read **INCONCLUSIVE (representation)**; nothing trained; 7,968,000 exact steps, 196.8 s, GPU time 0.
  By G1_RULE.md the candidate knee_s10_v2 stops on this host configuration; a changed host setting is a new
  candidate. Readout: `../knee_s10_candidate_v2/g1/G1_READOUT_20261101.md`. Attempt records uncommitted.

## Outside the gate chain: the OOD-action overestimation comparison (2026-11-01, after G1 NOT REPRESENTABLE)

- **`../knee_s10_candidate_v2/overestimation_v1/` (rule `OVERESTIMATION_RULE.md`, written first; the user's choice
  18:25 EST: "Yes, build it").** The literature's question on the pieces already built: native versus BCA (kappa 63)
  actors from the same 10k fork on the knee_s10_v2 CUTOFF stall3 data (digest c2a6d616…, W = |a0| > float32(0.8)
  withheld), Unifloral TD3+BC host through the native screen's worker and its seven bindings. Per 20k policy:
  128 common starts (returns, W exposure), 48 visited states (own-action critic error Q1 − Q^pi, own unclipped
  exact continuation), 192 dataset rows (logged action and three withheld a0 magnitudes 0.85/0.9/1.0: critic error,
  width U, and for BCA the penalized score's error). Paired per seed: dJ, de_own. Bars fixed: PROMISING = de_own ≤
  −0.5 and dJ ≥ −2.0 in every seed; HARMFUL = dJ ≤ −2.0 in two seeds; NO EFFECT = all inside; NOT REACHED = native
  W exposure < 0.01 everywhere; INCONCLUSIVE on the I1/I2 instrument checks. Seeds: the screen's unspent training
  and data-plan rows plus two new audited names. Caps: 1,800 s per training job, 3.5 h; 900 s and 200,000
  transitions per evaluation. 12 tests pass (the measurement runs on the real simulator with a fake policy; I1
  identity holds). Sealed; awaits the user's approval (`OE_APPROVAL.json`) and launch. Exploratory; supports no
  protocol claim; says nothing about the knee departure.
- **Design audit and r5 (2026-11-01 evening):** `DESIGN_AUDIT_r4_20261101.md` (defects D1-D8 against evidence on
  file: the native TD critic resolves a 0.1 step in a0 to 0.05-0.15 where the truth is smooth and misses the kink by
  about 85% when present; G1's regression at 20k was 10-50x coarser than the TD critic in support). `REVISION_r5.md`
  and the package `../knee_s10_candidate_v2/native_screen_r5/` (rule written first; 17 tests; unsealed, awaiting the
  WSL dry run, then the seal and the user's approval). G1 becomes a reported calibration unit; twin learnability
  becomes findings; returns reported; B2 judged against shuffled-U and constant-penalty nulls on net gain; pooled
  anchor reads; MIXED removed; stage 0 exempt from attempts; dry run required before the seal.
- **Overestimation comparison v1, attempt 1 (2026-11-01 18:39-20:00 EST, user-run, manifest 9834a461…):
  INCONCLUSIVE by the instrument tolerances alone; computed PROMISING in every seed**
  (`../knee_s10_candidate_v2/overestimation_v1/OE_READOUT_attempt1_20261101.md`; record `oe_run/attempt_1/`). All
  15 units completed (GPU 4,475 s; 335,448 exact transitions). I1 held to 1.1e-7 to 2.4e-7 relative against a 1e-9
  bar and I2 was not bit-equal: the float32 GPU actor differs in the last place across batch sizes (128 rollout, 48
  probes, shrinking continuation); the bar had been calibrated on the CPU float64 identity. Reads: native actor
  saturates a0 at −1 (median |a0| 0.993, W− side) for 67 to 70 percent of steps; own-action over-valuation about
  +180 (Q1 median −80 against exact −305, every probe over-valued); at dataset states e_data +24, e_W +25 (the
  error is carried by state shift, not the withheld action alone); U_W − U_data +0.01 to +0.03 (the width does not
  flag withheld actions; U is about 0.5 higher off-data). BCA at kappa 63: dJ +11.9, +41.3, +22.8 (base about
  −360); de_own −0.6, −16.2, −28.9; dEXPO +0.004, −0.062, −0.239. Amendment A1 (I1 1e-6 relative, I2 1e-6 per
  component, values recorded, attempt 2 after this named cause; training reused, evaluations rerun) written,
  13 tests pass, re-sealed; awaits the user's attempt-2 approval. Exploratory; no protocol claim; mechanism open.
- **r5 dry run (2026-11-02 01:05-01:33 UTC, user-run in WSL): COMPLETED.** preflight, i0 (median exact rho(s10)
  1.7652868718841468 on Linux, H1 0.0, both archives by digest, both data plans equal), smoke collection (515,600
  steps), and one job of every kind on smoke data: warmup 512 s, r0, branch 480 s, bc_only 16 s, evaluate, bca_branch
  493 s, evaluate_bca, all completed; device kind recorded; projected training 17,856 s of 36,000. Receipt
  `../knee_s10_candidate_v2/native_screen_r5/dry_run/DRY_RUN_RECEIPT.json`; the package was then sealed.
- **Overestimation comparison (`../knee_s10_candidate_v2/overestimation_v1/`, another package, user-run):** attempt 1
  INCONCLUSIVE on its instrument checks alone (I1 misses 1.1-2.4e-7 against 1e-9; I2 bit equality across GPU batch
  shapes); amendment A1 (tolerances to 1e-6, cause recorded) and attempt 2 (training reused, evaluations rerun):
  **PROMISING** by its rule (de_own -0.61/-16.2/-28.9, dJ +11.9/+41.3/+22.8). Exploratory; its canonical block shows
  BCA's e_W - e_data above native's in seeds 0 and 2.
- **Overestimation comparison v1, amendment A2 (2026-11-01 21:02 EST, the user: use the honeypot-edge and
  underestimation stress-test setup):** the honeypot half is attempt 1's stall3 design unchanged (W is a trap; the
  native actor walked in). The underestimation half is the same comparison in the no-stall world (the formal G2
  v2 twin dataset by digest e3549869…), where W is harmless: a penalty that keeps the actor out of W there and
  loses return over-corrects. Labels fixed before any no-stall number: OVER-CORRECTION (dEXPO ≤ −0.05 and dJ ≤
  −2.0 in two or more seeds), CALIBRATED (dJ ≥ −2.0 in every seed), PLACEBO WITHOUT STAKE (native W exposure
  < 0.01 everywhere), PLACEBO MIXED; the reading is the pair "honeypot / placebo". Nine new training jobs and six
  evaluations; attempt 1's stall3 training reused; stall3 evaluations rerun under A1's tolerances. 14 tests pass
  (the measurement runs on the real simulator in both worlds). Re-sealed; awaits the user's attempt-2 approval.
  Post-hoc on attempt 1 (reported only, appended to its readout): U correlates with the critic's error magnitude
  at 0.26 to 0.49 and is sign-blind (U on over- versus under-valued probes 1.01 to 1.06).
- **Overestimation comparison v1, attempt 2 under A1 (2026-11-01 20:37-20:40 EST, user-run, manifest 64d331e4…):
  PROMISING in every seed.** Training reused from attempt 1; six evaluations rerun; I1 0.0 relative, I2 ≤ 5.7e-7.
  Paired reads equal attempt 1's: dJ +11.94, +41.30, +22.79; de_own −0.61, −16.23, −28.91; dEXPO +0.004, −0.062,
  −0.239. The honeypot half is decided; exploratory, no protocol claim, mechanism open (U sign-blind). Because a
  decided comparison takes no further attempt, amendment A2 (the no-stall placebo) runs as its own comparison in
  `oe_run_a2/` with its own approval sentence, reusing the stall3 training units by path and rerunning every
  evaluation. Record `oe_run/attempt_2/`.
- **Overestimation comparison v1, A2 run (2026-11-01 21:26-22:47 EST, user-run, manifest 89cd48e8…): PROMISING /
  PLACEBO MIXED** (`../knee_s10_candidate_v2/overestimation_v1/OE_READOUT_a2_20261101.md`; record
  `oe_run_a2/attempt_1/`). Stall3 training reused; nine no-stall training units (4,477 s GPU); twelve evaluations
  (670,896 exact transitions); I1 ≤ 4e-8, I2 ≤ 5.7e-7. Honeypot (stall3): dJ +11.9, +41.3, +22.8; de_own −0.6,
  −16.2, −28.9. Placebo (no-stall, W harmless): native J about −62 with W exposure 0.79 to 0.85 and own-action
  over-valuation +4 to +11 (against −360 and +180 in stall3 for the same saturating behavior); BCA cuts W exposure
  by 0.25, 0.08, 0.32 there too (U is sign- and world-blind: U_W − U_data +0.01 to +0.03 in both worlds) at a
  return cost of −0.3, +3.4, −6.7: one seed past the 2.0 bar, so neither CALIBRATED nor OVER-CORRECTION. Mild
  underestimation signature in no-stall (BCA e_data −0.6, −0.2). Exploratory; three seeds; mechanism untested.
