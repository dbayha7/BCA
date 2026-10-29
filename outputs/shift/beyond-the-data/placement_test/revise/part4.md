
---

## 10. Module W: the known shift (recommended; decision D4)

**Why.** BCA's weights can earn credit only where the evidence law and the deployment law overlap with a known ratio.
On the main log the recorded current is U[0.1, 0.7] and strong current U[0.7, 1.1], so the supports are disjoint and no
weight exists (`thruster_chain.py:87-92`). The weighting test's world has an exact current weight `w(c)` on [0.1, 1.1]
(MAIN `experiments/shift_iw/world.py:7-16`), and its recorder's random-command part (`eps / 16`, CONN
`channel/command_le_train.py:73-84`) fills the rungs past 80% (on seed 0 arm B had 42, 54 and 78 thruster-1 tries in
the 80-85% rung on medium-replay, medium and medium-expert, and 35-67 in each higher rung on medium-expert; CONN
`outputs/shift_iw/explore/arms_abc_f0/SCREEN_seed0.md:41, 46-52`). So module W is also the only place where C's band can
act past the knee (section 2.3). It is stalling-only (`world.py:69`; `data.py` hard-codes the stalling world at :80,
336, 454, 646 and 652), so a no-stall twin must be built: the generator with `PHYSICS` passed in (`world.simulate` and
`HoldRecorder` already accept physics, `world.py:218, 324`).

**Design.** Quality medium-expert (the hosts' primary quality); weighting-test seeds 0-4 (pilot) and 25-29
(confirmatory); Unifloral TD3+BC hosts in S and the new twin N; tries = training-law log rows of the calibration split
outside the bank, or the evaluation split, outside the ball (`arm_b.py:8-9`); deployment = the strong-mix law; weights
exact `w(c)` (B-K2, C-K5) or estimated by a classifier on the weighting test's classifier samples (B-K3). NET in module
W is defined as in section 6.1, under the strong-mix law.

- **B arms**: B-RUNG with the residual score and BCA's weighted tail gate (the shipped Monte Carlo gate with its floor of
  36 is primary here, because with unequal weights no exact Beta form exists; a 10,000-draw version is reported)
  against rejection sampling (accept each try with probability `w / max w`, then the exact binomial test), weighted
  split conformal (Tibshirani et al. 2019), a self-normalized weighted mean with an empirical-Bernstein bound, the
  unweighted gate, and shuffled outcomes.
- **C arms** (S and N): C-BCA with exact state x command weights for the strong mix (primary); matched in S and locked
  in N: unweighted C-BCA, C-F0_m, C-RATIO_m, C-SC_m, C-SUP, C-HINGE, C-EVEN; C-ORACLE. About 51.5 runs per seed (9.2).
- **B-K2 / B-K3 pass iff**: in the stalling world BCA's weighted gate makes harmful certifications (true harm rate above
  alpha_B, from fresh tries) in at most 5% of decisions (pooled one-sided Clopper-Pearson upper bound <= 0.073); every
  rival above that rate is reported invalid and dropped; each remaining rival's level is set on the pilot so that its
  stalling-world rate of harmful certifications equals BCA's (matched harm), frozen in a confirmatory slot; and in the
  no-stall twin BCA's weighted gate certifies at least 10 percentage points more truly safe rungs than the best
  remaining rival at matched harm (pooled over seeds, Holm p <= 0.05 in {B-K2, B-K3}). With exact weights the toys gave
  33.6% against 19.0% for rejection sampling (LIVE 10811); evidence floors are about 29 / 39 / 83 tries for no, moderate
  and strong shift (LIVE 10987-10992).
- **C in module W**: C-R1W, `NET(C-BCA) - NET(pooled-best of C-HINGE, C-EVEN and clip always)`: mean >= +5 and LB > 0;
  C-V1W, C-V3W as on the main recorder; C-V2W judged overall, in the near-knee zone and **past 80%** (RS <= 0.50 and CAC
  >= 0.85 there; no certified draw past 80% fails it). Machinery family {C-K1W vs C-F0_m, C-K2W vs C-RATIO_m, C-K3W vs
  C-SC_m, C-K4W vs C-SUP, C-K5 weighted vs unweighted C-BCA}: each mean >= +5 and Holm p <= 0.05; C-K5 also needs C-V1W
  met by both arms.
- The weighted promise is empirical, not proven, at these sizes (PLACE `red.challenges[3]`; MAIN
  `experiments/wbcp/DEPENDENCE.md:1114-1133`); module W measures it, bank by bank (C-V3W).

---

## 11. Code plan

### 11.1 Base branch, reuse and what is not reused

- **Base.** A new branch from `origin/claude/shift-hosts` at `608eefa` (the hosts' approved version 1 bytes), with
  `origin/main` merged. HOSTS does not contain MAIN (7 MAIN commits are missing), but between their merge base and MAIN
  no file under the hosts' `EXTERNAL_CODE` paths differs (checked with `git diff`), so the hosts' approval still
  verifies there. HV2 edits `actors_f32.py` and `evaluate.py` (`git diff` HOSTS..HV2), so it is not the base; if the
  hosts' version 2 is frozen later, S1 catches the change through the code manifest.
- **Clone of the repository.** The test runs from a clone separate from the hosts' `/root/bca-hosts`, and every entry
  point sets `sys.dont_write_bytecode` first (stop rule S11).
- **Test references only (never imported)**: CONN `outputs/shift_iw/explore/arms_abc_f0/code/arm_b.py` and `arm_f0.py`;
  MAIN `step1lib.py`; MAIN `bca_mean_gate_toy.ipynb` and `bca_weights_toy.ipynb`. They are read with `git show
  <commit>:<path>`, their blob SHA-256 pinned in `prespec.json`, and the package's restatements are tested against them.

**Reused and imported (none of these calls a study's gate):**

| Need | Where |
|---|---|
| Simulator, start laws, banks, env (scoring environment) | MAIN `experiments/shift/thruster_chain.py` (`dynamics`, `dynamics_jax`, `draw_starts`, `bank_starts`, `world_physics`, `ThrusterChainEnv`) |
| Exact truth | MAIN `experiments/shift/truth.py` (`rollout`, `q_value`, `q_value_jax`, `points`, `attribution`); not `anchors`, which takes no starts |
| Data, gap, bank | MAIN `experiments/shift/dataset.py` (`generate`, `simulate`, `draw_streams`), `shift.py` (`define_gap`, `make_removal`), `recorder.py`, `calibration/bank.py` |
| WBCP | MAIN `calibration/wbcp.py` |
| Unifloral import | HOSTS `experiments/shift_hosts/unifloral_import.py` (`prepare`, `args_for`, `make_transition`, `init_like_training_block`, `make_unifloral_step`); MAIN `baselines/unifloral/algorithms/td3_bc.py`, `bc.py` |
| BCA arithmetic | HOSTS `bca_core.py` core functions (seed and key rules :96-157, `fit_scale_core` :282, `width` :364, `bank_scores` :387, `calibrate_bank` :399, `apply_refresh` :417); `unifloral_td3_bc_bca.py` `_own_target` :316, `_band_values` :343-349 and `band_parts` :352-359 as templates for the Q1 version |
| Threshold tables (training environment) | HOSTS `iw.py` (`calibrate` :94, `mass_grid` :136, `threshold_table` :157, `lookup` :180) |
| Policies and hand-off | HOSTS `policies.py` (`MLPAdapter`), `actors_f32.py`, `handoff.py` |
| Pure helpers | HOSTS `band_readout.py:85-157` (`width`, `refusals`, `outcomes`, `episode_bootstrap`, `summarize`, `groups`, `in_ball`); `evaluate.py` `paired` :356, `late_peak` :275, `costs` :282 |
| Coverage arithmetic (scoring environment only; it imports `shift_iw.world`, `coverage.py:42`) | MAIN `experiments/shift_iw/coverage.py`, with COR computed in `coverage_c.py` |
| Module W data | MAIN `experiments/shift_iw/data.py`, `world.py` |

**Not reused** (each calls a study's gate, or does not fit): `make_train_step` with BCA present, `refresh` and
`recompute_bank_scores` (HOSTS `unifloral_td3_bc_bca.py:243, 392, 408`); `iw.threshold_tables` and `iw.iw_refresh`
(`iw.py:266, 485`); `band_readout`'s `band_values`, `readout` and `coverage` (:296, 331, 354); `evaluate`'s
`score_actor` and `closed_loop_check`, and `score`, which knows only the standard and weighting banks with the base and
stall anchors and needs the laptop's scoring cache (`evaluate.py:171-263, 313`); `runner` and `stages`; `lower_edge.py`'s
band readers and `Runner`, which call the weighting test's gate (`lower_edge.py:19-22`). The hosts' gate opens only on
registered made-up arrays or on a hosts' Approval (HOSTS `guard.py:127-136`). A stage-0 test fails if any placement
module calls `guard.require_bca`, `gate.require_run_approval`, `gate.verify` or `learners_iw.require_gate`, or builds a
hosts' Approval. The package lives at `experiments/placement_test/`, outside the hosts' fingerprint list (HOSTS
`gate.py:56-65`); nothing is added under `experiments/shift`, `experiments/shift_iw` or `calibration`, and the hosts'
approval is not reused for this study.

### 11.2 To write

| Module | Does |
|---|---|
| `PRESPEC.md`, `prespec_pilot.json`, `freeze_pilot.py`, `freeze_confirm.py`, `CHANGES.md`, `gate.py`, `approve_placement.py` | the frozen rules, the two freezes, recorded changes and your terminal approval |
| `seeds.py` | the seed allow-list (S2); the stream registry with the padding check (section 3.5) |
| `data_main.py` (scoring) | gapped logs in S and N; spliced MW+ and MW-; evidence; B's conditioned tries and decoupled rest; per-row `mu(x)`, hard-water flags and `Z(x)`; Unifloral prepare and hand-off; leak checks |
| `hosts.py` (training) | the chunk loop (restating `td3_bc.py:329-341`); TD3+BC runs; `load_bc()` and BC runs in their own process; run records; a resumable per-run ledger |
| `dynamics_train.py` (training) | the restated dynamics from handed-off columns, under x64 (section 3.1) |
| `scores.py` (training) | the residual score on a Unifloral critic (min head, Q1, noisy-target variants) |
| `harm_switch.py` | A's rules at their own margins, shuffles, A-RAND by expectation, cells, kernel, A-KQ, A-KPERM, boundary shifts |
| `safety_cert.py` | B's exact and shipped gates, mean gates, HCOPE-style, looks, e-process, rung walk, maps, decoupled tau, boundary shifts |
| `recorder_density.py` (both) | the recorder's density, `Z(x)` by Monte Carlo (scoring), the query law's density and the weights (training), with the cross-environment test |
| `lower_edge_unifloral.py` (training) | the restated TD3+BC step with the lower-edge slot; Q1 band parts; its own refresh with per-query tables; C's penalties, C-SHUF and the oracles; matching; saving the networks of each declared refresh |
| `online.py` (scoring) | the episode runner (env.step arithmetic); the float32 wrapper for truth calls; mixed physics; fresh banks; `anchors_on(starts, physics, recorder_stream)`, tested to reproduce `truth.anchors` on the standard bank bit for bit; the no-stall scorer; the staged-B loop and the monitor replay with hand-offs; attribution |
| `truth_pop.py` | replicate blocks (split by physics for mixed worlds), staged-B replicate sets, populations, boundary shifts |
| `coverage_c.py` | COR, RS, CAC with soft refusals, the zones, C-V3, sharpness |
| `s4_checks.py` | the package's own open-loop and closed-loop float32 checks (stop rule S4) |
| `stats.py` | paired t bounds, Holm (the hosts' procedure), Clopper-Pearson, pooled marks with guards, bootstraps (episode, two-stage, unit), design effect, the size rule, the discriminability step |
| `report.py` | `results.json` and `results.md`, write-once |
| `tests/` | unit tests on made-up data for every module (stage 0): the gate-call test, the cross-environment tests, the weight identity, COR with refusals, stream distinctness, `anchors_on`'s reproduction, k = 0 reproduction |
| Module W only | the weighting-test no-stall twin; its prepare; the weighted gates |

### 11.3 Outputs and report

- Official outputs only under `runs/placement_test/<stage>/`, write-once, each with a SHA-256 sidecar; run records hold
  the fingerprint, seeds, streams, CPU or card, software and timings; every process pinned to one CPU with
  `OMP_NUM_THREADS=1` and no bytecode (step 1 PRESPEC.md:107-113).
- `decisions.json` (locked before any return in the confirmatory stage), `lock.json` (network fingerprints and
  strengths), `values.json` per seed-world, `results.json`, `results.md`. Until a stage is complete, only integrity checks
  and spend are shown.
- `results.md`: per claim a badge (passed, shelved, flagged, pending, not judged, no room, did not learn, vacuous,
  underpowered by design, dominated), the numbers against the mark, per-seed values, and the fixed wording; the three
  verdicts per option; then per option the write-up in your format: The idea / What we tested (and the kill rule) / What
  happened / Why (HOSTS PRESPEC.md:515-516). Every pilot page carries "EXPLORATORY: seeds 0-4 were seen; the margin was
  fitted on these seeds".

---

## 12. What this design rests on (verified for this document)

- Step 1's certificate, rules and process: MAIN `outputs/shift/beyond-the-data/restructure/step1/PRESPEC.md:5-8, 51-63,
  133-168, 192-268, 246-251, 340-347`; code `step1lib.py:174-352`.
- Round 1's routes (held-out rows 106.1; 55.7 without the margin; plain tests at the margin also 106.1; ladder +23.0
  over the middle strengths; small evidence; mixed world 43.2 against a cap at the knee 41.5 and a region oracle 52.6;
  run-time monitor): MAIN `restructure/routes/round1/ROUND1.md:13-146`.
- Step 2's arms, matching (1%, at most 5 rounds), marks and results (NET 102.9; best world-blind rival 77.7; conformal
  97.3; range hinge 76.5 with no-stall cost 30.9; "Bayesian or only larger cannot be told"): MAIN
  `restructure/step2/PRESPEC.md:97-257`; CONN `restructure/step2_results/results.md:9-38` and
  `addon_report/report.md:138-230`.
- Fallback: deployment clip +106.0 in strong current, -61.8 in the no-stall world; every step to the cloned actor +68.9:
  MAIN `outputs/shift/beyond-the-data/fallback.md:83-88, 167-178`. Gated clone fallback +73.8 / -30.4 and no-stall
  -109.5 against the clip's -61.6: HV2 `outputs/shift/beyond-the-data/results-2026-10-13/deploy_tables.md:17, 33, 79,
  95`.
- Stage 1's room (oracle 79.7, exact long-run harm 82.2, no-signal clip 93.1, no-stall costs 1.3 and 30.9; actor-only
  against a frozen critic): MAIN `stage1/RESULTS.md:28-65`; `stage1/code/harness.py:373-376, 530-533, 559, 791`.
- Stage 2 v6 (lower edge 24.9 against the matched hinge 70.5; no-stall cost 14.8; co-trained 28.0 against -16.8):
  MAIN `results-2026-10-14/README.md:20-36, 69-77`.
- ThrusterChain, truth, data, recorder: MAIN `experiments/shift/thruster_chain.py:1-53, 83-92, 143-149, 195-251,
  308-336, 389-481`; `truth.py:1-35, 77-108, 330-383`; `dataset.py:4-15, 50-65, 152-226`; `recorder.py:8-60, 210-270`;
  `shift.py:1-60`; `experiments/shift/__init__.py:81-111`.
- WBCP and its promise: MAIN `calibration/wbcp.py:1-117`; `experiments/wbcp/DEPENDENCE.md:1-16, 1110-1135`;
  `experiments/shift_iw/coverage.py:1-60`.
- Unifloral and the hosts: MAIN `baselines/unifloral/algorithms/td3_bc.py:23-46, 174-279, 326-342`, `bc.py:1-22`;
  `configs/unifloral.json`; HOSTS `experiments/shift_hosts/PRESPEC.md:138-226, 254-282, 466-606, 613-621`,
  `unifloral_td3_bc_bca.py:1-40, 102-413`, `bca_core.py:90-420`, `iw.py:60-490`, `band_readout.py:40-465`,
  `evaluate.py:1-125, 302-362`, `unifloral_import.py:55-76, 186-234`, `gate.py:1-70`, `guard.py:115-136`; HV2
  `PRESPEC.md:1-46, 866-930`; `git diff` HOSTS..HV2 and the merge base of HOSTS and MAIN.
- Arm B and the lower edge: CONN `arms_abc_f0/code/arm_b.py:1-155`, `arms_abc_f0/RESULTS.md:40-48, 118-126`;
  `channel/command_le_train.py:1-18, 73-118`; `channel/command_band_check.py:1-130`; MAIN
  `experiments/shift_iw/lower_edge.py:1-64, 196-224, 293-312`.
- Weighting-test world: MAIN `experiments/shift_iw/world.py:1-70, 218, 324`.
- Seeds: CONN `near80/code/near80lib.py:64-72`; `seed_registry.json`; mixed-world counts
  `routes/round1/scratch/_where/feas_amp_0_1_2_3_4.json`.
- Floors, power, marks and precision: `measure/floor_arith.out`; REV `revise_arith.out`, `budget_arith.out`;
  `review_cov/arith.out`; `review_stats/marks_arith.out`; `review_feas/card_arith.out`, `bench_wbcp.out`; LIVE 5104-5112
  (bank standard errors), 10502-10560 (tail-gate floor; 36 with 1,000 draws), 10985-10996 (weighted floors).
- Placement analysis and its red team: PLACE (theory findings 3, 5, 9; rank test_plan T1-T6; red challenges 1-13,
  missing options, revised recommendation).
- Benchmarks: `bench/conc_alone_*.out`, `conc_bca_*.out`, `conc_lower_*.out`, `rollout_1cpu.out`, `rollout_4cpu.out`.
- numpy key padding: checked in the scratch environment (numpy 2.2.6): `default_rng([20261019, 7, 25, 0])` and
  `default_rng([20261019, 7, 25])` give the same first integer; with five or more words a trailing word changes the
  stream.

---

## 13. Review round

Four reviews of draft 1 were read: coverage and validity (R1, 18 findings), methods and credit (R2, 18), feasibility
and code (R3, 17), statistics (R4, 17). Each finding was re-checked against the sources before it was applied. "Applied"
means as proposed; "applied in part" names what was changed or left out and why; "rejected" gives the reason.

**R1, coverage and validity**

| # | Finding | Status and where |
|---|---|---|
| R1.1 | C-V1 judged only average coverage; BCA promises coverage per bank | Applied: C-V3, primary (6.4, 7.4), also for C-SC and C-RATIO, with the per-bank credit reported. Part (ii) as a mark rejected (kept as a readout): an exact-95% rule fails "at most 1 of 10 seed-worlds" 8.6% of the time (REV section 2), and it adds ten draws to the 2,000 units of part (i). |
| R1.2 | C-V1's band is one that nothing uses | Applied: judged on the bands of 7 declared refreshes with that refresh's networks; staleness and the final band reported (5.1, 6.4). |
| R1.3 | Staged B's threshold and tries share episodes; B-V1 assumed a fixed tau | Applied in part: per-replicate error at the replicate's own tau (6.4); offline tau from calibration episodes that supply no try (3.2). Changed: staged tau comes from 50 fallback reference episodes, not 50 probe-map episodes; they are equally independent of the tries, cost no probing, and F runs first anyway (5.3). |
| R1.4 | B-V2 judged an event the certificate never promised | Applied: B-V2 is the share of departing steps with `s_t > tau`; the attribution shares and harm mass are readouts (6.4, 7.3). |
| R1.5 | The query law's states give the weights a spurious Z(x) | Applied: kept-row states, `w = q Z / beta` (section 1, 4.3). Corrected: the proposed check "mean bank weight is 1" fails whenever the query law has mass the recorder never covers; the identity is `mean(w) = 1 - P_query(w = +inf)`. |
| R1.6 | C-V2 can be passed by soft refusal and easy draws | Applied: soft refusals counted; judged overall and in the near-knee hard-water zone; sharpness readout (6.4, 7.4). Past 80%: reported on the main recorder, where refusal is the support rule by construction, and judged in module W (with R2.4). |
| R1.7 | B-V1 is vacuous far from the boundary | Applied in part: a judged boundary null at `pi(tau_pop) = 1.1 alpha_B` (6.4, 7.3); alpha_B, 1.2 and 1.5 alpha_B and the in-band ladder decisions are reported, not judged, since the judged shift sits next to the hardest point and more conjunctions would only fail exact rules more often. "Vacuous" badge applied to B-V2; a boundary null cannot be vacuous. |
| R1.8 | B-K1 confused "no outcome information" with the evidence floor | Applied: the permutation p-value test (7.3). |
| R1.9 | m_on was set on a different population from its tries | Applied: first uses against fallback reference episodes, as the tries and tau are built (3.6 step 7). |
| R1.10 | Looks at 10/30/100 cannot certify before the last | Applied: looks at 0.4N, 0.7N, N (40/70/100 at N = 100) (5.3). |
| R1.11 | "A true 5% passes" is false for every-seed-world marks | Applied in part: pooled bound <= 0.065 with a per-unit guard (7.2, 7.3). The guard is 73 in 1,000 (99.9th percentile), not 66 (99th): over 10 units a 66 guard alone fails an exact-5% rule about 10% of the time, 73 about 0.6% (REV section 1). |
| R1.12 | A-K3's expected NET for A-INS cannot be computed | Applied: realized NET; a bootstrap E-NET reported, labeled (7.2). |
| R1.13 | Attribution runs on other arithmetic than the episodes | Applied: float32 wrapper before every truth call; S8 extended (5.2, 7.1). |
| R1.14 | Staged B-RUNG horizon, parallelism and truth unspecified | Applied: parallel probing, at most 2 families (with R3.10), truth per realized map (5.3). |
| R1.15 | The reused coverage code counts refusals as misses | Applied: COR in `coverage_c.py` with a stage-0 test (6.4). |
| R1.16 | Fresh-bank anchors cannot come from `truth.anchors` | Applied: `anchors_on`, with a registered recorder stream and a reproduction test (3.5, 11.2). |
| R1.17 | World in the stream tags breaks shared random numbers | Applied: no world in tags 4, 5 and 6 (3.5). |
| R1.18 | Per-seed values appear before a stage completes | Applied (8.1, 11.3). |

**R2, methods and credit**

| # | Finding | Status and where |
|---|---|---|
| R2.1 | "BCA earns credit" could rest on checks that pass almost automatically | Applied: three verdicts (7.5); A-K2 is now the ladder claim on confirmatory seeds; the shuffle is a sanity check; C-K6 counts as outcome reading. |
| R2.2 | Clipping always is not the best world-blind rule | Applied: the set W (section 1, 4.1); A-R1 against its pooled-best; W reported beside B; domination rows (7.2-7.5). |
| R2.3 | C's primary bar let C "work" while far below A's switched clip | Applied: C-R1 against the best switched plain action; draft 1's C-R1 kept as C-R1b (reported); domination rule (7.4, 7.5). |
| R2.4 | On the main recorder C is refusal alone past 80%; module W lacked controls | Applied in part: the construction declared (2.3); module W's C arms extended to the matched set, one Holm family, recommended (10, D4). C-V2 past 80% is judged in module W but only reported on the main recorder: there it fails by construction, which the declaration already states. |
| R2.5 | Mean and shift rivals used a quantile-unit margin | Applied: per-statistic margins m_f (3.6, 4.1, 4.2). |
| R2.6 | C-K1 to C-K3 compared arms of different strength | Applied: matched C-F0_m, C-RATIO_m, C-SC_m; k = 1 versions reported (4.3, 9.1). |
| R2.7 | A-K4 lacked rivals, a metric, a reference and the decisive control | Applied in part: NET_mixed, A-ORACLE-REGION, A-KQ, A-KMEAN at its own margin, A-CELL and per-cell A-SC in the rival set, the A-KPERM condition (4.1, 6.1, 7.2). Rejected: the local cap (a cap where A-KERN is on) as a rival: it reuses A-KERN's own decisions, so it compares actions, not localizers; it is reported as A-KERN-CAP. |
| R2.8 | Validity checks far from the hardest case; A-V2's target | Applied: boundary nulls, the rivals' size table; A-V2's target declared (the 90%-quantile gap judged through A-V1, because the margin is in that unit; the n-matched gap reported with a wording row) (6.4, 7.2). |
| R2.9 | C had no shuffled-outcome control | Applied: C-SHUF_m and C-K6 (4.3, 7.4). |
| R2.10 | C's weights dropped the per-state factor | Applied (with R1.5). |
| R2.11 | A-K3's E-NET for A-INS | Applied (with R1.12). |
| R2.12 | B's primary gate was the Monte Carlo one | Applied on the main generator: the exact gate is primary, the shipped gate reported (4.2). Not in module W: with unequal weights no exact form exists, so the shipped weighted gate stays primary there (10). |
| R2.13 | B-K1 can fail for too few tries | Superseded by R1.8: the permutation test does not depend on certifying. Both certification rates are reported. |
| R2.14 | The margin depends on the action and the critic | Applied: m_hinge and A-PERFECT_hinge; S-host ladder readout (3.6). |
| R2.15 | B-GROUP's offline tries were the recorder's commands | Applied: tries conditioned on the host's command at the same state; unconditioned reported (3.2). |
| R2.16 | Clone-fallback numbers mis-cited | Applied: deploy_tables.md:17, 33, 79, 95 (4.1, 12). |
| R2.17 | Module W's mark failed BCA when a rival was unsafe, and was not at equal harm | Applied (10). |
| R2.18 | C-ORACLE changed stage 1's oracle; the value-error oracle was missing | Applied in part: relabeled; stage 1's one-step form (C-ORACLE-S1) and long-run form (C-ORACLE-V) added actor-only; K3 uses the largest (4.3, 7.1). Changed: C-ORACLE-V uses stage 1's own `Q1 - Q^c_250` (`harness.py:559`), not `Q1 - Q^pi` inside co-training, which would need a 1,000-step rollout per batch row per update for a moving actor. |

**R3, feasibility and code**

| # | Finding | Status and where |
|---|---|---|
| R3.1 | C-ORACLE needed the simulator inside training | Applied: `dynamics_train.py` as a declared exception with a cross-environment test, x64, S3 exemption; the actor-only alternative also run (3.1, 4.3). |
| R3.2 | The card estimate was low; the cap would trip S10 | Applied: cost rule, stage-0 card benchmark, caps re-derived (9.2). The suggested trimming taken in part: 3-rung ladders and C-RAND matched on the pilot only; C-EVEN stays matched, since it is the plain alpha retune you rejected; a trimmed scope is offered (D14). |
| R3.3 | No single branch held the code | Applied: base branch and test references (11.1). |
| R3.4 | Reused host modules refuse without the four-learner approval | Applied: the reuse and not-reused lists, the gate-call test, own entry points (11.1, 11.2). |
| R3.5 | This container cannot train Unifloral | Applied (3.1, 9.2, D11). |
| R3.6 | The stage order read returns before the lock | Applied (9.3). |
| R3.7 | Fresh anchors, mixed points and NET_mixed undefined | Applied (with R1.16, R2.7); NET_mixed is a sum, as NET is (6.1). |
| R3.8 | Online residuals need the critic across the split | Applied: hand-off protocol (5.4). |
| R3.9 | The weights' form and the smoothing law's clip atoms | Applied: the two ratios agree once the state law is fixed; the kept-row law is chosen because the band is used on training batches (section 1). The inner clip is dropped from the declared law. The density is computed in the training environment from handed-off inputs, with a cross-environment test (4.3). |
| R3.10 | Evaluation cost missed staged-B replicates; runs missing | Applied (9.1, 9.2; B-RUNG staged capped at 2 families). |
| R3.11 | Bytecode in `experiments/` would block the hosts | Applied: S11, a separate clone, `sys.dont_write_bytecode` (7.1, 11.1). |
| R3.12 | The laptop card is shared with the hosts' restart | Applied (9.2, D11). |
| R3.13 | Keys with a zero last word collide | Applied (3.5). |
| R3.14 | The clone needs its own loader and process | Applied (3.1). |
| R3.15 | Mixed-world tries per cell are thin | Applied: expected counts, a minimum of 10 per cell, replicates split by physics (3.3, 4.1). |
| R3.16 | Stage 0 was under-budgeted | Applied (9.2). |
| R3.17 | Citation slip on the Q1 line | Applied (3.1). |

**R4, statistics**

| # | Finding | Status and where |
|---|---|---|
| R4.1 | A realized random coin fails a perfect switch 21% of the time | Applied: exact expectation (7.2). |
| R4.2 | Validity checks could not fail; every-seed-world marks fail exact rules | Applied in part: judged boundary nulls and pooled marks (6.4, 7.2, 7.3); ladder and nulls in the confirmatory stage too (3.3). A-V1 shifts the group to the margin (R2.8's form) rather than re-dealing pooled scores, because the shift keeps the group's own shape and clustering and gives the exact identity with coverage; the re-dealt null is reported. B-V1 shifts the tries to `1.1 alpha_B` rather than fixing tau at the union's 88% quantile, because a fixed tau would hide the threshold's randomness; the fixed form is reported. |
| R4.3 | The verdict did not separate BCA from plain rules | Applied (with R2.1). |
| R4.4 | B's release had no power analysis | Applied: discriminability step, staged N slot, "underpowered by design" (8.5, 7.3). |
| R4.5 | "L > 0" marks beside Holm; A-K4 outside every family | Applied: Holm-adjusted p in the marks; A-K4 in A's machinery family (7, 8.3). |
| R4.6 | One seed spread for every claim; observed-power size rule | Applied (8.4). |
| R4.7 | C-R2's non-inferiority was underpowered | Applied option (ii), merged with R2.3: replaced by the domination check (7.5). |
| R4.8 | A-K3 undefined | Applied (with R1.12). |
| R4.9 | B-K1 could not pass with few tries; mixed sources; unnamed cell | Applied (with R1.8): cell named, both rates over the replicate sets (7.3). |
| R4.10 | B-V2's event, and steps pooled across seeds | Applied (with R1.4): per seed, two-stage bootstrap (6.4, 7.3). |
| R4.11 | tau not fixed across replicates | Applied in part: the per-replicate threshold is primary (with R1.3) because the deployed procedure computes tau from data; the fixed-tau form is reported; staged tau from separate fallback episodes. |
| R4.12 | The freeze contradicted the two-stage plan | Applied (9.4). |
| R4.13 | A-V2 judged against a quantile L does not target | Applied in part (with R2.8): the functional declared; the 90%-quantile gap judged, the n-matched gap reported. |
| R4.14 | Stream layouts collide; replicate and decision indices missing | Applied (3.5). |
| R4.15 | The seed guard was a deny-list kept in the scratchpad | Applied (3.4). |
| R4.16 | Safety marks and matching inconsistent | Matching applied (1%, at most 5 rounds, step 2 PRESPEC.md:181-182). One safety form rejected: B's safety is its own primary promise and needs a bound, while A's and C's side condition is step 2's G2 copied verbatim; the sd = 0 rule is stated (7.3, 8.1). |
| R4.17 | Undefined or overloaded quantities | Applied: NET_mixed, L_gap and LB, the pooled-best rule, A-RAND's pooled rate, percentage points (sections 1, 6, 7). |

---

## 14. Decisions for the owner

| # | Decision | Recommended |
|---|---|---|
| D1 | Generator for the primary test | the main study's (only one with a no-stall twin, the gap and every earlier A and C result) |
| D2 | Confirmatory seeds | main 25-29, with 30-34 by the size rule (8.4) |
| D3 | Margin source | the break-even rules on the pilot (3.6); alternatives: a fresh exploratory block 20-24 (the pilot's A and B results are then not fitted to their own margin), or the record's 0.10 (no meaning on a new host; not recommended) |
| D4 | Module W | include, medium-expert: the only place for the weights' credit and for C's band past 80%; about 14 card-hours per 5 seeds (17 with the allowance) |
| D5 | Evidence pool for A and B | calibration and evaluation splits; the calibration split alone reported |
| D6 | Where Q1 is read in C's objective | at `pi(x)` (k = 0 reproduces the host); Q1 at the smoothed command as information |
| D7 | C's protocol | co-trained as primary; actor-only arms as references |
| D8 | Mixed worlds | train hosts on the spliced logs (2 runs per seed); alternative: drop A-K4 |
| D9 | Staged B | 50 fallback reference episodes, then a batch decision at N = 100, raised to 150-300 by the pilot's discriminability step; H = 1,000 |
| D10 | alpha_B | 0.1 (arm B's 0.5 reported only) |
| D11 | Compute site and timing | the laptop card for training, after the hosts' version 2 stages or under a time-share you set; evaluation never overlapping card training (or in the cloud scoring environment); the CPU route only if a cloud training environment is built, on its own budget line |
| D12 | A pilot kill rule fires | that option's test ends at the pilot, with its write-up |
| D13 | Near-80 version 8 | it runs on main seeds 10-14; if it reports first, A here is partly a replication with new rivals on a Unifloral host |
| D14 | Scope of C's matched controls on the main recorder | full (recommended); trimmed (C-RATIO_m and C-SHUF_m at k = 1, reported) saves about 4 card-hours per 5 seeds, but then C-K2 is judged only in module W and C's outcome verdict is unavailable on the main recorder |
| D15 | Budget caps | card route: pilot 26 card-hours + 50 CPU-hours; confirmatory 24 + 50 per 5 seeds; module W 18 + 18 per 5 seeds; or the CPU route's 260 / 240 / 155 CPU-hours |
| D16 | Stage 0 | booked separately, about 3-5 card-hours and 20-25 CPU-hours |
