# BCA experiment log

This file records every experiment in one place. Each entry says why the experiment was run, what we expected,
what happened, what it means, and what comes next. Update it whenever an experiment starts, finishes or changes
plan. Details live in the linked write-ups; this file stays short.

Last updated: 2026-10-14.

## The big picture

- **BCA** (this repo) measures how wrong the critic is with a calibrated band. It then uses that band to scale one
  term of the host algorithm's loss; this multiplier is called the "dose".
- **The calibration** is the published WBCP (Lou & Luo, arXiv:2604.06464v3). BCA runs it with uniform weights, which
  is exactly BQ‑CP.
- **The overall question:** does BCA measure the critic error that matters to the actor, and does it use that
  information usefully?
- **How results are judged.** A calibration bank fails when its threshold misses more than 10% of test rows. A valid
  rule fails in at most 5% of banks.
- **Scope of all evidence so far:**
  - one frozen critic per host and dataset, trained for 100k updates with one seed;
  - logged actions only;
  - no training runs.

## Status at a glance

| # | Experiment | Status | One-line result |
|---|---|---|---|
| 1 | WBCP implementation | Done | The test atom fixed an overconfident radius |
| 2 | Paper reproduction | Done, one gap open | Table 1 matches; Table 2 is about 2 points high at n = 250 |
| 3 | Shift on D4RL scores (hopper) | Done | Weights fix most of the shift problem |
| 4 | WBCP's excess under strong shift | Done | A finite-sample effect: explained, not fixed |
| 5 | Estimated vs exact weights | Provisional | Weights are only as good as their classifier |
| 6 | Bank dependence | Done; rule change pending | Whole-episode banks fail 26–48%; thin banks hold |
| 7 | Host matrix (4 hosts × 7 datasets) | Done, all four hosts | K must be set per host × dataset |
| 8 | Critic health | Done | 7 of 28 critics flagged |
| 9 | Critic alignment (TD3+BC) | Done; switch on hold | A min band misses Q1's error in 8–82% of banks |
| 10 | Signal study, steps 1–3 | Done, audited | The dose acts as near-uniform extra BC |
| 11 | Signal study, step 4 (placement) | Main run computing (started 2026-10-02) | G2 moved to poor data (a pilot-informed amendment) |
| 12 | Signal study, steps 5–6 | Not started | — |
| 13 | Full D4RL training (return) | Not started; needs permission | No WBCP-era return evidence yet |
| 14 | Toy-world audit and the Gapped LQ world | Done: rejected twice (original gate, then Amendment A1 on fresh seeds) | The critic is wrong everywhere, so the loss is not where data are thin |
| 15 | D4RL rollout truth (option C) and the action-gap protocol | Built; real runs need data + checkpoints | Tested on synthetic data only |
| 16 | ThrusterChain: a Gymnasium environment with a controlled distribution shift | Stopped before BCA (no room); Stage 1 passed; Stage 2 weighted detection passed, policy test running | The shift costs 99 points in strong current; pull-toward-data caution cannot reach it, a value penalty at the actor's command can |
| 17 | ThrusterChain known-shift weighting test (spec version 5, five dataset qualities) | Done: claims 1–2 (coverage) hold; no quality qualified for BCA training; exploratory BCA arms run | Weighting fixes coverage under the known shift (83–87% → 91–95%). Weighting the band for the actor's own commands, with "no value credit" where the data lack the command, raises return under the strong shift by 2.0 points (medium-expert) and 5.8 (expert), every seed, with no cost without the shift |

## Experiments

### 1. WBCP implementation (done)
- **Why:** replace BCA's archived radius with the published method.
- **Expected:** an exact implementation of the paper's Algorithm 1.
- **Found:**
  - Uniform-weight WBCP (= BQ‑CP) with a test atom is now deployed.
  - The archived radius had no test atom. Its "95%" credibility was really 88.8–94.3%, and its no-shift failure
    rate was 9.7 / 6.8 / 5.6%, against WBCP's 3.4 / 3.6 / 4.4%.
- **Means:** the deployed threshold now has its stated coverage when test rows come from the same distribution as
  the bank.
- **Where:** calibration/wbcp.py; ALGORITHMS.md; experiments/wbcp/README.md.

### 2. Paper reproduction, Tables 1–2 (done, one gap open)
- **Why:** check our code before trusting any D4RL number.
- **Expected:** results inside the paper's intervals. Table 2 was pre-registered in runs/wbcp_paper/expectations.md.
- **Found:**
  - Table 1 under shift matches: WBCP fails 5.1% against the paper's 5.0%, and every rule lands within about
    2 points.
  - Table 2 at n = 250 does not: WBCP fails 10.2% against 7.9%. About 1.4 of the 2.3-point gap is unexplained.
  - Two of the paper's published numbers do not follow from its own definitions.
- **Means:** the implementation is right wherever it can be checked. The Table 2 gap is still open.
- **Where:** experiments/wbcp/reproduce_table1.py and reproduce_table2.py; results.ipynb §9.

### 3. Distribution shift on D4RL scores (done)
- **Why:** test the rules on real critic scores, under shifts we impose and therefore know exactly.
- **Expected:** shift-blind rules fail under shift, and WBCP stays near 5%.
- **Found:**
  - With no shift, uniform BCA fails 4.9% and WBCP 4.8%.
  - Under the density tilt at γ = 1, uniform BCA fails 89.5% and WBCP 7.3% (7.5% with exact weights).
  - At the strongest tilts WBCP partly abstains.
- **Means:** weighting fixes most of the shift problem. The scope is hopper with independent-row banks only.
- **Where:** runs/wbcp_bench; experiments/wbcp/README.md; results.ipynb §5.

### 4. Why WBCP still exceeds 5% under strong shift (done)
- **Why:** explain the 7–8% excess seen in entry 3.
- **Expected:** a finite-sample effect (pre-registered as Change 11). Verdicts: A mostly met, B met, C partly met.
- **Found:**
  - Heavy weights, 33–39× the mean, sit on the misses.
  - With exact weights, failure falls from 7.6% to 6.0% as the bank grows from 1,103 to 8,824 rows. It stays above
    5%.
  - On other hosts it reaches 14.2% (CQL halfcheetah) and 22.8% (ReBRAC pen-expert) at γ = 0.5.
- **Means:** the excess is explained but not fixed. The paper's frequentist bound guarantees nothing at our sizes:
  its slack is 0.32 even with no shift.
- **Where:** experiments/wbcp/DEPENDENCE.md, Change 11.

### 5. Estimated vs exact weights (provisional)
- **Why:** a real deployment only has estimated weights.
- **Found:**
  - A well-specified classifier is about as good as exact weights: 7.3% vs 7.5%.
  - A misspecified one fails 16.6–36.4% under the density and state tilts.
  - When the score is not pivotal, estimated weights lose coverage even with no shift. Walker2d fails 14.9% at
    n = 8,192; ten times more fitting data brings it down to 5.7%.
- **Means:** weights are only as good as their classifier and its data. BCA does not deploy weights.
- **Next:** no minimum fitting-sample size has been established yet.

### 6. Bank dependence, Changes 1–12 (done; rule change pending)
- **Why:** rows from the same episode are correlated, but the posterior assumes independent rows.
- **Expected:** the design-effect model predicts the failure rates. Change 4 was "largely" met, within about ±1–2
  points.
- **Found:**
  - Whole-episode banks fail 26–48%.
  - Thinned banks hold on hopper (5.2% at K = 5).
  - Within-episode correlation ρ runs from 0.016 (hopper) to 0.120 (walker2d).
  - At K = 5, walker2d fails 7.5%.
  - The configured "raise K" banks fail 17.7% and 23.4% on walker2d.
- **Means:** K must come from the measured ρ, and per host as well (entry 7). The ρ-based rule is not implemented
  yet.
- **Where:** experiments/wbcp/DEPENDENCE.md; calibration/bank.py; calibration/dependence_evidence.json.

### 7. Host matrix: 4 hosts × 7 datasets (done, all four hosts)
- **Why:** the dependence study (entry 6) used TD3+BC's critic only.
- **Expected:** pre-registered per host: Stage A (ρ ranks the datasets as TD3+BC's does, Spearman ≥ 0.7) and Stage B
  (failure rate for each bank design).
- **Found:**
  - Stage B predictions met on healthy critics: CQL 46 of 48, ReBRAC 91 of 96, IQL 92 of 98, TD3+BC 77 of 80.
  - The misses are banks drawn from few episodes. TD3+BC's Stage A item 3 forecast their direction and was met:
    halfcheetah's whole-episode banks came in lower than predicted (33.1% vs 39.0%) and pen-human's higher
    (36.7% vs 31.0%, a flagged critic). Its item 1 (within 0.7 points of Change 12) holds on every configured and
    whole-episode bank checked so far; a full design-by-design scoring is still to do.
  - Independent-row banks hold everywhere (CQL and ReBRAC: 3.8–5.0%).
  - Configured banks hold near 5% where ρ is low. They fail on walker2d (10.5–18.3% across hosts) and pen-human
    (33.3–52.4%).
  - The Stage A ranking forecast missed on every host (0.57–0.63), and the host alone moves ρ by up to 8×.
- **Means:** K must be chosen per host × dataset.
- **Where:** experiments/wbcp/host_matrix.py; experiments/wbcp/results_{cql,rebrac,iql,td3_bc}.ipynb;
  runs/wbcp_hosts/*/expectations.md.

### 8. Critic health (done)
- **Found:** 7 of 28 critics are flagged: CQL maze2d and its 3 pen pools, ReBRAC maze2d, and TD3+BC pen-human and
  pen-cloned. The thresholds were set post hoc.
- **Means:** flagged pools give exact numbers for their scores, but they are kept out of conclusions.

### 9. Critic alignment, TD3+BC min vs Q1 (done; switch on hold)
- **Why:** TD3+BC's actor climbs Q1, while BCA calibrates min(Q1, Q2).
- **Expected:** the pre-registration was partly wrong. Items 1 and 5b were met; items 2, 4 and 5c partly; items 3
  and 5a not.
- **Found:**
  - A min band fails for Q1 in 8.2–70.2% of banks with σ fit online, or 10.5–82.4% with σ refit (signal study
    step 2).
  - A Q1 band fails 4.0–5.9%, at 2–10% more width.
- **Means:** TD3+BC's deployed band does not certify the error its actor uses. The pre-registered rule calls for the
  Q1 switch; it is on hold until step 4.
- **Where:** experiments/wbcp/CRITIC_ALIGNMENT.md. The pre-registration is in the BCA-critic-alignment worktree.

### 10. Signal study, steps 1–3 (done; audited twice)
- **Why:** the user's six-step plan of 2026-10-01: does BCA measure, and usefully apply, the error that matters?
- **Step 1, a trustworthy baseline:**
  - Random remainder trim added.
  - An evidence registry replaces `dependence_validated`.
  - Hashes verified.
  - Verdict: partly met.
- **Step 2, which error signal:**
  - The min band misses Q1's error in 10–82% of banks; the q1 band is valid.
  - maxabs is the widest and was dropped by judgment.
  - Forecast item 2 missed on magnitude.
- **Step 3, the known-MDP (LQ) harness:**
  - The dose is about 1.3 everywhere (row SD 0.017–0.050 on D4RL).
  - BCA is worse than both same-mean controls in 121 of 216 cells and better in 60. Forecast E5 missed.
  - Post hoc, the oracle's advantage was mostly strength: its BC gradient was 2.25–3.06× the constant dose's, and
    a strength-only model correlates 0.85 with BCA's difference from the constant dose.
  - A bias both heads share is invisible to any residual.
- **Means:** through today's hook, a valid band reaches the actor almost as uniform extra BC.
- **Where:** experiments/signal/SIGNAL_STUDY.md; runs/wbcp_signal.

### 11. Signal study, step 4: where to apply the signal (main run computing)
- **Why:** test whether any placement of the signal beats plain extra BC. The three placements are BC amplification,
  actor-Q trust and a critic-side bound. They are compared at matched policy displacement, against shuffled,
  stratified, reversed and nuisance controls.
- **Expected:** pre-registered 2026-10-02. The forecast is sceptical: no placement is expected to show real targeting
  on the natural misleading cases.
- **Pilot results (replicate 99, excluded from analysis; Addendum A part 2, independently audited):**
  - All 1,926 built-in checks pass.
  - Met: the cone's shape in X-CS (it misses in X-EP), the fixed-point identities, fixed-point definiteness, and the
    BC-pull cosines.
  - Reachability is near complete.
  - **The power check missed.**
    - Block C's oracle positive control does not beat its shuffled control with expert data: pooled C = −0.004 (P1L)
      and −0.033 (P2L), against a forecast of ≥ 0.03.
    - The pre-registered retune (doubling κ) makes the reference fixed point undefined, so it cannot be scored. The
      cone fails too, even after its retune.
    - With poor data the same control works clearly: pooled C = 0.166 (P1L) and 0.082 (P2L).
  - **Timing:** the pilot ran under heavy load. A clean re-run is 3.8–5.6× faster, which puts the full main run at
    about 18 process-hours (about 27 CPU-h), against the earlier estimate of 15–20 CPU-h.
  - The addendum was appended 2026-10-02T11:28:59Z; the pre-registration's sha256 is now bcbeff47….
- **Means:** as registered, the decisive gate G2 (block C, expert data) will probably fail, and the pre-registration
  then says "stop and redesign". The pilot exists to catch exactly this before the main run.
- **Decided by the user** (Addendum A part 3, appended 2026-10-02T11:37:12Z, before any main-run J; the
  pre-registration's sha256 is now c86ed140…):
  - **Amendment P, informed by the pilot:** G2 reads block C's poor-data cells. The expert version is reported as
    G2_expert, descriptive only. G2 now certifies detection only where BC is harmful. The scorecard's sha256 is
    625af396…, and its 24 tests pass.
  - **Amendment Q:** the cone's κ set is {2, 3}. Stage 0 was rerun (sha256 5ff59adb…); every non-cone row and set is
    identical.
  - **Amendment R:** the full main run is approved. It is 120 processes, 12 at a time, about 2 h, output in
    runs/wbcp_signal/step4/main. It started 11:37Z.
- **Next:** J (score_values), the frozen scorecard, the s = 0.47 robustness run, then an independent audit of the
  write-up.
- **Where:** experiments/signal/STEP4_DESIGN.md; runs/wbcp_signal/step4/expectations.md (Addendum A part 2);
  experiments/signal/pilot_report.py; runs/wbcp_signal/step4/pilot_report.{json,md}.

### 12. Signal study, steps 5–6 (not started)
- **Step 5:** coverage at the actor's own actions, and reused vs fresh banks.
- **Step 6:** ensembles, with TD3-BC-N as a baseline.

### 13. Full D4RL training: return and regret (not started; needs permission)
- **Status:** no training run with WBCP calibration exists yet.
- **Only return evidence:** one archived-radius pair (TD3+BC walker2d, one seed, 77.2 → 82.9). It is descriptive
  only.

### 14. Toy-world audit and the Gapped LQ world (done: rejected by the original gate and by Amendment A1)
- **Why:**
  - The 2026-10-09/10 audit found that the 3-state LQ toy world could not test BCA. It has no thin data, its
    critic errors are planted formulas, its policy is 6 numbers, and the shift is tiny.
  - Even caution recovered 97–99.9% of the possible improvement, and perfect placement added −1.6 to +0.04 points.
  - The environment card is the artifact "LQ Toy World".
  - The user approved a replacement (option A): 10 states, 4 actuators and 5 contexts. Two rare contexts log
    actuators 3–4 only thinly, and in one of them (T_wrong) those actuators are reversed. A noisy decoy context is
    added. The critic is learned with td3_bc_update, and truth is exact per context.
- **Expected:** the gate, with 57 pass marks, was pre-registered and hashed before any training (sha256
  255369889b…). Stop rule: if step 5, 6 or 9 fails, try the next retune; after the last, reject the world.
- **Found** (lean path, seeds 0–4; 70 critic runs):
  - **Steps 1–4 pass.** Every build check passes, Stage 0 passes (after moving 4 free scales, closed form only),
    R0 state richness passes (participation ratio ≥ 8.96 of 10) and R1 thin data passes.
  - **Step 6 (R3) fails at alpha 2.5, 5 and 10.** The base world passes 0/1/0 of 5 seeds and the thin_share retune
    3 of 5 at its best, with 4 needed.
    - The failing mark: at least 60% of the host's shortfall against EXACT must sit in the thin contexts. It is
      0.40–0.60 (base) and 0.56–0.74 (retune).
    - The cost, the EXACT gap and the rich-NS marks pass at alpha 10.
  - **Retune 2 (wrong_gain) fails Stage 0** (the transplant is unstable, √γ·ρ = 1.0023). The world is rejected,
    and BCA is not run in it.
  - **R2 (information):** the learned critics miss the ≤ 5% accuracy mark by a wide margin (0.46–1.15). Most of it is
    a level offset, consistent with clipped double-Q underestimation at γ = 0.98. The co-trained critic's steering
    error is not concentrated in T_wrong (1.0–1.6× the rich contexts'); FQE's is (3.4–6.2×).
  - **Room to win (information only; base world, alpha 10; not counted by the gate):**
    - Caution placed exactly in T_wrong beats even caution at matched strength by +0.060 NS (p = 0.0001, 5 of 5
      seeds). Caution in both thin contexts beats it by +0.045, and best against best by +0.088.
    - Row-by-row error placement (OR-err, OR-D) gains only +0.016 to +0.024.
    - Within BCA's dose range [1, 1.5) the oracle cannot reach the matched strength.
- **Means:**
  - This world does what the toy world could not. Its data is thin, its critic is learned, the errors cost return
    in the predicted direction, and placing caution by situation clearly wins.
  - But its own gate rejects it: the learned critic is also wrong where data is rich, so the loss is not
    concentrated in the thin contexts.
  - The information-only result says the room lies in placing caution by situation (context), not in per-row error
    magnitude, the kind of signal BCA's band gives.
- **Amendment A1 (2026-10-12, post hoc, labelled):** the user re-registered four rules after seeing seeds 0–4
  (pre-registration 8528a613…): R2 gated on the fixed-policy critic's error concentration only; R3's thin-shortfall
  mark at 1.5 × the thin contexts' share of episodes; matching up to 10 rounds; fresh seeds 5–9.
  - **Found:** the world fails again in both allowed versions. R3 fails at every alpha (base 0/3/2 of 5 seeds at alpha
    2.5/5/10; thin_share 0/0/0). The thin share of the shortfall is 0.33–0.55 against 0.45 (base) and 0.51–0.71
    against 0.75 (thin_share). R2 fails too: the fixed-policy critic's error is 2.8–8.1× higher in the reversed
    context, but the thin contexts hold only 0.38–0.78 of the squared error (3 and 2 of 5 seeds pass). The co-trained
    critic's error is not concentrated at all (1.0–1.5×), and both critics are inaccurate where data are plentiful
    (the recommended action is off by 1.8–4.8 units against recording noise 0.3).
  - **Means:** the rejection is not a matter of where the marks were set. The learned critic is wrong almost
    everywhere in this world, so it cannot test whether caution placed where data are missing helps. Replaced by
    entry 16.
- **Where:** experiments/signal/gapped_lq.py, glq_train.py, glq_acceptance.py, GAPPED_LQ.md;
  outputs/glq/lean-v1/README.md (original verdict, R3, R2, R4); outputs/glq/lean-a1/README.md (Amendment A1).

### 15. D4RL rollout truth, option C (built; real runs need data and checkpoints)
- **Why:** measure true Q^π of real TD3+BC hosts (Hopper, Walker2d) by restoring logged MuJoCo states and rolling out.
  This shows whether real critic errors sit where data is thin.
- **Built:** experiments/truth/
  - state restore with a one-step reproduction gate;
  - a batched mujoco_py rollout labeller;
  - pools L, Pi and T, and R1–R7 stages;
  - extraction on frozen critics;
  - a pre-registration hash gate.
  53 tests pass. The synthetic end-to-end run uses simulator-made D4RL-format data only, and none of it is a finding.
- **Blocked in the cloud container:** the D4RL files (rail.eecs.berkeley.edu is blocked by the network policy) and
  the trained host checkpoints (on the local machine). The first real step, truth labels for the existing Hopper
  checkpoint seed 202609171 (R1, R2, R5-world, R6), costs about 1.5 process-hours. The commands are in
  experiments/truth/README.md.
- **Action-gap protocol (2026-10-12):** the user's D4RL test, "remove a commonly taken action and reintroduce it
  during evaluation". The densest 15% of training actions (the smallest ball in action space holding 15% of rows) is
  removed from training; gapped hosts are compared with full-data hosts of the same seed, then evaluated where the
  action is allowed again. Built in experiments/truth/gap.py, gap_train.py and gap_acceptance.py; adversarially
  reviewed (3 major and 8 minor findings fixed); 85 truth tests pass on synthetic data. Hopper-medium-v2 first, on
  the user's machine.

### 16. ThrusterChain: a Gymnasium environment with a controlled distribution shift (done: stopped before BCA, no room)
- **Why:** the user asked for a real Gymnasium environment in which a distribution shift is created on purpose and
  measured exactly, to show whether BCA finds where the critic's error causes harm and whether its caution helps the
  learned policy in situations the data did not cover, while it still improves on the recording.
- **Design** (judge panel of three designs; experiments/shift/SHIFT_ENV.md once built): five floats in a channel,
  pushed by a current and waves and held by four thrusters. Past 80% throttle a thruster stalls. The recording
  controller's most common command is a full-throttle command, so the D4RL 15% rule removes it, and the training data
  then show nothing past 80%. Tested in recorded conditions and in three shifted situations (stronger current, bigger
  waves, bigger starting pushes). Controls: nothing removed, the same number of rows removed at random, and a
  no-stall comparison world where the removed data hide nothing. Truth is exact by deterministic simulation.
- **Plan:** harm checks before BCA: the critic is wrong where data are missing (gapped vs full-data critic, same
  seed), acting on it costs return, and caution placed by the exact harm can help. If no harm appears at TD3+BC's
  alpha 2.5, alpha 5 and then 10 are tried, the same for every method and without looking at BCA. Then stop and ask
  the user before any BCA run.
- **Expected:** honest forecast from the design: the harm checks pass with about 25–30% probability; the most likely
  outcome is a stop before BCA, because TD3+BC stays close to the recorded commands.
- **Ran** 2026-10-12 under pre-registration `68d9e8779e148b7b…` (commit bdb8c8a), base world, gate seeds 0–4. No BCA
  arm, band fit or dose readout ran. Report: `outputs/shift/stop-report/README.md`.
- **Result:**
  - The critic is wrong where data are missing (H1, 6 of 6 marks). The gapped critic is 2.15–2.40× as wrong at the
    removed commands, 0.68–0.80× far from them. Random removal (0.96–1.21×) and another initialisation (0.89–1.09×)
    do not explain it, and the stall makes it 1.75–2.23× worse than the no-stall world. Pushing past 80% truly
    costs 0.33–0.35; the gapped critic overrates it by 0.33–0.36, so it sees no cost (full-log critic: within 0.06).
  - Acting on it costs return (H2). It failed at alpha 2.5 on one mark (pushes past 80% 1.46× as often, against
    1.5) and passed at alpha 5 on all five. In strong current the gapped critic's actor loses 98.9 points
    [96.2, 101.7] against the full-log critic's actor. Random removal explains 3.1 and the no-stall world 0.4. The
    loss sits at the past-80% steps. Elsewhere: 5.8 (recorded), 16.6 (big waves), 11.8 (big kicks).
  - Plain TD3+BC on the gapped log beats cloning by 27–47 points in three situations but loses 71 in strong current.
  - No room (H3). Caution placed by the exact harm, with factors in [1, 1.5), beats strength-matched even,
    amount-matched and shuffled caution by −1.9 to +1.1 points (mark: 3) in every situation. Factors up to 2 do no
    better. Only even ×3 recovers part of the strong-current loss (17 of 99 points), at a cost of 5–13 elsewhere.
    H4 refused, as designed.
- **Meaning:** the world is sound and the harm is real and large. But it lies in states outside the recorded range,
  where a stronger pull toward the recorded commands on training rows does not reach. The limit is in the mechanism,
  not in BCA's signal. BCA's detection was not tested.
- **The user's decision (2026-10-13): the stop is not accepted.** The question is now what must change for BCA to help
  with harm outside the data. Exploratory analysis, no BCA (`outputs/shift/beyond-the-data/PLAN.md`):
  - The harm is one wrong belief held at training states: in the full-throttle region the gapped critic rates full
    throttle +0.06 against 80% (truth −0.46). The actor learns it there (past 80% on 42% of region rows, full-log
    critic's actor 10%) and carries it into strong current.
  - BCA never reads its band where that error is. It reads at recorded rows, where the gapped critic is accurate
    (error 0.07). At the actor's own past-80% commands, the one-step error BCA calibrates averages −0.67 (typical
    size 0.105). Its dose pulls toward the recorded command, which cannot cancel the critic's false slope.
  - What must change, as a declared new method version: read the band at the actor's own command; act on the value
    the actor climbs (lower edge Q − k·U, k = 1 fixed); no 1.5 cap; "cannot certify" means the most caution, not
    none; let the gradient pass through the frozen scale network's command input; check the band widens past 80%
    before training, and otherwise use likelihood-ratio weights. Also a deployment check that reads the same band and
    clips into the recorded range.
  - Deployment diagnostic (no BCA): a distance-to-data check with a clip fallback recovers 57 of 99 points in strong
    current, 20 of them from targeting, at no cost in the recorded conditions. But clipping every command scores 91.0
    in strong current here, and costs 61.8 in the no-stall world. So in this test bed BCA can at best show that it
    flags unsupported commands. Claiming it identifies harm needs data that carry a trace of the harm.
  - Next: Stage 1, a room test of the lower-edge channel with an exact one-step oracle against matched untargeted
    controls, still without BCA. Stage 2 (BCA) needs the user's typed approval and a new labelled pre-registration.
- **Stage 1 result (2026-10-13; exploratory, no BCA; `outputs/shift/beyond-the-data/stage1/RESULTS.md`): all marks
  pass.** A penalty at the actor's own command, equal to the critic's exact extra overrating of that command over the
  recorded one, recovers 94 of the 99 strong-current points (−14.3 to 79.7). It beats matched even pull,
  push-less-everywhere and random penalty surfaces by 89–99 points (5 of 5 seeds), and improves the other situations
  by 8–20. The same information taken from another state recovers almost nothing (−0.3). In the no-stall world it
  costs 1.3 points, where a plain clip to the recorded range costs 30.9 (the clip scores 93.1 in the stalling
  world). It replicates on check 3's critic (84.7 against −2.8). The user asked for both claims, the flags claim
  first: the Stage 2 pre-registration is written and awaits the user's approval on their own machine; the trace test
  bed for the harm claim is running.
- **Results of 13 October (exploratory, seeds 0-4; `outputs/shift/beyond-the-data/results-2026-10-13/README.md`):**
  - Data-only rivals in the lower-edge slot: only the range hinge captures the room (84.6 of 98.9 points back in
    strong current, but 20.4 lost in the no-stall world); the classifier ratio wins back 36.8, the others under 10.
    The range hinge is Stage 2's rival.
  - Trace test bed (harm claim, seeds 0-2, shares 1-10% post hoc): no share passes. Harm stays above 50 points up to
    5% kept, but a data-only residual signal never separates the worlds (at most 0.08 against 0.2).
  - Deployment-check channel: open (a perfect detector beats random switching by 23.2 points in strong current).
  - **Stage 2 detection, uniform version: fails** direction (0.50 / 0.73) and size (0.0004): BCA's band barely
    changes past 80%. The pre-registered weighted version was declared.
  - **Stage 2 detection, weighted version: passes direction (1.00 / 1.00), size (1.46) and ranking (0.99 / 0.86)**,
    calibration 4.9%, but is 0.04 below the classifier ratio in strong current, so the claim rule is not met. Its
    width adds a term from the same classifier, so its detection mostly reflects that classifier.
  - Because a version passed direction and size, the policy test, deployment check and full host started on seeds
    0-4 (13 Oct, 13:56 UTC), as the pre-registration orders.
- **Known bug (reporting only):** `acceptance.report_cards` passes all four situations' anchors to `shift.shift_card`.
  The cards were recomputed outside the gate with that one change, labelled as such
  (`outputs/shift/stop-report/cards/`), and match the gated scores to 3e-14 points. The bug gets fixed only under a
  later pre-registration.

### 17. ThrusterChain known-shift weighting test (spec version 5; claims 1–2 hold; BCA arms exploratory)
- **Why:** the user's goal, verbatim: "verify that bca's mechanics are working and then show importance weighting
  empirical showing an improvement given that we know the test distribution. This is all controlled, we need to know
  every value, so we can simulate a distribution shift."
- **Design** (`experiments/shift_iw`, spec version 5, SHA-256 `a71af9bd…`, frozen with its sidecar; the user approved
  it in chat, "Yes, freeze and run"):
  - The training current follows a tilted density on [0.1, 1.1] (λ = −2). The tests use λ = −2 (none), 0 (mild) and
    +2 (strong), so the exact weight ω(c) = p_test(c)/p_train(c) is known.
  - The expert's 80% hold is cut from the training and bank rows (the action gap), so its exact ratio is infinite and
    weighted BCA must refuse those steps.
  - Five D4RL-style dataset qualities: random, medium-replay, medium, medium-expert and expert.
  - Two uses of the band: the capped pull (BCA's dose) and the lower edge (−λ·mean[Q1 − k·[U(s, π(s)) − U(s, a_i)]_+]
    + BC, k = 1).
  - Per-step weights through a tabulated R(t).
  - Gates before any BCA arm: room checks with the exact error, and "can weighting move return" predictions.
- **Ran** 2026-10-13 in WSL: data, hosts, room checks, band, coverage, report and summary, every stage rc = 0. Run
  folder: `runs/shift_iw_v5/` (not committed).
- **Result, pre-registered:**
  - Mechanics hold on all five qualities: no-shift coverage 91.2–92.5%, width tracks the exact critic error within
    current bands, and the dose follows the width.
  - Under the strong shift, unweighted coverage falls to 83.3–86.7% and exact-weighted per-step coverage restores
    91.0–94.6%. Estimated weights land within 2 points of exact. Weighted BCA refuses 100% of steps using the removed
    hold and 0% of the others.
  - One side check failed: on medium-expert, the unweighted band was not over-confident inside the gap.
  - No quality qualified for a BCA training arm. The capped pull's room check fails on the low qualities, where
    placed caution lowers return by 2.9–4.6 points. At the top two qualities weighting moves the dose only 0.04–0.05
    (predicted gain 0.4 return units against 1.8 needed). The lower edge's perfect penalty helps at the top (+8.5
    expert, +5.8 medium-expert), but return is already level in its strength, and BCA's width at the actor's commands
    is 1/160 to 1/30 of the true error and barely correlated with it.
- **Exploratory, at the user's request ("Run the arms anyway, labelled exploratory"):**
  - **BCA arms** (`iw_explore/RESULTS.md` in the work folder; runs in `runs/shift_iw_v5_explore/`). On the strong
    starts, weighting changed return by −26.4 to +0.1 return units (capped pull) and −1.1 to +2.7 (lower edge). No arm
    gained 1.8 with a lower bound above 0. The predictions held at the top qualities.
    - The capped pull costs 5–9 points against plain TD3+BC on the three lower qualities, because it pulls toward
      mostly random recorded commands; weighting makes that slightly worse.
    - Points, strong starts: plain TD3+BC 66.0 / 77.0 / 84.1 / 91.1 / 90.3; capped pull, weighted 57.1 / 69.1 / 77.7
      / 91.5 / 90.3; lower edge, weighted 67.1 / 77.8 / 82.2 / 92.2 / 90.6 (random → expert).
  - **Plain TD3+BC coverage** with its two critics' disagreement as its band (the user's choice; expectations written
    first, `iw_explore/PLAIN_TWIN_PREREG.md`). It covers 38.6–48.1% at every shift. Its band is 0.22–0.32 times BCA's
    width and no wider at the removed hold. All four expectations were met; on medium-expert its coverage rose 1.4
    points under the shift instead of dropping or staying flat.
  - **Data fact** (recomputed from the training files, data seeds 0–4): in the expert data no command of any thruster
    goes past 80%. In the other qualities, 3.25–10.11% of all training rows push thruster 1 upstream past 80%
    (medium-expert 3.25–3.53%, random 9.98–10.11%), and every one is a uniform random command. The critic therefore
    guesses past 80% with or without the hold removed (the page's "critic in action" section shows it). An earlier
    note quoted 3.6–9.9%, which does not reproduce exactly (`docs/thrusterchain/notes/p4_facts.md`, section 8).
- **Meaning:**
  - BCA's mechanics are verified, and weighting fixes coverage under a known shift.
  - Weighting did not raise return here. The band is calibrated on recorded commands. Rescaling it cannot create the
    width that is missing at commands the data never contain, which is where return is lost.
  - Next, if the user wants return gains: a band whose width grows at unsupported commands (for example read relative
    to the recorded command, as Stage 1's oracle was), then a new pre-registered test.
- **The user's request (2026-10-14):** "i wanr return gain as well. its good that the coverage aspect is working
  accordingly, but we need to identify the right channel for the penalty to actually impact policy performance".
- **Channel check (2026-10-14, exploratory, post hoc, no training;** expectations written first:
  `outputs/shift_iw/explore/channel/`).
  - **Which channel:** the room checks with exact information already answer it.
    - The lower edge (a penalty on the value at the actor's own command) gains +5.8 points (medium-expert) and
      +8.4 (expert) under the strong shift.
    - The capped pull gains at most 0.2 points even when placed perfectly.
    - On the three lower qualities neither channel has room: the perfect lower edge loses 0.8–1.1 points.
  - **Why BCA's lower edge did not move return:** the band is honest where it was calibrated, not where the lower
    edge reads it.
    - At the no-caution actor's own commands within 80%, it covers 95–96% at the top two qualities.
    - The actor pushes past 80% on 53–62% of hard-water rows. There the critic overrates its command by 3.4
      (medium-expert) and 1.7 (expert) per step, and the band covers 48% and 25% (strong law). It would have to be
      3.4× and 8.1× wider, but it widens only 15–25% past 80%.
    - On the three lower qualities the band is 10–18× wider than on expert data, and the actors rarely go past
      80%, so it covers 81–93% at their commands.
    - Expectations: E1 met; E2 and E3 partly met (they hold at the top two qualities only, and the weighting helped
      coverage more than predicted: +6 to +10 points).
  - **Why weighting by the current law cannot raise return:** it multiplies every row's band by a nearly constant
    factor (1.45× expert, 1.75× medium-expert). In the room checks, scaling caution by 1.5 moves return by
    0.0–0.1 points for the perfectly informed lower edge and 0.4–0.7 for the even pull. Weighting has to change where the band is wide, not only how wide it is.
  - **Consequence:** for weighting to reach return, the weights must describe where the band is read, which is the
    actor's commands under the test law. Both parts are known exactly in this world: the current law, and the
    recording controller's command density in closed form. Proposed to the user (decision 10).
- **Command-weighted band, check without training (2026-10-14; the user's go: "Yes, build and check it
  (Recommended)").** Pass marks were written first (`outputs/shift_iw/explore/channel/`).
  - **The band:** BCA's frozen band with one change, the weights. A (row, command) gets the exact ratio
    w(c) × π_h(a | s) × Z(s) / β(a | s):
    - π_h is the actor's command with BCA's own target-noise spread (0.2, clipped at 0.5);
    - β is the recording controller's command density, exact for each episode's random share;
    - Z is the share outside the removed ball.
  - **C1, met on all five qualities:** at draws from that test distribution, 94–100% of commands are covered or
    refused, by seed.
  - **C2, met at the top two qualities:** at the actor's own commands, 98–100% are covered or refused. Every
    past-80% command is refused, where version 5's band covered 25% and 48%.
  - **C3, missed:** within 80% the band also refuses 25% (expert) and 48% (medium-expert) of the actor's commands.
    15 and 9 points of that are commands inside the removed hold; the rest come from the small effective bank
    (3–16%). On the three lower qualities it refuses 68–98%, a blanket refusal.
  - **C4, missed:** version 5's refusal rule (the largest finite threshold, only 1.7–3.0× R_u) leaves the penalty
    too small and flat from 80% to 81%. Check (e) passes on 0 and 1 of 5 seeds.
  - **Post hoc, not a pass mark:** if a refused command gets no value credit over the recorded command, check (e)
    passes on 5 of 5 seeds at both qualities. The share of hard-water rows whose best throttle is past 80% falls
    from 80% to 1.8% (expert) and from 85% to 16.5% (medium-expert); the exact oracle reaches 34% and 52%.
- **Return test (2026-10-14; the user's approval: "Approve as drafted (Recommended)").** The pass marks were frozen
  first (`outputs/shift_iw/explore/channel/PASSMARKS_training_test.md`, SHA-256 `204f53ff…`). Setup: actor-only
  lower edge, version 5's protocol, medium-expert and expert, seeds 0–4, 30 runs on CPU (about 150 s each).
  - **Checks before training:** all passed. With the weights switched off the step equals version 5's bit for bit,
    and the in-graph density and refusals match the numpy check.
  - **Expert: T1–T3 met.** A beats version 5's unweighted lower edge under the strong shift by 78.6 return units
    (lower bound 72.8; 6.2 points; every seed +70 to +85), which is 73% of the exact oracle's gain. An untargeted
    control with the same amount of "no credit" gains 1.1. Without the shift A gains 16.3.
  - **Medium-expert: T1–T3 missed.** A loses 124.8 under the strong shift (9.8 points) and 35.6 without it.
    - Post hoc: refusals pile up in the strongest third of currents (62–68%) and on random-command episodes.
      59–65% of refused rows have a random recorded command, and "no credit" makes the actor copy them.
  - **Knowing the test current law adds nothing to return:** A minus B (training law) is +1.0 on expert (lower
    bound −0.3) and −15.1 on medium-expert. The gain comes from weighting for the actor's commands.
  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_training_test.md`.
- **Rerun with the refusal fallback fixed (2026-10-14; the user's go: "Fix the fallback, then rerun
  (Recommended)").** The pass marks were frozen first (`PASSMARKS_fallback_rerun.md`, SHA-256 `8b8f7a27…`).
  - **The change:** "no value credit" applies only where the exact ratio is infinite (the data never contain the
    command at that state and episode, or it is in the removed hold). A refusal because the bank is too small
    gets version 5's capped threshold. Checks before training passed (bit-equal step; refusal types match).
  - **U1–U4 met on both qualities.** Against version 5's unweighted lower edge under the strong shift:
    - medium-expert +25.0 return units (lower bound 22.6; 2.0 points; every seed +22.8 to +28.8);
    - expert +74.2 (lower bound 68.3; 5.8 points); that is 34% and 69% of the exact oracle's gain.
    An untargeted control of the same amount gains about 1. Without the shift return also rises (+4.5 and
    +15.6). Medium-expert's loss is gone (+149.8 over the first test).
  - **The current-law part of the weight adds nothing to return** (A2 minus B2: −0.2 and +0.5). The gain comes
    from weighting for the actor's commands, which here acts as a calibrated support check: it refuses where
    the recorded commands never go and certifies elsewhere, with no knee or threshold given to it.
  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_fallback_rerun.md`.
- **Estimated instead of exact weights (2026-10-14; the user's go: "Proceed with all three").** The pass marks
  were frozen first (`PASSMARKS_estimated_weights.md`, SHA-256 `4560769a…`).
  - **Weights:** the current-law part is version 5's logistic-regression estimate. The command part is a
    classifier separating recorded commands from the actor's command plus its spread, refitted every 4,000 steps.
  - **Refusal:** "no value credit" applies where the estimated odds exceed those of every recorded command, or the
    command is in the removed ball. Nothing exact about the recording process or the laws is used.
  - **Expert: V1, V2 and X1–X3 met; X4 missed.** +29.4 return units under the strong shift (lower bound 25.8,
    2.3 points, every seed +25 to +35), against +0.9 for an untargeted control. That is 40% of the exact
    weights' gain (the bar was half).
  - **Medium-expert: V1 and X3 met; V2, X1, X2 and X4 missed.** +1.5 (lower bound +0.2). The estimated rule finds
    only 5% of the past-80% commands the exact rule refuses, because those depend on the episode's random share,
    which the classifier cannot see.
  - **Meaning:** the estimated band stays valid (coverage held on both). The return gain depends on estimating
    where the data lack a command, which is easy on clean expert data and hard on mixed data.
  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_estimated_weights.md`.
- **Learning curves for every setup (2026-10-14; the user's request "im going to need to see learning curves as well
  comprehensively for all possible env setups" and go "Full set, ~3 h CPU").**
  - **Method:** every run of this test was retrained with checkpoints (actor-only every 1,000 of 20,000 steps; full
    TD3+BC every 5,000 of 100,000 updates), and every checkpoint was scored on the three start banks.
  - **Reproduction:** all 330 reruns with an earlier run end at its saved actor bit for bit. Version 5's runs
    reproduce only when pinned to one CPU, as they were trained.
  - **New runs:** the command-weighted arms (A2, B2, C2, A3, C3) on the three lower qualities, with expectations
    written first. Mostly met: within ±10 units except estimated weights on medium (−11.4); 3 of 6 controls beyond
    ±5.
  - **Shown:**
    - On expert, actor-only training against the critic lowers return (the cloned actor's 95.2 points on the strong
      test falls to 90.7). The command-weighted band prevents it and rises to 96.5.
    - The first test's medium-expert collapse comes after a peak at 3,000 steps.
    - Full TD3+BC on the lower qualities peaks at 5,000–15,000 updates and loses 4–17 points by 100,000; the
      capped pull loses 9–23.
  - Page: finding G (`docs/thrusterchain/`); code and data: `outputs/shift_iw/explore/curves/`.

## Decisions waiting on the user
1. Step 4 has its decisions (amendments P, Q, R). Next comes reading the main-run verdicts.
2. The TD3+BC Q1 switch: the pre-registered rule says now; it is on hold until step 4.
3. The K rule: K from measured ρ per host × dataset, or test the n/D posterior correction.
4. Importance weights: adopt them at all, and toward which target distribution?
5. Permissions: actor fine-tuning (4B-1), TD3+BC pools on the random datasets, and later full training (4B-2).
6. Provenance: commit each expectation file before its run.
7. Gapped LQ (entry 14): decided. The user amended the gate (A1); the world was rejected again on fresh seeds.
8. Option C and the action gap (entry 15): run on the local machine (hopper-medium-v2 first), or allow
   rail.eecs.berkeley.edu in the cloud environment and provide the checkpoints.
9. ThrusterChain (entry 16): the user did not accept the stop. Stage 1 of the plan (no BCA) runs first. Still to
   decide: the claim wanted ("flags reliance on unsupported commands" is testable now; "identifies the harm" needs a
   test bed whose data carry a trace of it). Stage 2 (BCA) needs typed approval.
10. Known-shift weighting (entry 17): the return gain is shown (rerun, U1–U4 met on medium-expert and expert).
   With estimated weights the gain holds on expert (+2.3 points) and vanishes on medium-expert. Learning curves
   for every setup are on the ThrusterChain page (finding G). Open: pushing; estimating support on mixed data.

## Deliverables
- **Notebooks:**
  - experiments/wbcp/results.ipynb;
  - results_{cql,rebrac,iql,td3_bc}.ipynb;
  - visualizations.ipynb: six figures of the theory in action, with explanations.
- **Advisor deck (50 slides):** https://claude.ai/artifact/3T2Pe8eKotkpNVp4Y7Gir2. Its source is in
  docs/decks/wbcp-essential-questions/.
- **Signal-study results page:** https://claude.ai/artifact/18TEwegNMBpTY9hfAxV69Q.
- **Commits:** cad0c61, 8d9ad4f and e3c4143 (not pushed).

## Known inconsistencies in the write-ups
- **min vs Q1 failure ranges:** there are two ranges, one per σ (fit online vs refit). Always say which.
- **The Q1 switch** reads as "now", "on hold" and "measurement signal" in different places. Keep these three
  decisions apart.
- **`dependence_validated`** is gone from the code, but DEPENDENCE.md and the host notebooks still describe it.
- **DEPENDENCE.md:**
  - its Summary and Limits sections predate Change 12 and the host matrices;
  - its "small excess" claim holds only on hopper.

## Change log
- **2026-09-29:** the WBCP rewrite began, replacing the archived radius.
- **2026-09-30:**
  - synthetic D4RL benchmark;
  - bank dependence study (Changes 1–11).
- **2026-10-01:**
  - Change 12, all seven datasets;
  - paper Tables 1–2 reproduced;
  - critic alignment;
  - CQL and ReBRAC host matrices;
  - the user's six-step signal plan, with steps 1–3 run and audited.
- **2026-10-02:**
  - step 4 pre-registered, Stage 0 run, pilot run;
  - IQL and TD3+BC host matrices done, completing all four hosts;
  - commits cad0c61, 8d9ad4f, e3c4143;
  - visualization notebook built;
  - advisor deck restructured to put the models first, add code snapshots, and lead through seven questions.
  - TD3+BC host matrix finished (all four hosts done);
  - step 4 pilot analysed, with retunes, a clean timing run and an audited Addendum A part 2. The expert-data positive
    control has no power, and the G2 decision goes to the user;
  - the user's decisions recorded as amendments P–R, and the step 4 main run launched (11:37Z);
  - the deck is being rebuilt to be visual-first, with charts and few words.
- **2026-10-11:**
  - options A (Gapped LQ) and C (D4RL rollout truth) built, tested (86 and 53 tests) and adversarially reviewed;
    the Gapped LQ gate was pre-registered (255369889b…);
  - the Gapped LQ lean gate ran: steps 1–4 pass, and R3 fails at every alpha in the base world and the thin_share
    retune. Retune 2 is unstable, so the world is rejected. The base world's information-only Tier A shows room to
    win for context-level caution (entry 14);
  - option C is blocked on D4RL data and checkpoints in the cloud (entry 15).
- **2026-10-12:**
  - the user re-registered the Gapped LQ gate (Amendment A1, post hoc, fresh seeds 5–9); the world is rejected again
    in both versions (entry 14);
  - the D4RL action-gap protocol built, reviewed and synthetic-tested (entry 15);
  - environment page for the Gapped LQ world published and rewritten in plain words;
  - ThrusterChain designed by a judge panel, built, reviewed and pre-registered (68d9e877…), then run to its stop:
    the harm checks pass at alpha 5 (99 points lost in strong current), but no situation has room for caution of
    BCA's kind, so the gate stopped before any BCA run (entry 16).
- **2026-10-13:**
  - the user did not accept the stop; exploratory analysis of what BCA needs to reach harm outside the data
    (mechanism, harm anatomy, deployment diagnostic, literature, judge panel of three designs): plan in
    `outputs/shift/beyond-the-data/PLAN.md` (entry 16);
  - the ThrusterChain page gained a task overview and Half-Cheetah-style component sections;
  - Stage 1 passed; rivals, trace sweep and deployment check finished; Stage 2 approved by the user, its uniform
    detection failed and its weighted detection passed direction, size and ranking; the policy test started
    (entry 16; `outputs/shift/beyond-the-data/results-2026-10-13/`);
  - the known-shift weighting test (spec version 5, five qualities) was frozen with the user's approval and run.
    Mechanics and weighted coverage hold; no quality qualified for BCA arms. The exploratory arms show no return gain
    from weighting. Plain TD3+BC's twin-critic band covers 39–48% (entry 17);
  - the ThrusterChain page went through versions 11–13: plain-language tooltips, specification sections first,
    Results rewritten for newcomers, and the "critic in action" section added. Version 13 answers the owner's 33
    page comments: one validated colour system, a five-step offline-RL pipeline strip tying every setting to its
    step, aligned symbol legends, two interactive charts, and the weighting test's plain TD3+BC coverage and
    exploratory BCA results. Page and sources: `docs/thrusterchain/`.
- **2026-10-14:** the user asked for a return gain from weighting and for the right channel. An exploratory check
  (no training) found that the lower edge is the channel with room, but BCA's band covers only 25–48% at the
  actor's past-80% commands at the top two qualities. The current-law weighting rescales the band nearly
  uniformly (1.45–1.75×), and in the room checks a rescale of that size moves return by at most 0.7 points,
  even for perfectly informed caution. Next: weights for the actor's commands under the test law (decision 10).
  With the user's go, that band was built and checked without training: it is honest where the lower edge reads
  it and refuses every past-80% command, but version 5's refusal rule makes the penalty too weak. A post hoc
  "no credit" rule removes most of the past-80% preference. The return test (approved, pass marks frozen first)
  met T1–T3 on expert data: +78.6 return units (6.2 points) under the strong shift, 73% of the exact oracle's
  gain, and nothing for an untargeted control of the same size. It missed them on medium-expert (−124.8), where
  refusals in strong currents make the actor copy random recorded commands. Knowing the test current law added
  nothing to return; the gain comes from weighting for the actor's commands.
  The user chose to fix the fallback: "no value credit" only where the data lack the command (infinite exact
  ratio), version 5's capped threshold where the bank is merely too small. The rerun met U1–U4 on both
  qualities: +25.0 return units (medium-expert) and +74.2 (expert) under the strong shift, with every seed
  positive, no gain for an untargeted control of the same size, and no cost without the shift.
  With estimated weights (pass marks frozen first): expert keeps a gain of +29.4 return units (2.3 points) under
  the strong shift, 40% of the exact weights' gain; medium-expert's gain vanishes (+1.5), because the classifier
  cannot see which episodes contain random commands. The user then asked for learning curves for every setup;
  every curve reruns the original training with checkpoints and checks it against the saved final actor.
  Version 5's runs reproduce only when pinned to one CPU, as they were trained.
  The ThrusterChain page, version 15 (merged onto another session's version 14, which answered the owner's critic-viewer
  and Setting 1 comments): the weighting test gains findings D-F (where the band is read, weighting for it, return),
  and its outcome, header line, timeline and navigation now report the return gain (`docs/thrusterchain/`).
  Learning curves for all 5 qualities × 3 tests: every run retrained with checkpoints (330 of 330 reproduce their
  saved actor bit for bit) and the command-weighted arms run on the three lower qualities. The ThrusterChain page,
  version 16, adds them as finding G.
