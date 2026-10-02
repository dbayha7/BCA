# BCA experiment log

This file records every experiment in one place. Each entry says why the experiment was run, what we expected,
what happened, what it means, and what comes next. Update it whenever an experiment starts, finishes or changes
plan. Details live in the linked write-ups; this file stays short.

Last updated: 2026-10-02.

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
| 11 | Signal study, step 4 (placement) | Pilot run; analysis next | Pre-registered; no outcomes yet |
| 12 | Signal study, steps 5–6 | Not started | — |
| 13 | Full D4RL training (return) | Not started; needs permission | No WBCP-era return evidence yet |

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

### 11. Signal study, step 4: where to apply the signal (pilot run; analysis next)
- **Why:** test whether any placement of the signal beats plain extra BC. The three placements are BC amplification,
  actor-Q trust and a critic-side bound. They are compared at matched policy displacement, against shuffled,
  stratified, reversed and nuisance controls.
- **Expected:** pre-registered 2026-10-02. The forecast is sceptical: no placement is expected to show real targeting
  on the natural misleading cases.
- **Progress:**
  - Stage 0 is done. Its boundary rule was corrected before any outcome existed.
  - The pilot ran all 12 jobs cleanly in 2 h 09 m. One replicate takes about 11 process-hours, so the 15–20 CPU-h
    estimate for the main run is too low.
- **Next:**
  1. Pilot analysis and Addendum A part 2.
  2. Re-cost the main run.
  3. Ask the user for compute.
- **Where:** experiments/signal/STEP4_DESIGN.md; runs/wbcp_signal/step4/expectations.md.

### 12. Signal study, steps 5–6 (not started)
- **Step 5:** coverage at the actor's own actions, and reused vs fresh banks.
- **Step 6:** ensembles, with TD3-BC-N as a baseline.

### 13. Full D4RL training: return and regret (not started; needs permission)
- **Status:** no training run with WBCP calibration exists yet.
- **Only return evidence:** one archived-radius pair (TD3+BC walker2d, one seed, 77.2 → 82.9). It is descriptive
  only.

## Decisions waiting on the user
1. Step 4 main-run compute, re-costed from the pilot.
2. The TD3+BC Q1 switch: the pre-registered rule says now; it is on hold until step 4.
3. The K rule: K from measured ρ per host × dataset, or test the n/D posterior correction.
4. Importance weights: adopt them at all, and toward which target distribution?
5. Permissions: actor fine-tuning (4B-1), TD3+BC pools on the random datasets, and later full training (4B-2).
6. Provenance: commit each expectation file before its run.

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
