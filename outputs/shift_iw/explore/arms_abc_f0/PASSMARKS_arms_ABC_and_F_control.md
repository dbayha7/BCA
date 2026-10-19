# Pass marks and kill rules: arms A, B, C and result F's no-band control (F0)

**EXPLORATORY.** Data seeds 0-4 are seen seeds. Nothing this file governs is a pre-registered or confirmatory result.
A confirmatory claim needs fresh data seeds 15-19, its own pre-registration and a new go from the owner.

- **Frozen:** 16 October 2026, before any code for these arms was written. Sidecar:
  `PASSMARKS_arms_ABC_and_F_control.md.sha256` (sha256sum format), in the same folder.
- **Never moved.** No mark, kill rule, comparator or wording below changes after freezing. Every script that trains,
  checks, scores or words these arms first compares this file with its sidecar and refuses on any difference. A code
  bug may be fixed; this file may not.
- **Sources.** `routes2/PROPOSAL.md` (it wins where sources disagree); the design workflow's journal
  (`wf_e5959f78-2c7/journal.jsonl`): fix-pull Arm 1 (arm A), transfer Arm 1 and skeptic Arm 1 (arm B, merged), skeptic
  Arm 2 (arm C), realistic-F Arm 1's P-same and skeptic Arm 3 part (ii) (F0); the judges' scores of both rounds;
  `routes2/diagnose/`. Section 10 lists every reading chosen where the sources were open or disagreed.

---

## 1. The owner's go and the two accepted changes

**The owner's go** (verbatim, 16 October 2026, in chat):

> i agree with 1 and 2 for suggested changes, and yes to A, B, C

**The two changes it accepts** (verbatim, as the session proposed them):

> 1. "Arm A's pre-training kill rule uses bars that were set after seeing seed 0. I'd apply it only to seeds 1-4."
> 2. "Add result F's missing control: the same 'no credit past 80%' rule with no band. The proposal pushed it to
>    later. But result F is going on the page, and this run is the only thing that tells us whether BCA's band or that
>    simple rule earned the gain."

**What the go covers:**
- the seed-0 screens of A, B and C, with A's band fits;
- seeds 1-4 of every arm whose kill checks hold;
- result F's no-band control F0, and result F's arm A2 on seeds 1-4, which F0's marks need;
- the data, reports, TD3+BC hosts, room checks and band fits these need, all in the sandbox.

**Not covered** (each needs its own go): co-training any arm; the matched-strength co-training check of the capped
pull; result F's other plain controls (P-knee, P-box); a second release weight; seeds 5-24; any confirmatory run.

---

## 2. Where it runs, and the standing rules

- **Sandbox:** `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun`, a byte-identical
  copy of the weighting test's code (spec version 5; 45 files, as its `run_seed0.sh verify-copy` checks).
- **Read only:** `/home/user/bca`. Result F's scripts run in place from
  `/home/user/bca/outputs/shift_iw/explore/channel/`; their outputs resolve into the sandbox's `repo/runs/`.
- **Files that are never created, edited or deleted:** any approval, spec, pass-mark or declined file of the existing
  tests. The committed approval (`outputs/shift_iw/v5/approval.json`, SHA-256 `c6b374d9…`, scope `band`, `coverage`,
  `bca`, `bca_lower_edge`) is only read, through the sandbox's existing link:
  - scope `band` covers the band fits (`run.py band`) and arm A's band reads;
  - scope `bca_lower_edge` covers A2 and F0, whose shared setup loads the band under that scope.
- **Allowed in the sandbox:** the package's own stage records and their `.sha256` sidecars under `repo/runs/`, and
  this run's arm records.
- **Where things go:**
  - new code (written only after this file is frozen): `routes2/arms/code/`;
  - logs and runner state: `routes2/arms/logs/` and `routes2/arms/state/`;
  - arm records: `<sandbox>/repo/runs/shift_iw_v5_explore/arms_abc_f0/<arm>/<quality>/seed<k>/`;
  - A2's seeds 1-4: where its unchanged script writes them (`.../channel/train_f1/<quality>/<arm>/seed<k>/`, seeds 1-4
    only). A2's seed 0 belongs to the result F rerun and is only read.
- **Every job's environment** (as `run_seed0.sh` and `run_resultF.sh`):
  `cd <sandbox>/repo; PYTHONPATH=<sandbox>/repo:<sandbox>/explore PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
  OMP_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES=<the AVX-512 list> XLA_FLAGS=--xla_cpu_max_isa=AVX2
  nice -n 10 taskset -c <one CPU> /tmp/claude-0/bcavenv/bin/python3 -B ...`
  - One CPU and one thread per job.
  - CPUs 1, 2 and 3 by default; CPU 0 is left to other sessions.
- **Waiting for the result F rerun.** The runner refuses to start while either of these holds, and never touches
  `resultF/state/` or `resultF/logs/`:
  - `resultF/state/run.lock` is held;
  - any process running `experiments.shift_iw.run`, `command_le_train_f1.py` or `run_resultF.sh` has its working
    directory in the sandbox.
- **Labels.** Every arm record carries:
  "EXPLORATORY: arms A/B/C and result F's no-band control; seen seeds 0-4; pass marks frozen in
  routes2/arms/PASSMARKS_arms_ABC_and_F_control.md (SHA-256 <its digest>) before this code was written".
  Package stage records cannot carry a label. So `EXPLORATORY_arms_abc_f0.txt` in each `runs/` root lists every file
  this run created, with its SHA-256.

---

## 3. Common ground for every arm

### 3.1 Test bed

- **The weighting test** (`experiments/shift_iw`, spec version 5), with the 80% hold removed: rows whose recorded
  command lies in the seed's ball are not trained on (the 'gap' removal).
- **Qualities:** random, medium-replay, medium, medium-expert and expert for A, B and C; medium-expert and expert for
  F0.
- **Data seeds 0-4.**
- **The host** of each seed and quality is plain TD3+BC trained on the gapped log (`run.py host --kinds gap`, 100,000
  updates). Its critic (two heads) is frozen; "the frozen critic" below always means this one.
- **The room check's protocol** for actor-only training:
  - The start is `learners.clone`: 20,000 cloning steps from TD3+BC's own initial actor.
  - Then 20,000 actor-only steps against the frozen critic, in 5 chunks of 4,000.
  - Batch 256, Adam 3e-4, alpha 2.5.
  - One shared stream of rows per seed (`learners.rng_np(settings, seed, TAG_ACTOR_ONLY)`, the rows of
    `learners.ActorOnly` and of result F).
  - The cloning term per row is `learners.actor_step`'s `bc_w`.
  - All arms of one program run vmapped side by side (`learners.ActorOnly.program`).
- **The three start banks** (`data.load_start_banks`), 1,000 fixed starts each:
  - **none:** no shift (lam −2, the training mix);
  - **mild:** lam 0;
  - **strong:** lam +2. This is "the strong test", and it is primary.
- **Scoring.**
  - Returns are exact float64 100-step returns from `truth.rollout` in the stalling world (`world.PHYSICS`).
  - Points are `world.points(J, anchor)` with the banks' anchors (0 = the random-quality data, 100 = the stall-aware
    controller), averaged over the 1,000 starts.
  - Unclipped actors are scored by `learners_iw.evaluate_actor`.
  - A ceiling is applied inside the policy as `clip(pi(x), −t_k, +t_k)` per thruster, before `truth.rollout`'s own clip
    to [−1, 1]. With every `t_k = 1` the code must reproduce `evaluate_actor` bit for bit (unit check).

### 3.2 How every mark is computed

- **Per seed** k: `d_k` = (the arm's points) − (the comparator's points), on the named bank, same seed and quality.
- **The mark's value** is the 5-seed mean of `d_0 ... d_4`.
- **Seed-sign count:** the number of seeds with `d_k > 0`. A tie counts as not positive.
- **Reported beside every mark, never deciding one:** each `d_k`, and the one-sided 95% lower bound
  `mean − 2.132 × SD / √5` (t, 4 degrees of freedom).
- **Thresholds are inclusive** ("at least +2.0" means mean ≥ 2.0). Values are compared at full precision; nothing is
  rounded before a comparison.
- **"Best" among rivals** means the rival with the highest 5-seed mean points on the named bank and quality (or
  summed over the qualities, where the mark is a sum). It is chosen after the fact, which favors the rival.
- **Missing numbers.** A mark that needs a missing number is not met. A number is missing when its seed was stopped,
  its actor failed or went non-finite, or a gate refused. A stopped seed is never replaced or rerun with other settings.

### 3.3 Kill rules: conventions

- Each kill rule is checked at the time it states, before any return it guards is computed. A rule that fires is never
  undone.
- **The seed-0 screen computes only what the kill rules read**, plus arm B's room check, which reads no BCA output.
- **No final return of an A, B or C arm is computed** until every kill rule that guards it has been checked on all
  five seeds. F0's seed-0 return is computed at the screen, because no later rule guards it.
- A stopped arm's returns may be computed later for information, marked "stopped by <rule>". They never enter a mark.
- The recording controller's own flags (`infos/random`, `infos/hard_water`, `infos/hold`) are read only by kill rules
  and diagnostics, never by an arm.

### 3.4 What the freezer had seen before freezing

- PROPOSAL.md, the design journal and the judges' scores.
- `routes2/diagnose/`: the seed-0 diagnosis, and the 5-seed learning curves and room check.
- `routes2/fixpull/`: the seed-0 release shares.
- **The result F rerun's seed-0 logs**, including A2's seed-0 scores on the strong test in the sandbox:
  - medium-expert: 94.69 points (J −148.15);
  - expert: 97.04 points (J −118.29);
  - its checks: on expert, A2 gives no credit to 100% of the no-caution actor's past-80% commands.

  F0's marks below are those set by the owner's change 2 and the session's task, not chosen after seeing these numbers.
- No code for these arms existed, and no arm had run.

---

## 4. Arm A: credible-target pull

**In one line.** TD3+BC's total pull, but no copying of a recorded command that the frozen critic rates below the
actor's own command by more than BCA's band width. Actor-only. BCA's band is read only at recorded commands, where it
is calibrated.

### 4.1 Mechanism (exact)

For each quality and seed:

1. **Inputs:**
   - the data (`run.load_data`);
   - `prep = data.prepare(data, 'gap', cfg)`: kept training rows i = 1..n, normalized states `s_i`, recorded commands
     `a_i`;
   - the host (`run.load_host_state(ctx, 'gap', seed, spec)`);
   - the band: `run.require_band`, loaded by `learners_iw.load_band(..., stage='band')`;
   - the band record's unweighted threshold record, `rec['thresholds']['unweighted']` (R_u).
2. **Fixed once per seed:**
   - **BCA's width at the recorded command:**
     `U_i = learners_iw.readout(band, learners_iw.reference_of(band, rec['thresholds']['unweighted']), prep.train)['width']`,
     that is, `R_u × unit × max(eta(s_i, a_i), 1e-6)`.
   - **Support guard bounds** per thruster k: `lo_k` and `hi_k`, the 0.5th and 99.5th percentiles (`numpy.quantile`,
     linear) of `a_ik` over the kept training rows. The guard uses no BCA, and every margin arm shares it.
   - **Plain residual spread s** (no BCA): over all kept training rows,
     `y_i = r_i + 0.98 (1 − done_i) min_h Q'_h(s'_i, clip(pi'(s'_i) + e_i, −1, 1))`, with:
     - the host's target networks Q' and pi';
     - target-policy noise `e_i ~ N(0, 0.2²)` clipped to ±0.5 (the arithmetic of `td3_bc_bca.native_target`), keyed
       from `numpy.random.default_rng([20261016, seed, 4302])`;
     - `q_i = min_h Q_h(s_i, a_i)`;
     - `s` = the population SD of `y_i − q_i`.

     The band's own `unit` and the ratio `mean U / s` are recorded beside s.
3. **Refreshes** at steps 0, 4,000, 8,000, 12,000 and 16,000 (each chunk's start). Each arm uses its own current actor
   `pi`:
   - `D_i = min_h Q_h(s_i, pi(s_i)) − min_h Q_h(s_i, a_i)`, from the frozen critic (`learners.critic_rows`,
     `learners.actor_rows`);
   - `guard_i`: `lo_k ≤ pi(s_i)_k ≤ hi_k` on all four thrusters;
   - **released rows:** `guard_i` and `D_i > margin_i`, with the arm's margin from the table below;
   - **weights:** `m_i = 0` on released rows and `m_i = n / (n − k)` on the others, where k is the number released. The
     mean weight is then exactly 1, TD3+BC's own total pull. Weights are computed in float64 and stored in float32. If
     every row is released, every weight is 0.
4. **Training:** each chunk's 4,000 steps with `learners.actor_step(bc_w = m)`.
5. **Saved actors:** the 8,000-step actors (after chunk 2) and the final 20,000-step actors of every arm.

**Arms** (one vmapped program of 12 arms per quality and seed):

| Arm | Release rule or table | Role |
|---|---|---|
| `none` | weight 1 on every row (`Arm('none', 'even', 1.0)`) | comparator ("TD3+BC") |
| `A1` | margin `U_i` | **the arm** |
| `margin0` | margin 0 | P4 control |
| `plain_1s`, `plain_2s`, `plain_4s` | margins 1s, 2s and 4s | P4 controls (the best of the three counts) |
| `matched_share` | releases the `k_A1` guard-passing rows with the largest `D_i`. `k_A1` is the number `A1` releases at the same refresh. Ties are broken by `ActorOnly.tie`. | P4 control: the band's level without its state shape |
| `untargeted` | releases `k_A1` rows drawn uniformly without replacement among this arm's guard-passing rows (all of them if there are fewer), from `default_rng([20261016, seed, 4301, r])`, r = refresh index 0-4 | P3 control: placement |
| `shuffled_band` | margin `U_perm(i)`, with `perm = ActorOnly.perms[0]` of the seed | information |
| `even_x1.33` | weight 1.33 on every row | information: the capped pull's actor-only twin |
| `A1_IW` | margin `U^IW_i`, BCA's width with each row's own threshold from the exact strong-mix table. The table is rebuilt exactly as `run.stage_band` builds `exact_strong` (`step_masses`, `mass_grid` with spec v5's per-step grid ratio, `threshold_tables`, `coverage.lookup`); then `learners_iw.per_step_readout(band, R^IW, prep.train)['width']`. A refused row (R = +inf) is never released. | information: exact weights are an oracle input |
| `A1c` | weight 0 where A1's rule releases (computed from A1c's own actor); otherwise BCA's own dose `1 + 0.5 U_i / (U_i + unit)` (readout's dose); no renormalization | information: the capped pull retargeted |

### 4.2 Comparator and controls

- **P1 and P2:** `none`, the actor-only TD3+BC with the same clone, rows and alpha.
- **P3:** `untargeted`.
- **P4, the band's own part, which decides whether BCA gets credit:** `margin0`; the best of `plain_1s`, `plain_2s`
  and `plain_4s` (best on that quality's strong test); and `matched_share`.

### 4.3 Pass marks (strong test unless a bank is named; 5-seed means over seeds 0-4)

- **P1, gain:** `A1 − none` is at least **+2.0** points on at least **2 of** medium-replay, medium and medium-expert,
  with at least **4 of 5** seeds positive on each counted quality. The counted qualities are "the passing qualities".
- **P2, no harm:** `A1 − none` is at least **−1.0** on **all five** qualities and on **every bank** (none, mild and
  strong).
- **P3, targeting:** `A1 − untargeted` is at least **+1.5** on every passing quality.
- **P4, the band's own part:** on every passing quality, all three hold:
  - `A1 − margin0` is at least +1.0;
  - `A1 − (best plain margin)` is at least +1.0;
  - `A1 − matched_share` is at least +1.0.
- **GO = P1 + P2 + P3 + P4.** A GO licenses only asking the owner for a confirmatory pre-registration on fresh seeds
  15-19. Co-training this rule is a separate go.
- **Information only:**
  - `A1 − shuffled_band`, `A1_IW − A1`, `A1c − none`, `even_x1.33 − none`;
  - `s`, the band's unit, `mean U / s`;
  - release shares per refresh among recorder rows and random rows, and the releases the guard blocked;
  - the co-trained host minus `none`.

### 4.4 Wording, one per outcome

- **P1-P4 all met:** "Exploratory: the credible-target pull raised return over TD3+BC on <qualities>, and BCA's band
  beat margin 0, every plain margin and a matched-share margin."
- **P1, P2 and P3 met, P4 not:** "Filtering helps; the band adds nothing over a plain margin." It names the rival that
  tied or won.
- **P1 and P2 met, P3 not:** "The gain does not come from targeting: releasing as many random rows did as well."
- **P1 not met:** "The credible-target pull did not raise return over TD3+BC."
- **P2 not met (any case):** add "It lowered return on <quality, bank> by <x> points." There is no GO.
- **A kill rule fired:** "Stopped by <rule> on <quality, seed>: <reason>." Marks that need that quality are not met.

### 4.5 Kill rules

- **K0, before training (change 1).** At refresh 0 (the clone actor), on medium-replay, medium and medium-expert, `A1`
  must release:
  - **at most 10%** of recorder rows (`infos/random` false), and
  - **at least 15%** of random rows (`infos/random` true).

  Seeds 1-4: binding. A failure stops A on that seed and quality, and the seed is never replaced. **Seed 0: recorded as
  information only**; it stops nothing. K0 is not checked on random (no recorder rows) or expert (no random rows). The
  releases the guard blocks are recorded.
- **K1, at the step-4,000 refresh, on every seed including seed 0.** A stops on that quality (all its seeds) if either
  holds:
  - `A1` releases more than **60%** of the kept rows;
  - the mean of `D_i` under A1's actor exceeds **3 times** the mean of `U_i` (the actor exploiting the critic).
- **K2, at 8,000 steps.** Checked once all five seeds of a quality have reached 8,000 steps, before any final A actor of
  that quality is scored. A stops on that quality if `A1 − none` on the strong test, from the 8,000-step actors, is
  **−3.0 points or lower on at least 3 of 5 seeds**. Seed 0's value is recorded at the screen.
- **K3, any time.** Any of these stops that seed and quality:
  - a non-finite actor in an arm that a mark needs;
  - an invalid band (`band_valid` false);
  - a refused gate check at any band read.

---

## 5. Arm B: certified throttle ceiling

**In one line.** TD3+BC is unchanged. When it acts, each thruster stops at the highest 0.05 step that BCA's
certificate, with estimated weights, says beat the step below on held-out tries. No tries, no step.

### 5.1 Mechanism (exact)

The consumer is the host actor `pi_H`. For each quality and seed:

1. **Tries (held out).** Rows of the seed's training-law log with:
   - `infos/split` = 1 (calibration, episodes 500-699) and not in the calibration bank (`log.bank_rows`); or
   - `infos/split` = 2 (held out, episodes 700-999).

   Rows whose recorded command lies in the seed's ball (`data.gap.contains`) are dropped, so the 80% hold is missing
   from the evidence just as it is from training. Transitions come from `learners_iw.transitions(log, rows, prep)`,
   with the host's normalization.
2. **Rungs** per thruster k = 1..4. The edges are `e_0..e_10 = float32(0.50, 0.55, ..., 1.00)`, and rung j is
   `(e_{j−1}, e_j]` for j = 1..10. The top edge of rung 6, `(0.75, 0.80]`, is `float32(0.8)`, the knee.
3. **The tries of rung j for thruster k** are tries where all three hold:
   - the recorded `|a_k|` (float32) lies in the rung;
   - the recorded `a_k` has the same nonzero sign as `pi_H(s)_k` at the try's own state;
   - `|pi_H(s)_k| > e_{j−1}`.
4. **Gain of a try:** `g = r + 0.98 (1 − done) min_h Q_h(s', pi_H(s')) − min_h Q_h(s, ã)`.
   - `ã` is the recorded command with thruster k set to `sign(a_k) × e_{j−1}`.
   - The critic and the actor are the frozen host's. There is no target noise.
5. **Weights:** BCA IW's estimated strong-mix weights.
   - Each try's weight is `w = clf.weight(the try's raw observation)`, with `clf = run.classifiers_of(data)['strong']`.
   - The test mass is `classifier.estimated_arm(clf, ...)['test_mass']` for 'strong', exactly as `run.estimated_arms`
     computes it.
6. **BCA's certificate:** `calibration.wbcp.calibrate(scores = −g, rng = default_rng([20261016, seed, 4311, k, j, 0]),
   weights = w, test_mass, alpha = 0.5, beta = 0.95, draws = 1000)`.
   - The rung is **certified** when the returned threshold is **below 0**. That means BCA is 95% sure that more than
     half of the deployment-weighted tries gained over stopping one rung lower, even counting one unseen try at the
     worst loss.
   - A rung with no tries is refused.
7. **Ceiling `t_k`:** walk j = 1, 2, ... and stop at the first refusal (a fixed sequence, so no multiplicity
   correction).
   - `t_k` is `e_j` of the last rung in the unbroken run of certified rungs from rung 1.
   - `t_k = 0.50` if rung 1 is refused; `t_k = 1.00` if all ten are certified.
8. **Acting:** `clip(pi_H(x), −t_k, t_k)` per thruster. One map per seed and quality, used on all three banks.

### 5.2 Rival ceilings

All use the same tries, gains, weights, rungs and walk; only the decision per rung differs.

**The plain tests** (no BCA):
- **weighted median:** certified if `Σ w 1[g > 0] > 0.5 Σ w`;
- **weighted mean:** certified if `Σ w g / Σ w > 0`;
- **exact sign test (unweighted):** certified if `P(X ≥ m) ≤ 0.05` for `X ~ Binomial(n, 1/2)`, with m = the number
  of tries with `g > 0` and n = the number of tries;
- **weighted t-test:** with `n_eff = (Σ w)² / Σ w²`, the weighted mean `mean_w` and
  `var_w = Σ w (g − mean_w)² / Σ w`, certified if `n_eff > 1` and `mean_w / sqrt(var_w / n_eff) > t_0.95(n_eff − 1)`;
- **weighted empirical Bernstein:** with `LB = mean_w − sqrt(2 var_w ln 40 / n_eff) − 7 B ln 40 / (3 (n_eff − 1))`
  (that is, δ = 0.05) and B = (largest g) − (smallest g) over all of thruster k's tries, certified if `n_eff > 1` and
  `LB > 0`.

**Ceilings that test placement and strength:**
- **range limit:** `t_k` = the 99.5th percentile of `|a_k|` over the kept training rows. It reads no outcomes.
- **shuffled outcomes:** BCA's rule (rng tag 2) after permuting g across all tries of thruster k (all rungs pooled),
  with the permutation from `default_rng([20261016, seed, 4312, k])`. The weights stay with their rows.
- **matched global ceiling:** one `t_g` for all four thrusters.
  - Let p be the share of thruster entries with `|a_tk| > t_k` (BCA's map) in the host's own unclipped strong-start
    rollouts (1,000 starts × 100 steps × 4 thrusters).
  - `t_g = numpy.quantile(those |a| values, 1 − p, method='higher')` if p > 0, else 1.00.

**Reported only:**
- **knee clip:** `t_k = float32(0.8)` for every k. It uses knowledge of the world; kill rule K0 also reads it.
- **BCA without weights:** `wbcp.calibrate` with no weights, rng tag 1.
- **BCA with exact weights:** `world.ratio(c, 'strong')`, test mass `data.test_mass('strong', gapped=True)`, rng tag 3.
- **BCA on a shifted grid:** edges 0.50, 0.525, 0.575, ..., 0.975, 1.00, so that no edge sits on the knee; rng tag 4.

### 5.3 Pass marks (strong test, 5-seed means, unless stated)

- **G1, gain:** `(host + BCA ceiling) − host` is at least **+2.0** on **medium-expert**, with at least **4 of 5**
  seeds positive.
- **G2, no harm:** `(host + BCA ceiling) − host` is at least **−1.0** on **every quality** and **every bank**.
- **G3, outcome reading, on medium-expert:**
  - `(host + BCA) − (host + range limit)` is at least **+1.5**;
  - `(host + BCA) − (host + shuffled outcomes)` is at least **+1.5**;
  - `(host + BCA) − (host + matched global ceiling)` is at least **+1.0**.
- **GO = G1 + G2 + G3.** As for A, a GO licenses only asking for a confirmatory pre-registration.
- **G4, BCA's own part, which decides whether BCA gets credit:** take the "best plain test", the one of the five with
  the highest sum over the five qualities of 5-seed mean strong-test points. Both must hold:
  - `Σ over the five qualities of [(host + BCA) − (host + best plain test)]` is at least **+2.0**;
  - no quality has `(host + BCA) − (host + best plain test)` below **−1.0**.
- **Information only:**
  - G1's value on expert. A gain there comes from having no tries past 0.80, which is the support rule, and is never
    credited to BCA.
  - BCA minus the knee clip; the unweighted, exact-weight and shifted-grid variants.
  - The ceilings per thruster and seed; tries per rung; the share of commands clipped.

### 5.4 Wording, one per outcome

- **GO and G4:** "Exploratory: BCA's certified throttle ceiling raised TD3+BC's return on medium-expert, and BCA's
  certificate beat the best plain test."
- **GO without G4:** "Reading held-out outcomes raised return on medium-expert; BCA's certificate did no better than
  <best plain test>." In short: outcome reading helped, not BCA.
- **G1 and G2 met, G3 not:** "A ceiling raised return, but not by reading outcomes: <the range limit / shuffled
  outcomes / a matched global ceiling> did as well."
- **G1 not met:** "The certified ceiling did not raise return over TD3+BC on medium-expert."
- **G2 not met:** add "It lowered return on <quality, bank> by <x> points." There is no GO.
- Never "BCA improves TD3+BC" on its own.

### 5.5 Kill rules

- **K0, room.** It reads no BCA output: `(host + knee clip) − host` on the strong test, medium-expert.
  - At the screen (seed 0): below **+1.0** stops B.
  - Again on the 5-seed mean, once the seeds 1-4 hosts exist and before any gated return: below +1.0 stops B.
- **K1, the certificate missed the stall.** Checked after each seed and quality's maps, before any gated return. B
  stops if BCA's **thruster-1 ceiling is 0.85 or higher** in any seed and quality whose rung (0.80, 0.85] holds **at
  least 30 tries**.
- **K2, over-refusal.** Checked at the same point. B stops if BCA's **thruster-1 ceiling is 0.60 or lower in 2 or
  more** seed-quality pairs, counted over all qualities and seeds.
- **K3, non-finite values.** A non-finite gain, weight, actor or return stops that seed and quality. Two such seeds stop
  B.
- **K4, nothing to change** (PROPOSAL's "under 2% clipped"). B stops if BCA's map clips **under 2%** of the host's
  thruster-1 commands in the host's unclipped strong-start rollouts on medium-expert. Checked on seed 0's value at the
  screen and on the 5-seed mean before scoring.
- **K1 and K2 are properties of the certificate.** If either fires, C's guard, which uses the same rule, is void too
  (section 6.4).

---

## 6. Arm C: guarded release

**In one line.** Half the pull toward the recorded commands (cloning weight 0.5), the capped pull's opposite, with B's
ceiling rule as the guard against stalling. There is no BCA in training. The release itself is a change of TD3+BC's
alpha and is never credited to BCA.

### 6.1 Mechanism (exact)

1. **Actors.** For each quality and seed, one program runs two arms from the clone, with the room check's protocol:
   - **rho 1:** `Arm('none', 'even', 1.0)`;
   - **rho 0.5:** `Arm('release_x0.5', 'even', 0.5)`.

   The 8,000-step and final actors are saved. `ActorOnly.run`'s chunk loop is restated only to save the 8,000-step
   actors.
2. **The guard: B's ceiling rule, built for C's own actor.** Steps 1-8 of section 5.1, with BCA and estimated
   strong-mix weights, except one change:
   - step 3's try selection uses the final rho 0.5 actor in place of `pi_H`.

   The gains are unchanged (host critic, host continuation `pi_H(s')`). Every rival ceiling is built the same way, for
   the rho 0.5 actor. The matched global ceiling is matched on the rho 0.5 actor's own unclipped strong-start commands.
   C's maps use B's random streams with 10 added to the rule tag (BCA 10, shuffled 12, and so on).
3. **Acting:** `clip(pi_0.5(x), −t_k, t_k)` per thruster.

### 6.2 Comparator and controls

- **Comparator:** rho 1 with no ceiling, the actor-only TD3+BC from the same clone and stream.
- **Controls:**
  - rho 0.5 alone;
  - rho 0.5 with the range limit, with shuffled outcomes, with the matched global ceiling, and with each of the five
    plain tests.

### 6.3 Pass marks (strong test, 5-seed means, unless stated)

- **B1, gain on weak data:** `(rho 0.5 + BCA) − rho 1` is at least **+3.0** on at least **2 of** random,
  medium-replay and medium, with at least **4 of 5** seeds positive on each counted quality. The counted qualities are
  "the passing qualities".
- **B2, the guard is needed:** `(rho 0.5 + BCA) − (rho 0.5 alone)` is at least **+1.5** on every passing quality.
- **B3, outcome reading:** on every passing quality, all three hold:
  - `(rho 0.5 + BCA) − (rho 0.5 + range limit)` is at least **+1.5**;
  - `(rho 0.5 + BCA) − (rho 0.5 + shuffled outcomes)` is at least **+1.5**;
  - `(rho 0.5 + BCA) − (rho 0.5 + matched global ceiling)` is at least **+1.0**.
- **B5, no harm:** `(rho 0.5 + BCA) − rho 1` is at least **−1.0** on **all five** qualities and **every bank**.
- **GO = B1 + B2 + B3 + B5.**
- **B4, BCA's own part, which decides whether BCA gets credit:** `(rho 0.5 + BCA) − (rho 0.5 + best plain test)` is at
  least **+1.0** on every passing quality. The best plain test is the one with the highest 5-seed mean on that quality.
- **Information only:**
  - rho 0.5 alone minus rho 1 (the release on its own);
  - rho 0.5 with the knee clip;
  - rho 1 with a BCA ceiling built for rho 1;
  - rho 0.5 with B's host-built map;
  - each of these against the co-trained host.

### 6.4 Kill rules

- **KB1, critic exploitation**, at 8,000 steps on every seed including seed 0. It stops rho 0.5 on that quality if:
  - the actor is non-finite; or
  - its critic gain `G = mean_i [min_h Q_h(s_i, pi(s_i)) − min_h Q_h(s_i, a_i)]` over the kept rows exceeds
    3 × G(rho 1) when G(rho 1) > 0, or exceeds 3s (arm A's plain residual spread for that seed and quality, computed
    as in section 4.1 even where A is stopped) when G(rho 1) ≤ 0.
- **KB0, the guard is idle.** Checked before any C return, on unclipped strong-start rollouts of the final actors. The
  rho 0.5 actor's share of thruster-1 commands past the knee (`|a_1| > float32(0.8)`) must be:
  - **at least 2%**, and
  - **at least twice** the rho 1 actor's,

  on **at least 2 of** random, medium-replay and medium. It is checked on seed 0's values at the screen and on the
  5-seed means before scoring. If it fails at either point, B2, B3 and B4 are not scored. B1 and B5 are still scored.
- **The guard is void, and B2-B4 are not scored, if either holds:**
  - B's K1 or K2 fires;
  - C's own map has a thruster-1 ceiling of 0.85 or higher with at least 30 tries in rung (0.80, 0.85] in any seed and
    quality.
- **K3** applies as in B.

### 6.5 Wording, one per outcome

- **B1 not met:** "Halving the pull did not raise return on the weak qualities, guarded or not."
- **B1 met, B2 or B3 not:** "Less pull helps; BCA adds nothing."
- **B1, B2, B3 and B5 met, B4 not:** "A held-out-outcome guard makes a bolder TD3+BC safe; BCA's certificate did no
  better than <best plain test>."
- **All met, including B4:** "Exploratory: with BCA's certificate as its guard, a TD3+BC with half the pull raised
  return on <qualities>." The release is TD3+BC's alpha. Never "BCA improves TD3+BC" on its own.
- **KB0 fired:** "The guard had nothing to guard: the bolder actor rarely went past 80%." Then B1 is reported alone.
- **B5 not met:** add the harm line, as for A. There is no GO.

---

## 7. Result F's no-band control F0 (change 2)

### 7.1 What it is and the one code path that differs

F0 is result F's arm A2 ("fixed fallback, exact weights", `command_le_train_f1.py`) with the band's finite part
deleted.

**A2's penalty** is `penalty_band_f1`, which ends in `return jnp.where(inf_ref, w_ref, w_cert)`, where:
- `w_ref = [Q1(s, pi) − Q1(s, a_i)]_+` is "no value credit over the recorded command";
- `inf_ref` is `band_read_f1`'s flag. It is set where the command is inside the removed ball, or where the recording
  process's exact density is 0 at the row's state and episode; there the exact weight is infinite;
- `w_cert = [U(s, pi) − U(s, a_i)]_+` is the band's certified width. U is read at the table's threshold, or at the cap
  where the bank is too small.

**F0's penalty** is `jnp.where(inf_ref, w_ref, 0.0)`: the same `w_ref` on the same `inf_ref` rows, and no penalty
anywhere else, so no band and no finite threshold. **That line is the only difference.**

Everything else is A2's, called unchanged:
- the setup: `T.World(quality, seed)` (data, the frozen host critic, the clone from `World.start()`, the row stream
  from `World.rows()`);
- the per-chunk tree: `command_le_train_f1.refresh(wd, actor, "strong")`, so the no-credit set comes from A2's own
  code;
- the program: `jax.jit(partial(lower_edge._loop, args, penalty))`;
- 5 chunks of 4,000 steps, a refresh at each chunk's start, alpha 2.5, k = 1, pull 1 on every row, Adam 3e-4,
  batch 256;
- the settings (spread 0.2, clip 0.5, grid ratio 1.05, floor 1e-4);
- scoring by `learners_iw.evaluate_actor` on the three banks.

The band-derived parts of the tree (`grid`, `thresholds`, `cap`, `u_logged`) and the band's `cal_params` are still
present, but F0's loss never reads them (check KF2b).

**What "the same rule" means on each quality:**
- **Expert:** A2's no-credit set is every past-80% command plus the ball. On seed 0, it covered 100% of the no-caution
  actor's past-80% commands. So F0 is "no credit past 80%" there.
- **Medium-expert:** the random-share episodes give past-80% commands a finite density. So A2's no-credit set covers
  only part of the actor's past-80% commands: 14-44% per chunk on seed 0 in the sandbox, 14% in the final chunk. The
  band's capped threshold handles the rest, and F0 drops that part.

F0 is A2's own no-credit rule without the band, as change 2 asks. A literal "any thruster past 80%" rule (P-knee, which
knows the knee) is not part of this go.

### 7.2 Runs

- **F0:** seeds 0-4 on medium-expert and expert. Records go to
  `<sandbox>/repo/runs/shift_iw_v5_explore/arms_abc_f0/F0/<quality>/F0_no_band/seed<k>/`, never to `train_f1/`.
- **A2 on seeds 1-4:** `command_le_train_f1.py train <Q> <k>`, then `evaluate <Q> <k>`, unchanged, for k = 1-4. As on
  the desktop, it also trains B2 and C2.
  - A2's seed 0 is the result F rerun's record and score, read only.
  - **This needs, for medium-expert and expert seeds 1-4:** the data (already built), plus the hosts, room checks and
    bands, which this run builds. The band stage needs that seed's room check.

### 7.3 Checks and kill rules (before each seed and quality trains)

- **KF1, the same no-credit set.** At the clone actor's commands on every kept row (and on seed 0 also at the result F
  rerun's no-caution actor's commands), F0's in-graph no-credit flags must equal `command_le_train_f1.refresh`'s numpy
  `inf_pi` on every row. The disagreement must be exactly 0.
- **KF2, nothing else differs.** Both must hold:
  - **(a)** with every no-credit flag forced false, 300 steps of F0's program from the clone equal 300 steps of
    `lower_edge._loop` with a zero penalty, bit for bit;
  - **(b)** 300 steps of F0's program are bit for bit unchanged when the tree's `grid`, `thresholds`, `cap` and
    `u_logged` and the band's `cal_params` are replaced by other finite values. This shows that no band input reaches
    F0.
- **KF3, the same start.** F0's clone SHA-256 must equal A2's clone SHA-256 for the same seed and quality.
- **KF4:** a non-finite actor stops that seed and quality.
- **When one fires:** a failing seed and quality is stopped and never replaced. Its quality's mark is then not met, and
  the wording for that quality is "not decided: <reason>".
- **No cross-machine bar.** A2 and F0 are compared in the same sandbox, with the same settings, paired by seed. The
  result F rerun's own `COMPARE.md` compares A2 with the desktop. A bar such as the journal's "within 3.0 return units
  of the desktop's A2" is not adopted: the sandbox's seed-0 A2 is already known to differ from the desktop's by 4.3
  return units on medium-expert, so such a bar would be set after seeing the number.

### 7.4 Pass marks and wording

Per seed, `d_k = A2 − F0` in points on the strong test. **BCA's band gets credit on a quality** when the 5-seed mean is
**at least +1.0** and at least **4 of 5** seeds are positive.

| Outcome | Wording |
|---|---|
| Credit on both medium-expert and expert | "BCA's band earned part of result F's gain: A2 beat the same no-credit rule without the band by <x> points on medium-expert and <y> on expert." |
| Credit on one quality | "Result F's gain comes partly from BCA's band (on <quality>); on <other quality> it comes from refusing credit past the data, not from BCA's band." |
| Credit on neither | "Result F's gain comes from refusing credit past the data, not from BCA's band." |
| Added on any quality where `A2 − F0` is −1.0 or lower with at least 4 of 5 seeds negative | "On <quality>, the band's finite part lowered return by <x> points." |

**Information only:**
- the mild and none banks;
- F0's no-credit share per chunk beside A2's;
- on seed 0, F0 and A2 each minus the result F rerun's "no extra caution" actor (`le_none`).

---

## 8. Order of work

0. **Freeze** this file and its sidecar. Nothing else is written before this.
1. **Code.** Write the code in `routes2/arms/code/`, then run the gates and the unit checks.
   - **Gates** (each refuses on failure):
     - spec sidecar (`a71af9bd…`);
     - the sandbox code copy is byte-identical (45 of 45 files);
     - the approval verifies for `band` (all five qualities) and `bca_lower_edge` (medium-expert, expert);
     - result F's scripts equal the code named in the desktop records (`command_le_train_f1.py` `f3f3b734…`,
       `command_le_train.py` `c91f4608…`, `command_band_check.py` `3c4d4dfa…`);
     - result F's pass marks are unchanged (`8b8f7a27…`);
     - this file matches its sidecar.
   - **Unit checks:**
     - **A:** with margin +inf on every row, `A1` is bit-identical to `none` over a full chunk; the mean weight is
       exactly 1 after renormalization; the guard blocks commands outside `[lo, hi]`; band reads are refused without
       the gate; a run on synthetic data (`learners_iw.synthetic_prepared`) works end to end.
     - **B and C:**
       - the all-1.00 map reproduces `evaluate_actor` bit for bit on all three banks;
       - an empty rung is refused under every rule;
       - all-negative gains are refused and 1,000 equal-weight all-positive gains are certified under every rule;
       - an identity permutation reproduces BCA's map.
     - **F0:** KF1 and KF2.
     - **Scorer:** it refuses when this file differs from its sidecar, and it refuses records without the label or
       with another `passmarks_sha256`.
   - The runner then waits for the result F rerun (section 2).
2. **Data for the remaining qualities.**
   - `run.py data --quality Q --seeds 1 2 3 4` and `run.py report --quality Q`, for random, medium-replay and medium.
   - D1 of seeds 0-4 is compared bit for bit with the committed `outputs/shift_iw/v5/<Q>/report.json`.
   - A report that does not say D1 passed, or any D1 difference, stops band fits on that quality until the owner
     decides.
3. **Seed-0 screens** (seed 0 data and hosts exist for all five qualities):
   - **A:**
     - `run.py room` and then `run.py band --kinds gap`, seed 0, for random, medium-replay and medium. Medium-expert
       and expert already have both from the result F rerun.
     - Then A's program on all five qualities: K0's seed-0 numbers (information), K1, K3, and K2's seed-0 value.
   - **B:** tries, gains and every map on all five qualities. Then K0 (seed 0), K1, K2, K3 and K4 (seed 0).
   - **C:** train rho 1 and rho 0.5 on all five qualities. Then KB1, KB0 (seed 0) and the guard-void check.
   - **F0:** seed 0 on medium-expert and expert, next to the result F rerun's A2. Then KF1-KF4, training, and the
     seed-0 score.
4. **Screen report.** `routes2/arms/SCREEN.md` (EXPLORATORY) lists every kill quantity and which arms and qualities go
   on. No final return of A, B or C is shown.
5. **Seeds 1-4**, for every arm that was not stopped:
   - **Hosts** (`run.py host --kinds gap`): all five qualities if A, B or C survives; otherwise medium-expert and
     expert only, for F.
   - **Room checks and bands:** all five qualities if A survives; medium-expert and expert in any case, for A2 and F0.
   - **A:** K0 now binding; K1 per seed; then K2 across the five seeds from the 8,000-step actors.
   - **B:** maps; K1, K2 and K3 per seed; K0 and K4 on the 5-seed means.
   - **C:** KB1 per seed; KB0 on the 5-seed means; the guard-void check.
   - **F:** A2 by its unchanged script; F0 after KF1-KF4.
6. **Scoring**, only after every kill rule above has been checked. The scorer checks this file's SHA-256, scores every
   surviving actor and ceiling on the three banks, computes every mark of sections 4-7 and writes
   `routes2/arms/RESULTS.md` and `results.json` (EXPLORATORY), with the wording fixed above.
   - **Information only, after the marks:** the main test's four situations for the primary arms (A1, none, host +
     BCA, host, rho 0.5 + BCA, rho 1, A2, F0).

---

## 9. Compute estimates

### 9.1 Measured on this machine, 15-16 October, one pinned CPU per job, other sessions present

| Job | Time |
|---|---|
| Data build | 110-125 s per seed |
| TD3+BC host | 1,354-1,796 s |
| Room check | 1,341-1,413 s |
| Band fit | 105-110 s |
| A2 + B2 + C2 (result F's script) | 443-493 s per seed and quality |
| Even actor-only arms | about 37 s per 4,000 steps for two vmapped arms, so about 1.2 CPU-minutes per arm per 20,000 steps |
| One policy on one bank | about 3.5 s |

### 9.2 Per seed and quality (estimates)

- **A:** 12 arms of about 1.6 CPU-minutes each, clone, refreshes, A1_IW's table, the K2 snapshot and scoring of 12
  actors on 3 banks: about 26 CPU-minutes.
- **B:** gates and about 14 ceilings on 3 banks: about 5 CPU-minutes.
- **C:** 2 arms, clone, maps and about 14 ceilings on 3 banks: about 9 CPU-minutes.
- **F0:** setup, KF1-KF2, training and scoring: about 5 CPU-minutes.

### 9.3 Totals

| Stage | CPU-hours | Wall clock on CPUs 1-3 |
|---|---|---|
| Data and reports, three qualities | about 0.45 | |
| Seed-0 screen: 3 room checks, 3 bands, A ×5, B ×5, C ×5, F0 ×2 (with the data) | **about 5.2** | **about 2-2.5 h** |
| Seeds 1-4, everything survives: 20 hosts (8.7), 20 room checks (7.7), 20 bands (0.7), A (8.7), B (1.7), C (3.0), A2 script ×8 (1.1), F0 ×8 (0.7) | **about 32** | **about 11 h** (8 h on all four CPUs) |
| Seeds 1-4 if A is stopped everywhere at the screen | about 18 | |
| Seeds 1-4 if only F remains | about 9 | |
| Scoring and reports | about 0.5 | |
| **Whole run, everything survives** | **about 38** | **about 13-14 h** |

**Range.** If the machine is busier than it was during the measurements, allow up to about 40 CPU-hours for seeds 1-4
(about 14 h on three CPUs).

**What the proposal left out.** The band stage on a gapped host needs that seed's room check (`run.stage_band`; about
23 minutes each here). This adds about 8.8 CPU-hours over 23 seed-quality pairs, about 3.1 of them for F's eight.

---

## 10. Readings chosen where the sources were open or disagreed

Where they disagreed, PROPOSAL.md won. Where they were open, the most conservative reading was chosen. Each choice is
listed here.

1. **Confirmatory seeds** are 15-19 (PROPOSAL), not 20-24 (fix-pull journal).
2. **Comparators.** A and C are judged against actor-only TD3+BC (`none` / rho 1, same clone and stream), as their
   journal designs state. B is judged against the co-trained host, which is its consumer. Differences against the
   co-trained host are information for A and C.
3. **No harm.** −1.0 on every quality and **every bank** (none, mild and strong) for all arms. PROPOSAL's −1.0 for B
   wins over the skeptic journal's −0.5. The journal named only strong and none for A; adding mild is the conservative
   reading.
4. **A's P4 rivals** are PROPOSAL's: margin 0, the best of 1, 2 and 4 residual spreads, and a matched-share constant.
   The first-round "plain 90% quantile" was dropped in PROPOSAL and is not used. The shuffled band is information.
5. **A's primary threshold** is the unweighted band threshold. `A1_IW` uses exact strong-mix weights, an oracle input,
   so it is information only, and the journal's wording mark P7 is not used.
6. **A's kill rules** are PROPOSAL's three, plus the journal's critic-exploitation half of K1 and its K3.
   - K0 follows change 1: binding on seeds 1-4, information on seed 0.
   - "3 points behind" means `A1 − none` of −3.0 or lower.
7. **A's `untargeted` control** draws among guard-passing rows, so it shares the support guard. This makes P3 harder
   to pass.
8. **A's plain spread s** uses the TD target with target-policy noise, as BCA's own unit does, but no BCA part. It is
   fully specified above (stream tag 4302).
9. **B's mechanism** is transfer Arm 1's: estimated weights, wbcp at alpha 0.5 and beta 0.95, 0.05 rungs, one ceiling
   per thruster and the host as consumer. These are PROPOSAL's "certificate (estimated weights)" and "each thruster".
   - Skeptic Arm 1's exact weights, current-band cells and mean gate are not used. Its kill rule "under 2% clipped"
     (K4) and its G1 and G3 on medium-expert are used, as PROPOSAL lists them.
   - One map, built with strong-mix weights, serves all three banks, as a deployed policy would.
10. **B's plain tests** are every fully defined plain test of the two merged designs: weighted median, weighted mean,
    exact sign test, weighted t-test and weighted empirical Bernstein (defined above). The best one is chosen after
    the fact. G4 adds the journal's floor (no quality below −1.0) to PROPOSAL's "+2.0 summed".
11. **B's K0** is judged on medium-expert, the quality G1 judges. It is checked on seed 0 and again on the 5-seed mean.
12. **C's guard** is B's rule built for C's own rho 0.5 actor. This is transfer Arm 1's consumer L, which is exactly
    this actor. B's host-built map on the rho 0.5 actor is information.
13. **C's B4** (BCA against the best plain test, +1.0 per passing quality, from the skeptic journal) is kept as the
    mark that decides whether BCA gets credit. PROPOSAL lists no such mark for C, but the task asks for one.
14. **C's KB1.** The journal's "mean critic value exceeds 3 times rho 1's" is undefined in sign, because the values
    are negative. It is read as the critic's claimed gain over the recorded command, with a fallback of 3s when rho 1's
    gain is not positive.
15. **C's guard is voided** by B's K1 or K2, or by the same K1 condition on C's own map. This rule is added, because a
    guard that failed its own sanity check should not be credited.
16. **Dropped:** the release weight 0.25 (PROPOSAL fixes 0.5; a second weight would be a tuned knob) and A's noise-map
    information arm (expensive exact rollouts that read the world's truth; not in PROPOSAL).
17. **Timing of returns.** No final return of A, B or C is computed before every kill rule guarding it has been checked
    on all five seeds. The seed-0 screen is a kill-check screen only. F0's seed-0 score is the exception, because no
    later rule guards it.
18. **Stopped seeds** are never replaced. Any mark that needs them is not met.
19. **F0** is A2's own no-credit rule with nothing else (section 7.1), as the task defines it. P-knee and P-box are not
    in this go. No cross-machine kill bar is adopted (section 7.3).
20. **The main test's four situations** are information only, after the marks.
