# Gapped LQ (GLQ-10x4x5)

A linear-quadratic test world for one question: does putting caution where the critic is actually wrong
improve return, compared with spreading the same caution evenly?

Every return and every Q-value is exact. The critic is learned from a log, never planted. In two rare
contexts the log barely touches two of the four actuators, so the critic must guess what they do. By the
true dynamics the guess is right in one of them (T_right) and has the wrong sign in the other (T_wrong).

Code: `gapped_lq.py` (world, Stage 0, data, exact truth), `glq_train.py` and `glq_match.py` (training,
Tier A, blind matching), `glq_acceptance.py` (the gate). Pre-registration: `glq_preregistration.json`.
Design: option A of the world audit, with the judge's grafts g1-g8.

## Description

The plant is five lightly damped 2-D oscillators (10 state numbers) pushed by 4 actuators. Each episode
draws one of five contexts and keeps it for 100 steps. The context is observed. Contexts differ only in
how the actuators act (and, for the decoy, in noise).

- **R0, R1 (rich):** the operator uses all 4 actuators, with exploration noise SD 0.3. It is a mediocre
  controller (0.2 x the optimal gain).
- **T_wrong, T_right (thin):** a competent operator that uses actuators 1-2 only. Actuators 3-4 are logged
  with SD 0.01 (the primary rung of the density ladder). In T_wrong, actuators 3-4 push the opposite way.
  The two thin logs have nearly the same distribution, so no function of the data can tell them apart.
- **DECOY (rich data, 3x noise):** like R0, but transition noise SD 0.3. The critic should be accurate
  here. A signal that tracks noise flags the decoy; a signal that tracks error flags the thin contexts.

## Observation space

`Box(-inf, inf, shape=(15,), float32)`. No normalisation.

| Index | Name | Meaning | Logged range |
|---|---|---|---|
| 0, 2, 4, 6, 8 | p1 ... p5 | oscillator positions | every \|s_i\| <= 5 on 99.98-100% of rows (10 seeds) |
| 1, 3, 5, 7, 9 | v1 ... v5 | oscillator velocities | as above |
| 10-14 | code | one-hot of (R0, R1, T_wrong, T_right, DECOY) | exactly 0 or 1 |

An optional noisy continuous code (one-hot + N(0, code_sd^2) per episode, graft g7) exists and is off by
default. Row-level truth needs the exact one-hot; returns stay exact per episode with the noisy code.

## Action space

`Box(-inf, inf, shape=(4,), float32)`. Actions are never clipped (TD3's target clip uses max_action = inf),
so Q^pi is exact for every action. Nominal scale is [-1, 1].

| Index | Actuator | Pushes | Gain |
|---|---|---|---|
| 0 | a1 | velocities of oscillators 1 and 2 (v1, v2) | 0.085 each |
| 1 | a2 | v2, v3 | 0.085 each |
| 2 | a3 | v3, v4 | 0.085 each (x -1 in T_wrong) |
| 3 | a4 | v4, v5 | 0.085 each (x -1 in T_wrong) |

Without actuators 3-4, oscillators 4 and 5 are reached only through the weak coupling. Stage 0: 99.3% of
logged actions and 97.1% of the optimal policy's actions lie in [-1, 1]^4 (mix-weighted). The decoy alone
has 80% of its optimal actions inside.

## Transition dynamics

`s' = A s + B_c a + eps`, `eps ~ N(0, sd_c^2 I_10)`, with sd_c = 0.1 (0.3 in DECOY).

- `A = blockdiag(rho_j Rot(theta_j)) + C`.
- Radii rho = (0.98, 0.975, 0.97, 0.965, 0.96).
- Angles theta = (0.10, 0.20, 0.35, 0.55, 0.80) rad, which are periods of 62.8, 31.4, 18.0, 11.4 and 7.9
  steps.
- C is a seeded Gaussian coupling (seed 20261011, first draw kept) with spectral norm 0.02. Its largest
  entry is 0.0106.
- Eigenvalues of A: moduli 0.9817, 0.9729, 0.9694, 0.9651, 0.9549 (spectral radius 0.9817).
- `B` has 0.085 at (v_k, a_k) and (v_{k+1}, a_k). R1 scales actuators 1-2 by 0.6. T_wrong negates
  columns 3-4. T_right and DECOY equal R0.
- Exact identity: J_T_wrong(K12, D) = J_T_right(K12, -D) for any gain block D on actuators 3-4.

Stage 0 chose these numbers outcome-free, from closed-form targets. The design's defaults failed them
(see `runs/glq/stage0/stage0.json`).

## Rewards

`r = -(|s|^2 + R |a|^2)`, R = 1.6056 (R / gain^2 = 222). No reward noise. Discount gamma = 0.98.

The headline return J is the exact expected discounted return over the 100-step episode, from a covariance
recursion. It is always finite. The normalised score is NS = (J - J_BC) / (J_opt - J_BC) per context and for
the mixture: 0 is the per-context behaviour clone, 1 is the per-context optimum K*_c.

## Starting state

`s0 ~ N(0, I_10)`. Context shares: R0 0.30, R1 0.30, T_wrong 0.15, T_right 0.15, DECOY 0.10. By default
every split holds exactly these shares of its episodes, in a seeded random order.

## Episode end

Truncation only, at t = 100 (`terminated = False`, `truncated = True`). Logged done = 0, so TD3+BC
bootstraps through the time limit.

## Contexts

| Context | Share | B_c | Noise SD | Behaviour (logged) |
|---|---|---|---|---|
| R0 | 0.30 | B | 0.1 | a = -0.2 K*_c s + 0.3 n |
| R1 | 0.30 | B, actuators 1-2 x 0.6 | 0.1 | a = -0.2 K*_c s + 0.3 n |
| T_wrong | 0.15 | B, actuators 3-4 x -1 | 0.1 | a_12 = -K2 s + 0.1 n, a_34 = 0.01 n |
| T_right | 0.15 | B | 0.1 | a_12 = -K2 s + 0.1 n, a_34 = 0.01 n |
| DECOY | 0.10 | B | 0.3 | a = -0.2 K*_c s + 0.3 n |

K*_c is the discounted LQR optimum with 4 actuators. K2 is the optimum with actuators 1-2 only (zero rows
3-4).

## Datasets

Per seed: 1,000 episodes of 100 steps, split by episode.

- Train: 500 episodes, 50,000 rows.
- Calibration bank: 200 episodes, 5 rows each by `calibration/bank.stratified_bank`, 1,000 rows.
- Evaluation: 300 episodes, 30,000 rows, for row-level diagnostics only. Returns need no rows.

Variants, all sharing the same random numbers:

- Density ladder (graft g2): actuator-3-4 SD in the thin contexts at 0, 0.01 (primary), 0.03, 0.1, 0.3.
- Null world (R7 anchor): rich behaviour at 1.0 K* (expert).
- Retunes (pre-registered, at most two, one at a time): thin shares 0.25 each; T_wrong actuator-3-4 gain x 1.5.

## Reference scores

Exact 100-step J (and J_inf in brackets), from the Stage-0 world.

| Policy | R0 | R1 | T_wrong | T_right | DECOY |
|---|---|---|---|---|---|
| K = 0 | -183.29 (-195.20) | -183.29 (-195.20) | -183.29 (-195.20) | -183.29 (-195.20) | -602.66 (-708.76) |
| Behaviour clone (NS 0) | -162.44 (-172.38) | -169.02 (-179.61) | -152.75 (-161.52) | -152.75 (-161.52) | -536.79 (-625.89) |
| K2 (actuators 1-2 only) | -152.75 (-161.52) | -166.36 (-176.28) | -152.75 (-161.52) | -152.75 (-161.52) | -507.53 (-586.46) |
| K* (NS 1) | -137.66 (-145.26) | -149.85 (-158.46) | -137.66 (-145.26) | -137.66 (-145.26) | -459.01 (-527.42) |

- Gap J_opt - J_BC per context: 24.78, 19.16, 15.09, 15.09, 77.77. Mixture gap: 25.49.
- The decoy holds 30.5% of the mixture gap on 10% of episodes. The rich-context marks use R0 and R1.
- K*_T_right used in T_wrong loses 126.7, which is 8.40 g (g = 15.09), with sqrt(gamma) x radius 0.987.

## Ground truth

For a per-context linear policy a = -K_c s, Q^pi_c(s, a) = z^T H_c z + h_c with z = (s, a), from the
discrete Lyapunov equation (`lq_harness.quadratic_q`). The critic error e1 = Q1 - Q^pi, its action
gradient, the advantage error e_A and OR-D are exact at any row. State laws are Gaussian, so density
ratios are exact too.

TD3's target adds clipped smoothing noise (SD 0.2, clip 0.5) to the next action. So even a perfect TD3
critic converges to Q^pi + c_c, not Q^pi, with c_c = gamma var tr(H_aa) / (1 - gamma), about -13.3 here.
That is 0.15-0.21 of the median |Q^pi| in R0 and R1. Action gradients do not change. R2's level accuracy
therefore measures the error against Q^pi + c_c (`World.td3_level`); the error against Q^pi itself is
reported too. Since Amendment A1 both are reported, not gated. EXACT's Q tables include c_c as well, so its
lambda is on the host critic's scale.

## The acceptance gate

One process per (stage, seed). Every stage first verifies the pre-registration and refuses to run on any
mismatch. It checks the sidecar; the SHA-256 of `glq_acceptance.py`, `gapped_lq.py`, `glq_train.py`,
`glq_match.py` and of the reused repository files; the pass marks; and every other field the code
determines (seeds, sizes, alpha ladder, retunes, rungs, world, training settings). An edited file with a
recomputed sidecar is therefore refused. A pooled mark counts only when every seed contributes: a failed
seed fails the mark and is never dropped from the mean. Outputs go to `runs/glq/<first 16 hex of the
pre-registration's SHA-256>/`. No stage trains a BCA arm. BCA's own band (R5(iii)) needs
`--allow-bca-band`, which the approved scope does not give.

| Step | Stage | Trains? | What it checks |
|---|---|---|---|
| 0 | `prereg` | no | writes the pass marks, seeds, retunes, alpha ladder and code hashes |
| 1 | `build_checks` | no | Bellman identity, Monte Carlo vs closed form, recursions, Riccati, bitwise steps, oracle identities, unit suites |
| 2 | `stage0` | no | S1-S6, decoy noise, stability (closed form) |
| 3-4 | `r0r1` | no | R0 state richness and R1 thinness on the log |
| 5 | `critics`, `r2` | yes | the FQE critic's errors sit in T_wrong and the thin contexts; dose-response over the density ladder (HOST and accuracy reported) |
| 6 | `r3` | reads | errors cost return (host vs EXACT); picks the host alpha (2.5, then 5, then 10) |
| 7 | `r6` | reads | the host leaves the data (and R1(c)) |
| 8 | `r5world` | fits two small regressors | confounds, population band, ideal vs noise-only regressor |
| 9 | `level1`, `tierA`, `r4` | yes (oracle actors) | first-order screen; placed vs even caution at matched strength (up to 10 matching rounds) |
| 10 | `r7` | reads | power, anchors, null world, global-linear class, matching quality, null SHUF pairs |
| 11 | `tierB` | yes | co-trained confirmation (full path only) |
| 12 | `verdict` | no | the table and the stop rule |

Stop rule: if step 5, 6 or 9 fails, the next pre-registered retune is tried; after the last, the world is
rejected and BCA is never run in it. If step 8 fails, the world tests thin-data placement only.

## Amendment A1 (2026-10-12)

The gate's lean run on seeds 0-4 rejected the world (`outputs/glq/lean-v1/README.md`). The user then
re-registered the gate with the four changes below, and nothing else.

**This amendment is post hoc.** The new thresholds were chosen after seeing seeds 0-4. Those seeds keep
their original verdict: the world was rejected under the original pre-registration (`255369889b400c2b...`).
The amended gate is tested only on fresh seeds: 5-9 (lean), 5-14 (full) and 5-7 (density ladder). These
seeds are the confirmatory test. None of them has been through a training stage or an R2-R7 evaluation.
The only earlier outputs on any of them are data-only r0r1 checks of seeds 5-9, and every mark passed:

- both earlier pre-registrations (`218e1af1...` and `255369889b400c2b...`) ran them in the base world on
  2026-10-11, as steps 3-4 of their full path;
- this amendment's first draft (`12db2dd5...`) ran them in the base and thin_share worlds. The review
  replaced that draft before anything trained, because the draft's record said these seeds had never been
  run.

Those marks are unchanged by the amendment and say nothing about R2-R7. The new pre-registration records
the amendment: its id, date, the original pre-registration's SHA-256, each change with its old and new
rule, the statement that it is post hoc, and these earlier outputs (`prior_outputs_on_fresh_seeds`). No
BCA arm and no BCA band fit is approved.

Unchanged: the world, Stage 0, the retunes (wrong_gain still fails Stage 0, so it cannot be used), the
training settings, every other pass mark and the stop rule.

### 1. R2 (step 5): gated on the FQE critic, on where its error sits

- **Old rule.** Both critics (HOST at the host alpha, and FQE) had to pass three marks in at least 4 of 5
  seeds. Mean E in T_wrong at least 3x the rich contexts' (R0, R1). The thin contexts carry at least 60% of
  sum E^2. In the rich contexts, the median level error |Q1 - Q_td3| / |Q^pi| is at most 5%. The
  density-ladder marks were judged on both critics.
- **New rule.** Only the FQE critic is gated, and only on concentration:
  - (a) mean E in T_wrong at least 3x the rich mean (same threshold as before);
  - (b) the thin contexts' share of sum E^2 (all 5 contexts pooled, as before) at least 1.5x their share
    of rows. That is 0.45 in the base world (rows 0.30) and 0.75 in thin_share (rows 0.50).

  Both must hold in at least 80% of seeds (4 of 5 on the lean path). The density-ladder marks stay, judged
  on FQE only.
- **Reported, not gated.** The HOST critic's R2 values. Accuracy, for both critics, as two numbers:
  - the level error |e1| / |Q^pi|, as before;
  - a new slope measure: the rich-context median of |H_aa^-1 (grad_a Q1 - grad_a Q^pi)| / 2 at
    (s, pi(s)). H_aa is the action block of the exact Q^pi of the policy the diagnostics evaluate. The
    number says how far the critic's error moves the action the critic recommends. For scale, the rich
    logging noise SD is 0.3.
- **What seeds 0-4 showed.** FQE's error had the designed pattern: E in T_wrong 3.4-6.2x the rich
  contexts', thin share of sum E^2 38-80%. HOST's did not: 1.0-1.6x and 13-37%. Every critic missed the 5%
  accuracy mark by far (0.46-1.15), mostly through a level offset. The slope is no better. Measured
  afterwards with the new slope measure on the saved rows of seeds 0-4, the critic error moves the
  recommended action by 1.7-3.6 units in the rich contexts of the base world (1.8-4.2 in thin_share; FQE,
  and HOST at every alpha), against logging noise 0.3.

### 2. R3 (step 6): where the return shortfall sits

- **Old rule.** The thin contexts hold at least 60% of the host-vs-EXACT shortfall.
- **New rule.** The thin contexts' share of that shortfall is at least 1.5x their mix share: 0.45 in the
  base world, 0.75 in thin_share. The other three R3 marks, the alpha ladder (2.5, 5, 10) and the
  80%-of-seeds rule are unchanged. In the base world the new mark is looser than before; in thin_share it
  is stricter (0.75 against 0.60).
- **What seeds 0-4 showed.** This was the mark that rejected the world. At alpha 10 the thin share was
  0.40-0.60 (base) and 0.56-0.74 (thin_share), while the other three marks passed. A fixed 60% takes no
  account of the thin contexts' share of the episodes (0.30 or 0.50).

### 3. Strength matching (R4 and R7)

- **Old rule.** At most 5 matching rounds. R4 was judged at S2 only if OR-wrong and OR-thin were matched
  at S2 in every seed, otherwise at S1. A seed with an unmatched family failed every pooled mark.
- **New rule.** At most 10 matching rounds. R4 is judged at S2 when OR-wrong and OR-thin are both matched
  there in at least 80% of seeds (4 of 5 on the lean path). Otherwise it is judged at S1 under the same
  rule. If neither level qualifies, R4 fails. At the chosen level, a seed whose family is unmatched counts
  as a non-positive seed; it is never dropped silently. It is left out only of the means and t-tests, and
  the output lists the seeds left out. The R4(4) and R7(b) anchors use the same level. R7(d)'s
  matching-quality mark is unchanged.
- **Two rules the amendment leaves as they were.** A seed whose Tier A run failed (its HOST critic
  stopped early) still fails every pooled mark. The null world (R7(c)) and the global-linear control keep
  their own level rule (S2 only when every seed matched OR-wrong there) and still need every seed.
- **What seeds 0-4 showed** (Tier A at alpha 10, information only). Matching hit its 5-round cap with 4-8
  unmatched pairs per seed. OR-wrong was matched at S2 in 4 of 5 seeds, so R4 fell back to S1. There the
  R4(4) anchors missed: 0.045 and 0.028 against 0.05.

### 4. Seeds

- **Old.** Lean 0-4, full 0-9, density ladder 0-2.
- **New.** Lean 5-9, full 5-14, density ladder 5-7. None has been through a training stage. The only
  earlier outputs are the data-only r0r1 checks of seeds 5-9 listed above.
- **Why.** Seeds 0-4 informed the new thresholds, so they cannot test them.

## How to run

```bash
cd /path/to/bca
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
GATE="python experiments/signal/glq_acceptance.py"

# every command of a path, in a safe order
$GATE plan --path lean
$GATE plan --path full
```

### Lean path (5 seeds, Tier A only)

Seeds 5-9 (Amendment A1; seeds 0-4 belong to the superseded pre-registration). The density ladder uses
seeds 5-7.

```bash
$GATE prereg                                   # once; never overwritten without --force
$GATE build_checks                             # steps 1-4: no training
$GATE stage0 --search
for k in 5 6 7 8 9; do $GATE r0r1 --seeds $k; done
# --- approval boundary: training below ---
for k in 5 6 7 8 9; do for m in HOST FQE EXACT; do echo "critics --seeds $k --mode $m"; done; done \
  | xargs -P 2 -L 1 $GATE
for r in rung0 rung0.03 rung0.1 rung0.3; do for k in 5 6 7; do for m in HOST FQE; do
  echo "critics --seeds $k --mode $m --variant $r"; done; done; done | xargs -P 2 -L 1 $GATE
$GATE r3 --path lean --alpha 2.5
# if r3 fails: critics --mode HOST and EXACT --alpha 5 for every seed, r3 --alpha 5; then alpha 10
$GATE r2 --path lean
for k in 5 6 7 8 9; do for s in r6 r5world level1 tierA; do echo "$s --seeds $k --path lean"; done; done \
  | xargs -P 2 -L 1 $GATE
for k in 5 6 7 8 9; do $GATE critics --seeds $k --mode HOST --variant null --alpha auto --path lean; done
for k in 5 6 7 8 9; do for v in null global_linear; do echo "tierA --seeds $k --variant $v --path lean"; done; done \
  | xargs -P 2 -L 1 $GATE
$GATE r4 --path lean
$GATE r7 --path lean
$GATE verdict --path lean
```

### Full gate (10 seeds plus Tier B)

The same commands with seeds 5-14 and `--path full`, then Tier B and the verdict:

```bash
for k in $(seq 5 14); do for a in EVEN OR-wrong OR-thin SHUF0 ANTI; do
  echo "tierB --seeds $k --arm $a --path full"; done; done | xargs -P 2 -L 1 $GATE
$GATE verdict --path full
```

Per-seed stages resume: a stage whose output exists with a matching sidecar is skipped (`--force` redoes
it). Pooled stages (`r2`, `r3`, `r4`, `r7`) and the verdict always recompute from the per-seed outputs.
A retuned world reruns the gate from Stage 0 with `--retune thin_share` (or `wrong_gain`).

## Cost per stage

Process times on 2 pinned CPUs (OMP_NUM_THREADS = 2). Four CPUs run two processes at once, so wall time is
about half the process time.

| Stage | Cost | Source |
|---|---|---|
| prereg | 1.0 s | measured (A3) |
| build_checks | 78-80 s (unit suites 67-68 s) | measured (A3, two runs) |
| stage0 --search | 182-190 s (3.4-4.7 s without the search) | measured (A3, two runs) |
| r0r1 | 0.7-1.0 s per seed; 9.4 s wall for 10 processes | measured (A3) |
| critics HOST | 257 s per 100k updates | measured (A2 pilot, full run) |
| critics FQE / EXACT | 220 s / 212 s | extrapolated from 5,000 timed updates (A2 pilot) |
| r2, r3, r4, r7, verdict | under 1 s | measured (A3 tests) |
| r6 | about 2 s per seed | measured components (A3: Kish ESS 1.1 s) |
| r5world | about 60 s per seed | measured components (A3: band 25.4 s per snapshot x 2, regressors 7.3 s) |
| level1 | 29 s per seed | measured (A2 pilot) |
| tierA main | 0.30 h per seed | ladder measured on 2,000 steps; matching assumes 3 rounds (A2 pilot). The lean run took about 21 min per seed with the old 5-round cap; Amendment A1 allows up to 10 |
| tierA null / global_linear | 0.06 h per seed each | extrapolated (A2 pilot) |
| tierB | 257 s per arm, 5 arms per seed | from the HOST run (A2 pilot) |

Path totals in process-hours (extrapolated from the rows above):

| Path | Critics | Density ladder | Null HOST | r5world, r6, level1 | Tier A | Tier A controls | Tier B | Total | Alpha ladder, worst case |
|---|---|---|---|---|---|---|---|---|---|
| Lean (5 seeds) | 0.96 | 1.59 | 0.36 | 0.13 | 1.49 | 0.59 | 0 | about 5.2 | +1.3 |
| Full (10 seeds) | 1.91 | 1.59 | 0.71 | 0.25 | 2.98 | 1.18 | 3.57 | about 12.3 | +2.6 |

The worst alpha case reruns HOST and EXACT at alpha 5 and 10 (FQE does not depend on alpha).

## Status

**Current pre-registration (Amendment A1):**
`8528a61385157a764e53b9b37fad764ed53bdc0b1089c9755344454429a81f32`, with outputs in
`runs/glq/8528a61385157a76/` (runs/ is gitignored). It supersedes `255369889b400c2b...`, whose outputs in
`runs/glq/255369889b400c2b/` are kept, and whose lean verdict on seeds 0-4 (the world is rejected) stands
for those seeds. It also replaces A1's first draft `12db2dd5...` (outputs kept in
`runs/glq/12db2dd5f33bbad4/`). The review corrected the draft's record, which had said seeds 5-14 were
never run; no rule changed, and nothing trained under the draft. Under A1 only the no-training stages
have run so far, on the fresh seeds 5-9. Their numbers are identical under both versions, and every mark
passes:

- Unit suites: test_gapped_lq 32, test_glq_train 34, test_glq_acceptance 30 tests, 0 failures.
- Build checks: Bellman residual 1.3e-15 (control 0.0044); max Monte Carlo |z| 2.02 (3,000 episodes per
  context); recursions agree to 5.6e-16 and 7.2e-15; Riccati gradient ratio 9.0e-9; 4 of 4 bitwise checks;
  13 of 13 oracle identities; unit suites (B7) 66 tests, 0 failures.
- Stage 0, base world: S1 max 0.459 (mark <= 0.5); S2 0.178 (>= 0.15); S3 8.40 g at radius 0.987 (>= 2 g);
  S4 0.971 (>= 0.95); decoy noise ratio 6.75 (>= 3); S6 0.422 (>= 0.30); the search reproduces the spec.
  thin_share: every target passes (S2 0.313).
- R0/R1, seeds 5-9, base world: participation ratio 9.06-9.30 (>= 6); transient share 0.36-0.41 (>= 0.30);
  IACT 16.8-17.3 steps (10-50); rows with every |s_i| <= 5: 99.98-100%; rich action eigenvalues >= 0.088
  (>= 0.05); thin actuator-3-4 SD 0.99-1.02 x every rung (rank 2 at rung 0); thin share 0.300.
- R0/R1, seeds 5-9, thin_share: participation ratio 9.10-9.36; transient share 0.35-0.39; IACT
  16.6-17.0; rows within 5: 99.98-100%; rich eigenvalues >= 0.088; thin SD 0.99-1.01 x every rung; thin
  share 0.500.

**Verdict under A1 (2026-10-12): the world is rejected again.** Steps 1–4 pass; R2 and R3 fail in the base world and
in thin_share, and wrong_gain still fails Stage 0, so no version is left. R3's thin share of the shortfall is
0.33–0.55 against 0.45 (base) and 0.51–0.71 against 0.75 (thin_share), with at most 3 of 5 seeds passing at any
alpha. See `outputs/glq/lean-a1/README.md`. BCA was never run in this world.

**History.** The adversarial review (A-review) changed three hashed files before any training: R2's
accuracy reference, EXACT's lambda scale, the pre-registration check and the pooled seed rule. The earlier
pre-registration `218e1af1...` is void. Its successor,
`255369889b400c2b93d5c92ca2e861c71d39554bf490b3c9489a5661be177baf`, ran the lean gate on seeds 0-4, and
the world was rejected (`outputs/glq/lean-v1/README.md`). Its steps
0-4 (10 seeds) had passed: participation ratio 9.06-9.75, transient share 0.36-0.41, IACT 16.7-17.7 steps,
S4 0.993 logged and 0.971 optimal.

Retune worlds, Stage 0: thin_share passes every target (S2 0.313). **wrong_gain fails S3 and stability:**
K*_T_right in that T_wrong has sqrt(gamma) x radius 1.0023, so the transferred policy diverges. Retune 2
cannot be used; Amendment A1 leaves this unchanged.
