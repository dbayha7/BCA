"""Gapped LQ acceptance gate (option A, part A3): the pre-registration and its hash check, the ordered gate stages
R0-R7 with the judge's grafts, and the verdict.

Design: option A of the world audit (scratchpad design_glq.json) and the judge's ordered acceptance gate (judge.json,
steps 0-12) with grafts g1 (decoy context), g2 (density ladder), g3 (ideal-regressor gate), g4 (OR-D, e_A), g5
(Level-1 screen), g6 (blind matching, null SHUF pairs), g7 (noisy code / gain-scheduled class, off) and g8 (R5
confound Spearmans within each context and pooled). The world and its exact truth are experiments/signal/gapped_lq.py
(A1); co-training, snapshots, exact critic errors, Tier A and the Level-1 screen are glq_train.py and glq_match.py
(A2). This module applies the pass marks. It edits no repository file; lq_harness, placement_harness and algorithms/
are imported and called. The environment card and the commands are experiments/signal/GAPPED_LQ.md.

PRE-REGISTRATION (stage `prereg`, gate step 0). Writes experiments/signal/glq_preregistration.json (strict JSON, sorted
keys) and its .sha256 sidecar: every pass mark (PASS_MARKS, the numbers the code reads), the seed-count rule of each
pooled mark, the lean (5) and full (10) seed lists, the density-ladder seeds and rungs, the primary rung (0.01), the
decoy settings, the retune list (RETUNES, at most two, applied one at a time to the Stage-0 world, in order), the
alpha ladder (2.5, then 5, then 10), every size and training setting (GateConfig and glq_train.TrainSettings), the
world spec and its SHA-256, and the SHA-256 of glq_acceptance.py, gapped_lq.py, glq_train.py and glq_match.py
(CODE_FILES; glq_match.py is added to the three the task names because Tier A's blind matching runs it) and of the
reused repository files (DEPENDENCY_FILES). Every later stage calls load_prereg first and refuses to run when the
file's SHA-256 differs from its sidecar, when any recorded code hash differs from the file on disk, when the recorded
pass marks differ from PASS_MARKS, when any other code-determined field (config, seeds, alpha ladder, retunes, rungs,
world, training settings: everything but PREREG_VOLATILE) differs from what prereg_payload writes for the recorded
preset (so an edit with a recomputed sidecar is refused), when the hashed dependency list differs from
DEPENDENCY_FILES, or (unless --allow-dependency-change, which is recorded) when a dependency changed;
every stage function re-checks the file's digest and the code hashes on entry (guard). Outputs go to
runs/glq/<first 16 hex of the pre-registration's SHA-256>/..., every JSON records the full digest, and a per-seed stage
skips work whose output exists with a matching sidecar and digest (resumable; --force redoes it); the pooled stages
(r2, r3, r4, r7) always recompute from the per-seed outputs, and every later stage and the verdict evaluate pooled
results (and the host alpha) live, so nothing pooled is ever stale. The pre-registration is never overwritten without
--force.

STAGES (one process per (stage, seed); `plan --path lean|full` prints every command in order). Training starts at
`critics` (the approval boundary). No stage trains a BCA arm. R5's BCA band (R5(iii)) is implemented (bca_band) but
refuses to run without --allow-bca-band, which the approved scope does not give.
 1 build_checks  No training. B1 Bellman identity V = r + gamma E V per context (K*, K2, behaviour, 0) to 1e-12
                 relative, with a negative control (another policy's (P, h) > 1e-3); B2 Monte Carlo against the closed
                 form, |z| <= 3 per context (the behaviour clone's and the noisy behaviour's 100-step J); B3 forward vs
                 backward finite-horizon recursion <= 1e-10 and 4000 steps vs Lyapunov <= 1e-9; B4 Riccati gradient at
                 K* <= 1e-5 x at 0.5 K*; B5 harness_update with every hook off, harness_update with a bc_multiplier
                 and Tier A's actor-only loop (against td3_bc_update with the critic, its target and the target actor
                 restored) are td3_bc_update bit for bit; B6 oracle-weight identities on a TierA instance (untrained
                 critic); B7 the unit suites test_gapped_lq and test_glq_train (subprocess; --no-unittests skips).
 2 stage0        gapped_lq.stage0_full on the (retuned) world: S1 <= 0.5, S2 >= 0.15, S3 >= 2 g and discounted-stable,
                 S4 >= 0.95, decoy aleatoric ratio >= 3, S5 (R0 in closed form), S6 >= 0.30, every policy stable.
                 --search reruns gapped_lq.stage0_search and requires that it returns STAGE0_SPEC.
 3-4 r0r1        Per seed, log only (training split): R0 participation ratio >= 6 in every context, pooled transient
                 share >= 0.30, IACT in [10, 50], >= 99% of rows with every |s_i| <= 5; R1(a) rich action-covariance
                 eigenvalues >= 0.05, thin actuator-3-4 SD within 5% of the rung (rank 2 at rung 0); R1(b) thin share
                 = the world's thin mix share +/- 0.03. Every rung of the ladder is checked. Required in every seed.
 -- approval boundary: every stage below reads or trains critics or oracle actors (no BCA arm) --
 5 critics       Per (seed, mode): glq_train.cotrain HOST / FQE / EXACT (100k updates, snapshots 25/50/100%, exact
                 diagnostics on the evaluation split). Variants: main (primary rung), rung<r> (density ladder, HOST and
                 FQE, ladder seeds, alpha = the ladder's first), null (expert rich behaviour f = 1, HOST only, R7(c)).
   r2            Pooled. Per seed and critic (HOST at the host alpha, FQE), 100% snapshot: mean E in T_wrong >= 3x the
                 rich (R0, R1) mean; thin share of sum E^2 >= 0.60; rich median |Q1 - Q_td3| / |Q^pi| at (s, pi(s))
                 <= 0.05, Q_td3 = Q^pi + c_c the fixed point of TD3's smoothed target (gapped_lq.World.td3_level;
                 against Q^pi itself a perfect TD3 critic reads 0.16-0.18 here, so that reading is reported). Pass:
                 all three in >= ceil(0.8 n) seeds for both critics. Ladder (ladder seeds, seed means of the thin rows'
                 mean E, HOST and FQE each): E falls between consecutive rungs in >= 3 of 4 steps, and at rung 0.3 the
                 thin / rich ratio is within [1 / 1.5, 1.5]. Reported: mean E T_wrong / T_right.
 6 r3            Pooled, per alpha. Per seed: w_Tw x (J_Tw(K_host with rows 3-4 zeroed) - J_Tw(K_host)) >= 0.05 x the
                 mixture gap sum_c w_c (J_opt,c - J_BC,c); NS_mix(host) <= NS_mix(EXACT) - 0.10; thin contexts hold >=
                 60% of the shortfall sum_c w_c (J_EXACT,c - J_host,c); rich (R0, R1) host NS >= 0.30. Pass: all four
                 in >= ceil(0.8 n) seeds. The host alpha is the smallest alpha of the ladder whose r3 passes; stages
                 below resolve --alpha auto to it and refuse when R3 failed at every alpha (the stop rule).
 7 r6            Per seed, host at 100%: (a) |pi(s)_34| >= 0.3 on >= 30% of thin evaluation states; (b) >= 20% of rich
                 (R0, R1) states with |pi(s) - mu_beh(s)| / 0.3 > 2; (c) Kish ESS fraction of the exact state density
                 ratio p_pi / p_log (100-step mixtures of the Gaussian state laws, on the evaluation states) in [0.05,
                 0.8] in at least one context; (d) the BC start (least-squares clone) has <= 5% on (a) and (b); R1(c)
                 >= 20% of the host's thin actions below 1% of the behaviour's peak density (floor SD 0.01). Pass: in
                 >= ceil(0.8 n) seeds.
 8 r5world       Per seed, host 100% snapshot (50% reported), evaluation rows: (i) |Spearman(E, x)| <= 0.5 for x in
                 (exact aleatoric target SD, |s|, |pi(s) - mu_beh(s)|), within each context and pooled; mean E in
                 T_right <= 0.5 x T_wrong; (ii) population band (conditional RMS of t - Q1 at the logged (s, a), Monte
                 Carlo over the exact next-state Gaussian and the smoothing noise, 256 draws per row, target networks):
                 Spearman with E >= 0.3, AUROC T_wrong vs rich >= 0.7, AUROC thin vs decoy >= 0.7; partial Spearman of
                 the band with E given the aleatoric SD reported; (g3) a Calibrator-shaped MLP fitted to |t - Q1| on the
                 training rows against one fitted to |t* - Q^pi| (exact target, exact critic): AUROC for high-error rows
                 (E >= its 75th percentile) of the first >= the second + 0.10. Pass: each in >= ceil(0.7 n) seeds. A
                 failure does not reject the world: it then tests thin-data placement only.
 9 level1        Per seed: glq_train.level1_screen at the host's 50% snapshot (Tier A's critic). Screen, not a stop:
                 OR-wrong's first-order gain minus EVEN's at m = 2 > 0 in >= ceil(0.8 n) seeds.
   tierA         Per (seed, variant): glq_train.tier_a on the 50% snapshot (blind record and its hashes written before
                 any return). Variants: main (every family), null (null critic and world; EVEN, OR-wrong, OR-thin) and
                 global_linear (main critic; the 40-number class; EVEN, OR-wrong, OR-thin).
   r4            Pooled. Level S2 (m = 2) when OR-wrong and OR-thin are matched there in every seed, else S1. (1) mean
                 NS(OR-wrong) - NS(EVEN) >= 0.05, one-sided paired t p < 0.01, >= ceil(0.8 n) seeds positive; (2) mean
                 NS(OR-thin) - NS(EVEN) >= 0.03, p < 0.05; (3) mean best OR-thin - best EVEN >= 0.02 and mean best
                 OR-wrong - NONE >= 0.03; (4) OR-thin - SHUF >= 0.05 and SHUF - ANTI >= 0.05 (SHUF = the SHUF
                 families' mean). Reported: CAP-wrong - EVEN at the level; below 0.02 the BCA test uses P1L placement.
 10 r7           Pooled. (a) MDE of the paired difference in (1) at 10 seeds (one-sided alpha 0.05, power 0.8; from
                 the observed SD) <= 0.02, else the seeds needed (<= 20); (b) OR-wrong - SHUF >= 0.05 and SHUF - ANTI
                 >= 0.05; (c) null world |mean NS(OR-wrong) - NS(EVEN)| < 0.02; global-linear mean difference < 0.02;
                 (d) |S / S* - 1| <= 5% for >= 90% of (family, target, seed) matches (CAP families excluded,
                 unreachable counts as a miss); (e) not applicable (no Monte Carlo actor class in this gate); null
                 SHUF pairs: <= 1 of the C(n_shuf, 2) pairs with |mean| > 2 SE.
 11 tierB        Per (seed, arm): co-trained HOST with the arm's static multiplier at its Tier A knob (the r4 level);
                 arms EVEN, OR-wrong (gated), OR-thin, SHUF0, ANTI (reported). Pass: NS(OR-wrong) - NS(EVEN) > 0 and >=
                 0.5 x the pooled Tier A mean of (1), in >= ceil(0.7 n) seeds.
 12 verdict      verdict.json and verdict.md (requirement, pass mark, measured, pass / fail, seeds) and the stop rule:
                 steps 1-4 failing is a build failure; step 5, 6 or 9 failing sends the gate to the next retune (or
                 rejects the world after the last); 7, 10 or 11 failing means not accepted; 8 failing restricts the
                 world to thin-data placement. Only when steps 1-11 pass may BCA's band be measured, under a separate
                 approval.

Seed counts: a mark "in >= f of 10 seeds" is applied as >= ceil(f n) of the n seeds of the path (lean: 4 of 5 for both
f = 0.8 and f = 0.7). R7(a) at the lean path projects the observed paired SD to 10 seeds. A pooled mean (R4 (1)-(4),
R7 (b), (c), global-linear) counts only when every seed of the path contributes: a seed whose run failed, or whose
family is unmatched at the level, fails the mark rather than leaving the mean. R7(c) is gated in the null world's own
NS units (its mixture gap is the thin contexts' only, 5.6x smaller than the main world's) and also reported in the main
world's units.

python experiments/signal/glq_acceptance.py prereg
python experiments/signal/glq_acceptance.py plan --path lean
python experiments/signal/glq_acceptance.py <stage> [--seeds 0 1 ..] [--mode HOST] [--variant main] [--alpha auto]
    [--path lean|full] [--retune base|thin_share|wrong_gain] [--force] [--prereg PATH] [--runs DIR]
Tests: JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_glq_acceptance
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
from scipy import special, stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

import algorithms.td3_bc as BASE  # noqa: E402
import algorithms.td3_bc_bca as P  # noqa: E402
from calibration.network import Calibrator  # noqa: E402
from calibration.reference import WBCPConfig, initial_reference, positive_scale  # noqa: E402
from experiments.signal import gapped_lq as G  # noqa: E402
from experiments.signal import glq_train as T  # noqa: E402
from experiments.signal import lq_harness as LQ  # noqa: E402
from experiments.signal import placement_harness as PH  # noqa: E402

DS, DA, NC, OBS_DIM = G.DS, G.DA, G.NC, G.OBS_DIM
PREREG_DEFAULT = ROOT / "experiments" / "signal" / "glq_preregistration.json"
RUNS_DEFAULT = ROOT / "runs" / "glq"
CODE_FILES = ("experiments/signal/glq_acceptance.py", "experiments/signal/gapped_lq.py",
              "experiments/signal/glq_train.py", "experiments/signal/glq_match.py")
DEPENDENCY_FILES = ("algorithms/td3_bc.py", "algorithms/td3_bc_bca.py", "experiments/signal/lq_harness.py",
                    "experiments/signal/placement_harness.py", "calibration/network.py", "calibration/reference.py",
                    "calibration/wbcp.py", "calibration/bank.py", "calibration/dose.py", "runtime/networks.py")
UNIT_SUITES = ("experiments.signal.test_gapped_lq", "experiments.signal.test_glq_train")
MODES = ("HOST", "FQE", "EXACT")
TIER_A_VARIANTS = ("main", "null", "global_linear")
RETUNES = {"base": {}, "thin_share": dict(mix=(0.20, 0.20, 0.25, 0.25, 0.10)), "wrong_gain": dict(wrong_gain=1.5)}
RETUNE_ORDER = ("thin_share", "wrong_gain")
RETUNE_TEXT = {"thin_share": "each thin context's episode share 0.15 -> 0.25 (mix R0 0.20, R1 0.20, T_wrong 0.25, "
                             "T_right 0.25, DECOY 0.10)",
               "wrong_gain": "T_wrong's actuator-3-4 gain x 1.5 (WorldSpec.wrong_gain = 1.5)"}
TAG_R5_BAND, TAG_R5_FIT, TAG_BCA = 41, 42, 43  # np.random.default_rng([cfg.train_seed, rep, TAG, ...])
BCA_BAND_OFF = ("R5(iii), BCA's own band, is switched off: pass --allow-bca-band only under a separate approval "
                "(the approved scope fits no BCA band)")
TRAINING_STAGES = ("critics", "tierA", "tierB", "r5world")

# ---------------------------------------------------------------------------------------------------------
# Pass marks (pre-registered; the code reads these numbers)

PASS_MARKS = {
    # step 1: build checks (no training)
    "B1": dict(step=1, name="Bellman identity per context", max=1e-12, control_min=1e-3,
               mark="max relative residual of V = r + gamma E V <= 1e-12 (K*, K2, behaviour, K = 0); another "
                    "policy's (P, h) gives > 1e-3"),
    "B2": dict(step=1, name="Monte Carlo against the closed form", max_abs_z=3.0,
               mark="|z| <= 3 per context: 100-step J of the behaviour clone and of the noisy behaviour"),
    "B3": dict(step=1, name="Finite-horizon recursion", max_forward_backward=1e-10, max_lyapunov=1e-9,
               mark="forward vs backward recursion <= 1e-10 relative; 4000 steps vs Lyapunov <= 1e-9"),
    "B4": dict(step=1, name="Riccati optimum gradient", max_ratio=1e-5,
               mark="max |grad J_inf| at K* <= 1e-5 x at 0.5 K*, per context"),
    "B5": dict(step=1, name="Harness steps equal td3_bc_update",
               mark="bit for bit: harness_update hooks off (it 1 and 2), with a bc_multiplier, and Tier A's loop "
                    "against td3_bc_update with the critic restored"),
    "B6": dict(step=1, name="Oracle-weight identities",
               mark="OR-wrong, OR-thin, ANTI, EVEN, SHUF multisets, knob 0 = NONE, CAP in [1, 1.5), OR-D mean 1, "
                    "OR-D = [-e_A]_+"),
    "B7": dict(step=1, name="Unit test suites", mark="test_gapped_lq and test_glq_train: 0 failures and 0 errors"),
    # step 2: Stage 0 (closed form)
    "S1": dict(step=2, name="Rich behaviour is mediocre", max=0.5,
               mark="(J(f K*) - J(0)) / (J(K*) - J(0)) <= 0.5 in every rich context"),
    "S2": dict(step=2, name="Thin contexts hold room", min=0.15, mark="thin mix share x g >= 0.15 of the total gap"),
    "S3": dict(step=2, name="Transfer is costly in T_wrong", min=2.0,
               mark="K*_T_right in T_wrong: discounted-stable and loses >= 2 g against the behaviour"),
    "S4": dict(step=2, name="Action scale", min=0.95, mark=">= 95% of logged and of optimal actions in [-1, 1]^4"),
    "DECOY": dict(step=2, name="Decoy noise", min=3.0, mark="decoy median aleatoric target SD >= 3x the rich median"),
    "S5": dict(step=2, name="R0 in closed form", mark="the R0 marks on the exact behaviour state laws"),
    "S6": dict(step=2, name="Ideal transferred policy leaves the log", min=0.30,
               mark="|K*_Tr,34 s| >= 0.3 on >= 30% of thin behaviour states"),
    "STAB": dict(step=2, name="Every policy used is stable", mark="sqrt(gamma) rho(A - B K) < 1 for all of them"),
    # step 3: R0
    "R0_pr": dict(step=3, name="R0 participation ratio", min=6.0, mark=">= 6.0 of 10 in every context, every seed"),
    "R0_transient": dict(step=3, name="R0 transient rows", min=0.30, mark="pooled transient share >= 0.30"),
    "R0_iact": dict(step=3, name="R0 autocorrelation time", lo=10.0, hi=50.0, mark="pooled IACT in [10, 50] steps"),
    "R0_scale": dict(step=3, name="R0 state scale", bound=5.0, min=0.99, mark=">= 99% of rows with every |s_i| <= 5"),
    # step 4: R1
    "R1a_rich": dict(step=4, name="R1(a) rich actions explore", min=0.05,
                     mark="all 4 action-covariance eigenvalues >= 0.05 in R0, R1, DECOY"),
    "R1a_thin": dict(step=4, name="R1(a) thin actions at the rung", rel_tol=0.05, rank_rel=1e-12,
                     mark="thin actuator-3-4 SD within 5% of the rung (rung 0: rank 2 and exact zeros), every rung"),
    "R1b": dict(step=4, name="R1(b) thin share", tol=0.03,
                mark="thin share of training rows = thin mix share +/- 0.03"),
    # step 5: R2
    "R2_ratio": dict(step=5, name="R2 error sits in T_wrong", min=3.0, seed_frac=0.8,
                     mark="mean E in T_wrong >= 3x the rich mean (HOST and FQE)"),
    "R2_thin_E2": dict(step=5, name="R2 thin share of E^2", min=0.60, seed_frac=0.8,
                       mark="thin contexts carry >= 60% of sum E^2"),
    "R2_rel": dict(step=5, name="R2 rich critic accurate", max=0.05, seed_frac=0.8, reference="td3_fixed_point",
                   mark="rich median |Q1(s, pi(s)) - Q_td3(s, pi(s))| / |Q^pi| <= 5%, Q_td3 = Q^pi + c_c the fixed "
                        "point of TD3's smoothed target (a perfect critic reads 0; against Q^pi alone it reads "
                        "0.16-0.18); the deterministic-reference value is reported"),
    "R2_ladder_falls": dict(step=5, name="R2 dose-response", min=3,
                            mark="thin mean E falls in >= 3 of the 4 rung steps (HOST and FQE)"),
    "R2_ladder_top": dict(step=5, name="R2 dose-response top rung", factor=1.5,
                          mark="at rung 0.3, thin / rich mean E within [1/1.5, 1.5]"),
    # step 6: R3
    "R3_cost": dict(step=6, name="R3 off-support cost", min=0.05, seed_frac=0.8,
                    mark="w_Tw x off-support cost in T_wrong >= 0.05 x the mixture gap"),
    "R3_ns_gap": dict(step=6, name="R3 host below EXACT", min=0.10, seed_frac=0.8, mark="NS(host) <= NS(EXACT) - 0.10"),
    "R3_thin_shortfall": dict(step=6, name="R3 shortfall in thin contexts", min=0.60, seed_frac=0.8,
                              mark=">= 60% of the host-vs-EXACT shortfall in the thin contexts"),
    "R3_rich_ns": dict(step=6, name="R3 critic useful where rich", min=0.30, seed_frac=0.8,
                       mark="host NS in the rich contexts (R0, R1) >= 0.30"),
    # step 7: R6 and R1(c)
    "R6a": dict(step=7, name="R6(a) thin off-support use", threshold=0.3, min=0.30, seed_frac=0.8,
                mark="|pi(s)_34| >= 0.3 on >= 30% of thin states"),
    "R6b": dict(step=7, name="R6(b) rich shift", distance=2.0, min=0.20, seed_frac=0.8,
                mark=">= 20% of rich states beyond 2 logged-noise SDs"),
    "R6c": dict(step=7, name="R6(c) Kish ESS", lo=0.05, hi=0.8, seed_frac=0.8,
                mark="ESS fraction in [0.05, 0.8] in at least one context"),
    "R6d": dict(step=7, name="R6(d) BC start does not shift", max=0.05, seed_frac=0.8,
                mark="BC start: <= 5% on (a) and on (b)"),
    "R1c": dict(step=7, name="R1(c) host leaves thin support", density=0.01, floor=0.01, min=0.20, seed_frac=0.8,
                mark=">= 20% of the host's thin actions below 1% of the behaviour's peak density"),
    # step 8: R5 world gates
    "R5_confound": dict(step=8, name="R5(i) confounds", max_abs=0.5, seed_frac=0.7,
                        mark="|Spearman(E, aleatoric SD / |s| / distance)| <= 0.5, each context and pooled"),
    "R5_tright": dict(step=8, name="R5(i) T_right below T_wrong", max=0.5, seed_frac=0.7,
                      mark="mean E in T_right <= 0.5 x T_wrong"),
    "R5_band_spearman": dict(step=8, name="R5(ii) band tracks E", min=0.3, seed_frac=0.7,
                             mark="Spearman(population band, E) >= 0.3"),
    "R5_auroc_wrong_rich": dict(step=8, name="R5(ii) band finds T_wrong", min=0.7, seed_frac=0.7,
                                mark="AUROC of the band, T_wrong vs rich rows >= 0.7"),
    "R5_auroc_thin_decoy": dict(step=8, name="R5(ii) band is not noise", min=0.7, seed_frac=0.7,
                                mark="AUROC of the band, thin vs decoy rows >= 0.7"),
    "R5_ideal": dict(step=8, name="R5 ideal regressor beats noise", min_margin=0.10, quantile=0.75, seed_frac=0.7,
                     mark="AUROC(high-error rows) of the |t - Q1| regressor >= the noise-only one + 0.10"),
    # step 9: Level-1 screen and R4 Tier A
    "L1": dict(step=9, name="Level-1 screen (not a stop)", min=0.0, seed_frac=0.8,
               mark="OR-wrong first-order gain - EVEN's at m = 2 > 0"),
    "R4_1": dict(step=9, name="R4(1) OR-wrong beats EVEN", min=0.05, p_max=0.01, positive_frac=0.8,
                 mark="mean NS(OR-wrong) - NS(EVEN) >= 0.05, p < 0.01, >= 80% of seeds positive"),
    "R4_2": dict(step=9, name="R4(2) OR-thin beats EVEN", min=0.03, p_max=0.05,
                 mark="mean NS(OR-thin) - NS(EVEN) >= 0.03, p < 0.05"),
    "R4_3a": dict(step=9, name="R4(3) best OR-thin vs best EVEN", min=0.02, mark="mean best-minus-best >= 0.02"),
    "R4_3b": dict(step=9, name="R4(3) caution helps at all", min=0.03, mark="mean best OR-wrong - NONE >= 0.03"),
    "R4_4": dict(step=9, name="R4(4) anchors", min_gap=0.05, mark="OR-thin - SHUF >= 0.05 and SHUF - ANTI >= 0.05"),
    "R4_cap": dict(step=9, name="Capped oracle (report)", decision=0.02,
                   mark="report NS(CAP-wrong) - NS(EVEN) at the R4 level (best - best when CAP-wrong cannot reach it); "
                        "< 0.02 means the BCA test uses P1L placement"),
    # step 10: R7
    "R7a": dict(step=10, name="R7(a) minimum detectable effect", max=0.02, at_seeds=10, alpha=0.05, power=0.8,
                max_seeds=20, mark="MDE <= 0.02 NS at 10 seeds (else the seeds needed, <= 20)"),
    "R7b": dict(step=10, name="R7(b) anchors", min_gap=0.05, mark="OR-wrong - SHUF >= 0.05 and SHUF - ANTI >= 0.05"),
    "R7c": dict(step=10, name="R7(c) expert-data null world", max_abs=0.02,
                mark="|mean NS(OR-wrong) - NS(EVEN)| < 0.02"),
    "R7_gl": dict(step=10, name="Global-linear class", max=0.02, mark="mean NS(OR-wrong) - NS(EVEN) < 0.02"),
    "R7d": dict(step=10, name="R7(d) strength matching", tol=0.05, min=0.90,
                mark="|S/S* - 1| <= 5% for >= 90% of (family, target, seed)"),
    "R7e": dict(step=10, name="R7(e) Monte Carlo SE", mark="not applicable: no Monte Carlo actor class in this gate"),
    "R7_null_pairs": dict(step=10, name="Null SHUF-vs-SHUF pairs", max=1, se=2.0, mark="<= 1 pair beyond 2 SE"),
    # step 11: Tier B
    "TB": dict(step=11, name="Tier B co-trained confirmation", min_fraction=0.5, seed_frac=0.7,
               mark="NS(OR-wrong) - NS(EVEN) > 0 and >= 0.5 x Tier A's mean, in >= 70% of seeds"),
}
STEP_NAMES = {1: "build tests", 2: "Stage 0", 3: "R0 state richness", 4: "R1 thinness", 5: "R2 learned errors",
              6: "R3 errors cost return", 7: "R6 policy shift", 8: "R5 world gates", 9: "R4 Tier A",
              10: "R7 measurement validity", 11: "R4 Tier B"}
STOP_STEPS = (5, 6, 9)  # the stop rule's steps: a failure goes to the next retune or rejects the world


# ---------------------------------------------------------------------------------------------------------
# Configuration

@dataclass(frozen=True)
class GateConfig:
    """Every size and setting of the gate (pre-registered). The 'real' preset is the gate; 'tiny' only tests it."""
    preset: str = "real"
    data_seed: int = G.DATA_SEED
    sizes: tuple = (500, 200, 300)
    lean_seeds: tuple = (0, 1, 2, 3, 4)
    full_seeds: tuple = tuple(range(10))
    ladder_seeds: tuple = (0, 1, 2)
    ladder_rungs: tuple = G.LADDER
    primary_rung: float = G.PRIMARY_RUNG
    alpha_ladder: tuple = (2.5, 5.0, 10.0)
    train_seed: int = 20261013
    updates: int = 100_000
    chunk: int = 1000
    snapshots: tuple = (0.25, 0.5, 1.0)
    r2_snapshot: float = 1.0
    tier_a_snapshot: float = 0.5
    r5_snapshot: float = 1.0
    r5_extra_snapshots: tuple = (0.5,)
    tier_a_steps: int = T.TIER_A_STEPS
    tier_a_refresh: int = T.ORACLE_REFRESH
    n_shuf: int = T.N_SHUF
    match_tol: float = 0.01
    match_rounds: int = 5
    control_families: tuple = ("EVEN", "OR-wrong", "OR-thin")
    tier_b_arms: tuple = ("EVEN", "OR-wrong", "OR-thin", "SHUF0", "ANTI")
    r5_draws: int = 256
    r5_chunk_rows: int = 256
    regressor_steps: int = 5000
    regressor_batch: int = 256
    regressor_lr: float = 1e-3
    bca_fit_steps: int = 5000
    mc_episodes: int = 3000
    unit_suites: tuple = UNIT_SUITES

    @property
    def path_seeds(self):
        return {"lean": tuple(self.lean_seeds), "full": tuple(self.full_seeds)}


PRESETS = {
    "real": GateConfig(),
    "tiny": GateConfig(preset="tiny", sizes=(20, 10, 10), lean_seeds=(0, 1), full_seeds=(0, 1), ladder_seeds=(0,),
                       updates=200, chunk=50, tier_a_steps=20, tier_a_refresh=10, match_rounds=2, r5_draws=8,
                       r5_chunk_rows=64, regressor_steps=40, bca_fit_steps=20, mc_episodes=400),
}


def config_dict(cfg):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}


def config_from_dict(d):
    return GateConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def train_settings(cfg, alpha, actor="context_linear"):
    return T.TrainSettings(seed=cfg.train_seed, updates=cfg.updates, chunk=cfg.chunk, alpha=float(alpha),
                           snapshots=tuple(cfg.snapshots), actor=actor)


def world_spec(cfg, retune="base", variant="main"):
    """The Stage-0 world with the retune's change and the variant's rung / behaviour fraction."""
    if retune not in RETUNES:
        raise ValueError("retune must be one of " + ", ".join(RETUNES))
    spec = replace(G.STAGE0_SPEC, thin_rung=float(cfg.primary_rung), **RETUNES[retune])
    if variant == "main":
        return spec
    if variant == "null":
        return replace(spec, behaviour_fraction=1.0)
    m = re.fullmatch(r"rung([0-9.]+)", variant)
    if m and float(m.group(1)) in [float(r) for r in cfg.ladder_rungs]:
        return replace(spec, thin_rung=float(m.group(1)))
    raise ValueError(f"unknown variant {variant!r}: main, null or rung<r> with r in {list(cfg.ladder_rungs)}")


def rung_variant(cfg, rung):
    return "main" if float(rung) == float(cfg.primary_rung) else f"rung{float(rung):g}"


_DATA = {}


def world_and_data(cfg, retune, variant, rep):
    """(World, data) of one (retune, variant, seed): gapped_lq.generate is deterministic in these."""
    key = (cfg, retune, variant, rep)
    if key not in _DATA:
        world = G.World(world_spec(cfg, retune, variant))
        _DATA.clear()
        _DATA[key] = (world, G.generate(world, rep, cfg.data_seed, sizes=tuple(cfg.sizes)))
    return _DATA[key]


# ---------------------------------------------------------------------------------------------------------
# Pre-registration and its hash check


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes(files=CODE_FILES):
    return {rel: file_sha256(ROOT / rel) for rel in files}


def canonical(payload):
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False)


@dataclass
class Prereg:
    path: Path
    digest: str
    payload: dict
    config: GateConfig
    runs: Path
    dependency_override: bool = False

    @property
    def root(self):
        return self.runs / self.digest[:16]


def prereg_payload(cfg):
    world = G.World(world_spec(cfg))
    return dict(
        kind="glq-preregistration", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        statement="Every pass mark, seed list, retune, alpha rung and setting of the Gapped LQ acceptance gate, "
                  "fixed before any acceptance stage trains anything. glq_acceptance.load_prereg refuses every stage "
                  "when this file, its sidecar or a hashed code file differs.",
        pass_marks=PASS_MARKS, config=config_dict(cfg), stop_steps=list(STOP_STEPS), step_names=STEP_NAMES,
        seed_rule="a mark 'in >= f of 10 seeds' needs >= ceil(f n) of the path's n seeds",
        seeds=dict(lean=list(cfg.lean_seeds), full=list(cfg.full_seeds), ladder=list(cfg.ladder_seeds)),
        alpha_ladder=list(cfg.alpha_ladder), alpha_rule="the host alpha is the smallest alpha of the ladder whose "
                                                        "R3 passes; each rung is tried once, in order",
        retunes=[dict(name=r, change=RETUNES[r], text=RETUNE_TEXT[r]) for r in RETUNE_ORDER],
        retune_rule="at most two retunes, applied one at a time to the Stage-0 world in the listed order, only when "
                    "step 5, 6 or 9 fails; a retuned world reruns the gate from Stage 0",
        density_ladder=dict(rungs=list(cfg.ladder_rungs), primary=cfg.primary_rung, seeds=list(cfg.ladder_seeds),
                            alpha=cfg.alpha_ladder[0], modes=["HOST", "FQE"]),
        decoy=dict(context="DECOY", noise_sd=world.spec.decoy_noise_sd, mix_share=world.spec.mix[G.DECOY],
                   behaviour="rich (f K* + 0.3 noise)", ns_reporting="mixture NS includes the decoy; rich marks use "
                                                                     "R0 and R1; per-context NS is always reported"),
        world=dict(spec=G.spec_dict(world.spec), spec_sha256=G.spec_digest(world.spec),
                   matrices_sha256=world.describe()["matrices_sha256"]),
        train_settings=asdict(train_settings(cfg, cfg.alpha_ladder[0])),
        bca=dict(training_arm="off (glq_train.cotrain refuses mode BCA without allow_bca)",
                 band="off (r5world refuses R5(iii) without --allow-bca-band)"),
        code_sha256=code_hashes(), dependency_sha256=code_hashes(DEPENDENCY_FILES),
        python=sys.version.split()[0], numpy=np.__version__, jax=jax.__version__)


def write_prereg(path=PREREG_DEFAULT, cfg=None, force=False):
    """Write the pre-registration (strict JSON, sorted keys) and its sidecar; returns the SHA-256."""
    cfg = PRESETS["real"] if cfg is None else cfg
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists: a pre-registration is not overwritten without --force")
    text = canonical(prereg_payload(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf8")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


class PreregError(RuntimeError):
    pass


# Fields of the pre-registration that the code does not determine: checked separately (code and dependency hashes)
# or informational (time and versions). Every other field must equal what prereg_payload writes for the preset.
PREREG_VOLATILE = ("created", "python", "numpy", "jax", "code_sha256", "dependency_sha256")


def load_prereg(path=PREREG_DEFAULT, runs=RUNS_DEFAULT, allow_dependency_change=False):
    """Verify the pre-registration (sidecar, code hashes, pass marks, dependencies) and return it."""
    path = Path(path)
    if not path.exists():
        raise PreregError(f"no pre-registration at {path}: run the prereg stage first")
    text = path.read_text(encoding="utf8")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != digest:
        raise PreregError(f"{path}: SHA-256 {digest} differs from its sidecar (the file was changed)")
    payload = json.loads(text)
    on_disk = code_hashes(tuple(payload["code_sha256"]))
    bad = [rel for rel, h in payload["code_sha256"].items() if on_disk[rel] != h]
    if set(payload["code_sha256"]) != set(CODE_FILES):
        bad.append("(the hashed file list differs from CODE_FILES)")
    if bad:
        raise PreregError("code changed since the pre-registration: " + ", ".join(bad) +
                          "; write a new pre-registration (its runs go to a new directory)")
    if json.loads(canonical(PASS_MARKS)) != payload["pass_marks"]:
        raise PreregError("the pre-registered pass marks differ from PASS_MARKS")
    # Everything else the file fixes (config, seeds, alpha ladder, retunes, rungs, world, training settings) must be
    # what the hashed code writes for its preset: an edit with a recomputed sidecar is refused, not run.
    preset = (payload.get("config") or {}).get("preset")
    if preset not in PRESETS:
        raise PreregError(f"unknown preset {preset!r} in the pre-registration")
    if path.resolve() == PREREG_DEFAULT.resolve() and preset != "real":
        raise PreregError("the default pre-registration path takes the real preset only")
    fresh = json.loads(canonical(prereg_payload(PRESETS[preset])))
    edited = sorted(k for k in set(fresh) | set(payload) if k not in PREREG_VOLATILE and fresh.get(k) != payload.get(k))
    if edited:
        raise PreregError("the pre-registration differs from what the hashed code writes for preset "
                          f"{preset!r} in: {', '.join(edited)} (edited after writing?); write a new pre-registration")
    if set(payload["dependency_sha256"]) != set(DEPENDENCY_FILES):
        raise PreregError("the hashed dependency list differs from DEPENDENCY_FILES")
    deps = code_hashes(tuple(payload["dependency_sha256"]))
    changed = [rel for rel, h in payload["dependency_sha256"].items() if deps[rel] != h]
    if changed and not allow_dependency_change:
        raise PreregError("reused repository files changed since the pre-registration: " + ", ".join(changed) +
                          " (--allow-dependency-change runs anyway and records it)")
    return Prereg(path, digest, payload, config_from_dict(payload["config"]), Path(runs), bool(changed))


def guard(prereg):
    """Re-verify at the start of every stage that neither the pre-registration nor a hashed code file changed since
    load_prereg (a stage may start long after its process loaded the file, or be called through the API)."""
    if hashlib.sha256(Path(prereg.path).read_bytes()).hexdigest() != prereg.digest:
        raise PreregError(f"{prereg.path} changed since it was loaded")
    now = code_hashes(tuple(prereg.payload["code_sha256"]))
    bad = [rel for rel, h in prereg.payload["code_sha256"].items() if now[rel] != h]
    if bad:
        raise PreregError("code changed since the pre-registration: " + ", ".join(bad))


# ---------------------------------------------------------------------------------------------------------
# Output helpers


def plain(x):
    """numpy -> Python, recursively; nonfinite floats -> None (strict JSON)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return G._write(path, plain(payload))


def finish(path, payload):
    """Write a stage record and return it as written (string keys, None for nonfinite), as a resumed stage would."""
    payload = plain(payload)
    write_json(path, payload)
    return payload


def read_json(path, verify=True):
    """Read a stage JSON; with verify, its sidecar must match (None when missing or mismatched)."""
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf8")
    if verify:
        side = path.with_name(path.name + ".sha256")
        if not side.exists() or side.read_text(encoding="utf8").split()[0] != hashlib.sha256(
                text.encode("utf8")).hexdigest():
            return None
    return json.loads(text)


def done(path, prereg):
    rec = read_json(path)
    return rec is not None and rec.get("prereg_sha256") == prereg.digest


def stage_payload(prereg, stage, started, **fields):
    guard(prereg)
    return dict(kind=f"glq-acceptance-{stage}", stage=stage, prereg_sha256=prereg.digest, prereg_path=str(prereg.path),
                preset=prereg.config.preset, dependency_override=prereg.dependency_override,
                machine=platform.node(), argv=sys.argv, seconds=time.perf_counter() - started,
                finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields)


def seed_need(frac, n):
    return int(math.ceil(frac * n - 1e-9))


def critic_dir(prereg, retune, variant, mode, alpha, rep):
    tag = "FQE" if mode == "FQE" else f"{mode}_alpha{float(alpha):g}"
    return prereg.root / "critics" / retune / variant / tag / f"rep{rep}"


def tier_a_dir(prereg, retune, variant, alpha, rep):
    return prereg.root / "tierA" / retune / variant / f"alpha{float(alpha):g}" / f"rep{rep}"


def tier_b_dir(prereg, retune, alpha, rep, arm):
    return prereg.root / "tierB" / retune / f"alpha{float(alpha):g}" / f"rep{rep}" / arm


def pooled_path(prereg, stage, retune, path, alpha=None):
    name = stage if alpha is None else f"{stage}_alpha{float(alpha):g}"
    return prereg.root / stage / retune / path / f"{name}.json"


def seed_path(prereg, stage, retune, alpha, rep):
    return prereg.root / stage / retune / f"alpha{float(alpha):g}" / f"seed{rep}.json"


def host_alpha(prereg, retune, path):
    """The host alpha: the smallest ladder alpha whose R3 passes on the path, evaluated live from the critics runs.
    Returns (alpha or None, note, {alpha: r3 evaluation}) and stops at the first alpha that passes or is incomplete."""
    evals = {}
    for a in prereg.config.alpha_ladder:
        res = r3_evaluate(prereg, retune, path, float(a))
        evals[float(a)] = res
        if not res["complete"]:
            return None, f"R3 at alpha {a:g} ({path}) is incomplete: " + "; ".join(res["missing"][:3]), evals
        if res["pass_"]:
            return float(a), f"R3 passes at alpha {a:g}", evals
    return None, "R3 fails at every alpha of the ladder", evals


def resolve_alpha(prereg, retune, path, alpha):
    """--alpha: a number on the ladder, or 'auto' = host_alpha. Returns (alpha, note)."""
    if alpha not in (None, "auto"):
        a = float(alpha)
        if a not in [float(x) for x in prereg.config.alpha_ladder]:
            raise ValueError(f"alpha {a} is not on the pre-registered ladder {list(prereg.config.alpha_ladder)}")
        return a, "explicit"
    a, note, _ = host_alpha(prereg, retune, path)
    if a is None:
        if note.startswith("R3 fails"):
            note += (": the stop rule applies (next retune, or the world is rejected). Pass --alpha explicitly to run "
                     "this stage for information only")
        raise PreregError("--alpha auto: " + note)
    return a, note


def load_critic(prereg, retune, variant, mode, alpha, rep, fraction, settings=None):
    """(run.json record, AgentTrainState at the snapshot nearest `fraction`, actor kind) of a finished critics run."""
    out = critic_dir(prereg, retune, variant, mode, alpha, rep)
    if not done(out / "stage.json", prereg):
        raise FileNotFoundError(f"critics run missing: {out} (run the critics stage for seed {rep}, mode {mode}, "
                                f"variant {variant}, alpha {alpha})")
    rec = read_json(out / "run.json")
    if rec is None:
        raise FileNotFoundError(f"{out / 'run.json'} is missing or does not match its sidecar")
    settings = train_settings(prereg.config, alpha) if settings is None else settings
    snap = next((s for s in rec["snapshots"] if abs(s["fraction"] - fraction) <= 1e-9), None)
    if snap is None:
        raise IncompleteRun(f"{out}: the run stopped at update {rec['final_update']} (nonfinite at "
                            f"{rec.get('nonfinite_at')}) and has no snapshot at fraction {fraction}")
    kind = T.actor_kind("context_linear" if mode == "FQE" else settings.actor, settings.gs_hidden)
    _, template, _ = T.initial_state(T.make_args(settings, rep), kind, None, settings.lr)
    state = T.load_snapshot(out / Path(snap["path"]).name, template, snap["sha256"])
    return rec, state, kind


class IncompleteRun(RuntimeError):
    """A co-trained run that stopped before its last update (nonfinite metrics), so a snapshot is missing. The gate
    counts its seed as failed, never as missing and never with another snapshot in its place."""


def run_complete(rec):
    return rec is not None and rec.get("final_update") == rec.get("updates") and rec.get("nonfinite_at") is None


def snapshot_key(rec, fraction):
    """The diagnostics key of the snapshot at exactly `fraction`, or None when the run never reached it."""
    for snap in rec["snapshots"]:
        if abs(snap["fraction"] - fraction) <= 1e-9 and str(snap["update"]) in rec["diagnostics"]:
            return str(snap["update"])
    return None


def final_gains(rec):
    """The actor's gains at the last update of a complete run (None when the run stopped early)."""
    if not run_complete(rec):
        return None
    return np.asarray(rec["diagnostics"][str(rec["final_update"])]["K"])


def stopped_note(rec):
    return f"the run stopped at update {rec.get('final_update')} (nonfinite at {rec.get('nonfinite_at')})"


# ---------------------------------------------------------------------------------------------------------
# Statistics


def auroc(score, positive):
    """P(score of a positive row > score of a negative row) (+ half the ties); None if a class is empty."""
    score, positive = np.asarray(score, np.float64), np.asarray(positive, bool)
    keep = np.isfinite(score)
    score, positive = score[keep], positive[keep]
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = stats.rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return None
    return float(stats.spearmanr(x[keep], y[keep])[0])


def partial_spearman(x, y, z):
    """Spearman of x and y given z: the partial correlation of the ranks."""
    x, y, z = (np.asarray(v, np.float64) for v in (x, y, z))
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if keep.sum() < 4:
        return None
    rx, ry, rz = (stats.rankdata(v[keep]) for v in (x, y, z))
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) > 0 and np.ptp(b) > 0 else None
    rxy, rxz, ryz = r(rx, ry), r(rx, rz), r(ry, rz)
    if None in (rxy, rxz, ryz) or rxz ** 2 >= 1 or ryz ** 2 >= 1:
        return None
    return float((rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def paired_t(d):
    """One-sided paired t test of mean d > 0: n, mean, sd, se, t, p (None where undefined)."""
    d = np.asarray([x for x in d if x is not None and math.isfinite(x)], np.float64)
    n = len(d)
    out = dict(n=n, mean=float(d.mean()) if n else None, sd=None, se=None, t=None, p=None,
               positive=int((d > 0).sum()), values=d.tolist())
    if n >= 2:
        sd = float(d.std(ddof=1))
        out.update(sd=sd, se=sd / math.sqrt(n))
        if sd > 0:
            t = out["mean"] / out["se"]
            out.update(t=float(t), p=float(stats.t.sf(t, n - 1)))
        else:
            out.update(p=0.0 if out["mean"] > 0 else 1.0)
    return out


def mde(sd, n, alpha=0.05, power=0.8):
    """Minimum detectable mean of a one-sided paired t test: (t_{1-alpha} + t_power) sd / sqrt(n), df n - 1."""
    if sd is None or n < 2:
        return None
    return float((stats.t.ppf(1 - alpha, n - 1) + stats.t.ppf(power, n - 1)) * sd / math.sqrt(n))


def seeds_for_mde(sd, target, max_seeds, alpha=0.05, power=0.8):
    for n in range(2, max_seeds + 1):
        if mde(sd, n, alpha, power) <= target:
            return n
    return None


def mixture_logpdf(s, covs, chunk=2048):
    """log of (1/T) sum_t N(s; 0, covs_t) per row (the state law over a uniform step of a 100-step episode)."""
    covs = np.asarray(covs, np.float64)
    L = np.linalg.cholesky(covs)
    Linv = np.linalg.inv(L)
    logdet = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
    d, out = covs.shape[1], []
    for i in range(0, len(s), chunk):
        z = np.einsum("tij,nj->nti", Linv, s[i:i + chunk])
        lp = -0.5 * (np.square(z).sum(-1) + logdet + d * math.log(2 * math.pi))
        out.append(special.logsumexp(lp, axis=1) - math.log(len(covs)))
    return np.concatenate(out) if out else np.zeros(0)


def kish_fraction(log_w):
    """Kish ESS / n = (sum w)^2 / (n sum w^2), from log weights."""
    log_w = np.asarray(log_w, np.float64)
    if len(log_w) == 0:
        return None
    return float(np.exp(2 * special.logsumexp(log_w) - special.logsumexp(2 * log_w) - math.log(len(log_w))))


# ---------------------------------------------------------------------------------------------------------
# Step 1: build checks (no training)


def _trees_equal(x, y):
    lx, ly = jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y)
    return len(lx) == len(ly) and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(lx, ly))


def _bellman_residual(system, K, P_, h, s):
    """Relative residual of V(s) = r(s, -K s) + gamma E V(s') (test_gapped_lq's definition)."""
    a = -s @ K.T
    m = s @ (system.A - system.B @ K).T
    lhs = np.einsum("ni,ij,nj->n", s, P_, s) + h
    rhs = LQ.reward(system, s, a) + system.gamma * (np.einsum("ni,ij,nj->n", m, P_, m) + np.trace(P_ @ system.noise)
                                                     + h)
    return float(np.max(np.abs(lhs - rhs) / np.maximum(1.0, np.abs(lhs))))


def closed_form_checks(world, mc_episodes, seed=0):
    """B1-B4 (float64, no training)."""
    rng = np.random.default_rng([seed, 1])
    s = rng.standard_normal((32, DS))
    marks = PASS_MARKS
    b1, b1_control, b2, b3_fb, b3_lyap, b4 = [], [], [], [], [], []
    for c, system in enumerate(world.systems):
        gains = dict(k_star=world.k_star[c], k_two=world.k_two[c], behaviour=world.k_behaviour[c],
                     zero=np.zeros((DA, DS)))
        for name, K in gains.items():
            _, h, P_ = LQ.exact_q(system, K)
            b1.append(_bellman_residual(system, K, P_, h, s))
            other = gains["zero" if name != "zero" else "k_star"]
            _, h2, P2 = LQ.exact_q(system, other)
            b1_control.append(_bellman_residual(system, K, P2, h2, s))
            fwd, bwd = G.finite_return(system, K), G.finite_return_backward(system, K)
            b3_fb.append(abs(fwd - bwd) / abs(fwd))
            long = G.finite_return(system, K, length=4000)
            b3_lyap.append(abs(long - LQ.value(system, K)) / abs(long))
        grad, ref = LQ.policy_gradient(system, world.k_star[c]), LQ.policy_gradient(system, 0.5 * world.k_star[c])
        b4.append(float(np.abs(grad).max() / np.abs(ref).max()))
        n, L = mc_episodes, world.spec.length
        r = np.random.default_rng([seed, 2, c])
        ctx = np.full(n, c)
        s0, shock = r.standard_normal((n, DS)), r.standard_normal((n, L, DS))
        gains_e = np.repeat(world.k_behaviour[c][None], n, 0)
        ret, _ = G.rollout_returns(world, gains_e, ctx, s0, shock)
        z_clone = (ret.mean() - world.j_bc[c]) / (ret.std(ddof=1) / math.sqrt(n))
        noise = r.standard_normal((n, L, DA))
        ep = G.simulate_episodes(world, ctx, gains_e, world.behaviour_sd[ctx], s0, noise, shock)
        ret = ep["reward"] @ (world.spec.gamma ** np.arange(L))
        exact = world.j_finite(world.k_behaviour, behaviour_noise=True)[c]
        z_noisy = (ret.mean() - exact) / (ret.std(ddof=1) / math.sqrt(n))
        b2.append(dict(context=G.CONTEXTS[c], z_clone=float(z_clone), z_noisy=float(z_noisy)))
    max_z = max(max(abs(x["z_clone"]), abs(x["z_noisy"])) for x in b2)
    return {
        "B1": dict(max_rel_residual=max(b1), min_control_residual=min(b1_control),
                   pass_=max(b1) <= marks["B1"]["max"] and min(b1_control) > marks["B1"]["control_min"]),
        "B2": dict(per_context=b2, max_abs_z=max_z, episodes=mc_episodes, pass_=max_z <= marks["B2"]["max_abs_z"]),
        "B3": dict(max_forward_backward=max(b3_fb), max_lyapunov=max(b3_lyap),
                   pass_=max(b3_fb) <= marks["B3"]["max_forward_backward"] and max(b3_lyap) <=
                   marks["B3"]["max_lyapunov"]),
        "B4": dict(ratio_per_context=b4, max_ratio=max(b4), pass_=max(b4) <= marks["B4"]["max_ratio"]),
    }


def harness_checks(world, data):
    """B5 and B6 on a small data set, with an untrained critic (no training happens here)."""
    settings = T.TrainSettings(updates=4, chunk=2)
    args = T.make_args(settings, 0)
    kind = T.actor_kind("context_linear")
    models = (kind, T._CRITIC)
    K_bc = T.bc_gains(data["train"])[0]
    _, state, _ = T.initial_state(args, kind, kind.params(K_bc), settings.lr)
    train = G.transitions(data["train"])
    n = len(train.obs)
    rows = np.random.default_rng([5, 0]).integers(n, size=(12, 256))
    batch = jax.tree_util.tree_map(lambda x: x[rows[0]], train)
    key = jax.random.PRNGKey(3)
    b5 = {}
    for it in (1, 2):
        ours, _ = T.harness_update(args, models, state, batch, it, key, T.MAX_ACTION)
        ref, _ = BASE.td3_bc_update(args, kind.apply, T._CRITIC.apply, state, batch, it, key, T.MAX_ACTION)
        b5[f"harness_update_hooks_off_it{it}"] = _trees_equal(ours, ref)
    w = jnp.asarray(1.0 + np.random.default_rng(6).random(256), jnp.float32)
    ours, _ = T.harness_update(args, models, state, batch, 2, key, T.MAX_ACTION, bc_w=w)
    ref, _ = BASE.td3_bc_update(args, kind.apply, T._CRITIC.apply, state, batch, 2, key, T.MAX_ACTION, bc_multiplier=w)
    b5["harness_update_bc_multiplier"] = _trees_equal(ours, ref)
    table = jnp.asarray(1.0 + 0.5 * np.random.default_rng(7).random(n), jnp.float32)
    rows_j = jnp.asarray(rows, jnp.int32)
    tier = jax.jit(lambda a, c: T._tier_a_loop(args, models, c, a, train, rows_j, table))(state.actor,
                                                                                        state.critic.params)[0]
    zero = optax.set_to_zero()
    frozen = state._replace(critic=state.critic.replace(tx=zero, opt_state=zero.init(state.critic.params)))

    def body(st, xs):
        t, r = xs
        b = jax.tree_util.tree_map(lambda x: x[r], train)
        new, _ = BASE.td3_bc_update(args, kind.apply, T._CRITIC.apply, st, b, args.policy_freq * t,
                                    jax.random.PRNGKey(0), T.MAX_ACTION, bc_multiplier=table[r])
        return new._replace(actor_target=frozen.actor_target, critic=frozen.critic,
                            critic_target=frozen.critic_target), None

    ref, _ = jax.jit(lambda s: jax.lax.scan(body, s, (jnp.arange(len(rows)), rows_j)))(frozen)
    b5["tier_a_loop_vs_td3_bc_update_restored"] = _trees_equal(tier.params, ref.actor.params)

    # B6: oracle-weight identities
    TA = T.TierA(world, data, state.critic.params, settings, 0, steps=2, refresh=1, n_shuf=4)
    ctx = TA.ctx
    thin = np.isin(ctx, G.THIN).astype(np.float64)
    oracle = TA.oracle_variables(TA.K_start)
    b6 = dict(
        or_wrong=bool(np.array_equal(TA.table("OR-wrong", 3.0), 1 + 3.0 * (ctx == G.T_WRONG))),
        or_thin=bool(np.array_equal(TA.table("OR-thin", 3.0), 1 + 3.0 * thin)),
        anti=bool(np.array_equal(TA.table("ANTI", 3.0), 1 + 3.0 * (1 - thin))),
        even=bool(np.all(TA.table("EVEN", 2.0) == 2.0)),
        none=bool(np.all(TA.table("NONE", 0.0) == 1.0)),
        knob_zero_is_none=all(bool(np.all(TA.table(f, 0.0, oracle) == 1.0))
                              for f in ("OR-wrong", "OR-thin", "ANTI", "SHUF0", "OR-err", "OR-D", "CAP-wrong")),
        shuf_multisets=all(bool(np.array_equal(np.sort(TA.table(f"SHUF{k}", 3.0)), np.sort(TA.table("OR-thin", 3.0))))
                           for k in range(4)),
        cap_range=all(bool(np.all((TA.table(f, k, oracle) >= 1) & (TA.table(f, k, oracle) < 1.5)))
                      for f in ("CAP-wrong", "CAP-err") for k in TA.families[f][1]),
        or_d_mean_one=bool(oracle["ord"].mean() > 0 and abs(TA.variable("ord", oracle).mean() - 1) < 1e-12),
        or_err_rank_normal=bool(np.array_equal(np.sort(TA.variable("err", oracle)),
                                               np.sort(PH.rank_normal(np.arange(len(ctx), dtype=float),
                                                                      tie_perm=TA.tie)[0]))),
        # the multiset alone holds for any assignment: the weights must also rise with the exact error E
        or_err_follows_E=bool(np.all(np.diff(TA.variable("err", oracle)[np.argsort(oracle["E"], kind="stable")]) >= 0)))
    err = T.error_rows(world, state.critic.params, TA.K_start, data["eval"])
    fin = np.isfinite(err["e_A"])
    b6["or_d_is_minus_e_A_positive_part"] = bool(np.allclose(err["or_d"][fin], np.maximum(0, -err["e_A"][fin]),
                                                             rtol=0, atol=1e-9))
    b6["e_A_identity"] = bool(np.allclose(err["e_A"][fin], (err["e1_logged"] - err["e1_pi"])[fin], rtol=0, atol=1e-9))
    return {"B5": dict(checks=b5, pass_=all(b5.values())), "B6": dict(checks=b6, pass_=all(b6.values()))}


def parse_unittest(stderr, returncode):
    """Counts from unittest's summary lines ('Ran N tests in Xs', 'OK' or 'FAILED (failures=a, errors=b)')."""
    ran = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", stderr)
    failed = re.search(r"FAILED \(([^)]*)\)", stderr)
    counts = dict(re.findall(r"(\w+)=(\d+)", failed.group(1))) if failed else {}
    skipped = re.search(r"skipped=(\d+)", stderr)
    out = dict(returncode=returncode, ran=int(ran.group(1)) if ran else None,
               failures=int(counts.get("failures", 0)), errors=int(counts.get("errors", 0)),
               skipped=int(skipped.group(1)) if skipped else 0, tail=stderr.strip().splitlines()[-6:])
    out["pass_"] = (returncode == 0 and out["ran"] is not None and out["ran"] > 0 and out["failures"] == 0 and
                    out["errors"] == 0 and failed is None)
    return out


def run_unit_suites(suites):
    """B7: the unit suites in a subprocess (the repository root as working directory)."""
    started = time.perf_counter()
    env = dict(os.environ, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-m", "unittest", *suites], cwd=ROOT, env=env, capture_output=True,
                          text=True)
    return dict(suites=list(suites), seconds=time.perf_counter() - started, **parse_unittest(proc.stderr,
                                                                                             proc.returncode))


def stage_build_checks(prereg, unittests=True, force=False, log=print):
    guard(prereg)
    out = prereg.root / "build_checks" / "build_checks.json"
    if not force and done(out, prereg):
        log(f"build_checks: done ({out})")
        return read_json(out)
    started = time.perf_counter()
    cfg = prereg.config
    world = G.World(world_spec(cfg))
    checks = closed_form_checks(world, cfg.mc_episodes)
    small = G.generate(world, 0, cfg.data_seed, sizes=(20, 10, 10))
    checks.update(harness_checks(world, small))
    if unittests:
        log("build_checks: running the unit suites " + " ".join(cfg.unit_suites))
        checks["B7"] = run_unit_suites(cfg.unit_suites)
    else:
        checks["B7"] = dict(pass_=None, note="skipped (--no-unittests)")
    for k, v in checks.items():
        log(f"build_checks {k}: {'pass' if v['pass_'] else ('not run' if v['pass_'] is None else 'FAIL')}")
    payload = stage_payload(prereg, "build_checks", started, step=1, checks=checks,
                            pass_=all(v["pass_"] for v in checks.values() if v["pass_"] is not None))
    return finish(out, payload)


# ---------------------------------------------------------------------------------------------------------
# Step 2: Stage 0 (closed form, outcome-free)


def stage0_marks(result):
    t, m = result["targets"], PASS_MARKS
    s1 = max(result["S1_share"].values())
    s4 = min(result["S4"]["logged_joint_mix"], result["S4"]["optimal_joint_mix"])
    full = result["S3"]["full"]
    return {
        "S1": dict(measured=s1, per_context=result["S1_share"], pass_=s1 <= m["S1"]["max"]),
        "S2": dict(measured=result["S2"]["thin_share_of_gap"], g=result["S2"]["g"], pass_=bool(t["S2"])),
        "S3": dict(measured=full["loss_over_g"], radius=full["radius_T_wrong"], stable=full["stable"],
                   pass_=bool(full["stable"] and full["loss_over_g"] >= m["S3"]["min"])),
        "S4": dict(measured=s4, logged=result["S4"]["logged_joint_mix"], optimal=result["S4"]["optimal_joint_mix"],
                   pass_=s4 >= m["S4"]["min"]),
        "DECOY": dict(measured=result["decoy"]["ratio_bc"], pass_=result["decoy"]["ratio_bc"] >= m["DECOY"]["min"]),
        "S5": dict(measured=result["S5"]["marks"], pooled=result["S5"]["per_context"]["pooled"],
                   min_participation_ratio=min(result["S5"]["per_context"][c]["participation_ratio"]
                                               for c in G.CONTEXTS), pass_=bool(t["S5"])),
        "S6": dict(measured=result["S6"]["share_full"], pass_=result["S6"]["share_full"] >= m["S6"]["min"]),
        "STAB": dict(measured=result["radius"], transplant=full["radius_T_wrong"], pass_=bool(t["stability"])),
    }


def stage_stage0(prereg, retune="base", search=False, force=False, log=print):
    guard(prereg)
    out = prereg.root / "stage0" / retune / "stage0.json"
    if not force and done(out, prereg):
        log(f"stage0: done ({out})")
        return read_json(out)
    started = time.perf_counter()
    spec = world_spec(prereg.config, retune)
    result = G.stage0_full(spec)
    marks = stage0_marks(result)
    search_rec = None
    if search:
        if retune != "base":
            raise ValueError("--search reproduces the base world's selection only")
        chosen, _ = G.stage0_search(log=log)
        search_rec = dict(chosen=G.spec_dict(chosen) if chosen else None,
                          reproduces_stage0_spec=chosen == G.STAGE0_SPEC)
        marks["SEARCH"] = dict(measured=search_rec["reproduces_stage0_spec"], pass_=bool(chosen == G.STAGE0_SPEC))
    world = G.World(spec)
    log(f"stage0 {retune}: " + ", ".join(f"{k} {'pass' if v['pass_'] else 'FAIL'}" for k, v in marks.items()))
    payload = stage_payload(prereg, "stage0", started, step=2, retune=retune, spec=G.spec_dict(spec),
                            spec_sha256=G.spec_digest(spec), marks=marks, result=result, search=search_rec,
                            world=dict(rho=list(world.rho), coupling_attempt=world.coupling_attempt,
                                       J_bc=world.j_bc, J_opt=world.j_opt,
                                       gap_mix=float(world.mix @ (world.j_opt - world.j_bc))),
                            pass_=all(v["pass_"] for v in marks.values()))
    return finish(out, payload)


# ---------------------------------------------------------------------------------------------------------
# Steps 3-4: R0 and R1 on the log


def r0r1_seed(cfg, retune, rep):
    """R0 at the primary rung and R1(a)(b) at every rung of the ladder, on the training split of one seed."""
    m = PASS_MARKS
    spec = world_spec(cfg, retune)
    thin_mix = float(sum(spec.mix[c] for c in G.THIN))
    rungs = {}
    for rung in cfg.ladder_rungs:
        world = G.World(replace(spec, thin_rung=float(rung)))
        data = G.generate(world, rep, cfg.data_seed, sizes=tuple(cfg.sizes))
        met = G.log_metrics(world, data["train"], float(rung))
        thin_sd = {G.CONTEXTS[c]: met["R1"][G.CONTEXTS[c]]["action_resid_sd"][2:] for c in G.THIN}
        rungs[f"{float(rung):g}"] = dict(
            R1a_thin=met["marks"]["R1a_thin_rung"], thin_actuator34_sd=thin_sd,
            thin_rank={G.CONTEXTS[c]: met["R1"][G.CONTEXTS[c]]["rank"] for c in G.THIN},
            R1a_rich=met["marks"]["R1a_rich_eigs"],
            rich_min_eig=min(min(met["R1"][G.CONTEXTS[c]]["action_cov_eigs"]) for c in G.RICH),
            thin_share=met["thin_share"], metrics=met if float(rung) == float(cfg.primary_rung) else None,
            context_episodes=data["meta"]["context_episodes"])
    prim = rungs[f"{float(cfg.primary_rung):g}"]["metrics"]
    r0 = prim["R0"]
    marks = {
        "R0_pr": dict(measured={c: r0[c]["participation_ratio"] for c in G.CONTEXTS},
                      pass_=all(r0[c]["participation_ratio"] >= m["R0_pr"]["min"] for c in G.CONTEXTS)),
        "R0_transient": dict(measured=r0["pooled"]["transient_share"],
                             pass_=r0["pooled"]["transient_share"] >= m["R0_transient"]["min"]),
        "R0_iact": dict(measured=r0["pooled"]["iact"],
                        pass_=m["R0_iact"]["lo"] <= r0["pooled"]["iact"] <= m["R0_iact"]["hi"]),
        "R0_scale": dict(measured=r0["pooled"]["row_share_within_5"],
                         pass_=r0["pooled"]["row_share_within_5"] >= m["R0_scale"]["min"]),
        "R1a_rich": dict(measured=min(v["rich_min_eig"] for v in rungs.values()),
                         pass_=all(v["R1a_rich"] for v in rungs.values())),
        "R1a_thin": dict(measured={k: v["thin_actuator34_sd"] for k, v in rungs.items()},
                         pass_=all(v["R1a_thin"] for v in rungs.values())),
        "R1b": dict(measured=prim["thin_share"], target=thin_mix,
                    pass_=abs(prim["thin_share"] - thin_mix) <= m["R1b"]["tol"]),
    }
    return dict(marks=marks, rungs=rungs, thin_mix_share=thin_mix)


def stage_r0r1(prereg, seed, retune="base", force=False, log=print):
    guard(prereg)
    out = prereg.root / "r0r1" / retune / f"seed{seed}.json"
    if not force and done(out, prereg):
        log(f"r0r1 seed {seed}: done")
        return read_json(out)
    started = time.perf_counter()
    rec = r0r1_seed(prereg.config, retune, seed)
    log(f"r0r1 {retune} seed {seed}: " + ", ".join(f"{k} {'pass' if v['pass_'] else 'FAIL'}"
                                                   for k, v in rec["marks"].items()))
    payload = stage_payload(prereg, "r0r1", started, step=3, retune=retune, seed=seed, **rec,
                            pass_=all(v["pass_"] for v in rec["marks"].values()))
    return finish(out, payload)


# ---------------------------------------------------------------------------------------------------------
# Step 5: critics (training starts here)


def stage_critics(prereg, seed, mode, variant="main", alpha=None, retune="base", path="lean", force=False,
                  log=print):
    """One co-trained critic run. alpha None: the ladder's first rung; 'auto': the host alpha chosen by r3."""
    guard(prereg)
    cfg = prereg.config
    if mode not in MODES:
        raise ValueError("mode must be one of " + ", ".join(MODES) + " (the BCA arm is not part of this gate)")
    alpha = float(cfg.alpha_ladder[0]) if alpha is None else resolve_alpha(prereg, retune, path, alpha)[0]
    if alpha not in [float(a) for a in cfg.alpha_ladder]:
        raise ValueError(f"alpha {alpha} is not on the pre-registered ladder")
    if variant == "null" and mode != "HOST":
        raise ValueError("the null world trains HOST only")
    if variant.startswith("rung") and (mode == "EXACT" or seed not in cfg.ladder_seeds or
                                       alpha != float(cfg.alpha_ladder[0])):
        raise ValueError("density-ladder runs are HOST and FQE on the ladder seeds at the ladder's first alpha")
    out = critic_dir(prereg, retune, variant, mode, alpha, seed)
    if not force and done(out / "stage.json", prereg):
        log(f"critics {retune}/{variant}/{mode} alpha {alpha:g} seed {seed}: done")
        return read_json(out / "stage.json")
    started = time.perf_counter()
    world, data = world_and_data(cfg, retune, variant, seed)
    settings = train_settings(cfg, alpha)
    rec = T.cotrain(world, data, settings, mode, seed, out_dir=out, log=log)
    summary = {k: dict(pooled=v["pooled"], NS_mix=v["score"]["NS_mix"] if v["score"] else None)
               for k, v in rec["diagnostics"].items()}
    payload = stage_payload(prereg, "critics", started, step=5, retune=retune, variant=variant, mode=mode,
                            alpha=alpha, seed=seed, run_json_sha256=rec["run_json_sha256"],
                            final_update=rec["final_update"], nonfinite_at=rec.get("nonfinite_at"),
                            exact_unstable_context_refreshes=rec["exact_unstable_context_refreshes"],
                            spec_sha256=G.spec_digest(world.spec), summary=summary)
    return finish(out / "stage.json", payload)


def _run_record(prereg, retune, variant, mode, alpha, rep):
    out = critic_dir(prereg, retune, variant, mode, alpha, rep)
    if not done(out / "stage.json", prereg):
        return None
    return read_json(out / "run.json")


def _thin_rich_E(diag):
    per = diag["per_context"]
    vals = [(per[G.CONTEXTS[c]]["mean_E_pi"], per[G.CONTEXTS[c]]["rows"]) for c in G.THIN]
    thin = (sum(v * n for v, n in vals) / sum(n for _, n in vals)) if all(v is not None for v, _ in vals) else None
    return thin, diag["pooled"]["mean_E_rich_main"]


def r2_values(rec, fraction):
    key = snapshot_key(rec, fraction)
    if key is None:  # a run that stopped early fails its seed
        return dict(ratio=None, thin_E2=None, rel=None, rel_logged=None, rel_deterministic=None,
                    rel_deterministic_logged=None, Twrong_over_Tright=None, thin_mean_E=None,
                    rich_mean_E=None, unstable_contexts=None, note=stopped_note(rec),
                    marks=dict(R2_ratio=False, R2_thin_E2=False, R2_rel=False), all=False)
    d = rec["diagnostics"][key]
    p = d["pooled"]
    thin, rich = _thin_rich_E(d)
    m = PASS_MARKS
    vals = dict(ratio=p["ratio_E_Twrong_rich_main"], thin_E2=p["thin_share_sum_E2"],
                rel=p["rich_main_median_rel_e1_td3_pi"], rel_logged=p["rich_main_median_rel_e1_td3_logged"],
                rel_deterministic=p["rich_main_median_rel_e1_pi"],
                rel_deterministic_logged=p["rich_main_median_rel_e1_logged"],
                Twrong_over_Tright=p["ratio_E_Twrong_Tright"], thin_mean_E=thin, rich_mean_E=rich,
                unstable_contexts=p["unstable_contexts"])
    ok = lambda x, f: x is not None and f(x)
    vals["marks"] = dict(R2_ratio=ok(vals["ratio"], lambda x: x >= m["R2_ratio"]["min"]),
                         R2_thin_E2=ok(vals["thin_E2"], lambda x: x >= m["R2_thin_E2"]["min"]),
                         R2_rel=ok(vals["rel"], lambda x: x <= m["R2_rel"]["max"]))
    vals["all"] = all(vals["marks"].values())
    return vals


def r2_evaluate(prereg, retune, path, alpha):
    """R2 from the finished critics runs (evaluated live; the r2 stage writes it)."""
    cfg = prereg.config
    seeds = cfg.path_seeds[path]
    per, missing = {}, []
    for mode in ("HOST", "FQE"):
        per[mode] = {}
        for rep in seeds:
            rec = _run_record(prereg, retune, "main", mode, alpha, rep)
            if rec is None:
                missing.append(f"critics {mode} seed {rep} alpha {alpha:g}")
                continue
            per[mode][rep] = r2_values(rec, cfg.r2_snapshot)
    need = seed_need(PASS_MARKS["R2_ratio"]["seed_frac"], len(seeds))
    counts = {mode: sum(v["all"] for v in per[mode].values()) for mode in per}
    ladder, ladder_missing = {}, []
    a0 = float(cfg.alpha_ladder[0])
    for mode in ("HOST", "FQE"):
        means = []
        for rung in cfg.ladder_rungs:
            vals = []
            for rep in cfg.ladder_seeds:
                rec = _run_record(prereg, retune, rung_variant(cfg, rung), mode, a0, rep)
                if rec is None:
                    ladder_missing.append(f"critics {mode} {rung_variant(cfg, rung)} seed {rep}")
                    continue
                key = snapshot_key(rec, cfg.r2_snapshot)
                vals.append(_thin_rich_E(rec["diagnostics"][key]) if key else (None, None))
            ok = len(vals) == len(cfg.ladder_seeds) and all(t is not None and r is not None for t, r in vals)
            means.append(dict(rung=float(rung), thin_mean_E=float(np.mean([t for t, _ in vals])) if ok else None,
                              rich_mean_E=float(np.mean([r for _, r in vals])) if ok else None))
        E = [x["thin_mean_E"] for x in means]
        complete = all(e is not None for e in E)
        falls = sum(E[k + 1] < E[k] for k in range(len(E) - 1)) if complete else None
        top = means[-1]
        ratio = top["thin_mean_E"] / top["rich_mean_E"] if complete and top["rich_mean_E"] else None
        f = PASS_MARKS["R2_ladder_top"]["factor"]
        ladder[mode] = dict(rungs=means, falls=falls, steps=len(E) - 1, top_ratio=ratio,
                            falls_pass=falls is not None and falls >= PASS_MARKS["R2_ladder_falls"]["min"],
                            top_pass=ratio is not None and 1 / f <= ratio <= f)
    complete = not missing and not ladder_missing
    marks = dict(main={mode: counts[mode] >= need for mode in counts},
                 ladder_falls=all(ladder[m]["falls_pass"] for m in ladder),
                 ladder_top=all(ladder[m]["top_pass"] for m in ladder))
    passed = complete and all(marks["main"].values()) and marks["ladder_falls"] and marks["ladder_top"]
    return dict(alpha=alpha, seeds=list(seeds), need=need, per_seed=per, passing_seeds=counts, ladder=ladder,
                missing=missing + ladder_missing, complete=complete, marks=marks, pass_=passed if complete else None)


def stage_r2(prereg, path="lean", retune="base", alpha="auto", force=False, log=print):
    """Pooled stages always recompute (cheap, never stale). With alpha 'auto' before R3 has chosen, R2 is evaluated
    at the ladder's first alpha and marked provisional."""
    guard(prereg)
    cfg = prereg.config
    try:
        alpha, note = resolve_alpha(prereg, retune, path, alpha)
    except PreregError as exc:
        alpha, note = float(cfg.alpha_ladder[0]), f"provisional (alpha {cfg.alpha_ladder[0]:g}): {exc}"
    started = time.perf_counter()
    res = r2_evaluate(prereg, retune, path, alpha)
    payload = stage_payload(prereg, "r2", started, step=5, retune=retune, path=path, alpha_note=note, **res)
    payload = finish(pooled_path(prereg, "r2", retune, path, alpha), payload)
    log(f"r2 {retune} {path} alpha {alpha:g}: " + ("incomplete" if not res["complete"] else
                                                   ("pass" if res["pass_"] else "FAIL")))
    return payload


# ---------------------------------------------------------------------------------------------------------
# Step 6: R3


R3_KEYS = ("off_support_cost_T_wrong", "cost_share_of_gap", "off_support_cost_T_right", "NS_host", "NS_exact",
           "NS_host_per_context", "NS_exact_per_context", "shortfall_per_context", "thin_shortfall_share",
           "rich_NS_host", "stable_host", "stable_exact")


def r3_values(world, host_rec, exact_rec):
    m = PASS_MARKS
    K_host, K_exact = final_gains(host_rec), final_gains(exact_rec)
    if K_host is None or K_exact is None:  # a run that stopped early fails its seed
        note = "; ".join(f"{name}: {stopped_note(r)}" for name, r, K in (("HOST", host_rec, K_host),
                                                                         ("EXACT", exact_rec, K_exact)) if K is None)
        return dict({k: None for k in R3_KEYS}, note=note, all=False,
                    marks=dict(R3_cost=False, R3_ns_gap=False, R3_thin_shortfall=False, R3_rich_ns=False))
    J_host, J_exact = world.j_finite(K_host), world.j_finite(K_exact)
    K0 = K_host.copy()
    K0[G.T_WRONG, 2:] = 0.0
    cost = float(G.finite_return(world.systems[G.T_WRONG], K0[G.T_WRONG]) - J_host[G.T_WRONG])
    gap = world.j_opt - world.j_bc
    total_gap = float(world.mix @ gap)
    ns_host, ns_exact = world.normalised(J_host)[1], world.normalised(J_exact)[1]
    short = world.mix * (J_exact - J_host)
    thin_share = float(short[list(G.THIN)].sum() / short.sum()) if short.sum() > 0 else None
    rich = list(G.RICH_MAIN)
    rich_ns = float(world.mix[rich] @ (J_host[rich] - world.j_bc[rich]) / (world.mix[rich] @ gap[rich]))
    cost_share = float(world.mix[G.T_WRONG] * cost / total_gap)
    vals = dict(off_support_cost_T_wrong=cost, cost_share_of_gap=cost_share,
                off_support_cost_T_right=float(G.finite_return(world.systems[G.T_RIGHT], np.vstack(
                    [K_host[G.T_RIGHT, :2], np.zeros((2, DS))])) - J_host[G.T_RIGHT]),
                NS_host=ns_host, NS_exact=ns_exact, NS_host_per_context=world.normalised(J_host)[0],
                NS_exact_per_context=world.normalised(J_exact)[0], shortfall_per_context=short,
                thin_shortfall_share=thin_share, rich_NS_host=rich_ns,
                stable_host=[bool(r < 1) for r in world.radius(K_host)],
                stable_exact=[bool(r < 1) for r in world.radius(K_exact)])
    vals["marks"] = dict(R3_cost=cost_share >= m["R3_cost"]["min"],
                         R3_ns_gap=ns_host <= ns_exact - m["R3_ns_gap"]["min"],
                         R3_thin_shortfall=thin_share is not None and thin_share >= m["R3_thin_shortfall"]["min"],
                         R3_rich_ns=rich_ns >= m["R3_rich_ns"]["min"])
    vals["all"] = all(vals["marks"].values())
    return vals


def r3_evaluate(prereg, retune, path, alpha):
    """R3 at one alpha from the finished HOST and EXACT runs (evaluated live; the r3 stage writes it)."""
    cfg = prereg.config
    seeds = cfg.path_seeds[path]
    world = G.World(world_spec(cfg, retune))
    per, missing = {}, []
    for rep in seeds:
        host = _run_record(prereg, retune, "main", "HOST", alpha, rep)
        exact = _run_record(prereg, retune, "main", "EXACT", alpha, rep)
        if host is None or exact is None:
            missing.append(f"seed {rep}: " + ", ".join(m for m, r in (("HOST", host), ("EXACT", exact)) if r is None))
            continue
        per[rep] = r3_values(world, host, exact)
    need = seed_need(PASS_MARKS["R3_cost"]["seed_frac"], len(seeds))
    count = sum(v["all"] for v in per.values())
    per_mark = {k: sum(v["marks"][k] for v in per.values()) for k in ("R3_cost", "R3_ns_gap", "R3_thin_shortfall",
                                                                      "R3_rich_ns")}
    complete = not missing
    passed = complete and count >= need
    return dict(alpha=alpha, seeds=list(seeds), need=need, per_seed=per, passing_seeds=count,
                passing_by_mark=per_mark, missing=missing, complete=complete, pass_=passed if complete else None)


def stage_r3(prereg, path="lean", alpha=None, retune="base", force=False, log=print):
    guard(prereg)
    cfg = prereg.config
    alpha = float(cfg.alpha_ladder[0] if alpha in (None, "auto") else alpha)
    if alpha not in [float(a) for a in cfg.alpha_ladder]:
        raise ValueError(f"alpha {alpha} is not on the pre-registered ladder")
    started = time.perf_counter()
    res = r3_evaluate(prereg, retune, path, alpha)
    payload = stage_payload(prereg, "r3", started, step=6, retune=retune, path=path, **res)
    payload = finish(pooled_path(prereg, "r3", retune, path, alpha), payload)
    log(f"r3 {retune} {path} alpha {alpha:g}: " + ("incomplete" if not res["complete"] else (
        "pass" if res["pass_"] else "FAIL")) + f" ({res['passing_seeds']}/{len(res['seeds'])} seeds, need "
        f"{res['need']})")
    return payload


# ---------------------------------------------------------------------------------------------------------
# Step 7: R6 and R1(c)


def shift_shares(world, K, rows):
    """R6 (a) and (b) shares of a per-context linear policy on one split's rows."""
    ctx, s = rows["context"], rows["state"]
    pi = world.policy(K, s, ctx)
    thin = np.isin(ctx, G.THIN)
    rich = np.isin(ctx, G.RICH_MAIN)
    m = PASS_MARKS
    off = np.linalg.norm(pi[thin, 2:], axis=1)
    mu = world.policy(world.k_behaviour, s, ctx)
    dist = np.linalg.norm(pi[rich] - mu[rich], axis=1) / world.spec.rich_noise_sd
    return dict(a=float(np.mean(off >= m["R6a"]["threshold"])), b=float(np.mean(dist > m["R6b"]["distance"])),
                median_off=float(np.median(off)), median_distance=float(np.median(dist)))


def kish_per_context(world, K, rows):
    out = {}
    K = world.per_context(K)
    for c in range(NC):
        s = rows["state"][rows["context"] == c]
        try:
            cov_pi = np.stack(world.state_covariances(K, c))
            cov_log = np.stack(world.state_covariances(world.k_behaviour, c, world.action_cov(c)))
            out[G.CONTEXTS[c]] = kish_fraction(mixture_logpdf(s, cov_pi) - mixture_logpdf(s, cov_log))
        except np.linalg.LinAlgError:
            out[G.CONTEXTS[c]] = None
    return out


def r6_seed(cfg, retune, rep, host_rec):
    m = PASS_MARKS
    world, data = world_and_data(cfg, retune, "main", rep)
    K_host = final_gains(host_rec)
    if K_host is None:  # a run that stopped early fails its seed
        return dict(host=None, bc_start=None, kish_host=None, kish_bc_start=None, R1c_share=None,
                    note=stopped_note(host_rec), marks=dict(R6a=False, R6b=False, R6c=False, R6d=False, R1c=False))
    K_bc = T.bc_gains(data["train"])[0]
    ev = data["eval"]
    host, start = shift_shares(world, K_host, ev), shift_shares(world, K_bc, ev)
    ess = kish_per_context(world, K_host, ev)
    ess_bc = kish_per_context(world, K_bc, ev)
    thin = np.isin(ev["context"], G.THIN)
    pi = world.policy(K_host, ev["state"], ev["context"])
    dens = world.behaviour_density_ratio(ev["state"][thin], pi[thin], ev["context"][thin], floor=m["R1c"]["floor"])
    r1c = float(np.mean(dens < m["R1c"]["density"]))
    lo, hi = m["R6c"]["lo"], m["R6c"]["hi"]
    marks = dict(R6a=host["a"] >= m["R6a"]["min"], R6b=host["b"] >= m["R6b"]["min"],
                 R6c=any(v is not None and lo <= v <= hi for v in ess.values()),
                 R6d=start["a"] <= m["R6d"]["max"] and start["b"] <= m["R6d"]["max"], R1c=r1c >= m["R1c"]["min"])
    return dict(host=host, bc_start=start, kish_host=ess, kish_bc_start=ess_bc, R1c_share=r1c, marks=marks,
                score_host=world.score(K_host), note=None)


def stage_r6(prereg, seed, path="lean", alpha="auto", retune="base", force=False, log=print):
    guard(prereg)
    alpha, note = resolve_alpha(prereg, retune, path, alpha)
    out = seed_path(prereg, "r6", retune, alpha, seed)
    if not force and done(out, prereg):
        return read_json(out)
    started = time.perf_counter()
    rec = _run_record(prereg, retune, "main", "HOST", alpha, seed)
    if rec is None:
        raise FileNotFoundError(f"r6 needs the HOST critics run of seed {seed} at alpha {alpha:g}")
    vals = r6_seed(prereg.config, retune, seed, rec)
    log(f"r6 {retune} seed {seed}: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in vals["marks"].items()))
    payload = stage_payload(prereg, "r6", started, step=7, retune=retune, alpha=alpha, alpha_note=note, seed=seed,
                            **vals)
    return finish(out, payload)


# ---------------------------------------------------------------------------------------------------------
# Step 8: R5 world gates (frozen snapshots; no BCA training, no BCA band unless explicitly allowed)


_PROGRAMS = {}


def _target_q_program(kind):
    """min over heads of the target critic at (next_obs, target actor(next_obs) + noise) (TD3's target, no clip)."""
    key = ("target_q", kind.name)
    if key not in _PROGRAMS:
        def run(actor_t, critic_t, next_obs, noise):
            action = kind.apply(actor_t, next_obs) + noise
            return T._CRITIC.apply(critic_t, next_obs, action).min(-1)

        _PROGRAMS[key] = jax.jit(run)
    return _PROGRAMS[key]


def population_band(world, state, kind, rows, settings, draws, rng, chunk_rows=256):
    """The conditional RMS of t - Q1(s, a) at each row's logged (s, a): t = r + gamma min Q_targ(s', pi_targ(s') +
    clip(N(0, 0.2^2), 0.5)), s' ~ N(A s + B_c a, sd_c^2 I) exactly; `draws` Monte Carlo draws per row. Returns
    (band, mean residual, SD of t, Q1 at the logged rows)."""
    obs32 = np.asarray(rows["obs"], np.float32)
    s, a, ctx = rows["state"], rows["action"], np.asarray(rows["context"], np.int64)
    q1 = T.critic_rows(state.critic.params, obs32, np.asarray(a, np.float32))[0][:, 0]
    mean_next = s @ world.A.T + np.einsum("nij,nj->ni", world.B[ctx], a)
    sd = world.noise_sd[ctx]
    code = obs32[:, DS:]
    program = _target_q_program(kind)
    n = len(ctx)
    band, mean_res, sd_t = np.empty(n), np.empty(n), np.empty(n)
    for i in range(0, n, chunk_rows):
        k = min(chunk_rows, n - i)
        sl = slice(i, i + k)
        shocks = rng.standard_normal((k, draws, DS))
        noise = np.clip(rng.standard_normal((k, draws, DA)) * settings.policy_noise, -settings.noise_clip,
                        settings.noise_clip)
        nxt = mean_next[sl, None, :] + sd[sl, None, None] * shocks
        nobs = np.concatenate([nxt, np.broadcast_to(code[sl, None, :], (k, draws, NC))], -1)
        pad = chunk_rows - k
        nobs = nobs.reshape(k * draws, OBS_DIM).astype(np.float32)
        nz = noise.reshape(k * draws, DA).astype(np.float32)
        if pad:
            nobs = np.concatenate([nobs, np.zeros((pad * draws, OBS_DIM), np.float32)])
            nz = np.concatenate([nz, np.zeros((pad * draws, DA), np.float32)])
        tq = np.asarray(program(state.actor_target.params, state.critic_target.params, jnp.asarray(nobs),
                                jnp.asarray(nz)), np.float64)[:k * draws].reshape(k, draws)
        t = rows["reward"][sl, None] + settings.gamma * tq
        res = t - q1[sl, None]
        band[sl] = np.sqrt(np.mean(res ** 2, 1))
        mean_res[sl] = res.mean(1)
        sd_t[sl] = t.std(1)
    return band, mean_res, sd_t, q1


def fit_regressor(train_rows, eval_rows, *, state=None, kind=None, y=None, settings, steps, batch, lr, rng_rows,
                  key):
    """A Calibrator-shaped MLP (calibration/network.Calibrator, state_dep, 64-64 ReLU, softplus + 0.01) fitted by
    Adam on the mean squared error to a per-row residual magnitude. y=None: the ideal target |t - Q1(s, a)| with t
    TD3's target from the snapshot (fresh smoothing noise each step); else the fixed targets y (NaN rows ignored).
    Returns predictions on eval_rows and the last loss."""
    obs = jnp.asarray(np.asarray(train_rows["obs"], np.float32))
    act = jnp.asarray(np.asarray(train_rows["action"], np.float32))
    mean, std = jnp.mean(obs, 0), jnp.std(obs, 0)
    cal = Calibrator(mean, std, state_dep=True)
    params = cal.init(key, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, DA)))
    tx = optax.adam(lr)
    idx = jnp.asarray(rng_rows.integers(len(obs), size=(steps, batch)), jnp.int32)
    keys = jax.random.split(jax.random.fold_in(key, 1), steps)
    if y is None:
        q1 = jnp.asarray(T.critic_rows(state.critic.params, np.asarray(obs), np.asarray(act))[0][:, 0], jnp.float32)
        nobs = jnp.asarray(np.asarray(train_rows["next_obs"], np.float32))
        rew = jnp.asarray(np.asarray(train_rows["reward"], np.float32))
        mask = jnp.ones(len(obs), jnp.float32)
        yv = jnp.zeros(len(obs), jnp.float32)
        nets = (state.actor_target.params, state.critic_target.params)
    else:
        y = np.asarray(y, np.float64)
        mask = jnp.asarray(np.isfinite(y), jnp.float32)
        yv = jnp.asarray(np.where(np.isfinite(y), y, 0.0), jnp.float32)
        q1 = nobs = rew = nets = None

    def target(r, k):
        if y is not None:
            return yv[r]
        noise = jnp.clip(jax.random.normal(k, (batch, DA)) * settings.policy_noise, -settings.noise_clip,
                         settings.noise_clip)
        a2 = kind.apply(nets[0], nobs[r]) + noise
        t = rew[r] + settings.gamma * T._CRITIC.apply(nets[1], nobs[r], a2).min(-1)
        return jnp.abs(t - q1[r])

    def body(carry, xs):
        p, opt = carry
        r, k = xs
        yt, w = jax.lax.stop_gradient(target(r, k)), mask[r]

        def loss_fn(q):
            return jnp.sum(w * jnp.square(cal.apply(q, obs[r], act[r]) - yt)) / jnp.maximum(jnp.sum(w), 1.0)

        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, opt = tx.update(g, opt, p)
        return (optax.apply_updates(p, upd), opt), loss

    (params, _), losses = jax.jit(lambda p: jax.lax.scan(body, (p, tx.init(p)), (idx, keys)))(params)
    pred = np.asarray(cal.apply(params, jnp.asarray(np.asarray(eval_rows["obs"], np.float32)),
                                jnp.asarray(np.asarray(eval_rows["action"], np.float32))), np.float64)
    return pred, float(losses[-1])


def bca_band(world, state, kind, data, settings, rep, cfg, allow=False, execute=True):
    """R5(iii), switched off: BCA's own band on a frozen snapshot, lq_harness.fit_scale_signal (q1 signal) for
    cfg.bca_fit_steps steps on the training rows, then calibrate_signal (WBCP alpha 0.1, beta 0.95, 1,000 draws) on the
    calibration bank; W = threshold x positive_scale(eta, unit) on the training and evaluation rows. Refuses without
    allow. execute=False returns the lowered fit program (traced, not run)."""
    if not allow:
        raise RuntimeError(BCA_BAND_OFF)
    args = T.make_args(settings, rep)
    config = P.Config(arm="bca", posterior=WBCPConfig(0.1, 0.95, 1000), blend=0.5)
    cal = Calibrator(jnp.zeros(OBS_DIM), jnp.ones(OBS_DIM), state_dep=True)
    models = (kind, T._CRITIC, cal)
    train, bank_rows, ev = (G.transitions(data[s]) for s in ("train", "cal", "eval"))
    key = jax.random.fold_in(jax.random.PRNGKey(cfg.train_seed), rep * 1000 + TAG_BCA)
    cal_state = TrainState.create(apply_fn=cal.apply, params=cal.init(jax.random.fold_in(key, LQ.KEY_INIT),
                                                                      jnp.zeros((1, OBS_DIM)), jnp.zeros((1, DA))),
                                  tx=optax.adam(config.cal_lr))
    st = P.State(state, cal_state, jnp.asarray(1.0), initial_reference(cal_state.params))
    rows = jnp.asarray(np.random.default_rng([cfg.train_seed, rep, TAG_BCA]).integers(
        len(train.obs), size=(cfg.bca_fit_steps, settings.batch_size)), jnp.int32)
    keys = jax.vmap(partial(jax.random.fold_in, jax.random.fold_in(key, 1)))(jnp.arange(cfg.bca_fit_steps))
    fit = jax.jit(partial(LQ.fit_loop, args, config, models, T.MAX_ACTION))
    if not execute:
        return fit.lower(st, train, rows, keys, jnp.int32(1))
    st, (accepted, _, _) = fit(st, train, rows, keys, jnp.int32(1))
    reference, diag, _, _ = LQ.calibrate_signal(args, config, models, st, bank_rows, jax.random.fold_in(key, 2), 1,
                                                T.MAX_ACTION)
    width = lambda tr: float(reference.threshold) * np.asarray(positive_scale(
        cal.apply(reference.cal_params, tr.obs, tr.action), reference.residual_scale), np.float64)
    return dict(threshold=float(reference.threshold), accepted_share=float(np.mean(np.asarray(accepted))),
                diagnostics={k: v for k, v in diag.items() if np.isscalar(v)}, W_train=width(train),
                W_eval=width(ev))


def r5_snapshot_values(prereg, retune, rep, alpha, fraction, regressors=True, allow_bca_band=False, log=print):
    cfg, m = prereg.config, PASS_MARKS
    world, data = world_and_data(cfg, retune, "main", rep)
    settings = train_settings(cfg, alpha)
    _, state, kind = load_critic(prereg, retune, "main", "HOST", alpha, rep, fraction, settings)
    K = kind.gains(state.actor.params)
    ev = data["eval"]
    err = T.error_rows(world, state.critic.params, K, ev)
    E, ctx = err["E_pi"], ev["context"]
    s, a = ev["state"], ev["action"]
    alea = world.aleatoric_sd(K, s, a, ctx) if all(r < 1 for r in world.radius(K)) else np.full(len(ctx), np.nan)
    lever = np.linalg.norm(s, axis=1)
    dist = np.linalg.norm(world.policy(K, s, ctx) - world.policy(world.k_behaviour, s, ctx), axis=1)
    confounds = dict(aleatoric_sd=alea, leverage=lever, distance=dist)
    groups = {"pooled": np.ones(len(ctx), bool), **{G.CONTEXTS[c]: ctx == c for c in range(NC)}}
    rho = {name: {g: spearman(E[mask], x[mask]) for g, mask in groups.items()} for name, x in confounds.items()}
    mean_E = {G.CONTEXTS[c]: T._nanmean(E[ctx == c]) for c in range(NC)}
    tr_tw = (mean_E["T_right"] / mean_E["T_wrong"]) if mean_E["T_wrong"] else None
    log(f"r5world seed {rep} snapshot {fraction}: population band ({cfg.r5_draws} draws per row)")
    band, mean_res, sd_t, _ = population_band(world, state, kind, ev, settings, cfg.r5_draws,
                                              np.random.default_rng([cfg.train_seed, rep, TAG_R5_BAND,
                                                                     int(round(1000 * fraction))]),
                                              cfg.r5_chunk_rows)
    rich, thin = np.isin(ctx, G.RICH_MAIN), np.isin(ctx, G.THIN)
    wrong_rich = (ctx == G.T_WRONG) | rich
    thin_decoy = thin | (ctx == G.DECOY)
    band_vals = dict(
        spearman_E=spearman(band, E), spearman_E_per_context={G.CONTEXTS[c]: spearman(band[ctx == c], E[ctx == c])
                                                               for c in range(NC)},
        partial_spearman_E_given_aleatoric=partial_spearman(band, E, alea),
        auroc_T_wrong_vs_rich=auroc(band[wrong_rich], (ctx == G.T_WRONG)[wrong_rich]),
        auroc_thin_vs_decoy=auroc(band[thin_decoy], thin[thin_decoy]),
        auroc_T_wrong_vs_T_right=auroc(band[thin], (ctx == G.T_WRONG)[thin]),
        mean_band={G.CONTEXTS[c]: float(band[ctx == c].mean()) for c in range(NC)},
        mean_residual={G.CONTEXTS[c]: float(mean_res[ctx == c].mean()) for c in range(NC)},
        mean_sd_target={G.CONTEXTS[c]: float(sd_t[ctx == c].mean()) for c in range(NC)},
        spearman_confounds={name: spearman(band, x) for name, x in confounds.items()})
    fin = np.isfinite(E)
    high = E >= np.quantile(E[fin], m["R5_ideal"]["quantile"]) if fin.any() else np.zeros(len(E), bool)
    reg = None
    if regressors:
        tr = data["train"]
        key = jax.random.fold_in(jax.random.PRNGKey(cfg.train_seed), rep * 1000 + TAG_R5_FIT)
        pred_ideal, loss_ideal = fit_regressor(tr, ev, state=state, kind=kind, settings=settings,
                                               steps=cfg.regressor_steps, batch=cfg.regressor_batch,
                                               lr=cfg.regressor_lr, key=key,
                                               rng_rows=np.random.default_rng([cfg.train_seed, rep, TAG_R5_FIT, 0]))
        ctx_tr = tr["context"]
        y_star = np.abs(tr["reward"] + settings.gamma * world.v(K, tr["next_state"], ctx_tr) -
                        world.q(K, tr["state"], tr["action"], ctx_tr)) if all(
            r < 1 for r in world.radius(K)) else np.full(len(ctx_tr), np.nan)
        # common random numbers with the ideal regressor (same initialisation and batch rows): the gate compares the
        # two AUROCs, so only their targets differ
        pred_noise, loss_noise = fit_regressor(tr, ev, y=y_star, settings=settings, steps=cfg.regressor_steps,
                                               batch=cfg.regressor_batch, lr=cfg.regressor_lr, key=key,
                                               rng_rows=np.random.default_rng([cfg.train_seed, rep, TAG_R5_FIT, 0]))
        a_ideal, a_noise = auroc(pred_ideal[fin], high[fin]), auroc(pred_noise[fin], high[fin])
        reg = dict(auroc_ideal=a_ideal, auroc_noise_only=a_noise,
                   margin=(a_ideal - a_noise) if a_ideal is not None and a_noise is not None else None,
                   spearman_ideal_E=spearman(pred_ideal, E), spearman_noise_E=spearman(pred_noise, E),
                   final_loss_ideal=loss_ideal, final_loss_noise=loss_noise, high_error_threshold=float(
                       np.quantile(E[fin], m["R5_ideal"]["quantile"])) if fin.any() else None)
    bca = dict(status="switched off (R5(iii) needs --allow-bca-band and a separate approval)")
    if allow_bca_band:
        res = bca_band(world, state, kind, data, settings, rep, cfg, allow=True)
        W = res.pop("W_eval")
        res.pop("W_train")
        bca = dict(status="run", **res, spearman_W_E=spearman(W, E),
                   W_T_wrong_over_T_right=float(W[ctx == G.T_WRONG].mean() / W[ctx == G.T_RIGHT].mean()),
                   spearman_W_confounds={name: spearman(W, x) for name, x in confounds.items()})
    ok = lambda x, f: x is not None and f(x)
    conf_ok = all(ok(v, lambda x: abs(x) <= m["R5_confound"]["max_abs"]) for d in rho.values() for v in d.values())
    marks = dict(R5_confound=conf_ok, R5_tright=ok(tr_tw, lambda x: x <= m["R5_tright"]["max"]),
                 R5_band_spearman=ok(band_vals["spearman_E"], lambda x: x >= m["R5_band_spearman"]["min"]),
                 R5_auroc_wrong_rich=ok(band_vals["auroc_T_wrong_vs_rich"],
                                        lambda x: x >= m["R5_auroc_wrong_rich"]["min"]),
                 R5_auroc_thin_decoy=ok(band_vals["auroc_thin_vs_decoy"],
                                        lambda x: x >= m["R5_auroc_thin_decoy"]["min"]),
                 R5_ideal=ok(reg and reg["margin"], lambda x: x >= m["R5_ideal"]["min_margin"]))
    return dict(fraction=fraction, confound_spearman=rho, mean_E=mean_E, T_right_over_T_wrong=tr_tw, band=band_vals,
                regressors=reg, bca_band=bca, marks=marks, nonfinite_E_rows=int((~fin).sum()))


def stage_r5world(prereg, seed, path="lean", alpha="auto", retune="base", allow_bca_band=False, force=False,
                  log=print):
    guard(prereg)
    alpha, note = resolve_alpha(prereg, retune, path, alpha)
    out = seed_path(prereg, "r5world", retune, alpha, seed)
    if not force and done(out, prereg) and not allow_bca_band:
        return read_json(out)
    started = time.perf_counter()
    cfg = prereg.config
    snaps = {}
    for frac in (cfg.r5_snapshot, *cfg.r5_extra_snapshots):
        try:
            snaps[f"{frac:g}"] = r5_snapshot_values(prereg, retune, seed, alpha, frac,
                                                    regressors=frac == cfg.r5_snapshot,
                                                    allow_bca_band=allow_bca_band and frac == cfg.r5_snapshot,
                                                    log=log)
        except IncompleteRun as exc:  # a run that stopped early fails its seed
            snaps[f"{frac:g}"] = dict(fraction=frac, note=str(exc), confound_spearman={}, T_right_over_T_wrong=None,
                                      band={}, regressors=None, bca_band=None,
                                      marks={k: False for k in ("R5_confound", "R5_tright", "R5_band_spearman",
                                                                "R5_auroc_wrong_rich", "R5_auroc_thin_decoy",
                                                                "R5_ideal")})
    primary = snaps[f"{cfg.r5_snapshot:g}"]
    log(f"r5world {retune} seed {seed}: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}"
                                                      for k, v in primary["marks"].items()))
    payload = stage_payload(prereg, "r5world", started, step=8, retune=retune, alpha=alpha, alpha_note=note, seed=seed,
                            primary_snapshot=cfg.r5_snapshot, snapshots=snaps, marks=primary["marks"],
                            allow_bca_band=allow_bca_band)
    return finish(out, payload)


# ---------------------------------------------------------------------------------------------------------
# Step 9: Level-1 screen and Tier A


def stage_level1(prereg, seed, path="lean", alpha="auto", retune="base", force=False, log=print):
    guard(prereg)
    alpha, note = resolve_alpha(prereg, retune, path, alpha)
    out = seed_path(prereg, "level1", retune, alpha, seed)
    if not force and done(out, prereg):
        return read_json(out)
    started = time.perf_counter()
    cfg = prereg.config
    world, data = world_and_data(cfg, retune, "main", seed)
    settings = train_settings(cfg, alpha)
    try:
        _, state, kind = load_critic(prereg, retune, "main", "HOST", alpha, seed, cfg.tier_a_snapshot, settings)
    except IncompleteRun as exc:  # a run that stopped early fails its seed
        payload = stage_payload(prereg, "level1", started, step=9, retune=retune, alpha=alpha, alpha_note=note,
                                seed=seed, screen=None, or_wrong_minus_even_m2=None, note=str(exc),
                                marks=dict(L1=False))
        return finish(out, payload)
    K_host = kind.gains(state.actor.params)
    screen = T.level1_screen(world, data, state.critic.params, K_host, settings, seed, n_shuf=cfg.n_shuf)
    level = screen["levels"].get("m=2.0", {})
    d = level.get("OR-wrong", {}).get("minus_EVEN")
    passed = d is not None and d > PASS_MARKS["L1"]["min"]
    log(f"level1 {retune} seed {seed}: OR-wrong minus EVEN at m = 2: {d}")
    payload = stage_payload(prereg, "level1", started, step=9, retune=retune, alpha=alpha, alpha_note=note, seed=seed,
                            screen=screen, or_wrong_minus_even_m2=d, marks=dict(L1=passed))
    return finish(out, payload)


def tier_a_families(cfg, variant):
    fam = T.family_ladders(cfg.n_shuf)
    if variant == "main":
        return fam
    return {f: fam[f] for f in cfg.control_families}


def stage_tier_a(prereg, seed, variant="main", path="lean", alpha="auto", retune="base", force=False, log=print):
    guard(prereg)
    if variant not in TIER_A_VARIANTS:
        raise ValueError("variant must be one of " + ", ".join(TIER_A_VARIANTS))
    alpha, note = resolve_alpha(prereg, retune, path, alpha)
    out = tier_a_dir(prereg, retune, variant, alpha, seed)
    if not force and done(out / "stage.json", prereg):
        return read_json(out / "stage.json")
    started = time.perf_counter()
    cfg = prereg.config
    critic_variant = "null" if variant == "null" else "main"
    world, data = world_and_data(cfg, retune, critic_variant, seed)
    settings = train_settings(cfg, alpha, "global_linear" if variant == "global_linear" else "context_linear")
    try:
        _, state, _ = load_critic(prereg, retune, critic_variant, "HOST", alpha, seed, cfg.tier_a_snapshot,
                                  train_settings(cfg, alpha))
    except IncompleteRun as exc:  # a run that stopped early fails its seed (counted, never dropped)
        payload = stage_payload(prereg, "tierA", started, step=9, retune=retune, variant=variant, alpha=alpha,
                                alpha_note=note, seed=seed, status="failed", note=str(exc))
        return finish(out / "stage.json", payload)
    blind, scores = T.tier_a(world, data, state.critic.params, settings, seed, out_dir=out, log=log,
                             steps=cfg.tier_a_steps, refresh=cfg.tier_a_refresh, n_shuf=cfg.n_shuf, tol=cfg.match_tol,
                             max_rounds=cfg.match_rounds, families=tier_a_families(cfg, variant))
    payload = stage_payload(prereg, "tierA", started, step=9, retune=retune, variant=variant, alpha=alpha,
                            alpha_note=note, seed=seed, critic_snapshot=cfg.tier_a_snapshot,
                            blind_sha256=blind["blind_sha256"], blind_file_sha256=blind["file_sha256"],
                            score_file_sha256=scores["file_sha256"], glq_match_sha256=blind["glq_match_sha256"],
                            matching_rounds=blind["rounds"], status="scored")
    return finish(out / "stage.json", payload)


def tier_a_scores(prereg, retune, variant, alpha, seeds):
    """{seed: tierA_score.json} of finished Tier A runs and the list of missing seeds."""
    got, missing = {}, []
    for rep in seeds:
        out = tier_a_dir(prereg, retune, variant, alpha, rep)
        stage = read_json(out / "stage.json") if done(out / "stage.json", prereg) else None
        sc = read_json(out / "tierA_score.json") if stage is not None else None
        if stage is not None and stage.get("status") == "failed":
            got[rep] = dict(score=None, blind=None, note=stage.get("note"))  # a failed seed counts against
        elif sc is None:
            missing.append(f"tierA {variant} seed {rep}")
        else:
            got[rep] = dict(score=sc, blind=read_json(out / "tierA_blind.json"))
    return got, missing


def _matched(score, family, tag):
    if score is None:
        return None
    rec = score["matched"].get(f"{family}@{tag}")
    return rec if rec is not None and rec.get("NS_mix") is not None and rec.get("reachable") else None


def r4_level(scores, families=("OR-wrong", "OR-thin")):
    """'m=2.0' when every scored seed matched the families at S2, else 'm=1.3' (pre-registered fallback)."""
    ok = all(_matched(v["score"], f, "m=2.0") for v in scores.values() if v["score"] for f in families)
    return "m=2.0" if ok else "m=1.3"


def _diff(score, family, tag):
    rec = _matched(score, family, tag)
    return rec["NS_minus_EVEN"] if rec else None


def _shuf_ns(score, tag):
    if score is None:
        return None
    vals = [_matched(score, f, tag) for f in sorted({f.split("@")[0] for f in score["matched"]}) if
            f.startswith("SHUF")]
    vals = [v["NS_mix"] for v in vals if v]
    return float(np.mean(vals)) if vals else None


def r4_evaluate(prereg, retune, path, alpha):
    cfg, m = prereg.config, PASS_MARKS
    seeds = cfg.path_seeds[path]
    scores, missing = tier_a_scores(prereg, retune, "main", alpha, seeds)
    if not scores:
        return dict(complete=False, missing=missing, pass_=None)
    tag = r4_level(scores)
    d1 = paired_t([_diff(v["score"], "OR-wrong", tag) for v in scores.values()])
    d2 = paired_t([_diff(v["score"], "OR-thin", tag) for v in scores.values()])
    best_thin = paired_t([(v["score"] or {}).get("best_OR_thin_minus_best_EVEN") for v in scores.values()])
    best_wrong = paired_t([(v["score"] or {}).get("best_OR_wrong_minus_NONE") for v in scores.values()])
    anchors = []
    for v in scores.values():
        sc = v["score"]
        thin, wrong, anti, shuf = (_matched(sc, "OR-thin", tag), _matched(sc, "OR-wrong", tag),
                                   _matched(sc, "ANTI", tag), _shuf_ns(sc, tag))
        anchors.append(dict(thin_minus_shuf=thin["NS_mix"] - shuf if thin and shuf is not None else None,
                            wrong_minus_shuf=wrong["NS_mix"] - shuf if wrong and shuf is not None else None,
                            shuf_minus_anti=shuf - anti["NS_mix"] if anti and shuf is not None else None))
    n = len(seeds)
    # a pooled mean counts only when every seed of the path contributes: a failed or unmatched seed fails the mark
    # instead of silently leaving the mean (the gate's rule: a failed seed is counted, never dropped)
    mean = lambda k: (lambda t: t["mean"] if t["n"] == n else None)(paired_t([a[k] for a in anchors]))
    cap = paired_t([_diff(v["score"], "CAP-wrong", tag) for v in scores.values()])
    cap_best = paired_t([((v["score"] or {}).get("best_capped_minus_best_EVEN") or {}).get("CAP-wrong")
                         for v in scores.values()])
    need_pos = seed_need(m["R4_1"]["positive_frac"], n)
    gap = m["R4_4"]["min_gap"]
    ge = lambda x, t: x is not None and x >= t
    marks = dict(
        R4_1=ge(d1["mean"], m["R4_1"]["min"]) and d1["p"] is not None and d1["p"] < m["R4_1"]["p_max"] and
        d1["positive"] >= need_pos and d1["n"] == n,
        R4_2=ge(d2["mean"], m["R4_2"]["min"]) and d2["p"] is not None and d2["p"] < m["R4_2"]["p_max"] and
        d2["n"] == n,
        R4_3a=ge(best_thin["mean"], m["R4_3a"]["min"]) and best_thin["n"] == n,
        R4_3b=ge(best_wrong["mean"], m["R4_3b"]["min"]) and best_wrong["n"] == n,
        R4_4=ge(mean("thin_minus_shuf"), gap) and ge(mean("shuf_minus_anti"), gap))
    if cap["n"] == len(scores):
        cap_margin, cap_form = cap["mean"], f"NS(CAP-wrong) - NS(EVEN) matched at {tag}"
    else:
        cap_margin, cap_form = cap_best["mean"], "best CAP-wrong - best EVEN (CAP-wrong unmatched at the level)"
    decision = ("P1L placement (the capped oracle's margin is below 0.02)" if cap_margin is None or cap_margin <
                m["R4_cap"]["decision"] else "the deployed hook's range (capped margin >= 0.02)")
    complete = not missing
    return dict(level=tag, n=n, seeds=list(seeds), need_positive=need_pos, R4_1=d1, R4_2=d2,
                best_OR_thin_minus_best_EVEN=best_thin, best_OR_wrong_minus_NONE=best_wrong, anchors=anchors,
                anchor_means=dict(thin_minus_shuf=mean("thin_minus_shuf"), wrong_minus_shuf=mean("wrong_minus_shuf"),
                                  shuf_minus_anti=mean("shuf_minus_anti")),
                capped=dict(matched=cap, best_minus_best_EVEN=cap_best, margin=cap_margin, form=cap_form,
                            bca_test_uses=decision),
                marks=marks, missing=missing, complete=complete,
                pass_=all(marks.values()) if complete else None)


def stage_r4(prereg, path="lean", alpha="auto", retune="base", force=False, log=print):
    guard(prereg)
    alpha, note = resolve_alpha(prereg, retune, path, alpha)
    out = pooled_path(prereg, "r4", retune, path, alpha)
    started = time.perf_counter()
    res = r4_evaluate(prereg, retune, path, alpha)
    log(f"r4 {retune} {path}: " + ("incomplete" if not res["complete"] else
                                   ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in res["marks"].items())))
    payload = stage_payload(prereg, "r4", started, step=9, retune=retune, path=path, alpha=alpha, alpha_note=note,
                            **res)
    return finish(out, payload)


# ---------------------------------------------------------------------------------------------------------
# Step 10: R7


def r7_evaluate(prereg, retune, path, alpha):
    cfg, m = prereg.config, PASS_MARKS
    seeds = cfg.path_seeds[path]
    r4 = r4_evaluate(prereg, retune, path, alpha)
    out = dict(missing=list(r4.get("missing", [])))
    if r4.get("level") is None:
        return dict(out, complete=False, pass_=None)
    tag, d1 = r4["level"], r4["R4_1"]
    ma = m["R7a"]
    at = mde(d1["sd"], ma["at_seeds"], ma["alpha"], ma["power"])
    out["R7a"] = dict(sd=d1["sd"], n=d1["n"], mde_at_10=at, mde_at_n=mde(d1["sd"], d1["n"], ma["alpha"], ma["power"]),
                      seeds_needed=seeds_for_mde(d1["sd"], ma["max"], ma["max_seeds"], ma["alpha"], ma["power"])
                      if d1["sd"] else None)
    out["R7b"] = dict(wrong_minus_shuf=r4["anchor_means"]["wrong_minus_shuf"],
                      shuf_minus_anti=r4["anchor_means"]["shuf_minus_anti"])
    ctrl = {}
    for variant in ("null", "global_linear"):
        sc, miss = tier_a_scores(prereg, retune, variant, alpha, seeds)
        out["missing"] += miss
        lvl = r4_level(sc, ("OR-wrong",)) if sc else None
        ctrl[variant] = dict(level=lvl, diff=paired_t([_diff(v["score"], "OR-wrong", lvl) for v in sc.values()])
                             if sc else None)
        ctrl[variant]["every_seed"] = bool(ctrl[variant]["diff"]) and ctrl[variant]["diff"]["n"] == len(seeds)
    # Reported, not gated: the null world's NS divides by its own mixture gap, which is the thin contexts' only (its
    # rich behaviour is the expert, so J_BC = J_opt there): 5.6x smaller than the main world's. The same difference in
    # the main world's NS units, for comparison with R4's 0.05:
    w_main, w_null = (G.World(world_spec(cfg, retune, v)) for v in ("main", "null"))
    gap_main, gap_null = (float(w.mix @ (w.j_opt - w.j_bc)) for w in (w_main, w_null))
    d_null = (ctrl["null"]["diff"] or {}).get("mean")
    ctrl["null"].update(mixture_gap_null=gap_null, mixture_gap_main=gap_main,
                        mean_in_main_world_units=d_null * gap_null / gap_main if d_null is not None else None)
    out["R7c"], out["R7_gl"] = ctrl["null"], ctrl["global_linear"]
    scores, _ = tier_a_scores(prereg, retune, "main", alpha, seeds)
    pairs = [(rep, name, rec) for rep, v in scores.items() if v["score"]
             for name, rec in v["score"]["matched"].items() if not rec["family"].startswith("CAP")]
    failed_seeds = [rep for rep, v in scores.items() if not v["score"]]
    good = sum(rec["relative_error"] is not None and rec["reachable"] and rec["relative_error"] <= m["R7d"]["tol"]
               for _, _, rec in pairs)
    if failed_seeds and pairs:  # a failed seed's would-be matches count as misses
        per_seed = len(pairs) / (len(scores) - len(failed_seeds))
        n_pairs = len(pairs) + int(round(per_seed * len(failed_seeds)))
    else:
        n_pairs = len(pairs)
    out["R7d"] = dict(pairs=n_pairs, within_tol=good, share=good / n_pairs if n_pairs else None,
                      unreachable=sum(not rec["reachable"] for _, _, rec in pairs), failed_seeds=failed_seeds)
    null_pairs = {}
    for v in scores.values():
        if not v["score"]:
            continue
        for rec in v["score"]["null_shuf_pairs"].get(tag, []):
            null_pairs.setdefault("-".join(rec["pair"]), []).append(rec["NS_difference"])
    stats_pairs = {k: paired_t(v) for k, v in null_pairs.items()}
    beyond = sum(s["se"] is not None and s["se"] > 0 and abs(s["mean"]) > m["R7_null_pairs"]["se"] * s["se"]
                 for s in stats_pairs.values())
    out["R7_null_pairs"] = dict(level=tag, pairs=stats_pairs, beyond_2se=beyond, n_pairs=len(stats_pairs))
    ge = lambda x, t: x is not None and x >= t
    lt = lambda x, t: x is not None and x < t
    gap = m["R7b"]["min_gap"]
    out["marks"] = dict(
        R7a=at is not None and at <= ma["max"],
        R7b=ge(out["R7b"]["wrong_minus_shuf"], gap) and ge(out["R7b"]["shuf_minus_anti"], gap),
        R7c=ctrl["null"]["every_seed"] and lt(abs(ctrl["null"]["diff"]["mean"]) if ctrl["null"]["diff"]["mean"]
                                               is not None else None, m["R7c"]["max_abs"]),
        R7_gl=ctrl["global_linear"]["every_seed"] and lt(ctrl["global_linear"]["diff"]["mean"], m["R7_gl"]["max"]),
        R7d=ge(out["R7d"]["share"], m["R7d"]["min"]),
        R7_null_pairs=len(stats_pairs) > 0 and beyond <= m["R7_null_pairs"]["max"])
    out["R7e"] = dict(status="not applicable", reason=PASS_MARKS["R7e"]["mark"])
    out["level"] = tag
    out["complete"] = not out["missing"]
    out["pass_"] = all(out["marks"].values()) if out["complete"] else None
    return out


def stage_r7(prereg, path="lean", alpha="auto", retune="base", force=False, log=print):
    guard(prereg)
    alpha, note = resolve_alpha(prereg, retune, path, alpha)
    out = pooled_path(prereg, "r7", retune, path, alpha)
    started = time.perf_counter()
    res = r7_evaluate(prereg, retune, path, alpha)
    log(f"r7 {retune} {path}: " + ("incomplete" if not res["complete"] else
                                   ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in res["marks"].items())))
    payload = stage_payload(prereg, "r7", started, step=10, retune=retune, path=path, alpha=alpha, alpha_note=note,
                            **res)
    return finish(out, payload)


# ---------------------------------------------------------------------------------------------------------
# Step 11: Tier B (co-trained oracle arms)


def stage_tier_b(prereg, seed, arm, path="full", alpha="auto", retune="base", force=False, log=print):
    guard(prereg)
    cfg = prereg.config
    if arm not in cfg.tier_b_arms:
        raise ValueError("arm must be one of " + ", ".join(cfg.tier_b_arms))
    alpha, note = resolve_alpha(prereg, retune, path, alpha)
    out = tier_b_dir(prereg, retune, alpha, seed, arm)
    if not force and done(out / "stage.json", prereg):
        return read_json(out / "stage.json")
    r4 = r4_evaluate(prereg, retune, path, alpha)
    if not r4.get("complete"):
        raise PreregError(f"tierB needs every main Tier A of the {path} path first (the knobs come from Tier A at "
                          f"R4's level): missing " + ", ".join(r4.get("missing", [])))
    tag = r4["level"]
    blind = read_json(tier_a_dir(prereg, retune, "main", alpha, seed) / "tierA_blind.json")
    if blind is None:
        raise FileNotFoundError(f"tierB needs the main Tier A of seed {seed}")
    m_even = float(tag.split("=")[1])
    knob = m_even if arm == "EVEN" else (blind["matches"].get(f"{arm}@{tag}") or {}).get("knob")
    started = time.perf_counter()
    if knob is None:
        payload = stage_payload(prereg, "tierB", started, step=11, retune=retune, alpha=alpha, seed=seed, arm=arm,
                                level=tag, knob=None, status="unreachable in Tier A at this level: not trained")
        return finish(out / "stage.json", payload)
    world, data = world_and_data(cfg, retune, "main", seed)
    settings = train_settings(cfg, alpha)
    try:
        _, state, _ = load_critic(prereg, retune, "main", "HOST", alpha, seed, cfg.tier_a_snapshot, settings)
    except IncompleteRun as exc:
        payload = stage_payload(prereg, "tierB", started, step=11, retune=retune, alpha=alpha, seed=seed, arm=arm,
                                level=tag, knob=float(knob), status="failed", note=str(exc), score=None)
        return finish(out / "stage.json", payload)
    TA = T.TierA(world, data, state.critic.params, settings, seed, steps=cfg.tier_a_refresh,
                 refresh=cfg.tier_a_refresh, n_shuf=cfg.n_shuf)
    table = TA.table(arm, float(knob))
    rec = T.cotrain(world, data, settings, "HOST", seed, out_dir=out, multiplier=table, log=log)
    K = final_gains(rec)
    K_host = final_gains(_run_record(prereg, retune, "main", "HOST", alpha, seed))
    S = float(T.M.strength(K, K_host, TA.Sigma_ev)) if K is not None and K_host is not None else None
    payload = stage_payload(prereg, "tierB", started, step=11, retune=retune, alpha=alpha, alpha_note=note, seed=seed,
                            arm=arm, level=tag, knob=float(knob), table_mean=float(table.mean()),
                            run_json_sha256=rec["run_json_sha256"], score=world.score(K) if K is not None else None,
                            S_vs_host=S, status="trained" if K is not None else "stopped early: " + stopped_note(rec))
    return finish(out / "stage.json", payload)


def tier_b_evaluate(prereg, retune, path, alpha):
    cfg, m = prereg.config, PASS_MARKS
    seeds = cfg.path_seeds[path]
    r4 = r4_evaluate(prereg, retune, path, alpha)
    margin_a = r4.get("R4_1", {}).get("mean") if r4.get("level") else None
    per, missing = {}, []
    for rep in seeds:
        recs = {arm: read_json(tier_b_dir(prereg, retune, alpha, rep, arm) / "stage.json") for arm in cfg.tier_b_arms}
        if any(r is None or r.get("prereg_sha256") != prereg.digest for r in recs.values()):
            missing.append(f"tierB seed {rep}")
            continue
        ns = {arm: (r["score"]["NS_mix"] if r.get("score") else None) for arm, r in recs.items()}
        d = ns["OR-wrong"] - ns["EVEN"] if ns["OR-wrong"] is not None and ns["EVEN"] is not None else None
        ok = d is not None and margin_a is not None and d > 0 and d >= m["TB"]["min_fraction"] * margin_a
        per[rep] = dict(NS=ns, d=d, pass_=ok, S={arm: r.get("S_vs_host") for arm, r in recs.items()})
    need = seed_need(m["TB"]["seed_frac"], len(seeds))
    count = sum(v["pass_"] for v in per.values())
    complete = not missing and margin_a is not None
    return dict(tier_a_margin=margin_a, level=r4.get("level"), per_seed=per, passing_seeds=count, need=need,
                missing=missing, complete=complete, pass_=(count >= need) if complete else None)


# ---------------------------------------------------------------------------------------------------------
# Step 12: verdict


def _fmt(x, digits=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.{digits}g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{digits}f}"
    return str(x)


def _row(step, mark_id, measured, result, seeds="", note=""):
    mk = PASS_MARKS.get(mark_id, {})
    return dict(step=step, id=mark_id, requirement=mk.get("name", mark_id), pass_mark=mk.get("mark", ""),
                measured=measured, result=result, seeds=seeds, note=note)


def _res(flag):
    return "not run" if flag is None else ("pass" if flag else "FAIL")


def verdict_rows(prereg, path, retune):
    """Every requirement's row (requirement, pass mark, measured, result, seeds) and each step's status."""
    cfg = prereg.config
    seeds = cfg.path_seeds[path]
    n = len(seeds)
    rows, step_status = [], {}

    def status(step, flags):
        step_status[step] = _combine(flags)

    # step 1
    bc = read_json(prereg.root / "build_checks" / "build_checks.json")
    flags = []
    for k in ("B1", "B2", "B3", "B4", "B5", "B6", "B7"):
        c = bc["checks"][k] if bc and bc.get("prereg_sha256") == prereg.digest else None
        meas = "-"
        if c:
            meas = {"B1": lambda c: (f"max residual {_fmt(c['max_rel_residual'])}; control "
                                     f"{_fmt(c['min_control_residual'])}"),
                    "B2": lambda c: f"max |z| {_fmt(c['max_abs_z'])} ({c['episodes']} episodes)",
                    "B3": lambda c: f"{_fmt(c['max_forward_backward'])}; {_fmt(c['max_lyapunov'])}",
                    "B4": lambda c: f"max ratio {_fmt(c['max_ratio'])}",
                    "B5": lambda c: f"{sum(c['checks'].values())}/{len(c['checks'])} bitwise",
                    "B6": lambda c: f"{sum(c['checks'].values())}/{len(c['checks'])} identities",
                    "B7": lambda c: (f"ran {c.get('ran')}, failures {c.get('failures')}, errors {c.get('errors')}"
                                     if c.get("pass_") is not None else c.get("note", "-"))}[k](c)
        flag = None if c is None else c["pass_"]
        flags.append(flag)
        rows.append(_row(1, k, meas, _res(flag)))
    status(1, flags)
    # step 2
    s0 = read_json(prereg.root / "stage0" / retune / "stage0.json")
    flags = []
    for k in ("S1", "S2", "S3", "S4", "DECOY", "S5", "S6", "STAB"):
        c = s0["marks"][k] if s0 and s0.get("prereg_sha256") == prereg.digest else None
        meas = "-"
        if c:
            meas = {"S5": lambda c: f"min PR {_fmt(c['min_participation_ratio'])}, transient "
                                    f"{_fmt(c['pooled']['transient_share'])}, IACT {_fmt(c['pooled']['iact'])}",
                    "STAB": lambda c: f"transplant sqrt(gamma) rho {_fmt(c['transplant'])}",
                    "S3": lambda c: f"{_fmt(c['measured'])} g, radius {_fmt(c['radius'])}"}.get(
                k, lambda c: _fmt(c["measured"]))(c)
        flags.append(None if c is None else c["pass_"])
        rows.append(_row(2, k, meas, _res(flags[-1])))
    status(2, flags)
    # steps 3-4
    recs = {rep: read_json(prereg.root / "r0r1" / retune / f"seed{rep}.json") for rep in seeds}
    recs = {k: v for k, v in recs.items() if v and v.get("prereg_sha256") == prereg.digest}
    for step, keys in ((3, ("R0_pr", "R0_transient", "R0_iact", "R0_scale")), (4, ("R1a_rich", "R1a_thin", "R1b"))):
        flags = []
        for k in keys:
            if len(recs) < n:
                flags.append(None)
                rows.append(_row(step, k, "-", "not run", f"{len(recs)}/{n} seeds run"))
                continue
            c = sum(r["marks"][k]["pass_"] for r in recs.values())
            vals = [r["marks"][k]["measured"] for r in recs.values()]
            if k == "R0_pr":
                meas = f"min {_fmt(min(min(v.values()) for v in vals))}"
            elif k == "R1a_thin":
                ratio = [x / float(rk) for v in vals for rk, ctxs in v.items() if float(rk) > 0
                         for pair in ctxs.values() for x in pair]
                meas = (f"{len(cfg.ladder_rungs)} rungs; SD / rung {_fmt(min(ratio))} to {_fmt(max(ratio))}"
                        if ratio else "-")
            else:
                meas = f"{_fmt(min(vals))} to {_fmt(max(vals))}"
            flags.append(c == n)
            rows.append(_row(step, k, meas, _res(c == n), f"{c}/{n} (need {n})"))
        status(step, flags)
    # step 5: R2 (at the host alpha; the ladder's first while R3 has not chosen)
    chosen, alpha_note, r3_evals = host_alpha(prereg, retune, path)
    r2 = r2_evaluate(prereg, retune, path, chosen if chosen is not None else float(cfg.alpha_ladder[0]))
    flags = []
    if r2["complete"]:
        for k in ("R2_ratio", "R2_thin_E2", "R2_rel"):
            parts = []
            ok_all = True
            for mode in ("HOST", "FQE"):
                per = r2["per_seed"][mode]
                c = sum(v["marks"][k] for v in per.values())
                key = {"R2_ratio": "ratio", "R2_thin_E2": "thin_E2", "R2_rel": "rel"}[k]
                vals = [v[key] for v in per.values() if v[key] is not None]
                parts.append(f"{mode} {_fmt(min(vals)) if vals else '-'}-{_fmt(max(vals)) if vals else '-'} ({c}/{n})")
                ok_all &= c >= r2["need"]
            rows.append(_row(5, k, "; ".join(parts), _res(ok_all), f"need {r2['need']}/{n} per critic",
                             "a seed passes when all three R2 marks hold"))
        flags.append(r2["marks"]["main"]["HOST"] and r2["marks"]["main"]["FQE"])
        lad = r2["ladder"]
        rows.append(_row(5, "R2_ladder_falls", "; ".join(f"{m_} falls {lad[m_]['falls']}/{lad[m_]['steps']}"
                                                       for m_ in lad), _res(r2["marks"]["ladder_falls"]),
                         f"{len(cfg.ladder_seeds)} ladder seeds"))
        rows.append(_row(5, "R2_ladder_top", "; ".join(f"{m_} {_fmt(lad[m_]['top_ratio'])}" for m_ in lad),
                         _res(r2["marks"]["ladder_top"]), f"{len(cfg.ladder_seeds)} ladder seeds"))
        flags += [r2["marks"]["ladder_falls"], r2["marks"]["ladder_top"]]
    else:
        for k in ("R2_ratio", "R2_thin_E2", "R2_rel", "R2_ladder_falls", "R2_ladder_top"):
            rows.append(_row(5, k, "-", "not run"))
        flags.append(None)
    status(5, flags)
    # step 6: R3 over the alpha ladder (live)
    r3s = [(a, r) for a, r in r3_evals.items() if r["complete"]]
    show = r3_evals.get(chosen) if chosen is not None else (r3s[-1][1] if r3s else None)
    if show:
        for k in ("R3_cost", "R3_ns_gap", "R3_thin_shortfall", "R3_rich_ns"):
            key = {"R3_cost": "cost_share_of_gap", "R3_ns_gap": None, "R3_thin_shortfall": "thin_shortfall_share",
                   "R3_rich_ns": "rich_NS_host"}[k]
            per = show["per_seed"].values()
            if key is None:
                vals = [v["NS_exact"] - v["NS_host"] for v in per if v["NS_host"] is not None]
                meas = (f"NS(EXACT) - NS(host) {_fmt(min(vals))} to {_fmt(max(vals))}" if vals else "-")
            else:
                vals = [v[key] for v in per if v[key] is not None]
                meas = f"{_fmt(min(vals)) if vals else '-'} to {_fmt(max(vals)) if vals else '-'}"
            c = show["passing_by_mark"][k]
            rows.append(_row(6, k, meas + f" (alpha {show['alpha']:g})", _res(c >= show["need"]),
                             f"{c}/{n} (need {show['need']})", "a seed passes when all four R3 marks hold"))
        tried = ", ".join(f"{a:g}: {'pass' if r['pass_'] else 'fail'}" for a, r in r3s)
        all_tried = len(r3s) == len(cfg.alpha_ladder)
        nxt = next((a for a in cfg.alpha_ladder if a not in [x for x, _ in r3s]), None)
        rows.append(dict(step=6, id="alpha", requirement="Host alpha (ladder 2.5, 5, 10)",
                         pass_mark="smallest alpha whose R3 passes", measured=tried,
                         result="pass" if chosen is not None else ("FAIL" if all_tried else "not run"), seeds="",
                         note=(f"host alpha {chosen:g}" if chosen is not None else
                               (f"next: critics HOST and EXACT and r3 at alpha {nxt:g}" if nxt else ""))))
        step_status[6] = True if chosen is not None else (False if all_tried else None)
    else:
        for k in ("R3_cost", "R3_ns_gap", "R3_thin_shortfall", "R3_rich_ns"):
            rows.append(_row(6, k, "-", "not run"))
        step_status[6] = None
    alpha = chosen if chosen is not None else float(cfg.alpha_ladder[0])
    # step 7: R6 and R1(c)
    _seed_marks(rows, step_status, prereg, "r6", retune, alpha, seeds, 7, ("R6a", "R6b", "R6c", "R6d", "R1c"),
                lambda r, k: {"R6a": r["host"]["a"], "R6b": r["host"]["b"],
                              "R6c": max([v for v in r["kish_host"].values() if v is not None], default=None),
                              "R6d": max(r["bc_start"]["a"], r["bc_start"]["b"]), "R1c": r["R1c_share"]}[k])
    # step 8: R5 world gates
    _seed_marks(rows, step_status, prereg, "r5world", retune, alpha, seeds, 8,
                ("R5_confound", "R5_tright", "R5_band_spearman", "R5_auroc_wrong_rich", "R5_auroc_thin_decoy",
                 "R5_ideal"), _r5_measure)
    # step 9: Level-1 screen (not a stop), R4
    _seed_marks(rows, {}, prereg, "level1", retune, alpha, seeds, 9, ("L1",),
                lambda r, k: r["or_wrong_minus_even_m2"], note="screen, not a stop")
    r4 = r4_evaluate(prereg, retune, path, alpha)
    if r4.get("complete"):
        t1, t2 = r4["R4_1"], r4["R4_2"]
        meas = {"R4_1": f"mean {_fmt(t1['mean'])}, p {_fmt(t1['p'])}, positive {t1['positive']}/{t1['n']}",
                "R4_2": f"mean {_fmt(t2['mean'])}, p {_fmt(t2['p'])}",
                "R4_3a": f"mean {_fmt(r4['best_OR_thin_minus_best_EVEN']['mean'])}",
                "R4_3b": f"mean {_fmt(r4['best_OR_wrong_minus_NONE']['mean'])}",
                "R4_4": f"OR-thin - SHUF {_fmt(r4['anchor_means']['thin_minus_shuf'])}, SHUF - ANTI "
                        f"{_fmt(r4['anchor_means']['shuf_minus_anti'])}"}
        for k in ("R4_1", "R4_2", "R4_3a", "R4_3b", "R4_4"):
            rows.append(_row(9, k, meas[k] + f" at {r4['level']}", _res(r4["marks"][k]), f"{n} seeds pooled"))
        rows.append(_row(9, "R4_cap", f"{r4['capped']['form']}: {_fmt(r4['capped']['margin'])}", "report",
                         f"{n} seeds", "BCA test uses " + r4["capped"]["bca_test_uses"]))
        step_status[9] = all(r4["marks"].values())
    else:
        for k in ("R4_1", "R4_2", "R4_3a", "R4_3b", "R4_4", "R4_cap"):
            rows.append(_row(9, k, "-", "not run"))
        step_status[9] = None
    # step 10: R7
    r7 = r7_evaluate(prereg, retune, path, alpha) if r4.get("level") else dict(complete=False)
    if r7.get("complete"):
        meas = {"R7a": f"MDE at 10 seeds {_fmt(r7['R7a']['mde_at_10'])} (SD {_fmt(r7['R7a']['sd'])}); seeds needed "
                       f"{_fmt(r7['R7a']['seeds_needed'])}",
                "R7b": f"OR-wrong - SHUF {_fmt(r7['R7b']['wrong_minus_shuf'])}, SHUF - ANTI "
                       f"{_fmt(r7['R7b']['shuf_minus_anti'])}",
                "R7c": f"mean {_fmt(r7['R7c']['diff']['mean'] if r7['R7c']['diff'] else None)}",
                "R7_gl": f"mean {_fmt(r7['R7_gl']['diff']['mean'] if r7['R7_gl']['diff'] else None)}",
                "R7d": f"{r7['R7d']['within_tol']}/{r7['R7d']['pairs']} within 5% ({r7['R7d']['unreachable']} "
                       f"unreachable)",
                "R7_null_pairs": f"{r7['R7_null_pairs']['beyond_2se']}/{r7['R7_null_pairs']['n_pairs']} beyond 2 SE"}
        for k in ("R7a", "R7b", "R7c", "R7_gl", "R7d", "R7_null_pairs"):
            rows.append(_row(10, k, meas[k], _res(r7["marks"][k]), f"{n} seeds pooled"))
        step_status[10] = all(r7["marks"].values())
    else:
        for k in ("R7a", "R7b", "R7c", "R7_gl", "R7d", "R7_null_pairs"):
            rows.append(_row(10, k, "-", "not run"))
        step_status[10] = None
    rows.append(_row(10, "R7e", "-", "n/a"))
    # step 11: Tier B (full path)
    tb = tier_b_evaluate(prereg, retune, path, alpha) if r4.get("level") else dict(complete=False)
    if tb.get("complete"):
        ds = [v["d"] for v in tb["per_seed"].values() if v["d"] is not None]
        rows.append(_row(11, "TB", f"OR-wrong - EVEN {_fmt(min(ds) if ds else None)} to {_fmt(max(ds) if ds else None)}"
                                   f"; Tier A margin {_fmt(tb['tier_a_margin'])}", _res(tb["pass_"]),
                         f"{tb['passing_seeds']}/{n} (need {tb['need']})"))
        step_status[11] = tb["pass_"]
    else:
        rows.append(_row(11, "TB", "-", "not run", "", "lean path: Tier A only" if path == "lean" else ""))
        step_status[11] = None
    return rows, step_status, dict(alpha=alpha, alpha_note=alpha_note, r4=r4 if r4.get("level") else None)


def _r5_measure(r, k):
    p = r["snapshots"][f"{r['primary_snapshot']:g}"]
    if k == "R5_confound":
        vals = [abs(v) for d in p["confound_spearman"].values() for v in d.values() if v is not None]
        return max(vals) if vals else None
    return {"R5_tright": p["T_right_over_T_wrong"], "R5_band_spearman": p["band"]["spearman_E"],
            "R5_auroc_wrong_rich": p["band"]["auroc_T_wrong_vs_rich"],
            "R5_auroc_thin_decoy": p["band"]["auroc_thin_vs_decoy"],
            "R5_ideal": (p["regressors"] or {}).get("margin")}[k]


def _seed_marks(rows, step_status, prereg, stage, retune, alpha, seeds, step, keys, measure, note=""):
    n = len(seeds)
    recs = {rep: read_json(seed_path(prereg, stage, retune, alpha, rep)) for rep in seeds}
    recs = {k: v for k, v in recs.items() if v and v.get("prereg_sha256") == prereg.digest}
    flags = []
    for k in keys:
        frac = PASS_MARKS[k].get("seed_frac", 1.0)
        need = seed_need(frac, n)
        if len(recs) < n:
            rows.append(_row(step, k, "-", "not run", f"{len(recs)}/{n} seeds run", note))
            flags.append(None)
            continue
        c = sum(bool(r["marks"][k]) for r in recs.values())
        vals = [measure(r, k) for r in recs.values()]
        vals = [v for v in vals if v is not None]
        meas = f"{_fmt(min(vals))} to {_fmt(max(vals))}" if vals else "-"
        rows.append(_row(step, k, meas, _res(c >= need), f"{c}/{n} (need {need})", note))
        flags.append(c >= need)
    if step_status is not None:
        step_status[step] = _combine(flags)


def _combine(flags):
    """A step's status from its marks: False if any fails, None if any is missing, else True."""
    if any(f is False for f in flags):
        return False
    if not flags or any(f is None for f in flags):
        return None
    return True


def outcome(prereg, retune, step_status, path="full"):
    s = step_status
    later = RETUNE_ORDER[RETUNE_ORDER.index(retune) + 1:] if retune in RETUNE_ORDER else RETUNE_ORDER
    if any(s.get(k) is False for k in (1, 2, 3, 4)):
        return "build failure: steps 1-4 must pass before any training (no retune applies)"
    stop = [k for k in STOP_STEPS if s.get(k) is False]
    if stop:
        nxt = f"next pre-registered retune: {later[0]} ({RETUNE_TEXT[later[0]]})" if later else \
            "the world is REJECTED (retunes exhausted); BCA is never run in it"
        return f"step(s) {', '.join(map(str, stop))} fail at this world ({retune}): {nxt}"
    required = range(1, 11) if path == "lean" else range(1, 12)
    failed = [k for k in (7, 10, 11) if s.get(k) is False]
    if failed:
        return f"NOT ACCEPTED: step(s) {', '.join(map(str, failed))} fail"
    pending = [k for k in required if s.get(k) is None]
    if pending:
        return "incomplete: step(s) " + ", ".join(map(str, pending)) + " not run (or not finished)"
    scope = "error-tracking and thin-data placement" if s.get(8) else \
        "thin-data placement only (R5 fails: no residual band can track this error here)"
    if path == "lean":
        return (f"LEAN PASS (steps 1-10, {scope}). The full gate adds 10 seeds and Tier B (step 11) before any BCA "
                "measurement")
    return (f"ACCEPTED for {scope}. Next, under a separate approval: measure BCA's band on frozen snapshots, then "
            "the BCA arms")


def verdict_markdown(prereg, path, retune, rows, step_status, verdict, extra):
    lines = [f"# Gapped LQ acceptance verdict ({path} path, world {retune})", "",
             f"Pre-registration SHA-256 `{prereg.digest}` (preset {prereg.config.preset}). Host alpha "
             f"{extra['alpha']:g} ({extra['alpha_note']}). Seeds: {list(prereg.config.path_seeds[path])}.", "",
             f"**Outcome:** {verdict}", "", "| Step | Requirement | Pass mark | Measured | Result | Seeds |",
             "|---|---|---|---|---|---|"]
    esc = lambda x: str(x).replace("|", "\\|")
    for r in rows:
        note = f" ({r['note']})" if r.get("note") else ""
        lines.append(f"| {r['step']} | {esc(r['requirement'])} | {esc(r['pass_mark'])} | "
                     f"{esc(r['measured'])}{esc(note)} | {r['result']} | {esc(r['seeds'])} |")
    lines += ["", "| Step | Status |", "|---|---|"]
    lines += [f"| {k} {STEP_NAMES[k]} | {_res(step_status.get(k))} |" for k in range(1, 12)]
    lines += ["", "Every number is from a pre-registered stage of this digest. Seeds: a mark 'in >= f of 10 seeds' "
              "needs >= ceil(f n) of the path's n seeds. 'not run' means the stage output is missing.", ""]
    return "\n".join(lines)


def stage_verdict(prereg, path="lean", retune="base", log=print):
    guard(prereg)
    started = time.perf_counter()
    rows, step_status, extra = verdict_rows(prereg, path, retune)
    verdict = outcome(prereg, retune, step_status, path)
    payload = stage_payload(prereg, "verdict", started, step=12, retune=retune, path=path, alpha=extra["alpha"],
                            rows=rows, step_status={str(k): v for k, v in step_status.items()}, outcome=verdict)
    out = prereg.root / "verdict" / retune / path
    payload = finish(out / "verdict.json", payload)
    md = verdict_markdown(prereg, path, retune, rows, step_status, verdict, extra)
    (out / "verdict.md").write_text(md, encoding="utf8")
    log(md)
    return payload


# ---------------------------------------------------------------------------------------------------------
# CLI


def plan(cfg, path="lean", retune="base", prefix="python experiments/signal/glq_acceptance.py"):
    """Every command of a path in order; blocks separated by comments can run in parallel within a block."""
    seeds = cfg.path_seeds[path]
    r = "" if retune == "base" else f" --retune {retune}"
    p = f" --path {path}"
    a0 = f"{cfg.alpha_ladder[0]:g}"
    out = [f"# {path} path, world {retune}: {len(seeds)} seeds {list(seeds)}",
           "# step 0 (once per pre-registration)", f"{prefix} prereg",
           "# steps 1-4 (no training)", f"{prefix} build_checks",
           f"{prefix} stage0{r}" + (" --search" if not r else "")]
    out += [f"{prefix} r0r1 --seeds {k}{r}" for k in seeds]
    out += ["# --- approval boundary: training below (critics and oracle actors; no BCA arm) ---",
            f"# step 5-6: critics at alpha {a0} (one process each)"]
    out += [f"{prefix} critics --seeds {k} --mode {m}{r}" for k in seeds for m in MODES]
    out += [f"{prefix} critics --seeds {k} --mode {m} --variant {rung_variant(cfg, rung)}{r}"
            for rung in cfg.ladder_rungs if rung_variant(cfg, rung) != "main" for k in cfg.ladder_seeds
            for m in ("HOST", "FQE")]
    out += [f"{prefix} r3{p} --alpha {a0}{r}",
            f"# if r3 fails at alpha {a0}: critics --mode HOST and EXACT at --alpha {cfg.alpha_ladder[1]:g} for every "
            f"seed, then r3 --alpha {cfg.alpha_ladder[1]:g}; then {cfg.alpha_ladder[2]:g} likewise",
            f"{prefix} r2{p}{r}", "# steps 7-9 (per seed; --alpha auto = the host alpha chosen by r3)"]
    out += [f"{prefix} {st} --seeds {k}{p}{r}" for st in ("r6", "r5world", "level1") for k in seeds]
    out += [f"{prefix} tierA --seeds {k}{p}{r}" for k in seeds]
    out += [f"{prefix} critics --seeds {k} --mode HOST --variant null --alpha auto{p}{r}" for k in seeds]
    out += [f"{prefix} tierA --seeds {k} --variant {v}{p}{r}" for v in ("null", "global_linear") for k in seeds]
    out += [f"{prefix} r4{p}{r}", f"{prefix} r7{p}{r}"]
    if path == "full":
        out += ["# step 11: Tier B (co-trained oracle arms at Tier A's knobs)"]
        out += [f"{prefix} tierB --seeds {k} --arm {arm}{p}{r}" for k in seeds for arm in cfg.tier_b_arms]
    out += ["# step 12", f"{prefix} verdict{p}{r}"]
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(p, seeds=False, path=False, alpha=False):
        p.add_argument("--prereg", default=str(PREREG_DEFAULT))
        p.add_argument("--runs", default=str(RUNS_DEFAULT))
        p.add_argument("--retune", default="base", choices=list(RETUNES))
        p.add_argument("--force", action="store_true")
        p.add_argument("--allow-dependency-change", action="store_true")
        if seeds:
            p.add_argument("--seeds", nargs="+", type=int, required=True)
        if path:
            p.add_argument("--path", default="lean", choices=("lean", "full"))
        if alpha:
            p.add_argument("--alpha", default="auto")

    p = sub.add_parser("prereg", help="write the pre-registration (step 0)")
    p.add_argument("--prereg", default=str(PREREG_DEFAULT))
    p.add_argument("--preset", default="real", choices=list(PRESETS))
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("plan", help="print every command of a path")
    p.add_argument("--prereg", default=str(PREREG_DEFAULT))
    p.add_argument("--path", default="lean", choices=("lean", "full"))
    p.add_argument("--retune", default="base", choices=list(RETUNES))
    p = sub.add_parser("build_checks", help="step 1 (no training)")
    common(p)
    p.add_argument("--no-unittests", action="store_true")
    p = sub.add_parser("stage0", help="step 2 (closed form)")
    common(p)
    p.add_argument("--search", action="store_true")
    common(sub.add_parser("r0r1", help="steps 3-4 (log only)"), seeds=True)
    p = sub.add_parser("critics", help="step 5-6 inputs: HOST / FQE / EXACT co-training (trains)")
    common(p, seeds=True)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--variant", default="main")
    p.add_argument("--alpha", default=None, help="a ladder alpha (default: the first), or auto (needs r3)")
    p.add_argument("--path", default="lean", choices=("lean", "full"))
    common(sub.add_parser("r2", help="step 5 (pooled)"), path=True, alpha=True)
    p = sub.add_parser("r3", help="step 6 (pooled, one alpha)")
    common(p, path=True)
    p.add_argument("--alpha", default=None)
    common(sub.add_parser("r6", help="step 7 (per seed)"), seeds=True, path=True, alpha=True)
    p = sub.add_parser("r5world", help="step 8 (per seed; frozen snapshots)")
    common(p, seeds=True, path=True, alpha=True)
    p.add_argument("--allow-bca-band", action="store_true", help="R5(iii): needs a separate approval")
    common(sub.add_parser("level1", help="step 9 screen (per seed)"), seeds=True, path=True, alpha=True)
    p = sub.add_parser("tierA", help="step 9 Tier A (per seed; trains oracle actors)")
    common(p, seeds=True, path=True, alpha=True)
    p.add_argument("--variant", default="main", choices=TIER_A_VARIANTS)
    common(sub.add_parser("r4", help="step 9 (pooled)"), path=True, alpha=True)
    common(sub.add_parser("r7", help="step 10 (pooled)"), path=True, alpha=True)
    p = sub.add_parser("tierB", help="step 11 (per seed and arm; trains)")
    common(p, seeds=True, path=True, alpha=True)
    p.add_argument("--arm", required=True)
    common(sub.add_parser("verdict", help="step 12"), path=True)
    return parser


def main(argv=None):
    ns = build_parser().parse_args(argv)
    if ns.stage == "prereg":
        if ns.preset != "real" and Path(ns.prereg).resolve() == PREREG_DEFAULT.resolve():
            raise SystemExit("the default pre-registration path takes the real preset only")
        digest = write_prereg(ns.prereg, PRESETS[ns.preset], ns.force)
        print(f"pre-registration {ns.prereg}\nsha256 {digest}\nruns go to {RUNS_DEFAULT / digest[:16]}")
        return
    if ns.stage == "plan":
        cfg = PRESETS["real"]
        if Path(ns.prereg).exists():
            cfg = config_from_dict(json.loads(Path(ns.prereg).read_text(encoding="utf8"))["config"])
        print("\n".join(plan(cfg, ns.path, ns.retune)))
        return
    prereg = load_prereg(ns.prereg, ns.runs, ns.allow_dependency_change)
    kw = dict(retune=ns.retune)
    st = ns.stage
    if st == "build_checks":
        stage_build_checks(prereg, unittests=not ns.no_unittests, force=ns.force)
    elif st == "stage0":
        stage_stage0(prereg, search=ns.search, force=ns.force, **kw)
    elif st == "r0r1":
        for k in ns.seeds:
            stage_r0r1(prereg, k, force=ns.force, **kw)
    elif st == "critics":
        for k in ns.seeds:
            stage_critics(prereg, k, ns.mode, ns.variant, ns.alpha, path=ns.path, force=ns.force, **kw)
    elif st == "r2":
        stage_r2(prereg, ns.path, alpha=ns.alpha, force=ns.force, **kw)
    elif st == "r3":
        stage_r3(prereg, ns.path, ns.alpha, force=ns.force, **kw)
    elif st in ("r6", "r5world", "level1", "tierA", "tierB"):
        for k in ns.seeds:
            if st == "r6":
                stage_r6(prereg, k, ns.path, ns.alpha, force=ns.force, **kw)
            elif st == "r5world":
                stage_r5world(prereg, k, ns.path, ns.alpha, allow_bca_band=ns.allow_bca_band, force=ns.force, **kw)
            elif st == "level1":
                stage_level1(prereg, k, ns.path, ns.alpha, force=ns.force, **kw)
            elif st == "tierA":
                stage_tier_a(prereg, k, ns.variant, ns.path, ns.alpha, force=ns.force, **kw)
            else:
                stage_tier_b(prereg, k, ns.arm, ns.path, ns.alpha, force=ns.force, **kw)
    elif st == "r4":
        stage_r4(prereg, ns.path, ns.alpha, force=ns.force, **kw)
    elif st == "r7":
        stage_r7(prereg, ns.path, ns.alpha, force=ns.force, **kw)
    elif st == "verdict":
        stage_verdict(prereg, ns.path, **kw)


if __name__ == "__main__":
    main()
