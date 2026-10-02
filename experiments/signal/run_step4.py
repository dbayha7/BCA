"""Step 4 runner: one (block, level, replicate) per process. Sigma fits, assignments, knob ladders, blind strength
matching, fixed-point and Adam K's and the Level-1/2 diagnostics of every step-4 arm; never J.

Design: experiments/signal/STEP4_DESIGN.md (sections 2-7 and 12); expectations: runs/wbcp_signal/step4/expectations_DRAFT.md.
The machinery is experiments/signal/placement_harness.py (PH), which calls lq_harness without editing it. This module
neither imports nor calls lq_harness.value(), lq_harness.policy_gradient() or any other J function (test_run_step4
scans the source and runs a process with both replaced by functions that raise). It writes the K's, knobs, strengths and
reachability, records their SHA-256, and score_values.py computes J afterwards (design section 5, "Blindness").

Blocks, levels and cells (design section 7). R: step 3's episodic data (lq_harness.Harness.data, seed 20261001, so
replicates 0-4 are step 3's), levels good / poor, start K_0 = K_b + offset D (offsets 0.5, 1, 2), cases clean,
noisy_reward, q1_optimistic and shared_bias at kappa 0.5 / 1 / 2, independent_errors, and q2_optimistic (kappa 0.5 / 1
/ 2) on replicate 0 only, a bitwise check against clean. X-CS / X-EP: PH.data_common / PH.data_episodic (seed
20261002), levels expert, mixed, medium, poor, start 0.5 K* + 0.5 D, cases clean, noisy_reward, tilt a* 0.9 / 0.5 / 0 /
-0.5, q1_optimistic and shared_bias at kappa 1, independent_errors, cone at kappa 1 / 2 (--cone-kappas, the pilot's
retune). C: common states with 4 contexts, levels expert / poor, cases clean, q1opt-local (kappa 1, --blockc-kappa),
tilt-local (a* -0.5). Clean runs first in each offset group, because its sigma is every other case's NUIS.

Per cell (CellRun):
1. Signal. sigma by PH.fit_signal (block R: the same fit through lq_harness's own compiled program and models, so the
   fit, threshold and dose are step 3's), the hook's dose (td3_bc_bca.bc_readout), step 3's same-mean controls
   (lq_harness.control_doses: constant, shuffled with step 3's permutation in block R, oracle |e1| at logged actions),
   and PH.assignments with 16 SHUF and 16 STRAT permutations (the Adam regimes use the first 4 of each, same seeds).
2. FP (fixed point; lambda locked at lambda_0 from Q1 at pi_0). NONE, EXACT, the P0 ladder and the 200-point uniform
   frontier; P1L, P2L, P1pi, P2pi for every assignment on the beta ladder; OR-raw(t) = 1 + t (oracle - 1) on a t
   ladder; P3: the closed form Q1 - c e1 for P3-OR where e1 >= 0 by construction, and BFGS
   (PH.fixed_point_penalty, warm-started along the c ladder, the design's 1e-5 criterion) for P3-SIG / NUIS (and P3-OR
   where e1 changes sign) in X-CS and C only (design section 10's estimate). S(c) of a BFGS arm can jump: the ReLU
   calibrator makes the penalty piecewise linear in a, so the minimiser can switch as c crosses a threshold. The tiny
   smoke had P3 matches stuck 0.1-0.3% off target after 40 bisection steps, unchanged when BFGS ran to 1e-9; they are
   kept, flagged (converged = False) and counted by the procedural check, never forced; P3 with native lambda (lambda = alpha / mean |Q1 - c W| at pi_0, the
   lambda side channel) on the ladder only; C2 (block C), the per-context LSTD penalty critic of W at logged actions,
   closed form. References HOOK, CONST-old, SHUF-old and OR-raw (t = 1). T-lin robustness (design section 2), FP
   only: P1L / P2L with M = W / mean W against SHUF-lin (the same magnitudes under SHUF_k's permutations). The s = 0.47
   multiset is a separate run with --rank-s 0.47 (the Adam programs read Settings.rank_s too).
3. A500 (500 Adam steps, step 3's batches and keys). One vmapped PH.program per Spec: NONE (weighted path and None
   path), EXACT, P0, P1L / P2L (rows of one weight table), P1pi / P2pi (re-ranked at the current actor, per kind), P3
   (sig, nuis, or), OR-raw and the references. Block R also runs step 3's anchors (none, bca, constant, shuffled,
   oracle) through lq_harness.actor_loop itself (Harness.actor), the rows its J continuity is read from.
4. Subsets (replicates 0-4, or --subsets on): A5000 (5,000 steps; the 5,000-step stream extends the 500-step one) for
   X-CS tilt and cone cells and block C's localized cells: NONE, P0, P1L / P2L / P2pi with SIG, SHUF x4, STRAT x4, P3
   SIG / NUIS, matched again; full-batch A500 (FB; every training row in every step) for X-CS tilt and cone: NONE, P0,
   P1L / P2L with SIG and SHUF x4, matched at S2.
5. Matching (design section 5), per regime, cell and arm, every permutation separately: S = sqrt(tr(delta Sigma_ev
   delta^T)) against the regime's NONE; targets S_k = S(P0(m_k)), m = 1.3 / 2 / 4; a target below max(2e-3, 0.02
   S_host) is below floor and not matched; PH.match_knob (first crossing, bisection in log knob; FP to 1e-3 in at most
   40 steps, Adam regimes one interpolated rerun plus at most 6 bisection steps to 5%; never extrapolated).
   batched_match runs PH.match_knob itself for every (arm, level) in lock-step, so each round's evaluations go
   through one batched solve or one vmapped program call; the knob sequence is match_knob's own (tested). P0 is also
   matched to S(HOOK) and S(OR-raw), on the branch of its ladder on the reference's side of m = 1 (p0_branch: P0(1)
   is NONE, so S is V-shaped in m and the whole ladder's first crossing would land on m < 1 for a dose mean above 1).
6. Diagnostics. Level 1 at K_0 on all training rows, exact: per-row K-gradients of each term (float64, from the
   float64 critic gradients at pi_0; the penalty's from JAX), so any arm's term gradients are weighted sums
   (test_run_step4 checks them against PH.terms), and PH.level1 without grad J (cos(-Delta g, grad J) and the
   efficiency need J: score_values.py adds them from the stored terms). Level 2 for every Adam run: the means over steps
   of the term gradient norms, BC:Q ratio, |dK|_F, Adam SNR and Kish n_eff, the path length, the last step's BC loss,
   mean Q1 and lambda. For every kept row: S, the displacement from K_0 (Sigma metric and Frobenius), the pull share
   cos_Sigma(delta, K_bc - K(NONE)); at FP the matrix's minimum eigenvalue and condition number (BFGS: its success flag)
   and the nearest point of the 200-point uniform frontier (m*, outcome-free targeting share). For every matched arm
   at S_k, the K's the decomposition G_reg / G_perp needs (K_N + a delta-hat_P0 and its complement, in the Sigma inner
   product), so score_values computes both orders without knowing the design. Outcome-free correlations: Spearman of
   SIG's weights with |e1| at the family's evaluation point, with the leverages and with context 0; corr with the
   poor-mode indicator.
7. Built-in [derived] checks (the cell's G0 inputs): same multiset (bitwise), zero knob (bitwise at FP and, for P1L,
   P2L, P1pi, P2pi and OR-raw, at A500; P3 at c = 0 only up to float32, kind 'eps' 1e-6: XLA compiles the
   zero-weighted penalty path into the gradient and rounds the Q-term differently, 2e-9 to 3e-8 absolute seen after 200
   steps), hooks off = td3_bc_update and P1 at the hook's dose = step 3's bca path (bitwise, against
   lq_harness.actor_loop; compiled single runs, as test_placement_harness; not for the cone or block C, whose critic
   step 3's QuadraticCritic does not represent),
   scale equivalence (FP 1e-6 every beta; A500 1e-4, eps), constant penalty = NONE (bitwise / identical), P3-OR = EXACT
   (closed form 1e-6, per-row action gradient 1e-4), the cancellation identities in their exact form (the
   finite-sample gap lambda_0 kappa A^-1 vec((K_bc,w - K-bar_b) Sigma_w) predicted, as test_placement_harness), block
   C's context locality (1e-10), NUIS = SIG in clean, q2_optimistic = clean bitwise (block R replicate 0), the
   multiset of the run's settings (rank_s, rank_clip; never the defaults) bit for bit: every arm's sorted global
   weights, and every step's sorted batch weights of every pi arm at A500 (a compiled, vmapped trace of the pi
   programs, pi_weight_trace; [derived]), and the matching tolerance (procedural: reachable matches that did not
   converge are counted).
8. OR-raw(t) = 1 + t (o_i - 1) would be negative on rows with o_i < 1 - 1/t (anti-BC there). The pre-registration
   (freeze decision G) clips every BC weight at 0, in every regime. Recorded, outcome-free: per cell the oracle's
   minimum, t_nonneg = 1 / (1 - min o) and the count of clipped rows at each ladder t ('or_raw'); the per-row
   neg_bc count of the (clipped) recipes is then 0 by construction.

Outputs, in OUTPUT/<block>/<level>/rep<replicate> (a new directory): arrays.npz ('K/<cell>/<regime>' every kept K of a
regime in table order, 'K/<cell>/K0', 'K/<cell>/FP_frontier'; 'D/...' diagnostics aligned with the table rows),
matched_knobs.json (per cell and regime: the row labels, the index of NONE / EXACT / ladders / matched rows / references
/ decomposition rows, knobs, S, reachability and floors; the checks; settings; timing) and, written last, HASHES.sha256
(sha256sum format, both files). score_values.py refuses a directory without it, or whose files no longer match it.

JAX_PLATFORMS=cpu python experiments/signal/run_step4.py --block X-CS --level expert --replicate 99 --output DIR
    [--plan full|reduced] [--cases ...] [--subsets auto|on|off] [--cone-kappas 1 2] [--blockc-kappa 1]
    [--train-episodes ... and every other PH.Settings field]
Compilation: every cell of a process reuses the same compiled programs, because the optimizer objects are shared
(optimizers(): TrainState's tx is a static pytree field, so a fresh optimizer per cell recompiled every program per
cell, about 6,000 memory maps each); caches are cleared only if the process nears vm.max_map_count.
Set the BLAS thread variables (OPENBLAS_NUM_THREADS etc.) before Python starts when several processes run at once.
Tests: JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_run_step4
"""

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

import algorithms.td3_bc as BASE  # noqa: E402
import algorithms.td3_bc_bca as P  # noqa: E402
from calibration.reference import initial_reference  # noqa: E402
from experiments.signal import lq_harness as LQ  # noqa: E402
from experiments.signal import placement_harness as PH  # noqa: E402

BLOCKS = ("R", "X-CS", "X-EP", "C")
R_SEED = 20261001
R_QUALITY = dict(good="expert", poor="poor")  # block R's per-level assignment seeds use these levels' indices
LEVELS = {"R": LQ.BEHAVIORS, "X-CS": PH.QUALITIES, "X-EP": PH.QUALITIES, "C": ("expert", "poor")}
X_KAPPA, CONE_KAPPAS, C_KAPPA, C_TILT = 1.0, (1.0, 2.0), 1.0, -0.5
PILOT_REPLICATE = 99
ANALYSIS_REPLICATES = tuple(range(10))
SUBSET_REPLICATES = tuple(range(5))
STREAM_TAG = {"X-CS": 100, "X-EP": 101, "C": 102}  # stream seeds [seed, rep, tag, 2 | 3], apart from every data tag
TAG_SHUF_OLD = 4  # step 3's shuffled-dose permutation [seed, rep, level, 4]
NATURAL = ("q1_optimistic", "shared_bias", "independent_errors")
NONNEGATIVE = ("q1_optimistic", "q2_optimistic", "shared_bias", "cone", "q1opt-local")  # e1 >= 0 by construction
LOCALIZED = ("q1opt-local", "tilt-local")
N_PERM_FP, N_PERM_ADAM = 16, 4
OR_T_LADDER = (0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0)
# P0's two branches around m = 1 (P0(1) is NONE, so S(P0(m)) is V-shaped there): the reference matches (S(HOOK),
# S(OR-raw)) search the branch on the reference's side of m = 1, never the whole ladder (see p0_branch).
P0_UP = tuple(m for m in PH.P0_LADDER if m >= 1.0)
P0_DOWN = tuple(m for m in PH.P0_LADDER if m <= 1.0)
LEVEL_NAMES = ("S1", "S2", "S3")
FLOOR_ABS, FLOOR_REL = 2e-3, 0.02
MATCH = dict(FP=dict(tol=1e-3, max_bisect=40, interpolate=False),
             ADAM=dict(tol=0.05, max_bisect=6, interpolate=True))
REGIMES = dict(A500=dict(steps=None, full_batch=False, diagnostics=True),
               A5000=dict(steps=5000, full_batch=False, diagnostics=False),
               FB=dict(steps=None, full_batch=True, diagnostics=False))
BUCKETS = (4, 8, 16, 32, 64, 96, 128, 160, 192, 256)  # vmapped run counts (padded), so few shapes compile
L_TABLE_ROWS = 24  # fixed weight-table height, so one compiled program serves every cell of a process
MAP_BUDGET = 40000  # clear JAX's caches before the process nears vm.max_map_count (65,530); values do not change
J_FUNCTIONS = ("value", "policy_gradient")  # lq_harness's J and grad J: never imported or called here
L2_MEANS = ("g_q_norm", "g_bc_norm", "g_pen_norm", "bc_q_ratio", "step_norm", "adam_snr", "kish_bc", "kish_q", "kish_w")
L2_FIELDS = tuple("mean_" + k for k in L2_MEANS) + ("path_length", "final_bc_loss", "final_q_mean", "final_lambda",
                                                    "min_kish_w", "max_kish_w")
L1_FIELDS = ("rho", "m_eff", "tau", "identified", "cos_correction", "cos_pull", "norm_g_Q", "norm_g_BC", "norm_g_pen",
             "norm_dg")
FP_FIELDS = ("min_eig", "cond", "finite", "bfgs_success")
PLACEMENT_KEYS = {("L", "P1"): "P1L", ("L", "P2"): "P2L", ("pi", "P1"): "P1pi", ("pi", "P2"): "P2pi"}


# ---------------------------------------------------------------------------------------------------------
# Grid, settings, data and cells


@dataclass(frozen=True)
class CellSpec:
    case: str
    kappa: float = None
    a_star: float = None
    offset: float = None

    @property
    def id(self):
        parts = [self.case]
        for tag, x in (("k", self.kappa), ("a", self.a_star), ("o", self.offset)):
            if x is not None:
                parts.append(f"{tag}{x:g}")
        return "_".join(parts)


def grid(block, rep, cone_kappas=CONE_KAPPAS, blockc_kappa=C_KAPPA, cases=None):
    """The block's cells in run order (clean first in each offset group; its sigma is the others' NUIS)."""
    if block == "R":
        out = []
        for o in LQ.DEFAULT_OFFSETS:
            out += [CellSpec("clean", offset=o), CellSpec("noisy_reward", offset=o)]
            out += [CellSpec(c, kappa=k, offset=o) for c in ("q1_optimistic", "shared_bias") for k in LQ.DEFAULT_KAPPAS]
            out.append(CellSpec("independent_errors", offset=o))
            if rep == 0:
                out += [CellSpec("q2_optimistic", kappa=k, offset=o) for k in LQ.DEFAULT_KAPPAS]
    elif block in ("X-CS", "X-EP"):
        out = [CellSpec("clean"), CellSpec("noisy_reward")]
        out += [CellSpec("tilt", a_star=a) for a in PH.TILT_ALIGNMENTS]
        out += [CellSpec("q1_optimistic", kappa=X_KAPPA), CellSpec("shared_bias", kappa=X_KAPPA),
                CellSpec("independent_errors")]
        out += [CellSpec("cone", kappa=float(k)) for k in cone_kappas]
    elif block == "C":
        out = [CellSpec("clean"), CellSpec("q1opt-local", kappa=float(blockc_kappa)),
               CellSpec("tilt-local", a_star=C_TILT)]
    else:
        raise ValueError("unknown block " + repr(block))
    return [c for c in out if cases is None or c.case in cases]


def settings_for(block, overrides=None):
    """PH.Settings of a block: step 3's seed 20261001 for R, 20261002 otherwise, then the overrides."""
    base = PH.Settings(seed=R_SEED) if block == "R" else PH.Settings()
    return replace(base, **{k: v for k, v in (overrides or {}).items() if v is not None})


def data_r(h, level, rep):
    """Block R: step 3's rows (lq_harness.Harness.data) with the fields and meta the placement harness reads."""
    data = h.lq.data(level, rep)
    quality = R_QUALITY[level]
    for split in data.values():
        split.update(state=split["obs"], next_state=split["next_obs"],
                     gain=np.full(len(split["obs"]), PH.QUALITY_GAIN[quality]))
    tr = data["train"]
    data["meta"] = dict(quality=quality, rep=rep, mode="step3", contexts=None, K_bar=h.lq.behavior_gain(level),
                        K_bc=PH.ls_gain(tr["state"], tr["action"]),  # as placement_harness._meta
                        tie_perm=PH.seed_rng(h, rep, PH.QUALITIES.index(quality), PH.TAG_TIE).permutation(len(tr["state"])))
    return data


def block_data(h, block, level, rep):
    if block == "R":
        return data_r(h, level, rep)
    if block == "X-EP":
        return PH.data_episodic(h, level, rep)
    return PH.data_common(h, level, rep, h.cfg.contexts if block == "C" else None)


def block_stream(h, block, level, rep, n_train, steps=None):
    """Batch rows and keys: step 3's stream in block R (tag = behaviour index), one tag per block otherwise."""
    tag = LQ.BEHAVIORS.index(level) if block == "R" else STREAM_TAG[block]
    owner = h if steps is None else SimpleNamespace(cfg=replace(h.cfg, actor_steps=steps))
    return PH.streams(owner, rep, tag, n_train)


def perm_old(h, block, level, rep, n):
    """SHUF-old's permutation: step 3's [seed, rep, behaviour, 4] in block R, [seed, rep, level, 4] otherwise."""
    if block == "R":
        return np.random.default_rng([h.cfg.seed, rep, LQ.BEHAVIORS.index(level), 4]).permutation(n)
    return PH.seed_rng(h, rep, PH.QUALITIES.index(level), TAG_SHUF_OLD).permutation(n)


@dataclass
class StartCell(PH.Cell):
    """A PH.Cell with its own start gain (block R: K_0 = K_b + offset D)."""
    start: Any = None

    @property
    def K0(self):
        return self.start


def critic_r(h, data, spec, level, rep, K_pi, K_b):
    """Block R's float64 critic, exactly as lq_harness.Harness.replicate builds it (independent_errors included)."""
    info, forms, scale = {}, None, None
    if spec.case == "independent_errors":
        forms, scale, ratio, attempt = LQ.independent_errors(h.system, data["train"], K_pi, h.args,
                                                             (h.cfg.seed, rep, LQ.BEHAVIORS.index(level), 6),
                                                             h.cfg.independent_ratio)
        info = dict(independent_scale=scale, independent_ratio=ratio, independent_redraws=attempt)
    return LQ.case_critic(h.system, spec.case, K_pi, K_b, spec.kappa, forms, scale), info


def make_cell(h, block, level, rep, spec, data):
    """The PH.Cell of one (block, level, replicate, case): float64 critic at the block's start, float32 agent state."""
    if block == "R":
        K_b = h.lq.behavior_gain(level)
        K_pi = K_b + spec.offset * LQ.DIRECTION
        params, info = critic_r(h, data, spec, level, rep, K_pi, K_b)
        return StartCell(h, data, params, PH.native(h, K_pi, params), info=info, start=K_pi)
    if block == "C":
        params, info = PH.critic_params_c(h, data, spec.case, spec.kappa, spec.a_star)
        return PH.make_cell(h, data, params, contexts=True, info=info)
    params, info = PH.critic_params_x(h, data, spec.case, spec.kappa, spec.a_star)
    return PH.make_cell(h, data, params, info=info)


def step3_alignment(params, s, K_pi, K_b):
    """alignment_K by lq_harness.Harness.replicate's own NumPy expressions (block R continuity)."""
    ds = s.shape[1]
    pi = -s @ K_pi.T
    z = np.concatenate([s, pi], 1)
    H = params["H"]
    g_true = 2 * z @ H[:, ds:]
    g_q1 = g_true + 2 * params["kappa"][0] * (pi + s @ K_b.T) + 2 * z @ params["W"][0][:, ds:]
    return LQ.cosine(g_q1.T @ -s, g_true.T @ -s)


def optimizers(shared, h):
    """The process's optimizer objects. TrainState's apply_fn and tx are static fields (part of the pytree
    structure), so a fresh torch_adam / adam / set_to_zero per cell, as PH.native, PH.fit_signal and
    lq_harness.Harness.native create, makes every compiled program recompile for every cell (about 6,000 new memory
    maps per cell, towards vm.max_map_count). Sharing them changes no array and no value."""
    if "optimizers" not in shared:
        shared["optimizers"] = dict(actor=BASE.C.torch_adam(h.args.lr), critic=optax.set_to_zero(),
                                    calibrator=optax.adam(h.config.cal_lr))
    return shared["optimizers"]


def native_state(shared, h, K, params, models):
    """PH.native (lq_harness.Harness.native with models = h.lq.models) with the process's optimizer objects."""
    tx = optimizers(shared, h)
    actor = TrainState.create(apply_fn=models[0].apply, params={"K": jnp.asarray(K, jnp.float32)}, tx=tx["actor"])
    critic = TrainState.create(apply_fn=models[1].apply, params=PH.to_jax32(params), tx=tx["critic"])
    return BASE.AgentTrainState(actor, actor, critic, critic)


def fit_signal(h, shared, native, train, cal, stream, contexts=False, lq_path=False, signal=1):
    """PH.fit_signal line for line with the process's calibrator optimizer; lq_path runs the fit through
    lq_harness's own compiled program and models (block R: step 3's sigma, threshold and dose)."""
    models = h.lq.models if lq_path else (h.context_models if contexts else h.models)
    fit = h.lq.fit if lq_path else h.fit[contexts]
    da = h.system.dims[1]
    cal_state = TrainState.create(apply_fn=models[2].apply,
                                  params=models[2].init(stream.init_key, jnp.zeros((1, train.obs.shape[1])),
                                                        jnp.zeros((1, da))),
                                  tx=optimizers(shared, h)["calibrator"])
    state = P.State(native, cal_state, jnp.asarray(1.0), initial_reference(cal_state.params))
    state, (accepted, loss, _) = fit(state, train, stream.fit_rows, stream.fit_keys, jnp.int32(signal))
    reference, diag, _, _ = LQ.calibrate_signal(h.args, h.config, models, state, cal, stream.cal_key, signal,
                                                LQ.MAX_ACTION)
    return reference, dict(diag, fit_accepted=float(np.mean(accepted)), fit_loss_final=float(np.mean(loss[-100:])))


# ---------------------------------------------------------------------------------------------------------
# Matching in lock-step


class _Pending(Exception):
    """Raised inside PH.match_knob when it asks for a strength that has not been evaluated yet."""

    def __init__(self, knob):
        super().__init__(knob)
        self.knob = knob


def batched_match(problems, evaluate, max_rounds=64):
    """PH.match_knob for many problems at once, each round's new evaluations in one call.

    problems: dicts of match_knob's arguments (ladder, S_ladder, target, tol, max_bisect, interpolate). Every round
    reruns match_knob from scratch for each unfinished problem with a memo of the strengths evaluated so far; the first
    knob missing from the memo is collected, and evaluate([(problem index, knob), ...]) returns their strengths. As
    match_knob is deterministic in those values, the knob sequence and the result are exactly match_knob's.
    """
    memo = [dict() for _ in problems]
    out = [None] * len(problems)
    for _ in range(max_rounds):
        need = []
        for i, p in enumerate(problems):
            if out[i] is not None:
                continue

            def S_of(k, m=memo[i]):
                if k not in m:
                    raise _Pending(k)
                return m[k]

            try:
                out[i] = PH.match_knob(p["ladder"], S_of, p["target"], p["tol"], p["max_bisect"], p["interpolate"],
                                       S_ladder=p["S_ladder"])
            except _Pending as e:
                need.append((i, e.knob))
        if not need:
            return out
        for (i, k), s in zip(need, evaluate(need)):
            memo[i][k] = float(s)
    raise RuntimeError("matching did not finish within max_rounds")


# ---------------------------------------------------------------------------------------------------------
# Rows, arms and programs


class Arm(NamedTuple):
    """A matchable arm of one regime: its knob ladder, executor group, executor item per knob and Level-1 recipe."""
    name: str
    ladder: tuple
    group: tuple
    item: Callable
    recipe: Callable


class Table:
    """Every gain one regime of a cell evaluates, in order: K (float64), a label (arm, knob, phase), the Level-1
    recipe (recipe function, knob) and per-row extras (FP solver flags, Level-2 summaries)."""

    def __init__(self, shape):
        self.shape = shape
        self.K, self.labels, self.recipes, self.fp, self.l2 = [], [], [], [], []
        self.index = dict(ladder={}, matched={}, refs={}, anchors={}, decomp={}, checks={})
        self.row_of, self.vbar = {}, {}

    def add(self, K, label, recipe=None, fp=None, l2=None, vbar=None):
        self.K.append(np.asarray(K, np.float64).reshape(self.shape))
        self.labels.append(label)
        self.recipes.append(recipe)
        self.fp.append(fp)
        self.l2.append(l2)
        if vbar is not None:
            self.vbar[len(self.K) - 1] = np.asarray(vbar, np.float64)
        return len(self.K) - 1

    def kept_rows(self):
        """Rows the index refers to; intermediate bisection rows are dropped when the table is written."""
        idx = self.index
        keep = {v for k, v in idx.items() if isinstance(v, int)}
        keep |= {r for rows in idx["ladder"].values() for r in rows}
        keep |= {m["row"] for levels in idx["matched"].values() for m in levels.values() if m.get("row") is not None}
        keep |= set(idx["refs"].values()) | set(idx["anchors"].values()) | set(idx["checks"].values())
        keep |= {r for levels in idx["decomp"].values() for parts in levels.values() for r in parts.values()}
        return sorted(keep)


def perm_index(name):
    """k of SHUF_k / STRAT_k; -1 for every other assignment."""
    m = re.fullmatch(r"(?:SHUF|STRAT)(\d+)", name)
    return int(m.group(1)) if m else -1


def bucket(n):
    return next(b for b in BUCKETS if b >= n)


def run_program(program, native, train, rows, keys, ties, ctx, items):
    """The vmapped program over run dicts (a, b, beta, c, assign), padded to a bucket size and chunked at 256."""
    Ks, vbars, metrics = [], [], []
    for start in range(0, len(items), BUCKETS[-1]):
        chunk = items[start:start + BUCKETS[-1]]
        padded = chunk + [chunk[-1]] * (bucket(len(chunk)) - len(chunk))
        runs = {k: jnp.asarray([float(it.get(k, d)) for it in padded], jnp.float32)
                for k, d in (("a", 1.0), ("b", 0.0), ("beta", 0.0), ("c", 0.0))}
        runs["assign"] = jnp.asarray([int(it.get("assign", 0)) for it in padded], jnp.int32)
        state, v_bar, m = program(native, train, rows, keys, ties, ctx, runs)
        Ks.append(np.asarray(state.actor.params["K"], np.float64)[:len(chunk)])
        vbars.append(np.asarray(v_bar["K"], np.float64)[:len(chunk)])
        metrics.append({k: np.asarray(v, np.float64)[:len(chunk)] for k, v in m.items()})
    return (np.concatenate(Ks), np.concatenate(vbars),
            {k: np.concatenate([m[k] for m in metrics]) for k in metrics[0]})


def l2_summary(metrics, j):
    """Level-2 summary of run j: means over steps, path length, the last step's BC loss, mean Q1 and lambda."""
    get = lambda k: metrics[k][j] if k in metrics else None
    out = [float(np.mean(get(k))) if get(k) is not None else math.nan for k in L2_MEANS]
    step = get("step_norm")
    kish_w = get("kish_w")
    out += [float(np.sum(step)) if step is not None else math.nan, float(get("bc_loss")[-1]),
            float(get("q_mean")[-1]), float(get("lambda")[-1]),
            float(np.min(kish_w)) if kish_w is not None else math.nan,
            float(np.max(kish_w)) if kish_w is not None else math.nan]
    return np.asarray(out)


def map_count():
    try:
        with open("/proc/self/maps", encoding="utf8") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def clear_if_needed(log=print):
    """jax.clear_caches() when the process's memory maps near the kernel limit (step 3's lesson); values unchanged."""
    n = map_count()
    if n is not None and n > MAP_BUDGET:
        jax.clear_caches()
        gc.collect()
        log(f"cleared JAX caches at {n} memory maps (now {map_count()})")


def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(stats.spearmanr(x, y)[0])


def inner_sigma(X, Y, Sigma):
    """<X, Y>_Sigma = tr(X Sigma Y^T), batched over leading axes; per-context blocks sum."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    if np.ndim(Sigma) == 2:
        return np.einsum("...ij,jk,...ik->...", X, Sigma, Y)
    return np.einsum("...cij,cjk,...cik->...", X, Sigma, Y)


def vec(K):
    """Column-major vec of one gain (da, ds), the fixed-point system's convention."""
    return np.asarray(K).T.reshape(-1)


def fp_matrix(T, lam0, v=1.0, rows=None, n=None):
    """Half the loss Hessian of the fixed point at Q weight 1 and BC weight v (symmetrised, as PH.solve_fp)."""
    sel = slice(None) if rows is None else rows
    count = T["Tq"][sel].shape[0] if n is None else n
    vv = np.broadcast_to(np.asarray(v, np.float64), (T["Tq"].shape[0],))[sel]
    m = T["ds"] * T["da"]
    A = ((-lam0 * T["Tq"][sel].sum(0) + (vv @ T["Tb"][sel]) / T["da"]) / count).reshape(m, m)
    return (A + A.T) / 2


def p0_branch(mean_weight):
    """The P0 ladder branch on which a reference with mean BC weight d-bar is matched: m >= 1 (1, 1.15, ..., 10) when
    d-bar >= 1, else m <= 1 (0.5, 0.75, 1). P0(1) equals NONE, so S(P0(m)) is 0 at m = 1 and grows on both sides (S(0.5)
    can exceed S(1.3)); a first crossing on the whole ladder lands on the weaker-BC branch m < 1 for any target below
    S(0.5), the opposite of a stronger-BC reference's uniform counterpart. Each branch starts at S = 0, so a target at
    or above the floor is bracketed whenever the branch's top reaches it. Returns (ladder, label)."""
    return (P0_UP, "m>=1") if mean_weight >= 1.0 else (P0_DOWN, "m<=1")


def negative_weights(recipe, n):
    """Rows with a negative BC weight v_i under a Level-1 recipe (OR-raw(t) = 1 + t (o_i - 1) for t > 1 / (1 - o_i);
    every other arm's v is positive by construction); None without a recipe."""
    if recipe is None:
        return None
    fn, knob = recipe
    v = np.broadcast_to(np.asarray(fn(knob).get("v", 1.0), np.float64), (n,))
    return int(np.sum(v < 0))


def pi_weight_trace(args, models, spec, native_state, train, indices, keys, ties, ctx, run):
    """PH.placement_loop for one pi-family run, line for line (no Adam second-moment sum), that also returns every
    step's batch weights w_t (steps, B): the vector the program feeds into bc_w = a + b w_t and q_w = 1 / (1 + beta
    w_t). w_t is the multiset gathered at the batch ranks and moved by perm[t] (no arithmetic), so its sorted values do
    not depend on the knob or the trajectory; test_run_step4 checks that the final state equals placement_loop's bit
    for bit. Used only by the pi-family multiset check (expectations Part 2 item 3)."""
    critic_params = native_state.critic.params
    pen_fn = PH.penalty_fn(spec.pen, models, critic_params, ctx)
    exact_fn = (lambda obs, a: models[1].exact(critic_params, obs, a)) if spec.exact else None
    pen_c = run["c"] if spec.pen is not None else None

    def body(state, xs):
        t, rows, key, tie = xs
        batch = jax.tree_util.tree_map(lambda x: x[rows], train)
        pi = jax.lax.stop_gradient(models[0].apply(state.actor.params, batch.obs))
        x = PH.assignment_at(spec.kind, models, critic_params, ctx, batch.obs, pi)
        w = PH.batch_rank_normal(x, spec.rank_s, spec.rank_clip, tie, ctx["reverse"][run["assign"]])
        w = w[ctx["perm"][run["assign"], t]]
        bc_w = None if spec.bc == "none" else jnp.maximum(run["a"] + run["b"] * w, 0.0)  # as placement_loop
        q_w = 1.0 / (1.0 + run["beta"] * w) if spec.trust else None
        new, _ = PH.placement_step(args, models, state, batch, key, bc_w, q_w, pen_fn, pen_c, exact_fn, None, False)
        return new, w

    return jax.lax.scan(body, native_state, (jnp.arange(len(indices)), indices, keys, ties))


def jsonable(x):
    """Strict-JSON copy: NumPy scalars and arrays to Python, NaN and +-inf to None, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if hasattr(x, "item") and np.ndim(x) == 0:
        return jsonable(x.item())
    return x


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------
# One cell


class CellRun:
    """Signal, fixed point, Adam regimes, diagnostics and checks of one (block, level, replicate, case)."""

    def __init__(self, h, block, level, rep, spec, data, stream, shared, plan="full", subsets=False, log=print):
        self.h, self.block, self.level, self.rep, self.spec = h, block, level, rep, spec
        self.data, self.stream, self.shared, self.plan, self.subsets, self.log = (data, stream, shared, plan, subsets,
                                                                                  log)
        self.contexts = block == "C"
        self.C = h.cfg.contexts if self.contexts else None
        self.cell = make_cell(h, block, level, rep, spec, data)
        self.cell.native = native_state(shared, h, self.cell.K0, self.cell.params, self.cell.models)
        tr, ev = data["train"], data["eval"]
        self.tr, self.n = tr, len(tr["state"])
        self.obs, self.s, self.a = tr["obs"], tr["state"], tr["action"]
        self.context = tr.get("context") if self.contexts else None
        self.K0 = np.asarray(self.cell.K0, np.float64)
        self.Kshape = self.K0.shape
        self.sigma_ev = PH.second_moment(ev["state"], ev.get("context") if self.contexts else None, self.C)
        self.sigma_tr = PH.second_moment(self.s, self.context, self.C)
        self.K_bc = data["meta"]["K_bc_context"] if self.contexts else data["meta"]["K_bc"]
        self.noisy = spec.case == "noisy_reward"
        self.train32, self.cal32 = PH.transitions(tr, self.noisy), PH.transitions(data["cal"], self.noisy)
        self.tables, self.checks, self.timing = {}, [], {}
        self.lam0 = PH.lambda0(self.cell.params, self.obs, self.K0, h.args.alpha)
        self.lq_ok = not self.contexts and "cone" not in self.cell.params  # step 3's QuadraticCritic is the critic
        self.native_lq = native_state(shared, h, self.K0, self.cell.params, h.lq.models) if self.lq_ok else None
        self.reduced = plan == "reduced" and block in ("R", "X-EP")
        self.executors = dict(solve=self.fp_solve, combo=self.fp_combo, bfgs=self.fp_bfgs, prog=self.adam_exec)

    # -- helpers --------------------------------------------------------------------------------------------

    def check(self, name, item, ok, measured=None, tol=None, kind="derived"):
        """kind: 'derived' (a miss fails G0), 'eps' ([derived up to float32/eps]: reported; fails G0 only above 100x
        tol) or 'procedural' (reported)."""
        self.checks.append(dict(name=name, item=item, kind=kind, ok=bool(ok), value=measured, tol=tol))

    def solve(self, T, u, v, lam=None):
        lam = self.lam0 if lam is None else lam
        if self.contexts:
            return PH.solve_fp_context(T, self.context, u, v, lam, self.C)
        return PH.solve_fp(T, u, v, lam)

    def strength(self, K, K_ref):
        return PH.strength(K, K_ref, self.sigma_ev)

    def S_row(self, reg, r):
        tab = self.tables[reg]
        return float(self.strength(tab.K[r], tab.K[tab.index["NONE"]]))

    def recipe_fn(self, name):
        """Level-1 recipe (Q weights u, BC weights v, Q read, penalty, c, lambda) of an arm at a knob."""
        if name == "P0":
            return lambda m: dict(v=m)
        if name == "ORraw":
            o = self.doses["oracle"].astype(np.float64)
            return lambda t: dict(v=np.maximum(1.0 + t * (o - 1.0), 0.0))
        if name.startswith(("P1", "P2")):
            placement, arm = name.split(":")
            fam = "L" if placement.endswith("L") else "pi"
            if arm.startswith(("TLIN", "SHUFLIN")):
                lin, perms = self.tlin_weights()
                w = lin if arm == "TLIN" else lin[perms[int(arm[7:])]]
            else:
                w = self.assign["weights"][fam][placement[:2]][arm]
            if placement.startswith("P1"):
                return lambda b: dict(v=1.0 + b * w)
            return lambda b: dict(u=1.0 / (1.0 + b * w))
        if name.startswith("P3:"):
            kind = name[3:].lower()
            return lambda c: dict(pen=kind, c=c)
        if name == "P3native:SIG":
            return lambda c: dict(pen="sig", c=c, lam=self.lam_native(c))
        if name == "C2":
            return lambda c: dict(pen="c2", c=c)
        raise ValueError(name)

    def lam_native(self, c):
        """P3's native lambda at pi_0: alpha / mean |Q1(s, pi_0) - c W(s, pi_0)| (the lambda side channel)."""
        return self.h.args.alpha / np.mean(np.abs(self.q1_pi0 - c * self.width_pi0))

    def execute(self, reg, items):
        """Evaluate (arm, knob) pairs, each executor group in one call; new rows go into the regime's table."""
        tab = self.tables[reg]
        rows = [tab.row_of.get((arm.name, k)) for arm, k in items]
        todo = {}
        for j, ((arm, k), r) in enumerate(zip(items, rows)):
            if r is None:
                todo.setdefault(arm.group, []).append(j)
        for group, js in todo.items():
            fresh = {}
            for j in js:
                fresh.setdefault((items[j][0].name, items[j][1]), j)
            order = list(fresh.values())
            results = self.executors[group[0]](group, [items[j] for j in order])
            for j, res in zip(order, results):
                arm, k = items[j]
                tab.row_of[(arm.name, k)] = tab.add(res["K"], (arm.name, k, "eval"), (arm.recipe, k),
                                                    fp=res.get("fp"), l2=res.get("l2"), vbar=res.get("vbar"))
            for j in js:
                rows[j] = tab.row_of[(items[j][0].name, items[j][1])]
        return rows

    def ladders(self, reg, arms):
        items = [(arm, k) for arm in arms for k in arm.ladder]
        rows = self.execute(reg, items)
        tab = self.tables[reg]
        for (arm, k), r in zip(items, rows):
            tab.labels[r] = (arm.name, k, "ladder")
            tab.index["ladder"].setdefault(arm.name, []).append(r)

    def targets(self, reg):
        """S_host, the floor and the targets S_k = S(P0(m_k)) of a regime (None: below floor or nonfinite)."""
        tab = self.tables[reg]
        K_none = tab.K[tab.index["NONE"]]
        S_host = float(self.strength(K_none, self.K0))
        floor = max(FLOOR_ABS, FLOOR_REL * S_host)
        p0 = tab.index["ladder"]["P0"]
        S_star = {name: self.S_row(reg, p0[PH.P0_LADDER.index(m)]) for name, m in zip(LEVEL_NAMES, PH.TARGET_M)}
        usable = {name: (S if np.isfinite(S) and S >= floor else None) for name, S in S_star.items()}
        for name, m in zip(LEVEL_NAMES, PH.TARGET_M):
            r = p0[PH.P0_LADDER.index(m)]
            tab.index["matched"].setdefault("P0", {})[name] = dict(
                knob=m, S=S_star[name], reachable=usable[name] is not None, converged=usable[name] is not None,
                monotone=None, evaluations=0, bracket=None, row=r if usable[name] is not None else None,
                no_target=usable[name] is None)
        info = dict(S_host=S_host, floor=floor, S_star=S_star, usable=usable)
        tab.index["targets"] = info
        return info

    def match(self, reg, arms, targets_of, kind):
        """Batched matching of every (arm, level) with a target; results and matched rows go into the index."""
        tab = self.tables[reg]
        cfg = MATCH[kind]
        problems, keys = [], []
        for arm in arms:  # the arm's own ladder (a P0 branch reads its subset of P0's ladder rows)
            S_ladder = [self.S_row(reg, tab.row_of[(arm.name, k)]) for k in arm.ladder]
            for level, target in targets_of(arm).items():
                if target is None:  # below floor or nonfinite: never matched
                    tab.index["matched"].setdefault(arm.name, {})[level] = dict(reachable=False, no_target=True,
                                                                               row=None, knob=None, S=None)
                    continue
                problems.append(dict(ladder=arm.ladder, S_ladder=S_ladder, target=target, **cfg))
                keys.append((arm, level))

        def evaluate(need):
            rows = self.execute(reg, [(keys[i][0], k) for i, k in need])
            return [self.S_row(reg, r) for r in rows]

        results = batched_match(problems, evaluate, max_rounds=cfg["max_bisect"] + 3)
        for (arm, level), m in zip(keys, results):
            m = {k: v for k, v in m.items() if k != "ladder_S"}
            m["row"] = tab.row_of.get((arm.name, m["knob"])) if m["reachable"] else None
            m["no_target"] = False
            tab.index["matched"].setdefault(arm.name, {})[level] = m

    # -- 1. signal and assignments --------------------------------------------------------------------------

    def signal(self):
        h, cell, spec = self.h, self.cell, self.spec
        t0 = time.perf_counter()
        if self.block == "R":
            ref, diag = fit_signal(h, self.shared, self.native_lq, self.train32, self.cal32, self.stream, lq_path=True)
        else:
            ref, diag = fit_signal(h, self.shared, cell.native, self.train32, self.cal32, self.stream, self.contexts)
        cell.sig = ref
        nuis = self.shared.setdefault("nuis", {})
        if spec.case == "clean":
            nuis[spec.offset] = ref
        cell.nuis = nuis.get(spec.offset)
        readout = P.bc_readout(cell.models, P.State(cell.native, None, None, ref), self.train32, h.cfg.blend)
        if not bool(readout.inputs_valid):
            raise FloatingPointError("invalid dose readout")
        dose = np.asarray(readout.dose, np.float64)
        err = np.abs(PH.head_error_any(cell.params, self.obs, self.a)[:, 0])
        self.perm_old = perm_old(h, self.block, self.level, self.rep, self.n)
        self.doses, degenerate = LQ.control_doses(dose, err, self.perm_old)  # float32: bca, constant, shuffled, oracle
        self.assign = PH.assignments(cell, N_PERM_FP, N_PERM_FP)
        self.has_error = self.assign["has_error"]
        pi0 = PH.policy_np(self.K0, self.obs)
        self.pi0 = pi0
        self.q1_pi0 = PH.q1_np(cell.params, self.obs, pi0)
        self.width_pi0 = PH.width_np(cell, ref, self.obs, pi0)
        thr = float(ref.threshold)
        sigma_ev = np.asarray(PH.width_np(cell, ref, self.data["eval"]["obs"], self.data["eval"]["action"])) / thr
        self.signal_record = dict(
            threshold=thr, residual_unit=float(ref.residual_scale), lambda_hat=diag.get("lambda_hat"),
            lambda_hpd=diag.get("lambda_hpd"), n_cal=diag.get("scores"), fit_accepted=diag["fit_accepted"],
            fit_loss_final=diag["fit_loss_final"], dose_mean=float(dose.mean()), dose_sd=float(dose.std()),
            dose_min=float(dose.min()), dose_max=float(dose.max()), dose_digest=LQ.digest(dose),
            sigma_eval_digest=LQ.digest(sigma_ev), oracle_degenerate=bool(degenerate),
            nuis="own" if spec.case == "clean" else ("clean" if cell.nuis is not None else "absent"),
            seconds=time.perf_counter() - t0)
        self.check_multiset()
        if spec.case == "clean" and cell.nuis is not None:
            same = all(np.array_equal(self.assign["weights"][f]["P1"]["NUIS"], self.assign["weights"][f]["P1"]["SIG"])
                       for f in PH.FAMILIES)
            self.check("nuis_is_sig_in_clean", "2.7", same)

    def check_multiset(self):
        """Part 2 item 3 (FP, global ranks): every arm's sorted weights are the multiset of the run's settings
        (PH.multiset(n, rank_s, rank_clip)) bit for bit, per family/placement (so also SIG's)."""
        ok = True
        ref = PH.multiset(self.n, self.h.cfg.rank_s, self.h.cfg.rank_clip)
        for fam in PH.FAMILIES:
            for p in PH.PLACEMENTS:
                arms = self.assign["weights"][fam][p]
                ok &= all(np.array_equal(np.sort(w), ref) for w in arms.values())
        self.check("same_multiset_global", "2.3", ok)

    def diagnostics(self):
        """Outcome-free correlations of SIG's weights (DR6a, items 2.17, 2.26, 3.3)."""
        cell, w = self.cell, self.assign["weights"]
        e_logged = np.abs(PH.head_error_any(cell.params, self.obs, self.a)[:, 0])
        e_pi = np.abs(PH.head_error_any(cell.params, self.obs, self.pi0)[:, 0])
        out = dict(spearman_sig_e1_L=spearman(w["L"]["P1"]["SIG"], e_logged),
                   spearman_sig_e1_pi=spearman(w["pi"]["P1"]["SIG"], e_pi),
                   spearman_sig_leverage_P1=spearman(w["L"]["P1"]["SIG"], self.assign["leverage"]["P1"]),
                   spearman_sig_leverage_P2=spearman(w["L"]["P2"]["SIG"], self.assign["leverage"]["P2"]),
                   corr_sig_poor_mode=(LQ.correlation(w["L"]["P1"]["SIG"], self.tr["gain"] == 0)
                                       if self.level == "mixed" else None),
                   spearman_sig_context0=(spearman(w["L"]["P1"]["SIG"], self.context == 0) if self.contexts
                                          else None))
        if not self.contexts:
            K_b = self.data["meta"]["K_bar"]
            qg, bg = self.h.lq.terms(PH.to_jax32(cell.params), jnp.asarray(self.K0, jnp.float32), self.train32.obs,
                                     self.train32.action, None) if self.lq_ok else (None, None)
            out.update(alignment_K=PH.alignment_K(self.assign["grads"], self.s),
                       alignment_K_step3=step3_alignment(cell.params, self.s, self.K0, K_b) if self.block == "R"
                       else None)
            if qg is not None:  # step 3's q_term_grad_norm and bc_term_grad_norm_* (rho_host = bc_none / q)
                out["q_term_grad_norm"] = float(jnp.linalg.norm(qg))
                for arm, d in (("none", None), *self.doses.items()):
                    bc = self.h.lq.terms(PH.to_jax32(cell.params), jnp.asarray(self.K0, jnp.float32),
                                         self.train32.obs, self.train32.action,
                                         None if d is None else jnp.asarray(d))[1]
                    out[f"bc_term_grad_norm_{arm}"] = float(jnp.linalg.norm(bc))
        self.diag = out

    # -- 2. fixed point -------------------------------------------------------------------------------------

    def fixed_point(self):
        t0 = time.perf_counter()
        cell, models = self.cell, self.cell.models
        reg = "FP"
        tab = self.tables[reg] = Table(self.Kshape)
        self.T = {part: PH.fp_terms(*PH.fp_rows(cell.params, self.obs, models[1], part), self.s, self.a)
                  for part in ("q1", "exact", "error")}
        self.bfgs_solved = {}
        self.ctx_pen = PH.loop_context(cell, const=0.7)
        none, exact = self.solve(self.T["q1"], 1.0, 1.0), self.solve(self.T["exact"], 1.0, 1.0)
        fpinfo = lambda r: dict(min_eig=float(np.min(r["min_eig"])), cond=float(np.max(r["cond"])),
                                finite=bool(r["finite"]))
        tab.index["NONE"] = tab.add(none["K"], ("NONE", None, "base"), (lambda _: dict(), None), fp=fpinfo(none))
        tab.index["EXACT"] = tab.add(exact["K"], ("EXACT", None, "base"), (lambda _: dict(qread="exact"), None),
                                     fp=fpinfo(exact))
        ones = np.ones(self.n)
        self.frontier = self.solve(self.T["q1"], 1.0, PH.FRONTIER_M[:, None] * ones)["K"]
        d = self.doses
        for name, v in (("HOOK", d["bca"]), ("CONST-old", d["constant"]), ("SHUF-old", d["shuffled"]),
                        ("OR-raw", d["oracle"])):
            v = v.astype(np.float64)
            r = self.solve(self.T["q1"], 1.0, v)
            tab.index["refs"][name] = tab.add(r["K"], (name, None, "ref"), (lambda _, v=v: dict(v=v), None),
                                              fp=fpinfo(r))
        arms = self.fp_arms()
        self.ladders(reg, arms)
        info = self.targets(reg)
        self.match(reg, [a for a in arms if a.name not in ("P0", "P3native:SIG")], lambda arm: info["usable"], "FP")
        self.match_p0_refs(reg, [a for a in arms if a.name == "P0"][0], info)
        self.fp_checks()
        self.timing["FP"] = time.perf_counter() - t0

    def match_p0_refs(self, reg, p0, info):
        """P0 matched to S(HOOK) and S(OR-raw) (DR7, Part 1 item 4), with the regime's floor, on the P0 branch on the
        reference's side of m = 1 (p0_branch: the hook's dose and OR-raw share the dose mean, about 1.3, so m >= 1).
        The branch, the reference's mean BC weight and the match (on the branch's own ladder, so its monotone flag is
        the branch's) are recorded under matched['P0']['S(HOOK)'] / ['S(OR-raw)']."""
        tab = self.tables[reg]
        weights = dict(HOOK=self.doses["bca"], **{"OR-raw": self.doses["oracle"]})
        by_branch = {}
        for name, v in weights.items():
            S = self.S_row(reg, tab.index["refs"][name])
            mean = float(np.mean(np.asarray(v, np.float64)))
            ladder, label = p0_branch(mean)
            by_branch.setdefault((ladder, label), {})[f"S({name})"] = (S if np.isfinite(S) and S >= info["floor"]
                                                                       else None, mean)
        for (ladder, label), targets in by_branch.items():
            self.match(reg, [p0._replace(ladder=ladder)], lambda arm, t=targets: {k: v[0] for k, v in t.items()},
                       "FP" if reg == "FP" else "ADAM")
            for level, (_, mean) in targets.items():
                tab.index["matched"]["P0"][level].update(branch=label, ref_mean_weight=mean)

    def fp_arms(self):
        """The fixed point's matchable arms (design sections 3, 6; plan 'reduced' trims blocks R and X-EP)."""
        arms = [Arm("P0", PH.P0_LADDER, ("solve", "q1"), lambda m: (1.0, m), self.recipe_fn("P0"))]
        o = self.doses["oracle"].astype(np.float64)
        if not self.reduced:
            arms.append(Arm("ORraw", OR_T_LADDER, ("solve", "q1"), lambda t: (1.0, np.maximum(1.0 + t * (o - 1.0), 0.0)),
                            self.recipe_fn("ORraw")))
        for (fam, p), key in PLACEMENT_KEYS.items():
            if self.reduced and fam == "pi":
                continue
            for name, w in self.assign["weights"][fam][p].items():
                if self.reduced and name not in ("SIG", "NUIS") and not name.startswith(("SHUF", "STRAT")):
                    continue
                item = (lambda b, w=w: (1.0, 1.0 + b * w)) if p == "P1" else (lambda b, w=w: (1.0 / (1.0 + b * w), 1.0))
                arms.append(Arm(f"{key}:{name}", PH.BETA_LADDER, ("solve", "q1"), item, self.recipe_fn(f"{key}:{name}")))
        if self.reduced:
            return arms
        lin, perms = self.tlin_weights()  # T-lin robustness (design section 2): raw W / mean W against SHUF-lin
        for p, key in (("P1", "P1L"), ("P2", "P2L")):
            for name, w in [("TLIN", lin)] + [(f"SHUFLIN{k}", lin[perm]) for k, perm in enumerate(perms)]:
                item = (lambda b, w=w: (1.0, 1.0 + b * w)) if p == "P1" else (lambda b, w=w: (1.0 / (1.0 + b * w), 1.0))
                arms.append(Arm(f"{key}:{name}", PH.BETA_LADDER, ("solve", "q1"), item, self.recipe_fn(f"{key}:{name}")))
        bfgs_ok = self.block in ("X-CS", "C")
        if self.has_error and self.spec.case in NONNEGATIVE:
            arms.append(Arm("P3:OR", PH.C_LADDER, ("combo", "error"), lambda c: c, self.recipe_fn("P3:OR")))
        elif self.has_error and bfgs_ok:
            arms.append(Arm("P3:OR", PH.C_LADDER, ("bfgs", "or", "locked"), lambda c: c, self.recipe_fn("P3:OR")))
        if bfgs_ok:
            arms.append(Arm("P3:SIG", PH.C_LADDER, ("bfgs", "sig", "locked"), lambda c: c, self.recipe_fn("P3:SIG")))
            if self.cell.nuis is not None:
                arms.append(Arm("P3:NUIS", PH.C_LADDER, ("bfgs", "nuis", "locked"), lambda c: c,
                                self.recipe_fn("P3:NUIS")))
            arms.append(Arm("P3native:SIG", PH.C_LADDER, ("bfgs", "sig", "native"), lambda c: c,
                            self.recipe_fn("P3native:SIG")))
        if self.contexts:
            self.c2_terms()
            arms.append(Arm("C2", PH.C_LADDER, ("combo", "c2"), lambda c: c, self.recipe_fn("C2")))
        return arms

    def tlin_weights(self):
        """T-lin: M_i = W(s_i, a_i) / mean W (SIG's own magnitudes, not rank-normal), and SHUF-lin's permutations
        (the same seeds as SHUF_k, so SHUFLIN_k permutes the magnitudes exactly as SHUF_k permutes the ranks)."""
        if not hasattr(self, "_tlin"):
            x = self.assign["x"]["L"]["SIG"]
            meta = self.data["meta"]
            qi = PH.QUALITIES.index(meta["quality"])
            perms = [PH.seed_rng(self.h, self.rep, qi, PH.TAG_SHUF + k).permutation(self.n) for k in range(N_PERM_FP)]
            self._tlin = (x / x.mean(), perms)
        return self._tlin

    def c2_terms(self):
        """C2: the per-context LSTD fixed point Q_p of the penalty W(s_i, a_i) under pi_0 (design section 3), as
        fixed-point rows (Q1 - c Q_p is quadratic in a, so P3-C2 is closed form) and its per-row action gradient."""
        p = PH.width_np(self.cell, self.cell.sig, self.obs, self.a)
        v_clip = PH.clipped_noise_variance(self.h.args.policy_noise, self.h.args.noise_clip)
        forms = PH.lstd_penalty(self.s, self.a, self.tr["next_state"], p, self.K0, self.h.system.gamma, v_clip,
                                self.context)
        da = self.a.shape[1]
        Pp, qp = np.zeros((self.n, da, da)), np.zeros((self.n, da))
        for c, (M, _, _) in enumerate(forms):
            rows = self.context == c
            Pc, qc = PH.quadratic_rows(M, self.s[rows])
            Pp[rows], qp[rows] = Pc, qc
        self.T["c2"] = PH.fp_terms(Pp, qp, self.s, self.a)
        self.c2_grad_pi0 = 2 * (np.einsum("nij,nj->ni", Pp, self.pi0) + qp)
        self.c2_info = dict(m0=[float(f[1]) for f in forms])

    def fp_solve(self, group, items):
        T, out = self.T[group[1]], []
        for start in range(0, len(items), 256):
            chunk = items[start:start + 256]
            pairs = [arm.item(k) for arm, k in chunk]
            U = np.stack([np.broadcast_to(np.asarray(u, np.float64), (self.n,)) for u, _ in pairs])
            V = np.stack([np.broadcast_to(np.asarray(v, np.float64), (self.n,)) for _, v in pairs])
            res = self.solve(T, U, V)
            mins = np.min(np.reshape(res["min_eig"], (len(chunk), -1)), axis=1)
            conds = np.max(np.reshape(res["cond"], (len(chunk), -1)), axis=1)
            out += [dict(K=res["K"][j], fp=dict(min_eig=float(mins[j]), cond=float(conds[j]),
                                                finite=bool(res["finite"][j]))) for j in range(len(chunk))]
        return out

    def fp_combo(self, group, items):
        """Closed form for a critic Q1 - c X with X quadratic in a (P3-OR where e1 >= 0; C2)."""
        T1, X, out = self.T["q1"], self.T[group[1]], []
        for arm, c in items:
            r = self.solve(dict(T1, Tq=T1["Tq"] - c * X["Tq"], bq=T1["bq"] - c * X["bq"]), 1.0, 1.0)
            out.append(dict(K=r["K"], fp=dict(min_eig=float(np.min(r["min_eig"])), cond=float(np.max(r["cond"])),
                                              finite=bool(r["finite"]))))
        return out

    def fp_bfgs(self, group, items):
        """P3 by BFGS, warm-started along increasing c from the nearest c already solved for the arm (first: NONE)."""
        _, kind, lam_mode = group
        tab = self.tables["FP"]
        K_none = tab.K[tab.index["NONE"]]
        if not np.all(np.isfinite(K_none)):  # unbounded NONE: start the ladder at K_0
            K_none = self.K0
        found = {}
        by_arm = {}
        for arm, c in items:
            by_arm.setdefault(arm.name, []).append(c)
        for name, cs in by_arm.items():
            solved = self.bfgs_solved.setdefault(name, {})
            cs = sorted(set(cs))
            if solved:
                near = min(solved, key=lambda x: abs(math.log(x) - math.log(cs[0])))
                start = solved[near]
            else:
                start = K_none
            if lam_mode == "locked":
                res = PH.fixed_point_penalty(self.cell.models, self.cell.params, self.obs, self.a, kind, self.ctx_pen,
                                             self.lam0, cs, start, self.K0)
            else:
                res = []
                for c in cs:
                    r = PH.fixed_point_penalty(self.cell.models, self.cell.params, self.obs, self.a, kind,
                                               self.ctx_pen, self.lam_native(c), (c,), start, self.K0)[0]
                    start = r["K"] if np.all(np.isfinite(r["K"])) else start
                    res.append(r)
            for r in res:
                solved[r["c"]] = r["K"]
                found[(name, r["c"])] = r
        out = []
        for arm, c in items:
            r = found[(arm.name, c)]
            out.append(dict(K=r["K"], fp=dict(min_eig=math.nan, cond=math.nan, finite=bool(np.all(np.isfinite(r["K"]))),
                                              bfgs_success=r["success"], bfgs_iterations=r["iterations"],
                                              bfgs_grad_norm=r["grad_norm"])))
        return out

    def fp_checks(self):
        """The fixed point's [derived] items (Part 0 item 4, Part 2 items 1, 4, 5, 6; Part 3 item 1)."""
        tab, T1, Te = self.tables["FP"], self.T["q1"], self.T["exact"]
        K_none = tab.K[tab.index["NONE"]]
        K_ex = tab.K[tab.index["EXACT"]]
        rel = lambda x, y: float(np.max(np.abs(np.asarray(x) - np.asarray(y))) / np.max(np.abs(np.asarray(y))))
        w = self.assign["weights"]["L"]["P1"]["SIG"]
        zero = [self.solve(T1, 1.0, 1.0 + 0.0 * w)["K"], self.solve(T1, 1.0 / (1.0 + 0.0 * w), 1.0)["K"]]
        self.check("zero_knob_fp", "2.1", all(np.array_equal(z, K_none, equal_nan=True) for z in zero))
        bounded = bool(np.all(np.isfinite(K_none)))
        betas = np.asarray(PH.BETA_LADDER)
        ones = np.ones(self.n)
        p2 = self.solve(T1, (1 / (1 + betas))[:, None] * ones, np.ones((len(betas), self.n)))
        p0 = self.solve(T1, np.ones((len(betas), self.n)), (1 + betas)[:, None] * ones)
        fin = p0["finite"] & p2["finite"]
        worst = max([rel(p2["K"][i], p0["K"][i]) for i in np.flatnonzero(fin)], default=0.0)
        self.check("scale_equivalence_fp", "2.4", np.array_equal(p0["finite"], p2["finite"]) and worst <= 1e-6, worst,
                   1e-6)
        if bounded:  # an unbounded NONE (minimum eigenvalue <= 0) has no fixed point to compare with
            res = PH.fixed_point_penalty(self.cell.models, self.cell.params, self.obs, self.a, "const", self.ctx_pen,
                                         self.lam0, (1.0, 30.0), K_none, self.K0)
            self.check("constant_penalty_fp", "2.5", all(r["success"] and np.array_equal(r["K"], K_none) for r in res))
        case, kappa, da = self.spec.case, self.spec.kappa, self.a.shape[1]
        if self.has_error and case in NONNEGATIVE:  # P3-OR at c = 1 reads Q1 - e1 = Q^pi
            r = self.solve(dict(T1, Tq=T1["Tq"] - self.T["error"]["Tq"], bq=T1["bq"] - self.T["error"]["bq"]), 1.0,
                           1.0)
            if r["finite"] and np.all(np.isfinite(K_ex)):
                self.check("p3_oracle_fp_is_exact", "2.6", rel(r["K"], K_ex) <= 1e-6, rel(r["K"], K_ex), 1e-6)
            self.check_p3_oracle_gradient()
        if case in ("q1_optimistic", "cone", "q1opt-local") and kappa:
            self.check_cancellation(case, kappa, da, K_ex)
        if self.contexts and case in LOCALIZED and "C_clean" in self.shared:
            clean_T, lam_clean = self.shared["C_clean"]
            k_err = self.solve(T1, 1.0, 1.0, lam_clean)["K"]
            k_clean = self.solve(clean_T, 1.0, 1.0, lam_clean)["K"]
            gap = float(np.max(np.abs(k_err[1:] - k_clean[1:])))
            self.check("context_locality_fp", "3.1", gap <= 1e-10, gap, 1e-10)
        if self.contexts and self.spec.case == "clean":
            self.shared["C_clean"] = (T1, self.lam0)

    def check_cancellation(self, case, kappa, da, K_ex):
        """Part 0 item 4 in its exact form: BC weights 1 + lambda_0 kappa d_a w (w = 1, c(s) or 1[c = 0]) give
        EXACT plus lambda_0 kappa A^-1 vec((K_bc,w - K-bar_b) Sigma_w), within 1e-6 relative in K (PD cells)."""
        s, a, T1, Te = self.s, self.a, self.T["q1"], self.T["exact"]
        K_bar = np.asarray(self.cell.params["Kb"], np.float64)
        if case == "q1_optimistic":
            w = np.ones(self.n)
        elif case == "cone":
            cone = self.cell.params["cone"]
            w = PH.cone_weight(s, cone["v"], cone["c0"], cone["sharpness"])
        else:
            w = (self.context == 0).astype(np.float64)
        arm = self.solve(T1, 1.0, 1.0 + self.lam0 * kappa * da * w)
        if not (np.all(arm["finite"]) and np.all(np.isfinite(K_ex))):
            self.check(f"cancellation_{case}", "0.4", True, None, 1e-6)  # not PD: the identity is not asserted
            return
        if self.contexts:
            rows = self.context == 0
            K_w = PH.ls_gain(s[rows], a[rows])
            S_w = s[rows].T @ s[rows] / self.n
            A = fp_matrix(Te, self.lam0, rows=np.flatnonzero(rows), n=self.n)
            pred = np.zeros(self.Kshape)
            pred[0] = np.linalg.solve(A, self.lam0 * kappa * vec((K_w - K_bar) @ S_w)).reshape(s.shape[1], -1).T
        else:
            K_w = PH.ls_gain(s, a, w)
            S_w = (s * w[:, None]).T @ s / self.n
            A = fp_matrix(Te, self.lam0)
            pred = np.linalg.solve(A, self.lam0 * kappa * vec((K_w - K_bar) @ S_w)).reshape(s.shape[1], -1).T
        miss = float(np.max(np.abs(arm["K"] - (K_ex + pred))) / np.max(np.abs(K_ex)))
        self.check(f"cancellation_{case}", "0.4", miss <= 1e-6, miss, 1e-6)

    def check_p3_oracle_gradient(self):
        """Part 2 item 6, first clause: at K_0, per row grad_a (Q1 - |e1|) = grad_a Q^pi within 1e-4 relative
        (float32 critic, as test_placement_harness)."""
        crit = self.cell.models[1]
        cp = PH.to_jax32(self.cell.params)
        obs = jnp.asarray(self.obs, jnp.float32)
        pi = jnp.asarray(self.pi0, jnp.float32)
        g = jax.grad(lambda x: (crit.apply(cp, obs, x)[..., 0] - jnp.abs(crit.error(cp, obs, x))).sum())(pi)
        ge = jax.grad(lambda x: crit.exact(cp, obs, x).sum())(pi)
        ge = np.asarray(ge, np.float64)
        err = np.linalg.norm(np.asarray(g, np.float64) - ge, axis=1) / np.maximum(np.linalg.norm(ge, axis=1), 1e-30)
        self.check("p3_oracle_action_gradient", "2.6", float(err.max()) <= 1e-4, float(err.max()), 1e-4)

    # -- 3. Adam regimes ------------------------------------------------------------------------------------

    def w_table(self):
        """The L-family weight table (float32 rows, fixed height): ONES, the hook's dose and step 3's controls, P1's
        assignments and P2's own (STRAT, OR2); returns the table and arm -> row maps per placement."""
        n = self.n
        rows = [np.ones(n), self.doses["bca"], self.doses["constant"], self.doses["shuffled"], self.doses["oracle"]]
        names = dict(ONES=0, DOSE=1, CONST=2, SHUF_OLD=3, ORACLE=4)
        maps = {"P1": {}, "P2": {}}
        for name, w in self.assign["weights"]["L"]["P1"].items():
            if perm_index(name) >= N_PERM_ADAM:
                continue
            maps["P1"][name] = len(rows)
            rows.append(w)
        for name, w in self.assign["weights"]["L"]["P2"].items():
            if perm_index(name) >= N_PERM_ADAM:
                continue
            if name.startswith("STRAT") or name == "OR2":
                maps["P2"][name] = len(rows)
                rows.append(w)
            else:
                maps["P2"][name] = maps["P1"][name]
        if len(rows) > L_TABLE_ROWS:
            raise ValueError("weight table taller than L_TABLE_ROWS")
        rows += [np.ones(n)] * (L_TABLE_ROWS - len(rows))
        return np.stack(rows).astype(np.float32), names, maps

    def adam_setup(self, reg):
        """Batch rows, keys, compiled programs and contexts (weight / permutation tables) of an Adam regime."""
        h, cell, cfg = self.h, self.cell, REGIMES[reg]
        if reg == "A5000":
            st = block_stream(h, self.block, self.level, self.rep, self.n, cfg["steps"])
            rows, keys = st.actor_rows, st.actor_keys
        elif cfg["full_batch"]:
            rows = jnp.asarray(np.tile(np.arange(self.n), (h.cfg.actor_steps, 1)), jnp.int32)
            keys = self.stream.actor_keys
        else:
            rows, keys = self.stream.actor_rows, self.stream.actor_keys
        W, names, maps = self.w_table()
        ctx_L = PH.loop_context(cell, w_table=W, const=0.7)
        dummy = jnp.zeros((rows.shape[0], 1), jnp.int32)
        diag = cfg["diagnostics"]
        spec = lambda **kw: PH.Spec(diagnostics=diag, rank_s=h.cfg.rank_s, rank_clip=h.cfg.rank_clip, **kw)
        progs = {"L": (spec(), ctx_L, dummy), "trust": (spec(trust=True), ctx_L, dummy),
                 "none": (spec(bc="none"), ctx_L, dummy), "exact": (spec(exact=True), ctx_L, dummy)}
        for kind in ("sig", "nuis", "or"):
            progs["pen:" + kind] = (spec(pen=kind), ctx_L, dummy)
        pi_maps = {}
        if not cfg["full_batch"]:
            for p in PH.PLACEMENTS:
                kinds = {"sig": ["SIG", "ANTI"] + [f"SHUF{k}" for k in range(N_PERM_ADAM)]
                         + [f"STRAT{k}" for k in range(N_PERM_ADAM)],
                         "nuis": ["NUIS"], "or1": ["OR1"], "or2_" + p.lower(): ["OR2"]}
                table, tie = PH.pi_tables(cell, self.assign, p, rows, arms=[a for v in kinds.values() for a in v])
                ties = jnp.asarray(tie, jnp.int32)
                for kind, arms in kinds.items():
                    arms = [a for a in arms if a in table]
                    if not arms:
                        continue
                    perm = np.stack([table[a][2] for a in arms])
                    reverse = np.array([table[a][1] for a in arms])
                    ctx = PH.loop_context(cell, perm_table=perm, reverse=reverse)
                    progs[f"pi:{p}:{kind}"] = (spec(family="pi", kind=table[arms[0]][0], trust=p == "P2"), ctx, ties)
                    for j, a in enumerate(arms):
                        pi_maps[(p, a)] = (f"pi:{p}:{kind}", j)
        self.adam_env = getattr(self, "adam_env", {})
        self.adam_env[reg] = dict(rows=rows, keys=keys, progs=progs, names=names, maps=maps, pi_maps=pi_maps)

    def adam_exec(self, group, items):
        _, reg, prog = group
        env = self.adam_env[reg]
        spec, ctx, ties = env["progs"][prog]
        program = self.h.program(spec, self.contexts)
        K, vbar, metrics = run_program(program, self.cell.native, self.train32, env["rows"], env["keys"], ties, ctx,
                                       [arm.item(k) for arm, k in items])
        return [dict(K=K[j], l2=l2_summary(metrics, j), vbar=vbar[j]) for j in range(len(items))]

    def adam_arms(self, reg):
        """The Adam regime's matchable arms (design sections 3, 4, 6; plan 'reduced' trims blocks R and X-EP)."""
        env = self.adam_env[reg]
        names, maps, pi_maps = env["names"], env["maps"], env["pi_maps"]
        g = lambda prog: ("prog", reg, prog)
        arms = [Arm("P0", PH.P0_LADDER, g("L"), lambda m: dict(a=m, assign=names["ONES"]), self.recipe_fn("P0"))]
        subset = reg != "A500"
        perm_names = [f"SHUF{k}" for k in range(N_PERM_ADAM)] + [f"STRAT{k}" for k in range(N_PERM_ADAM)]
        if reg == "A500":
            keep = None if not self.reduced else ["SIG", "NUIS"] + perm_names
            placements = [("L", "P1"), ("L", "P2")] + ([] if self.reduced else [("pi", "P1"), ("pi", "P2")])
        elif reg == "A5000":
            keep, placements = ["SIG"] + perm_names, [("L", "P1"), ("L", "P2"), ("pi", "P2")]
        else:
            keep, placements = ["SIG"] + [f"SHUF{k}" for k in range(N_PERM_ADAM)], [("L", "P1"), ("L", "P2")]
        if reg == "A500" and not self.reduced:
            arms.append(Arm("ORraw", OR_T_LADDER, g("L"), lambda t: dict(a=1.0 - t, b=t, assign=names["ORACLE"]),
                            self.recipe_fn("ORraw")))
        for fam, p in placements:
            key = PLACEMENT_KEYS[(fam, p)]
            for name in self.assign["weights"][fam][p]:
                if keep is not None and name not in keep:
                    continue
                if fam == "L":
                    if name not in maps[p]:
                        continue
                    row = maps[p][name]
                    item = ((lambda b, r=row: dict(a=1.0, b=b, assign=r)) if p == "P1"
                            else (lambda b, r=row: dict(beta=b, assign=r)))
                    group = g("L" if p == "P1" else "trust")
                else:
                    if (p, name) not in pi_maps:
                        continue
                    prog, j = pi_maps[(p, name)]
                    item = ((lambda b, j=j: dict(a=1.0, b=b, assign=j)) if p == "P1"
                            else (lambda b, j=j: dict(beta=b, assign=j)))
                    group = g(prog)
                arms.append(Arm(f"{key}:{name}", PH.BETA_LADDER, group, item, self.recipe_fn(f"{key}:{name}")))
        if not self.reduced and reg in ("A500", "A5000"):
            kinds = [("SIG", "sig"), ("NUIS", "nuis")] + ([("OR", "or")] if self.has_error and not subset else [])
            for name, kind in kinds:
                if name == "NUIS" and self.cell.nuis is None:
                    continue
                arms.append(Arm(f"P3:{name}", PH.C_LADDER, g("pen:" + kind), lambda c: dict(c=c),
                                self.recipe_fn(f"P3:{name}")))
        return arms

    def adam(self, reg):
        t0 = time.perf_counter()
        tab = self.tables[reg] = Table(self.Kshape)
        self.adam_setup(reg)
        env = self.adam_env[reg]
        names = env["names"]
        base = [Arm("NONE", (None,), ("prog", reg, "L"), lambda _: dict(assign=names["ONES"]), lambda _: dict())]
        if reg == "A500":
            base += [Arm("NONE_path", (None,), ("prog", reg, "none"), lambda _: dict(), lambda _: dict()),
                     Arm("EXACT", (None,), ("prog", reg, "exact"), lambda _: dict(), lambda _: dict(qread="exact"))]
            d = self.doses
            for name, row, v in (("HOOK", "DOSE", d["bca"]), ("CONST-old", "CONST", d["constant"]),
                                 ("SHUF-old", "SHUF_OLD", d["shuffled"]), ("OR-raw", "ORACLE", d["oracle"])):
                v = v.astype(np.float64)
                base.append(Arm(name, (None,), ("prog", reg, "L"), lambda _, r=names[row]: dict(a=0.0, b=1.0, assign=r),
                                lambda _, v=v: dict(v=v)))
            base.append(Arm("const_trust_0.5", (None,), ("prog", reg, "trust"),
                            lambda _: dict(beta=1.0, assign=names["ONES"]), lambda _: dict(u=0.5)))
        rows = self.execute(reg, [(arm, None) for arm in base])
        for arm, r in zip(base, rows):
            label = arm.name
            tab.labels[r] = (label, None, "base")
            if label in ("NONE", "NONE_path", "EXACT"):
                tab.index[label] = r
            elif label.startswith("const_trust"):
                tab.index["checks"][label] = r
            else:
                tab.index["refs"][label] = r
        if reg == "A500":
            self.vbar_none = tab.vbar[tab.index["NONE"]]
        arms = self.adam_arms(reg)
        self.ladders(reg, arms)
        info = self.targets(reg)
        if reg == "FB":
            targets = {k: (v if k == "S2" else None) for k, v in info["usable"].items()}
            self.match(reg, [a for a in arms if a.name != "P0"], lambda arm: targets, "ADAM")
        else:
            self.match(reg, [a for a in arms if a.name != "P0"], lambda arm: info["usable"], "ADAM")
        if reg == "A500":
            self.match_p0_refs(reg, arms[0], info)
            self.adam_checks(reg)
        self.timing[reg] = time.perf_counter() - t0

    def single(self, spec, run, ctx, ties):
        """One jitted (not vmapped) run of PH.placement_loop: the bitwise checks compare compiled single runs, as
        test_placement_harness does."""
        key = (spec, self.contexts)
        cache = self.shared.setdefault("single", {})
        if key not in cache:
            cache[key] = jax.jit(partial(PH.placement_loop, self.h.args, self.cell.models, spec))
        env = self.adam_env["A500"]
        run = {k: jnp.asarray(run.get(k, d), jnp.int32 if k == "assign" else jnp.float32)
               for k, d in (("a", 1.0), ("b", 0.0), ("beta", 0.0), ("c", 0.0), ("assign", 0))}
        out = cache[key](self.cell.native, self.train32, env["rows"], env["keys"], ties, ctx, run)
        return np.asarray(out[0].actor.params["K"])

    def adam_checks(self, reg):
        """A500's [derived] items: zero knob (Part 2 item 1), hooks off and the hook's dose against step 3's path
        (item 2), scale equivalence (item 4, eps), constant penalty (item 5) and the pi family's batch multiset (item
        3, bitwise at every step of every pi arm)."""
        env, tab = self.adam_env[reg], self.tables[reg]
        names, maps = env["names"], env["maps"]
        _, ctx_L, dummy = env["progs"]["L"]
        off = lambda **kw: PH.Spec(diagnostics=False, rank_s=self.h.cfg.rank_s, rank_clip=self.h.cfg.rank_clip, **kw)
        none = self.single(off(), dict(assign=names["ONES"]), ctx_L, dummy)
        zero = dict(P1L=self.single(off(), dict(b=0.0, assign=maps["P1"]["SIG"]), ctx_L, dummy),
                    P2L=self.single(off(trust=True), dict(beta=0.0, assign=maps["P2"]["SIG"]), ctx_L, dummy),
                    ORraw=self.single(off(), dict(a=1.0, b=0.0, assign=names["ORACLE"]), ctx_L, dummy))
        for p in PH.PLACEMENTS:
            if (p, "SIG") in env["pi_maps"]:
                prog, j = env["pi_maps"][(p, "SIG")]
                sp, ctx, ties = env["progs"][prog]
                zero[p + "pi"] = self.single(replace(sp, diagnostics=False),
                                             dict(b=0.0, assign=j) if p == "P1" else dict(beta=0.0, assign=j), ctx, ties)
        differ = [k for k, z in zero.items() if not np.array_equal(z, none)]
        self.check("zero_knob_a500", "2.1", not differ, differ or None)
        # P3 at c = 0: the zero-weighted penalty path is compiled into the gradient, and XLA may round the Q-term's
        # gradient differently (seen: 2e-9 to 3e-8 absolute after 200 steps in shared_bias and the cone), so P3's
        # zero knob is [derived up to float32], not bitwise.
        p3 = self.single(off(pen="sig"), dict(c=0.0), ctx_L, dummy)
        gap = float(np.max(np.abs(p3 - none)) / np.max(np.abs(none)))
        self.check("zero_knob_a500_p3", "2.1", gap <= 1e-6, gap, 1e-6, kind="eps")
        const = [self.single(off(pen="const"), dict(c=c), ctx_L, dummy) for c in (1.0, 30.0)]
        self.check("constant_penalty_a500", "2.5", all(np.array_equal(z, none) for z in const))
        trust = self.single(off(trust=True), dict(beta=1.0, assign=names["ONES"]), ctx_L, dummy)
        p0 = self.single(off(), dict(a=2.0, assign=names["ONES"]), ctx_L, dummy)
        gap = float(np.max(np.abs(trust - p0)) / np.max(np.abs(p0)))
        self.check("scale_equivalence_a500", "2.4", gap <= 1e-4, gap, 1e-4, kind="eps")
        self.none_single_gap = float(np.max(np.abs(none - tab.K[tab.index["NONE"]])))
        if self.lq_ok:
            st = self.stream
            lq_none = np.asarray(self.h.lq.actor[False](self.native_lq, self.train32, st.actor_rows, st.actor_keys,
                                                        jnp.ones(self.n, jnp.float32))[0].actor.params["K"])
            path = self.single(off(bc="none"), dict(), ctx_L, dummy)
            self.check("hooks_off_is_td3_bc_update", "2.2", np.array_equal(path, lq_none))
            anchors = {"none": lq_none}
            for arm, dose in self.doses.items():
                anchors[arm] = np.asarray(self.h.lq.actor[True](self.native_lq, self.train32, st.actor_rows,
                                                                st.actor_keys, jnp.asarray(dose))[0].actor.params["K"])
            hook = self.single(off(), dict(a=0.0, b=1.0, assign=names["DOSE"]), ctx_L, dummy)
            self.check("hook_dose_is_step3_bca_path", "2.2", np.array_equal(hook, anchors["bca"]))
            if self.block == "R":  # step 3's anchors through lq_harness.actor_loop itself: J continuity (Part 1.1)
                for arm, K in anchors.items():
                    tab.index["anchors"][arm] = tab.add(K, ("anchor:" + arm, None, "anchor"))
        if env["pi_maps"]:
            self.check_pi_multiset(reg)

    def pi_multiset_gaps(self, reg, knob=1.0):
        """Per pi arm of an Adam regime: whether every step's batch weights, sorted, equal the float32 multiset
        PH.multiset(B, rank_s, rank_clip) of the run's settings bit for bit, and the largest absolute difference.

        One compiled, vmapped pi_weight_trace per program (all its arms at once) at knob 1 (b = 1 for P1, beta = 1 for
        P2; on the ladder). The knob enters only after w_t is formed, so the sorted weights of every knob's run are
        these. Returns {'P1:SIG': (equal, max_abs_gap), ...}."""
        env, h = self.adam_env[reg], self.h
        M = np.asarray(PH.multiset(h.cfg.batch_size, h.cfg.rank_s, h.cfg.rank_clip), np.float32)
        by_prog = {}
        for (p, arm), (prog, j) in sorted(env["pi_maps"].items()):
            by_prog.setdefault(prog, []).append((p, arm, j))
        cache = self.shared.setdefault("pi_trace", {})
        out = {}
        for prog, items in by_prog.items():
            spec, ctx, ties = env["progs"][prog]
            spec = replace(spec, diagnostics=False)
            key = (spec, self.contexts)
            if key not in cache:
                cache[key] = jax.jit(jax.vmap(partial(pi_weight_trace, h.args, self.cell.models, spec),
                                              in_axes=(None, None, None, None, None, None, 0)))
            n = len(items)
            trust = items[0][0] == "P2"
            run = dict(a=jnp.ones(n, jnp.float32), b=jnp.full(n, 0.0 if trust else knob, jnp.float32),
                       beta=jnp.full(n, knob if trust else 0.0, jnp.float32), c=jnp.zeros(n, jnp.float32),
                       assign=jnp.asarray([j for _, _, j in items], jnp.int32))
            _, w = cache[key](self.cell.native, self.train32, env["rows"], env["keys"], ties, ctx, run)
            w = np.sort(np.asarray(w), axis=-1)  # (arms, steps, B)
            for i, (p, arm, _) in enumerate(items):
                same = w[i].shape[-1] == len(M) and np.array_equal(w[i], np.broadcast_to(M, w[i].shape))
                gap = float(np.max(np.abs(w[i] - M))) if w[i].shape[-1] == len(M) else math.inf
                out[f"{p}:{arm}"] = (bool(same), gap)
        return out

    def check_pi_multiset(self, reg):
        """Part 2 item 3, family pi at A500 ([derived], bitwise): every step's batch weights of every pi arm are the
        settings' multiset (rank_s and rank_clip of the run, so the s = 0.47 robustness run is checked against its own
        multiset). value: the arms that differ, with their largest absolute difference."""
        gaps = self.pi_multiset_gaps(reg)
        bad = {arm: g for arm, (same, g) in gaps.items() if not same}
        self.check("pi_batch_multiset", "2.3", not bad, bad or None, 0.0)

    # -- 4. diagnostics, decomposition rows and the record -------------------------------------------------------

    def pen_rows(self, kind):
        """Per-row grad_a pen(s_i, pi_0(s_i)) s_i^T (n, da, ds)."""
        cache = self.__dict__.setdefault("_pen_rows", {})
        if kind not in cache:
            if kind in ("sig", "nuis"):
                ref = self.cell.sig if kind == "sig" or self.cell.nuis is None else self.cell.nuis
                obs = jnp.asarray(self.obs, jnp.float32)
                g = jax.grad(lambda x: PH.width(self.cell.models, ref, obs, x).sum())(jnp.asarray(self.pi0, jnp.float32))
                g = np.asarray(g, np.float64)
            elif kind == "or":
                e = PH.head_error_any(self.cell.params, self.obs, self.pi0)[:, 0]
                g = np.sign(e)[:, None] * self.assign["grads"]["error"]
            else:
                g = self.c2_grad_pi0
            cache[kind] = np.einsum("ni,nj->nij", g, self.s)
        return cache[kind]

    def level1_terms(self, recipe):
        """The placement loss's term K-gradients at K_0 on all training rows (float64; PH.terms' conventions):
        g_Q = lambda mean u_i grad_a Q-read_i s_i^T, g_pen = -lambda c mean grad_a pen_i s_i^T, g_BC = -(2/d_a) mean
        v_i (pi_0(s_i) - a_i) s_i^T; per-context gains take their context's rows (normalised by n)."""
        G = self.__dict__.setdefault("_G", {})
        if not G:
            grads = self.assign["grads"]
            G["q1"] = np.einsum("ni,nj->nij", grads["q1"], self.s)
            G["exact"] = np.einsum("ni,nj->nij", grads["exact"], self.s)
            G["bc"] = np.einsum("ni,nj->nij", self.pi0 - self.a, self.s)
        lam = recipe.get("lam", self.lam0)
        da = self.a.shape[1]

        def agg(w, rows):
            w = np.broadcast_to(np.asarray(w, np.float64), (self.n,))
            if self.contexts:
                return np.einsum("n,nc,nij->cij", w, self.obs[:, -self.C:], rows) / self.n
            return np.einsum("n,nij->ij", w, rows) / self.n

        g_Q = lam * agg(recipe.get("u", 1.0), G["exact" if recipe.get("qread") == "exact" else "q1"])
        g_BC = -(2.0 / da) * agg(recipe.get("v", 1.0), G["bc"])
        g_pen = (-lam * recipe["c"] * agg(1.0, self.pen_rows(recipe["pen"])) if recipe.get("pen")
                 else np.zeros(self.Kshape))
        return dict(g_Q=g_Q, g_pen=g_pen, g_BC=g_BC, g=g_Q + g_pen + g_BC, lam=float(lam))

    def decomposition(self, reg):
        """Rows K_N + a delta-hat_P0 and K_N + delta - a delta-hat_P0 for every matched arm at S_k (P0 at the same S_k
        gives delta_P0; Sigma inner product), from which score_values forms G_reg and G_perp in both orders."""
        tab = self.tables[reg]
        K_N = tab.K[tab.index["NONE"]]
        p0 = tab.index["matched"].get("P0", {})
        for arm, levels in list(tab.index["matched"].items()):
            if arm == "P0":
                continue
            for level, m in levels.items():
                ref = p0.get(level)
                if m.get("row") is None or ref is None or ref.get("row") is None:
                    continue
                d0 = tab.K[ref["row"]] - K_N
                norm = math.sqrt(max(float(inner_sigma(d0, d0, self.sigma_ev)), 0.0))
                d = tab.K[m["row"]] - K_N
                if not (norm > 0 and np.all(np.isfinite(d))):
                    continue
                unit = d0 / norm
                coef = float(inner_sigma(d, unit, self.sigma_ev))
                tab.index["decomp"].setdefault(arm, {})[level] = dict(
                    reg=tab.add(K_N + coef * unit, ("decomp:" + arm, level, "reg")),
                    perp=tab.add(K_N + d - coef * unit, ("decomp:" + arm, level, "perp")))

    def finalize(self):
        """Prune the tables to kept rows; per-row S, displacement, pull share, FP flags and frontier, Level 1 and 2."""
        arrays, regimes = {}, {}
        cid = self.spec.id
        arrays[f"K/{cid}/K0"] = self.K0
        arrays[f"K/{cid}/FP_frontier"] = self.frontier
        if getattr(self, "vbar_none", None) is not None:
            arrays[f"D/{cid}/vbar_none"] = np.asarray(self.vbar_none)
        t_none = self.level1_terms(dict())
        t_exact = self.level1_terms(dict(qread="exact"))
        pull = np.einsum("...ij,...jk->...ik", self.K_bc - self.K0, self.sigma_tr)
        for reg, tab in self.tables.items():
            self.decomposition(reg)
            keep = tab.kept_rows()
            new = {old: i for i, old in enumerate(keep)}
            K = np.stack([tab.K[r] for r in keep])
            K_none = tab.K[tab.index["NONE"]]
            D, delta = self.K_bc - K_none, K - K_none
            S = self.strength(K, K_none)
            with np.errstate(invalid="ignore", divide="ignore"):
                pull_share = inner_sigma(delta, D, self.sigma_ev) / (np.sqrt(inner_sigma(delta, delta, self.sigma_ev))
                                                                     * math.sqrt(float(inner_sigma(D, D, self.sigma_ev))))
            arrays[f"K/{cid}/{reg}"] = K
            arrays[f"D/{cid}/{reg}/S"] = S
            arrays[f"D/{cid}/{reg}/disp_sigma"] = self.strength(K, self.K0)
            arrays[f"D/{cid}/{reg}/disp_frob"] = np.sqrt(np.sum((K - self.K0) ** 2, axis=tuple(range(1, K.ndim))))
            arrays[f"D/{cid}/{reg}/pull_share"] = pull_share
            arrays[f"D/{cid}/{reg}/neg_bc"] = np.array([np.nan if x is None else x for x in
                                                        (negative_weights(tab.recipes[r], self.n) for r in keep)])
            L1 = np.full((len(keep), 4) + self.Kshape, np.nan)
            L1lam = np.full(len(keep), np.nan)
            L1m = np.full((len(keep), len(L1_FIELDS)), np.nan)
            for i, r in enumerate(keep):
                rec = tab.recipes[r]
                if rec is None:
                    continue
                fn, knob = rec
                t = self.level1_terms(fn(knob))
                L1[i] = np.stack([t["g_Q"], t["g_pen"], t["g_BC"], t["g"]])
                L1lam[i] = t["lam"]
                m = PH.level1(t, t_none, t_exact, pull=pull)
                L1m[i] = [np.nan if m.get(k) is None else float(m[k]) for k in L1_FIELDS]
            arrays[f"D/{cid}/{reg}/L1"] = L1
            arrays[f"D/{cid}/{reg}/L1lam"] = L1lam
            arrays[f"D/{cid}/{reg}/L1m"] = L1m
            if reg == "FP":
                fp = np.array([[tab.fp[r].get(k, np.nan) if tab.fp[r] else np.nan for k in FP_FIELDS] for r in keep],
                              np.float64)
                arrays[f"D/{cid}/FP/fp"] = fp
                dist = PH.strength(self.frontier[None], K[:, None], self.sigma_ev)  # (rows, 200)
                with np.errstate(invalid="ignore", divide="ignore"):
                    idx = np.array([np.nanargmin(row) if np.any(np.isfinite(row)) else -1 for row in dist])
                    share = np.array([dist[i, j] / S[i] if j >= 0 and S[i] > 0 else np.nan for i, j in enumerate(idx)])
                arrays[f"D/{cid}/FP/frontier"] = np.stack([idx.astype(np.float64), share], axis=1)
            else:
                arrays[f"D/{cid}/{reg}/L2"] = np.array([tab.l2[r] if tab.l2[r] is not None
                                                        else np.full(len(L2_FIELDS), np.nan) for r in keep])
            index = remap(tab.index, new)
            regimes[reg] = dict(labels=[list(tab.labels[r]) for r in keep], index=index, n_evaluated=len(tab.K))
        record = dict(cell=cid, case=self.spec.case, kappa=self.spec.kappa, a_star=self.spec.a_star,
                      offset=self.spec.offset, block=self.block, level=self.level, replicate=self.rep,
                      K0=self.K0, lam0=self.lam0, has_error=self.has_error, info=self.cell.info,
                      signal=self.signal_record, diagnostics=self.diag, regimes=regimes, checks=self.checks,
                      timing=self.timing, none_single_gap=getattr(self, "none_single_gap", None),
                      lam_native={f"{c:g}": self.lam_native(c) for c in PH.C_LADDER},
                      c2=getattr(self, "c2_info", None), or_raw=self.or_raw_record())
        return record, arrays

    def or_raw_record(self):
        """OR-raw(t)'s BC weights v_i(t) = 1 + t (o_i - 1) turn negative (anti-BC on that row) where o_i < 1 - 1/t
        (open issue for the pre-registration: keep, cap t or clip v). Outcome-free: the oracle's minimum, the largest t
        with every v_i >= 0 (t_nonneg = 1 / (1 - min o); None when min o >= 1) and the count of negative rows at each
        ladder t (D/<cell>/<regime>/neg_bc has the count of every kept row)."""
        o = np.asarray(self.doses["oracle"], np.float64)
        lo = float(o.min())
        return dict(min_oracle=lo, t_nonneg=1.0 / (1.0 - lo) if lo < 1.0 else None, n=self.n,
                    neg_rows={f"{t:g}": int(np.sum(1.0 + t * (o - 1.0) < 0)) for t in OR_T_LADDER})

    def run(self):
        t0 = time.perf_counter()
        self.signal()
        self.diagnostics()
        self.fixed_point()
        self.adam("A500")
        if self.subsets:
            if self.block == "X-CS" and self.spec.case in ("tilt", "cone"):
                self.adam("A5000")
                self.adam("FB")
            elif self.block == "C" and self.spec.case in LOCALIZED:
                self.adam("A5000")
        matched = [m for tab in self.tables.values() for levels in tab.index["matched"].values()
                   for m in levels.values()]
        stuck = sum(1 for m in matched if m.get("reachable") and not m.get("converged"))
        self.check("matches_within_tolerance", "2.8", stuck == 0, stuck, 0, kind="procedural")
        record, arrays = self.finalize()
        record["timing"]["total"] = time.perf_counter() - t0
        return record, arrays


def remap(index, new):
    """The index with row numbers renumbered after pruning (rows: ints, ladder lists, 'row' fields, decomposition)."""
    out = {}
    for key, val in index.items():
        if isinstance(val, int):
            out[key] = new[val]
        elif key == "ladder":
            out[key] = {arm: [new[r] for r in rows] for arm, rows in val.items()}
        elif key == "matched":
            out[key] = {arm: {lv: dict(m, row=new[m["row"]] if m.get("row") is not None else None)
                              for lv, m in levels.items()} for arm, levels in val.items()}
        elif key in ("refs", "anchors", "checks"):
            out[key] = {name: new[r] for name, r in val.items()}
        elif key == "decomp":
            out[key] = {arm: {lv: {part: new[r] for part, r in parts.items()} for lv, parts in levels.items()}
                        for arm, levels in val.items()}
        else:
            out[key] = val
    return out


# ---------------------------------------------------------------------------------------------------------
# One process


def run_process(block, level, rep, output, settings=None, plan="full", cases=None, subsets="auto",
                cone_kappas=CONE_KAPPAS, blockc_kappa=C_KAPPA, log=print):
    """Every cell of one (block, level, replicate); writes arrays.npz, matched_knobs.json and HASHES.sha256."""
    if level not in LEVELS[block]:
        raise ValueError(f"level {level!r} is not one of block {block}'s {LEVELS[block]}")
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat()
    settings = settings or settings_for(block)
    out = Path(output) / block / level / f"rep{rep}"
    out.mkdir(parents=True, exist_ok=False)
    h = PH.PlacementHarness(settings)
    data = block_data(h, block, level, rep)
    n = len(data["train"]["state"])
    stream = block_stream(h, block, level, rep, n)
    do_subsets = subsets == "on" or (subsets == "auto" and rep in SUBSET_REPLICATES)
    shared, records, arrays = {}, {}, {}
    for spec in grid(block, rep, cone_kappas, blockc_kappa, cases):
        t0 = time.perf_counter()
        run = CellRun(h, block, level, rep, spec, data, stream, shared, plan, do_subsets, log)
        record, arr = run.run()
        records[spec.id] = record
        arrays.update(arr)
        log(f"{block} {level} rep{rep} {spec.id}: {time.perf_counter() - t0:.1f}s "
            + " ".join(f"{k} {v:.1f}s" for k, v in record["timing"].items() if k != "total")
            + f" maps {map_count()}")
        clear_if_needed(log)
    for cid, record in records.items():  # q2_optimistic = clean bitwise (block R replicate 0; Part 1 item 1)
        if record["case"] == "q2_optimistic":
            clean = CellSpec("clean", offset=record["offset"]).id
            if clean in records:
                ok = all(np.array_equal(arrays[f"K/{cid}/{reg}"], arrays[f"K/{clean}/{reg}"])
                         for reg in record["regimes"]) and record["signal"]["threshold"] == records[clean]["signal"][
                    "threshold"]
                record["checks"].append(dict(name="q2_optimistic_is_clean", item="1.1", kind="derived", ok=bool(ok),
                                             value=None, tol=None))
    if block == "X-CS":
        digests = {split: LQ.digest(*(data[split][k] for k in ("state", "noise", "shock")))
                   for split in ("train", "cal", "eval")}
    else:
        digests = {split: LQ.digest(data[split]["state"]) for split in ("train", "cal", "eval")}
    meta = dict(block=block, level=level, replicate=rep, pilot=rep == PILOT_REPLICATE,
                analysis=rep in ANALYSIS_REPLICATES, settings=asdict(settings), plan=plan, subsets=do_subsets,
                cone_kappas=list(cone_kappas), blockc_kappa=blockc_kappa, cases=[c.id for c in grid(
                    block, rep, cone_kappas, blockc_kappa, cases)], started_utc=stamp,
                finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - started,
                argv=sys.argv, jax=jax.__version__, numpy=np.__version__, data_digests=digests,
                constants=dict(P0_LADDER=PH.P0_LADDER, BETA_LADDER=PH.BETA_LADDER, C_LADDER=PH.C_LADDER,
                               OR_T_LADDER=OR_T_LADDER, TARGET_M=PH.TARGET_M, LEVEL_NAMES=LEVEL_NAMES,
                               FRONTIER_M=PH.FRONTIER_M, FLOOR=(FLOOR_ABS, FLOOR_REL), MATCH=MATCH,
                               L1_FIELDS=L1_FIELDS, L2_FIELDS=L2_FIELDS, FP_FIELDS=FP_FIELDS,
                               L1_TERMS=("g_Q", "g_pen", "g_BC", "g")))
    np.savez_compressed(out / "arrays.npz", **arrays)
    (out / "matched_knobs.json").write_text(json.dumps(jsonable(dict(meta=meta, cells=records)), indent=1,
                                                       allow_nan=False), encoding="utf8")
    lines = [f"{sha256_file(out / name)}  {name}" for name in ("matched_knobs.json", "arrays.npz")]
    (out / "HASHES.sha256").write_text("\n".join(lines) + "\n", encoding="utf8")  # written last: marks completion
    log(f"{block} {level} rep{rep}: {len(records)} cells in {time.perf_counter() - started:.0f}s -> {out}")
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", required=True, choices=BLOCKS)
    parser.add_argument("--level", required=True)
    parser.add_argument("--replicate", type=int, required=True, help="99 is the pilot; 0-9 the analysis")
    parser.add_argument("--output", required=True, help="root; the process writes OUTPUT/<block>/<level>/rep<r>")
    parser.add_argument("--plan", choices=("full", "reduced"), default="full")
    parser.add_argument("--cases", nargs="+", default=None, help="restrict the block's cases (pilot / tests)")
    parser.add_argument("--subsets", choices=("auto", "on", "off"), default="auto",
                        help="A5000 / full-batch subsets: auto = replicates 0-4")
    parser.add_argument("--cone-kappas", nargs="+", type=float, default=list(CONE_KAPPAS))
    parser.add_argument("--blockc-kappa", type=float, default=C_KAPPA)
    for name, default in asdict(PH.Settings()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=None)
    return parser


def main(argv=None):
    ns = build_parser().parse_args(argv)
    overrides = {name: getattr(ns, name) for name in asdict(PH.Settings())}
    settings = settings_for(ns.block, overrides)
    run_process(ns.block, ns.level, ns.replicate, ns.output, settings, ns.plan, ns.cases, ns.subsets,
                tuple(ns.cone_kappas), ns.blockc_kappa)


if __name__ == "__main__":
    main()
