"""Gapped LQ training harness (option A, part A2): co-trained TD3+BC, FQE and EXACT runs, snapshots, exact critic-error
diagnostics, Tier A actor-only arms with blind strength matching, and the first-order Level-1 screen.

Design: option A of the world audit (scratchpad design_glq.json) with the judge's grafts g4 (OR-D and the advantage
error), g5 (Level-1 screen), g6 (blind matching, null SHUF pairs) and g7 (gain-scheduled actor, off by default). The
world, its exact truth and its data are experiments/signal/gapped_lq.py (A1); the gate stages, the pre-registration and
its hash check are glq_acceptance.py (A3). This module trains and measures; it applies no pass mark and its CLI runs
only the timing pilot, so every acceptance stage that trains goes through glq_acceptance.py's hash check.
algorithms/td3_bc.py, algorithms/td3_bc_bca.py, lq_harness.py and placement_harness.py are imported and called, never
edited.

HOST (co-trained TD3+BC, the setting BCA runs in). td3_bc.DualCritic (two heads, 256-256 ReLU, input obs 15 + action 4),
trained by algorithms/td3_bc.td3_bc_update unchanged: clipped double-Q target, target smoothing noise 0.2 clipped at
0.5, gamma 0.98, tau 0.005, Adam (runtime.networks.torch_adam) 3e-4 for actor and critic, batch 256, actor and target
updates every policy_freq = 2 updates, alpha 2.5, max_action = inf (lq_harness.MAX_ACTION: nothing is clipped, so every
Q^pi is exact for every action). Each update is td3_bc.make_train_step's step: (rng, rng_batch, rng_noise) =
split(rng, 3), batch rows randint(rng_batch, (256,), 0, n), it counted from 1 (tested bitwise equal to a scan of
make_train_step). Per-row bc_multiplier hook: a table over training rows (fixed, or refreshed from the current state
every `every` updates by a Multiplier) is indexed by the batch rows and passed as td3_bc_update's bc_multiplier; without
a table the host takes td3_bc's unweighted path. Initialisation follows td3_bc.initialize's key scheme: rng =
fold_in(PRNGKey(args.seed), 1414676809), (rng, k_actor, k_critic) = split(rng, 3); the critic is
DualCritic.init(k_critic) and args.seed = default_rng([settings.seed, rep, TAG_INIT]).integers(2^31 - 1), the same for
every mode of a rep, so HOST, FQE and EXACT share their critic initialisation and batch stream (common random
numbers).
Actor classes (pi(obs) for obs = [s (10), code (5)]):
- context_linear (primary): placement_harness.ContextLinearActor, pi = -sum_c code_c K_c s, K of shape (5, 4, 10)
  (160 numbers); with the exact one-hot code it is the per-context linear policy -K_c s.
- global_linear (negative-control class): pi = -K s, one K of shape (4, 10) for every context (40 numbers).
- gain_scheduled (g7, off by default): pi = -K_theta(code) s, K_theta a 5-64-64-40 ReLU MLP (torch_dense layers), so
  the policy is linear in s within an episode and its truth stays exact per context (per episode with a noisy code).
actor_init: 'zero' (linear classes start at K = 0, the analogue of td3_bc's near-zero initial outputs; gain_scheduled
uses its random Flax initialisation) or 'bc' (the least-squares behaviour-clone gains of the training split;
gain_scheduled is first fitted to them by Adam on the 5 one-hot codes).

FQE. The same update with the actor frozen at a fixed probe policy: after each td3_bc_update call the actor and the
target actor are restored, so the critic and its target follow td3_bc exactly while the actor that the targets read is
the probe. Probe 'kstar' (default): K*_c in every context (lq_harness.lqr_gain with all 4 actuators), which uses
actuators 3-4 everywhere and is discounted-stable by construction; 'mid': (K_beh,c + K*_c) / 2. FQE always uses the
context_linear class (the probe is linear).

EXACT. The same critic step, but the actor climbs the exact Q^pi of the current actor: every exact_refresh = 100 updates
the per-context (H_c, h_c) = lq_harness.exact_q(system_c, K_c) of the current actor's gains are recomputed in float64
(NumPy, discrete Lyapunov) and handed to the compiled program in float32; the actor loss reads Q^pi(s, a) = sum_c
code_c (z^T H_c z + h_c), z = (s, a), instead of Q1, with the same BC term and alpha. With exact_level 'td3'
(default) h_c includes TD3's fixed-point level c_c (gapped_lq.World.td3_level, about -13.3): the tables hold the value
a perfect TD3 critic reads, Q^pi + c_c, so that lambda below is on the host critic's scale (without it EXACT's lambda
is 9-10% larger than a perfect host's, an alpha confound in R3); the action gradients are the same either way
('deterministic' omits c_c). lambda = alpha / mean |Q(s, pi(s))| uses these tables by default (exact_lambda 'exact':
TD3+BC with Q1 replaced by the exact value everywhere) or the co-trained Q1 ('q1', placement_harness's lambda
lock). A context whose current gain is not discounted-stable keeps its previous tables (counted as exact_unstable).
The update (harness_update) is the critic step of td3_bc_update itself (called with it = 1, which takes td3_bc's
skip branch: critic step only, so policy_freq >= 2 is required), then, when it % policy_freq == 0, the actor step
(actor_step: td3_bc's _actor_loss_fn expression by expression, with the hooks) and
td3_bc's soft target updates. With every hook off, harness_update equals td3_bc_update bit for bit (tested).

BCA (switched off). The co-trained BCA arm (td3_bc_bca.update with fit_scale every update, refresh of the WBCP
threshold on the calibration bank every 5,000 updates, blend 0.5, alpha 0.1, beta 0.95, 1,000 draws) is implemented in
_bca_cotrain but refuses to run unless allow_bca=True is passed. Nothing in this module or its tests runs it (a test
traces its program without executing it). Running it needs a separate approval.

Snapshots. At 25%, 50% and 100% of the updates the whole AgentTrainState (actor, target actor, critic, target critic,
with their Adam states) is written as Flax msgpack (flax.serialization.to_bytes) to <out_dir>/snapshot_<pct>.msgpack;
its SHA-256 goes into run.json (strict JSON, allow_nan=False, with a .sha256 sidecar). load_snapshot restores it.

Exact critic-error diagnostics (error_rows, summarize_errors; evaluation split, exact one-hot codes). At the actor's
gains K (FQE: the probe, in the float32 the critic's targets read), pi(s) = -K_c s, evaluated by the critic at
float32 inputs and by the truth at the same float32-rounded inputs in float64:
- e1(s, a) = Q1(s, a) - Q^pi_c(s, a) at pi(s) and at the logged a; E = |grad_a e1(s, pi(s))| (the error the actor
  climbs) and the same at the logged action; |e1| / |Q^pi|; the level error against TD3's fixed point, e1_td3 = Q1 -
  (Q^pi_c + c_c) (World.td3_level with the run's smoothing noise), and |e1_td3| / |Q^pi|, which R2's accuracy mark
  reads (a perfect TD3 critic has e1 = c_c, 0.15-0.21 of |Q^pi| in R0 and R1, but e1_td3 = 0);
- the advantage error e_A = e1(s, a) - e1(s, pi(s)) (g4) and OR-D = [(Q1(s, pi) - Q1(s, a)) - (Q^pi(s, pi) - Q^pi(s,
  a))]_+ = [-e_A]_+, what BC caution corrects (per-state level errors cancel in both);
- per context: means of E, |e1|, |e_A|, OR-D, medians of |e1| / |Q^pi|, each context's share of sum E^2; pooled: the
  thin contexts' share of sum E^2 and of rows, mean E in T_wrong over the rich contexts (R0, R1, and with DECOY) and
  over T_right. Rows in a context whose closed loop is not discounted-stable have no Q^pi; they are NaN and counted.
grad_a Q1 comes from jax.vjp of DualCritic with cotangent (1, 0), grad_a Q^pi = 2 (H_c z)_a.

Tier A (tier_a; R4's gate, actor-only, frozen critic). The critic is a frozen snapshot (default the host's 50%
snapshot). Every arm starts from the behaviour-clone gains K_bc (least squares per context on the training split) with
a fresh Adam state and takes `steps` (10,000) actor steps (actor_step, td3_bc's actor loss; the critic, its target and
the target actor are never updated, which equals td3_bc_update followed by restoring them, tested) on one shared
stream of batch rows (default_rng([settings.seed, rep, TAG_TIER_A_ROWS])). Arms are vmapped (one compiled program per
arm count); every arm uses the weighted BC path, bc_w = its row multiplier (NONE = 1 on every row; the unweighted
td3_bc path is run once as none_path for comparison). Row multipliers w_i(knob) by family, x the assignment variable:
- EVEN(m) = m on every row (an alpha retune), m in EVEN_LADDER = (1.3, 2, 4, 8, 16);
- OR-wrong = 1 + b x, x = 1[c = T_wrong]; OR-thin: x = 1[c thin]; ANTI: x = 1[c not thin] (R0, R1, DECOY);
- SHUF_k (k < n_shuf = 4): OR-thin's x permuted across training rows by default_rng([seed, rep, TAG_SHUF, k]);
- OR-err: x = placement_harness.rank_normal (s 0.83, clip 2.5, ties by default_rng([seed, rep, TAG_TIE])) of the exact
  E_i = |grad_a e1(s_i, pi(s_i))| at the arm's current actor on every training row, refreshed every 1,000 steps;
- OR-D (g4): x = OR-D_i / mean OR-D at the current actor, refreshed likewise (all-zero OR-D gives x = 0);
- b in B_LADDER = (0.3, 1, 3, 10, 30);
- capped oracles (multiplier in [1, 1.5), BCA's hook range): CAP-wrong = 1 + b x, x = 1[T_wrong], b in (0.1, 0.2, 0.3,
  0.4, 0.49); CAP-err = 1 + 0.5 c x / (1 + c x), x = E_i / mean E (refreshed), c in (0.1, 0.3, 1, 3, 10), the hook's
  saturating form with the exact error in place of the width.
A refreshed variable whose arm is not discounted-stable in some context keeps that context's previous values (counted).
Strength S = glq_match.strength(K_arm, K_NONE, Sigma_ev): the RMS action displacement from NONE on the evaluation
states (Sigma_ev = per-context second moments). Targets S1 = S(EVEN 1.3) and S2 = S(EVEN 2) (S2 primary). Matching is
glq_match.match_parallel (first crossing on [0] + ladder, then interpolation and bisection rounds, tolerance 1%, at
most 5 rounds, each round one vmapped program over every open (family, target) pair, padded with NONE arms to the next
power of two so that few programs compile). Blindness (g6): until the chosen knobs, every arm's gains and
strengths, glq_match.py's SHA-256 and the record's own SHA-256 are written (tierA_blind.json), no return is computed;
glq_match imports no evaluation code. Then tier_a_score adds the exact 100-step J and NS per context and mixed for
every arm (gapped_lq.World.score), the paired differences against EVEN at each target, best-over-ladder values (and
the capped oracles' best minus EVEN's best), and the null SHUF-vs-SHUF pairs at each target.

Level-1 screen (level1_screen, g5; no training). At the host actor K_host and its critic (the snapshot Tier A uses), on
every training row: placement_harness.terms gives the K-gradients of td3_bc's actor-loss terms (Q term, BC term) for
row weights 1 and for each family's x (computed at K_host); the BC gradient is linear in the weights, so for P0(m)
rho = |g_BC| / |g_Q| = m rho_host and a family 1 + b x is matched to the same rho by the positive root of |u + b v| = m
|u| (u = g_BC[1], v = g_BC[x]). Per arm: the increment Delta g over NONE, cos(-Delta g, grad J), the first-order gain
-<grad J, Delta g> (per unit learning rate, unpreconditioned) and its difference from EVEN at the same rho; grad J is
the exact gradient of the mixture's 100-step J (central differences of gapped_lq.finite_return per context).

Timing pilot (pilot; outcome-blind: only times are recorded, no error, return or NS). It times the co-trained HOST,
FQE and EXACT updates, snapshot I/O, the exact diagnostics, the actor-only step at several arm counts, an oracle
refresh and the Level-1 terms, and extrapolates the gate's cost. Part 'full' runs one 100k-update HOST run and the
full Tier A ladder program for 2,000 steps (times only; its snapshots are deleted).

API for the gate stages (glq_acceptance.py). TrainSettings (alpha for the R3 ladder, actor class, probe, ...);
cotrain(world, data, settings, mode, rep, out_dir=..., multiplier=...) -> record (snapshots on disk and in
record['states'], per-snapshot summarize_errors and, for HOST / EXACT, World.score of the actor); load_snapshot;
error_rows / summarize_errors / save_error_rows (R2, R5 rows); exact_rows, critic_rows, bc_gains; Multiplier (Tier B's
co-trained arms: a table function of the state, refreshed every `every` updates); tier_a(world, data, critic_params,
settings, rep, out_dir) -> (blind record, scores), or tier_a_train then tier_a_score; TierA (arm tables and runs);
level1_screen; grad_j. None of these checks a pre-registration: the stage that calls them does.

python experiments/signal/glq_train.py pilot --output runs/glq/pilot [--seconds 900] [--part 1|full]
Tests: JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_glq_train
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax import serialization  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

import algorithms.td3_bc as BASE  # noqa: E402
import algorithms.td3_bc_bca as P  # noqa: E402
from calibration.network import Calibrator  # noqa: E402
from calibration.reference import WBCPConfig, initial_reference  # noqa: E402
from experiments.signal import gapped_lq as G  # noqa: E402
from experiments.signal import glq_match as M  # noqa: E402
from experiments.signal import lq_harness as LQ  # noqa: E402
from experiments.signal import placement_harness as PH  # noqa: E402

DS, DA, NC, OBS_DIM = G.DS, G.DA, G.NC, G.OBS_DIM
MODES = ("HOST", "FQE", "EXACT", "BCA")
ACTORS = ("context_linear", "global_linear", "gain_scheduled")
MAX_ACTION = LQ.MAX_ACTION
KEY_TD3_INIT = 1414676809  # td3_bc.initialize's fold
# np.random.default_rng([settings.seed, rep, TAG, ...])
TAG_INIT, TAG_TIER_A_ROWS, TAG_TIE, TAG_SHUF = 31, 32, 33, 34
EVEN_LADDER = (1.3, 2.0, 4.0, 8.0, 16.0)
B_LADDER = (0.3, 1.0, 3.0, 10.0, 30.0)
CAP_B_LADDER = (0.1, 0.2, 0.3, 0.4, 0.49)
CAP_C_LADDER = (0.1, 0.3, 1.0, 3.0, 10.0)
TARGET_M = (1.3, 2.0)  # S1, S2 (primary)
TIER_A_STEPS, ORACLE_REFRESH, N_SHUF = 10_000, 1_000, 4
REFRESHED = ("err", "ord", "err_ratio")
BCA_OFF = ("the BCA training arm is switched off: pass allow_bca=True only under a separate approval "
           "(the approved scope runs no BCA arm and fits no BCA band)")


@dataclass(frozen=True)
class TrainSettings:
    seed: int = 20261013
    updates: int = 100_000
    gamma: float = 0.98
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    policy_freq: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    alpha: float = 2.5
    actor: str = "context_linear"
    actor_init: str = "zero"
    exact_refresh: int = 100
    exact_lambda: str = "exact"
    exact_level: str = "td3"
    probe: str = "kstar"
    snapshots: tuple = (0.25, 0.5, 1.0)
    chunk: int = 1000
    gs_hidden: int = 64

    def __post_init__(self):
        if self.actor not in ACTORS:
            raise ValueError("actor must be one of " + ", ".join(ACTORS))
        if self.actor_init not in ("zero", "bc") or self.exact_lambda not in ("exact", "q1"):
            raise ValueError("actor_init must be zero|bc and exact_lambda exact|q1")
        if self.probe not in ("kstar", "mid"):
            raise ValueError("probe must be kstar|mid")
        if self.exact_level not in ("td3", "deterministic"):
            raise ValueError("exact_level must be td3|deterministic")
        if self.policy_freq < 2:
            raise ValueError("policy_freq >= 2: harness_update takes td3_bc_update's skip branch for its critic step")


def make_args(settings, rep=0):
    """td3_bc.Args of a rep: the settings' TD3+BC numbers and the rep's seed."""
    seed = int(np.random.default_rng([settings.seed, rep, TAG_INIT]).integers(2 ** 31 - 1))
    return replace(BASE.Args(), seed=seed, discount=settings.gamma, tau=settings.tau, lr=settings.lr,
                   batch_size=settings.batch_size, policy_freq=settings.policy_freq,
                   policy_noise=settings.policy_noise, noise_clip=settings.noise_clip, alpha=settings.alpha)


_TX = {}


def adam(lr):
    """One shared torch_adam object per learning rate, so that train states stack (vmap) and programs are reused."""
    if lr not in _TX:
        _TX[lr] = BASE.C.torch_adam(lr)
    return _TX[lr]


# ---------------------------------------------------------------------------------------------------------
# Actor classes


class ContextLinear:
    """placement_harness.ContextLinearActor: pi = -sum_c code_c K_c s, params {'K': (5, 4, 10)}."""

    name = "context_linear"
    apply = staticmethod(PH.ContextLinearActor.apply)

    def params(self, K):
        return {"K": jnp.asarray(G.World.per_context(K), jnp.float32)}

    def init(self, key):
        return self.params(np.zeros((NC, DA, DS)))

    def gains(self, params):
        return np.asarray(params["K"], np.float64)


class GlobalLinear:
    """pi = -K s for every context, params {'K': (4, 10)} (the negative-control class)."""

    name = "global_linear"

    @staticmethod
    def apply(params, obs):
        K = params["K"]
        return -jnp.einsum("ij,...j->...i", K, obs[..., :K.shape[-1]])

    def params(self, K):
        K = np.asarray(K, np.float64)
        return {"K": jnp.asarray(K if K.shape == (DA, DS) else K.mean(0), jnp.float32)}

    def init(self, key):
        return self.params(np.zeros((DA, DS)))

    def gains(self, params):
        return np.repeat(np.asarray(params["K"], np.float64)[None], NC, 0)


class GainNet(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, code):
        x = nn.relu(BASE.C.torch_dense(self.hidden, NC)(code))
        x = nn.relu(BASE.C.torch_dense(self.hidden, self.hidden)(x))
        return BASE.C.torch_dense(DA * DS, self.hidden)(x).reshape(code.shape[:-1] + (DA, DS))


class GainScheduled:
    """g7: pi = -K_theta(code) s with K_theta = GainNet (linear in s within an episode)."""

    name = "gain_scheduled"

    def __init__(self, hidden=64):
        self.net = GainNet(hidden)
        self.apply = self._apply

    def _apply(self, params, obs):
        return -jnp.einsum("...ij,...j->...i", self.net.apply(params, obs[..., DS:]), obs[..., :DS])

    def init(self, key):
        return self.net.init(key, jnp.zeros((NC,)))

    def gains(self, params, codes=None):
        codes = np.eye(NC) if codes is None else codes
        return np.asarray(self.net.apply(params, jnp.asarray(codes, jnp.float32)), np.float64)

    def fit(self, params, K_target, steps=3000, lr=1e-3):
        """Adam on mean |K_theta(e_c) - K_target,c|^2 over the one-hot codes; returns (params, final loss)."""
        codes, target = jnp.eye(NC), jnp.asarray(K_target, jnp.float32)
        tx = optax.adam(lr)

        def body(carry, _):
            p, opt = carry
            loss, g = jax.value_and_grad(lambda q: jnp.mean(jnp.square(self.net.apply(q, codes) - target)))(p)
            upd, opt = tx.update(g, opt, p)
            return (optax.apply_updates(p, upd), opt), loss

        (params, _), losses = jax.jit(lambda p: jax.lax.scan(body, (p, tx.init(p)), None, length=steps))(params)
        return params, float(losses[-1])


_KINDS = {}


def actor_kind(name, hidden=64):
    """One shared instance per actor class (programs are cached per class)."""
    key = (name, hidden)
    if key not in _KINDS:
        if name == "context_linear":
            _KINDS[key] = ContextLinear()
        elif name == "global_linear":
            _KINDS[key] = GlobalLinear()
        elif name == "gain_scheduled":
            _KINDS[key] = GainScheduled(hidden)
        else:
            raise ValueError("actor must be one of " + ", ".join(ACTORS))
    return _KINDS[key]


_CRITIC = BASE.DualCritic(OBS_DIM, DA)


def bc_gains(rows):
    """Least-squares behaviour-clone gains per context (5, 4, 10) and pooled (4, 10) on one split's rows."""
    s, a, ctx = rows["state"], rows["action"], rows["context"]
    per = np.stack([PH.ls_gain(s[ctx == c], a[ctx == c]) for c in range(NC)])
    return per, PH.ls_gain(s, a)


def start_params(kind, settings, rows, key):
    """The actor's initial parameters (actor_init) and, for 'bc', the fit record."""
    if settings.actor_init == "zero":
        return kind.init(key), None
    per, pooled = bc_gains(rows)
    if kind.name == "gain_scheduled":
        params, loss = kind.fit(kind.init(key), per)
        return params, dict(gain_fit_loss=loss)
    return kind.params(pooled if kind.name == "global_linear" else per), None


def probe_gains(world, settings):
    if settings.probe == "kstar":
        return world.k_star.copy()
    return 0.5 * (world.k_behaviour + world.k_star)


# ---------------------------------------------------------------------------------------------------------
# Exact Q^pi tables and row-level truth


def exact_tables(world, K, previous=None, smoothing=None):
    """Float32 per-context (H_c, h_c) of the policy -K_c s and the stable flags. A context whose closed loop is not
    discounted-stable keeps `previous`'s table (zeros without one). smoothing = (policy_noise, noise_clip) adds TD3's
    fixed-point level c_c (World.td3_level) to h_c: the tables then hold Q^pi + c_c, the value a perfect TD3 critic
    reads, so lambda = alpha / mean |Q| is on the host critic's scale; H and every action gradient are unchanged."""
    K = G.World.per_context(K)
    H = np.zeros((NC, DS + DA, DS + DA)) if previous is None else np.asarray(previous[0], np.float64).copy()
    h = np.zeros(NC) if previous is None else np.asarray(previous[1], np.float64).copy()
    stable = np.zeros(NC, bool)
    level = np.zeros(NC) if smoothing is None else world.td3_level(K, *smoothing)
    for c, system in enumerate(world.systems):
        try:
            H[c], h[c], _ = LQ.exact_q(system, K[c])
            h[c] += level[c]
            stable[c] = True
        except ValueError:
            pass
    return (jnp.asarray(H, jnp.float32), jnp.asarray(h, jnp.float32)), stable


def exact_q_jax(tables, obs, action):
    """sum_c code_c (z^T H_c z + h_c), z = (s, a): Q^pi with the exact one-hot code (float32)."""
    H, h = tables
    z = jnp.concatenate([obs[..., :DS], action], axis=-1)
    per = jnp.einsum("...i,cij,...j->...c", z, H, z) + h
    return jnp.sum(obs[..., DS:] * per, axis=-1)


def exact_rows(world, K, s, a, context):
    """Q^pi_c(s, a) and grad_a Q^pi_c per row (float64); NaN where context c is not discounted-stable."""
    K = G.World.per_context(K)
    context = np.asarray(context, np.int64)
    q, g = np.full(len(context), np.nan), np.full((len(context), DA), np.nan)
    stable = np.zeros(NC, bool)
    for c, system in enumerate(world.systems):
        rows = context == c
        try:
            H, h, _ = LQ.exact_q(system, K[c])
        except ValueError:
            continue
        stable[c] = True
        if rows.any():
            z = np.concatenate([s[rows], a[rows]], 1)
            q[rows] = LQ.qform(H, h, s[rows], a[rows])
            g[rows] = 2 * z @ H[:, DS:]
    return q, g, stable


# ---------------------------------------------------------------------------------------------------------
# Updates


def _soft(args, src, tgt):
    """td3_bc_update's soft target update."""
    return tgt.replace(step=tgt.step + 1, params=optax.incremental_update(src.params, tgt.params, args.tau))


def actor_step(args, models, actor, critic_params, batch, bc_w=None, exact=None, lam="exact"):
    """td3_bc's actor step (its _actor_loss_fn and apply_gradients) with hooks: bc_w per-row BC multipliers (None is
    td3_bc's unweighted path), exact (H, h) tables (the actor reads Q^pi instead of Q1; lam 'exact' takes lambda from
    Q^pi, 'q1' from Q1). Returns the new actor TrainState and (actor_loss, q_mean, lambda, bc_loss)."""

    def _actor_loss_fn(params, critic_params):
        pi = models[0].apply(params, batch.obs)
        if exact is None or lam == "q1":
            q = models[1].apply(critic_params, batch.obs, pi)[..., 0]
        q_read = q if exact is None else exact_q_jax(exact, batch.obs, pi)
        q_lam = q if exact is None or lam == "q1" else q_read
        lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q_lam).mean())
        if bc_w is None:
            bc = jnp.square(pi - batch.action).mean()
        else:
            bc = (jax.lax.stop_gradient(bc_w) * jnp.square(pi - batch.action).mean(axis=-1)).mean()
        return (-lmbda * q_read.mean() + bc, (q_read.mean(), lmbda, bc))

    grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
    (a_loss, aux), a_grad = grad_fn(actor.params, critic_params)
    return actor.apply_gradients(grads=a_grad), (a_loss, *aux)


def harness_update(args, models, state, batch, it, rng_noise, max_action, bc_w=None, exact=None, lam="exact"):
    """td3_bc_update with the actor hooks: td3_bc_update's own critic step (called with it = 1, its skip branch), then
    every policy_freq updates actor_step and td3_bc's soft target updates. Hooks off: td3_bc_update bit for bit."""
    if args.policy_freq < 2:
        raise ValueError("harness_update needs policy_freq >= 2")
    state, metrics = BASE.td3_bc_update(args, models[0].apply, models[1].apply, state, batch, 1, rng_noise, max_action)

    def _do_actor_update(st):
        actor, aux = actor_step(args, models, st.actor, st.critic.params, batch, bc_w, exact, lam)
        st = st._replace(actor=actor)
        st = st._replace(actor_target=_soft(args, st.actor, st.actor_target),
                         critic_target=_soft(args, st.critic, st.critic_target))
        return st, aux

    def _skip_actor_update(st):
        zero = jnp.float32(0.0)
        return st, (zero, zero, zero, zero)

    state, (a_loss, q_mean, lmbda, bc) = jax.lax.cond(it % args.policy_freq == 0, _do_actor_update,
                                                      _skip_actor_update, state)
    metrics.update(actor_loss=a_loss, q_mean=q_mean, bc_loss=bc, **{"lambda": lmbda})
    return state, metrics


def _train_step(args, models, mode, max_action, train, table, tables, lam, runner, _):
    """One update in td3_bc.make_train_step's layout (rng split 3, randint rows, it + 1) for HOST, FQE or EXACT."""
    rng, state, it = runner
    it = it + 1
    rng, rng_batch, rng_noise = jax.random.split(rng, 3)
    idx = jax.random.randint(rng_batch, (args.batch_size,), 0, train.obs.shape[0])
    batch = jax.tree_util.tree_map(lambda x: x[idx], train)
    mult = None if table is None else table[idx]
    if mode == "HOST":
        new, metrics = BASE.td3_bc_update(args, models[0].apply, models[1].apply, state, batch, it, rng_noise,
                                          max_action, bc_multiplier=mult)
    elif mode == "FQE":
        new, metrics = BASE.td3_bc_update(args, models[0].apply, models[1].apply, state, batch, it, rng_noise,
                                          max_action)
        new = new._replace(actor=state.actor, actor_target=state.actor_target)
    elif mode == "EXACT":
        new, metrics = harness_update(args, models, state, batch, it, rng_noise, max_action, mult, tables, lam)
    else:
        raise ValueError("unknown mode " + repr(mode))
    return (rng, new, it), dict(metrics, it=it)


_PROGRAMS = {}


def chunk_program(args, models, mode, length, has_table, has_tables, lam):
    """Compiled lax.scan of `length` updates, cached per (args, actor class, mode, length, hook presence)."""
    key = (tuple(sorted(asdict(args).items())), models[0].name, mode, length, has_table, has_tables, lam)
    if key not in _PROGRAMS:
        def run(runner, train, table, tables):
            step = partial(_train_step, args, models, mode, MAX_ACTION, train, table if has_table else None,
                           tables if has_tables else None, lam)
            return jax.lax.scan(step, runner, None, length=length)

        _PROGRAMS[key] = jax.jit(run)
    return _PROGRAMS[key]


# ---------------------------------------------------------------------------------------------------------
# State, snapshots and multipliers


def initial_state(args, kind, actor_params, lr):
    """(rng, AgentTrainState) with td3_bc.initialize's key scheme; the critic is DualCritic.init(k_critic)."""
    rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), KEY_TD3_INIT)
    rng, ka, kc = jax.random.split(rng, 3)
    obs, action = jnp.zeros(OBS_DIM), jnp.zeros(DA)
    if actor_params is None:
        actor_params = kind.init(ka)
    actor = TrainState.create(apply_fn=kind.apply, params=actor_params, tx=adam(lr))
    critic = TrainState.create(apply_fn=_CRITIC.apply, params=_CRITIC.init(kc, obs, action), tx=adam(lr))
    return rng, BASE.AgentTrainState(actor, actor, critic, critic), ka


def run_dir(root, tag, rep, mode):
    """The conventional output directory of one co-trained run: <root>/<tag>/rep<rep>/<mode> (e.g. root runs/glq)."""
    return Path(root) / tag / f"rep{rep}" / mode


def save_snapshot(path, state):
    """Flax msgpack of the whole AgentTrainState; returns its SHA-256."""
    blob = serialization.to_bytes(state)
    Path(path).write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def load_snapshot(path, template, sha256=None):
    blob = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError(f"{path}: SHA-256 differs from the recorded one")
    return serialization.from_bytes(template, blob)


def tree_sha256(tree):
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()


@dataclass
class Multiplier:
    """A per-row bc_multiplier table over training rows: table(state, it) -> (n_train,) array, recomputed every
    `every` updates (None: once, at the start)."""
    name: str
    table: Callable[[Any, int], np.ndarray]
    every: int = None


# ---------------------------------------------------------------------------------------------------------
# Critic diagnostics


_ROW_PROGRAMS = {}


def critic_rows(critic_params, obs, action, chunk=8192):
    """Both heads Q (n, 2) and grad_a Q1 (n, 4) (jax.vjp with cotangent (1, 0)), float32 in, float64 out, chunked."""
    if "rows" not in _ROW_PROGRAMS:
        def program(p, o, a):
            q, vjp = jax.vjp(lambda x: _CRITIC.apply(p, o, x), a)
            cot = jnp.zeros_like(q).at[:, 0].set(1.0)
            return q, vjp(cot)[0]

        _ROW_PROGRAMS["rows"] = jax.jit(program)
    program = _ROW_PROGRAMS["rows"]
    obs, action = np.asarray(obs, np.float32), np.asarray(action, np.float32)
    n = len(obs)
    qs, gs = [], []
    for i in range(0, n, chunk):
        o, a = obs[i:i + chunk], action[i:i + chunk]
        pad = chunk - len(o)
        if pad:
            o, a = np.concatenate([o, np.zeros((pad, o.shape[1]), np.float32)]), np.concatenate(
                [a, np.zeros((pad, a.shape[1]), np.float32)])
        q, g = program(critic_params, jnp.asarray(o), jnp.asarray(a))
        qs.append(np.asarray(q, np.float64)[:chunk - pad])
        gs.append(np.asarray(g, np.float64)[:chunk - pad])
    return np.concatenate(qs), np.concatenate(gs)


def _check_one_hot(rows):
    code = rows["obs"][:, DS:]
    if not np.array_equal(code, np.eye(NC)[rows["context"]]):
        raise NotImplementedError("row-level exact diagnostics need the exact one-hot code (g7's noisy code is per "
                                  "episode: use World.j_episodes for returns)")


def error_rows(world, critic_params, K, rows, smoothing=(G.TD3_POLICY_NOISE, G.TD3_NOISE_CLIP)):
    """Per-row exact critic errors at the gains K on one split's rows (module docstring). float64 arrays.

    smoothing = (policy_noise, noise_clip) of the TD3 target that trained the critic: e1_td3 = Q1 - (Q^pi + c_c) is
    the level error against TD3's fixed point (World.td3_level; a perfect critic has e1_td3 = 0 and e1 = c_c)."""
    _check_one_hot(rows)
    K = G.World.per_context(K)
    ctx = np.asarray(rows["context"], np.int64)
    obs32 = np.asarray(rows["obs"], np.float32)
    s = obs32[:, :DS].astype(np.float64)
    a32 = np.asarray(rows["action"], np.float32)
    pi32 = (-np.einsum("nij,nj->ni", K[ctx], s)).astype(np.float32)
    q_pi, g_pi = critic_rows(critic_params, obs32, pi32)
    q_log, g_log = critic_rows(critic_params, obs32, a32)
    pi, a = pi32.astype(np.float64), a32.astype(np.float64)
    Qe_pi, ge_pi, stable = exact_rows(world, K, s, pi, ctx)
    Qe_log, ge_log, _ = exact_rows(world, K, s, a, ctx)
    e1_pi, e1_log = q_pi[:, 0] - Qe_pi, q_log[:, 0] - Qe_log
    level = world.td3_level(K, *smoothing)
    lvl = level[ctx]
    with np.errstate(invalid="ignore"):
        out = dict(context=ctx, q1_pi=q_pi[:, 0], q2_pi=q_pi[:, 1], q1_logged=q_log[:, 0], q2_logged=q_log[:, 1],
                   qpi_pi=Qe_pi, qpi_logged=Qe_log, e1_pi=e1_pi, e1_logged=e1_log,
                   emin_logged=q_log.min(1) - Qe_log,
                   E_pi=np.linalg.norm(g_pi - ge_pi, axis=1), E_logged=np.linalg.norm(g_log - ge_log, axis=1),
                   grad_q1_pi=g_pi, grad_qpi_pi=ge_pi, e_A=e1_log - e1_pi,
                   or_d=np.maximum(0.0, e1_pi - e1_log), rel_e1_pi=np.abs(e1_pi) / np.abs(Qe_pi),
                   rel_e1_logged=np.abs(e1_log) / np.abs(Qe_log), e1_td3_pi=e1_pi - lvl, e1_td3_logged=e1_log - lvl,
                   rel_e1_td3_pi=np.abs(e1_pi - lvl) / np.abs(Qe_pi),
                   rel_e1_td3_logged=np.abs(e1_log - lvl) / np.abs(Qe_log), pi=pi)
    out["stable"] = stable
    out["td3_level"] = level
    return out


def _nanmean(x):
    x = np.asarray(x, np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def _nanmedian(x):
    x = np.asarray(x, np.float64)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else None


def summarize_errors(err):
    """Per-context and pooled summaries of error_rows (strict-JSON friendly; None where undefined)."""
    ctx, E = err["context"], err["E_pi"]
    sq = np.where(np.isfinite(E), E ** 2, 0.0)
    total = float(sq.sum())
    per = {}
    for c in range(NC):
        m = ctx == c
        per[G.CONTEXTS[c]] = dict(
            rows=int(m.sum()), stable=bool(err["stable"][c]), nonfinite_rows=int(np.sum(~np.isfinite(E[m]))),
            mean_E_pi=_nanmean(E[m]), mean_E_logged=_nanmean(err["E_logged"][m]),
            mean_abs_e1_pi=_nanmean(np.abs(err["e1_pi"][m])), mean_abs_e1_logged=_nanmean(np.abs(err["e1_logged"][m])),
            mean_e1_pi=_nanmean(err["e1_pi"][m]), mean_e1_logged=_nanmean(err["e1_logged"][m]),
            median_rel_e1_pi=_nanmedian(err["rel_e1_pi"][m]), median_rel_e1_logged=_nanmedian(err["rel_e1_logged"][m]),
            td3_level=_nanmean(err["td3_level"][c:c + 1]), mean_e1_td3_pi=_nanmean(err["e1_td3_pi"][m]),
            median_rel_e1_td3_pi=_nanmedian(err["rel_e1_td3_pi"][m]),
            median_rel_e1_td3_logged=_nanmedian(err["rel_e1_td3_logged"][m]),
            mean_abs_eA=_nanmean(np.abs(err["e_A"][m])), mean_or_d=_nanmean(err["or_d"][m]),
            share_or_d_positive=_nanmean((err["or_d"][m] > 0).astype(np.float64)),
            share_sum_E2=float(sq[m].sum() / total) if total > 0 else None)
    mean_of = lambda cs: _nanmean(E[np.isin(ctx, cs)])
    ratio = lambda x, y: float(x / y) if x is not None and y not in (None, 0.0) else None
    thin = np.isin(ctx, G.THIN)
    pooled = dict(
        thin_share_sum_E2=float(sq[thin].sum() / total) if total > 0 else None, thin_share_rows=float(thin.mean()),
        mean_E_rich_main=mean_of(G.RICH_MAIN), mean_E_rich_all=mean_of(G.RICH),
        ratio_E_Twrong_rich_main=ratio(mean_of([G.T_WRONG]), mean_of(G.RICH_MAIN)),
        ratio_E_Twrong_rich_all=ratio(mean_of([G.T_WRONG]), mean_of(G.RICH)),
        ratio_E_Twrong_Tright=ratio(mean_of([G.T_WRONG]), mean_of([G.T_RIGHT])),
        ratio_E_Tright_Twrong=ratio(mean_of([G.T_RIGHT]), mean_of([G.T_WRONG])),
        rich_main_median_rel_e1_pi=_nanmedian(err["rel_e1_pi"][np.isin(ctx, G.RICH_MAIN)]),
        rich_main_median_rel_e1_logged=_nanmedian(err["rel_e1_logged"][np.isin(ctx, G.RICH_MAIN)]),
        rich_main_median_rel_e1_td3_pi=_nanmedian(err["rel_e1_td3_pi"][np.isin(ctx, G.RICH_MAIN)]),
        rich_main_median_rel_e1_td3_logged=_nanmedian(err["rel_e1_td3_logged"][np.isin(ctx, G.RICH_MAIN)]),
        unstable_contexts=[G.CONTEXTS[c] for c in range(NC) if not err["stable"][c]])
    return dict(per_context=per, pooled=pooled)


def save_error_rows(path, err):
    """The per-row arrays (float32) for later stages (R5); npz is gitignored."""
    np.savez_compressed(path, **{k: np.asarray(v, np.float32) if np.asarray(v).dtype.kind == "f" else np.asarray(v)
                                 for k, v in err.items()})


# ---------------------------------------------------------------------------------------------------------
# Co-training


def _event_steps(total, chunk, snapshots, every=()):
    events = set(range(chunk, total + 1, chunk)) | set(snapshots) | {total}
    for e in every:
        if e:
            events |= set(range(e, total + 1, e))
    return sorted(x for x in events if 0 < x <= total)


def snapshot_steps(settings, updates):
    steps = sorted({int(round(f * updates)) for f in settings.snapshots})
    if any(s <= 0 for s in steps):
        raise ValueError("snapshot fractions must give positive update counts")
    return steps


def _chunk_metrics(metrics, policy_freq):
    it = np.asarray(metrics["it"])
    actor = it % policy_freq == 0
    out = dict(it_last=int(it[-1]), critic_loss=float(np.mean(np.asarray(metrics["critic_loss"]))))
    for name in ("actor_loss", "q_mean", "lambda", "bc_loss"):
        x = np.asarray(metrics[name])[actor]
        out[name] = float(np.mean(x)) if x.size else None
    return out


def cotrain(world, data, settings, mode, rep, *, out_dir=None, multiplier=None, allow_bca=False, diagnostics=True,
            updates=None, save_rows=True, log=print):
    """One co-trained run (HOST, FQE, EXACT; BCA only with allow_bca) on one seed's data. Returns a record with the
    final state, the in-memory snapshots {update: state}, chunk metrics, snapshot files and exact diagnostics."""
    if mode not in MODES:
        raise ValueError("mode must be one of " + ", ".join(MODES))
    if mode == "BCA":
        if not allow_bca:
            raise RuntimeError(BCA_OFF)
        return _bca_cotrain(world, data, settings, rep, out_dir=out_dir, updates=updates, log=log)
    if mode == "EXACT" and data["meta"].get("code_sd") is not None:
        raise NotImplementedError("EXACT reads Q^pi through the one-hot code; the noisy code has no per-row table")
    started = time.perf_counter()
    args = make_args(settings, rep)
    smoothing = (settings.policy_noise, settings.noise_clip)  # the TD3 target's smoothing (td3_level)
    kind = actor_kind("context_linear" if mode == "FQE" else settings.actor, settings.gs_hidden)
    models = (kind, _CRITIC)
    train = G.transitions(data["train"])
    total = int(settings.updates if updates is None else updates)
    snaps = snapshot_steps(settings, total)
    rng0 = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed), KEY_TD3_INIT), 3)[1]
    fit = None
    if mode == "FQE":
        actor_params = kind.params(probe_gains(world, settings))
    else:
        actor_params, fit = start_params(kind, settings, data["train"], rng0)
    rng, state, _ = initial_state(args, kind, actor_params, settings.lr)
    runner = (rng, state, jnp.int32(0))
    every = []
    if mode == "EXACT":
        every.append(settings.exact_refresh)
    if isinstance(multiplier, Multiplier) and multiplier.every:
        every.append(multiplier.every)
    events = _event_steps(total, settings.chunk, snaps, every)
    table = tables = None
    if multiplier is not None:
        if mode == "FQE":
            raise ValueError("FQE's actor is frozen: a BC multiplier has no effect")
        table = multiplier.table(state, 0) if isinstance(multiplier, Multiplier) else multiplier
        table = jnp.asarray(np.asarray(table, np.float32))
        if table.shape != (len(train.obs),):
            raise ValueError("a multiplier table has one value per training row")
    exact_unstable, previous = 0, None
    record = dict(kind="glq-cotrain", mode=mode, rep=rep, settings=asdict(settings), args=asdict(args),
                  actor_class=kind.name, updates=total, snapshot_steps=snaps, data_meta=data["meta"],
                  spec_sha256=G.spec_digest(world.spec), multiplier=getattr(multiplier, "name", None if multiplier is
                                                                             None else "fixed"),
                  start_fit=fit, chunks=[], snapshots=[], diagnostics={})
    if mode == "FQE":
        record["probe"] = dict(name=settings.probe, K=probe_gains(world, settings).tolist())
    states, done = {}, 0
    out_dir = Path(out_dir) if out_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for step in events:
        if mode == "EXACT" and done % settings.exact_refresh == 0:
            tables, stable = exact_tables(world, kind.gains(runner[1].actor.params), previous,
                                          smoothing if settings.exact_level == "td3" else None)
            exact_unstable += int((~stable).sum())
            previous = tables
        if isinstance(multiplier, Multiplier) and multiplier.every and done and done % multiplier.every == 0:
            table = jnp.asarray(np.asarray(multiplier.table(runner[1], done), np.float32))
        program = chunk_program(args, models, mode, step - done, table is not None, tables is not None,
                                settings.exact_lambda)
        runner, metrics = program(runner, train, table, tables)
        record["chunks"].append(_chunk_metrics(metrics, args.policy_freq))
        done = step
        if step in snaps:
            states[step] = runner[1]
            pct = int(round(100 * step / total))
            snap = dict(update=step, fraction=step / total, actor_sha256=tree_sha256(runner[1].actor.params),
                        critic_sha256=tree_sha256(runner[1].critic.params))
            if out_dir is not None:
                path = out_dir / f"snapshot_{pct:03d}.msgpack"
                snap.update(path=str(path), sha256=save_snapshot(path, runner[1]))
            record["snapshots"].append(snap)
            if diagnostics:
                K = kind.gains(runner[1].actor.params)  # FQE: the probe as the critic's targets read it (float32)
                err = error_rows(world, runner[1].critic.params, K, data["eval"], smoothing)
                record["diagnostics"][str(step)] = dict(summarize_errors(err), K=K.tolist(),
                                                        score=world.score(K) if mode != "FQE" else None)
                if out_dir is not None and save_rows:
                    save_error_rows(out_dir / f"errors_{pct:03d}.npz", err)
            if log:
                last = record["chunks"][-1]
                log(f"{mode} rep {rep}: update {step}/{total}, critic loss {last['critic_loss']:.4g}")
        if not all(np.isfinite(v) for v in jax.tree_util.tree_leaves(record["chunks"][-1]) if v is not None):
            record["nonfinite_at"] = step
            log(f"{mode} rep {rep}: nonfinite metrics at update {step}; stopping")
            break
    record.update(exact_unstable_context_refreshes=exact_unstable if mode == "EXACT" else None,
                  seconds=time.perf_counter() - started, final_update=done)
    if out_dir is not None:
        record["run_json_sha256"] = G._write(out_dir / "run.json", G._clean(record))
    record["state"], record["states"] = runner[1], states
    return record


def _bca_cotrain(world, data, settings, rep, *, out_dir=None, updates=None, log=print, refresh_every=5000,
                 execute=True):
    """The co-trained BCA arm (switched off; module docstring). td3_bc_bca.make_train_step scanned in chunks of
    refresh_every updates, td3_bc_bca.refresh on the calibration bank between chunks (the first after the first
    chunk, as the runtime's refresh events). execute=False returns the lowered first-chunk program and does not run."""
    args = make_args(settings, rep)
    kind = actor_kind(settings.actor, settings.gs_hidden)
    config = P.Config(arm="bca", posterior=WBCPConfig(0.1, 0.95, 1000), blend=0.5)
    cal = Calibrator(jnp.zeros(OBS_DIM), jnp.ones(OBS_DIM), state_dep=True)
    models = (kind, _CRITIC, cal)
    train, bank_rows = G.transitions(data["train"]), G.transitions(data["cal"])
    rng0 = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed), KEY_TD3_INIT), 3)[1]
    actor_params, _ = start_params(kind, settings, data["train"], rng0)
    rng, native, _ = initial_state(args, kind, actor_params, settings.lr)
    cal_state = TrainState.create(apply_fn=cal.apply, params=cal.init(jax.random.fold_in(rng, LQ.KEY_INIT),
                                                                      jnp.zeros((1, OBS_DIM)), jnp.zeros((1, DA))),
                                  tx=optax.adam(config.cal_lr))
    state = P.State(native, cal_state, jnp.asarray(1.0), initial_reference(cal_state.params))
    step = P.make_train_step(args, config, models, train, MAX_ACTION)
    program = jax.jit(lambda carry: jax.lax.scan(step, carry, None, length=refresh_every))
    carry = (rng, state, jnp.int32(0))
    if not execute:
        return program.lower(carry)
    total = int(settings.updates if updates is None else updates)
    if total % refresh_every:
        raise ValueError("updates must be a multiple of refresh_every")
    record = dict(kind="glq-cotrain", mode="BCA", rep=rep, settings=asdict(settings), chunks=[], refreshes=[])
    for k in range(total // refresh_every):
        carry, metrics = program(carry)
        P.require_valid(metrics)
        record["chunks"].append(dict(it_last=int(carry[2]), critic_loss=float(np.mean(metrics["critic_loss"]))))
        new, diag, _ = P.refresh(args, config, models, carry[1], bank_rows, jax.random.fold_in(rng, k),
                                 MAX_ACTION, heldout_ids=np.arange(len(bank_rows.obs)))
        carry = (carry[0], new, carry[2])
        record["refreshes"].append({k2: bool(v) for k2, v in diag.items()})
    record["state"] = carry[1]
    return record


# ---------------------------------------------------------------------------------------------------------
# Tier A: actor-only arms on a frozen critic


def _tier_a_loop(args, models, critic_params, actor, train, rows, table):
    """lax.scan of actor_step over precomputed batch rows for one arm; table is the arm's per-row multiplier."""
    def body(a, r):
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        a, aux = actor_step(args, models, a, critic_params, batch, table[r])
        return a, jnp.stack(aux)

    return jax.lax.scan(body, actor, rows)


def _none_path_loop(args, models, critic_params, actor, train, rows):
    def body(a, r):
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        a, aux = actor_step(args, models, a, critic_params, batch)
        return a, jnp.stack(aux)

    return jax.lax.scan(body, actor, rows)


def family_ladders(n_shuf=N_SHUF):
    """{family: (assignment variable, knob ladder, weight form)}; EVEN sets the targets, the rest are matched."""
    fam = {"EVEN": (None, EVEN_LADDER, "const"), "OR-wrong": ("wrong", B_LADDER, "linear"),
           "OR-thin": ("thin", B_LADDER, "linear"), "ANTI": ("rich", B_LADDER, "linear")}
    fam.update({f"SHUF{k}": (f"shuf{k}", B_LADDER, "linear") for k in range(n_shuf)})
    fam.update({"OR-err": ("err", B_LADDER, "linear"), "OR-D": ("ord", B_LADDER, "linear"),
                "CAP-wrong": ("wrong", CAP_B_LADDER, "linear"), "CAP-err": ("err_ratio", CAP_C_LADDER, "saturating")})
    return fam


def weights(form, knob, x):
    """Row multipliers of one arm (float64)."""
    if form == "const":
        return np.full(len(x), float(knob))
    if form == "linear":
        return 1.0 + knob * x
    if form == "saturating":
        return 1.0 + 0.5 * (knob * x) / (1.0 + knob * x)
    raise ValueError(form)


class TierA:
    """One seed's Tier A: the frozen critic, the start, the shared batch stream and the compiled arm programs."""

    def __init__(self, world, data, critic_params, settings, rep, *, steps=TIER_A_STEPS, refresh=ORACLE_REFRESH,
                 n_shuf=N_SHUF, families=None):
        if steps % refresh:
            raise ValueError("steps must be a multiple of the oracle refresh interval")
        _check_one_hot(data["train"])
        self.world, self.settings, self.rep = world, settings, rep
        self.steps, self.refresh = steps, refresh
        self.args = make_args(settings, rep)
        self.kind = actor_kind(settings.actor, settings.gs_hidden)
        self.models = (self.kind, _CRITIC)
        self.critic_params = critic_params
        self.train = G.transitions(data["train"])
        tr = data["train"]
        self.s, self.ctx = np.asarray(tr["obs"], np.float32)[:, :DS].astype(np.float64), np.asarray(tr["context"])
        self.obs32, self.a32 = np.asarray(tr["obs"], np.float32), np.asarray(tr["action"], np.float32)
        n = len(self.ctx)
        self.K_bc, self.K_bc_pooled = bc_gains(tr)
        rng = lambda *t: np.random.default_rng([settings.seed, rep, *t])
        self.tie = rng(TAG_TIE).permutation(n)
        thin = np.isin(self.ctx, G.THIN).astype(np.float64)
        self.static = dict(wrong=(self.ctx == G.T_WRONG).astype(np.float64), thin=thin, rich=1.0 - thin)
        self.static.update({f"shuf{k}": thin[rng(TAG_SHUF, k).permutation(n)] for k in range(n_shuf)})
        self.families = family_ladders(n_shuf) if families is None else families
        self.rows = rng(TAG_TIER_A_ROWS).integers(n, size=(steps // refresh, refresh, settings.batch_size))
        self.rows = jnp.asarray(self.rows, jnp.int32)
        ev = data["eval"]
        self.Sigma_ev = M.second_moments(np.asarray(ev["obs"], np.float32)[:, :DS].astype(np.float64), ev["context"],
                                         NC)
        key = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(self.args.seed), KEY_TD3_INIT), 3)[1]
        if self.kind.name == "gain_scheduled":
            params, loss = self.kind.fit(self.kind.init(key), self.K_bc)
            self.start_fit = dict(gain_fit_loss=loss)
        else:
            params = self.kind.params(self.K_bc_pooled if self.kind.name == "global_linear" else self.K_bc)
            self.start_fit = None
        self.start = TrainState.create(apply_fn=self.kind.apply, params=params, tx=adam(settings.lr))
        self.K_start = self.kind.gains(params)
        self._programs = {}
        self.unstable_refreshes = 0
        self._oracle_cache = {}
        self._q_logged = None

    def program(self, n):
        if n not in self._programs:
            loop = partial(_tier_a_loop, self.args, self.models)
            self._programs[n] = jax.jit(jax.vmap(loop, in_axes=(None, 0, None, None, 0)))
        return self._programs[n]

    def oracle_variables(self, K, previous=None):
        """Exact E_i and OR-D_i at the gains K on every training row; contexts without a stable closed loop keep
        `previous`'s values (zeros without one)."""
        key = hashlib.sha256(np.ascontiguousarray(K).tobytes()).hexdigest()
        if key in self._oracle_cache:
            return self._oracle_cache[key]
        K = G.World.per_context(K)
        pi32 = (-np.einsum("nij,nj->ni", K[self.ctx], self.s)).astype(np.float32)
        q_pi, g_pi = critic_rows(self.critic_params, self.obs32, pi32)
        if self._q_logged is None:  # the critic is frozen: Q1 at the logged actions is computed once
            self._q_logged = critic_rows(self.critic_params, self.obs32, self.a32)[0]
        q_log = self._q_logged
        Qe_pi, ge_pi, stable = exact_rows(self.world, K, self.s, pi32.astype(np.float64), self.ctx)
        Qe_log, _, _ = exact_rows(self.world, K, self.s, self.a32.astype(np.float64), self.ctx)
        E = np.linalg.norm(g_pi - ge_pi, axis=1)
        ord_ = np.maximum(0.0, (q_pi[:, 0] - q_log[:, 0]) - (Qe_pi - Qe_log))
        bad = ~stable[self.ctx]
        if bad.any():
            self.unstable_refreshes += 1
            E[bad] = 0.0 if previous is None else previous["E"][bad]
            ord_[bad] = 0.0 if previous is None else previous["ord"][bad]
        out = dict(E=E, ord=ord_)
        self._oracle_cache = {key: out}
        return out

    def variable(self, name, oracle=None):
        if name in self.static:
            return self.static[name]
        if name == "err":
            return PH.rank_normal(oracle["E"], tie_perm=self.tie)[0]
        if name == "err_ratio":
            m = oracle["E"].mean()
            return oracle["E"] / m if m > 0 else np.zeros_like(oracle["E"])
        if name == "ord":
            m = oracle["ord"].mean()
            return oracle["ord"] / m if m > 0 else np.zeros_like(oracle["ord"])
        raise ValueError(name)

    def table(self, family, knob, oracle=None):
        """The per-row multipliers of one arm (NONE: 1 on every row)."""
        if family == "NONE":
            return np.ones(len(self.ctx))
        var, _, form = self.families[family]
        x = np.zeros(len(self.ctx)) if var is None else self.variable(var, oracle)
        return weights(form, knob, x)

    def run(self, arms, log=None):
        """Train every arm [(family, knob)] for `steps` actor steps (vmapped). Returns gains (n, 5, 4, 10) float64,
        per-chunk mean metrics (n, chunks, 4: actor loss, q mean, lambda, bc loss) and the refresh count."""
        n = len(arms)
        refreshed = [f != "NONE" and self.families[f][0] in REFRESHED for f, _ in arms]
        oracle = self.oracle_variables(self.K_start) if any(refreshed) else None
        prev = [oracle] * n
        tables = np.stack([self.table(f, k, oracle) for f, k in arms]).astype(np.float32)
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), self.start)
        program, means = self.program(n), []
        for chunk in range(self.steps // self.refresh):
            states, aux = program(self.critic_params, states, self.train, self.rows[chunk], jnp.asarray(tables))
            means.append(np.asarray(aux, np.float64).mean(1))
            if chunk + 1 < self.steps // self.refresh and any(refreshed):
                for i, (f, k) in enumerate(arms):
                    if refreshed[i]:
                        K_i = self.kind.gains(jax.tree_util.tree_map(lambda x: x[i], states.params))
                        prev[i] = self.oracle_variables(K_i, prev[i])
                        tables[i] = self.table(f, k, prev[i])
            if log:
                log(f"tier A: {n} arms, step {(chunk + 1) * self.refresh}/{self.steps}")
        gains = np.stack([self.kind.gains(jax.tree_util.tree_map(lambda x: x[i], states.params)) for i in range(n)])
        return gains, np.stack(means, 1)

    def none_path(self):
        """NONE through td3_bc's unweighted BC path (bc_multiplier=None), on the same batches."""
        loop = jax.jit(partial(_none_path_loop, self.args, self.models))
        actor = self.start
        for chunk in range(self.steps // self.refresh):
            actor, _ = loop(self.critic_params, actor, self.train, self.rows[chunk])
        return self.kind.gains(actor.params)

    def strength(self, K, K_none):
        return M.strength(K, K_none, self.Sigma_ev)


def tier_a_train(world, data, critic_params, settings, rep, *, steps=TIER_A_STEPS, refresh=ORACLE_REFRESH,
                 n_shuf=N_SHUF, targets=TARGET_M, tol=M.MATCH_TOL, max_rounds=M.MATCH_ROUNDS, families=None,
                 match=True, log=print):
    """Tier A's blind phase: the ladders, the targets and the matched knobs. Computes no return."""
    started = time.perf_counter()
    T = TierA(world, data, critic_params, settings, rep, steps=steps, refresh=refresh, n_shuf=n_shuf,
              families=families)
    arms = [("NONE", 0.0)] + [(f, float(k)) for f, (_, ladder, _) in T.families.items() for k in ladder]
    for m in targets:
        if ("EVEN", float(m)) not in arms:
            arms.append(("EVEN", float(m)))
    gains, means = T.run(arms, log)
    K_none = gains[0]
    S = T.strength(gains, K_none)
    ladder = [dict(family=f, knob=k, S=float(s), K=g.tolist(), chunk_means=mm.tolist())
              for (f, k), s, g, mm in zip(arms, S, gains, means)]
    target_S = {f"m={m}": float(S[arms.index(("EVEN", float(m)))]) for m in targets}
    by = lambda fam: [(r["knob"], r["S"], np.asarray(r["K"])) for r in ladder if r["family"] == fam]
    evaluated = {(r["family"], r["knob"]): np.asarray(r["K"]) for r in ladder}
    matches, rounds = {}, 0
    if match:
        matched = [f for f in T.families if f != "EVEN"]
        ladders = {f: ([0.0] + [k for k, _, _ in by(f)], [0.0] + [s for _, s, _ in by(f)]) for f in matched}
        def evaluate(proposals):
            pad = 1 << (len(proposals) - 1).bit_length()  # next power of two: at most log2(pairs) + 1 programs
            batch = list(proposals) + [("NONE", 0.0)] * (pad - len(proposals))
            g, _ = T.run(batch, None)
            for (f, k), gi in zip(proposals, g):
                evaluated[(f, float(k))] = gi
            return [float(T.strength(gi, K_none)) for gi in g[:len(proposals)]]

        states, rounds = M.match_parallel(ladders, target_S, evaluate, tol, max_rounds, log=log)
        for (f, tag), st in states.items():
            rec = M.public(st)
            rec["K"] = (evaluated[(f, rec["knob"])].tolist() if rec["knob"] is not None and rec["knob"] > 0 else
                        (K_none.tolist() if rec["knob"] == 0.0 else None))
            matches[f"{f}@{tag}"] = dict(rec, family=f, target_tag=tag)
    record = dict(kind="glq-tierA-blind", rep=rep, settings=asdict(settings), args=asdict(T.args),
                  actor_class=T.kind.name, steps=steps, refresh=refresh, n_shuf=n_shuf, targets=target_S,
                  target_m=list(targets), tol=tol, max_rounds=max_rounds, rounds=rounds, start_fit=T.start_fit,
                  K_bc=T.K_bc.tolist(), K_start=T.K_start.tolist(), critic_sha256=tree_sha256(critic_params),
                  rows_sha256=G.array_digest(np.asarray(T.rows)), spec_sha256=G.spec_digest(world.spec),
                  data_meta=data["meta"], families={f: dict(variable=v, ladder=list(l), form=fm)
                                                    for f, (v, l, fm) in T.families.items()},
                  ladder=ladder, matches=matches, oracle_unstable_refreshes=T.unstable_refreshes,
                  glq_match_sha256=M.source_sha256(), seconds=time.perf_counter() - started)
    record["blind_sha256"] = hashlib.sha256(json.dumps(G._clean(record), sort_keys=True,
                                                       default=G._json_default).encode()).hexdigest()
    return record, T


def tier_a_score(world, blind):
    """Tier A's scoring phase, after the blind record is fixed: exact J and NS per arm, paired differences against
    EVEN at each target, best over each ladder, and the null SHUF-vs-SHUF pairs."""
    score = lambda K: world.score(np.asarray(K))
    ladder = [dict(family=r["family"], knob=r["knob"], S=r["S"], **score(r["K"])) for r in blind["ladder"]]
    even = {f"m={m}": next(r for r in ladder if r["family"] == "EVEN" and r["knob"] == float(m))
            for m in blind["target_m"]}
    matched = {}
    for name, rec in blind["matches"].items():
        sc = score(rec["K"]) if rec["K"] is not None else None
        ref = even[rec["target_tag"]]
        matched[name] = dict(family=rec["family"], target_tag=rec["target_tag"], knob=rec["knob"], S=rec["S"],
                             relative_error=rec["relative_error"], converged=rec["converged"],
                             reachable=rec["reachable"], **(sc or {}),
                             NS_minus_EVEN=(sc["NS_mix"] - ref["NS_mix"]) if sc else None,
                             NS_minus_EVEN_per_context=([None if a is None or b is None else a - b for a, b in
                                                         zip(sc["NS"], ref["NS"])] if sc else None))
    none = next(r for r in ladder if r["family"] == "NONE")
    best = {}
    for fam in sorted({r["family"] for r in ladder}):
        rows = [r for r in ladder if r["family"] == fam]
        top = max(rows, key=lambda r: r["NS_mix"])
        best[fam] = dict(knob=top["knob"], NS_mix=top["NS_mix"], S=top["S"])
    shuf = sorted({r["family"] for r in ladder if r["family"].startswith("SHUF")})
    nulls = {}
    for tag in even:
        vals = {f: matched.get(f"{f}@{tag}", {}).get("NS_mix") for f in shuf}
        nulls[tag] = [dict(pair=[a, b], NS_difference=(vals[a] - vals[b]) if vals[a] is not None and
                           vals[b] is not None else None) for i, a in enumerate(shuf) for b in shuf[i + 1:]]
    return dict(kind="glq-tierA-score", rep=blind["rep"], blind_sha256=blind["blind_sha256"], ladder=ladder,
                none=dict(NS_mix=none["NS_mix"], NS=none["NS"]), even_at_target={k: v["NS_mix"] for k, v in
                                                                                  even.items()},
                matched=matched, best_over_ladder=best,
                best_OR_thin_minus_best_EVEN=best["OR-thin"]["NS_mix"] - best["EVEN"]["NS_mix"]
                if "OR-thin" in best and "EVEN" in best else None,
                best_OR_wrong_minus_NONE=best["OR-wrong"]["NS_mix"] - none["NS_mix"] if "OR-wrong" in best else None,
                best_capped_minus_best_EVEN={f: best[f]["NS_mix"] - best["EVEN"]["NS_mix"] for f in ("CAP-wrong",
                                                                                                     "CAP-err")
                                             if f in best and "EVEN" in best},
                null_shuf_pairs=nulls)


def tier_a(world, data, critic_params, settings, rep, out_dir=None, none_path=True, log=print, **kwargs):
    """Both phases: the blind record (written, with its SHA-256, before any return is computed), then the scores."""
    blind, T = tier_a_train(world, data, critic_params, settings, rep, log=log, **kwargs)
    if none_path:
        blind["none_path_K"] = T.none_path().tolist()
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        blind["file_sha256"] = G._write(Path(out_dir) / "tierA_blind.json", G._clean(blind))
    scores = tier_a_score(world, blind)
    if none_path:
        scores["none_path"] = dict(world.score(np.asarray(blind["none_path_K"])))
        scores["none_path_minus_none_weighted_NS"] = scores["none_path"]["NS_mix"] - scores["none"]["NS_mix"]
    if out_dir is not None:
        scores["file_sha256"] = G._write(Path(out_dir) / "tierA_score.json", G._clean(scores))
    return blind, scores


# ---------------------------------------------------------------------------------------------------------
# Level-1 screen (g5): first order, no training


def grad_j(world, K, step=1e-6):
    """Exact gradient of the mixture's 100-step J with respect to the per-context gains (central differences)."""
    K = G.World.per_context(K)
    return np.stack([world.mix[c] * G.finite_gradient(world.systems[c], K[c], step) for c in range(NC)])


def _positive_root(u, v, m):
    """b >= 0 with |u + b v| = m |u| (m >= 1); None when v = 0."""
    vv, uv, uu = float(v @ v), float(u @ v), float(u @ u)
    if vv == 0:
        return None
    disc = uv * uv + vv * uu * (m * m - 1)
    return float((-uv + math.sqrt(max(disc, 0.0))) / vv)


def level1_screen(world, data, critic_params, K_host, settings, rep=0, *, levels=TARGET_M, n_shuf=N_SHUF):
    """The first-order screen at the host actor (module docstring). context_linear class only."""
    if settings.actor != "context_linear":
        raise NotImplementedError("the Level-1 screen uses placement_harness.terms with params {'K': (5, 4, 10)}")
    T = TierA(world, data, critic_params, settings, rep, steps=1, refresh=1, n_shuf=n_shuf)
    args, models = T.args, (PH.ContextLinearActor(), _CRITIC)
    K_host = G.World.per_context(K_host)
    obs, action = T.obs32, T.a32
    t = lambda w: PH.terms(args, models, critic_params, K_host, obs, action, bc_w=w)
    t_none = t(np.ones(len(obs)))
    oracle = T.oracle_variables(K_host)
    variables = sorted({v for v, _, _ in T.families.values() if v is not None and v != "err_ratio"})
    gx = {v: t(T.variable(v, oracle))["g_BC"].ravel() for v in variables}
    u, gq = t_none["g_BC"].ravel(), t_none["g_Q"].ravel()
    gJ = grad_j(world, K_host).ravel()
    rho_host = float(np.linalg.norm(u) / np.linalg.norm(gq))
    out = dict(rho_host=rho_host, norm_g_Q=float(np.linalg.norm(gq)), norm_g_BC=float(np.linalg.norm(u)),
               cos_gQ_gBC=LQ.cosine(gq, u), norm_grad_J=float(np.linalg.norm(gJ)), lam=t_none["lam"], levels={})
    for m in levels:
        rows = {}
        dg_even = (m - 1) * u
        rows["EVEN"] = dict(knob=m, dg=dg_even)
        for fam, (v, _, form) in T.families.items():
            if v is None or v == "err_ratio" or form != "linear" or fam.startswith("CAP"):
                continue
            b = _positive_root(u, gx[v], m)
            rows[fam] = dict(knob=b, dg=None if b is None else b * gx[v])
        level = {}
        gain_even = float(-(gJ @ dg_even))
        for fam, r in rows.items():
            dg = r["dg"]
            if dg is None:
                level[fam] = dict(knob=None)
                continue
            gain = float(-(gJ @ dg))
            level[fam] = dict(knob=r["knob"], norm_dg=float(np.linalg.norm(dg)), cos_minus_dg_gradJ=LQ.cosine(-dg, gJ),
                              first_order_gain=gain, minus_EVEN=gain - gain_even,
                              rho=float(np.linalg.norm(u + dg) / np.linalg.norm(gq)) if fam != "EVEN" else m * rho_host)
        out["levels"][f"m={m}"] = level
    return out


# ---------------------------------------------------------------------------------------------------------
# Timing pilot (outcome-blind)


def _timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    jax.block_until_ready([x for x in jax.tree_util.tree_leaves(out) if isinstance(x, jax.Array)])
    return out, time.perf_counter() - t


def pilot(output, budget_seconds=900, log=print):
    """Timing pilot: times only (no error, return or NS is recorded). Writes pilot.json."""
    started = time.perf_counter()
    world = G.World()
    settings = TrainSettings()
    data, t_gen = _timed(G.generate, world, 0)
    out = dict(kind="glq-timing-pilot", host=platform.node(), cpu_count=os.cpu_count(),
               affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
               env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS")},
               jax=jax.__version__, seconds_generate=t_gen, timings={})
    tm = out["timings"]

    def cotrain_time(mode, warm, timed):
        rec = cotrain(world, data, replace(settings, snapshots=(1.0,)), mode, 0, updates=warm, diagnostics=False,
                      log=None)
        t0 = time.perf_counter()
        rec2 = cotrain(world, data, replace(settings, snapshots=(1.0,)), mode, 0, updates=timed, diagnostics=False,
                       log=None)
        dt = time.perf_counter() - t0
        return dict(ms_per_update=1000 * dt / timed, updates=timed, first_call_seconds=rec["seconds"]), rec2

    for mode, warm, timed in (("HOST", 1000, 20000), ("FQE", 1000, 5000), ("EXACT", 1000, 5000)):
        tm[mode], rec = cotrain_time(mode, warm, timed)
        log(f"{mode}: {tm[mode]['ms_per_update']:.2f} ms/update")
    state = rec["state"]
    tmp = Path(output) / "pilot_snapshot.msgpack"
    Path(output).mkdir(parents=True, exist_ok=True)
    _, tm["snapshot_save_seconds"] = _timed(save_snapshot, tmp, state)
    _, tm["snapshot_load_seconds"] = _timed(load_snapshot, tmp, state)
    tmp.unlink()
    K = ContextLinear().gains(state.actor.params)
    error_rows(world, state.critic.params, K, data["eval"])  # compile
    _, tm["diagnostics_eval_rows_seconds"] = _timed(error_rows, world, state.critic.params, K, data["eval"])
    T = TierA(world, data, state.critic.params, settings, 0, steps=1000, refresh=1000)
    _, tm["oracle_refresh_one_arm_seconds"] = _timed(T.oracle_variables, K)
    tm["tier_a"] = {}
    for n in (1, 16, 64):
        arms = [("NONE", 0.0)] * n
        T.run(arms)  # compile
        _, dt = _timed(T.run, arms)
        tm["tier_a"][str(n)] = dict(ms_per_step=dt, ms_per_arm_step=dt / n)  # 1000 steps: seconds == ms per step
        log(f"tier A {n} arms: {dt:.2f} ms/step, {dt / n:.3f} ms/arm-step")
        if time.perf_counter() - started > budget_seconds:
            break
    T2 = TierA(world, data, state.critic.params, settings, 0, steps=2000, refresh=1000)
    arms = [("OR-err", float(b)) for b in B_LADDER] + [("OR-D", 1.0), ("CAP-err", 1.0), ("NONE", 0.0)]
    T2.run(arms)  # compile (and one refresh per refreshed arm)
    T2._oracle_cache = {}
    _, dt = _timed(T2.run, arms)
    tm["tier_a_refreshed_8_arms_2000_steps_seconds"] = dt
    log(f"tier A, 8 arms with 7 refreshed, 2000 steps: {dt:.1f} s")
    _, tm["level1_screen_seconds"] = _timed(level1_screen, world, data, state.critic.params, K, settings)
    ev = data["eval"]
    obs, act = np.asarray(ev["obs"], np.float32), np.asarray(ev["action"], np.float32)
    fwd = jax.jit(lambda p, o, a: _CRITIC.apply(p, o, a).min(-1))
    fwd(state.critic.params, obs, act).block_until_ready()
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        fwd(state.critic.params, obs, act).block_until_ready()
    tm["critic_forward_rows_per_second"] = reps * len(obs) / (time.perf_counter() - t0)
    out["extrapolation"] = extrapolate(tm)
    out["seconds"] = time.perf_counter() - started
    G._write(Path(output) / "pilot.json", G._clean(out))
    return out


def pilot_full(output, log=print):
    """Pilot part 2 (outcome-blind, times only): one full-length HOST run (100k updates, snapshots written to a
    temporary directory and deleted, no diagnostics) and the full Tier A ladder program (every family's ladder + NONE,
    vmapped) for 2,000 steps with one oracle refresh. Writes pilot_full.json."""
    import shutil
    started = time.perf_counter()
    world, settings = G.World(), TrainSettings()
    data = G.generate(world, 0)
    out = dict(kind="glq-timing-pilot-full", affinity=sorted(os.sched_getaffinity(0)) if hasattr(
        os, "sched_getaffinity") else None, env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "JAX_PLATFORMS")},
               timings={})
    tm = out["timings"]
    tmp = Path(output) / "pilot_full_tmp"
    rec = cotrain(world, data, settings, "HOST", 0, out_dir=tmp, diagnostics=False, log=None)
    tm["host_100k_seconds"] = rec["seconds"]
    tm["host_100k_ms_per_update_incl_compile"] = 1000 * rec["seconds"] / settings.updates
    log(f"HOST 100k updates: {rec['seconds']:.0f} s")
    shutil.rmtree(tmp)
    critic = rec["states"][settings.updates // 2].critic.params
    T = TierA(world, data, critic, settings, 0, steps=2000, refresh=1000)
    arms = [("NONE", 0.0)] + [(f, float(k)) for f, (_, ladder, _) in T.families.items() for k in ladder]
    _, dt_first = _timed(T.run, arms)
    T._oracle_cache = {}
    _, dt = _timed(T.run, arms)
    tm["tier_a_ladder_arms"] = len(arms)
    tm["tier_a_ladder_2000_steps_first_call_seconds"] = dt_first
    tm["tier_a_ladder_2000_steps_seconds"] = dt
    tm["tier_a_ladder_projected_10k_steps_hours"] = dt * TIER_A_STEPS / 2000 / 3600
    log(f"tier A ladder, {len(arms)} arms, 2000 steps with one refresh: {dt:.1f} s")
    out["seconds"] = time.perf_counter() - started
    G._write(Path(output) / "pilot_full.json", G._clean(out))
    return out


def extrapolate(tm, updates=100_000, rung_seeds=3, rungs=4, tier_a_steps=TIER_A_STEPS, n_shuf=N_SHUF,
                match_rounds=3, tier_b_arms=7, band_draws=256, eval_rows=30_000):
    """The gate's compute in process-hours at the pilot's measured speeds and thread count.

    Per seed: HOST, FQE and EXACT co-trained runs (R2, R3; the HOST 50% snapshot is Tier A's critic), exact diagnostics
    at 3 snapshots of 3 runs, R5's population band (band_draws target-critic forwards per evaluation row at 3
    snapshots), the Level-1 screen, Tier A's ladders (every family's 5 knobs + NONE, vmapped; refreshed oracle arms
    add one oracle refresh per 1,000 steps) and match_rounds matching rounds over every (matched family, target) pair.
    The density ladder (g2) adds HOST + FQE for `rungs` other rungs on rung_seeds seeds. The expert-rich null world
    adds a HOST run and a Tier A per seed; the global-linear class a Tier A per seed (on the host critic). Lean: 5
    seeds, Tier A only. Full: 10 seeds plus Tier B (tier_b_arms co-trained arms per seed). Not included: the alpha
    ladder's retries (reported separately as a worst case), the ideal-regressor fits of R5 (g3) and any BCA work."""
    hours = lambda ms, n: ms * n / 3.6e6
    host, fqe, exact = (hours(tm[k]["ms_per_update"], updates) for k in ("HOST", "FQE", "EXACT"))
    fams = family_ladders(n_shuf)
    n_ladder = 1 + sum(len(l) for _, l, _ in fams.values())
    n_refreshed = sum(len(l) for v, l, _ in fams.values() if v in REFRESHED)
    n_pairs = (len(fams) - 1) * len(TARGET_M)
    arm_ms = {int(k): v["ms_per_arm_step"] for k, v in tm["tier_a"].items()}
    big, small = arm_ms[max(arm_ms)], arm_ms[min((k for k in arm_ms if k >= 8), default=max(arm_ms))]
    refresh_h = tm["oracle_refresh_one_arm_seconds"] * (tier_a_steps // ORACLE_REFRESH - 1) / 3600
    ladder = hours(big, n_ladder * tier_a_steps) + n_refreshed * refresh_h
    share_refreshed = sum(1 for f, (v, _, _) in fams.items() if f != "EVEN" and v in REFRESHED) / (len(fams) - 1)
    matching = match_rounds * n_pairs * (hours(small, tier_a_steps) + share_refreshed * refresh_h)
    tier_a_h = ladder + matching
    diag = 9 * tm["diagnostics_eval_rows_seconds"] / 3600
    band = 3 * band_draws * eval_rows / tm["critic_forward_rows_per_second"] / 3600
    screen = tm["level1_screen_seconds"] / 3600
    per_seed = dict(HOST=host, FQE=fqe, EXACT=exact, diagnostics=diag, r5_band=band, level1_screen=screen,
                    tier_a_ladder=ladder, tier_a_matching=matching, tier_a=tier_a_h)

    # the null world and the global-linear class need only EVEN, OR-wrong and OR-thin (their marks compare OR-wrong
    # with EVEN): 16 ladder arms and 4 matched pairs
    control_h = hours(big, 16 * tier_a_steps) + match_rounds * 4 * hours(arm_ms[min(arm_ms)], tier_a_steps)
    per_seed["tier_a_controls_reduced"] = control_h

    def path(seeds, tier_b, controls=tier_a_h):
        out = dict(seeds=seeds, critic_runs=seeds * (host + fqe + exact + diag + band + screen),
                   density_ladder=rung_seeds * rungs * (host + fqe), tier_a=seeds * tier_a_h,
                   null_world=seeds * (host + controls), global_linear=seeds * controls,
                   tier_b=seeds * tier_b_arms * host if tier_b else 0.0)
        out["total_hours"] = sum(v for k, v in out.items() if k != "seeds")
        out["alpha_ladder_worst_extra"] = 2 * seeds * (host + fqe + exact)
        return out

    return dict(hours_per_seed=per_seed, n_ladder_arms=n_ladder, n_refreshed_arms=n_refreshed,
                n_match_pairs=n_pairs, lean=path(5, False), full=path(10, True),
                lean_reduced_controls=path(5, False, control_h), full_reduced_controls=path(10, True, control_h),
                note="process-hours at the pilot's thread count (wall hours for one process at a time)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pilot", help="outcome-blind timing pilot; writes pilot.json")
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--part", choices=("1", "full"), default="1",
                   help="1: per-update and per-step times; full: a 100k-update HOST run and the full Tier A ladder")
    ns = parser.parse_args(argv)
    if ns.part == "full":
        print(json.dumps(pilot_full(ns.output)["timings"], indent=1))
    else:
        out = pilot(ns.output, ns.seconds)
        print(json.dumps(out["extrapolation"], indent=1))


if __name__ == "__main__":
    main()
