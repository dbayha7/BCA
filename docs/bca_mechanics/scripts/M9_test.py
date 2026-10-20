"""M9 test: the lower edge, the penalty W = [U(s, pi(s)) - U(s, a_i)]_+ in the actor's objective (k = 1).

What this script does (module map: docs/bca_mechanics/MODULES.md, section M9; write-up: docs/bca_mechanics/modules/M9.md)
  Part S  Synthetic tests with a known truth (JAX, float32, the project's error-size network class
          calibration.network.Calibrator when it imports, else a re-typed copy of it):
          S1 arithmetic [M9.1-M9.4, M9.6]: U at the actor's command is R * u_f * max(eta_f, 1e-6); the paired
             penalty is never negative and is exactly 0 at the recorded command; a per-row threshold scales the
             penalty row by row, W(R_i) = (R_i / R_u) W(R_u); with k = 0 the loss and its gradient are the plain
             actor loss's, bit for bit.
          S2 gradient path [M9.1]: the penalty's gradient reaches the command only through eta_f's command input;
             with those input weights set to 0 the gradient is exactly 0 while Q1's is unchanged.
          S3 refusal [M9.5]: the weighting test's rule (a row whose threshold is +infinity takes the largest finite
             threshold of its table) on a hand-built table, and Stage 2's rule (U = U_max everywhere), under which the
             paired penalty of a refused standard band is 0 at every command: no caution at all.
          S4 the mechanism with a known truth: one throttle, a critic that keeps rising past the 80% knee where the
             truth falls, a band whose slope past the knee is a set fraction of the critic's; the actor loss of M9.4 is
             minimized exactly and by gradient descent; the actor stays at the knee only when the band's slope is of
             the critic's order (ratio about 0.85 or more at k = 1); at the ratio measured on the records (about
             0.01) it runs to full throttle.
  Part R  Real-data tests on the recorded ThrusterChain files (nothing trained, no band fitted):
          R1 the width recompute in the pinned environment is quoted from data/M9/width_recompute_pinned.out
             (docs/bca_mechanics/scripts/M9_width_recompute.py through M9_width_recompute.sh).
          R2 Stage 2's detection pairs (seeds 0-4, stalling world, fixed critic): BCA's penalty at the no-caution
             actor's command against the critic's over-preference measured by exact simulation, for the standard
             band and the classifier-weighted band: effective k (ratio of means), Spearman correlation, share of
             harmful commands penalized, and the band's slope relative to the critic's (U(cmd) - U(cap) over
             Q1(cmd) - Q1(cap)). Needs --runs or --copy; skipped otherwise.
          R3 from the records (data/M9/M9_records.json, written by M9_records.py): the penalty term during training
             against the value term and the cloning term; Stage 2's recorded penalty shape past 80%; the weighting
             test's proxy check (BCA's penalty against the exact one-step oracle) and its L1-L3 marks.
          R4 from the records: the k = 0 plumbing check on every seed-world; the trained actors' strength and points
             in strong current (near-80, standard band; Stage 2, weighted band) against no caution and the rivals.
  Figure  docs/bca_mechanics/figures/M9.svg (three panels).

How to run (Windows venv used for the numbers in M9.md: Python 3.14, numpy 2.5.4, scipy 1.18.1, jax 0.12.0,
flax 0.12.10, matplotlib 3.11.2; under a minute):
    set PYTHONPATH=M:\\  (for calibration.network; optional)
    python docs/bca_mechanics/scripts/M9_test.py --runs \\\\wsl.localhost\\Ubuntu-24.04\\home\\user\\bca\\runs\\shift_explore
  --runs <root>  holds detect/seed<K>/stall/{pairs,labels}.npz and
                 stage2/detect/{uniform,weighted}/stall/fixed/seed<K>/scores/bca_width_<ver>_fixed.npz (read only)
  --copy <dir>   the same files as pairs/seed<K>/{pairs,labels}.npz and {uniform,weighted}/stall/seed<K>/scores/...
  --records <json>  default docs/bca_mechanics/data/M9/M9_records.json
  Without --runs or --copy, Part R2 is skipped and the figure's recorded values for it are read from --r2 (the JSON
  this script writes to data/M9/M9_r2.json when R2 runs).
The pinned environment (Python 3.10, numpy 2.2.6, jax 0.6.2) is needed only for M9_width_recompute.py and the
module's own unit tests (M9_unit_tests.sh); the synthetic parts here use random networks and need no exact match.
Lines marked [M9.x] carry the module map's labels.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(DOCS))
FIG = os.path.join(DOCS, "figures", "M9.svg")
DATA = os.path.join(DOCS, "data", "M9")
WSL = r"\\wsl.localhost\Ubuntu-24.04\home\user\bca\runs\shift_explore"
for p in (REPO,):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402

jax.config.update("jax_platforms", "cpu") if hasattr(jax.config, "update") else None
DX, DA, EPS = 13, 4, 1e-6
ALPHA_STAGE2 = 5.0              # stage 2 and near-80 (the weighting test used 2.5)
KNEE = 0.8


def banner(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


# ----------------------------------------------------------------------------------------------------------------
# The error-size network eta (M2), the width (M9.1), the paired penalty (M9.3) and the actor loss (M9.4)
# ----------------------------------------------------------------------------------------------------------------
def make_calibrator():
    """The project's Calibrator (calibration/network.py:8-32) as the gate builds it, else a re-typed copy."""
    try:
        from calibration.network import Calibrator          # the project's class
        return Calibrator(jnp.zeros(DX), jnp.ones(DX), state_dep=True), "calibration.network.Calibrator"
    except Exception as e:                                   # noqa: BLE001
        import flax.linen as nn

        class Calibrator(nn.Module):                         # re-typed from calibration/network.py:8-32
            obs_mean: jax.Array
            obs_std: jax.Array
            state_dep: bool = True
            hidden: int = 64
            depth: int = 2

            @nn.compact
            def __call__(self, obs=None, action=None):
                obs_n = (obs - self.obs_mean) / (self.obs_std + 0.001)
                x = jnp.concatenate([obs_n, action], axis=-1)
                for _ in range(self.depth):
                    x = nn.relu(nn.Dense(self.hidden)(x))
                log_eta = nn.Dense(1)(x).squeeze(-1)
                return jax.nn.softplus(log_eta) + 0.01
        return Calibrator(jnp.zeros(DX), jnp.ones(DX), state_dep=True), f"re-typed copy ({type(e).__name__})"


def band_width(cal, cal_params, unit, obs, a, R):
    """[M9.1] U = R x (max(eta_f(s, a), 1e-6) x u_f), float32, eta_f's parameters frozen
    (experiments/shift_iw/lower_edge.py:196-201; stage2/code/band.py:146-155 associates it as (R u_f) e)."""
    eta = cal.apply(jax.lax.stop_gradient(cal_params), obs, a)
    return jax.lax.stop_gradient(R) * (jnp.maximum(eta, EPS) * jax.lax.stop_gradient(unit))


def paired_penalty(cal, cal_params, unit, obs, a, R, u_rec):
    """[M9.2-M9.3] W = [U(s, a) - U(s, a_i)]_+ with U(s, a_i) a constant per row (lower_edge.py:204-208)."""
    return jnp.maximum(band_width(cal, cal_params, unit, obs, a, R) - jax.lax.stop_gradient(u_rec), 0.0)


def critic_init(key, hidden=32):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(w1=jax.random.normal(k1, (DX + DA, hidden)) * 0.3, b1=jnp.zeros(hidden),
                w2=jax.random.normal(k2, (hidden, hidden)) * 0.3, b2=jnp.zeros(hidden),
                w3=jax.random.normal(k3, (hidden, 1)) * 0.3)


def critic_q1(cp, obs, a):
    x = jnp.concatenate([obs, a], -1)
    x = jnp.tanh(x @ cp["w1"] + cp["b1"])
    x = jnp.tanh(x @ cp["w2"] + cp["b2"])
    return (x @ cp["w3"])[..., 0] * 10.0 - 15.0           # values of the recorded order (Q1 mean about -15)


def actor_loss(pi, cp, obs, a_rec, alpha, k, penalty):
    """[M9.4] -lambda mean_i[Q1(s_i, pi_i) - k W_i] + mean_i mean_j (pi_ij - a_ij)^2, lambda = alpha / mean|Q1|
    held constant (lower_edge.py:293-312; stage2/code/policy.py:83-93 with pull 1)."""
    q = critic_q1(cp, obs, pi)
    lmbda = alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
    w = penalty(pi).astype(q.dtype)
    kw = k * w
    bc = jnp.square(pi - a_rec).mean(axis=-1).mean()
    return -lmbda * (q - kw).mean() + bc, (q.mean(), lmbda, bc, w.mean())


def plain_actor_loss(pi, cp, obs, a_rec, alpha):
    """TD3+BC's actor loss with no penalty (algorithms/td3_bc.py's expression, re-typed)."""
    q = critic_q1(cp, obs, pi)
    lmbda = alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
    return -lmbda * q.mean() + jnp.square(pi - a_rec).mean(axis=-1).mean()


def capped_thresholds(R_rows, finite_pool):
    """[M9.5, weighting test] re-typed from experiments/shift_iw/lower_edge.py:211-224."""
    R = np.asarray(R_rows, np.float64)
    if np.any(np.isnan(R)) or np.any(R < 0):
        raise ValueError("thresholds must be non-negative numbers or +inf")
    pool = np.asarray(finite_pool, np.float64).ravel()
    pool = pool[np.isfinite(pool)]
    if not pool.size:
        raise ValueError("the band certifies nothing here: no finite threshold to read a width from")
    cap = float(pool.max())
    refused = np.isinf(R)
    return np.where(refused, cap, R), cap, refused


def width_stage2(cal, cal_params, unit, R, R_c, U_max, certified, obs, a):
    """[M9.5, stage 2 / near-80] re-typed from stage2/code/band.py:146-155: U_max where the band cannot certify."""
    e = jnp.maximum(cal.apply(jax.lax.stop_gradient(cal_params), obs, a), EPS)
    R_safe = jnp.where(certified, R, R_c)
    return jnp.where(certified, R_safe * unit * e, U_max * jnp.ones_like(e))


# ----------------------------------------------------------------------------------------------------------------
# Part S: synthetic tests
# ----------------------------------------------------------------------------------------------------------------
def part_s(results):
    banner("PART S  synthetic tests with a known truth (random networks, float32)")
    cal, which = make_calibrator()
    print("error-size network:", which)
    key = jax.random.PRNGKey(20261020)
    k_cal, k_cr, k_obs, k_act, k_pi, k_R = jax.random.split(key, 6)
    n = 512
    obs = jax.random.normal(k_obs, (n, DX), jnp.float32)
    a_rec = jnp.clip(jax.random.normal(k_act, (n, DA), jnp.float32) * 0.4, -1, 1)
    pi = jnp.clip(a_rec + 0.3 * jax.random.normal(k_pi, (n, DA), jnp.float32), -1, 1)   # the actor's commands
    cal_params = cal.init(k_cal, obs[:1], a_rec[:1])
    cp = critic_init(k_cr)
    R_u, unit = jnp.float32(1.5), jnp.float32(0.17)       # a threshold and a unit of the recorded order
    R_rows = jnp.full((n,), R_u, jnp.float32)
    u_rec = band_width(cal, cal_params, unit, obs, a_rec, R_rows)                         # [M9.2]
    U_pi = band_width(cal, cal_params, unit, obs, pi, R_rows)                              # [M9.1]
    W = paired_penalty(cal, cal_params, unit, obs, pi, R_rows, u_rec)                      # [M9.3]
    eta_pi = cal.apply(cal_params, obs, pi)
    U_retyped = (R_u * (jnp.maximum(eta_pi, EPS) * unit)).astype(jnp.float32)

    print("\nS1  arithmetic [M9.1-M9.4, M9.6]")
    s1 = {}
    s1["width_equals_formula"] = bool(jnp.array_equal(U_pi, U_retyped))
    s1["penalty_nonnegative"] = bool(jnp.all(W >= 0))
    W_at_rec = paired_penalty(cal, cal_params, unit, obs, a_rec, R_rows, u_rec)
    s1["penalty_zero_at_recorded"] = bool(jnp.all(W_at_rec == 0))
    s1["penalty_is_positive_part"] = bool(jnp.array_equal(W, jnp.maximum(U_pi - u_rec, 0)))
    f = jax.random.uniform(k_R, (n,), jnp.float32, 0.7, 2.5)                                 # per-row ratios R_i / R_u
    R_i = (R_u * f).astype(jnp.float32)
    u_rec_i = band_width(cal, cal_params, unit, obs, a_rec, R_i)
    W_i = paired_penalty(cal, cal_params, unit, obs, pi, R_i, u_rec_i)                     # [M9.6]
    err = np.abs(np.asarray(W_i) - np.asarray(f * W)) / float(np.mean(np.asarray(U_pi)))   # in units of the mean width
    s1["per_row_threshold_scales_penalty_max_err_over_mean_width"] = float(err.max())
    pen = lambda p: paired_penalty(cal, cal_params, unit, obs, p, R_rows, u_rec)          # noqa: E731
    loss0, aux0 = actor_loss(pi, cp, obs, a_rec, ALPHA_STAGE2, jnp.float32(0.0), pen)
    lossp = plain_actor_loss(pi, cp, obs, a_rec, ALPHA_STAGE2)
    g0 = jax.grad(lambda p: actor_loss(p, cp, obs, a_rec, ALPHA_STAGE2, jnp.float32(0.0), pen)[0])(pi)
    gp = jax.grad(lambda p: plain_actor_loss(p, cp, obs, a_rec, ALPHA_STAGE2))(pi)
    s1["k0_loss_equals_plain_bitwise"] = bool(loss0 == lossp)
    s1["k0_gradient_equals_plain_bitwise"] = bool(jnp.array_equal(g0, gp))
    loss1, aux1 = actor_loss(pi, cp, obs, a_rec, ALPHA_STAGE2, jnp.float32(1.0), pen)
    s1["k1_minus_k0_loss_equals_lambda_times_meanW_rel_err"] = float(abs((loss1 - loss0) - aux1[1] * aux1[3]) / max(float(abs(loss1 - loss0)), 1e-9))
    for k_, v in s1.items():
        print(f"   {k_:58s} {v}")
    ok1 = (s1["width_equals_formula"] and s1["penalty_nonnegative"] and s1["penalty_zero_at_recorded"] and
           s1["penalty_is_positive_part"] and s1["per_row_threshold_scales_penalty_max_err_over_mean_width"] < 1e-5 and
           s1["k0_loss_equals_plain_bitwise"] and s1["k0_gradient_equals_plain_bitwise"] and
           s1["k1_minus_k0_loss_equals_lambda_times_meanW_rel_err"] < 1e-4)
    print("   S1:", "PASS" if ok1 else "FAIL")

    print("\nS2  gradient path [M9.1]: the penalty reaches the command only through eta_f's command input")
    dW = jax.vmap(jax.grad(lambda a_, o_, r_, u_: paired_penalty(cal, cal_params, unit, o_[None], a_[None], r_[None],
                                                                 u_[None])[0]))(pi, obs, R_rows, u_rec)
    dQ = jax.vmap(jax.grad(lambda a_, o_: critic_q1(cp, o_[None], a_[None])[0]))(pi, obs)
    # zero the command-input rows of eta's first layer: the last DA rows of Dense_0's kernel
    import flax
    p2 = jax.tree_util.tree_map(lambda x: jnp.array(x), cal_params)       # a copy of eta_f's parameters
    p2 = flax.core.unfreeze(p2) if hasattr(flax.core, "unfreeze") else dict(p2)   # a plain, mutable dict
    kern = p2["params"]["Dense_0"]["kernel"]                               # (DX + DA, hidden): state rows, then command rows
    p2["params"]["Dense_0"]["kernel"] = kern.at[DX:].set(0.0)
    u_rec2 = band_width(cal, p2, unit, obs, a_rec, R_rows)
    dW0 = jax.vmap(jax.grad(lambda a_, o_, r_, u_: paired_penalty(cal, p2, unit, o_[None], a_[None], r_[None],
                                                                  u_[None])[0]))(pi, obs, R_rows, u_rec2)
    W2 = paired_penalty(cal, p2, unit, obs, pi, R_rows, u_rec2)
    s2 = dict(share_rows_with_nonzero_dW_da=float(np.mean(np.abs(np.asarray(dW)).sum(1) > 0)),
              rms_dW_da=float(np.sqrt(np.mean(np.asarray(dW) ** 2))), rms_dQ1_da=float(np.sqrt(np.mean(np.asarray(dQ) ** 2))),
              dW_da_with_command_input_zeroed_max_abs=float(np.max(np.abs(np.asarray(dW0)))),
              penalty_with_command_input_zeroed_max=float(np.max(np.asarray(W2))))
    for k_, v in s2.items():
        print(f"   {k_:58s} {v:.6g}")
    ok2 = s2["share_rows_with_nonzero_dW_da"] > 0.3 and s2["dW_da_with_command_input_zeroed_max_abs"] == 0.0 and s2["penalty_with_command_input_zeroed_max"] == 0.0
    print("   S2:", "PASS" if ok2 else "FAIL",
          "(with eta_f blind to the command, U(s, a) = U(s, a_i) for every a: the penalty and its gradient vanish)")

    print("\nS3  refusal [M9.5]")
    R_rows_np = np.array([1.2, np.inf, 0.9, np.inf])
    capped, cap, refused = capped_thresholds(R_rows_np, [1.2, 0.9, np.inf])
    s3 = dict(weighting_test_capped=capped.tolist(), cap=cap, refused=refused.tolist())
    try:
        capped_thresholds([np.inf], [np.inf])
        s3["all_infinite_refuses"] = False
    except ValueError:
        s3["all_infinite_refuses"] = True
    U_max = float(jnp.max(u_rec)) * 1.1
    Ur = width_stage2(cal, cal_params, unit, jnp.float32(np.inf), R_u, jnp.float32(U_max), jnp.asarray(False), obs, pi)
    Urec = width_stage2(cal, cal_params, unit, jnp.float32(np.inf), R_u, jnp.float32(U_max), jnp.asarray(False), obs, a_rec)
    Wr = jnp.maximum(Ur - Urec, 0.0)
    s3["stage2_refused_width_is_U_max_everywhere"] = bool(jnp.all(Ur == jnp.float32(U_max)))
    s3["stage2_refused_paired_penalty_max"] = float(jnp.max(Wr))
    Wc = paired_penalty(cal, cal_params, unit, obs, pi, jnp.full((n,), cap, jnp.float32),
                        band_width(cal, cal_params, unit, obs, a_rec, jnp.full((n,), cap, jnp.float32)))
    s3["weighting_test_refused_row_penalty_over_unweighted_median"] = float(np.median(np.asarray(Wc)[np.asarray(W) > 0] / np.asarray(W)[np.asarray(W) > 0]))
    s3["cap_over_R_u"] = cap / float(R_u)
    for k_, v in s3.items():
        print(f"   {k_:58s} {v}")
    ok3 = (capped.tolist() == [1.2, 1.2, 0.9, 1.2] and cap == 1.2 and refused.tolist() == [False, True, False, True]
           and s3["all_infinite_refuses"] and s3["stage2_refused_width_is_U_max_everywhere"]
           and s3["stage2_refused_paired_penalty_max"] == 0.0)
    print("   S3:", "PASS" if ok3 else "FAIL",
          "(stage 2 / near-80: a refused standard band gives a flat width U_max, so the paired penalty is 0 everywhere;"
          " the weighting test: the refused row's penalty is the unweighted one times cap / R_u)")

    print("\nS4  the mechanism with a known truth: one throttle, critic rising past the knee, band of a set slope")
    lam, s_Q, a_i, U0 = 0.32, 2.0, 0.75, 0.30     # lambda, the critic's slope and the recorded command of the recorded order
    ratios = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0, 1.5, 3.0]
    rows = []
    for k_ in (1.0, 2.0):
        for r_ in ratios:
            s_U = r_ * s_Q

            def loss(a, k_=k_, s_U=s_U):
                q = s_Q * a                                               # the critic keeps rising past the knee
                U = U0 + s_U * jnp.maximum(a - KNEE, 0.0)                 # the band: flat below the knee, slope s_U past it
                W = jnp.maximum(U - U0, 0.0)                              # paired against the recorded command (below the knee)
                return -lam * (q - k_ * W) + (a - a_i) ** 2
            a = jnp.float32(a_i)
            g = jax.jit(jax.grad(loss))
            for step in range(6000):                                     # plain gradient descent, step 0.01 then 0.0005
                a = jnp.clip(a - (0.01 if step < 3000 else 0.0005) * g(a), 0.0, 1.0)
            # closed form: past the knee dL/da = -lam (s_Q - k s_U) + 2 (a - a_i); below it the loss falls all the way
            # to the knee (a_i + lam s_Q / 2 > knee), so the minimizer is the interior point when it lies past the knee,
            # else the knee itself
            interior = a_i + lam * (s_Q - k_ * s_U) / 2
            closed = min(1.0, interior) if interior > KNEE else KNEE
            rows.append(dict(k=k_, slope_ratio=r_, final_throttle=float(a), closed_form=float(closed)))
    print(f"   lambda {lam}, critic slope {s_Q} per unit throttle, recorded command {a_i}, knee {KNEE}; the truth falls past the knee,"
          f" so the right answer is a throttle of at most {KNEE}")
    print(f"   {'k':>3s} {'band slope / critic slope':>26s} {'final throttle (descent)':>26s} {'closed form':>12s}  verdict")
    for r in rows:
        v = ("stays at the knee" if abs(r["final_throttle"] - KNEE) <= 0.01 else
             "held back" if r["final_throttle"] < 0.999 else "full throttle (harm)")
        print(f"   {r['k']:3.0f} {r['slope_ratio']:26.3f} {r['final_throttle']:26.3f} {r['closed_form']:12.3f}  {v}")
    need = (s_Q - 2 * (KNEE - a_i) / lam) / s_Q
    print(f"   the band's slope must be at least {need:.2f} x the critic's at k = 1 ({need / 2:.2f} x at k = 2) to hold the actor at the knee")
    results["S"] = dict(S1=s1, S2=s2, S3=s3, S4=dict(rows=rows, lam=lam, s_Q=s_Q, a_i=a_i, needed_ratio_k1=need),
                        pass_S1=ok1, pass_S2=ok2, pass_S3=ok3)


# ----------------------------------------------------------------------------------------------------------------
# Part R2: the recorded pairs
# ----------------------------------------------------------------------------------------------------------------
def r2_paths(args, seed, ver):
    if args.copy:
        return (os.path.join(args.copy, "pairs", f"seed{seed}", "pairs.npz"),
                os.path.join(args.copy, "pairs", f"seed{seed}", "labels.npz"),
                os.path.join(args.copy, ver, "stall", f"seed{seed}", "scores", f"bca_width_{ver}_fixed.npz"))
    r = args.runs
    return (os.path.join(r, "detect", f"seed{seed}", "stall", "pairs.npz"),
            os.path.join(r, "detect", f"seed{seed}", "stall", "labels.npz"),
            os.path.join(r, "stage2", "detect", ver, "stall", "fixed", f"seed{seed}", "scores", f"bca_width_{ver}_fixed.npz"))


def part_r2(args, results):
    from scipy.stats import spearmanr
    banner("PART R2  Stage 2's recorded pairs: BCA's penalty at the actor's command against the critic's over-preference")
    print("rows: T1 = 1,500 training states in the full-throttle region, T3 = 3,000 strong-current rollout states; per state the"
          " no-caution actor's command, that command capped at 80% throttle and (T1) the recorded command.")
    print("over-preference = [Q1(cmd) - Q1(cap)] - [true(cmd) - true(cap)] (labels.npz over_pref, exact simulation);"
          " harmful = the cap is truly better and the critic over-prefers the command (labels.npz harmful).")
    print("penalty (size form) = [U(cmd) - U(cap)]_+ ; penalty (paired, as trained) = [U(cmd) - U(recorded)]_+ (T1 only).")
    out = {}
    hdr = (f"{'band':9s} {'seed':4s} {'set':3s} {'harmful':>7s} {'mean W size':>11s} {'mean W pair':>11s} {'mean over-pref':>14s} "
           f"{'eff. k':>8s} {'W>0 share':>9s} {'Spearman':>8s} {'slope ratio':>11s} {'U(cmd)-U(rec), not harmful':>26s}")
    print(hdr)
    for ver in ("uniform", "weighted"):
        for seed in range(5):
            pp, lp, sp = r2_paths(args, seed, ver)
            pz, lz, sc = np.load(pp), np.load(lp), np.load(sp)["scores"]
            st, role, pair = pz["set"], pz["role"], pz["pair"]
            for S in ("T1", "T3"):
                mc, mk = (st == S) & (role == "command"), (st == S) & (role == "cap")
                oc, ok = np.argsort(pair[mc]), np.argsort(pair[mk])
                assert np.array_equal(pair[mc][oc], pair[mk][ok])
                Uc, Uk = sc[mc][oc], sc[mk][ok]
                harm = lz["harmful"][mc][oc]
                over = lz["over_pref"][mc][oc]
                dq = lz["critic_pref"][mc][oc]
                W_size = np.maximum(Uc - Uk, 0.0)
                rec = dict(harmful=int(harm.sum()), rows=int(mc.sum()),
                           mean_W_size_harmful=float(W_size[harm].mean()), mean_over_pref_harmful=float(over[harm].mean()),
                           eff_k=float(W_size[harm].mean() / over[harm].mean()),
                           share_W_positive_harmful=float(np.mean(W_size[harm] > 0)),
                           spearman_harmful=float(spearmanr(W_size[harm], over[harm])[0]),
                           spearman_all=float(spearmanr(W_size, over)[0]),
                           slope_ratio_median_harmful=float(np.median((Uc - Uk)[harm] / dq[harm])),
                           mean_dU_cap_not_harmful=float((Uc - Uk)[~harm].mean()))
                if S == "T1":
                    mr = (st == S) & (role == "recorded")
                    orr = np.argsort(pair[mr])
                    assert np.array_equal(pair[mc][oc], pair[mr][orr])
                    Ur = sc[mr][orr]
                    W_pair = np.maximum(Uc - Ur, 0.0)
                    rec.update(mean_W_pair_harmful=float(W_pair[harm].mean()), mean_W_pair_not_harmful=float(W_pair[~harm].mean()),
                               share_W_pair_positive_harmful=float(np.mean(W_pair[harm] > 0)),
                               share_W_pair_positive_not_harmful=float(np.mean(W_pair[~harm] > 0)),
                               eff_k_pair=float(W_pair[harm].mean() / over[harm].mean()),
                               spearman_pair_harmful=float(spearmanr(W_pair[harm], over[harm])[0]),
                               mean_dU_rec_not_harmful=float((Uc - Ur)[~harm].mean()))
                out[f"{ver}/{seed}/{S}"] = rec
                wp = f"{rec['mean_W_pair_harmful']:11.4f}" if S == "T1" else f"{'-':>11s}"
                nh = f"{rec['mean_dU_rec_not_harmful']:26.4f}" if S == "T1" else f"{'-':>26s}"
                print(f"{ver:9s} {seed:4d} {S:3s} {rec['harmful']:7d} {rec['mean_W_size_harmful']:11.4f} {wp} {rec['mean_over_pref_harmful']:14.3f} "
                      f"{rec['eff_k']:8.4f} {rec['share_W_positive_harmful']:9.3f} {rec['spearman_harmful']:8.3f} {rec['slope_ratio_median_harmful']:11.4f} {nh}")
    # summaries
    for ver in ("uniform", "weighted"):
        for S in ("T1", "T3"):
            ks = [out[f"{ver}/{s}/{S}"]["eff_k"] for s in range(5)]
            sr = [out[f"{ver}/{s}/{S}"]["slope_ratio_median_harmful"] for s in range(5)]
            rho = [out[f"{ver}/{s}/{S}"]["spearman_harmful"] for s in range(5)]
            print(f"   {ver:9s} {S}: effective k {min(ks):.4f}-{max(ks):.4f} (mean {np.mean(ks):.4f}); band slope / critic slope, median per seed "
                  f"{min(sr):.4f}-{max(sr):.4f}; Spearman {min(rho):.2f}-{max(rho):.2f}")
    results["R2"] = out
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "M9_r2.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("   written:", os.path.join(DATA, "M9_r2.json"))


# ----------------------------------------------------------------------------------------------------------------
# Parts R3, R4: from the records
# ----------------------------------------------------------------------------------------------------------------
def part_r3_r4(rec, results):
    banner("PART R3  from the records: the penalty during training, the penalty's shape, the proxy check")
    print("R3a  the three terms of the actor loss in the last 4,000-step chunk (chunk_metrics of the trained arm):"
          " value term lambda*|Q1|, cloning term, penalty term lambda*k*W")
    print(f"{'study':8s} {'band':9s} {'world':9s} {'seed':4s} {'lambda':>7s} {'|Q1| mean':>9s} {'cloning':>8s} {'k W mean':>9s} {'lambda k W':>10s} {'penalty / cloning':>17s} {'penalty / value':>15s}")
    r3a = {}
    for study, key, band in (("near-80", "near80", "standard"), ("stage 2", "stage2", "weighted")):
        for sw, e in rec[key]["seeds"].items():
            cm = e["chunk_metrics"][-1]
            loss, q, lam, bc, kw = cm
            r3a[f"{study}/{sw}"] = dict(lam=lam, q=q, bc=bc, kw=kw, pen=lam * kw, pen_over_bc=lam * kw / bc, pen_over_value=lam * kw / (lam * abs(q)))
            w, s = sw.split("/")
            print(f"{study:8s} {band:9s} {w:9s} {s:4s} {lam:7.3f} {abs(q):9.2f} {bc:8.4f} {kw:9.5f} {lam * kw:10.5f} {lam * kw / bc:17.4f} {lam * kw / (lam * abs(q)):15.6f}")
    for study in ("near-80", "stage 2"):
        v = [x for k_, x in r3a.items() if k_.startswith(study)]
        print(f"   {study}: penalty / cloning {min(x['pen_over_bc'] for x in v):.4f}-{max(x['pen_over_bc'] for x in v):.4f};"
              f" penalty / value term {min(x['pen_over_value'] for x in v):.6f}-{max(x['pen_over_value'] for x in v):.6f}")
    results["R3a"] = r3a

    print("\nR3b  Stage 2's recorded penalty shape (weighted band, the no-caution actor, 2,000 full-throttle-region training states,"
          " thruster 1 moved along the actor's push; lock.json bca_penalty_shape)")
    print(f"{'seed':4s} {'mean W at 80%':>13s} {'at 81%':>8s} {'at 100%':>8s} {'rises 80->81':>12s} {'falls 80->81':>12s} {'best past 80%: Q1 - W':>21s} {'Q1 alone':>9s}")
    r3b = {}
    for s in range(5):
        sh = rec["stage2"]["seeds"][f"stall/{s}"].get("penalty_shape")
        if not sh:
            continue
        mw = sh["mean_W"]
        r3b[s] = dict(w80=mw["0.8"], w81=mw["0.81"], w100=mw["1"], rise=sh["rise_share_80_to_81"], fall=sh["fall_share_80_to_81"],
                      best=sh["best_past80_share"], alone=sh["q1_alone_best_past80_share"])
        print(f"{s:4d} {mw['0.8']:13.3f} {mw['0.81']:8.3f} {mw['1']:8.3f} {sh['rise_share_80_to_81']:12.3f} {sh['fall_share_80_to_81']:12.3f} {sh['best_past80_share']:21.3f} {sh['q1_alone_best_past80_share']:9.3f}")
    print("   (no such record exists for the standard band in Stage 2: its lower-edge arms ran on the weighted band)")
    results["R3b"] = r3b

    print("\nR3c  the weighting test (standard band; report.json marks L1, L2 with the proxy check, L3), five qualities x seeds 0-4")
    print(f"{'quality':14s} {'L1 oracle - none':>16s} {'L1':>5s} {'L2 predicted':>12s} {'L2':>5s} {'eff. k (band/oracle)':>21s} {'Pearson':>13s} {'Spearman':>13s} {'check (e) on BCA':>17s} {'L3':>8s}")
    r3c = {}
    for q, e in rec["weighting_test"]["qualities"].items():
        pc = e["L2"]["proxy_check"]
        k = pc["band_over_oracle_per_seed"]
        pe, sp = pc["pearson_per_seed"], pc["spearman_per_seed"]
        ce = pc["band_check_e_passed"]
        r3c[q] = dict(L1=e["L1"]["status"], L1_mean=e["L1"]["oracle_minus_none_mean"], L2=e["L2"]["status"], L2_pred=e["L2"]["predicted_return"],
                      eff_k=k, pearson=pe, spearman=sp, check_e=ce, L3=e["L3"]["status"],
                      band_mean=pc["band_mean"], oracle_mean=pc["oracle_mean"])
        print(f"{q:14s} {e['L1']['oracle_minus_none_mean']:16.1f} {e['L1']['status']:>5s} {e['L2']['predicted_return']:12.2f} {e['L2']['status']:>5s} "
              f"{min(k):10.4f}-{max(k):.4f} {min(pe):6.2f}..{max(pe):5.2f} {min(sp):6.2f}..{max(sp):5.2f} {sum(ce):8d} of {len(ce)} pass {e['L3']['status']:>8s}")
    print("   L1: the same penalty driven by the exact one-step error, return units against no caution (bar 1.8); L2: the return the"
          " weighted lower edge was predicted to add (bar 1.8 in absolute value); eff. k = mean of BCA's penalty / mean of the oracle's"
          " at the no-caution actor's commands on all kept training rows (the prediction assumed about 1).")
    results["R3c"] = r3c

    banner("PART R4  from the records: the k = 0 plumbing check, the trained actors' strength and points")
    r4 = dict(plumbing={}, near80={}, stage2={})
    print("R4a  k = 0 (the BCA program with every coefficient 0) reproduces the no-caution actor: largest parameter difference")
    for study, key, field in (("near-80", "near80", "check_b_w0"), ("stage 2", "stage2", "check_b")):
        vals = {sw: e[field]["max_abs_diff"] for sw, e in rec[key]["seeds"].items()}
        r4["plumbing"][study] = vals
        print(f"   {study}: {len(vals)} seed-worlds, max |difference| {max(vals.values()):.1e}, all passed: {all(e[field]['passed'] for e in rec[key]['seeds'].values())}")
    print("\nR4b  near-80 (standard band, version 6's arm 'version6_bca' at k = 1), strength = RMS command change from the no-caution actor;"
          " points in strong current and recorded conditions")
    print(f"{'world':9s} {'seed':4s} {'R':>6s} {'u_f':>6s} {'strength: BCA':>13s} {'ramp (M11)':>10s} {'range limit':>11s} {'points strong: none':>19s} {'BCA':>7s} {'ramp':>7s} {'range limit':>11s} {'recorded: none':>14s} {'BCA':>7s}")
    for sw, e in rec["near80"]["seeds"].items():
        w, s = sw.split("/")
        st, p = e["strengths"], e["points"]
        r4["near80"][sw] = dict(strength=st["version6_bca"], ramp=st["obca_k1"], hinge=st["range_hinge"],
                                pts_none=p["none"]["strong_current"], pts_bca=p["version6_bca"]["strong_current"],
                                pts_ramp=p["obca_k1"]["strong_current"], pts_hinge=p["range_hinge"]["strong_current"],
                                rec_none=p["none"]["recorded"], rec_bca=p["version6_bca"]["recorded"], R=e["band"]["R"], unit=e["band"]["unit"])
        print(f"{w:9s} {s:4s} {e['band']['R']:6.3f} {e['band']['unit']:6.3f} {st['version6_bca']:13.4f} {st['obca_k1']:10.4f} {st['range_hinge']:11.4f} "
              f"{p['none']['strong_current']:19.1f} {p['version6_bca']['strong_current']:7.1f} {p['obca_k1']['strong_current']:7.1f} {p['range_hinge']['strong_current']:11.1f} "
              f"{p['none']['recorded']:14.1f} {p['version6_bca']['recorded']:7.1f}")
    print("\nR4c  Stage 2 (classifier-weighted band, arm 'bca_k1' at k = 1; the standard band failed the detection marks, so the"
          " pre-registered weighted version ran); the rivals matched to BCA's strength before any return")
    print(f"{'world':9s} {'seed':4s} {'strength: BCA':>13s} {'range limit':>11s} {'perfect band':>12s} {'points strong: none':>19s} {'BCA':>7s} {'range limit':>11s} {'perfect':>8s} {'clone':>7s} {'recorded: none':>14s} {'BCA':>7s}")
    for sw, e in rec["stage2"]["seeds"].items():
        w, s = sw.split("/")
        st, p = e["strengths"], e["points"]
        r4["stage2"][sw] = dict(strength=st["bca_k1"], hinge=st["range_hinge"], perfect=st["perfect_band"],
                                pts_none=p["none"]["strong_current"], pts_bca=p["bca_k1"]["strong_current"],
                                pts_hinge=p["range_hinge"]["strong_current"], pts_perfect=p["perfect_band"]["strong_current"],
                                pts_clone=p["clone"]["strong_current"], rec_none=p["none"]["recorded"], rec_bca=p["bca_k1"]["recorded"])
        print(f"{w:9s} {s:4s} {st['bca_k1']:13.4f} {st['range_hinge']:11.4f} {st['perfect_band']:12.4f} {p['none']['strong_current']:19.1f} {p['bca_k1']['strong_current']:7.1f} "
              f"{p['range_hinge']['strong_current']:11.1f} {p['perfect_band']['strong_current']:8.1f} {p['clone']['strong_current']:7.1f} {p['none']['recorded']:14.1f} {p['bca_k1']['recorded']:7.1f}")
    sp = rec["stage2_pooled"]
    print(f"   Stage 2 pooled (seeds 0-4, exploratory): M1 BCA - matched even pull {sp['M1_even_pull']['mean']:+.1f} ({'pass' if sp['M1_even_pull']['passed'] else 'fail'});"
          f" M4 BCA - matched range limit {sp['M4_best_standin']['mean']:+.1f} ({'pass' if sp['M4_best_standin']['passed'] else 'fail'}); "
          f"recovers {sp['claim_numbers']['X']:.1f} of {sp['claim_numbers']['harm']:.1f} points; no-stall cost {sp['no_stall_cost']['bca_k1']:.1f}")
    results["R4"] = r4


# ----------------------------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------------------------
def figure(results, rec, r2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # the data-viz reference palette's first three categorical slots (validated for colorblind separation) and ink
    BLUE, ORANGE, GREEN, INK = "#2a78d6", "#eb6834", "#1baf7a", "#222222"
    GRAY = INK                                                   # the no-caution baseline is a reference mark, not a series hue
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.2, top=0.86, wspace=0.3)

    # Panel A: effective k
    ax = axes[0]
    groups = []
    if r2:
        groups.append(("Stage 2\ntraining states", [r2[f"uniform/{s}/T1"]["eff_k"] for s in range(5)], [r2[f"weighted/{s}/T1"]["eff_k"] for s in range(5)]))
        groups.append(("Stage 2\nstrong current", [r2[f"uniform/{s}/T3"]["eff_k"] for s in range(5)], [r2[f"weighted/{s}/T3"]["eff_k"] for s in range(5)]))
    for q, e in rec["weighting_test"]["qualities"].items():
        groups.append((f"weighting test\n{q}", e["L2"]["proxy_check"]["band_over_oracle_per_seed"], None))
    rng = np.random.default_rng(0)
    for i, (name, std, wt) in enumerate(groups):
        x = i + rng.uniform(-0.12, 0.12, len(std))
        ax.scatter(x, std, color=BLUE, marker="o", s=34, zorder=3, label="standard band" if i == 0 else None)
        if wt is not None:
            ax.scatter(x + 0.3, wt, color=ORANGE, marker="s", s=34, zorder=3, label="classifier-weighted band" if i == 0 else None)
    ax.axhline(1.0, color=GREEN, lw=1.4, ls="--")
    ax.text(len(groups) - 0.5, 1.25, "as intended: about 1 (the exact-error penalty at k = 1)", ha="right", fontsize=7.5, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 20)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("BCA's penalty per unit of the critic's over-preference\n(ratio of means, log scale)", fontsize=9)
    ax.set_xlabel("study and setting (one dot per seed)", fontsize=9)
    ax.set_title("A. How much of the critic's error the penalty cancels\n(at the no-caution actor's command, nothing trained)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    # Panel B: the mechanism
    ax = axes[1]
    S4 = results["S"]["S4"]
    for k_, col, mk in ((1.0, BLUE, "o"), (2.0, ORANGE, "s")):
        rows = [r for r in S4["rows"] if r["k"] == k_ and r["slope_ratio"] > 0]
        ax.plot([r["slope_ratio"] for r in rows], [r["final_throttle"] for r in rows], color=col, marker=mk, ms=5, lw=2, label=f"k = {k_:.0f}")
    ax.axhline(KNEE, color=GREEN, lw=1.4, ls="--")
    ax.text(0.0032, KNEE - 0.012, "the truth: stay at or below the 80% knee", fontsize=7.5, color=INK, va="top")
    if r2:
        lo = min(r2[f"uniform/{s}/{S}"]["slope_ratio_median_harmful"] for s in range(5) for S in ("T1", "T3"))
        hi = max(r2[f"uniform/{s}/{S}"]["slope_ratio_median_harmful"] for s in range(5) for S in ("T1", "T3"))
        ax.axvspan(0.003, hi, color=BLUE, alpha=0.15, lw=0)
        ax.text(math.sqrt(0.003 * hi), 0.985, f"standard band, Stage 2 records:\nat most {hi:.3f} ({lo:.3f} to {hi:.3f})",
                ha="center", va="top", fontsize=7.5, color=BLUE)
        lo = min(r2[f"weighted/{s}/{S}"]["slope_ratio_median_harmful"] for s in range(5) for S in ("T1", "T3"))
        hi = max(r2[f"weighted/{s}/{S}"]["slope_ratio_median_harmful"] for s in range(5) for S in ("T1", "T3"))
        ax.axvspan(lo, min(hi, 3.0), color=ORANGE, alpha=0.15, lw=0)
        ax.text(math.sqrt(lo * min(hi, 3.0)), 0.985, "weighted band,\nStage 2 records", ha="center", va="top", fontsize=7.5, color="#9A6700")
    ax.set_xscale("log")
    ax.set_xlim(0.003, 3.2)
    ax.set_ylim(0.74, 1.02)
    ax.set_xlabel("band's slope past the knee / critic's slope (log scale)", fontsize=9)
    ax.set_ylabel("throttle the actor settles at (recorded command 0.75)", fontsize=9)
    ax.set_title("B. Synthetic, known truth: the lower edge holds the actor back\nonly when the band's slope is of the critic's order", fontsize=10)
    ax.legend(fontsize=8, loc="center right")

    # Panel C: strength and points
    ax = axes[2]
    n8, s2 = results["R4"]["near80"], results["R4"]["stage2"]
    stall8 = [v for k_, v in n8.items() if k_.startswith("stall/")]
    stall2 = [v for k_, v in s2.items() if k_.startswith("stall/")]
    ax.scatter([0] * len(stall8) + [0] * len(stall2), [v["pts_none"] for v in stall8] + [v["pts_none"] for v in stall2], color=GRAY, marker="x", s=40, zorder=3, label="no caution (near-80 and Stage 2)")
    ax.scatter([v["strength"] for v in stall8], [v["pts_bca"] for v in stall8], color=BLUE, marker="o", s=44, zorder=4, label="lower edge, standard band (near-80)")
    ax.scatter([v["strength"] for v in stall2], [v["pts_bca"] for v in stall2], color=ORANGE, marker="s", s=44, zorder=4, label="lower edge, weighted band (Stage 2)")
    ax.scatter([v["ramp"] for v in stall8], [v["pts_ramp"] for v in stall8], color=GREEN, marker="^", s=46, zorder=3, label="BCA's switched ramp, M11 (near-80)")
    ax.scatter([v["hinge"] for v in stall8] + [v["hinge"] for v in stall2], [v["pts_hinge"] for v in stall8] + [v["pts_hinge"] for v in stall2],
               facecolors="none", edgecolors=GREEN, marker="D", s=42, lw=1.5, zorder=3, label="hand-set range limit (both)")
    ax.scatter([v["perfect"] for v in stall2], [v["pts_perfect"] for v in stall2], facecolors="none", edgecolors=INK, marker="o", s=40, lw=1.2, zorder=3, label="perfect band, exact error (Stage 2)")
    ax.set_xlabel("strength: RMS change of the command from the no-caution actor", fontsize=9)
    ax.set_ylabel("points in strong current, stalling world (0 = recorder, 100 = best)", fontsize=9)
    ax.set_title("C. What the trained actor did (one marker per seed, seeds 0-4)", fontsize=10)
    ax.set_xlim(-0.01, 0.2)
    ax.legend(fontsize=7.2, loc="lower right")
    for a in axes:
        a.grid(color="#DDDDDD", lw=0.6)
        a.set_axisbelow(True)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
    fig.suptitle("Module M9, the lower edge: the arithmetic is right, the standard band's penalty is about 1% of the critic's error, and the actor does not move",
                 fontsize=11)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, format="svg")
    print("figure written:", FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="run-file root (the WSL clone's runs/shift_explore, read only)")
    ap.add_argument("--copy", default=None, help="a folder of copies (see the header)")
    ap.add_argument("--records", default=os.path.join(DATA, "M9_records.json"))
    ap.add_argument("--r2", default=os.path.join(DATA, "M9_r2.json"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    if args.runs is None and args.copy is None and os.path.exists(os.path.join(WSL, "detect")):
        args.runs = WSL
    t0 = time.time()
    print(f"python {sys.version.split()[0]} numpy {np.__version__} jax {jax.__version__}")
    results = {}
    part_s(results)
    rec = json.load(open(args.records, encoding="utf-8"))
    banner("PART R1  the width at the actor's command recomputed in the pinned environment (quoted)")
    p = os.path.join(DATA, "width_recompute_pinned.out")
    print(open(p, encoding="utf-8").read().strip() if os.path.exists(p) else f"(missing: {p})")
    r2 = None
    if args.runs or args.copy:
        part_r2(args, results)
        r2 = results["R2"]
    elif os.path.exists(args.r2):
        r2 = json.load(open(args.r2, encoding="utf-8"))
        print("\nPART R2 skipped (no --runs or --copy); the figure uses the saved values in", args.r2)
    else:
        print("\nPART R2 skipped (no --runs or --copy, and no saved values)")
    part_r3_r4(rec, results)
    if not args.no_figure:
        figure(results, rec, r2)
    banner("SUMMARY")
    print("S1-S3 (arithmetic, gradient path, refusal):", "PASS" if all(results["S"][k] for k in ("pass_S1", "pass_S2", "pass_S3")) else "FAIL")
    if r2:
        ks = [r2[f"uniform/{s}/{S}"]["eff_k"] for s in range(5) for S in ("T1", "T3")]
        sr = [r2[f"uniform/{s}/{S}"]["slope_ratio_median_harmful"] for s in range(5) for S in ("T1", "T3")]
        print(f"standard band on Stage 2's records: penalty / over-preference {min(ks):.4f}-{max(ks):.4f}; band slope / critic slope {min(sr):.4f}-{max(sr):.4f}")
    n8 = results["R4"]["near80"]
    print(f"near-80 standard band: strength {min(v['strength'] for v in n8.values()):.3f}-{max(v['strength'] for v in n8.values()):.3f}; "
          f"stall strong current BCA {np.mean([v['pts_bca'] for k, v in n8.items() if k.startswith('stall')]):.1f} vs none "
          f"{np.mean([v['pts_none'] for k, v in n8.items() if k.startswith('stall')]):.1f} vs ramp {np.mean([v['pts_ramp'] for k, v in n8.items() if k.startswith('stall')]):.1f}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
