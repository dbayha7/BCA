"""Module M2 of MODULES.md, the error-size network eta(s, a) and the unit u: a synthetic test that drives the project's
own fit on rows whose truth is known, the real-data check on the twenty recorded ThrusterChain bands of the near-80 run,
the pinned-environment recompute on the training rows, and the weighting test's pre-registered mark M2. Prints the
tables that docs/bca_mechanics/modules/M2.md quotes and writes the figure docs/bca_mechanics/figures/M2.svg.

What M2 computes (calibration/network.py:8-46; algorithms/td3_bc_bca.py:159-231):
    eta(s, a) = softplus(net(s / 1.001, a)) + 0.01      net: two layers of 64 ReLU units, one output   (network.py:16-31)
    loss L    = (sum_i m_i c_i - 0.9)^2 + 0.005 sum_i m_i eta_i^2                                       (td3_bc_bca.py:184-188)
        with m_i = E_i / sum_j E_j, E_i ~ Exponential(1), one draw per batch of 256 rows                 (network.py:44-46)
        and  c_i = sigmoid(20 (y_i/u - q_i/u + eta_i)) * sigmoid(20 (q_i/u + eta_i - y_i/u))           (network.py:34-37)
    one Adam step (learning rate 0.001) on eta's parameters; u <- 0.99 u + 0.01 max(SD(y - q), 1e-6);
    the step is kept only if everything is finite, eta > 0, the masses sum to 1 and u > 0 (td3_bc_bca.py:199-227).

HOW TO RUN (no training of any critic or actor; the only thing fitted is the error-size network on SYNTHETIC rows in
part A; nothing is written outside docs/bca_mechanics/figures):
    cd <repository root>
    set PYTHONPATH=<repository root>          (PowerShell: $env:PYTHONPATH = (Get-Location).Path)
    set JAX_PLATFORMS=cpu
    python docs/bca_mechanics/scripts/M2_test.py
Environment used for the committed numbers: the Windows venv, Python 3.14.2, numpy 2.5.4, scipy 1.18.1, jax 0.12.0
(CPU), flax 0.12.10, optax 0.2.8, matplotlib 3.11.2. The pinned environment (Python 3.10.20, numpy 2.2.6, jax 0.6.2,
flax 0.10.7) was used by the companion job scripts/M2_pinned.py (run through the WSL bridge by
scripts/M2_pinned_job.sh); its report docs/bca_mechanics/data/M2_pinned_recompute.json is read here when present
(part C). Every number this script prints comes from:
  - synthetic rows generated here and the project's own fit_scale on them (part A), or
  - the recorded near-80 band files on branch claude/near80-run-files (fetched with `git show`; part B), or
  - the pinned-environment report (part C), or
  - the weighting test's reports outputs/shift_iw/v5/<quality>/report.json on branch main (`git show`; part D).
Options: --repo DIR (repository root; default: three levels above this file), --runfiles DIR (a folder holding
<setting>/<world>/seed<k>/{v6_band.json, v6_bank_scores.npz} instead of git show), --pinned FILE, --v5 DIR (a folder
holding <quality>/report.json), --figure FILE, --no-figure, --steps N (synthetic fit steps; default 20000, the
study's value), --quick (2000 steps, for a smoke test).
"""
import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import namedtuple
from types import SimpleNamespace

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    from scipy import stats
    import algorithms.td3_bc as BASE
    import algorithms.td3_bc_bca as P
    from calibration.network import Calibrator, bayesian_bootstrap_weights, soft_coverage
    from calibration.reference import WBCPConfig, freeze_reference, initial_reference, positive_scale
    from runtime.networks import Transition
except ImportError as error:
    sys.exit(f"set PYTHONPATH to the repository root (algorithms/ and calibration/ must import) and install jax, flax, "
             f"optax, scipy: {error}")

DX, DA = 13, 4                                   # ThrusterChain's state and command sizes (experiments/shift/thruster_chain.py)
BATCH, STEPS = 256, 20_000                       # the study's batch and the frozen-critic fit's steps (learners.py:146)
KEY_CALIBRATOR, TAG_BAND = 1128352841, 41        # td3_bc_bca.initialize's fold (line 135); learners.bca_band's tag
ETA_FLOOR = 0.01                                 # network.py:30
Q90_HALF_NORMAL = 1.6448536269514722             # the 90% point of |Z|, Z standard normal
NEAR80_REFS = ("origin/claude/near80-run-files", "claude/near80-run-files")
NEAR80_DIR = "runs/shift_explore/near80/learners"
MAIN_REFS = ("origin/main", "main")
V5_DIR = "outputs/shift_iw/v5"
QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
SETTINGS, WORLDS, SEEDS = ("near80", "random_rows"), ("stall", "no_stall"), (0, 1, 2, 3, 4)
COLORS = dict(blue="#0072B2", vermilion="#D55E00", green="#009E73")   # Okabe-Ito; validated colorblind-safe
INK = "#333333"


def verdict(ok):
    return "PASS" if ok else "FAIL"


def q90(x):
    return float(np.quantile(np.asarray(x, np.float64), 0.9))


def deciles_by(values, n=10):
    return np.array_split(np.argsort(np.asarray(values, np.float64), kind="stable"), n)


# ============================================================================================ the module, driven as is
Net = namedtuple("Net", "params")


class ZeroActor:
    """The target actor of the synthetic host: command 0 at every next state (its noise is added to 0)."""
    @staticmethod
    def apply(params, obs):
        return jnp.zeros(obs.shape[:-1] + (DA,), jnp.float32)


class ZeroCritic:
    """The synthetic host's critic pair: both heads 0 everywhere. Then td3_bc_bca.native_target gives
    y = r + 0.98 * 0 = r and q = 0, so the residual y - q IS the reward column, which we fill with a known error."""
    @staticmethod
    def apply(params, obs, action):
        return jnp.zeros(obs.shape[:-1] + (2,), jnp.float32)


def bca_config():
    """experiments/shift/learners.py:104-105 (BCA_SETTINGS) as td3_bc_bca.Config."""
    return P.Config(arm="bca", posterior=WBCPConfig(0.1, 0.95, 1000), blend=0.5, cal_lr=0.001, cal_beta=20.0,
                    width_penalty=0.005, scale_ema=0.99)


def fresh_state(seed):
    """td3_bc_bca.initialize's calibrator state (lines 132-141): the network from fold_in(rng, 1128352841), Adam 0.001,
    unit 1, the not-ready reference; the host part is the synthetic zero host."""
    rng = jax.random.PRNGKey(seed)
    cal = Calibrator(jnp.zeros(DX), jnp.ones(DX), state_dep=True)
    params = cal.init(jax.random.fold_in(rng, KEY_CALIBRATOR), jnp.zeros(DX)[None], jnp.zeros(DA)[None])
    cal_state = TrainState.create(apply_fn=cal.apply, params=params, tx=optax.adam(0.001))
    native = BASE.AgentTrainState(Net({}), Net({}), Net({}), Net({}))
    return rng, P.State(native, cal_state, jnp.asarray(1.0), initial_reference(params)), (ZeroActor, ZeroCritic, cal)


ARGS = SimpleNamespace(policy_noise=0.2, noise_clip=0.5, discount=0.98)     # native_target's three settings


def fit(train, steps, seed):
    """learners.bca_band's fit loop (learners.py:1253-1272): fit_scale on `steps` batches of 256 rows drawn with
    default_rng, keys split from fold_in(rng, 41); returns the state, the accepted flags and the losses."""
    rng, state, models = fresh_state(seed)
    config = bca_config()
    rows = np.random.default_rng([20261020, seed, TAG_BAND]).integers(len(train.obs), size=(steps, BATCH))
    keys = jax.random.split(jax.random.fold_in(rng, TAG_BAND), steps)

    def body(st, inp):
        r, k = inp
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        st, diag = P.fit_scale(ARGS, config, models, st, batch, k, 1.0)
        return st, (diag["scale_fit_accepted"], diag["scale_loss"])

    run = jax.jit(lambda st, r, k: jax.lax.scan(body, st, (r, k)))
    state, (accepted, losses) = run(state, jnp.asarray(rows, jnp.int32), keys)
    return state, models, np.asarray(accepted), np.asarray(losses, np.float64)


def eta_of(state, models, trans):
    return np.asarray(models[2].apply(state.calibrator.params, trans.obs, trans.action), np.float64)


def transition(obs, act, e):
    z = np.zeros(len(obs), np.float32)
    return Transition(jnp.asarray(obs), jnp.asarray(act), jnp.asarray(e.astype(np.float32)), jnp.asarray(obs), jnp.asarray(z))


def synthetic_rows(g, n, case):
    """Rows whose error size is known. The state is 13 standard normal numbers (already normalized, as the data pipeline
    hands them to the network), the command 4 uniform numbers in [-1, 1]. The error e = sigma * Z, Z standard normal.
    'visible': log sigma = log 0.1 + 0.6 s_0 + 0.4 a_0, a size the network can read off its inputs;
    'constant': sigma = 0.1 everywhere;
    'hidden': log sigma = log 0.1 + 0.6 h, h a standard normal number the network never sees."""
    obs = g.standard_normal((n, DX)).astype(np.float32)
    act = g.uniform(-1.0, 1.0, (n, DA)).astype(np.float32)
    h = g.standard_normal(n)
    if case == "visible":
        log_sigma = 0.6 * obs[:, 0].astype(np.float64) + 0.4 * act[:, 0].astype(np.float64)
    elif case == "constant":
        log_sigma = np.zeros(n)
    elif case == "hidden":
        log_sigma = 0.6 * h
    else:
        raise ValueError(case)
    sigma = 0.1 * np.exp(log_sigma)
    e = sigma * g.standard_normal(n)
    return obs, act, e, sigma


def threshold_on(eta, unit, e, key):
    """Module M5 on a set of rows: the threshold R from the scores |e| / (u * eta) (reference.freeze_reference)."""
    ref, valid, _ = freeze_reference({"w": jnp.ones(2)}, jnp.asarray(unit, jnp.float32), jnp.asarray(eta, jnp.float32),
                                     jnp.asarray(e, jnp.float32), key, WBCPConfig(0.1, 0.95, 1000))
    return float(ref.threshold), float(ref.lambda_hat), bool(valid)


def penalty_equilibrium():
    """The coverage the loss settles at when the error size is constant (sigma = u): minimize (C(eta) - 0.9)^2 +
    0.005 eta^2 with C(eta) = P(|Z| <= eta) = 2 Phi(eta) - 1; stationarity 4 (C - 0.9) phi(eta) + 0.01 eta = 0."""
    etas = np.linspace(1.0, 2.0, 100001)
    C = 2 * stats.norm.cdf(etas) - 1
    g = 4 * (C - 0.9) * stats.norm.pdf(etas) + 0.01 * etas
    i = int(np.argmin(np.abs(g)))
    return float(etas[i]), float(C[i])


# ============================================================================================ part A: synthetic
def part_a(steps, quick):
    print("=" * 110)
    print("PART A. Synthetic test: the project's own fit (td3_bc_bca.fit_scale) on rows whose error size is known")
    print("=" * 110)
    results = []
    key = jax.random.fold_in(jax.random.PRNGKey(2026), 10)

    # A0. the pieces, one at a time
    cal = Calibrator(jnp.zeros(DX), jnp.ones(DX), state_dep=True)
    prm = cal.init(jax.random.PRNGKey(0), jnp.zeros(DX)[None], jnp.zeros(DA)[None])
    extreme = jnp.asarray([[1e3] * DX, [-1e3] * DX, [0.0] * DX, [1e-3] * DX], jnp.float32)
    eta_x = np.asarray(cal.apply(prm, extreme, jnp.asarray([[1.0] * DA, [-1.0] * DA, [0.0] * DA, [0.5] * DA])))
    floor_ok = bool(np.all(np.isfinite(eta_x)) and np.all(eta_x >= np.float32(ETA_FLOOR)) and np.all(eta_x > 0))
    floor_reached = bool(np.any(eta_x == np.float32(ETA_FLOOR)))
    c_edge = float(soft_coverage(jnp.asarray(1.2), jnp.asarray(0.0), jnp.asarray(1.2))[()])        # |e|/u = eta: the edge
    c_in = float(soft_coverage(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.2))[()])          # the center
    c_out = float(soft_coverage(jnp.asarray(2.4), jnp.asarray(0.0), jnp.asarray(1.2))[()])         # twice the band
    m = np.asarray(bayesian_bootstrap_weights(jax.random.PRNGKey(3), BATCH), np.float64)
    masses_ok = bool(abs(m.sum() - 1) < 1e-6 and np.all(m >= 0) and m.max() < 0.1)
    ok0 = floor_ok and abs(c_edge - 0.5) < 0.01 and c_in > 0.999 and c_out < 1e-9 and masses_ok
    results.append(("A0", ok0))
    print(f"A0 {verdict(ok0)}  pieces: eta at extreme inputs (states +-1000, commands +-1) finite, positive and at or above"
          f" the 0.01 floor: {floor_ok} (values {np.round(eta_x, 4).tolist()}; at state +1000 the float32 softplus underflows and"
          f" eta EQUALS the floor: {floor_reached}, so the exact statement is eta >= 0.01, not eta > 0.01); soft coverage = {c_in:.4f} at the center, {c_edge:.3f} at the"
          f" band's edge, {c_out:.1e} at twice the band (a smoothed indicator of |y - q| <= u*eta, slope 20);"
          f" bootstrap masses: sum {m.sum():.6f}, min {m.min():.2e}, max {m.max():.4f} (all >= 0)")

    # A0b. the acceptance rule: a batch of zero residuals is accepted (the 1e-6 floor keeps u > 0); a NaN batch is not
    rng, st0, models = fresh_state(0)
    config = bca_config()
    g = np.random.default_rng(1)
    obs, act, _, _ = synthetic_rows(g, BATCH, "constant")
    zero_batch = transition(obs, act, np.zeros(BATCH))
    st1, d1 = P.fit_scale(ARGS, config, models, st0, zero_batch, jax.random.PRNGKey(5), 1.0)
    nan_e = np.zeros(BATCH)
    nan_e[0] = np.nan
    st2, d2 = P.fit_scale(ARGS, config, models, st0, transition(obs, act, nan_e), jax.random.PRNGKey(5), 1.0)
    same = jax.tree_util.tree_all(jax.tree_util.tree_map(lambda a, b: bool(np.array_equal(np.asarray(a), np.asarray(b))),
                                                         st2.calibrator.params, st0.calibrator.params))
    u1 = float(st1.residual_scale)
    ok0b = bool(d1["scale_fit_accepted"]) and abs(u1 - (0.99 * 1.0 + 0.01 * 1e-6)) < 1e-6 and u1 > 0 \
        and not bool(d2["scale_fit_accepted"]) and same and float(st2.residual_scale) == 1.0
    results.append(("A0b", ok0b))
    print(f"A0b {verdict(ok0b)}  acceptance rule (M2.8): a batch with residual 0 everywhere is accepted "
          f"({bool(d1['scale_fit_accepted'])}) and u becomes 0.99*1 + 0.01*1e-6 = {u1:.8f} in float32 (> 0);"
          f" a batch with one NaN residual is rejected ({not bool(d2['scale_fit_accepted'])}) and eta's parameters and u"
          f" stay as they were ({same}, u = {float(st2.residual_scale)})")

    # A1-A3: the full fit on three worlds
    n_train, n_held = 40_000, 4_000
    table = []
    fitted = {}
    for case, label in (("visible", "error size visible to the network"), ("constant", "error size constant"),
                        ("hidden", "error size hidden from the network")):
        g = np.random.default_rng({"visible": 11, "constant": 12, "hidden": 13}[case])
        obs, act, e, sigma = synthetic_rows(g, n_train, case)
        obs_h, act_h, e_h, sigma_h = synthetic_rows(g, n_held, case)
        t0 = time.perf_counter()
        state, models, accepted, losses = fit(transition(obs, act, e), steps, seed=0)
        secs = time.perf_counter() - t0
        u = float(state.residual_scale)
        eta_t, eta_h = eta_of(state, models, transition(obs, act, e)), eta_of(state, models, transition(obs_h, act_h, e_h))
        share_t = float(np.mean(np.abs(e) <= u * eta_t))
        share_h = float(np.mean(np.abs(e_h) <= u * eta_h))
        R, lam, valid = threshold_on(eta_h, u, e_h, key)
        rho = float(stats.spearmanr(eta_h, sigma_h)[0]) if case != "constant" else float("nan")
        ratio = eta_h * u / sigma_h                      # eta in units of the row's true sigma; 1.645 = the row's 90% point
        parts = deciles_by(sigma_h)
        cov_by_sigma = [float(np.mean(np.abs(e_h[p]) <= u * eta_h[p])) for p in parts]
        row = dict(case=case, label=label, steps=steps, seconds=secs, accepted_share=float(accepted.mean()),
                   eta_finite_above_floor=bool(np.all(np.isfinite(eta_t)) and np.all(eta_t > ETA_FLOOR)
                                               and np.all(np.isfinite(eta_h)) and np.all(eta_h > ETA_FLOOR)),
                   unit=u, sd_e_train=float(e.std()), unit_over_sd=u / float(e.std()),
                   eta_min=float(eta_h.min()), eta_median=float(np.median(eta_h)), eta_max=float(eta_h.max()),
                   eta_cv=float(eta_h.std() / eta_h.mean()),
                   share_train=share_t, share_heldout=share_h, R=R, lambda_hat=lam, valid=valid,
                   spearman_eta_sigma=rho, ratio_median=float(np.median(ratio)),
                   ratio_q10=float(np.quantile(ratio, 0.1)), ratio_q90=float(np.quantile(ratio, 0.9)),
                   coverage_by_sigma_decile=cov_by_sigma, loss_first=float(losses[:100].mean()),
                   loss_last=float(losses[-100:].mean()))
        table.append(row)
        fitted[case] = dict(state=state, models=models, eta_h=eta_h, sigma_h=sigma_h, e_h=e_h, u=u,
                            obs_h=obs_h, act_h=act_h)
    print()
    print("Table A. The fit on three synthetic worlds (40,000 training rows, 4,000 held-out rows from the same law; "
          f"{steps} fit steps of 256 rows; the study's settings)")
    print(f"{'world':36s} {'accepted':>8s} {'eta>0.01':>8s} {'u/SD(e)':>8s} {'eta med [min,max]':>22s} "
          f"{'share<=u*eta train':>18s} {'held-out':>8s} {'R (M5)':>7s} {'Spearman(eta,sigma)':>19s} {'eta*u/sigma med [q10,q90]':>26s}")
    for r in table:
        print(f"{r['label']:36s} {r['accepted_share']:8.3f} {str(r['eta_finite_above_floor']):>8s} {r['unit_over_sd']:8.3f} "
              f"{r['eta_median']:7.3f} [{r['eta_min']:.2f},{r['eta_max']:.2f}]   {r['share_train']:14.3f} {r['share_heldout']:8.3f}"
              f" {r['R']:7.3f} {r['spearman_eta_sigma']:19.3f}  {r['ratio_median']:6.3f} [{r['ratio_q10']:.2f},{r['ratio_q90']:.2f}]")
    eta_eq, cov_eq = penalty_equilibrium()
    print(f"  (eta*u/sigma = the band at threshold 1 in units of the row's true error scale; 1.645 would be the row's own 90% point."
          f" The loss's own equilibrium with a constant error size is eta = {eta_eq:.3f} sigma/u and coverage {cov_eq:.3f}, not"
          f" 0.900: the width penalty 0.005 trades about {(0.9 - cov_eq) * 100:.0f} points of coverage for a narrower band.)")
    print()
    print("Coverage (share with |e| <= u*eta) inside each tenth of the held-out rows by TRUE error size, smallest to largest:")
    for r in table:
        print(f"  {r['label']:36s} " + " ".join(f"{c:.2f}" for c in r["coverage_by_sigma_decile"]))
    v, c, h = (next(r for r in table if r["case"] == k) for k in ("visible", "constant", "hidden"))
    ok1 = all(r["accepted_share"] == 1.0 and r["eta_finite_above_floor"] and abs(r["unit_over_sd"] - 1) < 0.03 for r in table)
    results.append(("A1", ok1))
    print(f"A1 {verdict(ok1)}  mechanics: every fit step accepted (accepted share 1.000 in all three worlds), eta finite and"
          f" above 0.01 at every row, the unit within 3% of SD(e) on the training rows"
          f" ({v['unit_over_sd']:.3f}, {c['unit_over_sd']:.3f}, {h['unit_over_sd']:.3f})")
    ok2a = all(abs(r["share_train"] - cov_eq) <= 0.02 for r in table)
    results.append(("A2a", ok2a))
    print(f"A2a {verdict(ok2a)}  coverage on the rows the fit saw: the share with |e| <= u*eta is {v['share_train']:.3f},"
          f" {c['share_train']:.3f}, {h['share_train']:.3f}, within 0.02 of the loss's own equilibrium {cov_eq:.3f} (not 0.900:"
          f" the width penalty's trade)")
    gap_a = np.mean([r["share_train"] - r["share_heldout"] for r in table]) * 100
    ok2b = all(r["share_heldout"] >= 0.85 and 0.95 <= r["R"] <= 1.25 for r in table)
    results.append(("A2b", ok2b))
    print(f"A2b {verdict(ok2b)}  coverage on held-out rows from the SAME law: {v['share_heldout']:.3f}, {c['share_heldout']:.3f},"
          f" {h['share_heldout']:.3f}, and M5's threshold R = {v['R']:.3f}, {c['R']:.3f}, {h['R']:.3f}. The strict statement"
          f" 'held-out share at least 0.85 and R at most 1.25' {'holds' if ok2b else 'does not hold'}: the network overfits the"
          f" training rows' particular errors by {gap_a:.1f} points on average (most where there is nothing to learn, the hidden"
          f" case), and R absorbs it. On ThrusterChain eta is far flatter and this cost is under half a point (part C, C4).")
    lo, hi = h["coverage_by_sigma_decile"][0], h["coverage_by_sigma_decile"][-1]
    ok3 = v["spearman_eta_sigma"] > 0.8 and c["eta_cv"] < 0.2 and h["eta_cv"] < 0.2 and lo > 0.95 and hi < 0.75
    results.append(("A3", ok3))
    print(f"A3 {verdict(ok3)}  tracking: where the error size can be read off (s, a), eta follows it (Spearman"
          f" {v['spearman_eta_sigma']:.3f}; coverage by size decile {v['coverage_by_sigma_decile'][0]:.2f} to"
          f" {v['coverage_by_sigma_decile'][-1]:.2f}); with a constant size eta is nearly constant (spread/mean"
          f" {c['eta_cv']:.3f}); where the size is hidden eta is flat (spread/mean {h['eta_cv']:.3f}) and the band is"
          f" wrong row by row: coverage {lo:.2f} on the smallest-error tenth, {hi:.2f} on the largest, while the average"
          f" stays {h['share_heldout']:.3f}")

    # A4. the real data's situation: held-out errors larger than the training errors the network was fitted to
    f = fitted["visible"]
    out4 = []
    for factor in (1.0, 1.2, 1.3, 1.4):
        e_big = f["e_h"] * factor
        share = float(np.mean(np.abs(e_big) <= f["u"] * f["eta_h"]))
        R, lam, _ = threshold_on(f["eta_h"], f["u"], e_big, key)
        out4.append((factor, share, R, float(np.mean(np.abs(e_big) <= R * f["u"] * f["eta_h"]))))
    expected = {fac: 2 * (0.5 * (1 + math.erf(1.555 / fac / math.sqrt(2)))) - 1 for fac, _, _, _ in out4}
    ok4 = all(abs(s - expected[fac]) < 0.04 for fac, s, _, _ in out4) and out4[2][2] > 1.25 and out4[0][2] < 1.25
    results.append(("A4", ok4))
    print(f"A4 {verdict(ok4)}  the real data's situation: the same fitted eta and u, held-out errors scaled up by a factor"
          " (the recorded critics' held-out residual SD is 1.2-1.6 times the training one):")
    print(f"      {'factor':>6s} {'share<=u*eta':>12s} {'expected':>8s} {'R (M5)':>7s} {'share<=R*u*eta':>14s}")
    for fac, s, R, sb in out4:
        print(f"      {fac:6.1f} {s:12.3f} {expected[fac]:8.3f} {R:7.3f} {sb:14.3f}")
    print("      (expected = P(|Z| <= 1.555/factor): the held-out share falls to about 0.8 and the threshold rises to about"
          " 1.3-1.4, which is exactly the recorded pattern of part B; the threshold step M5 restores 0.9)")
    return results, table, out4, fitted


# ============================================================================================ part B: recorded bands
def git_show(repo, refs, path):
    for ref in refs:
        out = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True)
        if out.returncode == 0:
            return out.stdout
    return None


def load_bands(repo, runfiles):
    bands = []
    for setting in SETTINGS:
        for world in WORLDS:
            for seed in SEEDS:
                rel = f"{setting}/{world}/seed{seed}"
                if runfiles:
                    try:
                        with open(os.path.join(runfiles, rel, "v6_band.json"), "rb") as fh:
                            rec = json.loads(fh.read())
                        with open(os.path.join(runfiles, rel, "v6_bank_scores.npz"), "rb") as fh:
                            raw = fh.read()
                    except OSError:
                        continue
                else:
                    js = git_show(repo, NEAR80_REFS, f"{NEAR80_DIR}/{rel}/v6_band.json")
                    raw = git_show(repo, NEAR80_REFS, f"{NEAR80_DIR}/{rel}/v6_bank_scores.npz")
                    if js is None or raw is None:
                        continue
                    rec = json.loads(js)
                with np.load(io.BytesIO(raw)) as z:
                    scores, eta = np.array(z["scores"]), np.array(z["eta"])
                bands.append(dict(setting=setting, world=world, seed=seed, rec=rec, scores=scores, eta=eta))
    return bands


def part_b(bands):
    print()
    print("=" * 110)
    print("PART B. Real data: the twenty recorded version-6 bands of the near-80 run (frozen critic, seeds 0-4, both worlds,"
          " settings near80 and random_rows), read from v6_band.json and v6_bank_scores.npz on branch claude/near80-run-files")
    print("=" * 110)
    if not bands:
        print("  (no band files found: give --runfiles or fetch branch claude/near80-run-files)")
        return [], []
    rows = []
    for b in bands:
        rec, sc, eta = b["rec"], b["scores"], b["eta"]
        u, R = float(rec["uniform"]["unit"]), float(rec["uniform"]["R"])
        abs_e = sc * u * np.maximum(eta, 1e-6)                 # |e| recovered from the score (M3 inverted; float64)
        parts = deciles_by(eta)
        rows.append(dict(setting=b["setting"], world=b["world"], seed=b["seed"], n=int(len(sc)),
                         accepted=float(rec["accepted_share"]), steps=int(rec["steps"]), unit=u, R=R,
                         lambda_hat=float(rec["uniform"]["lambda_hat"]), certified=bool(rec["uniform"]["certified"]),
                         eta_min=float(eta.min()), eta_median=float(np.median(eta)), eta_max=float(eta.max()),
                         eta_ok=bool(np.all(np.isfinite(eta)) and np.all(eta > ETA_FLOOR)),
                         share_unit_eta=float(np.mean(sc <= 1.0)), share_band=float(np.mean(sc <= R)),
                         score_q90=q90(sc), spearman=float(stats.spearmanr(eta, abs_e)[0]),
                         score_q90_by_eta_decile=[q90(sc[p]) for p in parts],
                         eta_by_decile=[float(eta[p].mean()) for p in parts],
                         refresh_valid=bool(rec["refresh"]["posterior_inputs_valid"])))
    print(f"{'setting':11s} {'world':9s} {'seed':>4s} {'n':>4s} {'accepted':>8s} {'unit':>7s} {'R':>7s} {'eta min':>7s} "
          f"{'median':>7s} {'max':>6s} {'>0.01':>5s} {'<=u*eta':>8s} {'<=R*u*eta':>9s} {'score q90':>9s} {'Spearman(eta,|e|)':>17s}")
    for r in rows:
        print(f"{r['setting']:11s} {r['world']:9s} {r['seed']:4d} {r['n']:4d} {r['accepted']:8.3f} {r['unit']:7.4f} {r['R']:7.4f} "
              f"{r['eta_min']:7.3f} {r['eta_median']:7.3f} {r['eta_max']:6.3f} {str(r['eta_ok']):>5s} {r['share_unit_eta']:8.3f} "
              f"{r['share_band']:9.3f} {r['score_q90']:9.3f} {r['spearman']:17.3f}")
    su = np.array([r["share_unit_eta"] for r in rows])
    sb = np.array([r["share_band"] for r in rows])
    results = []
    okb1 = all(r["accepted"] == 1.0 and r["eta_ok"] and r["refresh_valid"] and r["steps"] == 20_000 for r in rows)
    results.append(("B1", okb1))
    print(f"B1 {verdict(okb1)}  on all {len(rows)} bands: accepted share 1.000 over 20,000 fit steps, eta finite and above 0.01"
          f" at every held-out row (range over bands {min(r['eta_min'] for r in rows):.3f} to {max(r['eta_max'] for r in rows):.3f}),"
          f" the refresh valid")
    okb2 = bool(np.all(su < 0.85)) and bool(np.all(np.abs(sb - 0.917) < 0.005))
    results.append(("B2", okb2))
    print(f"B2 {verdict(okb2)}  on the held-out rows the share with |e| <= u*eta is {su.min():.3f} to {su.max():.3f} (mean"
          f" {su.mean():.3f}), BELOW the 0.9 the fit aims at; the threshold R = {min(r['R'] for r in rows):.2f} to"
          f" {max(r['R'] for r in rows):.2f} brings the band to {sb.min():.3f}-{sb.max():.3f} (M5 and M6 do their job; whether"
          " the shortfall is the fit's or the critic's is answered by the training rows, part C)")
    sp = np.array([r["spearman"] for r in rows])
    print(f"B3 (description)  Spearman rank correlation of eta with |e| on the held-out rows: {sp.min():.2f} to {sp.max():.2f}"
          f" (mean {sp.mean():.2f}); eta's spread across rows: median {np.mean([r['eta_median'] for r in rows]):.2f}, typical range"
          f" {np.mean([r['eta_min'] for r in rows]):.2f} to {np.mean([r['eta_max'] for r in rows]):.2f} (times u = {np.mean([r['unit'] for r in rows]):.3f})")
    return results, rows


# ============================================================================================ part C: pinned recompute
def part_c(pinned_path):
    print()
    print("=" * 110)
    print("PART C. Pinned environment (Python 3.10.20, numpy 2.2.6, jax 0.6.2): eta and the residual recomputed from the saved"
          " critic and band at every kept TRAINING row (the rows the fit drew its batches from) and every held-out row")
    print("=" * 110)
    if not pinned_path or not os.path.exists(pinned_path):
        print(f"  (report not found: {pinned_path}; run scripts/M2_pinned_job.sh through the WSL bridge)")
        return [], None
    with open(pinned_path) as fh:
        rep = json.load(fh)
    bands = [b for b in rep["bands"] if not b.get("failed")]
    print(f"  report written {rep['written']}, environment {rep['environment']}, {len(bands)} bands")
    print(f"{'setting':11s} {'world':9s} {'seed':>4s} {'n train':>7s} {'hashes':>6s} {'eta=rec':>7s} {'sc=rec':>6s} "
          f"{'eta train min':>13s} {'max':>6s} {'u/SD train':>10s} {'SD held/SD train':>16s} {'<=u*eta train':>13s} {'held':>6s} "
          f"{'<=R*u*eta train':>15s} {'held':>6s} {'q90 train':>9s} {'held':>6s} {'rho train':>9s} {'held':>6s}")
    for b in bands:
        t, h = b["train"], b["bank"]
        print(f"{b['setting']:11s} {b['world']:9s} {b['seed']:4d} {b['n_train']:7d} {str(b['rows_hashes_equal']):>6s} "
              f"{str(b['eta_bank_byte_equal']):>7s} {str(b['scores_byte_equal']):>6s} {t['eta_min']:13.3f} {t['eta_max']:6.3f} "
              f"{b['unit_over_sd_e_train']:10.3f} {h['sd_e'] / t['sd_e']:16.3f} {t['share_within_unit_eta']:13.3f} "
              f"{h['share_within_unit_eta']:6.3f} {t['share_within_band']:15.3f} {h['share_within_band']:6.3f} "
              f"{t['score_q90']:9.3f} {h['score_q90']:6.3f} {t['spearman_eta_abs_e']:9.3f} {h['spearman_eta_abs_e']:6.3f}")
    st = np.array([b["train"]["share_within_unit_eta"] for b in bands])
    sh = np.array([b["bank"]["share_within_unit_eta"] for b in bands])
    usd = np.array([b["unit_over_sd_e_train"] for b in bands])
    gap = np.array([b["bank"]["sd_e"] / b["train"]["sd_e"] for b in bands])
    results = []
    okc1 = all(b["rows_hashes_equal"] and b["native_sha_equal"] and b["eta_bank_byte_equal"] and b["scores_byte_equal"]
               and b["unit_equals_record"] and b["train"]["eta_finite"] and b["train"]["eta_above_floor"] for b in bands)
    results.append(("C1", okc1))
    print(f"C1 {verdict(okc1)}  integrity: prepared rows' hashes, the frozen host and the unit equal the records; eta at the"
          f" held-out rows and the scores equal the stored arrays byte for byte; eta finite and above 0.01 at all"
          f" {sum(b['n_train'] for b in bands):,} training rows ({len(bands)} of {len(bands)} bands)")
    okc2 = bool(np.all(np.abs(usd - 1) < 0.07)) and bool(np.all(st > 0.86)) and bool(np.all(st < 0.90))
    results.append(("C2", okc2))
    print(f"C2 {verdict(okc2)}  the fit did its job on the rows it saw: unit/SD(e) = {usd.min():.3f} to {usd.max():.3f} on the"
          f" training rows (the unit averages batch SDs, which sit a few percent below the whole-set SD when the residual has"
          f" heavy tails; the largest recorded scores reach 9); share with |e| <= u*eta on the training rows {st.min():.3f} to"
          f" {st.max():.3f} (mean {st.mean():.3f}; the 0.9 target less the width penalty's pull, as in part A)")
    okc3 = bool(np.all(gap > 1.1)) and bool(np.all(sh < st - 0.04))
    results.append(("C3", okc3))
    print(f"C3 {verdict(okc3)}  the held-out shortfall is the critic's generalization gap, not the fit's: the critic's residual"
          f" SD on held-out rows is {gap.min():.2f} to {gap.max():.2f} times its training-row SD, and the held-out share drops to"
          f" {sh.min():.3f}-{sh.max():.3f} (mean {sh.mean():.3f}), {np.mean(st - sh) * 100:.0f} points below the training share")
    ce = [b.get("constant_eta") for b in bands]
    if all(ce):
        dt = np.array([b["train"]["share_within_unit_eta"] - c["train_share_within_unit_eta"] for b, c in zip(bands, ce)])
        db = np.array([b["bank"]["share_within_unit_eta"] - c["bank_share_within_unit_eta"] for b, c in zip(bands, ce)])
        dbb = np.array([b["bank"]["share_within_band"] - c["bank_share_within_band"] for b, c in zip(bands, ce)])
        qr = np.array([b["q90_abs_e_ratio_bank_over_train"] for b in bands])
        okc4 = bool(np.all(np.abs(dt) < 0.02) and np.all(np.abs(db) < 0.02) and np.all(qr > 1.1))
        results.append(("C4", okc4))
        print(f"C4 {verdict(okc4)}  the caveat, measured: with eta(s, a) replaced by ONE number (its mean over the training rows,"
              f" {np.mean([c['eta_mean_train'] for c in ce]):.3f}), the share inside u*eta changes by {dt.mean() * 100:+.1f} points on"
              f" the training rows (range {dt.min() * 100:+.1f} to {dt.max() * 100:+.1f}) and {db.mean() * 100:+.1f} points on the"
              f" held-out rows ({db.min() * 100:+.1f} to {db.max() * 100:+.1f}); inside the full band R*u*eta, {dbb.mean() * 100:+.1f}"
              f" points on the held-out rows. The network's row-by-row shape adds almost nothing to a constant width on"
              f" ThrusterChain. The held-out shortfall is the critic's: its 90% error point is {qr.min():.2f} to {qr.max():.2f}"
              f" times larger on held-out rows than on training rows (mean {qr.mean():.2f}).")
    # the row-by-row view: the 90% point of the score inside tenths of the rows by eta (1 = eta right in that tenth)
    def pooled(key, which):
        return np.mean([b[which]["by_eta_decile"][key] for b in bands], axis=0)
    print()
    print("  Inside each tenth of the rows by eta (smallest to largest), pooled over the 20 bands: the 90% point of the score"
          " |e|/(u*eta) (1.00 = eta right for that tenth) and the mean eta")
    print("    mean eta, training   " + " ".join(f"{v:5.2f}" for v in pooled("eta_mean", "train")))
    print("    q90 score, training  " + " ".join(f"{v:5.2f}" for v in pooled("score_q90", "train")))
    print("    q90 score, held-out  " + " ".join(f"{v:5.2f}" for v in pooled("score_q90", "bank")))
    print("    coverage, training   " + " ".join(f"{v:5.2f}" for v in pooled("share_within_unit_eta", "train")))
    print("    coverage, held-out   " + " ".join(f"{v:5.2f}" for v in pooled("share_within_unit_eta", "bank")))
    def pooled_c(key, which):
        return np.mean([b[which]["by_current_decile"][key] for b in bands], axis=0)
    print("  Inside each tenth of the rows by WATER CURRENT (weakest to strongest), pooled over the 20 bands:")
    print("    mean current         " + " ".join(f"{v:5.2f}" for v in pooled_c("current_mean", "train")))
    print("    mean eta, training   " + " ".join(f"{v:5.2f}" for v in pooled_c("eta_mean", "train")))
    print("    mean |e|, training   " + " ".join(f"{v:5.3f}" for v in pooled_c("abs_e_mean", "train")))
    print("    coverage, training   " + " ".join(f"{v:5.2f}" for v in pooled_c("share_within_unit_eta", "train")))
    print("    coverage, held-out   " + " ".join(f"{v:5.2f}" for v in pooled_c("share_within_unit_eta", "bank")))
    ft = [b["train"]["full_throttle"] for b in bands if b["train"]["full_throttle"]["rows"]]
    if ft:
        print(f"  Full-throttle training rows (some thruster at +-1; {int(np.mean([f['rows'] for f in ft])):,} rows per band on"
              f" average): mean eta {np.mean([f['eta_mean'] for f in ft]):.2f} against"
              f" {np.mean([b['train']['eta_mean'] for b in bands]):.2f} overall; coverage {np.mean([f['share_within_unit_eta'] for f in ft]):.3f}"
              f" against {st.mean():.3f} overall")
    rho_t = np.array([b["train"]["spearman_eta_abs_e"] for b in bands])
    rho_c = np.array([b["train"]["spearman_eta_current"] for b in bands])
    rho_ec = np.array([b["train"]["spearman_abs_e_current"] for b in bands])
    print(f"  Rank correlations on the training rows: eta with |e| {rho_t.min():.2f} to {rho_t.max():.2f}; eta with the current"
          f" {rho_c.min():.2f} to {rho_c.max():.2f}; |e| with the current {rho_ec.min():.2f} to {rho_ec.max():.2f}")
    return results, dict(bands=bands, pooled_train=pooled("score_q90", "train"), pooled_bank=pooled("score_q90", "bank"),
                         pooled_eta=pooled("eta_mean", "train"))


# ============================================================================================ part D: the M2 mark
def part_d(repo, v5dir):
    print()
    print("=" * 110)
    print("PART D. The weighting test's pre-registered mark M2 (outputs/shift_iw/v5/<quality>/report.json, branch main):"
          " Spearman rank correlation of the band width U with the EXACT critic error |q - Q^pi| (by simulation), inside each"
          " tenth of the held-out truth rows by water current, averaged; pass = one-sided lower bound above 0")
    print("=" * 110)
    rows = []
    for qlt in QUALITIES:
        if v5dir:
            try:
                with open(os.path.join(v5dir, qlt, "report.json")) as fh:
                    rep = json.load(fh)
            except OSError:
                continue
        else:
            js = git_show(repo, MAIN_REFS, f"{V5_DIR}/{qlt}/report.json")
            if js is None:
                continue
            rep = json.loads(js)
        m = rep["marks"]["M2"]
        rows.append(dict(quality=qlt, status=m["status"], within=m["within_deciles"], pooled=m["pooled"],
                         given=m["given_float_radius_within_deciles"], noise=m["noise_map_within_deciles"]))
    if not rows:
        print("  (reports not found: fetch branch main or give --v5)")
        return [], []
    print(f"{'quality':14s} {'status':>6s} {'within deciles: mean':>20s} {'lower bound':>11s} {'per seed':>36s} {'pooled':>7s} "
          f"{'given float radius':>18s} {'noise map':>9s}")
    for r in rows:
        print(f"{r['quality']:14s} {r['status']:>6s} {r['within']['mean']:20.3f} {r['within']['lower_one_sided']:11.3f} "
              f"{str([round(x, 2) for x in r['within']['per_seed']]):>36s} {r['pooled']['mean']:7.3f} "
              f"{r['given']['mean']:9.3f} ({r['given']['lower_one_sided']:.3f}) {r['noise']['mean']:9.3f}")
    okd = all(r["status"] == "pass" and r["within"]["lower_one_sided"] > 0 for r in rows)
    print(f"D1 {verdict(okd)}  the mark passes on {sum(r['status'] == 'pass' for r in rows)} of {len(rows)} qualities, but the"
          f" link is weak: within-decile correlation {min(r['within']['mean'] for r in rows):.2f} to"
          f" {max(r['within']['mean'] for r in rows):.2f}; after removing the float radius (position and speed size), "
          f"{min(r['given']['mean'] for r in rows):.2f} to {max(r['given']['mean'] for r in rows):.2f} (expert: 0.063, lower"
          f" bound 0.018). The width follows the noise map (how noisy the recorded outcomes are) much more closely:"
          f" {min(r['noise']['mean'] for r in rows):.2f} to {max(r['noise']['mean'] for r in rows):.2f}")
    return [("D1", okd)], rows


# ============================================================================================ figure
def figure(path, fitted, rows_b, pinned, png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9, "axes.edgecolor": "#888888", "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.titlesize": 9.5,
                         "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    # (a) synthetic: fitted eta against the true error size
    ax = axes[0]
    for case, color, marker, label in (("visible", COLORS["blue"], "o", "error size visible to the network"),
                                       ("hidden", COLORS["vermilion"], "^", "error size hidden from the network")):
        f = fitted[case]
        x = f["sigma_h"] / f["u"]
        idx = np.random.default_rng(0).choice(len(x), 700, replace=False)
        ax.scatter(x[idx], f["eta_h"][idx], s=9, marker=marker, color=color, alpha=0.55, linewidths=0, label=label)
    xs = np.linspace(0.2, 2.2, 50)
    ax.plot(xs, Q90_HALF_NORMAL * xs, color=INK, lw=1, ls="--", label="row's own 90% point (1.645 x size)")
    ax.set_xlabel("true error size of the row, in units of u  (sigma / u)")
    ax.set_ylabel("fitted eta(s, a)")
    ax.set_title("(a) Synthetic rows, the fit as the study runs it:\neta follows the error size only where it can see it")
    ax.set_xlim(0.2, 2.3)
    ax.set_ylim(0, 3)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.legend(loc="upper left", frameon=False)
    # (b) real data: shares per band
    ax = axes[1]
    labels = [f"{r['setting'][0:2]}-{'s' if r['world'] == 'stall' else 'n'}{r['seed']}" for r in rows_b]
    xi = np.arange(len(rows_b))
    if pinned:
        key = {(b["setting"], b["world"], b["seed"]): b for b in pinned["bands"]}
        tr = [key[(r["setting"], r["world"], r["seed"])]["train"]["share_within_unit_eta"]
              if (r["setting"], r["world"], r["seed"]) in key else np.nan for r in rows_b]
        ax.plot(xi, tr, marker="s", ms=5, lw=1, color=COLORS["vermilion"], label="training rows, |e| <= u*eta (the fit's own rows)")
    ax.plot(xi, [r["share_unit_eta"] for r in rows_b], marker="o", ms=5, lw=1, color=COLORS["blue"],
            label="held-out rows, |e| <= u*eta (before the threshold)")
    ax.plot(xi, [r["share_band"] for r in rows_b], marker="^", ms=5, lw=1, color=COLORS["green"],
            label="held-out rows, |e| <= R*u*eta (the band, M6)")
    ax.axhline(0.9, color=INK, lw=0.8, ls="--", label="the fit's target, 0.90")
    ax.set_xticks(xi)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("recorded band  (ne = near80, ra = random_rows; s = stalling world, n = no-stall; seed)")
    ax.set_ylabel("share of rows inside the band")
    ax.set_title("(b) ThrusterChain, 20 recorded bands:\nshare of rows inside the band, before and after the threshold R")
    ax.set_ylim(0.6, 1.0)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.legend(loc="lower center", frameon=False)
    # (c) real data: the 90% point of the score by eta decile
    ax = axes[2]
    d = np.arange(1, 11)
    if pinned:
        ax.plot(d, pinned["pooled_train"], marker="s", ms=5, lw=1.2, color=COLORS["vermilion"], label="training rows (pinned recompute)")
        held = pinned["pooled_bank"]
    else:
        held = np.mean([r["score_q90_by_eta_decile"] for r in rows_b], axis=0)
    ax.plot(d, held, marker="o", ms=5, lw=1.2, color=COLORS["blue"], label="held-out rows (recorded scores)")
    ax.axhline(1.0, color=INK, lw=0.8, ls="--")
    ax.text(10.3, 1.02, "eta right for the tenth", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(d)
    ax.set_xlabel("tenth of the rows by eta  (1 = smallest eta, 10 = largest)")
    ax.set_ylabel("90% point of the score |e| / (u*eta) in that tenth")
    ax.set_title("(c) The 20 recorded bands pooled, rows grouped by eta:\neta is right on average, not row by row")
    ax.set_ylim(0.6, 3.0)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.legend(loc="upper left", frameon=False)
    fig.suptitle("Module M2: the error-size network eta(s, a) and the unit u", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, format="svg")
    if png:
        fig.savefig(png, format="png", dpi=110)
    plt.close(fig)
    print(f"\nfigure written: {path}")


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    repo_default = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--repo", default=repo_default)
    ap.add_argument("--runfiles", default=None)
    ap.add_argument("--pinned", default=os.path.join(repo_default, "docs", "bca_mechanics", "data", "M2_pinned_recompute.json"))
    ap.add_argument("--v5", default=None)
    ap.add_argument("--figure", default=os.path.join(repo_default, "docs", "bca_mechanics", "figures", "M2.svg"))
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--png", default=None, help="also write a PNG preview of the figure")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps = 2000 if a.quick else a.steps
    print(f"M2_test.py  python {sys.version.split()[0]}, numpy {np.__version__}, jax {jax.__version__} ({jax.default_backend()})"
          f", {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; repo {a.repo}")
    results = []
    ra, table_a, out4, fitted = part_a(steps, a.quick)
    results += ra
    bands = load_bands(a.repo, a.runfiles)
    rb, rows_b = part_b(bands)
    results += rb
    rc, pinned = part_c(a.pinned)
    results += rc
    rd, rows_d = part_d(a.repo, a.v5)
    results += rd
    print()
    print("=" * 110)
    print("SUMMARY  " + "  ".join(f"{k} {verdict(ok)}" for k, ok in results))
    print("=" * 110)
    if not a.no_figure and rows_b:
        figure(a.figure, fitted, rows_b, pinned, a.png)


if __name__ == "__main__":
    main()
