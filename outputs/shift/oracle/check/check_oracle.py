#!/usr/bin/env python
"""Independent, adversarial check of outputs/shift/oracle/ (the exact optimal reference for ThrusterChain).

Nothing here imports oracle.py or reuses its matrices. The world is only touched through the repository's own
experiments/shift/thruster_chain.py (dynamics() in numpy float64 for every reported number, dynamics_jax under
jax_enable_x64 for gradients) and the bank starts. No learner is trained, no BCA band is fitted. Everything written
goes under outputs/shift/oracle/check/.

Items
  1  optimum   a different solver for the 100-step optimum: L-BFGS-B on the 400 raw commands, cost = minus the sum of
               the rewards of a 100-step rollout of dynamics_jax (float64), gradient by JAX reverse-mode autodiff of that
               rollout. 125 starts per (world, situation): 100 spread over the bank (indices 5, 15, ..., 995) plus the
               25 starts where the oracle's plan sits on the throttle limit most often (the hardest active sets). Both
               plans are replayed through numpy thruster_chain.dynamics() and their returns compared. Also: an
               independent optimality certificate (Frank-Wolfe gap from the JAX gradient, no use of their matrix M)
               for ALL 8,000 of the oracle's plans.
  2  nonconvex the stalling world over the FULL range [-1, 1] with the TRUE stalling thrust: (a) the elementwise facts
               the dominance argument rests on, on every float32 command in [0.5, 1] and 10 million float64 commands;
               (b) 500 random full-range plans on each of 80 starts, replayed with and without the 'replace each
               command by its own push' map; (c) 8 local searches (L-BFGS-B, true thrust, bounds +-1) from different
               starting plans on each of 80 starts, each then mapped and polished inside the knee box.
  3  points    anchors recomputed with an own re-implementation of the recording, stall-aware and rising-thrust
               controllers (own textbook gain from scipy's Riccati solver, own noise and coin streams, own rollout loop);
               every saved gate and Stage 2 actor re-scored with an own numpy forward pass and the normalisation from
               the cloned actors' records; the oracle's optimum points and headline regret table recomputed.
  4  code      numeric checks of the definitions the oracle relies on (horizon, reward coordinates, knee, bounds,
               bank draw, last command).

Usage (from /tmp, PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=/home/user/bca):
    /root/bcavenv/bin/python /home/user/bca/outputs/shift/oracle/check/check_oracle.py all
or one phase: code, optimum, nonconvex, points, report. CHECK_WORKERS sets the process count (6).
"""

import os
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent
ROOT = ORACLE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import thruster_chain as TC  # noqa: E402

WORKERS = int(os.environ.get("CHECK_WORKERS", "6"))
L, DA, DS, DX = 100, 4, 10, 13
WORLDS = ("stall", "no_stall")
SITS = TC.SITUATIONS
BOUND = {"stall": TC.KNEE, "no_stall": 1.0}
SHRINK = 1.0 - 1e-12   # JAX gradients are taken at (1 - 1e-12) u: strictly inside the knee / the +-1 clip, where
                       # autodiff of jnp.clip at an exact tie would return half (or, at the knee, a wrong-signed) slope
GATE = ROOT / "runs" / "shift" / "68d9e8779e148b7b"
PREREG = ROOT / "runs" / "shift" / "preregistration.json"
STAGE2 = ROOT / "runs" / "shift_explore" / "stage2" / "policy"
OUT = HERE / "cache"
SPREAD = np.arange(5, 1000, 10)          # 100 starts spread over each bank
HARD = 25                                # plus the 25 with the most at-limit commands in the oracle's plan
NC_STARTS = 20                           # nonconvex: starts per situation (taken from the item-1 set)
NC_INITS = ("zero", "uniform_1", "uniform_2", "uniform_3", "past_knee", "full_throttle_pattern", "all_plus_one",
            "check_optimum_pushed")
NC_MAXITER = 3000
NC_SAMPLES = 500


def phys(world):
    return TC.world_physics(world, 1.5)


def wpath(name):
    p = (HERE / name).resolve()
    if HERE != p and HERE not in p.parents:
        raise SystemExit(f"refused: {p} is outside {HERE}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json(name, payload):
    wpath(name).write_text(json.dumps(plain(payload), indent=1, sort_keys=True), encoding="utf8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pool_map(fn, jobs, workers=None):
    import multiprocessing as mp

    workers = min(int(workers or WORKERS), max(1, len(jobs)))
    if workers <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=1)


def oracle_plans():
    z = np.load(ORACLE / "cache" / "optimum100.npz")
    return {k: z[k] for k in z.files}


# ================================================================================================ own rollout

def replay(U, X, physics, record=False):
    """Own 100-step loop through the repository's numpy dynamics(): J = sum_{t=0}^{99} r_t (float64)."""
    X = np.asarray(X, np.float64)
    U = np.asarray(U, np.float64).reshape(len(X), L, DA)
    x, J, S = X.copy(), np.zeros(len(X)), [X.copy()]
    for t in range(L):
        x, r = TC.dynamics(x, U[:, t], physics)
        J = J + r
        if record:
            S.append(x)
    return (J, np.stack(S, 1)) if record else J


def own_banks():
    """The banks redrawn from the docstring's rule, without calling bank_starts / draw_starts."""
    out = {}
    for s, law in TC.START_LAWS.items():
        rng = np.random.default_rng([20261012, 2, 9])
        sv = law["kick"] * rng.standard_normal((1000, DS))
        c = rng.uniform(law["current"][0], law["current"][1], size=1000)
        amp = rng.uniform(law["wave"][0], law["wave"][1], size=1000)
        ph = rng.uniform(0.0, 2 * math.pi, size=1000)
        out[s] = np.concatenate([sv, c[:, None], (amp * np.cos(ph))[:, None], (amp * np.sin(ph))[:, None]], 1)
    return out


# ================================================================================================ JAX cost

def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


def make_cost(world):
    """u (400,) -> minus the 100-step return of a dynamics_jax rollout, evaluated at SHRINK * u (true thrust)."""
    jax = _jax()
    import jax.numpy as jnp

    ph = phys(world)

    def cost(u, x0):
        U = (u * SHRINK).reshape(L, DA)

        def body(x, a):
            x2, r = TC.dynamics_jax(x, a, ph)
            return x2, r

        _, rs = jax.lax.scan(body, x0, U)
        return -jnp.sum(rs)

    return cost


def lbfgsb(fun, u0, b, maxiter=20000):
    from scipy.optimize import minimize

    r = minimize(fun, u0, jac=True, method="L-BFGS-B", bounds=[(-b, b)] * (L * DA),
                 options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-16, gtol=1e-11, maxcor=50))
    return np.clip(r.x, -b, b), r


# ================================================================================================ item 1

def job_box(job):
    """My own box optimum for some starts of one (world, situation)."""
    world, sit, idx = job
    jax = _jax()
    import jax.numpy as jnp

    vg = jax.jit(jax.value_and_grad(make_cost(world)))
    b = BOUND[world]
    X = TC.bank_starts(sit)[idx]
    U, info = np.zeros((len(idx), L * DA)), []
    for i, x0 in enumerate(X):
        xj = jnp.asarray(x0)

        def fun(u):
            f, g = vg(jnp.asarray(u), xj)
            return float(f), np.asarray(g, np.float64)

        t0 = time.perf_counter()
        u, nit, rounds = np.zeros(L * DA), 0, 0
        for rounds in range(1, 5):                        # warm restarts until the certificate is tight
            u, r = lbfgsb(fun, u, b)
            nit += int(r.nit)
            f, g = fun(u)
            fw = float(np.sum(g * u + b * np.abs(g)))
            if fw / f < 1e-12:
                break
        pg = float(np.abs(u - np.clip(u - g, -b, b)).max())
        U[i] = u
        info.append(dict(start=int(idx[i]), iterations=nit, rounds=rounds, fw_gap=fw, fw_gap_relative=fw / f,
                         projected_gradient=pg, seconds=time.perf_counter() - t0))
    return world, sit, idx, U, info


def job_cert(job):
    """Independent certificate of the oracle's 1,000 plans of one (world, situation): Frank-Wolfe gap from JAX."""
    world, sit = job
    jax = _jax()
    import jax.numpy as jnp

    vgb = jax.jit(jax.vmap(jax.value_and_grad(make_cost(world))))
    U = oracle_plans()[f"U_{world}_{sit}"]
    X = TC.bank_starts(sit)
    b = BOUND[world]
    f, g = vgb(jnp.asarray(U), jnp.asarray(X))
    f, g = np.asarray(f, np.float64), np.asarray(g, np.float64)
    fw = np.sum(g * U + b * np.abs(g), 1)
    pg = np.abs(U - np.clip(U - g, -b, b)).max(1)
    return world, sit, fw, f, pg


def phase_optimum():
    opt = oracle_plans()
    jobs, chosen = [], {}
    for w in WORLDS:
        for s in SITS:
            Uo = opt[f"U_{w}_{s}"]
            at = (np.abs(Uo) >= BOUND[w] - 1e-9).sum(1)
            hard = [int(i) for i in np.argsort(-at, kind="stable") if int(i) not in set(SPREAD.tolist())][:HARD]
            idx = np.array(sorted(set(SPREAD.tolist()) | set(hard)))
            chosen[(w, s)] = (idx, hard)
            for k in range(0, len(idx), 25):
                jobs.append((w, s, idx[k:k + 25]))
    t0 = time.perf_counter()
    res = pool_map(job_box, jobs)
    certs = pool_map(job_cert, [(w, s) for w in WORLDS for s in SITS])
    seconds = time.perf_counter() - t0
    arrays, per = {}, {}
    mine = {}
    for w, s, idx, U, info in res:
        mine.setdefault((w, s), []).append((idx, U, info))
    for w in WORLDS:
        for s in SITS:
            parts = mine[(w, s)]
            idx = np.concatenate([p[0] for p in parts])
            U = np.concatenate([p[1] for p in parts])
            info = [x for p in parts for x in p[2]]
            X = TC.bank_starts(s)[idx]
            Uo = opt[f"U_{w}_{s}"][idx]
            Jm, Sm = replay(U, X, phys(w), record=True)
            Jo = replay(Uo, X, phys(w))
            rel = (Jm - Jo) / np.abs(Jo)                 # > 0: my plan is better than theirs
            hard = chosen[(w, s)][1]
            arrays[f"idx_{w}_{s}"], arrays[f"U_{w}_{s}"] = idx, U
            arrays[f"J_mine_{w}_{s}"], arrays[f"J_oracle_{w}_{s}"] = Jm, Jo
            per[f"{w}/{s}"] = dict(
                starts=len(idx), hard_starts=hard,
                max_abs_relative_return_difference=float(np.abs(rel).max()),
                worst_start=int(idx[int(np.argmax(np.abs(rel)))]),
                mine_better_by_max_relative=float(rel.max()), theirs_better_by_max_relative=float(-rel.min()),
                max_abs_return_difference=float(np.abs(Jm - Jo).max()),
                max_command_difference=float(np.abs(U - Uo).max()),
                mine_max_abs_command=float(np.abs(U).max()), mine_last_command_max_abs=float(np.abs(U[:, -DA:]).max()),
                mine_past_knee_commands=int(np.sum(np.abs(U) > TC.KNEE)),
                mine_fw_gap_relative_max=float(max(x["fw_gap_relative"] for x in info)),
                mine_projected_gradient_max=float(max(x["projected_gradient"] for x in info)),
                mine_iterations_mean=float(np.mean([x["iterations"] for x in info])),
                mine_rounds_max=int(max(x["rounds"] for x in info)),
                mean_return_mine=float(Jm.mean()), mean_return_oracle_same_starts=float(Jo.mean()))
    cert = {}
    for w, s, fw, f, pg in certs:
        U = opt[f"U_{w}_{s}"]
        cert[f"{w}/{s}"] = dict(fw_gap_max=float(fw.max()), fw_gap_relative_max=float((fw / f).max()),
                                projected_gradient_max=float(pg.max()), within_bounds=bool(np.all(np.abs(U) <= BOUND[w])),
                                max_abs_command=float(np.abs(U).max()), last_command_max_abs=float(np.abs(U[:, -DA:]).max()))
        arrays[f"certfw_{w}_{s}"] = fw
    np.savez_compressed(wpath("cache/optimum_check.npz"), **arrays)
    write_json("cache/optimum_check.json", dict(per=per, oracle_certificate=cert, seconds=seconds, workers=WORKERS))


# ================================================================================================ item 2

def job_nonconvex(job):
    sit, idx, Ubox, seed = job
    jax = _jax()
    import jax.numpy as jnp

    ph = phys("stall")
    knee = TC.KNEE
    vg = jax.jit(jax.value_and_grad(make_cost("stall")))
    X = TC.bank_starts(sit)[idx]
    rng = np.random.default_rng(seed)
    K = _own_gain()
    pattern = -K[:, TC.CURRENT] / abs(K[0, TC.CURRENT])
    rows = []
    for i, x0 in enumerate(X):
        ub = Ubox[i]
        cstar = -float(replay(ub[None], x0[None], ph)[0])
        at = np.abs(ub) >= knee - 1e-9
        # (b) random full-range plans, with and without the dominance map
        n4 = NC_SAMPLES // 5
        kinds = [rng.uniform(-1, 1, (n4, L * DA)),
                 np.sign(rng.uniform(-1, 1, (n4, L * DA))) * rng.uniform(knee, 1.0, (n4, L * DA)),
                 np.clip(ub + rng.choice([0.01, 0.05, 0.2], (n4, 1)) * rng.standard_normal((n4, L * DA)), -1, 1),
                 np.where(at & (rng.uniform(size=(n4, L * DA)) < 0.7),
                          np.sign(ub) * rng.uniform(knee, 1.0, (n4, L * DA)), ub),
                 np.where(rng.uniform(size=(n4, L * DA)) < 0.3, np.sign(ub + 1e-300) * rng.uniform(0.79, 1.0, (n4, L * DA)),
                          ub)]
        P = np.vstack(kinds)
        Pd = TC.thrust(np.clip(P, -1, 1), ph)
        Jp, Sp = replay(P, np.repeat(x0[None], len(P), 0), ph, record=True)
        Jd, Sd = replay(Pd, np.repeat(x0[None], len(P), 0), ph, record=True)
        rec = dict(start=int(idx[i]), box_optimum_cost=cstar, samples=len(P),
                   samples_using_past_knee=int(np.sum(np.any(np.abs(P) > knee, 1))),
                   samples_mapped_states_identical=bool(np.array_equal(Sp, Sd)),
                   samples_mapped_never_costlier=bool(np.all(-Jd <= -Jp)),
                   samples_mapped_strictly_cheaper_when_past_knee=bool(np.all((-Jd < -Jp)[np.any(np.abs(P) > knee, 1)])),
                   samples_min_relative_to_optimum=float((-Jp.max() - cstar) / cstar),
                   samples_below_optimum=int(np.sum(-Jp < cstar * (1 - 1e-12))))
        # (c) local searches over [-1, 1] with the true thrust
        inits = {
            "zero": np.zeros(L * DA),
            "uniform_1": rng.uniform(-1, 1, L * DA), "uniform_2": rng.uniform(-1, 1, L * DA),
            "uniform_3": rng.uniform(-1, 1, L * DA),
            "past_knee": np.sign(rng.uniform(-1, 1, L * DA)) * rng.uniform(knee, 1.0, L * DA),
            "full_throttle_pattern": np.clip(np.tile(pattern, L) + 0.03 * rng.standard_normal(L * DA), -1, 1),
            "all_plus_one": np.ones(L * DA),
            "check_optimum_pushed": np.where(at | (rng.uniform(size=L * DA) < 0.1),
                                             np.sign(ub + 1e-300) * rng.uniform(knee + 1e-3, 1.0, L * DA), ub),
        }
        xj = jnp.asarray(x0)

        def fun(u):
            f, g = vg(jnp.asarray(u), xj)
            return float(f), np.asarray(g, np.float64)

        for name in NC_INITS:
            t0 = time.perf_counter()
            u, r = lbfgsb(fun, inits[name], 1.0, maxiter=NC_MAXITER)
            c = -float(replay(u[None], x0[None], ph)[0])
            ud = TC.thrust(u, ph)
            cd = -float(replay(ud[None], x0[None], ph)[0])
            up, _ = lbfgsb(fun, ud, knee)                    # polish inside the knee box from the mapped plan
            cp = -float(replay(up[None], x0[None], ph)[0])
            rec[name] = dict(relative=(c - cstar) / cstar, mapped_relative=(cd - cstar) / cstar,
                             polished_relative=(cp - cstar) / cstar, past_knee_commands=int(np.sum(np.abs(u) > knee)),
                             iterations=int(r.nit), seconds=time.perf_counter() - t0)
        rows.append(rec)
    return sit, rows


def elementwise_thrust_check():
    """(a) on every float32 command in [0.5, 1] (both signs), 10 million float64 commands in [-1, 1] and the 2,001
    float64 neighbours of the knee: |thrust| <= knee, |thrust| <= |a| (strictly past the knee), thrust(thrust(a)) ==
    thrust(a) bit for bit, and thrust(a) == a at and below the knee."""
    ph = phys("stall")
    knee = TC.KNEE
    lo, hi = np.float32(0.5).view(np.uint32), np.float32(1.0).view(np.uint32)
    f32 = np.arange(lo, hi + 1, dtype=np.uint32).view(np.float32).astype(np.float64)
    near = [knee]
    a, b_ = knee, knee
    for _ in range(1000):
        a, b_ = np.nextafter(a, 2.0), np.nextafter(b_, 0.0)
        near += [a, b_]
    rnd = np.random.default_rng(99).uniform(-1, 1, 10_000_000)
    out = {}
    for name, v in (("float32_0.5_to_1", np.concatenate([f32, -f32])), ("knee_neighbours", np.array(near + [-x for x in near])),
                    ("random_float64", rnd)):
        th = TC.thrust(v, ph)
        th2 = TC.thrust(th, ph)
        past = np.abs(v) > knee
        out[name] = dict(values=int(v.size), past_knee=int(past.sum()),
                         max_abs_thrust=float(np.abs(th).max()), thrust_within_knee=bool(np.all(np.abs(th) <= knee)),
                         thrust_not_above_command=bool(np.all(np.abs(th) <= np.abs(v))),
                         strictly_below_past_knee=bool(np.all(np.abs(th[past]) < np.abs(v[past]))),
                         idempotent_bitwise=bool(np.array_equal(th2, th)),
                         identity_at_or_below_knee=bool(np.array_equal(th[~past], v[~past])),
                         same_sign=bool(np.all(np.sign(th) == np.sign(v))),
                         min_abs_thrust_past_knee=float(np.abs(th[past]).min()) if past.any() else None)
    out["knee"] = dict(KNEE=TC.KNEE, float32_of_0_8=float(np.float32(0.8)), thrust_at_knee=float(TC.thrust(TC.KNEE, ph)),
                       stalled_at_knee=bool(TC.stalled(np.array([TC.KNEE]), ph)[0]),
                       thrust_at_full=float(TC.thrust(1.0, ph)), thrust_at_0_9=float(TC.thrust(0.9, ph)))
    return out


def phase_nonconvex():
    mine = np.load(HERE / "cache" / "optimum_check.npz")
    jobs = []
    for k, s in enumerate(SITS):
        idx = mine[f"idx_stall_{s}"]
        U = mine[f"U_stall_{s}"]
        pick = np.linspace(0, len(idx) - 1, NC_STARTS).round().astype(int)
        for j in range(0, NC_STARTS, 4):
            p = pick[j:j + 4]
            jobs.append((s, idx[p], U[p], 4242 + 10 * k + j))
    t0 = time.perf_counter()
    res = pool_map(job_nonconvex, jobs)
    rows = [r for _, rr in res for r in rr]
    elem = elementwise_thrust_check()
    summary = dict(starts=len(rows), elementwise=elem,
                   samples=dict(total=sum(r["samples"] for r in rows),
                                using_past_knee=sum(r["samples_using_past_knee"] for r in rows),
                                mapped_states_identical_all=all(r["samples_mapped_states_identical"] for r in rows),
                                mapped_never_costlier_all=all(r["samples_mapped_never_costlier"] for r in rows),
                                mapped_strictly_cheaper_when_past_knee_all=all(
                                    r["samples_mapped_strictly_cheaper_when_past_knee"] for r in rows),
                                below_optimum=sum(r["samples_below_optimum"] for r in rows),
                                min_relative_to_optimum=min(r["samples_min_relative_to_optimum"] for r in rows)))
    for name in NC_INITS:
        rel = np.array([r[name]["relative"] for r in rows])
        mrel = np.array([r[name]["mapped_relative"] for r in rows])
        prel = np.array([r[name]["polished_relative"] for r in rows])
        summary[name] = dict(min_relative=float(rel.min()), median_relative=float(np.median(rel)),
                             max_relative=float(rel.max()), runs_below_optimum=int(np.sum(rel < -1e-10)),
                             runs_ending_past_knee=int(sum(r[name]["past_knee_commands"] > 0 for r in rows)),
                             mapped_never_costlier=bool(np.all(mrel <= rel + 1e-15)),
                             polished_max_abs_relative=float(np.abs(prel).max()),
                             polished_min_relative=float(prel.min()))
    summary["seconds"] = time.perf_counter() - t0
    write_json("cache/nonconvex_check.json", dict(summary=summary, rows=rows))


# ================================================================================================ item 3

def _own_gain():
    """The discounted (0.98) LQR gain of the rising-thrust model, state cost I on the 10 floats, throttle cost 0.3 I,
    from scipy's Riccati solver (recorder.py's docstring definition)."""
    from scipy import linalg

    A, B = (np.array(m, np.float64) for m in TC.matrices(TC.STALL))
    g, R = 0.98, 0.3 * np.eye(DA)
    Qc = np.zeros((DX, DX))
    Qc[:DS, :DS] = np.eye(DS)
    P = linalg.solve_discrete_are(math.sqrt(g) * A, math.sqrt(g) * B, Qc, R)
    return np.linalg.solve(R + g * B.T @ P @ B, g * B.T @ P @ A)


def own_anchor_returns(world, X, K):
    """Mean-return building blocks: per-start 100-step returns of my recording / stall-aware / rising-thrust."""
    ph = phys(world)
    rng = np.random.default_rng([7, 1])
    noise = rng.standard_normal((len(X), L, DA))
    coin = rng.uniform(size=(len(X), L))
    pattern = -K[:, TC.CURRENT] / abs(K[0, TC.CURRENT])
    out = {}
    for name in ("recording", "stall_aware", "rising_thrust"):
        x, J = X.copy(), np.zeros(len(X))
        for t in range(L):
            if name == "recording":
                u = -(0.25 * (x[:, :DS] @ K[:, :DS].T) + x[:, DS:] @ K[:, DS:].T) + 0.3 * noise[:, t]
                hand = 0.8 * np.tanh(u / 0.8)
                full = np.clip(pattern + 0.03 * noise[:, t], -1, 1)
                press = (x[:, TC.CURRENT] + x[:, TC.WAVE1] > 0.565) & (coin[:, t] < 0.5)
                a = np.where(press[:, None], full, hand)
            else:
                a = np.clip(-(x @ K.T), -0.8 if name == "stall_aware" else -1.0, 0.8 if name == "stall_aware" else 1.0)
            x, r = TC.dynamics(x, a, ph)
            J = J + r
        out[name] = J
    return out


def _restore(path):
    from flax import serialization

    return serialization.msgpack_restore(Path(path).read_bytes())


def _mlp(params, mean, std):
    p = params.get("params", params)
    names = sorted(p, key=lambda k: int(str(k).rsplit("_", 1)[1]))
    W = [np.asarray(p[k]["kernel"], np.float64) for k in names]
    b = [np.asarray(p[k]["bias"], np.float64) for k in names]
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)

    def act(x):
        h = (np.asarray(x, np.float32).astype(np.float64) - mean) / std
        h = np.maximum(h @ W[0] + b[0], 0.0)
        h = np.maximum(h @ W[1] + b[1], 0.0)
        return np.tanh(h @ W[2] + b[2])

    return act


def job_actor(job):
    """Per-start 100-step returns of one actor on the four banks (own loop, own forward pass)."""
    world, params, mean, std = job["world"], job["params"], job["mean"], job["std"]
    act = _mlp(params, mean, std)
    ph = phys(world)
    out = {}
    for s in SITS:
        x = TC.bank_starts(s)
        J = np.zeros(len(x))
        for t in range(L):
            x, r = TC.dynamics(x, np.clip(act(x), -1, 1), ph)
            J = J + r
        out[s] = J
    return job["label"], out


def collect_actors():
    """Every saved actor with recorded points: the gate (alpha 2.5 and 5, seeds 0-4) and Stage 2 (seeds 0-1)."""
    norms, files = {}, {}
    for w in WORLDS:
        for r in range(5):
            rec = read_json(GATE / "clone" / "base" / w / f"seed{r}" / "record.json")
            d = rec["result"]["data"]
            norms[(w, r)] = (np.asarray(d["obs_mean"], np.float32), np.asarray(d["obs_std"], np.float32))
    jobs, records = [], []
    for alpha in ("2.5", "5"):
        for r in range(5):
            d = GATE / "harm_return" / "base" / f"alpha{alpha}" / f"seed{r}"
            vals = read_json(d / "values.json")
            files[str(d / "actors.msgpack")] = sha256(d / "actors.msgpack") == vals["actors_sha256"]
            acts = _restore(d / "actors.msgpack")
            for name in vals["actor_names"]:
                if vals["failed"].get(name) or vals["evaluations"].get(name) is None:
                    continue
                w = "no_stall" if name.startswith("no_stall") else "stall"
                label = f"gate:alpha{alpha}:seed{r}:{name}"
                jobs.append(dict(label=label, world=w, params=acts[name], mean=norms[(w, r)][0], std=norms[(w, r)][1]))
                records.append(dict(label=label, world=w, rep=r, source="gate", alpha=alpha, name=name,
                                    family=f"gate alpha {alpha}: {name}",
                                    recorded={s: vals["evaluations"][name][s]["points"] for s in SITS}))
    for r in (0, 1):
        for w in WORLDS:
            d = STAGE2 / f"seed{r}" / ("" if w == "stall" else "no_stall")
            lock, vals = read_json(d / "lock.json"), read_json(d / "values.json")
            for nm in ("lock.json", "values.json"):
                side = (d / f"{nm}.sha256").read_text(encoding="utf8").split()[0]
                files[str(d / nm)] = hashlib.sha256((d / nm).read_bytes()).hexdigest() == side
            files[str(d / lock["actors_file"])] = sha256(d / lock["actors_file"]) == lock["actors_file_sha256"]
            acts = _restore(d / lock["actors_file"])
            matched = {v: k for k, v in (lock.get("matched_names") or {}).items() if v}
            standin = (lock.get("standin") or {}).get("arm")
            for name in lock["scored_names"]:
                if name not in vals["evaluations"]:
                    continue
                if name in matched:
                    fam = f"stage2:{matched[name]} (matched)"
                elif name == standin or name == "standin_locked":
                    fam = "stage2:standin (matched)" if w == "stall" else "stage2:standin (locked)"
                else:
                    fam = f"stage2:{name}"
                label = f"stage2:seed{r}:{w}:{name}"
                jobs.append(dict(label=label, world=w, params=acts[name], mean=norms[(w, r)][0], std=norms[(w, r)][1]))
                records.append(dict(label=label, world=w, rep=r, source="stage2", name=name, family=fam,
                                    recorded={s: vals["evaluations"][name][s]["points"] for s in SITS}))
    return jobs, records, files


def phase_points():
    t0 = time.perf_counter()
    pre = read_json(PREREG)
    R = read_json(ORACLE / "results.json")
    K = _own_gain()
    from experiments.shift import recorder as RC
    gain_diff = float(np.abs(K - RC.textbook_gain(TC.STALL)).max())
    # (a) anchors
    anchors, anchor_cmp, ref_J = {}, {}, {}
    for w in WORLDS:
        anchors[w] = {}
        for s in SITS:
            X = TC.bank_starts(s)
            Js = own_anchor_returns(w, X, K)
            ref_J[(w, s)] = Js
            top = "stall_aware" if w == "stall" else "rising_thrust"
            anchors[w][s] = dict(recording=float(Js["recording"].mean()), top=float(Js[top].mean()))
            pa = pre["anchors"][f"base/{w}"][s]
            oa = R["anchors"][w][s]
            anchor_cmp[f"{w}/{s}"] = dict(
                mine=anchors[w][s], prereg=dict(recording=pa["recording"], top=pa["top"]),
                mine_minus_prereg=max(abs(anchors[w][s]["recording"] - pa["recording"]), abs(anchors[w][s]["top"] - pa["top"])),
                oracle_minus_prereg=max(abs(oa["recording"] - pa["recording"]), abs(oa["top"] - pa["top"])),
                prereg_top_controller=pa["top_controller"])
    pa = {w: {s: pre["anchors"][f"base/{w}"][s] for s in SITS} for w in WORLDS}

    def pts(J, w, s):
        a = pa[w][s]
        return 100.0 * (np.asarray(J) - a["recording"]) / (a["top"] - a["recording"])

    # (b) actors re-scored
    jobs, records, files = collect_actors()
    res = dict(pool_map(job_actor, jobs))
    diffs, worst = [], None
    for rec in records:
        J = res[rec["label"]]
        rec["mine"] = {s: float(pts(J[s], rec["world"], s).mean()) for s in SITS}
        for s in SITS:
            d = abs(rec["mine"][s] - rec["recorded"][s])
            diffs.append(d)
            if worst is None or d > worst[0]:
                worst = (d, rec["label"], s)
    # (c) the oracle's optimum points and regret table, from its plans replayed in my loop
    opt = oracle_plans()
    J_opt, opt_pts, opt_cmp = {}, {}, {}
    for w in WORLDS:
        for s in SITS:
            J_opt[(w, s)] = replay(opt[f"U_{w}_{s}"], TC.bank_starts(s), phys(w))
            opt_pts[(w, s)] = float(pts(J_opt[(w, s)], w, s).mean())
            opt_cmp[f"{w}/{s}"] = dict(points_mine=opt_pts[(w, s)], points_oracle=R["optimum"][w][s]["points"],
                                       return_mine=float(J_opt[(w, s)].mean()),
                                       return_oracle=R["optimum"][w][s]["return_"]["mean"])
    # upper bound: every re-scored actor and my reference controllers at or below the optimum, start by start
    min_regret = math.inf
    for rec in records:
        for s in SITS:
            min_regret = min(min_regret, float((J_opt[(rec["world"], s)] - res[rec["label"]][s]).min()))
    for (w, s), Js in ref_J.items():
        for J in Js.values():
            min_regret = min(min_regret, float((J_opt[(w, s)] - J).min()))
    # family regret means (as the oracle groups them)
    fams = {}
    for rec in records:
        fams.setdefault((rec["world"], rec["family"]), {})
        fams[(rec["world"], rec["family"])].setdefault(rec["rep"], rec)
    fam_out = {}
    for (w, fam), byrep in fams.items():
        recs = list(byrep.values())
        key = f"{w} | {fam}"
        e = R["families"].get(key)
        fam_out[key] = dict(n=len(recs), per_situation={})
        for s in SITS:
            mine = float(np.mean([opt_pts[(w, s)] - r["mine"][s] for r in recs]))
            theirs = None if e is None else e[s]["regret_points_mean"]
            fam_out[key]["per_situation"][s] = dict(regret_points_mine=mine, regret_points_oracle=theirs,
                                                    difference=None if theirs is None else mine - theirs)
    for w in WORLDS:
        top = "stall_aware" if w == "stall" else "rising_thrust"
        for name in ("recording", top):
            key = f"{w} | reference:{name}"
            e = R["families"].get(key)
            fam_out[key] = dict(n=1, per_situation={
                s: dict(regret_points_mine=opt_pts[(w, s)] - float(pts(ref_J[(w, s)][name], w, s).mean()),
                        regret_points_oracle=None if e is None else e[s]["regret_points_mean"]) for s in SITS})
            for s in SITS:
                d = fam_out[key]["per_situation"][s]
                d["difference"] = None if d["regret_points_oracle"] is None else d["regret_points_mine"] - d["regret_points_oracle"]
    fam_max = max(abs(d["difference"]) for f in fam_out.values() for d in f["per_situation"].values()
                  if d["difference"] is not None)
    write_json("cache/points_check.json", dict(
        textbook_gain_mine_minus_repo=gain_diff, anchors=anchor_cmp,
        anchors_max_mine_minus_prereg=max(v["mine_minus_prereg"] for v in anchor_cmp.values()),
        anchors_max_oracle_minus_prereg=max(v["oracle_minus_prereg"] for v in anchor_cmp.values()),
        actors_rescored=len(records), recorded_numbers=len(diffs), recorded_points_max_abs_difference=float(max(diffs)),
        recorded_points_worst=worst, files_sha256_ok=files, all_files_ok=all(files.values()),
        optimum_points=opt_cmp,
        optimum_points_max_abs_difference=max(abs(v["points_mine"] - v["points_oracle"]) for v in opt_cmp.values()),
        every_controller_at_or_below_optimum_min_regret=min_regret, families=fam_out,
        families_max_abs_difference=fam_max, records=[{k: v for k, v in r.items()} for r in records],
        seconds=time.perf_counter() - t0))


# ================================================================================================ item 4

def phase_code():
    out = {}
    banks = own_banks()
    out["bank_redraw_equal"] = {s: bool(np.array_equal(banks[s], TC.bank_starts(s))) for s in SITS}
    # reward: minus (sum of squares of coordinates 0..9 of the state BEFORE the step + 0.3 |clipped a|^2)
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0, 2, (5000, DS)), rng.uniform(0, 1.5, (5000, 1)), rng.uniform(-0.7, 0.7, (5000, 2))], 1)
    a = rng.uniform(-1.5, 1.5, (5000, DA))
    for w in WORLDS:
        _, r = TC.dynamics(x, a, phys(w))
        ac = np.clip(a, -1, 1)
        ref = -(np.sum(x[:, :DS] ** 2, 1) + 0.3 * np.sum(ac ** 2, 1))
        out[f"reward_definition_max_relative_{w}"] = float(np.max(np.abs(r - ref) / np.abs(ref)))
        wrong = -(np.sum(x ** 2, 1) + 0.3 * np.sum(ac ** 2, 1))
        out[f"reward_if_all_13_counted_would_differ_{w}"] = float(np.max(np.abs(r - wrong)))
    # horizon: their reported returns are sum_{t=0}^{99} r_t, not a shifted window
    opt = oracle_plans()
    R = read_json(ORACLE / "results.json")
    hz = {}
    for w in WORLDS:
        for s in SITS:
            U = opt[f"U_{w}_{s}"]
            X = TC.bank_starts(s)
            J, S = replay(U, X, phys(w), record=True)
            Ur = U.reshape(-1, L, DA)
            energy = 0.3 * np.sum(Ur ** 2, (1, 2))
            J_shift = -(np.sum(S[:, 1:L + 1, :DS] ** 2, (1, 2)) + energy)      # s_1..s_100 charged instead
            J_101 = J - np.sum(S[:, L, :DS] ** 2, 1)                            # s_0..s_100 charged
            hz[f"{w}/{s}"] = dict(reported=R["optimum"][w][s]["return_"]["mean"], t0_to_99=float(J.mean()),
                                  s1_to_100=float(J_shift.mean()), s0_to_100=float(J_101.mean()),
                                  max_abs_command=float(np.abs(U).max()), bound=BOUND[w],
                                  commands_above_bound=int(np.sum(np.abs(U) > BOUND[w])),
                                  last_command_max_abs=float(np.abs(Ur[:, -1]).max()),
                                  share_at_bound=float(np.mean(np.abs(U) >= BOUND[w] - 1e-9)),
                                  thruster1_at_bound_share=float(np.mean(np.abs(Ur[:, :, 0]) >= BOUND[w] - 1e-9)))
    out["horizon"] = hz
    out["knee"] = dict(KNEE=TC.KNEE, repr=repr(TC.KNEE), equals_float32_0_8=TC.KNEE == float(np.float32(0.8)),
                       oracle_stall_bound=read_json(ORACLE / "results.json")["optimum"]["stall"]["recorded"]["bound"])
    write_json("cache/code_check.json", out)


# ================================================================================================ report

def phase_report():
    code = read_json(HERE / "cache/code_check.json")
    optc = read_json(HERE / "cache/optimum_check.json")
    nc = read_json(HERE / "cache/nonconvex_check.json")["summary"]
    pc = read_json(HERE / "cache/points_check.json")
    per = optc["per"]
    item1 = dict(
        starts_checked=sum(v["starts"] for v in per.values()),
        worst_abs_relative_return_difference=max(v["max_abs_relative_return_difference"] for v in per.values()),
        worst_case=max(per.items(), key=lambda kv: kv[1]["max_abs_relative_return_difference"])[0],
        mine_better_by_max_relative=max(v["mine_better_by_max_relative"] for v in per.values()),
        theirs_better_by_max_relative=max(v["theirs_better_by_max_relative"] for v in per.values()),
        max_command_difference=max(v["max_command_difference"] for v in per.values()),
        mine_fw_gap_relative_max=max(v["mine_fw_gap_relative_max"] for v in per.values()),
        oracle_all_8000_fw_gap_relative_max=max(v["fw_gap_relative_max"] for v in optc["oracle_certificate"].values()),
        oracle_all_8000_projected_gradient_max=max(v["projected_gradient_max"] for v in optc["oracle_certificate"].values()),
        oracle_all_within_bounds=all(v["within_bounds"] for v in optc["oracle_certificate"].values()))
    item2 = dict(elementwise=nc["elementwise"], samples=nc["samples"],
                 local_search={k: nc[k] for k in NC_INITS},
                 runs=nc["starts"] * len(NC_INITS),
                 runs_below_optimum=sum(nc[k]["runs_below_optimum"] for k in NC_INITS),
                 min_relative=min(nc[k]["min_relative"] for k in NC_INITS),
                 polished_max_abs_relative=max(nc[k]["polished_max_abs_relative"] for k in NC_INITS))
    item3 = dict(textbook_gain_mine_minus_repo=pc["textbook_gain_mine_minus_repo"],
                 anchors_max_mine_minus_prereg=pc["anchors_max_mine_minus_prereg"],
                 anchors_max_oracle_minus_prereg=pc["anchors_max_oracle_minus_prereg"],
                 actors_rescored=pc["actors_rescored"], recorded_numbers=pc["recorded_numbers"],
                 recorded_points_max_abs_difference=pc["recorded_points_max_abs_difference"],
                 all_files_ok=pc["all_files_ok"],
                 optimum_points_max_abs_difference=pc["optimum_points_max_abs_difference"],
                 every_controller_at_or_below_optimum_min_regret=pc["every_controller_at_or_below_optimum_min_regret"],
                 families_max_abs_difference=pc["families_max_abs_difference"])
    write_json("results.json", dict(item1_optimum=item1, item1_per=per, item2_nonconvex=item2, item3_points=item3,
                                    item4_code=code, optimum_points=pc["optimum_points"], families=pc["families"]))


PHASES = ("code", "optimum", "nonconvex", "points", "report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=PHASES + ("all",))
    args = ap.parse_args()
    for ph in (PHASES if args.phase == "all" else (args.phase,)):
        t0 = time.perf_counter()
        globals()[f"phase_{ph}"]()
        print(f"[{ph}] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
