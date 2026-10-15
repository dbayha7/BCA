#!/usr/bin/env python
"""ThrusterChain's exact optimal reference controller ("the optimum") and the regret of saved controllers against it.

Evaluator work only: no learner is trained and no BCA band is fitted. Every number is float64 simulation through the
repository's own experiments/shift code, which is imported and never edited. Outputs go only to this folder
(outputs/shift/oracle/); the run files under runs/ are only read.

1. What is being optimised (the headline objective)
---------------------------------------------------
For one start x0 the headline score of a controller is its 100-step undiscounted return
    J = sum_{t=0}^{99} r_t,   r_t = -(|s_t|^2 + 0.3 |a_t|^2),
summed exactly as truth.rollout sums it (learners.evaluate_policy and acceptance._evaluate call truth.rollout): the reward
of step t uses the 10 float coordinates s_t of the state BEFORE the step and the clipped command a_t; the state after the
last step, s_100, is never charged. Points are 100 (J - J_recording) / (J_top - J_recording) per start, averaged over the
bank, with the gate's anchors read from the pre-registration (truth.points, acceptance.anchors_of).

The world (thruster_chain.py): x' = A x + B thrust(a), x = (s, current c, wave w1, w2), no noise. The current and the wave
move on their own (linearly; no command changes them). thrust(a) = a while |a| <= knee = float32(0.8) =
0.800000011920929; past the knee in the stalling world thrust(a) = sign(a) (knee - 1.5 (|a| - knee)), i.e. LESS push for
MORE throttle, while the energy cost 0.3 a^2 keeps rising. In the no-stall world thrust(a) = a on all of [-1, 1].

Why the stalling world's optimum never goes past the knee: for any command a with |a| > knee, the command a' = thrust(a)
gives exactly the same push (so exactly the same states) with |a'| < knee < |a| (so a strictly smaller energy cost).
Every plan that uses a past-knee command is therefore beaten by one that does not, and the best plan lies in
|a_k| <= knee, where thrust(a) = a. There the problem is linear-quadratic with box bounds, which is convex: a local
optimum is the global optimum. Section (c) below checks this numerically with the true (non-convex) thrust anyway.

2. How the optimum is computed
------------------------------
Because the dynamics are linear and the same at every step, the 99 charged future float states are a linear function of
the 400 commands u = (a_0, ..., a_99):
    (s_1, ..., s_99) = Ms u + ms(x0),     ms(x0) = (S A^t x0)_{t=1..99},  Ms[t, k] = S A^(t-1-k) B for k < t,
so the cost -J = |s_0|^2 + ||Ms u + ms(x0)||^2 + 0.3 ||u||^2 = |s_0|^2 + ||M u + m(x0)||^2 with M = [Ms; sqrt(0.3) I],
the SAME matrix M for every start and for both worlds (only the bounds differ: +-knee or +-1). Each start is one bounded
least-squares problem, solved with scipy.optimize.lsq_linear(method="bvls", tol=1e-14), an exact active-set method. To
make it fast the 1,390 x 400 matrix M is first reduced by one QR factorisation M = Q R (R is 400 x 400, condition number
about 16): ||M u + m||^2 = ||R u + Q^T m||^2 + (a constant), so both problems have the same minimiser. The objective
that is reported and checked is always recomputed from M itself and by replaying the commands through the world.
The last command a_99 only moves s_100, which is never charged, so the optimum sets it to 0.

The plan is open loop (a fixed list of 100 commands per start). Nothing in the world is random, so no feedback
controller can do better from the same start: whatever it does produces one command list, and the optimum is the best
of all command lists. The optimum is therefore an upper bound on every controller's return, start by start (checked).

3. Checks (validate, nonconvex)
-------------------------------
(a) replay: every optimal command list is replayed through thruster_chain.dynamics() (via truth.rollout) in float64; the
    summed reward must equal the least-squares objective to 1e-8 relative.
(b) optimality: the gradient g = 2 M^T (M u + m); projected-gradient norm |u - clip(u - g)|; the Frank-Wolfe gap
    sum_i (g_i u_i + b |g_i|), which by convexity is a guaranteed upper bound on (cost(u) - best cost); a sign check of
    the bound conditions; a second solver (lsq_linear "trf") on a subset; random feasible perturbations (and every
    single-command nudge) that must never lower the cost.
(c) stalling world, over the FULL range [-1, 1] with the true non-convex thrust: the one-sided slope of the cost for
    pushing each at-knee optimal command past the knee (all 4,000 starts; must be > 0); 2,000 sampled full-range plans
    per start on 400 starts (the optimum with at-knee commands pushed past the knee, with random commands replaced,
    plus noise, and uniform random plans), none of which may cost less, each also mapped through "replace each command
    by its own push" to confirm the dominance argument; and L-BFGS-B started at the optimum and at four other plans
    (at-knee commands pushed past the knee, at full throttle, the optimum plus noise, all-zero), at most 5,000
    iterations each: no run may find a lower cost.

4. Secondary objective (discounted)
-----------------------------------
The learners and truth.py value a controller by the discounted sum sum_{t=0}^{999} 0.98^t r_t (truth.HORIZON = 1,000
steps). Its optimum over 1,000-step plans is computed for the first 200 starts of each bank by an accelerated projected
gradient method (FISTA) in time-scaled variables v_t = 0.98^(t/2) a_t (which make the problem well conditioned:
condition number about 230), with the gradient from one forward and one backward pass through the linear world, and
certified by the same Frank-Wolfe gap. Controllers get their exact discounted value from truth.value (horizon 1,000).
The two objectives' rankings of the controllers are compared.

5. Controllers scored (score)
-----------------------------
- the reference controllers (recorder.py): the recording controller (0 points), the stall-aware controller (100 points
  in the stalling world), the rising-thrust controller (100 points in the no-stall world) and normal driving only;
- the gate's saved actors (runs/shift/68d9e8779e148b7b/harm_return/base/alpha{2.5,5}/seed{0..4}/actors.msgpack, SHA-256
  from values.json; the cloned actors of clone/base/<world>/seed<s>/ match the 'clone' entries);
- the v6 Stage 2 locked arms of seeds 0-1 (runs/shift_explore/stage2/policy/seed<s>/[no_stall/]actors.msgpack; the file
  SHA-256 and every actor's tree SHA-256 checked against lock.json); the scored names of each lock.
Every controller is scored with learners.evaluate_policy / evaluate_actor (the gate's own code path), and its points are
compared with the points recorded in its values.json. The actors' observation normalisation comes from
acceptance.load_seed(...).prepared("gap", ...) (the gate's verified loader) and is cross-checked against the SHA-256 in
the cloned actor's record.

6. Float32 deployment check (float32)
-------------------------------------
For saved trained actors: the deployed path (ThrusterChainEnv.reset(options={"state": x0}) and env.step, float32
observations, the float32 JAX actor, commands cast to float32 by env.step) against the float64 evaluator
(truth.MLPPolicy in truth.rollout), over the full four banks.

Usage (from /tmp, with PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=/home/user/bca):
    /root/bcavenv/bin/python /home/user/bca/outputs/shift/oracle/oracle.py all
or one phase at a time: solve, validate, nonconvex, discounted, score, float32, report (each phase caches its output
under cache/, and report writes results.json, results.md and timings.json). ORACLE_WORKERS sets the process count (8).
"""

import os
import sys

# no bytecode anywhere (a later integrity check fingerprints pinned folders), CPU JAX, one BLAS thread per process
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402

CACHE = HERE / "cache"
WORKERS = int(os.environ.get("ORACLE_WORKERS", "8"))

DS, DA, DX = TC.DS, TC.DA, TC.DX
LENGTH = TC.STALL.length                 # 100 steps per episode
WORLDS = ("stall", "no_stall")
SITUATIONS = TC.SITUATIONS               # recorded, strong_current, big_waves, big_kicks
VARIANT = "base"                         # the gate's outcome variant (stall slope 1.5)
STALL_SLOPE = 1.5
BOUND = {"stall": TC.KNEE, "no_stall": 1.0}
GAMMA = TC.STALL.gamma                   # 0.98
DISC_HORIZON = TR.HORIZON                # 1,000 steps, truth.py's horizon
DISC_STARTS = 200                        # the first 200 starts of each bank
BVLS_TOL = 1e-14
AT_BOUND = 1e-9                          # |a| >= bound - 1e-9 counts as "at the bound"
LATE_PEAK_FROM = 10                      # "late peak": the largest float deviation over steps 10..99 (after the
                                         # start's kick has been damped; the peak over all steps is the start itself)
GATE_DIGEST16 = "68d9e8779e148b7b"
GATE_ALPHAS = ("2.5", "5")
GATE_SEEDS = (0, 1, 2, 3, 4)
STAGE2_SEEDS = (0, 1)
STAGE2_POLICY = ROOT / "runs" / "shift_explore" / "stage2" / "policy"
NONCONVEX_STARTS = 100                   # per situation, stalling world
PERTURB_STARTS = 50                      # per world and situation
TRF_STARTS = 25                          # per world and situation
FLOAT32_ACTORS = (("gate", "5", 0, "gapped"), ("stage2", None, 0, "bca_k1"))   # (source, alpha, seed, name), stall


def physics_of(world):
    return TC.world_physics(world, STALL_SLOPE)


def out_path(name):
    """Every file this script writes goes under this folder (refuses anything else)."""
    p = (HERE / name).resolve()
    if HERE != p and HERE not in p.parents:
        raise SystemExit(f"refused: {p} is outside {HERE}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plain(x):
    """numpy -> JSON-safe Python, recursively (non-finite floats -> None)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(name, payload):
    p = out_path(name)
    p.write_text(json.dumps(plain(payload), indent=1, sort_keys=True), encoding="utf8")
    return p


def read_json(name):
    return json.loads((HERE / name).read_text(encoding="utf8"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json_sidecar(path):
    """A run-file JSON whose '<name>.sha256' sidecar must match (stage2_common.read_json's rule)."""
    path = Path(path)
    text = path.read_text(encoding="utf8")
    side = path.with_name(path.name + ".sha256").read_text(encoding="utf8").split()[0]
    if hashlib.sha256(text.encode("utf8")).hexdigest() != side:
        raise SystemExit(f"refused: {path} differs from its SHA-256 sidecar")
    return json.loads(text)


def pool_map(fn, jobs, workers=None):
    """Run fn over jobs in fresh 'spawn' processes (no fork after JAX starts), keeping the job order."""
    import multiprocessing as mp

    workers = min(int(workers or WORKERS), max(1, len(jobs)))
    if workers <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=1)


class Timer:
    def __init__(self):
        self.t = {}

    def run(self, name, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.t[name] = time.perf_counter() - t0
        print(f"[{name}] {self.t[name]:.1f}s", flush=True)
        return out


# ================================================================================================ the 100-step problem

class Plan100:
    """The matrices of the 100-step problem (the same for every start and for both worlds).

    free_response[t - 1] = S A^t (t = 1..99): how the start moves the charged float states with no commands.
    Ms (990 x 400): Ms[t-1 block, k block] = S A^(t-1-k) B for k < t: how command k moves float state t.
    M = [Ms; sqrt(0.3) I] (1,390 x 400): -J = |s_0|^2 + ||M u + m(x0)||^2. Q, R: the reduced QR factors of M."""

    def __init__(self, length=LENGTH):
        A, B = (np.array(m, np.float64) for m in TC.matrices(TC.STALL))
        A2, B2 = TC.matrices(TC.NO_STALL)
        assert np.array_equal(A, A2) and np.array_equal(B, B2), "the two worlds share A and B"
        self.A, self.B, self.L = A, B, int(length)
        L = self.L
        P = [np.eye(DX)]
        for _ in range(L):
            P.append(A @ P[-1])
        self.free_response = np.stack([P[t][:DS] for t in range(1, L)])
        Ms = np.zeros(((L - 1) * DS, L * DA))
        for t in range(1, L):
            for k in range(t):
                Ms[(t - 1) * DS:t * DS, k * DA:(k + 1) * DA] = (P[t - 1 - k] @ B)[:DS]
        self.Ms = Ms
        self.energy_weight = TC.STALL.throttle_cost
        self.M = np.vstack([Ms, math.sqrt(self.energy_weight) * np.eye(L * DA)])
        self.Q, self.R = np.linalg.qr(self.M)

    def ms(self, X):
        """(N, 13) starts -> (N, 990): the charged float states s_1..s_99 under all-zero commands."""
        X = np.asarray(X, np.float64)
        return np.einsum("tij,nj->nti", self.free_response, X).reshape(len(X), -1)

    def m(self, X):
        return np.concatenate([self.ms(X), np.zeros((len(X), self.L * DA))], 1)

    def cost(self, U, X):
        """-J from M itself: |s_0|^2 + ||M u + m||^2, U (N, 400)."""
        X = np.asarray(X, np.float64)
        res = np.asarray(U, np.float64) @ self.M.T + self.m(X)
        return np.sum(X[:, :DS] ** 2, 1) + np.sum(res ** 2, 1)

    def grad(self, U, X):
        res = np.asarray(U, np.float64) @ self.M.T + self.m(X)
        return 2.0 * res @ self.M

    def true_cost_grad(self, u, x0, physics):
        """Cost and gradient of ONE plan u (400,) over the full range with the world's TRUE thrust (non-convex in the
        stalling world). The derivative of thrust is 1 up to the knee (inclusive) and -slope past it."""
        th = TC.thrust(u, physics)
        res = self.Ms @ th + self.ms(x0[None])[0]
        c = float(x0[:DS] @ x0[:DS] + res @ res + self.energy_weight * u @ u)
        dth = np.where(np.abs(u) <= physics.thrust_knee, 1.0, -physics.thrust_slope)
        g = dth * (2.0 * (self.Ms.T @ res)) + 2.0 * self.energy_weight * u
        return c, g


_PLAN = {}


def plan100():
    if "p" not in _PLAN:
        _PLAN["p"] = Plan100()
    return _PLAN["p"]


class OpenLoop(RC.Controller):
    """A fixed command list per start: act(x, t) returns U[:, t] (U (N, L, 4)). Used to replay plans through
    truth.rollout, i.e. through thruster_chain.dynamics()."""

    def __init__(self, U, name="open-loop plan"):
        self.U = np.asarray(U, np.float64)
        self.name = name

    def act(self, x, t):
        return self.U[:, t]

    def __call__(self, x):
        raise TypeError("an open-loop plan needs the step index; call act(x, t)")


def fw_gap(U, G, b):
    """Frank-Wolfe gap per start: sum_i (g_i u_i + b |g_i|) >= cost(u) - best cost (convexity)."""
    return np.sum(G * U + b * np.abs(G), axis=-1)


def projected_gradient(U, G, b):
    return U - np.clip(U - G, -b, b)


def _solve_chunk(job):
    """BVLS on the reduced problem for some starts of one (world, situation)."""
    from scipy.optimize import lsq_linear

    world, situation, idx = job
    plan = plan100()
    X = TC.bank_starts(situation)[idx]
    b = BOUND[world]
    rhs = -(plan.m(X) @ plan.Q)
    U = np.empty((len(X), plan.L * DA))
    info = np.empty((len(X), 3))
    for i in range(len(X)):
        t0 = time.perf_counter()
        res = lsq_linear(plan.R, rhs[i], bounds=(-b, b), method="bvls", tol=BVLS_TOL, max_iter=20000)
        U[i] = np.clip(res.x, -b, b)
        info[i] = (res.status, res.nit, time.perf_counter() - t0)
    return world, situation, idx, U, info


def _trf_chunk(job):
    from scipy.optimize import lsq_linear

    world, situation, idx = job
    plan = plan100()
    X = TC.bank_starts(situation)[idx]
    b = BOUND[world]
    rhs = -(plan.m(X) @ plan.Q)
    U = np.empty((len(X), plan.L * DA))
    for i in range(len(X)):
        res = lsq_linear(plan.R, rhs[i], bounds=(-b, b), method="trf", lsq_solver="exact", tol=BVLS_TOL,
                         max_iter=20000)
        U[i] = np.clip(res.x, -b, b)
    return world, situation, idx, U


def phase_solve():
    """The 100-step optimum for all 8,000 bank starts -> cache/optimum100.npz."""
    plan = plan100()
    chunks = [(w, s, np.arange(i, min(i + 125, TC.BANK_SIZE))) for w in WORLDS for s in SITUATIONS
              for i in range(0, TC.BANK_SIZE, 125)]
    t0 = time.perf_counter()
    out = pool_map(_solve_chunk, chunks)
    U = {(w, s): np.empty((TC.BANK_SIZE, plan.L * DA)) for w in WORLDS for s in SITUATIONS}
    info = {(w, s): np.empty((TC.BANK_SIZE, 3)) for w in WORLDS for s in SITUATIONS}
    for w, s, idx, u, inf in out:
        U[(w, s)][idx], info[(w, s)][idx] = u, inf
    wall = time.perf_counter() - t0
    # a second exact solver on a subset (TRF, interior trust region)
    t1 = time.perf_counter()
    trf = pool_map(_trf_chunk, [(w, s, np.arange(TRF_STARTS)) for w in WORLDS for s in SITUATIONS])
    trf_wall = time.perf_counter() - t1
    arrays = {}
    for w in WORLDS:
        for s in SITUATIONS:
            arrays[f"U_{w}_{s}"] = U[(w, s)]
            arrays[f"info_{w}_{s}"] = info[(w, s)]
    for w, s, idx, u in trf:
        arrays[f"trf_{w}_{s}"] = u
    np.savez_compressed(out_path("cache/optimum100.npz"), **arrays)
    write_json("cache/solve_meta.json", dict(wall_seconds=wall, trf_wall_seconds=trf_wall, workers=WORKERS,
                                             chunks=len(chunks), bvls_tol=BVLS_TOL, qr_condition=float(
                                                 np.linalg.cond(plan.R))))


def load_optimum100():
    z = np.load(HERE / "cache/optimum100.npz")
    return {k: z[k] for k in z.files}


# ================================================================================================ checks of the optimum

def _perturb_job(job):
    """Random feasible perturbations and every single-command nudge must not lower the cost."""
    world, situation, U, seed = job
    plan = plan100()
    X = TC.bank_starts(situation)[:len(U)]
    b = BOUND[world]
    rng = np.random.default_rng(seed)
    base = plan.cost(U, X)
    worst_rel, worst_abs, improved, tried = np.inf, np.inf, 0, 0
    radii = (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)
    for i in range(len(U)):
        # 600 random directions (100 per radius), clipped back into the box
        d = rng.standard_normal((600, U.shape[1]))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = np.repeat(radii, 100)[:, None]
        cand = np.clip(U[i] + r * d, -b, b)
        # every single command nudged by +-1e-4 and +-1e-2 (inside the box)
        eye = np.eye(U.shape[1])
        for eps in (1e-4, 1e-2, -1e-4, -1e-2):
            cand = np.vstack([cand, np.clip(U[i] + eps * eye, -b, b)])
        c = plan.cost(cand, np.repeat(X[i:i + 1], len(cand), 0))
        delta = c - base[i]
        worst_abs = min(worst_abs, float(delta.min()))
        worst_rel = min(worst_rel, float(delta.min() / base[i]))
        improved += int(np.sum(delta < -1e-12 * base[i]))
        tried += len(cand)
    return dict(world=world, situation=situation, starts=len(U), tried=tried, improved=improved,
                worst_delta=worst_abs, worst_relative_delta=worst_rel)


def phase_validate():
    """Checks (a) and (b) on all 8,000 optimal plans -> cache/validate.json."""
    plan = plan100()
    opt = load_optimum100()
    out = {}
    perturb_jobs = []
    for w in WORLDS:
        phys, b = physics_of(w), BOUND[w]
        for s in SITUATIONS:
            X = TC.bank_starts(s)
            U = opt[f"U_{w}_{s}"]
            info = opt[f"info_{w}_{s}"]
            C = plan.cost(U, X)
            run = TR.rollout(OpenLoop(U.reshape(-1, LENGTH, DA)), X, phys)        # replay through dynamics()
            rel = np.abs(-run.J - C) / np.abs(C)
            G = plan.grad(U, X)
            fw = fw_gap(U, G, b)
            pg = np.abs(projected_gradient(U, G, b)).max(1)
            free = np.abs(U) < b - AT_BOUND
            up, lo = U >= b - AT_BOUND, U <= -b + AT_BOUND
            sign_violation = max(float(np.abs(G[free]).max()) if free.any() else 0.0,
                                 float(np.maximum(G[up], 0).max()) if up.any() else 0.0,
                                 float(np.maximum(-G[lo], 0).max()) if lo.any() else 0.0)
            trf = opt[f"trf_{w}_{s}"]
            Ct = plan.cost(trf, X[:len(trf)])
            out[f"{w}/{s}"] = dict(
                replay_max_relative_difference=float(rel.max()),
                replay_max_absolute_difference=float(np.abs(-run.J - C).max()),
                frank_wolfe_gap_max=float(fw.max()), frank_wolfe_gap_max_relative=float((fw / C).max()),
                frank_wolfe_gap_min=float(fw.min()),
                projected_gradient_max=float(pg.max()), kkt_sign_violation_max=sign_violation,
                bvls_status_counts={str(int(k)): int(v) for k, v in zip(*np.unique(info[:, 0], return_counts=True))},
                bvls_iterations_mean=float(info[:, 1].mean()), bvls_iterations_max=int(info[:, 1].max()),
                bvls_seconds_mean=float(info[:, 2].mean()),
                trf_starts=int(len(trf)), trf_max_relative_cost_difference=float(np.max(np.abs(Ct - C[:len(trf)]) / C[:len(trf)])),
                trf_cost_minus_bvls_cost_min=float(np.min(Ct - C[:len(trf)])),
                trf_max_command_difference=float(np.abs(trf - U[:len(trf)]).max()),
                max_command=float(np.abs(U).max()), bound=b,
                last_command_max_abs=float(np.abs(U[:, -DA:]).max()))
            perturb_jobs.append((w, s, U[:PERTURB_STARTS], 20261014 + len(perturb_jobs)))
    pert = pool_map(_perturb_job, perturb_jobs)
    for p in pert:
        out[f"{p['world']}/{p['situation']}"]["random_perturbation"] = p
    write_json("cache/validate.json", out)


NONCONVEX_INITS = ("optimum", "optimum_pushed_past_knee", "optimum_full_throttle", "optimum_plus_noise", "zero")
NONCONVEX_MAXITER = 5000
SAMPLES_PER_START = 2000


def true_cost_batch(plan, U, x0, physics):
    """True-thrust cost of many plans U (K, 400) from one start x0 (vectorised true_cost_grad without gradient)."""
    th = TC.thrust(U, physics)
    res = th @ plan.Ms.T + plan.ms(x0[None])[0]
    return float(x0[:DS] @ x0[:DS]) + np.sum(res * res, 1) + plan.energy_weight * np.sum(U * U, 1)


def _nonconvex_job(job):
    """Check (c) in the stalling world, over the FULL range [-1, 1]^400 with the TRUE (non-convex) thrust:
    (1) the one-sided slope of the cost for pushing each at-knee optimal command past the knee (must be > 0);
    (2) 2,000 sampled plans per start that use past-knee commands (the optimum with some at-knee commands pushed past
        the knee, the optimum with random commands replaced, the optimum plus noise, uniform random plans): none may cost
        less than the optimum; each is also mapped through 'replace every command by its own push';
    (3) L-BFGS-B started at the optimum and at four other plans (at most 5,000 iterations each)."""
    from scipy.optimize import minimize

    situation, idx, U, seed = job
    plan = plan100()
    phys = physics_of("stall")
    knee = phys.knee
    X = TC.bank_starts(situation)[idx]
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(X)):
        x0, ustar = X[i], U[i]
        cstar = float(plan.cost(ustar[None], x0[None])[0])
        at = np.abs(ustar) >= knee - AT_BOUND
        # (1) outward slope at the knee: d cost / d|a| just past the knee = -slope sign(a) dC/dthrust + 0.6 |a|
        res = plan.Ms @ ustar + plan.ms(x0[None])[0]
        h = 2.0 * (plan.Ms.T @ res)
        out_slope = -phys.thrust_slope * np.sign(ustar[at]) * h[at] + 2 * plan.energy_weight * np.abs(ustar[at])
        # (2) sampled plans that use the past-knee range
        K = SAMPLES_PER_START // 4
        push = np.where(at & (rng.uniform(size=(K, ustar.size)) < 0.5),
                        np.sign(ustar) * rng.uniform(knee, 1.0, (K, ustar.size)), ustar)
        repl = np.where(rng.uniform(size=(K, ustar.size)) < rng.uniform(0.01, 0.5, (K, 1)),
                        rng.uniform(-1, 1, (K, ustar.size)), ustar)
        noise = np.clip(ustar + rng.choice([0.01, 0.1, 0.3], (K, 1)) * rng.standard_normal((K, ustar.size)), -1, 1)
        unif = rng.uniform(-1, 1, (K, ustar.size))
        samples = np.vstack([push, repl, noise, unif])
        cs = true_cost_batch(plan, samples, x0, phys)
        cdom = plan.cost(TC.thrust(samples, phys), np.repeat(x0[None], len(samples), 0))
        rec = dict(start=int(idx[i]), best_cost=cstar, past_knee_at_optimum=int(np.sum(np.abs(ustar) > knee)),
                   at_knee_commands=int(at.sum()),
                   min_outward_slope=float(out_slope.min()) if at.any() else None,
                   samples=int(len(samples)), samples_using_past_knee=int(np.sum(np.any(np.abs(samples) > knee, 1))),
                   samples_min_relative=float((cs.min() - cstar) / cstar),
                   samples_lower_than_best=int(np.sum(cs < cstar * (1 - 1e-12))),
                   samples_dominance_violations=int(np.sum(cdom > cs * (1 + 1e-12) + 1e-12)),
                   samples_dominated_min_relative=float((cdom.min() - cstar) / cstar))
        starts = {
            "optimum": ustar.copy(),
            "optimum_pushed_past_knee": np.where(at, np.sign(ustar) * rng.uniform(knee + 1e-3, 1.0, ustar.shape), ustar),
            "optimum_full_throttle": np.where(at, np.sign(ustar), ustar),
            "optimum_plus_noise": np.clip(ustar + rng.uniform(-0.3, 0.3, ustar.shape), -1, 1),
            "zero": np.zeros_like(ustar),
        }
        for name in NONCONVEX_INITS:
            res = minimize(plan.true_cost_grad, starts[name], args=(x0, phys), jac=True, method="L-BFGS-B",
                           bounds=[(-1.0, 1.0)] * len(ustar),
                           options=dict(maxiter=NONCONVEX_MAXITER, maxfun=2 * NONCONVEX_MAXITER, ftol=1e-16,
                                        gtol=1e-12, maxcor=30))
            u = np.clip(res.x, -1, 1)
            c_true, _ = plan.true_cost_grad(u, x0, phys)
            # replay the final plan through the world itself (true stalling thrust)
            run = TR.rollout(OpenLoop(u.reshape(1, LENGTH, DA)), x0[None], phys)
            # dominance: replacing each command by its own push gives the same states, less energy, a feasible plan
            u_dom = TC.thrust(u, phys)
            c_dom = float(plan.cost(u_dom[None], x0[None])[0])
            rec[name] = dict(cost_minus_best=c_true - cstar, relative=(c_true - cstar) / cstar,
                             replay_relative=abs(-float(run.J[0]) - c_true) / c_true,
                             past_knee_commands=int(np.sum(np.abs(u) > knee)),
                             dominated_cost_minus_cost=c_dom - c_true, dominated_cost_minus_best=c_dom - cstar,
                             iterations=int(res.nit))
        rows.append(rec)
    return situation, rows


def phase_nonconvex():
    opt = load_optimum100()
    jobs = []
    for k, s in enumerate(SITUATIONS):
        U = opt[f"U_stall_{s}"]
        quarter = NONCONVEX_STARTS // 4
        for j in range(4):
            idx = np.arange(j * quarter, (j + 1) * quarter)
            jobs.append((s, idx, U[idx], 777 + 10 * k + j))
    out = pool_map(_nonconvex_job, jobs)
    rows = [r for _, rr in out for r in rr]
    inits = NONCONVEX_INITS
    slopes = [r["min_outward_slope"] for r in rows if r["min_outward_slope"] is not None]
    # (1) for ALL 4,000 stalling-world starts: the outward slope at every at-knee optimal command
    plan, opt, phys = plan100(), load_optimum100(), physics_of("stall")
    all_slopes, n_at = [], 0
    for s in SITUATIONS:
        Uall, Xall = opt[f"U_stall_{s}"], TC.bank_starts(s)
        H = 2.0 * ((Uall @ plan.Ms.T + plan.ms(Xall)) @ plan.Ms)
        at = np.abs(Uall) >= phys.knee - AT_BOUND
        n_at += int(at.sum())
        all_slopes.append((-phys.thrust_slope * np.sign(Uall) * H + 2 * plan.energy_weight * np.abs(Uall))[at])
    all_slopes = np.concatenate(all_slopes)
    summary = dict(starts=len(rows), per_situation=NONCONVEX_STARTS, bounds="[-1, 1] (full range)",
                   thrust="true stalling thrust (non-convex)",
                   optimiser=f"scipy L-BFGS-B, ftol 1e-16, gtol 1e-12, at most {NONCONVEX_MAXITER} iterations",
                   optimum_commands_past_knee=int(sum(r["past_knee_at_optimum"] for r in rows)),
                   outward_slope_all_starts=dict(at_knee_commands=n_at, starts=4 * TC.BANK_SIZE,
                                                 min=float(all_slopes.min()) if n_at else None,
                                                 all_positive=bool(n_at == 0 or all_slopes.min() > 0)),
                   outward_slope_min_checked_starts=float(min(slopes)) if slopes else None,
                   sampled_plans=dict(total=int(sum(r["samples"] for r in rows)),
                                      using_past_knee=int(sum(r["samples_using_past_knee"] for r in rows)),
                                      lower_than_best=int(sum(r["samples_lower_than_best"] for r in rows)),
                                      min_relative=float(min(r["samples_min_relative"] for r in rows)),
                                      dominance_violations=int(sum(r["samples_dominance_violations"] for r in rows)),
                                      dominated_min_relative=float(min(r["samples_dominated_min_relative"]
                                                                       for r in rows))))
    for name in inits:
        d = np.array([r[name]["cost_minus_best"] for r in rows])
        rel = np.array([r[name]["relative"] for r in rows])
        pk = np.array([r[name]["past_knee_commands"] for r in rows])
        dom = np.array([r[name]["dominated_cost_minus_cost"] for r in rows])
        domb = np.array([r[name]["dominated_cost_minus_best"] for r in rows])
        rp = np.array([r[name]["replay_relative"] for r in rows])
        summary[name] = dict(min_cost_minus_best=float(d.min()), min_relative=float(rel.min()),
                             median_relative=float(np.median(rel)), max_relative=float(rel.max()),
                             runs_lower_than_best=int(np.sum(d < -1e-9 * np.array([r["best_cost"] for r in rows]))),
                             runs_ending_with_past_knee_commands=int(np.sum(pk > 0)),
                             past_knee_commands_mean=float(pk.mean()),
                             max_dominated_cost_minus_cost=float(dom.max()),
                             min_dominated_cost_minus_best=float(domb.min()), max_replay_relative=float(rp.max()))
    write_json("cache/nonconvex.json", dict(summary=summary, rows=rows))


# ================================================================================================ discounted optimum

class Discounted:
    """sum_{t<H} g^t (|s_t|^2 + 0.3 |a_t|^2) over H-step plans, in scaled variables v_t = g^(t/2) a_t, z_t = g^(t/2) x_t:
    z_{t+1} = sqrt(g) A z_t + sqrt(g) B v_t, cost sum_t |S z_t|^2 + 0.3 |v_t|^2, bounds |v_t| <= g^(t/2) b."""

    def __init__(self, horizon=DISC_HORIZON, gamma=GAMMA):
        A, B = (np.array(m, np.float64) for m in TC.matrices(TC.STALL))
        self.H, self.g = int(horizon), float(gamma)
        sg = math.sqrt(self.g)
        self.Ap, self.Bp = sg * A, sg * B
        self.ApT, self.BpT = self.Ap.T.copy(), self.Bp.T.copy()
        self.scale = self.g ** (0.5 * np.arange(self.H))
        self.w = TC.STALL.throttle_cost

    def cost_grad(self, z0, v):
        H = self.H
        Z = np.empty((H,) + z0.shape)
        z = z0
        for t in range(H):
            Z[t] = z
            z = z @ self.ApT + v[t] @ self.BpT
        S = Z[:, :, :DS]
        f = np.sum(S * S, axis=(0, 2)) + self.w * np.sum(v * v, axis=(0, 2))
        G = np.empty_like(v)
        p = np.zeros_like(z0)                      # derivative of the cost after step t with respect to z_{t+1}
        for t in range(H - 1, -1, -1):
            G[t] = 2 * self.w * v[t] + p @ self.Bp
            q = p @ self.Ap
            q[:, :DS] += 2 * Z[t][:, :DS]
            p = q
        return f, G

    def lipschitz(self, iters=80):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((self.H, 1, DA))
        z0 = np.zeros((1, DX))
        lam = 0.0
        for _ in range(iters):
            _, G = self.cost_grad(z0, v)
            lam = math.sqrt(np.sum(G * G) / np.sum(v * v))
            v = G / math.sqrt(np.sum(G * G))
        return lam

    def solve(self, X, b, L, max_iter=4000, check=50, target=1e-12):
        N = len(X)
        hi = np.broadcast_to((self.scale * b)[:, None, None], (self.H, N, DA))
        mu = 2 * self.w
        q = mu / L
        beta = (1 - math.sqrt(q)) / (1 + math.sqrt(q))
        v = np.zeros((self.H, N, DA))
        y = v.copy()
        it, gap = 0, None
        for it in range(1, max_iter + 1):
            _, G = self.cost_grad(X, y)
            vn = np.clip(y - G / L, -hi, hi)
            y = vn + beta * (vn - v)
            v = vn
            if it % check == 0:
                f, Gv = self.cost_grad(X, v)
                gap = np.sum(Gv * v + hi * np.abs(Gv), axis=(0, 2))
                if np.max(gap / f) < target:
                    break
        f, Gv = self.cost_grad(X, v)
        gap = np.sum(Gv * v + hi * np.abs(Gv), axis=(0, 2))
        pg = np.sqrt(np.sum((v - np.clip(v - Gv, -hi, hi)) ** 2, axis=(0, 2)))
        u = (v / self.scale[:, None, None]).transpose(1, 0, 2)       # (N, H, 4) commands
        return u, f, gap, pg, it


def _discounted_job(job):
    world, situation, L = job
    t0 = time.perf_counter()
    d = Discounted()
    X = TC.bank_starts(situation)[:DISC_STARTS]
    b = BOUND[world]
    u, f, gap, pg, it = d.solve(X, b, L)
    u = np.clip(u, -b, b)
    phys = physics_of(world)
    long = TR.rollout(OpenLoop(u), X, phys, length=DISC_HORIZON)          # replay 1,000 steps through dynamics()
    short = TR.rollout(OpenLoop(u[:, :LENGTH]), X, phys, length=LENGTH)   # the discount-optimal plan's 100-step return
    return dict(world=world, situation=situation, value=-f, fw_gap=gap, pg=pg, iterations=it, replay_Jd=long.Jd,
                J100_of_discount_plan=short.J, first_commands=u[:, :LENGTH], seconds=time.perf_counter() - t0,
                at_bound_share=float(np.mean(np.abs(u[:, :LENGTH]) >= b - AT_BOUND)))


def phase_discounted():
    d = Discounted()
    L = 1.02 * d.lipschitz()
    out = pool_map(_discounted_job, [(w, s, L) for w in WORLDS for s in SITUATIONS])
    arrays, meta = {}, dict(lipschitz=L, strong_convexity=2 * d.w, condition=L / (2 * d.w), horizon=d.H, gamma=d.g,
                            starts=DISC_STARTS, per={})
    for r in out:
        k = f"{r['world']}_{r['situation']}"
        arrays[f"value_{k}"] = r["value"]
        arrays[f"J100plan_{k}"] = r["J100_of_discount_plan"]
        arrays[f"plan100_{k}"] = r["first_commands"]
        rel = np.abs(r["replay_Jd"] - r["value"]) / np.abs(r["value"])
        meta["per"][f"{r['world']}/{r['situation']}"] = dict(
            iterations=r["iterations"], seconds=r["seconds"], fw_gap_max=float(r["fw_gap"].max()),
            fw_gap_max_relative=float(np.max(r["fw_gap"] / -r["value"])), projected_gradient_max=float(r["pg"].max()),
            replay_max_relative_difference=float(rel.max()), value_mean=float(r["value"].mean()),
            first_100_at_bound_share=r["at_bound_share"])
    np.savez_compressed(out_path("cache/discounted.npz"), **arrays)
    write_json("cache/discounted_meta.json", meta)


# ================================================================================================ controllers

def _norm_job(job):
    """acceptance.load_seed(...).prepared('gap', ...) of one (world, seed): the actors' observation normalisation."""
    import experiments.shift.acceptance as A
    import experiments.shift.learners as L

    world, rep = job
    prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(prereg)
    sd = A.load_seed(prereg.runs, prereg.config, VARIANT, world, rep)
    prep = sd.prepared("gap", prereg.config.learn)
    return dict(world=world, rep=rep, obs_mean=np.asarray(prep.obs_mean), obs_std=np.asarray(prep.obs_std),
                sha256=prep.sha256["normalization"], physics=prep.physics.as_dict(),
                check=L.tree_sha256((prep.obs_mean, prep.obs_std)))


def _tree_np(tree):
    import jax

    return jax.tree_util.tree_map(lambda a: np.asarray(a), tree)


def collect_controllers():
    """Every controller to score: reference controllers, the gate's saved actors, the Stage 2 locked arms.

    Returns (jobs, records, meta): one job per distinct (world, normalisation, actor) and, per recorded appearance,
    which job it is and its recorded points."""
    import experiments.shift.acceptance as A
    import experiments.shift.learners as L

    prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(prereg)
    if prereg.digest[:16] != GATE_DIGEST16:
        raise SystemExit(f"refused: the gate's pre-registration is {prereg.digest[:16]}, expected {GATE_DIGEST16}")
    outcome = A.gate_outcome(prereg)
    anchors = {w: A.anchors_of(prereg, VARIANT, w) for w in WORLDS}
    spec = A.stage0_spec(prereg.runs, VARIANT)
    # normalisations through the gate's loader, cross-checked against the cloned actors' records
    norm_list = pool_map(_norm_job, [(w, r) for w in WORLDS for r in GATE_SEEDS], workers=min(WORKERS, 10))
    norms = {(n["world"], n["rep"]): n for n in norm_list}
    norm_checks = {}
    jobs, records, by_key = [], [], {}

    def add_actor(params, world, rep, label, source, recorded, extra):
        np_params = _tree_np(params)
        tsha = L.tree_sha256(np_params)
        key = (world, rep, tsha)
        if key not in by_key:
            by_key[key] = len(jobs)
            n = norms[(world, rep)]
            jobs.append(dict(kind="actor", world=world, rep=rep, params=np_params, obs_mean=n["obs_mean"],
                             obs_std=n["obs_std"], anchors=anchors[world], tree_sha256=tsha, label=label))
        records.append(dict(job=by_key[key], world=world, rep=rep, label=label, source=source, tree_sha256=tsha,
                            recorded_points=recorded, **extra))

    for (w, r), n in norms.items():
        d = prereg.root / "clone" / VARIANT / w / f"seed{r}"
        rec = A.finished(d / "record.json", prereg)
        data = rec["result"]["data"]
        same = (np.array_equal(np.asarray(data["obs_mean"], np.float32), n["obs_mean"])
                and np.array_equal(np.asarray(data["obs_std"], np.float32), n["obs_std"]))
        norm_checks[f"{w}/seed{r}"] = dict(loader_sha256=n["sha256"], record_sha256=data["sha256"]["normalization"],
                                           sha_match=n["sha256"] == data["sha256"]["normalization"] == n["check"],
                                           arrays_match_record=bool(same))
        clone = L.load_tree(d / "actor.msgpack", L.actor_template(), rec["result"]["file_sha256"])
        norm_checks[f"{w}/seed{r}"]["clone_tree_sha256"] = L.tree_sha256(_tree_np(clone))

    # the reference controllers (no data): one job per world
    for w in WORLDS:
        for name in ("recording", "stall_aware", "rising_thrust", "hand_only"):
            jobs.append(dict(kind="reference", world=w, name=name, anchors=anchors[w], spec=spec.as_dict(),
                             label=f"reference:{name}"))
            records.append(dict(job=len(jobs) - 1, world=w, rep=None, label=f"reference:{name}", source="reference",
                                tree_sha256=None, recorded_points=None, family=f"reference:{name}", name=name))

    # the gate's saved actors (check 2, both alpha rungs), SHA-256 from values.json
    gate_files = {}
    for alpha in GATE_ALPHAS:
        for r in GATE_SEEDS:
            d = prereg.root / "harm_return" / VARIANT / f"alpha{alpha}" / f"seed{r}"
            rec = A.finished(d / "values.json", prereg)
            if rec is None:
                raise SystemExit(f"refused: no verified values.json at {d}")
            acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]},
                               rec["actors_sha256"])
            gate_files[f"alpha{alpha}/seed{r}"] = dict(path=str(d / "actors.msgpack"), sha256=rec["actors_sha256"],
                                                       verified=True)
            for name in rec["actor_names"]:
                if rec["failed"].get(name):
                    continue
                world = "no_stall" if name.startswith("no_stall") else "stall"
                ev = rec["evaluations"].get(name)
                recorded = None if ev is None else {s: ev[s]["points"] for s in SITUATIONS}
                add_actor(acts[name], world, r, f"gate:alpha{alpha}:{name}", "gate", recorded,
                          dict(alpha=alpha, name=name, family=f"gate:{name}"))

    # the Stage 2 (v6) locked arms of seeds 0-1
    stage2_files = {}
    for r in STAGE2_SEEDS:
        for world in WORLDS:
            d = STAGE2_POLICY / f"seed{r}" / ("" if world == "stall" else "no_stall")
            lock = read_json_sidecar(d / "lock.json")
            vals = read_json_sidecar(d / "values.json")
            acts = L.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                               lock["actors_file_sha256"])
            bad = [k for k in lock["names"] if L.tree_sha256(acts[k]) != lock["actor_sha256"][k]]
            if bad:
                raise SystemExit(f"refused: {d}: actors differ from the lock: {bad}")
            stage2_files[f"seed{r}/{world}"] = dict(path=str(d / lock["actors_file"]),
                                                    file_sha256=lock["actors_file_sha256"],
                                                    actors_checked=len(lock["names"]), all_tree_sha256_match=True)
            matched = {v: k for k, v in (lock.get("matched_names") or {}).items() if v}
            standin_arm = (lock.get("standin") or {}).get("arm")
            for name in lock["scored_names"]:
                ev = vals["evaluations"].get(name)
                if ev is None:
                    continue
                if name in matched:
                    family = f"stage2:{matched[name]} (matched)"
                elif name == standin_arm or name == "standin_locked":
                    family = "stage2:standin (matched)" if world == "stall" else "stage2:standin (locked)"
                else:
                    family = f"stage2:{name}"
                add_actor(acts[name], world, r, f"stage2:seed{r}:{world}:{name}", "stage2",
                          {s: ev[s]["points"] for s in SITUATIONS}, dict(alpha="5", name=name, family=family))
    meta = dict(prereg_digest=prereg.digest, gate_outcome=outcome, anchors=anchors, recorder_spec=spec.as_dict(),
                normalisation_checks=norm_checks, gate_files=gate_files, stage2_files=stage2_files)
    return jobs, records, meta


def _controller_policy(job):
    import experiments.shift.learners as L

    phys = physics_of(job["world"])
    if job["kind"] == "reference":
        name = job["name"]
        spec = RC.RecorderSpec(**job["spec"])            # the gate's recorder settings (stage0_base.json)
        if name == "recording":
            return TR.recorder_on_bank(phys, spec), phys, None
        return {"stall_aware": RC.stall_aware, "rising_thrust": RC.rising_thrust,
                "hand_only": lambda p: RC.hand_only(spec, p)}[name](phys), phys, None
    prep = SimpleNamespace(obs_mean=np.asarray(job["obs_mean"], np.float32),
                           obs_std=np.asarray(job["obs_std"], np.float32), physics=phys)
    return L.mlp_policy(job["params"], prep), phys, prep


def _score_job(job):
    """One controller: the gate's evaluation (learners.evaluate_policy / evaluate_actor), per-start returns, tracking
    and energy costs, peak float deviation, and the exact discounted value on the first 200 starts (truth.value)."""
    import experiments.shift.learners as L

    t0 = time.perf_counter()
    policy, phys, prep = _controller_policy(job)
    an = job["anchors"]
    if prep is not None:
        ev = L.evaluate_actor(job["params"], prep, an)          # the gate's code path (acceptance._evaluate)
    else:
        ev = L.evaluate_policy(policy, phys, an)
    if job["kind"] == "reference" and job["name"] == "recording":
        policy = TR.recorder_on_bank(phys, RC.RecorderSpec(**job["spec"]))   # fresh streams for the detail rollout
    per = {}
    for s in SITUATIONS:
        X = TC.bank_starts(s)
        if job["kind"] == "reference" and job["name"] == "recording":
            policy = TR.recorder_on_bank(phys, RC.RecorderSpec(**job["spec"]))
        run = TR.rollout(policy, X, phys, record=True)
        S = run.states[:, :LENGTH, :DS]
        tracking = np.sum(S * S, axis=(1, 2))
        energy = phys.throttle_cost * np.sum(run.commands ** 2, axis=(1, 2))
        radius = np.sqrt(np.sum(S * S, axis=2))
        peak, late_peak = radius.max(1), radius[:, LATE_PEAK_FROM:].max(1)
        at_bound = np.abs(run.commands) >= BOUND[job["world"]] - AT_BOUND
        d = dict(J=run.J, tracking=tracking, energy=energy, peak=peak, late_peak=late_peak,
                 J_mean_matches_evaluator=bool(float(run.J.mean()) == ev[s]["J"]),
                 identity_error=float(np.max(np.abs(tracking + energy + run.J))),
                 past_knee_step_share=float(TC.past_knee(run.commands, phys).mean()),
                 at_bound_share=float(at_bound.mean()), mean_abs_command=float(np.abs(run.commands).mean()))
        deterministic = not (job["kind"] == "reference" and job["name"] == "recording")
        if deterministic:
            d["value_disc"] = TR.value(policy, X[:DISC_STARTS], phys, horizon=DISC_HORIZON)
        per[s] = d
    return dict(ev=ev, per=per, seconds=time.perf_counter() - t0)


def phase_score():
    t0 = time.perf_counter()
    jobs, records, meta = collect_controllers()
    meta["collect_seconds"] = time.perf_counter() - t0
    print(f"scoring {len(jobs)} distinct controllers ({len(records)} recorded appearances)", flush=True)
    order = sorted(range(len(jobs)), key=lambda i: 0 if jobs[i]["kind"] == "reference" else 1)
    t1 = time.perf_counter()
    res = pool_map(_score_job, [jobs[i] for i in order])
    results = [None] * len(jobs)
    for i, r in zip(order, res):
        results[i] = r
    meta["score_seconds"] = time.perf_counter() - t1
    arrays = {}
    summary = []
    per_start = ("J", "tracking", "energy", "peak", "late_peak", "value_disc")
    for i, (job, r) in enumerate(zip(jobs, results)):
        for s in SITUATIONS:
            for k in per_start:
                if k in r["per"][s]:
                    arrays[f"{i}_{s}_{k}"] = r["per"][s][k]
        summary.append(dict(job=i, kind=job["kind"], world=job["world"], rep=job.get("rep"),
                            label=job.get("label", job.get("name")), tree_sha256=job.get("tree_sha256"),
                            ev=r["ev"], seconds=r["seconds"],
                            per={s: {k: v for k, v in r["per"][s].items() if k not in per_start}
                                 for s in SITUATIONS}))
    np.savez_compressed(out_path("cache/controllers.npz"), **arrays)
    write_json("cache/controllers.json", dict(meta=meta, jobs=summary, records=records))


# ================================================================================================ float32 deployment

def _float32_job(job):
    """Deployed float32 path (ThrusterChainEnv + the float32 JAX actor) against the float64 evaluator."""
    import experiments.shift.learners as L

    t0 = time.perf_counter()
    world = job["world"]
    phys = physics_of(world)
    params = job["params"]
    prep = SimpleNamespace(obs_mean=np.asarray(job["obs_mean"], np.float32),
                           obs_std=np.asarray(job["obs_std"], np.float32), physics=phys)
    an = job["anchors"]
    pol64 = L.mlp_policy(params, prep)

    class Cast32(RC.Controller):
        """The float64 actor with its command cast to float32 (only the command rounding of env.step)."""

        def __call__(self, x):
            return pol64(x).astype(np.float32).astype(np.float64)

    out = {}
    for s in SITUATIONS:
        X = TC.bank_starts(s)
        N = len(X)
        envs = [TC.ThrusterChainEnv(stall=world == "stall", start=s) for _ in range(N)]
        obs = np.stack([env.reset(options={"state": X[i]})[0] for i, env in enumerate(envs)])
        J32 = np.zeros(N)
        for _ in range(LENGTH):
            a = L.actor_rows(params, (obs.astype(np.float32) - prep.obs_mean) / prep.obs_std)   # float32 actor
            for i, env in enumerate(envs):
                o, r, terminated, truncated, _ = env.step(a[i])
                obs[i] = o
                J32[i] += r
        J64 = TR.rollout(pol64, X, phys).J
        J64c = TR.rollout(Cast32(), X, phys).J
        scale = 100.0 / (an[s]["top"] - an[s]["recording"])
        diff = np.abs(J32 - J64)
        out[s] = dict(max_abs_return_difference=float(diff.max()),
                      p999_abs_return_difference=float(np.quantile(diff, 0.999)),
                      mean_abs_return_difference=float(diff.mean()),
                      max_abs_points_difference=float(diff.max() * scale),
                      p999_abs_points_difference=float(np.quantile(diff, 0.999) * scale),
                      bank_mean_points_difference=float((J32.mean() - J64.mean()) * scale),
                      command_rounding_only_max_abs_return_difference=float(np.abs(J64c - J64).max()),
                      truncated_at_100=bool(truncated))
    return dict(label=job["label"], world=world, per=out, seconds=time.perf_counter() - t0)


def phase_float32():
    import experiments.shift.acceptance as A
    import experiments.shift.learners as L

    prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(prereg)
    an = A.anchors_of(prereg, VARIANT, "stall")
    norm = _norm_job(("stall", 0))
    jobs = []
    for source, alpha, rep, name in FLOAT32_ACTORS:
        if source == "gate":
            d = prereg.root / "harm_return" / VARIANT / f"alpha{alpha}" / f"seed{rep}"
            rec = A.finished(d / "values.json", prereg)
            acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]},
                               rec["actors_sha256"])
            label = f"gate:alpha{alpha}:seed{rep}:{name}"
        else:
            d = STAGE2_POLICY / f"seed{rep}"
            lock = read_json_sidecar(d / "lock.json")
            acts = L.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                               lock["actors_file_sha256"])
            assert L.tree_sha256(acts[name]) == lock["actor_sha256"][name]
            label = f"stage2:seed{rep}:stall:{name}"
        jobs.append(dict(world="stall", params=_tree_np(acts[name]), obs_mean=norm["obs_mean"],
                         obs_std=norm["obs_std"], anchors=an, label=label))
    out = pool_map(_float32_job, jobs)
    write_json("cache/float32.json", out)


# ================================================================================================ report

def _stats(J):
    J = np.asarray(J)
    k = max(1, int(round(0.05 * len(J))))
    return dict(mean=float(J.mean()), se=float(J.std() / math.sqrt(len(J))), p5=float(np.quantile(J, 0.05)),
                worst5_mean=float(np.sort(J)[:k].mean()), worst=float(J.min()))


def _points(J, an):
    return 100.0 * (np.asarray(J, np.float64) - an["recording"]) / (an["top"] - an["recording"])


def build_report(timer):
    from scipy import stats as SS

    plan = plan100()
    opt = load_optimum100()
    val = read_json("cache/validate.json")
    nonconvex = read_json("cache/nonconvex.json")["summary"]
    dmeta = read_json("cache/discounted_meta.json")
    dz = np.load(HERE / "cache/discounted.npz")
    ctrl = read_json("cache/controllers.json")
    cz = np.load(HERE / "cache/controllers.npz")
    f32 = read_json("cache/float32.json")
    solve_meta = read_json("cache/solve_meta.json")
    anchors = ctrl["meta"]["anchors"]

    # ---------------------------------------------------------------- the optimum per world and situation
    optimum = {}
    J_opt = {}
    for w in WORLDS:
        b = BOUND[w]
        optimum[w] = {}
        for s in SITUATIONS:
            X = TC.bank_starts(s)
            U = opt[f"U_{w}_{s}"].reshape(-1, LENGTH, DA)
            run = TR.rollout(OpenLoop(U), X, physics_of(w), record=True)
            J_opt[(w, s)] = run.J
            S = run.states[:, :LENGTH, :DS]
            tracking = np.sum(S * S, axis=(1, 2))
            energy = 0.3 * np.sum(U ** 2, axis=(1, 2))
            radius = np.sqrt(np.sum(S * S, axis=2))
            peak, late_peak = radius.max(1), radius[:, LATE_PEAK_FROM:].max(1)
            start_is_peak = float(np.mean(radius.argmax(1) == 0))
            at = np.abs(U) >= b - AT_BOUND
            pts = _points(run.J, anchors[w][s])
            optimum[w][s] = dict(
                return_=_stats(run.J), points=float(pts.mean()), points_se=float(pts.std() / math.sqrt(len(pts))),
                tracking_cost_mean=float(tracking.mean()), energy_cost_mean=float(energy.mean()),
                peak_deviation_mean=float(peak.mean()), peak_deviation_max=float(peak.max()),
                peak_is_the_start_share=start_is_peak,
                late_peak_deviation_mean=float(late_peak.mean()), late_peak_deviation_max=float(late_peak.max()),
                thruster_steps_at_bound_share=float(at.mean()),
                steps_with_any_thruster_at_bound_share=float(at.any(-1).mean()),
                thruster_steps_at_bound_share_by_thruster=at.mean(axis=(0, 1)).tolist(),
                starts_using_the_bound_share=float(at.any(axis=(1, 2)).mean()),
                mean_abs_command=float(np.abs(U).mean()),
                mean_abs_command_by_thruster=np.abs(U).mean(axis=(0, 1)).tolist(),
                share_past_0_8=float(np.mean(np.abs(U) > TC.KNEE)),
                share_steps_any_past_0_8=float(TC.past_knee(U, physics_of(w)).mean()),
                at_bound_share_by_step_decile=[float(at[:, i:i + 10].mean()) for i in range(0, LENGTH, 10)],
                bound=b, anchors=anchors[w][s])

    # ---------------------------------------------------------------- controllers vs the optimum
    jobs = ctrl["jobs"]
    per_job = []
    lower_bound_violation = math.inf          # the smallest per-start regret over every controller and start
    for j in jobs:
        w = j["world"]
        an = anchors[w]
        d = {}
        for s in SITUATIONS:
            i = j["job"]
            J = cz[f"{i}_{s}_J"]
            reg = J_opt[(w, s)] - J
            lower_bound_violation = min(lower_bound_violation, float(reg.min()))
            tr, en, pk = cz[f"{i}_{s}_tracking"], cz[f"{i}_{s}_energy"], cz[f"{i}_{s}_peak"]
            lpk = cz[f"{i}_{s}_late_peak"]
            pts = _points(J, an[s])
            e = dict(return_=_stats(J), points=j["ev"][s]["points"], points_from_per_start=float(pts.mean()),
                     regret_return=float(reg.mean()), regret_points=float(optimum[w][s]["points"] - j["ev"][s]["points"]),
                     regret_min_over_starts=float(reg.min()),
                     regret_share_of_optimum=float(reg.mean() / -J_opt[(w, s)].mean()),
                     tracking_cost_mean=float(tr.mean()), energy_cost_mean=float(en.mean()),
                     peak_deviation_mean=float(pk.mean()), peak_deviation_max=float(pk.max()),
                     late_peak_deviation_mean=float(lpk.mean()), late_peak_deviation_max=float(lpk.max()),
                     capped_points=j["ev"][s]["capped_points"],
                     past_knee_step_share=j["per"][s]["past_knee_step_share"],
                     at_bound_share=j["per"][s]["at_bound_share"], mean_abs_command=j["per"][s]["mean_abs_command"],
                     J_mean_matches_evaluator=j["per"][s]["J_mean_matches_evaluator"])
            key = f"{i}_{s}_value_disc"
            if key in cz.files:
                Vd = cz[key]
                Vopt = dz[f"value_{w}_{s}"]
                e["discounted_value_mean"] = float(Vd.mean())
                e["discounted_regret_mean"] = float((Vopt - Vd).mean())
                e["discounted_regret_min_over_starts"] = float((Vopt - Vd).min())
                e["J100_first200_mean"] = float(J[:DISC_STARTS].mean())
            d[s] = e
        per_job.append(d)

    # recorded points check (values.json) for every recorded appearance
    rec_diffs = []
    for rec in ctrl["records"]:
        if rec["recorded_points"] is None:
            continue
        j = jobs[rec["job"]]
        for s in SITUATIONS:
            rec_diffs.append(abs(j["ev"][s]["points"] - rec["recorded_points"][s]))
    rec_diffs = np.array(rec_diffs)

    # the gate's saved cloned actors (clone/base/<world>/seed<s>/actor.msgpack) are the 'clone' entries scored here
    clone_same = []
    for rec in ctrl["records"]:
        if rec["source"] == "gate" and rec["name"] in ("clone", "no_stall_clone"):
            nc_ = ctrl["meta"]["normalisation_checks"][f"{rec['world']}/seed{rec['rep']}"]
            clone_same.append(rec["tree_sha256"] == nc_["clone_tree_sha256"])
    # anchors recomputed through truth.anchors (should equal the pre-registered ones)
    anchor_check = {}
    for w in WORLDS:
        an2 = TR.anchors(physics_of(w), RC.RecorderSpec(**ctrl["meta"]["recorder_spec"]))
        anchor_check[w] = max(max(abs(an2[s]["recording"] - anchors[w][s]["recording"]),
                                  abs(an2[s]["top"] - anchors[w][s]["top"])) for s in SITUATIONS)

    # ---------------------------------------------------------------- per-arm summaries (mean over seeds)
    fam_rows = {}
    for rec in ctrl["records"]:
        fam = rec["family"]
        if rec["source"] == "gate":
            fam = f"gate alpha {rec['alpha']}: {rec['name']}"
        key = (rec["world"], fam)
        fam_rows.setdefault(key, [])
        if rec["job"] not in [x for x in fam_rows[key]]:
            fam_rows[key].append(rec["job"])
    families = {}
    for (w, fam), job_ids in sorted(fam_rows.items()):
        entry = dict(world=w, family=fam, seeds=[jobs[i]["rep"] for i in job_ids], n=len(job_ids))
        for s in SITUATIONS:
            pts = [per_job[i][s]["points"] for i in job_ids]
            rg = [per_job[i][s]["regret_points"] for i in job_ids]
            rr = [per_job[i][s]["regret_return"] for i in job_ids]
            entry[s] = dict(points_mean=float(np.mean(pts)), regret_points_mean=float(np.mean(rg)),
                            regret_points_by_seed=rg, points_by_seed=pts, regret_return_mean=float(np.mean(rr)),
                            tracking_cost_mean=float(np.mean([per_job[i][s]["tracking_cost_mean"] for i in job_ids])),
                            energy_cost_mean=float(np.mean([per_job[i][s]["energy_cost_mean"] for i in job_ids])),
                            peak_deviation_mean=float(np.mean([per_job[i][s]["peak_deviation_mean"] for i in job_ids])),
                            late_peak_deviation_mean=float(np.mean([per_job[i][s]["late_peak_deviation_mean"]
                                                                    for i in job_ids])),
                            late_peak_deviation_max=float(np.max([per_job[i][s]["late_peak_deviation_max"]
                                                                  for i in job_ids])),
                            p5_return_mean=float(np.mean([per_job[i][s]["return_"]["p5"] for i in job_ids])),
                            worst5_return_mean=float(np.mean([per_job[i][s]["return_"]["worst5_mean"]
                                                              for i in job_ids])),
                            past_knee_step_share_mean=float(np.mean([per_job[i][s]["past_knee_step_share"]
                                                                     for i in job_ids])))
            if "discounted_regret_mean" in per_job[job_ids[0]][s]:
                entry[s]["discounted_regret_mean"] = float(np.mean([per_job[i][s]["discounted_regret_mean"]
                                                                    for i in job_ids]))
                entry[s]["discounted_value_mean"] = float(np.mean([per_job[i][s]["discounted_value_mean"]
                                                                   for i in job_ids]))
        families[f"{w} | {fam}"] = entry

    # ---------------------------------------------------------------- discounted vs 100-step rankings
    ranking = {}
    for w in WORLDS:
        ranking[w] = {}
        ids = [k for k, j in enumerate(jobs) if j["world"] == w and "discounted_value_mean" in per_job[k][SITUATIONS[0]]]
        for s in SITUATIONS:
            a = np.array([per_job[k][s]["J100_first200_mean"] for k in ids])
            bvals = np.array([per_job[k][s]["discounted_value_mean"] for k in ids])
            tau = SS.kendalltau(a, bvals)
            rho = SS.spearmanr(a, bvals)
            n = len(ids)
            disc = sum(1 for p in range(n) for q in range(p + 1, n) if (a[p] - a[q]) * (bvals[p] - bvals[q]) < 0)
            ranking[w][s] = dict(controllers=n, kendall_tau=float(tau.statistic), spearman=float(rho.statistic),
                                 discordant_pairs=int(disc), pairs=n * (n - 1) // 2,
                                 discordant_share=disc / (n * (n - 1) / 2))
    # key paired contrasts per seed: do the two objectives agree on the sign?
    def find(world, source, rep, name, alpha=None):
        for rec in ctrl["records"]:
            if (rec["world"] == world and rec["source"] == source and rec["rep"] == rep and rec.get("name") == name
                    and (alpha is None or rec.get("alpha") == alpha)):
                return rec["job"]
        return None

    contrasts = []
    pairs = [("stall", "gate", "5", "full", "gapped", GATE_SEEDS, "harm (full log minus gapped log, alpha 5)"),
             ("stall", "gate", "2.5", "full", "gapped", GATE_SEEDS, "harm (full log minus gapped log, alpha 2.5)"),
             ("no_stall", "gate", "5", "no_stall_full", "no_stall_gapped", GATE_SEEDS, "no-stall harm (alpha 5)"),
             ("stall", "gate", "5", "gapped", "clone", GATE_SEEDS, "critic-guided minus cloning (alpha 5)"),
             ("stall", "stage2", None, "bca_k1", "none", STAGE2_SEEDS, "BCA (bca_k1) minus none"),
             ("stall", "stage2", None, "range_hinge", "none", STAGE2_SEEDS, "range hinge minus none"),
             ("stall", "stage2", None, "perfect_band", "none", STAGE2_SEEDS, "perfect band minus none"),
             ("no_stall", "stage2", None, "bca_k1", "none", STAGE2_SEEDS, "no-stall BCA (bca_k1) minus none")]
    for w, src, alpha, n1, n2, seeds, label in pairs:
        for r in seeds:
            j1, j2 = find(w, src, r, n1, alpha), find(w, src, r, n2, alpha)
            if j1 is None or j2 is None:
                continue
            for s in SITUATIONS:
                d100 = per_job[j1][s]["J100_first200_mean"] - per_job[j2][s]["J100_first200_mean"]
                dd = per_job[j1][s]["discounted_value_mean"] - per_job[j2][s]["discounted_value_mean"]
                dpts = per_job[j1][s]["points"] - per_job[j2][s]["points"]
                contrasts.append(dict(contrast=label, world=w, seed=r, situation=s, points_difference_full_bank=dpts,
                                      return100_difference_first200=d100, discounted_difference_first200=dd,
                                      same_sign=bool(d100 * dd > 0)))
    # the discount-optimal plan scored on the headline (100-step) objective
    disc_vs_100 = {}
    for w in WORLDS:
        disc_vs_100[w] = {}
        for s in SITUATIONS:
            J100plan = dz[f"J100plan_{w}_{s}"]
            Jopt = J_opt[(w, s)][:DISC_STARTS]
            an = anchors[w][s]
            gap = Jopt - J100plan
            disc_vs_100[w][s] = dict(
                optimum100_points_first200=float(_points(Jopt, an).mean()),
                discount_optimal_plan_points_first200=float(_points(J100plan, an).mean()),
                regret_points=float(_points(Jopt, an).mean() - _points(J100plan, an).mean()),
                regret_return=float(gap.mean()), regret_min_over_starts=float(gap.min()),
                discounted_optimum_value_mean=float(dz[f"value_{w}_{s}"].mean()))

    # ---------------------------------------------------------------- validation summary
    validation = dict(
        replay_max_relative_difference=max(v["replay_max_relative_difference"] for v in val.values()),
        frank_wolfe_gap_max=max(v["frank_wolfe_gap_max"] for v in val.values()),
        frank_wolfe_gap_max_relative=max(v["frank_wolfe_gap_max_relative"] for v in val.values()),
        projected_gradient_max=max(v["projected_gradient_max"] for v in val.values()),
        kkt_sign_violation_max=max(v["kkt_sign_violation_max"] for v in val.values()),
        trf_max_relative_cost_difference=max(v["trf_max_relative_cost_difference"] for v in val.values()),
        trf_cost_minus_bvls_cost_min=min(v["trf_cost_minus_bvls_cost_min"] for v in val.values()),
        random_perturbations_tried=sum(v["random_perturbation"]["tried"] for v in val.values()),
        random_perturbations_that_improved=sum(v["random_perturbation"]["improved"] for v in val.values()),
        random_perturbation_worst_relative_delta=min(v["random_perturbation"]["worst_relative_delta"]
                                                    for v in val.values()),
        stall_optimum_max_command=max(val[f"stall/{s}"]["max_command"] for s in SITUATIONS),
        knee=TC.KNEE,
        last_command_max_abs=max(v["last_command_max_abs"] for v in val.values()),
        every_controller_at_or_below_optimum_min_regret=lower_bound_violation,
        recorded_points_appearances_checked=int(len(rec_diffs)),
        recorded_points_max_abs_difference=float(rec_diffs.max()) if len(rec_diffs) else None,
        anchors_recomputed_max_abs_difference=anchor_check,
        evaluator_mean_return_identical_to_detail_rollout=all(
            j["per"][s]["J_mean_matches_evaluator"] for j in jobs for s in SITUATIONS),
        normalisation_checks=ctrl["meta"]["normalisation_checks"],
        gate_clone_files_equal_scored_clones=dict(checked=len(clone_same), all_equal=all(clone_same)),
        per_world_situation=val, nonconvex=nonconvex,
        discounted=dmeta, qr_condition=solve_meta["qr_condition"])

    import scipy

    results = dict(
        about="ThrusterChain's exact optimal reference controller (the best possible 100-step plan from each bank "
              "start) and the regret of saved controllers against it. Evaluator work only: no learner trained, no "
              "BCA band fitted. Exploratory; changes nothing in v6/v7.",
        definitions=dict(
            return_="100-step undiscounted return J = sum_{t=0}^{99} -(|s_t|^2 + 0.3|a_t|^2) (truth.rollout)",
            points="100 (J - J_recording) / (J_top - J_recording), the gate's anchors (pre-registration 68d9e877)",
            regret="optimum minus controller (return or points); >= 0 start by start",
            tracking_cost="sum_t |s_t|^2 over t = 0..99 (how far the floats are from rest)",
            energy_cost="0.3 sum_t |a_t|^2 (throttle cost); tracking + energy = -J",
            peak_deviation="per start, the largest |s_t| over t = 0..99, then mean (or max) over the bank",
            p5="5th percentile of per-start returns", worst5_mean="mean return of the worst 5% of starts",
            discounted_value="sum_{t=0}^{999} 0.98^t r_t (truth.value, horizon 1,000), first 200 starts per bank",
            at_bound="|a| >= bound - 1e-9; bound = float32(0.8) (stall) or 1 (no-stall)"),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         machine=platform.node(), workers=WORKERS, repository=str(ROOT),
                         gate_prereg=ctrl["meta"]["prereg_digest"], gate_outcome=ctrl["meta"]["gate_outcome"]),
        optimum=optimum, validation=validation, controllers=[
            dict(job=j["job"], label=j["label"], world=j["world"], rep=j["rep"], kind=j["kind"],
                 tree_sha256=j["tree_sha256"], seconds=j["seconds"], per=per_job[k]) for k, j in enumerate(jobs)],
        records=ctrl["records"], families=families, ranking=ranking, contrasts=contrasts,
        discount_optimal_plan_on_100_steps=disc_vs_100, float32_check=f32,
        files=dict(gate=ctrl["meta"]["gate_files"], stage2=ctrl["meta"]["stage2_files"]),
        anchors=anchors)
    return results


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


SIT_SHORT = {"recorded": "recorded", "strong_current": "strong cur.", "big_waves": "big waves", "big_kicks": "big kicks"}

KEY_FAMILIES = {
    "stall": [("reference:recording", "recording controller (0 pts)"), ("reference:hand_only", "normal driving only"),
              ("reference:stall_aware", "stall-aware (100 pts)"), ("reference:rising_thrust", "rising-thrust"),
              ("gate alpha 5: clone", "clone"), ("gate alpha 5: gapped", "none (gapped critic)"),
              ("gate alpha 5: full", "full (full-log critic)"), ("gate alpha 5: random", "random thinning"),
              ("gate alpha 5: td3bc_gapped", "plain TD3+BC"),
              ("gate alpha 2.5: gapped", "none, alpha 2.5"), ("gate alpha 2.5: full", "full, alpha 2.5"),
              ("stage2:none", "none, seeds 0-1 only"), ("stage2:full_none", "full, seeds 0-1 only"),
              ("stage2:bca_k1", "bca_k1"), ("stage2:bca_k2", "bca_k2"), ("stage2:bca_unpaired", "bca_unpaired"),
              ("stage2:bca_shuffled", "bca_shuffled"), ("stage2:perfect_band", "perfect band"),
              ("stage2:range_hinge", "range hinge"), ("stage2:even_pull (matched)", "even pull (matched)"),
              ("stage2:push_less (matched)", "push-less (matched)"), ("stage2:random_1 (matched)", "random 1 (matched)"),
              ("stage2:random_2 (matched)", "random 2 (matched)"), ("stage2:standin (matched)", "stand-in (matched)")],
    "no_stall": [("reference:recording", "recording controller (0 pts)"),
                 ("reference:hand_only", "normal driving only"), ("reference:stall_aware", "stall-aware"),
                 ("reference:rising_thrust", "rising-thrust (100 pts)"),
                 ("gate alpha 5: no_stall_clone", "clone"), ("gate alpha 5: no_stall_gapped", "none (gapped critic)"),
                 ("gate alpha 5: no_stall_full", "full (full-log critic)"),
                 ("stage2:none", "none, seeds 0-1 only"), ("stage2:bca_k1", "bca_k1"), ("stage2:perfect_band", "perfect band"),
                 ("stage2:range_hinge", "range hinge"), ("stage2:even_pull_locked", "even pull (locked)"),
                 ("stage2:standin (locked)", "stand-in (locked)")],
}


def write_markdown(R, timings):
    L_ = []
    a = L_.append
    a("# ThrusterChain: the exact best controller, and how far each controller falls short of it")
    a("")
    a("All numbers are exact float64 simulations. Nothing was trained; no BCA band was fitted. "
      "Points: 0 = the recording controller, 100 = the gate's top reference controller (stall-aware in the stalling "
      "world, rising-thrust in the no-stall world). 'Regret' = the optimum's score minus the controller's score, so a "
      "smaller regret is better and 0 would be perfect.")
    a("")
    a("## 1. The optimum's score (1,000 starts per situation)")
    a("")
    for w in WORLDS:
        a(f"**{TC.WORLD_LABELS[w]}**")
        a("")
        a("| situation | optimum points | optimum return | top ref. return |")
        a("|---|---|---|---|")
        for s in SITUATIONS:
            o = R["optimum"][w][s]
            a(f"| {SIT_SHORT[s]} | {fmt(o['points'])} | {fmt(o['return_']['mean'], 2)} | "
              f"{fmt(o['anchors']['top'], 2)} |")
        a("")
    a("## 2. Regret in points (optimum minus controller), mean over seeds")
    a("")
    a("Gate actors: seeds 0-4. Stage 2 (v6) arms: seeds 0-1. The controller's own points are in results.json.")
    a("")
    fam = R["families"]
    for w in WORLDS:
        a(f"**{TC.WORLD_LABELS[w]}**")
        a("")
        a("| controller | rec. | strong | waves | kicks |")
        a("|---|---|---|---|---|")
        a("| optimum (its points) | " + " | ".join(fmt(R["optimum"][w][s]["points"]) for s in SITUATIONS) + " |")
        for key, label in KEY_FAMILIES[w]:
            e = fam.get(f"{w} | {key}")
            if e is None:
                continue
            a(f"| {label} (n={e['n']}) | " + " | ".join(fmt(e[s]["regret_points_mean"]) for s in SITUATIONS) + " |")
        a("")
    a("**The same regret in return units** (optimum's mean return minus the controller's; the bank-mean return of "
      "the optimum is in section 1)")
    a("")
    for w in WORLDS:
        a(f"**{TC.WORLD_LABELS[w]}**")
        a("")
        a("| controller | rec. | strong | waves | kicks |")
        a("|---|---|---|---|---|")
        for key, label in KEY_FAMILIES[w]:
            e = fam.get(f"{w} | {key}")
            if e is None:
                continue
            a(f"| {label} | " + " | ".join(fmt(e[s]["regret_return_mean"], 2) for s in SITUATIONS) + " |")
        a("")
    a("## 3. Points of the same controllers (mean over seeds)")
    a("")
    for w in WORLDS:
        a(f"**{TC.WORLD_LABELS[w]}**")
        a("")
        a("| controller | rec. | strong | waves | kicks |")
        a("|---|---|---|---|---|")
        for key, label in KEY_FAMILIES[w]:
            e = fam.get(f"{w} | {key}")
            if e is None:
                continue
            a(f"| {label} | " + " | ".join(fmt(e[s]["points_mean"]) for s in SITUATIONS) + " |")
        a("")
    a("## 4. Checks")
    a("")
    v = R["validation"]
    nc = v["nonconvex"]
    a("| check | result |")
    a("|---|---|")
    a(f"| replay through the world vs the least-squares objective (largest relative gap, 8,000 plans) | "
      f"{v['replay_max_relative_difference']:.1e} |")
    a(f"| largest optimality gap guaranteed by the Frank-Wolfe bound (relative) | "
      f"{v['frank_wolfe_gap_max_relative']:.1e} |")
    a(f"| largest projected gradient | {v['projected_gradient_max']:.1e} |")
    a(f"| second solver (TRF) vs BVLS, largest relative cost gap (200 plans) | "
      f"{v['trf_max_relative_cost_difference']:.1e} |")
    a(f"| random and single-command nudges tried / that lowered the cost | "
      f"{v['random_perturbations_tried']:,} / {v['random_perturbations_that_improved']} |")
    a(f"| largest stalling-world optimal command (knee = 0.800000012) | {v['stall_optimum_max_command']:.9f} |")
    worst_nc = min(nc[k]["min_relative"] for k in NONCONVEX_INITS)
    lower = sum(nc[k]["runs_lower_than_best"] for k in NONCONVEX_INITS)
    sp = nc["sampled_plans"]
    os_ = nc["outward_slope_all_starts"]
    a(f"| stalling world, every at-knee optimal command ({os_['at_knee_commands']:,}): smallest slope of the cost "
      f"for pushing it past the knee (must be > 0) | {os_['min']:.3f} |")
    a(f"| stalling world, {sp['total']:,} sampled full-range plans with the true thrust ({sp['using_past_knee']:,} "
      f"use past-knee commands): plans cheaper than the optimum | {sp['lower_than_best']} (closest relative "
      f"difference {sp['min_relative']:.1e}: rounding) |")
    a(f"| stalling world, local search (L-BFGS-B) with the true thrust: runs ({nc['starts']} starts x "
      f"{len(NONCONVEX_INITS)} starting plans) that beat the optimum | {lower} (closest relative difference "
      f"{worst_nc:.1e}: rounding) |")
    a(f"| every controller's return <= the optimum's, start by start ({len(R['controllers'])} controllers x "
      f"4,000 starts; smallest regret seen, must be >= 0) | {v['every_controller_at_or_below_optimum_min_regret']:.1e} |")
    a(f"| the gate's saved cloned actors equal the 'clone' actors scored here | "
      f"{v['gate_clone_files_equal_scored_clones']['checked']} checked, all equal: "
      f"{v['gate_clone_files_equal_scored_clones']['all_equal']} |")
    a(f"| controllers' points vs their recorded values.json points ({v['recorded_points_appearances_checked']} "
      f"numbers) | largest difference {v['recorded_points_max_abs_difference']:.1e} |")
    a(f"| anchors recomputed vs pre-registered (largest difference) | "
      f"{max(v['anchors_recomputed_max_abs_difference'].values()):.1e} |")
    dm = v["discounted"]["per"]
    a(f"| discounted optimum: largest guaranteed relative gap / replay gap | "
      f"{max(x['fw_gap_max_relative'] for x in dm.values()):.1e} / "
      f"{max(x['replay_max_relative_difference'] for x in dm.values()):.1e} |")
    a("")
    a("## 5. How the optimum uses the thrusters")
    a("")
    a("Share of thruster-steps at the limit (0.8 in the stalling world, 1.0 in the no-stall world), and mean |command|.")
    a("")
    a("| world, situation | at limit | steps w/ any at limit | mean abs cmd | past 0.8 |")
    a("|---|---|---|---|---|")
    for w in WORLDS:
        for s in SITUATIONS:
            o = R["optimum"][w][s]
            a(f"| {'stall' if w == 'stall' else 'no-stall'}, {SIT_SHORT[s]} | "
              f"{100 * o['thruster_steps_at_bound_share']:.1f}% | "
              f"{100 * o['steps_with_any_thruster_at_bound_share']:.1f}% | {o['mean_abs_command']:.3f} | "
              f"{100 * o['share_past_0_8']:.1f}% |")
    a("")
    a("## 6. Where the loss comes from: tracking cost, energy cost, peak, worst starts")
    a("")
    a("Tracking = sum of squared float deviations; energy = 0.3 x sum of squared commands; both per episode "
      "(tracking + energy = minus the return). Late peak = the largest float deviation |s| over steps 10-99 of an "
      "episode, mean over starts (over all steps the peak is usually the start itself, which no controller "
      "changes: " + ", ".join(f"{100 * R['optimum'][w][s]['peak_is_the_start_share']:.0f}%" for w in ("stall",)
                              for s in SITUATIONS) + " of starts in the stalling world's four situations). "
      "Worst 5% = mean return of the worst 50 of 1,000 starts.")
    a("")
    rows = [("optimum", None)] + [(k, l) for k, l in KEY_FAMILIES["stall"]
                                  if k in ("reference:stall_aware", "gate alpha 5: clone", "gate alpha 5: gapped",
                                           "gate alpha 5: full", "stage2:bca_k1", "stage2:range_hinge")]
    for w in WORLDS:
        rws = rows if w == "stall" else [("optimum", None)] + [(k, l) for k, l in KEY_FAMILIES["no_stall"] if k in (
            "reference:rising_thrust", "gate alpha 5: no_stall_clone", "gate alpha 5: no_stall_gapped",
            "gate alpha 5: no_stall_full", "stage2:bca_k1")]
        for s in SITUATIONS:
            a(f"**{TC.WORLD_LABELS[w]}, {TC.SITUATION_LABELS[s]}**")
            a("")
            a("| controller | tracking | energy | late peak | worst 5% |")
            a("|---|---|---|---|---|")
            for k, label in rws:
                if k == "optimum":
                    o = R["optimum"][w][s]
                    a(f"| optimum | {fmt(o['tracking_cost_mean'])} | {fmt(o['energy_cost_mean'])} | "
                      f"{fmt(o['late_peak_deviation_mean'], 2)} | {fmt(o['return_']['worst5_mean'])} |")
                    continue
                e = fam.get(f"{w} | {k}")
                if e is None:
                    continue
                a(f"| {label} | {fmt(e[s]['tracking_cost_mean'])} | {fmt(e[s]['energy_cost_mean'])} | "
                  f"{fmt(e[s]['late_peak_deviation_mean'], 2)} | {fmt(e[s]['worst5_return_mean'])} |")
            a("")
    a("## 7. The discounted objective (what the learners optimise) vs the 100-step score")
    a("")
    a("Discounted value = sum over 1,000 steps of 0.98^t x reward (truth.py), first 200 starts of each bank.")
    a("")
    a("**7a. A perfect discounted optimiser, scored on the 100-step objective** (points on the same 200 starts)")
    a("")
    a("| world, situation | 100-step optimum | discount-optimal plan | regret |")
    a("|---|---|---|---|")
    for w in WORLDS:
        for s in SITUATIONS:
            d = R["discount_optimal_plan_on_100_steps"][w][s]
            a(f"| {'stall' if w == 'stall' else 'no-stall'}, {SIT_SHORT[s]} | {fmt(d['optimum100_points_first200'])} | "
              f"{fmt(d['discount_optimal_plan_points_first200'])} | {fmt(d['regret_points'], 2)} |")
    a("")
    a("**7b. Do the two objectives rank the controllers the same way?** (all deterministic controllers of the world)")
    a("")
    a("| world, situation | controllers | Kendall tau | pairs ordered differently |")
    a("|---|---|---|---|")
    for w in WORLDS:
        for s in SITUATIONS:
            r = R["ranking"][w][s]
            a(f"| {'stall' if w == 'stall' else 'no-stall'}, {SIT_SHORT[s]} | {r['controllers']} | "
              f"{r['kendall_tau']:.3f} | {r['discordant_pairs']} of {r['pairs']} ({100 * r['discordant_share']:.1f}%) |")
    a("")
    a("**7c. Key comparisons: same sign under both objectives?** (count of seed x situation cases)")
    a("")
    a("| comparison | same sign | cases |")
    a("|---|---|---|")
    groups = {}
    for c in R["contrasts"]:
        groups.setdefault(c["contrast"], []).append(c["same_sign"])
    for k, vv in groups.items():
        a(f"| {k} | {sum(vv)} | {len(vv)} |")
    a("")
    a("**7d. Discounted regret (discounted optimum minus controller), mean over seeds**")
    a("")
    for w in WORLDS:
        a(f"**{TC.WORLD_LABELS[w]}** (discounted optimum value: " + ", ".join(
            f"{SIT_SHORT[s]} {R['validation']['discounted']['per'][f'{w}/{s}']['value_mean']:.1f}"
            for s in SITUATIONS) + ")")
        a("")
        a("| controller | rec. | strong | waves | kicks |")
        a("|---|---|---|---|---|")
        for key, label in KEY_FAMILIES[w]:
            e = fam.get(f"{w} | {key}")
            if e is None or "discounted_regret_mean" not in e[SITUATIONS[0]]:
                continue
            a(f"| {label} | " + " | ".join(fmt(e[s]["discounted_regret_mean"], 2) for s in SITUATIONS) + " |")
        a("")
    a("## 8. Float32 deployment vs the float64 evaluator (full banks)")
    a("")
    a("| actor, situation | max diff (return) | 99.9% (return) | max diff (points) | bank mean (points) |")
    a("|---|---|---|---|---|")
    for f in R["float32_check"]:
        for s in SITUATIONS:
            p = f["per"][s]
            a(f"| {f['label'].split(':')[-1]}, {SIT_SHORT[s]} | {p['max_abs_return_difference']:.1e} | "
              f"{p['p999_abs_return_difference']:.1e} | {p['max_abs_points_difference']:.1e} | "
              f"{p['bank_mean_points_difference']:.1e} |")
    a("")
    a("## 9. Timings (seconds, wall clock)")
    a("")
    a("| phase | seconds |")
    a("|---|---|")
    for k, t in timings.items():
        a(f"| {k} | {t:.0f} |")
    a("")
    (out_path("results.md")).write_text("\n".join(L_) + "\n", encoding="utf8")


def phase_report(timer):
    R = build_report(timer)
    prior = {}
    tp = HERE / "timings.json"
    if tp.exists():
        prior = json.loads(tp.read_text(encoding="utf8"))
    prior.update(timer.t)
    R["timings"] = prior
    write_json("results.json", R)
    write_json("timings.json", prior)
    write_markdown(R, prior)


PHASES = ("solve", "validate", "nonconvex", "discounted", "score", "float32", "report")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=PHASES + ("all",))
    args = ap.parse_args()
    timer = Timer()
    phases = PHASES if args.phase == "all" else (args.phase,)
    for ph in phases:
        if ph == "report":
            timer.run("report", phase_report, timer)
        else:
            timer.run(ph, globals()[f"phase_{ph}"])
        if ph != "report":
            tp = HERE / "timings.json"
            prior = json.loads(tp.read_text(encoding="utf8")) if tp.exists() else {}
            prior[ph] = timer.t[ph]
            write_json("timings.json", prior)


if __name__ == "__main__":
    main()
