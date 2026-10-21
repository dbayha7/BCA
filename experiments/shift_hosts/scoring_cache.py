"""The scoring cache: fixed, non-learned reference numbers per bank that every score needs (plan sections 2.3, 5.4 and
9.2), computed once, before any learner runs. Scoring environment (/opt/bca-shift-venv, numpy 2.2.6).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. No learned
policy is involved anywhere here: only the zero command, the stall-aware controller, the two 0-point controllers and
the exact best plans. Everything is float64 simulation through truth.rollout (thruster_chain.dynamics) in the stalling
world.

Per bank (the 7 banks of INTERFACES.md section 3), runs/shift_hosts/scoring_cache/banks/<index>_<bank>.npz (handoff.py:
manifest and SHA-256 sidecars) holds:
  starts               float64 [1000, 13], the bank's starts (evaluate.starts)
  zero/J               the zero-command controller's 100-step return per start (F1: a policy fails a start when its
                       return is below this; the weighting test's ZeroCommand, experiments/shift_iw/data.py:609-615)
  stall_aware/J        the stall-aware controller (recorder.stall_aware: the textbook gain capped at 80%) per start
  stall_aware/late_peak  its late peak per start (evaluate.late_peak: oracle.py:140-141, 878-884)
  f2/threshold         0-d: the 99th percentile of stall_aware/late_peak (numpy's default linear interpolation), F2's
                       threshold; the controller itself exceeds it on 1% of the starts by construction
  anchor0/J            the 0-point controller per start: the weighting test's random quality on its fixed streams
                       (data.quality_on_bank("random"), data.py:630-636) on the weighting banks; the main study's
                       recording controller on its fixed streams (truth.recorder_on_bank) on the four situations
  optimum/J, optimum/Jd, optimum/U, optimum/status, optimum/nit, optimum/fw_gap
                       the exact best 100-command plan per start (regret): outputs/shift/oracle/oracle.py's Plan100 and
                       scipy's lsq_linear (BVLS, bounded-variable least squares, an exact active-set method; tol
                       1e-14, bounds +-knee), imported unchanged. For the weighting banks the loop over starts
                       (_solve_chunk, oracle.py:321-337) is restated with these banks' starts, because _solve_chunk
                       reads only TC.bank_starts; for the four situations _solve_chunk itself is called (plan section
                       2.3). Each plan is replayed through truth.rollout (J, Jd) and certified by the oracle's own
                       checks (phase_validate, oracle.py:420-459): replay against the least-squares cost,
                       the Frank-Wolfe gap (a bound on the distance to the best cost), the projected gradient and the
                       gradient's signs at the bounds (the optimality conditions of a bounded problem)
  viewer/starts, viewer/percentile, viewer/target, viewer/J, viewer/extra
                       the viewer's 10 fixed starts (plan section 9.2): the starts whose stall-aware return is nearest
                       its 95th, 85th, ..., 5th percentile on the bank (easy to hard; numpy's percentile, linear; ties
                       go to the lowest index), and for the four situations the page's existing start 11 (D.start, the
                       0-based index into the main study's banks) unless it is already among them
And per bank, runs/shift_hosts/scoring_cache/cql_noise/<index>_<bank>.npz: CQL's declared scoring noise eps float32
[1000, 100, 4] (policies.cql_noise; choice 7). runs/shift_hosts/scoring_cache/cache.json (with sidecar) records every
check, the sources and their SHA-256, the timings and the environment.

Checks (cache.json "checks"; "passed" is their conjunction):
  anchors    the weighting banks' random-quality, stall-aware and zero-command means recomputed here equal
             start_banks.json (on the strong bank -1,351.41 and -80.61); the situations' stall-aware and recording
             means equal preregistration.json's "base/stall" anchors
  optimum    every BVLS status positive; replay within 1e-8 relative (the oracle's check (a)); Frank-Wolfe gap within
             1e-9 of the cost; no command beyond the knee; on the four situations the optimum's points equal the
             oracle's published results.json (100.8 / 109.0 / 102.6 / 106.7) within 1e-6 points
  regret     every reference controller here (zero command, stall-aware, 0-point controller) is at or below the
             optimum on every start (within 1e-9 relative)
  F2, viewer the stall-aware controller exceeds its own threshold on exactly 1% of starts; the viewer's starts are
             distinct
Run (scoring environment, up to 4 processes on cores 20-23):
  cd /root/bca-hosts && env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu taskset -c 20-23 \\
      /opt/bca-shift-venv/bin/python -m experiments.shift_hosts.scoring_cache --workers 4
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401  (dependency path and version checks; numpy 2.2.6 only)
from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC
from experiments.shift import truth as TR
from experiments.shift_iw import data as D
from experiments.shift_hosts import evaluate as E
from experiments.shift_hosts import handoff as H
from experiments.shift_hosts import policies as P

SCHEMA = "shift-hosts-scoring-cache-v1"
REPO = E.REPO
ORACLE_PATH = REPO / "outputs" / "shift" / "oracle" / "oracle.py"
ORACLE_SHA256 = "01bda76ce243432d3a65ff7059b88a065b6e1ba606132b8baae18b1e6af8c861"      # commit 9aae4df
ORACLE_RESULTS = REPO / "outputs" / "shift" / "oracle" / "results.json"
ORACLE_RESULTS_SHA256 = "816d582e7588188f9ca4ee2b3262ea85aeea38ed9650deb31861237807df75a2"
ORACLE_MODULE = "shift_hosts_oracle"
CHUNK = 125                                   # starts per solve job (oracle.py phase_solve uses 125)
PERCENTILES = (95, 85, 75, 65, 55, 45, 35, 25, 15, 5)   # easy to hard (plan section 9.2)
PAGE_START = 11                               # the page's D.start, a 0-based index into the main study's banks
REPLAY_TOL = 1e-8                             # the oracle's check (a): replay within 1e-8 relative
FW_TOL = 1e-9                                 # Frank-Wolfe gap relative to the cost
REGRET_TOL = 1e-9                             # a reference controller may exceed the optimum by rounding only
ANCHOR_TOL = 1e-9                             # relative; the recomputed anchors are expected to be exact
PUBLISHED_POINTS_TOL = 1e-6

_ORACLE = {}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle():
    """outputs/shift/oracle/oracle.py imported unchanged by path (it is not in a package), once per process, after
    checking its SHA-256. Importing it sets only environment defaults (PYTHONDONTWRITEBYTECODE, JAX_PLATFORMS=cpu, one
    BLAS thread; oracle.py:111-116) and puts the repository root on sys.path."""
    if "module" not in _ORACLE:
        digest = sha256_file(ORACLE_PATH)
        if digest != ORACLE_SHA256:
            raise RuntimeError(f"{ORACLE_PATH} differs from its pinned SHA-256 (stop rule S1)")
        spec = importlib.util.spec_from_file_location(ORACLE_MODULE, ORACLE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[ORACLE_MODULE] = module
        spec.loader.exec_module(module)
        if module.LATE_PEAK_FROM != E.LATE_PEAK_FROM:
            raise RuntimeError("evaluate.LATE_PEAK_FROM differs from the oracle's")
        if module.physics_of("stall") != E.PHYSICS:
            raise RuntimeError("the oracle's stalling world differs from evaluate.PHYSICS")
        _ORACLE["module"] = module
    return _ORACLE["module"]


# ------------------------------------------------------------------------------------------------ the optimum

def _solve_weighting_chunk(bank, idx, X):
    """BVLS on the reduced problem for some starts of a weighting bank: oracle.py:321-337 (_solve_chunk) restated
    line for line, with the weighting bank's starts X in place of TC.bank_starts(situation)[idx] (oracle.py:328), in
    the stalling world."""
    from scipy.optimize import lsq_linear

    OR = oracle()
    plan = OR.plan100()
    b = OR.BOUND["stall"]
    rhs = -(plan.m(X) @ plan.Q)
    U = np.empty((len(X), plan.L * OR.DA))
    info = np.empty((len(X), 3))
    for i in range(len(X)):
        t0 = time.perf_counter()
        res = lsq_linear(plan.R, rhs[i], bounds=(-b, b), method="bvls", tol=OR.BVLS_TOL, max_iter=20000)
        U[i] = np.clip(res.x, -b, b)
        info[i] = (res.status, res.nit, time.perf_counter() - t0)
    return bank, idx, U, info


def _solve_job(job):
    """One solve job in a worker process: ("weighting", bank, idx, X) or ("situation", bank, idx)."""
    kind, bank, idx = job[:3]
    if kind == "weighting":
        return _solve_weighting_chunk(bank, idx, job[3])
    world, situation, idx, U, info = oracle()._solve_chunk(("stall", bank, idx))
    return situation, idx, U, info


def solve(banks, workers, chunk=CHUNK):
    """{bank: (U [1000, 400], info [1000, 3])} for the banks, solved in `workers` fresh spawn processes (no fork; the
    oracle's own pool_map rule, oracle.py:207-216)."""
    import multiprocessing as mp

    jobs = []
    for bank in banks:
        n = len(E.starts(bank))
        X = E.starts(bank)
        for i in range(0, n, chunk):
            idx = np.arange(i, min(i + chunk, n))
            if bank in E.WEIGHTING:
                jobs.append(("weighting", bank, idx, X[idx]))
            else:
                jobs.append(("situation", bank, idx))
    if workers <= 1:
        out = [_solve_job(j) for j in jobs]
    else:
        pool = mp.get_context("spawn").Pool(int(workers))
        try:
            out = pool.map(_solve_job, jobs, chunksize=1)
        finally:
            pool.close()
            pool.join()
    plan = oracle().plan100()
    res = {b: (np.full((len(E.starts(b)), plan.L * oracle().DA), np.nan), np.full((len(E.starts(b)), 3), np.nan))
           for b in banks}
    for bank, idx, U, info in out:
        res[bank][0][idx], res[bank][1][idx] = U, info
    return res


def certify(bank, X, U, info):
    """The oracle's checks (a) and (b) on one bank's plans (phase_validate, oracle.py:420-459, without the random
    perturbations), and the plans' replayed returns. Returns (J, Jd, fw_gap, record)."""
    OR = oracle()
    plan = OR.plan100()
    b = OR.BOUND["stall"]
    phys = OR.physics_of("stall")
    C = plan.cost(U, X)
    run = TR.rollout(OR.OpenLoop(U.reshape(-1, OR.LENGTH, OR.DA)), X, phys)        # replay through dynamics()
    rel = np.abs(-run.J - C) / np.abs(C)
    G = plan.grad(U, X)
    fw = OR.fw_gap(U, G, b)
    pg = np.abs(OR.projected_gradient(U, G, b)).max(1)
    free = np.abs(U) < b - OR.AT_BOUND
    up, lo = U >= b - OR.AT_BOUND, U <= -b + OR.AT_BOUND
    sign_violation = max(float(np.abs(G[free]).max()) if free.any() else 0.0,
                         float(np.maximum(G[up], 0).max()) if up.any() else 0.0,
                         float(np.maximum(-G[lo], 0).max()) if lo.any() else 0.0)
    statuses, counts = np.unique(info[:, 0], return_counts=True)
    record = dict(replay_max_relative_difference=float(rel.max()), frank_wolfe_gap_max=float(fw.max()),
                  frank_wolfe_gap_max_relative=float((fw / C).max()), frank_wolfe_gap_min=float(fw.min()),
                  projected_gradient_max=float(pg.max()), kkt_sign_violation_max=sign_violation,
                  bvls_status_counts={str(int(k)): int(v) for k, v in zip(statuses, counts)},
                  bvls_iterations_mean=float(info[:, 1].mean()), bvls_iterations_max=int(info[:, 1].max()),
                  bvls_seconds_mean=float(info[:, 2].mean()), bvls_seconds_total=float(info[:, 2].sum()),
                  max_command=float(np.abs(U).max()), bound=float(b),
                  last_command_max_abs=float(np.abs(U[:, -OR.DA:]).max()))
    record["passed"] = bool(np.all(info[:, 0] > 0) and rel.max() <= REPLAY_TOL and (fw / C).max() <= FW_TOL
                            and np.abs(U).max() <= b and np.all(np.isfinite(run.J)))
    return run.J, run.Jd, fw, record


# ------------------------------------------------------------------------------------------------ fixed controllers

def anchor_controller(bank, n):
    """The 0-point controller of a bank (a fresh one per bank: its streams start at the bank's first step)."""
    if bank in E.WEIGHTING:
        return D.quality_on_bank("random", n, hold=True)
    return TR.recorder_on_bank(E.PHYSICS, n=n)


def viewer(J_sa, bank):
    """The viewer's fixed starts (module docstring): (starts, percentile, target, extra)."""
    target = np.percentile(J_sa, PERCENTILES)
    picks = np.array([int(np.argmin(np.abs(J_sa - q))) for q in target], np.int64)
    extra = np.zeros(0, np.int64)
    if bank in E.SITUATIONS and PAGE_START not in set(picks.tolist()):
        extra = np.array([PAGE_START], np.int64)
    return picks, np.asarray(PERCENTILES, np.float64), target, extra


def reference_bank(bank, U, info):
    """Every cached array of one bank and its checks (module docstring)."""
    X = E.starts(bank)
    n = len(X)
    phys = E.PHYSICS
    zero = TR.rollout(D.ZeroCommand(), X, phys).J
    sa_run = TR.rollout(RC.stall_aware(phys), X, phys, record=True)
    sa_J, sa_peak = sa_run.J, E.late_peak(sa_run.states)
    a0 = TR.rollout(anchor_controller(bank, n), X, phys).J
    threshold = float(np.quantile(sa_peak, E.F2_QUANTILE))
    J_opt, Jd_opt, fw, opt_record = certify(bank, X, U, info)
    picks, pct, target, extra = viewer(sa_J, bank)
    arrays = {"starts": X, "zero/J": zero, "stall_aware/J": sa_J, "stall_aware/late_peak": sa_peak,
              "f2/threshold": np.asarray(threshold, np.float64), "anchor0/J": a0, "optimum/J": J_opt,
              "optimum/Jd": Jd_opt, "optimum/U": U.reshape(n, E.LENGTH, TC.DA),
              "optimum/status": info[:, 0].astype(np.int8), "optimum/nit": info[:, 1].astype(np.int32),
              "optimum/fw_gap": fw, "viewer/starts": picks.astype(np.int32), "viewer/percentile": pct,
              "viewer/target": target, "viewer/J": sa_J[picks], "viewer/extra": extra.astype(np.int32)}
    # checks
    regret_min = {}
    for name, J in (("zero_command", zero), ("stall_aware", sa_J), ("anchor0", a0)):
        regret_min[name] = float(np.min((J_opt - J) / np.maximum(1.0, np.abs(J_opt))))
    a = E.anchor(bank)
    checks = dict(optimum=opt_record, regret_min_relative=regret_min,
                  regret_passed=bool(min(regret_min.values()) >= -REGRET_TOL),
                  f2_threshold=threshold, f2_stall_aware_share=float(np.mean(sa_peak > threshold)),
                  f2_passed=bool(np.mean(sa_peak > threshold) == 0.01),
                  viewer_distinct=bool(len(set(picks.tolist()) | set(extra.tolist())) == len(picks) + len(extra)))
    means = dict(zero_command=float(zero.mean()), stall_aware=float(sa_J.mean()), anchor0=float(a0.mean()),
                 optimum=float(J_opt.mean()))
    if bank in E.WEIGHTING:
        rec = E.weighting_banks()[1]["anchors"][bank]
        want = dict(anchor0=rec["anchor"]["random"], stall_aware=rec["anchor"]["top"],
                    zero_command=rec["controllers"]["zero_command"]["mean"])
        source = "start_banks.json"
    else:
        pre = E.prereg_anchors()[bank]
        want = dict(anchor0=pre["recording"], stall_aware=pre["top"])
        source = "preregistration.json anchors base/stall"
    diffs = {k: float(means[k] - v) for k, v in want.items()}
    checks["anchors"] = dict(source=source, recomputed={k: means[k] for k in want}, recorded=want, differences=diffs,
                             passed=bool(all(abs(d) <= ANCHOR_TOL * abs(want[k]) for k, d in diffs.items())))
    opt_points = float(np.mean(E.points(J_opt, bank)))
    checks["optimum_points"] = opt_points
    if bank in E.SITUATIONS:
        pub = published_optimum()[bank]
        checks["published"] = dict(points=pub["points"], return_mean=pub["return_mean"],
                                   points_difference=opt_points - pub["points"],
                                   return_mean_difference=means["optimum"] - pub["return_mean"],
                                   rounded=round(opt_points, 1),
                                   passed=bool(abs(opt_points - pub["points"]) <= PUBLISHED_POINTS_TOL))
    if bank == "strong":
        checks["plan_quoted_strong_anchors"] = dict(
            random=round(a["bottom"], 2), stall_aware=round(a["top"], 2),
            passed=bool(round(a["bottom"], 2) == -1351.41 and round(a["top"], 2) == -80.61))
    checks["passed"] = bool(opt_record["passed"] and checks["regret_passed"] and checks["f2_passed"]
                            and checks["viewer_distinct"] and checks["anchors"]["passed"]
                            and checks.get("published", {}).get("passed", True)
                            and checks.get("plan_quoted_strong_anchors", {}).get("passed", True))
    meta = dict(schema=SCHEMA, bank=bank, index=P.bank_index(bank), kind="weighting" if bank in E.WEIGHTING
                else "situation", world="stalling world", physics=E.PHYSICS.as_dict(), anchor=a, means=means,
                controllers=dict(zero="the zero command (data.ZeroCommand)",
                                 stall_aware=RC.stall_aware(E.PHYSICS).name,
                                 anchor0=anchor_controller(bank, 1).name if bank in E.WEIGHTING
                                 else "the main study's recording controller (truth.recorder_on_bank)",
                                 optimum="exact best 100-command plan (oracle.py Plan100 + lsq_linear BVLS)"),
                late_peak=f"max over steps {E.LATE_PEAK_FROM}-99 of the norm of the 10 float position and speed "
                          "numbers (oracle.py:140-141, 878-884)",
                f2_quantile=E.F2_QUANTILE, viewer=dict(percentiles=list(PERCENTILES), order="easy to hard",
                                                       method="numpy.percentile (linear); nearest return, lowest "
                                                              "index on ties",
                                                       page_start=PAGE_START if bank in E.SITUATIONS else None),
                checks=checks)
    return arrays, meta


def published_optimum():
    """The oracle's published optimum per situation (results.json "optimum" "stall"), after its pinned SHA-256."""
    if sha256_file(ORACLE_RESULTS) != ORACLE_RESULTS_SHA256:
        raise RuntimeError(f"{ORACLE_RESULTS} differs from its pinned SHA-256")
    opt = json.loads(ORACLE_RESULTS.read_text(encoding="utf8"))["optimum"]["stall"]
    return {s: dict(points=float(opt[s]["points"]), return_mean=float(opt[s]["return_"]["mean"])) for s in opt}


def processor_seconds():
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the scoring cache (fixed controllers and exact best plans).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=str(E.CACHE))
    p.add_argument("--banks", nargs="*", default=list(P.BANKS))
    a = p.parse_args(argv)
    out = Path(a.out).resolve()
    allowed = (REPO / "runs" / "shift_hosts").resolve()
    if allowed != out and allowed not in out.parents:
        raise SystemExit(f"refused: {out} is outside {allowed}")
    banks = [E.bank_name(b) for b in a.banks]
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0, cpu0 = time.perf_counter(), processor_seconds()
    oracle()
    timings = {}
    t = time.perf_counter()
    solved = solve(banks, a.workers)
    timings["solve_seconds"] = time.perf_counter() - t
    summary = dict(schema=SCHEMA, started_utc=started_utc, banks={}, files={})
    for bank in banks:
        t = time.perf_counter()
        U, info = solved[bank]
        arrays, meta = reference_bank(bank, U, info)
        meta["starts_source"] = (E.weighting_banks()[2] if bank in E.WEIGHTING
                                 else dict(source="thruster_chain.bank_starts", stream=list(TC.BANK_SEED)))
        meta["starts_sha256"] = H.array_sha256(arrays["starts"])
        man = H.write(E.cache_path(bank, out), arrays, meta=meta)
        eps = P.cql_noise(bank)
        nman = H.write(E.noise_path(bank, out), {"eps": eps},
                       meta=dict(schema=SCHEMA, kind="cql scoring noise", bank=bank, index=P.bank_index(bank),
                                 rule="np.random.default_rng([20261016, 1004, index]).standard_normal((1000, 100, 4))"
                                      ".astype(np.float32)", stream=[P.MASTER_SEED, P.CQL_NOISE_STREAM,
                                                                     P.bank_index(bank)]))
        timings[f"{bank}_seconds"] = time.perf_counter() - t
        summary["banks"][bank] = dict(checks=meta["checks"], means=meta["means"], anchor=meta["anchor"],
                                      viewer_starts=arrays["viewer/starts"].tolist(),
                                      viewer_extra=arrays["viewer/extra"].tolist(), f2_threshold=meta["checks"][
                                          "f2_threshold"], starts_sha256=meta["starts_sha256"])
        summary["files"][bank] = dict(cache=str(E.cache_path(bank, out).relative_to(REPO)),
                                      cache_npz_sha256=man["npz_sha256"],
                                      cql_noise=str(E.noise_path(bank, out).relative_to(REPO)),
                                      cql_noise_npz_sha256=nman["npz_sha256"],
                                      cql_noise_eps_sha256=nman["arrays"]["eps"]["sha256"])
        c = meta["checks"]
        print(f"{bank:15s} optimum {meta['means']['optimum']:9.3f} ({c['optimum_points']:7.3f} pts) "
              f"stall-aware {meta['means']['stall_aware']:9.3f} zero {meta['means']['zero_command']:9.3f} "
              f"F2 {c['f2_threshold']:.4f} viewer {arrays['viewer/starts'].tolist()} "
              f"{'passed' if c['passed'] else 'FAILED'}", flush=True)
    summary["passed"] = bool(all(v["checks"]["passed"] for v in summary["banks"].values()))
    timings["wall_seconds"] = time.perf_counter() - t0
    timings["processor_seconds"] = processor_seconds() - cpu0
    summary.update(timings=timings, workers=a.workers, affinity=sorted(os.sched_getaffinity(0)),
                   environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                                    scipy=__import__("scipy").__version__, executable=sys.executable,
                                    host=platform.node(), omp_num_threads=os.environ.get("OMP_NUM_THREADS")),
                   oracle=dict(path=str(ORACLE_PATH.relative_to(REPO)), sha256=ORACLE_SHA256,
                               results_sha256=ORACLE_RESULTS_SHA256, bvls_tol=oracle().BVLS_TOL),
                   code_sha256=E.code_sha256(), tolerances=dict(replay=REPLAY_TOL, frank_wolfe=FW_TOL,
                                                                regret=REGRET_TOL, anchors=ANCHOR_TOL,
                                                                published_points=PUBLISHED_POINTS_TOL))
    H.write_json(out / "cache.json", summary)
    print(f"scoring cache {'PASSED' if summary['passed'] else 'FAILED'}: {out} "
          f"({timings['wall_seconds']:.0f} s, {timings['processor_seconds']:.0f} processor-s)", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
