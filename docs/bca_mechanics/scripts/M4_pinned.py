"""Module M4 (the weights) on RECORDED ThrusterChain inputs, in the project's pinned environment.

What this script does (nothing is trained, no band is fitted): for each of the ten near-80 version-6 band records
(setting "near80", worlds stall and no_stall, data seeds 0-4) it
  1. loads the recorded held-out scores (v6_bank_scores.npz, 821-823 rows) and checks their SHA-256 against the band
     record (v6_band.json: bank_scores_sha256);
  2. regenerates the seed's recorded log with the project's own loader (the near-80 code's settingdata.gate_seed, which
     checks the log against the gate's data record), applies the near-80 removal, and checks that the kept held-out rows
     are exactly the rows the band was calibrated on (v6_band.json: data.sha256.bank_rows);
  3. rebuilds the posterior random stream the band used (stage2/code/band.py:244-249 and 317-320) and recomputes the
     unweighted threshold: it must equal the recorded one;
  4. M4 checks on those recorded scores: (a) weights of 1 and test mass 1 reproduce the unweighted threshold bit for
     bit; (b) a common factor of 1,000 on weights and test mass changes nothing; (c) exact weights for a declared tilt
     of the recorded current law (uniform on [0.1, 0.7]; tilt slopes 2 and 4 in log w per unit of current, the slopes
     of the weighting test's mild and strong shifts) give the weighted threshold, its ratio to the unweighted one, the
     Kish effective size and the two per-step ends; (d) the estimated-weight classifier (experiments/shift_iw/
     classifier.fit) fitted on recorded observations resampled with the known tilt recovers the slope 0, 2, 4, and its
     weights give a threshold close to the exact one; (e) every held-out (evaluation-split) row whose recorded command
     lies inside the removed ball is refused by a weighted arm and none outside (experiments/shift_iw/coverage.refusals).

How to run (inside WSL, the pinned environment: Python 3.10.20, numpy 2.2.6, jax 0.6.2 on CPU), from the repository root,
with exactly the gate's pre-registered flags (JAX_PLATFORMS=cpu and OMP_NUM_THREADS=1 set; JAX_ENABLE_X64, XLA_FLAGS and
JAX_DEFAULT_MATMUL_PRECISION unset: the data loader's software check, experiments/shift/acceptance.check_software,
refuses anything else):
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 \\
      /tmp/claude-0/bcavenv/bin/python docs/bca_mechanics/scripts/M4_pinned.py --out <file.json>
From Windows it was run through the bridge with docs/bca_mechanics/scripts/M4_pinned_job.sh (job name m4pinned).
The output JSON is committed as docs/bca_mechanics/data/M4_real_data.json and read by M4_test.py.
Inputs read: runs/shift_explore/near80/learners/near80/<world>/seed<k>/{v6_band.json, v6_bank_scores.npz} (also on the
branch claude/near80-run-files) and the gate's data records under runs/shift (through acceptance.load_seed).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
ROOT = "/home/user/bca"
for _flag, _want in (("JAX_PLATFORMS", "cpu"), ("OMP_NUM_THREADS", "1")):
    if os.environ.get(_flag) != _want:
        sys.exit(f"set {_flag}={_want} (the gate's pre-registered numeric flags)")
for _flag in ("JAX_ENABLE_X64", "XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION"):
    if os.environ.get(_flag) is not None:
        sys.exit(f"unset {_flag} (the gate's pre-registered numeric flags)")
NEAR80_CODE = ROOT + "/outputs/shift/beyond-the-data/near80/code"
for p in (NEAR80_CODE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import jax  # noqa: E402
from scipy import stats  # noqa: E402

from calibration import wbcp  # noqa: E402
import settingdata as SD  # noqa: E402  (the near-80 code's data loader)
import near80lib as N  # noqa: E402
from experiments.shift import acceptance as A  # noqa: E402
from experiments.shift import learners as L  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift_iw import classifier as CL  # noqa: E402
from experiments.shift_iw import coverage as CV  # noqa: E402

ALPHA, BETA, DRAWS = 0.1, 0.95, 1000           # calibration/reference.py:17-29 (WBCPConfig)
TAG_BAND, POSTERIOR_FOLD = 41, 1347375956      # band.py:85, 320; reference.py:177
C_LO, C_HI = TC.START_LAWS["recorded"]["current"]   # the recorded current law: uniform on [0.1, 0.7]
TILTS = (2.0, 4.0)                              # slopes of log w in c: the weighting test's mild and strong shifts
N_SIDE = 30_000                                 # classifier sample per side: 300 episodes x 100 steps (data.py:14)
SEEDS, WORLDS = (0, 1, 2, 3, 4), ("stall", "no_stall")


def M(lam):
    """The integral of exp(lam c) over the recorded current range [C_LO, C_HI] (world.py:95-100's normaliser, on this
    range)."""
    return (C_HI - C_LO) if lam == 0.0 else (math.exp(lam * C_HI) - math.exp(lam * C_LO)) / lam


def w_tilt(c, lam):
    """Exact weight p_test(c) / p_train(c) of a tilt exp(lam c) of the uniform recorded law (world.py:128-135's form
    with the recorded law in place of the weighting test's training law)."""
    return (C_HI - C_LO) * np.exp(lam * np.asarray(c, np.float64)) / M(lam)


def test_mass(lam):
    """E_test[w] = E_train[w^2] in closed form (world.py:150-154's form)."""
    return (C_HI - C_LO) * M(2.0 * lam) / M(lam) ** 2


def posterior_stream(seed, steps):
    """The posterior generator of a near-80 version-6 band: band.py:244-249 (refresh key) and 317-320 (the fold)."""
    rep_seed = int(np.random.default_rng([20261015, int(seed), TAG_BAND, 1]).integers(2 ** 31 - 1))
    refresh_key = jax.random.fold_in(jax.random.PRNGKey(rep_seed), int(steps))
    words = np.asarray(jax.random.fold_in(refresh_key, POSTERIOR_FOLD), np.uint32).ravel()
    return lambda: np.random.default_rng(words)


def cal(scores, stream, weights=None, t=None):
    return wbcp.calibrate(scores, stream(), weights, t, alpha=ALPHA, beta=BETA, draws=DRAWS)


def same(a, b):
    return bool(a.threshold == b.threshold and a.lambda_hat == b.lambda_hat and a.lambda_hpd == b.lambda_hpd
                and np.array_equal(a.crossings, b.crossings))


def load_seed(seed, world):
    try:
        return SD.gate_seed(seed, world)
    except Exception as error:  # the guard refused: load the gate's record without it
        print(f"   settingdata.gate_seed failed ({error!r}); using acceptance.load_seed directly", flush=True)
        p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
        return A.load_seed(p.runs, p.config, "base", world, int(seed))


def one_band(seed, world):
    t0 = time.perf_counter()
    d = f"{ROOT}/runs/shift_explore/near80/learners/near80/{world}/seed{seed}"
    rec = json.load(open(f"{d}/v6_band.json"))
    with np.load(f"{d}/v6_bank_scores.npz") as z:
        scores = np.asarray(z["scores"], np.float64)
    n = len(scores)
    out = dict(seed=seed, world=world, rows=n,
               scores_sha256_equal=hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest()
               == rec["bank_scores_sha256"])

    sd = load_seed(seed, world)
    log = sd.log
    rm = N.band_removal(log, sd.gap)
    kept_bank = rm.bank_rows(log)
    out["bank_rows_sha256_equal"] = bool(L.tree_sha256(kept_bank) == rec["data"]["sha256"]["bank_rows"]
                                         and len(kept_bank) == n == rec["data"]["bank_rows"])
    c = np.asarray(log["infos/state"][kept_bank], np.float64)[:, TC.CURRENT]
    obs_bank = np.asarray(log["observations"][kept_bank], np.float32)
    out["current_range_bank"] = [float(c.min()), float(c.max())]
    out["bank_rows_in_ball"] = int(sd.gap.contains(log["actions"][kept_bank]).sum())

    # 3. the recorded unweighted threshold from the recorded scores and the band's own stream
    stream = posterior_stream(seed, rec["steps"])
    cu = cal(scores, stream)
    out["recorded"] = dict(R=rec["uniform"]["R"], lambda_hat=rec["uniform"]["lambda_hat"],
                           lambda_hpd=rec["uniform"]["lambda_hpd"], unit=rec["uniform"]["unit"],
                           recomputed_by_the_run=rec["uniform"]["threshold_recomputed"]["recomputed"])
    out["unweighted"] = dict(R=cu.threshold, lambda_hat=cu.lambda_hat, lambda_hpd=cu.lambda_hpd, n_eff=cu.n_eff,
                             equals_recorded_float32=bool(np.float32(cu.threshold) == np.float32(rec["uniform"]["R"])),
                             equals_runs_recomputation=bool(cu.threshold
                                                            == rec["uniform"]["threshold_recomputed"]["recomputed"]))

    # 4a. weights of 1 and test mass 1 (mark W0's unit check, coverage.py:78-87)
    c1 = cal(scores, stream, np.ones(n), 1.0)
    out["weights_of_1_identical"] = same(cu, c1)
    # 4b. a common factor on weights and test mass (wbcp.py:83-85)
    w_rand = np.random.default_rng([20261020, 4, seed]).uniform(0.2, 5.0, n)
    out["common_factor_identical"] = same(cal(scores, stream, w_rand, 2.0), cal(scores, stream, 1000.0 * w_rand, 2000.0))

    # 4c. exact weights for a declared tilt of the recorded current law
    out["exact"] = {}
    for lam in TILTS:
        w = w_tilt(c, lam)
        t = test_mass(lam)
        cw = cal(scores, stream, w, t)
        ends = [cal(scores, stream, w, float(w_tilt(x, lam))).threshold for x in (C_LO, C_HI)]
        out["exact"][str(lam)] = dict(
            slope=lam, weight_range=[float(w.min()), float(w.max())], test_mass=t,
            R=cw.threshold, R_ratio=cw.threshold / cu.threshold, lambda_hat_ratio=cw.lambda_hat / cu.lambda_hat,
            margin_unweighted=cu.lambda_hpd / cu.lambda_hat, margin_weighted=cw.lambda_hpd / cw.lambda_hat,
            kish=cw.n_eff, kish_fraction=cw.n_eff / n, kish_fraction_formula=1.0 / t,
            per_step_R_at_smallest_mass=ends[0], per_step_R_at_largest_mass=ends[1],
            refused=bool(not math.isfinite(cw.threshold)))

    # 4d. the estimated-weight classifier on recorded observations resampled with a known tilt
    pool_rows = rm.kept_train_rows(log)
    pool_obs = np.asarray(log["observations"][pool_rows], np.float32)
    pool_c = np.asarray(log["infos/state"][pool_rows], np.float64)[:, TC.CURRENT]
    g = np.random.default_rng([20261020, 4, seed, WORLDS.index(world), 7])
    out["estimated"] = {}
    for lam in (0.0,) + TILTS:
        i0 = g.choice(len(pool_rows), N_SIDE, replace=True)
        p = w_tilt(pool_c, lam)
        i1 = g.choice(len(pool_rows), N_SIDE, replace=True, p=p / p.sum())
        clf = CL.fit(pool_obs[i0], pool_obs[i1])
        others = np.delete(np.abs(np.asarray(clf.coef)), TC.CURRENT)       # per standardised input: effect of 1 SD
        est = clf.log_weight(obs_bank)
        exact = np.log(w_tilt(c, lam))
        dd = (est - est.mean()) - (exact - exact.mean())
        rank = float(stats.spearmanr(est, exact)[0]) if np.ptp(exact) > 0 else None
        w_est = clf.weight(obs_bank)
        t_est = float(clf.weight(pool_obs[i1]).mean())      # classifier.estimated_arm's rule (no ball among kept rows)
        ce = cal(scores, stream, w_est, t_est)
        exact_R = cu.threshold if lam == 0.0 else out["exact"][str(lam)]["R"]
        out["estimated"][str(lam)] = dict(
            exact_slope=lam, fitted_slope_on_c=clf.slope_on_c, largest_other_standardised_coef=float(others.max()),
            current_standardised_coef=float(clf.coef[TC.CURRENT]), converged=bool(clf.converged),
            max_score=float(clf.max_score), iterations=int(clf.iterations), n0=int(clf.n0), n1=int(clf.n1),
            rank_correlation_at_bank=rank, rms_log_error_at_bank=float(np.sqrt(np.mean(dd ** 2))),
            test_mass=t_est, R=ce.threshold, R_ratio=ce.threshold / cu.threshold,
            R_over_exact=ce.threshold / exact_R, kish=ce.n_eff, kish_fraction=ce.n_eff / n)

    # 4e. refusal inside the ball on the held-out evaluation split's recorded rows (coverage.py:251-259)
    ev = log.split_rows("evaluation")
    in_ball = np.asarray(sd.gap.contains(log["actions"][ev]), bool)
    ratio_exact = np.where(in_ball, np.inf, w_tilt(np.asarray(log["infos/state"][ev], np.float64)[:, TC.CURRENT], 4.0))
    r_w, r_u = CV.refusals(in_ball, "exact"), CV.refusals(in_ball, "unweighted")
    out["refusal"] = dict(rows=int(len(ev)), in_ball_share=float(in_ball.mean()),
                          weighted_refused_inside=float(r_w[in_ball].mean()),
                          weighted_refused_outside=float(r_w[~in_ball].mean()),
                          unweighted_refused=float(r_u.mean()),
                          infinite_ratio_exactly_inside=bool(np.array_equal(np.isinf(ratio_exact), in_ball)))
    out["seconds"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--worlds", default="stall,no_stall")
    args = ap.parse_args()
    bands = []
    for world in args.worlds.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            print(f"band {world} seed {seed} ...", flush=True)
            b = one_band(seed, world)
            bands.append(b)
            e2, e4 = b["exact"]["2.0"], b["exact"]["4.0"]
            s = b["estimated"]
            print(f"   rows {b['rows']}  scores sha ok {b['scores_sha256_equal']}  bank rows sha ok "
                  f"{b['bank_rows_sha256_equal']}  R_u {b['unweighted']['R']:.5f} (recorded {b['recorded']['R']:.5f}, "
                  f"equal f32 {b['unweighted']['equals_recorded_float32']})  w=1 identical {b['weights_of_1_identical']}"
                  f"  x1000 identical {b['common_factor_identical']}", flush=True)
            print(f"   exact slope 2: R ratio {e2['R_ratio']:.3f} kish {e2['kish_fraction']:.3f} | slope 4: R ratio "
                  f"{e4['R_ratio']:.3f} kish {e4['kish_fraction']:.3f} ends {e4['per_step_R_at_smallest_mass']:.3f}.."
                  f"{e4['per_step_R_at_largest_mass']:.3f}", flush=True)
            print(f"   fitted slopes {s['0.0']['fitted_slope_on_c']:+.3f} / {s['2.0']['fitted_slope_on_c']:.3f} / "
                  f"{s['4.0']['fitted_slope_on_c']:.3f} (exact 0/2/4); estimated R ratio "
                  f"{s['0.0']['R_ratio']:.3f} / {s['2.0']['R_ratio']:.3f} / {s['4.0']['R_ratio']:.3f}; refusal inside "
                  f"{b['refusal']['weighted_refused_inside']:.2f} outside {b['refusal']['weighted_refused_outside']:.2f} "
                  f"({b['seconds']:.0f} s)", flush=True)
    env = dict(python=sys.version.split()[0], numpy=np.__version__, jax=jax.__version__,
               recorded_current_law=[C_LO, C_HI], tilt_slopes=list(TILTS), classifier_rows_per_side=N_SIDE,
               alpha=ALPHA, credibility=BETA, draws=DRAWS, written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(dict(environment=env, bands=bands), open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
