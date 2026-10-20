"""Module M4 (the weights): synthetic checks with a known truth, the real-data table, and the figure.

How to run (Windows scratch venv: Python 3.14, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2; any Python 3.10+ with
numpy, scipy and matplotlib works), from the repository root:
    set PYTHONPATH=M:\\   (the repository root: the threshold is the project's own calibration/wbcp.py)
    python docs\\bca_mechanics\\scripts\\M4_test.py
Options: --real <json> (default docs/bca_mechanics/data/M4_real_data.json, written by M4_pinned.py in the pinned
environment through the bridge; the real-data table is skipped when the file is missing), --figure <svg> (default
docs/bca_mechanics/figures/M4.svg), --fast (fewer repeats; the bars below are set for the full run), --no-figure.
The weighting test's recorded slopes (panel B of the figure) are read from outputs/shift_iw/v5/<quality>/report.json
when the worktree has them, else from branch origin/main with git show.

Bars, fixed before the run (each check prints PASS or FAIL against them):
  S1  every constant within 1e-9 of spec.json (ranges and Kish rows within the rounding printed there)
  S2  identical on every bank (threshold, both quantiles and all 1,000 crossings)
  S3  identical in every case
  S4  independent rows: mean fitted slope within 0.05 of exact; 300 episodes x 100 steps a side (the weighting test's
      shape): within 0.35 (the record's own deviations are 0.19-0.28); 3,000 episodes x 10 steps a side: within 0.15
      and with a smaller spread than the 300-episode fits (the same rows, ten times as many distinct currents); the
      bias of each shape is printed, not judged
  S5  exactly the in-ball steps refused by a weighted arm; none by the unweighted arm
  S6  no shift, unweighted: coverage in [0.89, 0.95]; strong shift, unweighted: below 0.88; strong shift, the plain
      weighted 90% quantile alone (reweighting, no margin): in [0.88, 0.92]; the full exact-weighted threshold: at
      least 0.90 and at most 0.97; |estimated - exact| at most 0.02; weights of 1 identical on every bank

Nothing here trains a learner or fits a band on study data. The synthetic checks build their own observations, scores
and weights with a known truth. The estimated-weight classifier is the project's experiments/shift_iw/classifier.fit
when that module imports (it needs the pinned numpy 2.2.6 because it imports experiments.shift); otherwise the same
Newton fit is used from a line-for-line transcription below (classifier.py:69-101), and the output says which one ran.

Checks (each prints PASS or FAIL):
  S1  the known shift's constants: M(-2), M(2), the weight constants, ranges, test masses and Kish fractions of
      experiments/iw/spec.json follow from the formulas of experiments/shift_iw/world.py:95-154
  S2  weights of 1 and a test mass of 1 reproduce the unweighted threshold bit for bit (mark W0's unit check)
  S3  a common factor on the weights and the test mass changes nothing (wbcp.py:83-85)
  S4  the classifier recovers the slope of log w in the current (0, 2, 4) on synthetic observations with a known tilt:
      rows drawn independently (tight); 300 episodes of 100 steps a side as in the weighting test (one current per
      episode: a wide spread, like the record's); 3,000 episodes of 10 steps (the spread shrinks)
  S5  a weighted arm refuses exactly the test steps whose command lies inside the ball; the unweighted arm none
      (world.py:160-164, coverage.py:251-259)
  S6  the purpose: under a known shift the unweighted band's coverage falls below 90%; the plain weighted 90% quantile
      (the weights alone, no Bayesian margin) brings it back to about 90%; the full exact-weighted threshold keeps the
      promise with the margin on top; the estimated-weighted one is within 0.02 of the exact one (marks W1, W2, W4)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from calibration import wbcp
except ImportError as error:
    sys.exit(f"set PYTHONPATH to the repository root (calibration/wbcp.py): {error}")

ALPHA, BETA, DRAWS = 0.1, 0.95, 1000
C_LO, C_HI = 0.1, 1.1                 # the weighting test's current range (world.py:69)
LAMBDA_TRAIN = -2.0                   # the training law's exponent (world.py:72)
TILT = {"none": -2.0, "mild": 0.0, "strong": 2.0}
CURRENT = 10                          # the current's index in the 13-number observation (thruster_chain.py:70)
DX = 13
SEED = 20261020

# ------------------------------------------------------------------------------------------ the project's classifier

CLASSIFIER_SOURCE = "experiments/shift_iw/classifier.py (imported)"
try:
    from experiments.shift_iw import classifier as _CL
    fit_classifier = _CL.fit
except Exception:  # the pinned numpy guard of experiments.shift, or a missing dependency
    CLASSIFIER_SOURCE = "transcription of experiments/shift_iw/classifier.py:27-101 (the import needs the pinned numpy)"

    class _Classifier:
        def __init__(self, intercept, coef, mean, std, n0, n1, iterations, converged, max_score):
            self.intercept, self.coef, self.mean, self.std = intercept, coef, mean, std
            self.n0, self.n1, self.iterations, self.converged, self.max_score = n0, n1, iterations, converged, max_score

        def log_odds(self, obs):                                            # classifier.py:38-40
            z = (np.asarray(obs, np.float64) - self.mean) / self.std
            return self.intercept + z @ self.coef

        def log_weight(self, obs):                                          # classifier.py:42-43
            return self.log_odds(obs) + math.log(self.n0 / self.n1)

        def weight(self, obs):                                              # classifier.py:45-46
            return np.exp(self.log_weight(obs))

        @property
        def slope_on_c(self):                                               # classifier.py:48-51
            return float(self.coef[CURRENT] / self.std[CURRENT])

        @property
        def raw_coef(self):                                                 # classifier.py:53-56
            return self.coef / self.std

    def _loglik(X, y, beta):                                                # classifier.py:64-66
        eta = X @ beta
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    def fit_classifier(obs0, obs1, max_iter=100, tol=1e-10):               # classifier.py:69-101
        x0, x1 = np.asarray(obs0, np.float64), np.asarray(obs1, np.float64)
        X = np.concatenate([x0, x1])
        mean = X.mean(0)
        std = X.std(0)
        std = np.where(std > 0, std, 1.0)
        Z = np.concatenate([np.ones((len(X), 1)), (X - mean) / std], 1)
        y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))])
        beta = np.zeros(Z.shape[1])
        beta[0] = math.log(len(x1) / len(x0))
        ll = _loglik(Z, y, beta)
        converged, it = False, 0
        for it in range(1, int(max_iter) + 1):
            p = 1.0 / (1.0 + np.exp(-(Z @ beta)))
            g = Z.T @ (y - p)
            H = (Z * (p * (1 - p))[:, None]).T @ Z
            step = np.linalg.solve(H, g)
            t = 1.0
            while True:
                new = beta + t * step
                ll_new = _loglik(Z, y, new)
                if ll_new >= ll - 1e-12 * abs(ll) or t < 1e-8:
                    break
                t *= 0.5
            beta, ll = new, ll_new
            if np.max(np.abs(t * step)) < tol:
                converged = True
                break
        p = 1.0 / (1.0 + np.exp(-(Z @ beta)))
        score = float(np.max(np.abs(Z.T @ (y - p))))
        return _Classifier(float(beta[0]), beta[1:], mean, std, len(x0), len(x1), it, converged, score)


# ------------------------------------------------------------------------------------------ the known shift (world.py)

def normaliser(lam):
    """M(lam) = integral of exp(lam c) over [0.1, 1.1]; M(0) = 1 (world.py:95-100)."""
    return (C_HI - C_LO) if lam == 0.0 else math.exp(lam * C_LO) * math.expm1(lam * (C_HI - C_LO)) / lam


def ratio(c, lam):
    """Exact w(c) = p_test / p_train = (M(-2) / M(lam)) exp((lam + 2) c) (world.py:128-135)."""
    c = np.asarray(c, np.float64)
    if lam == LAMBDA_TRAIN:
        return np.ones_like(c)
    return (normaliser(LAMBDA_TRAIN) / normaliser(lam)) * np.exp((lam - LAMBDA_TRAIN) * c)


def full_test_mass(lam):
    """E_test[w] = E_train[w^2] = M(-2) M(2 lam + 2) / M(lam)^2 (world.py:150-154)."""
    return normaliser(LAMBDA_TRAIN) * normaliser(2 * lam - LAMBDA_TRAIN) / normaliser(lam) ** 2


def inverse_cdf(u, lam):
    """The law's quantile function (world.py:117-125)."""
    u = np.asarray(u, np.float64)
    if lam == 0.0:
        return C_LO + (C_HI - C_LO) * u
    return np.clip(C_LO + np.log1p(u * math.expm1(lam * (C_HI - C_LO))) / lam, C_LO, C_HI)


def calibrate(scores, stream, weights=None, t=None):
    return wbcp.calibrate(np.asarray(scores, np.float64), np.random.default_rng(stream), weights, t,
                          alpha=ALPHA, beta=BETA, draws=DRAWS)


def identical(a, b):
    return bool(a.threshold == b.threshold and a.lambda_hat == b.lambda_hat and a.lambda_hpd == b.lambda_hpd
                and np.array_equal(a.crossings, b.crossings))


def verdict(ok):
    return "PASS" if ok else "FAIL"


# ------------------------------------------------------------------------------------------ synthetic observations

def observations(rng, n_episodes, length, lam):
    """Synthetic 13-number observations with the structure of ThrusterChain's: the current c (index 10) is drawn once
    per episode from the law exp(lam c) on [0.1, 1.1]; the ten float numbers and the two wave numbers vary step by step
    and depend on c the same way under every law, so the only difference between laws is how often each c occurs, as
    in the weighting test (world.py's docstring). Returns (observations (n_episodes*length, 13), c per row)."""
    c_ep = inverse_cdf(rng.random(n_episodes), lam)
    c = np.repeat(c_ep, length)
    n = n_episodes * length
    floats = 0.5 * rng.standard_normal((n, 10)) + np.outer(c, np.linspace(-0.4, 0.4, 10))   # depend on c
    amp = np.repeat(rng.uniform(0.1, 0.4, n_episodes), length)
    phase = rng.uniform(0.0, 2 * math.pi, n)
    obs = np.concatenate([floats, c[:, None], (amp * np.cos(phase))[:, None], (amp * np.sin(phase))[:, None]], 1)
    return obs.astype(np.float32), c


# ------------------------------------------------------------------------------------------ checks

def s1_constants():
    spec = dict(M_minus2=0.35396379735782396, M_2=3.901805370636976, const_mild=0.35396379735782396,
                const_strong=0.09071795328941248, range_mild=(0.43, 3.19), range_strong=(0.135, 7.39),
                mass=(1.0, 1.381097845541816, 2.8414637940557546), kish=(1000, 724, 352))     # spec.json laws.*
    got = dict(M_minus2=normaliser(-2.0), M_2=normaliser(2.0),
               const_mild=normaliser(-2.0) / normaliser(0.0), const_strong=normaliser(-2.0) / normaliser(2.0),
               range_mild=(float(ratio(C_LO, 0.0)), float(ratio(C_HI, 0.0))),
               range_strong=(float(ratio(C_LO, 2.0)), float(ratio(C_HI, 2.0))),
               mass=tuple(full_test_mass(TILT[t]) for t in ("none", "mild", "strong")),
               kish=tuple(1000.0 / full_test_mass(TILT[t]) for t in ("none", "mild", "strong")))
    ok = (abs(got["M_minus2"] - spec["M_minus2"]) < 1e-12 and abs(got["M_2"] - spec["M_2"]) < 1e-12
          and abs(got["const_mild"] - spec["const_mild"]) < 1e-12 and abs(got["const_strong"] - spec["const_strong"]) < 1e-12
          and all(abs(a - b) < 0.005 for a, b in zip(got["range_mild"], spec["range_mild"]))
          and all(abs(a - b) < 0.005 for a, b in zip(got["range_strong"], spec["range_strong"]))
          and all(abs(a - b) < 1e-9 for a, b in zip(got["mass"], spec["mass"]))
          and all(abs(a - b) < 0.5 for a, b in zip(got["kish"], spec["kish"])))
    print("S1  the known shift's constants from world.py's formulas against experiments/iw/spec.json (laws):")
    print(f"    M(-2) = {got['M_minus2']:.6f}, M(2) = {got['M_2']:.6f}; weight = {got['const_mild']:.5f} e^(2c) (mild), "
          f"{got['const_strong']:.6f} e^(4c) (strong)")
    print(f"    weight range mild {got['range_mild'][0]:.2f}-{got['range_mild'][1]:.2f}, strong "
          f"{got['range_strong'][0]:.3f}-{got['range_strong'][1]:.2f}; test mass E_test[w] = "
          + ", ".join(f"{m:.4f}" for m in got["mass"]) + "; Kish rows of 1,000 = "
          + ", ".join(f"{k:.0f}" for k in got["kish"]) + " (none, mild, strong)")
    print("   ", verdict(ok))
    return ok, got


def s2_weights_of_one(fast):
    rng = np.random.default_rng([SEED, 2])
    sizes = [30, 45, 100, 300, 843, 3000] * (2 if fast else 7)
    bad = 0
    for i, n in enumerate(sizes):
        scores = np.abs(rng.standard_normal(n)) * rng.uniform(0.5, 2.0)
        a = calibrate(scores, [SEED, 2, i])
        b = calibrate(scores, [SEED, 2, i], np.ones(n), 1.0)
        bad += not identical(a, b)
    ok = bad == 0
    print(f"S2  weights of 1 and test mass 1 reproduce the unweighted threshold, credible quantile, plain quantile and "
          f"all 1,000 crossings bit for bit: {len(sizes) - bad} of {len(sizes)} banks (sizes 30 to 3,000)")
    print("   ", verdict(ok))
    return ok


def s3_common_factor(fast):
    rng = np.random.default_rng([SEED, 3])
    reps = 4 if fast else 12
    bad = 0
    for i in range(reps):
        n = int(rng.integers(40, 1200))
        scores = np.abs(rng.standard_normal(n))
        w = rng.uniform(0.05, 8.0, n)
        t = float(rng.uniform(0.5, 4.0))
        a = calibrate(scores, [SEED, 3, i], w, t)
        for f in (1e-3, 37.0, 1e3):
            bad += not identical(a, calibrate(scores, [SEED, 3, i], f * w, f * t))
    ok = bad == 0
    print(f"S3  a common factor (0.001, 37, 1000) on the weights and the test mass changes nothing: "
          f"{3 * reps - bad} of {3 * reps} cases")
    print("   ", verdict(ok))
    return ok


def s4_slope(fast):
    rng = np.random.default_rng([SEED, 4])
    reps_iid, reps_ep = (3, 6) if fast else (6, 20)
    out = {}
    print(f"S4  the classifier's fitted slope of log w on the current (exact 0, 2, 4), using {CLASSIFIER_SOURCE}")
    print("    30,000 rows a side in every shape; bars: independent rows 0.05, 300 episodes 0.35, 3,000 episodes 0.15")
    ok = True
    for name in ("none", "mild", "strong"):
        lam = TILT[name]
        exact = lam - LAMBDA_TRAIN
        iid, ep, many = [], [], []
        for _ in range(reps_iid):             # independent rows: 30,000 a side, every row its own current
            o0, _ = observations(rng, 30_000, 1, LAMBDA_TRAIN)
            o1, _ = observations(rng, 30_000, 1, lam)
            iid.append(fit_classifier(o0, o1).slope_on_c)
        for _ in range(reps_ep):              # the weighting test's shape: 300 episodes x 100 steps a side
            o0, _ = observations(rng, 300, 100, LAMBDA_TRAIN)
            o1, _ = observations(rng, 300, 100, lam)
            ep.append(fit_classifier(o0, o1).slope_on_c)
        for _ in range(reps_ep):              # ten times as many distinct currents: 3,000 episodes x 10 steps a side
            o0, _ = observations(rng, 3_000, 10, LAMBDA_TRAIN)
            o1, _ = observations(rng, 3_000, 10, lam)
            many.append(fit_classifier(o0, o1).slope_on_c)
        iid, ep, many = np.array(iid), np.array(ep), np.array(many)
        out[name] = dict(exact=exact, iid=iid.tolist(), episodes=ep.tolist(), many_episodes=many.tolist())
        good = (abs(iid.mean() - exact) <= 0.05 and abs(ep.mean() - exact) <= 0.35 and abs(many.mean() - exact) <= 0.15
                and many.std(ddof=1) < ep.std(ddof=1))
        ok &= good
        print(f"    {name:6s} exact {exact:.0f}: independent rows {iid.mean():+.3f} +- {iid.std(ddof=1):.3f} "
              f"({reps_iid} fits); 300 episodes a side {ep.mean():+.3f} +- {ep.std(ddof=1):.3f} ({reps_ep} fits, "
              f"range {ep.min():+.2f} to {ep.max():+.2f}, bias {ep.mean() - exact:+.2f}); 3,000 episodes a side "
              f"{many.mean():+.3f} +- {many.std(ddof=1):.3f} (bias {many.mean() - exact:+.2f})")
    print("    (the weighting test's record, 300 episodes a side, 5 seeds x 5 qualities: 0.04-0.16 for 0, 2.19-2.25 for"
          " 2, 4.21-4.28 for 4 by quality)")
    print("   ", verdict(ok))
    return ok, out


def s5_refusal():
    rng = np.random.default_rng([SEED, 5])
    n = 20_000
    actions = rng.uniform(-1.0, 1.0, (n, 4)).astype(np.float32)
    centre, rho = np.array([-0.76, -0.14, -0.54, -0.38]), 0.27        # a ball like the main test's (data.json gap)
    inside = np.linalg.norm(actions.astype(np.float64) - centre, axis=1) <= rho
    c = inverse_cdf(rng.random(n), LAMBDA_TRAIN)
    Z = 0.85
    exact_ratio = np.where(inside, np.inf, Z * ratio(c, 2.0))             # world.py:160-164 (exact_ratio)
    refused_w = inside.copy()                                             # coverage.py:251-259 (refusals, weighted)
    refused_u = np.zeros(n, bool)                                         # coverage.py:251-259 (unweighted)
    ok = (np.array_equal(np.isinf(exact_ratio), inside) and refused_w[inside].all() and not refused_w[~inside].any()
          and not refused_u.any() and np.all(np.isfinite(exact_ratio[~inside])) and inside.any())
    print(f"S5  refusal in the ball: {inside.sum()} of {n} synthetic commands inside the ball; the exact ratio is "
          f"+infinity on exactly those ({np.array_equal(np.isinf(exact_ratio), inside)}); a weighted arm refuses "
          f"{refused_w[inside].mean():.0%} inside and {refused_w[~inside].mean():.0%} outside; the unweighted arm "
          f"refuses {refused_u.mean():.0%}")
    print("   ", verdict(ok))
    return ok


def s6_purpose(fast):
    """Scores whose size grows with the current (as the record found for TD3+BC's critic: mark M2): held-out rows from
    the training law, test rows from the strong law."""
    rng = np.random.default_rng([SEED, 6])
    reps = 5 if fast else 20
    n_bank, n_test = 843, 20_000
    rows = []
    for r in range(reps):
        c_bank = inverse_cdf(rng.random(n_bank), LAMBDA_TRAIN)
        scores = np.abs(rng.standard_normal(n_bank)) * (0.5 + c_bank)
        c_test = inverse_cdf(rng.random(n_test), 2.0)
        test = np.abs(rng.standard_normal(n_test)) * (0.5 + c_test)
        c_same = inverse_cdf(rng.random(n_test), LAMBDA_TRAIN)
        same = np.abs(rng.standard_normal(n_test)) * (0.5 + c_same)
        stream = [SEED, 6, r]
        cu = calibrate(scores, stream)
        c1 = calibrate(scores, stream, np.ones(n_bank), 1.0)
        w = ratio(c_bank, 2.0)
        cw = calibrate(scores, stream, w, full_test_mass(2.0))
        o0, _ = observations(rng, 300, 100, LAMBDA_TRAIN)
        o1, _ = observations(rng, 300, 100, 2.0)
        clf = fit_classifier(o0, o1)
        bank_obs, _ = observations(rng, n_bank, 1, LAMBDA_TRAIN)
        bank_obs[:, CURRENT] = c_bank                                      # the bank rows' own currents
        w_hat = clf.weight(bank_obs)
        ce = calibrate(scores, stream, w_hat, float(clf.weight(o1).mean()))
        rows.append((np.mean(same <= cu.threshold), np.mean(test <= cu.threshold), np.mean(test <= cw.threshold),
                     np.mean(test <= ce.threshold),
                     np.mean(same <= cu.lambda_hat), np.mean(test <= cu.lambda_hat), np.mean(test <= cw.lambda_hat),
                     identical(cu, c1), cw.threshold / cu.threshold, ce.threshold / cu.threshold, clf.slope_on_c,
                     cw.lambda_hat / cu.lambda_hat, cu.n_eff, cw.n_eff, ce.n_eff))
    a = np.array(rows, dtype=float)
    cov_same, cov_u, cov_w, cov_e = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean()
    plain_same, plain_u, plain_w = a[:, 4].mean(), a[:, 5].mean(), a[:, 6].mean()
    ok = (0.89 <= cov_same <= 0.95 and cov_u < 0.88 and 0.88 <= plain_w <= 0.92 and 0.90 <= cov_w <= 0.97
          and abs(cov_e - cov_w) <= 0.02 and bool(np.all(a[:, 7] == 1)))
    print(f"S6  purpose ({reps} banks of {n_bank} rows, scores whose size grows with the current, 20,000 test rows each):")
    print(f"    full threshold R: coverage with no shift, unweighted {cov_same:.3f}; under the strong shift: unweighted "
          f"{cov_u:.3f}, exact-weighted {cov_w:.3f}, estimated-weighted {cov_e:.3f} (promise 0.90)")
    print(f"    plain 90% quantile alone (no Bayesian margin): no shift {plain_same:.3f}; strong shift: unweighted "
          f"{plain_u:.3f}, exact-weighted {plain_w:.3f} (the weights' own job: back to about 0.90)")
    print(f"    weights of 1 identical on {int(a[:, 7].sum())} of {reps} banks; R_w/R_u {a[:, 8].mean():.3f} "
          f"(range {a[:, 8].min():.3f}-{a[:, 8].max():.3f}), of which the plain quantile {a[:, 11].mean():.3f}; "
          f"R_est/R_u {a[:, 9].mean():.3f}; fitted slope {a[:, 10].mean():.2f} +- {a[:, 10].std(ddof=1):.2f} (exact 4); "
          f"Kish effective rows: unweighted {a[:, 12].mean():.0f}, exact {a[:, 13].mean():.0f}, estimated "
          f"{a[:, 14].mean():.0f} of {n_bank}")
    print("   ", verdict(ok))
    return ok, dict(no_shift=cov_same, unweighted=cov_u, exact=cov_w, estimated=cov_e, plain_no_shift=plain_same,
                    plain_unweighted=plain_u, plain_exact=plain_w, ratio_exact=a[:, 8].tolist(),
                    ratio_estimated=a[:, 9].tolist(), kish_exact=float(a[:, 13].mean()))


# ------------------------------------------------------------------------------------------ the real-data table

def real_table(path):
    if not Path(path).exists():
        print(f"\nreal data: {path} not found (run M4_pinned.py through the bridge first); table skipped")
        return None
    d = json.load(open(path))
    env, bands = d["environment"], d["bands"]
    print(f"\nREAL DATA: the ten near-80 version-6 bands (setting near80), recorded held-out scores and the recorded "
          f"rows' currents; pinned environment Python {env['python']}, numpy {env['numpy']}, jax {env['jax']}; "
          f"recorded current law uniform on {env['recorded_current_law']}")
    hdr = ("world  seed  rows  scores  rows   R_u      R_u=rec  w=1   x1000  | exact 2: R_w/R_u  Kish | exact 4: "
           "R_w/R_u  Kish  ends          | fitted slope 0 / 2 / 4         | est. R/R_u 0 / 4     | refused in/out")
    print(hdr)
    for b in bands:
        e2, e4, s = b["exact"]["2.0"], b["exact"]["4.0"], b["estimated"]
        print(f"{b['world']:8s} {b['seed']}   {b['rows']}   {str(b['scores_sha256_equal'])[0]}      "
              f"{str(b['bank_rows_sha256_equal'])[0]}     {b['unweighted']['R']:.4f}  "
              f"{str(b['unweighted']['equals_recorded_float32'])[0]}        {str(b['weights_of_1_identical'])[0]}     "
              f"{str(b['common_factor_identical'])[0]}      |          {e2['R_ratio']:.3f}   {e2['kish_fraction']:.3f} | "
              f"         {e4['R_ratio']:.3f}   {e4['kish_fraction']:.3f} {e4['per_step_R_at_smallest_mass']:.2f}-"
              f"{e4['per_step_R_at_largest_mass']:.2f}    | {s['0.0']['fitted_slope_on_c']:+.2f} / "
              f"{s['2.0']['fitted_slope_on_c']:.2f} / {s['4.0']['fitted_slope_on_c']:.2f}           | "
              f"{s['0.0']['R_ratio']:.3f} / {s['4.0']['R_ratio']:.3f}        | "
              f"{b['refusal']['weighted_refused_inside']:.2f} / {b['refusal']['weighted_refused_outside']:.2f}")
    n = len(bands)
    allb = lambda k: sum(bool(b[k]) for b in bands)
    r2 = np.array([b["exact"]["2.0"]["R_ratio"] for b in bands])
    r4 = np.array([b["exact"]["4.0"]["R_ratio"] for b in bands])
    k2 = np.array([b["exact"]["2.0"]["kish_fraction"] for b in bands])
    k4 = np.array([b["exact"]["4.0"]["kish_fraction"] for b in bands])
    f0 = np.array([b["estimated"]["0.0"]["fitted_slope_on_c"] for b in bands])
    f2 = np.array([b["estimated"]["2.0"]["fitted_slope_on_c"] for b in bands])
    f4 = np.array([b["estimated"]["4.0"]["fitted_slope_on_c"] for b in bands])
    e0 = np.array([b["estimated"]["0.0"]["R_ratio"] for b in bands])
    e4 = np.array([b["estimated"]["4.0"]["R_over_exact"] for b in bands])
    rec_eq = sum(b["unweighted"]["equals_recorded_float32"] for b in bands)
    print(f"\nsummary over {n} bands: scores file matches its record {allb('scores_sha256_equal')}/{n}; the kept held-out "
          f"rows match the record {allb('bank_rows_sha256_equal')}/{n}; the recorded threshold reproduced {rec_eq}/{n}; "
          f"weights of 1 identical {allb('weights_of_1_identical')}/{n}; common factor identical "
          f"{allb('common_factor_identical')}/{n}")
    print(f"  exact slope 2: R_w/R_u {r2.mean():.3f} (range {r2.min():.3f}-{r2.max():.3f}), Kish fraction {k2.mean():.3f} "
          f"(formula {bands[0]['exact']['2.0']['kish_fraction_formula']:.3f}); exact slope 4: R_w/R_u {r4.mean():.3f} "
          f"(range {r4.min():.3f}-{r4.max():.3f}), Kish fraction {k4.mean():.3f} (formula "
          f"{bands[0]['exact']['4.0']['kish_fraction_formula']:.3f})")
    print(f"  fitted slopes on recorded observations: {f0.mean():+.3f} +- {f0.std(ddof=1):.3f} (exact 0), "
          f"{f2.mean():.3f} +- {f2.std(ddof=1):.3f} (exact 2), {f4.mean():.3f} +- {f4.std(ddof=1):.3f} (exact 4)")
    print(f"  estimated weights: no-shift R_est/R_u {e0.mean():.3f} (range {e0.min():.3f}-{e0.max():.3f}); slope-4 "
          f"R_est/R_exact {e4.mean():.3f} (range {e4.min():.3f}-{e4.max():.3f})")
    print(f"  refusal on the held-out evaluation split: inside the ball "
          f"{np.mean([b['refusal']['weighted_refused_inside'] for b in bands]):.2f}, outside "
          f"{np.mean([b['refusal']['weighted_refused_outside'] for b in bands]):.2f}, unweighted "
          f"{np.mean([b['refusal']['unweighted_refused'] for b in bands]):.2f}; in-ball share "
          f"{np.mean([b['refusal']['in_ball_share'] for b in bands]):.3f}")
    return d


def recorded_slopes():
    """The weighting test's fitted slopes per seed and quality (outputs/shift_iw/v5/<quality>/report.json, marks.W4),
    from the worktree when it has them, else from branch origin/main (git show)."""
    import subprocess
    out = {"none": [], "mild": [], "strong": []}
    for q in ("random", "medium-replay", "medium", "medium-expert", "expert"):
        rel = f"outputs/shift_iw/v5/{q}/report.json"
        p = ROOT / rel
        try:
            if p.exists():
                text = p.read_text()
            else:
                text = subprocess.run(["git", "-C", str(ROOT), "show", f"origin/main:{rel}"], check=True,
                                      capture_output=True, text=True).stdout
        except Exception:
            continue
        s = json.loads(text)["marks"]["W4"]["classifier_slope_on_c"]
        for t in out:
            out[t] += s[t]["fitted"]["per_seed"]
    return out


# ------------------------------------------------------------------------------------------ the figure

def figure(path, s4, s6, real):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"  # keep the figure's text as text, not outlines
    import matplotlib.ticker
    # Okabe-Ito colorblind-safe palette
    BLUE, ORANGE, GREEN, PURPLE, BLACK, SKY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#000000", "#56B4E9"
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    fig.suptitle("Module M4, the weights: what each held-out row counts, and the checks on it", fontsize=12)

    # A: the weight curves of the known shift
    a = ax[0, 0]
    c = np.linspace(C_LO, C_HI, 300)
    for name, col in (("none", BLACK), ("mild", BLUE), ("strong", ORANGE)):
        k = 1000.0 / full_test_mass(TILT[name])
        a.plot(c, ratio(c, TILT[name]), color=col, lw=2,
               label=f"{name} shift: slope {TILT[name] - LAMBDA_TRAIN:.0f}, {k:.0f} effective rows of 1,000")
    a.set_yscale("log")
    a.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    a.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("water current c of the held-out row")
    a.set_ylabel("weight w(c): how often the current occurs at test time\ndivided by how often it occurred in training")
    a.set_title("A. Exact weights of the weighting test (world.py:128-135)", fontsize=10)
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3)

    # B: slope recovery
    b = ax[0, 1]
    rec = recorded_slopes()
    for i, name in enumerate(("none", "mild", "strong")):
        exact = TILT[name] - LAMBDA_TRAIN
        ys = rec[name]
        if ys:
            b.scatter(np.full(len(ys), exact - 0.25) + np.random.default_rng(1).uniform(-0.08, 0.08, len(ys)), ys,
                      s=18, color=BLUE, alpha=0.7, label="weighting test record (25 seed-qualities)" if i == 0 else None)
        ys = s4[name]["episodes"]
        b.scatter(np.full(len(ys), exact) + np.random.default_rng(2).uniform(-0.08, 0.08, len(ys)), ys, s=18,
                  color=ORANGE, alpha=0.7, marker="s", label="synthetic, 300 episodes a side" if i == 0 else None)
    if real:
        for i, (lam, exact) in enumerate((("0.0", 0.0), ("2.0", 2.0), ("4.0", 4.0))):
            ys = [bd["estimated"][lam]["fitted_slope_on_c"] for bd in real["bands"]]
            b.scatter(np.full(len(ys), exact + 0.25) + np.random.default_rng(3).uniform(-0.08, 0.08, len(ys)), ys,
                      s=22, color=GREEN, marker="^", label="recorded observations, known resampling tilt (10 bands)"
                      if i == 0 else None)
    b.plot([-0.6, 4.6], [-0.6, 4.6], color=BLACK, lw=1, ls="--", label="fitted = exact")
    b.set_xlabel("exact slope of log w in the current (0 = no shift, 2 = mild, 4 = strong)")
    b.set_ylabel("slope the estimated-weight classifier fitted")
    b.set_title("B. Estimated weights recover the known slope (classifier.py:69-101)", fontsize=10)
    b.set_xticks([0, 2, 4])
    b.legend(fontsize=7.5, loc="upper left")
    b.grid(alpha=0.3)

    # C: threshold ratios on the recorded scores
    cx = ax[1, 0]
    if real:
        bands = real["bands"]
        x = np.arange(len(bands))
        labels = [f"{bd['world'][:2]}{bd['seed']}" for bd in bands]
        ones = [1.0 if bd["weights_of_1_identical"] else np.nan for bd in bands]
        cx.bar(x - 0.3, [bd["exact"]["2.0"]["R_ratio"] for bd in bands], 0.2, color=BLUE, label="exact weights, slope 2")
        cx.bar(x - 0.1, [bd["exact"]["4.0"]["R_ratio"] for bd in bands], 0.2, color=ORANGE, label="exact weights, slope 4")
        cx.bar(x + 0.1, [bd["estimated"]["4.0"]["R_ratio"] for bd in bands], 0.2, color=GREEN,
               label="estimated weights, slope 4")
        cx.bar(x + 0.3, [bd["estimated"]["0.0"]["R_ratio"] for bd in bands], 0.2, color=PURPLE,
               label="estimated weights, no shift")
        cx.scatter(x, ones, color=BLACK, marker="_", s=120, lw=2, label="weights of 1 (= standard, exactly 1.000)",
                   zorder=5)
        cx.set_xticks(x)
        cx.set_xticklabels(labels, fontsize=8)
        cx.set_xlabel("band: st = stalling world, no = no-stall world; data seed")
        cx.set_ylim(0.9, max(1.6, max(bd["exact"]["4.0"]["R_ratio"] for bd in bands) * 1.08))
    else:
        cx.text(0.5, 0.5, "real-data JSON not found", ha="center", va="center", transform=cx.transAxes)
    cx.axhline(1.0, color=BLACK, lw=0.8)
    cx.set_ylabel("weighted threshold / unweighted threshold (R_w / R_u)")
    cx.set_title("C. Weights on the recorded held-out scores (near-80 bands, 821-823 rows)", fontsize=10)
    cx.legend(fontsize=7.5, loc="upper left", ncol=2)
    cx.grid(alpha=0.3, axis="y")

    # D: the purpose, synthetic
    dx = ax[1, 1]
    names = ["no shift,\nunweighted", "strong shift,\nunweighted", "strong shift,\nexact weights",
             "strong shift,\nestimated weights"]
    vals = [s6["no_shift"], s6["unweighted"], s6["exact"], s6["estimated"]]
    cols = [BLACK, SKY, ORANGE, GREEN]
    dx.bar(range(4), vals, color=cols, label="full threshold R (quantile plus Bayesian margin)")
    for i, v in enumerate(vals):
        dx.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plain = [s6["plain_no_shift"], s6["plain_unweighted"], s6["plain_exact"]]
    dx.scatter(range(3), plain, marker="D", s=55, facecolor="white", edgecolor=BLACK, lw=1.5, zorder=5,
               label="plain 90% quantile alone (the weights' own effect)")
    for i, v in enumerate(plain):
        dx.text(i + 0.1, v, f"{v:.3f}", ha="left", va="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    dx.axhline(0.90, color=BLACK, lw=1, ls="--", label="promised 90%")
    dx.set_xticks(range(4))
    dx.set_xticklabels(names, fontsize=8)
    dx.set_ylim(0.7, 1.0)
    dx.set_xlabel("condition (synthetic scores whose size grows with the current)")
    dx.set_ylabel("share of test rows whose score is below the threshold (coverage)")
    dx.set_title("D. Why weight: coverage under a known shift (synthetic, check S6)", fontsize=10)
    dx.legend(fontsize=7.5, loc="lower right")
    dx.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    print(f"\nfigure written: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(HERE.parent / "data" / "M4_real_data.json"))
    ap.add_argument("--figure", default=str(HERE.parent / "figures" / "M4.svg"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    print(f"numpy {np.__version__}; calibration/wbcp.py from {wbcp.__file__}; classifier: {CLASSIFIER_SOURCE}")
    ok1, _ = s1_constants()
    ok2 = s2_weights_of_one(args.fast)
    ok3 = s3_common_factor(args.fast)
    ok4, s4 = s4_slope(args.fast)
    ok5 = s5_refusal()
    ok6, s6 = s6_purpose(args.fast)
    results = [ok1, ok2, ok3, ok4, ok5, ok6]
    print("\nsynthetic checks:", verdict(all(results)), "(S1-S6:", " ".join(verdict(r) for r in results) + ")")
    real = real_table(args.real)
    if not args.no_figure:
        figure(args.figure, s4, s6, real)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
