"""Module M6, the band: synthetic checks where the truth is known, the real-data tables from recorded inputs, and the
figure docs/bca_mechanics/figures/M6.svg.

Command (run from the repository root; the same text is in docs/bca_mechanics/modules/M6.md):

    PYTHONPATH=. python docs/bca_mechanics/scripts/M6_test.py

Environment used for the record: Python 3.14, NumPy 2.5.4, SciPy 1.18.1, JAX 0.12.0, Matplotlib 3.11.2 on Windows
(the project's pinned environment is Python 3.10, NumPy 2.2.6, JAX 0.6.2; the synthetic checks do not need it, and the
one real-data computation that does, the fresh-row coverage, was run there by M6_fresh_rows.py and is read here from
its JSON). Options:
    --repo PATH        the repository root (default: the current directory); `git -C PATH show` reads the side branches
    --fresh PATH       the JSON written by M6_fresh_rows.py (default docs/bca_mechanics/data/M6_fresh_rows_coverage.json)
    --runfiles DIR     a folder holding runs/shift_explore/near80/learners/... (the near80-run-files branch checked out);
                       without it the 20 band records are read with `git show origin/claude/near80-run-files:<path>`
    --stage2 DIR       a copy of runs/shift_explore/stage2/detect/uniform/<world>/fixed/seed<k>/{band.json, bank_scores.npz,
                       tau.json} laid out as DIR/<world>/seed<k>/ (lives only in the owner's WSL clone; optional)
    --figure PATH      where to write the SVG (default docs/bca_mechanics/figures/M6.svg)
    --quick            fewer synthetic banks (for a smoke run)

What is checked. Module M6 is: U(s, a) = R * u_f * max(eta_f(s, a), 1e-6) in float32 (0 before the first refresh,
+infinity on refusal); a row is covered when q - U <= y <= q + U; coverage = covered rows / all rows, refused rows
counted against it. "Working as intended" = on fresh rows from the same law as the calibration rows the share covered
is about 0.90 (the 95%-credible margin makes it a little more), and the width is exactly the formula.

Synthetic (part A; nothing read from study data):
  A1  the width through calibration/dose.py's frozen_level_dose equals R * (max(eta, 1e-6) * u_f) bit for bit in
      float32; 0 everywhere before the first refresh; +infinity (and dose 1, support false) on refusal; and the other
      order of multiplication used by stage 2 and near-80, (R * u_f) * eta, differs by a few float32 steps at most
      (2 steps, 1.6e-7 relative, in this run; a rounding-order difference, not a different formula).
  A2  coverage of fresh rows at the study's bank size (843 rows): the share of banks whose realized coverage reaches
      0.90 with the full threshold R, against the plain 90% quantile alone.
  A3  coverage against bank size (20 to 3,000 rows), with the refusal boundary near 29 rows: the left panel of the figure.
  A4  the band holds on average over rows, not per row: with a flat error-size network on rows whose error spread
      differs by group, overall coverage is 0.90 while the groups get about 1.00 / 0.93 / 0.68.
  A5  a shift in the water current: unweighted coverage falls below 0.90 and exact importance weights restore it; rows
      from outside the recorded range of the current are not covered and have no finite weight (refusal).

Real data (part B; every number from a recorded file, named in the output):
  B1  the 20 near-80 version-6 bands: the share of recorded held-out scores at or below R (>= 0.90 by construction),
      the width's size, and U_max recomputed from the recorded eta.
  B2  the fresh-row coverage of those bands at the evaluation split of the log (M6_fresh_rows.py's JSON): recorded
      support, removed ball, hidden 70-90% band, past 80% throttle, and by current.
  B3  the weighting test (outputs/shift_iw/v5/<quality>/report.json): coverage per seed with no shift, under the strong
      and mild shifts unweighted, with exact weights, and inside the removed ball.
  B4  the four-learner study (branch claude/shift-hosts): coverage at recorded, actor and visited rows, standard vs IW.
  B5  the channel check (branch main, outputs/shift_iw/explore/channel): coverage at the actor's own commands.
  B6  stage 2's ten uniform bands (optional): bank coverage and the split-conformal check of the width.
"""
import argparse
import io
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from calibration import wbcp
    from calibration import dose as DOSE
    from calibration.reference import FrozenReference
except ImportError as error:
    sys.exit(f"set PYTHONPATH to the repository root (calibration/): {error}")

import jax
import jax.numpy as jnp

ALPHA, BETA, DRAWS = 0.1, 0.95, 1000
SEED = 20261020
QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
RUNFILES_BRANCH = "origin/claude/near80-run-files"
HOSTS_BRANCH = "origin/claude/shift-hosts"
MAIN_BRANCH = "origin/main"
# the weighting test's laws (experiments/shift_iw/world.py): current c uniform-tilted on [0.1, 1.1], density
# proportional to exp(lambda c); training lambda = -2; 'mild' 0; 'strong' +2. Exact weight w(c) = p_test / p_train.
C_LO, C_HI, LAMBDA_TRAIN = 0.1, 1.1, -2.0
LAWS = {"none": -2.0, "mild": 0.0, "strong": 2.0}


def verdict(ok):
    return "PASS" if ok else "FAIL"


def f32(x):
    return np.float32(x)


# ------------------------------------------------------------------------------------------------ helpers

def calibrate(scores, rng, weights=None, test_mass=None):
    return wbcp.calibrate(np.asarray(scores, np.float64), rng, weights, test_mass, alpha=ALPHA, beta=BETA, draws=DRAWS)


def covered_share(y, q, U):
    """coverage.outcomes + summarize: covered when q - U <= y <= q + U, a refused row (U not finite) never covered."""
    y, q, U = (np.asarray(v, np.float64) for v in (y, q, U))
    refused = ~np.isfinite(U)
    return float(np.mean((y >= q - U) & (y <= q + U) & ~refused))


def normaliser(lam):
    if lam == 0.0:
        return C_HI - C_LO
    return (math.exp(lam * C_HI) - math.exp(lam * C_LO)) / lam


def draw_current(rng, lam, n):
    """Draw c on [C_LO, C_HI] with density proportional to exp(lam c) by inverse transform."""
    u = rng.uniform(size=n)
    if lam == 0.0:
        return C_LO + u * (C_HI - C_LO)
    a, b = math.exp(lam * C_LO), math.exp(lam * C_HI)
    return np.log(a + u * (b - a)) / lam


def weight(c, law):
    lam = LAWS[law]
    if lam == LAMBDA_TRAIN:
        return np.ones_like(np.asarray(c, np.float64))
    return (normaliser(LAMBDA_TRAIN) / normaliser(lam)) * np.exp((lam - LAMBDA_TRAIN) * np.asarray(c, np.float64))


def test_mass(law):
    """E_test[w] = M(-2) M(2 lam + 2) / M(lam)^2 (world.full_test_mass): 1, 1.381, 2.841."""
    lam = LAWS[law]
    return normaliser(LAMBDA_TRAIN) * normaliser(2 * lam - LAMBDA_TRAIN) / normaliser(lam) ** 2


def git_show(repo, ref, path, binary=False):
    out = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode("utf-8")


# ------------------------------------------------------------------------------------------------ part A: synthetic

def A1_width_identity():
    print("A1  the width through calibration/dose.py (frozen_level_dose) against the formula, float32")
    rng = np.random.default_rng([SEED, 61])
    R, u = 1.4691808, 0.15669172
    eta = rng.uniform(0.3, 3.0, 20000).astype(np.float32)
    eta[:5] = np.float32([1e-9, 5e-7, 1e-6, 2e-6, 1.0])        # the floor 1e-6 must bind on the first two
    ref = FrozenReference({"w": jnp.zeros(3, jnp.float32)}, jnp.asarray(u, jnp.float32), jnp.asarray(R, jnp.float32),
                          jnp.asarray(1.36, jnp.float32), jnp.asarray(R, jnp.float32), jnp.asarray(843.0, jnp.float32),
                          jnp.asarray(True))
    out = DOSE.frozen_level_dose(ref, jnp.asarray(eta), 0.5)
    width = np.asarray(out.width)
    expected = f32(R) * (np.maximum(eta, f32(1e-6)) * f32(u))             # dose.py:75-76, this order
    other = (f32(R) * f32(u)) * np.maximum(eta, f32(1e-6))                 # stage-2 band.py:157, near-80: this order
    same = np.array_equal(width, expected) and width.dtype == np.float32
    rel = np.max(np.abs(expected.astype(np.float64) - other) / expected)
    ulp = np.max(np.abs(expected.astype(np.float64) - other) / np.spacing(expected))
    floor_ok = bool(np.all(width[:2] == f32(R) * (f32(1e-6) * f32(u))))
    # not ready: width 0, dose 1
    out0 = DOSE.frozen_level_dose(ref._replace(ready=jnp.asarray(False)), jnp.asarray(eta), 0.5)
    not_ready_ok = bool(np.all(np.asarray(out0.width) == 0) and np.all(np.asarray(out0.dose) == 1.0))
    # refusal: width +inf, support false, dose 1 (the standard rule)
    outinf = DOSE.frozen_level_dose(ref._replace(threshold=jnp.asarray(jnp.inf, jnp.float32),
                                                 lambda_hpd=jnp.asarray(jnp.inf, jnp.float32)), jnp.asarray(eta), 0.5)
    refusal_ok = bool(np.all(np.isposinf(np.asarray(outinf.width))) and not np.any(np.asarray(outinf.support_mask))
                      and np.all(np.asarray(outinf.dose) == 1.0) and bool(outinf.inputs_valid))
    print(f"    width == R * (max(eta, 1e-6) * u) bit for bit on 20,000 rows: {same}; floor 1e-6 binds: {floor_ok}")
    print(f"    the other order (R * u) * eta differs by at most {rel:.2e} relative = {ulp:.0f} float32 step(s)")
    print(f"    before the first refresh: width 0 and dose 1 everywhere: {not_ready_ok}")
    print(f"    refusal (R = +inf): width +inf, support false, dose 1, inputs valid: {refusal_ok}")
    ok = same and floor_ok and not_ready_ok and refusal_ok and ulp <= 4.0
    print("   ", verdict(ok))
    return ok, dict(order_rel=float(rel), order_ulp=float(ulp))


def one_bank_coverage(rng, n, n_fresh=20000, u=0.15, sigma=0.12, eta_lo=0.5, eta_hi=1.5):
    """One synthetic bank: residual = sigma * eta * N(0,1) with eta known; returns (R, lambda_hat, coverage with R,
    coverage with lambda_hat alone) on fresh rows from the same law."""
    eta = rng.uniform(eta_lo, eta_hi, n)
    resid = rng.standard_normal(n) * sigma * eta
    scores = np.abs(resid) / (np.maximum(eta, 1e-6) * u)
    cal = calibrate(scores, np.random.default_rng(rng.integers(2 ** 31)))
    eta_new = rng.uniform(eta_lo, eta_hi, n_fresh)
    fresh = rng.standard_normal(n_fresh) * sigma * eta_new
    U = cal.threshold * u * np.maximum(eta_new, 1e-6)
    U_hat = cal.lambda_hat * u * np.maximum(eta_new, 1e-6)
    return cal.threshold, cal.lambda_hat, covered_share(fresh, 0.0, U), covered_share(fresh, 0.0, U_hat)


def A2_fresh_rows_843(quick):
    banks = 60 if quick else 300
    rng = np.random.default_rng([SEED, 62])
    rows = [one_bank_coverage(rng, 843) for _ in range(banks)]
    cov = np.array([r[2] for r in rows])
    cov_hat = np.array([r[3] for r in rows])
    R = np.array([r[0] for r in rows])
    lam = np.array([r[1] for r in rows])
    print(f"A2  coverage of 20,000 fresh rows from the same law, {banks} synthetic banks of 843 rows")
    print(f"    with R = max(plain quantile, credible quantile): mean {cov.mean():.3f}, 5%-95% {np.quantile(cov, 0.05):.3f}"
          f"-{np.quantile(cov, 0.95):.3f}, banks at or above 0.90: {np.mean(cov >= 0.90):.1%}")
    print(f"    with the plain 90% quantile alone:             mean {cov_hat.mean():.3f}, 5%-95% "
          f"{np.quantile(cov_hat, 0.05):.3f}-{np.quantile(cov_hat, 0.95):.3f}, banks at or above 0.90: "
          f"{np.mean(cov_hat >= 0.90):.1%}")
    print(f"    R / plain quantile: mean {np.mean(R / lam):.3f} (the Bayesian margin at 843 rows)")
    ok = 0.90 <= cov.mean() <= 0.93 and np.mean(cov >= 0.90) >= 0.85
    print("   ", verdict(ok))
    return ok, dict(banks=banks, mean=float(cov.mean()), lo=float(np.quantile(cov, 0.05)),
                    hi=float(np.quantile(cov, 0.95)), share_at_least_090=float(np.mean(cov >= 0.90)),
                    plain_mean=float(cov_hat.mean()), plain_share=float(np.mean(cov_hat >= 0.90)),
                    margin=float(np.mean(R / lam)))


def A3_by_bank_size(quick):
    sizes = (20, 26, 28, 29, 30, 36, 50, 100, 300, 843, 3000)
    banks = 40 if quick else 200
    rng = np.random.default_rng([SEED, 63])
    table = []
    print(f"A3  coverage of fresh rows against the bank size, {banks} synthetic banks per size (a refused bank covers "
          f"0: refused rows count against coverage)")
    print("    rows   refused   mean   median   5%     95%    mean among certified")
    for n in sizes:
        rows = [one_bank_coverage(rng, n, n_fresh=5000) for _ in range(banks)]
        cov = np.array([r[2] for r in rows])
        refused = np.array([not math.isfinite(r[0]) for r in rows])
        cert = cov[~refused]
        table.append(dict(n=n, refused=float(refused.mean()), mean=float(cov.mean()), median=float(np.median(cov)),
                          lo=float(np.quantile(cov, 0.05)), hi=float(np.quantile(cov, 0.95)),
                          mean_certified=float(cert.mean()) if len(cert) else None))
        t = table[-1]
        mc = "-" if t["mean_certified"] is None else f"{t['mean_certified']:.3f}"
        print(f"    {n:5d}   {t['refused']:5.1%}   {t['mean']:.3f}  {t['median']:.3f}   {t['lo']:.3f}  {t['hi']:.3f}  {mc}")
    big = [t for t in table if t["n"] >= 100]
    means = [t["mean"] for t in table if t["n"] >= 36]
    # no refusal from 100 rows up; coverage between 0.89 and 0.97 there (the credible margin gives about 0.95 at 100
    # rows, 0.93 at 300, 0.92 at 843, 0.91 at 3,000) and falling toward 0.90 as the bank grows; every bank of 20 refuses
    ok = all(t["refused"] == 0 for t in big) and all(0.89 <= t["mean"] <= 0.97 for t in big) \
        and all(a >= b for a, b in zip(means, means[1:])) and table[0]["refused"] == 1.0
    print("   ", verdict(ok))
    return ok, table


def A4_marginal(quick):
    rng = np.random.default_rng([SEED, 64])
    n, n_fresh, u = 843, 30000, 0.15
    spreads = np.array([0.5, 1.0, 2.0])          # three groups of rows whose error spread differs
    print("A4  the band holds on average over rows: three groups with error spreads 0.5 / 1.0 / 2.0 (relative)")
    results = {}
    for name, eta_fn in (("eta right (tracks the spread)", lambda g: spreads[g]),
                         ("eta flat (1 everywhere)", lambda g: np.ones(len(g)))):
        g = rng.integers(3, size=n)
        eta = eta_fn(g)
        resid = rng.standard_normal(n) * 0.12 * spreads[g]
        scores = np.abs(resid) / (np.maximum(eta, 1e-6) * u)
        R = calibrate(scores, np.random.default_rng(rng.integers(2 ** 31))).threshold
        gf = rng.integers(3, size=n_fresh)
        fresh = rng.standard_normal(n_fresh) * 0.12 * spreads[gf]
        U = R * u * np.maximum(eta_fn(gf), 1e-6)
        overall = covered_share(fresh, 0.0, U)
        per = [covered_share(fresh[gf == k], 0.0, U[gf == k]) for k in range(3)]
        results[name] = dict(overall=overall, groups=per)
        print(f"    {name:32s}: overall {overall:.3f}; by group {per[0]:.3f} / {per[1]:.3f} / {per[2]:.3f}")
    right, flat = results["eta right (tracks the spread)"], results["eta flat (1 everywhere)"]
    ok = (0.88 <= right["overall"] <= 0.94 and 0.88 <= flat["overall"] <= 0.94
          and max(abs(c - 0.90) for c in right["groups"]) < 0.04 and flat["groups"][2] < 0.80
          and flat["groups"][0] > 0.97)
    print("   ", verdict(ok))
    return ok, results


def A5_shift(quick):
    rng = np.random.default_rng([SEED, 65])
    n, n_fresh, u = 843, 30000, 0.15
    # the critic's error grows with the current: spread 0.1 (1 + 2 c); eta is flat, so a shift toward high currents
    # meets a band calibrated mostly on low currents (the training law tilts toward low c)
    spread = lambda c: 0.1 * (1.0 + 2.0 * c)
    c_bank = draw_current(rng, LAMBDA_TRAIN, n)
    scores = np.abs(rng.standard_normal(n) * spread(c_bank)) / u
    stream = int(rng.integers(2 ** 31))
    R_u = calibrate(scores, np.random.default_rng(stream)).threshold
    print("A5  a shift in the current (the weighting test's laws): unweighted against exact-weighted coverage of fresh rows")
    print("    law      unweighted R   coverage   weighted R (single test mass)   coverage   test mass")
    out = {}
    for law in ("none", "mild", "strong"):
        c = draw_current(rng, LAWS[law], n_fresh)
        fresh = rng.standard_normal(n_fresh) * spread(c)
        cov_u = covered_share(fresh, 0.0, R_u * u)
        w = weight(c_bank, law)
        R_w = calibrate(scores, np.random.default_rng(stream), w, test_mass(law)).threshold
        cov_w = covered_share(fresh, 0.0, R_w * u)
        out[law] = dict(R_u=R_u, cov_u=cov_u, R_w=R_w, cov_w=cov_w, t=test_mass(law))
        print(f"    {law:7s}  {R_u:10.3f}     {cov_u:.3f}      {R_w:10.3f}                   {cov_w:.3f}      {test_mass(law):.3f}")
    # rows from outside the recorded range of the current: no calibration row there, the weight is undefined (+inf)
    c_out = rng.uniform(C_HI, C_HI + 0.5, n_fresh)
    fresh = rng.standard_normal(n_fresh) * spread(c_out)
    cov_out = covered_share(fresh, 0.0, R_u * u)
    out["outside_range"] = dict(cov_u=cov_out)
    print(f"    rows with a current outside the recorded range (1.1 to 1.6): unweighted coverage {cov_out:.3f}; the exact "
          f"weight is +infinity there, so a weighted arm refuses them (coverage.outcomes: refused = not covered)")
    ok = (out["none"]["cov_u"] >= 0.89 and out["none"]["R_w"] == out["none"]["R_u"]
          and out["strong"]["cov_u"] < 0.88 and out["strong"]["cov_w"] >= 0.89 and cov_out < out["strong"]["cov_u"])
    print("   ", verdict(ok))
    return ok, out


# ------------------------------------------------------------------------------------------------ part B: real data

def read_near80_bands(repo, runfiles):
    bands = []
    for setting in ("near80", "random_rows"):
        for world in ("stall", "no_stall"):
            for seed in range(5):
                rel = f"runs/shift_explore/near80/learners/{setting}/{world}/seed{seed}"
                if runfiles:
                    rec = json.loads((Path(runfiles) / rel / "v6_band.json").read_text())
                    z = np.load(Path(runfiles) / rel / "v6_bank_scores.npz")
                else:
                    rec = json.loads(git_show(repo, RUNFILES_BRANCH, rel + "/v6_band.json"))
                    z = np.load(io.BytesIO(git_show(repo, RUNFILES_BRANCH, rel + "/v6_bank_scores.npz", binary=True)))
                bands.append(dict(setting=setting, world=world, seed=seed, rec=rec, scores=z["scores"], eta=z["eta"]))
    return bands


def B1_near80_banks(bands):
    print("B1  the 20 near-80 version-6 bands (runs/shift_explore/near80/learners/<setting>/<world>/seed<k>/v6_band.json "
          "and v6_bank_scores.npz, branch claude/near80-run-files): held-out rows, R, the share of recorded scores at or "
          "below R and below the plain quantile, R / plain quantile, the width at the held-out rows, U_max recomputed")
    print("    setting      world     seed  rows   R       <=R     <=plain  R/plain  U median  U 95%   U_max rec / recomputed")
    rows = []
    for b in bands:
        u = b["rec"]["uniform"]
        R, lam, unit = u["R"], u["lambda_hat"], u["unit"]
        s, e = b["scores"], b["eta"]
        U = f32(R) * (np.maximum(e.astype(np.float32), f32(1e-6)) * f32(unit))
        umax = float(f32(R) * f32(unit) * f32(np.maximum(e, 1e-6).max()))
        row = dict(setting=b["setting"], world=b["world"], seed=b["seed"], rows=int(len(s)), R=R, lambda_hat=lam,
                   unit=unit, share_R=float(np.mean(s <= R)), share_plain=float(np.mean(s <= lam)), margin=R / lam,
                   U_median=float(np.median(U)), U_95=float(np.quantile(U, 0.95)), U_max_recorded=u["U_max"],
                   U_max_recomputed=umax, scores_equal_recorded=u["threshold_recomputed"]["equal"])
        rows.append(row)
        print(f"    {row['setting']:12s} {row['world']:9s} {row['seed']}     {row['rows']}   {R:.3f}   {row['share_R']:.3f}"
              f"   {row['share_plain']:.3f}    {row['margin']:.3f}    {row['U_median']:.3f}     {row['U_95']:.3f}"
              f"   {u['U_max']:.4f} / {umax:.4f}")
    ok = all(r["share_R"] >= 0.90 for r in rows) and all(abs(r["U_max_recorded"] - r["U_max_recomputed"]) <= 2e-6
                                                         for r in rows)
    sh = [r["share_R"] for r in rows]
    print(f"    share at or below R over the 20 bands: {min(sh):.3f} to {max(sh):.3f} (mean {statistics.mean(sh):.3f}); "
          f"R / plain quantile {min(r['margin'] for r in rows):.3f} to {max(r['margin'] for r in rows):.3f}; "
          f"recorded threshold_recomputed.equal true on {sum(r['scores_equal_recorded'] for r in rows)} of 20")
    print("   ", verdict(ok))
    return ok, rows


def B2_fresh_rows(fresh_path):
    p = Path(fresh_path)
    if not p.is_file():
        print(f"B2  fresh-row coverage: {p} not found (run M6_fresh_rows.py in the pinned environment); skipped")
        return None, None
    d = json.loads(p.read_text())
    res = [r for r in d["results"] if not r.get("failed")]
    failed = [r for r in d["results"] if r.get("failed")]
    print(f"B2  coverage of the saved bands at rows they never saw ({p.name}; pinned environment Python {d['python']}, "
          f"NumPy {d['numpy']}, JAX {d['jax']}); {len(res)} seed-worlds, {len(failed)} failed")
    print("    setting      world     seed  scores=rec  bank    fresh:all  support  ball    band    past80  | current low/mid/high (support) | train kept  hidden")
    for r in res:
        ev = r["evaluation"]
        band = ev["band_not_ball"].get("coverage")
        print(f"    {r['setting']:12s} {r['world']:9s} {r['seed']}     {str(r['bank_scores_check']['equal']):5s}      "
              f"{r['bank_direct']['coverage']:.3f}   {ev['all']['coverage']:.3f}      {ev['support']['coverage']:.3f}    "
              f"{ev['ball']['coverage']:.3f}   {('%.3f' % band) if band is not None else '  -  '}   "
              f"{ev['past_80']['coverage']:.3f}   | {ev['support_current_low']['coverage']:.3f} / "
              f"{ev['support_current_mid']['coverage']:.3f} / {ev['support_current_high']['coverage']:.3f}          | "
              f"{r['train_kept']['coverage']:.3f}       {r['train_hidden'].get('coverage', float('nan')):.3f}")

    def agg(key, sub=None, only=None, world=None):
        vals = []
        for r in res:
            if only and r["setting"] != only:
                continue
            if world and r["world"] != world:
                continue
            v = r["evaluation"][key] if sub is None else r[key]
            if v.get("rows", 0) and v.get("coverage") is not None:
                vals.append(v["coverage"])
        return vals

    summary = {}
    for label, key, sub, only, world in (
            ("fresh rows, all", "all", None, None, None),
            ("fresh rows, recorded support", "support", None, None, None),
            ("fresh rows, removed ball", "ball", None, None, None),
            ("fresh rows, removed ball, stalling world", "ball", None, None, "stall"),
            ("fresh rows, removed ball, no-stall world", "ball", None, None, "no_stall"),
            ("fresh rows, hidden 70-90% band (near-80 setting)", "band_not_ball", None, "near80", None),
            ("fresh rows, hidden 70-90% band, stalling world", "band_not_ball", None, "near80", "stall"),
            ("fresh rows, hidden 70-90% band, no-stall world", "band_not_ball", None, "near80", "no_stall"),
            ("fresh rows, past 80% throttle", "past_80", None, None, None),
            ("fresh rows, past 80% throttle, stalling world", "past_80", None, None, "stall"),
            ("fresh rows, past 80% throttle, no-stall world", "past_80", None, None, "no_stall"),
            ("fresh support rows, low current", "support_current_low", None, None, None),
            ("fresh support rows, middle current", "support_current_mid", None, None, None),
            ("fresh support rows, high current", "support_current_high", None, None, None),
            ("training rows kept (in-sample for the critic)", "train_kept", True, None, None),
            ("training rows hidden by the removal", "train_hidden", True, None, None),
            ("training rows hidden, stalling world", "train_hidden", True, None, "stall"),
            ("training rows hidden, no-stall world", "train_hidden", True, None, "no_stall"),
            ("held-out calibration rows (in-sample for R)", "bank_direct", True, None, None)):
        vals = agg(key, sub, only, world)
        if vals:
            summary[label] = dict(n=len(vals), mean=statistics.mean(vals), lo=min(vals), hi=max(vals), values=vals)
    print("    summary over seed-worlds (mean; smallest to largest):")
    for k, v in summary.items():
        print(f"      {k:52s} {v['mean']:.3f}  ({v['lo']:.3f} to {v['hi']:.3f}; n = {v['n']})")
    ev = lambda r, k: r["evaluation"][k]["coverage"]
    stall = [r for r in res if r["world"] == "stall"]
    nost = [r for r in res if r["world"] == "no_stall"]
    print("    consistent or exceptional, counted over seed-worlds:")
    print(f"      fresh rows of the recorded support at or above 0.89: {sum(ev(r, 'support') >= 0.89 for r in res)} of {len(res)}")
    print(f"      high-current third of the support covered less than the low-current third: "
          f"{sum(ev(r, 'support_current_high') < ev(r, 'support_current_low') for r in res)} of {len(res)}")
    print(f"      removed ball below 0.60: {sum(ev(r, 'ball') < 0.60 for r in stall)} of {len(stall)} stalling-world bands; "
          f"at or above 0.85: {sum(ev(r, 'ball') >= 0.85 for r in nost)} of {len(nost)} no-stall bands")
    print(f"      past 80% throttle below 0.55: {sum(ev(r, 'past_80') < 0.55 for r in stall)} of {len(stall)} stalling-world "
          f"bands; at or above 0.85: {sum(ev(r, 'past_80') >= 0.85 for r in nost)} of {len(nost)} no-stall bands")
    print(f"      hidden 70-90% band below 0.85: {sum(ev(r, 'band_not_ball') < 0.85 for r in res if r['setting'] == 'near80')}"
          f" of {sum(r['setting'] == 'near80' for r in res)} near-80 bands")
    print(f"      kept training rows covered more than the held-out rows: "
          f"{sum(r['train_kept']['coverage'] > r['bank_direct']['coverage'] for r in res)} of {len(res)}")
    checks = [r["bank_scores_check"]["equal"] for r in res]
    order = max(r["width_order_check_train"]["max_rel_difference"] for r in res)
    print(f"    recorded held-out scores reproduced bit for bit: {sum(checks)} of {len(checks)}; the run's width "
          f"(stage2/code/band.py width_uniform, (R * u_f) * eta) against dose.py's order of multiplication "
          f"(R * (eta * u_f)): at most {order:.1e} relative, about {order / 1.19e-7:.0f} float32 steps")
    sup = summary.get("fresh rows, recorded support")
    ok = bool(sup and all(checks) and 0.88 <= sup["mean"] <= 0.93 and sup["lo"] >= 0.86)
    print("   ", verdict(ok), "(the promise holds on the recorded support; the removed regions are the caveat)")
    return ok, summary


def B3_weighting_test(repo):
    print("B3  the weighting test, outputs/shift_iw/v5/<quality>/report.json (TD3+BC co-trained with BCA; test steps "
          "outside the removed ball unless said; per seed 0-4)")
    print("    quality         no shift unw.  strong unw.   strong exact   mild unw.*  mild exact   inside ball: none / mild / strong (unw.)   est. none")
    table = {}
    for q in QUALITIES:
        d = json.loads((Path(repo) / "outputs/shift_iw/v5" / q / "report.json").read_text())
        mk, sh = d["marks"], d["shown"]
        w3 = mk["W3"]["tilts"]
        row = dict(none_unw=mk["M1"]["coverage"]["per_seed"], strong_unw=mk["W2"]["coverage"]["per_seed"],
                   strong_exact=mk["W1"]["coverage"]["strong"]["per_seed"], mild_exact=mk["W1"]["coverage"]["mild"]["per_seed"],
                   mild_unw=w3["mild"]["unweighted"]["outside"]["per_seed"],
                   inside_none=w3["none"]["unweighted"]["inside"]["per_seed"],
                   inside_mild=w3["mild"]["unweighted"]["inside"]["per_seed"],
                   inside_strong=w3["strong"]["unweighted"]["inside"]["per_seed"],
                   est_none=sh["single_threshold_coverage"]["none"]["estimated_single"]["per_seed"],
                   strong_exact_single=sh["single_threshold_coverage"]["strong"]["exact_single"]["per_seed"],
                   marks=dict(M1=mk["M1"]["status"], W1=mk["W1"]["status"], W2=mk["W2"]["status"], W3=mk["W3"]["status"]))
        table[q] = row
        m = lambda k: statistics.mean(row[k])
        print(f"    {q:14s}  {m('none_unw'):.3f}          {m('strong_unw'):.3f}         {m('strong_exact'):.3f}          "
              f"{m('mild_unw'):.3f}       {m('mild_exact'):.3f}        {m('inside_none'):.3f} / {m('inside_mild'):.3f} / "
              f"{m('inside_strong'):.3f}                  {m('est_none'):.3f}")
    print("    * the mild unweighted column is W3's 'outside' value (not a mark). Marks: M1 (no shift in [0.89, 0.94], "
          "every seed >= 0.86), W1 (weighted in [0.89, 0.96]), W2 (strong unweighted below 0.88), W3 (inside the "
          "removed ball the unweighted band covers less than outside it, judged per tilt).")
    print("    marks per quality (report.json marks.<mark>.status):")
    for q, row in table.items():
        print(f"      {q:14s} " + "  ".join(f"{k} {v}" for k, v in row["marks"].items()))
    core = all(row["marks"][k] == "pass" for row in table.values() for k in ("M1", "W1", "W2"))
    w3_fail = [q for q, row in table.items() if row["marks"]["W3"] != "pass"]
    print(f"    M1, W1, W2 pass on all five qualities: {core}; W3 fails on: {', '.join(w3_fail) if w3_fail else 'none'}"
          " (there the band covered the removed ball as well as the rest: the inside-ball drop is not universal)")
    print("   ", verdict(core), "(judged on M1, W1, W2; W3 is reported)")
    return core, table


def B4_four_learners(repo):
    try:
        d = json.loads(git_show(repo, HOSTS_BRANCH, "outputs/shift_hosts_v3/results.json"))
    except subprocess.CalledProcessError as error:
        print(f"B4  four-learner study: cannot read {HOSTS_BRANCH} ({error}); skipped")
        return None, None
    print("B4  the four-learner study (branch claude/shift-hosts, outputs/shift_hosts_v3/results.json; exploratory, seeds "
          "0-4; under the strong shift): coverage at recorded rows, at the actor's own command, at visited states; "
          "standard band vs the importance-weighted band (IW, exact strong weights); mean over seeds (range)")
    table = {}
    for q in ("medium", "medium-expert"):
        for learner in ("td3_bc", "cql", "iql", "rebrac"):
            E = d["qualities"][q]["expanded"].get(learner)
            if not E:
                continue
            for setup in ("standard", "iw"):
                S = E["setups"].get(setup)
                cov = (S or {}).get("coverage") or []
                groups = {}
                for c in cov:
                    if not c or "other" not in c:
                        continue
                    for g, gv in c["other"]["groups"].items():
                        for arm in ("standard", "iw"):
                            if arm in gv:
                                groups.setdefault((g, arm), []).append(gv[arm]["all"]["certified_coverage"])
                if groups:
                    table[(q, learner, setup)] = groups
                    parts = []
                    for (g, arm), vals in sorted(groups.items()):
                        parts.append(f"{g} {arm}: {statistics.mean(vals):.3f} ({min(vals):.3f}-{max(vals):.3f})")
                    print(f"    {q:14s} {learner:7s} {setup:9s} | " + "; ".join(parts))
    print("    (TD3+BC's IW setup and IQL were stopped by stop rule S7 and are not judged; ReBRAC's critic did not learn.)")
    return True, {f"{k[0]}|{k[1]}|{k[2]}": {f"{g}|{a}": v for (g, a), v in gr.items()} for k, gr in table.items()}


def B5_channel(repo):
    print("B5  the channel check (branch main, outputs/shift_iw/explore/channel/results/<quality>.json; exploratory): "
          "coverage of the unweighted band at the recorded command and at the actor's own command (within and past 80% "
          "throttle), under the training law and the strong law; mean over seeds 0-4")
    print("    quality         @recorded train/strong   @actor all train/strong   within 80% train/strong   past 80% train/strong   share past 80%")
    table = {}
    for q in QUALITIES:
        try:
            d = json.loads(git_show(repo, MAIN_BRANCH, f"outputs/shift_iw/explore/channel/results/{q}.json"))
        except subprocess.CalledProcessError:
            print(f"    {q}: not readable; skipped")
            continue
        seeds = d["seeds"]
        g = lambda key, grp, law: statistics.mean(s[key][grp][f"coverage_{law}_law"] for s in seeds)
        row = dict(rec_train=g("unweighted@recorded", "all", "train"), rec_strong=g("unweighted@recorded", "all", "strong"),
                   act_train=g("unweighted@actor", "all", "train"), act_strong=g("unweighted@actor", "all", "strong"),
                   within_train=g("unweighted@actor", "actor_within_80", "train"),
                   within_strong=g("unweighted@actor", "actor_within_80", "strong"),
                   past_train=g("unweighted@actor", "actor_past_80", "train"),
                   past_strong=g("unweighted@actor", "actor_past_80", "strong"),
                   past_share=statistics.mean(s["actor_past_80_share"] for s in seeds),
                   past_strong_weighted=g("exact_strong@actor", "actor_past_80", "strong"))
        table[q] = row
        print(f"    {q:14s}  {row['rec_train']:.3f} / {row['rec_strong']:.3f}            {row['act_train']:.3f} / "
              f"{row['act_strong']:.3f}             {row['within_train']:.3f} / {row['within_strong']:.3f}            "
              f"{row['past_train']:.3f} / {row['past_strong']:.3f}            {row['past_share']:.3f}")
    return True, table


def B6_stage2(stage2):
    if not stage2:
        print("B6  stage 2's uniform bands: no --stage2 folder given; skipped (the files live only in the owner's WSL clone)")
        return None, None
    print("B6  stage 2's uniform bands (runs/shift_explore/stage2/detect/uniform/<world>/fixed/seed<k>/, WSL only): held-out "
          "rows, R, share of recorded scores at or below R, R / plain quantile, and tau.json's check (the share of "
          "calibration-split rows outside the ball whose width is above the split-conformal 95% level: about 0.05)")
    rows = []
    for world in ("stall", "no_stall"):
        for seed in range(5):
            d = Path(stage2) / world / f"seed{seed}"
            if not (d / "band.json").is_file():
                continue
            rec = json.loads((d / "band.json").read_text())["uniform"]
            s = np.load(d / "bank_scores.npz")["scores"]
            tau = json.loads((d / "tau.json").read_text()) if (d / "tau.json").is_file() else {}
            row = dict(world=world, seed=seed, rows=int(len(s)), R=rec["R"], share_R=float(np.mean(s <= rec["R"])),
                       margin=rec["R"] / rec["lambda_hat"], tau=tau.get("tau"), rate_above=tau.get("check_rows_rate_above"),
                       heldout_rows=tau.get("heldout_rows"))
            rows.append(row)
            print(f"    {world:9s} {seed}   {row['rows']}   {row['R']:.3f}   {row['share_R']:.3f}   {row['margin']:.3f}   "
                  f"tau {row['tau']}, share above {row['rate_above']} of {row['heldout_rows']} rows")
    ok = all(r["share_R"] >= 0.90 for r in rows)
    print("   ", verdict(ok))
    return ok, rows


# ------------------------------------------------------------------------------------------------ the figure

def figure(path, a3, b1, b2, b3, b5, png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 9
    # Okabe-Ito, colorblind-safe
    BLUE, ORANGE, GREEN, VERMILION, PURPLE, SKY, BLACK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.8), gridspec_kw=dict(width_ratios=[1, 1.5]))

    # left: synthetic coverage against bank size
    n = [t["n"] for t in a3]
    ax1.fill_between(n, [t["lo"] for t in a3], [t["hi"] for t in a3], color=SKY, alpha=0.35,
                     label="5% to 95% of banks (a refused bank covers 0)")
    ax1.plot(n, [t["median"] for t in a3], "o-", color=BLUE, label="median bank")
    cert = [(t["n"], t["mean_certified"]) for t in a3 if t["mean_certified"] is not None]
    ax1.plot([c[0] for c in cert], [c[1] for c in cert], "s--", color=GREEN, ms=4, lw=1,
             label="mean of the banks that did not refuse")
    ax1.axhline(0.90, color=BLACK, lw=0.8, ls="--", label="promised 0.90")
    ax1.axvspan(15, 29, color=VERMILION, alpha=0.12, label="refusal zone (fewer than about 29 rows)")
    ax1.set_xscale("log")
    ax1.set_xlim(17, 3600)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xticks([20, 29, 50, 100, 300, 843, 3000])
    ax1.set_xticklabels(["20", "29", "50", "100", "300", "843", "3000"])
    ax1.set_xlabel("held-out rows used to set the threshold (log scale)")
    ax1.set_ylabel("share of fresh rows inside the band (coverage)")
    ax1.set_title("Synthetic rows, truth known: coverage against bank size", fontsize=10)
    ax1.legend(loc="lower right", fontsize=7.5, frameon=False)

    # right: real data, one dot per seed (or seed-world), one row per condition
    conds = []
    if b2:
        get = lambda k: b2[k]["values"] if k in b2 else []
        conds += [("Fresh rows, recorded support (near-80 run, 20 bands)", get("fresh rows, recorded support"), BLUE),
                  ("Fresh rows, high-current third of the support (20)", get("fresh support rows, high current"), BLUE),
                  ("Fresh rows, hidden 70-90% throttle band (near-80 setting, 10)",
                   get("fresh rows, hidden 70-90% band (near-80 setting)"), VERMILION),
                  ("Fresh rows, removed ball, no-stall world (10)", get("fresh rows, removed ball, no-stall world"), ORANGE),
                  ("Fresh rows, removed ball, stalling world (10)", get("fresh rows, removed ball, stalling world"), VERMILION),
                  ("Fresh rows past 80% throttle, no-stall world (10)", get("fresh rows, past 80% throttle, no-stall world"), ORANGE),
                  ("Fresh rows past 80% throttle, stalling world (10)", get("fresh rows, past 80% throttle, stalling world"), VERMILION)]
    if b3:
        col = lambda k: [v for q in QUALITIES for v in b3[q][k]]
        conds += [("Weighting test: no shift, unweighted (25 seeds)", col("none_unw"), GREEN),
                  ("Weighting test: strong shift, unweighted", col("strong_unw"), ORANGE),
                  ("Weighting test: strong shift, exact weights", col("strong_exact"), GREEN),
                  ("Weighting test: inside the ball, strong, unweighted", col("inside_strong"), VERMILION)]
    if b5:
        conds += [("Actor's commands past 80%, strong law, unweighted (5 qualities)",
                   [b5[q]["past_strong"] for q in b5], PURPLE)]
    if b1:
        conds += [("Held-out calibration rows themselves (in-sample, 20 bands)", [r["share_R"] for r in b1], BLACK)]
    rng = np.random.default_rng(3)
    for i, (label, vals, color) in enumerate(conds):
        yv = i + rng.uniform(-0.18, 0.18, len(vals))
        ax2.scatter(vals, yv, s=18, color=color, alpha=0.8, edgecolor="none")
        if vals:
            ax2.plot([statistics.mean(vals)] * 2, [i - 0.3, i + 0.3], color=color, lw=2)
    ax2.axvline(0.90, color=BLACK, lw=0.8, ls="--")
    ax2.set_yticks(range(len(conds)))
    ax2.set_yticklabels([c[0] for c in conds], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, 1.02)
    ax2.set_xlabel("share of rows inside the band (coverage)")
    ax2.set_ylabel("")
    ax2.set_title("Recorded ThrusterChain rows: where the band covers 90% and where it does not", fontsize=10)
    from matplotlib.lines import Line2D
    ax2.legend(handles=[Line2D([], [], marker="o", color="gray", ls="none", ms=4, label="one dot = one seed (or seed-world)"),
                        Line2D([], [], color="gray", lw=2, label="bar = mean over the dots"),
                        Line2D([], [], color=BLACK, lw=0.8, ls="--", label="promised 0.90")],
               loc="lower left", fontsize=7.5, frameon=False)
    fig.suptitle("Module M6, the band U = R u max(eta, 1e-6): share of rows whose target y lies inside q +- U", fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    print(f"figure written: {path}")
    if png:
        fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
        print(f"preview written: {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--fresh", default="docs/bca_mechanics/data/M6_fresh_rows_coverage.json")
    ap.add_argument("--runfiles", default=None)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--figure", default="docs/bca_mechanics/figures/M6.svg")
    ap.add_argument("--png", default=None, help="also write a PNG preview of the figure here")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print(f"numpy {np.__version__}, jax {jax.__version__}, python {sys.version.split()[0]}; calibration/wbcp.py from "
          f"{wbcp.__file__}; calibration/dose.py from {DOSE.__file__}")
    print()
    print("PART A: synthetic checks (the truth is known)")
    okA1, a1 = A1_width_identity()
    okA2, a2 = A2_fresh_rows_843(a.quick)
    okA3, a3 = A3_by_bank_size(a.quick)
    okA4, a4 = A4_marginal(a.quick)
    okA5, a5 = A5_shift(a.quick)
    print()
    print("PART B: recorded ThrusterChain inputs")
    bands = read_near80_bands(a.repo, a.runfiles)
    okB1, b1 = B1_near80_banks(bands)
    okB2, b2 = B2_fresh_rows(a.fresh)
    okB3, b3 = B3_weighting_test(a.repo)
    okB4, b4 = B4_four_learners(a.repo)
    okB5, b5 = B5_channel(a.repo)
    okB6, b6 = B6_stage2(a.stage2)
    print()
    figure(a.figure, a3, b1, b2, b3, b5, a.png)
    results = dict(A1=okA1, A2=okA2, A3=okA3, A4=okA4, A5=okA5, B1=okB1, B2=okB2, B3=okB3, B6=okB6)
    print("checks:", ", ".join(f"{k} {verdict(v) if v is not None else 'skipped'}" for k, v in results.items()))
    all_ok = all(v for v in results.values() if v is not None)
    print("all run checks:", verdict(all_ok))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
