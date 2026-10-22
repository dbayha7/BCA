"""Module M5 test: BCA's threshold with refusal (docs/bca_mechanics/modules/M5.md).

What this checks, in plain words. BCA's threshold R is the score below which 90% of the held-out scores fall, with 95%
credibility; it is the larger of the plain 90% quantile and the 950th smallest of 1,000 random reweightings
("crossings"); R = +infinity means "cannot certify". The code under test is the repository's own
calibration/wbcp.py (calibrate), used unchanged.

Part 1, synthetic (the truth is known): refusal by group size against the binomial prediction; the 95% credibility
claim on exchangeable scores from a known law (the share of calibration sets whose threshold really covers 90% of the
population); small-sample behavior; stream-to-stream spread; weights of 1 and a common factor change nothing; the
exact binomial form; R(t) does not fall along the test-mass grid.

Part 2, recorded ThrusterChain inputs (nothing trained, no new band fit): the 20 near-80 version-6 bands and the 10
stage-2 uniform bands are recomputed from their recorded held-out scores with the recorded random-key rule and
compared with the recorded R, plain quantile, credible quantile, posterior spread and n_eff; the plain quantile is
checked against an independent order statistic; the stream-to-stream spread on real scores; refusal on subsamples
of real scores; the 40 recorded certificates' share of crossing draws against 1 - 0.9^n; the 18 refreshes of the
weighting test's co-trained runs; the four-learner study's refresh counts; R(t) on recorded scores.

ENVIRONMENT AND COMMANDS (both were run on 2026-10-20).
  (1) The Windows venv (Python 3.14.2, NumPy 2.5.4, JAX 0.12.0, Matplotlib 3.11.2), from the worktree M:\\ ; this run
      writes the figure and the table file:
      set PYTHONPATH=M:\\
      set PYTHONDONTWRITEBYTECODE=1
      python M:\\docs\\bca_mechanics\\scripts\\M5_test.py --root M:\\ --figure M:\\docs\\bca_mechanics\\figures\\M5.svg
          --json M:\\docs\\bca_mechanics\\data\\M5_results.json
          --near80-root \\\\wsl.localhost\\Ubuntu-24.04\\home\\user\\bca\\runs\\shift_explore\\near80
          --stage2-root \\\\wsl.localhost\\Ubuntu-24.04\\home\\user\\bca\\runs\\shift_explore\\stage2
      (without --near80-root the near-80 files are fetched with git show from origin/claude/near80-run-files; the
      stage-2 run files exist only in the owner's WSL clone, so without them Part 2 runs on the 20 near-80 bands)
  (2) The project's pinned environment (Python 3.10.20, NumPy 2.2.6, JAX 0.6.2 on CPU) inside WSL, through the owner's
      bridge (scratchpad/wslrun.ps1); the job file is docs/bca_mechanics/scripts/M5_pinned_job.sh and its output
      docs/bca_mechanics/data/M5_pinned_output.txt. The job also runs the module's own unit tests
      (python -m unittest experiments.wbcp.test_wbcp). The pinned run is the one whose numbers must match the records;
      the Windows run reproduced the same 30 thresholds bit for bit as well.
Inputs: the near-80 score files (runs/shift_explore/near80/learners/{near80,random_rows}/{stall,no_stall}/seedK/
{v6_band.json,v6_bank_scores.npz}) are read from --near80-root when present, else fetched with
`git show origin/claude/near80-run-files:<path>` into --cache; the stage-2 files from --stage2-root
(detect/uniform/<world>/fixed/seedK/{band.json,bank_scores.npz}); the certificates and the weighting test's refresh
record from the repository's outputs/ folder; the four-learner record with `git show origin/claude/shift-hosts:...`.
JAX is needed only to derive the recorded random key (jax.random.fold_in); Matplotlib only for the figure.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

import numpy as np

try:
    from calibration import wbcp
except ImportError as error:
    sys.exit(f"set PYTHONPATH to the repository root (calibration/wbcp.py): {error}")

ALPHA, BETA, DRAWS = 0.1, 0.95, 1000         # calibration/reference.py:17-20 (WBCPConfig)
NEED = math.ceil(Fraction("0.95") * DRAWS)   # 950 crossing draws are needed for a finite credible quantile
SEED = 20261020
POSTERIOR_FOLD = 1347375956                  # algorithms/td3_bc_bca.py:268
TAG_BAND = 41                                # stage2/code/band.py:85, 243-248 (the frozen-critic band's key rule)
NEAR80_BRANCH = "origin/claude/near80-run-files"
NEAR80_PREFIX = "runs/shift_explore/near80/learners"
HOSTS_BRANCH = "origin/claude/shift-hosts"
HOSTS_RESULTS = "outputs/shift_hosts_v3/results.json"
CERT_GLOB = os.path.join("outputs", "shift", "beyond-the-data", "near80", "results-20261020T115246Z", "files",
                         "learners", "*", "*", "seed*", "switch.json")
STEP1 = os.path.join("outputs", "shift", "beyond-the-data", "restructure", "step1", "results", "results.json")
DOSE_CHUNKS = os.path.join("outputs", "shift_iw", "explore", "routes2", "diagnose", "dose_chunks.json")

RESULTS = {}      # everything the figure and the write-up need
CHECKS = []       # (name, passed)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    print(f"   -> {name}: {'PASS' if ok else 'FAIL'}{('  (' + detail + ')') if detail else ''}")
    return bool(ok)


def cal(scores, stream, weights=None, test_mass=None):
    return wbcp.calibrate(scores, np.random.default_rng(stream), weights, test_mass, alpha=ALPHA, beta=BETA, draws=DRAWS)


def finite_probability(n):
    """P(at least 950 of 1,000 draws cross) when each draw crosses with probability 1 - 0.9^n (calibration/reference.py
    certifiable's reasoning): the chance that the threshold of n equally weighted scores is finite."""
    p = 1.0 - (1.0 - ALPHA) ** n
    total = 0.0
    for k in range(NEED, DRAWS + 1):
        total += math.exp(math.lgamma(DRAWS + 1) - math.lgamma(k + 1) - math.lgamma(DRAWS - k + 1)
                          + k * math.log(p) + (DRAWS - k) * math.log1p(-p))
    return total


def exact_threshold(scores):
    """The infinite-draw form of the same rule: the largest x with P(Binomial(n, 0.1) <= x) <= 0.05 scores may lie above
    the threshold (bca_channel_toy.ipynb cell 8; the toy notebooks' test 4a)."""
    s = np.sort(np.asarray(scores, np.float64))
    n = s.size
    cdf, best = 0.0, -1
    for x in range(n + 1):
        cdf += math.exp(math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
                        + x * math.log(ALPHA) + (n - x) * math.log1p(-ALPHA))
        if cdf > 1 - BETA:
            break
        best = x
    hpd = s[n - best - 1] if best >= 0 else math.inf
    lam_hat = s[math.ceil((1 - Fraction(str(ALPHA))) * n) - 1]
    return max(lam_hat, hpd)


# ----------------------------------------------------------------------------------------------------- part 1: synthetic

def synthetic_refusal(sets):
    print(f"\n[1a] refusal by group size: share of {sets} random score sets with a finite threshold, against the binomial "
          f"prediction P(Binomial(1000, 1 - 0.9^n) >= 950)")
    print("     (with equal weights, whether a draw crosses depends on the random draws only, never on the score values)")
    rng = np.random.default_rng([SEED, 11])
    rows, ok = [], True
    for n in (10, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 45, 100):
        finite = sum(math.isfinite(cal(rng.exponential(size=n), rng.integers(2 ** 31)).threshold) for _ in range(sets))
        share, pred = finite / sets, finite_probability(n)
        tol = 3 * math.sqrt(max(pred * (1 - pred), 1e-9) / sets) + 0.005
        good = abs(share - pred) <= tol and (share == 1.0 if n >= 36 else True) and (share == 0.0 if n <= 24 else True)
        ok &= good
        rows.append(dict(n=n, share=share, predicted=pred))
        print(f"     n = {n:3d}: finite {share:6.1%}   predicted {pred:6.1%}   exact form: "
              f"{'finite' if (1 - ALPHA) ** n <= 1 - BETA else '+inf'}{'' if good else '   <-- outside tolerance'}")
    RESULTS["synthetic_refusal"] = rows
    check("1a refusal by size matches the binomial prediction (0 at n <= 24, 1 at n >= 36)", ok)


def synthetic_credibility(sets):
    print(f"\n[1b] the 95% credibility claim on exchangeable scores from a known law (Exponential(1), true 90% quantile "
          f"ln 10 = {math.log(10):.4f}): over {sets} calibration sets, how often does the threshold really cover 90% of the "
          f"population?")
    rng = np.random.default_rng([SEED, 12])
    rows, ok = [], True
    for n in (36, 100, 843):
        cover_R, cover_hat, finite, ratio, top = [], [], 0, [], 0
        for _ in range(sets):
            s = rng.exponential(size=n)
            r = cal(s, rng.integers(2 ** 31))
            if not math.isfinite(r.threshold):
                continue
            finite += 1
            cover_R.append(1.0 - math.exp(-r.threshold))          # the true share of the population below R
            cover_hat.append(1.0 - math.exp(-r.lambda_hat))
            ratio.append(r.threshold / math.log(10))
            top += r.threshold == s.max()
        cover_R, cover_hat = np.array(cover_R), np.array(cover_hat)
        share_R, share_hat = float(np.mean(cover_R >= 0.9)), float(np.mean(cover_hat >= 0.9))
        row = dict(n=n, finite=finite / sets, share_R=share_R, share_hat=share_hat, mean_cov_R=float(cover_R.mean()),
                   mean_cov_hat=float(cover_hat.mean()), R_over_q90=float(np.mean(ratio)), top_share=top / finite)
        rows.append(row)
        print(f"     n = {n:3d}: finite {finite / sets:5.1%}; true coverage >= 90% in {share_R:5.1%} of sets with R "
              f"(plain quantile alone: {share_hat:5.1%}); mean true coverage {cover_R.mean():.3f} (plain {cover_hat.mean():.3f}); "
              f"R / true 90% quantile {np.mean(ratio):.3f}; R is the single largest score in {top / finite:5.1%}")
        if n == 843:
            ok &= share_R >= 0.92 and share_hat <= 0.70 and 0.90 <= cover_R.mean() <= 0.935
        else:
            ok &= share_R >= 0.92
    RESULTS["synthetic_credibility"] = rows
    check("1b with 95% credibility the threshold covers 90% of the population in at least 92% of sets (expected 95%)", ok)


def synthetic_stream_spread():
    print("\n[1c] stream-to-stream spread: one fixed score set, 50 random streams; how far can R move by chance?")
    rng = np.random.default_rng([SEED, 13])
    rows, ok = [], True
    for n in (36, 45, 100, 843):
        s = rng.exponential(size=n)
        R = np.array([cal(s, int(k)).threshold for k in rng.integers(2 ** 31, size=50)])
        R = R[np.isfinite(R)]
        spread = float(R.max() / R.min() - 1) if R.size else math.inf
        ex = exact_threshold(s)
        rows.append(dict(n=n, spread=spread, exact=ex, equal_exact=float(np.mean(R == ex))))
        print(f"     n = {n:3d}: R from {R.min():.4f} to {R.max():.4f} (max/min - 1 = {spread:.3f}); exact form {ex:.4f}, "
              f"hit in {np.mean(R == ex):5.1%} of streams")
        if n == 843:
            ok &= spread <= 0.05
    RESULTS["synthetic_spread"] = rows
    check("1c at 843 scores the threshold moves by at most 5% between streams", ok)


def synthetic_identities():
    print("\n[1d] identities: weights of 1 and test mass 1 reproduce the unweighted threshold bit for bit (40 sets); a common "
          "factor of 1,000 on weights and test mass changes nothing (40 weighted sets)")
    rng = np.random.default_rng([SEED, 14])
    unit = factor = 0
    for _ in range(40):
        n = int(rng.integers(30, 900))
        s = rng.exponential(size=n)
        k = int(rng.integers(2 ** 31))
        a, b = cal(s, k), cal(s, k, np.ones(n), 1.0)
        unit += (a.threshold == b.threshold and a.lambda_hat == b.lambda_hat and a.lambda_hpd == b.lambda_hpd
                 and np.array_equal(a.crossings, b.crossings))
        w = rng.uniform(0.2, 5.0, n)
        c, d = cal(s, k, w, 2.0), cal(s, k, 1000.0 * w, 2000.0)
        factor += c.threshold == d.threshold and np.array_equal(c.crossings, d.crossings) and abs(c.n_eff - d.n_eff) < 1e-9
    print(f"     weights of 1 identical: {unit} of 40; common factor identical: {factor} of 40")
    RESULTS["identities"] = dict(unit=unit, factor=factor)
    check("1d weights of 1 and a common factor change nothing", unit == 40 and factor == 40)


def synthetic_exact_form(sets=300):
    print(f"\n[1e] the exact binomial form: decisions 'R is at most a claimed value' agree between the 1,000-draw threshold and "
          f"the infinite-draw form ({sets} sets per n)")
    rng = np.random.default_rng([SEED, 15])
    rows, ok = [], True
    for n in (36, 45, 100, 843):
        agree = equal = 0
        for _ in range(sets):
            s = rng.exponential(size=n) * rng.uniform(0.5, 2.0)
            r = cal(s, rng.integers(2 ** 31))
            ex = exact_threshold(s)
            claimed = rng.uniform(s.min(), s.max())
            agree += (r.threshold <= claimed) == (ex <= claimed)
            equal += r.threshold == ex
        rows.append(dict(n=n, agree=agree / sets, equal=equal / sets))
        print(f"     n = {n:3d}: decisions agree {agree / sets:5.1%}; values equal {equal / sets:5.1%}")
        ok &= agree / sets >= 0.90 and (agree / sets >= 0.99 if n >= 800 else True)
    RESULTS["exact_form"] = rows
    check("1e decisions agree with the exact form in at least 90% of sets (99% at 843)", ok)


def tilt_weights(rng, n):
    """The strong tilt's exact weight shape, w(c) = 0.0907 exp(4c), on a synthetic current c ~ U(0.1, 1.1)
    (experiments/shift_iw/world.py:128-135; experiments/iw/spec.json laws.tilts). Range 0.135 to 7.39."""
    c = rng.uniform(0.1, 1.1, n)
    return 0.0907 * np.exp(4 * c)


def table_R_of_t(scores, weights, stream):
    """R(t) on the per-step grid t = 1.01^k over the strong tilt's range (402 points, as the weighting test builds it,
    experiments/shift_iw/coverage.py:153-186), then on a coarser 1.05^k extension up to t = 1000 to find the refusal
    point; every t sees the same draws (the same stream)."""
    grid = 1.01 ** np.arange(-201, 201)
    ext = 1.05 ** np.arange(41, 142)
    R = np.array([cal(scores, stream, weights, float(t)).threshold for t in grid])
    Re = np.array([cal(scores, stream, weights, float(t)).threshold for t in ext])
    w = weights / weights.max()
    t_ref = ALPHA / (1 - ALPHA) * float(w.sum()) * weights.max()   # closed-form refusal point, stage2 band.py:343
    first_inf = float(ext[np.argmax(~np.isfinite(Re))]) if (~np.isfinite(Re)).any() else math.inf
    return dict(grid=grid, R=R, ext=ext, Re=Re, monotone=bool(np.all(R[1:] >= R[:-1]) and np.all(Re[1:] >= Re[:-1])),
                t_ref=t_ref, first_inf=first_inf, R_at_1=float(R[np.argmin(np.abs(grid - 1.0))]),
                n_eff=float(w.sum() ** 2 / np.dot(w, w)))


def synthetic_monotone():
    print("\n[1f] R(t) along the per-step test-mass grid (synthetic scores, strong-tilt weights on a synthetic current): the "
          "threshold must not fall as the test mass t grows, with the same draws for every t")
    rng = np.random.default_rng([SEED, 16])
    s = rng.exponential(size=843)
    w = tilt_weights(rng, 843)
    tab = table_R_of_t(s, w, 11)
    print(f"     monotone on the 1.01 grid (0.135 to 7.39) and the 1.05 extension (up to 1000): {tab['monotone']}; "
          f"R(t = 1) = {tab['R_at_1']:.4f}; R(0.135) = {tab['R'][0]:.4f}; R(7.39) = {tab['R'][-1]:.4f}; "
          f"first +inf at t = {tab['first_inf']:.1f}; closed-form refusal point alpha/(1-alpha) * sum w = {tab['t_ref']:.1f}; "
          f"Kish effective size {tab['n_eff']:.0f} of 843")
    RESULTS["synthetic_monotone"] = tab
    check("1f R(t) never falls along the grid", tab["monotone"])


def synthetic_weighted_validity(sets=60):
    print(f"\n[1g] weights on exchangeable scores (the advisor's weighting question at the threshold level): {sets} sets of "
          "843 Exponential(1) scores; the strong tilt's weight shape on a synthetic current the scores do not depend on; "
          "one test mass t = the mean weight. Is the weighted threshold still valid, and how much wider is it?")
    rng = np.random.default_rng([SEED, 17])
    ratio, cov_w, cov_u, neff, mar_w, mar_u, hat_ratio = [], [], [], [], [], [], []
    for _ in range(sets):
        s = rng.exponential(size=843)
        w = tilt_weights(rng, 843)
        k = int(rng.integers(2 ** 31))
        u, v = cal(s, k), cal(s, k, w, float(w.mean()))
        ratio.append(v.threshold / u.threshold)
        hat_ratio.append(v.lambda_hat / u.lambda_hat)
        cov_w.append(1.0 - math.exp(-v.threshold))
        cov_u.append(1.0 - math.exp(-u.threshold))
        neff.append(v.n_eff)
        mar_w.append(v.lambda_hpd / v.lambda_hat - 1)
        mar_u.append(u.lambda_hpd / u.lambda_hat - 1)
    ratio, cov_w, cov_u = np.array(ratio), np.array(cov_w), np.array(cov_u)
    share_w, share_u = float(np.mean(cov_w >= 0.9)), float(np.mean(cov_u >= 0.9))
    print(f"     R_weighted / R_unweighted: mean {ratio.mean():.3f}, from {ratio.min():.3f} to {ratio.max():.3f} "
          f"(above 1 in {np.mean(ratio > 1):.0%} of sets); plain weighted / plain unweighted quantile: mean {np.mean(hat_ratio):.3f}")
    print(f"     Bayesian margin (credible / plain - 1): weighted {np.mean(mar_w):.1%}, unweighted {np.mean(mar_u):.1%}; "
          f"Kish effective size {np.mean(neff):.0f} of 843")
    print(f"     true coverage >= 90% in {share_w:.1%} of sets (weighted) and {share_u:.1%} (unweighted); mean true coverage "
          f"{cov_w.mean():.3f} (weighted), {cov_u.mean():.3f} (unweighted)")
    RESULTS["weighted_exchangeable"] = dict(sets=sets, ratio_mean=float(ratio.mean()), ratio_min=float(ratio.min()),
                                            ratio_max=float(ratio.max()), ratio_above_1=float(np.mean(ratio > 1)),
                                            hat_ratio_mean=float(np.mean(hat_ratio)), margin_w=float(np.mean(mar_w)),
                                            margin_u=float(np.mean(mar_u)), n_eff=float(np.mean(neff)), share_w=share_w,
                                            share_u=share_u, cov_w=float(cov_w.mean()), cov_u=float(cov_u.mean()))
    check("1g with shift weights on exchangeable scores the threshold stays valid (true coverage >= 90% in at least 92% "
          "of sets); any widening comes from the smaller effective sample, not from the scores", share_w >= 0.92)


# ------------------------------------------------------------------------------------------------- part 2: recorded data

def posterior_stream(rep, steps):
    """The recorded key rule of the frozen-critic band fits: refresh key = fold_in(PRNGKey(default_rng([20261015, rep,
    41, 1]).integers(2^31 - 1)), steps) (stage2/code/band.py:243-248; near80/code/v6bca.py:57), then the posterior
    stream = uint32 words of fold_in(refresh key, 1347375956) (algorithms/td3_bc_bca.py:268, reference.py:177)."""
    import jax
    seed_int = int(np.random.default_rng([20261015, int(rep), TAG_BAND, 1]).integers(2 ** 31 - 1))
    refresh_key = jax.random.fold_in(jax.random.PRNGKey(seed_int), int(steps))
    return np.asarray(jax.random.fold_in(refresh_key, POSTERIOR_FOLD), np.uint32).ravel()


def git_show(root, spec, dest):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        data = subprocess.run(["git", "-C", root, "show", spec], check=True, capture_output=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def band_records(args):
    """(label, setting, world, seed, steps, record dict, scores) for the 20 near-80 bands and the 10 stage-2 bands."""
    out = []
    for setting in ("near80", "random_rows"):
        for world in ("stall", "no_stall"):
            for seed in range(5):
                rel = f"{setting}/{world}/seed{seed}"
                local = os.path.join(args.near80_root, "learners", setting, world, f"seed{seed}")
                files = {}
                for name in ("v6_band.json", "v6_bank_scores.npz"):
                    p = os.path.join(local, name)
                    if not os.path.exists(p):
                        p = git_show(args.root, f"{NEAR80_BRANCH}:{NEAR80_PREFIX}/{rel}/{name}",
                                     os.path.join(args.cache, "near80", rel, name))
                    files[name] = p
                if all(files.values()):
                    d = json.load(open(files["v6_band.json"], encoding="utf8"))
                    s = np.load(files["v6_bank_scores.npz"])["scores"]
                    out.append(("near-80 " + setting, setting, world, seed, int(d["steps"]), d, s))
    for world in ("stall", "no_stall"):
        for seed in range(5):
            d = os.path.join(args.stage2_root, "detect", "uniform", world, "fixed", f"seed{seed}")
            pj, pn = os.path.join(d, "band.json"), os.path.join(d, "bank_scores.npz")
            if os.path.exists(pj) and os.path.exists(pn):
                rec = json.load(open(pj, encoding="utf8"))
                out.append(("stage 2 uniform", "stage2", world, int(rec["rep"]), int(rec["steps"]), rec,
                            np.load(pn)["scores"]))
    return out


def real_reproduction(bands):
    print(f"\n[2a] reproduction: {len(bands)} recorded bands, each threshold recomputed from its recorded held-out scores with "
          f"the recorded key rule, against the record (float64 'refresh' block and float32 'uniform' block)")
    print(f"     {'band':22s} {'world':8s} seed    n   plain q   credible q   R recorded   R recomputed  eq64 eq32  margin  "
          f"share<=R  share<=q")
    rows, ok = [], True
    try:
        import jax  # noqa: F401
    except ImportError:
        print("     JAX is not installed: the recorded key cannot be derived; skipping the recompute")
        check("2a thresholds reproduce bit for bit", False, "jax missing")
        return rows
    for label, setting, world, seed, steps, d, s in bands:
        r = cal(s, posterior_stream(seed, steps))
        rec, uni = d["refresh"], d["uniform"]
        n = s.size
        eq64 = (r.threshold == rec["threshold"] and r.lambda_hat == rec["lambda_hat"] and r.lambda_hpd == rec["lambda_hpd"]
                and abs(r.sigma_post - rec["sigma_post"]) <= 1e-12 and r.n_eff == rec["n_eff"])
        eq32 = bool(np.float32(r.threshold) == np.float32(uni["R"]))
        srt = np.sort(s)
        indep_hat = float(srt[math.ceil(0.9 * n) - 1])                 # the ceil(0.9 n)-th smallest score
        indep_hpd = float(np.sort(r.crossings)[NEED - 1])                # the 950th smallest crossing
        consistent = (indep_hat == r.lambda_hat and indep_hpd == r.lambda_hpd and r.threshold == max(r.lambda_hat, r.lambda_hpd)
                      and np.isfinite(r.crossings).all())
        margin = r.lambda_hpd / r.lambda_hat - 1
        share_R, share_hat = float(np.mean(s <= r.threshold)), float(np.mean(s <= r.lambda_hat))
        ok &= eq64 and eq32 and consistent
        rows.append(dict(label=label, setting=setting, world=world, seed=seed, n=n, lambda_hat=r.lambda_hat,
                         lambda_hpd=r.lambda_hpd, R=r.threshold, R_recorded=rec["threshold"], eq64=eq64, eq32=eq32,
                         consistent=bool(consistent), margin=margin, share_R=share_R, share_hat=share_hat,
                         sigma_post=r.sigma_post, R32=uni["R"]))
        print(f"     {label:22s} {world:8s} {seed:3d}  {n:4d}  {r.lambda_hat:8.4f}   {r.lambda_hpd:8.4f}   "
              f"{rec['threshold']:10.6f}   {r.threshold:10.6f}   {str(eq64)[0]}    {str(eq32)[0]}   {margin:6.1%}   "
              f"{share_R:6.3f}    {share_hat:6.3f}")
    if rows:
        m = np.array([x["margin"] for x in rows])
        print(f"     Bayesian margin (credible / plain - 1): {m.min():.1%} to {m.max():.1%}, mean {m.mean():.1%}; "
              f"the plain quantile never bound (R = credible quantile in {sum(x['R'] == x['lambda_hpd'] for x in rows)} of {len(rows)})")
    RESULTS["real_reproduction"] = rows
    check(f"2a all {len(rows)} thresholds reproduce bit for bit (float64 and float32) and R = max(plain, credible) with "
          f"independent order statistics", ok and bool(rows))


def real_stream_spread(bands, streams=20):
    print(f"\n[2b] stream-to-stream spread on the recorded scores: each band recomputed with {streams} other random streams")
    rng = np.random.default_rng([SEED, 22])
    rows, worst = [], 0.0
    for label, setting, world, seed, steps, d, s in bands:
        R0 = d["refresh"]["threshold"]
        R = np.array([cal(s, int(k)).threshold for k in rng.integers(2 ** 31, size=streams)])
        lo, hi = float(R.min() / R0 - 1), float(R.max() / R0 - 1)
        worst = max(worst, abs(lo), abs(hi))
        rows.append(dict(label=label, world=world, seed=seed, Rmin=float(R.min()), Rmax=float(R.max()), R0=R0))
    print(f"     largest relative move of R across {len(bands)} bands x {streams} streams: {worst:.1%}")
    RESULTS["real_spread"] = rows
    check("2b on about 820 real scores R moves by at most 5% between streams", worst <= 0.05, f"worst {worst:.1%}")


def real_subsample_refusal(bands, sets):
    print(f"\n[2c] refusal on real scores: subsamples (without replacement) of a recorded score set, {sets} per size, share with "
          f"a finite threshold against the binomial prediction (the same curve as 1a, now on real scores)")
    picks = [b for b in bands if b[1] == "near80" and b[3] == 0] or bands[:2]
    rng = np.random.default_rng([SEED, 23])
    rows, ok = [], True
    for label, setting, world, seed, steps, d, s in picks[:2]:
        print(f"     {label} {world} seed {seed} ({s.size} scores):")
        for n in (20, 24, 26, 28, 29, 30, 32, 36):
            finite = sum(math.isfinite(cal(rng.choice(s, n, replace=False), rng.integers(2 ** 31)).threshold)
                         for _ in range(sets))
            share, pred = finite / sets, finite_probability(n)
            tol = 3 * math.sqrt(max(pred * (1 - pred), 1e-9) / sets) + 0.005
            good = abs(share - pred) <= tol
            ok &= good
            rows.append(dict(world=world, seed=seed, n=n, share=share, predicted=pred))
            print(f"       n = {n:2d}: finite {share:6.1%}   predicted {pred:6.1%}{'' if good else '   <-- outside tolerance'}")
    RESULTS["real_subsample"] = rows
    check("2c refusal on real scores follows the same binomial curve", ok)


def real_certificates(root):
    print("\n[2d] the 40 recorded certificates (near-80 run, 30; step 1, 10): the recorded share of the 1,000 draws that cross "
          "in the full-throttle group against 1 - 0.9^n_tail, and the rule 'R_tail finite iff at least 950 cross'")
    num = lambda v: float(v) if isinstance(v, str) else v
    rows = []
    for path in sorted(glob.glob(os.path.join(root, CERT_GLOB))):
        d = json.load(open(path, encoding="utf8"))
        c = d["certificate"]
        rows.append(dict(source="near-80", setting=d["setting"], world=d["world"], seed=d["seed"], n_tail=c["n_tail"],
                         n_rest=c["n_rest"], share_tail=c["finite_share_tail"], share_rest=c["finite_share_rest"],
                         R_tail=num(c["R_tail"]), on=c["on"]))
    p1 = os.path.join(root, STEP1)
    if os.path.exists(p1):
        s1 = json.load(open(p1, encoding="utf8"))
        for key in sorted(s1["per_seed_world"]):
            world, seed = key.split("/seed")
            c = s1["per_seed_world"][key]["certificate"]
            rows.append(dict(source="step 1", setting="main", world=world, seed=int(seed), n_tail=c["n_tail"],
                             n_rest=c["n_rest"], share_tail=c["finite_share_tail"], share_rest=c["finite_share_rest"],
                             R_tail=num(c["R_tail"]), on=c["on"]))
    if not rows:
        check("2d certificates", False, "no certificate records found")
        return
    print(f"     {'source':8s} {'setting':12s} {'world':9s} seed  n_tail  share crossing  1-0.9^n  within 3 SE  R_tail   rule")
    ok_share = ok_rule = 0
    for r in sorted(rows, key=lambda x: x["n_tail"]):
        pred = 1 - 0.9 ** r["n_tail"]
        se = math.sqrt(pred * (1 - pred) / DRAWS)
        within = abs(r["share_tail"] - pred) <= 3 * se
        rule = math.isfinite(r["R_tail"]) == (r["share_tail"] >= 0.95)
        ok_share += within
        ok_rule += rule
        r.update(predicted=pred, within=within, rule=rule)
        print(f"     {r['source']:8s} {r['setting']:12s} {r['world']:9s} {r['seed']:4d}  {r['n_tail']:5d}   {r['share_tail']:8.3f}     "
              f"{pred:7.3f}   {str(within):5s}      {('%.3f' % r['R_tail']) if math.isfinite(r['R_tail']) else '+inf':>7s}  {str(rule)}")
    ninf = sum(not math.isfinite(r["R_tail"]) for r in rows)
    sizes_inf = sorted(r["n_tail"] for r in rows if not math.isfinite(r["R_tail"]))
    sizes_fin = sorted(r["n_tail"] for r in rows if math.isfinite(r["R_tail"]))
    rest_ok = all(r["share_rest"] == 1.0 for r in rows)
    print(f"     share within 3 standard errors of 1 - 0.9^n: {ok_share} of {len(rows)}; rule 'finite iff >= 950 cross' holds: "
          f"{ok_rule} of {len(rows)}; infinite R_tail in {ninf} of {len(rows)} (group sizes {sizes_inf}); finite group sizes "
          f"from {min(sizes_fin)} to {max(sizes_fin)}; the rest group (n_rest {min(r['n_rest'] for r in rows)}-"
          f"{max(r['n_rest'] for r in rows)}) had every draw cross in all {len(rows)}: {rest_ok}")
    RESULTS["certificates"] = rows
    check("2d the recorded crossing shares follow 1 - 0.9^n and refusal follows the 950 rule in every certificate",
          ok_share == len(rows) and ok_rule == len(rows) and rest_ok)


def real_cotraining(root):
    print("\n[2e] the refresh schedule in co-training (weighting test, exploratory rerun, seed 0, five data qualities, two "
          "arms): the unweighted threshold at each of the 18 refreshes (update 10,000, then every 5,000)")
    p = os.path.join(root, DOSE_CHUNKS)
    if not os.path.exists(p):
        check("2e co-training refreshes", False, "dose_chunks.json not found")
        return
    d = json.load(open(p, encoding="utf8"))
    rows, all_finite, first_equal = [], True, True
    for key, v in d.items():
        thr = [float(x) for x in v["thr"]]
        all_finite &= all(math.isfinite(x) for x in thr) and len(thr) == 18
        rows.append(dict(run=key, thr=thr))
        print(f"     {key:28s} refreshes {len(thr):2d}, all finite {all(math.isfinite(x) for x in thr)}; first {thr[0]:.4f}, "
              f"last {thr[-1]:.4f}, min {min(thr):.4f}, max {max(thr):.4f}")
    for q in ("random", "medium-replay", "medium", "medium-expert", "expert"):
        a, b = d.get(f"{q}/unweighted"), d.get(f"{q}/exact_strong")
        if a and b:
            first_equal &= a["thr"][0] == b["thr"][0]
    print(f"     the two arms of a quality share the first refresh's threshold exactly (identical runs until the first "
          f"refresh): {first_equal}")
    RESULTS["cotraining"] = rows
    check("2e every co-training refresh certified (18 of 18 in all 10 runs)", all_finite)


def real_hosts(args):
    print("\n[2f] the four-learner study's record (standard and weighted arms): certified refreshes and the stop rule S3 "
          "(the unweighted threshold reproduced bit for bit from the bank scores at every refresh)")
    p = git_show(args.root, f"{HOSTS_BRANCH}:{HOSTS_RESULTS}", os.path.join(args.cache, "hosts", "results.json"))
    if not p:
        print("     the shift-hosts branch is not reachable; skipped")
        return
    d = json.load(open(p, encoding="utf8"))
    # 'alone' and 'room' run without a band; 'sham_alone' is a no-band control whose record carries 0 refreshes by
    # design; stop rule S3 (the IW step reproduces the standard threshold) is recorded for the weighted arm 'iw' only
    # (the standard arm stores the flag as false because it never runs the IW step).
    cert, s3_true, s3_all, runs, per = [], 0, 0, 0, {}
    for q, body in d.get("qualities", {}).items():
        for learner, e in body.get("expanded", {}).items():
            for setup, s in e.get("setups", {}).items():
                if setup not in ("standard", "iw"):
                    continue
                band = s.get("band") or {}
                if "certified_refreshes" in band:
                    c = band["certified_refreshes"]["per_seed"]
                    cert.extend(c)
                    runs += len(c)
                    per[(q, learner, setup)] = (len(c), min(c), max(c))
                if setup == "iw":
                    for flags in band.get("s3_recomputed_equals_refresh_scores", []):
                        s3_all += len(flags)
                        s3_true += sum(bool(x) for x in flags)
    for (q, learner, setup), (k, lo, hi) in sorted(per.items()):
        print(f"     {q:14s} {learner:7s} {setup:9s} {k} seeds: certified refreshes {lo:.0f}-{hi:.0f} of 18")
    print(f"     runs with a band (standard and iw arms): {runs}; certified refreshes per run: min {min(cert) if cert else '-'}, "
          f"max {max(cert) if cert else '-'} (18 scheduled); S3 reproduced {s3_true} of {s3_all} (iw arms)")
    RESULTS["hosts"] = dict(runs=runs, cert=cert, s3_true=s3_true, s3_all=s3_all,
                            per={f"{q}/{l}/{s}": v for (q, l, s), v in per.items()})
    check("2f every recorded four-learner refresh certified (standard and iw arms) and S3 reproduced (iw arms)",
          bool(cert) and min(cert) == 18 and s3_all > 0 and s3_true == s3_all)


def real_monotone(bands):
    print("\n[2g] R(t) along the test-mass grid on recorded scores (near-80 stall seed 0; weights from the strong tilt's shape "
          "on a synthetic current: mechanics only, not a claim about the run)")
    picks = [b for b in bands if b[1] == "near80" and b[2] == "stall" and b[3] == 0]
    if not picks:
        return
    s = picks[0][6]
    rng = np.random.default_rng([SEED, 27])
    tab = table_R_of_t(s, tilt_weights(rng, s.size), 11)
    print(f"     monotone: {tab['monotone']}; R(1) = {tab['R_at_1']:.4f} (recorded unweighted R {picks[0][5]['refresh']['threshold']:.4f}); "
          f"R(0.135) = {tab['R'][0]:.4f}; R(7.39) = {tab['R'][-1]:.4f}; first +inf at t = {tab['first_inf']:.1f}; closed-form "
          f"refusal point {tab['t_ref']:.1f}; Kish effective size {tab['n_eff']:.0f} of {s.size}")
    RESULTS["real_monotone"] = tab
    check("2g R(t) never falls along the grid on recorded scores", tab["monotone"])


# ------------------------------------------------------------------------------------------------------------- figure

def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"  # keep the figure's text as text, not outlines
    blue, orange, green, verm, grey = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6E6E6E"
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    fig.suptitle("Module M5: the threshold with refusal", fontsize=13, y=0.995)

    ax = axes[0, 0]
    n = np.arange(10, 61)
    ax.plot(n, [finite_probability(k) for k in n], color=grey, lw=2, label="binomial prediction P(Bin(1000, 1 - 0.9^n) >= 950)")
    if "synthetic_refusal" in RESULTS:
        r = RESULTS["synthetic_refusal"]
        ax.plot([x["n"] for x in r], [x["share"] for x in r], "o", color=blue, ms=7, mfc="white", mew=2, label="synthetic scores")
    if "real_subsample" in RESULTS:
        r = [x for x in RESULTS["real_subsample"] if x["world"] == "stall"]
        ax.plot([x["n"] for x in r], [x["share"] for x in r], "s", color=verm, ms=6, label="subsamples of recorded scores")
    for k, txt in ((29, "29"), (36, "36")):
        ax.axvline(k, color=grey, lw=0.8, ls=":")
        ax.text(k + 0.3, 0.05, txt, color=grey, fontsize=9)
    ax.set_xlim(9, 60)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("held-out rows n (equal weights)")
    ax.set_ylabel("share of score sets with a finite threshold R")
    ax.set_title("A. Refusal depends on the group size only", fontsize=11, loc="left")
    ax.legend(loc="center right", fontsize=8.5, frameon=False)

    ax = axes[0, 1]
    if "certificates" in RESULTS:
        c = RESULTS["certificates"]
        nn = np.arange(15, 65)
        ax.plot(nn, 1 - 0.9 ** nn, color=grey, lw=2, label="1 - 0.9^n (chance that one draw crosses)")
        ax.axhline(0.95, color=green, lw=1.2, ls="--", label="0.95: at least 950 of 1,000 must cross")
        fin = [x for x in c if math.isfinite(x["R_tail"])]
        inf = [x for x in c if not math.isfinite(x["R_tail"])]
        ax.plot([x["n_tail"] for x in fin], [x["share_tail"] for x in fin], "o", color=blue, ms=7, mfc="white", mew=2,
                label=f"R_tail finite ({len(fin)} certificates)")
        ax.plot([x["n_tail"] for x in inf], [x["share_tail"] for x in inf], "x", color=verm, ms=9, mew=2.5,
                label=f"R_tail = +inf ({len(inf)} certificates)")
        ax.set_xlim(15, 62)
        ax.set_ylim(0.80, 1.005)
    ax.set_xlabel("rows in the full-throttle group (n_tail)")
    ax.set_ylabel("recorded share of the 1,000 draws that cross")
    ax.set_title("B. The 40 recorded certificates follow the same rule", fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    ax = axes[1, 0]
    if "real_reproduction" in RESULTS:
        rows = RESULTS["real_reproduction"]
        spread = {(x["label"], x["world"], x["seed"]): x for x in RESULTS.get("real_spread", [])}
        xs = np.arange(len(rows))
        for i, x in enumerate(rows):
            sp = spread.get((x["label"], x["world"], x["seed"]))
            if sp:
                ax.plot([i, i], [sp["Rmin"], sp["Rmax"]], color=grey, lw=1.2, solid_capstyle="butt",
                        label="range over 20 other random streams" if i == 0 else None)
        ax.plot(xs, [x["lambda_hat"] for x in rows], "_", color=orange, ms=12, mew=2.5, label="plain 90% quantile (recomputed)")
        ax.plot(xs, [x["R_recorded"] for x in rows], "o", color=blue, ms=8, mfc="white", mew=2, label="R recorded")
        ax.plot(xs, [x["R"] for x in rows], "x", color=verm, ms=7, mew=2, label="R recomputed from the recorded scores")
        n80 = sum(x["setting"] != "stage2" for x in rows)
        if 0 < n80 < len(rows):
            ax.axvline(n80 - 0.5, color=grey, lw=0.8, ls=":")
            ax.text(n80 / 2, 1.215, f"near-80 ({n80} bands)", ha="center", fontsize=9, color=grey)
            ax.text(n80 + (len(rows) - n80) / 2, 1.215, f"stage 2 ({len(rows) - n80})", ha="center", fontsize=9, color=grey)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x['world'][:2]}{x['seed']}" for x in rows], fontsize=7, rotation=90)
        ax.set_ylim(1.20, 1.80)
    ax.set_xlabel("recorded band (st = stalling world, no = no-stall world; digit = seed)")
    ax.set_ylabel("threshold (score units)")
    if RESULTS.get("real_reproduction"):
        rows = RESULTS["real_reproduction"]
        m = np.array([x["margin"] for x in rows])
        ax.set_title(f"C. {len(rows)} recorded thresholds reproduce bit for bit;\nthe credible quantile sits "
                     f"{m.min():.0%}-{m.max():.0%} above the plain one (mean {m.mean():.0%})", fontsize=11, loc="left")
    else:
        ax.set_title("C. recorded thresholds (no band records found)", fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)

    ax = axes[1, 1]
    for key, color, label in (("synthetic_monotone", blue, "synthetic scores (n = 843)"),
                              ("real_monotone", verm, "recorded scores (near-80 stall seed 0)")):
        if key in RESULTS:
            t = RESULTS[key]
            tt = np.concatenate([t["grid"], t["ext"]])
            RR = np.concatenate([t["R"], t["Re"]])
            fin = np.isfinite(RR)
            ax.plot(tt[fin], RR[fin], color=color, lw=2, label=label)
            if (~fin).any():
                ax.axvline(tt[np.argmax(~fin)], color=color, lw=1, ls="--")
    ax.axvspan(0.135, 7.39, color=grey, alpha=0.12, lw=0, label="the strong tilt's mass range (the per-step grid, ratio 1.01)")
    ax.set_xscale("log")
    ax.set_xlabel("test mass t (the weight of the row being certified)")
    ax.set_ylabel("threshold R(t)")
    ax.set_title("D. R(t) never falls as the test mass grows;\n+inf beyond the dashed line (refusal)", fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    for a in axes.ravel():
        a.grid(True, color="#DDDDDD", lw=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fmt = os.path.splitext(path)[1].lstrip(".").lower() or "svg"
    fig.savefig(path, format=fmt, dpi=110)
    print(f"\nfigure written: {path}")


def jsonable(x):
    """RESULTS as plain JSON: arrays to lists, non-finite floats to strings."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x):
            return "nan"
        return x if math.isfinite(x) else ("inf" if x > 0 else "-inf")
    return x


# --------------------------------------------------------------------------------------------------------------- main

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..", "..")), help="repository root")
    ap.add_argument("--part", choices=("all", "synthetic", "real"), default="all")
    ap.add_argument("--figure", default=None, help="write the SVG figure here")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "bca_m5_cache"))
    ap.add_argument("--near80-root", default=None, help="runs/shift_explore/near80 (default: under --root)")
    ap.add_argument("--stage2-root", default=None, help="runs/shift_explore/stage2 (default: under --root)")
    ap.add_argument("--sets", type=int, default=400, help="synthetic score sets per group size")
    ap.add_argument("--json", default=None, help="write every number behind the tables and the figure here")
    args = ap.parse_args()
    args.near80_root = args.near80_root or os.path.join(args.root, "runs", "shift_explore", "near80")
    args.stage2_root = args.stage2_root or os.path.join(args.root, "runs", "shift_explore", "stage2")
    t0 = time.time()
    print(f"python {sys.version.split()[0]}; numpy {np.__version__}; calibration/wbcp.py from {wbcp.__file__}")
    print(f"alpha {ALPHA}, credibility {BETA}, draws {DRAWS}: a finite credible quantile needs {NEED} crossing draws")
    if args.part in ("all", "synthetic"):
        print("\n=== Part 1: synthetic scores, truth known ===")
        synthetic_refusal(args.sets)
        synthetic_credibility(min(args.sets, 500))
        synthetic_stream_spread()
        synthetic_identities()
        synthetic_exact_form()
        synthetic_monotone()
        synthetic_weighted_validity()
    if args.part in ("all", "real"):
        print("\n=== Part 2: recorded ThrusterChain inputs ===")
        bands = band_records(args)
        print(f"recorded bands found: {sum(b[1] != 'stage2' for b in bands)} near-80, {sum(b[1] == 'stage2' for b in bands)} stage 2")
        if bands:
            real_reproduction(bands)
            real_stream_spread(bands)
            real_subsample_refusal(bands, args.sets)
        real_certificates(args.root)
        real_cotraining(args.root)
        real_hosts(args)
        if bands:
            real_monotone(bands)
    print("\n=== Summary ===")
    for name, ok in CHECKS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"all checks: {'PASS' if all(ok for _, ok in CHECKS) else 'FAIL'}  ({time.time() - t0:.0f} s)")
    if args.figure and not args.no_figure:
        figure(args.figure)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf8") as f:
            json.dump(dict(script="docs/bca_mechanics/scripts/M5_test.py", python=sys.version.split()[0],
                           numpy=np.__version__, part=args.part, sets=args.sets, seed=SEED,
                           checks=[dict(name=n, passed=ok) for n, ok in CHECKS], results=jsonable(RESULTS)),
                      f, indent=1)
        print(f"table file written: {args.json}")
    return 0 if all(ok for _, ok in CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
