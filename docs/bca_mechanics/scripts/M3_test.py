"""Module M3 of MODULES.md, the scores: synthetic checks with a known truth, and the real-data checks on the recorded
ThrusterChain bands. Prints the tables that docs/bca_mechanics/modules/M3.md quotes and writes the figure
docs/bca_mechanics/figures/M3.svg.

What M3 computes (calibration/reference.py:148-149 and 164-175): on each held-out calibration row j,
    sigma_j = max(eta_f(s_j, a_j), 1e-6) * u_f        in float32, then converted to float64        (reference.py:164)
    rho_j   = |y_j - q_j| / sigma_j                    in float64 (|y - q| from the float32 residual)  (reference.py:175)
with the error-size network eta_f and the unit u_f frozen at the refresh.

HOW TO RUN (no training, nothing written outside docs/bca_mechanics/figures):
    cd <repository root>
    set PYTHONPATH=<repository root>          (PowerShell: $env:PYTHONPATH = (Get-Location).Path)
    python docs/bca_mechanics/scripts/M3_test.py
Environment used for the committed output: the Windows venv, Python 3.14.2, numpy 2.5.4, jax 0.12.0 (CPU),
matplotlib 3.11.2; about 90 seconds. Parts A and B and the figure run anywhere the repository imports.
The PINNED environment (WSL, Python 3.10.20, numpy 2.2.6, jax 0.6.2 on CPU) was used for the two companion jobs,
run through the owner's WSL bridge (scratchpad/wslrun.ps1) from /home/user/bca:
    scripts/M3_recompute_pinned.sh  -> scripts/M3_recompute_pinned.py -> data/M3_pinned_recompute.json  (part C)
    scripts/M3_bank_rows_pinned.sh  -> scripts/M3_bank_rows_pinned.py -> data/M3_bank_rows.npz, .json   (part D)
Both reports are read here when present; without them parts C and D say so and are skipped.
Every number this script prints comes from:
  - synthetic rows generated here (part A), or
  - the recorded near-80 version-6 band files on branch claude/near80-run-files (fetched with `git show`; part B), or
  - the recorded stage-2 band files, which exist only on the WSL side (runs/shift_explore/stage2/detect/uniform;
    read through \\wsl.localhost when reachable, else skipped; part B), or
  - the two pinned-environment reports (parts C and D).
Options: --repo DIR (repository root; default: three levels above this file), --runfiles DIR (a checkout of the
run-files branch or the WSL mount, instead of git show), --stage2 DIR, --pinned FILE, --rows FILE, --figure FILE,
--no-figure.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from calibration import wbcp
    from calibration.reference import WBCPConfig, freeze_reference, positive_scale
except ImportError as error:
    sys.exit(f"set PYTHONPATH to the repository root (calibration/ must import) and install jax: {error}")

ALPHA, BETA, DRAWS = 0.1, 0.95, 1000
KEY_TARGET, KEY_POSTERIOR = 1213156420, 1347375956           # td3_bc_bca.refresh's fold-in constants (lines 249, 268)
TAG_BAND, BAND_FIT_STEPS = 41, 20_000                          # band_rows_and_keys (stage 2 band.py:85, 89)
NEAR80_BRANCH = ("origin/claude/near80-run-files", "claude/near80-run-files")
NEAR80_DIR = "runs/shift_explore/near80/learners/near80"
STAGE2_WSL = r"\\wsl.localhost\Ubuntu-24.04\home\user\bca\runs\shift_explore\stage2\detect\uniform"
Q90_HALF_NORMAL = 1.6448536269514722                           # the 90% quantile of |Z|, Z standard normal
CURRENT_INDEX = 10                                             # the water current is the 11th state number


def verdict(ok):
    return "PASS" if ok else "FAIL"


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def f32eq(a, b):
    return bool(np.float32(a).tobytes() == np.float32(b).tobytes())


def scores_of(eta, unit, residual):
    """M3 exactly as freeze_reference computes it (reference.py:164, 175): float32 scale, then float64."""
    scale = np.asarray(positive_scale(jnp.asarray(eta, jnp.float32), jnp.asarray(unit, jnp.float32)), np.float64)
    return np.abs(np.asarray(residual, np.float64)) / scale, scale


def refresh_key(seed):
    """The frozen-critic bands' refresh key (stage 2 band.py:243-250, used by near-80's v6bca.fit as well):
    fold_in(PRNGKey(default_rng([20261015, seed, 41, 1]).integers(2**31 - 1)), 20000)."""
    base = int(np.random.default_rng([20261015, int(seed), TAG_BAND, 1]).integers(2 ** 31 - 1))
    return jax.random.fold_in(jax.random.PRNGKey(base), BAND_FIT_STEPS)


def words(key):
    return np.asarray(key, np.uint32).ravel()


# ============================================================================================ part A: synthetic
def part_a():
    print("=" * 100)
    print("PART A. Synthetic checks of M3 (the truth is known by construction)")
    print("=" * 100)
    g = np.random.default_rng(20261020)
    n = 2000
    unit = np.float32(0.37)
    eta = g.uniform(0.5, 2.0, n).astype(np.float32)          # a made-up error-size network's output, > 0.01
    z = g.standard_normal(n)
    config = WBCPConfig()
    key = jax.random.fold_in(jax.random.PRNGKey(7), 11)
    results = []

    # A1. known truth: residual = u * eta * Z, so the score is |Z| up to float32 rounding; 90% of |Z| <= 1.645.
    residual = (unit * eta * z).astype(np.float32)
    rho, scale = scores_of(eta, unit, residual)
    rel = np.abs(rho - np.abs(z)) / np.abs(z)
    share = float(np.mean(rho <= Q90_HALF_NORMAL))
    ref, valid, diag = freeze_reference({"w": jnp.ones(2)}, jnp.asarray(unit), jnp.asarray(eta),
                                        jnp.asarray(residual), key, config)
    ok1 = bool(np.max(rel) < 1e-5 and abs(share - 0.90) < 0.02 and valid and 1.55 < float(ref.threshold) < 1.80)
    results.append(("A1", ok1))
    print(f"A1 {verdict(ok1)}  residual = u*eta*Z: the scores equal |Z| (largest relative difference {np.max(rel):.1e},"
          f" float32 rounding); share of scores <= 1.645 = {share:.3f} (truth 0.900);"
          f" freeze_reference: R = {float(ref.threshold):.4f}, plain 90% quantile {float(ref.lambda_hat):.4f},"
          f" credible {float(ref.lambda_hpd):.4f} (truth of the plain quantile 1.645; R adds the 95% margin)")

    # A2. an error-size network that matches the 90% level gives R near 1.
    residual2 = (unit * eta * z / Q90_HALF_NORMAL).astype(np.float32)
    rho2, _ = scores_of(eta, unit, residual2)
    ref2, valid2, _ = freeze_reference({"w": jnp.ones(2)}, jnp.asarray(unit), jnp.asarray(eta),
                                       jnp.asarray(residual2), key, config)
    share2 = float(np.mean(rho2 <= 1.0))
    ok2 = bool(valid2 and abs(share2 - 0.90) < 0.02 and 0.95 < float(ref2.threshold) < 1.10)
    results.append(("A2", ok2))
    print(f"A2 {verdict(ok2)}  residual = u*eta*Z/1.645 (eta matches the 90% level): share <= 1 = {share2:.3f},"
          f" R = {float(ref2.threshold):.4f} (near 1: the threshold only corrects what eta and u miss)")

    # A3. only the product u*eta matters, and a common factor on u and the residual cancels (exact for powers of 2).
    rho3a, _ = scores_of(eta / 2, unit * 2, residual)
    rho3b, _ = scores_of(eta, unit * 8, residual * 8)
    rho3c, _ = scores_of(eta, unit * 3, residual * 3)
    ok3 = bool(np.array_equal(rho3a, rho) and np.array_equal(rho3b, rho)
               and np.max(np.abs(rho3c - rho) / rho) < 1e-6)
    results.append(("A3", ok3))
    print(f"A3 {verdict(ok3)}  (u*2, eta/2) gives the same scores bit for bit: {np.array_equal(rho3a, rho)};"
          f" residual*8 with u*8: bit for bit {np.array_equal(rho3b, rho)}; factor 3: largest relative difference"
          f" {np.max(np.abs(rho3c - rho) / rho):.1e} (float32 rounding only)")

    # A4. the floor 1e-6 never binds for the real network (eta >= 0.01 by construction, network.py:28-31), and
    #     when eta is forced to 0 the floor keeps the scale positive and the score finite.
    eta4 = eta.copy()
    eta4[:5] = 0.0
    rho4, scale4 = scores_of(eta4, unit, residual)
    floor_binds_real = int(np.sum(np.maximum(eta, 1e-6) != eta))
    ok4 = bool(floor_binds_real == 0 and np.all(scale4 > 0) and np.all(np.isfinite(rho4))
               and np.allclose(scale4[:5], 1e-6 * unit, rtol=1e-6))
    results.append(("A4", ok4))
    print(f"A4 {verdict(ok4)}  the floor max(eta, 1e-6) changes {floor_binds_real} of {n} rows when eta >= 0.01;"
          f" with eta forced to 0 on 5 rows the scale is 1e-6*u = {float(scale4[0]):.2e} > 0 and the scores stay"
          f" finite ({float(rho4[:5].max()):.3g} at most). Note: the host's refresh refuses eta <= 0 before this point"
          f" (td3_bc_bca.py:258-262).")

    # A5. validity: NaN residual, infinite eta, u = 0, and a float32 overflow of R make the refresh invalid
    #     (the caller keeps the previous reference); nothing is silently scored.
    cases = [("NaN residual", np.where(np.arange(n) == 3, np.nan, residual), eta, unit),
             ("infinite eta", residual, np.where(np.arange(n) == 3, np.inf, eta).astype(np.float32), unit),
             ("u = 0", residual, eta, np.float32(0.0)),
             ("R overflows float32", np.full(60, 1e30, np.float32), np.full(60, 1e-6, np.float32), np.float32(1e-6))]
    invalid = []
    for name, r_, e_, u_ in cases:
        ref5, valid5, _ = freeze_reference({"w": jnp.ones(2)}, jnp.asarray(u_), jnp.asarray(e_), jnp.asarray(r_),
                                           key, config)
        invalid.append(bool((not valid5) and (not bool(ref5.ready))))
    ok5 = all(invalid)
    results.append(("A5", ok5))
    print(f"A5 {verdict(ok5)}  invalid inputs refused (valid=False, reference not ready): "
          + ", ".join(f"{c[0]}: {v}" for c, v in zip(cases, invalid)))

    # A6. determinism and the float32-then-float64 order of operations: the same inputs give the same array; a
    #     float64-throughout version differs in the last bits on many rows, which is why a reproduction must copy
    #     the arithmetic to be bit for bit.
    rho6, _ = scores_of(eta, unit, residual)
    scale64 = np.maximum(eta.astype(np.float64), 1e-6) * float(unit)
    rho_alt = np.abs(residual.astype(np.float64)) / scale64
    differ = int(np.sum(rho_alt != rho))
    rel_alt = float(np.max(np.abs(rho_alt - rho) / rho))
    cal_a = wbcp.calibrate(rho, np.random.default_rng(words(key)), alpha=ALPHA, beta=BETA, draws=DRAWS)
    cal_b = wbcp.calibrate(rho_alt, np.random.default_rng(words(key)), alpha=ALPHA, beta=BETA, draws=DRAWS)
    ok6 = bool(np.array_equal(rho6, rho) and differ > 0 and rel_alt < 1e-6)
    results.append(("A6", ok6))
    print(f"A6 {verdict(ok6)}  two calls give identical arrays: {np.array_equal(rho6, rho)}; a float64-throughout"
          f" scale changes {differ} of {n} scores by at most {rel_alt:.1e} relative; the thresholds from the two"
          f" versions: {cal_a.threshold:.17g} vs {cal_b.threshold:.17g} (equal in float32: "
          f"{f32eq(cal_a.threshold, cal_b.threshold)})")

    # A7. what M5 reads from M3: the plain 90% quantile lambda_hat is the ceil(0.9 n)-th smallest score, so only the
    #     ranks of the scores set it; a strictly increasing change of every score (x -> x^1.5) leaves the rank
    #     unchanged and lambda_hat follows the transformation exactly.
    k = math.ceil(0.9 * n) - 1
    srt = np.sort(rho)
    cal_pow = wbcp.calibrate(rho ** 1.5, np.random.default_rng(words(key)), alpha=ALPHA, beta=BETA, draws=DRAWS)
    ok7 = bool(cal_a.lambda_hat == srt[k] and cal_pow.lambda_hat == srt[k] ** 1.5
               and cal_pow.lambda_hpd == cal_a.lambda_hpd ** 1.5)
    results.append(("A7", ok7))
    print(f"A7 {verdict(ok7)}  lambda_hat = the {k + 1}-th smallest of {n} scores = {srt[k]:.6f}; after x -> x^1.5"
          f" it is {cal_pow.lambda_hat:.6f} = {srt[k] ** 1.5:.6f} and the credible quantile transforms the same way:"
          f" the threshold depends on the scores only through their order and the values at two ranks")

    # A8. weights never enter M3: the scores are the same array whatever weights M4 later supplies, and with weights
    #     of 1 the weighted call on them equals the unweighted call bit for bit (the weighting test's mark W0 and the
    #     four-learner study's stop rule S2, on synthetic scores).
    cal_w1 = wbcp.calibrate(rho, np.random.default_rng(words(key)), weights=np.ones(n), test_mass=1.0,
                            alpha=ALPHA, beta=BETA, draws=DRAWS)
    ok8 = bool(cal_w1.threshold == cal_a.threshold and cal_w1.lambda_hat == cal_a.lambda_hat
               and cal_w1.lambda_hpd == cal_a.lambda_hpd and cal_w1.n_eff == cal_a.n_eff)
    results.append(("A8", ok8))
    print(f"A8 {verdict(ok8)}  weights of 1 on the same scores give the unweighted threshold bit for bit in float64:"
          f" {cal_w1.threshold:.17g} = {cal_a.threshold:.17g}; the score array itself takes no weight")
    print("Part A:", "all PASS" if all(ok for _, ok in results) else "FAILURES: " + str([n_ for n_, ok in results if not ok]))
    return results, dict(rho=rho, eta=eta, unit=unit, R=float(ref.threshold), lam=float(ref.lambda_hat))


# ============================================================================================ part B: real data
def git_show(repo, ref, path):
    out = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True)
    if out.returncode != 0:
        raise FileNotFoundError(out.stderr.decode(errors="replace").strip())
    return out.stdout


def load_near80(repo, runfiles):
    """[(label, study, world, critic, seed, record dict, scores, eta)] for the ten near-80 version-6 bands."""
    bands = []
    tmp = Path(tempfile.mkdtemp(prefix="m3_near80_"))
    for world in ("stall", "no_stall"):
        for seed in range(5):
            rel = f"{NEAR80_DIR}/{world}/seed{seed}"
            if runfiles:
                d = Path(runfiles) / rel
                rec = json.loads((d / "v6_band.json").read_text())
                z = np.load(d / "v6_bank_scores.npz")
            else:
                rec = z = None
                for ref in NEAR80_BRANCH:
                    try:
                        rec = json.loads(git_show(repo, ref, rel + "/v6_band.json"))
                        p = tmp / f"{world}_{seed}.npz"
                        p.write_bytes(git_show(repo, ref, rel + "/v6_bank_scores.npz"))
                        z = np.load(p)
                        break
                    except FileNotFoundError:
                        continue
                if rec is None:
                    raise SystemExit(f"cannot read {rel} from {NEAR80_BRANCH}; pass --runfiles")
            bands.append((f"near-80 {world:8s} seed {seed}", "near-80", world, "fixed", seed, rec,
                          np.asarray(z["scores"], np.float64), np.asarray(z["eta"], np.float64)))
    return bands


def load_stage2(stage2_dir):
    bands = []
    root = Path(stage2_dir)
    if not root.exists():
        return bands
    for world, critic in (("stall", "fixed"), ("no_stall", "fixed"), ("stall", "host")):
        for seed in range(5):
            d = root / world / critic / f"seed{seed}"
            if not (d / "band.json").exists():
                continue
            rec = json.loads((d / "band.json").read_text())
            z = np.load(d / "bank_scores.npz")
            bands.append((f"stage 2 {world:8s} {critic} seed {seed}", "stage 2", world, critic, seed, rec,
                          np.asarray(z["scores"], np.float64), np.asarray(z["eta"], np.float64)))
    return bands


def part_b(bands):
    print()
    print("=" * 100)
    print("PART B. Real data: the recorded held-out score arrays of every frozen-critic band on record")
    print("=" * 100)
    print("Checks per band: (1) SHA-256 of the stored score array equals the hash in its record; (2) every score is"
          " finite and >= 0, the count equals the record's bank_rows; (3) the smallest eta on record is above 0.01,"
          " so the 1e-6 floor never bound; (4) the plain 90% quantile = the ceil(0.9 n)-th smallest score equals the"
          " recorded lambda_hat (float32); (5) wbcp.calibrate on the stored scores with the refresh's own posterior"
          " stream (this machine's jax) gives the recorded R, lambda_hat, lambda_hpd and n_eff in float32.")
    hdr = (f"{'band':34s} {'n':>4s} {'sha':>4s} {'finite':>6s} {'eta_min':>7s} {'lam_hat':>8s} {'R_rec':>8s} "
           f"{'R_recomp':>9s} {'eq5':>4s} {'<=1':>6s} {'<=lam':>6s} {'<=R':>6s} {'max':>6s}")
    print(hdr)
    rows = []
    for label, study, world, critic, seed, rec, sc, eta in bands:
        u = rec["uniform"]
        n = sc.size
        c1 = sha(sc) == rec["bank_scores_sha256"]
        c2 = bool(np.all(np.isfinite(sc)) and np.all(sc >= 0) and n == u["bank_rows"])
        c3 = bool(eta.min() > 0.01)
        k = math.ceil(0.9 * n) - 1
        lam = float(np.sort(sc)[k])
        c4 = f32eq(lam, u["lambda_hat"])
        key = refresh_key(seed)
        post = words(jax.random.fold_in(key, KEY_POSTERIOR))
        cal = wbcp.calibrate(sc, np.random.default_rng(post), alpha=ALPHA, beta=BETA, draws=DRAWS)
        c5 = bool(f32eq(cal.threshold, u["R"]) and f32eq(cal.lambda_hat, u["lambda_hat"])
                  and f32eq(cal.lambda_hpd, u["lambda_hpd"]) and f32eq(cal.n_eff, rec["refresh"]["n_eff"]))
        c5_f64 = bool(cal.threshold == rec["refresh"]["threshold"])
        share1, sharel, shareR = (float(np.mean(sc <= t)) for t in (1.0, u["lambda_hat"], u["R"]))
        i = int(np.argmax(sc))
        rows.append(dict(label=label, study=study, world=world, critic=critic, seed=seed, n=n, sha_equal=c1,
                         finite=c2, eta_min=float(eta.min()), eta_max=float(eta.max()), floor_never_bound=c3,
                         lambda_hat_rank=lam, lambda_hat_equal=c4, R_record=u["R"], R_recomputed=cal.threshold,
                         recompute_equal_f32=c5, recompute_equal_f64=c5_f64, posterior_words=post.tolist(),
                         share_le_1=share1, share_le_lambda=sharel, share_le_R=shareR, score_max=float(sc.max()),
                         unit=u["unit"], record_threshold_recomputed_equal=u["threshold_recomputed"]["equal"],
                         max_row=i, max_row_eta=float(eta[i]),
                         max_row_abs_residual=float(sc[i] * max(eta[i], 1e-6) * u["unit"]),
                         rows_above_5=int(np.sum(sc > 5)), rows_eta_below_0_9=int(np.sum(eta < 0.9)),
                         scores=sc, eta=eta))
        print(f"{label:34s} {n:4d} {verdict(c1)[:4]:>4s} {verdict(c2)[:4]:>6s} {eta.min():7.3f} {lam:8.4f} "
              f"{u['R']:8.5f} {cal.threshold:9.5f} {verdict(c5)[:4]:>4s} {share1:6.3f} {sharel:6.3f} {shareR:6.3f} "
              f"{sc.max():6.2f}")
    ok = all(r["sha_equal"] and r["finite"] and r["floor_never_bound"] and r["lambda_hat_equal"] and
             r["recompute_equal_f32"] for r in rows)
    print(f"Part B: {len(rows)} bands; all five checks pass on every band: {ok}; share <= 1 ranges"
          f" {min(r['share_le_1'] for r in rows):.3f}-{max(r['share_le_1'] for r in rows):.3f}, share <= R"
          f" {min(r['share_le_R'] for r in rows):.3f}-{max(r['share_le_R'] for r in rows):.3f}; R ranges"
          f" {min(r['R_record'] for r in rows):.3f}-{max(r['R_record'] for r in rows):.3f}; float64 threshold equal"
          f" to the record's refresh block on {sum(r['recompute_equal_f64'] for r in rows)} of {len(rows)}; the"
          f" record's own threshold_recomputed.equal flag is true on"
          f" {sum(bool(r['record_threshold_recomputed_equal']) for r in rows)} of {len(rows)}")
    print()
    print("B2. The largest score of each band (the exceptional rows): |y - q| recovered as score * eta * u from the"
          " stored arrays and the recorded unit; a typical row has |y - q| near 0.1 (the medians are 0.089-0.106,"
          " part C).")
    print(f"{'band':34s} {'max score':>9s} {'eta there':>9s} {'|y-q| there':>11s} {'eta_min':>7s} {'rows>5':>6s} {'eta<0.9':>7s}")
    for r in rows:
        print(f"{r['label']:34s} {r['score_max']:9.2f} {r['max_row_eta']:9.3f} {r['max_row_abs_residual']:11.3f}"
              f" {r['eta_min']:7.3f} {r['rows_above_5']:6d} {r['rows_eta_below_0_9']:7d}")
    return rows, ok


# ============================================================================================ part C: pinned report
def part_c(pinned_path, rows_b):
    print()
    print("=" * 100)
    print("PART C. The pinned-environment recompute from the saved critic, eta_f, u_f and the refresh key"
          " (scripts/M3_recompute_pinned.py through the WSL bridge)")
    print("=" * 100)
    p = Path(pinned_path)
    if not p.exists():
        print(f"no report at {p}; run the bridge job first (see M3_recompute_pinned.sh). Skipped.")
        return None
    rep = json.loads(p.read_text())
    env = rep["environment"]
    print(f"environment: Python {env['python']}, numpy {env['numpy']}, jax {env['jax']} ({env['platform']}),"
          f" repository commit {env.get('repo_commit')}; {len(rep['bands'])} bands, {len(rep['errors'])} errors,"
          f" {rep['seconds']:.0f} s")
    print("Columns: (a) the study's own score function from the saved critic + band equals the stored array / the"
          " stored eta; (b) the scores by hand (float32 scale, float64 division) equal (a); 'f64 rows' = rows that a"
          " float64-throughout scale would change, 'max rel' = by how much; (c) freeze_reference on the recomputed"
          " pieces equals the record's R, lambda_hat, lambda_hpd, n_eff in float32 and all float64 diagnostics;"
          " (d) wbcp.calibrate on the STORED scores equals the record's float64 threshold; 'rows=rec' = the rebuilt"
          " held-out rows carry the record's SHA-256.")
    print(f"{'band':34s} {'n':>4s} {'(a)scores':>9s} {'(a)eta':>6s} {'(b)hand':>7s} {'f64 rows':>8s} {'max rel':>8s} "
          f"{'(c)R':>4s} {'lam':>4s} {'hpd':>4s} {'neff':>4s} {'f64all':>6s} {'(d)stored':>9s} {'rows=rec':>8s}")
    words_match = []
    by_label = {(r["study"], r["world"], r["critic"], r["seed"]): r for r in rows_b}
    all_ok = True
    medians = []
    for b in rep["bands"]:
        label = f"{b['study'].split()[0]:8s} {b['world']:8s} {b['critic']} seed {b['seed']}"
        d = b["d_from_stored_scores"]
        if b.get("full_recompute"):
            c = b["c_bit_for_bit"]
            prep_eq = all(b["prepared_rows_equal_record"].values())
            f64 = all(c[k] for k in ("f64_threshold", "f64_lambda_hat", "f64_lambda_hpd", "f64_sigma_post",
                                      "recomputed_scores_threshold_f64"))
            ok = (b["a_scores_equal_stored"] and b["a_eta_equal_stored"] and b["b_hand_equals_study_path"]
                  and all(c[k] for k in ("R", "lambda_hat", "lambda_hpd", "n_eff")) and f64
                  and d["equal_f64_threshold"] and prep_eq)
            medians.append(b["summary"]["median_abs_residual"])
            print(f"{label:34s} {b['n']:4d} {str(b['a_scores_equal_stored']):>9s} {str(b['a_eta_equal_stored']):>6s}"
                  f" {str(b['b_hand_equals_study_path']):>7s} {b['b_alt_scale64_rows_differ']:8d}"
                  f" {b['b_alt_scale64_max_rel']:8.1e} {str(c['R'])[:4]:>4s} {str(c['lambda_hat'])[:4]:>4s}"
                  f" {str(c['lambda_hpd'])[:4]:>4s} {str(c['n_eff'])[:4]:>4s} {str(f64):>6s}"
                  f" {str(d['equal_f64_threshold']):>9s} {str(prep_eq):>8s}")
        else:
            ok = d["equal_f64_threshold"] and d["equal_f32_R"]
            print(f"{label:34s} {b['n']:4d} {'(no critic loaded: (d) only)':>52s} {str(d['equal_f64_threshold']):>9s}")
        all_ok = all_ok and ok
        key = ("near-80" if b["study"].startswith("near") else "stage 2", b["world"], b["critic"], b["seed"])
        rb = by_label.get(key)
        if rb is not None:
            words_match.append(rb["posterior_words"] == b["posterior_words"])
    print(f"Part C: every check passes on every band: {all_ok}; this machine's jax ({jax.__version__}) produced the"
          f" same posterior stream words as the pinned jax on {sum(words_match)} of {len(words_match)} bands compared;"
          f" median |y - q| on the held-out rows {min(medians):.3f}-{max(medians):.3f} critic units over the"
          f" {len(medians)} fully recomputed bands")
    if rep["errors"]:
        for e in rep["errors"]:
            print("ERROR", e["label"], e["error"])
    return rep, all_ok


# ============================================================================================ part D: by current
def part_d(rows_path, rows_b):
    print()
    print("=" * 100)
    print("PART D. The recorded scores read by water current (held-out rows rebuilt in the pinned environment,"
          " scripts/M3_bank_rows_pinned.py; exploratory description, no claim was pre-registered)")
    print("=" * 100)
    p = Path(rows_path)
    if not p.exists():
        print(f"no rows file at {p}; run the bridge job first (see M3_bank_rows_pinned.sh). Skipped.")
        return None
    z = np.load(p)
    meta = json.loads(p.with_suffix(".json").read_text()) if p.with_suffix(".json").exists() else {}
    checks = meta.get("checks", {})
    print("Within each band the held-out rows are split into thirds by their current (low / middle / high third)."
          " Shown per third: the mean score, the plain 90% quantile of the scores, and the share of rows the band"
          " misses (score > R). 'Ratio' = the 90% quantile in the high third over the low third.")
    print(f"{'band':34s} {'n':>4s} {'c range':>13s} {'mean L/M/H':>20s} {'q90 L/M/H':>20s} {'miss L/M/H':>20s} {'ratio':>6s} {'>80% thr':>8s}")
    out = []
    for r in rows_b:
        study = "near80" if r["study"] == "near-80" else "stage2"
        label = f"{study}_{r['world']}_{r['critic']}_seed{r['seed']}"
        if f"current_{label}" not in z.files:
            continue
        c = np.asarray(z[f"current_{label}"], np.float64)
        thr = np.asarray(z[f"throttle_{label}"], np.float64)
        sc = r["scores"]
        if c.size != sc.size:
            print(f"{r['label']:34s} rows file has {c.size} rows, the score array {sc.size}: skipped")
            continue
        chk = checks.get(label, {})
        sha_ok = all(chk.get("sha_equal", {}).values()) if chk else None
        lo, hi = np.quantile(c, [1 / 3, 2 / 3])
        parts = [sc[c <= lo], sc[(c > lo) & (c <= hi)], sc[c > hi]]
        mean = [float(np.mean(s)) for s in parts]
        q90 = [float(np.quantile(s, 0.9)) for s in parts]
        miss = [float(np.mean(s > r["R_record"])) for s in parts]
        ratio = q90[2] / q90[0]
        out.append(dict(label=r["label"], study=r["study"], world=r["world"], seed=r["seed"], n=int(sc.size),
                        c_min=float(c.min()), c_max=float(c.max()), mean=mean, q90=q90, miss=miss, ratio=ratio,
                        share_past_80=float(np.mean(thr > 0.8)), rows_sha_equal=sha_ok,
                        corr_current=float(np.corrcoef(c, sc)[0, 1])))
        print(f"{r['label']:34s} {sc.size:4d} {c.min():6.2f}..{c.max():5.2f} {mean[0]:6.3f}/{mean[1]:6.3f}/{mean[2]:6.3f}"
              f" {q90[0]:6.3f}/{q90[1]:6.3f}/{q90[2]:6.3f} {miss[0]:6.3f}/{miss[1]:6.3f}/{miss[2]:6.3f} {ratio:6.2f}"
              f" {np.mean(thr > 0.8):8.3f}")
    if not out:
        print("no band matched the rows file")
        return None
    print()
    print("D2. Means over seeds 0-4 by group:")
    print(f"{'group':26s} {'bands':>5s} {'mean L/M/H':>20s} {'q90 L/M/H':>20s} {'miss L/M/H':>20s} {'ratio':>6s} {'ratio>1':>7s} {'r(c,score)':>10s}")
    groups = {}
    for o in out:
        groups.setdefault((o["study"], o["world"]), []).append(o)
    for (study, world), members in groups.items():
        m = np.mean([o["mean"] for o in members], axis=0)
        q = np.mean([o["q90"] for o in members], axis=0)
        ms = np.mean([o["miss"] for o in members], axis=0)
        ratios = [o["ratio"] for o in members]
        corr = np.mean([o["corr_current"] for o in members])
        print(f"{study + ' ' + world:26s} {len(members):5d} {m[0]:6.3f}/{m[1]:6.3f}/{m[2]:6.3f}"
              f" {q[0]:6.3f}/{q[1]:6.3f}/{q[2]:6.3f} {ms[0]:6.3f}/{ms[1]:6.3f}/{ms[2]:6.3f} {np.mean(ratios):6.2f}"
              f" {sum(x > 1 for x in ratios):3d}/{len(ratios):<3d} {corr:10.3f}")
    print(f"Part D: {len(out)} bands; the rebuilt held-out rows carry the record's SHA-256 on"
          f" {sum(bool(o['rows_sha_equal']) for o in out)} of {len(out)}; the 90% quantile is higher in the high-current"
          f" third than in the low-current third on {sum(o['ratio'] > 1 for o in out)} of {len(out)} bands"
          f" (ratio {min(o['ratio'] for o in out):.2f}-{max(o['ratio'] for o in out):.2f}); share of held-out rows"
          f" past 80% throttle {min(o['share_past_80'] for o in out):.3f}-{max(o['share_past_80'] for o in out):.3f}")
    return out


# ============================================================================================ the figure
def figure(path, bands, rows_b, rows_d):
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 9.5
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    seq = ("#9ecae1", "#4292c6", "#08306b")       # low / middle / high current: ordered by lightness
    ncol = 3 if rows_d else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol + 0.8, 4.4),
                             gridspec_kw=dict(width_ratios=[1.0, 1.3, 1.0][:ncol]))
    ax1, ax2 = axes[0], axes[1]
    # panel A: the scores of near-80, stalling world, seed 0
    label, study, world, critic, seed, rec, sc, eta = bands[0]
    u = rec["uniform"]
    edges = np.arange(0, 4.0001, 0.1)
    ax1.hist(np.minimum(sc, 3.95), bins=edges, color=blue, edgecolor="white", linewidth=0.5)
    lines = ((1.0, ":", f"score 1 (u*eta alone): {np.mean(sc <= 1):.1%} of rows at or below"),
             (u["lambda_hat"], "--", f"plain 90% quantile {u['lambda_hat']:.3f}: {np.mean(sc <= u['lambda_hat']):.1%}"),
             (u["R"], "-", f"threshold R {u['R']:.3f}: {np.mean(sc <= u['R']):.1%}"))
    for x, ls, name in lines:
        ax1.axvline(x, ymax=0.74, color=ink, linestyle=ls, linewidth=1.3, label=name)
    ax1.set_xlabel(f"score = |y - q| / (u * eta), dimensionless\n(last bar: every score above 3.9; largest {sc.max():.1f})")
    ax1.set_ylabel("number of held-out rows")
    ax1.set_title(f"A. The {sc.size} held-out scores of one band\n(near-80, stalling world, seed 0; recorded array)", fontsize=10)
    ax1.legend(loc="upper right", frameon=False, fontsize=8.2)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    # panel B: coverage of the held-out rows by u*eta alone and by R*u*eta, every band on record
    groups = [("near-80", "stall", "fixed"), ("near-80", "no_stall", "fixed"), ("stage 2", "stall", "fixed"),
              ("stage 2", "no_stall", "fixed"), ("stage 2", "stall", "host")]
    names = {("near-80", "stall", "fixed"): "near-80\nstalling", ("near-80", "no_stall", "fixed"): "near-80\nno stall",
             ("stage 2", "stall", "fixed"): "stage 2\nstalling", ("stage 2", "no_stall", "fixed"): "stage 2\nno stall",
             ("stage 2", "stall", "host"): "stage 2\nhost critic"}
    x, s1, sR, ticks, tlabels, bounds = [], [], [], [], [], []
    pos = 0
    for grp in groups:
        members = [r for r in rows_b if (r["study"], r["world"], r["critic"]) == grp]
        if not members:
            continue
        start = pos
        for r in sorted(members, key=lambda r: r["seed"]):
            x.append(pos)
            s1.append(r["share_le_1"])
            sR.append(r["share_le_R"])
            pos += 1
        ticks.append((start + pos - 1) / 2)
        tlabels.append(names[grp])
        bounds.append(pos - 0.5)
        pos += 1
    ax2.axhline(0.90, color=muted, linestyle="--", linewidth=1.0)
    ax2.text(x[-1] + 0.6, 0.902, "target 0.90", color=muted, fontsize=8.5, va="bottom", ha="right")
    for b in bounds[:-1]:
        ax2.axvline(b + 0.5, color="#dddddd", linewidth=0.8)
    ax2.scatter(x, s1, s=42, marker="o", color=blue, label="u*eta alone (score <= 1)", zorder=3)
    ax2.scatter(x, sR, s=42, marker="s", color=orange, label="R*u*eta, the band (score <= R)", zorder=3)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tlabels, fontsize=8.5)
    ax2.set_ylim(0.70, 0.96)
    ax2.set_xlim(-0.8, x[-1] + 0.8)
    ax2.set_ylabel("share of held-out rows covered")
    ax2.set_xlabel("band (seeds 0-4 within each group, left to right)")
    ax2.set_title(f"B. Every frozen-critic band on record ({len(x)} bands):\neta alone under-covers; the threshold R lifts the band past 0.90", fontsize=10)
    ax2.legend(loc="lower right", frameon=False, fontsize=8.5)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    # panel C: the 90% quantile of the scores by current third, per group
    if rows_d:
        ax3 = axes[2]
        order = [("near-80", "stall"), ("near-80", "no_stall"), ("stage 2", "stall"), ("stage 2", "no_stall")]
        gnames = {("near-80", "stall"): "near-80\nstalling", ("near-80", "no_stall"): "near-80\nno stall",
                  ("stage 2", "stall"): "stage 2\nstalling", ("stage 2", "no_stall"): "stage 2\nno stall"}
        width = 0.26
        xt, xl = [], []
        for gi, key in enumerate(order):
            members = [o for o in rows_d if (o["study"], o["world"]) == key]
            if not members:
                continue
            for ti, (col, tname) in enumerate(zip(seq, ("low current third", "middle third", "high current third"))):
                vals = [o["q90"][ti] for o in members]
                xpos = gi + (ti - 1) * width
                ax3.bar(xpos, np.mean(vals), width=width * 0.92, color=col, edgecolor="white", linewidth=0.5,
                        label=tname if gi == 0 else None, zorder=2)
                ax3.scatter(np.full(len(vals), xpos) + np.linspace(-0.05, 0.05, len(vals)), vals, s=12, color=ink,
                            zorder=3)
            xt.append(gi)
            xl.append(gnames[key])
        ax3.axhline(1.0, color=muted, linestyle=":", linewidth=1.0)
        ax3.text(len(xt) - 0.55, 1.01, "score 1", color=muted, fontsize=8.5, ha="right", va="bottom")
        ax3.set_xticks(xt)
        ax3.set_xticklabels(xl, fontsize=8.5)
        ax3.set_ylabel("plain 90% quantile of the scores in that third")
        ax3.set_xlabel("band group (bars: mean of seeds 0-4; dots: seeds)")
        ax3.set_title("C. The same scores by water current:\nlarger in the high-current third of the held-out rows", fontsize=10)
        ax3.legend(loc="upper left", frameon=False, fontsize=8.2)
        ax3.set_ylim(0, max(max(o["q90"]) for o in rows_d) * 1.25)
        for s in ("top", "right"):
            ax3.spines[s].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    print(f"figure written: {path}")


def main():
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(here.parents[3]))
    ap.add_argument("--runfiles", default=None)
    ap.add_argument("--stage2", default=STAGE2_WSL)
    ap.add_argument("--pinned", default=str(here.parents[1] / "data" / "M3_pinned_recompute.json"))
    ap.add_argument("--rows", default=str(here.parents[1] / "data" / "M3_bank_rows.npz"))
    ap.add_argument("--figure", default=str(here.parents[1] / "figures" / "M3.svg"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    print(f"Python {sys.version.split()[0]}, numpy {np.__version__}, jax {jax.__version__}; repository {args.repo}")
    a_results, _ = part_a()
    bands = load_near80(args.repo, args.runfiles) + load_stage2(args.stage2)
    rows_b, ok_b = part_b(bands)
    c = part_c(args.pinned, rows_b)
    rows_d = part_d(args.rows, rows_b)
    if not args.no_figure:
        figure(args.figure, bands, rows_b, rows_d)
    print()
    print("SUMMARY: part A", "all PASS" if all(ok for _, ok in a_results) else "FAIL", "| part B", verdict(ok_b),
          f"on {len(rows_b)} bands", "| part C", ("skipped" if c is None else verdict(c[1])),
          "| part D", ("skipped" if rows_d is None else f"{len(rows_d)} bands described"))


if __name__ == "__main__":
    main()
