"""Recompute BCA's threshold from recorded held-out scores, changing one input at a time (the penalty investigation,
docs/bca_mechanics/PENALTY_EVIDENCE.md, section "The mechanism"). Nothing is trained: the error-size network and the
unit are untouched, the scores are read exactly as the refresh computed them, and only the weights, the test mass and
the random stream of module M5 (calibration/wbcp.calibrate) are varied.

Inputs, read only (either tree works; the near-80 files are also on branch claude/near80-run-files):
  <runs>/near80/learners/{near80,random_rows}/{stall,no_stall}/seed<K>/{v6_bank_scores.npz, v6_band.json}
  <runs>/stage2/detect/uniform/{stall,no_stall}/fixed/seed<K>/{bank_scores.npz, band.json}
Usage (the pinned environment, numpy 2.2.6 and jax 0.6.2, reproduces the recorded stream exactly):
  PYTHONPATH=<repository root> python docs/bca_mechanics/scripts/penalty_recompute.py \\
      --runs /home/user/bca/runs/shift_explore --out penalty_recompute.json
Needs numpy and jax (jax only to rebuild the refresh's random key as the band code did).

What is computed for every recorded bank of scores (n = 800-850 rows):
  0. the recorded threshold R, lambda_hat and lambda_hpd (band.json) and the recorded recomputation check;
  A. weights of 1 and test mass 1 with the refresh's own random stream: must equal the record (module M4/M5, the
     unit check of the weighting test's mark W0);
  B. weights of 1 and the strong tilt's test mass alone (the "unseen test point" atom made heavier): the test-atom
     part of the weighted threshold;
  C. the strong tilt's weights given to rows AT RANDOM (20 random currents per bank, no link between a row's weight
     and its score) with the strong tilt's test mass: the smaller-effective-sample part (Kish size about 0.35 n) plus
     the test atom, with no reweighting toward high-error rows (the four-learner study's shuffled-weights placebo,
     restated on recorded scores);
  C-mild. the same with the mild tilt's weights;
  D. the strong tilt's weights SORTED onto the scores, heaviest on the largest scores (the most the reweighting part
     can add) and heaviest on the smallest scores (the most it can take away);
  E. the classifier's recorded fitted slopes under the no-shift law (25 seed-quality values from the weighting test's
     mark W4), applied as exp(slope * c) to random currents: the move an "estimated no-shift" weight makes when the
     scores do not depend on the current;
  F. weights of 1 on random subsets of 100 and 300 rows: the Bayesian margin lambda_hpd / lambda_hat as the bank shrinks.

The tilts are the weighting test's (experiments/shift_iw/world.py: the training law of the current has exponent -2 on
[0.1, 1.1]; the mild tilt exponent 0, the strong tilt exponent +2; w(c) = p_test(c) / p_train(c)); the test mass is
E_test[w] = E_train[w^2], by quadrature. The currents drawn here are synthetic (the bank rows' own currents are not
stored next to the scores), so cases B to F describe the arithmetic of the threshold on real scores, not a recorded
study result; case A is the recorded reproduction.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

try:
    from calibration import wbcp
except ImportError as error:
    sys.exit(f"set PYTHONPATH to the repository root (calibration/wbcp.py): {error}")

ALPHA, BETA, DRAWS = 0.1, 0.95, 1000
C_LO, C_HI = 0.1, 1.1
LAM_TRAIN = -2.0
TILTS = {"mild": 0.0, "strong": 2.0}
TAG_BAND, BAND_FIT_STEPS, POSTERIOR_FOLD = 41, 20_000, 1347375956
# W4.classifier_slope_on_c.none.fitted, outputs/shift_iw/v5/<quality>/report.json (branch main), seeds 0-4
NONE_SLOPES = {
    "random": [0.1867, -0.1197, -0.2822, 0.0847, 0.39],
    "medium-replay": [0.2866, -0.0195, -0.117, 0.0984, 0.4192],
    "medium": [0.1083, -0.0807, -0.2802, 0.1444, 0.2917],
    "medium-expert": [0.2842, 0.113, 0.0048, 0.0964, 0.3252],
    "expert": [0.2019, -0.2267, 0.0055, 0.1473, 0.3665],
}


def M(lam):
    return (C_HI - C_LO) if lam == 0 else math.exp(lam * C_LO) * math.expm1(lam * (C_HI - C_LO)) / lam


def sample_c(rng, n, lam=LAM_TRAIN):
    u = rng.random(n)
    if lam == 0:
        return C_LO + u * (C_HI - C_LO)
    return C_LO + np.log1p(u * math.expm1(lam * (C_HI - C_LO))) / lam


def tilt_weight(c, lam):
    """w(c) = p_test(c) / p_train(c) for a test law with exponent lam (experiments/shift_iw/world.ratio)."""
    return (M(LAM_TRAIN) / M(lam)) * np.exp((lam - LAM_TRAIN) * np.asarray(c))


def test_mass(weight_fn):
    """E_test[w] = E_train[w^2] / E_train[w], by quadrature over the training law of the current."""
    cg = np.linspace(C_LO, C_HI, 40001)
    p_train = np.exp(LAM_TRAIN * cg) / M(LAM_TRAIN)
    w = weight_fn(cg)
    return float(np.trapezoid(w * w * p_train, cg) / np.trapezoid(w * p_train, cg))


def refresh_stream(rep, steps=BAND_FIT_STEPS):
    """The posterior's NumPy generator as the band code built it: band_rows_and_keys (stage 2's band.py:243-249,
    reused by the near-80 run) gives refresh_key = fold_in(PRNGKey(default_rng([20261015, rep, 41, 1]).integers(2^31 - 1)),
    steps); freeze_reference / uniform_threshold_check then seed NumPy with the words of fold_in(refresh_key, 1347375956)."""
    import jax
    seed = int(np.random.default_rng([20261015, int(rep), TAG_BAND, 1]).integers(2 ** 31 - 1))
    refresh_key = jax.random.fold_in(jax.random.PRNGKey(seed), int(steps))
    words = np.asarray(jax.random.fold_in(refresh_key, POSTERIOR_FOLD), np.uint32).ravel()
    return lambda: np.random.default_rng(words)


def cal(scores, stream, weights=None, t=None):
    if weights is None:
        return wbcp.calibrate(scores, stream(), alpha=ALPHA, beta=BETA, draws=DRAWS)
    return wbcp.calibrate(scores, stream(), np.asarray(weights, np.float64), float(t), alpha=ALPHA, beta=BETA, draws=DRAWS)


def pack(r):
    return dict(R=r.threshold, lambda_hat=r.lambda_hat, lambda_hpd=r.lambda_hpd, n_eff=r.n_eff,
                margin=(r.lambda_hpd / r.lambda_hat if r.lambda_hat > 0 and math.isfinite(r.lambda_hpd) else None))


def find_banks(runs):
    banks = []
    for path in sorted(glob.glob(os.path.join(runs, "near80", "learners", "*", "*", "seed*", "v6_bank_scores.npz"))):
        d = os.path.dirname(path)
        parts = os.path.normpath(d).split(os.sep)
        rec = json.load(open(os.path.join(d, "v6_band.json"), encoding="utf8"))
        banks.append(dict(study="near-80 version 6", setting=parts[-3], world=parts[-2], seed=int(parts[-1][4:]),
                          scores_file=path, record_file=os.path.join(d, "v6_band.json"), rep=rec["data"]["rep"],
                          steps=rec["steps"], uniform=rec["uniform"], refresh=rec["refresh"]))
    for path in sorted(glob.glob(os.path.join(runs, "stage2", "detect", "uniform", "*", "fixed", "seed*", "bank_scores.npz"))):
        d = os.path.dirname(path)
        parts = os.path.normpath(d).split(os.sep)
        rec = json.load(open(os.path.join(d, "band.json"), encoding="utf8"))
        banks.append(dict(study="stage 2 uniform", setting="main", world=parts[-3], seed=int(parts[-1][4:]),
                          scores_file=path, record_file=os.path.join(d, "band.json"), rep=rec["data"]["rep"],
                          steps=rec["steps"], uniform=rec["uniform"], refresh=rec["refresh"]))
    return banks


def analyse(bank, shuffles=20):
    scores = np.load(bank["scores_file"])["scores"].astype(np.float64)
    n = scores.size
    stream = refresh_stream(bank["rep"], bank["steps"])
    rec = bank["uniform"]
    out = dict(study=bank["study"], setting=bank["setting"], world=bank["world"], seed=bank["seed"], n=int(n),
               scores_file=bank["scores_file"], record_file=bank["record_file"],
               recorded=dict(R=rec["R"], lambda_hat=rec["lambda_hat"], lambda_hpd=rec["lambda_hpd"],
                             recomputed=rec["threshold_recomputed"]["recomputed"],
                             recomputed_equal=rec["threshold_recomputed"]["equal"]))
    # A. the recorded reproduction: weights of 1, test mass 1, the refresh's stream
    a = cal(scores, stream)
    a1 = cal(scores, stream, np.ones(n), 1.0)
    out["A_ones"] = dict(pack(a), equals_recorded_recompute=bool(a.threshold == rec["threshold_recomputed"]["recomputed"]),
                         equals_recorded_R_float32=bool(np.float32(a.threshold) == np.float32(rec["R"])),
                         explicit_ones_bit_equal=bool(a.threshold == a1.threshold and a.lambda_hat == a1.lambda_hat
                                                      and a.lambda_hpd == a1.lambda_hpd
                                                      and np.array_equal(a.crossings, a1.crossings)))
    R_u, lh_u = a.threshold, a.lambda_hat
    # B. the test atom alone
    masses = {k: test_mass(lambda c, lam=lam: tilt_weight(c, lam)) for k, lam in TILTS.items()}
    out["test_mass"] = masses
    b = cal(scores, stream, np.ones(n), masses["strong"])
    out["B_test_mass_only"] = dict(pack(b), ratio=b.threshold / R_u)
    # C. shuffled tilt weights (no link to the scores)
    for name, lam in TILTS.items():
        rows = []
        for k in range(shuffles):
            c = sample_c(np.random.default_rng([20261020, 5, bank["seed"], k]), n)
            w = tilt_weight(c, lam)
            r = cal(scores, stream, w, masses[name])
            rows.append(dict(pack(r), ratio=r.threshold / R_u, lambda_hat_ratio=r.lambda_hat / lh_u,
                             kish_share=r.n_eff / n))
        rat = np.array([x["ratio"] for x in rows])
        out[f"C_shuffled_{name}"] = dict(ratio_mean=float(rat.mean()), ratio_min=float(rat.min()), ratio_max=float(rat.max()),
                                         above_one=int((rat > 1).sum()), below_one=int((rat < 1).sum()), equal_one=int((rat == 1).sum()),
                                         lambda_hat_ratio_mean=float(np.mean([x["lambda_hat_ratio"] for x in rows])),
                                         margin_mean=float(np.mean([x["margin"] for x in rows])),
                                         kish_share_mean=float(np.mean([x["kish_share"] for x in rows])), draws=rows)
    # D. sorted strong weights: the bounds of the reweighting part
    c_sorted = np.sort(sample_c(np.random.default_rng([20261020, 6, bank["seed"]]), n))
    w_sorted = tilt_weight(c_sorted, TILTS["strong"])          # ascending in c, so ascending in w
    order = np.argsort(scores, kind="stable")
    w_up, w_down = np.empty(n), np.empty(n)
    w_up[order] = w_sorted                                      # heaviest weights on the largest scores
    w_down[order] = w_sorted[::-1]                              # heaviest weights on the smallest scores
    up, down = cal(scores, stream, w_up, masses["strong"]), cal(scores, stream, w_down, masses["strong"])
    out["D_sorted_strong"] = dict(heaviest_on_largest_scores=dict(pack(up), ratio=up.threshold / R_u),
                                  heaviest_on_smallest_scores=dict(pack(down), ratio=down.threshold / R_u))
    # E. the classifier's no-shift slopes on random currents
    rows = []
    for quality, slopes in NONE_SLOPES.items():
        for j, slope in enumerate(slopes):
            for k in range(4):
                c = sample_c(np.random.default_rng([20261020, 7, bank["seed"], j, k]), n)
                w = np.exp(slope * c)
                t = test_mass(lambda cg, s=slope: np.exp(s * cg))
                r = cal(scores, stream, w, t)
                rows.append(dict(quality=quality, seed=j, slope=slope, ratio=r.threshold / R_u, kish_share=r.n_eff / n))
    rat = np.array([x["ratio"] for x in rows])
    out["E_none_slopes"] = dict(ratio_mean=float(rat.mean()), ratio_min=float(rat.min()), ratio_max=float(rat.max()),
                                beyond_5pct=int((np.abs(rat - 1) > 0.05).sum()), cases=len(rows),
                                kish_share_min=float(min(x["kish_share"] for x in rows)), draws=rows)
    # F. the margin as the bank shrinks
    out["F_subsets"] = {}
    for m in (100, 300):
        margins, finite = [], 0
        for k in range(20):
            idx = np.random.default_rng([20261020, 8, bank["seed"], m, k]).choice(n, m, replace=False)
            r = cal(scores[idx], stream)
            finite += math.isfinite(r.threshold)
            if math.isfinite(r.lambda_hpd) and r.lambda_hat > 0:
                margins.append(r.lambda_hpd / r.lambda_hat)
        out["F_subsets"][str(m)] = dict(margin_mean=float(np.mean(margins)) if margins else None, finite=finite, of=20)
    out["margin_recorded_n"] = a.lambda_hpd / a.lambda_hat
    return out


def table(results):
    hdr = (f"{'study':18} {'setting':11} {'world':8} {'seed':>4} {'n':>4} {'R rec':>7} {'A=rec':>5} {'B t-only':>8} "
           f"{'C strong':>8} {'[min,max]':>15} {'lam^ x':>7} {'Kish':>5} {'marg u':>6} {'marg w':>6} "
           f"{'D up':>6} {'D down':>6} {'C mild':>6} {'E none':>14} {'marg100':>7} {'marg300':>7}")
    print(hdr)
    print("-" * len(hdr))
    for o in results:
        cs, cm, e = o["C_shuffled_strong"], o["C_shuffled_mild"], o["E_none_slopes"]
        m100 = o["F_subsets"]["100"]["margin_mean"] or float("nan")
        m300 = o["F_subsets"]["300"]["margin_mean"] or float("nan")
        print(f"{o['study']:18} {o['setting']:11} {o['world']:8} {o['seed']:>4} {o['n']:>4} {o['recorded']['R']:7.4f} "
              f"{str(o['A_ones']['equals_recorded_recompute'])[:5]:>5} {o['B_test_mass_only']['ratio']:8.4f} "
              f"{cs['ratio_mean']:8.4f} [{cs['ratio_min']:6.3f},{cs['ratio_max']:6.3f}] {cs['lambda_hat_ratio_mean']:7.4f} "
              f"{cs['kish_share_mean']:5.3f} {o['margin_recorded_n']:6.3f} {cs['margin_mean']:6.3f} "
              f"{o['D_sorted_strong']['heaviest_on_largest_scores']['ratio']:6.3f} "
              f"{o['D_sorted_strong']['heaviest_on_smallest_scores']['ratio']:6.3f} {cm['ratio_mean']:6.3f} "
              f"[{e['ratio_min']:5.3f},{e['ratio_max']:5.3f}] {m100:7.3f} {m300:7.3f}")


def summary(results):
    n = len(results)
    a_ok = sum(o["A_ones"]["equals_recorded_recompute"] for o in results)
    a32 = sum(o["A_ones"]["equals_recorded_R_float32"] for o in results)
    ones = sum(o["A_ones"]["explicit_ones_bit_equal"] for o in results)
    cs_up = sum(o["C_shuffled_strong"]["above_one"] for o in results)
    cs_dn = sum(o["C_shuffled_strong"]["below_one"] for o in results)
    cs_eq = sum(o["C_shuffled_strong"]["equal_one"] for o in results)
    e_beyond = sum(o["E_none_slopes"]["beyond_5pct"] for o in results)
    e_cases = sum(o["E_none_slopes"]["cases"] for o in results)
    s = dict(banks=n, A_equals_recorded_recompute=a_ok, A_equals_recorded_R_float32=a32, A_explicit_ones_bit_equal=ones,
             B_test_mass_only_ratio_mean=float(np.mean([o["B_test_mass_only"]["ratio"] for o in results])),
             C_strong_ratio_mean=float(np.mean([o["C_shuffled_strong"]["ratio_mean"] for o in results])),
             C_strong_ratio_min=float(min(o["C_shuffled_strong"]["ratio_min"] for o in results)),
             C_strong_ratio_max=float(max(o["C_shuffled_strong"]["ratio_max"] for o in results)),
             C_strong_draws_above_one=cs_up, C_strong_draws_below_one=cs_dn, C_strong_draws_equal_one=cs_eq,
             C_strong_lambda_hat_ratio_mean=float(np.mean([o["C_shuffled_strong"]["lambda_hat_ratio_mean"] for o in results])),
             C_strong_margin_mean=float(np.mean([o["C_shuffled_strong"]["margin_mean"] for o in results])),
             margin_unweighted_mean=float(np.mean([o["margin_recorded_n"] for o in results])),
             C_strong_kish_share_mean=float(np.mean([o["C_shuffled_strong"]["kish_share_mean"] for o in results])),
             C_mild_ratio_mean=float(np.mean([o["C_shuffled_mild"]["ratio_mean"] for o in results])),
             D_up_ratio_mean=float(np.mean([o["D_sorted_strong"]["heaviest_on_largest_scores"]["ratio"] for o in results])),
             D_down_ratio_mean=float(np.mean([o["D_sorted_strong"]["heaviest_on_smallest_scores"]["ratio"] for o in results])),
             E_none_ratio_min=float(min(o["E_none_slopes"]["ratio_min"] for o in results)),
             E_none_ratio_max=float(max(o["E_none_slopes"]["ratio_max"] for o in results)),
             E_none_beyond_5pct=e_beyond, E_none_cases=e_cases,
             F_margin_100_mean=float(np.mean([o["F_subsets"]["100"]["margin_mean"] for o in results if o["F_subsets"]["100"]["margin_mean"]])),
             F_margin_300_mean=float(np.mean([o["F_subsets"]["300"]["margin_mean"] for o in results if o["F_subsets"]["300"]["margin_mean"]])),
             test_mass=results[0]["test_mass"] if results else None)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder holding near80/ and stage2/ (for example runs/shift_explore)")
    ap.add_argument("--out", default=None, help="write the full results as JSON here")
    ap.add_argument("--shuffles", type=int, default=20)
    args = ap.parse_args()
    banks = find_banks(args.runs)
    if not banks:
        sys.exit(f"no recorded banks under {args.runs}")
    print(f"numpy {np.__version__}; calibration/wbcp.py from {wbcp.__file__}; {len(banks)} recorded banks")
    results = [analyse(b, args.shuffles) for b in banks]
    table(results)
    s = summary(results)
    print()
    print("summary:", json.dumps(s, indent=1))
    if args.out:
        slim = []
        for o in results:
            o2 = dict(o)
            for k in ("C_shuffled_strong", "C_shuffled_mild", "E_none_slopes"):
                o2[k] = {kk: vv for kk, vv in o[k].items() if kk != "draws"}
            slim.append(o2)
        json.dump(dict(summary=s, banks=slim), open(args.out, "w", encoding="utf8"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
