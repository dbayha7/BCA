"""The mechanism behind the weighted threshold, tested with the banks' RECORDED water currents (PENALTY_EVIDENCE.md,
section 5, case G). Nothing is trained: the held-out scores are the recorded ones, the error-size network and the unit
are untouched, and only the weights handed to module M5 (calibration/wbcp.calibrate) change.

Question. When the weights point at the high-current rows, the weighted threshold rises above the unweighted one.
How much of that rise is the TILT (the heavy rows are the rows with the larger scores) and how much is the smaller
effective sample plus the heavier test atom (which would raise the threshold even if the weights had nothing to do with
the scores)? The earlier recompute (penalty_recompute.py, cases B-F) answered with synthetic currents. This script
answers with the recorded currents of 20 banks: the same weights by the rows' real current, against the same weights
PERMUTED across the rows (20 permutations, the same posterior stream), against the unweighted threshold.

Inputs, read only:
  docs/bca_mechanics/data/M3_bank_rows.npz   the normalized water current of every held-out row of the 20 banks whose
                                              rows module M3 rebuilt through the studies' own loader (SHA-256 equal to
                                              the band records; 10 near-80 version-6 bands and 10 stage-2 uniform bands)
  docs/bca_mechanics/data/M4_real_data.json  current_range_bank: the smallest and largest RAW current among the held-out
                                              rows of each near-80 band (module M4 read them from the log). The normalized
                                              current is a linear map of the raw one with the gate's constants, shared by
                                              every setting of a seed and world (near80/code/settingdata.py:7), so the map
                                              recovered from the near-80 band of a seed and world also serves stage 2's
                                              band of the same seed and world (checked: the recovered currents of every
                                              bank lie inside the recorded law's range, uniform on [0.1, 0.7]).
  <runs>/near80/learners/near80/{stall,no_stall}/seed<K>/{v6_bank_scores.npz, v6_band.json}
  <runs>/stage2/detect/uniform/{stall,no_stall}/fixed/seed<K>/{bank_scores.npz, band.json}
                                              the recorded scores and the refresh's key rule (penalty_recompute.find_banks)

Weights. w(c) = exp(k c) for k = 2, 4 and 9.4, a tilt of the recorded (uniform) current law; the test mass is
E_test[w] = E_train[w^2] / E_train[w] by quadrature over [0.1, 0.7], as in M4_pinned.py. Slopes 2 and 4 are the
weighting test's mild and strong slopes (its current range is [0.1, 1.1]); 9.4 makes the Kish effective size about 0.32 of
the rows on the recorded range, the weighting test's strong Kish share (0.352). Only ratios of weights matter, so the
additive constant of the linear map drops out; its scale does not, which is why the raw current is recovered.

Usage (Windows venv or the pinned environment; the repository root on PYTHONPATH; the posterior stream is rebuilt with jax
as the band code did, and both environments draw the same stream, M3.md section 5):
  python docs/bca_mechanics/scripts/penalty_real_current.py --runs //wsl.localhost/Ubuntu-24.04/home/user/bca/runs/shift_explore \\
      --out docs/bca_mechanics/data/penalty_real_current.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import penalty_recompute as PR  # noqa: E402  (find_banks, refresh_stream, cal: the same calls as the pinned recompute)

C_LO, C_HI = 0.1, 0.7          # the recorded current law of these logs: uniform on [0.1, 0.7] (M4_pinned.py:68)
SLOPES = (2.0, 4.0, 9.4)
PERMUTATIONS = 20


def M(lam):
    return (C_HI - C_LO) if lam == 0.0 else (math.exp(lam * C_HI) - math.exp(lam * C_LO)) / lam


def test_mass(k):
    """E_test[w] = E_train[w^2] / E_train[w] for w = exp(k c), c uniform on [C_LO, C_HI] (M4_pinned.py:86-88)."""
    return (C_HI - C_LO) * M(2.0 * k) / M(k) ** 2


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="runs/shift_explore (holds near80/ and stage2/)")
    ap.add_argument("--rows", default=os.path.join(HERE, "..", "data", "M3_bank_rows.npz"))
    ap.add_argument("--m4", default=os.path.join(HERE, "..", "data", "M4_real_data.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = ap.parse_args()

    rows = np.load(args.rows)
    m4 = json.load(open(args.m4, encoding="utf8"))
    raw_range = {(b["world"], b["seed"]): b["current_range_bank"] for b in m4["bands"]}
    banks = [b for b in PR.find_banks(args.runs) if b["setting"] in ("near80", "main")]
    if not banks:
        sys.exit(f"no banks under {args.runs}")
    print(f"numpy {np.__version__}; {len(banks)} banks; slopes {SLOPES}; {args.permutations} permutations per bank and slope")
    print(f"test masses: " + ", ".join(f"k = {k}: {test_mass(k):.3f}" for k in SLOPES))
    results = []
    for b in banks:
        study = "near80" if b["study"].startswith("near-80") else "stage2"
        label = f"{study}_{b['world']}_fixed_seed{b['seed']}"
        z = rows[f"current_{label}"].astype(np.float64)
        scores = np.load(b["scores_file"])["scores"].astype(np.float64)
        n = scores.size
        assert z.size == n, (label, z.size, n)
        # undo the gate's normalization with the raw range of the near-80 band of the same seed and world
        z_ref = rows[f"current_near80_{b['world']}_fixed_seed{b['seed']}"].astype(np.float64)
        c_min, c_max = raw_range[(b["world"], b["seed"])]
        scale = (c_max - c_min) / (z_ref.max() - z_ref.min())
        c = c_min + (z - z_ref.min()) * scale
        range_ok = bool(c.min() > C_LO - 0.02 and c.max() < C_HI + 0.02)
        stream = PR.refresh_stream(b["rep"], b["steps"])
        u = PR.cal(scores, stream)
        R_u, lh_u = u.threshold, u.lambda_hat
        assert np.float32(R_u) == np.float32(b["uniform"]["R"]), (label, R_u, b["uniform"]["R"])
        out = dict(study=study, world=b["world"], seed=b["seed"], n=int(n), R_unweighted=R_u, lambda_hat_unweighted=lh_u,
                   c_range=[float(c.min()), float(c.max())], range_ok=range_ok,
                   spearman_score_current=spearman(scores, c))
        for k in SLOPES:
            w = np.exp(k * c)
            t = test_mass(k)
            r = PR.cal(scores, stream, w, t)
            perm = []
            for j in range(args.permutations):
                rng = np.random.default_rng([20261020, 9, int(b["seed"]), 1 if b["world"] == "stall" else 2, j])
                rp = PR.cal(scores, stream, w[rng.permutation(n)], t)
                perm.append(dict(R=rp.threshold, lambda_hat=rp.lambda_hat))
            pr = np.array([p["R"] for p in perm]) / R_u
            plh = np.array([p["lambda_hat"] for p in perm]) / lh_u
            out[f"k{k:g}"] = dict(test_mass=t, kish_share=r.n_eff / n, R=r.threshold, ratio_real=r.threshold / R_u,
                                  lambda_hat_ratio_real=r.lambda_hat / lh_u, margin_real=r.lambda_hpd / r.lambda_hat,
                                  perm_ratio_mean=float(pr.mean()), perm_ratio_min=float(pr.min()), perm_ratio_max=float(pr.max()),
                                  perm_lambda_hat_ratio_mean=float(plh.mean()),
                                  tilt_part=float(r.threshold / R_u / pr.mean()),
                                  real_above_perm_mean=bool(r.threshold / R_u > pr.mean()),
                                  real_above_perm_max=bool(r.threshold / R_u > pr.max()))
        results.append(out)
        print(f"  {label:32s} n {n:4d} R_u {R_u:.4f} c [{c.min():.3f}, {c.max():.3f}] rank(score, c) {out['spearman_score_current']:+.3f} | "
              + " | ".join(f"k={k:g}: real {out[f'k{k:g}']['ratio_real']:.3f} perm {out[f'k{k:g}']['perm_ratio_mean']:.3f} "
                           f"tilt {out[f'k{k:g}']['tilt_part']:.3f} Kish {out[f'k{k:g}']['kish_share']:.2f}" for k in SLOPES), flush=True)

    print()
    print("Summary over the banks (means; ranges in brackets):")
    print(f"{'slope k (Kish share)':22} {'R_real / R_u':>22} {'R_permuted / R_u':>18} {'tilt part real/perm':>24} "
          f"{'real > perm mean':>16} {'real > perm max':>15} {'plain quantile real / perm':>26}")
    summary = dict(banks=len(results), permutations=args.permutations, test_masses={f"k{k:g}": test_mass(k) for k in SLOPES},
                   range_ok_all=all(o["range_ok"] for o in results),
                   spearman_score_current_range=[min(o["spearman_score_current"] for o in results),
                                                 max(o["spearman_score_current"] for o in results)], slopes={})
    for k in SLOPES:
        key = f"k{k:g}"
        real = np.array([o[key]["ratio_real"] for o in results])
        perm = np.array([o[key]["perm_ratio_mean"] for o in results])
        tilt = np.array([o[key]["tilt_part"] for o in results])
        kish = np.array([o[key]["kish_share"] for o in results])
        lhr = np.array([o[key]["lambda_hat_ratio_real"] for o in results])
        lhp = np.array([o[key]["perm_lambda_hat_ratio_mean"] for o in results])
        above_mean = int(sum(o[key]["real_above_perm_mean"] for o in results))
        above_max = int(sum(o[key]["real_above_perm_max"] for o in results))
        summary["slopes"][key] = dict(kish_share_mean=float(kish.mean()), real_mean=float(real.mean()), real_min=float(real.min()),
                                      real_max=float(real.max()), perm_mean=float(perm.mean()), perm_min=float(perm.min()),
                                      perm_max=float(perm.max()), tilt_mean=float(tilt.mean()), tilt_min=float(tilt.min()),
                                      tilt_max=float(tilt.max()), real_above_perm_mean=above_mean, real_above_perm_max=above_max,
                                      plain_quantile_real_mean=float(lhr.mean()), plain_quantile_perm_mean=float(lhp.mean()))
        print(f"{k:<4g} ({kish.mean():.2f}){'':13} {real.mean():6.3f} [{real.min():.3f}, {real.max():.3f}] "
              f"{perm.mean():10.3f} [{perm.min():.3f}, {perm.max():.3f}] {tilt.mean():8.3f} [{tilt.min():.3f}, {tilt.max():.3f}] "
              f"{above_mean:>9d} of {len(results)} {above_max:>8d} of {len(results)} {lhr.mean():12.3f} / {lhp.mean():.3f}")
    print("Reading: 'real' gives every held-out row the weight exp(k c) by its recorded current; 'permuted' gives the same weights")
    print("to the rows in a random order (the smaller effective sample and the heavier test atom are kept, the link to the row is")
    print("broken); 'tilt part' is real over the permuted mean, the part of the rise that comes from pointing the weights at the")
    print("high-current rows, where the scores are larger (module M3, Part D).")
    if args.out:
        json.dump(dict(summary=summary, banks=results), open(args.out, "w", encoding="utf8"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
