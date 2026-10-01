"""Reproduce Table 1 of Lou and Luo, arXiv:2604.06464v3 (section 4.1, Figure 3).

Synthetic heteroskedastic regression under covariate shift: X ~ U[0,4],
Y | X ~ N(0, X^2), predictor 0, score |Y|, interval [-lambda, lambda], miscoverage
loss, alpha = 0.1, beta = 0.95, n = 200 scored calibration points. Test covariates
have density proportional to exp(gamma x) on [0,4]; gamma = 1 gives the paper's
shortest valid length 2 lambda* = 10.6, gamma = 0 is the exchangeable control.
Realized risk under the tilted law is computed by quadrature, not simulation.

Weights follow the paper's compliant recipe: a logistic discriminator fit on
covariate samples Cw (calibration law) and Tw (test law), disjoint from the scored
calibration set Cs and from the unlabeled test covariates Ts, on which the test
mass wbar = E_test[w] is estimated. The paper does not state |Cw|, |Tw|, |Ts| or
the number of posterior draws; they are command-line settings here.

Two oracle rows are reported. "WBCP (oracle w)" uses Eq. (6) as written, test mass
wbar = E_test[w*] (about 2.07 at gamma = 1). The paper's oracle row is matched
instead by wbar = E_cal[w*] = 1, reported as "WBCP (oracle, wbar=1)".

python experiments/wbcp/reproduce_table1.py --gamma 1
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import optimize, special, stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from calibration.wbcp import calibrate  # noqa: E402

PAPER = {  # Table 1 (gamma = 1): failure frequency, mean risk, mean length
    "BQ-CP": (0.923, 0.144, 9.35), "RCPS": (0.852, 0.133, 9.67), "W-CRC": (0.428, 0.095, 10.98),
    "WBCP": (0.050, 0.050, 13.41), "WBCP (oracle w)": (0.067, 0.054, 13.09),
    "WBCP (oracle, wbar=1)": (0.067, 0.054, 13.09),
}


def oracle_ratio(x, gamma):
    """dP_test/dP_cal for the exp(gamma x) tilt of U[0,4], normalized so E_cal = 1."""
    return np.ones_like(x) if gamma == 0 else 4 * gamma * np.exp(gamma * x) / np.expm1(4 * gamma)


def oracle_test_mass(gamma):
    """E_test[w*] = E_cal[w*^2] in closed form."""
    return 1.0 if gamma == 0 else 2 * gamma * (np.exp(4 * gamma) + 1) / np.expm1(4 * gamma)


def sample_test_covariates(rng, size, gamma):
    if gamma == 0:
        return rng.uniform(0, 4, size)
    return np.log1p(rng.uniform(size=size) * np.expm1(4 * gamma)) / gamma  # inverse CDF


def risk_curve(gamma, nodes=400):
    """lambda -> Pr_test(|Y| > lambda) by Gauss-Legendre quadrature over x in (0, 4)."""
    x, c = np.polynomial.legendre.leggauss(nodes)
    x, c = 2 * (x + 1), 2 * c
    density = oracle_ratio(x, gamma) / 4
    return lambda lam: np.sum(2 * stats.norm.sf(np.divide.outer(lam, x)) * density * c, axis=-1)


def logistic_ratio(cal_x, test_x):
    """Unregularized 1-D logistic discriminator (Newton), turned into a density ratio."""
    features = np.column_stack([np.ones(cal_x.size + test_x.size), np.concatenate([cal_x, test_x])])
    labels = np.concatenate([np.zeros(cal_x.size), np.ones(test_x.size)])
    theta = np.zeros(2)
    for _ in range(50):
        p = special.expit(features @ theta)
        step = np.linalg.solve(features.T @ (features * (p * (1 - p))[:, None]), features.T @ (labels - p))
        theta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    prior = cal_x.size / test_x.size  # class-prior correction
    return lambda x: prior * np.exp(theta[0] + theta[1] * x)


def rcps_index(n, alpha, delta):
    """RCPS with the Hoeffding-Bentkus bound (Bates et al., 2021), shift-blind.

    The unweighted risk at the j-th sorted score is (n-j)/n, so the selection is a
    fixed order statistic: the first j whose HB p-value falls below delta, or None
    (abstain) if no threshold is certifiable.
    """
    exceed = n - np.arange(1, n + 1)  # integer exceedance counts; ceil(n * risk) = exceed exactly
    clipped = np.minimum(exceed / n, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = special.xlogy(clipped, clipped / alpha) + special.xlogy(1 - clipped, (1 - clipped) / (1 - alpha))
    certified = np.minimum(np.exp(-n * h1), math.e * stats.binom.cdf(exceed, n, alpha)) < delta
    return int(np.argmax(certified)) if certified.any() else None


def weighted_crc(sorted_scores, sorted_weights, test_mass, alpha):
    """W-CRC for miscoverage: Tibshirani's weighted quantile with test mass wbar."""
    covered = np.cumsum(sorted_weights) >= (1 - alpha) * (sorted_weights.sum() + test_mass)
    return sorted_scores[covered.argmax()] if covered.any() else math.inf


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--n", type=int, default=200, help="scored calibration size |Cs|")
    parser.add_argument("--weight-fit", type=int, default=200, help="|Cw| = |Tw|")
    parser.add_argument("--test-mass-size", type=int, default=200, help="|Ts|, unlabeled test covariates")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026092901)
    parser.add_argument("--output", help="optional JSON path; must not exist")
    args = parser.parse_args()
    if args.output and os.path.exists(args.output):
        parser.error("--output already exists")

    rng = np.random.default_rng(args.seed)
    risk = risk_curve(args.gamma)
    lam_star = optimize.brentq(lambda lam: risk(lam) - args.alpha, 1e-6, 100)
    rcps_j = rcps_index(args.n, args.alpha, 1 - args.beta)
    arms = {name: [] for name in PAPER}
    n_eff = []

    for _ in range(args.trials):
        x = rng.uniform(0, 4, args.n)  # scored calibration set Cs
        scores = np.abs(rng.normal(0, x))
        estimated = logistic_ratio(rng.uniform(0, 4, args.weight_fit),  # Cw
                                   sample_test_covariates(rng, args.weight_fit, args.gamma))  # Tw
        test_x = sample_test_covariates(rng, args.test_mass_size, args.gamma)  # Ts, unlabeled
        order = np.argsort(scores, kind="stable")
        weights = estimated(x)
        wbar = float(np.mean(estimated(test_x)))

        arms["BQ-CP"].append(calibrate(scores, rng, alpha=args.alpha, beta=args.beta,
                                       draws=args.draws).lambda_hpd)
        arms["RCPS"].append(math.inf if rcps_j is None else scores[order][rcps_j])
        arms["W-CRC"].append(weighted_crc(scores[order], weights[order], wbar, args.alpha))
        result = calibrate(scores, rng, weights, wbar, alpha=args.alpha, beta=args.beta, draws=args.draws)
        arms["WBCP"].append(result.threshold)
        n_eff.append(result.n_eff)
        oracle = oracle_ratio(x, args.gamma)
        for name, oracle_wbar in (("WBCP (oracle w)", oracle_test_mass(args.gamma)),
                                  ("WBCP (oracle, wbar=1)", 1.0)):
            arms[name].append(calibrate(scores, rng, oracle, oracle_wbar,
                                        alpha=args.alpha, beta=args.beta, draws=args.draws).threshold)

    print(f"gamma={args.gamma}  trials={args.trials}  n={args.n}  draws={args.draws}  "
          f"|Cw|=|Tw|={args.weight_fit}  |Ts|={args.test_mass_size}")
    print(f"lambda*={lam_star:.4f} (shortest valid length {2 * lam_star:.3f})  "
          f"mean n_eff={np.mean(n_eff):.1f}")
    header = f"{'rule':<23}{'fail freq':>10}{'95% CI':>18}{'mean risk':>11}{'mean len':>10}{'abstain':>9}"
    print(header + ("   paper (freq / risk / len)" if args.gamma == 1 else ""))
    summary = {}
    for name, thresholds in arms.items():
        thresholds = np.asarray(thresholds)
        certified = thresholds[np.isfinite(thresholds)]
        abstain = 1 - certified.size / thresholds.size
        if not certified.size:  # every trial abstained: nothing deployed to score
            summary[name] = dict(fail=None, ci=None, risk=None, length=None, abstain=abstain)
            print(f"{name:<23}{'-':>10}{'-':>18}{'-':>11}{'-':>10}{abstain:>9.1%}")
            continue
        fails = int(np.sum(certified < lam_star))  # risk decreases in lambda: violation iff lambda < lambda*
        low, high = stats.binomtest(fails, certified.size).proportion_ci(method="exact")
        row = dict(fail=fails / certified.size, ci=(low, high), risk=float(np.mean(risk(certified))),
                   length=float(np.mean(2 * certified)), abstain=abstain)
        summary[name] = row
        line = (f"{name:<23}{row['fail']:>10.1%}{f'[{low:.1%}, {high:.1%}]':>18}{row['risk']:>11.1%}"
                f"{row['length']:>10.2f}{row['abstain']:>9.1%}")
        if args.gamma == 1:
            line += "   {:.1%} / {:.1%} / {:.2f}".format(*PAPER[name])
        print(line)
    if args.output:
        with open(args.output, "x") as handle:
            json.dump(dict(settings=vars(args), lambda_star=lam_star, mean_n_eff=float(np.mean(n_eff)),
                           summary=summary), handle, indent=1)


if __name__ == "__main__":
    main()
