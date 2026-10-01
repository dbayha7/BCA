"""Reproduce Table 2 of Lou and Luo, arXiv:2604.06464v3 (section 4.2, Appendix C).

Synthetic multilabel classification under covariate shift: x ~ N(0,1); each unit carries
K = 4 outcomes T_k | x ~ Exp(rate(x)); the bounded monotone loss is the exceedance fraction
l(lambda; x) = #{k : T_k > lambda} / K (B = 1), alpha = 0.4, beta = 0.95. Test covariates
follow the exponential tilt exp(gamma x), i.e. N(gamma, 1), with oracle ratio
w*(x) = exp(gamma x - gamma^2 / 2). Realized risk under the tilted law is
E_test[exp(-lambda rate(x))], computed by Gauss-Hermite quadrature, not simulation.

The paper does not state rate(x). This script uses rate(x) = exp(offset - slope x), with
slope and offset inferred from the shift-blind rows of Table 2 alone: BQ-CP and RCPS see
only calibration data, so their thresholds depend on the calibration law alone. The ratios
of their mean thresholds at n = 10 and n = 250 point to slope 0.69-0.70 and their n = 10
failure rates to about 0.67; the default slope ln 2 follows the ratios, and offset -0.019
then fits the four mean thresholds. Every W-CRC and WBCP number is a prediction, not a fit.

RCPS is reported twice. "RCPS" is the published Hoeffding-Bentkus bound (Bates et al.,
2021), whose Bentkus term is P(Bin(n, alpha) <= ceil(n R_hat)). "RCPS (floor)" uses
floor(n R_hat) instead, which is what the paper's RCPS numbers imply; scipy's binom.cdf
floors a non-integer count silently. The count loss makes n R_hat a multiple of 1/K, so the
two differ here and agree for the miscoverage loss of Table 1.

Weights follow the paper's compliant recipe, as in reproduce_table1.py: a logistic
discriminator fit on covariate samples Cw and Tw disjoint from the scored set Cs, with the
test mass wbar = E_test[w] estimated on unlabeled test covariates Ts. Two oracle rows are
reported, test mass E_test[w*] = exp(gamma^2) (Eq. 6) and test mass 1.

python experiments/wbcp/reproduce_table2.py --n 250
"""

import argparse
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np
from scipy import optimize, special, stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.wbcp.reproduce_table1 import logistic_ratio  # noqa: E402

PAPER = {  # Table 2 (gamma = 1): failure frequency, mean lambda over certified trials
    10: {"BQ-CP": (0.282, 2.44), "RCPS": (0.057, 3.49), "RCPS (floor)": (0.057, 3.49),
         "W-CRC": (0.132, 3.31), "WBCP": (0.000, 7.61), "WBCP (oracle w)": (0.000, 7.94),
         "WBCP (oracle, wbar=1)": (0.000, 7.94)},
    250: {"BQ-CP": (1.000, 1.05), "RCPS": (1.000, 1.16), "RCPS (floor)": (1.000, 1.16),
          "W-CRC": (0.437, 1.92), "WBCP": (0.079, 2.36), "WBCP (oracle w)": (0.056, 2.35),
          "WBCP (oracle, wbar=1)": (0.056, 2.35)},
}
# Table 2 caption: "At n = 10 WBCP finds no certifiable threshold in 93% of trials (W-CRC 4%, others 0)
# ... At n = 250 every rule always certifies." Read literally, "others 0" includes the oracle row, but
# its interval [0.0, 0.9] for 0 failures needs about 410 certified trials, i.e. about 96% abstention.
# The oracle rows are left out here for that reason.
PAPER_ABSTAIN = {10: {"BQ-CP": 0.0, "RCPS": 0.0, "RCPS (floor)": 0.0, "W-CRC": 0.04, "WBCP": 0.93},
                 250: dict.fromkeys(PAPER[250], 0.0)}
_CHUNK_ELEMENTS = 4_000_000  # bounds the draws-by-grid matrix held at once


def rate(x, slope, offset):
    return np.exp(offset - slope * x)


def risk_curve(gamma, slope, offset, nodes=120):
    """lambda -> E_{x ~ N(gamma, 1)}[Pr(T > lambda | x)] = E[exp(-lambda rate(x))]."""
    z, c = np.polynomial.hermite_e.hermegauss(nodes)
    r, c = rate(z + gamma, slope, offset), c / c.sum()
    return lambda lam: np.sum(c * np.exp(-np.multiply.outer(lam, r)), axis=-1)


def _settle(masses, alpha, extra=Fraction(0)):
    """Smallest index k with sum(masses[:k+1]) >= (1 - alpha) (sum(masses) + extra), or None.

    The float search is settled in exact rationals, since equality counts as covered.
    """
    cumulative = np.cumsum(masses)
    hit = cumulative >= (1.0 - alpha) * (cumulative[-1] + float(extra))
    k = int(np.argmax(hit)) if hit.any() else masses.size - 1
    exact = [Fraction(m) for m in masses]
    target = (1 - Fraction(str(alpha))) * (sum(exact) + extra)
    covered = sum(exact[:k + 1])
    while k > 0 and covered - exact[k] >= target:
        covered -= exact[k]
        k -= 1
    while covered < target:
        if k + 1 == masses.size:
            return None
        k += 1
        covered += exact[k]
    return k


def calibrate_count(outcomes, rng, weights, test_mass, *, alpha, beta, draws):
    """Algorithm 1 for the exceedance-fraction loss; returns (lambda_dep, lambda_hat, lambda_hpd).

    `outcomes` is n-by-K. The grid is every observed outcome; at grid point lambda a unit's
    loss is the share of its outcomes above lambda. One exponential E_i per unit (Eq. 6), so
    a unit's K outcomes share its posterior mass. L+(lambda) <= alpha exactly when the
    posterior mass of outcomes at or below lambda reaches (1 - alpha) of the total, the test
    atom included. inf means nothing is certifiable.
    """
    n, k = outcomes.shape
    scale = weights.max()
    weights, test_mass = weights / scale, test_mass / scale
    flat = outcomes.ravel()
    order = np.argsort(flat, kind="stable")
    grid, unit = flat[order], order // k

    # Eq. (1): smallest grid point whose weighted empirical risk is at most alpha.
    index = _settle(weights[unit] / k, alpha)
    lambda_hat = math.inf if index is None else float(grid[index])

    chunk = max(1, _CHUNK_ELEMENTS // (n * k + 1))
    crossings = []
    for start in range(0, draws, chunk):
        exponentials = rng.standard_exponential((min(chunk, draws - start), n + 1))
        unit_mass = exponentials[:, :n] * weights
        total = unit_mass.sum(axis=1) + exponentials[:, n] * test_mass
        covered = np.cumsum(unit_mass[:, unit] / k, axis=1) >= (1.0 - alpha) * total[:, None]
        crossings.append(np.where(covered.any(axis=1), grid[covered.argmax(axis=1)], np.inf))
    posterior = np.sort(np.concatenate(crossings))
    lambda_hpd = float(posterior[math.ceil(Fraction(str(beta)) * int(draws)) - 1])  # Eq. (7)
    return max(lambda_hat, lambda_hpd), lambda_hat, lambda_hpd


def rcps(outcomes, alpha, delta, rounding):
    """Shift-blind RCPS with the Hoeffding-Bentkus bound over the n units.

    The unweighted risk at the j-th sorted outcome is exceed_j / (nK) with integer exceed_j,
    so n R_hat = exceed_j / K exactly; `rounding` is "ceil" (published) or "floor".
    """
    n, k = outcomes.shape
    grid = np.sort(outcomes.ravel())
    exceed = n * k - np.arange(1, n * k + 1)
    risk = exceed / (n * k)
    count = -(-exceed // k) if rounding == "ceil" else exceed // k
    clipped = np.minimum(risk, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = special.xlogy(clipped, clipped / alpha) + special.xlogy(1 - clipped, (1 - clipped) / (1 - alpha))
    certified = np.minimum(np.exp(-n * h1), math.e * stats.binom.cdf(count, n, alpha)) < delta
    return float(grid[certified.argmax()]) if certified.any() else math.inf


def weighted_crc(outcomes, weights, test_mass, alpha):
    """W-CRC: smallest lambda with (sum w l + wbar B) / (sum w + wbar) <= alpha.

    The test point enters with the plug-in mass wbar, the alpha-crossing of E[L+_w] in the
    paper's section 3.1. Appendix C instead names CRC Prop. 2's transductive form, which uses
    the test point's own weight w(X_{n+1}). Only the wbar form matches Table 2's W-CRC row: the
    transductive form fails about 33% at n = 10 with about 11% abstention (paper 13.2%, 4%).
    """
    n, k = outcomes.shape
    scale = weights.max()
    flat = outcomes.ravel()
    order = np.argsort(flat, kind="stable")
    index = _settle(weights[order // k] / scale / k, alpha, Fraction(test_mass / scale))
    return math.inf if index is None else float(flat[order][index])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=250, help="scored calibration units |Cs|")
    parser.add_argument("--outcomes", type=int, default=4, help="K outcomes per unit")
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--weight-fit", type=int, default=200, help="|Cw| = |Tw|")
    parser.add_argument("--test-mass-size", type=int, default=200, help="|Ts|, unlabeled test covariates")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--rate-slope", type=float, default=math.log(2), help="rate(x) = exp(offset - slope x)")
    parser.add_argument("--rate-offset", type=float, default=-0.019)
    parser.add_argument("--seed", type=int, default=2026100101)
    parser.add_argument("--output", help="optional JSON path; must not exist")
    args = parser.parse_args()
    if args.output and os.path.exists(args.output):
        parser.error("--output already exists")

    rng = np.random.default_rng(args.seed)
    gamma, n, k = args.gamma, args.n, args.outcomes
    risk = risk_curve(gamma, args.rate_slope, args.rate_offset)
    lam_star = optimize.brentq(lambda lam: risk(lam) - args.alpha, 1e-9, 1e6)
    lam_star_cal = optimize.brentq(lambda lam: risk_curve(0.0, args.rate_slope, args.rate_offset)(lam) - args.alpha,
                                   1e-9, 1e6)
    names = ["BQ-CP", "RCPS", "RCPS (floor)", "W-CRC", "WBCP", "WBCP (oracle w)", "WBCP (oracle, wbar=1)"]
    arms = {name: [] for name in names}
    n_eff = []
    posterior = dict(alpha=args.alpha, beta=args.beta, draws=args.draws)

    for _ in range(args.trials):
        x = rng.normal(size=n)  # scored calibration set Cs
        outcomes = rng.standard_exponential((n, k)) / rate(x, args.rate_slope, args.rate_offset)[:, None]
        estimated = logistic_ratio(rng.normal(size=args.weight_fit),  # Cw
                                   rng.normal(gamma, 1.0, args.weight_fit))  # Tw
        weights = estimated(x)
        wbar = float(np.mean(estimated(rng.normal(gamma, 1.0, args.test_mass_size))))  # Ts, unlabeled
        n_eff.append(float(weights.sum() ** 2 / np.dot(weights, weights)))

        arms["BQ-CP"].append(calibrate_count(outcomes, rng, np.ones(n), 1.0, **posterior)[2])
        arms["RCPS"].append(rcps(outcomes, args.alpha, 1 - args.beta, "ceil"))
        arms["RCPS (floor)"].append(rcps(outcomes, args.alpha, 1 - args.beta, "floor"))
        arms["W-CRC"].append(weighted_crc(outcomes, weights, wbar, args.alpha))
        arms["WBCP"].append(calibrate_count(outcomes, rng, weights, wbar, **posterior)[0])
        oracle = np.exp(gamma * x - gamma ** 2 / 2)
        arms["WBCP (oracle w)"].append(calibrate_count(outcomes, rng, oracle, math.exp(gamma ** 2), **posterior)[0])
        arms["WBCP (oracle, wbar=1)"].append(calibrate_count(outcomes, rng, oracle, 1.0, **posterior)[0])

    paper = PAPER.get(n) if gamma == 1 and k == 4 else None
    print(f"gamma={gamma}  n={n}  K={k}  trials={args.trials}  draws={args.draws}  "
          f"|Cw|=|Tw|={args.weight_fit}  |Ts|={args.test_mass_size}  rate(x)=exp({args.rate_offset:+.3f} - {args.rate_slope:.4f} x)")
    print(f"lambda*_test={lam_star:.4f}  lambda*_cal={lam_star_cal:.4f}  mean n_eff (estimated w)={np.mean(n_eff):.1f}")
    header = f"{'rule':<23}{'fail freq':>10}{'95% CI':>18}{'mean lam':>10}{'mean risk':>11}{'abstain':>9}"
    print(header + ("   paper (freq / lam / abstain)" if paper else ""))
    summary = {}
    for name, thresholds in arms.items():
        thresholds = np.asarray(thresholds)
        certified = thresholds[np.isfinite(thresholds)]
        abstain = 1 - certified.size / thresholds.size
        if not certified.size:  # every trial abstained: nothing deployed to score
            summary[name] = dict(fail=None, ci=None, mean_lambda=None, risk=None, abstain=abstain)
            print(f"{name:<23}{'-':>10}{'-':>18}{'-':>10}{'-':>11}{abstain:>9.1%}")
            continue
        fails = int(np.sum(certified < lam_star))  # risk decreases in lambda: violation iff lambda < lambda*
        low, high = stats.binomtest(fails, certified.size).proportion_ci(method="exact")
        row = dict(fail=fails / certified.size, ci=(low, high), mean_lambda=float(np.mean(certified)),
                   risk=float(np.mean(risk(certified))), abstain=abstain, certified=int(certified.size))
        summary[name] = row
        line = (f"{name:<23}{row['fail']:>10.1%}{f'[{low:.1%}, {high:.1%}]':>18}{row['mean_lambda']:>10.2f}"
                f"{row['risk']:>11.1%}{abstain:>9.1%}")
        if paper:
            freq, lam = paper[name]
            line += f"   {freq:.1%} / {lam:.2f}"
            if PAPER_ABSTAIN[n].get(name):
                line += f" / {PAPER_ABSTAIN[n][name]:.0%}"
        print(line)
    if args.output:
        with open(args.output, "x") as handle:
            json.dump(dict(settings=vars(args), lambda_star=lam_star, lambda_star_cal=lam_star_cal,
                           mean_n_eff=float(np.mean(n_eff)), summary=summary), handle, indent=1)


if __name__ == "__main__":
    main()
