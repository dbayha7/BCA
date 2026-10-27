"""The estimated-weight classifier: logistic regression on the 13-number observation, fitted by Newton's method with no
penalty, separating a training-law sample (class 0) from a test-law sample (class 1).

Why it is correctly specified: the current c is one of the 13 numbers (index 10) and, given c, a step looks the same
under every law, so the true log-odds of "test" are log w(c) + log(n1 / n0) = const + (lam_test + 2) c: linear in the
observation, with slope 0, 2 or 4 on c and 0 on everything else.

Estimated weight of a step: w_hat(x) = exp(fitted log-odds) x n0 / n1 (the class-size correction; 1 for equal
samples). Estimated test mass: the mean of w_hat over the test-law sample's steps whose recorded command lies outside
the ball. The commands are used only for that and for the refusal: a step inside the (public) ball is refused, as in
the exact arm, never given a weight of 0. The inputs are standardised by the pooled sample mean and SD for numerical
conditioning only; an unpenalised fit's predictions do not depend on it. Nothing here reads rewards, next states or
values, and nothing here is BCA.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from experiments.shift import thruster_chain as TC
from experiments.shift_iw import world as W


@dataclass
class Classifier:
    intercept: float
    coef: np.ndarray        # per standardised input (13)
    mean: np.ndarray
    std: np.ndarray
    n0: int
    n1: int
    iterations: int
    converged: bool
    max_score: float        # largest |gradient of the log-likelihood| at the fit (0 at the exact optimum)

    def log_odds(self, obs):
        z = (np.asarray(obs, np.float64) - self.mean) / self.std
        return self.intercept + z @ self.coef

    def log_weight(self, obs):
        return self.log_odds(obs) + math.log(self.n0 / self.n1)

    def weight(self, obs):
        return np.exp(self.log_weight(obs))

    @property
    def slope_on_c(self):
        """The fitted slope of the log-odds on c in raw units (exact: 0, 2 or 4)."""
        return float(self.coef[TC.CURRENT] / self.std[TC.CURRENT])

    @property
    def raw_coef(self):
        """Slopes on the 13 raw observation numbers."""
        return self.coef / self.std

    def describe(self):
        return dict(intercept=float(self.intercept), raw_coef=self.raw_coef.tolist(), slope_on_c=self.slope_on_c,
                    n0=int(self.n0), n1=int(self.n1), iterations=int(self.iterations), converged=bool(self.converged),
                    max_score=float(self.max_score))


def _loglik(X, y, beta):
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit(obs0, obs1, max_iter=100, tol=1e-10):
    """Logistic regression of class (0: obs0, 1: obs1) on the observation by Newton's method (float64, no penalty,
    step halving if the log-likelihood would fall). Converged when the largest Newton step is below tol."""
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
    return Classifier(float(beta[0]), beta[1:], mean, std, len(x0), len(x1), it, converged, score)


def fit_samples(samples, tilt):
    """The classifier of a tilt from one seed's classifier samples (data.classifier_samples): training-law sample vs
    the tilt's sample; for 'none' (the no-tilt control), training-law sample vs the second training-law sample."""
    other = "train2" if tilt == "none" else tilt
    return fit(samples["train"]["observations"], samples[other]["observations"])


def estimated_arm(clf, bank_obs, test_obs, test_actions, gap):
    """The estimated arm's inputs: bank weights w_hat at the bank rows' observations, the test mass (mean w_hat over
    the test sample's steps outside the ball) and the refusal of test-sample steps inside the ball."""
    inside = gap.contains(test_actions)
    w_test = clf.weight(test_obs)
    return dict(bank_weights=clf.weight(bank_obs), test_mass=float(w_test[~inside].mean()),
                test_sample_in_ball_share=float(inside.mean()))


def compare(clf, bank_obs, c_bank, tilt):
    """Estimated against exact weights at the bank rows: rank correlation and root-mean-square error of log weights
    after removing the common factor (only ratios matter), plus the fitted and exact slopes on c."""
    est = clf.log_weight(bank_obs)
    exact = np.log(W.ratio(c_bank, tilt))
    d = (est - est.mean()) - (exact - exact.mean())
    rank = stats.spearmanr(est, exact)[0] if np.ptp(exact) > 0 else float("nan")
    return dict(slope_on_c=clf.slope_on_c, exact_slope=W.ratio_slope(tilt), rank_correlation=float(rank),
                rms_log_error=float(np.sqrt(np.mean(d ** 2))))
