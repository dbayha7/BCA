"""Outcome-free settings of the step-4 design: closed form, or Monte Carlo over Gaussian states; no actor run, no J of
any arm.

Consolidates the designers' scratch scripts in experiments/signal/step4_design_scripts (synth_settings.py,
synth_pull.py, lq_common.py, lq_levels.py, lq_mixed.py, lq_sigma.py) on the harness's own code: the system, Q^pi and
J come from lq_harness (quadratic_q, value), and the multiset, tilt scale and cone threshold from placement_harness.
J is evaluated only at fixed gains (the behaviour gains, the BC targets and the start), never at an arm.

- Weight multiset (design section 2; expectations B6): CV, range, Kish n_eff / n and p90 / p10 of the rank-normal
  multiset at n = 10,000 and 256 for s = 0.47, 0.83 and 1.
- Values (Part 0 item 6): J*, J(0), J(K_0) and the gap J* - J(K_0), J(0.1338 K*) and J(0.5 K*) (the BC targets in
  common-state mode), and J of the episodic mixed least-squares target K* Sigma_e (Sigma_e + Sigma_p)^-1; the medium
  gain as the gain whose J is halfway between J(0) and J*; sqrt(gamma) rho(A - B K_0) and the eigenvalues of Q^pi's
  action block H_aa at K_0.
- State second moments of 50-step rollouts with behaviour noise (tr Sigma under K_0, K*, 0.1338 K*, 0).
- Tilt scale c for alignment a* (design section 7, placement_harness.tilt_c).
- cos(BC pull, Q^pi0 ascent) = cos((K_bc - K_0) Sigma, +grad_K E Q^pi(s, -K s) at K_0) (B5), on common states
  (Sigma(K_0)) and on each level's own states. The ascent is the gradient itself, E[grad_a Q(s, -K_0 s) (-s)^T] = -2
  (H_as - H_aa K_0) Sigma. synth_pull.py printed the cosine with its negative (the descent direction): its signs are
  flipped here, and the draft's B5 already carries the corrected signs.
- Cone (design section 7): c0 with mean c(s) = 0.25, cos(D_q Sigma, D_q Sigma_c) with D_q = K_0 - K-bar_b, and
  corr(c(s), |s|^2), on Gaussian states N(0, Sigma(K_0)) (400,000 draws, seed 0, as synth_settings.py).
- lambda_0 = 2.5 / mean |Q1(s, pi_0(s))| on those states, and the fixed point's definiteness at K_0 (minimum eigenvalue
  of half the loss Hessian at BC weight 1): clean and tilt (H_aa unchanged), q1_optimistic kappa 0.5 / 1 / 2 (with the
  clean lambda_0, as synth_settings.py, and with the case's own lambda_0 per level), the cone kappa 1 / 2.
- Two numbers the draft's reasoning quotes: block C's anti-BC strength lambda_0 kappa d_a at kappa = 1, and the cone's
  uncancellable share sin(arccos cos) at expert.

Every number is compared with the value quoted in runs/wbcp_signal/step4/expectations_DRAFT.md (B5, B6, Part 0 item 6
and the reasoning of Part 0) or in STEP4_DESIGN.md: a match is agreement to the quoted number's last decimal (half a
unit). Differences are listed and explained in the printout.

python experiments/signal/step4_settings.py [--output runs/wbcp_signal/step4/settings.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.signal import lq_harness as LQ  # noqa: E402
from experiments.signal import placement_harness as PH  # noqa: E402

GAUSSIAN_DRAWS, GAUSSIAN_SEED = 400000, 0
# Values quoted in expectations_DRAFT.md / STEP4_DESIGN.md, as strings so that the quoted precision sets the tolerance.
QUOTED = {
    "multiset_s0.83_cv": "0.914", "multiset_s0.83_lo": "0.090", "multiset_s0.83_hi": "5.714",
    "multiset_s0.83_kish": "0.545", "multiset_s0.83_p90_p10": "8.39",
    "multiset_s0.47_cv": "0.482", "multiset_s0.47_lo": "0.277", "multiset_s0.47_hi": "2.908",
    "multiset_s0.47_kish": "0.811",
    "J_opt": "-2.2951", "J_zero": "-4.4335", "J_K0": "-2.7824", "gap": "0.4873", "J_medium": "-3.3641",
    "J_half": "-2.4985", "sqrt_gamma_rho_K0": "0.688", "H_aa_eig_1": "-1.897", "H_aa_eig_2": "-1.453",
    "lambda0_clean": "1.68", "medium_gain": "0.1338",
    "tr_sigma_K0": "0.136", "tr_sigma_expert": "0.105", "tr_sigma_poor": "0.321",
    "cos_pull_common_expert": "0.99", "cos_pull_common_mixed": "0.53", "cos_pull_common_medium": "-0.11",
    "cos_pull_common_poor": "-0.24", "cos_pull_own_expert": "0.99", "cos_pull_own_medium": "-0.57",
    "cos_pull_own_poor": "-0.78",
    "tilt_c_0.9": "0.461", "tilt_c_0.5": "0.897", "tilt_c_0.0": "1.414", "tilt_c_-0.5": "3.346",
    "cone_c0": "0.390", "cone_corr_s2": "-0.105",
    "q1opt_k2_H1aa_eig_1": "0.10", "q1opt_k2_H1aa_eig_2": "0.55", "blockC_anti_bc": "3.4",
    "cone_uncancellable": "0.38",
}
# Quoted as a range: Part 0 "Why 3", "the Gaussian approximation gives 0.92-0.93" (expert, medium, poor; B2.3 says
# "about 0.92"). The designers' script did not compute the mixed level (0.908 here; Part 0 item 3's forecast range
# [0.88, 0.96] contains it).
QUOTED_RANGES = {f"cone_cos_{name}": ("0.92", "0.93") for name in ("expert", "medium", "poor")}
NOTES = {
    "cos_pull_*": "ascent = +grad_K E Q^pi; synth_pull.py's printed values are the synth_pull_descent_* entries "
                  "(opposite sign). The draft's B5 already quotes the corrected (ascent) signs, which match.",
    "cone_cos_mixed": "not quoted: the designers' script computed expert, medium and poor only.",
    "lambda0_q1opt_* / lambda0_cone_*": "the case's own lambda_0 (Q1 includes the error at pi_0); synth_settings.py "
                                        "used the clean lambda_0 for the q1_optimistic definiteness (the "
                                        "*_clean_lambda entries). The sign of the minimum eigenvalue is the same.",
}


def rollout_second_moment(system, K, sigma_b):
    """Time-averaged E[s s^T] over L-step episodes from s0 ~ N(0, Sigma_0) under a = -K s + N(0, sigma_b^2 I)."""
    S, acc = system.init.copy(), np.zeros_like(system.init)
    M = system.A - system.B @ K
    for _ in range(system.length):
        acc += S
        S = M @ S @ M.T + sigma_b ** 2 * system.B @ system.B.T + system.noise
    return acc / system.length


def ascent(H, K, Sigma):
    """grad_K E[Q(s, -K s)] = E[grad_a Q (-s)^T] = -2 (H_as - H_aa K) Sigma for Q = z^T H z + h."""
    ds = K.shape[1]
    return -2 * (H[ds:, :ds] - H[ds:, ds:] @ K) @ Sigma


def multiset_stats(n, s, clip=PH.RANK_CLIP):
    m = PH.multiset(n, s, clip)
    return dict(cv=float(m.std() / m.mean()), lo=float(m.min()), hi=float(m.max()),
                kish=float(m.sum() ** 2 / (n * np.sum(m ** 2))),
                p90_p10=float(np.percentile(m, 90) / np.percentile(m, 10)))


def medium_gain(system, k_opt, target=0.5):
    """The gain g with (J(g K*) - J(0)) / (J* - J(0)) = target (bisection on [0, 1], as lq_levels.py)."""
    J0, J1 = LQ.value(system, 0 * k_opt), LQ.value(system, k_opt)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (LQ.value(system, mid * k_opt) - J0) / (J1 - J0) < target else (lo, mid)
    return hi


def compute(settings=PH.Settings()):
    """Every outcome-free settings number, as a flat dict (keys as in QUOTED, plus extras)."""
    h = PH.PlacementHarness(settings)
    system, k_opt, K0 = h.system, h.k_opt, h.K0
    sb, ds = settings.behavior_noise, system.dims[0]
    out = {}
    for s in (0.47, 0.83, 1.0):
        for n in (10000, 256):
            for key, value in multiset_stats(n, s).items():
                out[f"multiset_s{s:.2f}{'' if n == 10000 else '_n256'}_{key}"] = value
    value = lambda K: LQ.value(system, K)
    out.update(J_opt=value(k_opt), J_zero=value(0 * k_opt), J_K0=value(K0), J_medium=value(0.1338 * k_opt),
               J_half=value(0.5 * k_opt))
    out["gap"] = out["J_opt"] - out["J_K0"]
    out["medium_gain"] = medium_gain(system, k_opt)
    rho = np.max(np.abs(np.linalg.eigvals(system.A - system.B @ K0)))
    out["sqrt_gamma_rho_K0"] = float(math.sqrt(system.gamma) * rho)
    H, h0, Pv = LQ.exact_q(system, K0)
    out["H_aa_eig_1"], out["H_aa_eig_2"] = (float(x) for x in np.linalg.eigvalsh(H[ds:, ds:]))
    sig = {name: rollout_second_moment(system, g * k_opt, sb) for name, g in
           (("expert", 1.0), ("medium", 0.1338), ("poor", 0.0))}
    sig["K0"] = rollout_second_moment(system, K0, sb)
    for name, S in sig.items():
        out[f"tr_sigma_{name}"] = float(np.trace(S))
    K_mixed_ep = k_opt @ sig["expert"] @ np.linalg.inv(sig["expert"] + sig["poor"])
    out["J_mixed_episodic_target"] = value(K_mixed_ep)
    out["mixed_episodic_target_proj_Kstar"] = float(np.sum(K_mixed_ep * k_opt) / np.sum(k_opt * k_opt))
    for a in PH.TILT_ALIGNMENTS:
        out[f"tilt_c_{a:.1f}"] = PH.tilt_c(a)
    # cos(BC pull, ascent): common states (Sigma(K_0)) with each level's least-squares target, and own states.
    targets = dict(expert=k_opt, mixed=0.5 * k_opt, medium=0.1338 * k_opt, poor=0 * k_opt)
    for name, K_bc in targets.items():
        c = LQ.cosine((K_bc - K0) @ sig["K0"], ascent(H, K0, sig["K0"]))
        out[f"cos_pull_common_{name}"], out[f"synth_pull_descent_common_{name}"] = c, -c
    for name, K_bc in (("expert", k_opt), ("medium", 0.1338 * k_opt), ("poor", 0 * k_opt)):
        c = LQ.cosine((K_bc - K0) @ sig[name], ascent(H, K0, sig[name]))
        out[f"cos_pull_own_{name}"], out[f"synth_pull_descent_own_{name}"] = c, -c
    S_mix = (sig["expert"] + sig["poor"]) / 2
    out["cos_pull_own_mixed_episodic"] = LQ.cosine((K_mixed_ep - K0) @ S_mix, ascent(H, K0, S_mix))
    # Cone and lambda_0 on Gaussian states N(0, Sigma(K_0)) (states only; no outcomes).
    X = np.random.default_rng(GAUSSIAN_SEED).multivariate_normal(np.zeros(ds), sig["K0"], size=GAUSSIAN_DRAWS)
    c0 = PH.cone_c0(X)
    w = PH.cone_weight(X, PH.CONE_V, c0)
    Sx, Sw = X.T @ X / len(X), (X * w[:, None]).T @ X / len(X)
    out["cone_c0"], out["cone_mean"] = c0, float(w.mean())
    for name, K_b in targets.items():
        d = K0 - K_b
        out[f"cone_cos_{name}"] = LQ.cosine(d @ Sx, d @ Sw)
    out["cone_corr_s2"] = float(np.corrcoef(w, np.sum(X ** 2, 1))[0, 1])
    out["cone_uncancellable"] = float(math.sqrt(1 - out["cone_cos_expert"] ** 2))
    v_on = np.einsum("ni,ij,nj->n", X, Pv, X) + h0  # Q^pi(s, pi_0(s)) = V(s)
    lam_clean = 2.5 / np.mean(np.abs(v_on))
    out["lambda0_clean"], out["blockC_anti_bc"] = float(lam_clean), float(lam_clean * 1.0 * system.dims[1])
    Haa, I2 = H[ds:, ds:], np.eye(system.dims[1])
    half_hessian = lambda lam, Paa: -lam * np.kron(Sx, Paa) + 0.5 * np.kron(Sx, I2)
    out["fp_min_eig_clean_and_tilt"] = float(np.linalg.eigvalsh(half_hessian(lam_clean, Haa)).min())
    for kappa in (0.5, 1.0, 2.0):
        eig = np.linalg.eigvalsh(Haa + kappa * I2)
        out[f"q1opt_k{kappa:g}_H1aa_eig_1"], out[f"q1opt_k{kappa:g}_H1aa_eig_2"] = (float(x) for x in eig)
        out[f"fp_min_eig_q1opt_k{kappa:g}_clean_lambda"] = float(
            np.linalg.eigvalsh(half_hessian(lam_clean, Haa + kappa * I2)).min())
        for name, K_b in targets.items():
            dev = np.sum(np.square(X @ (K_b - K0).T), 1)  # |pi_0(s) + K_b s|^2
            lam = 2.5 / np.mean(np.abs(v_on + kappa * dev))
            out[f"lambda0_q1opt_k{kappa:g}_{name}"] = float(lam)
            out[f"fp_min_eig_q1opt_k{kappa:g}_{name}"] = float(
                np.linalg.eigvalsh(half_hessian(lam, Haa + kappa * I2)).min())
    for kappa in (1.0, 2.0):
        for name, K_b in targets.items():
            dev = np.sum(np.square(X @ (K_b - K0).T), 1)
            lam = 2.5 / np.mean(np.abs(v_on + kappa * w * dev))
            A = half_hessian(lam, Haa) - lam * kappa * np.kron(Sw, I2)
            out[f"lambda0_cone_k{kappa:g}_{name}"] = float(lam)
            out[f"fp_min_eig_cone_k{kappa:g}_{name}"] = float(np.linalg.eigvalsh(A).min())
    return out


def matches(value, quoted):
    """Agreement to the quoted number's last decimal (half a unit)."""
    decimals = len(quoted.split(".")[1]) if "." in quoted else 0
    return abs(value - float(quoted)) <= 0.5 * 10 ** -decimals + 1e-12


def comparison(out):
    table = {key: dict(computed=out[key], quoted=q, match=matches(out[key], q)) for key, q in QUOTED.items()}
    for key, (lo, hi) in QUOTED_RANGES.items():
        half = 0.5 * 10 ** -len(lo.split(".")[1])
        table[key] = dict(computed=out[key], quoted=f"{lo}-{hi}",
                          match=float(lo) - half <= out[key] <= float(hi) + half)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", help="write the numbers and the comparison as JSON (default: print only)")
    args = parser.parse_args(argv)
    out = compute()
    table = comparison(out)
    width = max(len(k) for k in out)
    for key, value in out.items():
        note = ""
        if key in table:
            note = f"   quoted {table[key]['quoted']:>8}  {'match' if table[key]['match'] else 'DIFFERS'}"
        print(f"{key:{width}s} {value: .6f}{note}")
    differ = [k for k, v in table.items() if not v["match"]]
    print(f"\n{len(table) - len(differ)} of {len(table)} quoted numbers match"
          + (f"; differ: {differ}" if differ else ""))
    for key, note in NOTES.items():
        print(f"note {key}: {note}")
    if args.output:
        Path(args.output).write_text(json.dumps(dict(values=out, comparison=table, notes=NOTES), indent=1,
                                                allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
