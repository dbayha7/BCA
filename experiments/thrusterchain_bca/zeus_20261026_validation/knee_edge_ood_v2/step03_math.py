"""Pure computations of step 0.3 (protocol Part 5, step 0.3; approval A2). No model, no simulator, no file access.

Every gated number of step 0.3 is computed here from arrays, so it can be tested on synthetic values:
- R_lin, the steep-probe selector, from the SAVED per-probe Q^pi at the slots -0.5 and -0.25;
- F, the fold error at full throttle, from Q1 at -1 and -0.5 (a1..a3 = 0);
- the onset contrast ratio (warning only);
- the reach forecast: edge slope, curvature, combined slope, equilibrium pi0 and the forecast fraction;
- the label of the step, in the 5.0 precedence INCOMPLETE > INCONCLUSIVE > FAIL > PASS.
"""
import numpy as np

from core import require
from knee_region import KNEE, SLOTS
from probes import G05_POINTS, REACH_POINTS

T15_M1 = -(KNEE - 1.5 * (1.0 - KNEE))                   # T15(-1) = -0.5000000298...
FOLD_CONSTANT = 0.3 * (1.0 - T15_M1 ** 2)               # 0.22499999..., the protocol's 0.22499999
F_BAR = 0.5
R_LIN_BAR = 1.0
FLAG_SHARE, STOP_SHARE = 0.5, 0.25
ONSET_FLAG = 0.3
M_ANCHOR = -0.7263                                      # CUTOFF's high-load BC anchor (Part 3.7)
LAMBDA_SCALE = 0.751                                    # lambda correction for the new reset law
DEEP_EDGE = -(KNEE + 0.10)
PROBE_BAR, VISITED_BAR = 0.35, 0.25
EDGE_POINTS = (float(np.float32(-0.5)), float(np.float32(-0.7)))          # -0.5 and -0.69999999
EDGE_SPAN = EDGE_POINTS[0] - EDGE_POINTS[1]                                # 0.19999998807907104
QUAD_POINTS = tuple(float(np.float32(x)) for x in (-0.5, -0.25, 0.0, 0.25, 0.5))
FOUR_POINTS = (float(np.float32(-0.7)), float(np.float32(-0.6)), -0.5, -0.25)          # descriptive only
SIX_POINTS = (float(np.float32(-0.7)), float(np.float32(-0.6)), -0.5, -0.25, 0.0, 0.25)  # descriptive only


def _column(points, value):
    """Index of an exact float32 point in a point list."""
    bits = np.asarray(points, np.float32).view(np.uint32)
    hit = np.flatnonzero(bits == np.asarray([value], np.float32).view(np.uint32)[0])
    require(len(hit) == 1, f"Point {value} not in the list")
    return int(hit[0])


def r_lin(saved_qpi, saved_actions):
    """R_lin = 2 [Q^pi(-0.5) - Q^pi(-0.25)] + 0.22499999, from the saved per-probe values (slots 1 and 2)."""
    saved_qpi, saved_actions = np.asarray(saved_qpi, np.float64), np.asarray(saved_actions, np.float32)
    require(saved_qpi.shape[1:] == (6,) and saved_actions.shape[1:] == (6, 4), "Saved per-probe layout (n, 6)")
    require(bool(np.all(saved_actions[:, 1, 0] == np.float32(-0.5))) and bool(np.all(saved_actions[:, 2, 0] == np.float32(-0.25))),
            "Saved slots 1 and 2 are not -0.5 and -0.25")
    require(bool(np.all(saved_actions[:, 1:, 1:] == 0)), "Saved slots carry a1..a3 = 0")
    return 2.0 * (saved_qpi[:, 1] - saved_qpi[:, 2]) + FOLD_CONSTANT


def steep(saved_qpi, saved_actions, high_load):
    return (r_lin(saved_qpi, saved_actions) >= R_LIN_BAR) & np.asarray(high_load, bool)


def fold_error(q1_g05):
    """F(s) = Q1(s, -1) - Q1(s, -0.5) + 0.22499999, from Q1 at the G0.5 points (rows, 7) with a1..a3 = 0."""
    q = np.asarray(q1_g05, np.float64)
    require(q.ndim == 2 and q.shape[1] == len(G05_POINTS), "Q1 at the seven G0.5 points")
    return q[:, _column(G05_POINTS, -1.0)] - q[:, _column(G05_POINTS, -0.5)] + FOLD_CONSTANT


def g05_seed_summary(f_values):
    f = np.asarray(f_values, np.float64)
    require(f.ndim == 1 and len(f) > 0, "G0.5 needs at least one steep ordinary probe")
    return {"probes": int(len(f)), "median_F": float(np.median(f)), "share_F_le_0.5": float(np.mean(f <= F_BAR)),
            "mean_abs_F": float(np.mean(np.abs(f)))}


def onset_ratio(q1_s05, q1_knee, qpi15_s05, qpi15_knee, qpi_nostall_s05):
    kc = (np.asarray(q1_s05, np.float64) - np.asarray(q1_knee, np.float64)) - \
         (np.asarray(qpi15_s05, np.float64) - np.asarray(qpi15_knee, np.float64))
    rho = np.asarray(qpi_nostall_s05, np.float64) - np.asarray(qpi15_s05, np.float64)
    require(len(kc) > 0 and kc.shape == rho.shape, "Onset contrast inputs")
    denom = float(np.median(rho))
    return {"median_abs_KC15_s05": float(np.median(np.abs(kc))), "median_rho15_s05": denom,
            "ratio": float(np.median(np.abs(kc)) / denom) if denom > 0 else None, "rows": int(len(kc))}


def _fit_slope(points, values):
    a = np.asarray(points, np.float64)
    design = np.stack([np.ones_like(a), a], axis=1)
    return np.linalg.lstsq(design, np.asarray(values, np.float64).T, rcond=None)[0][1]


def reach_rows(q1_reach, lam):
    """Equilibrium pi0 per row from Q1 at the 11 reach points (rows, 11), all at the actor's own a1..a3.

    s_edge = [Q1(-0.5) - Q1(-0.69999999)] / 0.19999999; q2 from the quadratic fit on {-0.5, -0.25, 0, 0.25, 0.5};
    q1' = s_edge + 1.2 q2; pi0 = clip((m + 2 lambda' q1') / (1 - 4 lambda' q2), -1, 1) with m = -0.7263 and
    lambda' = lam x 0.751; a row with 1 - 4 lambda' q2 <= 0 reaches -1."""
    q = np.asarray(q1_reach, np.float64)
    require(q.ndim == 2 and q.shape[1] == len(REACH_POINTS), "Q1 at the eleven reach points")
    require(np.isfinite(lam) and lam > 0, "Saved lambda must be positive")
    col = {p: _column(REACH_POINTS, p) for p in set(EDGE_POINTS + QUAD_POINTS + FOUR_POINTS + SIX_POINTS)}
    s_edge = (q[:, col[EDGE_POINTS[0]]] - q[:, col[EDGE_POINTS[1]]]) / EDGE_SPAN
    a = np.asarray(QUAD_POINTS, np.float64)
    design = np.stack([np.ones_like(a), a, a * a], axis=1)
    coef = np.linalg.lstsq(design, q[:, [col[p] for p in QUAD_POINTS]].T, rcond=None)[0]
    q2 = coef[2]
    lam_prime = lam * LAMBDA_SCALE
    q1p = s_edge + 1.2 * q2
    denom = 1.0 - 4.0 * lam_prime * q2
    pi0 = np.where(denom > 0, np.clip((M_ANCHOR + 2.0 * lam_prime * q1p) / np.where(denom > 0, denom, 1.0), -1.0, 1.0), -1.0)
    return {"pi0": pi0, "s_edge": s_edge, "q2": q2, "q1_prime": q1p, "lambda_prime": lam_prime,
            "nonpositive_denominator_rows": int(np.sum(denom <= 0)),
            "descriptive_four_point_slope": _fit_slope(FOUR_POINTS, q[:, [col[p] for p in FOUR_POINTS]]),
            "descriptive_six_point_slope": _fit_slope(SIX_POINTS, q[:, [col[p] for p in SIX_POINTS]])}


def forecast(pi0):
    pi0 = np.asarray(pi0, np.float64)
    require(pi0.ndim == 1 and len(pi0) > 0, "Forecast needs rows")
    return float(np.mean(pi0 <= DEEP_EDGE))


def label(*, known_answer_ok, caps_ok, g05_by_seed, probe_forecast_by_cell, visited_forecast_by_cell, onset_by_seed):
    """The step 0.3 label and its reasons, in 5.0 precedence. `onset_by_seed` values may be None ("not computed")."""
    reasons = []
    if not known_answer_ok or not caps_ok:
        reasons += (["known-answer mismatch"] if not known_answer_ok else []) + (["gated cap exhausted"] if not caps_ok else [])
        return "INCOMPLETE", reasons, False
    require(len(g05_by_seed) == 3 and len(probe_forecast_by_cell) == 9 and len(visited_forecast_by_cell) == 9,
            "Three seeds and nine cells required")
    shares = {s: v["share_F_le_0.5"] for s, v in g05_by_seed.items()}
    if any(x < STOP_SHARE for x in shares.values()):
        return "INCONCLUSIVE", [f"under 25% of steep ordinary probes have F <= 0.5 in seed {s}" for s, x in shares.items()
                                if x < STOP_SHARE], False
    low_probe = {c: f for c, f in probe_forecast_by_cell.items() if f < PROBE_BAR}
    low_visit = {c: f for c, f in visited_forecast_by_cell.items() if f < VISITED_BAR}
    if low_probe or low_visit:
        return "FAIL (reach)", [f"probe forecast {f:.3f} < 0.35 in {c}" for c, f in low_probe.items()] + \
               [f"visited forecast {f:.3f} < 0.25 in {c}" for c, f in low_visit.items()], False
    flagged = False
    for s, v in g05_by_seed.items():
        # The pass row (median F <= 0.5) and the flag row (25-50% have F <= 0.5) overlap at a share of exactly 0.5;
        # the conservative reading flags it: flagged when the share is <= 0.5 or the median (even count: the
        # average of the two middle values) is > 0.5. A flag only adds the written prediction at A3.
        if v["median_F"] > F_BAR or v["share_F_le_0.5"] <= FLAG_SHARE:
            flagged = True
            reasons.append(f"seed {s}: median F {v['median_F']:.3f}, share F <= 0.5 is {v['share_F_le_0.5']:.3f}")
    for s, v in onset_by_seed.items():
        if v is not None and v.get("ratio") is not None and v["ratio"] > ONSET_FLAG:
            flagged = True
            reasons.append(f"onset-contrast ratio {v['ratio']:.3f} > 0.3 in seed {s}")
    return ("PASS (flagged)" if flagged else "PASS"), reasons, flagged
