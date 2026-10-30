"""Pure computations of step 0.4 (protocol Part 2 L1.5 and L2, Part 5 step 0.4; approval A3). No model or file access.

- L2.3 neighbor residual: predictors P1 (quadratic on S9), P2 (line on the five edge slots in [-KNEE, -0.69999999]),
  P3 (cubic on S9), fitted per anchor to Q_stall at the supported slots; rho_hat(slot) = min_k [P_k(slot) - Q_stall(slot)];
  the sign control uses max_k.
- The L2 PASS rule: item 5 (estimator validity) first, then items 1-4 on predictor-valid anchors, and item 6.
- The Stage 1 reachability forecast (reported): medians with non-predictor-valid valid anchors at minus infinity.
- L1.5 state typicality: 64-nearest-neighbor distance on the 14 normalized inputs, leave-one-episode-out, against a
  hash-selected reference sample of training rows, compared with its 0.99 quantile.
- The step 0.4 label in 5.0 precedence.
"""
import hashlib

import numpy as np

from core import require
from knee_region import GATED_PAST_KNEE, S9, SLOTS

EDGE5 = ("edge_knee", "edge_775", "edge_750", "edge_725", "edge_700")
ITEM5_TOL = {"s05": 0.10, "s10": 0.15, "s20": 0.30}
NEIGHBORS, QUANTILE = 64, 0.99
REFERENCE_ROWS = 640


def _fit(xs, ys, degree):
    """Least-squares polynomial coefficients per anchor; ys is (anchors, len(xs))."""
    x = np.asarray(xs, np.float64)
    design = np.stack([x ** p for p in range(degree + 1)], axis=1)
    return np.linalg.lstsq(design, np.asarray(ys, np.float64).T, rcond=None)[0].T          # (anchors, degree + 1)


def _eval(coef, x):
    return sum(coef[:, p] * float(x) ** p for p in range(coef.shape[1]))


def predictors(q_stall_s9):
    """{name: callable(slot value) -> (anchors,)} for P1, P2, P3, fitted to Q_stall at the S9 slots (anchors, 9)."""
    q = np.asarray(q_stall_s9, np.float64)
    require(q.ndim == 2 and q.shape[1] == len(S9), "Q_stall at the nine S9 slots")
    xs = np.asarray([SLOTS[n] for n in S9], np.float64)
    edge = [S9.index(n) for n in EDGE5]
    c1, c3 = _fit(xs, q, 2), _fit(xs, q, 3)
    c2 = _fit(xs[edge], q[:, edge], 1)
    return {"P1": lambda a: _eval(c1, a), "P2": lambda a: _eval(c2, a), "P3": lambda a: _eval(c3, a)}


def neighbor_residuals(q_stall_s9, q_stall_w, q_nostall_w):
    """rho_hat and item-5 errors per anchor at the past-knee slots.

    q_stall_w and q_nostall_w: {slot name: (anchors,)} for GATED_PAST_KNEE (Q_nostall needed at s05, s10, s20)."""
    preds = predictors(q_stall_s9)
    out = {"rho_hat": {}, "rho_hat_max": {}, "item5_error": {}}
    for name in GATED_PAST_KNEE:
        values = np.stack([p(SLOTS[name]) for p in preds.values()])                     # (3, anchors)
        q_stall = np.asarray(q_stall_w[name], np.float64)
        out["rho_hat"][name] = np.min(values - q_stall, axis=0)
        out["rho_hat_max"][name] = np.max(values - q_stall, axis=0)
        if name in ITEM5_TOL:
            out["item5_error"][name] = np.max(np.abs(values - np.asarray(q_nostall_w[name], np.float64)), axis=0)
    return out


def l2_rule(res, q_stall_knee, q_stall_s10, valid):
    """The L2 PASS rule on the gated anchors of one continuation seed. `valid` marks valid anchors."""
    valid = np.asarray(valid, bool)
    require(valid.any(), "No valid anchor")
    item5_ok = np.ones(len(valid), bool)
    for name, tol in ITEM5_TOL.items():
        item5_ok &= res["item5_error"][name] <= tol
    predictor_valid = valid & item5_ok
    share5 = float(np.mean(item5_ok[valid]))
    pv = predictor_valid
    items = {"item5_share": share5, "item5": share5 >= 0.90, "predictor_valid": int(pv.sum()), "valid": int(valid.sum())}
    if pv.any():
        r05, r10, r20 = (res["rho_hat"][n][pv] for n in ("s05", "s10", "s20"))
        drop = (np.asarray(q_stall_knee, np.float64) - np.asarray(q_stall_s10, np.float64))[pv]
        items.update(
            item1=bool(np.mean(r05 >= 0.4) >= 0.70 and np.median(r05) >= 0.75), item1_share=float(np.mean(r05 >= 0.4)),
            item1_median=float(np.median(r05)),
            item2=bool(np.mean(r10 >= 0.5) >= 0.70 and np.median(r10) >= 0.75), item2_share=float(np.mean(r10 >= 0.5)),
            item2_median=float(np.median(r10)),
            item3=bool(np.mean(r20 >= 1.0) >= 0.70), item3_share=float(np.mean(r20 >= 1.0)),
            item4=bool(np.mean(drop >= 0.25) >= 0.70), item4_share=float(np.mean(drop >= 0.25)))
    else:
        items.update(item1=False, item2=False, item3=False, item4=False)
    p10 = res["rho_hat_max"]["p10"][valid]
    items.update(item6=bool(np.mean(p10 < 0) >= 0.90), item6_share=float(np.mean(p10 < 0)))
    return items, predictor_valid


def reach_forecast(res, valid, predictor_valid):
    """Reported Stage 1 reachability forecast: over valid anchors, non-predictor-valid entering as minus infinity."""
    valid, pv = np.asarray(valid, bool), np.asarray(predictor_valid, bool)
    out = {"valid_anchors": int(valid.sum()), "minus_infinity": int((valid & ~pv).sum())}
    for name, bar in (("s05", 0.60), ("s10", 0.65)):
        values = np.where(pv, res["rho_hat"][name], -np.inf)[valid]
        med = float(np.median(values)) if len(values) else float("-inf")
        finite = bool(np.isfinite(med))                   # JSON-safe: a minus-infinity median is a flag, not a float
        out[name] = {"median_rho_hat": med if finite else None, "median_is_minus_infinity": not finite,
                     "bar": bar, "at_or_above_bar": bool(med >= bar)}
    return out


def reference_rows(row_ids, count=REFERENCE_ROWS):
    """Outcome-independent reference sample: the rows whose sha256('knee-edge-l15-v1:' + id) is smallest, at most one
    per episode first (ids are trajectory_id * 100 + t)."""
    ids = np.asarray(row_ids, np.int64)
    keys = np.asarray([hashlib.sha256(f"knee-edge-l15-v1:{int(i)}".encode()).hexdigest() for i in ids])
    order = np.argsort(keys, kind="stable")
    chosen, seen = [], set()
    for i in order:
        episode = int(ids[i]) // 100
        if episode not in seen:
            chosen.append(int(i))
            seen.add(episode)
        if len(chosen) == count:
            break
    return np.asarray(chosen, np.int64)


def knn_distance(queries, rows, row_episodes, exclude_episodes=None, k=NEIGHBORS, chunk=256):
    """k-th nearest-neighbor Euclidean distance from each query to `rows`, excluding rows of the query's own episode."""
    rows, queries = np.asarray(rows, np.float64), np.asarray(queries, np.float64)
    row_episodes = np.asarray(row_episodes, np.int64)
    out = np.empty(len(queries))
    rn = np.sum(rows * rows, axis=1)
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]
        d2 = np.maximum(np.sum(q * q, axis=1)[:, None] + rn[None, :] - 2.0 * q @ rows.T, 0.0)
        if exclude_episodes is not None:
            d2[np.asarray(exclude_episodes)[start:start + chunk, None] == row_episodes[None, :]] = np.inf
        out[start:start + chunk] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])
    return out


def l15(train_obs, train_episodes, train_times, train_deep, start_obs, anchor_obs, anchor_times, train_row_ids=None):
    """L1.5 for one world, as amended (PROTOCOL_AMENDMENT_1.md): each E2 state's 64-NN distance to all fitting rows,
    against the 0.99 quantile of the leave-one-episode-out 64-NN distances of training rows in its own group
    (deep-load episodes at the same time step). Bars: 95% of starts (t = 0) and 90% of gated anchors.
    The general (unamended) threshold is reported alongside, descriptive only."""
    train = np.asarray(train_obs, np.float64)
    episodes, times, deep = np.asarray(train_episodes), np.asarray(train_times), np.asarray(train_deep, bool)
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = (train - mean) / std
    thresholds = {}
    for t in sorted({0} | set(int(x) for x in np.asarray(anchor_times))):
        group = np.flatnonzero(deep & (times == t))
        require(len(group) >= NEIGHBORS, f"Group at t = {t} has too few rows")
        d = knn_distance(z[group], z, episodes, exclude_episodes=episodes[group])
        thresholds[t] = {"threshold": float(np.quantile(d, QUANTILE, method="higher")), "group_rows": int(len(group))}
    starts = knn_distance((np.asarray(start_obs, np.float64) - mean) / std, z, episodes)
    anchors = knn_distance((np.asarray(anchor_obs, np.float64) - mean) / std, z, episodes)
    a_thr = np.asarray([thresholds[int(t)]["threshold"] for t in np.asarray(anchor_times)])
    share_s = float(np.mean(starts <= thresholds[0]["threshold"]))
    share_a = float(np.mean(anchors <= a_thr))
    out = {"amendment": 1, "thresholds_by_t": thresholds, "start_share": share_s, "anchor_share": share_a,
           "pass": bool(share_s >= 0.95 and share_a >= 0.90)}
    if train_row_ids is not None:                                  # the unamended rule, reported only
        ref = reference_rows(train_row_ids)
        general = float(np.quantile(knn_distance(z[ref], z, episodes, exclude_episodes=episodes[ref]), QUANTILE, method="higher"))
        out["descriptive_general_threshold"] = {"threshold": general, "start_share": float(np.mean(starts <= general)),
                                                "anchor_share": float(np.mean(anchors <= general))}
    return out


def label(*, plumbing_ok, identity_ok, equal_ok, cross_check_ok, item5_by_seed, l14_ok, l15_ok, expected_ok,
          items_by_seed):
    """The step 0.4 label and its reasons, in 5.0 precedence."""
    incomplete = [n for n, ok in (("plumbing (4.7(i), twins, splits, a1..a3)", plumbing_ok), ("identity residual", identity_ok),
                                  ("stall = no-stall on supported slots", equal_ok), ("evaluator cross-check", cross_check_ok)) if not ok]
    if incomplete:
        return "INCOMPLETE", incomplete
    inconclusive = ([f"L2 item 5 fails for seed {s}" for s, ok in item5_by_seed.items() if not ok]
                    + ([] if l14_ok else ["L1.4 outside its bands"]) + ([] if l15_ok else ["L1.5 fails"])
                    + ([] if expected_ok else ["4.7(ii) expected-value check outside its bounds"]))
    if inconclusive:
        return "INCONCLUSIVE", inconclusive
    named = ("item1", "item2", "item3", "item4", "item6")
    fails = [f"L2 {item} fails for seed {s}" for s, items in items_by_seed.items() for item in named if not items[item]]
    if fails:
        consistent = any(all(not items[item] for items in items_by_seed.values()) for item in named)
        return ("FAIL-consistent" if consistent else "FAIL-mixed"), fails     # 5.0 subtype; "not shown model-relevant"
    return "PASS", []
