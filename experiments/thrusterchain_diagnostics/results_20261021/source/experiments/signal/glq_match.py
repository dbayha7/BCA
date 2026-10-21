"""Blind strength matching for the Gapped LQ world's Tier A (graft g6): knobs are chosen from strengths alone.

What it is. Tier A (experiments/signal/glq_train.py) trains every arm of a knob family on one frozen critic and must
then compare families at the same realized policy movement. This module picks the knob of each family at each target
strength. It is deliberately the only code that makes that choice, and it imports nothing but the standard library and
NumPy: no world, no exact truth, no return or value function (test_glq_train checks its import statements). The
caller hands it strengths and an evaluator that turns knobs into strengths; returns are computed elsewhere, after the
chosen knobs and this file's SHA-256 (source_sha256) are recorded (glq_train.tier_a writes them before scoring).

Strength (STEP4_DESIGN.md section 5, placement_harness.strength): S = sqrt(sum_c tr(delta_c Sigma_c delta_c^T)),
delta_c = K_c(arm) - K_c(NONE) for per-context gains K of shape (C, da, ds), Sigma_c = (1/n) sum_{i in c} s_i s_i^T
over the evaluation states (second_moments; the blocks sum to the pooled second moment). S is the RMS action
displacement from NONE on the evaluation states. strength and second_moments reproduce placement_harness's functions
(tested equal); they are restated here so that this module imports no evaluation code.

Matching (placement_harness.match_knob's rule, run for many (family, target) pairs in parallel rounds). Each family has
an ascending knob ladder whose strengths are known; knob 0 (the NONE arm, S = 0 exactly) may lead the ladder. Per
target S*: the first ladder knob with |S / S* - 1| <= tol is taken as it is (converged); otherwise the first adjacent
pair whose S values straddle S* is the bracket; a target outside every bracket is unreachable and never extrapolated.
In a bracket, round 1 proposes the log-linear interpolation (linear when the bracket starts at knob 0), later rounds
the geometric midpoint (arithmetic from 0). All pending proposals of a round are evaluated together by the caller's
evaluator (one vmapped Tier A program per round). A nonfinite S counts as overshooting on the low side, as in
placement_harness. A pair stops when |S / S* - 1| <= tol or after max_rounds; the best evaluated knob is reported with
converged, reachable, monotone (the ladder's finite S values are monotone) and the evaluation count.
"""

import hashlib
import math
from pathlib import Path

import numpy as np

MATCH_TOL = 0.01  # |S / S* - 1| at which a pair stops; R7(d)'s mark is 5%
MATCH_ROUNDS = 5


def source_sha256():
    """SHA-256 of this file's bytes: recorded with the matched knobs, before any return is computed."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def second_moments(states, context, n_contexts):
    """(C, ds, ds) blocks (1/n) sum_{i in c} s_i s_i^T (placement_harness.second_moment with contexts)."""
    s = np.asarray(states, np.float64)
    context = np.asarray(context)
    return np.stack([s[context == c].T @ s[context == c] / len(s) for c in range(n_contexts)])


def strength(K_arm, K_none, Sigma):
    """S = sqrt(sum_c tr(delta_c Sigma_c delta_c^T)), delta = K_arm - K_none; batched over leading axes of K_arm."""
    d = np.asarray(K_arm, np.float64) - np.asarray(K_none, np.float64)
    S = np.asarray(Sigma, np.float64)
    if S.ndim == 2:
        return np.sqrt(np.einsum("...ij,jk,...ik->...", d, S, d))
    return np.sqrt(np.einsum("...cij,cjk,...cik->...", d, S, d))


def _within(s, target, tol):
    return bool(np.isfinite(s) and abs(s / target - 1) <= tol)


def start(knobs, S, target, tol=MATCH_TOL):
    """One (family, target) pair after the ladder: converged at a ladder knob, an open bracket, or unreachable."""
    knobs = np.asarray(knobs, np.float64)
    S = np.asarray(S, np.float64)
    if np.any(np.diff(knobs) <= 0) or knobs[0] < 0:
        raise ValueError("a knob ladder must be ascending and nonnegative")
    if not target > 0:
        raise ValueError("the target strength must be positive")
    fin = S[np.isfinite(S)]
    monotone = bool(np.all(np.diff(fin) >= 0) or np.all(np.diff(fin) <= 0))
    state = dict(target=float(target), tol=tol, knob=None, S=None, reachable=False, converged=False,
                 monotone=monotone, evaluations=0, ladder_knobs=knobs.tolist(), ladder_S=S.tolist(), bracket=None,
                 done=True, history=[])
    for j in range(len(knobs)):
        if _within(S[j], target, tol):
            return dict(state, knob=float(knobs[j]), S=float(S[j]), reachable=True, converged=True, bracket=[j, j])
        if j + 1 < len(knobs) and np.all(np.isfinite(S[j:j + 2])) and (S[j] - target) * (S[j + 1] - target) < 0:
            return dict(state, bracket=[j, j + 1], lo=float(knobs[j]), hi=float(knobs[j + 1]), s_lo=float(S[j]),
                        s_hi=float(S[j + 1]), done=False, best=None)
    return state


def propose(state, interpolate=True):
    """The next knob of an open bracket: interpolation in round 1 (log-linear; linear from knob 0), then midpoints."""
    lo, hi, s_lo, s_hi, target = state["lo"], state["hi"], state["s_lo"], state["s_hi"], state["target"]
    if interpolate and state["evaluations"] == 0:
        frac = (target - s_lo) / (s_hi - s_lo)
        if lo == 0:
            return float(lo + frac * (hi - lo))
        return float(np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))))
    return float((lo + hi) / 2) if lo == 0 else float(math.sqrt(lo * hi))


def update(state, knob, s, max_rounds=MATCH_ROUNDS):
    """Record one evaluation (knob, S) of an open bracket; close it when within tol or out of rounds."""
    state = dict(state)
    state["evaluations"] += 1
    state["history"] = state["history"] + [[float(knob), float(s) if np.isfinite(s) else None]]
    target, tol = state["target"], state["tol"]
    best = state.get("best")
    if np.isfinite(s) and (best is None or abs(s / target - 1) < abs(best[1] / target - 1)):
        state["best"] = best = (float(knob), float(s))
    if not np.isfinite(s) or (s - target) * (state["s_lo"] - target) > 0:
        state["lo"], state["s_lo"] = float(knob), (float(s) if np.isfinite(s) else state["s_lo"])
    else:
        state["hi"], state["s_hi"] = float(knob), float(s)
    if _within(s, target, tol) or state["evaluations"] >= max_rounds:
        state["done"] = True
    if best is not None:
        state.update(knob=best[0], S=best[1], reachable=True, converged=_within(best[1], target, tol))
    return state


def match_parallel(ladders, targets, evaluate, tol=MATCH_TOL, max_rounds=MATCH_ROUNDS, interpolate=True, log=None):
    """Match every family of `ladders` ({name: (knobs, S)}) to every target of `targets` ({tag: S*}).

    evaluate(list of (name, knob)) -> list of S, called once per round with every open pair's proposal (the caller
    runs them as one batch). Returns ({(name, tag): state}, rounds run)."""
    states = {(name, tag): start(knobs, S, target, tol)
              for name, (knobs, S) in ladders.items() for tag, target in targets.items()}
    rounds = 0
    while rounds < max_rounds:
        open_pairs = [key for key, st in states.items() if not st["done"]]
        if not open_pairs:
            break
        proposals = [(key[0], propose(states[key], interpolate)) for key in open_pairs]
        values = evaluate(proposals)
        if len(values) != len(proposals):
            raise ValueError("the evaluator must return one strength per proposal")
        for key, (_, knob), s in zip(open_pairs, proposals, values):
            states[key] = update(states[key], knob, float(s), max_rounds)
        rounds += 1
        if log:
            log(f"matching round {rounds}: {len(open_pairs)} open pairs")
    for key, st in states.items():
        st.pop("best", None)
        st["done"] = True
    return states, rounds


def public(state):
    """The strict-JSON record of a matching state."""
    keep = ("target", "tol", "knob", "S", "reachable", "converged", "monotone", "evaluations", "ladder_knobs",
            "ladder_S", "bracket", "history")
    out = {k: state.get(k) for k in keep}
    out["ladder_S"] = [float(x) if x is not None and np.isfinite(x) else None for x in out["ladder_S"]]
    out["relative_error"] = (abs(state["S"] / state["target"] - 1) if state.get("S") is not None else None)
    return out
