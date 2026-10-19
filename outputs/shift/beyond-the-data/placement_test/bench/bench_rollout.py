"""Micro-benchmark of the exact evaluator's ARITHMETIC on MADE-UP arrays (no ThrusterChain matrices, banks or data; no
repository import).

Restates the per-step work of experiments/shift/truth.rollout (truth.py:77-95) with thruster_chain.dynamics
(thruster_chain.py:201-221: fixed-order column sums over 13 state columns and 4 thrust columns, the stall thrust,
reward) and truth.MLPPolicy.__call__ (truth.py:241-248: float32 rounding of the observation, normalisation, 13-256-256-4
ReLU MLP in float64, tanh), on a random stable 13x13 matrix and random weights. Also the capped rollout
(recorder.CappedController: clip to +-0.8) that evaluate.score / learners.evaluate_policy run beside it, and
Q^pi(s, a) by a 1,000-step float64 rollout (truth.q_value, truth.py:98-108) for a batch of (s, a).

Usage: python -B bench_rollout.py
"""
import time

import numpy as np

rs = np.random.default_rng(7)
DX, DA = 13, 4
M = rs.standard_normal((DX, DX))
A = 0.97 * M / np.max(np.abs(np.linalg.eigvals(M)))
B = 0.3 * rs.standard_normal((DX, DA))
acols = tuple(A[:, j].copy() for j in range(DX))
bcols = tuple(B[:, k].copy() for k in range(DA))
W = [rs.standard_normal((13, 256)) / 4, rs.standard_normal((256, 256)) / 16, rs.standard_normal((256, 4)) / 16]
b = [np.zeros(256), np.zeros(256), np.zeros(4)]
mean, std = np.zeros(13), np.ones(13)


def policy(x):
    o = np.asarray(x, np.float64).astype(np.float32).astype(np.float64)
    h = (o - mean) / std
    h = np.maximum(h @ W[0] + b[0], 0.0)
    h = np.maximum(h @ W[1] + b[1], 0.0)
    return np.tanh(h @ W[2] + b[2])


def thrust(a, knee=0.8, slope=1.5):
    return np.clip(a, -knee, knee) - slope * np.sign(a) * np.maximum(np.abs(a) - knee, 0.0)


def dynamics(x, a):
    a = np.clip(a, -1.0, 1.0)
    th = thrust(a)
    nxt = x[..., 0:1] * acols[0]
    for j in range(1, DX):
        nxt = nxt + x[..., j:j + 1] * acols[j]
    for k in range(DA):
        nxt = nxt + th[..., k:k + 1] * bcols[k]
    s2 = x[..., 0] * x[..., 0]
    for i in range(1, 10):
        s2 = s2 + x[..., i] * x[..., i]
    a2 = a[..., 0] * a[..., 0]
    for k in range(1, DA):
        a2 = a2 + a[..., k] * a[..., k]
    return np.clip(nxt, -50, 50), -(s2 + 0.3 * a2)


def rollout(pol, x0, L=100, record=True):
    x = x0.copy()
    J = np.zeros(len(x))
    S, C, R = [x], [], []
    for _ in range(L):
        a = np.clip(pol(x), -1.0, 1.0)
        x2, r = dynamics(x, a)
        J = J + r
        if record:
            S.append(x2), C.append(a), R.append(r)
        x = x2
    return J


def q_value(x, a, horizon=1000, g=0.98):
    xx, q = dynamics(x, a)
    disc = 1.0
    for _ in range(1, horizon):
        disc *= g
        xx, rr = dynamics(xx, policy(xx))
        q = q + disc * rr
    return q


def timed(fn, *a, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


x0 = rs.standard_normal((1000, DX)) * 0.5
t_bank = timed(rollout, policy, x0)
t_capped = timed(rollout, lambda x: np.clip(policy(x), -0.8, 0.8), x0, 100, False)
t_200 = timed(rollout, policy, x0[:200])
t_one = timed(rollout, policy, x0[:1], reps=5)
xs, as_ = rs.standard_normal((1000, DX)) * 0.5, np.tanh(rs.standard_normal((1000, DA)))
t_q = timed(q_value, xs, as_, 100, reps=1) * 10.0       # 100 steps timed, x10 for the 1,000-step horizon
t_one_step = timed(lambda: dynamics(xs, as_), reps=5)
print(f"bank rollout, 1,000 starts x 100 steps, recorded: {t_bank:.3f} s")
print(f"capped rollout, 1,000 starts x 100 steps: {t_capped:.3f} s")
print(f"=> one score() of one policy on one bank (rollout + capped): {t_bank + t_capped:.3f} s")
print(f"200 starts x 100 steps: {t_200:.3f} s;  1 start x 100 steps (online episode, batch 1): {t_one * 1000:.1f} ms")
print(f"Q^pi by 1,000-step rollout for 1,000 (s, a) pairs: {t_q:.2f} s (from 100 steps x 10)")
print(f"exact one-step target arithmetic (dynamics only) for 1,000 rows: {t_one_step * 1000:.2f} ms")
