"""Edit 3 (comment C): recompute, independently of coupling_facts.py, every coupling number the new Transition
Dynamics text uses, from the environment's own matrices, and compare with notes/coupling_facts.json. Reads only.

Run (WSL):  cd /root/bca-shift && PYTHONPATH=/root/bca-shift /opt/bca-shift-venv/bin/python \
            /mnt/c/Users/David/dev/bca-shift-work/page/scripts/cc3_facts_check.py
"""
import json

import numpy as np

import experiments.shift  # noqa: F401  (must come first)
from experiments.shift import thruster_chain as T
from experiments.signal import gapped_lq as GLQ

F = json.load(open("/mnt/c/Users/David/dev/bca-shift-work/page/notes/coupling_facts.json"))
DS, NF = T.DS, 5
A13, B13 = T.matrices(T.STALL)
A = np.array(A13[:DS, :DS])
A0 = GLQ.oscillators(GLQ.STAGE0_SPEC)
C = A - A0
spec = GLQ.STAGE0_SPEC
print("coupling seed", spec.coupling_seed, "coupling norm", spec.coupling_norm)
assert spec.coupling_seed == 20261011 and spec.coupling_norm == 0.02
X = np.random.default_rng([spec.coupling_seed, 0]).standard_normal((DS, DS))
assert np.allclose(C, 0.02 * X / np.linalg.norm(X, 2), rtol=0, atol=1e-15), "C is not the first draw"
print("first draw kept; spectral radius of the first draw's A:", round(max(abs(np.linalg.eigvals(A))), 6))

P = lambda i: slice(2 * i, 2 * i + 2)
own = [np.linalg.norm(A0[P(i), P(i)], 2) for i in range(NF)]
print("C spectral norm %.6f; largest |entry| %.4f; mean |entry| %.4f; nonzero %d; own swing blocks %s; 0.02/0.96 = %.4f"
      % (np.linalg.norm(C, 2), abs(C).max(), abs(C).mean(), np.count_nonzero(C), np.round(own, 3), 0.02 / min(own)))

# design turns and shrinks (without C) and the swings with C
ev0 = sorted({(round(abs(z), 6), round(abs(np.angle(z)), 6)) for z in np.linalg.eigvals(A0)}, key=lambda t: t[1])
ev = sorted({(round(abs(z), 6), round(abs(np.angle(z)), 6)) for z in np.linalg.eigvals(A)}, key=lambda t: t[1])
half = lambda m: np.log(0.5) / np.log(m)
print("design (shrink, turn):", ev0)
print("with C (shrink, turn):", ev)
print("steps to halve, with C:", [round(half(m), 1) for m, _ in ev], "without C:", [round(half(m), 1) for m, _ in ev0])

# a push on one float: unit kick to its speed number, thrusters off, no water
tab20, tab100 = np.zeros((NF, NF)), np.zeros((NF, NF))
for j in range(NF):
    x = np.zeros(DS); x[2 * j + 1] = 1.0
    x0 = x.copy()
    for t in range(1, 101):
        x = A @ x; x0 = A0 @ x0
        for i in range(NF):
            sz = np.linalg.norm(x[P(i)])
            if i != j:
                assert np.linalg.norm(x0[P(i)]) == 0.0
            if t <= 20:
                tab20[j, i] = max(tab20[j, i], sz)
            tab100[j, i] = max(tab100[j, i], sz)
off = ~np.eye(NF, dtype=bool)
print("push table, largest swing size of each other float within 20 steps (row = float pushed), in %:")
print(np.round(100 * np.where(off, tab20, np.nan), 1))
print("range %.4f to %.4f, median %.4f; sorted: %s" % (tab20[off].min(), tab20[off].max(), np.median(tab20[off]),
                                                       np.round(np.sort(tab20[off]), 4)))
grow = [(j + 1, i + 1, round(tab20[j, i], 4), round(tab100[j, i], 4)) for j in range(NF) for i in range(NF)
        if i != j and round(tab100[j, i], 4) > round(tab20[j, i], 4)]
print("pairs that grow after 20 steps (from, to, max20, max100):", grow)
for j in range(NF):
    for i in range(NF):
        if i != j:
            assert abs(F["impulse"]["table"][f"push_on_float_{j + 1}"][i]["max_size_20"] - tab20[j, i]) < 1e-6

# a one-step thrust of 1 from each thruster
Bf = np.array(B13[:DS])
worst = 0.0
for k in range(4):
    x = Bf[:, k].copy()
    sizes = np.zeros(NF)
    for t in range(1, 101):
        for i in range(NF):
            sizes[i] = max(sizes[i], np.linalg.norm(x[P(i)]))
        x = A @ x
    mine = [i for i in range(NF) if Bf[2 * i + 1, k] != 0]
    other = max(sizes[i] for i in range(NF) if i not in mine)
    worst = max(worst, other)
    print(f"thruster {k + 1}: own floats {[i + 1 for i in mine]} moved {sizes[mine]}, any other float at most {other:.4f}")
print("worst other float %.4f = %.1f%% of 0.3" % (worst, 100 * worst / 0.3))

# no-throttle episodes from the recorded-conditions bank: share of each float's motion due to C, and the return
A13n = np.array(A13); A13n[:DS, :DS] = A0
starts = np.array(T.bank_starts("recorded"), np.float64)
print("bank starts:", starts.shape)
x, xn = starts.copy(), starts.copy()
Rw, Rn, S, Sn = [], [], [], []
for _ in range(T.STALL.length):
    x, rw = T.dynamics(x, np.zeros((len(x), T.DA)), T.STALL)
    Rw.append(rw)
    Rn.append(-np.sum(xn[:, :DS] ** 2, axis=1))       # zero throttle: the reward is minus the float cost
    xn = xn @ A13n.T
    S.append(x[:, :DS].copy()); Sn.append(xn[:, :DS].copy())
# the manual reward formula, checked against the environment on the run with C
x2, Rm = starts.copy(), []
for _ in range(T.STALL.length):
    Rm.append(-np.sum(x2[:, :DS] ** 2, axis=1))
    x2 = x2 @ np.array(A13).T
assert np.allclose(np.sum(Rm, 0), np.sum(Rw, 0), rtol=0, atol=1e-8), "manual reward differs from the environment's"
S, Sn = np.array(S), np.array(Sn)
D = S - Sn
share = [np.sqrt(np.sum(D[:, :, P(i)] ** 2) / np.sum(S[:, :, P(i)] ** 2)) for i in range(NF)]
print("share of each float's motion due to C (recorded, no throttle):", [f"{100 * v:.1f}%" for v in share],
      "all floats %.1f%%" % (100 * np.sqrt(np.sum(D ** 2) / np.sum(S ** 2))))
rw, rn = np.sum(Rw, 0), np.sum(Rn, 0)
print("mean return with C %.2f, without C %.2f, change %.2f%%, largest single-episode change %.1f%%"
      % (rw.mean(), rn.mean(), 100 * abs(rw.mean() - rn.mean()) / abs(rn.mean()), 100 * np.max(abs(rw - rn) / abs(rn))))

# reachability
direct = {i + 1: [k + 1 for k in range(4) if Bf[2 * i + 1, k] != 0] for i in range(NF)}
print("thrusters pushing each float directly:", direct)
def rank(M):
    K = np.hstack([np.linalg.matrix_power(M, n) @ Bf for n in range(DS)])
    return np.linalg.matrix_rank(K)
print("the four thrusters steer all ten float numbers: rank with C", rank(A), "without C", rank(A0))
v = np.array(T.push_pattern())
print("water pushes every speed number:", bool(np.all(v[1::2] == 1.0)), "and no position number:", bool(np.all(v[0::2] == 0.0)))

# the wave's effect on float 2, per unit of amplitude, with and without C: simulate a unit wave with no current and no
# throttle for 3,000 steps from rest and take the largest swing size of each float over the last 500 steps
A13c = np.array(A13)
turn = T.STALL.wave_turn
for name, Af in (("with C", A), ("without C", A0)):
    s = np.zeros(DS); w = np.array([1.0, 0.0]); g = T.STALL.gain
    peak = np.zeros(NF)
    for t in range(3000):
        s = Af @ s + g * v[:DS] * w[0]
        w = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]) @ w
        if t >= 2500:
            peak = np.maximum(peak, [np.linalg.norm(s[P(i)]) for i in range(NF)])
    print("wave per unit amplitude,", name, np.round(peak, 2))
