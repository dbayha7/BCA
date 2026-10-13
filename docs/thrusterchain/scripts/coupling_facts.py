"""Coupling facts for the Transition Dynamics paragraph (comment C), computed with the environment's own matrices.

Run (WSL):  cd /root/bca-shift && PYTHONPATH=/root/bca-shift /opt/bca-shift-venv/bin/python \
            /mnt/c/Users/David/dev/bca-shift-work/page/scripts/coupling_facts.py
Writes /mnt/c/Users/David/dev/bca-shift-work/page/notes/coupling_facts.json and prints a summary.

Definitions used below.
- A10: the 10 x 10 float block of thruster_chain.matrices()[0] (the environment's own A).
- A0: gapped_lq.oscillators(STAGE0_SPEC), the five fading swings alone (block diagonal).
- C = A10 - A0, checked against a fresh draw of gapped_lq.transition_matrix's recipe.
- "Swing size" of float i at step t: the length of its pair (position_i, speed_i), sqrt(p_i^2 + v_i^2).
- A push on float j: a unit kick to its speed number (the coordinate the thrusters and the water push).
"""

import json
import math

import numpy as np

import experiments.shift  # noqa: F401  (must come first)
from experiments.shift import thruster_chain as T
from experiments.signal import gapped_lq as GLQ

OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/coupling_facts.json"
NF = 5
DS = T.DS


def pair(i):
    return slice(2 * i, 2 * i + 2)


def r(x, n=6):
    return float(round(float(x), n))


facts = {}

# ------------------------------------------------------------------ the matrices
A13, B13 = T.matrices(T.STALL)
A10 = np.array(A13[:DS, :DS])
A0 = GLQ.oscillators(GLQ.STAGE0_SPEC)
C = A10 - A0
A_check, attempt = GLQ.transition_matrix(GLQ.STAGE0_SPEC)
assert np.array_equal(A_check, A10), "env A differs from gapped_lq.transition_matrix(STAGE0_SPEC)"
X = np.random.default_rng([GLQ.STAGE0_SPEC.coupling_seed, attempt]).standard_normal((DS, DS))
C_direct = GLQ.STAGE0_SPEC.coupling_norm * X / np.linalg.norm(X, 2)
assert np.allclose(C, C_direct, atol=1e-15, rtol=0), np.abs(C - C_direct).max()

blk = np.array([[np.linalg.norm(C[pair(i), pair(j)], 2) for j in range(NF)] for i in range(NF)])
diag_mask = np.eye(NF, dtype=bool)
own_blocks = [np.linalg.norm(A0[pair(i), pair(i)], 2) for i in range(NF)]

ev = np.linalg.eigvals(A10)
ev0 = np.linalg.eigvals(A0)


def swings(evals):
    """One (modulus, angle) per conjugate pair, sorted by angle."""
    out = sorted({(round(abs(z), 12), round(abs(np.angle(z)), 12)) for z in evals}, key=lambda t: t[1])
    return [dict(shrink=r(m, 6), turn=r(a, 6), steps_to_halve=r(math.log(0.5) / math.log(m), 3),
                 period=r(2 * math.pi / a, 3)) for m, a in out]


# eigenvector concentration: share of each swing (eigenvector, squared length) in its own float's pair
w, V = np.linalg.eig(A10)
conc = []
for k in range(DS):
    v = V[:, k]
    e = np.array([np.sum(np.abs(v[pair(i)]) ** 2) for i in range(NF)])
    e = e / e.sum()
    conc.append((abs(np.angle(w[k])), int(np.argmax(e)), float(e.max())))
conc = sorted({(round(a, 9), i, round(s, 9)) for a, i, s in conc})

facts["matrices"] = dict(
    coupling_attempt=int(attempt),
    C_spectral_norm=r(np.linalg.norm(C, 2), 8),
    C_largest_entry_abs=r(np.abs(C).max(), 6),
    C_largest_entry_at=[int(t) for t in np.unravel_index(np.abs(C).argmax(), C.shape)],
    C_mean_entry_abs=r(np.abs(C).mean(), 6),
    C_rms_entry=r(np.sqrt((C ** 2).mean()), 6),
    C_frobenius=r(np.linalg.norm(C), 6),
    C_nonzero_entries=int(np.count_nonzero(C)),
    C_block_norms_rows_to_cols_from=[[r(x, 5) for x in row] for row in blk],
    C_largest_cross_block=r(blk[~diag_mask].max(), 5),
    C_largest_cross_block_at=[int(t) + 1 for t in np.unravel_index(np.where(~diag_mask, blk, -1).argmax(), blk.shape)],
    C_smallest_cross_block=r(blk[~diag_mask].min(), 5),
    C_largest_own_block=r(blk[diag_mask].max(), 5),
    own_swing_block_norms=[r(x, 5) for x in own_blocks],
    ratio_C_to_smallest_own_block=r(np.linalg.norm(C, 2) / min(own_blocks), 5),
    spectral_radius_with_C=r(np.max(np.abs(ev)), 6),
    spectral_radius_without_C=r(np.max(np.abs(ev0)), 6),
    swings_with_C=swings(ev),
    swings_without_C=swings(ev0),
    own_pair_share_of_each_swing=[dict(turn=r(a, 4), float=i + 1, share=r(s, 6)) for a, i, s in conc],
)

# ------------------------------------------------------------------ impulse responses
def powers(M, n):
    out, P = [], np.eye(DS)
    for _ in range(n):
        P = M @ P
        out.append(P.copy())
    return out


PW = powers(A10, 100)
PW0 = powers(A0, 100)
assert all(np.abs(P0[pair(i), pair(j)]).max() == 0 for P0 in PW0 for i in range(NF) for j in range(NF) if i != j)

imp = {}
for j in range(NF):
    e = np.zeros(DS)
    e[2 * j + 1] = 1.0                      # a unit kick to float j's speed
    traj = np.array([P @ e for P in PW])    # steps 1..100
    traj0 = np.array([P @ e for P in PW0])
    rows = []
    for i in range(NF):
        size = np.linalg.norm(traj[:, pair(i)], axis=1)
        size0 = np.linalg.norm(traj0[:, pair(i)], axis=1)
        diff = np.linalg.norm(traj[:, pair(i)] - traj0[:, pair(i)], axis=1)
        # worst direction inside float j's pair: spectral norm of the (i, j) block of A^t
        worst = np.array([np.linalg.norm(P[pair(i), pair(j)], 2) for P in PW])
        rows.append(dict(
            float=i + 1,
            max_size_20=r(size[:20].max(), 6), step_of_max_20=int(size[:20].argmax()) + 1,
            max_size_100=r(size.max(), 6), step_of_max_100=int(size.argmax()) + 1,
            size_at_20=r(size[19], 6), size_at_100=r(size[99], 6),
            max_size_20_without_C=r(size0[:20].max(), 6), max_size_100_without_C=r(size0.max(), 6),
            max_change_by_C_20=r(diff[:20].max(), 6), max_change_by_C_100=r(diff.max(), 6),
            worst_direction_max_20=r(worst[:20].max(), 6), worst_direction_max_100=r(worst.max(), 6),
        ))
    imp[f"push_on_float_{j + 1}"] = rows

cross20 = [(row["max_size_20"], j + 1, row["float"]) for j in range(NF)
           for row in imp[f"push_on_float_{j + 1}"] if row["float"] != j + 1]
cross100 = [(row["max_size_100"], j + 1, row["float"]) for j in range(NF)
            for row in imp[f"push_on_float_{j + 1}"] if row["float"] != j + 1]
own_change20 = [(imp[f"push_on_float_{j + 1}"][j]["max_change_by_C_20"], j + 1) for j in range(NF)]
own_change100 = [(imp[f"push_on_float_{j + 1}"][j]["max_change_by_C_100"], j + 1) for j in range(NF)]
facts["impulse"] = dict(
    note="unit kick to float j's speed at step 0; swing size of float i = |(p_i, v_i)|; without C every cross "
         "entry is exactly 0",
    table=imp,
    largest_cross_within_20=dict(size=max(cross20)[0], from_float=max(cross20)[1], to_float=max(cross20)[2]),
    smallest_cross_within_20=dict(size=min(cross20)[0], from_float=min(cross20)[1], to_float=min(cross20)[2]),
    median_cross_within_20=r(np.median([c[0] for c in cross20]), 6),
    largest_cross_within_100=dict(size=max(cross100)[0], from_float=max(cross100)[1], to_float=max(cross100)[2]),
    smallest_cross_within_100=dict(size=min(cross100)[0], from_float=min(cross100)[1], to_float=min(cross100)[2]),
    median_cross_within_100=r(np.median([c[0] for c in cross100]), 6),
    largest_change_to_pushed_float_20=dict(size=max(own_change20)[0], float=max(own_change20)[1]),
    largest_change_to_pushed_float_100=dict(size=max(own_change100)[0], float=max(own_change100)[1]),
    propagator_gap_20=r(np.linalg.norm(PW[19] - PW0[19], 2), 6),
    propagator_norm_20_without_C=r(np.linalg.norm(PW0[19], 2), 6),
    propagator_gap_100=r(np.linalg.norm(PW[99] - PW0[99], 2), 6),
    propagator_norm_100_without_C=r(np.linalg.norm(PW0[99], 2), 6),
)

# ------------------------------------------------------------------ steady current and the wave, per unit
v = T.push_pattern()
g = T.STALL.gain
ss = np.linalg.solve(np.eye(DS) - A10, g * v)
ss0 = np.linalg.solve(np.eye(DS) - A0, g * v)
facts["steady_current_per_unit"] = [
    dict(float=i + 1, offset_with_C=r(np.linalg.norm(ss[pair(i)]), 5),
         offset_without_C=r(np.linalg.norm(ss0[pair(i)]), 5),
         change_by_C=r(np.linalg.norm(ss[pair(i)] - ss0[pair(i)]), 5),
         position_with_C=r(ss[2 * i], 5), position_without_C=r(ss0[2 * i], 5))
    for i in range(NF)]

# forced response to a unit-amplitude wave: complex steady state of s' = A s + g v w1, w1 = Re(e^{i 0.27 t})
z = np.exp(1j * T.STALL.wave_turn)
fw = np.linalg.solve(z * np.eye(DS) - A10, g * v)
fw0 = np.linalg.solve(z * np.eye(DS) - A0, g * v)


def peak(cvec):
    # largest |(p_i, v_i)| over the cycle for s_t = Re(cvec e^{i phi}); sample the cycle finely
    ph = np.linspace(0, 2 * np.pi, 20001)
    re = np.real(np.outer(np.exp(1j * ph), cvec))
    return np.linalg.norm(re, axis=1).max()


facts["wave_per_unit_amplitude"] = [
    dict(float=i + 1, peak_with_C=r(peak(fw[pair(i)]), 4), peak_without_C=r(peak(fw0[pair(i)]), 4),
         peak_change_by_C=r(peak(fw[pair(i)] - fw0[pair(i)]), 4))
    for i in range(NF)]

# ------------------------------------------------------------------ no-throttle episodes from the evaluation banks
A13_noC = np.array(A13)
A13_noC[:DS, :DS] = A0


def rollout(starts, use_env):
    x = np.array(starts, np.float64)
    a = np.zeros((len(x), T.DA))
    states, rewards = [x[:, :DS].copy()], []
    for _ in range(T.STALL.length):
        if use_env:
            x, rew = T.dynamics(x, a, T.STALL)
        else:
            s2 = np.sum(x[:, :DS] ** 2, axis=1)
            rew = -(s2 + 0.0)
            x = x @ A13_noC.T
        rewards.append(rew)
        states.append(x[:, :DS].copy())
    return np.array(states), np.array(rewards)   # states (101, n, 10): s_0..s_100; rewards (100, n)


# check: our linear rollout with the env's A reproduces dynamics() at zero throttle
_st = T.bank_starts("recorded", 50)
xa, xb = np.array(_st), np.array(_st)
for _ in range(100):
    xa, _ = T.dynamics(xa, np.zeros((50, 4)), T.STALL)
    xb = xb @ np.array(A13).T
assert np.allclose(xa, xb, rtol=0, atol=1e-10), np.abs(xa - xb).max()

episodes = {}
for sit in T.SITUATIONS:
    starts = T.bank_starts(sit)
    S, R = rollout(starts, True)
    S0, R0 = rollout(starts, False)
    D = S - S0
    per_float = []
    for i in range(NF):
        m = np.sum(S[1:, :, pair(i)] ** 2)          # steps 1..100
        m0 = np.sum(S0[1:, :, pair(i)] ** 2)
        d = np.sum(D[1:, :, pair(i)] ** 2)
        per_float.append(dict(
            float=i + 1,
            rms_size_with_C=r(np.sqrt(m / (100 * len(starts))), 5),
            rms_size_without_C=r(np.sqrt(m0 / (100 * len(starts))), 5),
            rms_change_by_C=r(np.sqrt(d / (100 * len(starts))), 5),
            change_share_of_motion=r(np.sqrt(d / m), 5),          # RMS(coupling part) / RMS(motion)
            change_energy_share=r(d / m, 6),
            largest_change_any_step=r(np.linalg.norm(D[:, :, pair(i)], axis=2).max(), 5),
        ))
    ret, ret0 = R.sum(0), R0.sum(0)
    tot = np.sum(S[1:] ** 2)
    totd = np.sum(D[1:] ** 2)
    episodes[sit] = dict(
        per_float=per_float,
        all_floats_change_share=r(np.sqrt(totd / tot), 5),
        mean_return_no_throttle_with_C=r(ret.mean(), 4),
        mean_return_no_throttle_without_C=r(ret0.mean(), 4),
        mean_return_change_pct=r(100 * (ret.mean() - ret0.mean()) / abs(ret0.mean()), 4),
        largest_episode_return_change_pct=r(100 * np.max(np.abs(ret - ret0) / np.abs(ret0)), 4),
    )
facts["no_throttle_episodes"] = dict(
    note="1,000 bank starts per situation (bank_starts, seed [20261012, 2, 9]); zero throttle; 100 steps; "
         "share = RMS over steps 1-100 and starts of (with C - without C) / RMS of the motion with C; return = "
         "undiscounted 100-step sum",
    situations=episodes)

# ------------------------------------------------------------------ reachability
Bf = np.array(B13[:DS])


def ctrb_rank(M, Bm):
    cols, P = [], Bm.copy()
    for _ in range(DS):
        cols.append(P)
        P = M @ P
    K = np.hstack(cols)
    sv = np.linalg.svd(K, compute_uv=False)
    return int(np.sum(sv > 1e-10 * sv[0])), float(sv[-1])


direct = {f"float_{i + 1}": [k + 1 for k in range(T.DA) if Bf[2 * i + 1, k] != 0] for i in range(NF)}
rank_C, smin_C = ctrb_rank(A10, Bf)
rank_0, smin_0 = ctrb_rank(A0, Bf)
# per float, without C: can the four thrusters move it at all? (its own pair reachable from B through A0)
reach0 = {}
for i in range(NF):
    blockB = Bf[pair(i)]
    reach0[f"float_{i + 1}"] = bool(np.abs(blockB).max() > 0)
# single-thruster subsets: which floats can thruster k move without C, and with C
single = {}
for k in range(T.DA):
    bk = Bf[:, [k]]
    without = [i + 1 for i in range(NF) if any(np.abs((P @ bk)[pair(i)]).max() > 0 for P in [np.eye(DS)] + PW0[:20])]
    with_c = [i + 1 for i in range(NF) if any(np.abs((P @ bk)[pair(i)]).max() > 1e-14 for P in [np.eye(DS)] + PW[:20])]
    single[f"thruster_{k + 1}"] = dict(moves_without_C=without, moves_with_C=with_c)
facts["reachability"] = dict(
    thrusters_pushing_each_float_directly=direct,
    water_pushes_every_float_directly=bool(np.all(v[1::2] == 1.0)),
    controllable_rank_with_C=rank_C, controllable_rank_without_C=rank_0,
    smallest_singular_value_with_C=r(smin_C, 8), smallest_singular_value_without_C=r(smin_0, 8),
    every_float_moved_directly_without_C=reach0,
    single_thruster=single,
    float_to_float_without_C="none: A0 is block diagonal, so a push on one float never reaches another",
)

with open(OUT, "w") as f:
    json.dump(facts, f, indent=1)

# ------------------------------------------------------------------ summary print
m = facts["matrices"]
print("C: spectral norm", m["C_spectral_norm"], "largest entry", m["C_largest_entry_abs"], "at", m["C_largest_entry_at"],
      "mean |entry|", m["C_mean_entry_abs"], "nonzero", m["C_nonzero_entries"], "attempt", m["coupling_attempt"])
print("block norms (row float <- column float):")
for row in m["C_block_norms_rows_to_cols_from"]:
    print("  ", row)
print("own swing block norms", m["own_swing_block_norms"], "ratio", m["ratio_C_to_smallest_own_block"])
print("spectral radius with/without", m["spectral_radius_with_C"], m["spectral_radius_without_C"])
print("swings with C   ", m["swings_with_C"])
print("swings without C", m["swings_without_C"])
print("own-pair share", m["own_pair_share_of_each_swing"])
print("\nimpulse (unit speed kick on float j): max swing size of float i within 20 / 100 steps")
for j in range(NF):
    rows = facts["impulse"]["table"][f"push_on_float_{j + 1}"]
    print(f" push on float {j + 1}:",
          "  ".join(f"f{x['float']}: {x['max_size_20']:.4f}/{x['max_size_100']:.4f}" for x in rows))
    print("   at step 20 / 100:", "  ".join(f"f{x['float']}: {x['size_at_20']:.4f}/{x['size_at_100']:.4f}" for x in rows))
    print("   change by C to own float 20/100:", rows[j]["max_change_by_C_20"], rows[j]["max_change_by_C_100"],
          " without C own max:", rows[j]["max_size_20_without_C"])
im = facts["impulse"]
for k in ("largest_cross_within_20", "smallest_cross_within_20", "median_cross_within_20", "largest_cross_within_100",
          "smallest_cross_within_100", "median_cross_within_100", "largest_change_to_pushed_float_20",
          "largest_change_to_pushed_float_100", "propagator_gap_20", "propagator_norm_20_without_C",
          "propagator_gap_100", "propagator_norm_100_without_C"):
    print(" ", k, im[k])
print("\nsteady current per unit:")
for x in facts["steady_current_per_unit"]:
    print("  ", x)
print("wave per unit amplitude:")
for x in facts["wave_per_unit_amplitude"]:
    print("  ", x)
print("\nno-throttle episodes:")
for sit, e in episodes.items():
    print(f" {sit}: all-float share {e['all_floats_change_share']}, return {e['mean_return_no_throttle_with_C']} vs "
          f"{e['mean_return_no_throttle_without_C']} ({e['mean_return_change_pct']}%), largest episode change "
          f"{e['largest_episode_return_change_pct']}%")
    for x in e["per_float"]:
        print("    ", x)
print("\nreachability:", json.dumps(facts["reachability"], indent=1))
