"""Expected values for the knee-edge arms under the exact stratified RANDOM rule (Monte Carlo, no simulator dynamics).

The RANDOM rule copies conditional_ood_gate_central_quartile_v1/geometry.py control_masks: in each stratum
(episode x ten-step bin x load band) keep exactly as many rows as CUTOFF keeps, chosen uniformly from ALL rows of the
stratum, withheld rows included. Load is action-independent, so only the reset law, wave and recorder are drawn.
Run: .venv/Scripts/python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_design_v1/screens/random_control_mc.py
"""
from pathlib import Path

import numpy as np

KNEE = float(np.float32(0.8))
EDGE = float(np.float32(-0.9))          # recorder window's lowest float32 command
DEEP = -(KNEE + 0.10)                   # R_deep- boundary (inclusive)
EPISODES, T, REPS = 768, 100, 200
rng = np.random.default_rng(20261029)


def band(g):
    return np.where(g < 0.4, 0, np.where(g < 1.0, 1, 2))


def draw():
    deep_ep = rng.random(EPISODES) < 0.15
    c = np.where(deep_ep, rng.uniform(0.9, 1.1, EPISODES), rng.uniform(0.1, 0.7, EPISODES))
    amp, phase = rng.uniform(0.1, 0.4, EPISODES), rng.uniform(0, 2 * np.pi, EPISODES)
    g = c[:, None] + amp[:, None] * np.cos(phase[:, None] + 0.27 * np.arange(T))
    u, v = rng.random((EPISODES, T)), rng.random((EPISODES, T))
    high = g >= 0.4
    a0 = np.where(high & (u < 0.98), -0.9 + 0.2 * v, -1 + 2 * v).astype(np.float32).astype(np.float64)
    return g, a0


def stratified_keep(strata_id, keep_count):
    order = np.lexsort((rng.random(len(strata_id)), strata_id))
    sid = strata_id[order]
    first = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    rank = np.arange(len(sid)) - np.repeat(first, np.diff(np.r_[first, len(sid)]))
    keep = np.zeros(len(sid), bool)
    keep[order] = rank < keep_count[sid]
    return keep


stats = {k: [] for k in ("w_ret", "w_ret_hi", "w_ret_lo", "dens_mid", "dens_deep", "dens_hi", "anchor_r", "anchor_c",
                         "rdeep_lo", "rdeep_mid", "rdeep_deep", "removed", "removed_hi", "removed_lo",
                         "dens_hi_unstrat", "dens_hi_bandblind", "dens_hi_binary", "dens_deep_binary",
                         "share_hi", "share_deep", "episodes_with_deep", "cal_replacements")}
first_w = []
for _ in range(REPS):
    g, a0 = draw()
    b = band(g).ravel(); a = a0.ravel(); w = np.abs(a) > KNEE
    ep = np.repeat(np.arange(EPISODES), T); tb = np.tile(np.arange(T) // 10, EPISODES)
    sid = np.unique(ep * 100 + tb * 3 + b, return_inverse=True)[1]
    cut = ~w
    keep_count = np.bincount(sid, weights=cut).astype(int)
    ran = stratified_keep(sid, keep_count)
    hi = b >= 1
    in_band = (a >= EDGE) & (a < -KNEE)
    n_hi = hi.sum()
    stats["w_ret"].append(ran[w].mean()); stats["w_ret_hi"].append(ran[w & hi].mean()); stats["w_ret_lo"].append(ran[w & ~hi].mean())
    for name, m in (("dens_mid", b == 1), ("dens_deep", b == 2), ("dens_hi", hi)):
        stats[name].append((ran & in_band & m).sum() / (m.sum() * 0.1))
    stats["anchor_r"].append(a[ran & hi].mean()); stats["anchor_c"].append(a[cut & hi].mean())
    for name, k in (("rdeep_lo", 0), ("rdeep_mid", 1), ("rdeep_deep", 2)):
        stats[name].append((ran & (a <= DEEP) & (b == k)).sum())
    stats["removed"].append((~cut).sum()); stats["removed_hi"].append((~cut & hi).mean() / hi.mean()); stats["removed_lo"].append((~cut & ~hi).mean() / (~hi).mean())
    unstrat = np.zeros(len(a), bool); unstrat[rng.choice(len(a), cut.sum(), replace=False)] = True
    stats["dens_hi_unstrat"].append((unstrat & in_band & hi).sum() / (n_hi * 0.1))
    sid2 = np.unique(ep * 100 + tb * 3 + 0, return_inverse=True)[1]
    blind = stratified_keep(sid2, np.bincount(sid2, weights=cut).astype(int))
    stats["dens_hi_bandblind"].append((blind & in_band & hi).sum() / (n_hi * 0.1))
    sid3 = np.unique(ep * 100 + tb * 3 + hi.astype(int), return_inverse=True)[1]   # prior geometry.py: binary high flag
    binary = stratified_keep(sid3, np.bincount(sid3, weights=cut).astype(int))
    stats["dens_hi_binary"].append((binary & in_band & hi).sum() / (n_hi * 0.1))
    stats["dens_deep_binary"].append((binary & in_band & (b == 2)).sum() / ((b == 2).sum() * 0.1))
    stats["share_hi"].append(hi.mean()); stats["share_deep"].append((b == 2).mean())
    stats["episodes_with_deep"].append((b.reshape(EPISODES, T) == 2).any(1).mean())
    cal = rng.integers(0, T, 128)                       # 128 calibration episodes: original uniform time per episode
    stats["cal_replacements"].append(int((np.abs(a0[:128][np.arange(128), cal]) > KNEE).sum()))
    fw = np.argmax(np.abs(a0) > KNEE, axis=1).astype(float); fw[~(np.abs(a0) > KNEE).any(1)] = np.inf
    first_w.append(fw)

lines = [f"Monte Carlo: {REPS} datasets x {EPISODES} fitting episodes x {T} steps; reset knee_mix15; recorder window C",
         f"{'quantity':22} {'mean':>10} {'sd':>9} {'mean-6sd':>10} {'mean+6sd':>10}"]
for k, vals in stats.items():
    x = np.array(vals, float)
    lines.append(f"{k:22} {x.mean():10.4f} {x.std(ddof=1):9.4f} {x.mean() - 6 * x.std(ddof=1):10.4f} {x.mean() + 6 * x.std(ddof=1):10.4f}")
fw = np.concatenate(first_w)
lines.append("first W step: P(<=0)=%.3f P(<=1)=%.3f P(<=4)=%.3f P(<=9)=%.3f median=%g" % tuple(
    [np.mean(fw <= k) for k in (0, 1, 4, 9)] + [np.median(fw)]))
out = "\n".join(lines)
print(out)
Path(__file__).with_name("random_control_mc_output.txt").write_text(out + "\n", encoding="utf-8")
