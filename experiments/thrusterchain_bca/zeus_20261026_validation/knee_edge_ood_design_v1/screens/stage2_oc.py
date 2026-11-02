"""Operating characteristics of the Stage 2 screens R1, R2, R3 under a branch-level noise model (Monte Carlo).

Model: each Stage 2 branch return J = cell mean + effect + e, e ~ N(0, sb^2) independent across branches, with
sb = 3.55 / sqrt(2) so that a same-fork difference J(BCA) - J(rep) has SD 3.55 (saved same-fork proxy).
Cells: stall CUTOFF (sC), stall RANDOM (sR), no-stall CUTOFF (nsC). Branches: BCA and replicate in each cell, distance in sC.
  R1: G_i = D_sC,i - D_sR,i > 0 for all 3 seeds and mean(G) >= +3.0      (D = J(BCA) - J(rep))
  R2: J(BCA,sC) - J(dist,sC) >= 0 for all 3 seeds
  R3: H_i = D_nsC,i - D_sR,i; mean(H) >= bar    (shares D_sR with R1: H = G + (D_nsC - D_sC))
Run: .venv/Scripts/python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_design_v1/screens/stage2_oc.py
"""
from pathlib import Path

import numpy as np

SB = 3.55 / np.sqrt(2)
N, SEEDS = 400_000, 3
rng = np.random.default_rng(20261029)


def screens(g_true, cost, dist_gaps, bar):
    """One noise draw per row; R2 is evaluated for every distance gap from that same draw (shared bca_sC term)."""
    e = {k: rng.normal(0, SB, (N, SEEDS)) for k in ("bca_sC", "rep_sC", "bca_sR", "rep_sR", "bca_nsC", "rep_nsC", "dist_sC")}
    d_sc = g_true + e["bca_sC"] - e["rep_sC"]           # region-specific gain carried in stall CUTOFF
    d_sr = e["bca_sR"] - e["rep_sR"]                     # generic gain set to 0 (cancels in every contrast)
    d_nsc = cost + e["bca_nsC"] - e["rep_nsC"]           # cost of caution in no-stall CUTOFF
    g, h = d_sc - d_sr, d_nsc - d_sr
    r1 = (g > 0).all(1) & (g.mean(1) >= 3.0)
    r2 = [((gap + e["bca_sC"] - e["dist_sC"]) >= 0).all(1) for gap in dist_gaps]
    r3 = h.mean(1) >= bar
    return r1, r2, r3


lines = ["Stage 2 operating characteristics (branch SD %.3f; same-fork difference SD 3.55; %d draws)" % (SB, N), ""]
lines.append(f"{'bar':>5} {'G':>4} {'cost':>5} {'P(R1)':>7} {'P(R3)':>7} {'P(R3|R1)':>9} {'joint d=0':>10} {'joint d=3':>10} {'R1&R3':>7}")
for bar in (-3.0, -2.0, -1.5, -1.0):
    for g_true, cost in ((0, 0), (3, 0), (3, -3), (3, -6), (0, -3)):
        r1, (r2a, r2b), r3 = screens(g_true, cost, (0.0, 3.0), bar)
        lines.append(f"{bar:5.1f} {g_true:4d} {cost:5d} {r1.mean():7.3f} {r3.mean():7.3f} {r3[r1].mean():9.3f} "
                     f"{(r1 & r2a & r3).mean():10.3f} {(r1 & r2b & r3).mean():10.3f} {(r1 & r3).mean():7.3f}")
lines += ["", "joint d=0: distance arm equal to BCA; joint d=3: BCA beats distance by 3; R1&R3: R2 ignored.",
          "Rows with G=0 are null for R1; rows with cost=0 are null for R3."]
out = "\n".join(lines)
print(out)
Path(__file__).with_name("stage2_oc_output.txt").write_text(out + "\n", encoding="utf-8")
