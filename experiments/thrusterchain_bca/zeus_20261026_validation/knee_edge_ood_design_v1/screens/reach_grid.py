"""Reach feasibility grid for the knee-edge design (rule: ../REACH_SCREEN_RULE.md, written first).

Reads saved JSON from the finished central-quartile study only. Opens no NPZ and no checkpoint, runs no simulator.
Run: .venv/Scripts/python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_design_v1/screens/reach_grid.py
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

V = Path(__file__).resolve().parents[2]
EV = V / "central_quartile_training_v1_evaluation_rerun1" / "run" / "evaluation"
RUNS = [V / "central_quartile_training_v1_authorized1" / "run", V / "central_quartile_training_v1_continuation1" / "run"]
KNEE = float(np.float32(0.8))
SLOTS = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
DESIGN = np.vstack([np.ones(5), SLOTS, SLOTS ** 2]).T
CELLS = [(c, s) for c in ("full", "hole", "random") for s in range(3)]

RESETS = {"wide": [(1.0, 0.1, 1.1)], "mix25": [(0.75, 0.1, 0.7), (0.25, 0.9, 1.1)],
          "mix15": [(0.85, 0.1, 0.7), (0.15, 0.9, 1.1)]}
WINDOWS = {"A": (-1.0, -0.6, 0.05), "B": (-0.9, -0.7, 0.05), "C": (-0.9, -0.7, 0.02), "D": (-0.85, -0.75, 0.02)}
DEPTHS = (0.10, 0.05)
PROBE_BAR, VISITED_BAR = 0.40, 0.30
inputs = []


def load(path):
    inputs.append(path)
    return json.loads(path.read_text(encoding="utf-8"))


def second_moment(law):
    return sum(w * (((lo + hi) / 2) ** 2 + (hi - lo) ** 2 / 12) for w, lo, hi in law)


def anchor(lo, hi, e):
    """Exact mean of retained high-load a0 under CUTOFF: window part with a0 >= -KNEE, floor part with |a0| <= KNEE."""
    w_mass = (1 - e) * (hi + KNEE) / (hi - lo)
    f_mass = e * KNEE
    return w_mass * (hi - KNEE) / 2 / (w_mass + f_mass), (1 - e) * (-KNEE - lo) / (hi - lo) + e * (1 - KNEE)


def audit_lambda(cond, seed):
    for run in RUNS:
        path = run / f"seed{seed}_{cond}" / "native" / "gradient_audit_020000.json"
        if path.exists():
            return float(np.mean([b["lambda_from_unpenalized_Q1"] for b in load(path)["batches"]]))
    raise FileNotFoundError(f"no 20k gradient audit for seed{seed}_{cond}")


mean_abs_q = {}
for ctx in ("ordinary", "strong_current"):
    vals = [np.abs(np.array(load(EV / f"seed{s}_{c}__native__20000__{ctx}.json")["q"]["per_probe"]["q1"])[:, 0]).mean()
            for c, s in CELLS]
    mean_abs_q[ctx] = float(np.mean(vals))
e_ord, e_str = second_moment([(1.0, 0.1, 0.7)]), second_moment([(1.0, 0.7, 1.1)])
slope = (mean_abs_q["strong_current"] - mean_abs_q["ordinary"]) / (e_str - e_ord)
icept = mean_abs_q["ordinary"] - slope * e_ord

cells = []
for c, s in CELLS:
    d = load(EV / f"seed{s}_{c}__native__20000__strong_current.json")
    pp, vr = d["q"]["per_probe"], d["visited"]["per_row"]
    hl = np.array(pp["high_load"], bool)
    coef = np.linalg.lstsq(DESIGN, np.array(pp["q1"])[:, 1:].T, rcond=None)[0].T
    cells.append(dict(name=f"{c}/s{s}", lam=audit_lambda(c, s), b=coef[hl, 1], c=coef[hl, 2],
                      visited=np.array(vr["actions"])[np.array(vr["high_load"], bool), 0]))
assert len(cells) == 9

lines = [f"mean|Q1| ordinary {mean_abs_q['ordinary']:.1f}, strong current {mean_abs_q['strong_current']:.1f}; "
         f"fit mean|Q1| = {icept:.1f} + {slope:.1f} E[c^2]", ""]
header = f"{'reset':6} {'win':3} {'d':>4} {'scale':>5} {'m':>7} {'remove_hi':>9} {'probe_min':>9} {'probe_max':>9} " \
         f"{'visit_min':>9} {'visit_max':>9} feasible"
lines.append(header)
feasible = []
for (rname, law), (wname, (lo, hi, e)), d in itertools.product(RESETS.items(), WINDOWS.items(), DEPTHS):
    scale = mean_abs_q["ordinary"] / (icept + slope * second_moment(law))
    m, removed = anchor(lo, hi, e)
    gate = -(KNEE + d)
    probe, visit = [], []
    for cell in cells:
        lam = cell["lam"] * scale
        pi0 = np.clip((m + 2 * lam * cell["b"]) / (1 - 4 * lam * cell["c"]), -1, 1)
        probe.append(np.mean(pi0 <= gate))
        visit.append(np.mean(np.clip(scale * cell["visited"] + 0.87 * m, -1, 1) <= gate))
    ok = min(probe) >= PROBE_BAR and min(visit) >= VISITED_BAR
    if ok:
        feasible.append((d, e, rname, wname))
    lines.append(f"{rname:6} {wname:3} {d:4.2f} {scale:5.3f} {m:7.4f} {removed:9.3f} {min(probe):9.3f} {max(probe):9.3f} "
                 f"{min(visit):9.3f} {max(visit):9.3f} {'YES' if ok else 'no'}")

order = lambda f: (-f[0], -f[1], list(RESETS).index(f[2]), list(WINDOWS).index(f[3]))
lines += ["", f"feasible candidates: {len(feasible)}",
          "selected by the rule: " + (str(min(feasible, key=order)) if feasible else "none (take to the user)"), "",
          "inputs (sha256):"]
lines += [f"  {hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(V)}" for p in sorted(set(inputs))]
out = "\n".join(lines)
print(out)
(Path(__file__).with_name("reach_grid_output.txt")).write_text(out + "\n", encoding="utf-8")
