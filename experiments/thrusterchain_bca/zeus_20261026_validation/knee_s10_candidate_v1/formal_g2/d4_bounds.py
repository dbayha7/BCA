"""Two-sided six-SD bounds for D4 of the formal G2 (OOD definition v4 r3: "D4 needs two-sided six-SD Monte Carlo
bounds computed for the candidate's own recorder"). Monte Carlo, no simulator dynamics, written and sealed before the
collection runs.

The twin is the superset arm (every row), so D4's density is the share of the band's fitting rows whose executed a0
lies in the window [-0.95, -0.85] (s10's depth), divided by the window width 0.1, per load band (mid 0.4 <= g < 1,
deep g >= 1) and pooled over high load. The recorder (recorder_wide.py) draws, at high load with probability 0.98,
a0 = -1 + 0.4 v, so the designed density is 0.98 / 0.4 + 0.01 = 2.46 everywhere in the window.

The load model is the one knee_edge_ood_design_v1/screens/random_control_mc.py used for the sealed knee-edge bounds:
g = c + amp cos(phase + 0.27 t), c from the knee_mix15 reset mixture (deep with probability 0.15 in [0.9, 1.1],
else [0.1, 0.7]), amp in [0.1, 0.4]. The executed command is the float32 cast of the request, as in the data.

Run (writes d4_bounds.json once):
  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B knee_s10_candidate_v1/formal_g2/d4_bounds.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "d4_bounds.json"
WINDOW = (-0.95, -0.85)
EPISODES, T, REPS, MC_SEED = 768, 100, 200, 20261031
WINDOW_PROBABILITY, LOW, WIDTH = 0.98, -1.0, 0.4
QUANTITIES = ("dens_mid", "dens_deep", "dens_hi", "share_hi", "share_deep", "rows_mid", "rows_deep")


def band(g):
    return np.where(g < 0.4, 0, np.where(g < 1.0, 1, 2))


def draw(rng):
    deep_ep = rng.random(EPISODES) < 0.15
    c = np.where(deep_ep, rng.uniform(0.9, 1.1, EPISODES), rng.uniform(0.1, 0.7, EPISODES))
    amp, phase = rng.uniform(0.1, 0.4, EPISODES), rng.uniform(0, 2 * np.pi, EPISODES)
    g = c[:, None] + amp[:, None] * np.cos(phase[:, None] + 0.27 * np.arange(T))
    u, v = rng.random((EPISODES, T)), rng.random((EPISODES, T))
    high = g >= 0.4
    a0 = np.where(high & (u < WINDOW_PROBABILITY), LOW + WIDTH * v, -1 + 2 * v).astype(np.float32).astype(np.float64)
    return g, a0


def statistics(g, a0):
    b = band(g).ravel()
    a = a0.ravel()
    in_window = (a >= WINDOW[0]) & (a <= WINDOW[1])
    hi = b >= 1
    out = {}
    for name, m in (("dens_mid", b == 1), ("dens_deep", b == 2), ("dens_hi", hi)):
        out[name] = float((in_window & m).sum() / (m.sum() * (WINDOW[1] - WINDOW[0]))) if m.any() else 0.0
    out["share_hi"], out["share_deep"] = float(hi.mean()), float((b == 2).mean())
    out["rows_mid"], out["rows_deep"] = int((b == 1).sum()), int((b == 2).sum())
    return out


def simulate(reps=REPS, seed=MC_SEED):
    rng = np.random.default_rng(seed)
    stats = {k: [] for k in QUANTITIES}
    for _ in range(reps):
        s = statistics(*draw(rng))
        for k in QUANTITIES:
            stats[k].append(s[k])
    return {k: np.asarray(v, float) for k, v in stats.items()}


def bounds(stats, decimals=3):
    out = {}
    for k, x in stats.items():
        mean, sd = float(x.mean()), float(x.std(ddof=1))
        lo, hi = mean - 6 * sd, mean + 6 * sd
        scale = 10.0 ** decimals
        out[k] = {"mean": mean, "sd": sd, "lower": float(np.floor(lo * scale) / scale), "upper": float(np.ceil(hi * scale) / scale)}
    return out


def main():
    record = {"schema": "knee_s10_candidate_v1.formal_g2.d4_bounds.v1",
              "rule": "two-sided bounds mean -/+ 6 SD of the Monte Carlo, rounded outward to 3 decimals",
              "monte_carlo": {"datasets": REPS, "fitting_episodes": EPISODES, "steps": T, "seed": MC_SEED,
                              "window": list(WINDOW), "designed_density": WINDOW_PROBABILITY / WIDTH + (1 - WINDOW_PROBABILITY) / 2,
                              "load_model": "random_control_mc.py: g = c + amp cos(phase + 0.27 t), knee_mix15 reset mixture"},
              "gated": ["dens_mid", "dens_deep"], "reported": ["dens_hi", "share_hi", "share_deep", "rows_mid", "rows_deep"],
              "bounds": bounds(simulate())}
    raw = (json.dumps(record, indent=2) + "\n").encode()
    with OUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"written": OUT.name, "sha256": hashlib.sha256(raw).hexdigest(),
                      "bounds": {k: [v["lower"], v["upper"]] for k, v in record["bounds"].items()}}, indent=1))


if __name__ == "__main__":
    main()
