"""Read-only facts for page pass 4, edit 2. Writes tmp/p4/edit2/netvalue.json.

1. The exact value of a command at one example state (floats at rest, current 0.6, wave w1 = 0.3, w2 = 0: inside the
   recorded conditions and in the full-throttle region), with thruster 1 swept from 0 to full throttle upstream in steps
   of 5% and thrusters 2-4 at the stall-aware controller's command, in both worlds, the stall-aware controller driving
   afterwards (the page's true value: discounted 0.98 sum over 1,000 steps). Reproduces notes/p4_facts.md section 4 at
   80/90/100%.
2. The recording controller's normal-driving command with the floats at rest, current 0.5, no wave and no noise
   (the widget's "Normal driving" preset), and the worked examples of the Normal driving legend.
3. Distances of the widget's presets to the main test's removed ball (seed 900).

Run from /mnt/c/Users/David/dev/bca-shift-work/page with
  PYTHONPATH=/root/bca-shift /opt/bca-shift-venv/bin/python scripts/p4_e2_netvalue.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/bca-shift")
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402

OUT = Path("/mnt/c/Users/David/dev/bca-shift-work/page/tmp/p4/edit2/netvalue.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
out = {}

sa = RC.stall_aware()
x = np.zeros(13)
x[TC.CURRENT], x[TC.WAVE1] = 0.6, 0.3
base = sa(x[None])[0]
rows = []
for k in range(21):
    mag = round(0.05 * k, 2)
    a = base.copy()
    a[0] = -mag
    row = dict(throttle=mag)
    for world, phys in (("stall", TC.STALL), ("no_stall", TC.NO_STALL)):
        x1, r0 = TC.dynamics(x[None], a[None], phys)
        q = TR.q_value(sa, x[None], a[None], phys)[0]
        row[world] = dict(push=float(TC.thrust(np.array([-mag]), phys)[0]), reward=float(r0[0]),
                          float_cost_next=float(np.sum(x1[0, :10] ** 2)), value=float(q))
    rows.append(row)
v80 = {w: next(r for r in rows if r["throttle"] == 0.8)[w]["value"] for w in ("stall", "no_stall")}
for r in rows:
    for w in ("stall", "no_stall"):
        r[w]["value_minus_80"] = r[w]["value"] - v80[w]
best = {w: max(rows, key=lambda r: r[w]["value"])["throttle"] for w in ("stall", "no_stall")}
out["net_value"] = dict(state="floats at rest, current 0.6, wave w1 = 0.3, w2 = 0", push_water=float(TC.push(x[None])[0]),
                        others=[float(v) for v in base[1:]], stall_aware_t1=float(base[0]), rows=rows, best=best)
print("stall-aware command here", np.round(base, 4), "water push", TC.push(x[None]))
for r in rows:
    print(f"{r['throttle']:.2f}  stall push {r['stall']['push']:.3f} value {r['stall']['value']:.4f} "
          f"d80 {r['stall']['value_minus_80']:+.4f} | no-stall push {r['no_stall']['push']:.3f} "
          f"value {r['no_stall']['value']:.4f} d80 {r['no_stall']['value_minus_80']:+.4f}  reward {r['stall']['reward']:.4f}")
print("best throttle by exact value:", best)

# 2. normal driving at rest, current 0.5, no wave, no noise
K = RC.textbook_gain(TC.STALL)
Kc = K[:, TC.CURRENT]
print("K current column", np.round(Kc, 4))
u = -(Kc * 0.5)
a_nd = 0.8 * np.tanh(u / 0.8)
out["normal_driving_rest_c05"] = dict(u=[float(v) for v in u], a=[float(v) for v in a_nd], K_current=[float(v) for v in Kc])
print("normal driving preset u", np.round(u, 4), "a", np.round(a_nd, 4))
u2 = u[0] - 0.6
out["normal_driving_two_sd"] = dict(u1=float(u2), a1=float(0.8 * np.tanh(u2 / 0.8)))
print("two-sd example", round(u2, 4), round(0.8 * np.tanh(u2 / 0.8), 4))
P = RC.full_throttle_pattern(K)
print("full-throttle pattern", np.round(P, 4))
out["pattern"] = [float(v) for v in P]

# 3. distances to the main test's ball (centre and radius from notes/p4_facts.json)
F = json.load(open("/mnt/c/Users/David/dev/bca-shift-work/page/notes/p4_facts.json"))
ball = F["four_throttle_widget"]["main_test_seed900"]["removed_ball"]
c, rad = np.array(ball["centre"]), ball["radius"]
presets = {"off": np.zeros(4), "normal": a_nd, "capped": np.array([-0.8, P[1], P[2], P[3]]), "full": P}
out["ball"] = dict(centre=list(c), radius=rad,
                   presets={k: dict(command=[float(v) for v in p], distance=float(np.linalg.norm(p - c)),
                                    cost=float(0.3 * np.sum(p ** 2))) for k, p in presets.items()})
for k, p in presets.items():
    print(k, np.round(p, 3), "distance", round(float(np.linalg.norm(p - c)), 4), "cost", round(float(0.3 * np.sum(p ** 2)), 4))
OUT.write_text(json.dumps(out, indent=1))
print("wrote", OUT)
