"""Probe for the sweep encoding: how many decimals the 7-digit values carry, how large the embedded data would be at
each unit, and whether decoding gives back exactly the 7-digit values. Prints only."""
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "precise" / "q_data.json").read_text())
F = {f: i for i, f in enumerate(d["episode_fields"])}


def decimals(v):
    t = Decimal(repr(float(v))).as_tuple()
    return max(0, -t.exponent)


mins, maxd, mags = [], 0, []
hist = {}
for e in d["episodes"]:
    for key in ("q_actor", "qpi_actor"):
        for v in e[F[key]]:
            k = decimals(v); hist[k] = hist.get(k, 0) + 1; maxd = max(maxd, k); mags.append(abs(v))
    for key in ("q_sweep", "qpi_sweep"):
        for row in e[F[key]]:
            for v in row:
                k = decimals(v); hist[k] = hist.get(k, 0) + 1; maxd = max(maxd, k); mags.append(abs(v))
print("smallest |value|", min(mags), "largest", max(mags), "max decimals", maxd, "decimals histogram", sorted(hist.items()))

for dd in (3, 4, 5, 6):
    sc = 10 ** dd
    parts, exact_bad = [], 0
    for e in d["episodes"]:
        for key, ref in (("q_sweep", "q_actor"), ("qpi_sweep", "qpi_actor")):
            rows = []
            for row, a in zip(e[F[key]], e[F[ref]]):
                ra = round(a * sc)
                r = [round(v * sc) - ra for v in row]
                rows.append([r[0]] + [r[k] - r[k - 1] for k in range(1, len(r))])
                acc = 0
                for k, v in enumerate(row):
                    acc = r[0] if k == 0 else acc + rows[-1][k]
                    if (ra + acc) / sc != v:
                        exact_bad += 1
            parts.append(rows)
    size = len(json.dumps(parts, separators=(",", ":")))
    print(f"unit 1e-{dd}: sweeps {size / 1e6:.3f} MB; decoded values not exactly the 7-digit value: {exact_bad}")


# second differences: r_0, r_1 - r_0, then (r_k - r_(k-1)) - (r_(k-1) - r_(k-2))
for dd in (5, 6):
    sc = 10 ** dd
    parts = []
    for e in d["episodes"]:
        for key, ref in (("q_sweep", "q_actor"), ("qpi_sweep", "qpi_actor")):
            rows = []
            for row, a in zip(e[F[key]], e[F[ref]]):
                ra = round(a * sc)
                r = [round(v * sc) - ra for v in row]
                d1 = [r[k] - r[k - 1] for k in range(1, len(r))]
                rows.append([r[0], d1[0]] + [d1[k] - d1[k - 1] for k in range(1, len(d1))])
            parts.append(rows)
    print(f"unit 1e-{dd}, second differences: sweeps {len(json.dumps(parts, separators=(',', ':'))) / 1e6:.3f} MB")
