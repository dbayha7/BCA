"""The harm switch's certificates (module M10 of MODULES.md), read from the recorded files; nothing is computed anew.

Prints one row per seed and world: whether the gap rule was defined, whether the switch is on, the statistic q05, the
height R_tail - R_rest, the group sizes n_tail and n_rest, and the two thresholds. Beside each row it prints what the
threshold's own arithmetic (module M5) says about the tail group's size: a BCA threshold with 1,000 draws and
credibility 0.95 can never be finite with 28 or fewer scores (0.9^28 = 0.052 > 0.05), is sometimes finite with 29 to 35,
and is finite with 36 or more (calibration/reference.certifiable, tolerance 1e-6). It then prints the near-80 run's
recorded stops.

Standard library only. Run from anywhere:
    python docs/bca_mechanics/scripts/certificate_table.py [repository root]
The root defaults to the folder three levels above this script.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "..", "..", ".."))
NEAR80 = os.path.join(ROOT, "outputs", "shift", "beyond-the-data", "near80", "results-20261020T115246Z", "files")
STEP1 = os.path.join(ROOT, "outputs", "shift", "beyond-the-data", "restructure", "step1", "results", "results.json")


def num(v):
    """Records write +inf as the string 'inf' (strict JSON has no infinity)."""
    if isinstance(v, str):
        return float(v)
    return v


def expectation(n_tail):
    if n_tail <= 28:
        return "never finite (n <= 28)"
    if n_tail <= 35:
        return "sometimes finite (29-35)"
    return "finite (n >= 36)"


def fmt(v, width=9):
    if v is None:
        return "-".rjust(width)
    if isinstance(v, bool):
        return str(v).rjust(width)
    if isinstance(v, float):
        return (f"{v:.4g}" if abs(v) != float("inf") else "inf").rjust(width)
    return str(v).rjust(width)


def row(label, setting, world, seed, defined, cert):
    on, q05, h = cert.get("on"), num(cert.get("q05")), num(cert.get("height"))
    n_tail, n_rest = cert.get("n_tail"), cert.get("n_rest")
    r_tail, r_rest = num(cert.get("R_tail")), num(cert.get("R_rest"))
    note = expectation(n_tail) if n_tail is not None else "-"
    flag = ""
    if on and h is not None and h == float("inf"):
        flag = "  <-- switch ON with height +inf (the ramp arm is refused)"
    print(f"{label:8} {setting:12} {world:9} {str(seed):>4} {fmt(defined, 8)} {fmt(on, 6)} {fmt(q05)} {fmt(h)} "
          f"{fmt(n_tail, 7)} {fmt(n_rest, 7)} {fmt(r_tail)} {fmt(r_rest)}  {note}{flag}")


def main():
    print(f"repository root: {ROOT}")
    print()
    print(f"{'source':8} {'setting':12} {'world':9} {'seed':>4} {'defined':>8} {'on':>6} {'q05':>9} {'height':>9} "
          f"{'n_tail':>7} {'n_rest':>7} {'R_tail':>9} {'R_rest':>9}  M5's expectation for R_tail")
    print("-" * 150)
    pattern = os.path.join(NEAR80, "learners", "*", "*", "seed*", "switch.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no near-80 switch records under {NEAR80}")
    for path in files:
        d = json.load(open(path, encoding="utf8"))
        row("near-80", d["setting"], d["world"], d["seed"], d.get("defined"), d.get("certificate", {}))
    print()
    if os.path.exists(STEP1):
        s1 = json.load(open(STEP1, encoding="utf8"))
        for key in sorted(s1["per_seed_world"]):
            world, seed = key.split("/seed")
            e = s1["per_seed_world"][key]
            row("step 1", "main", world, int(seed), True, e["certificate"])
        print()
        print("step 1 go/no-go:", json.dumps(s1.get("go_no_go")))
    else:
        print(f"no step-1 results at {STEP1}")
    print()
    stops = sorted(glob.glob(os.path.join(NEAR80, "stops", "*.json")))
    print(f"near-80 recorded stops: {len(stops)}")
    for path in stops:
        d = json.load(open(path, encoding="utf8"))
        print(f"  {d.get('phase_id')}: {d.get('reason')}: {(d.get('detail') or {}).get('error')}")


if __name__ == "__main__":
    main()
