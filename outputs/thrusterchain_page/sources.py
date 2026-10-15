"""Collect the page's remaining literal numbers from their source files (read only) into sources_data.json, so that
every number on the page has a recorded source. Nothing is imported from the repository, trained or written under
/home/user/bca: each value is read from the text of its file with a pattern, and the build refuses if one is missing."""

import json
import pathlib
import re

REPO = pathlib.Path("/home/user/bca")
SCRATCH = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).with_name("sources_data.json")


def grab(path, pattern, cast=float, group=1, flags=0):
    text = pathlib.Path(path).read_text()
    m = re.search(pattern, text, flags)
    if not m:
        raise SystemExit(f"{path}: pattern {pattern!r} not found")
    return cast(m.group(group)) if isinstance(group, int) else tuple(cast(m.group(g)) for g in group)


out = {}

# Transition Dynamics: the float coupling of the earlier world, imported read-only by thruster_chain.py
p = REPO / "experiments/signal/gapped_lq.py"
out["coupling_seed"] = dict(value=grab(p, r"coupling_seed: int = (\d+)", int), source=str(p))

# H1's far rows (shift.py constants)
p = REPO / "experiments/shift/shift.py"
out["far_rows"] = dict(removed_share_max=grab(p, r"^FAR_MS_MAX = ([\d.]+)", float, flags=re.M),
                       ball_radii=grab(p, r"^FAR_RADII = ([\d.]+)", float, flags=re.M),
                       pool_size=grab(p, r"^POOL_SIZE = (\d+)", int, flags=re.M), source=str(p))

# Version History: the 3-state toy world (experiments/signal/lq_harness.py) and its placement result
p = REPO / "experiments/signal/lq_harness.py"
gamma, length = grab(p, r"def default_system\(gamma=([\d.]+), length=(\d+)\)", float, group=(1, 2))
ds, da = grab(p, r"ds = (\d+), da = (\d+)", int, group=(1, 2))
p2 = REPO / "outputs/glq/lean-v1/README.md"
lo, hi = grab(p2, r"perfect placement added (−[\d.]+) to \+([\d.]+) points", str, group=(1, 2))
out["toy_world"] = dict(states=ds, throttles=da, discount=gamma, steps=int(length),
                        perfect_placement_points=[float(lo.replace("−", "-")), float(hi)],
                        source=[str(p), str(p2)])

# Methods, "Measuring mu": the design's own measurement (scratch script f2_refs.py on its prototype log)
f2 = SCRATCH / "shiftenv/final/f2.out"
f2py = SCRATCH / "shiftenv/final/f2_refs.py"
fw = SCRATCH / "shiftenv/final/fworld.py"
mq, mql = grab(f2, r"mean\|Q\| ([\d.]+) \(with TD3 smoothing level ([\d.]+)", float, group=(1, 2))
out["design_mu"] = dict(
    size_without_noise=mq, size_with_noise=mql,
    curvature=grab(f2, r"median \|H_aa\| ([\d.]+)"),
    mu=grab(f2, r"with smoothing level ([\d.]+)\)"),
    rows=grab(f2py, r"choice\(len\(xk\), (\d+),", int),
    data_seed=grab(f2py, r"D = gen\(WS, TRIG, (\d+)\)", int),
    episodes=grab(fw, r"def gen\(W, trig, rep, n=(\d+)", int),
    source=[str(f2), str(f2py), str(fw)])

# Results: the design's own odds for the gate (judgements written in the design record, not computations)
rec = json.loads((SCRATCH / "shift_design_final.json").read_text())
eo = rec["expected_outcomes"]


def grab_text(text, pattern):
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"design record: pattern {pattern!r} not found")
    return m.group(1)


out["design_odds"] = dict(
    h2_harm=grab_text(eo, r"H2 harm >= 5 points in strong current: about (\d+%)"),
    h1_ratio=grab_text(eo, r"H1, gap error ratio >= 2: about (\d+%)"),
    h1_specificity=grab_text(eo, r"H1 specificity \(far rows, random, placebo\): about (\d+%)"),
    harm_certified=grab_text(eo, r"Harm certified \(H1 and H2, including the one retune\): about ([\d\-]+%)"),
    room_given_harm=grab_text(eo, r"H3 room on at least one situation, given harm: about (\d+%)"),
    whole_gate=grab_text(eo, r"The whole gate: about (\d+%)"),
    source=str(SCRATCH / "shift_design_final.json") + " (expected_outcomes)")

# H4: what the factor 1.25 is for (design record, harm_checks)
hc = json.dumps(rec)
out["power_inflation"] = dict(
    text=grab_text(hc, r"a 3-point effect, and (a 25% inflation for the SD's own uncertainty)"),
    holm=grab_text(hc, r"This is one-sided alpha 0\.05/3 \((worst-case Holm)\)"))

OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
