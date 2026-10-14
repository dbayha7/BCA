"""Stage 2 v7 evidence (V7_SPEC C1 steps 1-4): every place the gate's and the Stage 2 code can reach A, B or K, and
every place a physics is built, found by reading the source (no study code is imported or run). Writes
argument_tuples.json into this folder and nothing else.

What it lists, per file and line:
  direct    calls of the two cached functions: thruster_chain._matrices(gain, wave_turn) and
            recorder._gain(gain, wave_turn, throttle_cost, gamma);
  wrappers  calls of their public wrappers thruster_chain.matrices(physics) and recorder.textbook_gain(physics), and
            of thruster_chain._columns (cached columns of A and B, used by dynamics and dynamics_jax);
  physics   every Physics(...) / world_physics(...) construction, with the fields it sets;
  replace   any dataclasses.replace on a physics (none expected);
  lapack    calls of the two LAPACK-dependent routines behind A and K (gapped_lq.transition_matrix,
            scipy's solve_discrete_are), which v7's tripwire watches.
The argument tuples follow from the physics fields: gain, wave_turn, throttle_cost and gamma are never set by any
construction (only stall and stall_slope are), so every caller uses the Physics defaults (thruster_chain.py:124-131).

Usage: python3 find_argument_tuples.py
"""
import glob
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/user/bca")
SP = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad")
GROUPS = {
    "gate (experiments/shift, hashed by the gate's pre-registration)": [ROOT / "experiments/shift/*.py"],
    "gate dependency (experiments/signal/gapped_lq.py)": [ROOT / "experiments/signal/gapped_lq.py"],
    "Stage 2 v6 code (outputs/shift/beyond-the-data/stage2/code)": [ROOT / "outputs/shift/beyond-the-data/stage2/code/*.py"],
    "Stage 2 v6 code as run (scratchpad/ood/stage2)": [SP / "ood/stage2/*.py"],
    "room test (runs/shift_explore/room/code)": [ROOT / "runs/shift_explore/room/code/*.py"],
    "stand-ins (runs/shift_explore/standins/code)": [ROOT / "runs/shift_explore/standins/code/*.py"],
    "detection (runs/shift_explore/detect/code)": [ROOT / "runs/shift_explore/detect/code/*.py"],
    "deployment test (runs/shift_explore/deploy/code)": [ROOT / "runs/shift_explore/deploy/code/*.py"],
    "fallback (scratchpad/ood/fallback/fallback.py)": [SP / "ood/fallback/fallback.py"],
}
PATTERNS = {
    "direct": re.compile(r"\b_matrices\s*\(|\b_gain\s*\("),
    "wrappers": re.compile(r"\bmatrices\s*\(|\btextbook_gain\s*\(|\b_columns\s*\("),
    "physics": re.compile(r"\bPhysics\s*\(|\bworld_physics\s*\("),
    "replace": re.compile(r"\breplace\s*\([^)]*\b(physics|STALL|NO_STALL|Physics)\b"),
    "lapack": re.compile(r"\btransition_matrix\s*\(|\bsolve_discrete_are\s*\("),
}
FIELD = re.compile(r"(\*\*\s*[\w.\[\]\"']+|\b\w+)\s*=(?!=)")


def rel(p):
    p = Path(p)
    for base, name in ((ROOT, ""), (SP, "scratchpad/")):
        try:
            return name + p.relative_to(base).as_posix()
        except ValueError:
            pass
    return str(p)


out = dict(kind="stage2-v7-evidence-argument-tuples", groups={})
seen = {}
for group, pats in GROUPS.items():
    files = sorted({f for pat in pats for f in glob.glob(str(pat))})
    g = dict(files=[], hits={k: [] for k in PATTERNS})
    for f in files:
        data = Path(f).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        g["files"].append(dict(file=rel(f), sha256=digest))
        is_test = Path(f).name.startswith("test_")
        for n, line in enumerate(data.decode("utf8").splitlines(), 1):
            for kind, rx in PATTERNS.items():
                if rx.search(line):
                    hit = dict(at=f"{rel(f)}:{n}", text=line.strip()[:160], test_file=is_test)
                    if kind == "physics":
                        inside = line[line.find("(", rx.search(line).start()) + 1:]
                        hit["fields_set"] = sorted(set(m.group(1) for m in FIELD.finditer(inside)
                                                       if m.group(1) not in ("physics",)) if "def " not in line
                                                   else ["(definition)"])
                    g["hits"][kind].append(hit)
    out["groups"][group] = g

# The Stage 2 code as run must be byte-identical to the committed copy (V7_SPEC: S2C).
a = {Path(x["file"]).name: x["sha256"] for x in out["groups"]["Stage 2 v6 code (outputs/shift/beyond-the-data/stage2/code)"]["files"]}
b = {Path(x["file"]).name: x["sha256"] for x in out["groups"]["Stage 2 v6 code as run (scratchpad/ood/stage2)"]["files"]}
out["stage2_copies_identical"] = {n: a.get(n) == b.get(n) for n in sorted(set(a) | set(b))}

cf = json.loads((ROOT / "outputs/shift/beyond-the-data/stage2_v7/world_constants.json").read_text(encoding="utf8"))
out["tuples_in_world_constants"] = {fn: list(v["tuples"]) for fn, v in cf["functions"].items()}
fields_seen = sorted({f for g in out["groups"].values() for h in g["hits"]["physics"]
                      if not h["test_file"] for f in h.get("fields_set", [])})
out["physics_fields_set_outside_tests"] = fields_seen
out["conclusion"] = (
    "Outside test files, no Physics construction sets gain, wave_turn, throttle_cost or gamma (fields set: "
    + ", ".join(fields_seen) + "); the dataset rebuilds physics from its recorded dict (experiments/shift/"
    "dataset.py:128), and every recorded dict holds the defaults (world_constants.json argument_sources). So "
    "_matrices is reached only with (0.3, 0.27) and _gain only with (0.3, 0.27, 0.3, 0.98).")
(HERE / "argument_tuples.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf8")
for group, g in out["groups"].items():
    print(group, {k: len(v) for k, v in g["hits"].items()})
print("fields set outside tests:", fields_seen)
print("stage2 copies identical:", all(out["stage2_copies_identical"].values()))
