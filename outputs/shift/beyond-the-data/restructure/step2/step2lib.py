"""Step 2 of the restructured BCA (the policy pilot): shared definitions.

EXPLORATORY, designed after seeing seeds 0-4 and after step 1; written before any step-2 actor is trained. See
PRESPEC.md and prespec.json beside this file. Everything here is a pure NumPy function (the penalty also runs on
jax.numpy, passed in as `xp`), a small path / hash / JSON / verified-import helper, or the go/no-go and reporting
arithmetic. It is tested in
test_step2.py on synthetic data. The heavy loaders (the gate, the room harness, version 6's policy code, JAX) live in
machinery.py.

Nothing here writes outside the step-2 output folder (runs/shift_explore/restructure/step2) or, in scratch mode, a
folder outside the repository. Bytecode writing is switched off before any import so that no __pycache__ is written
into a pinned folder.
"""
import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path("/home/user/bca")
HERE = Path(__file__).resolve().parent
PRESPEC_NAME = "prespec.json"
MANIFEST_NAME = "code_manifest.json"
CODE_FILES = ("step2lib.py", "machinery.py", "train_policy.py", "evaluate_policy.py", "pooled.py", "run_step2.sh",
              "test_step2.py", "freeze.py", "PRESPEC.md", "prespec.json")
OFFICIAL_OUT = ROOT / "runs" / "shift_explore" / "restructure" / "step2"
STEP1_RESULTS = ROOT / "outputs" / "shift" / "beyond-the-data" / "restructure" / "step1" / "results" / "results.json"
# Pinned folders: never written. The official output folder above is the only repository path step 2 writes.
PINNED = tuple(ROOT / p for p in (
    "experiments", "runs/shift", "runs/shift_explore/stage2", "runs/shift_explore/stage2_v7",
    "runs/shift_explore/room", "runs/shift_explore/standins", "runs/shift_explore/detect",
    "runs/shift_explore/deploy", "runs/shift_explore/fallback", "runs/shift_explore/trace",
    "runs/shift_explore/anatomy", "runs/shift_explore/empirical", "runs/shift_explore/principled",
    "runs/shift_explore/restructure/step1", "outputs/shift/beyond-the-data/stage2",
    "outputs/shift/beyond-the-data/stage2_v7", "outputs/shift/beyond-the-data/stage2_gpu",
    "outputs/shift/beyond-the-data/stage2_v7_design", "outputs/shift/beyond-the-data/restructure/step1",
    "outputs/shift/beyond-the-data/restructure/analysis", "calibration", "algorithms", "runtime"))
WORLDS = ("stall", "no_stall")
SEEDS = (0, 1, 2, 3, 4)
RESERVED = {**{s: "version 7's confirmatory run (seeds 5-9)" for s in range(5, 10)},
            **{s: "the planned version 8 confirmation on fresh seeds (seeds 10-14)" for s in range(10, 15)}}
SITUATIONS = ("recorded", "strong_current", "big_waves", "big_kicks")
RAMP_ARMS = ("obca_k1", "obca_k2", "obca_forced", "fixed_gated", "conformal_k1", "meangap_k1")
PRIMARY = "obca_k1"
W0 = "w0_check"
WORLD_BLIND = ("standin", "even_pull", "random_1", "random_2", "obca_forced")
LABEL = ("EXPLORATORY: restructured BCA, step 2 (the policy pilot). Designed after seeing seeds 0-4 and after step 1; "
         "the specification was written and frozen before any step-2 actor was trained. Outside the gate; changes no "
         "pre-registered rule and no v6/v7/step-1 file.")


# ------------------------------------------------------------------------------------------------ paths and hashes

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def inside(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    return p == r or r in p.parents


def guard_out(path, official):
    """Refuse any output path except: official mode, inside runs/shift_explore/restructure/step2; scratch mode,
    outside the repository altogether. A pinned folder is refused in both modes."""
    p = Path(path).resolve()
    for pin in PINNED:
        if inside(p, pin):
            raise SystemExit(f"refused: {p} is inside the pinned folder {pin}")
    if official:
        if not inside(p, OFFICIAL_OUT):
            raise SystemExit(f"refused: official outputs go under {OFFICIAL_OUT}, not {p}")
    elif inside(p, ROOT):
        raise SystemExit(f"refused: scratch outputs must lie outside {ROOT} (got {p})")
    return p


def exec_verified(name, path, sha256):
    """Execute a pinned file from its verified bytes as module `name` (no bytecode is written anywhere). Refuses when
    the bytes differ from `sha256`, or when a module of that name is already loaded and is not this verified one.
    Once loaded, a plain `import <name>` anywhere binds to this verified module."""
    path = Path(path)
    data = path.read_bytes()
    if sha256_bytes(data) != sha256:
        raise SystemExit(f"refused: {path} differs from the SHA-256 prespec.json pins")
    if name in sys.modules:
        mod = sys.modules[name]
        if getattr(mod, "__step2_sha256__", None) != sha256:
            raise SystemExit(f"refused: a module named {name!r} is already loaded from elsewhere")
        return mod
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    mod.__step2_sha256__ = sha256
    sys.modules[name] = mod
    exec(compile(data, str(path), "exec"), mod.__dict__)
    return mod


def local_imports(path, folder):
    """The modules a Python file imports anywhere in it (also inside functions) whose file <name>.py lies in
    `folder`: the sibling files it would load through the normal import system."""
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return sorted(n for n in names if (Path(folder) / f"{n}.py").is_file())


def unverified_modules(folders):
    """Loaded modules whose file lies inside one of `folders` but which exec_verified did not load (no
    __step2_sha256__): files that ran through the normal import system with no hash check. Empty when all is well."""
    out = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and getattr(mod, "__step2_sha256__", None) is None and any(inside(f, d) for d in folders):
            out.append(f"{name} ({f})")
    return sorted(out)


def jsonable(x):
    """Plain JSON types; non-finite floats become the strings 'inf', '-inf', 'nan' (strict JSON has none)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isfinite(v):
            return v
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(x, Path):
        return str(x)
    return x


def num(x):
    """A number as stored by step 1's jsonable: a float, or one of the strings 'inf', '-inf', 'nan'."""
    if isinstance(x, str):
        if x in ("inf", "-inf", "nan"):
            return float(x)
        raise ValueError(f"not a number: {x!r}")
    if x is None:
        raise ValueError("a missing number")
    return float(x)


def write_json_once(path, payload, official):
    """Strict JSON with a .sha256 sidecar; refuses to overwrite (write-once)."""
    p = guard_out(path, official)
    if p.exists():
        raise SystemExit(f"refused: {p} exists (write-once)")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(payload), indent=1, sort_keys=True, allow_nan=False)
    with open(p, "x", encoding="utf8") as handle:          # 'x': fails if another process wrote it meanwhile
        handle.write(text)
    digest = sha256_bytes(text.encode("utf8"))
    p.with_name(p.name + ".sha256").write_text(f"{digest}  {p.name}\n", encoding="utf8")
    return digest


def write_bytes_once(path, blob, official):
    p = guard_out(path, official)
    if p.exists():
        raise SystemExit(f"refused: {p} exists (write-once)")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "xb") as handle:
        handle.write(blob)
    return sha256_bytes(blob)


def read_json_checked(path, required=True):
    """A JSON with a matching '.sha256' sidecar (refuses when the sidecar is missing or differs)."""
    p = Path(path)
    if not p.exists():
        if required:
            raise SystemExit(f"refused: {p} is missing")
        return None
    text = p.read_text(encoding="utf8")
    side = p.with_name(p.name + ".sha256")
    if not side.exists() or side.read_text().split()[0] != sha256_bytes(text.encode("utf8")):
        raise SystemExit(f"refused: {p} differs from its .sha256 sidecar (or has none)")
    return json.loads(text)


def load_prespec(folder=HERE):
    """prespec.json, verified against prespec.json.sha256 (written by freeze.py). Returns (spec, its SHA-256)."""
    folder = Path(folder)
    path, side = folder / PRESPEC_NAME, folder / (PRESPEC_NAME + ".sha256")
    text = path.read_bytes()
    if not side.exists():
        raise SystemExit("refused: prespec.json.sha256 is missing; run freeze.py first")
    if sha256_bytes(text) != side.read_text().split()[0]:
        raise SystemExit("refused: prespec.json changed after it was frozen")
    return json.loads(text), sha256_bytes(text)


def verify_manifest(folder=HERE, names=CODE_FILES):
    """code_manifest.json (written by freeze.py): it must list exactly `names`, and every listed file must still hash
    to its frozen value. Returns (manifest, its SHA-256)."""
    folder = Path(folder)
    path = folder / MANIFEST_NAME
    if not path.exists():
        raise SystemExit("refused: code_manifest.json is missing; run freeze.py first")
    man = json.loads(path.read_text())
    if sorted(man.get("files", {})) != sorted(names):
        raise SystemExit("refused: code_manifest.json does not list exactly the step-2 files")
    bad = [n for n, h in man["files"].items() if not (folder / n).exists() or file_sha256(folder / n) != h]
    if bad:
        raise SystemExit(f"refused: files changed after freeze.py: {bad}")
    return man, file_sha256(path)


def official_outputs_exist(out=OFFICIAL_OUT):
    """Any official step-2 output (a lock, a stop, a values file, the pooled results): after that the specification
    is final and freeze.py refuses."""
    out = Path(out)
    if not out.exists():
        return []
    found = [str(p) for pat in ("policy/seed*/*/lock.json", "policy/seed*/*/stop.json", "policy/seed*/*/values.json",
                                "policy/seed*/*/actors.msgpack") for p in sorted(out.glob(pat))]
    found += [str(p) for p in (out / "results.json",) if p.exists()]
    return found


def cpus():
    return sorted(os.sched_getaffinity(0))


def require_one_cpu(what):
    c = cpus()
    if len(c) != 1:
        raise SystemExit(f"refused: {what} must run pinned to one CPU (taskset -c <cpu> ...); this process may use {c}")
    return c


def check_environment(spec):
    for k, v in spec["protocol"]["environment"].items():
        if os.environ.get(k) != v:
            raise SystemExit(f"refused: {k} must be {v!r} (got {os.environ.get(k)!r})")


def check_seed(seed, allowed=SEEDS):
    """Seeds 0-4 only. Seeds 5-9 (version 7's confirmatory run) and 10-14 (the planned version 8 confirmation) are
    reserved: no step-2 code may touch them."""
    s = int(seed)
    if s in RESERVED:
        raise SystemExit(f"refused: seed {s} is reserved for {RESERVED[s]}; step 2 runs on seeds 0-4 only")
    if s not in tuple(allowed):
        raise SystemExit(f"refused: seed {s} is not one of step 2's seeds {list(allowed)}")
    return s


def check_world(world):
    if world not in WORLDS:
        raise SystemExit(f"refused: world {world!r} is not one of {WORLDS}")
    return world


def vmhwm_mb():
    """Peak resident memory of this process (VmHWM from /proc/self/status), in MB; None where unavailable."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def seed_dir(root, seed, world):
    return Path(root) / "policy" / f"seed{int(seed)}" / world


# ------------------------------------------------------------------------------------------------ step 1's numbers

def load_step1(path, want):
    """Step 1's frozen results (outputs/.../restructure/step1/results/results.json), read-only: its SHA-256 must be the
    one prespec.json pins (want['sha256']), it must be an official run under step 1's frozen prespec.json
    (want['prespec_sha256']) with verdict GO. Returns the parsed results."""
    p = Path(path)
    raw = p.read_bytes()
    if sha256_bytes(raw) != want["sha256"]:
        raise SystemExit(f"refused: {p} is not the step-1 results file prespec.json pins")
    r = json.loads(raw.decode("utf8"))
    if r.get("mode") != "official":
        raise SystemExit("refused: the step-1 results are not an official run")
    if r.get("prespec_sha256") != want["prespec_sha256"]:
        raise SystemExit("refused: the step-1 results were computed under another step-1 prespec.json")
    if (r.get("go_no_go") or {}).get("verdict") != "GO":
        raise SystemExit("refused: step 1's verdict is not GO; step 2 is not licensed")
    return r


def arm_table(step1, seeds=SEEDS, rule=None):
    """The six outcome-ramp arms of every seed and world, read from step 1's results with no refit (PRESPEC.md, ARMS).

    Per seed-world, from step1['per_seed_world']['<world>/seed<k>']:
      t_lo, t_up                 the gap rule's ends (the ramp runs from t_lo to t_up)
      BCA switch / height        certificate.on / certificate.height (= R_tail - R_rest; 'inf' when not certified)
      split conformal gap        groups.q90_full_throttle - groups.q90_rest
      rank test p                rank_test.p (one-sided Mann-Whitney)
      mean gap                   groups.mean_full_throttle - groups.mean_rest
    Arms (switch, height h, multiplier k):
      obca_k1       BCA switch;  h = BCA height when on, else 0;  k = 1   (PRIMARY)
      obca_k2       as obca_k1;  k = 2
      obca_forced   always on;   h = this seed's STALLING-world BCA height, in both worlds;  k = 1
      fixed_gated   BCA switch;  h = rule['fixed_height'] (1.0 value unit) when on, else 0;  k = 1
      conformal_k1  switch = conformal gap > rule['conformal_margin'] (0.10);  h = conformal gap when on;  k = 1
      meangap_k1    switch = p < rule['rank_p'] (0.05);  h = max(mean gap, 0) when on;  k = 1
    Refuses (SystemExit) when an arm's switch is on and its h is not finite, or when an h is negative."""
    rule = {**dict(fixed_height=1.0, conformal_margin=0.10, rank_p=0.05), **(rule or {})}
    per = step1["per_seed_world"]
    out = {w: {} for w in WORLDS}
    for s in seeds:
        stall_h = num(per[f"stall/seed{s}"]["certificate"]["height"])
        for w in WORLDS:
            key = f"{w}/seed{s}"
            e = per[key]
            c, g, rt = e["certificate"], e["groups"], e["rank_test"]
            on_bca, h_bca = bool(c["on"]), num(c["height"])
            gap90 = num(g["q90_full_throttle"]) - num(g["q90_rest"])
            on_conf = bool(gap90 > float(rule["conformal_margin"]))
            p = num(rt["p"])
            on_rank = bool(p < float(rule["rank_p"]))
            meangap = num(g["mean_full_throttle"]) - num(g["mean_rest"])
            arms = dict(
                obca_k1=dict(switch=on_bca, h=h_bca if on_bca else 0.0, k=1.0),
                obca_k2=dict(switch=on_bca, h=h_bca if on_bca else 0.0, k=2.0),
                obca_forced=dict(switch=True, h=stall_h, k=1.0),
                fixed_gated=dict(switch=on_bca, h=float(rule["fixed_height"]) if on_bca else 0.0, k=1.0),
                conformal_k1=dict(switch=on_conf, h=gap90 if on_conf else 0.0, k=1.0),
                meangap_k1=dict(switch=on_rank, h=max(meangap, 0.0) if on_rank else 0.0, k=1.0))
            for name, a in arms.items():
                if a["switch"] and not math.isfinite(a["h"]):
                    raise SystemExit(f"refused: {key}: arm {name} is switched on but its height {a['h']} is not "
                                     f"finite (step 1 could not certify the full-throttle group)")
                if a["h"] < 0:
                    raise SystemExit(f"refused: {key}: arm {name} has a negative height {a['h']}")
            out[w][s] = dict(
                t_lo=num(e["t_lo"]), t_up=num(e["t_up"]), arms=arms,
                sources=dict(bca_on=on_bca, bca_height=h_bca, stall_bca_height=stall_h, conformal_gap=gap90,
                             plain_gap_information_only=num(c.get("plain_gap", "nan")), rank_p=p, mean_gap=meangap))
    return out


def table_matches(table, expected, tol=0.0):
    """True when every (switch, h, k, t_lo, t_up) of `table` equals prespec.json's expected table (tol on numbers)."""
    for w in WORLDS:
        for s, row in table[w].items():
            e = expected[w][str(s)]
            if abs(row["t_lo"] - e["t_lo"]) > tol or abs(row["t_up"] - e["t_up"]) > tol:
                return False
            for name in RAMP_ARMS:
                a, b = row["arms"][name], e["arms"][name]
                if bool(a["switch"]) != bool(b["switch"]) or abs(a["h"] - b["h"]) > tol or a["k"] != b["k"]:
                    return False
    return True


# ------------------------------------------------------------------------------------------------ the penalty

def throttle(a, xp=np):
    """Throttle index t(a) = max over the four thrusters of |a_j| (last axis)."""
    return xp.max(xp.abs(a), axis=-1)


def ramp(t, t_lo, t_up, xp=np):
    """clip((t - t_lo) / (t_up - t_lo), 0, 1): 0 at or below t_lo, 1 at or above t_up, linear between."""
    return xp.clip((t - t_lo) / (t_up - t_lo), 0.0, 1.0)


def outcome_ramp(a, h, t_lo, t_up, xp=np):
    """U(a) = h * ramp(t(a))."""
    return h * ramp(throttle(a, xp), t_lo, t_up, xp)


def ramp_penalty(pi, a_rec, h, k, t_lo, t_up, xp=np):
    """The paired, one-sided outcome-ramp penalty per row: k * [U(pi(s)) - U(a_rec)]_+ (v6's paired form, change A1
    of the room test), with a_rec the row's recorded command. With h = 0 it is 0 and so is its gradient."""
    return k * xp.maximum(outcome_ramp(pi, h, t_lo, t_up, xp) - outcome_ramp(a_rec, h, t_lo, t_up, xp), 0.0)


# ------------------------------------------------------------------------------------------------ pooled measures

def mean_or_none(xs):
    return None if not xs or any(x is None for x in xs) else float(np.mean(xs))


def diff(a, b):
    return None if a is None or b is None else float(a) - float(b)


def pts_of(pts, world, seed, arm, situation="strong_current"):
    """points of logical arm `arm` (None when missing, unmatched, stopped or failed)."""
    return (((pts.get(world) or {}).get(seed) or {}).get(arm) or {}).get(situation)


def net(pts, arm, seeds, situation="strong_current"):
    """NET(x) = mean over seeds of { [pts_stall(x) - pts_stall(none)] - [pts_nostall(none) - pts_nostall(x)] }.
    Every seed is needed: a missing value gives None."""
    gain = [diff(pts_of(pts, "stall", s, arm, situation), pts_of(pts, "stall", s, "none", situation)) for s in seeds]
    loss = [diff(pts_of(pts, "no_stall", s, "none", situation), pts_of(pts, "no_stall", s, arm, situation))
            for s in seeds]
    per = [None if g is None or l_ is None else g - l_ for g, l_ in zip(gain, loss)]
    return dict(per_seed=per, net=mean_or_none(per), gain_stall=mean_or_none(gain), loss_no_stall=mean_or_none(loss))


def paired(pts, world, a, b, seeds, situation):
    x = [diff(pts_of(pts, world, s, a, situation), pts_of(pts, world, s, b, situation)) for s in seeds]
    return dict(per_seed=x, mean=mean_or_none(x), positive=sum(1 for v in x if v is not None and v > 0))


def go_no_go(pts, seeds, rule):
    """The fixed go/no-go (PRESPEC.md, GO / NO-GO). Each needs every seed; a missing value fails the condition.
      G1  NET(obca_k1) >= max over the world-blind rivals W of NET + g1_margin (10)
      G2  stalling world, obca_k1 - none >= g2_floor (-3) in recorded conditions, big waves and big kicks
      G3  stalling world, strong current, obca_k1 - matched even pull >= g3_margin (15)"""
    n_p = net(pts, PRIMARY, seeds)
    n_w = {x: net(pts, x, seeds) for x in WORLD_BLIND}
    w_vals = [v["net"] for v in n_w.values()]
    best_w = None if any(v is None for v in w_vals) else max(w_vals)
    g1 = bool(n_p["net"] is not None and best_w is not None and n_p["net"] >= best_w + float(rule["g1_margin"]))
    keeps = {s: paired(pts, "stall", PRIMARY, "none", seeds, s) for s in ("recorded", "big_waves", "big_kicks")}
    g2 = bool(all(v["mean"] is not None and v["mean"] >= float(rule["g2_floor"]) for v in keeps.values()))
    ev = paired(pts, "stall", PRIMARY, "even_pull", seeds, "strong_current")
    g3 = bool(ev["mean"] is not None and ev["mean"] >= float(rule["g3_margin"]))
    go = bool(g1 and g2 and g3)
    best_name = None
    if best_w is not None:
        best_name = [x for x in WORLD_BLIND if n_w[x]["net"] == best_w][0]
    return dict(G1=g1, G2=g2, G3=g3, verdict="GO" if go else "NO-GO",
                G1_detail=dict(net_primary=n_p, net_world_blind=n_w, best_world_blind=best_name,
                               best_world_blind_net=best_w, margin=float(rule["g1_margin"])),
                G2_detail=dict(keeps, floor=float(rule["g2_floor"])),
                G3_detail=dict(ev, margin=float(rule["g3_margin"])))


# The attribution's fixed wordings (review of 15 Oct 2026, before any official training; prespec.json
# attribution.wording holds the same text, and test_step2.py checks that the two agree). On seeds 0-4 the three
# rivals have (almost) obca_k1's switch and a smaller stalling-world height in every seed (attribution_facts), so
# each A compares one penalty shape at two doses: a failed A is worded as dose, never as credit to a part of BCA.
ATTRIBUTION = (
    ("A1", "conformal_k1",
     "the Bayesian part adds nothing measurable (plain split conformal does as well)",
     "a larger dose did better: conformal_k1's smaller height did worse. Nothing is attributed to the Bayesian "
     "part: the switch is the same as obca_k1's in every seed-world, so the comparison differs only in the "
     "stalling-world height, and whether BCA's height did better because it is Bayesian or only because it is "
     "larger cannot be told here"),
    ("A2", "fixed_gated",
     "the calibrated height adds nothing beyond the switch",
     "a larger dose did better: the fixed height 1.0 did worse. Nothing is attributed to calibration: the switch is "
     "the same as obca_k1's in every seed-world, so the comparison differs only in the stalling-world height"),
    ("A3", "meangap_k1",
     "calibration adds nothing beyond raw average residuals",
     "a larger dose did better: meangap_k1's smaller height did worse. Nothing is attributed to calibration: the "
     "switch is the same as obca_k1's except in no_stall/seed2 (h = 0.048 there), so the comparison differs almost "
     "only in the stalling-world height"))
DOSE_ARMS = ("meangap_k1", "fixed_gated", "conformal_k1", "obca_k1", "obca_k2")
DOSE_READING = dict(
    rising="obca_k2 (twice BCA's dose) did better by more than 5: BCA's height lies on the rising part of the dose "
           "curve, so it is not an optimum; more of this penalty helps here",
    plateau="obca_k2 (twice BCA's dose) is within 5 of obca_k1: BCA's height reaches the plateau of the dose curve; "
            "this says it is large enough, not that it is calibrated",
    falling="obca_k2 (twice BCA's dose) did worse by more than 5: the best dose of this penalty lies below twice "
            "BCA's height; this says where BCA's height lies on the dose curve, not that it is calibrated",
    undefined="undefined: a value is missing")
CALIBRATION_CREDIT = (
    "none on these seeds: in the stalling world the height of every rival (fixed_gated, conformal_k1, meangap_k1) "
    "is below BCA's in every seed, and their switches are (almost) BCA's, so no comparison separates a calibrated "
    "height from a larger one. A failed A is worded as dose (a larger dose did better), never as credit to the "
    "Bayesian part or to calibration; obca_k2 against obca_k1 says only where BCA's height lies on the dose curve")


def _row(table, world, seed):
    """One seed-world row of an arm table (step2lib.arm_table: int seed keys; prespec.json: string seed keys)."""
    t = table[world]
    return t[seed] if seed in t else t[str(seed)]


def attribution_facts(table, seeds):
    """What the A comparisons can tell apart, from the arm table alone (no outcome): per A, the seed-worlds where the
    rival's switch differs from obca_k1's, and whether its stalling-world dose k*h is below obca_k1's in every seed
    (with the per-seed ratio of obca_k1's dose to the rival's)."""
    per = {}
    for tag, arm, _, _ in ATTRIBUTION:
        differs = [f"{w}/seed{s}" for w in WORLDS for s in seeds
                   if bool(_row(table, w, s)["arms"][arm]["switch"]) !=
                   bool(_row(table, w, s)["arms"][PRIMARY]["switch"])]
        dose = {s: float(_row(table, "stall", s)["arms"][arm]["k"]) * float(_row(table, "stall", s)["arms"][arm]["h"])
                for s in seeds}
        dose_p = {s: float(_row(table, "stall", s)["arms"][PRIMARY]["k"]) *
                  float(_row(table, "stall", s)["arms"][PRIMARY]["h"]) for s in seeds}
        per[tag] = dict(arm=arm, switch_differs_in=differs,
                        stall_dose_below_primary_in_every_seed=bool(all(dose[s] < dose_p[s] for s in seeds)),
                        stall_dose_ratio_primary_over_arm={s: (dose_p[s] / dose[s] if dose[s] > 0 else math.inf)
                                                           for s in seeds})
    return dict(per_comparison=per, no_rival_height_reaches_bca=bool(
        all(v["stall_dose_below_primary_in_every_seed"] for v in per.values())))


def dose_readout(pts, seeds, rule):
    """Reported: obca_k2 (k = 2) against obca_k1, which says where BCA's height lies on the dose curve."""
    a, b = net(pts, "obca_k2", seeds), net(pts, PRIMARY, seeds)
    m = float(rule["attribution_margin"])
    d = diff(a["net"], b["net"])
    reading = "undefined" if d is None else ("rising" if d > m else ("falling" if d < -m else "plateau"))
    return dict(net_obca_k2=a["net"], net_obca_k1=b["net"], difference=d, reading=reading,
                wording=DOSE_READING[reading], per_seed_difference=[diff(x, y) for x, y in
                                                                    zip(a["per_seed"], b["per_seed"])])


def dose_table(pts, seeds, table, situation="strong_current"):
    """Reported: per seed, for the arms of one penalty shape at different doses (DOSE_ARMS), the dose k*h in each
    world, the stalling-world gain over none, the no-stall loss against none and that seed's NET term."""
    out = {}
    for s in seeds:
        out[s] = {}
        for arm in DOSE_ARMS:
            st, ns = _row(table, "stall", s)["arms"][arm], _row(table, "no_stall", s)["arms"][arm]
            gain = diff(pts_of(pts, "stall", s, arm, situation), pts_of(pts, "stall", s, "none", situation))
            loss = diff(pts_of(pts, "no_stall", s, "none", situation), pts_of(pts, "no_stall", s, arm, situation))
            out[s][arm] = dict(dose_stall=float(st["k"]) * float(st["h"]),
                               dose_no_stall=float(ns["k"]) * float(ns["h"]),
                               gain_stall=gain, loss_no_stall=loss, net=diff(gain, loss))
    return out


def attribution(pts, seeds, rule, table=None):
    """Reported; decides only the claim's wording. A<i> holds when NET(rival) >= NET(obca_k1) - margin (5): then that
    part of BCA adds nothing measurable here. When it fails, the fixed wording says only that a larger dose did
    better (ATTRIBUTION). Also: the per-seed differences, the dose readout (obca_k2 against obca_k1), the fixed
    statement that nothing is credited to calibration on these seeds and, given the arm table, the facts behind it."""
    rp = net(pts, PRIMARY, seeds)
    n_p = rp["net"]
    margin = float(rule["attribution_margin"])
    out = {}
    for tag, arm, if_holds, if_fails in ATTRIBUTION:
        rx = net(pts, arm, seeds)
        n_x = rx["net"]
        holds = None if n_p is None or n_x is None else bool(n_x >= n_p - margin)
        per = [diff(x, y) for x, y in zip(rx["per_seed"], rp["per_seed"])]
        out[tag] = dict(arm=arm, net_arm=n_x, net_primary=n_p, holds=holds, meaning_if_holds=if_holds,
                        meaning_if_fails=if_fails,
                        wording=if_holds if holds else (if_fails if holds is False else DOSE_READING["undefined"]),
                        per_seed_difference=per,
                        holds_in_seeds=sum(1 for v in per if v is not None and v >= -margin),
                        seeds_defined=sum(1 for v in per if v is not None))
    out["dose"] = dose_readout(pts, seeds, rule)
    out["calibration_credit"] = dict(credited=False, statement=CALIBRATION_CREDIT,
                                     facts=None if table is None else attribution_facts(table, seeds))
    return out


def analogues(pts, seeds, rule):
    """Version 6's marks restated for obca_k1 (reported, not decisive; G3 is M1's margin)."""
    sc = "strong_current"
    m1 = paired(pts, "stall", PRIMARY, "even_pull", seeds, sc)
    m2 = {f: paired(pts, "stall", PRIMARY, f, seeds, sc) for f in ("random_1", "random_2")}
    m3 = dict(recorded_vs_none=paired(pts, "stall", PRIMARY, "none", seeds, "recorded"),
              recorded_vs_clone=paired(pts, "stall", PRIMARY, "clone", seeds, "recorded"),
              big_waves_vs_none=paired(pts, "stall", PRIMARY, "none", seeds, "big_waves"),
              big_kicks_vs_none=paired(pts, "stall", PRIMARY, "none", seeds, "big_kicks"))
    m4 = paired(pts, "stall", PRIMARY, "standin", seeds, sc)

    def ok(x, bar):
        return bool(x is not None and x >= bar)
    return dict(
        M1_even_pull=dict(m1, passed=bool(ok(m1["mean"], rule["m1_margin"]) and m1["positive"] >= rule["m1_positive_min"])),
        M2_random_surfaces=dict(m2, passed=bool(all(ok(v["mean"], rule["m2_margin"]) for v in m2.values()))),
        M3_keeps_improvement=dict(m3, passed=bool(
            ok(m3["recorded_vs_none"]["mean"], rule["m3_floor"]) and ok(m3["recorded_vs_clone"]["mean"],
                                                                         rule["m3_clone_margin"])
            and ok(m3["big_waves_vs_none"]["mean"], rule["m3_floor"])
            and ok(m3["big_kicks_vs_none"]["mean"], rule["m3_floor"]))),
        M4_best_standin=dict(m4, passed=ok(m4["mean"], rule["m4_floor"])))


# ------------------------------------------------------------------------------------------------ record checks

def lock_problems(lock, spec_sha, man_sha, seed, world, official, steps):
    """Why a lock cannot be used (empty list: usable): mode, the frozen prespec.json and code manifest, seed, world,
    the step count, and the built-in checks."""
    problems = []
    if lock.get("kind") != "step2-policy-lock":
        problems.append("not a step-2 policy lock")
    if official and lock.get("mode") != "official":
        problems.append("not an official run")
    if lock.get("prespec_sha256") != spec_sha:
        problems.append("trained under another prespec.json")
    if lock.get("code_manifest_sha256") != man_sha:
        problems.append("trained under another code manifest")
    if lock.get("seed") != int(seed) or lock.get("world") != world:
        problems.append(f"lock is for {lock.get('world')}/seed {lock.get('seed')}")
    if official and lock.get("steps") != int(steps):
        problems.append(f"not {steps} actor steps")
    if not all(bool(v.get("passed")) for v in (lock.get("checks") or {}).values()):
        problems.append("a built-in check did not pass")
    if official and len(lock.get("cpus") or []) != 1:
        problems.append("not run on one CPU")
    return problems
