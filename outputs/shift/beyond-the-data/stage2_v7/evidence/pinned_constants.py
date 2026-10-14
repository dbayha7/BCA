"""Stage 2 v7 evidence probe (V7_SPEC C1 "Evidence status", "Tested by (b)"). Read-only: writes one JSON (--out) and
nothing else; nothing under runs/; no BCA code; no Stage 2 phase.

This is the earlier session's probe scratchpad/v7_hw_probe/pinned_constants.py (SHA-256 in EARLIER_PROBE_SHA256),
changed in these ways only:
  * the constants come from the pinned file outputs/shift/beyond-the-data/stage2_v7/world_constants.json (checked
    against its sidecar and against --constants-sha256), keyed by the exact argument tuples, instead of from
    plant_default.npz with the arguments ignored; an argument tuple not in the file writes the JSON and exits 97;
  * before substituting, it recomputes A, B and K locally with the gate's unchanged functions and records the
    differences from the file (V7_SPEC C1 step 2), under the OpenBLAS kernel and numpy dispatch of this run;
  * it counts calls to gapped_lq.transition_matrix and scipy.linalg.solve_discrete_are after that point, and records
    the original lru_cache counters at the end (no caller may reach the original functions once they are replaced);
  * it compares the 64 anchors one by one (as scratchpad/v7_hw_probe/anchors_ulp.py does) besides load_prereg's
    whole-table check; without the constants this count is the self-test's ST5 number;
  * it records the full environment and machine facts and writes everything to --out (the earlier one only printed).

The gate's checks themselves run unchanged, with the gate's software check on: acceptance.load_prereg(
acceptance.refuse_protected(acceptance.RUNS_DEFAULT)) and acceptance.guard (as Stage 2 does:
outputs/shift/beyond-the-data/stage2/code/fresh.py:42-46, runs/shift_explore/room/code/harness.py:215-216), then
acceptance.load_seed for gate seeds 0-4 in both worlds (as runs/shift_explore/room/code/harness.py:225).

Usage: python pinned_constants.py --out OUT.json --constants-sha256 SHA [--no-pin] [--seeds-anyway] [--label L]
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import ctypes  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

T0 = time.perf_counter()
ROOT = Path("/home/user/bca")
CONSTANTS = ROOT / "outputs/shift/beyond-the-data/stage2_v7/world_constants.json"
EARLIER_PROBE_SHA256 = "5466bf35c40f37bad514ea94fe8d58365e0e6b44e34d087fdbbd411eb26a39f3"  # scratchpad/v7_hw_probe/pinned_constants.py, copied to earlier_probes_14oct/
MAT_FN = "experiments.shift.thruster_chain._matrices"
GAIN_FN = "experiments.shift.recorder._gain"
TOL_A, TOL_K = 1e-14, 1e-12  # V7_SPEC C1 step 2 (information here: the probe never refuses on them)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--constants-sha256", required=True)
ap.add_argument("--no-pin", action="store_true")
ap.add_argument("--seeds-anyway", action="store_true")
ap.add_argument("--no-seeds", action="store_true")
ap.add_argument("--label", default="")
a = ap.parse_args()
OUT = Path(a.out)
if OUT.exists():
    raise SystemExit(f"refused: {OUT} exists")
pin = not a.no_pin
out = dict(kind="stage2-v7-evidence-pinned-constants", label=a.label, pinned=pin, argv=sys.argv[1:],
           script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           derived_from=dict(script="scratchpad/v7_hw_probe/pinned_constants.py", sha256=EARLIER_PROBE_SHA256))


def write_and_exit(code):
    out["seconds_total"] = time.perf_counter() - T0
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False), encoding="utf8")
    sys.stdout.flush()
    os._exit(code)


# ------------------------------------------------------------------------------ environment and machine facts
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__  # noqa: E402


def openblas():
    base = os.path.dirname(os.path.dirname(np.__file__))
    res = {}
    for pkg, pattern, fns in (("numpy", "numpy.libs/libscipy_openblas64_*.so",
                               ("scipy_openblas_get_corename64_", "scipy_openblas_get_config64_")),
                              ("scipy", "scipy.libs/libscipy_openblas-*.so",
                               ("scipy_openblas_get_corename", "scipy_openblas_get_config"))):
        lib = ctypes.CDLL(sorted(glob.glob(os.path.join(base, pattern)))[0])
        row = {}
        for key, fn in zip(("kernel", "config"), fns):
            f = getattr(lib, fn)
            f.restype = ctypes.c_char_p
            row[key] = f().decode("ascii", "replace").strip()
        res[pkg] = row
    return res


def cpu_model():
    with open("/proc/cpuinfo", encoding="utf8") as h:
        for line in h:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()


libc = ctypes.CDLL("libc.so.6").gnu_get_libc_version
libc.restype = ctypes.c_char_p
out["environment"] = dict(
    variables=dict(sorted(os.environ.items())),
    OPENBLAS_CORETYPE=os.environ.get("OPENBLAS_CORETYPE"),
    NPY_DISABLE_CPU_FEATURES=os.environ.get("NPY_DISABLE_CPU_FEATURES"),
    XLA_FLAGS=os.environ.get("XLA_FLAGS"),
    gate_software_check="on (acceptance.check_software unchanged, called by load_prereg and guard)",
    cwd=os.getcwd(), python=sys.version.split()[0], executable=sys.executable,
    processors_usable=len(os.sched_getaffinity(0)), cpu_model=cpu_model(), glibc=libc().decode(),
    kernel_release=platform.release(), numpy=np.__version__, scipy=scipy.__version__, openblas=openblas(),
    numpy_dispatch_targets_active=[t for t in __cpu_dispatch__ if __cpu_features__.get(t)])

# ------------------------------------------------------------------------------ the constants file
text = CONSTANTS.read_bytes()
digest = hashlib.sha256(text).hexdigest()
side = CONSTANTS.with_name(CONSTANTS.name + ".sha256").read_text(encoding="utf8").split()[0]
out["constants_file"] = dict(path=str(CONSTANTS.relative_to(ROOT)), sha256=digest, sidecar=side,
                             expected=a.constants_sha256, ok=digest == side == a.constants_sha256)
if not out["constants_file"]["ok"]:
    out["refused"] = "constants file SHA-256 differs from its sidecar or from --constants-sha256"
    write_and_exit(1)
CF = json.loads(text)


def from_hex(block):
    M = np.array([[float.fromhex(s) for s in row] for row in block["hex"]], np.float64)
    M.setflags(write=False)
    return M


def args_of(entry, names):
    return tuple(float.fromhex(entry["args_hex"][n]) for n in names)


MAT_NAMES = tuple(CF["functions"][MAT_FN]["argument_names"])
GAIN_NAMES = tuple(CF["functions"][GAIN_FN]["argument_names"])
FILE_AB = {args_of(e, MAT_NAMES): (from_hex(e["A"]), from_hex(e["B"]))
           for e in CF["functions"][MAT_FN]["tuples"].values()}
FILE_K = {args_of(e, GAIN_NAMES): from_hex(e["K"]) for e in CF["functions"][GAIN_FN]["tuples"].values()}

# ------------------------------------------------------------------------------ the gate's modules, as imported by Stage 2
sys.path.insert(0, str(ROOT))
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.recorder as RC  # noqa: E402
import experiments.signal.gapped_lq as GLQ  # noqa: E402
from scipy import linalg as SL  # noqa: E402

ORIG_MATRICES, ORIG_GAIN, ORIG_COLUMNS = TC._matrices, RC._gain, TC._columns
out["caches_at_import"] = dict(matrices=ORIG_MATRICES.cache_info()._asdict(), gain=ORIG_GAIN.cache_info()._asdict(),
                               columns=ORIG_COLUMNS.cache_info()._asdict())


def ulps(x, y):
    xi = np.asarray(x, np.float64).view(np.int64).astype(object)
    yi = np.asarray(y, np.float64).view(np.int64).astype(object)
    same = np.signbit(x) == np.signbit(y)
    return [abs(int(p) - int(q)) if s else None for p, q, s in zip(xi.ravel(), yi.ravel(), same.ravel())]


def compare(local, pinned):
    d = np.abs(np.asarray(local) - np.asarray(pinned))
    neq = np.asarray(local).view(np.uint64) != np.asarray(pinned).view(np.uint64)
    u = [v for v, n in zip(ulps(local, pinned), neq.ravel()) if n]
    return dict(entries=int(d.size), differing=int(neq.sum()), max_abs=float(d.max()),
                max_ulp=max((v for v in u if v is not None), default=0), bit_identical=not bool(neq.any()),
                local_hex=[[float(v).hex() for v in row] for row in np.asarray(local)])


# Step 2 of C1: recompute locally with the gate's functions (through the wrappers, as every caller does).
local = {}
for key, (Af, Bf) in FILE_AB.items():
    Al, Bl = TC.matrices(TC.Physics(**dict(zip(MAT_NAMES, key))))
    local[f"A {key}"] = compare(Al, Af)
    local[f"B {key}"] = compare(Bl, Bf)
for key, Kf in FILE_K.items():
    Kl = RC.textbook_gain(TC.Physics(**dict(zip(GAIN_NAMES, key))))
    local[f"K {key}"] = compare(Kl, Kf)
out["local_vs_file"] = local
maxA = max(v["max_abs"] for k, v in local.items() if k.startswith("A "))
maxK = max(v["max_abs"] for k, v in local.items() if k.startswith("K "))
B_same = all(v["bit_identical"] for k, v in local.items() if k.startswith("B "))
out["local_vs_file_summary"] = dict(max_abs_dA=maxA, max_abs_dK=maxK, B_bit_identical=B_same,
                                    within_v7_tolerances=bool(maxA <= TOL_A and maxK <= TOL_K and B_same),
                                    tolerances=dict(A=TOL_A, K=TOL_K, B="bit-identical"))

# Step 3 of C1 (probe form): substitute the file's arrays, keyed by the exact tuple.
calls = {}


def _record(fn, key):
    calls.setdefault(fn, {}).setdefault(repr(key), 0)
    calls[fn][repr(key)] += 1


if pin:
    def pinned_matrices(gain, wave_turn):
        key = (gain, wave_turn)
        _record(MAT_FN, key)
        if key not in FILE_AB:
            out["refused"] = f"argument tuple {key!r} for {MAT_FN} is not in the constants file"
            write_and_exit(97)
        return FILE_AB[key]

    def pinned_gain(gain, wave_turn, throttle_cost, gamma):
        key = (gain, wave_turn, throttle_cost, gamma)
        _record(GAIN_FN, key)
        if key not in FILE_K:
            out["refused"] = f"argument tuple {key!r} for {GAIN_FN} is not in the constants file"
            write_and_exit(97)
        return FILE_K[key]

    TC._matrices = pinned_matrices
    TC._columns.cache_clear()
    RC._gain = pinned_gain

# Tripwire counters (information): no call is expected after this point in either mode, because the originals' caches
# already hold the local values (unpinned) or the originals are replaced (pinned).
trip = {"experiments.signal.gapped_lq.transition_matrix": 0, "scipy.linalg.solve_discrete_are": 0}
_tm, _dare = GLQ.transition_matrix, SL.solve_discrete_are


def counted_tm(*args, **kw):
    trip["experiments.signal.gapped_lq.transition_matrix"] += 1
    return _tm(*args, **kw)


def counted_dare(*args, **kw):
    trip["scipy.linalg.solve_discrete_are"] += 1
    return _dare(*args, **kw)


GLQ.transition_matrix = counted_tm
SL.solve_discrete_are = counted_dare
mark_matrices = ORIG_MATRICES.cache_info()._asdict()
mark_gain = ORIG_GAIN.cache_info()._asdict()

# ------------------------------------------------------------------------------ the gate's unchanged checks
import experiments.shift.acceptance as A  # noqa: E402

t = time.perf_counter()
p = None
try:
    p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(p)
    out["gate_accepts"] = True
except Exception as e:  # noqa: BLE001 (recorded, as the earlier probe did)
    out["gate_accepts"], out["gate_error"] = False, f"{type(e).__name__}: {str(e)[:600]}"
    p = None
out["seconds_gate"] = time.perf_counter() - t

payload = json.loads((A.RUNS_DEFAULT / A.PREREG_NAME).read_text(encoding="utf8"))
cfg = A.config_from_dict(payload["config"])

# The 64 anchors one by one, recomputed as load_prereg does (anchor_table + canonical JSON round trip).
t = time.perf_counter()
now = json.loads(A.canonical(A.anchor_table(A.RUNS_DEFAULT, cfg)))
rec = payload["anchors"]
rows = []
for k in sorted(rec):
    for s in sorted(rec[k]):
        for q in ("recording", "top", "recording_se", "top_se"):
            x, y = now[k][s][q], rec[k][s][q]
            rows.append(dict(key=f"{k}/{s}/{q}", now=x, rec=y, equal=x == y, abs=abs(x - y),
                             ulp=ulps(np.float64(x), np.float64(y))[0]))
diff = [r for r in rows if not r["equal"]]
out["anchors"] = dict(n=len(rows), n_differing=len(diff), max_abs=max((r["abs"] for r in diff), default=0.0),
                      max_ulp=max((r["ulp"] or 0 for r in diff), default=0),
                      by_table={k: sum(1 for r in diff if r["key"].startswith(k + "/")) for k in sorted(rec)},
                      differing=diff, seconds=time.perf_counter() - t)

seeds = {}
if (p is not None and not a.no_seeds) or a.seeds_anyway:
    for rep in (0, 1, 2, 3, 4):
        for w in ("stall", "no_stall"):
            t = time.perf_counter()
            try:
                A.load_seed(A.RUNS_DEFAULT, cfg, "base", w, rep)
                seeds[f"{w}/seed{rep}"] = dict(ok=True, seconds=time.perf_counter() - t)
            except Exception as e:  # noqa: BLE001
                seeds[f"{w}/seed{rep}"] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}",
                                               seconds=time.perf_counter() - t)
out["load_seed"] = seeds
out["load_seed_summary"] = dict(run=len(seeds), matched=sum(1 for v in seeds.values() if v["ok"]))

# ------------------------------------------------------------------------------ after: nothing bypassed the substitution
out["substitute_calls"] = calls
out["tripwire_calls_after_local_recompute"] = trip
out["original_caches"] = dict(
    matrices_before_gate=mark_matrices, matrices_end=ORIG_MATRICES.cache_info()._asdict(),
    gain_before_gate=mark_gain, gain_end=ORIG_GAIN.cache_info()._asdict(),
    note=("in the pinned run, unchanged counters from before_gate to end mean no caller reached the original "
          "functions after they were replaced"))
out["software_state_equals_gate_record"] = A.software_state() == payload["software"]
write_and_exit(0)
