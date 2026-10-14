"""Stage 2, version 7: make the pinned world-constants file (V7_SPEC C1, "Evidence status" and "Tested by (b)").

What it does, in order:
  1. Refuses unless it runs in the cloud's setting: the exact eight-variable environment of launch_v7.sh (so
     OPENBLAS_CORETYPE, NPY_DISABLE_CPU_FEATURES and XLA_FLAGS are unset), a processor with AVX-512, OpenBLAS's
     default SkylakeX kernels in both numpy and scipy, the pinned Python /tmp/claude-0/bcavenv, the gate's
     pre-registration 68d9e877... and the gate's three constants files exactly as that pre-registration hashed them.
  2. Lists every physics the gate and the Stage 2 code build (the module constants STALL and NO_STALL, world_physics
     for both worlds at every stall slope the gate records, and every physics the gate's pre-registration and data
     records hold) and turns each into the argument tuples exactly as the gate's wrappers do:
     thruster_chain.matrices -> _matrices(float(gain), float(wave_turn)) (experiments/shift/thruster_chain.py:181)
     and recorder.textbook_gain -> _gain(float(gain), float(wave_turn), float(throttle_cost), float(gamma))
     (experiments/shift/recorder.py:89).
  3. Calls the gate's own unchanged functions through those wrappers and checks that each wrapper returned the very
     cached object _matrices / _gain hold for the tuple (so the tuple is the one the gate used).
  4. Runs the gate's own unchanged acceptance.load_prereg + guard with NO substitution, as Stage 2 does
     (outputs/shift/beyond-the-data/stage2/code/fresh.py:42-46; runs/shift_explore/room/code/harness.py:215-216).
     They accept only if this machine reproduces all 64 recorded anchors to the last bit. It then checks that the
     gate's anchor computation asked for no other tuple (the cache sizes did not change).
  5. Writes outputs/shift/beyond-the-data/stage2_v7/world_constants.json and its sidecar world_constants.json.sha256:
     A, B and K as float64 hexadecimal strings (float.hex, exact to the last bit) keyed by the exact argument tuples,
     with the kernels used, numpy/scipy versions, the processor model and flags.

The file holds no time and no timing, so a rerun on the same machine gives the same bytes. Written once: if the file
exists, the new text must equal it byte for byte, otherwise the script refuses and writes nothing. It writes nothing
else (in particular nothing under runs/) and runs no BCA code.

Usage (from /home/user/bca):
  env -i PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=/home/user/bca \\
      PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
      taskset -c 0 nice -n 10 /tmp/claude-0/bcavenv/bin/python \\
      outputs/shift/beyond-the-data/stage2_v7/code/make_world_constants.py
"""

import sys

sys.dont_write_bytecode = True

import ctypes  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path("/home/user/bca")
V7 = ROOT / "outputs" / "shift" / "beyond-the-data" / "stage2_v7"
OUT = V7 / "world_constants.json"
SCRIPT_REL = "outputs/shift/beyond-the-data/stage2_v7/code/make_world_constants.py"
PYTHON = "/tmp/claude-0/bcavenv/bin/python"
GATE_PREREG = ROOT / "runs" / "shift" / "preregistration.json"
GATE_PREREG_SHA256 = "68d9e8779e148b7bfd2fcc6dc32e0a1091e5bfbbcb83cef2e4b58e4f6ec60fab"
# The three files that compute A, B and K, and where the gate's pre-registration records their SHA-256.
CONSTANTS_CODE = (("experiments/shift/thruster_chain.py", "code_sha256"),
                  ("experiments/shift/recorder.py", "code_sha256"),
                  ("experiments/signal/gapped_lq.py", "dependency_sha256"))
# launch_v7.sh's exact environment (V7_SPEC C4): these eight and nothing else.
ENV_EXACT = {"PATH": None, "HOME": "/root", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONPATH": str(ROOT),
             "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"}
KERNEL = "SkylakeX"


def refuse(message):
    print(f"refused: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def canonical(payload):
    """The same strict sorted JSON as the gate and v6 (experiments/shift/acceptance.py:362-363)."""
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False)


# --------------------------------------------------------------------------------------------- machine facts

def cpuinfo():
    fields, flags = {}, []
    with open("/proc/cpuinfo", encoding="utf8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                break  # first processor only
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "flags":
                flags = sorted(value.split())
            elif key in ("vendor_id", "cpu family", "model", "model name", "stepping"):
                fields[key.replace(" ", "_")] = value
    return dict(fields, flags=flags, avx512_flags=[f for f in flags if f.startswith("avx512")],
                has_avx512f="avx512f" in flags, logical_processors=os.cpu_count(),
                processors_usable_by_this_process=len(os.sched_getaffinity(0)), machine=platform.machine())


def openblas_facts():
    """For numpy's and scipy's bundled OpenBLAS: the kernel family it picked, its build string and the library's SHA-256,
    read through the libraries' own functions (as outputs/shift/beyond-the-data/stage2_gpu/lib/envfacts.py:93-123)."""
    import numpy
    base = os.path.dirname(os.path.dirname(numpy.__file__))
    out = {}
    for pkg, pattern, core_fns, config_fns in (
            ("numpy", "numpy.libs/libscipy_openblas64_*.so",
             ("scipy_openblas_get_corename64_", "openblas_get_corename64_"),
             ("scipy_openblas_get_config64_", "openblas_get_config64_")),
            ("scipy", "scipy.libs/libscipy_openblas-*.so",
             ("scipy_openblas_get_corename", "openblas_get_corename"),
             ("scipy_openblas_get_config", "openblas_get_config"))):
        libs = sorted(glob.glob(os.path.join(base, pattern)))
        if len(libs) != 1:
            refuse(f"expected one OpenBLAS library for {pkg}, found {libs}")
        lib = ctypes.CDLL(libs[0])
        row = dict(library=os.path.basename(libs[0]), library_sha256=sha256_file(libs[0]))
        for key, names in (("kernel", core_fns), ("config", config_fns)):
            row[key] = None
            for fn in names:
                f = getattr(lib, fn, None)
                if f is not None:
                    f.restype = ctypes.c_char_p
                    row[key] = f().decode("ascii", "replace").strip()
                    break
        out[pkg] = row
    return out


def numpy_dispatch():
    from numpy._core._multiarray_umath import __cpu_baseline__, __cpu_dispatch__, __cpu_features__
    return dict(baseline=list(__cpu_baseline__), dispatch_targets=list(__cpu_dispatch__),
                dispatch_targets_active=[t for t in __cpu_dispatch__ if __cpu_features__.get(t)])


def glibc_version():
    try:
        f = ctypes.CDLL("libc.so.6").gnu_get_libc_version
        f.restype = ctypes.c_char_p
        return f().decode("ascii")
    except (OSError, AttributeError):
        return None


# --------------------------------------------------------------------------------------------- encoding

def hex_matrix(M):
    import numpy as np
    M = np.asarray(M)
    if M.dtype != np.float64 or M.ndim != 2:
        refuse(f"expected a float64 matrix, got {M.dtype} {M.shape}")
    rows = [[float(v).hex() for v in row] for row in M]
    back = np.array([[float.fromhex(s) for s in row] for row in rows], np.float64)
    if not np.array_equal(back.view(np.uint64), np.ascontiguousarray(M).view(np.uint64)):
        refuse("the hexadecimal strings do not round-trip bit for bit")
    return dict(shape=list(M.shape), dtype="float64", hex=rows,
                sha256_le_c_order=sha256_bytes(np.ascontiguousarray(M, "<f8").tobytes()))


def key_text(names, values):
    return ",".join(f"{n}={float(v).hex()}" for n, v in zip(names, values))


A_NAMES = ("gain", "wave_turn")
K_NAMES = ("gain", "wave_turn", "throttle_cost", "gamma")


def a_key(p):
    """thruster_chain.matrices' tuple, the same conversion as experiments/shift/thruster_chain.py:181."""
    return (float(p.gain), float(p.wave_turn))


def k_key(p):
    """recorder.textbook_gain's tuple, the same conversion as experiments/shift/recorder.py:89."""
    return (float(p.gain), float(p.wave_turn), float(p.throttle_cost), float(p.gamma))


# --------------------------------------------------------------------------------------------- main

def check_setting():
    env = dict(os.environ)
    extra = sorted(set(env) - set(ENV_EXACT))
    if extra:
        refuse(f"variables outside launch_v7.sh's eight are set: {extra}")
    wrong = sorted(k for k, v in ENV_EXACT.items() if v is not None and env.get(k) != v)
    if wrong or "PATH" not in env:
        refuse(f"environment differs from launch_v7.sh's in {wrong or ['PATH']}")
    if os.path.realpath(sys.executable) != os.path.realpath(PYTHON) or sys.prefix != "/tmp/claude-0/bcavenv":
        refuse(f"not the pinned Python {PYTHON} (running {sys.executable}, prefix {sys.prefix})")
    if Path.cwd() != ROOT:
        refuse(f"run from {ROOT}")
    if len(os.sched_getaffinity(0)) != 1:
        refuse("run on exactly one processor (taskset -c <n>), as every Stage 2 phase does")
    text = GATE_PREREG.read_bytes()
    if sha256_bytes(text) != GATE_PREREG_SHA256:
        refuse("the gate's pre-registration is not 68d9e877...")
    payload = json.loads(text)
    for rel, block in CONSTANTS_CODE:
        if sha256_file(ROOT / rel) != payload[block][rel]:
            refuse(f"{rel} differs from the gate's pre-registration")
    return payload


def main():
    t0 = time.perf_counter()
    gate_payload = check_setting()

    import numpy as np
    import scipy

    cpu = cpuinfo()
    if not cpu["has_avx512f"]:
        refuse("this processor has no AVX-512: the constants must be made on the cloud's processor")
    blas = openblas_facts()
    if any(blas[p]["kernel"] != KERNEL for p in blas):
        refuse(f"OpenBLAS kernels are {[(p, blas[p]['kernel']) for p in blas]}, not the default {KERNEL}")
    dispatch = numpy_dispatch()

    # The gate's own modules, imported as the gate imports them.
    import experiments.shift.thruster_chain as TC
    import experiments.shift.recorder as RC

    caches_at_import = dict(matrices=TC._matrices.cache_info().currsize, columns=TC._columns.cache_info().currsize,
                            gain=RC._gain.cache_info().currsize)
    if any(caches_at_import.values()):
        refuse(f"A, B or K was computed while importing the gate's modules: {caches_at_import}")

    # Every physics the gate and the Stage 2 code build (V7_SPEC C1; grep in the evidence folder's README).
    sources = [("experiments.shift.thruster_chain.STALL", TC.STALL),
               ("experiments.shift.thruster_chain.NO_STALL", TC.NO_STALL)]
    slopes = {1.5}
    for variant, block in sorted((gate_payload.get("world") or {}).items()):
        if not block:
            continue
        for name in ("stall_physics", "no_stall_physics"):
            if name in block:
                sources.append((f"runs/shift/preregistration.json world/{variant}/{name}", TC.Physics(**block[name])))
                slopes.add(float(block[name]["stall_slope"]))
    for slope in sorted(slopes):
        for world in TC.WORLDS:
            sources.append((f"experiments.shift.thruster_chain.world_physics({world!r}, {slope!r})",
                            TC.world_physics(world, slope)))
    records = sorted((ROOT / "runs" / "shift" / "pre" / "data").rglob("record.json"))
    n_records = 0
    for path in records:
        rec = json.loads(path.read_text(encoding="utf8"))
        if "physics" in rec:
            sources.append((f"{path.relative_to(ROOT).as_posix()} physics", TC.Physics(**rec["physics"])))
            n_records += 1

    # Compute through the gate's wrappers, exactly as every caller does, and pin each tuple to the cached object.
    matrices, gains, rows = {}, {}, []
    for label, physics in sources:
        ak, kk = a_key(physics), k_key(physics)
        AB = TC.matrices(physics)
        K = RC.textbook_gain(physics)
        if AB is not TC._matrices(*ak) or K is not RC._gain(*kk):
            refuse(f"{label}: the wrapper did not return the object cached for {ak} / {kk}")
        matrices.setdefault(ak, AB)
        gains.setdefault(kk, K)
        rows.append((key_text(A_NAMES, ak), key_text(K_NAMES, kk), label))
    sizes = dict(matrices=TC._matrices.cache_info().currsize, gain=RC._gain.cache_info().currsize)
    if sizes != dict(matrices=len(matrices), gain=len(gains)):
        refuse(f"cache sizes {sizes} differ from the tuples found {len(matrices)}, {len(gains)}")

    # The gate's own exact check, with nothing substituted: it accepts only if all 64 anchors match to the last bit.
    t1 = time.perf_counter()
    import experiments.shift.acceptance as A
    prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(prereg)
    gate_seconds = time.perf_counter() - t1
    after = dict(matrices=TC._matrices.cache_info().currsize, gain=RC._gain.cache_info().currsize)
    if after != sizes:
        refuse(f"the gate's checks asked for argument tuples not listed here: cache sizes {sizes} -> {after}")
    for ak, AB in matrices.items():
        if TC._matrices(*ak) is not AB:
            refuse("a cached A, B changed during the gate's checks")
    n_anchors = sum(1 for t in gate_payload["anchors"].values() for s in t.values()
                    for q in ("recording", "top", "recording_se", "top_se") if q in s)

    payload = dict(
        kind="stage2-v7-world-constants",
        version=1,
        statement=(
            "The cloud's exact ThrusterChain plant matrices A (13 x 13) and B (13 x 4) and textbook gain K (4 x 13), "
            "as returned by the gate's own unchanged functions experiments.shift.thruster_chain._matrices(gain, "
            "wave_turn) and experiments.shift.recorder._gain(gain, wave_turn, throttle_cost, gamma) on this machine "
            "(AVX-512, OpenBLAS SkylakeX kernels), for every argument tuple the gate and the Stage 2 code use. Each "
            "number is float.hex of the float64 value, exact to the last bit. Before this file was written, the gate's "
            "own load_prereg and guard accepted on this machine with nothing substituted, so these are the values "
            "behind the gate's 64 recorded anchors. v7 substitutes them at run time (V7_SPEC C1); no pinned file is "
            "edited."),
        encoding=("Matrices are lists of rows; each entry is Python's float.hex() of a float64 (read back with "
                  "float.fromhex, bit for bit). Keys are the argument tuples as 'name=float.hex(value)' joined by "
                  "commas, in the function's argument order; 'args' gives the same values as JSON numbers. "
                  "sha256_le_c_order is the SHA-256 of the matrix's little-endian float64 bytes in C order."),
        functions={
            "experiments.shift.thruster_chain._matrices": dict(
                argument_names=list(A_NAMES), returns="(A, B), read-only float64",
                called_by_wrapper="experiments/shift/thruster_chain.py:181 matrices(physics) -> "
                                  "_matrices(float(physics.gain), float(physics.wave_turn))",
                tuples={key_text(A_NAMES, k): dict(args=dict(zip(A_NAMES, k)),
                                                   args_hex=dict(zip(A_NAMES, (float(v).hex() for v in k))),
                                                   A=hex_matrix(AB[0]), B=hex_matrix(AB[1]))
                        for k, AB in sorted(matrices.items())}),
            "experiments.shift.recorder._gain": dict(
                argument_names=list(K_NAMES), returns="K, read-only float64",
                called_by_wrapper="experiments/shift/recorder.py:89 textbook_gain(physics) -> "
                                  "_gain(float(physics.gain), float(physics.wave_turn), float(physics.throttle_cost), "
                                  "float(physics.gamma))",
                tuples={key_text(K_NAMES, k): dict(args=dict(zip(K_NAMES, k)),
                                                   args_hex=dict(zip(K_NAMES, (float(v).hex() for v in k))),
                                                   K=hex_matrix(K))
                        for k, K in sorted(gains.items())}),
        },
        argument_sources=dict(
            physics_listed=len(rows), gate_data_records_read=n_records,
            by_tuple=[dict(matrices_key=mk, gain_key=gk, sources=[r[2] for r in rows if r[:2] == (mk, gk)])
                      for mk, gk in sorted({r[:2] for r in rows})],
            rule=("Every physics listed was turned into its tuples with the wrappers' own conversion, computed "
                  "through the public wrappers, and checked to return the object cached for that tuple. The gate's "
                  "load_prereg and guard asked for no other tuple.")),
        made_on=dict(
            cpu=cpu, openblas=blas, numpy=np.__version__, scipy=scipy.__version__, numpy_dispatch=dispatch,
            python=sys.version.split()[0], python_executable=PYTHON, glibc=glibc_version(),
            kernel_release=platform.release(),
            environment={k: os.environ.get(k) for k in sorted(set(ENV_EXACT) | {
                "OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "XLA_FLAGS", "JAX_ENABLE_X64",
                "JAX_DEFAULT_MATMUL_PRECISION", "BCA_SHIFT_DEPS"})},
            gate_dependency_directory=gate_payload["dependency_directory"]["path"]),
        gate=dict(
            preregistration="runs/shift/preregistration.json", preregistration_sha256=GATE_PREREG_SHA256,
            constants_code_sha256={rel: sha256_file(ROOT / rel) for rel, _ in CONSTANTS_CODE},
            load_prereg_and_guard="accepted, nothing substituted (acceptance.load_prereg(acceptance.refuse_protected("
                                  "acceptance.RUNS_DEFAULT)) then acceptance.guard, unchanged)",
            anchors_in_preregistration=n_anchors, software_check="on (unchanged acceptance.check_software)"),
        made_by=dict(script=SCRIPT_REL, script_sha256=sha256_file(ROOT / SCRIPT_REL),
                     spec="V7_SPEC.md C1 (Evidence status; Tested by (b) and (d))"),
    )
    text = canonical(payload)
    digest = sha256_bytes(text.encode("utf8"))
    if OUT.exists():
        old = OUT.read_text(encoding="utf8")
        if old != text:
            refuse(f"{OUT} exists with different content (SHA-256 {sha256_bytes(old.encode('utf8'))}); "
                   f"it is written once")
        status = "unchanged (the file already held exactly these bytes)"
    else:
        OUT.write_text(text, encoding="utf8")
        status = "written"
    side = OUT.with_name(OUT.name + ".sha256")
    side_text = f"{digest}  {OUT.name}\n"
    if side.exists() and side.read_text(encoding="utf8") != side_text:
        refuse(f"{side} exists with different content")
    if not side.exists():
        side.write_text(side_text, encoding="utf8")
    print(json.dumps(dict(file=str(OUT.relative_to(ROOT)), sha256=digest, status=status,
                          matrices_tuples=list(payload["functions"]["experiments.shift.thruster_chain._matrices"]
                                               ["tuples"]),
                          gain_tuples=list(payload["functions"]["experiments.shift.recorder._gain"]["tuples"]),
                          physics_sources=len(rows), kernels={p: blas[p]["kernel"] for p in blas},
                          gate_seconds=round(gate_seconds, 2), seconds=round(time.perf_counter() - t0, 2),
                          finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
