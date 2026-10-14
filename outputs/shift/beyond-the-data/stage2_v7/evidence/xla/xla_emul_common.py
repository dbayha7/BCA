"""Shared set-up for the information-only XLA evidence in this folder (V7_SPEC.md, version 3: C1 "Still to do", C7
"Cloud evidence" and C7 step 2b). NOTHING HERE IS A v7 RESULT. These runs are made on the cloud's Xeon, outside the
v7 wrapper (launch_v7.sh and run_stage2_v7.py are not used), before prereg_v7.json exists or pins anything. The
laptop's own self-test (V7_SPEC C11, P1 and P2) replaces every number made here.

Two modes, checked before any computation:
  default   the cloud's processor as it is: exactly launch_v7.sh's eight variables (v7_common.PHASE_ENV), nothing else.
  emulated  an AVX2-only processor (like the laptop's Core Ultra 9 275HX, which has no AVX-512) emulated on the cloud's
            Xeon, as in scratchpad/v7_hw_probe/haswell_npyoff.json and V7/evidence/run_probes.sh: the eight variables
            plus XLA_FLAGS=--xla_cpu_max_isa=AVX2 (XLA, the compiler behind JAX, generates no AVX-512 code),
            OPENBLAS_CORETYPE=Haswell (the matrix routines of numpy and scipy use their AVX2 kernels) and
            NPY_DISABLE_CPU_FEATURES=<the AVX-512 list> (numpy's own AVX-512 code paths off). After numpy is imported
            the emulation is confirmed: no AVX-512 dispatch target active in numpy, and both OpenBLAS libraries report
            the Haswell kernel.

What an emulation cannot show: XLA still tunes its code for the Xeon's model (only the instruction set is capped), and
libraries compiled ahead of time (inside jaxlib) are not affected by the flag. The laptop's Arrow Lake processor may
make other choices. So the emulation is a forecast only.

Every process also:
  * runs on one processor (os.sched_getaffinity must have exactly one element; the driver uses taskset -c 0);
  * substitutes the world constants of V7/world_constants.json exactly as V7/code/v7_constants.py does
    (v7_constants.apply: the file's SHA-256 and sidecar checked, A, B and K recomputed and compared with the file,
    the two gate functions replaced, the tripwire installed) -- except the feasibility repeat, which runs the gate
    with nothing substituted, as the review run it repeats did;
  * in emulated mode only, runs the gate's software check (experiments/shift/acceptance.py:491-502, called by
    load_prereg and guard, :958 and :980) with the XLA_FLAGS entry excused: the check compares the software state with
    XLA_FLAGS reported as the gate's pre-registration recorded it (unset), and still checks everything else in it (the
    dependency folder, the Python and library versions, the other numeric flags, float64 mode, the backend). Unexcused
    it refuses any XLA_FLAGS. In default mode the check runs unchanged. Each process records which.
No pinned file is edited, moved or deleted; nothing is written under runs/.
"""

import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("/home/user/bca")
V7 = ROOT / "outputs" / "shift" / "beyond-the-data" / "stage2_v7"
XLA_DIR = V7 / "evidence" / "xla"
SCRATCH = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/v7_build2")
if str(V7 / "code") not in sys.path:
    sys.path.insert(1, str(V7 / "code"))
import v7_common as K  # noqa: E402

NPYOFF = "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
EMULATION = {"XLA_FLAGS": "--xla_cpu_max_isa=AVX2", "OPENBLAS_CORETYPE": "Haswell", "NPY_DISABLE_CPU_FEATURES": NPYOFF}
MODES = ("default", "emulated")
INFO = ("Information only: run on the cloud's Xeon outside the v7 wrapper, before anything is pinned; not a v7 result. "
        "The laptop's own self-test replaces it.")


def check_mode(mode):
    """Problems with this process's environment for the mode (an empty list when it is exact)."""
    problems = []
    if mode not in MODES:
        return [f"unknown mode {mode!r}"]
    env = dict(os.environ)
    allowed = set(K.PHASE_ENV) | (set(EMULATION) if mode == "emulated" else set())
    extra = sorted(set(env) - allowed)
    if extra:
        problems.append(f"variables outside the mode: {extra}")
    for k, v in K.PHASE_ENV.items():
        if env.get(k) != v:
            problems.append(f"{k}={env.get(k)!r}, not {v!r}")
    if mode == "emulated":
        for k, v in EMULATION.items():
            if env.get(k) != v:
                problems.append(f"{k}={env.get(k)!r}, not {v!r}")
    aff = sorted(os.sched_getaffinity(0))
    if len(aff) != 1:
        problems.append(f"{len(aff)} processors visible ({aff}); exactly one is required")
    if os.getcwd() != str(ROOT):
        problems.append(f"working folder {os.getcwd()}, not {ROOT}")
    return problems


def env_facts(mode):
    """The environment and the processor, after numpy is imported; confirms the emulation (or its absence)."""
    import numpy
    import scipy
    import jax
    import jaxlib
    import v7_verify as VF
    f = VF.machine_facts()
    avx512 = [t for t in (f.get("numpy_dispatch_active") or []) if t.upper().startswith("AVX512")]
    kernels = {k: (v or "") for k, v in (f.get("openblas") or {}).items()}
    if mode == "emulated":
        confirmed = not avx512 and bool(kernels) and all(v.lower() == "haswell" for v in kernels.values())
    else:
        confirmed = all(k not in os.environ for k in EMULATION)
    f.update(mode=mode, variables=dict(sorted(os.environ.items())), cwd=os.getcwd(), executable=sys.executable,
             numpy=numpy.__version__, scipy=scipy.__version__, jax=jax.__version__, jaxlib=jaxlib.__version__,
             numpy_avx512_dispatch_active=avx512, mode_confirmed=bool(confirmed),
             mode_meaning=("AVX2-only processor emulated on the cloud's Xeon (XLA capped at AVX2, OpenBLAS Haswell "
                           "kernels, numpy's AVX-512 paths off)" if mode == "emulated"
                           else "the cloud's Xeon as it is (no emulation)"))
    return f


def substitute_constants():
    """V7_SPEC C1 steps 1-4, exactly as the wrapper does them (V7/code/v7_constants.py)."""
    import v7_constants
    return v7_constants.apply(K.ROOT / K.CONSTANTS_REL, K.CONSTANTS_SHA256)


def gate_software_check(mode):
    """Default mode: the gate's check unchanged. Emulated mode: the gate's check with only XLA_FLAGS excused."""
    import experiments.shift.acceptance as A
    if mode != "emulated":
        return dict(state="unchanged (acceptance.check_software as written; XLA_FLAGS unset)")
    original_check, original_state = A.check_software, A.software_state
    seen = []

    def check_software(payload):
        recorded = ((payload.get("software") or {}).get("flags") or {}).get("XLA_FLAGS", "<not recorded>")

        def state():
            s = original_state()
            seen.append(s["flags"].get("XLA_FLAGS"))
            s["flags"] = dict(s["flags"], XLA_FLAGS=recorded)
            return s

        A.software_state = state
        try:
            return original_check(payload)
        finally:
            A.software_state = original_state

    A.check_software = check_software
    return dict(state=("XLA_FLAGS excused: acceptance.check_software runs with XLA_FLAGS reported as the gate's "
                       "pre-registration recorded it (unset); every other item it checks is still checked. Unexcused, "
                       "it refuses any XLA_FLAGS (experiments/shift/acceptance.py:115, :491-502)."),
                xla_flags_seen_by_the_check=seen)


def harness():
    """The room test's harness from its pinned bytes, through v6's stage2_common (itself checked against v6's pin)."""
    return K.load_v6_common().load_room_harness()


def write(path, payload):
    """Strict sorted JSON with a .sha256 sidecar, written once (V7/code/v7_common.write_json_once)."""
    return K.write_json_once(path, payload)


def refuse(path, out, reason, code=2):
    out["refused"] = str(reason)
    try:
        write(path, out)
    finally:
        sys.stderr.write(f"refused: {reason}\n")
        sys.stderr.flush()
        os._exit(code)


def now():
    return time.perf_counter()
