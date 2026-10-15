"""Stage 2, version 7, change C1: the pinned world constants (V7_SPEC.md C1, steps 1-4).

Every v7 phase or probe process calls apply() first, before any Stage 2 or gate computation:
  1. load_constants(): V7/world_constants.json, refused unless its SHA-256 equals the pin in prereg_v7.json (and its
     sidecar). A, B and K are read from float64 hexadecimal strings (float.fromhex, bit for bit), keyed by the exact
     argument tuples the gate uses: (gain, wave_turn) for A, B and (gain, wave_turn, throttle_cost, gamma) for K.
  2. local_differences() + check_tolerances(): A, B and K recomputed here with the gate's unchanged functions; refused
     unless max |A_local - A_file| <= 1e-14, max |K_local - K_file| <= 1e-12 and B is bit-identical. This only
     confirms that the file describes this world; it is never used to accept an anchor or a log.
  3. install(): experiments.shift.thruster_chain._matrices and experiments.shift.recorder._gain are replaced by
     functions that return the file's arrays (read-only, float64, the same objects on every call, as lru_cache gives);
     thruster_chain._columns' cache is cleared (it then rebuilds from the file's A and B), and so are the originals'
     caches, so that any stray call of an original reaches the tripwire. An argument tuple not in the file stops the
     process (exit 97). There is no fallback to computing it.
  4. install_tripwire(): experiments.signal.gapped_lq.transition_matrix and scipy.linalg.solve_discrete_are are
     replaced by functions whose first call writes one ledger line and calls os._exit(97). They never raise, because
     pinned v6 code catches any Exception around the stand-in rival (S2C/policy.py:419, :558).
No pinned file is edited: the replacements live in memory, in this process only. The gate's checks (load_prereg,
guard, load_seed) then run unchanged and stay exact to the last bit.

apply() imports the gate's thruster_chain (and through experiments/signal/gapped_lq.py JAX and Flax) but runs no JAX
computation: JAX picks its backend only at first use.
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as K  # noqa: E402

MAT_FN = "experiments.shift.thruster_chain._matrices"
GAIN_FN = "experiments.shift.recorder._gain"
TRIPWIRES = ("experiments.signal.gapped_lq.transition_matrix", "scipy.linalg.solve_discrete_are")


class Constants:
    """The file's arrays: ab {(gain, wave_turn): (A, B)}, k {(gain, wave_turn, throttle_cost, gamma): K}."""

    def __init__(self, path, sha256, ab, k, mat_names, gain_names):
        self.path, self.sha256, self.ab, self.k = str(path), sha256, ab, k
        self.mat_names, self.gain_names = tuple(mat_names), tuple(gain_names)


def _array(block):
    import numpy as np
    rows = block["hex"]
    M = np.array([[float.fromhex(s) for s in row] for row in rows], dtype=np.float64)
    if list(M.shape) != list(block["shape"]):
        raise K.V7Refusal(f"a matrix in the constants file has shape {M.shape}, not {block['shape']}")
    if K.sha256_bytes(np.ascontiguousarray(M, "<f8").tobytes()) != block["sha256_le_c_order"]:
        raise K.V7Refusal("a matrix in the constants file does not match its own recorded SHA-256")
    M = np.ascontiguousarray(M)
    M.setflags(write=False)
    return M


def load_constants(path, expected_sha256):
    """Step 1. V7Refusal unless the file's SHA-256 equals expected_sha256 and its sidecar."""
    path = Path(path)
    if not path.is_file():
        raise K.V7Refusal(f"the constants file {path} is missing")
    raw = path.read_bytes()
    digest = K.sha256_bytes(raw)
    if digest != expected_sha256:
        raise K.V7Refusal(f"the constants file's SHA-256 {digest[:16]}... is not the pinned {str(expected_sha256)[:16]}...")
    if K.sidecar_digest(path) != digest:
        raise K.V7Refusal("the constants file differs from its sidecar")
    body = json.loads(raw.decode("utf8"))
    if body.get("kind") != "stage2-v7-world-constants":
        raise K.V7Refusal("the constants file is not a v7 world-constants file")
    fm, fg = body["functions"][MAT_FN], body["functions"][GAIN_FN]
    mnames, gnames = tuple(fm["argument_names"]), tuple(fg["argument_names"])
    if mnames != ("gain", "wave_turn") or gnames != ("gain", "wave_turn", "throttle_cost", "gamma"):
        raise K.V7Refusal("the constants file's argument names are not the gate's")
    ab, kk = {}, {}
    for entry in fm["tuples"].values():
        key = tuple(float.fromhex(entry["args_hex"][n]) for n in mnames)
        ab[key] = (_array(entry["A"]), _array(entry["B"]))
    for entry in fg["tuples"].values():
        key = tuple(float.fromhex(entry["args_hex"][n]) for n in gnames)
        kk[key] = _array(entry["K"])
    if not ab or not kk:
        raise K.V7Refusal("the constants file holds no argument tuple")
    return Constants(path, digest, ab, kk, mnames, gnames)


def _gate_modules():
    import experiments.shift.thruster_chain as TC
    import experiments.shift.recorder as RC
    import experiments.signal.gapped_lq as GLQ
    import scipy.linalg as SL
    return TC, RC, GLQ, SL


def _compare(local, pinned):
    import numpy as np
    local, pinned = np.asarray(local, np.float64), np.asarray(pinned, np.float64)
    if local.shape != pinned.shape:
        return dict(shape_local=list(local.shape), shape_file=list(pinned.shape), max_abs=float("inf"),
                    differing=None, bit_identical=False)
    neq = np.ascontiguousarray(local).view(np.uint64) != np.ascontiguousarray(pinned).view(np.uint64)
    return dict(entries=int(local.size), differing=int(neq.sum()), max_abs=float(np.max(np.abs(local - pinned))),
                bit_identical=not bool(neq.any()))


def local_differences(consts, TC=None, RC=None):
    """Step 2's measurement: A, B, K recomputed with the gate's unchanged functions (through the public wrappers, as
    every caller does) against the file. The originals' caches are cleared afterwards."""
    if TC is None or RC is None:
        TC, RC = _gate_modules()[:2]
    if getattr(TC._matrices, "v7_pinned", False) or getattr(RC._gain, "v7_pinned", False):
        raise K.V7Refusal("the constants are already substituted in this process")
    rows = {}
    for key, (Af, Bf) in consts.ab.items():
        Al, Bl = TC.matrices(TC.Physics(**dict(zip(consts.mat_names, key))))
        rows[f"A {key}"] = _compare(Al, Af)
        rows[f"B {key}"] = _compare(Bl, Bf)
    for key, Kf in consts.k.items():
        Kl = RC.textbook_gain(TC.Physics(**dict(zip(consts.gain_names, key))))
        rows[f"K {key}"] = _compare(Kl, Kf)
    for fn in (TC._matrices, RC._gain, TC._columns):
        fn.cache_clear()
    summary = dict(max_abs_dA=max(v["max_abs"] for k, v in rows.items() if k.startswith("A ")),
                   max_abs_dK=max(v["max_abs"] for k, v in rows.items() if k.startswith("K ")),
                   B_bit_identical=all(v["bit_identical"] for k, v in rows.items() if k.startswith("B ")),
                   A_bit_identical=all(v["bit_identical"] for k, v in rows.items() if k.startswith("A ")),
                   K_bit_identical=all(v["bit_identical"] for k, v in rows.items() if k.startswith("K ")))
    return dict(rows=rows, summary=summary)


def check_tolerances(diffs, tol_a=K.TOL_A, tol_k=K.TOL_K):
    """Step 2's rule. V7Refusal unless max |dA| <= 1e-14, max |dK| <= 1e-12 and B is bit-identical."""
    s = diffs["summary"]
    if not s["B_bit_identical"]:
        raise K.V7Refusal("B computed here differs from the constants file (it must be bit-identical)")
    if not s["max_abs_dA"] <= tol_a:
        raise K.V7Refusal(f"A computed here differs from the constants file by {s['max_abs_dA']:.3g} > {tol_a:g}")
    if not s["max_abs_dK"] <= tol_k:
        raise K.V7Refusal(f"K computed here differs from the constants file by {s['max_abs_dK']:.3g} > {tol_k:g}")
    return s


def _on_refusal(reason):
    import v7_ledger
    v7_ledger.hard_exit(reason)


def install(consts, TC=None, RC=None, on_unknown=None):
    """Step 3. Replace the two gate functions by read-only cached returns of the file's arrays."""
    if TC is None or RC is None:
        TC, RC = _gate_modules()[:2]
    on_unknown = on_unknown or _on_refusal
    ab, kk = consts.ab, consts.k
    calls = {"matrices": 0, "gain": 0}

    def _matrices(gain, wave_turn):
        calls["matrices"] += 1
        got = ab.get((gain, wave_turn))
        if got is None:
            on_unknown(f"argument tuple {(gain, wave_turn)!r} for {MAT_FN} is not in the constants file")
        return got

    def _gain(gain, wave_turn, throttle_cost, gamma):
        calls["gain"] += 1
        got = kk.get((gain, wave_turn, throttle_cost, gamma))
        if got is None:
            on_unknown(f"argument tuple {(gain, wave_turn, throttle_cost, gamma)!r} for {GAIN_FN} is not in the "
                       f"constants file")
        return got

    for fn, name in ((_matrices, "_matrices"), (_gain, "_gain")):
        fn.v7_pinned = True
        fn.v7_calls = calls
        fn.__qualname__ = fn.__name__ = name
    originals = dict(matrices=TC._matrices, gain=RC._gain)
    originals["matrices"].cache_clear()
    originals["gain"].cache_clear()
    TC._matrices = _matrices
    RC._gain = _gain
    TC._columns.cache_clear()
    return dict(originals=originals, calls=calls)


def install_tripwire(GLQ=None, SL=None, on_call=None):
    """Step 4. The first call of either function stops the process at once (exit 97)."""
    if GLQ is None or SL is None:
        _, _, GLQ, SL = _gate_modules()
    on_call = on_call or _on_refusal
    state = {name: 0 for name in TRIPWIRES}

    def wire(name):
        def tripped(*args, **kwargs):
            state[name] += 1
            on_call(f"tripwire: {name} was called after the constants were substituted (no caller may reach it)")
            raise SystemExit(K.EXIT_V7)   # only reached when on_call returns (tests); the real on_call never returns
        tripped.v7_tripwire = True
        return tripped

    GLQ.transition_matrix = wire(TRIPWIRES[0])
    SL.solve_discrete_are = wire(TRIPWIRES[1])
    solvers = sys.modules.get("scipy.linalg._solvers")
    if solvers is not None and hasattr(solvers, "solve_discrete_are"):
        solvers.solve_discrete_are = SL.solve_discrete_are
    return state


def apply(constants_path, expected_sha256, on_refusal=None):
    """Steps 1-4 in order. Returns the facts the ledger records (file SHA-256, the differences, the tripwire state).
    V7Refusal on step 1-2 failures; steps 3-4 stop the process themselves (exit 97) when they fire later."""
    for name in ("experiments.shift.thruster_chain", "experiments.shift.recorder"):
        mod = sys.modules.get(name)
        if mod is not None and (getattr(getattr(mod, "_matrices", None), "v7_pinned", False)
                                or getattr(getattr(mod, "_gain", None), "v7_pinned", False)):
            raise K.V7Refusal("the constants were substituted already in this process")
    consts = load_constants(constants_path, expected_sha256)
    TC, RC, GLQ, SL = _gate_modules()
    at_import = dict(matrices=TC._matrices.cache_info().currsize, gain=RC._gain.cache_info().currsize,
                     columns=TC._columns.cache_info().currsize)
    if any(at_import.values()):
        raise K.V7Refusal(f"A, B or K was computed before the constants could be substituted: caches {at_import}")
    diffs = local_differences(consts, TC, RC)
    summary = check_tolerances(diffs)
    installed = install(consts, TC, RC, on_unknown=on_refusal)
    trip = install_tripwire(GLQ, SL, on_call=on_refusal)
    return dict(file=consts.path, sha256=consts.sha256, tolerances=dict(A=K.TOL_A, K=K.TOL_K, B="bit-identical"),
                differences=summary, per_tuple={k: {kk: v[kk] for kk in ("max_abs", "differing", "bit_identical")}
                                                for k, v in diffs["rows"].items()},
                tuples=dict(matrices=[list(k) for k in consts.ab], gain=[list(k) for k in consts.k]),
                caches_at_import=at_import, tripwire=dict(installed=list(TRIPWIRES), calls=trip),
                substitute_calls=installed["calls"])
