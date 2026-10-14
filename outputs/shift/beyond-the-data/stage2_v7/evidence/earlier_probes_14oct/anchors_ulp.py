"""Read-only probe for the v7 design (hardware track). Recomputes the gate's anchor table exactly as
acceptance.load_prereg does (anchor_table + canonical JSON round trip) and compares every value with the recorded
pre-registration: count, absolute, relative and ULP differences. Also the textbook LQR gain K's bits, and (with
--log) whether one gate seed's log regenerates to its recorded hashes (acceptance.load_seed's check). Writes only to
the path given by --out (scratchpad). No BCA, no phase, nothing under runs/ is written."""
import sys
sys.dont_write_bytecode = True
import argparse, json, os, time
sys.path.insert(0, "/home/user/bca")
sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/lib")
import numpy as np
import envfacts as E

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--log", action="store_true")
a = ap.parse_args()
t0 = time.perf_counter()
import experiments.shift.acceptance as A
import experiments.shift.recorder as RC
import experiments.shift.thruster_chain as TC
import experiments.shift.dataset as DS
payload = json.loads((A.RUNS_DEFAULT / A.PREREG_NAME).read_text(encoding="utf8"))
cfg = A.config_from_dict(payload["config"])
K = np.asarray(RC.textbook_gain(TC.STALL), np.float64)
now = json.loads(A.canonical(A.anchor_table(A.RUNS_DEFAULT, cfg)))
rec = payload["anchors"]

def ulp(x, y):
    xi, yi = np.array([x], np.float64).view(np.int64)[0], np.array([y], np.float64).view(np.int64)[0]
    return int(abs(int(xi) - int(yi))) if (x >= 0) == (y >= 0) else None

rows = []
for k in sorted(rec):
    for s in sorted(rec[k]):
        for q in ("recording", "top", "recording_se", "top_se"):
            x, y = now[k][s][q], rec[k][s][q]
            rows.append(dict(key=f"{k}/{s}/{q}", now=x, rec=y, equal=x == y, abs=abs(x - y),
                             rel=abs(x - y) / abs(y) if y else None, ulp=ulp(x, y)))
diff = [r for r in rows if not r["equal"]]
out = dict(env={k: os.environ.get(k) for k in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "JAX_PLATFORMS",
                                                  "OMP_NUM_THREADS")},
           openblas=E.openblas_cores(), cpus=len(os.sched_getaffinity(0)),
           K_hex=[float(v).hex() for v in K.ravel()],
           n=len(rows), n_diff=len(diff), max_abs=max((r["abs"] for r in diff), default=0.0),
           max_rel=max((r["rel"] or 0 for r in diff), default=0.0),
           max_ulp=max((r["ulp"] or 0 for r in diff), default=0), ulps=sorted(r["ulp"] for r in diff),
           by_quantity={q: sum(1 for r in diff if r["key"].endswith("/" + q)) for q in
                        ("recording", "top", "recording_se", "top_se")},
           by_table={k: sum(1 for r in diff if r["key"].startswith(k + "/")) for k in sorted(rec)},
           worst=sorted(diff, key=lambda r: -r["abs"])[:3], seconds_anchors=time.perf_counter() - t0)
if a.log:
    t1 = time.perf_counter()
    r0 = A.read_json(A.data_dir(A.RUNS_DEFAULT, "base", "stall", 0) / "record.json")
    spec = RC.RecorderSpec(**r0["recorder"])
    lg = DS.generate(0, "stall", spec, tuple(r0["sizes"]), r0["stall_slope"])
    h = lg.array_hashes()
    out["log_seed0_stall"] = dict(equal=h == r0["log_sha256"], differing=sorted(k for k in h if h[k] != r0["log_sha256"].get(k)),
                                  seconds=time.perf_counter() - t1)
open(a.out, "w").write(json.dumps(out, indent=1, sort_keys=True))
print(json.dumps({k: out[k] for k in ("env", "openblas", "n_diff", "max_abs", "max_rel", "max_ulp", "seconds_anchors")}
                 | ({"log": out["log_seed0_stall"]} if a.log else {})))
