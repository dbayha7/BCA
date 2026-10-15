"""Stage 2 v7 evidence (information only; V7_SPEC C1 "Tested by (d)"). Reads files, imports no study code, writes two
JSON files into this folder and nothing else:

  crosscheck_constants.json: world_constants.json against the earlier session's own dumps and printouts
    - scratchpad/v7_hw_probe/plant_default.npz (A, B, K under the default SkylakeX kernels, 13 Oct 23:59 UTC):
      expected bit-identical;
    - scratchpad/v7_hw_probe/plant_Haswell.npz (Haswell kernels, numpy AVX-512 dispatch off): the differences the spec
      quotes (71 of A's 169 entries, largest 1.7e-18; all 52 of K, largest 7.0e-15);
    - the K_hex lists in scratchpad/v7_hw_probe/default.json, haswell_npyoff.json and core_*.json: K's differences per
      kernel family (the spec: at most 1.04e-14 across the laptop's families).
  npz_sha256.json: the SHA-256 and size of every .npz file the v7 probes produced (the .npz files stay out of git:
    .gitignore:12), in scratchpad/v7_hw_probe/, scratchpad/isa_probe/ and scratchpad/v7_build/.

Usage: /tmp/claude-0/bcavenv/bin/python -I crosscheck_constants.py
"""
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/user/bca")
CONSTANTS = ROOT / "outputs/shift/beyond-the-data/stage2_v7/world_constants.json"
SP = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad")
PROBE = SP / "v7_hw_probe"
NPZ_DIRS = (PROBE, SP / "isa_probe", SP / "v7_build")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def from_hex(rows):
    return np.array([[float.fromhex(s) for s in r] for r in rows], np.float64)


def ulp_gap(x, y):
    xi, yi = np.asarray(x, np.float64).view(np.int64), np.asarray(y, np.float64).view(np.int64)
    same = np.signbit(x) == np.signbit(y)
    g = np.abs(xi.astype(object) - yi.astype(object))
    return int(max((int(v) for v, s in zip(g.ravel(), same.ravel()) if s), default=0))


def compare(file_m, other):
    file_m, other = np.asarray(file_m, np.float64), np.asarray(other, np.float64)
    neq = file_m.view(np.uint64) != other.view(np.uint64)
    d = np.abs(file_m - other)
    return dict(entries=int(neq.size), differing=int(neq.sum()), max_abs=float(d.max()),
                max_ulp=ulp_gap(file_m[neq], other[neq]) if neq.any() else 0, bit_identical=not bool(neq.any()))


cf_text = CONSTANTS.read_bytes()
cf = json.loads(cf_text)
mats = list(cf["functions"]["experiments.shift.thruster_chain._matrices"]["tuples"].values())
gains = list(cf["functions"]["experiments.shift.recorder._gain"]["tuples"].values())
assert len(mats) == 1 and len(gains) == 1, "the earlier dumps hold one tuple each"
A, B, K = from_hex(mats[0]["A"]["hex"]), from_hex(mats[0]["B"]["hex"]), from_hex(gains[0]["K"]["hex"])

out = dict(kind="stage2-v7-evidence-crosscheck", constants_sha256=hashlib.sha256(cf_text).hexdigest(),
           constants_tuples=dict(matrices=mats[0]["args"], gain=gains[0]["args"]))
for name in ("plant_default.npz", "plant_Haswell.npz"):
    p = PROBE / name
    with np.load(p, allow_pickle=False) as z:
        out[name] = dict(sha256=sha256_file(p), A=compare(A, z["A"]), B=compare(B, z["B"]), K=compare(K, z["K"]))
rows = {}
for p in sorted(PROBE.glob("*.json")):
    d = json.loads(p.read_text(encoding="utf8"))
    if "K_hex" not in d:
        continue
    k_other = np.array([float.fromhex(s) for s in d["K_hex"]], np.float64).reshape(K.shape)
    rows[p.name] = dict(sha256=sha256_file(p), OPENBLAS_CORETYPE=d.get("env", {}).get("OPENBLAS_CORETYPE"),
                        kernels=d.get("openblas"), K=compare(K, k_other), anchors_differing=d.get("n_diff"))
out["K_hex_in_earlier_jsons"] = rows
fam = {}
for r in rows.values():
    kern = (r["kernels"] or {}).get("numpy")
    if kern is None:
        continue
    f = fam.setdefault(kern, dict(files=0, max_abs_dK=0.0, K_differing=set(), anchors_differing=set()))
    f["files"] += 1
    f["max_abs_dK"] = max(f["max_abs_dK"], r["K"]["max_abs"])
    f["K_differing"].add(r["K"]["differing"])
    f["anchors_differing"].add(r["anchors_differing"])
out["by_kernel_family"] = {k: dict(v, K_differing=sorted(v["K_differing"]), anchors_differing=sorted(
    x for x in v["anchors_differing"] if x is not None)) for k, v in sorted(fam.items())}
(HERE / "crosscheck_constants.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf8")

npz = []
for d in NPZ_DIRS:
    for p in sorted(glob.glob(str(d / "**" / "*.npz"), recursive=True)):
        npz.append(dict(path=str(Path(p).relative_to(SP)), bytes=os.path.getsize(p), sha256=sha256_file(p)))
(HERE / "npz_sha256.json").write_text(json.dumps(dict(
    kind="stage2-v7-evidence-npz-sums", root=str(SP), note=("the .npz files stay in the cloud scratchpad, out of git "
                                                            "(.gitignore:12); none is copied into V7/"),
    files=npz), indent=1, sort_keys=True), encoding="utf8")
print(json.dumps(dict(plant_default=out["plant_default.npz"], plant_Haswell=out["plant_Haswell.npz"],
                      by_kernel_family=out["by_kernel_family"], npz_files=len(npz)), indent=1))
