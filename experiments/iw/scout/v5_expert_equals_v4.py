"""Proof that version 5's expert column is version 4's data bit for bit (no learning, no BCA).

Version 4's data were built by version 4's own code (a frozen copy of shift_iw/world.py and data.py taken before any
version 5 edit), at the given size and seed, into V4_DIR/seed<k> with hashes_seed<k>_<size>.json beside it. This script
builds version 5's expert column with the current code at the same size and seed and compares:
  - every array of every file that version 4 wrote (train_law, test_mild, test_strong, samples, truth), by SHA-256
    and by the arrays themselves (np.array_equal on the bytes);
  - the ball (gap), its find_gap record, the quadrature, D1, the removal summary's version 4 fields and the test logs'
    in-ball shares of world.json;
  - the arrays version 5 adds (infos/random) must be all False for the expert.
It also builds every other quality at the same size and seed and checks that the ball and D1 are identical across the
five qualities and equal to version 4's.
  python iw/scout/v5_expert_equals_v4.py V4_DIR SEED SIZE OUT_DIR      SIZE: small | tiny
Writes OUT_DIR/v5_expert_equals_v4_seed<k>_<size>.json and prints the verdict."""

import json
import sys
import time
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401
from experiments.shift_iw import data as D
from experiments.shift_iw import world as W

TINY = D.Sizes(train=24, calibration=8, held_out=8, test=8, sample=8, grid_points=5, per_point=20, truth_rows=100,
               noise_map_horizon=50, start_bank=20)
FILES = ("train_law.hdf5", "test_mild.hdf5", "test_strong.hdf5", "samples.hdf5", "truth.hdf5")


def norm(x):
    """A record as JSON reads it back (tuples become lists)."""
    return json.loads(json.dumps(x, sort_keys=True))


def main():
    v4_dir, seed, size, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    sizes = {"small": D.SMALL, "tiny": TINY}[size]
    ref = json.loads((v4_dir / f"hashes_seed{seed}_{size}.json").read_text())
    if ref["sizes"] != sizes.as_dict():
        raise SystemExit("the version 4 build used other sizes")
    t0 = time.perf_counter()
    worlds = {}
    for q in W.QUALITIES:
        worlds[q], _ = D.build_seed(seed, D.data_dir(out, q), sizes, q, log=print)
    v5 = D.data_dir(out, "expert") / f"seed{seed}"
    v4 = v4_dir / f"seed{seed}"
    res = dict(seed=seed, size=size, sizes=sizes.as_dict(), files={}, world={}, qualities={})
    ok = True
    for name in FILES:
        a4, m4 = D.read_hdf5(v4 / name)
        a5, m5 = D.read_hdf5(v5 / name)
        same_hash = {k: D.sha256_array(a5[k]) == h for k, h in ref["arrays"][name].items() if k in a5}
        same_bytes = {k: bool(k in a5 and a4[k].dtype == a5[k].dtype and a4[k].shape == a5[k].shape
                              and a4[k].tobytes() == a5[k].tobytes()) for k in a4}
        missing = sorted(set(a4) - set(a5))
        added = sorted(set(a5) - set(a4))
        added_ok = all(not np.any(a5[k]) for k in added if k == "infos/random") and set(added) <= {"infos/random"}
        good = all(same_hash.values()) and all(same_bytes.values()) and not missing and added_ok \
            and len(same_hash) == len(ref["arrays"][name])
        res["files"][name] = dict(arrays_compared=len(a4), identical=good, missing_in_v5=missing, added_in_v5=added,
                                  added_all_false=added_ok,
                                  meta_keys_added=sorted(set(m5) - set(m4)),
                                  meta_keys_changed=sorted(k for k in set(m4) & set(m5) if m4[k] != m5[k]))
        ok &= good
    w5 = worlds["expert"]
    checks = dict(gap=norm(w5["gap"]) == norm(ref["gap"]), quadrature=w5["quadrature"] == ref["quadrature"],
                  quadrature_expert=w5["quadrature_expert"] == ref["quadrature"], D1=w5["D1"]["checks"] ==
                  ref["D1"]["checks"] and w5["D1"]["passes"] == ref["D1"]["passes"],
                  test_in_ball_share=w5["test_in_ball_share"] == ref["test_in_ball_share"],
                  removal_v4_fields={k: w5["removal"].get(k) == v for k, v in ref["removal"].items()})
    checks["removal_v4_fields_all"] = all(checks["removal_v4_fields"].values())
    res["world"] = checks
    ok &= all(v for k, v in checks.items() if k != "removal_v4_fields")
    for q, w in worlds.items():
        same = dict(gap=norm(w["gap"]) == norm(ref["gap"]), D1=w["D1"] == w5["D1"],
                    quadrature_expert=w["quadrature_expert"] == ref["quadrature"])
        res["qualities"][q] = dict(same, Z_own=w["quadrature"]["Z"]["value"],
                                   holds_removed_own=w["quadrature"]["holds_removed_train"]["value"],
                                   removal=w["removal"], test_in_ball_steps=w["test_in_ball_steps"])
        ok &= all(same.values())
    res["identical"] = bool(ok)
    res["seconds"] = time.perf_counter() - t0
    path = out / f"v5_expert_equals_v4_seed{seed}_{size}.json"
    path.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(dict(identical=res["identical"], files={k: v["identical"] for k, v in res["files"].items()},
                          world={k: v for k, v in checks.items() if k != "removal_v4_fields"},
                          qualities={q: {k: v for k, v in r.items() if k in ("gap", "D1", "quadrature_expert")}
                                     for q, r in res["qualities"].items()}), indent=1))
    print(f"written {path} in {res['seconds']:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
