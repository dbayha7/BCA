"""Spot check (read only): rebuild BCA's thruster ceilings (rule 'bca' only) for arm B's host at seed 0 and for arm C's
final rho 0.5 actor at every seed, with the frozen code's own functions, and compare with the records' maps, the
thruster-1 tries in rung (0.80, 0.85] and the K1 / guard-void decision. Writes only verify/spot_maps.json.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import arm_b as B  # noqa: E402
import arm_c  # noqa: E402
import common as C  # noqa: E402

def norm(v):
    if v is None:
        return None
    return float(v)


def stats(rungs):
    return [norm(r["statistic"]) for r in rungs]


QS = ("random", "medium-replay", "medium", "medium-expert", "expert")
out = {}
for q in QS:
    for k in range(5):
        t0 = time.time()
        S = C.Seed(q, k)
        tries = B.Tries(S)
        res = {}
        if k == 0:
            pi = C.actor_out(S.host.actor.params, tries.obs)
            maps, det = B.maps_for(tries, pi, 0, rules=("bca",))
            rec = json.loads((C.arm_dir("B", q, 0) / "record.json").read_text())
            k1 = B.k1_numbers(det, maps)
            res["B"] = dict(map=maps["bca"], recorded=rec["maps"]["bca"], equal=maps["bca"] == rec["maps"]["bca"],
                            t1=k1["t1"], tries_080_085=k1["tries_080_085"], K1=k1["fires"],
                            recorded_K1=rec["kills"]["K1"],
                            stats_equal=stats(det["rules"]["bca"][0]["rungs"]) == stats(rec["details"]["rules"]["bca"][0]["rungs"]),
                            stats_all_thrusters_equal=all(stats(det["rules"]["bca"][t]["rungs"]) == stats(rec["details"]["rules"]["bca"][t]["rungs"]) for t in range(4)),
                            tries_all_equal=all([r["tries"] for r in det["rules"]["bca"][t]["rungs"]] == [r["tries"] for r in rec["details"]["rules"]["bca"][t]["rungs"]] for t in range(4)))
        crec, cf = arm_c.load_actors(q, k, "final")
        pi05 = C.actor_out(cf["rho0.5"], tries.obs)
        maps, det = B.maps_for(tries, pi05, arm_c.TAG_OFFSET, rules=("bca",))
        k1 = B.k1_numbers(det, maps)
        res["C"] = dict(map=maps["bca"], recorded=crec["maps"]["bca"], equal=maps["bca"] == crec["maps"]["bca"],
                        t1=k1["t1"], tries_080_085=k1["tries_080_085"], void=k1["fires"],
                        recorded_void=crec["kills"]["guard_void_own_map"],
                        stats_equal=stats(det["rules"]["bca"][0]["rungs"]) == stats(crec["details"]["rules"]["bca"][0]["rungs"]),
                        stats_all_thrusters_equal=all(stats(det["rules"]["bca"][t]["rungs"]) == stats(crec["details"]["rules"]["bca"][t]["rungs"]) for t in range(4)),
                        tries_all_equal=all([r["tries"] for r in det["rules"]["bca"][t]["rungs"]] == [r["tries"] for r in crec["details"]["rules"]["bca"][t]["rungs"]] for t in range(4)))
        res["seconds"] = time.time() - t0
        out[f"{q}:{k}"] = res
        print(f"{q}:{k}", json.dumps(res, default=float), flush=True)
(Path(__file__).resolve().parent / "spot_maps.json").write_text(json.dumps(out, indent=1, default=float))
