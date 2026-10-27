"""Spot check (read only): recompute arm A's K2 value (A1 - none, strong test points, 8,000-step actors) and arm C's KB1
critic gains from the saved actors, and compare with the records. Writes only verify/spot_check.json.

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<iw>/repo:<iw>/explore:<arms>/code python3 -B spot_check.py Q:K ...
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import arm_a  # noqa: E402
import arm_c  # noqa: E402
import common as C  # noqa: E402

out = {}
for item in sys.argv[1:]:
    q, k = item.split(":")
    k = int(k)
    t0 = time.time()
    S = C.Seed(q, k)
    rec, p8 = arm_a.load_actors(q, k, "8000")
    strong = {"strong": S.banks["strong"]}
    e_none = C.evaluate_policy(C.actor_policy(p8["none"], S.prep), strong, S.anchors)["strong"]
    e_a1 = C.evaluate_policy(C.actor_policy(p8["A1"], S.prep), strong, S.anchors)["strong"]
    d = e_a1["points"] - e_none["points"]
    rec_d = rec["kills"]["K2_value"]["d"]
    crec, c8 = arm_c.load_actors(q, k, "8000")
    critic = S.host.critic.params
    q_rec = C.q_min(critic, S.obs, S.a_rec)
    G = {n: float(np.mean(C.q_min(critic, S.obs, C.actor_out(c8[n], S.obs)) - q_rec)) for n in ("rho1", "rho0.5")}
    out[item] = dict(K2_recomputed=d, K2_recorded=rec_d, K2_equal=bool(d == rec_d), K2_absdiff=abs(d - rec_d),
                     none_points=e_none["points"], A1_points=e_a1["points"],
                     G_rho1_recomputed=G["rho1"], G_rho1_recorded=crec["kills"]["KB1"]["G_rho1"],
                     G_rho05_recomputed=G["rho0.5"], G_rho05_recorded=crec["kills"]["KB1"]["G_rho05"],
                     G_equal=bool(G["rho1"] == crec["kills"]["KB1"]["G_rho1"] and G["rho0.5"] == crec["kills"]["KB1"]["G_rho05"]),
                     seconds=time.time() - t0)
    print(item, json.dumps(out[item]), flush=True)
p = Path(__file__).resolve().parent / "spot_check.json"
p.write_text(json.dumps(out, indent=1))
