"""Does lower_edge.Runner reproduce the saved version-5 band arm (expert seed 0, unweighted) with 4,000-step chunks, and
how far off is the 1,000-step split?"""
import importlib.util, json, sys
import jax, jax.numpy as jnp
sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/curves")
import curves as C
from experiments.shift import learners as LR
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import run as RN

wd = C.T.World("expert", 0)
clone, _ = wd.start()
a = RN.lower_edge_arm(wd.data, wd.prep, wd.brec, "unweighted", wd.R_u, None)
R, _, _ = LE.capped_thresholds(a["R"], a["pool"])
R32 = LE.as_float32_thresholds(R)
spec_ = dict(name="unweighted", k=1.0, R=R32, u_logged=LE.logged_widths(wd.band, wd.prep, R32, stage=LE.SCOPE))
p = C.EX / "expert" / "bca_lower_edge" / "unweighted" / "seed0"
saved = LR.load_tree(p / "actor.msgpack", LR.actor_template(), json.loads((p / "record.json").read_text())["actor_sha256"])
rec = json.loads((p / "record.json").read_text())
print("saved clone sha", rec["result"]["clone"].get("actor_sha256"), "this clone sha", LR.tree_sha256(clone))
runner = LE.Runner(wd.prep, clone, wd.host.critic.params, wd.ctx.cfg, 0, "band", LE.band_shared(wd.band, LE.SCOPE),
                   band=wd.band, stage=LE.SCOPE)
params, info = runner.run([spec_], separate=True)
print("Runner.run full chunks: max diff vs saved", LE.max_param_diff(params["unweighted"], saved))
print("saved R sha", rec["result"]["arm"].get("R_sha256"), "this R sha", LR.tree_sha256(R32))
