"""EXPLORATORY dry check for a seed-0 rerun of result F in the cloud sandbox. Read only: no training, no record written.

Imports the sandbox's byte-identical copy of the weighting test's code, then probes every gate that result F's scripts
(and the stages they need) check, for data seed 0, medium-expert and expert. Prints JSON.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
IW = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun")
CHANNEL = Path("/home/user/bca/outputs/shift_iw/explore/channel")
COMMITTED_V5 = Path("/home/user/bca/outputs/shift_iw/v5")
sys.path.insert(0, str(IW / "repo"))

from experiments.shift_iw import run as RN          # noqa: E402
from experiments.shift_iw import learners_iw as LI  # noqa: E402

out = {"label": "EXPLORATORY dry check (read only)", "run_module": RN.__file__, "root": str(RN.ROOT_DEFAULT)}


def probe(name, fn):
    try:
        v = fn()
        out[name] = {"ok": True, "value": v}
    except BaseException as e:  # noqa: BLE001  (Refused, SystemExit, ...)
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, CHANNEL / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


code_now = LI.code_hashes()
for q in ("medium-expert", "expert"):
    ctx = RN.Context(quality=q)
    spec = RN.spec_digest(ctx)
    body = RN.frozen_spec(ctx, spec)
    for stage in ("band", "coverage", "bca", "bca_lower_edge", "bca_estimated", "bca_lower_edge_estimated"):
        probe(f"{q}:approval:{stage}", lambda: (RN.approval_for(ctx, stage), "verified")[1])
    probe(f"{q}:data_record", lambda: RN.require(RN.data_record_path(ctx, 0), spec, "data")["finished_at"])
    probe(f"{q}:host_gap_record", lambda: RN.require_host(ctx, "gap", 0, spec)["finished_at"])
    probe(f"{q}:room_record", lambda: RN.require(RN.room_dir(ctx, 0) / "record.json", spec, "room")["finished_at"])
    probe(f"{q}:room_lower_edge_record",
          lambda: RN.require(RN.room_le_dir(ctx, 0) / "record.json", spec, "room-lower-edge")["finished_at"])
    probe(f"{q}:band_gap_record", lambda: RN.require_band(ctx, "gap", 0, spec)["finished_at"])
    probe(f"{q}:report_gate_for_band(require_design)", lambda: RN.require_design(ctx, spec, body))
    # the committed version-5 report, as a candidate read-only input to require_design (not linked; only read here)
    rep_path = COMMITTED_V5 / q / "report.json"
    rep = json.loads(rep_path.read_text())
    side = (COMMITTED_V5 / q / "report.json.sha256").read_text().split()[0]
    out[f"{q}:committed_report"] = dict(
        path=str(rep_path), sidecar_matches=side == hashlib.sha256(rep_path.read_bytes()).hexdigest(),
        status=rep.get("status"), spec_matches=rep.get("spec_sha256") == spec, seeds=rep.get("seeds"),
        D1=rep["marks"]["D1"]["status"], code_matches_sandbox_copy=rep.get("code_sha256") == code_now,
        M1=rep["marks"]["M1"]["status"], L1=rep["marks"]["L1"]["status"], L2=rep["marks"]["L2"]["status"],
        bca_lower_edge_allowed=rep.get("bca_lower_edge_allowed"))

# result F's own scripts: import from the repository folder (read only), with the sandbox's package
F1 = load("cle_f1", "command_le_train_f1.py")
T = F1.T
probe("f1:passmarks_ok", F1.passmarks_ok)
probe("first_test:passmarks_ok", T.passmarks_ok)
out["f1:OUT"], out["first_test:OUT"], out["command_band_check:OUT"] = str(F1.OUT), str(T.OUT), str(T.CB.OUT)
out["f1:ARMS"], out["f1:this_code"] = list(F1.ARMS), F1.this_code()
EST = load("cle_est", "command_le_train_est.py")
probe("est:passmarks_ok", EST.passmarks_ok)
out["est:OUT"] = str(EST.OUT)
cfg = LI.settings()
out["settings"] = {k: getattr(cfg, k) for k in ("seed", "alpha", "updates", "batch_size", "discount", "lr",
                                                 "clone_steps", "actor_only_steps", "actor_only_chunk")
                   if hasattr(cfg, k)}
# T.World: loads data and host (read only), then stops at the first missing input
probe("f1:World(expert,0)", lambda: (T.World("expert", 0), "constructed")[1])
print(json.dumps(out, indent=1, default=str))
