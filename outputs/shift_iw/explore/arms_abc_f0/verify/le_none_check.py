#!/usr/bin/env python3
"""Is arm A's 'none' actor (medium-expert, expert, seed 0) the same actor as result F's 'no extra caution' actor
(le_none)? Compares protocol fields from the records, the actor parameters themselves, the first chunk's training
metrics and the seed-0 scores. Verification only: reads records and actors, writes verify/le_none_check.json.

Run with the sandbox's interpreter (it has flax for msgpack):
  PYTHONDONTWRITEBYTECODE=1 /tmp/claude-0/bcavenv/bin/python3 -I le_none_check.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from flax.serialization import msgpack_restore

SP = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad")
V5 = SP / "iw_rerun" / "repo" / "runs" / "shift_iw_v5"
EX = SP / "iw_rerun" / "repo" / "runs" / "shift_iw_v5_explore" / "arms_abc_f0"
OUT = SP / "routes2" / "arms" / "verify"
RF = json.loads((SP / "resultF" / "results" / "scores_seed0.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def leaves(t, prefix=""):
    if isinstance(t, dict):
        out = {}
        for k in sorted(t):
            out.update(leaves(t[k], f"{prefix}/{k}"))
        return out
    return {prefix: np.asarray(t)}


def compare(a, b):
    la, lb = leaves(a), leaves(b)
    assert la.keys() == lb.keys()
    mx = max(float(np.max(np.abs(la[k].astype(np.float64) - lb[k].astype(np.float64)))) for k in la)
    same = all(la[k].dtype == lb[k].dtype and la[k].shape == lb[k].shape and
               la[k].tobytes() == lb[k].tobytes() for k in la)
    return dict(bit_identical=same, max_abs_param_diff=mx)


out = {}
for q in ("medium-expert", "expert"):
    arec = json.loads((EX / "A" / q / "seed0" / "record.json").read_text())
    afile = EX / "A" / q / "seed0" / "actors_final.msgpack"
    lrec = json.loads((V5 / q / "room_lower_edge" / "seed0" / "record.json").read_text())
    lfile = V5 / q / "room_lower_edge" / "seed0" / "actors.msgpack"
    rrec = json.loads((V5 / q / "room" / "seed0" / "record.json").read_text())
    rfile = V5 / q / "room" / "seed0" / "actors.msgpack"
    A = msgpack_restore(afile.read_bytes())
    L = msgpack_restore(lfile.read_bytes())
    Rm = msgpack_restore(rfile.read_bytes())
    le = lrec["room_lower_edge"]
    ev = json.loads((EX / "eval" / "A" / q / "seed0" / "record.json").read_text())["evaluations"]["none"]
    lev = RF["scores"][q]["le_none"]
    out[q] = dict(
        files=dict(arm_A_actors=str(afile), arm_A_actors_sha_ok=sha(afile) == arec["actors_final_sha256"],
                   le_actors=str(lfile), le_actors_sha_ok=sha(lfile) == lrec["actors_sha256"],
                   room_actors=str(rfile), room_actors_sha_ok=sha(rfile) == rrec["actors_sha256"]),
        protocol=dict(
            clone_sha256=dict(arm_A=arec["start_sha256"], le_none=le["training"]["start_sha256"],
                              le_clone_record=le["clone"]["actor_sha256"],
                              resultF_fact=RF["learner_facts"][f"{q}/le_none"]["clone_actor_sha256"]),
            rows_sha256=dict(arm_A=arec["rows_sha256"], le_none=le["training"]["rows_sha256"]),
            steps=dict(arm_A=f"5 chunks x 4,000 (record chunk_metrics: {len(arec['chunk_metrics'])} chunks)",
                       le_none=f"{le['training']['steps']} in chunks of {le['training']['chunk']}"),
            program=dict(arm_A="learners.ActorOnly.program (float32 step), 12 arms vmapped side by side "
                               f"({', '.join(arec['arms'])})",
                         le_none="lower_edge.Runner kind 'oracle': one float64-step program under jax enable_x64 "
                                 "(Adam's bias correction in float64, then rounded), 3 arms vmapped "
                                 f"({', '.join(a['name'] for a in le['training']['arms'])})"),
            penalty=dict(arm_A="none: Arm('none', 'even', 1.0), cloning weight 1 on every row, no penalty term",
                         le_none=f"k = {le['k']['le_none']} times the exact one-step oracle width W (the term is "
                                 "multiplied by 0)"),
            alpha="both runners take settings.alpha (2.5); the first chunk's lambda agrees to about 1e-6",
            first_chunk_metrics=dict(names=arec["metric_names"], arm_A=arec["chunk_metrics"][0][0],
                                     le_none=le["training"]["chunk_metrics"][0][0][:4])),
        params=dict(armA_none_vs_le_none=compare(A["none"], L["le_none"]),
                    armA_none_vs_room_none=compare(A["none"], Rm["none"]),
                    le_none_vs_room_none=compare(L["le_none"], Rm["none"]),
                    le_record_check_b=le["check_b"]["max_param_diff"]),
        scores=dict(arm_A_none={law: ev[law]["points"] for law in ("none", "mild", "strong")},
                    le_none={law: lev[law]["points"] for law in ("none", "mild", "strong")},
                    arm_A_none_J_strong=ev["strong"]["J"], le_none_J_strong=lev["strong"]["J"],
                    strong_points_diff_A_minus_le=ev["strong"]["points"] - lev["strong"]["points"],
                    sources=dict(arm_A=str(EX / "eval" / "A" / q / "seed0" / "record.json"),
                                 le_none=str(SP / "resultF" / "results" / "scores_seed0.json"))))
(OUT / "le_none_check.json").write_text(json.dumps(out, indent=1))
for q, v in out.items():
    print(q, json.dumps(v["files"]), json.dumps(v["params"]), json.dumps(v["scores"]))
    print("  clone", v["protocol"]["clone_sha256"]["arm_A"] == v["protocol"]["clone_sha256"]["le_none"],
          "rows", v["protocol"]["rows_sha256"]["arm_A"] == v["protocol"]["rows_sha256"]["le_none"])
    print("  first chunk", v["protocol"]["first_chunk_metrics"])
