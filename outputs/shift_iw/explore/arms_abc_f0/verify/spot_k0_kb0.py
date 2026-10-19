"""Spot check (read only): recompute arm A's K0 release shares at refresh 0 (the clone actor) and arm C's KB0 shares
(thruster-1 commands past float32(0.8) in each final actor's unclipped strong-start rollouts) from the inputs and the
saved actors, and compare with the records. Writes only verify/spot_k0_kb0_<mode>.json.

  python -B spot_k0_kb0.py k0 Q:K ...     K0 (rebuilds the clone, reads the band at the recorded commands)
  python -B spot_k0_kb0.py kb0 Q:K ...    KB0 shares from C's saved final actors
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import arm_a as A  # noqa: E402
import arm_c  # noqa: E402
import common as C  # noqa: E402

mode = sys.argv[1]
out = {}
for item in sys.argv[2:]:
    q, k = item.split(":")
    k = int(k)
    t0 = time.time()
    S = C.Seed(q, k)
    if mode == "k0":
        from experiments.shift import learners as LR
        from experiments.shift_iw import learners_iw as LI
        from experiments.shift_iw import run as R
        rec = json.loads((C.arm_dir("A", q, k) / "record.json").read_text())
        brec = R.finished(R.band_dir(S.ctx, "gap", k) / "record.json", S.spec)
        approval = R.approval_for(S.ctx, "band")
        band = LI.load_band(R.band_dir(S.ctx, "gap", k) / "band.msgpack", brec["band_sha256"], S.host, S.prep, S.cfg, k,
                            approval, brec["band"], stage="band")
        rd = LI.readout(band, LI.reference_of(band, brec["thresholds"]["unweighted"]), S.prep.train, stage="band")
        U = np.asarray(rd["width"], np.float64)
        a64 = np.asarray(S.a_rec, np.float32).astype(np.float64)
        lo, hi = np.quantile(a64, 0.005, axis=0), np.quantile(a64, 0.995, axis=0)
        rf = np.asarray(S.data.train_log.diagnostic("infos/random"), bool)[S.prep.rows]
        clone_params, clone_rec = LR.clone(S.prep, S.cfg, k)
        q_rec = C.q_min(S.host.critic.params, S.obs, S.a_rec)
        pi = C.actor_out(clone_params, S.obs)
        D = C.q_min(S.host.critic.params, S.obs, pi) - q_rec
        guard = A.guard_of(pi, lo, hi)
        rel = A.margin_release(D, guard, U)
        rec_share = float(rel[~rf].mean()) if (~rf).any() else None
        rnd_share = float(rel[rf].mean()) if rf.any() else None
        h0 = rec["history"][0]["arms"]["A1"]
        out[item] = dict(clone_sha256=LR.tree_sha256(clone_params), start_sha256_recorded=rec["start_sha256"],
                         clone_equal=LR.tree_sha256(clone_params) == rec["start_sha256"],
                         U_mean=float(U.mean()), U_mean_recorded=rec["inputs"]["U_mean"],
                         released_share=float(rel.mean()), released_share_recorded=h0["released_share"],
                         recorder_share=rec_share, recorder_share_recorded=h0.get("released_share_recorder_rows"),
                         random_share=rnd_share, random_share_recorded=h0.get("released_share_random_rows"),
                         equal=bool(rec_share == h0.get("released_share_recorder_rows")
                                    and rnd_share == h0.get("released_share_random_rows")
                                    and float(rel.mean()) == h0["released_share"]),
                         K0_pass=(rec_share is not None and rnd_share is not None and rec_share <= 0.10
                                  and rnd_share >= 0.15) if q in ("medium-replay", "medium", "medium-expert") else None,
                         seconds=time.time() - t0)
    else:
        crec, cf = arm_c.load_actors(q, k, "final")
        shares = {}
        for n in ("rho1", "rho0.5"):
            cmds = C.strong_commands(C.actor_policy(cf[n], S.prep, name=f"{q} seed {k} {n} (float64 copy)"), S.banks)
            shares[n] = C.past_share(cmds, [float(C.KNEE32)] * 4, thruster=0)
        kb = crec["kills"]["KB0_seed"]
        out[item] = dict(share_rho05=shares["rho0.5"], share_rho1=shares["rho1"], recorded_rho05=kb["share_rho05"],
                         recorded_rho1=kb["share_rho1"],
                         equal=bool(shares["rho0.5"] == kb["share_rho05"] and shares["rho1"] == kb["share_rho1"]),
                         seconds=time.time() - t0)
    print(item, json.dumps(out[item]), flush=True)
(Path(__file__).resolve().parent / f"spot_k0_kb0_{mode}.json").write_text(json.dumps(out, indent=1))
