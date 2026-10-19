"""EXPLORATORY diagnosis, seed 0 of the cloud rerun (no training, no fit, no new threshold): on hard-water training
states (the recording controller's own flag; where the 80% hold was the good command before the removal), read the
capped pull's final frozen band U(s, a) = R x eta(s, a) x unit and both critics' values at five commands per state:
  recorded   the row's own recorded command (the only place the capped pull ever reads the band)
  actor      the plain TD3+BC actor's command
  hold60     thruster 1 at 60%, thrusters 2-4 in the hold pattern scaled to it
  hold80     the 80% hold (thruster 1 at -0.8, thrusters 2-4 at -0.110, -0.468, -0.325): removed from the data
  full100    thruster 1 at 100%, the hold pattern scaled to it (past the knee: stalls)
Gate: learners_iw.require_gate (scope 'bca') before the band is read. Writes diag_band_at_commands.json."""
import json, time
from pathlib import Path
import numpy as np
import jax.numpy as jnp

import experiments.shift  # noqa: F401
from experiments.shift import learners as LR
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import run as R
import algorithms.td3_bc_bca as P
import explore_arms as XA
import explore_eval as XE
import jobs as J

OUT = Path(__file__).resolve().parent / "diag_band_at_commands.json"
spec = R.spec_digest(J.ctx_of("expert"))
approval = LI.verify_approval(J.V5 / "approval.json", stage="bca")
PAT = np.array([-0.8, -0.110, -0.468, -0.325]) / 0.8   # per unit of thruster-1 throttle
res = {}
for q in J.QUALITIES:
    ctx = J.ctx_of(q)
    data = R.load_data(ctx, J.SEED, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    LI.require_gate(approval, prep, "bca")
    hard = np.asarray(data.train_log.diagnostic("infos/hard_water"))[prep.rows].astype(bool)
    idx = np.flatnonzero(hard)
    args = LR.make_args(LI.settings(), J.SEED)
    config = LR.bca_config()
    _, _, mods = LR.bca_initial_state(args, config)
    d = XA.capped_dir(J.EXPLORE_OUT, q, "unweighted", J.SEED)
    rec = XE.explore_record(d / "record.json", spec)
    st = LR.load_tree(d / "snapshot_100.msgpack", LI.bca_template(LI.settings(), J.SEED), rec["result"]["snapshots"][-1]["file_sha256"])
    learners = J.learners_of(ctx, spec)
    host_actor, host_critic, _ = learners["host_gap"]
    obs = prep.train.obs[idx]
    n = len(idx)
    cmds = dict(recorded=np.asarray(prep.train.action[idx]),
                actor=np.clip(np.asarray(mods[0].apply(host_actor, obs)), -1, 1),
                hold60=np.tile(PAT * 0.6, (n, 1)), hold80=np.tile(PAT * 0.8, (n, 1)), full100=np.tile(PAT * 1.0, (n, 1)))
    thr, unit = float(st.posterior.threshold), float(st.posterior.residual_scale)
    body = dict(rows=int(n), threshold=thr, unit=unit)
    Uk = {}
    for k, a in cmds.items():
        a = jnp.asarray(a, jnp.float32)
        eta = np.asarray(mods[2].apply(st.posterior.cal_params, obs, a), np.float64)
        U = thr * np.maximum(eta, 1e-6) * unit
        dose = 1 + 0.5 * U / (U + unit)
        qb = np.asarray(mods[1].apply(st.native.critic.params, obs, a).min(-1), np.float64)
        qh = np.asarray(mods[1].apply(host_critic, obs, a).min(-1), np.float64)
        Uk[k] = U
        body[k] = dict(U_over_unit=float((U / unit).mean()), dose_equiv=float(dose.mean()), q_bca=float(qb.mean()), q_host=float(qh.mean()))
    rel = (Uk["full100"] - Uk["hold80"]) / Uk["hold80"]
    sv = (Uk["hold80"] - Uk["hold80"].mean()) / Uk["hold80"].mean()
    body["full100_vs_hold80"] = dict(rel_mean=float(rel.mean()), rel_p5=float(np.percentile(rel, 5)), rel_p95=float(np.percentile(rel, 95)),
                                     share_wider_past80=float(np.mean(rel > 0)),
                                     state_spread_p5=float(np.percentile(sv, 5)), state_spread_p95=float(np.percentile(sv, 95)))
    res[q] = body
    print(q, n, " | ".join(f"{k}: U {body[k]['U_over_unit']:.2f} dose {body[k]['dose_equiv']:.3f} Qh {body[k]['q_host']:.1f}" for k in cmds), "| full100 vs hold80", {k: round(v, 3) for k, v in body["full100_vs_hold80"].items()}, flush=True)
OUT.write_text(json.dumps(dict(kind="capped-pull-band-at-commands", label="EXPLORATORY: seed 0 of the cloud rerun; no training, no fit", res=res), indent=1))
print("written", OUT)
