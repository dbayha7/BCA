"""EXPLORATORY diagnosis of BCA's capped pull, seed 0 of the cloud rerun. Trains nothing, fits no band, computes no new
threshold. It loads the finished seed-0 states (plain TD3+BC on the gapped log; BCA's capped pull, unweighted) and reads,
at every kept training row:
  - the capped pull's dose at its final frozen band (td3_bc_bca.bc_readout: the exact readout the last 5,000 updates
    used), after the package's own gate (learners_iw.require_gate, scope 'bca', the committed approval);
  - the recording controller's own flags: random command, 80% hold (none survive the removal), hard water;
  - the band's raw input at the row, |y - q| (one target-noise draw, the refresh's formula), in units of the band unit;
  - each actor's command at the row's state.
Writes diag_rows.json beside this file. Run from <iw_rerun>/repo with
PYTHONPATH=<iw_rerun>/repo:<iw_rerun>/explore:<iw_rerun>/driver."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

import experiments.shift  # noqa: F401
from experiments.shift import learners as LR
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import run as R
import algorithms.td3_bc_bca as P
import explore_arms as XA
import explore_eval as XE
import jobs as J

OUT = Path(__file__).resolve().parent / "diag_rows.json"
t0 = time.time()
ctx0 = J.ctx_of("expert")
spec = R.spec_digest(ctx0)
approval = LI.verify_approval(J.V5 / "approval.json", stage="bca")


def pct(x, qs=(5, 25, 50, 75, 95)):
    return {str(q): float(np.percentile(x, q)) for q in qs}


def grp(dose, mask):
    return dict(n=int(mask.sum()), dose_mean=float(dose[mask].mean()) if mask.any() else None)


out = {}
for q in J.QUALITIES:
    ctx = J.ctx_of(q)
    data = R.load_data(ctx, J.SEED, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    LI.require_gate(approval, prep, "bca")
    rows = prep.rows
    log = data.train_log
    rand = np.asarray(log.diagnostic("infos/random"))[rows].astype(bool)
    hold = np.asarray(log.diagnostic("infos/hold"))[rows].astype(bool)
    hard = np.asarray(log.diagnostic("infos/hard_water"))[rows].astype(bool)
    a = np.asarray(prep.actions, np.float64)
    learners = J.learners_of(ctx, spec)
    # full BCA state for the dose
    args = LR.make_args(LI.settings(), J.SEED)
    config = LR.bca_config()
    _, _, mods = LR.bca_initial_state(args, config)
    d = XA.capped_dir(J.EXPLORE_OUT, q, "unweighted", J.SEED)
    rec = XE.explore_record(d / "record.json", spec)
    snap = rec["result"]["snapshots"][-1]
    st = LR.load_tree(d / "snapshot_100.msgpack", LI.bca_template(LI.settings(), J.SEED), snap["file_sha256"])
    tr = prep.train
    ro = P.bc_readout(mods, st, tr, config.blend)
    dose = np.asarray(ro.dose, np.float64)
    width = np.asarray(ro.width, np.float64)
    unit = float(st.posterior.residual_scale)
    thr = float(st.posterior.threshold)
    eta = width / (thr * unit)
    # band's raw input at the row: |y - q| / unit, with the BCA arm's own critic and target (one noise draw)
    y = np.asarray(P.native_target(args, mods, st.native, tr, jax.random.PRNGKey(7), 1.0), np.float64)
    qv = np.asarray(mods[1].apply(st.native.critic.params, tr.obs, tr.action).min(axis=-1), np.float64)
    err = np.abs(y - qv) / unit
    obs = tr.obs
    pi = {}
    for name, (actor, critic, f) in learners.items():
        pi[name] = np.asarray(mods[0].apply(actor, obs), np.float64)
    ph, pb, pw = pi["host_gap"], pi["bca_unweighted"], pi["bca_exact_strong"]
    past = np.any(np.abs(a) > 0.8 + 1e-6, axis=1)
    t1_past = a[:, 0] < -0.8 - 1e-6
    t1_near = (a[:, 0] <= -0.75) & ~t1_past
    # where the actor's commands sit relative to the rows' conditional mean of recorded commands
    k_fit = float(np.sum(pb * ph) / np.sum(ph * ph))
    kw_fit = float(np.sum(pw * ph) / np.sum(ph * ph))
    cos = float(np.sum(pb * ph) / math.sqrt(np.sum(pb * pb) * np.sum(ph * ph)))
    body = dict(
        rows=int(len(rows)), random_share=float(rand.mean()), hold_rows=int(hold.sum()), hard_water_share=float(hard.mean()),
        threshold=thr, unit=unit,
        dose=dict(mean=float(dose.mean()), sd=float(dose.std()), pct=pct(dose), min=float(dose.min()), max=float(dose.max()),
                  share_uniform=float((dose.mean() - 1.0) ** 2 / ((dose.mean() - 1.0) ** 2 + dose.var())),
                  equivalent_alpha=2.5 / float(dose.mean())),
        eta=dict(mean=float(eta.mean()), pct=pct(eta)),
        dose_by=dict(random=grp(dose, rand), recorder=grp(dose, ~rand), past80_any=grp(dose, past), not_past80=grp(dose, ~past),
                     t1_past80=grp(dose, t1_past), t1_near80=grp(dose, t1_near), hard_water=grp(dose, hard), calm=grp(dose, ~hard),
                     recorder_hard=grp(dose, ~rand & hard), recorder_calm=grp(dose, ~rand & ~hard),
                     random_hard=grp(dose, rand & hard), random_calm=grp(dose, rand & ~hard)),
        corr=dict(dose_vs_err=float(np.corrcoef(dose, err)[0, 1]),
                  dose_vs_cmd_norm=float(np.corrcoef(dose, np.linalg.norm(a, axis=1))[0, 1]),
                  dose_vs_dist_host_to_record=float(np.corrcoef(dose, np.linalg.norm(ph - a, axis=1))[0, 1]),
                  err_vs_dist_host_to_record=float(np.corrcoef(err, np.linalg.norm(ph - a, axis=1))[0, 1])),
        err=dict(mean=float(err.mean()), random=float(err[rand].mean()) if rand.any() else None,
                 recorder=float(err[~rand].mean()) if (~rand).any() else None),
        commands=dict(norm_host=float(np.linalg.norm(ph, axis=1).mean()), norm_bca=float(np.linalg.norm(pb, axis=1).mean()),
                      norm_bca_iw=float(np.linalg.norm(pw, axis=1).mean()), norm_recorded=float(np.linalg.norm(a, axis=1).mean()),
                      shrink_bca_vs_host=k_fit, shrink_bca_iw_vs_host=kw_fit, cos_bca_host=cos,
                      t1_host_mean=float(ph[:, 0].mean()), t1_bca_mean=float(pb[:, 0].mean()),
                      t1_host_hard=float(ph[hard, 0].mean()) if hard.any() else None,
                      t1_bca_hard=float(pb[hard, 0].mean()) if hard.any() else None,
                      t1_bca_iw_hard=float(pw[hard, 0].mean()) if hard.any() else None,
                      host_past80_share=float(np.mean(np.any(np.abs(ph) > 0.8, axis=1))),
                      bca_past80_share=float(np.mean(np.any(np.abs(pb) > 0.8, axis=1))),
                      sq_dist_to_record_host=float(np.mean(np.sum((ph - a) ** 2, 1))),
                      sq_dist_to_record_bca=float(np.mean(np.sum((pb - a) ** 2, 1))),
                      sq_dist_to_record_bca_iw=float(np.mean(np.sum((pw - a) ** 2, 1))),
                      # distance to the conditional mean of the recorded command: estimated by the recorder's own command
                      # share (1 - eps) on non-random rows only, so use the host->bca move direction against -host
                      move_toward_zero=float(np.mean(np.sum((pb - ph) * (-ph), 1)) / max(1e-12, np.mean(np.sum(ph * ph, 1))))))
    out[q] = body
    print(q, json.dumps(dict(dose=body["dose"]["mean"], sd=body["dose"]["sd"], by=dict((k, v["dose_mean"]) for k, v in body["dose_by"].items()),
                              corr=body["corr"], cmd=body["commands"]), default=lambda x: round(x, 4)), f"{time.time()-t0:.0f}s", flush=True)

OUT.write_text(json.dumps(dict(kind="capped-pull-diagnosis-rows", label="EXPLORATORY: seed 0 of the cloud rerun; no training, no band fit", by_quality=out), indent=1))
print("written", OUT)
