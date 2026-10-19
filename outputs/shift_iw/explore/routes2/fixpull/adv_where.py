"""EXPLORATORY (seed 0; host critic and actor only; no band): where are the rows whose recorded command the critic
rates below the actor's by more than BCA's mean band width U (U from diag_band_at_commands.json)? Hard water, actor's
thruster 1 past 80%, random rows. Also the recorded per-thruster command range (0.5-99.5 percentiles)."""
import json
from pathlib import Path
import numpy as np, jax.numpy as jnp
import experiments.shift  # noqa
from experiments.shift import learners as LR
from experiments.shift_iw import data as D
from experiments.shift_iw import run as R
import jobs as J
HERE = Path(__file__).resolve().parent
band = json.load(open(HERE.parent / "diagnose" / "diag_band_at_commands.json"))["res"]
ctx0 = J.ctx_of("expert"); spec = R.spec_digest(ctx0)
rng = np.random.default_rng(12345); out = {}
for q in J.QUALITIES:
    ctx = J.ctx_of(q); data = R.load_data(ctx, J.SEED, spec); prep = D.prepare(data, "gap", ctx.cfg)
    rows = prep.rows; log = data.train_log
    rand = np.asarray(log.diagnostic("infos/random"))[rows].astype(bool)
    hard = np.asarray(log.diagnostic("infos/hard_water"))[rows].astype(bool)
    host = R.load_host_state(ctx, "gap", J.SEED, spec)
    A = np.asarray(prep.train.action, np.float64)
    rngp = [np.percentile(A[:, j], [0.5, 99.5]).round(3).tolist() for j in range(4)]
    n = len(prep.train.obs); idx = rng.choice(n, size=min(8192, n), replace=False)
    obs = jnp.asarray(prep.train.obs[idx]); a = jnp.asarray(prep.train.action[idx])
    Q = lambda o, c: np.asarray(LR.CRITIC.apply(host.critic.params, o, c).min(axis=-1), np.float64)
    pi = np.clip(np.asarray(LR.ACTOR.apply(host.actor.params, obs)), -1, 1)
    d = Q(obs, a) - Q(obs, jnp.asarray(pi, jnp.float32))
    U = band[q]["recorded"]["U_over_unit"] * band[q]["unit"]
    low = d < -U; r, h = rand[idx], hard[idx]; p80 = np.abs(pi[:, 0]) > 0.8
    body = dict(share_low=float(low.mean()), low_hard=float(h[low].mean()), all_hard=float(h.mean()),
                low_actor_past80=float(p80[low].mean()), all_actor_past80=float(p80.mean()),
                low_random=float(r[low].mean()) if low.any() else None,
                recorder_low_hard=float(h[low & ~r].mean()) if (low & ~r).any() else None,
                recorder_low_actor_past80=float(p80[low & ~r].mean()) if (low & ~r).any() else None,
                recorded_range_pct=rngp,
                # shift in the mean pulled command if rows with d < -U were dropped (thruster 1, hard water)
                t1_mean_all_hard=float(A[idx][h, 0].mean()), t1_mean_kept_hard=float(A[idx][h & ~low, 0].mean()),
                t1_actor_hard=float(pi[h, 0].mean()))
    out[q] = body; print(q, {k: v for k, v in body.items()}, flush=True)
(HERE / "adv_where.json").write_text(json.dumps(dict(label="EXPLORATORY seed 0; host critic only; no band", res=out), indent=1))
