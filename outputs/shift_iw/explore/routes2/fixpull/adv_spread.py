"""EXPLORATORY (seed 0 of the cloud rerun; no training, no band, no threshold): with plain TD3+BC's final critic and
actor only, how far apart are the critic's values of different commands at one state, compared with the size of
BCA's band (U = (U/unit) x unit from the diagnosis, read from diag_band_at_commands.json, not recomputed)?
Per quality, on 8,192 kept training rows: q(s, a_i) at the recorded command, q(s, pi(s)) at the actor's command, and
q at 32 uniform random commands. Split by the recorder's own flags (random command / recorder command, hard water)."""
import json, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import experiments.shift  # noqa
from experiments.shift import learners as LR
from experiments.shift_iw import data as D
from experiments.shift_iw import run as R
import jobs as J

HERE = Path(__file__).resolve().parent
band = json.load(open(HERE.parent / "diagnose" / "diag_band_at_commands.json"))["res"]
ctx0 = J.ctx_of("expert"); spec = R.spec_digest(ctx0)
rng = np.random.default_rng(12345)
out = {}
def pct(x): return [float(np.percentile(x, p)) for p in (5, 25, 50, 75, 95)]
for q in J.QUALITIES:
    ctx = J.ctx_of(q); data = R.load_data(ctx, J.SEED, spec); prep = D.prepare(data, "gap", ctx.cfg)
    rows = prep.rows; log = data.train_log
    rand = np.asarray(log.diagnostic("infos/random"))[rows].astype(bool)
    hard = np.asarray(log.diagnostic("infos/hard_water"))[rows].astype(bool)
    host = R.load_host_state(ctx, "gap", J.SEED, spec)
    n = len(prep.train.obs); idx = rng.choice(n, size=min(8192, n), replace=False)
    obs = jnp.asarray(prep.train.obs[idx]); a = jnp.asarray(prep.train.action[idx])
    Q = lambda o, c: np.asarray(LR.CRITIC.apply(host.critic.params, o, c).min(axis=-1), np.float64)
    pi = np.clip(np.asarray(LR.ACTOR.apply(host.actor.params, obs)), -1, 1)
    qa, qpi = Q(obs, a), Q(obs, jnp.asarray(pi, jnp.float32))
    K = 32; us = rng.uniform(-1, 1, size=(K, len(idx), 4)).astype(np.float32)
    qu = np.stack([Q(obs, jnp.asarray(u)) for u in us])
    Vu, Su = qu.mean(0), qu.std(0)
    U = band[q]["recorded"]["U_over_unit"] * band[q]["unit"]
    d = qa - qpi; au = qa - Vu
    r, h = rand[idx], hard[idx]
    body = dict(rows=int(len(idx)), U_mean=U, unit=band[q]["unit"], random_share=float(r.mean()),
                delta_pct=pct(d), delta_over_U_pct=pct(d / U),
                share_delta_below_minusU=float(np.mean(d < -U)),
                share_delta_below_minusU_random=float(np.mean(d[r] < -U)) if r.any() else None,
                share_delta_below_minusU_recorder=float(np.mean(d[~r] < -U)) if (~r).any() else None,
                share_delta_below_minus_quarterU=float(np.mean(d < -U / 4)),
                spread_over_commands_pct=pct(Su), spread_over_commands_over_U_median=float(np.median(Su) / U),
                adv_vs_uniform_pct=pct(au),
                delta_random_median=float(np.median(d[r])) if r.any() else None,
                delta_recorder_median=float(np.median(d[~r])) if (~r).any() else None)
    out[q] = body
    print(q, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in body.items() if not isinstance(v, list)}),
          "delta pct", [round(x, 2) for x in body["delta_pct"]], "spread pct", [round(x, 2) for x in body["spread_over_commands_pct"]], flush=True)
(HERE / "adv_spread.json").write_text(json.dumps(dict(label="EXPLORATORY seed 0; host critic only; no band", res=out), indent=1))
