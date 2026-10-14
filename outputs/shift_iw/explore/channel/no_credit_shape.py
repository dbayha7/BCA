"""POST HOC (added after command_band_check.py's pass marks were scored; not a pass mark): check (e) on the
command-weighted lower edge with a different refusal rule. A command the band cannot certify gets no value credit
over the row's recorded command, W = [Q1(s, a) - Q1(s, a_i)]_+ (so Q1 - W = min(Q1(s, a), Q1(s, a_i))), instead of
version 5's rule (a) (the table's largest finite threshold). Certified commands keep [U(s, a) - U(s, a_i)]_+.
Same rows, grid, actor and rule as check (e). Reads only; writes runs/shift_iw_v5_explore/channel/command_band/
no_credit_<quality>.json.

Usage (from /root/bca-shift, PYTHONPATH=/root/bca-shift, JAX_PLATFORMS=cpu): python no_credit_shape.py <quality> ...
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.shift import learners as LR
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import run as RN

_spec = importlib.util.spec_from_file_location("cbc", Path(__file__).with_name("command_band_check.py"))
CB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CB)
LABEL = "POST HOC: not a pass mark; the refusal rule 'no credit over the recorded command' in the lower edge"


def one(ctx, spec, seed, rule):
    t0 = time.perf_counter()
    data = RN.load_data(ctx, seed, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    host = RN.load_host_state(ctx, "gap", seed, spec)
    brec = RN.require_band(ctx, "gap", seed, spec)
    band = LI.load_band(RN.band_dir(ctx, "gap", seed) / "band.msgpack", brec["band_sha256"], host, prep, ctx.cfg,
                        seed, RN.approval_for(ctx, CB.STAGE), brec["band"], stage=CB.STAGE)
    pi = RN.controllers_of(ctx, "room_lower_edge", seed, spec)["le_none"]
    log, gap = data.train_log, data.gap
    rng = lambda part: np.random.default_rng([CB.W.MASTER_SEED, int(seed), CB.TAG, part])
    consts = CB.ball_constants(gap, rng(0))
    bank = CB.Rows(log, prep.bank_rows, gap, consts, rng(1))
    train = CB.Rows(log, prep.rows, gap, consts, rng(2))
    a_bank = np.asarray(log["actions"][prep.bank_rows], np.float32).astype(np.float64)
    pi_bank = np.asarray(LR.actor_rows(pi, np.asarray(prep.bank.obs, np.float32)), np.float64)
    lw_bank = bank.log_ratio(np.arange(len(a_bank)), a_bank, pi_bank)
    hold = np.asarray(log.diagnostic("infos/hold")[prep.bank_rows], bool)
    w_bank = np.where(hold | ~np.isfinite(lw_bank), 0.0, np.exp(np.clip(lw_bank, -700, 700)))
    scores, stream = LI.bank_scores(band, prep.bank, ctx.cfg)
    th = CB.Thresholds(scores, stream, w_bank)

    obs = np.asarray(prep.train.obs, np.float32)
    a_pi = np.asarray(LR.actor_rows(pi, obs), np.float32)
    idx = np.arange(prep.n)
    a_rec = np.asarray(prep.actions, np.float64)
    R_rec, _ = th.read(np.where(gap.contains(a_rec), np.inf, train.log_ratio(idx, a_rec, a_pi.astype(np.float64))))
    u_rec = LE.logged_widths(band, prep, LE.as_float32_thresholds(R_rec), stage=CB.STAGE)
    q_rec = np.asarray(LR.critic_rows(host.critic.params, obs, prep.actions.astype(np.float32))[0][:, 0], np.float64)
    shared = LE.band_shared(band, CB.STAGE)
    critic = LI.models()[1]
    band_v = jax.jit(lambda sh, o, a, R, u: jnp.maximum(LE.band_width(sh["cal_params"], sh["unit"], o, a, R) - u, 0.0))
    band_g = jax.jit(lambda sh, o, a, R, u: jax.vmap(jax.grad(
        lambda aa, oo, rr, uu: jnp.maximum(LE.band_width(sh["cal_params"], sh["unit"], oo[None], aa[None], rr) - uu,
                                           0.0)[0]))(a, o, R, u))
    q_v = jax.jit(lambda p, o, a, qr: jnp.maximum(critic.apply(p, o, a)[:, 0] - qr, 0.0))
    q_g = jax.jit(lambda p, o, a, qr: jax.vmap(jax.grad(
        lambda aa, oo, qq: jnp.maximum(critic.apply(p, oo[None], aa[None])[0, 0] - qq, 0.0)))(a, o, qr))

    def read(r, a):
        a64 = np.asarray(a, np.float64)
        lm = np.where(gap.contains(a64), np.inf, train.log_ratio(r, a64, a_pi[r].astype(np.float64)))
        R, refused = th.read(lm)
        return R.astype(np.float32), refused

    def values(r, a):
        R, refused = read(r, a)
        wb = np.asarray(band_v(shared, jnp.asarray(obs[r]), jnp.asarray(a), jnp.asarray(R), jnp.asarray(u_rec[r])))
        wq = np.asarray(q_v(host.critic.params, jnp.asarray(obs[r]), jnp.asarray(a), jnp.asarray(q_rec[r], jnp.float32)))
        return np.where(refused, wq, wb).astype(np.float64)

    def slope(r, a):
        R, refused = read(r, a)
        gb = np.asarray(band_g(shared, jnp.asarray(obs[r]), jnp.asarray(a), jnp.asarray(R), jnp.asarray(u_rec[r])))
        gq = np.asarray(q_g(host.critic.params, jnp.asarray(obs[r]), jnp.asarray(a),
                            jnp.asarray(q_rec[r], jnp.float32)))
        return np.where(refused[:, None], gq, gb).astype(np.float64)

    rows_e = LE.shape_rows(prep, seed, rule["rows"])
    rec = LE.penalty_shape(values, slope, host.critic.params, prep, pi, rows_e, rule)
    rec.update(seed=int(seed), seconds=time.perf_counter() - t0)
    return rec


def main(argv):
    for quality in argv:
        ctx = RN.Context(quality=quality)
        spec = RN.spec_digest(ctx)
        rule = RN.spec_lower_edge(RN.frozen_spec(ctx, spec))["rule"].check_e
        out = []
        for seed in RN.SEEDS:
            r = one(ctx, spec, seed, rule)
            ctx.log(f"{quality} seed {seed}: check (e) {'pass' if r['passed'] else 'FAIL ' + str(r['failed'])}; best "
                    f"past 80% {r['best_past80_share']:.3f} vs Q1 alone {r['q1_alone_best_past80_share']:.3f}")
            out.append(r)
        (CB.OUT / f"no_credit_{quality}.json").write_text(json.dumps(dict(label=LABEL, quality=quality, seeds=out),
                                                                     indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
