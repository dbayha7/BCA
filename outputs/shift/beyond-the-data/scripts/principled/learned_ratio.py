"""EXPLORATORY (outside the gate; changes no pre-registered rule). NO BCA code: a generic conditional density of the
recorded commands given the state (mixture density network, fitted on the gapped training rows only), used as the
behaviour model of a weighted conformal band's likelihood ratio. Only the score-free abstention set is computed.

Target law at a state: the fitted recorded-command law, shifted from its own mean m(s) to the actor's command pi(s).
At the actor's own command: log w = log p(m(s) | s) - log p(pi(s) | s). Bank weights: log w_j = log p(a_j - Delta_j | s_j)
- log p(a_j | s_j), Delta_j = pi(s_j) - m(s_j). Abstain iff w > alpha / (1 - alpha) * sum_j w_j (plain weighted split
conformal, alpha 0.1). Compared with the exact-recorder ratio of readout.py on the same pools.

Usage: python learned_ratio.py <seed>
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/principled")
from readout import (Ratio, auroc, boundary, residual, Seed, actor_cmd, true_q, visited_states, jsonable, log,  # noqa
                     ALPHA, HARM_TOL)
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.recorder as RC  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402

OUT = Path("/home/user/bca/runs/shift_explore/principled")
LABEL = ("EXPLORATORY: generic mixture-density behaviour model fitted on gapped training rows; stand-in weights for a "
         "weighted conformal band; score-free abstention only; no BCA code was run")
KMIX, HID, STEPS, BATCH, LR = 8, 128, 15000, 256, 1e-3
LOG_SD_MIN = float(np.log(float(sys.argv[2]))) if len(sys.argv) > 2 else -7.0  # floor on each component's spread
TAG = f"_floor{sys.argv[2]}" if len(sys.argv) > 2 else ""
CREDIBLE_SHIFT = float(np.log(np.log(20.0)))  # WBCP's test atom (Exp(1) mass, 95% credibility) abstains about ln(20)x
# earlier than plain weighted split conformal when the bank sum is stable: P(E > c) = exp(-c) > 0.05 iff c < ln 20
N = 3000


class MDN(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(HID)(x))
        h = nn.relu(nn.Dense(HID)(h))
        logits = nn.Dense(KMIX)(h)
        mu = nn.Dense(KMIX * 4)(h).reshape(x.shape[:-1] + (KMIX, 4))
        ls = jnp.clip(nn.Dense(KMIX * 4)(h).reshape(x.shape[:-1] + (KMIX, 4)), LOG_SD_MIN, 1.0)
        return logits, mu, ls


def logp(params, x, a):
    logits, mu, ls = MDN().apply(params, x)
    z = (a[..., None, :] - mu) / jnp.exp(ls)
    comp = jnp.sum(-0.5 * z * z - ls - 0.5 * jnp.log(2 * jnp.pi), -1)
    return jax.scipy.special.logsumexp(jax.nn.log_softmax(logits) + comp, -1)


def mean(params, x):
    logits, mu, _ = MDN().apply(params, x)
    return jnp.sum(jax.nn.softmax(logits)[..., None] * mu, -2)


def fit(obs, act, seed):
    key = jax.random.PRNGKey(seed)
    params = MDN().init(key, obs[:2])
    opt = optax.adam(LR)
    st = opt.init(params)
    obs, act = jnp.asarray(obs), jnp.asarray(act)

    @jax.jit
    def step(params, st, idx):
        loss, g = jax.value_and_grad(lambda p: -logp(p, obs[idx], act[idx]).mean())(params)
        up, st = opt.update(g, st, params)
        return optax.apply_updates(params, up), st, loss

    rng = np.random.default_rng([20261013, seed, 61])
    losses = []
    for i in range(STEPS):
        params, st, loss = step(params, st, jnp.asarray(rng.integers(len(obs), size=BATCH)))
        if (i + 1) % 1000 == 0:
            losses.append(float(loss))
    return params, losses


def main(rep):
    t0 = time.perf_counter()
    rng = np.random.default_rng([20261013, int(rep), 62])
    S = Seed(rep, worlds=("stall",))
    pg, ph, sd = S.pg, S.physics, S.st
    sa = RC.stall_aware(ph)
    exact = Ratio(S.recorder)
    critics = {"gapped": S.h2_critic("gapped")}
    acts, _ = S.h2_actors()
    obs_tr = np.asarray(pg.train.obs, np.float32)
    params, losses = fit(obs_tr, np.asarray(pg.actions, np.float32), rep)
    log(f"seed {rep}: mixture density fitted, final train NLL {losses[-1]:.3f} ({time.perf_counter() - t0:.0f}s)")
    lp = jax.jit(lambda x, a: logp(params, x, a))
    mfn = jax.jit(lambda x: mean(params, x))

    def P(x, a):
        return np.asarray(lp(jnp.asarray(pg.normalize(x)), jnp.asarray(np.asarray(a, np.float32))), np.float64)

    def M(x):
        return np.asarray(mfn(jnp.asarray(pg.normalize(x))), np.float64)

    lg = sd.log
    xb = np.asarray(lg["infos/state"][pg.bank_rows], np.float64)
    ab = np.asarray(lg["actions"][pg.bank_rows], np.float64)
    res = dict(seed=int(rep), alpha=ALPHA, model=dict(kind="mixture density network", components=KMIX, hidden=HID,
                                                       log_sd_min=LOG_SD_MIN,
                                                       steps=STEPS, batch=BATCH, lr=LR, train_nll_every_1000=losses),
               bank_nll=float(-P(xb, ab).mean()), pools={}, boundaries={})
    bnd, bnd_exact = {}, {}
    for k in ("gapped", "full", "clone"):
        cb = actor_cmd(acts[k], pg, xb)
        mb = M(xb)
        lw = P(xb, ab - (cb - mb)) - P(xb, ab)
        bnd[k] = boundary(lw)
        bnd_exact[k] = boundary(exact.log_w(xb, ab, cb))
    res["boundaries"] = dict(learned=bnd, exact=bnd_exact)

    def score(name, x, k):
        a = actor_cmd(acts[k], pg, x)
        ell = P(x, M(x)) - P(x, a)
        abst = ell > bnd[k]["log_threshold"]
        ell_x = exact.ell_own(x, a)
        abst_x = ell_x > bnd_exact[k]["log_threshold"]
        past = TC.past_knee(a)
        d_g, dis, q_g = residual(critics["gapped"], pg, ph, sa, x, a)
        rec = dict(n=int(len(x)), past_80_share=float(past.mean()), abstain_learned=float(abst.mean()),
                   abstain_exact=float(abst_x.mean()),
                   catch_past80_learned=float(abst[past].mean()) if past.any() else None,
                   flag_not_past80_learned=float(abst[~past].mean()) if (~past).any() else None,
                   agreement_learned_exact=float((abst == abst_x).mean()),
                   auroc_past80_by_learned_ell=auroc(ell, past))
        thr = bnd[k]["log_threshold"]
        abst_c = ell > thr - CREDIBLE_SHIFT
        rec.update(abstain_learned_credible=float(abst_c.mean()),
                   catch_past80_learned_credible=float(abst_c[past].mean()) if past.any() else None,
                   flag_not_past80_learned_credible=float(abst_c[~past].mean()) if (~past).any() else None)
        q = lambda v: [float(t) for t in np.quantile(v, (0.1, 0.25, 0.5, 0.75, 0.9))] if len(v) else None
        rec["ell_minus_boundary_quantiles_past80"] = q(ell[past] - thr)
        rec["ell_minus_boundary_quantiles_not_past80"] = q(ell[~past] - thr)
        if k in ("gapped", "full"):
            ac = np.clip(a, -TC.KNEE, TC.KNEE)
            harm = (true_q(sa, x, ac, ph) - true_q(sa, x, a, ph)) > HARM_TOL
            rec.update(harmful_share=float(harm.mean()),
                       catch_harmful_learned=float(abst[harm].mean()) if harm.any() else None,
                       flag_harmless_learned=float(abst[~harm].mean()) if (~harm).any() else None,
                       catch_harmful_exact=float(abst_x[harm].mean()) if harm.any() else None,
                       flag_harmless_exact=float(abst_x[~harm].mean()) if (~harm).any() else None,
                       auroc_harm_by_learned_ell=auroc(ell, harm), auroc_harm_by_exact_ell=auroc(ell_x, harm),
                       catch_harmful_learned_credible=float(abst_c[harm].mean()) if harm.any() else None,
                       flag_harmless_learned_credible=float(abst_c[~harm].mean()) if (~harm).any() else None)
        res["pools"][name] = rec
        log(f"  {name}: past80 {rec['past_80_share']:.3f} abstain learned {rec['abstain_learned']:.3f} exact "
            f"{rec['abstain_exact']:.3f}; catch past80 {rec['catch_past80_learned']} flag other "
            f"{rec['flag_not_past80_learned']}; harm {rec.get('harmful_share')} catch {rec.get('catch_harmful_learned')} "
            f"flag harmless {rec.get('flag_harmless_learned')} auroc {rec.get('auroc_harm_by_learned_ell')}; credible: catch "
            f"{rec.get('catch_harmful_learned_credible')} flag {rec.get('flag_harmless_learned_credible')}; "
            f"ell-thr past80 q {rec['ell_minus_boundary_quantiles_past80']}")

    tr = np.sort(rng.choice(pg.n, N, replace=False))
    for k in ("gapped", "full", "clone"):
        score(f"train|{k}", pg.states[tr], k)
    cal = lg.split_rows("calibration")
    cal = np.sort(rng.choice(cal[~sd.gap.contains(lg["actions"][cal])], N, replace=False))
    xh, ah = np.asarray(lg["infos/state"][cal], np.float64), np.asarray(lg["actions"][cal], np.float64)
    for k in ("gapped", "full"):
        lw = P(xh, ah - (actor_cmd(acts[k], pg, xh) - M(xh))) - P(xh, ah)
        res[f"heldout_recorded_abstain_{k}"] = float((lw > bnd[k]["log_threshold"]).mean())
        res[f"heldout_recorded_abstain_credible_{k}"] = float((lw > bnd[k]["log_threshold"] - CREDIBLE_SHIFT).mean())
    for s in ("recorded", "strong_current", "big_waves", "big_kicks"):
        for k in ("gapped", "full", "clone"):
            x, _ = visited_states(L.mlp_policy(acts[k], pg, k), s, ph, 200, take=N, rng=rng)
            score(f"{s}|{k}", x, k)
    res["seconds"] = time.perf_counter() - t0
    p = OUT / f"learned_ratio_seed{rep}{TAG}.json"
    p.write_text(json.dumps(jsonable(dict(label=LABEL, **res)), indent=1, sort_keys=True))
    log(f"wrote {p}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
