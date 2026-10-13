"""EXPLORATORY (outside the gate; changes no pre-registered rule). NO BCA code. The likelihood ratio of a weighted
conformal band estimated the standard way (probabilistic classification, as in Tibshirani et al. 2019), with no density
model: the target law is 'the recorder's own scatter, recentred on the actor's command', sampled from the data:

    target sample at training row i:   (s_i, clip(pi(s_i) + a_i - clone(s_i), -1, 1))
    recorded sample at training row i: (s_i, a_i)

A classifier separates the two; w(s, a) = c / (1 - c). Score-free abstention as in readout.py: abstain at x iff
w(x) > alpha / (1 - alpha) * sum_j w_j over the calibration bank. Compared with the exact recorder ratio.

Usage: python classifier_ratio.py <seed>
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
LABEL = ("EXPLORATORY: classifier likelihood ratio (actor-recentred recorded scatter against recorded commands), fitted "
         "on gapped training rows; stand-in weights for a weighted conformal band; score-free abstention only; no BCA "
         "code was run")
HID, STEPS, BATCH, LR = 128, 10000, 512, 1e-3
N = 3000
CREDIBLE_SHIFT = float(np.log(np.log(20.0)))


class Clf(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(HID)(x))
        h = nn.relu(nn.Dense(HID)(h))
        return nn.Dense(1)(h)[..., 0]


def fit(xs, a_rec, a_tgt, seed):
    X = jnp.asarray(np.concatenate([np.concatenate([xs, a_rec], 1), np.concatenate([xs, a_tgt], 1)]), jnp.float32)
    y = jnp.asarray(np.concatenate([np.zeros(len(xs)), np.ones(len(xs))]), jnp.float32)
    params = Clf().init(jax.random.PRNGKey(seed), X[:2])
    opt = optax.adam(LR)
    st = opt.init(params)

    @jax.jit
    def step(params, st, idx):
        def lf(p):
            z = Clf().apply(p, X[idx])
            return optax.sigmoid_binary_cross_entropy(z, y[idx]).mean()
        loss, g = jax.value_and_grad(lf)(params)
        up, st = opt.update(g, st, params)
        return optax.apply_updates(params, up), st, loss

    rng = np.random.default_rng([20261013, seed, 71])
    losses = []
    for i in range(STEPS):
        params, st, loss = step(params, st, jnp.asarray(rng.integers(len(X), size=BATCH)))
        if (i + 1) % 1000 == 0:
            losses.append(float(loss))
    return params, losses


def main(rep):
    t0 = time.perf_counter()
    rng = np.random.default_rng([20261013, int(rep), 72])
    S = Seed(rep, worlds=("stall",))
    pg, ph, sd = S.pg, S.physics, S.st
    sa = RC.stall_aware(ph)
    exact = Ratio(S.recorder)
    critics = {"gapped": S.h2_critic("gapped")}
    acts, _ = S.h2_actors()
    obs_tr = np.asarray(pg.train.obs, np.float32)
    a_tr = np.asarray(pg.actions, np.float32)
    clone_tr = actor_cmd(acts["clone"], pg, pg.states).astype(np.float32)
    lg = sd.log
    xb = np.asarray(lg["infos/state"][pg.bank_rows], np.float64)
    ab = np.asarray(lg["actions"][pg.bank_rows], np.float64)
    res = dict(seed=int(rep), alpha=ALPHA, model=dict(kind="classifier", hidden=HID, steps=STEPS, batch=BATCH, lr=LR),
               pools={}, boundaries={}, losses={})
    tr_pick = np.sort(rng.choice(pg.n, N, replace=False))
    cal = lg.split_rows("calibration")
    cal = np.sort(rng.choice(cal[~sd.gap.contains(lg["actions"][cal])], N, replace=False))
    xh, ah = np.asarray(lg["infos/state"][cal], np.float64), np.asarray(lg["actions"][cal], np.float64)
    visited = {}
    for s in ("recorded", "strong_current", "big_waves", "big_kicks"):
        for k in ("gapped", "full", "clone"):
            visited[(s, k)] = visited_states(L.mlp_policy(acts[k], pg, k), s, ph, 200, take=N, rng=rng)[0]

    for k in ("gapped", "full", "clone"):
        pi_tr = actor_cmd(acts[k], pg, pg.states).astype(np.float32)
        tgt = np.clip(pi_tr + a_tr - clone_tr, -1.0, 1.0)
        params, losses = fit(obs_tr, a_tr, tgt, rep * 10 + len(k))
        res["losses"][k] = losses
        f = jax.jit(lambda X: Clf().apply(params, X))

        def logw(x, a):
            X = np.concatenate([pg.normalize(x), np.asarray(a, np.float32)], 1)
            return np.asarray(f(jnp.asarray(X, jnp.float32)), np.float64)  # logit = log c/(1-c), equal class sizes

        bnd = boundary(logw(xb, ab))
        bx = boundary(exact.log_w(xb, ab, actor_cmd(acts[k], pg, xb)))
        res["boundaries"][k] = dict(classifier=bnd, exact=bx)
        thr = bnd["log_threshold"]
        res[f"heldout_recorded_abstain_{k}"] = float((logw(xh, ah) > thr).mean())
        log(f"seed {rep} actor {k}: classifier fitted (final loss {losses[-1]:.4f}); boundary {thr:.2f}, ESS "
            f"{bnd['kish_ess']:.0f}; held-out recorded flagged {res[f'heldout_recorded_abstain_{k}']:.4f} "
            f"({time.perf_counter() - t0:.0f}s)")
        pools = [("train", pg.states[tr_pick])] + [(s, visited[(s, k)]) for s in
                                                    ("recorded", "strong_current", "big_waves", "big_kicks")]
        for name, x in pools:
            a = actor_cmd(acts[k], pg, x)
            ell = logw(x, a)
            abst, abst_c = ell > thr, ell > thr - CREDIBLE_SHIFT
            ell_x = exact.ell_own(x, a)
            abst_x = ell_x > bx["log_threshold"]
            past = TC.past_knee(a)
            hole = sd.gap.contains(np.asarray(a, np.float32))
            rec = dict(n=int(len(x)), past_80_share=float(past.mean()), abstain=float(abst.mean()),
                       abstain_exact=float(abst_x.mean()),
                       catch_past80=float(abst[past].mean()) if past.any() else None,
                       flag_not_past80=float(abst[~past].mean()) if (~past).any() else None,
                       flag_in_ball_not_past80=float(abst[hole & ~past].mean()) if (hole & ~past).any() else None,
                       catch_past80_credible=float(abst_c[past].mean()) if past.any() else None,
                       flag_not_past80_credible=float(abst_c[~past].mean()) if (~past).any() else None,
                       auroc_past80=auroc(ell, past))
            if k in ("gapped", "full"):
                ac = np.clip(a, -TC.KNEE, TC.KNEE)
                harm = (true_q(sa, x, ac, ph) - true_q(sa, x, a, ph)) > HARM_TOL
                rec.update(harmful_share=float(harm.mean()),
                           catch_harmful=float(abst[harm].mean()) if harm.any() else None,
                           flag_harmless=float(abst[~harm].mean()) if (~harm).any() else None,
                           catch_harmful_credible=float(abst_c[harm].mean()) if harm.any() else None,
                           flag_harmless_credible=float(abst_c[~harm].mean()) if (~harm).any() else None,
                           catch_harmful_exact=float(abst_x[harm].mean()) if harm.any() else None,
                           flag_harmless_exact=float(abst_x[~harm].mean()) if (~harm).any() else None,
                           auroc_harm=auroc(ell, harm), auroc_harm_exact=auroc(ell_x, harm))
            res["pools"][f"{name}|{k}"] = rec
            log(f"  {name}|{k}: past80 {rec['past_80_share']:.3f} abstain {rec['abstain']:.3f} (exact "
                f"{rec['abstain_exact']:.3f}); catch past80 {rec['catch_past80']} flag other {rec['flag_not_past80']}; "
                f"harm {rec.get('harmful_share')} catch {rec.get('catch_harmful')} flag harmless "
                f"{rec.get('flag_harmless')} auroc {rec.get('auroc_harm')}")
    res["seconds"] = time.perf_counter() - t0
    p = OUT / f"classifier_ratio_seed{rep}.json"
    p.write_text(json.dumps(jsonable(dict(label=LABEL, **res)), indent=1, sort_keys=True))
    log(f"wrote {p}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
