"""READ-ONLY. Deployment consumers of the SAVED weighted band, by exact rollouts of SAVED actors.

Same starts (first 200 of each situation's bank), same arithmetic and same clip fallback (recorded range: 0.5% and
99.5% quantiles of the gapped training commands) as stage2/code/deploy_check.run_rule. Rules per step, with
a = clip(pi(s), -1, 1) the actor's proposal, a_clip = clip(a, lo, hi), s~ = s clamped to the training-state box
(per-dimension 0.5% / 99.5% quantiles of the gapped training states):

  never                    a
  tau_raw                  Stage 2's check as run: U~(s, a) > tau -> a_clip            (plumbing: must reproduce)
  refuse_clamped           BCA cannot certify (log w(s~, a) >= l*) -> a_clip
  project_raw              certified projection, band read at s: if log w(s, a) >= l*, the furthest point of the
                           segment a_clip -> a with log w < l* (12 bisection steps); a_clip if a_clip is refused too
  project_clamped          the same, band read at s~                                      (PRIMARY consumer)
  project_clamped_plain    project_clamped with l* replaced by the closed-form refusal limit of PLAIN weighted
                           split conformal on the same threshold half (log(a/(1-a)) + log sum w): BCA's Bayesian
                           threshold removed, same classifier                               (attribution)
  random:project_clamped   a_clip with probability = project_clamped's firing share (the M5 control)
  clip_always              a_clip at every step (the recorded-range hinge applied at deployment)
  cap80                    clip(a, -0.8, 0.8) (knows the gap; reference)
  lower_edge_select        argmax over {a, project_clamped(a), a_clip, clone c(s)} of min_h Q(s, .) - U~(s~, .)
                           (pessimistic selection by BCA's lower edge, k = 1; critic = the frozen gapped critic)

Actors: the Stage 2 policy test's saved 'none' (no caution) and 'bca_k1' in each world. Band: the world's saved
weighted fixed-critic band (the one the deployment check used). Usage: python deploy.py <seed> [n_starts]
"""
import json
import math
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import common as C

TC, TR, L = C.TC, C.TR, C.L
N_STARTS = 200
SITS = ("strong_current", "recorded", "big_waves", "big_kicks")
RULES = ("never", "tau_raw", "refuse_clamped", "project_raw", "project_clamped", "project_clamped_plain",
         "random:project_clamped", "clip_always", "cap80", "lower_edge_select")


def make_projector(band):
    cls_p = band.b["cls_params"]

    def lw(o, a):
        return C.CLS.apply(cls_p, o, a)

    def proj(o, a, aref, lstar):
        ok_a = lw(o, a) < lstar
        ok_ref = lw(o, aref) < lstar

        def body(_, c):
            lo_t, hi_t = c
            mid = 0.5 * (lo_t + hi_t)
            am = aref + mid[:, None] * (a - aref)
            ok = lw(o, am) < lstar
            return jnp.where(ok, mid, lo_t), jnp.where(ok, hi_t, mid)

        lo_t, _ = jax.lax.fori_loop(0, 12, body, (jnp.zeros(a.shape[0]), jnp.ones(a.shape[0])))
        ap = aref + lo_t[:, None] * (a - aref)
        out = jnp.where(ok_a[:, None], a, jnp.where(ok_ref[:, None], ap, aref))
        return out, ~ok_a
    return jax.jit(proj)


def run_rule(rule, actor, clone_pol, band, proj, crit, physics, anchor, x0, lo, hi, blo, bhi, l_plain, rate=None,
             coin=None):
    x = np.asarray(x0, np.float64).copy()
    n, T = len(x), physics.length
    J = np.zeros(n)
    fired = np.zeros((n, T), bool)
    past_prop = np.zeros((n, T), bool)
    past_app = np.zeros((n, T), bool)
    ge95_app = np.zeros((n, T), bool)
    move = np.zeros((n, T))
    rng = np.random.default_rng(coin) if coin is not None else None
    for t in range(T):
        a_act = np.clip(actor(x), -1.0, 1.0)
        a_clip = np.clip(a_act, lo, hi)
        a = a_act.copy()
        if rule == "never":
            fire = np.zeros(n, bool)
        elif rule == "tau_raw":
            fire = ~(band.width(band.norm(x), a_act) <= band.tau)
            a[fire] = a_clip[fire]
        elif rule == "refuse_clamped":
            fire = band.logw(band.norm(np.clip(x, blo, bhi)), a_act) >= band.l_star
            a[fire] = a_clip[fire]
        elif rule in ("project_raw", "project_clamped", "project_clamped_plain"):
            xs = x if rule == "project_raw" else np.clip(x, blo, bhi)
            ls = l_plain if rule == "project_clamped_plain" else band.l_star
            ap, f = proj(jnp.asarray(band.norm(xs)), jnp.asarray(a_act, jnp.float32),
                         jnp.asarray(a_clip, jnp.float32), jnp.float32(ls))
            fire = np.asarray(f)
            a = np.where(fire[:, None], np.asarray(ap, np.float64), a_act)
        elif rule.startswith("random:"):
            fire = rng.uniform(size=n) < rate
            a[fire] = a_clip[fire]
        elif rule == "clip_always":
            fire = np.any(a_clip != a_act, 1)
            a = a_clip
        elif rule == "cap80":
            a = np.clip(a_act, -0.8, 0.8)
            fire = np.any(a != a_act, 1)
        elif rule == "lower_edge_select":
            xs = np.clip(x, blo, bhi)
            on = band.norm(xs)
            ap, _ = proj(jnp.asarray(on), jnp.asarray(a_act, jnp.float32), jnp.asarray(a_clip, jnp.float32),
                         jnp.float32(band.l_star))
            cands = [a_act, np.asarray(ap, np.float64), a_clip, np.clip(clone_pol(x), -1.0, 1.0)]
            o_raw = band.norm(x)
            le = np.stack([C.qmin(crit, o_raw, c) - band.width(on, c) for c in cands], 1)
            pick = le.argmax(1)
            a = np.stack(cands, 1)[np.arange(n), pick]
            fire = pick != 0
        else:
            raise ValueError(rule)
        fired[:, t] = fire
        past_prop[:, t] = TC.past_knee(a_act, physics)
        past_app[:, t] = TC.past_knee(a, physics)
        ge95_app[:, t] = np.any(np.abs(a) >= 0.95, 1)
        move[:, t] = np.abs(a - a_act).max(1)
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    pts = TR.points(J, anchor)
    fp = fired & past_prop
    return dict(points=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)), fire_share=float(fired.mean()),
                past80_proposed=float(past_prop.mean()), past80_applied=float(past_app.mean()),
                ge95_applied=float(ge95_app.mean()), mean_move_when_fired=float(move[fired].mean()) if fired.any()
                else 0.0, recall_past80=float(fp.sum() / max(past_prop.sum(), 1)),
                precision_past80=float(fp.sum() / max(fired.sum(), 1)),
                flag_rate_not_past80=float((fired & ~past_prop).sum() / max((~past_prop).sum(), 1)),
                rate=rate, per_start_points=np.round(pts, 4).tolist())


def main(rep, n):
    t0 = time.perf_counter()
    out = dict(rep=rep, n_starts=n, worlds={})
    for wi, world in enumerate(("stall", "no_stall")):
        sd, pg = C.seed_data(world, rep)
        physics = pg.physics
        band = C.Band(world, rep)
        proj = make_projector(band)
        crit = C.critic(sd)
        lo, hi = C.recorded_range(pg)
        blo, bhi = C.state_box(pg)
        l_plain = C.plain_wcp_limit(band, pg, rep)
        acts, lock, vals = C.policy_actors(world, rep)
        clone_pol = L.mlp_policy(C.clone(world, rep), pg, "clone")
        an = C.anchors(world)
        saved = json.loads((C.R2 / "deploy" / world / f"seed{rep}" / "values.json").read_text())
        W = dict(l_star=band.l_star, l_plain=l_plain, tau=band.tau, recorded_range=[lo.tolist(), hi.tolist()],
                 actors={})
        for ai, name in enumerate(("none", "bca_k1")):
            pol = L.mlp_policy(acts[name], pg, name)
            R = {}
            for si, s in enumerate(SITS):
                x0 = TC.bank_starts(s, TC.BANK_SIZE)[:n]
                r = {}
                for rule in RULES:
                    kw = {}
                    if rule.startswith("random:"):
                        kw = dict(rate=r[rule.split(":", 1)[1]]["fire_share"],
                                  coin=[20261014, rep, wi, ai, si, 777])
                    r[rule] = run_rule(rule, pol, clone_pol, band, proj, crit, physics, an[s], x0, lo, hi, blo, bhi,
                                       l_plain, **kw)
                sv = saved["results"][name][s]
                r["plumbing"] = dict(saved_never=sv["none"]["points"], saved_check=sv["bca"]["points"],
                                     saved_check_fire=sv["bca"]["fire_share"],
                                     never_diff=r["never"]["points"] - sv["none"]["points"],
                                     check_diff=r["tau_raw"]["points"] - sv["bca"]["points"])
                r["project_minus_random"] = r["project_clamped"]["points"] - r["random:project_clamped"]["points"]
                R[s] = r
                print(f"seed {rep} {world:8s} {name:7s} {s:14s} " + " ".join(
                    f"{k.replace('project', 'P')}={v['points']:.1f}/{v['fire_share']:.2f}" for k, v in r.items()
                    if isinstance(v, dict) and "points" in v) + f" | plumbing never {r['plumbing']['never_diff']:+.2e}"
                    f" check {r['plumbing']['check_diff']:+.2e} ({time.perf_counter() - t0:.0f}s)", flush=True)
            W["actors"][name] = R
        out["worlds"][world] = W
    out["seconds"] = time.perf_counter() - t0
    C.dump(out, f"deploy_seed{rep}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else N_STARTS)
