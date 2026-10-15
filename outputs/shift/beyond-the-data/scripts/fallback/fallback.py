"""EXPLORATORY, no training, no BCA: deployment-time switching rules for the saved ThrusterChain actors.

At every step of an exact 100-step rollout the actor proposes a command; a trigger decides whether to replace it by a
fallback command. Nothing is learned; every return is exact simulation. Gate outputs are read through the gate's own
verified loaders and nothing under experiments/shift or runs/shift is written.

Fallbacks:  clone    the cloned actor's command (the actor every H2 actor started from)
            clip99   the actor's command clipped, per thruster, to the range holding 99% of the recorded commands
                     (0.5% trimmed each side) of the gapped training rows
            clipmm   the same with the full recorded range (minimum to maximum); a few leftover full-throttle rows
                     keep thruster 1's range at -1
            project  the recorded command, among the 10 training rows nearest in state, closest to the actor's command
Triggers:   none / always
            state95, state99   10th-nearest-neighbour distance of the state to the training states (standard units)
                               above the 95th / 99th split-conformal quantile of held-out recorded rows
            sa95, sa99         the same for (state, actor's command) against the training (state, command) rows
            oracle             the fallback's command has a higher exact remaining-episode return than the actor's own
                               command, both followed by the actor (one exact policy-improvement step)
            random95, random99 a coin at the rate sa95 / sa99 fired for the same actor, situation, seed and fallback
  extras:   local95, local99   the actor's command lies outside the per-thruster range of the commands recorded at
                               the 10 training rows nearest in state by more than the 95th / 99th split-conformal
                               quantile of the same score on held-out recorded rows (a command-given-state check)
            rlocal95           a coin at the rate local95 fired
            unrec              the actor proposes a command past 80% on any thruster: a perfect detector of the
                               commands the gapped data lack (uses knowledge of the gap; a reference, not a method)
Held-out recorded rows: the calibration split (200 recorded episodes the learners never saw) with the same removal
rule applied (rows whose command is in the removed ball are dropped), so they are exchangeable with training rows.

Usage (from /home/user/bca): python fallback.py <seed> <world: stall|no_stall> <actor> [--n N] [--situations ...]
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import experiments.shift.acceptance as A
import experiments.shift.learners as L
import experiments.shift.thruster_chain as TC
import experiments.shift.truth as TR

OUT = Path("/home/user/bca/runs/shift_explore/fallback")
K = 10
LEVELS = (95, 99)
TRIM = 0.005
EPS = 1.0            # approximate kd-tree search for the cheap first pass; decisions are refined to exact
EVAL_CHECK_ROWS = 5000


def conformal_threshold(scores, level):
    s = np.sort(np.asarray(scores, np.float64))
    n = len(s)
    r = math.ceil((n + 1) * level / 100.0)
    return float(s[r - 1]) if r <= n else math.inf


class Novelty:
    """10th-nearest-neighbour distance to training rows in standard units, with split-conformal thresholds."""

    def __init__(self, train, held_scores, check_scores):
        train = np.asarray(train, np.float64)
        self.mu, self.sd = train.mean(0), train.std(0)
        self.tree = cKDTree((train - self.mu) / self.sd)
        self.set_scores(held_scores, check_scores)

    def set_scores(self, held, check):
        self.held = np.asarray(held, np.float64)
        self.thr = {lv: conformal_threshold(self.held, lv) for lv in LEVELS}
        dc = np.asarray(check, np.float64)
        self.check_rate = {lv: float(np.mean(dc > self.thr[lv])) for lv in LEVELS}
        self.check_quantiles = {q: float(np.quantile(dc, q / 100)) for q in (50, 90, 95, 99)}
        self.held_quantiles = {q: float(np.quantile(self.held, q / 100)) for q in (50, 90, 95, 99)}

    def z(self, q):
        return (np.asarray(q, np.float64) - self.mu) / self.sd

    def dist(self, q):
        return self.tree.query(self.z(q), k=K)[0][:, -1]

    def novel(self, q, thr):
        """Exact decision 'the 10th-neighbour distance exceeds thr': an approximate search (its 10th distance is at
        least the true one and at most (1 + EPS) times it) settles most rows; the rest get an exact bounded search."""
        z = self.z(q)
        d = self.tree.query(z, k=K, eps=EPS)[0][:, -1]
        out = d > thr
        amb = out & (d <= (1.0 + EPS) * thr)
        if amb.any():
            de = self.tree.query(z[amb], k=K, distance_upper_bound=np.nextafter(thr, np.inf))[0][:, -1]
            out[np.flatnonzero(amb)] = ~(de <= thr)
        return out


class Local:
    """How far a command lies outside the per-thruster range of the commands recorded at the 10 training rows nearest
    in state (raw command units, the largest over thrusters; 0 inside), with split-conformal thresholds."""

    def __init__(self, state, actions, held_scores, check_scores):
        self.state, self.actions = state, actions
        Novelty.set_scores(self, held_scores, check_scores)

    def score(self, x, a):
        idx = self.state.tree.query(self.state.z(x), k=K)[1]
        C = self.actions[idx]
        a = np.asarray(a, np.float64)
        return np.max(np.maximum(np.maximum(C.min(1) - a, a - C.max(1)), 0.0), 1)

    def novel(self, x, a, thr):
        return self.score(x, a) > thr


class Reference:
    def __init__(self, sd, cache_dir):
        log, rem, gap = sd.log, sd.removals["gap"], sd.gap
        kept = rem.kept_train_rows(log)
        self.states = np.asarray(log["infos/state"][kept], np.float64)
        self.actions = np.asarray(log["actions"][kept], np.float64)
        cal = log.split_rows("calibration")
        cal = cal[~gap.contains(log["actions"][cal])]
        ev = log.split_rows("evaluation")
        ev = ev[~gap.contains(log["actions"][ev])]
        ev = ev[np.sort(np.random.default_rng([TC.DATA_SEED, sd.rep, 77]).choice(len(ev), EVAL_CHECK_ROWS,
                                                                                  replace=False))]
        xc, ac = np.asarray(log["infos/state"][cal], np.float64), np.asarray(log["actions"][cal], np.float64)
        xe, ae = np.asarray(log["infos/state"][ev], np.float64), np.asarray(log["actions"][ev], np.float64)
        self.n_cal, self.n_check = int(len(cal)), int(len(ev))
        key = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in
                                      (self.states, self.actions, xc, ac, xe, ae))).hexdigest()[:16]
        cache = Path(cache_dir) / f"scores_{sd.world}_seed{sd.rep}_{key}.npz"
        self.state = Novelty(self.states, [0.0], [0.0])
        self.sa = Novelty(np.concatenate([self.states, self.actions], 1), [0.0], [0.0])
        self.local = Local(self.state, self.actions, [0.0], [0.0])
        if cache.exists():
            with np.load(cache) as z:
                sc = {k: z[k] for k in z.files}
        else:
            sc = dict(state_held=self.state.dist(xc), state_check=self.state.dist(xe),
                      sa_held=self.sa.dist(np.concatenate([xc, ac], 1)),
                      sa_check=self.sa.dist(np.concatenate([xe, ae], 1)),
                      local_held=self.local.score(xc, ac), local_check=self.local.score(xe, ae))
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **sc)
        self.state.set_scores(sc["state_held"], sc["state_check"])
        self.sa.set_scores(sc["sa_held"], sc["sa_check"])
        Novelty.set_scores(self.local, sc["local_held"], sc["local_check"])
        self.lo99, self.hi99 = np.quantile(self.actions, TRIM, axis=0), np.quantile(self.actions, 1 - TRIM, axis=0)
        self.lomm, self.himm = self.actions.min(0), self.actions.max(0)
        self.train_past80 = int(TC.past_knee(self.actions).sum())

    def describe(self):
        return dict(training_rows=int(len(self.states)), heldout_rows=self.n_cal, check_rows=self.n_check,
                    training_rows_past_80=self.train_past80,
                    state=dict(threshold=self.state.thr, heldout_quantiles=self.state.held_quantiles,
                               evaluation_split_rate_above=self.state.check_rate,
                               evaluation_split_quantiles=self.state.check_quantiles),
                    state_command=dict(threshold=self.sa.thr, heldout_quantiles=self.sa.held_quantiles,
                                       evaluation_split_rate_above=self.sa.check_rate,
                                       evaluation_split_quantiles=self.sa.check_quantiles),
                    local_command=dict(threshold=self.local.thr, heldout_quantiles=self.local.held_quantiles,
                                       evaluation_split_rate_above=self.local.check_rate,
                                       evaluation_split_quantiles=self.local.check_quantiles),
                    clip99=dict(low=self.lo99.tolist(), high=self.hi99.tolist()),
                    clipmm=dict(low=self.lomm.tolist(), high=self.himm.tolist()))


# ------------------------------------------------------------------------------------------------ exact oracle

_RQ = {}


def remaining_q(actor_params, prepared, physics, x, a, steps):
    """r(x, a) + the actor's next steps - 1 rewards (undiscounted): the exact rest-of-episode return of command a."""
    import jax
    import jax.numpy as jnp

    ctx, _ = TR._precision(True)
    with ctx:
        if physics not in _RQ:
            def f(Ws, bs, mean, std, x, a, steps):
                def pi(z):
                    o = z.astype(jnp.float32).astype(z.dtype)
                    h = (o - mean) / std
                    h = jnp.maximum(h @ Ws[0] + bs[0], 0.0)
                    h = jnp.maximum(h @ Ws[1] + bs[1], 0.0)
                    return jnp.tanh(h @ Ws[2] + bs[2])

                x1, r0 = TC.dynamics_jax(x, a, physics)

                def body(i, c):
                    xx, acc = c
                    xx2, rr = TC.dynamics_jax(xx, pi(xx), physics)
                    return xx2, acc + rr

                return jax.lax.fori_loop(1, steps, body, (x1, r0))[1]

            _RQ[physics] = jax.jit(f)
        Ws, bs = L._mlp_layers(actor_params, jnp.float64)
        mean = jnp.asarray(np.asarray(prepared.obs_mean, np.float64))
        std = jnp.asarray(np.asarray(prepared.obs_std, np.float64))
        return np.asarray(_RQ[physics](Ws, bs, mean, std, jnp.asarray(np.asarray(x, np.float64)),
                                       jnp.asarray(np.asarray(a, np.float64)), jnp.asarray(int(steps), jnp.int32)))


# ------------------------------------------------------------------------------------------------ rollouts

def fallback_command(name, ctx, x, a_act):
    if name == "clone":
        return ctx["clone"](x)
    ref = ctx["ref"]
    if name == "clip99":
        return np.clip(a_act, ref.lo99, ref.hi99)
    if name == "clipmm":
        return np.clip(a_act, ref.lomm, ref.himm)
    if name == "cap80":
        return np.clip(a_act, -0.8, 0.8)
    if name == "project":
        idx = ref.state.tree.query(ref.state.z(x), k=K)[1]
        cand = ref.actions[idx]                                  # (M, K, 4)
        j = np.argmin(np.sum((cand - a_act[:, None, :]) ** 2, -1), 1)
        return cand[np.arange(len(x)), j]
    raise ValueError(name)


def run_rule(ctx, x0, fb, trig, rate=None, coin_seed=None):
    actor, physics, ref = ctx["actor"], ctx["physics"], ctx["ref"]
    x = np.asarray(x0, np.float64).copy()
    n = len(x)
    L_ = physics.length
    J = np.zeros(n)
    fired = np.zeros((n, L_), bool)
    changed = np.zeros((n, L_), bool)
    past_prop = np.zeros((n, L_), bool)
    past_app = np.zeros((n, L_), bool)
    rng = np.random.default_rng(coin_seed) if coin_seed is not None else None
    for t in range(L_):
        a_act = np.clip(actor(x), -1.0, 1.0)
        a = a_act.copy()
        if trig == "none":
            fire = np.zeros(n, bool)
        elif trig == "always":
            fire = np.ones(n, bool)
        elif trig.startswith("state"):
            fire = ref.state.novel(x, ref.state.thr[int(trig[5:])])
        elif trig.startswith("sa"):
            fire = ref.sa.novel(np.concatenate([x, a_act], 1), ref.sa.thr[int(trig[2:])])
        elif trig.startswith("local"):
            fire = ref.local.novel(x, a_act, ref.local.thr[int(trig[5:])])
        elif trig == "unrec":
            fire = TC.past_knee(a_act, physics)
        elif trig.startswith("random") or trig.startswith("rlocal"):
            fire = rng.uniform(size=n) < rate
        elif trig == "oracle":
            a_fb = np.clip(fallback_command(fb, ctx, x, a_act), -1.0, 1.0)
            diff = np.any(a_fb != a_act, 1)
            fire = np.zeros(n, bool)
            if diff.any():
                rows = np.flatnonzero(diff)
                xx = np.concatenate([x[rows], x[rows]])
                aa = np.concatenate([a_act[rows], a_fb[rows]])
                q = remaining_q(ctx["actor_params"], ctx["prepared"], physics, xx, aa, L_ - t)
                q_act, q_fb = q[:len(rows)], q[len(rows):]
                fire[rows] = q_fb > q_act + 1e-9 * (1.0 + np.abs(q_act))
            a[fire] = a_fb[fire]
        else:
            raise ValueError(trig)
        if trig != "oracle" and fire.any():
            a[fire] = np.clip(fallback_command(fb, ctx, x[fire], a_act[fire]), -1.0, 1.0)
        fired[:, t] = fire
        changed[:, t] = np.any(a != a_act, 1)
        past_prop[:, t] = TC.past_knee(a_act, physics)
        past_app[:, t] = TC.past_knee(a, physics)
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    pts = TR.points(J, ctx["anchor"])
    fp = fired & past_prop
    return dict(points=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)), J=float(J.mean()),
                fire_share=float(fired.mean()), changed_share=float(changed.mean()),
                past80_proposed=float(past_prop.mean()), past80_applied=float(past_app.mean()),
                recall_past80=float(fp.sum() / max(past_prop.sum(), 1)),
                precision_past80=float(fp.sum() / max(fired.sum(), 1)),
                rate=rate, per_start_points=np.round(pts, 5).tolist())


FALLBACKS = ("clone", "clip99", "project")
PLANS = dict(
    full=(("none", None), ("always", "clone"), ("always", "clip99"), ("always", "project"), ("always", "clipmm"),
          ("state95", "clone"), ("state95", "clip99"), ("state95", "project"), ("state99", "clone"),
          ("state99", "clip99"),
          ("sa95", "clone"), ("sa95", "clip99"), ("sa95", "project"), ("sa99", "clone"), ("sa99", "clip99"),
          ("local95", "clone"), ("local95", "clip99"), ("local99", "clone"), ("local99", "clip99"),
          ("unrec", "clone"), ("unrec", "clip99"), ("oracle", "clone"), ("oracle", "clip99"),
          ("random95", "clone"), ("random95", "clip99"), ("random95", "project"), ("random99", "clone"),
          ("random99", "clip99"), ("rlocal95", "clone"), ("rlocal95", "clip99")),
    medium=(("none", None), ("always", "clone"), ("always", "clip99"), ("always", "project"), ("always", "clipmm"),
            ("state95", "clone"), ("state95", "clip99"), ("sa95", "clone"), ("sa95", "clip99"), ("sa99", "clone"),
            ("sa99", "clip99"), ("local95", "clip99"), ("unrec", "clone"), ("unrec", "clip99"), ("oracle", "clone"),
            ("oracle", "clip99"), ("random95", "clone"), ("random95", "clip99"), ("random99", "clone"),
            ("random99", "clip99"), ("rlocal95", "clip99")),
    short=(("none", None), ("always", "clone"), ("always", "clip99"), ("sa95", "clone"), ("sa95", "clip99"),
           ("sa99", "clip99"), ("local95", "clip99"), ("unrec", "clip99"), ("oracle", "clip99"),
           ("random95", "clone"), ("random95", "clip99"), ("random99", "clip99"), ("rlocal95", "clip99")),
)
MATCH = {"random95": "sa95", "random99": "sa99", "rlocal95": "local95"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("world", choices=("stall", "no_stall"))
    ap.add_argument("actor")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--situations", nargs="*", default=list(TC.SITUATIONS))
    ap.add_argument("--plan", default="full", choices=tuple(PLANS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rep = args.seed
    started = time.perf_counter()
    prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(prereg)
    cfg = prereg.config
    outcome = A.gate_outcome(prereg)
    variant = outcome["variant"]
    alpha = outcome.get("alpha") or A._last_alpha_run(prereg, variant)
    sd = A.load_seed(prereg.runs, cfg, variant, args.world, rep)
    pg = sd.prepared("gap", cfg.learn)
    d = A._variant_root(prereg, "harm_return", variant) / f"alpha{alpha:g}" / f"seed{rep}"
    rec = A.finished(d / "values.json", prereg)
    acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]},
                       rec["actors_sha256"])
    clone_key = "no_stall_clone" if args.world == "no_stall" else "clone"
    if rec["failed"].get(args.actor) or args.actor not in acts:
        raise SystemExit(f"actor {args.actor} failed or missing in seed {rep}")
    an = A.anchors_of(prereg, variant, args.world)
    t0 = time.perf_counter()
    ref = Reference(sd, Path(__file__).resolve().parent / "cache")
    print(f"seed {rep} {args.world}: reference built in {time.perf_counter() - t0:.1f}s", flush=True)
    ctx = dict(actor=L.mlp_policy(acts[args.actor], pg, args.actor), actor_params=acts[args.actor], prepared=pg,
               clone=L.mlp_policy(acts[clone_key], pg, clone_key), physics=sd.physics, ref=ref)
    plan = PLANS[args.plan]
    out = dict(kind="shift-explore-fallback", exploratory=True,
               note="EXPLORATORY deployment-time switching diagnostic; no training, no BCA; changes no "
                    "pre-registered rule. Gate outputs read through the gate's verified loaders.",
               prereg_sha256=prereg.digest, variant=variant, alpha=alpha, rep=rep, world=args.world,
               actor=args.actor, n_starts=args.n, plan=args.plan, start_subset="the first n starts of each 1,000-start bank",
               reference=ref.describe(), situations={})
    path = OUT / f"seed{rep}_{args.world}_{args.actor}{args.tag}.json"
    for s in args.situations:
        ctx["anchor"] = an[s]
        bank = TC.bank_starts(s)
        # check against the gate: no switching and capping at 80% on the whole bank reproduce H2's numbers
        full_none = run_rule(ctx, bank, None, "none")
        full_cap = run_rule(ctx, bank, "cap80", "always")
        gate = rec["evaluations"][args.actor][s]
        check = dict(points=full_none["points"], gate_points=gate["points"], capped=full_cap["points"],
                     gate_capped=gate["capped_points"],
                     max_abs_diff=max(abs(full_none["points"] - gate["points"]),
                                      abs(full_cap["points"] - gate["capped_points"])))
        print(f"  {s}: gate check diff {check['max_abs_diff']:.2e}", flush=True)
        x0 = bank[:args.n]
        res = {}
        for trig, fb in plan:
            t1 = time.perf_counter()
            rate, coin = None, None
            if trig in MATCH:
                src = res.get(f"{MATCH[trig]}:{fb}")
                if src is None:
                    continue
                rate = src["fire_share"]
                coin = [20261012, rep, TC.SITUATIONS.index(s), FALLBACKS.index(fb), sum(map(ord, trig)),
                        sum(map(ord, args.actor)), 0 if args.world == "stall" else 1]
            r = run_rule(ctx, x0, fb, trig, rate, coin)
            r["seconds"] = time.perf_counter() - t1
            res[f"{trig}:{fb}"] = r
            print(f"  {s} {trig:9s} {str(fb):8s} pts {r['points']:7.2f} fire {r['fire_share']:.3f} "
                  f"chg {r['changed_share']:.3f} past80 {r['past80_applied']:.3f} ({r['seconds']:.0f}s)", flush=True)
        out["situations"][s] = dict(gate_check=check, rules=res)
        out["seconds"] = time.perf_counter() - started
        A.write_json(path, out)
    print(f"wrote {path} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
