"""Route 'runtime_monitor', phase 1 (per seed): plumbing checks, exact forward rollouts, per-step scores.

EXPLORATORY, designed after seeing seeds 0-4. Seeds 0-4 only. No training, no new log: saved actors and critics are
run forward. Kill rule fixed beforehand in scratchpad/routes/runtime_monitor/KILL_RULE.md.

Four worlds (wi = world index):
  S  (1): stalling log,  stalling physics at deployment
  N  (2): no-stall log,  no-stall physics
  M3 (3): stall only when current > 0.7  -> log = the no-stall log, strong-current deployment = stalling physics
  M4 (4): stall only when current <= 0.7 -> log = the stalling log, strong-current deployment = no-stall physics
Writes only under runs/shift_explore/routes/runtime_monitor/seed<k>/.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, "/home/user/bca")
sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/restructure/step1")
import step1lib as S1  # noqa: E402  (read-only use: gap_rule, throttle, overestimate, guards)

import jax.numpy as jnp  # noqa: E402
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402

OUT = Path("/home/user/bca/runs/shift_explore/routes/runtime_monitor")
R2 = Path("/home/user/bca/runs/shift_explore/stage2")
STEP1 = Path("/home/user/bca/outputs/shift/beyond-the-data/restructure/step1")
WORLDS4 = {  # key: (world index, log world, deployment physics world)
    "S": (1, "stall", "stall"), "N": (2, "no_stall", "no_stall"),
    "M3": (3, "no_stall", "stall"), "M4": (4, "stall", "no_stall")}
CUR_CUT = 0.7
TOL = 1e-9


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def guard(path):
    p = Path(path).resolve()
    for pin in S1.PINNED:
        if S1.inside(p, pin):
            raise SystemExit(f"refused: {p} inside pinned {pin}")
    if not S1.inside(p, OUT):
        raise SystemExit(f"refused: {p} outside {OUT}")
    return p


def none_actor(world, seed):
    d = R2 / "policy" / f"seed{seed}"
    if world == "no_stall":
        d = d / "no_stall"
    lock = S1.read_json_checked(d / "lock.json")
    acts = L.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    if L.tree_sha256(acts["none"]) != lock["actor_sha256"]["none"]:
        raise SystemExit(f"refused: the 'none' actor of {d} differs from its lock")
    return acts["none"], lock["actor_sha256"]["none"]


def make_dyn(key, phys):
    """Deployment dynamics of a world key. M3/M4 are written out as current-dependent physics."""
    if key == "S":
        return lambda x, a: TC.dynamics(x, a, phys["stall"])
    if key == "N":
        return lambda x, a: TC.dynamics(x, a, phys["no_stall"])

    def dyn(x, a):
        cur = np.asarray(x, np.float64)[:, TC.CURRENT]
        stall_rows = cur > CUR_CUT if key == "M3" else cur <= CUR_CUT
        n1, r1 = TC.dynamics(x, a, phys["stall"])
        n2, r2 = TC.dynamics(x, a, phys["no_stall"])
        return np.where(stall_rows[:, None], n1, n2), np.where(stall_rows, r1, r2)
    return dyn


def rollout(policy, dyn, length, x0, lo, hi, t_lo, mode, record=False):
    """Same arithmetic as step1 evaluate.rollout ('never' / 'on'); per-start returns; optional trajectory."""
    x = np.asarray(x0, np.float64).copy()
    n = len(x)
    J = np.zeros(n)
    fired = np.zeros((n, length), bool)
    X, Aex, R, Xn = [], [], [], []
    for k in range(length):
        a_act = np.clip(policy(x), -1.0, 1.0)
        fire = (np.abs(a_act).max(1) > t_lo) if mode == "on" else np.zeros(n, bool)
        a = np.where(fire[:, None], np.clip(a_act, lo, hi), a_act)
        fired[:, k] = fire
        xn, r = dyn(x, a)
        if record:
            X.append(x); Aex.append(a); R.append(r); Xn.append(xn)
        x = xn
        J = J + r
    out = dict(J=J, fire_share=float(fired.mean()), fired=fired)
    if record:
        out.update(X=np.stack(X, 1), A=np.stack(Aex, 1), R=np.stack(R, 1), Xn=np.stack(Xn, 1))
    return out


def scores(critic, fixed, pg, X, Aex, R, Xn, gamma):
    """One-sided overestimate of every executed step (flattened row-major: episode, step)."""
    x, a, r, xn = (v.reshape(-1, v.shape[-1]) if v.ndim == 3 else v.reshape(-1) for v in (X, Aex, R, Xn))
    o = pg.normalize(x)
    on = pg.normalize(xn)
    q = L.critic_rows(critic, o, a.astype(np.float32))[0].min(1)
    c = np.asarray(fixed.apply(fixed.params(), jnp.asarray(on, jnp.float32)), np.float32)
    qn = L.critic_rows(critic, on, c)[0].min(1)
    r32 = r.astype(np.float32).astype(np.float64)
    return S1.overestimate(q, r32, np.zeros(len(r32)), qn, gamma)


def main(seed):
    t_cpu0 = time.process_time()
    t0 = time.perf_counter()
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise SystemExit("refused: PYTHONDONTWRITEBYTECODE must be 1")
    spec, _ = S1.load_prespec()
    seen = spec["deployment"]["seen_design_values"]
    p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(p)
    variant = A.gate_outcome(p)["variant"]
    learn = p.config.learn
    gamma = float(learn.gamma)
    assert abs(gamma - 0.98) == 0
    out_dir = guard(OUT / f"seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "prep.json").exists():
        raise SystemExit(f"refused: {out_dir}/prep.json exists (write-once)")
    checks, info = {}, {}
    ctx = {}
    for world in ("stall", "no_stall"):
        sd = A.load_seed(p.runs, p.config, variant, world, seed)
        pg = sd.prepared("gap", learn)
        lg = sd.log
        # plumbing 1: the simulator reproduces every logged row
        st = np.asarray(lg["infos/state"], np.float64)
        nst = np.asarray(lg["infos/next_state"], np.float64)
        act = np.asarray(lg["actions"])
        rew = np.asarray(lg["rewards"])
        xn, r = TC.dynamics(st, act, pg.physics)
        d_state = float(np.max(np.abs(xn - nst)))
        d_rew = float(np.max(np.abs(r.astype(np.float32).astype(np.float64) - rew.astype(np.float64))))
        cur_max = float(st[:, TC.CURRENT].max())
        # the M3 / M4 dynamics on the log equal the log world's own physics (every log current <= 0.7)
        ok_log_cur = bool(cur_max <= CUR_CUT and float(nst[:, TC.CURRENT].max()) <= CUR_CUT)
        checks[f"{world}/sim_reproduces_log"] = bool(d_state <= TOL and d_rew <= TOL)
        checks[f"{world}/log_current_le_0.7"] = ok_log_cur
        info[f"{world}/sim_max_abs_diff"] = dict(next_state=d_state, reward_float32=d_rew, rows=int(len(st)),
                                                  log_current_max=cur_max)
        if not (checks[f"{world}/sim_reproduces_log"] and ok_log_cur):
            raise SystemExit(f"PLUMBING STOP: {world} seed {seed}: {info[f'{world}/sim_max_abs_diff']}")
        gcfg = spec["groups"]["gap_rule"]
        gr = S1.gap_rule(S1.throttle(pg.actions), gcfg["upper_quantile"], gcfg["min_gap"], gcfg["min_tail"])
        if not gr["defined"]:
            raise SystemExit("gap rule undefined")
        a64 = np.asarray(pg.actions, np.float64)
        lo, hi = np.quantile(a64, 0.005, axis=0), np.quantile(a64, 0.995, axis=0)
        actor, actor_sha = none_actor(world, seed)
        pol = L.mlp_policy(actor, pg, "none")
        grec, gcrit = A.get_critic(p, sd, "gap", "stall_aware", train=False)
        if gcrit is None:
            raise SystemExit("gate critic failed")
        fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
        # offline bank rest group (reported only): bank rows with t <= t_lo, gate-critic scores
        bk = pg.bank
        bq = L.critic_rows(gcrit, np.asarray(bk.obs), np.asarray(bk.action))[0].min(1)
        bc = np.asarray(fixed.apply(fixed.params(), jnp.asarray(np.asarray(bk.next_obs), jnp.float32)), np.float32)
        bqn = L.critic_rows(gcrit, np.asarray(bk.next_obs), bc)[0].min(1)
        bs = S1.overestimate(bq, np.asarray(bk.reward, np.float64), np.asarray(bk.done, np.float64), bqn, gamma)
        bt = S1.throttle(np.asarray(bk.action))
        ctx[world] = dict(sd=sd, pg=pg, gr=gr, lo=lo, hi=hi, pol=pol, crit=gcrit, fixed=fixed, actor_sha=actor_sha,
                          critic_sha=grec["result"]["file_sha256"], anchors=A.anchors_of(p, variant, world),
                          bank_rest=bs[~S1.full_throttle(bt, gr["t_lo"])], bank_tail=bs[S1.full_throttle(bt, gr["t_lo"])])
    phys = {w: ctx[w]["pg"].physics for w in ("stall", "no_stall")}
    if not (phys["stall"].stall and not phys["no_stall"].stall):
        raise SystemExit("physics mix-up")
    res = dict(seed=seed, checks=checks, info=info, worlds={})
    arrays = {}
    jobs = [(k, "strong_current") for k in WORLDS4] + [("S", "recorded"), ("N", "recorded")]
    for key, sit in jobs:
        wi, lw, dw = WORLDS4[key]
        c = ctx[lw]
        dyn = make_dyn(key, phys)
        x0 = TC.bank_starts(sit, TC.BANK_SIZE)[:200]
        anchor = ctx[dw]["anchors"][sit]
        length = phys[dw].length
        nev = rollout(c["pol"], dyn, length, x0, c["lo"], c["hi"], c["gr"]["t_lo"], "never", record=True)
        on = rollout(c["pol"], dyn, length, x0, c["lo"], c["hi"], c["gr"]["t_lo"], "on", record=False)
        pn, po = TR.points(nev["J"], anchor), TR.points(on["J"], anchor)
        tag = f"{key}/{sit}"
        if key in ("M3", "M4"):
            cur = nev["X"][..., TC.CURRENT]
            ok = bool(cur.min() > CUR_CUT and np.all(cur == cur[:, :1]))
            # exactness: the current-dependent dynamics equal plain deployment physics on these states
            same = make_dyn("S" if dw == "stall" else "N", phys)
            on2 = rollout(c["pol"], same, length, x0, c["lo"], c["hi"], c["gr"]["t_lo"], "on")
            nev2 = rollout(c["pol"], same, length, x0, c["lo"], c["hi"], c["gr"]["t_lo"], "never")
            ok = ok and bool(np.array_equal(on2["J"], on["J"]) and np.array_equal(nev2["J"], nev["J"]))
            checks[f"{tag}/deployment_current_gt_0.7_and_exact"] = ok
            if not ok:
                raise SystemExit(f"PLUMBING STOP: {tag} current check")
        if key in ("S", "N"):
            sv = seen[f"{lw}/seed{seed}/{sit}"]
            dn, do = abs(float(pn.mean()) - sv["never"]), abs(float(po.mean()) - sv["on"])
            checks[f"{tag}/rollouts_reproduce_step1_design"] = bool(dn <= TOL and do <= TOL)
            info[f"{tag}/reproduction_abs_diff"] = dict(never=dn, on=do)
            if not checks[f"{tag}/rollouts_reproduce_step1_design"]:
                raise SystemExit(f"PLUMBING STOP: {tag} rollouts {dn} {do}")
        sc = scores(c["crit"], c["fixed"], c["pg"], nev["X"], nev["A"], nev["R"], nev["Xn"], gamma)
        thr = np.abs(nev["A"]).max(-1)                       # float64 throttle of the executed command
        tailm = thr > c["gr"]["t_lo"]
        # the 'on' rule fires exactly on these steps of the never rollout (same command before any clip)
        assert np.array_equal(tailm, np.abs(np.clip(nev["A"], -1, 1)).max(-1) > c["gr"]["t_lo"])
        arrays[f"{key}_{sit}_score"] = sc.reshape(200, length)
        arrays[f"{key}_{sit}_tail"] = tailm
        arrays[f"{key}_{sit}_pts_never"] = pn
        arrays[f"{key}_{sit}_pts_on"] = po
        res["worlds"][tag] = dict(world_index=wi, log_world=lw, deploy_physics=dw, t_lo=c["gr"]["t_lo"],
                                  never=float(pn.mean()), on=float(po.mean()), fire_share_on=on["fire_share"],
                                  tail_share_never=float(tailm.mean()), score_mean_tail=float(sc.reshape(200, -1)[tailm].mean()),
                                  score_mean_rest=float(sc.reshape(200, -1)[~tailm].mean()) if (~tailm).any() else None,
                                  actor_sha256=c["actor_sha"], critic_sha256=c["critic_sha"])
        log(f"seed {seed} {tag}: never {pn.mean():.2f} on {po.mean():.2f} tail share {tailm.mean():.3f} "
            f"mean score tail {res['worlds'][tag]['score_mean_tail']:+.3f} rest {res['worlds'][tag]['score_mean_rest']}")
    for world in ("stall", "no_stall"):
        arrays[f"bank_rest_{world}"] = ctx[world]["bank_rest"]
        arrays[f"bank_tail_{world}"] = ctx[world]["bank_tail"]
    np.savez_compressed(guard(out_dir / "arrays.npz"), **arrays)
    res["cpu_seconds"] = time.process_time() - t_cpu0
    res["wall_seconds"] = time.perf_counter() - t0
    res["label"] = "EXPLORATORY route test (runtime_monitor), designed after seeing seeds 0-4; seeds 0-4 only."
    (guard(out_dir / "prep.json")).write_text(json.dumps(S1.jsonable(res), indent=1))
    log(f"seed {seed} done: cpu {res['cpu_seconds']:.0f}s wall {res['wall_seconds']:.0f}s; checks all "
        f"{all(checks.values())}")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(int(s))
