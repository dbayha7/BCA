"""Step 2: train ONE world of one seed with version 6's actor-only policy-test protocol and the outcome-ramp arms.
EXPLORATORY; see PRESPEC.md.

  --world stall      every arm of the stalling world: the w0 check, the six outcome-ramp arms, the perfect band, the
                     room's range hinge, the controls' ladders, the controls and the best stand-in matched blind to
                     obca_k1's strength; then lock.json (no return is computed here).
  --world no_stall   after the stalling world's lock of the same seed: the six outcome-ramp arms with their no-stall
                     (switch, height), the w0 check, the perfect band, the range hinge, and the even pull, push less,
                     random surfaces and stand-in at the stalling world's locked coefficients; then lock.json.

Official mode (no --out-root): writes only runs/shift_explore/restructure/step2/policy/seed<k>/<world>/
{actors.msgpack, lock.json(.sha256)} or stop.json(.sha256), write-once, after checking prespec.json's SHA-256 and the
code manifest. 20,000 actor steps in 4,000-step chunks. It must run pinned to one CPU (taskset -c <cpu>), with
OMP_NUM_THREADS=1, JAX_PLATFORMS=cpu and PYTHONDONTWRITEBYTECODE=1.

Scratch mode (--out-root DIR outside the repository): the same code path with --steps / --chunk (smoke test), and
--w0-repro (the reproduction check: at full length, the ramp program with every height 0 must reproduce check 2's
saved 'none' actor; no scored arm is trained). Records are labelled scratch; the official evaluation refuses them.

Usage:
  taskset -c 0 python train_policy.py --seed 0 --world stall
  taskset -c 0 python train_policy.py --seed 0 --world no_stall
  taskset -c 0 python train_policy.py --seed 0 --world stall --out-root /root/step2_scratch/smoke --steps 40 --chunk 20
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2lib as S  # noqa: E402

LOCKED_FAMILIES = ("even_pull", "push_less", "random_1", "random_2")


def reference(MC, runner, trained, none, full):
    """The actor every height-0 arm must reproduce: check 2's saved 'none' actor at full length (version 6's check
    (b)); in a shortened scratch run, the room's own W = 0 program (kind plain, pull 1) at the same step count."""
    if full:
        return none, "check 2's saved 'none' actor (version 6's check (b))"
    if "plain_ref" not in trained:
        trained.update(runner.run([MC.H.RoomArm("plain_ref", "plain", (0.0,), 1.0)], MC.log))
    return trained["plain_ref"]["params"], "scratch only: the room's W = 0 program at the same step count"


def h0_check(MC, trained, names, ref, ref_name):
    rows = {}
    for n in names:
        d = MC.H.max_param_diff(trained[n]["params"], ref)
        rows[n] = dict(max_abs_diff=d, passed=bool(trained[n]["finite"] and d <= MC.W0_TOL))
    return dict(arms=rows, tol=MC.W0_TOL, reference=ref_name, passed=bool(all(r["passed"] for r in rows.values())))


def strengths(MC, trained, extra, none, pg):
    s, sr = MC.P.strengths_of(trained, none, pg)
    for k, p in extra.items():
        s[k] = float(MC.L.strength(p, none, pg.eval_obs))
        sr[k] = MC.H.region_strength(p, none, pg)
    return s, sr


def train_stall(MC, seed, table, d, base, official, steps, chunk, full, t_start):
    H, L, P = MC.H, MC.L, MC.P
    row = table["stall"][seed]
    t0 = time.perf_counter()
    ctx = MC.context(seed, "stall")
    pg = ctx.prepared("stall")
    critic = ctx.critic("stall", "gap")[0]
    clone = ctx.clone("stall")[0]
    none = ctx.none("stall")
    full_none = ctx.full_none()
    best = MC.best_standin()
    checks = dict(gap_rule=MC.gap_rule_check(pg, row))
    base = dict(base, gate_prereg_sha256=ctx.prereg.digest, arm_table=row)
    if not checks["gap_rule"]["passed"]:
        MC.stop(d, dict(base, checks=checks), official, "step 1's gap rule gives another t_lo or t_up on this data")
    seconds = dict(load=time.perf_counter() - t0)
    t1 = time.perf_counter()
    cfg = H.ProgramConfig()
    shared, sinfo = H.build_shared(pg, critic, seed, cfg=cfg, need_harm=False, log_fn=MC.log)
    nets = {j: H.random_surface_params(seed, j) for j in (1, 2)}
    g, scale = P.room_scales(pg, critic, shared, clone, seed, cfg, nets)
    seconds["shared_and_scales"] = time.perf_counter() - t1
    runner = MC.RampRunner(pg, clone, critic, ctx.learn, seed, cfg=cfg, shared=shared,
                           random_nets={j: nets[j][0] for j in (1, 2)}, steps=steps, chunk=chunk,
                           t_lo=row["t_lo"], t_up=row["t_up"])
    t1 = time.perf_counter()
    trained = runner.run([MC.ramp_arm(S.W0, 0.0, 0.0)], MC.log)          # alone first, as version 6's check (b)
    ref, ref_name = reference(MC, runner, trained, none, full)
    w0 = H.max_param_diff(trained[S.W0]["params"], ref)
    checks["b_w0"] = dict(max_abs_diff=w0, tol=MC.W0_TOL, reference=ref_name,
                          passed=bool(trained[S.W0]["finite"] and w0 <= MC.W0_TOL))
    if not checks["b_w0"]["passed"]:
        MC.stop(d, dict(base, checks=checks), official,
                "check (b) failed: the ramp program with height 0 differs from the reference actor")
    arms = [MC.ramp_arm(n, row["arms"][n]["h"], row["arms"][n]["k"]) for n in S.RAMP_ARMS]
    arms += [H.RoomArm("perfect_band", "band", H.MAIN_BAND), H.RoomArm("range_hinge", "hinge", (1.0,))]
    arms += [H.family_arm("even_pull", nu, scale) for nu in H.EVEN_LADDER]
    arms += [H.family_arm(f, k, scale) for f in ("push_less", "random_1", "random_2") for k in H.SLOPE_LADDER]
    trained.update(runner.run(arms, MC.log))
    zero = [n for n in S.RAMP_ARMS if row["arms"][n]["h"] == 0.0]
    checks["h0_reproduces_reference"] = h0_check(MC, trained, zero, ref, ref_name)
    if not checks["h0_reproduces_reference"]["passed"]:
        MC.stop(d, dict(base, checks=checks), official, "an outcome-ramp arm with height 0 differs from the reference")
    seconds["arms"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    target = float(L.strength(trained[S.PRIMARY]["params"], none, pg.eval_obs)) if trained[S.PRIMARY]["finite"] \
        else float("nan")
    if np.isfinite(target) and target > 0:
        matches = P.match_controls(runner, trained, none, pg, target, scale, P.MATCHED, MC.log)
    else:
        matches = {f: dict(record=None, arm=None, matched=False, reason="obca_k1 failed or has no positive strength")
                   for f in P.MATCHED}
    seconds["matching"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    standin = dict(family=best, matched=False, arm=None)
    s_actors, s_record = {}, {}
    if np.isfinite(target) and target > 0:
        try:
            SM = P.standins_module(MC.v6_prereg())
            s_trained, s_match, s_info = MC.standin_matched(SM, ctx, "stall", pg, critic, clone, none, best, target,
                                                            shared, cfg, steps, chunk, MC.log)
            name = s_match["arm"]
            ok = name is not None and s_trained[name]["finite"]
            standin.update(record=s_match["record"], matched=bool(ok), arm=f"standin:{name}" if ok else None,
                           coefficient=list(s_trained[name]["arm"]["coef"]) if ok else None,
                           knob=s_trained[name]["arm"]["knob"] if ok else None,
                           reason=None if ok else "no finite matched arm", data_side=s_info["info"],
                           scale=s_info["scale"], seconds=s_info["seconds"])
            s_actors = {f"standin:{k}": v["params"] for k, v in s_trained.items() if v["finite"]}
            s_record = {f"standin:{k}": dict(arm=v["arm"], finite=v["finite"], chunk_metrics=v["chunk_metrics"])
                        for k, v in s_trained.items()}
        except Exception as error:   # a rival that cannot run fails every measure that needs it (counts against BCA)
            standin.update(error=repr(error))
            MC.log(f"stand-in error: {error!r}")
    seconds["standin"] = time.perf_counter() - t1
    S_, SR = strengths(MC, trained, s_actors, none, pg)
    shape = MC.info_only("ramp shape", lambda: MC.ramp_shape(
        pg, critic, none, seed, row["t_lo"], row["t_up"],
        {n: (row["arms"][n]["h"], row["arms"][n]["k"]) for n in S.RAMP_ARMS}))
    ids = dict(none="none", clone="clone", full_none="full_none", perfect_band="perfect_band",
               range_hinge="range_hinge", **{n: n for n in S.RAMP_ARMS})
    ids.update({f: matches[f]["arm"] for f in P.MATCHED})
    ids["standin"] = standin["arm"]
    actors = {k: v["params"] for k, v in trained.items()}
    actors.update(s_actors)
    actors.update(none=none, clone=clone, full_none=full_none)
    finite = {k: bool(v["finite"]) for k, v in trained.items()}
    finite.update({k: True for k in s_actors}, none=True, clone=True, full_none=True)
    scored = sorted({v for v in ids.values() if v})
    payload = dict(base, kind="step2-policy-lock", world="stall", checks=checks,
                   arms=dict(P.arms_record(trained), **s_record), strengths=S_, region_strengths=SR,
                   target_strength=target, matches=matches, matched_names={f: m["arm"] for f, m in matches.items()},
                   standin=standin, arm_ids=ids, scored_names=scored, finite={k: finite[k] for k in scored},
                   slope_rms=g, coefficient_per_knob=scale, random_surface_seeds={j: nets[j][1] for j in (1, 2)},
                   shared=sinfo, ramp_shape=shape, chunk_seconds=runner.chunk_seconds,
                   seconds=dict(seconds, by_kind=runner.seconds, total=time.perf_counter() - t_start),
                   vmhwm_mb=S.vmhwm_mb(), **MC.machine_record())
    MC.save_and_lock(d, actors, payload, official)
    MC.log(f"stall/seed {seed}: locked; target strength {target:.4f}; matched {payload['matched_names']}; stand-in "
           f"{standin['arm']}; {time.perf_counter() - t_start:.0f}s; VmHWM {S.vmhwm_mb():.0f} MB")


def locked_controls(MC, stall_lock):
    """The even pull, push less and random surfaces at the stalling world's locked knobs and coefficients (the same
    penalty in both worlds; never re-matched)."""
    H = MC.H
    arms, info = [], {}
    for f in LOCKED_FAMILIES:
        m = stall_lock["matches"][f]
        if not m.get("matched") or not m.get("arm"):
            info[f] = dict(locked=False, reason="not matched in the stalling world")
            continue
        rec = stall_lock["arms"][m["arm"]]["arm"]
        name = f"{f}_locked"
        if f == "even_pull":
            nu = float(rec["knob"])
            arms.append(H.RoomArm(name, "plain", (0.0,), 1.0 + nu, family=f, knob=nu))
        elif f == "push_less":
            arms.append(H.RoomArm(name, "push_less", tuple(float(c) for c in rec["coef"]), family=f,
                                  knob=float(rec["knob"])))
        else:
            arms.append(H.RoomArm(name, "random", tuple(float(c) for c in rec["coef"]), surface=int(rec["surface"]),
                                  family=f, knob=float(rec["knob"])))
        info[f] = dict(locked=True, name=name, stall_arm=m["arm"], knob=rec["knob"], coef=rec["coef"],
                       pull=rec["pull"], surface=rec["surface"])
    return arms, info


def train_no_stall(MC, seed, table, d, base, official, steps, chunk, full, t_start, root, spec_sha, man_sha):
    H, L, P = MC.H, MC.L, MC.P
    row = table["no_stall"][seed]
    base = dict(base, arm_table=row)
    stall_d = S.seed_dir(root, seed, "stall")
    if (stall_d / "stop.json").exists():
        MC.stop(d, base, official, "the stalling world of this seed stopped: no locked coefficients exist")
    stall_lock, _, stall_lock_sha = MC.verify_lock(stall_d, spec_sha, man_sha, seed, "stall", official, steps)
    if stall_lock.get("steps") != steps or stall_lock.get("mode") != base["mode"]:
        raise SystemExit("refused: the stalling world's lock was trained with another step count or mode")
    t0 = time.perf_counter()
    ctx = MC.context(seed, "no_stall")
    pt = ctx.prepared("no_stall")
    critic = ctx.critic("no_stall", "gap")[0]
    clone = ctx.clone("no_stall")[0]
    none = ctx.none("no_stall")
    checks = dict(gap_rule=MC.gap_rule_check(pt, row))
    base = dict(base, gate_prereg_sha256=ctx.prereg.digest, stall_lock_sha256=stall_lock_sha)
    if not checks["gap_rule"]["passed"]:
        MC.stop(d, dict(base, checks=checks), official, "step 1's gap rule gives another t_lo or t_up on this data")
    seconds = dict(load=time.perf_counter() - t0)
    t1 = time.perf_counter()
    cfg = H.ProgramConfig(physics=pt.physics, next_command="stall_aware")
    shared, sinfo = H.build_shared(pt, critic, seed, cfg=cfg, need_harm=False, log_fn=MC.log)
    nets = {j: H.random_surface_params(seed, j) for j in (1, 2)}
    seconds["shared"] = time.perf_counter() - t1
    runner = MC.RampRunner(pt, clone, critic, ctx.learn, seed, cfg=cfg, shared=shared,
                           random_nets={j: nets[j][0] for j in (1, 2)}, steps=steps, chunk=chunk,
                           t_lo=row["t_lo"], t_up=row["t_up"])
    locked, locked_info = locked_controls(MC, stall_lock)
    arms = [MC.ramp_arm(S.W0, 0.0, 0.0)] + [MC.ramp_arm(n, row["arms"][n]["h"], row["arms"][n]["k"])
                                            for n in S.RAMP_ARMS]
    arms += [H.RoomArm("perfect_band", "band", H.MAIN_BAND), H.RoomArm("range_hinge", "hinge", (1.0,))] + locked
    t1 = time.perf_counter()
    trained = runner.run(arms, MC.log)
    ref, ref_name = reference(MC, runner, trained, none, full)
    w0 = H.max_param_diff(trained[S.W0]["params"], ref)
    checks["b_w0"] = dict(max_abs_diff=w0, tol=MC.W0_TOL, reference=ref_name,
                          passed=bool(trained[S.W0]["finite"] and w0 <= MC.W0_TOL))
    zero = [n for n in S.RAMP_ARMS if row["arms"][n]["h"] == 0.0]
    checks["h0_reproduces_reference"] = h0_check(MC, trained, zero, ref, ref_name)
    if not (checks["b_w0"]["passed"] and checks["h0_reproduces_reference"]["passed"]):
        MC.stop(d, dict(base, checks=checks), official,
                "check (b) failed in the no-stall world: a height-0 ramp arm differs from the reference actor")
    seconds["arms"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    standin = dict(family=stall_lock["standin"].get("family"), coefficient=stall_lock["standin"].get("coefficient"),
                   stall_arm=stall_lock["standin"].get("arm"), arm=None)
    s_actors, s_record = {}, {}
    if standin["coefficient"] is not None:
        try:
            SM = P.standins_module(MC.v6_prereg())
            st, s_info = MC.standin_locked(SM, ctx, "no_stall", pt, critic, clone, none, standin["family"],
                                           standin["coefficient"], steps, chunk, MC.log)
            if st["finite"]:
                s_actors = {"standin_locked": st["params"]}
                standin["arm"] = "standin_locked"
            s_record = {"standin_locked": dict(arm=st["arm"], finite=st["finite"], chunk_metrics=st["chunk_metrics"])}
            standin.update(no_stall_data_side=s_info["info"], seconds=s_info["seconds"])
        except Exception as error:
            standin["error"] = repr(error)
            MC.log(f"stand-in error: {error!r}")
    seconds["standin"] = time.perf_counter() - t1
    S_, SR = strengths(MC, trained, s_actors, none, pt)
    shape = MC.info_only("ramp shape", lambda: MC.ramp_shape(
        pt, critic, none, seed, row["t_lo"], row["t_up"],
        {n: (row["arms"][n]["h"], row["arms"][n]["k"]) for n in S.RAMP_ARMS}))
    ids = dict(none="none", clone="clone", perfect_band="perfect_band", range_hinge="range_hinge",
               **{n: n for n in S.RAMP_ARMS})
    ids.update({f: (locked_info[f]["name"] if locked_info[f]["locked"] else None) for f in LOCKED_FAMILIES})
    ids["standin"] = standin["arm"]
    actors = {k: v["params"] for k, v in trained.items()}
    actors.update(s_actors)
    actors.update(none=none, clone=clone)
    finite = {k: bool(v["finite"]) for k, v in trained.items()}
    finite.update({k: True for k in s_actors}, none=True, clone=True)
    scored = sorted({v for v in ids.values() if v})
    payload = dict(base, kind="step2-policy-lock", world="no_stall", checks=checks,
                   arms=dict(P.arms_record(trained), **s_record), strengths=S_, region_strengths=SR,
                   locked_controls=locked_info, standin=standin, arm_ids=ids, scored_names=scored,
                   finite={k: finite[k] for k in scored}, random_surface_seeds={j: nets[j][1] for j in (1, 2)},
                   shared=sinfo, ramp_shape=shape, chunk_seconds=runner.chunk_seconds,
                   seconds=dict(seconds, by_kind=runner.seconds, total=time.perf_counter() - t_start),
                   vmhwm_mb=S.vmhwm_mb(), **MC.machine_record())
    MC.save_and_lock(d, actors, payload, official)
    MC.log(f"no_stall/seed {seed}: locked; locked controls {[a.name for a in locked]}; stand-in {standin['arm']}; "
           f"{time.perf_counter() - t_start:.0f}s; VmHWM {S.vmhwm_mb():.0f} MB")


def w0_repro(MC, seed, world, table, d, base, steps, chunk, t_start):
    """Scratch only, full length: the ramp program with every height set to 0, as wide as the phase's own first ramp
    program (stalling world: the w0 check alone; no-stall world: the w0 check and the six ramp arms side by side),
    must reproduce check 2's saved 'none' actor within 1e-5 (bit for bit expected). No scored arm is trained."""
    row = table[world][seed]
    ctx = MC.context(seed, world)
    pg = ctx.prepared(world)
    critic = ctx.critic(world, "gap")[0]
    clone = ctx.clone(world)[0]
    none = ctx.none(world)
    cfg = MC.H.ProgramConfig(physics=pg.physics, next_command="stall_aware")
    runner = MC.RampRunner(pg, clone, critic, ctx.learn, seed, cfg=cfg, shared=None, steps=steps, chunk=chunk,
                           t_lo=row["t_lo"], t_up=row["t_up"])
    names = [S.W0] if world == "stall" else [S.W0] + list(S.RAMP_ARMS)
    trained = runner.run([MC.ramp_arm(n, 0.0, 0.0 if n == S.W0 else row["arms"][n]["k"]) for n in names], MC.log)
    rows = {n: dict(max_abs_diff=MC.H.max_param_diff(trained[n]["params"], none),
                    sha256_equal=MC.L.tree_sha256(trained[n]["params"]) == MC.L.tree_sha256(none)) for n in names}
    rec = dict(base, kind="step2-w0-reproduction", world=world, n_arms=len(names), arms=rows,
               passed=bool(all(r["max_abs_diff"] <= MC.W0_TOL for r in rows.values())), tol=MC.W0_TOL,
               seconds=dict(by_kind=runner.seconds, total=time.perf_counter() - t_start), vmhwm_mb=S.vmhwm_mb(),
               **MC.machine_record())
    S.write_json_once(Path(d) / "w0_repro.json", rec, False)
    MC.log(f"w0 reproduction {world}/seed {seed}: {rows}; passed {rec['passed']}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--world", choices=S.WORLDS, required=True)
    ap.add_argument("--out-root", default=None, help="scratch mode: a folder outside the repository")
    ap.add_argument("--steps", type=int, default=None, help="scratch only: fewer actor steps (smoke test)")
    ap.add_argument("--chunk", type=int, default=None, help="scratch only: chunk length (divides --steps)")
    ap.add_argument("--w0-repro", action="store_true", help="scratch only: the reproduction check (full length)")
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    official = a.out_root is None
    if official and (a.steps is not None or a.chunk is not None or a.w0_repro):
        raise SystemExit("refused: --steps, --chunk and --w0-repro are for scratch runs only (--out-root)")
    spec, spec_sha = S.load_prespec()
    man, man_sha = S.verify_manifest()
    seed = S.check_seed(a.seed)
    cpu = S.require_one_cpu("an actor training")
    S.check_environment(spec)
    pr = spec["protocol"]
    steps = int(a.steps) if a.steps is not None else int(pr["actor_steps"])
    chunk = int(a.chunk) if a.chunk is not None else min(int(pr["chunk"]), steps)
    if steps <= 0 or chunk <= 0 or steps % chunk or steps > int(pr["actor_steps"]):
        raise SystemExit(f"refused: {steps} steps in chunks of {chunk} (steps must be a multiple of the chunk, at most "
                         f"{pr['actor_steps']})")
    full = steps == int(pr["actor_steps"])
    if a.w0_repro and not full:
        raise SystemExit("refused: the reproduction check runs at full length only")
    root = S.OFFICIAL_OUT if official else Path(a.out_root)
    d = S.guard_out((root / "w0_repro" / f"seed{seed}" / a.world) if a.w0_repro else S.seed_dir(root, seed, a.world),
                    official)
    for n in ("lock.json", "stop.json", "actors.msgpack", "w0_repro.json"):
        if (d / n).exists():
            raise SystemExit(f"refused: {d}/{n} exists (write-once); an actors.msgpack without a lock.json means an "
                             f"interrupted run: inspect it by hand")
    step1 = S.load_step1(S.STEP1_RESULTS, spec["inputs"]["step1_results"])
    table = S.arm_table(step1, S.SEEDS, spec["arms"]["rule"])
    if not S.table_matches(table, spec["arm_table_expected"]):
        raise SystemExit("refused: the arm table read from step 1's results differs from prespec.json's")
    import machinery as MC  # noqa: E402  (heavy: the gate, the room harness, version 6's code, JAX)
    if MC.SPEC_SHA != spec_sha:
        raise SystemExit("refused: prespec.json changed while starting")
    base = dict(label=S.LABEL, mode="official" if official else "scratch", seed=seed, world=a.world,
                prespec_sha256=spec_sha, code_manifest_sha256=man_sha,
                step1_results_sha256=spec["inputs"]["step1_results"]["sha256"],
                v6_prereg_sha256=spec["inputs"]["v6_prereg_sha256"], steps=steps, chunk=chunk, full_length=full,
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    MC.log(f"{a.world}/seed {seed}: {'official' if official else 'scratch'} -> {d}; {steps} steps in chunks of "
           f"{chunk}; CPU {cpu}")
    if a.w0_repro:
        return w0_repro(MC, seed, a.world, table, d, base, steps, chunk, t_start)
    if a.world == "stall":
        return train_stall(MC, seed, table, d, base, official, steps, chunk, full, t_start)
    return train_no_stall(MC, seed, table, d, base, official, steps, chunk, full, t_start, root, spec_sha, man_sha)


if __name__ == "__main__":
    main()
