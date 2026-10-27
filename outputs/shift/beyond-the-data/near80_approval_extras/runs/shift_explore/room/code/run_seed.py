"""EXPLORATORY: one seed of the Stage-1 room test (plan sections 4.0, 4.1, 7). No BCA. Outside the gate.

Phases (each refuses unless spec.json exists, every gate file still has the SHA-256 the spec recorded, and every
room-test script and its copy match runs/shift_explore/room/code_manifest.json, which must name the current spec):
  train              stalling world: check (a) (the gate's own code reproduces check 2's gapped-critic actor by hash),
                     check (e) (the main oracle's penalty rises past the knee; critic surface only), every arm
                     trained, check (b) (the new program with W = 0 within 1e-5 of that actor), strengths, blind
                     matching of the gating controls, then lock.json with every actor's SHA-256. No return. A failed
                     check writes stop.json and stops the seed.
  eval               verifies lock.json, then exact returns (acceptance._evaluate, full 1,000-start banks, four
                     situations) of the locked actors -> values.json. Refuses when values.json exists or the seed
                     stopped.
  no_stall_train / no_stall_eval     the no-stall comparison world at the stalling world's locked coefficients.
  replication_train / replication_eval   check 3's critic: check (e), the perfect band and the matched even pull.
  futility           seeds 0 and 1: the futility rule -> futility.json (each seed needs a verified values.json or
                     stop.json; a missing difference counts as less than 5 points).
  pooled             seeds 0-4: the room, recovery and keeps-improvement marks -> pooled.json (needs futility.json
                     saying continue and a verified values.json or stop.json for every seed).

Usage (from /home/user/bca; normally through launch.sh, which pins the process and logs to
runs/shift_explore/room/logs/):
  python run_seed.py --seed 0 --phase train
  python run_seed.py --phase futility
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402
from experiments.signal import glq_match as M  # noqa: E402

import jax  # noqa: E402

SITUATIONS = TC.SITUATIONS
FIXED_SCORED = ("perfect_band", "perfect_band_half", "perfect_band_centred", "underrating_half", "perfect_band_literal",
                "overrating_half_literal", "state_shuffled_band", "long_run_harm", "range_hinge")


def seed_dir(rep, sub=None):
    d = H.OUT / f"seed{int(rep)}"
    return d / sub if sub else d


class Stop(SystemExit):
    pass


def stop(path, rep, reason, spec_sha, man_sha, **fields):
    H.write_json(path, dict(label=H.LABEL, kind="room-stop", rep=int(rep), stopped=True, reason=reason,
                            spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                            written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields))
    H.log(f"STOP: {reason}")
    raise Stop(2)


def begin():
    """Every phase starts here: the spec, the gate files and the code manifest. Returns (spec, spec_sha, man_sha)."""
    spec, spec_sha = H.load_spec()
    H.check_gate_files(spec)
    _, man_sha = H.check_code(spec_sha)
    return spec, spec_sha, man_sha


def refuse_if_done(d, what):
    if (d / "stop.json").exists():
        raise SystemExit(f"refused: {d} has stopped (stop.json); {what} may not run")


# ------------------------------------------------------------------------------------------------ check (a)

def reproduce_h2(ctx, world="stall", log_fn=H.log):
    """acceptance.h2_seed's swap() for one world, through the gate's unchanged code (learners.ActorOnly.run,
    unweighted, the critics side by side), compared with check 2's saved actors by parameter hash."""
    t0 = time.perf_counter()
    acts, rec, d = ctx.h2()
    pg = ctx.prepared(world)
    if world == "stall":
        named = [("gapped", ctx.critic("stall", "gap")[0]), ("full", ctx.critic("stall", "none")[0]),
                 ("random", ctx.critic("stall", "random")[0])]
        clone_key = "clone"
    else:
        named = [("no_stall_gapped", ctx.critic("no_stall", "gap")[0]),
                 ("no_stall_full", ctx.critic("no_stall", "none")[0])]
        clone_key = "no_stall_clone"
    clone, _ = ctx.clone(world)
    ao = L.ActorOnly(pg, clone, ctx.learn, ctx.rep, alpha=H.ALPHA)
    stacked, info = ao.run([L.Arm(n, critic=i) for i, (n, _) in enumerate(named)], [c for _, c in named],
                           weighted=False, log=log_fn)
    out = dict(clone_equals_check2_clone=L.tree_sha256(clone) == L.tree_sha256(acts[clone_key]),
               rows_sha256=info["rows_sha256"], arms={})
    for i, (n, _) in enumerate(named):
        p = ao.arm_params(stacked, i)
        out["arms"][n] = dict(reproduced_sha256=L.tree_sha256(p), saved_sha256=L.tree_sha256(acts[n]),
                              equal=L.tree_sha256(p) == L.tree_sha256(acts[n]),
                              max_abs_diff=H.max_param_diff(p, acts[n]))
    key = "gapped" if world == "stall" else "no_stall_gapped"
    out["passed"] = bool(out["arms"][key]["equal"] and out["clone_equals_check2_clone"])
    out["seconds"] = time.perf_counter() - t0
    out["h2_values_sha256"] = H.file_sha256(d / "values.json")
    out["h2_actors_file_sha256"] = rec["actors_sha256"]
    return out, acts


# ------------------------------------------------------------------------------------------------ matching

def strengths_of(trained, none, prepared):
    S, SR = {}, {}
    for name, v in trained.items():
        if v["finite"]:
            S[name] = L.strength(v["params"], none, prepared.eval_obs)
            SR[name] = H.region_strength(v["params"], none, prepared)
        else:
            S[name], SR[name] = float("nan"), float("nan")
    return S, SR


def match_gating(runner, trained, none, prepared, target, scale, families, log_fn=H.log):
    """Blind matching of every gating family to the target strength: the ladder (knob 0 = no caution, S = 0), the
    declared extension, then glq_match.match_parallel. Trains proposals as arms; strengths only."""
    by_knob = {f: {} for f in families}
    for name, v in trained.items():
        f = v["arm"].get("family")
        if f in by_knob:
            by_knob[f][float(v["arm"]["knob"])] = name

    def S_of(name):
        return float(L.strength(trained[name]["params"], none, prepared.eval_obs)) if trained[name]["finite"] \
            else float("nan")

    extensions = {f: [] for f in families}
    for _ in range(H.MAX_EXTENSIONS):
        need = []
        for f in families:
            knobs = sorted(by_knob[f])
            S = [S_of(by_knob[f][k]) for k in knobs]
            fin = [s for s in S if np.isfinite(s)]
            if fin and target > max(fin):
                need.append(H.family_arm(f, knobs[-1] * H.EXTEND_FACTOR[f], scale))
        if not need:
            break
        log_fn(f"ladder extension: {[a.name for a in need]}")
        new = runner.run(need, log_fn)
        trained.update(new)
        for a in need:
            by_knob[a.family][a.knob] = a.name
            extensions[a.family].append(a.knob)
    ladders = {}
    for f in families:
        knobs = sorted(by_knob[f])
        ladders[f] = ([0.0] + knobs, [0.0] + [S_of(by_knob[f][k]) for k in knobs])

    def evaluate(proposals):
        arms = []
        for f, k in proposals:
            if float(k) in by_knob[f]:
                continue
            arms.append(H.family_arm(f, float(k), scale))
        if arms:
            new = runner.run(arms, log_fn)
            trained.update(new)
            for a in arms:
                by_knob[a.family][a.knob] = a.name
        return [S_of(by_knob[f][float(k)]) for f, k in proposals]

    states, rounds = M.match_parallel(ladders, {"perfect_band": float(target)}, evaluate, M.MATCH_TOL,
                                      M.MATCH_ROUNDS, log=log_fn)
    out = {}
    for f in families:
        rec = M.public(states[(f, "perfect_band")])
        rec["rounds"] = rounds
        rec["extensions"] = extensions[f]
        rec["glq_match_sha256"] = M.source_sha256()
        name = None if rec["knob"] is None else by_knob[f].get(float(rec["knob"]))
        out[f] = dict(record=rec, arm=name, matched=name is not None)
    return out


# ------------------------------------------------------------------------------------------------ locks

def save_and_lock(d, lock_name, actors, payload):
    """Save every actor (actors.msgpack) and write the lock with every actor's parameter SHA-256."""
    names = sorted(actors)
    file_sha = H.save_tree(d / f"{lock_name}_actors.msgpack", {k: actors[k] for k in names})
    payload = dict(payload, names=names, actor_sha256={k: L.tree_sha256(actors[k]) for k in names},
                   actors_file=f"{lock_name}_actors.msgpack", actors_file_sha256=file_sha,
                   written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   about=("the lock: every actor trained and every gating control matched in strength, written before "
                          "any return, points or score of any trained arm is computed"))
    digest = H.write_json(d / f"{lock_name}.json", payload)
    H.log(f"wrote {d / (lock_name + '.json')} ({digest[:16]})")
    return digest


def verify_lock(d, lock_name, spec_sha):
    """The lock must exist, match its sidecar, name the current spec, and its actors file and every actor must match
    their recorded SHA-256. Returns (lock, actors)."""
    lock = H.read_json(d / f"{lock_name}.json")
    if lock is None:
        raise SystemExit(f"refused: no lock at {d / (lock_name + '.json')}: no return may be computed before it")
    if lock["spec_sha256"] != spec_sha:
        raise SystemExit("refused: the lock names another spec")
    acts = L.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    bad = [k for k in lock["names"] if L.tree_sha256(acts[k]) != lock["actor_sha256"][k]]
    if bad:
        raise SystemExit(f"refused: actors differ from the lock: {bad}")
    return lock, acts, H.file_sha256(d / f"{lock_name}.json")


def arms_record(trained):
    return {k: dict(arm=v["arm"], finite=v["finite"], chunk_metrics=v["chunk_metrics"],
                    metric_names=H.METRIC_NAMES) for k, v in trained.items()}


def common_lock(ctx, spec_sha, man_sha, rep):
    return dict(label=H.LABEL, rep=int(rep), spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                code_sha256=H.code_hashes(), gate_prereg_sha256=ctx.prereg.digest, alpha=H.ALPHA, steps=H.STEPS,
                chunk=H.CHUNK)


# ------------------------------------------------------------------------------------------------ stalling world

def phase_train(rep):
    t0 = time.perf_counter()
    spec, spec_sha, man_sha = begin()
    d = seed_dir(rep)
    if (d / "lock.json").exists():
        raise SystemExit(f"refused: seed {rep} is already locked")
    refuse_if_done(d, "train")
    ctx = H.SeedContext(rep)
    pg = ctx.prepared()
    critic, crec = ctx.critic()
    clone, clrec = ctx.clone()
    H.log(f"seed {rep}: loaded ({ctx.load_seconds:.0f}s), {pg.n} training rows")
    repro, h2acts = reproduce_h2(ctx)
    H.log(f"check (a): {repro['arms']['gapped']}")
    if not repro["passed"]:
        stop(d / "stop.json", rep, "check (a) failed: the gate's code did not reproduce check 2's actor", spec_sha,
             man_sha, check_a=repro)
    none = h2acts["gapped"]
    cfg = H.ProgramConfig()
    shared, sinfo = H.build_shared(pg, critic, rep, cfg=cfg, log_fn=H.log)
    check_e = H.oracle_shape(critic, shared, cfg, pg, none, rep)
    H.log(f"check (e): passed {check_e['passed']}; main form {summary_e(check_e)}")
    if not check_e["passed"]:
        stop(d / "stop.json", rep, "check (e) failed: the main oracle's penalty does not rise past the knee", spec_sha,
             man_sha, check_a=repro, check_e=check_e)
    nets = {j: H.random_surface_params(rep, j) for j in (1, 2)}
    idx = H.slope_sample(pg, rep)
    a_clone = L.actor_rows(clone, np.asarray(pg.train.obs)[idx])
    g = dict(band=H.slope_rms("band", cfg, critic, shared, idx, a_clone),
             push_less=H.slope_rms("push_less", cfg, critic, shared, idx, a_clone),
             random_1=H.slope_rms("random", cfg, critic, shared, idx, a_clone, net=nets[1][0]),
             random_2=H.slope_rms("random", cfg, critic, shared, idx, a_clone, net=nets[2][0]))
    scale = {"even_pull": 1.0, "push_less": g["band"] / g["push_less"], "random_1": g["band"] / g["random_1"],
             "random_2": g["band"] / g["random_2"]}
    H.log(f"slopes {g}; scales {scale}")
    runner = H.RoomRunner(pg, clone, critic, ctx.learn, rep, cfg=cfg, shared=shared,
                          random_nets={j: nets[j][0] for j in (1, 2)})
    if runner.rows_sha256 != repro["rows_sha256"]:
        stop(d / "stop.json", rep, "the batch-row stream differs from check 2's", spec_sha, man_sha)
    trained = {}
    fixed = {a.name: a for a in H.fixed_arms()}
    trained.update(runner.run([fixed["w0_check"]] + [H.family_arm("even_pull", nu, scale) for nu in H.EVEN_LADDER],
                              H.log))
    w0diff = H.max_param_diff(trained["w0_check"]["params"], none)
    check_b = dict(max_abs_diff=w0diff, tol=H.W0_TOL, passed=bool(trained["w0_check"]["finite"] and w0diff <= H.W0_TOL),
                   bitwise_equal=L.tree_sha256(trained["w0_check"]["params"]) == L.tree_sha256(none))
    H.log(f"check (b): {check_b}")
    if not check_b["passed"]:
        stop(d / "stop.json", rep, "check (b) failed: the new program with W = 0 differs from check 2's actor",
             spec_sha, man_sha, check_a=repro, check_e=check_e, check_b=check_b)
    rest = [a for n, a in fixed.items() if n != "w0_check"]
    rest += [H.family_arm(f, k, scale) for f in ("push_less", "random_1", "random_2") for k in H.SLOPE_LADDER]
    trained.update(runner.run(rest, H.log))
    target = float(L.strength(trained["perfect_band"]["params"], none, pg.eval_obs)) \
        if trained["perfect_band"]["finite"] else float("nan")
    H.log(f"perfect band strength {target:.5f}")
    if np.isfinite(target) and target > 0:
        matches = match_gating(runner, trained, none, pg, target, scale, H.GATING)
    else:
        matches = {f: dict(record=None, arm=None, matched=False,
                           reason="the perfect band failed or has no positive strength") for f in H.GATING}
    S, SR = strengths_of(trained, none, pg)
    actors = {k: v["params"] for k, v in trained.items()}
    actors["none"], actors["clone"] = none, clone
    scored = ["none", "clone"] + list(FIXED_SCORED) + [m["arm"] for m in matches.values() if m["arm"]]
    payload = dict(common_lock(ctx, spec_sha, man_sha, rep), kind="room-lock", world="stall",
                   checks=dict(a=repro, b=check_b, e=check_e), arms=arms_record(trained), strengths=S,
                   region_strengths=SR,
                   target_strength=target, matches=matches,
                   matched_names={f: m["arm"] for f, m in matches.items()}, scored_names=scored,
                   slope_rms=g, coefficient_per_knob=scale,
                   random_surface_seeds={j: nets[j][1] for j in (1, 2)}, shared=sinfo,
                   gate_artifacts=dict(critic_record_sha256=H.file_sha256(A.critic_dir(
                       ctx.prereg, ctx.variant, "stall", "gap", "stall_aware", False, rep) / "record.json"),
                       critic_file_sha256=crec["result"]["file_sha256"],
                       clone_file_sha256=clrec["result"]["file_sha256"]),
                   seconds=dict(total=time.perf_counter() - t0, by_kind=runner.seconds))
    save_and_lock(d, "lock", actors, payload)


def thruster_shares(params, prepared, physics):
    """Reported: the share of steps past 80% per thruster, from one exact rollout per situation's full bank."""
    pol = L.mlp_policy(params, prepared)
    out = {}
    for s in SITUATIONS:
        run = TR.rollout(pol, TC.bank_starts(s, TC.BANK_SIZE), physics, record=True)
        out[s] = (np.abs(run.commands) > physics.knee).mean(axis=(0, 1)).tolist()
    return out


def evaluate_locked(ctx, lock, acts, prepared, world):
    an = ctx.anchors(world)
    ev, shares = {}, {}
    for name in lock["scored_names"]:
        finite = name in ("none", "clone") or lock["arms"][name]["finite"]
        t = time.perf_counter()
        ev[name] = A._evaluate(acts[name], prepared, an, ctx.cfg, failed=not finite)
        shares[name] = thruster_shares(acts[name], prepared, prepared.physics) if finite else None
        H.log(f"evaluated {name} ({time.perf_counter() - t:.0f}s)")
    return ev, shares


def pts(ev, name, s):
    v = ev.get(name) if name else None
    return None if v is None else v[s]["points"]


def diff(a, b):
    return None if a is None or b is None else a - b


def phase_eval(rep):
    spec, spec_sha, man_sha = begin()
    d = seed_dir(rep)
    refuse_if_done(d, "eval")
    if (d / "values.json").exists():
        raise SystemExit(f"refused: {d / 'values.json'} exists; an evaluation is computed once")
    lock, acts, lock_sha = verify_lock(d, "lock", spec_sha)
    ctx = H.SeedContext(rep)
    pg = ctx.prepared()
    t0 = time.perf_counter()
    ev, shares = evaluate_locked(ctx, lock, acts, pg, "stall")
    m = lock["matched_names"]
    values = {}
    for s in SITUATIONS:
        p = {k: pts(ev, k, s) for k in lock["scored_names"]}
        pb = p.get("perfect_band")
        values[s] = dict(
            points=p,
            perfect_band_minus_control={f: diff(pb, pts(ev, m.get(f), s)) for f in H.GATING},
            perfect_band_minus_none=diff(pb, p.get("none")),
            perfect_band_minus_clone=diff(pb, p.get("clone")),
            arm_minus_none={k: diff(v, p.get("none")) for k, v in p.items()},
            unrecorded_throttle_share={k: None if ev[k] is None else ev[k][s]["unrecorded_throttle_share"] for k in ev},
            capped_points={k: None if ev[k] is None else ev[k][s]["capped_points"] for k in ev},
            past_80_share_per_thruster={k: None if shares[k] is None else shares[k][s] for k in shares})
    rec = dict(label=H.LABEL, kind="room-values", rep=int(rep), world="stall", spec_sha256=spec_sha,
               code_manifest_sha256=man_sha, lock_sha256=lock_sha, matched_names=m, evaluations=ev,
               situations=values,
               seconds=time.perf_counter() - t0,
               about="exact returns of the locked actors, computed after the lock was verified")
    H.write_json(d / "values.json", rec)
    sc = values["strong_current"]
    H.log(f"seed {rep} strong current: perfect band {sc['points'].get('perfect_band')}, none "
          f"{sc['points'].get('none')}, minus controls {sc['perfect_band_minus_control']}")


# ------------------------------------------------------------------------------------------------ no-stall world

def phase_no_stall_train(rep):
    t0 = time.perf_counter()
    spec, spec_sha, man_sha = begin()
    d = seed_dir(rep, "no_stall")
    if (d / "lock.json").exists():
        raise SystemExit(f"refused: seed {rep} no-stall is already locked")
    refuse_if_done(d, "no_stall_train")
    stall_lock, _, stall_lock_sha = verify_lock(seed_dir(rep), "lock", spec_sha)
    ctx = H.SeedContext(rep, worlds=("stall", "no_stall"))
    pt = ctx.prepared("no_stall")
    critic, crec = ctx.critic("no_stall", "gap")
    clone, clrec = ctx.clone("no_stall")
    repro, h2acts = reproduce_h2(ctx, "no_stall")
    H.log(f"check (a), no-stall: {repro['arms']['no_stall_gapped']}")
    if not repro["passed"]:
        stop(d / "stop.json", rep, "check (a) failed in the no-stall world", spec_sha, man_sha, check_a=repro)
    none = h2acts["no_stall_gapped"]
    cfg_s = H.ProgramConfig(physics=pt.physics, next_command="stall_aware")
    cfg_r = H.ProgramConfig(physics=pt.physics, next_command="rising_thrust")
    shared, sinfo = H.build_shared(pt, critic, rep, cfg=cfg_s, need_harm=False, log_fn=H.log)
    shared_r, rinfo = H.with_logged_delta(shared, critic, cfg_r, pt)
    sinfo["rising_thrust_next_command"] = rinfo
    shape = dict(stall_aware=H.oracle_shape(critic, shared, cfg_s, pt, none, rep),
                 rising_thrust=H.oracle_shape(critic, shared_r, cfg_r, pt, none, rep),
                 note="reported only: in the no-stall world the missing commands help")
    runner = H.RoomRunner(pt, clone, critic, ctx.learn, rep, cfg=cfg_s, shared=shared)
    runner_r = H.RoomRunner(pt, clone, critic, ctx.learn, rep, cfg=cfg_r, shared=shared_r)
    em = stall_lock["matches"]["even_pull"]
    nu = None if not em["matched"] else float(stall_lock["arms"][em["arm"]]["arm"]["knob"])
    arms = [H.RoomArm("w0_check", "plain", (0.0,), 1.0)]
    if nu is not None:
        arms.append(H.RoomArm("even_pull_locked", "plain", (0.0,), 1.0 + nu, family="even_pull", knob=nu))
    arms += [H.RoomArm("perfect_band", "band", H.MAIN_BAND), H.RoomArm("range_hinge", "hinge", (1.0,))]
    trained = runner.run(arms, H.log)
    w0diff = H.max_param_diff(trained["w0_check"]["params"], none)
    check_b = dict(max_abs_diff=w0diff, tol=H.W0_TOL, passed=bool(trained["w0_check"]["finite"] and w0diff <= H.W0_TOL))
    H.log(f"check (b), no-stall: {check_b}")
    if not check_b["passed"]:
        stop(d / "stop.json", rep, "check (b) failed in the no-stall world", spec_sha, man_sha, check_a=repro,
             check_b=check_b)
    trained.update(runner_r.run([H.RoomArm("perfect_band_rising", "band", H.MAIN_BAND)], H.log))
    S, SR = strengths_of(trained, none, pt)
    actors = {k: v["params"] for k, v in trained.items()}
    actors["none"], actors["clone"] = none, clone
    scored = ["none", "clone", "perfect_band", "perfect_band_rising", "range_hinge"] + \
        (["even_pull_locked"] if nu is not None else [])
    payload = dict(common_lock(ctx, spec_sha, man_sha, rep), kind="room-lock-no-stall", world="no_stall",
                   stall_lock_sha256=stall_lock_sha, locked_nu=nu, checks=dict(a=repro, b=check_b, e_reported=shape),
                   arms=arms_record(trained), strengths=S, region_strengths=SR, scored_names=scored, shared=sinfo,
                   gate_artifacts=dict(critic_file_sha256=crec["result"]["file_sha256"],
                                       clone_file_sha256=clrec["result"]["file_sha256"]),
                   seconds=dict(total=time.perf_counter() - t0, by_kind=runner.seconds))
    save_and_lock(d, "lock", actors, payload)


def phase_no_stall_eval(rep):
    spec, spec_sha, man_sha = begin()
    d = seed_dir(rep, "no_stall")
    refuse_if_done(d, "no_stall_eval")
    if (d / "values.json").exists():
        raise SystemExit(f"refused: {d / 'values.json'} exists; an evaluation is computed once")
    lock, acts, lock_sha = verify_lock(d, "lock", spec_sha)
    ctx = H.SeedContext(rep, worlds=("stall", "no_stall"))
    pt = ctx.prepared("no_stall")
    t0 = time.perf_counter()
    ev, shares = evaluate_locked(ctx, lock, acts, pt, "no_stall")
    values = {}
    for s in SITUATIONS:
        p = {k: pts(ev, k, s) for k in lock["scored_names"]}
        values[s] = dict(points=p, arm_minus_none={k: diff(v, p.get("none")) for k, v in p.items()},
                         unrecorded_throttle_share={k: None if ev[k] is None else ev[k][s]["unrecorded_throttle_share"]
                                                    for k in ev},
                         past_80_share_per_thruster={k: None if shares[k] is None else shares[k][s] for k in shares})
    H.write_json(d / "values.json", dict(label=H.LABEL, kind="room-values-no-stall", rep=int(rep), world="no_stall",
                                         spec_sha256=spec_sha, code_manifest_sha256=man_sha, lock_sha256=lock_sha,
                                         evaluations=ev,
                                         situations=values, seconds=time.perf_counter() - t0,
                                         about="the price of caution where the missing commands help"))


# ------------------------------------------------------------------------------------------------ replication

def phase_replication_train(rep):
    t0 = time.perf_counter()
    spec, spec_sha, man_sha = begin()
    d = seed_dir(rep, "replication")
    if (d / "lock.json").exists():
        raise SystemExit(f"refused: seed {rep} replication is already locked")
    refuse_if_done(d, "replication_train")
    ctx = H.SeedContext(rep)
    pg = ctx.prepared()
    host, hrec = ctx.host()
    critic = host.critic.params
    h3acts, blind, vals, d3 = ctx.h3()
    none = h3acts["none"]
    clone, clrec = ctx.clone()
    cfg = H.ProgramConfig(next_command="host_actor", bc_form="weighted")
    shared, sinfo = H.build_shared(pg, critic, rep, cfg=cfg, host_actor=host.actor_target.params, need_harm=False,
                                   log_fn=H.log)
    check_e = H.oracle_shape(critic, shared, cfg, pg, none, rep)
    H.log(f"check (e), replication: passed {check_e['passed']}; main form {summary_e(check_e)}")
    if not check_e["passed"]:
        stop(d / "stop.json", rep, "check (e) failed on check 3's critic", spec_sha, man_sha, check_e=check_e)
    runner = H.RoomRunner(pg, clone, critic, ctx.learn, rep, cfg=cfg, shared=shared)
    if runner.rows_sha256 != blind["rows_sha256"]:
        stop(d / "stop.json", rep, "the batch-row stream differs from check 3's", spec_sha, man_sha)
    scale = {"even_pull": 1.0}
    trained = runner.run([H.RoomArm("w0_check", "plain", (0.0,), 1.0)] +
                         [H.family_arm("even_pull", nu, scale) for nu in H.EVEN_LADDER], H.log)
    w0diff = H.max_param_diff(trained["w0_check"]["params"], none)
    check_b = dict(max_abs_diff=w0diff, tol=H.W0_TOL, passed=bool(trained["w0_check"]["finite"] and w0diff <= H.W0_TOL),
                   against="check 3's saved 'none' actor")
    even_vs_check3 = {f"even_x{1 + nu:g}": H.max_param_diff(trained[H.family_arm("even_pull", nu, scale).name]["params"],
                                                         h3acts[f"even_x{1 + nu:g}"])
                      for nu in H.EVEN_LADDER if f"even_x{1 + nu:g}" in h3acts}
    H.log(f"check (b), replication: {check_b}; even arms against check 3's: {even_vs_check3}")
    if not check_b["passed"]:
        stop(d / "stop.json", rep, "check (b) failed on check 3's critic", spec_sha, man_sha, check_b=check_b,
             check_e=check_e)
    trained.update(runner.run([H.RoomArm("perfect_band", "band", H.MAIN_BAND)], H.log))
    target = float(L.strength(trained["perfect_band"]["params"], none, pg.eval_obs)) \
        if trained["perfect_band"]["finite"] else float("nan")
    if np.isfinite(target) and target > 0:
        matches = match_gating(runner, trained, none, pg, target, scale, ("even_pull",))
    else:
        matches = {"even_pull": dict(record=None, arm=None, matched=False, reason="no positive perfect-band strength")}
    S, SR = strengths_of(trained, none, pg)
    actors = {k: v["params"] for k, v in trained.items()}
    actors["none"] = none
    scored = ["perfect_band"] + ([matches["even_pull"]["arm"]] if matches["even_pull"]["arm"] else [])
    payload = dict(common_lock(ctx, spec_sha, man_sha, rep), kind="room-lock-replication", world="stall",
                   critic="check 3's frozen co-trained plain TD3+BC critic (gapped log, alpha 5)",
                   checks=dict(b=check_b, e=check_e, even_arms_against_check3_max_abs_diff=even_vs_check3),
                   arms=arms_record(trained), strengths=S, region_strengths=SR, target_strength=target,
                   matches=matches, matched_names={f: m["arm"] for f, m in matches.items()}, scored_names=scored,
                   check3_strengths=blind["strengths"], shared=sinfo,
                   gate_artifacts=dict(host_record_sha256=H.file_sha256(A.host_dir(
                       ctx.prereg, ctx.variant, "stall", "gap", H.ALPHA, "plain", rep) / "record.json"),
                       check3_lock_sha256=H.file_sha256(d3 / "room_blind.json"),
                       check3_values_sha256=H.file_sha256(d3 / "values.json")),
                   seconds=dict(total=time.perf_counter() - t0, by_kind=runner.seconds))
    save_and_lock(d, "lock", actors, payload)


def phase_replication_eval(rep):
    spec, spec_sha, man_sha = begin()
    d = seed_dir(rep, "replication")
    refuse_if_done(d, "replication_eval")
    if (d / "values.json").exists():
        raise SystemExit(f"refused: {d / 'values.json'} exists; an evaluation is computed once")
    lock, acts, lock_sha = verify_lock(d, "lock", spec_sha)
    ctx = H.SeedContext(rep)
    pg = ctx.prepared()
    _, blind, vals, _ = ctx.h3()
    t0 = time.perf_counter()
    ev, shares = evaluate_locked(ctx, lock, acts, pg, "stall")
    reused = ("none", "harm_placed", "strength_matched_even", "cap2")
    old = {k: vals["evaluations"].get(k) for k in reused}
    m = lock["matched_names"].get("even_pull")
    values = {}
    for s in SITUATIONS:
        p = {k: pts(ev, k, s) for k in lock["scored_names"]}
        p.update({f"check3_{k}": None if old[k] is None else old[k][s]["points"] for k in reused})
        values[s] = dict(points=p, perfect_band_minus_matched_even=diff(p.get("perfect_band"), p.get(m)),
                         perfect_band_minus_none=diff(p.get("perfect_band"), p.get("check3_none")),
                         matched_even_minus_none=diff(p.get(m), p.get("check3_none")),
                         past_80_share_per_thruster={k: None if shares[k] is None else shares[k][s] for k in shares})
    H.write_json(d / "values.json", dict(label=H.LABEL, kind="room-values-replication", rep=int(rep),
                                         spec_sha256=spec_sha, code_manifest_sha256=man_sha, lock_sha256=lock_sha,
                                         evaluations=ev,
                                         check3_reused=sorted(reused), situations=values,
                                         seconds=time.perf_counter() - t0))


# ------------------------------------------------------------------------------------------------ pooled looks

def summary_e(check_e):
    m = check_e["forms"]["main (perfect_band)"]
    return dict(fall_share_80_to_81=round(m["fall_share_80_to_81"], 4), mean_W_80=round(m["mean_W"]["0.8"], 4),
                mean_W_81=round(m["mean_W"]["0.81"], 4), slope_805=round(m["mean_dW_dpush"]["0.805"]["mean"], 3),
                best_past80=round(m["best_past80_share"], 4),
                q1_alone_best_past80=round(check_e["q1_alone_best_past80_share"], 4))


def load_result(rep, spec_sha):
    """A seed's stalling-world result for the futility and pooled looks: ('values', values), ('stopped', stop) or
    ('missing', None). values.json is accepted only when it matches its sidecar, is labelled EXPLORATORY, is of kind
    room-values for this seed, names the current spec, and its lock_sha256 is the SHA-256 of that seed's lock.json
    (itself sidecar-verified and naming the current spec). stop.json is accepted only when it matches its sidecar,
    is labelled EXPLORATORY and names the current spec. Anything else refuses."""
    d = seed_dir(rep)
    v = H.read_json(d / "values.json")       # ValueError when the sidecar does not match
    st = H.read_json(d / "stop.json")
    if v is not None and st is not None:
        raise SystemExit(f"refused: seed {rep} has both values.json and stop.json")
    if v is not None:
        if not str(v.get("label", "")).startswith("EXPLORATORY") or v.get("kind") != "room-values" \
                or v.get("rep") != int(rep):
            raise SystemExit(f"refused: seed {rep} values.json is not this harness's labelled output")
        if v.get("spec_sha256") != spec_sha:
            raise SystemExit(f"refused: seed {rep} values.json names another spec")
        lock = H.read_json(d / "lock.json")
        if lock is None or lock.get("spec_sha256") != spec_sha:
            raise SystemExit(f"refused: seed {rep} has no lock naming the current spec")
        if v.get("lock_sha256") != H.file_sha256(d / "lock.json"):
            raise SystemExit(f"refused: seed {rep} values.json does not name that seed's lock.json")
        return "values", v
    if st is not None:
        if not str(st.get("label", "")).startswith("EXPLORATORY") or st.get("kind") != "room-stop" \
                or st.get("spec_sha256") != spec_sha:
            raise SystemExit(f"refused: seed {rep} stop.json is not this harness's labelled output for this spec")
        return "stopped", st
    return "missing", None


def futility_difference(status, rec):
    """(difference or None, why it is missing)."""
    if status == "stopped":
        return None, "seed stopped: " + str(rec.get("reason"))
    sc = rec["situations"]["strong_current"]
    dlt = sc["perfect_band_minus_control"]["even_pull"]
    if dlt is not None:
        return float(dlt), None
    if not rec["matched_names"].get("even_pull"):
        return None, "the even pull has no matched arm"
    if sc["points"].get("perfect_band") is None:
        return None, "the perfect band is not finite"
    return None, "the difference is missing"


def phase_futility():
    spec, spec_sha, man_sha = begin()
    path = H.OUT / "futility.json"
    if path.exists():
        raise SystemExit(f"refused: {path} exists; the futility look is taken once")
    out = dict(label=H.LABEL, kind="room-futility", spec_sha256=spec_sha, code_manifest_sha256=man_sha,
               rule=spec["marks"]["futility"], seeds={})
    below = []
    for rep in H.FUTILITY_SEEDS:
        status, rec = load_result(rep, spec_sha)
        if status == "missing":
            raise SystemExit(f"refused: seed {rep} has neither values.json nor stop.json yet")
        dlt, why = futility_difference(status, rec)
        f = "values.json" if status == "values" else "stop.json"
        out["seeds"][rep] = dict(status=status, perfect_band_minus_matched_even_pull=dlt, missing_reason=why,
                                 counts_as_below_5=bool(dlt is None or dlt < 5.0),
                                 file=f, file_sha256=H.file_sha256(seed_dir(rep) / f))
        below.append(dlt is None or dlt < 5.0)
    out["stop"] = bool(all(below))
    out["decision"] = ("STOP: go to plan section 4.5 (seeds 2-4 are not run)" if out["stop"] else
                       "continue: run seeds 2-4 and the no-stall arms")
    H.write_json(path, out)
    H.log(out["decision"], out["seeds"])


def _mean(xs):
    return None if any(x is None for x in xs) or not xs else float(np.mean(xs))


def phase_pooled():
    spec, spec_sha, man_sha = begin()
    path = H.OUT / "pooled.json"
    if path.exists():
        raise SystemExit(f"refused: {path} exists")
    fut = H.read_json(H.OUT / "futility.json")
    if fut is None or fut.get("spec_sha256") != spec_sha or fut.get("stop") is not False:
        raise SystemExit("refused: the pooled look needs futility.json for this spec saying continue")
    results = {rep: load_result(rep, spec_sha) for rep in H.SEEDS}
    missing = [rep for rep, (status, _) in results.items() if status == "missing"]
    if missing:
        raise SystemExit(f"refused: seeds {missing} have neither values.json nor stop.json yet")
    vals = {rep: (rec if status == "values" else None) for rep, (status, rec) in results.items()}
    complete = all(v is not None for v in vals.values())
    n = len(H.SEEDS)

    def col(situation, key, sub=None):
        out = []
        for rep in H.SEEDS:
            v = vals[rep]
            if v is None:
                out.append(None)
                continue
            x = v["situations"][situation][key]
            out.append(x.get(sub) if sub is not None else x)
        return out

    room = {}
    for f in H.GATING:
        dl = col("strong_current", "perfect_band_minus_control", f)
        mean = _mean(dl)
        positive = sum(1 for x in dl if x is not None and x > 0)
        room[f] = dict(values=dl, mean=mean, positive_seeds=positive,
                       passed=bool(complete and mean is not None and mean >= 20.0 and positive >= 4))
    rec_d = col("strong_current", "perfect_band_minus_none")
    recovery = dict(values=rec_d, mean=_mean(rec_d), passed=bool(complete and _mean(rec_d) is not None
                                                                   and _mean(rec_d) >= 49.5))
    r_none = col("recorded", "perfect_band_minus_none")
    r_clone = col("recorded", "perfect_band_minus_clone")
    keeps = dict(recorded_vs_none=dict(values=r_none, mean=_mean(r_none)),
                 recorded_vs_clone=dict(values=r_clone, mean=_mean(r_clone)))
    ok = (complete and _mean(r_none) is not None and _mean(r_clone) is not None and _mean(r_none) >= -2.0
          and _mean(r_clone) >= 25.0)
    for s in ("big_waves", "big_kicks"):
        x = col(s, "perfect_band_minus_none")
        keeps[f"{s}_vs_none"] = dict(values=x, mean=_mean(x))
        ok = ok and _mean(x) is not None and _mean(x) >= -3.0
    keeps["passed"] = bool(ok)
    out = dict(label=H.LABEL, kind="room-pooled", spec_sha256=spec_sha, code_manifest_sha256=man_sha,
               futility_sha256=H.file_sha256(H.OUT / "futility.json"), seeds=list(H.SEEDS), complete=complete,
               seed_status={rep: status for rep, (status, _) in results.items()},
               n=n, room=dict(controls=room, passed=bool(all(r["passed"] for r in room.values()))),
               recovery=recovery, keeps_improvement=keeps, marks_text=spec["marks"])
    H.write_json(path, out)
    H.log(f"room {out['room']['passed']}, recovery {recovery['passed']}, keeps {keeps['passed']}")


PHASES = dict(train=phase_train, eval=phase_eval, no_stall_train=phase_no_stall_train,
              no_stall_eval=phase_no_stall_eval, replication_train=phase_replication_train,
              replication_eval=phase_replication_eval)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--phase", required=True, choices=sorted(PHASES) + ["futility", "pooled"])
    a = ap.parse_args()
    if a.phase == "futility":
        return phase_futility()
    if a.phase == "pooled":
        return phase_pooled()
    if a.seed not in H.SEEDS:
        raise SystemExit(f"--seed must be one of {H.SEEDS}")
    PHASES[a.phase](a.seed)


if __name__ == "__main__":
    main()
