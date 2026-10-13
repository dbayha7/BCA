"""EXPLORATORY: one seed of the Stage-1 data-only stand-ins (plan section 4.2). No BCA. Outside the gate.

Phases (each refuses unless spec.json exists, every gate file still has the SHA-256 the spec recorded, the room harness
imports from its pinned bytes, every stand-in script and its copy match runs/shift_explore/standins/
code_manifest.json, which must name the current spec, and runs/shift_explore/standins/approval.json holds the user's
choice of mark forms for this spec and this manifest, written by approve.py before the first launch):
  train            stalling world. Refuses unless the room test's pooled look passed its room, recovery and
                   keeps-improvement marks. Reads the perfect band's strength from the room's lock.json of the seed
                   (strengths only). Fits every stand-in's data side, runs check (e) on every penalty (reported),
                   the W = 0 check (b) (stop on failure), every 4-rung ladder, the blind matching, then lock.json with
                   every actor's SHA-256. No return.
  eval             verifies lock.json, then exact returns of the locked actors -> values.json (once).
  no_stall_train   the no-stall world, every stand-in at its locked coefficient (not re-matched), fitted on that
                   world's own gapped log; check (b) against check 2's no-stall actor; lock.
  no_stall_eval    exact returns in the no-stall world -> values.json (once).
  marks            seeds 0-4: plan 4.2's mark "captures the room" (three conditions) for every stand-in, the no-stall
                   loss and the selectivity -> marks.json; captured_by lists the forms the approval chose to carry
                   the mark ('approve F1-F3': re-referenced; 'decline F1-F3': the plan's literal forms). Reads the
                   room's verified values.json of every seed (the
                   perfect band, the matched even pull, no caution) only here.

Usage (from /home/user/bca; normally through launch.sh, which pins the process and logs to
runs/shift_explore/standins/logs/):
  python run_standins.py --seed 0 --phase train
  python run_standins.py --phase marks
"""
import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import standins as S  # noqa: E402

import numpy as np  # noqa: E402

H, A, L, TC = S.H, S.A, S.L, S.TC
SITUATIONS = TC.SITUATIONS
log = S.log


def seed_dir(rep, sub=None, out=None):
    d = Path(out or S.OUT) / f"seed{int(rep)}"
    return d / sub if sub else d


class Stop(SystemExit):
    pass


def stop(path, rep, reason, spec_sha, man_sha, **fields):
    S.write_json(path, dict(label=S.LABEL, kind="standins-stop", rep=int(rep), stopped=True, reason=reason,
                            spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                            written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields), out=S.OUT)
    log(f"STOP: {reason}")
    raise Stop(2)


def refuse_if_stopped(d, what):
    if (d / "stop.json").exists():
        raise SystemExit(f"refused: {d} has stopped (stop.json); {what} may not run")


def save_and_lock(d, actors, payload):
    names = sorted(actors)
    file_sha = S.save_tree(d / "lock_actors.msgpack", {k: actors[k] for k in names})
    S.guard_out(d / "lock_actors.msgpack.sha256").write_text(f"{file_sha}  lock_actors.msgpack\n", encoding="utf8")
    payload = dict(payload, names=names, actor_sha256={k: L.tree_sha256(actors[k]) for k in names},
                   actors_file="lock_actors.msgpack", actors_file_sha256=file_sha,
                   written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   about=("the lock: every stand-in trained and matched in strength (or, in the no-stall world, trained "
                          "at its locked coefficient), written before any return, points or score of any trained arm "
                          "is computed"))
    digest = S.write_json(d / "lock.json", payload)
    log(f"wrote {d / 'lock.json'} ({digest[:16]})")
    return digest


def verify_lock(d, spec_sha, appr=None):
    lock = S.read_json(d / "lock.json")
    if lock is None:
        raise SystemExit(f"refused: no lock at {d / 'lock.json'}: no return may be computed before it")
    if lock["spec_sha256"] != spec_sha or not str(lock.get("label", "")).startswith("EXPLORATORY"):
        raise SystemExit("refused: the lock names another spec or is not labelled")
    if appr is not None and (lock.get("approval") or {}).get("sha256") != appr["sha256"]:
        raise SystemExit("refused: the lock was written under another approval.json")
    acts = L.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    bad = [k for k in lock["names"] if L.tree_sha256(acts[k]) != lock["actor_sha256"][k]]
    if bad:
        raise SystemExit(f"refused: actors differ from the lock: {bad}")
    return lock, acts, S.file_sha256(d / "lock.json")


def arms_record(trained):
    return {k: dict(arm=v["arm"], finite=v["finite"], chunk_metrics=v["chunk_metrics"], metric_names=S.METRIC_NAMES,
                    classifier_refits=v.get("refits"), classifier_refit_names=S.CLS_DIAG_NAMES if v.get("refits")
                    else None, classifier_refit_check_e=v.get("refit_check_e")) for k, v in trained.items()}


def refit_check_summary(trained):
    """Per classifier arm: check (e) passed at each refit (reported, spec change C2)."""
    return {k: [r["passed"] for r in v["refit_check_e"]] for k, v in trained.items() if v.get("refit_check_e")}


def strengths_of(trained, none, prepared):
    St, SR = {}, {}
    for name, v in trained.items():
        if v["finite"]:
            St[name] = L.strength(v["params"], none, prepared.eval_obs)
            SR[name] = H.region_strength(v["params"], none, prepared)
        else:
            St[name], SR[name] = float("nan"), float("nan")
    return St, SR


def start():
    """Every run phase starts here: the spec, the gate files, the room import, the code manifest (standins.begin)
    and the user's approval of this spec and manifest (spec change C3)."""
    spec, spec_sha, man_sha = S.begin()
    appr = S.require_approval(spec_sha, man_sha)
    log(f"approval {appr['sha256'][:16]} ({appr['mode']} mode): mark forms {appr['mark_forms']} "
        f"{appr['mark_families']}")
    return spec, spec_sha, man_sha, appr


def approval_record(appr):
    return dict(sha256=appr["sha256"], mode=appr["mode"], mark_forms=appr["mark_forms"],
                mark_families=appr["mark_families"])


def common(ctx, spec_sha, man_sha, rep, appr):
    return dict(label=S.LABEL, rep=int(rep), spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                code_sha256=S.code_hashes(), room_import=S.ROOM_IMPORT, gate_prereg_sha256=ctx.prereg.digest,
                alpha=S.ALPHA, steps=S.STEPS, chunk=S.CHUNK, approval=approval_record(appr))


def check_b(trained, none):
    diff = S.max_param_diff(trained["w0_check"]["params"], none)
    return dict(max_abs_diff=diff, tol=S.W0_TOL, passed=bool(trained["w0_check"]["finite"] and diff <= S.W0_TOL),
                bitwise_equal=L.tree_sha256(trained["w0_check"]["params"]) == L.tree_sha256(none))


def world_setup(ctx, world, pg, critic, clone, none, rep):
    """The data-side fits, the classifier fits with the no-caution and the cloned actor, check (e) on every penalty
    (reported; not a stop condition, spec 'check_e')."""
    shared, sinfo, cmds = S.build_shared(ctx, world, pg, critic, clone, none, rep, log)
    sinfo["density_levels"] = S.density_levels(pg, shared, none, rep)
    cls_none = S.classifier_fit_for(none, shared)
    cls_clone = S.classifier_fit_for(clone, shared)
    sinfo["classifier_fit_no_caution_actor"] = dict(l_star=cls_none[1], diag=cls_none[2])
    sinfo["classifier_fit_cloned_actor"] = dict(l_star=cls_clone[1], diag=cls_clone[2])
    return shared, sinfo, cmds, cls_none, cls_clone


def run_check_e(pg, critic, shared, none, rep, scale, cls_none, cls_clone):
    out = {}
    for f in S.FAMILIES:
        net, lstar = (cls_none[0], cls_none[1]) if f == "classifier_ratio" else (None, None)
        out[f] = S.check_e(f, pg, critic, shared, none, rep, scale, net, lstar)
        log(f"check (e) {f}: passed {out[f]['passed']}, fall {out[f]['fall_share_80_to_81']:.3f}, "
            f"mean W 80/81 {out[f]['mean_W']['0.8']:.4f}/{out[f]['mean_W']['0.81']:.4f}, "
            f"slope {out[f]['mean_dW_dpush']['0.805']['mean']:.3f}, best>80 {out[f]['best_past80_share_knob1']:.3f} "
            f"(Q1 alone {out[f]['q1_alone_best_past80_share']:.3f})")
    out["classifier_ratio[fitted with the cloned actor]"] = S.check_e(
        "classifier_ratio", pg, critic, shared, none, rep, scale, cls_clone[0], cls_clone[1])
    out["note"] = ("reported for every stand-in, not a stop condition (spec 'check_e'); the classifier ratio is read "
                   "with its classifier fitted with the no-caution actor, and also with the cloned actor (the "
                   "classifier every arm starts from)")
    return out


# ------------------------------------------------------------------------------------------------ stalling world

def phase_train(rep):
    t0 = time.perf_counter()
    spec, spec_sha, man_sha, appr = start()
    d = seed_dir(rep)
    if (d / "lock.json").exists():
        raise SystemExit(f"refused: seed {rep} is already locked")
    refuse_if_stopped(d, "train")
    gate = S.room_gate()
    ctx = H.SeedContext(rep)
    pg = ctx.prepared()
    critic, crec = ctx.critic()
    clone, clrec = ctx.clone()
    h2acts, h2rec, h2d = ctx.h2()
    none = h2acts["gapped"]
    if L.tree_sha256(clone) != L.tree_sha256(h2acts["clone"]):
        stop(d / "stop.json", rep, "the saved cloned actor is not check 2's start actor", spec_sha, man_sha)
    target, rlock = S.room_lock_target(rep, none)
    log(f"seed {rep}: loaded ({ctx.load_seconds:.0f}s), {pg.n} training rows; perfect band strength from the room "
        f"lock: {target}")
    if target is None:
        stop(d / "stop.json", rep, "the room's perfect band has no positive strength to match", spec_sha, man_sha,
             room_lock=rlock)
    shared, sinfo, cmds, cls_none, cls_clone = world_setup(ctx, "stall", pg, critic, clone, none, rep)
    room_cfg = H.ProgramConfig()
    room_shared, _ = H.build_shared(pg, critic, rep, cfg=room_cfg, need_harm=False)
    scales = S.slope_scales(pg, critic, shared, room_shared, cmds["a_none"], rep, cls_none, room_cfg)
    log(f"ladder scales {scales['scale']}")
    ce = run_check_e(pg, critic, shared, none, rep, scales["scale"], cls_none, cls_clone)
    runner = S.StandinRunner(pg, clone, critic, ctx.learn, rep, shared=shared,
                             refit_check=dict(none=none, scale=scales["scale"]))
    if runner.rows_sha256 != rlock["rows_sha256"]:
        stop(d / "stop.json", rep, "the batch-row stream differs from check 2's (the room lock's check (a))", spec_sha,
             man_sha)
    trained = runner.run([S.StandinArm("w0_check", "plain", (0.0,))], log)
    cb = check_b(trained, none)
    log(f"check (b): {cb}")
    if not cb["passed"]:
        stop(d / "stop.json", rep, "check (b) failed: the stand-in program with W = 0 differs from check 2's actor",
             spec_sha, man_sha, check_b=cb, check_e=ce)
    trained.update(runner.run([S.family_arm(f, k, scales["scale"]) for f in S.FAMILIES for k in S.LADDER], log))
    matches = S.match_standins(runner, trained, none, pg, target, scales["scale"], S.FAMILIES)
    St, SR = strengths_of(trained, none, pg)
    actors = {k: v["params"] for k, v in trained.items()}
    actors["none"], actors["clone"] = none, clone
    scored = ["none", "clone"] + [m["arm"] for m in matches.values() if m["arm"]]
    reported_unmatched = {f: m["strongest_finite_rung"] for f, m in matches.items()
                          if not m["matched"] and m["strongest_finite_rung"]}
    scored += sorted(set(reported_unmatched.values()) - set(scored))
    payload = dict(common(ctx, spec_sha, man_sha, rep, appr), kind="standins-lock", world="stall", room_gate=gate,
                   room_lock=rlock, target_strength=target, checks=dict(b=cb, e=ce), shared=sinfo,
                   ladder_scale=scales, arms=arms_record(trained), strengths=St, region_strengths=SR,
                   matches=matches, matched_names={f: m["arm"] for f, m in matches.items()},
                   unmatched_strongest_rung=reported_unmatched, scored_names=scored,
                   locked_coefficients={f: (trained[m["arm"]]["arm"]["coef"] if m["arm"] else None)
                                        for f, m in matches.items()},
                   gate_artifacts=dict(critic_file_sha256=crec["result"]["file_sha256"],
                                       clone_file_sha256=clrec["result"]["file_sha256"],
                                       check2_actors_file_sha256=h2rec["actors_sha256"]),
                   classifier_refit_check_e_passed=refit_check_summary(trained),
                   seconds=dict(total=time.perf_counter() - t0, by_kind=runner.seconds, cpu_by_kind=runner.cpu_seconds,
                                classifier_refit_cpu=runner.refit_cpu_seconds,
                                classifier_refit_check_e_cpu=runner.refit_check_cpu_seconds))
    save_and_lock(d, actors, payload)


def evaluate_locked(ctx, lock, acts, prepared, world):
    an = ctx.anchors(world)
    ev, shares = {}, {}
    for name in lock["scored_names"]:
        finite = name in ("none", "clone") or lock["arms"][name]["finite"]
        t = time.perf_counter()
        ev[name] = A._evaluate(acts[name], prepared, an, ctx.cfg, failed=not finite)
        shares[name] = S.thruster_shares(acts[name], prepared, prepared.physics) if finite else None
        log(f"evaluated {name} ({time.perf_counter() - t:.0f}s)")
    return ev, shares


def values_of(ev, shares, scored):
    values = {}
    for s in SITUATIONS:
        p = {k: (None if ev.get(k) is None else ev[k][s]["points"]) for k in scored}
        values[s] = dict(points=p, arm_minus_none={k: (None if v is None or p.get("none") is None else v - p["none"])
                                                   for k, v in p.items()},
                         unrecorded_throttle_share={k: None if ev[k] is None else ev[k][s]["unrecorded_throttle_share"]
                                                    for k in ev},
                         capped_points={k: None if ev[k] is None else ev[k][s]["capped_points"] for k in ev},
                         past_80_share_per_thruster={k: None if shares[k] is None else shares[k][s] for k in shares})
    return values


def phase_eval(rep, world="stall"):
    spec, spec_sha, man_sha, appr = start()
    d = seed_dir(rep) if world == "stall" else seed_dir(rep, "no_stall")
    refuse_if_stopped(seed_dir(rep), "eval")
    refuse_if_stopped(d, "eval")
    if (d / "values.json").exists():
        raise SystemExit(f"refused: {d / 'values.json'} exists; an evaluation is computed once")
    lock, acts, lock_sha = verify_lock(d, spec_sha, appr)
    ctx = H.SeedContext(rep, worlds=("stall",) if world == "stall" else ("stall", "no_stall"))
    pg = ctx.prepared(world)
    t0 = time.perf_counter()
    ev, shares = evaluate_locked(ctx, lock, acts, pg, world)
    rec = dict(label=S.LABEL, kind="standins-values" if world == "stall" else "standins-values-no-stall",
               rep=int(rep), world=world, spec_sha256=spec_sha, code_manifest_sha256=man_sha, lock_sha256=lock_sha,
               approval=approval_record(appr),
               matched_names=lock.get("matched_names"), family_names=lock.get("family_names"),
               unmatched_strongest_rung=lock.get("unmatched_strongest_rung"), evaluations=ev,
               situations=values_of(ev, shares, lock["scored_names"]), seconds=time.perf_counter() - t0,
               about="exact returns of the locked actors, computed after the lock was verified")
    S.write_json(d / "values.json", rec)
    log(f"wrote {d / 'values.json'}")


# ------------------------------------------------------------------------------------------------ no-stall world

def phase_no_stall_train(rep):
    t0 = time.perf_counter()
    spec, spec_sha, man_sha, appr = start()
    d = seed_dir(rep, "no_stall")
    if (d / "lock.json").exists():
        raise SystemExit(f"refused: seed {rep} no-stall is already locked")
    refuse_if_stopped(seed_dir(rep), "no_stall_train")
    refuse_if_stopped(d, "no_stall_train")
    stall_lock, _, stall_lock_sha = verify_lock(seed_dir(rep), spec_sha, appr)
    ctx = H.SeedContext(rep, worlds=("stall", "no_stall"))
    pt = ctx.prepared("no_stall")
    critic, crec = ctx.critic("no_stall", "gap")
    clone, clrec = ctx.clone("no_stall")
    h2acts, h2rec, _ = ctx.h2()
    none = h2acts["no_stall_gapped"]
    if L.tree_sha256(clone) != L.tree_sha256(h2acts["no_stall_clone"]):
        stop(d / "stop.json", rep, "the saved no-stall cloned actor is not check 2's no-stall start actor", spec_sha,
             man_sha)
    shared, sinfo, cmds, cls_none, cls_clone = world_setup(ctx, "no_stall", pt, critic, clone, none, rep)
    scale_stall = stall_lock["ladder_scale"]["scale"]
    ce = run_check_e(pt, critic, shared, none, rep, scale_stall, cls_none, cls_clone)
    ce["note_no_stall"] = "reported only: in the no-stall world the missing commands help"
    runner = S.StandinRunner(pt, clone, critic, ctx.learn, rep, shared=shared,
                             refit_check=dict(none=none, scale=scale_stall))
    trained = runner.run([S.StandinArm("w0_check", "plain", (0.0,))], log)
    cb = check_b(trained, none)
    log(f"check (b), no-stall: {cb}")
    if not cb["passed"]:
        stop(d / "stop.json", rep, "check (b) failed in the no-stall world", spec_sha, man_sha, check_b=cb)
    arms, source = [], {}
    for f in S.FAMILIES:
        name = stall_lock["matched_names"].get(f)
        tag = "matched"
        if not name:
            name, tag = stall_lock["unmatched_strongest_rung"].get(f), "unmatched: the stalling world's strongest rung"
        if not name:
            source[f] = dict(arm=None, why="no finite arm in the stalling world")
            continue
        coef = stall_lock["arms"][name]["arm"]["coef"]
        arms.append(S.locked_arm(f, coef, name=f"{f}@locked"))
        source[f] = dict(stall_arm=name, coef=coef, status=tag)
    trained.update(runner.run(arms, log))
    St, SR = strengths_of(trained, none, pt)
    actors = {k: v["params"] for k, v in trained.items()}
    actors["none"], actors["clone"] = none, clone
    scored = ["none", "clone"] + [a.name for a in arms]
    payload = dict(common(ctx, spec_sha, man_sha, rep, appr), kind="standins-lock-no-stall", world="no_stall",
                   stall_lock_sha256=stall_lock_sha, locked_from_stall=source, checks=dict(b=cb, e_reported=ce),
                   shared=sinfo, arms=arms_record(trained), strengths=St, region_strengths=SR, scored_names=scored,
                   family_names={f: f"{f}@locked" for f in source if source[f].get("coef") is not None},
                   gate_artifacts=dict(critic_file_sha256=crec["result"]["file_sha256"],
                                       clone_file_sha256=clrec["result"]["file_sha256"],
                                       check2_actors_file_sha256=h2rec["actors_sha256"]),
                   classifier_refit_check_e_passed=refit_check_summary(trained),
                   seconds=dict(total=time.perf_counter() - t0, by_kind=runner.seconds, cpu_by_kind=runner.cpu_seconds,
                                classifier_refit_cpu=runner.refit_cpu_seconds,
                                classifier_refit_check_e_cpu=runner.refit_check_cpu_seconds))
    save_and_lock(d, actors, payload)


# ------------------------------------------------------------------------------------------------ marks

def room_values(rep):
    """The room's stalling-world values.json of a seed, accepted only as the room's own futility and pooled looks
    accept it (sidecar, label, kind room-values, seed, the pinned spec, and lock_sha256 = that seed's lock.json)."""
    dr = S.ROOM_OUT / f"seed{int(rep)}"
    v = S.read_json(dr / "values.json")
    if v is None:
        return None, None
    if not str(v.get("label", "")).startswith("EXPLORATORY") or v.get("kind") != "room-values" \
            or v.get("rep") != int(rep) or v.get("spec_sha256") != S.ROOM_PINNED["spec_sha256"]:
        raise SystemExit(f"refused: room seed {rep} values.json is not the room harness's labelled output")
    if v.get("lock_sha256") != S.file_sha256(dr / "lock.json"):
        raise SystemExit(f"refused: room seed {rep} values.json does not name that seed's lock.json")
    return v, S.file_sha256(dr / "values.json")


def load_own(rep, spec_sha, out=None, appr=None):
    """This component's values of a seed: (stall values or None, no-stall values or None, stop record or None)."""
    d = seed_dir(rep, out=out)
    st = S.read_json(d / "stop.json") or S.read_json(d / "no_stall" / "stop.json")
    vs, vn = S.read_json(d / "values.json"), S.read_json(d / "no_stall" / "values.json")
    for v, kind in ((vs, "standins-values"), (vn, "standins-values-no-stall")):
        if v is None:
            continue
        if not str(v.get("label", "")).startswith("EXPLORATORY") or v.get("kind") != kind or v.get("rep") != int(rep) \
                or v.get("spec_sha256") != spec_sha:
            raise SystemExit(f"refused: seed {rep} {kind} is not this component's labelled output for this spec")
        lock_dir = d if kind == "standins-values" else d / "no_stall"
        if v.get("lock_sha256") != S.file_sha256(lock_dir / "lock.json"):
            raise SystemExit(f"refused: seed {rep} {kind} does not name its lock.json")
        if appr is not None and (v.get("approval") or {}).get("sha256") != appr["sha256"]:
            raise SystemExit(f"refused: seed {rep} {kind} was written under another approval.json")
    return vs, vn, st


def _mean(xs):
    return None if not xs or any(x is None for x in xs) else float(np.mean(xs))


def compute_marks(per_seed, seeds=S.SEEDS, mark_families=S.MAIN_FAMILIES):
    """Plan 4.2's mark and reports from per-seed inputs (pure; tested on synthetic numbers by test_marks.py).
    per_seed[rep] = dict(own_stall={family: {situation: points}} or None for a missing seed, own_none_stall
    {situation: points}, own_no_stall {family: {situation: points}}, own_none_no_stall, room_pb, room_none,
    room_even {situation: points} (None when missing)). Every mark needs every seed; a missing value fails it.
    mark_families: the forms that carry the mark (the approval's choice); the others are labelled reported."""
    out = {}
    for f in S.FAMILIES:
        g, pb, rec, ev, ns_loss, sel, ns_by_s = [], [], [], [], [], [], {s: [] for s in SITUATIONS}
        why = []
        for rep in seeds:
            x = per_seed.get(rep) or {}
            st = (x.get("own_stall") or {}).get(f)
            none_s = x.get("own_none_stall")
            rpb, rnone, reven = x.get("room_pb"), x.get("room_none"), x.get("room_even")
            nst = (x.get("own_no_stall") or {}).get(f)
            none_n = x.get("own_none_no_stall")

            def d(a, b, s):
                return None if a is None or b is None or a.get(s) is None or b.get(s) is None else a[s] - b[s]

            gi = d(st, none_s, "strong_current")
            g.append(gi)
            pb.append(d(rpb, rnone, "strong_current"))
            rec.append(d(st, none_s, "recorded"))
            ev.append(d(st, reven, "strong_current"))
            li = d(none_n, nst, "strong_current")
            ns_loss.append(li)
            sel.append(None if gi is None or li is None else gi - li)
            for s in SITUATIONS:
                ns_by_s[s].append(d(none_n, nst, s))
            if st is None:
                why.append(f"seed {rep}: no matched stalling-world arm or no values")
            if rpb is None or reven is None:
                why.append(f"seed {rep}: the room's perfect band or matched even pull is missing")
        mg, mpb, mrec, mev = _mean(g), _mean(pb), _mean(rec), _mean(ev)
        c1 = bool(mg is not None and mpb is not None and mg >= 0.5 * mpb)
        c2 = bool(mrec is not None and mrec >= -3.0)
        c3 = bool(mev is not None and mev >= 15.0)
        out[f] = dict(
            role=("stand-in (mark)" if f in mark_families else
                  ("the plan's literal form (reported, same mark computed)" if f in S.REPORTED_FAMILIES else
                   "re-referenced form (reported, same mark computed)")),
            gain_strong_current=dict(values=g, mean=mg), perfect_band_gain_strong_current=dict(values=pb, mean=mpb),
            c1_gain_at_least_half_of_perfect_band=c1,
            recorded_minus_none=dict(values=rec, mean=mrec), c2_recorded_at_least_none_minus_3=c2,
            minus_matched_even_pull_strong_current=dict(values=ev, mean=mev), c3_beats_even_pull_by_15=c3,
            captures_the_room=bool(c1 and c2 and c3), missing=why,
            no_stall_loss_strong_current=dict(values=ns_loss, mean=_mean(ns_loss)),
            no_stall_loss_by_situation={s: dict(values=v, mean=_mean(v)) for s, v in ns_by_s.items()},
            selectivity=dict(values=sel, mean=_mean(sel)))
    hs = out["range_hinge"]["selectivity"]["values"]
    for f in S.FAMILIES:
        v = out[f]["selectivity"]["values"]
        dv = [None if a is None or b is None else a - b for a, b in zip(v, hs)]
        out[f]["selectivity_minus_range_hinge"] = dict(values=dv, mean=_mean(dv))
    return out


def phase_marks():
    spec, spec_sha, man_sha, appr = start()
    mark_families = tuple(appr["mark_families"])
    path = S.OUT / "marks.json"
    if path.exists():
        raise SystemExit(f"refused: {path} exists")
    gate = S.room_gate()
    per_seed, files, consistency = {}, {}, {}
    for rep in S.SEEDS:
        vs, vn, st = load_own(rep, spec_sha, appr=appr)
        if vs is None and vn is None and st is None:
            raise SystemExit(f"refused: seed {rep} has neither values nor a stop record yet")
        if st is None and (vs is None or vn is None):
            raise SystemExit(f"refused: seed {rep} is missing its stalling or no-stall values")
        rv, rv_sha = room_values(rep)
        x = {}
        if vs is not None:
            pts = {s: vs["situations"][s]["points"] for s in SITUATIONS}
            x["own_none_stall"] = {s: pts[s]["none"] for s in SITUATIONS}
            x["own_stall"] = {f: {s: pts[s][n] for s in SITUATIONS} for f, n in vs["matched_names"].items() if n}
        if vn is not None:
            pts = {s: vn["situations"][s]["points"] for s in SITUATIONS}
            x["own_none_no_stall"] = {s: pts[s]["none"] for s in SITUATIONS}
            x["own_no_stall"] = {f: {s: pts[s].get(n) for s in SITUATIONS} for f, n in vn["family_names"].items()}
        if rv is not None:
            rp = {s: rv["situations"][s]["points"] for s in SITUATIONS}
            x["room_pb"] = {s: rp[s].get("perfect_band") for s in SITUATIONS}
            x["room_none"] = {s: rp[s].get("none") for s in SITUATIONS}
            en = rv["matched_names"].get("even_pull")
            x["room_even"] = {s: rp[s].get(en) for s in SITUATIONS} if en else None
            if vs is not None:
                consistency[rep] = {s: (None if x["room_none"][s] is None else x["own_none_stall"][s] - x["room_none"][s])
                                    for s in SITUATIONS}
        per_seed[rep] = x
        files[rep] = dict(room_values_sha256=rv_sha, stop=None if st is None else st.get("reason"))
    marks = compute_marks(per_seed, seeds=S.SEEDS, mark_families=mark_families)
    reported = [f for f in S.FAMILIES if f not in mark_families]
    out = dict(label=S.LABEL, kind="standins-marks", spec_sha256=spec_sha, code_manifest_sha256=man_sha,
               approval=approval_record(appr), mark_forms=appr["mark_forms"], mark_families=list(mark_families),
               reported_families=reported, room_gate=gate, seeds=list(S.SEEDS), files=files, marks=marks,
               none_points_own_minus_room=consistency,
               captured_by=[f for f in mark_families if marks[f]["captures_the_room"]],
               captured_by_reported_forms=[f for f in reported if marks[f]["captures_the_room"]],
               marks_text=spec["marks"])
    S.write_json(path, out)
    log(f"captures the room: {out['captured_by']} (reported forms: {out['captured_by_reported_forms']})")


PHASES = dict(train=phase_train, eval=lambda r: phase_eval(r, "stall"), no_stall_train=phase_no_stall_train,
              no_stall_eval=lambda r: phase_eval(r, "no_stall"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--phase", required=True, choices=sorted(PHASES) + ["marks"])
    a = ap.parse_args()
    if a.phase == "marks":
        return phase_marks()
    if a.seed not in S.SEEDS:
        raise SystemExit(f"--seed must be one of {S.SEEDS}")
    PHASES[a.phase](a.seed)


if __name__ == "__main__":
    main()
