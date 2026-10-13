"""EXPLORATORY: the stand-ins' seed-0 checks, run after spec.json and the code manifest were written. NO RETURN,
POINTS OR SCORE is computed here (no rollout return, no evaluation, no strength). Nothing trained here is saved or
scored. The room test's outcome files are not read (no lock, values, futility or pooled file).

  f  each penalty (the 8 families) is finite and nonnegative, with a finite gradient, on every training row at the
     recorded, cloned, no-caution and uniform-random commands, in both worlds;
  e  check (e) on each penalty in the stalling world (the no-stall world's reported), before any training; the ladder
     scales (slope normalisation) that check (e)'s knob-1 coefficient uses;
  d  a 400-step run of every arm (every family at ladder knob 1, and the W = 0 arm) in both worlds, in 200-step chunks
     so that the classifier is refitted twice; finite losses; the W = 0 arm against the room harness's own RoomRunner
     over the same 400 steps (parameter hash); timing per 1,000 steps per kind (CPU seconds of a second, compiled
     run), the data-side fits' timing and the projected cost of the full run.

Usage: python checks_seed0.py --world stall      then  --world no_stall      then  --summarise
       (--dev [--world W] [f e d]: without the spec, writes nothing)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import standins as S  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402

H, A, L, TC = S.H, S.A, S.L, S.TC
REP = 0
log = S.log


def cpu():
    return time.process_time()


class World:
    """One world's seed-0 objects and data-side fits (timed)."""

    def __init__(self, ctx, world):
        t, c = time.perf_counter(), cpu()
        self.world = world
        self.pg = ctx.prepared(world)
        self.critic, _ = ctx.critic(world, "gap")
        self.clone, _ = ctx.clone(world)
        acts, _, _ = ctx.h2()
        self.none = acts["gapped" if world == "stall" else "no_stall_gapped"]
        self.shared, self.info, self.cmds, self.cls_none, self.cls_clone = None, None, None, None, None
        c1 = cpu()
        self.shared, self.info, self.cmds = S.build_shared(ctx, world, self.pg, self.critic, self.clone, self.none, REP,
                                                           log)
        c2 = cpu()
        self.cls_none = S.classifier_fit_for(self.none, self.shared)
        c3 = cpu()
        self.cls_clone = S.classifier_fit_for(self.clone, self.shared)
        c4 = cpu()
        self.cls_clone2 = S.classifier_fit_for(self.clone, self.shared)
        c5 = cpu()
        self.timing = dict(build_shared_cpu=c2 - c1, classifier_fit_first_cpu=c3 - c2, classifier_fit_cpu=c5 - c4,
                           surface_cpu=self.info["distance"]["surface"]["cpu_seconds"],
                           knn_cpu=self.info["distance"]["surface"]["knn_cpu_seconds"], total_cpu=cpu() - c,
                           total_wall=time.perf_counter() - t)


def check_f(w):
    rng = np.random.default_rng([20261015, REP, 87])
    a_rows = dict(recorded=np.asarray(w.pg.actions, np.float32), cloned=np.asarray(w.cmds["a_clone"]),
                  no_caution=np.asarray(w.cmds["a_none"]),
                  uniform=rng.uniform(-1, 1, size=(w.pg.n, TC.DA)).astype(np.float32))
    out = {}
    for f in S.FAMILIES:
        net, lstar = (w.cls_none[0], w.cls_none[1]) if f == "classifier_ratio" else (None, None)
        out[f] = S.finite_check(f, w.pg, w.critic, w.shared, a_rows, net, lstar)
    cl = S.finite_check("classifier_ratio", w.pg, w.critic, w.shared, a_rows, w.cls_clone[0], w.cls_clone[1])
    out["classifier_ratio[fitted with the cloned actor]"] = cl
    out["passed"] = bool(all(v["passed"] for v in out.values() if isinstance(v, dict)))
    out["at_recorded_commands_zero"] = {f: out[f]["recorded"]["max"] for f in ("density", "distance", "two_head")}
    return out


def check_e_world(w, ctx):
    cfg = H.ProgramConfig() if w.world == "stall" else H.ProgramConfig(physics=w.pg.physics)
    room_shared, _ = H.build_shared(w.pg, w.critic, REP, cfg=cfg, need_harm=False)
    scales = S.slope_scales(w.pg, w.critic, w.shared, room_shared, w.cmds["a_none"], REP, w.cls_none, cfg)
    out = dict(ladder_scale=scales, forms={})
    for f in S.FAMILIES:
        net, lstar = (w.cls_none[0], w.cls_none[1]) if f == "classifier_ratio" else (None, None)
        out["forms"][f] = S.check_e(f, w.pg, w.critic, w.shared, w.none, REP, scales["scale"], net, lstar)
    out["forms"]["classifier_ratio[fitted with the cloned actor]"] = S.check_e(
        "classifier_ratio", w.pg, w.critic, w.shared, w.none, REP, scales["scale"], w.cls_clone[0], w.cls_clone[1])
    out["passed"] = {f: out["forms"][f]["passed"] for f in out["forms"]}
    return out, scales


def run_400(w, ctx, scale):
    """Every family at knob 1 and the W = 0 arm, 400 steps in 200-step chunks, twice (the second run is timed)."""
    res = dict(steps=S.CHECK_STEPS, chunk=200, arms={})
    runner = S.StandinRunner(w.pg, w.clone, w.critic, ctx.learn, REP, shared=w.shared, steps=S.CHECK_STEPS, chunk=200)
    arms = [S.StandinArm("w0_check", "plain", (0.0,))] + [S.family_arm(f, 1.0, scale) for f in S.FAMILIES]
    first = runner.run(arms)
    cpu0, ref0 = dict(runner.cpu_seconds), runner.refit_cpu_seconds
    second = runner.run(arms)
    per_kind = {}
    for k in S.KINDS:
        n = sum(1 for a in arms if a.kind == k)
        if not n:
            continue
        sec = runner.cpu_seconds[k] - cpu0[k]
        refit = (runner.refit_cpu_seconds - ref0) if k == "classifier" else 0.0
        per_kind[k] = dict(n_arms=n, cpu_seconds_per_1000_steps_per_arm=(sec - refit) / S.CHECK_STEPS * 1000 / n,
                           refit_cpu_seconds_per_arm=(refit / (S.CHECK_STEPS // 200) / n) if refit else None)
    same = all(L.tree_sha256(first[a.name]["params"]) == L.tree_sha256(second[a.name]["params"]) for a in arms)
    for a in arms:
        v = second[a.name]
        res["arms"][a.name] = dict(kind=a.kind, coef=list(a.coef), finite=v["finite"],
                                   last_chunk_metrics=dict(zip(S.METRIC_NAMES, v["chunk_metrics"][-1])),
                                   classifier_refits=[dict(zip(S.CLS_DIAG_NAMES, r)) for r in v["refits"]]
                                   if v.get("refits") else None)
    res["timing"] = per_kind
    res["repeat_is_bitwise_identical"] = bool(same)
    res["all_finite"] = bool(all(v["finite"] for v in res["arms"].values()))
    # the W = 0 arm against the room harness's own program over the same 400 steps
    room_shared, _ = H.build_shared(w.pg, w.critic, REP, cfg=H.ProgramConfig(physics=w.pg.physics), need_harm=False)
    rr = H.RoomRunner(w.pg, w.clone, w.critic, ctx.learn, REP, cfg=H.ProgramConfig(physics=w.pg.physics),
                      shared=room_shared, steps=S.CHECK_STEPS, chunk=200)
    room = rr.run([H.RoomArm("w0_check", "plain", (0.0,), 1.0)])
    res["w0_against_room_runner"] = dict(
        equal=L.tree_sha256(room["w0_check"]["params"]) == L.tree_sha256(second["w0_check"]["params"]),
        max_abs_diff=S.max_param_diff(room["w0_check"]["params"], second["w0_check"]["params"]),
        rows_sha256_equal=rr.rows_sha256 == runner.rows_sha256)
    return res


def projection(timing_stall, timing_ns, fits, room_eval_seconds):
    """Projected CPU seconds of one seed (one pinned process), from the measured per-1,000-step costs, assuming
    linear cost in the number of vmapped arms, the room's typical (3) and worst (5) matching rounds and 1 (typical) or
    3 (worst) ladder extensions per family."""
    steps = S.STEPS / 1000.0
    fam_kind = {f: S.FAMILY_KIND[f][0] for f in S.FAMILIES}

    def arm_cost(t, f):
        k = fam_kind[f]
        c = t[k]["cpu_seconds_per_1000_steps_per_arm"] * steps
        if k == "classifier":
            c += t[k]["refit_cpu_seconds_per_arm"] * (S.STEPS // S.CHUNK)
        return c

    ladder = sum(arm_cost(timing_stall, f) * len(S.LADDER) for f in S.FAMILIES)
    plain = timing_stall["plain"]["cpu_seconds_per_1000_steps_per_arm"] * steps
    out = {}
    for label, rounds, ext in (("typical", 3, 1), ("worst", 5, 3)):
        match = sum(arm_cost(timing_stall, f) * (rounds + ext) for f in S.FAMILIES)
        scored = 2 + len(S.FAMILIES)
        stall_train = fits["stall"] + plain + ladder + match + 60.0 * 8 + 120.0
        stall_eval = scored * room_eval_seconds * 2 + 60.0
        ns_train = fits["no_stall"] + plain + sum(arm_cost(timing_ns, f) for f in S.FAMILIES) + 60.0 * 8 + 120.0
        ns_eval = scored * room_eval_seconds * 2 + 60.0
        total = stall_train + stall_eval + ns_train + ns_eval
        out[label] = dict(stall_train_minutes=stall_train / 60, stall_eval_minutes=stall_eval / 60,
                          no_stall_train_minutes=ns_train / 60, no_stall_eval_minutes=ns_eval / 60,
                          seed_minutes=total / 60, five_seeds_process_hours=5 * total / 3600,
                          arm_runs_stall=1 + len(S.FAMILIES) * (len(S.LADDER) + rounds + ext))
    out["assumptions"] = ("per-arm cost linear in the vmapped group; matching rounds 3 (typical) / 5 (worst) and "
                          "ladder extensions 1 / 3 per family, each one 20,000-step arm; check (e) about 60 CPU s per "
                          "family; overhead 120 s per train phase; evaluation per actor = 2 x the room's measured "
                          "evaluation cost (the exact evaluation and the per-thruster rollout), from the room's "
                          "timing.json (a cost estimate, no return)")
    return out


def part_path(world):
    return S.OUT / f"checks_seed0_{world}.json"


def summarise(spec_sha, man_sha):
    """checks_seed0.json and timing.json from the two worlds' part files (no computation)."""
    parts = {w: S.read_json(part_path(w)) for w in ("stall", "no_stall")}
    if any(p is None for p in parts.values()):
        raise SystemExit("refused: run the checks of both worlds first")
    for w, p in parts.items():
        if p["spec_sha256"] != spec_sha or p["code_manifest_sha256"] != man_sha:
            raise SystemExit(f"refused: the {w} part was computed under another spec or code manifest")
    room_timing = S.read_json(S.ROOM_OUT / "timing.json")
    ev_sec = room_timing["assumptions"]["evaluation_seconds_per_actor"]
    fits = {w: p["fit_timing"]["total_cpu"] for w, p in parts.items()}
    proj = projection(parts["stall"]["checks"]["d"]["timing"], parts["no_stall"]["checks"]["d"]["timing"], fits, ev_sec)
    proj["room_timing_file_sha256"] = S.file_sha256(S.ROOM_OUT / "timing.json")
    passed = {w: p["passed"] for w, p in parts.items()}
    out = dict(label=S.LABEL, kind="standins-checks-seed0", rep=REP, spec_sha256=spec_sha, code_manifest_sha256=man_sha,
               parts={w: dict(file=part_path(w).name, sha256=S.file_sha256(part_path(w))) for w in parts},
               passed=passed, no_return_computed=True,
               check_e_stall={f: dict(passed=r["passed"], e1=r["e1_fall_share_ok"], e2=r["e2_mean_rises"],
                                      e3=r["e3_slope_past_knee_positive"], e4=r["e4_fewer_best_past80_than_q1"],
                                      fall_share_80_to_81=r["fall_share_80_to_81"],
                                      rise_share_80_to_81=r["rise_share_80_to_81"],
                                      mean_W_80=r["mean_W"]["0.8"], mean_W_81=r["mean_W"]["0.81"],
                                      mean_W_100=r["mean_W"]["1"], slope_805=r["mean_dW_dpush"]["0.805"]["mean"],
                                      best_past80_knob1=r["best_past80_share_knob1"],
                                      q1_alone_best_past80=r["q1_alone_best_past80_share"],
                                      active_share=r["active_share"])
                             for f, r in parts["stall"]["checks"]["e"]["forms"].items()},
               ladder_scale_stall=parts["stall"]["checks"]["e"]["ladder_scale"],
               projection=proj)
    digest = S.write_json(S.OUT / "checks_seed0.json", out)
    S.write_json(S.OUT / "timing.json", dict(label=S.LABEL, kind="standins-timing", spec_sha256=spec_sha,
                                             code_manifest_sha256=man_sha, checks_sha256=digest,
                                             per_kind={w: p["checks"]["d"]["timing"] for w, p in parts.items()},
                                             fits={w: p["fit_timing"] for w, p in parts.items()}, projection=proj))
    log("projection: " + json.dumps(proj))
    log(f"wrote {S.OUT / 'checks_seed0.json'} ({digest[:16]}) and timing.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="run without the spec and write nothing")
    ap.add_argument("--world", choices=("stall", "no_stall"))
    ap.add_argument("--summarise", action="store_true", help="combine the two worlds' part files")
    ap.add_argument("checks", nargs="*", default=["f", "e", "d"])
    a = ap.parse_args()
    if a.dev:
        spec_sha = man_sha = None
    else:
        spec, spec_sha, man_sha = S.begin()
    if a.summarise:
        return summarise(spec_sha, man_sha)
    if not a.world:
        raise SystemExit("--world stall|no_stall (or --summarise)")
    t0, c0 = time.perf_counter(), cpu()
    ctx = H.SeedContext(REP, worlds=("stall", "no_stall") if a.world == "no_stall" else ("stall",))
    w = World(ctx, a.world)
    log(f"{w.world}: fits {json.dumps(w.timing)}")
    out = dict(label=S.LABEL, kind="standins-checks-seed0-part", rep=REP, world=a.world, spec_sha256=spec_sha,
               code_manifest_sha256=man_sha, code_sha256=S.code_hashes(), room_import=S.ROOM_IMPORT,
               no_return_computed=True, shared=w.info, fit_timing=w.timing,
               classifier_fits=dict(no_caution_actor=dict(l_star=w.cls_none[1], diag=w.cls_none[2]),
                                    cloned_actor=dict(l_star=w.cls_clone[1], diag=w.cls_clone[2])), checks={})
    passed = {}
    if "f" in a.checks:
        out["checks"]["f"] = check_f(w)
        passed["f"] = out["checks"]["f"]["passed"]
        log(f"check (f): {passed['f']}; at recorded commands (max W) {out['checks']['f']['at_recorded_commands_zero']}")
    scale = None
    if "e" in a.checks or "d" in a.checks:
        e_out, scales = check_e_world(w, ctx)
        scale = scales["scale"]
        e_out["note"] = ("reported for every stand-in, not a stop condition (spec 'check_e')" +
                         ("; no-stall world: reported only, and its ladder scale is for information (the no-stall arms "
                          "run at the stalling world's locked coefficients)" if a.world == "no_stall" else ""))
        out["checks"]["e"] = e_out
        passed["e"] = e_out["passed"]
        for f, r in e_out["forms"].items():
            log(f"check (e) {a.world} {f}: passed {r['passed']} fall {r['fall_share_80_to_81']:.3f} rise "
                f"{r['rise_share_80_to_81']:.3f} W80 {r['mean_W']['0.8']:.4f} W81 {r['mean_W']['0.81']:.4f} W100 "
                f"{r['mean_W']['1']:.4f} slope {r['mean_dW_dpush']['0.805']['mean']:.4f} best>80 k1 "
                f"{r['best_past80_share_knob1']:.3f} (Q1 {r['q1_alone_best_past80_share']:.3f}) by knob "
                f"{r['best_past80_share_by_knob']}")
        log(f"ladder scales {a.world}: " + json.dumps(scale))
    if "d" in a.checks:
        used = "this world's own ladder scale"
        if a.world == "no_stall" and not a.dev:
            st = S.read_json(part_path("stall"))
            if st is not None and st.get("spec_sha256") == spec_sha:
                scale, used = st["checks"]["e"]["ladder_scale"]["scale"], "the stalling world's ladder scale"
        d = run_400(w, ctx, scale)
        d["scale_used"] = used
        out["checks"]["d"] = d
        passed["d"] = bool(d["all_finite"] and d["w0_against_room_runner"]["equal"])
        log(f"check (d) {a.world}: finite {d['all_finite']}, repeat identical {d['repeat_is_bitwise_identical']}, "
            f"w0 vs room {d['w0_against_room_runner']}, timing {json.dumps(d['timing'])}")
    out["passed"] = passed
    out["seconds"], out["cpu_seconds"] = time.perf_counter() - t0, cpu() - c0
    if a.dev:
        log("DEV run: nothing written")
        print(json.dumps(A.plain(passed), indent=1))
        return
    if sorted(a.checks) != ["d", "e", "f"]:
        raise SystemExit("an official part runs all of f, e and d")
    digest = S.write_json(part_path(a.world), out)
    log(f"wrote {part_path(a.world)} ({digest[:16]})")


if __name__ == "__main__":
    main()
