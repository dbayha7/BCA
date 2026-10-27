"""EXPLORATORY: the seed-0 checks of plan section 7 step 3 (and 4.0's reproduction check), run after spec.json was
written. NO RETURN, POINTS OR SCORE is computed here (no rollout return, no evaluation; no strength either). Nothing
trained here is saved or scored.

  (a) the no-caution arm through the gate's own unchanged code reproduces check 2's saved gapped-critic actor by
      parameter hash;
  (b) the new program with W = 0 matches it within 1e-5 on every parameter (in the run's own configuration: the
      plain group with the even-pull ladder; and alone, for information);
  (c) the oracle. Plumbing: on the earlier readout's own 3,000 training rows (readout_seed0.json) the readout's Q1
      form reproduces its mean at the gapped critic's actor's past-80% commands (-0.7298) and its median size
      elsewhere (0.1066) within 0.01. The trained form (min over heads on both sides, BCA's residual) against the same
      numbers is reported as met or not met. The amended main oracle [delta*(s, a_i) - delta*(s, a)]_+ is read out on
      the same rows and on every row. float32 in-program against float64 at every training row (<= 1e-3); the
      long-run harm term's float32 rollout against float64; the gradients of delta* and of the main penalty along
      thruster-1 throttle finite on both sides of the 80% knee (and against float64 central differences);
  (d) a 400-step run of every arm kind (stalling world, no-stall world, check 3's critic) gives finite losses;
      timing per 1,000 steps per kind, and the projected wall-clock of one seed;
  (e) the shape of the main oracle's penalty past the knee (harness.oracle_shape), on check 2's critic and on check
      3's critic. 'e' runs seed 0 (part of the seed-0 checks); 'e1' ... 'e4' run that seed alone, reported, so that
      a seed the launch would stop is known in advance. Critic surfaces only.

Every check refuses unless the spec, the gate files and the code manifest check out.

Usage: python checks_seed0.py [a b c d e e1 e2 e3 e4]   (default: a b c d e)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import run_seed as R  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.recorder as RC  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

REP = 0
READOUT = Path("/home/user/bca/runs/shift_explore/principled/readout_seed0.json")


def readout_rows(ctx, pg, n=3000):
    """principled/readout.py's training-row sample, regenerated with its own random sequence."""
    sd = ctx.sd["stall"]
    rng = np.random.default_rng([20261013, REP, 51])
    lg = sd.log
    cal = lg.split_rows("calibration")
    cal = cal[~sd.gap.contains(lg["actions"][cal])]
    rng.choice(cal, n, replace=False)
    return np.sort(rng.choice(pg.n, n, replace=False))


def check_c(ctx, pg, critic, gapped, shared, out):
    t0 = time.perf_counter()
    cfg = H.ProgramConfig()
    rd = json.loads(READOUT.read_text())["pools"]["train|gapped"]
    res = dict(readout_file=str(READOUT), readout_sha256=H.file_sha256(READOUT),
               readout_mean_past80=rd["residual_gapped_mean_past_80"],
               readout_median_abs_other=rd["abs_residual_gapped_median_not_past_80"])
    # the readout's rows
    tr = readout_rows(ctx, pg)
    a = L.actor_rows(gapped, np.asarray(pg.train.obs)[tr]).astype(np.float64)
    past = TC.past_knee(a)
    d_q1 = H.oracle64(critic, pg, pg.states[tr], a, pg.physics, q_head="q1")
    d_min = H.oracle64(critic, pg, pg.states[tr], a, pg.physics, q_head="min")
    d32 = H.oracle32(critic, shared, cfg, tr, a)
    res["readout_rows"] = dict(
        n=int(len(tr)), past80_share=float(past.mean()), readout_past80_share=rd["past_80_share"],
        q1_form=dict(mean_past80=float(d_q1[past].mean()), median_abs_other=float(np.median(np.abs(d_q1[~past])))),
        min_form=dict(mean_past80=float(d_min[past].mean()), median_abs_other=float(np.median(np.abs(d_min[~past])))),
        min_form_float32_program=dict(mean_past80=float(d32[past].mean()),
                                      median_abs_other=float(np.median(np.abs(d32[~past])))))
    ref_m, ref_a = rd["residual_gapped_mean_past_80"], rd["abs_residual_gapped_median_not_past_80"]
    q1f, mnf = res["readout_rows"]["q1_form"], res["readout_rows"]["min_form"]
    res["plumbing_reproduces_readout"] = bool(abs(q1f["mean_past80"] - ref_m) <= 0.01 and
                                              abs(q1f["median_abs_other"] - ref_a) <= 0.01)
    res["trained_form_vs_readout"] = dict(
        met=bool(abs(mnf["mean_past80"] - ref_m) <= 0.01 and abs(mnf["median_abs_other"] - ref_a) <= 0.01),
        diff_mean_past80=mnf["mean_past80"] - ref_m, diff_median_abs_other=mnf["median_abs_other"] - ref_a, tol=0.01,
        note=("reported, not a stop condition (spec change A3): the readout took Q1 on the Q(s, a) side; the trained "
              "residual takes the minimum over both heads, as BCA's own residual does; the plan's 7.3(b) reference "
              "values apply only to the Q1 form"))
    d_rec_tr = H.oracle64(critic, pg, pg.states[tr], pg.actions[tr].astype(np.float64), pg.physics, q_head="min")
    w_tr = H.band_terms(np.asarray(H.MAIN_BAND), d_min, d_rec_tr)
    res["readout_rows"]["main_oracle"] = dict(
        W="[delta*(s, a_i) - delta*(s, a)]_+",
        mean_past80=float(w_tr[past].mean()), median_other=float(np.median(w_tr[~past])),
        mean_other=float(w_tr[~past].mean()), share_zero_other=float(np.mean(w_tr[~past] == 0.0)),
        delta_at_recorded_mean=float(d_rec_tr.mean()))
    # every training row
    idx = np.arange(pg.n)
    a_all = L.actor_rows(gapped, np.asarray(pg.train.obs)).astype(np.float64)
    past_all = TC.past_knee(a_all)
    d64_all = H.oracle64(critic, pg, pg.states, a_all, pg.physics, q_head="min")
    d32_all = H.oracle32(critic, shared, cfg, idx, a_all)
    q1_all = H.oracle64(critic, pg, pg.states, a_all, pg.physics, q_head="q1")
    d64_rec = H.oracle64(critic, pg, pg.states, pg.actions.astype(np.float64), pg.physics, q_head="min")
    d32_rec = H.oracle32(critic, shared, cfg, idx, pg.actions)
    reg = pg.region
    w_all64 = H.band_terms(np.asarray(H.MAIN_BAND), d64_all, d64_rec)
    w_all32 = H.band_terms(np.asarray(H.MAIN_BAND), d32_all, np.asarray(shared["d_logged"], np.float64))
    res["all_rows_main_oracle"] = dict(
        mean_past80=float(w_all64[past_all].mean()), mean_past80_region=float(w_all64[past_all & reg].mean()),
        median_other=float(np.median(w_all64[~past_all])), mean_other=float(w_all64[~past_all].mean()),
        at_recorded_commands="0 by construction",
        max_abs_float32_program_vs_float64=float(np.abs(w_all32 - w_all64).max()),
        d_logged_vs_float64_max_abs=float(np.abs(np.asarray(shared["d_logged"], np.float64) - d64_rec).max()))
    res["all_rows"] = dict(
        n=int(pg.n), past80_share=float(past_all.mean()), region_share=float(reg.mean()),
        past80_share_in_region=float(past_all[reg].mean()),
        min_form=dict(mean_past80=float(d64_all[past_all].mean()),
                      median_abs_other=float(np.median(np.abs(d64_all[~past_all]))),
                      mean_past80_region=float(d64_all[past_all & reg].mean())),
        q1_form=dict(mean_past80=float(q1_all[past_all].mean()),
                     median_abs_other=float(np.median(np.abs(q1_all[~past_all])))),
        at_recorded_commands=dict(median_abs=float(np.median(np.abs(d64_rec))), mean=float(d64_rec.mean())),
        float32_vs_float64=dict(max_abs_diff_actor_commands=float(np.abs(d32_all - d64_all).max()),
                                max_abs_diff_recorded_commands=float(np.abs(d32_rec - d64_rec).max()),
                                tol=H.ORACLE_PRECISION_TOL))
    res["precision_passed"] = bool(max(res["all_rows"]["float32_vs_float64"]["max_abs_diff_actor_commands"],
                                       res["all_rows"]["float32_vs_float64"]["max_abs_diff_recorded_commands"])
                                   <= H.ORACLE_PRECISION_TOL)
    # the long-run harm term: float32 in-program 250-step value against float64 (truth.q_value_jax, same horizon)
    sub = np.sort(np.random.default_rng([20261015, REP, 64]).choice(pg.n, 4000, replace=False))
    sa = RC.stall_aware(pg.physics)
    for name, acts in (("recorded", pg.actions[sub].astype(np.float64)), ("gapped_actor", a_all[sub])):
        q32 = H.controller_value_rows(np.asarray(pg.states[sub], np.float32), acts.astype(np.float32), cfg)
        q64 = TR.q_value_jax(sa, pg.states[sub], acts.astype(np.float32).astype(np.float64), pg.physics,
                             horizon=H.HARM_HORIZON, chunk=4000)
        q1000 = TR.q_value_jax(sa, pg.states[sub], acts.astype(np.float32).astype(np.float64), pg.physics,
                               horizon=TR.HORIZON, chunk=4000)
        res[f"harm_value_{name}"] = dict(rows=int(len(sub)), max_abs_float32_vs_float64=float(np.abs(q32 - q64).max()),
                                         max_rel=float(np.max(np.abs(q32 - q64) / np.maximum(np.abs(q64), 1e-9))),
                                         max_abs_truncation_250_vs_1000=float(np.abs(q64 - q1000).max()),
                                         median_abs_value=float(np.median(np.abs(q64))))
    # gradient along thruster-1 throttle on both sides of the knee, at region training states
    rows = np.flatnonzero(reg)
    rows = np.sort(np.random.default_rng([20261015, REP, 65]).choice(rows, min(2000, len(rows)), replace=False))
    base = a_all[rows].copy()
    sign = np.where(base[:, 0] < 0, -1.0, 1.0)
    grad, wgrad = {}, {}
    h = 1e-4
    d_rec_rows = H.oracle64(critic, pg, pg.states[rows], pg.actions[rows].astype(np.float64), pg.physics)
    for side, mag in (("inside_0.79", 0.79), ("past_0.805", 0.805), ("past_0.81", 0.81)):
        aa = base.copy()
        aa[:, 0] = sign * mag
        g32 = H.oracle_grad32(critic, shared, cfg, rows, aa)[:, 0]
        ap, am = aa.copy(), aa.copy()
        ap[:, 0] += h
        am[:, 0] -= h
        dp = H.oracle64(critic, pg, pg.states[rows], ap, pg.physics)
        dm = H.oracle64(critic, pg, pg.states[rows], am, pg.physics)
        fd = (dp - dm) / (2 * h)
        # along the push direction (sign): d delta* / d |a_1|
        grad[side] = dict(finite_share=float(np.isfinite(g32).mean()), mean_d_delta_d_push=float((g32 * sign).mean()),
                          median_d_delta_d_push=float(np.median(g32 * sign)),
                          central_difference_float64_mean=float((fd * sign).mean()),
                          median_abs_autodiff_minus_central=float(np.median(np.abs(g32 - fd))),
                          rows=int(len(rows)), negative_push_share=float((sign < 0).mean()))
        # the main penalty W = [delta*(s, a_i) - delta*(s, a)]_+ (what the actor feels)
        gw = H.penalty_grad32(critic, shared, cfg, rows, aa, H.MAIN_BAND)[:, 0]
        c = np.asarray(H.MAIN_BAND)
        fdw = (H.band_terms(c, dp, d_rec_rows) - H.band_terms(c, dm, d_rec_rows)) / (2 * h)
        wgrad[side] = dict(finite_share=float(np.isfinite(gw).mean()), mean_dW_d_push=float((gw * sign).mean()),
                           central_difference_float64_mean=float((fdw * sign).mean()),
                           share_dW_d_push_negative=float(np.mean(gw * sign < 0)))
    res["thruster1_gradient"] = grad
    res["thruster1_gradient_main_penalty"] = wgrad
    res["gradient_finite_both_sides"] = bool(all(v["finite_share"] == 1.0 for v in list(grad.values()) +
                                                 list(wgrad.values())))
    res["seconds"] = time.perf_counter() - t0
    out["c"] = res
    H.log(f"check (c): {json.dumps(H.A.plain(res))[:3000]}")


def check_d(ctx, pg, critic, clone, shared, out, scale, nets):
    """400 steps of every arm kind in every configuration; finite losses; timing (second call, compiled)."""
    t0 = time.perf_counter()
    res = dict(steps=H.CHECK_STEPS, configs={})

    def go(label, runner, arms):
        first = runner.run(arms)
        t = time.perf_counter()
        sec0 = dict(runner.seconds)
        second = runner.run(arms)
        per_kind = {}
        for k in H.KINDS:
            n = sum(1 for a in arms if a.kind == k)
            if n:
                per_kind[k] = dict(n_arms=n, seconds_per_1000_steps_per_arm=(runner.seconds[k] - sec0[k]) /
                                   H.CHECK_STEPS * 1000 / n)
        same = all(L.tree_sha256(first[a.name]["params"]) == L.tree_sha256(second[a.name]["params"]) for a in arms)
        res["configs"][label] = dict(
            arms={a.name: dict(kind=a.kind, finite=second[a.name]["finite"],
                               last_chunk_metrics=dict(zip(H.METRIC_NAMES, second[a.name]["chunk_metrics"][-1])))
                  for a in arms},
            all_finite=bool(all(v["finite"] for v in second.values())), repeat_is_bitwise_identical=bool(same),
            timing=per_kind, seconds=time.perf_counter() - t)
        H.log(f"check (d) {label}: finite {res['configs'][label]['all_finite']}, timing {per_kind}")

    cfg = H.ProgramConfig()
    runner = H.RoomRunner(pg, clone, critic, ctx.learn, REP, cfg=cfg, shared=shared,
                          random_nets={j: nets[j][0] for j in (1, 2)}, steps=H.CHECK_STEPS, chunk=H.CHECK_STEPS)
    arms = H.fixed_arms() + [H.family_arm(f, 1.0, scale) for f in H.GATING]
    go("stall", runner, arms)
    # no-stall world
    pt = ctx.prepared("no_stall")
    ct, _ = ctx.critic("no_stall", "gap")
    clone_t, _ = ctx.clone("no_stall")
    for nc in ("stall_aware", "rising_thrust"):
        c2 = H.ProgramConfig(physics=pt.physics, next_command=nc)
        sh2, _ = H.build_shared(pt, ct, REP, cfg=c2, need_harm=False)
        r2 = H.RoomRunner(pt, clone_t, ct, ctx.learn, REP, cfg=c2, shared=sh2, steps=H.CHECK_STEPS,
                          chunk=H.CHECK_STEPS)
        a2 = [H.RoomArm("perfect_band", "band", H.MAIN_BAND)]
        if nc == "stall_aware":
            a2 += [H.RoomArm("w0_check", "plain"), H.RoomArm("even_pull_locked", "plain", (0.0,), 1.5),
                   H.RoomArm("range_hinge", "hinge", (1.0,))]
        go(f"no_stall_{nc}", r2, a2)
    # check 3's critic
    host, _ = ctx.host()
    ch = H.ProgramConfig(next_command="host_actor", bc_form="weighted")
    shh, _ = H.build_shared(pg, host.critic.params, REP, cfg=ch, host_actor=host.actor_target.params, need_harm=False)
    rh = H.RoomRunner(pg, clone, host.critic.params, ctx.learn, REP, cfg=ch, shared=shh, steps=H.CHECK_STEPS,
                      chunk=H.CHECK_STEPS)
    go("replication", rh, [H.RoomArm("w0_check", "plain"), H.RoomArm("even_pull@1", "plain", (0.0,), 2.0),
                           H.RoomArm("perfect_band", "band", H.MAIN_BAND)])
    res["all_finite"] = bool(all(c["all_finite"] for c in res["configs"].values()))
    res["seconds"] = time.perf_counter() - t0
    out["d"] = res


def evaluation_cost_estimate(pg, clone):
    """Seconds per scored actor, estimated WITHOUT computing any return: the float64 MLP on 1,000 states 100 times
    (one rollout's policy cost), times 12 rollouts (4 situations x (uncapped, capped, per-thruster pass))."""
    pol = L.mlp_policy(clone, pg)
    x = TC.bank_starts("recorded", TC.BANK_SIZE)
    t = time.perf_counter()
    for _ in range(100):
        pol(x)
    one = time.perf_counter() - t
    return dict(policy_seconds_per_rollout=one, estimated_seconds_per_actor=12 * one * 1.3,
                note="1.3 x for the dynamics and bookkeeping; no return was computed")


def check_e(rep, spec_sha, man_sha, ctx=None):
    """Check (e) on one seed: the main oracle's penalty shape on check 2's critic (no-caution actor = check 2's saved
    gapped actor) and on check 3's critic (check 3's saved 'none' actor, next command = the host's actor target).
    Writes checks_seed0_e.json (seed 0) or checks_e_seed<k>.json (other seeds, reported)."""
    t0 = time.perf_counter()
    ctx = ctx or H.SeedContext(rep)
    pg = ctx.prepared()
    critic, _ = ctx.critic()
    none = ctx.h2()[0]["gapped"]
    cfg = H.ProgramConfig()
    shared, sinfo = H.build_shared(pg, critic, rep, cfg=cfg, need_harm=False)
    stall = H.oracle_shape(critic, shared, cfg, pg, none, rep)
    H.log(f"check (e) seed {rep} stall: passed {stall['passed']} {R.summary_e(stall)}")
    host, _ = ctx.host()
    ch = H.ProgramConfig(next_command="host_actor", bc_form="weighted")
    shh, hinfo = H.build_shared(pg, host.critic.params, rep, cfg=ch, host_actor=host.actor_target.params,
                                need_harm=False)
    repl = H.oracle_shape(host.critic.params, shh, ch, pg, ctx.h3()[0]["none"], rep)
    H.log(f"check (e) seed {rep} replication: passed {repl['passed']} {R.summary_e(repl)}")
    out = dict(label=H.LABEL + " (checks; no return, points, score or strength computed)",
               kind=f"room-check-e-seed{rep}", rep=int(rep), spec_sha256=spec_sha, code_manifest_sha256=man_sha,
               e=dict(stall=stall, replication=repl, passed=bool(stall["passed"] and repl["passed"]),
                      d_logged=dict(stall=sinfo, replication=hinfo),
                      rule=H.read_json(H.SPEC_PATH)["checks"]["e"]),
               seconds=time.perf_counter() - t0)
    path = H.OUT / ("checks_seed0_e.json" if rep == 0 else f"checks_e_seed{rep}.json")
    H.write_json(path, out)
    H.log(f"wrote {path}")


def main(which):
    """Each check writes its own file checks_seed0_<letter>.json, so the checks can run as separate processes."""
    spec, spec_sha = H.load_spec()
    H.check_gate_files(spec)
    man, man_sha = H.check_code(spec_sha)
    for tok in sorted(w for w in which if w.startswith("e") and w != "e"):
        check_e(int(tok[1:]), spec_sha, man_sha)
    which = {w for w in which if not (w.startswith("e") and w != "e")}
    if not which:
        return
    ctx = H.SeedContext(REP, worlds=("stall", "no_stall"))
    pg = ctx.prepared()
    critic, _ = ctx.critic()
    clone, _ = ctx.clone()
    acts, _, _ = ctx.h2()
    gapped = acts["gapped"]
    cfg = H.ProgramConfig()
    shared, sinfo = H.build_shared(pg, critic, REP, cfg=cfg, log_fn=H.log)
    nets = {j: H.random_surface_params(REP, j) for j in (1, 2)}
    idx = H.slope_sample(pg, REP)
    a_clone = L.actor_rows(clone, np.asarray(pg.train.obs)[idx])
    g = dict(band=H.slope_rms("band", cfg, critic, shared, idx, a_clone),
             push_less=H.slope_rms("push_less", cfg, critic, shared, idx, a_clone),
             random_1=H.slope_rms("random", cfg, critic, shared, idx, a_clone, net=nets[1][0]),
             random_2=H.slope_rms("random", cfg, critic, shared, idx, a_clone, net=nets[2][0]))
    scale = {"even_pull": 1.0, "push_less": g["band"] / g["push_less"], "random_1": g["band"] / g["random_1"],
             "random_2": g["band"] / g["random_2"]}
    for letter in sorted(which):
        if letter == "e":
            check_e(REP, spec_sha, man_sha, ctx=ctx)
            continue
        t0 = time.perf_counter()
        out = dict(label=H.LABEL + " (seed-0 checks; no return, points or score computed)",
                   kind=f"room-check-{letter}-seed0", rep=REP, spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                   code_sha256=man["files"], shared=sinfo, slope_rms=g, coefficient_per_knob=scale)
        if letter == "a":
            repro, _ = R.reproduce_h2(ctx)
            out["a"] = repro
            H.log(f"check (a): passed {repro['passed']} {repro['arms']} ({repro['seconds']:.0f}s)")
        elif letter == "b":
            runner = H.RoomRunner(pg, clone, critic, ctx.learn, REP, cfg=cfg, shared=shared)
            check2_rows = L.ActorOnly(pg, clone, ctx.learn, REP, alpha=H.ALPHA).rows_sha256
            t = time.perf_counter()
            group = runner.run([H.RoomArm("w0_check", "plain")] +
                               [H.family_arm("even_pull", nu, scale) for nu in H.EVEN_LADDER], H.log)
            t_group = time.perf_counter() - t
            alone = runner.run([H.RoomArm("w0_alone", "plain")], H.log)
            d_group = H.max_param_diff(group["w0_check"]["params"], gapped)
            d_alone = H.max_param_diff(alone["w0_alone"]["params"], gapped)
            out["b"] = dict(tol=H.W0_TOL,
                            run_configuration=dict(max_abs_diff=d_group, passed=bool(d_group <= H.W0_TOL),
                                                   bitwise_equal=L.tree_sha256(group["w0_check"]["params"])
                                                   == L.tree_sha256(gapped)),
                            alone=dict(max_abs_diff=d_alone, bitwise_equal=L.tree_sha256(alone["w0_alone"]["params"])
                                       == L.tree_sha256(gapped)),
                            rows_sha256_equals_check2=runner.rows_sha256 == check2_rows,
                            plain_group_seconds=t_group,
                            plain_seconds_per_1000_steps_per_arm=t_group / H.STEPS * 1000 / 6,
                            all_finite=bool(all(v["finite"] for v in group.values())))
            out["b"]["passed"] = bool(out["b"]["run_configuration"]["passed"] and out["b"]["rows_sha256_equals_check2"])
            H.log(f"check (b): {out['b']}")
        elif letter == "c":
            check_c(ctx, pg, critic, gapped, shared, out)
        elif letter == "d":
            check_d(ctx, pg, critic, clone, shared, out, scale, nets)
            out["evaluation_cost"] = evaluation_cost_estimate(pg, clone)
        out["seconds"] = time.perf_counter() - t0
        path = H.OUT / f"checks_seed0_{letter}.json"
        H.write_json(path, out)
        H.log(f"wrote {path}")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"a", "b", "c", "d", "e"})
