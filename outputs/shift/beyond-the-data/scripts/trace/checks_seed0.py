"""EXPLORATORY seed-0 checks of the trace test bed (plan 4.6), spec v2. No BCA. No return of a trained actor and no
signal at an actor's command is computed. Refuses unless the process is pinned to one CPU and spec.json, the gate
files and the code manifest check out.

  (a) build the seed-0 trace logs of both worlds at shares 0, 0.5% and 2% (run_job.phase_build: the official
      trace_rows.json of seed 0) and verify the row counts, the build checks, the realised share of the shared
      candidates, the past-80% share of the kept training rows, that the 0.5% rows are a subset of the 2% rows, and
      that both worlds add back the same rows (paired, review finding 3); share 0 must equal the gate's gapped log
      (removal record and prepared data SHA-256). The world-specific candidates are reported.
  (a2) the paired T1 states: 1,500 rows drawn from the region rows both worlds keep; the T1c command (check 1's
      full-throttle pattern) is the same in both worlds and past 80%. States only; no score.
  (b) the pull past 80% of the gate's gapped and full-log critics, recomputed with tracebed.region_pull, equals check
      1's saved rows bit for bit (both worlds): the overrating readout is check 1's own.
  (c) learners.clone on the share-0 log reproduces the gate's saved cloned actor of the stalling world (parameter
      SHA-256): the clone path through the new removal is the gate's.
  (f) pinned to one CPU, run_job.train_actors (the train phase's own actor-only call: three arms side by side) on
      the share-0 log, with the gate's gapped critic in the 'trace' slot, reproduces check 2's saved gapped-critic
      and full-log-critic actors by parameter SHA-256 (review finding 2). Its time is the 20,000-step actor timing.
  (g) learners.fixed_critic on the share-0 log for one 5,000-update chunk (the train phase's first compiled chunk)
      reproduces the gate's gapped critic's initial parameters (SHA-256) and first-chunk mean critic loss exactly.
      Its time gives the critic timing.
  (h) the J-by-region diagnostic (review finding 1) on the gate's gapped and full-log critics, both worlds, with the
      official noise draws (signals.next_state_targets): J at region rows and outside. Reads the critics only at
      next states; no one-step error, residual or flag. Reported.
  (d) timing on the stalling world's 0.5% log (results discarded): one situation's exact evaluation (1,000 starts,
      uncapped and capped) of an untrained actor, the stand-ins' set-up (with the 32 noise draws) on the (g)
      critic, their T4 scores, one 1,500-state score call and one 200-start rollout. Then the projected cost of
      every job.
Writes checks_seed0.json and timing.json.

Spec v2 only: these checks ran under spec v2 and their outputs are kept as they are. Under spec v3 (post-hoc shares)
this script refuses; spec v3's checks are checks_v3.py.

Usage (from /home/user/bca): taskset -c <cpu> python checks_seed0.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracebed as T  # noqa: E402
import signals as S  # noqa: E402
import run_job as R  # noqa: E402

L, TC = T.L, T.TC
REP = 0


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


def main():
    cpu = T.require_one_cpu("the seed-0 checks")
    spec, spec_sha, man_sha = T.begin()
    if spec.get("post_hoc_extension"):
        raise SystemExit("refused: the seed-0 checks of spec v2 are complete and kept as they are; spec v3's checks "
                         "are checks_v3.py")
    load0 = os.getloadavg()
    t_all = time.perf_counter()
    g = R.load("stall", REP, spec)          # both worlds: the load time of one job
    checks, timing = {}, dict(load_one_job_s=g.load_seconds, cpus=cpu)
    T.log(f"loaded seed {REP}, both worlds, in {g.load_seconds:.1f} s (CPU {cpu[0]})")

    # (a) builds
    built = {}
    for world in T.WORLDS:
        for share in (T.REFERENCE_SHARE,) + T.V2_SHARES:
            built[(world, share)] = R.phase_build(g, world, REP, share, spec_sha, man_sha)
    a = {}
    for world in T.WORLDS:
        sd = g.sd[world]
        r0, r1, r2 = (built[(world, s)] for s in (0.0, 0.005, 0.02))
        per = {}
        for share in (0.0, 0.005, 0.02):
            r = built[(world, share)]
            n = r["shared_candidates"]
            per[T.SHARE_DIR[share]] = dict(
                candidates=r["candidates"], shared_candidates=n, world_specific_candidates=r["world_specific_candidates"],
                added=r["added"], expected_added=T.trace_count(share, n),
                realised_share_of_shared=r["realised_share_of_shared_candidates"],
                realised_share_within_half_a_row=bool(abs(r["added"] - share * n) <= 0.5),
                kept_training_rows=r["kept_training_rows"], gate_kept_training_rows=r["gate_kept_training_rows"],
                kept_equals_gate_plus_added=bool(r["kept_training_rows"] == r["gate_kept_training_rows"] + r["added"]),
                past80_kept_rows=r["past80_kept_rows"], past80_kept_share=r["past80_kept_share"],
                gate_past80_kept_rows=r["gate_past80_kept_rows"],
                past80_rows_added_equals_added=bool(r["past80_kept_rows"] == r["gate_past80_kept_rows"] + r["added"]),
                bank_candidates=r["bank_candidates"], shared_bank_candidates=r["shared_bank_candidates"],
                bank_added=r["bank_added"],
                added_rows_recorded_command_same_in_both_worlds=r["added_rows_recorded_command_same_in_both_worlds"],
                prepared_rows=r["prepared"]["rows"], prepared_rows_equals_kept=bool(r["prepared"]["rows"] ==
                                                                                    r["kept_training_rows"]),
                build_checks=r["checks"], passed=r["passed"])
        nested = bool(set(r1["added_rows"]) <= set(r2["added_rows"])
                      and set(r1["added_bank_rows"]) <= set(r2["added_bank_rows"]))
        share0 = bool(r0["added"] == 0 and r0["kept_rows_sha256"] == sd.record["removals"]["gap"]["kept_rows_sha256"]
                      and r0["checks"].get("share0_prepared_equals_gate_gapped", False))
        ok = (all(p["passed"] and p["realised_share_within_half_a_row"] and p["kept_equals_gate_plus_added"]
                  and p["past80_rows_added_equals_added"] and p["prepared_rows_equals_kept"]
                  and p["added"] == p["expected_added"] for p in per.values()) and nested and share0)
        a[world] = dict(shares=per, nested_0p5_in_2=nested, share0_is_gate_gapped_log=share0, passed=bool(ok))
        T.log(f"(a) {world}: " + "; ".join(f"{k}: {v['added']}/{v['shared_candidates']} shared added "
                                           f"({v['world_specific_candidates']} world-specific), "
                                           f"{v['kept_training_rows']} rows, {v['past80_kept_rows']} past 80%"
                                           for k, v in per.items())
              + f"; nested {nested}; share 0 = gate {share0}")
    paired = {T.SHARE_DIR[s]: bool(built[("stall", s)]["added_rows"] == built[("no_stall", s)]["added_rows"]
                                   and built[("stall", s)]["added_bank_rows"] == built[("no_stall", s)]["added_bank_rows"])
              for s in (0.0,) + T.V2_SHARES}
    a["paired_same_rows_in_both_worlds"] = paired
    a["passed"] = bool(all(a[w]["passed"] for w in T.WORLDS) and all(paired.values()))
    checks["a_builds"] = a
    T.log(f"(a) paired across worlds: {paired}")

    # (a2) the paired T1 states and the T1c command (no score)
    rows1, pool1 = S.t1_rows(g.sd, REP)
    pat = {w: S.common_pattern(g.sd[w]) for w in T.WORLDS}
    a2 = dict(t1_rows=int(len(rows1)), t1_shared_pool=pool1, t1_rows_sha256=T.DS.sha256_array(rows1),
              t1_rows_kept_and_in_region_in_both_worlds=bool(all(np.isin(rows1, S.t1_pool(g.sd[w])).all()
                                                                  for w in T.WORLDS)),
              t1c_pattern=pat["stall"].tolist(),
              t1c_pattern_same_in_both_worlds=bool(np.array_equal(pat["stall"], pat["no_stall"])),
              t1c_pattern_past80=bool(T.past80(pat["stall"][None], g.sd["stall"].physics).all()))
    a2["passed"] = bool(a2["t1_rows"] == S.T1_STATES and a2["t1_rows_kept_and_in_region_in_both_worlds"]
                        and a2["t1c_pattern_same_in_both_worlds"] and a2["t1c_pattern_past80"])
    checks["a2_pairs"] = a2
    T.log(f"(a2) T1: {len(rows1)} of {pool1} shared region rows; T1c pattern {pat['stall'].round(3).tolist()}, "
          f"same in both worlds {a2['t1c_pattern_same_in_both_worlds']}, past 80% {a2['t1c_pattern_past80']}")

    # (b) region pull reproduces check 1's rows
    h1 = g.h1_rows()
    b = {}
    for world in T.WORLDS:
        prep = g.gapped_prepared(world)
        for removal, key in (("gap", "gap"), ("none", "full")):
            params, _ = g.critic(world, removal)
            mine = T.region_pull(params, g.sd[world], prep, h1)
            saved = h1[f"{world}:{T.CONTROLLER}:{key}|pull|region"]
            b[f"{world}/{key}"] = dict(max_abs_difference=float(np.max(np.abs(mine - saved))),
                                       bit_identical=bool(np.array_equal(mine, saved)), n=int(len(saved)))
    checks["b_region_pull"] = dict(rows=b, passed=bool(all(v["bit_identical"] for v in b.values())))
    T.log(f"(b) region pull identical: {checks['b_region_pull']['passed']}")

    # (c) the gate's clone reproduced at share 0
    sd = g.sd["stall"]
    tr0 = T.build_trace(g.sd, "stall", 0.0)
    p0 = T.trace_prepared(sd, tr0, g.learn)
    (clone_new, crec), t_clone = timed(L.clone, p0, g.learn, REP)
    gate_clone, gate_rec = g.clone("stall")
    same = L.tree_sha256(clone_new) == gate_rec["result"]["actor_sha256"] == L.tree_sha256(gate_clone)
    checks["c_clone"] = dict(new_sha256=L.tree_sha256(clone_new), gate_sha256=gate_rec["result"]["actor_sha256"],
                             rows_sha256_equal=bool(crec["rows_sha256"] == gate_rec["result"]["rows_sha256"]),
                             passed=bool(same))
    timing["clone_20k_steps_s"] = t_clone
    T.log(f"(c) clone reproduced: {same} ({t_clone:.0f} s for 20,000 steps)")

    # (f) the train phase's actor-only call reproduces check 2 at share 0, pinned
    full, _ = g.critic("stall", "none")
    gapped, gapped_rec = g.critic("stall", "gap")
    acts, _, _ = g.h2()
    trained, failed, info, t_actor = R.train_actors(p0, gate_clone, [("trace", gapped), ("full", full),
                                                                      ("gapped", gapped)], g.learn, REP)
    hashes = {n: L.tree_sha256(p) for n, p in trained.items()}
    ref = {"trace": L.tree_sha256(acts["gapped"]), "full": L.tree_sha256(acts["full"]),
           "gapped": L.tree_sha256(acts["gapped"])}
    checks["f_actor_protocol"] = dict(
        what=("run_job.train_actors on the share-0 log of the stalling world, arms trace (the gate's gapped critic), "
              "full, gapped, from the gate's clone, pinned to one CPU; compared with check 2's saved actors"),
        cpus=T.cpus(), arm_sha256=hashes, check2_sha256=ref,
        equal={n: bool(hashes[n] == ref[n]) for n in ref}, failed=failed, rows_sha256=info["rows_sha256"],
        passed=bool(all(hashes[n] == ref[n] for n in ref) and not any(failed.values())))
    timing["actor_20k_steps_3_arms_s"] = t_actor
    T.log(f"(f) actor protocol reproduces check 2: {checks['f_actor_protocol']['equal']} ({t_actor:.0f} s)")

    # (g) the trace critic's protocol: first chunk reproduces the gate's gapped critic
    (krec, kstate), t_critic = timed(L.fixed_critic, p0, T.CONTROLLER, g.learn, REP, updates=g.learn.chunk)
    gres = gapped_rec["result"]
    checks["g_critic_protocol"] = dict(
        what=(f"learners.fixed_critic on the share-0 log of the stalling world, {g.learn.chunk} updates (the first "
              "compiled chunk), against the gate's gapped critic's record"),
        initial_sha256=krec["initial_sha256"], gate_initial_sha256=gres["initial_sha256"],
        first_chunk_loss=krec["chunks"][0]["critic_loss"], gate_first_chunk_loss=gres["chunks"][0]["critic_loss"],
        passed=bool(krec["initial_sha256"] == gres["initial_sha256"]
                    and krec["chunks"][0]["critic_loss"] == gres["chunks"][0]["critic_loss"]))
    timing["critic_first_chunk_s"] = t_critic
    T.log(f"(g) critic protocol: initial {krec['initial_sha256'] == gres['initial_sha256']}, first-chunk loss "
          f"{krec['chunks'][0]['critic_loss']:.6g} against {gres['chunks'][0]['critic_loss']:.6g} "
          f"({t_critic:.0f} s for {g.learn.chunk} updates)")

    # (h) the J-by-region diagnostic on the gate's critics (reported)
    h = {}
    for world in T.WORLDS:
        prep = g.gapped_prepared(world)
        reg = np.asarray(prep.region, bool)
        n_log = len(g.sd[world].log["actions"])
        for removal, key in (("gap", "gapped"), ("none", "full")):
            params, _ = g.critic(world, removal)
            plain, mean = S.next_state_targets(params, prep, g.learn, REP, n_log)
            J = plain - mean
            h[f"{world}/{key}"] = dict(all=float(J.mean()), region_rows=float(J[reg].mean()),
                                       outside_region_rows=float(J[~reg].mean()), region_rows_n=int(reg.sum()))
            T.log(f"(h) {world}/{key}: J region {J[reg].mean():+.3f}, outside {J[~reg].mean():+.3f}")
    checks["h_offset_diagnostic"] = dict(
        rows=h, draws=S.DRAWS, reported=True,
        note=("J(s') = min_h Q(s', c(s')) - mean over the training noise of min_h Q(s', clip(c(s') + noise)), at the "
              "next states of the gapped log's training rows; the noiseless one-step error exceeds the training-form "
              "one by 0.98 J. Reviewer's figures (16 draws, own seed): stalling full-log +0.269 / +0.132, no-stall "
              "full-log +0.123 / +0.113, gapped +0.133 / +0.113 and +0.128 / +0.113"))

    # (d) timing on the stalling world's 0.5% log
    tr = T.build_trace(g.sd, "stall", 0.005)
    prep = T.trace_prepared(sd, tr, g.learn)
    untrained = L.initial_actor(g.learn, REP)
    _, t_ev = timed(L.evaluate_policy, L.mlp_policy(untrained, prep), sd.physics, g.anchors("stall"),
                    situations=("strong_current",), n=g.cfg.bank_size)
    timing["eval_one_situation_1000_starts_s"] = t_ev
    T.log(f"(d) evaluation: one situation, 1,000 starts, uncapped and capped: {t_ev:.1f} s")
    si, t_si = timed(S.StandIns, sd, prep, kstate.critic.params, gate_clone, g.learn, REP)
    rows4 = S.t4_rows(sd)
    _, t_t4 = timed(si.scores, sd.log["infos/state"][rows4], np.asarray(sd.log["actions"][rows4], np.float32))
    x3, t_roll = timed(S.visited_states, L.mlp_policy(untrained, prep), sd, 0)
    a3 = L.actor_rows(untrained, prep.normalize(x3))
    _, t_sc = timed(si.scores, x3, a3)
    timing.update(standins_setup_with_noise_draws_s=t_si, t4_scores_5000_s=t_t4, rollout_200_starts_s=t_roll,
                  scores_1500_s=t_sc)
    T.log(f"(d) stand-ins: set-up {t_si:.1f} s, T4 {t_t4:.1f} s, rollout {t_roll:.1f} s, 1,500 scores {t_sc:.1f} s")

    # projection
    load_s = g.load_seconds
    critic_100k = (g.learn.updates / g.learn.chunk) * t_critic
    eval_s = len(T.EVAL_ACTORS) * len(TC.SITUATIONS) * t_ev
    n_score_calls = 1 + 2 * len(S.SETS)          # T1 at the recorded command, then command and cap per set
    signals_s = t_si + t_t4 + 2 * t_roll + n_score_calls * t_sc
    job = load_s + t_clone + critic_100k + t_actor + eval_s + signals_s
    ref = load_s + signals_s
    n_jobs, n_ref = len(T.WORLDS) * len(T.SEEDS) * len(T.V2_SHARES), len(T.WORLDS) * len(T.SEEDS)
    total_h = (n_jobs * job + n_ref * ref) / 3600
    per_seed_h = total_h / len(T.SEEDS)
    timing["projection"] = dict(
        per_trace_job_s=dict(load_both_worlds=load_s, clone=t_clone, critic_100k=critic_100k,
                             actor_20k_3_arms=t_actor, eval_4_actors_4_situations=eval_s, signals=signals_s, total=job),
        per_reference_job_s=ref, trace_jobs=n_jobs, reference_jobs=n_ref, total_process_hours=total_h,
        per_seed_process_hours=per_seed_h,
        wall_hours_launch_all_on_2_cpus=per_seed_h + per_seed_h / 2,
        plan_estimate_process_hours=30,
        gate_idle_machine_reference_s=dict(critic_100k=403, clone_20k=32, check2_seed=2279),
        note=("measured on a shared machine, pinned to one CPU at nice 15 (load average at start {:.1f}, {:.1f}, "
              "{:.1f}); critic: 20 x the first 5,000-update chunk (compilation included each time); actor: the full "
              "20,000-step three-arm run of check f; 'launch all' runs seeds 0 and 1 side by side, then seed 2 split "
              "by world").format(*load0))
    timing["load_average_start"], timing["load_average_end"] = list(load0), list(os.getloadavg())
    T.log(f"projection: {job / 60:.1f} min per trace job, {ref / 60:.1f} min per reference job, "
          f"{total_h:.1f} process-hours in all, about {per_seed_h * 1.5:.1f} hours of wall time with 'launch all'")
    passed = all(checks[k]["passed"] for k in ("a_builds", "a2_pairs", "b_region_pull", "c_clone", "f_actor_protocol",
                                                "g_critic_protocol"))
    T.write_json(T.OUT / "checks_seed0.json", T.jsonable(dict(
        label=T.LABEL, kind="trace-checks-seed0", spec_sha256=spec_sha, code_manifest_sha256=man_sha, rep=REP,
        checks=checks, passed=bool(passed), cpus=cpu, seconds=time.perf_counter() - t_all,
        note=("no return of a trained actor and no signal at an actor's command was computed; check h reads the "
              "gate's critics only at next states"),
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))))
    T.write_json(T.OUT / "timing.json", T.jsonable(dict(
        label=T.LABEL, kind="trace-timing", spec_sha256=spec_sha, code_manifest_sha256=man_sha, rep=REP,
        timing=timing, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))))
    T.log(f"seed-0 checks {'PASSED' if passed else 'FAILED'}")
    T.begin()
    return passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
