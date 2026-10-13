"""EXPLORATORY, POST HOC (spec v3: chosen after seeing the 0.5% and 2% results): the small checks of the trace test
bed's extension to the shares 1%, 3%, 5% and 10%. No BCA. One process, pinned to one CPU (it refuses otherwise), run
at nice 15. No return of a trained actor and no signal score of a real job is computed. The smoke step's numbers come
from tiny settings in a scratch folder; they are never printed and the folder is deleted.

  (a) build the seed-0 logs of the four post-hoc shares in both worlds (run_job.phase_build: their official
      trace_rows.json) and verify, per world and share: the count rule (floor(share x N + 0.5) of the N shared
      candidates, training and bank), the build checks, kept rows = the gate's kept rows plus the added rows, past-80%
      kept rows = the gate's plus the added rows, prepared rows = kept rows, the post-hoc label. Rebuild the spec-v2
      logs (share 0, 0.5%, 2%) in memory, without writing, and compare them with their saved records. Then the nesting
      of all six shares (each share's added rows and bank rows inside the next larger share's), that every share's
      added rows are exactly the first rows of the seed's one permutation (recomputed here independently), and that
      both worlds add back the same rows. (a2) T1's 1,500 rows are kept in every new log and equal spec v2's T1 rows.
  (b) one smoke step: the whole job chain (build, train, eval, signals) on the 10% log of the stalling world with tiny
      settings (200 critic updates, 200 clone steps, 200 actor steps, 20 evaluation starts), written to a scratch
      folder, checked for its files, labels and lock, then deleted; the phases' own progress lines are not printed. (b2) the refusals: a spec-v2 share is not a job
      under spec v3, guard_out refuses writes into spec-v2 outputs, launch.sh refuses a spec-v2 share and two equal
      CPUs, and (code manifest version 4) every launch naming CPU 2 or 3, which run the stand-ins. The launch
      refusals run dry: a stub interpreter, a scratch output folder and a taskset on PATH that records the CPU each
      call asks for and pins nothing, so a missing refusal could not start a real job; each must exit 2 with no job
      call and no CPU 2 or 3 asked for. (b3) the pooled look and the map over the saved spec-v2 jobs reproduce pooled.json's entries exactly; the
      post-hoc jobs are reported missing (none has run).
  (c) timing: the critic's first 5,000-update chunk and the stand-ins' set-up, T4 scores and one 1,500-pair score
      call on the 0.5% and the 10% logs in this process (the ratio between them is the share's own effect, whatever
      the load), against spec v2's timing.json (the load factor now); then the projected cost of the 24 post-hoc jobs
      from the spec-v2 trace jobs' measured wall times.
Writes checks_v3_seed0.json and timing_v3.json.

Usage (from /home/user/bca): nice -n 15 taskset -c <cpu> python checks_v3.py
"""
import copy
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracebed as T  # noqa: E402
import signals as S  # noqa: E402
import run_job as R  # noqa: E402

L, TC, DS = T.L, T.TC, T.DS
REP = 0
SMOKE_DIR = T.HERE / "dev" / "smoke_v3_out"
SMOKE_SHARE = 0.1
SHARE_OF = {v: k for k, v in T.SHARE_DIR.items()}


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


def permutation_prefix(shared, share, tags):
    """Independent recomputation of the selection: the first floor(share x N + 0.5) entries of
    default_rng(tags).permutation(N) over the shared candidates, sorted."""
    perm = np.random.default_rng([int(t) for t in tags]).permutation(len(shared))
    k = int(np.floor(float(share) * len(shared) + 0.5))
    return np.sort(np.asarray(shared, np.int64)[perm[:k]])


# ------------------------------------------------------------------------------------------------ (a) builds

def check_builds(g, spec_sha, man_sha):
    built = {}
    for world in T.WORLDS:
        for share in T.POST_HOC_SHARES:
            built[(world, share)] = R.phase_build(g, world, REP, share, spec_sha, man_sha)
    saved_v2 = {(w, s): T.read_json(T.job_dir(w, REP, s) / "trace_rows.json") for w in T.WORLDS for s in T.FROZEN_SHARES}
    out, rows_by = {}, {}
    for world in T.WORLDS:
        sd = g.sd[world]
        lg = sd.log
        per = {}
        # the spec-v2 logs, rebuilt in memory and compared with their saved records (nothing is written)
        for share in T.FROZEN_SHARES:
            rec = saved_v2[(world, share)]
            tr = T.build_trace(g.sd, world, share)
            same = dict(
                added_rows=DS.sha256_array(tr.added_rows) == rec["added_rows_sha256"],
                added_bank_rows=DS.sha256_array(tr.added_bank) == rec["added_bank_rows_sha256"],
                kept_rows=DS.sha256_array(tr.removal.kept_train_rows(lg)) == rec["kept_rows_sha256"],
                bank_rows=DS.sha256_array(tr.removal.bank_rows(lg)) == rec["bank_rows_sha256"],
                shared_candidates=DS.sha256_array(tr.shared) == rec["shared_candidates_sha256"])
            per[T.SHARE_DIR[share]] = dict(origin=T.share_origin(share), spec_v2_record_reproduced=same,
                                           added=rec["added"], bank_added=rec["bank_added"],
                                           shared_candidates=rec["shared_candidates"],
                                           kept_training_rows=rec["kept_training_rows"],
                                           past80_kept_rows=rec["past80_kept_rows"],
                                           passed=bool(all(same.values()) and rec["passed"]))
            rows_by[(world, share)] = (np.asarray(rec["added_rows"], np.int64),
                                       np.asarray(rec["added_bank_rows"], np.int64), tr.shared, tr.shared_bank)
        for share in T.POST_HOC_SHARES:
            r = built[(world, share)]
            n, nb = r["shared_candidates"], r["shared_bank_candidates"]
            tr = T.build_trace(g.sd, world, share)
            prefix_ok = bool(np.array_equal(permutation_prefix(tr.shared, share, T.selection_tags(REP)),
                                            np.asarray(r["added_rows"], np.int64))
                             and np.array_equal(permutation_prefix(tr.shared_bank, share, T.selection_tags(REP, True)),
                                                np.asarray(r["added_bank_rows"], np.int64)))
            p = dict(
                origin=T.share_origin(share), share_label=r.get("share_label"),
                candidates=r["candidates"], shared_candidates=n, world_specific_candidates=r["world_specific_candidates"],
                added=r["added"], expected_added=T.trace_count(share, n),
                bank_added=r["bank_added"], expected_bank_added=T.trace_count(share, nb),
                realised_share_of_shared=r["realised_share_of_shared_candidates"],
                realised_share_within_half_a_row=bool(abs(r["added"] - share * n) <= 0.5),
                kept_training_rows=r["kept_training_rows"], gate_kept_training_rows=r["gate_kept_training_rows"],
                kept_equals_gate_plus_added=bool(r["kept_training_rows"] == r["gate_kept_training_rows"] + r["added"]),
                past80_kept_rows=r["past80_kept_rows"], past80_kept_share=r["past80_kept_share"],
                gate_past80_kept_rows=r["gate_past80_kept_rows"],
                past80_rows_added_equals_added=bool(r["past80_kept_rows"] == r["gate_past80_kept_rows"] + r["added"]),
                prepared_rows=r["prepared"]["rows"],
                prepared_rows_equals_kept=bool(r["prepared"]["rows"] == r["kept_training_rows"]),
                added_rows_recorded_command_same_in_both_worlds=r["added_rows_recorded_command_same_in_both_worlds"],
                first_rows_of_the_permutation=prefix_ok, build_checks=r["checks"])
            p["passed"] = bool(r["passed"] and p["added"] == p["expected_added"]
                               and p["bank_added"] == p["expected_bank_added"] and p["realised_share_within_half_a_row"]
                               and p["kept_equals_gate_plus_added"] and p["past80_rows_added_equals_added"]
                               and p["prepared_rows_equals_kept"] and prefix_ok
                               and r.get("share_label") == T.POST_HOC_LABEL)
            per[T.SHARE_DIR[share]] = p
            rows_by[(world, share)] = (np.asarray(r["added_rows"], np.int64), np.asarray(r["added_bank_rows"], np.int64),
                                       tr.shared, tr.shared_bank)
        order = sorted(T.SHARES)
        nesting = {f"{T.SHARE_DIR[a]} in {T.SHARE_DIR[b]}": bool(
            np.isin(rows_by[(world, a)][0], rows_by[(world, b)][0]).all()
            and np.isin(rows_by[(world, a)][1], rows_by[(world, b)][1]).all())
            for a, b in zip(order[:-1], order[1:])}
        same_pool = bool(all(np.array_equal(rows_by[(world, s)][2], rows_by[(world, order[0])][2]) for s in order))
        out[world] = dict(shares=per, nesting=nesting, same_selection_pool_at_every_share=same_pool,
                          passed=bool(all(v["passed"] for v in per.values()) and all(nesting.values()) and same_pool))
        T.log(f"(a) {world}: " + "; ".join(
            f"{k}: {v['added']}/{v['shared_candidates']} added (bank {v['bank_added']}), {v['kept_training_rows']} rows, "
            f"{v['past80_kept_rows']} past 80%" for k, v in sorted(per.items(), key=lambda kv: SHARE_OF[kv[0]])))
        T.log(f"(a) {world}: nested {all(nesting.values())}; spec-v2 records reproduced "
              f"{all(per[T.SHARE_DIR[s]]['passed'] for s in T.FROZEN_SHARES)}")
    paired = {T.SHARE_DIR[s]: bool(np.array_equal(rows_by[("stall", s)][0], rows_by[("no_stall", s)][0])
                                   and np.array_equal(rows_by[("stall", s)][1], rows_by[("no_stall", s)][1]))
              for s in T.FROZEN_SHARES + T.POST_HOC_SHARES}
    out["paired_same_rows_in_both_worlds"] = paired
    out["passed"] = bool(all(out[w]["passed"] for w in T.WORLDS) and all(paired.values()))
    T.log(f"(a) both worlds add back the same rows at every share: {all(paired.values())}")
    return out, built


def check_t1(g, built):
    rows1, pool1 = S.t1_rows(g.sd, REP)
    sig_v2 = T.read_json(T.job_dir("stall", REP, T.V2_SHARES[0]) / "signals.json")
    kept_in_every_log = {}
    for world in T.WORLDS:
        for share in T.POST_HOC_SHARES:
            tr = T.build_trace(g.sd, world, share)
            kept_in_every_log[f"{world}/{T.SHARE_DIR[share]}"] = bool(
                np.isin(rows1, tr.removal.kept_train_rows(g.sd[world].log)).all())
    out = dict(t1_rows=int(len(rows1)), t1_shared_pool=pool1, t1_rows_sha256=DS.sha256_array(rows1),
               equal_to_spec_v2_t1_rows=bool(DS.sha256_array(rows1) == sig_v2["readout"]["t1_rows_sha256"]),
               kept_in_every_new_log=kept_in_every_log)
    out["passed"] = bool(out["t1_rows"] == S.T1_STATES and out["equal_to_spec_v2_t1_rows"]
                         and all(kept_in_every_log.values()))
    T.log(f"(a2) T1: {len(rows1)} rows, equal to spec v2's {out['equal_to_spec_v2_t1_rows']}, kept in every new log "
          f"{all(kept_in_every_log.values())}")
    return out


# ------------------------------------------------------------------------------------------------ (b) smoke

def smoke(g, official_10pct):
    """The whole job chain on the 10% log of the stalling world with tiny settings, in a scratch folder (deleted)."""
    small = replace(L.LearnSettings(), updates=200, chunk=100, clone_steps=200, actor_only_steps=200,
                    actor_only_chunk=100)
    gs = copy.copy(g)
    gs.learn = small
    gs.cfg = replace(g.cfg, bank_size=20, learn=small)
    real_out, real_log = T.OUT, T.log
    quiet = []
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True)
    t = time.perf_counter()
    try:
        T.OUT = SMOKE_DIR
        T.log = lambda *a: quiet.append(1)   # the phases' own lines (tiny-setting losses, flag levels) are not printed
        tag = "smoke-not-a-result"
        R.phase_build(gs, "stall", REP, SMOKE_SHARE, tag, tag)
        R.phase_train(gs, "stall", REP, SMOKE_SHARE, tag, tag)
        R.phase_eval(gs, "stall", REP, SMOKE_SHARE, tag, tag)
        R.phase_signals(gs, "stall", REP, SMOKE_SHARE, tag, tag)
        d = T.job_dir("stall", REP, SMOKE_SHARE)
        files = sorted(p.name for p in d.iterdir())
        rows, lock = T.read_json(d / "trace_rows.json"), T.read_json(d / "lock.json")
        values, sig = T.read_json(d / "values.json"), T.read_json(d / "signals.json")
        _, actors, critic = R.read_lock(d, tag)
        arrays = T.load_npz(d / "signals.npz", sig["arrays_sha256"])
        labels = {k: v.get("share_label") for k, v in dict(trace_rows=rows, lock=lock, values=values,
                                                           signals=sig).items()}
        out = dict(
            what=("build, train, eval and signals on the 10% log of the stalling world, seed 0, with tiny settings "
                  "(200 critic updates, 200 clone steps, 200 actor steps, 20 evaluation starts), in a scratch folder; "
                  "no number is kept or printed; the folder is deleted"),
            folder=d.name, files=files, labels=labels,
            labels_post_hoc=bool(all(v == T.POST_HOC_LABEL for v in labels.values())),
            added_rows_equal_official_build=bool(rows["added_rows_sha256"] == official_10pct["added_rows_sha256"]),
            lock_verified=bool(critic is not None and sorted(actors) == sorted(T.EVAL_ACTORS)),
            situations=sorted(values["situations"]), actors_evaluated=sorted(values["evaluations"]),
            signal_sets_pairs={k: v["pairs"] for k, v in sig["readout"]["sets"].items()},
            signal_arrays=len(arrays), progress_lines_not_printed=len(quiet), seconds=time.perf_counter() - t)
        out["passed"] = bool(out["labels_post_hoc"] and out["added_rows_equal_official_build"] and out["lock_verified"]
                             and set(T.EVAL_ACTORS) <= set(out["actors_evaluated"])
                             and all(n > 0 for n in out["signal_sets_pairs"].values())
                             and {"values.json", "signals.json", "lock.json", "actors.msgpack"} <= set(files))
    finally:
        T.OUT, T.log = real_out, real_log
        shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    out["scratch_folder_deleted"] = not SMOKE_DIR.exists()
    T.log(f"(b) smoke on {T.SHARE_DIR[SMOKE_SHARE]}: passed {out['passed']} ({out['seconds']:.0f} s; files "
          f"{len(out['files'])}; pairs per set {out['signal_sets_pairs']})")
    return out


def refusals(spec, cpu):
    out = {}
    for share in T.FROZEN_SHARES:
        try:
            R.check_job(spec, "stall", REP, share)
            out[f"check_job_{T.SHARE_DIR[share]}"] = False
        except SystemExit as e:
            out[f"check_job_{T.SHARE_DIR[share]}"] = "spec-v2 job" in str(e)
    for name, path in (("job_folder_2pct", T.job_dir("stall", REP, 0.02) / "x.json"),
                       ("job_folder_0pct", T.job_dir("no_stall", 2, 0.0) / "values.json"),
                       ("pooled_json", T.OUT / T.POOLED_V2_NAME), ("timing_json", T.OUT / "timing.json"),
                       ("checks_seed0_json", T.OUT / "checks_seed0.json")):
        try:
            T.guard_out(path)
            out[f"guard_{name}"] = False
        except SystemExit as e:
            out[f"guard_{name}"] = "spec-v2 output" in str(e)
    T.guard_out(T.job_dir("stall", REP, 0.1) / "x.json")   # a post-hoc folder is writable
    out["guard_allows_post_hoc_folder"] = True
    dry = Path(tempfile.mkdtemp(prefix="launch_dry_", dir=T.HERE / "dev"))
    try:
        calls, pins = dry / "calls.txt", dry / "pins.txt"
        (dry / "bin").mkdir()
        for path, text in ((dry / "py", f'#!/bin/sh\necho "$*" >> "{calls}"\nexit 0\n'),
                           (dry / "bin" / "taskset",
                            f'#!/bin/sh\n[ "$1" = -c ] || exit 9\necho "$2" >> "{pins}"\nshift 2\nexec "$@"\n')):
            path.write_text(text)
            path.chmod(0o755)
        env = dict(os.environ, NICE="15", PY=str(dry / "py"), OUTDIR=str(dry / "out"),
                   PATH=f"{dry / 'bin'}:{os.environ.get('PATH', '')}")
        for name, cases in (("launch_refuses_v2_share", [["0", str(cpu), "stall", "0.02"]]),
                            ("launch_refuses_equal_cpus", [["all", str(cpu), str(cpu)]]),
                            ("launch_refuses_cpu_range", [["0", "0-3"]]),
                            ("launch_refuses_standin_cpus_2_and_3",
                             [["0", "2", "stall", "0.01"], ["1", "3", "no_stall", "0.1"], ["pooled", "2"],
                              ["all", "3"], ["all", "0", "3"], ["all", "1", "2"], ["all", "2", "1"]])):
            ok = True
            for args in cases:
                p = subprocess.run(["sh", str(T.HERE / "launch.sh")] + args, env=env, capture_output=True,
                                   text=True, timeout=60)
                ok = ok and p.returncode == 2
            out[name] = bool(ok)
        asked = pins.read_text().split() if pins.exists() else []
        out["launch_dry_no_job_called"] = not calls.exists()
        out["launch_dry_no_cpu_2_or_3_asked_for"] = not (set(asked) & {"2", "3"})
    finally:
        shutil.rmtree(dry, ignore_errors=True)
    out["passed"] = bool(all(out.values()))
    T.log(f"(b2) refusals: {out}")
    return out


def pooled_v2(spec, spec_sha):
    jobs, missing = R.collect_jobs(spec, spec_sha, list(T.FROZEN_SHARES))
    if missing:
        raise SystemExit(f"refused: spec-v2 jobs missing: {missing}")
    out = R.pool_jobs(jobs, list(T.FROZEN_SHARES))
    v2 = T.read_json(T.OUT / T.POOLED_V2_NAME)
    equal, same = R.v2_entries_check(out, v2)
    the_map = R.share_map(out, list(T.FROZEN_SHARES))
    for line in R.map_lines(the_map):
        T.log("(b3) " + line)
    _, post_missing = R.collect_jobs(spec, spec_sha, list(T.POST_HOC_SHARES))
    res = dict(entries_equal_pooled_json=equal, map_rows=len(the_map["rows"]),
               map_of_spec_v2_shares=the_map, post_hoc_jobs_missing=len(post_missing),
               passed=bool(same and len(the_map["rows"]) == len(T.FROZEN_SHARES)
                           and len(post_missing) == len(T.POST_HOC_SHARES) * len(T.SEEDS) * len(T.WORLDS)))
    T.log(f"(b3) pooled look of the spec-v2 jobs equals pooled.json: {equal}; post-hoc jobs not yet run: "
          f"{len(post_missing)}")
    return res


# ------------------------------------------------------------------------------------------------ (c) timing

def v2_job_walls():
    """The spec-v2 trace jobs' wall times (s) from their launch logs (start to done of build train eval signals)."""
    walls, refs = [], []
    pat = re.compile(r"^(\S+) (start|done): (--world \S+ --seed \d+ --share (\S+) --phase .*?)(?: \(cpu.*)?$")
    for f in sorted((T.OUT / "logs").glob("seed*.log")):
        started = {}
        for line in f.read_text().splitlines():
            m = pat.match(line)
            if not m:
                continue
            ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ"))
            if m.group(2) == "start":
                started[m.group(3)] = ts
            elif m.group(3) in started:
                (refs if float(m.group(4)) == 0.0 else walls).append(ts - started.pop(m.group(3)))
    return walls, refs


def timing(g, built, cpu):
    load0 = os.getloadavg()
    v2t = T.read_json(T.OUT / "timing.json")["timing"]
    sd = g.sd["stall"]
    meas = {}
    for share in (T.V2_SHARES[0], SMOKE_SHARE):
        tr = T.build_trace(g.sd, "stall", share)
        prep, t_prep = timed(T.trace_prepared, sd, tr, g.learn)
        (krec, kstate), t_critic = timed(L.fixed_critic, prep, T.CONTROLLER, g.learn, REP, updates=g.learn.chunk)
        clone, _ = g.clone("stall")
        si, t_si = timed(S.StandIns, sd, prep, kstate.critic.params, clone, g.learn, REP)
        rows4 = S.t4_rows(sd)
        _, t_t4 = timed(si.scores, sd.log["infos/state"][rows4], np.asarray(sd.log["actions"][rows4], np.float32))
        rows1, _ = S.t1_rows(g.sd, REP)
        x1 = sd.log["infos/state"][rows1]
        _, t_sc = timed(si.scores, x1, np.asarray(sd.log["actions"][rows1], np.float32))
        meas[T.SHARE_DIR[share]] = dict(training_rows=int(prep.rows.size), prepare_s=t_prep, critic_first_chunk_s=t_critic,
                                        standins_setup_with_noise_draws_s=t_si, t4_scores_5000_s=t_t4,
                                        scores_1500_s=t_sc)
        T.log(f"(c) {T.SHARE_DIR[share]}: {prep.rows.size} training rows; critic chunk {t_critic:.1f} s, stand-ins "
              f"{t_si:.1f} s, T4 {t_t4:.1f} s, 1,500 scores {t_sc:.1f} s")
    a, b = meas[T.SHARE_DIR[T.V2_SHARES[0]]], meas[T.SHARE_DIR[SMOKE_SHARE]]
    keys = ("critic_first_chunk_s", "standins_setup_with_noise_draws_s", "t4_scores_5000_s", "scores_1500_s")
    share_ratio = {k: b[k] / a[k] for k in keys}
    load_factor = {k: a[k] / v2t[k] for k in keys if v2t.get(k)}
    walls, refs = v2_job_walls()
    job_s = statistics.median(walls)
    # the critic, clone and actor steps are fixed in number; only the signals' set-up and the neighbour search grow
    # with the training rows, so the share's effect is the critic and signal ratios weighted by their part of a job
    # (spec v2's projected job: timing.json)
    v2job = v2t["projection"]["per_trace_job_s"]
    critic_part = v2job["critic_100k"] / v2job["total"]
    signals_part = v2job["signals"] / v2job["total"]
    scale = 1.0 + critic_part * (share_ratio["critic_first_chunk_s"] - 1.0) + signals_part * (
        statistics.mean([share_ratio[k] for k in keys[1:]]) - 1.0)
    per_job = job_s * max(scale, 1.0)
    n_jobs = len(T.POST_HOC_SHARES) * len(T.WORLDS) * len(T.SEEDS)
    jobs_per_cpu = n_jobs // 2
    pooled_s = 60.0
    proj = dict(
        share_factor=scale, parts_of_a_job=dict(critic=critic_part, signals=signals_part),
        spec_v2_trace_job_wall_s=dict(n=len(walls), median=job_s, mean=statistics.mean(walls), min=min(walls),
                                      max=max(walls)),
        spec_v2_reference_job_wall_s=dict(n=len(refs), median=statistics.median(refs)) if refs else None,
        share_ratio_10pct_over_0p5pct=share_ratio, load_factor_now_over_spec_v2_timing=load_factor,
        per_trace_job_s=per_job, trace_jobs=n_jobs, reference_jobs=0,
        total_process_hours=n_jobs * per_job / 3600.0,
        wall_hours_launch_all_on_2_free_cpus=(jobs_per_cpu * per_job + pooled_s) / 3600.0,
        wall_hours_launch_all_on_1_free_cpu=(n_jobs * per_job + pooled_s) / 3600.0,
        wall_hours_launch_all_on_2_cpus_shared_like_now=(jobs_per_cpu * per_job * statistics.mean(load_factor.values())
                                                         + pooled_s) / 3600.0,
        note=("per job: the median wall time of spec v2's 12 trace jobs (each pinned alone to one CPU, nice 15, on a "
              "shared machine), scaled by the measured change from the 0.5% to the 10% log (the critic, clone and "
              "actor steps are fixed in number); 'launch all' on two CPUs gives each CPU 12 jobs (8 for seeds 0 and 1 "
              "side by side, then 4 for seed 2 split by world), then the pooled look. The load factor compares the same "
              "steps now (on CPU {}, machine load average {:.1f} at the start of the timing) with spec v2's "
              "timing.json; if CPUs 0 and 1 are as loaded when the sweep starts as CPU {} is now, the wall time "
              "changes by about that factor").format(cpu[0], load0[0], cpu[0]))
    T.log(f"(c) projection: {per_job / 60:.1f} min per trace job (spec v2 median {job_s / 60:.1f} min x share factor "
          f"{scale:.3f}); {n_jobs} jobs, {proj['total_process_hours']:.1f} process-hours; 'launch all' on 2 free CPUs "
          f"about {proj['wall_hours_launch_all_on_2_free_cpus']:.1f} h of wall time "
          f"({proj['wall_hours_launch_all_on_2_cpus_shared_like_now']:.1f} h if shared like now)")
    return dict(measured=meas, cpus=cpu, load_average_start=list(load0), load_average_end=list(os.getloadavg()),
                spec_v2_timing=dict((k, v2t[k]) for k in keys), projection=proj)


# ------------------------------------------------------------------------------------------------ main

def main():
    cpu = T.require_one_cpu("the spec-v3 checks")
    spec, spec_sha, man_sha = T.begin()
    if not spec.get("post_hoc_extension"):
        raise SystemExit("refused: checks_v3.py needs spec v3 (post_hoc_extension)")
    t_all = time.perf_counter()
    g = R.load("stall", REP, spec)
    T.log(f"loaded seed {REP}, both worlds, in {g.load_seconds:.1f} s (CPU {cpu[0]}, nice {os.nice(0)})")
    checks = {}
    checks["a_builds"], built = check_builds(g, spec_sha, man_sha)
    checks["a2_t1"] = check_t1(g, built)
    checks["b_smoke"] = smoke(g, built[("stall", SMOKE_SHARE)])
    checks["b2_refusals"] = refusals(spec, cpu[0])
    checks["b3_pooled_v2_reproduced"] = pooled_v2(spec, spec_sha)
    tim = timing(g, built, cpu)
    tim["load_one_job_s"] = g.load_seconds
    passed = bool(all(c["passed"] for c in checks.values()))
    T.write_json(T.OUT / "checks_v3_seed0.json", T.jsonable(dict(
        label=T.LABEL, post_hoc_label=T.POST_HOC_LABEL, kind="trace-checks-v3-seed0", spec_sha256=spec_sha,
        code_manifest_sha256=man_sha, rep=REP, checks=checks, passed=passed, cpus=cpu, nice=os.nice(0),
        seconds=time.perf_counter() - t_all,
        note=("no return of a trained actor and no signal score of a real job was computed; the smoke step's numbers "
              "came from tiny settings in a scratch folder, were not printed and were deleted; (b3) re-pools the "
              "spec-v2 results already seen"),
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))))
    T.write_json(T.OUT / "timing_v3.json", T.jsonable(dict(
        label=T.LABEL, post_hoc_label=T.POST_HOC_LABEL, kind="trace-timing-v3", spec_sha256=spec_sha,
        code_manifest_sha256=man_sha, rep=REP, timing=tim, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))))
    T.log(f"spec-v3 checks {'PASSED' if passed else 'FAILED'}: " + ", ".join(f"{k} {v['passed']}" for k, v in checks.items()))
    T.begin()
    return passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
