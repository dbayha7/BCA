"""EXPLORATORY trace test bed (plan section 4.6): one job = one world, one gate seed, one share. No BCA. Outside the
gate. Every phase refuses unless spec.json exists, every gate file still has the SHA-256 the spec recorded, and every
script here and its copy match runs/shift_explore/trace/code_manifest.json, which must name the current spec.

Phases of a trace job (share 0.005 or 0.02), in order, each in runs/shift_explore/trace/<world>/seed<k>/<share>/:
  build      the trace removal (tracebed.build_trace: the same rows in both worlds, drawn from the candidates both
             worlds share) -> trace_rows.json: the exact added rows, their SHA-256, the row counts, the world-specific
             candidates (never added) and the build checks. A failed check writes stop.json.
  train      the cloned actor (learners.clone), the stall-aware controller's fixed-controller critic
             (learners.fixed_critic, 100,000 updates), then check 2's actor-only protocol (learners.ActorOnly.run,
             unweighted, alpha 5, from that clone, on the trace log's rows) with three arms side by side: the trace
             critic, the gate's full-log critic and the gate's gapped critic (both read-only). Then lock.json with
             every SHA-256. No return and no signal is computed before the lock exists. Resumable: a saved clone or
             critic (verified by SHA-256) is reused.
  eval       verifies the lock; exact returns of the cloned actor and the three actors (acceptance._evaluate: the full
             1,000-start banks of the four situations, the world's anchors, capped at 80%, the share past 80%) and
             the trace critic's pull past 80% on check 1's frozen region pool -> values.json.
  signals    verifies the lock; the detection readout with the data-only stand-ins (signals.readout) ->
             signals.json and signals.npz.
Reference job (share 0, the gate's own gapped log; no training), in <world>/seed<k>/share_0pct/:
  reference  the share-0 build check (the removal and prepared data equal the gate's), check 2's saved points of its
             cloned, gapped-critic and full-log-critic actors, check 1's saved pull past 80%, and the detection
             readout on the gate's gapped critic, clone and check 2's gapped-critic actor -> trace_rows.json,
             values.json, signals.json, signals.npz.
Pooled (all jobs):
  pooled     Q1 and Q2 per share over seeds 0-2. Refuses until every job has values.json and signals.json, or
             stop.json. Spec v2: -> pooled.json. Spec v3: -> pooled_v3.json, every share 0-10% side by side (its
             entries for share 0, 0.5% and 2% must equal pooled.json's exactly) and a short map over every share
             (share_map: harm, T1 selectivity of every signal, usable yes or no), each share labelled reference,
             planned (spec v2) or post hoc (spec v3). pooled.json is kept as it is.

Spec v3 (POST HOC: chosen after seeing the 0.5% and 2% results) adds the shares 0.01, 0.03, 0.05 and 0.1 with the
same phases. The spec-v2 jobs (share 0, 0.005, 0.02) are complete and kept as they are: every job phase refuses them
(check_job), and their outputs are read with spec v2's SHA-256 (tracebed.share_spec_sha256). Every record of a
post-hoc job carries share_label = the post-hoc label.

Every job phase refuses unless the process is pinned to ONE CPU (tracebed.require_one_cpu): the results depend on how
many threads the computation uses; pinned to one CPU, the actor-only call reproduces check 2's actors bit for bit, and
on 4 CPUs it does not (review finding 2; seed-0 check f). launch.sh pins with taskset; direct use must too.
A job loads both worlds of its seed (the trace rows and T1 states are paired across worlds, spec v2).

Usage (from /home/user/bca; normally through launch.sh, which allows only CPU 0 or CPU 1: CPUs 2 and 3 run the
stand-ins and must not be used):
  nice -n 15 taskset -c 1 python run_job.py --world stall --seed 0 --share 0.01 --phase build train eval signals
  nice -n 15 taskset -c 1 python run_job.py --phase pooled
(Under spec v3 every phase refuses the spec-v2 shares 0, 0.005 and 0.02, the reference phase included.)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracebed as T  # noqa: E402
import signals as S  # noqa: E402

A, L, TC = T.A, T.L, T.TC
SITUATIONS = TC.SITUATIONS


class Stop(SystemExit):
    pass


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stop(d, reason, spec_sha, man_sha, **fields):
    T.write_json(d / "stop.json", dict(label=T.LABEL, kind="trace-stop", share_label=T.share_label(d_share(d)),
                                       stopped=True, reason=reason,
                                       spec_sha256=spec_sha, code_manifest_sha256=man_sha, written=now(), **fields))
    T.log(f"STOP: {reason}")
    raise Stop(2)


def d_share(d):
    """The share of a job folder (its name), for the stop record's label."""
    return {v: k for k, v in T.SHARE_DIR.items()}[Path(d).name]


def refuse_if_stopped(d):
    if (d / "stop.json").exists():
        raise SystemExit(f"refused: {d} has stopped (stop.json)")


def check_job(spec, world, rep, share):
    if world not in spec["worlds"] or int(rep) not in spec["seeds"] or float(share) not in (
            T.spec_shares(spec) + [float(spec["reference_share"])]):
        raise SystemExit(f"refused: {world}/seed {rep}/share {share} is not a job of the spec")
    ext = spec.get("post_hoc_extension")
    if ext and float(share) in [float(s) for s in ext["frozen_v2"]["shares"]]:
        raise SystemExit(f"refused: share {share} is a spec-v2 job, complete and kept as it is; spec v3 adds the "
                         f"post-hoc shares {ext['shares']} only")


# ------------------------------------------------------------------------------------------------ build

def load(world, rep, spec):
    """The seed's gate artifacts, both worlds (the trace rows and T1 are paired across worlds)."""
    g = T.GateSeed(rep, worlds=T.WORLDS)
    if T.A.canonical(T.jsonable(g.learn.as_dict())) != T.A.canonical(spec["training"]["settings"]):
        raise SystemExit("refused: the pre-registered learner settings differ from the spec's")
    return g


def rebuild(g, world, share, rows_rec):
    """The trace removal and prepared data, checked against trace_rows.json."""
    sd = g.sd[world]
    tr = T.build_trace(g.sd, world, share)
    prepared = T.trace_prepared(sd, tr, g.learn)
    if (T.DS.sha256_array(tr.added_rows) != rows_rec["added_rows_sha256"]
            or T.DS.sha256_array(tr.removal.kept_train_rows(sd.log)) != rows_rec["kept_rows_sha256"]
            or prepared.sha256 != rows_rec["prepared"]["sha256"]):
        raise SystemExit(f"refused: the rebuilt trace log differs from {rows_rec['share_dir']}/trace_rows.json")
    return sd, tr, prepared


def phase_build(g, world, rep, share, spec_sha, man_sha):
    d = T.job_dir(world, rep, share)
    refuse_if_stopped(d)
    sd = g.sd[world]
    tr = T.build_trace(g.sd, world, share)
    rec = T.trace_record(g.sd, tr)
    prepared = T.trace_prepared(sd, tr, g.learn)
    rec["prepared"] = prepared.describe()
    if share == 0.0:
        gp = g.gapped_prepared(world)
        rec["checks"]["share0_prepared_equals_gate_gapped"] = bool(prepared.sha256 == gp.sha256)
        rec["passed"] = bool(all(rec["checks"].values()))
    payload = dict(label=T.LABEL, kind="trace-rows", share_label=T.share_label(share), spec_sha256=spec_sha,
                   code_manifest_sha256=man_sha, **rec)
    old = T.read_json(d / "trace_rows.json")
    if old is not None:
        volatile = ("written", "code_manifest_sha256")
        if (T.A.canonical({k: v for k, v in old.items() if k not in volatile})
                != T.A.canonical({k: v for k, v in T.jsonable(payload).items() if k not in volatile})):
            raise SystemExit(f"refused: {d}/trace_rows.json exists and differs from the rebuilt trace log")
        T.log(f"build {world}/seed {rep}/{T.SHARE_DIR[share]}: trace_rows.json already written and identical")
        return old
    if not rec["passed"]:
        T.write_json(d / "trace_rows.json", T.jsonable(dict(payload, written=now())))
        stop(d, "a build check failed: " + ", ".join(k for k, v in rec["checks"].items() if not v), spec_sha, man_sha)
    T.write_json(d / "trace_rows.json", T.jsonable(dict(payload, written=now())))
    T.log(f"build {world}/seed {rep}/{T.SHARE_DIR[share]}: {rec['added']} of the {rec['shared_candidates']} removed "
          f"past-80% rows both worlds share added back, the same rows in both worlds "
          f"({rec['world_specific_candidates']} world-specific never added; {rec['kept_training_rows']} training rows, "
          f"{rec['past80_kept_rows']} past 80%)")
    return payload


def rows_record(d, spec_sha):
    rec = T.read_json(d / "trace_rows.json")
    if rec is None:
        raise SystemExit(f"refused: no trace_rows.json in {d}; run the build phase first")
    if rec["spec_sha256"] != spec_sha:
        raise SystemExit(f"refused: {d}/trace_rows.json was written under another spec")
    if not rec["passed"]:
        raise SystemExit(f"refused: {d}/trace_rows.json failed its build checks")
    return rec


# ------------------------------------------------------------------------------------------------ train

def train_actors(prepared, clone, named, learn, rep, log=None):
    """Check 2's actor-only protocol with the named critics side by side (critic swap, unweighted, alpha 5), exactly
    as phase_train runs it: named = [(arm name, critic params or None)]; a None critic gets no arm. Must run pinned to
    one CPU (seed-0 check f: with the gate's gapped critic in the 'trace' slot on the share-0 log it reproduces check
    2's gapped-critic and full-log-critic actors bit for bit). Returns ({name: actor}, {name: failed}, info, seconds)."""
    T.require_one_cpu("the actor-only protocol")
    ok = [(n, c) for n, c in named if c is not None]
    failed = {n: c is None for n, c in named}
    t = time.perf_counter()
    ao = L.ActorOnly(prepared, clone, learn, rep, alpha=T.ALPHA)
    stacked, info = ao.run([L.Arm(n, critic=i) for i, (n, _) in enumerate(ok)], [c for _, c in ok], weighted=False,
                           log=log)
    actors = {}
    for i, (n, _) in enumerate(ok):
        actors[n] = ao.arm_params(stacked, i)
        failed[n] = bool(info["failed"][i])
    info["arms"] = [n for n, _ in ok]
    return actors, failed, info, time.perf_counter() - t


def phase_train(g, world, rep, share, spec_sha, man_sha):
    d = T.job_dir(world, rep, share)
    refuse_if_stopped(d)
    if (d / "lock.json").exists():
        raise SystemExit(f"refused: {d}/lock.json exists; the job is trained and locked")
    rows_rec = rows_record(d, spec_sha)
    sd, tr, prepared = rebuild(g, world, share, rows_rec)
    learn = g.learn
    started = time.perf_counter()
    # the cloned actor of the trace log
    crec = T.read_json(d / "clone.json")
    if crec is not None and crec["spec_sha256"] == spec_sha:
        clone = L.load_tree(d / "clone.msgpack", L.actor_template(), crec["file_sha256"])
        T.log("train: the saved cloned actor is reused (SHA-256 checked)")
    else:
        t = time.perf_counter()
        clone, result = L.clone(prepared, learn, rep, log=T.log)
        crec = dict(label=T.LABEL, kind="trace-clone", spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                    result=result, file_sha256=T.save_tree(d / "clone.msgpack", clone),
                    seconds=time.perf_counter() - t, written=now())
        T.write_json(d / "clone.json", T.jsonable(crec))
    # the trace critic: the stall-aware controller's fixed-controller critic on the trace log
    krec = T.read_json(d / "critic.json")
    if krec is not None and krec["spec_sha256"] == spec_sha:
        trace_critic = None if krec["result"]["failed"] else L.load_tree(
            d / "critic.msgpack", L.critic_template(), krec["result"]["file_sha256"])
        T.log("train: the saved trace critic is reused (SHA-256 checked)")
    else:
        t = time.perf_counter()
        result, state = L.fixed_critic(prepared, T.CONTROLLER, learn, rep, out_dir=T.guard_out(d), log=T.log)
        trace_critic = None if result["failed"] else state.critic.params
        krec = dict(label=T.LABEL, kind="trace-critic", spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                    result=result, seconds=time.perf_counter() - t, written=now())
        T.write_json(d / "critic.json", T.jsonable(krec))
    # check 2's actor-only protocol on the trace log, three critics side by side
    full, full_rec = g.critic(world, "none")
    gapped, gapped_rec = g.critic(world, "gap")
    named = [("trace", trace_critic), ("full", full), ("gapped", gapped)]
    trained, failed, info, actor_seconds = train_actors(prepared, clone, named, learn, rep, log=T.log)
    actors = {"clone": clone, **trained}
    actors_sha = T.save_tree(d / "actors.msgpack", actors)
    lock = dict(label=T.LABEL, kind="trace-lock", share_label=T.share_label(share), world=world, rep=int(rep),
                share=float(share),
                spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                trace_rows_sha256=T.file_sha256(d / "trace_rows.json"),
                clone_file_sha256=crec["file_sha256"], clone_actor_sha256=L.tree_sha256(clone),
                trace_critic_file_sha256=krec["result"].get("file_sha256"),
                trace_critic_sha256=krec["result"].get("critic_sha256"), trace_critic_failed=krec["result"]["failed"],
                reused_critics=dict(full=dict(file_sha256=full_rec["result"]["file_sha256"],
                                              critic_sha256=full_rec["result"]["critic_sha256"]),
                                    gapped=dict(file_sha256=gapped_rec["result"]["file_sha256"],
                                                critic_sha256=gapped_rec["result"]["critic_sha256"])),
                actors_file_sha256=actors_sha, actor_names=sorted(actors),
                actor_sha256={k: L.tree_sha256(v) for k, v in actors.items()}, failed=failed,
                actor_only=dict(rows_sha256=info["rows_sha256"], start_sha256=info["start_sha256"],
                                chunk_metrics=info["chunk_metrics"], metric_names=info["metric_names"],
                                arms=info["arms"], seconds=actor_seconds),
                cpus=T.require_one_cpu("the train phase"),
                seconds=time.perf_counter() - started, written=now(),
                note="written after training and before any return or signal is computed")
    T.write_json(d / "lock.json", T.jsonable(lock))
    T.log(f"train {world}/seed {rep}/{T.SHARE_DIR[share]}: locked ({', '.join(sorted(actors))})")
    return lock


def read_lock(d, spec_sha):
    lock = T.read_json(d / "lock.json")
    if lock is None:
        raise SystemExit(f"refused: no lock.json in {d}; run the train phase first")
    if lock["spec_sha256"] != spec_sha:
        raise SystemExit(f"refused: {d}/lock.json was written under another spec")
    if T.file_sha256(d / "trace_rows.json") != lock["trace_rows_sha256"]:
        raise SystemExit(f"refused: {d}/trace_rows.json changed after the lock")
    actors = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in lock["actor_names"]},
                         lock["actors_file_sha256"])
    if {k: L.tree_sha256(v) for k, v in actors.items()} != lock["actor_sha256"]:
        raise SystemExit(f"refused: {d}/actors.msgpack differs from the lock")
    critic = None
    if not lock["trace_critic_failed"]:
        critic = L.load_tree(d / "critic.msgpack", L.critic_template(), lock["trace_critic_file_sha256"])
    return lock, actors, critic


# ------------------------------------------------------------------------------------------------ eval

def points_table(ev, names):
    out = {}
    for s in SITUATIONS:
        p = {n: None if ev.get(n) is None else ev[n][s]["points"] for n in names}
        diff = lambda a, b: None if p.get(a) is None or p.get(b) is None else p[a] - p[b]  # noqa: E731
        out[s] = dict(points=p,
                      capped_points={n: None if ev.get(n) is None else ev[n][s]["capped_points"] for n in names},
                      unrecorded_throttle_share={n: None if ev.get(n) is None else
                                                 ev[n][s]["unrecorded_throttle_share"] for n in names},
                      loss_from_unrecorded_throttle={n: None if ev.get(n) is None else
                                                     ev[n][s]["loss_from_unrecorded_throttle"] for n in names},
                      harm=diff("full", "trace"), harm_gapped_critic=diff("full", "gapped"),
                      trace_minus_gapped=diff("trace", "gapped"), trace_minus_clone=diff("trace", "clone"))
    return out


def pulls(g, world, prepared, trace_critic):
    sd, h1 = g.sd[world], g.h1_rows()
    out = {k: T.summarise(h1[f"{world}:{T.CONTROLLER}:{k}|pull|region"]) for k in ("gap", "full")}
    out = {"gate_gapped_critic": out["gap"], "gate_full_log_critic": out["full"]}
    if trace_critic is not None:
        out["trace_critic"] = T.summarise(T.region_pull(trace_critic, sd, prepared, h1))
    out["true_price_of_pushing_past_80"] = T.summarise(h1[f"{world}:{T.CONTROLLER}:price|region"])
    return out


def phase_eval(g, world, rep, share, spec_sha, man_sha):
    d = T.job_dir(world, rep, share)
    refuse_if_stopped(d)
    if (d / "values.json").exists():
        raise SystemExit(f"refused: {d}/values.json exists")
    lock, actors, critic = read_lock(d, spec_sha)
    sd, tr, prepared = rebuild(g, world, share, rows_record(d, spec_sha))
    started = time.perf_counter()
    anchors = g.anchors(world)
    ev = {}
    for n in T.EVAL_ACTORS:
        failed = lock["failed"].get(n, False) or n not in actors
        ev[n] = A._evaluate(actors.get(n), prepared, anchors, g.cfg, failed)
        T.log(f"eval {world}/seed {rep}/{T.SHARE_DIR[share]}: {n} done")
    values = dict(label=T.LABEL, kind="trace-values", share_label=T.share_label(share), world=world, rep=int(rep),
                  share=float(share), spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                  lock_sha256=T.file_sha256(d / "lock.json"),
                  evaluations=ev, situations=points_table(ev, T.EVAL_ACTORS),
                  pull_past_80=pulls(g, world, prepared, critic),
                  cpus=T.cpus(), seconds=time.perf_counter() - started, written=now())
    T.write_json(d / "values.json", T.jsonable(values))
    T.log(f"eval {world}/seed {rep}/{T.SHARE_DIR[share]}: written")
    return values


# ------------------------------------------------------------------------------------------------ signals

def phase_signals(g, world, rep, share, spec_sha, man_sha):
    d = T.job_dir(world, rep, share)
    refuse_if_stopped(d)
    if (d / "signals.json").exists():
        raise SystemExit(f"refused: {d}/signals.json exists")
    lock, actors, critic = read_lock(d, spec_sha)
    if critic is None or lock["failed"].get("trace"):
        stop(d, "the trace critic or its actor failed; no detection readout", spec_sha, man_sha)
    sd, tr, prepared = rebuild(g, world, share, rows_record(d, spec_sha))
    started = time.perf_counter()
    summary, arrays = S.readout(g.sd, world, prepared, critic, actors["clone"], actors["trace"], g.learn, log_fn=T.log)
    sha = T.save_npz(d / "signals.npz", **arrays)
    T.write_json(d / "signals.json", T.jsonable(dict(
        label=T.LABEL, kind="trace-signals", share_label=T.share_label(share), world=world, rep=int(rep),
        share=float(share), spec_sha256=spec_sha, code_manifest_sha256=man_sha,
        lock_sha256=T.file_sha256(d / "lock.json"), arrays_sha256=sha,
        readout=summary, cpus=T.cpus(), seconds=time.perf_counter() - started, written=now())))
    T.log(f"signals {world}/seed {rep}/{T.SHARE_DIR[share]}: written")


# ------------------------------------------------------------------------------------------------ reference (share 0)

H2_NAMES = {"stall": dict(clone="clone", gapped="gapped", full="full"),
            "no_stall": dict(clone="no_stall_clone", gapped="no_stall_gapped", full="no_stall_full")}


def phase_reference(g, world, rep, spec_sha, man_sha):
    share = T.REFERENCE_SHARE
    d = T.job_dir(world, rep, share)
    refuse_if_stopped(d)
    if (d / "signals.json").exists():
        raise SystemExit(f"refused: {d} already holds the reference readout")
    rows_rec = phase_build(g, world, rep, share, spec_sha, man_sha)
    sd = g.sd[world]
    prepared = g.gapped_prepared(world)
    if prepared.sha256 != rows_rec["prepared"]["sha256"]:
        raise SystemExit("refused: the share-0 log is not the gate's gapped log")
    started = time.perf_counter()
    acts, h2rec, _ = g.h2()
    names = H2_NAMES[world]
    clone, clone_rec = g.clone(world)
    if L.tree_sha256(clone) != L.tree_sha256(acts[names["clone"]]):
        raise SystemExit("refused: the gate's saved clone differs from check 2's cloned actor")
    gapped, _ = g.critic(world, "gap")
    ev = {k: h2rec["evaluations"].get(v) for k, v in names.items()}
    ev["trace"] = ev["gapped"]   # at share 0 the trace critic is the gate's gapped critic
    values = dict(label=T.LABEL, kind="trace-values", world=world, rep=int(rep), share=share, spec_sha256=spec_sha,
                  code_manifest_sha256=man_sha, source="check 2's saved values (runs/shift, read-only); no training",
                  check2_values_sha256=T.file_sha256(A._variant_root(g.prereg, "harm_return", g.variant)
                                                     / f"alpha{T.ALPHA:g}" / f"seed{rep}" / "values.json"),
                  evaluations=ev, situations=points_table(ev, T.EVAL_ACTORS),
                  pull_past_80=pulls(g, world, prepared, None), written=now())
    values["pull_past_80"]["trace_critic"] = values["pull_past_80"]["gate_gapped_critic"]
    old = T.read_json(d / "values.json")
    if old is None:
        T.write_json(d / "values.json", T.jsonable(values))
    elif old["spec_sha256"] != spec_sha:
        raise SystemExit(f"refused: {d}/values.json was written under another spec")
    else:
        T.log(f"reference {world}/seed {rep}: values.json already written (kept); the readout follows")
    summary, arrays = S.readout(g.sd, world, prepared, gapped, clone, acts[names["gapped"]], g.learn, log_fn=T.log)
    sha = T.save_npz(d / "signals.npz", **arrays)
    T.write_json(d / "signals.json", T.jsonable(dict(
        label=T.LABEL, kind="trace-signals", world=world, rep=int(rep), share=share, spec_sha256=spec_sha,
        code_manifest_sha256=man_sha, source="the gate's gapped critic, its clone and check 2's gapped-critic actor",
        arrays_sha256=sha, readout=summary, cpus=T.cpus(), seconds=time.perf_counter() - started, written=now())))
    T.log(f"reference {world}/seed {rep}: written")


# ------------------------------------------------------------------------------------------------ pooled

def job_files(spec, spec_sha, world, rep, share):
    """One job's values and signals (or its stop record); None when missing. Under spec v3 a spec-v2 job's outputs
    must name spec v2 (tracebed.share_spec_sha256), a post-hoc job's the current spec."""
    d = T.job_dir(world, rep, share)
    want = T.share_spec_sha256(spec, spec_sha, share)
    if (d / "stop.json").exists():
        st = T.read_json(d / "stop.json")
        if st["spec_sha256"] != want:
            raise SystemExit(f"refused: {d}/stop.json was written under another spec")
        return dict(stopped=True, reason=st["reason"])
    v, s = T.read_json(d / "values.json"), T.read_json(d / "signals.json")
    if v is None or s is None:
        return None
    if v["spec_sha256"] != want or s["spec_sha256"] != want:
        raise SystemExit(f"refused: {d} holds outputs of another spec")
    if share != T.REFERENCE_SHARE:
        lock_sha = T.file_sha256(d / "lock.json")
        if v["lock_sha256"] != lock_sha or s["lock_sha256"] != lock_sha:
            raise SystemExit(f"refused: {d}: values or signals do not name the job's lock")
    return dict(stopped=False, values=v, signals=s)


def mean_or_none(xs):
    return None if not xs or any(x is None for x in xs) else float(np.mean(xs))


def collect_jobs(spec, spec_sha, shares):
    """{(share, seed, world): job files} of the given shares, and the jobs still missing."""
    jobs, missing = {}, []
    for share in shares:
        for rep in T.SEEDS:
            for world in T.WORLDS:
                f = job_files(spec, spec_sha, world, rep, share)
                if f is None:
                    missing.append(f"{world}/seed{rep}/{T.SHARE_DIR[share]}")
                jobs[(share, rep, world)] = f
    return jobs, missing


def v2_entries_check(out, v2_pooled):
    """Spec v3's pooled entries of the spec-v2 shares must equal pooled.json's (spec v2) exactly."""
    names = [T.SHARE_DIR[s] for s in T.FROZEN_SHARES]
    equal = {n: bool(T.A.canonical(T.jsonable(out[n])) == T.A.canonical(v2_pooled["shares"][n])) for n in names}
    return equal, bool(all(equal.values()))


def phase_pooled(spec, spec_sha, man_sha):
    ext = spec.get("post_hoc_extension")
    shares = [T.REFERENCE_SHARE] + sorted(T.spec_shares(spec))
    jobs, missing = collect_jobs(spec, spec_sha, shares)
    if missing:
        raise SystemExit("refused: jobs without values.json and signals.json (or stop.json): " + ", ".join(missing))
    out = pool_jobs(jobs, shares)
    if not ext:
        T.write_json(T.OUT / T.POOLED_V2_NAME, T.jsonable(dict(label=T.LABEL, kind="trace-pooled", spec_sha256=spec_sha,
                                                              code_manifest_sha256=man_sha, shares=out,
                                                              written=now())))
        T.log("pooled: written")
        return
    v2_path = T.OUT / T.POOLED_V2_NAME
    v2_pooled = T.read_json(v2_path)
    if v2_pooled is None or T.file_sha256(v2_path) != ext["frozen_v2"]["pooled_sha256"]:
        raise SystemExit(f"refused: {v2_path} is missing or differs from the spec-v2 pooled look the spec records")
    equal, same = v2_entries_check(out, v2_pooled)
    if not same:
        raise SystemExit(f"refused: the spec-v2 shares pooled again differ from pooled.json: {equal}")
    the_map = share_map(out, shares)
    T.write_json(T.OUT / T.POOLED_V3_NAME, T.jsonable(dict(
        label=T.LABEL, post_hoc_label=T.POST_HOC_LABEL, kind="trace-pooled-v3", spec_sha256=spec_sha,
        spec_version=int(spec["version"]), code_manifest_sha256=man_sha,
        spec_v2_pooled=dict(path=str(v2_path), sha256=T.file_sha256(v2_path), entries_equal=equal),
        map=the_map, shares=out, written=now())))
    for line in map_lines(the_map):
        T.log(line)
    T.log(f"pooled (spec v3): written {T.POOLED_V3_NAME}; pooled.json (spec v2) kept as it is")


def offset_row(job):
    """The J-by-region diagnostic of one job's readout (reported beside the selectivity)."""
    if job["stopped"]:
        return None
    r = job["signals"]["readout"]
    od = r["stand_ins"]["offset_diagnostic"]
    get = lambda d: None if d is None else d["mean"]  # noqa: E731
    return dict(J_region_rows=get(od["J"]["region_rows"]), J_outside_region_rows=get(od["J"]["outside_region_rows"]),
                J_past80_rows=get(od["J"]["past80_rows"]),
                neighbour_offset_T1_past80=r["sets"][S.PRIMARY_SET]["neighbour_offset_mean_past80"],
                neighbour_offset_T1c=r["sets"][S.COMMON_SET]["neighbour_offset_mean_past80"],
                neighbour_offset_T4=r["T4"]["neighbour_offset_mean"])


def pool_jobs(jobs, shares=None):
    """Q1 and Q2 per share from {(share, seed, world): dict(stopped, values, signals)} (pure; unit-tested). Judged:
    Q1 (strong-current harm, stalling world) and Q2 (T1 selectivity of residual evidence, training form, against
    the range hinge). Reported only: every other set and signal, the re-referenced differences on T1 and T1c, and
    the J-by-region diagnostic of both worlds. shares: the shares to pool (default: share 0 and every trace share);
    each share is pooled on its own, so adding shares never changes another share's entry."""
    out = {}
    for share in ([T.REFERENCE_SHARE] + list(T.SHARES)) if shares is None else shares:
        per_seed, harm = {}, []
        sel = {s: {k: [] for k in S.SIGNALS} for s in S.SETS}
        reref = {s: {k: [] for k in S.SIGNALS} for s in S.REREFERENCED_SETS}
        for rep in T.SEEDS:
            st, ns = jobs[(share, rep, "stall")], jobs[(share, rep, "no_stall")]
            both = not (st["stopped"] or ns["stopped"])
            row = dict(stall_stopped=st["stopped"], no_stall_stopped=ns["stopped"])
            h = None if st["stopped"] else st["values"]["situations"]["strong_current"]["harm"]
            harm.append(h)
            row["harm_strong_current"] = h
            row["no_stall_harm_strong_current"] = (None if ns["stopped"] else
                                                   ns["values"]["situations"]["strong_current"]["harm"])
            row["points_strong_current"] = None if st["stopped"] else st["values"]["situations"]["strong_current"]["points"]
            row["trace_critic_pull_past_80"] = None if st["stopped"] else st["values"]["pull_past_80"].get("trace_critic")
            row["selectivity"], row["rereferenced_difference"] = {}, {}
            for sname in S.SETS:
                row["selectivity"][sname] = {}
                for k in S.SIGNALS:
                    v = None
                    if both:
                        v = S.selectivity(st["signals"]["readout"]["sets"][sname], ns["signals"]["readout"]["sets"][sname], k)
                    row["selectivity"][sname][k] = v
                    sel[sname][k].append(v)
            for sname in S.REREFERENCED_SETS:
                row["rereferenced_difference"][sname] = {}
                for k in S.SIGNALS:
                    v = None
                    if both:
                        v = S.rereferenced_difference(st["signals"]["readout"]["sets"][sname],
                                                      ns["signals"]["readout"]["sets"][sname], k)
                    row["rereferenced_difference"][sname][k] = v
                    reref[sname][k].append(v)
            row["offset_diagnostic"] = {"stall": offset_row(st), "no_stall": offset_row(ns)}
            per_seed[rep] = row
        pooled_sel = {s: {k: mean_or_none(v) for k, v in d.items()} for s, d in sel.items()}
        pooled_reref = {s: {k: mean_or_none(v) for k, v in d.items()} for s, d in reref.items()}
        q1_mean = mean_or_none(harm)
        re, hinge = pooled_sel[S.PRIMARY_SET][S.PRIMARY_SIGNAL], pooled_sel[S.PRIMARY_SET][S.RIVAL_SIGNAL]
        q1 = q1_mean is not None and q1_mean >= T.HARM_MIN
        q2 = re is not None and hinge is not None and re >= T.SELECTIVITY_MIN and hinge < T.SELECTIVITY_MIN
        out[T.SHARE_DIR[share]] = dict(
            share=share, per_seed=per_seed, harm_strong_current_mean=q1_mean, selectivity_mean=pooled_sel,
            rereferenced_difference_mean=pooled_reref,
            Q1_harm_kept=bool(q1), Q2_residual_evidence_separates=bool(q2),
            usable_for_a_harm_claim=bool(q1 and q2) if share != T.REFERENCE_SHARE else None,
            note=("reference: the gate's own gapped log (no trace added); reported, not judged"
                  if share == T.REFERENCE_SHARE else "judged by the spec's questions (Q1, Q2); the rest is reported"))
    return out


def yes_no(flag, answered):
    return "yes" if flag else ("no" if answered else "no answer (a seed is missing)")


def share_map(out, shares=None):
    """The map over every share (pure; unit-tested): per share, smallest first, its origin (reference, planned in spec
    v2, post hoc in spec v3), the strong-current harm in the stalling world (mean over seeds 0-2 and per seed), the T1
    selectivity of every signal (mean over seeds 0-2), and Q1, Q2 and 'usable' as yes or no. Descriptive: nothing is
    chosen from it."""
    shares = sorted(float(o["share"]) for o in out.values()) if shares is None else sorted(float(s) for s in shares)
    rows = []
    for share in shares:
        o = out[T.SHARE_DIR[share]]
        sel = o["selectivity_mean"][S.PRIMARY_SET]
        harm = o["harm_strong_current_mean"]
        q2_answered = sel.get(S.PRIMARY_SIGNAL) is not None and sel.get(S.RIVAL_SIGNAL) is not None
        q1 = yes_no(o["Q1_harm_kept"], harm is not None)
        q2 = yes_no(o["Q2_residual_evidence_separates"], q2_answered)
        if share == T.REFERENCE_SHARE:
            usable = "not judged (reference)"
        else:
            usable = yes_no(o["usable_for_a_harm_claim"], harm is not None and q2_answered)
        rows.append(dict(
            share=share, percent=round(100.0 * share, 6), share_dir=T.SHARE_DIR[share], origin=T.share_origin(share),
            harm_strong_current_mean=harm,
            harm_strong_current_per_seed=[o["per_seed"][r]["harm_strong_current"] for r in T.SEEDS],
            selectivity_T1_mean={k: sel.get(k) for k in S.SIGNALS},
            residual_evidence_selectivity_T1_per_seed=[o["per_seed"][r]["selectivity"][S.PRIMARY_SET][S.PRIMARY_SIGNAL]
                                                        for r in T.SEEDS],
            Q1_harm_at_least_50=q1, Q2_trace_detected=q2, usable=usable))
    judged = [r for r in rows if r["share"] != T.REFERENCE_SHARE]
    return dict(
        rows=rows,
        marks=dict(harm_min_points=T.HARM_MIN, selectivity_min=T.SELECTIVITY_MIN, set=S.PRIMARY_SET,
                   primary_signal=S.PRIMARY_SIGNAL, rival_signal=S.RIVAL_SIGNAL),
        shares_with_harm_kept=[r["percent"] for r in judged if r["Q1_harm_at_least_50"] == "yes"],
        shares_with_trace_detected=[r["percent"] for r in judged if r["Q2_trace_detected"] == "yes"],
        usable_shares=[r["percent"] for r in judged if r["usable"] == "yes"],
        note=("descriptive only. Shares 1%, 3%, 5% and 10% are " + T.POST_HOC_LABEL + ". The questions and marks are "
              "spec v2's; no share, question, mark, signal or set is chosen from this map; any harm claim on any share "
              "needs its own pre-registration and fresh seeds, and must say that the share was found post hoc. "
              "Seeds 0-2 only."))


MAP_ORIGIN = {"reference": "reference", "planned": "planned v2", "post hoc": "post hoc v3"}
MAP_SIGNAL = {"residual_evidence": "resid", "residual_evidence_noiseless": "resid-nl", "distance": "distance",
              "range_hinge": "hinge", "density": "density", "heads": "heads"}


def map_lines(the_map):
    """The map as plain text lines for the log."""
    def f(x, nd=3):
        return "-" if x is None else f"{x:+.{nd}f}"
    sig = list(S.SIGNALS)
    origin = lambda o: next(v for k, v in MAP_ORIGIN.items() if o.startswith(k))  # noqa: E731
    head = ("share  origin       harm  " + "  ".join(f"{MAP_SIGNAL.get(k, k[:8]):>8}" for k in sig)
            + "  Q1   Q2   usable")
    out = ["map over every share (selectivity on T1, mean over seeds 0-2; harm in points, strong current, stalling "
           "world; resid = residual evidence, the judged signal; hinge = range hinge, the rival)", head]
    for r in the_map["rows"]:
        h = "-" if r["harm_strong_current_mean"] is None else f"{r['harm_strong_current_mean']:.1f}"
        out.append(f"{r['percent']:4g}%  {origin(r['origin']):11} {h:>5}  "
                   + "  ".join(f"{f(r['selectivity_T1_mean'][k]):>8}" for k in sig)
                   + f"  {r['Q1_harm_at_least_50']:4} {r['Q2_trace_detected']:4} {r['usable']}")
    out.append(f"usable shares (descriptive; 1%, 3%, 5% and 10% are post hoc): {the_map['usable_shares']}")
    return out


# ------------------------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=T.WORLDS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--share", type=float)
    ap.add_argument("--phase", nargs="+", required=True,
                    choices=("build", "train", "eval", "signals", "reference", "pooled"))
    a = ap.parse_args(argv)
    spec, spec_sha, man_sha = T.begin()
    if a.phase == ["pooled"]:
        return phase_pooled(spec, spec_sha, man_sha)
    if a.world is None or a.seed is None or a.share is None:
        raise SystemExit("--world, --seed and --share are needed")
    check_job(spec, a.world, a.seed, a.share)
    if (a.share == T.REFERENCE_SHARE) != (a.phase == ["reference"]):
        raise SystemExit("refused: share 0 runs the reference phase only, and the reference phase needs share 0")
    cpu = T.require_one_cpu(f"job {a.world}/seed {a.seed}/share {a.share}")
    T.log(f"job {a.world}/seed {a.seed}/{T.SHARE_DIR[a.share]}: phases {a.phase}; spec {spec_sha[:16]}, "
          f"manifest {man_sha[:16]}; pinned to CPU {cpu[0]}")
    g = load(a.world, a.seed, spec)
    for ph in a.phase:
        if ph == "reference":
            phase_reference(g, a.world, a.seed, spec_sha, man_sha)
            continue
        fn = dict(build=phase_build, train=phase_train, eval=phase_eval, signals=phase_signals)[ph]
        fn(g, a.world, a.seed, a.share, spec_sha, man_sha)
        T.begin()   # the spec, gate files and code still check out after each phase


if __name__ == "__main__":
    main()
