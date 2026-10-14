"""Stage 2, version 7: the self-test's probes, the references they make, and the re-make rule (V7_SPEC.md C3, C7, C11).

No BCA code runs here except in R1 (information only, after the v7 import). Every gate input is read through the
gate's read-only paths: the room harness's SeedContext (acceptance.load_prereg + guard + load_seed), prepared('gap'),
acceptance.get_critic(..., train=False), the clone's read path (ROOM/harness.py:240-246: acceptance.finished on its
record, then load_tree; never get_clone, whose other branch trains and writes in the gate's folder), check 2's saved
actors (ROOM/harness.py:248-255) and get_host(..., train=False) (ROOM/harness.py:267-271).

  P1 (one process per seed s in 2-4 and world w):
    P1a  the gate's check-2 call (GATE/acceptance.py:1459-1484): L.ActorOnly(prepared, clone, learn, s, alpha=5.0).run(
         [Arm(name, critic=i)], critics, weighted=False), the arms side by side as in the gate (stall: gapped, full,
         random against the gap, full-log and random-removal critics; no-stall: no_stall_gapped, no_stall_full), a
         missing gate critic left out as the gate's swap() does. Writes R7/refs/seed<s>/check2_<w>.msgpack (a dictionary
         of the arms, L.save_tree), the re-made policy reference.
    P1b  the check (b) forecast: the room harness's no-BCA W = 0 arm RoomArm('w0_check', 'plain', (0.0,), 1.0) through
         RoomRunner with check (b)'s start (the clone), critic (the gap critic), batch-row stream and program settings
         (S2C/policy.py:371-372, :385; :527-528, :537); alone in the stalling world, beside RoomArm('pull_2', 'plain',
         (0.0,), 2.0) in the no-stall world (check (b) runs w0_check beside bca_k1 there, S2C/policy.py:540).
    P1c  the seeds 5-9 forecast: the fresh seeds' actor-only structure on seed s's gate inputs (S2C/fresh.py:245-253):
         stall [none (gap critic), full_none (full-log critic)], no-stall [none_no_stall (no-stall gap critic)].
    Record R7/selftest/p1_seed<s>_<w>.json: per actor the laptop's and the gate's parameter SHA-256 (L.tree_sha256),
    the largest absolute parameter difference, 'differs', the failures; P1b's and P1c's SHA-256 and their differences
    from the gate's actor, P1a's and each other; the forecast gaps F_ref and F_fresh (C7 step 2b).
  P2 (one process per seed): L.cotrain(prepared('gap'), learn, s, alpha=5.0, table=None, stream=0,
    out_dir=R7/refs/seed<s>/host_plain, name='TD3+BC plain (stall, gap log)') -- the gate's call for the arm 'plain'
    (GATE/acceptance.py:1071-1072). The gate's code writes snapshot_050/100.msgpack; v7 writes record.json. Record
    R7/selftest/p2_seed<s>.json compares the final actor (and reports the critic) with the gate's, and records check
    2's 'td3bc_gapped' SHA-256 and whether it equals the gate's host actor.
  P3 a fixed 1,000-step ActorOnly run (prints the parameters' SHA-256; the self-test runs it idle and under load on
    two lane processors and requires identical results, C3).
  R1 (information, after the v7 import): the weighted band recomputed with v6's pinned band code from the gate's 'none'
    and, where re-made, from the laptop's copy, compared with the cloud's band; written to R7/info/.
  remake: 'selftest_v7.sh --remake <probe>' (C7, a damaged reference): the damaged file is moved (never deleted) to
    R7/interrupted/refs/<UTC time>/, the probe's call reruns into R7/refs_remake/<UTC time>/, and the new file is renamed
    into the original path only if its SHA-256 equals the pin; otherwise the run stays halted.

Records and reference files are written once. An interrupted probe (no record yet) has its files moved aside (C5) and
reruns; a written record is never replaced. Each record holds the fingerprints of its inputs: the constants file, this
script, the gate's pre-registration, v6's code manifest and the re-make and forecast block of the draft prereg_v7.json.

Command line (each probe in its own process, in launch_v7.sh's environment, on one processor):
  env -i <the eight variables> taskset -c <cpu> /tmp/claude-0/bcavenv/bin/python v7_probes.py p1 --seed 2 --world stall
  ... p2 --seed 2 | p3 --seed 2 | r1 --seed 2 --world stall | remake --probe p1_seed2_stall | forecast
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import v7_common as K  # noqa: E402
import v7_ledger as LG  # noqa: E402
import v7_verify as VF  # noqa: E402

ALPHA = 5.0
PROBE_SEEDS = K.REMAKE_SEEDS
WORLDS = ("stall", "no_stall")
P1A_ARMS = {"stall": (("gapped", "gap"), ("full", "none"), ("random", "random")),       # GATE/acceptance.py:1482-1483
            "no_stall": (("no_stall_gapped", "gap"), ("no_stall_full", "none"))}          # :1484
CHECK_B_KEY = {"stall": "gapped", "no_stall": "no_stall_gapped"}                         # S2C/fresh.py:65-66
ENTRY_KEYS = {"stall": ("gapped", "full"), "no_stall": ("no_stall_gapped",)}            # C7 step 2
P1C_ARMS = {"stall": (("none", "gap"), ("full_none", "none")), "no_stall": (("none_no_stall", "gap"),)}
HOST_NAME = "TD3+BC plain (stall, gap log)"                                               # GATE/acceptance.py:1072
P3_STEPS = 1000


def probe_id(kind, seed, world=None):
    return f"{kind}_seed{int(seed)}" + (f"_{world}" if world else "")


def all_probe_ids():
    return [probe_id("p1", s, w) for s in PROBE_SEEDS for w in WORLDS] + [probe_id("p2", s) for s in PROBE_SEEDS]


def record_path(r7, pid):
    return Path(r7) / "selftest" / f"{pid}.json"


def p1_reference(r7, seed, world):
    return Path(r7) / "refs" / f"seed{int(seed)}" / f"check2_{world}.msgpack"


def p2_dir(r7, seed):
    return Path(r7) / "refs" / f"seed{int(seed)}" / "host_plain"


def _diff(H, a, b):
    if a is None or b is None:
        return None
    try:
        v = float(H.max_param_diff(a, b))
    except Exception:  # noqa: BLE001 - a shape mismatch is reported as no comparison
        return None
    return v if v == v and v != float("inf") else None


def _sha(L, tree):
    return None if tree is None else L.tree_sha256(tree)


def _prepare_outputs(r7, pid, rels):
    """C7 'Probe records': a written record is never replaced; files of an interrupted probe (no record yet) are moved
    aside (C5) before it reruns."""
    r7 = Path(r7)
    if record_path(r7, pid).exists() or Path(str(record_path(r7, pid)) + ".sha256").exists():
        raise K.V7Refusal(f"the probe record {record_path(r7, pid)} exists; a probe record is never replaced")
    present = [rel for rel in rels if (r7 / rel).is_file()]
    if present:
        stamp = K.utc_stamp()
        d = r7 / "interrupted" / f"probe_{pid}"
        return LG.move_files_aside(present, r7, d / stamp, d / f"{stamp}.json", phase_id=f"probe_{pid}",
                                   interruption="probe interrupted before its record", why="no record yet")
    return None


# --------------------------------------------------------------------------------------------- P1

def p1a_call(H, ctx, world, seed, log=None):
    """The gate's check-2 swap for one world (GATE/acceptance.py:1459-1484): returns (params by name, failed by name,
    info). A gate critic that failed (params None) is left out, and its actor counts as failed, as swap() does."""
    A, L = H.A, H.L
    prep = ctx.prepared(world)
    clone = ctx.clone(world)[0]
    named = [(n, A.get_critic(ctx.prereg, ctx.sd[world], removal, "stall_aware", train=False)[1])
             for n, removal in P1A_ARMS[world]]
    failed = {n: c is None for n, c in named}
    ok = [(n, c) for n, c in named if c is not None]
    params, info = {}, None
    if ok:
        ao = L.ActorOnly(prep, clone, ctx.learn, seed, alpha=ALPHA)
        stacked, info = ao.run([L.Arm(n, critic=i) for i, (n, _) in enumerate(ok)], [c for _, c in ok],
                               weighted=False, log=log)
        for i, (n, _) in enumerate(ok):
            params[n] = ao.arm_params(stacked, i)
            failed[n] = bool(info["failed"][i])
    return params, failed, info


def p1b_call(H, ctx, world, seed, log=None):
    """The room program at W = 0 with check (b)'s inputs and group size (C7 step 1, P1b). Returns (params, finite)."""
    prep = ctx.prepared(world)
    clone = ctx.clone(world)[0]
    critic = ctx.critic(world, "gap")[0]
    cfg = H.ProgramConfig() if world == "stall" else H.ProgramConfig(physics=prep.physics, next_command="stall_aware")
    shared, _ = H.build_shared(prep, critic, seed, cfg=cfg, need_harm=False, log_fn=log)
    runner = H.RoomRunner(prep, clone, critic, ctx.learn, seed, cfg=cfg, shared=shared)
    arms = [H.RoomArm("w0_check", "plain", (0.0,), 1.0)]
    if world == "no_stall":
        arms.append(H.RoomArm("pull_2", "plain", (0.0,), 2.0))
    out = runner.run(arms, log)
    return out["w0_check"]["params"], bool(out["w0_check"]["finite"])


def p1c_call(H, ctx, world, seed, log=None):
    """The fresh seeds' actor-only structure (S2C/fresh.py:245-253) on the gate's inputs. Returns (params of 'none'
    or 'none_no_stall', failed)."""
    A, L = H.A, H.L
    prep = ctx.prepared(world)
    clone = ctx.clone(world)[0]
    critics = []
    for _, removal in P1C_ARMS[world]:
        c = A.get_critic(ctx.prereg, ctx.sd[world], removal, "stall_aware", train=False)[1]
        if c is None:
            return None, True
        critics.append(c)
    ao = L.ActorOnly(prep, clone, ctx.learn, seed, alpha=ALPHA)
    stacked, info = ao.run([L.Arm(n, critic=i) for i, (n, _) in enumerate(P1C_ARMS[world])], critics, weighted=False,
                           log=log)
    return ao.arm_params(stacked, 0), bool(info["failed"][0])


def forecast_of(p1a_actors, p1b, p1c, world, tol=K.W0_TOL):
    """C7 step 2b for one seed and world (assuming the phase is not inherited): F_ref against the actor check (b) would
    read (P1a's copy when step 2 makes an entry, the gate's otherwise) and F_fresh against P1c."""
    keys = ENTRY_KEYS[world]
    differs = any(p1a_actors[k]["differs"] for k in keys if k in p1a_actors)
    failed = any(p1a_actors[k]["laptop_failed"] and not p1a_actors[k]["gate_failed"] for k in keys if k in p1a_actors)
    entry = bool(differs and not p1b["reproduces_gate"] and not failed)
    f_ref = p1b["max_abs_diff_vs_p1a"] if entry else p1b["max_abs_diff_vs_gate"]
    f_fresh = p1b["max_abs_diff_vs_p1c"]
    above = [n for n, v in (("F_ref", f_ref), ("F_fresh", f_fresh)) if v is None or v > tol]
    return dict(entry_if_not_inherited=entry, remake_failed=failed, p1a_differs=differs,
                p1b_reproduces_gate=p1b["reproduces_gate"], F_ref=f_ref, F_fresh=f_fresh, tolerance=tol,
                above_tolerance=above, F_ref_against="P1a's copy" if entry else "the gate's saved actor")


def run_p1(H, seed, world, r7, inputs, log=None):
    """One P1 process: P1a (writes the reference file), P1b, P1c; the record. Returns the record."""
    seed, r7 = int(seed), Path(r7)
    if seed not in PROBE_SEEDS or world not in WORLDS:
        raise K.V7Refusal(f"P1 runs for seeds 2-4 and both worlds only, not ({seed}, {world})")
    pid = probe_id("p1", seed, world)
    ref = p1_reference(r7, seed, world)
    moved = _prepare_outputs(r7, pid, [ref.relative_to(r7).as_posix()])
    t0 = time.perf_counter()
    L = H.L
    ctx = H.SeedContext(seed, worlds=("stall", "no_stall"))
    params, failed, info = p1a_call(H, ctx, world, seed, log)
    if ref.exists():
        raise K.V7Refusal(f"{ref} exists; a reference file is written once")
    file_sha = L.save_tree(ref, {n: params[n] for n in sorted(params)}) if params else None
    t1 = time.perf_counter()
    gate_acts, gate_rec, gate_dir = ctx.h2()
    gate_failed = gate_rec.get("failed") or {}
    actors = {}
    for n, _ in P1A_ARMS[world]:
        lap, gate = params.get(n), gate_acts.get(n)
        row = dict(laptop_sha256=_sha(L, lap), gate_sha256=_sha(L, gate), max_abs_diff=_diff(H, lap, gate),
                   laptop_failed=bool(failed.get(n)), gate_failed=bool(gate_failed.get(n)) or gate is None)
        row["differs"] = row["laptop_sha256"] != row["gate_sha256"]
        actors[n] = row
    p1b_params, p1b_finite = p1b_call(H, ctx, world, seed, log)
    t2 = time.perf_counter()
    p1c_params, p1c_failed = p1c_call(H, ctx, world, seed, log)
    t3 = time.perf_counter()
    k = CHECK_B_KEY[world]
    p1b = dict(arms=["w0_check"] + (["pull_2"] if world == "no_stall" else []), actor_sha256=_sha(L, p1b_params),
               finite=p1b_finite, max_abs_diff_vs_gate=_diff(H, p1b_params, gate_acts.get(k)),
               max_abs_diff_vs_p1a=_diff(H, p1b_params, params.get(k)),
               max_abs_diff_vs_p1c=_diff(H, p1b_params, p1c_params))
    p1b["reproduces_gate"] = p1b["actor_sha256"] is not None and p1b["actor_sha256"] == actors[k]["gate_sha256"]
    p1c = dict(arms=[n for n, _ in P1C_ARMS[world]], actor_sha256=_sha(L, p1c_params), failed=p1c_failed,
               max_abs_diff_vs_gate=_diff(H, p1c_params, gate_acts.get(k)),
               max_abs_diff_vs_p1a=_diff(H, p1c_params, params.get(k)),
               max_abs_diff_vs_p1b=_diff(H, p1c_params, p1b_params))
    record = dict(
        kind="stage2-v7-probe-p1", probe=pid, seed=seed, world=world, inputs=inputs,
        p1a=dict(call=("L.ActorOnly(prepared(world), clone(world), learn, seed, alpha=5.0).run([L.Arm(name, critic=i) "
                       "...], [critics], weighted=False); ao.arm_params(stacked, i) (GATE/acceptance.py:1459-1484)"),
                 names_written=sorted(params), actors=actors,
                 reference_file=dict(path=ref.relative_to(r7).as_posix(), sha256=file_sha),
                 rows_sha256=None if info is None else info.get("rows_sha256"),
                 start_sha256=None if info is None else info.get("start_sha256")),
        p1b=p1b, p1c=p1c, forecast=forecast_of(actors, p1b, p1c, world),
        gate=dict(check2_dir=str(gate_dir), actors_sha256=gate_rec.get("actors_sha256")),
        moved_aside=moved, seconds=dict(p1a=t1 - t0, p1b=t2 - t1, p1c=t3 - t2), written_utc=K.utc_now())
    K.write_json_once(record_path(r7, pid), record)
    return record


# --------------------------------------------------------------------------------------------- P2

def p2_call(H, ctx, seed, out_dir, log=None):
    """The gate's call for the host arm 'plain' (GATE/acceptance.py:1066-1072, HOST_ARMS['plain']: stream 0, no
    table): returns (result, final state)."""
    L = H.L
    result, state, _ = L.cotrain(ctx.prepared("stall"), ctx.learn, seed, alpha=ALPHA, table=None, stream=0,
                                 out_dir=out_dir, log=log, name=HOST_NAME)
    return result, state


def run_p2(H, seed, r7, inputs, log=None):
    seed, r7 = int(seed), Path(r7)
    if seed not in PROBE_SEEDS:
        raise K.V7Refusal(f"P2 runs for seeds 2-4 only, not {seed}")
    pid = probe_id("p2", seed)
    d = p2_dir(r7, seed)
    rels = [(d / n).relative_to(r7).as_posix() for n in ("snapshot_050.msgpack", "snapshot_100.msgpack",
                                                          "record.json", "record.json.sha256")]
    moved = _prepare_outputs(r7, pid, rels)
    t0 = time.perf_counter()
    L = H.L
    ctx = H.SeedContext(seed, worlds=("stall",))
    result, state = p2_call(H, ctx, seed, d, log)
    K.write_json_once(d / "record.json", dict(kind="stage2-v7-p2-run", probe=pid, seed=seed,
                                              call=f"L.cotrain(prepared('gap'), learn, {seed}, alpha=5.0, table=None, "
                                                   f"stream=0, out_dir=..., name={HOST_NAME!r})",
                                              result=result, inputs=inputs))
    t1 = time.perf_counter()
    gate_state, gate_rec = ctx.host()
    gate_acts, h2rec, _ = ctx.h2()
    lap_actor = None if state is None else state.actor.params
    actor = dict(laptop_sha256=_sha(L, lap_actor), gate_sha256=_sha(L, gate_state.actor.params),
                 max_abs_diff=_diff(H, lap_actor, gate_state.actor.params), laptop_failed=bool(result.get("failed")),
                 gate_failed=gate_state is None)
    actor["differs"] = actor["laptop_sha256"] != actor["gate_sha256"]
    lap_critic = None if state is None else state.critic.params
    critic = dict(laptop_sha256=_sha(L, lap_critic), gate_sha256=_sha(L, gate_state.critic.params),
                  max_abs_diff=_diff(H, lap_critic, gate_state.critic.params))
    critic["identical"] = critic["laptop_sha256"] == critic["gate_sha256"]
    snaps = {}
    for n in ("snapshot_050.msgpack", "snapshot_100.msgpack"):
        p = d / n
        snaps[n] = dict(path=p.relative_to(r7).as_posix(), sha256=K.file_sha256(p) if p.is_file() else None)
    td3 = gate_acts.get("td3bc_gapped")
    record = dict(
        kind="stage2-v7-probe-p2", probe=pid, seed=seed, world="stall", inputs=inputs, call=f"L.cotrain(... {HOST_NAME})",
        actor=actor, critic=critic, reference_file=snaps["snapshot_100.msgpack"], snapshots=snaps,
        run_record=dict(path=(d / "record.json").relative_to(r7).as_posix(), sha256=K.file_sha256(d / "record.json")),
        gate_snapshot_sha256=[s.get("file_sha256") for s in (gate_rec.get("result") or {}).get("snapshots", [])],
        td3bc_gapped=dict(sha256=_sha(L, td3), equals_gate_host_actor=_sha(L, td3) == actor["gate_sha256"]),
        moved_aside=moved, seconds=dict(train=t1 - t0, compare=time.perf_counter() - t1), written_utc=K.utc_now())
    K.write_json_once(record_path(r7, pid), record)
    return record


# --------------------------------------------------------------------------------------------- P3

def run_p3(H, seed=2, steps=P3_STEPS, log=None):
    """A fixed actor-only run (no record). Returns its parameters' SHA-256 (tree and serialized bytes)."""
    L = H.L
    ctx = H.SeedContext(int(seed), worlds=("stall",))
    prep = ctx.prepared("stall")
    critic = ctx.critic("stall", "gap")[0]
    ao = L.ActorOnly(prep, ctx.clone("stall")[0], ctx.learn, int(seed), alpha=ALPHA, steps=steps, chunk=steps)
    stacked, info = ao.run([L.Arm("none", critic=0)], [critic], weighted=False, log=log)
    p = ao.arm_params(stacked, 0)
    from flax import serialization
    return dict(seed=int(seed), steps=int(steps), tree_sha256=L.tree_sha256(p),
                bytes_sha256=K.sha256_bytes(serialization.to_bytes(p)), failed=bool(info["failed"][0]),
                cpu=sorted(os.sched_getaffinity(0)))


def p3_compare(results):
    """C3: every run identical, byte for byte. (ok, detail)."""
    shas = sorted({(r["tree_sha256"], r["bytes_sha256"]) for r in results})
    return len(shas) == 1 and len(results) > 0, dict(runs=len(results), distinct=len(shas))


# --------------------------------------------------------------------------------------------- C7 steps 2 and 2b

def forecast_all(records, q7="stop", tol=K.W0_TOL):
    """The six P1 records -> rows and the verdict: NOT READY (P1b) when any F_ref or F_fresh is above 1e-5, unless the
    draft records the user's choice to proceed (Q7)."""
    rows, above = {}, []
    for pid, rec in sorted(records.items()):
        if rec.get("kind") != "stage2-v7-probe-p1":
            continue
        f = rec["forecast"]
        rows[pid] = dict(F_ref=f["F_ref"], F_fresh=f["F_fresh"], above=f["above_tolerance"])
        if f["above_tolerance"]:
            above.append(pid)
    not_ready = bool(above) and q7 != "proceed"
    return dict(rows=rows, above=above, tolerance=tol, q7=q7, not_ready=not_ready,
                line=("NOT READY (P1b: check (b) forecast above 1e-5 in " + ", ".join(above) + ")") if not_ready
                else ("READY as to the check (b) forecast" + (" (gaps above 1e-5 accepted under Q7: " + ", ".join(above)
                                                                + ")" if above else "")))


def result_line(rec):
    """The self-test's words for one probe record (C11)."""
    if rec["kind"] == "stage2-v7-probe-p1":
        f = rec["forecast"]
        if f["remake_failed"]:
            return "re-make failed (the laptop's run failed where the gate's did not)"
        if not f["p1a_differs"]:
            return "bit-identical"
        if f["p1b_reproduces_gate"]:
            return "gate's file kept (P1b reproduced it)"
        return "reference will be re-made if not inherited"
    a = rec["actor"]
    if a["laptop_failed"] and not a["gate_failed"]:
        return "re-make failed (the laptop's run failed where the gate's did not)"
    return "reference will be re-made if not inherited" if a["differs"] else "bit-identical"


COMPONENTS = (("policy stall", "policy_train_seed{s}"), ("policy no_stall", "policy_no_stall_train_seed{s}"),
              ("host stall", "host_train_seed{s}_stall"), ("host no_stall", "host_train_seed{s}_no_stall"),
              ("deploy stall", "deploy_seed{s}_stall"), ("deploy no_stall", "deploy_seed{s}_no_stall"))


def remake_table(records, record_sha256, inherited):
    """C7 step 2, from pinned inputs only: (entries, labels). records {probe id: record}, record_sha256 {probe id:
    SHA-256 of the record file}, inherited: the set of phase ids HANDOFF.json lists as inherited."""
    entries, labels = [], {}
    inherited = set(inherited)
    for s in PROBE_SEEDS:
        for world, phase in (("stall", "policy_train"), ("no_stall", "policy_no_stall_train")):
            pid = probe_id("p1", s, world)
            rec = records[pid]
            f = rec["forecast"]
            comp = f"seed{s} policy {world}"
            if f"{phase}_seed{s}" in inherited:
                labels[comp] = "inherited (cloud, v6)"
                continue
            if f["entry_if_not_inherited"]:
                acts = rec["p1a"]["actors"]
                lock_names = {"gapped": "none", "full": "full_none", "no_stall_gapped": "none"}
                entries.append(dict(
                    phase=phase, seed=s, world=world, reference_file=rec["p1a"]["reference_file"]["path"],
                    file_sha256=rec["p1a"]["reference_file"]["sha256"], names_in_file=rec["p1a"]["names_written"],
                    replace={k: dict(actor_sha256=acts[k]["laptop_sha256"], lock_name=lock_names[k])
                             for k in ENTRY_KEYS[world]},
                    probe_record=pid, probe_record_sha256=record_sha256[pid],
                    largest_parameter_difference={k: acts[k]["max_abs_diff"] for k in ENTRY_KEYS[world]}))
                labels[comp] = "laptop, reference re-made on the laptop"
            elif f["remake_failed"]:
                labels[comp] = "laptop, re-make failed"
            elif f["p1a_differs"]:
                labels[comp] = "laptop, gate's reference kept (P1b reproduced it)"
            else:
                labels[comp] = "laptop, gate's reference (v6 as written)"
        pid = probe_id("p2", s)
        rec = records[pid]
        a = rec["actor"]
        comp = f"seed{s} host stall"
        if f"host_train_seed{s}_stall" in inherited:
            labels[comp] = "inherited (cloud, v6)"
        elif a["laptop_failed"] and not a["gate_failed"]:
            labels[comp] = "laptop, re-make failed"
        elif a["differs"]:
            entries.append(dict(
                phase="host_train", seed=s, world="stall", reference_file=rec["reference_file"]["path"],
                file_sha256=rec["reference_file"]["sha256"],
                replace={"actor": dict(actor_sha256=a["laptop_sha256"], lock_name="plain")},
                probe_record=pid, probe_record_sha256=record_sha256[pid],
                largest_parameter_difference={"actor": a["max_abs_diff"]}))
            labels[comp] = "laptop, reference re-made on the laptop"
        else:
            labels[comp] = "laptop, gate's reference (v6 as written)"
        labels[f"seed{s} host no_stall"] = ("inherited (cloud, v6)" if f"host_train_seed{s}_no_stall" in inherited
                                            else "laptop, own run")
        for w in WORLDS:
            labels[f"seed{s} deploy {w}"] = ("inherited (cloud, v6)" if f"deploy_seed{s}_{w}" in inherited
                                             else "laptop, gate's reference (v6 as written)")
    return entries, labels


def read_probe_records(r7, pins=None):
    """All nine records, each verified against its sidecar (and against pins {probe id: SHA-256} when given)."""
    out, shas = {}, {}
    for pid in all_probe_ids():
        p = record_path(r7, pid)
        out[pid] = K.read_json_verified(p, f"probe record {pid}")
        shas[pid] = K.file_sha256(p)
        if pins is not None and pins.get(pid) != shas[pid]:
            raise K.V7Refusal(f"HALT: probe record {pid} differs from its pin")
    return out, shas


# --------------------------------------------------------------------------------------------- R1 (information)

def run_r1(seed, world, r7, entry, s_folder=None, out_root=None, log=None):
    """C7 R1: the weighted band refit from the gate's 'none' (and the laptop's copy where re-made), compared with the
    cloud's weighted band. Uses v6's pinned band code (BCA code: only after the v7 import, under v6's approval)."""
    import numpy as np
    s_folder = str(s_folder or K.S_FOLDER)
    out_root = Path(out_root or K.OUT)
    if s_folder not in sys.path:
        sys.path.insert(0, s_folder)
    import stage2_common as C
    import band as B
    import fresh as F
    import jax
    approval = C.require_approval()
    ctx = F.context(int(seed), worlds=("stall", "no_stall"))
    pg = ctx.prepared(world)
    d_u = out_root / "detect" / "uniform" / world / "fixed" / f"seed{int(seed)}"
    d_w = out_root / "detect" / "weighted" / world / "fixed" / f"seed{int(seed)}"
    band_u, brec = B.load_band(d_u, approval, B.TC_DX(), B.TC_DA())
    cloud, _ = B.load_band(d_w, approval, B.TC_DX(), B.TC_DA())
    refit = B.weighted_refitter(ctx, world, pg, band_u, brec, d_u, approval)

    def compare(band):
        diffs, same = [], True
        for x, y in zip(jax.tree_util.tree_leaves(band), jax.tree_util.tree_leaves(cloud)):
            x, y = np.asarray(x), np.asarray(y)
            same = same and x.shape == y.shape and x.tobytes() == y.tobytes()
            if x.shape == y.shape and x.size:
                diffs.append(float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
        return dict(bit_identical=bool(same), max_abs_diff=max(diffs, default=0.0))

    band_g, _, diag_g = refit(ctx.none(world))
    rows = dict(gate_none=compare(band_g))
    if entry is not None:
        L = F._gate()[1]
        key = CHECK_B_KEY[world]
        ref = L.load_tree(Path(r7) / entry["reference_file"], {k: L.actor_template() for k in entry["names_in_file"]},
                          entry["file_sha256"])
        band_l, _, _ = refit(ref[key])
        rows["laptop_none"] = compare(band_l)
    record = dict(kind="stage2-v7-r1", seed=int(seed), world=world, information_only=True, comparisons=rows,
                  cloud_band=str(d_w / "band.msgpack"), written_utc=K.utc_now(),
                  note="routes nothing; the policy lock does not keep this refit's diagnostics (S2C/policy.py:377)")
    K.write_json_once(Path(r7) / "info" / f"r1_seed{int(seed)}_{world}.json", record)
    return record


# --------------------------------------------------------------------------------------------- a damaged reference

def remake(H, pid, r7, pins, log=None):
    """C7 'A damaged reference'. pins {path relative to R7: SHA-256} for the probe's files. Moves each damaged file
    (missing or with another SHA-256) aside, reruns the probe's call into R7/refs_remake/<UTC time>/, and renames each
    new file into the original path only if its SHA-256 equals the pin. V7Refusal otherwise (the run stays halted)."""
    r7 = Path(r7)
    damaged = [rel for rel, h in sorted(pins.items()) if not (r7 / rel).is_file() or K.file_sha256(r7 / rel) != h]
    if not damaged:
        return dict(probe=pid, damaged=[], note="every file matches its pin; nothing to do")
    stamp = K.utc_stamp()
    moved = LG.move_files_aside([rel for rel in damaged if (r7 / rel).is_file()], r7,
                                r7 / "interrupted" / "refs" / stamp, r7 / "interrupted" / "refs" / f"{stamp}.json",
                                probe=pid, why="a reference file no longer matched its pin")
    tmp = r7 / "refs_remake" / stamp
    kind, seed = pid.split("_")[0], int(pid.split("_")[1][4:])
    ctx = H.SeedContext(seed, worlds=("stall", "no_stall"))
    if kind == "p1":
        world = pid.split("_", 2)[2]
        params, _, _ = p1a_call(H, ctx, world, seed, log)
        H.L.save_tree(tmp / "refs" / f"seed{seed}" / f"check2_{world}.msgpack", {n: params[n] for n in sorted(params)})
    elif kind == "p2":
        p2_call(H, ctx, seed, tmp / "refs" / f"seed{seed}" / "host_plain", log)
    else:
        raise K.V7Refusal(f"no remake for probe {pid}")
    renamed, bad = [], []
    for rel in damaged:
        new = tmp / rel
        if new.is_file() and K.file_sha256(new) == pins[rel]:
            os.rename(new, r7 / rel)
            renamed.append(rel)
        else:
            bad.append(rel)
    result = dict(kind="stage2-v7-remake", probe=pid, damaged=damaged, moved=moved, renamed=renamed, not_reproduced=bad,
                  temporary=str(tmp), utc=K.utc_now())
    K.write_json_once(r7 / "refs_remake" / f"{stamp}.json", result)
    if bad:
        raise K.V7Refusal(f"HALT: the remake of {pid} did not reproduce the pinned SHA-256 of {bad}; the run stays halted")
    return result


# --------------------------------------------------------------------------------------------- the process

def inputs_block(prereg, constants_sha256, root=None):
    """The fingerprints a probe record binds to (C7 'Probe records')."""
    root = Path(root or K.ROOT)
    gate = K.file_sha256(root / "runs" / "shift" / "preregistration.json")
    if gate != K.GATE_PREREG_SHA256:
        raise K.V7Refusal("the gate's pre-registration is not 68d9e877...")
    return dict(constants_sha256=constants_sha256, probe_script_sha256=K.file_sha256(Path(__file__)),
                gate_prereg_sha256=gate, v6_code_manifest_sha256=K.V6["code_manifest_sha256"],
                remake_block_sha256=K.block_sha256(prereg.payload.get("remake")),
                prereg_v7_sha256_at_run=prereg.sha256, prereg_v7_was_final=prereg.final)


def setup_process(pid, r7, cpu=None, need_final=False, ledger_path=None, load_harness=True):
    """What every probe process does first: the phase environment and one processor (C3, C4), prereg_v7.json and the
    v7 code (draft allowed for P1-P3), its own ledger attempt with the audit hook (C10; the attempt is the one
    'v7_ledger.py begin-probe' allocated when --ledger is given, else a new one), the constants (C1), then the pinned
    room harness. Returns (H, prereg, inputs, ledger)."""
    VF.check_environment()
    VF.check_one_processor(cpu)
    prereg = VF.load_prereg_v7()
    if need_final:
        VF.require_final(prereg)
    VF.check_v7_code(prereg)
    VF.check_v6_pins()
    if ledger_path is None:
        _, path = LG.allocate_attempt(r7, f"probe_{pid}")
        LG.append_event(path, "launch-start", probe=pid, argv=sys.argv[1:], cpu=cpu, boot_id=K.boot_id(),
                        r7=str(Path(r7).resolve()), out_root=str(K.OUT))
    else:
        path = Path(ledger_path)
        if path.parent != Path(r7) / "ledger" / f"probe_{pid}":
            raise K.V7Refusal(f"the ledger file {path} is not an attempt of probe_{pid}")
    ledger = LG.Ledger(path)
    ledger.install_audit_hook()
    ledger.event("wrapper-start", probe=pid, argv=sys.argv[1:], boot_id=K.boot_id())
    block = prereg.payload.get("constants") or {}
    if block.get("file") != K.CONSTANTS_REL or block.get("sha256") != K.CONSTANTS_SHA256:
        raise K.V7Refusal("prereg_v7.json does not pin the constants file 6602f9d1...")
    import v7_constants
    facts = v7_constants.apply(K.ROOT / K.CONSTANTS_REL, K.CONSTANTS_SHA256)
    ledger.event("constants", facts=facts)
    ledger.event("facts", facts=VF.machine_facts())
    H = K.load_v6_common().load_room_harness() if load_harness else None
    return H, prereg, inputs_block(prereg, facts["sha256"]), ledger


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("p1", "p2", "p3", "r1", "remake", "forecast"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--world", choices=WORLDS)
    ap.add_argument("--probe")
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--ledger", help="the attempt file 'v7_ledger.py begin-probe' allocated")
    a = ap.parse_args(argv)
    r7 = K.R7
    ledger = None
    try:
        if a.cmd == "forecast":
            prereg = VF.load_prereg_v7()
            records = {pid: K.read_json_verified(record_path(r7, pid)) for pid in all_probe_ids()
                       if record_path(r7, pid).exists() and pid.startswith("p1")}
            out = forecast_all(records, (prereg.payload.get("remake") or {}).get("q7_if_forecast_above_tolerance"))
            print(json.dumps(out, indent=1, sort_keys=True))
            return 3 if out["not_ready"] else 0
        if a.cmd in ("p1", "r1") and (a.seed is None or a.world is None):
            raise K.V7Refusal(f"{a.cmd} needs --seed and --world")
        if a.cmd in ("p2", "p3") and a.seed is None:
            raise K.V7Refusal(f"{a.cmd} needs --seed")
        pid = {"p1": lambda: probe_id("p1", a.seed, a.world), "p2": lambda: probe_id("p2", a.seed),
               "p3": lambda: f"p3_seed{a.seed}", "r1": lambda: f"r1_seed{a.seed}_{a.world}",
               "remake": lambda: f"remake_{a.probe}"}[a.cmd]()
        H, prereg, inputs, ledger = setup_process(pid, r7, a.cpu, need_final=a.cmd in ("r1", "remake"),
                                                  ledger_path=a.ledger, load_harness=a.cmd != "r1")
        if a.cmd == "p1":
            rec = run_p1(H, a.seed, a.world, r7, inputs, log=K_log)
            print(json.dumps(dict(probe=pid, result=result_line(rec), forecast=rec["forecast"]), sort_keys=True))
        elif a.cmd == "p2":
            rec = run_p2(H, a.seed, r7, inputs, log=K_log)
            print(json.dumps(dict(probe=pid, result=result_line(rec), actor=rec["actor"]), sort_keys=True))
        elif a.cmd == "p3":
            print(json.dumps(run_p3(H, a.seed), sort_keys=True))
        elif a.cmd == "r1":
            VF.verify_v7_chain(prereg)
            import v7_refs
            entry = v7_refs.lookup(v7_refs.table(prereg.payload),
                                   dict(phase="policy_train" if a.world == "stall" else "policy_no_stall_train",
                                        seed=a.seed, world=a.world))
            print(json.dumps(run_r1(a.seed, a.world, r7, entry, log=K_log), sort_keys=True))
        else:
            handoff = (prereg.payload.get("handoff") or {})
            pins = (handoff.get("probe_records") or {}).get(a.probe, {}).get("file_pins")
            if not pins:
                raise K.V7Refusal(f"the final prereg_v7.json pins no files for probe {a.probe}")
            print(json.dumps(remake(H, a.probe, r7, pins, log=K_log), sort_keys=True))
    except K.V7Refusal as error:
        LG.hard_exit(str(error), ledger=ledger)
    if ledger is not None:
        ledger.event("py-exit", code=0, **LG.usage())
    return 0


def K_log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


if __name__ == "__main__":
    sys.exit(main())
