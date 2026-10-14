"""Stage 2, version 7: the provenance files, the results to commit, and the list of run files for the orphan branch
(V7_SPEC.md C10 'The cost of this choice', 'Returning results'; Q6). Read only outside R7/provenance_*.json and
V7/results/. Never edits or recomputes a result: every number here is copied from a record.

  make_results_v7.py results [--partial]
      Copies into V7/results/files/ (flat names, '/' -> '__'; .log -> .txt because .gitignore:13 ignores *.log, as
      GPU/lib/make_results.py:132 does): every JSON record the laptop's phases wrote under OUT/ with its sidecar; the
      ledger (R7/ledger/: attempt records, sidecars and write logs); the provenance files; the probe records and the
      self-test reports; the retry, review, interruption, remake and R1 records; the restore and status records; the
      logs. Writes V7/results/BUNDLE.json (+ .sha256): an index of every file with its SHA-256, the fingerprints, the
      machine facts, the reference files' SHA-256, the SHA-256 of every inherited file as found on the laptop now, the
      process-hours, and the list of unfinished phases (empty when the run finished). No model (.msgpack) or array
      (.npz) file is copied (.gitignore:11-12). Without --partial it refuses until every phase is done, inherited,
      skipped on record or unfinished.
  make_results_v7.py provenance --group exploratory|confirmatory
      R7/provenance_<group>.json (+ .sha256), rewritten each time: per seed and component (detection, policy per
      world, deployment per world, host per world) the source (inherited from the cloud under v6, or made on the laptop
      under v7), the reference used and its label (C7), the records' SHA-256, the machine facts, retries and unfinished
      phases, the probe, forecast and R1 results, the labels 'machine-caused: memory' (C2) and 'machine-caused (re-make
      failed)' (C7), the gate's and the re-made actor's points side by side, the process-hours, and the status note.
  make_results_v7.py runfiles-plan --dir DIR      (push_runfiles_v7.sh, step 14)
      The files for the orphan branch stage2-v7-run-files: every file under R7/ (lock files left out) and every file
      under OUT/ that is not in the final snapshot; each new OUT/ file must be named, with the same SHA-256, by a ledger
      attempt's writes. Refuses anything outside R7/ and those new OUT/ files, and refuses when an inherited file
      changed. Writes DIR/r7.list and DIR/out.list (NUL-separated), DIR/SUMS and DIR/env.sh.
  make_results_v7.py runfiles-check --tree TREE --dir DIR
      Refuses unless the git tree holds exactly the planned paths plus RUNFILES_SHA256SUMS.
"""

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
LIB = Path(__file__).resolve().parent
LAPTOP = LIB.parent
V7_DIR = LAPTOP.parent
CODE_DIR = V7_DIR / "code"
for _p in (str(LIB), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import v7_common as K  # noqa: E402
import v7_ledger as LG  # noqa: E402
import runner_v7 as RN  # noqa: E402

SITUATIONS = ("recorded", "big_waves", "big_kicks", "strong_current")
GATE_CHECK2 = "harm_return/base/alpha5/seed{s}/values.json"                     # R-stall's folder (C7)
ORPHAN_BRANCH = "stage2-v7-run-files"
APPROVER_NAME = "David Bayha"
SUMS_NAME = "RUNFILES_SHA256SUMS"
STATUS_NOTE = ("The pooled look's own status sentence is version 6's and cannot change. The exploratory group (seeds "
               "0-4) mixes machines: phases inherited from the cloud and phases run on the laptop under v7{remade}. "
               "Seeds 0-4 carry no claim.")


def _rec(path):
    state, body = K.read_record(path)
    return body if state == "ok" else None


def rel_to(path, base):
    return Path(os.path.abspath(path)).relative_to(os.path.abspath(base)).as_posix()


# --------------------------------------------------------------------------------------------- the ledger

def ledger_rows(cfg):
    """Every attempt record in R7/ledger/, as {phase id: [records sorted by attempt]} (probes included)."""
    rows = {}
    d = cfg.r7 / "ledger"
    if not d.is_dir():
        return rows
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        recs = []
        for n, w in RN.attempt_files(cfg.r7, sub.name):
            rec = RN.attempt_record(w)
            recs.append(dict(attempt=n, record=rec, writes_file=str(w)))
        rows[sub.name] = recs
    return rows


def _hours(rec):
    """(hours, estimated): the attempt's wall time; when its launch shell did not survive (no end time), up to the
    time of the last line of its write log (a lower bound, marked estimated)."""
    from datetime import datetime
    end, estimated = rec.get("end_utc"), False
    if not end and rec.get("end_utc_estimated"):
        end, estimated = rec["end_utc_estimated"], True
    try:
        a = datetime.strptime(rec["start_utc"], "%Y-%m-%dT%H:%M:%SZ")
        b = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
        return max(0.0, (b - a).total_seconds() / 3600.0), estimated
    except (KeyError, TypeError, ValueError):
        return None, False


def process_hours(cfg, rows=None):
    """Laptop process-hours from the ledger (wall time of each one-processor attempt), split as C2 asks. A killed
    attempt whose launch shell did not survive counts up to the last line of its write log (estimated_end_attempts)."""
    rows = ledger_rows(cfg) if rows is None else rows
    out = dict(phases_done=0.0, killed_or_interrupted=0.0, failed_or_refused=0.0, probes=0.0, unknown_attempts=0,
               estimated_end_attempts=0)
    for pid, recs in rows.items():
        for r in recs:
            rec = r["record"]
            h, estimated = _hours(rec) if rec else (None, False)
            if h is None:
                out["unknown_attempts"] += 1
                continue
            if estimated:
                out["estimated_end_attempts"] += 1
            kind = (rec.get("classification") or {}).get("kind")
            if pid.startswith("probe_"):
                out["probes"] += h
            elif kind == "done":
                out["phases_done"] += h
            elif kind in RN.INTERRUPTED:
                out["killed_or_interrupted"] += h
            else:
                out["failed_or_refused"] += h
    out = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}
    out["total"] = round(sum(v for k, v in out.items() if k not in ("unknown_attempts", "estimated_end_attempts")), 3)
    out["note"] = ("at the laptop's speed; wall time of one-processor processes, from the ledger's start and end times; "
                   "an attempt whose launch shell did not survive counts up to the last line of its write log (a lower "
                   "bound)")
    return out


def machine_summary(rows):
    keys = ("cpu_model", "avx512f", "openblas", "glibc", "kernel_release", "python", "logical_processors")
    seen = {}
    for recs in rows.values():
        for r in recs:
            facts = (r["record"] or {}).get("facts") or {}
            for k in keys:
                if k in facts:
                    v = json.dumps(facts[k], sort_keys=True)
                    seen.setdefault(k, set()).add(v)
    return {k: [json.loads(v) for v in sorted(vs)] for k, vs in seen.items()}


# --------------------------------------------------------------------------------------------- the plan's state

def plan_state(cfg):
    """(tasks, phases state from HANDOFF.json, version, training, unfinished phase ids) for the whole plan."""
    version, training = RN.run_conditions(cfg)
    tasks = RN.build_plan(version, training, cfg.seeds, cfg.pooled)
    phases = RN.handoff_phases(cfg)
    unfinished = []
    for t in tasks:
        if phases.get(t.pid) == "inherited" or LG.phase_done(t.key, cfg.out)[0]:
            continue
        mode, _, _ = RN.need(cfg, t.pid, write=False)
        if mode == "unfinished":
            unfinished.append(t.pid)
    return tasks, phases, version, training, sorted(unfinished)


def settle_everything(cfg, out=print):
    """The record of every attempt left without one (C5, C10): phase attempts whose phase lock is free, and probe
    attempts when no self-test is running."""
    import selftest_v7 as ST
    written = RN.settle_all(cfg)
    try:
        with ST.SelfTestLock(cfg):
            written += RN.settle_all(cfg, probes=True, phases=False)
    except K.V7Refusal:
        out("WARN a self-test is running; its probe attempts are settled later")
    if written:
        out(f"wrote {len(written)} missing attempt records: {', '.join(f'{p} {n} ({k})' for p, n, k in written[:8])}")
    return written


def source_of(cfg, t, state, rows):
    """C10 provenance 'source', from the ledger's evidence, not from HANDOFF.json alone."""
    if state == "inherited":
        return "v6-cloud inherited"
    if state in ("done", "failed on record"):
        recs = {r["attempt"]: r["record"] for r in rows.get(t.pid, [])}
        ok, why = RN.ledger_evidence(cfg, t.key, recs)
        return "v7-laptop" if ok else f"UNEXPLAINED: done on file, but {why}"
    if state == "skipped":
        return "skipped on record (v6's rule)"
    return "v7-laptop (not done yet)"


def task_state(cfg, t, phases, version):
    if phases.get(t.pid) == "inherited":
        return "inherited"
    done, how, _ = LG.phase_done(t.key, cfg.out)
    if done:
        return "done" if how == "final" else "failed on record"
    if RN.skip_reason(cfg, t, version):
        return "skipped"
    return "not done"


# --------------------------------------------------------------------------------------------- provenance

COMPONENTS = (
    ("detection", lambda k: k["phase"].startswith(("detect_", "fresh_"))),
    ("policy stall", lambda k: k["phase"] in ("policy_train", "policy_eval")),
    ("policy no_stall", lambda k: k["phase"] in ("policy_no_stall_train", "policy_no_stall_eval")),
    ("deploy stall", lambda k: k["phase"] == "deploy" and k.get("world") == "stall"),
    ("deploy no_stall", lambda k: k["phase"] == "deploy" and k.get("world") == "no_stall"),
    ("host stall", lambda k: k["phase"] in ("host_train", "host_eval") and k.get("world") == "stall"),
    ("host no_stall", lambda k: k["phase"] in ("host_train", "host_eval") and k.get("world") == "no_stall"),
)


def _final_shas(cfg, key):
    return {f: K.file_sha256(cfg.out / f) for f in LG.row_existing_files(key, cfg.out) if f.endswith(".json")}


def _points_gate(cfg, s, key):
    rec = _rec(Path(cfg.extra.get("gate_variant_dir", K.GATE_VARIANT_DIR)) / GATE_CHECK2.format(s=s))
    if rec is None:
        return None
    ev = (rec.get("evaluations") or {}).get(key) or {}
    return {sit: (ev.get(sit) or {}).get("points") for sit in SITUATIONS}


def _points_laptop(cfg, entry):
    s = entry["seed"]
    if entry["phase"] == "host_train":
        rec = _rec(cfg.out / "host" / "stall" / f"seed{s}" / "values.json")
        pts = ((rec or {}).get("points") or {}).get("plain") or {}
        return {"plain": {sit: pts.get(sit) for sit in SITUATIONS}} if rec else None
    rel = f"policy/seed{s}/values.json" if entry["phase"] == "policy_train" else f"policy/seed{s}/no_stall/values.json"
    rec = _rec(cfg.out / rel)
    if rec is None:
        return None
    out = {}
    for k, spec in entry["replace"].items():
        ev = (rec.get("evaluations") or {}).get(spec["lock_name"]) or {}
        out[spec["lock_name"]] = {sit: (ev.get(sit) or {}).get("points") for sit in SITUATIONS}
    return out


def side_by_side(cfg, entry):
    """C7 Labels: the gate's recorded check-2 points of each replaced actor and the points the evaluation gives the
    re-made actor, per situation both report. Both values already exist; nothing is recomputed."""
    gate_key = {"gapped": "gapped", "full": "full", "no_stall_gapped": "no_stall_gapped", "actor": "td3bc_gapped"}
    rows = {}
    lap = _points_laptop(cfg, entry) or {}
    for k, spec in entry["replace"].items():
        rows[spec["lock_name"]] = dict(gate_actor=gate_key[k], gate_points=_points_gate(cfg, entry["seed"], gate_key[k]),
                                       laptop_points=lap.get(spec["lock_name"]),
                                       source_of_gate_points=GATE_CHECK2.format(s=entry["seed"]))
    return rows


def write_provenance(cfg, group):
    """R7/provenance_<group>.json (+ sidecar), rewritten each time. Returns the path."""
    import v7_verify as VF
    prereg = VF.load_prereg_v7(cfg.prereg)
    h = prereg.payload.get("handoff") or {}
    handoff = RN.read_handoff(cfg) or {}
    labels = h.get("labels") or {}
    entries = {(e["phase"], e["seed"], e["world"]): e for e in (h.get("remake_table") or [])}
    tasks, phases, version, training, unfinished = plan_state(cfg)
    rows = ledger_rows(cfg)
    seeds = LG.EXPLORATORY if group == "exploratory" else LG.CONFIRMATORY
    errors = RN.standin_errors(cfg, phases)
    seeds_out = {}
    for s in seeds:
        comps = {}
        for name, match in COMPONENTS:
            mine = [t for t in tasks if t.seed == s and match(t.key)]
            if name == "detection" and s in LG.EXPLORATORY:
                comps[name] = dict(source="inherited (cloud, v6)", note="detection on seeds 0-4 does not run again")
                continue
            if not mine:
                continue
            ph = {}
            for t in mine:
                st = task_state(cfg, t, phases, version)
                atts = [dict(attempt=r["attempt"], classification=(r["record"] or {}).get("classification"),
                             start_utc=(r["record"] or {}).get("start_utc"), end_utc=(r["record"] or {}).get("end_utc"),
                             reference=((r["record"] or {}).get("reference") or {}).get("label"),
                             retry_request=(r["record"] or {}).get("retry_request"))
                        for r in rows.get(t.pid, [])]
                ph[t.pid] = dict(source=source_of(cfg, t, st, rows), state=st,
                                 records_sha256=_final_shas(cfg, t.key), attempts=atts,
                                 retries=sorted(RN.retry_requests(cfg.r7, t.pid)),
                                 unfinished=t.pid in unfinished)
            label = labels.get(f"seed{s} {name}") or ("laptop, own run (fresh seed)" if s in LG.CONFIRMATORY else
                                                      "inherited (cloud, v6)" if all(
                                                          v["state"] == "inherited" for v in ph.values())
                                                      else "laptop, gate's reference (v6 as written)")
            comp = dict(phases=ph, reference_label=label, labels=[])
            world = "no_stall" if "no_stall" in name else "stall"
            if name.startswith("policy"):
                train = "policy_train" if world == "stall" else "policy_no_stall_train"
                for e in errors:
                    if e["pid"] == f"{train}_seed{s}":
                        comp["labels"].append(dict(label="machine-caused: memory" if e["memory"] else
                                                   "stand-in error (not memory)", field=e["field"], error=e["error"]))
                stop = cfg.out / "policy" / f"seed{s}" / ("no_stall" if world == "no_stall" else "") / "stop.json"
                if label == "laptop, re-make failed" and _rec(stop) is not None:
                    comp["labels"].append(dict(label="machine-caused (re-make failed)", stop=rel_to(stop, cfg.out)))
                e = entries.get((train, s, world))
                if e:
                    comp["side_by_side"] = side_by_side(cfg, e)
                    comp["entry"] = e
            if name == "host stall":
                e = entries.get(("host_train", s, "stall"))
                if e:
                    comp["side_by_side"] = side_by_side(cfg, e)
                    comp["entry"] = e
                    comp["labels"].append(dict(
                        label="the host test's 'plain' (laptop) and the deployment check's 'td3bc' (the gate's) are "
                              "different actors",
                        largest_parameter_difference=e.get("largest_parameter_difference"),
                        lock_text_note=("the lock's fixed text records['plain']['source'] ('the gate's saved run', "
                                        "S2C/host.py:355) does not describe this seed: its 'plain' was re-made on the "
                                        "laptop")))
            comps[name] = comp
        seeds_out[f"seed{s}"] = comps
    remade = [f"seed {e['seed']} {e['phase']} ({e['world']})" for e in entries.values() if e["seed"] in seeds]
    info = {p.name: _rec(p) for p in sorted((cfg.r7 / "info").glob("r1_*.json"))} if (cfg.r7 / "info").is_dir() else {}
    body = dict(
        kind="stage2-v7-provenance", group=group, written_utc=K.utc_now(), recomputes_nothing=True,
        prereg_v7_sha256=prereg.sha256, v6_prereg_sha256=K.V6["prereg_sha256"],
        v6_approval_sha256=K.V6["runs_side_approval_sha256"], version=version, training=training,
        status_note=STATUS_NOTE.format(remade=(", and references re-made on the laptop (" + "; ".join(remade) + ")")
                                       if remade else "") if group == "exploratory" else
        "The confirmatory group (seeds 5-9) ran entirely on the laptop under v7; check (b) compared two laptop programs.",
        seeds=seeds_out, unfinished=[t.pid for t in tasks if t.group == group and t.pid in unfinished],
        probes=dict(records=h.get("probe_records"), forecast=h.get("forecast"), remake_table=h.get("remake_table"),
                    labels=labels),
        r1=info, machine=machine_summary(rows),
        process_hours=dict(laptop=process_hours(cfg, rows), cloud=handoff.get("cloud_process_hours"),
                           v6_estimate="about 72-92 process-hours (an estimate that stops nothing)"),
        retries={pid: sorted(RN.retry_requests(cfg.r7, pid)) for pid in rows if RN.retry_requests(cfg.r7, pid)},
        standin_errors=errors)
    path = cfg.r7 / f"provenance_{group}.json"
    import selftest_v7 as ST
    ST.write_status_json(path, body)
    return path


# --------------------------------------------------------------------------------------------- the results

def encode(rel):
    return rel.replace("/", "__")


def collect(cfg, tasks, phases):
    """(absolute path, repository-relative path, kind) of everything the bundle holds."""
    out, seen = [], set()

    def add(p, kind):
        p = Path(p)
        if not p.is_file() or p.suffix in (".msgpack", ".npz") or p.name.endswith((".lock", ".tmp")):
            return
        rel = rel_to(p, cfg.root)
        if rel not in seen:
            seen.add(rel)
            out.append((p, rel, kind))

    for t in tasks:
        if phases.get(t.pid) == "inherited":
            continue
        for f in LG.row_existing_files(t.key, cfg.out):
            if f.endswith((".json", ".json.sha256")):
                add(cfg.out / f, "result" if f.endswith(".json") else "sidecar")
    r7 = cfg.r7
    for sub, kind in (("ledger", "ledger"), ("selftest", "self-test"), ("retry", "retry"), ("reviewed", "review"),
                      ("info", "R1"), ("refs_remake", "remake"), ("logs", "log")):
        d = r7 / sub
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if sub == "selftest" and p.suffix == ".msgpack":
                    continue
                add(p, kind if not p.name.endswith(".log") else "log")
    d = r7 / "interrupted"
    if d.is_dir():                      # the move records (<phase>/<UTC time>.json), not the moved files themselves
        for p in sorted(d.glob("*/*")):
            if p.name.endswith((".json", ".json.sha256")):
                add(p, "interruption record")
    for name in ("provenance_exploratory.json", "provenance_confirmatory.json", "restore_last.json", "status.json",
                 "runner.log", "launch_stderr.txt"):
        for p in (r7 / name, r7 / (name + ".sha256")):
            add(p, "record")
    return out


def inherited_now(cfg):
    """{path: SHA-256 as found now} for every inherited run file (the final snapshot's sums minus OUT/logs/ and the
    partial files), and the snapshot facts."""
    import selftest_v7 as ST
    handoff = RN.read_handoff(cfg)
    if handoff is None:
        return {}, None
    snap = handoff.get("snapshot") or {}
    code, data = ST.git(cfg, "show", f"{snap.get('commit')}:{SUMS_NAME}", binary=True)
    if code != 0 or K.sha256_bytes(data) != snap.get("sums_sha256"):
        raise K.V7Refusal("the final snapshot's sums are not readable here (run restore_v7.sh)")
    sums = ST.parse_sums(data)
    partial = set(handoff.get("partial_files") or [])
    now = {}
    for path in sorted(sums):
        if path.startswith(ST.LOGS_REL + "/") or path in partial:
            continue
        p = cfg.root / path
        now[path] = K.file_sha256(p) if p.is_file() else None
    return now, dict(commit=snap.get("commit"), sums_sha256=snap.get("sums_sha256"),
                     unchanged=all(now[p] == sums[p] for p in now), changed=[p for p in now if now[p] != sums[p]][:20])


def make_results(cfg, partial=False, out=print):
    import v7_verify as VF
    if RN.held_phase_locks(cfg.r7):
        raise K.V7Refusal(f"phases are running ({RN.held_phase_locks(cfg.r7)}); gather the results after they end")
    settle_everything(cfg, out)
    tasks, phases, version, training, unfinished = plan_state(cfg)
    states = {t.pid: task_state(cfg, t, phases, version) for t in tasks}
    open_ = [p for p, s in states.items() if s == "not done" and p not in unfinished]
    if open_ and not partial:
        raise K.V7Refusal(f"{len(open_)} phases are neither done nor unfinished ({open_[:5]}); use --partial for an "
                          f"interim copy")
    for g in ("exploratory", "confirmatory"):
        try:
            write_provenance(cfg, g)
        except K.V7Refusal as error:
            out(f"WARN provenance {g}: {error}")
    prereg = VF.load_prereg_v7(cfg.prereg)
    results = cfg.v7_root / "results"
    files_dir = results / "files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    files_dir.mkdir(parents=True)
    rows, total = [], 0
    for src, rel, kind in collect(cfg, tasks, phases):
        data = src.read_bytes()
        name = encode(rel) + (".txt" if rel.endswith(".log") else "")
        (files_dir / name).write_bytes(data)
        row = dict(name=name, path=rel, kind=kind, sha256=K.sha256_bytes(data), bytes=len(data))
        if rel.endswith(".json"):           # a record with a sidecar must match it; every result has one
            side = K.sidecar_digest(src)
            row["sidecar_ok"] = None if side is None and kind != "result" else side == row["sha256"]
        rows.append(row)
        total += len(data)
    inh, snap = inherited_now(cfg)
    refs = {rel_to(p, cfg.r7): K.file_sha256(p) for p in sorted((cfg.r7 / "refs").rglob("*")) if p.is_file()} \
        if (cfg.r7 / "refs").is_dir() else {}
    lrows = ledger_rows(cfg)
    status = _rec(cfg.r7 / "status.json") or (json.loads((cfg.r7 / "status.json").read_text())
                                              if (cfg.r7 / "status.json").exists() else {})
    finished = not unfinished and not open_
    bundle = dict(
        kind="stage2-v7-results", partial=not finished,
        status=("FINAL: every phase is done, inherited or skipped on record" if finished else
                "PARTIAL: phases are unfinished or not done" + (f" (unfinished: {', '.join(unfinished)})"
                                                                 if unfinished else "")),
        created=K.utc_now(), test=cfg.test, unfinished=unfinished, not_done=open_,
        fingerprints=dict(prereg_v7_sha256=prereg.sha256, code_manifest_v7_sha256=(prereg.payload.get("code") or {})
                          .get("code_manifest_v7_sha256"), v6_prereg_sha256=K.V6["prereg_sha256"],
                          v6_code_manifest_sha256=K.V6["code_manifest_sha256"],
                          v6_approval_sha256=K.V6["runs_side_approval_sha256"], constants_sha256=K.CONSTANTS_SHA256,
                          handoff_sha256=K.file_sha256(cfg.handoff) if cfg.handoff.exists() else None,
                          gate_prereg_sha256=K.GATE_PREREG_SHA256),
        version=version, training=training, phases=states, machine=machine_summary(lrows),
        machine_now=dict(kernel=platform.release(), python=platform.python_version()),
        process_hours=process_hours(cfg, lrows), reference_files=refs, inherited_snapshot=snap,
        inherited_files=inh, runner=dict(lanes=status.get("lanes"), cpus=status.get("cpus"), halted=status.get("halted")),
        files=rows, total_bytes=total,
        excluded="model files (.msgpack) and arrays (.npz) stay on the laptop; they go to the orphan branch (step 14)")
    text = K.canonical(bundle).encode("utf8")
    (results / "BUNDLE.json").write_bytes(text)
    (results / "BUNDLE.json.sha256").write_text(f"{K.sha256_bytes(text)}  BUNDLE.json\n", encoding="utf8")
    out(f"results: {len(rows)} files, {total} bytes in {results} ({'final' if finished else 'PARTIAL'}"
        + (f"; UNFINISHED: {', '.join(unfinished)}" if unfinished else "") + ")")
    return bundle


# --------------------------------------------------------------------------------------------- the orphan branch (Q6)

def runfiles_plan(cfg, out_dir):
    """The run files for stage2-v7-run-files. V7Refusal on anything outside R7/ and the laptop's new OUT/ files."""
    import selftest_v7 as ST
    handoff = RN.read_handoff(cfg)
    if handoff is None:
        raise K.V7Refusal("no HANDOFF.json: there is nothing of v7's to push yet")
    if RN.held_phase_locks(cfg.r7):
        raise K.V7Refusal(f"phases are running ({RN.held_phase_locks(cfg.r7)}); push the run files after they end")
    settle_everything(cfg, out=lambda *a: None)
    bundle = _rec(cfg.v7_root / "results" / "BUNDLE.json")
    if bundle is None:
        raise K.V7Refusal("results/BUNDLE.json is missing: do step 13 (the results) first")
    snap = handoff.get("snapshot") or {}
    code, data = ST.git(cfg, "show", f"{snap.get('commit')}:{SUMS_NAME}", binary=True)
    if code != 0 or K.sha256_bytes(data) != snap.get("sums_sha256"):
        raise K.V7Refusal("the final snapshot's sums are not readable here (run restore_v7.sh)")
    sums = ST.parse_sums(data)
    partial = set(handoff.get("partial_files") or [])
    inherited = {p: s for p, s in sums.items() if not p.startswith(ST.LOGS_REL + "/") and p not in partial}
    out_rel, r7_rel = rel_to(cfg.out, cfg.root), rel_to(cfg.r7, cfg.root)
    changed = [p for p, sha in inherited.items() if p.startswith(out_rel + "/") and (cfg.root / p).is_file()
               and K.file_sha256(cfg.root / p) != sha]
    if changed:
        raise K.V7Refusal(f"inherited files changed on this laptop: {changed[:5]}; nothing is pushed; ask the cloud")
    written = {}
    for recs in ledger_rows(cfg).values():
        for r in recs:
            for w in (r["record"] or {}).get("writes") or []:
                if w.get("sha256"):
                    written.setdefault(os.path.abspath(w["path"]), set()).add(w["sha256"])
    new_out, bad = [], []
    for p in sorted(cfg.out.rglob("*")) if cfg.out.is_dir() else []:
        if not p.is_file():
            continue
        rel = rel_to(p, cfg.root)
        if rel in inherited or rel.startswith(ST.LOGS_REL + "/"):    # a partial file's path is the laptop's now
            continue
        sha = K.file_sha256(p)
        if sha not in written.get(os.path.abspath(p), set()):
            bad.append(rel)
        else:
            new_out.append(rel)
    if bad:
        raise K.V7Refusal(f"files under OUT/ that no v7 ledger attempt wrote (with this SHA-256): {bad[:5]}; nothing is "
                          f"pushed; ask the cloud")
    r7_files = [rel_to(p, cfg.root) for p in sorted(cfg.r7.rglob("*"))
                if p.is_file() and not p.name.endswith((".lock", ".tmp")) and not p.name.startswith(".restore-")]
    for rel in r7_files + new_out:
        if not (rel.startswith(r7_rel + "/") or (rel.startswith(out_rel + "/") and rel not in inherited)):
            raise K.V7Refusal(f"refused: {rel} lies outside R7/ and the new OUT/ files")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r7.list").write_bytes(b"".join(p.encode("utf8") + b"\0" for p in r7_files))
    (out_dir / "out.list").write_bytes(b"".join(p.encode("utf8") + b"\0" for p in new_out))
    lines = [f"{K.file_sha256(cfg.root / p)}  {p}\n" for p in sorted(r7_files + new_out)]
    (out_dir / SUMS_NAME).write_text("".join(lines), encoding="utf8")
    date = bundle.get("created") or K.utc_now()
    env = dict(REPO_DIR=str(cfg.root), ORPHAN_BRANCH=cfg.extra.get("orphan_branch", ORPHAN_BRANCH),
               COMMIT_DATE=date.replace("T", " ").replace("Z", " +0000"), AUTHOR_NAME=APPROVER_NAME,
               AUTHOR_EMAIL=K.APPROVER_EMAIL, R7_COUNT=str(len(r7_files)), OUT_COUNT=str(len(new_out)),
               SNAPSHOT=str(snap.get("commit")))
    (out_dir / "env.sh").write_text("".join(f"{k}='{v}'\n" for k, v in env.items()), encoding="utf8")
    return dict(r7=len(r7_files), out=len(new_out), bytes=sum((cfg.root / p).stat().st_size for p in r7_files + new_out))


def runfiles_check(cfg, tree, plan_dir):
    import selftest_v7 as ST
    plan_dir = Path(plan_dir)
    planned = set()
    for name in ("r7.list", "out.list"):
        planned |= {p.decode("utf8") for p in (plan_dir / name).read_bytes().split(b"\0") if p}
    code, listing = ST.git(cfg, "ls-tree", "-r", "--name-only", tree)
    if code != 0:
        raise K.V7Refusal(f"git ls-tree {tree} failed")
    have = set(listing.splitlines())
    extra = sorted(have - planned - {SUMS_NAME})
    missing = sorted(planned - have)
    if extra or missing:
        raise K.V7Refusal(f"refused: the tree to push holds paths outside the plan {extra[:5]} or misses {missing[:5]}")
    return len(have)


def main(argv=None, out=print):
    argv = list(sys.argv[1:] if argv is None else argv)
    fixture = None
    if argv[:1] == ["--v7-test"]:
        fixture, argv = argv[1], argv[2:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("results")
    r.add_argument("--partial", action="store_true")
    p = sub.add_parser("provenance")
    p.add_argument("--group", required=True, choices=("exploratory", "confirmatory"))
    q = sub.add_parser("runfiles-plan")
    q.add_argument("--dir", required=True)
    c = sub.add_parser("runfiles-check")
    c.add_argument("--tree", required=True)
    c.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    try:
        cfg = RN.config(fixture)
        if a.cmd == "results":
            make_results(cfg, a.partial, out)
        elif a.cmd == "provenance":
            out(str(write_provenance(cfg, a.group)))
        elif a.cmd == "runfiles-plan":
            res = runfiles_plan(cfg, a.dir)
            out(f"run files to push: {res['r7']} under R7/ and {res['out']} new under OUT/ ({res['bytes']} bytes)")
        else:
            out(f"tree {a.tree}: {runfiles_check(cfg, a.tree, a.dir)} paths, all planned")
    except K.V7Refusal as error:
        text = str(error)
        out(text if text.startswith("refused") else f"refused: {text}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
