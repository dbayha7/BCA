"""EXPLORATORY: result F's seed-0 sandbox rerun -- the driver's small checks. Standard library only (no JAX), so the
runner can call it often and fast (run it with python -I -B).

It only reads, except: <resultF>/state/*.json (each written atomically: a temporary file, then os.replace), the start
marker, the EXPLORATORY label file in each of the sandbox's two runs/ roots, and (quarantine only) moving an
interrupted arm folder that this run itself wrote under the sandbox's runs/ tree into <resultF>/state/quarantine/. It
never writes into the repository (/home/user/bca), never touches a sandbox file that existed before the run, and never
writes or edits an approval, spec, pass-mark or declined file.

  rf.py done JOB         exit 0 if the job's output is finished and current, 1 if not (job names: run_resultF.sh).
                         Every record is parsed: a record that does not parse, lacks a key the check reads, or fails
                         its record checksum counts as not finished. Extra names: 'evalF-partial' (eval_resultF.py's
                         outputs from `run --allow-missing`), 'a2ok_<Q>' (result F's actor A2 trained and intact)
  rf.py d1 Q [--dry]     the new data seeds' D1 numbers (seeds 0-4) against the committed version-5 report's, bit for
                         bit; writes state/d1_<Q>.json (--dry: state/dry_d1_<Q>.json); exit 0 (identical or not: the
                         verdict is in the file), 2 if a data record is missing
  rf.py baseline         once: SHA-256 of every file of the sandbox (iw_rerun, outside __pycache__) and of the
                         repository files this run reads in place; and the start marker
  rf.py provenance       once: the files data_seeds1to4.sh (started by the main session before this run) built in the
                         sandbox: data seeds 1-4 and the report of each quality, with SHA-256, size and time; writes
                         state/data_seeds1to4.json
  rf.py untouched        no file of the baseline changed or vanished; no new file in the sandbox outside
                         repo/runs/ or in the repository folders read in place; exit 0 only if all hold. Paths in the
                         repository (outside .git) newer than the start marker are listed as a warning. Every path
                         listed comes with its time and the jobs of this run that were running then
                         (state/job_times.tsv). Writes state/untouched.json
  rf.py label [--final]  the EXPLORATORY label file in both runs/ roots; --final (at the end of every run) adds the
                         explicit list of every file this run created and of data_seeds1to4.sh's outputs
  rf.py quarantine JOB   (f1train_Q, esttrain_Q) move an interrupted arm folder of this run (record.json that does not
                         parse, or an actor that does not match it) to state/quarantine/, so the script retrains it
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

RF = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("RF_SRC", "/home/user/bca"))
IW = Path(os.environ.get("RF_IW", "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun"))
QUALITIES = tuple(os.environ.get("RF_QUALITIES", "medium-expert expert").split())
REPO = IW / "repo"
V5 = REPO / "runs" / "shift_iw_v5"
EX = REPO / "runs" / "shift_iw_v5_explore"
F1 = EX / "channel" / "train_f1"
ESTD = EX / "channel" / "train_est"
CHAN = SRC / "outputs" / "shift_iw" / "explore" / "channel"
STATE = RF / "state"
# baseline and untouched keep their files here (the dry mode uses its own folder, so the real run takes its
# baseline at its own first start)
BASE = Path(os.environ.get("RF_BASELINE_DIR", str(STATE)))
SEED = 0
SEEDS = (0, 1, 2, 3, 4)
F1_ARMS = ("A2_cmd_strong", "B2_cmd_train", "C2_untargeted")
EST_ARMS = ("A3_estimated", "C3_untargeted")
LAWS = ("none", "mild", "strong")
DATA_SCRIPT = RF / "data_seeds1to4.sh"
# the repository files this run reads in place (scripts run from their own folder; the approval through the
# sandbox's symbolic link), checked unchanged at the end
SRC_READ = ("outputs/shift_iw/explore/channel", "outputs/shift_iw/explore/explore_arms.py",
            "outputs/shift_iw/explore/explore_eval.py", "outputs/shift_iw/explore/RESULTS.json",
            "outputs/shift_iw/explore/curves/results", "outputs/shift_iw/v5", "experiments/iw",
            "experiments/shift_iw")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_atomic(path, text):
    """Write a state file whole or not at all: a temporary file beside it, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def when(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))


# ------------------------------------------------------------------------------------------------ records

def spec_sha():
    """The specification digest every package record carries (the sandbox's iw/spec.json sidecar; gates.py checks it
    equals the committed one)."""
    return (REPO / "experiments" / "iw" / "spec.json.sha256").read_text().split()[0]


def rec_ok(path):
    """A record and its sha256sum record checksum, matching (the package's own write_json pair)."""
    path = Path(path)
    side = Path(str(path) + ".sha256")
    if not path.is_file() or not side.is_file():
        return False
    words = side.read_text().split()
    return bool(words) and words[0] == sha(path)


def parsed(path):
    """The JSON body of a file, or None if it is missing or does not parse into an object."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None


def finished(path):
    """The package's run.finished: record checksum matches, parses, status finished, this specification."""
    if not rec_ok(path):
        return None
    rec = parsed(path)
    return rec if rec is not None and rec.get("status") == "finished" and rec.get("spec_sha256") == spec_sha() else None


def mtime(path):
    return Path(path).stat().st_mtime


def arms_trained(root, quality, arms, passmarks_file):
    want = (CHAN / (passmarks_file + ".sha256")).read_text().split()[0]
    for arm in arms:
        d = root / quality / arm / f"seed{SEED}"
        rec = parsed(d / "record.json")
        if rec is None or not (d / "actor.msgpack").is_file():
            return False
        if not (rec.get("exploratory") is True and str(rec.get("label", "")).startswith("EXPLORATORY")
                and rec.get("finite") is True and rec.get("passmarks_sha256") == want and rec.get("arm") == arm
                and rec.get("actor_sha256") == sha(d / "actor.msgpack") and rec.get("seed") == SEED
                and rec.get("quality") == quality and isinstance(rec.get("chunks"), list)
                and isinstance((rec.get("clone") or {}).get("actor_sha256"), str)):
            return False
    return True


def evaluated(root, quality, arms):
    """The script's evaluate_seed0.json holds every arm, finite on every bank, written after every arm's record."""
    p = root / quality / f"evaluate_seed{SEED}.json"
    body = parsed(p)
    if body is None or body.get("quality") != quality or body.get("seed") != SEED:
        return False
    ev = body.get("evaluations")
    if not isinstance(ev, dict):
        return False
    for arm in arms:
        r = root / quality / arm / f"seed{SEED}" / "record.json"
        if arm not in ev or not r.is_file() or mtime(p) < mtime(r):
            return False
        if not all(isinstance(ev[arm].get(law), dict) and ev[arm][law].get("finite") is True
                   and isinstance(ev[arm][law].get("J"), (int, float)) for law in LAWS):
            return False
    return True


def checked(path, keys, *inputs):
    """A checks file that parses, holds every key, and is newer than each of its inputs."""
    body = parsed(path)
    return (body is not None and all(k in body for k in keys)
            and all(Path(i).is_file() and mtime(path) >= mtime(i) for i in inputs))


def data_done(q, s):
    rec = finished(V5 / q / "data" / f"seed{s}" / "stage.json")
    return rec is not None and rec.get("seed") == s and rec.get("quality") == q and isinstance(rec.get("D1"), dict)


def report_done(q):
    """The quality's report on seeds 0-4 (run.py report, as data_seeds1to4.sh or this run built it): finished, made by
    the code that made the data, and newer than every data record (a report from before a data seed was built is
    stale; run.require_design would refuse it)."""
    p = V5 / q / "report" / "report.json"
    rep = finished(p)
    if rep is None or rep.get("quality") != q or sorted(rep.get("seeds") or []) != list(SEEDS):
        return False
    if not isinstance(((rep.get("marks") or {}).get("D1") or {}).get("status"), str):
        return False
    data = [V5 / q / "data" / f"seed{s}" / "stage.json" for s in SEEDS]
    if not all(data_done(q, s) for s in SEEDS):
        return False
    d0 = parsed(data[0])
    return rep.get("code_sha256") == d0.get("code_sha256") and all(mtime(p) >= mtime(d) for d in data)


def roomle_done(q):
    """room-lower-edge finished and did not stop the seed (check (e)): 'No extra caution' exists and is intact."""
    d = V5 / q / "room_lower_edge" / f"seed{SEED}"
    rec = finished(d / "record.json")
    if rec is None or rec.get("seed") != SEED:
        return False
    le = rec.get("room_lower_edge") or {}
    return (not le.get("stopped") and "le_none" in (le.get("names") or [])
            and isinstance((le.get("clone") or {}).get("actor_sha256"), str)
            and (d / "actors.msgpack").is_file() and rec.get("actors_sha256") == sha(d / "actors.msgpack"))


def a2_ok(q):
    """Result F's actor A2 trained and intact (its record parses and names its actor's SHA-256, finite, pass marks)."""
    return arms_trained(F1, q, ("A2_cmd_strong",), "PASSMARKS_fallback_rerun.md")


def done(job):
    if job == "evalF":
        return evalf_done(partial_ok=False)
    if job == "evalF-partial":
        return evalf_done(partial_ok=True)
    kind, rest = job.split("_", 1)
    q = rest.split("_s")[0]
    if q not in QUALITIES:
        raise SystemExit(f"unknown quality in job {job!r}")
    band = V5 / q / "band" / "gap" / f"seed{SEED}" / "record.json"
    roomle = V5 / q / "room_lower_edge" / f"seed{SEED}" / "record.json"
    data = [V5 / q / "data" / f"seed{s}" / "stage.json" for s in SEEDS]
    if kind == "a2ok":
        return a2_ok(q)
    if kind == "data":
        return data_done(q, int(rest.rsplit("_s", 1)[1]))
    if kind == "d1":
        body = parsed(STATE / f"d1_{q}.json")
        return (body is not None and body.get("kind") == "resultF-d1-check" and body.get("quality") == q
                and isinstance(body.get("all_identical"), bool)
                and sorted((body.get("seeds") or {}).keys()) == [str(s) for s in SEEDS]
                and checked(STATE / f"d1_{q}.json", (), *data))
    if kind == "report":
        return report_done(q)
    if kind == "room":
        rec = finished(V5 / q / "room" / f"seed{SEED}" / "record.json")
        return rec is not None and rec.get("seed") == SEED and isinstance(rec.get("room"), dict)
    if kind == "band":
        rec = finished(band)
        return rec is not None and rec.get("seed") == SEED
    if kind == "roomle":
        return roomle_done(q)
    if kind == "roomleeval":
        rec = finished(V5 / q / "evaluate" / f"seed{SEED}" / "room_lower_edge.json")
        key = f"{q}/room_lower_edge/seed{SEED}/record.json"
        ev = (rec or {}).get("evaluations") or {}
        return (bool(rec) and roomle.is_file() and (rec.get("inputs_sha256") or {}).get(key) == sha(roomle)
                and all(isinstance((ev.get(n) or {}).get(law, {}).get("J"), (int, float))
                        for n in ("le_none", "le_oracle") for law in LAWS))
    if kind == "f1train":
        return arms_trained(F1, q, F1_ARMS, "PASSMARKS_fallback_rerun.md")
    if kind == "f1eval":
        return evaluated(F1, q, F1_ARMS)
    if kind == "f1checks":
        return checked(F1 / q / f"checks_seed{SEED}.json",
                       ("check1_max_param_diff_vs_v5_unweighted_300_steps", "check2_refusal_types"), band, roomle)
    if kind == "esttrain":
        return arms_trained(ESTD, q, EST_ARMS, "PASSMARKS_estimated_weights.md")
    if kind == "esteval":
        return evaluated(ESTD, q, EST_ARMS)
    if kind == "estchecks":
        return checked(ESTD / q / f"checks_seed{SEED}.json", ("V1", "V2", "in_graph_agreement"), band, roomle)
    if kind == "v5le":
        d = EX / q / "bca_lower_edge" / "unweighted" / f"seed{SEED}"
        rec = finished(d / "record.json")
        return (rec is not None and rec.get("exploratory") is True and (d / "actor.msgpack").is_file()
                and rec.get("actor_sha256") == sha(d / "actor.msgpack"))
    if kind == "v5leeval":
        rec = finished(EX / q / "evaluate" / f"seed{SEED}" / "bca_lower_edge.json")
        r = EX / q / "bca_lower_edge" / "unweighted" / f"seed{SEED}" / "record.json"
        key = f"{q}/bca_lower_edge/unweighted/seed{SEED}/record.json"
        return (bool(rec) and "le_bca_unweighted" in (rec.get("evaluations") or {}) and r.is_file()
                and (rec.get("inputs_sha256") or {}).get(key) == sha(r))
    raise SystemExit(f"unknown job {job!r}")


def evalf_done(partial_ok):
    """eval_resultF.py's outputs exist and parse, and every file it read is unchanged since (scores_seed0.json,
    written last, names their SHA-256). Without partial_ok, also complete: every core learner on every quality
    (`eval_resultF.py run`, not `run --allow-missing`)."""
    files = {f: RF / "results" / f for f in ("scores_seed0.json", "episodes_seed0.json", "situations_seed0.json")}
    bodies = {f: parsed(p) for f, p in files.items()}
    if any(b is None for b in bodies.values()) or not (RF / "results" / "COMPARE.md").is_file():
        return False
    s = bodies["scores_seed0.json"]
    if not all(k in s for k in ("overall", "scores", "comparison", "complete", "inputs_sha256")):
        return False
    if not all(k in bodies["situations_seed0.json"] for k in ("avg", "episodes", "learner_facts", "seconds")):
        return False
    if not all(k in bodies["episodes_seed0.json"] for k in ("episodes", "learner_facts")):
        return False
    if not partial_ok and s.get("complete") is not True:
        return False
    ins = s.get("inputs_sha256") or {}
    return bool(ins) and all(Path(k).is_file() and sha(k) == v for k, v in ins.items())


# ------------------------------------------------------------------------------------------------ D1

def same_numbers(a, b):
    """Every number present in both equal exactly (as the sandbox's own arith check compares D1)."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) & set(b)
        return bool(keys) and all(same_numbers(a[k], b[k]) for k in keys)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return a == b
    return True


def d1(q, dry=False):
    rep_path = SRC / "outputs" / "shift_iw" / "v5" / q / "report.json"
    rep = json.loads(rep_path.read_text())
    desk = rep["marks"]["D1"]["per_seed"]
    out = dict(kind="resultF-d1-check", label="EXPLORATORY: a read-only check of the sandbox's new data",
               quality=q, committed_report=str(rep_path), committed_report_sha256=sha(rep_path),
               committed_D1_status=rep["marks"]["D1"]["status"], seeds={})
    for s in SEEDS:
        rec = finished(V5 / q / "data" / f"seed{s}" / "stage.json")
        if rec is None:
            print(f"d1 {q}: the data record of seed {s} is missing", file=sys.stderr)
            return 2
        out["seeds"][str(s)] = dict(identical=same_numbers(rec["D1"], desk[s]), sandbox=rec["D1"], desktop=desk[s])
    out["all_identical"] = all(v["identical"] for v in out["seeds"].values())
    dest = STATE / (f"dry_d1_{q}.json" if dry else f"d1_{q}.json")    # the dry mode's preview is not the run's file
    write_atomic(dest, json.dumps(out, indent=1))
    print(f"d1 {q}: seeds 0-4 identical to the desktop's bit for bit: {out['all_identical']} -> {dest}")
    return 0


# ------------------------------------------------------------------------------------------------ untouched

def sandbox_files():
    for p in sorted(IW.rglob("*")):
        if "__pycache__" in p.parts or p.is_dir():
            continue
        yield p


def src_files():
    for rel in SRC_READ:
        p = SRC / rel
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from (f for f in sorted(p.rglob("*")) if f.is_file() and "__pycache__" not in f.parts)


def manifest(files):
    rows = {}
    for p in files:
        if p.is_symlink():
            rows[str(p)] = "symlink:" + os.readlink(p)
        else:
            rows[str(p)] = sha(p)
    return rows


def baseline():
    BASE.mkdir(parents=True, exist_ok=True)
    marker = BASE / "start.marker"
    if not marker.exists():
        write_atomic(marker, time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    for name, files in (("baseline_sandbox.json", sandbox_files), ("baseline_repository.json", src_files)):
        p = BASE / name
        if not p.exists():
            write_atomic(p, json.dumps(dict(made_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), files=manifest(files())),
                                       indent=0))
    print(f"baseline: {BASE / 'baseline_sandbox.json'}, {BASE / 'baseline_repository.json'}, {marker}")
    return 0


def job_times():
    """[(job, cpu, start, end, exit code)] of this run's jobs (run_resultF.sh appends one line per job as it ends;
    a job still running has no line yet, so its pid file gives its start and 'now' its end)."""
    rows = []
    p = BASE / "job_times.tsv"
    if p.is_file():
        for line in p.read_text().splitlines():
            w = line.split("\t")
            if len(w) == 5:
                try:
                    rows.append((w[0], w[1], float(w[2]), float(w[3]), w[4]))
                except ValueError:
                    continue
    for f in STATE.glob("*.pid"):
        w = f.read_text().split()
        if f.name != "runner.pid" and len(w) == 3:
            rows.append((f.stem, w[1], float(w[2]), time.time(), "running"))
    return rows


def explain(paths, times, start):
    """Each path's modification and change times (lstat; None if gone) and the jobs of this run running then."""
    out = []
    for k in paths:
        row = dict(path=k)
        try:
            st = os.lstat(k)
        except FileNotFoundError:
            row.update(exists=False)
            out.append(row)
            continue
        t = max(st.st_mtime, st.st_ctime)
        row.update(exists=True, mtime=when(st.st_mtime), ctime=when(st.st_ctime), before_run_start=t < start,
                   jobs_running_then=[f"{j} (cpu {c}, {when(a)} to {when(b)}, exit {rc})"
                                      for j, c, a, b, rc in times if a - 2 <= t <= b + 2])
        out.append(row)
    return out


def repo_newer_than(start, limit=50):
    """Paths in the repository (outside every .git folder) modified or changed after the start marker."""
    newer = []
    for root, dirs, files in os.walk(SRC):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for name in sorted(dirs + files):
            p = os.path.join(root, name)
            try:
                st = os.lstat(p)
            except FileNotFoundError:
                continue
            if st.st_mtime > start or st.st_ctime > start:
                newer.append(p)
                if len(newer) >= limit:
                    return newer
    return newer


def untouched():
    out = dict(kind="resultF-untouched-check", checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    ok = True
    marker = BASE / "start.marker"
    start = marker.stat().st_mtime
    times = job_times()
    for name, files, label in (("baseline_sandbox.json", sandbox_files, "sandbox"),
                               ("baseline_repository.json", src_files, "repository")):
        base = json.loads((BASE / name).read_text())["files"]
        now = manifest(files())
        changed = sorted(k for k in base if k in now and now[k] != base[k])
        gone = sorted(k for k in base if k not in now)
        new = sorted(k for k in now if k not in base)
        if label == "sandbox":
            runs = str(REPO / "runs") + "/"
            new_outside = [k for k in new if not k.startswith(runs)]
            out[label] = dict(changed=changed, removed=gone, new_under_runs=len(new) - len(new_outside),
                              new_outside_runs=new_outside,
                              new_under_runs_files=[k for k in new if k.startswith(runs)])   # this run's outputs
            bad = changed + gone + new_outside
        else:
            out[label] = dict(changed=changed, removed=gone, new=new)
            bad = changed + gone + new
        if bad:
            ok = False
            out[label]["what_and_when"] = explain(bad, times, start)
    newer = repo_newer_than(start)
    # every job runs with -B and only reads the repository, so a path newer than the start is not expected; it is
    # listed as a warning (another session may have written there), not a failure of this run
    out["repository_newer_than_start_WARNING"] = explain(newer, times, start)
    out["run_started"] = when(start)
    out["all_untouched"] = bool(ok)
    write_atomic(BASE / "untouched.json", json.dumps(out, indent=1))
    if not ok:
        for label in ("sandbox", "repository"):
            for row in out[label].get("what_and_when", []):
                print(f"untouched FAILED, {label}: {row['path']}: " + (
                    "removed" if not row["exists"] else
                    f"mtime {row['mtime']}, ctime {row['ctime']}; jobs of this run running then: "
                    f"{row['jobs_running_then'] or 'none'}"), file=sys.stderr)
    print(f"untouched: {ok} (sandbox changed {len(out['sandbox']['changed'])}, removed "
          f"{len(out['sandbox']['removed'])}, new outside runs {len(out['sandbox']['new_outside_runs'])}, new under "
          f"runs {out['sandbox']['new_under_runs']}; repository files read in place changed "
          f"{len(out['repository']['changed'])}, removed {len(out['repository']['removed'])}, new "
          f"{len(out['repository']['new'])}; WARNING paths in the repository outside .git newer than the start: "
          f"{len(newer)}; details {BASE / 'untouched.json'})")
    return 0 if ok else 6


# ------------------------------------------------------------------------------------------------ provenance

def provenance():
    """Once, at the run's first start: the files data_seeds1to4.sh built in the sandbox before this run (the main
    session started it; this run's done checks accept them and do not rebuild them)."""
    p = STATE / "data_seeds1to4.json"
    if p.exists():
        print(f"provenance: {p} (kept from the first start)")
        return 0
    rows = []
    for q in QUALITIES:
        for d in [V5 / q / "data" / f"seed{s}" for s in (1, 2, 3, 4)] + [V5 / q / "report"]:
            for f in sorted(d.glob("*")) if d.is_dir() else []:
                if f.is_file():
                    st = f.stat()
                    rows.append(dict(path=str(f), sha256=sha(f), bytes=st.st_size, mtime=when(st.st_mtime)))
    body = dict(kind="resultF-data-seeds1to4-provenance",
                label="EXPLORATORY: scratch inputs of result F's seed-0 sandbox rerun, not version-5 results",
                script=str(DATA_SCRIPT), script_sha256=sha(DATA_SCRIPT) if DATA_SCRIPT.is_file() else None,
                logs=[str(x) for x in sorted((RF / "logs").glob("data_*.log")) + sorted((RF / "logs").glob("report_*.log"))
                      if x.name != "data_seeds1to4.log"],
                how=("data_seeds1to4.sh: run.py data --quality Q --seeds K for K = 1-4, then run.py report --quality Q, "
                     "for medium-expert and expert, with this run's processor settings (ARITH=desktop), "
                     "OMP_NUM_THREADS=1 and nice -n 10, but without taskset (not pinned); rf.py d1 checks their D1 "
                     "against the committed report bit for bit"),
                recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), files=rows)
    write_atomic(p, json.dumps(body, indent=1))
    print(f"provenance: {len(rows)} files built by data_seeds1to4.sh -> {p}")
    return 0


# ------------------------------------------------------------------------------------------------ labels

LABEL_TEXT = """EXPLORATORY: outside the version-5 pre-registration.

Every file in this folder tree created on or after {start} by result F's seed-0 sandbox rerun
({rf}/run_resultF.sh; plan: {rf}/PLAN.md) is EXPLORATORY: a cloud rerun of data seed 0, medium-expert and expert,
of the weighting test's result F ("Fixed fallback, exact weights", A2_cmd_strong) and its comparison rows, with the
committed code unchanged. The version-5 package's own stage records written here (data seeds 1-4, report, room,
room_lower_edge, band, evaluate) carry the package's labels; they are scratch inputs of that rerun, not version-5
results. Data seeds 1-4 and the report of each quality were built just before the rerun by {script} (started by the
main session); they are listed below by path.
"""


def own_files(start):
    """resultF's own files this run wrote (results, state, logs), not the dry mode's or the smoke tests'."""
    rows = []
    for sub in ("results", "state", "logs"):
        d = RF / sub
        for f in sorted(d.rglob("*")) if d.is_dir() else []:
            if (f.is_file() and f.stat().st_mtime >= start and "dry_baseline" not in f.parts
                    and not f.name.startswith(("dry", "gates_dry", "report_dry", "eval_check", "eval_smoke"))
                    and f.suffix != ".lock"):
                rows.append(str(f))
    return rows


def label(final=False):
    """The EXPLORATORY label file in each of the sandbox's two runs/ roots (the package's own stage records cannot
    carry the label themselves without changing the code). Written at the start; rewritten with --final at the end of
    every run, with the explicit list of every file this run created so far."""
    marker = BASE / "start.marker"
    start = marker.read_text().strip() if marker.is_file() else "this run's start"
    text = LABEL_TEXT.format(start=start, rf=RF, script=DATA_SCRIPT)
    if final:
        un = parsed(BASE / "untouched.json") or {}
        made = (un.get("sandbox") or {}).get("new_under_runs_files") or []
        prov = parsed(STATE / "data_seeds1to4.json") or {}
        mine = own_files(marker.stat().st_mtime) if marker.is_file() else []
        text += (f"\nFiles this rerun created in the sandbox's runs/ tree ({len(made)}; from {BASE / 'untouched.json'}, "
                 f"checked {un.get('checked_at')}):\n" + "".join(f"  {p}\n" for p in made)
                 + f"\nFiles this rerun wrote in its own folder ({len(mine)}; plus {RF / 'DONE'} when it finishes):\n"
                 + "".join(f"  {p}\n" for p in mine)
                 + f"\nFiles built before this rerun by {prov.get('script', DATA_SCRIPT)} (data seeds 1-4 and the "
                   f"report of each quality; {len(prov.get('files') or [])} files; recorded "
                   f"{prov.get('recorded_at')}; logs {', '.join(prov.get('logs') or [])}):\n"
                 + "".join(f"  {r['path']}  sha256 {r['sha256']}  {r['mtime']}\n" for r in prov.get("files") or [])
                 + f"\nList written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}.\n")
    for root in (V5, EX):
        p = root / "EXPLORATORY_resultF_rerun.txt"
        if final or not p.exists():
            write_atomic(p, text)
        print(f"label: {p}" + (" (with the file list)" if final else ""))
    return 0


# ------------------------------------------------------------------------------------------------ quarantine

def quarantine(job):
    """Before f1train / esttrain (re)starts: result F's scripts skip an arm whose record.json merely exists, so an arm
    interrupted while its record was being written would be skipped forever. Move such an arm folder (record.json that
    does not parse, or no actor.msgpack, or an actor whose SHA-256 differs from the record's) out of the sandbox into
    <resultF>/state/quarantine/<time>/, so the script trains it again. Only folders this run created are moved (never
    a path in the sandbox baseline); a parsed, consistent record is left alone even if its actor is non-finite (that is
    a result, and the done check reports it)."""
    kind, q = job.split("_", 1)
    root, arms = {"f1train": (F1, F1_ARMS), "esttrain": (ESTD, EST_ARMS)}[kind]
    base_file = BASE / "baseline_sandbox.json"
    base = json.loads(base_file.read_text())["files"] if base_file.is_file() else {}
    moved = []
    for arm in arms:
        d = root / q / arm / f"seed{SEED}"
        rec = d / "record.json"
        if not rec.exists():
            continue
        try:
            body = json.loads(rec.read_text())
            ok = (d / "actor.msgpack").is_file() and body.get("actor_sha256") == sha(d / "actor.msgpack")
        except (ValueError, OSError, AttributeError):
            ok = False
        if ok:
            continue
        if any(k.startswith(str(d) + "/") for k in base):
            print(f"quarantine {job}: {d} is incomplete but existed before this run; not moved", file=sys.stderr)
            return 5
        dest = STATE / "quarantine" / time.strftime("%Y%m%dT%H%M%S") / d.relative_to(IW)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(d, dest)
        moved.append(f"{d} -> {dest}")
    print(f"quarantine {job}: " + ("; ".join(moved) if moved else "nothing incomplete"))
    return 0


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "label":
        return label(final="--final" in argv[1:])
    if cmd == "done":
        try:
            ok = done(argv[1])
        except SystemExit:
            raise
        except Exception as error:  # noqa: BLE001  (a record that does not parse or lacks a key: not finished)
            print(f"done {argv[1]}: not finished ({type(error).__name__}: {error})", file=sys.stderr)
            ok = False
        return 0 if ok else 1
    if cmd == "quarantine":
        return quarantine(argv[1])
    if cmd == "d1":
        return d1(argv[1], dry="--dry" in argv[2:])
    if cmd == "baseline":
        return baseline()
    if cmd == "provenance":
        return provenance()
    if cmd == "untouched":
        return untouched()
    print("usage: rf.py done JOB | d1 Q | baseline | provenance | untouched | label [--final] | "
          "quarantine f1train_Q|esttrain_Q", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
