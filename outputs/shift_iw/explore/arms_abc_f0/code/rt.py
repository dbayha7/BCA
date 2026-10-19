"""EXPLORATORY: arms A, B, C and F0 -- the runner's small helper. Standard library only (no JAX, no numpy), run with
python -I -B, so run_arms.sh can call it often and fast.

It only reads, except: routes2/arms/state/*.json (each written whole: a temporary file, then os.replace), the
EXPLORATORY_arms_abc_f0.txt label in each of the sandbox's two runs/ roots, and (quarantine only) moving an interrupted
A2/B2/C2 folder of seeds 1-4 that this run created under channel/train_f1/ into routes2/arms/state/quarantine/. It never
writes into the repository, never touches a sandbox file that existed before this run's first start, never touches
result F's rerun folder, and never writes an approval, spec, pass-mark or declined file.

  rt.py jobs start|continue    the job list (one line per job: name TAB comma-separated dependencies)
  rt.py done JOB               exit 0 if the job is settled: its output is finished (or stopped by a kill rule) and
                               current, or an input failed for good so it can never run (a TD3+BC host that went
                               non-finite: its room check, band and A2 are settled; an invalid band: A2 is settled).
                               The screen, the kill checks and the scoring are done only when complete and built on
                               the current versions of every file they read (SHA-256 of each, recorded in them)
  rt.py d1 Q                   the new data's D1 numbers (seeds 0-4) against the committed report, bit for bit, and the
                               report's D1 status; writes state/d1_Q.json; exit 0 only if identical and D1 passed
  rt.py baseline               once: SHA-256 of every sandbox file and of the repository files read in place
  rt.py untouched              no baseline file changed or vanished; no new sandbox file outside repo/runs/
  rt.py label                  the EXPLORATORY label in both runs/ roots, listing every file this run created
                               (refreshed after every job; hashes cached by size and modification time)
  rt.py quarantine A2_Q_sK     move an interrupted A2/B2/C2 folder of seed K (1-4) created by this run aside
  rt.py screen-allows          exit 0 if SCREEN_seed0.json is complete, verified and allows seeds 1-4 for some arm
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
ARMS = CODE.parent
STATE = ARMS / "state"
SRC = Path(os.environ.get("ARMS_SRC", "/home/user/bca"))
IW = Path(os.environ.get("ARMS_IW", str(ARMS.parent.parent / "iw_rerun")))
REPO = IW / "repo"
V5 = REPO / "runs" / "shift_iw_v5"
EX = REPO / "runs" / "shift_iw_v5_explore"
OUT = EX / "arms_abc_f0"
F1 = EX / "channel" / "train_f1"
CHAN = SRC / "outputs" / "shift_iw" / "explore" / "channel"
PASSMARKS = ARMS / "PASSMARKS_arms_ABC_and_F_control.md"
PASSMARKS_SHA256 = "a073be97003938c112e9ba77581a311e812fe5851f7e64211a11fb4bf47d9ecf"
LABEL = ("EXPLORATORY: arms A/B/C and result F's no-band control; seen seeds 0-4; pass marks frozen in "
         "routes2/arms/PASSMARKS_arms_ABC_and_F_control.md (SHA-256 {sha}) before this code was written")
QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
NEW_DATA = ("random", "medium-replay", "medium")
F0_Q = ("medium-expert", "expert")
SEEDS = (0, 1, 2, 3, 4)
LATER = (1, 2, 3, 4)
F1_ARMS = ("A2_cmd_strong", "B2_cmd_train", "C2_untargeted")
LAWS = ("none", "mild", "strong")
SRC_READ = ("outputs/shift_iw/explore/channel", "outputs/shift_iw/v5", "experiments/iw", "experiments/shift_iw")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parsed(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None


def rec_ok(path):
    path = Path(path)
    side = Path(str(path) + ".sha256")
    if not path.is_file() or not side.is_file():
        return False
    words = side.read_text().split()
    return bool(words) and words[0] == sha(path)


def spec_sha():
    return (REPO / "experiments" / "iw" / "spec.json.sha256").read_text().split()[0]


def finished(path):
    """The package's run.finished: record checksum matches, parses, status finished, this specification."""
    if not rec_ok(path):
        return None
    rec = parsed(path)
    return rec if rec is not None and rec.get("status") == "finished" and rec.get("spec_sha256") == spec_sha() else None


def ours(path):
    """One of this run's arm records: checksum, label, this frozen file, status finished or stopped."""
    if not rec_ok(path):
        return None
    rec = parsed(path)
    if rec is None or rec.get("label") != LABEL.format(sha=PASSMARKS_SHA256):
        return None
    if rec.get("passmarks_sha256") != PASSMARKS_SHA256 or rec.get("status") not in ("finished", "stopped"):
        return None
    return rec


def passmarks_ok():
    side = Path(str(PASSMARKS) + ".sha256")
    return (PASSMARKS.is_file() and side.is_file() and sha(PASSMARKS) == PASSMARKS_SHA256
            and side.read_text().split()[0] == PASSMARKS_SHA256)


def mtime(path):
    return Path(path).stat().st_mtime


def arm_path(arm, q, k):
    if arm == "F0":
        return OUT / "F0" / q / "F0_no_band" / f"seed{k}" / "record.json"
    return OUT / arm / q / f"seed{k}" / "record.json"


# ------------------------------------------------------------------------------------------------ done checks

def data_done(q, k):
    rec = finished(V5 / q / "data" / f"seed{k}" / "stage.json")
    return rec is not None and rec.get("seed") == k and rec.get("quality") == q and isinstance(rec.get("D1"), dict)


def report_done(q):
    p = V5 / q / "report" / "report.json"
    rep = finished(p)
    if rep is None or rep.get("quality") != q or sorted(rep.get("seeds") or []) != list(SEEDS):
        return False
    if not all(data_done(q, k) for k in SEEDS):
        return False
    data = [V5 / q / "data" / f"seed{k}" / "stage.json" for k in SEEDS]
    d0 = parsed(data[0])
    return rep.get("code_sha256") == d0.get("code_sha256") and all(mtime(p) >= mtime(d) for d in data)


def f1_trained(q, k):
    want = (CHAN / "PASSMARKS_fallback_rerun.md.sha256").read_text().split()[0]
    for arm in F1_ARMS:
        d = F1 / q / arm / f"seed{k}"
        rec = parsed(d / "record.json")
        if rec is None or not (d / "actor.msgpack").is_file():
            return False
        if not (rec.get("exploratory") is True and rec.get("finite") in (True, False) and rec.get("arm") == arm
                and rec.get("passmarks_sha256") == want and rec.get("actor_sha256") == sha(d / "actor.msgpack")
                and rec.get("seed") == k and rec.get("quality") == q):
            return False
    return True


def f1_evaluated(q, k):
    p = F1 / q / f"evaluate_seed{k}.json"
    body = parsed(p)
    if body is None or body.get("quality") != q or body.get("seed") != k:
        return False
    ev = body.get("evaluations") or {}
    for arm in F1_ARMS:
        r = F1 / q / arm / f"seed{k}" / "record.json"
        if arm not in ev or not r.is_file() or mtime(p) < mtime(r):
            return False
        if not all(isinstance((ev[arm].get(law) or {}).get("J"), (int, float)) for law in LAWS):
            return False
    return True


def host_failed(q, k):
    """The gapped host of (q, k) finished but failed (non-finite): final, the package never retrains it."""
    rec = finished(V5 / q / "host" / "gap" / f"seed{k}" / "record.json")
    return rec is not None and rec.get("seed") == k and bool((rec.get("result") or {}).get("failed"))


def band_invalid(q, k):
    rec = finished(V5 / q / "band" / "gap" / f"seed{k}" / "record.json")
    return rec is not None and rec.get("seed") == k and not rec.get("band_valid")


def fresh(rec):
    """Every file the summary read still has the SHA-256 it had then."""
    for p, h in (rec.get("inputs_sha256") or {}).items():
        try:
            if sha(p) != h:
                return False
        except OSError:
            return False
    return True


def split_job(job):
    parts = job.split("_")
    kind = parts[0]
    q = parts[1] if len(parts) > 1 else None
    k = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("s") else None
    if kind == "eval":   # eval_<arm>_<Q>_s<K>
        return kind, dict(arm=parts[1], q=parts[2], k=int(parts[3][1:]))
    return kind, dict(q=q, k=k)


def done(job):
    kind, a = split_job(job)
    q, k = a.get("q"), a.get("k")
    if kind == "data":
        return data_done(q, k)
    if kind == "report":
        return report_done(q)
    if kind == "d1":
        body = parsed(STATE / f"d1_{q}.json")
        data = [V5 / q / "data" / f"seed{s}" / "stage.json" for s in SEEDS]
        rep = V5 / q / "report" / "report.json"
        return (body is not None and body.get("allows_band") is True and rep.is_file()
                and all(p.is_file() and mtime(STATE / f"d1_{q}.json") >= mtime(p) for p in data + [rep]))
    if kind == "room":
        rec = finished(V5 / q / "room" / f"seed{k}" / "record.json")
        return (rec is not None and rec.get("seed") == k) or host_failed(q, k)
    if kind == "band":
        rec = finished(V5 / q / "band" / "gap" / f"seed{k}" / "record.json")
        return (rec is not None and rec.get("seed") == k) or host_failed(q, k)
    if kind == "host":            # a finished host that failed is final: its dependents record the stop
        rec = finished(V5 / q / "host" / "gap" / f"seed{k}" / "record.json")
        return rec is not None and rec.get("seed") == k
    if kind in ("A", "B", "C", "F0"):
        rec = ours(arm_path(kind, q, k))
        return rec is not None and rec.get("quality") == q and rec.get("seed") == k
    if kind == "A2":              # result F's World refuses a failed host or an invalid band on every retry
        return f1_trained(q, k) or host_failed(q, k) or band_invalid(q, k)
    if kind == "A2eval":
        return f1_evaluated(q, k) or host_failed(q, k) or band_invalid(q, k)
    # a summary is done only while the summary it was built on is done too, so a resumed run that redoes the screen
    # or the kill checks also redoes everything built on them (load_jobs marks jobs done once, at load)
    if kind in ("screen", "kills"):
        rec = ours_doc(ARMS / ("SCREEN_seed0.json" if kind == "screen" else "KILLS.json"))
        return (rec is not None and rec.get("complete") is True and fresh(rec)
                and (kind == "screen" or done("screen")))
    if kind == "eval":
        rec = ours_doc(OUT / "eval" / a["arm"] / a["q"] / f"seed{a['k']}" / "record.json")
        kills = ARMS / "KILLS.json"
        return rec is not None and kills.is_file() and rec.get("kills_sha256") == sha(kills) and done("kills")
    if kind == "marks":
        rec = ours_doc(ARMS / "results.json")
        kills = ARMS / "KILLS.json"
        return (rec is not None and kills.is_file() and rec.get("kills_sha256") == sha(kills) and fresh(rec)
                and done("kills"))
    raise SystemExit(f"unknown job {job!r}")


def ours_doc(path):
    if not rec_ok(path):
        return None
    rec = parsed(path)
    if rec is None or rec.get("label") != LABEL.format(sha=PASSMARKS_SHA256) or \
            rec.get("passmarks_sha256") != PASSMARKS_SHA256:
        return None
    return rec


# ------------------------------------------------------------------------------------------------ job lists

def jobs_start():
    """The seed-0 screen (section 8, steps 2-4)."""
    out = []
    for q in NEW_DATA:                                   # the room checks are the longest: first
        out.append((f"room_{q}_s0", []))
    for q in NEW_DATA:
        for k in LATER:
            out.append((f"data_{q}_s{k}", []))
    for q in NEW_DATA:
        out.append((f"report_{q}", [f"data_{q}_s{k}" for k in LATER]))
        out.append((f"d1_{q}", [f"report_{q}"]))
        out.append((f"band_{q}_s0", [f"room_{q}_s0", f"report_{q}", f"d1_{q}"]))
    for q in QUALITIES:
        out.append((f"A_{q}_s0", [f"band_{q}_s0"] if q in NEW_DATA else []))
    for q in QUALITIES:
        out.append((f"C_{q}_s0", []))
    for q in F0_Q:
        out.append((f"F0_{q}_s0", []))
    for q in QUALITIES:
        out.append((f"B_{q}_s0", []))
    out.append(("screen", [j for j, _ in out]))
    return out


def screen_survivors():
    rec = ours_doc(ARMS / "SCREEN_seed0.json")
    if rec is None:
        raise SystemExit("refused: no verified SCREEN_seed0.json (run the screen first: run_arms.sh start)")
    if rec.get("complete") is not True:
        raise SystemExit("refused: the screen is not complete; seeds 1-4 wait until every kill check of the screen is "
                         f"computed ({rec.get('missing')})")
    return rec["survivors"]


def jobs_continue():
    """Seeds 1-4 for every arm the screen did not stop, then the kill checks across seeds, then scoring (section 8,
    steps 5-6). Arm B's maps of seeds 1-4 are built for all five qualities when B survives (K1 and K2 count over every
    seed and quality) or when C survives and B's certificate was not voided at the screen (C's guard is void if B's K1
    or K2 fires; kills.py b_map_qualities is the same rule). Scoring starts only when every evaluation is done."""
    sv = screen_survivors()
    screen = ours_doc(ARMS / "SCREEN_seed0.json")
    A, Bq, Cq, Fq = list(sv["A"]), list(sv["B"]), list(sv["C"]), list(sv["F0"])
    b_maps = list(QUALITIES) if (Bq or (Cq and not screen["C_rules"]["guard_void"]["by_B"])) else []
    hosts = list(QUALITIES) if (A or Bq or Cq) else [q for q in F0_Q if q in Fq]
    bands = [q for q in QUALITIES if q in A or q in Fq]
    out = []
    for q in hosts:
        for k in LATER:
            out.append((f"host_{q}_s{k}", []))
    for q in bands:
        for k in LATER:
            out.append((f"room_{q}_s{k}", [f"host_{q}_s{k}"]))
    for q in bands:
        for k in LATER:
            out.append((f"band_{q}_s{k}", [f"room_{q}_s{k}"]))
    for q in A:
        for k in LATER:
            out.append((f"A_{q}_s{k}", [f"band_{q}_s{k}"]))
    for q in Cq:
        for k in LATER:
            out.append((f"C_{q}_s{k}", [f"host_{q}_s{k}"]))
    for q in b_maps:
        for k in LATER:
            out.append((f"B_{q}_s{k}", [f"host_{q}_s{k}"]))
    for q in Fq:
        for k in LATER:
            out.append((f"A2_{q}_s{k}", [f"band_{q}_s{k}"]))
            out.append((f"A2eval_{q}_s{k}", [f"A2_{q}_s{k}"]))
            out.append((f"F0_{q}_s{k}", [f"A2_{q}_s{k}"]))
    trained = [j for j, _ in out]
    out.append(("kills", trained))
    evals = []
    for arm, qs in (("A", A), ("B", Bq), ("C", Cq), ("F", Fq)):
        for q in qs:
            for k in SEEDS:
                evals.append((f"eval_{arm}_{q}_s{k}", ["kills"]))
    out += evals
    out.append(("marks", ["kills"] + [j for j, _ in evals]))
    return out


# ------------------------------------------------------------------------------------------------ D1

def same_numbers(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) & set(b)
        return bool(keys) and all(same_numbers(a[k], b[k]) for k in keys)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return a == b
    return True


def d1(q):
    rep_path = SRC / "outputs" / "shift_iw" / "v5" / q / "report.json"
    rep = json.loads(rep_path.read_text())
    desk = rep["marks"]["D1"]["per_seed"]
    sandbox_rep = finished(V5 / q / "report" / "report.json")
    status = (((sandbox_rep or {}).get("marks") or {}).get("D1") or {}).get("status")
    out = dict(kind="arms-d1-check", label=LABEL.format(sha=PASSMARKS_SHA256), passmarks_sha256=PASSMARKS_SHA256,
               quality=q, committed_report=str(rep_path), committed_report_sha256=sha(rep_path),
               committed_D1_status=rep["marks"]["D1"]["status"], sandbox_report_D1_status=status, seeds={})
    for s in SEEDS:
        rec = finished(V5 / q / "data" / f"seed{s}" / "stage.json")
        if rec is None:
            print(f"d1 {q}: the data record of seed {s} is missing", file=sys.stderr)
            return 2
        out["seeds"][str(s)] = dict(identical=same_numbers(rec["D1"], desk[s]), sandbox=rec["D1"], desktop=desk[s])
    out["all_identical"] = all(v["identical"] for v in out["seeds"].values())
    out["allows_band"] = bool(out["all_identical"] and status == "pass")
    out["rule"] = ("section 8 step 2: a report that does not say D1 passed, or any D1 difference, stops band fits on "
                   "this quality until the owner decides")
    write_atomic(STATE / f"d1_{q}.json", json.dumps(out, indent=1))
    print(f"d1 {q}: seeds 0-4 identical to the committed report: {out['all_identical']}; the sandbox report says D1 "
          f"{status}; band fits allowed: {out['allows_band']}")
    return 0 if out["allows_band"] else 5


# ------------------------------------------------------------------------------------------------ untouched, label

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
    return {str(p): ("symlink:" + os.readlink(p)) if p.is_symlink() else sha(p) for p in files}


def baseline():
    marker = STATE / "start.marker"
    if not marker.exists():
        write_atomic(marker, time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    for name, files in (("baseline_sandbox.json", sandbox_files), ("baseline_repository.json", src_files)):
        p = STATE / name
        if not p.exists():
            write_atomic(p, json.dumps(dict(made_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), files=manifest(files()))))
    print(f"baseline: {STATE / 'baseline_sandbox.json'}, {STATE / 'baseline_repository.json'}")
    return 0


def untouched():
    out = dict(kind="arms-untouched-check", checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    ok = True
    for name, files, lab in (("baseline_sandbox.json", sandbox_files, "sandbox"),
                             ("baseline_repository.json", src_files, "repository")):
        base = json.loads((STATE / name).read_text())["files"]
        now = manifest(files())
        changed = sorted(k for k in base if k in now and now[k] != base[k])
        gone = sorted(k for k in base if k not in now)
        new = sorted(k for k in now if k not in base)
        if lab == "sandbox":
            runs = str(REPO / "runs") + "/"
            outside = [k for k in new if not k.startswith(runs)]
            out[lab] = dict(changed=changed, removed=gone, new_outside_runs=outside,
                            new_under_runs={k: now[k] for k in new if k.startswith(runs)})
            bad = changed + gone + outside
        else:
            out[lab] = dict(changed=changed, removed=gone, new=new)
            bad = changed + gone + new
        ok &= not bad
    out["all_untouched"] = bool(ok)
    write_atomic(STATE / "untouched.json", json.dumps(out, indent=1))
    print(f"untouched: {ok} (sandbox: changed {len(out['sandbox']['changed'])}, removed {len(out['sandbox']['removed'])}"
          f", new outside runs/ {len(out['sandbox']['new_outside_runs'])}, new under runs/ "
          f"{len(out['sandbox']['new_under_runs'])}; repository: changed {len(out['repository']['changed'])}, removed "
          f"{len(out['repository']['removed'])}, new {len(out['repository']['new'])})")
    return 0 if ok else 6


LABEL_FILE = "EXPLORATORY_arms_abc_f0.txt"


def created_under_runs():
    """Every file under the sandbox's runs/ tree that is not in the baseline taken at this run's first start (the
    label files themselves excluded), with SHA-256; hashes are cached by size and modification time, so a refresh after
    each job rehashes only new or changed files."""
    base = set(json.loads((STATE / "baseline_sandbox.json").read_text())["files"])
    cache_path = STATE / "label_cache.json"
    cache = parsed(cache_path) or {}
    out, new_cache = {}, {}
    for p in sorted((REPO / "runs").rglob("*")):
        if "__pycache__" in p.parts or p.name == LABEL_FILE or p.name.startswith(f".{LABEL_FILE}.tmp"):
            continue
        try:
            if p.is_dir() or str(p) in base:
                continue
            st = p.lstat()
        except OSError:
            continue
        key = f"{st.st_size}:{st.st_mtime_ns}"
        hit = cache.get(str(p))
        try:
            h = hit[1] if hit and hit[0] == key else (("symlink:" + os.readlink(p)) if p.is_symlink() else sha(p))
        except OSError:
            continue                                      # vanished while listing (a temporary file)
        out[str(p)] = h
        new_cache[str(p)] = [key, h]
    write_atomic(cache_path, json.dumps(new_cache))
    return out


def label():
    """EXPLORATORY_arms_abc_f0.txt in both runs/ roots: the label and every file this run created there, with
    SHA-256. run_arms.sh refreshes it after every job and when the runner stops or exits."""
    if not (STATE / "baseline_sandbox.json").is_file():
        print("label: no baseline yet (nothing created by this run)")
        return 0
    made = created_under_runs()
    start = (STATE / "start.marker").read_text().strip() if (STATE / "start.marker").is_file() else "?"
    text = (LABEL.format(sha=PASSMARKS_SHA256) + ".\n\n"
            f"Every file listed below was created in this sandbox's runs/ tree on or after {start} by "
            f"{ARMS / 'run_arms.sh'} (code: {CODE}). The package's own stage records (data, report, host, room, band) "
            "cannot carry the label themselves; they are scratch inputs of this exploratory run, not version-5 results. "
            "Result F's arm A2 of seeds 1-4 was trained by its unchanged script (channel/command_le_train_f1.py).\n\n"
            f"Files ({len(made)}):\n"
            + "".join(f"  {p}  sha256 {h}\n" for p, h in sorted(made.items()))
            + f"\nList written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}.\n")
    for root in (V5, EX):
        write_atomic(root / LABEL_FILE, text)
    print(f"label: {V5 / LABEL_FILE} and {EX / LABEL_FILE} ({len(made)} files)")
    return 0


def quarantine(job):
    kind, a = split_job(job)
    q, k = a["q"], a["k"]
    if kind != "A2" or k not in LATER:
        raise SystemExit(f"quarantine handles A2 jobs of seeds 1-4 only, not {job!r}")
    base_file = STATE / "baseline_sandbox.json"
    base = json.loads(base_file.read_text())["files"] if base_file.is_file() else None
    if base is None:
        print(f"quarantine {job}: no baseline; nothing moved", file=sys.stderr)
        return 5
    moved = []
    for arm in F1_ARMS:
        d = F1 / q / arm / f"seed{k}"
        rec = d / "record.json"
        if not rec.exists():
            continue
        try:
            body = json.loads(rec.read_text())
            good = (d / "actor.msgpack").is_file() and body.get("actor_sha256") == sha(d / "actor.msgpack")
        except (ValueError, OSError, AttributeError):
            good = False
        if good:
            continue
        if any(x.startswith(str(d) + "/") for x in base):
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
    if cmd == "jobs":
        if not passmarks_ok():
            print("refused: the frozen pass marks differ from their sidecar or from the frozen digest", file=sys.stderr)
            return 3
        try:
            rows = jobs_start() if argv[1] == "start" else jobs_continue()
        except SystemExit as error:
            print(str(error), file=sys.stderr)
            return 4
        for j, deps in rows:
            print(f"{j}\t{','.join(deps)}")
        return 0
    if cmd == "done":
        try:
            return 0 if done(argv[1]) else 1
        except SystemExit:
            raise
        except Exception as error:  # noqa: BLE001
            print(f"done {argv[1]}: not finished ({type(error).__name__}: {error})", file=sys.stderr)
            return 1
    if cmd == "screen-allows":
        try:
            sv = screen_survivors()
        except SystemExit as error:
            print(str(error), file=sys.stderr)
            return 4
        print(json.dumps(sv))
        return 0
    if cmd == "d1":
        return d1(argv[1])
    if cmd == "baseline":
        return baseline()
    if cmd == "untouched":
        return untouched()
    if cmd == "label":
        return label()
    if cmd == "quarantine":
        return quarantine(argv[1])
    if cmd == "passmarks":
        ok = passmarks_ok()
        print(f"frozen pass marks: {'unchanged' if ok else 'CHANGED OR MISSING'} ({PASSMARKS})")
        return 0 if ok else 3
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
