"""Independent check of the kill rules and the run's integrity for arms A, B, C and F0 (EXPLORATORY run).

Reads only: the frozen pass marks, the raw arm records, A2's records and evaluations, the package's stage records,
the runner's logs and state, KILLS.json/SCREEN_seed0.json/results.json (only to compare against), the sandbox and the
repository files listed in the run's baselines. Writes only verify/kills_check_raw.json next to this script.
Standard library only. Run with: PYTHONDONTWRITEBYTECODE=1 python3 -I -B kills_check.py
"""
import hashlib
import json
import os
import re
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SP = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad")
ARMS = SP / "routes2" / "arms"
LOGS, STATE = ARMS / "logs", ARMS / "state"
IW = SP / "iw_rerun"
RUNS = IW / "repo" / "runs"
V5 = RUNS / "shift_iw_v5"
EX = RUNS / "shift_iw_v5_explore"
OUT = EX / "arms_abc_f0"
F1 = EX / "channel" / "train_f1"
SRC = Path("/home/user/bca")
PASSMARKS_SHA = "a073be97003938c112e9ba77581a311e812fe5851f7e64211a11fb4bf47d9ecf"
LABEL = ("EXPLORATORY: arms A/B/C and result F's no-band control; seen seeds 0-4; pass marks frozen in "
         f"routes2/arms/PASSMARKS_arms_ABC_and_F_control.md (SHA-256 {PASSMARKS_SHA}) before this code was written")
Q5 = ("random", "medium-replay", "medium", "medium-expert", "expert")
K0Q = ("medium-replay", "medium", "medium-expert")
WEAK = ("random", "medium-replay", "medium")
FQ = ("medium-expert", "expert")
SEEDS = (0, 1, 2, 3, 4)


def num(v):
    if isinstance(v, str):
        return float(v)
    return v


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


EDGES = [f32(v) for v in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)]
T085, T060, KNEE = f32(0.85), f32(0.60), f32(0.8)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def sidecar_ok(p):
    p = Path(p)
    side = Path(str(p) + ".sha256")
    if not side.is_file():
        return None
    words = side.read_text().split()
    return bool(words) and words[0] == sha(p) and (len(words) < 2 or words[1].lstrip("*") == p.name)


def iso(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def fmt(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


R = dict(made_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()), problems=[], notes=[])


def problem(msg):
    R["problems"].append(msg)


def note(msg):
    R["notes"].append(msg)


def arm_rec(arm, q, k):
    p = OUT / "F0" / q / "F0_no_band" / f"seed{k}" / "record.json" if arm == "F0" else OUT / arm / q / f"seed{k}" / "record.json"
    if not p.is_file():
        return p, None
    return p, load(p)


# ================================================================================================ integrity
I = R.setdefault("integrity", {})
pm = ARMS / "PASSMARKS_arms_ABC_and_F_control.md"
I["passmarks_sha256"] = sha(pm)
I["passmarks_ok"] = I["passmarks_sha256"] == PASSMARKS_SHA and (Path(str(pm) + ".sha256").read_text().split()[0] == PASSMARKS_SHA)
if not I["passmarks_ok"]:
    problem("frozen pass marks differ from the frozen digest")

# every sidecar under the arms' output tree, train_f1 and the package's new stage records, and the docs
side_bad, side_n = [], 0
for root in (OUT, F1, V5):
    for side in root.rglob("*.sha256"):
        target = Path(str(side)[: -len(".sha256")])
        if not target.is_file():
            continue
        side_n += 1
        if not sidecar_ok(target):
            side_bad.append(str(target))
for name in ("KILLS.json", "results.json", "SCREEN_seed0.json"):
    side_n += 1
    if not sidecar_ok(ARMS / name):
        side_bad.append(str(ARMS / name))
I["sidecars_checked"], I["sidecars_bad"] = side_n, side_bad
if side_bad:
    problem(f"sidecar mismatch: {side_bad}")

# labels, digests and actor hashes of every arm record
lab_bad, actor_bad, n_arm = [], [], 0
for arm, qs in (("A", Q5), ("B", Q5), ("C", Q5), ("F0", FQ)):
    for q in qs:
        for k in SEEDS:
            p, r = arm_rec(arm, q, k)
            if r is None:
                continue
            n_arm += 1
            if r.get("label") != LABEL or r.get("passmarks_sha256") != PASSMARKS_SHA or r.get("quality") != q or r.get("seed") != k:
                lab_bad.append(str(p))
            d = p.parent
            for key, fn in (("actors_8000_sha256", "actors_8000.msgpack"), ("actors_final_sha256", "actors_final.msgpack"),
                            ("actor_sha256", "actor.msgpack")):
                if key in r and (d / fn).is_file() and sha(d / fn) != r[key]:
                    actor_bad.append(str(d / fn))
                if key in r and not (d / fn).is_file():
                    actor_bad.append(f"missing {d / fn}")
for q in FQ:
    for k in SEEDS:
        for arm in ("A2_cmd_strong", "B2_cmd_train", "C2_untargeted"):
            d = F1 / q / arm / f"seed{k}"
            if (d / "record.json").is_file():
                r = load(d / "record.json")
                if sha(d / "actor.msgpack") != r.get("actor_sha256"):
                    actor_bad.append(str(d / "actor.msgpack"))
I["arm_records"], I["label_bad"], I["actor_hash_bad"] = n_arm, lab_bad, actor_bad
if lab_bad:
    problem(f"label/digest mismatch: {lab_bad}")
if actor_bad:
    problem(f"actor hash mismatch: {actor_bad}")

# recorded input hashes still current (nothing they read was overwritten later)
for name in ("SCREEN_seed0.json", "KILLS.json", "results.json"):
    doc = load(ARMS / name)
    stale = [p for p, h in (doc.get("inputs_sha256") or {}).items() if not Path(p).is_file() or sha(p) != h]
    I[f"{name}_inputs"] = dict(n=len(doc.get("inputs_sha256") or {}), stale=stale)
    if stale:
        problem(f"{name}: inputs changed after it was built: {stale}")
kills_sha = sha(ARMS / "KILLS.json")
res = load(ARMS / "results.json")
I["results_kills_sha256_current"] = res.get("kills_sha256") == kills_sha
if not I["results_kills_sha256_current"]:
    problem("results.json was built on another KILLS.json")

# eval records
ev_bad, ev_list = [], []
for arm_dir in sorted((OUT / "eval").glob("*")):
    for rp in sorted(arm_dir.rglob("record.json")):
        r = load(rp)
        arm = arm_dir.name
        q, k = rp.parent.parent.name, int(rp.parent.name[4:])
        src_arm = "F0" if arm == "F" else arm
        ap, _ = arm_rec(src_arm, q, k)
        ok = (r.get("kills_sha256") == kills_sha and sidecar_ok(rp) and r.get("label") == LABEL
              and (arm == "F" or r.get("arm_record_sha256") == sha(ap)))
        ev_list.append(dict(path=str(rp), arm=arm, q=q, k=k, started_at=r.get("started_at"), finished_at=r.get("finished_at"),
                            arm_record_sha256_ok=(r.get("arm_record_sha256") == sha(ap)) if r.get("arm_record_sha256") else None))
        if not ok:
            ev_bad.append(str(rp))
I["eval_records"] = len(ev_list)
I["eval_arms"] = sorted({e["arm"] for e in ev_list})
I["eval_bad"] = ev_bad
if ev_bad:
    problem(f"eval records not tied to the current KILLS.json or arm record: {ev_bad}")
if any(e["arm"] == "B" for e in ev_list):
    problem("an eval record exists for arm B")

# untouched: recompute against the baselines now
def manifest_now(paths):
    out = {}
    for p in paths:
        pp = Path(p)
        if pp.is_symlink():
            out[p] = "symlink:" + os.readlink(pp)
        elif pp.is_file():
            out[p] = sha(pp)
    return out


bs = load(STATE / "baseline_sandbox.json")["files"]
br = load(STATE / "baseline_repository.json")["files"]
now_s, now_r = manifest_now(bs), manifest_now(br)
s_changed = sorted(p for p in bs if p in now_s and now_s[p] != bs[p])
s_gone = sorted(p for p in bs if p not in now_s)
r_changed = sorted(p for p in br if p in now_r and now_r[p] != br[p])
r_gone = sorted(p for p in br if p not in now_r)
new_outside = []
for p in sorted(IW.rglob("*")):
    if "__pycache__" in p.parts or p.is_dir():
        continue
    if str(p) not in bs and not str(p).startswith(str(RUNS) + "/"):
        new_outside.append(str(p))
pyc = [str(p) for p in IW.rglob("__pycache__")]
# repository: files under the read roots now vs baseline (new files), and anything under /home/user/bca newer than
# the run's start marker (outside .git)
SRC_READ = ("outputs/shift_iw/explore/channel", "outputs/shift_iw/v5", "experiments/iw", "experiments/shift_iw")
r_new = []
for rel in SRC_READ:
    p = SRC / rel
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
    r_new += [str(f) for f in files if str(f) not in br]
start_marker = STATE / "start.marker"
t_start = start_marker.stat().st_mtime
repo_newer = []
for dirpath, dirnames, filenames in os.walk(SRC):
    if ".git" in dirnames:
        dirnames.remove(".git")
    for fn in filenames:
        fp = Path(dirpath) / fn
        try:
            if fp.lstat().st_mtime > t_start:
                repo_newer.append(str(fp))
        except OSError:
            pass
u = load(STATE / "untouched.json")
I["untouched_recorded"] = dict(checked_at=u["checked_at"], all_untouched=u["all_untouched"],
                               sandbox={k: len(v) for k, v in u["sandbox"].items()},
                               repository={k: len(v) for k, v in u["repository"].items()})
I["untouched_now"] = dict(sandbox_baseline_files=len(bs), sandbox_changed=s_changed, sandbox_removed=s_gone,
                          sandbox_new_outside_runs=new_outside, sandbox_pycache_dirs=pyc,
                          repository_baseline_files=len(br), repository_changed=r_changed, repository_removed=r_gone,
                          repository_new_under_read_roots=r_new,
                          repository_files_modified_after_start_marker=repo_newer[:50],
                          repository_files_modified_after_start_marker_count=len(repo_newer),
                          start_marker=start_marker.read_text().strip())
if not u["all_untouched"]:
    problem("the run's own final untouched check failed")
if s_changed or s_gone or new_outside or r_changed or r_gone or r_new:
    problem("untouched recheck now finds differences")

# ================================================================================================ job windows
jt = []
for line in (STATE / "job_times.tsv").read_text().splitlines():
    j, cpu, t0, t1, rc = line.split("\t")
    jt.append(dict(job=j, cpu=int(cpu), t0=int(t0), t1=int(t1), rc=int(rc)))
JOB = {}
for e in jt:
    JOB.setdefault(e["job"], []).append(e)
R["jobs"] = dict(rows=len(jt), distinct=len(JOB), nonzero_rc=[e for e in jt if e["rc"] != 0],
                 more_than_one_row=[j for j, v in JOB.items() if len(v) > 1])
if R["jobs"]["nonzero_rc"]:
    problem(f"jobs with a nonzero exit code: {R['jobs']['nonzero_rc']}")

# per-job logs: count 'start on cpu' lines (attempts)
attempts = {}
for lf in LOGS.glob("*.log"):
    n = sum(1 for line in lf.read_text(errors="replace").splitlines() if re.match(r"^\[\S+ \S+\] start on cpu", line))
    if n:
        attempts[lf.stem] = n
R["jobs"]["multi_attempt_logs"] = {j: n for j, n in attempts.items() if n > 1}


def window(job):
    rows = JOB.get(job)
    if not rows:
        return None
    return rows[-1]["t0"], rows[-1]["t1"]


# record mtimes inside their job's final window (nothing finished was rewritten later)
mt_bad, mt_n = [], 0


def check_in_window(job, paths):
    global mt_n
    w = window(job)
    if w is None:
        mt_bad.append(f"{job}: no job_times row")
        return
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        mt_n += 1
        m = p.stat().st_mtime
        if not (w[0] - 2 <= m <= w[1] + 2):
            mt_bad.append(f"{job}: {p} mtime {fmt(m)} outside [{fmt(w[0])}, {fmt(w[1])}]")


for arm, qs, seeds in (("A", Q5, SEEDS), ("B", Q5, (0,)), ("C", Q5, SEEDS), ("F0", FQ, SEEDS)):
    for q in qs:
        for k in seeds:
            p, r = arm_rec(arm, q, k)
            if r is None:
                problem(f"missing record {p}")
                continue
            check_in_window(f"{arm}_{q}_s{k}", [x for x in p.parent.iterdir()])
for q in Q5:
    for k in (1, 2, 3, 4):
        check_in_window(f"host_{q}_s{k}", list((V5 / q / "host" / "gap" / f"seed{k}").glob("*")))
        check_in_window(f"room_{q}_s{k}", list((V5 / q / "room" / f"seed{k}").glob("*")))
        check_in_window(f"band_{q}_s{k}", list((V5 / q / "band" / "gap" / f"seed{k}").glob("*")))
for q in WEAK:
    check_in_window(f"room_{q}_s0", list((V5 / q / "room" / "seed0").glob("*")))
    check_in_window(f"band_{q}_s0", list((V5 / q / "band" / "gap" / "seed0").glob("*")))
    for k in (1, 2, 3, 4):
        check_in_window(f"data_{q}_s{k}", list((V5 / q / "data" / f"seed{k}").glob("*")))
for q in FQ:
    for k in (1, 2, 3, 4):
        for a in ("A2_cmd_strong", "B2_cmd_train", "C2_untargeted"):
            check_in_window(f"A2_{q}_s{k}", list((F1 / q / a / f"seed{k}").glob("*")))
        check_in_window(f"A2eval_{q}_s{k}", [F1 / q / f"evaluate_seed{k}.json"])
for e in ev_list:
    p = Path(e["path"])
    check_in_window(f"eval_{e['arm']}_{e['q']}_s{e['k']}", list(p.parent.glob("*")))
check_in_window("kills", [ARMS / "KILLS.json", ARMS / "KILLS.json.sha256", ARMS / "KILLS.md"])
check_in_window("screen", [ARMS / "SCREEN_seed0.json", ARMS / "SCREEN_seed0.json.sha256", ARMS / "SCREEN_seed0.md"])
check_in_window("marks", [ARMS / "results.json", ARMS / "results.json.sha256"])
R["mtimes"] = dict(files_checked=mt_n, outside_window=mt_bad)
if mt_bad:
    note(f"files whose mtime lies outside their job's final window: {mt_bad}")

# ================================================================================================ restarts
RS = R.setdefault("restarts", {})
interrupted = {}
runner_logs = ["nohup_start.out", "nohup_continue.out", "nohup_continue2.out", "nohup_continue3.out"]
for lf in runner_logs:
    started, done = {}, set()
    for line in (LOGS / lf).read_text().splitlines():
        m = re.match(r"^\[(\S+ \S+)\] started: (\S+) on cpu", line)
        if m:
            started[m.group(2)] = m.group(1)
        m = re.match(r"^\[(\S+ \S+)\] done: (\S+)$", line)
        if m:
            done.add(m.group(2))
    open_jobs = {j: t for j, t in started.items() if j not in done}
    interrupted[lf] = open_jobs
RS["started_not_done_per_runner_log"] = interrupted
# for each interrupted job: its log's attempts, first attempt's last line, the rerun's start, file mtimes
det = {}
for lf, jobs in interrupted.items():
    for j, t in jobs.items():
        log = (LOGS / f"{j}.log").read_text().splitlines()
        starts = [i for i, line in enumerate(log) if re.match(r"^\[\S+ \S+\] start on cpu", line)]
        first = log[starts[0]:starts[1]] if len(starts) > 1 else log[starts[0]:]
        second = log[starts[1]:] if len(starts) > 1 else []
        w = window(j)
        kind = j.split("_")[0]
        if kind == "room":
            q, k = j.split("_")[1], int(j.split("_s")[-1])
            d = V5 / q / "room" / f"seed{k}"
        elif kind in ("A", "C", "B"):
            q, k = j.split("_")[1], int(j.split("_s")[-1])
            d = OUT / kind / q / f"seed{k}"
        else:
            d = None
        files = {}
        if d is not None and d.is_dir():
            for f in sorted(d.iterdir()):
                files[f.name] = fmt(f.stat().st_mtime)
        first_progress = [x for x in first[1:] if not x.startswith("[")]
        second_progress = [x for x in second[1:] if not x.startswith("[")]
        det[j] = dict(runner_log=lf, started=t, attempts=len(starts), first_attempt_lines=len(first) - 1,
                      first_attempt_last=first[-1] if len(first) > 1 else None,
                      rerun_first_progress=second_progress[0] if second_progress else None,
                      first_attempt_first_progress=first_progress[0] if first_progress else None,
                      rerun_finished_line=second[-1] if second else None,
                      final_window=[fmt(w[0]), fmt(w[1])] if w else None, files=files,
                      files_older_than_rerun=[n for n, m in files.items() if w and datetime.strptime(m, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() < w[0] - 2])
        if det[j]["files_older_than_rerun"]:
            problem(f"{j}: files older than the rerun's start remain: {det[j]['files_older_than_rerun']}")
        if len(starts) < 2:
            problem(f"{j}: interrupted but its log shows no rerun")
RS["interrupted_jobs"] = det
RS["quarantine_dir_exists"] = (STATE / "quarantine").exists()
RS["leftover_pid_or_failed_files"] = [p.name for p in STATE.glob("*.pid")] + [p.name for p in STATE.glob("*.failed")]

# ================================================================================================ kill rules
K = load(ARMS / "KILLS.json")
S = load(ARMS / "SCREEN_seed0.json")
KR = R.setdefault("kills", {})

# ---------------- arm A
A = {}
for q in Q5:
    per = {}
    for k in SEEDS:
        p, r = arm_rec("A", q, k)
        h = r.get("history") or []
        h0 = next((x for x in h if x.get("refresh") == 0), None)
        h1 = next((x for x in h if x.get("refresh") == 1), None)
        a1_0 = h0["arms"]["A1"] if h0 else {}
        a1_1 = h1["arms"]["A1"] if h1 else {}
        rec_share, rnd_share = a1_0.get("released_share_recorder_rows"), a1_0.get("released_share_random_rows")
        k0_applies = q in K0Q
        k0_pass = (rec_share is not None and rnd_share is not None and rec_share <= 0.10 and rnd_share >= 0.15) if k0_applies else None
        k1_share, k1_D, k1_U = a1_1.get("released_share"), a1_1.get("D_mean"), a1_1.get("U_mean")
        k1_fires = bool(k1_share > 0.60 or k1_D > 3.0 * k1_U) if a1_1 else None
        k2 = ((r.get("kills") or {}).get("K2_value") or {})
        fin = r.get("finite") or {}
        mark_arms = ("none", "A1", "margin0", "plain_1s", "plain_2s", "plain_4s", "matched_share", "untargeted")
        k3 = not all(fin.get(a) for a in mark_arms)
        # band validity of the band this seed read
        band = load(V5 / q / "band" / "gap" / f"seed{k}" / "record.json")
        per[k] = dict(status=r["status"], stopped_by=r.get("stopped_by"), step0=h0.get("step") if h0 else None,
                      step1=h1.get("step") if h1 else None, n=(r.get("inputs") or {}).get("n"),
                      k0_recorder=rec_share, k0_random=rnd_share, k0_applies=k0_applies,
                      k0_binding=k0_applies and k >= 1, k0_pass=k0_pass,
                      k0_guard_blocked_recorder=a1_0.get("guard_blocked_recorder_rows"),
                      k0_guard_blocked_random=a1_0.get("guard_blocked_random_rows"),
                      k0_recorded=(r.get("kills") or {}).get("K0"),
                      k1_share=k1_share, k1_D=k1_D, k1_U=k1_U, k1_ratio=(k1_D / k1_U) if a1_1 else None, k1_fires=k1_fires,
                      k1_recorded=((r.get("kills") or {}).get("K1") or {}).get("fires"),
                      k2_d=k2.get("d"), k2_finite=k2.get("finite"), k3_nonfinite=k3, band_valid=band.get("band_valid"),
                      band_record_sha_match=r.get("band_record_sha256") == sha(V5 / q / "band" / "gap" / f"seed{k}" / "record.json"),
                      s=(r.get("inputs") or {}).get("s"), finished_at=r.get("finished_at"),
                      final_return_keys=[x for x in r if x in ("evaluation", "evaluations", "returns", "points")])
        if per[k]["k0_recorded"] and k0_applies and per[k]["k0_recorded"].get("passes") != k0_pass:
            problem(f"A {q} seed {k}: K0 recorded {per[k]['k0_recorded'].get('passes')} vs recomputed {k0_pass}")
        if per[k]["k1_recorded"] is not None and per[k]["k1_recorded"] != k1_fires:
            problem(f"A {q} seed {k}: K1 recorded differs from recomputed")
        if per[k]["final_return_keys"]:
            problem(f"A {q} seed {k}: the training record holds a final return {per[k]['final_return_keys']}")
    d = [per[k]["k2_d"] for k in SEEDS]
    k2_bad = [k for k in SEEDS if per[k]["k2_d"] is not None and per[k]["k2_d"] <= -3.0]
    stop_k0 = [k for k in SEEDS if per[k]["k0_binding"] and per[k]["k0_pass"] is False]
    stop_k1 = [k for k in SEEDS if per[k]["k1_fires"]]
    stop_k3 = [k for k in SEEDS if per[k]["k3_nonfinite"] or per[k]["band_valid"] is not True]
    mine = "K1" if stop_k1 else ("K2" if len(k2_bad) >= 3 else None)
    A[q] = dict(per_seed=per, k2_values=d, k2_seeds_at_or_below_minus3=k2_bad, k2_fires=len(k2_bad) >= 3,
                k0_binding_failures=stop_k0, k1_fires_on=stop_k1, k3_on=stop_k3, quality_stopped_by=mine,
                seeds_usable=[k for k in SEEDS if per[k]["status"] == "finished" and k not in stop_k0 and k not in stop_k3 and not mine],
                kills_md=dict(stopped_by=K["A"][q].get("stopped_by"), seeds_usable=K["A"][q].get("seeds_usable"),
                              k2_values=K["A"][q].get("K2", {}).get("values")))
    if (K["A"][q].get("stopped_by") or None) != mine:
        problem(f"A {q}: KILLS stopped_by {K['A'][q].get('stopped_by')} vs recomputed {mine}")
    if sorted(K["A"][q].get("seeds_usable") or []) != A[q]["seeds_usable"]:
        problem(f"A {q}: KILLS seeds_usable {K['A'][q].get('seeds_usable')} vs recomputed {A[q]['seeds_usable']}")
    kv = {int(k): v for k, v in (K["A"][q].get("K2") or {}).get("values", {}).items()}
    if any(kv.get(k) != per[k]["k2_d"] for k in SEEDS):
        problem(f"A {q}: KILLS K2 values differ from the records")
    # the screen's seed-0 K2 value equals the record
    sv = (S["A"].get(q) or {})
KR["A"] = A

# ---------------- arm B (seed 0 only; B's seeds 1-4 not built)
B = {}
b_k1, b_k2, b_k3 = [], [], []
for q in Q5:
    p, r = arm_rec("B", q, 0)
    bca = r["details"]["rules"]["bca"]
    rungs = bca[0]["rungs"]
    cert = [bool(x["certified"]) for x in rungs]
    stat_ok = all((x["certified"] == (x["statistic"] is not None and num(x["statistic"]) < 0 and x["tries"] > 0)) for x in rungs)
    walk = EDGES[0]
    for j, ok in enumerate(cert, start=1):
        if not ok:
            break
        walk = EDGES[j]
    t_all = []
    for th in range(4):
        cc = [bool(x["certified"]) for x in bca[th]["rungs"]]
        w = EDGES[0]
        for j, ok in enumerate(cc, start=1):
            if not ok:
                break
            w = EDGES[j]
        t_all.append(w)
    t1 = r["maps"]["bca"][0]
    tries7 = rungs[6]["tries"]
    assert abs(rungs[6]["lower"] - EDGES[6]) < 1e-9 and abs(rungs[6]["upper"] - EDGES[7]) < 1e-9
    k1 = bool(t1 >= T085 and tries7 >= 30)
    k2 = bool(t1 <= T060)
    kk = r.get("kills") or {}
    k3 = bool((kk.get("K3") or {}).get("fires")) or r.get("stopped_by") == "K3"
    if k1:
        b_k1.append(q)
    if k2:
        b_k2.append(q)
    if k3:
        b_k3.append(q)
    ret_keys = []

    def walk_keys(o, path=""):
        if isinstance(o, dict):
            for kx, v in o.items():
                if kx in ("points", "J") and isinstance(v, (int, float)):
                    ret_keys.append(path + "/" + kx)
                walk_keys(v, path + "/" + kx)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk_keys(v, f"{path}[{i}]")
    walk_keys(r)
    B[q] = dict(t_map=r["maps"]["bca"], t_walk_recomputed=t_all, walk_matches=all(abs(a - b) < 1e-9 for a, b in zip(t_all, r["maps"]["bca"])),
                certified_equals_statistic_below_0=stat_ok, thruster1_tries_per_rung=[x["tries"] for x in rungs],
                thruster1_certified=cert, thruster1_statistics=[x["statistic"] for x in rungs],
                t1=t1, tries_080_085=tries7, rung_080_085_certified=rungs[6]["certified"], rung_085_090=dict(tries=rungs[7]["tries"], certified=rungs[7]["certified"], statistic=rungs[7]["statistic"]),
                k1_fires=k1, k2_t1_le_060=k2, k3=k3, recorded_k1=kk.get("K1"),
                K0_room=kk.get("K0_room") if q == "medium-expert" else None,
                K4=kk.get("K4_clip_share_thruster1"), return_like_fields=ret_keys, status=r["status"], finished_at=r.get("finished_at"))
    if not B[q]["walk_matches"] or not stat_ok:
        problem(f"B {q}: ceiling walk or certificate flags inconsistent")
    if (kk.get("K1") or {}).get("fires") != k1:
        problem(f"B {q}: recorded K1 differs from recomputed")
KR["B"] = dict(per_quality=B, k1_fires_on=b_k1, k2_pairs=b_k2, k3_seeds=b_k3,
               k0_room_seed0=B["medium-expert"]["K0_room"]["room"], k4_seed0=B["medium-expert"]["K4"],
               stopped=bool(b_k1) or len(b_k2) >= 2 or len(b_k3) >= 2 or B["medium-expert"]["K0_room"]["room"] < 1.0 or B["medium-expert"]["K4"] < 0.02,
               screen_B_stopped_by=S.get("B_stopped_by"), screen_K1_where=S["B_rules"]["K1"]["where"],
               kills_B=dict(stopped_by=K["B"].get("stopped_by"), built_for=K["B"].get("built_for"), stops=K["B"].get("stops"),
                            rules=K["B"].get("rules")),
               b_records_seeds_1_4=[str(p) for p in OUT.joinpath("B").rglob("seed[1-4]")],
               b_jobs_in_job_times=sorted(j for j in JOB if j.startswith("B_")))
if sorted(S["B_rules"]["K1"]["where"]) != sorted(b_k1):
    problem("B K1 qualities differ between screen and recomputation")
if K["B"].get("stopped_by") != ["K1"]:
    problem(f"KILLS B stopped_by {K['B'].get('stopped_by')}")
# no B return in results.json marks


def numbers_in(o, path=""):
    out = []
    if isinstance(o, dict):
        for kx, v in o.items():
            out += numbers_in(v, path + "/" + kx)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out += numbers_in(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out.append((path, o))
    return out


b_nums = [(p, v) for p, v in numbers_in(res["B"]) if re.search(r"per_seed|mean|lower_bound|positives|value|sum", p)]
KR["B"]["results_B_numbers"] = b_nums[:40]
KR["B"]["results_B_wording"] = res["B"].get("wording")
KR["B"]["results_B_GO"] = res["B"].get("GO")
KR["B"]["results_B_met"] = res["B"].get("met")
if b_nums:
    problem(f"results.json B holds numbers: {b_nums[:5]}")

# ---------------- arm C
Cd = {}
own_void = []
for q in Q5:
    per = {}
    for k in SEEDS:
        p, r = arm_rec("C", q, k)
        kk = r.get("kills") or {}
        kb1 = kk.get("KB1") or {}
        G1, G05 = kb1.get("G_rho1"), kb1.get("G_rho05")
        s_c = (r.get("spread") or {}).get("s")
        _, ra = arm_rec("A", q, k)
        s_a = (ra.get("inputs") or {}).get("s")
        bar = 3 * G1 if G1 > 0 else 3 * s_c
        kb1_fires = bool(G05 > bar)
        rungs = r["details"]["rules"]["bca"][0]["rungs"]
        cert = [bool(x["certified"]) for x in rungs]
        w = EDGES[0]
        for j, ok in enumerate(cert, start=1):
            if not ok:
                break
            w = EDGES[j]
        t1 = r["maps"]["bca"][0]
        tries7 = rungs[6]["tries"]
        void = bool(t1 >= T085 and tries7 >= 30)
        if void:
            own_void.append(f"{q} seed {k}")
        kb0 = kk.get("KB0_seed") or {}
        ret_keys = [x for x in r if x in ("evaluation", "evaluations", "returns", "points")]
        per[k] = dict(status=r["status"], stopped_by=r.get("stopped_by"), G_rho1=G1, G_rho05=G05, s_C=s_c, s_A=s_a,
                      s_equal=s_c == s_a, bar=bar, ratio=G05 / G1 if G1 else None, kb1_fires=kb1_fires,
                      kb1_recorded=kb1.get("fires"), finite=r.get("finite"), k3=(kk.get("K3") or {}).get("fires"),
                      share_rho05=kb0.get("share_rho05"), share_rho1=kb0.get("share_rho1"),
                      kb0_seed_pass=(kb0.get("share_rho05") >= 0.02 and kb0.get("share_rho05") >= 2 * kb0.get("share_rho1")) if kb0 else None,
                      t1=t1, t1_walk=w, tries_080_085=tries7, own_void=void, own_void_recorded=(kk.get("guard_void_own_map") or {}).get("fires"),
                      map=r["maps"]["bca"], final_return_keys=ret_keys, finished_at=r.get("finished_at"))
        if kb1.get("fires") != kb1_fires:
            problem(f"C {q} seed {k}: KB1 recorded differs")
        if abs(w - t1) > 1e-9:
            problem(f"C {q} seed {k}: guard map walk differs")
        if per[k]["own_void_recorded"] != void:
            problem(f"C {q} seed {k}: guard-void own map recorded differs")
        if ret_keys:
            problem(f"C {q} seed {k}: training record holds returns")
    stop = [k for k in SEEDS if per[k]["kb1_fires"]]
    Cd[q] = dict(per_seed=per, kb1_fires_on=stop, stopped_by="KB1" if stop else None,
                 kills_md=dict(stopped_by=K["C"][q].get("stopped_by"), seeds_usable=K["C"][q].get("seeds_usable")))
    if (K["C"][q].get("stopped_by") or None) != (Cd[q]["stopped_by"]):
        problem(f"C {q}: KILLS stopped_by differs")
kb0_means = {}
for q in WEAK:
    s05 = [Cd[q]["per_seed"][k]["share_rho05"] for k in SEEDS]
    s1 = [Cd[q]["per_seed"][k]["share_rho1"] for k in SEEDS]
    m05, m1 = sum(s05) / 5, sum(s1) / 5
    kb0_means[q] = dict(shares_rho05=s05, shares_rho1=s1, mean_rho05=m05, mean_rho1=m1, twice_rho1=2 * m1,
                        passes=bool(m05 >= 0.02 and m05 >= 2 * m1),
                        kills=K["C_rules"]["KB0"]["per_quality"][q])
kb0_seed0 = {q: Cd[q]["per_seed"][0]["kb0_seed_pass"] for q in WEAK}
kb0_holds_means = sum(v["passes"] for v in kb0_means.values()) >= 2
kb0_holds_seed0 = sum(bool(v) for v in kb0_seed0.values()) >= 2
void = bool(b_k1 or len(b_k2) >= 2 or own_void)
KR["C"] = dict(per_quality=Cd, kb0_means=kb0_means, kb0_holds_on_means=kb0_holds_means, kb0_seed0=kb0_seed0,
               kb0_holds_seed0=kb0_holds_seed0, own_map_void=own_void, guard_void=void, void_by_B_K1=bool(b_k1),
               B2_B4_scored_should_be=bool(kb0_holds_means and kb0_holds_seed0 and not void),
               kills_guard_void=K["C_rules"]["guard_void"], kills_B2_B4_scored=K["C_rules"]["B2_B4_scored"],
               kills_KB0_holds=K["C_rules"]["KB0"]["holds"],
               results_C_B2_B4_not_scored=res["C"].get("B2_B4_not_scored"),
               results_C_marks={m: {kx: v for kx, v in (res["C"].get(m) or {}).items() if kx in ("met", "scored", "status", "note")}
                                for m in ("B1", "B2", "B3", "B4", "B5")})
if sorted(K["C_rules"]["guard_void"]["by_own_map"]) != sorted(own_void):
    problem("C guard-void own-map list differs from KILLS")
if K["C_rules"]["B2_B4_scored"] != KR["C"]["B2_B4_scored_should_be"]:
    problem("C B2-B4 scored flag differs")
for q in WEAK:
    if kb0_means[q]["passes"] != K["C_rules"]["KB0"]["per_quality"][q]["passes"]:
        problem(f"C KB0 {q} differs")

# ---------------- F0
F = {}
for q in FQ:
    per = {}
    for k in SEEDS:
        p, r = arm_rec("F0", q, k)
        ch = r.get("checks") or {}
        a2p = F1 / q / "A2_cmd_strong" / f"seed{k}" / "record.json"
        a2 = load(a2p)
        a2_clone = (a2.get("clone") or {}).get("actor_sha256")
        f0_clone = (r.get("clone") or {}).get("actor_sha256")
        kf1 = ch.get("KF1") or {}
        kfn = ch.get("KF1_no_caution")
        kf2 = ch.get("KF2") or {}
        actor_ok = sha(p.parent / "actor.msgpack") == r.get("actor_sha256")
        per[k] = dict(status=r["status"], stopped_by=r.get("stopped_by"),
                      KF1_disagreement=kf1.get("disagreement"), KF1_rows=kf1.get("rows_disagreeing"),
                      KF1_share=kf1.get("no_credit_numpy"),
                      KF1_no_caution=(dict(disagreement=kfn.get("disagreement"), rows=kfn.get("rows_disagreeing")) if kfn else None),
                      KF2a=dict(passes=(kf2.get("a") or {}).get("passes"), max_param_diff=(kf2.get("a") or {}).get("max_param_diff")),
                      KF2b=dict(passes=(kf2.get("b") or {}).get("passes"), max_param_diff=(kf2.get("b") or {}).get("max_param_diff"),
                                replaced=(kf2.get("b") or {}).get("replaced")),
                      KF2_steps=kf2.get("steps"),
                      KF3_f0_clone=f0_clone, KF3_a2_clone_from_A2_record=a2_clone, KF3_equal=f0_clone == a2_clone and f0_clone is not None,
                      KF3_recorded_clone_equals=(ch.get("KF3") or {}).get("clone_sha256") == f0_clone,
                      KF4_finite=r.get("finite"), actor_hash_ok=actor_ok,
                      a2_record_sha_match=r.get("a2_record_sha256") == sha(a2p), a2_finished_at=a2.get("finished_at"),
                      f0_started_at=r.get("started_at"))
        ok = (kf1.get("disagreement") == 0.0 and kf1.get("rows_disagreeing") == 0
              and (k != 0 or (kfn and kfn.get("disagreement") == 0.0 and kfn.get("rows_disagreeing") == 0))
              and (kf2.get("a") or {}).get("max_param_diff") == 0.0 and (kf2.get("b") or {}).get("max_param_diff") == 0.0
              and per[k]["KF3_equal"] and r.get("finite") is True)
        per[k]["all_pass"] = bool(ok)
        if not ok:
            problem(f"F0 {q} seed {k}: a KF check fails")
        if not per[k]["a2_record_sha_match"]:
            problem(f"F0 {q} seed {k}: A2 record changed after F0 read it")
    F[q] = dict(per_seed=per, decided=all(v["all_pass"] for v in per.values()), kills=K["F0"][q].get("decided"))
    if F[q]["decided"] != K["F0"][q].get("decided"):
        problem(f"F0 {q}: decided differs")
KR["F0"] = F

# ================================================================================================ ordering
O = R.setdefault("ordering", {})
kills_w = window("kills")
screen_w = window("screen")
marks_w = window("marks")
train_jobs = [j for j in JOB if j.split("_")[0] in ("A", "B", "C", "F0", "A2", "A2eval", "host", "room", "band")]
late = [j for j in train_jobs if JOB[j][-1]["t1"] > kills_w[0]]
evals = [j for j in JOB if j.startswith("eval_")]
early_eval = [j for j in evals if JOB[j][-1]["t0"] < kills_w[1]]
eval_after_marks = [j for j in evals if JOB[j][-1]["t1"] > marks_w[0]]
O["kills_window"] = [fmt(kills_w[0]), fmt(kills_w[1])]
O["screen_window"] = [fmt(screen_w[0]), fmt(screen_w[1])]
O["marks_window"] = [fmt(marks_w[0]), fmt(marks_w[1])]
O["last_training_job_done"] = max((JOB[j][-1]["t1"], j) for j in train_jobs)
O["last_training_job_done"] = [fmt(O["last_training_job_done"][0]), O["last_training_job_done"][1]]
O["first_eval_start"] = min((JOB[j][-1]["t0"], j) for j in evals)
O["first_eval_start"] = [fmt(O["first_eval_start"][0]), O["first_eval_start"][1]]
O["last_eval_done"] = max((JOB[j][-1]["t1"], j) for j in evals)
O["last_eval_done"] = [fmt(O["last_eval_done"][0]), O["last_eval_done"][1]]
O["training_jobs_done_after_kills_start"] = late
O["eval_jobs_started_before_kills_done"] = early_eval
O["eval_jobs_done_after_marks_start"] = eval_after_marks
O["eval_counts"] = {a: sum(1 for j in evals if j.split("_")[1] == a) for a in ("A", "B", "C", "F")}
k_made = iso(K["made_at"])
ev_early_rec = [e["path"] for e in ev_list if iso(e["started_at"]) < k_made]
O["KILLS_made_at"] = K["made_at"]
O["eval_records_started_before_KILLS_made_at"] = ev_early_rec
O["results_made_at"] = res.get("made_at")
O["results_after_last_eval_record"] = iso(res["made_at"]) >= max(iso(e["finished_at"]) for e in ev_list)
# arm records (A and C, all seeds) finished before KILLS made_at, and they hold no final returns (checked above)
arm_fin = []
for arm, qs, seeds in (("A", Q5, SEEDS), ("C", Q5, SEEDS), ("B", Q5, (0,)), ("F0", FQ, SEEDS)):
    for q in qs:
        for k in seeds:
            _, r = arm_rec(arm, q, k)
            arm_fin.append((iso(r["finished_at"]), f"{arm} {q} seed {k}"))
O["last_arm_record_finished"] = [fmt(max(arm_fin)[0]), max(arm_fin)[1]]
O["arm_records_after_KILLS"] = [n for t, n in arm_fin if t > k_made]
# eval logs: exactly one attempt each; no eval job in the earlier runner logs
earlier = []
for lf in ("nohup_start.out", "nohup_continue.out", "nohup_continue2.out"):
    earlier += [line for line in (LOGS / lf).read_text().splitlines() if re.search(r"started: (eval_|kills|marks)", line)]
O["eval_kills_marks_in_earlier_runner_logs"] = earlier
O["eval_logs_with_more_than_one_attempt"] = [j for j, n in attempts.items() if j.startswith(("eval_", "kills", "marks")) and n > 1]
# screen: what return-like numbers it holds
scr_nums = [p for p, v in numbers_in(S) if re.search(r"/(points|J)$|points", p)]
O["screen_return_like_paths"] = scr_nums
for x in (late, early_eval, ev_early_rec, O["arm_records_after_KILLS"], earlier, O["eval_logs_with_more_than_one_attempt"]):
    if x:
        problem(f"ordering: {x}")

out = ARMS / "verify" / "kills_check_raw.json"
tmp = out.with_name(out.name + ".tmp")
tmp.write_text(json.dumps(R, indent=1, default=str))
os.replace(tmp, out)
print(json.dumps(dict(problems=R["problems"], notes=R["notes"]), indent=1))
