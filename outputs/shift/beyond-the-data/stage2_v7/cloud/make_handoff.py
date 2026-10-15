"""CLOUD ONLY. Stage 2, version 7: the handoff from the cloud to the laptop (V7_SPEC.md C6) and the way back.

Order at the user's go-ahead (README step 9; C6 step 1 first: the watchdog Routine and every other Routine that could
start a Stage 2 phase deleted and confirmed with the Routines list, the driver's sh processes stopped, the running
phases stopped at once unless the user chose 'let them finish', no run_stage2.py left):

  make_handoff.py import-selftest --from-transcript [JSONL] | --paste FILE
      Reads the block the laptop's full self-test printed, byte for byte: with --from-transcript, from the last
      message of the user's in this cloud session's own transcript that holds a block (default: the newest top-level
      transcript under ~/.claude/projects/); with --paste, from FILE, which a script saved (never retype the block: one
      wrong character fails its SHA-256 check). A block that is not a FULL self-test whose verdict starts with READY
      and whose repeatability test P3 passed is refused (C3, C11: a P3 failure ends v7; a NOT READY self-test is fixed
      on the laptop and the full self-test run again, which reruns no written probe record); the refusal is recorded
      in V7/handoff/probes/selftest_refused_<UTC>.json (its summary and why), and no record of it is stored. Otherwise
      it checks every probe record's SHA-256 and that it binds to the committed draft prereg_v7.json (its re-make and
      forecast block, the constants file, the gate's pre-registration, v6's code manifest, the probe script), all
      before storing any, and writes each record, byte for byte, once, to V7/handoff/probes/selftest/<probe>.json
      (+ .sha256), and an import record V7/handoff/probes/selftest_import_<UTC>.json (+ .sha256): the summary, its
      SHA-256, where the block was read and the cloud's own time of the import. A record already there must be
      identical (never replaced). It warns when the self-test ran under another draft than the committed one.
  make_handoff.py ready
      Before the cloud stops anything (C6 step 1): whether the self-test side of the handoff holds. The most recent
      full self-test pasted (accepted or refused) must be an accepted import that ran under the committed draft; no
      pasted full self-test may have failed P3 (a P3 failure ends v7); all nine probe records must be imported and
      bind to the draft. 'write' checks the same. Reads only.
  make_handoff.py snapshot [--push] [--trailer TEXT]
      The 'part 4' commit on claude/stage2-run-files (C6 step 2): the run folders of parts 1-3 as they are now (logs
      included, lock files left out) on top of the branch's tip, with a new RUNFILES_SHA256SUMS computed from the
      committed bytes, through a temporary git index (the main checkout and index are not touched). Refuses while a
      run_stage2.py process or the driver explore_policy.sh runs, and when a file of the tip other than a log changed
      or disappeared (write-once).
  make_handoff.py write --snapshot COMMIT --stop-rule 'stop at once'|'let them finish'
                        --stop-rule-chosen YYYY-MM-DDTHH:MM:SSZ|default --go-ahead UTC --stop UTC --routines FILE
                        --processes FILE
      V7/handoff/HANDOFF.json (+ .sha256), written once (C6 step 3): the snapshot commit and the SHA-256 of its sums
      file; the stop time; the stop rule, when it was chosen (a UTC time in exactly that form, or 'default'), the write
      times of the nine P1/P2 records and the cloud's time of the first self-test paste (the choice must come before
      both, or be the default 'stop at once'; against the records' laptop times it allows the clock tolerance the full
      self-test checks, ST9 clock, and records the measured offsets); every self-test pasted and the full self-test
      this handoff rests on (as 'ready' checks; its import record's SHA-256, verdict, P3 result and the draft it ran
      under); the go-ahead's time; each Routine's id, name and final
      state (FILE: a JSON list of {id, name, final_state, starts_phases}; every Routine that can start a phase must be
      'deleted'); the processes stopped (FILE: a JSON list), the check that no run_stage2.py and no explore_policy.sh
      process is left, and the driver log's last lines; for every seed and phase, 'inherited', 'partial' or 'not
      started' by the phase table on the snapshot's own bytes (C6 step 4), and the partial files; the cloud's
      process-hours from its logs (C2); the draft prereg_v7.json it goes with (write_prereg_v7.py final refuses
      another draft, and write_prereg_v7.py draft then refuses to write any other draft: HANDOFF.json is written once).
      Then: write_prereg_v7.py final --handoff V7/handoff/HANDOFF.json --records V7/handoff/probes
  make_handoff.py withdraw --reason TEXT --no-laptop-phase-started --user-quote TEXT --user-utc UTC
                           [--ledger-listing FILE]
      V7/handoff/WITHDRAWN.json (+ .sha256), once (C6 step 6): v7 is not approved, or the laptop refused before its
      first phase started. It records the user's own confirmation, quoted, with the UTC time of their message, that no
      v7 phase has started on the laptop; with --ledger-listing (the output of
      'ls -1 /home/user/bca/runs/shift_explore/stage2_v7/ledger', pasted by the user) it is stored and refused when it
      names any folder other than a probe's. The v7 runner then refuses; the cloud may resume v6 with
      explore_policy.sh, at most 2 processes, with a new watchdog only if the user asks.

'--v7-test <fixture>' before the command is for test_v7_laptop.py only (a fixture may give 'proc_root', a synthetic
/proc, for the process check).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V7_DIR = HERE.parent
LIB = V7_DIR / "laptop" / "lib"
CODE_DIR = V7_DIR / "code"
for _p in (str(LIB), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import v7_common as K  # noqa: E402
import runner_v7 as RN  # noqa: E402
import selftest_v7 as ST  # noqa: E402

WATCHDOG = "trig_01YCP5oPNVkUPRgta25ag1y4"                  # "Stage 2 run watchdog (hourly)" (V7_SPEC.md section 2)
STOP_RULES = ("stop at once", "let them finish")
FOLDERS = ("runs/shift", "runs/shift_explore/room", "runs/shift_explore/standins", "runs/shift_explore/fallback",
           "runs/shift_explore/deploy", "runs/shift_explore/anatomy", "runs/shift_explore/principled",
           "runs/shift_explore/empirical", "runs/shift_explore/detect", "runs/shift_explore/stage2")
DRIVER_LOG = ST.LOGS_REL + "/explore_policy.log"
LINE_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (start|done|FAILED)\b")
AGENT = ("Claude", "noreply@anthropic.com")


def handoff_dir(cfg):
    return cfg.v7_root / "handoff"


def probes_dir(cfg):
    return handoff_dir(cfg) / "probes"


def cloud_processes(proc_root="/proc"):
    """Read only (/proc): every process whose command line names run_stage2.py (the cloud's v6 phases) or the driver
    explore_policy.sh (which starts the next v6 phase as soon as one ends, C6 step 1). This process and any process
    naming make_handoff are left out. proc_root: a folder laid out as /proc (the tests give a synthetic one)."""
    out = dict(run_stage2_py=[], explore_policy_sh=[])
    for d in sorted(Path(proc_root).iterdir(), key=lambda p: p.name):
        if not d.name.isdigit() or (str(proc_root) == "/proc" and int(d.name) == os.getpid()):
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf8", "replace")
        except OSError:
            continue
        if "make_handoff" in cmd:
            continue
        if "run_stage2.py" in cmd:
            out["run_stage2_py"].append(dict(pid=int(d.name), cmd=cmd.strip()[:300]))
        if "explore_policy.sh" in cmd:
            out["explore_policy_sh"].append(dict(pid=int(d.name), cmd=cmd.strip()[:300]))
    return out


def stopped_check(procs):
    """C6 step 1 on a process listing: refuses while a v6 phase or the driver runs; else the check, for HANDOFF.json."""
    if procs["run_stage2_py"] or procs["explore_policy_sh"]:
        raise K.V7Refusal(f"Stage 2 processes still run (run_stage2.py: {procs['run_stage2_py']}; the driver "
                          f"explore_policy.sh: {procs['explore_policy_sh']}); stop the driver first, then the phases "
                          f"(C6 step 1)")
    return dict(procs, checked_utc=K.utc_now())


def require_cloud_stopped(cfg):
    """C6 step 1: no v6 phase and no driver left. Returns the check, for HANDOFF.json. A test fixture reads its own
    synthetic process folder (runner.json 'proc_root'), or nothing."""
    if cfg.test:
        if not cfg.extra.get("proc_root"):
            return dict(checked_utc=K.utc_now(), run_stage2_py=[], explore_policy_sh=[], note="test fixture: not checked")
        return dict(stopped_check(cloud_processes(cfg.extra["proc_root"])), note="test fixture: synthetic processes")
    return stopped_check(cloud_processes())


UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def parse_utc(text, what):
    """A UTC time in exactly the form YYYY-MM-DDTHH:MM:SSZ, as a datetime; V7Refusal otherwise."""
    from datetime import datetime
    try:
        return datetime.strptime(str(text), UTC_FORMAT)
    except ValueError:
        raise K.V7Refusal(f"{what} must be a UTC time in exactly the form 2026-10-14T09:00:00Z (got {text!r})") from None


def selftest_problems(summary):
    """Why a pasted self-test summary is not one a handoff may rest on (C3, C11): it must be the FULL form, its
    verdict must start with READY, and P3 must have passed."""
    problems = []
    if (summary or {}).get("form") != "full":
        problems.append(f"it is the {(summary or {}).get('form')!r} form, not the full self-test")
    if not str((summary or {}).get("verdict") or "").startswith("READY"):
        problems.append(f"its verdict is {(summary or {}).get('verdict')!r}")
    if ((summary or {}).get("p3") or {}).get("ok") is not True:
        problems.append("its repeatability test P3 did not pass")
    return problems


# --------------------------------------------------------------------------------------------- import-selftest

def bind_check(prereg, rec):
    """The fingerprints a probe record must bind to (as write_prereg_v7.build_final checks)."""
    import v7_probes as PR
    return PR.binding_problems(prereg.payload, rec)


def transcript_block(path=None):
    """(text, source): the last message a person sent this cloud session that holds a V7 self-test block, read byte
    for byte from the session's own transcript (a .jsonl), so nothing is retyped. path: the transcript; default the
    most recently written top-level transcript under ~/.claude/projects/ (this session's)."""
    if path in (None, "", "auto"):
        cands = sorted(Path.home().glob(".claude/projects/*/*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise K.V7Refusal("no session transcript under ~/.claude/projects/; save the user's message with a script "
                              "and give --paste FILE (never retype the block)")
        path = cands[-1]
    found = None
    for line in Path(path).read_text(encoding="utf8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict) or d.get("type") != "user" or d.get("isSidechain") or d.get("isMeta"):
            continue
        content = (d.get("message") or {}).get("content")
        texts = [content] if isinstance(content, str) else [
            c.get("text") for c in (content or []) if isinstance(c, dict) and c.get("type") == "text"]
        for text in texts:
            if isinstance(text, str) and ST.BLOCK_BEGIN in text:
                found = (text, d.get("uuid"), d.get("timestamp"))
    if found is None:
        raise K.V7Refusal(f"no message of the user's in {path} holds a V7 self-test block")
    return found[0], dict(transcript=str(path), message_uuid=found[1], message_timestamp=found[2])


def stamp_of(path):
    """The UTC stamp in a 'selftest_import_<stamp>.json' or 'selftest_refused_<stamp>.json' name (sorts in time)."""
    return Path(path).name.rsplit("_", 1)[-1][:-len(".json")]


def pasted_selftests(cfg):
    """Every self-test block pasted to the cloud, accepted or refused, oldest first: [(stamp, 'accepted' |
    'refused', path, record)] (the cloud's own clock orders them)."""
    rows = []
    for kind, pattern in (("accepted", "selftest_import_*.json"), ("refused", "selftest_refused_*.json")):
        for p in probes_dir(cfg).glob(pattern):
            rows.append((stamp_of(p), kind, p, K.read_json_verified(p, f"the self-test record {p.name}")))
    return sorted(rows, key=lambda r: (r[0], r[2].name))


def import_selftest(cfg, paste=None, out=print, text=None, source=None):
    """paste: a file holding the user's message; or text (from transcript_block) with its source."""
    import v7_probes as PR
    import v7_verify as VF
    if text is None:
        text, source = Path(paste).read_text(encoding="utf8", errors="replace"), dict(file=str(paste))
    summary, records = ST.decode_block(text)
    problems = selftest_problems(summary)
    if problems:
        rec = dict(kind="stage2-v7-selftest-refused", imported_utc=K.utc_now(), summary=summary,
                   summary_sha256=K.block_sha256(summary), problems=problems, records=sorted(records), source=source,
                   rule=("kept so that 'make_handoff.py write' rests on the most recent full self-test and sees every "
                         "P3 failure (C3, C11); no probe record of this block was stored"))
        K.write_json_once(probes_dir(cfg) / f"selftest_refused_{K.utc_stamp()}.json", rec)
        raise K.V7Refusal("the pasted block is not from a READY full self-test with P3 passed (" + "; ".join(problems) +
                          "); no probe record was stored (the refusal is recorded). If P3 failed, v7 ends as "
                          "specified; otherwise fix what the self-test names and run the full self-test again (it "
                          "reruns no written probe record)")
    prereg = VF.load_prereg_v7(cfg.prereg)
    pins = ((prereg.payload.get("handoff") or {}).get("probe_records") or {}) if prereg.final else {}
    d = probes_dir(cfg) / "selftest"
    checked = []
    for pid, data in sorted(records.items()):                  # every record is checked before any is stored
        if pid not in PR.all_probe_ids():
            raise K.V7Refusal(f"the block holds an unknown record {pid!r}; nothing was stored")
        rec = json.loads(data.decode("utf8"))
        if rec.get("probe") != pid:
            raise K.V7Refusal(f"the record named {pid} is the record of {rec.get('probe')}; nothing was stored")
        bad = bind_check(prereg, rec)
        if bad:
            raise K.V7Refusal(f"probe record {pid} does not bind to the committed prereg_v7.json ({bad}); nothing was "
                              f"stored: " + PR.binding_advice([rec]))
        sha = K.sha256_bytes(data)
        if pins and (pins.get(pid) or {}).get("sha256") != sha:
            raise K.V7Refusal(f"HALT: probe record {pid} differs from its pin in the final prereg_v7.json")
        p = d / f"{pid}.json"
        if p.exists() and p.read_bytes() != data:
            raise K.V7Refusal(f"{p} exists with other bytes; a probe record is never replaced; nothing was stored")
        checked.append((pid, data, sha, p))
    new, same = [], []
    for pid, data, sha, p in checked:
        if p.exists():
            same.append(pid)
            continue
        K.write_once(p, data)
        K.write_once(Path(str(p) + ".sha256"), f"{sha}  {p.name}\n".encode("utf8"))
        new.append(pid)
    other_draft = summary.get("prereg_v7_sha256") != prereg.sha256
    imp = dict(kind="stage2-v7-selftest-import", imported_utc=K.utc_now(), summary=summary,
               summary_sha256=K.block_sha256(summary), records=sorted(records), new=new, same=same, source=source,
               committed_prereg_v7_sha256=prereg.sha256, ran_under_committed_prereg=not other_draft,
               rule="the cloud's own time of the import; the stop rule must be chosen before it (C6)")
    K.write_json_once(probes_dir(cfg) / f"selftest_import_{K.utc_stamp()}.json", imp)
    out(f"self-test ({summary.get('form')}): {summary.get('verdict')}; P3 passed")
    for line in summary.get("probe_lines") or []:
        out(line)
    out(f"probe records: {len(new)} new, {len(same)} already here and identical; "
        f"{len(PR.all_probe_ids()) - len(new) - len(same)} not in this block")
    clock = summary.get("clock") or {}
    if clock.get("offset_seconds") is None:
        out(f"WARN the laptop's clock was not compared with a reference ({clock.get('error')}); the stop rule's time is "
            f"compared with the records' laptop times as they are")
    if other_draft:
        out(f"WARN this self-test ran under prereg_v7 {str(summary.get('prereg_v7_sha256'))[:16]}..., not the committed "
            f"draft {prereg.sha256[:16]}...: the handoff needs a full self-test under the committed draft. Ask the user "
            f"to type 'git -C /home/user/bca pull --ff-only' and run step 8 again (a written probe record is not rerun)")
    return summary, new, same


# --------------------------------------------------------------------------------------------- snapshot (part 4)

def run_git(cfg, *args, env=None, input=None, binary=False):
    full = dict(RN.BASE_ENV, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", **(env or {}))
    done = subprocess.run(["git", "-C", str(cfg.root)] + list(args), capture_output=True, env=full, input=input,
                          timeout=7200)
    if done.returncode != 0:
        raise K.V7Refusal(f"git {' '.join(str(a) for a in args[:3])} failed: "
                          f"{done.stderr.decode('utf8', 'replace').strip()[-300:]}")
    return done.stdout if binary else done.stdout.decode("utf8", "replace").strip()


def remote_tip(cfg, branch):
    text = run_git(cfg, "ls-remote", "origin", f"refs/heads/{branch}")
    return text.split()[0] if text else None


def tree_sums(cfg, tree):
    """{path: SHA-256 of the committed bytes} for every file of a tree, streamed (the run files are about 0.8 GB)."""
    import hashlib
    import threading
    rows = [ln.split("\t", 1) for ln in run_git(cfg, "ls-tree", "-r", tree).splitlines()]
    blobs = [(meta.split()[2], path) for meta, path in rows if meta.split()[1] == "blob"]
    req = "".join(f"{b}\n" for b, _ in blobs).encode("utf8")
    proc = subprocess.Popen(["git", "-C", str(cfg.root), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, env=dict(RN.BASE_ENV, GIT_OPTIONAL_LOCKS="0"))

    def feed():
        proc.stdin.write(req)
        proc.stdin.close()

    threading.Thread(target=feed, daemon=True).start()
    out = {}
    for _, path in blobs:
        head = proc.stdout.readline().split()
        size = int(head[2])
        h, left = hashlib.sha256(), size
        while left:
            chunk = proc.stdout.read(min(1 << 20, left))
            if not chunk:
                raise K.V7Refusal("git cat-file ended early")
            h.update(chunk)
            left -= len(chunk)
        proc.stdout.read(1)
        out[path] = h.hexdigest()
    proc.stdout.close()
    if proc.wait() != 0:
        raise K.V7Refusal("git cat-file --batch failed")
    return out


def snapshot(cfg, push=False, trailer="", out=print):
    require_cloud_stopped(cfg)
    branch = cfg.extra.get("runfiles_branch", ST.RUNFILES_BRANCH)
    parent = remote_tip(cfg, branch)
    if parent is None:
        raise K.V7Refusal(f"origin has no branch {branch}")
    if subprocess.run(["git", "-C", str(cfg.root), "cat-file", "-e", f"{parent}^{{commit}}"]).returncode != 0:
        run_git(cfg, "fetch", "--no-tags", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
    old = tree_sums(cfg, parent)
    files = []
    for folder in FOLDERS:
        base = cfg.root / folder
        if base.is_dir():
            files += [p.relative_to(cfg.root).as_posix() for p in sorted(base.rglob("*"))
                      if p.is_file() and not p.name.endswith(".lock")]
    have = set(files)
    gone = [p for p in old if p.startswith("runs/") and p not in have]
    changed = [p for p in old if p in have and not p.startswith(ST.LOGS_REL + "/") and p != ST.SUMS_NAME
               and K.file_sha256(cfg.root / p) != old[p]]
    if gone or changed:
        raise K.V7Refusal(f"write-once files of the snapshot changed {changed[:5]} or disappeared {gone[:5]}")
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(GIT_INDEX_FILE=str(Path(tmp) / "index"), GIT_AUTHOR_NAME=AGENT[0], GIT_AUTHOR_EMAIL=AGENT[1],
                   GIT_COMMITTER_NAME=AGENT[0], GIT_COMMITTER_EMAIL=AGENT[1])
        run_git(cfg, "read-tree", parent, env=env)
        run_git(cfg, "update-index", "--add", "-z", "--stdin", env=env,
                input=b"".join(p.encode("utf8") + b"\0" for p in files))
        tree = run_git(cfg, "write-tree", env=env)
        sums = {p: h for p, h in tree_sums(cfg, tree).items() if p.startswith("runs/")}
        text = "".join(f"{sums[p]}  {p}\n" for p in sorted(sums)).encode("utf8")
        blob = run_git(cfg, "hash-object", "-w", "--stdin", input=text)
        run_git(cfg, "update-index", "--add", "--cacheinfo", f"100644,{blob},{ST.SUMS_NAME}", env=env)
        tree = run_git(cfg, "write-tree", env=env)
        msg = (f"Stage 2 run files, part 4: the final snapshot at the handoff to the laptop (V7_SPEC.md C6 step 2)\n\n"
               f"{len(sums)} files under runs/, logs included, and the SHA-256 of every file (RUNFILES_SHA256SUMS). "
               f"Taken {K.utc_now()} after the cloud's Stage 2 processes stopped.")
        if trailer:
            msg += "\n\n" + trailer.strip()
        commit = run_git(cfg, "commit-tree", tree, "-p", parent, "-m", msg, env=env)
    res = dict(commit=commit, parent=parent, files=len(sums), sums_sha256=K.sha256_bytes(text), branch=branch)
    if push:
        for n in range(5):
            try:
                run_git(cfg, "push", "origin", f"{commit}:refs/heads/{branch}")
                break
            except K.V7Refusal as error:
                if n == 4:
                    raise
                out(f"WARN push failed ({error}); retrying")
                time.sleep(2 ** (n + 1))
        res["pushed"] = True
    out(f"snapshot commit {commit} on {branch} (parent {parent[:12]}): {len(sums)} files, sums "
        f"{res['sums_sha256']}" + (" (pushed)" if push else " (not pushed: add --push)"))
    return res


# --------------------------------------------------------------------------------------------- HANDOFF.json

def cloud_process_hours(cfg, commit, sums):
    """From the snapshot's own logs (launch.sh's 'start:', 'done:' and 'FAILED:' lines): completed attempts' hours,
    and the number of attempts with no end line (killed; their length is unknown)."""
    from datetime import datetime
    logs = [p for p in sums if p.startswith(ST.LOGS_REL + "/") and p.rsplit("/", 1)[-1].startswith("phase_")]
    blobs = ST.cat_files(cfg, commit, logs)
    per, total, killed = {}, 0.0, 0
    for path, data in sorted(blobs.items()):
        if data is None:
            continue
        start, hours, n_k, n_done = None, 0.0, 0, 0
        for line in data.decode("utf8", "replace").splitlines():
            m = LINE_RE.match(line)
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ")
            if m.group(2) == "start":
                if start is not None:
                    n_k += 1
                start = t
            elif start is not None:
                hours += (t - start).total_seconds() / 3600.0
                n_done += 1
                start = None
        if start is not None:
            n_k += 1
        per[path.rsplit("/", 1)[-1]] = dict(completed_attempts=n_done, hours=round(hours, 3), without_end=n_k)
        total += hours
        killed += n_k
    return dict(total_completed_hours=round(total, 3), attempts_without_end=killed, per_log=per,
                note="the cloud's processor; attempts without an end line were killed and their length is unknown")


def p3_differs(p3):
    """True when a self-test's P3 found different results (C3: v7 ends); a summary without 'differ' counts any
    failure."""
    p3 = p3 or {}
    return p3.get("differ") is True or ("differ" not in p3 and p3.get("ok") is False)


def chosen_in_time(chosen, first_record, first_paste, tolerance_s=None):
    """C6: the stop rule's choice (the cloud's time of the user's message) came before the first P1/P2 record (the
    laptop's clock, which the full self-test checks to within CLOCK_TOLERANCE_S before any probe runs, so that
    tolerance is allowed) and before the first self-test paste (the cloud's clock, compared exactly)."""
    tol = timedelta(seconds=ST.CLOCK_TOLERANCE_S if tolerance_s is None else tolerance_s)
    return chosen < first_record + tol and chosen < first_paste


def handoff_selftest(cfg, prereg):
    """The self-test facts a handoff rests on (C3, C6 step 3, C11), checked without changing anything: the most recent
    full self-test pasted to the cloud must be an accepted (READY, P3 passed) import that ran under the committed
    draft; no pasted full self-test may have failed P3 (a P3 failure ends v7); all nine probe records must be
    imported and bind to the draft. 'make_handoff.py ready' runs this before the cloud stops anything."""
    import v7_probes as PR
    pasted = pasted_selftests(cfg)
    full = [r for r in pasted if ((r[3].get("summary") or {}).get("form")) == "full"]
    # P3 failed: finished runs gave different results (a run that did not finish shows nothing and only needs a rerun)
    p3_failed = [r for r in full if p3_differs((r[3].get("summary") or {}).get("p3"))]
    if p3_failed:
        raise K.V7Refusal(f"a full self-test pasted at {p3_failed[0][3].get('imported_utc')} failed the repeatability "
                          f"test P3: v7 ends (C3, C11). Nothing is handed off; the cloud may resume v6 after "
                          f"make_handoff.py withdraw")
    imports = [r for r in pasted if r[1] == "accepted"]
    if not imports:
        raise K.V7Refusal("no full self-test was imported (make_handoff.py import-selftest)")
    stamp, kind, last_path, last = full[-1]
    if kind != "accepted":
        raise K.V7Refusal(f"the most recent full self-test pasted (at {last.get('imported_utc')}) was refused "
                          f"({'; '.join(last.get('problems') or [])}); the handoff rests on the most recent full "
                          f"self-test: fix what it names, run step 8 again and import it")
    problems = selftest_problems(last.get("summary"))
    if problems:
        raise K.V7Refusal(f"the last imported self-test cannot carry a handoff: {'; '.join(problems)}")
    ran_under = (last.get("summary") or {}).get("prereg_v7_sha256")
    if ran_under != prereg.sha256:
        raise K.V7Refusal(f"the most recent full self-test ran under prereg_v7 {str(ran_under)[:16]}..., not the "
                          f"committed draft {prereg.sha256[:16]}...: the user types 'git -C /home/user/bca pull "
                          f"--ff-only', runs step 8 again (a written probe record is not rerun) and the cloud imports "
                          f"it (C11)")
    # every paste counts, refused ones too: each shows the probe results to the cloud (C6: the stop rule comes first)
    first_import = min((parse_utc(i.get("imported_utc"), "an import time"), i.get("imported_utc"))
                       for _, _, _, i in pasted)
    probes = {}
    for pid in PR.all_probe_ids():
        p = probes_dir(cfg) / "selftest" / f"{pid}.json"
        rec = K.read_json_verified(p, f"probe record {pid} (import the self-test block first)")
        bad = bind_check(prereg, rec)
        if bad:
            raise K.V7Refusal(f"probe record {pid} does not bind to the committed draft ({bad}): "
                              + PR.binding_advice([rec]))
        probes[pid] = dict(sha256=K.file_sha256(p), written_utc=rec.get("written_utc"))
    offsets = {r[0]: ((r[3].get("summary") or {}).get("clock") or {}).get("offset_seconds") for r in full}
    rows = [dict(stamp=r[0], kind=r[1], record=r[2].name, imported_utc=r[3].get("imported_utc"),
                 form=(r[3].get("summary") or {}).get("form"), verdict=(r[3].get("summary") or {}).get("verdict"),
                 p3_ok=((r[3].get("summary") or {}).get("p3") or {}).get("ok")) for r in pasted]
    return dict(last_path=last_path, last=last, first_import=first_import, probes=probes, clock_offsets=offsets,
                pasted=rows)


def ready(cfg, out=print):
    """'make_handoff.py ready': before the cloud stops anything (C6 step 1), whether the self-test side of the handoff
    holds (handoff_selftest); reads only."""
    import v7_verify as VF
    prereg = VF.load_prereg_v7(cfg.prereg)
    if prereg.final:
        raise K.V7Refusal("prereg_v7.json is final already; the handoff goes with the draft")
    st = handoff_selftest(cfg, prereg)
    out(f"ready: the most recent full self-test ({st['last'].get('imported_utc')}) is READY with P3 passed and ran "
        f"under the committed draft {prereg.sha256[:16]}...; all nine probe records are imported and bind to it; first "
        f"self-test paste {st['first_import'][1]}; laptop clock offsets {st['clock_offsets']}")
    return st


def write_handoff(cfg, a, out=print):
    import v7_verify as VF
    path = handoff_dir(cfg) / "HANDOFF.json"
    if path.exists() or (handoff_dir(cfg) / "WITHDRAWN.json").exists():
        raise K.V7Refusal(f"{path} or WITHDRAWN.json exists; HANDOFF.json is written once and never edited")
    if a.stop_rule not in STOP_RULES:
        raise K.V7Refusal(f"--stop-rule must be one of {STOP_RULES}")
    if a.stop_rule_chosen == "default" and a.stop_rule != "stop at once":
        raise K.V7Refusal("with no message from the user the default holds: 'stop at once'")
    chosen = None if a.stop_rule_chosen == "default" else parse_utc(a.stop_rule_chosen, "--stop-rule-chosen")
    prereg = VF.load_prereg_v7(cfg.prereg)
    if prereg.final:
        raise K.V7Refusal("prereg_v7.json is final already; the handoff goes with the draft")
    process_check = require_cloud_stopped(cfg)
    st = handoff_selftest(cfg, prereg)
    last_path, last, first_import = st["last_path"], st["last"], st["first_import"]
    branch = cfg.extra.get("runfiles_branch", ST.RUNFILES_BRANCH)
    tip = remote_tip(cfg, branch)
    if tip != a.snapshot:
        raise K.V7Refusal(f"the snapshot {a.snapshot[:12]} is not the tip of origin/{branch} ({str(tip)[:12]}); push it")
    code, sums_bytes = ST.git(cfg, "show", f"{a.snapshot}:{ST.SUMS_NAME}", binary=True)
    if code != 0:
        raise K.V7Refusal(f"{ST.SUMS_NAME} is not in {a.snapshot[:12]}")
    sums = ST.parse_sums(sums_bytes)
    if tree_sums(cfg, a.snapshot) != dict(sums, **{ST.SUMS_NAME: K.sha256_bytes(sums_bytes)}):
        raise K.V7Refusal(f"{ST.SUMS_NAME} of {a.snapshot[:12]} does not match the committed files")
    states, partial = ST.snapshot_states(cfg, a.snapshot, sums)
    probes = st["probes"]
    first_dt, first = min((parse_utc(v["written_utc"], f"the write time of probe record {pid}"), v["written_utc"])
                          for pid, v in probes.items())
    if chosen is not None and not chosen_in_time(chosen, first_dt, first_import[0]):
        raise K.V7Refusal(f"the stop rule was chosen at {a.stop_rule_chosen}, not before the first P1/P2 record "
                          f"({first}, the laptop's clock, which the full self-test checks to within "
                          f"{ST.CLOCK_TOLERANCE_S} s; measured offsets {st['clock_offsets']}) and the first self-test "
                          f"import ({first_import[1]}, the cloud's clock); it cannot be changed after any record exists "
                          f"(C6)")
    routines = json.loads(Path(a.routines).read_text(encoding="utf8"))
    if not any(r.get("id") == WATCHDOG for r in routines):
        raise K.V7Refusal(f"the routines list does not name the watchdog {WATCHDOG}")
    live = [r for r in routines if r.get("starts_phases") and r.get("final_state") != "deleted"]
    if live:
        raise K.V7Refusal(f"Routines that can start a Stage 2 phase are not deleted: {live}")
    processes = json.loads(Path(a.processes).read_text(encoding="utf8"))
    log = ST.cat_files(cfg, a.snapshot, [DRIVER_LOG]).get(DRIVER_LOG) or b""
    payload = dict(
        kind="stage2-v7-handoff", written_utc=K.utc_now(),
        snapshot=dict(commit=a.snapshot, branch=branch, sums_sha256=K.sha256_bytes(sums_bytes), files=len(sums)),
        stop_utc=a.stop, stop_rule=a.stop_rule, stop_rule_chosen_utc=a.stop_rule_chosen,
        probe_records=probes, first_probe_record_utc=first, first_selftest_import_utc=first_import[1],
        laptop_clock=dict(tolerance_seconds=ST.CLOCK_TOLERANCE_S, offsets_by_selftest=st["clock_offsets"],
                          rule=("the full self-test compares the laptop's clock with a reference before any probe runs "
                                "and runs no probe when they differ by more than the tolerance; the stop rule's time "
                                "is compared with the records' laptop times allowing the tolerance (C6, C11)")),
        pasted_selftests=st["pasted"],
        selftest=dict(import_record=last_path.relative_to(cfg.v7_root).as_posix(),
                      import_record_sha256=K.file_sha256(last_path), imported_utc=last.get("imported_utc"),
                      summary_sha256=last.get("summary_sha256"), form=last["summary"].get("form"),
                      verdict=last["summary"].get("verdict"), p3=last["summary"].get("p3"),
                      selftest_utc=last["summary"].get("utc"),
                      prereg_v7_sha256_at_selftest=last["summary"].get("prereg_v7_sha256")),
        go_ahead_utc=a.go_ahead, routines=routines,
        processes_stopped=processes, process_check=process_check,
        last_log_lines=log.decode("utf8", "replace").splitlines()[-30:],
        phases=states, partial_files=partial,
        counts={s: sum(1 for v in states.values() if v == s) for s in ("inherited", "partial", "not started")},
        cloud_process_hours=cloud_process_hours(cfg, a.snapshot, sums),
        draft_prereg_v7_sha256=prereg.sha256,
        rule=("a phase is inherited when its final record or failed-on-record file is in the final snapshot with a "
              "matching sidecar (C6 step 4); partial files stay in the snapshot and are not extracted on the laptop"))
    digest = K.write_json_once(path, payload)
    out(f"wrote {path} ({digest[:16]}): {payload['counts']}; {len(partial)} partial files; stop rule "
        f"'{a.stop_rule}'. Next: write_prereg_v7.py final --handoff {path} --records {probes_dir(cfg)}")
    return payload


LISTING_PROBE = re.compile(r"probe_[A-Za-z0-9_]+/?")


def listing_phase_folders(text):
    """The folders a pasted 'ls -1 .../stage2_v7/ledger' names that are not a probe's (the command line itself, a
    prompt and empty lines are ignored)."""
    out = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or "ls " in line or line.endswith(("#", "$")) or line.startswith(("ls:", "total ")):
            continue
        for word in line.split():
            if not LISTING_PROBE.fullmatch(word):
                out.append(word)
    return out


def withdraw(cfg, reason, confirmed, out=print, user_quote=None, user_utc=None, listing=None):
    path = handoff_dir(cfg) / "WITHDRAWN.json"
    if path.exists():
        raise K.V7Refusal(f"{path} exists (written once)")
    if not confirmed:
        raise K.V7Refusal("--no-laptop-phase-started is required: the user confirms that no v7 phase has started on "
                          "the laptop (no ledger file of a phase exists); otherwise the cloud never resumes Stage 2")
    if not (user_quote or "").strip():
        raise K.V7Refusal("--user-quote is required: the user's own words confirming that no v7 phase has started on "
                          "the laptop, quoted from their message")
    parse_utc(user_utc, "--user-utc (the time of the user's message)")
    listing_text = None
    if listing is not None:
        listing_text = Path(listing).read_text(encoding="utf8", errors="replace")
        phases = listing_phase_folders(listing_text)
        if phases:
            raise K.V7Refusal(f"the pasted ledger listing names phase folders {phases[:5]}: a v7 phase has started on "
                              f"the laptop, so the cloud never resumes Stage 2 (C6 step 6); any further route is the "
                              f"user's decision, as a new version")
    hand = handoff_dir(cfg) / "HANDOFF.json"
    payload = dict(kind="stage2-v7-withdrawn", written_utc=K.utc_now(), reason=reason,
                   handoff_sha256=K.file_sha256(hand) if hand.exists() else None, no_laptop_phase_started=True,
                   user_confirmation=dict(quote=user_quote.strip(), utc=user_utc),
                   ledger_listing=listing_text,
                   rule=("v7 is withdrawn: the v7 runner refuses from now on; the cloud may resume v6 with "
                         "explore_policy.sh, at most 2 processes, with a new watchdog only if the user asks (C6 step 6). "
                         "HANDOFF.json is never edited."))
    digest = K.write_json_once(path, payload)
    out(f"wrote {path} ({digest[:16]})")
    return payload


def main(argv=None, out=print):
    argv = list(sys.argv[1:] if argv is None else argv)
    fixture = None
    if argv[:1] == ["--v7-test"]:
        fixture, argv = argv[1], argv[2:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("import-selftest")
    src = i.add_mutually_exclusive_group(required=True)
    src.add_argument("--paste", help="a file holding the user's message, saved by a script (never retyped)")
    src.add_argument("--from-transcript", nargs="?", const="auto", metavar="JSONL",
                     help="read the user's message from this session's transcript (default: the newest one)")
    sub.add_parser("ready")
    s = sub.add_parser("snapshot")
    s.add_argument("--push", action="store_true")
    s.add_argument("--trailer", default="")
    w = sub.add_parser("write")
    w.add_argument("--snapshot", required=True)
    w.add_argument("--stop-rule", required=True)
    w.add_argument("--stop-rule-chosen", required=True)
    w.add_argument("--go-ahead", required=True)
    w.add_argument("--stop", required=True)
    w.add_argument("--routines", required=True)
    w.add_argument("--processes", required=True)
    d = sub.add_parser("withdraw")
    d.add_argument("--reason", required=True)
    d.add_argument("--no-laptop-phase-started", action="store_true")
    d.add_argument("--user-quote", help="the user's own words confirming that no v7 phase started on the laptop")
    d.add_argument("--user-utc", help="the UTC time of that message, e.g. 2026-10-14T09:00:00Z")
    d.add_argument("--ledger-listing", help="a file holding the user's pasted 'ls -1 .../stage2_v7/ledger'")
    a = ap.parse_args(argv)
    try:
        cfg = RN.config(fixture)
        if a.cmd == "import-selftest":
            if a.paste:
                import_selftest(cfg, a.paste, out)
            else:
                text, source = transcript_block(a.from_transcript)
                import_selftest(cfg, None, out, text=text, source=source)
        elif a.cmd == "ready":
            ready(cfg, out)
        elif a.cmd == "snapshot":
            snapshot(cfg, a.push, a.trailer, out)
        elif a.cmd == "write":
            write_handoff(cfg, a, out)
        else:
            withdraw(cfg, a.reason, a.no_laptop_phase_started, out, a.user_quote, a.user_utc, a.ledger_listing)
    except K.V7Refusal as error:
        out(f"refused: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
