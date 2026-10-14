"""CLOUD ONLY. Stage 2, version 7: the handoff from the cloud to the laptop (V7_SPEC.md C6) and the way back.

Order at the user's go-ahead (README step 9; C6 step 1 first: the watchdog Routine and every other Routine that could
start a Stage 2 phase deleted and confirmed with the Routines list, the driver's sh processes stopped, the running
phases stopped at once unless the user chose 'let them finish', no run_stage2.py left):

  make_handoff.py import-selftest --paste FILE
      Reads the block the laptop's full self-test printed (pasted into FILE). It refuses unless the block's summary is
      a FULL self-test whose verdict starts with READY and whose repeatability test P3 passed (C3, C11: a P3 failure
      ends v7; a NOT READY self-test is fixed on the laptop and the full self-test run again, which reruns no written
      probe record). It checks every probe record's SHA-256 and that it binds to the committed draft prereg_v7.json
      (its re-make and forecast block, the constants file, the gate's pre-registration, v6's code manifest, the probe
      script), and writes each record, byte for byte, once, to V7/handoff/probes/selftest/<probe>.json (+ .sha256), and
      an import record V7/handoff/probes/selftest_import_<UTC>.json (+ .sha256): the summary, its SHA-256 and the
      cloud's own time of the import. A record already there must be identical (never replaced).
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
      times of the nine P1/P2 records and the cloud's time of the first self-test import (the choice must come before
      both, or be the default 'stop at once'); the full self-test this handoff rests on (its import record's SHA-256,
      verdict and P3 result; it must be READY with P3 passed); the go-ahead's time; each Routine's id, name and final
      state (FILE: a JSON list of {id, name, final_state, starts_phases}; every Routine that can start a phase must be
      'deleted'); the processes stopped (FILE: a JSON list), the check that no run_stage2.py and no explore_policy.sh
      process is left, and the driver log's last lines; for every seed and phase, 'inherited', 'partial' or 'not
      started' by the phase table on the snapshot's own bytes (C6 step 4), and the partial files; the cloud's
      process-hours from its logs (C2); the draft prereg_v7.json it goes with (write_prereg_v7.py final refuses
      another draft).
      Then: write_prereg_v7.py final --handoff V7/handoff/HANDOFF.json --records V7/handoff/probes
  make_handoff.py withdraw --reason TEXT --no-laptop-phase-started --user-quote TEXT --user-utc UTC
                           [--ledger-listing FILE]
      V7/handoff/WITHDRAWN.json (+ .sha256), once (C6 step 6): v7 is not approved, or the laptop refused before its
      first phase started. It records the user's own confirmation, quoted, with the UTC time of their message, that no
      v7 phase has started on the laptop; with --ledger-listing (the output of
      'ls -1 /home/user/bca/runs/shift_explore/stage2_v7/ledger', pasted by the user) it is stored and refused when it
      names any folder other than a probe's. The v7 runner then refuses; the cloud may resume v6 with
      explore_policy.sh, at most 2 processes, with a new watchdog only if the user asks.

'--v7-test <fixture>' before the command is for test_v7_laptop.py only.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
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


def cloud_processes():
    """Read only (/proc): every process whose command line names run_stage2.py (the cloud's v6 phases) or the driver
    explore_policy.sh (which starts the next v6 phase as soon as one ends, C6 step 1)."""
    out = dict(run_stage2_py=[], explore_policy_sh=[])
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
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


def require_cloud_stopped(cfg):
    """C6 step 1: no v6 phase and no driver left. Returns the check, for HANDOFF.json."""
    if cfg.test:
        return dict(checked_utc=K.utc_now(), run_stage2_py=[], explore_policy_sh=[], note="test fixture: not checked")
    procs = cloud_processes()
    if procs["run_stage2_py"] or procs["explore_policy_sh"]:
        raise K.V7Refusal(f"Stage 2 processes still run (run_stage2.py: {procs['run_stage2_py']}; the driver "
                          f"explore_policy.sh: {procs['explore_policy_sh']}); stop the driver first, then the phases "
                          f"(C6 step 1)")
    return dict(procs, checked_utc=K.utc_now())


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
    inp = rec.get("inputs") or {}
    want = dict(remake_block_sha256=K.block_sha256(prereg.payload["remake"]),
                constants_sha256=prereg.payload["constants"]["sha256"], gate_prereg_sha256=K.GATE_PREREG_SHA256,
                v6_code_manifest_sha256=K.V6["code_manifest_sha256"],
                probe_script_sha256=(prereg.payload["code"]["files"] or {}).get("code/v7_probes.py"))
    return sorted(k for k, v in want.items() if inp.get(k) != v)


def import_selftest(cfg, paste, out=print):
    import v7_probes as PR
    import v7_verify as VF
    summary, records = ST.decode_block(Path(paste).read_text(encoding="utf8", errors="replace"))
    problems = selftest_problems(summary)
    if problems:
        raise K.V7Refusal("the pasted block is not from a READY full self-test with P3 passed (" + "; ".join(problems) +
                          "); nothing was stored. If P3 failed, v7 ends as specified; otherwise fix what the self-test "
                          "names and run the full self-test again (it reruns no written probe record)")
    prereg = VF.load_prereg_v7(cfg.prereg)
    pins = ((prereg.payload.get("handoff") or {}).get("probe_records") or {}) if prereg.final else {}
    d = probes_dir(cfg) / "selftest"
    new, same = [], []
    for pid, data in sorted(records.items()):
        if pid not in PR.all_probe_ids():
            raise K.V7Refusal(f"the block holds an unknown record {pid!r}")
        rec = json.loads(data.decode("utf8"))
        if rec.get("probe") != pid:
            raise K.V7Refusal(f"the record named {pid} is the record of {rec.get('probe')}")
        bad = bind_check(prereg, rec)
        if bad:
            raise K.V7Refusal(f"probe record {pid} does not bind to the committed prereg_v7.json ({bad})")
        sha = K.sha256_bytes(data)
        if pins and (pins.get(pid) or {}).get("sha256") != sha:
            raise K.V7Refusal(f"HALT: probe record {pid} differs from its pin in the final prereg_v7.json")
        p = d / f"{pid}.json"
        if p.exists():
            if p.read_bytes() != data:
                raise K.V7Refusal(f"{p} exists with other bytes; a probe record is never replaced")
            same.append(pid)
            continue
        K.write_once(p, data)
        K.write_once(Path(str(p) + ".sha256"), f"{sha}  {p.name}\n".encode("utf8"))
        new.append(pid)
    imp = dict(kind="stage2-v7-selftest-import", imported_utc=K.utc_now(), summary=summary,
               summary_sha256=K.block_sha256(summary), records=sorted(records), new=new, same=same,
               rule="the cloud's own time of the import; the stop rule must be chosen before it (C6)")
    K.write_json_once(probes_dir(cfg) / f"selftest_import_{K.utc_stamp()}.json", imp)
    out(f"self-test ({summary.get('form')}): {summary.get('verdict')}; P3 passed")
    for line in summary.get("probe_lines") or []:
        out(line)
    out(f"probe records: {len(new)} new, {len(same)} already here and identical; "
        f"{len(PR.all_probe_ids()) - len(new) - len(same)} not in this block")
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


def write_handoff(cfg, a, out=print):
    import v7_probes as PR
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
    imports = []
    for p in sorted(probes_dir(cfg).glob("selftest_import_*.json")):
        imports.append((p, K.read_json_verified(p, "a self-test import record")))
    if not imports:
        raise K.V7Refusal("no full self-test was imported (make_handoff.py import-selftest)")
    last_path, last = imports[-1]
    problems = selftest_problems(last.get("summary"))
    if problems:
        raise K.V7Refusal(f"the last imported self-test cannot carry a handoff: {'; '.join(problems)}")
    first_import = min((parse_utc(i.get("imported_utc"), "an import time"), i.get("imported_utc")) for _, i in imports)
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
    probes = {}
    for pid in PR.all_probe_ids():
        p = probes_dir(cfg) / "selftest" / f"{pid}.json"
        rec = K.read_json_verified(p, f"probe record {pid} (import the self-test block first)")
        if bind_check(prereg, rec):
            raise K.V7Refusal(f"probe record {pid} does not bind to the committed draft")
        probes[pid] = dict(sha256=K.file_sha256(p), written_utc=rec.get("written_utc"))
    first_dt, first = min((parse_utc(v["written_utc"], f"the write time of probe record {pid}"), v["written_utc"])
                          for pid, v in probes.items())
    if chosen is not None and not (chosen < first_dt and chosen < first_import[0]):
        raise K.V7Refusal(f"the stop rule was chosen at {a.stop_rule_chosen}, not before the first P1/P2 record "
                          f"({first}, the laptop's clock) and the first self-test import ({first_import[1]}, the "
                          f"cloud's clock); it cannot be changed after any record exists (C6)")
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
    i.add_argument("--paste", required=True)
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
            import_selftest(cfg, a.paste, out)
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
