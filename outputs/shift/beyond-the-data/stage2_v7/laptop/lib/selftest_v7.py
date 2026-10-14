"""Stage 2, version 7: the restore, the placement of the v6 code, and the laptop self-test (V7_SPEC.md C4, C6, C7,
C11). No BCA code runs here (P1b runs the room's program with W = 0) and no phase starts, so none of it needs an
approval.

  selftest_v7.py restore [--no-fetch]   (restore_v7.sh) fetch ONLY the run-files branch, check that the pinned
        snapshot commit is on it and its RUNFILES_SHA256SUMS has the pinned SHA-256, extract with
        'git archive <commit> runs | tar -x --skip-old-files', leaving out OUT/logs/ and the files of every phase that is
        not done in the snapshot (partial files, C6 step 4), check every extracted file against the sums, then place
        the v6 code. Before the handoff the snapshot is 901c877; after it, the one HANDOFF.json names. It never touches
        the main branch's index or tracked files (runs/ is ignored, .gitignore:6). Writes R7/restore_last.json.
  selftest_v7.py place                  the v6 code (17 files, byte for byte, from S2/code/) and fallback.py (from
        outputs/shift/beyond-the-data/scripts/fallback/) at the scratchpad paths launch.sh fixes, on every start because
        WSL empties /tmp; never overwrites a file that differs.
  selftest_v7.py full | short           the self-test (C11). Full: ST1-ST9 and the probes P1, P2, P3 (P1 and P2 make
        the seed 2-4 reference copies; each record is written once and never rerun). Short (before every run and after
        every restore): ST1, ST2 (versions), ST3, ST4, ST6, ST7, ST8 and a check that the P1 and P2 records and the
        reference files exist and, after the handoff, match the final prereg_v7.json.
  selftest_v7.py remake --probe ID      C7 'A damaged reference' (v7_probes.py remake, ledgered).
  selftest_v7.py r1                     C7's R1, information only, after the v7 import (run_v7.sh runs it once every
        phase is done): for seeds 2-4 and both worlds, the weighted band recomputed with v6's pinned band code from the
        gate's 'none' and, where re-made, from the laptop's copy, compared with the cloud's; R7/info/ (v7_probes.py r1,
        ledgered like a probe; a written record is not rerun). It routes nothing.
The last lines: 'READY', 'READY (waiting for handoff/approval)' or 'NOT READY (<what failed>)', then the probe results
and a block to paste into the cloud session (the nine probe records, byte for byte, compressed; cloud/make_handoff.py
import-selftest reads it).

It writes only to R7/selftest/ (reports, probe records, probe logs), R7/refs/ and R7/ledger/ (with --remake, also
R7/interrupted/refs/ and R7/refs_remake/; the files of an interrupted probe and a torn attempt record are moved, never
deleted, to R7/interrupted/); restore writes R7/restore_last.json and the run files under runs/. full, short, remake
and r1 hold R7/selftest/selftest.lock (one at a time) and first write the record of every probe attempt an earlier,
stopped self-test left without one. The 'restore' download is done by restore_v7.sh in the shell, where git can ask
for the user's sign-in (it then calls 'restore --no-fetch').
Probes and checks that touch the gate run as their own process, on one processor, in launch_v7.sh's exact eight-variable
environment ('--probe NAME', internal).
"""

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
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

RUNFILES_BRANCH = "claude/stage2-run-files"
PRE_HANDOFF = dict(commit="901c87766ab03bbd33819ff9e770302060501e79",
                   sums_sha256="157b0cdd018c5a1d1133aa941493480b1b16b13e3c0f27ad18011e14bdf96cc1")
SUMS_NAME = "RUNFILES_SHA256SUMS"
OUT_REL = "runs/shift_explore/stage2"
LOGS_REL = OUT_REL + "/logs"
FALLBACK_SRC_REL = "outputs/shift/beyond-the-data/scripts/fallback/fallback.py"
GPU_ENV_RECORD_REL = "outputs/shift/beyond-the-data/stage2_gpu/env_record.json"
GPU_PY = "/root/bcagpu/bin/python"
APPROVAL_COMMIT = K.V6["approval_commit"]
MIN_FREE_DISK_GB = 20
MIN_WSL_MEMORY_GIB = 40                 # README step 0 gives WSL 48 GB (it shows as about 47 GiB); below this, a warning
LANE_CANDIDATES = 16
BLOCK_BEGIN = "==== BEGIN V7 SELF-TEST BLOCK: paste every line from here down to the END line into the cloud session ===="
BLOCK_END = "==== END V7 SELF-TEST BLOCK ===="
CHUNK = 96


# --------------------------------------------------------------------------------------------- git

def git(cfg, *args, binary=False, check=False, timeout=600, input=None):
    env = dict(RN.BASE_ENV, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    try:
        done = subprocess.run(["git", "-C", str(cfg.root)] + list(args), capture_output=True, timeout=timeout, env=env,
                              input=input)
    except subprocess.TimeoutExpired:
        if check:
            raise K.V7Refusal(f"git {' '.join(args[:3])} did not finish within {timeout} s") from None
        msg = f"git {' '.join(args[:3])} did not finish within {timeout} s"
        return (124, msg.encode("utf8")) if binary else (124, msg)
    if check and done.returncode != 0:
        raise K.V7Refusal(f"git {' '.join(args[:3])} failed: {done.stderr.decode('utf8', 'replace').strip()[-300:]}")
    if binary:
        return done.returncode, done.stdout
    return done.returncode, done.stdout.decode("utf8", "replace").strip()


def cat_files(cfg, commit, paths):
    """{path: bytes or None} for 'commit:path', through one 'git cat-file --batch'."""
    paths = list(paths)
    if not paths:
        return {}
    req = "".join(f"{commit}:{p}\n" for p in paths).encode("utf8")
    code, data = git(cfg, "cat-file", "--batch", binary=True, input=req)
    if code != 0:
        raise K.V7Refusal("git cat-file --batch failed")
    out, i = {}, 0
    for p in paths:
        nl = data.index(b"\n", i)
        head = data[i:nl].decode("utf8", "replace").split()
        i = nl + 1
        if len(head) >= 2 and head[-1] == "missing":
            out[p] = None
            continue
        size = int(head[2])
        out[p] = data[i:i + size]
        i += size + 1
    return out


def parse_sums(data):
    sums = {}
    for line in data.decode("utf8").splitlines():
        if not line.strip():
            continue
        m = re.match(r"([0-9a-f]{64}) [ *](.+)$", line)
        if not m:
            raise K.V7Refusal(f"{SUMS_NAME} holds a line that is not '<sha256>  <path>': {line[:80]!r}")
        sums[m.group(2)] = m.group(1)
    return sums


# --------------------------------------------------------------------------------------------- the phase table on a snapshot

def all_phase_keys():
    """Every phase key of the v6 study (a superset of what v7 runs), for classifying a snapshot's files by the phase
    table: seeds 0-9, both worlds, both critics and versions, both groups."""
    keys = []

    def k(*args):
        keys.append(LG.phase_key(LG.parse_phase_args([str(a) for a in args])))

    for s in LG.EXPLORATORY + LG.CONFIRMATORY:
        if s in LG.CONFIRMATORY:
            k("--phase", "fresh_data", "--seed", s)
            k("--phase", "fresh_learners", "--seed", s)
            for w in RN.WORLDS:
                for p in ("detect_pairs", "detect_labels", "detect_standins"):
                    k("--phase", p, "--seed", s, "--world", w)
        for w in RN.WORLDS:
            for c in ("fixed", "host"):
                k("--phase", "detect_fit", "--seed", s, "--world", w, "--critic", c)
                k("--phase", "detect_fit_weighted", "--seed", s, "--world", w, "--critic", c)
                for v in ("uniform", "weighted"):
                    k("--phase", "detect_score", "--seed", s, "--world", w, "--critic", c, "--version", v)
            k("--phase", "deploy", "--seed", s, "--world", w)
            k("--phase", "host_train", "--seed", s, "--world", w)
            k("--phase", "host_eval", "--seed", s, "--world", w)
        for p in ("policy_train", "policy_eval", "policy_no_stall_train", "policy_no_stall_eval"):
            k("--phase", p, "--seed", s)
    for g in ("exploratory", "confirmatory"):
        for v in ("uniform", "weighted"):
            k("--phase", "detect_marks", "--group", g, "--version", v)
        k("--phase", "pooled", "--group", g)
    return keys


def _verified(blobs, path):
    data, side = blobs.get(path), blobs.get(path + ".sha256")
    if data is None or side is None:
        return None
    words = side.decode("utf8", "replace").split()
    if not words or words[0] != hashlib.sha256(data).hexdigest():
        return None
    try:
        return json.loads(data.decode("utf8"))
    except ValueError:
        return None


def snapshot_states(cfg, commit, sums):
    """C6 step 4 on a snapshot: {phase id: 'inherited' | 'partial' | 'not started'} and the sorted partial files
    (repository-relative), from the phase table and the snapshot's own bytes (git cat-file), never from folders."""
    keys = all_phase_keys()
    need = set()
    for key in keys:
        r = LG.row(key)
        for alt in r["final"]:
            need.update(alt)
        need.update(f["file"] for f in r["failed"])
    need = sorted(f"{OUT_REL}/{p}" for p in need)
    blobs = cat_files(cfg, commit, need + [p + ".sha256" for p in need])
    out_paths = [p[len(OUT_REL) + 1:] for p in sums if p.startswith(OUT_REL + "/")]
    states, partial = {}, set()
    for key in keys:
        r = LG.row(key)
        pid = LG.phase_id(key)
        done = False
        for f in r["failed"]:
            body = _verified(blobs, f"{OUT_REL}/{f['file']}")
            if body is not None and (f.get("field") is None or body.get(f["field"]) == f["equals"]):
                done = True
        for alt in r["final"]:
            if all(_verified(blobs, f"{OUT_REL}/{f}") is not None for f in alt):
                done = True
        files = [p for p in out_paths if LG.in_row(p, key)]
        if done:
            states[pid] = "inherited"
        elif files:
            states[pid] = "partial"
            partial.update(f"{OUT_REL}/{p}" for p in files)
        else:
            states[pid] = "not started"
    return states, sorted(partial)


# --------------------------------------------------------------------------------------------- restore (C6)

def restore_target(cfg):
    """(commit, sums SHA-256, source) to restore: HANDOFF.json's snapshot (verified, and the final prereg_v7.json's
    pin when one exists), else the pre-handoff snapshot 901c877."""
    import v7_verify as VF
    if cfg.handoff.exists():
        body = K.read_json_verified(cfg.handoff, "HANDOFF.json")
        if cfg.prereg.exists():
            prereg = VF.load_prereg_v7(cfg.prereg)
            if prereg.final and (prereg.payload.get("handoff") or {}).get("handoff_sha256") != K.file_sha256(cfg.handoff):
                raise K.V7Refusal("HANDOFF.json differs from the one the final prereg_v7.json pins")
        snap = body.get("snapshot") or {}
        if not snap.get("commit") or not snap.get("sums_sha256"):
            raise K.V7Refusal("HANDOFF.json names no snapshot commit and sums")
        return snap["commit"], snap["sums_sha256"], "HANDOFF.json", body
    pre = cfg.extra.get("pre_handoff") or PRE_HANDOFF
    return pre["commit"], pre["sums_sha256"], "pre-handoff snapshot", None


def fetch_runfiles(cfg, out=print, tries=5):
    """The fetch of the run-files branch from Python (prompts switched off). restore_v7.sh fetches in the shell
    instead, where git can ask for the user's sign-in and shows its progress, and then calls 'restore --no-fetch'."""
    branch = cfg.extra.get("runfiles_branch", RUNFILES_BRANCH)
    for n in range(tries):
        code, text = git(cfg, "fetch", "--no-tags", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                         timeout=3600)
        if code == 0:
            return True
        out(f"WARN fetch of {branch} failed (try {n + 1} of {tries}): {text[-200:]}")
        time.sleep(min(60, 2 ** (n + 1)))
    return False


def restore(cfg, fetch=True, out=print):
    """C6 'Restore'. Returns the record written to R7/restore_last.json; V7Refusal on any problem."""
    branch = cfg.extra.get("runfiles_branch", RUNFILES_BRANCH)
    commit, sums_pin, source, handoff = restore_target(cfg)
    if fetch and not fetch_runfiles(cfg, out):
        raise K.V7Refusal(f"restore: the run-files branch {branch} could not be downloaded (network, or git could not "
                          f"sign in: run restore_v7.sh, which lets git ask)")
    code, tip = git(cfg, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}")
    if code != 0:
        raise K.V7Refusal(f"restore: origin/{branch} is not here (run restore_v7.sh with network access)")
    if git(cfg, "cat-file", "-e", f"{commit}^{{commit}}")[0] != 0:
        raise K.V7Refusal(f"restore: the pinned commit {commit[:12]} is not on {branch}")
    if tip != commit and git(cfg, "merge-base", "--is-ancestor", commit, tip)[0] != 0:
        raise K.V7Refusal(f"restore: the pinned commit {commit[:12]} is not on {branch} (tip {tip[:12]})")
    code, sums_bytes = git(cfg, "show", f"{commit}:{SUMS_NAME}", binary=True)
    if code != 0 or K.sha256_bytes(sums_bytes) != sums_pin:
        raise K.V7Refusal(f"restore: {SUMS_NAME} of {commit[:12]} does not have the pinned SHA-256 {sums_pin[:16]}...")
    sums = parse_sums(sums_bytes)
    code, listing = git(cfg, "ls-tree", "-r", "--name-only", commit, "--", "runs")
    if set(listing.splitlines()) != set(sums):
        raise K.V7Refusal(f"restore: the files under runs/ in {commit[:12]} are not exactly those {SUMS_NAME} lists")
    states, partial = snapshot_states(cfg, commit, sums)
    if handoff is not None and sorted(handoff.get("partial_files") or []) != partial:
        raise K.V7Refusal("restore: the snapshot's partial files differ from HANDOFF.json's list")
    excluded = set(partial) | {p for p in sums if p.startswith(LOGS_REL + "/")}
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(cfg.r7) if cfg.r7.exists() else None,
                                     prefix=".restore-exclude-") as handle:
        handle.write(LOGS_REL + "\n" + "".join(p + "\n" for p in sorted(partial)))
        exfile = handle.name
    try:
        arch = subprocess.Popen(["git", "-C", str(cfg.root), "archive", "--format=tar", commit, "runs"],
                                stdout=subprocess.PIPE, env=dict(RN.BASE_ENV, GIT_OPTIONAL_LOCKS="0"))
        tar = subprocess.run(["tar", "-x", "--skip-old-files", "--anchored", "--no-wildcards", "-X", exfile, "-C",
                              str(cfg.root)], stdin=arch.stdout, capture_output=True)
        arch.stdout.close()
        if arch.wait() != 0 or tar.returncode != 0:
            raise K.V7Refusal(f"restore: extraction failed: {tar.stderr.decode('utf8', 'replace')[-300:]}")
    finally:
        os.unlink(exfile)
    bad, missing = verify_files(cfg, sums, excluded)
    if missing or bad:
        raise K.V7Refusal(f"restore: {len(bad)} run files differ from the pinned sums ({bad[:3]}) and {len(missing)} "
                          f"are missing ({missing[:3]}); nothing is overwritten: ask the cloud")
    leaked = [p for p in sorted(partial) if (cfg.root / p).exists()]
    placed = place(cfg)
    rec = dict(kind="stage2-v7-restore", ok=True, utc=K.utc_now(), boot_id=K.boot_id(), source=source, commit=commit,
               branch=branch, branch_tip=tip, sums_sha256=sums_pin, files_in_snapshot=len(sums),
               files_verified=len(sums) - len(excluded), logs_left_out=len([p for p in sums if p.startswith(LOGS_REL)]),
               partial_left_out=partial, partial_present_from_elsewhere=leaked, phase_states=states, placed=placed)
    write_status_json(cfg.r7 / "restore_last.json", rec)
    return rec


def verify_files(cfg, sums, excluded):
    bad, missing = [], []
    for path, sha in sorted(sums.items()):
        if path in excluded:
            continue
        p = cfg.root / path
        if not p.is_file():
            missing.append(path)
        elif K.file_sha256(p) != sha:
            bad.append(path)
    return bad, missing


def write_status_json(path, payload):
    """A status record that is rewritten (never a study record): JSON with its sidecar, replaced atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = K.canonical(payload).encode("utf8")
    for target, data in ((path, text), (Path(str(path) + ".sha256"),
                                        f"{K.sha256_bytes(text)}  {path.name}\n".encode("utf8"))):
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)


# --------------------------------------------------------------------------------------------- placement (C4)

def place(cfg):
    """The v6 code and fallback.py at the scratchpad paths, byte for byte; refuses when a file there differs."""
    root = cfg.root
    man = json.loads((root / K.V6["code_manifest"]).read_bytes().decode("utf8"))
    if K.file_sha256(root / K.V6["code_manifest"]) != K.V6["code_manifest_sha256"]:
        raise K.V7Refusal("place: v6's committed code manifest is not the pinned one")
    prereg = root / K.V6["prereg"]
    if K.file_sha256(prereg) != K.V6["prereg_sha256"]:
        raise K.V7Refusal("place: v6's committed prereg.json is not the pinned one")
    fb_pin = json.loads(prereg.read_text(encoding="utf8"))["pins"]["fallback_py"]["sha256"]
    todo = [(root / K.S2_REL / "code" / n, cfg.s_folder / n, h) for n, h in sorted(man["files"].items())]
    fb_target = Path(cfg.extra.get("fallback_target", str(K.FALLBACK_PY)))
    todo.append((root / FALLBACK_SRC_REL, fb_target, fb_pin))
    copied = unchanged = 0
    for src, dst, pin in todo:
        data = src.read_bytes()
        if K.sha256_bytes(data) != pin:
            raise K.V7Refusal(f"place: {src} is not the pinned file ({pin[:16]}...)")
        if dst.is_symlink():
            raise K.V7Refusal(f"place: {dst} is a symbolic link; refusing to write through it")
        if dst.exists():
            if dst.read_bytes() != data:
                raise K.V7Refusal(f"place: {dst} exists with different contents; it is never overwritten (delete "
                                  f"/tmp/claude-0/-home-user and run again)")
            unchanged += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dst.parent), prefix=".place-")
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, dst)
        copied += 1
    return dict(copied=copied, unchanged=unchanged, folder=str(cfg.s_folder), fallback=str(fb_target))


# --------------------------------------------------------------------------------------------- probes in the phase environment

def _quiet_path():
    for p in (str(LIB),):
        while p in sys.path:
            sys.path.remove(p)
    if str(K.ROOT) not in sys.path:
        sys.path.insert(0, str(K.ROOT))


def _with_constants():
    import v7_constants
    return v7_constants.apply(K.ROOT / K.CONSTANTS_REL, K.CONSTANTS_SHA256)


def _anchor_diffs(A):
    payload = json.loads((A.RUNS_DEFAULT / A.PREREG_NAME).read_text(encoding="utf8"))
    cfg = A.config_from_dict(payload["config"])
    now = json.loads(A.canonical(A.anchor_table(A.RUNS_DEFAULT, cfg)))
    rec = payload["anchors"]
    diff = 0
    for k in sorted(set(now) | set(rec)):
        for s in sorted(set(now.get(k, {})) | set(rec.get(k, {}))):
            for q in ("recording", "top", "recording_se", "top_se"):
                if now.get(k, {}).get(s, {}).get(q) != rec.get(k, {}).get(s, {}).get(q):
                    diff += 1
    return diff


def probe_gate(constants=True):
    """ST4 (with the constants) / ST5 (without): the gate's own load_prereg + guard, and the anchors."""
    import envfacts as E
    _quiet_path()
    out = dict(cpus=len(os.sched_getaffinity(0)), affinity=sorted(os.sched_getaffinity(0)),
               openblas=E.openblas_cores())
    if constants:
        facts = _with_constants()
        out["constants"] = dict(sha256=facts["sha256"], differences=facts["differences"])
    import experiments.shift.acceptance as A
    out["anchors_differing"] = _anchor_diffs(A)
    try:
        A.guard(A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)))
        out.update(gate_accepts=True, gate_error=None)
    except Exception as error:  # noqa: BLE001 - the gate's refusal is the result
        out.update(gate_accepts=False, gate_error=str(error)[:600])
    return out


def probe_constants():
    """ST3: A and K computed here against the pinned file (C1 step 2), nothing substituted."""
    _quiet_path()
    import v7_constants as VC
    consts = VC.load_constants(K.ROOT / K.CONSTANTS_REL, K.CONSTANTS_SHA256)
    diffs = VC.local_differences(consts)
    try:
        VC.check_tolerances(diffs)
        ok, why = True, None
    except K.V7Refusal as error:
        ok, why = False, str(error)
    return dict(ok=ok, why=why, summary=diffs["summary"],
                rows={k: dict(max_abs=v["max_abs"], differing=v["differing"]) for k, v in diffs["rows"].items()})


def probe_load_seed(seed):
    """ST6: the gate's load_seed for one seed in both worlds, with the constants (the room harness's SeedContext)."""
    _quiet_path()
    facts = _with_constants()
    t = time.perf_counter()
    H = K.load_v6_common().load_room_harness()
    ctx = H.SeedContext(int(seed), worlds=("stall", "no_stall"))
    return dict(seed=int(seed), worlds=sorted(ctx.sd), seconds=round(time.perf_counter() - t, 1),
                constants_sha256=facts["sha256"])


def probe_software():
    """ST2: versions, installed files, the dependency folder (required); the rest for information."""
    import envfacts as E
    _quiet_path()
    import experiments.shift as S
    import experiments.shift.acceptance as A
    gate = json.loads((K.ROOT / "runs" / "shift" / "preregistration.json").read_text(encoding="utf8"))
    ref = json.loads((K.ROOT / GPU_ENV_RECORD_REL).read_text(encoding="utf8"))
    soft = A.software_state()
    deps = S.dependency_record()
    lock = E.read_lock(LAPTOP / "requirements-lock.txt")
    wrong = [f"{n} {E.installed_version(n)} (want {v})" for n, v in lock if E.installed_version(n) != v]
    numeric, other = [], []
    for name, facts in sorted(ref["packages"].items()):
        if E.dist_content_sha256(name) != facts["installed_files"]:
            (numeric if name in E.NUMERIC_PACKAGES else other).append(name)
    try:
        from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__
        dispatch = [t for t in __cpu_dispatch__ if __cpu_features__.get(t)]
    except ImportError:
        dispatch = None
    import platform
    import jax
    dev = jax.devices()[0]
    return dict(python=E.python_facts(), python_reference=ref["python"]["full"], software=soft,
                software_equal=soft == gate["software"],
                software_differs={k: [soft.get(k), gate["software"].get(k)] for k in set(soft) | set(gate["software"])
                                  if soft.get(k) != gate["software"].get(k)},
                dependency_record=deps, dependency_equal=deps == gate["dependency_directory"],
                lock_packages=len(lock), wrong_version=wrong, files_differ_numeric=numeric, files_differ_other=other,
                cpu=E.cpu_info(), openblas=E.openblas_cores(), numpy_dispatch_active=dispatch,
                glibc=" ".join(platform.libc_ver()), kernel=platform.release(),
                xla_device=dict(platform=dev.platform, kind=getattr(dev, "device_kind", None),
                                version=getattr(getattr(dev, "client", None), "platform_version", None)))


PROBES = dict(gate=lambda a: probe_gate(True), gate_raw=lambda a: probe_gate(False), constants=lambda a: probe_constants(),
              load_seed=lambda a: probe_load_seed(a.seed), software=lambda a: probe_software())


def run_check_probe(cfg, name, cpu, seed=None, timeout=1800):
    """One probe as its own process: one processor, launch_v7.sh's eight variables, the working folder ROOT."""
    stubs = cfg.extra.get("stub_probes") if cfg.test else None
    if stubs is not None:
        res = stubs.get(name if seed is None else f"{name}:{seed}", stubs.get(name))
        return dict(ok=True, **res) if isinstance(res, dict) and "error" not in res else dict(ok=False, **(res or {}))
    cmd = ["env", "-i"] + [f"{k}={v}" for k, v in K.PHASE_ENV.items()] + \
        ["nice", "-n", "5", "taskset", "-c", str(cpu), K.PYTHON, str(Path(__file__)), "--probe", name]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    try:
        done = subprocess.run(cmd, cwd=str(K.ROOT), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return dict(ok=False, error=f"could not run: {error}")
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        return dict(ok=False, error=" | ".join(tail)[:600] or f"exit {done.returncode}")
    return dict(ok=True, **json.loads(lines[-1]))


def run_parallel(cfg, jobs):
    """jobs: [(label, name, cpu, seed)] run side by side (each on its own processor). {label: result}."""
    if cfg.test and cfg.extra.get("stub_probes") is not None:
        return {lab: run_check_probe(cfg, name, cpu, seed) for lab, name, cpu, seed in jobs}
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futs = {lab: pool.submit(run_check_probe, cfg, name, cpu, seed) for lab, name, cpu, seed in jobs}
        return {lab: f.result() for lab, f in futs.items()}


# --------------------------------------------------------------------------------------------- P1-P3 (v7_probes.py)

def probe_command(cfg, args, cpu, ledger):
    stub = cfg.extra.get("probe_script") if cfg.test else None
    script = stub or str(CODE_DIR / "v7_probes.py")
    head = [K.PYTHON] if not cfg.test else [sys.executable]
    env_prefix = ["env", "-i"] + [f"{k}={v}" for k, v in K.PHASE_ENV.items()]
    if cfg.test:
        env_prefix = ["env", "-i"] + [f"{k}={v}" for k, v in RN.BASE_ENV.items()]
        return env_prefix + head + [script] + list(args) + ["--cpu", str(cpu), "--ledger", str(ledger),
                                                             "--r7", str(cfg.r7)]
    return env_prefix + ["nice", "-n", "5", "taskset", "-c", str(cpu)] + head + [script] + list(args) + \
        ["--cpu", str(cpu), "--ledger", str(ledger)]


def ledger_cli(cfg, *args):
    cmd = ["env", "-i"] + [f"{k}={v}" for k, v in RN.BASE_ENV.items()] + \
        [sys.executable if cfg.test else K.PYTHON, str(CODE_DIR / "v7_ledger.py")] + [str(a) for a in args]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cfg.root if cfg.test else K.ROOT))
    if done.returncode != 0:
        raise K.V7Refusal(f"v7_ledger.py {args[0]} refused: {done.stderr.strip()[-300:]}")
    return done.stdout.strip()


def start_probe(cfg, probe_name, args, cpu):
    """Allocate the probe's ledger attempt (v7_ledger.py begin-probe) and start the probe process. Returns a handle."""
    ledger = ledger_cli(cfg, "begin-probe", "--r7", cfg.r7, "--probe", probe_name, "--cpu", cpu)
    logs = cfg.r7 / "selftest" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"probe_{probe_name}.log"
    handle = open(log, "a", encoding="utf8")
    handle.write(f"{K.utc_now()} start: {' '.join(args)} (cpu {cpu}, ledger {ledger})\n")
    handle.flush()
    proc = subprocess.Popen(probe_command(cfg, args, cpu, ledger), cwd=str(cfg.root if cfg.test else K.ROOT),
                            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    handle.close()
    return dict(name=probe_name, proc=proc, ledger=ledger, log=log, cpu=cpu)


def end_probe(cfg, h, record=None):
    """Wait for the probe and write its ledger record (v7_ledger.py end). record: the file whose presence (with a
    matching sidecar) means the probe finished; a callable gets the exit status and returns that file."""
    status = h["proc"].wait()
    if callable(record):
        record = record(status)
    args = ["end", "--ledger", h["ledger"], "--status", status, "--log", h["log"], "--r7", cfg.r7]
    if record is not None:
        args += ["--probe-record", record]
    ledger_cli(cfg, *args)
    return status


def records_p1p2(cfg):
    import v7_probes as PR
    out = {}
    for pid in PR.all_probe_ids():
        state, rec = K.read_record(PR.record_path(cfg.r7, pid))
        out[pid] = (state, rec)
    return out


def run_p1p2(cfg, cpus, out=print):
    """P1 (seeds 2-4 x both worlds) and P2 (seeds 2-4), side by side, each on its own processor. A probe whose record
    exists is never rerun (C7 'Probe records')."""
    import v7_probes as PR
    todo = []
    for s in PR.PROBE_SEEDS:
        for w in PR.WORLDS:
            todo.append((PR.probe_id("p1", s, w), ["p1", "--seed", str(s), "--world", w]))
        todo.append((PR.probe_id("p2", s), ["p2", "--seed", str(s)]))
    have = records_p1p2(cfg)
    run = [(pid, args) for pid, args in todo if have[pid][0] == "missing"]
    for pid, _ in todo:
        if have[pid][0] == "partial":
            raise K.V7Refusal(f"HALT: probe record {pid} differs from its sidecar (R7 was damaged); ask the cloud")
    if run:
        out(f"INFO probes: running {', '.join(p for p, _ in run)}, up to {len(cpus)} side by side, each on its own "
            f"processor (about 6-15 minutes each at the cloud's speed)")
    queue, running, results = list(run), {}, {}
    while queue or running:
        free = [c for c in cpus if c not in running]
        while queue and free:
            pid, args = queue.pop(0)
            cpu = free.pop(0)
            running[cpu] = (pid, start_probe(cfg, pid, args, cpu))
        for cpu, (pid, h) in list(running.items()):
            if h["proc"].poll() is not None:
                status = end_probe(cfg, h, PR.record_path(cfg.r7, pid))
                results[pid] = dict(status=status, log=str(h["log"]))
                del running[cpu]
        time.sleep(0.2)
    return results


def run_p3(cfg, cpus, out=print):
    """C3's repeatability test: three runs with the machine idle, three with the other lane processors busy, on two
    lane processors. Every run must give identical bytes."""
    import v7_probes as PR
    a, b = cpus[0], cpus[1] if len(cpus) > 1 else cpus[0]
    rows = []

    def one(cpu, load):
        h = start_probe(cfg, "p3_seed2", ["p3", "--seed", "2"], cpu)
        res = {}

        def record(status):
            text = Path(h["log"]).read_text(encoding="utf8", errors="replace")
            lines = [ln for ln in text.splitlines() if ln.startswith("{") and "tree_sha256" in ln]
            res.update(json.loads(lines[-1]) if (status == 0 and lines) else dict(error=f"exit {status}; log {h['log']}"))
            res.update(cpu=cpu, load=load)
            if "error" in res:
                return None
            path = cfg.r7 / "selftest" / "p3" / f"attempt_{Path(h['ledger']).name.split('.')[0]}.json"
            K.write_json_once(path, dict(res, kind="stage2-v7-probe-p3-run", utc=K.utc_now()))
            return path

        end_probe(cfg, h, record)
        rows.append(res)

    for cpu in (a, b, a):
        one(cpu, "idle")
    loaders = []
    if not cfg.test:
        code = "import numpy as np\na = np.random.default_rng(0).random((300, 300))\nwhile True:\n    a = (a @ a) / 300.0\n"
        for cpu in [c for c in cpus[2:]] + ([a] if len(cpus) < 3 else []):
            loaders.append(subprocess.Popen(["env", "-i"] + [f"{k}={v}" for k, v in K.PHASE_ENV.items()] +
                                            ["nice", "-n", "10", "taskset", "-c", str(cpu), K.PYTHON, "-c", code],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    try:
        for cpu in (b, a, b):
            one(cpu, f"busy ({len(loaders)} other lanes loaded)")
    finally:
        for p in loaders:
            p.kill()
            p.wait()
    ok = all("error" not in r for r in rows) and PR.p3_compare(rows)[0]
    return dict(ok=ok, runs=rows, distinct=len({(r.get("tree_sha256"), r.get("bytes_sha256")) for r in rows}))


# --------------------------------------------------------------------------------------------- the report

class Report:
    def __init__(self, out=print):
        self.lines, self.failed, self.data, self.out = [], [], {}, out

    def add(self, status, name, detail, tag=None):
        line = f"{status:<4} {name}: {detail}"
        self.lines.append(line)
        self.out(line)
        if status == "FAIL":
            self.failed.append(tag or name)


def git_checks(cfg, r):
    """ST8 (as GPU/lib/selftest_checks.py:203-231)."""
    problems = []
    if git(cfg, "rev-parse", "--is-shallow-repository")[1] != "false":
        problems.append("the clone is shallow (run: git -C /home/user/bca fetch --unshallow)")
    approval_commit = cfg.extra.get("approval_commit") or APPROVAL_COMMIT
    if git(cfg, "cat-file", "-e", approval_commit + "^{commit}")[0] != 0:
        problems.append(f"the approval commit {approval_commit[:7]} is not in this clone")
    code, tip = git(cfg, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{K.BRANCH}")
    if code != 0:
        problems.append(f"origin/{K.BRANCH} does not exist here (clone with --branch {K.BRANCH})")
    email = git(cfg, "config", "--get", "user.email")[1]
    if email and email.lower() in (K.APPROVER_EMAIL.lower(), "noreply@anthropic.com"):
        problems.append(f"git's user.email here is {email} (remove it: git config --global --unset user.email)")
    if git(cfg, "status", "--porcelain", "--untracked-files=no")[1]:
        problems.append("tracked files are changed in this clone (git status)")
    if problems:
        r.add("FAIL", "ST8 git", "; ".join(problems), "git: " + problems[0][:60])
    else:
        r.add("PASS", "ST8 git", f"full clone, approval commit {approval_commit[:7]} present, origin/{K.BRANCH} at "
                                 f"{tip[:7]}, git user.email {'unset' if not email else email}, no tracked file changed")


def st1(cfg, r, prereg):
    """ST1: the run files against the pinned sums, the v6 code placement, the v7 files against the v7 manifest, the
    v6 chain (read only)."""
    import v7_verify as VF
    problems = []
    try:
        rec = K.read_json_verified(cfg.r7 / "restore_last.json", "the last restore record")
        commit, pin, _, handoff = restore_target(cfg)
        if rec.get("commit") != commit or rec.get("sums_sha256") != pin:
            problems.append(f"the last restore was {str(rec.get('commit'))[:12]}, not the snapshot to use {commit[:12]} "
                            f"(rerun restore_v7.sh)")
        else:
            code, sums_bytes = git(cfg, "show", f"{commit}:{SUMS_NAME}", binary=True)
            if code != 0 or K.sha256_bytes(sums_bytes) != pin:
                problems.append(f"{SUMS_NAME} of {commit[:12]} is not readable here")
            else:
                sums = parse_sums(sums_bytes)
                excluded = set(rec.get("partial_left_out") or []) | {p for p in sums if p.startswith(LOGS_REL + "/")}
                bad, missing = verify_files(cfg, sums, excluded)
                if bad or missing:
                    problems.append(f"{len(bad)} run files differ from the pinned sums {bad[:2]}, {len(missing)} "
                                    f"missing {missing[:2]}")
    except K.V7Refusal as error:
        problems.append(str(error))
    try:
        place(cfg)
        if not cfg.test:
            VF.check_s_folder(cfg.s_folder)
    except K.V7Refusal as error:
        problems.append(str(error))
    try:
        VF.check_v7_code(prereg, cfg.v7_root)
    except K.V7Refusal as error:
        problems.append(str(error))
    if not cfg.test:
        try:
            VF.verify_v6_chain()
        except K.V7Refusal as error:
            problems.append(str(error))
    if problems:
        r.add("FAIL", "ST1 restore and fingerprints", "; ".join(problems), "restore: " + problems[0][:80])
    else:
        r.add("PASS", "ST1 restore and fingerprints", "the run files match the pinned sums; the v6 code and fallback.py "
                                                      "are in place; the v7 files match the v7 manifest; version 6's "
                                                      "approval chain verifies")


def st2(r, sw):
    if not sw.get("ok"):
        r.add("FAIL", "ST2 versions", f"the probe could not run: {sw.get('error')}", "versions: probe")
        return
    bad = []
    if sw["python"]["version"] != "3.10.20":
        bad.append(("versions", f"Python {sw['python']['version']} (want 3.10.20)"))
    if not sw["software_equal"]:
        bad.append(("versions", f"the gate's software block differs: {sw['software_differs']}"))
    if sw["wrong_version"]:
        bad.append(("versions", f"versions differ from the lock: {sw['wrong_version'][:6]}"))
    if sw["files_differ_numeric"]:
        bad.append(("installed files", f"installed files differ from the cloud's for {sw['files_differ_numeric']}"))
    if not sw["dependency_equal"]:
        bad.append(("dependency folder", f"the dependency folder {sw['dependency_record']} differs from the gate's"))
    for tag, text in bad:
        r.add("FAIL", "ST2 versions", text, tag if tag != "versions" else f"versions: {text[:60]}")
    if not bad:
        r.add("PASS", "ST2 versions", f"Python {sw['python']['version']}, the gate's eight libraries and flags as "
                                      f"recorded, {sw['lock_packages']} locked packages, installed files and the "
                                      f"dependency folder as recorded")
    cpu = sw.get("cpu") or {}
    r.add("INFO", "ST2 machine", f"{cpu.get('model')}; AVX-512 {'yes' if (cpu.get('flags') or {}).get('avx512f') else 'no'}"
                                 f"; {cpu.get('logical_processors')} logical processors; OpenBLAS kernels "
                                 f"{sw.get('openblas')}; numpy dispatch {sw.get('numpy_dispatch_active')}; "
                                 f"{sw.get('glibc')}; kernel {sw.get('kernel')}; XLA device {sw.get('xla_device')}")
    if sw.get("files_differ_other"):
        r.add("INFO", "ST2 other packages", f"installed files differ for packages that do not touch the numbers: "
                                            f"{sw['files_differ_other']}")


def st_records(cfg, r, prereg, full):
    """The probe records and reference files exist and, after the handoff, match the final prereg_v7.json."""
    import v7_probes as PR
    have = records_p1p2(cfg)
    missing = [p for p, (s, _) in have.items() if s == "missing"]
    damaged = [p for p, (s, _) in have.items() if s == "partial"]
    if damaged:
        r.add("FAIL", "probe records", f"HALT: probe record {damaged} differs from its sidecar", "HALT: probe record")
        return None
    if missing:
        if not full:
            r.add("FAIL", "probe records", f"missing {missing}: run the full self-test first (README step 8)",
                  "P1/P2: records missing")
        elif not any(t.startswith(("P1 ", "P2 ")) for t in r.failed):
            r.add("FAIL", "probe records", f"missing {missing} after the probes ran", "P1/P2: records missing")
        return None
    recs = {p: rec for p, (s, rec) in have.items()}
    problems = []
    for pid, rec in recs.items():
        files = [rec["p1a"]["reference_file"]] if rec["kind"] == "stage2-v7-probe-p1" else \
            list(rec["snapshots"].values())
        for f in files:
            if f.get("sha256") is None:
                continue
            p = cfg.r7 / f["path"]
            if not p.is_file() or K.file_sha256(p) != f["sha256"]:
                problems.append(f"HALT: reference {f['path']} differs from probe record {pid}")
    if prereg.final:
        pins = (prereg.payload.get("handoff") or {}).get("probe_records") or {}
        for pid in recs:
            row = pins.get(pid) or {}
            if row.get("sha256") != K.file_sha256(PR.record_path(cfg.r7, pid)):
                problems.append(f"HALT: probe record {pid} differs from its pin in the final prereg_v7.json")
            for rel, sha in (row.get("file_pins") or {}).items():
                p = cfg.r7 / rel
                if not p.is_file() or K.file_sha256(p) != sha:
                    problems.append(f"HALT: reference {rel} differs from its pin")
    if problems:
        r.add("FAIL", "probe records and reference files", "; ".join(problems[:4]), problems[0][:60])
    else:
        r.add("PASS", "probe records and reference files", f"all nine records verify; every reference file matches "
                                                           f"its record{' and the final prereg_v7.json' if prereg.final else ''}")
    return recs


def probe_lines(recs, q7):
    """C11: per seed and world, the result and the two forecast gaps."""
    import v7_probes as PR
    lines = []
    for pid in PR.all_probe_ids():
        rec = recs[pid]
        res = PR.result_line(rec)
        if rec["kind"] == "stage2-v7-probe-p1":
            f = rec["forecast"]
            word = {"bit-identical": "bit-identical", "gate's file kept (P1b reproduced it)":
                    f"gate's file kept (P1b): seed {rec['seed']} world {rec['world']}"}.get(
                res, f"re-made: seed {rec['seed']} world {rec['world']} (if not inherited)" if "re-made" in res else res)
            lines.append(f"  {pid}: {word}; F_ref {f['F_ref']} (against {f['F_ref_against']}); F_fresh {f['F_fresh']}")
        else:
            a = rec["actor"]
            word = "bit-identical" if res == "bit-identical" else (
                f"re-made: seed {rec['seed']} world stall host (if not inherited)" if "re-made" in res else res)
            lines.append(f"  {pid}: {word}; largest parameter difference {a.get('max_abs_diff')}; critic identical "
                         f"{(rec.get('critic') or {}).get('identical')}")
    fc = PR.forecast_all(recs, q7)
    lines.append(f"  check (b) forecast (tolerance {fc['tolerance']}, Q7 '{q7}'): {fc['line']}")
    return lines, fc


# --------------------------------------------------------------------------------------------- the paste block

def encode_block(summary, records):
    """Lines to paste: the summary and each record's exact bytes (gzip, base64, in short lines), with SHA-256."""
    lines = [BLOCK_BEGIN]
    items = [("_summary", K.canonical(summary).encode("utf8"))] + sorted(records.items())
    for name, data in items:
        b64 = base64.b64encode(gzip.compress(data, mtime=0)).decode("ascii")
        chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)] or [""]
        lines.append(f"V7RECORD {name} {K.sha256_bytes(data)} {len(data)} {len(chunks)}")
        lines += [f"V7DATA {name} {i} {c}" for i, c in enumerate(chunks)]
    lines.append(BLOCK_END)
    return lines


def decode_block(text):
    """(summary, {name: bytes}) from pasted text; V7Refusal on any damage (each record's SHA-256 is checked)."""
    heads, data = {}, {}
    inside = False
    for raw in text.splitlines():
        line = raw.strip()
        if BLOCK_BEGIN in line:
            inside, heads, data = True, {}, {}
            continue
        if BLOCK_END in line:
            inside = False
            continue
        if not inside:
            continue
        parts = line.split()
        if parts[:1] == ["V7RECORD"] and len(parts) == 5:
            heads[parts[1]] = (parts[2], int(parts[3]), int(parts[4]))
        elif parts[:1] == ["V7DATA"] and len(parts) in (3, 4):
            data.setdefault(parts[1], {})[int(parts[2])] = parts[3] if len(parts) == 4 else ""
    if "_summary" not in heads:
        raise K.V7Refusal("no complete V7 self-test block was found in the pasted text")
    out = {}
    for name, (sha, size, n) in heads.items():
        chunks = data.get(name, {})
        if sorted(chunks) != list(range(n)):
            raise K.V7Refusal(f"the pasted block is missing lines of {name}")
        try:
            raw = gzip.decompress(base64.b64decode("".join(chunks[i] for i in range(n)), validate=True))
        except (ValueError, OSError) as error:
            raise K.V7Refusal(f"the pasted block's {name} is damaged: {error}") from None
        if len(raw) != size or K.sha256_bytes(raw) != sha:
            raise K.V7Refusal(f"the pasted block's {name} does not match its SHA-256")
        out[name] = raw
    summary = json.loads(out.pop("_summary").decode("utf8"))
    return summary, out


# --------------------------------------------------------------------------------------------- the self-test lock

class SelfTestLock:
    """R7/selftest/selftest.lock: one self-test (or remake, or R1) at a time, so that the probe attempts of an earlier,
    killed self-test can be settled safely at the start (their records written, C5, C10)."""

    def __init__(self, cfg):
        self.path = cfg.r7 / "selftest" / "selftest.lock"
        self.handle = None

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "w")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise K.V7Refusal("another self-test is running (R7/selftest/selftest.lock); wait for it to end") from None
        return self

    def __exit__(self, *exc):
        self.handle.close()
        return False


def settle_probes(cfg, out=print):
    """The record of every probe attempt an earlier, killed self-test left without one (under the self-test lock)."""
    written = RN.settle_all(cfg, probes=True, phases=False)
    if written:
        out(f"INFO probe attempts: wrote {len(written)} missing attempt records of an earlier self-test that was "
            f"stopped: {', '.join(f'{p} {n} ({k})' for p, n, k in written[:6])}")
    return written


# --------------------------------------------------------------------------------------------- the self-test

def self_test(cfg, full, out=print):
    """Runs the checks; returns (verdict, report). Writes R7/selftest/report_<form>_<UTC>.json."""
    import v7_verify as VF
    import v7_probes as PR
    r = Report(out)
    cfg.r7.mkdir(parents=True, exist_ok=True)
    (cfg.r7 / "selftest").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    try:
        prereg = VF.load_prereg_v7(cfg.prereg)
    except K.V7Refusal as error:
        r.add("FAIL", "prereg_v7.json", str(error), "restore: prereg_v7.json")
        return finish(cfg, r, None, full, stamp, out)
    q7 = (prereg.payload.get("remake") or {}).get("q7_if_forecast_above_tolerance", "stop")
    cpus = list(cfg.cpus) if cfg.cpus else RN.pick_cpus(LANE_CANDIDATES)
    r.data["lane_processors"] = cpus
    st1(cfg, r, prereg)
    sw = run_check_probe(cfg, "software", cpus[0])
    r.data["software"] = sw
    st2(r, sw)
    c = run_check_probe(cfg, "constants", cpus[0])
    r.data["constants"] = c
    if c.get("ok") and c.get("summary") is not None and c.get("why") is None:
        s = c["summary"]
        r.add("PASS", "ST3 constants", f"A differs from the file by at most {s['max_abs_dA']:.3g} (tolerance 1e-14), "
                                       f"K by {s['max_abs_dK']:.3g} (1e-12); B bit-identical")
    else:
        r.add("FAIL", "ST3 constants", str(c.get("why") or c.get("error")), "constants: beyond tolerance")
    gates = run_parallel(cfg, [(cpu, "gate", cpu, None) for cpu in cpus])
    r.data["gate"] = gates
    bad = [cpu for cpu, g in gates.items() if not (g.get("ok") and g.get("gate_accepts"))]
    multi = [cpu for cpu, g in gates.items() if g.get("ok") and g.get("cpus") != 1]
    if bad:
        g = gates[bad[0]]
        r.add("FAIL", "ST4 gate with the constants", f"refused on processors {bad}: "
                                                     f"{str(g.get('gate_error') or g.get('error'))[:300]}",
              f"gate: processors {bad}")
    else:
        r.add("PASS", "ST4 gate with the constants", f"load_prereg and guard accept all 64 anchors exactly on every lane "
                                                     f"processor {cpus}")
    if multi:
        r.add("FAIL", "ST7 one processor", f"processors {multi} show more than one processor to a lane",
              f"one processor: {multi}")
    elif not bad:
        r.add("PASS", "ST7 one processor", "each lane sees exactly one processor")
    if full:
        raw = run_check_probe(cfg, "gate_raw", cpus[0])
        r.data["gate_raw"] = raw
        r.add("INFO", "ST5 gate without the constants", f"{raw.get('anchors_differing')} of 64 anchors differ on this "
                                                        f"laptop; gate accepts {raw.get('gate_accepts')}")
    seeds = run_parallel(cfg, [(s, "load_seed", cpus[i % len(cpus)], s) for i, s in enumerate(LG.EXPLORATORY)])
    r.data["load_seed"] = seeds
    bad = [s for s, v in seeds.items() if not v.get("ok")]
    if bad:
        r.add("FAIL", "ST6 load_seed", f"seeds {bad}: {str(seeds[bad[0]].get('error'))[:300]}", f"load_seed: seeds {bad}")
    else:
        r.add("PASS", "ST6 load_seed", "every log array of seeds 0-4 in both worlds matches the gate's SHA-256 (with "
                                       "the constants)")
    recs = None
    if full:
        try:
            res = run_p1p2(cfg, cpus, out)
            r.data["p1p2_runs"] = res
            failed = [p for p, v in res.items() if K.read_record(PR.record_path(cfg.r7, p))[0] != "ok"]
            if failed:
                r.add("FAIL", "P1/P2", f"{failed} did not finish (logs {[res[p]['log'] for p in failed]}); run the "
                                       f"self-test again (a written record is never rerun)",
                      f"{'P1' if failed[0].startswith('p1') else 'P2'} {failed[0]} did not finish")
        except K.V7Refusal as error:
            r.add("FAIL", "P1/P2", str(error), str(error)[:60])
    recs = st_records(cfg, r, prereg, full)
    fc = None
    if recs is not None:
        lines, fc = probe_lines(recs, q7)
        r.data["probe_lines"] = lines
        r.data["forecast"] = fc
        if fc["not_ready"]:
            r.add("FAIL", "P1b check (b) forecast", fc["line"], "P1b: check (b) forecast above 1e-5")
        else:
            r.add("PASS" if not fc["above"] else "INFO", "P1b check (b) forecast", fc["line"])
    if full:
        p3 = run_p3(cfg, cpus, out)
        r.data["p3"] = p3
        if p3["ok"]:
            secs = [x.get("seconds") for x in p3["runs"] if x.get("seconds") is not None]
            r.add("PASS", "P3 repeatability", f"{len(p3['runs'])} runs (idle and under load, processors "
                                              f"{sorted({x['cpu'] for x in p3['runs']})}) give identical bytes")
        else:
            r.add("FAIL", "P3 repeatability", f"{p3['distinct']} different results in {len(p3['runs'])} runs: "
                                              f"{[x.get('error') for x in p3['runs'] if x.get('error')][:2]}",
                  "P3: results depend on lane or load")
    git_checks(cfg, r)
    resources(cfg, r, full, cpus[0])
    return finish(cfg, r, prereg, full, stamp, out, recs, fc, q7)


def resources(cfg, r, full, cpu):
    total, avail = RN.meminfo(cfg.meminfo)
    disk = shutil.disk_usage(str(cfg.root))
    free = disk.free / 1e9
    low_mem = total / RN.GIB < MIN_WSL_MEMORY_GIB
    r.add("INFO" if free >= MIN_FREE_DISK_GB and not low_mem else "WARN", "ST9 resources",
          f"WSL memory {total / RN.GIB:.1f} GiB (available {avail / RN.GIB:.1f})"
          + (f": the 48 GB setting of README step 0 is not in effect (run 'wsl --shutdown' in PowerShell, then open "
             f"Ubuntu again); fewer phases will run at once" if low_mem else "")
          + f"; free disk {free:.1f} GB (at least {MIN_FREE_DISK_GB} GB wanted)")
    p3 = r.data.get("p3") or {}
    secs = [x.get("seconds") for x in p3.get("runs", []) if x.get("seconds") is not None]
    if secs:
        r.add("INFO", "ST9 speed", f"one 1,000-step actor-only training run took {min(secs):.1f}-{max(secs):.1f} s "
                                   f"(P3; idle and under load); information only, no lane-hour estimate is made")
    if full and not cfg.test and os.path.exists(GPU_PY):
        try:
            done = subprocess.run(["taskset", "-c", str(cpu), GPU_PY, "-I", str(LIB / "gpu_probe.py")],
                                  capture_output=True, text=True, timeout=1800, env=dict(RN.BASE_ENV))
            lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
            res = json.loads(lines[-1]) if lines else dict(error=(done.stderr or "").strip()[-300:])
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            res = dict(error=str(error))
        r.data["gpu"] = res
        r.add("INFO", "ST9 graphics card", str(res)[:300] + " (information only; every phase uses the processor)")


def ready_after_handoff(cfg, prereg):
    """True when the final prereg_v7.json, HANDOFF.json and the v7 approval chain all verify (then 'READY')."""
    import v7_verify as VF
    if prereg is None or not prereg.final:
        return False
    try:
        VF.check_handoff(prereg, cfg.v7_root)
        if cfg.check_chain:
            VF.verify_v7_chain(prereg, cfg.v7_root)
    except K.V7Refusal:
        return False
    return True


def finish(cfg, r, prereg, full, stamp, out, recs=None, fc=None, q7="stop"):
    if r.failed:
        verdict = f"NOT READY ({r.failed[0]}" + (f"; and {len(r.failed) - 1} more" if len(r.failed) > 1 else "") + ")"
    elif ready_after_handoff(cfg, prereg):
        verdict = "READY"
    else:
        verdict = "READY (waiting for handoff/approval)"
    out(verdict)
    for line in r.data.get("probe_lines") or []:
        out(line)
    summary = dict(kind="stage2-v7-selftest-summary", form="full" if full else "short", utc=K.utc_now(),
                   verdict=verdict, failed=r.failed, q7=q7, lane_processors=r.data.get("lane_processors"),
                   git_head=git(cfg, "rev-parse", "HEAD")[1], machine=(r.data.get("software") or {}).get("cpu"),
                   openblas=(r.data.get("software") or {}).get("openblas"),
                   p3=dict(ok=(r.data.get("p3") or {}).get("ok"), distinct=(r.data.get("p3") or {}).get("distinct")),
                   forecast=fc, probe_lines=r.data.get("probe_lines"),
                   prereg_v7_sha256=None if prereg is None else prereg.sha256)
    records = {}
    if recs is not None:
        import v7_probes as PR
        for pid in recs:
            records[pid] = PR.record_path(cfg.r7, pid).read_bytes()
    for line in encode_block(summary, records):
        out(line)
    report = dict(summary, lines=r.lines, data=r.data)
    try:
        write_status_json(cfg.r7 / "selftest" / f"report_{summary['form']}_{stamp}.json", report)
    except OSError:
        pass
    return verdict, report


def remake_cmd(cfg, probe, out=print):
    import v7_probes as PR
    if probe not in PR.all_probe_ids():
        out(f"refused: {probe} is not a probe with reference files ({PR.all_probe_ids()})")
        return 1
    cpu = (list(cfg.cpus) if cfg.cpus else RN.pick_cpus(1))[0]
    before = set((cfg.r7 / "refs_remake").glob("*.json")) if (cfg.r7 / "refs_remake").is_dir() else set()
    h = start_probe(cfg, f"remake_{probe}", ["remake", "--probe", probe], cpu)

    def record(status):
        new = sorted(set((cfg.r7 / "refs_remake").glob("*.json")) - before) if (cfg.r7 / "refs_remake").is_dir() else []
        return new[-1] if (status == 0 and new) else None

    status = end_probe(cfg, h, record)
    out(Path(h["log"]).read_text(encoding="utf8", errors="replace").strip().splitlines()[-1:][0] if status == 0
        else f"HALT: the remake of {probe} did not reproduce the pinned SHA-256 (exit {status}); the run stays halted; "
             f"log {h['log']}")
    return 0 if status == 0 else 1


def r1_cmd(cfg, out=print):
    """C7 R1 (information only): six processes side by side, each ledgered; existing records are kept."""
    todo = [(s, w) for s in K.REMAKE_SEEDS for w in ("stall", "no_stall")
            if not (cfg.r7 / "info" / f"r1_seed{s}_{w}.json").exists()]
    cpus = list(cfg.cpus) if cfg.cpus else RN.pick_cpus(max(1, len(todo)))
    queue, running, bad = list(todo), {}, []
    while queue or running:
        free = [c for c in cpus if c not in running]
        while queue and free:
            s, w = queue.pop(0)
            cpu = free.pop(0)
            running[cpu] = ((s, w), start_probe(cfg, f"r1_seed{s}_{w}", ["r1", "--seed", str(s), "--world", w], cpu))
        for cpu, ((s, w), h) in list(running.items()):
            if h["proc"].poll() is not None:
                status = end_probe(cfg, h, cfg.r7 / "info" / f"r1_seed{s}_{w}.json")
                if status != 0:
                    bad.append(f"r1_seed{s}_{w} (exit {status}; log {h['log']})")
                del running[cpu]
        time.sleep(0.2)
    out(f"R1 (information only): {len(todo) - len(bad)} written, {6 - len(todo)} already there" +
        (f"; did not finish: {bad}" if bad else ""))
    return 0 if not bad else 1


def main(argv=None, out=print):
    argv = list(sys.argv[1:] if argv is None else argv)
    fixture = None
    if argv[:1] == ["--v7-test"]:
        fixture, argv = argv[1], argv[2:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("restore", "place", "full", "short", "remake", "r1"))
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--probe")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    if a.cmd is None and a.probe in PROBES:                    # internal: one check in the phase environment
        print(json.dumps(PROBES[a.probe](a), sort_keys=True, default=str))
        return 0
    try:
        cfg = RN.config(fixture)
    except K.V7Refusal as error:
        out(f"refused: {error}")
        return 1
    try:
        if a.cmd in ("full", "short", "remake", "r1"):
            with SelfTestLock(cfg):
                settle_probes(cfg, out)
                if a.cmd == "remake":
                    return remake_cmd(cfg, a.probe, out)
                if a.cmd == "r1":
                    return r1_cmd(cfg, out)
                verdict, _ = self_test(cfg, full=a.cmd == "full", out=out)
                return 0 if verdict.startswith("READY") else 1
        if a.cmd == "restore":
            rec = restore(cfg, fetch=not a.no_fetch, out=out)
            out(f"PASS restore: {rec['files_verified']} run files of {rec['commit'][:12]} ({rec['source']}) match the "
                f"pinned sums; {len(rec['partial_left_out'])} partial files and the cloud's logs left out; v6 code "
                f"placed ({rec['placed']['copied']} copied, {rec['placed']['unchanged']} already there)")
            return 0
        if a.cmd == "place":
            place(cfg)
            return 0
        out("refused: give one of restore, place, full, short, remake, r1")
        return 1
    except K.V7Refusal as error:
        text = str(error)
        out(f"NOT READY ({text})" if text.startswith("restore") else f"refused: {text}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
