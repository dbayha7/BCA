"""The self-test's checks (selftest.sh runs restore.sh first, then this). Read only: it runs no Stage 2 phase, fits
nothing, trains nothing and writes only its own report under runs/shift_explore/stage2_offcloud/.

Each check that touches the pinned code runs in its own process, pinned to one processor with taskset and given
exactly the environment launch.sh gives a phase (JAX_PLATFORMS=cpu, OMP_NUM_THREADS=1, PYTHONPATH=/home/user/bca,
nothing else), so it sees what a phase will see. The first such check is the gate's own check of its
pre-registration, which recomputes the anchor returns with numpy and compares them exactly: on some processors the
last bits differ, and then no phase can run here at all.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
LIB = Path(__file__).resolve().parent
PKG = LIB.parent
ROOT = Path("/home/user/bca")
LINK_PY = "/tmp/claude-0/bcavenv/bin/python"
GPU_PY = "/root/bcagpu/bin/python"
OFFCLOUD = ROOT / "runs" / "shift_explore" / "stage2_offcloud"
BRANCH = "claude/main-bca-connection-x4o46s"
APPROVAL_COMMIT = "7dab7d545c49e0310176fc4f00b6a41492b30ea6"
AGENT_EMAILS = ("noreply@anthropic.com",)
GATE_FLAGS = ("JAX_ENABLE_X64", "JAX_PLATFORMS", "XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION", "OMP_NUM_THREADS")
BASE_ENV = dict(PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", HOME="/root", LANG="C.UTF-8",
                LC_ALL="C.UTF-8")
PHASE_ENV = dict(BASE_ENV, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu",
                 OMP_NUM_THREADS="1")


def code_dir():
    man = json.loads((ROOT / "runs" / "shift_explore" / "stage2" / "code_manifest.json").read_text(encoding="utf8"))
    return man["source_dir"]


# ------------------------------------------------------------------------------------------------ probes (one process
# each, started by run_probe under taskset with PHASE_ENV)

def _quiet_path():
    here = str(LIB)
    while here in sys.path:
        sys.path.remove(here)
    sys.path.insert(0, str(ROOT))


def probe_anchors():
    sys.path.insert(0, str(LIB))
    import envfacts as E
    _quiet_path()
    out = dict(cpus=len(os.sched_getaffinity(0)), openblas=E.openblas_cores(),
               env={k: os.environ.get(k) for k in GATE_FLAGS + ("OPENBLAS_CORETYPE",)})
    import experiments.shift.acceptance as A
    payload = json.loads((A.RUNS_DEFAULT / A.PREREG_NAME).read_text(encoding="utf8"))
    cfg = A.config_from_dict(payload["config"])
    now = json.loads(A.canonical(A.anchor_table(A.RUNS_DEFAULT, cfg)))
    rec = payload["anchors"]
    diff = []
    for k in sorted(set(now) | set(rec)):
        for s in sorted(set(now.get(k, {})) | set(rec.get(k, {}))):
            for q in ("recording", "top", "recording_se", "top_se"):
                a, b = now.get(k, {}).get(s, {}).get(q), rec.get(k, {}).get(s, {}).get(q)
                if a != b:
                    diff.append([k, s, q, a, b])
    out.update(anchors_equal=not diff, anchors_differing=len(diff), first_difference=diff[:1])
    try:
        prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
        A.guard(prereg)
        out.update(gate_accepts=True, gate_error=None)
    except Exception as error:  # noqa: BLE001 - the gate's refusal is the result
        out.update(gate_accepts=False, gate_error=str(error)[:600])
    return out


def probe_software():
    sys.path.insert(0, str(LIB))
    import envfacts as E
    _quiet_path()
    import experiments.shift as S
    import experiments.shift.acceptance as A
    gate = json.loads((ROOT / "runs" / "shift" / "preregistration.json").read_text(encoding="utf8"))
    ref = json.loads((PKG / "env_record.json").read_text(encoding="utf8"))
    soft = A.software_state()
    deps = S.dependency_record()
    lock = E.read_lock(PKG / "requirements-lock.txt")
    wrong_version = [f"{n} {E.installed_version(n)} (want {v})" for n, v in lock if E.installed_version(n) != v]
    numeric, other = [], []
    for name, facts in sorted(ref["packages"].items()):
        got = E.dist_content_sha256(name)
        if got != facts["installed_files"]:
            (numeric if name in E.NUMERIC_PACKAGES else other).append(name)
    import importlib.metadata as md
    prefix = os.path.realpath(sys.prefix)
    in_env = {E.canonical_name(d.metadata["Name"]) for d in md.distributions()
              if os.path.realpath(str(d.locate_file(""))).startswith(prefix + os.sep)}
    extra = sorted(in_env - {n for n, _ in lock})
    return dict(python=E.python_facts(), python_reference=ref["python"]["full"],
                software=soft, software_equal=soft == gate["software"],
                software_differs={k: [soft.get(k), gate["software"].get(k)] for k in set(soft) | set(gate["software"])
                                  if soft.get(k) != gate["software"].get(k)},
                dependency_record=deps, dependency_equal=deps == gate["dependency_directory"],
                lock_packages=len(lock), wrong_version=wrong_version, not_in_lock=extra,
                files_differ_numeric=numeric, files_differ_other=other)


def probe_check_env():
    _quiet_path()
    import warnings
    import experiments.shift.thruster_chain as TC
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    out = {}
    for eid in TC.ENV_IDS:
        env = gym.make(eid, render_mode="ansi")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env.unwrapped, skip_render_check=False)
        out[eid] = [str(w.message)[:200] for w in caught]
        env.close()
    return dict(ids=len(out), warnings={k: v for k, v in out.items() if v})


def probe_pinned():
    _quiet_path()
    sys.path.insert(0, code_dir())
    import prereg_payload as PP
    import run_stage2 as RS
    import stage2_common as C
    out = {}
    prereg = RS.final_prereg()
    out["prereg_sha256"] = prereg.sha256
    approval = C.require_approval()                       # the approval, the import record and git, read only
    out["approval"] = dict(mode=approval.mode, file_sha256=approval.file_sha256)
    PP.check_pins(prereg)                                 # room, fallback, stand-ins, detection script
    out["pins"] = "match"
    C.load_room_harness()
    C.load_fallback()
    RS.detect_module(prereg)
    pin = prereg.payload["pins"]
    C.verify_component(C.STANDINS_OUT, pin["standins_manifest_sha256"])
    C.load_pinned_module("stage2_standins", pin["standins_py"]["path"], pin["standins_py"]["sha256"])
    out["imports"] = "room harness, fallback diagnostic, detection script and stand-ins loaded from their pinned bytes"
    out["room_condition"] = RS.room_condition()
    late = (C.PATHS.out / "best_standin.json").exists()
    out["late_files_present"] = late
    if late:
        out["best_standin"] = C.read_json(C.PATHS.out / "best_standin.json")["best"]
        out["version_for_seeds_5_9"] = RS.chosen_version()
    return out


def probe_onecpu():
    return dict(cpus=len(os.sched_getaffinity(0)), pid_affinity=sorted(os.sched_getaffinity(0)))


PROBES = dict(anchors=probe_anchors, software=probe_software, check_env=probe_check_env, pinned=probe_pinned,
              onecpu=probe_onecpu)


def run_probe(name, cpu, extra_env=None, timeout=1800):
    env = dict(PHASE_ENV, **(extra_env or {}))
    cmd = ["taskset", "-c", str(cpu), LINK_PY, str(Path(__file__)), "--probe", name]
    try:
        done = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return dict(ok=False, error=f"could not run: {error}")
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        return dict(ok=False, error=" | ".join(tail)[:600] or f"exit {done.returncode}")
    return dict(ok=True, **json.loads(lines[-1]))


# ------------------------------------------------------------------------------------------------ report

class Report:
    def __init__(self):
        self.lines, self.failed, self.data = [], [], {}

    def add(self, status, name, detail):
        line = f"{status:<4} {name}: {detail}"
        self.lines.append(line)
        print(line, flush=True)
        if status == "FAIL":
            self.failed.append(name)


def git(*args):
    try:
        done = subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True, timeout=60,
                              env=dict(BASE_ENV, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0"))
        return done.returncode, done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, ""


def git_checks(r):
    problems = []
    code, out = git("rev-parse", "--is-shallow-repository")
    if out != "false":
        problems.append("the clone is shallow (run: git -C /home/user/bca fetch --unshallow)")
    code, _ = git("cat-file", "-e", APPROVAL_COMMIT + "^{commit}")
    if code != 0:
        problems.append(f"your approval commit {APPROVAL_COMMIT[:7]} is not in this clone")
    code, tip = git("rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{BRANCH}")
    if code != 0:
        problems.append(f"origin/{BRANCH} does not exist here (clone with --branch {BRANCH}, then git fetch)")
    code, email = git("config", "--get", "user.email")
    approval = json.loads((ROOT / "runs/shift_explore/stage2/approval.json").read_text(encoding="utf8"))
    refused = {m.lower() for m in AGENT_EMAILS}
    for role in ("author", "committer"):
        m = re.search(r"<([^<>]+)>", approval["git"][role])
        if m:
            refused.add(m.group(1).lower())
    if email and email.lower() in refused:
        problems.append(f"git's user.email here is {email}; the Stage 2 code refuses to run under the identity that "
                        f"made the approval commit (remove it: git config --global --unset user.email)")
    code, dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        problems.append("tracked files are changed in this clone (git status); the pinned files must be unchanged")
    if problems:
        r.add("FAIL", "git", "; ".join(problems))
    else:
        r.add("PASS", "git", f"full clone, approval commit {APPROVAL_COMMIT[:7]} present, origin/{BRANCH} at "
                             f"{tip[:7]}, git user.email {'unset' if not email else email}, no tracked file changed")


def anchors_check(r, cpu, cpuinfo):
    res = run_probe("anchors", cpu)
    r.data["anchors"] = res
    model = cpuinfo.get("model")
    if not res.get("ok"):
        r.add("FAIL", "anchors (first check)", f"the probe could not run: {res.get('error')}")
        return
    core = res["openblas"]
    if res["gate_accepts"]:
        r.add("PASS", "anchors (first check)", f"the gate accepts its pre-registration on this processor ({model}; "
                                               f"matrix kernels {core.get('numpy')}/{core.get('scipy')}; anchors equal)")
        return
    why = res["gate_error"] or ""
    r.add("FAIL", "anchors (first check)", f"the gate refuses on this processor ({model}; matrix kernels "
                                           f"{core.get('numpy')}/{core.get('scipy')}): {why[:300]}; "
                                           f"{res['anchors_differing']} anchor values differ in their last bits")
    if res["anchors_differing"] and cpuinfo["flags"].get("avx512f") and core.get("numpy") != "SkylakeX":
        again = run_probe("anchors", cpu, dict(OPENBLAS_CORETYPE="SkylakeX"))
        r.data["anchors_skylakex"] = again
        r.add("INFO", "anchors with OPENBLAS_CORETYPE=SkylakeX",
              f"anchors equal {again.get('anchors_equal')}, gate accepts {again.get('gate_accepts')} "
              f"(this processor has AVX-512; ask the cloud before using this setting)")
    elif res["anchors_differing"] and not cpuinfo["flags"].get("avx512f"):
        r.add("INFO", "anchors", "this processor has no AVX-512, so the matrix library cannot reproduce the cloud's "
                                 "last bits; the pinned code cannot run here without a code change (a new "
                                 "pre-registration version and a new approval)")


def gpu_checks(r, cpu):
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi")
                                         else None)
    if smi is None:
        r.add("INFO", "graphics card", "nvidia-smi not found inside WSL (update the NVIDIA Windows driver)")
    else:
        try:
            out = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            out = f"nvidia-smi failed: {error}"
        r.add("INFO", "graphics card", out or "nvidia-smi printed nothing")
    if not os.path.exists(GPU_PY):
        r.add("INFO", "graphics card timing", "not measured (optional: README, 'Optional: measure the graphics card')")
        return
    try:
        done = subprocess.run(["taskset", "-c", str(cpu), GPU_PY, "-I", str(LIB / "gpu_probe.py")],
                              capture_output=True, text=True, timeout=1800, env=dict(BASE_ENV))
        lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
        res = json.loads(lines[-1]) if lines else dict(error=(done.stderr or "").strip()[-300:])
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        res = dict(error=str(error))
    r.data["gpu"] = res
    if res.get("error"):
        r.add("INFO", "graphics card timing", f"failed: {res['error']}")
    else:
        r.add("INFO", "graphics card timing", f"JAX sees {res['devices']}; actor step: {res['gpu_steps_per_s']} per "
                                              f"second on the graphics card, {res['cpu_steps_per_s']} on one processor "
                                              f"(x{res['ratio']}); information only, the pre-registration requires the "
                                              f"processor")


def resource_info(r):
    mem = None
    try:
        for line in open("/proc/meminfo", encoding="utf8"):
            if line.startswith("MemTotal"):
                mem = round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    disk = shutil.disk_usage(str(ROOT))
    tmp_rule = None
    for p in ("/etc/tmpfiles.d/tmp.conf", "/usr/lib/tmpfiles.d/tmp.conf"):
        if os.path.exists(p):
            tmp_rule = " ".join(ln.strip() for ln in open(p, encoding="utf8") if ln.strip() and
                                not ln.startswith("#"))
            break
    r.add("INFO", "resources", f"memory {mem} GB, free disk {round(disk.free / 1e9, 1)} GB, /tmp rule "
                               f"'{tmp_rule}' (restore.sh puts /tmp back every run)")


def preflight(cpus, extra_env=None):
    """The checks run_confirmatory.sh repeats before any phase starts: git, and on every processor a lane will use,
    the gate's own check (anchors, versions, flags) and the one-processor rule; then the approval and the pins through
    the pinned code. Returns True when all pass (each line is printed)."""
    r = Report()
    git_checks(r)
    for cpu in cpus:
        res = run_probe("anchors", cpu, extra_env)
        if res.get("ok") and res.get("gate_accepts"):
            r.add("PASS", f"gate check on processor {cpu}", f"accepted (matrix kernels {res['openblas']})")
        else:
            r.add("FAIL", f"gate check on processor {cpu}", str(res.get("gate_error") or res.get("error"))[:400])
        one = run_probe("onecpu", cpu, extra_env)
        if not (one.get("ok") and one.get("cpus") == 1):
            r.add("FAIL", f"one processor ({cpu})", str(one)[:200])
    pin = run_probe("pinned", cpus[0], extra_env)
    if pin.get("ok") and pin.get("late_files_present"):
        r.add("PASS", "approval and pins", f"approved ({pin['approval']['mode']}), pins match, best stand-in "
                                           f"{pin['best_standin']}, version {pin['version_for_seeds_5_9']}")
    else:
        r.add("FAIL", "approval and pins", str(pin.get("error") or "the late files are not in place")[:400])
    return not r.failed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=tuple(PROBES))
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.probe:
        print(json.dumps(PROBES[a.probe](), sort_keys=True, default=str))
        return 0
    sys.path.insert(0, str(LIB))
    import envfacts as E
    r = Report()
    cpuinfo = E.cpu_info()
    r.add("INFO", "processor", f"{cpuinfo['model']}; AVX-512 {'yes' if cpuinfo['flags']['avx512f'] else 'no'}, "
                               f"AVX2 {'yes' if cpuinfo['flags']['avx2'] else 'no'}; {cpuinfo['logical_processors']} "
                               f"logical processors visible to WSL")
    restore = json.loads((OFFCLOUD / "restore_last.json").read_text(encoding="utf8")) \
        if (OFFCLOUD / "restore_last.json").exists() else dict(ok=False, error="restore.sh did not run")
    if restore.get("ok"):
        r.add("PASS", "pinned hashes", f"payload ({restore['payload_files']} files) in place; " +
              "; ".join(restore["chain"][:2]) + f"; and {len(restore['chain']) - 2} more groups of pins")
    else:
        r.add("FAIL", "pinned hashes", restore.get("error"))
    if not os.path.exists(LINK_PY):
        r.add("FAIL", "environment", f"{LINK_PY} does not exist (README, step 6)")
        return finish(r, a)
    anchors_check(r, a.cpu, cpuinfo)
    sw = run_probe("software", a.cpu)
    r.data["software"] = sw
    if not sw.get("ok"):
        r.add("FAIL", "versions", f"the probe could not run: {sw.get('error')}")
    else:
        bad = []
        if sw["python"]["version"] != "3.10.20":
            bad.append(f"Python {sw['python']['version']} (want 3.10.20)")
        if not sw["software_equal"]:
            bad.append(f"gate software block differs: {sw['software_differs']}")
        if not sw["dependency_equal"]:
            bad.append(f"dependency folder {sw['dependency_record']} differs from the recorded one")
        if sw["wrong_version"]:
            bad.append(f"versions differ from the lock: {sw['wrong_version'][:6]}")
        if sw["files_differ_numeric"]:
            bad.append(f"installed files differ from the cloud's for {sw['files_differ_numeric']}")
        if bad:
            r.add("FAIL", "versions", "; ".join(bad))
        else:
            r.add("PASS", "versions", f"Python {sw['python']['version']}, jax {sw['software']['packages']['jax']}, "
                                      f"jaxlib, flax, optax, numpy, scipy, h5py, gymnasium as recorded; "
                                      f"{sw['lock_packages']} locked packages; dependency folder "
                                      f"{sw['dependency_record']['sha256'][:12]} as recorded; flags "
                                      f"{sw['software']['flags']}; backend {sw['software']['jax_backend']}")
        notes = []
        if sw["python"]["full"] != sw["python_reference"]:
            notes.append(f"Python build '{sw['python']['full']}' differs from the cloud's '{sw['python_reference']}'")
        if sw["files_differ_other"]:
            notes.append(f"installed files differ for packages that do not touch the numbers: "
                         f"{sw['files_differ_other']}")
        if sw["not_in_lock"]:
            notes.append(f"packages installed that the lock does not list: {sw['not_in_lock']}")
        if notes:
            r.add("WARN", "versions", "; ".join(notes))
    ce = run_probe("check_env", a.cpu)
    if ce.get("ok") and not ce["warnings"] and ce["ids"] == 8:
        r.add("PASS", "ThrusterChain check_env", "all 8 environment ids pass with no warnings")
    else:
        r.add("FAIL", "ThrusterChain check_env", str(ce.get("warnings") or ce.get("error"))[:400])
    pin = run_probe("pinned", a.cpu)
    r.data["pinned"] = pin
    if pin.get("ok"):
        r.add("PASS", "approval and pins (pinned code, read only)",
              f"prereg {pin['prereg_sha256'][:16]}... approved ({pin['approval']['mode']}); import record on "
              f"origin; pins match; {pin['imports']}; room condition passed {pin['room_condition'].get('passed')}")
    else:
        r.add("FAIL", "approval and pins (pinned code, read only)", pin.get("error"))
    git_checks(r)
    one = run_probe("onecpu", a.cpu)
    if one.get("ok") and one["cpus"] == 1:
        r.add("PASS", "one processor per phase", f"taskset -c {a.cpu} gives each phase exactly 1 processor "
                                                 f"(the detection script's rule)")
    else:
        r.add("FAIL", "one processor per phase", str(one))
    if restore.get("ok"):
        if restore.get("late_status") == "present":
            late = restore["late"]
            r.add("PASS", "late files", f"present: best stand-in {late['best_standin']}, version for seeds 5-9 "
                                        f"{late['chosen_version']}")
        else:
            r.add("WAIT", "late files", "not here yet: the cloud commits best_standin.json and the exploratory "
                                        "detection marks after seeds 0-4; then git pull")
    gpu_checks(r, a.cpu)
    resource_info(r)
    return finish(r, a, restore)


def finish(r, a, restore=None):
    if r.failed:
        verdict = f"NOT READY ({len(r.failed)} failed: {', '.join(r.failed)})"
    elif restore and restore.get("late_status") != "present":
        verdict = "READY, WAITING FOR THE CLOUD'S LATE FILES"
    else:
        verdict = "READY"
    line = f"RESULT: {verdict}"
    print(line)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(lines=r.lines + [line], data=r.data, failed=r.failed,
                                                verdict=verdict), indent=1, sort_keys=True, default=str),
                                encoding="utf8")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
