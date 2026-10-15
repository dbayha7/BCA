"""Stage 2, version 7: the checks every v7 phase makes before the hand-over (V7_SPEC C3, C4, C6, C10). Read-only.

  check_environment()   C4: exactly launch_v7.sh's eight variables (PATH, HOME=/root, LANG=C.UTF-8, LC_ALL=C.UTF-8,
                        PYTHONPATH=/home/user/bca, PYTHONDONTWRITEBYTECODE=1, JAX_PLATFORMS=cpu, OMP_NUM_THREADS=1) and
                        nothing else (no OPENBLAS_*, NPY_*, XLA_*, CUDA_*, LD_*, BCA_SHIFT_DEPS, PWD, OLDPWD), the
                        working folder /home/user/bca and the pinned Python path.
  check_one_processor() C3: os.sched_getaffinity(0) holds exactly one processor (the one launch_v7.sh was given).
  load_prereg_v7(), check_v7_code()   V7/prereg_v7.json against its sidecar; the v7 code manifest it pins; every v7
                        file against the manifest.
  check_v6_pins(), check_s_folder()   version 6's committed files and the scratchpad copy of its code (the folder
                        launch.sh runs from) against v6's pinned code manifest.
  verify_v6_chain()     version 6's approval route, by v6's own unchanged require_approval (read-only git).
  verify_v7_chain()     version 7's approval route (C10): the user's committed approval_v7_<prefix>.json, exactly one
                        commit on origin's branch that added it, its author and committer neither the cloud session's
                        identity nor its configured user.email (v6's rule, S2C/stage2_common.py:83, :483-521, :524-549,
                        called from v6's module, not copied) AND both equal to v6's approver (the new positive check),
                        the import record on origin's branch. Git does not authenticate authors (S2/prereg.json:47).
  check_handoff()       C6: V7/handoff/HANDOFF.json present with the SHA-256 the final prereg_v7.json pins, no
                        WITHDRAWN.json, and the phase not inherited.
  machine_facts()       information for the ledger: processor model and AVX-512 flags, OpenBLAS kernels, numpy's
                        dispatch, glibc, the (WSL) kernel release.
Every check raises v7_common.V7Refusal; the wrapper turns it into one ledger line and exit 97.
"""

import ctypes
import glob
import json
import os
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as K  # noqa: E402


# --------------------------------------------------------------------------------------------- C3, C4

def check_environment(environ=None, cwd=None, executable=None, expected=None):
    env = dict(os.environ if environ is None else environ)
    expected = dict(K.PHASE_ENV if expected is None else expected)
    extra = sorted(set(env) - set(expected))
    if extra:
        raise K.V7Refusal(f"the environment holds variables outside launch_v7.sh's eight: {extra}")
    wrong = sorted(k for k, v in expected.items() if env.get(k) != v)
    if wrong:
        raise K.V7Refusal(f"the environment differs from launch_v7.sh's in {wrong}")
    cwd = Path(os.getcwd() if cwd is None else cwd)
    if cwd != K.ROOT:
        raise K.V7Refusal(f"the working folder is {cwd}, not {K.ROOT}")
    executable = sys.executable if executable is None else executable
    if executable != K.PYTHON:
        raise K.V7Refusal(f"the Python is {executable}, not the pinned {K.PYTHON}")
    return dict(variables=sorted(env), cwd=str(cwd), python=executable)


def check_one_processor(cpu=None, affinity=None):
    aff = sorted(os.sched_getaffinity(0) if affinity is None else affinity)
    if len(aff) != 1:
        raise K.V7Refusal(f"this process may use {len(aff)} processors {aff}; every phase runs on exactly one (C3)")
    if cpu is not None and aff != [int(cpu)]:
        raise K.V7Refusal(f"this process runs on processor {aff[0]}, not the processor {cpu} it was given")
    return aff[0]


# --------------------------------------------------------------------------------------------- the v7 files

class PreregV7:
    def __init__(self, path, sha256, payload):
        self.path, self.sha256, self.payload = Path(path), sha256, payload

    @property
    def prefix(self):
        return K.prefix_of(self.sha256)

    @property
    def final(self):
        return self.payload.get("approvable") is True and not self.payload.get("missing")


def load_prereg_v7(path=None):
    path = Path(path or (K.V7 / "prereg_v7.json"))
    if not path.is_file():
        raise K.V7Refusal(f"no v7 pre-registration at {path}")
    raw = path.read_bytes()
    digest = K.sha256_bytes(raw)
    if K.sidecar_digest(path) != digest:
        raise K.V7Refusal(f"{path} differs from its SHA-256 sidecar (the file was changed)")
    payload = json.loads(raw.decode("utf8"))
    if payload.get("kind") != K.PREREG_KIND or payload.get("label") != K.LABEL:
        raise K.V7Refusal(f"{path} is not the Stage 2 v7 pre-registration")
    return PreregV7(path, digest, payload)


def require_final(prereg):
    if not prereg.final:
        raise K.V7Refusal(f"prereg_v7.json is a draft (missing: {prereg.payload.get('missing')}); no phase runs "
                          f"before the final, approved version")
    if not isinstance(prereg.payload.get("handoff"), dict):
        raise K.V7Refusal("the final prereg_v7.json holds no handoff block")
    return prereg


def check_v7_code(prereg, v7_root=None):
    """The v7 code manifest prereg_v7.json pins, and every file it lists, unchanged."""
    v7_root = Path(v7_root or K.V7)
    pin = (prereg.payload.get("code") or {}).get("code_manifest_v7_sha256")
    path = v7_root / "code_manifest_v7.json"
    if not path.is_file():
        raise K.V7Refusal(f"{path} is missing")
    raw = path.read_bytes()
    digest = K.sha256_bytes(raw)
    if digest != pin or K.sidecar_digest(path) != digest:
        raise K.V7Refusal("the v7 code manifest is not the one prereg_v7.json pins (or differs from its sidecar)")
    man = json.loads(raw.decode("utf8"))
    bad = [rel for rel, h in man.get("files", {}).items()
           if not (v7_root / rel).is_file() or K.file_sha256(v7_root / rel) != h]
    if bad:
        raise K.V7Refusal(f"v7 files differ from the pinned manifest: {bad}")
    return man


def check_v6_pins(root=None):
    """Version 6's committed pre-registration, manifest, approval file and import record have the SHA-256 v7 pins."""
    root = Path(root or K.ROOT)
    bad = []
    for name in ("prereg", "code_manifest", "approval_file", "import_record"):
        p = root / K.V6[name]
        want = K.V6[name + "_sha256"]
        if not p.is_file() or K.file_sha256(p) != want or K.sidecar_digest(p) != want:
            bad.append(K.V6[name])
    if bad:
        raise K.V7Refusal(f"version 6's committed files differ from v7's pins: {bad}")
    return dict(prereg_sha256=K.V6["prereg_sha256"], code_manifest_sha256=K.V6["code_manifest_sha256"])


def check_s_folder(s_folder=None):
    """The folder launch.sh runs from holds every v6 file of v6's code manifest, byte for byte (C4: placed on every
    start because WSL empties /tmp)."""
    s_folder = Path(s_folder or K.S_FOLDER)
    files = K.v6_manifest()["files"]
    bad = [n for n, h in sorted(files.items()) if not (s_folder / n).is_file() or K.file_sha256(s_folder / n) != h]
    if bad:
        raise K.V7Refusal(f"{s_folder} does not hold v6's pinned code: {bad}")
    return len(files)


# --------------------------------------------------------------------------------------------- approval chains

def verify_v6_chain(v6c=None):
    """v6's own unchanged require_approval (approval.json, the user's commit, the import record, every pinned script),
    read-only. V7Refusal when it refuses."""
    v6c = v6c or K.load_v6_common()
    try:
        approval = v6c.require_approval(v6c.PATHS)
    except (v6c.ApprovalError, SystemExit) as error:
        raise K.V7Refusal(f"version 6's approval does not verify: {error}") from None
    if approval.prereg_sha256 != K.V6["prereg_sha256"] or approval.file_sha256 != K.V6["runs_side_approval_sha256"]:
        raise K.V7Refusal("version 6's approval is not the one v7 pins")
    return dict(prereg_sha256=approval.prereg_sha256, approval_sha256=approval.file_sha256, mode=approval.mode)


def approval_rel(prefix):
    return K.V7_REL + "/" + K.APPROVAL_NAME.format(prefix=prefix)


def import_rel(prefix):
    return K.V7_REL + "/" + K.IMPORT_NAME.format(prefix=prefix)


class ChainProblem(K.V7Refusal):
    pass


def check_local_approval_v7(raw, prereg):
    """The user's committed approval file (approve_local_v7.py's output) approves THIS prereg_v7.json."""
    try:
        body = json.loads(bytes(raw).decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise ChainProblem("the v7 approval file is not UTF-8 JSON") from None
    code = prereg.payload.get("code") or {}
    checks = [
        (isinstance(body, dict) and body.get("kind") == K.LOCAL_KIND, f"it is not a v7 local approval ({K.LOCAL_KIND})"),
        (body.get("mode") == K.LOCAL_MODE, "its mode is not local-terminal"),
        (body.get("label") == K.LABEL, "it names another study"),
        (body.get("prereg_path") == K.V7_REL + "/prereg_v7.json", "it names another pre-registration file"),
        (body.get("prereg_sha256") == prereg.sha256, "it approves another version of prereg_v7.json"),
        (body.get("typed") == prereg.prefix, "the typed characters are not exactly the first 16 of the fingerprint"),
        (body.get("stdin_isatty") is True, "it was not typed at a terminal"),
        (bool(body.get("approved_at_utc")), "it holds no time"),
        (body.get("v6_prereg_sha256") == K.V6["prereg_sha256"], "it names another version 6"),
        (body.get("code_manifest_sha256") == code.get("code_manifest_v7_sha256"), "it was made against another manifest"),
        (body.get("approve_local_sha256") == (code.get("files") or {}).get("approve_local_v7.py"),
         "it was made with another approve_local_v7.py"),
        (prereg.sha256 in str(body.get("statement", "")), "its statement does not name this fingerprint"),
    ]
    for ok, why in checks:
        if not ok:
            raise ChainProblem(f"the v7 approval file does not approve this pre-registration: {why}")
    return body


def _email(who):
    import re
    m = re.fullmatch(r"\s*(\S.*?)\s*<([^<>\s]+@[^<>\s]+)>\s*", str(who or ""))
    return m.group(2).lower() if m else None


def check_approver(record, approver=K.APPROVER_EMAIL):
    """The new, positive check (C10): author and committer are both v6's approver."""
    for role in ("author", "committer"):
        if _email(record.get(role)) != approver.lower():
            raise ChainProblem(f"the commit's {role} is {record.get(role)!r}, not v6's approver <{approver}>")
    return record


def verify_v7_chain(prereg, v7_root=None, v6c=None):
    """C10 step 4: the v7 approval chain, read-only (git queries through v6's read-only git())."""
    v7_root = Path(v7_root or K.V7)
    v6c = v6c or K.load_v6_common()
    require_final(prereg)
    section = prereg.payload.get("approval") or {}
    branch = section.get("branch")
    prefix = prereg.prefix
    a_rel, i_rel = approval_rel(prefix), import_rel(prefix)
    local = v7_root / K.APPROVAL_NAME.format(prefix=prefix)
    if not local.is_file():
        raise ChainProblem(f"no v7 approval of this fingerprint at {local} (the user approves at their own terminal)")
    raw = local.read_bytes()
    local_sha = K.sha256_bytes(raw)
    if K.sidecar_digest(local) != local_sha:
        raise ChainProblem(f"{local} differs from its sidecar")
    check_local_approval_v7(raw, prereg)
    imp = v7_root / K.IMPORT_NAME.format(prefix=prefix)
    if not imp.is_file():
        raise ChainProblem(f"no import record at {imp} (written by 'approve_v7.py import')")
    raw_imp = imp.read_bytes()
    if K.sidecar_digest(imp) != K.sha256_bytes(raw_imp):
        raise ChainProblem(f"{imp} differs from its sidecar")
    body = json.loads(raw_imp.decode("utf8"))
    want = dict(kind=K.IMPORT_KIND, label=K.LABEL, prereg_sha256=prereg.sha256, approval_file=a_rel,
                approval_file_sha256=local_sha, branch=branch)
    bad = sorted(k for k, v in want.items() if body.get(k) != v)
    if bad:
        raise ChainProblem(f"the import record does not match this approval ({', '.join(bad)})")
    record = dict(commit=body.get("commit"), author=body.get("author"), committer=body.get("committer"),
                  committed_sha256=local_sha)
    try:
        v6c.check_commit_record(record, local_sha, v6c.agent_emails(v6c.git_session_email()))
        check_approver(record)
        v6c.check_commit_in_git(record, a_rel, branch)
        adders = v6c.git_adding_commits(v6c.remote_ref(branch), a_rel)
        if adders != [record["commit"]]:
            raise ChainProblem(f"exactly one commit on origin's branch must add {a_rel}: git found {adders}")
        if v6c.git_blob(v6c.remote_ref(branch), i_rel) != raw_imp:
            raise ChainProblem(f"the import record {i_rel} is not on origin's branch {branch} as written")
    except v6c.LocalApprovalProblem as problem:
        raise ChainProblem(str(problem)) from None
    return dict(prereg_sha256=prereg.sha256, approval_file=a_rel, approval_sha256=local_sha, import_record=i_rel,
                import_sha256=K.sha256_bytes(raw_imp), commit=record["commit"], author=record["author"])


# --------------------------------------------------------------------------------------------- C6

def check_handoff(prereg, v7_root=None, pid=None):
    v7_root = Path(v7_root or K.V7)
    if (v7_root / "handoff" / "WITHDRAWN.json").exists():
        raise K.V7Refusal("v7 was withdrawn (V7/handoff/WITHDRAWN.json exists); no v7 phase runs (C6)")
    pin = (prereg.payload.get("handoff") or {}).get("handoff_sha256")
    path = v7_root / "handoff" / "HANDOFF.json"
    if not path.is_file():
        raise K.V7Refusal("refused: handoff missing (V7/handoff/HANDOFF.json); steps 9-11 are not complete")
    raw = path.read_bytes()
    digest = K.sha256_bytes(raw)
    if digest != pin or K.sidecar_digest(path) != digest:
        raise K.V7Refusal("refused: handoff differs from the one the final prereg_v7.json pins")
    body = json.loads(raw.decode("utf8"))
    if pid is not None and (body.get("phases") or {}).get(pid) == "inherited":
        raise K.V7Refusal(f"{pid} is inherited from the cloud (HANDOFF.json); it never runs on the laptop (C6)")
    return body


# --------------------------------------------------------------------------------------------- machine facts

def _openblas():
    out = {}
    try:
        import numpy
        base = os.path.dirname(os.path.dirname(numpy.__file__))
    except ImportError:
        return out
    for pkg, pattern, names in (("numpy", "numpy.libs/libscipy_openblas64_*.so",
                                 ("scipy_openblas_get_corename64_", "openblas_get_corename64_")),
                                ("scipy", "scipy.libs/libscipy_openblas-*.so",
                                 ("scipy_openblas_get_corename", "openblas_get_corename"))):
        libs = sorted(glob.glob(os.path.join(base, pattern)))
        if not libs:
            out[pkg] = None
            continue
        try:
            lib = ctypes.CDLL(libs[0])
        except OSError as error:
            out[pkg] = f"not loadable: {error}"
            continue
        out[pkg] = None
        for fn in names:
            f = getattr(lib, fn, None)
            if f is not None:
                f.restype = ctypes.c_char_p
                out[pkg] = f().decode("ascii", "replace").strip()
                break
    return out


def machine_facts():
    model, flags = None, set()
    try:
        with open("/proc/cpuinfo", encoding="utf8", errors="replace") as handle:
            for line in handle:
                if model is None and line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                if line.startswith("flags") and not flags:
                    flags = set(line.split(":", 1)[1].split())
    except OSError:
        pass
    try:
        from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__
        dispatch = [t for t in __cpu_dispatch__ if __cpu_features__.get(t)]
    except ImportError:
        dispatch = None
    try:
        f = ctypes.CDLL("libc.so.6").gnu_get_libc_version
        f.restype = ctypes.c_char_p
        glibc = f().decode("ascii")
    except (OSError, AttributeError):
        glibc = None
    return dict(cpu_model=model, avx512f="avx512f" in flags, avx512_flags=sorted(f for f in flags if f.startswith("avx512")),
                logical_processors=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)), openblas=_openblas(),
                numpy_dispatch_active=dispatch, glibc=glibc, kernel_release=platform.release(),
                python=sys.version.split()[0], boot_id=K.boot_id(), hostname=platform.node())
