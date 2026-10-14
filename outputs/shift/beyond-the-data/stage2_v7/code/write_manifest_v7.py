"""Stage 2, version 7: write V7/code_manifest_v7.json (+ .sha256): the SHA-256 of every v7 file, and the diff of
launch_v7.sh against v6's launch.sh (V7_SPEC.md section 4).

What it hashes: every file under outputs/shift/beyond-the-data/stage2_v7/ except the ones that pin it or are written
later (prereg_v7.json, this manifest, prereg_v7_history/, handoff/, results/, return_check.json, the approval and import
records) and files git ignores here (*.pyc, __pycache__, *.npz, *.msgpack, *.log). It also:
  - writes diff_from_v6/launch.sh.diff (a unified diff of S2C/launch.sh against code/launch_v7.sh; generated, so it is
    rewritten whenever it differs);
  - places V7/approve_local_v7.py, a byte copy of code/approve_local_v7.py (the user runs the one next to prereg_v7.json),
    and refuses if a different copy is there;
  - refuses if the laptop package's byte copies of the GPU/ files (lib/envfacts.py, lib/gpu_probe.py,
    requirements-lock.txt) differ from GPU/'s;
  - lists the required files that do not exist yet (missing_required): write_prereg_v7.py writes no approvable
    pre-registration while any is missing.
It refuses once an approval or import record of the current prereg_v7.json's fingerprint exists (the code is frozen).
The bytes depend only on the files (no time), so a rerun with no change gives the same manifest.

  python write_manifest_v7.py
"""

import argparse
import difflib
import fnmatch
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as K  # noqa: E402

EXCLUDE_TOP = {"prereg_v7.json", "prereg_v7.json.sha256", "code_manifest_v7.json", "code_manifest_v7.json.sha256",
               "return_check.json", "return_check.json.sha256"}
EXCLUDE_DIRS = {"handoff", "results", "prereg_v7_history", "__pycache__"}
EXCLUDE_NAMES = ("approval_v7_*", "import_v7_*", "*.pyc", "*.pyo", "*.npz", "*.msgpack", "*.log")
CODE_REQUIRED = ("v7_common.py", "v7_constants.py", "v7_refs.py", "v7_verify.py", "v7_ledger.py", "v7_probes.py",
                 "run_stage2_v7.py", "launch_v7.sh", "approve_v7.py", "approve_local_v7.py", "write_prereg_v7.py",
                 "write_manifest_v7.py", "make_world_constants.py", "test_v7.py")
REQUIRED = tuple("code/" + n for n in CODE_REQUIRED) + (
    "approve_local_v7.py", "diff_from_v6/launch.sh.diff", "world_constants.json", "world_constants.json.sha256",
    "README.md", "PREREGISTRATION_V7.md",
    "laptop/restore_v7.sh", "laptop/selftest_v7.sh", "laptop/run_v7.sh", "laptop/push_runfiles_v7.sh",
    "laptop/lib/common_v7.sh", "laptop/lib/runner_v7.py", "laptop/lib/selftest_v7.py", "laptop/lib/make_results_v7.py",
    "laptop/lib/envfacts.py", "laptop/lib/gpu_probe.py", "laptop/requirements-lock.txt",
    "cloud/make_handoff.py", "cloud/check_return_v7.py")
BYTE_COPIES = {"laptop/lib/envfacts.py": "outputs/shift/beyond-the-data/stage2_gpu/lib/envfacts.py",
               "laptop/lib/gpu_probe.py": "outputs/shift/beyond-the-data/stage2_gpu/lib/gpu_probe.py",
               "laptop/requirements-lock.txt": "outputs/shift/beyond-the-data/stage2_gpu/requirements-lock.txt"}
DIFF_REL = "diff_from_v6/launch.sh.diff"


def launch_diff(v7_root=None, root=None):
    v7_root, root = Path(v7_root or K.V7), Path(root or K.ROOT)
    old = (root / K.S2_REL / "code" / "launch.sh").read_text(encoding="utf8").splitlines(keepends=True)
    new = (v7_root / "code" / "launch_v7.sh").read_text(encoding="utf8").splitlines(keepends=True)
    return "".join(difflib.unified_diff(old, new, fromfile="outputs/shift/beyond-the-data/stage2/code/launch.sh",
                                        tofile="outputs/shift/beyond-the-data/stage2_v7/code/launch_v7.sh"))


def excluded(rel):
    parts = rel.split("/")
    if len(parts) == 1 and parts[0] in EXCLUDE_TOP:
        return True
    if any(p in EXCLUDE_DIRS for p in parts[:-1]):
        return True
    return any(fnmatch.fnmatchcase(parts[-1], pat) for pat in EXCLUDE_NAMES)


def collect(v7_root):
    v7_root = Path(v7_root)
    files = {}
    for p in sorted(v7_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(v7_root).as_posix()
            if not excluded(rel):
                files[rel] = K.file_sha256(p)
    return files


def frozen(v7_root):
    """True when an approval or import record of the current prereg_v7.json's fingerprint exists."""
    v7_root = Path(v7_root)
    p = v7_root / "prereg_v7.json"
    if not p.is_file():
        return False
    prefix = K.prefix_of(K.file_sha256(p))
    return any((v7_root / n.format(prefix=prefix)).exists() for n in (K.APPROVAL_NAME, K.IMPORT_NAME))


def main(argv=None, v7_root=None, root=None, log=print):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    v7_root, root = Path(v7_root or K.V7), Path(root or K.ROOT)
    if frozen(v7_root):
        raise SystemExit("refused: an approval or import record of the current prereg_v7.json exists; the v7 code is "
                         "frozen (any change needs a new version)")
    src, copy = v7_root / "code" / "approve_local_v7.py", v7_root / "approve_local_v7.py"
    if copy.exists() and copy.read_bytes() != src.read_bytes():
        raise SystemExit(f"refused: {copy} differs from {src}; it must be a byte copy (remove it to recopy)")
    if not copy.exists():
        shutil.copyfile(src, copy)
    for rel, gpu_rel in BYTE_COPIES.items():
        p = v7_root / rel
        if p.exists() and p.read_bytes() != (root / gpu_rel).read_bytes():
            raise SystemExit(f"refused: {rel} is not a byte copy of {gpu_rel}")
    diff = launch_diff(v7_root, root)
    d = v7_root / DIFF_REL
    if not d.exists() or d.read_text(encoding="utf8") != diff:
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(diff, encoding="utf8")
    files = collect(v7_root)
    missing = sorted(r for r in REQUIRED if r not in files)
    body = dict(kind=K.MANIFEST_KIND, label=K.LABEL, files=files, missing_required=missing,
                required=list(REQUIRED),
                launch_diff=dict(file=DIFF_REL, sha256=K.sha256_bytes(diff.encode("utf8")),
                                 v6_launch_sh=K.S2_REL + "/code/launch.sh",
                                 v6_launch_sh_sha256=K.file_sha256(root / K.S2_REL / "code" / "launch.sh"),
                                 v7_launch_sh_sha256=files.get("code/launch_v7.sh")),
                byte_copies_of_gpu=BYTE_COPIES, v6_code_manifest_sha256=K.V6["code_manifest_sha256"],
                rule=("every v7 phase and the self-test refuse unless each file here has this SHA-256 and this "
                      "manifest's SHA-256 is the one prereg_v7.json pins"))
    text = K.canonical(body).encode("utf8")
    digest = K.sha256_bytes(text)
    man = v7_root / "code_manifest_v7.json"
    man.write_bytes(text)
    man.with_name(man.name + ".sha256").write_text(f"{digest}  {man.name}\n", encoding="utf8")
    log(f"wrote {man} ({digest[:16]}); {len(files)} files; missing required: {missing or 'none'}")
    return body


if __name__ == "__main__":
    main()
