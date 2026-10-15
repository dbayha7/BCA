"""CLOUD ONLY. Copy the files the confirmatory phases (fresh seeds 5-9) need from the cloud's runs/ folder (which git
ignores) into payload/files/, and write payload/MANIFEST.json (with a .sha256 sidecar) that names each file's place
on the user's machine and its SHA-256. Also lists the committed files that restore.sh copies to the places the pinned
code expects (the Stage 2 code, its pre-registration and code manifest, the fallback diagnostic).

    /tmp/claude-0/bcavenv/bin/python -I outputs/shift/beyond-the-data/stage2_gpu/cloud/build_payload.py

Why each file is needed (traced from the pinned code; see README.md, "What the payload holds"):
  - the gate's pre-registration and the records it is checked against: acceptance.load_prereg, called by every
    phase (through fresh.gate_prereg), re-reads stage0_*.json, build_checks.json and every data record it lists;
  - the room test's code manifest, spec and pinned harness copy: imported from its pinned bytes by Stage 2, the
    detection script and the stand-ins module; its pooled look: the run condition of the policy test and full host;
  - the detection script's spec, code manifest and every pinned file: verified whole, then imported;
  - the stand-ins test's code manifest and every pinned file: verified whole, then imported by the policy test;
  - Stage 2's approval.json, the import of the user's approval (checked with git by every BCA phase).
Nothing large and no model file: the payload holds JSON records and source files only. Writes only under payload/.
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = Path("/home/user/bca")
RUNS = ROOT / "runs"
STAGE2_REL = "outputs/shift/beyond-the-data/stage2"
FALLBACK_REL = "outputs/shift/beyond-the-data/scripts/fallback/fallback.py"
LIMIT_BYTES = 20 * 1000 * 1000


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def encode(rel):
    """A flat file name for a path under the repository (git ignores folders named 'runs' or 'data' anywhere)."""
    return rel.replace("/", "__")


def needed_runs_files():
    """(repository-relative path, role) of every runs/ file the confirmatory phases read."""
    out = []

    def add(rel, role, sidecar=False):
        out.append((rel, role))
        if sidecar:
            out.append((rel + ".sha256", role + " (its SHA-256 sidecar, checked when it is read)"))

    gate = json.loads((RUNS / "shift" / "preregistration.json").read_text(encoding="utf8"))
    add("runs/shift/preregistration.json", "the gate's pre-registration (acceptance.load_prereg, every phase)", True)
    for name in ("stage0_base.json", "stage0_retune.json", "build_checks.json"):
        add(f"runs/shift/pre/{name}", "a gate record the pre-registration is re-derived from (load_prereg)", True)
    for key in sorted(gate["data"]):
        add(f"runs/shift/pre/data/{key}/record.json", "a gate data record the pre-registration lists (load_prereg)",
            True)
    room = "runs/shift_explore/room"
    add(f"{room}/code_manifest.json", "the room test's code manifest (pinned by prereg.json)", True)
    add(f"{room}/spec.json", "the room test's spec (pinned by prereg.json)")
    add(f"{room}/code/harness.py", "the room test's harness, imported from its pinned bytes")
    add(f"{room}/pooled.json", "the room test's pooled look: the run condition of the policy test and full host", True)
    for comp, extra in (("detect", True), ("standins", False)):
        base = f"runs/shift_explore/{comp}"
        man = json.loads((ROOT / base / "code_manifest.json").read_text(encoding="utf8"))
        add(f"{base}/code_manifest.json", f"the {comp} component's code manifest (pinned by prereg.json)", True)
        if extra:
            add(f"{base}/spec.json", "the detection script's spec (read by every detection call)", True)
        for name in sorted(man["files"]):
            add(f"{base}/code/{name}", f"a pinned file of the {comp} component (its whole manifest is verified)")
    add("runs/shift_explore/stage2/approval.json", "the import of the user's approval (every BCA phase)", True)
    return out


def committed_sources():
    """Committed files that restore.sh copies to where the pinned code expects them: (source, [targets])."""
    prereg = json.loads((ROOT / STAGE2_REL / "prereg.json").read_text(encoding="utf8"))
    man = json.loads((ROOT / STAGE2_REL / "code_manifest.json").read_text(encoding="utf8"))
    code_dir = man["source_dir"]                       # where every path of the pinned code is fixed to
    fallback_path = prereg["pins"]["fallback_py"]["path"]
    out = [(f"{STAGE2_REL}/{n}", [f"runs/shift_explore/stage2/{n}"])
           for n in ("prereg.json", "prereg.json.sha256", "code_manifest.json", "code_manifest.json.sha256")]
    for name in sorted(man["files"]):
        out.append((f"{STAGE2_REL}/code/{name}", [f"runs/shift_explore/stage2/code/{name}", f"{code_dir}/{name}"]))
    out.append((FALLBACK_REL, [fallback_path]))
    return out, code_dir, fallback_path


def main():
    prereg_sha = sha256_file(RUNS / "shift_explore" / "stage2" / "prereg.json")
    files_dir = PKG / "payload" / "files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    files_dir.mkdir(parents=True)
    entries, total = [], 0
    for rel, role in needed_runs_files():
        src = ROOT / rel
        data = src.read_bytes()
        name = encode(rel)
        (files_dir / name).write_bytes(data)
        entries.append(dict(name=name, target=rel, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                            role=role))
        total += len(data)
    sources, code_dir, fallback_path = committed_sources()
    committed = []
    for src, targets in sources:
        data = (ROOT / src).read_bytes()
        committed.append(dict(source=src, targets=targets, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    manifest = dict(
        kind="stage2-offcloud-payload-manifest",
        about=("files the confirmatory phases (fresh seeds 5-9) of Stage 2 need that git does not carry, and where "
               "they go; restore.sh copies them and verifies every hash, then checks the pre-registration's own pins"),
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        repository_root=str(ROOT), stage2_code_dir=code_dir, fallback_path=fallback_path,
        venv_link="/tmp/claude-0/bcavenv", prereg_sha256=prereg_sha,
        gate_prereg_sha256=sha256_file(RUNS / "shift" / "preregistration.json"),
        payload_files=entries, payload_bytes=total, committed_sources=committed,
        late_files=dict(
            folder="payload/late", index="LATE.json",
            files={"best_standin.json": "runs/shift_explore/stage2/best_standin.json",
                   "marks_exploratory_uniform.json": "runs/shift_explore/stage2/detect/marks_exploratory_uniform.json",
                   "marks_exploratory_weighted.json":
                       "runs/shift_explore/stage2/detect/marks_exploratory_weighted.json"},
            rule=("written by the cloud from seeds 0-4 (cloud/publish_late.py): best_standin.json and the uniform "
                  "version's exploratory detection marks always; the weighted version's marks only when the uniform "
                  "version failed direction or size; each with its .sha256 sidecar")))
    text = json.dumps(manifest, indent=1, sort_keys=True)
    path = PKG / "payload" / "MANIFEST.json"
    path.write_text(text, encoding="utf8")
    path.with_name("MANIFEST.json.sha256").write_text(
        f"{hashlib.sha256(text.encode('utf8')).hexdigest()}  MANIFEST.json\n", encoding="utf8")
    if total > LIMIT_BYTES:
        sys.exit(f"refused: the payload is {total} bytes, over {LIMIT_BYTES}")
    print(f"payload: {len(entries)} files, {total} bytes; {len(committed)} committed files to place")


if __name__ == "__main__":
    main()
