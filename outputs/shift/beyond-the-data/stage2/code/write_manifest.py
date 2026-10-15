"""Stage 2: write runs/shift_explore/stage2/code_manifest.json (+ .sha256): the SHA-256 of every Stage 2 script
(stage2_common.CODE_FILES) with a copy of each under runs/shift_explore/stage2/code/, and of every repository file Stage 2
imports read-only (BCA's own code and the gate's hashed files, as the room harness lists them).

Refuses after approval: once approval.json exists the code is frozen (any change needs a new pre-registration and a
new approval). An earlier manifest is kept in code_manifest_history/ with the reason for the rewrite.

  python write_manifest.py --reason "first manifest"
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_common as C  # noqa: E402


def repository_files():
    H = C.load_room_harness()
    files = dict(H.gate_file_hashes())
    files.update(C.repository_hashes())
    files["runtime/networks.py"] = C.file_sha256(C.ROOT / "runtime" / "networks.py")
    files["experiments/signal/glq_match.py"] = C.file_sha256(C.ROOT / "experiments" / "signal" / "glq_match.py")
    return dict(sorted(files.items()))


def main(argv=None, paths=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reason", required=True)
    a = ap.parse_args(argv)
    paths = paths or C.PATHS
    if paths.approval.exists():
        raise SystemExit("refused: approval.json exists; the approved code is frozen")
    old = C.read_json(paths.manifest, required=False)
    history = []
    if old is not None:
        hist = paths.out / "code_manifest_history"
        n = len(old.get("history", [])) + 1
        C.guard_out(hist, paths).mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.manifest, hist / f"code_manifest_v{n}.json")
        history = old.get("history", []) + [dict(version=n, sha256=C.file_sha256(paths.manifest),
                                                 archived=f"code_manifest_history/code_manifest_v{n}.json")]
    files = {n: C.file_sha256(paths.code_src / n) for n in C.CODE_FILES}
    copies = C.guard_out(paths.code_copies, paths)
    copies.mkdir(parents=True, exist_ok=True)
    for n in C.CODE_FILES:
        shutil.copy2(paths.code_src / n, copies / n)
    body = dict(kind="stage2-code-manifest", label=C.LABEL, files=files, source_dir=str(paths.code_src),
                copies_dir=str(copies), repository_files=repository_files(), history=history, reason=a.reason,
                rule=("every Stage 2 BCA entry point refuses unless each file here and its copy have these SHA-256 "
                      "and this manifest's SHA-256 is the one prereg.json pins"),
                written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    digest = C.write_json(paths.manifest, body, paths, overwrite=True)
    print(f"wrote {paths.manifest} ({digest[:16]}); {len(files)} Stage 2 files, "
          f"{len(body['repository_files'])} repository files")


if __name__ == "__main__":
    main()
