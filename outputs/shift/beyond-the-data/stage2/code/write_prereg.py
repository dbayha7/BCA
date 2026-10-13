"""Stage 2: write runs/shift_explore/stage2/prereg.json (+ .sha256) from prereg_payload.payload().

Order: write_manifest.py first (the pre-registration pins the manifest's SHA-256), then this. When a pin is missing
(the detection script or the stand-ins test not locked yet) the file is written as a draft (approvable: false) that
approve.py refuses; rerun both scripts once they are locked. Refuses after approval. An earlier version is kept in
prereg_history/ (no earlier version was ever approved: approval would have frozen it).

  python write_prereg.py
"""

import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_common as C  # noqa: E402
import prereg_payload as PP  # noqa: E402


def main(paths=None):
    paths = paths or C.PATHS
    if paths.approval.exists():
        raise SystemExit("refused: approval.json exists; the approved pre-registration is frozen")
    version = 1
    if paths.prereg.exists():
        old = C.load_prereg(paths)
        version = int(old.payload.get("version", 0)) + 1
        hist = C.guard_out(paths.out / "prereg_history", paths)
        hist.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.prereg, hist / f"prereg_v{version - 1}.json")
        shutil.copy2(paths.prereg.with_name("prereg.json.sha256"), hist / f"prereg_v{version - 1}.json.sha256")
    body = PP.payload(version, paths)
    digest = C.write_json(paths.prereg, body, paths, overwrite=True)
    print(f"wrote {paths.prereg} version {version}: SHA-256 {digest}\n  first 16 characters: {digest[:16]}\n"
          f"  approvable: {body['approvable']}" + ("" if body["approvable"] else f"; missing pins: {body['missing_pins']}"))


if __name__ == "__main__":
    main()
