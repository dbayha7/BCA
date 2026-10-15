"""CLOUD ONLY. Once seeds 0-4 have decided the version (the exploratory detection marks) and best_standin.json exists,
copy those files with their .sha256 sidecars into payload/late/ and write payload/late/LATE.json (with a sidecar).
The orchestrating agent then commits and pushes payload/late/, and the user's run_confirmatory.sh stops refusing.

    /tmp/claude-0/bcavenv/bin/python -I outputs/shift/beyond-the-data/stage2_gpu/cloud/publish_late.py

Refuses: when a file or its sidecar is missing or does not match; when a file is not this pre-registration's; when the
uniform version failed direction or size and the weighted version's exploratory marks do not exist yet; when
payload/late/ already holds different files. Reads runs/, writes only payload/late/.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
PKG = Path(__file__).resolve().parent.parent
OUT = Path("/home/user/bca/runs/shift_explore/stage2")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def verified(path):
    p = Path(path)
    side = p.with_name(p.name + ".sha256")
    if not p.exists() or not side.exists():
        sys.exit(f"refused: {p} or its sidecar is missing")
    raw = p.read_bytes()
    if side.read_text(encoding="utf8").split()[:1] != [sha(raw)]:
        sys.exit(f"refused: {p} differs from its sidecar")
    return raw, side.read_bytes(), json.loads(raw.decode("utf8"))


def main():
    fp = sha((OUT / "prereg.json").read_bytes())
    want = {"best_standin.json": OUT / "best_standin.json",
            "marks_exploratory_uniform.json": OUT / "detect" / "marks_exploratory_uniform.json"}
    got = {n: verified(p) for n, p in want.items()}
    bs, uni = got["best_standin.json"][2], got["marks_exploratory_uniform.json"][2]
    if bs.get("kind") != "stage2-best-standin" or bs.get("prereg_sha256") != fp:
        sys.exit("refused: best_standin.json is not this pre-registration's")
    if uni.get("kind") != "stage2-detection-marks" or uni.get("group") != "exploratory" or \
            uni.get("version") != "uniform" or uni.get("prereg_sha256") != fp or uni.get("seeds") != [0, 1, 2, 3, 4]:
        sys.exit("refused: marks_exploratory_uniform.json is not this pre-registration's exploratory uniform marks")
    chosen = "uniform"
    if uni["marks"].get("direction_or_size_failed"):
        p = OUT / "detect" / "marks_exploratory_weighted.json"
        if not p.exists():
            sys.exit("refused: the uniform version failed direction or size; the weighted version's exploratory marks "
                     "must exist before seeds 5-9 can start")
        got["marks_exploratory_weighted.json"] = verified(p)
        w = got["marks_exploratory_weighted.json"][2]
        if w.get("kind") != "stage2-detection-marks" or w.get("group") != "exploratory" or \
                w.get("version") != "weighted" or w.get("prereg_sha256") != fp:
            sys.exit("refused: marks_exploratory_weighted.json is not this pre-registration's")
        chosen = None if w["marks"].get("direction_or_size_failed") else "weighted"
    late = PKG / "payload" / "late"
    late.mkdir(parents=True, exist_ok=True)
    for name, (raw, side, _) in got.items():
        for fname, data in ((name, raw), (name + ".sha256", side)):
            dest = late / fname
            if dest.exists() and dest.read_bytes() != data:
                sys.exit(f"refused: {dest} exists with different contents")
            dest.write_bytes(data)
    index = dict(kind="stage2-offcloud-late-files", prereg_sha256=fp,
                 files={n: sha(raw) for n, (raw, _, _) in got.items()},
                 best_standin=bs.get("best"), version_for_seeds_5_9=chosen,
                 written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 note="copied unchanged from the cloud's runs/shift_explore/stage2 after seeds 0-4 decided")
    text = json.dumps(index, indent=1, sort_keys=True)
    path = late / "LATE.json"
    if path.exists() and path.read_text(encoding="utf8") != text:
        old = json.loads(path.read_text(encoding="utf8"))
        if {k: v for k, v in old.items() if k != "written"} != {k: v for k, v in index.items() if k != "written"}:
            sys.exit(f"refused: {path} exists with different contents")
        text = path.read_text(encoding="utf8")
    path.write_text(text, encoding="utf8")
    (late / "LATE.json.sha256").write_text(f"{sha(text.encode('utf8'))}  LATE.json\n", encoding="utf8")
    print(f"late files ready in {late}: best stand-in {bs.get('best')}, version for seeds 5-9 {chosen}")


if __name__ == "__main__":
    main()
