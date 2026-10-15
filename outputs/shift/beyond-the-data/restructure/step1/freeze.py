"""Freeze step 1's specification and code: writes prespec.json.sha256 and code_manifest.json (SHA-256 of every file
in step1lib.CODE_FILES). train_critics.py and evaluate.py refuse to run when either no longer matches.

Refuses once any official step-1 output exists (a critic record under runs/shift_explore/restructure/step1/critics/
or results.json): after that the specification cannot be changed. Writes only inside this folder.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1lib as S  # noqa: E402


def main():
    existing = sorted(str(p) for p in S.OFFICIAL_CRITICS.glob("*/seed*/record.json")) if S.OFFICIAL_CRITICS.exists() \
        else []
    existing += [str(p) for p in (S.OFFICIAL_OUT / "results.json",) if p.exists()]
    existing += sorted(str(p) for p in S.OFFICIAL_CRITICS.glob("*/seed*/critic.msgpack")) if S.OFFICIAL_CRITICS.exists() \
        else []
    if existing:
        raise SystemExit(f"refused: official step-1 outputs exist ({existing[:3]}...); the specification is final")
    json.loads(S.PRESPEC_PATH.read_text())          # must parse
    digest = S.file_sha256(S.PRESPEC_PATH)
    S.PRESPEC_SHA_PATH.write_text(f"{digest}  prespec.json\n")
    files = {n: S.file_sha256(S.HERE / n) for n in S.CODE_FILES}
    man = dict(label=S.LABEL, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), prespec_sha256=digest, files=files)
    S.MANIFEST_PATH.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print("prespec.json", digest)
    for n, h in files.items():
        print(f"  {h[:16]}  {n}")
    print("code_manifest.json", S.file_sha256(S.MANIFEST_PATH))


if __name__ == "__main__":
    main()
