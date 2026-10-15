"""Freeze step 2's specification and code: writes prespec.json.sha256 and code_manifest.json (SHA-256 of every file
in step2lib.CODE_FILES). train_policy.py, evaluate_policy.py and pooled.py refuse to run when either no longer
matches.

Refuses once any official step-2 output exists (a lock, a stop, an actors file or a values file under
runs/shift_explore/restructure/step2/policy/, or results.json): after that the specification cannot be changed.
Writes only inside this folder.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2lib as S  # noqa: E402


def main():
    existing = S.official_outputs_exist()
    if existing:
        raise SystemExit(f"refused: official step-2 outputs exist ({existing[:3]}...); the specification is final")
    path = S.HERE / S.PRESPEC_NAME
    json.loads(path.read_text())          # must parse
    digest = S.file_sha256(path)
    (S.HERE / (S.PRESPEC_NAME + ".sha256")).write_text(f"{digest}  prespec.json\n")
    files = {n: S.file_sha256(S.HERE / n) for n in S.CODE_FILES}
    man = dict(label=S.LABEL, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), prespec_sha256=digest, files=files)
    (S.HERE / S.MANIFEST_NAME).write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print("prespec.json", digest)
    for n, h in files.items():
        print(f"  {h[:16]}  {n}")
    print("code_manifest.json", S.file_sha256(S.HERE / S.MANIFEST_NAME))


if __name__ == "__main__":
    main()
