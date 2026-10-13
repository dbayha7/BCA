"""EXPLORATORY: write runs/shift_explore/standins/spec.json (+ .sha256) before any stand-in run.

The spec may be rewritten only while no run output exists (no seed directory, no marks.json). A rewrite keeps the
previous spec in spec_history/ and moves what was computed under it (the seed-0 check files, timing.json, the code
manifest and its copies) to superseded/v<k>/; the new spec must be version k + 1 and name the replaced spec's
SHA-256 in previous_spec, with a reason.

Usage: python write_spec.py [--reason TEXT]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import standins as S  # noqa: E402

UNDER_OLD_SPEC = ("checks_seed0.json", "checks_seed0_stall.json", "checks_seed0_no_stall.json", "timing.json",
                  S.MANIFEST_NAME)


def any_run_output():
    found = list(S.OUT.glob("seed*"))
    found += [S.OUT / n for n in ("marks.json",) if (S.OUT / n).exists()]
    return found


def main(reason=None):
    if any_run_output():
        raise SystemExit("refused: stand-in run outputs exist; the spec may not change after any run: "
                         + ", ".join(str(p) for p in any_run_output()))
    payload = S.spec_payload()
    if S.SPEC_PATH.exists():
        old = S.read_json(S.SPEC_PATH)
        old_sha = S.file_sha256(S.SPEC_PATH)
        version = int(old.get("version", 1))
        trial = dict(payload, version=old.get("version"), previous_spec=old.get("previous_spec"))
        if trial.get("previous_spec") is None:
            trial.pop("previous_spec")
        if S.A.canonical(old) == S.A.canonical(trial):
            print("spec.json already written and identical")
            return
        if not reason:
            raise SystemExit("refused: a spec exists; a new version needs --reason")
        payload = dict(payload, version=version + 1,
                       previous_spec=dict(sha256=old_sha, archived=f"spec_history/spec_v{version}.json",
                                          replaced_because=reason,
                                          history=(old.get("previous_spec") or {}).get("history", []) +
                                          [dict(version=version, sha256=old_sha, replaced_because=reason)]))
        hist = S.OUT / "spec_history"
        hist.mkdir(exist_ok=True)
        for suffix in ("", ".sha256"):
            shutil.copy2(str(S.SPEC_PATH) + suffix, hist / f"spec_v{version}.json{suffix}")
        sup = S.OUT / "superseded" / f"v{version}"
        sup.mkdir(parents=True, exist_ok=True)
        for name in UNDER_OLD_SPEC:
            for suffix in ("", ".sha256"):
                p = S.OUT / (name + suffix)
                if p.exists():
                    shutil.move(str(p), sup / p.name)
        if (S.OUT / "code").exists():
            shutil.move(str(S.OUT / "code"), sup / "code")
        print(f"archived spec v{version} ({old_sha[:16]}) to {hist}; moved its outputs to {sup}")
    digest = S.write_json(S.SPEC_PATH, payload)
    print(f"wrote {S.SPEC_PATH} version {payload['version']} sha256 {digest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason")
    main(ap.parse_args().reason)
