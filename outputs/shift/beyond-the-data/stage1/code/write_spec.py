"""EXPLORATORY: write runs/shift_explore/room/spec.json (+ .sha256) before any room-test training.

The spec may be rewritten only while no seed output, lock, values, futility or pooled file exists (no training run of
the room test has happened). A rewrite keeps the previous spec in spec_history/ and moves the outputs that were
computed under it (the seed-0 checks, the timing summary, the code manifest) to superseded/v<k>/, each with its
sidecar; the new spec lists its changes in changes_before_any_run (harness.CHANGES_BEFORE_ANY_RUN).

Usage: python write_spec.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

UNDER_OLD_SPEC = ("checks_seed0.json", "checks_seed0_a.json", "checks_seed0_b.json", "checks_seed0_c.json",
                  "checks_seed0_d.json", "checks_seed0_e.json", "timing.json", H.MANIFEST_NAME)


def any_run_output():
    found = [p for p in H.OUT.glob("seed*")]
    found += [H.OUT / n for n in ("futility.json", "pooled.json") if (H.OUT / n).exists()]
    return found


def build_payload():
    """The spec as harness.spec_payload states it, with the pre-spec timing record."""
    t = H.read_json(H.OUT / "pre_spec_timing.json")
    timing = dict(file=str(H.OUT / "pre_spec_timing.json"), sha256=H.file_sha256(H.OUT / "pre_spec_timing.json"),
                  note="random start actor (not the cloned actor), results discarded, no strength or return",
                  steps=t["steps"], seconds_per_1000_steps_per_arm={k: v["seconds_per_1000_steps_per_arm"]
                                                                    for k, v in t["kinds"].items()})
    return H.spec_payload(timing)


def main():
    if any_run_output():
        raise SystemExit("refused: room-test run outputs exist; the spec may not change after any training: "
                         + ", ".join(str(p) for p in any_run_output()))
    payload = build_payload()
    if H.SPEC_PATH.exists():
        old = H.read_json(H.SPEC_PATH)
        if H.A.canonical(old) == H.A.canonical(payload):
            print("spec.json already written and identical")
            return
        old_sha = H.file_sha256(H.SPEC_PATH)
        version = int(old.get("version", 1))
        if payload["version"] != version + 1:
            raise SystemExit(f"refused: the new spec must be version {version + 1}")
        if payload.get("previous_spec", {}).get("sha256") != old_sha:
            raise SystemExit("refused: the new spec must name the SHA-256 of the spec it replaces")
        hist = H.OUT / "spec_history"
        hist.mkdir(exist_ok=True)
        for suffix in ("", ".sha256"):
            shutil.copy2(str(H.SPEC_PATH) + suffix, hist / f"spec_v{version}.json{suffix}")
        sup = H.OUT / "superseded" / f"v{version}"
        sup.mkdir(parents=True, exist_ok=True)
        for name in UNDER_OLD_SPEC:
            for suffix in ("", ".sha256"):
                p = H.OUT / (name + suffix)
                if p.exists():
                    shutil.move(str(p), sup / p.name)
        if (H.OUT / "code").exists():
            shutil.move(str(H.OUT / "code"), sup / "code")
        print(f"archived spec v{version} ({old_sha[:16]}) to {hist}; moved its outputs to {sup}")
    digest = H.write_json(H.SPEC_PATH, payload)
    print(f"wrote {H.SPEC_PATH} version {payload['version']} sha256 {digest}")


if __name__ == "__main__":
    main()
