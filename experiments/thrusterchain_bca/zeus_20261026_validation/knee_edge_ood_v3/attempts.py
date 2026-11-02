"""Approval and attempt rules of protocol 5.0, shared by the A3-A5 drivers (A2's driver carries its own copy).

- A step runs only with the user's approval file naming the step's sealed manifest hash and the attempt number k;
  attempts 1..k-1 must exist with reports, and the previous one must have ended INCOMPLETE.
- A unit is sealed when an earlier attempt completed it with a valid outcome (its sealed file exists and its label is
  not INCOMPLETE); a re-attempt runs only the units not yet sealed.
- Only a pre-computation (preflight) failure is exempt; a second counted failure of the same unit under an unchanged
  manifest is INCONCLUSIVE; at most one code-only fix (one manifest change) per step.
"""
from datetime import datetime, timezone
import json

from core import require, sha_file


def verify_upstream(here, pins):
    """The sealed packages a step runs on: each pinned manifest equals its committed hash and every file it lists is
    unchanged; rerun1's reused sources equal their recorded hashes (as step03.check_approval does for A2)."""
    import step03
    checked = {}
    for name, digest in pins.items():
        require(sha_file(here / name) == digest, f"{name} differs from the committed manifest")
        for f, d in json.loads((here / name).read_text(encoding="utf-8"))["files"].items():
            require(sha_file(here / f) == d, f"Sealed file changed: {f}")
        checked[name] = digest
    for name, digest in step03.RERUN1_SOURCES.items():
        require(sha_file(step03.RERUN1 / name) == digest, f"Reused rerun1 file changed: {name}")
    return {"manifests": checked, "rerun1_sources": len(step03.RERUN1_SOURCES)}


def history(run_dir):
    if not run_dir.is_dir():
        return []
    return sorted((p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_")), key=lambda p: int(p.name[8:]))


def check(*, approval_path, manifest_path, run_dir, words, units, sealed_files, required_files, here):
    """Validate the approval and return the attempt plan. `sealed_files` maps unit -> file name in an attempt dir."""
    require(approval_path.is_file(), f"{approval_path.name} is missing: the user's own approval is required")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    require(words in approval.get("approval", ""), "Approval wording missing")
    manifest_sha = sha_file(manifest_path)
    require(approval.get("package_manifest_sha256") == manifest_sha, "Approval names a different package")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(set(required_files) <= set(manifest["files"]), "Manifest lacks a required file")
    for name, digest in manifest["files"].items():
        require(sha_file(here / name) == digest, f"Sealed file changed: {name}")
    previous = history(run_dir)
    attempt = approval.get("attempt")
    require(type(attempt) is int and attempt == len(previous) + 1, f"Approval must name attempt {len(previous) + 1}")
    rows, sealed = [], {}
    for p in previous:
        require((p / "report.json").is_file(), f"{p.name} has no report; resolve it before a new attempt")
        prior = json.loads((p / "report.json").read_text(encoding="utf-8"))
        rows.append({"attempt": p.name, "label": prior.get("label"), "failed_unit": prior.get("failed_unit"),
                     "counts_under_5_0": prior.get("counts_under_5_0"),
                     "manifest_sha256": prior.get("approval", {}).get("manifest_sha256")})
        for unit, name in sealed_files.items():
            path = p / name
            if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("unit_label") != "INCOMPLETE":
                sealed[unit] = p.name
    if rows:
        require(rows[-1]["label"] == "INCOMPLETE", f"No re-attempt after {rows[-1]['label']} (5.0)")
        require(len({r["manifest_sha256"] for r in rows} | {manifest_sha}) - 1 <= 1, "At most one code-only fix (5.0)")
    return {"approval": approval, "attempt": attempt, "manifest_sha256": manifest_sha, "history": rows,
            "units": [u for u in units if u not in sealed], "sealed_units": sealed,
            "started_utc": datetime.now(timezone.utc).isoformat()}


def finish(report):
    """Apply the exemption and the second-failure rule to an INCOMPLETE report (in place)."""
    if report.get("label") == "INCOMPLETE":
        unit = report.get("failed_unit") or "preflight"
        report["failed_unit"] = unit
        report["counts_under_5_0"] = unit != "preflight"
        earlier = [h for h in report["approval"]["history"] if h.get("failed_unit") == unit and h.get("counts_under_5_0")]
        if report["counts_under_5_0"] and earlier and earlier[-1]["manifest_sha256"] == report["approval"]["manifest_sha256"]:
            report["label"] = "INCONCLUSIVE"
            report.setdefault("reasons", []).append(f"second failure of unit {unit!r} under unchanged code (5.0)")
    return report
