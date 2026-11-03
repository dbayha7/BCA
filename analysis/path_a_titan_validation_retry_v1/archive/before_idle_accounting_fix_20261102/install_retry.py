"""Add the approved retry files to the isolated Titan tree, without overwrites."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

PREFIX = "analysis/path_a_titan_validation_retry_v1/"
STUDY = "experiments/thrusterchain_bca/two_scenario_path_a"
TARGET = Path("/mnt/c/Users/David Bayha/BCA-PathA" if os.name == "posix"
              else "C:/Users/David Bayha/BCA-PathA")
MANIFEST_SHA = "32718856c6b4714ea1a4004554d61b80223d502e3517b5563c372a97047a15ed"
APPROVAL_SHA = "4fc24a5bb025d54372cdd14736cd24894b4053516e4393130bb4e8712dd3a156"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def destination(repo):
    repo = Path(repo).resolve()
    target = repo / PREFIX
    require(target.resolve().is_relative_to(repo), "Patch destination escapes the isolated repository")
    return target


def inspected_payload(source, expected_manifest):
    source = Path(source).resolve()
    manifest_path = source / "RETRY_MANIFEST.json"
    require(sha(manifest_path) == expected_manifest, "Patch manifest differs from the supplied digest")
    manifest = read(manifest_path)
    require(manifest["schema"] == "path_a.titan_validation_retry.v1", "Wrong retry schema")
    require(manifest["titan_manifest_sha256"] == MANIFEST_SHA, "Wrong original Titan manifest")
    approval = read(source / "RETRY_APPROVAL.json")
    require(approval["retry_manifest_sha256"] == expected_manifest and
            approval["validation_attempt"] == 2 and approval["prior_pid"] == 34106 and
            approval["zero_update_retry_only"] is True and
            approval["conditional_existing_recovery"] is True, "Retry approval binding differs")
    payload = {}
    for relative, digest in manifest["files"].items():
        require(relative.startswith(PREFIX), "Patch tries to modify a frozen or unrelated file")
        name = relative[len(PREFIX):]
        require(name and Path(name).name == name and name not in (".", "..") and
                "/" not in name and "\\" not in name, "Unsafe patch filename")
        path = source / name
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "Patch file changed: " + name)
        payload[name] = path.read_bytes()
    require({"titan_retry.py", "install_retry.py", "README.md"} <= set(payload), "Incomplete patch")
    for name in ("RETRY_MANIFEST.json", "RETRY_APPROVAL.json"):
        payload[name] = (source / name).read_bytes()
    return payload


def inspect_target(repo, confirm_helper_absent):
    repo = Path(repo).resolve()
    require(confirm_helper_absent, "Resolve path_a_a1.py provenance before installing this retry")
    # This checks the project only, not the entire computer. Operator confirmation
    # covers helpers supplied or used outside it. Never execute a discovered helper.
    require(not any(repo.rglob("path_a_a1.py")), "Existing path_a_a1.py must be reviewed first")
    study = repo / STUDY
    require(sha(study / "TITAN_MANIFEST.json") == MANIFEST_SHA, "Original Titan manifest changed")
    require(sha(study / "TITAN_APPROVAL.json") == APPROVAL_SHA, "Original Titan approval changed")
    for relative, expected in read(study / "TITAN_MANIFEST.json")["files"].items():
        candidate = (repo / relative).resolve()
        require(candidate.is_relative_to(repo) and candidate.is_file() and sha(candidate) == expected,
                "Original Titan pinned file changed: " + relative)
    for name in ("run_titan_recovery_1", "run_titan_recovery_2", "run_titan_recovery_validation2", "titan_validation_2"):
        require(not (study / name).exists(), "A validation/production reservation already exists: " + name)
    require(not destination(repo).exists(), "Retry installation already exists; preserve it and do not retry")
    prior = study / "titan_validation_1"
    require(prior.is_dir() and {p.name for p in prior.iterdir()} == {"ATTEMPT.json", "progress.jsonl"},
            "Validation 1 has changed or has terminal evidence; reconcile it first")
    attempt = read(prior / "ATTEMPT.json")
    require(attempt["pid"] == 34106 and attempt["process_cap_seconds"] == 120 and
            attempt["titan_manifest_sha256"] == MANIFEST_SHA and
            attempt["titan_approval_sha256"] == APPROVAL_SHA,
            "This is not the approved interrupted validation")
    require(attempt["target_identity"]["repository"] == "/mnt/c/Users/David Bayha/BCA-PathA" and
            attempt["target_identity"]["executable"] == "/home/dbayha/bca-gpu/bin/python",
            "Validation 1 target differs")
    return {name: sha(prior / name) for name in ("ATTEMPT.json", "progress.jsonl")}


def install(source, repo, expected_manifest, confirm_helper_absent):
    started = time.monotonic()
    payload = inspected_payload(source, expected_manifest)
    prior_hashes = inspect_target(repo, confirm_helper_absent)
    target = destination(repo)
    target.mkdir(parents=True, exist_ok=False)
    # Partial installation is preserved on any error. No cleanup, replacement,
    # import of experiment code, process start or automatic retry is performed.
    for name, content in payload.items():
        require(destination(repo).resolve() == target.resolve() and
                (target / name).resolve().is_relative_to(Path(repo).resolve()),
                "Installation destination changed")
        with (target / name).open("xb") as stream:
            stream.write(content)
    require(prior_hashes == {name: sha(Path(repo) / STUDY / "titan_validation_1" / name)
                            for name in prior_hashes}, "Validation 1 changed during installation")
    receipt = {"schema": "path_a.titan_validation_retry_install.v1",
               "status": "installed_without_overwriting",
               "repository": "/mnt/c/Users/David Bayha/BCA-PathA",
               "manifest_sha256": expected_manifest,
               "installation_elapsed_seconds": time.monotonic() - started,
               "timing_known": True, "path_a_a1_absent": True,
               "first_validation_files": prior_hashes, "first_validation_pid": 34106,
               "no_scientific_work": True}
    destination(repo)
    with (target / "INSTALL_RECEIPT.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repository", type=Path, default=TARGET)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--confirm-helper-absent-not-used", action="store_true")
    args = parser.parse_args()
    require(args.repository.resolve() == TARGET.resolve(), "Use only the isolated Titan target")
    print(json.dumps(install(args.source, args.repository, args.manifest_sha256,
                             args.confirm_helper_absent_not_used), sort_keys=True))
