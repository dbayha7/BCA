"""Write confirmation_v2/CF_APPROVAL.json with the rule's exact sentences, the manifest's sha256 and the attempt.

Lives outside the sealed folder. Run it yourself, at your terminal, only when you approve; it refuses to overwrite.

  python -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/approve_confirmation_v2.py --attempt 1
"""
import argparse
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent / "confirmation_v2"
TEXT = ("I approve confirmation v2 of the OOD-action overestimation comparison on the knee_s10_v2 CUTOFF data, Unifloral "
        "TD3+BC host, kappa 189, both worlds, five fresh seeds, attempt {attempt}. And the BCA arm at kappa 189 is approved, "
        "the confirmation v2 seeds are fresh: no run has used them, kappa 189 was chosen from confirmation v1's profile, so "
        "this run, not that one, is its confirmation, and the I1 tolerance is 1e-5 relative.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attempt", type=int, required=True)
    attempt = parser.parse_args().attempt
    manifest = hashlib.sha256((PACKAGE / "CF_MANIFEST.json").read_bytes()).hexdigest()
    body = {"approval": TEXT.format(attempt=attempt), "package_manifest_sha256": manifest, "attempt": attempt}
    with (PACKAGE / "CF_APPROVAL.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(body, indent=2) + "\n")
    print(json.dumps({"written": "CF_APPROVAL.json", "attempt": attempt, "manifest_sha256": manifest}))


if __name__ == "__main__":
    main()
