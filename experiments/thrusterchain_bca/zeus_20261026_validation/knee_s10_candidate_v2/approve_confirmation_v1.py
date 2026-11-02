"""Write confirmation_v1/CF_APPROVAL.json with the rule's exact sentences, the manifest's sha256 and the attempt.

Lives outside the sealed folder. Run it yourself, at your terminal, only when you approve; it refuses to overwrite.

  python -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/approve_confirmation_v1.py --attempt 2
"""
import argparse
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent / "confirmation_v1"
TEXT = ("I approve the confirmation of the OOD-action overestimation comparison on the knee_s10_v2 CUTOFF data, Unifloral "
        "TD3+BC host, doses kappa 21, 63 and 189, both worlds, attempt {attempt}. And the BCA arms at kappa 21, 63 and 189 are "
        "approved, the confirmation seeds are fresh: no run has used them, and the dose being confirmed is 63; the other "
        "doses are a profile.")


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
