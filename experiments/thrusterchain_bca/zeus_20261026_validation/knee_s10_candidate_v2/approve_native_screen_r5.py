"""Write native_screen_r5/SCREEN_APPROVAL.json with the r5 approval words (SCREEN_RULE_r5.md), bound to the sealed
manifest's sha256. Run by the user at their terminal; refuses to overwrite. Lives outside the sealed package so the
worker's added-file check never sees it.

  .venv/Scripts/python.exe experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/approve_native_screen_r5.py [--attempt N]
"""
import argparse
import hashlib
import json
from pathlib import Path

PKG = Path(__file__).resolve().parent / "native_screen_r5"
WORDS = ("I approve the native screen for candidate knee_s10_v2 under OOD definition v4 r5 at kappa 63, attempt {attempt}. "
         "I adopt definition v4 r5 for this candidate. "
         "RS1: the scored anchors are retained-supported; I overrule the start-share primary miss (121 of 128 against 122, "
         "a one-draw threshold); the secondary's agreement with D3 is on record. "
         "And the BCA arm at kappa 63 is approved, and the screen is exploratory and supports no protocol claim.")

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", type=int, default=1)
attempt = parser.parse_args().attempt
manifest = PKG / "SCREEN_MANIFEST.json"
if not manifest.is_file():
    raise SystemExit("SCREEN_MANIFEST.json is missing: the package is not sealed")
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
out = PKG / "SCREEN_APPROVAL.json"
with out.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps({"approval": WORDS.format(attempt=attempt), "package_manifest_sha256": digest, "attempt": attempt}, indent=2) + "\n")
print(json.dumps({"written": str(out), "attempt": attempt, "package_manifest_sha256": digest}, indent=2))
