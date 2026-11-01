"""Write G2_FORMAL_MANIFEST.json: the sha256 of every file the formal G2 of candidate knee_s10_v2 runs on. `files`
are this package, the v2 prefix rule and its development record, and the v1 files v2 inherits (pi_ref, recorder,
formal driver, seeds, bounds, v1's manifest and rule) plus the definition; `upstream` are the knee_edge_ood_v3
modules the drivers import. Run once, after g2_seeds_v2.json exists and the tests pass; the approval names the
manifest's hash.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B knee_s10_candidate_v2/formal_g2/seal.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("G2_FORMAL_RULE_v2.md", "g2_formal_v2.py", "g2_seeds_v2.py", "g2_seeds_v2.json", "d4_bounds.json",
         "test_g2_formal_v2.py", "seal.py", "README.md",
         "../prefix_rule.py", "../G2_DEV_RULE_v2.md", "../g2_development_result_v2.json", "../g2_dev_v2.py", "../test_v2.py",
         "../README.md",
         "../../knee_s10_candidate_v1/pi_ref.py", "../../knee_s10_candidate_v1/recorder_wide.py",
         "../../knee_s10_candidate_v1/test_pi_ref.py", "../../knee_s10_candidate_v1/G2_RULE_r3.md",
         "../../knee_s10_candidate_v1/formal_g2/g2_formal.py", "../../knee_s10_candidate_v1/formal_g2/g2_seeds.py",
         "../../knee_s10_candidate_v1/formal_g2/g2_seeds.json", "../../knee_s10_candidate_v1/formal_g2/d4_bounds.py",
         "../../knee_s10_candidate_v1/formal_g2/d4_bounds.json", "../../knee_s10_candidate_v1/formal_g2/G2_FORMAL_MANIFEST.json",
         "../../knee_s10_candidate_v1/formal_g2/G2_FORMAL_RULE.md",
         "../../ood_definition_v4/DRAFT_DEFINITION.md")
UPSTREAM = ("core.py", "worlds.py", "knee_region.py", "dataset_knee.py", "masks.py", "probes.py", "qpi.py",
            "region_independent.py", "stage1_eval.py", "stage1_math.py", "step04_math.py", "step04.py", "seeds.py",
            "seeds.json", "fixtures.py", "stage1_smoke.py", "bindings.py", "step03.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = {"schema": "knee_s10_candidate_v2.formal_g2.manifest.v1",
                "files": {f: sha(HERE / f) for f in FILES},
                "upstream": {f"../../knee_edge_ood_v3/{f}": sha(HERE / "../../knee_edge_ood_v3" / f) for f in UPSTREAM}}
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    out = HERE / "G2_FORMAL_MANIFEST.json"
    with out.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"written": out.name, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                      "files": len(manifest["files"]), "upstream": len(manifest["upstream"])}, indent=1))


if __name__ == "__main__":
    main()
