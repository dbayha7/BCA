"""Write G2_FORMAL_MANIFEST.json: the sha256 of every file the formal G2 runs on. `files` are this package and the
candidate's committed pi_ref and recorder; `upstream` are the knee_edge_ood_v3 modules the driver imports. Run once,
after g2_seeds.json and d4_bounds.json exist and the tests pass; the approval names the manifest's hash.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B knee_s10_candidate_v1/formal_g2/seal.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("G2_FORMAL_RULE.md", "g2_formal.py", "g2_seeds.py", "g2_seeds.json", "d4_bounds.py", "d4_bounds.json",
         "test_g2_formal.py", "seal.py", "README.md", "../pi_ref.py", "../recorder_wide.py", "../test_pi_ref.py",
         "../G2_RULE_r3.md", "../../ood_definition_v4/DRAFT_DEFINITION.md")
UPSTREAM = ("core.py", "worlds.py", "knee_region.py", "dataset_knee.py", "masks.py", "probes.py", "qpi.py",
            "region_independent.py", "stage1_eval.py", "stage1_math.py", "step04_math.py", "step04.py", "seeds.py",
            "seeds.json", "fixtures.py", "stage1_smoke.py", "bindings.py", "step03.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = {"schema": "knee_s10_candidate_v1.formal_g2.manifest.v1",
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
