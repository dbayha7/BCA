"""Write collection/collection.json for the native screen: the two formal G2 v2 datasets by digest and the Stage 1
data plan of both arms in both worlds (bindings.plan_streams, with the screen's data-plan seeds), in the shape
stage1_worker.load_world reads. Exact arithmetic, CPU, no training; run once before sealing.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B knee_s10_candidate_v2/native_screen/prepare.py
"""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "formal_g2"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import screen_bind as SB                                    # noqa: E402  (imports g2_formal_v2 first: recorder substituted)
import bindings as B                                        # noqa: E402
import step03                                               # noqa: E402
from core import canonical_json, sha_file                   # noqa: E402

OUT = HERE / "collection"


def main():
    SB.patch_bindings()
    record = {"schema": "knee_s10_candidate_v2.native_screen.collection.v1", "unit_label": "COMPLETE",
              "source": "formal G2 v2 attempt 1 datasets, by digest", "formal_manifest_sha256": SB.FORMAL_REPORT["manifest_sha256"],
              "datasets": {}, "data_plan": {}, "seeds_sha256": sha_file(SB.SS.SEEDS_PATH)}
    for world in ("stall3", "nostall"):
        ds = SB.load_dataset(world)
        plan = B.plan_streams(ds, world)
        receipts = {arm: json.loads(json.dumps(plan["plans"][arm]["receipt"], default=step03._plain)) for arm in B.ARMS}
        rel = Path("..") / ".." / "formal_g2" / "formal_g2_run" / "attempt_1" / SB.DATASET_FILES[world].name
        record["datasets"][world] = {"file": str(rel).replace("\\", "/"), "sha256": ds.digest, "entropy": SB.ENTROPY}
        record["data_plan"][world] = {"receipts": receipts, "mask_receipt_digests": plan["mask_receipt"]["retained_row_digests"],
                                      "normalization_rows": plan["normalization_rows"],
                                      "rows": {arm: receipts[arm]["streams"]["train"]["rows"] for arm in B.ARMS}}
    OUT.mkdir(exist_ok=True)
    path = OUT / "collection.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"written": str(path), "sha256": sha_file(path),
                      "rows": {w: record["data_plan"][w]["rows"] for w in record["data_plan"]}}, indent=1))


if __name__ == "__main__":
    main()
