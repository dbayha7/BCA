"""Named seeds of the overestimation comparison (OVERESTIMATION_RULE.md). Namespace
`knee_s10_candidate_v2/overestimation_v1/`; derivation as the native screen's: the first 8 bytes of
sha256(namespace + name), numpy seeds & (2**63 - 1). Two names: `visited_states` (stage1_eval.visited_states) and
`dataset_rows` (the 192 dataset probe rows). Training rows and data-plan seeds are the native screen's
(`../native_screen/screen_seeds.json`), never used for training because the screen never trained. Audited against
every integer on record plus the screen's values."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
for _p in (str(CAND.parent / "knee_edge_ood_v3"), str(CAND / "formal_g2"), str(CAND / "native_screen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import g2_seeds_v2 as GS2                                   # noqa: E402
import screen_seeds as SS                                   # noqa: E402

SEEDS_PATH = HERE / "oe_seeds.json"
NAMESPACE = "knee_s10_candidate_v2/overestimation_v1/"
NAMES = ("visited_states", "dataset_rows")


def _word(name):
    return int.from_bytes(hashlib.sha256((NAMESPACE + name).encode()).digest()[:8], "big")


def numpy_seed(name):
    return _word(name) & ((1 << 63) - 1)


def build():
    return {"schema": "knee_s10_candidate_v2.overestimation_v1.seeds.v1",
            "derivation": f"first 8 bytes of sha256('{NAMESPACE}' + name); numpy & (2**63 - 1)",
            "numpy": {name: numpy_seed(name) for name in NAMES},
            "training_and_data_plan": "../native_screen/screen_seeds.json (unused by any training before this package)",
            "screen_seeds_sha256": hashlib.sha256(SS.SEEDS_PATH.read_bytes()).hexdigest()}


def audit(record, prior):
    prior = set(prior)
    values = list(record["numpy"].values())
    raw = sorted(set(values) & prior)
    return {"prior_integers_scanned": len(prior), "named_values": len(values), "named_values_distinct": len(set(values)) == len(values),
            "raw_collisions": raw, "clean": len(set(values)) == len(values) and not raw}


def write(experiments_root):
    record = build()
    import stage1_smoke
    screen = SS.load()
    prior = (GS2.prior_integers(experiments_root) | set(GS2.load()["numpy"].values()) | stage1_smoke.sealed_values()
             | set(SS.new_values(screen)) | set(screen["numpy"].values()))
    record["audit"] = audit(record, prior)
    if not record["audit"]["clean"]:
        raise SystemExit("Seed collision: " + json.dumps(record["audit"]))
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    with SEEDS_PATH.open("xb") as stream:
        stream.write(raw)
    return {"path": str(SEEDS_PATH), "sha256": hashlib.sha256(raw).hexdigest(), "audit": record["audit"]}


def load():
    record = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))
    fresh = build()
    if json.dumps({k: record[k] for k in fresh}, sort_keys=True) != json.dumps(fresh, sort_keys=True):
        raise SystemExit("oe_seeds.json differs from its derivation (or the screen seeds changed)")
    if not record["audit"]["clean"]:
        raise SystemExit("oe_seeds.json audit is not clean")
    return record


if __name__ == "__main__":
    print(json.dumps(write(HERE.parents[3]), indent=1))
