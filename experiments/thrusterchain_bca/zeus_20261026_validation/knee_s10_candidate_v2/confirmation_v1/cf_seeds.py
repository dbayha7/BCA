"""Fresh seeds of the confirmation (CONFIRMATION_RULE.md). Namespace `knee_s10_candidate_v2/confirmation/`, reserved
by the native screen's seeds and drawn from by no run before this package. Derivation as the screen's: the first 8
bytes of sha256(namespace + name); training seeds 1 + word % (2**31 - 2); numpy seeds & (2**63 - 1). Three training
rows of Stage 1's fields plus `visited_states` and `dataset_rows`. Audited against every integer on record, the
screen's values and the exploratory package's."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
for _p in (str(CAND.parent / "knee_edge_ood_v3"), str(CAND / "formal_g2"), str(CAND / "native_screen"), str(CAND / "overestimation_v1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import g2_seeds_v2 as GS2                                   # noqa: E402
import screen_seeds as SS                                   # noqa: E402
import oe_seeds as OS                                       # noqa: E402

SEEDS_PATH = HERE / "cf_seeds.json"
NAMESPACE = "knee_s10_candidate_v2/confirmation/"
STAGE1_FIELDS = SS.STAGE1_FIELDS
NUMPY_NAMES = ("visited_states", "dataset_rows")
SEEDS = (0, 1, 2)


def _word(name):
    return int.from_bytes(hashlib.sha256((NAMESPACE + name).encode()).digest()[:8], "big")


def numpy_seed(name):
    return _word(name) & ((1 << 63) - 1)


def training_seed(name):
    return 1 + _word(name) % ((1 << 31) - 2)


def build():
    return {"schema": "knee_s10_candidate_v2.confirmation_v1.seeds.v1",
            "derivation": f"first 8 bytes of sha256('{NAMESPACE}' + name); training 1 + word % (2**31 - 2); numpy & (2**63 - 1)",
            "numpy": {name: numpy_seed(name) for name in NUMPY_NAMES},
            "training": [{"index": i, **{f: training_seed(f"cf_seed{i}/{f}") for f in STAGE1_FIELDS}} for i in SEEDS],
            "data_plan": "../native_screen/screen_seeds.json (the CUTOFF rows must match the sealed collection record)",
            "screen_seeds_sha256": hashlib.sha256(SS.SEEDS_PATH.read_bytes()).hexdigest(),
            "oe_seeds_sha256": hashlib.sha256(OS.SEEDS_PATH.read_bytes()).hexdigest()}


def new_values(record):
    return list(record["numpy"].values()) + [r[f] for r in record["training"] for f in STAGE1_FIELDS]


def audit(record, prior):
    prior = set(prior)
    values = new_values(record)
    raw = sorted(set(values) & prior)
    return {"prior_integers_scanned": len(prior), "named_values": len(values), "named_values_distinct": len(set(values)) == len(values),
            "raw_collisions": raw, "clean": len(set(values)) == len(values) and not raw}


def write(experiments_root):
    record = build()
    import stage1_smoke
    screen, oe = SS.load(), OS.load()
    prior = (GS2.prior_integers(experiments_root) | set(GS2.load()["numpy"].values()) | stage1_smoke.sealed_values()
             | set(SS.new_values(screen)) | set(screen["numpy"].values()) | set(oe["numpy"].values()))
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
        raise SystemExit("cf_seeds.json differs from its derivation (or an upstream seed file changed)")
    if not record["audit"]["clean"]:
        raise SystemExit("cf_seeds.json audit is not clean")
    return record


def training_rows(record=None):
    record = record or load()
    return [dict(r) for r in record["training"]]


if __name__ == "__main__":
    print(json.dumps(write(HERE.parents[3]), indent=1))
