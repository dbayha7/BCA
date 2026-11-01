"""Named seeds of the native screen for candidate knee_s10_v2 (SCREEN_RULE.md). Written before any screen number
exists. Namespace `knee_s10_candidate_v2/native_screen/`; derivation as knee_edge_ood_v3's seeds.py: the first 8
bytes of sha256(namespace + name); numpy seeds & (2**63 - 1); training seeds 1 + word % (2**31 - 2); G1 fit seeds
& (2**32 - 1) (JAX PRNGKey). Audited against every integer on record (the formal G2 v2 scanner), raw collisions,
plus the derived reset-seed check for the data-plan seeds.

- training: three rows of Stage 1's fields (native_seed, aux_seed, branch_native_seed, branch_aux_seed,
  bc_only_seed), one per screen seed index; never the knee-edge sealed training seeds.
- data plan: calibration_original, calibration_replacement, gradient_probes, scale_bank are new; random_control is
  the formal G2 v2's value, copied, so the screen's RANDOM arm is drawn by the same seed D1 used (CUTOFF does not
  depend on it). dataset_entropy is the formal G2 v2's collection entropy (the data are its datasets, by digest).
- g1: three fit seeds for the supervised positive-control unit.
- The confirmation namespace `knee_s10_candidate_v2/confirmation/` is reserved and nothing here draws from it.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
F2 = HERE.parent / "formal_g2"
if str(F2) not in sys.path:
    sys.path.insert(0, str(F2))
import g2_seeds_v2 as GS2                                   # noqa: E402

SEEDS_PATH = HERE / "screen_seeds.json"
NAMESPACE = "knee_s10_candidate_v2/native_screen/"
STAGE1_FIELDS = ("native_seed", "aux_seed", "branch_native_seed", "branch_aux_seed", "bc_only_seed")
NEW_DATA_NAMES = ("calibration_original", "calibration_replacement", "gradient_probes", "scale_bank")
G1_NAMES = ("g1_fit_seed_0", "g1_fit_seed_1", "g1_fit_seed_2")
SEEDS = (0, 1, 2)


def _word(name):
    return int.from_bytes(hashlib.sha256((NAMESPACE + name).encode()).digest()[:8], "big")


def numpy_seed(name):
    return _word(name) & ((1 << 63) - 1)


def training_seed(name):
    return 1 + _word(name) % ((1 << 31) - 2)


def g1_seed(name):
    return _word(name) & ((1 << 32) - 1)


def build():
    v2 = GS2.load()["numpy"]
    data = {"dataset_entropy": v2["dataset_entropy"], "random_control": v2["random_control"]}
    data.update({name: numpy_seed(name) for name in NEW_DATA_NAMES})
    return {"schema": "knee_s10_candidate_v2.native_screen.seeds.v1",
            "derivation": f"first 8 bytes of sha256('{NAMESPACE}' + name); numpy & (2**63 - 1); training 1 + word % (2**31 - 2); "
                          "g1 & (2**32 - 1); dataset_entropy and random_control copied from the formal G2 v2 seeds",
            "numpy": data,
            "training": [{"index": i, **{f: training_seed(f"screen_seed{i}/{f}") for f in STAGE1_FIELDS}} for i in SEEDS],
            "g1": {name: g1_seed(name) for name in G1_NAMES},
            "copied_from_formal_g2_v2": ["dataset_entropy", "random_control"],
            "reserved_untouched_namespace": "knee_s10_candidate_v2/confirmation/"}


def new_values(record):
    """Every value this namespace derived (the copied two are excluded from the collision check)."""
    out = [v for k, v in record["numpy"].items() if k not in record["copied_from_formal_g2_v2"]]
    out += [r[f] for r in record["training"] for f in STAGE1_FIELDS]
    out += list(record["g1"].values())
    return out


def audit(record, prior):
    prior = set(prior)
    values = new_values(record)
    distinct = len(set(values)) == len(values)
    raw = sorted(set(values) & prior)
    return {"prior_integers_scanned": len(prior), "named_values": len(values), "named_values_distinct": distinct,
            "raw_collisions": raw, "note": "raw collisions only: no value here serves as a collection entropy or bank seed",
            "clean": distinct and not raw}


def write(experiments_root, extra_prior=()):
    record = build()
    import stage1_smoke
    prior = GS2.prior_integers(experiments_root) | set(GS2.load()["numpy"].values()) | stage1_smoke.sealed_values() | set(extra_prior)
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
        raise SystemExit("screen_seeds.json differs from its derivation")
    if not record["audit"]["clean"]:
        raise SystemExit("screen_seeds.json audit is not clean")
    return record


def data_seeds(record=None):
    """The keys bindings.plan_streams reads (knee_edge_ood_v3 seeds.DATA_NAMES)."""
    record = record or load()
    return dict(record["numpy"])


def training_rows(record=None):
    record = record or load()
    return [dict(r) for r in record["training"]]


def g1_seeds(record=None):
    record = record or load()
    return [record["g1"][n] for n in G1_NAMES]
