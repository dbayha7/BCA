"""Named seeds for knee_edge_ood_v1, the sealed draw rules of every bank and anchor set, and the collision audit
(protocol 3.7, 3.9, 4.7 "Seeds", Stage 1 "Arms", Stage 2 "Seeds").

Every seed is derived from its name alone, before any data exist: the first 8 bytes of
sha256("knee_edge_ood_v1/" + name), masked to 63 bits for NumPy entropies and reduced to [1, 2**31 - 1) for JAX
training seeds (the prior runtime's seeds fit in int32). Nothing about an outcome can move a seed.

Stage 1 seeds are paired across arms and worlds within a seed index (rows of `training`, the shape rerun1
`runtime.py` reads at lines 136-138 and 226-228). Stage 2 has its own branch_native_seed and branch_aux_seed per
fork, keyed by (seed index, data arm, world) and shared by that fork's BCA, distance and replicate branches.

Bank and anchor draws (`BANKS`): item i of bank `name` resets with
SeedSequence(numpy_seed(name), spawn_key=(1901, i)) (the word & (2**63 - 1), as for datasets); a collection-law
(E1) start draws its mixture uniform from spawn_key (1904, i); an anchor's recorder prefix uses the five-uniform
recorder with spawn_key (1903, i) for its first t steps. Anchor times follow `t_schedule`, in blocks.

The audit compares (i) raw values with every integer in a seed- or entropy-like assignment anywhere under
experiments/ outside this package (every integer of a seed list, and every integer under a seed-like JSON key), and
(ii) derived reset seeds: the dataset's and every bank item's, against those every prior integer would give as an
entropy under stream keys 1701 (core schema), 1801 (this schema) and 1901 (these banks).
"""
import hashlib
import json
import re

import numpy as np

import recorder
from core import HERE, REPO, canonical_json, require, sha_bytes

SEEDS_PATH = HERE / "seeds.json"
BANK_KEYS = {"reset": 1901, "prefix": 1903, "mixture": 1904}
E2_CURRENT, E1 = (0.9, 1.1), "knee_mix15"
FOUR_T, THREE_T = (0, 25, 50, 75), (0, 25, 50)
BANKS = {
    "step04_e2_anchors_t0": {"kind": "anchors", "count": 16, "law": "e2", "t_schedule": [0]},
    "step04_e2_anchors_t25": {"kind": "anchors", "count": 16, "law": "e2", "t_schedule": [25]},
    "step04_e2_anchors_t50": {"kind": "anchors", "count": 16, "law": "e2", "t_schedule": [50]},
    "step04_e2_anchors_t75": {"kind": "anchors", "count": 16, "law": "e2", "t_schedule": [75]},
    "step04_e1_anchors": {"kind": "anchors", "count": 32, "law": "e1", "t_schedule": list(FOUR_T)},
    "stage1_e2_bank": {"kind": "starts", "count": 128, "law": "e2"},
    "stage1_e1_bank": {"kind": "starts", "count": 64, "law": "e1"},
    "stage1_e2_anchors": {"kind": "anchors", "count": 64, "law": "e2", "t_schedule": list(FOUR_T)},
    "stage1_e1_anchors": {"kind": "anchors", "count": 32, "law": "e1", "t_schedule": list(FOUR_T)},
    "r0_e1_starts": {"kind": "starts", "count": 32, "law": "e1"},
    "r0_e2_starts": {"kind": "starts", "count": 32, "law": "e2"},
    "stage2_e2_bank": {"kind": "starts", "count": 128, "law": "e2"},
    "stage2_e1_bank": {"kind": "starts", "count": 64, "law": "e1"},
    "stage2_e2_anchors": {"kind": "anchors", "count": 64, "law": "e2", "t_schedule": list(FOUR_T)},
    "d12_e2_anchors": {"kind": "anchors", "count": 64, "law": "e2", "t_schedule": list(FOUR_T)},
}
SELECTIONS = {"stage1_visited_sample": {"kind": "visited", "per_arm": 48, "t_values": list(THREE_T),
                                        "rule": "16 free episodes per t, without replacement, from PCG64("
                                                "SeedSequence(seed, spawn_key=(1905, t))) over the Stage 1 E2 bank's 128 episodes"}}
DATA_NAMES = ("dataset_entropy", "random_control", "calibration_original", "calibration_replacement",
              "gradient_probes", "scale_bank")
NUMPY_NAMES = DATA_NAMES + tuple(BANKS) + tuple(SELECTIONS)
STAGE1_FIELDS = ("native_seed", "aux_seed", "branch_native_seed", "branch_aux_seed", "bc_only_seed")
FORKS = [(i, arm, world) for i in range(3) for arm in ("cutoff", "random") for world in ("stall3", "nostall")]
_ASSIGNMENT = re.compile(r"(?i)(?:seed|entropy)[A-Za-z0-9_\"']*\s*[:=]\s*(\[[^\]]*\]|\([^)]*\)|-?\d+)")


def _word(name):
    return int.from_bytes(hashlib.sha256(("knee_edge_ood_v1/" + name).encode()).digest()[:8], "big")


def numpy_seed(name):
    return _word(name) & ((1 << 63) - 1)


def training_seed(name):
    return 1 + _word(name) % ((1 << 31) - 2)


def build():
    return {"schema": "knee_edge_ood_v1.seeds.v2",
            "derivation": "first 8 bytes of sha256('knee_edge_ood_v1/' + name); numpy: & (2**63 - 1); "
                          "training: 1 + word % (2**31 - 2)",
            "numpy": {name: numpy_seed(name) for name in NUMPY_NAMES},
            "training": [{"index": i, **{f: training_seed(f"stage1_seed{i}/{f}") for f in STAGE1_FIELDS}} for i in range(3)],
            "stage2_forks": [{"seed_index": i, "arm": arm, "world": world,
                              "branch_native_seed": training_seed(f"stage2_seed{i}/{arm}/{world}/branch_native_seed"),
                              "branch_aux_seed": training_seed(f"stage2_seed{i}/{arm}/{world}/branch_aux_seed")}
                             for i, arm, world in FORKS],
            "banks": BANKS, "bank_stream_keys": BANK_KEYS, "selections": SELECTIONS,
            "e2_current": list(E2_CURRENT), "e1_law": E1,
            "dataset_streams": dict(recorder.STREAM_KEYS), "worlds_share_dataset_entropy": True}


def stage2_protocol_seeds(record, seed_index, arm, world):
    """The per-fork view the Stage 2 worker passes to the frozen branch(): its seeds[seed_index] holds the fork's
    Stage 2 seeds, and every branch of the fork (BCA, distance, replicate) receives the same row."""
    rows = [r for r in record["stage2_forks"] if (r["seed_index"], r["arm"], r["world"]) == (seed_index, arm, world)]
    require(len(rows) == 1, "Unknown fork")
    view = [dict(r) for r in record["training"]]
    view[seed_index] = {**view[seed_index], "branch_native_seed": rows[0]["branch_native_seed"],
                        "branch_aux_seed": rows[0]["branch_aux_seed"]}
    return view


def bank_reset_seed(name, item):
    word = np.random.SeedSequence(numpy_seed(name), spawn_key=(BANK_KEYS["reset"], int(item)), pool_size=4)
    return int(word.generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)


def bank_item(name, item):
    """The sealed draw rule of one bank item: reset seed, current range, anchor time and prefix-draw stream."""
    spec = BANKS[name]
    require(0 <= int(item) < spec["count"], "Item outside the bank")
    if spec["law"] == "e2":
        current = E2_CURRENT
    else:
        mix = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            numpy_seed(name), spawn_key=(BANK_KEYS["mixture"], int(item)), pool_size=4))).random()
        current = recorder.CURRENT_RANGES[int(mix < recorder.DEEP_PROBABILITY)]
    out = {"bank": name, "item": int(item), "reset_seed": bank_reset_seed(name, item), "current": tuple(current)}
    if spec["kind"] == "anchors":
        block = spec["count"] // len(spec["t_schedule"])
        out["t"] = spec["t_schedule"][int(item) // block]
        out["prefix_spawn_key"] = [BANK_KEYS["prefix"], int(item)]
    return out


def prefix_uniforms(name, item, t):
    """The (t, 5) recorder uniforms of an anchor's prefix."""
    seq = np.random.SeedSequence(numpy_seed(name), spawn_key=(BANK_KEYS["prefix"], int(item)), pool_size=4)
    return np.random.Generator(np.random.PCG64(seq)).random((int(t), 5))


def _ints_from_json(value, seedlike=False):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _ints_from_json(v, seedlike or bool(re.search("seed|entropy", str(k), re.I)))
    elif isinstance(value, list):
        for v in value:
            yield from _ints_from_json(v, seedlike)
    elif seedlike and isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 1000:
        yield abs(value)


def prior_integers():
    """Every integer of four or more digits in a seed/entropy assignment or under a seed-like JSON key, under
    experiments/ but outside this package."""
    found = set()
    for path in (REPO / "experiments").rglob("*"):
        if path.suffix not in (".py", ".json", ".md") or HERE in path.parents or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in _ASSIGNMENT.finditer(text):
            found.update(int(x) for x in re.findall(r"\d{4,}", match.group(1)))
        if path.suffix == ".json" and len(text) < 50_000_000:
            try:
                found.update(_ints_from_json(json.loads(text)))
            except ValueError:
                pass
    return found


def _derived(entropy, key, episodes):
    out = set()
    for e in range(episodes):
        word = np.random.SeedSequence(int(entropy), spawn_key=(key, e), pool_size=4).generate_state(1, dtype=np.uint64)[0]
        out.add(int(word) & ((1 << 63) - 1))
    return out


def new_reset_seeds(record):
    out = {recorder.reset_seed(record["numpy"]["dataset_entropy"], e) for e in range(1280)}
    for name in BANKS:
        out |= {bank_reset_seed(name, i) for i in range(BANKS[name]["count"])}
    return out


def audit(record, prior=None, episodes=1300):
    prior = prior_integers() if prior is None else set(prior)
    values = list(record["numpy"].values())
    values += [v for row in record["training"] for k, v in row.items() if k != "index"]
    values += [r[k] for r in record["stage2_forks"] for k in ("branch_native_seed", "branch_aux_seed")]
    distinct = len(set(values)) == len(values)
    raw = sorted(set(values) & prior)
    mine = new_reset_seeds(record)
    derived = set()
    for value in prior:
        for key in (1701, 1801, 1901):
            derived |= _derived(value, key, episodes) & mine
    return {"prior_integers_scanned": len(prior), "named_values": len(values), "named_values_distinct": distinct,
            "raw_collisions": raw, "new_reset_seeds": len(mine), "derived_reset_seed_collisions": sorted(derived),
            "stream_keys_checked": [1701, 1801, 1901], "episodes_per_prior_entropy": episodes,
            "clean": distinct and not raw and not derived}


def write():
    record = build()
    record["audit"] = audit(record)
    require(record["audit"]["clean"], "Seed collision: " + json.dumps(record["audit"]))
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    with SEEDS_PATH.open("xb") as stream:
        stream.write(raw)
    return {"path": str(SEEDS_PATH), "sha256": sha_bytes(raw), "audit": record["audit"]}


def load():
    record = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))
    fresh = build()
    require(canonical_json({k: record[k] for k in fresh}) == canonical_json(fresh), "seeds.json differs from its derivation")
    require(record["audit"]["clean"], "seeds.json audit is not clean")
    return record


if __name__ == "__main__":
    print(json.dumps(write(), indent=2))
