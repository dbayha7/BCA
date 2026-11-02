"""Named seeds of the formal G2 for candidate knee_s10_v2 (OOD definition v4 r3, section 6a step 3: "new entropy,
new streams"). Written before any formal number exists. The derivation is knee_s10_v1's g2_seeds.py with the
namespace, file name and schema changed; the audit scans v1's formal seeds too.

Every seed is derived from its name alone: the first 8 bytes of sha256("knee_s10_candidate_v2/" + name), masked to
63 bits. The namespace differs from knee_edge_ood_v1's, so no value can equal a sealed knee-edge seed by derivation,
and the audit below checks it against every integer on record anyway. Nothing about an outcome can move a seed.

- dataset_entropy: the collection entropy of both worlds (they share it, as the knee-edge twins do);
- random_control: the stratified RANDOM arm's draw seed for masks.arm_masks (D1 reads only the CUTOFF mask);
- banks: g2_e2_anchors (128 anchors, 32 per time in 0 / 25 / 50 / 75; the 96 at t in {0, 25, 50} are gated) and
  g2_e2_bank (128 start states), both under the E2 law (current in [0.9, 1.1]). Item i resets with
  SeedSequence(numpy_seed(bank), spawn_key=(1901, i)); an anchor's recorder prefix draws from spawn_key (1903, i).

The development screens used test entropy 424263 and test stream 707070707; both are in the audit's prior set.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS_PATH = HERE / "g2_seeds_v2.json"
NAMESPACE = "knee_s10_candidate_v2/"
NAMES = ("dataset_entropy", "random_control")
BANK_KEYS = {"reset": 1901, "prefix": 1903}
E2_CURRENT = (0.9, 1.1)
FOUR_T, GATED_T = (0, 25, 50, 75), (0, 25, 50)
BANKS = {"g2_e2_anchors": {"kind": "anchors", "count": 128, "law": "e2", "t_schedule": list(FOUR_T)},
         "g2_e2_bank": {"kind": "starts", "count": 128, "law": "e2"}}
EPISODES = 1280
DEVELOPMENT_VALUES = (424263, 707_070_707, 424242, 777_000_777, 99)      # the development screens' and smoke test's
_ASSIGNMENT = re.compile(r"(?i)(?:seed|entropy)[A-Za-z0-9_\"']*\s*[:=]\s*(\[[^\]]*\]|\([^)]*\)|-?\d+)")


def _word(name):
    return int.from_bytes(hashlib.sha256((NAMESPACE + name).encode()).digest()[:8], "big")


def numpy_seed(name):
    return _word(name) & ((1 << 63) - 1)


def bank_reset_seed(name, item):
    word = np.random.SeedSequence(numpy_seed(name), spawn_key=(BANK_KEYS["reset"], int(item)), pool_size=4)
    return int(word.generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)


def bank_item(name, item):
    """The sealed draw rule of one bank item: reset seed, current range and, for anchors, the time and prefix key."""
    spec = BANKS[name]
    if not 0 <= int(item) < spec["count"]:
        raise ValueError("Item outside the bank")
    out = {"bank": name, "item": int(item), "reset_seed": bank_reset_seed(name, item), "current": tuple(E2_CURRENT)}
    if spec["kind"] == "anchors":
        block = spec["count"] // len(spec["t_schedule"])
        out["t"] = spec["t_schedule"][int(item) // block]
        out["prefix_spawn_key"] = [BANK_KEYS["prefix"], int(item)]
    return out


def gated_items(name="g2_e2_anchors"):
    return [i for i in range(BANKS[name]["count"]) if bank_item(name, i)["t"] in GATED_T]


def prefix_uniforms(name, item, t):
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


def prior_integers(experiments_root):
    """Every integer of four or more digits in a seed/entropy assignment or under a seed-like JSON key, under
    experiments/ but outside this package (the knee-edge seeds.json is inside that scan)."""
    found = set(DEVELOPMENT_VALUES)
    for path in Path(experiments_root).rglob("*"):
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
    out = _derived(record["numpy"]["dataset_entropy"], 1801, EPISODES)
    for name in BANKS:
        out |= {bank_reset_seed(name, i) for i in range(BANKS[name]["count"])}
    return out


def build():
    return {"schema": "knee_s10_candidate_v2.formal_g2.seeds.v1",
            "derivation": f"first 8 bytes of sha256('{NAMESPACE}' + name) & (2**63 - 1)",
            "numpy": {name: numpy_seed(name) for name in NAMES}, "episodes": EPISODES,
            "banks": BANKS, "bank_stream_keys": BANK_KEYS, "gated_times": list(GATED_T),
            "e2_current": list(E2_CURRENT), "worlds_share_dataset_entropy": True}


def audit(record, prior, episodes=1300):
    """Raw and derived collisions of the new seeds against every prior integer (v3 seeds.audit's two checks)."""
    prior = set(prior)
    values = list(record["numpy"].values())
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


def write(experiments_root, extra_prior=()):
    record = build()
    record["audit"] = audit(record, prior_integers(experiments_root) | set(extra_prior))
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
        raise SystemExit("g2_seeds.json differs from its derivation")
    if not record["audit"]["clean"]:
        raise SystemExit("g2_seeds.json audit is not clean")
    return record
