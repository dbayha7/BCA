"""Proof that every learner received the masked arrays (protocol 4.4), and the knee-edge data provider for the
reused Stage 1 pipeline.

Stage 1 reuses the prior pipeline, which has the shadow auxiliary: rerun1 `runtime._initialize_data` and
`make_adapter` (lines 117-215), unchanged and imported read-only. That pipeline samples rows directly from
`context.train_transition`, so the assertions run on the exact arrays it receives, after float32 conversion:
- the plan is built from a validated dataset whose metadata world and physics equal the requested world tag, and
  research data must use the sealed seeds;
- every stream row is traced to its source (episode, t): obs = observations[e, t], next_obs = observations[e, t+1],
  action = actions[e, t], reward = rewards[e, t], done = terminated[e, t];
- for CUTOFF (and for the common calibration and normalization streams), 0 rows in W, by both region implementations;
- the length equals the reconciled retained count, and a sha256 of obs, action, reward, next_obs and done, in that
  fixed order and dtype, equals the plan receipt;
- the adapter's binding is compared with an expectation the driver holds independently (world, arm, dataset digest,
  stream digests), never with a copy of the adapter's own receipt.
"""
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np

import masks as M
import region_independent as second
import seeds as S
import worlds
from core import RERUN1, canonical_json, module, require, sha_file
from knee_region import SOURCE_SHA256 as REGION_SHA256, past_knee_a0

FIELDS = ("obs", "action", "reward", "next_obs", "done")
ARMS = ("cutoff", "random")
SCALE_ROWS = 256
STREAMS = ("train", "heldout", "probe", "scale_bank", "normalization_fixture")
COMMON_STREAMS = ("heldout", "normalization_fixture")


def transition_digest(fields):
    """sha256 over obs, action, reward, next_obs, done in that order, each with its dtype and shape."""
    h = hashlib.sha256()
    for name in FIELDS:
        value = np.ascontiguousarray(np.asarray(fields[name]))
        header = json.dumps([name, value.dtype.str, list(value.shape)]).encode()
        h.update(len(header).to_bytes(8, "big") + header + value.tobytes())
    return h.hexdigest()


def _frozen(value):
    array = np.asarray(value, np.float32)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _rows(arrays, episodes, local_rows):
    """Float32 transitions for local rows (episode ordinal * 100 + t) of the given dataset episodes."""
    obs = np.asarray(arrays["observations"])[episodes]
    intact = {"obs": obs[:, :-1].reshape(-1, 14), "action": np.asarray(arrays["actions"])[episodes].reshape(-1, 4),
              "reward": np.asarray(arrays["rewards"])[episodes].reshape(-1),
              "next_obs": obs[:, 1:].reshape(-1, 14), "done": np.asarray(arrays["terminated"])[episodes].reshape(-1)}
    chosen = {k: _frozen(v[np.asarray(local_rows, np.int64)]) for k, v in intact.items()}
    require(all(bool(np.all(np.isfinite(v))) for v in chosen.values()), "Float32 conversion overflow")
    return chosen


def _source(episodes, local_rows):
    local_rows = np.asarray(local_rows, np.int64)
    return np.stack((np.asarray(episodes)[local_rows // 100], local_rows % 100), axis=1)


def trace(stream, arrays, source):
    """Recompute every row from the dataset by its (episode, t); raises on any offset or substitution."""
    e, t = source[:, 0], source[:, 1]
    obs = np.asarray(arrays["observations"])
    expected = {"obs": obs[e, t], "next_obs": obs[e, t + 1], "action": np.asarray(arrays["actions"])[e, t],
                "reward": np.asarray(arrays["rewards"])[e, t].astype(np.float32),
                "done": np.asarray(arrays["terminated"])[e, t].astype(np.float32)}
    for name in FIELDS:
        require(np.array_equal(np.asarray(stream[name]), expected[name]), f"Row trace failed on {name}")
    require(np.array_equal(np.asarray(stream["done"]) == 1, t == 99), "done must hold exactly at t = 99")


def _one_per_episode(rows, n_episodes, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    out = []
    for e in range(n_episodes):
        candidates = rows[rows // 100 == e]
        require(len(candidates) > 0, "Fitting episode with no retained row; no backfill")
        out.append(int(candidates[rng.integers(len(candidates))]))
    return np.asarray(out, np.int64)


def _seeds_for(dataset, test_seeds):
    if dataset.metadata["provenance"] == "research":
        import dataset_knee
        require(type(dataset) is dataset_knee.KneeDataset, "Research data must be a validated KneeDataset")
        require(test_seeds is None, "Research data use the sealed seeds only")
        record = S.load()
        require(dataset.metadata["seed"] == record["numpy"]["dataset_entropy"], "Dataset entropy is not the sealed one")
        return {k: record["numpy"][k] for k in S.DATA_NAMES}, sha_file(S.SEEDS_PATH)
    require(test_seeds is not None, "Smoke data need explicit test seeds")
    sealed = S.load()["numpy"]
    require(not set(test_seeds.values()) & set(sealed.values()), "Smoke data must not use sealed seeds")
    return dict(test_seeds), None


def plan_streams(dataset, world, *, test_seeds=None):
    """Every stream for both arms of one world, with receipts. Pure: no model, no simulator."""
    require(world in ("stall3", "nostall"), "Training worlds are stall3 and nostall")
    meta = dataset.metadata
    require(meta.get("world") == world, f"Dataset of world {meta.get('world')!r} bound as {world!r}")
    require(meta.get("physics") == worlds.WORLD_DICTS[world], "Dataset physics differ from the world tag")
    seeds, seeds_sha = _seeds_for(dataset, test_seeds)
    arrays = dataset.arrays
    rows = M.fitting_rows(arrays)
    masks, mask_receipt = M.arm_masks(rows["obs"], rows["actions"], rows["times"], rows["episode_ids"],
                                      seed=seeds["random_control"])
    calibration = M.calibration_rows(arrays, seed=seeds["calibration_original"],
                                     replacement_seed=seeds["calibration_replacement"])
    normalization = M.normalization_rows(masks)
    fitting, n_fit = rows["episodes"], len(rows["episodes"])
    trajectory = np.asarray(arrays["trajectory_ids"])
    cal_source = _source(calibration["episodes"], calibration["local_rows"])
    heldout_ids = {"row_ids": trajectory[cal_source[:, 0]] * 100 + cal_source[:, 1],
                   "episode_ids": np.asarray(arrays["group_ids"])[cal_source[:, 0]]}
    plans = {}
    for arm in ARMS:
        retained = np.flatnonzero(masks[arm])
        probe = _one_per_episode(retained, n_fit, seeds["gradient_probes"])
        scale = np.random.Generator(np.random.PCG64(seeds["scale_bank"])).choice(retained, SCALE_ROWS, replace=False)
        local = {"train": (fitting, retained), "heldout": (calibration["episodes"], calibration["local_rows"]),
                 "probe": (fitting, probe), "scale_bank": (fitting, scale), "normalization_fixture": (fitting, normalization)}
        streams, sources = {}, {}
        for name, (episodes, rows_) in local.items():
            streams[name], sources[name] = _rows(arrays, episodes, rows_), _source(episodes, rows_)
            trace(streams[name], arrays, sources[name])
        receipt = {name: {"rows": len(s["obs"]), "sha256": transition_digest(s),
                          "w_rows": int(np.sum(past_knee_a0(s["action"])[0]))} for name, s in streams.items()}
        codes = np.asarray(arrays["split_codes"])
        protected = int(sum(np.isin(codes[src[:, 0]], (2, 3)).sum() for src in sources.values()))
        require(protected == 0, "Selection or final-calibration rows reached a learner stream")
        unused_tuning = set(np.flatnonzero(codes == 1).tolist()) - set(calibration["episodes"].tolist())
        require(not any(np.isin(src[:, 0], list(unused_tuning)).any() for src in sources.values()),
                "Sealed tuning episodes reached a learner stream")
        plans[arm] = {"streams": streams, "sources": sources,
                      "receipt": {"world": world, "arm": arm, "dataset_sha256": dataset.digest, "streams": receipt,
                                  "seeds": seeds, "seeds_json_sha256": seeds_sha, "region_sha256": REGION_SHA256,
                                  "protected_transition_rows_bound_to_learner": protected}}
    return {"plans": plans, "mask_receipt": mask_receipt, "calibration": calibration, "heldout_ids": heldout_ids,
            "normalization_rows": len(normalization), "world": world, "dataset_sha256": dataset.digest}


def assert_stream(fields, expected, arm, name):
    """The 4.4 assertions on the exact arrays a learner receives."""
    arrays = {k: np.asarray(getattr(fields, k) if not isinstance(fields, dict) else fields[k]) for k in FIELDS}
    require(len(arrays["obs"]) == expected["rows"], f"{name}: length differs from the reconciled count")
    require(transition_digest(arrays) == expected["sha256"], f"{name}: array digest differs from the mask receipt")
    primary = past_knee_a0(arrays["action"])[0]
    require(np.array_equal(primary, second.agree(arrays["action"])), f"{name}: region implementations disagree")
    if arm == "cutoff" or name in COMMON_STREAMS:
        require(not bool(np.any(primary)), f"{name}: W rows reached a CUTOFF-visible stream")
    return {"stream": name, "rows": expected["rows"], "sha256": expected["sha256"], "w_rows": int(np.sum(primary))}


def expectation(plan, arm):
    """What the driver holds independently for one binding: copy it before building the adapter."""
    r = plan["plans"][arm]["receipt"]
    return json.loads(canonical_json({"world": r["world"], "arm": arm, "dataset_sha256": r["dataset_sha256"],
                                      "streams": r["streams"]}))


def assert_adapter(adapter, expected):
    """Run the assertions on every stream bound in a freshly made adapter (warmup, branch, BC-only or check)."""
    context = adapter.context
    bound_receipt = context.data.receipt
    for key in ("world", "arm", "dataset_sha256"):
        require(bound_receipt.get(key) == expected[key], f"Adapter bound to another {key}")
    bound = {"train": context.train_transition, "heldout": context.heldout_transition, "probe": context.probe,
             "scale_bank": context.data.scale_bank, "normalization_fixture": context.data.normalization_fixture}
    return [assert_stream(bound[name], expected["streams"][name], expected["arm"], name) for name in STREAMS]


def provide(native, plan, arm):
    """The data namespace rerun1 `_initialize_data` expects, built from a knee-edge plan."""
    p = plan["plans"][arm]
    transitions = {name: native.Transition(**{k: p["streams"][name][k] for k in FIELDS}) for name in STREAMS}
    receipt = dict(p["receipt"])
    return SimpleNamespace(train=transitions["train"], heldout=transitions["heldout"], probe=transitions["probe"],
                           scale_bank=transitions["scale_bank"], normalization_fixture=transitions["normalization_fixture"],
                           heldout_ids={k: np.asarray(v, np.int64) for k, v in plan["heldout_ids"].items()},
                           metadata=receipt, receipt=receipt)


def rerun1_runtime():
    """rerun1 runtime.py, imported read-only (bytecode writing is off; see core.py)."""
    if str(RERUN1) not in sys.path:
        sys.path.insert(0, str(RERUN1))
    return module("_knee_rerun1_runtime", RERUN1 / "runtime.py")


def create(protocol, seed_index, arm, plan, expected):
    """Stage 1 initialization on knee-edge data through the unchanged prior pipeline, with the 4.4 assertions.
    `expected` is the driver's independently held expectation for (world, arm, dataset)."""
    runtime = rerun1_runtime()
    native = runtime.load_native()
    receipt = runtime.runtime_receipt(native)
    data = provide(native, plan, arm)
    adapter, state = runtime._initialize_data(protocol, seed_index, arm, native, data, receipt)
    return adapter, state, assert_adapter(adapter, expected)
