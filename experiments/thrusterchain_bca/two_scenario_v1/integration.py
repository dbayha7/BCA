"""Retained tuple binding to the unchanged UniFloral/BCA runtime.

Import is stdlib-only. No training CLI is provided. Model initialization requires
an explicit later, hash-bound training review; setup approval is insufficient.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNTIME_RELATIVE = "experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/runtime.py"
CONDITIONS = ("honeypot", "bottleneck")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_protocol(protocol):
    expected = {"gamma": .98, "horizon": 100, "kappa": 3., "warmup_updates": 10000, "branch_updates": 10000}
    require(all(protocol.get(k) == v for k, v in expected.items()), "Unreviewed learner/evaluation settings")
    seeds = protocol["seeds"]
    fields = ("native_seed", "aux_seed", "branch_native_seed", "branch_aux_seed")
    require(len(seeds) == 3 and all(all(type(s.get(k)) is int and 0 <= s[k] < 2**31 for k in fields) for s in seeds),
            "Three explicit paired initialization/branch seeds are required")
    require(all(len({s[k] for s in seeds}) == 3 for k in fields), "Independent seed streams were repeated")
    return {"conditions": 2, "final_policies": 12, "training_jobs": 18,
            "critic_updates": 180000, "actor_updates": 90000, "auxiliary_updates": 180000,
            "calibrations": 18, "td3_alpha": 2.5, "bc_coefficient": 1., "batch_size": 256,
            "warmup": "native", "branch_objectives": {"native": "Q1", "bca": "Q1 - 3 U"},
            "critic_target": "unchanged native twin-min, target-policy smoothing"}


def _frozen(value, dtype):
    import numpy as np
    return np.frombuffer(np.asarray(value, dtype=dtype).tobytes(), dtype=dtype).reshape(np.shape(value))


def prepare_learner_data(native, records, retained, *, geometry, condition, calibration_seed, dataset_sha256, geometry_sha256):
    """Select intact source tuples, never adjacent retained rows or a fabricated next action.

    The passed mask must equal data.retained_mask(records, geometry, condition).
    This adapter applies that same mask to every learner stream. The caller
    authenticates the geometry file hash; physical validity belongs to setup.
    """
    import numpy as np
    try:
        from . import data as source_data
    except ImportError:
        import data as source_data
    require(Path(source_data.__file__).resolve() == HERE / "data.py", "Wrong raw-data validator imported")
    raw_validation = source_data.validate_arrays(records)
    require(condition in CONDITIONS, "Unknown data condition")
    require(type(calibration_seed) is int and 0 <= calibration_seed < 2**31, "Invalid calibration selection seed")
    require(all(isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
                for s in (dataset_sha256, geometry_sha256)), "Missing exact dataset/geometry identity")
    required = ("observations", "actions", "rewards", "next_observations", "terminals", "episode_ids", "row_ids", "times", "roles")
    require(all(k in records for k in required), "Missing intact transition/provenance field")
    n = len(records["actions"])
    shapes = {"observations": (n, 14), "actions": (n, 4), "rewards": (n,), "next_observations": (n, 14),
              "terminals": (n,), "episode_ids": (n,), "row_ids": (n,), "times": (n,), "roles": (n,)}
    for k, shape in shapes.items():
        a = np.asarray(records[k])
        require(a.shape == shape and a.dtype.kind in "biuf" and np.isfinite(a).all(), "Invalid source array: " + k)
    mask = np.asarray(retained)
    require(mask.dtype == np.bool_ and mask.shape == (n,), "Retention mask must be exact bool[N]")
    require(np.array_equal(mask, source_data.retained_mask(records, geometry, condition)),
            "Retention differs from the declared source/next/crossing geometry mask")
    roles, episodes, rows, times = (np.asarray(records[k]) for k in ("roles", "episode_ids", "row_ids", "times"))
    require(all(a.dtype.kind in "iu" for a in (roles, episodes, rows, times)), "Integer tuple identities/time required")
    require(np.all(np.isin(roles, (0, 1, 2))) and np.all((times >= 0) & (times < 100)), "Unknown role/time")
    require(len(np.unique(rows)) == n and np.all(rows >= 0) and np.all(episodes >= 0), "Repeated/negative source identities")
    require(len(np.unique(np.stack([episodes, times], axis=1), axis=0)) == n, "Repeated parent/time tuple identity")
    require(all(np.asarray(records[k]).dtype == np.float32 for k in ("observations", "next_observations", "actions")),
            "Source deployment precision must remain float32")
    require(np.all(np.isin(records["terminals"], (False, True))) and np.all(np.abs(records["actions"]) <= 1),
            "Source terminal/action values are invalid")
    for ep in np.unique(episodes):
        require(len(np.unique(roles[episodes == ep])) == 1, "A parent episode crosses data roles")
    expected_t = ((100 - times) / 100).astype(np.float32)
    expected_next = ((99 - times) / 100).astype(np.float32)
    require(np.array_equal(np.asarray(records["observations"])[:, -1], expected_t) and
            np.array_equal(np.asarray(records["next_observations"])[:, -1], expected_next), "Original full time was not preserved")
    require(np.all(np.asarray(records["terminals"])[times == 99]), "Finite-horizon final tuples must be terminal")
    train_rows = np.flatnonzero(mask & (roles == 0))
    cal_rows = np.flatnonzero(mask & (roles == 1))
    require(len(train_rows) >= 256, "Fewer than one retained training batch")
    eligible_parents = np.sort(np.unique(episodes[cal_rows]))
    require(len(eligible_parents) >= 128, "Fewer than 128 eligible calibration parents; no mask weakening/backfill")
    rng = np.random.Generator(np.random.PCG64(calibration_seed))
    selected = np.asarray([int(rng.choice(cal_rows[episodes[cal_rows] == ep])) for ep in eligible_parents[:128]], np.int64)
    require(not np.intersect1d(episodes[train_rows], episodes[selected]).size, "Calibration episode leaked into fitting")

    def transition(indices):
        # Explicit original next_observations/terminals survive masked fragment cuts.
        return native.Transition(*(_frozen(np.asarray(records[k])[indices], np.float32)
                                   for k in ("observations", "actions", "rewards", "next_observations", "terminals")))

    first_by_parent = np.asarray([idx[np.argmin(rows[idx])] for ep in np.sort(np.unique(episodes[train_rows]))
                                 for idx in [train_rows[episodes[train_rows] == ep]]], np.int64)
    train = transition(train_rows)
    metadata = {"condition": condition, "dataset_sha256": dataset_sha256, "geometry_sha256": geometry_sha256,
        "geometry_content_sha256": digest(geometry),
        "training_rows": len(train_rows), "training_parents": len(np.unique(episodes[train_rows])),
        "calibration_rows": 128, "calibration_parents": 128, "calibration_seed": calibration_seed,
        "calibration_rule": "first 128 eligible parents sorted by id; one uniform retained tuple per parent",
        "normalization": "all condition-retained fitting observations only; identical native/BCA pair",
        "training_row_ids_sha256": hashlib.sha256(np.asarray(rows[train_rows], dtype='<i8').tobytes()).hexdigest(),
        "calibration_row_ids_sha256": hashlib.sha256(np.asarray(rows[selected], dtype='<i8').tobytes()).hexdigest(),
        "retention_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "complete_raw_parent_validation": raw_validation,
        "next_action_used": False, "fragment_joining": False,
        "calibration_claim": "No shifted-region or selected-action coverage guarantee",
        "freshness": "New task development study; final evaluation is separate and not used here"}
    return SimpleNamespace(train=train, heldout=transition(selected), probe=transition(first_by_parent),
        scale_bank=transition(train_rows[:256]), normalization_fixture=train,
        source_row_ids=_frozen(rows[train_rows], np.int64),
        normalization_source_row_ids=_frozen(rows[train_rows], np.int64),
        heldout_ids={"row_ids": _frozen(rows[selected], np.int64), "episode_ids": _frozen(episodes[selected], np.int64)},
        metadata=metadata, receipt=metadata)


def initialize(runtime, native, protocol, seed_index, condition, prepared, *, training_review=None):
    """Later production binding only. Setup approval does not authorize model creation."""
    validate_protocol(protocol)
    require(training_review is not None and training_review.get("training_authorized") is True and
            training_review.get("protocol_sha256") == digest(protocol) and
            training_review.get("dataset_sha256") == prepared.metadata["dataset_sha256"] and
            training_review.get("geometry_sha256") == prepared.metadata["geometry_sha256"],
            "Training requires the later reviewed geometry/protocol approval")
    require(condition == prepared.metadata["condition"] and condition in CONDITIONS and
            type(seed_index) is int and 0 <= seed_index < 3, "Wrong condition/seed binding")
    receipt = runtime.runtime_receipt(native)
    adapter, state = runtime._initialize_data(protocol, seed_index, condition, native, prepared, receipt)
    require(adapter.context.args.td3_alpha == 2.5 and adapter.context.args.gamma == .98 and
            adapter.context.args.batch_size == 256 and adapter.kappa == 0., "Native settings changed")
    # Correct inherited description only: the numerical normalization fixture is unchanged.
    adapter.context.normalization["source"] = prepared.metadata["normalization"]
    adapter.context.normalization["task_condition"] = condition
    return adapter, state


def fork_pair(runtime, adapter, warm_state, protocol, seed_index):
    """Fresh shared 10k fork; preserve optimizer/targets/reference and matched branch RNG."""
    validate_protocol(protocol)
    before = runtime.tree_digest(warm_state)
    native_adapter, native_state = runtime.branch(adapter, warm_state, protocol, seed_index, "native", 1.)
    bca_adapter, bca_state = runtime.branch(adapter, warm_state, protocol, seed_index, "bca", 1.)
    require(runtime.tree_digest(warm_state) == before, "Warmup mutated while forking")
    require(runtime.tree_digest(native_state) == runtime.tree_digest(bca_state), "Paired full-state fork differs")
    require(native_adapter.kappa == 0. and bca_adapter.kappa == 3. and
            runtime.config(native_adapter.context.args) == runtime.config(bca_adapter.context.args), "Unmatched actor settings")
    return {"native": (native_adapter, native_state), "bca": (bca_adapter, bca_state)}


def restore_bound(runtime, adapter, template, raw, receipt, *, prepared, expected_sha256, protocol, seed_index):
    """Restore only this new task's reviewed checkpoint, never an old study's model."""
    validate_protocol(protocol)
    require(type(seed_index) is int and 0 <= seed_index < 3, "Invalid expected checkpoint seed")
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Checkpoint bytes changed")
    require(receipt.get("dataset_sha256") == prepared.metadata["dataset_sha256"] and
            receipt.get("geometry_sha256") == prepared.metadata["geometry_sha256"] and
            receipt.get("condition") == prepared.metadata["condition"], "Checkpoint belongs to another task/data condition")
    require(receipt.get("protocol_sha256") == digest(protocol) and receipt.get("seed_index") == seed_index,
            "Checkpoint protocol or initialization seed differs")
    require(receipt.get("normalization_sha256") == digest(runtime.host(adapter.context.normalization)),
            "Checkpoint normalization closure differs")
    diagnostic = {}
    restored = runtime.restore_exact(adapter, template, raw, diagnostic=diagnostic)
    require(runtime.tree_digest(restored) == receipt["state_digest"], "Decoded full-state identity differs")
    return restored, diagnostic
