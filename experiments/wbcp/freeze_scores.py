"""Freeze BCA's TD3+BC nonconformity scores on a D4RL population for the WBCP benchmark.

The semi-synthetic benchmark (d4rl_benchmark.py) tilts a fixed population by a known
function of (s, a). Resampling rows never changes a row's own (r, s') or score, so the
shift is pure covariate shift with exact ground truth. This script builds that
population: a frozen model gives every row the score BCA itself would give it,
|y - q| / (max(eta, 1e-6) u). All of it is the repo's own host code: train.prepare and
runtime/td3_bc.py for the data (exact D4RL conversion, effective dependency components,
observation mean/std + 1e-3 fit on the training pool), and algorithms/td3_bc_bca.py for
the BCA arm, its updates, its targets, its critics and its live scale network.

Two modes.

short-train (default): the converted rows are split by whole effective dependency
components with a seeded permutation. The split is the runtime's own reservation, with
the target set to (1 - train_fraction) of the rows and rows_per_episode=None, so every
withheld row is in the population (a population split, not a WBCP calibration bank,
which thins each withheld component to K rows). The BCA arm trains for --updates host
updates on the training part with P.make_train_step inside jax.lax.scan blocks. No refresh
runs: the scale network fits at every update, and the LIVE network and residual unit are
frozen at the end. The other part is then scored. The run also writes
training_log.json, checkpoint_<updates>.msgpack (flax bytes, as in train.py) and
resolved.json (the resolved row with the split reservation), so checkpoint mode can
score the same state again.

checkpoint (--checkpoint RUN_DIR): scores a finished train.py run of td3_bc bca. The
run's resolved row is prepared again with train.prepare and checked against the run's
recorded input hashes. The checkpoint is restored into the P.initialize template and
checked byte for byte. --population all scores every converted row (in_training marks the
run's training pool; withheld rows outside the calibration bank, which the preparation
drops, are rebuilt from the cached file with the runtime's own conversion and transforms);
heldout scores the run's calibration bank only (K rows of each withheld component).

The target noise is one draw for the whole population, in ascending row order. It is
exactly the draw algorithms.td3_bc_bca.refresh makes for a held-out bank of these rows at
key PRNGKey(score_seed): noise key fold_in(PRNGKey(score_seed), 1213156420). That key is
not any refresh event of a train.py run (those use fold_in(PRNGKey(event seed), step)), and
because the draw spans the population, a row's residual depends on the population rule
(heldout vs all); sigma, q and eta do not. Only the chunking of the forward passes
(--score-batch) differs, up to float32 rounding in the last chunk.

Artifact read by d4rl_benchmark.py: <output>/frozen.npz holds score, residual and sigma
(float64), obs, action and policy_action (float32), episode, timestep and row (int64),
and in_training (bool), one entry per population row. <output>/frozen.json (schema
wbcp-frozen-scores-v1) holds provenance, constants, row counts and diagnostics.
--output names a new directory. The script honors JAX_PLATFORMS.

JAX_PLATFORMS=cpu python experiments/wbcp/freeze_scores.py --dataset hopper --output runs/wbcp_frozen/hopper-s202609171
JAX_PLATFORMS=cpu python experiments/wbcp/freeze_scores.py --checkpoint RUN_DIR --output runs/wbcp_frozen/hopper-run
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name, _value in {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "WANDB_MODE": "disabled",
                      "D4RL_SUPPRESS_IMPORT_ERROR": "1", "MUJOCO_GL": "egl"}.items():
    os.environ.setdefault(_name, _value)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from flax import serialization  # noqa: E402

import algorithms.td3_bc_bca as P  # noqa: E402
import runtime.td3_bc as R  # noqa: E402
import train as T  # noqa: E402
from calibration.reference import positive_scale  # noqa: E402
from runtime.config import resolve, typed  # noqa: E402
from runtime.provenance import sha, source_files, write  # noqa: E402
from runtime.validation import checkpoint_counts  # noqa: E402

SCHEMA = "wbcp-frozen-scores-v1"
ALGORITHM = "td3_bc"
CONFIG = ROOT / "configs" / "td3_bc.yaml"
DEFAULT_DATA = Path.home() / ".d4rl" / "datasets"
KEY_REFRESH_NOISE = 1213156420  # algorithms.td3_bc_bca.refresh folds this into its key for target noise
KEY_REFRESH_WBCP = 1347375956  # ... and this for its WBCP draws (recorded only; no threshold is selected here)
MAX_FRACTION = float(np.nextafter(np.float32(1), np.float32(0)))  # largest float32 below one
ARRAYS = {"score": np.float64, "residual": np.float64, "sigma": np.float64, "obs": np.float32,
          "action": np.float32, "policy_action": np.float32, "episode": np.int64,
          "timestep": np.int64, "row": np.int64, "in_training": np.bool_}
OBS_RULE = "current training-observation mean/std + 1e-3"
EPISODE_RULE = ("effective dependency component (runtime.td3_bc.dependency_maps): contiguous raw rows, "
                "broken after terminals and at dropped timeout rows; timestep counts rows from its start")


def declared_seeds():
    return yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf8"))["seeds"]


def resolve_row(dataset, seed, output):
    """The declared BCA row for a td3_bc dataset key (e.g. hopper) and a declared seed."""
    return resolve(CONFIG, "bca", seed, str(Path(output).resolve()), dataset)


def _data_dir(path):
    return Path(path).expanduser().resolve()


def converted_rows(row, data_dir):
    """Row count of runtime.td3_bc._convert on the cached file: with timeouts present (train.prepare
    always reads them) it keeps every raw row but the last, minus the timeout rows."""
    import h5py

    with h5py.File(_data_dir(data_dir) / row["cache"]["filename"], "r") as f:
        timeouts = np.asarray(f["timeouts"][()]).astype(bool)
    return int(np.count_nonzero(~timeouts[:-1]))


def split_row(row, converted, train_fraction, split_seed):
    """Replace the declared reservation by the population target. With rows_per_episode=None the
    runtime reserves whole effective components in seeded-permutation order until (1 - train_fraction)
    of the rows are covered and keeps every withheld row as the held-out pool, the population."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between zero and one")
    population = converted - int(round(train_fraction * converted))
    if not 0 < population < converted:
        raise ValueError("train_fraction leaves one part empty")
    row = copy.deepcopy(row)
    row["protocol"]["reservation"] = dict(target_size=population, seed=int(split_seed),
                                          max_fraction=MAX_FRACTION, dependency_contract=R.DEPENDENCY_CONTRACT,
                                          rows_per_episode=None)
    return row


def prepare(row, data_dir):
    """typed(row) and train.prepare(row, objects): the exact inputs a train.py run of this row sees."""
    os.environ["D4RL_DATASET_DIR"] = str(_data_dir(data_dir))
    objects = typed(row)
    return objects, T.prepare(row, objects)


def train_arm(args, config, prepared, updates, block):
    """Run `updates` BCA host updates on the training pool in jax.lax.scan blocks, as run_prepared does."""
    rng, state, models = P.initialize(args, config, prepared.training.obs.shape[1],
                                      prepared.training.action.shape[1], prepared.max_action)
    R._accept(state, {})
    step = P.make_train_step(args, config, models, prepared.training, prepared.max_action)
    carry, done, log = (rng, state, jnp.int32(0)), 0, []
    while done < updates:
        length = min(block, updates - done)
        started = time.perf_counter()
        proposal, metrics = jax.lax.scan(step, carry, None, length=length)
        proposal = jax.block_until_ready(proposal)
        R._accept(proposal[1], metrics)
        if int(proposal[2]) != done + length:
            raise RuntimeError("an update was rejected: the host step counter fell behind")
        carry, done = proposal, done + length
        log.append(_block_summary(metrics, done - length, done, args.policy_freq, carry[1],
                                  time.perf_counter() - started))
    return carry, models, log


def _block_summary(metrics, first, last, policy_freq, state, seconds):
    m = {name: np.asarray(value) for name, value in metrics.items()}
    actor = np.arange(first + 1, last + 1) % policy_freq == 0  # the host's delayed actor updates

    def mean(name, mask=None):
        values = m[name].astype(np.float64)
        values = values if mask is None else values[mask]
        return float(values.mean()) if values.size else None

    return dict(step=last, seconds=seconds, critic_loss=mean("critic_loss"), actor_loss=mean("actor_loss", actor),
                bc_loss=mean("bc_loss", actor), q_mean=mean("q_mean", actor), scale_loss=mean("scale_loss"),
                scale_fit_accepted=mean("scale_fit_accepted"), bc_multiplier_mean=mean("bc_multiplier_mean"),
                residual_unit=float(state.residual_scale))


def write_checkpoint(output, carry, policy_freq):
    payload = serialization.to_bytes({"state": carry[1], "training_rng": carry[0], "step": carry[2]})
    step = int(carry[2])
    path = Path(output) / f"checkpoint_{step}.msgpack"
    with path.open("xb") as f:
        f.write(payload)
    counters = checkpoint_counts(serialization.msgpack_restore(payload), "td3", step, policy_freq)
    return dict(step=step, path=path.name, sha256=sha(path), size_bytes=len(payload), counters=counters,
                format="flax serialization.to_bytes({'state', 'training_rng', 'step'}), as train.py")


def score_rows(args, models, state, data, key, max_action, batch_size):
    """Score `data` as algorithms.td3_bc_bca.refresh scores a held-out bank at refresh key `key`.

    The target noise is drawn once for all rows, with refresh's key fold and shape. The forward
    passes then run in chunks, and each chunk repeats P.native_target line by line with its own
    slice of that noise. Returns float32 NumPy arrays.
    """
    n, action_dim = data.action.shape
    noise = jax.random.normal(jax.random.fold_in(key, KEY_REFRESH_NOISE), (n, action_dim)) * args.policy_noise
    noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    native = state.native
    parts = {name: [] for name in ("target", "q", "residual", "eta", "sigma", "policy_action")}
    for start in range(0, n, batch_size):
        chunk = jax.tree_util.tree_map(lambda x: x[start:start + batch_size], data)
        action = models[0].apply(native.actor_target.params, chunk.next_obs)
        action = jnp.clip(action + noise[start:start + batch_size], -max_action, max_action)
        q_next = models[1].apply(native.critic_target.params, chunk.next_obs, action).min(axis=-1)
        target = jax.lax.stop_gradient(chunk.reward + (1.0 - chunk.done) * args.discount * q_next)
        q = models[1].apply(native.critic.params, chunk.obs, chunk.action).min(axis=-1)
        eta = models[2].apply(state.calibrator.params, chunk.obs, chunk.action)
        values = dict(target=target, q=q, residual=target - q, eta=eta,
                      sigma=positive_scale(eta, state.residual_scale),
                      policy_action=models[0].apply(native.actor.params, chunk.obs))
        for name, value in values.items():
            parts[name].append(np.asarray(value))
    scored = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    if not all(np.all(np.isfinite(value)) for value in scored.values()) or not np.all(scored["sigma"] > 0):
        raise FloatingPointError("nonfinite score inputs or a nonpositive frozen scale")
    return scored


def components(prepared):
    """Effective dependency component of every converted row and the row's index within it."""
    component = np.asarray(prepared.metadata["dependency_maps"]["effective_components"], np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(component)) + 1]
    if len(starts) != component[-1] + 1:
        raise ValueError("effective components are not consecutive contiguous blocks")
    return component, np.arange(len(component), dtype=np.int64) - starts[component]


def population_arrays(prepared, ids, data, in_training, scored):
    """The frozen.npz contract. Scores are computed as calibration.reference.freeze_reference does."""
    component, timestep = components(prepared)
    residual = scored["residual"].astype(np.float64)
    sigma = scored["sigma"].astype(np.float64)
    arrays = dict(score=np.abs(residual) / sigma, residual=residual, sigma=sigma,
                  obs=np.asarray(data.obs, np.float32), action=np.asarray(data.action, np.float32),
                  policy_action=scored["policy_action"].astype(np.float32), episode=component[ids],
                  timestep=timestep[ids], row=np.asarray(ids, np.int64), in_training=np.asarray(in_training, bool))
    check_contract(arrays)
    return arrays


def check_contract(arrays):
    if set(arrays) != set(ARRAYS):
        raise ValueError("frozen arrays differ from the contract")
    n = len(arrays["row"])
    for name, dtype in ARRAYS.items():
        value = arrays[name]
        if value.dtype != dtype or len(value) != n or value.ndim != (2 if name in ("obs", "action", "policy_action") else 1):
            raise ValueError("contract violation: " + name)
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise FloatingPointError("nonfinite " + name)
    if (n == 0 or arrays["action"].shape != arrays["policy_action"].shape or not np.all(arrays["sigma"] > 0)
            or np.any(np.diff(arrays["row"]) <= 0)
            or not np.array_equal(arrays["score"], np.abs(arrays["residual"]) / arrays["sigma"])):
        raise ValueError("contract violation: rows, scales or scores")


def _diagnostics(arrays, scored):
    score, levels = arrays["score"], (0.5, 0.8, 0.9, 0.95, 0.99)
    gap = arrays["policy_action"].astype(np.float64) - arrays["action"]
    return dict(
        score_mean=float(score.mean()), score_quantiles={str(p): float(np.quantile(score, p)) for p in levels},
        fraction_score_at_most_one=float(np.mean(score <= 1.0)),  # the live scale fit targets 1 - alpha
        residual_mean=float(arrays["residual"].mean()), residual_std=float(arrays["residual"].std()),
        sigma_quantiles={str(p): float(np.quantile(arrays["sigma"], p)) for p in (0.05, 0.5, 0.95)},
        target_mean=float(scored["target"].astype(np.float64).mean()),
        q_mean=float(scored["q"].astype(np.float64).mean()), eta_mean=float(scored["eta"].astype(np.float64).mean()),
        policy_action_rms_gap=float(np.sqrt(np.mean(np.square(gap)))),
        population_episodes=int(np.unique(arrays["episode"]).size),
    )


def git_identity():
    def git(*command):
        return subprocess.run(["git", "-c", "safe.directory=" + ROOT.as_posix(), "-C", str(ROOT), *command],
                              capture_output=True, text=True, check=True,
                              env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout.strip()

    try:
        return dict(branch=git("branch", "--show-current"), commit=git("rev-parse", "HEAD"),
                    dirty=bool(git("status", "--porcelain")))
    except (OSError, subprocess.CalledProcessError) as error:
        return dict(branch=None, commit=None, dirty=None, error=str(error))


def _source_digest(files):
    return R._digest({name: digest for name, digest in files.items() if name != "runtime"})


def build_metadata(*, source, row, args, prepared, arrays, scored, state, score_seed, score_batch,
                   updates, population_rule, checkpoint, timing, **extra):
    raw, m = prepared.metadata["raw_identity"], prepared.metadata
    component, _ = components(prepared)
    episodes = np.asarray(m["original_terminal_timeout_episode_ids"], np.int64)
    refresh_key = jax.random.PRNGKey(score_seed)
    return dict(
        schema=SCHEMA, algorithm=ALGORITHM, method="bca", run_id=row["run_id"], dataset=args.dataset,
        dataset_file=dict(filename=Path(raw["path"]).name, path=raw["path"], size_bytes=raw["size_bytes"],
                          sha256=raw["sha256"], declared_sha256=row["cache"]["sha256"]),
        dataset_sha256=raw["sha256"], source=source, updates=int(updates), seed=args.seed, score_seed=score_seed,
        target_noise=dict(
            rule=("noise = clip(normal(fold_in(PRNGKey(score_seed), 1213156420), (N, action_dim)) * policy_noise, "
                  "-noise_clip, noise_clip): one draw over the population in ascending row order, the draw "
                  "algorithms.td3_bc_bca.refresh makes for a held-out bank of these rows at key PRNGKey(score_seed)"),
            refresh_key=np.asarray(refresh_key, np.uint32), noise_fold=KEY_REFRESH_NOISE,
            noise_key=np.asarray(jax.random.fold_in(refresh_key, KEY_REFRESH_NOISE), np.uint32),
            refresh_wbcp_fold=KEY_REFRESH_WBCP, policy_noise=args.policy_noise, noise_clip=args.noise_clip,
            discount=args.discount, max_action=prepared.max_action, score_batch=score_batch),
        score_definition=dict(
            score="|residual| / sigma in float64 from float32 y - q and sigma, as calibration.reference.freeze_reference",
            residual="y - q", sigma="max(eta, 1e-6) * u (calibration.reference.positive_scale)",
            y=("reward + (1 - done) * discount * min_k Q_target_k(s', clip(pi_target(s') + noise, -max_action, "
               "max_action)) (algorithms.td3_bc_bca.native_target)"),
            q="min over the two critic heads at (obs, action)", eta="live scale network models[2] at (obs, action)",
            u=float(state.residual_scale), policy_action="deterministic actor output at obs"),
        residual_unit=float(state.residual_scale),
        obs_normalization=dict(rule=OBS_RULE, fit=m["normalization_fit"], obs_mean=m["obs_mean"],
                               obs_std=m["obs_std"]),
        population=dict(rule=population_rule, episode_rule=EPISODE_RULE,
                        components_match_raw_terminal_timeout_episodes=bool(
                            np.array_equal(np.diff(component) != 0, np.diff(episodes) != 0))),
        rows=dict(raw=m["raw_rows"], converted=m["converted_rows"], training=len(m["training_converted_ids"]),
                  withheld=len(m["withheld_converted_ids"]), heldout=len(m["heldout_converted_ids"]),
                  population=len(arrays["row"]),
                  population_in_training=int(arrays["in_training"].sum()),
                  population_episodes=int(np.unique(arrays["episode"]).size),
                  converted_components=int(component[-1] + 1)),
        checkpoint=checkpoint,
        preparation=dict(schema=m["schema"], metadata_sha256=prepared.metadata_sha256,
                         settings_sha256=m["settings_sha256"], raw_dataset_sha256=m["raw_dataset_sha256"],
                         run_input_hashes=m["run_input_hashes"], dependency_contract=m["dependency_contract"],
                         reservation=row["protocol"]["reservation"],
                         training_ids_sha256=R._array_hash(np.asarray(prepared.training_ids, np.int64)),
                         source_files_sha256=_source_digest(m["source_files"])),
        configuration=dict(native_args=row["native_args"], bca=row["specification"]["posterior"]),
        arrays={name: dict(dtype=str(value.dtype), shape=list(value.shape)) for name, value in arrays.items()},
        diagnostics=_diagnostics(arrays, scored), git=git_identity(),
        code=dict(script_sha256=sha(Path(__file__).resolve()), source_files_sha256=_source_digest(source_files())),
        timing=timing, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra)


def write_artifact(output, arrays, metadata):
    check_contract(arrays)
    path = Path(output) / "frozen.npz"
    with path.open("xb") as f:
        np.savez(f, **arrays)
    metadata = dict(metadata, npz_sha256=sha(path))
    write(Path(output) / "frozen.json", R._json_value(metadata))
    return metadata


def freeze_short_train(row, output, *, updates=100_000, train_fraction=0.5, split_seed=None, score_seed=None,
                       block=1000, score_batch=65536, data_dir=DEFAULT_DATA):
    """Split, train the BCA arm, freeze its live scale and score the population part."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed = row["protocol"]["seed"]
    split_seed = seed if split_seed is None else split_seed
    score_seed = seed if score_seed is None else score_seed
    started = time.perf_counter()
    converted = converted_rows(row, data_dir)
    row = split_row(row, converted, train_fraction, split_seed)
    (_, args, spec, _), prepared = prepare(row, data_dir)
    if prepared.metadata["converted_rows"] != converted:
        raise RuntimeError("converted row count differs from the timeout rule")
    if prepared.metadata["withheld_converted_ids"] != prepared.metadata["heldout_converted_ids"]:
        raise RuntimeError("the population must be every withheld row")
    prepared_at = time.perf_counter()
    carry, models, log = train_arm(args, spec.config(), prepared, updates, block)
    trained_at = time.perf_counter()
    state = carry[1]
    checkpoint = write_checkpoint(output, carry, args.policy_freq)
    ids = np.asarray(prepared.heldout_ids, np.int64)
    scored = score_rows(args, models, state, prepared.heldout, jax.random.PRNGKey(score_seed),
                        prepared.max_action, score_batch)
    arrays = population_arrays(prepared, ids, prepared.heldout, np.zeros(len(ids), bool), scored)
    scored_at = time.perf_counter()
    # Throughput after the first block, which carries the compilation.
    steady_rate = (log[-1]["step"] - log[0]["step"]) / sum(e["seconds"] for e in log[1:]) if len(log) > 1 else None
    timing = dict(prepare_seconds=prepared_at - started, training_seconds=trained_at - prepared_at,
                  first_block_seconds_with_compile=log[0]["seconds"], steady_updates_per_second=steady_rate,
                  projected_seconds_per_100k_updates=None if steady_rate is None else 1e5 / steady_rate,
                  scoring_seconds=scored_at - trained_at)
    write(output / "resolved.json", row)
    write(output / "training_log.json", R._json_value(dict(
        schema="wbcp-frozen-training-log-v1", updates=updates, block=block, policy_freq=args.policy_freq,
        refreshes=0, bc_multiplier="1 throughout: no refresh, so the frozen reference never becomes ready",
        blocks=log, residual_unit_final=float(state.residual_scale), **timing)))
    split = dict(train_fraction=train_fraction, split_seed=split_seed,
                 rule=("runtime reservation (calibration.reference.reserve_calibration via runtime.td3_bc.prepare) "
                       "with rows_per_episode=None: whole effective components in "
                       "np.random.default_rng(split_seed).permutation order until (1 - train_fraction) of the "
                       "converted rows are covered; every row of them forms the population, the complement the "
                       "training pool"))
    metadata = build_metadata(
        source="short-train", row=row, args=args, prepared=prepared, arrays=arrays, scored=scored, state=state,
        score_seed=score_seed, score_batch=score_batch, updates=updates, checkpoint=checkpoint,
        population_rule="short-train population: every row of the reserved part, none used in training",
        timing=dict(timing, total_seconds=time.perf_counter() - started), split=split,
        training=dict(block=block, refreshes=0, scale_network="fit at every update; frozen live at the end",
                      log="training_log.json"))
    metadata = write_artifact(output, arrays, metadata)
    return dict(arrays=arrays, metadata=metadata, state=state, models=models, population=prepared.heldout,
                prepared=prepared, args=args, config=spec.config(), row=row, training_rng=carry[0])


def verify_preparation(run_dir, prepared):
    """Compare the new preparation with the run's record: train.py's preparation.json or a short-train frozen.json."""
    current = prepared.metadata
    if (run_dir / "preparation.json").is_file():
        saved = json.loads((run_dir / "preparation.json").read_text(encoding="utf8"))["metadata"]
        record, recorded = "preparation.json", saved["run_input_hashes"]
        changed = sorted(name for name in set(saved["source_files"]) | set(current["source_files"])
                         if name != "runtime" and saved["source_files"].get(name) != current["source_files"].get(name))
    elif (run_dir / "frozen.json").is_file():
        saved = json.loads((run_dir / "frozen.json").read_text(encoding="utf8"))["preparation"]
        record, recorded = "frozen.json", saved["run_input_hashes"]
        changed = None if saved["source_files_sha256"] == _source_digest(current["source_files"]) else "changed"
    else:
        raise FileNotFoundError("RUN_DIR has no preparation.json or frozen.json to verify the preparation against")
    if recorded != current["run_input_hashes"]:
        raise ValueError("the new preparation differs from the run's recorded inputs")
    return dict(record=record, run_input_hashes_match=True, source_files_changed_since_run=changed)


def locate_checkpoint(run_dir, step=None):
    found = {int(path.stem[len("checkpoint_"):]): path for path in run_dir.glob("checkpoint_*.msgpack")
             if path.stem[len("checkpoint_"):].isdigit()}
    if not found:
        raise FileNotFoundError("no checkpoint_<step>.msgpack in " + str(run_dir))
    step = max(found) if step is None else step
    if step not in found:
        raise FileNotFoundError(f"no checkpoint at step {step}; found {sorted(found)}")
    recorded = None
    if (run_dir / "result.json").is_file():
        entries = json.loads((run_dir / "result.json").read_text(encoding="utf8")).get("checkpoints", [])
        recorded = next((entry["sha256"] for entry in entries if entry["step"] == step), None)
    elif (run_dir / "frozen.json").is_file():
        entry = json.loads((run_dir / "frozen.json").read_text(encoding="utf8")).get("checkpoint") or {}
        recorded = entry.get("sha256") if entry.get("step") == step else None
    return step, found[step], recorded


def same_leaves(payload, restored):
    """Leaf-for-leaf identity of a checkpoint and a restored tree: paths, dtypes, shapes and bytes.
    Dict key order is not compared: fresh flax initializers order parameters (kernel, bias) while
    states that went through jax.lax.scan come back sorted, so whole-byte equality is too strict."""
    flat = [jax.tree_util.tree_flatten_with_path(serialization.msgpack_restore(value))[0]
            for value in (payload, serialization.to_bytes(restored))]
    return len(flat[0]) == len(flat[1]) and all(
        path_a == path_b and np.asarray(a).dtype == np.asarray(b).dtype and np.shape(a) == np.shape(b)
        and np.asarray(a).tobytes() == np.asarray(b).tobytes() for (path_a, a), (path_b, b) in zip(*flat))


def all_rows(prepared, args):
    """Every converted row in order. The prepared pools hold only the training rows and the
    calibration bank, so the rows are rebuilt from the prepared raw file with the runtime's own
    conversion and transforms, then checked against the recorded raw and converted array hashes,
    the normalization and both pools."""
    import h5py

    m = prepared.metadata
    path = Path(m["raw_identity"]["path"])
    if sha(path) != m["raw_identity"]["sha256"]:
        raise ValueError("the cached file differs from the prepared raw identity")
    with h5py.File(path, "r") as f:
        raw = {name: f[name][()] for name in m["raw_array_hashes"]}
    if {name: R._array_hash(value) for name, value in raw.items()} != m["raw_array_hashes"]:
        raise ValueError("raw arrays differ from the prepared raw identity")
    converted, _, _ = R._convert(raw, prepared.max_episode_steps)
    if {name: R._array_hash(value) for name, value in converted.items()} != m["converted_array_hashes"]:
        raise ValueError("converted arrays differ from the preparation")
    data, mean, std = R.transform_rows(converted, args, np.asarray(prepared.training_ids), prepared.max_episode_steps)
    take = lambda ids: jax.tree_util.tree_map(lambda x: x[np.asarray(ids)], data)
    if (not np.array_equal(mean, prepared.obs_mean) or not np.array_equal(std, prepared.obs_std)
            or R._tree_hash(take(prepared.training_ids)) != R._tree_hash(prepared.training)
            or R._tree_hash(take(prepared.heldout_ids)) != R._tree_hash(prepared.heldout)):
        raise ValueError("rebuilt rows differ from the prepared pools")
    return data


def freeze_checkpoint(run_dir, output, *, step=None, population="all", score_seed=None, score_batch=65536,
                      data_dir=DEFAULT_DATA):
    """Prepare the run's resolved row again, restore its checkpoint and score the chosen population."""
    if population not in ("all", "heldout"):
        raise ValueError("population is all or heldout")
    run_dir = Path(run_dir).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    row = json.loads((run_dir / "resolved.json").read_text(encoding="utf8"))
    if (row.get("algorithm"), row.get("method")) != (ALGORITHM, "bca"):
        raise ValueError("checkpoint mode scores td3_bc bca runs only")
    score_seed = row["protocol"]["seed"] if score_seed is None else score_seed
    step, path, recorded_sha = locate_checkpoint(run_dir, step)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if recorded_sha is not None and digest != recorded_sha:
        raise ValueError("checkpoint bytes differ from the run's recorded sha256")
    (_, args, spec, _), prepared = prepare(row, data_dir)
    verification = verify_preparation(run_dir, prepared)
    prepared_at = time.perf_counter()
    config = spec.config()
    rng, template, models = P.initialize(args, config, prepared.training.obs.shape[1],
                                         prepared.training.action.shape[1], prepared.max_action)
    counters = checkpoint_counts(serialization.msgpack_restore(payload), "td3", step, args.policy_freq)
    restored = serialization.from_bytes({"state": template, "training_rng": rng, "step": jnp.int32(0)}, payload)
    if int(restored["step"]) != step or not same_leaves(payload, restored):
        raise ValueError("the checkpoint did not restore exactly into the P.initialize template")
    state = restored["state"]
    R._accept(state, {})
    training_ids = np.asarray(prepared.training_ids, np.int64)
    if population == "heldout":
        ids, data = np.asarray(prepared.heldout_ids, np.int64), prepared.heldout
    else:
        ids, data = np.arange(prepared.metadata["converted_rows"], dtype=np.int64), all_rows(prepared, args)
    scored = score_rows(args, models, state, data, jax.random.PRNGKey(score_seed), prepared.max_action, score_batch)
    arrays = population_arrays(prepared, ids, data, np.isin(ids, training_ids), scored)
    checkpoint = dict(step=step, path=str(path), sha256=digest, recorded_sha256=recorded_sha,
                      size_bytes=len(payload), counters=counters, restored_leaves_identical=True,
                      posterior_ready=bool(state.posterior.ready), posterior_threshold=float(state.posterior.threshold))
    rule = ("every converted row; in_training marks the run's training pool" if population == "all"
            else "the run's held-out calibration bank (a subset of its withheld rows); none of it in the training pool")
    metadata = build_metadata(
        source="checkpoint", row=row, args=args, prepared=prepared, arrays=arrays, scored=scored, state=state,
        score_seed=score_seed, score_batch=score_batch, updates=step, checkpoint=checkpoint,
        population_rule=rule, run_dir=str(run_dir), population_choice=population, verification=verification,
        timing=dict(prepare_seconds=prepared_at - started, total_seconds=time.perf_counter() - started),
        scale="live scale network and residual unit of the checkpoint state (not the frozen reference)")
    metadata = write_artifact(output, arrays, metadata)
    return dict(arrays=arrays, metadata=metadata, state=state, models=models, population=data,
                prepared=prepared, args=args, config=config, row=row, restored=restored)


def summary(metadata):
    keys = ("source", "dataset", "updates", "seed", "score_seed", "rows", "residual_unit", "diagnostics", "timing")
    return dict({key: metadata[key] for key in keys}, checkpoint_sha256=metadata["checkpoint"]["sha256"],
                npz_sha256=metadata["npz_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=("short-train", "checkpoint"),
                        help="default: checkpoint when --checkpoint is given, else short-train")
    parser.add_argument("--checkpoint", type=Path, help="train.py run directory of td3_bc bca (checkpoint mode)")
    parser.add_argument("--dataset", help="td3_bc dataset key in configs/td3_bc.yaml (short-train; default hopper)")
    parser.add_argument("--seed", type=int, help="declared seed from configs/experiment.yaml (short-train)")
    parser.add_argument("--updates", type=int, help="host updates (short-train; default 100000)")
    parser.add_argument("--train-fraction", type=float, help="training share of the rows (short-train; default 0.5)")
    parser.add_argument("--split-seed", type=int, help="component permutation seed (short-train; default --seed)")
    parser.add_argument("--block", type=int, help="host updates per jax.lax.scan block (short-train; default 1000)")
    parser.add_argument("--step", type=int, help="checkpoint step (checkpoint mode; default the latest)")
    parser.add_argument("--population", choices=("all", "heldout"), help="rows to score (checkpoint mode; default all)")
    parser.add_argument("--score-seed", type=int, help="refresh key seed for target noise (default the run seed)")
    parser.add_argument("--score-batch", type=int, default=65536, help="rows per forward-pass chunk")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="cached D4RL HDF5 directory")
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory; must not exist")
    opt = parser.parse_args(argv)
    mode = opt.mode or ("checkpoint" if opt.checkpoint is not None else "short-train")
    if opt.output.exists():
        parser.error("--output already exists")
    short_only = ("dataset", "seed", "updates", "train_fraction", "split_seed", "block")
    if mode == "checkpoint":
        if opt.checkpoint is None:
            parser.error("checkpoint mode needs --checkpoint RUN_DIR")
        given = [name for name in short_only if getattr(opt, name) is not None]
        if given:
            parser.error("short-train options in checkpoint mode: " + ", ".join(given))
    else:
        if opt.checkpoint is not None or opt.step is not None or opt.population is not None:
            parser.error("--checkpoint, --step and --population apply to checkpoint mode only")
        opt.dataset = opt.dataset or "hopper"
        opt.seed = declared_seeds()[0] if opt.seed is None else opt.seed
        opt.updates = 100_000 if opt.updates is None else opt.updates
        opt.train_fraction = 0.5 if opt.train_fraction is None else opt.train_fraction
        opt.block = 1000 if opt.block is None else opt.block
        if opt.updates < 1 or opt.block < 1 or not 0 < opt.train_fraction < 1:
            parser.error("--updates and --block must be positive and --train-fraction inside (0, 1)")
        if opt.seed not in declared_seeds():
            parser.error("--seed must be one of the declared seeds " + str(declared_seeds()))
    for name in ("split_seed", "score_seed"):
        value = getattr(opt, name)
        if value is not None and not 0 <= value < 2**32:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 2**32)")
    if opt.score_batch < 1:
        parser.error("--score-batch must be positive")
    try:
        if mode == "short-train":
            result = freeze_short_train(
                resolve_row(opt.dataset, opt.seed, opt.output), opt.output, updates=opt.updates,
                train_fraction=opt.train_fraction, split_seed=opt.split_seed, score_seed=opt.score_seed,
                block=opt.block, score_batch=opt.score_batch, data_dir=opt.data_dir)
        else:
            result = freeze_checkpoint(opt.checkpoint, opt.output, step=opt.step, population=opt.population or "all",
                                       score_seed=opt.score_seed, score_batch=opt.score_batch, data_dir=opt.data_dir)
    except BaseException as error:
        if opt.output.is_dir() and not (opt.output / "failure.json").exists():
            write(opt.output / "failure.json", dict(type=type(error).__name__, message=str(error),
                                                    traceback=traceback.format_exc()))
        raise
    print(json.dumps(R._json_value(summary(result["metadata"])), indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
