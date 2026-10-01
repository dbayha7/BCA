"""Freeze BCA's ReBRAC nonconformity scores on a D4RL population for the WBCP benchmark.

The ReBRAC counterpart of freeze_scores.py's short-train mode. It writes the same artifact
(frozen.npz + frozen.json, schema wbcp-frozen-scores-v1), so d4rl_benchmark.load_pool,
d4rl_benchmark.py and dependence.py read it unchanged. All of it is the repo's own host
code: train.prepare and runtime/rebrac.py for the data (exact D4RL next-action conversion,
effective dependency components, ReBRAC's reward handling and its own state normalization
setting), and algorithms/rebrac_bca.py for the BCA arm, its updates, its targets, its
critics and its live scale network.

Split. The declared rebrac bca row's reservation is replaced by freeze_scores.py's
population split: the same target, (1 - train_fraction) of the converted rows
(freeze_scores.split_row, imported), the same seed and the same rule, whole effective
components in np.random.default_rng(split_seed).permutation order until the target is
covered (calibration.reference.reserve_calibration with rows_per_episode=None, reached
through algorithms.rebrac_bca.reserve_pool and runtime.rebrac._selection). runtime.rebrac
converts and labels rows exactly as runtime.td3_bc does (same timeout rule, same effective
components), so for a given split seed the population rows are TD3+BC's. The reservation
is a runtime.rebrac.PopulationSplit, the only type through which runtime.rebrac accepts
rows_per_episode=None (runtime.rebrac.run_prepared refuses it), and every withheld row is
the population.

Training. The BCA arm runs --updates host updates on the training part with
P.make_train_step inside jax.lax.scan blocks, as runtime.rebrac.run_prepared does (after
runtime.rebrac.validate_prepared, as there). No refresh runs, so the frozen reference never
becomes ready and every actor BC multiplier stays 1 (native ReBRAC) while the scale network
sigma(s, a) is fit at every update. The LIVE network and residual unit are frozen at the end
and the population is scored as algorithms.rebrac_bca.refresh scores a held-out bank
(score_rows).

Target noise: one draw over the population in ascending row order, exactly the draw refresh
makes for a held-out bank of these rows at key PRNGKey(score_seed): noise =
clip(normal(fold_in(PRNGKey(score_seed), 1380470604), (N, action_dim)) * policy_noise,
-noise_clip, noise_clip). The target's critic BC penalty uses each row's recorded next
action (TransitionNA.next_action), as the host's does.

obs field: the population's observations normalized by TD3+BC's rule (training-pool
mean/std + 1e-3, benchmark_obs), the field every other host's artifact holds, so the
benchmark's state tilt is the same function of s for every host. ReBRAC's own networks read
raw observations (normalize_states=False) and are scored on them.

Outputs in --output (a new directory): frozen.npz, frozen.json, resolved.json (the row with
the split), training_log.json and checkpoint_<updates>.msgpack (flax bytes, as train.py).
The script honors JAX_PLATFORMS.

JAX_PLATFORMS=cpu python experiments/wbcp/freeze_rebrac.py --dataset hopper --output runs/wbcp_frozen/rebrac-hopper-s202609171
"""

import argparse
import copy
import dataclasses
import json
import os
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
from flax import serialization  # noqa: E402

import algorithms.rebrac_bca as P  # noqa: E402
import runtime.rebrac as R  # noqa: E402
import train as T  # noqa: E402
from calibration.reference import positive_scale  # noqa: E402
from experiments.wbcp import freeze_scores as F  # noqa: E402
from runtime.config import read_config, resolve, typed  # noqa: E402
from runtime.provenance import sha, source_files, write  # noqa: E402
from runtime.validation import checkpoint_counts  # noqa: E402

ALGORITHM = "rebrac"
CONFIG = ROOT / "configs" / "rebrac.yaml"
DEFAULT_DATA = F.DEFAULT_DATA
KEY_REFRESH_NOISE = P.KEY_REFRESH_NOISE  # algorithms.rebrac_bca.refresh folds this into its key for target noise
KEY_REFRESH_WBCP = P.KEY_POSTERIOR  # ... and this for its WBCP draws (recorded only; no threshold is selected here)
EPISODE_RULE = ("effective dependency component (runtime.rebrac.dependency_maps): contiguous raw rows, broken after "
                "terminals and at dropped timeout rows; timestep counts rows from its start. runtime.rebrac converts "
                "and breaks rows exactly as runtime.td3_bc does, so the labels are TD3+BC's")


def resolve_row(dataset, seed, output):
    """The declared BCA row for a rebrac dataset key (e.g. hopper) and a declared seed."""
    return resolve(CONFIG, "bca", seed, str(Path(output).resolve()), dataset)


def split_row(row, converted, train_fraction, split_seed):
    """freeze_scores.split_row's population reservation (target, seed, cap, rows_per_episode=None)
    with ReBRAC's dependency contract, so the target and rule are TD3+BC's by construction."""
    reservation = F.split_row(row, converted, train_fraction, split_seed)["protocol"]["reservation"]
    row = copy.deepcopy(row)
    row["protocol"]["reservation"] = dict(reservation, dependency_contract=R.DEPENDENCY_CONTRACT)
    return row


def typed_population(row):
    """typed(row) with the reservation as runtime.rebrac.PopulationSplit.

    runtime.config.typed builds runtime.rebrac.Reservation, which refuses rows_per_episode=None,
    so the row is typed with K = 1 and the protocol is rebuilt with the split's own fields.
    """
    probe = copy.deepcopy(row)
    probe["protocol"]["reservation"]["rows_per_episode"] = 1
    runner, args, spec, protocol = typed(probe)
    protocol = dataclasses.replace(protocol, reservation=R.PopulationSplit(**row["protocol"]["reservation"]))
    R.validate_protocol(args, spec, protocol)
    return runner, args, spec, protocol


def prepare(row, data_dir):
    """typed_population(row) and train.prepare(row, objects): the inputs a train.py run would see."""
    os.environ["D4RL_DATASET_DIR"] = str(F._data_dir(data_dir))
    objects = typed_population(row)
    return objects, T.prepare(row, objects)


def train_arm(args, config, prepared, updates, block):
    """Run `updates` BCA host updates on the training pool in jax.lax.scan blocks, as run_prepared does."""
    rng, state, models = P.initialize(args, config, prepared.training)
    R._accept(state, {}, models)
    if R._tree_hash((state.cal_obs_mean, state.cal_obs_std)) != R._tree_hash(
            (prepared.cal_obs_mean, prepared.cal_obs_std)):
        raise ValueError("initialized calibrator statistics differ from preparation")
    step = P.make_train_step(args, config, models, prepared.training)
    carry, done, log = (rng, state, jnp.int32(0)), 0, []
    while done < updates:
        length = min(block, updates - done)
        started = time.perf_counter()
        proposal, metrics = jax.lax.scan(step, carry, None, length=length)
        proposal = jax.block_until_ready(proposal)
        R._accept(proposal[1], metrics, models)
        if int(proposal[2]) != done + length:
            raise RuntimeError("an update was rejected: the host step counter fell behind")
        carry, done = proposal, done + length
        log.append(_block_summary(metrics, done - length, done, args.policy_freq, carry[1],
                                  time.perf_counter() - started))
    return carry, models, log


def _block_summary(metrics, first, last, policy_freq, state, seconds):
    m = {name: np.asarray(value).astype(np.float64) for name, value in metrics.items()}
    actor = np.arange(first, last) % policy_freq == 0  # ReBRAC updates the actor when the pre-update counter is 0 mod policy_freq

    def mean(name, mask=None):
        if name not in m:
            return None
        values = m[name] if mask is None else m[name][mask]
        return float(values.mean()) if values.size else None

    return dict(step=last, seconds=seconds, critic_loss=mean("critic_loss"), q_min=mean("q_min"),
                actor_loss=mean("actor_loss", actor), bc_mse_policy=mean("bc_mse_policy", actor),
                action_mse=mean("action_mse", actor), scale_loss=mean("scale_loss"),
                scale_fit_accepted=mean("scale_fit_accepted"), bc_multiplier_mean=mean("bc_multiplier_mean"),
                posterior_ready=mean("posterior_ready"), residual_unit=float(state.residual_scale))


def write_checkpoint(output, carry, policy_freq):
    payload = serialization.to_bytes({"state": carry[1], "training_rng": carry[0], "step": carry[2]})
    step = int(carry[2])
    path = Path(output) / f"checkpoint_{step}.msgpack"
    with path.open("xb") as f:
        f.write(payload)
    counters = checkpoint_counts(serialization.msgpack_restore(payload), "rebrac", step, policy_freq)
    return dict(step=step, path=path.name, sha256=sha(path), size_bytes=len(payload), counters=counters,
                format="flax serialization.to_bytes({'state', 'training_rng', 'step'}), as train.py")


def score_rows(args, models, state, data, key, max_action, batch_size):
    """Score `data` as algorithms.rebrac_bca.refresh scores a held-out bank at refresh key `key`.

    refresh (algorithms/rebrac_bca.py) computes
        target = native_target(args, models, state.native, heldout, fold_in(key, 1380470604)),
        q = models[1].apply(state.native.critic.params, obs, action).min(0),
        cal = models[2].apply(state.calibrator.params, obs, action),
    and calibration.reference.freeze_reference scores |target - q| / positive_scale(cal, residual_scale)
    in float64. native_target (algorithms/rebrac_bca.py) draws its target-policy noise once for the
    whole bank, shape (N, action_dim), clips the noisy target action to the literal unit bound and
    subtracts critic_bc_coef * ||a' - next_action||^2, next_action being the row's recorded next
    action. Here that draw is made once for all rows and the forward passes run in chunks of
    batch_size rows, each repeating native_target line by line with its slice of the draw; with one
    chunk this is refresh's computation exactly. refresh's validity conditions (finite transitions
    within the unit bounds, valid calibration storage, finite targets, positive scale outputs) are
    enforced by raising.

    policy_action is the deterministic actor at obs, as runtime.td3_bc.evaluate_episodes (ReBRAC's
    evaluator) acts: tanh bounds it, the host applies no clip and refuses |a| > max_action, so the
    same bound is checked here. Returns float32 NumPy arrays.
    """
    if max_action != 1.0:
        raise ValueError("native ReBRAC acts in the literal unit box")
    if not bool(P.transition_valid(data)) or not bool(P.calibration_storage_valid(models, state)):
        raise ValueError("population rows or calibration storage fail refresh's validity checks")
    n, action_dim = data.action.shape
    noise = jnp.clip(jax.random.normal(jax.random.fold_in(key, KEY_REFRESH_NOISE), (n, action_dim)) * args.policy_noise,
                     -args.noise_clip, args.noise_clip)
    native = state.native
    parts = {name: [] for name in ("target", "q", "residual", "eta", "sigma", "policy_action")}
    for start in range(0, n, batch_size):
        batch = jax.tree_util.tree_map(lambda x: x[start:start + batch_size], data)
        # algorithms.rebrac_bca.native_target, line by line, with this chunk's slice of the noise
        action = models[0].apply(native.actor.target_params, batch.next_obs)
        action = jnp.clip(action + noise[start:start + batch_size], -1, 1)
        penalty = ((action - batch.next_action) ** 2).sum(-1)
        q_next = models[1].apply(native.critic.target_params, batch.next_obs, action).min(0)
        q_next = q_next - args.critic_bc_coef * penalty
        target = jax.lax.stop_gradient(batch.reward + (1 - batch.done) * args.gamma * q_next)
        # algorithms.rebrac_bca.refresh
        q = models[1].apply(native.critic.params, batch.obs, batch.action).min(0)
        eta = models[2].apply(state.calibrator.params, batch.obs, batch.action)
        values = dict(target=target, q=q, residual=target - q, eta=eta,
                      sigma=positive_scale(eta, state.residual_scale),
                      policy_action=models[0].apply(native.actor.params, batch.obs))
        for name, value in values.items():
            parts[name].append(np.asarray(value))
    scored = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    if (not all(np.all(np.isfinite(value)) for value in scored.values()) or not np.all(scored["eta"] > 0)
            or not np.all(scored["sigma"] > 0)):
        raise FloatingPointError("nonfinite score inputs or a nonpositive frozen scale")
    if np.any(np.abs(scored["policy_action"]) > max_action):
        raise ValueError("deterministic action bound violation; the host does not repair actions")
    return scored


def benchmark_obs(prepared, ids, normalize_states):
    """The artifact's obs field: the population's observations under freeze_scores.py's rule
    (F.OBS_RULE, runtime.common.compute_mean_std(training rows, eps=1e-3) and normalize_states).

    obs is only the benchmark's tilt input (d4rl_benchmark.tilt_feature), and its state tilt,
    ||obs||, changes with how obs are scaled. ReBRAC's declared normalize_states=False means its
    networks read raw observations, but with TD3+BC's rule on the same training rows the field
    is the one TD3+BC's (and CQL's and IQL's) artifacts hold, so every host is tilted by the same
    function of s. Scores, sigma and policy_action still use the host's own inputs.
    Returns (obs, mean, std).
    """
    obs = np.asarray(prepared.converted["observations"], np.float32)
    mean, std = R.C.compute_mean_std(obs[np.asarray(prepared.training_ids)], eps=0.001)
    normalized = np.asarray(R.C.normalize_states(obs[ids], mean, std), np.float32)
    if normalize_states and not np.array_equal(normalized, np.asarray(prepared.heldout.obs)):
        raise ValueError("the host's own state normalization differs from freeze_scores.py's rule")
    return normalized, mean, std


def build_metadata(*, row, args, prepared, arrays, scored, state, score_seed, score_batch, updates, checkpoint,
                   split, timing, training, obs_mean, obs_std):
    m, raw = prepared.metadata, prepared.metadata["raw_identity"]
    component, _ = F.components(prepared)
    episodes = np.asarray(m["original_terminal_timeout_episode_ids"], np.int64)
    refresh_key = jax.random.PRNGKey(score_seed)
    return dict(
        schema=F.SCHEMA, algorithm=ALGORITHM, host="rebrac", method="bca", run_id=row["run_id"], dataset=args.dataset,
        dataset_file=dict(filename=Path(raw["path"]).name, path=raw["path"], size_bytes=raw["size_bytes"],
                          sha256=raw["sha256"], declared_sha256=row["cache"]["sha256"]),
        dataset_sha256=raw["sha256"], source="short-train", updates=int(updates), seed=args.seed, score_seed=score_seed,
        target_noise=dict(
            rule=("noise = clip(normal(fold_in(PRNGKey(score_seed), 1380470604), (N, action_dim)) * policy_noise, "
                  "-noise_clip, noise_clip): one draw over the population in ascending row order, the draw "
                  "algorithms.rebrac_bca.refresh makes for a held-out bank of these rows at key PRNGKey(score_seed)"),
            refresh_key=np.asarray(refresh_key, np.uint32), noise_fold=KEY_REFRESH_NOISE,
            noise_key=np.asarray(jax.random.fold_in(refresh_key, KEY_REFRESH_NOISE), np.uint32),
            refresh_wbcp_fold=KEY_REFRESH_WBCP, policy_noise=args.policy_noise, noise_clip=args.noise_clip,
            gamma=args.gamma, critic_bc_coef=args.critic_bc_coef, max_action=prepared.max_action,
            score_batch=score_batch),
        score_definition=dict(
            score="|residual| / sigma in float64 from float32 y - q and sigma, as calibration.reference.freeze_reference",
            residual="y - q", sigma="max(eta, 1e-6) * u (calibration.reference.positive_scale)",
            y=("reward + (1 - done) * gamma * (min_k Q_target_k(s', a') - critic_bc_coef * ||a' - a_next||^2), "
               "a' = clip(pi_target(s') + noise, -1, 1), a_next the row's recorded next action "
               "(algorithms.rebrac_bca.native_target)"),
            q="min over the two critic heads at (obs, action)", eta="live scale network models[2] at (obs, action)",
            u=float(state.residual_scale),
            policy_action="deterministic actor output at obs (tanh-bounded; ReBRAC's evaluator applies no clip)"),
        residual_unit=float(state.residual_scale),
        obs_normalization=dict(
            rule=F.OBS_RULE, fit="training_complement (the training pool)", obs_mean=obs_mean, obs_std=obs_std,
            field=("the artifact's obs only (freeze_rebrac.benchmark_obs): TD3+BC's rule on the same training rows, "
                   "so the benchmark tilts every host by the same function of s"),
            host_input=(F.OBS_RULE if args.normalize_states else
                        "raw float32 observations: normalize_states=False (configs/rebrac.yaml)"),
            normalize_states=args.normalize_states, scale_network=m["cal_normalization"]),
        reward=dict(preprocessing=m["reward_preprocessing"], reward_transform=args.reward_transform,
                    reward_scale=args.reward_scale, reward_bias=args.reward_bias,
                    note="runtime.rewards.apply_reward_transform changes antmaze rewards only"),
        population=dict(rule="short-train population: every row of the reserved part, none used in training",
                        episode_rule=EPISODE_RULE,
                        components_match_raw_terminal_timeout_episodes=bool(
                            np.array_equal(np.diff(component) != 0, np.diff(episodes) != 0))),
        rows=dict(raw=m["raw_rows"], converted=m["converted_rows"], training=len(m["training_converted_ids"]),
                  withheld=len(m["withheld_converted_ids"]), heldout=len(m["heldout_converted_ids"]),
                  population=len(arrays["row"]), population_in_training=int(arrays["in_training"].sum()),
                  population_episodes=int(np.unique(arrays["episode"]).size),
                  converted_components=int(component[-1] + 1)),
        split=split, checkpoint=checkpoint, training=training,
        preparation=dict(schema=m["schema"], metadata_sha256=prepared.metadata_sha256,
                         settings_sha256=m["settings_sha256"], run_input_hashes=m["run_input_hashes"],
                         dependency_contract=m["dependency_contract"], reservation=row["protocol"]["reservation"],
                         reservation_type="runtime.rebrac.PopulationSplit", reservation_record=m["reservation"],
                         training_ids_sha256=R._array_hash(np.asarray(prepared.training_ids, np.int64)),
                         source_files_sha256=F._source_digest(m["source_files"])),
        configuration=dict(native_args=row["native_args"], bca=row["specification"]["posterior"]),
        arrays={name: dict(dtype=str(value.dtype), shape=list(value.shape)) for name, value in arrays.items()},
        diagnostics=F._diagnostics(arrays, scored), git=F.git_identity(),
        code=dict(script_sha256=sha(Path(__file__).resolve()), source_files_sha256=F._source_digest(source_files())),
        timing=timing, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def freeze_short_train(row, output, *, updates=100_000, train_fraction=0.5, split_seed=None, score_seed=None,
                       block=1000, score_batch=65536, data_dir=DEFAULT_DATA):
    """Split, train the BCA arm, freeze its live scale and score the population part."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed = row["protocol"]["seed"]
    split_seed = seed if split_seed is None else split_seed
    score_seed = seed if score_seed is None else score_seed
    started = time.perf_counter()
    converted = F.converted_rows(row, data_dir)
    row = split_row(row, converted, train_fraction, split_seed)
    (_, args, spec, protocol), prepared = prepare(row, data_dir)
    m = prepared.metadata
    if m["converted_rows"] != converted:
        raise RuntimeError("converted row count differs from the timeout rule")
    if m["withheld_converted_ids"] != m["heldout_converted_ids"]:
        raise RuntimeError("the population must be every withheld row")
    config = R.validate_prepared(args, spec, protocol, prepared)  # what run_prepared checks before training
    prepared_at = time.perf_counter()
    load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    carry, models, log = train_arm(args, config, prepared, updates, block)
    trained_at = time.perf_counter()
    state = carry[1]
    if bool(state.posterior.ready):
        raise RuntimeError("the frozen reference became ready without a refresh")
    checkpoint = write_checkpoint(output, carry, args.policy_freq)
    ids = np.asarray(prepared.heldout_ids, np.int64)
    scored = score_rows(args, models, state, prepared.heldout, jax.random.PRNGKey(score_seed),
                        prepared.max_action, score_batch)
    obs, obs_mean, obs_std = benchmark_obs(prepared, ids, args.normalize_states)
    arrays = F.population_arrays(prepared, ids, prepared.heldout._replace(obs=obs), np.zeros(len(ids), bool), scored)
    scored_at = time.perf_counter()
    # Throughput after the first block, which carries the compilation.
    steady = sum(e["seconds"] for e in log[1:])
    rate = (log[-1]["step"] - log[0]["step"]) / steady if len(log) > 1 else None
    timing = dict(prepare_seconds=prepared_at - started, training_seconds=trained_at - prepared_at,
                  first_block_seconds_with_compile=log[0]["seconds"], steady_updates_per_second=rate,
                  seconds_per_1000_updates=None if rate is None else 1e3 / rate,
                  projected_seconds_per_100k_updates=None if rate is None else 1e5 / rate,
                  scoring_seconds=scored_at - trained_at, load_average_at_training_start=load,
                  cpu_count=os.cpu_count())
    write(output / "resolved.json", row)
    write(output / "training_log.json", R._json_value(dict(
        schema="wbcp-frozen-training-log-v1", host="rebrac", updates=updates, block=block,
        policy_freq=args.policy_freq, batch_size=args.batch_size, refreshes=0,
        bc_multiplier="1 throughout: no refresh, so the frozen reference never becomes ready",
        blocks=log, residual_unit_final=float(state.residual_scale), **timing)))
    reservation = row["protocol"]["reservation"]
    split = dict(train_fraction=train_fraction, split_seed=split_seed, target_size=reservation["target_size"],
                 max_fraction=reservation["max_fraction"],
                 rule=("runtime reservation (calibration.reference.reserve_calibration via algorithms.rebrac_bca.reserve_pool "
                       "and runtime.rebrac._selection, reservation runtime.rebrac.PopulationSplit) with "
                       "rows_per_episode=None: whole effective components in np.random.default_rng(split_seed).permutation "
                       "order until (1 - train_fraction) of the converted rows are covered; every row of them forms the "
                       "population, the complement the training pool. Target (freeze_scores.split_row) and rule are "
                       "freeze_scores.py's, and the converted rows and components are TD3+BC's"))
    metadata = build_metadata(
        row=row, args=args, prepared=prepared, arrays=arrays, scored=scored, state=state, score_seed=score_seed,
        score_batch=score_batch, updates=updates, checkpoint=checkpoint, split=split, obs_mean=obs_mean, obs_std=obs_std,
        timing=dict(timing, total_seconds=time.perf_counter() - started),
        training=dict(block=block, refreshes=0, scale_network="fit at every update; frozen live at the end",
                      bc_multiplier="1 (native ReBRAC) throughout", log="training_log.json"))
    metadata = F.write_artifact(output, arrays, metadata)
    return dict(arrays=arrays, metadata=metadata, state=state, models=models, population=prepared.heldout,
                prepared=prepared, args=args, spec=spec, protocol=protocol, config=config, row=row,
                training_rng=carry[0])


def summary(metadata):
    keys = ("source", "dataset", "updates", "seed", "score_seed", "rows", "residual_unit", "diagnostics", "timing")
    return dict({key: metadata[key] for key in keys}, checkpoint_sha256=metadata["checkpoint"]["sha256"],
                npz_sha256=metadata["npz_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="hopper", help="rebrac dataset key in configs/rebrac.yaml (default hopper)")
    parser.add_argument("--seed", type=int, help="declared seed from configs/experiment.yaml (default the first)")
    parser.add_argument("--updates", type=int, default=100_000, help="host updates (default 100000)")
    parser.add_argument("--train-fraction", type=float, default=0.5, help="training share of the rows (default 0.5)")
    parser.add_argument("--split-seed", type=int, help="component permutation seed (default --seed)")
    parser.add_argument("--block", type=int, default=1000, help="host updates per jax.lax.scan block (default 1000)")
    parser.add_argument("--score-seed", type=int, help="refresh key seed for target noise (default --seed)")
    parser.add_argument("--score-batch", type=int, default=65536, help="rows per forward-pass chunk")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="cached D4RL HDF5 directory")
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory; must not exist")
    opt = parser.parse_args(argv)
    if opt.output.exists():
        parser.error("--output already exists")
    datasets = list(read_config(CONFIG)["datasets"])
    if opt.dataset not in datasets:
        parser.error("--dataset must be one of " + str(datasets))
    opt.seed = F.declared_seeds()[0] if opt.seed is None else opt.seed
    if opt.seed not in F.declared_seeds():
        parser.error("--seed must be one of the declared seeds " + str(F.declared_seeds()))
    if opt.updates < 1 or opt.block < 1 or not 0 < opt.train_fraction < 1:
        parser.error("--updates and --block must be positive and --train-fraction inside (0, 1)")
    for name in ("split_seed", "score_seed"):
        value = getattr(opt, name)
        if value is not None and not 0 <= value < 2**32:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 2**32)")
    if opt.score_batch < 1:
        parser.error("--score-batch must be positive")
    try:
        result = freeze_short_train(
            resolve_row(opt.dataset, opt.seed, opt.output), opt.output, updates=opt.updates,
            train_fraction=opt.train_fraction, split_seed=opt.split_seed, score_seed=opt.score_seed,
            block=opt.block, score_batch=opt.score_batch, data_dir=opt.data_dir)
    except BaseException as error:
        if opt.output.is_dir() and not (opt.output / "failure.json").exists():
            write(opt.output / "failure.json", dict(type=type(error).__name__, message=str(error),
                                                    traceback=traceback.format_exc()))
        raise
    print(json.dumps(R._json_value(summary(result["metadata"])), indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
