"""Freeze BCA's IQL nonconformity scores on a D4RL population for the WBCP benchmark.

The IQL counterpart of freeze_scores.py's short-train mode. It writes the same artifact
(frozen.npz + frozen.json, schema wbcp-frozen-scores-v1), so d4rl_benchmark.load_pool,
d4rl_benchmark.py and dependence.py read it unchanged. All of it is the repo's own host
code: runtime.config.typed and the IQL pair driver (runtime/iql_pair.py) for the resolved
arguments, train.prepare's IQL branch for the data (the D4RL environment's get_dataset,
d4rl.qlearning_dataset and calibration.iql_reference.prepare_dataset: training-only return
range and observation mean/std + 1e-3), and algorithms/iql_bca.py for the shared Q/V system,
its two actors and its Bayesian bootstrap scale fitter.

Split. The declared iql bca row's reservation is replaced by the population split that
freeze_scores.py uses for TD3+BC: the same target, (1 - train_fraction) of the converted
rows, the same seed and the same rule, whole episodes in
np.random.default_rng(split_seed).permutation order until the target is covered
(calibration.reference.reserve_calibration, rows_per_episode=None, through
prepare_dataset(..., population_split=True); the cap is freeze_scores.MAX_FRACTION). IQL
reserves by its raw terminal/timeout episode IDs; the script checks that they break at
exactly the rows where TD3+BC's effective dependency components break, and the converted
rows are the same D4RL conversion, so for a given split seed the population rows are the
same for both hosts. Every withheld row is the population and none of it trains.

Training. The shared system (Q/V, the host and BCA actors, one scale fitter) runs --updates
updates on the training part with algorithms.iql_bca.make_shared_train_step inside
jax.lax.scan blocks, jitted as runtime.iql_pair.PairRuntime does. No refresh runs, so the
frozen reference never becomes ready, the BCA actor uses the native AWR weights (it stays
identical to the host actor; recorded) and the scale network sigma(s, a) is fit at every
update. Every block passes the pair driver's own evidence checks. The LIVE scale network and
residual unit are frozen at the end and the population is scored as
calibration.iql_reference.refresh scores a held-out bank (score_rows). IQL's refresh target
r + (1 - d) gamma V(s') draws no noise, so no key enters a score; --score-seed is accepted
for the common CLI and only recorded.

The actor learning-rate schedule decays over the declared num_updates, as in a train.py run
of this row; --updates trains the first of them.

Outputs in --output (a new directory): frozen.npz, frozen.json, resolved.json (the row with
the split), training_log.json and checkpoint_<updates>.msgpack (flax bytes of the shared
carry, as runtime.iql_pair.PairRuntime.checkpoint). The script honors JAX_PLATFORMS.

JAX_PLATFORMS=cpu python experiments/wbcp/freeze_iql.py --dataset hopper --output runs/wbcp_frozen/iql-hopper-s202609171
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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

from calibration.reference import positive_scale  # noqa: E402
from experiments.wbcp import freeze_scores as F  # noqa: E402
from runtime.config import read_config, resolve, typed  # noqa: E402
from runtime.iql_checkpoint import checkpoint_counts  # noqa: E402
from runtime.provenance import sha, source_files, write  # noqa: E402

ALGORITHM = "iql"
CONFIG = ROOT / "configs" / "iql.yaml"
DEFAULT_DATA = F.DEFAULT_DATA
POSTERIOR_FOLD = 1347375956  # calibration.iql_reference.refresh folds this into the training key for its WBCP draws
OBS_RULE = "training-observation mean/std + 1e-3 (calibration.iql_reference.prepare_dataset)"
EPISODE_RULE = ("effective dependency component (runtime.td3_bc.dependency_maps of the converted rows): contiguous "
                "raw rows, broken after terminals and at dropped timeout rows; timestep counts rows from its start. "
                "It breaks where IQL's raw terminal/timeout episode IDs do (checked), so the labels are TD3+BC's")


def resolve_row(dataset, seed, output):
    """The declared BCA row for an iql dataset key (e.g. hopper) and a declared seed."""
    return resolve(CONFIG, "bca", seed, str(Path(output).resolve()), dataset)


def split_row(row, converted, train_fraction, split_seed):
    """IQL's reservation fields set to freeze_scores.split_row's population target, seed and cap.

    F.split_row computes them for TD3+BC (target = converted - round(train_fraction * converted)).
    IQL keeps them in the driver options (reserve_size) and the posterior parameters
    (reserve_seed, reserve_max_fraction). The declared K stays, since PosteriorArgs only accepts a
    positive integer, and is unused: prepare_dataset(population_split=True) reserves with None.
    """
    reservation = F.split_row({"protocol": {}}, converted, train_fraction, split_seed)["protocol"]["reservation"]
    row = copy.deepcopy(row)
    row["options"]["reserve_size"] = reservation["target_size"]
    row["options"]["posterior_parameters"].update(reserve_seed=reservation["seed"],
                                                  reserve_max_fraction=reservation["max_fraction"])
    return row


def prepare(row, data_dir):
    """train.prepare's IQL branch, with the population split.

    train.prepare checks the cached file's SHA256 and calls m.H.load_data(args)
    (calibration/iql_state.py), which makes the D4RL environment, reads env.get_dataset(),
    converts it with d4rl.qlearning_dataset(env, dataset=raw) and calls
    calibration.iql_reference.prepare_dataset(args, converted, raw). The same calls run here,
    with two differences: get_dataset reads the checked file by path (h5path, its own
    argument), and prepare_dataset gets population_split=True.
    """
    path = F._data_dir(data_dir) / row["cache"]["filename"]
    if not path.is_file():
        raise FileNotFoundError("Required D4RL dataset: " + str(path))
    if sha(path) != row["cache"]["sha256"]:
        raise ValueError("Dataset SHA256 mismatch: " + str(path))
    os.environ["D4RL_DATASET_DIR"] = str(F._data_dir(data_dir))
    from runtime.environment import setup

    setup()  # as train.main: reuse the installed MuJoCo extension before any environment is made
    driver, m, options = typed(row)
    args, variants, arms, _ = driver.resolved_arguments(options, m)
    env = m.H.C.gym.make(args.dataset)
    try:
        raw = env.get_dataset(h5path=str(path))
        converted = m.H.C.d4rl.qlearning_dataset(env, dataset=raw)
        spaces = dict(obs_dim=int(env.observation_space.shape[0]), action_dim=int(env.action_space.shape[0]),
                      max_action=float(env.action_space.high[0]))
    finally:
        env.close()
    data = m.P.prepare_dataset(args, converted, raw, population_split=True)
    return SimpleNamespace(driver=driver, m=m, options=options, args=args, variants=variants, arms=arms, raw=raw,
                           converted=converted, data=data, spaces=spaces, path=path)


def dependency_maps(raw, converted, episode_ids):
    """runtime.td3_bc.dependency_maps of IQL's converted rows, checked against IQL's episode IDs.

    With raw timeouts, d4rl.qlearning_dataset keeps raw row i < N - 1 unless it is a timeout row,
    with next observation row i + 1 (checked). The effective components define TD3+BC's population
    split and the artifact's episode and timestep. IQL reserved by its raw terminal/timeout episode
    IDs; the two must break at the same rows, or the population would differ between the hosts.
    """
    if "timeouts" not in raw:
        raise ValueError("the population labels need raw timeouts (every declared dataset has them)")
    rows = np.flatnonzero(~np.asarray(raw["timeouts"][:-1], bool))
    obs = np.asarray(raw["observations"])
    if (len(rows) != len(converted["rewards"])
            or not np.array_equal(converted["observations"], obs[rows].astype(np.float32))
            or not np.array_equal(converted["next_observations"], obs[rows + 1].astype(np.float32))
            or not np.array_equal(np.asarray(converted["terminals"], bool), np.asarray(raw["terminals"], bool)[rows])):
        raise ValueError("the D4RL conversion differs from the timeout row map")
    maps = F.R.dependency_maps(rows, converted["terminals"])
    component = np.asarray(maps["effective_components"])
    if not np.array_equal(np.diff(component) != 0, np.diff(np.asarray(episode_ids)) != 0):
        raise ValueError("IQL's episode IDs and TD3+BC's effective components break at different rows")
    return maps


def train_shared(prepared, updates, block):
    """Run `updates` shared IQL updates on the training part in jitted jax.lax.scan blocks, no refresh.

    The carry, fitters and step are runtime.iql_pair.PairRuntime's (algorithms.iql_bca), and every
    block passes the driver's checks: no invalid flag (first_failure), the block evidence of the
    pair (verify_block_evidence, verify_variant_evidence: every scale fit accepted), and the
    reference never ready.
    """
    p, m, np_ = prepared, prepared.m, prepared.m.np
    D, args, variants, arms = p.driver.D, p.args, p.variants, p.arms
    od, ad, max_action = p.spaces["obs_dim"], p.spaces["action_dim"], p.spaces["max_action"]
    state, key, actor = m.H.initialize_agent(args, od, ad, max_action)
    fitters = m.X.make_fitters(args, state, od, ad, variants)
    carry = m.X.initialize_shared(args, state, key, fitters, arms)
    step = m.X.make_shared_train_step(args, p.data.train, fitters, variants, arms)
    scan = jax.jit(lambda c, n: jax.lax.scan(step, c, None, n), static_argnums=1)
    arm_dicts, variant_dicts = [asdict(a) for a in arms], [asdict(v) for v in variants]
    actor_applied, cal_applied = np_.zeros(len(arms), np_.int64), np_.zeros(len(variants), np_.int64)
    done, log = 0, []
    while done < updates:
        count = min(block, updates - done)
        started = time.perf_counter()
        carry, metrics = scan(carry, count)
        metrics = jax.device_get(metrics)
        seconds = time.perf_counter() - started
        failed = D.first_failure(metrics, done + 1, np_)
        if failed is not None:
            raise FloatingPointError("invalid update in the shared IQL system: " + json.dumps(D.plain(failed)))
        D.verify_block_evidence(metrics, count, arm_dicts, "shared", np_)
        cal_applied += p.driver.verify_variant_evidence(metrics, count, variant_dicts, args.batch_size, np_)
        applied, _, post = D.block_counts(metrics, count, len(arms), np_)
        actor_applied += applied
        if np_.any(metrics["posterior_ready"]) or post.any() or int(carry.step) != done + count:
            raise RuntimeError("the reference became ready or the step counter fell behind")
        done += count
        log.append(_block_summary(metrics, done, carry, seconds))
    return SimpleNamespace(carry=carry, fitters=fitters, actor=actor, log=log, actor_applied=actor_applied,
                           cal_applied=cal_applied)


def _block_summary(metrics, last, carry, seconds):
    m = {name: np.asarray(value).astype(np.float64) for name, value in metrics.items()}
    return dict(step=last, seconds=seconds, q_loss=float(m["q_loss"].mean()), value_loss=float(m["value_loss"].mean()),
                actor_loss=m["actor_loss"].mean(axis=0).tolist(), weight_mean=m["weight_mean"].mean(axis=0).tolist(),
                scale_loss=float(m["scale_loss"].mean()), scale_coverage=float(m["scale_coverage"].mean()),
                scale_fit_accepted=float(m["cal_variant_accepted"].mean()),
                posterior_ready=float(m["posterior_ready"].mean()),
                residual_unit=float(carry.extras[0].calibration.resid_scale))


def write_checkpoint(output, trained):
    """The shared carry as flax bytes (PairRuntime.checkpoint) and its decoded counters (train.py's check)."""
    payload = serialization.to_bytes(trained.carry)
    step = int(trained.carry.step)
    path = Path(output) / f"checkpoint_{step}.msgpack"
    with path.open("xb") as f:
        f.write(payload)
    counters = checkpoint_counts(serialization.msgpack_restore(payload), "shared", step,
                                 [int(x) for x in trained.actor_applied], [int(x) for x in trained.cal_applied])
    return dict(step=step, path=path.name, sha256=sha(path), size_bytes=len(payload), counters=counters,
                format="flax serialization.to_bytes(SharedPairCarry), as runtime.iql_pair.PairRuntime.checkpoint")


def score_rows(fitter, cal_state, agent_state, actor, actor_params, data, discount, batch_size):
    """Score `data` as calibration.iql_reference.refresh scores a held-out bank `cal`.

    refresh (calibration/iql_reference.py, run eagerly by runtime.iql_pair.PairRuntime.refresh with
    cal_state = the variant's live BCAState and agent_state = carry.nuisance) computes
        pred_cal = stop_gradient(fitter.predictions(cal_state, cal_state.calibrator.params, cal))
        target = cal.reward + (1.0 - cal.done) * discount * agent_state.vf.apply_fn(vf.params, cal.next_obs)
        q = jnp.min(agent_state.qf.apply_fn(qf.params, cal.obs, cal.action), axis=-1)
        residual = stop_gradient(target - q)
    and calibration.reference.freeze_reference scores |residual| / positive_scale(pred_cal,
    cal_state.resid_scale). The target is deterministic: no noise and no key. Each chunk of
    batch_size rows repeats those lines eagerly; with one chunk this is refresh's computation op
    for op. policy_action is the host actor's evaluation action at the (normalized) obs,
    actor.act(params, obs) = clip(max_action * mean, -max_action, max_action): the GaussianPolicy
    mean, or the DeterministicPolicy output (algorithms/iql.py), as runtime/iql.py's evaluator
    applies it. Returns float32 NumPy arrays.
    """
    n = len(data.reward)
    parts = {name: [] for name in ("target", "q", "residual", "eta", "sigma", "policy_action")}
    for start in range(0, n, batch_size):
        cal = jax.tree_util.tree_map(lambda x: x[start:start + batch_size], data)
        pred_cal = jax.lax.stop_gradient(fitter.predictions(cal_state, cal_state.calibrator.params, cal))
        target = cal.reward + (1.0 - cal.done) * discount * agent_state.vf.apply_fn(agent_state.vf.params, cal.next_obs)
        q = jnp.min(agent_state.qf.apply_fn(agent_state.qf.params, cal.obs, cal.action), axis=-1)
        residual = jax.lax.stop_gradient(target - q)
        values = dict(target=target, q=q, residual=residual, eta=pred_cal,
                      sigma=positive_scale(pred_cal, cal_state.resid_scale),
                      policy_action=actor.act(actor_params, cal.obs))
        for name, value in values.items():
            parts[name].append(np.asarray(value))
    scored = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    if not all(np.all(np.isfinite(value)) for value in scored.values()) or not np.all(scored["sigma"] > 0):
        raise FloatingPointError("nonfinite score inputs or a nonpositive frozen scale")
    return scored


def score_population(prepared, trained, batch_size):
    """score_rows on the population with the trained state; policy_action from the host actor."""
    carry, arms = trained.carry, prepared.arms
    host = next(i for i, a in enumerate(arms) if a.mode == "off")
    extra = carry.extras[0]  # the one scale variant (algorithms.iql_bca.default_design)
    host_actor = prepared.m.S.actor_at(carry.actors, host)
    return score_rows(trained.fitters[0], extra.calibration, carry.nuisance, trained.actor, host_actor.params,
                      prepared.data.calibration, prepared.args.discount, batch_size)


def actors_identical(carry):
    """Whether the BCA actor still equals the host actor (no refresh: identical weights and batches)."""
    leaves = jax.tree_util.tree_leaves(carry.actors.params)
    return all(bool(np.array_equal(np.asarray(x)[0], np.asarray(x)[1])) for x in leaves)


def build_metadata(*, row, prepared, maps, arrays, scored, trained, score_seed, score_batch, updates, checkpoint,
                   split, timing, training):
    p, args, m = prepared, prepared.args, prepared.data.metadata
    component = np.asarray(maps["effective_components"], np.int64)
    carry = trained.carry
    unit = float(carry.extras[0].calibration.resid_scale)
    file_sha = sha(p.path)
    return dict(
        schema=F.SCHEMA, algorithm=ALGORITHM, host="iql", method="bca", run_id=row["run_id"], dataset=args.dataset,
        dataset_file=dict(filename=p.path.name, path=str(p.path), size_bytes=p.path.stat().st_size, sha256=file_sha,
                          declared_sha256=row["cache"]["sha256"]),
        dataset_sha256=file_sha, source="short-train", updates=int(updates), seed=args.seed, score_seed=score_seed,
        target_noise=dict(
            rule=("none: calibration.iql_reference.refresh's target r + (1 - d) * discount * V(s') draws no noise, "
                  "so no key enters a score; score_seed is recorded only"),
            refresh_posterior_fold=POSTERIOR_FOLD, discount=args.discount, max_action=p.spaces["max_action"],
            score_batch=score_batch),
        score_definition=dict(
            score="|residual| / sigma in float64 from float32 y - q and sigma, as calibration.reference.freeze_reference",
            residual="y - q", sigma="max(eta, 1e-6) * u (calibration.reference.positive_scale)",
            y="reward + (1 - done) * discount * V(s') with the current value network (calibration.iql_reference.refresh)",
            q="min over the twin online Q heads at (obs, action) (calibration.iql_reference.refresh)",
            eta=("live scale network at (obs, action): BootstrapScaleFitter.predictions with the live calibrator "
                 "parameters, as refresh freezes them"),
            u=unit,
            policy_action=("host actor's evaluation action actor.act(params, obs) = clip(max_action * mean, "
                           "-max_action, max_action) at the normalized obs: the "
                           + ("DeterministicPolicy output" if args.iql_deterministic else "GaussianPolicy mean")
                           + " (algorithms/iql.py; runtime/iql.py evaluates this action)")),
        residual_unit=unit,
        obs_normalization=dict(rule=OBS_RULE, fit=m["normalization_fit"], obs_mean=m["obs_mean"],
                               obs_std=m["obs_std_with_epsilon"]),
        reward_normalization=m["reward"],
        population=dict(rule="short-train population: every row of the reserved part, none used in training",
                        episode_rule=EPISODE_RULE, components_match_raw_terminal_timeout_episodes=True),
        rows=dict(raw=len(p.raw["rewards"]), converted=int(m["dataset_rows"]), training=int(m["training_size"]),
                  withheld=int(m["withheld_size"]), heldout=int(m["calibration_size"]), population=len(arrays["row"]),
                  population_in_training=int(arrays["in_training"].sum()),
                  population_episodes=int(np.unique(arrays["episode"]).size),
                  converted_components=int(component[-1] + 1)),
        split=split, checkpoint=checkpoint, training=training,
        preparation=dict(metadata=m, metadata_sha256=F.R._digest(m), population_split=True,
                         function="calibration.iql_reference.prepare_dataset(args, converted, raw, population_split=True)",
                         reservation=dict(target_size=row["options"]["reserve_size"],
                                          seed=row["options"]["posterior_parameters"]["reserve_seed"],
                                          max_fraction=row["options"]["posterior_parameters"]["reserve_max_fraction"],
                                          rows_per_episode=m["rows_per_episode"]),
                         training_ids_sha256=F.R._array_hash(np.asarray(p.data.train_indices, np.int64)),
                         population_ids_sha256=F.R._array_hash(np.asarray(arrays["row"], np.int64))),
        configuration=dict(args=asdict(args), variants=[asdict(v) for v in p.variants],
                           arms=[asdict(a) for a in p.arms], options=row["options"]),
        actors=dict(applied_updates=[int(x) for x in trained.actor_applied],
                    scale_fits=[int(x) for x in trained.cal_applied],
                    bca_actor_identical_to_host=actors_identical(carry)),
        arrays={name: dict(dtype=str(value.dtype), shape=list(value.shape)) for name, value in arrays.items()},
        diagnostics=F._diagnostics(arrays, scored), git=F.git_identity(),
        code=dict(script_sha256=sha(Path(__file__).resolve()), source_files_sha256=F._source_digest(source_files())),
        timing=timing, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def freeze_short_train(row, output, *, updates=100_000, train_fraction=0.5, split_seed=None, score_seed=None,
                       block=1000, score_batch=65536, data_dir=DEFAULT_DATA):
    """Split, train the shared IQL system, freeze its live scale and score the population part."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed = row["options"]["seed"]
    split_seed = seed if split_seed is None else split_seed
    score_seed = seed if score_seed is None else score_seed
    started = time.perf_counter()
    converted = F.converted_rows(row, data_dir)
    row = split_row(row, converted, train_fraction, split_seed)
    prepared = prepare(row, data_dir)
    data, m = prepared.data, prepared.data.metadata
    if m["dataset_rows"] != converted:
        raise RuntimeError("converted row count differs from the timeout rule")
    if not np.array_equal(data.withheld_indices, data.calibration_indices):
        raise RuntimeError("the population must be every withheld row")
    maps = dependency_maps(prepared.raw, prepared.converted, data.episode_ids)
    prepared_at = time.perf_counter()
    trained = train_shared(prepared, updates, block)
    trained_at = time.perf_counter()
    checkpoint = write_checkpoint(output, trained)
    ids = np.asarray(data.calibration_indices, np.int64)
    scored = score_population(prepared, trained, score_batch)
    arrays = F.population_arrays(SimpleNamespace(metadata={"dependency_maps": maps}), ids, data.calibration,
                                 np.zeros(len(ids), bool), scored)
    scored_at = time.perf_counter()
    log = trained.log
    # Throughput after the first block, which carries the compilation.
    steady = sum(e["seconds"] for e in log[1:])
    rate = (log[-1]["step"] - log[0]["step"]) / steady if len(log) > 1 else None
    timing = dict(prepare_seconds=prepared_at - started, training_seconds=trained_at - prepared_at,
                  first_block_seconds_with_compile=log[0]["seconds"], steady_updates_per_second=rate,
                  seconds_per_1000_updates=None if rate is None else 1e3 / rate,
                  projected_seconds_per_100k_updates=None if rate is None else 1e5 / rate,
                  scoring_seconds=scored_at - trained_at,
                  load_average=list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                  cpu_count=os.cpu_count())
    write(output / "resolved.json", row)
    write(output / "training_log.json", F.R._json_value(dict(
        schema="wbcp-frozen-training-log-v1", host="iql", updates=updates, block=block, refreshes=0,
        bca_actor_weights="native AWR weights throughout: no refresh, so the frozen reference never becomes ready",
        blocks=log, residual_unit_final=float(trained.carry.extras[0].calibration.resid_scale), **timing)))
    split = dict(train_fraction=train_fraction, split_seed=split_seed, target_size=row["options"]["reserve_size"],
                 rule=("calibration.reference.reserve_calibration via calibration.iql_reference.prepare_dataset "
                       "(population_split=True) with rows_per_episode=None: whole episodes in "
                       "np.random.default_rng(split_seed).permutation order until (1 - train_fraction) of the converted "
                       "rows are covered; every row of them forms the population, the complement the training pool. "
                       "Target and rule are freeze_scores.py's, and IQL's episodes are TD3+BC's effective components"))
    metadata = build_metadata(
        row=row, prepared=prepared, maps=maps, arrays=arrays, scored=scored, trained=trained, score_seed=score_seed,
        score_batch=score_batch, updates=updates, checkpoint=checkpoint, split=split,
        timing=dict(timing, total_seconds=time.perf_counter() - started),
        training=dict(block=block, refreshes=0, scale_network="fit at every update; frozen live at the end",
                      bca_actor="native AWR weights (reference never ready); shares Q/V with the host actor",
                      log="training_log.json"))
    metadata = F.write_artifact(output, arrays, metadata)
    return dict(arrays=arrays, metadata=metadata, prepared=prepared, trained=trained, scored=scored, maps=maps,
                row=row)


def summary(metadata):
    keys = ("source", "dataset", "updates", "seed", "score_seed", "rows", "residual_unit", "diagnostics", "timing")
    return dict({key: metadata[key] for key in keys}, checkpoint_sha256=metadata["checkpoint"]["sha256"],
                npz_sha256=metadata["npz_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="hopper", help="iql dataset key in configs/iql.yaml (default hopper)")
    parser.add_argument("--seed", type=int, help="declared seed from configs/experiment.yaml (default the first)")
    parser.add_argument("--updates", type=int, default=100_000, help="shared IQL updates (default 100000)")
    parser.add_argument("--train-fraction", type=float, default=0.5, help="training share of the rows (default 0.5)")
    parser.add_argument("--split-seed", type=int, help="episode permutation seed (default --seed)")
    parser.add_argument("--block", type=int, default=1000, help="updates per jax.lax.scan block (default 1000)")
    parser.add_argument("--score-seed", type=int,
                        help="recorded only: IQL's refresh target draws no noise (default --seed)")
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
    print(json.dumps(F.R._json_value(summary(result["metadata"])), indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
