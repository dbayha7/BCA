"""Freeze BCA's CQL nonconformity scores on a D4RL population for the WBCP benchmark.

The CQL counterpart of freeze_scores.py's short-train mode. It writes the same artifact
(frozen.npz + frozen.json, schema wbcp-frozen-scores-v1), so d4rl_benchmark.load_pool,
d4rl_benchmark.py and dependence.py read it unchanged. All of it is the repo's own host
code: train.prepare and runtime/cql.py for the data (exact D4RL conversion, CQL's reward
handling, observation mean/std + 1e-3 fit on the training pool), and algorithms/cql_bca.py
for the BCA arm, its updates, its targets, its critics and its live scale network.

Split. The declared cql bca row's reservation is replaced by the population split that
freeze_scores.py uses for TD3+BC: the same target, (1 - train_fraction) of the converted
rows, the same seed and the same rule, whole episodes in
np.random.default_rng(split_seed).permutation order until the target is covered
(calibration.reference.reserve_calibration, rows_per_episode=None). runtime.cql reserves by
its raw terminal/timeout episode IDs; the script checks that they break at exactly the rows
where TD3+BC's effective dependency components break, so for a given split seed the
population rows are the same for both hosts. The protocol is a
runtime.cql.PopulationSplitProtocol, the only type through which runtime.cql accepts
rows_per_episode=None, and every withheld row is the population.

Training. The BCA arm runs --updates host updates on the training part with
P.make_train_step inside jax.lax.scan blocks, as runtime.cql.run_prepared does. No refresh
runs, so the frozen reference never becomes ready and every conservative-gap multiplier
stays 1 (native CQL) while the scale network sigma(s, a) is fit at every update. The LIVE
network and residual unit are frozen at the end and the population is scored as
algorithms.cql_bca.refresh scores a held-out bank (score_rows).

Target noise: one draw over the population in ascending row order, exactly the draw refresh
makes for a held-out bank of these rows at key PRNGKey(score_seed): the tanh-Gaussian
next-action noise normal(fold_in(PRNGKey(score_seed), 1213156420), (N, action_dim)).

Outputs in --output (a new directory): frozen.npz, frozen.json, resolved.json (the row with
the split), training_log.json and checkpoint_<updates>.msgpack (flax bytes, as train.py).
The script honors JAX_PLATFORMS.

JAX_PLATFORMS=cpu python experiments/wbcp/freeze_cql.py --dataset hopper --output runs/wbcp_frozen/cql-hopper-s202609171
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
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

import algorithms.cql_bca as P  # noqa: E402
import runtime.cql as R  # noqa: E402
import train as T  # noqa: E402
from calibration.reference import positive_scale  # noqa: E402
from experiments.wbcp import freeze_scores as F  # noqa: E402
from runtime.config import read_config, resolve, typed  # noqa: E402
from runtime.provenance import sha, source_files, write  # noqa: E402
from runtime.validation import checkpoint_counts  # noqa: E402

ALGORITHM = "cql"
CONFIG = ROOT / "configs" / "cql.yaml"
DEFAULT_DATA = F.DEFAULT_DATA
EPISODE_RULE = ("effective dependency component (runtime.td3_bc.dependency_maps of the converted rows): contiguous "
                "raw rows, broken after terminals and at dropped timeout rows; timestep counts rows from its start. "
                "It breaks where runtime.cql's raw terminal/timeout episode IDs do (checked), so the labels are "
                "TD3+BC's")


def resolve_row(dataset, seed, output):
    """The declared BCA row for a cql dataset key (e.g. hopper) and a declared seed."""
    return resolve(CONFIG, "bca", seed, str(Path(output).resolve()), dataset)


def split_row(row, converted, train_fraction, split_seed):
    """CQL's protocol fields set to freeze_scores.split_row's population target, seed and cap.

    F.split_row computes them for TD3+BC (target = converted - round(train_fraction * converted));
    rows_per_episode=None selects whole episodes (calibration.reference.reserve_calibration).
    """
    reservation = F.split_row(row, converted, train_fraction, split_seed)["protocol"]["reservation"]
    row = copy.deepcopy(row)
    row["protocol"].update(calibration_target_size=reservation["target_size"], calibration_seed=reservation["seed"],
                           calibration_max_fraction=reservation["max_fraction"], calibration_rows_per_episode=None)
    return row


def typed_population(row):
    """typed(row) with the protocol as runtime.cql.PopulationSplitProtocol.

    runtime.config.typed builds RunProtocol, which refuses rows_per_episode=None, so the row is
    typed with K = 1 and the protocol is rebuilt from the same fields with K = None.
    """
    probe = copy.deepcopy(row)
    probe["protocol"]["calibration_rows_per_episode"] = 1
    runner, args, spec, protocol = typed(probe)
    fields = {name: getattr(protocol, name) for name in protocol.__dataclass_fields__}
    protocol = runner.PopulationSplitProtocol(**dict(fields, calibration_rows_per_episode=None))
    runner.validate_protocol(args, spec, protocol)
    return runner, args, spec, protocol


def prepare(row, data_dir):
    """typed_population(row) and train.prepare(row, objects): the inputs a train.py run would see."""
    os.environ["D4RL_DATASET_DIR"] = str(F._data_dir(data_dir))
    objects = typed_population(row)
    return objects, T.prepare(row, objects)


def dependency_maps(prepared, path):
    """runtime.td3_bc.dependency_maps of CQL's converted rows, checked against CQL's episode IDs.

    The effective components define TD3+BC's population split and the artifact's episode and
    timestep. runtime.cql reserved by its raw terminal/timeout episode IDs instead; the two must
    break at the same rows, or the population would differ between the hosts.
    """
    import h5py

    with h5py.File(path, "r") as f:
        terminals = np.asarray(f["terminals"][()]).astype(bool)
    m = prepared.metadata
    rows = np.asarray(m["conversion_raw_indices"], np.int64)
    maps = F.R.dependency_maps(rows, terminals[rows])
    component, ids = np.asarray(maps["effective_components"]), np.asarray(m["converted_episode_ids"])
    if not np.array_equal(np.diff(component) != 0, np.diff(ids) != 0):
        raise ValueError("CQL's episode IDs and TD3+BC's effective components break at different rows")
    return maps


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
        log.append(_block_summary(metrics, done, carry[1], time.perf_counter() - started))
    return carry, models, log


def _block_summary(metrics, last, state, seconds):
    m = {name: np.asarray(value).astype(np.float64) for name, value in metrics.items()}
    mean = lambda name: float(m[name].mean()) if name in m else None  # noqa: E731
    return dict(step=last, seconds=seconds, qf_loss=mean("qf_loss"), policy_loss=mean("policy_loss"),
                alpha=mean("alpha"), average_qf1=mean("average_qf1"), average_target_q=mean("average_target_q"),
                cql_min_qf1_loss=mean("cql_min_qf1_loss"), scale_loss=mean("scale_loss"),
                scale_fit_accepted=mean("scale_fit_accepted"), critic_dose_mean=mean("critic_dose_mean"),
                posterior_ready=mean("posterior_ready"), residual_unit=float(state.residual_scale))


def write_checkpoint(output, carry):
    payload = serialization.to_bytes({"state": carry[1], "training_rng": carry[0], "step": carry[2]})
    step = int(carry[2])
    path = Path(output) / f"checkpoint_{step}.msgpack"
    with path.open("xb") as f:
        f.write(payload)
    counters = checkpoint_counts(serialization.msgpack_restore(payload), "cql", step)
    return dict(step=step, path=path.name, sha256=sha(path), size_bytes=len(payload), counters=counters,
                format="flax serialization.to_bytes({'state', 'training_rng', 'step'}), as train.py")


def score_rows(args, models, state, data, key, max_action, batch_size):
    """Score `data` as algorithms.cql_bca.refresh scores a held-out bank at refresh key `key`.

    refresh (algorithms/cql_bca.py) computes target = native_target(args, models, state.native,
    heldout, fold_in(key, 1213156420), max_action), q = min(critic1, critic2) at (obs, action)
    and eta = models[3] at (obs, action); calibration.reference.freeze_reference then scores
    |target - q| / positive_scale(eta, residual_scale). native_target draws its tanh-Gaussian
    noise once for the whole bank. Here that draw is made once for all rows and the forward
    passes run in chunks of batch_size rows, each repeating native_target line by line with
    its slice of the draw; with one chunk this is refresh's computation exactly.
    policy_action is runtime.cql.evaluate's deterministic action, max_action * tanh(mean): the
    tanh bounds it and the host applies no further clip. Returns float32 NumPy arrays.
    """
    actor, c1, c2, cal = models
    n, action_dim = data.action.shape
    shape = (n, args.cql_n_actions, action_dim) if args.cql_max_target_backup else (n, action_dim)
    noise = jax.random.normal(jax.random.fold_in(key, F.KEY_REFRESH_NOISE), shape)
    native = state.native
    parts = {name: [] for name in ("target", "q", "residual", "eta", "sigma", "policy_action")}
    for start in range(0, n, batch_size):
        batch = jax.tree_util.tree_map(lambda x: x[start:start + batch_size], data)
        # algorithms.cql_bca.native_target, line by line, with this chunk's slice of the noise
        b, a = batch.action.shape
        obs = (jnp.broadcast_to(batch.next_obs[:, None, :], (b, args.cql_n_actions, batch.obs.shape[-1]))
               if args.cql_max_target_backup else batch.next_obs)
        action, logp = P.BASE.tanh_gaussian_sample(*actor.apply(native.actor.params, obs),
                                                   noise[start:start + batch_size], max_action)
        q_next = jnp.minimum(c1.apply(native.critic1_target.params, batch.next_obs, action),
                             c2.apply(native.critic2_target.params, batch.next_obs, action))
        if args.cql_max_target_backup:
            idx = jnp.argmax(q_next, axis=-1)
            q_next = jnp.max(q_next, axis=-1)
            logp = jnp.take_along_axis(logp, idx[:, None], axis=-1).squeeze(-1)
        if args.backup_entropy:
            alpha = (jnp.exp(P.BASE.scalar_value(native.log_alpha)) * args.alpha_multiplier
                     if args.use_automatic_entropy_tuning else args.alpha_multiplier)
            q_next = q_next - alpha * logp
        target = jax.lax.stop_gradient(batch.reward + (1.0 - batch.done) * args.discount * q_next)
        # algorithms.cql_bca.refresh
        q = jnp.minimum(c1.apply(native.critic1.params, batch.obs, batch.action),
                        c2.apply(native.critic2.params, batch.obs, batch.action))
        eta = cal.apply(state.calibrator.params, batch.obs, batch.action)
        mean, _ = actor.apply(native.actor.params, batch.obs)
        values = dict(target=target, q=q, residual=target - q, eta=eta,
                      sigma=positive_scale(eta, state.residual_scale), policy_action=max_action * jnp.tanh(mean))
        for name, value in values.items():
            parts[name].append(np.asarray(value))
    scored = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    if not all(np.all(np.isfinite(value)) for value in scored.values()) or not np.all(scored["sigma"] > 0):
        raise FloatingPointError("nonfinite score inputs or a nonpositive frozen scale")
    return scored


def population_arrays(maps, ids, data, in_training, scored):
    """freeze_scores.population_arrays (the frozen.npz contract and its score computation).

    It reads the components from prepared.metadata['dependency_maps'], which runtime.cql does not
    record, so it gets a view holding the maps computed by dependency_maps().
    """
    return F.population_arrays(SimpleNamespace(metadata={"dependency_maps": maps}), ids, data, in_training, scored)


def _reward_normalization(record):
    """The scalar fields of runtime.cql's reward-normalization record (its row lists are in the preparation)."""
    summary = {k: v for k, v in record.items() if not isinstance(v, (list, dict))}
    for name in ("selected_segment_ids", "excluded_segment_ids"):
        if name in record:
            summary[name.replace("_ids", "s")] = len(record[name])
    return summary


def build_metadata(*, row, args, prepared, maps, arrays, scored, state, path, score_seed, score_batch, updates,
                   checkpoint, split, timing, training):
    m = prepared.metadata
    component = np.asarray(maps["effective_components"], np.int64)
    refresh_key = jax.random.PRNGKey(score_seed)
    file_sha = sha(path)
    return dict(
        schema=F.SCHEMA, algorithm=ALGORITHM, host="cql", method="bca", run_id=row["run_id"], dataset=args.dataset,
        dataset_file=dict(filename=path.name, path=str(path), size_bytes=path.stat().st_size, sha256=file_sha,
                          declared_sha256=row["cache"]["sha256"]),
        dataset_sha256=file_sha, source="short-train", updates=int(updates), seed=args.seed, score_seed=score_seed,
        target_noise=dict(
            rule=("noise = normal(fold_in(PRNGKey(score_seed), 1213156420), (N, action_dim)) "
                  "[(N, cql_n_actions, action_dim) with cql_max_target_backup]: one draw over the population in "
                  "ascending row order, the draw algorithms.cql_bca.refresh makes for a held-out bank of these rows "
                  "at key PRNGKey(score_seed)"),
            refresh_key=np.asarray(refresh_key, np.uint32), noise_fold=F.KEY_REFRESH_NOISE,
            noise_key=np.asarray(jax.random.fold_in(refresh_key, F.KEY_REFRESH_NOISE), np.uint32),
            refresh_wbcp_fold=F.KEY_REFRESH_WBCP, cql_max_target_backup=args.cql_max_target_backup,
            cql_n_actions=args.cql_n_actions, backup_entropy=args.backup_entropy, discount=args.discount,
            max_action=prepared.max_action, score_batch=score_batch),
        score_definition=dict(
            score="|residual| / sigma in float64 from float32 y - q and sigma, as calibration.reference.freeze_reference",
            residual="y - q", sigma="max(eta, 1e-6) * u (calibration.reference.positive_scale)",
            y=("reward + (1 - done) * discount * min(Q1_target, Q2_target)(s', a'), a' = max_action * tanh(mean(s') + "
               "noise * exp(clip(log_std(s'), -20, 2))) from the current actor"
               + (", minus alpha * log pi(a'|s')" if args.backup_entropy else "")
               + " (algorithms.cql_bca.native_target)"),
            q="min(Q1, Q2) at (obs, action)", eta="live scale network models[3] at (obs, action)",
            u=float(state.residual_scale),
            policy_action="max_action * tanh(actor mean at obs): runtime.cql.evaluate's deterministic action"),
        residual_unit=float(state.residual_scale),
        obs_normalization=dict(rule=F.OBS_RULE, fit=m["normalization_fit"], obs_mean=m["obs_mean"],
                               obs_std=m["obs_std"]),
        reward_normalization=_reward_normalization(m["reward_normalization"]),
        population=dict(rule="short-train population: every row of the reserved part, none used in training",
                        episode_rule=EPISODE_RULE, components_match_raw_terminal_timeout_episodes=True),
        rows=dict(raw=m["raw_rows"], converted=m["converted_rows"], training=len(m["training_converted_indices"]),
                  withheld=len(m["withheld_converted_indices"]), heldout=len(m["heldout_converted_indices"]),
                  population=len(arrays["row"]), population_in_training=int(arrays["in_training"].sum()),
                  population_episodes=int(np.unique(arrays["episode"]).size),
                  converted_components=int(component[-1] + 1)),
        split=split, checkpoint=checkpoint, training=training,
        preparation=dict(schema=m["schema"], metadata_sha256=prepared.metadata_sha256,
                         settings_sha256=m["settings_sha256"], raw_dataset_sha256=m["raw_dataset_sha256"],
                         run_input_hashes=m["run_input_hashes"], reservation=m["reservation"],
                         protocol_type="runtime.cql.PopulationSplitProtocol",
                         training_ids_sha256=F.R._array_hash(np.asarray(m["training_converted_indices"], np.int64)),
                         source_files_sha256=F._source_digest(m["source_files"])),
        configuration=dict(native_args=row["native_args"], bca=row["config"]),
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
    (_, args, config, _), prepared = prepare(row, data_dir)
    m = prepared.metadata
    if m["converted_rows"] != converted:
        raise RuntimeError("converted row count differs from the timeout rule")
    if m["withheld_converted_indices"] != m["heldout_converted_indices"]:
        raise RuntimeError("the population must be every withheld row")
    path = F._data_dir(data_dir) / row["cache"]["filename"]
    maps = dependency_maps(prepared, path)
    prepared_at = time.perf_counter()
    carry, models, log = train_arm(args, config, prepared, updates, block)
    trained_at = time.perf_counter()
    state = carry[1]
    if bool(state.posterior.ready):
        raise RuntimeError("the frozen reference became ready without a refresh")
    checkpoint = write_checkpoint(output, carry)
    ids = np.asarray(m["heldout_converted_indices"], np.int64)
    scored = score_rows(args, models, state, prepared.heldout, jax.random.PRNGKey(score_seed),
                        prepared.max_action, score_batch)
    arrays = population_arrays(maps, ids, prepared.heldout, np.zeros(len(ids), bool), scored)
    scored_at = time.perf_counter()
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
    write(output / "training_log.json", R._json_value(dict(
        schema="wbcp-frozen-training-log-v1", host="cql", updates=updates, block=block, refreshes=0,
        conservative_multiplier="1 throughout: no refresh, so the frozen reference never becomes ready",
        blocks=log, residual_unit_final=float(state.residual_scale), **timing)))
    split = dict(train_fraction=train_fraction, split_seed=split_seed, target_size=row["protocol"]["calibration_target_size"],
                 rule=("runtime reservation (calibration.reference.reserve_calibration via algorithms.cql_bca.reserve_pool "
                       "and runtime.cql.prepare, PopulationSplitProtocol) with rows_per_episode=None: whole episodes in "
                       "np.random.default_rng(split_seed).permutation order until (1 - train_fraction) of the converted "
                       "rows are covered; every row of them forms the population, the complement the training pool. "
                       "Target and rule are freeze_scores.py's, and CQL's episodes are TD3+BC's effective components"))
    metadata = build_metadata(
        row=row, args=args, prepared=prepared, maps=maps, arrays=arrays, scored=scored, state=state, path=path,
        score_seed=score_seed, score_batch=score_batch, updates=updates, checkpoint=checkpoint, split=split,
        timing=dict(timing, total_seconds=time.perf_counter() - started),
        training=dict(block=block, refreshes=0, scale_network="fit at every update; frozen live at the end",
                      conservative_multiplier="1 (native CQL) throughout", log="training_log.json"))
    metadata = F.write_artifact(output, arrays, metadata)
    return dict(arrays=arrays, metadata=metadata, state=state, models=models, population=prepared.heldout,
                prepared=prepared, maps=maps, args=args, config=config, row=row, training_rng=carry[0])


def summary(metadata):
    keys = ("source", "dataset", "updates", "seed", "score_seed", "rows", "residual_unit", "diagnostics", "timing")
    return dict({key: metadata[key] for key in keys}, checkpoint_sha256=metadata["checkpoint"]["sha256"],
                npz_sha256=metadata["npz_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="hopper", help="cql dataset key in configs/cql.yaml (default hopper)")
    parser.add_argument("--seed", type=int, help="declared seed from configs/experiment.yaml (default the first)")
    parser.add_argument("--updates", type=int, default=100_000, help="host updates (default 100000)")
    parser.add_argument("--train-fraction", type=float, default=0.5, help="training share of the rows (default 0.5)")
    parser.add_argument("--split-seed", type=int, help="episode permutation seed (default --seed)")
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
