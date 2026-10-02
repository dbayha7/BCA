"""Three calibration signals on frozen TD3+BC critics, each with its own BCA scale (signal study, step 2).

BCA calibrates the residual of min(Q1, Q2) (algorithms/td3_bc_bca.py), while TD3+BC's actor climbs
critic head 0, Q1, alone (algorithms/td3_bc.py). This script isolates the signal definition. It
restores a short-train TD3+BC pool (runs/wbcp_frozen/<pool>, written by freeze_scores.py) byte for
byte and freezes its actor, critic and target heads, so every signal shares one native Bellman
target t (algorithms.td3_bc_bca.native_target). Each signal is a residual t - v, v a per-row critic
value (signal_value):
  min     v = min(Q1, Q2), BCA today
  q1      v = Q1, the head the actor climbs
  maxabs  v = the head farther from t (Q1 on ties), so |t - v| = max(|t - Q1|, |t - Q2|) and the
          score is max(|t - Q1|, |t - Q2|) / sigma
maxabs's sign only enters its residual unit u = EMA of std(t - v). That keeps u the spread of a
signed residual, as for the other two. The std of the magnitude max(|r1|, |r2|) would be about 0.6
of it for a centred residual, and u reaches the dose through eta = sigma / u, so that choice alone
would raise maxabs's dose.

fit: every signal gets a FRESH scale network and residual unit, the P.initialize template (the
parent run's own initial calibrator, u = 1), fit with BCA's objective on the pool's training half
for --updates updates (default: the parent's count) at the host batch size. fit_signal_scale is
fit_scale with the residual made a parameter. With signal 'min' it reproduces fit_scale bit for bit
(test_frozen_signals.py). All signals update in one jax.lax.scan step from one batch, one target and
one bootstrap prior (fit_inputs), so they get identical inputs by construction, and every signal
gets the same updates, optimizer (adam at cal_lr), seeds and batch order. The step replays the parent
run's training stream (freeze_scores.train_arm's rng: the same batch indices and per-update keys,
hence fit_scale's per-batch target noise fold_in(key, 1179210836)), but against the final frozen
critic instead of the co-trained one. An invalid update keeps the previous scale, as fit_scale does,
and is counted. Every population row is then scored with refresh's noise convention: one draw over
the population at fold_in(PRNGKey(score_seed), 1213156420), with eager forward passes in the parent's
chunks, as freeze_scores.score_rows does. A gate first requires the restored min residual, the
parent's sigma, the observations and the policy action to reproduce frozen.npz exactly.
Outputs in DIR (new): <signal>/frozen.npz + frozen.json (schema wbcp-frozen-scores-v1; residual
t - v and sigma = max(eta, 1e-6) u of that signal, so d4rl_benchmark.py and dependence.py read it
unchanged; the metadata records the signal, the parent pool and its sha256s and the fit settings),
<signal>/scale.msgpack (the fitted scale state and u), heads.npz (t, both heads, each signal's eta
and sigma, float32) and fit.json (settings, parent hashes, the gate, per-block losses, sha256s of
the batch indices and targets, timing).

evaluate: B banks per design, with the same rows for every signal: independent rows (n = 1,024) and
the configured TD3+BC reservation bank (n and K from configs/td3_bc.yaml, BCA's sampler). Each
signal's WBCP threshold (lambda_dep = max(lambda_hat, lambda_hpd), the threshold BCA deploys) is
calibrated on that signal's scores, with the same posterior draws for every signal (common random
numbers). A row is covered for residual r under signal s's band when |r| / sigma_s <= threshold_s.
That is the score form WBCP certifies: |r| <= threshold * sigma_s can misplace the threshold's own
atom by one rounding. maxabs's residual is covered when both |t - Q1| and |t - Q2| are. The exact
population miscoverage of every residual (min, q1, q2, maxabs) under every band gives a failure
matrix with exact binomial CIs (failure: miscoverage above alpha; an abstaining bank covers
everything). The output also has mean band half-widths, thresholds relative to each signal's
lambda*, paired width ratios between signals, the head disagreement, and BCA's BC dose at every
population row (calibration.dose.frozen_level_dose, blend 0.5: 1 + 0.5 lambda eta / (1 + lambda
eta)), at lambda* and averaged over banks. The parent's critic health (host_matrix.critic_health)
flags pools whose critic did not converge (TD3+BC pen-cloned and pen-human).

Logged actions only: D4RL cannot label an action the data never took, so actor-proposed actions
and every oracle check of the true critic error live in the linear-quadratic harness (step 3).
Pre-registration: runs/wbcp_signal/expectations.md. The fit honors JAX_PLATFORMS. evaluate imports
JAX only after its NumPy workers have finished, so they can fork.

JAX_PLATFORMS=cpu python experiments/signal/frozen_signals.py fit --pool runs/wbcp_frozen/<pool> --signals min q1 maxabs --output DIR
JAX_PLATFORMS=cpu python experiments/signal/frozen_signals.py evaluate DIR --banks 1000 [--workers 8]
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from calibration.wbcp import calibrate  # noqa: E402
from experiments.wbcp import host_matrix as H  # noqa: E402
from experiments.wbcp.d4rl_benchmark import ExactRisk, draw_calibration, episode_groups, load_pool, stream  # noqa: E402

SIGNALS = ("min", "q1", "maxabs")
RESIDUALS = ("min", "q1", "q2", "maxabs")  # residual types evaluated under every band; maxabs = both heads
SCHEMA = "wbcp-frozen-scores-v1"
FIT_SCHEMA = "wbcp-signal-fit-v1"
EVALUATION_SCHEMA = "wbcp-signal-evaluation-v1"
KEY_FIT_NOISE = 1179210836  # algorithms.td3_bc_bca.fit_scale folds this into its key for target noise
KEY_FIT_PRIOR = 1128352850  # ... and this for the Bayesian bootstrap prior
IID_N = 1024
SIGNAL_TEXT = {
    "min": "t - min(Q1, Q2): BCA's residual (algorithms.td3_bc_bca)",
    "q1": "t - Q1 (critic head 0, the head TD3+BC's actor climbs)",
    "maxabs": "t - Q_k, k the head farther from t (Q1 on ties): |residual| = max(|t - Q1|, |t - Q2|)",
}
HEADS = ("target", "q", "row")  # heads.npz, plus eta_<signal>, sigma_<signal>, unit and signals


def signal_value(signal, target, q, xp=np):
    """The critic value v whose residual t - v the signal calibrates; q holds (Q1, Q2) on its last axis.
    Elementwise, so NumPy (xp=np) and JAX (xp=jax.numpy, traced or eager) give the same float32 bits."""
    if signal == "min":
        return q.min(axis=-1)
    if signal == "q1":
        return q[..., 0]
    if signal == "maxabs":
        return xp.where(xp.abs(target - q[..., 0]) >= xp.abs(target - q[..., 1]), q[..., 0], q[..., 1])
    raise ValueError(f"unknown signal {signal!r}")


# ---- fit (JAX; imported inside the functions so that evaluate can fork before JAX starts) ----

def fit_inputs(args, models, native, batch, rng, max_action=1.0):
    """What fit_scale computes before its objective: the native target with fit_scale's per-batch noise,
    both critic heads at the logged actions and the Bayesian bootstrap prior. It does not depend on the
    signal, so every signal gets the same values."""
    import jax

    import algorithms.td3_bc_bca as P
    from calibration.network import bayesian_bootstrap_weights

    target = P.native_target(args, models, native, batch, jax.random.fold_in(rng, KEY_FIT_NOISE), max_action)
    q = jax.lax.stop_gradient(models[1].apply(native.critic.params, batch.obs, batch.action))
    prior = jax.lax.stop_gradient(bayesian_bootstrap_weights(jax.random.fold_in(rng, KEY_FIT_PRIOR), len(batch.obs)))
    return target, q, prior


def scale_update(config, models, state, batch, target, q, prior):
    """fit_scale's update of state.calibrator and state.residual_scale on the residual target - q, line
    for line: soft coverage of the residual in units u against eta, the prior-weighted width penalty, an
    adam step, the EMA of std(target - q) into u, and the same validity check (an invalid update keeps
    the state)."""
    import jax
    import jax.numpy as jnp

    from calibration.dose import tree_finite
    from calibration.network import soft_coverage

    unit = state.residual_scale

    def objective(params):
        eta = models[2].apply(params, batch.obs, batch.action)
        if eta.shape != (len(batch.obs),):
            raise ValueError("scale predictor must emit one positive value per row")
        coverage = soft_coverage(target / unit, q / unit, eta, config.cal_beta)
        width = jnp.sum(prior * jnp.square(eta))
        return (jnp.square(jnp.sum(prior * coverage) - (1.0 - config.posterior.alpha))
                + config.width_penalty * width, eta)

    (loss, eta), grad = jax.value_and_grad(objective, has_aux=True)(state.calibrator.params)
    proposed = state.calibrator.apply_gradients(grads=grad)
    unit_new = config.scale_ema * unit + (1.0 - config.scale_ema) * jnp.maximum(jnp.std(target - q), 1e-06)
    valid = (tree_finite((batch, state.native, state.calibrator, target, q, prior, unit, loss, grad, proposed,
                          unit_new))
             & jnp.all(jnp.isfinite(eta) & (eta > 0)) & jnp.all(prior >= 0) & jnp.any(prior > 0)
             & jnp.isclose(prior.sum(), 1.0, rtol=1e-05, atol=1e-06) & (unit > 0) & (unit_new > 0))
    result = jax.lax.cond(valid, lambda _: state._replace(calibrator=proposed, residual_scale=unit_new),
                          lambda _: state, None)
    return result, dict(scale_inputs_valid=valid, scale_fit_accepted=valid, scale_loss=loss)


def fit_signal_scale(args, config, models, state, batch, rng, signal, max_action=1.0):
    """algorithms.td3_bc_bca.fit_scale with the residual made a parameter; signal 'min' is fit_scale."""
    import jax.numpy as jnp

    if config.arm != "bca":
        raise ValueError("only posterior arms fit a scale")
    target, q, prior = fit_inputs(args, models, state.native, batch, rng, max_action)
    return scale_update(config, models, state, batch, target, signal_value(signal, target, q, jnp), prior)


def make_fit_step(args, config, models, training, signals, max_action=1.0):
    """One scan step: make_train_step's key split and batch draw, then every signal's scale update on the
    same batch, target and prior. Carry: (rng, frozen native state, ((calibrator, u) per signal))."""
    import jax
    import jax.numpy as jnp

    import algorithms.td3_bc_bca as P

    if not len(training.obs):
        raise ValueError("training pool must be nonempty")

    def step(carry, _):
        rng, native, scales = carry
        kr, kb, kn = jax.random.split(rng, 3)
        idx = jax.random.randint(kb, (args.batch_size,), 0, len(training.obs))
        batch = jax.tree_util.tree_map(lambda x: x[idx], training)
        target, q, prior = fit_inputs(args, models, native, batch, kn, max_action)
        updated, losses, accepted = [], [], []
        for signal, (calibrator, unit) in zip(signals, scales):
            state, diag = scale_update(config, models, P.State(native, calibrator, unit, None), batch, target,
                                       signal_value(signal, target, q, jnp), prior)
            updated.append((state.calibrator, state.residual_scale))
            losses.append(diag["scale_loss"])
            accepted.append(diag["scale_fit_accepted"])
        return (kr, native, tuple(updated)), dict(idx=idx, target=target, loss=jnp.stack(losses),
                                                   accepted=jnp.stack(accepted))

    return step


def fit_scales(args, config, models, native, template, training, signals, updates, block, rng, max_action=1.0):
    """Fit a fresh scale (template.calibrator, template.residual_scale) per signal for `updates` updates in
    jax.lax.scan blocks. Returns ({signal: (calibrator, u)}, per-block log, input fingerprints)."""
    import jax

    step = make_fit_step(args, config, models, training, tuple(signals), max_action)
    run = jax.jit(lambda carry, length: jax.lax.scan(step, carry, None, length=length), static_argnums=1)
    carry = (rng, native, tuple((template.calibrator, template.residual_scale) for _ in signals))
    hashes = dict(batch_indices=hashlib.sha256(), targets=hashlib.sha256())
    accepted, done, log = np.zeros(len(signals), np.int64), 0, []
    while done < updates:
        length = min(block, updates - done)
        started = time.perf_counter()
        carry, out = run(carry, length)
        out = jax.device_get(jax.block_until_ready(out))
        seconds = time.perf_counter() - started
        hashes["batch_indices"].update(np.ascontiguousarray(out["idx"], "<i4").tobytes())
        hashes["targets"].update(np.ascontiguousarray(out["target"], "<f4").tobytes())
        accepted += out["accepted"].sum(axis=0)
        done += length
        log.append(dict(step=done, updates=length, seconds=seconds,
                        scale_loss={s: float(np.mean(out["loss"][:, i], dtype=np.float64)) for i, s in enumerate(signals)},
                        accepted={s: int(out["accepted"][:, i].sum()) for i, s in enumerate(signals)},
                        residual_unit={s: float(carry[2][i][1]) for i, s in enumerate(signals)},
                        target_mean=float(np.mean(out["target"], dtype=np.float64))))
    fingerprints = dict(batch_indices_sha256=hashes["batch_indices"].hexdigest(),
                        targets_sha256=hashes["targets"].hexdigest(),
                        rule="sha256 over the per-update batch indices (int32) and targets (float32), in update order",
                        accepted_updates={s: int(accepted[i]) for i, s in enumerate(signals)})
    return {s: carry[2][i] for i, s in enumerate(signals)}, log, fingerprints


def restore_pool(pool_dir, data_dir=None):
    """Restore a short-train TD3+BC pool exactly: prepared data, checkpoint state, models, the P.initialize
    template and rng (the parent run's initial calibrator and training stream) and the parent's records."""
    import jax.numpy as jnp
    from flax import serialization

    from experiments.wbcp import freeze_scores as FS

    pool_dir = Path(pool_dir).resolve()
    meta = json.loads((pool_dir / "frozen.json").read_text(encoding="utf8"))
    if (meta.get("schema"), meta.get("algorithm"), meta.get("source")) != (SCHEMA, "td3_bc", "short-train"):
        raise ValueError("expected a short-train td3_bc pool written by freeze_scores.py")
    row = json.loads((pool_dir / "resolved.json").read_text(encoding="utf8"))
    # Pools frozen before the thinned bank existed (hopper, 2026-09-30) lack the field; None is the whole-component
    # population split freeze_scores.split_row writes, and verify_preparation checks the run's input hashes below.
    row["protocol"]["reservation"].setdefault("rows_per_episode", None)
    step, path, recorded = FS.locate_checkpoint(pool_dir)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if recorded is None or digest != recorded:
        raise ValueError("checkpoint bytes differ from the sha256 recorded in frozen.json")
    (_, args, spec, _), prepared = FS.prepare(row, FS.DEFAULT_DATA if data_dir is None else data_dir)
    verification = FS.verify_preparation(pool_dir, prepared)
    config = spec.config()
    rng, template, models = FS.P.initialize(args, config, prepared.training.obs.shape[1],
                                            prepared.training.action.shape[1], prepared.max_action)
    restored = serialization.from_bytes({"state": template, "training_rng": rng, "step": jnp.int32(0)}, payload)
    if int(restored["step"]) != step or not FS.same_leaves(payload, restored):
        raise ValueError("the checkpoint did not restore exactly into the P.initialize template")
    state = restored["state"]
    FS.R._accept(state, {})
    return dict(pool_dir=pool_dir, meta=meta, row=row, args=args, config=config, prepared=prepared, state=state,
                template=template, rng=rng, models=models, step=step, checkpoint_path=path, checkpoint_sha256=digest,
                verification=verification)


def score_population(args, models, state, scales, data, key, max_action, batch_size):
    """Every population row as freeze_scores.score_rows scores it (one refresh noise draw over the rows,
    eager forward passes in chunks), with both heads, the parent's scale for the gate and each fitted
    signal's eta and sigma. Returns float32 NumPy arrays."""
    import jax
    import jax.numpy as jnp

    from calibration.reference import positive_scale
    from experiments.wbcp.freeze_scores import KEY_REFRESH_NOISE

    n, action_dim = data.action.shape
    noise = jax.random.normal(jax.random.fold_in(key, KEY_REFRESH_NOISE), (n, action_dim)) * args.policy_noise
    noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    native, parts = state.native, {}
    for start in range(0, n, batch_size):
        chunk = jax.tree_util.tree_map(lambda x: x[start:start + batch_size], data)
        action = models[0].apply(native.actor_target.params, chunk.next_obs)
        action = jnp.clip(action + noise[start:start + batch_size], -max_action, max_action)
        q_next = models[1].apply(native.critic_target.params, chunk.next_obs, action).min(axis=-1)
        target = jax.lax.stop_gradient(chunk.reward + (1.0 - chunk.done) * args.discount * q_next)
        values = dict(target=target, q=models[1].apply(native.critic.params, chunk.obs, chunk.action),
                      policy_action=models[0].apply(native.actor.params, chunk.obs),
                      parent_sigma=positive_scale(models[2].apply(state.calibrator.params, chunk.obs, chunk.action),
                                                  state.residual_scale))
        for signal, (calibrator, unit) in scales.items():
            eta = models[2].apply(calibrator.params, chunk.obs, chunk.action)
            values["eta_" + signal], values["sigma_" + signal] = eta, positive_scale(eta, unit)
        for name, value in values.items():
            parts.setdefault(name, []).append(np.asarray(value))
    scored = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    if scored["q"].shape != (n, 2):
        raise ValueError("expected TD3+BC's twin critic")
    if not all(np.all(np.isfinite(v)) for v in scored.values()) or not all(
            np.all(scored["sigma_" + s] > 0) for s in scales):
        raise FloatingPointError("nonfinite scores or a nonpositive fitted scale")
    return scored


def reproduction_gate(scored, parent, data):
    """Largest absolute difference from the parent's frozen.npz; the fit refuses anything but zeros."""
    residual = (scored["target"] - scored["q"].min(axis=-1)).astype(np.float64)
    return dict(residual=float(np.max(np.abs(residual - parent["residual"]))),
                sigma=float(np.max(np.abs(scored["parent_sigma"].astype(np.float64) - parent["sigma"]))),
                policy_action=float(np.max(np.abs(scored["policy_action"] - parent["policy_action"]))),
                obs=float(np.max(np.abs(np.asarray(data.obs) - parent["obs"]))),
                action=float(np.max(np.abs(np.asarray(data.action) - parent["action"]))))


def signal_arrays(parent, scored, signal):
    """The frozen.npz contract for one signal: residual t - v and that signal's sigma, scores as freeze_scores."""
    from experiments.wbcp import freeze_scores as FS

    residual = (scored["target"] - signal_value(signal, scored["target"], scored["q"])).astype(np.float64)
    sigma = scored["sigma_" + signal].astype(np.float64)
    arrays = dict(score=np.abs(residual) / sigma, residual=residual, sigma=sigma,
                  **{name: parent[name] for name in ("obs", "action", "policy_action", "episode", "timestep", "row",
                                                       "in_training")})
    FS.check_contract(arrays)
    return arrays


def _sha_file(path):
    from runtime.provenance import sha

    return sha(path) if Path(path).is_file() else None


def dataset_key(environment):
    """The configs/td3_bc.yaml dataset key (hopper, ...) of a D4RL environment name (hopper-medium-v2, ...)."""
    found = [key for key, entry in H.config("td3_bc")["datasets"].items() if entry["environment"] == environment]
    if len(found) != 1:
        raise ValueError(f"no unique td3_bc dataset for environment {environment!r}")
    return found[0]


def _critic_health(pool_dir, key):
    try:
        return H.critic_health(str(pool_dir), key)
    except (OSError, KeyError, ValueError, TypeError) as error:  # a pool without a training log, or an older format
        return dict(status="unknown", error=f"{type(error).__name__}: {error}")


def fit(pool_dir, output, *, signals=SIGNALS, updates=None, block=1000, score_batch=None, data_dir=None):
    """Restore the pool, fit a fresh scale per signal, score the population and write DIR."""
    import jax
    from flax import serialization

    from experiments.wbcp import freeze_scores as FS
    from runtime.provenance import write

    signals = tuple(signals)
    if not signals or len(set(signals)) != len(signals) or not set(signals) <= set(SIGNALS):
        raise ValueError(f"signals must be distinct names from {SIGNALS}")
    if any(value is not None and value < 1 for value in (updates, block, score_batch)):
        raise ValueError("updates, block and score_batch must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pool = restore_pool(pool_dir, data_dir)
    meta, args, config, prepared, state = pool["meta"], pool["args"], pool["config"], pool["prepared"], pool["state"]
    pool_dir = pool["pool_dir"]
    with np.load(pool_dir / "frozen.npz", allow_pickle=False) as archive:
        parent = {name: archive[name] for name in archive.files}
    npz_sha = _sha_file(pool_dir / "frozen.npz")
    if npz_sha != meta["npz_sha256"]:
        raise ValueError("the parent's frozen.npz differs from the sha256 in its frozen.json")
    if not np.array_equal(np.asarray(prepared.heldout_ids, np.int64), parent["row"]) or parent["in_training"].any():
        raise ValueError("the population rows differ from the parent's frozen.npz")
    updates = int(meta["updates"]) if updates is None else int(updates)
    score_batch = int(meta["target_noise"]["score_batch"]) if score_batch is None else int(score_batch)
    restored_at = time.perf_counter()

    scales, log, fingerprints = fit_scales(args, config, pool["models"], state.native, pool["template"],
                                           prepared.training, signals, updates, block, pool["rng"],
                                           prepared.max_action)
    fitted_at = time.perf_counter()
    scored = score_population(args, pool["models"], state, scales, prepared.heldout,
                              jax.random.PRNGKey(meta["score_seed"]), prepared.max_action, score_batch)
    gate = reproduction_gate(scored, parent, prepared.heldout)
    if max(gate.values()) > 0:
        raise ValueError(f"reproduction gate not exact: {gate}; nothing is written")
    scored_at = time.perf_counter()

    key = dataset_key(meta["dataset"])
    steady = [entry for entry in log[1:] if entry["updates"] == log[0]["updates"]]  # other lengths recompile
    per_thousand = 1000 * sum(e["seconds"] for e in steady) / sum(e["updates"] for e in steady) if steady else None
    timing = dict(restore_seconds=restored_at - started, fit_seconds=fitted_at - restored_at,
                  first_block_seconds_with_compile=log[0]["seconds"], steady_seconds_per_1000_updates=per_thousand,
                  projected_fit_seconds_for_parent_count=None if per_thousand is None
                  else per_thousand * int(meta["updates"]) / 1000, scoring_seconds=scored_at - fitted_at)
    settings = dict(
        signals=list(signals), updates=updates, batch_size=args.batch_size, block=block,
        optimizer=f"optax.adam(cal_lr={config.cal_lr}) (algorithms.td3_bc_bca.initialize)",
        cal_beta=config.cal_beta, width_penalty=config.width_penalty, scale_ema=config.scale_ema,
        coverage_target=1.0 - config.posterior.alpha, posterior=vars(config.posterior), blend=config.blend,
        initialization=("fresh: the P.initialize template's calibrator (the parent run's initial scale network, "
                        "key fold_in(rng, 1128352841)) and residual unit 1, the same for every signal"),
        stream=("the parent run's training stream (freeze_scores.train_arm): carry rng from P.initialize, each update "
                "splits (kr, kb, kn), draws batch indices randint(kb, (batch_size,), 0, training rows) and gives kn to "
                "the fit: target noise fold_in(kn, 1179210836), bootstrap prior fold_in(kn, 1128352850)"),
        objective="algorithms.td3_bc_bca.fit_scale with residual t - v (fit_signal_scale / scale_update)",
        critic="frozen: actor, critic and target heads of the parent checkpoint", training_rows=len(prepared.training.obs))
    parent_record = dict(
        pool=str(pool_dir), dataset_key=key, frozen_json_sha256=_sha_file(pool_dir / "frozen.json"), npz_sha256=npz_sha,
        resolved_json_sha256=_sha_file(pool_dir / "resolved.json"),
        training_log_sha256=_sha_file(pool_dir / "training_log.json"),
        checkpoint=dict(step=pool["step"], path=str(pool["checkpoint_path"]), sha256=pool["checkpoint_sha256"]),
        residual_unit=meta["residual_unit"], critic_health=_critic_health(pool_dir, key))
    for signal in signals:
        directory = output / signal
        directory.mkdir()
        arrays = signal_arrays(parent, scored, signal)
        v = signal_value(signal, scored["target"], scored["q"])
        unit = float(scales[signal][1])
        metadata = dict(
            schema=SCHEMA, algorithm="td3_bc", method="bca", source="frozen-signal", signal=signal,
            signal_definition=SIGNAL_TEXT[signal], run_id=meta["run_id"], dataset=meta["dataset"],
            dataset_file=meta["dataset_file"], dataset_sha256=meta["dataset_sha256"],
            updates=meta["updates"], seed=meta["seed"], score_seed=meta["score_seed"],
            target_noise=dict(meta["target_noise"], score_batch=score_batch),
            score_definition=dict(
                score="|residual| / sigma in float64 from float32 t - v and sigma, as freeze_scores",
                residual=SIGNAL_TEXT[signal], sigma="max(eta, 1e-6) * u of this signal's fitted scale",
                y=meta["score_definition"]["y"], u=unit,
                updates="updates is the frozen critic's training count; sigma_fit.updates the scale's"),
            residual_unit=unit, parent=parent_record, sigma_fit=dict(settings, accepted_updates=fingerprints[
                "accepted_updates"][signal], fingerprints={k: fingerprints[k] for k in ("batch_indices_sha256",
                                                                                       "targets_sha256")}),
            obs_normalization=meta["obs_normalization"], population=meta["population"], rows=meta["rows"],
            preparation=dict(meta["preparation"], verification=pool["verification"]),
            configuration=meta["configuration"], split=meta.get("split"), gate=gate,
            arrays={name: dict(dtype=str(value.dtype), shape=list(value.shape)) for name, value in arrays.items()},
            diagnostics=FS._diagnostics(arrays, dict(target=scored["target"], q=v, eta=scored["eta_" + signal])),
            git=FS.git_identity(), code=dict(script_sha256=_sha_file(Path(__file__).resolve()),
                                             source_files_sha256=FS._source_digest(FS.source_files())),
            timing=timing, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        FS.write_artifact(directory, arrays, metadata)
        with (directory / "scale.msgpack").open("xb") as handle:
            handle.write(serialization.to_bytes({"calibrator": scales[signal][0], "residual_scale": scales[signal][1]}))
    heads = dict(target=scored["target"], q=scored["q"], row=parent["row"], signals=np.array(signals),
                 unit=np.array([np.float32(scales[s][1]) for s in signals], np.float32),
                 **{f"{kind}_{s}": scored[f"{kind}_{s}"] for s in signals for kind in ("eta", "sigma")})
    with (output / "heads.npz").open("xb") as handle:
        np.savez(handle, **heads)
    record = dict(schema=FIT_SCHEMA, signals=list(signals), dataset=meta["dataset"], parent=parent_record,
                  settings=settings, fingerprints=fingerprints, gate=gate, verification=pool["verification"],
                  residual_unit={s: float(scales[s][1]) for s in signals}, blocks=log,
                  heads_sha256=_sha_file(output / "heads.npz"),
                  artifacts={s: dict(npz_sha256=_sha_file(output / s / "frozen.npz"),
                                     scale_sha256=_sha_file(output / s / "scale.msgpack")) for s in signals},
                  timing=dict(timing, total_seconds=time.perf_counter() - started),
                  git=FS.git_identity(), created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    write(output / "fit.json", FS.R._json_value(record))
    return dict(record=record, scored=scored, scales=scales, pool=pool, parent=parent)


# ---- evaluate (NumPy; JAX only for the dose, after the workers) ----

def residual_magnitudes(heads):
    """|t - v| in float64 for each residual type, from the float32 heads as the artifacts compute them;
    maxabs = max(|t - Q1|, |t - Q2|), so covering it means covering both heads."""
    t, q = heads["target"], heads["q"]
    out = {name: np.abs((t - signal_value(name, t, q)).astype(np.float64)) for name in ("min", "q1")}
    out["q2"] = np.abs((t - q[:, 1]).astype(np.float64))
    out["maxabs"] = np.maximum(out["q1"], out["q2"])
    return out


@dataclass(frozen=True)
class Population:
    signals: tuple
    scores: dict  # (signal, residual) -> |residual| / sigma_signal, float64 [N]
    risks: dict  # (signal, residual) -> ExactRisk with uniform mass: population miscoverage at a threshold
    groups: tuple  # episode grouping for the reservation sampler
    sigma_mean: dict  # signal -> population mean of sigma

    @property
    def size(self):
        return self.scores[self.signals[0], self.signals[0]].size


def cross_scores(heads, signals):
    """(signal, residual) -> |residual| / sigma_signal in float64; (s, s) is signal s's own artifact score."""
    mags = residual_magnitudes(heads)
    return {(s, r): mags[r] / heads["sigma_" + s].astype(np.float64) for s in signals for r in RESIDUALS}


def population(heads, episode, timestep, signals):
    scores = cross_scores(heads, signals)
    ones = np.ones(len(heads["target"]))
    risks = {key: ExactRisk(value, ones) for key, value in scores.items()}
    return Population(tuple(signals), scores, risks, episode_groups(np.asarray(episode), np.asarray(timestep)),
                      {s: float(np.mean(heads["sigma_" + s].astype(np.float64))) for s in signals})


def run_bank(pop, design, b, seed, options):
    """One bank: the same rows for every signal, each signal's threshold on its own scores (same posterior draws),
    then the exact population miscoverage of every residual type under every band: [signal, residual]."""
    groups = pop.groups if design["per_episode"] is not None else None
    rows = draw_calibration(stream(seed, "rows", design["name"], b), design["n"], pop.size, groups,
                            design["per_episode"], design["spacing"])
    thresholds = np.array([calibrate(pop.scores[s, s][rows], stream(seed, "wbcp", design["name"], b), **options).threshold
                           for s in pop.signals])
    miscoverage = np.array([[float(pop.risks[s, r](t)) for r in RESIDUALS] for s, t in zip(pop.signals, thresholds)])
    return rows.size, thresholds, miscoverage


_TASK = None  # inherited by forked workers


def _bank_chunk(bounds):
    pop, design, seed, options = _TASK
    return [run_bank(pop, design, b, seed, options) for b in range(*bounds)]


def run_banks(pop, design, banks, seed, options, workers=1):
    """Every bank of a design; each bank's streams are keyed by (seed, design, bank), so --workers changes nothing."""
    chunk = max(1, math.ceil(banks / (4 * workers)))
    tasks = [(start, min(start + chunk, banks)) for start in range(0, banks, chunk)]
    global _TASK
    _TASK = (pop, design, seed, options)
    try:
        if workers == 1:
            parts = [_bank_chunk(task) for task in tasks]
        else:
            with multiprocessing.get_context("fork").Pool(workers) as executor:
                parts = executor.map(_bank_chunk, tasks, chunksize=1)
    finally:
        _TASK = None
    records = [record for part in parts for record in part]
    return (np.array([r[0] for r in records]), np.stack([r[1] for r in records]), np.stack([r[2] for r in records]))


def failure_summary(miscoverage, alpha):
    trials, failures = miscoverage.size, int(np.sum(miscoverage > alpha))
    low, high = stats.binomtest(failures, trials).proportion_ci(method="exact")
    return dict(fail=failures / trials, ci=[float(low), float(high)], failures=failures, trials=trials,
                mean_miscoverage=float(np.mean(miscoverage)), miscoverage_q95=float(np.quantile(miscoverage, 0.95)),
                excess_given_fail=float(np.mean(miscoverage[miscoverage > alpha] - alpha)) if failures else None)


def width_ratios(pop, thresholds):
    """Paired mean over banks (both signals certified) of (threshold_a mean sigma_a) / (threshold_b mean sigma_b)."""
    out = {}
    for i, a in enumerate(pop.signals):
        for j, b in enumerate(pop.signals):
            if i != j:
                both = np.isfinite(thresholds[:, i]) & np.isfinite(thresholds[:, j])
                ratio = (thresholds[both, i] * pop.sigma_mean[a]) / (thresholds[both, j] * pop.sigma_mean[b])
                out[f"{a}/{b}"] = float(np.mean(ratio)) if both.any() else None
    return out


def summarize_design(pop, design, sizes, thresholds, miscoverage, lambda_star, alpha):
    signals = {}
    for i, s in enumerate(pop.signals):
        certified = np.isfinite(thresholds[:, i])
        mean_threshold = float(np.mean(thresholds[certified, i])) if certified.any() else None
        signals[s] = dict(
            abstain=1.0 - float(np.mean(certified)), threshold_mean=mean_threshold,
            threshold_over_lambda_star=None if mean_threshold is None else mean_threshold / lambda_star[s],
            half_width_mean=None if mean_threshold is None else mean_threshold * pop.sigma_mean[s],
            failure={r: failure_summary(miscoverage[:, i, k], alpha) for k, r in enumerate(RESIDUALS)})
    return dict(design=design, bank_size=dict(mean=float(np.mean(sizes)), min=int(sizes.min()), max=int(sizes.max())),
                signals=signals, width_ratio=width_ratios(pop, thresholds))


def dose_tables(heads, signals, lambda_star, thresholds, blend, dose_banks=None):
    """BCA's BC dose at every population row, from each signal's eta, u and threshold, through
    calibration.dose.frozen_level_dose (a ready FrozenReference; only its threshold, unit and finite
    parameters enter): exact statistics at lambda*, and per-bank statistics averaged over the first
    dose_banks banks of each design (default all). Within a bank, the dose is increasing in eta, so it
    is computed on eta sorted ascending and its percentiles are read in that order. A float32 rounding
    that breaks the order falls back to np.quantile."""
    import jax
    import jax.numpy as jnp

    from calibration.dose import frozen_level_dose
    from calibration.reference import FrozenReference

    @jax.jit
    def dose(eta, unit, threshold):
        unit, threshold = jnp.asarray(unit, jnp.float32), jnp.asarray(threshold, jnp.float32)
        reference = FrozenReference({}, unit, threshold, threshold, threshold, jnp.asarray(0.0, jnp.float32),
                                    jnp.asarray(True))
        out = frozen_level_dose(reference, eta, blend)
        return out.dose, out.inputs_valid, jnp.all(jnp.diff(out.dose) >= 0)

    def describe(values, ordered):
        values = np.asarray(values, np.float64)
        if ordered:
            position = np.array([0.05, 0.95]) * (values.size - 1)
            low = np.floor(position).astype(np.int64)
            high = np.minimum(low + 1, values.size - 1)
            p05, p95 = values[low] + (position - low) * (values[high] - values[low])
        else:
            p05, p95 = np.quantile(values, [0.05, 0.95])
        return dict(mean=float(values.mean()), sd=float(values.std()), p05=float(p05), p95=float(p95))

    out = {}
    for i, s in enumerate(signals):
        eta = np.sort(np.asarray(heads["eta_" + s], np.float32))
        unit = np.float32(heads["unit"][i])
        values, valid, _ = dose(eta, unit, lambda_star[s])
        if not bool(valid):
            raise FloatingPointError(f"invalid dose inputs for signal {s}")
        entry = dict(lambda_star=describe(values, False),
                     lambda_eta_mean_at_lambda_star=float(np.mean(lambda_star[s] * np.maximum(eta.astype(np.float64), 1e-6))),
                     banks={})
        for name, table in thresholds.items():
            column = table[:, i] if dose_banks is None else table[:dose_banks, i]
            rows = []
            for threshold in column:
                values, valid, ordered = dose(eta, unit, threshold)
                if not bool(valid):
                    raise FloatingPointError(f"invalid dose inputs for signal {s}")
                rows.append(describe(values, bool(ordered)))
            means = np.array([r["mean"] for r in rows])
            entry["banks"][name] = dict({k: float(np.mean([r[k] for r in rows])) for k in ("mean", "sd", "p05", "p95")},
                                        banks=len(rows), mean_across_banks_min=float(means.min()),
                                        mean_across_banks_max=float(means.max()))
        out[s] = entry
    return out


def head_disagreement(heads):
    """The critic-alignment study's head comparison at the logged actions (expectations B2)."""
    q = heads["q"].astype(np.float64)
    gap = q[:, 0] - q[:, 1]
    residual_min = residual_magnitudes(heads)["min"]
    return dict(q1_above_q2=float(np.mean(gap > 0)), median_abs_gap=float(np.median(np.abs(gap))),
                median_abs_residual_min=float(np.median(residual_min)),
                gap_over_residual=float(np.median(np.abs(gap)) / np.median(residual_min)))


def evaluate_arrays(heads, episode, timestep, signals, designs, *, banks, seed, alpha=0.1, beta=0.95, draws=1000,
                    blend=0.5, workers=1, dose_banks=None):
    """Cross-coverage, widths and doses for float32 heads (target, q, eta_<s>, sigma_<s>, unit) and episode ids."""
    started = time.perf_counter()
    pop = population(heads, episode, timestep, signals)
    lambda_star = {s: pop.risks[s, s].lambda_star(alpha) for s in signals}
    options = dict(alpha=alpha, beta=beta, draws=draws)
    results, thresholds = [], {}
    for design in designs:
        sizes, table, miscoverage = run_banks(pop, design, banks, seed, options, workers)
        thresholds[design["name"]] = table
        results.append(summarize_design(pop, design, sizes, table, miscoverage, lambda_star, alpha))
    banked = time.perf_counter()
    doses = dose_tables(heads, signals, lambda_star, thresholds, blend, dose_banks)
    at_star = dict(
        lambda_star=lambda_star, sigma_mean=pop.sigma_mean,
        half_width={s: lambda_star[s] * pop.sigma_mean[s] for s in signals},
        half_width_ratio={f"{a}/{b}": (lambda_star[a] * pop.sigma_mean[a]) / (lambda_star[b] * pop.sigma_mean[b])
                          for a in signals for b in signals if a != b},
        miscoverage={s: {r: float(pop.risks[s, r](lambda_star[s])) for r in RESIDUALS} for s in signals},
        fraction_own_score_at_most_one={s: float(np.mean(pop.scores[s, s] <= 1.0)) for s in signals})
    return dict(rows=pop.size, episodes=int(np.unique(episode).size), population=at_star, designs=results,
                doses=doses, head_disagreement=head_disagreement(heads),
                residual_unit={s: float(heads["unit"][i]) for i, s in enumerate(signals)},
                timing=dict(bank_seconds=banked - started, dose_seconds=time.perf_counter() - banked))


def load_signal_dir(directory):
    """fit.json, heads.npz and every signal artifact (through d4rl_benchmark.load_pool), cross-checked."""
    directory = Path(directory)
    record = json.loads((directory / "fit.json").read_text(encoding="utf8"))
    signals = tuple(record["signals"])
    with np.load(directory / "heads.npz", allow_pickle=False) as archive:
        heads = {name: archive[name] for name in archive.files}
    if tuple(heads["signals"]) != signals:
        raise ValueError("heads.npz and fit.json list different signals")
    pools = {s: load_pool(str(directory / s), "heldout") for s in signals}
    first = pools[signals[0]]
    if first.size != len(heads["target"]) or not np.array_equal(first.row, heads["row"]):
        raise ValueError("the artifacts' population differs from heads.npz (rows in training?)")
    scores = cross_scores(heads, signals)
    for s, pool in pools.items():
        if not (np.array_equal(pool.row, first.row) and np.array_equal(pool.episode, first.episode)
                and np.array_equal(pool.timestep, first.timestep)):
            raise ValueError("signal artifacts disagree on the population rows")
        if not np.array_equal(pool.scores["normalized"], scores[s, s]):
            raise ValueError(f"heads.npz does not reproduce the {s} artifact's scores")
    return record, heads, first, pools


def default_designs(key, iid_n=IID_N):
    n, k, _ = H.bank("td3_bc", key)
    return [dict(name="iid", label=f"Independent rows, n={iid_n}", n=iid_n, per_episode=None, spacing="random"),
            dict(name="reservation", label=f"Configured bank, n={n}, K={k}, BCA sampler", n=n, per_episode=k,
                 spacing="reservation")]


def evaluate(directory, *, banks, seed=2026100201, workers=1, iid_n=IID_N, designs=None, dose_banks=None,
             output=None):
    directory = Path(directory)
    record, heads, first, _ = load_signal_dir(directory)
    key = record["parent"]["dataset_key"]
    posterior, blend = record["settings"]["posterior"], record["settings"]["blend"]
    designs = default_designs(key, iid_n) if designs is None else designs
    result = evaluate_arrays(heads, first.episode, first.timestep, tuple(record["signals"]), designs, banks=banks,
                             seed=seed, alpha=posterior["alpha"], beta=posterior["credibility"],
                             draws=posterior["draws"], blend=blend, workers=workers, dose_banks=dose_banks)
    parent = record["parent"]["pool"]
    health = _critic_health(parent, key) if Path(parent).is_dir() else record["parent"]["critic_health"]
    result = dict(schema=EVALUATION_SCHEMA, directory=str(directory.resolve()), dataset=record["dataset"],
                  dataset_key=key, parent=parent, critic_health=health,
                  critic_converged=health.get("status") == "ok" if health else None,
                  settings=dict(banks=banks, seed=seed, alpha=posterior["alpha"], beta=posterior["credibility"],
                                draws=posterior["draws"], blend=blend, workers=workers, dose_banks=dose_banks,
                                threshold="lambda_dep = max(lambda_hat, lambda_hpd) (BCA's deployed threshold)",
                                coverage="|r| / sigma_s <= threshold_s; maxabs residual: both heads",
                                fit=dict(updates=record["settings"]["updates"],
                                         accepted_updates=record["fingerprints"]["accepted_updates"])),
                  fit_sha256=_sha_file(directory / "fit.json"), heads_sha256=_sha_file(directory / "heads.npz"),
                  script_sha256=_sha_file(Path(__file__).resolve()),
                  created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), **result)
    output = directory / f"evaluation_b{banks}_s{seed}.json" if output is None else Path(output)
    with output.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=1, allow_nan=False)
    return result, output


def print_evaluation(result):
    print(f"{result['dataset']}  rows={result['rows']}  episodes={result['episodes']}  "
          f"critic={result['critic_health'].get('status') if result['critic_health'] else '?'}  "
          f"banks={result['settings']['banks']}  fit updates={result['settings']['fit']['updates']}")
    pop = result["population"]
    print("lambda*: " + "  ".join(f"{s}={v:.4g}" for s, v in pop["lambda_star"].items())
          + "   head gap / residual: " + f"{result['head_disagreement']['gap_over_residual']:.3f}")
    for block in result["designs"]:
        print(f"\n[{block['design']['label']}]  bank size {block['bank_size']['mean']:.0f}")
        print(f"  {'band':<8}" + "".join(f"{'fail ' + r:>24}" for r in RESIDUALS) + f"{'thr/lambda*':>13}{'abstain':>9}")
        for s, entry in block["signals"].items():
            cells = "".join(f"{f['fail']:>8.1%} [{f['ci'][0]:.1%}, {f['ci'][1]:.1%}]".rjust(24)
                            for f in (entry["failure"][r] for r in RESIDUALS))
            ratio = "-" if entry["threshold_over_lambda_star"] is None else f"{entry['threshold_over_lambda_star']:.3f}"
            print(f"  {s:<8}{cells}{ratio:>13}{entry['abstain']:>9.1%}")
        print("  width ratios: " + "  ".join(f"{k}={v:.3f}" for k, v in block["width_ratio"].items() if v is not None))
    print("\ndose (blend 0.5): mean / row SD / p05-p95")
    for s, entry in result["doses"].items():
        cells = [f"lambda* {entry['lambda_star']['mean']:.3f}/{entry['lambda_star']['sd']:.3f}/"
                 f"{entry['lambda_star']['p05']:.3f}-{entry['lambda_star']['p95']:.3f}"]
        cells += [f"{name} {d['mean']:.3f}/{d['sd']:.3f}/{d['p05']:.3f}-{d['p95']:.3f}" for name, d in entry["banks"].items()]
        print(f"  {s:<8}" + "   ".join(cells))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("fit", help="fit a fresh scale per signal on a frozen pool and write the artifacts")
    f.add_argument("--pool", type=Path, required=True, help="short-train TD3+BC pool (runs/wbcp_frozen/<pool>)")
    f.add_argument("--signals", nargs="+", choices=SIGNALS, default=list(SIGNALS))
    f.add_argument("--updates", type=int, help="scale updates per signal (default: the parent's count)")
    f.add_argument("--block", type=int, default=1000, help="updates per jax.lax.scan block")
    f.add_argument("--score-batch", type=int, help="rows per forward-pass chunk (default: the parent's, for the gate)")
    f.add_argument("--data-dir", type=Path, help="cached D4RL HDF5 directory (default ~/.d4rl/datasets)")
    f.add_argument("--output", type=Path, required=True, help="new directory")
    e = commands.add_parser("evaluate", help="cross-coverage, widths and doses on a fit directory")
    e.add_argument("directory", type=Path)
    e.add_argument("--banks", type=int, required=True)
    e.add_argument("--seed", type=int, default=2026100201)
    e.add_argument("--n-iid", type=int, default=IID_N, help="rows of the independent-row design")
    e.add_argument("--dose-banks", type=int, help="banks per design whose dose statistics are averaged (default all)")
    e.add_argument("--workers", type=int, default=1, help="forked processes for the banks (Linux)")
    e.add_argument("--output", type=Path, help="JSON path (default DIR/evaluation_b<banks>_s<seed>.json); must not exist")
    opt = parser.parse_args(argv)
    if opt.command == "fit":
        if opt.output.exists():
            parser.error("--output already exists")
        if len(set(opt.signals)) != len(opt.signals):
            parser.error("--signals has repeated names")
        if (opt.updates is not None and opt.updates < 1) or opt.block < 1 or (opt.score_batch or 1) < 1:
            parser.error("--updates, --block and --score-batch must be positive")
        try:
            result = fit(opt.pool, opt.output, signals=opt.signals, updates=opt.updates, block=opt.block,
                         score_batch=opt.score_batch, data_dir=opt.data_dir)
        except BaseException as error:
            if opt.output.is_dir() and not (opt.output / "failure.json").exists():
                with (opt.output / "failure.json").open("w", encoding="utf8") as handle:
                    json.dump(dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()),
                              handle, indent=1)
            raise
        r = result["record"]
        print(json.dumps(dict(dataset=r["dataset"], signals=r["signals"], updates=r["settings"]["updates"],
                              accepted=r["fingerprints"]["accepted_updates"], residual_unit=r["residual_unit"],
                              gate=r["gate"], critic_health=r["parent"]["critic_health"], timing=r["timing"]),
                         indent=1, default=str))
        return result
    if opt.banks < 1 or opt.workers < 1 or opt.n_iid < 1 or (opt.dose_banks is not None and opt.dose_banks < 1):
        parser.error("--banks, --workers, --n-iid and --dose-banks must be positive")
    if opt.output is not None and opt.output.exists():
        parser.error("--output already exists")
    result, path = evaluate(opt.directory, banks=opt.banks, seed=opt.seed, workers=opt.workers, iid_n=opt.n_iid,
                            dose_banks=opt.dose_banks, output=opt.output)
    print_evaluation(result)
    print(f"\nwritten {path}  (banks {result['timing']['bank_seconds']:.1f} s, doses {result['timing']['dose_seconds']:.1f} s)")
    return result


BLAS_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

if __name__ == "__main__":
    # evaluate forks NumPy workers: one BLAS thread per process (a site hook may load NumPy before this line, so the
    # limit needs a fresh interpreter, as in d4rl_benchmark.py). The fit runs on XLA, which these do not limit.
    if "evaluate" in sys.argv[1:2] and any(os.environ.get(name) != "1" for name in BLAS_THREADS):
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, **dict.fromkeys(BLAS_THREADS, "1")))
    main()
