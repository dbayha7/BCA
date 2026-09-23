"""Run and verify the plain IQL host; shared evaluation/checkpoint utilities."""

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

PRIMARY = (
    "hopper-medium-v2",
    "halfcheetah-medium-expert-v2",
    "walker2d-medium-replay-v2",
    "maze2d-large-v1",
    "pen-human-v1",
    "pen-cloned-v1",
    "pen-expert-v1",
)
EVAL_SEED = 20260920
CURVE_BASE = 1129644032
FINAL_BASE = 1179189248
DRIVER_REL = Path("runtime/iql.py")
CHECKPOINT_VERIFICATION = "Base driver verifies SHA256; train.py additionally writes decoded optimizer-counter checks."


@dataclass(frozen=True)
class Options:
    dataset: str
    seed: int
    output_dir: str
    mode: str = "shared"
    scope: str = "development"
    updates: int = 100000
    eval_interval: int = 5000
    curve_episodes: int = 2
    final_episodes: int = 20
    reserve_size: int = 8192
    warmup: int = 10000
    refresh_interval: int = 5000
    eval_seed: int = EVAL_SEED
    prepare_only: bool = False
    host_parameters: dict | None = None
    posterior_parameters: dict | None = None


def validate_options(o):
    if (
        o.dataset not in PRIMARY
        or o.mode not in ("shared", "native")
        or o.scope not in ("development", "diagnostic")
    ):
        raise ValueError("unknown dataset, mode or experiment scope")
    if not Path(o.output_dir).is_absolute():
        raise ValueError("output_dir must be absolute")
    if not 0 <= o.seed <= 4294967295 or o.eval_seed != EVAL_SEED:
        raise ValueError(
            "uint32 training seed and fixed evaluation seed20260920 required"
        )
    for name in (
        "updates",
        "eval_interval",
        "curve_episodes",
        "final_episodes",
        "refresh_interval",
    ):
        if getattr(o, name) < 1:
            raise ValueError(name + " must be positive")
    if o.reserve_size < 1:
        raise ValueError("Both methods require a positive held-out reservation.")
    if o.warmup < 0 or (o.mode == "shared" and o.warmup >= o.updates):
        raise ValueError("shared warmup must leave at least one posterior actor update")
    if (
        o.curve_episodes >= 65536
        or CURVE_BASE + o.updates // o.eval_interval * 65536 >= FINAL_BASE
    ):
        raise ValueError("curve episode IDs would overlap the final bank")
    if o.final_episodes > 4294967295 - FINAL_BASE:
        raise ValueError("too many final episodes")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources(root, inventory):
    for name, digest in inventory.items():
        if sha(Path(root) / name) != digest:
            raise ValueError("source hash mismatch: " + name)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, float) and (not math.isfinite(value)):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def write_json(path, value, *, progress=False):
    path = Path(path)
    target = path.with_suffix(path.suffix + ".tmp") if progress else path
    with open(target, "w" if progress else "x") as f:
        json.dump(plain(value), f, indent=2, allow_nan=False)
    if progress:
        os.replace(target, path)


def load_modules():
    alg = Path(__file__).resolve().parent
    sys.path[:0] = [str(alg)]
    h = importlib.import_module("calibration.iql_state")
    s = importlib.import_module("calibration.iql_actors")
    if Path(h.__file__).resolve() != alg.parent / "calibration/iql_state.py":
        raise RuntimeError("host import did not resolve to this source copy")
    import numpy as np
    from flax import serialization
    from runtime.evaluation_keys import evaluation_key

    return SimpleNamespace(
        H=h,
        S=s,
        P=h.P,
        jax=h.jax,
        jnp=h.jnp,
        np=np,
        serialization=serialization,
        evaluation_key=evaluation_key,
    )


def resolved_arguments(o, m):
    if not isinstance(o.host_parameters, dict) or not isinstance(
        o.posterior_parameters, dict
    ):
        raise ValueError("Explicit host and BCA configuration required.")
    values = dict(o.host_parameters)
    values.update(
        algorithm="iql_host",
        posterior=m.P.PosteriorArgs(
            **o.posterior_parameters,
            mode="full" if o.mode == "shared" else "off",
            reserve_size=o.reserve_size,
            warmup=o.warmup,
            refresh_interval=o.refresh_interval,
        ),
    )
    args = m.H.Args(**values)
    args.posterior.validate()
    m.H.C.check_reward_transform(
        "corl_iql",
        o.dataset,
        args.reward_transform,
        args.reward_scale,
        args.reward_bias,
    )
    differences = m.H._published_config.diff_against_published(
        "corl_iql", o.dataset, args
    )
    if differences is None or any(
        (field != "num_updates" or got != o.updates for field, got, want in differences)
    ):
        raise ValueError(
            "IQL host hyperparameters differ from the recorded native host."
        )
    arms = (m.S.Arm("host", "off", args.beta, 1.0),)
    return (
        args,
        arms,
        [
            dict(
                arm="host",
                actual_differences=differences,
                declaration="Native host on the shared reserved pool; explicit update budget.",
            )
        ],
    )


def episode_ids(final, index, count):
    base = FINAL_BASE if final else CURVE_BASE + index * 65536
    return list(range(base, base + count))


def environment_seed(m, seed, identity):
    key = m.evaluation_key(None, seed, identity)
    return int(m.jax.random.randint(key, (), 0, 2**31 - 1, dtype=m.jnp.int32))


def evaluation_schedule(o, m):

    def point(final, index, count, step):
        ids = episode_ids(final, index, count)
        return {
            "step": step,
            "evaluation_ids": ids,
            "environment_seeds": [environment_seed(m, o.eval_seed, i) for i in ids],
        }

    return {
        "mapping": "fold_in(PRNGKey(eval_seed), uint32 episode_id); randint scalar int32 [0, 2**31-1)",
        "curve": [
            point(False, i, o.curve_episodes, step)
            for i, step in enumerate(
                range(o.eval_interval, o.updates + 1, o.eval_interval)
            )
        ],
        "final": point(True, 0, o.final_episodes, o.updates),
    }


def block_counts(metrics, count, actors, np):
    applied = np.asarray(metrics["actor_updated"])
    ready = np.asarray(metrics["posterior_ready"])
    if (
        applied.dtype != np.bool_
        or applied.shape != (count, actors)
        or ready.dtype != np.bool_
        or (ready.shape != (count,))
    ):
        raise ValueError("invalid actor update or posterior-ready diagnostics")
    valid = np.asarray(metrics["actor_inputs_valid"]) & np.asarray(
        metrics["actor_proposal_valid"]
    )
    if valid.shape != applied.shape:
        raise ValueError("misaligned actor validity flags")
    return (
        applied.sum(axis=0, dtype=np.int64),
        (valid & ~applied).sum(axis=0, dtype=np.int64),
        (applied & ready[:, None]).sum(axis=0, dtype=np.int64),
    )


class ScalarEvaluator:

    def __init__(
        self, env, policy, normalize_scores, seed_for_id, np, eval_seed=EVAL_SEED
    ):
        self.env, self.policy, self.normalize_scores, self.seed_for_id, self.np = (
            env,
            policy,
            normalize_scores,
            seed_for_id,
            np,
        )
        self.eval_seed = eval_seed
        self.max_steps = int(env.spec.max_episode_steps)
        if self.max_steps < 1:
            raise ValueError("environment must declare a positive max episode horizon")
        self.obs_shape = tuple(env.observation_space.shape)
        self.action_shape = tuple(env.action_space.shape)

    def _finite_state(self, obs):
        np = self.np
        if np.shape(obs) != self.obs_shape or not np.all(np.isfinite(obs)):
            raise FloatingPointError("nonfinite or malformed evaluation observation")
        sim = getattr(self.env, "sim", None)
        if sim is not None:
            for name in ("qpos", "qvel"):
                if not np.all(np.isfinite(getattr(sim.data, name))):
                    raise FloatingPointError("nonfinite simulator " + name)

    def evaluate(self, params, ids):
        np = self.np
        returns = []
        lengths = []
        seeds = []
        terminated = []
        truncated = []
        limited = []
        for identity in ids:
            seed = int(self.seed_for_id(self.eval_seed, identity))
            seeds.append(seed)
            py_state = random.getstate()
            np_state = np.random.get_state()
            try:
                random.seed(seed)
                np.random.seed(seed)
                try:
                    result = self.env.reset(seed=seed)
                except TypeError:
                    self.env.seed(seed)
                    result = self.env.reset()
                obs = (
                    result[0]
                    if isinstance(result, tuple)
                    and len(result) == 2
                    and isinstance(result[1], dict)
                    else result
                )
                self._finite_state(obs)
                total = 0.0
                term = False
                trunc = False
                for length in range(1, self.max_steps + 1):
                    policy_obs = np.asarray(obs, dtype=np.float32)
                    if not np.all(np.isfinite(policy_obs)):
                        raise FloatingPointError("nonfinite float32 policy observation")
                    proposal = self.policy(params, policy_obs)
                    if isinstance(proposal, tuple):
                        proposal, input_valid = proposal
                        if not bool(input_valid):
                            raise FloatingPointError(
                                "nonfinite normalized policy observation"
                            )
                    action = np.asarray(proposal)
                    if action.shape != self.action_shape or not np.all(
                        np.isfinite(action)
                    ):
                        raise FloatingPointError(
                            "nonfinite or malformed evaluation action"
                        )
                    result = self.env.step(action)
                    if len(result) == 5:
                        obs, reward, term, trunc, info = result
                    elif len(result) == 4:
                        obs, reward, done, info = result
                        trunc = bool(info.get("TimeLimit.truncated", False))
                        term = bool(done) and (not trunc)
                    else:
                        raise ValueError("unknown step API")
                    self._finite_state(obs)
                    if np.ndim(reward) != 0 or not np.isfinite(reward):
                        raise FloatingPointError("nonfinite reward")
                    total += float(reward)
                    if not math.isfinite(total):
                        raise FloatingPointError("nonfinite cumulative reward")
                    if term or trunc:
                        break
                returns.append(total)
                lengths.append(length)
                terminated.append(bool(term))
                truncated.append(bool(trunc))
                limited.append(length == self.max_steps and (not (term or trunc)))
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
        returns = np.asarray(returns, dtype=np.float64)
        scores = np.asarray(self.normalize_scores(returns), dtype=np.float64)
        if scores.shape != returns.shape or not np.all(np.isfinite(scores)):
            raise FloatingPointError("invalid normalized scores")
        return dict(
            returns=returns,
            scores=scores,
            evaluation_ids=np.asarray(ids, np.uint32),
            environment_seeds=np.asarray(seeds, np.uint32),
            episode_lengths=np.asarray(lengths, np.int32),
            terminated=np.asarray(terminated, bool),
            truncated=np.asarray(truncated, bool),
            horizon_limited=np.asarray(limited, bool),
        )


def first_failure(metrics, start_step, np):
    required = (
        "nuisance_valid",
        "calibration_valid",
        "actor_inputs_valid",
        "actor_proposal_valid",
        "inputs_valid",
    )
    if any((k not in metrics for k in required)):
        raise ValueError("missing separate validity diagnostics")
    flags = {k: np.asarray(metrics[k]) for k in required}
    if any((v.dtype != np.bool_ for v in flags.values())):
        raise TypeError("validity diagnostics must be boolean")
    n = flags["inputs_valid"].shape[0]
    valid = np.ones(n, bool)
    for value in flags.values():
        if value.shape[0] != n:
            raise ValueError("misaligned block diagnostics")
        valid &= value.reshape(n, -1).all(axis=1)
    bad = np.flatnonzero(~valid)
    if not len(bad):
        return None
    i = int(bad[0])
    return {
        "step": start_step + i,
        "block_offset": i,
        "flags": {k: plain(v[i]) for k, v in flags.items()},
    }


def run_schedule(runtime, o, on_block):
    step = 0
    next_refresh = o.warmup if o.mode == "shared" else o.updates + 1
    while step < o.updates:
        if step == next_refresh:
            runtime.refresh(step)
            next_refresh += o.refresh_interval
        end = min(
            o.updates, (step // o.eval_interval + 1) * o.eval_interval, next_refresh
        )
        metrics = runtime.advance(end - step)
        failed = on_block(step + 1, end, metrics)
        step = end
        if failed:
            return {
                "status": "failed",
                "steps_completed": step,
                "first_failure": failed,
                "stop_further_dispatch": True,
            }
        if step % o.eval_interval == 0:
            runtime.curve(step, step // o.eval_interval - 1)
    runtime.finish(step)
    return {
        "status": "complete",
        "steps_completed": step,
        "first_failure": None,
        "stop_further_dispatch": False,
    }


def save_npz(path, np, **arrays):
    converted = {k: np.asarray(v) for k, v in arrays.items()}
    if any((v.dtype.hasobject for v in converted.values())):
        raise TypeError("object arrays are forbidden")
    with open(path, "xb") as f:
        np.savez_compressed(f, **converted)
    return sha(path)


def verify_final(
    path, count, steps, eval_seed, ids, np, *, training_seed=None, scope=None
):
    with np.load(path, allow_pickle=False) as data:
        for name in ("final_returns", "final_scores"):
            a = data[name]
            if a.shape != (count,) or not np.all(np.isfinite(a)):
                raise ValueError("invalid numeric final " + name)
        for name, want in [
            ("steps_completed", steps),
            ("eval_seed", eval_seed),
            ("training_seed", training_seed),
        ]:
            if want is not None and (
                data[name].shape != ()
                or data[name].dtype.kind not in "iu"
                or int(data[name]) != want
            ):
                raise ValueError("final step/seed mismatch: " + name)
        if scope is not None and str(data["experiment_scope"]) != scope:
            raise ValueError("final scope mismatch")
        if data["evaluation_ids"].dtype != np.uint32 or not np.array_equal(
            data["evaluation_ids"], ids
        ):
            raise ValueError("final episode IDs mismatch")
    return sha(path)


def verify_evaluation(path, point, o, np, *, final):
    count = o.final_episodes if final else o.curve_episodes
    prefix = "final_" if final else ""
    with np.load(path, allow_pickle=False) as data:
        for name in ("returns", "scores"):
            x = data[prefix + name]
            if (
                x.shape != (count,)
                or x.dtype.kind not in "fiu"
                or (not np.isfinite(x).all())
            ):
                raise ValueError("invalid evaluation numeric " + name)
        for name, want in [
            ("steps_completed", point["step"]),
            ("eval_seed", o.eval_seed),
            ("training_seed", o.seed),
        ]:
            x = data[name]
            if x.shape != () or x.dtype.kind not in "iu" or int(x) != want:
                raise ValueError("evaluation step/seed mismatch")
        if str(data["experiment_scope"]) != o.scope:
            raise ValueError("evaluation scope mismatch")
        for name in ("evaluation_ids", "environment_seeds"):
            x = data[name]
            if (
                x.dtype != np.uint32
                or x.shape != (count,)
                or (not np.array_equal(x, point[name]))
            ):
                raise ValueError("evaluation IDs/seeds mismatch")
        lengths = data["episode_lengths"]
        if (
            lengths.shape != (count,)
            or lengths.dtype.kind not in "iu"
            or np.any(lengths < 1)
        ):
            raise ValueError("invalid evaluation episode lengths")
        endings = []
        for name in ("terminated", "truncated", "horizon_limited"):
            x = data[name]
            if x.shape != (count,) or x.dtype != np.bool_:
                raise ValueError("invalid evaluation termination flags")
            endings.append(x)
        if not np.logical_or.reduce(endings).all():
            raise ValueError("evaluation episode has no ending")
    return sha(path)


def verify_block_evidence(metrics, count, arms, mode, np):
    actors = len(arms)
    for name, shape in [
        ("inputs_valid", (count,)),
        ("nuisance_valid", (count,)),
        ("calibration_valid", (count,)),
        ("actor_inputs_valid", (count, actors)),
        ("actor_proposal_valid", (count, actors)),
    ]:
        x = np.asarray(metrics.get(name))
        if x.shape != shape or x.dtype != np.bool_:
            raise ValueError("invalid diagnostic flag shape/type: " + name)

    def numeric(name, shape, low=None, high=None, *, roundoff=False):
        x = np.asarray(metrics.get(name))
        if x.shape != shape or x.dtype.kind not in "fiu" or (not np.isfinite(x).all()):
            raise ValueError("missing/nonfinite/malformed diagnostic: " + name)
        margin = 8 * np.finfo(x.dtype).eps if roundoff and x.dtype.kind == "f" else 0.0
        if (
            low is not None
            and np.any(x < low)
            or (high is not None and np.any(x > high + margin))
        ):
            raise ValueError("diagnostic outside range: " + name)
        return x

    numeric("q_loss", (count,), 0.0)
    numeric("value_loss", (count,), 0.0)
    numeric("actor_loss", (count, actors) if mode == "shared" else (count,))
    applied = np.asarray(metrics.get("actor_updated"))
    if applied.shape != (count, actors) or applied.dtype != np.bool_:
        raise ValueError("invalid actor update flags")
    if mode == "shared":
        shape = (count, actors)
        mean = numeric("weight_mean", shape, 0.0, 100.0)
        ess = numeric("weight_ess_fraction", shape, 0.0, 1.0, roundoff=True)
        cap = numeric("weight_cap_fraction", shape, 0.0, 1.0)
        support = numeric("supported_fraction", shape, 0.0, 1.0)
        numeric("scale_loss", (count,))
        numeric("scale_coverage", (count,), 0.0, 1.0, roundoff=True)
        if np.any(cap > support):
            raise ValueError("cap fraction exceeds support")
        unsupported = support == 0
        if np.any(unsupported & ((mean != 0) | (ess != 0) | (cap != 0))):
            raise ValueError("unsupported actor has nonzero weight evidence")
        if not np.array_equal(applied, support > 0):
            raise ValueError("support/update mismatch")
        off = [i for i, a in enumerate(arms) if a["mode"] == "off"]
        if not np.all(support[:, off] == 1):
            raise ValueError("native actor must retain row support")
    if any(
        (not np.all(applied[:, i]) for i, a in enumerate(arms) if a["mode"] == "off")
    ):
        raise ValueError("native actor skipped an update")


def _evidence_count(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError("invalid evidence count: " + name)
    return value


def _evidence_number(value, name, *, infinity=False):
    if infinity and value == "Infinity":
        return math.inf
    if type(value) not in (int, float):
        raise ValueError("invalid numeric reference: " + name)
    try:
        value = float(value)
    except OverflowError as e:
        raise ValueError("unrepresentable reference: " + name) from e
    if not math.isfinite(value) and (not (infinity and value == math.inf)):
        raise ValueError("nonfinite reference: " + name)
    return value


def verify_reference_evidence(record, posterior, data):
    train = _evidence_count(data.get("training_size"), "training_size")
    cal = _evidence_count(data.get("calibration_size"), "calibration_size")
    if _evidence_count(data.get("dataset_rows"), "dataset_rows") != train + cal:
        raise ValueError("training/calibration count partition mismatch")
    if _evidence_count(record.get("calibration_rows"), "calibration_rows") != cal:
        raise ValueError("reference calibration count mismatch")
    fit = _evidence_count(posterior.get("fit_size"), "fit_size")
    if _evidence_count(record.get("fit_rows"), "fit_rows") != min(fit, train):
        raise ValueError("reference fitting count mismatch")
    alpha = _evidence_number(posterior.get("alpha"), "alpha")
    credibility = _evidence_number(posterior.get("credibility"), "credibility")
    draws = _evidence_count(posterior.get("draws"), "draws")
    if not 0 < alpha < 1 or not 0 < credibility < 1 or draws < 2:
        raise ValueError("invalid declared posterior configuration")
    rank = int(
        (Decimal(cal + 1) * (Decimal(1) - Decimal(str(alpha)))).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    if rank > cal:
        raise ValueError(
            "global calibration bank cannot support configured finite rank"
        )
    if posterior.get("mode") != "full" or posterior.get("use_bootstrap") is not True:
        raise ValueError(
            "shared reference must retain full posterior and Bayesian scale fitting"
        )
    unit = _evidence_number(record.get("residual_unit"), "residual_unit")
    gain = _evidence_number(record.get("decision_gain"), "decision_gain")
    if unit <= 0 or gain <= 0 or gain != posterior.get("decision_gain"):
        raise ValueError("invalid reference unit or decision gain")
    radii = {
        name: _evidence_number(record.get(name), name, infinity=True)
        for name in ("full_radius", "bayesian_radius", "conformal_radius")
    }
    if any((x < 0 for x in radii.values())) or radii["full_radius"] != max(
        radii["bayesian_radius"], radii["conformal_radius"]
    ):
        raise ValueError(
            "reference must satisfy nonnegative full=max(Bayesian,conformal)"
        )
    for name in ("training_key", "posterior_key"):
        key = record.get(name)
        if (
            not isinstance(key, list)
            or len(key) != 2
            or any((type(x) is not int or not 0 <= x <= 4294967295 for x in key))
        ):
            raise ValueError("invalid reference uint32 key: " + name)
    components = record.get("component_sha256", {})
    required = [
        name + "_" + kind
        for name in ("actor", "online_q", "target_q", "value", "calibrator")
        for kind in ("params", "optimizer")
    ]
    required += [
        "residual_unit",
        "posterior_key",
        "posterior_quantile_draws",
        "fit_indices",
        "cal_residuals",
        "cal_predictions",
        "fit_predictions",
        "radii",
    ]
    if not isinstance(components, dict) or any(
        (
            not isinstance(components.get(k), str)
            or len(components[k]) != 64
            or any((c not in "0123456789abcdef" for c in components[k]))
            for k in required
        )
    ):
        raise ValueError("missing or invalid reference component digest")
    if record.get("common_component_sha256") != {
        k: v for k, v in components.items() if not k.startswith("actor_")
    }:
        raise ValueError("common/reference component digest mismatch")
    if (
        record.get("bootstrap_weights_saved") is not False
        or record.get("posterior_quantile_draws_hashed") is not True
    ):
        raise ValueError("incorrect bootstrap/draw provenance declaration")
    return radii


def verify_result(output, result, o, np):
    output = Path(output)
    declared = json.loads((output / "resolved_args.json").read_text())
    if declared["driver_options"] != asdict(o):
        raise ValueError("resolved driver options mismatch")
    data = json.loads((output / "data_metadata.json").read_text())
    _evidence_count(data.get("training_size"), "training_size")
    if _evidence_count(data.get("calibration_size"), "calibration_size") < 1:
        raise ValueError("The matched host requires the declared held-out reservation.")
    names = [a["name"] for a in declared["arms"]]
    final = result.get("artifacts", {}).get("final", {})
    if (
        len(names) != (2 if o.mode == "shared" else 1)
        or len(set(names)) != len(names)
        or set(final) != set(names)
    ):
        raise ValueError("missing or unexpected final actor artifacts")
    if result["status"] != "complete" or result["steps_completed"] != o.updates:
        raise ValueError("group did not complete its declared budget")
    if "training_end" not in result["artifacts"]:
        raise ValueError("missing final training state")
    refs = (
        list(range(o.warmup, o.updates, o.refresh_interval))
        if o.mode == "shared"
        else []
    )
    ends = sorted(
        {o.updates, *range(o.eval_interval, o.updates + 1, o.eval_interval), *refs}
        - {0}
    )
    blocks = result["blocks"]
    if [(b["first_step"], b["last_step"]) for b in blocks] != list(
        zip([1] + [x + 1 for x in ends[:-1]], ends)
    ):
        raise ValueError("block boundaries do not match complete schedule")
    applied = np.zeros(len(names), np.int64)
    abstained = applied.copy()
    post_applied = applied.copy()
    for b in blocks:
        path = output / b["diagnostics_npz"]
        if sha(path) != b["sha256"]:
            raise ValueError("block hash mismatch")
        first, last = (b["first_step"], b["last_step"])
        count = last - first + 1
        with np.load(path, allow_pickle=False) as archive:
            metrics = {
                k: archive[k]
                for k in archive.files
                if k not in ("experiment_scope", "training_seed", "eval_seed")
            }
            if (
                str(archive["experiment_scope"]) != o.scope
                or int(archive["training_seed"]) != o.seed
                or int(archive["eval_seed"]) != o.eval_seed
            ):
                raise ValueError("block scope/seed mismatch")
        if (
            first_failure(metrics, first, np) is not None
            or b["first_failure"] is not None
        ):
            raise ValueError("invalid block diagnostics")
        verify_block_evidence(metrics, count, declared["arms"], o.mode, np)
        if len(metrics["inputs_valid"]) != count:
            raise ValueError("block diagnostic row count mismatch")
        expected_ready = (
            np.arange(first, last + 1) > o.warmup
            if o.mode == "shared"
            else np.zeros(count, bool)
        )
        if not np.array_equal(metrics["posterior_ready"], expected_ready):
            raise ValueError("posterior readiness schedule mismatch")
        a, b_count, p = block_counts(metrics, count, len(names), np)
        applied += a
        abstained += b_count
        post_applied += p
        actor_steps = np.asarray(metrics["actor_step"])
        if (
            actor_steps.shape != (count, len(names))
            or actor_steps.dtype.kind not in "iu"
            or (
                not np.array_equal(
                    actor_steps,
                    applied - a + np.cumsum(metrics["actor_updated"], axis=0),
                )
            )
        ):
            raise ValueError("actor step diagnostics mismatch")
    for key, counts in [
        ("actor_applied_updates", applied),
        ("actor_valid_abstentions", abstained),
        ("actor_post_ready_applied_updates", post_applied),
    ]:
        if result[key] != {name: int(n) for name, n in zip(names, counts)}:
            raise ValueError("actor counts mismatch: " + key)
    if result["state_counters"] != {
        "qf": o.updates,
        "vf": o.updates,
        "qf_target": o.updates,
        "calibrator": o.updates if o.mode == "shared" else None,
    }:
        raise ValueError("actual state counter mismatch")
    if result["nuisance_valid_updates"] != o.updates or result[
        "calibration_valid_updates"
    ] != (o.updates if o.mode == "shared" else 0):
        raise ValueError("common valid update count mismatch")
    schedule = declared["evaluation_schedule"]
    curve_points = schedule["curve"]
    if [p["step"] for p in curve_points] != list(
        range(o.eval_interval, o.updates + 1, o.eval_interval)
    ):
        raise ValueError("curve schedule mismatch")
    curves = result["artifacts"]["curves"]
    if set(curves) != {str(p["step"]) for p in curve_points}:
        raise ValueError("missing curve artifacts")
    for i, point in enumerate(curve_points):
        if point["evaluation_ids"] != episode_ids(False, i, o.curve_episodes):
            raise ValueError("curve IDs mismatch")
        if set(curves[str(point["step"])]) != set(names):
            raise ValueError("missing curve actor")
        for name in names:
            entry = curves[str(point["step"])][name]
            if (
                entry["path"] != f"curve_{point['step']}_{name}.npz"
                or verify_evaluation(output / entry["path"], point, o, np, final=False)
                != entry["sha256"]
            ):
                raise ValueError("curve archive hash/path mismatch")
    references = result["artifacts"]["references"]
    if set(references) != {str(s) for s in refs}:
        raise ValueError("reference refresh list mismatch")
    reference_radii = {}
    for step in refs:
        entry = references[str(step)]
        path = output / f"reference_{step}.json"
        if entry["path"] != path.name or sha(path) != entry["sha256"]:
            raise ValueError("reference JSON hash/path mismatch")
        record = json.loads(path.read_text())
        if (
            record["step"] != step
            or record["training_seed"] != o.seed
            or record["eval_seed"] != o.eval_seed
            or (record["experiment_scope"] != o.scope)
        ):
            raise ValueError("reference step/scope/seed mismatch")
        if (
            record["actor_states"]
            != result["artifacts"][f"reference_{step}"]["actor_states"]
        ):
            raise ValueError("reference actor state hash mismatch")
        reference_radii[step] = verify_reference_evidence(
            record, declared["args"]["posterior"], data
        )
    if o.mode == "shared":
        for b in blocks:
            consumed = [s for s in refs if s < b["first_step"]]
            radius = reference_radii[max(consumed)] if consumed else None
            with np.load(output / b["diagnostics_npz"], allow_pickle=False) as archive:
                support = archive["supported_fraction"]
                for i, arm in enumerate(declared["arms"]):
                    r = (
                        0.0
                        if radius is None or arm["mode"] == "off"
                        else radius[
                            (
                                "full_radius"
                                if arm["mode"] == "full"
                                else "conformal_radius"
                            )
                        ]
                    )
                    if not np.all(support[:, i] == (0.0 if math.isinf(r) else 1.0)):
                        raise ValueError("consumed global radius/support mismatch")
    ids = np.asarray(episode_ids(True, 0, o.final_episodes), np.uint32)
    point = schedule["final"]
    if point["step"] != o.updates or point["evaluation_ids"] != ids.tolist():
        raise ValueError("final schedule mismatch")
    for name in names:
        path = output / f"final_{name}.npz"
        digest = verify_final(
            path,
            o.final_episodes,
            o.updates,
            o.eval_seed,
            ids,
            np,
            training_seed=o.seed,
            scope=o.scope,
        )
        if digest != final[name]["sha256"]:
            raise ValueError("final archive hash mismatch")
        if verify_evaluation(path, point, o, np, final=True) != digest:
            raise ValueError("final verification mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if (
                int(archive["actor_applied_updates"])
                != result["actor_applied_updates"][name]
            ):
                raise ValueError("final actor counter mismatch")
    for record in result["artifacts"].values():
        if "path" in record and sha(output / record["path"]) != record["sha256"]:
            raise ValueError("checkpoint hash mismatch")
    if (
        o.scope == "diagnostic"
        and o.mode == "shared"
        and any(
            (
                post_applied[i] < 1
                for i, a in enumerate(declared["arms"])
                if a["mode"] != "off"
            )
        )
    ):
        raise ValueError(
            "diagnostic exposure gate: no post-ready applied update for a posterior actor; not numeric invalidity"
        )
    return True


def runtime_dependencies():
    result = {}
    for name in ("jax", "flax", "optax", "numpy", "gym", "d4rl"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        result[name] = {
            "version": importlib.metadata.version(name),
            "module_entry": str(path),
            "module_entry_sha256": sha(path),
        }
    return result


class GroupRuntime:

    def __init__(self, o, m, output, args, arms):
        self.o, self.m, self.output, self.args, self.arms = (
            o,
            m,
            Path(output),
            args,
            arms,
        )
        h, s, np = (m.H, m.S, m.np)
        self.data = h.load_data(args)
        write_json(self.output / "data_metadata.json", self.data.metadata)
        self.env = h.C.gym.make(o.dataset)
        try:
            state, key, actor = h.initialize_agent(
                args,
                self.env.observation_space.shape[0],
                self.env.action_space.shape[0],
                float(self.env.action_space.high[0]),
            )
            self.fitter = None
            self.carry = (key, state, m.jnp.int32(0))
            native_step = h.BASE.make_train_step(
                args,
                state.actor.apply_fn,
                state.qf.apply_fn,
                state.vf.apply_fn,
                self.data.train,
            )

            def step(carry, unused):
                result, diag = native_step(carry, unused)
                agent = result[1]
                nuisance = m.P.finite_tree(
                    (
                        agent.qf,
                        agent.qf_target,
                        agent.vf,
                        diag["q_loss"],
                        diag["value_loss"],
                    )
                )
                actor_valid = m.P.finite_tree((agent.actor, diag["actor_loss"]))
                return (
                    result,
                    {
                        **diag,
                        "nuisance_valid": nuisance,
                        "calibration_valid": m.jnp.asarray(True),
                        "actor_inputs_valid": m.jnp.array([actor_valid]),
                        "actor_proposal_valid": m.jnp.array([actor_valid]),
                        "actor_updated": m.jnp.array([actor_valid]),
                        "actor_step": m.jnp.array([agent.actor.step]),
                        "posterior_ready": m.jnp.asarray(False),
                        "inputs_valid": nuisance & actor_valid,
                    },
                )

            self.scan = m.jax.jit(
                lambda carry, n: m.jax.lax.scan(step, carry, None, n), static_argnums=1
            )

            def policy_fn(params, obs):
                normalized = (obs - self.data.obs_mean) / self.data.obs_std
                return (
                    actor.act(params, normalized),
                    m.jnp.all(m.jnp.isfinite(normalized)),
                )

            policy = m.jax.jit(policy_fn)
            self.evaluator = ScalarEvaluator(
                self.env,
                policy,
                lambda r: h.C.normalized_scores(o.dataset, r),
                lambda seed, identity: environment_seed(m, seed, identity),
                np,
                o.eval_seed,
            )
            self.actor_applied = np.zeros(len(arms), np.int64)
            self.actor_abstained = self.actor_applied.copy()
            self.actor_post_applied = self.actor_applied.copy()
            self.nuisance_valid_count = 0
            self.calibration_valid_count = 0
            self.artifacts = {"curves": {}, "references": {}}
            self.phase = "initialized"
            self.timings = {
                "training_scan_seconds": 0.0,
                "evaluation_seconds": 0.0,
                "posterior_refresh_seconds": 0.0,
            }
        except BaseException:
            self.env.close()
            raise

    def actors(self):
        return (
            [self.m.S.actor_at(self.carry.actors, i) for i in range(len(self.arms))]
            if self.o.mode == "shared"
            else [self.carry[1].actor]
        )

    def checkpoint(self, label):
        payload = self.m.serialization.to_bytes(self.carry)
        path = self.output / (label + ".msgpack")
        with open(path, "xb") as f:
            f.write(payload)
        record = {
            "path": path.name,
            "sha256": sha(path),
            "format": "Flax state dict msgpack; numeric arrays, no pickle",
            "experiment_scope": self.o.scope,
            "training_seed": self.o.seed,
            "eval_seed": self.o.eval_seed,
            "actor_states": {
                arm.name: {
                    "params_sha256": self.m.P.fingerprint(actor.params),
                    "optimizer_sha256": self.m.P.fingerprint(actor.opt_state),
                    "applied_steps": int(actor.step),
                }
                for arm, actor in zip(self.arms, self.actors())
            },
        }
        self.artifacts[label] = record
        return record

    def advance(self, count):
        self.phase = "training"
        start = time.perf_counter()
        self.carry, metrics = self.scan(self.carry, count)
        metrics = self.m.jax.device_get(metrics)
        applied, abstained, post = block_counts(
            metrics, count, len(self.arms), self.m.np
        )
        self.actor_applied += applied
        self.actor_abstained += abstained
        self.actor_post_applied += post
        self.nuisance_valid_count += int(
            self.m.np.asarray(metrics["nuisance_valid"]).sum()
        )
        self.timings["training_scan_seconds"] += time.perf_counter() - start
        return metrics

    def curve(self, step, index):
        self.phase = "curve_evaluation"
        start = time.perf_counter()
        ids = episode_ids(False, index, self.o.curve_episodes)
        rows = {
            arm.name: self.evaluator.evaluate(actor.params, ids)
            for arm, actor in zip(self.arms, self.actors())
        }
        entries = {}
        for name, row in rows.items():
            path = self.output / f"curve_{step}_{name}.npz"
            digest = save_npz(
                path,
                self.m.np,
                **row,
                steps_completed=step,
                eval_seed=self.o.eval_seed,
                training_seed=self.o.seed,
                experiment_scope=self.o.scope,
            )
            entries[name] = {"path": path.name, "sha256": digest}
        self.artifacts["curves"][str(step)] = entries
        self.timings["evaluation_seconds"] += time.perf_counter() - start

    def finish(self, steps):
        self.phase = "final_validation"
        if steps != self.o.updates or int(self.carry[2]) != steps:
            raise ValueError("finalization requires all declared updates")
        for arm, actor, applied in zip(self.arms, self.actors(), self.actor_applied):
            if int(actor.step) != int(applied) or not bool(self.m.P.finite_tree(actor)):
                raise ValueError("actor budget/state mismatch: " + arm.name)
            if arm.mode == "off" and int(applied) != steps:
                raise ValueError("native actor skipped an update")
        if self.nuisance_valid_count != steps:
            raise ValueError("invalid nuisance update budget")
        counters = self.state_counters()
        if any((counters[k] != steps for k in ("qf", "vf", "qf_target"))):
            raise ValueError("actual Q/V/target step counter mismatch")
        self.checkpoint("training_end")
        self.phase = "final_evaluation"
        start = time.perf_counter()
        ids = episode_ids(True, 0, self.o.final_episodes)
        rows = {
            arm.name: self.evaluator.evaluate(actor.params, ids)
            for arm, actor in zip(self.arms, self.actors())
        }
        final = {}
        for arm in self.arms:
            row = rows[arm.name]
            path = self.output / f"final_{arm.name}.npz"
            save_npz(
                path,
                self.m.np,
                final_returns=row.pop("returns"),
                final_scores=row.pop("scores"),
                **row,
                steps_completed=steps,
                eval_seed=self.o.eval_seed,
                training_seed=self.o.seed,
                experiment_scope=self.o.scope,
                actor_applied_updates=self.actor_applied[list(self.arms).index(arm)],
            )
            final[arm.name] = {
                "path": path.name,
                "sha256": verify_final(
                    path,
                    self.o.final_episodes,
                    steps,
                    self.o.eval_seed,
                    self.m.np.asarray(ids, self.m.np.uint32),
                    self.m.np,
                    training_seed=self.o.seed,
                    scope=self.o.scope,
                ),
                "episodes": self.o.final_episodes,
            }
        self.artifacts["final"] = final
        self.phase = "complete"
        self.timings["evaluation_seconds"] += time.perf_counter() - start

    def state_counters(self):
        agent = self.carry[1]
        return {
            **{
                name: int(getattr(agent, name).step)
                for name in ("qf", "vf", "qf_target")
            },
            "calibrator": (
                int(self.carry.extra.calibration.calibrator.step)
                if self.o.mode == "shared"
                else None
            ),
        }


def execute_group(o, m, output, runtime_factory=GroupRuntime):
    output = Path(output)
    start = time.perf_counter()
    runtime = None
    record = {
        "status": "running",
        "started_utc": utc(),
        "experiment_scope": o.scope,
        "selection_allowed": False,
        "checkpoint_verification": CHECKPOINT_VERIFICATION,
        "training_seed": o.seed,
        "eval_seed": o.eval_seed,
        "stop_further_dispatch": False,
        "blocks": [],
        "steps_completed": 0,
        "first_failure": None,
    }
    try:
        args, arms, differences = resolved_arguments(o, m)
        write_json(
            output / "resolved_args.json",
            {
                "args": asdict(args),
                "arms": [asdict(a) for a in arms],
                "published_differences": differences,
                "driver_options": asdict(o),
                "unused_native_posterior_schedule_fields": (
                    ["warmup", "refresh_interval"] if o.mode == "native" else []
                ),
                "evaluation_schedule": evaluation_schedule(o, m),
                "evaluation_deviation": "fixed20260920 per-episode scalar seeds, same for all arms/native control; native standalone schedule not used",
                "runtime_dependencies": runtime_dependencies(),
                "dependency_hash_scope": "imported package entry files and installed versions; not complete wheel hashes",
            },
        )
        runtime = runtime_factory(o, m, output, args, arms)

        def block(first, last, metrics):
            failed = first_failure(metrics, first, m.np)
            path = output / f"block_{first}_{last}.npz"
            digest = save_npz(
                path,
                m.np,
                **metrics,
                experiment_scope=o.scope,
                training_seed=o.seed,
                eval_seed=o.eval_seed,
            )
            record["blocks"].append(
                {
                    "first_step": first,
                    "last_step": last,
                    "diagnostics_npz": path.name,
                    "sha256": digest,
                    "means": {
                        k: m.np.asarray(v).mean(axis=0) for k, v in metrics.items()
                    },
                    "first_failure": failed,
                }
            )
            record["steps_completed"] = last
            write_json(output / "progress.json", record, progress=True)
            print(
                json.dumps(plain({"scope": o.scope, "block": record["blocks"][-1]})),
                flush=True,
            )
            return failed

        record.update(run_schedule(runtime, o, block))
        if record["status"] != "complete":
            runtime.checkpoint("failed_block_end")
    except BaseException as error:
        record.update(
            status="failed",
            error=repr(error),
            phase=getattr(runtime, "phase", "setup"),
            stop_further_dispatch=True,
            traceback=traceback.format_exc(),
        )
        if runtime is not None:
            try:
                runtime.checkpoint("failed_attempt")
            except Exception as e:
                record["checkpoint_error"] = repr(e)
    finally:
        if runtime is not None:
            try:
                runtime.env.close()
                record["environment_closed"] = True
            except Exception as e:
                record.update(
                    status="failed",
                    close_error=repr(e),
                    environment_closed=False,
                    stop_further_dispatch=True,
                )
            record.update(
                artifacts=runtime.artifacts,
                nuisance_attempted_updates=int(runtime.carry[2]),
                nuisance_valid_updates=runtime.nuisance_valid_count,
                actor_attempted_updates=int(runtime.carry[2]) * len(runtime.arms),
                actor_applied_updates={
                    a.name: int(n) for a, n in zip(runtime.arms, runtime.actor_applied)
                },
                calibration_valid_updates=runtime.calibration_valid_count,
                actor_valid_abstentions={
                    a.name: int(n)
                    for a, n in zip(runtime.arms, runtime.actor_abstained)
                },
                actor_post_ready_applied_updates={
                    a.name: int(n)
                    for a, n in zip(runtime.arms, runtime.actor_post_applied)
                },
                state_counters=runtime.state_counters(),
                diagnostic_exposure={
                    "required": o.scope == "diagnostic" and o.mode == "shared",
                    "all_posterior_actors_applied_after_ready": all(
                        (
                            n > 0
                            for a, n in zip(runtime.arms, runtime.actor_post_applied)
                            if a.mode != "off"
                        )
                    ),
                    "semantics": "acceptance exposure gate, separate from numerical validity",
                },
                timings=getattr(runtime, "timings", {}),
            )
        record.update(finished_utc=utc(), elapsed_seconds=time.perf_counter() - start)
        write_json(output / "result.json", record)
    return record
