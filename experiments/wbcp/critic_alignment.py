"""Does calibrating min(Q1, Q2) hide error in the Q1 that TD3+BC's actor climbs?

TD3+BC's actor maximizes Q1 alone (algorithms/td3_bc.py), while BCA scores the residual of
min(Q1, Q2) (algorithms/td3_bc_bca.py). This script works on frozen TD3+BC pools
(runs/wbcp_frozen/<run>, written by freeze_scores.py in short-train mode). It restores each
checkpoint byte for byte and recomputes both critic heads at the logged action and at the policy
action pi(s). It checks that their min reproduces the frozen residual exactly, then compares three
scores under the frozen scale sigma:
- s_min = |t - min(Q1, Q2)| / sigma, BCA's score today;
- s_1 = |t - Q1| / sigma, the actor-aligned score;
- s_max = max(s_min, s_1).

The comparison uses population coverage and 1,000 WBCP banks of independent rows per pool.
Expectations: runs/wbcp_critic_alignment/expectations.md.

JAX_PLATFORMS=cpu python experiments/wbcp/critic_alignment.py runs/wbcp_frozen/<run> [...]
python experiments/wbcp/critic_alignment.py --analyze-only runs/wbcp_critic_alignment/<run>/heads.npz [...]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from calibration.wbcp import calibrate  # noqa: E402

OUTPUT = ROOT / "runs" / "wbcp_critic_alignment"
HEADS = ("target", "q", "q_pi", "sigma", "sigma_pi", "episode")  # heads.npz contract


def extract(run_dir, data_dir=None):
    """Restore a short-train TD3+BC pool and return per-row heads, with the reproduction gate."""
    import jax
    import jax.numpy as jnp
    from flax import serialization

    from experiments.wbcp import freeze_scores as FS

    run_dir = Path(run_dir).resolve()
    meta = json.loads((run_dir / "frozen.json").read_text(encoding="utf8"))
    if (meta["algorithm"], meta["source"]) != ("td3_bc", "short-train"):
        raise ValueError("expected a short-train td3_bc pool")
    frozen = np.load(run_dir / "frozen.npz")
    row = json.loads((run_dir / "resolved.json").read_text(encoding="utf8"))
    # Pools frozen before the thinned bank existed (hopper, 2026-09-30) lack the field. None is the
    # whole-component population split FS.split_row writes; verify_preparation below checks the result
    # against the run's recorded input hashes.
    row["protocol"]["reservation"].setdefault("rows_per_episode", None)
    step, path, recorded = FS.locate_checkpoint(run_dir)
    payload = path.read_bytes()
    if recorded is not None and hashlib.sha256(payload).hexdigest() != recorded:
        raise ValueError("checkpoint bytes differ from the recorded sha256")
    (_, args, spec, _), prepared = FS.prepare(row, FS.DEFAULT_DATA if data_dir is None else data_dir)
    verification = FS.verify_preparation(run_dir, prepared)
    rng, template, models = FS.P.initialize(args, spec.config(), prepared.training.obs.shape[1],
                                            prepared.training.action.shape[1], prepared.max_action)
    restored = serialization.from_bytes({"state": template, "training_rng": rng, "step": jnp.int32(0)}, payload)
    if int(restored["step"]) != step or not FS.same_leaves(payload, restored):
        raise ValueError("the checkpoint did not restore exactly")
    state, data = restored["state"], prepared.heldout
    if not np.array_equal(np.asarray(prepared.heldout_ids, np.int64), frozen["row"]):
        raise ValueError("the population rows differ from frozen.npz")

    # As FS.score_rows: one target-noise draw over the population, then chunked forward passes.
    n, action_dim = data.action.shape
    batch = meta["target_noise"]["score_batch"]
    key = jax.random.PRNGKey(meta["score_seed"])
    noise = jax.random.normal(jax.random.fold_in(key, FS.KEY_REFRESH_NOISE), (n, action_dim)) * args.policy_noise
    noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    native, parts = state.native, {}
    for start in range(0, n, batch):
        chunk = jax.tree_util.tree_map(lambda x: x[start:start + batch], data)
        action = models[0].apply(native.actor_target.params, chunk.next_obs)
        action = jnp.clip(action + noise[start:start + batch], -prepared.max_action, prepared.max_action)
        q_next = models[1].apply(native.critic_target.params, chunk.next_obs, action).min(axis=-1)
        target = jax.lax.stop_gradient(chunk.reward + (1.0 - chunk.done) * args.discount * q_next)
        q = models[1].apply(native.critic.params, chunk.obs, chunk.action)
        pi = models[0].apply(native.actor.params, chunk.obs)
        values = dict(target=target, q=q, residual_min=target - q.min(axis=-1), policy_action=pi,
                      q_pi=models[1].apply(native.critic.params, chunk.obs, pi),
                      sigma=FS.positive_scale(models[2].apply(state.calibrator.params, chunk.obs, chunk.action),
                                              state.residual_scale),
                      sigma_pi=FS.positive_scale(models[2].apply(state.calibrator.params, chunk.obs, pi),
                                                 state.residual_scale))
        for name, value in values.items():
            parts.setdefault(name, []).append(np.asarray(value))
    out = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    if out["q"].shape != (n, 2):
        raise ValueError("expected TD3+BC's twin critic")
    gate = dict(
        residual=float(np.max(np.abs(out["residual_min"].astype(np.float64) - frozen["residual"]))),
        sigma=float(np.max(np.abs(out["sigma"].astype(np.float64) - frozen["sigma"]))),
        policy_action=float(np.max(np.abs(out["policy_action"] - frozen["policy_action"]))),
        checkpoint_step=step, verification=verification)
    heads = {name: out[name] for name in HEADS if name != "episode"}
    heads["episode"] = frozen["episode"]
    return heads, gate, meta


def upper_quantile(sorted_scores, alpha):
    """Smallest population value with at most alpha of the population strictly above it."""
    return float(sorted_scores[math.ceil((1 - alpha) * sorted_scores.size) - 1])


def miscoverage(sorted_scores, thresholds):
    """Population share strictly above each threshold."""
    return 1.0 - np.searchsorted(sorted_scores, thresholds, side="right") / sorted_scores.size


def analyze(heads, rng, *, banks=1000, n=1024, alpha=0.1, beta=0.95, draws=1000):
    # Residuals are float32 differences cast afterwards, as the frozen scores are, so s_min is exactly the
    # gated score.
    r_min = (heads["target"] - heads["q"].min(axis=1)).astype(np.float64)
    r_1 = (heads["target"] - heads["q"][:, 0]).astype(np.float64)
    t = heads["target"].astype(np.float64)
    q = heads["q"].astype(np.float64)
    q_pi = heads["q_pi"].astype(np.float64)
    sigma, sigma_pi = heads["sigma"].astype(np.float64), heads["sigma_pi"].astype(np.float64)
    scores = {"min": np.abs(r_min) / sigma, "q1": np.abs(r_1) / sigma}
    scores["max"] = np.maximum(scores["min"], scores["q1"])
    ordered = {name: np.sort(s) for name, s in scores.items()}
    d, d_pi = q[:, 0] - q[:, 1], q_pi[:, 0] - q_pi[:, 1]

    lam_min = upper_quantile(ordered["min"], alpha)
    covered_min, missed_q1 = scores["min"] <= lam_min, scores["q1"] > lam_min
    raw = {"min": np.abs(r_min), "q1": np.abs(r_1), "max": np.maximum(np.abs(r_min), np.abs(r_1))}
    out = dict(
        rows=int(t.size),
        logged=dict(q1_above_q2=float(np.mean(d > 0)), mean_gap_over_sigma=float(np.mean(d / sigma)),
                    median_abs_gap=float(np.median(np.abs(d))), median_abs_residual_min=float(np.median(np.abs(r_min))),
                    gap_over_residual=float(np.median(np.abs(d)) / np.median(np.abs(r_min))),
                    residual_min_negative=float(np.mean(r_min < 0))),
        policy=dict(q1_above_q2=float(np.mean(d_pi > 0)), mean_gap_over_sigma=float(np.mean(d_pi / sigma_pi)),
                    median_abs_gap=float(np.median(np.abs(d_pi))),
                    mean_q1_pi_minus_q1_logged_over_sigma=float(np.mean((q_pi[:, 0] - q[:, 0]) / sigma))),
        population=dict(lambda_min=lam_min, q1_miscoverage_at_lambda_min=float(np.mean(missed_q1)),
                        hidden_misses=float(np.mean(covered_min & missed_q1)),
                        reverse=float(np.mean(~covered_min & ~missed_q1)),
                        quantile_scaled={name: upper_quantile(s, alpha) for name, s in ordered.items()},
                        quantile_raw={name: upper_quantile(np.sort(r), alpha) for name, r in raw.items()}),
    )
    thresholds = {name: np.empty(banks) for name in scores}
    for b in range(banks):
        rows = rng.integers(t.size, size=n)
        for name, s in scores.items():
            thresholds[name][b] = calibrate(s[rows], rng, alpha=alpha, beta=beta, draws=draws).threshold
    out["banks"] = dict(banks=banks, n=n, alpha=alpha, beta=beta, draws=draws, by_score={})
    for name, th in thresholds.items():
        finite = th[np.isfinite(th)]
        out["banks"]["by_score"][name] = dict(
            abstain=float(1 - finite.size / banks), mean_threshold=float(np.mean(finite)),
            **{f"fail_{resid}": float(np.mean(miscoverage(ordered[resid], finite) > alpha)) for resid in ("min", "q1")},
            **{f"mean_miscoverage_{resid}": float(np.mean(miscoverage(ordered[resid], finite))) for resid in ("min", "q1")})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", help="frozen pool directories, or heads.npz files with --analyze-only")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--banks", type=int, default=1000)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026100102)
    args = parser.parse_args()
    for item in args.inputs:
        started = time.perf_counter()
        if args.analyze_only:
            path = Path(item)
            heads, out_dir = dict(np.load(path)), path.parent
            gate = json.loads((out_dir / "summary.json").read_text())["gate"]
        else:
            heads, gate, meta = extract(item)
            out_dir = OUTPUT / Path(item).name
            out_dir.mkdir(parents=True, exist_ok=False)
            np.savez(out_dir / "heads.npz", **heads)
            (out_dir / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        if max(gate["residual"], gate["sigma"], gate["policy_action"]) > 0:
            raise SystemExit(f"reproduction gate not exact for {item}: {gate}; nothing is analyzed")
        stats = analyze(heads, np.random.default_rng(args.seed), banks=args.banks, n=args.n)
        summary = dict(run=Path(item).name if not args.analyze_only else out_dir.name, gate=gate, **stats,
                       seconds=time.perf_counter() - started)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        b = stats["banks"]["by_score"]
        print(f"{summary['run']}: gate {gate['residual']:.3g}/{gate['sigma']:.3g}/{gate['policy_action']:.3g}  "
              f"Q1>Q2 logged {stats['logged']['q1_above_q2']:.1%} policy {stats['policy']['q1_above_q2']:.1%}  "
              f"Q1 miscov at lambda_min {stats['population']['q1_miscoverage_at_lambda_min']:.1%} "
              f"hidden {stats['population']['hidden_misses']:.2%}  banks fail_q1 min/q1/max "
              f"{b['min']['fail_q1']:.1%}/{b['q1']['fail_q1']:.1%}/{b['max']['fail_q1']:.1%}  "
              f"({summary['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
