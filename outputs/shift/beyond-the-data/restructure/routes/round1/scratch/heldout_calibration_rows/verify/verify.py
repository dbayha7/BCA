"""Adversarial re-check of route heldout_calibration_rows. Read-only on every repository path; writes only
verify/out.json beside this file. Gate seeds 0-4 only, forward passes only.

Recomputes, independently of evaluate.overestimates: the full-throttle rows, their one-step overestimates (own code
path from critic_rows), the rest scores, the certificate (S.certificate, same tags), the shuffle mean; then probes
attribution: margin sweep, simple non-Bayesian rules on the same scores, leave-one-row-out, row bootstrap, episodes."""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

ROOT = Path("/home/user/bca")
STEP1 = ROOT / "outputs/shift/beyond-the-data/restructure/step1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STEP1))
import step1lib as S  # noqa: E402
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = ROOT / "runs/shift_explore/routes/heldout_calibration_rows/results.json"
SEEDS = [0, 1, 2, 3, 4]


def own_scores(critic, lg, rows, mean, std, fixed, gamma):
    obs = (np.asarray(lg["observations"][rows], np.float32) - mean) / std
    nxt = (np.asarray(lg["next_observations"][rows], np.float32) - mean) / std
    act = np.asarray(lg["actions"][rows], np.float32)
    rew = np.asarray(lg["rewards"][rows], np.float64)
    done = np.asarray(lg["terminals"][rows], np.float64)
    q = L.critic_rows(critic, obs, act)[0].min(1)
    c = np.asarray(fixed.apply(fixed.params(), jnp.asarray(nxt, jnp.float32)), np.float32)
    qn = L.critic_rows(critic, nxt, c)[0].min(1)
    return q - (rew + gamma * (1.0 - done) * qn), q, rew + gamma * (1.0 - done) * qn


def main():
    t0 = time.process_time()
    r = S.read_json_checked(RES)
    spec, _ = S.load_prespec()
    S.verify_manifest()
    cfg = spec["certificate"]
    gcfg = spec["groups"]["gap_rule"]
    p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(p)
    variant = A.gate_outcome(p)["variant"]
    learn = p.config.learn
    gamma = float(spec["scores"]["gamma"])
    seen = spec["deployment"]["seen_design_values"]
    out = {}
    tails, rests = {}, {}
    for seed in SEEDS:
        for world in S.WORLDS:
            key = f"{world}/seed{seed}"
            w = S.WORLD_TAG[world]
            sd = A.load_seed(p.runs, p.config, variant, world, seed)
            lg = sd.log
            pg = sd.prepared("gap", learn)
            gr = S.gap_rule(S.throttle(pg.actions), gcfg["upper_quantile"], gcfg["min_gap"], gcfg["min_tail"])
            t_lo = gr["t_lo"]
            split = np.asarray(lg["infos/split"])
            cal = np.flatnonzero(split == 1)
            cal = cal[~sd.gap.contains(lg["actions"][cal])]
            tt = np.abs(np.asarray(lg["actions"][cal], np.float32)).max(1).astype(np.float64)
            ft = cal[tt > t_lo]
            stored = r["per_seed_world"][key]
            rows_equal = bool(np.array_equal(ft, np.asarray(stored["rows"]["log_row"])))
            # episodes: calibration episodes disjoint from training episodes?
            ep = np.asarray(lg["infos/episode"])
            ep_train = np.unique(ep[pg.rows])
            ep_ft = np.unique(ep[ft])
            bank = np.asarray(pg.bank_rows, np.int64)
            bank_t = S.throttle(np.asarray(lg["actions"][bank]))
            rest_rows = bank[bank_t <= t_lo]
            rec, critic = A.get_critic(p, sd, "gap", "stall_aware", train=False)
            fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
            s_ft, q_ft, tgt_ft = own_scores(critic, lg, ft, pg.obs_mean, pg.obs_std, fixed, gamma)
            s_rest, q_rest, tgt_rest = own_scores(critic, lg, rest_rows, pg.obs_mean, pg.obs_std, fixed, gamma)
            score_maxdiff = float(np.max(np.abs(s_ft - np.asarray(stored["rows"]["score"]))))
            cert = S.certificate(s_ft, s_rest, [20261015, 9900, seed, w], cfg)
            tails[key], rests[key] = s_ft, s_rest
            # leave-one-row-out of the full-throttle group
            loo = [S.certificate(np.delete(s_ft, i), s_rest, [20261015, 9900, seed, w], cfg)["q05"]
                   for i in range(len(s_ft))]
            # row bootstrap of the full-throttle group (rest fixed), 100 resamples
            g = np.random.default_rng([20261015, 7777, seed, w])
            boot_on = []
            for b in range(100):
                idx = g.integers(0, len(s_ft), len(s_ft))
                boot_on.append(bool(S.certificate(s_ft[idx], s_rest, [20261015, 9900, seed, w], cfg)["on"]))
            # simple rules on the same scores
            q90 = lambda x: float(np.quantile(x, 0.9, method="inverted_cdf"))  # noqa: E731
            out[key] = dict(
                t_lo=t_lo, n_ft=int(len(ft)), rows_equal_stored=rows_equal, score_max_abs_diff=score_maxdiff,
                n_rest=int(len(rest_rows)), n_rest_stored=stored["n_rest"],
                q05=cert["q05"], q05_stored=stored["certificate"]["q05"], q50=cert["q50"], on=cert["on"],
                height=cert["height"], finite_share_tail=cert["finite_share_tail"],
                episodes_ft_in_train=int(np.intersect1d(ep_ft, ep_train).size),
                rows_ft_in_train=int(np.intersect1d(ft, pg.rows).size),
                all_ft_calibration=bool((split[ft] == 1).all()),
                mean_diff=float(s_ft.mean() - s_rest.mean()), median_diff=float(np.median(s_ft) - np.median(s_rest)),
                q90_diff=q90(s_ft) - q90(s_rest),
                mean_pred_ft=float(q_ft.mean()), mean_pred_rest=float(q_rest.mean()),
                mean_target_ft=float(tgt_ft.mean()), mean_reward_ft=float(np.asarray(lg["rewards"][ft]).mean()),
                loo_q05_min=float(np.min(loo)), loo_q05_max=float(np.max(loo)),
                boot_on_rate=float(np.mean(boot_on)),
                critic_sha=rec["result"]["file_sha256"])
            print(time.strftime("%H:%M:%S"), key, {k: (round(v, 4) if isinstance(v, float) else v)
                                                     for k, v in out[key].items() if k != "critic_sha"}, flush=True)
            del sd, lg, pg
    pts = {wd: {s: dict(never=seen[f"{wd}/seed{s}/strong_current"]["never"],
                        on=seen[f"{wd}/seed{s}/strong_current"]["on"]) for s in SEEDS} for wd in S.WORLDS}

    def net_of(rule):
        certs = {wd: {s: bool(rule(f"{wd}/seed{s}")) for s in SEEDS} for wd in S.WORLDS}
        return S.deployment_net(pts, certs, SEEDS)["net"], certs

    summary = {}
    for m in (0.0, 0.025, 0.05, 0.10, 0.20, 0.30, 0.36, 0.40):
        summary[f"certificate_margin_{m}"] = net_of(lambda k, m=m: out[k]["q05"] > m)[0]
    for m in (0.0, 0.05, 0.10, 0.20):
        summary[f"mean_diff_gt_{m}"] = net_of(lambda k, m=m: out[k]["mean_diff"] > m)[0]
        summary[f"q90_diff_gt_{m}"] = net_of(lambda k, m=m: out[k]["q90_diff"] > m)[0]
        summary[f"median_diff_gt_{m}"] = net_of(lambda k, m=m: out[k]["median_diff"] > m)[0]
    # the shift-by-margin rank test: Mann-Whitney of (tail - 0.10) vs rest at p < 0.05
    summary["rank_test_shifted_by_0.10_p<0.05"] = net_of(
        lambda k: S.rank_test(tails[k] - 0.10, rests[k])["p"] < 0.05)[0]
    summary["rank_test_p<0.05"] = net_of(lambda k: S.rank_test(tails[k], rests[k])["p"] < 0.05)[0]
    summary["real_net"] = net_of(lambda k: out[k]["on"])[0]
    # shuffle recompute (200, same tags) for the main certificate
    sh_on = {wd: {} for wd in S.WORLDS}
    for seed in SEEDS:
        for world in S.WORLDS:
            key, w = f"{world}/seed{seed}", S.WORLD_TAG[world]
            sh = S.shuffled_certificates(tails[key], rests[key], [20261015, 9900, seed, w],
                                         [20261015, 9901, seed, w], 200, cfg)
            sh_on[world][seed] = sh["on"]
    summary["shuffle"] = S.shuffle_summary(pts, sh_on, SEEDS, summary["real_net"])
    summary["cpu_seconds"] = time.process_time() - t0
    (HERE / "out.json").write_text(json.dumps(S.jsonable(dict(per=out, summary=summary)), indent=1))
    print(json.dumps(S.jsonable(summary), indent=1))


if __name__ == "__main__":
    main()
