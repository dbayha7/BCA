"""Route 1 (heldout_calibration_rows): BCA's world-level certificate fed by its own held-out calibration rows.

EXPLORATORY, designed after seeing seeds 0-4. Gate seeds 0-4 only. Forward passes only: no training, no rollouts.
Kill rule: KILL_RULE.md beside this file (written and hashed before any score of these rows was computed).

Evidence rows: every calibration-split row whose command lies outside the removed ball and past t_lo (the bank's
full-throttle rows included). No critic ever trained on the calibration split. Rest: the bank rows at or below t_lo
(step 1's rest group). Critic: the gate's own gapped fixed-controller critic. Certificate, shuffle, net and rank test:
step 1's frozen functions (step1lib, prespec.json), imported unchanged; the scoring function is step 1's
evaluate.overestimates, imported unchanged.

Writes only runs/shift_explore/routes/heldout_calibration_rows/results.json (+ .sha256), write-once.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path("/home/user/bca")
STEP1 = ROOT / "outputs/shift/beyond-the-data/restructure/step1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STEP1))
import step1lib as S  # noqa: E402
import evaluate as E  # noqa: E402  (step 1's evaluation module; only its helpers are used, main() is not run)
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402

SCR = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes")
HERE = SCR / "heldout_calibration_rows"
OUT = ROOT / "runs/shift_explore/routes/heldout_calibration_rows"
STEP1_RESULTS = ROOT / "runs/shift_explore/restructure/step1/results.json"
SEEDS = [0, 1, 2, 3, 4]
N_PERM = 200
LABEL = ("EXPLORATORY route test (heldout_calibration_rows), designed after seeing seeds 0-4; gate seeds 0-4 only; "
         "forward passes only. Outside the gate; changes no pre-registered rule and no v6/v7 file.")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def guard_out(path):
    p = Path(path).resolve()
    for pin in S.PINNED:
        if S.inside(p, pin):
            raise SystemExit(f"refused: {p} is inside pinned {pin}")
    if not S.inside(p, OUT):
        raise SystemExit(f"refused: outputs only under {OUT}")
    return p


def main():
    t_start, c_start = time.perf_counter(), time.process_time()
    kr = (HERE / "KILL_RULE.md").read_bytes()
    kr_sha = S.sha256_bytes(kr)
    want = (HERE / "KILL_RULE.md.sha256").read_text().split()[0]
    if kr_sha != want:
        raise SystemExit("refused: KILL_RULE.md changed after it was hashed")
    out_json = guard_out(OUT / "results.json")
    if out_json.exists():
        raise SystemExit(f"refused: {out_json} exists (write-once)")
    spec, spec_sha = S.load_prespec()
    man, man_sha = S.verify_manifest()
    cfg = spec["certificate"]
    gcfg = spec["groups"]["gap_rule"]
    p, variant = E.gate_context(spec)
    learn = p.config.learn
    gamma = float(spec["scores"]["gamma"])
    if abs(learn.gamma - gamma) > 0:
        raise SystemExit("refused: discount differs")
    feas = {}
    for f in ("count_rows_0.json", "count_rows_1_2_3_4.json"):
        feas.update(json.loads((SCR / "_feasibility" / f).read_text()))
    step1 = S.read_json_checked(STEP1_RESULTS)
    seen = spec["deployment"]["seen_design_values"]

    per, certs, shuffled_on, checks = {}, {w: {} for w in S.WORLDS}, {w: {} for w in S.WORLDS}, {}
    var_t4, var_restall, rank_on = ({w: {} for w in S.WORLDS} for _ in range(3))
    for seed in SEEDS:
        for world in S.WORLDS:
            key = f"{world}/seed{seed}"
            t0 = time.perf_counter()
            w = S.WORLD_TAG[world]
            sd = A.load_seed(p.runs, p.config, variant, world, seed)
            lg = sd.log
            pg = sd.prepared("gap", learn)
            gr = S.gap_rule(S.throttle(pg.actions), gcfg["upper_quantile"], gcfg["min_gap"], gcfg["min_tail"])
            exp = spec["groups"]["expected_from_design"][key]
            ok = bool(gr["defined"] and gr["n_tail"] == exp["n_survivors"] and abs(gr["t_lo"] - exp["t_lo"]) <= 1e-12)
            checks[f"{key}/gap_rule_as_step1"] = ok
            if not ok:
                raise SystemExit(f"PLUMBING STOP: gap rule on {key} gives t_lo {gr['t_lo']} n {gr['n_tail']}, "
                                 f"step 1 {exp}")
            t_lo = gr["t_lo"]
            cal = lg.split_rows("calibration")
            cal = cal[~sd.gap.contains(lg["actions"][cal])]
            t_cal = S.throttle(np.asarray(lg["actions"][cal]))
            ft_rows = cal[t_cal > t_lo]
            rest_all_rows = cal[t_cal <= t_lo]
            bank = np.asarray(pg.bank_rows, np.int64)
            bank_t = S.throttle(np.asarray(pg.bank.action))
            bank_ft = S.full_throttle(bank_t, t_lo)
            fz = feas[f"{seed}_{world}"]
            c = dict(
                bank_in_cal_outside_ball=bool(np.isin(bank, cal).all()),
                bank_throttle_consistent=bool(np.array_equal(bank_t, S.throttle(np.asarray(lg["actions"][bank])))),
                bank_tail_subset=bool(np.isin(bank[bank_ft], ft_rows).all()),
                no_training_row=bool(np.intersect1d(ft_rows, pg.rows).size == 0
                                     and np.intersect1d(rest_all_rows, pg.rows).size == 0),
                size_as_feasibility=bool(len(ft_rows) == fz["n_heldcal_tail"] + fz["n_bank_tail"]),
                t_lo_as_feasibility=bool(abs(fz["t_lo"] - t_lo) <= 1e-12),
                all_calibration_split=bool((np.asarray(lg["infos/split"])[ft_rows] == 1).all()))
            for k, v in c.items():
                checks[f"{key}/{k}"] = v
            if not all(c.values()):
                raise SystemExit(f"PLUMBING STOP: {key}: {c}")
            rec, critic = A.get_critic(p, sd, "gap", "stall_aware", train=False)
            fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
            if critic is None:
                certs[world][seed] = False
                shuffled_on[world][seed] = [False] * N_PERM
                per[key] = dict(certificate=dict(on=False, reason="gate critic failed"))
                log(f"{key}: GATE CRITIC FAILED -> off")
                continue
            tr_ft = L._transition(lg, ft_rows, pg.obs_mean, pg.obs_std)
            s_ft = E.overestimates(critic, tr_ft, fixed, gamma)
            s_bank = E.overestimates(critic, pg.bank, fixed, gamma)
            # consistency: bank full-throttle rows scored both ways must agree exactly
            pos = np.searchsorted(ft_rows, bank[bank_ft])
            checks[f"{key}/bank_tail_scores_agree"] = bool(np.array_equal(s_ft[pos], s_bank[bank_ft]))
            if not checks[f"{key}/bank_tail_scores_agree"]:
                raise SystemExit(f"PLUMBING STOP: {key}: bank-tail scores differ between the two transition builds")
            rest = s_bank[~bank_ft]
            cert = S.certificate(s_ft, rest, [20261015, 9900, seed, w], cfg)
            certs[world][seed] = bool(cert["on"])
            rank = S.rank_test(s_ft, rest)
            rank_on[world][seed] = bool(rank["p"] is not None and rank["p"] < 0.05)
            sh = S.shuffled_certificates(s_ft, rest, [20261015, 9900, seed, w], [20261015, 9901, seed, w], N_PERM, cfg)
            shuffled_on[world][seed] = sh["on"]
            # reported only: T4 rows left out
            pz = np.load(ROOT / f"runs/shift_explore/detect/seed{seed}/{world}/pairs.npz")
            t4 = np.asarray(pz["log_row"][pz["set"] == "T4"], np.int64)
            is_t4 = np.isin(ft_rows, t4)
            keep = ~is_t4
            c_t4 = S.certificate(s_ft[keep], rest, [20261015, 9902, seed, w], cfg)
            var_t4[world][seed] = bool(c_t4["on"])
            # reported only: rest = every held-out calibration row at or below t_lo
            tr_ra = L._transition(lg, rest_all_rows, pg.obs_mean, pg.obs_std)
            s_ra = E.overestimates(critic, tr_ra, fixed, gamma)
            c_ra = S.certificate(s_ft, s_ra, [20261015, 9903, seed, w], cfg)
            var_restall[world][seed] = bool(c_ra["on"])
            # reported only: in-sample survivors under the same critic (the fade reference)
            surv_idx = np.flatnonzero(gr["tail"])
            s_surv_in = E.overestimates(critic, pg.train, fixed, gamma, rows=surv_idx)
            st1 = step1["per_seed_world"][key]
            sh_q = np.asarray([v for v in sh["q05"] if v is not None], np.float64)
            per[key] = dict(
                t_lo=t_lo, n_full_throttle=int(len(s_ft)), n_full_throttle_from_bank=int(bank_ft.sum()),
                n_full_throttle_t4=int(is_t4.sum()), n_rest=int(len(rest)), n_rest_all_held_out=int(len(s_ra)),
                critic=dict(file_sha256=rec["result"]["file_sha256"]),
                certificate=cert, rank_test=rank,
                shuffle=dict(on_rate=float(np.mean(sh["on"])), q05_median=float(np.median(sh_q)),
                             q05_q95=float(np.quantile(sh_q, 0.95))),
                variant_without_t4=dict(n=int(keep.sum()), q05=c_t4["q05"], on=c_t4["on"], height=c_t4["height"],
                                        rank_p=S.rank_test(s_ft[keep], rest)["p"]),
                variant_rest_all_held_out=dict(q05=c_ra["q05"], on=c_ra["on"], height=c_ra["height"],
                                               mean_rest=float(s_ra.mean()), rank_p=S.rank_test(s_ft, s_ra)["p"]),
                groups=dict(mean_full_throttle=float(s_ft.mean()), median_full_throttle=float(np.median(s_ft)),
                            mean_rest=float(rest.mean()),
                            q90_full_throttle=float(np.quantile(s_ft, 0.9, method="inverted_cdf")),
                            q90_rest=float(np.quantile(rest, 0.9, method="inverted_cdf")),
                            share_full_throttle_above_rest_q90=float(np.mean(
                                s_ft > np.quantile(rest, 0.9, method="inverted_cdf")))),
                in_sample_survivors_same_critic=dict(n=int(len(surv_idx)), mean=float(s_surv_in.mean())),
                step1_beside=dict(q05=st1["certificate"]["q05"], on=st1["certificate"]["on"],
                                  height=st1["certificate"]["height"], n_full=st1["groups"]["n_full_throttle"],
                                  mean_full=st1["groups"]["mean_full_throttle"]),
                rows=dict(log_row=ft_rows.tolist(), throttle=S.throttle(np.asarray(lg["actions"][ft_rows])).tolist(),
                          score=s_ft.tolist(), in_bank=np.isin(ft_rows, bank).tolist(), t4=is_t4.tolist(),
                          episode=np.asarray(lg["infos/episode"][ft_rows]).tolist()))
            log(f"{key}: n_full {len(s_ft)} (bank {int(bank_ft.sum())}, T4 {int(is_t4.sum())}) n_rest {len(rest)}; "
                f"mean full {s_ft.mean():+.3f} rest {rest.mean():+.3f}; q05 {cert['q05']:+.3f} -> "
                f"{'ON' if cert['on'] else 'off'}; height {cert['height']:.3f}; MW p {rank['p']:.2g}; "
                f"shuffled on-rate {np.mean(sh['on']):.3f}; noT4 q05 {c_t4['q05']:+.3f}; restall q05 "
                f"{c_ra['q05']:+.3f}; step1 q05 {st1['certificate']['q05']:+.3f}; {time.perf_counter() - t0:.0f}s")
            del sd, lg, pg

    pts = {w: {s: dict(never=seen[f"{w}/seed{s}/strong_current"]["never"],
                       on=seen[f"{w}/seed{s}/strong_current"]["on"]) for s in SEEDS} for w in S.WORLDS}
    net = S.deployment_net(pts, certs, SEEDS)
    sh_sum = S.shuffle_summary(pts, shuffled_on, SEEDS, net["net"])
    k1 = not (all(certs["stall"][s] for s in SEEDS) and not any(certs["no_stall"][s] for s in SEEDS))
    k2 = net["net"] < 90.0
    k3 = sh_sum["mean"] > 0.5 * net["net"]
    verdict = "SHELVED" if (k1 or k2 or k3) else "ALIVE (no kill condition met)"
    info = dict(
        forced_on_net=S.deployment_net(pts, {w: {s: True for s in SEEDS} for w in S.WORLDS}, SEEDS),
        rank_test_switched=dict(on=rank_on, net=S.deployment_net(pts, rank_on, SEEDS)),
        variant_without_t4=dict(on=var_t4, net=S.deployment_net(pts, var_t4, SEEDS)),
        variant_rest_all_held_out=dict(on=var_restall, net=S.deployment_net(pts, var_restall, SEEDS)),
        step1_net=step1["net"])
    ru = resource.getrusage(resource.RUSAGE_SELF)
    results = dict(
        label=LABEL, kill_rule_sha256=kr_sha, prespec_sha256=spec_sha, step1_code_manifest_sha256=man_sha,
        gate_prereg_sha256=p.digest, certificates={w: {str(s): certs[w][s] for s in SEEDS} for w in S.WORLDS},
        net=net, shuffle=sh_sum,
        kill=dict(rule1_pattern_wrong=k1, rule2_net_below_90=k2, rule3_shuffle_mean_above_half=k3, verdict=verdict),
        information_only=info, per_seed_world=per, checks=checks,
        cpu_seconds=dict(process=time.process_time() - c_start, user=ru.ru_utime, system=ru.ru_stime),
        wall_seconds=time.perf_counter() - t_start, machine=platform.node(),
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    digest = _write(out_json, results)
    log(f"certificates {results['certificates']}")
    log(f"NET {net['net']:.1f} (gain {net['gain_stall']:.1f}, loss {net['loss_no_stall']:.1f}); shuffled mean "
        f"{sh_sum['mean']:.1f}; kill {results['kill']}; results {out_json} sha {digest}")
    log(f"info: forced-on {info['forced_on_net']['net']:.1f}; rank-switched {info['rank_test_switched']['net']['net']:.1f}; "
        f"no-T4 {info['variant_without_t4']['net']['net']:.1f} {var_t4}; rest-all "
        f"{info['variant_rest_all_held_out']['net']['net']:.1f} {var_restall}; cpu {results['cpu_seconds']}")


def _write(path, payload):
    p = guard_out(path)
    if p.exists():
        raise SystemExit(f"refused: {p} exists (write-once)")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(S.jsonable(payload), indent=1, sort_keys=True, allow_nan=False)
    p.write_text(text, encoding="utf8")
    d = S.sha256_bytes(text.encode("utf8"))
    p.with_name(p.name + ".sha256").write_text(f"{d}  {p.name}\n", encoding="utf8")
    return d


if __name__ == "__main__":
    main()
