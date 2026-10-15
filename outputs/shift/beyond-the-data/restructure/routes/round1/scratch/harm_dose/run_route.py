"""Route 3 (harm_dose): does BCA's outcome-reading certificate switch caution on where clipping starts to pay, as the
stall strength sigma varies?

EXPLORATORY, designed after seeing seeds 0-4. Gate seeds 0-4 only. No training: one-step dynamics recomputations,
critic forward passes and rollouts of saved actors. Kill rule: KILL_RULE.md beside this file (hashed in
KILL_RULE.md.sha256 before any result of this route was computed; this script refuses if it changed).

World sigma = thruster_chain.Physics(stall=True, stall_slope=sigma): push = clip(a, +-0.8) - sigma sign(a)
max(|a| - 0.8, 0). sigma = -1 is the no-stall world (push = a), 1.5 the stalling world, 3 the pre-registered retune.

Evidence (primary): the no-stall world's gate log; full-throttle group = calibration-split rows outside the removed ball
past t_lo; rest = bank rows at or below t_lo; each row's one-step outcome recomputed exactly under sigma from its logged
state; scored by the no-stall gate critic (never trained on any of these rows); step 1's frozen certificate.
Value: points of 'clip past t_lo' minus 'never' for the no-stall 'none' actor at 200 strong-current starts in world
sigma, on the stalling world's points scale.

Writes only runs/shift_explore/routes/harm_dose/results.json (+ .sha256), write-once.
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
import evaluate as E  # noqa: E402  (step 1's evaluation module; only helpers are used, main() is not run)
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402

HERE = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/harm_dose")
OUT = ROOT / "runs/shift_explore/routes/harm_dose"
SEEDS = [0, 1, 2, 3, 4]
SIGMAS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 3.0]
KNOWN = (-1.0, 1.5)
INTERIOR = [s for s in SIGMAS if s not in KNOWN]
TOL = 1e-9
N_STARTS = 200
LABEL = ("EXPLORATORY route test (harm_dose), designed after seeing seeds 0-4; gate seeds 0-4 only; no training. "
         "Outside the gate; changes no pre-registered rule and no v6/v7 file.")


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


def phys_of(sigma):
    return TC.Physics(stall=True, stall_slope=float(sigma))


def recompute(lg, rows, phys):
    """Exact one-step outcome of log rows under phys, from their logged float64 state and float32 command."""
    x = np.asarray(lg["infos/state"][rows], np.float64)
    a = np.asarray(lg["actions"][rows], np.float32).astype(np.float64)
    return TC.dynamics(x, a, phys)


def transition(lg, rows, next_state, pg):
    """learners._transition on the logged rows with next_observations replaced by float32(next_state)."""
    fake = {"observations": np.asarray(lg["observations"][rows], np.float32),
            "next_observations": np.asarray(next_state, np.float64).astype(np.float32),
            "actions": np.asarray(lg["actions"][rows], np.float32),
            "rewards": np.asarray(lg["rewards"][rows], np.float32),
            "terminals": np.asarray(lg["terminals"][rows], np.float32)}
    return L._transition(fake, np.arange(len(rows)), pg.obs_mean, pg.obs_std)


def groups(sd, pg, world, seed, spec):
    """t_lo (checked against step 1's prespec), the primary full-throttle rows, the rest rows (bank <= t_lo) and the
    removed training rows past t_lo (rich evidence)."""
    lg = sd.log
    gcfg = spec["groups"]["gap_rule"]
    gr = S.gap_rule(S.throttle(pg.actions), gcfg["upper_quantile"], gcfg["min_gap"], gcfg["min_tail"])
    exp = spec["groups"]["expected_from_design"][f"{world}/seed{seed}"]
    if not (gr["defined"] and gr["n_tail"] == exp["n_survivors"] and abs(gr["t_lo"] - exp["t_lo"]) <= 1e-12):
        raise SystemExit(f"PLUMBING STOP: gap rule {world}/seed{seed}: {gr['t_lo']} {gr['n_tail']} vs {exp}")
    t_lo = gr["t_lo"]
    cal = lg.split_rows("calibration")
    cal = cal[~sd.gap.contains(lg["actions"][cal])]
    ft_rows = cal[S.throttle(np.asarray(lg["actions"][cal])) > t_lo]
    bank = np.asarray(pg.bank_rows, np.int64)
    bank_t = S.throttle(np.asarray(pg.bank.action))
    if not np.array_equal(bank_t, S.throttle(np.asarray(lg["actions"][bank]))):
        raise SystemExit("PLUMBING STOP: bank commands differ")
    rest_rows = bank[bank_t <= t_lo]
    rem = sd.removals["gap"].removed_train_rows(lg)
    rem_ft = rem[S.throttle(np.asarray(lg["actions"][rem])) > t_lo]
    c = dict(bank_tail_in_group=bool(np.isin(bank[bank_t > t_lo], ft_rows).all()),
             no_training_row=bool(np.intersect1d(ft_rows, pg.rows).size == 0
                                  and np.intersect1d(rest_rows, pg.rows).size == 0
                                  and np.intersect1d(rem_ft, pg.rows).size == 0),
             calibration_split=bool((np.asarray(lg["infos/split"])[ft_rows] == 1).all()
                                    and (np.asarray(lg["infos/split"])[rest_rows] == 1).all()))
    if not all(c.values()):
        raise SystemExit(f"PLUMBING STOP: groups {world}/seed{seed}: {c}")
    return dict(t_lo=t_lo, ft=ft_rows, rest=rest_rows, removed_ft=rem_ft, checks=c,
                n_ft_at_or_below_knee=int(np.sum(S.throttle(np.asarray(lg["actions"][ft_rows])) <= TC.KNEE)))


def check_outcomes(lg, rows, phys, what):
    nxt, r = recompute(lg, rows, phys)
    dx = float(np.max(np.abs(nxt - np.asarray(lg["infos/next_state"][rows], np.float64)))) if len(rows) else 0.0
    rew_ok = bool(np.array_equal(np.asarray(r, np.float64).astype(np.float32), np.asarray(lg["rewards"][rows],
                                                                                          np.float32)))
    if not (dx <= TOL and rew_ok):
        raise SystemExit(f"PLUMBING STOP: {what}: max |x' - logged| {dx}, rewards equal {rew_ok}")
    return dict(max_abs_next_state_diff=dx, rewards_equal=rew_ok)


def scores(critic, fixed, gamma, lg, rows, phys, pg):
    nxt, _ = recompute(lg, rows, phys)
    return E.overestimates(critic, transition(lg, rows, nxt, pg), fixed, gamma)


def rollout_J(policy, phys, x0, lo, hi, t_lo, mode):
    """step 1's evaluate.rollout ('never' / 'on'), restated to return the per-start returns J and the fire share."""
    x = np.asarray(x0, np.float64).copy()
    n = len(x)
    J = np.zeros(n)
    fired = np.zeros((n, phys.length), bool)
    for k in range(phys.length):
        a_act = np.clip(policy(x), -1.0, 1.0)
        fire = np.abs(a_act).max(1) > t_lo if mode == "on" else np.zeros(n, bool)
        a = np.where(fire[:, None], np.clip(a_act, lo, hi), a_act)
        fired[:, k] = fire
        x, r = TC.dynamics(x, a, phys)
        J = J + r
    return J, float(fired.mean())


def pts(J, anchor):
    return float(TR.points(J, anchor).mean())


def actor_setup(world, seed, pg):
    actor, info = E.none_actor(world, seed)
    pol = L.mlp_policy(actor, pg, "none")
    a = np.asarray(pg.actions, np.float64)
    return pol, np.quantile(a, 0.005, axis=0), np.quantile(a, 0.995, axis=0), info


def cert_summary(c):
    keep = ("on", "q05", "q50", "plain_gap", "height", "R_tail", "R_rest", "lambda_hat_tail", "lambda_hat_rest",
            "mean_tail", "mean_rest", "n_tail", "n_rest", "tail_certified", "finite_share_tail")
    return {k: c.get(k) for k in keep}


def main():
    t_start, c_start = time.perf_counter(), time.process_time()
    kr = (HERE / "KILL_RULE.md").read_bytes()
    kr_sha = S.sha256_bytes(kr)
    if kr_sha != (HERE / "KILL_RULE.md.sha256").read_text().split()[0]:
        raise SystemExit("refused: KILL_RULE.md changed after it was hashed")
    out_json = guard_out(OUT / "results.json")
    if out_json.exists():
        raise SystemExit(f"refused: {out_json} exists (write-once)")
    spec, spec_sha = S.load_prespec()
    man, man_sha = S.verify_manifest()
    cfg = spec["certificate"]
    p, variant = E.gate_context(spec)
    learn = p.config.learn
    gamma = float(spec["scores"]["gamma"])
    if abs(learn.gamma - gamma) > 0:
        raise SystemExit("refused: discount differs")
    seen = spec["deployment"]["seen_design_values"]
    an_stall = A.anchors_of(p, variant, "stall")["strong_current"]
    an_nost = A.anchors_of(p, variant, "no_stall")["strong_current"]
    x0 = TC.bank_starts("strong_current", TC.BANK_SIZE)[:N_STARTS]

    per = {}          # per[seed][sigma] = dict
    checks = {}
    meta = {}
    for seed in SEEDS:
        t0 = time.perf_counter()
        per[seed] = {s: {} for s in SIGMAS}
        # ---------------- the no-stall world's log, critic and actor (primary)
        sd = A.load_seed(p.runs, p.config, variant, "no_stall", seed)
        lg, pg = sd.log, sd.prepared("gap", learn)
        g = groups(sd, pg, "no_stall", seed, spec)
        checks[f"no_stall/seed{seed}/groups"] = g["checks"]
        checks[f"no_stall/seed{seed}/sigma-1_outcomes"] = check_outcomes(
            lg, np.concatenate([g["ft"], g["rest"]]), phys_of(-1.0), f"no_stall/seed{seed} sigma -1")
        rec, critic = A.get_critic(p, sd, "gap", "stall_aware", train=False)
        fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
        # logged-transition scores (for the sigma = -1 consistency report)
        s_ft_logged = E.overestimates(critic, L._transition(lg, g["ft"], pg.obs_mean, pg.obs_std), fixed, gamma) \
            if critic is not None else None
        pol, lo, hi, ainfo = actor_setup("no_stall", seed, pg)
        meta[seed] = dict(t_lo_no_stall=g["t_lo"], n_full_throttle=int(len(g["ft"])), n_rest=int(len(g["rest"])),
                          n_removed_full_throttle=int(len(g["removed_ft"])),
                          n_full_throttle_at_or_below_knee=g["n_ft_at_or_below_knee"],
                          critic_no_stall=None if rec is None else rec["result"]["file_sha256"],
                          critic_failed=critic is None, actor_no_stall=ainfo,
                          recorded_range_no_stall=dict(lo=lo.tolist(), hi=hi.tolist()))
        for sigma in SIGMAS:
            ph = phys_of(sigma)
            d = per[seed][sigma]
            d["rest_outcomes"] = check_outcomes(lg, g["rest"], ph, f"no_stall/seed{seed} rest sigma {sigma}")
            Jn, _ = rollout_J(pol, ph, x0, lo, hi, g["t_lo"], "never")
            Jo, fs = rollout_J(pol, ph, x0, lo, hi, g["t_lo"], "on")
            d["points_never"], d["points_on"], d["fire_share"] = pts(Jn, an_stall), pts(Jo, an_stall), fs
            d["v"] = d["points_on"] - d["points_never"]
            if sigma == -1.0:
                nv, on = pts(Jn, an_nost), pts(Jo, an_nost)
                sv = seen[f"no_stall/seed{seed}/strong_current"]
                ok = abs(nv - sv["never"]) <= TOL and abs(on - sv["on"]) <= TOL
                checks[f"no_stall/seed{seed}/rollouts_reproduce_design"] = dict(
                    ok=bool(ok), never=nv, on=on, seen_never=sv["never"], seen_on=sv["on"])
                if not ok:
                    raise SystemExit(f"PLUMBING STOP: no-stall rollouts seed {seed}: {nv} {on} vs {sv}")
            if critic is None:
                d["cert"] = dict(on=False, reason="no-stall gate critic failed")
                d["rank_p"] = None
                d["rich"] = dict(on=False)
                continue
            s_ft = scores(critic, fixed, gamma, lg, g["ft"], ph, pg)
            s_rest = scores(critic, fixed, gamma, lg, g["rest"], ph, pg)
            if sigma == -1.0:
                d["max_abs_score_diff_vs_logged_transitions"] = float(np.max(np.abs(s_ft - s_ft_logged)))
            c = S.certificate(s_ft, s_rest, [20261015, 9970, seed, 2], cfg)
            d["cert"] = cert_summary(c)
            d["rank_p"] = S.rank_test(s_ft, s_rest)["p"]
            s_rem = scores(critic, fixed, gamma, lg, g["removed_ft"], ph, pg)
            cr = S.certificate(np.concatenate([s_ft, s_rem]), s_rest, [20261015, 9971, seed, 2], cfg)
            d["rich"] = dict(on=cr["on"], q05=cr["q05"], height=cr["height"], mean_tail=cr["mean_tail"],
                             n_tail=cr["n_tail"], rank_p=S.rank_test(np.concatenate([s_ft, s_rem]), s_rest)["p"])
            d["scores_full_throttle"] = s_ft.tolist()
        del sd, lg, pg, critic
        # ---------------- the stalling world's log, critic and actor (plumbing (b), (d) and sensitivity)
        sd = A.load_seed(p.runs, p.config, variant, "stall", seed)
        lg, pg = sd.log, sd.prepared("gap", learn)
        g = groups(sd, pg, "stall", seed, spec)
        checks[f"stall/seed{seed}/groups"] = g["checks"]
        checks[f"stall/seed{seed}/sigma1.5_outcomes"] = check_outcomes(
            lg, np.concatenate([g["ft"], g["rest"]]), phys_of(1.5), f"stall/seed{seed} sigma 1.5")
        rec, critic = A.get_critic(p, sd, "gap", "stall_aware", train=False)
        fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
        pol, lo, hi, ainfo = actor_setup("stall", seed, pg)
        meta[seed].update(t_lo_stall=g["t_lo"], n_full_throttle_stall_log=int(len(g["ft"])),
                          critic_stall=None if rec is None else rec["result"]["file_sha256"], actor_stall=ainfo)
        for sigma in SIGMAS:
            ph = phys_of(sigma)
            d = per[seed][sigma]
            Jn, _ = rollout_J(pol, ph, x0, lo, hi, g["t_lo"], "never")
            Jo, fs = rollout_J(pol, ph, x0, lo, hi, g["t_lo"], "on")
            d["stall_actor"] = dict(points_never=pts(Jn, an_stall), points_on=pts(Jo, an_stall), fire_share=fs)
            d["stall_actor"]["v"] = d["stall_actor"]["points_on"] - d["stall_actor"]["points_never"]
            if sigma == 1.5:
                sv = seen[f"stall/seed{seed}/strong_current"]
                ok = (abs(d["stall_actor"]["points_never"] - sv["never"]) <= TOL
                      and abs(d["stall_actor"]["points_on"] - sv["on"]) <= TOL)
                checks[f"stall/seed{seed}/rollouts_reproduce_design"] = dict(ok=bool(ok), **d["stall_actor"],
                                                                            seen_never=sv["never"], seen_on=sv["on"])
                if not ok:
                    raise SystemExit(f"PLUMBING STOP: stalling rollouts seed {seed}: {d['stall_actor']} vs {sv}")
            if critic is None:
                d["stall_critic"] = dict(on=False, reason="stalling gate critic failed")
                continue
            s_ft = scores(critic, fixed, gamma, lg, g["ft"], ph, pg)
            s_rest = scores(critic, fixed, gamma, lg, g["rest"], ph, pg)
            cs = S.certificate(s_ft, s_rest, [20261015, 9972, seed, 1], cfg)
            d["stall_critic"] = dict(on=cs["on"], q05=cs["q05"], height=cs["height"], mean_tail=cs["mean_tail"],
                                     mean_rest=cs["mean_rest"], rank_p=S.rank_test(s_ft, s_rest)["p"])
        del sd, lg, pg, critic
        log(f"seed {seed}: " + "; ".join(
            f"s{sg:+.2f} v {per[seed][sg]['v']:+.1f} q05 {float(per[seed][sg]['cert'].get('q05') or float('nan')):+.3f} "
            f"{'ON' if per[seed][sg]['cert']['on'] else 'off'}" for sg in SIGMAS)
            + f"; {time.perf_counter() - t0:.0f}s wall, cpu so far {time.process_time() - c_start:.0f}s")

    # ---------------- decision arithmetic (KILL_RULE.md)
    def ladder(dec, val):
        """dec(seed, sigma) -> bool, val(seed, sigma) -> float: per-sigma seed-mean of d*v."""
        return {s: float(np.mean([val(k, s) if dec(k, s) else 0.0 for k in SEEDS])) for s in SIGMAS}

    v = lambda k, s: per[k][s]["v"]  # noqa: E731
    v_sa = lambda k, s: per[k][s]["stall_actor"]["v"]  # noqa: E731
    m = {s: float(np.mean([v(k, s) for k in SEEDS])) for s in SIGMAS}
    m_sa = {s: float(np.mean([v_sa(k, s) for k in SEEDS])) for s in SIGMAS}

    def nets(lad):
        return dict(per_sigma=lad, sum9=float(sum(lad[s] for s in SIGMAS)),
                    sum7=float(sum(lad[s] for s in INTERIOR)))

    rules = {
        "bca": lambda k, s: bool(per[k][s]["cert"]["on"]),
        "always": lambda k, s: True,
        "never": lambda k, s: False,
        "rank_test_p_lt_0.05": lambda k, s: per[k][s]["rank_p"] is not None and per[k][s]["rank_p"] < 0.05,
        "plain_split_conformal_gap_gt_0.10": lambda k, s: (per[k][s]["cert"].get("plain_gap") is not None
                                                           and per[k][s]["cert"]["plain_gap"] > 0.10),
        "rich_evidence_bca": lambda k, s: bool(per[k][s]["rich"]["on"]),
        "stalling_critic_bca": lambda k, s: bool(per[k][s]["stall_critic"]["on"]),
    }
    oracle_lad = {s: max(m[s], 0.0) for s in SIGMAS}
    primary = {name: nets(ladder(r, v)) for name, r in rules.items()}
    primary["oracle"] = nets(oracle_lad)
    sens_actor = {name: nets(ladder(r, v_sa)) for name, r in rules.items()}
    sens_actor["oracle"] = nets({s: max(m_sa[s], 0.0) for s in SIGMAS})

    bca9, bca7 = primary["bca"]["sum9"], primary["bca"]["sum7"]
    orc9 = primary["oracle"]["sum9"]
    best_const7 = max(primary["always"]["sum7"], 0.0)
    k1 = bool(bca9 < 0.8 * orc9)
    k2 = bool(bca7 - best_const7 < 10.0)
    verdict = "SHELVED" if (k1 or k2) else "ALIVE (no kill condition met)"

    # switch-on sigma and break-even sigma (reported only)
    switch_on = {k: next((s for s in SIGMAS if per[k][s]["cert"]["on"]), None) for k in SEEDS}
    monotone = {k: all((not per[k][a]["cert"]["on"]) or per[k][b]["cert"]["on"]
                       for a, b in zip(SIGMAS[:-1], SIGMAS[1:])) for k in SEEDS}
    first_pos = next((s for s in SIGMAS if m[s] > 0), None)
    cross = None
    for a, b in zip(SIGMAS[:-1], SIGMAS[1:]):
        if m[a] <= 0 < m[b]:
            cross = float(a + (0 - m[a]) * (b - a) / (m[b] - m[a]))
            break
    first_pos_seed = {k: next((s for s in SIGMAS if v(k, s) > 0), None) for k in SEEDS}

    ru = resource.getrusage(resource.RUSAGE_SELF)
    results = dict(
        label=LABEL, kill_rule_sha256=kr_sha, prespec_sha256=spec_sha, step1_code_manifest_sha256=man_sha,
        gate_prereg_sha256=p.digest, sigmas=SIGMAS, interior_sigmas=INTERIOR,
        value_seed_mean=m, value_seed_mean_stalling_actor=m_sa,
        ladder_nets=primary, ladder_nets_stalling_actor_values=sens_actor,
        kill=dict(bca_sum9=bca9, oracle_sum9=orc9, ratio=(bca9 / orc9 if orc9 > 0 else None),
                  rule1_below_80pct_of_oracle=k1, bca_sum7=bca7, always_sum7=primary["always"]["sum7"],
                  best_constant_sum7=best_const7, margin_over_best_constant_sum7=bca7 - best_const7,
                  rule2_less_than_10_above_best_constant=k2, verdict=verdict),
        switching=dict(bca_switch_on_sigma=switch_on, bca_monotone=monotone, break_even_first_ladder_sigma=first_pos,
                       break_even_linear_crossing=cross, break_even_first_ladder_sigma_per_seed=first_pos_seed),
        per_seed_sigma={str(k): {str(s): per[k][s] for s in SIGMAS} for k in SEEDS}, meta=meta, checks=checks,
        cpu_seconds=dict(process=time.process_time() - c_start, user=ru.ru_utime, system=ru.ru_stime),
        wall_seconds=time.perf_counter() - t_start, machine=platform.node(),
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    digest = _write(out_json, results)
    log(f"value seed-mean by sigma: {json.dumps({str(s): round(m[s], 2) for s in SIGMAS})}")
    log(f"BCA ladder by sigma: {json.dumps({str(s): round(primary['bca']['per_sigma'][s], 2) for s in SIGMAS})}")
    log(f"KILL: {results['kill']}")
    log(f"switch-on {switch_on}; break-even first {first_pos}, crossing {cross}")
    for name in rules:
        log(f"  {name}: sum9 {primary[name]['sum9']:.1f} sum7 {primary[name]['sum7']:.1f}")
    log(f"  oracle: sum9 {orc9:.1f} sum7 {primary['oracle']['sum7']:.1f}")
    log(f"results {out_json} sha {digest}; cpu {results['cpu_seconds']}")


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
