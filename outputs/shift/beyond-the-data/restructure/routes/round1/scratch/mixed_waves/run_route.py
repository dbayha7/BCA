"""Route 5 (mixed_waves): a spliced mixed world where full throttle stalls in big-wave episodes and helps in small-wave
ones (W1), and the mirror (W2). Can BCA's own WBCP posterior, localised by a Gaussian kernel in wave amplitude, switch
caution on only where full throttle stalls?

EXPLORATORY, designed after seeing seeds 0-4. Gate seeds 0-4 only. Read-only: forward passes and rollouts of saved
critics and actors, no training. Kill rule: KILL_RULE.md beside this file (written and hashed before any score).

Writes only runs/shift_explore/routes/mixed_waves/results.json (+ .sha256), write-once.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path("/home/user/bca")
STEP1 = ROOT / "outputs/shift/beyond-the-data/restructure/step1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STEP1))
import step1lib as S  # noqa: E402
import evaluate as E  # noqa: E402  (step 1's evaluation module; only its helpers are used, main() is not run)
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402
from calibration import wbcp  # noqa: E402

SCR = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes")
HERE = SCR / "mixed_waves"
OUT = ROOT / "runs/shift_explore/routes/mixed_waves"
SMOKE = "--smoke" in sys.argv
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
CUT = 0.25
MWORLDS = (1, 2)
N_STARTS = 20 if "--smoke" in sys.argv else 200
N_EXTRA_PERM = 20
EV = {1: "primary", 2: "rich"}
LABEL = ("EXPLORATORY route test (mixed_waves), designed after seeing seeds 0-4; gate seeds 0-4 only; read-only "
         "(forward passes and rollouts of saved networks). Spliced worlds built from whole episodes of the two gate "
         "worlds. Outside the gate; changes no pre-registered rule and no v6/v7 file.")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def guard_out(path):
    p = Path(path).resolve()
    for pin in S.PINNED:
        if S.inside(p, pin):
            raise SystemExit(f"refused: {p} is inside pinned {pin}")
    if SMOKE:
        if not S.inside(p, HERE):
            raise SystemExit(f"refused: smoke outputs only under {HERE}")
    elif not S.inside(p, OUT):
        raise SystemExit(f"refused: outputs only under {OUT}")
    return p


def stalls(amp, m):
    """True where full throttle stalls in spliced world W_m."""
    amp = np.asarray(amp, np.float64)
    return amp > CUT if m == 1 else amp <= CUT


def world_of(amp, m):
    return np.where(stalls(amp, m), "stall", "no_stall")


def silverman(x):
    x = np.asarray(x, np.float64)
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.quantile(x, [0.75, 0.25])
    return 0.9 * min(sd, float(q75 - q25) / 1.34) * len(x) ** (-0.2)


def kernel(amp, a0, h):
    return np.exp(-0.5 * ((np.asarray(amp, np.float64) - a0) / h) ** 2)


def wpost(scores, w, tags, cfg):
    return wbcp.calibrate(np.asarray(scores, np.float64), S.rng(tags), weights=np.asarray(w, np.float64),
                          test_mass=1.0, alpha=float(cfg["alpha"]), beta=float(cfg["credibility"]),
                          draws=int(cfg["draws"]))


def upost(scores, tags, cfg):
    return wbcp.calibrate(np.asarray(scores, np.float64), S.rng(tags), alpha=float(cfg["alpha"]),
                          beta=float(cfg["credibility"]), draws=int(cfg["draws"]))


def rollout_J(policy, physics, x0, lo, hi, t_lo, on):
    """Step 1's rollout rule (evaluate.rollout, modes 'never' / 'on'), returning the per-start return J."""
    x = np.asarray(x0, np.float64).copy()
    n = len(x)
    J = np.zeros(n)
    fired = np.zeros((n, physics.length), bool)
    for k in range(physics.length):
        a_act = np.clip(policy(x), -1.0, 1.0)
        fire = (np.abs(a_act).max(1) > t_lo) if on else np.zeros(n, bool)
        a = np.where(fire[:, None], np.clip(a_act, lo, hi), a_act)
        fired[:, k] = fire
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    return J, fired


def main():
    t_start, c_start = time.perf_counter(), time.process_time()
    kr = (HERE / "KILL_RULE.md").read_bytes()
    kr_sha = S.sha256_bytes(kr)
    if kr_sha != (HERE / "KILL_RULE.md.sha256").read_text().split()[0]:
        raise SystemExit("refused: KILL_RULE.md changed after it was hashed")
    out_json = guard_out(HERE / "smoke" / "results.json" if SMOKE else OUT / "results.json")
    if out_json.exists():
        raise SystemExit(f"refused: {out_json} exists (write-once)")
    spec, spec_sha = S.load_prespec()
    man, man_sha = S.verify_manifest()
    cfg = spec["certificate"]
    gcfg = spec["groups"]["gap_rule"]
    margin = float(cfg["margin"])
    p, variant = E.gate_context(spec)
    learn = p.config.learn
    gamma = float(spec["scores"]["gamma"])
    if abs(learn.gamma - gamma) > 0:
        raise SystemExit("refused: discount differs")
    feas = json.loads((SCR / "_where" / "feas_amp_0_1_2_3_4.json").read_text())
    seen = spec["deployment"]["seen_design_values"]
    anchors = {w: A.anchors_of(p, variant, w)["strong_current"] for w in S.WORLDS}
    x0 = np.asarray(TC.bank_starts("strong_current", TC.BANK_SIZE)[:N_STARTS], np.float64)
    a0s = TC.wave_amplitude(x0)

    checks, per = {}, {}
    # nets[arm][(seed, m)] = mean over starts of (arm points - never points); actor = stalling world's 'none' actor
    nets = {}
    nets_ns = {}       # the same with the no-stall world's 'none' actor (reported only)
    region_rates = {}

    def put(store, arm, seed, m, v):
        store.setdefault(arm, {})[f"{seed}_{m}"] = float(v)

    for seed in SEEDS:
        t_seed = time.perf_counter()
        W = {}
        for world in S.WORLDS:
            sd = A.load_seed(p.runs, p.config, variant, world, seed)
            lg = sd.log
            pg = sd.prepared("gap", learn)
            st = np.asarray(lg["infos/state"], np.float64)
            amp = TC.wave_amplitude(st)
            gr_own = S.gap_rule(S.throttle(pg.actions), gcfg["upper_quantile"], gcfg["min_gap"], gcfg["min_tail"])
            exp = spec["groups"]["expected_from_design"][f"{world}/seed{seed}"]
            checks[f"{world}/seed{seed}/own_gap_rule_as_step1"] = bool(
                gr_own["defined"] and gr_own["n_tail"] == exp["n_survivors"] and abs(gr_own["t_lo"] - exp["t_lo"]) <= 1e-12)
            cal = lg.split_rows("calibration")
            cal = cal[~sd.gap.contains(lg["actions"][cal])]
            t_cal = S.throttle(np.asarray(lg["actions"][cal]))
            bank = np.asarray(pg.bank_rows, np.int64)
            t_bank = S.throttle(np.asarray(pg.bank.action))
            rem = sd.removals["gap"].removed_train_rows(lg)
            t_rem = S.throttle(np.asarray(lg["actions"][rem]))
            # plumbing: amplitude splits with the world's own t_lo reproduce the feasibility counts
            fz = feas[f"{seed}_{world}"]
            held_tail = cal[(t_cal > gr_own["t_lo"]) & ~np.isin(cal, bank)]
            rem_tail = rem[t_rem > gr_own["t_lo"]]
            c = dict(bank_in_cal_outside_ball=bool(np.isin(bank, cal).all()),
                     bank_throttle_consistent=bool(np.array_equal(t_bank, S.throttle(np.asarray(lg["actions"][bank])))),
                     t_lo_as_feas=bool(abs(fz["t_lo"] - gr_own["t_lo"]) <= 1e-12),
                     cal_tail_amp_split_as_feas=bool(int((amp[held_tail] > CUT).sum()) == fz["cal_tail"]["amp_hi"]
                                                     and int((amp[held_tail] <= CUT).sum()) == fz["cal_tail"]["amp_lo"]),
                     removed_amp_split_as_feas=bool(
                         int((amp[rem_tail] > CUT).sum()) == fz["removed_train_full_throttle"]["amp_hi"]
                         and int((amp[rem_tail] <= CUT).sum()) == fz["removed_train_full_throttle"]["amp_lo"]),
                     no_training_row=bool(np.intersect1d(cal, pg.rows).size == 0
                                          and np.intersect1d(rem, pg.rows).size == 0))
            for k, v in c.items():
                checks[f"{world}/seed{seed}/{k}"] = v
            if not (all(c.values()) and checks[f"{world}/seed{seed}/own_gap_rule_as_step1"]):
                raise SystemExit(f"PLUMBING STOP: {world}/seed{seed}: {c}")
            rec, critic = A.get_critic(p, sd, "gap", "stall_aware", train=False)
            if critic is None:
                raise SystemExit(f"PLUMBING STOP: gate critic of {world}/seed{seed} failed")
            fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
            # score a superset: calibration rows outside the ball with throttle > 0.75, every removed row, every bank row
            cal_pick = cal[t_cal > 0.75]
            s_cal = E.overestimates(critic, L._transition(lg, cal_pick, pg.obs_mean, pg.obs_std), fixed, gamma)
            s_rem = E.overestimates(critic, L._transition(lg, rem, pg.obs_mean, pg.obs_std), fixed, gamma)
            s_bank = E.overestimates(critic, pg.bank, fixed, gamma)
            # consistency: bank rows inside cal_pick scored both ways agree
            inb = np.isin(cal_pick, bank)
            pos = np.searchsorted(bank, cal_pick[inb])
            checks[f"{world}/seed{seed}/bank_scores_agree"] = bool(np.array_equal(s_cal[inb], s_bank[pos]))
            if not checks[f"{world}/seed{seed}/bank_scores_agree"]:
                raise SystemExit(f"PLUMBING STOP: {world}/seed{seed}: bank scores differ between transition builds")
            actor, actor_info = E.none_actor(world, seed)
            W[world] = dict(
                pg=pg, physics=pg.physics, crit_sha=rec["result"]["file_sha256"], actor_info=actor_info,
                policy=L.mlp_policy(actor, pg, "none"), own_t_lo=gr_own["t_lo"],
                kept_amp=amp[pg.rows], kept_act=np.asarray(pg.actions, np.float64), kept_t=S.throttle(pg.actions),
                cal=dict(rows=cal_pick, t=S.throttle(np.asarray(lg["actions"][cal_pick])), amp=amp[cal_pick],
                         s=s_cal, obs=np.asarray(lg["observations"][cal_pick], np.float32)),
                rem=dict(rows=rem, t=t_rem, amp=amp[rem], s=s_rem, obs=np.asarray(lg["observations"][rem], np.float32)),
                bank=dict(rows=bank, t=t_bank, amp=amp[bank], s=s_bank))
            del sd, lg
        # plumbing: the per-start rollout code reproduces step 1's saved stalling-world values
        Ws = W["stall"]
        lo_s, hi_s = np.quantile(Ws["kept_act"], 0.005, axis=0), np.quantile(Ws["kept_act"], 0.995, axis=0)
        x0_full = np.asarray(TC.bank_starts("strong_current", TC.BANK_SIZE)[:200], np.float64)
        for mode in ("never", "on"):
            J, _ = rollout_J(Ws["policy"], Ws["physics"], x0_full, lo_s, hi_s, Ws["own_t_lo"], mode == "on")
            v = float(TR.points(J, anchors["stall"]).mean())
            ok = abs(v - seen[f"stall/seed{seed}/strong_current"][mode]) <= 1e-9
            checks[f"stall/seed{seed}/rollout_{mode}_reproduces_step1"] = bool(ok)
            if not ok:
                raise SystemExit(f"PLUMBING STOP: rollout {mode} seed {seed}: {v} vs "
                                 f"{seen[f'stall/seed{seed}/strong_current'][mode]}")
        # never rollouts in both physics, both actors
        never_J = {(act, w): rollout_J(W[act]["policy"], W[w]["physics"], x0, 0, 0, 9.0, False)[0]
                   for act in S.WORLDS for w in S.WORLDS}
        nobs_mean, nobs_std = W["stall"]["pg"].obs_mean, W["stall"]["pg"].obs_std
        x0_n = (np.asarray(x0, np.float32) - nobs_mean) / nobs_std
        per_seed = {}
        for m in MWORLDS:
            t_m = time.perf_counter()
            # spliced log: kept training commands, t_lo and recorded range
            km = [stalls(W[w]["kept_amp"], m) == (w == "stall") for w in S.WORLDS]
            kept_act = np.concatenate([W[w]["kept_act"][k] for w, k in zip(S.WORLDS, km)])
            gr = S.gap_rule(S.throttle(kept_act.astype(np.float32)), gcfg["upper_quantile"], gcfg["min_gap"],
                            gcfg["min_tail"])
            if not gr["defined"]:
                raise SystemExit(f"PLUMBING STOP: gap rule undefined on spliced W{m} seed {seed}: {gr.get('reason')}")
            t_lo = gr["t_lo"]
            lo, hi = np.quantile(kept_act, 0.005, axis=0), np.quantile(kept_act, 0.995, axis=0)

            def gather(kind, tail):
                out = dict(s=[], amp=[], obs=[], world=[])
                for w in S.WORLDS:
                    d = W[w][kind]
                    sel = (stalls(d["amp"], m) == (w == "stall")) & ((d["t"] > t_lo) if tail else (d["t"] <= t_lo))
                    out["s"].append(d["s"][sel])
                    out["amp"].append(d["amp"][sel])
                    if "obs" in d:
                        out["obs"].append(d["obs"][sel])
                    out["world"].append(np.full(int(sel.sum()), w))
                return {k: np.concatenate(v) if len(v) else None for k, v in out.items()}

            groups = {1: gather("cal", True), 2: gather("rem", True)}
            rest = gather("bank", False)
            hs = {e: silverman(groups[e]["amp"]) for e in (1, 2)}
            # world-level certificate (rival), per evidence type
            wl = {e: S.certificate(groups[e]["s"], rest["s"], [20261015, 9961, seed, m, e], cfg) for e in (1, 2)}
            # permutations of the primary full-throttle scores (amplitudes stay)
            perm_main = S.rng([20261015, 9950, seed, m]).permutation(len(groups[1]["s"]))
            perms = [perm_main] + [S.rng([20261015, 9951, seed, m, j + 1]).permutation(len(groups[1]["s"]))
                                   for j in range(N_EXTRA_PERM)]
            wl_perm = S.certificate(groups[1]["s"][perm_main], rest["s"], [20261015, 9961, seed, m, 1], cfg)
            # kNN neighbours (reported only): normalised observation, stalling world's normalisation
            knn_obs = {e: (groups[e]["obs"] - nobs_mean) / nobs_std for e in (1, 2)}
            rest_u = {e: upost(rest["s"], [20261015, 9962, seed, m, e, 2], cfg) for e in (1, 2)}
            dec = {k: np.zeros(N_STARTS, bool) for k in ("local_primary", "local_rich", "perm_primary", "knn_primary",
                                                           "knn_rich", "mw_primary", "mw_rich")}
            dec_extra = np.zeros((N_EXTRA_PERM, N_STARTS), bool)
            stat = {k: np.zeros(N_STARTS) for k in ("local_primary", "local_rich", "perm_primary")}
            n_eff = {k: np.zeros(N_STARTS) for k in ("tail_primary", "tail_rich", "rest")}
            fin_tail = np.zeros(N_STARTS)
            for i, a0 in enumerate(a0s):
                cr = wpost(rest["s"], kernel(rest["amp"], a0, hs[1]), [20261015, 9960, seed, m, 3, 2], cfg)
                n_eff["rest"][i] = cr.n_eff
                for e in (1, 2):
                    if e == 2 and hs[2] != hs[1]:
                        cr_e = wpost(rest["s"], kernel(rest["amp"], a0, hs[2]), [20261015, 9960, seed, m, 3, 2], cfg)
                    else:
                        cr_e = cr
                    wt = kernel(groups[e]["amp"], a0, hs[e])
                    ct = wpost(groups[e]["s"], wt, [20261015, 9960, seed, m, e, 1], cfg)
                    q = S.pairwise_quantile(ct.crossings, cr_e.crossings, float(cfg["statistic_quantile"]))
                    key = "local_" + EV[e]
                    stat[key][i] = q
                    dec[key][i] = q > margin
                    n_eff["tail_" + EV[e]][i] = ct.n_eff
                    if e == 1:
                        fin_tail[i] = float(np.mean(np.isfinite(ct.crossings)))
                        for j, pm in enumerate(perms):
                            ctp = wpost(groups[1]["s"][pm], wt, [20261015, 9960, seed, m, 1, 1], cfg)
                            qp = S.pairwise_quantile(ctp.crossings, cr_e.crossings, float(cfg["statistic_quantile"]))
                            if j == 0:
                                stat["perm_primary"][i] = qp
                                dec["perm_primary"][i] = qp > margin
                            else:
                                dec_extra[j - 1, i] = qp > margin
                    # kNN (uniform weights, whole rest group)
                    k = min(20, len(groups[e]["s"]))
                    dist = np.sqrt(((knn_obs[e] - x0_n[i]) ** 2).sum(1))
                    nn = np.argsort(dist, kind="stable")[:k]
                    ck = upost(groups[e]["s"][nn], [20261015, 9962, seed, m, e, 1], cfg)
                    dec["knn_" + EV[e]][i] = S.pairwise_quantile(ck.crossings, rest_u[e].crossings,
                                                                 float(cfg["statistic_quantile"])) > margin
                    # local Mann-Whitney
                    tw = np.abs(groups[e]["amp"] - a0) <= hs[e]
                    rw = np.abs(rest["amp"] - a0) <= hs[e]
                    if tw.sum() >= 1 and rw.sum() >= 1:
                        pmw = stats.mannwhitneyu(groups[e]["s"][tw], rest["s"][rw], alternative="greater").pvalue
                        dec["mw_" + EV[e]][i] = pmw < 0.05
            # rollouts with the check, both physics, both actors
            on_J = {(act, w): rollout_J(W[act]["policy"], W[w]["physics"], x0, lo, hi, t_lo, True)
                    for act in S.WORLDS for w in S.WORLDS}
            wstart = world_of(a0s, m)
            stall_region = wstart == "stall"

            def pts(Jd, act):
                return np.where(stall_region, TR.points(Jd[(act, "stall")] if not isinstance(Jd[(act, "stall")], tuple)
                                                        else Jd[(act, "stall")][0], anchors["stall"]),
                                TR.points(Jd[(act, "no_stall")] if not isinstance(Jd[(act, "no_stall")], tuple)
                                          else Jd[(act, "no_stall")][0], anchors["no_stall"]))

            arms_dec = dict(dec)
            arms_dec["clip_always"] = np.ones(N_STARTS, bool)
            arms_dec["never"] = np.zeros(N_STARTS, bool)
            arms_dec["world_level_primary"] = np.full(N_STARTS, bool(wl[1]["on"]))
            arms_dec["world_level_rich"] = np.full(N_STARTS, bool(wl[2]["on"]))
            arms_dec["world_level_primary_permuted"] = np.full(N_STARTS, bool(wl_perm["on"]))
            arms_dec["oracle_region"] = stall_region.copy()
            entry = dict(t_lo=t_lo, t_up=gr["t_up"], n_tail_spliced_training=gr["n_tail"], recorded_range=dict(
                lo=lo.tolist(), hi=hi.tolist()), h=dict(primary=hs[1], rich=hs[2]),
                n_full_throttle={EV[e]: dict(total=int(len(groups[e]["s"])),
                                             stall_region=int((groups[e]["world"] == "stall").sum()),
                                             help_region=int((groups[e]["world"] == "no_stall").sum()))
                                 for e in (1, 2)},
                n_rest=dict(total=int(len(rest["s"])), stall_region=int((rest["world"] == "stall").sum()),
                            help_region=int((rest["world"] == "no_stall").sum())),
                mean_score={EV[e]: dict(stall_region=float(groups[e]["s"][groups[e]["world"] == "stall"].mean()),
                                        help_region=float(groups[e]["s"][groups[e]["world"] == "no_stall"].mean()))
                            for e in (1, 2)},
                mean_score_rest=dict(stall_region=float(rest["s"][rest["world"] == "stall"].mean()),
                                     help_region=float(rest["s"][rest["world"] == "no_stall"].mean())),
                q90_rest=float(np.quantile(rest["s"], 0.9, method="inverted_cdf")),
                world_level={EV[e]: dict(on=wl[e]["on"], q05=wl[e]["q05"]) for e in (1, 2)},
                world_level_primary_permuted=dict(on=wl_perm["on"], q05=wl_perm["q05"]),
                share_stall_region_starts=float(stall_region.mean()),
                n_eff_median={k: float(np.median(v)) for k, v in n_eff.items()},
                finite_share_tail_primary_median=float(np.median(fin_tail)),
                statistic={k: dict(stall_region_median=float(np.median(v[stall_region])),
                                   help_region_median=float(np.median(v[~stall_region])))
                           for k, v in stat.items()})
            for act, store in (("stall", nets), ("no_stall", nets_ns)):
                nv = pts(never_J, act)
                ov = pts(on_J, act)
                gain_if_on = ov - nv
                if act == "stall":
                    entry["gain_if_on"] = dict(stall_region_mean=float(gain_if_on[stall_region].mean()),
                                               help_region_mean=float(gain_if_on[~stall_region].mean()))
                    entry["fire_share_on"] = dict(
                        stall_region=float(on_J[(act, "stall")][1][stall_region].mean()),
                        help_region=float(on_J[(act, "no_stall")][1][~stall_region].mean()))
                else:
                    entry["gain_if_on_no_stall_actor"] = dict(stall_region_mean=float(gain_if_on[stall_region].mean()),
                                                              help_region_mean=float(gain_if_on[~stall_region].mean()))
                for arm, d in arms_dec.items():
                    put(store, arm, seed, m, float(np.mean(np.where(d, gain_if_on, 0.0))))
                if act == "stall":
                    extra = [float(np.mean(np.where(d, gain_if_on, 0.0))) for d in dec_extra]
                    put(store, "perm_primary_extra_mean", seed, m, float(np.mean(extra)))
            for arm, d in arms_dec.items():
                region_rates.setdefault(arm, {})[f"{seed}_{m}"] = dict(
                    on_rate_stall_region=float(d[stall_region].mean()), on_rate_help_region=float(d[~stall_region].mean()))
            region_rates.setdefault("perm_primary_extra", {})[f"{seed}_{m}"] = dict(
                on_rate_stall_region=float(dec_extra[:, stall_region].mean()),
                on_rate_help_region=float(dec_extra[:, ~stall_region].mean()))
            per_seed[f"W{m}"] = entry
            log(f"seed {seed} W{m}: t_lo {t_lo:.4f} h {hs[1]:.4f}/{hs[2]:.4f}; n_full primary "
                f"{entry['n_full_throttle']['primary']} rich {entry['n_full_throttle']['rich']['total']}; "
                f"mean score primary {entry['mean_score']['primary']} rest {entry['mean_score_rest']}; "
                f"gain_if_on {entry['gain_if_on']}; world-level {entry['world_level']}; "
                f"local_primary {nets['local_primary'][f'{seed}_{m}']:+.1f} "
                f"(on stall {region_rates['local_primary'][f'{seed}_{m}']['on_rate_stall_region']:.2f} "
                f"help {region_rates['local_primary'][f'{seed}_{m}']['on_rate_help_region']:.2f}); "
                f"local_rich {nets['local_rich'][f'{seed}_{m}']:+.1f} "
                f"(on stall {region_rates['local_rich'][f'{seed}_{m}']['on_rate_stall_region']:.2f} "
                f"help {region_rates['local_rich'][f'{seed}_{m}']['on_rate_help_region']:.2f}); "
                f"perm {nets['perm_primary'][f'{seed}_{m}']:+.1f}; clip {nets['clip_always'][f'{seed}_{m}']:+.1f}; "
                f"oracle {nets['oracle_region'][f'{seed}_{m}']:+.1f}; {time.perf_counter() - t_m:.0f}s")
        per[f"seed{seed}"] = dict(per_seed, critics={w: W[w]["crit_sha"] for w in S.WORLDS},
                                  actors={w: W[w]["actor_info"] for w in S.WORLDS})
        del W
        log(f"seed {seed} done in {time.perf_counter() - t_seed:.0f}s; cpu so far {time.process_time() - c_start:.0f}s")

    def mixed(store, arm):
        return float(np.mean(list(store[arm].values())))

    def by_m(store, arm):
        return {f"W{m}": float(np.mean([store[arm][f"{s}_{m}"] for s in SEEDS])) for m in MWORLDS}

    summary = {arm: dict(mixed_world_net=mixed(nets, arm), by_world=by_m(nets, arm),
                         per_seed_world=nets[arm]) for arm in nets}
    summary_ns = {arm: dict(mixed_world_net=mixed(nets_ns, arm), by_world=by_m(nets_ns, arm)) for arm in nets_ns}
    rates = {arm: dict(on_rate_stall_region=float(np.mean([v["on_rate_stall_region"] for v in d.values()])),
                       on_rate_help_region=float(np.mean([v["on_rate_help_region"] for v in d.values()])))
             for arm, d in region_rates.items()}
    best, margins = {}, {}
    for e in (1, 2):
        cands = {f"world_level_{EV[e]}": summary[f"world_level_{EV[e]}"]["mixed_world_net"],
                 "clip_always": summary["clip_always"]["mixed_world_net"], "never": 0.0}
        b = max(cands, key=cands.get)
        best[EV[e]] = dict(arm=b, net=cands[b], candidates=cands)
        margins[EV[e]] = summary[f"local_{EV[e]}"]["mixed_world_net"] - cands[b]
    rival_perm = max(summary["world_level_primary_permuted"]["mixed_world_net"],
                     summary["clip_always"]["mixed_world_net"], 0.0)
    ctrl_margin = summary["perm_primary"]["mixed_world_net"] - best["primary"]["net"]
    k1 = margins["rich"] < 10.0
    k2 = margins["primary"] < 10.0
    k3 = ctrl_margin >= 0.5 * margins["primary"]
    verdict = "dead end" if (k1 or k2 or k3) else "alive"
    ru = resource.getrusage(resource.RUSAGE_SELF)
    results = dict(
        label=LABEL, kill_rule_sha256=kr_sha, prespec_sha256=spec_sha, step1_code_manifest_sha256=man_sha,
        gate_prereg_sha256=p.digest, cut=CUT, starts=N_STARTS,
        summary=summary, best_rival=best, margins=margins,
        permuted_control=dict(net=summary["perm_primary"]["mixed_world_net"], margin=ctrl_margin,
                              rival_recomputed_with_permuted_scores=rival_perm,
                              extra_permutations_mean_net=summary["perm_primary_extra_mean"]["mixed_world_net"]),
        kill=dict(rule1_rich_margin_below_10=k1, rule2_primary_margin_below_10=k2,
                  rule3_permuted_keeps_half=k3, verdict=verdict),
        on_rates_by_region=rates, no_stall_actor=summary_ns, per_seed=per, checks=checks,
        cpu_seconds=dict(process=time.process_time() - c_start, user=ru.ru_utime, system=ru.ru_stime),
        wall_seconds=time.perf_counter() - t_start, machine=platform.node(),
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    digest = _write(out_json, results)
    for arm in sorted(summary):
        log(f"{arm:32s} mixed net {summary[arm]['mixed_world_net']:+7.1f}  {summary[arm]['by_world']}  "
            f"on-rates {rates.get(arm)}")
    log(f"best rival {best}; margins {margins}; control margin {ctrl_margin:+.1f}; kill {results['kill']}")
    log(f"no-stall actor: { {a: round(v['mixed_world_net'], 1) for a, v in summary_ns.items()} }")
    log(f"cpu {results['cpu_seconds']}; results {out_json} sha {digest}")


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
