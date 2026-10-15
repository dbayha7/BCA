"""Step 1 evaluation, read-only on everything except its own result files. EXPLORATORY; see PRESPEC.md.

Reads prespec.json (SHA-256 checked) and never changes it. For each seed 0-4 and world it loads the gate's data
(acceptance.load_seed: regenerated and checked), the step-1 critic (SHA-256 checked against its record), the gate's
own gapped critic (read path only, for the in-sample references), and the saved no-caution actor of v6's Stage 2
policy test (SHA-256 checked). It then computes, exactly as prespec.json fixes them:
  the held-out one-sided overestimates of the full-throttle group (survivors + bank rows past t_lo) and of the rest
  (bank rows at or below t_lo); BCA's WBCP posterior on each; the certificate (5% posterior gap > 0.10); the height;
  the plain one-sided Mann-Whitney test; the gated deployment check (never / on / random at the check's own rate,
  strong current and recorded, first 200 starts); the net across worlds; the outcome-shuffle control (200
  permutations); the go/no-go; and the information-only items (trace test bed dose-response, e11 form, noisy target,
  in-sample references, fade, forced-on and rank-switched nets).
Writes runs/shift_explore/restructure/step1/results.json (+ .sha256) and results.md, write-once.

Scratch mode (--critics-root and --out outside the repository): the same code on scratch critics (reproduction check,
smoke test), optionally with fewer permutations / starts; results are labelled SCRATCH and never count.

It must run pinned to one CPU (critic forward passes and rollouts are then bit-reproducible).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1lib as S  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.dataset as DS  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402

R2 = S.ROOT / "runs" / "shift_explore" / "stage2"
TRACE_CODE = S.ROOT / "runs" / "shift_explore" / "trace" / "code"
SITUATIONS = ("strong_current", "recorded")
TOL = 1e-9


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def info_only(what, fn):
    """Run an INFORMATION-ONLY item (I1 trace test bed, I3 noisy target). An error or a refusal inside it is recorded
    in results.json and does not stop the evaluation: after the official critics exist the code can no longer be
    changed (freeze.py refuses), so an information-only item must never be able to block the go/no-go. The decisive
    items and the integrity checks (gap rule, rollout reproduction, e11 reproduction, critic hashes and records) still
    refuse."""
    try:
        return fn()
    except (Exception, SystemExit) as e:
        log(f"INFORMATION-ONLY ITEM FAILED ({what}): {e!r}; recorded, evaluation continues")
        return dict(error=f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------------------------------------ loaders

def gate_context(spec):
    p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(p)
    if p.digest != spec["gate"]["prereg_sha256"]:
        raise SystemExit("refused: the gate's pre-registration differs from the one prespec.json names")
    variant = A.gate_outcome(p)["variant"]
    if variant != spec["gate"]["variant"]:
        raise SystemExit("refused: the gate's variant differs from prespec.json")
    return p, variant


def step1_critic(root, world, seed, spec_sha, man_sha, official, spec):
    """(record, params) of a step-1 critic; (record or None, None) when missing or failed."""
    d = Path(root) / world / f"seed{seed}"
    if not (d / "record.json").exists():
        return None, None
    rec = S.read_json_checked(d / "record.json")
    problems = S.record_problems(rec, spec, spec_sha, man_sha, world, seed) if official else []
    if problems:
        raise SystemExit(f"refused: {d}: {problems}")
    if rec["failed"]:
        return rec, None
    params = L.load_tree(d / "critic.msgpack", L.critic_template(), rec["file_sha256"])
    return rec, params


def none_actor(world, seed):
    d = R2 / "policy" / f"seed{seed}"
    if world == "no_stall":
        d = d / "no_stall"
    lock = S.read_json_checked(d / "lock.json")
    acts = L.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    if L.tree_sha256(acts["none"]) != lock["actor_sha256"]["none"]:
        raise SystemExit(f"refused: the 'none' actor of {d} differs from its lock")
    return acts["none"], dict(dir=str(d), actors_file_sha256=lock["actors_file_sha256"],
                              actor_sha256=lock["actor_sha256"]["none"])


# ------------------------------------------------------------------------------------------------ scores

def overestimates(critic, tr, fixed, gamma, rows=None, noise=None):
    """One-sided overestimates at the transitions tr (runtime Transition, normalised) [rows]: noiseless c(o') or, with
    noise (draws, n, 4), the mean over draws of min_h Q(o', clip(c(o') + noise, -1, 1))."""
    obs, act, rew, nxt, done = (np.asarray(x) for x in (tr.obs, tr.action, tr.reward, tr.next_obs, tr.done))
    if rows is not None:
        obs, act, rew, nxt, done = obs[rows], act[rows], rew[rows], nxt[rows], done[rows]
    q = L.critic_rows(critic, obs, act)[0].min(1)
    c = np.asarray(fixed.apply(fixed.params(), jnp.asarray(nxt, jnp.float32)), np.float32)
    if noise is None:
        qn = L.critic_rows(critic, nxt, c)[0].min(1)
    else:
        qn = np.zeros(len(obs))
        for k in range(len(noise)):
            qn += L.critic_rows(critic, nxt, np.clip(c + noise[k], -1.0, 1.0).astype(np.float32))[0].min(1)
        qn /= len(noise)
    return S.overestimate(q, rew.astype(np.float64), done.astype(np.float64), qn, gamma)


def smoothing_noise(tags, draws, n):
    z = S.rng(tags).standard_normal((draws, n, 4)) * 0.2
    return np.clip(z, -0.5, 0.5).astype(np.float32)


# ------------------------------------------------------------------------------------------------ deployment

def rollout(policy, physics, anchor, x0, lo, hi, t_lo, mode, rate=None, coin=None):
    """The gated deployment check (calibration design e12, restated): each step a = clip(pi(x), -1, 1); 'on' fires
    when max_k |a_k| > t_lo, 'random' fires with probability rate (coin), 'never' never fires; a fired command is
    clipped to the recorded range [lo, hi]."""
    x = np.asarray(x0, np.float64).copy()
    n = len(x)
    J = np.zeros(n)
    fired = np.zeros((n, physics.length), bool)
    gen = np.random.default_rng(coin) if mode == "random" else None
    for k in range(physics.length):
        a_act = np.clip(policy(x), -1.0, 1.0)
        if mode == "random":
            fire = gen.uniform(size=n) < rate
        elif mode == "on":
            fire = np.abs(a_act).max(1) > t_lo
        else:
            fire = np.zeros(n, bool)
        a = np.where(fire[:, None], np.clip(a_act, lo, hi), a_act)
        fired[:, k] = fire
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    pts = TR.points(J, anchor)
    return dict(points=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)), fire_share=float(fired.mean()))


def deployment(p, variant, world, seed, pg, t_lo, n_starts):
    actor, actor_info = none_actor(world, seed)
    pol = L.mlp_policy(actor, pg, "none")
    a = np.asarray(pg.actions, np.float64)
    lo, hi = np.quantile(a, 0.005, axis=0), np.quantile(a, 0.995, axis=0)
    an = A.anchors_of(p, variant, world)
    wi = 0 if world == "stall" else 1
    out = dict(actor=actor_info, recorded_range=dict(lo=lo.tolist(), hi=hi.tolist()), t_lo=t_lo)
    for si, s in enumerate(SITUATIONS):
        x0 = TC.bank_starts(s, TC.BANK_SIZE)[:n_starts]
        r = dict(never=rollout(pol, pg.physics, an[s], x0, lo, hi, t_lo, "never"),
                 on=rollout(pol, pg.physics, an[s], x0, lo, hi, t_lo, "on"))
        r["random_at_rate"] = rollout(pol, pg.physics, an[s], x0, lo, hi, t_lo, "random",
                                      rate=r["on"]["fire_share"], coin=[20261014, seed, wi, si, 993])
        out[s] = r
    return out


# ------------------------------------------------------------------------------------------------ trace (info only)

def trace_info(p, sds, seed, gr_by_world, cfg, gamma):
    """I1: the step-1 statistic on the saved trace test bed critics (in-sample full-throttle rows)."""
    if str(TRACE_CODE) not in sys.path:
        sys.path.insert(0, str(TRACE_CODE))
    import tracebed as T  # noqa: E402  (read-only use: build_trace, trace_prepared, job_dir)
    out = {}
    learn = p.config.learn
    for world in S.WORLDS:
        sd = sds[world]
        t_lo = gr_by_world[world]["t_lo"]
        fixed = L.FixedController("stall_aware", sd.physics, *sd.norm())
        rows_out = {}
        for si, share in enumerate(T.SHARES if 0.0 in T.SHARES else (0.0,) + tuple(T.SHARES)):
            d = T.job_dir(world, seed, share)
            if share == 0.0:
                prep = sd.prepared("gap", learn)
                rec, critic = A.get_critic(p, sd, "gap", "stall_aware", train=False)
                src = "the gate's gapped critic"
            else:
                rows_rec = S.read_json_checked(d / "trace_rows.json")
                tr = T.build_trace(sds, world, share)
                prep = T.trace_prepared(sd, tr, learn)
                if (DS.sha256_array(tr.added_rows) != rows_rec["added_rows_sha256"]
                        or DS.sha256_array(tr.removal.kept_train_rows(sd.log)) != rows_rec["kept_rows_sha256"]
                        or prep.sha256 != rows_rec["prepared"]["sha256"]):
                    raise SystemExit(f"refused: the rebuilt trace log differs from {d}/trace_rows.json")
                lock = S.read_json_checked(d / "lock.json")
                critic = L.load_tree(d / "critic.msgpack", L.critic_template(), lock["trace_critic_file_sha256"])
                src = f"trace critic {d}"
            t_tr = S.throttle(prep.actions)
            ft = S.full_throttle(t_tr, t_lo)
            s_tr = overestimates(critic, prep.train, fixed, gamma)
            s_bk = overestimates(critic, prep.bank, fixed, gamma)
            bt = S.full_throttle(S.throttle(np.asarray(prep.bank.action)), t_lo)
            w = S.WORLD_TAG[world]
            ca = S.certificate(s_tr[ft], s_tr[~ft], [20261015, 9950, seed, w, si + 1, 1], cfg)
            cb = S.certificate(s_tr[ft], s_bk[~bt], [20261015, 9950, seed, w, si + 1, 2], cfg)
            own = S.gap_rule(t_tr)
            rows_out[f"{share:g}"] = dict(
                source=src, n_full_throttle_in_sample=int(ft.sum()), n_rest_in_sample=int((~ft).sum()),
                n_bank_rest=int((~bt).sum()),
                in_sample_rest=dict(q05=ca["q05"], q50=ca["q50"], on=ca["on"], height=ca["height"],
                                    e11_form_q05=ca["e11_form_q05"]),
                held_out_bank_rest=dict(q05=cb["q05"], q50=cb["q50"], on=cb["on"], height=cb["height"]),
                mean_overestimate_full_throttle=float(s_tr[ft].mean()),
                gap_rule_on_trace_commands=dict(defined=own["defined"], t_lo=own["t_lo"], t_up=own["t_up"],
                                                n_tail=own["n_tail"]))
            log(f"  trace {world}/seed {seed}/share {share:g}: in-sample q05 {ca['q05']:+.3f} ({'on' if ca['on'] else 'off'}), "
                f"bank-rest q05 {cb['q05']:+.3f}; n_full {int(ft.sum())}")
        out[world] = rows_out
    return out


# ------------------------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--critics-root", default=None, help="scratch mode: folder of scratch critics")
    ap.add_argument("--out", default=None, help="scratch mode: output folder outside the repository")
    ap.add_argument("--perms", type=int, default=None, help="scratch only")
    ap.add_argument("--starts", type=int, default=None, help="scratch only")
    ap.add_argument("--skip-trace", action="store_true", help="scratch only")
    ap.add_argument("--allow-missing", action="store_true",
                    help="evaluate although a critic record is missing (missing counts as NO-GO)")
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    official = a.critics_root is None and a.out is None
    if official and (a.perms is not None or a.starts is not None or a.skip_trace):
        raise SystemExit("refused: --perms, --starts and --skip-trace are for scratch runs only")
    if (a.critics_root is None) != (a.out is None):
        raise SystemExit("refused: scratch mode needs both --critics-root and --out")
    spec, spec_sha = S.load_prespec()
    man, man_sha = S.verify_manifest()
    cpu = S.require_one_cpu("the evaluation")
    for k, v in spec["training"]["environment"].items():
        if os.environ.get(k) != v:
            raise SystemExit(f"refused: {k} must be {v!r}")
    critics_root = S.OFFICIAL_CRITICS if official else Path(a.critics_root)
    out_dir = S.guard_out(S.OFFICIAL_OUT if official else Path(a.out), official)
    if (out_dir / "results.json").exists():
        raise SystemExit(f"refused: {out_dir}/results.json exists (write-once)")
    seeds = list(spec["seeds"])
    cfg = spec["certificate"]
    n_perm = a.perms if a.perms is not None else spec["shuffle"]["permutations"]
    n_starts = a.starts if a.starts is not None else 200
    if official:
        missing = [f"{w}/seed{s}" for w in S.WORLDS for s in seeds
                   if not (critics_root / w / f"seed{s}" / "record.json").exists()]
        if missing and not a.allow_missing:
            raise SystemExit(f"refused: critic records missing for {missing}; finish (or rerun interrupted) "
                             f"trainings first, or pass --allow-missing to evaluate them as NO-GO")
        if missing and (S.OFFICIAL_OUT / "run.lock").exists():
            # Review fix: results.json is write-once, so an evaluation with missing critics while run_step1.sh is
            # still training would fix a premature NO-GO for good. run_step1.sh never passes --allow-missing.
            raise SystemExit(f"refused: {S.OFFICIAL_OUT / 'run.lock'} exists (a run may still be training {missing}); "
                             f"--allow-missing is only for after every training has ended")

    p, variant = gate_context(spec)
    learn = p.config.learn
    gamma = float(spec["scores"]["gamma"])
    if abs(learn.gamma - gamma) > 0:
        raise SystemExit("refused: the gate's discount differs from prespec.json")
    per, deploy, certs, shuffled_on, checks, trace = {}, {}, {w: {} for w in S.WORLDS}, {w: {} for w in S.WORLDS}, \
        {}, {}
    for seed in seeds:
        sds = {w: A.load_seed(p.runs, p.config, variant, w, seed) for w in S.WORLDS}
        grs = {}
        for world in S.WORLDS:
            key = f"{world}/seed{seed}"
            t0 = time.perf_counter()
            w = S.WORLD_TAG[world]
            sd = sds[world]
            pg = sd.prepared("gap", learn)
            gcfg = spec["groups"]["gap_rule"]
            t_tr = S.throttle(pg.actions)
            gr = S.gap_rule(t_tr, gcfg["upper_quantile"], gcfg["min_gap"], gcfg["min_tail"])
            grs[world] = gr
            exp = spec["groups"]["expected_from_design"][key]
            checks[f"{key}/gap_rule_as_design"] = bool(gr["defined"] and gr["n_tail"] == exp["n_survivors"]
                                                       and abs(gr["t_lo"] - exp["t_lo"]) <= 1e-12)
            if not checks[f"{key}/gap_rule_as_design"]:
                raise SystemExit(f"refused: the gap rule on {key} differs from the design's")
            surv_idx = np.flatnonzero(gr["tail"])                 # positions among the gapped training rows
            surv_rows = np.asarray(pg.rows[surv_idx], np.int64)
            fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
            bank_t = S.throttle(np.asarray(pg.bank.action))
            bank_ft = S.full_throttle(bank_t, gr["t_lo"])
            # deployment rollouts (independent of every critic)
            dep = deployment(p, variant, world, seed, pg, gr["t_lo"], n_starts)
            deploy[key] = dep
            if n_starts == 200:
                for s in SITUATIONS:
                    seen = spec["deployment"]["seen_design_values"][f"{key}/{s}"]
                    ok = (abs(dep[s]["never"]["points"] - seen["never"]) <= TOL
                          and abs(dep[s]["on"]["points"] - seen["on"]) <= TOL)
                    if world == "stall":
                        ok = ok and abs(dep[s]["random_at_rate"]["points"] - seen["random_at_rate"]) <= TOL
                    checks[f"{key}/{s}/rollouts_reproduce_design"] = bool(ok)
                    if not ok:
                        raise SystemExit(f"refused: the rollouts of {key}/{s} do not reproduce the design's values")
            # in-sample references with the gate's own critic (I4, I5) and the e11 reproduction check
            grec, gcrit = A.get_critic(p, sd, "gap", "stall_aware", train=False)
            g_tr = overestimates(gcrit, pg.train, fixed, gamma)
            g_bk = overestimates(gcrit, pg.bank, fixed, gamma)
            e11_c = S.wbcp.calibrate(g_tr[gr["tail"]], np.random.default_rng([20261015, 9400, seed, ord("B")]),
                                     alpha=0.1, beta=0.95, draws=1000)
            bulk_q = float(np.quantile(g_tr[~gr["tail"]], 0.9, method="inverted_cdf"))
            e11_q05 = float(np.quantile(e11_c.crossings - bulk_q, 0.05))
            seen_q05 = spec["certificate"]["seen_in_sample_e11_form"][key]["q05"]
            checks[f"{key}/e11_reproduced"] = bool(abs(e11_q05 - seen_q05) <= TOL)
            if not checks[f"{key}/e11_reproduced"]:
                raise SystemExit(f"refused: the in-sample e11 number of {key} is {e11_q05}, the design saved "
                                 f"{seen_q05}: the residuals do not match the design's")
            ins_tail = np.concatenate([g_tr[gr["tail"]], g_bk[bank_ft]])
            ins = S.certificate(ins_tail, g_bk[~bank_ft], [20261015, 9700, seed, w], cfg)
            # the step-1 critic
            rec, critic = step1_critic(critics_root, world, seed, spec_sha, man_sha, official, spec)
            entry = dict(t_lo=gr["t_lo"], t_up=gr["t_up"], n_survivors=int(len(surv_rows)),
                         n_bank_full_throttle=int(bank_ft.sum()), n_bank_rest=int((~bank_ft).sum()),
                         critic=None if rec is None else dict(
                             file_sha256=rec["file_sha256"], failed=rec["failed"], mode=rec["mode"],
                             updates=rec["result"]["updates"], seconds=rec["seconds"],
                             equal_to_gate_critic=rec["compare_with_gate"]["file_sha256_equal_gate"],
                             initial_params_equal_gate=rec["compare_with_gate"]["initial_params_equal_gate"]),
                         in_sample_reference=dict(e11_form_q05=e11_q05, step1_statistic_vs_bank=ins))
            if rec is not None and S.array_sha256(surv_rows) != rec["survivors_sha256"]:
                raise SystemExit(f"refused: {key}: the survivors differ from those removed in training")
            if (official and rec is not None
                    and S.array_sha256(np.setdiff1d(pg.rows, surv_rows)) != rec["training_rows_sha256"]):
                raise SystemExit(f"refused: {key}: the critic's training rows are not the gapped rows minus survivors")
            if critic is None:
                entry["certificate"] = dict(on=False, reason="critic missing" if rec is None else "critic failed")
                certs[world][seed] = False
                shuffled_on[world][seed] = [False] * n_perm
                per[key] = entry
                log(f"{key}: NO CRITIC ({entry['certificate']['reason']})")
                continue
            s_tr_surv = overestimates(critic, pg.train, fixed, gamma, rows=surv_idx)
            s_bk = overestimates(critic, pg.bank, fixed, gamma)
            tail = np.concatenate([s_tr_surv, s_bk[bank_ft]])
            rest = s_bk[~bank_ft]
            cert = S.certificate(tail, rest, [20261015, 9700, seed, w], cfg)
            certs[world][seed] = bool(cert["on"])
            rank = S.rank_test(tail, rest)
            sh = S.shuffled_certificates(tail, rest, [20261015, 9700, seed, w], [20261015, 9800, seed, w],
                                         n_perm, cfg)
            shuffled_on[world][seed] = sh["on"]
            ns = len(surv_idx)

            def noisy_target():      # I3, information only
                noise = smoothing_noise([20261015, 9900, seed, w, 1], 32, ns + len(bank_t))   # survivors, then bank
                nz_tr = overestimates(critic, pg.train, fixed, gamma, rows=surv_idx, noise=noise[:, :ns])
                nz_bk = overestimates(critic, pg.bank, fixed, gamma, noise=noise[:, ns:])
                c = S.certificate(np.concatenate([nz_tr, nz_bk[bank_ft]]), nz_bk[~bank_ft],
                                  [20261015, 9700, seed, w], cfg)
                return dict(q05=c["q05"], on=c["on"], height=c["height"])

            noisy = info_only(f"{key} noisy target", noisy_target)
            e11_held = S.certificate(tail, rest, [20261015, 9700, seed, w], dict(cfg, margin=0.0))
            sh_q = np.asarray([v for v in sh["q05"] if v is not None], np.float64)
            sh_h = np.asarray([v for v in sh["height"] if v is not None], np.float64)
            entry.update(
                certificate=cert, rank_test=rank,
                shuffle=dict(on_rate=float(np.mean(sh["on"])), q05_median=float(np.median(sh_q)),
                             q05_q95=float(np.quantile(sh_q, 0.95)),
                             height_median_information_only=float(np.median(sh_h)) if len(sh_h) else None),
                groups=dict(n_full_throttle=int(len(tail)), n_rest=int(len(rest)),
                            mean_full_throttle=float(tail.mean()), mean_rest=float(rest.mean()),
                            q90_full_throttle=float(np.quantile(tail, 0.9, method="inverted_cdf")),
                            q90_rest=float(np.quantile(rest, 0.9, method="inverted_cdf"))),
                information=dict(
                    e11_form_held_out=dict(q05_margin_0_10=cert["e11_form_q05"], on_margin_0_10=cert["e11_form_on"],
                                           on_margin_0=e11_held["e11_form_on"]),
                    certificate_margin_0=dict(on=e11_held["on"], q05=e11_held["q05"]),
                    noisy_target=noisy,
                    fade=dict(survivor_mean_overestimate_gate_critic_in_sample=float(g_tr[gr["tail"]].mean()),
                              survivor_mean_overestimate_step1_critic_held_out=float(s_tr_surv.mean()),
                              per_survivor_gate=g_tr[gr["tail"]].tolist(), per_survivor_step1=s_tr_surv.tolist())))
            per[key] = entry
            log(f"{key}: n_tail {len(tail)} n_rest {len(rest)}; q05 {cert['q05']:+.3f} -> {'ON' if cert['on'] else 'off'}; "
                f"height {cert['height']:.3f}; MW p {rank['p']:.2g}; shuffled on-rate {np.mean(sh['on']):.3f}; "
                f"in-sample q05 {ins['q05']:+.3f}; {time.perf_counter() - t0:.0f}s")
        if not a.skip_trace and seed in (0, 1, 2):
            trace[f"seed{seed}"] = info_only(f"seed {seed} trace test bed",
                                             lambda: trace_info(p, sds, seed, grs, cfg, gamma))
        del sds

    # ---------------------------------------------------------------------------------------- aggregate
    pts = {w: {s: dict(never=deploy[f"{w}/seed{s}"]["strong_current"]["never"]["points"],
                       on=deploy[f"{w}/seed{s}"]["strong_current"]["on"]["points"]) for s in seeds} for w in S.WORLDS}
    pts_rec = {w: {s: dict(never=deploy[f"{w}/seed{s}"]["recorded"]["never"]["points"],
                           on=deploy[f"{w}/seed{s}"]["recorded"]["on"]["points"]) for s in seeds} for w in S.WORLDS}
    net = S.deployment_net(pts, certs, seeds)
    net_recorded = S.deployment_net(pts_rec, certs, seeds)
    sh_sum = S.shuffle_summary(pts, shuffled_on, seeds, net["net"])
    verdict = S.go_no_go(certs, net["net"], sh_sum["mean"], seeds, spec["go_no_go"])
    forced = S.deployment_net(pts, {w: {s: True for s in seeds} for w in S.WORLDS}, seeds)
    rank_on = {w: {s: bool(per[f"{w}/seed{s}"].get("rank_test", {}).get("p") is not None
                           and per[f"{w}/seed{s}"]["rank_test"]["p"] < 0.05) for s in seeds} for w in S.WORLDS}
    rank_net = S.deployment_net(pts, rank_on, seeds)
    rand = {}
    for w in S.WORLDS:
        rows = []
        for s in seeds:
            d = deploy[f"{w}/seed{s}"]["strong_current"]
            on = certs[w][s]
            rows.append(dict(seed=s, certificate=on, check=d["on"]["points"] if on else d["never"]["points"],
                             random_at_rate=d["random_at_rate"]["points"] if on else d["never"]["points"],
                             fire_share=d["on"]["fire_share"] if on else 0.0))
        rand[w] = dict(per_seed=rows, check_mean=float(np.mean([r["check"] for r in rows])),
                       random_mean=float(np.mean([r["random_at_rate"] for r in rows])))
    results = dict(
        label=S.LABEL, mode="official" if official else "SCRATCH (does not count)",
        prespec_sha256=spec_sha, code_manifest_sha256=man_sha, gate_prereg_sha256=p.digest,
        settings=dict(permutations=n_perm, starts=n_starts, trace=not a.skip_trace),
        per_seed_world=per, certificates={w: {str(s): certs[w][s] for s in seeds} for w in S.WORLDS},
        deployment=deploy, net=net, net_recorded_conditions=net_recorded, shuffle=sh_sum,
        random_switching_strong_current=rand, go_no_go=verdict,
        information_only=dict(forced_on_net=forced, rank_test_switched=dict(on=rank_on, net=rank_net),
                              trace_test_bed=trace),
        checks=checks, cpus=cpu, machine=platform.node(), software=A.software_state(),
        seconds=time.perf_counter() - t_start, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    digest = S.write_json_once(out_dir / "results.json", results, official)
    try:
        md = results_md(results, spec, seeds, digest)
    except Exception as e:      # the report must never cost the verdict (results.json is already written)
        md = (f"# Step 1 results: {verdict['verdict']}\n\nThe full report could not be formatted ({e!r}); "
              f"read results.json (SHA-256 {digest}).\n")
    (out_dir / "results.md").write_text(md, encoding="utf8")
    log(f"verdict {verdict['verdict']}: net {net['net']:.1f} (gain {net['gain_stall']:.1f}, loss "
        f"{net['loss_no_stall']:.1f}); shuffled mean {sh_sum['mean']:.1f}; results {out_dir}/results.json")
    return results


def f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "on" if x else "off"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return f"{x:+.{nd}f}" if nd else f"{x:.0f}"


def results_md(r, spec, seeds, digest):
    v = r["go_no_go"]
    L_ = [f"# Step 1 results: {v['verdict']}", "",
          f"{r['label']}", "", f"Mode: {r['mode']}. results.json SHA-256 {digest}. prespec.json {r['prespec_sha256']}.",
          "", "## Conditions (fixed in prespec.json before any step-1 critic was trained)", "",
          f"- C1 certificate on in 5 of 5 stalling seeds: {v['C1_certificate_on_5_of_5_stall']}; "
          f"off in 5 of 5 no-stall seeds: {v['C1_certificate_off_5_of_5_no_stall']}",
          f"- C2 net across worlds {r['net']['net']:.1f} >= 90: {v['C2_net_at_least_90']} "
          f"(stalling gain {r['net']['gain_stall']:.1f}, no-stall loss {r['net']['loss_no_stall']:.1f})",
          f"- C3 mean shuffled net {r['shuffle']['mean']:.1f} <= half of {r['net']['net']:.1f}: "
          f"{v['C3_shuffled_at_most_half']} (95th percentile {r['shuffle']['q95']:.1f}; "
          f"{r['shuffle']['n_perm']} permutations); information only, under the other reading of 'half of that' "
          f"(at most 45, half of the 90-point bar): {v['info_C3_also_at_most_half_of_90']}",
          "- C4 plain rank test reported beside the certificate: below", "",
          "## Per seed and world (held-out overestimates, value units)", "",
          "| world | seed | n full | n rest | 5% gap | certificate | height | MW p | shuffled on-rate |",
          "|---|---|---|---|---|---|---|---|---|"]
    for w in S.WORLDS:
        for s in seeds:
            e = r["per_seed_world"][f"{w}/seed{s}"]
            c = e.get("certificate", {})
            mw = e.get("rank_test", {}).get("p")
            sh = e.get("shuffle", {}).get("on_rate")
            L_.append(f"| {w} | {s} | {e.get('groups', {}).get('n_full_throttle', '-')} | "
                      f"{e.get('groups', {}).get('n_rest', '-')} | {f(c.get('q05'))} | {f(c.get('on'))} | "
                      f"{f(c.get('height'))} | {'-' if mw is None else f'{mw:.2g}'} | "
                      f"{'-' if sh is None else f'{sh:.3f}'} |")
    rs = r["random_switching_strong_current"]["stall"]
    L_ += ["", f"Stalling world, strong current: check {rs['check_mean']:.1f} points against random switching at the "
           f"check's own rate {rs['random_mean']:.1f}. Recorded conditions net {r['net_recorded_conditions']['net']:.1f}.",
           f"Information only: forced-on net {r['information_only']['forced_on_net']['net']:.1f}; rank-test-switched "
           f"net {r['information_only']['rank_test_switched']['net']['net']:.1f}.", "",
           "## Meaning", "", spec["go_no_go"]["meaning_go"] if v["verdict"] == "GO" else spec["go_no_go"]["meaning_no_go"],
           ""]
    return "\n".join(L_)


if __name__ == "__main__":
    main()
