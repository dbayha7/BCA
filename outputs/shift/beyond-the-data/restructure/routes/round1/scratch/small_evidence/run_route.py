"""Route 4 (small_evidence): BCA's posterior certificate against plain tests when only n full-throttle outcomes exist.

EXPLORATORY, designed after seeing seeds 0-4. Gate seeds 0-4 only. Forward passes of saved gate critics only: no
training, no rollouts. Kill rule: KILL_RULE.md beside this file (hashed in KILL_RULE.md.sha256 before any draw).

Two phases, run as separate processes:
  prep   load each gate seed (both worlds), rebuild the pool of removed full-throttle training rows (held out from the
         gate critic) and the rest group (bank rows at or below t_lo), score them with the gate's gapped critic
         (noiseless one-sided overestimate, as testbed heldout.py), check against the saved test-bed numbers, and save
         the scores to this folder (scores.npz). Needs JAX.
  run    for each seed-world, n in the ladder and 400 random draws: rules B (step 1's certificate), B0 (margin 0),
         C (plain gap of empirical 90% quantiles > 0.10), R (Mann-Whitney p < 0.05); expected nets; the kill rule.
         NumPy/SciPy only; two worker processes (one per allowed CPU). Writes runs/shift_explore/routes/small_evidence/
         results.json (+ .sha256), write-once.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path("/home/user/bca")
STEP1 = ROOT / "outputs/shift/beyond-the-data/restructure/step1"
for p_ in (str(ROOT), str(STEP1)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import step1lib as S  # noqa: E402

SCR = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad")
HERE = SCR / "routes" / "small_evidence"
TB = SCR / "restructure" / "testbed"
OUT = ROOT / "runs/shift_explore/routes/small_evidence"
STEP1_RESULTS = ROOT / "runs/shift_explore/restructure/step1/results.json"
HCR_RESULTS = ROOT / "runs/shift_explore/routes/heldout_calibration_rows/results.json"
SCORES = HERE / "scores.npz"
SEEDS = [0, 1, 2, 3, 4]
LADDER = [3, 5, 8, 12, 20, 30, 45]
DECIDING_N = [12, 20, 30, 45]
FLAG_N = [12, 20, 30]
DRAWS = 400
MARGIN_C = 0.10
LABEL = ("EXPLORATORY route test (small_evidence), designed after seeing seeds 0-4; gate seeds 0-4 only; forward "
         "passes of saved gate critics only (no training, no rollouts). Outside the gate; changes no pre-registered "
         "rule and no v6/v7 file.")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def check_kill_rule():
    kr = (HERE / "KILL_RULE.md").read_bytes()
    want = (HERE / "KILL_RULE.md.sha256").read_text().split()[0]
    if S.sha256_bytes(kr) != want:
        raise SystemExit("refused: KILL_RULE.md changed after it was hashed")
    return want


def guard_out(path):
    p = Path(path).resolve()
    for pin in S.PINNED:
        if S.inside(p, pin):
            raise SystemExit(f"refused: {p} is inside pinned {pin}")
    if not (S.inside(p, OUT) or S.inside(p, HERE)):
        raise SystemExit(f"refused: outputs only under {OUT} or {HERE}")
    return p


# ------------------------------------------------------------------------------------------------ phase 1: prep

def noiseless_overestimate(L, jnp, cp, lg, rows, mean, std, fixed):
    """Exactly testbed heldout.py's noiseless form, returned as the overestimate (= -deltanl)."""
    x = np.asarray(lg["observations"][rows], np.float32)
    a = np.asarray(lg["actions"][rows], np.float32)
    r = np.asarray(lg["rewards"][rows], np.float64)
    x1 = np.asarray(lg["next_observations"][rows], np.float32)
    dn = np.asarray(lg["terminals"][rows], np.float64)
    obs, nxt = (x - mean) / std, (x1 - mean) / std
    c = np.asarray(fixed.apply(fixed.params(), jnp.asarray(nxt)), np.float32)
    q = L.critic_rows(cp, obs, a)[0].min(1)
    plain = L.critic_rows(cp, nxt, c)[0].min(1)
    disc = 0.98 * (1.0 - dn)
    deltanl = r + disc * plain - q
    return -deltanl


def prep():
    t0, c0 = time.perf_counter(), time.process_time()
    check_kill_rule()
    if SCORES.exists():
        raise SystemExit(f"refused: {SCORES} exists (write-once)")
    import jax.numpy as jnp
    sys.path.insert(0, str(ROOT / "runs/shift_explore/trace/code"))
    import tracebed as T
    import evaluate as E
    import experiments.shift.acceptance as A
    import experiments.shift.learners as L
    spec, spec_sha = S.load_prespec()
    S.verify_manifest()
    gcfg = spec["groups"]["gap_rule"]
    p, variant = E.gate_context(spec)
    learn = p.config.learn
    if abs(float(learn.gamma) - 0.98) > 0 or float(spec["scores"]["gamma"]) != 0.98:
        raise SystemExit("refused: discount differs from 0.98")
    step1 = S.read_json_checked(STEP1_RESULTS)
    arrays, meta, checks = {}, {}, {}
    for seed in SEEDS:
        sds = {w: A.load_seed(p.runs, p.config, variant, w, seed) for w in S.WORLDS}
        pool = np.asarray(T.shared_candidates(sds)[0], np.int64)
        for world in S.WORLDS:
            key = f"{world}/seed{seed}"
            sd = sds[world]
            lg = sd.log
            pg = sd.prepared("gap", learn)
            gr = S.gap_rule(S.throttle(pg.actions), gcfg["upper_quantile"], gcfg["min_gap"], gcfg["min_tail"])
            exp = spec["groups"]["expected_from_design"][key]
            if not (gr["defined"] and gr["n_tail"] == exp["n_survivors"] and abs(gr["t_lo"] - exp["t_lo"]) <= 1e-12):
                raise SystemExit(f"PLUMBING STOP: gap rule on {key}: {gr['t_lo']} {gr['n_tail']} vs {exp}")
            t_lo = gr["t_lo"]
            z = np.load(TB / f"heldout_seed{seed}_{world}.npz")
            zrows = np.asarray(z["rows"], np.int64)
            bank = np.asarray(zrows[z["pos_bank"]], np.int64)
            rg = sd.removals["gap"]
            c = {}
            c["bank_as_npz"] = bool(np.array_equal(np.sort(bank), np.sort(np.asarray(rg.bank_rows(lg), np.int64))))
            bank_t = S.throttle(np.asarray(lg["actions"][bank]))
            rest_rows = bank[bank_t <= t_lo]
            pgb = np.asarray(pg.bank_rows, np.int64)
            pgb_rest = pgb[S.throttle(np.asarray(pg.bank.action)) <= t_lo]
            c["rest_as_pg_bank"] = bool(np.array_equal(np.sort(rest_rows), np.sort(pgb_rest)))
            c["n_rest_as_step1"] = bool(len(rest_rows) == step1["per_seed_world"][key]["groups"]["n_rest"])
            pool_t = S.throttle(np.asarray(lg["actions"][pool]))
            c["pool_past_t_lo"] = bool((pool_t > t_lo).all())
            c["pool_not_training"] = bool(np.intersect1d(pool, np.asarray(pg.rows, np.int64)).size == 0)
            if seed <= 2:
                c["pool_as_npz"] = bool(np.array_equal(pool, np.asarray(zrows[z["pos_cand"]], np.int64)))
            rec, cp = A.get_critic(p, sd, "gap", "stall_aware", train=False)
            if cp is None:
                raise SystemExit(f"{key}: gate critic failed (would count as always off; not expected)")
            mean, std = pg.obs_mean, pg.obs_std
            fixed = L.FixedController("stall_aware", sd.physics, mean, std)
            # Forward pass on the npz's own row set (same batch as the saved numbers) for the reproduction check.
            s_npz_rows = noiseless_overestimate(L, jnp, cp, lg, zrows, mean, std, fixed)
            ref = -np.asarray(z["deltanl_gap"], np.float64)
            pos_bank = np.asarray(z["pos_bank"])
            c["bank_scores_reproduce_npz"] = bool(np.max(np.abs(s_npz_rows[pos_bank] - ref[pos_bank])) <= 1e-9)
            # Pool scores: seeds 0-2 from the npz (checked); seeds 3-4 from the forward pass on the pool rows.
            s_pool_fp = noiseless_overestimate(L, jnp, cp, lg, pool, mean, std, fixed)
            if seed <= 2:
                pos_cand = np.asarray(z["pos_cand"])
                c["pool_scores_reproduce_npz"] = bool(np.max(np.abs(s_pool_fp - ref[pos_cand])) <= 1e-9)
                s_pool = ref[pos_cand]
            else:
                s_pool = s_pool_fp
            # Rest scores from the npz bank positions (identical to the forward pass by the check above).
            bpos = np.searchsorted(zrows, rest_rows)
            s_rest = ref[bpos]
            for k, v in c.items():
                checks[f"{key}/{k}"] = v
            if not all(c.values()):
                raise SystemExit(f"PLUMBING STOP: {key}: {c}")
            arrays[f"{world}_{seed}_pool"] = s_pool
            arrays[f"{world}_{seed}_pool_rows"] = pool
            arrays[f"{world}_{seed}_rest"] = s_rest
            arrays[f"{world}_{seed}_rest_rows"] = rest_rows
            meta[key] = dict(t_lo=t_lo, n_pool=int(len(pool)), n_rest=int(len(rest_rows)),
                             critic_file_sha256=rec["result"]["file_sha256"])
            log(f"{key}: pool {len(pool)} rest {len(rest_rows)}; checks ok")
        del sds
    np.savez(guard_out(SCORES), **arrays)
    sha = S.file_sha256(SCORES)
    (HERE / "scores_meta.json").write_text(json.dumps(dict(
        label=LABEL, scores_sha256=sha, meta=meta, checks=checks, prespec_sha256=spec_sha,
        cpu_seconds=time.process_time() - c0, wall_seconds=time.perf_counter() - t0), indent=1, sort_keys=True))
    log(f"prep done: {SCORES} sha {sha}; cpu {time.process_time() - c0:.0f}s wall {time.perf_counter() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ phase 2: run

def conformal_corrected(scores, alpha=0.1):
    """Textbook split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score; +inf if that exceeds n."""
    s = np.sort(np.asarray(scores, np.float64))
    k = math.ceil((len(s) + 1) * (1.0 - alpha) - 1e-12)
    return float(s[k - 1]) if k <= len(s) else math.inf


def one_seed_world(args):
    """All draws of one seed-world. Returns per-n arrays of q05, plain_gap, corrected gap and Mann-Whitney p."""
    world, seed, pool, rest, cfg = args
    t0 = time.process_time()
    w = S.WORLD_TAG[world]
    cr = S.posterior(rest, [20261015, 9931, seed, w, 2], cfg)
    rest_corr = conformal_corrected(rest, float(cfg["alpha"]))
    out = {}
    for n in LADDER:
        g = S.rng([20261015, 9930, seed, w, n])
        q05, pgap, cgap, pval, idx_all = [], [], [], [], []
        for j in range(DRAWS):
            idx = g.choice(len(pool), n, replace=False)
            tail = pool[idx]
            ct = S.posterior(tail, [20261015, 9931, seed, w, n, j + 1, 1], cfg)
            q05.append(S.pairwise_quantile(ct.crossings, cr.crossings, float(cfg["statistic_quantile"])))
            pgap.append(float(ct.lambda_hat - cr.lambda_hat))
            cgap.append(conformal_corrected(tail, float(cfg["alpha"])) - rest_corr)
            pval.append(S.rank_test(tail, rest)["p"])
            idx_all.append(idx)
        out[n] = dict(q05=np.asarray(q05), plain_gap=np.asarray(pgap), corrected_gap=np.asarray(cgap),
                      p=np.asarray(pval), idx=np.asarray(idx_all))
    return world, seed, out, time.process_time() - t0


def equivalence_check(cfg):
    """Synthetic numbers: the reused-rest-posterior form equals step1lib.certificate (same posteriors)."""
    r = np.random.default_rng(7)
    for n in (3, 12, 45):
        tail, rest = r.normal(size=n) + 0.3, r.normal(size=300)
        tags = [20261015, 9939, n]
        ref = S.certificate(tail, rest, tags, cfg)
        ct = S.posterior(tail, tags + [1], cfg)
        cr = S.posterior(rest, tags + [2], cfg)
        q = S.pairwise_quantile(ct.crossings, cr.crossings, float(cfg["statistic_quantile"]))
        if q != ref["q05"] or float(ct.lambda_hat - cr.lambda_hat) != ref["plain_gap"] or (q > 0.10) != ref["on"]:
            raise SystemExit(f"PLUMBING STOP: equivalence with step1lib.certificate fails at n={n}")
    return True


def run():
    t0, c0 = time.perf_counter(), time.process_time()
    kr_sha = check_kill_rule()
    out_json = guard_out(OUT / "results.json")
    if out_json.exists():
        raise SystemExit(f"refused: {out_json} exists (write-once)")
    spec, spec_sha = S.load_prespec()
    S.verify_manifest()
    cfg = spec["certificate"]
    if float(cfg["margin"]) != 0.10:
        raise SystemExit("refused: the frozen certificate margin is not 0.10")
    smeta = json.loads((HERE / "scores_meta.json").read_text())
    if S.file_sha256(SCORES) != smeta["scores_sha256"]:
        raise SystemExit("refused: scores.npz differs from its recorded hash")
    equivalence_check(cfg)
    z = np.load(SCORES)
    seen = spec["deployment"]["seen_design_values"]
    gain = {s: seen[f"stall/seed{s}/strong_current"]["on"] - seen[f"stall/seed{s}/strong_current"]["never"]
            for s in SEEDS}
    loss = {s: seen[f"no_stall/seed{s}/strong_current"]["never"] - seen[f"no_stall/seed{s}/strong_current"]["on"]
            for s in SEEDS}
    jobs = [(w, s, np.asarray(z[f"{w}_{s}_pool"], np.float64), np.asarray(z[f"{w}_{s}_rest"], np.float64), cfg)
            for s in SEEDS for w in S.WORLDS]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    res = {}
    cpu_children = 0.0
    with ctx.Pool(2) as pool:
        for world, seed, out, cpu in pool.imap_unordered(one_seed_world, jobs):
            res[(world, seed)] = out
            cpu_children += cpu
            log(f"{world}/seed{seed} done ({cpu:.0f} cpu-s): " + "; ".join(
                f"n{n} B {np.mean(o['q05'] > 0.10):.2f} C {np.mean(o['plain_gap'] > MARGIN_C):.2f} "
                f"R {np.mean(o['p'] < 0.05):.2f}" for n, o in out.items()))

    rules = {
        "B": lambda o: o["q05"] > 0.10,
        "B0": lambda o: o["q05"] > 0.0,
        "C": lambda o: o["plain_gap"] > MARGIN_C,
        "R": lambda o: o["p"] < 0.05,
        "C_corrected_info": lambda o: o["corrected_gap"] > MARGIN_C,
    }
    on_rate = {r: {n: {w: {s: float(np.mean(f(res[(w, s)][n]))) for s in SEEDS} for w in S.WORLDS} for n in LADDER}
               for r, f in rules.items()}

    def net_of(rule, n):
        return float(np.mean([on_rate[rule][n]["stall"][s] * gain[s] - on_rate[rule][n]["no_stall"][s] * loss[s]
                              for s in SEEDS]))

    def se_of_diff(rule_a, rule_b, n):
        """Draw-level Monte Carlo standard error of net_a - net_b at n (paired draws)."""
        v = 0.0
        for s in SEEDS:
            for w, k in (("stall", gain[s]), ("no_stall", -loss[s])):
                o = res[(w, s)][n]
                d = k * (rules[rule_a](o).astype(float) - (rules[rule_b](o).astype(float) if rule_b else 0.0))
                v += np.var(d, ddof=1) / len(d)
        return float(math.sqrt(v) / len(SEEDS))

    nets = {r: {n: net_of(r, n) for n in LADDER} for r in rules}
    se = {r: {n: se_of_diff(r, None, n) for n in LADDER} for r in rules}
    D = {}
    for X in ("C", "R"):
        diffs = [nets["B"][n] - nets[X][n] for n in DECIDING_N]
        se_d = math.sqrt(sum(se_of_diff("B", X, n) ** 2 for n in DECIDING_N)) / len(DECIDING_N)
        D[X] = dict(value=float(np.mean(diffs)), per_n={n: d for n, d in zip(DECIDING_N, diffs)}, mc_se=se_d)
    shelve_C = D["C"]["value"] < 5.0
    shelve_R = D["R"]["value"] < 5.0
    route_shelved = bool(shelve_C and shelve_R)
    proposer = {n: nets["B"][n] - max(nets["C"][n], nets["R"][n]) for n in LADDER}
    proposer_shelve = not any(v >= 5.0 for v in proposer.values())
    smallest90 = {r: next((n for n in LADDER if nets[r][n] >= 90.0), None) for r in rules}
    flag = {n: nets["R"][n] - nets["B"][n] for n in FLAG_N}
    flag_on = any(v >= 5.0 for v in flag.values())

    # Group summaries (information): pool and rest score distributions per seed-world.
    groups = {}
    for s in SEEDS:
        for w in S.WORLDS:
            pl, rs = np.asarray(z[f"{w}_{s}_pool"]), np.asarray(z[f"{w}_{s}_rest"])
            q90r = float(np.quantile(rs, 0.9, method="inverted_cdf"))
            groups[f"{w}/seed{s}"] = dict(
                n_pool=int(len(pl)), n_rest=int(len(rs)), mean_pool=float(pl.mean()), mean_rest=float(rs.mean()),
                median_pool=float(np.median(pl)), median_rest=float(np.median(rs)),
                q90_pool=float(np.quantile(pl, 0.9, method="inverted_cdf")), q90_rest=q90r,
                share_pool_above_rest_q90=float(np.mean(pl > q90r)),
                share_pool_above_rest_q90_plus_0p1=float(np.mean(pl > q90r + 0.10)),
                auc_pool_vs_rest=float(S.rank_test(pl, rs)["auc"]))

    step1 = S.read_json_checked(STEP1_RESULTS)
    beside = {}
    for key, v in step1["per_seed_world"].items():
        beside[key] = dict(step1_q05=v["certificate"]["q05"], step1_on=v["certificate"]["on"],
                           step1_plain_gap=v["certificate"].get("plain_gap"), step1_rank_p=v["rank_test"]["p"],
                           step1_n_full=v["groups"]["n_full_throttle"])
    if HCR_RESULTS.exists():
        h = S.read_json_checked(HCR_RESULTS)
        for key, v in h.get("per_seed_world", {}).items():
            if "certificate" in v and "q05" in v["certificate"]:
                beside.setdefault(key, {}).update(
                    heldout_cal_q05=v["certificate"]["q05"], heldout_cal_on=v["certificate"]["on"],
                    heldout_cal_plain_gap=v["certificate"].get("plain_gap"),
                    heldout_cal_rank_p=v["rank_test"]["p"], heldout_cal_n_full=v.get("n_full_throttle"))

    def realistic_net(on_key):
        certs = {w: {s: bool(beside[f"{w}/seed{s}"][on_key]) for s in SEEDS} for w in S.WORLDS}
        pts = {w: {s: dict(never=seen[f"{w}/seed{s}/strong_current"]["never"],
                           on=seen[f"{w}/seed{s}/strong_current"]["on"]) for s in SEEDS} for w in S.WORLDS}
        return S.deployment_net(pts, certs, SEEDS)["net"]

    for key in beside:
        b = beside[key]
        b["step1_C_on"] = bool(b["step1_plain_gap"] is not None and b["step1_plain_gap"] > MARGIN_C)
        b["step1_R_on"] = bool(b["step1_rank_p"] < 0.05)
        if "heldout_cal_q05" in b:
            b["heldout_cal_C_on"] = bool(b["heldout_cal_plain_gap"] is not None and b["heldout_cal_plain_gap"] > MARGIN_C)
            b["heldout_cal_R_on"] = bool(b["heldout_cal_rank_p"] < 0.05)
    realistic = {}
    for src in ("step1", "heldout_cal"):
        if all(f"{src}_q05" in beside[f"{w}/seed{s}"] for w in S.WORLDS for s in SEEDS):
            realistic[src] = {r: realistic_net(f"{src}_{r}_on") for r in ("C", "R")}
            realistic[src]["B"] = realistic_net(f"{src}_on")

    cpu_total = time.process_time() - c0 + cpu_children + smeta["cpu_seconds"]
    results = dict(
        label=LABEL, kill_rule_sha256=kr_sha, prespec_sha256=spec_sha, scores_sha256=smeta["scores_sha256"],
        settings=dict(ladder=LADDER, deciding_n=DECIDING_N, draws=DRAWS, margin_B=0.10, margin_C=MARGIN_C,
                      rank_p=0.05, draw_tags="[20261015, 9930, seed, w, n]",
                      tail_posterior_tags="[20261015, 9931, seed, w, n, j+1, 1]",
                      rest_posterior_tags="[20261015, 9931, seed, w, 2]"),
        gain_stall=gain, loss_no_stall=loss, nets=nets, net_mc_se=se, on_rate=on_rate,
        decision=dict(D_C=D["C"], D_R=D["R"], shelve_claim_posterior_beats_plain_conformal=bool(shelve_C),
                      shelve_claim_certificate_beats_rank_test=bool(shelve_R), route_shelved=route_shelved,
                      verdict=("dead end" if route_shelved else "alive")),
        reported=dict(proposer_rule=dict(B_minus_best_of_C_R=proposer, would_shelve=bool(proposer_shelve)),
                      smallest_n_net_90=smallest90, R_minus_B_at_12_30=flag, flag_R_beats_B_by_5=bool(flag_on),
                      realistic_operating_points=dict(per_seed_world=beside, nets=realistic)),
        groups=groups, checks=smeta["checks"],
        cpu_seconds=dict(prep=smeta["cpu_seconds"], run_parent=time.process_time() - c0, run_children=cpu_children,
                         total=cpu_total),
        wall_seconds_run=time.perf_counter() - t0,
        written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(S.jsonable(results), indent=1, sort_keys=True, allow_nan=False)
    out_json.write_text(text)
    (OUT / "results.json.sha256").write_text(f"{S.sha256_bytes(text.encode())}  results.json\n")
    # draw-level arrays for later inspection (write-once beside the results)
    np.savez(guard_out(OUT / "draws.npz"), **{f"{w}_{s}_n{n}_{k}": v for (w, s), out in res.items()
                                             for n, o in out.items() for k, v in o.items()})
    log("nets: " + " | ".join(f"{r}: " + ", ".join(f"n{n} {nets[r][n]:.1f}" for n in LADDER) for r in rules))
    log(f"D_C {D['C']['value']:.2f} (se {D['C']['mc_se']:.2f}) -> shelve claim C: {shelve_C}; "
        f"D_R {D['R']['value']:.2f} (se {D['R']['mc_se']:.2f}) -> shelve claim R: {shelve_R}; route shelved: "
        f"{route_shelved}")
    log(f"proposer rule shelve: {proposer_shelve}; smallest n to 90: {smallest90}; R-B at 12-30: {flag}, flag "
        f"{flag_on}; realistic: {realistic}")
    log(f"cpu total {cpu_total / 60:.1f} min; results {out_json}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    {"prep": prep, "run": run}[sys.argv[1]]()
