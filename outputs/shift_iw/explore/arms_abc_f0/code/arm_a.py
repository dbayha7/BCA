"""EXPLORATORY arm A: the credible-target pull (frozen pass marks, section 4).

TD3+BC's total pull, but no copying of a recorded command that the frozen critic rates below the actor's own command by
more than BCA's band width U_i (read only at recorded commands, where it is calibrated). Actor-only, the room check's
protocol: learners.clone (20,000 steps), then 20,000 actor-only steps against the frozen gapped host critic in 5 chunks
of 4,000, one shared stream of rows (learners.ActorOnly), batch 256, Adam 3e-4, alpha 2.5, the per-row cloning weight
of learners.actor_step (bc_w), all 12 arms side by side in one vmapped program (ActorOnly.program).

At each refresh (steps 0, 4,000, ..., 16,000), for each arm's own current actor pi:
  D_i = min_h Q_h(s_i, pi(s_i)) - min_h Q_h(s_i, a_i)                       (the frozen critic)
  guard_i: lo_k <= pi(s_i)_k <= hi_k on all four thrusters (0.5th / 99.5th percentiles of a_k; no BCA)
  released: guard_i and D_i > margin_i;  weights 0 there and n / (n - k) elsewhere (mean 1), all 0 if k = n
The arms and their margins are section 4.1's table (ARMS below). The arms never read the recording controller's flags;
only the kill rule K0 and the diagnostics do (infos/random, through IWLog.diagnostic).

Kill rules checked here: K0 at refresh 0 (binding on seeds 1-4 for medium-replay, medium, medium-expert; information on
seed 0), K1 at the step-4,000 refresh (stops A on the quality), K3 at any time (stops the seed and quality). K2's
per-seed value (A1 - none on the strong test from the 8,000-step actors) is computed here and judged by screen.py
(seed 0, recorded) and kills.py (all five seeds).

  python -B arm_a.py --quality Q --seed K [--dry]
Writes <sandbox>/repo/runs/shift_iw_v5_explore/arms_abc_f0/A/<Q>/seed<K>/{actors_8000.msgpack, actors_final.msgpack,
record.json (+ .sha256)}. No final return is computed here.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402

ARMS = ("none", "A1", "margin0", "plain_1s", "plain_2s", "plain_4s", "matched_share", "untargeted", "shuffled_band",
        "even_x1.33", "A1_IW", "A1c")
MARK_ARMS = ("none", "A1", "margin0", "plain_1s", "plain_2s", "plain_4s", "matched_share", "untargeted")
INFO_ARMS = tuple(a for a in ARMS if a not in MARK_ARMS)
STEP_8000_CHUNK = 1          # after chunk index 1 (steps 4,001-8,000)
K0_RECORDER_MAX, K0_RANDOM_MIN = 0.10, 0.15
K1_SHARE_MAX, K1_GAIN_FACTOR = 0.60, 3.0


# ------------------------------------------------------------------------------------------------ pure rules

def guard_of(pi, lo, hi):
    """The support guard: every thruster of the actor's command within [lo_k, hi_k] (float64 comparisons)."""
    pi = np.asarray(pi, np.float32).astype(np.float64)
    return np.all((pi >= np.asarray(lo, np.float64)) & (pi <= np.asarray(hi, np.float64)), axis=1)


def margin_release(D, guard, margin):
    """Released rows: guard and D_i > margin_i (a margin of +inf releases nothing; NaN never releases)."""
    return np.asarray(guard, bool) & (np.asarray(D, np.float64) > np.asarray(margin, np.float64))


def renormalized(released):
    """Weights 0 on released rows and n / (n - k) elsewhere (float64; mean exactly 1 up to float64 rounding, TD3+BC's
    total pull); every weight 0 when every row is released."""
    released = np.asarray(released, bool)
    n, k = len(released), int(released.sum())
    if k == n:
        return np.zeros(n)
    return np.where(released, 0.0, n / (n - k))


def matched_release(D, guard, k, tie):
    """The k guard-passing rows with the largest D (ties broken by the fixed permutation tie; all of them if fewer)."""
    idx = np.flatnonzero(np.asarray(guard, bool))
    out = np.zeros(len(D), bool)
    if k <= 0 or not len(idx):
        return out
    order = idx[np.lexsort((np.asarray(tie)[idx], -np.asarray(D, np.float64)[idx]))]
    out[order[:min(int(k), len(idx))]] = True
    return out


def untargeted_release(guard, k, rng):
    """k rows drawn uniformly without replacement among the guard-passing rows (all of them if fewer)."""
    idx = np.flatnonzero(np.asarray(guard, bool))
    out = np.zeros(len(guard), bool)
    take = min(int(k), len(idx))
    if take > 0:
        out[rng.choice(idx, size=take, replace=False)] = True
    return out


def a1c_weights(released, dose):
    """A1c: weight 0 where A1's rule releases, else BCA's own dose 1 + 0.5 U / (U + unit); no renormalization."""
    return np.where(np.asarray(released, bool), 0.0, np.asarray(dose, np.float64))


class Inputs:
    """Everything fixed once per seed: rows, recorded critic values, BCA's widths (unweighted, IW, shuffled), the
    readout's dose, the plain spread s, the guard bounds, the tie and shuffle permutations, and the K0 flags (read only
    by the kill rule and diagnostics)."""

    def __init__(self, obs, q_rec, U, U_iw, dose, s, lo, hi, tie, perm, seed, random_flags=None):
        self.obs, self.q_rec = np.asarray(obs, np.float32), np.asarray(q_rec, np.float64)
        self.U, self.U_iw = np.asarray(U, np.float64), np.asarray(U_iw, np.float64)
        self.dose, self.s = np.asarray(dose, np.float64), float(s)
        self.lo, self.hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        self.tie, self.perm, self.seed = np.asarray(tie), np.asarray(perm), int(seed)
        self.n = len(self.q_rec)
        self.random_flags = None if random_flags is None else np.asarray(random_flags, bool)

    def margins(self):
        return {"A1": self.U, "margin0": 0.0, "plain_1s": 1.0 * self.s, "plain_2s": 2.0 * self.s,
                "plain_4s": 4.0 * self.s, "shuffled_band": self.U[self.perm], "A1_IW": self.U_iw}


def refresh(X, pis, D, r):
    """Every arm's per-row weights at refresh r from each arm's own commands pis and gains D ({arm: array}).
    Returns (tables (12, n) float32, info per arm)."""
    n = X.n
    guard = {a: guard_of(pis[a], X.lo, X.hi) for a in ARMS}
    margins = X.margins()
    rel = {a: margin_release(D[a], guard[a], m) for a, m in margins.items()}
    k_a1 = int(rel["A1"].sum())
    rel["matched_share"] = matched_release(D["matched_share"], guard["matched_share"], k_a1, X.tie)
    rel["untargeted"] = untargeted_release(guard["untargeted"], k_a1,
                                           np.random.default_rng([C.MASTER, X.seed, C.TAG_UNTARGETED, int(r)]))
    rel["A1c"] = margin_release(D["A1c"], guard["A1c"], X.U)
    rel["none"] = np.zeros(n, bool)
    rel["even_x1.33"] = np.zeros(n, bool)
    weights = {}
    for a in ARMS:
        if a == "none":
            weights[a] = np.ones(n)
        elif a == "even_x1.33":
            weights[a] = np.full(n, 1.33)
        elif a == "A1c":
            weights[a] = a1c_weights(rel[a], X.dose)
        else:
            weights[a] = renormalized(rel[a])
    tables = np.stack([np.asarray(weights[a], np.float32) for a in ARMS])
    info = {}
    for a in ARMS:
        row = dict(released=int(rel[a].sum()), released_share=float(rel[a].mean()),
                   guard_pass_share=float(guard[a].mean()), D_mean=float(np.mean(D[a])),
                   weight_mean_float64=float(np.mean(weights[a])), weight_mean_float32=float(np.mean(tables[ARMS.index(a)],
                                                                                                    dtype=np.float64)),
                   finite=bool(np.all(np.isfinite(D[a])) and np.all(np.isfinite(pis[a]))))
        m = margins.get(a, X.U if a == "A1c" else None)
        if m is not None:
            row["guard_blocked"] = int(np.sum(~guard[a] & (np.asarray(D[a]) > np.asarray(m, np.float64))))
        if X.random_flags is not None:      # diagnostics only (the arms never read the flags)
            rf = X.random_flags
            row["released_share_recorder_rows"] = float(rel[a][~rf].mean()) if (~rf).any() else None
            row["released_share_random_rows"] = float(rel[a][rf].mean()) if rf.any() else None
            if "guard_blocked" in row:
                blocked = ~guard[a] & (np.asarray(D[a]) > np.asarray(m, np.float64))
                row["guard_blocked_recorder_rows"] = int(np.sum(blocked & ~rf))
                row["guard_blocked_random_rows"] = int(np.sum(blocked & rf))
        info[a] = row
    info["A1"]["k_A1"] = k_a1
    info["A1"]["U_mean"] = float(np.mean(X.U))
    return tables, info


def k0_check(info_a1, quality, seed):
    """K0 (change 1): at refresh 0, A1 releases at most 10% of recorder rows and at least 15% of random rows; checked
    on medium-replay, medium, medium-expert; binding on seeds 1-4, information on seed 0."""
    applies = quality in C.K0_QUALITIES
    rec_share = info_a1.get("released_share_recorder_rows")
    rnd_share = info_a1.get("released_share_random_rows")
    if not applies:
        return dict(applies=False, binding=False, passes=None, recorder_share=rec_share, random_share=rnd_share,
                    note="K0 is not checked on random (no recorder rows) or expert (no random rows)")
    ok = (rec_share is not None and rnd_share is not None and rec_share <= K0_RECORDER_MAX
          and rnd_share >= K0_RANDOM_MIN)
    return dict(applies=True, binding=int(seed) >= 1, passes=bool(ok), recorder_share=rec_share,
                random_share=rnd_share, guard_blocked_recorder_rows=info_a1.get("guard_blocked_recorder_rows"),
                guard_blocked_random_rows=info_a1.get("guard_blocked_random_rows"),
                bars=dict(recorder_at_most=K0_RECORDER_MAX, random_at_least=K0_RANDOM_MIN))


def k1_check(info_a1):
    share = info_a1["released_share"]
    gain, U = info_a1["D_mean"], info_a1["U_mean"]
    fires = bool(share > K1_SHARE_MAX or gain > K1_GAIN_FACTOR * U)
    return dict(fires=fires, released_share=share, D_mean=gain, U_mean=U,
                bars=dict(released_share_above=K1_SHARE_MAX, D_mean_above_times_U_mean=K1_GAIN_FACTOR))


# ------------------------------------------------------------------------------------------------ inputs of a seed

def iw_widths(S, band, brec):
    """A1_IW's widths: the exact strong-mix per-step table rebuilt as run.stage_band builds 'exact_strong'
    (step_masses, mass_grid with spec v5's grid ratio, threshold_tables, coverage.lookup), read with
    learners_iw.per_step_readout (scope 'band'). A refused row (R = +inf) has an infinite width: never released."""
    from experiments.shift_iw import coverage as CV
    from experiments.shift_iw import learners_iw as LI
    from experiments.shift_iw import run as R
    ratio = R.spec_per_step(R.frozen_spec(S.ctx, S.spec))["grid_ratio"]
    scores, stream = LI.bank_scores(band, S.prep.bank, S.cfg)
    m = R.step_masses(S.data, S.prep, "gap", "exact", "strong")
    grid = CV.mass_grid(np.concatenate([m["train"], m["test"]]), ratio)
    tab = LI.threshold_tables(band, scores, stream,
                              {"exact_strong": (R.bank_weights(S.data, S.prep, "exact", "strong"), grid)})["exact_strong"]
    R_iw, _ = CV.lookup(tab, m["train"])
    rd = LI.per_step_readout(band, R_iw, S.prep.train, stage="band")
    rec_tab = CV.table_from_summary(brec["tables"]["exact_strong"])
    same = bool(np.array_equal(rec_tab.grid, tab.grid) and np.array_equal(rec_tab.thresholds, tab.thresholds))
    return np.asarray(rd["width"], np.float64), dict(grid_points=int(len(grid)), grid_ratio=ratio,
                                                     refused_rows=int(np.sum(np.isinf(R_iw))),
                                                     matches_band_record_table=same, valid=bool(rd["valid"]))


def band_inputs(S):
    """The band (scope 'band'), BCA's width and dose at every recorded command with the unweighted threshold, and the
    IW widths. Returns (dict, None) or (None, K3 reason)."""
    from experiments.shift_iw import learners_iw as LI
    from experiments.shift_iw import run as R
    path = R.band_dir(S.ctx, "gap", S.seed) / "record.json"
    brec = R.finished(path, S.spec)
    if brec is None:
        raise C.Refused(f"refused: the gapped band of {S.quality} seed {S.seed} has not finished ({path})")
    if not brec.get("band_valid"):
        return None, "invalid band (band_valid false)"
    try:
        approval = R.approval_for(S.ctx, "band")
        band = LI.load_band(R.band_dir(S.ctx, "gap", S.seed) / "band.msgpack", brec["band_sha256"], S.host, S.prep,
                            S.cfg, S.seed, approval, brec["band"], stage="band")
        rd = LI.readout(band, LI.reference_of(band, brec["thresholds"]["unweighted"]), S.prep.train, stage="band")
        if not rd["valid"]:
            return None, "invalid band (the readout's inputs are not valid)"
        U_iw, iw_info = iw_widths(S, band, brec)
    except (R.Refused, LI.ApprovalError) as error:
        return None, f"a refused gate check at a band read: {error}"
    U = np.asarray(rd["width"], np.float64)
    return dict(U=U, dose=np.asarray(rd["dose"], np.float64), unit=float(band.unit), U_iw=U_iw, iw=iw_info,
                R_u=C.num(brec["thresholds"]["unweighted"]["threshold"]), band_record=path,
                support_share=float(rd["support"].mean())), None


def sibling_stop(quality, seed):
    """A rule that stops A on the whole quality (K1 on another seed, or K2 from kills.py) already fired."""
    for k in C.SEEDS:
        if k == seed:
            continue
        p = C.arm_dir("A", quality, k) / "record.json"
        rec = C.read_record(p)
        if rec is not None and rec.get("stopped_by") == "K1":
            return f"K1 fired on {quality} seed {k}"
    kills = C.read_record(C.docs() / "KILLS.json")
    if kills is not None and (kills.get("A") or {}).get(quality, {}).get("stopped_by") == "K2":
        return f"K2 fired on {quality}"
    return None


# ------------------------------------------------------------------------------------------------ the job

def run_job(quality, seed, dry=False):
    import jax
    import jax.numpy as jnp
    from experiments.shift import learners as LR
    digest = C.require_passmarks()
    started = time.perf_counter()
    d = C.arm_dir("A", quality, seed)
    try:
        S = C.Seed(quality, seed)
    except C.InputFailed as error:     # an input failed for good: the seed's numbers are missing (section 3.2)
        if dry:
            raise
        return C.write_stop(d / "record.json", C.base_record("A", quality, seed, digest), "input", str(error), started)
    rec = C.base_record("A", quality, seed, digest)

    def stop(rule, reason, **extra):
        rec.update(status="stopped", stopped_by=rule, reason=reason, seconds=time.perf_counter() - started,
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        C.write_record(d / "record.json", rec)
        C.log(f"A {quality} seed {seed}: STOPPED by {rule}: {reason}")
        return 0

    other = sibling_stop(quality, seed)
    if other and not dry:
        return stop(other.split()[0], f"not run: {other} (A stops on the whole quality)", not_run=True)
    bi, k3 = band_inputs(S)
    if dry:
        C.log(f"A {quality} seed {seed} DRY: inputs present; band "
              f"{'readable' if bi else 'NOT readable: ' + str(k3)}; nothing trained or written")
        return 0
    if bi is None:
        return stop("K3", k3)
    s, s_info = C.plain_spread(S.host, S.prep, S.cfg, seed)
    a64 = np.asarray(S.a_rec, np.float32).astype(np.float64)
    lo, hi = np.quantile(a64, 0.005, axis=0), np.quantile(a64, 0.995, axis=0)
    random_flags = np.asarray(S.data.train_log.diagnostic("infos/random"), bool)[S.prep.rows]
    clone_params, clone_rec = LR.clone(S.prep, S.cfg, seed)
    runner = LR.ActorOnly(S.prep, clone_params, S.cfg, seed, alpha=S.cfg.alpha)
    q_rec = C.q_min(S.host.critic.params, S.obs, S.a_rec)
    X = Inputs(S.obs, q_rec, bi["U"], bi["U_iw"], bi["dose"], s, lo, hi, runner.tie, runner.perms[0], seed,
               random_flags)
    rec.update(inputs=dict(n=X.n, s=s, spread=s_info, unit=bi["unit"], R_u=bi["R_u"], U_mean=float(X.U.mean()),
                           U_mean_over_s=float(X.U.mean() / s) if s > 0 else None, U_finite=bool(np.all(np.isfinite(X.U))),
                           support_share=bi["support_share"], iw=bi["iw"], guard_lo=lo, guard_hi=hi,
                           random_rows=int(random_flags.sum()), recorder_rows=int((~random_flags).sum())),
               clone=clone_rec, rows_sha256=runner.rows_sha256, start_sha256=LR.tree_sha256(clone_params),
               arms=list(ARMS), band_record_sha256=C.sha256_file(bi["band_record"]),
               inputs_sha256=S.links([S.host_record_path(), bi["band_record"]]))
    if clone_rec.get("failed"):
        return stop("K3", "the cloned start actor is not finite")
    n_arms = len(ARMS)
    program = runner.program(n_arms, True, True)
    states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_arms,) + jnp.shape(x)), runner.start)
    critic = S.host.critic.params
    history, metrics, kills = [], [], {}
    for c in range(runner.steps // runner.chunk):
        params = {a: runner.arm_params(states.params, i) for i, a in enumerate(ARMS)}
        pis = {a: C.actor_out(params[a], S.obs) for a in ARMS}
        D = {a: C.q_min(critic, S.obs, pis[a]) - q_rec for a in ARMS}
        bad = [a for a in MARK_ARMS if not (np.all(np.isfinite(pis[a])) and np.all(np.isfinite(D[a])))]
        if bad:
            return stop("K3", f"non-finite actor in arms a mark needs: {bad} at step {c * runner.chunk}",
                        history=history)
        tables, info = refresh(X, pis, D, c)
        for a in INFO_ARMS:      # an information arm that failed keeps training on even weights, marked failed
            if not info[a]["finite"]:
                tables[ARMS.index(a)] = np.ones(X.n, np.float32)
        history.append(dict(step=c * runner.chunk, refresh=c, arms=info))
        if c == 0:
            kills["K0"] = k0_check(info["A1"], quality, seed)
            if kills["K0"]["applies"] and kills["K0"]["binding"] and not kills["K0"]["passes"]:
                return stop("K0", "A1's release shares at the clone fail K0's bars (binding on seeds 1-4)",
                            history=history, kills=kills)
        if c == 1:
            kills["K1"] = k1_check(info["A1"])
            if kills["K1"]["fires"]:
                return stop("K1", "at the step-4,000 refresh A1 releases more than 60% of the kept rows or its "
                                  "critic gain exceeds 3 times the mean width (A stops on this quality)",
                            history=history, kills=kills)
        states, aux = program(critic, states, S.prep.train, runner.rows[c], jnp.asarray(tables))
        metrics.append(np.asarray(aux, np.float64).mean(1).tolist())
        C.log(f"A {quality} seed {seed}: chunk {c + 1}/5, A1 released {info['A1']['released_share']:.3f}, "
              f"K0/K1 so far {({k: v.get('passes', v.get('fires')) for k, v in kills.items()})}")
        if c == STEP_8000_CHUNK:
            p8 = {a: runner.arm_params(states.params, i) for i, a in enumerate(ARMS)}
            rec["actors_8000_sha256"] = LR.save_tree(d / "actors_8000.msgpack", p8)
            fin8 = {a: C.finite_tree(p8[a]) for a in ARMS}
            if not all(fin8[a] for a in MARK_ARMS):
                return stop("K3", f"non-finite 8,000-step actor in arms a mark needs: "
                                  f"{[a for a in MARK_ARMS if not fin8[a]]}", history=history, kills=kills)
            strong = {"strong": S.banks["strong"]}
            e_none = C.evaluate_policy(C.actor_policy(p8["none"], S.prep), strong, S.anchors)["strong"]
            e_a1 = C.evaluate_policy(C.actor_policy(p8["A1"], S.prep), strong, S.anchors)["strong"]
            kills["K2_value"] = dict(d=e_a1["points"] - e_none["points"], at_or_below_minus_3=bool(
                e_a1["points"] - e_none["points"] <= -3.0), finite=e_none["finite"] and e_a1["finite"],
                note="A1 - none, strong test points, 8,000-step actors; judged across the five seeds")
    final = {a: runner.arm_params(states.params, i) for i, a in enumerate(ARMS)}
    fin = {a: C.finite_tree(final[a]) for a in ARMS}
    rec["actors_final_sha256"] = LR.save_tree(d / "actors_final.msgpack", final)
    rec.update(history=history, chunk_metrics=metrics, metric_names=["actor_loss", "q_mean", "lambda", "bc"],
               kills=kills, finite=fin)
    if not all(fin[a] for a in MARK_ARMS):
        return stop("K3", f"non-finite final actor in arms a mark needs: {[a for a in MARK_ARMS if not fin[a]]}")
    rec.update(status="finished", stopped_by=None, seconds=time.perf_counter() - started,
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    C.write_record(d / "record.json", rec)
    C.log(f"A {quality} seed {seed}: finished in {rec['seconds']:.0f}s -> {d / 'record.json'}")
    return 0


def load_actors(quality, seed, which="final"):
    """{arm: actor params} of a finished A record (which: 'final' or '8000'), checked against the record."""
    from experiments.shift import learners as LR
    d = C.arm_dir("A", quality, seed)
    rec = C.arm_record(d / "record.json")
    key = "actors_final_sha256" if which == "final" else "actors_8000_sha256"
    return rec, LR.load_tree(d / f"actors_{which}.msgpack", {a: LR.actor_template() for a in ARMS}, rec[key])


def main(argv=None):
    p = argparse.ArgumentParser(description="EXPLORATORY arm A (credible-target pull): one quality and seed")
    p.add_argument("--quality", required=True, choices=C.QUALITIES)
    p.add_argument("--seed", type=int, required=True, choices=C.SEEDS)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args(argv)
    try:
        return run_job(a.quality, a.seed, a.dry)
    except C.Refused as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
