"""EXPLORATORY arm B: the certified throttle ceiling (frozen pass marks, section 5). Also the ceiling library arm C's
guard uses (section 6.1, step 2).

TD3+BC (the gapped host) is unchanged. When it acts, each thruster k stops at the highest 0.05 step that BCA's
certificate (wbcp.calibrate, alpha 0.5, beta 0.95, estimated strong-mix weights) says beat the step below on held-out
tries; no tries, no step. One map per seed and quality, used on all three banks.

  tries   rows of the training-law log with infos/split 1 (calibration episodes) outside the calibration bank, or
          infos/split 2 (held out); rows whose recorded command is in the seed's ball are dropped
  rungs   edges e_0..e_10 = float32(0.50, 0.55, ..., 1.00); rung j = (e_{j-1}, e_j]
  try     of rung j for thruster k: |a_k| in the rung, a_k with the same nonzero sign as the consumer's command at the
          try's own state, and |pi(s)_k| > e_{j-1}
  gain    g = r + 0.98 (1 - done) min_h Q_h(s', pi_H(s')) - min_h Q_h(s, a~), a~ the recorded command with thruster k at
          sign(a_k) e_{j-1}; the frozen host's critic and actor, no target noise
  rule    BCA: certified when wbcp.calibrate(-g, default_rng([20261016, seed, 4311, k, j, tag]), w, test mass) has a
          threshold below 0 (k = 1..4, j = 1..10; tag 0 BCA, 1 unweighted, 2 shuffled outcomes, 3 exact weights,
          4 shifted grid; arm C adds 10); the plain tests of section 5.2; a rung with no tries is refused
  walk    j = 1, 2, ... until the first refusal; t_k = e_j of the last certified rung (0.50 if rung 1 is refused,
          1.00 if all are certified)

  python -B arm_b.py --quality Q --seed K [--dry]
Writes <sandbox>/repo/runs/shift_iw_v5_explore/arms_abc_f0/B/<Q>/seed<K>/record.json (+ .sha256). Computes no return
of the host with a BCA map; on medium-expert it computes the room check K0 (host and host + knee clip, strong test),
which reads no BCA output.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402

EDGES = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00], np.float32)
SHIFTED = np.array([0.50, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975, 1.00], np.float32)
assert EDGES[6] == C.KNEE32 and not np.any(SHIFTED == C.KNEE32)
RUNG_080_085 = 7                      # rung (0.80, 0.85] of the standard grid (K1)
T085, T060 = float(EDGES[7]), float(EDGES[2])   # "0.85 or higher", "0.60 or lower": ceilings are float32 edges
ALPHA, BETA, DRAWS = 0.5, 0.95, 1000
LN40 = math.log(40.0)                 # ln(2 / delta), delta = 0.05
BCA_RULES = {"bca": 0, "bca_unweighted": 1, "shuffled": 2, "bca_exact": 3, "bca_shifted": 4}
PLAIN_TESTS = ("wmedian", "wmean", "sign", "wttest", "wbernstein")
WALK_RULES = tuple(BCA_RULES) + PLAIN_TESTS
MAPS = WALK_RULES + ("range_limit", "matched_global", "knee_clip")
DISTINCTS = dict(bca="BCA's certificate (estimated strong-mix weights)", bca_unweighted="BCA without weights",
                 shuffled="BCA's rule on outcomes shuffled within each thruster", bca_exact="BCA with exact weights",
                 bca_shifted="BCA on the shifted grid (no edge on the knee)", wmedian="weighted median",
                 wmean="weighted mean", sign="exact sign test (unweighted)", wttest="weighted t-test",
                 wbernstein="weighted empirical Bernstein", range_limit="range limit (99.5th percentile of |a_k|)",
                 matched_global="matched global ceiling", knee_clip="knee clip (float32(0.8) on every thruster)")


# ------------------------------------------------------------------------------------------------ pure rules

def rung_index(abs_a, edges=EDGES):
    """Rung j (1..J) of |a| in (e_{j-1}, e_j], 0 outside every rung (float32 comparisons)."""
    v = np.asarray(abs_a, np.float32)
    j = np.searchsorted(edges, v, side="left")
    return np.where((j >= 1) & (j <= len(edges) - 1), j, 0)


def try_mask(a_k, pi_k, j, edges=EDGES):
    """A row is a try of its rung for the consumer pi: same nonzero sign and |pi_k| > the rung's lower edge."""
    a_k = np.asarray(a_k, np.float32)
    pi_k = np.asarray(pi_k, np.float32)
    lower = edges[np.maximum(np.asarray(j) - 1, 0)]
    return (np.asarray(j) > 0) & (a_k != 0) & (np.sign(a_k) == np.sign(pi_k)) & (np.abs(pi_k) > lower)


def bca_decide(g, w, test_mass, rng):
    """BCA's certificate on one rung: certified when wbcp's threshold on the scores -g is below 0 (w None: without
    weights). Returns (certified, threshold); a rung with no tries is refused (False, None), and so is a rung whose
    tries all carry weight 0 (no weighted tries; wbcp needs positive mass). Non-finite gains or weights never reach
    here: maps_for raises NonFinite first (K3)."""
    from calibration import wbcp
    g = np.asarray(g, np.float64)
    if g.size == 0:
        return False, None
    if w is not None and not np.asarray(w, np.float64).max() > 0:
        return False, None
    if w is None:
        cal = wbcp.calibrate(-g, rng, alpha=ALPHA, beta=BETA, draws=DRAWS)
    else:
        cal = wbcp.calibrate(-g, rng, np.asarray(w, np.float64), float(test_mass), alpha=ALPHA, beta=BETA,
                             draws=DRAWS)
    return bool(cal.threshold < 0.0), float(cal.threshold)


def plain_decide(rule, g, w, B=None):
    """The plain tests of section 5.2 (no BCA). Returns (certified, statistic); an empty rung is refused."""
    g = np.asarray(g, np.float64)
    if g.size == 0:
        return False, None
    w = np.ones_like(g) if w is None else np.asarray(w, np.float64)
    sw = w.sum()
    if rule == "wmedian":
        stat = float(w[g > 0].sum() / sw) if sw > 0 else float("nan")
        return bool(w[g > 0].sum() > 0.5 * sw), stat
    if rule == "wmean":
        stat = float((w * g).sum() / sw) if sw > 0 else float("nan")
        return bool(stat > 0), stat
    if rule == "sign":
        m, n = int(np.sum(g > 0)), int(g.size)
        p = float(stats.binom.sf(m - 1, n, 0.5))          # P(X >= m), X ~ Binomial(n, 1/2)
        return bool(p <= 0.05), p
    n_eff = float(sw ** 2 / np.dot(w, w)) if sw > 0 else 0.0
    if not n_eff > 1:
        return False, None
    mean_w = float((w * g).sum() / sw)
    var_w = float((w * (g - mean_w) ** 2).sum() / sw)
    if rule == "wttest":
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = mean_w / math.sqrt(var_w / n_eff) if var_w > 0 else (math.inf if mean_w > 0 else
                                                                          (-math.inf if mean_w < 0 else math.nan))
        crit = float(stats.t.ppf(0.95, n_eff - 1))
        return bool(tstat > crit), float(tstat)
    if rule == "wbernstein":
        if B is None or not math.isfinite(B):
            raise ValueError("the empirical Bernstein test needs the thruster's range of gains B")
        lb = mean_w - math.sqrt(2.0 * var_w * LN40 / n_eff) - 7.0 * B * LN40 / (3.0 * (n_eff - 1.0))
        return bool(lb > 0), float(lb)
    raise ValueError(f"unknown plain test {rule!r}")


def walk(certified, edges):
    """The ceiling from a fixed sequence of rung decisions (j = 1..J): e_j of the last rung of the unbroken run of
    certified rungs from rung 1; e_0 if rung 1 is refused."""
    t = float(edges[0])
    for j, ok in enumerate(certified, start=1):
        if not ok:
            break
        t = float(edges[j])
    return t


def matched_global(commands, t_map):
    """One ceiling for all four thrusters: p = share of thruster entries with |a| > t_k (the given map) in the
    commands; t_g = quantile(|a|, 1 - p, method 'higher') if p > 0, else 1.00."""
    a = np.abs(np.asarray(commands, np.float64)).reshape(-1, 4)
    p = float(np.mean(a > np.asarray(t_map, np.float64)))
    if p > 0:
        return float(np.quantile(a.ravel(), 1.0 - p, method="higher")), p
    return 1.0, p


def range_limit(prep):
    """t_k = 99.5th percentile (numpy.quantile, linear) of |a_k| over the kept training rows (reads no outcomes)."""
    a = np.abs(np.asarray(prep.actions, np.float32).astype(np.float64))
    return [float(v) for v in np.quantile(a, 0.995, axis=0)]


class NonFinite(RuntimeError):
    """K3 (section 5.5): a non-finite gain, weight, test mass or consumer command; the seed and quality stop."""

    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


def finite_inputs(tries, pi, rules=None):
    """K3's reasons for one consumer before any rung is decided (wbcp.calibrate refuses non-finite scores or weights):
    the gains of every grid the rules use, the estimated weights and test mass, and the consumer's commands. The exact
    weights feed only the information rule 'bca_exact' and are checked there."""
    rules = WALK_RULES if rules is None else rules
    reasons = []
    for grid in sorted({"shifted" if r == "bca_shifted" else "standard" for r in rules}):
        for kk, (_, _, g) in enumerate(tries.gains(grid)):
            if not np.all(np.isfinite(g)):
                reasons.append(f"non-finite gain (thruster {kk + 1}, {grid} grid, {int(np.sum(~np.isfinite(g)))} tries)")
    if not np.all(np.isfinite(tries.w) & (tries.w >= 0)):
        reasons.append("non-finite or negative estimated weight")
    if not (math.isfinite(tries.test_mass) and tries.test_mass > 0):
        reasons.append(f"estimated test mass not finite and positive ({tries.test_mass})")
    if not np.all(np.isfinite(np.asarray(pi, np.float64))):
        reasons.append("non-finite consumer command at the tries' states")
    return reasons


def rung_rng(seed, k, j, tag):
    return np.random.default_rng([C.MASTER, int(seed), C.TAG_CERT, int(k), int(j), int(tag)])


def shuffle_perm(seed, k, n):
    return np.random.default_rng([C.MASTER, int(seed), C.TAG_SHUFFLE, int(k)]).permutation(int(n))


def decide_rule(rule, g, w, w_exact, tm, tm_exact, seed, k, j, tag_offset, B):
    """One rung's decision under one walk rule (g, w already restricted to the rung's tries)."""
    if rule in BCA_RULES:
        rng = rung_rng(seed, k, j, BCA_RULES[rule] + tag_offset)
        if rule == "bca_unweighted":
            return bca_decide(g, None, None, rng)
        if rule == "bca_exact":
            return bca_decide(g, w_exact, tm_exact, rng)
        return bca_decide(g, w, tm, rng)
    return plain_decide(rule, g, w, B)


def thruster_map(rule, j, g, w, w_exact, tm, tm_exact, seed, k, tag_offset, edges=EDGES, perm=None):
    """One thruster's rung decisions and ceiling under one rule. j, g, w, w_exact: the thruster's tries (all rungs)
    for the consumer; perm: the permutation of g ('shuffled'; identity reproduces BCA's map with the same tag)."""
    j = np.asarray(j)
    g = np.asarray(g, np.float64)
    if perm is not None:
        g = g[np.asarray(perm)]
    B = float(g.max() - g.min()) if g.size else math.nan
    rungs, decisions = [], []
    for r in range(1, len(edges)):
        m = j == r
        ok, stat = decide_rule(rule, g[m], None if w is None else w[m], None if w_exact is None else w_exact[m],
                               tm, tm_exact, seed, k, r, tag_offset, B)
        decisions.append(ok)
        rungs.append(dict(rung=r, lower=float(edges[r - 1]), upper=float(edges[r]), tries=int(m.sum()),
                          certified=ok, statistic=stat))
    return walk(decisions, edges), rungs


# ------------------------------------------------------------------------------------------------ tries

class Tries:
    """One seed's held-out tries with everything that does not depend on the consumer: the rows, their transitions
    (the host's normalisation), the host's continuation value, the estimated and exact strong-mix weights and test
    masses, and per grid and thruster the candidate rows, their rungs and gains."""

    def __init__(self, S):
        from experiments.shift_iw import classifier as CL
        from experiments.shift_iw import learners_iw as LI
        from experiments.shift_iw import run as R
        from experiments.shift_iw import world as W
        log, data = S.data.train_log, S.data
        split = np.asarray(log["infos/split"])
        cal = np.flatnonzero(split == 1)
        cal = cal[~np.isin(cal, log.bank_rows)]
        rows = np.union1d(cal, np.flatnonzero(split == 2))
        if C.smoke_settings():
            rows = rows[:3000]
        acts = np.asarray(log["actions"][rows])
        self.dropped_in_ball = int(np.sum(data.gap.contains(acts)))
        self.rows = rows[~data.gap.contains(acts)]
        self.n = int(len(self.rows))
        trans = LI.transitions(log, self.rows, S.prep)
        self.obs = np.asarray(trans.obs, np.float32)
        self.a32 = np.asarray(trans.action, np.float32)
        host = S.host
        pi_next = C.actor_out(host.actor.params, np.asarray(trans.next_obs, np.float32))
        q_next = C.q_min(host.critic.params, np.asarray(trans.next_obs, np.float32), pi_next)
        r = np.asarray(trans.reward, np.float32).astype(np.float64)
        done = np.asarray(trans.done, np.float32).astype(np.float64)
        gamma = float(S.cfg.gamma)
        self.cont = r + gamma * (1.0 - done) * q_next
        self.critic = host.critic.params
        clf = R.classifiers_of(data)["strong"]
        raw = np.asarray(log["observations"][self.rows], np.float64)
        self.w = np.asarray(clf.weight(raw), np.float64)
        bank_obs = np.asarray(log["observations"][S.prep.bank_rows])
        self.test_mass = float(CL.estimated_arm(clf, bank_obs, data.samples["strong"]["observations"],
                                                data.samples["strong"]["actions"], data.gap)["test_mass"])
        self.w_exact = np.asarray(W.ratio(log.current[self.rows], "strong"), np.float64)
        self.test_mass_exact = float(data.test_mass("strong", gapped=True))
        self.seed = S.seed
        self._gains = {}

    def gains(self, grid):
        """Per thruster k (0-based): (candidate try indices, rung j, gain g) on a grid ('standard' or 'shifted'),
        computed once (they do not depend on the consumer)."""
        if grid not in self._gains:
            edges = EDGES if grid == "standard" else SHIFTED
            out = []
            for kk in range(4):
                j = rung_index(np.abs(self.a32[:, kk]), edges)
                cand = np.flatnonzero(j > 0)
                at = self.a32[cand].copy()
                at[:, kk] = (np.sign(at[:, kk]) * edges[j[cand] - 1]).astype(np.float32)
                q = C.q_min(self.critic, self.obs[cand], at)
                out.append((cand, j[cand], self.cont[cand] - q))
            self._gains[grid] = out
        return self._gains[grid]

    def describe(self):
        return dict(rows=self.n, dropped_in_ball=self.dropped_in_ball, test_mass_estimated=self.test_mass,
                    test_mass_exact=self.test_mass_exact, weights_finite=bool(np.all(np.isfinite(self.w))),
                    weight_range=[float(self.w.min()), float(self.w.max())] if self.n else None,
                    rows_sha256=hashlib.sha256(np.ascontiguousarray(self.rows).tobytes()).hexdigest())


def maps_for(tries, pi, tag_offset, commands=None, range_t=None, rules=WALK_RULES, perm_fn=shuffle_perm):
    """Every ceiling map of one consumer (pi: its float32 commands at the tries' states): the walk rules (the BCA
    variants with tag offset tag_offset: 0 for arm B, 10 for arm C), the range limit, the matched global ceiling (on
    the consumer's own unclipped strong-start commands) and the knee clip. Returns (maps {name: [t_1..t_4]},
    details)."""
    pi = np.asarray(pi, np.float32)
    seed = tries.seed
    bad = finite_inputs(tries, pi, rules)
    if bad:
        raise NonFinite(bad)
    maps, details = {}, dict(rules={}, finite={}, skipped={})
    finite = True
    exact_ok = bool(np.all(np.isfinite(tries.w_exact) & (tries.w_exact >= 0)) and math.isfinite(tries.test_mass_exact)
                    and tries.test_mass_exact > 0)
    for rule in rules:
        if rule == "bca_exact" and not exact_ok:      # information only: recorded as not built, never a crash
            details["skipped"][rule] = "the exact weights or exact test mass are not finite and positive"
            continue
        grid = "shifted" if rule == "bca_shifted" else "standard"
        edges = SHIFTED if grid == "shifted" else EDGES
        ts, per = [], []
        for kk in range(4):
            cand, j, g = tries.gains(grid)[kk]
            sel = try_mask(tries.a32[cand, kk], pi[cand, kk], j, edges)
            idx, jj, gg = cand[sel], j[sel], g[sel]
            finite &= bool(np.all(np.isfinite(gg)))
            perm = perm_fn(seed, kk + 1, len(gg)) if rule == "shuffled" else None
            t, rungs = thruster_map(rule, jj, gg, tries.w[idx], tries.w_exact[idx], tries.test_mass,
                                    tries.test_mass_exact, seed, kk + 1, tag_offset, edges, perm)
            ts.append(t)
            per.append(dict(thruster=kk + 1, ceiling=t, tries=int(len(gg)), rungs=rungs,
                            gain_range=[float(gg.min()), float(gg.max())] if len(gg) else None))
        maps[rule] = ts
        details["rules"][rule] = per
    details["finite"] = dict(gains=finite, weights=bool(np.all(np.isfinite(tries.w))),
                             consumer=bool(np.all(np.isfinite(pi))))
    if range_t is not None:
        maps["range_limit"] = [float(v) for v in range_t]
    if commands is not None and "bca" in maps:
        tg, p = matched_global(commands, maps["bca"])
        maps["matched_global"] = [tg] * 4
        details["matched_global"] = dict(p=p, t_g=tg, entries=int(np.asarray(commands).size))
    maps["knee_clip"] = [float(C.KNEE32)] * 4
    return maps, details


def k1_numbers(details, maps, rule="bca"):
    """K1's inputs: the thruster-1 ceiling and the tries of rung (0.80, 0.85] for thruster 1."""
    t1 = maps[rule][0]
    tries = details["rules"][rule][0]["rungs"][RUNG_080_085 - 1]["tries"]
    return dict(t1=t1, tries_080_085=tries, fires=bool(t1 >= T085 and tries >= 30), over_refusal=bool(t1 <= T060))


# ------------------------------------------------------------------------------------------------ the job

def run_job(quality, seed, dry=False):
    digest = C.require_passmarks()
    started = time.perf_counter()
    path = C.arm_dir("B", quality, seed) / "record.json"
    try:
        S = C.Seed(quality, seed)
    except C.InputFailed as error:     # the host is B's actor: a non-finite actor is K3 (section 5.5)
        if dry:
            raise
        return C.write_stop(path, C.base_record("B", quality, seed, digest), "K3", str(error), started,
                            kills=dict(K3=dict(fires=True, reasons=[str(error)])))
    rec = C.base_record("B", quality, seed, digest)
    tries = Tries(S)
    rec["tries"] = tries.describe()
    if dry:
        C.log(f"B {quality} seed {seed} DRY: inputs present; {tries.n} tries; nothing computed or written")
        return 0
    host = S.host
    pi = C.actor_out(host.actor.params, tries.obs)
    pol = C.actor_policy(host.actor.params, S.prep, name=f"{quality} seed {seed} gapped host (float64 copy)")
    cmds = C.strong_commands(pol, S.banks)          # the host's own unclipped strong-start commands
    finite_cmds = bool(np.all(np.isfinite(cmds)))
    try:
        maps, details = maps_for(tries, pi, 0, commands=cmds, range_t=range_limit(S.prep))
    except NonFinite as error:
        return C.write_stop(path, rec, "K3", f"non-finite values: {error}", started,
                            kills=dict(K3=dict(fires=True, reasons=error.reasons)))
    k1 = k1_numbers(details, maps)
    k3_reasons = [k for k, v in dict(details["finite"], actor=C.finite_tree(host.actor.params),
                                     commands=finite_cmds).items() if not v]
    k4 = C.past_share(cmds, maps["bca"], thruster=0)
    rec.update(status="finished", stopped_by=None, maps=maps, map_names=DISTINCTS, details=details,
               kills=dict(K1=k1, K2_over_refusal=k1["over_refusal"], K3=dict(fires=bool(k3_reasons),
                                                                              reasons=k3_reasons),
                          K4_clip_share_thruster1=k4,
                          K4_clip_share_all=C.past_share(cmds, maps["bca"])),
               host_record_sha256=C.sha256_file(S.host_record_path()),
               inputs_sha256=S.links([S.host_record_path()]))
    if quality == "medium-expert":   # K0, the room check: reads no BCA output
        strong = {"strong": S.banks["strong"]}
        h = C.evaluate_policy(pol, strong, S.anchors)["strong"]
        kc = C.evaluate_policy(C.ceiling_policy(pol, maps["knee_clip"]), strong, S.anchors)["strong"]
        rec["kills"]["K0_room"] = dict(host_points=h["points"], knee_clip_points=kc["points"],
                                       room=kc["points"] - h["points"], finite=h["finite"] and kc["finite"],
                                       fires_at_seed=bool(kc["points"] - h["points"] < 1.0))
    rec["seconds"] = time.perf_counter() - started
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    d = C.arm_dir("B", quality, seed)
    C.write_record(d / "record.json", rec)
    C.log(f"B {quality} seed {seed}: BCA map {maps['bca']}, K1 {k1}, K4 share {k4:.4f}; "
          f"{rec['seconds']:.0f}s -> {d / 'record.json'}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="EXPLORATORY arm B (certified throttle ceiling): one quality and seed")
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
