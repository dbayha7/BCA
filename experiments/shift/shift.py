"""The controlled shift: which training rows are removed, the frozen row pools, and the per-state shift card.

Removal (dial 1), applied to the training split and the calibration bank, never to the evaluation split:
  'none'    the zero-shift control: nothing removed.
  'gap'     the D4RL rule (primary): experiments/truth/gap.find_gap unchanged with gap.DESIGN, except
            search.score_rows = None so all 50,000 training rows are scored; float64 of the stored float32 training
            actions; the densest 15% (inclusive ball, mass band [0.14, 0.16], radius rho >= 0.05, gap.py's tie and
            next-centre fallbacks). Every training row whose command is in the ball leaves training; a removed row's
            true next observation stays in the previous row, rows are never spliced. Gap rows are dropped from the
            calibration bank too.
  'random'  the same number of training rows drawn uniformly by default_rng([20261012, rep, 99]), independent of the
            commands (and, for the bank, the same number of bank rows by default_rng([20261012, rep, 99, 1])).
The competing-ball margin: the smallest 15% ball centred on one of gap.py's 4,096 candidate training commands lying
more than 2 gap radii from the gap centre, divided by the gap radius (T3 asks >= 1.2).

Removed share m(s): the probability that the recording controller's command at state s (rounded to float32) falls in
the removed ball, by Monte Carlo over its wobble noise and coin (4,096 draws: SE <= 0.008). 'none' gives 0
everywhere; 'random' gives the removed fraction (0.15) everywhere.

Frozen pools (make_pools), from the data before any training; the same rows serve every critic of a seed:
  removed  3,000 removed training rows at their logged commands;
  region   3,000 kept rows with the water's push above the trigger (the full-throttle region), each scored at
           c_hi (the full-throttle pattern) and c_cap (the same pattern capped at 80%);
  far      3,000 kept rows with m(s) <= 0.01 whose logged command lies more than 2 gap radii from the gap centre.

The shift card (shift_card), for one (log, policy, situation); a visited state is a state of the 100-step rollout
from each of the situation's 1,000 bank starts:
  1 removed share m(s) at visited states (Monte Carlo on a seeded subsample);
  2 unrecorded-throttle share: steps commanding past 80% on any thruster;
  3 full-throttle-region share: visited states with push above the trigger;
  4 outside the recorded conditions: current > 0.7, or wave amplitude > 0.4, or floats beyond the training 99th
    percentile of the standardised float radius;
  5 inside the removed ball, split into past 80% (not inferable from the gapped data) and at or below 80%;
  6 distance to the data: the 10th-nearest kept training row in standardised (state, command) space, beside the same
    for held-out evaluation-split rows at their logged commands;
  7 covered share: of the unrecorded-throttle steps, the share inside the recorded conditions;
  8 loss from unrecorded throttle: J(the same policy with commands capped at 80% at evaluation) - J(policy), exact.
"""

import copy
import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from experiments.shift import dataset as DS
from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC
from experiments.truth import gap as GAP

GAP_Q = 0.15
TAG_RANDOM, TAG_POOLS, TAG_MS, TAG_CARD = 99, 12, 13, 14
POOL_SIZE = 3000
MS_DRAWS = 4096
FAR_MS_MAX = 0.01
FAR_RADII = 2.0
MARGIN_CANDIDATES = 4096
REMOVALS = ("none", "gap", "random")


def gap_settings(q=GAP_Q):
    """gap.DESIGN with every training row scored (score_rows None); another q uses the band q +- 0.01."""
    s = copy.deepcopy(GAP.DESIGN)
    s["search"]["score_rows"] = None
    if q != GAP_Q:
        s["q"] = float(q)
        s["band"] = [float(q) - 0.01, float(q) + 0.01]
    return s


@dataclass(frozen=True)
class Gap:
    """The removed ball: every command a with ||a - centre|| <= rho (float64 of the command; inclusive).
    mass: its share of the training rows; rule: gap.py's radius rule ('base' or 'tie_block_fallback');
    competing_ratio: the best competing 15% ball's radius over rho; record_sha256: hash of gap.find_gap's record."""

    centre: tuple
    rho: float
    mass: float
    rule: str
    q: float
    competing_ratio: float
    record_sha256: str

    def contains(self, actions):
        """Per row: is the command inside the ball? float32 commands are compared at their exact float64 value."""
        return GAP.in_gap(np.asarray(actions).astype(np.float64), np.asarray(self.centre), self.rho)

    def as_dict(self):
        return asdict(self)


def _canonical_sha(value):
    return GAP.canonical_sha256(value)


def competing_ratio(actions, centre, rho, q=GAP_Q):
    """The smallest q-ball centred on one of gap.py's candidate training commands (default_rng([20261012, 1]),
    4,096 rows) farther than 2 rho from the centre, over rho."""
    a = np.asarray(actions, np.float64)
    pick = np.random.default_rng([GAP.DESIGN["search"]["seed"], 1]).choice(len(a), min(MARGIN_CANDIDATES, len(a)),
                                                                           replace=False)
    cand = a[pick]
    far = cand[GAP.distance(cand, centre) > FAR_RADII * rho]
    if not len(far):
        return math.inf
    return float(GAP.rho_q(a, far, q).min() / rho)


def define_gap(log, q=GAP_Q):
    """The gap of a log: gap.find_gap on the float64 of its float32 training commands. Returns (Gap, record); Gap is
    None (with record['reason']) when gap.py refuses (no centre in the band, or the point-mass guard)."""
    train = log.split_rows("train")
    a = np.asarray(log["actions"][train], np.float32).astype(np.float64)
    record = GAP.find_gap(a, gap_settings(q))
    if not record["defined"]:
        return None, record
    centre = np.asarray(record["centre"], np.float64)
    gap = Gap(centre=tuple(float(v) for v in centre), rho=float(record["rho"]), mass=float(record["mass_train"]),
              rule=record["rule"], q=float(q), competing_ratio=competing_ratio(a, centre, record["rho"], q),
              record_sha256=_canonical_sha(record))
    return gap, record


@dataclass
class Removal:
    """One removal applied to a log. removed: bool per log row (True only on training rows); bank_removed: bool per
    calibration-bank row. kind: 'none', 'gap' or 'random'; gap: the Gap (for 'gap'; for 'random' the gap whose count
    it matches)."""

    kind: str
    removed: np.ndarray
    bank_removed: np.ndarray
    gap: object = None

    def kept_train_rows(self, log):
        train = log.split_rows("train")
        return train[~self.removed[train]]

    def removed_train_rows(self, log):
        train = log.split_rows("train")
        return train[self.removed[train]]

    def bank_rows(self, log):
        return log.bank_rows[~self.bank_removed]

    @property
    def count(self):
        return int(self.removed.sum())

    def rate(self, log):
        return self.count / len(log.split_rows("train"))

    def summary(self, log):
        return dict(kind=self.kind, removed_training_rows=self.count, removed_share=self.rate(log),
                    kept_training_rows=int(len(self.kept_train_rows(log))), bank_rows=int((~self.bank_removed).sum()),
                    bank_removed=int(self.bank_removed.sum()),
                    gap=None if self.gap is None else self.gap.as_dict(),
                    kept_rows_sha256=DS.sha256_array(self.kept_train_rows(log)),
                    bank_rows_sha256=DS.sha256_array(self.bank_rows(log)))


def make_removal(log, kind, gap=None):
    """A Removal of kind 'none', 'gap' (needs gap) or 'random' (needs gap: it removes the same count). Never touches
    the evaluation split."""
    n = log.n_rows
    train = log.split_rows("train")
    bank = log.bank_rows
    removed = np.zeros(n, bool)
    bank_removed = np.zeros(len(bank), bool)
    if kind == "none":
        return Removal("none", removed, bank_removed, None)
    if gap is None:
        raise ValueError(f"removal {kind!r} needs the gap")
    in_ball = gap.contains(log["actions"])
    if kind == "gap":
        removed[train] = in_ball[train]
        bank_removed = in_ball[bank]
    elif kind == "random":
        rep = log.meta["rep"]
        count = int(in_ball[train].sum())
        pick = np.random.default_rng([TC.DATA_SEED, rep, TAG_RANDOM]).choice(len(train), count, replace=False)
        removed[train[pick]] = True
        bank_count = int(in_ball[bank].sum())
        bpick = np.random.default_rng([TC.DATA_SEED, rep, TAG_RANDOM, 1]).choice(len(bank), bank_count, replace=False)
        bank_removed[bpick] = True
    else:
        raise ValueError(f"unknown removal {kind!r}; one of {', '.join(REMOVALS)}")
    if removed[log.split_rows("evaluation")].any() or removed[log.split_rows("calibration")].any():
        raise AssertionError("removal must touch training rows only")
    return Removal(kind, removed, bank_removed, gap)


def leak_count(rows, log, gap):
    """How many of the given rows have a command inside the removed ball (0 for a correctly gapped row set)."""
    rows = np.asarray(rows, np.int64)
    return int(gap.contains(log["actions"][rows]).sum())


# ------------------------------------------------------------------------------------------------ m(s)

def removed_share(states, removal, recorder, draws=MS_DRAWS, seed=(TC.DATA_SEED, TAG_MS), draw_chunk=256,
                  row_chunk=2048, rate=None):
    """m(s) at each state (module docstring) and its Monte Carlo SE. removal: a Removal, or the strings 'none' /
    'random' (then `rate`, the removed fraction, is needed for 'random')."""
    x = np.asarray(states, np.float64)
    n = len(x)
    kind = removal if isinstance(removal, str) else removal.kind
    if kind == "none":
        return np.zeros(n), np.zeros(n)
    if kind == "random":
        if rate is None:
            raise ValueError("random removal: pass rate (the removed fraction of training rows)")
        return np.full(n, float(rate)), np.zeros(n)
    gap = removal.gap
    centre, rho = np.asarray(gap.centre), gap.rho
    hits = np.zeros(n)
    rng = np.random.default_rng(list(seed))
    for d0 in range(0, draws, draw_chunk):
        d = min(draw_chunk, draws - d0)
        noise = rng.standard_normal((d, n, TC.DA))
        coin = rng.uniform(size=(d, n))
        for r0 in range(0, n, row_chunk):
            sl = slice(r0, min(n, r0 + row_chunk))
            cmd = recorder(x[None, sl], noise[:, sl], coin[:, sl]).astype(np.float32).astype(np.float64)
            hits[sl] += (np.sqrt(np.sum((cmd - centre) ** 2, axis=-1)) <= rho).sum(0)
    m = hits / draws
    return m, np.sqrt(np.maximum(m * (1 - m), 0.0) / draws)


# ------------------------------------------------------------------------------------------------ pools

def _pool_sha(rows, *extra):
    h = hashlib.sha256()
    for v in (rows,) + extra:
        h.update(DS.sha256_array(v).encode())
    return h.hexdigest()


def make_pools(log, gap, recorder, size=POOL_SIZE, ms_draws=MS_DRAWS, candidate_chunk=1000):
    """The three frozen pools (module docstring) as ascending log-row indices, with c_hi / c_cap for the region pool,
    m(s) of the far rows and a SHA-256 per pool. Seeds: default_rng([20261012, rep, 12, k]) for k = 0, 1, 2 (row
    choice) and [20261012, rep, 13, chunk] for m(s)."""
    rep = log.meta["rep"]
    train = log.split_rows("train")
    a = log["actions"][train]
    x = log["infos/state"][train]
    ing = gap.contains(a)
    rng = lambda k: np.random.default_rng([TC.DATA_SEED, rep, TAG_POOLS, k])
    gap_rows = train[ing]
    removed = np.sort(gap_rows[rng(0).choice(len(gap_rows), min(size, len(gap_rows)), replace=False)])
    region_rows = train[~ing & recorder.in_region(x)]
    region = np.sort(region_rows[rng(1).choice(len(region_rows), min(size, len(region_rows)), replace=False)])
    c_hi = np.asarray(recorder.pattern, np.float64)
    c_cap = np.clip(c_hi, -0.8, 0.8)
    far_cand = train[~ing & (GAP.distance(np.asarray(a, np.float64), np.asarray(gap.centre)) > FAR_RADII * gap.rho)]
    order = far_cand[rng(2).permutation(len(far_cand))]
    removal = Removal("gap", np.zeros(log.n_rows, bool), np.zeros(0, bool), gap)
    far, far_m = [], []
    for c0 in range(0, len(order), candidate_chunk):
        rows = order[c0:c0 + candidate_chunk]
        m, _ = removed_share(log["infos/state"][rows], removal, recorder, draws=ms_draws,
                             seed=(TC.DATA_SEED, rep, TAG_MS, c0 // candidate_chunk))
        ok = m <= FAR_MS_MAX
        far.extend(rows[ok].tolist())
        far_m.extend(m[ok].tolist())
        if len(far) >= size:
            break
    far, far_m = np.asarray(far[:size], np.int64), np.asarray(far_m[:size])
    o = np.argsort(far)
    far, far_m = far[o], far_m[o]
    return dict(removed=removed, region=region, far=far, c_hi=c_hi, c_cap=c_cap, far_removed_share=far_m,
                sha256=dict(removed=_pool_sha(removed), region=_pool_sha(region, c_hi, c_cap), far=_pool_sha(far)),
                rule=dict(size=size, far_removed_share_max=FAR_MS_MAX, far_gap_radii=FAR_RADII, ms_draws=ms_draws))


# ------------------------------------------------------------------------------------------------ the card

@dataclass
class CardReference:
    """What the shift card compares against, from one log and removal: the kept training rows' float mean / SD and
    99th percentile of the standardised float radius; the standardisation and k-d tree of kept training rows in
    (state, command) space; the held-out rows' 10th-neighbour distances; the recorder, removal and gap."""

    removal: Removal
    recorder: object
    float_mean: np.ndarray
    float_sd: np.ndarray
    float_radius_p99: float
    xa_mean: np.ndarray
    xa_sd: np.ndarray
    tree: object
    heldout_distance: np.ndarray
    removed_rate: float
    k: int = 10


def card_reference(log, removal, recorder, heldout_rows=5000, k=10):
    from scipy.spatial import cKDTree

    kept = removal.kept_train_rows(log)
    xk = log["infos/state"][kept]
    ak = np.asarray(log["actions"][kept], np.float64)
    fm, fs = xk[:, :TC.DS].mean(0), xk[:, :TC.DS].std(0)
    r99 = float(np.percentile(np.sqrt(np.sum(((xk[:, :TC.DS] - fm) / fs) ** 2, axis=1)), 99))
    xa = np.concatenate([xk, ak], 1)
    mu, sd = xa.mean(0), xa.std(0)
    tree = cKDTree((xa - mu) / sd)
    ev = log.split_rows("evaluation")
    pick = np.sort(np.random.default_rng([TC.DATA_SEED, log.meta["rep"], TAG_CARD]).choice(
        len(ev), min(heldout_rows, len(ev)), replace=False))
    q = np.concatenate([log["infos/state"][ev[pick]], np.asarray(log["actions"][ev[pick]], np.float64)], 1)
    dho = tree.query((q - mu) / sd, k=k)[0][:, -1]
    return CardReference(removal, recorder, fm, fs, r99, mu, sd, tree, dho, removal.rate(log), k)


def _quantiles(v, qs=(0.1, 0.5, 0.9, 0.99)):
    v = np.asarray(v, np.float64)
    return {f"p{int(round(100 * q))}": float(np.quantile(v, q)) for q in qs} if len(v) else None


def shift_card(policy, situation, physics, ref, anchors=None, ms_states=2000, knn_states=5000,
               seed=(TC.DATA_SEED, TAG_CARD)):
    """Measures 1-8 (module docstring) for a policy (a recorder.Controller; the recording controller on a bank is a
    recorder.StreamedRecorder) on a situation's bank. anchors: optional dict(operator=J, top=J) to report points."""
    from experiments.shift import truth as TR

    x0 = TC.bank_starts(situation)
    run = TR.rollout(policy, x0, physics, record=True)
    capped = TR.rollout(RC.CappedController(policy), x0, physics)
    xs = run.states[:, :-1].reshape(-1, TC.DX)
    cs = run.commands.reshape(-1, TC.DA)
    over = TC.past_knee(cs, physics)
    region = ref.recorder.in_region(xs)
    outside, by_c, by_w, by_f = TC.outside_recorded_conditions(xs, ref.float_mean, ref.float_sd,
                                                              ref.float_radius_p99)
    gap = ref.removal.gap
    in_ball = gap.contains(cs) if gap is not None else np.zeros(len(cs), bool)
    rng = np.random.default_rng(list(seed))
    ms_pick = np.sort(rng.choice(len(xs), min(ms_states, len(xs)), replace=False))
    m, m_se = removed_share(xs[ms_pick], ref.removal, ref.recorder, rate=ref.removed_rate)
    kn_pick = np.sort(rng.choice(len(xs), min(knn_states, len(xs)), replace=False))
    q = np.concatenate([xs[kn_pick], cs[kn_pick]], 1)
    dist = ref.tree.query((q - ref.xa_mean) / ref.xa_sd, k=ref.k)[0][:, -1]
    loss = capped.J - run.J
    out = dict(
        situation=situation, policy=getattr(policy, "name", str(policy)), world="stall" if physics.stall else "no_stall",
        removal=ref.removal.kind, starts=int(len(x0)), visited_states=int(len(xs)),
        removed_share=dict(mean=float(m.mean()), quantiles=_quantiles(m), states=int(len(m)),
                           max_se=float(m_se.max()) if len(m_se) else 0.0),
        unrecorded_throttle_share=float(over.mean()),
        full_throttle_region_share=float(region.mean()),
        outside_recorded_conditions=dict(share=float(outside.mean()), by_current=float(by_c.mean()),
                                         by_wave=float(by_w.mean()), by_floats=float(by_f.mean()),
                                         float_radius_p99=ref.float_radius_p99),
        inside_removed_ball=dict(share=float(in_ball.mean()), past_80=float(np.mean(in_ball & over)),
                                 at_or_below_80=float(np.mean(in_ball & ~over))),
        distance_to_data=dict(visited=_quantiles(dist, (0.5, 0.9)), heldout_log_rows=_quantiles(
            ref.heldout_distance, (0.5, 0.9)), k=ref.k, space="standardised (state, command)"),
        covered_share=float(np.mean(~outside[over])) if over.any() else None,
        loss_from_unrecorded_throttle=dict(J=float(loss.mean()), se=float(loss.std() / math.sqrt(len(loss)))),
        J=dict(policy=float(run.J.mean()), capped=float(capped.J.mean()), policy_se=float(run.J.std() / math.sqrt(
            len(run.J)))),
    )
    if anchors is not None:
        scale = 100.0 / (anchors["top"] - anchors["operator"])
        out["points"] = dict(policy=float((run.J.mean() - anchors["operator"]) * scale),
                             capped=float((capped.J.mean() - anchors["operator"]) * scale),
                             loss_from_unrecorded_throttle=float(loss.mean() * scale))
    return out


# ------------------------------------------------------------------------------------------------ D4RL note

def density_check_note(log, gap, query_rows=5000, gap_rows=3000):
    """The D4RL protocol's density check (gap.g0b_counts / g0b_passes, as for Hopper) on this log, reported only:
    states standardised by the training rows (eps 1e-3, as the runtime does), x = (standardised state, command);
    r and r_s are the median 10th-neighbour distances of seeded withheld rows (calibration + evaluation) to the
    training rows; counts over up to 3,000 seeded gap rows of the training split."""
    from scipy.spatial import cKDTree

    rep = log.meta["rep"]
    x = log["infos/state"]
    a = np.asarray(log["actions"], np.float64)
    ep = log["infos/episode"].astype(np.int64)
    train = log["infos/split"] == 0
    held = ~train
    ss = (x - x[train].mean(0)) / (x[train].std(0) + 1e-3)
    xsa = np.concatenate([ss, a], 1)
    q = np.random.default_rng([TC.DATA_SEED, rep, TAG_CARD, 1]).choice(np.flatnonzero(held), query_rows, replace=False)
    r = float(np.median(cKDTree(xsa[train]).query(xsa[q], k=10)[0][:, -1]))
    r_s = float(np.median(cKDTree(ss[train]).query(ss[q], k=10)[0][:, -1]))
    ing = gap.contains(log["actions"])
    kept = train & ~ing
    gq = np.random.default_rng([TC.DATA_SEED, rep, TAG_CARD, 2]).choice(np.flatnonzero(train & ing),
                                                                       min(gap_rows, int((train & ing).sum())),
                                                                       replace=False)
    values, _ = GAP.g0b_counts(xsa[gq], ss[gq], ep[gq], GAP.ArrayThinness(xsa[train], ep[train]),
                               GAP.ArrayThinness(xsa[kept], ep[kept]), GAP.ArrayThinness(ss[train], ep[train]),
                               GAP.ArrayThinness(ss[kept], ep[kept]), r, r_s, 2)
    values["match_rate"] = None
    passes = GAP.g0b_passes(values, GAP.DESIGN["g0_marks"])
    return dict(values={k: v for k, v in values.items() if not k.startswith("nn")}, passes=passes, r=r, r_s=r_s,
                r_over_rho=r / gap.rho, note="reported only: the D4RL density check's count marks (ii)-(iv) are "
                "geometrically hard to reach when the 10-neighbour radius in (state, command) space is several gap "
                "radii; check this on Hopper's data before GPU time")
