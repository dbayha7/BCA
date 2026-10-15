"""EXPLORATORY trace test bed: the detection readout with data-only stand-ins (plan sections 4.4 and 4.6). NO BCA:
no band, calibrator, threshold, dose or BCA update is called; every score here is a stand-in computed from the log,
the frozen fixed-controller critic and the cloned actor. The detect component of Stage 1 was not ready when this was
written (scratchpad/ood/detect/ did not exist), so these are this test bed's own implementations, written to the
plan's definitions.

One-step error at training row i, with the log's frozen critic (spec v2, review finding 1): the critic's own training
form,
  delta_i = r_i + 0.98 (1 - done_i) * mean_d min_h Q_h(s'_i, clip(c(s'_i) + eps_id, -1, 1)) - min_h Q_h(s_i, a_i),
recorded r_i and s'_i, c the stall-aware controller as the critic's target reads it (learners.FixedController on the
normalised next observation), eps_id = clip(0.2 z_id, -0.5, 0.5) with z_id standard normal: the training noise of
td3_bc_update (policy noise 0.2, clipped at 0.5, then the command clipped to +-1), averaged over 32 fixed draws.
Draw d of log row i is row i of default_rng([20261015, seed, 94, d + 1]).standard_normal((log rows, 4)), so a row
gets the same draws in both worlds and at every share. The critic was trained toward exactly this target (its
target copy in place of its final parameters), so at a well-fitted row delta is about 0 whatever the critic's shape
along the command at s'.
Why not the noiseless next command (spec v1, the room test's form): then delta_noiseless_i = delta_i + 0.98 J(s'_i),
J(s') = min_h Q(s', c(s')) - mean_d min_h Q(s', clip(c(s') + eps_d)). J is not a constant: at seed 0 it is +0.27 at
region rows against +0.13 elsewhere for a critic that has learned the stall (the full-log critic, stalling world)
and +0.12 / +0.11 in the no-stall world, because where the stall-aware controller sits at its 80% cap the noise
reaches past-80% commands, whose value a stall-aware critic drops. That offset lowers residual evidence at region
rows in the stalling world only, against the separation Q2 looks for, and T4 (84% outside the region) cannot cancel
it. The noiseless form is kept as a reported signal (residual_evidence_noiseless); 0.98 (1 - done) J is reported per
row group and at every pair (neighbour_offset): the J-by-region diagnostic of both worlds.

Stand-ins, each a score at a (state, command) pair (s, a); larger means "more suspect":
  residual_evidence  PRIMARY. Minus the mean one-step error (training form, above) of the 10 nearest training rows of
                     the log, in standardised (state, command) space (the gate's distance-to-data space:
                     shift.card_reference's columns, standardised by the log's kept training rows). A critic that
                     overrates the commands near (s, a) leaves negative errors there, so the score is high.
  residual_evidence_noiseless  reported: the same with the noiseless next command (spec v1's form).
  distance           the distance to the 10th-nearest training row in the same space (the gate's distance to data).
  range_hinge        sum_j ([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2), lo_j and hi_j the 0.5% and 99.5% quantiles of the
                     log's training commands per thruster (the recorded range; the data-only rival with no signal).
  density            behaviour density: 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2, c the log's cloned actor, sigma_k the
                     maximum-likelihood spread (root mean square of a_ik - c_k(s_i) over the log's training rows).
  heads              two-head disagreement |Q1 - Q2|(s, a) of the log's frozen critic.
Also at every pair, reported: neighbour_offset (the mean of 0.98 (1 - done) J over the same 10 nearest rows; the
noiseless score is residual_evidence - neighbour_offset exactly) and neighbour_past80 (a past-80% training row among
the 10 nearest: does the trace reach the pair at all?).

Flag level: the ceil((n + 1) * 0.95)-th smallest score on T4 (split-conformal 95% level). A pair is flagged when
its score is above the level.

Pair sets (fixed before any run; spec.json):
  T1   1,500 full-throttle-region training states, PAIRED across worlds (spec v2): log rows that are kept training
       rows in the region in BOTH worlds' gapped logs (so kept in every trace log), chosen by
       default_rng([20261015, seed, 92]); the same log positions in both worlds (the state at a position can differ
       between the worlds). At each, the log's trace critic's actor's command and that command capped at 80%. Primary set.
       Also scored at the row's recorded command a_i (re-referencing, reported: score(s, a) - score(s, a_i)).
  T1c  the T1 states at a command common to both worlds: check 1's full-throttle pattern (recorder.pattern, float32)
       and the same pattern capped at 80% (check 1's c_hi and c_cap). Every pair is past 80%. Reported: unlike T1,
       its past-80% pairs are not selected by what each world's critic taught its actor.
  T3a  1,500 strong-current states visited by the trace critic's actor (exact rollouts from the first 200 starts of
       the strong-current bank, 100 steps each), chosen by default_rng([20261015, seed, 93, world index, 0]); the
       actor's command and its cap. Reported.
  T3b  the same with states visited by the stall-aware controller (default_rng([.., 93, world index, 1])). Reported.
  T4   5,000 held-out recorded rows: the evaluation split outside the gap ball, chosen by
       default_rng([20261012, seed, 77]) (the fallback diagnostic's check rows), at their recorded commands. Sets the
       flag levels; never trained on.
Past-80% pairs: pairs whose command is past 80% on any thruster (thruster_chain.past_knee).
Per set and signal: the flag rate at past-80% pairs (the selectivity's input), at their 80% caps, at the other
pairs; paired widening (the share of past-80% pairs whose score at the command exceeds the score at the cap); on T1
and T1c the re-referenced score at past-80% pairs (mean of score(s, a) - score(s, a_i), and the share above 0).
Selectivity (run_job.py, pooled phase): the flag rate at past-80% pairs in the stalling world minus the same in the
no-stall world, per seed, share, set and signal; a world with fewer than 20 past-80% pairs in a set gives no value.
"""

import math

import numpy as np
from scipy.spatial import cKDTree

import tracebed as T

A, L, RC, TC, TR = T.A, T.L, T.RC, T.TC, T.TR

import jax.numpy as jnp  # noqa: E402

K_NEIGHBOURS = 10
LEVEL = 0.95
RANGE_QUANTILES = (0.005, 0.995)
DRAWS = 32
SIGNALS = ("residual_evidence", "residual_evidence_noiseless", "distance", "range_hinge", "density", "heads")
EXTRAS = ("neighbour_past80", "neighbour_offset")
PRIMARY_SIGNAL, RIVAL_SIGNAL = "residual_evidence", "range_hinge"
SETS = ("T1", "T1c", "T3a", "T3b")
PRIMARY_SET, COMMON_SET = "T1", "T1c"
REREFERENCED_SETS = ("T1", "T1c")
T1_STATES, T3_STARTS, T3_STATES, T4_ROWS = 1500, 200, 1500, 5000
T4_TAG = 77
STRONG = "strong_current"


def conformal_level(scores, level=LEVEL):
    """The ceil((n + 1) * level)-th smallest score (+inf when that exceeds n)."""
    s = np.sort(np.asarray(scores, np.float64))
    k = int(math.ceil((len(s) + 1) * level))
    return float(s[k - 1]) if 1 <= k <= len(s) else math.inf


# ------------------------------------------------------------------------------------------------ one-step error

def target_noise_draw(rep, draw, rows, n_log, n_act=4):
    """Standard normal z for draw `draw` at the given log rows: row r of
    default_rng([20261015, seed, 94, draw + 1]).standard_normal((n_log, n_act)). Depends only on (seed, draw, row)."""
    z = np.random.default_rng([L.LEARNER_SEED, int(rep), T.TAG_NOISE, int(draw) + 1]).standard_normal((int(n_log), n_act))
    return z[np.asarray(rows, np.int64)]


def noisy_next_commands(c, z, policy_noise, noise_clip, max_action=1.0):
    """td3_bc_update's target command: clip(c + clip(policy_noise * z, -noise_clip, noise_clip), -max, max), float32."""
    eps = np.clip(np.asarray(z, np.float64) * float(policy_noise), -float(noise_clip), float(noise_clip))
    return np.clip(np.asarray(c, np.float64) + eps, -max_action, max_action).astype(np.float32)


def one_step_from_q(reward, done, gamma, q_sa, q_next_plain, q_next_mean):
    """The two one-step errors and the offset between them, from per-row values (pure; unit-tested):
    trained = r + g (1 - done) mean_d min Q(s', c + eps_d) - min Q(s, a); noiseless = r + g (1 - done) min Q(s', c)
    - min Q(s, a); J = min Q(s', c) - mean_d min Q(s', c + eps_d); offset = g (1 - done) J = noiseless - trained."""
    r, dn = np.asarray(reward, np.float64), np.asarray(done, np.float64)
    disc = float(gamma) * (1.0 - dn)
    q_sa, qp, qm = (np.asarray(v, np.float64) for v in (q_sa, q_next_plain, q_next_mean))
    J = qp - qm
    return dict(trained=r + disc * qm - q_sa, noiseless=r + disc * qp - q_sa, J=J, offset=disc * J)


def next_state_targets(critic, prepared, settings, rep, n_log, draws=DRAWS):
    """At the next state of every training row of prepared: min_h Q(s', c(s')) and its mean over the training noise
    (draws fixed per log row). Reads the critic only at next states."""
    fixed = L.FixedController(T.CONTROLLER, prepared.physics, prepared.obs_mean, prepared.obs_std)
    nxt = np.asarray(prepared.train.next_obs, np.float32)
    c = np.asarray(fixed.apply(fixed.params(), jnp.asarray(nxt)), np.float32)
    plain = L.critic_rows(critic, nxt, c)[0].min(1)
    acc = np.zeros(len(nxt))
    for d in range(int(draws)):
        z = target_noise_draw(rep, d, prepared.rows, n_log, c.shape[1])
        acc += L.critic_rows(critic, nxt, noisy_next_commands(c, z, settings.policy_noise, settings.noise_clip))[0].min(1)
    return plain, acc / int(draws)


def one_step_errors(critic, prepared, settings, rep, n_log, draws=DRAWS):
    """one_step_from_q at every training row of prepared (module docstring)."""
    plain, mean = next_state_targets(critic, prepared, settings, rep, n_log, draws)
    q = L.critic_rows(critic, np.asarray(prepared.train.obs, np.float32), prepared.actions)[0].min(1)
    return one_step_from_q(np.asarray(prepared.train.reward), np.asarray(prepared.train.done), settings.gamma, q,
                           plain, mean)


# ------------------------------------------------------------------------------------------------ stand-ins

def neighbour_scores(Z_train, delta, past_train, Z_query, k=K_NEIGHBOURS, tree=None, delta_noiseless=None,
                     offset=None):
    """Residual evidence (both forms), distance, the neighbour offset and the past-80% neighbour indicator at
    standardised query points. Pure NumPy and SciPy (unit-tested on synthetic arrays)."""
    tree = tree if tree is not None else cKDTree(Z_train)
    d, idx = tree.query(Z_query, k=k)
    d, idx = np.atleast_2d(d), np.atleast_2d(idx)
    out = dict(residual_evidence=-np.asarray(delta, np.float64)[idx].mean(1), distance=d[:, -1],
               neighbour_past80=np.asarray(past_train, bool)[idx].any(1))
    if delta_noiseless is not None:
        out["residual_evidence_noiseless"] = -np.asarray(delta_noiseless, np.float64)[idx].mean(1)
    if offset is not None:
        out["neighbour_offset"] = np.asarray(offset, np.float64)[idx].mean(1)
    return out


def range_hinge(a, lo, hi):
    a = np.asarray(a, np.float64)
    return (np.maximum(a - hi, 0.0) ** 2 + np.maximum(lo - a, 0.0) ** 2).sum(1)


def density(a, c, sigma):
    return 0.5 * (((np.asarray(a, np.float64) - np.asarray(c, np.float64)) / sigma) ** 2).sum(1)


def _summary_or_none(v, mask):
    return T.summarise(np.asarray(v)[mask]) if np.asarray(mask).any() else None


class StandIns:
    """Every stand-in of one log (its prepared rows, frozen critic and cloned actor)."""

    def __init__(self, sd, prepared, critic, clone, settings, rep, draws=DRAWS):
        self.sd, self.prepared, self.critic, self.clone = sd, prepared, critic, clone
        lg = sd.log
        x = np.asarray(lg["infos/state"][prepared.rows], np.float64)
        a = prepared.actions.astype(np.float64)
        z = np.concatenate([x, a], 1)
        self.mu, sdev = z.mean(0), z.std(0)
        self.sdev = np.where(sdev > 0, sdev, 1.0)
        self.tree = cKDTree((z - self.mu) / self.sdev)
        self.draws = int(draws)
        self.one_step = one_step_errors(critic, prepared, settings, rep, len(lg["actions"]), self.draws)
        self.delta = self.one_step["trained"]
        self.past_train = T.past80(prepared.actions, sd.physics)
        self.lo = np.quantile(a, RANGE_QUANTILES[0], axis=0)
        self.hi = np.quantile(a, RANGE_QUANTILES[1], axis=0)
        c = L.actor_rows(clone, np.asarray(prepared.train.obs, np.float32)).astype(np.float64)
        self.sigma = np.sqrt(np.mean((a - c) ** 2, axis=0))

    def offset_diagnostic(self):
        """The J-by-region diagnostic: J and 0.98 (1 - done) J at the training rows, by the row's region and command."""
        o, reg, past = self.one_step, np.asarray(self.prepared.region, bool), self.past_train
        return dict(
            draws=self.draws,
            J=dict(all=T.summarise(o["J"]), region_rows=_summary_or_none(o["J"], reg),
                   outside_region_rows=_summary_or_none(o["J"], ~reg), past80_rows=_summary_or_none(o["J"], past)),
            offset=dict(all=T.summarise(o["offset"]), region_rows=_summary_or_none(o["offset"], reg),
                        outside_region_rows=_summary_or_none(o["offset"], ~reg)),
            identity_max_abs_error=float(np.max(np.abs(o["noiseless"] - o["trained"] - o["offset"]))),
            region_share_of_training_rows=float(reg.mean()))

    def describe(self):
        o = self.one_step
        return dict(training_rows=int(len(self.delta)), past80_training_rows=int(self.past_train.sum()),
                    one_step_error=T.summarise(o["trained"]),
                    one_step_error_past80_rows=_summary_or_none(o["trained"], self.past_train),
                    one_step_error_noiseless=T.summarise(o["noiseless"]),
                    one_step_error_noiseless_past80_rows=_summary_or_none(o["noiseless"], self.past_train),
                    offset_diagnostic=self.offset_diagnostic(),
                    recorded_range=dict(lo=self.lo.tolist(), hi=self.hi.tolist()), density_sigma=self.sigma.tolist(),
                    standardisation=dict(mean=self.mu.tolist(), sd=self.sdev.tolist()), k=K_NEIGHBOURS)

    def scores(self, x, a):
        x = np.asarray(x, np.float64)
        a32 = np.asarray(a, np.float32)
        a64 = a32.astype(np.float64)
        z = (np.concatenate([x, a64], 1) - self.mu) / self.sdev
        out = neighbour_scores(None, self.delta, self.past_train, z, tree=self.tree,
                               delta_noiseless=self.one_step["noiseless"], offset=self.one_step["offset"])
        obs = self.prepared.normalize(x)
        out["range_hinge"] = range_hinge(a64, self.lo, self.hi)
        c = L.actor_rows(self.clone, obs).astype(np.float64)
        out["density"] = density(a64, c, self.sigma)
        q = L.critic_rows(self.critic, obs, a32)[0]
        out["heads"] = np.abs(q[:, 0] - q[:, 1])
        return out


# ------------------------------------------------------------------------------------------------ pair sets

def t1_pool(sd):
    """One world's T1 candidates: the gate's gapped log's kept training rows in the full-throttle region (as
    prepared.region reads them: float64 of the float32 observation)."""
    lg = sd.log
    kept = sd.removals["gap"].kept_train_rows(lg)
    x = np.asarray(lg["observations"][kept], np.float32).astype(np.float64)
    return kept[RC.Recorder(sd.spec, sd.physics).in_region(x)]


def t1_rows(sds, rep):
    """T1's log rows, paired across worlds: the rows in both worlds' pools, default_rng([20261015, seed, 92]).
    Returns (rows, size of the shared pool)."""
    pool = np.intersect1d(t1_pool(sds["stall"]), t1_pool(sds["no_stall"]))
    rng = np.random.default_rng([L.LEARNER_SEED, int(rep), T.TAG_T1])
    return np.sort(pool[rng.choice(len(pool), min(T1_STATES, len(pool)), replace=False)]), int(len(pool))


def t4_rows(sd):
    """T4's log rows: the evaluation split outside the gap ball, default_rng([20261012, seed, 77])."""
    lg = sd.log
    ev = lg.split_rows("evaluation")
    ev = ev[~sd.gap.contains(lg["actions"][ev])]
    rng = np.random.default_rng([TC.DATA_SEED, int(sd.rep), T4_TAG])
    return np.sort(ev[rng.choice(len(ev), min(T4_ROWS, len(ev)), replace=False)])


def visited_states(policy, sd, which):
    """States visited along exact 100-step rollouts from the first 200 strong-current bank starts, 1,500 chosen by
    default_rng([20261015, seed, 93, world index, which])."""
    x0 = TC.bank_starts(STRONG)[:T3_STARTS]
    run = TR.rollout(policy, x0, sd.physics, record=True)
    S = run.states[:, :-1].reshape(-1, TC.DX)
    rng = np.random.default_rng([L.LEARNER_SEED, int(sd.rep), T.TAG_T3, T.WORLD_INDEX[sd.world], int(which)])
    return S[np.sort(rng.choice(len(S), min(T3_STATES, len(S)), replace=False))]


def cap80(a):
    """The command capped at 80% (float64 clip, then float32, as check 1's c_cap)."""
    return np.clip(np.asarray(a, np.float64), -0.8, 0.8).astype(np.float32)


def common_pattern(sd):
    """Check 1's full-throttle pattern as the critic sees it (float32): the T1c command."""
    return np.asarray(RC.Recorder(sd.spec, sd.physics).pattern, np.float64).astype(np.float32)


def _mean_or_none(v, mask):
    return float(np.asarray(v)[mask].mean()) if np.asarray(mask).any() else None


def set_summary(sc_a, sc_cap, past, levels, sc_rec=None):
    """Per signal: flag rates at past-80% pairs, their caps and the other pairs, paired widening, mean scores; with
    sc_rec (scores at the row's recorded command), the re-referenced score at past-80% pairs."""
    out = dict(pairs=int(len(past)), past80_pairs=int(past.sum()), past80_share=float(past.mean()) if len(past) else None)
    p = past
    for name in SIGNALS:
        sa, sc, lv = sc_a[name], sc_cap[name], levels[name]
        out[name] = dict(
            flag_past80=float((sa[p] > lv).mean()) if p.any() else None,
            flag_cap_of_past80=float((sc[p] > lv).mean()) if p.any() else None,
            flag_not_past80=float((sa[~p] > lv).mean()) if (~p).any() else None,
            paired_widening=float((sa[p] > sc[p]).mean()) if p.any() else None,
            mean_past80=_mean_or_none(sa, p), mean_cap_of_past80=_mean_or_none(sc, p))
        if sc_rec is not None:
            sr = sc_rec[name]
            out[name]["rereferenced"] = dict(
                mean_past80=_mean_or_none(sa - sr, p),
                share_above_recorded_past80=float((sa[p] > sr[p]).mean()) if p.any() else None,
                mean_cap_of_past80=_mean_or_none(sc - sr, p))
    out["neighbour_past80_share_of_past80_pairs"] = _mean_or_none(sc_a["neighbour_past80"], p)
    out["neighbour_past80_share_of_caps"] = _mean_or_none(sc_cap["neighbour_past80"], p)
    out["neighbour_offset_mean_past80"] = _mean_or_none(sc_a["neighbour_offset"], p)
    out["neighbour_offset_mean_cap_of_past80"] = _mean_or_none(sc_cap["neighbour_offset"], p)
    return out


def readout(sds, world, prepared, critic, clone, actor, settings, log_fn=None, draws=DRAWS):
    """The detection readout of one log of `world` (sds: both worlds' loaded seeds, for the paired T1 rows): levels
    on T4, the pair sets and every stand-in. Returns (summary, arrays)."""
    log_fn = log_fn or (lambda *a: None)
    sd = sds[world]
    si = StandIns(sd, prepared, critic, clone, settings, sd.rep, draws)
    lg = sd.log
    rows4 = t4_rows(sd)
    s4 = si.scores(lg["infos/state"][rows4], np.asarray(lg["actions"][rows4], np.float32))
    levels = {k: conformal_level(s4[k]) for k in SIGNALS}
    t4 = dict(rows=int(len(rows4)), rows_sha256=T.DS.sha256_array(rows4), levels=levels,
              flag_rate={k: float((s4[k] > levels[k]).mean()) for k in SIGNALS},
              past80_rows=int(T.past80(lg["actions"][rows4], sd.physics).sum()),
              neighbour_offset_mean=float(s4["neighbour_offset"].mean()))
    log_fn(f"signals: T4 levels {levels}")
    rows1, pool1 = t1_rows(sds, sd.rep)
    x1 = np.asarray(lg["infos/state"][rows1], np.float64)
    rec1 = np.asarray(lg["actions"][rows1], np.float32)
    s_rec = si.scores(x1, rec1)
    pattern = common_pattern(sd)
    states = {"T1": x1, "T1c": x1,
              "T3a": visited_states(L.mlp_policy(actor, prepared), sd, 0),
              "T3b": visited_states(RC.stall_aware(sd.physics), sd, 1)}
    summary = dict(stand_ins=si.describe(), T4=t4, t1_rows_sha256=T.DS.sha256_array(rows1), t1_shared_pool=pool1,
                   t1c_pattern=pattern.tolist(), sets={})
    arrays = {"T4_rows": rows4, "T1_rows": rows1, "T1|recorded": rec1}
    for k in SIGNALS + EXTRAS:
        arrays[f"T4|{k}"] = s4[k]
        arrays[f"T1|{k}|recorded"] = s_rec[k]
    for name in SETS:
        x = states[name]
        a = (np.broadcast_to(pattern, (len(x), len(pattern))).copy() if name == COMMON_SET
             else L.actor_rows(actor, prepared.normalize(x)))
        cap = cap80(a)
        past = T.past80(a, sd.physics)
        sa, sc = si.scores(x, a), si.scores(x, cap)
        summary["sets"][name] = set_summary(sa, sc, past, levels, s_rec if name in REREFERENCED_SETS else None)
        summary["sets"][name]["states_sha256"] = T.DS.sha256_array(x)
        arrays[f"{name}|states"], arrays[f"{name}|command"], arrays[f"{name}|past80"] = x, a, past
        for k in SIGNALS + EXTRAS:
            arrays[f"{name}|{k}|command"], arrays[f"{name}|{k}|cap"] = sa[k], sc[k]
        log_fn(f"signals: {name} {int(past.sum())} past-80% pairs of {len(past)}")
    return summary, arrays


# ------------------------------------------------------------------------------------------------ selectivity

def selectivity(stall_set, no_stall_set, signal, min_pairs=T.MIN_PAST80_PAIRS):
    """Flag rate at past-80% pairs, stalling world minus no-stall world, for one set summary pair; None when either
    world has fewer than min_pairs past-80% pairs."""
    if stall_set["past80_pairs"] < min_pairs or no_stall_set["past80_pairs"] < min_pairs:
        return None
    a, b = stall_set[signal]["flag_past80"], no_stall_set[signal]["flag_past80"]
    return None if a is None or b is None else float(a - b)


def rereferenced_difference(stall_set, no_stall_set, signal, min_pairs=T.MIN_PAST80_PAIRS):
    """Reported: the mean re-referenced score at past-80% pairs, stalling world minus no-stall world (T1, T1c)."""
    if stall_set["past80_pairs"] < min_pairs or no_stall_set["past80_pairs"] < min_pairs:
        return None
    a = (stall_set[signal].get("rereferenced") or {}).get("mean_past80")
    b = (no_stall_set[signal].get("rereferenced") or {}).get("mean_past80")
    return None if a is None or b is None else float(a - b)
