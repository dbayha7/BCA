"""BCA's band weighted for the actor's own commands under the strong test law: a check without training (EXPLORATORY;
pass marks in channel/PASSMARKS_command_weighted_band.md, written before this ran; the owner's go of 2026-10-14).

Reads (never writes) runs/shift_iw_v5: per quality and seed, the gapped host, its frozen band and band record, the data
and the lower-edge room check's no-caution actor ('le_none'). Only the weights and test masses differ from version 5:
the exact ratio of a (row, command) is
    omega = w(c) x pi_h(a | s) x Z_eps(s) / beta_eps(a | s)
(the strong law's w(c); the actor's spread pi_h, BCA's own target noise N(0, 0.2^2) clipped at +-0.5; the recording
controller's command density beta_eps for the row's episode's random share eps; Z_eps(s) the recording process's share
outside the removed ball). Writes runs/shift_iw_v5_explore/channel/command_band/<quality>.json.

Usage (from /root/bca-shift, PYTHONPATH=/root/bca-shift, JAX_PLATFORMS=cpu):
    python command_band_check.py <quality> [<quality> ...]
"""
import json
import math
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import enable_x64

from experiments.shift import learners as LR
from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC
from experiments.shift_iw import coverage as CV
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import run as RN
from experiments.shift_iw import world as W
from runtime.networks import Transition

OUT = RN.ROOT_DEFAULT.parent / "shift_iw_v5_explore" / "channel" / "command_band"
LABEL = ("EXPLORATORY: outside the version-5 pre-registration; pass marks in "
         "channel/PASSMARKS_command_weighted_band.md, written before this ran")
STAGE = "coverage"
TILT = "strong"
SPREAD, CLIP = 0.2, 0.5                 # BCA's own target noise (td3_bc policy_noise, noise_clip)
GRID_RATIO = 1.05                       # coarser than version 5's 1.01; lookups round the mass up (conservative)
FLOOR = 1e-4                            # test masses floored at FLOOR x the mean bank weight
MC_NORMAL, MC_HOLD, MC_RANDOM = 256, 20000, 200000
TAG = 4242                              # this check's random streams: default_rng([20261016, seed, TAG, part])
REC = W.HoldRecorder()
LOG_PHI0 = -0.5 * math.log(2.0 * math.pi)


# ------------------------------------------------------------------------------------------------ exact densities

def recorder_mean(x):
    fl, ex = RC._linear(np.asarray(x, np.float64), REC.K)
    return -(REC.spec.damping * fl + ex)


def normal_logdensity(a, x):
    """log n(a | x): normal driving a = 0.8 tanh(u / 0.8), u ~ N(mu(x), 0.3^2 I); -inf where any |a_k| >= 0.8."""
    a = np.asarray(a, np.float64)
    cap, s = REC.spec.normal_driving_cap, REC.spec.wobble
    inside = np.all(np.abs(a) < cap, axis=-1)
    r = np.clip(a / cap, -1.0 + 1e-15, 1.0 - 1e-15)
    z = (cap * np.arctanh(r) - recorder_mean(x)) / s
    lp = np.sum(LOG_PHI0 - 0.5 * z * z - math.log(s) - np.log1p(-r * r), axis=-1)
    return np.where(inside, lp, -np.inf)


def recorder_logdensity(a, x, eps, hard):
    """log beta_eps(a | x): (1 - eps)(1 - 0.5 hard) n(a | x) + eps / 16 (the hold is a point mass: density 0)."""
    eps = np.asarray(eps, np.float64)
    p_norm = (1.0 - eps) * (1.0 - REC.spec.hold_probability * np.asarray(hard, np.float64))
    with np.errstate(divide="ignore"):
        terms = np.stack([np.log(p_norm) + normal_logdensity(a, x), np.log(eps / 16.0)], 0)
    m = np.max(terms, 0)
    with np.errstate(invalid="ignore"):
        out = m + np.log(np.sum(np.exp(terms - np.where(np.isfinite(m), m, 0.0)), 0))
    return np.where(np.isfinite(m), out, -np.inf)


def spread_logdensity(a, centre):
    """log pi_h(a | s): prod_k phi((a_k - centre_k) / 0.2) / 0.2 where every |a_k - centre_k| < 0.5; -inf elsewhere."""
    d = np.asarray(a, np.float64) - np.asarray(centre, np.float64)
    inside = np.all(np.abs(d) < CLIP, axis=-1)
    lp = np.sum(LOG_PHI0 - 0.5 * (d / SPREAD) ** 2 - math.log(SPREAD), axis=-1)
    return np.where(inside, lp, -np.inf)


def ball_constants(gap, rng):
    """P(hold in the ball) and P(uniform random command in the ball), by Monte Carlo."""
    h = np.broadcast_to(REC.hold, (MC_HOLD, TC.DA)).copy()
    h[:, 1:] += REC.spec.jitter * rng.standard_normal((MC_HOLD, TC.DA - 1))
    h[:, 0] = REC.spec.hold_thruster1
    p_hold = float(np.mean(gap.contains(W.round32(h))))
    r = rng.uniform(-1.0, 1.0, size=(MC_RANDOM, TC.DA))
    p_rand = float(np.mean(gap.contains(W.round32(r))))
    return p_hold, p_rand


def normal_in_ball(x, gap, rng, chunk=4096):
    """N_ball(x): the share of normal driving's commands at each state inside the ball (MC_NORMAL draws per state)."""
    x = np.asarray(x, np.float64)
    out = np.empty(len(x))
    cap, s = REC.spec.normal_driving_cap, REC.spec.wobble
    for i0 in range(0, len(x), chunk):
        mu = recorder_mean(x[i0:i0 + chunk])
        u = mu[:, None, :] + s * rng.standard_normal((len(mu), MC_NORMAL, TC.DA))
        a = W.round32(cap * np.tanh(u / cap)).reshape(-1, TC.DA)
        out[i0:i0 + chunk] = gap.contains(a).reshape(len(mu), MC_NORMAL).mean(1)
    return out


def outside_share(eps, hard, n_ball, p_hold, p_rand):
    """Z_eps(s): the recording process's share of commands at s outside the ball."""
    eps, hard = np.asarray(eps, np.float64), np.asarray(hard, np.float64)
    q = REC.spec.hold_probability * hard
    return 1.0 - ((1.0 - eps) * q * p_hold + (1.0 - eps) * (1.0 - q) * n_ball + eps * p_rand)


class Rows:
    """The exact-ratio ingredients of a set of log rows: states, current, random share, hard water, Z."""

    def __init__(self, log, rows, gap, consts, rng):
        self.x = np.asarray(log["infos/state"][rows], np.float64)
        self.c = np.asarray(log.current[rows], np.float64)
        shares = np.asarray(log.meta["random_shares"], np.float64)
        self.eps = shares[np.asarray(log["infos/episode"][rows])]
        self.hard = REC.hard_water(self.x)
        self.Z = outside_share(self.eps, self.hard, normal_in_ball(self.x, gap, rng), *consts)

    def log_ratio(self, idx, a, centre):
        """log omega at rows idx and commands a, with the actor's spread around centre; +inf where beta = 0."""
        lb = recorder_logdensity(a, self.x[idx], self.eps[idx], self.hard[idx])
        ls = spread_logdensity(a, centre)
        with np.errstate(divide="ignore", invalid="ignore"):
            lw = np.log(W.ratio(self.c[idx], TILT)) + ls + np.log(self.Z[idx]) - lb
        return np.where(np.isfinite(lb), lw, np.where(np.isfinite(ls), np.inf, -np.inf))


# ------------------------------------------------------------------------------------------------ thresholds

class Thresholds:
    """R(test mass) for one set of bank weights: a table on a log grid (GRID_RATIO) from the floor to the onset of
    +inf; masses beyond the grid are refused; a lookup takes the grid point at or above the mass."""

    def __init__(self, scores, stream, weights):
        self.weights = np.asarray(weights, np.float64)
        self.floor = FLOOR * float(self.weights.mean())
        t = max(1.0, float(self.weights.mean()))
        for _ in range(80):
            if not math.isfinite(CV.calibrate(scores, stream, self.weights, t).threshold):
                break
            t *= 2.0
        self.onset_upper = t
        self.table = CV.threshold_table(scores, stream, self.weights,
                                        CV.mass_grid(np.array([self.floor, t]), GRID_RATIO))
        finite = self.table.thresholds[np.isfinite(self.table.thresholds)]
        self.cap = float(finite.max()) if finite.size else math.nan
        self.no_atom = float(self.table.thresholds[0])

    def read(self, log_mass):
        """(R, refused): R per row (the cap where refused, as version 5's rule (a) does in the lower edge)."""
        lm = np.asarray(log_mass, np.float64)
        refused = ~np.isfinite(lm) & (lm > 0)
        m = np.where(np.isfinite(lm), np.exp(np.clip(lm, -700, 700)), self.floor)
        m = np.maximum(m, self.floor)
        beyond = m > self.table.grid[-1]
        R = np.full(len(m), np.inf)
        ok = ~refused & ~beyond
        if ok.any():
            R[ok] = CV.lookup(self.table, m[ok])[0]
        refused = refused | beyond | ~np.isfinite(R)
        return np.where(refused, self.cap, R), refused


# ------------------------------------------------------------------------------------------------ truth

def exact_step(states, actions):
    with enable_x64():
        x1, r = TC.dynamics_jax(jnp.asarray(states, jnp.float64),
                                jnp.asarray(np.asarray(actions, np.float32).astype(np.float64)), W.PHYSICS)
        return np.asarray(x1, np.float64), np.asarray(r, np.float64)


def truth_error(ctx, host, prep, seed, actions, tag):
    """y - q at every kept training row's state and the given commands (BCA's target rule, one exact step)."""
    x1, r = exact_step(prep.states, actions)
    trans = Transition(jnp.asarray(prep.train.obs), jnp.asarray(np.asarray(actions, np.float32)),
                       jnp.asarray(r.astype(np.float32)), jnp.asarray(prep.normalize(x1)),
                       jnp.zeros(prep.n, jnp.float32))
    y, q = LI.critic_parts(host, trans, ctx.cfg, seed, LI.coverage_key(ctx.cfg, seed, tag))
    return y - q


def wshare(flag, w, mask):
    m = np.asarray(mask, bool)
    return float(np.sum(np.asarray(w)[m] * np.asarray(flag)[m]) / np.sum(np.asarray(w)[m])) if m.any() else None


def outcome_stats(err, U, refused, w, mask):
    """Covered / refused / missed shares (weighted by w) on the rows in mask."""
    m = np.asarray(mask, bool)
    if not m.any():
        return dict(rows=0)
    ww = np.asarray(w, np.float64)[m]
    ref = np.asarray(refused, bool)[m]
    cov = (~ref) & (np.abs(err[m]) <= U[m])
    tot = ww.sum()
    cert = ~ref
    return dict(rows=int(m.sum()), weight_share=float(tot / np.sum(w)),
                refused=float(ww[ref].sum() / tot), covered=float(ww[cov].sum() / tot),
                covered_or_refused=float(ww[cov | ref].sum() / tot),
                covered_among_certified=float(ww[cov].sum() / ww[cert].sum()) if cert.any() else None)


# ------------------------------------------------------------------------------------------------ one seed

def one(ctx, spec, seed, rule):
    t0 = time.perf_counter()
    data = RN.load_data(ctx, seed, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    host = RN.load_host_state(ctx, "gap", seed, spec)
    brec = RN.require_band(ctx, "gap", seed, spec)
    approval = RN.approval_for(ctx, STAGE)
    band = LI.load_band(RN.band_dir(ctx, "gap", seed) / "band.msgpack", brec["band_sha256"], host, prep, ctx.cfg,
                        seed, approval, brec["band"], stage=STAGE)
    R_u = RN.num(brec["thresholds"]["unweighted"]["threshold"])
    pi = RN.controllers_of(ctx, "room_lower_edge", seed, spec)["le_none"]
    log, gap = data.train_log, data.gap
    rng = lambda part: np.random.default_rng([W.MASTER_SEED, int(seed), TAG, part])

    consts = ball_constants(gap, rng(0))
    bank = Rows(log, prep.bank_rows, gap, consts, rng(1))
    train = Rows(log, prep.rows, gap, consts, rng(2))

    # bank weights: omega at each bank row's own recorded command, the actor's spread around pi(bank state)
    a_bank = np.asarray(log["actions"][prep.bank_rows], np.float32).astype(np.float64)
    pi_bank = np.asarray(LR.actor_rows(pi, np.asarray(prep.bank.obs, np.float32)), np.float64)
    lw_bank = bank.log_ratio(np.arange(len(a_bank)), a_bank, pi_bank)
    hold = np.asarray(log.diagnostic("infos/hold")[prep.bank_rows], bool)
    zero_beta = np.isposinf(lw_bank)
    w_bank = np.where(hold | zero_beta | ~np.isfinite(lw_bank), 0.0, np.exp(np.clip(lw_bank, -700, 700)))
    scores, stream = LI.bank_scores(band, prep.bank, ctx.cfg)
    th = Thresholds(scores, stream, w_bank)

    obs = np.asarray(prep.train.obs, np.float32)
    a_pi = np.asarray(LR.actor_rows(pi, obs), np.float32)
    idx = np.arange(prep.n)
    w_law = W.ratio(train.c, TILT)

    def width(actions, log_mass):
        R, refused = th.read(log_mass)
        U = LE.widths_at(band, prep, actions, LE.as_float32_thresholds(R), stage=STAGE).astype(np.float64)
        return U, refused, R

    # (b) at the actor's own command
    lm_pi = train.log_ratio(idx, a_pi.astype(np.float64), a_pi.astype(np.float64))
    lm_pi = np.where(gap.contains(a_pi.astype(np.float64)), np.inf, lm_pi)
    U_pi, ref_pi, R_pi = width(a_pi, lm_pi)
    err_pi = truth_error(ctx, host, prep, seed, a_pi, 90)
    # (a) one draw from the test distribution per row
    z = np.clip(SPREAD * rng(3).standard_normal(a_pi.shape), -CLIP, CLIP)
    a_q = np.clip(a_pi.astype(np.float64) + z, -1.0, 1.0).astype(np.float32)
    lm_q = train.log_ratio(idx, a_q.astype(np.float64), a_pi.astype(np.float64))
    lm_q = np.where(gap.contains(a_q.astype(np.float64)), np.inf, lm_q)
    U_q, ref_q, _ = width(a_q, lm_q)
    err_q = truth_error(ctx, host, prep, seed, a_q, 92)
    # version 5's exact-strong per-step width at the actor's command (for C3)
    arm = RN.lower_edge_arm(data, prep, brec, "exact_strong", R_u)
    Rv5, _, _ = LE.capped_thresholds(arm["R"], arm["pool"])
    U_v5 = LE.widths_at(band, prep, a_pi, LE.as_float32_thresholds(Rv5), stage=STAGE).astype(np.float64)

    past = np.asarray(TC.stalled(a_pi), bool).any(1)
    groups = dict(all=np.ones(prep.n, bool), within_80=~past, past_80=past, hard_water=train.hard)
    # why the actor's commands are refused (strong-law weighted shares of each group)
    in_ball_pi = np.asarray(gap.contains(a_pi.astype(np.float64)), bool)
    beta_zero_pi = ~in_ball_pi & np.isposinf(lm_pi)
    beyond_pi = ref_pi & ~in_ball_pi & ~beta_zero_pi
    at_knee_pi = np.any(np.abs(a_pi.astype(np.float64)) >= REC.spec.normal_driving_cap, axis=1) & ~past
    why = {g: dict(in_ball=wshare(in_ball_pi, w_law, m), recorder_density_zero=wshare(beta_zero_pi, w_law, m),
                   mass_beyond_onset=wshare(beyond_pi, w_law, m), at_or_past_knee_not_stalled=wshare(at_knee_pi,
                                                                                                  w_law, m))
           for g, m in (("within_80", ~past), ("past_80", past), ("hard_water", train.hard))}
    q_past = np.asarray(TC.stalled(a_q), bool).any(1)
    cert_within = (~past) & (~ref_pi)

    # (d) check (e) on the new lower-edge penalty [U(s, a) - U(s, a_i)]_+
    a_rec = np.asarray(prep.actions, np.float64)
    lm_rec = train.log_ratio(idx, a_rec, a_pi.astype(np.float64))
    R_rec, _ = th.read(lm_rec)
    u_rec = LE.logged_widths(band, prep, LE.as_float32_thresholds(R_rec), stage=STAGE)
    shared = LE.band_shared(band, STAGE)
    val = jax.jit(lambda sh, o, a, R, u: jnp.maximum(LE.band_width(sh["cal_params"], sh["unit"], o, a, R) - u, 0.0))
    grad = jax.jit(lambda sh, o, a, R, u: jax.vmap(jax.grad(
        lambda aa, oo, rr, uu: jnp.maximum(LE.band_width(sh["cal_params"], sh["unit"], oo[None], aa[None], rr) - uu,
                                           0.0)[0]))(a, o, R, u))

    def R_at(r, a):
        lm = train.log_ratio(r, np.asarray(a, np.float64), a_pi[r].astype(np.float64))
        lm = np.where(gap.contains(np.asarray(a, np.float64)), np.inf, lm)
        return th.read(lm)[0].astype(np.float32)

    def values(r, a):
        return np.asarray(val(shared, jnp.asarray(obs[r]), jnp.asarray(a), jnp.asarray(R_at(r, a)),
                              jnp.asarray(u_rec[r])), np.float64)

    def slope(r, a):
        return np.asarray(grad(shared, jnp.asarray(obs[r]), jnp.asarray(a), jnp.asarray(R_at(r, a)),
                               jnp.asarray(u_rec[r])), np.float64)

    rows_e = LE.shape_rows(prep, seed, rule["rows"])
    check_e = LE.penalty_shape(values, slope, host.critic.params, prep, pi, rows_e, rule)
    refused_on_grid = {f"{m:g}": float(np.mean(th.read(np.where(gap.contains(aa := _grid_command(
        a_pi[rows_e], m, rule)), np.inf, train.log_ratio(rows_e, aa, a_pi[rows_e].astype(np.float64))))[1]))
        for m in rule["grid"]}

    out = dict(
        seed=int(seed), rows=int(prep.n), bank_rows=int(len(a_bank)), R_u=R_u,
        ball=dict(p_hold_in_ball=consts[0], p_random_in_ball=consts[1],
                  Z_train=dict(mean=float(train.Z.mean()), min=float(train.Z.min()))),
        bank_weights=dict(kish=CV.kish(w_bank), kish_share=CV.kish(w_bank) / len(w_bank),
                          zero_share=float(np.mean(w_bank == 0)), hold_rows=int(hold.sum()),
                          beta_zero_rows=int((zero_beta & ~hold).sum()), mean=float(w_bank.mean()),
                          max_share=float(w_bank.max() / w_bank.sum())),
        thresholds=dict(no_atom=th.no_atom, no_atom_over_R_u=th.no_atom / R_u, cap=th.cap, cap_over_R_u=th.cap / R_u,
                        onset_upper=th.onset_upper, floor=th.floor, grid_points=int(len(th.table.grid)),
                        monotone=th.table.monotone),
        at_actor={g: outcome_stats(err_pi, U_pi, ref_pi, w_law, m) for g, m in groups.items()},
        refusal_reasons_at_actor=why,
        at_test_draw=dict(all=outcome_stats(err_q, U_q, ref_q, w_law, np.ones(prep.n, bool)),
                          past_80=outcome_stats(err_q, U_q, ref_q, w_law, q_past),
                          within_80=outcome_stats(err_q, U_q, ref_q, w_law, ~q_past)),
        within_80_width_over_v5=dict(
            mean=float(np.average(U_pi[cert_within] / U_v5[cert_within], weights=w_law[cert_within]))
            if cert_within.any() else None,
            median=float(np.median(U_pi[cert_within] / U_v5[cert_within])) if cert_within.any() else None),
        R_at_actor_over_R_u=dict(
            within_80_certified=float(np.average(R_pi[cert_within], weights=w_law[cert_within]) / R_u)
            if cert_within.any() else None),
        check_e=check_e, refused_share_on_check_e_grid=refused_on_grid,
        actor_past_80_share=float(past.mean()), seconds=time.perf_counter() - t0)
    return out


def _grid_command(base, m, rule):
    """check (e)'s command: the actor's command with thruster 1 set along its push to throttle m (the knee at 80%)."""
    base = np.asarray(base, np.float64).copy()
    sign = np.where(base[:, 0] < 0, -1.0, 1.0)
    base[:, 0] = sign * (TC.KNEE if float(m) == float(rule["knee"]) else float(m))
    return base.astype(np.float32).astype(np.float64)


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    for quality in argv:
        ctx = RN.Context(quality=quality)
        spec = RN.spec_digest(ctx)
        rule = RN.spec_lower_edge(RN.frozen_spec(ctx, spec))["rule"].check_e
        seeds = []
        for seed in RN.SEEDS:
            r = one(ctx, spec, seed, rule)
            a = r["at_actor"]
            ctx.log(f"{quality} seed {seed}: kish {r['bank_weights']['kish_share']:.3f}, cap/R_u "
                    f"{r['thresholds']['cap_over_R_u']:.2f}; at actor covered-or-refused all "
                    f"{a['all']['covered_or_refused']:.3f}, within {a['within_80']['covered_or_refused']:.3f} "
                    f"(refused {a['within_80']['refused']:.3f}), past "
                    f"{a['past_80'].get('covered_or_refused', float('nan')):.3f} (refused "
                    f"{a['past_80'].get('refused', float('nan')):.3f}); test draw "
                    f"{r['at_test_draw']['all']['covered_or_refused']:.3f}; check (e) "
                    f"{'pass' if r['check_e']['passed'] else 'FAIL ' + str(r['check_e']['failed'])}; "
                    f"best past 80% {r['check_e']['best_past80_share']:.3f} vs Q1 alone "
                    f"{r['check_e']['q1_alone_best_past80_share']:.3f} ({r['seconds']:.0f}s)")
            seeds.append(r)
            body = dict(label=LABEL, quality=quality, spec_sha256=spec, code_sha256=LI.code_hashes(),
                        settings=dict(tilt=TILT, spread=SPREAD, clip=CLIP, grid_ratio=GRID_RATIO, floor=FLOOR,
                                      mc=dict(normal=MC_NORMAL, hold=MC_HOLD, random=MC_RANDOM), tag=TAG),
                        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seeds=seeds)
            (OUT / f"{quality}.json").write_text(json.dumps(body, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
