"""Stage 0: the outcome-free world card. No critic, actor or BCA component is trained here; every number comes from
the data, exact simulation of fixed controllers, or closed form.

Trigger (bisect_trigger): 14 bisection steps on [0, 1.2] for the trigger at which the recording controller gives the
full-throttle command on 13% of steps, averaged over calibration seeds 900-902 (300 episodes each, vectorised
simulation through dynamics()). Rounded to 4 decimals it must still return recorder.TRIGGER (0.565). The share depends
only on the water's push, which no thruster changes, so the bisection always uses seeds 900-902, also for the retune.

Marks, on every calibration seed (900-904; the stall-slope-3.0 retune uses 905-909). The design's computed values
are in brackets; the rules, not the values, are what this module reruns:
  T1 full-throttle commands removed by the gap >= 99% [99.1-99.8], and gap centre on thruster 1 <= -0.6
     [-0.74 to -0.78]
  T2 kept rows past 80% <= 0.2% of kept rows [0.03-0.14], and random removal keeps >= 80% of the past-80% rows [84-85]
  T3 best competing 15% ball >= 1.2 gap radii [1.24-1.44]
  T4 kept training transitions that follow the identical rule in both worlds >= 99.8% [99.87-99.97]
  T5 full-throttle-region rows kept >= 40% [44-47]
  T6 normal driving only <= 85 points in recorded conditions [78.4]
  T7 stall-aware minus recording controller >= 20 J in every situation [53.4 / 150.1 / 70.6 / 99.7]
  T8 TD3 target-smoothing SD <= the SD of Q over the recording controller's wobble [0.295 vs 0.363]
  T9 forecast harm in strong current >= 5 points under the bending stand-in [9.5] and >= 20 under the edge stand-in
     [40.4]
If a mark fails, only the pre-registered fallback grid may be searched (fallback_grid): full-throttle share target
{0.12, 0.13} x wobble {0.3, 0.35} x normal-driving damping {0.25, 0.35}, default first.

Forecasts (reported, never pass rules). The TD3+BC actor's fixed point on a critic with action curvature |H| is
pi = (m a + k c) / (m + k) with k = d lambda |H| / 2, lambda = alpha / mean|Q| (alpha 2.5, d = 4 throttles); mean|Q|
uses TD3's smoothed level. k is measured per seed on 1,000 kept rows with the normal-driving continuation (the
stand-in for cloning the gapped log). Two stand-ins for the actor that trusts a rising-thrust critic: 'bend' (goes
k / (m + k) of the way from normal driving toward the rising-thrust command) and 'edge' (min(k / m, 1): a critic that
keeps its slope past the data edge). The informed actor is the same actor capped at 80%; harm = its points minus the
trusting actor's. Room: caution placed only where the trusting actor crosses 80% (dose cap 1.5, and 2.0 as the cap-2
diagnostic) against even caution matched in strength, in amount, the best even caution and a shuffled placement.
Dose-cap leverage: the share of the way BCA's whole dose range [1, 1.5) can move a row back toward the recorded
command, k/(1+k) - k/(1.5+k) under bending (at most 0.101, at k = 1.22) and min(k, 1) - min(k/1.5, 1) at the edge.
"""

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import replace

import numpy as np

from experiments.shift import dataset as DS
from experiments.shift import recorder as RC
from experiments.shift import shift as SH
from experiments.shift import thruster_chain as TC
from experiments.shift import truth as TR

SCHEMA = "bca-shift-stage0-v1"
CALIBRATION_SEEDS = (900, 901, 902, 903, 904)
BISECTION_SEEDS = (900, 901, 902)
RETUNE_SEEDS = (905, 906, 907, 908, 909)
RETUNE_STALL_SLOPE = 3.0
BISECTION = dict(episodes=300, lo=0.0, hi=1.2, iterations=14, decimals=4)
ALPHA, CRITIC_K_REFERENCE = 2.5, 0.58
CONTINUATION_HORIZON = 600
MARKS = dict(T1_full_throttle_removed_min=0.99, T1_centre_thruster1_max=-0.6, T2_kept_past80_share_max=0.002,
             T2_random_keeps_past80_min=0.80, T3_competing_ratio_min=1.2, T4_identical_rule_min=0.998,
             T5_region_kept_min=0.40, T6_normal_driving_points_max=85.0, T7_stall_aware_minus_recording_min=20.0,
             T9_harm_bend_min=5.0, T9_harm_edge_min=20.0)
COMPUTED = dict(T1="99.1-99.8%, centre -0.74 to -0.78", T2="0.03-0.14%; random keeps 84-85%", T3="1.24-1.44",
                T4="99.87-99.97%", T5="44-47%", T6="78.4", T7="53.4 / 150.1 / 70.6 / 99.7", T8="0.295 vs 0.363",
                T9="9.5 (forecast with a critic that bends like the truth past 80%) / 40.4 (forecast with a critic "
                   "that keeps its slope past the edge of the data)")
STAND_INS = dict(
    bend="forecast stand-in 'bend': a critic that bends like the truth past 80%; the actor that trusts it goes "
         "k / (m + k) of the way from normal driving toward the rising-thrust command (m the cloning dose)",
    edge="forecast stand-in 'edge': a critic that keeps its slope past the edge of the data; the actor goes "
         "min(k / m, 1) of the way",
    critic_weight_k_uncorrected="the critic weight by the formula the earlier design drafts used, k = d lambda |H| "
                                "(twice the corrected value and without TD3's own level), shown for comparison only")
LADDER = (1.0, 1.05, 1.1, 1.15, 1.25, 1.5, 2.0, 3.0, 4.0)
EVEN_FORECAST = (1.25, 1.5, 2.0, 3.0, 4.0)


def fallback_grid():
    """The pre-registered fallback grid, default first: (full_throttle_share_target, wobble, normal-driving
    damping)."""
    default = (0.13, 0.3, 0.25)
    rest = [g for g in itertools.product((0.12, 0.13), (0.3, 0.35), (0.25, 0.35)) if g != default]
    return [default] + rest


def bisect_trigger(spec=RC.DEFAULT_SPEC, physics=TC.STALL, seeds=BISECTION_SEEDS, **kw):
    """The trigger for the target share of full-throttle commands (module docstring). Returns (trigger, history)."""
    b = dict(BISECTION, **kw)
    streams = [DS.draw_streams(rep, b["episodes"], physics.length) for rep in seeds]
    lo, hi, history = float(b["lo"]), float(b["hi"]), []
    for _ in range(int(b["iterations"])):
        mid = (lo + hi) / 2
        rec = RC.Recorder(replace(spec, trigger=mid), physics)
        share = float(np.mean([DS.simulate(rec, physics, *s, length=physics.length)["full_throttle"].mean()
                               for s in streams]))
        history.append(dict(trigger=mid, full_throttle_share=share))
        lo, hi = (mid, hi) if share > spec.full_throttle_share_target else (lo, mid)
    return round((lo + hi) / 2, int(b["decimals"])), history


# ------------------------------------------------------------------------------------------------ T1-T5

def world_card(rep, spec=RC.DEFAULT_SPEC, stall_slope=1.5, with_no_stall=True):
    """T1-T5 and the gap of one data seed (stalling world), plus the no-stall world's gap for comparison."""
    physics = TC.world_physics("stall", stall_slope)
    log = DS.generate(rep, "stall", spec, stall_slope=stall_slope)
    gap, record = SH.define_gap(log)
    if gap is None:
        return dict(rep=rep, defined=False, reason=record["reason"]), log, None
    train = log.split_rows("train")
    a = np.asarray(log["actions"][train], np.float64)
    x = log["infos/state"][train]
    press = log.diagnostic("infos/full_throttle_command")[train]
    recorder = RC.Recorder(spec, physics)
    region = recorder.in_region(x)
    ing = gap.contains(a)
    past = TC.past_knee(a, physics)
    rnd = SH.make_removal(log, "random", gap)
    rnd_removed = rnd.removed[train]
    same = np.all(TC.dynamics(x[~ing], a[~ing], physics)[0] == TC.dynamics(x[~ing], a[~ing], TC.NO_STALL)[0], axis=1)
    ep_return = log["rewards"][train].astype(np.float64).reshape(-1, physics.length).sum(1)
    card = dict(
        rep=rep, defined=True, gap=gap.as_dict(), rows=int(len(train)),
        region_share=float(region.mean()), full_throttle_share=float(press.mean()),
        full_throttle_commands_removed=float(ing[press].mean()), removed_rows_full_throttle_share=float(press[ing].mean()),
        region_rows_kept=float(np.mean(~ing[region])), kept_past80_rows=int(past[~ing].sum()),
        kept_past80_share=float(past[~ing].mean()), full_log_past80_share=float(past.mean()),
        random_keeps_past80=float(past[~rnd_removed].sum() / max(1, past.sum())),
        identical_rule_share=float(same.mean()), recording_return=float(ep_return.mean()),
        every_full_throttle_command_past80=bool(np.all(past[press])), nothing_else_past80=bool(not np.any(past[~press])),
    )
    if with_no_stall:
        tlog = DS.generate(rep, "no_stall", spec, stall_slope=stall_slope)
        tgap, trecord = SH.define_gap(tlog)
        tpress = tlog.diagnostic("infos/full_throttle_command")[tlog.split_rows("train")]
        tin = tgap.contains(tlog["actions"][tlog.split_rows("train")]) if tgap is not None else None
        card["no_stall_world"] = dict(
            defined=tgap is not None, gap=None if tgap is None else tgap.as_dict(),
            centre_distance=None if tgap is None else float(np.linalg.norm(np.subtract(tgap.centre, gap.centre))),
            full_throttle_commands_removed=None if tgap is None else float(tin[tpress].mean()),
            same_full_throttle_commands_at_first_step=bool(np.array_equal(
                log.diagnostic("infos/full_throttle_command").reshape(-1, physics.length)[:, 0],
                tlog.diagnostic("infos/full_throttle_command").reshape(-1, physics.length)[:, 0])),
            full_throttle_share=float(tpress.mean()))
    return card, log, gap


# ------------------------------------------------------------------------------------------------ T6, T7

def reference_returns(physics=TC.STALL, spec=RC.DEFAULT_SPEC):
    """Exact bank returns of the reference controllers per situation and their points."""
    an = TR.anchors(physics, spec)
    refs = dict(normal_driving_only=RC.hand_only(spec, physics), rising_thrust=RC.rising_thrust(physics),
                stall_aware=RC.stall_aware(physics))
    out = {}
    for s in TC.SITUATIONS:
        row = dict(recording=an[s]["recording"], recording_se=an[s]["recording_se"], top=an[s]["top"],
                   top_se=an[s]["top_se"])
        for name, pol in refs.items():
            J = TR.bank_rollout(pol, s, physics).J.mean()
            row[name] = float(J)
            row[name + "_points"] = float(TR.points(J, an[s]))
        out[s] = row
    return out, an


# ------------------------------------------------------------------------------------------------ T8, k

def learnability(log, gap, spec=RC.DEFAULT_SPEC, physics=TC.STALL, rows=1000, sub=300, h=1e-3,
                 horizon=CONTINUATION_HORIZON):
    """T8 and the critic weight k on the gapped log's kept training rows (module docstring; seeds as the design's
    scratch: row choice default_rng(3), target noise default_rng(11), wobble default_rng([12, i]) / ([13, i])).
    critic_weight_k is the corrected weight k = d lambda |H| / 2 with lambda from TD3's smoothed level;
    critic_weight_k_without_level the same without that level; critic_weight_k_uncorrected the formula the earlier
    design drafts used, k = d lambda |H| (no factor 1/2, no level), reported for comparison only."""
    removal = SH.make_removal(log, "gap", gap)
    kept = removal.kept_train_rows(log)
    xk = log["infos/state"][kept]
    ak = np.asarray(log["actions"][kept], np.float64)
    sel = np.random.default_rng(3).choice(len(xk), rows, replace=False)
    xs, al = xk[sel], ak[sel]
    hand = RC.hand_only(spec, physics)
    qa = TR.q_value(hand, xs, al, physics, horizon)
    qv = TR.q_value(hand, xs, hand(xs), physics, horizon)
    hd = []
    for j in range(TC.DA):
        e = np.zeros(TC.DA)
        e[j] = h
        hd.append((TR.q_value(hand, xs, hand(xs) + e, physics, horizon)
                   + TR.q_value(hand, xs, hand(xs) - e, physics, horizon) - 2 * qv) / h ** 2)
    curvature = float(np.median(np.abs(np.stack(hd, 1))))
    mean_q = float(np.abs(qv).mean())
    qs, qse = TR.smoothed_q(hand, xs[:sub], hand(xs[:sub]), physics, paths=16, horizon=horizon, seed=TR.SMOOTH_SEED)
    mean_q_level = float(np.abs(qs).mean())
    x1, _ = TC.dynamics(xs[:sub], al[:sub], physics)
    eps = np.clip(TR.TD3_NOISE * np.random.default_rng(11).standard_normal((16, sub, TC.DA)), -TR.TD3_CLIP, TR.TD3_CLIP)
    tq = np.stack([TR.q_value(hand, x1, np.clip(hand(x1) + eps[i], -1, 1), physics, horizon) for i in range(16)])
    rec = RC.Recorder(spec, physics)
    jq = np.stack([TR.q_value(hand, xs[:sub], rec(xs[:sub], np.random.default_rng([12, i]).standard_normal(
        (sub, TC.DA)), np.random.default_rng([13, i]).uniform(size=sub)), physics, horizon) for i in range(8)])
    smoothing_sd = float(np.median(physics.gamma * tq.std(0)))
    wobble_sd = float(np.median(jq.std(0)))
    k = TC.DA * (ALPHA / mean_q_level) * curvature / 2.0
    rising = RC.rising_thrust(physics)
    wrong = lambda z: TC.past_knee(rising(z), physics)
    return dict(
        rows=rows, smoothing_rows=sub, horizon=horizon, mean_abs_q=mean_q, mean_abs_q_with_td3_level=mean_q_level,
        level_mc_se_max=float(qse.max()), median_action_curvature=curvature,
        action_share_of_q_variance=float(np.var(qa - qv) / np.var(qa)), q_sd=float(qa.std()),
        advantage_sd=float(np.std(qa - qv)), target_smoothing_sd=smoothing_sd, wobble_sd_of_q=wobble_sd,
        critic_weight_k=k, critic_weight_k_without_level=TC.DA * (ALPHA / mean_q) * curvature / 2.0,
        critic_weight_k_uncorrected=TC.DA * (ALPHA / mean_q) * curvature,
        affected_kept_rows=float(wrong(xk).mean()),
        affected_kept_rows_s_or_next=float(np.mean(wrong(xk) | wrong(TC.dynamics(xk, ak, physics)[0]))))


# ------------------------------------------------------------------------------------------------ forecasts

class _Blend(RC.Controller):
    """Forecast stand-in: normal driving moved a share w of the way toward the rising-thrust command, w = k / (m + k)
    ('bend') or min(k / m, 1) ('edge') for the cloning dose m(x); optional cap at 80% (the informed actor)."""

    def __init__(self, hand, rising, k, shape, dose, cap=None):
        self.hand, self.rising, self.k, self.shape, self.dose, self.cap = hand, rising, float(k), shape, dose, cap
        self.name = f"blend stand-in ({shape})"

    def __call__(self, x):
        m = np.asarray(self.dose(x), np.float64)[..., None]
        ab, ac = self.hand(x), self.rising(x)
        w = self.k / (m + self.k) if self.shape == "bend" else np.minimum(self.k / m, 1.0)
        a = np.clip(ab + w * (ac - ab), -1, 1)
        return a if self.cap is None else np.clip(a, -self.cap, self.cap)


def _const(v):
    return lambda x: np.full(np.asarray(x).shape[:-1], float(v))


def forecasts(k, log_states, anchors, spec=RC.DEFAULT_SPEC, physics=TC.STALL):
    """Harm and room forecasts at critic weight k for both stand-ins on every situation (points; never pass rules).
    log_states: the training log's states (all 50,000 rows), used for strength and amount matching."""
    hand, rising = RC.hand_only(spec, physics), RC.rising_thrust(physics)
    sub = log_states[np.random.default_rng(1).choice(len(log_states), 5000, replace=False)]
    out = {}
    for shape in ("bend", "edge"):
        mk = lambda dose, cap=None: _Blend(hand, rising, k, shape, dose, cap)
        trusting = mk(_const(1.0))
        cross = lambda z: TC.past_knee(trusting(z), physics)
        strength = lambda p: float(np.sqrt(np.mean(np.sum((p(sub) - trusting(sub)) ** 2, 1))))
        s_ladder = np.array([strength(mk(_const(m))) for m in LADDER])
        share = float(cross(log_states).mean())

        def shuffled(z, cap):
            hsh = (np.abs(np.sin(z[..., :TC.DS] @ np.arange(1, TC.DS + 1) * 37.1)) * 1e4) % 1.0
            return np.where(hsh < share, cap, 1.0)

        rows = {}
        for s in TC.SITUATIONS:
            pts = lambda p: float(TR.points(TR.bank_rollout(p, s, physics).J.mean(), anchors[s]))
            p_trust, p_inf, p_bc = pts(trusting), pts(mk(_const(1.0), 0.8)), pts(hand)
            even = {m: pts(mk(_const(m))) for m in LADDER}
            row = dict(cloning=p_bc, trusting=p_trust, informed=p_inf, harm=p_inf - p_trust,
                       best_even=max(even.values()), best_even_multiplier=max(even, key=even.get))
            for cap in (1.5, 2.0):
                oracle = lambda z, cap=cap: np.where(cross(z), cap, 1.0)
                po = mk(oracle)
                m_str = float(np.interp(strength(po), s_ladder, LADDER))
                amount = float(oracle(log_states).mean())
                p_or = pts(po)
                tag = "cap15" if cap == 1.5 else "cap2"
                row[f"harm_placed_{tag}"] = p_or
                row[f"harm_placed_minus_strength_even_{tag}"] = p_or - pts(mk(_const(m_str)))
                row[f"harm_placed_minus_amount_even_{tag}"] = p_or - pts(mk(_const(amount)))
                row[f"harm_placed_minus_shuffled_{tag}"] = p_or - pts(mk(lambda z, cap=cap: shuffled(z, cap)))
                row[f"harm_placed_minus_best_even_{tag}"] = p_or - row["best_even"]
                row[f"strength_matched_multiplier_{tag}"] = m_str
            rows[s] = row
        out[shape] = dict(share_of_log_states_crossing_80=share, situations=rows)
    return out


def dose_cap_leverage(k):
    bend = lambda kk, m: kk / (m + kk)
    edge = lambda kk, m: min(kk / m, 1.0)
    kmax = math.sqrt(1.5)
    return dict(k=k, share_of_way_dose1=bend(k, 1.0), share_of_way_dose15=bend(k, 1.5),
                share_of_way_dose2=bend(k, 2.0), leverage_bend=bend(k, 1.0) - bend(k, 1.5),
                leverage_edge=edge(k, 1.0) - edge(k, 1.5), leverage_cap2_bend=bend(k, 1.0) - bend(k, 2.0),
                leverage_bend_max=bend(kmax, 1.0) - bend(kmax, 1.5), leverage_bend_max_at_k=kmax)


# ------------------------------------------------------------------------------------------------ marks

def judge(card, learn, refs, fc):
    """Pass/fail of T1-T9 for one seed."""
    g = card["gap"]
    sc = fc["bend"]["situations"]["strong_current"]["harm"], fc["edge"]["situations"]["strong_current"]["harm"]
    t7 = {s: refs[s]["stall_aware"] - refs[s]["recording"] for s in TC.SITUATIONS}
    marks = dict(
        T1=card["full_throttle_commands_removed"] >= MARKS["T1_full_throttle_removed_min"]
        and g["centre"][0] <= MARKS["T1_centre_thruster1_max"],
        T2=card["kept_past80_share"] <= MARKS["T2_kept_past80_share_max"]
        and card["random_keeps_past80"] >= MARKS["T2_random_keeps_past80_min"],
        T3=g["competing_ratio"] >= MARKS["T3_competing_ratio_min"],
        T4=card["identical_rule_share"] >= MARKS["T4_identical_rule_min"],
        T5=card["region_rows_kept"] >= MARKS["T5_region_kept_min"],
        T6=refs["recorded"]["normal_driving_only_points"] <= MARKS["T6_normal_driving_points_max"],
        T7=all(v >= MARKS["T7_stall_aware_minus_recording_min"] for v in t7.values()),
        T8=learn["target_smoothing_sd"] <= learn["wobble_sd_of_q"],
        T9=sc[0] >= MARKS["T9_harm_bend_min"] and sc[1] >= MARKS["T9_harm_edge_min"])
    values = dict(
        T1=dict(full_throttle_commands_removed=card["full_throttle_commands_removed"], centre_thruster1=g["centre"][0]),
        T2=dict(kept_past80_share=card["kept_past80_share"], kept_past80_rows=card["kept_past80_rows"],
                random_keeps_past80=card["random_keeps_past80"]),
        T3=dict(competing_ratio=g["competing_ratio"]), T4=dict(identical_rule_share=card["identical_rule_share"]),
        T5=dict(region_rows_kept=card["region_rows_kept"]),
        T6=dict(normal_driving_only_points_recorded=refs["recorded"]["normal_driving_only_points"]),
        T7=dict(stall_aware_minus_recording=t7),
        T8=dict(target_smoothing_sd=learn["target_smoothing_sd"], wobble_sd_of_q=learn["wobble_sd_of_q"]),
        T9=dict(harm_bend=sc[0], harm_edge=sc[1], critic_weight_k=learn["critic_weight_k"]))
    return marks, values


def expected_trigger(spec):
    """The trigger the bisection must return: recorder.TRIGGER (0.565) for the default 13% target; None (no fixed
    value) for the fallback grid's 12% target. The trigger depends only on the water's push, which no thruster
    changes, so neither the stall slope nor the wobble or damping can move it."""
    return RC.TRIGGER if spec.full_throttle_share_target == RC.DEFAULT_SPEC.full_throttle_share_target else None


def run_stage0(seeds=CALIBRATION_SEEDS, spec=RC.DEFAULT_SPEC, stall_slope=1.5, log=print, with_no_stall=True,
               with_density=True, bisection_seeds=BISECTION_SEEDS):
    """Bisection (always on calibration seeds 900-902), then T1-T9 and the forecasts on every seed. Returns the
    stage-0 record (JSON-ready). The stall-slope-3.0 retune passes stall_slope=3.0 and seeds=RETUNE_SEEDS."""
    t0 = time.time()
    physics = TC.world_physics("stall", stall_slope)
    trigger, history = bisect_trigger(spec, physics, bisection_seeds)
    expected = expected_trigger(spec)
    trigger_ok = expected is None or trigger == expected
    spec = replace(spec, trigger=trigger)
    log(f"trigger {trigger} (expected {expected}); {time.time() - t0:.0f}s")
    refs, an = reference_returns(physics, spec)
    k_ref_fc, k_ref_seed = None, None
    per_seed = []
    for rep in seeds:
        card, lg, gap = world_card(rep, spec, stall_slope, with_no_stall)
        if not card["defined"]:
            per_seed.append(dict(rep=rep, card=card, marks=dict(T1=False), passed=False))
            log(f"seed {rep}: gap not defined: {card['reason']}")
            continue
        learn = learnability(lg, gap, spec, physics)
        states = lg["infos/state"][lg.split_rows("train")]
        fc = forecasts(learn["critic_weight_k"], states, an, spec, physics)
        if k_ref_fc is None:
            k_ref_fc, k_ref_seed = forecasts(CRITIC_K_REFERENCE, states, an, spec, physics), rep
        marks, values = judge(card, learn, refs, fc)
        rec = dict(rep=rep, card=card, learnability=learn, forecasts=fc,
                   dose_cap_leverage=dose_cap_leverage(learn["critic_weight_k"]), marks=marks, values=values,
                   passed=all(marks.values()))
        if with_density:
            rec["density_check_note"] = SH.density_check_note(lg, gap)
        per_seed.append(rec)
        failed = [k for k, v in marks.items() if not v]
        log(f"seed {rep}: {'every mark T1-T9 passes' if not failed else 'marks failed: ' + ', '.join(failed)}; "
            f"critic weight k {learn['critic_weight_k']:.3f}; {time.time() - t0:.0f}s")
    return dict(schema=SCHEMA, seeds=list(seeds), stall_slope=stall_slope, recorder=spec.as_dict(),
                trigger=dict(value=trigger, expected=expected, matches=trigger_ok, history=history,
                             bisection_seeds=list(bisection_seeds)),
                reference_returns=refs, anchors=an, marks_rule=MARKS, computed_in_design=COMPUTED,
                stand_ins=STAND_INS,
                forecasts_at_reference_k=dict(k=CRITIC_K_REFERENCE, forecasts=k_ref_fc, log_states_seed=k_ref_seed,
                                              dose_cap_leverage=dose_cap_leverage(CRITIC_K_REFERENCE)),
                per_seed=per_seed, passed=bool(per_seed) and all(r["passed"] for r in per_seed) and trigger_ok,
                fallback_grid=[dict(full_throttle_share_target=a, wobble=b, damping=c) for a, b, c in fallback_grid()],
                seconds=time.time() - t0)


def search_fallback(seeds=CALIBRATION_SEEDS, stall_slope=1.5, log=print, **kw):
    """The only search allowed when a mark fails: the pre-registered grid, in fallback_grid() order (default first),
    stopping at the first setting that passes on every seed. Returns (chosen spec or None, every record)."""
    records = []
    for target, wobble, damping in fallback_grid():
        spec = replace(RC.DEFAULT_SPEC, full_throttle_share_target=target, wobble=wobble, damping=damping)
        log(f"fallback grid: full-throttle share target {target}, wobble {wobble}, normal-driving damping {damping}")
        record = run_stage0(seeds, spec, stall_slope, log, **kw)
        records.append(record)
        if record["passed"]:
            return RC.RecorderSpec(**record["recorder"]), records
    return None, records


# ------------------------------------------------------------------------------------------------ build checks

def adversarial_command(x):
    """Full throttle on every thruster, each pushing the way its two floats are already moving (sign of B^T x): the
    command that pumps the most energy into the floats per step; used to stress the observation bound."""
    _, B = TC.matrices(TC.NO_STALL)
    return np.where(B[:TC.DS].T @ np.asarray(x, np.float64)[:TC.DS] >= 0, 1.0, -1.0).astype(np.float32)


def build_checks(replay_seed=900, adversarial_steps=10_000):
    """The environment's part of the build checks (no learning): check_env with no warnings on all 8 ids, seeded
    determinism, truncation at 100, render, the proven bound under adversarial full-throttle steps in strong current,
    step() against dynamics() (bitwise) and against a separately written matrix form (1e-12), every row of one log
    replayed bitwise, regeneration identical, and the truth anchors."""
    import warnings

    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    out = {}
    env_ok = {}
    for eid in TC.ENV_IDS:
        env = gym.make(eid, render_mode="ansi")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env.unwrapped, skip_render_check=False)
        env_ok[eid] = [str(w.message) for w in caught]
        env.close()
    out["check_env_warnings"] = env_ok
    e1, e2 = gym.make(TC.env_id("stall", "recorded")), gym.make(TC.env_id("stall", "recorded"))
    out["seeded_reset_equal"] = bool(np.array_equal(e1.reset(seed=5)[0], e2.reset(seed=5)[0]))
    trunc_at = None
    e1.reset(seed=1)
    for t in range(TC.STALL.length):
        _, _, term, trunc, _ = e1.step(e1.action_space.sample())
        if term:
            break
        if trunc:
            trunc_at = t + 1
            break
    out["truncated_at"] = trunc_at
    env = TC.ThrusterChainEnv(stall=False, start="strong_current", render_mode="ansi")
    env.reset(options={"start": dict(current=(1.5, 1.5), wave=(1.0, 1.0))}, seed=0)
    worst, bitwise, matrix_gap = 0.0, True, 0.0
    A, B = TC.matrices(TC.NO_STALL)
    for k in range(adversarial_steps):
        if k % TC.STALL.length == 0:
            env.reset(options={"start": dict(current=(1.5, 1.5), wave=(1.0, 1.0))})
        x = env._x.copy()
        a = adversarial_command(x)
        _, r, _, _, info = env.step(a)
        nxt, rr = TC.dynamics(x, a.astype(np.float64), env.physics)
        bitwise &= bool(np.array_equal(nxt, info["state"]) and rr == r)
        matrix_gap = max(matrix_gap, float(np.abs(A @ x + B @ a.astype(np.float64) - info["state"]).max()))
        worst = max(worst, float(TC.float_radius(info["state"])))
    out["adversarial"] = dict(steps=adversarial_steps, largest_float_radius=worst, bound=TC.observation_bound(),
                              step_equals_dynamics_bitwise=bitwise, matrix_form_max_gap=matrix_gap,
                              render=env.render())
    log = DS.generate(replay_seed)
    again = DS.generate(replay_seed)
    out["regeneration_identical"] = log.array_hashes() == again.array_hashes()
    out["replay"] = DS.replay(log)
    aw = RC.stall_aware()
    xs = log["infos/state"][:200]
    al = np.asarray(log["actions"][:200], np.float64)
    q1, q2 = TR.q_value(aw, xs, al, horizon=1000), TR.q_value(aw, xs, al, horizon=2000)
    out["truth"] = dict(truncation=float(np.abs(q1 - q2).max()), bellman=TR.bellman_residual(aw, xs, al),
                        closed_form=TR.closed_form_check(),
                        jax_vs_numpy=float(np.abs(TR.q_value_jax(aw, xs, al) - q1).max()))
    out["passed"] = bool(all(not w for w in env_ok.values()) and out["seeded_reset_equal"] and trunc_at == 100
                         and bitwise and matrix_gap <= 1e-12 and worst <= TC.FLOAT_BOUND
                         and out["regeneration_identical"] and out["replay"]["mismatches"] == 0
                         and out["truth"]["truncation"] <= 1e-6 and out["truth"]["bellman"] <= 1e-9
                         and out["truth"]["closed_form"]["max_rel"] <= 1e-10
                         and out["truth"]["closed_form"]["continuation_inside"]
                         and out["truth"]["jax_vs_numpy"] <= 1e-12)
    return out


def _json_default(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))


def main(argv=None):
    p = argparse.ArgumentParser(description="ThrusterChain stage 0 (outcome-free): build checks or marks T1-T9.")
    p.add_argument("command", choices=("checks", "marks"))
    p.add_argument("--out", required=True, help="output directory (never under runs/glq, runs/shift or experiments)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    p.add_argument("--stall-slope", type=float, default=1.5, help="3.0 for the pre-registered retune (seeds 905-909)")
    p.add_argument("--grid", action="store_true",
                   help="search the pre-registered fallback grid (only if a mark failed)")
    a = p.parse_args(argv)
    out = DS.refuse_protected(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.command == "checks":
        result = build_checks(replay_seed=a.seeds[0])
        name = "build_checks.json"
    elif a.grid:
        chosen, records = search_fallback(tuple(a.seeds), stall_slope=a.stall_slope)
        result = dict(schema=SCHEMA, search="fallback grid", chosen=None if chosen is None else chosen.as_dict(),
                      records=records, passed=chosen is not None)
        name = "stage0_grid.json"
    else:
        result = run_stage0(tuple(a.seeds), stall_slope=a.stall_slope)
        name = "stage0.json"
    (out / name).write_text(json.dumps(result, indent=2, default=_json_default, allow_nan=True))
    print(f"wrote {out / name}; passed: {result['passed']}")


if __name__ == "__main__":
    sys.exit(main())
