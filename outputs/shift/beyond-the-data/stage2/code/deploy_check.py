"""Stage 2, Variant B: the deployment check that reads BCA's band (plan sections 3 and 5.3). No training.

At every step of an exact 100-step rollout the actor proposes a command a = pi(s). The check reads the band there:

    a_exec(s) = pi(s)                               if U(s, pi(s)) <= tau
                clip_j(pi(s)_j, lo_j, hi_j)        otherwise (and whenever U is not finite)

  - U is band.width at the float32-rounded, normalised state (WidthReader), uniform or weighted as the detection test
    decided, from the band fitted on the frozen critic of the actor being checked: the fixed-controller gapped
    critic's band for the actor-only actors (no caution, bca_k1), the as-written band on the host's own critic for
    plain TD3+BC;
  - tau is the split-conformal 95% level (the ceil((n + 1) 0.95)-th smallest value, fallback.conformal_threshold) of U
    at the held-out recorded rows: every row of the calibration split (200 recorded episodes no learner saw) outside
    the removed ball, outside the calibration bank and outside the detection script's T4 rows (so the detection
    test's calibration-sanity mark reads T4 on rows tau never saw). Its rate above tau on the fallback diagnostic's
    5,000 evaluation-split check rows is reported;
  - lo_j, hi_j: the recorded range, the 0.5% and 99.5% quantiles of the gapped training commands per thruster (the
    fallback diagnostic's clip99, identical to the room's hinge bounds).
  - a uniform band that cannot certify has U = U_max everywhere, so tau = U_max and the check never fires: "maximum
    caution" does not reach Variant B with uniform weights (declared; unreachable with about 850 bank rows).
  - the band's version is the one the detection test chose; run_stage2.phase_deploy refuses until that decision
    exists, and uses the uniform band only if neither version passed direction and size.

Compared, on the same starts and in the same arithmetic (the fallback diagnostic's run_rule, restated with a pluggable
trigger), with: never switching; random switching at the BCA check's own firing rate (a coin per start and step,
seeded per seed, world, actor and situation); the fallback diagnostic's distance check (sa95: the (state, command)
10th-nearest-neighbour distance above its 95% level) and random switching at the distance check's rate. All with the
clip fallback. Starts: the first 200 of each situation's 1,000-start bank. Every return is exact simulation.

Every function that loads data requires a verified approval.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_common as C  # noqa: E402

N_STARTS = 200
LEVEL = 95
TAG_COIN = 92
RULES = ("none", "bca", "random:bca", "sa95", "random:sa95")


def _mods():
    H = C.load_room_harness()
    FB = C.load_fallback()
    return H, FB, H.L, H.TC, H.TR


def heldout_rows(sd, exclude_log_rows=None):
    """Every calibration-split row outside the removed ball, outside the calibration bank and outside
    exclude_log_rows (the detection script's T4 rows): float64 states and commands."""
    lg = sd.log
    cal = lg.split_rows("calibration")
    cal = cal[~sd.gap.contains(lg["actions"][cal])]
    held = np.setdiff1d(cal, lg.bank_rows)
    if exclude_log_rows is not None:
        held = np.setdiff1d(held, np.asarray(exclude_log_rows, np.int64))
    return held, np.asarray(lg["infos/state"][held], np.float64), np.asarray(lg["actions"][held], np.float64)


def check_rows(sd):
    """The fallback diagnostic's 5,000 evaluation-split check rows (fallback.Reference's rule)."""
    H, FB, L, TC, TR = _mods()
    lg = sd.log
    ev = lg.split_rows("evaluation")
    ev = ev[~sd.gap.contains(lg["actions"][ev])]
    ev = ev[np.sort(np.random.default_rng([TC.DATA_SEED, sd.rep, 77]).choice(len(ev), FB.EVAL_CHECK_ROWS,
                                                                             replace=False))]
    return ev, np.asarray(lg["infos/state"][ev], np.float64), np.asarray(lg["actions"][ev], np.float64)


def conformal_level(scores, level=LEVEL):
    s = np.sort(np.asarray(scores, np.float64))
    r = math.ceil((len(s) + 1) * level / 100.0)
    return float(s[r - 1]) if r <= len(s) else math.inf


def fit_tau(ctx, world, reader, approval, out_dir, exclude_log_rows=None):
    """tau for one band (written once to <band dir>/tau.json). Reads the band only at held-out recorded rows; the
    detection script's T4 rows (exclude_log_rows) are left out, so that the share of T4 above tau is read on rows tau
    never saw."""
    C.recheck(approval)
    sd = ctx.sd[world]
    held, xh, ah = heldout_rows(sd, exclude_log_rows)
    u = reader(xh, ah)
    tau = conformal_level(u)
    _, xe, ae = check_rows(sd)
    ue = reader(xe, ae)
    rec = dict(label=C.LABEL_LONG, kind="stage2-tau", rep=int(ctx.rep), world=world, level=LEVEL, tau=tau,
               heldout_rows=int(len(held)), heldout_rows_sha256=C.sha256_bytes(np.ascontiguousarray(held).tobytes()),
               heldout_quantiles={q: float(np.quantile(u, q / 100)) for q in (50, 90, 95, 99)},
               check_rows_rate_above=float(np.mean(~(ue <= tau))), band_file_sha256=reader.sha256,
               prereg_sha256=approval.prereg_sha256,
               excluded_rows=0 if exclude_log_rows is None else int(len(exclude_log_rows)),
               rule=("split-conformal 95% level of U at every calibration-split row outside the removed ball, the bank "
                     "and the detection script's T4 rows"))
    C.write_json(Path(out_dir) / "tau.json", rec)
    return rec


def run_rule(actor, physics, anchor, x0, trig, *, ref, FB, reader=None, tau=None, rate=None, coin=None):
    """fallback.run_rule's arithmetic with the clip fallback and a pluggable trigger: 'none', 'bca' (U > tau or not
    finite), 'sa95' (the fallback's distance check), 'random:*' (a coin with probability `rate`)."""
    H, _, L, TC, TR = _mods()
    t0 = time.perf_counter()
    x = np.asarray(x0, np.float64).copy()
    n = len(x)
    T = physics.length
    J = np.zeros(n)
    fired = np.zeros((n, T), bool)
    past_prop = np.zeros((n, T), bool)
    past_app = np.zeros((n, T), bool)
    rng = np.random.default_rng(coin) if coin is not None else None
    ctx = {"ref": ref}
    for t in range(T):
        a_act = np.clip(actor(x), -1.0, 1.0)
        a = a_act.copy()
        if trig == "none":
            fire = np.zeros(n, bool)
        elif trig == "bca":
            fire = ~(reader(x, a_act) <= tau)
        elif trig == "sa95":
            fire = ref.sa.novel(np.concatenate([x, a_act], 1), ref.sa.thr[95])
        elif trig.startswith("random:"):
            fire = rng.uniform(size=n) < rate
        else:
            raise ValueError(trig)
        if fire.any():
            a[fire] = np.clip(FB.fallback_command("clip99", ctx, x[fire], a_act[fire]), -1.0, 1.0)
        fired[:, t] = fire
        past_prop[:, t] = TC.past_knee(a_act, physics)
        past_app[:, t] = TC.past_knee(a, physics)
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    pts = TR.points(J, anchor)
    fp = fired & past_prop
    return dict(points=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)), fire_share=float(fired.mean()),
                past80_proposed=float(past_prop.mean()), past80_applied=float(past_app.mean()),
                recall_past80=float(fp.sum() / max(past_prop.sum(), 1)),
                precision_past80=float(fp.sum() / max(fired.sum(), 1)),
                flag_rate_not_past80=float((fired & ~past_prop).sum() / max((~past_prop).sum(), 1)),
                rate=rate, coin_seed=coin, per_start_points=np.round(pts, 5).tolist(),
                seconds=time.perf_counter() - t0)


def coin_seed(rep, world, actor_name, situation, trig):
    H, FB, L, TC, TR = _mods()
    return [20261013, int(rep), TAG_COIN, ("stall", "no_stall").index(world),
            ("none", "bca_k1", "td3bc").index(actor_name), TC.SITUATIONS.index(situation), RULES.index(trig)]


def run(ctx, world, actors, readers, approval, out_dir, n=N_STARTS, log_fn=C.log, paths=None):
    """Variant B for one seed and world. actors: {name: params}; readers: {name: (WidthReader, tau)}. Writes
    values.json once. Reported per actor and situation: every rule's points and counts, check - never, check - random
    at the check's rate, distance check - its random."""
    C.recheck(approval)
    H, FB, L, TC, TR = _mods()
    out = Path(out_dir) / "values.json"
    if out.exists():
        raise SystemExit(f"refused: {out} exists; the deployment check is evaluated once")
    t0 = time.perf_counter()
    sd = ctx.sd[world]
    prep = ctx.prepared(world)
    ref = FB.Reference(sd, C.guard_out((paths or C.PATHS).out / "cache", paths))
    an = ctx.anchors(world)
    res = {}
    for name, params in actors.items():
        pol = L.mlp_policy(params, prep, name)
        reader, tau = readers[name]
        res[name] = {}
        for s in TC.SITUATIONS:
            x0 = TC.bank_starts(s, TC.BANK_SIZE)[:n]
            r = {}
            r["none"] = run_rule(pol, prep.physics, an[s], x0, "none", ref=ref, FB=FB)
            r["bca"] = run_rule(pol, prep.physics, an[s], x0, "bca", ref=ref, FB=FB, reader=reader, tau=tau)
            r["random:bca"] = run_rule(pol, prep.physics, an[s], x0, "random:bca", ref=ref, FB=FB,
                                       rate=r["bca"]["fire_share"],
                                       coin=coin_seed(ctx.rep, world, name, s, "random:bca"))
            r["sa95"] = run_rule(pol, prep.physics, an[s], x0, "sa95", ref=ref, FB=FB)
            r["random:sa95"] = run_rule(pol, prep.physics, an[s], x0, "random:sa95", ref=ref, FB=FB,
                                        rate=r["sa95"]["fire_share"],
                                        coin=coin_seed(ctx.rep, world, name, s, "random:sa95"))
            r["check_minus_never"] = r["bca"]["points"] - r["none"]["points"]
            r["check_minus_random"] = r["bca"]["points"] - r["random:bca"]["points"]
            r["distance_minus_random"] = r["sa95"]["points"] - r["random:sa95"]["points"]
            r["check_minus_distance"] = r["bca"]["points"] - r["sa95"]["points"]
            res[name][s] = r
            log_fn(f"deploy seed {ctx.rep} {world} {name} {s}: done")
    rec = dict(label=C.LABEL_LONG, kind="stage2-deploy-values", rep=int(ctx.rep), world=world, starts=int(n),
               prereg_sha256=approval.prereg_sha256, approval_sha256=approval.file_sha256,
               taus={k: float(v[1]) for k, v in readers.items()},
               bands={k: v[0].sha256 for k, v in readers.items()}, reference=ref.describe(), results=res,
               seconds=time.perf_counter() - t0)
    C.write_json(out, rec, paths)
    return rec


def summary_for_marks(rec):
    """{actor: {situation: {check_minus_never, check_minus_random}}} for policy.policy_marks' M5."""
    return {a: {s: dict(check_minus_never=v["check_minus_never"], check_minus_random=v["check_minus_random"])
                for s, v in sits.items()} for a, sits in rec["results"].items()}
