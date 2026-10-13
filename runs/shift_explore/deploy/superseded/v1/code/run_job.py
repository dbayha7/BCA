"""EXPLORATORY (plan section 4.3, the deployment-check channel; no training, no BCA). One job: one gate seed, one
world, one saved check-2 actor. At every step of an exact 100-step rollout from each of the first 200 starts of each
1,000-start evaluation bank, the actor proposes a command; a rule decides whether the fallback replaces it for that
step. Every return is exact simulation (thruster_chain.dynamics), scored in points against the gate's anchors.

Rules (RULES): never switch; every step -> fallback; and, for every trigger T of signals.TRIGGERS and every fallback F,
T -> F and random switching at T's own firing rate -> F (a coin per start and step with probability equal to the share
of steps on which T -> F fired, for the same seed, world, actor, situation and fallback). The distance check (sa95)
runs with the clip fallback only. Fallbacks: clip99, the actor's command clipped per thruster to the recorded range
(0.5% and 99.5% quantiles of the gapped training commands; primary); clone, the cloned actor's command (reported).

A full job writes runs/shift_explore/deploy/partial/<job>.json after each situation and <job>.json only when all four
are done. Before the rules, a gate check (world-matched actors only): never switching and capping at 80% on the whole 1,000-start
bank must reproduce check 2's saved points within 1e-6; otherwise the job stops (stop_<job>.json).

Full mode refuses unless: the deploy spec, the code manifest, the pinned imports and the gate files check out; the
output does not exist yet; and the room test's pooled look (runs/shift_explore/room/pooled.json, for the room spec
pinned here) exists and reports the room, recovery and keeps-improvement marks passed. With --after-room-failure it
runs after a pooled look that did not pass (plan 4.8: "the deployment check remains"); that is recorded in the output.

Smoke mode (--smoke): the seed-0 check. Same code, --n starts (default 20), outputs under smoke/ with every point
value withheld (only whether they are finite and a hash), plus timing and an equivalence check against the fallback
diagnostic's own run_rule on the rules both programs share. It needs the spec and the manifest, not the room's look.

Usage (from /home/user/bca):
  python run_job.py --seed K --world stall|no_stall --actor NAME [--after-room-failure]
  python run_job.py --seed 0 --world stall --actor gapped --smoke [--n 20] [--situations ...]
"""

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import signals as S  # noqa: E402

H, FB, L, TC, TR = S.H, S.FB, S.L, S.TC, S.TR
import numpy as np  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N_STARTS = 200
SMOKE_STARTS = 20
FALLBACKS = ("clip99", "clone")
HOME = dict(gapped="stall", td3bc_gapped="stall", no_stall_gapped="no_stall")
JOBS = (("stall", "gapped"), ("stall", "td3bc_gapped"), ("no_stall", "no_stall_gapped"),
        ("no_stall", "td3bc_gapped"), ("no_stall", "gapped"))
ROLE = {
    ("stall", "gapped"): "the gapped critic's actor (check 2), in its own world; the channel-open mark is read here",
    ("stall", "td3bc_gapped"): "plain TD3+BC on the gapped log (check 2's td3bc_gapped), in its own world",
    ("no_stall", "no_stall_gapped"): "the gapped critic's actor of the no-stall world (check 2), in its own world",
    ("no_stall", "td3bc_gapped"): ("plain TD3+BC trained on the stalling world's gapped log, deployed in the no-stall "
                                   "world: the gate trained no TD3+BC in the no-stall world (deviation D1)"),
    ("no_stall", "gapped"): ("the stalling world's gapped critic's actor deployed in the no-stall world: the same-actor "
                             "companion of TD3+BC's no-stall row (reported)"),
}
GATE_CHECK_TOL = 1e-6
TAG_COIN = 91
EQUIV_RULES = (("none", None, "none"), ("always", "clip99", "always"), ("always", "clone", "always"),
               ("unrec", "clip99", "unrec"), ("sa95", "clip99", "sa95"), ("random:unrec", "clip99", "random95"))
EQUIV_RATE = 0.25


def rules():
    out = [("none", None), ("always", "clip99"), ("always", "clone")]
    for t in S.TRIGGERS:
        for fb in (("clip99",) if t == "sa95" else FALLBACKS):
            out += [(t, fb), (f"random:{t}", fb)]
    return tuple(out)


RULES = rules()


def key(trig, fb):
    return f"{trig}:{fb}"


def job_name(rep, world, actor):
    return f"seed{int(rep)}_{world}_{actor}"


def coin_seed(rep, world, actor, situation, fb, trig):
    return [20261013, int(rep), TAG_COIN, S.WORLDS.index(world), S.ACTOR_TAG[actor], TC.SITUATIONS.index(situation),
            FALLBACKS.index(fb), S.TRIGGERS.index(trig)]


# ------------------------------------------------------------------------------------------------ one rule

def run_rule(ctx, sig, x0, fb, trig, rate=None, coin=None):
    """fallback.run_rule with a pluggable trigger (signals.Signals.fire) and two more counts; the same arithmetic."""
    actor, physics = ctx["actor"], ctx["physics"]
    t0 = time.perf_counter()
    x = np.asarray(x0, np.float64).copy()
    n = len(x)
    T = physics.length
    J = np.zeros(n)
    fired = np.zeros((n, T), bool)
    changed = np.zeros((n, T), bool)
    past_prop = np.zeros((n, T), bool)
    past_app = np.zeros((n, T), bool)
    rng = np.random.default_rng(coin) if coin is not None else None
    for t in range(T):
        a_act = np.clip(actor(x), -1.0, 1.0)
        a = a_act.copy()
        if trig == "none":
            fire = np.zeros(n, bool)
        elif trig == "always":
            fire = np.ones(n, bool)
        elif trig.startswith("random:"):
            fire = rng.uniform(size=n) < rate
        else:
            fire = np.asarray(sig.fire(trig, x, a_act), bool)
        if fire.any():
            a[fire] = np.clip(FB.fallback_command(fb, ctx, x[fire], a_act[fire]), -1.0, 1.0)
        fired[:, t] = fire
        changed[:, t] = np.any(a != a_act, 1)
        past_prop[:, t] = TC.past_knee(a_act, physics)
        past_app[:, t] = TC.past_knee(a, physics)
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    pts = TR.points(J, ctx["anchor"])
    fp = fired & past_prop
    return dict(points=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)), J=float(J.mean()),
                fire_share=float(fired.mean()), changed_share=float(changed.mean()),
                past80_proposed=float(past_prop.mean()), past80_applied=float(past_app.mean()),
                recall_past80=float(fp.sum() / max(past_prop.sum(), 1)),
                precision_past80=float(fp.sum() / max(fired.sum(), 1)),
                flag_rate_not_past80=float((fired & ~past_prop).sum() / max((~past_prop).sum(), 1)),
                rate=rate, coin_seed=coin, per_start_points=np.round(pts, 5).tolist(),
                seconds=time.perf_counter() - t0)


def blind(r):
    """Smoke mode: withhold every point value; keep whether they are finite and a hash."""
    r = dict(r)
    pts = np.asarray(r.pop("per_start_points"), np.float64)
    for k in ("points", "points_se", "J"):
        r.pop(k)
    r["points_withheld"] = True
    r["points_finite"] = bool(np.all(np.isfinite(pts)))
    r["per_start_points_sha256"] = hashlib.sha256(np.ascontiguousarray(pts).tobytes()).hexdigest()
    return r


# ------------------------------------------------------------------------------------------------ launch condition

def launch_condition(after_room_failure):
    pooled = C.read_json(C.ROOM_POOLED)
    if pooled is None:
        raise SystemExit("refused: the room test's pooled look (runs/shift_explore/room/pooled.json) has not run; the "
                         "deploy runs only after it")
    pins = C.pins()
    if (pooled.get("kind") != "room-pooled" or not str(pooled.get("label", "")).startswith("EXPLORATORY")
            or pooled.get("spec_sha256") != pins["room_spec_sha256"]):
        raise SystemExit("refused: room pooled.json is not the labelled pooled look of the pinned room spec")
    passed = dict(room=bool(pooled["room"]["passed"]), recovery=bool(pooled["recovery"]["passed"]),
                  keeps_improvement=bool(pooled["keeps_improvement"]["passed"]))
    ok = all(passed.values())
    if not ok and not after_room_failure:
        raise SystemExit("refused: the room test's pooled look did not pass all of the room, recovery and "
                         "keeps-improvement marks; rerun with --after-room-failure only if the deployment check is to "
                         "run regardless (plan 4.8), which is recorded")
    return dict(room_pooled_sha256=C.file_sha256(C.ROOM_POOLED), room_marks_passed=passed, all_passed=ok,
                after_room_failure=bool(after_room_failure and not ok))


# ------------------------------------------------------------------------------------------------ the job

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--world", required=True, choices=S.WORLDS)
    ap.add_argument("--actor", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--situations", nargs="*", default=None)
    ap.add_argument("--after-room-failure", action="store_true")
    a = ap.parse_args()
    started = time.perf_counter()
    rep, world, actor = a.seed, a.world, a.actor
    if (world, actor) not in JOBS:
        raise SystemExit(f"refused: ({world}, {actor}) is not a job of the spec")
    spec, spec_sha = C.load_spec()
    pins = C.check_pins(spec)
    man, man_sha = C.check_code(spec_sha)
    name = job_name(rep, world, actor)
    if a.smoke:
        n = SMOKE_STARTS if a.n is None else int(a.n)
        if not 1 <= n <= N_STARTS:
            raise SystemExit("refused: smoke starts must be between 1 and 200")
        situations = list(a.situations or TC.SITUATIONS)
        tag = "" if situations == list(TC.SITUATIONS) else "_" + "_".join(situations)
        path = C.OUT / "smoke" / f"{name}_n{n}{tag}.json"
        cond = dict(smoke=True, note="seed-0 check; the room's pooled look is not required; points withheld")
    else:
        if a.n is not None or a.situations is not None:
            raise SystemExit("refused: a full job runs the spec's 200 starts in all four situations")
        n, situations = N_STARTS, list(TC.SITUATIONS)
        path = C.OUT / f"{name}.json"
        if path.exists() or (C.OUT / f"stop_{name}.json").exists():
            raise SystemExit(f"refused: {path} (or its stop file) exists")
        cond = launch_condition(a.after_room_failure)
    final = path
    if not a.smoke:
        path = C.OUT / "partial" / f"{name}.json"     # written after each situation; the final file only at the end
    C.log(f"{name}: {'smoke' if a.smoke else 'full'} n={n} situations={situations}")

    sc = H.SeedContext(rep, worlds=tuple(sorted({world, HOME[actor]}, key=S.WORLDS.index)))
    prep, actor_prep = sc.prepared(world), sc.prepared(HOME[actor])
    critic, _ = sc.critic(world)
    acts, rec, h2_dir = sc.h2()
    if rec["failed"].get(actor) or actor not in acts:
        raise SystemExit(f"refused: actor {actor} failed or is missing in check 2, seed {rep}")
    clone_key = "no_stall_clone" if world == "no_stall" else "clone"
    ref = FB.Reference(sc.sd[world], C.FALLBACK_CACHE)
    lo, hi = H.hinge_bounds(prep.actions)
    C.log(f"  loaded ({time.perf_counter() - started:.0f}s)")
    sig = S.Signals(rep, world, actor, acts[actor], actor_prep, prep, sc.sd[world], critic, acts[clone_key], ref)
    C.log(f"  signals ready ({sig.seconds['total']:.0f}s): " + ", ".join(
        f"{k} {v['threshold']:.4g}" for k, v in sig.levels.items()))
    ctx = dict(actor=sig.actor, actor_params=acts[actor], prepared=actor_prep, clone=sig.clone,
               physics=prep.physics, ref=ref)
    an = sc.anchors(world)
    out = dict(label=C.LABEL, kind="deploy-smoke" if a.smoke else "deploy-values", exploratory=True,
               spec_sha256=spec_sha, code_manifest_sha256=man_sha, pinned=pins, launch_condition=cond,
               rep=int(rep), world=world, actor=actor, actor_home_world=HOME[actor], role=ROLE[(world, actor)],
               actor_sha256=L.tree_sha256(acts[actor]), check2_actors_sha256=rec["actors_sha256"],
               check2_dir=str(h2_dir), critic_params_sha256=L.tree_sha256(critic),
               clone=clone_key, n_starts=int(n), situations_run=situations,
               start_subset="the first n starts of each 1,000-start evaluation bank (thruster_chain.bank_starts)",
               reference=ref.describe(),
               recorded_range_check=dict(max_abs_diff_vs_room_hinge_bounds=float(max(np.max(np.abs(lo - ref.lo99)),
                                                                                    np.max(np.abs(hi - ref.hi99))))),
               signals=sig.info, rules=[key(t, f) for t, f in RULES], situations={})
    world_matched = HOME[actor] == world
    for s in situations:
        ctx["anchor"] = an[s]
        bank = TC.bank_starts(s)
        t1 = time.perf_counter()
        if world_matched:
            full_none = run_rule(ctx, sig, bank, None, "none")
            full_cap = run_rule(ctx, sig, bank, "cap80", "always")
            g = rec["evaluations"][actor][s]
            diff = max(abs(full_none["points"] - g["points"]), abs(full_cap["points"] - g["capped_points"]))
            gate = dict(applicable=True, max_abs_diff=float(diff), tolerance=GATE_CHECK_TOL, passed=bool(diff <= GATE_CHECK_TOL),
                        seconds=time.perf_counter() - t1)
        else:
            gate = dict(applicable=False, note="cross-world actor: check 2 never scored it in this world")
        C.log(f"  {s}: gate check {gate}")
        if gate.get("applicable") and not gate["passed"]:
            stop = dict(label=C.LABEL, kind="deploy-stop", spec_sha256=spec_sha, code_manifest_sha256=man_sha,
                        rep=int(rep), world=world, actor=actor, situation=s, gate_check=gate,
                        reason="never switching / capping at 80% on the whole bank did not reproduce check 2")
            if not a.smoke:
                C.write_json(C.OUT / f"stop_{name}.json", stop)
            raise SystemExit(f"stopped: gate check failed in {s} ({diff:.3g})")
        x0 = bank[:n]
        res = {}
        for trig, fb in RULES:
            rate, coin = None, None
            if trig.startswith("random:"):
                rate = res[key(trig[7:], fb)]["fire_share"]
                coin = coin_seed(rep, world, actor, s, fb, trig[7:])
            r = run_rule(ctx, sig, x0, fb, trig, rate, coin)
            res[key(trig, fb)] = r
            C.log(f"  {s} {trig:20s} {str(fb):7s} fire {r['fire_share']:.3f} chg {r['changed_share']:.3f} "
                  f"recall80 {r['recall_past80']:.3f} prec80 {r['precision_past80']:.3f} ({r['seconds']:.1f}s)")
        sit = dict(gate_check=gate, seconds=time.perf_counter() - t1)
        if a.smoke:
            eq = {}
            for mine, fb, theirs in EQUIV_RULES:
                coin = [20261013, 999, TC.SITUATIONS.index(s)] if theirs == "random95" else None
                rate = EQUIV_RATE if theirs == "random95" else None
                m = run_rule(ctx, sig, x0, fb, mine, rate, coin)
                f = FB.run_rule(ctx, x0, fb, theirs, rate, coin)
                eq[f"{mine}:{fb}"] = dict(
                    max_abs_diff_per_start_points=float(np.max(np.abs(np.asarray(m["per_start_points"]) -
                                                                      np.asarray(f["per_start_points"])))),
                    abs_diff_J=float(abs(m["J"] - f["J"])), same_fire_share=bool(m["fire_share"] == f["fire_share"]))
            sit["equivalence_with_fallback_run_rule"] = eq
            C.log(f"  {s}: equivalence {eq}")
            res = {k: blind(v) for k, v in res.items()}
        sit["rules"] = res
        out["situations"][s] = sit
        out["seconds"] = time.perf_counter() - started
        C.write_json(path, out)
    if final != path:
        C.write_json(final, out)
    C.log(f"wrote {final} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
