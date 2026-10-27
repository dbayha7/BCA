"""EXPLORATORY arm C: the guarded release (frozen pass marks, section 6).

Half the pull toward the recorded commands (cloning weight 0.5, the capped pull's opposite), with B's ceiling rule as
the guard against stalling. There is no BCA in training; the release is a change of TD3+BC's alpha and is never
credited to BCA.

  actors  one learners.ActorOnly program from the clone with two even arms, rho 1 (Arm('none', 'even', 1.0)) and
          rho 0.5 (Arm('release_x0.5', 'even', 0.5)); ActorOnly.run's chunk loop is restated only to save the 8,000-step
          actors (unit-tested equal to ActorOnly.run bit for bit)
  guard   B's ceiling rule built for C's own final rho 0.5 actor (try selection with that actor; gains unchanged: host
          critic, host continuation), estimated strong-mix weights, every rival built the same way; C's maps use B's
          random streams with 10 added to the rule tag (BCA 10, unweighted 11, shuffled 12, exact 13, shifted grid 14;
          the shuffle's permutation stream [20261016, seed, 4312, k] has no rule tag and is B's). Information: a BCA map
          built for rho 1's final actor (tag 10 as well) and B's host-built map (read from B's record at scoring)

Kill rules computed here: KB1 at 8,000 steps (stops rho 0.5 on the quality: non-finite actor, or critic gain
G(rho 0.5) > 3 G(rho 1) when G(rho 1) > 0, else > 3 s with s arm A's plain residual spread); KB0's per-seed shares
(thruster-1 commands past float32(0.8) in each final actor's unclipped strong-start rollouts; judged by screen.py and
kills.py); the guard-void quantity (C's own thruster-1 ceiling >= 0.85 with >= 30 tries in rung (0.80, 0.85]); K3.

  python -B arm_c.py --quality Q --seed K [--dry]
Writes <sandbox>/repo/runs/shift_iw_v5_explore/arms_abc_f0/C/<Q>/seed<K>/{actors_8000.msgpack, actors_final.msgpack,
record.json (+ .sha256)}. No return of a C actor is recorded here (the rollouts give commands only).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import arm_b as B  # noqa: E402
import common as C  # noqa: E402

RHO = (("rho1", 1.0, "none"), ("rho0.5", 0.5, "release_x0.5"))
TAG_OFFSET = 10
STEP_8000_CHUNK = 1


def arms():
    from experiments.shift import learners as LR
    return [LR.Arm(name, "even", value) for _, value, name in RHO]


def train(runner, critic, train_rows, save_8000=None, check_8000=None, log=None):
    """ActorOnly.run's loop for even arms (weighted program, tables from ActorOnly.tables), restated only to hand the
    8,000-step actors to save_8000 and to stop when check_8000 returns a reason. Returns (stacked final params,
    info, stopped reason or None)."""
    import jax
    import jax.numpy as jnp
    A = arms()
    program = runner.program(len(A), True, True)
    states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (len(A),) + jnp.shape(x)), runner.start)
    tables = jnp.asarray(runner.tables(A, {}))
    metrics = []
    for c in range(runner.steps // runner.chunk):
        states, aux = program(critic, states, train_rows, runner.rows[c], tables)
        metrics.append(np.asarray(aux, np.float64).mean(1).tolist())
        if log:
            log(f"C: step {(c + 1) * runner.chunk}/{runner.steps}")
        if c == STEP_8000_CHUNK:
            if save_8000 is not None:
                save_8000(states.params)
            if check_8000 is not None:
                reason = check_8000(states.params)
                if reason:
                    return states.params, dict(chunk_metrics=metrics), reason
    return states.params, dict(chunk_metrics=metrics, rows_sha256=runner.rows_sha256), None


def kb1(G1, G05, s, finite05):
    """KB1, critic exploitation at 8,000 steps (reading 14 of section 10)."""
    if not finite05:
        return dict(fires=True, reason="the rho 0.5 actor is not finite", G_rho1=G1, G_rho05=G05, s=s)
    bar = 3.0 * G1 if G1 > 0 else 3.0 * s
    return dict(fires=bool(G05 > bar), G_rho1=G1, G_rho05=G05, s=s, bar=bar,
                bar_rule="3 G(rho 1)" if G1 > 0 else "3 s (G(rho 1) <= 0)")


def run_job(quality, seed, dry=False):
    from experiments.shift import learners as LR
    digest = C.require_passmarks()
    started = time.perf_counter()
    d = C.arm_dir("C", quality, seed)
    try:
        S = C.Seed(quality, seed)
    except C.InputFailed as error:     # the host's critic gives C's gains: non-finite gains are K3 (section 6.4)
        if dry:
            raise
        return C.write_stop(d / "record.json", C.base_record("C", quality, seed, digest), "K3", str(error), started,
                            kills=dict(K3=dict(fires=True, reasons=[str(error)])))
    rec = C.base_record("C", quality, seed, digest)
    if dry:
        tries = B.Tries(S)
        C.log(f"C {quality} seed {seed} DRY: inputs present; {tries.n} tries; nothing trained or written")
        return 0

    def stop(rule, reason, **extra):
        rec.update(status="stopped", stopped_by=rule, reason=reason, seconds=time.perf_counter() - started,
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        C.write_record(d / "record.json", rec)
        C.log(f"C {quality} seed {seed}: STOPPED by {rule}: {reason}")
        return 0

    clone_params, clone_rec = LR.clone(S.prep, S.cfg, seed)
    if clone_rec.get("failed"):
        return stop("K3", "the cloned start actor is not finite", clone=clone_rec)
    runner = LR.ActorOnly(S.prep, clone_params, S.cfg, seed, alpha=S.cfg.alpha)
    critic = S.host.critic.params
    q_rec = C.q_min(critic, S.obs, S.a_rec)
    s, s_info = C.plain_spread(S.host, S.prep, S.cfg, seed)
    rec.update(clone=clone_rec, start_sha256=LR.tree_sha256(clone_params), arms=[n for n, _, _ in RHO],
               cloning_weights={n: v for n, v, _ in RHO}, spread=s_info,
               inputs_sha256=S.links([S.host_record_path()]))
    kills = {}

    def save_8000(stacked):
        p8 = {n: runner.arm_params(stacked, i) for i, (n, _, _) in enumerate(RHO)}
        rec["actors_8000_sha256"] = LR.save_tree(d / "actors_8000.msgpack", p8)

    def check_8000(stacked):
        p8 = {n: runner.arm_params(stacked, i) for i, (n, _, _) in enumerate(RHO)}
        G = {}
        for n in ("rho1", "rho0.5"):
            pi = C.actor_out(p8[n], S.obs)
            G[n] = float(np.mean(C.q_min(critic, S.obs, pi) - q_rec)) if np.all(np.isfinite(pi)) else float("nan")
        fin05 = C.finite_tree(p8["rho0.5"]) and np.isfinite(G["rho0.5"])
        kills["KB1"] = kb1(G["rho1"], G["rho0.5"], s, fin05)
        if not C.finite_tree(p8["rho1"]):
            kills["K3"] = dict(fires=True, reason="the rho 1 actor is not finite at 8,000 steps")
            return "K3"
        return "KB1" if kills["KB1"]["fires"] else None

    stacked, info, stopped = train(runner, critic, S.prep.train, save_8000, check_8000,
                                   lambda m: C.log(f"C {quality} seed {seed}: {m}"))
    rec.update(training=info)
    if stopped:
        return stop(stopped, kills.get(stopped, {}).get("reason") or
                    "KB1: the rho 0.5 actor's critic gain exceeds its bar at 8,000 steps (C stops on this quality)",
                    kills=kills)
    final = {n: runner.arm_params(stacked, i) for i, (n, _, _) in enumerate(RHO)}
    rec["actors_final_sha256"] = LR.save_tree(d / "actors_final.msgpack", final)
    fin = {n: C.finite_tree(p) for n, p in final.items()}
    if not all(fin.values()):
        return stop("K3", f"non-finite final actor: {[n for n, f in fin.items() if not f]}", kills=kills, finite=fin)
    tries = B.Tries(S)
    pols = {n: C.actor_policy(p, S.prep, name=f"{quality} seed {seed} {n} (float64 copy)") for n, p in final.items()}
    cmds = {n: C.strong_commands(pols[n], S.banks) for n in final}       # commands only; no return is kept
    pi05 = C.actor_out(final["rho0.5"], tries.obs)
    pi1 = C.actor_out(final["rho1"], tries.obs)
    try:
        maps, details = B.maps_for(tries, pi05, TAG_OFFSET, commands=cmds["rho0.5"], range_t=B.range_limit(S.prep))
        maps1, details1 = B.maps_for(tries, pi1, TAG_OFFSET, commands=cmds["rho1"], rules=("bca",))
    except B.NonFinite as error:
        return stop("K3", f"non-finite values: {error}", kills=dict(kills, K3=dict(fires=True, reasons=error.reasons)))
    knee = float(C.KNEE32)
    share = {n: C.past_share(cmds[n], [knee] * 4, thruster=0) for n in final}
    kills["KB0_seed"] = dict(share_rho05=share["rho0.5"], share_rho1=share["rho1"],
                             passes=bool(share["rho0.5"] >= 0.02 and share["rho0.5"] >= 2.0 * share["rho1"]),
                             note="judged on at least 2 of random, medium-replay, medium (screen.py, kills.py)")
    k1 = B.k1_numbers(details, maps)
    kills["guard_void_own_map"] = dict(t1=k1["t1"], tries_080_085=k1["tries_080_085"], fires=k1["fires"])
    k3 = [k for k, v in dict(details["finite"], commands=all(bool(np.all(np.isfinite(c))) for c in cmds.values())
                             ).items() if not v]
    kills["K3"] = dict(fires=bool(k3), reasons=k3)
    rec.update(status="finished", stopped_by=None, maps=maps, map_names=B.DISTINCTS, details=details,
               rho1_bca_map=maps1["bca"], rho1_details=details1, tries=tries.describe(), kills=kills, finite=fin,
               seconds=time.perf_counter() - started, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    C.write_record(d / "record.json", rec)
    C.log(f"C {quality} seed {seed}: guard map {maps['bca']}, KB0 shares {share}, KB1 "
          f"{kills.get('KB1', {}).get('fires')}; {rec['seconds']:.0f}s -> {d / 'record.json'}")
    return 0


def load_actors(quality, seed, which="final"):
    from experiments.shift import learners as LR
    d = C.arm_dir("C", quality, seed)
    rec = C.arm_record(d / "record.json")
    key = "actors_final_sha256" if which == "final" else "actors_8000_sha256"
    return rec, LR.load_tree(d / f"actors_{which}.msgpack", {n: LR.actor_template() for n, _, _ in RHO}, rec[key])


def main(argv=None):
    p = argparse.ArgumentParser(description="EXPLORATORY arm C (guarded release): one quality and seed")
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
