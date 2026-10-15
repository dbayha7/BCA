"""Step 2: exact returns of one seed-world's locked actors. EXPLORATORY; see PRESPEC.md.

Verifies the lock (its sidecar, its record checks and every actor's SHA-256), then, for every scored actor, computes
exactly what version 6's policy evaluation computes: acceptance._evaluate (learners.evaluate_actor: the full
1,000-start banks of all four situations, exact 100-step returns, points against the gate's anchors, the same actor
capped at 80%, the share of steps past 80%), and from one more exact rollout per situation the share of visited steps
past 80% per thruster and the share of visited steps whose throttle index lies above this seed-world's t_lo. Writes
values.json (+ .sha256), once.

Official mode refuses until EVERY seed-world (seeds 0-4, both worlds) has a lock.json or a stop.json: no return of
any trained arm is computed before every lock exists. It must run pinned to one CPU.

Usage: taskset -c 0 python evaluate_policy.py --seed 0 --world stall [--out-root DIR (scratch)]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2lib as S  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--world", choices=S.WORLDS, required=True)
    ap.add_argument("--out-root", default=None, help="scratch mode: the scratch folder the training wrote")
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    official = a.out_root is None
    spec, spec_sha = S.load_prespec()
    man, man_sha = S.verify_manifest()
    seed = S.check_seed(a.seed)
    cpu = S.require_one_cpu("the evaluation")
    S.check_environment(spec)
    root = S.OFFICIAL_OUT if official else Path(a.out_root)
    d = S.guard_out(S.seed_dir(root, seed, a.world), official)
    if (d / "values.json").exists():
        raise SystemExit(f"refused: {d}/values.json exists (an evaluation is computed once)")
    if (d / "stop.json").exists():
        raise SystemExit(f"refused: {d} stopped (stop.json); there is nothing to evaluate")
    if official:
        missing = [f"{w}/seed{s}" for s in S.SEEDS for w in S.WORLDS
                   if not ((S.seed_dir(root, s, w) / "lock.json").exists()
                           or (S.seed_dir(root, s, w) / "stop.json").exists())]
        if missing:
            raise SystemExit(f"refused: no return is computed before every seed-world is locked or stopped; missing "
                             f"{missing}")
    import machinery as MC  # noqa: E402
    if MC.SPEC_SHA != spec_sha:
        raise SystemExit("refused: prespec.json changed while starting")
    lock, acts, lock_sha = MC.verify_lock(d, spec_sha, man_sha, seed, a.world, official,
                                          spec["protocol"]["actor_steps"])
    ctx = MC.context(seed, a.world)
    prep = ctx.prepared(a.world)
    an = ctx.anchors(a.world)
    t_lo = float(lock["arm_table"]["t_lo"])
    t0 = time.perf_counter()
    ev, vs = {}, {}
    for name in lock["scored_names"]:
        finite = bool(lock["finite"][name])
        ev[name] = MC.A._evaluate(acts[name], prep, an, ctx.cfg, failed=not finite)
        vs[name] = MC.visit_shares(acts[name], prep, t_lo) if finite else None
        MC.log(f"evaluated {name}")
    sits = {}
    for s in MC.TC.SITUATIONS:
        sits[s] = dict(
            points={k: None if ev[k] is None else ev[k][s]["points"] for k in ev},
            capped_points={k: None if ev[k] is None else ev[k][s]["capped_points"] for k in ev},
            unrecorded_throttle_share={k: None if ev[k] is None else ev[k][s]["unrecorded_throttle_share"]
                                       for k in ev},
            past_80_share_per_thruster={k: None if vs[k] is None else vs[k][s]["past_80_share_per_thruster"]
                                        for k in vs},
            past_t_lo_share={k: None if vs[k] is None else vs[k][s]["past_t_lo_share"] for k in vs})
    rec = dict(label=S.LABEL, kind="step2-policy-values", mode="official" if official else "scratch", seed=seed,
               world=a.world, prespec_sha256=spec_sha, code_manifest_sha256=man_sha, lock_sha256=lock_sha,
               arm_ids=lock["arm_ids"], t_lo=t_lo, evaluations=ev, situations=sits,
               seconds=dict(evaluation=time.perf_counter() - t0, total=time.perf_counter() - t_start),
               vmhwm_mb=S.vmhwm_mb(), about="exact returns of the locked actors, computed after every lock",
               **MC.machine_record())
    S.write_json_once(d / "values.json", rec, official)
    sc = sits["strong_current"]["points"]
    MC.log(f"{a.world}/seed {seed}: values written; strong current none {sc.get('none')}, obca_k1 "
           f"{sc.get('obca_k1')}; {time.perf_counter() - t_start:.0f}s; VmHWM {S.vmhwm_mb():.0f} MB")
    return rec


if __name__ == "__main__":
    main()
