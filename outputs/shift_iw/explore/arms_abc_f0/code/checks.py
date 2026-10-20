"""EXPLORATORY: arms A, B, C and F0 -- the unit checks that need the sandbox's real seed-0 inputs (frozen pass marks,
section 8 step 1), beside the synthetic ones in test_arms.py. Trains no arm and writes no record; writes only
routes2/arms/state/checks_real.json.

  F0      KF3 (same clone as A2), KF1 (in-graph no-credit flags equal refresh's numpy flags at the clone's and at the
          result F rerun's no-caution actor's commands) and KF2 (a) and (b) (300 steps each), on medium-expert and
          expert, seed 0
  B and C the all-1.00 map reproduces learners_iw.evaluate_actor bit for bit on all three start banks (1,000 fixed
          starts each), with the expert seed-0 gapped host as the actor

  python -B checks.py
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402

import common as C  # noqa: E402


def main():
    from experiments.shift_iw import learners_iw as LI
    import arm_f0
    C.require_passmarks()
    out = dict(kind="arms-real-unit-checks", label=C.label(), checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               checks={})
    for q in C.F0_QUALITIES:
        t0 = time.perf_counter()
        ch = arm_f0.pretraining_checks(q, 0)
        ch["seconds"] = time.perf_counter() - t0
        out["checks"][f"F0_{q}_seed0"] = ch
        C.log(f"F0 {q} seed 0: KF3 {ch['KF3']['passes']}, KF1 {ch['KF1']['passes']} (disagreement "
              f"{ch['KF1']['disagreement']}), KF1 no-caution {ch.get('KF1_no_caution', {}).get('passes')}, KF2 (a) "
              f"{ch['KF2']['a']['passes']} (b) {ch['KF2']['b']['passes']}")
    S = C.Seed("expert", 0)
    t0 = time.perf_counter()
    ref = LI.evaluate_actor(S.host.actor.params, S.prep, S.banks, S.anchors)
    mine = C.evaluate_policy(C.ceiling_policy(C.actor_policy(S.host.actor.params, S.prep), [1.0] * 4), S.banks,
                             S.anchors)
    same = json.dumps(ref, sort_keys=True) == json.dumps(mine, sort_keys=True)
    out["checks"]["all_ones_map_three_banks"] = dict(passes=same, banks=sorted(ref), starts=[v["starts"] for v in
                                                                                          ref.values()],
                                                     seconds=time.perf_counter() - t0,
                                                     note="the host's returns themselves are not recorded here")
    C.log(f"all-1.00 map reproduces evaluate_actor bit for bit on the three banks: {same}")
    out["all_passed"] = all(v.get("passes") for v in out["checks"].values())
    C.write_atomic(C.STATE / "checks_real.json", json.dumps(C.plain(out), indent=1))
    C.log(f"real-data unit checks: {'all passed' if out['all_passed'] else 'FAILED'} -> {C.STATE / 'checks_real.json'}")
    return 0 if out["all_passed"] else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.Refused as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
