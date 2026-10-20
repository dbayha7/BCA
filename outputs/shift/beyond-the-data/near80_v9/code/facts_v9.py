"""Near-80 (version 9, DRAFT): the no-training facts and the no-training checks (PREREGISTRATION.md section 8.1).

No learner is trained, no outcome of any learner is read, and no reward or next state is read: everything here reads
commands and split labels only. Seeds 0-4 (and, with --spent, version 8's spent seeds 10-14, for information); this
script refuses seeds 5-9 and every seed of version 9's pool (35-49). It reads the gate's verified logs through
acceptance.load_seed (seeds 0-4: it regenerates each log and checks it, its ball and both gate removals against the
gate's data record) or regenerates a spent seed with the gate's generator, builds BCA's two pooled groups
(near80lib.outcome_free_count), checks them (near80lib.group_checks) and, on seeds 0-4, checks that the as-built
bank-only groups reproduce step 1's recorded group sizes. Writes one JSON, write-once, outside the repository or under
runs/shift_explore/near80_v9/.

Usage (from /tmp, one CPU):
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \
    python /home/user/bca/outputs/shift/beyond-the-data/near80_v9/code/facts_v9.py --out OUT.json \
      [--seeds 0 1 2 3 4] [--worlds stall no_stall] [--spent]

Cost on seeds 0-4, both worlds: about 15 s per seed-world (the log is regenerated).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80lib as N  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402

RUNS = N.ROOT / "runs" / "shift"
GAP_RULE = dict(upper_quantile=0.99, min_gap=0.05, min_tail=9)      # near80_constants.GAP_RULE (test_v9 checks)


def gate_checks_of(record, world):
    """The gate's data checks as its data record holds them (seeds 0-4): the ball's leak checks and, in the stalling
    world, the five world checks T1-T5."""
    checks = record.get("checks") or {}
    leak = bool(checks.get("gap_rows_in_gapped_training") == 0 and checks.get("gap_rows_in_gapped_bank") == 0
                and all(v for v in checks.values() if isinstance(v, bool)))
    marks = record.get("t1_t5") if world == "stall" else None
    passed = bool(record.get("passed") and leak and (all((marks or {}).values()) if world == "stall" else True))
    return dict(ball_leak_checks_passed=leak, world_marks=None if marks is None else {k: bool(v) for k, v in
                                                                                       marks.items()},
                passed=passed)


def step1_groups(world, seed, s1=None):
    """Step 1's recorded group sizes of one seed-world (seeds 0-4), or None."""
    if s1 is None:
        try:
            s1 = json.loads((N.STEP1_DIR / "results" / "results.json").read_text(encoding="utf8"))
        except (OSError, ValueError):
            return None
    e = (s1.get("per_seed_world") or {}).get(f"{world}/seed{int(seed)}")
    if not e:
        return None
    g = e.get("groups") or {}
    return dict(n_survivors=e.get("n_survivors"), n_tail=g.get("n_full_throttle"), n_rest=g.get("n_rest"),
                t_lo=e.get("t_lo"), t_up=e.get("t_up"))


def seed_world_facts(log, gap, world, step1, gate_checks, s1_groups=None):
    """Every no-training fact of one seed-world: the gate's checks, the outcome-free count of BCA's pooled groups (with
    its group checks), and, when step 1's recorded sizes are given, whether the bank-only groups reproduce them."""
    gapped = SH.make_removal(log, "gap", gap)
    kept = gapped.kept_train_rows(log)
    count = N.outcome_free_count(log, gap, kept, step1, GAP_RULE)
    f = dict(rep=int(log.meta["rep"]), world=world, gate_checks=gate_checks, count=count,
             gap=dict(centre=list(gap.centre), rho=gap.rho, mass=gap.mass))
    if s1_groups is not None and count["defined"]:
        mine = dict(n_survivors=count["n_survivors"], n_tail=count["bank_only"]["n_tail"],
                    n_rest=count["bank_only"]["n_rest"], t_lo=count["t_lo"], t_up=count["t_up"])
        f["step1_groups"] = s1_groups
        f["bank_only_groups"] = mine
        f["bank_only_reproduces_step1"] = bool(all(mine[k] == s1_groups[k] for k in mine))
    elif s1_groups is not None:
        f["bank_only_reproduces_step1"] = False
    else:
        f["bank_only_reproduces_step1"] = None
    f["diagnostic_reads"] = int(log.diagnostic_reads)
    return f


def _bank(situation, n):
    from experiments.shift import thruster_chain as TC
    return TC.bank_starts(situation) if n is None else TC.bank_starts(situation, int(n))


def anchors_for(world, spec=None, bank_n=None):
    """{situation: (J of the recording controller on the bank, J of the world's top controller)}, exact (facts_near80's
    anchors_for, restated; the stub mode's anchors)."""
    from experiments.shift import recorder as RC
    from experiments.shift import thruster_chain as TC
    from experiments.shift import truth as TR
    spec = RC.DEFAULT_SPEC if spec is None else spec
    physics = TC.world_physics(world)
    top = RC.stall_aware(physics) if world == "stall" else RC.rising_thrust(physics)
    n = TC.BANK_SIZE if bank_n is None else int(bank_n)
    out = {}
    for s in TC.SITUATIONS:
        x0 = _bank(s, bank_n)
        out[s] = (float(TR.rollout(TR.recorder_on_bank(physics, spec, n), x0, physics).J.mean()),
                  float(TR.rollout(top, x0, physics).J.mean()))
    return out


def spent_seed(seed, world):
    """A spent seed (10-14) regenerated with the gate's generator and ball (commands only are read)."""
    from experiments.shift import acceptance as AC
    from experiments.shift import dataset as DS
    p = AC.load_prereg(AC.refuse_protected(AC.RUNS_DEFAULT))
    spec = AC.stage0_spec(p.runs, "base")
    lg = DS.generate(int(seed), world, spec, tuple(p.config.sizes), float(p.config.slope("base")))
    gap, _ = SH.define_gap(lg)
    return lg, gap


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=list(N.EXPLORATORY_SEEDS))
    p.add_argument("--worlds", nargs="+", default=["stall", "no_stall"], choices=["stall", "no_stall"])
    p.add_argument("--spent", action="store_true", help="also allow version 8's spent seeds 10-14 (information only)")
    a = p.parse_args(argv)
    seeds = N.check_seeds(a.seeds, allow_spent=a.spent)
    out = N.guard_out(a.out)
    from experiments.shift import acceptance as AC
    t0 = time.perf_counter()
    step1 = N.load_step1lib()
    s1 = json.loads((N.STEP1_DIR / "results" / "results.json").read_text(encoding="utf8"))
    results = []
    for world in a.worlds:
        for rep in seeds:
            t1 = time.perf_counter()
            if rep in N.EXPLORATORY_SEEDS:
                sd = AC.load_seed(RUNS, None, "base", world, rep)
                lg, gap, gc, sg = sd.log, sd.gap, gate_checks_of(sd.record, world), step1_groups(world, rep, s1)
            else:
                lg, gap = spent_seed(rep, world)
                gc, sg = dict(passed=gap is not None, note="a spent seed: the gate's ball only (information)"), None
            f = seed_world_facts(lg, gap, world, step1, gc, sg)
            f["seconds"] = time.perf_counter() - t1
            results.append(f)
            c = f["count"]
            print(f"{world} seed {rep}: full-throttle group {c.get('n_tail')} rows (rest {c.get('n_rest')}); "
                  f"certifiable {c.get('certifiable')}; {f['seconds']:.0f} s", flush=True)
    explo = [f for f in results if f["rep"] in N.EXPLORATORY_SEEDS]
    payload = dict(status=N.STATUS, label="EXPLORATORY no-training facts (commands only), seeds "
                                          + ", ".join(str(s) for s in seeds),
                   seeds=seeds, worlds=a.worlds, results=results, no_training_checks=N.no_training_checks(explo),
                   seconds=time.perf_counter() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"wrote {out}; no-training checks {'PASSED' if payload['no_training_checks']['passed'] else 'NOT PASSED'}"
          f" (not run: {payload['no_training_checks']['not_run'] or 'none'})")
    return payload


if __name__ == "__main__":
    main()
