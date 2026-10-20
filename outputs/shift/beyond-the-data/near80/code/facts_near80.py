"""Near-80 (version 9, main setting): the outcome-free counts of the pooled groups on the gate's seeds 0-4 (the
freeze's step 7, PREREGISTRATION.md section 8.2), and the anchors of the fixed test start sets. DRAFT, not frozen.

No learner is trained, no BCA band is fitted, no outcome of any learner is read. Seeds 0-4 only: this script refuses
every other seed (the fresh seeds are never generated, loaded or read by it). It reads the gate's verified logs
through acceptance.load_seed (which regenerates each log and checks it, its ball and both gate removals against
runs/shift/pre/data/base/<world>/seed<k>/record.json, read only), reads step 1's gap rule (verified bytes) on the
ball-kept training commands and counts the pooled groups (near80lib.pooled_count_record). Writes one JSON,
write-once, outside the repository or under the official output folder.

Usage (from /tmp, one CPU):
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=<repo> \
    python <repo>/outputs/shift/beyond-the-data/near80/code/facts_near80.py --out OUT.json \
      [--seeds 0 1 2 3 4] [--worlds stall no_stall]

Cost on seeds 0-4, both worlds: about 13 s per seed-world (the log is regenerated).
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

import near80_constants as K  # noqa: E402
import near80lib as N  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

RUNS = N.ROOT / "runs" / "shift"
KIND = "near80-v9-pooled-group-counts"


def seed_world_counts(log, gap, world, removal, step1):
    """The outcome-free count record of one seed-world (near80lib.pooled_count_record) with its identifiers."""
    rec = N.pooled_count_record(log, gap, removal, step1)
    rec.update(rep=int(log.meta["rep"]), world=world, kept_training_rows=int(len(removal.kept_train_rows(log))),
               bank_rows_kept=int(len(removal.bank_rows(log))),
               calibration_rows=int(len(log.split_rows("calibration"))),
               evaluation_rows=int(len(log.split_rows("evaluation"))))
    return rec


def _bank(situation, n):
    return TC.bank_starts(situation) if n is None else TC.bank_starts(situation, int(n))


def anchors_for(world, spec=RC.DEFAULT_SPEC, bank_n=None):
    """{situation: (J of the recording controller on the bank, J of the world's top controller)}, exact. The
    points' anchors, as the gate computes them (acceptance.anchors_of) and as stub mode replaces them."""
    from experiments.shift import truth as TR
    physics = TC.world_physics(world)
    top = RC.stall_aware(physics) if world == "stall" else RC.rising_thrust(physics)
    n = TC.BANK_SIZE if bank_n is None else int(bank_n)
    out = {}
    for s in TC.SITUATIONS:
        x0 = _bank(s, bank_n)
        out[s] = (float(TR.rollout(TR.recorder_on_bank(physics, spec, n), x0, physics).J.mean()),
                  float(TR.rollout(top, x0, physics).J.mean()))
    return out


def summary(results):
    """The smallest and largest pooled full-throttle group per world, and how many seed-worlds reach CERT_MIN_ROWS."""
    out = {}
    for w in K.WORLDS:
        rows = [r for r in results if r["world"] == w and r.get("groups")]
        sizes = [r["groups"]["n_full_throttle"] for r in rows]
        out[w] = dict(seed_worlds=len(rows), n_full_throttle_min=min(sizes) if sizes else None,
                      n_full_throttle_max=max(sizes) if sizes else None,
                      reach_min_rows=sum(1 for r in rows if r["certification"]["reaches_min_rows"]),
                      version8_min=min((r["version8_groups"]["n_full_throttle"] for r in rows), default=None),
                      version8_max=max((r["version8_groups"]["n_full_throttle"] for r in rows), default=None))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--worlds", nargs="+", default=["stall", "no_stall"], choices=["stall", "no_stall"])
    a = p.parse_args(argv)
    seeds = N.check_seeds(a.seeds)
    out = N.guard_out(a.out)
    from experiments.shift import acceptance as AC
    t0 = time.perf_counter()
    step1 = N.load_step1lib()
    results = []
    for world in a.worlds:
        for rep in seeds:
            t1 = time.perf_counter()
            sd = AC.load_seed(RUNS, None, "base", world, rep)
            f = seed_world_counts(sd.log, sd.gap, world, sd.removals["gap"], step1)
            f["seconds"] = time.perf_counter() - t1
            results.append(f)
            g = f.get("groups") or {}
            print(f"{world} seed {rep}: pooled full-throttle group {g.get('n_full_throttle')} "
                  f"(version 8's {((f.get('version8_groups') or {}).get('n_full_throttle'))}), rest {g.get('n_rest')}, "
                  f"{f['seconds']:.0f} s", flush=True)
    payload = dict(kind=KIND, status=N.STATUS, label=K.LABEL, seeds=seeds, worlds=a.worlds, results=results,
                   summary=summary(results), min_rows=K.CERT_MIN_ROWS, seconds=time.perf_counter() - t0,
                   note="outcome-free: commands and split labels only; nothing trained, no reward or critic read")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf8")
    print(f"wrote {out}")
    return payload


if __name__ == "__main__":
    main()
