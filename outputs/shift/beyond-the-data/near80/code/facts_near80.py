"""Near-80 version (version 8): the no-training facts and the no-training checks (PREREGISTRATION.md section 8.1).

No learner is trained, no BCA band is fitted, no outcome of any learner is read. Seeds 0-4 only: this script
refuses seeds 5-9 and 10-14. It reads the gate's verified logs through acceptance.load_seed (which regenerates each
log and checks it, its ball and both gate removals against runs/shift/pre/data/base/<world>/seed<k>/record.json,
read only), builds the near-80 removal and its random-rows control (near80lib), runs the leak checks, counts rows,
reads step 1's gap rule (verified bytes), restates the gate's world card for each removal and, with --rollouts, runs
fixed scripted controllers exactly on the four fixed test start sets (stall-aware, textbook gain held to the band's
lower edge, textbook gain clipped to each removal's recorded range). Writes one JSON, write-once, outside the
repository or under runs/shift_explore/near80/.

Usage (from /tmp, one CPU):
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \
    python /home/user/bca/outputs/shift/beyond-the-data/near80/code/facts_near80.py --out OUT.json \
      [--seeds 0 1 2 3 4] [--worlds stall no_stall] [--rollouts]

Cost on seeds 0-4, both worlds: about 13 s per seed-world for the counts (the log is regenerated), plus about
2-3 CPU-minutes in all with --rollouts.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80lib as N  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

RUNS = N.ROOT / "runs" / "shift"


def _gap_rule(step1, actions):
    g = step1.gap_rule(step1.throttle(actions))
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in g.items() if k != "tail"}


def seed_world_facts(log, gap, world, physics, spec, step1, rec=None, lo=N.BAND_LO, hi=N.BAND_HI, scope=N.SCOPE,
                     anchors=None, bank_n=None):
    """Every no-training fact of one seed-world. rec: the gate's data record (for the reproduction check); None
    compares with shift.make_removal's own gap removal instead. anchors: {situation: (J_recorder, J_top)} of this
    world, to run the data-edge rollouts (stalling world only); None skips them."""
    acts = log["actions"]
    train, cal, ev = log.split_rows("train"), log.split_rows("calibration"), log.split_rows("evaluation")
    bank = log.bank_rows
    ball = gap.contains(acts)
    band = N.in_band(acts, lo, hi, scope)
    base = SH.make_removal(log, "gap", gap)
    near = N.band_removal(log, gap, lo, hi, scope)
    rnd = N.random_rows_removal(log, gap, near, spec, physics)
    empty = N.band_removal(log, gap, lo, hi, scope, empty=True)
    k_base, k_near, k_rnd = base.kept_train_rows(log), near.kept_train_rows(log), rnd.kept_train_rows(log)
    t_near = step1.throttle(acts[k_near])
    want = (rec["removals"]["gap"] if rec is not None else base.summary(log))
    es = empty.summary(log)
    f = dict(rep=int(log.meta["rep"]), world=world, band=[lo, hi], scope=scope,
             gap=dict(center=list(gap.centre), rho=gap.rho, mass=gap.mass),
             leak_check_band=N.leak_check(log, near, gap, lo=lo, hi=hi, scope=scope),
             leak_check_random=N.leak_check(log, rnd, gap, lo=lo, hi=hi, scope=scope, band_rm=near, spec=spec,
                                            physics=physics),
             reproduces_gate_removal_with_empty_band=bool(es["kept_rows_sha256"] == want["kept_rows_sha256"]
                                                         and es["bank_rows_sha256"] == want["bank_rows_sha256"]))
    f["training_split"] = dict(
        rows=int(len(train)), ball_removed=int(base.removed[train].sum()), band_rows=int(band[train].sum()),
        band_and_ball=int((band & ball)[train].sum()), band_only=int((band & ~ball)[train].sum()),
        band_rows_per_thruster=[int(v) for v in ((N.throttle_abs(acts[train]) >= lo)
                                                 & (N.throttle_abs(acts[train]) <= hi)).sum(0)])
    f["ball_alone"] = dict(kept_rows=int(len(k_base)), bank_rows_kept=int((~base.bank_removed).sum()),
                           recorded_range=N.recorded_range(acts[k_base]), gap_rule=_gap_rule(step1, acts[k_base]))
    f["near80"] = dict(
        removed=int(near.count), extra=int(near.count - base.count),
        extra_share_of_ball_kept=float((near.count - base.count) / len(k_base)), kept_rows=int(len(k_near)),
        bank_rows_kept=int((~near.bank_removed).sum()), bank_extra=int(near.bank_removed.sum() - base.bank_removed.sum()),
        recorded_range=N.recorded_range(acts[k_near]), gap_rule=_gap_rule(step1, acts[k_near]),
        max_throttle_index_below_band=float(t_near[t_near < lo].max()),
        kept_rows_past_knee=int(TC.past_knee(np.asarray(acts[k_near], np.float64)).sum()),
        kept_rows_sha256=near.summary(log)["kept_rows_sha256"], bank_rows_sha256=near.summary(log)["bank_rows_sha256"])
    f["random_rows"] = dict(
        removed=int(rnd.count), kept_rows=int(len(k_rnd)), bank_rows_kept=int((~rnd.bank_removed).sum()),
        recorded_range=N.recorded_range(acts[k_rnd]), gap_rule=_gap_rule(step1, acts[k_rnd]),
        kept_rows_in_band=int(band[k_rnd].sum()),
        kept_rows_sha256=rnd.summary(log)["kept_rows_sha256"], bank_rows_sha256=rnd.summary(log)["bank_rows_sha256"])
    f["calibration_split"] = dict(rows=int(len(cal)), band_rows=int(band[cal].sum()),
                                  bank_band_rows=int(band[bank].sum()), bank_band_only=int((band & ~ball)[bank].sum()))
    f["evaluation_split"] = dict(rows=int(len(ev)), band_rows=int(band[ev].sum()),
                                 band_rows_in_ball=int((band & ball)[ev].sum()))
    if world == "stall":
        from experiments.shift import acceptance as AC
        card, marks = AC.world_marks(log, gap, spec, physics)
        f["gate_world_marks_on_ball"] = {k: bool(v) for k, v in marks.items()}
        f["gate_world_card_on_ball"] = {k: float(card[k]) for k in ("full_throttle_commands_removed",
                                                                    "kept_past80_share", "identical_rule_share",
                                                                    "region_rows_kept")}
        f["world_card_ball"] = N.world_card(log, base.removed[train], spec, physics)
        f["world_card_near80"] = N.world_card(log, near.removed[train], spec, physics)
        f["world_card_random_rows"] = N.world_card(log, rnd.removed[train], spec, physics)
        region = RC.Recorder(spec, physics).in_region(log["infos/state"][train])
        kept_region = region & ~base.removed[train]
        f["region_band_share_of_ball_kept"] = float(band[train][kept_region].mean()) if kept_region.any() else 0.0
        if anchors is not None:
            from experiments.shift import truth as TR
            K = RC.textbook_gain(physics)
            pts = {}
            for name, rows in (("ball_alone", k_base), ("near80", k_near), ("random_rows", k_rnd)):
                rr = np.asarray(N.recorded_range(acts[rows]))
                ctl = N.RangeClip(K, rr[:, 0], rr[:, 1], f"textbook gain clipped to the recorded range ({name})")
                pts[name] = {s: _points(TR.rollout(ctl, _bank(s, bank_n), physics).J.mean(), anchors[s])
                             for s in TC.SITUATIONS}
            f["data_edge_points"] = pts
            f["data_edge_loss_strong_current"] = float(pts["ball_alone"]["strong_current"]
                                                       - pts["near80"]["strong_current"])
    f["diagnostic_reads"] = int(log.diagnostic_reads)
    return f


def _bank(situation, n):
    return TC.bank_starts(situation) if n is None else TC.bank_starts(situation, int(n))


def _points(J, anchor):
    r, t = anchor
    return float(100.0 * (float(J) - r) / (t - r))


def anchors_for(world, spec=RC.DEFAULT_SPEC, bank_n=None):
    """{situation: (J of the recording controller on the bank, J of the world's top controller)}, exact."""
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


def fixed_bank_facts(anchors, lo=N.BAND_LO, hi=N.BAND_HI, bank_n=None):
    """Seed-free facts on the fixed test start sets, stalling world: how often the 100-point controller needs the band, and
    what holding the textbook rule to the band's lower edge costs. Nothing trained."""
    from experiments.shift import truth as TR
    physics = TC.world_physics("stall")
    K = RC.textbook_gain(physics)
    sa = RC.stall_aware(physics)
    held = RC.LinearController(K, lo, f"textbook gain held to {lo:.2f}")
    out = dict(situations={})
    for s in TC.SITUATIONS:
        x0 = _bank(s, bank_n)
        run = TR.rollout(sa, x0, physics, record=True)
        c = run.commands
        z = N.throttle_zones(c.reshape(-1, TC.DA), lo=lo)
        m1 = np.abs(c[..., 0])
        out["situations"][s] = dict(
            stall_aware_points=_points(run.J.mean(), anchors[s]),
            stall_aware_thruster1_band_share=float(((m1 >= lo) & (m1 <= hi)).mean()),
            stall_aware_zone_shares=[float((z == k).mean()) for k in (0, 1, 2)],
            textbook_held_points=_points(TR.rollout(held, x0, physics).J.mean(), anchors[s]))
    sc = out["situations"]["strong_current"]
    out["stall_aware_thruster1_band_share_strong_current"] = sc["stall_aware_thruster1_band_share"]
    out["hold_loss_strong_current"] = float(sc["stall_aware_points"] - sc["textbook_held_points"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=list(N.EXPLORATORY_SEEDS))
    p.add_argument("--worlds", nargs="+", default=["stall", "no_stall"], choices=["stall", "no_stall"])
    p.add_argument("--rollouts", action="store_true", help="also run the fixed controllers on the four banks")
    p.add_argument("--lo", type=float, default=N.BAND_LO)
    p.add_argument("--hi", type=float, default=N.BAND_HI)
    p.add_argument("--scope", default=N.SCOPE, choices=N.SCOPES)
    a = p.parse_args(argv)
    seeds = N.check_seeds(a.seeds)
    out = N.guard_out(a.out)
    from experiments.shift import acceptance as AC
    t0 = time.perf_counter()
    step1 = N.load_step1lib()
    anchors = {"stall": anchors_for("stall")} if (a.rollouts and "stall" in a.worlds) else {}   # only the stalling world is scored
    results = []
    for world in a.worlds:
        for rep in seeds:
            t1 = time.perf_counter()
            sd = AC.load_seed(RUNS, None, "base", world, rep)
            f = seed_world_facts(sd.log, sd.gap, world, sd.physics, sd.spec, step1, rec=sd.record, lo=a.lo, hi=a.hi,
                                 scope=a.scope, anchors=anchors.get(world) if world == "stall" else None)
            f["seconds"] = time.perf_counter() - t1
            results.append(f)
            print(f"{world} seed {rep}: extra {f['near80']['extra']}, kept {f['near80']['kept_rows']}, "
                  f"leak check {'pass' if f['leak_check_band']['passed'] else 'FAIL'}, {f['seconds']:.0f} s", flush=True)
    fixed = fixed_bank_facts(anchors["stall"], a.lo, a.hi) if "stall" in anchors else None
    variant = (a.lo, a.hi, a.scope) != (N.BAND_LO, N.BAND_HI, N.SCOPE)
    payload = dict(status=N.STATUS, label=("EXPLORATORY no-training facts, seeds 0-4" +
                                           (" (a VARIANT band, not the pre-registered one)" if variant else "")),
                   band=dict(lo=a.lo, hi=a.hi, scope=a.scope), seeds=seeds, worlds=a.worlds, results=results,
                   fixed_banks=fixed, anchors={w: v for w, v in anchors.items()},
                   no_training_checks=N.no_training_checks(results, fixed), seconds=time.perf_counter() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"wrote {out}; no-training checks {'PASSED' if payload['no_training_checks']['passed'] else 'NOT PASSED'}"
          f" (not run: {payload['no_training_checks']['not_run'] or 'none'})")
    return payload


if __name__ == "__main__":
    main()
