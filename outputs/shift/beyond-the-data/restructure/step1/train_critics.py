"""Step 1: train ONE fixed-controller critic of one seed and world on the gate's gapped log minus its surviving
full-throttle rows, with the gate's own protocol (learners.fixed_critic, unchanged). EXPLORATORY; see PRESPEC.md.

Official mode (no --out-root): writes only runs/shift_explore/restructure/step1/critics/<world>/seed<k>/
{critic.msgpack, record.json(.sha256)}, write-once, after checking prespec.json's SHA-256 and the code manifest. The
protocol is fixed: survivors removed, 100,000 updates. It must run pinned to one CPU (taskset -c <cpu>), with
OMP_NUM_THREADS=1 and JAX_PLATFORMS=cpu as the gate ran.

Scratch mode (--out-root DIR outside the repository): the same code path, for the reproduction check
(--no-removal: the gate's exact gapped log, which must reproduce the gate's saved critic bit for bit) and for the smoke
test (--updates N). Records are labelled scratch and evaluate.py refuses them in official mode.

Usage:
  taskset -c 0 python train_critics.py --seed 0 --world stall
  taskset -c 0 python train_critics.py --seed 0 --world stall --out-root /tmp/.../repro --no-removal
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1lib as S  # noqa: E402

import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.shift as SH  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gate_context(spec):
    p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(p)
    if p.digest != spec["gate"]["prereg_sha256"]:
        raise SystemExit("refused: the gate's pre-registration differs from the one prespec.json names")
    variant = A.gate_outcome(p)["variant"]
    if variant != spec["gate"]["variant"]:
        raise SystemExit(f"refused: the gate's variant is {variant}, prespec.json says {spec['gate']['variant']}")
    return p, variant


def survivors_of(sd, learn, spec):
    """The gate's gapped prepared data and its surviving full-throttle training rows by the frozen gap rule."""
    pg = sd.prepared("gap", learn)
    gr_cfg = spec["groups"]["gap_rule"]
    gr = S.gap_rule(S.throttle(pg.actions), gr_cfg["upper_quantile"], gr_cfg["min_gap"], gr_cfg["min_tail"])
    if not gr["defined"]:
        raise SystemExit(f"refused: the gap rule is not defined on {sd.world}/seed {sd.rep}: {gr.get('reason')}")
    surv = np.asarray(pg.rows[gr["tail"]], np.int64)
    exp = spec["groups"]["expected_from_design"][f"{sd.world}/seed{sd.rep}"]
    if len(surv) != exp["n_survivors"] or abs(gr["t_lo"] - exp["t_lo"]) > 1e-12:
        raise SystemExit(f"refused: the gap rule found {len(surv)} survivors at t_lo {gr['t_lo']}; the design found "
                         f"{exp['n_survivors']} at {exp['t_lo']}")
    return pg, gr, surv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--world", choices=S.WORLDS, required=True)
    ap.add_argument("--out-root", default=None, help="scratch mode: a folder outside the repository")
    ap.add_argument("--no-removal", action="store_true", help="scratch only: train on the gate's exact gapped log")
    ap.add_argument("--updates", type=int, default=None, help="scratch only: fewer updates (smoke test)")
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    official = a.out_root is None
    if official and (a.no_removal or a.updates is not None):
        raise SystemExit("refused: --no-removal and --updates are for scratch runs only (--out-root)")
    spec, spec_sha = S.load_prespec()
    man, man_sha = S.verify_manifest()
    if a.seed not in spec["seeds"]:
        raise SystemExit(f"refused: seed {a.seed} is not one of {spec['seeds']}")
    cpu = S.require_one_cpu("critic training")
    for k, v in spec["training"]["environment"].items():
        if os.environ.get(k) != v:
            raise SystemExit(f"refused: {k} must be {v!r} (got {os.environ.get(k)!r})")
    root = S.OFFICIAL_CRITICS if official else Path(a.out_root)
    d = S.guard_out(root / a.world / f"seed{a.seed}", official)
    if (d / "record.json").exists() or (d / "critic.msgpack").exists():
        raise SystemExit(f"refused: {d} already holds a critic or a record (write-once); a critic.msgpack without a "
                         f"record means an interrupted run: inspect it by hand")

    p, variant = gate_context(spec)
    learn = p.config.learn
    tr = spec["training"]
    if learn.updates != tr["updates"] or learn.batch_size != tr["batch_size"] or learn.gamma != tr["gamma"]:
        raise SystemExit("refused: the gate's learner settings differ from prespec.json")
    t0 = time.perf_counter()
    sd = A.load_seed(p.runs, p.config, variant, a.world, a.seed)
    pg, gr, surv = survivors_of(sd, learn, spec)
    rg = sd.removals["gap"]
    if a.no_removal:
        removal, kind = rg, "gap (no rows removed: reproduction check)"
    else:
        removed = rg.removed.copy()
        if removed[surv].any():
            raise AssertionError("a survivor is already removed")
        removed[surv] = True
        removal = SH.Removal("gap_minus_survivors", removed, rg.bank_removed.copy(), sd.gap)
        kind = "gap minus survivors"
    prep = L.prepare(sd.log, removal, sd.norm(), learn)      # the gate's normalisation: the gapped log's kept rows
    checks = dict(
        rows=bool(prep.n == pg.n - (0 if a.no_removal else len(surv))),
        no_survivor_in_training=bool(a.no_removal or not np.isin(surv, prep.rows).any()),
        bank_unchanged=bool(np.array_equal(prep.bank_rows, pg.bank_rows)),
        normalisation_unchanged=bool(np.array_equal(prep.obs_mean, pg.obs_mean)
                                     and np.array_equal(prep.obs_std, pg.obs_std)),
        eval_rows_unchanged=bool(np.array_equal(prep.eval_rows, pg.eval_rows)),
        training_rows_are_gapped_minus_survivors=bool(np.array_equal(
            prep.rows, pg.rows if a.no_removal else np.setdiff1d(pg.rows, surv))))
    if not all(checks.values()):
        raise SystemExit(f"refused: the training data checks failed: {checks}")
    load_seconds = time.perf_counter() - t0

    gate_dir = A.critic_dir(p, variant, a.world, "gap", tr["controller"], False, a.seed)
    gate_rec = A.finished(gate_dir / "record.json", p)
    if gate_rec is None:
        raise SystemExit(f"refused: the gate's critic record {gate_dir} is missing")
    # Review: refuse BEFORE training (no compute spent) unless the learner arguments that fixed_critic will build are
    # the gate critic's; the initial parameters are a function of these arguments (args.seed) and are compared again
    # after training (record.compare_with_gate; evaluate.py refuses a record where either differs).
    planned_args = json.loads(json.dumps(S.jsonable(asdict(L.make_args(learn, a.seed))), sort_keys=True))
    if planned_args != gate_rec["result"]["args"]:
        raise SystemExit("refused: the learner arguments differ from the gate critic's (not the exact gate protocol)")

    log(f"{a.world}/seed {a.seed}: {kind}; {prep.n} training rows ({len(surv)} survivors, t_lo {gr['t_lo']:.4f}); "
        f"{'official' if official else 'scratch'} -> {d}; CPU {cpu}")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t1 = time.perf_counter()
    result, state = L.fixed_critic(prep, tr["controller"], learn, a.seed, placebo=False, updates=a.updates,
                                   out_dir=d, log=log)
    train_seconds = time.perf_counter() - t1
    file_sha = S.file_sha256(d / "critic.msgpack")
    if file_sha != result.get("file_sha256"):
        raise SystemExit("the saved critic file does not hash to the value learners.fixed_critic returned")
    g = gate_rec["result"]
    same_args = json.loads(json.dumps(S.jsonable(result["args"]), sort_keys=True)) == g["args"]
    compare = dict(initial_params_equal_gate=bool(result["initial_sha256"] == g["initial_sha256"]),
                   args_equal_gate=bool(same_args),
                   critic_sha256_equal_gate=bool(result["critic_sha256"] == g["critic_sha256"]),
                   file_sha256_equal_gate=bool(file_sha == g["file_sha256"]))
    rec = dict(label=S.LABEL, kind="step1-critic", mode="official" if official else "scratch",
               variant_of_run=kind, seed=a.seed, world=a.world,
               prespec_sha256=spec_sha, code_manifest_sha256=man_sha, gate_prereg_sha256=p.digest,
               gap_rule=dict(t_lo=gr["t_lo"], t_up=gr["t_up"], n_survivors=int(len(surv))),
               survivors=surv.tolist(), survivors_sha256=S.array_sha256(surv),
               training_rows_sha256=S.array_sha256(prep.rows), data=prep.describe(), checks=checks,
               result=result, file_sha256=file_sha, failed=bool(result["failed"]),
               gate_critic=dict(dir=str(gate_dir), critic_sha256=g["critic_sha256"], file_sha256=g["file_sha256"],
                                initial_sha256=g["initial_sha256"], seconds=g.get("seconds")),
               compare_with_gate=compare, cpus=cpu, machine=platform.node(), software=A.software_state(),
               started=started, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               seconds=dict(load=load_seconds, train=train_seconds, total=time.perf_counter() - t_start))
    S.write_json_once(d / "record.json", rec, official)
    log(f"{a.world}/seed {a.seed}: {'FAILED' if result['failed'] else 'done'} in {train_seconds:.0f}s; "
        f"file {file_sha[:16]}; equal to the gate's critic: {compare['file_sha256_equal_gate']}")
    return rec


if __name__ == "__main__":
    main()
