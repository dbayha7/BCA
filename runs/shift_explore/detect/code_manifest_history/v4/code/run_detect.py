"""EXPLORATORY detection rehearsal (plan section 4.4), NO BCA: the phases, one per process.

  python run_detect.py --phase pairs    --seed K --world stall|no_stall|both
  python run_detect.py --phase signals  --seed K --world ... [--signals a,b,...]   (default: every Stage 1 signal)
  python run_detect.py --phase labels   --seed K --world ...
  python run_detect.py --phase measures --seed K --world ...
  python run_detect.py --phase pooled

Order per seed and world: pairs -> signals -> labels -> measures; pooled after every seed and world.
  - every phase first checks the spec, the gate files, the room pins and the code manifest (common.begin);
  - pairs, scores, labels and measures are written once. An existing output is skipped only when it verifies AND
    names the current spec and code manifest (its spec_sha256 / code_manifest_sha256 and the module's stamp,
    common.check_version); one written under another spec or manifest refuses, as does a half-written one. A new spec
    version therefore starts in a fresh output folder or archives the old outputs first;
  - 'signals' refuses when the labels of that seed and world exist (every Stage 1 signal is scored blind to the
    labels; signals.score_signal enforces the same in the rehearsal's folder) and writes signals_done.json; 'labels'
    refuses until signals_done.json exists (labels.run enforces the same), runs in a process with no signal code
    loaded (this file imports labels.py only in that phase) and never opens a score file;
  - 'measures' joins each hashed scores record with the hashed labels record (measures.py);
  - 'pooled' reads only measures written under the current spec and code manifest;
  - every phase refuses unless the process may run on exactly one CPU (launch.sh pins it): the float32 networks'
    last bits depend on the CPU count (spec change C9).
--out: a folder under runs/shift_explore/detect for checks (default: runs/shift_explore/detect itself).
Nothing is written outside runs/shift_explore/detect. Launch: launch.sh, from the pinned copies in
runs/shift_explore/detect/code (common.check_code accepts either the copies or the scratchpad sources).
"""

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

SIGNALS_DONE = C.SIGNALS_DONE
POOLED = "pooled.json"
KNEE_KEYS = tuple(f"knee_{k}_T1_{grp}{suf}" for grp in ("past80", "harmful") for k, suf in (
    ("rise", ""), ("fall", ""), ("mean_change", ""), ("rise", "_paired"), ("fall", "_paired"), ("tie", "_paired"),
    ("mean_change", "_paired")))
POOLED_KEYS = ("widening_T1_harmful", "widening_T1_past80", "widening_T1_harmless_past80", "widening_T2_past80",
               "widening_T3_harmful", "widening_T3_past80", "widening_T3_harmless_past80", "widening_T3a_harmful",
               "widening_T3b_harmful", "widening_T1_harmful_paired", "size_T1_raw", "size_T1_paired", "size_T3_raw",
               *KNEE_KEYS, "ranking_T12", "ranking_T3", "ranking_T1", "ranking_T2", "ranking_T3a", "ranking_T3b",
               "ranking_T12_among_past80", "ranking_T3_among_past80", "rate_above_level_T4", "catch_rate_T1_harmful",
               "catch_rate_T3_harmful", "catch_rate_T3_past80", "catch_rate_T3a_past80", "catch_rate_T3b_past80",
               "false_flag_T1", "false_flag_T2", "false_flag_T3", "false_flag_T1_cap", "false_flag_T1_recorded",
               "false_flag_T3_cap", "flag_rate_past80", "flag_rate_past80_T1", "flag_rate_past80_T3", "n_harmful_T1",
               "n_harmful_T3", "n_past80_T1", "n_past80_T3", "level")
SANITY = dict(signal="perfect_band", world="stall", widening_T1_harmful_at_least=0.9, ranking_T12_at_least=0.9)
BARS = dict(
    key="catch_rate_T3a_past80", distance_check=0.35, exact_one_step_check=0.67,
    source=("runs/shift_explore/fallback (fallback.md, table 'Which harmful commands each check catches (strong "
            "current, clip fallback)'): the gapped critic's actor, strong current, first 200 starts, seeds 0-4"),
    distance_check_readout=("trigger (b), new (state, command): the distance to the 10th-nearest training (state, "
                            "command) row above its 95% split-conformal level, set on the held-out calibration-split "
                            "rows (about 16,800 rows, the removal rule applied); 35% = the share of the actor's "
                            "past-80% proposals it caught, counted along rollouts that the clip fallback changed "
                            "whenever the trigger fired"),
    exact_one_step_check_readout=("trigger (c), an exact-return comparison, not a scored signal and with no level: "
                                  "switch when the clipped command followed by the actor has a higher exact return "
                                  "than the actor's own command followed by the actor; 67% = the share of past-80% "
                                  "proposals on which it switched, counted along rollouts that the clip fallback "
                                  "changed"),
    comparability=("carried forward from a DIFFERENT readout and reported beside catch_rate_T3a_past80 for reference "
                   "only: T3a scores 1,500 states of the actor's own UNCHANGED rollouts (first 200 strong-current "
                   "starts), each signal flagged above its 95% level set on T4 (5,000 rows of the calibration "
                   "split, gap removed, bank rows left out); the bars came from rule-modified rollouts, an "
                   "exact-return switch (the 67%) and a different threshold pool (the 35%)"))
def out_root(arg):
    if arg is None:
        return C.OUT
    p = (C.OUT / arg).resolve()
    if C.OUT.resolve() not in p.parents:
        raise SystemExit("refused: --out must be a folder under runs/shift_explore/detect")
    return p


def worlds_of(arg):
    return C.WORLDS if arg == "both" else (arg,)


def check_run(rec, run, what):
    """An existing output is reused only when it names the current spec and code manifest (the phase's own run
    fields and the module's stamp); otherwise refuse (a new spec version starts in a fresh folder or archives)."""
    C.check_version(rec, what)
    bad = [k for k in ("spec_sha256", "code_manifest_sha256") if rec.get(k) != run[k]]
    if bad:
        raise SystemExit(f"refused: {what} was written under another spec or code manifest ({', '.join(bad)}); archive "
                         f"the old outputs or use a fresh output folder")
    return rec


def phase_pairs(ctx, world, out, run):
    import pairs as P

    d = C.seed_dir(ctx.rep, world, out)
    if (d / P.RECORD).exists() or (d / P.FILE).exists():
        _, rec = P.load(ctx.rep, world, out)
        check_run(rec, run, f"the pair sets in {d}")
        C.log(f"pairs seed {ctx.rep} {world}: exist, verified, current spec and manifest")
        return
    t0 = time.perf_counter()
    arrays, info = P.build(ctx, world)
    info["seconds"] = time.perf_counter() - t0
    rec = P.save(arrays, info, ctx.rep, world, out, extra=run)
    C.log(f"pairs seed {ctx.rep} {world}: {info['n_pairs']} pairs, content {rec['content_sha256'][:16]} "
          f"({info['seconds']:.0f}s)")


def phase_signals(ctx, world, out, run, names):
    import pairs as P
    import signals as S

    d = C.seed_dir(ctx.rep, world, out)
    if (d / "labels.json").exists() or (d / "labels.npz").exists():
        raise SystemExit(f"refused: labels of seed {ctx.rep} {world} exist; every Stage 1 signal is scored before them")
    _, prec = P.load(ctx.rep, world, out)
    check_run(prec, run, f"the pair sets in {d}")
    names = tuple(names)
    bad = [n for n in names if n not in S.ALL_NAMES]
    if bad:
        raise SystemExit(f"refused: unknown signals {bad}")
    todo = []
    for n in names:
        js, nz = d / S.SCORES_DIR / f"{n}.json", d / S.SCORES_DIR / f"{n}.npz"
        if js.exists() and nz.exists():
            rec = C.read_json(js)
            check_run(rec, run, f"the scores of {n} in {d / S.SCORES_DIR}")
            C.load_npz(nz, rec["file_sha256"], rec["content_sha256"])
            C.log(f"scores {n} seed {ctx.rep} {world}: exist, verified, current spec and manifest")
        elif js.exists() or nz.exists():
            raise SystemExit(f"refused: half-written scores of {n} in {d / S.SCORES_DIR}")
        else:
            todo.append(n)
    t0, c0 = time.perf_counter(), time.process_time()
    sigs = S.stand_in_signals(ctx, world, names=tuple(todo)) if todo else {}
    fit = time.perf_counter() - t0
    recs = {}
    for n in todo:
        recs[n] = S.score_signal(n, sigs[n], ctx, world, pairs_out=out, extra=run)
    done_prev = C.read_json(d / SIGNALS_DONE)
    if done_prev is not None:
        check_run(done_prev, run, f"{d / SIGNALS_DONE}")
    scored = sorted(set(names) | set((done_prev or {}).get("signals", [])))
    C.write_json(d / SIGNALS_DONE, dict(label=C.LABEL, kind="detect-signals-done", rep=int(ctx.rep), world=world,
                                        signals=scored, fit_seconds=fit, seconds=time.perf_counter() - t0,
                                        cpu_seconds=time.process_time() - c0,
                                        score_seconds={n: r["score_seconds"] for n, r in recs.items()}, **run,
                                        **C.stamp()))


def phase_labels(ctx, world, out, run):
    import labels as LB

    LB.refuse_if_signals_loaded()
    d = C.seed_dir(ctx.rep, world, out)
    if not (d / SIGNALS_DONE).exists():
        raise SystemExit(f"refused: the signals of seed {ctx.rep} {world} are not scored yet; labels come after them")
    check_run(C.read_json(d / SIGNALS_DONE), run, f"{d / SIGNALS_DONE}")
    if (d / LB.RECORD).exists() and (d / LB.FILE).exists():
        _, rec = LB.load(ctx.rep, world, out)
        check_run(rec, run, f"the labels in {d}")
        C.log(f"labels seed {ctx.rep} {world}: exist, verified, current spec and manifest")
        return
    if (d / LB.RECORD).exists() or (d / LB.FILE).exists():
        raise SystemExit(f"refused: half-written labels in {d}")
    LB.run(ctx, world, out, extra=run)
    LB.refuse_if_signals_loaded()


def phase_measures(rep, world, out, run):
    import measures as M

    d = C.seed_dir(rep, world, out)
    lab = d / "labels.json"
    if not lab.exists():
        raise SystemExit(f"refused: no labels for seed {rep} {world}")
    check_run(C.read_json(lab), run, f"the labels in {d}")
    for js in sorted((d / "scores").glob("*.json")):
        name = js.stem
        mo = d / "measures" / f"{name}.json"
        if mo.exists():
            check_run(C.read_json(mo), run, f"the measures of {name} in {d}")
            C.log(f"measures {name} seed {rep} {world}: exist, verified, current spec and manifest")
            continue
        srec = check_run(C.read_json(js), run, f"the scores of {name} in {d}")
        extra = srec.get("signal_info", {}).get("extra_level") or {}
        m = M.measures(js, lab, extra_levels=extra)
        m.update(run)
        C.write_json(mo, m)
        mm = m["measures"]
        C.log(f"measures {name} seed {rep} {world}: widening T1/T3 harmful {mm['widening_T1_harmful']}/"
              f"{mm['widening_T3_harmful']}, ranking T12/T3 {mm['ranking_T12']}/{mm['ranking_T3']}")


def phase_pooled(out, run, seeds=C.SEEDS):
    import signals as S

    if (out / POOLED).exists():
        raise SystemExit(f"refused: {out / POOLED} exists")
    table, missing = {}, []
    for w in C.WORLDS:
        for n in S.ALL_NAMES:
            per = []
            for rep in seeds:
                rec = C.read_json(C.seed_dir(rep, w, out) / "measures" / f"{n}.json")
                if rec is None:
                    missing.append(f"seed{rep}/{w}/{n}")
                else:
                    check_run(rec, run, f"the measures of seed {rep} {w} {n}")
                per.append(None if rec is None else rec["measures"])
            table[(w, n)] = per
    if missing:
        raise SystemExit("refused: measures missing: " + ", ".join(missing))

    def col(w, n, k):
        vals = [m.get(k) for m in table[(w, n)]]
        ok = [v for v in vals if v is not None]
        return dict(values=vals, mean=(sum(ok) / len(ok)) if ok and len(ok) == len(vals) else None,
                    seeds_with_value=len(ok))

    res = {}
    for n in S.ALL_NAMES:
        res[n] = {w: {k: col(w, n, k) for k in POOLED_KEYS} for w in C.WORLDS}
        sel = {}
        for k in ("flag_rate_past80", "flag_rate_past80_T1", "flag_rate_past80_T3", "catch_rate_T3a_past80"):
            vals = [None if a.get(k) is None or b.get(k) is None else a[k] - b[k]
                    for a, b in zip(table[("stall", n)], table[("no_stall", n)])]
            ok = [v for v in vals if v is not None]
            sel[k] = dict(values=vals, mean=(sum(ok) / len(ok)) if ok and len(ok) == len(vals) else None)
        res[n]["selectivity"] = sel
    pb = res[SANITY["signal"]][SANITY["world"]]
    sanity = dict(SANITY, widening_T1_harmful=pb["widening_T1_harmful"]["mean"], ranking_T12=pb["ranking_T12"]["mean"])
    sanity["met"] = bool(sanity["widening_T1_harmful"] is not None and sanity["ranking_T12"] is not None
                         and sanity["widening_T1_harmful"] >= SANITY["widening_T1_harmful_at_least"]
                         and sanity["ranking_T12"] >= SANITY["ranking_T12_at_least"])
    bars = dict(BARS, signals={n: res[n]["stall"][BARS["key"]]["mean"] for n in S.ALL_NAMES})
    best = {}
    for k in ("ranking_T12", "ranking_T3"):
        per = {n: res[n]["stall"][k]["mean"] for n in S.STANDIN_NAMES}
        per = {a: b for a, b in per.items() if b is not None}
        best[k] = dict(name=max(per, key=per.get) if per else None, value=max(per.values()) if per else None, all=per)
    C.write_json(out / POOLED, dict(label=C.LABEL, kind="detect-pooled", seeds=list(seeds), worlds=list(C.WORLDS),
                                    signals=res, perfect_band_sanity=sanity, bars_carried_forward=bars,
                                    best_standin_ranking=best, **run, **C.stamp()))
    C.log(f"pooled written: perfect band sanity {'met' if sanity['met'] else 'NOT met'}; best stand-in ranking "
          f"T12 {best['ranking_T12']['name']} {best['ranking_T12']['value']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("pairs", "signals", "labels", "measures", "pooled"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--world", default="both", choices=("stall", "no_stall", "both"))
    ap.add_argument("--signals", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    C.require_one_cpu()
    t0 = time.perf_counter()
    b = C.begin()
    out = out_root(a.out)
    run = dict(spec_sha256=b["spec_sha256"], code_manifest_sha256=b["manifest_sha256"],
               room_manifest_sha256=b["room_pins"]["room_manifest_sha256"])
    if a.phase == "pooled":
        phase_pooled(out, run)
        return
    if a.seed not in C.SEEDS:
        raise SystemExit("refused: --seed must be a gate seed 0-4")
    worlds = worlds_of(a.world)
    if a.phase == "measures":
        for w in worlds:
            phase_measures(a.seed, w, out, run)
    else:
        ctx = C.context(a.seed, worlds)
        for w in worlds:
            if a.phase == "pairs":
                phase_pairs(ctx, w, out, run)
            elif a.phase == "signals":
                import signals as S
                names = S.ALL_NAMES if a.signals is None else tuple(x for x in a.signals.split(",") if x)
                phase_signals(ctx, w, out, run, names)
            elif a.phase == "labels":
                phase_labels(ctx, w, out, run)
    C.log(f"phase {a.phase} seed {a.seed} {a.world} done ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
