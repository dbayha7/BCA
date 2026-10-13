"""EXPLORATORY detection rehearsal (plan section 4.4), NO BCA: the seed-0 check, a small run under the written spec
and code manifest, into runs/shift_explore/detect/checks/seed0 (never the real seed folders).

  1. the real phases, each its own process (run_detect.py --out checks/seed0): pairs (both worlds), signals with the
     one cheap signal only (two_head), labels, measures; wall time of each;
  2. invariants of the pair sets (counts, region membership, caps, knee, T4 rows, references, float32 states,
     determinism by a rebuild);
  3. invariants of the labels (caps harmless, harmful within past-80%, finite; the JAX float64 truth against the numpy
     float64 truth on 16 pairs);
  4. the label script's isolation (its import list; a process that loads signal code is refused) and the join's
     refusal of a tampered score file;
  5. Stage 2's contract (scratchpad/ood/stage2/detect_contract.py, read-only) against this module and the measures;
  6. timing of every other signal: its fit and its scoring of a 2,000-pair subsample of the stalling world's pairs
     (the scores are discarded: not saved, never joined with a label), extrapolated to every pair;
  7. the projected cost of the full run (5 seeds x 2 worlds).
Writes checks/checks_seed0.json and timing.json (with sidecars).
Usage: python checks_seed0.py
"""

import ast
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import common as C  # noqa: E402

REL = "checks/seed0"
SUB = 2000
PY = sys.executable


def run_phase(phase, extra=()):
    env = dict(os.environ, PYTHONPATH=str(C.ROOT), PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="1")
    cmd = [PY, str(C.HERE / "run_detect.py"), "--phase", phase, "--seed", "0", "--world", "both", "--out", REL,
           *extra]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(C.ROOT), env=env, capture_output=True, text=True)
    sec = time.perf_counter() - t0
    log = C.OUT / "checks" / "logs" / f"seed0_{phase}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + r.stderr, encoding="utf8")
    if r.returncode != 0:
        raise SystemExit(f"check phase {phase} failed (see {log}):\n{r.stderr[-3000:]}")
    C.log(f"phase {phase}: {sec:.0f}s")
    return dict(seconds=sec, log=str(log))


def main():
    b = C.begin()
    t_all = time.perf_counter()
    out = C.OUT / REL
    if out.exists():
        raise SystemExit(f"refused: {out} exists (the seed-0 check runs once per spec; remove it with a reason)")
    res = dict(label=C.LABEL, kind="detect-checks-seed0", spec_sha256=b["spec_sha256"],
               code_manifest_sha256=b["manifest_sha256"], room_pins=b["room_pins"], phases={})
    # 1. the real phases, each its own process
    res["phases"]["pairs"] = run_phase("pairs")
    res["phases"]["signals_two_head"] = run_phase("signals", ("--signals", "two_head"))
    res["phases"]["labels"] = run_phase("labels")
    res["phases"]["measures"] = run_phase("measures")

    import pairs as P
    import labels as LB
    H, A, L, RC, TC, TR = C.gate()
    ctx = C.context(0)
    inv = {}
    for w in C.WORLDS:
        arrays, prec = P.load(0, w, out)
        lab, lrec = LB.load(0, w, out)
        pg = ctx.prepared(w)
        v = P.view(arrays)
        st, rl = arrays["set"], arrays["role"]
        a = arrays["commands"]
        chk = {}
        chk["counts_as_spec"] = bool(all(int(np.sum((st == s) & (rl == r))) == n for s, r, n in
                                         [("T1", r, P.N_T1) for r in P.SET_ROLES["T1"]] +
                                         [("T2", r, P.N_T2) for r in P.SET_ROLES["T2"]] +
                                         [("T3", "command", 2 * P.N_T3_EACH), ("T3", "cap", 2 * P.N_T3_EACH),
                                          ("T4", "recorded", P.N_T4)]))
        t1 = arrays["train_index"][v["T1"]["command"]]
        t2 = arrays["train_index"][v["T2"]["command"]]
        chk["T1_in_region"] = bool(pg.region[t1].all())
        chk["T2_outside_region"] = bool((~pg.region[t2]).all())
        chk["T1_T2_disjoint_unique"] = bool(len(set(t1) | set(t2)) == len(t1) + len(t2))
        chk["states_match_training_rows"] = bool(np.array_equal(arrays["states"][v["T1"]["command"]], pg.states[t1]))
        chk["states_float32_exact"] = bool(np.array_equal(arrays["states"].astype(np.float32).astype(np.float64),
                                                          arrays["states"]))
        chk["caps_never_past_80"] = bool(not TC.past_knee(a[rl == "cap"]).any())
        cmd_rows = np.concatenate([v[s]["command"] for s in ("T1", "T2", "T3")])
        cap_rows = np.concatenate([v[s]["cap"] for s in ("T1", "T2", "T3")])
        past = TC.past_knee(a[cmd_rows])
        chk["cap_equals_command_iff_not_past"] = bool(np.array_equal(np.all(a[cap_rows] == a[cmd_rows], 1), ~past))
        kn = a[v["T1"]["knee"]]
        chk["knee_past_iff_command_past"] = bool(np.array_equal(TC.past_knee(kn), TC.past_knee(a[v["T1"]["command"]]))
                                                 and np.abs(kn).max() <= np.float32(0.81))
        chk["commands_are_the_float32_actor"] = bool(np.array_equal(
            a[v["T1"]["command"]], L.actor_rows(ctx.none(w), pg.normalize(arrays["states"][v["T1"]["command"]]))))
        lg, sd = ctx.sd[w].log, ctx.sd[w]
        t4 = arrays["log_row"][v["T4"]["recorded"]]
        cal = set(lg.split_rows("calibration").tolist())
        chk["T4_in_calibration_split"] = bool(all(int(r) in cal for r in t4))
        chk["T4_not_bank"] = bool(not np.isin(t4, lg.bank_rows).any())
        chk["T4_not_in_gap"] = bool(not sd.gap.contains(lg["actions"][t4]).any())
        chk["T4_not_training"] = bool(not np.isin(t4, pg.rows).any())
        chk["refs_recorded_at_T1_T2"] = bool(all(np.array_equal(arrays["refs"][v[s]["command"]], a[v[s]["recorded"]])
                                                 for s in ("T1", "T2")))
        t3 = v["T3"]["command"]
        chk["refs_cloned_at_T3"] = bool(np.array_equal(arrays["refs"][t3], L.actor_rows(
            ctx.clone(w)[0], pg.normalize(arrays["states"][t3]))))
        # labels
        caps = np.flatnonzero(rl == "cap")
        chk["labels_caps_harmless"] = bool(not lab["harmful"][caps].any() and not lab["unsupported"][caps].any())
        chk["labels_harmful_within_past80"] = bool(not np.any(lab["harmful"] & ~lab["unsupported"]))
        chk["labels_finite"] = bool(np.isfinite(lab["over_pref"]).all() and np.isfinite(lab["true_pref"]).all())
        chk["labels_simulated_iff_past80"] = bool(np.array_equal(lab["simulated"], lab["unsupported"]))
        sim = np.flatnonzero(lab["simulated"])
        pick = sim[np.linspace(0, len(sim) - 1, 16).astype(int)] if len(sim) else sim
        pol = RC.stall_aware(pg.physics)
        ref = TR.q_value(pol, arrays["states"][pick], a[pick].astype(np.float64), pg.physics, horizon=LB.HORIZON)
        chk["truth_jax_vs_numpy_max_abs"] = float(np.max(np.abs(ref - lab["true_command"][pick]))) if len(pick) else 0.0
        chk["truth_jax_vs_numpy_ok"] = bool(chk["truth_jax_vs_numpy_max_abs"] < 1e-8)
        chk["pairs_content_sha256"] = prec["content_sha256"]
        chk["labels_content_sha256"] = lrec["content_sha256"]
        chk["label_counts"] = lrec["info"]["counts"]
        chk["labels_seconds"] = lrec["info"]["seconds"]
        chk["pairs_seconds"] = prec["info"].get("seconds")
        if w == "stall":
            t0 = time.perf_counter()
            again, _ = P.build(ctx, w, log_fn=None)
            chk["deterministic_rebuild"] = bool(C.content_sha256(again) == prec["content_sha256"])
            chk["rebuild_seconds"] = time.perf_counter() - t0
        inv[w] = chk
    res["invariants"] = inv
    # measures of the cheap signal
    res["two_head_measures"] = {}
    for w in C.WORLDS:
        m = C.read_json(C.seed_dir(0, w, out) / "measures" / "two_head.json")
        res["two_head_measures"][w] = m["measures"]
    # 4. isolation and refusals
    tree = ast.parse((C.HERE / "labels.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {x.name.split(".")[0] for x in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    iso = dict(labels_imports=sorted(names), labels_imports_ok=bool(names <= {"sys", "time", "numpy", "common",
                                                                               "pairs"}))
    code = ("import sys; sys.path.insert(0, %r); import signals, labels, common as C; "
            "labels.run(type('Ctx', (), {'rep': 0})(), 'stall', C.OUT / %r)") % (str(C.HERE), REL)
    r = subprocess.run([PY, "-c", code], cwd=str(C.ROOT), capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(C.ROOT), PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu",
                                OMP_NUM_THREADS="1"))
    iso["label_script_refused_with_signals_loaded"] = bool(r.returncode != 0 and "refused" in (r.stdout + r.stderr))
    import measures as M
    tam = C.OUT / "checks" / "tamper"
    if tam.exists():
        shutil.rmtree(tam)
    shutil.copytree(C.seed_dir(0, "stall", out), tam / "seed0" / "stall")
    p = tam / "seed0" / "stall" / "scores" / "two_head.npz"
    bb = bytearray(p.read_bytes())
    bb[-9] ^= 1
    p.write_bytes(bytes(bb))
    try:
        M.load_join(tam / "seed0" / "stall" / "scores" / "two_head.json", tam / "seed0" / "stall" / "labels.json")
        iso["join_refused_tampered_scores"] = False
    except SystemExit:
        iso["join_refused_tampered_scores"] = True
    shutil.rmtree(tam)
    res["isolation"] = iso
    # 5. Stage 2's contract
    import detect as D
    contract = C.HERE.parent / "stage2" / "detect_contract.py"
    if contract.exists():
        ns = {}
        exec(compile(contract.read_bytes(), str(contract), "exec"), ns)
        try:
            ns["check_detect_module"](D)
            ns["check_measures"](res["two_head_measures"]["stall"])
            res["stage2_contract"] = dict(ok=True, sha256=C.file_sha256(contract))
        except SystemExit as e:
            res["stage2_contract"] = dict(ok=False, error=str(e), sha256=C.file_sha256(contract))
    else:
        res["stage2_contract"] = dict(ok=None, note="stage2/detect_contract.py not present")
    # 6. timing of every other signal (scores discarded; never joined with a label)
    import signals as S
    arrays, _ = P.load(0, "stall", out)
    n_all = len(arrays["set"])
    idx = np.sort(C.rng(0, 199).choice(n_all, SUB, replace=False))
    tim = {}
    t0 = time.perf_counter()
    sigs = S.stand_in_signals(ctx, "stall", names=S.ALL_NAMES, log_fn=None)      # as the signals phase builds them
    fit_all = time.perf_counter() - t0
    for name in S.ALL_NAMES:
        sig = sigs[name]
        t1 = time.perf_counter()
        args = (arrays["states"][idx], arrays["commands"][idx]) + ((arrays["refs"][idx],) if sig.needs_ref else ())
        sc = sig(*args)
        score = time.perf_counter() - t1
        tim[name] = dict(fit_seconds=fit_all if name == S.ALL_NAMES[0] else 0.0, score_seconds_subsample=score,
                         subsample=SUB, score_seconds_projected=score * n_all / SUB,
                         finite=bool(np.isfinite(sc).all()))
        del sc
    res["signal_timing_stall"] = dict(per_signal=tim, fit_all_seconds=fit_all,
                                      note=("every signal fitted in one call, as the signals phase does; residual "
                                            "evidence reuses the distance's neighbour search"))
    # 7. projection
    ph = res["phases"]
    per_world_labels = np.mean([inv[w]["labels_seconds"]["total"] for w in C.WORLDS])
    sig_total = fit_all + sum(t["score_seconds_projected"] for t in tim.values())
    per_seed_wall = (ph["pairs"]["seconds"] + ph["labels"]["seconds"] + ph["measures"]["seconds"] * len(S.ALL_NAMES)
                     + ph["signals_two_head"]["seconds"] + 2 * sig_total)
    timing = dict(label=C.LABEL, kind="detect-timing", spec_sha256=b["spec_sha256"],
                  code_manifest_sha256=b["manifest_sha256"],
                  measured_under=("the room test running on all 4 CPUs; this check at nice 15, one process at a time; "
                                  "wall seconds"),
                  seed0_phases_wall_seconds={k: v["seconds"] for k, v in ph.items()},
                  labels_compute_seconds_per_world=per_world_labels,
                  signal_fit_and_score_seconds_per_world=sig_total, signal_timing_stall=tim,
                  per_seed_both_worlds_projected_wall_seconds=per_seed_wall,
                  all_seeds_projected_wall_seconds=5 * per_seed_wall,
                  all_seeds_projected_process_hours=5 * per_seed_wall / 3600.0,
                  plan_estimate_process_hours=2.5,
                  rule=("per seed: the measured pairs, labels and two-head signals phases (both worlds, each phase "
                        "including its process start and gate loading), the measures phase scaled by the number of "
                        "signals, plus every other signal's fit and projected full scoring in both worlds"))
    res["seconds"] = time.perf_counter() - t_all
    res["all_invariants_ok"] = bool(all(all(v for k, v in inv[w].items() if isinstance(v, bool)) for w in C.WORLDS)
                                    and iso["labels_imports_ok"] and iso["label_script_refused_with_signals_loaded"]
                                    and iso["join_refused_tampered_scores"]
                                    and res["stage2_contract"].get("ok") is not False)
    C.write_json(C.OUT / "checks" / "checks_seed0.json", res)
    C.write_json(C.OUT / "timing.json", timing)
    C.log(f"seed-0 check: invariants {'ok' if res['all_invariants_ok'] else 'FAILED'}; projected full run "
          f"{timing['all_seeds_projected_process_hours']:.2f} process-hours ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
