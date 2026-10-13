"""Synthetic tests of the detect component (no gate data, no BCA): the measures on a hand-made pair set with known
answers (the ranking pools without the knee pairs, the knee measures), the join's refusals, the label script's
isolation, the output guard, the .npz hashing, the Stage 2 hook's refusals, the record versions (reuse refused under
another spec or code manifest), the labels chosen by folder, the order rule enforced in the module and the labelled
logs. Writes only under scratchpad/ood/detect/_test_out (removed first). Runs only from the scratchpad sources.
Usage: python test_detect.py
"""

import ast
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import pairs as P  # noqa: E402
import labels as LB  # noqa: E402
import measures as M  # noqa: E402
import detect as D  # noqa: E402

KNEE = float(np.float32(0.8))
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def refuses(fn, what):
    try:
        fn()
    except SystemExit:
        check(True, what)
        return
    except Exception as e:  # noqa: BLE001
        check(False, f"{what} (raised {type(e).__name__}: {e})")
        return
    check(False, f"{what} (did not refuse)")


class Ctx:
    rep = 0


def synthetic():
    """T1: 6 states (pairs 0-3 past 80%; 0-2 harmful), T2: 4 states (none past), T3: 3 actor + 3 controller states
    (T3 pairs 0, 1, 3 past 80%; 0 and 3 harmful), T4: 20 rows."""
    rng = np.random.default_rng(0)
    cols = {k: [] for k in ("states", "commands", "refs", "set", "role", "pair", "log_row", "train_index",
                            "t3_source", "t3_start", "t3_step")}
    lab = {k: [] for k in ("unsupported", "harmful", "over_pref")}
    sc = []

    def add(st, role, pid, cmd, score, harm=False, over=0.0, src=""):
        cols["states"].append(rng.normal(size=13).astype(np.float32).astype(np.float64))
        a = np.array([cmd, 0.1, 0.0, -0.2], np.float32)
        cols["commands"].append(a)
        cols["refs"].append(np.array([-0.4, 0.1, 0.0, -0.2], np.float32))
        cols["set"].append(st)
        cols["role"].append(role)
        cols["pair"].append(pid)
        cols["log_row"].append(-1)
        cols["train_index"].append(-1)
        cols["t3_source"].append(src)
        cols["t3_start"].append(-1)
        cols["t3_step"].append(-1)
        lab["unsupported"].append(bool(abs(a[0]) > KNEE))
        lab["harmful"].append(bool(harm))
        lab["over_pref"].append(float(over))
        sc.append(float(score))

    for p in range(6):
        past, harm = p < 4, p < 3
        cmd = -1.0 if past else -0.5
        add("T1", "command", p, cmd, 2.0 if harm else 0.9, harm, 0.5 if harm else 0.0)
        add("T1", "cap", p, max(cmd, -0.8), 1.0)
        add("T1", "recorded", p, -0.4, 1.2)
        add("T1", "cloned", p, -0.45, 0.3)
        add("T1", "knee", p, max(cmd, -0.81), 1.5 if past else 0.9, harm, 0.4 if harm else 0.0)
    for p in range(4):
        add("T2", "command", p, -0.3, 0.2)
        add("T2", "cap", p, -0.3, 0.2)
        add("T2", "recorded", p, -0.35, 0.1)
        add("T2", "cloned", p, -0.33, 0.1)
    for p in range(6):
        src = "actor" if p < 3 else "controller"
        past, harm = p in (0, 1, 3), p in (0, 3)
        cmd = -1.0 if past else -0.6
        add("T3", "command", p, cmd, 3.0 if harm else (0.5 if past else 0.4), harm, 0.6 if harm else 0.0, src)
        add("T3", "cap", p, max(cmd, -0.8), 1.0 if past else 0.4, src=src)
    for p in range(20):
        add("T4", "recorded", p, -0.3, p / 10.0)
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    arrays["set"] = arrays["set"].astype("<U2")
    arrays["role"] = arrays["role"].astype("<U8")
    arrays["t3_source"] = arrays["t3_source"].astype("<U10")
    arrays["commands"] = arrays["commands"].astype(np.float32)
    labs = dict(unsupported=np.asarray(lab["unsupported"]), harmful=np.asarray(lab["harmful"]),
                over_pref=np.asarray(lab["over_pref"]), critic_pref=np.asarray(lab["over_pref"]),
                true_pref=np.zeros(len(sc)), simulated=np.asarray(lab["unsupported"]))
    return arrays, labs, np.asarray(sc)


def test_basics():
    s = np.arange(1, 5001, dtype=float)
    check(M.conformal_level(s) == 4751.0, "conformal level is the ceil((n+1) 0.95)-th smallest")
    check(M.auroc([2, 1], [True, False]) == 1.0 and M.auroc([1, 2], [True, False]) == 0.0, "auroc order")
    check(M.auroc([1, 1, 1], [True, False, False]) == 0.5, "auroc ties count one half")
    check(M.auroc([1, 2], [True, True]) is None, "auroc with an empty class is None")
    cap = P.cap80(np.array([[0.85, -1.0, 0.8, KNEE]], np.float32))
    check(not np.any(np.abs(cap) > KNEE) and cap[0, 3] == np.float32(0.8), "the cap is never past the knee")


def test_measures():
    arrays, labs, sc = synthetic()
    m = M.compute(arrays, sc, labs, units="value")
    check(m["level"] == 1.9, "level on T4 (n = 20: the 20th smallest)")
    check(m["rate_above_level_T4"] == 0.0, "T4 rate above its own level (n = 20: none above the max)")
    check(m["n_harmful_T1"] == 3 and m["n_past80_T1"] == 4, "T1 counts")
    check(m["widening_T1_harmful"] == 1.0, "widening on harmful T1 pairs (2.0 > 1.0)")
    check(m["widening_T1_past80"] == 0.75, "widening on past-80% T1 pairs (0.9 < 1.0 for the harmless one)")
    check(abs(m["size_T1_raw"] - 2.0) < 1e-12, "size raw = (2 - 1) / 0.5")
    check(abs(m["size_T1_paired"] - 1.6) < 1e-12, "size paired = ([2 - 1.2]_+ - [1 - 1.2]_+) / 0.5")
    check(m["knee_rise_T1_harmful"] == 1.0 and m["knee_rise_T1_harmful_paired"] == 1.0, "knee rise (1.5 > 1.0)")
    check(m["widening_T3_harmful"] == 1.0 and m["n_harmful_T3"] == 2, "T3 widening and count")
    check(m["widening_T3_harmless_past80"] == 0.0, "T3 harmless past-80% pair does not widen (0.5 < 1.0)")
    check(m["catch_rate_T3_harmful"] == 1.0 and m["catch_rate_T3_past80"] == 2 / 3, "T3 catch rates at level 1.9")
    check(m["catch_rate_T3a_past80"] == 0.5 and m["catch_rate_T3b_past80"] == 1.0, "T3a / T3b catch rates")
    check(m["false_flag_T3"] == 0.0 and m["false_flag_T1"] == 0.0, "no harmless pair above 1.9")
    check(m["ranking_T3"] == 1.0, "T3 ranking (harmful 3.0 above every other)")
    h12 = np.flatnonzero(np.isin(arrays["set"], ["T1", "T2"]))
    plan = h12[arrays["role"][h12] != "knee"]
    with_knee = M.auroc(sc[h12], labs["harmful"][h12])
    check(m["ranking_T12"] == 1.0 and abs(m["ranking_T12"] - M.auroc(sc[plan], labs["harmful"][plan])) < 1e-12,
          "T12 ranking over the plan's roles only (1.0: every harmful command above every harmless pair)")
    check(with_knee < 1.0 and abs(m["ranking_T12"] - with_knee) > 1e-3,
          f"the knee pairs would change T12 ranking ({with_knee:.4f} with them): they are left out")
    t1 = np.flatnonzero(arrays["set"] == "T1")
    t1p = t1[arrays["role"][t1] != "knee"]
    check(abs(m["ranking_T1"] - M.auroc(sc[t1p], labs["harmful"][t1p])) < 1e-12, "T1 ranking without the knee pairs")
    pp = h12[(arrays["role"][h12] != "knee") & labs["unsupported"][h12]]
    check(m["ranking_T12_among_past80"] == M.auroc(sc[pp], labs["harmful"][pp]), "among-past-80% pool without knee")
    check("false_flag_T1_knee" not in m and m["n_pool_T12"] == 4 * 6 + 4 * 4, "no knee false flags; pool size")
    check(m["knee_rise_T1_harmful_paired"] == 1.0 and m["knee_fall_T1_harmful_paired"] == 0.0
          and m["knee_tie_T1_harmful_paired"] == 0.0 and abs(m["knee_mean_change_T1_harmful_paired"] - 0.3) < 1e-12,
          "knee measures, paired: W(knee) = 0.3 > W(cap) = 0 on every harmful pair")
    # knee measures with a rise, a tie and a fall (scores changed in a copy)
    def row(st, role, pid):
        return int(np.flatnonzero((arrays["set"] == st) & (arrays["role"] == role) & (arrays["pair"] == pid))[0])
    sc2 = sc.copy()
    sc2[row("T1", "knee", 1)] = 1.1                      # W(knee) 0 = W(cap) 0: a tie; raw +0.1
    sc2[row("T1", "knee", 2)], sc2[row("T1", "cap", 2)] = 1.3, 1.4   # W 0.1 < W 0.2: a fall; raw -0.1
    k = M.compute(arrays, sc2, labs, units="value")
    check(abs(k["knee_rise_T1_harmful_paired"] - 1 / 3) < 1e-12 and abs(k["knee_fall_T1_harmful_paired"] - 1 / 3)
          < 1e-12 and abs(k["knee_tie_T1_harmful_paired"] - 1 / 3) < 1e-12
          and abs(k["knee_mean_change_T1_harmful_paired"] - 0.2 / 3) < 1e-12,
          "knee paired: rise, tie and fall shares 1/3 each, mean change (0.3 + 0 - 0.1) / 3")
    check(abs(k["knee_rise_T1_harmful"] - 2 / 3) < 1e-12 and abs(k["knee_fall_T1_harmful"] - 1 / 3) < 1e-12
          and abs(k["knee_mean_change_T1_harmful"] - 0.5 / 3) < 1e-12, "knee raw: rise 2/3, fall 1/3, mean 0.5/3")
    check(m["flag_rate_past80"] == 5 / 7, "flag rate at past-80% commands of T1 and T3 (3 + 2 of 4 + 3)")
    m2 = M.compute(arrays, sc, labs, units="other")
    check(m2["size_T1_raw"] is None and m2["size_T1_paired"] is None, "no size for a signal not in value units")
    m3 = M.compute(arrays, sc, labs, units="value", level=0.95)
    check(m3["level"] == 0.95 and m3["rate_above_level_T4"] == 0.5, "a given level (Stage 2's tau) is used")
    for k in D.MEASURE_KEYS:
        check(k in m, f"measure key {k} present")


def test_files_and_join():
    out = C.TEST_OUT
    if out.exists():
        shutil.rmtree(out)
    arrays, labs, sc = synthetic()
    f1, c1 = C.save_npz(out / "x" / "a.npz", {"a": np.arange(3)})
    f2, c2 = C.save_npz(out / "x" / "b.npz", {"a": np.arange(3)})
    check(f1 == f2 and c1 == c2, "save_npz is byte-reproducible")
    prec = P.save(arrays, dict(n_pairs=len(sc)), 0, "stall", out)
    now = C.versions()
    check(all(prec.get(k) == v for k, v in now.items()) and prec["label"] == C.LABEL,
          "pair sets stamped with the current spec and manifest, Stage 1 label")
    lrec = LB.save(labs, dict(test=True), prec, 0, "stall", out)
    refuses(lambda: LB.save(labs, {}, prec, 0, "stall", out), "labels are written once")
    d = C.seed_dir(0, "stall", out)

    srec = D.score_signal("synthetic", lambda s, a: sc, Ctx(), "stall", pairs_out=out, units="value", log_fn=None)
    check(srec["reads_labels"] is False and srec["n"] == len(sc), "scores written blind, every pair")
    refuses(lambda: D.score_signal("synthetic", lambda s, a: sc, Ctx(), "stall", pairs_out=out, log_fn=None),
            "scores are written once")
    res = D.measures(d / "scores" / "synthetic.json", d / "labels.json")
    check(res["measures"]["size_T1_paired"] is not None and abs(res["measures"]["size_T1_paired"] - 1.6) < 1e-12,
          "the join reproduces the in-memory measures")
    check(res["joined"]["labels_content_sha256"] == lrec["content_sha256"], "the join records the labels' SHA-256")
    # tamper with the scores file: the join refuses
    tam = out / "tamper"
    shutil.copytree(d, tam / "seed0" / "stall")
    p = tam / "seed0" / "stall" / "scores" / "synthetic.npz"
    b = bytearray(p.read_bytes())
    b[-20] ^= 1
    p.write_bytes(bytes(b))
    refuses(lambda: M.load_join(tam / "seed0" / "stall" / "scores" / "synthetic.json",
                                tam / "seed0" / "stall" / "labels.json"), "the join refuses a tampered score file")
    # a labels record naming other pairs: refuses
    lj = tam / "seed0" / "stall" / "labels.json"
    other = C.read_json(lj)
    other["pairs_content_sha256"] = "f" * 64
    C.write_json(lj, other)
    shutil.copy2(d / "scores" / "synthetic.npz", tam / "seed0" / "stall" / "scores" / "synthetic.npz")
    refuses(lambda: M.load_join(tam / "seed0" / "stall" / "scores" / "synthetic.json", lj),
            "the join refuses scores and labels of different pair sets")
    check(all(res.get(k) == v for k, v in C.versions().items()), "measures stamped with the current versions")
    # records written under another spec or code manifest are never reused
    shutil.rmtree(tam)
    shutil.copytree(d, tam / "seed0" / "stall")
    td = tam / "seed0" / "stall"

    def stale(path, key):
        rec = C.read_json(path)
        rec[key] = "0" * 64
        C.write_json(path, rec)
    stale(td / "pairs.json", "detect_spec_sha256")
    refuses(lambda: P.load(0, "stall", tam), "pair sets of another spec are not reused")
    stale(td / "scores" / "synthetic.json", "detect_code_manifest_sha256")
    refuses(lambda: M.load_join(td / "scores" / "synthetic.json", d / "labels.json"),
            "the join refuses scores of another code manifest")
    stale(td / "labels.json", "detect_spec_sha256")
    refuses(lambda: LB.load(0, "stall", tam), "labels of another spec are not reused")
    refuses(lambda: M.load_join(d / "scores" / "synthetic.json", td / "labels.json"),
            "the join refuses labels of another spec")
    rec = C.read_json(d / "pairs.json")
    del rec["detect_spec_sha256"]
    C.write_json(td / "pairs.json", rec)
    refuses(lambda: P.load(0, "stall", tam), "a record without a version stamp is not reused")


def test_labels_by_folder():
    s2 = C.TEST_OUT / "stage2_standin"                   # a stand-in for runs/shift_explore/stage2
    old = C.STAGE2_OUT
    C.STAGE2_OUT = s2
    try:
        check(C.label_for(s2 / "x") == C.LABEL_STAGE2_FRESH and C.label_for(C.TEST_OUT / "y") == C.LABEL,
              "the default label follows the folder")
        refuses(lambda: C.label_for(s2 / "x", C.LABEL), "the Stage 1 label is refused in Stage 2's folder")
        refuses(lambda: C.label_for(C.OUT / "seed0", C.LABEL_STAGE2), "a Stage 2 label is refused in the rehearsal's")
        refuses(lambda: C.label_for(s2 / "x", "made up"), "a label must start with EXPLORATORY")
        arrays, labs, sc = synthetic()
        refuses(lambda: P.save(arrays, {}, 0, "stall", s2, label=C.LABEL), "pairs.save refuses the Stage 1 label there")
        prec = P.save(arrays, dict(n_pairs=len(sc)), 0, "stall", s2)
        lrec = LB.save(labs, {}, prec, 0, "stall", s2)
        srec = D.score_signal("synthetic", lambda s, a: sc, Ctx(), "stall", pairs_out=s2, units="value", log_fn=None)
        check(prec["label"] == lrec["label"] == srec["label"] == C.LABEL_STAGE2_FRESH,
              "pair sets, labels and stand-in scores in Stage 2's folder carry Stage 2's label (callers pass none)")
        res = D.measures(C.seed_dir(0, "stall", s2) / "scores" / "synthetic.json",
                         C.seed_dir(0, "stall", s2) / "labels.json")
        check(res["label"] == C.LABEL_STAGE2_FRESH, "their measures keep Stage 2's label")
        check(srec["n"] == len(sc), "Stage 2's folder is exempt from the order rule (scores after labels, spec C7)")
    finally:
        C.STAGE2_OUT = old


def test_order_rule_in_module():
    r = C.TEST_OUT / "rehearsal_standin"                 # a stand-in for runs/shift_explore/detect
    C.ORDER_RULE_EXTRA.append(r)
    try:
        arrays, labs, sc = synthetic()
        prec = P.save(arrays, dict(n_pairs=len(sc)), 0, "stall", r)
        d = C.seed_dir(0, "stall", r)
        refuses(lambda: LB.refuse_before_signals(0, "stall", r), "labels wait for signals_done.json (module, D12)")
        D.score_signal("synthetic", lambda s, a: sc, Ctx(), "stall", pairs_out=r, units="value", log_fn=None)
        C.write_json(d / C.SIGNALS_DONE, dict(label=C.LABEL, kind="detect-signals-done", signals=["synthetic"],
                                              **C.stamp()))
        try:
            LB.refuse_before_signals(0, "stall", r)
            check(True, "labels may run once signals_done.json exists")
        except SystemExit as e:
            check(False, f"labels may run once signals_done.json exists ({e})")
        LB.save(labs, {}, prec, 0, "stall", r)
        refuses(lambda: D.score_signal("synthetic_late", lambda s, a: sc, Ctx(), "stall", pairs_out=r,
                                       log_fn=None), "a signal is refused once the labels exist (module, D12)")
        rec = C.read_json(d / C.SIGNALS_DONE)
        rec["detect_code_manifest_sha256"] = "0" * 64
        C.write_json(d / C.SIGNALS_DONE, rec)
        refuses(lambda: LB.refuse_before_signals(0, "stall", r), "a signals_done.json of another manifest is refused")
    finally:
        C.ORDER_RULE_EXTRA.remove(r)


def test_logs():
    p = C.TEST_OUT / "logs" / "x.log"
    C.write_log(p, "line\n")
    data = p.read_bytes()
    side = p.with_name("x.log.sha256").read_text().split()
    check(data.decode().splitlines()[0] == C.LOG_LABEL and side == [C.sha256_bytes(data), "x.log"],
          "a log starts with the EXPLORATORY label and has a matching sidecar")
    sh = (C.HERE / "launch.sh").read_text()
    line = [x for x in sh.splitlines() if x.startswith("LABEL=")][0]
    check(line == f'LABEL="{C.LOG_LABEL}"', "launch.sh writes the same label as common.LOG_LABEL")
    check("D=$OUTROOT/code" in sh and "OUTROOT=" + str(C.OUT) in sh, "launch.sh runs the pinned copies")


def test_guard_and_isolation():
    for p in (C.ROOT / "runs" / "shift" / "x.json", C.ROOT / "experiments" / "x.json", C.ROOM_OUT / "x.json",
              C.ROOT / "runs" / "shift_explore" / "standins" / "x.json", Path("/tmp/x.json")):
        refuses(lambda p=p: C.guard_out(p), f"guard refuses {p}")
    tree = ast.parse((C.HERE / "labels.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    check(names <= {"sys", "time", "numpy", "common", "pairs"}, f"labels.py imports only {sorted(names)}")
    refuses(LB.refuse_if_signals_loaded, "the label script refuses in a process with signal code loaded")


def test_hook():
    arrays, labs, sc = synthetic()

    class W:
        name, sha256 = "bca_width_uniform_fixed", "a" * 64

        def __call__(self, s, a):
            return np.abs(a[:, 0]).astype(np.float64)
    w = W()
    good = "b" * 64
    refuses(lambda: D.score_bca_width(w, Ctx(), "stall", C.OUT / "x", stage2_prereg_sha256=good, version="uniform"),
            "hook refuses an output outside Stage 2's folder")

    class Bad(W):
        name = "distance"
    refuses(lambda: D.score_bca_width(Bad(), Ctx(), "stall", C.TEST_OUT / "s2", stage2_prereg_sha256=good,
                                      version="uniform", pairs_out=C.TEST_OUT), "hook refuses a non-BCA name")
    refuses(lambda: D.score_bca_width(w, Ctx(), "stall", C.TEST_OUT / "s2", stage2_prereg_sha256="x",
                                      version="uniform", pairs_out=C.TEST_OUT), "hook refuses a malformed prereg SHA")
    refuses(lambda: D.score_bca_width(w, Ctx(), "stall", C.TEST_OUT / "s2", stage2_prereg_sha256=good,
                                      version="other", pairs_out=C.TEST_OUT), "hook refuses an unknown version")
    rec = D.score_bca_width(w, Ctx(), "stall", C.TEST_OUT / "s2", stage2_prereg_sha256=good, version="uniform",
                            pairs_out=C.TEST_OUT, log_fn=None)
    check(rec["units"] == "value" and rec["bca_version"] == "uniform" and rec["label"].startswith("EXPLORATORY/STAGE 2"),
          "hook writes a labelled, value-unit width record")
    res = D.measures(C.TEST_OUT / "s2" / f"{W.name}.json", C.seed_dir(0, "stall", C.TEST_OUT) / "labels.json",
                     level=0.85)
    check(res["measures"]["level"] == 0.85, "Stage 2's tau is used as the level")
    contract = C.HERE.parent / "stage2" / "detect_contract.py"
    if contract.exists():
        ns = {}
        exec(compile(contract.read_bytes(), str(contract), "exec"), ns)
        try:
            ns["check_detect_module"](D)
            ns["check_measures"](res["measures"])
            check(True, "Stage 2's contract accepts this module and its measures")
        except SystemExit as e:
            check(False, f"Stage 2's contract: {e}")


if __name__ == "__main__":
    if C.in_rehearsal_folder(C.HERE):
        raise SystemExit("refused: run the synthetic tests from the scratchpad sources, not from the pinned copies")
    test_basics()
    test_measures()
    test_files_and_join()
    test_guard_and_isolation()
    test_hook()
    test_labels_by_folder()
    test_order_rule_in_module()
    test_logs()
    if C.TEST_OUT.exists():
        shutil.rmtree(C.TEST_OUT)
    print(f"{'FAILED: ' + str(len(FAILS)) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
