"""Step 2 reporting add-on: regret, paired standard errors, cost split and lower tail. EXPLORATORY; DECIDES NOTHING.

What it is
  A separate report placed beside step 2 of the restructured BCA (the policy pilot). It never changes NET, G1-G3,
  A1-A3, version 6's marks or step 2's verdict: it only adds information beside them. It reads step 2's frozen code
  by import, after checking the code manifest's SHA-256 against the one pinned below, and it never writes into
  step 2's folders. It was written and committed before step 2's official results existed (README.md beside this
  file); its author never read step 2's official outputs.

What it adds (outputs/shift/oracle/REVIEW_RESPONSE.md, section 4, "Step 2, now")
  1. Regret beside points: the best possible points (the exact optimal reference, outputs/shift/oracle) minus each
     arm's points, for every arm, situation, world and seed, and the total regret across both worlds in strong
     current. NET(x) = total regret of none - total regret of x (checked).
  2. The paired standard error of every pooled difference step 2 reports (NET, G1's comparisons, G2, G3 = M1,
     A1-A3 and the dose readout, M2, M3's clone part, M4, the NET parts and NET in recorded conditions), from one
     extra exact rollout of every scored locked actor on the full banks, keeping per-start returns.
  3. From the same rollouts: tracking cost, energy cost, late peak, 10th percentile and worst-10% mean of per-start
     return.
  4. Labels of the four situations; 5. a fixed text block (NOTES).

Usage (official: only after step 2's results.json exists; pinned to one CPU):
  cd /tmp && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \\
    taskset -c 20 /root/bcavenv/bin/python \\
    /home/user/bca/outputs/shift/beyond-the-data/restructure/step2_report/report.py
  -> runs/shift_explore/restructure/step2_report/{report.json (+ .sha256), report.md, per_start.npz}, write-once.
Scratch (a smoke run's folder and an output folder, both outside the repository):
  ... report.py --root /root/step2_review_scratch/smoke --out /root/step2_report_scratch [--allow-no-results]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path("/home/user/bca")
STEP2_CODE = REPO / "outputs" / "shift" / "beyond-the-data" / "restructure" / "step2"
OFFICIAL_ROOT = REPO / "runs" / "shift_explore" / "restructure" / "step2"
OFFICIAL_OUT = REPO / "runs" / "shift_explore" / "restructure" / "step2_report"
ORACLE_RESULTS = REPO / "outputs" / "shift" / "oracle" / "results.json"
# step 2 as frozen (prespec.json version 2 and its final code manifest); the add-on refuses any other step 2
PINNED_PRESPEC_SHA256 = "839ed716acb7a12646c49295c9319406ce86bba4267b2c9b6829bdb17cd8da04"
PINNED_MANIFEST_SHA256 = "3222aedcfe80b72d95ea3c78ef65a0cecf4a81a76446daccccbed3fd97819fe8"
LATE_PEAK_FROM = 10          # the late peak looks at the states after the 10th step (s_10 .. s_99), as oracle.py
TAIL_SHARE = 0.10            # the 10th percentile and the worst-10% mean
CHECK_TOL = 1e-9             # agreement asked of numbers that must be equal (points, pooled values, anchors)
STATED_TOL = 0.05            # the optimum's points must round to the numbers stated when this add-on was written
# the best possible points as stated (rounded) in outputs/shift/oracle/REVIEW_RESPONSE.md when this was written;
# the exact values are read from outputs/shift/oracle/results.json at run time and compared with these
STATED_OPTIMUM = dict(stall=dict(recorded=100.8, strong_current=109.0, big_waves=102.6, big_kicks=106.7),
                      no_stall=dict(recorded=100.6, strong_current=104.2, big_waves=101.0, big_kicks=103.9))
LABEL = ("EXPLORATORY reporting add-on to step 2 of the restructured BCA (the policy pilot). It decides nothing: NET, "
         "G1-G3, A1-A3, version 6's marks and step 2's verdict are step 2's own and are not changed. Written and "
         "committed before step 2's official results existed.")


def _sha256(b):
    return hashlib.sha256(b).hexdigest()


def _import_step2():
    """step2lib and pooled from step 2's frozen folder, after checking the code manifest against the pinned SHA-256
    and the two files (and prespec.json) against the manifest. Read-only: nothing in that folder is written."""
    raw = (STEP2_CODE / "code_manifest.json").read_bytes()
    if _sha256(raw) != PINNED_MANIFEST_SHA256:
        raise SystemExit("refused: step 2's code_manifest.json is not the frozen one this add-on was written against")
    files = json.loads(raw)["files"]
    for n in ("step2lib.py", "pooled.py", "machinery.py", "prespec.json"):
        if _sha256((STEP2_CODE / n).read_bytes()) != files[n]:
            raise SystemExit(f"refused: step 2's {n} differs from its frozen code manifest")
    if str(STEP2_CODE) not in sys.path:
        sys.path.insert(0, str(STEP2_CODE))
    import step2lib
    import pooled
    for m in (step2lib, pooled):
        if Path(m.__file__).resolve().parent != STEP2_CODE:
            raise SystemExit(f"refused: {m.__name__} was not loaded from step 2's frozen folder")
    return step2lib, pooled


S, POOLED = _import_step2()
SITUATIONS = S.SITUATIONS
WORLDS = S.WORLDS
PRIMARY = S.PRIMARY
ALL_IDS = POOLED.ALL_IDS
ARMS = {"stall": ("none", "full_none") + ALL_IDS, "no_stall": ("none",) + ALL_IDS}
BOTH_WORLDS = ("none",) + ALL_IDS
WORLD_NAME = dict(stall="stalling world", no_stall="no-stall world")
SIT_NAME = dict(recorded="recorded conditions", strong_current="strong current", big_waves="big waves",
                big_kicks="big kicks")

# ------------------------------------------------------------------------------------------------ fixed text

NOTES = (
    "100 points is the stall-aware reference, not the best known controller. In the stalling world it is the "
    "stall-aware controller (the textbook linear rule cut off at 80% throttle); in the no-stall world it is the "
    "rising-thrust controller (the same rule, allowed full throttle). Step 2's frozen PRESPEC.md calls it \"the best "
    "known controller\"; read that as \"the stall-aware reference\". The best possible scores are 100.8 / 109.0 / "
    "102.6 / 106.7 points in the stalling world and 100.6 / 104.2 / 101.0 / 103.9 in the no-stall world (recorded "
    "conditions / strong current / big waves / big kicks); section 1 gives the exact values this report used.",
    "All points are measured on four fixed banks of 1,000 starts, one bank per situation. Every arm, seed and world "
    "is scored on the same starts. The numbers describe these starts; the start-to-start standard error in section 4 "
    "says how much a mean could move with another bank of 1,000 starts.",
    "Penalties are in the critic's discounted value units (one unit of predicted discounted reward), while scores "
    "are undiscounted 100-step points. A penalty's size (the height h, or the dose k*h) therefore does not translate "
    "into points.",
    "BCA treats the 5 calibration rows it takes from each episode as independent. The measured design effect (how "
    "much the likeness of rows from one episode inflates the uncertainty; 1 means none) is 1.0-1.4. Step 1's "
    "certificate survives thinning to one row per episode in all 10 combinations of seed and world; this was "
    "measured with the gate's critic as a stand-in, because step 1's own critics are not on this machine.",
    "Read the attribution ladder support-first. The range hinge and the matched stand-in are the support rule "
    "alone: they look only at where the recorded data are, never at an outcome. obca_forced is BCA's switch test: "
    "it keeps BCA's stalling-world height in both worlds, so in the stalling world it is identical to obca_k1 (same "
    "switch, height and k; section 6 checks the trained actors), and NET(obca_k1) - NET(obca_forced) comes only "
    "from the no-stall world, that is, from BCA switching itself off there. fixed_gated is the fixed height, "
    "meangap_k1 the uncalibrated error scale and conformal_k1 plain split conformal. In the stalling world the best "
    "possible plan never goes past 80% throttle, so refusing unrecorded throttle already has the right shape there: "
    "the stalling half of NET can be won without any calibration, and the no-stall half is what tests whether BCA "
    "knows when not to refuse.",
    "Regret changes no ranking, no difference and no verdict: on a given bank the best possible score is one number "
    "for every arm, so a difference between two arms is the same in regret as in points. In particular "
    "NET(x) = total regret of none - total regret of x (both worlds, strong current).",
)
LABELS = dict(
    recorded="Recorded conditions: the conditions of the training log, inside its range.",
    strong_current="Strong current: currents 0.70-1.10, against the training log's 0.10-0.70. It lies entirely "
                   "outside the training range.",
    big_waves="Big waves: wave sizes 0.40-0.70, against the training log's 0.10-0.40. It lies entirely outside the "
              "training range.",
    big_kicks="Big kicks: almost every start (98.6-99.1%) lies beyond every recorded float state, and the floats "
              "settle back within about 10-30 steps. It is a short excursion beyond the data, not a heavier tail.")
WORDS = (
    ("Points", "the score of an actor on one bank: 0 for the recording controller (the controller that made the "
               "training log), 100 for the stall-aware reference (note 1), from exact 100-step returns."),
    ("Return", "the score of one 100-step episode: the sum of its rewards. Each step's reward is minus (the sum of "
               "squares of the 10 float numbers, the five floats' positions and speeds, plus 0.3 times the sum of "
               "squares of the four throttles). Closer to 0 is better."),
    ("Best possible", "the score of the exact best list of 100 commands from each start (the optimal reference, "
                      "outputs/shift/oracle). No controller can beat it on any start."),
    ("Regret", "best possible points minus the arm's points, on the same bank. 0 would be perfect; smaller is "
               "better."),
    ("Total regret across both worlds", "regret in the stalling world plus regret in the no-stall world, strong "
                                        "current."),
    ("Paired difference", "the difference between two arms measured on the same starts, which cancels most of the "
                          "luck of the starts."),
    ("Standard error (SE)", "how much a mean would typically move if the experiment were repeated with other "
                            "random draws. Section 4 gives three kinds (SE starts, SE seeds, SE)."),
    ("Tracking cost", "per episode, the sum over the 100 steps of the squared distance of the floats from rest "
                      "(the sum of squares of the 10 float numbers): the part of the cost from the floats being "
                      "away from rest."),
    ("Energy cost", "per episode, 0.3 times the sum over the 100 steps of the squared throttles. Tracking cost plus "
                    "energy cost equals minus the return."),
    ("Late peak", "per episode, the largest distance of the floats from rest (the square root of the sum of squares "
                  "of the 10 float numbers) over the states after the 10th step (steps 10-99), once the start's own "
                  "push has had time to die down; then the mean over the bank. Over all steps the largest distance "
                  "is usually the start itself, which no controller changes."),
    ("10th percentile", "of per-start return: the return that 10% of the 1,000 starts fall below (linear "
                        "interpolation, numpy's default)."),
    ("Worst-10% mean", "the mean return of the worst 100 of the 1,000 starts."),
    ("NET", "step 2's score of an arm x: [points of x minus points of none, stalling world] minus [points of none "
            "minus points of x, no-stall world], strong current, mean over seeds."),
    ("G1-G3, A1-A3, M1-M4", "step 2's go/no-go conditions, its attribution comparisons and version 6's marks "
                            "restated (step 2's PRESPEC.md). This report adds standard errors beside the "
                            "differences they read; it does not re-decide them."),
    ("none, clone, full_none", "the actor trained against the frozen critic with no caution; the actor that copies "
                               "the recorded commands; the no-caution actor of a critic trained on the full log "
                               "(stalling world only, a reference)."),
    ("Arms", "obca_k1 (BCA's outcome ramp, the primary arm), obca_k2 (twice its dose), obca_forced (always on), "
             "fixed_gated, conformal_k1, meangap_k1, the matched controls even_pull, push_less, random_1 and "
             "random_2, the matched stand-in (range hinge), the room's range_hinge and the perfect band (a "
             "reference that knows the true error, not a method)."),
)

# ------------------------------------------------------------------------------------------------ pure arithmetic


def inside(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    return p == r or r in p.parents


def paired_se(E):
    """Standard errors of a pooled paired difference, from E (n_seeds, n_starts): E[s, i] is the paired difference
    on start i in seed s (the same n starts serve every seed). The pooled value is the mean of E (= the mean over
    seeds of each seed's bank mean).
      se_starts   = sd_i(mean_s E[s, i]) / sqrt(n_starts): start-to-start; the start luck is shared by the seeds
      se_seeds    = sd_s(mean_i E[s, i]) / sqrt(n_seeds): the plain seed-to-seed SE (None with one seed)
      se          = sqrt(se_starts^2 + se_seeds^2): both together (the usual formula for a seeds-by-starts design;
                    slightly conservative: it counts the seed-by-start interaction once more than needed)
      se_two_way  = sqrt(max(se_seeds^2 + se_starts^2 - MS_interaction / (n_seeds n_starts), 0)): the exact
                    two-way random-effects estimate (crossed seeds and starts), for comparison
      se_within_seed[s] = sd_i(E[s, i]) / sqrt(n_starts): the paired SE inside one seed
    Every sd uses n - 1."""
    E = np.asarray(E, np.float64)
    if E.ndim != 2 or E.shape[1] < 2:
        raise ValueError("E must be (n_seeds, n_starts) with at least 2 starts")
    k, n = E.shape
    d = E.mean(1)
    ebar = E.mean(0)
    value = float(d.mean())
    se_starts = float(ebar.std(ddof=1) / math.sqrt(n))
    within = [float(x) for x in E.std(1, ddof=1) / math.sqrt(n)]
    if k >= 2:
        se_seeds = float(d.std(ddof=1) / math.sqrt(k))
        se = float(math.sqrt(se_starts ** 2 + se_seeds ** 2))
        resid = E - d[:, None] - ebar[None, :] + value
        ms_int = float((resid ** 2).sum() / ((k - 1) * (n - 1)))
        se_two_way = float(math.sqrt(max(se_seeds ** 2 + se_starts ** 2 - ms_int / (k * n), 0.0)))
    else:
        se_seeds = se = se_two_way = None
    return dict(value=value, per_seed=[float(x) for x in d], se_starts=se_starts, se_seeds=se_seeds, se=se,
                se_two_way=se_two_way, se_within_seed=within, n_seeds=int(k), n_starts=int(n))


def tail_stats(J, share=TAIL_SHARE):
    """The lower tail of per-start returns: the `share` quantile (numpy's linear interpolation) and the mean of the
    worst round(share * n) starts."""
    J = np.sort(np.asarray(J, np.float64))
    k = max(1, int(round(share * len(J))))
    return dict(p10=float(np.quantile(J, share)), worst10_mean=float(J[:k].mean()), worst10_count=int(k),
                mean=float(J.mean()), worst=float(J[0]))


def episode_costs(states, commands, J, throttle_cost, length, n_float, late_from=LATE_PEAK_FROM):
    """Per start, from one recorded rollout (truth.rollout(..., record=True)): states (N, L + 1, 13), the applied
    (clipped) commands (N, L, 4) and the returns J (N,). Tracking = sum_{t=0}^{L-1} |s_t|^2 over the first n_float
    state numbers; energy = throttle_cost sum_t |a_t|^2; late peak = max over t = late_from..L-1 of |s_t|. Also the
    largest |tracking + energy + J| (0 up to rounding: tracking + energy = -J)."""
    Sx = np.asarray(states, np.float64)[:, :int(length), :int(n_float)]
    A = np.asarray(commands, np.float64)
    tracking = np.sum(Sx * Sx, axis=(1, 2))
    energy = float(throttle_cost) * np.sum(A * A, axis=(1, 2))
    radius = np.sqrt(np.sum(Sx * Sx, axis=2))
    late_peak = radius[:, int(late_from):].max(1)
    identity = float(np.max(np.abs(tracking + energy + np.asarray(J, np.float64))))
    return dict(tracking=tracking, energy=energy, late_peak=late_peak, identity_error=identity)


def mean_all(xs):
    """The mean when every value is present, else None (step 2's rule: every seed is needed)."""
    return None if not xs or any(x is None for x in xs) else float(np.mean(xs))


def regret_tables(pts, optimum, seeds):
    """Regret = best possible points - points, per world, arm, situation and seed (None where step 2 has no value),
    and the total regret across both worlds in strong current per arm."""
    per_world = {}
    for w in WORLDS:
        per_world[w] = {}
        for arm in ARMS[w]:
            per_world[w][arm] = {}
            for sit in SITUATIONS:
                per = [S.diff(optimum[w][sit], S.pts_of(pts, w, s, arm, sit)) for s in seeds]
                per_world[w][arm][sit] = dict(per_seed=per, mean=mean_all(per))
    total = {}
    for arm in BOTH_WORLDS:
        per = [None if a is None or b is None else a + b for a, b in
               zip(per_world["stall"][arm]["strong_current"]["per_seed"],
                   per_world["no_stall"][arm]["strong_current"]["per_seed"])]
        total[arm] = dict(per_seed=per, mean=mean_all(per))
    references = {w: {sit: dict(best_possible=optimum[w][sit], reference_100_points=optimum[w][sit] - 100.0,
                                recording_controller_0_points=optimum[w][sit]) for sit in SITUATIONS}
                  for w in WORLDS}
    references["total_strong_current"] = dict(
        reference_100_points=sum(optimum[w]["strong_current"] - 100.0 for w in WORLDS),
        recording_controller_0_points=sum(optimum[w]["strong_current"] for w in WORLDS))
    return dict(per_world=per_world, total_strong_current=total, references=references)


def net_identity_error(pts, regret, seeds):
    """The largest |NET(x) per seed - [total regret of none - total regret of x] per seed| over arms and seeds (0 up
    to rounding)."""
    worst = 0.0
    tot = regret["total_strong_current"]
    for arm in ALL_IDS:
        n = S.net(pts, arm, seeds)["per_seed"]
        for i, v in enumerate(n):
            a, b = tot["none"]["per_seed"][i], tot[arm]["per_seed"][i]
            if v is not None and a is not None and b is not None:
                worst = max(worst, abs(v - (a - b)))
    return worst


def net_terms(arm):
    """NET(x) as a sum of arm points: +x - none in the stalling world, +x - none in the no-stall world."""
    return {("stall", arm): 1.0, ("stall", "none"): -1.0, ("no_stall", arm): 1.0, ("no_stall", "none"): -1.0}


def pair_terms(world, a, b):
    return {(world, a): 1.0, (world, b): -1.0}


def combine(*parts):
    """sum_j c_j * terms_j for (c_j, terms_j), dropping terms that cancel."""
    out = {}
    for c, t in parts:
        for key, v in t.items():
            out[key] = out.get(key, 0.0) + float(c) * float(v)
    return {key: v for key, v in out.items() if v != 0.0}


def contrast_specs():
    """Every pooled difference step 2 reports, as a linear combination of arm points (terms) on one situation, with
    where step 2's results.json holds its value (ref)."""
    sc = "strong_current"
    P = PRIMARY
    out = []

    def add(group, name, situation, terms, ref):
        out.append(dict(group=group, name=name, situation=situation, terms=terms, ref=ref))

    for x in ALL_IDS:
        add("NET, strong current", f"NET({x})", sc, net_terms(x), ("net_strong_current", x, "net"))
    for w in S.WORLD_BLIND:
        add("G1: NET(obca_k1) against each world-blind rival", f"NET({P}) - NET({w})", sc,
            combine((1, net_terms(P)), (-1, net_terms(w))), ("G1", w))
    for sit in ("recorded", "big_waves", "big_kicks"):
        add("G2 (and M3's none parts): stalling world, obca_k1 - none", f"{SIT_NAME[sit]}: {P} - none", sit,
            pair_terms("stall", P, "none"), ("G2", sit))
    add("G3 (= M1): stalling world, strong current", f"{P} - even_pull (matched)", sc,
        pair_terms("stall", P, "even_pull"), ("G3",))
    for tag, arm, *_ in S.ATTRIBUTION:
        add("A1-A3 and the dose readout", f"{tag}: NET({arm}) - NET({P})", sc,
            combine((1, net_terms(arm)), (-1, net_terms(P))), ("A", tag))
    add("A1-A3 and the dose readout", f"dose: NET(obca_k2) - NET({P})", sc,
        combine((1, net_terms("obca_k2")), (-1, net_terms(P))), ("dose",))
    for f in ("random_1", "random_2"):
        add("M2: stalling world, strong current", f"{P} - {f} (matched)", sc, pair_terms("stall", P, f), ("M2", f))
    add("M3's clone part: stalling world, recorded conditions", f"{P} - clone", "recorded",
        pair_terms("stall", P, "clone"), ("M3", "recorded_vs_clone"))
    add("M4: stalling world, strong current", f"{P} - standin (matched)", sc, pair_terms("stall", P, "standin"),
        ("M4",))
    for x in ALL_IDS:
        add("NET parts, strong current: stalling gain over none", f"{x} - none, stalling world", sc,
            pair_terms("stall", x, "none"), ("net_strong_current", x, "gain_stall"))
    for x in ALL_IDS:
        add("NET parts, strong current: no-stall loss against none", f"none - {x}, no-stall world", sc,
            pair_terms("no_stall", "none", x), ("net_strong_current", x, "loss_no_stall"))
    for x in ALL_IDS:
        add("NET in recorded conditions (information only in step 2)", f"NET({x}), recorded", "recorded",
            net_terms(x), ("net_recorded_conditions_information_only", x, "net"))
    return out


def step2_value(results, ref):
    """The value step 2's results.json holds for a contrast (None when it holds none, or results is None)."""
    if results is None:
        return None
    try:
        kind = ref[0]
        if kind in ("net_strong_current", "net_recorded_conditions_information_only"):
            v = results[kind][ref[1]][ref[2]]
        elif kind == "G1":
            d = results["go_no_go"]["G1_detail"]
            v = d["net_primary"]["net"] - d["net_world_blind"][ref[1]]["net"]
        elif kind == "G2":
            v = results["go_no_go"]["G2_detail"][ref[1]]["mean"]
        elif kind == "G3":
            v = results["go_no_go"]["G3_detail"]["mean"]
        elif kind == "A":
            a = results["attribution"][ref[1]]
            v = a["net_arm"] - a["net_primary"]
        elif kind == "dose":
            v = results["attribution"]["dose"]["difference"]
        elif kind == "M2":
            v = results["v6_marks_restated"]["M2_random_surfaces"][ref[1]]["mean"]
        elif kind == "M3":
            v = results["v6_marks_restated"]["M3_keeps_improvement"][ref[1]]["mean"]
        elif kind == "M4":
            v = results["v6_marks_restated"]["M4_best_standin"]["mean"]
        else:
            return None
    except (KeyError, TypeError):
        return None
    return None if v is None or isinstance(v, str) else float(v)


def contrast_value_from_means(pts, spec, seeds):
    """The pooled value from step 2's bank means (values.json): the same linear combination, mean over seeds."""
    per = []
    for s in seeds:
        acc = 0.0
        for (w, arm), c in spec["terms"].items():
            p = S.pts_of(pts, w, s, arm, spec["situation"])
            if p is None:
                acc = None
                break
            acc += c * p
        per.append(acc)
    return mean_all(per), per


def contrast_from_starts(per_start, spec, seeds):
    """E (n_seeds, n_starts) of a contrast from per-start points (None when any term is missing in any seed)."""
    rows = []
    for s in seeds:
        acc = None
        for (w, arm), c in spec["terms"].items():
            x = (((per_start.get(w) or {}).get(s) or {}).get(arm) or {}).get(spec["situation"])
            if x is None:
                return None
            acc = c * x if acc is None else acc + c * x
        rows.append(acc)
    return np.stack(rows)


def resolve_paths(root_arg, out_arg, allow_no_results, official_root=OFFICIAL_ROOT, official_out=OFFICIAL_OUT,
                  repo=REPO):
    """(official, step-2 root, output folder). Official mode (no --root): step 2's own output folder, which must
    already hold results.json, and the fixed report folder; --out and --allow-no-results are refused. Scratch mode:
    --root and --out both outside the repository; results.json is required unless --allow-no-results. The report
    is written once (an existing report.json is refused)."""
    official = root_arg is None
    if official:
        if allow_no_results:
            raise SystemExit("refused: --allow-no-results is for scratch runs only (the official report runs only "
                             "after step 2's results.json exists)")
        if out_arg is not None:
            raise SystemExit(f"refused: --out is for scratch runs only; the official report goes to {official_out}")
        root, out = Path(official_root), Path(official_out)
        if not (root / "results.json").exists():
            raise SystemExit(f"refused: step 2's results.json does not exist yet in {root}; this add-on runs only "
                             f"after step 2's own results are written")
        if inside(out, root):
            raise SystemExit(f"refused: {out} lies inside step 2's own output folder")
    else:
        root = Path(root_arg).resolve()
        if inside(root, repo):
            raise SystemExit(f"refused: a scratch --root must lie outside {repo} (got {root}); the official step 2 "
                             f"folder is read only in official mode, after its results exist")
        if out_arg is None:
            raise SystemExit("refused: a scratch run needs --out (a folder outside the repository)")
        out = Path(out_arg).resolve()
        if inside(out, repo):
            raise SystemExit(f"refused: scratch outputs must lie outside {repo} (got {out})")
        if not (root / "results.json").exists() and not allow_no_results:
            raise SystemExit(f"refused: {root}/results.json does not exist (pass --allow-no-results to test on "
                             f"scratch without it)")
    if (out / "report.json").exists():
        raise SystemExit(f"refused: {out}/report.json exists (the report is written once)")
    return official, root, out


def write_once(path, data):
    """Bytes or text, refusing to overwrite (mode 'x')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = data.encode("utf8") if isinstance(data, str) else data
    with open(path, "xb") as handle:
        handle.write(blob)
    return _sha256(blob)


# ------------------------------------------------------------------------------------------------ inputs


def read_oracle(path=ORACLE_RESULTS):
    """The optimal reference's best possible points (bank means), anchors and reference metrics, read only from
    outputs/shift/oracle/results.json (nothing of the oracle is recomputed)."""
    raw = Path(path).read_bytes()
    R = json.loads(raw.decode("utf8"))
    opt = {w: {sit: float(R["optimum"][w][sit]["points"]) for sit in SITUATIONS} for w in WORLDS}
    anchors = {w: {sit: dict(recording=float(R["anchors"][w][sit]["recording"]),
                             top=float(R["anchors"][w][sit]["top"])) for sit in SITUATIONS} for w in WORLDS}
    metrics = {}
    for w in WORLDS:
        metrics[w] = {}
        for sit in SITUATIONS:
            e = R["optimum"][w][sit]
            metrics[w][sit] = dict(tracking_mean=e.get("tracking_cost_mean"), energy_mean=e.get("energy_cost_mean"),
                                   late_peak_mean=e.get("late_peak_deviation_mean"),
                                   return_mean=(e.get("return_") or {}).get("mean"))
    stated = max(abs(opt[w][sit] - STATED_OPTIMUM[w][sit]) for w in WORLDS for sit in SITUATIONS)
    st = os.stat(path)
    return dict(points=opt, anchors=anchors, metrics=metrics, sha256=_sha256(raw), path=str(path),
                modified=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime)),
                stated=STATED_OPTIMUM, largest_difference_from_stated=stated)


def read_step2_results(root, official, spec_sha, man_sha, seeds):
    """Step 2's results.json (sidecar-checked) when it exists: (results or None, its SHA-256 or None, notes)."""
    p = Path(root) / "results.json"
    if not p.exists():
        return None, None, ["results.json not present (scratch, --allow-no-results): step 2's pooled values were "
                            "recomputed from values.json with step2lib's own functions"]
    r = S.read_json_checked(p)
    notes, hard = [], []
    if r.get("prespec_sha256") != spec_sha:
        hard.append("results.json was computed under another prespec.json")
    if r.get("code_manifest_sha256") != man_sha:
        (hard if official else notes).append("results.json was computed under another code manifest")
    if official and (r.get("mode") != "official" or list(r.get("seeds") or []) != list(seeds)):
        hard.append("results.json is not step 2's official pooled look over seeds 0-4")
    if hard:
        raise SystemExit(f"refused: {p}: {hard}")
    return r, S.file_sha256(p), notes


def read_seed_world(root, seed, world, official, spec, spec_sha, man_sha, L):
    """One seed-world: stop.json, or lock.json + values.json (each sidecar-checked) and the locked actors, every
    one checked against the SHA-256 the lock records. Returns a dict (status, lock, values, actors, notes)."""
    d = S.seed_dir(root, seed, world)
    key = f"{world}/seed{seed}"
    st = S.read_json_checked(d / "stop.json", required=False)
    v = S.read_json_checked(d / "values.json", required=False)
    if st is not None and v is not None:
        raise SystemExit(f"refused: {d} has both values.json and stop.json")
    if st is not None:
        if st.get("prespec_sha256") != spec_sha or (official and st.get("mode") != "official"):
            raise SystemExit(f"refused: {d}/stop.json is not this specification's output")
        return dict(key=key, status="stopped", reason=st.get("reason"), stop_sha256=S.file_sha256(d / "stop.json"))
    if v is None:
        raise SystemExit(f"refused: {key} has neither values.json nor stop.json")
    lock = S.read_json_checked(d / "lock.json")
    problems = S.lock_problems(lock, spec_sha, man_sha, seed, world, official, spec["protocol"]["actor_steps"])
    notes = []
    if not official and "trained under another code manifest" in problems:
        problems.remove("trained under another code manifest")
        notes.append("scratch: trained under another code manifest (tolerated in scratch only)")
    lock_sha = S.file_sha256(d / "lock.json")
    if v.get("kind") != "step2-policy-values" or v.get("lock_sha256") != lock_sha:
        problems.append("values.json is not the evaluation of this lock")
    if v.get("prespec_sha256") != spec_sha:
        problems.append("values.json was computed under another prespec.json")
    if v.get("code_manifest_sha256") != man_sha:
        if official:
            problems.append("values.json was computed under another code manifest")
        else:
            notes.append("scratch: values.json computed under another code manifest (tolerated in scratch only)")
    if official and v.get("mode") != "official":
        problems.append("values.json is not official")
    if v.get("seed") != int(seed) or v.get("world") != world:
        problems.append("values.json is for another seed-world")
    if problems:
        raise SystemExit(f"refused: {key}: {problems}")
    acts = L.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    bad = [k for k in lock["names"] if L.tree_sha256(acts[k]) != lock["actor_sha256"][k]]
    if bad:
        raise SystemExit(f"refused: {key}: actors differ from the SHA-256 in lock.json: {bad}")
    return dict(key=key, status="values", lock=lock, values=v, actors=acts, lock_sha256=lock_sha,
                values_sha256=S.file_sha256(d / "values.json"), actors_verified=len(lock["names"]), notes=notes)


# ------------------------------------------------------------------------------------------------ the run


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 2 reporting add-on (decides nothing)")
    ap.add_argument("--root", default=None, help="scratch: a step-2 smoke folder outside the repository")
    ap.add_argument("--out", default=None, help="scratch: the output folder, outside the repository")
    ap.add_argument("--allow-no-results", action="store_true", help="scratch only: run without results.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="scratch only: a subset of seeds 0-4")
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    official, root, out = resolve_paths(a.root, a.out, a.allow_no_results)
    if official and a.seeds is not None:
        raise SystemExit("refused: --seeds is for scratch runs only; the official report covers seeds 0-4")
    spec, spec_sha = S.load_prespec()
    man, man_sha = S.verify_manifest()
    if spec_sha != PINNED_PRESPEC_SHA256 or man_sha != PINNED_MANIFEST_SHA256:
        raise SystemExit("refused: step 2's prespec.json or code manifest is not the frozen one this add-on pins")
    S.check_environment(spec)
    cpu = S.require_one_cpu("the step-2 report")
    if official:
        seeds = [S.check_seed(s) for s in spec["seeds"]]
    elif a.seeds is not None:
        seeds = [S.check_seed(s) for s in a.seeds]
    else:
        seeds = [s for s in S.SEEDS if all(S.seed_dir(root, s, w).is_dir() for w in WORLDS)]
        if not seeds:
            raise SystemExit(f"refused: {root} holds no seed with both worlds")
    results, results_sha, results_notes = read_step2_results(root, official, spec_sha, man_sha, seeds)
    oracle = read_oracle()
    print(f"step-2 report ({'official' if official else 'SCRATCH'}): seeds {seeds}, root {root}, out {out}",
          flush=True)

    import machinery as MC  # noqa: E402  (verified bytes of version 6's code; JAX)
    if MC.SPEC_SHA != spec_sha:
        raise SystemExit("refused: prespec.json changed while starting")
    L, TR, TC = MC.L, MC.TR, MC.TC

    pts = {w: {s: {} for s in seeds} for w in WORLDS}               # step 2's bank means (values.json)
    per_start = {w: {s: {} for s in seeds} for w in WORLDS}         # this report's per-start points
    metrics = {w: {s: {} for s in seeds} for w in WORLDS}           # per seed: tracking, energy, late peak, tail
    status, inputs_sw, notes_sw, rollout_seconds, context_seconds = {}, {}, {}, [], {}
    repro = dict(rollouts=0, J_mean_exactly_equal=0, J_mean_max_abs_difference=0.0, points_max_abs_difference=0.0,
                 identity_max_error=0.0, non_finite_returns=0)
    anchors_diff = 0.0
    forced_equal, actors_verified = {}, 0
    npz = {}
    for s in seeds:
        for w in WORLDS:
            sw = read_seed_world(root, s, w, official, spec, spec_sha, man_sha, L)
            key = sw["key"]
            if sw["status"] == "stopped":
                status[key] = dict(status="stopped", reason=sw["reason"])
                inputs_sw[key] = dict(stop_sha256=sw["stop_sha256"])
                continue
            lock, v, acts = sw["lock"], sw["values"], sw["actors"]
            actors_verified += sw["actors_verified"]
            status[key] = dict(status="values")
            inputs_sw[key] = dict(lock_sha256=sw["lock_sha256"], values_sha256=sw["values_sha256"],
                                  actors_file_sha256=lock["actors_file_sha256"], actors_verified=sw["actors_verified"])
            notes_sw[key] = sw["notes"]
            if w == "stall":
                forced_equal[s] = bool(lock["actor_sha256"].get("obca_forced") == lock["actor_sha256"].get(PRIMARY))
            t0 = time.perf_counter()
            ctx = MC.context(s, w)
            prep = ctx.prepared(w)
            an = ctx.anchors(w)
            n = int(ctx.cfg.bank_size)
            if n != TC.BANK_SIZE:
                raise SystemExit(f"refused: the gate's bank size {n} is not {TC.BANK_SIZE}")
            context_seconds[key] = time.perf_counter() - t0
            for sit in SITUATIONS:
                for k in ("recording", "top"):
                    anchors_diff = max(anchors_diff, abs(float(an[sit][k]) - oracle["anchors"][w][sit][k]))
            by_name = {}
            for name in lock["scored_names"]:
                ev = v["evaluations"].get(name)
                if not bool(lock["finite"][name]) or ev is None:
                    by_name[name] = None
                    continue
                pol = L.mlp_policy(acts[name], prep)
                rec = {}
                for sit in SITUATIONS:
                    t1 = time.perf_counter()
                    run = TR.rollout(pol, TC.bank_starts(sit, n), prep.physics, record=True)
                    rollout_seconds.append(time.perf_counter() - t1)
                    J = np.asarray(run.J, np.float64)
                    c = episode_costs(run.states, run.commands, J, prep.physics.throttle_cost, prep.physics.length,
                                      TC.DS)
                    p = np.asarray(TR.points(J, an[sit]), np.float64)
                    repro["rollouts"] += 1
                    repro["J_mean_exactly_equal"] += int(float(J.mean()) == float(ev[sit]["J"]))
                    repro["J_mean_max_abs_difference"] = max(repro["J_mean_max_abs_difference"],
                                                             abs(float(J.mean()) - float(ev[sit]["J"])))
                    repro["points_max_abs_difference"] = max(repro["points_max_abs_difference"],
                                                             abs(float(p.mean()) - float(ev[sit]["points"])))
                    repro["identity_max_error"] = max(repro["identity_max_error"], c["identity_error"])
                    repro["non_finite_returns"] += int(np.sum(~np.isfinite(J)))
                    tail = tail_stats(J)
                    rec[sit] = dict(points=p, metrics=dict(
                        tracking_mean=float(c["tracking"].mean()), energy_mean=float(c["energy"].mean()),
                        late_peak_mean=float(c["late_peak"].mean()), late_peak_max=float(c["late_peak"].max()),
                        p10=tail["p10"], worst10_mean=tail["worst10_mean"], return_mean=tail["mean"],
                        worst=tail["worst"]))
                    for m, arr in (("J", J), ("tracking", c["tracking"]), ("energy", c["energy"]),
                                   ("late_peak", c["late_peak"])):
                        npz[f"{w}|seed{s}|{name}|{sit}|{m}"] = arr
                by_name[name] = rec
                print(f"  {key} {name}: rolled out", flush=True)
            for arm, name in lock["arm_ids"].items():
                pts[w][s][arm] = None if name is None else {
                    sit: v["situations"][sit]["points"].get(name) for sit in SITUATIONS}
                r = None if name is None else by_name.get(name)
                if r is not None:
                    per_start[w][s][arm] = {sit: r[sit]["points"] for sit in SITUATIONS}
                    metrics[w][s][arm] = {sit: r[sit]["metrics"] for sit in SITUATIONS}
            del acts, sw

    # 1. regret beside points
    regret = regret_tables(pts, oracle["points"], seeds)
    net_identity = net_identity_error(pts, regret, seeds)
    points_table = {w: {arm: {sit: dict(per_seed=[S.pts_of(pts, w, s, arm, sit) for s in seeds],
                                        mean=mean_all([S.pts_of(pts, w, s, arm, sit) for s in seeds]))
                              for sit in SITUATIONS} for arm in ARMS[w]} for w in WORLDS}

    # 2. paired standard errors
    contrasts, pooled_vs_values, pooled_vs_results, compared_with_results = [], 0.0, 0.0, 0
    for spec_c in contrast_specs():
        E = contrast_from_starts(per_start, spec_c, seeds)
        st = None if E is None else paired_se(E)
        v_means, v_per = contrast_value_from_means(pts, spec_c, seeds)
        v_res = step2_value(results, spec_c["ref"])
        if st is not None and v_means is not None:
            pooled_vs_values = max(pooled_vs_values, abs(st["value"] - v_means))
        if results is not None and v_means is not None and v_res is not None:
            pooled_vs_results = max(pooled_vs_results, abs(v_means - v_res))
            compared_with_results += 1
        contrasts.append(dict(group=spec_c["group"], name=spec_c["name"], situation=spec_c["situation"],
                              terms=[dict(world=w, arm=arm, coefficient=c) for (w, arm), c in spec_c["terms"].items()],
                              value_step2=v_res, value_from_values=v_means, per_seed_from_values=v_per,
                              **(st or dict(value=None, per_seed=None, se_starts=None, se_seeds=None, se=None,
                                            se_two_way=None, se_within_seed=None))))

    # 3. tracking, energy, late peak and the lower tail, mean over seeds
    keys_m = ("tracking_mean", "energy_mean", "late_peak_mean", "late_peak_max", "p10", "worst10_mean", "return_mean")
    episode = {}
    for w in WORLDS:
        episode[w] = {}
        for arm in ARMS[w]:
            episode[w][arm] = {}
            for sit in SITUATIONS:
                per = [((metrics[w][s].get(arm) or {}).get(sit)) for s in seeds]
                mean = None if any(x is None for x in per) else {
                    k: (float(np.max([x[k] for x in per])) if k == "late_peak_max" else
                        float(np.mean([x[k] for x in per]))) for k in keys_m}
                episode[w][arm][sit] = dict(per_seed=per, mean=mean)

    # checks (reported; a failed check is flagged in report.md, never hidden)
    checks = dict(
        frozen_step2=dict(passed=True, prespec_sha256=spec_sha, code_manifest_sha256=man_sha),
        locks_and_actors=dict(passed=True, actors_verified_against_lock=actors_verified,
                              seed_worlds={k: x["status"] for k, x in status.items()}),
        rollouts_reproduce_step2=dict(passed=bool(repro["rollouts"] > 0 and repro["J_mean_exactly_equal"] ==
                                                  repro["rollouts"] and repro["points_max_abs_difference"] <= CHECK_TOL
                                                  and repro["non_finite_returns"] == 0), **repro),
        tracking_plus_energy_is_minus_return=dict(passed=bool(repro["identity_max_error"] <= 1e-6),
                                                  max_error=repro["identity_max_error"]),
        anchors_equal_the_oracle_anchors=dict(passed=bool(anchors_diff <= CHECK_TOL), max_abs_difference=anchors_diff),
        oracle_points_as_stated=dict(passed=bool(oracle["largest_difference_from_stated"] <= STATED_TOL),
                                     largest_difference=oracle["largest_difference_from_stated"], tolerance=STATED_TOL,
                                     stated=STATED_OPTIMUM),
        net_equals_total_regret_difference=dict(passed=bool(net_identity <= CHECK_TOL), max_abs_difference=net_identity),
        pooled_values_from_rollouts_equal_values_json=dict(passed=bool(pooled_vs_values <= CHECK_TOL),
                                                           max_abs_difference=pooled_vs_values),
        pooled_values_equal_results_json=(
            dict(passed=bool(pooled_vs_results <= CHECK_TOL), max_abs_difference=pooled_vs_results,
                 compared=compared_with_results) if results is not None else
            dict(passed=None, note="results.json not present (scratch, --allow-no-results)")),
        obca_forced_actor_equals_obca_k1_stalling_world=dict(passed=None, per_seed=forced_equal,
                                                             note="reported fact, not a requirement"))
    failed = [k for k, c in checks.items() if c.get("passed") is False]

    verified = MC.check_imports()
    t_roll = float(np.sum(rollout_seconds)) if rollout_seconds else 0.0
    timing = dict(rollouts=len(rollout_seconds), rollout_seconds_total=t_roll,
                  seconds_per_rollout_mean=float(np.mean(rollout_seconds)) if rollout_seconds else None,
                  seconds_per_rollout_max=float(np.max(rollout_seconds)) if rollout_seconds else None,
                  rollout_means="one rollout = one actor on one situation's full bank of 1,000 starts, 100 steps",
                  context_seconds=context_seconds, total_seconds=None, vmhwm_mb=S.vmhwm_mb())
    verdict = None
    if results is not None:
        g = results.get("go_no_go") or {}
        verdict = dict(verdict=g.get("verdict"), G1=g.get("G1"), G2=g.get("G2"), G3=g.get("G3"),
                       note="copied from step 2's results.json, unchanged; this report does not re-decide it")
    report = dict(
        label=LABEL, kind="step2-report", mode="official" if official else "SCRATCH (does not count)",
        decides="nothing: NET, G1-G3, A1-A3, version 6's marks and step 2's verdict are step 2's own", seeds=seeds,
        step2_verdict_copied=verdict, notes=list(NOTES), situation_labels=LABELS, words=[list(x) for x in WORDS],
        optimum=dict(points=oracle["points"], stated=STATED_OPTIMUM, reference_metrics=oracle["metrics"],
                     source=dict(path=oracle["path"], sha256=oracle["sha256"], modified=oracle["modified"])),
        points=points_table, regret=regret, paired_differences=contrasts, episode_metrics=episode,
        definitions=dict(
            regret="best possible points (bank mean, outputs/shift/oracle/results.json optimum.<world>.<situation>."
                   "points) minus the arm's points (step 2's values.json)",
            paired_se=paired_se.__doc__, tail=tail_stats.__doc__, costs=episode_costs.__doc__,
            late_peak_from_step=LATE_PEAK_FROM, tail_share=TAIL_SHARE),
        checks=checks, failed_checks=failed, status=status,
        inputs=dict(step2_root=str(root), prespec_sha256=spec_sha, code_manifest_sha256=man_sha,
                    results_json_sha256=results_sha, results_notes=results_notes, seed_worlds=inputs_sw,
                    seed_world_notes=notes_sw, report_py_sha256=_sha256(Path(__file__).read_bytes())),
        timing=timing, verified_imports=verified, cpus=cpu, machine=platform.node(), python=platform.python_version(),
        numpy=np.__version__, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    out.mkdir(parents=True, exist_ok=True)
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **npz)
    report["inputs"]["per_start_npz_sha256"] = write_once(out / "per_start.npz", buf.getvalue())
    report["timing"]["total_seconds"] = time.perf_counter() - t_start
    text = json.dumps(S.jsonable(report), indent=1, sort_keys=True, allow_nan=False)
    digest = write_once(out / "report.json", text)
    write_once(out / "report.json.sha256", f"{digest}  report.json\n")
    try:
        md = report_md(json.loads(text), digest)
    except Exception as e:          # the markdown must never cost the numbers (report.json is already written)
        md = f"# Step 2 add-on report\n\nThe report could not be formatted ({e!r}); read report.json ({digest}).\n"
    write_once(out / "report.md", md)
    print(f"report written: {out}/report.json ({digest}); failed checks: {failed or 'none'}; "
          f"{len(rollout_seconds)} rollouts, {t_roll:.0f}s of rollouts, {time.perf_counter() - t_start:.0f}s in all",
          flush=True)
    return report


# ------------------------------------------------------------------------------------------------ report.md


def f_(x, nd=1, sign=False):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, str):
        return x
    x = float(x)
    if not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def report_md(r, digest):
    scratch = r["mode"] != "official"
    seeds = r["seeds"]
    opt = r["optimum"]["points"]
    reg = r["regret"]
    pts = r["points"]
    a = []
    a.append(f"# Step 2 add-on: regret, paired standard errors, cost split and lower tail"
             f"{' (SCRATCH: does not count)' if scratch else ''}")
    a += ["", r["label"], "",
          f"Mode: {r['mode']}. Seeds {seeds}. report.json SHA-256 {digest}. Written {r['written']}.", ""]
    v = r.get("step2_verdict_copied")
    if v:
        a += [f"**This report decides nothing.** Step 2's verdict, copied from its results.json and unchanged: "
              f"**{v['verdict']}** (G1 {f_(v['G1'])}, G2 {f_(v['G2'])}, G3 {f_(v['G3'])}).", ""]
    else:
        a += ["**This report decides nothing.** Step 2's results.json was not read (scratch, --allow-no-results).", ""]
    if r["failed_checks"]:
        a += [f"**CHECKS THAT DID NOT PASS: {', '.join(r['failed_checks'])}** (section 6). Read the numbers below "
              f"with that in mind.", ""]
    else:
        a += ["Every check in section 6 passed.", ""]
    a += ["## Read this first", ""]
    a += [f"{i}. {t}" for i, t in enumerate(r["notes"], 1)]
    a += ["", "## The four situations", ""]
    a += [f"- {r['situation_labels'][s]}" for s in SITUATIONS]
    a += ["", "## Words used here", ""]
    a += [f"- **{k}**: {t}" for k, t in r["words"]]

    a += ["", "## 1. The best possible points", "",
          "From the exact optimal reference (outputs/shift/oracle/results.json, bank means; nothing of it was "
          "recomputed here). 100 points is the stall-aware reference, not the best possible.", "",
          "| world | recorded | strong current | big waves | big kicks |", "|---|---|---|---|---|"]
    for w in WORLDS:
        a.append(f"| {WORLD_NAME[w]} | " + " | ".join(f_(opt[w][s], 2) for s in SITUATIONS) + " |")
    src = r["optimum"]["source"]
    a += ["", f"Read from {src['path']} (SHA-256 {src['sha256'][:16]}..., modified {src['modified']}).", ""]

    def cell(w, arm, sit):
        p = pts[w].get(arm, {}).get(sit, {}).get("mean")
        g = reg["per_world"][w].get(arm, {}).get(sit, {}).get("mean")
        return f"{f_(p)} / {f_(g)}"

    a += ["## 2. Regret beside points, strong current", "",
          f"Each cell is points / regret, mean over seeds {seeds}. The points are step 2's own (values.json); the "
          f"regret is the best possible points (section 1) minus them. Smaller regret is better. \"Both worlds\" is "
          f"the total regret (stalling plus no-stall). NET(x) = total regret of none - total regret of x.", "",
          "| arm | stalling world | no-stall world | both worlds |", "|---|---|---|---|"]
    refs = reg["references"]
    a.append(f"| 100-point reference | 100.0 / {f_(refs['stall']['strong_current']['reference_100_points'])} | "
             f"100.0 / {f_(refs['no_stall']['strong_current']['reference_100_points'])} | "
             f"{f_(refs['total_strong_current']['reference_100_points'])} |")
    a.append(f"| recording controller | 0.0 / {f_(refs['stall']['strong_current']['recording_controller_0_points'])} | "
             f"0.0 / {f_(refs['no_stall']['strong_current']['recording_controller_0_points'])} | "
             f"{f_(refs['total_strong_current']['recording_controller_0_points'])} |")
    for arm in ARMS["stall"]:
        ns = cell("no_stall", arm, "strong_current") if arm in ARMS["no_stall"] else "-"
        tot = reg["total_strong_current"].get(arm, {}).get("mean")
        a.append(f"| {arm} | {cell('stall', arm, 'strong_current')} | {ns} | {f_(tot)} |")
    a += ["", "Per-seed values of every arm, situation and world are in report.json (points, regret).", ""]
    a += ["## 3. Regret beside points, the other situations", ""]
    for w in WORLDS:
        a += [f"**{WORLD_NAME[w].capitalize()}** (points / regret, mean over seeds)", "",
              "| arm | recorded | big waves | big kicks |", "|---|---|---|---|"]
        a.append("| 100-point reference | " + " | ".join(
            f"100.0 / {f_(refs[w][s]['reference_100_points'])}" for s in ("recorded", "big_waves", "big_kicks")) + " |")
        for arm in ARMS[w]:
            a.append(f"| {arm} | " + " | ".join(cell(w, arm, s) for s in ("recorded", "big_waves", "big_kicks")) + " |")
        a.append("")

    a += ["## 4. Paired standard errors of step 2's pooled differences", "",
          "How they are computed. One extra exact rollout of every scored locked actor on the full banks keeps each "
          "start's return. For one difference (for example obca_k1 minus none, or a NET), each start i of the bank "
          "gives a paired difference e(s, i) in seed s: the arms' points on that same start, combined the way step 2 "
          "combines the bank means (for a NET: [x - none, stalling world] + [x - none, no-stall world], on the same "
          "start, since both worlds use the same bank). The value is the mean over seeds and starts, which equals "
          "step 2's number (checked in section 6).", "",
          "- **SE starts** = sd over starts of (the mean over seeds of e(s, i)) / sqrt(1,000): how much the value "
          "would move with another bank of 1,000 starts. The same bank serves every seed, so the start luck is "
          "shared by the seeds and is counted once.",
          "- **SE seeds** = sd over seeds of (the mean over starts of e(s, i)) / sqrt(number of seeds): the plain "
          "seed-to-seed standard error, how much the value would move with another set of seeds on this bank.",
          "- **SE** = sqrt(SE starts^2 + SE seeds^2): both kinds of luck together (the usual formula for a "
          "seeds-by-starts design). It is slightly conservative: it counts the seed-by-start interaction once more "
          "than an exact two-way analysis, which report.json also gives (se_two_way).",
          "- Every sd divides by n - 1. With one seed, SE seeds and SE are not defined (\"-\").", ""]
    groups = []
    for c in r["paired_differences"]:
        if c["group"] not in groups:
            groups.append(c["group"])
    for g in groups:
        a += [f"**{g}**", "", "| difference | value | SE | SE starts | SE seeds |", "|---|---|---|---|---|"]
        for c in r["paired_differences"]:
            if c["group"] == g:
                val = c["value"] if c["value"] is not None else c["value_from_values"]
                a.append(f"| {c['name']} | {f_(val, 1, True)} | {f_(c['se'], 2)} | {f_(c['se_starts'], 2)} | "
                         f"{f_(c['se_seeds'], 2)} |")
        a.append("")

    a += ["## 5. Where the cost comes from, and the lower tail", "",
          f"From the same rollouts, mean over seeds {seeds}, in return units per episode (not points). Tracking + "
          f"energy = minus the mean return. Late peak: mean over starts (largest over starts in report.json). "
          f"10th percentile and worst-10% mean: of per-start return (more negative is worse). The best possible "
          f"row is from the oracle's results.json, which has no 10th percentile or worst-10% mean.", ""]

    def metric_table(w, sit):
        rows = [f"**{WORLD_NAME[w].capitalize()}, {SIT_NAME[sit]}**", "",
                "| arm | tracking | energy | late peak | 10th pct | worst 10% |", "|---|---|---|---|---|---|"]
        om = r["optimum"]["reference_metrics"][w][sit]
        rows.append(f"| best possible | {f_(om['tracking_mean'])} | {f_(om['energy_mean'])} | "
                    f"{f_(om['late_peak_mean'], 2)} | - | - |")
        for arm in ARMS[w]:
            m = r["episode_metrics"][w][arm][sit]["mean"]
            if m is None:
                rows.append(f"| {arm} | - | - | - | - | - |")
            else:
                rows.append(f"| {arm} | {f_(m['tracking_mean'])} | {f_(m['energy_mean'])} | "
                            f"{f_(m['late_peak_mean'], 2)} | {f_(m['p10'])} | {f_(m['worst10_mean'])} |")
        return rows + [""]

    for w in WORLDS:
        a += metric_table(w, "strong_current")

    ch = r["checks"]
    rp = ch["rollouts_reproduce_step2"]
    a += ["## 6. Checks", "",
          f"- Step 2 as frozen: prespec.json {r['inputs']['prespec_sha256'][:16]}..., code manifest "
          f"{r['inputs']['code_manifest_sha256'][:16]}... (both pinned in report.py): passed.",
          f"- Every lock's actors checked against the SHA-256 in lock.json: {ch['locks_and_actors']['actors_verified_against_lock']} "
          f"actors, passed. Seed-worlds: " + ", ".join(f"{k} {x}" for k, x in ch["locks_and_actors"]["seed_worlds"].items()) + ".",
          f"- The extra rollouts reproduce step 2's evaluation: {rp['J_mean_exactly_equal']} of {rp['rollouts']} bank "
          f"means of the return exactly equal to values.json; largest points difference "
          f"{rp['points_max_abs_difference']:.1e}: {f_(rp['passed'])}.",
          f"- Tracking + energy = minus the return: largest error {ch['tracking_plus_energy_is_minus_return']['max_error']:.1e}: "
          f"{f_(ch['tracking_plus_energy_is_minus_return']['passed'])}.",
          f"- Step 2's anchors equal the oracle's: largest difference "
          f"{ch['anchors_equal_the_oracle_anchors']['max_abs_difference']:.1e}: "
          f"{f_(ch['anchors_equal_the_oracle_anchors']['passed'])}.",
          f"- The oracle's best possible points equal those stated when this add-on was written (to "
          f"{STATED_TOL}): largest difference {ch['oracle_points_as_stated']['largest_difference']:.3f}: "
          f"{f_(ch['oracle_points_as_stated']['passed'])}.",
          f"- NET = total regret of none - total regret of x: largest difference "
          f"{ch['net_equals_total_regret_difference']['max_abs_difference']:.1e}: "
          f"{f_(ch['net_equals_total_regret_difference']['passed'])}.",
          f"- Pooled values from the per-start rollouts equal those from values.json: largest difference "
          f"{ch['pooled_values_from_rollouts_equal_values_json']['max_abs_difference']:.1e}: "
          f"{f_(ch['pooled_values_from_rollouts_equal_values_json']['passed'])}."]
    pr = ch["pooled_values_equal_results_json"]
    if pr.get("passed") is None:
        a.append(f"- Pooled values equal step 2's results.json: not compared ({pr['note']}).")
    else:
        a.append(f"- Pooled values equal step 2's results.json: {pr['compared']} compared, largest difference "
                 f"{pr['max_abs_difference']:.1e}: {f_(pr['passed'])}.")
    fe = ch["obca_forced_actor_equals_obca_k1_stalling_world"]["per_seed"]
    a.append("- Stalling world, obca_forced's trained actor is bit-identical to obca_k1's (same SHA-256): " +
             ", ".join(f"seed {s} {f_(x)}" for s, x in fe.items()) + " (a reported fact).")
    extra = list(r["inputs"]["results_notes"]) + [f"{k}: {n}" for k, ns in r["inputs"]["seed_world_notes"].items()
                                                  for n in ns]
    if extra:
        a.append("- Notes on the inputs: " + "; ".join(extra) + ".")
    t = r["timing"]
    a += ["", "## 7. Time", "",
          f"- {t['rollouts']} rollouts ({t['rollout_means']}): {f_(t['rollout_seconds_total'], 0)} s, "
          f"{f_(t['seconds_per_rollout_mean'], 2)} s each on average (largest {f_(t['seconds_per_rollout_max'], 2)} s).",
          f"- Loading the gate's data per seed-world: " +
          ", ".join(f"{k} {x:.0f} s" for k, x in t["context_seconds"].items()) + ".",
          f"- In all {f_(t['total_seconds'], 0)} s; peak memory {f_(t['vmhwm_mb'], 0)} MB.", ""]
    a += ["## Appendix: the other situations", ""]
    for w in WORLDS:
        for sit in ("recorded", "big_waves", "big_kicks"):
            a += metric_table(w, sit)
    a += ["## Files", "",
          "- report.json: every number (per seed, per arm, per situation), the checks, the inputs' SHA-256.",
          "- per_start.npz: the per-start return, tracking cost, energy cost and late peak of every rollout (keys "
          "world|seed|actor|situation|quantity).",
          "- The code: outputs/shift/beyond-the-data/restructure/step2_report/report.py (README.md beside it).", ""]
    return "\n".join(a)


if __name__ == "__main__":
    main()
