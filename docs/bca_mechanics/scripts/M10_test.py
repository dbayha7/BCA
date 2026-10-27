r"""Module M10 of MODULES.md, the harm switch certificate: synthetic tests with a known truth, the real-data test on the
recorded near-80 and step-1 certificates, a table and the figure docs/bca_mechanics/figures/M10.svg.

WHAT IT RUNS
  Part A  synthetic data (the truth is known): a known gap between two groups' 90% quantiles at the recorded spread
          of the scores, the false-on rate with no gap, the refusal boundary by group size against the binomial
          prediction, the shuffle control, the dependence on the random stream, and the gap rule on synthetic throttles.
  Part B  recorded data: the 40 recorded certificates (near-80 switch.json x 30, step 1 results.json x 10) and the
          ordering q05 <= plain gap <= height, the pinned-environment recomputation of the 30 near-80 certificates
          (data/M10/recomputed_certificates.json, made by M10_recompute_pinned.py through the WSL bridge), the same
          certificates recomputed HERE from the saved per-row overestimates (data/M10/overestimates.npz), the shuffle
          control on those scores, and the certificate under 20 other random streams (does the decision depend on
          the stream?).
  Part C  the figure and the table.
The certificate code is the project's own: outputs/shift/beyond-the-data/restructure/step1/step1lib.py (imported, not
copied) on top of calibration/wbcp.py. Nothing is trained and no critic is touched here.

COMMAND (Windows, from the repository root; the paths below are the ones used on 20 October 2026)
  set PYTHONPATH=M:\
  set PYTHONDONTWRITEBYTECODE=1
  C:\...\scratchpad\venv\Scripts\python.exe docs\bca_mechanics\scripts\M10_test.py [--root M:\] [--fast] [--no-figure]
ENVIRONMENT USED
  Python 3.14, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2 (Windows 11). The script also runs in the pinned
  environment (Python 3.10.20, numpy 2.2.6). Only the recomputation of the per-row overestimates needs JAX and the
  pinned environment; that step is M10_recompute_pinned.py + M10_recompute_job.sh and its outputs are committed in
  data/M10/, so this script runs from the repository alone.
OUTPUTS
  prints the tables; writes figures/M10.svg and data/M10/M10_test_results.json (every number printed).
  --fast uses fewer repetitions (for a quick look; the committed numbers are from the full run).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, ".."))
parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.path.abspath(os.path.join(DOCS, "..", "..")), help="repository root")
parser.add_argument("--fast", action="store_true", help="fewer repetitions")
parser.add_argument("--no-figure", action="store_true")
parser.add_argument("--png", default=None, help="also save the figure as PNG at this path (for a quick look)")
parser.add_argument("--figure-only", action="store_true", help="redraw the figure from data/M10/M10_test_results.json")
ARGS = parser.parse_args()
ROOT = ARGS.root
STEP1_DIR = os.path.join(ROOT, "outputs", "shift", "beyond-the-data", "restructure", "step1")
NEAR80 = os.path.join(ROOT, "outputs", "shift", "beyond-the-data", "near80", "results-20261020T115246Z", "files")
STEP1_RESULTS = os.path.join(STEP1_DIR, "results", "results.json")
DATA = os.path.join(DOCS, "data", "M10")
FIG = os.path.join(DOCS, "figures", "M10.svg")

sys.path.insert(0, ROOT)
sys.path.insert(0, STEP1_DIR)
import step1lib as S  # noqa: E402  (the certificate code itself: throttle, gap_rule, certificate, shuffled_certificates)
from calibration import wbcp  # noqa: E402

# The constants of the near-80 run (near80_constants.py:109-111) and of step 1 (prespec.json): copied here so that the
# script needs no JAX import; checked against step 1's prespec.json below.
CFG = dict(alpha=0.1, credibility=0.95, draws=1000, statistic_quantile=0.05, margin=0.10)
GAP_RULE = dict(upper_quantile=0.99, min_gap=0.05, min_tail=9)
CERT_TAG_HEAD = (20261015, 9700)      # certificate streams: [20261015, 9700, seed, world tag] + [1] (tail) / [2] (rest)
PERM_TAG_HEAD = (20261015, 9800)      # step 1's shuffle streams: [20261015, 9800, seed, world tag, j + 1]
WORLD_TAG = {"stall": 1, "no_stall": 2}
SCALE = 0.15                          # the recorded spread (standard deviation) of the rest group's scores: 0.12-0.18
#                                       value units in all 30 near-80 seed-worlds (data/M10/overestimates.npz); the
#                                       margin 0.10 is about two thirds of it
N_TAIL, N_REST = 40, 830              # typical recorded group sizes (tails 18-60, rest 803-878)
R = (dict(known_gap=40, false_on=60, refusal=40, shuffle=50, streams=20) if ARGS.fast
     else dict(known_gap=200, false_on=300, refusal=200, shuffle=200, streams=20))
OUT = dict(settings=dict(cfg=CFG, gap_rule=GAP_RULE, reps=R, numpy=np.__version__, python=sys.version.split()[0]))
T0 = time.time()


def say(*a):
    print(*a, flush=True)


def num(v):
    """Records write non-finite numbers as the strings 'inf', '-inf', 'nan'."""
    return float(v) if isinstance(v, str) else v


def p_finite(n, draws=1000, alpha=0.1, beta=0.95):
    """M5's rule: a draw's crossing is finite with probability 1 - (1 - alpha)^n; the credible quantile (and so R)
    is finite iff at least ceil(beta * draws) of the draws are finite (calibration/reference.py:124-136)."""
    need = math.ceil(beta * draws)
    return float(stats.binom.sf(need - 1, draws, 1.0 - (1.0 - alpha) ** n))


def check_prespec():
    p = json.load(open(os.path.join(STEP1_DIR, "prespec.json"), encoding="utf8"))
    c = p.get("certificate", {})
    got = dict(alpha=c.get("alpha"), credibility=c.get("credibility"), draws=c.get("draws"),
               statistic_quantile=c.get("statistic_quantile"), margin=c.get("margin"))
    OUT["settings"]["prespec_certificate"] = got
    same = all(got.get(k) == v for k, v in CFG.items() if got.get(k) is not None)
    say(f"step 1 prespec.json certificate block: {got}; equal to CFG where present: {same}")


# ================================================================================================ PART A: synthetic
def synthetic_groups(rng, gap, n_tail, n_rest=850, scale=1.0):
    """Two groups whose 90% quantiles differ by exactly `gap`: rest ~ N(0, scale), tail ~ N(gap, scale)."""
    return rng.normal(gap, scale, n_tail), rng.normal(0.0, scale, n_rest)


def cert(tail, rest, tags):
    return S.certificate(tail, rest, list(tags), CFG)


def part_a():
    say("\n" + "=" * 100 + "\nPART A  synthetic tests (the truth is known)\n" + "=" * 100)
    A = OUT["A"] = {}

    # A1 a known gap between the two 90% quantiles ------------------------------------------------------------
    say(f"\nA1  a known gap: tail ~ N(gap, {SCALE}) with {N_TAIL} rows, rest ~ N(0, {SCALE}) with {N_REST} rows (the recorded "
        f"spread and sizes); the true gap between the two 90% quantiles IS `gap`; {R['known_gap']} data sets per gap, "
        "certificate streams as in the run")
    say(f"{'gap':>5} {'on rate':>8} {'q05<=gap':>9} {'q05 median':>11} {'height-gap median':>18} {'height-gap 5%..95%':>20} {'finite':>7}")
    a1 = A["A1_known_gap"] = []
    for gap in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0):
        rows = []
        for r in range(R["known_gap"]):
            rng = np.random.default_rng([101, int(round(gap * 100)), r])
            tail, rest = synthetic_groups(rng, gap, N_TAIL, N_REST, SCALE)
            c = cert(tail, rest, (*CERT_TAG_HEAD, r, 1))
            rows.append((c["on"], c["q05"], c["height"]))
        on = np.array([x[0] for x in rows]); q05 = np.array([x[1] for x in rows]); h = np.array([x[2] for x in rows])
        fin = np.isfinite(h)
        d = (h - gap)[fin]
        rec = dict(gap=gap, scale=SCALE, n_tail=N_TAIL, n_rest=N_REST, on_rate=float(on.mean()),
                   q05_at_most_gap=float(np.mean(q05 <= gap)), q05_median=float(np.median(q05)),
                   q05_q05=float(np.quantile(q05, 0.05)), q05_q95=float(np.quantile(q05, 0.95)),
                   height_minus_gap_median=float(np.median(d)),
                   height_minus_gap_q05=float(np.quantile(d, 0.05)), height_minus_gap_q95=float(np.quantile(d, 0.95)),
                   finite_share=float(fin.mean()), n=len(rows))
        a1.append(rec)
        say(f"{gap:5.2f} {rec['on_rate']:8.3f} {rec['q05_at_most_gap']:9.3f} {rec['q05_median']:11.3f} "
            f"{rec['height_minus_gap_median']:18.3f} {rec['height_minus_gap_q05']:9.3f}..{rec['height_minus_gap_q95']:<9.3f} {rec['finite_share']:7.3f}")

    # A2 the false-on rate with no gap, by group size ------------------------------------------------------------
    say(f"\nA2  no gap at all (both groups N(0, {SCALE}), rest {N_REST} rows); the switch should stay off: {R['false_on']} data sets per size")
    say(f"{'n_tail':>6} {'on rate':>8} {'q05 95th pct':>13} {'finite height':>14} {'binomial prediction':>20}")
    a2 = A["A2_false_on"] = []
    for n in (18, 29, 36, 50, 100, 300):
        on, q05, fin = [], [], []
        for r in range(R["false_on"]):
            rng = np.random.default_rng([102, n, r])
            tail, rest = synthetic_groups(rng, 0.0, n, N_REST, SCALE)
            c = cert(tail, rest, (*CERT_TAG_HEAD, r, 2))
            on.append(c["on"]); q05.append(c["q05"]); fin.append(math.isfinite(c["height"]))
        rec = dict(n_tail=n, scale=SCALE, on_rate=float(np.mean(on)), q05_q95=float(np.quantile(q05, 0.95)),
                   q05_max=float(np.max(q05)), finite_share=float(np.mean(fin)), finite_predicted=p_finite(n), n=len(on))
        a2.append(rec)
        say(f"{n:6d} {rec['on_rate']:8.3f} {rec['q05_q95']:13.3f} {rec['finite_share']:14.3f} {rec['finite_predicted']:20.3f}")

    # A3 the refusal boundary with a real gap: the switch is on, the height may be infinite ----------------------
    say(f"\nA3  a clear gap (2.0) and a small full-throttle group: is the switch on, and is the height finite? "
        f"{R['refusal']} data sets per size; prediction = P(Binomial(1000, 1 - 0.9^n) >= 950)")
    say(f"{'n_tail':>6} {'on rate':>8} {'finite height':>14} {'predicted':>10} {'on & height inf':>16} {'finite share of tail draws':>27}")
    a3 = A["A3_refusal"] = []
    for n in (15, 18, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 40, 45, 60):
        on, fin, fs = [], [], []
        for r in range(R["refusal"]):
            rng = np.random.default_rng([103, n, r])
            tail, rest = synthetic_groups(rng, 2.0, n)
            c = cert(tail, rest, (*CERT_TAG_HEAD, r, 1))
            on.append(c["on"]); fin.append(math.isfinite(c["height"])); fs.append(c["finite_share_tail"])
        on, fin = np.array(on), np.array(fin)
        rec = dict(n_tail=n, on_rate=float(on.mean()), finite_share=float(fin.mean()), finite_predicted=p_finite(n),
                   on_with_infinite_height=float(np.mean(on & ~fin)), mean_finite_share_tail=float(np.mean(fs)),
                   expected_finite_share_tail=1.0 - 0.9 ** n, n=int(len(on)))
        a3.append(rec)
        say(f"{n:6d} {rec['on_rate']:8.3f} {rec['finite_share']:14.3f} {rec['finite_predicted']:10.3f} "
            f"{rec['on_with_infinite_height']:16.3f} {rec['mean_finite_share_tail']:12.3f} (1-0.9^n = {rec['expected_finite_share_tail']:.3f})")

    # A4 the shuffle control -------------------------------------------------------------------------------------
    say(f"\nA4  shuffle control: one data set with gap 2.0 (50 tail rows); the pooled scores are dealt back at random "
        f"{R['shuffle']} times (step 1's streams) and the certificate recomputed with the same posterior streams")
    rng = np.random.default_rng([104])
    tail, rest = synthetic_groups(rng, 2.0, 50)
    real = cert(tail, rest, (*CERT_TAG_HEAD, 0, 1))
    sh = S.shuffled_certificates(tail, rest, [*CERT_TAG_HEAD, 0, 1], [*PERM_TAG_HEAD, 0, 1], R["shuffle"], CFG)
    rec = A["A4_shuffle"] = dict(real_on=real["on"], real_q05=real["q05"], real_height=real["height"],
                                shuffled_on_rate=float(np.mean(sh["on"])), shuffled_q05_median=float(np.median(sh["q05"])),
                                shuffled_q05_max=float(np.max(sh["q05"])), n_perm=R["shuffle"])
    say(f"real: on {real['on']}, q05 {real['q05']:.3f}, height {real['height']:.3f};  shuffled: on rate "
        f"{rec['shuffled_on_rate']:.3f}, q05 median {rec['shuffled_q05_median']:.3f}, q05 max {rec['shuffled_q05_max']:.3f}")

    # A5 the random stream: same tags -> same numbers; other tags -> how much do q05 and the height move? ---------
    say(f"\nA5  the posterior's random stream: the same tags give the same certificate; {R['streams']} other streams on "
        f"one fixed data set (gap 1.0) show how much q05 and the height move with the stream")
    a5 = A["A5_streams"] = {}
    for n in (30, 50, 300):
        rng = np.random.default_rng([105, n])
        tail, rest = synthetic_groups(rng, 1.0, n)
        c1, c2 = cert(tail, rest, (*CERT_TAG_HEAD, 7, 1)), cert(tail, rest, (*CERT_TAG_HEAD, 7, 1))
        same = (c1["q05"] == c2["q05"]) and (c1["height"] == c2["height"]) and (c1["R_tail"] == c2["R_tail"])
        q, h = [], []
        for k in range(R["streams"]):
            c = cert(tail, rest, (*CERT_TAG_HEAD, 1000 + k, 1))
            q.append(c["q05"]); h.append(c["height"])
        q, h = np.array(q), np.array(h)
        hf = h[np.isfinite(h)]
        rec = dict(n_tail=n, same_tags_same_numbers=bool(same), q05_min=float(q.min()), q05_max=float(q.max()),
                   height_min=float(hf.min()) if len(hf) else None, height_max=float(hf.max()) if len(hf) else None,
                   height_finite_share=float(np.isfinite(h).mean()),
                   height_relative_spread=float((hf.max() - hf.min()) / np.median(hf)) if len(hf) else None)
        a5[str(n)] = rec
        say(f"n_tail {n:4d}: same tags -> identical: {same}; q05 over streams {rec['q05_min']:.3f}..{rec['q05_max']:.3f}; "
            f"height {rec['height_min']}..{rec['height_max']} (spread / median = {rec['height_relative_spread']})")

    # A6 the gap rule on synthetic throttles ---------------------------------------------------------------------
    say("\nA6  the gap rule on synthetic throttle indices (reads commands only)")
    r6 = np.random.default_rng([106])
    bulk = r6.uniform(0.0, 0.78, 40000)
    t = np.concatenate([bulk, np.full(50, 1.0)])
    g = S.gap_rule(t, **GAP_RULE)
    t8 = np.concatenate([bulk, np.full(8, 1.0)])
    g8 = S.gap_rule(t8, **GAP_RULE)
    tn = np.concatenate([bulk, np.full(50, 0.81)])
    gn = S.gap_rule(tn, **GAP_RULE)
    rec = A["A6_gap_rule"] = dict(
        defined=bool(g["defined"]), t_lo=g["t_lo"], t_lo_is_max_bulk=bool(g["t_lo"] == float(bulk.max())),
        t_up=g["t_up"], n_tail=int(g["n_tail"]), eight_rows_defined=bool(g8["defined"]), narrow_gap_defined=bool(gn["defined"]),
        narrow_gap=gn["gap"])
    say(f"40,000 throttles in [0, 0.78] + 50 at 1.0: defined {rec['defined']}, t_lo = {rec['t_lo']:.5f} (= largest bulk "
        f"throttle: {rec['t_lo_is_max_bulk']}), t_up = {rec['t_up']}, n_tail = {rec['n_tail']}; with only 8 rows at 1.0: "
        f"defined {rec['eight_rows_defined']}; with 50 rows at 0.81 (gap {rec['narrow_gap']:.3f} < 0.05): defined {rec['narrow_gap_defined']}")
    say(f"(part A took {time.time() - T0:.0f} s)")


# ================================================================================================ PART B: recorded
def load_recorded():
    """The 40 recorded certificates: 30 near-80 switch.json (seeds 0-4 in the near80 and random_rows settings, seeds
    10-14 in the main setting) and step 1's 10 (main setting, seeds 0-4)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(NEAR80, "learners", "*", "*", "seed*", "switch.json"))):
        d = json.load(open(path, encoding="utf8"))
        c = d["certificate"]
        rows.append(dict(source="near-80", setting=d["setting"], world=d["world"], seed=int(d["seed"]), defined=d["defined"],
                         on=bool(c["on"]), q05=num(c["q05"]), height=num(c["height"]), n_tail=c["n_tail"], n_rest=c["n_rest"],
                         R_tail=num(c["R_tail"]), R_rest=num(c["R_rest"]), finite_share_tail=c["finite_share_tail"],
                         lambda_hat_tail=num(c["lambda_hat_tail"]), lambda_hat_rest=num(c["lambda_hat_rest"]),
                         plain_gap=num(c["plain_gap"]), q50=num(c["q50"]),
                         mean_tail=num(c["mean_tail"]), mean_rest=num(c["mean_rest"]), n_survivors=d["n_survivors"],
                         t_lo=d["t_lo"], t_up=d["t_up"], rank_auc=d["rank_test"]["auc"], rank_p=num(d["rank_test"]["p"]),
                         file=os.path.relpath(path, ROOT)))
    s1 = json.load(open(STEP1_RESULTS, encoding="utf8"))
    for key in sorted(s1["per_seed_world"]):
        world, seed = key.split("/seed")
        e = s1["per_seed_world"][key]
        c = e["certificate"]
        rows.append(dict(source="step 1", setting="main", world=world, seed=int(seed), defined=True,
                         on=bool(c["on"]), q05=num(c["q05"]), height=num(c["height"]), n_tail=c["n_tail"], n_rest=c["n_rest"],
                         R_tail=num(c["R_tail"]), R_rest=num(c["R_rest"]), finite_share_tail=c["finite_share_tail"],
                         lambda_hat_tail=num(c["lambda_hat_tail"]), lambda_hat_rest=num(c["lambda_hat_rest"]),
                         plain_gap=num(c["plain_gap"]), q50=num(c["q50"]),
                         mean_tail=num(c["mean_tail"]), mean_rest=num(c["mean_rest"]), n_survivors=e["n_survivors"],
                         t_lo=e["t_lo"], t_up=e["t_up"], rank_auc=e["rank_test"]["auc"], rank_p=num(e["rank_test"]["p"]),
                         shuffle_on_rate=e["shuffle"]["on_rate"], shuffle_q05_median=e["shuffle"]["q05_median"],
                         file=os.path.relpath(STEP1_RESULTS, ROOT)))
    return rows, s1


def part_b():
    say("\n" + "=" * 100 + "\nPART B  the recorded certificates and their recomputation\n" + "=" * 100)
    B = OUT["B"] = {}
    rows, s1 = load_recorded()
    B["recorded"] = rows

    # B1 the 40 recorded certificates -----------------------------------------------------------------------------
    say(f"\nB1  the {len(rows)} recorded certificates")
    say(f"{'source':8} {'setting':12} {'world':9} {'seed':>4} {'on':>5} {'q05':>8} {'plain gap':>9} {'height':>8} {'n_tail':>6} {'R_tail':>8} "
        f"{'finite share':>12} {'1-0.9^n':>8} {'mean tail':>9} {'mean rest':>9} {'AUC':>6}  M5 expectation")
    for r in rows:
        n = r["n_tail"]
        exp = "never finite (n <= 24)" if n <= 24 else (f"by chance (25-35; P = {p_finite(n):.2f})" if n <= 35 else "finite (n >= 36)")
        flag = "  <-- ON with height +inf" if r["on"] and not math.isfinite(r["height"]) else ""
        say(f"{r['source']:8} {r['setting']:12} {r['world']:9} {r['seed']:4d} {str(r['on']):>5} {r['q05']:8.3f} {r['plain_gap']:9.3f} "
            f"{('inf' if not math.isfinite(r['height']) else f'{r['height']:.3f}'):>8} {n:6d} "
            f"{('inf' if not math.isfinite(r['R_tail']) else f'{r['R_tail']:.3f}'):>8} {r['finite_share_tail']:12.3f} "
            f"{1 - 0.9 ** n:8.3f} {r['mean_tail']:9.3f} {r['mean_rest']:9.3f} {r['rank_auc']:6.3f}  {exp}{flag}")
    stall = [r for r in rows if r["world"] == "stall"]
    nost = [r for r in rows if r["world"] == "no_stall"]
    checks = B["B1_checks"] = dict(
        n=len(rows), on_in_every_stalling_world=all(r["on"] for r in stall), n_stall=len(stall),
        off_in_every_no_stall_world=not any(r["on"] for r in nost), n_no_stall=len(nost),
        q05_stall_min=min(r["q05"] for r in stall), q05_stall_max=max(r["q05"] for r in stall),
        q05_no_stall_min=min(r["q05"] for r in nost), q05_no_stall_max=max(r["q05"] for r in nost),
        finite_heights_stall=sorted(r["height"] for r in stall if math.isfinite(r["height"])),
        finite_heights_no_stall=sorted(r["height"] for r in nost if math.isfinite(r["height"])),
        R_tail_infinite_iff_rule=dict(
            never_finite_at_most_24=all(not math.isfinite(r["R_tail"]) for r in rows if r["n_tail"] <= 24),
            always_finite_at_least_36=all(math.isfinite(r["R_tail"]) for r in rows if r["n_tail"] >= 36),
            zone_25_to_35=[f"{r['source']} {r['setting']} {r['world']} seed {r['seed']}: n_tail {r['n_tail']} (P finite {p_finite(r['n_tail']):.2f}): "
                           + ("finite" if math.isfinite(r["R_tail"]) else "inf")
                           for r in sorted(rows, key=lambda r: r["n_tail"]) if 25 <= r["n_tail"] <= 35],
            n_at_most_24=sum(r["n_tail"] <= 24 for r in rows), n_at_least_36=sum(r["n_tail"] >= 36 for r in rows),
            p_finite_by_n={str(n): round(p_finite(n), 3) for n in range(20, 37)}),
        finite_share_tail_vs_prediction_max_abs=max(abs(r["finite_share_tail"] - (1 - 0.9 ** r["n_tail"])) for r in rows),
        on_with_infinite_height=[f"{r['source']} {r['setting']} {r['world']} seed {r['seed']} (n_tail {r['n_tail']})"
                                 for r in rows if r["on"] and not math.isfinite(r["height"])],
        infinite_height_all=[f"{r['setting']} {r['world']} seed {r['seed']} (n_tail {r['n_tail']})"
                             for r in rows if not math.isfinite(r["height"])],
        mean_tail_above_mean_rest_stall=sum(r["mean_tail"] > r["mean_rest"] for r in stall),
        mean_tail_above_mean_rest_no_stall=sum(r["mean_tail"] > r["mean_rest"] for r in nost),
        # the lower bound sits below the point estimate of the gap, which sits below the height (where finite)
        q05_at_most_plain_gap=sum(r["q05"] <= r["plain_gap"] for r in rows),
        plain_gap_at_most_height=sum(r["plain_gap"] <= r["height"] for r in rows if math.isfinite(r["height"])),
        n_finite_height=sum(math.isfinite(r["height"]) for r in rows),
        plain_gap_stall=[round(r["plain_gap"], 3) for r in stall], plain_gap_no_stall=[round(r["plain_gap"], 3) for r in nost],
        height_minus_plain_gap_no_stall=sorted(round(r["height"] - r["plain_gap"], 3) for r in nost if math.isfinite(r["height"])),
        height_minus_plain_gap_stall=sorted(round(r["height"] - r["plain_gap"], 3) for r in stall if math.isfinite(r["height"])),
        step1_go_no_go=s1["go_no_go"], step1_net=s1["net"], step1_shuffle=s1["shuffle"],
        step1_rank_test_switched_net=s1["information_only"]["rank_test_switched"]["net"],
        step1_rank_test_false_on=[k for k, v in s1["information_only"]["rank_test_switched"]["on"]["no_stall"].items() if v])
    say(f"\non in every stalling world: {checks['on_in_every_stalling_world']} ({len(stall)} of {len(stall)}); off in every "
        f"no-stall world: {checks['off_in_every_no_stall_world']} ({len(nost)} of {len(nost)})")
    say(f"q05: stalling {checks['q05_stall_min']:.3f}..{checks['q05_stall_max']:.3f}; no-stall {checks['q05_no_stall_min']:.3f}.."
        f"{checks['q05_no_stall_max']:.3f}; the margin is 0.10")
    say(f"finite heights: stalling {min(checks['finite_heights_stall']):.2f}..{max(checks['finite_heights_stall']):.2f} "
        f"({len(checks['finite_heights_stall'])} of {len(stall)}); no-stall {min(checks['finite_heights_no_stall']):.2f}.."
        f"{max(checks['finite_heights_no_stall']):.2f} ({len(checks['finite_heights_no_stall'])} of {len(nost)})")
    say(f"R_tail infinite for every n_tail <= 24: {checks['R_tail_infinite_iff_rule']['never_finite_at_most_24']} "
        f"({checks['R_tail_infinite_iff_rule']['n_at_most_24']} cases); finite for every n_tail >= 36: "
        f"{checks['R_tail_infinite_iff_rule']['always_finite_at_least_36']} ({checks['R_tail_infinite_iff_rule']['n_at_least_36']} cases)")
    say("the zone 25-35, where M5's rule makes certification a matter of chance (same seed, world and n_tail = same draws):")
    for line in checks["R_tail_infinite_iff_rule"]["zone_25_to_35"]:
        say("   " + line)
    say(f"P(R finite) by group size (Binomial(1000, 1 - 0.9^n) >= 950): {checks['R_tail_infinite_iff_rule']['p_finite_by_n']}")
    say(f"recorded finite share of the tail's draws against 1 - 0.9^n_tail: largest difference "
        f"{checks['finite_share_tail_vs_prediction_max_abs']:.3f}")
    say(f"ordering: q05 <= plain gap (the gap of the two plain 90% quantiles) in {checks['q05_at_most_plain_gap']} of {len(rows)}; "
        f"plain gap <= height in {checks['plain_gap_at_most_height']} of {checks['n_finite_height']} finite heights")
    say(f"plain gap: stalling {min(checks['plain_gap_stall']):.3f}..{max(checks['plain_gap_stall']):.3f}; no-stall "
        f"{min(checks['plain_gap_no_stall']):.3f}..{max(checks['plain_gap_no_stall']):.3f}; height minus plain gap: stalling "
        f"{checks['height_minus_plain_gap_stall'][0]:.3f}..{checks['height_minus_plain_gap_stall'][-1]:.3f}, no-stall "
        f"{checks['height_minus_plain_gap_no_stall'][0]:.3f}..{checks['height_minus_plain_gap_no_stall'][-1]:.3f} (the small-sample margin of the tail's threshold)")
    say(f"switch ON with height +inf (the exceptional case): {checks['on_with_infinite_height']}")
    say(f"all infinite heights: {checks['infinite_height_all']}")
    say(f"step 1 go/no-go: {checks['step1_go_no_go']['verdict']} (C1 {checks['step1_go_no_go']['C1']}, C2 {checks['step1_go_no_go']['C2_net_at_least_90']}, "
        f"C3 {checks['step1_go_no_go']['C3_shuffled_at_most_half']}); net {checks['step1_net']['net']:.1f}; shuffled net mean "
        f"{checks['step1_shuffle']['mean']:.3f}, max {checks['step1_shuffle']['max']:.1f}; rank test as the switch: net "
        f"{checks['step1_rank_test_switched_net']['net']:.1f}, false on in no-stall seeds {checks['step1_rank_test_false_on']}")

    # B2 the pinned-environment recomputation -----------------------------------------------------------------
    say("\nB2  recomputation from the recorded inputs (saved second critic applied to the recorded rows; pinned environment)")
    rc_path = os.path.join(DATA, "recomputed_certificates.json")
    npz_path = os.path.join(DATA, "overestimates.npz")
    if not (os.path.exists(rc_path) and os.path.exists(npz_path)):
        say(f"  not available: {rc_path} / {npz_path} missing (run M10_recompute_pinned.py through the bridge first)")
        B["B2"] = dict(available=False)
        return rows
    rc = json.load(open(rc_path, encoding="utf8"))["certificates"]
    ov = np.load(npz_path)
    B["B2_pinned"] = []
    say(f"{'setting':12} {'world':9} {'seed':>4} {'n_tail':>6} {'q05 recorded':>13} {'q05 recomputed':>15} {'height recorded':>15} "
        f"{'height recomputed':>17} {'pinned equal':>12} {'here equal':>10} {'max |diff| here':>15}")
    n_equal_pinned = n_equal_here = 0
    max_here = 0.0
    for e in rc:
        if "error" in e:
            say(f"{e['setting']:12} {e['world']:9} {e['seed']:4d}  ERROR {e['error']}")
            B["B2_pinned"].append(dict(setting=e["setting"], world=e["world"], seed=e["seed"], error=e["error"]))
            continue
        key = f"{e['setting']}_{e['world']}_seed{e['seed']}"
        tail, rest = ov[key + "_tail"], ov[key + "_rest"]
        here = cert(tail, rest, (*CERT_TAG_HEAD, e["seed"], WORLD_TAG[e["world"]]))
        rec = e["recorded"]
        # which score each threshold IS (rank 1 = the group's largest score): the credible threshold of a small group
        # is one of its top scores, the rest's is near its 90% quantile
        ts, rs = np.sort(tail)[::-1], np.sort(rest)[::-1]
        rank_R_tail = int(np.argmin(np.abs(ts - here["R_tail"]))) + 1 if math.isfinite(here["R_tail"]) else None
        rank_R_rest = int(np.argmin(np.abs(rs - here["R_rest"]))) + 1
        rank_lhat_tail = int(np.argmin(np.abs(ts - here["lambda_hat_tail"]))) + 1
        keys = ("q05", "q50", "height", "R_tail", "R_rest", "lambda_hat_tail", "lambda_hat_rest", "finite_share_tail",
                "finite_share_rest", "mean_tail", "mean_rest", "plain_gap", "P_gap_le_margin", "e11_form_q05")
        dmax = 0.0
        eq = True
        for k in keys:
            a, b = here[k], num(rec[k])
            if math.isinf(a) or math.isinf(b):
                eq = eq and (a == b)
            else:
                dmax = max(dmax, abs(a - b))
                eq = eq and (abs(a - b) <= 1e-9 * max(1.0, abs(b)))
        eq = eq and here["on"] == rec["on"] and here["n_tail"] == rec["n_tail"] and here["n_rest"] == rec["n_rest"]
        n_equal_pinned += bool(e["all_equal"]); n_equal_here += bool(eq); max_here = max(max_here, dmax)
        B["B2_pinned"].append(dict(setting=e["setting"], world=e["world"], seed=e["seed"], n_tail=e["n_tail"], n_rest=e["n_rest"],
                                   sd_tail=float(np.std(tail)), sd_rest=float(np.std(rest)), max_tail=float(np.max(tail)),
                                   R_tail_rank_from_top=rank_R_tail, R_rest_rank_from_top=rank_R_rest, lambda_hat_tail_rank_from_top=rank_lhat_tail,
                                   q90_tail=float(np.quantile(tail, 0.9, method="inverted_cdf")), q90_rest=float(np.quantile(rest, 0.9, method="inverted_cdf")),
                                   recorded_q05=num(rec["q05"]), pinned_q05=num(e["recomputed"]["q05"]), here_q05=here["q05"],
                                   recorded_height=num(rec["height"]), pinned_height=num(e["recomputed"]["height"]), here_height=here["height"],
                                   pinned_all_equal=bool(e["all_equal"]), here_equal=bool(eq), here_max_abs_diff=dmax,
                                   survivors_sha256_equal=e["survivors_sha256_equal"], gap_rule=e["gap_rule"], pinned_seconds=e["seconds"]))
        say(f"{e['setting']:12} {e['world']:9} {e['seed']:4d} {e['n_tail']:6d} {num(rec['q05']):13.6f} {num(e['recomputed']['q05']):15.6f} "
            f"{num(rec['height']):15.6f} {num(e['recomputed']['height']):17.6f} {str(bool(e['all_equal'])):>12} {str(bool(eq)):>10} {dmax:15.2e}")
    B["B2_summary"] = dict(n=len(rc), pinned_equal=n_equal_pinned, here_equal=n_equal_here, here_max_abs_diff=max_here,
                           gap_rule_t_lo_range=[min(e["gap_rule"]["t_lo"] for e in rc if "gap_rule" in e), max(e["gap_rule"]["t_lo"] for e in rc if "gap_rule" in e)],
                           gap_rule_t_up_range=[min(e["gap_rule"]["t_up"] for e in rc if "gap_rule" in e), max(e["gap_rule"]["t_up"] for e in rc if "gap_rule" in e)],
                           bank_rows_past_t_lo=[e["gap_rule"]["n_bank_past_t_lo"] for e in rc if "gap_rule" in e],
                           n_above_99th=[e["gap_rule"]["n_above_quantile"] for e in rc if "gap_rule" in e],
                           n_training=[e["gap_rule"]["n_training"] for e in rc if "gap_rule" in e],
                           t_lo_equal_recorded=all(e["gap_rule"]["t_lo"] == e["gap_rule"]["recorded_t_lo"] and e["gap_rule"]["t_up"] == e["gap_rule"]["recorded_t_up"] for e in rc if "gap_rule" in e),
                           survivors_equal=all(e.get("survivors_sha256_equal") for e in rc if "gap_rule" in e))
    s = B["B2_summary"]
    bp = B["B2_pinned"]
    s["sd_rest_range"] = [min(x["sd_rest"] for x in bp), max(x["sd_rest"] for x in bp)]
    s["sd_tail_range_stall"] = [min(x["sd_tail"] for x in bp if x["world"] == "stall"), max(x["sd_tail"] for x in bp if x["world"] == "stall")]
    s["sd_tail_range_no_stall"] = [min(x["sd_tail"] for x in bp if x["world"] == "no_stall"), max(x["sd_tail"] for x in bp if x["world"] == "no_stall")]
    s["max_tail_range_stall"] = [min(x["max_tail"] for x in bp if x["world"] == "stall"), max(x["max_tail"] for x in bp if x["world"] == "stall")]
    finr = [x for x in bp if x["R_tail_rank_from_top"] is not None]
    s["n_finite_R_tail"] = len(finr)
    s["R_tail_rank_counts"] = {str(k): int(sum(x["R_tail_rank_from_top"] == k for x in finr)) for k in (1, 2, 3)}
    s["lambda_hat_tail_rank_range"] = [min(x["lambda_hat_tail_rank_from_top"] for x in bp), max(x["lambda_hat_tail_rank_from_top"] for x in bp)]
    s["R_rest_rank_range"] = [min(x["R_rest_rank_from_top"] for x in bp), max(x["R_rest_rank_from_top"] for x in bp)]
    s["n_rest_range"] = [min(x["n_rest"] for x in bp), max(x["n_rest"] for x in bp)]
    say(f"\nwhich score the thresholds are: where finite, R_tail is the full-throttle group's largest score in "
        f"{s['R_tail_rank_counts']['1']} of {s['n_finite_R_tail']} seed-worlds and its second largest in {s['R_tail_rank_counts']['2']} "
        f"(the plain 90% quantile lambda_hat_tail has rank {s['lambda_hat_tail_rank_range'][0]} to {s['lambda_hat_tail_rank_range'][1]} from the top); "
        f"R_rest has rank {s['R_rest_rank_range'][0]} to {s['R_rest_rank_range'][1]} from the top among {s['n_rest_range'][0]}-{s['n_rest_range'][1]} rest scores")
    say(f"spread of the scores: rest {s['sd_rest_range'][0]:.3f}..{s['sd_rest_range'][1]:.3f} (standard deviation, value units); "
        f"tail in the stalling world {s['sd_tail_range_stall'][0]:.2f}..{s['sd_tail_range_stall'][1]:.2f} (largest score "
        f"{s['max_tail_range_stall'][0]:.1f}..{s['max_tail_range_stall'][1]:.1f}); tail in the no-stall world "
        f"{s['sd_tail_range_no_stall'][0]:.3f}..{s['sd_tail_range_no_stall'][1]:.3f}")
    say(f"pinned environment: {s['pinned_equal']} of {s['n']} certificates equal to the record in every field (1e-9 relative); "
        f"this machine (numpy {np.__version__}) from the saved overestimates with the recorded streams: {s['here_equal']} of {s['n']} "
        f"equal, largest absolute difference {s['here_max_abs_diff']:.2e}")
    say(f"gap rule recomputed: t_lo {s['gap_rule_t_lo_range'][0]:.4f}..{s['gap_rule_t_lo_range'][1]:.4f}, t_up "
        f"{s['gap_rule_t_up_range'][0]:.4f}..{s['gap_rule_t_up_range'][1]:.4f}, equal to the records: {s['t_lo_equal_recorded']}; "
        f"survivors equal (SHA-256): {s['survivors_equal']}; held-out rows past t_lo per seed-world: {s['bank_rows_past_t_lo']}; "
        f"rows above the 99th percentile the rule sorts: {min(s['n_above_99th'])}..{max(s['n_above_99th'])} of "
        f"{min(s['n_training'])}..{max(s['n_training'])} kept training rows")

    # B3 the shuffle control on the recomputed scores -------------------------------------------------------------
    n_perm = R["shuffle"]
    say(f"\nB3  shuffle control on the 30 near-80 seed-worlds' recomputed scores: {n_perm} permutations each (step 1's streams)")
    say(f"{'setting':12} {'world':9} {'seed':>4} {'real on':>7} {'real q05':>9} {'shuffled on rate':>16} {'shuffled q05 median':>19} {'shuffled q05 max':>16}")
    B["B3_shuffle"] = []
    for e in rc:
        if "error" in e:
            continue
        key = f"{e['setting']}_{e['world']}_seed{e['seed']}"
        tail, rest = ov[key + "_tail"], ov[key + "_rest"]
        w = WORLD_TAG[e["world"]]
        sh = S.shuffled_certificates(tail, rest, [*CERT_TAG_HEAD, e["seed"], w], [*PERM_TAG_HEAD, e["seed"], w], n_perm, CFG)
        rec = dict(setting=e["setting"], world=e["world"], seed=e["seed"], real_on=e["recorded"]["on"], real_q05=num(e["recorded"]["q05"]),
                   shuffled_on_rate=float(np.mean(sh["on"])), shuffled_q05_median=float(np.median(sh["q05"])),
                   shuffled_q05_max=float(np.max(sh["q05"])), n_perm=n_perm)
        B["B3_shuffle"].append(rec)
        say(f"{e['setting']:12} {e['world']:9} {e['seed']:4d} {str(rec['real_on']):>7} {rec['real_q05']:9.3f} {rec['shuffled_on_rate']:16.3f} "
            f"{rec['shuffled_q05_median']:19.3f} {rec['shuffled_q05_max']:16.3f}")
    ons = [r["shuffled_on_rate"] for r in B["B3_shuffle"]]
    B["B3_summary"] = dict(n_seed_worlds=len(ons), n_perm=n_perm, max_on_rate=max(ons), mean_on_rate=float(np.mean(ons)),
                           total_on=int(round(sum(ons) * n_perm)), total=len(ons) * n_perm)
    say(f"shuffled: {B['B3_summary']['total_on']} of {B['B3_summary']['total']} permutations switched on (largest per-seed-world "
        f"rate {B['B3_summary']['max_on_rate']:.3f}); step 1 recorded 0.0 in 9 of 10 seed-worlds and 0.005 in stall/seed 4")

    # B4 other random streams on the real scores: does the decision depend on the stream? -------------------------
    n_str = R["streams"]
    say(f"\nB4  the same 30 score sets under {n_str} other random streams for the two posteriors (tags [20261015, 9700, "
        "1000 + k, world tag]): the recorded decision should not depend on the stream")
    say(f"{'setting':12} {'world':9} {'seed':>4} {'n_tail':>6} {'recorded q05':>12} {'q05 min':>8} {'q05 max':>8} {'on in':>6} "
        f"{'height min':>10} {'height max':>10} {'finite':>6}")
    B["B4_streams"] = []
    for e in rc:
        if "error" in e:
            continue
        key = f"{e['setting']}_{e['world']}_seed{e['seed']}"
        tail, rest = ov[key + "_tail"], ov[key + "_rest"]
        w = WORLD_TAG[e["world"]]
        q, h = [], []
        for k in range(n_str):
            c = cert(tail, rest, (*CERT_TAG_HEAD, 1000 + k, w))
            q.append(c["q05"]); h.append(c["height"])
        q, h = np.array(q), np.array(h)
        hf = h[np.isfinite(h)]
        rec = dict(setting=e["setting"], world=e["world"], seed=e["seed"], n_tail=e["n_tail"], recorded_on=e["recorded"]["on"],
                   recorded_q05=num(e["recorded"]["q05"]), q05_min=float(q.min()), q05_max=float(q.max()),
                   on_count=int(np.sum(q > CFG["margin"])), n_streams=n_str,
                   decision_stable=bool(all((q > CFG["margin"]) == e["recorded"]["on"])),
                   height_min=float(hf.min()) if len(hf) else None, height_max=float(hf.max()) if len(hf) else None,
                   height_finite_count=int(len(hf)))
        B["B4_streams"].append(rec)
        say(f"{e['setting']:12} {e['world']:9} {e['seed']:4d} {e['n_tail']:6d} {rec['recorded_q05']:12.3f} {rec['q05_min']:8.3f} {rec['q05_max']:8.3f} "
            f"{rec['on_count']:3d}/{n_str:<2d} {(f'{rec['height_min']:.3f}' if rec['height_min'] is not None else 'inf'):>10} "
            f"{(f'{rec['height_max']:.3f}' if rec['height_max'] is not None else 'inf'):>10} {rec['height_finite_count']:3d}/{n_str:<2d}")
    stable = sum(r["decision_stable"] for r in B["B4_streams"])
    spread = max(r["q05_max"] - r["q05_min"] for r in B["B4_streams"])
    closest = min(abs(r["q05_min"] - CFG["margin"]) if r["recorded_on"] else abs(CFG["margin"] - r["q05_max"]) for r in B["B4_streams"])
    B["B4_summary"] = dict(n_seed_worlds=len(B["B4_streams"]), n_streams=n_str, decision_stable_in=stable, max_q05_spread=spread,
                           closest_approach_to_margin=closest,
                           height_finite_count_by_n_tail={str(r["n_tail"]): r["height_finite_count"] for r in sorted(B["B4_streams"], key=lambda r: r["n_tail"])})
    say(f"the recorded decision holds under every stream in {stable} of {len(B['B4_streams'])} seed-worlds; largest spread of q05 across "
        f"streams {spread:.3f}; closest any stream came to the margin on the wrong side: {closest:.3f} value units")
    say(f"(part B took {time.time() - T0:.0f} s so far)")
    return rows


# ================================================================================================ PART C: figure
def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"  # keep the figure's text as text, not outlines
    from matplotlib.lines import Line2D

    BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    col = {"stall": BLUE, "no_stall": ORANGE}
    A = OUT["A"]
    reps = OUT["settings"]["reps"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#b5b4ae"); ax.spines["bottom"].set_color("#b5b4ae")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)

    # panel 1: the switch statistic q05 against the group size, all 40 recorded certificates
    ax = axes[0, 0]
    for w in ("stall", "no_stall"):
        rr = [r for r in rows if r["world"] == w]
        ax.scatter([r["n_tail"] for r in rr], [r["q05"] for r in rr], s=42, color=col[w], edgecolor="#fcfcfb", lw=1.2, zorder=3,
                   marker="o" if w == "stall" else "s")
    ax.axhline(0.10, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(61, 0.13, "switch-on margin 0.10", fontsize=9, color=INK, ha="right")
    ax.text(61, 0.95, "stalling world: on in 20 of 20", fontsize=9.5, color=BLUE, ha="right", fontweight="bold")
    ax.text(61, -0.22, "no-stall world: off in 20 of 20", fontsize=9.5, color=ORANGE, ha="right", fontweight="bold")
    ax.set_xlabel("rows in the full-throttle group (n_tail)", fontsize=10)
    ax.set_ylabel("q05: 5% lower bound of the gap between\nthe two 90% quantiles (value units)", fontsize=10)
    ax.set_title("A. The switch reads the gap: 40 recorded certificates", fontsize=11, loc="left", color=INK)
    ax.set_xlim(14, 62); ax.set_ylim(-0.3, 1.4)

    # panel 2: the height against the group size; infinite heights at the top
    ax = axes[0, 1]
    top = 7.2
    for w in ("stall", "no_stall"):
        rr = [r for r in rows if r["world"] == w]
        fin = [r for r in rr if math.isfinite(r["height"])]
        inf = [r for r in rr if not math.isfinite(r["height"])]
        ax.scatter([r["n_tail"] for r in fin], [r["height"] for r in fin], s=42, color=col[w], edgecolor="#fcfcfb", lw=1.2, zorder=3,
                   marker="o" if w == "stall" else "s")
        ax.scatter([r["n_tail"] for r in inf], [top] * len(inf), s=60, facecolor="none", edgecolor=col[w], lw=1.8, zorder=3,
                   marker="o" if w == "stall" else "s")
    ax.axvspan(14, 24.5, color="#efeeea", zorder=1)
    ax.axvspan(24.5, 35.5, color="#f6f5f2", zorder=1)
    ax.text(19.2, 4.6, "n <= 24:\nnever certified\n(height +inf)", fontsize=8.5, color=MUTED, ha="center")
    ax.text(30, 3.0, "25-35: certified\nby chance\n(0.4 at 28, 0.7 at 29,\n0.9 at 30 rows)", fontsize=8, color=MUTED, ha="center")
    ax.text(48, 4.6, "n >= 36: always certified", fontsize=8.5, color=MUTED, ha="center")
    ax.axhline(top, color="#b5b4ae", lw=0.8, ls=":")
    ax.text(61, top + 0.15, "open markers: height = +inf (the group could not be certified)", fontsize=8.5, color=MUTED, ha="right")
    ax.annotate("switch ON, height +inf:\nfresh seeds 10, 12, 13 (the stops)", xy=(19.5, top), xytext=(24, 6.1), fontsize=8.5, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.set_xlabel("rows in the full-throttle group (n_tail)", fontsize=10)
    ax.set_ylabel("height = R_tail - R_rest (value units)", fontsize=10)
    ax.set_title("B. The height: large where full throttle stalls, near zero where it does not", fontsize=11, loc="left", color=INK)
    ax.set_xlim(14, 62); ax.set_ylim(-0.3, 7.8)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])

    # panel 3: synthetic refusal boundary
    ax = axes[1, 0]
    a3 = A["A3_refusal"]
    n = [r["n_tail"] for r in a3]
    ax.plot(n, [r["finite_predicted"] for r in a3], color=AQUA, lw=2, zorder=2)
    ax.plot(n, [r["finite_share"] for r in a3], color=BLUE, lw=0, marker="o", ms=7, zorder=3)
    ax.plot(n, [r["on_rate"] for r in a3], color=INK, lw=2, ls="-", zorder=2, marker="D", ms=5)
    ax.text(61, 0.50, "share of data sets with a finite height:\nblue dots observed, green line the\nbinomial prediction from module M5", fontsize=8.5, color=BLUE, ha="right")
    ax.text(61, 1.045, "share switched on (black): 1.0 at every size", fontsize=8.5, color=INK, ha="right")
    ax.axvline(29, color="#b5b4ae", lw=0.8, ls=":")
    ax.text(29.6, 0.25, "about 29 rows", fontsize=8.5, color=MUTED)
    ax.set_xlabel(f"rows in the full-throttle group (n_tail)\nsynthetic scores with a true gap of 2.0; {reps['refusal']} data sets per size", fontsize=10)
    ax.set_ylabel("share of synthetic data sets", fontsize=10)
    ax.set_title("C. Synthetic: the switch is on at every size, the height needs about 29 rows", fontsize=11, loc="left", color=INK)
    ax.set_xlim(14, 62); ax.set_ylim(-0.05, 1.12)

    # panel 4: synthetic known gap at the recorded spread
    ax = axes[1, 1]
    a1 = A["A1_known_gap"]
    g = np.array([r["gap"] for r in a1])
    gmax = float(g.max())
    ax.plot([0, gmax * 1.1], [0, gmax * 1.1], color="#b5b4ae", lw=1, ls="--", zorder=1)
    ax.plot(g, g + np.array([r["height_minus_gap_median"] for r in a1]), color=BLUE, lw=2, marker="o", ms=7, zorder=3)
    ax.fill_between(g, g + np.array([r["height_minus_gap_q05"] for r in a1]), g + np.array([r["height_minus_gap_q95"] for r in a1]),
                    color=BLUE, alpha=0.15, lw=0, zorder=2)
    ax.plot(g, [r["q05_median"] for r in a1], color=ORANGE, lw=2, marker="s", ms=6, zorder=3)
    ax.fill_between(g, [r["q05_q05"] for r in a1], [r["q05_q95"] for r in a1], color=ORANGE, alpha=0.15, lw=0, zorder=2)
    ax2 = ax.twinx()
    ax2.plot(g, [r["on_rate"] for r in a1], color=INK, lw=1.5, ls=":", marker="D", ms=4, zorder=3)
    ax2.set_ylim(-0.1, 1.25); ax2.set_ylabel("share of data sets switched on (dotted)", fontsize=10, color=MUTED)
    ax2.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color("#b5b4ae")
    ax.axhline(0.10, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(gmax * 1.07, 0.125, "margin 0.10", fontsize=8.5, color=INK, ha="right")
    ax.text(gmax * 0.80, gmax * 0.56, "true gap (diagonal)", fontsize=8.5, color=MUTED, ha="left")
    ax.text(0.0, gmax * 1.23, "height (blue; median and 5-95% band):\nthe true gap plus a small-sample margin", fontsize=8.5, color=BLUE, va="top")
    ax.text(gmax * 0.60, gmax * 0.20, "q05 (orange; median and 5-95% band):\na lower bound on the gap", fontsize=8.5, color=ORANGE)
    ax.text(gmax * 0.25, gmax * 0.78, "share switched on\n(black dotted, right axis)", fontsize=8.5, color=INK)
    ax.set_xlabel(f"true gap between the two groups' 90% quantiles (value units)\nsynthetic; spread {SCALE}, {N_TAIL} tail rows, "
                  f"{N_REST} rest rows, {reps['known_gap']} data sets per gap", fontsize=10)
    ax.set_ylabel("certificate output (value units)", fontsize=10)
    ax.set_title("D. Synthetic: q05 and the height follow a known gap", fontsize=11, loc="left", color=INK)
    ax.set_xlim(-0.03, gmax * 1.1); ax.set_ylim(-0.2, gmax * 1.25)

    handles = [Line2D([], [], marker="o", color=BLUE, lw=0, ms=7, label="stalling world (full throttle is harmful)"),
               Line2D([], [], marker="s", color=ORANGE, lw=0, ms=7, label="no-stall world (full throttle is harmless)")]
    axes[0, 0].legend(handles=handles, loc="center right", fontsize=9, frameon=False)
    fig.suptitle("M10, the harm switch certificate: recorded certificates (top) and synthetic checks (bottom)", fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, format="svg")
    say(f"figure written: {FIG}")
    if ARGS.png:
        fig.savefig(ARGS.png, format="png", dpi=110)
        say(f"PNG preview written: {ARGS.png}")


def main():
    say(f"M10 test; repository root {ROOT}; numpy {np.__version__}; python {sys.version.split()[0]}; fast={ARGS.fast}")
    if ARGS.figure_only:
        saved = json.load(open(os.path.join(DATA, "M10_test_results.json"), encoding="utf8"))
        OUT["A"], OUT["settings"] = saved["A"], saved["settings"]
        figure(load_recorded()[0])
        return
    check_prespec()
    part_a()
    rows = part_b()
    if not ARGS.no_figure:
        figure(rows)
    os.makedirs(DATA, exist_ok=True)
    out_path = os.path.join(DATA, "M10_test_results.json")
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(OUT, f, indent=1, default=lambda o: (None if isinstance(o, float) and not math.isfinite(o) else str(o)))
    say(f"results written: {out_path}; total {time.time() - T0:.0f} s")


if __name__ == "__main__":
    main()
