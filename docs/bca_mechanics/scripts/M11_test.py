"""Module M11 of MODULES.md: the ramp height. Mechanical tests on synthetic data and on the recorded ThrusterChain
inputs; prints the tables of docs/bca_mechanics/modules/M11.md and writes docs/bca_mechanics/figures/M11.svg.

Run from the repository root (branch claude/bca-mechanics or any checkout holding the near-80 results folder):
    set PYTHONPATH=<repository root> and PYTHONDONTWRITEBYTECODE=1
    python docs/bca_mechanics/scripts/M11_test.py [--no-figure]
Environment used: Python 3.14.2, NumPy 2.5.4, Matplotlib 3.11.2, jax 0.12.0 (the scratch venv on Windows). The ramp code
is imported from the repository's own outputs/shift/beyond-the-data/restructure/step2/step2lib.py (NumPy only), so the
pinned environment is not needed for anything printed here. The one part that needed the pinned environment (loading the
saved actors and the critic: Python 3.10.20, numpy 2.2.6, jax 0.6.2 on CPU, in WSL) ran once through the bridge with
    pwsh -NoProfile -File <scratchpad>/wslrun.ps1 -ScriptFile docs/bca_mechanics/scripts/M11_bridge_job.sh -TimeoutSec 1500 -Name m11ramp
and its outputs are the 24 JSON files in docs/bca_mechanics/data/M11/ (one per locked seed-world of the near-80 run),
which this script reads and summarizes (sections R2 second half and R3 second half). Nothing is trained.

Sections (each check prints PASS or FAIL):
  S1  ramp shape: 0 at or below t_lo, 1 at or above t_up, linear between (exact values at dyadic points)
  S2  the paired one-sided charge: only for going further up the ramp than the recorded command
  S3  height 0 gives penalty exactly 0 and gradient exactly 0; the penalty is exactly linear in h and in k
  S4  the arm-table rules on made-up step-1 records: on + finite -> h = height; off + infinite -> h = 0 (no refusal);
      on + infinite -> refused; negative -> refused; the always-on arm carries the stalling height
  S5  toys with the truth known: (a) ramp starting where the harm starts and a straight-line critic: every height
      at or above the critic's over-optimism along the ramp gives the same (best) choice, so an outcome can only show
      "large enough"; (b) the same with the ramp starting 0.10 below the harm: every height that removes the harm also
      gives up the useful stretch, no height recovers it; (c) a curved critic: the picked throttle falls smoothly as
      the height grows, so there IS a calibrated height, with too small leaving the harm and too large losing the
      useful stretch; the true best height equals the critic's slope at the harm times the ramp's length
  R1  every recorded arm table (24 locked seed-worlds of the near-80 run, 10 of step 2) is rebuilt from the switch
      records by step2lib.arm_table; the three fresh-seed stops are reproduced with the recorded message
  R2  the height-0 plumbing check: every recorded difference (34 seed-worlds, 104 arms) and the differences recomputed
      from the saved actors in the pinned environment (24 seed-worlds, 70 arms), all 0.0 against a tolerance of 1e-5
  R3  the penalty's size on recorded rows: the recorded ramp-shape readouts are proportional to h*k; the penalty the
      no-caution actor would pay under BCA's ramp against what the trained ramp actor pays
  R4  the outcome by height: strong-current points in the stalling world for five heights per seed, the net across
      worlds, the dose readout k = 2 against k = 1, and the fresh seeds
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STEP2 = ROOT / "outputs" / "shift" / "beyond-the-data" / "restructure" / "step2"
STEP1_RESULTS = ROOT / "outputs" / "shift" / "beyond-the-data" / "restructure" / "step1" / "results" / "results.json"
STEP2_RESULTS = ROOT / "outputs" / "shift" / "beyond-the-data" / "restructure" / "step2_results" / "results.json"
NEAR80 = ROOT / "outputs" / "shift" / "beyond-the-data" / "near80" / "results-20261020T115246Z" / "files"
BRIDGE_OUT = HERE.parent / "data" / "M11"
FIGURE = HERE.parent / "figures" / "M11.svg"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STEP2))
import step2lib as S  # noqa: E402  (NumPy only; the ramp code the runs used)

ARM_RULE = dict(fixed_height=1.0, conformal_margin=0.10, rank_p=0.05)   # near80_constants.ARM_RULE = step 2's
MEANDIFF_MARGIN = 0.10                                                   # near80_constants.MEANDIFF_MARGIN
KNEE32 = float(np.float32(0.8))                                          # near80_constants.KNEE32
W0_TOL = 1e-5                                                            # near80_constants.W0_TOL
HEIGHT_ARMS = ("meangap_k1", "fixed_gated", "conformal_k1", "obca_k1", "obca_k2")
LABEL = dict(meangap_k1="mean gap", fixed_gated="fixed 1.0", conformal_k1="conformal gap", obca_k1="BCA height",
             obca_k2="2 x BCA height", obca_forced="always on", none="no caution", knee_ramp="ramp from 80%")

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def jl(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float) and not math.isfinite(x):
        return "+inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return f"{x:.{nd}f}"


# ================================================================================================= synthetic
def synthetic():
    print("\n== S1  ramp shape (step2lib.ramp, throttle)")
    t_lo, t_up = 0.75, 1.0
    t = np.array([0.0, 0.5, 0.75, 0.875, 0.9375, 1.0, 1.25])
    want = np.array([0.0, 0.0, 0.0, 0.5, 0.75, 1.0, 1.0])
    check("ramp is 0 up to t_lo, linear to 1 at t_up, flat after", np.array_equal(S.ramp(t, t_lo, t_up), want))
    a = np.array([[0.1, -0.9, 0.2, 0.0], [-1.0, 0.5, 0.0, 0.3], [0.0, 0.0, 0.0, 0.5]])
    check("throttle index = largest |thruster command|", np.array_equal(S.throttle(a), [0.9, 1.0, 0.5]))
    grid = np.linspace(0, 1.2, 1201)
    hand = np.minimum(1.0, np.maximum(0.0, (grid - t_lo) / (t_up - t_lo)))
    check("ramp equals the hand formula on 1,201 grid points", np.allclose(S.ramp(grid, t_lo, t_up), hand, atol=0, rtol=0))

    print("\n== S2  the paired one-sided charge k [U(pi(s)) - U(a_rec)]_+")
    pi = np.array([[0.875, 0, 0, 0], [1.0, 0, 0, 0], [0.5, 0, 0, 0], [0.875, 0, 0, 0]])
    rec = np.array([[0.5, 0, 0, 0], [0.875, 0, 0, 0], [1.0, 0, 0, 0], [0.0, 0.0, -1.0, 0]])
    p = S.ramp_penalty(pi, rec, 2.0, 3.0, t_lo, t_up)        # h = 2, k = 3
    # 3[2*0.5 - 0]+ = 3 ; 3[2*1 - 2*0.5]+ = 3 ; 3[0 - 2]+ = 0 ; 3[1 - 2]+ = 0
    check("charged only for going further up the ramp than the recorded command", np.array_equal(p, [3.0, 3.0, 0.0, 0.0]))
    r = np.random.default_rng(11)
    pi2, rec2 = r.uniform(-1, 1, (2000, 4)), r.uniform(-1, 1, (2000, 4))
    p2 = S.ramp_penalty(pi2, rec2, 1.5, 1.0, t_lo, t_up)
    below = S.throttle(pi2) <= S.throttle(rec2)
    check("never charged when the actor's throttle is at or below the recorded one",
          float(np.abs(p2[below]).max()) == 0.0, f"{int(below.sum())} such rows of 2,000")
    check("penalty is never negative and never above k*h", p2.min() >= 0 and p2.max() <= 1.5 + 1e-12)

    print("\n== S3  height 0, and linearity in h and k")
    p0 = S.ramp_penalty(pi2, rec2, 0.0, 2.0, t_lo, t_up)
    check("h = 0: the penalty is exactly 0 on 2,000 random rows", float(np.abs(p0).max()) == 0.0)
    # gradient by finite differences at h = 0 (no jax needed): a bump of 1e-3 in any command changes nothing
    bumped = pi2.copy()
    bumped[:, 0] += 1e-3
    check("h = 0: a bump of the command changes the penalty by exactly 0 (zero gradient)",
          float(np.abs(S.ramp_penalty(bumped, rec2, 0.0, 2.0, t_lo, t_up) - p0).max()) == 0.0)
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        g0 = jax.grad(lambda x: S.ramp_penalty(x, jnp.asarray(rec2), 0.0, 2.0, t_lo, t_up, xp=jnp).sum())(jnp.asarray(pi2))
        g1 = jax.grad(lambda x: S.ramp_penalty(x, jnp.asarray(rec2), 1.0, 2.0, t_lo, t_up, xp=jnp).sum())(jnp.asarray(pi2))
        check("jax: the gradient is exactly 0 at h = 0 and non-zero at h = 1", float(jnp.abs(g0).max()) == 0.0
              and float(jnp.abs(g1).max()) > 0.0)
        t_pi = S.throttle(pi2)
        outside = (t_pi <= t_lo) | (t_pi >= t_up)
        check("jax: at h = 1 the gradient is 0 wherever the actor's throttle is outside the ramp",
              float(np.abs(np.asarray(g1)[outside]).max()) == 0.0)
    except ImportError:
        print("  (jax not installed: the gradient checks by jax are skipped; the finite-difference check above stands)")
    p1 = S.ramp_penalty(pi2, rec2, 1.3, 1.0, t_lo, t_up)
    check("linear in h: P(2h) = 2 P(h) exactly", np.array_equal(S.ramp_penalty(pi2, rec2, 2.6, 1.0, t_lo, t_up), 2 * p1))
    check("linear in k: P(k=2) = 2 P(k=1) exactly", np.array_equal(S.ramp_penalty(pi2, rec2, 1.3, 2.0, t_lo, t_up), 2 * p1))

    print("\n== S4  the arm-table rules (step2lib.arm_table) on made-up step-1 records")

    def entry(on, height, q90_full=1.0, q90_rest=0.0, p=1e-6, mean_full=0.5, mean_rest=0.0, t_lo=0.786, t_up=0.995):
        return dict(certificate=dict(on=on, height=height), t_lo=t_lo, t_up=t_up, rank_test=dict(p=p),
                    groups=dict(q90_full_throttle=q90_full, q90_rest=q90_rest, mean_full_throttle=mean_full,
                                mean_rest=mean_rest))

    rec_ok = {"per_seed_world": {"stall/seed0": entry(True, 2.5), "no_stall/seed0": entry(False, "inf", 0.01, 0.0, 0.7)}}
    tab = S.arm_table(rec_ok, seeds=[0], rule=ARM_RULE)
    st, ns = tab["stall"][0]["arms"], tab["no_stall"][0]["arms"]
    check("switch on + finite height -> h = the height, k = 1 (k = 2 for the dose arm)",
          st["obca_k1"] == dict(switch=True, h=2.5, k=1.0) and st["obca_k2"] == dict(switch=True, h=2.5, k=2.0))
    check("switch off + INFINITE height -> h = 0 and nothing refused (the no-stall seed-3 case)",
          ns["obca_k1"] == dict(switch=False, h=0.0, k=1.0))
    check("the always-on arm carries the stalling world's height into the no-stall world",
          ns["obca_forced"] == dict(switch=True, h=2.5, k=1.0))
    check("fixed height 1.0 when on; conformal gap when > 0.10; mean gap when the rank test p < 0.05",
          st["fixed_gated"]["h"] == 1.0 and st["conformal_k1"] == dict(switch=True, h=1.0, k=1.0)
          and st["meangap_k1"] == dict(switch=True, h=0.5, k=1.0) and ns["conformal_k1"]["h"] == 0.0 and ns["meangap_k1"]["h"] == 0.0)
    rec_bad = {"per_seed_world": {"stall/seed0": entry(True, "inf"), "no_stall/seed0": entry(False, 0.1)}}
    refused = None
    try:
        S.arm_table(rec_bad, seeds=[0], rule=ARM_RULE)
    except SystemExit as e:
        refused = str(e)
    check("switch on + INFINITE height -> refused (the fresh seeds 10, 12, 13 case)",
          refused is not None and "switched on but its height inf is not finite" in refused, refused or "no refusal")
    rec_neg = {"per_seed_world": {"stall/seed0": entry(True, -0.5), "no_stall/seed0": entry(False, 0.1)}}
    try:
        S.arm_table(rec_neg, seeds=[0], rule=ARM_RULE)
        neg = False
    except SystemExit as e:
        neg = "negative height" in str(e)
    check("a negative height is refused", neg)

    print("\n== S5  a toy with the truth known: which throttle an actor picks under the ramp")
    # One state. Throttles m on a grid. The critic (frozen) PROMISES q(m) = m (more throttle, more value) because it
    # never saw outcomes past t_lo; the TRUE value is q(m) for m <= knee and falls by d past the knee (the stall).
    # The critic's over-optimism at full throttle is therefore h_true = 1 - (1 - d) = d. The actor picks
    # argmax_m [q(m) - h ramp(m)] (the recorded command is at or below t_lo, so U(a_rec) = 0).
    grid = np.round(np.arange(0.0, 1.0001, 0.005), 3)

    def pick(h, t_lo, t_up, q):
        return float(grid[np.argmax(q - h * S.ramp(grid, t_lo, t_up))])

    # (a) main-setting pattern: the ramp starts where the harm starts (t_lo = knee = 0.80)
    d = 1.2
    q = grid.copy()
    knee = 0.80
    truth = np.where(grid > knee, grid - d, grid)
    best_true = float(grid[np.argmax(truth)])
    hs = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
    picks = {h: pick(h, 0.80, 1.0, q) for h in hs}
    # with t_lo = t_up - 0.2 and q(m) = m, the slope of the ramp is h / 0.2: the actor leaves full throttle once
    # h / 0.2 > 1, i.e. h > 0.2; it stays at or below t_lo once the whole climb is cancelled: the smallest such
    # h is the critic's over-optimism along the ramp, 0.2 here (q(1) - q(0.8)); the harm is removed for every h >= 0.2
    harmful = {h: picks[h] > knee for h in hs}
    check("toy (a): heights below the critic's over-optimism along the ramp leave the actor at full throttle",
          harmful[0.0] and harmful[0.1], f"picks {picks[0.0]}, {picks[0.1]}")
    check("toy (a): every height at or above it gives the SAME choice (the true best, 0.80): large enough is all "
          "the outcome can show", all(picks[h] == best_true for h in hs if h >= 0.2), f"picks {[picks[h] for h in hs if h >= 0.2]}")
    # (b) near-80 pattern: the data stop at 0.70 but the harm starts at 0.80; the ramp runs 0.70 -> 1.0; the true value
    # keeps rising from 0.70 to 0.80 (useful unseen throttles), then falls. With a straight-line critic and a
    # straight-line ramp the objective is linear on the ramp, so the actor sits at one END of the ramp: full throttle
    # (height below the over-optimism along the ramp, 0.3 here) or t_lo = 0.70 (any larger height). No height picks
    # the true best, 0.80: the ramp charges by throttle, not by harm, so the useful stretch is lost with the harm.
    truth_b = np.where(grid > knee, grid - d, grid)
    hs_b = [0.0, 0.2, 0.3, 0.5, 1.0, 2.0, 6.0]
    picks_b = {h: pick(h, 0.70, 1.0, q) for h in hs_b}
    true_value_b = {h: float(truth_b[np.argmin(np.abs(grid - m))]) for h, m in picks_b.items()}
    check("toy (b): with the ramp starting 0.10 below the stall, every height that removes the harm (h >= 0.3) also "
          "gives up the useful 0.70-0.80 stretch (picked 0.700, true value 0.70 against the true best 0.80), and no "
          "height on the grid recovers it",
          picks_b[0.0] > knee and all(picks_b[h] == 0.70 for h in hs_b if h >= 0.3)
          and max(true_value_b.values()) < float(truth_b.max()),
          "height -> picked throttle, true value: " + ", ".join(f"{h:g} -> {picks_b[h]:.3f}, {true_value_b[h]:.2f}" for h in hs_b)
          + f"; true best {float(truth_b.max()):.2f} at 0.80")
    # (c) a CURVED critic: the promise q(m) = 1 - (1 - m)^2 flattens toward full throttle (slope 2(1 - m)), the truth
    # falls by d past the knee 0.80, the ramp runs 0.70 -> 1.0. Now the objective q - h ramp is concave on the ramp and
    # the actor's pick moves smoothly down as h grows: q'(m) = h / 0.3, so m = 1 - h / 0.6 (reaching t_lo at h = 0.18). The pick equals the knee at
    # h* = q'(0.80) * 0.3 = 0.4 * 0.3 = 0.12 exactly: the calibrated height. Below it the actor is past the knee (harm);
    # above it the actor is below the knee (useful stretch lost). This is the shape under which "a calibrated height
    # beats both a smaller and a larger one" can be seen at all.
    q_c = 1.0 - (1.0 - grid) ** 2
    d_c = 0.5
    truth_c = np.where(grid > knee, q_c - d_c, q_c)
    hs_c = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.24, 0.3, 0.6, 1.2]
    picks_c = {h: pick(h, 0.70, 1.0, q_c) for h in hs_c}
    tv_c = {h: float(truth_c[np.argmin(np.abs(grid - m))]) for h, m in picks_c.items()}
    best_c = max(tv_c, key=tv_c.get)
    h_star = 2 * (1 - knee) * (1.0 - 0.70)
    check("toy (c): with a curved critic the picked throttle falls smoothly as the height grows (strictly between the "
          "ramp's ends for 0 < h < 0.18)", all(0.70 < picks_c[h] < 1.0 for h in hs_c if 0 < h < 0.18)
          and all(picks_c[a] >= picks_c[b] for a, b in zip(hs_c, hs_c[1:])),
          "height -> picked throttle: " + ", ".join(f"{h:g} -> {picks_c[h]:.3f}" for h in hs_c))
    check(f"toy (c): the true best height is {h_star:.2f} = the critic's slope at the knee (0.4) x the ramp's length "
          f"(0.3); smaller heights leave the actor past the knee (harm), larger ones pull it below the knee (useful "
          f"stretch lost): a calibrated height beats both", abs(best_c - h_star) < 1e-12
          and all(picks_c[h] > knee for h in hs_c if h < h_star) and all(picks_c[h] < knee for h in hs_c if h > h_star)
          and all(tv_c[h] < tv_c[h_star] for h in hs_c if h != h_star),
          "height -> true value: " + ", ".join(f"{h:g} -> {tv_c[h]:.3f}" for h in hs_c))
    print("  what the three toys mean for the recorded data: the step-2 and near-80 arms compare heights of 0.5 to 12 "
          "on real critics; whether the outcome has a peak (c) or a plateau (a, b) is an empirical question, read in R4")


# ================================================================================================= real data
def near80_cases():
    out = []
    for setting, seeds in (("near80", range(5)), ("random_rows", range(5)), ("main", (10, 11, 12, 13, 14))):
        for s in seeds:
            out.append((setting, s))
    return out


def switch_entries(setting, seed):
    per = {}
    for w in ("stall", "no_stall"):
        p = NEAR80 / "learners" / setting / w / f"seed{seed}" / "switch.json"
        if p.exists():
            per[f"{w}/seed{seed}"] = jl(p)["entry"]
    return per


def near80_row(setting, seed, world, per):
    """policy_phase.arm_row_from restated: step2lib.arm_table on both worlds' switch entries, then meandiff_k1 and
    (near-80 setting) knee_ramp; the random-rows setting keeps obca_k1 alone when arm_table refuses for another arm."""
    try:
        row = S.arm_table({"per_seed_world": per}, seeds=[seed], rule=ARM_RULE)[world][seed]
    except SystemExit as e:
        if setting != "random_rows":
            raise
        c = per[f"{world}/seed{seed}"]["certificate"]
        on, h = bool(c["on"]), S.num(c["height"])
        if on and not math.isfinite(h):
            raise
        g = per[f"{world}/seed{seed}"]["groups"]
        row = dict(t_lo=S.num(per[f"{world}/seed{seed}"]["t_lo"]), t_up=S.num(per[f"{world}/seed{seed}"]["t_up"]),
                   arms={"obca_k1": dict(switch=on, h=h if on else 0.0, k=1.0)},
                   sources=dict(mean_gap=S.num(g["mean_full_throttle"]) - S.num(g["mean_rest"]), refusal=str(e)))
    gap_mean = float(row["sources"]["mean_gap"])
    on = bool(np.isfinite(gap_mean) and gap_mean > MEANDIFF_MARGIN)
    row["arms"]["meandiff_k1"] = dict(switch=on, h=gap_mean if on else 0.0, k=1.0)
    if setting == "near80":
        o = row["arms"]["obca_k1"]
        row["arms"]["knee_ramp"] = dict(switch=o["switch"], h=o["h"], k=1.0, t_lo=KNEE32, t_up=row["t_up"])
    return row


def r1_arm_tables():
    print("\n== R1  every recorded arm table rebuilt from the switch records (pinned step2lib.arm_table, no refit)")
    n_rows = n_arms = 0
    mism = []
    stops = {}
    for setting, seed in near80_cases():
        per = switch_entries(setting, seed)
        for world in ("stall", "no_stall"):
            lock = NEAR80 / "policy" / setting / world / f"seed{seed}" / "lock.json"
            stop = NEAR80 / "policy" / setting / world / f"seed{seed}" / "stop.json"
            if stop.exists():
                rec = jl(stop)["detail"]["error"]
                try:
                    near80_row(setting, seed, world, per)
                    stops[(setting, world, seed)] = (rec, "NOT refused")
                except SystemExit as e:
                    stops[(setting, world, seed)] = (rec, str(e))
                continue
            if not lock.exists():
                continue
            want = jl(lock)["arm_table"]
            got = near80_row(setting, seed, world, per)
            n_rows += 1
            same = got["t_lo"] == want["t_lo"] and got["t_up"] == want["t_up"]
            for name, a in want["arms"].items():
                n_arms += 1
                b = got["arms"].get(name)
                ok = b is not None and bool(a["switch"]) == bool(b["switch"]) and a["h"] == b["h"] and a["k"] == b["k"] \
                    and a.get("t_lo", got["t_lo"]) == b.get("t_lo", got["t_lo"])
                same = same and ok
                if not ok:
                    mism.append((setting, world, seed, name, a, b))
            if not same:
                mism.append((setting, world, seed, "t_lo/t_up", (want["t_lo"], want["t_up"]), (got["t_lo"], got["t_up"])))
    n_locks = len(list(NEAR80.glob("policy/*/*/seed*/lock.json")))
    check(f"near-80 run: {n_rows} locked seed-worlds ({n_locks} lock.json files on disk), {n_arms} arm rows, every "
          f"switch, height, k, t_lo, t_up equal to the lock (exact float equality)",
          n_rows == n_locks == 24 and not mism, f"{len(mism)} mismatches")
    for m in mism[:10]:
        print("   mismatch:", m)
    ok_stops = all(rec == got for rec, got in stops.values()) and len(stops) == 3
    check("the three fresh-seed stops (main, stalling world, seeds 10, 12, 13) are reproduced with the recorded message",
          ok_stops, "; ".join(f"seed {k[2]}: {v[1][:60]}..." for k, v in sorted(stops.items())))
    # step 2's own table against prespec.json's expected table
    step1 = jl(STEP1_RESULTS)
    expected = jl(STEP2 / "prespec.json")["arm_table_expected"]
    tab = S.arm_table(step1, seeds=S.SEEDS, rule=ARM_RULE)
    check("step 2: the arm table rebuilt from step 1's results.json equals prespec.json's expected table "
          "(10 seed-worlds, 6 arms each)", S.table_matches(tab, expected, tol=0.0))
    return tab


def r2_plumbing():
    print("\n== R2  the height-0 plumbing check (every height-0 ramp arm must equal the no-caution actor within 1e-5)")
    rows = []
    s2 = jl(STEP2_RESULTS)["checks"]
    for key, c in sorted(s2.items()):
        w, s = key.split("/")
        rows.append(("step 2 (main)", w, int(s[4:]), "w0_check", c["b_w0"]["max_abs_diff"], "recorded"))
        for a, v in (c["h0_reproduces_reference"].get("arms") or {}).items():
            rows.append(("step 2 (main)", w, int(s[4:]), a, v["max_abs_diff"], "recorded"))
    for lock in sorted(NEAR80.glob("policy/*/*/seed*/lock.json")):
        L = jl(lock)
        c = L["checks"]
        rows.append((f"near-80 ({L['setting']})", L["world"], L["seed"], "w0_check", c["b_w0"]["max_abs_diff"], "recorded"))
        for a, v in (c["h0_reproduces_reference"].get("arms") or {}).items():
            rows.append((f"near-80 ({L['setting']})", L["world"], L["seed"], a, v["max_abs_diff"], "recorded"))
    rec = [r for r in rows if r[5] == "recorded"]
    worst = max(r[4] for r in rec)
    check(f"recorded: {len(rec)} height-0 arms in {len(set((r[0], r[1], r[2]) for r in rec))} seed-worlds, largest "
          f"parameter difference {worst:g} (tolerance {W0_TOL:g})", worst <= W0_TOL)
    by = {}
    for r in rec:
        by.setdefault((r[0], r[1]), [0, 0.0])
        by[(r[0], r[1])][0] += 1
        by[(r[0], r[1])][1] = max(by[(r[0], r[1])][1], r[4])
    print("  | study | world | height-0 arms checked | largest difference |")
    print("  |---|---|---|---|")
    for (st, w), (n, m) in sorted(by.items()):
        print(f"  | {st} | {w} | {n} | {m:g} |")
    # recomputed from the saved actors (bridge)
    files = sorted(BRIDGE_OUT.glob("*.json"))
    if not files:
        print("  (no bridge outputs in", BRIDGE_OUT, ": the recompute from the saved actors is not shown)")
        return
    n = 0
    worst = 0.0
    agree = True
    for f in files:
        b = jl(f)
        for a, dlt in b["plumbing_recomputed"].items():
            n += 1
            worst = max(worst, dlt)
            r = b["plumbing_recorded"]["b_w0"]["max_abs_diff"] if a == "w0_check" else \
                b["plumbing_recorded"]["h0"]["arms"][a]["max_abs_diff"]
            agree = agree and (r == dlt)
    check(f"recomputed from the saved actors (pinned environment, {len(files)} seed-worlds, {n} arms): largest "
          f"difference {worst:g}, equal to the recorded value in every case", worst <= W0_TOL and agree)


def r3_penalty_size():
    print("\n== R3  the penalty's size on recorded rows")
    # (a) the recorded ramp-shape readouts: mean penalty at each grid throttle, per arm, on the room's states
    n_pairs = 0
    worst_rel = 0.0
    full = []
    for lock in sorted(NEAR80.glob("policy/*/stall/seed*/lock.json")):
        L = jl(lock)
        shape = (L.get("ramp_shape") or {}).get("arms") or {}
        ref = shape.get("fixed_gated")
        if not ref or ref["h"] == 0:
            continue
        for name, a in shape.items():
            if a.get("t_lo") or name == "knee_ramp" or a["h"] == 0:
                continue
            scale = a["h"] * a["k"] / (ref["h"] * ref["k"])
            for m, w in a["mean_W"].items():
                if ref["mean_W"][m] > 0:
                    n_pairs += 1
                    worst_rel = max(worst_rel, abs(w / ref["mean_W"][m] - scale) / scale)
        o = shape.get("obca_k1")
        full.append((L["setting"], L["seed"], o["h"], o["mean_W"]["1"], o["mean_W"]["0.8"], o["best_past80_share"],
                     shape["fixed_gated"]["best_past80_share"] if "fixed_gated" in shape else None,
                     L["ramp_shape"].get("q1_alone_best_past80_share")))
    check(f"recorded ramp-shape readouts: the mean penalty at every grid throttle is proportional to h*k across the "
          f"arms of a seed-world ({n_pairs} ratios, largest relative deviation {worst_rel:.1e})", worst_rel < 1e-9)
    print("  | setting | seed | BCA height h | mean penalty at full throttle | at 80% | share of states whose best "
          "throttle under Q1 - P lies past 80%: BCA / fixed 1.0 / no penalty |")
    print("  |---|---|---|---|---|---|")
    for st, s, h, w1, w8, b, bf, b0 in full:
        print(f"  | {st} | {s} | {h:.2f} | {w1:.2f} | {w8:.2f} | {b:.3f} / {fmt(bf, 3)} / {fmt(b0, 3)} |")
    # (b) the bridge: what each saved actor pays on the recorded training rows
    allb = [jl(f) for f in sorted(BRIDGE_OUT.glob("*.json"))]
    stall = [b for b in allb if b["world"] == "stall"]
    nostall = [b for b in allb if b["world"] == "no_stall"]
    if not allb:
        print("  (no bridge outputs: the penalty at the saved actors' commands is not shown)")
        return
    print("  Stalling worlds (every 4th recorded training row; the penalty in critic value units, mean |Q1| about 15):")
    print("  | setting | seed | rows | BCA height h | recorded commands past t_lo | no caution: share past t_lo, "
          "penalty it would pay under BCA's ramp (mean; share > 0) | BCA ramp actor: share past t_lo, penalty paid "
          "(mean) | mean penalty while training, chunk 1 -> 5 |")
    print("  |---|---|---|---|---|---|---|---|")
    big = small = falling = True
    n_shown = 0
    for b in stall:
        a = b["arms"]
        n_, o_ = a["none"], a.get("obca_k1")
        if o_ is None:
            continue
        n_shown += 1
        kw = b.get("training_kW_mean_per_chunk_obca_k1") or []
        print(f"  | {b['setting']} | {b['seed']} | {b['rows_used']} | {o_['own_h']:.2f} | {b['recorded']['share_past_t_lo']:.4f} | "
              f"{n_['share_past_t_lo']:.3f}, {n_['penalty_under_bca_ramp_mean']:.3f}; {n_['penalty_under_bca_ramp_share_pos']:.3f} | "
              f"{o_['share_past_t_lo']:.3f}, {o_['own_penalty_mean']:.4f} | " + " -> ".join(f"{x:.4f}" for x in kw) + " |")
        big = big and n_["penalty_under_bca_ramp_mean"] > 10 * max(o_["own_penalty_mean"], 1e-12)
        small = small and o_["own_penalty_mean"] < 0.05 * o_["q1_abs_mean"]
        falling = falling and len(kw) == 5 and kw[0] > kw[-1] and kw[-1] < 0.01
    check(f"on every stalling seed-world ({n_shown}) the no-caution actor would pay at least 10x the penalty the trained "
          "ramp actor pays, and the trained actor's mean penalty is under 5% of mean |Q1| (the penalty bites at the "
          "margin, then is almost never paid)", big and small)
    check("while training, the mean penalty per chunk falls from the first chunk to the last and ends below 0.01 value "
          "units (the actor learns to stay below the ramp's start)", falling)
    # the no-stall worlds: the trained BCA arm pays exactly 0 (h = 0)
    z = all(b["arms"]["obca_k1"]["own_penalty_mean"] == 0.0 for b in nostall if "obca_k1" in b["arms"])
    check(f"no-stall worlds ({len(nostall)} seed-worlds): BCA's switched ramp pays exactly 0 (h = 0), and the "
          "no-caution actor would pay 0 under it too", z and all(b["arms"]["none"]["penalty_under_bca_ramp_mean"] == 0.0 for b in nostall))


def points_tables(step2_tab):
    """Strong-current points per arm from the near-80 evaluation records (seeds 0-4 of the main setting are step 2's
    saved actors re-evaluated there; every such file records step2_points_compared = "equal to step 2's, exactly").
    Heights h*k per arm: the near-80 locks, and step 2's arm table (rebuilt in R1) for main seeds 0-4."""
    pts = {}
    same = []
    for f in sorted(NEAR80.glob("eval/*/*/seed*/values.json")):
        v = jl(f)
        for arm, e in v["evaluations"].items():
            pts[(v["setting"], v["world"], v["seed"], arm)] = {s: e[s]["points"] for s in e}
        if v["setting"] == "main" and v["seed"] in range(5):
            same.append(v.get("step2_points_compared") == "equal to step 2's, exactly")
    check(f"main seeds 0-4: the near-80 run's evaluation of step 2's saved actors equals step 2's own points exactly "
          f"({len(same)} seed-world files)", bool(same) and all(same))
    heights = {}
    for w in ("stall", "no_stall"):
        for s, row in step2_tab[w].items():
            for arm, a in row["arms"].items():
                heights[("main", w, s, arm)] = a["h"] * a["k"]
            heights[("main", w, s, "none")] = 0.0
    for lock in sorted(NEAR80.glob("policy/*/*/seed*/lock.json")):
        L = jl(lock)
        for arm, a in L["arm_table"]["arms"].items():
            heights[(L["setting"], L["world"], L["seed"], arm)] = a["h"] * a["k"]
        heights[(L["setting"], L["world"], L["seed"], "none")] = 0.0
    return pts, heights


def r4_outcomes(pts, heights):
    print("\n== R4  the outcome by height (strong current; points: 0 = recording controller, 100 = best scripted)")
    print("  Stalling world, per seed: points of the five heights (the dose table); seeds 0-4 exploratory, 11 and 14 fresh "
          "(description only; seeds 10, 12, 13 stopped).")
    print("  | setting | seed | no caution | " + " | ".join(LABEL[a] + " (h)" for a in HEIGHT_ARMS) + " |")
    print("  |---|---|---|" + "---|" * len(HEIGHT_ARMS))
    table = {}
    for setting in ("main", "near80"):
        seeds = sorted({k[2] for k in pts if k[0] == setting and k[1] == "stall"})
        for s in seeds:
            cells = []
            for a in HEIGHT_ARMS:
                p = pts.get((setting, "stall", s, a), {}).get("strong_current")
                h = heights.get((setting, "stall", s, a))
                table[(setting, s, a)] = (h, p)
                cells.append(f"{fmt(p)} ({fmt(h, 2)})" if p is not None else "-")
            n0 = pts.get((setting, "stall", s, "none"), {}).get("strong_current")
            print(f"  | {setting} | {s} | {fmt(n0)} | " + " | ".join(cells) + " |")
    # the ordering checks
    for setting in ("main", "near80"):
        seeds = sorted({k[1] for k in table if k[0] == setting})
        k2_minus_k1 = [table[(setting, s, "obca_k2")][1] - table[(setting, s, "obca_k1")][1] for s in seeds]
        bca_vs_conf = [table[(setting, s, "obca_k1")][1] - table[(setting, s, "conformal_k1")][1] for s in seeds]
        bca_vs_fixed = [table[(setting, s, "obca_k1")][1] - table[(setting, s, "fixed_gated")][1] for s in seeds]
        print(f"  {setting}: 2xBCA minus BCA per seed {[round(x, 1) for x in k2_minus_k1]} (mean {np.mean(k2_minus_k1):+.1f}); "
              f"BCA minus conformal {[round(x, 1) for x in bca_vs_conf]}; BCA minus fixed 1.0 {[round(x, 1) for x in bca_vs_fixed]}")
    main_seeds = sorted({k[1] for k in table if k[0] == "main"})
    check("main setting: BCA's height beat the fixed height 1.0 in every seed (7 of 7, seeds 0-4, 11, 14)",
          all(table[("main", s, "obca_k1")][1] > table[("main", s, "fixed_gated")][1] for s in main_seeds))
    up = sum(table[("main", s, "obca_k2")][1] > table[("main", s, "obca_k1")][1] for s in main_seeds)
    check(f"main setting: twice BCA's height did BETTER than BCA's in {up} of {len(main_seeds)} seeds: BCA's height is "
          "not shown to be an optimum there", up >= len(main_seeds) - 1)
    n80 = sorted({k[1] for k in table if k[0] == "near80"})
    down = sum(table[("near80", s, "obca_k2")][1] < table[("near80", s, "obca_k1")][1] for s in n80)
    worse_than_conf = sum(table[("near80", s, "obca_k1")][1] < table[("near80", s, "conformal_k1")][1] for s in n80)
    check(f"near-80 setting: twice BCA's height did WORSE in {down} of 5 seeds and the smaller conformal height did "
          f"better in {worse_than_conf} of 5: the dose curve is not monotone when the ramp starts 10 points below the stall",
          down >= 3 and worse_than_conf >= 2)
    # nets across worlds (step 2's record, and the near-80 run's reproduction of them)
    s2 = jl(STEP2_RESULTS)
    net = s2["net_strong_current"]
    print("  Step 2 (main setting, seeds 0-4), net across worlds = gain in the stalling world minus loss in the no-stall world:")
    print("  | arm | height in the stalling world | net | gain (stall) | loss (no-stall) |")
    print("  |---|---|---|---|---|")
    for a in ("obca_k1", "obca_k2", "conformal_k1", "fixed_gated", "meangap_k1", "obca_forced", "standin", "range_hinge", "perfect_band", "even_pull"):
        n = net[a]
        hs = "BCA's" if a in ("obca_k1", "obca_forced") else ("2 x BCA's" if a == "obca_k2" else "-")
        print(f"  | {LABEL.get(a, a)} | {hs} | {n['net']:.1f} | {n['gain_stall']:.1f} | {n['loss_no_stall']:.1f} |")
    g = s2["go_no_go"]
    check("step 2's go marks G1, G2, G3 are all true in the record and the verdict is GO",
          g["G1"] and g["G2"] and g["G3"] and g["verdict"] == "GO")
    att = s2["attribution"]
    check("step 2's attribution: every smaller height did worse (A1, A2, A3 all fail), so the record credits 'a larger "
          "dose', not calibration", not att["A1"]["holds"] and not att["A2"]["holds"] and not att["A3"]["holds"]
          and not att["calibration_credit"]["credited"])
    dose = net["obca_k2"]["net"] - net["obca_k1"]["net"]
    print(f"  dose readout NET(k=2) - NET(k=1) = {dose:+.1f} (the prespec reads within 5 as a plateau: 'large enough', "
          f"not calibrated)")
    # the no-stall world: the switched arms equal no caution exactly (h = 0)
    eq = all(pts[(st, "no_stall", s, "obca_k1")]["strong_current"] == pts[(st, "no_stall", s, "none")]["strong_current"]
             for (st, w, s, a) in pts if w == "no_stall" and a == "obca_k1" and (st, "no_stall", s, "none") in pts)
    check("no-stall worlds: BCA's switched ramp scores exactly the no-caution points in every evaluated seed-world "
          "(the same actor)", eq)
    # fresh seeds
    print("  Fresh seeds, main setting (pre-registered claim 4): seeds 10, 12, 13 stopped (switch on, height +inf; "
          "19, 22, 18 full-throttle rows); seeds 11 and 14 (description only, a selected subset):")
    for s in (11, 14):
        o = pts.get(("main", "stall", s, "obca_k1"), {}).get("strong_current")
        f_ = pts.get(("main", "stall", s, "obca_forced"), {}).get("strong_current")
        n0 = pts.get(("main", "stall", s, "none"), {}).get("strong_current")
        on = pts.get(("main", "no_stall", s, "none"), {}).get("strong_current")
        of = pts.get(("main", "no_stall", s, "obca_forced"), {}).get("strong_current")
        print(f"    seed {s}: stalling world BCA {fmt(o)} vs no caution {fmt(n0)} (height {fmt(heights.get(('main', 'stall', s, 'obca_k1')), 2)}); "
              f"no-stall world always-on ramp {fmt(of)} vs no caution {fmt(on)}; net BCA {o - n0:.1f}, always-on {f_ - n0 - (on - of):.1f}")
    return table


# ================================================================================================= figure
def figure(table, pts, heights):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"  # keep the figure's text as text, not outlines
    # Okabe-Ito, colorblind-safe
    C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", purple="#CC79A7", black="#000000", sky="#56B4E9", verm="#D55E00")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    # left: the ramp penalty surface for near-80 stalling seed 0 and main (step 2) stalling seed 0
    L80 = jl(NEAR80 / "policy" / "near80" / "stall" / "seed0" / "lock.json")["arm_table"]
    Lmain = jl(NEAR80 / "policy" / "main" / "stall" / "seed11" / "lock.json")["arm_table"]
    t = np.linspace(0.6, 1.0, 801)
    for name, col, ls in (("obca_k2", C["purple"], ":"), ("obca_k1", C["blue"], "-"), ("conformal_k1", C["orange"], "-"),
                          ("fixed_gated", C["green"], "-"), ("meangap_k1", C["sky"], "-")):
        a = L80["arms"][name]
        ax1.plot(t, a["h"] * a["k"] * S.ramp(t, L80["t_lo"], L80["t_up"]), color=col, ls=ls, lw=2,
                 label=f"{LABEL[name]} (h·k = {a['h'] * a['k']:.2f})")
    a = Lmain["arms"]["obca_k1"]
    ax1.plot(t, a["h"] * S.ramp(t, Lmain["t_lo"], Lmain["t_up"]), color=C["black"], ls="--", lw=1.5,
             label=f"main setting, seed 11: BCA height {a['h']:.2f}")
    ax1.axvline(0.8, color=C["verm"], lw=1, ls="-.")
    ax1.text(0.806, 6.3, "stall begins (80% throttle)", color=C["verm"], fontsize=8, rotation=90, va="bottom", ha="left")
    ax1.axvline(L80["t_lo"], color="gray", lw=0.8)
    ax1.axvline(L80["t_up"], color="gray", lw=0.8)
    ax1.text(0.605, 3.0, "t_lo = 0.70 (gray line):\nwhere the recorded\nthrottles stop", fontsize=8, color="gray", va="bottom")
    ax1.text(L80["t_up"] - 0.004, 3.0, "t_up = 0.995:\nwhere the survivors begin", fontsize=8, color="gray", ha="right", va="bottom")
    ax1.set_xlabel("throttle index of the command, t(a) = largest |thruster command|")
    ax1.set_ylabel("penalty U(a) charged to the actor, critic value units\n(recorded command below t_lo, so U(a_rec) = 0)")
    ax1.set_title("What the actor is charged (near-80 setting, stalling world, seed 0)", fontsize=10)
    ax1.set_xlim(0.6, 1.0)
    ax1.set_ylim(0, 12.5)
    ax1.legend(fontsize=8, loc="upper left", framealpha=1.0)
    # right: points against height, one thin line per seed through its five heights
    mk = dict(meangap_k1="v", fixed_gated="s", conformal_k1="D", obca_k1="o", obca_k2="^")
    for setting, col in (("main", C["blue"]), ("near80", C["orange"])):
        seeds = sorted({k[1] for k in table if k[0] == setting})
        for s in seeds:
            xy = sorted((table[(setting, s, a)][0], table[(setting, s, a)][1], a) for a in HEIGHT_ARMS)
            ax2.plot([x for x, _, _ in xy], [y for _, y, _ in xy], color=col, lw=0.8, alpha=0.6)
            for x, y, a in xy:
                ax2.scatter([x], [y], color=col, marker=mk[a], s=28, zorder=3, edgecolor="white", lw=0.4)
        n0 = np.mean([pts[(setting, "stall", s, "none")]["strong_current"] for s in seeds])
        ax2.axhline(n0, color=col, ls="--", lw=1)
        ax2.text(0.42, n0 + 2.0 if setting == "main" else n0 - 7.0,
                 f"{'main' if setting == 'main' else 'near-80'} setting: no caution (height 0), mean {n0:.0f} points", color=col, fontsize=8)
    for a in HEIGHT_ARMS:
        ax2.scatter([], [], color="gray", marker=mk[a], s=28, label=LABEL[a])
    ax2.plot([], [], color=C["blue"], lw=2, label="main setting (ramp starts at 0.79, the data's edge = the stall; seeds 0-4, 11, 14)")
    ax2.plot([], [], color=C["orange"], lw=2, label="near-80 setting (ramp starts at 0.70, 10 points below the stall; seeds 0-4)")
    ax2.set_xscale("log")
    ax2.set_xlim(0.4, 14)
    ax2.set_xlabel("height of the ramp, h x k (critic value units, log scale)")
    ax2.set_ylabel("points in strong current, stalling world\n(0 = recording controller, 100 = best scripted controller)")
    ax2.set_title("Outcome against height: one line per seed through its five heights", fontsize=10)
    ax2.legend(fontsize=7.5, loc="center right", framealpha=1.0)
    ax2.grid(alpha=0.25, which="both")
    fig.suptitle("M11, the ramp height: the penalty surface (left) and the outcome against height (right)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="svg")
    print(f"\nfigure written: {FIGURE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    print("M11 ramp height: mechanical tests\n  repository root:", ROOT)
    synthetic()
    step2_tab = r1_arm_tables()
    r2_plumbing()
    r3_penalty_size()
    pts, heights = points_tables(step2_tab)
    table = r4_outcomes(pts, heights)
    if not args.no_figure:
        figure(table, pts, heights)
    n_fail = sum(not ok for _, ok in results)
    print(f"\n{len(results) - n_fail} of {len(results)} checks PASS" + (f"; FAILED: {[n for n, ok in results if not ok]}" if n_fail else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
