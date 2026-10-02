"""BCA from training to testing: one diagram of the whole pipeline, drawn from the code (no run data).

What it shows, in three bands:

Training (the host algorithm). A D4RL dataset is split by calibration/reference.py reserve_calibration (lines
49-121) into a training complement and a held-out calibration bank: about n/K length-weighted episodes are withheld
and K spaced rows of each enter the bank (calibration/bank.py stratified_bank), n = 248 to 8,192 rows (configs/
<host>.yaml reservation). The host (runtime/<host>.py, algorithms/<host>_bca.py) trains its twin critics and its
actor for 1M updates (configs/experiment.yaml num_updates) on the complement.

Calibration (BCA). Every update fits the live residual scale sigma(s, a) = u * eta_psi(s, a) on the training
minibatch: psi by gradient toward soft coverage 1 - alpha, the unit u by an EMA of the residual spread (fit_scale in
algorithms/{td3_bc,rebrac,cql}_bca.py, calibration/iql_scale.py; ALGORITHMS.md Procedure B).
At each refresh (update 10,000, then every 5,000 to 995,000: 198 refreshes; configs/experiment.yaml refresh_steps,
ALGORITHMS.md "Schedules") the live scale is frozen, the bank is scored rho = |y - min_k Q_k| / sigma with the
current critic, and WBCP (calibration/wbcp.py calibrate, uniform weights = BQ-CP) selects
R = max(lambda_hat, lambda_HPD) at alpha = 0.1, beta = 0.95, M = 1,000 draws (calibration/reference.py
freeze_reference, lines 152-194). Between refreshes each training row gets the frozen width U = R * sigma_f(s, a)
and the detached dose m = 1 + 0.5 U / (U + u) (calibration/dose.py lines 23-95; blend 0.5 in configs), which
multiplies one host loss term: the actor BC term for TD3+BC and ReBRAC, the conservative critic gap for CQL; for
IQL the capped actor weight's excess above 1 is shrunk by A / (A + U) (calibration/advantage.py
postcap_level_actor_weights, lines 32-71). Before update 10,000 there is no dose: hosts keep their native
weighting (calibration/reference.py initial_reference; calibration/dose.py line 90).

Testing. (1) calibration validity on frozen score pools (experiments/wbcp/d4rl_benchmark.py, 2,000 banks in the
hopper study of experiments/wbcp/README.md, 4,000 per design in experiments/wbcp/host_matrix.py TRIALS); (2) the
known-MDP linear-quadratic harness (experiments/signal/lq_harness.py, SIGNAL_STUDY.md step 3); (3) the step-4
placement study (experiments/signal/run_step4.py; pilot running, no outcomes); (4) full D4RL training for return
and regret (train.py; not run, needs approval). Status of (1) per host comes from the result files under
runs/wbcp_hosts/<host>/ (ReBRAC 60, CQL 60 and IQL 55 benchmark runs on all 7 datasets; TD3+BC's host matrix
started on 2026-10-02, after IQL finished) and DEPENDENCE.md Change 12 (TD3+BC pools, seven datasets).

Numbered tags tie each test to the part of the pipeline it checks. This is a schematic of code paths, not data.
"""

from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Circle, FancyBboxPatch

from experiments.wbcp.viz import style as S

W, H = 16.64, 9.4  # inches; the axes spans the figure, so data units are inches

TITLE, BODY = S.BASE, S.SMALL  # 19 pt box titles, 17 pt everything else
LINE = 0.30  # body line pitch (in)
PAD = 0.2  # inner box padding (in)


def _mix(color, t):
    """`color` laid over PAPER at opacity t, as an opaque hex (keeps band tints in the deck palette)."""
    a, b = to_rgb(color), to_rgb(S.PAPER)
    return to_hex(tuple(t * x + (1 - t) * y for x, y in zip(a, b)))


TRAIN_BAND, CAL_BAND = _mix(S.INK, 0.06), _mix(S.BLUE, 0.11)
PHASE = {  # box edge colour, width and style per phase
    "train": (S.INK, 1.6, "-"),
    "cal": (S.BLUE, 2.4, "-"),
    "test": (S.SOFT, 1.5, "-"),
}


def _round_box(ax, x0, y0, w, h, face, edge, lw, ls="-", r=0.12, z=2):
    patch = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=face,
                           edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(patch)
    return patch


def _box(ax, x0, y0, w, h, phase, title, body, title_x=None):
    """A card with a semibold title and short body lines, left-aligned from the top."""
    edge, lw, ls = PHASE[phase]
    _round_box(ax, x0, y0, w, h, S.CARD, edge, lw, ls)
    top = y0 + h
    ax.text(title_x if title_x is not None else x0 + PAD, top - 0.29, title, fontsize=TITLE, fontweight="semibold",
            color=S.INK, ha="left", va="center", zorder=4)
    for k, line in enumerate(body):
        ax.text(x0 + PAD, top - 0.68 - k * LINE, line, fontsize=BODY, color=S.SOFT, ha="left", va="center", zorder=4)


def _tag(ax, x, y, n):
    """A numbered badge: the same mark sits on a pipeline box and on the test that checks it."""
    ax.add_patch(Circle((x, y), 0.2, facecolor=S.INK, edgecolor=S.CARD, linewidth=2.0, zorder=5))
    ax.text(x, y - 0.01, str(n), fontsize=BODY, fontweight="semibold", color=S.CARD, ha="center", va="center",
            zorder=6)


CHIP_W, CHIP_H = 1.08, 0.36
CHIP = {  # status chips differ in fill lightness, edge style and label
    "done": dict(face=S.BLUE, edge=S.BLUE, ls="-", lw=1.5, text=S.CARD, weight="semibold"),
    "running": dict(face=S.BLUE_LIGHT, edge=S.BLUE, ls=(0, (4, 2)), lw=1.6, text=S.INK, weight="semibold"),
    "next": dict(face=S.CARD, edge=S.MUTED, ls=(0, (1.2, 1.6)), lw=2.0, text=S.SOFT, weight="semibold"),
}


def _chip(ax, x, y, kind, note=None):
    """Status chip with its left edge at x, centred on y, and an optional note after it; returns the note's end x."""
    c = CHIP[kind]
    _round_box(ax, x, y - CHIP_H / 2, CHIP_W, CHIP_H, c["face"], c["edge"], c["lw"], c["ls"], r=0.1, z=4)
    ax.text(x + CHIP_W / 2, y - 0.005, kind, fontsize=BODY, fontweight=c["weight"], color=c["text"], ha="center",
            va="center", zorder=5)
    if note:
        ax.text(x + CHIP_W + 0.13, y, note, fontsize=BODY, color=S.INK, ha="left", va="center", zorder=5)


def _arrow(ax, points, color, lw=2.4, label=None, label_at=None, ha="left"):
    """Polyline with an arrowhead on its last segment; optional label (17 pt) at label_at."""
    xs, ys = zip(*points)
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="butt", solid_joinstyle="miter", zorder=3)
    ax.annotate("", xy=points[-1], xytext=points[-2], zorder=3,
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.3", mutation_scale=24, color=color,
                                lw=lw, shrinkA=0, shrinkB=0, joinstyle="miter"))
    if label:
        ax.text(*label_at, label, fontsize=BODY, color=S.SOFT, ha=ha, va="center", zorder=4, linespacing=1.25)


def make():
    fig, ax = S.figure(figsize=(W, H), constrained_layout=False)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Bands: training (neutral tint), calibration (blue tint), testing (plain, ruled).
    _round_box(ax, 0.06, 7.12, W - 0.12, 2.22, TRAIN_BAND, "none", 0, r=0.18, z=0)
    _round_box(ax, 0.06, 3.50, W - 0.12, 3.56, CAL_BAND, "none", 0, r=0.18, z=0)
    _round_box(ax, 0.06, 0.06, W - 0.12, 3.30, S.PAPER, S.RULE, 1.4, r=0.18, z=0)
    head = dict(fontsize=TITLE, fontweight="semibold", ha="left", va="center", zorder=4)
    sub = dict(fontsize=BODY, color=S.SOFT, ha="left", va="center", zorder=4)
    for y, name, color, note in [(9.07, "Training", S.INK, "host algorithm, offline data"),
                                 (6.80, "Calibration", S.BLUE, "BCA"),
                                 (3.10, "Testing", S.INK, "numbers match the tags above")]:
        ax.text(0.3, y, name, color=color, **head)
        ax.text(1.95, y, note, **sub)

    # Four columns for the pipeline boxes.
    gap, bw, bh = 0.6, (W - 0.5 - 3 * 0.6) / 4, 1.52
    x0 = [0.25 + i * (bw + gap) for i in range(4)]
    cx = [x + bw / 2 for x in x0]
    y_train, y_cal = 7.36, 4.80  # box bottoms
    top_train, top_cal = y_train + bh, y_cal + bh

    # Training row.
    _box(ax, x0[0], y_train, bw, bh, "train", "D4RL dataset",
         ["logged transitions", "(s, a, r, s′)", "7 datasets per host"])
    _box(ax, x0[1], y_train, bw, bh, "train", "Episode reservation",
         ["withhold ~n/K episodes", "bank: K spaced rows each", "train on the other episodes"])
    _box(ax, x0[2], y_train, bw, bh, "train", "Host training",
         ["critics Q₁, Q₂ and actor π", "TD3+BC, ReBRAC, CQL or IQL", "1M updates"])
    _box(ax, x0[3], y_train, bw, bh, "train", "Dose scales one term",
         ["TD3+BC, ReBRAC: actor BC", "CQL: conservative critic gap", "IQL: actor weight above 1"])

    # Calibration row: scale fit (every update) -> refresh (freeze, score, WBCP) -> width and dose (every update).
    bx0, bx1, by0, by1 = x0[0] - 0.15, x0[1] + bw + 0.15, y_cal - 0.64, top_cal + 0.15
    _round_box(ax, bx0, by0, bx1 - bx0, by1 - by0, "none", S.BLUE, 1.6, ls=(0, (5, 3)), r=0.16, z=1)
    _box(ax, x0[2], y_cal, bw, bh, "cal", "Scale fit σ = u · η(s, a)",
         ["every update, on minibatch", "soft coverage target 1 − α", "kept live between refreshes"])
    _box(ax, x0[1], y_cal, bw, bh, "cal", "Score the held-out bank",
         ["freeze σ; current critic", "ρ = |y − min Q| / σ", "y: the host's Bellman target"])
    _box(ax, x0[0], y_cal, bw, bh, "cal", "WBCP threshold R",
         ["smallest λ: Pr(miss ≤ α) ≥ β", "and ≥ empirical 90% quantile", "α 0.1, β 0.95; uniform = BQ-CP"])
    _box(ax, x0[3], y_cal, bw, bh, "cal", "Width → dose",
         ["U = R · σ at logged (s, a)", "m = 1 + 0.5 · U / (U + u)", "IQL: shrink by A / (A + U)"])
    ax.text(x0[0] + 0.62, y_cal - 0.34, "refresh: at update 10k, then every 5k to 995k (198)", fontsize=BODY,
            color=S.BLUE, ha="left", va="center", zorder=4)

    # Arrows. Training flow in ink, BCA flow in blue.
    mid_tr, mid_cal, mid_gap = y_train + bh / 2, y_cal + bh / 2, (y_train + top_cal) / 2 + 0.02
    _arrow(ax, [(x0[0] + bw, mid_tr), (x0[1], mid_tr)], S.INK)
    _arrow(ax, [(x0[1] + bw, mid_tr), (x0[2], mid_tr)], S.INK)
    _arrow(ax, [(x0[3], mid_tr), (x0[2] + bw, mid_tr)], S.BLUE, lw=3.0)
    _arrow(ax, [(cx[1], y_train), (cx[1], top_cal)], S.INK, label="held-out bank\nn = 248 to 8,192 rows",
           label_at=(cx[1] - 0.16, mid_gap), ha="right")
    _arrow(ax, [(cx[2], y_train), (cx[2], top_cal)], S.INK, label="residuals\nevery update",
           label_at=(cx[2] + 0.16, mid_gap))
    _arrow(ax, [(x0[2], mid_cal), (x0[1] + bw, mid_cal)], S.BLUE)
    _arrow(ax, [(x0[1], mid_cal), (x0[0] + bw, mid_cal)], S.BLUE)
    lane = by0 - 0.36
    _arrow(ax, [(x0[0] + 0.35, y_cal), (x0[0] + 0.35, lane), (cx[3], lane), (cx[3], y_cal)], S.BLUE,
           label="frozen R and σ", label_at=(x0[2] + 0.35, lane + 0.21))
    _arrow(ax, [(cx[3], top_cal), (cx[3], y_train)], S.BLUE, lw=3.0, label="detached dose\nnone before 10k",
           label_at=(cx[3] - 0.16, mid_gap), ha="right")

    # Tags: which test checks which part (badge on the title line, top right).
    for n, (x, top) in {1: (x0[0], top_cal), 2: (x0[3], top_cal), 3: (x0[3], top_train),
                        4: (x0[2], top_train)}.items():
        _tag(ax, x + bw - 0.34, top - 0.29, n)

    # Testing row: four studies, status chips on a common bottom line.
    tg, tw1 = 0.3, 5.4
    tw = (W - 0.5 - tw1 - 3 * tg) / 3
    tx = [0.25] + [0.25 + tw1 + tg + k * (tw + tg) for k in range(3)]
    ty, th = 0.22, 2.62
    chip_y = ty + 0.36
    tests = [
        (tx[0], tw1, "Calibration validity", ["resampled banks from frozen pools", "2,000–4,000 banks, with shifts",
                                              "pass: ≤ 5% of banks miss > 10%"]),
        (tx[1], tw, "Known-MDP test", ["linear-quadratic, exact Q", "true Q error, value change"]),
        (tx[2], tw, "Placement study", ["where to apply the dose", "vs same-strength controls",
                                         "no outcomes yet"]),
        (tx[3], tw, "Full D4RL training", ["return and regret vs host", "needs approval"]),
    ]
    for k, (x, w, title, body) in enumerate(tests, start=1):
        _box(ax, x, ty, w, th, "test", title, body, title_x=x + PAD + 0.52)
        _tag(ax, x + PAD + 0.2, ty + th - 0.29, k)
    _chip(ax, tx[0] + PAD, chip_y + 0.44, "done", "TD3+BC study; ReBRAC, CQL, IQL")
    _chip(ax, tx[0] + PAD, chip_y, "running", "TD3+BC matrix")
    _chip(ax, tx[1] + PAD, chip_y, "done", "step 3")
    _chip(ax, tx[2] + PAD, chip_y, "running", "pilot")
    _chip(ax, tx[3] + PAD, chip_y, "next")

    facts = _facts()
    return fig, facts


def _facts():
    f = lambda what, value, source: {"what": what, "value": value, "source": source}
    return [
        # Boxes and the code they come from.
        f("box: D4RL dataset (HDF5 checked by SHA-256, host transition conversion)", "7 datasets per host",
          "runtime/{td3_bc,rebrac,cql,iql}.py preparation; configs/<host>.yaml datasets; "
          "experiments/wbcp/host_matrix.py:39 DATASETS"),
        f("box: Episode reservation (withhold ceil(n/K) length-weighted episodes, K spaced rows each, train on "
          "the complement)", "reserve_calibration",
          "calibration/reference.py:49-121; calibration/bank.py:45 stratified_bank; algorithms/<host>_bca.py "
          "reserve_pool; calibration/iql_reference.py:265 prepare_dataset"),
        f("held-out bank size n across hosts and datasets", "248 to 8,192 rows",
          "configs/{td3_bc,rebrac,cql,iql}.yaml reservation.size; ALGORITHMS.md:319"),
        f("box: Host training (twin critics + actor; TD3+BC, ReBRAC, CQL, IQL)", "1M updates",
          "configs/experiment.yaml:2 num_updates 1000000; runtime/{td3_bc,rebrac,cql,iql_pair}.py; "
          "algorithms/{td3_bc,rebrac,cql,iql}.py"),
        f("box: Dose scales one host loss term: TD3+BC actor BC (bc_multiplier), ReBRAC actor BC "
          "(actor_bc_multiplier; critic BC fixed), CQL conservative gap (conservative_multiplier), IQL actor "
          "weight excess above 1", "one term per host",
          "ALGORITHMS.md:43-48 and 321-363; INTEGRATION.md:15-20; algorithms/td3_bc.py:100; "
          "algorithms/rebrac.py:172; algorithms/cql.py:162,300; calibration/iql_state.py actor_update"),
        f("box: Scale fit sigma(s,a) = u * max(eta_psi(s,a), 1e-6), every update on the training minibatch: "
          "psi by Adam on (soft coverage - (1 - alpha))^2 + width penalty; unit u by EMA "
          "u <- 0.99 u + 0.01 std(y - q) (IQL: of its per-batch unit)", "every update",
          "algorithms/td3_bc_bca.py:159, rebrac_bca.py:283, cql_bca.py:176 fit_scale; calibration/iql_scale.py; "
          "calibration/network.py:34 soft_coverage; ALGORITHMS.md:232-265"),
        f("box: Score the held-out bank, rho = |y - min_k Q_k| / sigma_f with the current critic "
          "(IQL: min over target heads)", "freeze_reference",
          "calibration/reference.py:152-175; algorithms/td3_bc_bca.py:234 refresh, rebrac_bca.py:340, "
          "cql_bca.py:227; calibration/iql_reference.py:184; INTEGRATION.md:895-"),
        f("box: WBCP threshold R = max(lambda_hat, lambda_HPD): the smallest score that is beta-credible "
          "(lambda_HPD = ceil(beta M)-th sorted per-draw crossing, Dirichlet masses plus a worst-case test atom) "
          "and not below the empirical (1 - alpha) = 90% quantile lambda_hat; uniform weights = BQ-CP",
          "calibrate", "calibration/wbcp.py:43-54 crossings, 89-102 lambda_hat, 115-117 lambda_hpd and max; "
          "calibration/reference.py:175-179; ALGORITHMS.md Procedure C steps 4-5"),
        f("alpha (target miscoverage)", "0.1", "calibration/reference.py:18; configs/<host>.yaml bca.alpha"),
        f("beta (credibility)", "0.95", "calibration/reference.py:19; configs/<host>.yaml bca.credibility"),
        f("posterior draws M", "1,000", "calibration/reference.py:20; configs/<host>.yaml bca.draws"),
        f("box: Width -> dose, U = R * sigma_f(s,a) at the recorded (logged) minibatch actions, "
          "m = 1 + blend * U / (U + u_f), u = frozen unit u_f",
          "blend 0.5 (dose between 1 and 1.5)",
          "calibration/dose.py:23-62 level_critic_dose, 65-95 frozen_level_dose; configs/{td3_bc,rebrac,cql}.yaml "
          "bca.blend 0.5; ALGORITHMS.md:334-353"),
        f("IQL: positive-advantage weight 1 + A/(A+U) * (w_host - 1)", "shrink by A/(A+U)",
          "calibration/advantage.py:32-71 postcap_level_actor_weights; ALGORITHMS.md:342-349"),
        f("first frozen reference (after update 10,000, used from 10,001); before it no dose: multiplier 1 for "
          "TD3+BC, ReBRAC, CQL and native capped AWR weights for IQL", "update 10k (warmup 10000)",
          "configs/experiment.yaml:5 warmup, refresh_steps first_step 10000; calibration/reference.py:139-145; "
          "calibration/dose.py:90; ALGORITHMS.md:664"),
        f("refresh interval and last refresh", "every 5k to 995k",
          "configs/experiment.yaml:6 refresh_interval 5000; refresh_steps last_step 995000"),
        f("number of refreshes per run", "198", "ALGORITHMS.md:665 (10k, 15k, ..., 995k)"),
        f("refresh call sites (eager, between jitted scan blocks)", "4 hosts",
          "runtime/td3_bc.py:1064, runtime/rebrac.py:840, runtime/cql.py:816, runtime/iql_pair.py:286"),
        # Tests.
        f("test 1: calibration validity on frozen score pools, resampled banks, imposed shifts",
          "2,000 to 4,000 banks per design",
          "experiments/wbcp/d4rl_benchmark.py; experiments/wbcp/README.md:185 (2000 trials); "
          "experiments/wbcp/host_matrix.py:40 TRIALS = 4000"),
        f("test 1 pass rule: a bank fails when its threshold misses more than alpha = 10% of test rows; a valid "
          "rule fails in at most 1 - beta = 5% of banks", "≤ 5% of banks miss > 10%",
          "calibration/bank.py:28 DEPENDENCE_BUDGET 0.05; experiments/wbcp/d4rl_benchmark.py docstring"),
        f("test 1 status: TD3+BC pools, seven datasets (dependence study)", "done",
          "experiments/wbcp/DEPENDENCE.md Change 12"),
        f("test 1 status: ReBRAC and CQL host matrices", "done (60 JSON files each, all 7 datasets)",
          "runs/wbcp_hosts/rebrac/*.json, runs/wbcp_hosts/cql/*.json; experiments/wbcp/results_rebrac.ipynb, "
          "results_cql.ipynb"),
        f("test 1 status: IQL host matrix", "done (55 of 55 benchmark runs, 2026-10-02)",
          "host_matrix.py --host iql --stage status; experiments/wbcp/results_iql.ipynb"),
        f("test 1 status: TD3+BC host matrix", "running (benchmark started 2026-10-02 after IQL finished)",
          "host_matrix.py --host td3_bc --stage status; runs/wbcp_hosts/td3_bc/"),
        f("test 2: known-MDP linear-quadratic harness, exact Q^pi, true Q error and true value change J", "done",
          "experiments/signal/lq_harness.py; experiments/signal/SIGNAL_STUDY.md step 3"),
        f("test 3: step-4 placement study (P1, P2, P3, P1pi, P2pi at matched strength vs controls)",
          "running (pilot; no outcomes)", "experiments/signal/run_step4.py; experiments/signal/STEP4_DESIGN.md"),
        f("test 4: full D4RL training for return and regret vs host", "next (not run; needs approval)",
          "train.py; ALGORITHMS.md 'Schedules, counters, and saved evidence'"),
    ]
