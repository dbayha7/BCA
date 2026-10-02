"""Calibrating the critic the actor actually follows (key: critic_alignment).

TD3+BC's actor climbs the online Q1 alone (algorithms/td3_bc.py:130), while BCA as recorded in the baseline scores the
residual of min(Q1, Q2) (algorithms/td3_bc_bca.py:173 in the scale fit, :255 in the refresh). Where Q1 is optimistic
and Q2 accurate, the min equals Q2, so the residual BCA calibrates carries none of Q1's error.

Panel a (Illustration, computed here). One action slice of the linear-quadratic harness's q1_optimistic case
(experiments/signal/lq_harness.py), with exact Q. The system matrices, the LQR gain K* and the perturbation direction
D are read from the step-3 run (runs/wbcp_signal/lq/results.json, meta), and Q^pi is recomputed in numpy with the
harness's closed form (quadratic_q, lq_harness.py:175-191; reward_form :194-195; lqr_gain :227-231, checked against
the recorded K*). Settings are cells of the run's own grid: poor behaviour (K_b = poor_gain K* = 0, :582 and
Settings :516-517, behaviour noise 0.2), pi_offset 1 (K_pi = K_b + offset D, :657) and kappa 1. The heads are
Q1 = Q^pi + kappa ||a - beta(s)||^2 and Q2 = Q^pi (case_critic :267-286, head_error :288-292), so min(Q1, Q2) = Q2 =
Q^pi exactly (the run's built-in check 4). The slice runs through the behaviour mean beta(s) and Q1's maximiser at the
state s = (0.5, 0.5, 0.5), one initial-state SD per coordinate (the state is chosen for display).

Panel b (real data). Signal study step 2 on frozen D4RL TD3+BC critics (runs/wbcp_signal/frozen/<pool>/
evaluation_b4000_s2026100201.json, design "iid": 4,000 banks of 1,024 independent held-out rows, uniform-weight
WBCP at alpha = 0.1, beta = 0.95; each signal with its own refit sigma). For the five pools whose critic converged:
the share of banks whose threshold, calibrated on min(Q1, Q2) or on Q1, misses more than 10% of Q1's residuals, with
the 95% intervals stored in the same files. pen-cloned and pen-human are excluded, as in SIGNAL_STUDY.md (critic not
converged). The annotation "Q1 > Q2 on 46-54% of rows" is head_disagreement.q1_above_q2 from the same files: on
D4RL the heads disagree in both directions (by about one residual), so panel a's one-sided optimism is a stylized
case of the mechanism, not a description of these critics. CRITIC_ALIGNMENT.md's earlier measurement (online sigma, 1,000 banks: 8-70%) is not shown; its run folder
runs/wbcp_critic_alignment is not on disk to verify.
"""

import json
import math

import numpy as np
from scipy import linalg

from experiments.wbcp.viz import style as S

LQ_RESULTS = S.ROOT / "runs" / "wbcp_signal" / "lq" / "results.json"
FROZEN = S.ROOT / "runs" / "wbcp_signal" / "frozen"
EVALUATION = "evaluation_b4000_s2026100201.json"
POOLS = {"hopper-medium-v2": "hopper", "walker2d": "walker2d", "halfcheetah": "halfcheetah", "maze2d": "maze2d",
         "pen-expert": "pen-expert"}  # folder -> label; the five pools with a converged critic

BEHAVIOR, OFFSET, KAPPA = "poor", 1.0, 1.0  # one cell of the step-3 grid
STATE = np.array([0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------------------------------------------------
# Panel a: the LQ harness's closed forms in numpy


def _quadratic_q(system, K, G):
    """lq_harness.quadratic_q (lq_harness.py:175-191): exact (H, h) of a = -K s for the per-step reward z^T G z."""
    A, B, gamma = system["A"], system["B"], system["gamma"]
    M = A - B @ K
    if not math.sqrt(gamma) * np.max(np.abs(np.linalg.eigvals(M))) < 1:
        raise ValueError("the closed loop is not discounted-stable")
    Pi = np.vstack([np.eye(A.shape[0]), -K])
    P = linalg.solve_discrete_lyapunov(math.sqrt(gamma) * M.T, Pi.T @ G @ Pi)
    P = (P + P.T) / 2
    F = np.hstack([A, B])
    H = G + gamma * F.T @ P @ F
    return (H + H.T) / 2, gamma * float(np.trace(P @ system["noise"])) / (1 - gamma)


def _lqr_gain(system):
    """lq_harness.lqr_gain (lq_harness.py:227-231): the discounted LQR optimum K*."""
    g, A, B = system["gamma"], system["A"], system["B"]
    P = linalg.solve_discrete_are(math.sqrt(g) * A, math.sqrt(g) * B, system["Qc"], system["Rc"])
    return np.linalg.solve(system["Rc"] + g * B.T @ P @ B, g * B.T @ P @ A)


def lq_slice(points=400):
    meta = json.loads(LQ_RESULTS.read_text(encoding="utf8"))["meta"]
    system = {k: (np.asarray(v, np.float64) if isinstance(v, list) else v) for k, v in meta["system"].items()}
    k_opt, direction = np.asarray(meta["k_opt"]), np.asarray(meta["direction"])
    if not np.allclose(_lqr_gain(system), k_opt, atol=1e-10):
        raise ValueError("recomputed K* differs from the run's")
    settings = meta["settings"]
    K_b = k_opt * {"good": 1.0, "poor": settings["poor_gain"]}[BEHAVIOR]
    K_pi = K_b + OFFSET * direction
    G = -linalg.block_diag(system["Qc"], system["Rc"])  # reward_form
    H, h = _quadratic_q(system, K_pi, G)
    Haa, Has = H[3:, 3:], H[3:, :3]
    if np.max(np.linalg.eigvalsh(Haa + KAPPA * np.eye(2))) >= 0:
        raise ValueError("Q1 is not concave in a; its maximiser is undefined")
    beta = -K_b @ STATE
    # Q1(s, a) = Q^pi(s, a) + kappa ||a - beta||^2; its gradient in a vanishes at (Haa + kappa I) a = kappa beta - Has s.
    a_q1 = np.linalg.solve(Haa + KAPPA * np.eye(2), KAPPA * beta - Has @ STATE)
    a_true = -np.linalg.solve(Haa, Has @ STATE)  # the maximiser of Q^pi over both action dimensions
    e = (a_q1 - beta) / np.linalg.norm(a_q1 - beta)
    t = np.linspace(-0.75, 1.85, points)
    actions = beta + t[:, None] * e
    z = np.concatenate([np.broadcast_to(STATE, (points, 3)), actions], axis=1)
    q_true = np.einsum("ni,ij,nj->n", z, H, z) + h
    q1 = q_true + KAPPA * t ** 2
    # On the slice Q^pi(t) = c + b t + q t^2; its peak is at -b / 2q.
    curv = float(e @ Haa @ e)
    lin = float(2 * e @ (Haa @ beta + Has @ STATE))
    t_true = -lin / (2 * curv)
    t_q1 = float(np.linalg.norm(a_q1 - beta))
    at = lambda x: float(np.concatenate([STATE, beta + x * e]) @ H @ np.concatenate([STATE, beta + x * e]) + h)
    return dict(t=t, q_true=q_true, q1=q1, t_true=t_true, q_at_true=at(t_true), t_q1=t_q1,
                q_true_at_q1=at(t_q1), q1_at_q1=at(t_q1) + KAPPA * t_q1 ** 2, err_at_q1=KAPPA * t_q1 ** 2,
                logged=2 * settings["behavior_noise"], sigma_b=settings["behavior_noise"], a_q1=a_q1,
                a_true=a_true, beta=beta, dist_true=float(np.linalg.norm(a_true - beta)))


# ---------------------------------------------------------------------------------------------------------------------
# Panel b: step 2's failure rates on the frozen D4RL critics


def pool_rates():
    rows = []
    for folder, label in POOLS.items():
        path = FROZEN / folder / EVALUATION
        data = json.loads(path.read_text(encoding="utf8"))
        if not data["critic_converged"]:
            raise ValueError(folder + ": critic not converged")
        design = next(d for d in data["designs"] if d["design"]["name"] == "iid")
        signals = design["signals"]
        rows.append(dict(
            label=label, dataset=data["dataset"], path=path.relative_to(S.ROOT).as_posix(),
            banks=design["signals"]["min"]["failure"]["q1"]["trials"], n=design["design"]["n"],
            alpha=data["settings"]["alpha"], beta=data["settings"]["beta"],
            min_band=signals["min"]["failure"]["q1"]["fail"], min_ci=signals["min"]["failure"]["q1"]["ci"],
            q1_band=signals["q1"]["failure"]["q1"]["fail"], q1_ci=signals["q1"]["failure"]["q1"]["ci"],
            q1_band_min=signals["q1"]["failure"]["min"]["fail"],
            q1_mis_min_band=signals["min"]["failure"]["q1"]["mean_miscoverage"],
            min_mis_min_band=signals["min"]["failure"]["min"]["mean_miscoverage"],
            q1_above=data["head_disagreement"]["q1_above_q2"],
            gap_ratio=data["head_disagreement"]["gap_over_residual"],
            width=signals["q1"]["half_width_mean"] / signals["min"]["half_width_mean"]))
    return sorted(rows, key=lambda r: -r["min_band"])


# ---------------------------------------------------------------------------------------------------------------------


def _panel_a(ax, L, facts):
    t = L["t"]
    ax.axvspan(-L["logged"], L["logged"], color=S.MUTED, alpha=0.16, lw=0, zorder=0)
    ax.fill_between(t, L["q_true"], L["q1"], color=S.BLUE_LIGHT, alpha=0.45, lw=0, zorder=1)
    ax.plot(t, L["q_true"], color=S.ORANGE, lw=3.4, zorder=3)
    ax.plot(t, L["q1"], color=S.BLUE, lw=3.4, ls=(0, (6, 2.2)), zorder=4)

    ax.plot([L["t_true"]], [L["q_at_true"]], "o", ms=13, color=S.ORANGE, mec=S.PAPER, mew=2, zorder=5)
    ax.plot([L["t_q1"]], [L["q1_at_q1"]], "o", ms=13, color=S.BLUE, mec=S.PAPER, mew=2, zorder=5)
    ax.annotate("", xy=(L["t_q1"], L["q_true_at_q1"]), xytext=(L["t_q1"], L["q1_at_q1"]),
                arrowprops=dict(arrowstyle="<->", color=S.INK, lw=2, shrinkA=7, shrinkB=1), zorder=6)

    ymin = min(L["q_true"].min(), L["q1"].min())
    ymax = max(L["q1"].max(), L["q_true"].max())
    pad = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + 0.55 * (ymax - ymin))
    ax.set_xlim(t[0], t[-1])
    top = ax.get_ylim()[1]

    ax.text(0, top - 0.04 * (top - ax.get_ylim()[0]), "logged\nactions", ha="center", va="top", fontsize=S.SMALL,
            color=S.SOFT)
    ax.text(L["t_q1"] + 0.05, L["q1_at_q1"] + 0.1 * (ymax - ymin), "actor's Q term\npulls to here", ha="center",
            va="bottom", fontsize=S.SMALL, color=S.BLUE)
    ax.text(L["t_true"], L["q_at_true"] - 0.07 * (ymax - ymin), "true Q's peak", ha="center", va="top",
            fontsize=S.SMALL, color=S.ORANGE)
    ax.text(L["t_q1"] + 0.06, (L["q_true_at_q1"] + L["q1_at_q1"]) / 2,
            f"Q1's error {L['err_at_q1']:.2f}:\nnot in t − min(Q1, Q2)", ha="left", va="center", fontsize=S.SMALL,
            color=S.INK)
    q_at = lambda x, curve: float(np.interp(x, t, L[curve]))
    ax.text(-0.08, q_at(-0.08, "q1") + 0.13 * (ymax - ymin), "Q1, optimistic\noff the data", ha="right",
            va="bottom", fontsize=S.SMALL, color=S.BLUE)
    ax.text(0.95, q_at(0.95, "q_true") - 0.35 * (ymax - ymin), "Q2 = min(Q1, Q2)\n= true Q", ha="center", va="top",
            fontsize=S.SMALL, color=S.ORANGE)
    ax.text(t[-1] - 0.03, top - 0.04 * (top - ax.get_ylim()[0]),
            f"LQ harness, exact Q, {BEHAVIOR} data\nq1_optimistic, κ = {KAPPA:g}", ha="right", va="top",
            fontsize=S.SMALL, color=S.SOFT)

    ax.set_xlabel("action on the slice, distance from β(s)")
    ax.set_ylabel("Q(s, a)")
    ax.tick_params(length=4)
    S.panel_label(ax, "a  Illustration: min(Q1, Q2) hides Q1's optimism")

    src = "computed in fig_critic_alignment.lq_slice from runs/wbcp_signal/lq/results.json meta (system, k_opt, " \
          "direction, settings) with lq_harness.py's closed forms"
    facts += [
        dict(what="panel a setting: LQ harness case, behaviour, pi_offset, kappa, state",
             value=f"q1_optimistic, {BEHAVIOR}, {OFFSET:g}, {KAPPA:g}, s = (0.5, 0.5, 0.5)",
             source="cells of the step-3 grid (runs/wbcp_signal/lq/results.json meta: kappas, pi_offsets, behaviors); "
                    "state chosen for display"),
        dict(what="panel a heads: Q1 = Q^pi + kappa ||a - beta(s)||^2, Q2 = Q^pi, so min(Q1, Q2) = Q^pi",
             value="construction", source="experiments/signal/lq_harness.py:267-292 (case_critic, head_error)"),
        dict(what="panel a logged-action span along the slice, beta(s) +/- 2 sigma_b", value=f"+/-{L['logged']:.1f}",
             source="behavior_noise 0.2, runs/wbcp_signal/lq/results.json meta.settings"),
        dict(what="panel a peak of true Q on the slice (distance from beta(s))", value=f"{L['t_true']:.2f}",
             source=src),
        dict(what="panel a peak of Q1 (Q1's global maximiser; distance from beta(s))", value=f"{L['t_q1']:.2f}",
             source=src),
        dict(what="panel a Q1's error at its peak, Q1 - Q^pi", value=f"{L['err_at_q1']:.2f}", source=src),
        dict(what="panel a axis ranges: Q(s, a) and the slice coordinate",
             value=f"Q {L['q_true'].min():.2f} to {L['q1'].max():.2f}; slice {t[0]:.2f} to {t[-1]:.2f}", source=src),
    ]


def _panel_b(ax, rows, facts):
    y = np.arange(len(rows))[::-1].astype(float)
    h = 0.36
    for yi, r in zip(y, rows):
        lo, hi = 100 * np.array(r["min_ci"])
        ax.barh(yi + h / 2, 100 * r["min_band"], height=h, color=S.ORANGE, zorder=2)
        ax.errorbar(100 * r["min_band"], yi + h / 2, xerr=[[100 * r["min_band"] - lo], [hi - 100 * r["min_band"]]],
                    fmt="none", ecolor=S.INK, elinewidth=1.6, capsize=4, zorder=3)
        ax.text(hi + 1.2, yi + h / 2, f"{100 * r['min_band']:.1f}%", ha="left", va="center", fontsize=S.SMALL,
                color=S.INK)
        lo, hi = 100 * np.array(r["q1_ci"])
        ax.barh(yi - h / 2, 100 * r["q1_band"], height=h, color=S.BLUE, zorder=2)
        ax.errorbar(100 * r["q1_band"], yi - h / 2, xerr=[[100 * r["q1_band"] - lo], [hi - 100 * r["q1_band"]]],
                    fmt="none", ecolor=S.INK, elinewidth=1.6, capsize=4, zorder=3)
        ax.text(hi + 1.2, yi - h / 2, f"{100 * r['q1_band']:.1f}%", ha="left", va="center", fontsize=S.SMALL,
                color=S.INK)
    ax.axvline(5, color=S.INK, lw=2, ls=(0, (4, 3)), zorder=4)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=S.TICK, labelcolor=S.INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(y[-1] - 0.75, y[0] + 0.95)
    ax.set_xticks([0, 5, 25, 50, 75, 100])
    ax.set_xlabel("banks whose threshold misses > 10% of Q1's residuals (%)")
    ax.spines["left"].set_visible(False)

    first = y[0]
    ax.text(6.5, first + h / 2, "calibrated on min(Q1, Q2)", ha="left", va="center", fontsize=S.SMALL,
            color=S.INK, zorder=5)
    ax.text(98, y[0] + 0.62, f"{rows[0]['banks']:,} banks of {rows[0]['n']:,} rows per pool", ha="right",
            va="center", fontsize=S.SMALL, color=S.SOFT)
    ax.text(100 * rows[0]["q1_ci"][1] + 12.5, first - h / 2, "calibrated on Q1", ha="left", va="center",
            fontsize=S.SMALL, color=S.BLUE)
    lo_a, hi_a = min(r["q1_above"] for r in rows), max(r["q1_above"] for r in rows)
    ax.text(98, y[2], f"Real heads err both ways:\nQ1 > Q2 on {100 * lo_a:.0f}-{100 * hi_a:.0f}% of rows",
            ha="right", va="center", fontsize=S.SMALL, color=S.SOFT, linespacing=1.3)
    ax.text(5.8, y[0] + 0.62, "5% budget (1 − β)", ha="left", va="center", fontsize=S.SMALL, color=S.INK)
    lo_w, hi_w = min(r["width"] for r in rows), max(r["width"] for r in rows)
    lo_f, hi_f = min(r["q1_band"] for r in rows), max(r["q1_band"] for r in rows)
    ax.text(98, y[-1] - 0.35, f"Calibrating Q1 instead:\n{100 * lo_f:.1f}-{100 * hi_f:.1f}% of banks fail,\n"
            f"band {100 * (lo_w - 1):.0f}-{100 * (hi_w - 1):.0f}% wider", ha="right", va="bottom", fontsize=S.SMALL,
            color=S.SOFT, linespacing=1.3)
    S.panel_label(ax, "b  D4RL critics: banks failing for Q1")

    for r in rows:
        facts += [
            dict(what=f"panel b {r['label']} ({r['dataset']}): banks failing for Q1's residual, min(Q1, Q2)-calibrated "
                      "band, with 95% interval",
                 value=f"{100 * r['min_band']:.1f}% [{100 * r['min_ci'][0]:.1f}, {100 * r['min_ci'][1]:.1f}]",
                 source=r["path"] + ' designs[iid].signals.min.failure.q1'),
            dict(what=f"panel b {r['label']}: banks failing for Q1's residual, Q1-calibrated band, with 95% interval",
                 value=f"{100 * r['q1_band']:.1f}% [{100 * r['q1_ci'][0]:.1f}, {100 * r['q1_ci'][1]:.1f}]",
                 source=r["path"] + ' designs[iid].signals.q1.failure.q1'),
            dict(what=f"panel b {r['label']}: mean half-width, Q1 band / min band",
                 value=f"{r['width']:.3f}", source=r["path"] + " designs[iid].signals.{q1,min}.half_width_mean"),
        ]
    facts += [
        dict(what="panel b banks per pool, rows per bank, alpha, beta",
             value=f"{rows[0]['banks']}, {rows[0]['n']}, {rows[0]['alpha']}, {rows[0]['beta']}",
             source="runs/wbcp_signal/frozen/<pool>/" + EVALUATION + " (settings; designs[iid])"),
        dict(what="panel b 5% line: the failure budget 1 - beta", value="5%", source="beta = 0.95"),
        dict(what="panel b annotation: Q1 band width over min band width, range over the five pools",
             value=f"{100 * (lo_w - 1):.0f}-{100 * (hi_w - 1):.0f}% wider",
             source="designs[iid].signals.{q1,min}.half_width_mean"),
        dict(what="panel b annotation: banks failing for Q1's residual under the Q1 band, range over the five pools",
             value=f"{100 * lo_f:.1f}-{100 * hi_f:.1f}%", source="designs[iid].signals.q1.failure.q1.fail"),
        dict(what="panel b annotation: share of logged held-out rows with Q1 > Q2, range over the five pools",
             value=f"{100 * lo_a:.0f}-{100 * hi_a:.0f}% (" + ", ".join(
                 f"{r['label']} {100 * r['q1_above']:.1f}" for r in rows) + ")",
             source="runs/wbcp_signal/frozen/<pool>/" + EVALUATION + " head_disagreement.q1_above_q2; "
                    "CRITIC_ALIGNMENT.md table (46-54%)"),
        dict(what="notebook text: median |Q1 - Q2| / median |min residual|, range over the five pools",
             value=f"{min(r['gap_ratio'] for r in rows):.2f}-{max(r['gap_ratio'] for r in rows):.2f}",
             source="head_disagreement.gap_over_residual; CRITIC_ALIGNMENT.md (0.52-1.15)"),
        dict(what="notebook text: Q1's mean miscoverage under the min band; the min's own mean miscoverage",
             value=f"{100 * min(r['q1_mis_min_band'] for r in rows):.1f}-"
                   f"{100 * max(r['q1_mis_min_band'] for r in rows):.1f}%; "
                   f"{100 * min(r['min_mis_min_band'] for r in rows):.1f}-"
                   f"{100 * max(r['min_mis_min_band'] for r in rows):.1f}%",
             source="designs[iid].signals.min.failure.{q1,min}.mean_miscoverage"),
        dict(what="notebook text: Q1 band's failure for the min residual, per pool",
             value=", ".join(f"{r['label']} {100 * r['q1_band_min']:.1f}%" for r in rows),
             source="designs[iid].signals.q1.failure.min.fail; SIGNAL_STUDY.md step 2"),
    ]


def make():
    L = lq_slice()
    rows = pool_rates()
    facts = []
    fig, (ax_a, ax_b) = S.figure(1, 2, gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    _panel_a(ax_a, L, facts)
    _panel_b(ax_b, rows, facts)
    return fig, facts
