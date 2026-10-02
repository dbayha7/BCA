"""From calibrated width to BC dose, and why the same mean dose is not the same strength.

Panel a: BCA's dose map m = 1 + b * U / (U + u) (calibration/dose.py, level_critic_dose l.53-60, applied by
frozen_level_dose l.65-94 with U = R * u * max(eta, 1e-6), l.75-76), drawn against U/u = R * eta with b = 0.5
(configs/td3_bc.yaml l.20; also cql.yaml and rebrac.yaml). Beside it, the real per-row doses of the five healthy
TD3+BC pools of the signal study's step 2 (runs/wbcp_signal/frozen/<pool>/heads.npz, the Q1 signal's eta and unit),
at each pool's mean deployed WBCP threshold over its 4,000 independent-row banks (evaluation_b4000_s2026100201.json,
designs[iid].signals.q1.threshold_mean). The dose is recomputed here with a float32 NumPy port of dose.py; the port
is checked against the JSON's exact dose statistics at lambda*, and the per-row mean and SD it gives match the
step-2 table of experiments/signal/SIGNAL_STUDY.md (l.217-223) to within 2e-4.

Panel b (post hoc, not pre-registered): in the step-3 linear-quadratic harness, the BC term's gradient norm at the
starting actor relative to a constant dose with the same mean, for the oracle dose and for BCA's dose (Q1 signal;
each bar spans every replicate of every kappa/offset cell of the case, 15 values for independent errors and 45 for
the others; the marker is the median), from runs/wbcp_signal/strength_reanalysis.json. The right column gives the
oracle's change in true value J against the constant dose (Q1 signal, mean over the case's cells), from
runs/wbcp_signal/lq/results.json. The strength-only model's correlation (0.85 over 216 cells) is read from the same
JSON and recomputed here from lq/results.json with experiments/signal/strength_reanalysis.py's formula (l.49-54).
"""

import json
import math
import statistics

import numpy as np
import yaml

from experiments.wbcp.viz import style as S

FROZEN = S.ROOT / "runs" / "wbcp_signal" / "frozen"
EVAL = "evaluation_b4000_s2026100201.json"
POOLS = [("hopper-medium-v2", "hopper"), ("walker2d", "walker2d"), ("halfcheetah", "halfcheetah"),
         ("maze2d", "maze2d"), ("pen-expert", "pen-expert")]
STRENGTH = S.ROOT / "runs" / "wbcp_signal" / "strength_reanalysis.json"
LQ = S.ROOT / "runs" / "wbcp_signal" / "lq" / "results.json"
CASES = [("independent_errors", "independent errors"), ("shared_bias", "shared bias"),
         ("q1_optimistic", "Q1 optimistic")]


def bc_dose(eta, unit, threshold, blend):
    """NumPy float32 port of calibration/dose.py for a ready reference with finite, positive inputs.

    frozen_level_dose l.75-76: U = R * (max(eta, 1e-6) * u); level_critic_dose l.54-60: with normalizer
    max(U, u) (the 0.25 rescale only applies above float32 max / 4), m = 1 + b * U' / (U' + u')."""
    f = np.float32
    eta = np.asarray(eta, f)
    unit, threshold, blend = f(unit), f(threshold), f(blend)
    width = threshold * (np.maximum(eta, f(1e-6)) * unit)
    normalizer = np.maximum(width, unit)
    u_norm, c_norm = width / normalizer, unit / normalizer
    return (f(1.0) + blend * (u_norm / (u_norm + c_norm))).astype(np.float64)


def load_pools(blend):
    pools = []
    for directory, name in POOLS:
        with open(FROZEN / directory / EVAL, encoding="utf-8") as handle:
            ev = json.load(handle)
        if ev["settings"]["blend"] != blend:
            raise ValueError(f"{directory}: evaluation blend {ev['settings']['blend']} != config {blend}")
        heads = np.load(FROZEN / directory / "heads.npz")
        signals = [str(s) for s in heads["signals"]]
        eta = np.asarray(heads["eta_q1"], np.float32)
        unit = np.float32(heads["unit"][signals.index("q1")])
        # check the port against the evaluation's exact statistics at lambda* (frozen_signals.dose_tables)
        star = ev["population"]["lambda_star"]["q1"]
        at_star = bc_dose(eta, unit, star, blend)
        ref = ev["doses"]["q1"]["lambda_star"]
        if abs(at_star.mean() - ref["mean"]) > 1e-6 or abs(at_star.std() - ref["sd"]) > 1e-6:
            raise AssertionError(f"{directory}: dose port does not reproduce the evaluation's lambda* statistics")
        iid = next(d for d in ev["designs"] if d["design"]["name"] == "iid")
        r_bar = iid["signals"]["q1"]["threshold_mean"]
        dose = bc_dose(eta, unit, r_bar, blend)
        x = r_bar * np.maximum(eta.astype(np.float64), 1e-6)
        banks = ev["doses"]["q1"]["banks"]["iid"]
        pools.append(dict(directory=directory, name=name, rows=int(eta.size), r_bar=r_bar, dose=dose, x=x,
                          json_mean=banks["mean"], json_sd=banks["sd"], star_mean=ref["mean"]))
    return pools


def load_strength():
    with open(STRENGTH, encoding="utf-8") as handle:
        strength = json.load(handle)
    with open(LQ, encoding="utf-8") as handle:
        lq = json.load(handle)
    # oracle - constant change in J, Q1 signal, mean over the case's cells
    gap = {}
    for r in lq["results"]:
        if r["signal"] == "q1":
            m = r["metrics"]
            gap.setdefault((r["case"], r["behavior"]), []).append(
                m["J_change_oracle"]["mean"] - m["J_change_constant"]["mean"])
    gap = {k: statistics.mean(v) for k, v in gap.items()}
    # strength-only model, as experiments/signal/strength_reanalysis.py l.49-54
    model, observed = [], []
    for r in lq["results"]:
        m = r["metrics"]
        const, bca = m["bc_term_grad_norm_constant"]["values"], m["bc_term_grad_norm_bca"]["values"]
        jc, jn, d = m["J_change_constant"]["mean"], m["J_change_none"]["mean"], m["dose_mean"]["mean"]
        ratio = statistics.mean(a / c for c, a in zip(const, bca) if c)
        if d > 1:
            model.append((jc - jn) * (ratio - 1) * d / (d - 1))
            observed.append(m["J_gain_bca_minus_constant"]["mean"])
    mx, my = statistics.mean(model), statistics.mean(observed)
    pearson = sum((a - mx) * (b - my) for a, b in zip(model, observed)) / math.sqrt(
        sum((a - mx) ** 2 for a in model) * sum((b - my) ** 2 for b in observed))
    saved = strength["strength_only_model"]
    if len(model) != saved["cells"] or abs(pearson - saved["pearson"]) > 1e-9:
        raise AssertionError("strength-only model does not reproduce strength_reanalysis.json")
    return strength["bc_gradient_ratio"], gap, saved


def _signed(v, digits):
    return ("+" if v >= 0 else "−") + f"{abs(v):.{digits}f}"


def make():
    S.setup()
    with open(S.ROOT / "configs" / "td3_bc.yaml", encoding="utf-8") as handle:
        blend = float(yaml.safe_load(handle)["bca"]["blend"])
    pools = load_pools(blend)
    ratios, gap, model = load_strength()
    facts = []

    def fact(what, value, source):
        facts.append(dict(what=what, value=value, source=source))

    fig, axes = S.figure(1, 4, gridspec_kw=dict(width_ratios=[4.4, 2.0, 0.2, 6.0]))
    ax, axm, spacer, axb = axes
    spacer.axis("off")

    # ---- a: the dose map ----
    xmax, ylo, yhi = 8.0, 0.97, 1.56
    xs = np.linspace(0.0, xmax, 400)
    ax.plot(xs, 1.0 + blend * xs / (1.0 + xs), color=S.BLUE, lw=3.5, zorder=3, solid_capstyle="round")
    ax.axhline(1.0 + blend, color=S.INK, lw=1.8, ls=(0, (6, 4)), zorder=2)
    ax.axhline(1.0, color=S.INK, lw=1.8, ls=(0, (1.5, 3)), zorder=2)
    ax.text(0.15, 1.0 + blend + 0.012, f"cap 1 + b = {1 + blend:g} (b = {blend:g}), never reached", color=S.INK, fontsize=S.SMALL,
            va="bottom", ha="left")
    ax.text(xmax - 0.1, 1.0 + 0.012, "1 = host's own BC weight", color=S.INK, fontsize=S.SMALL, va="bottom",
            ha="right")
    ax.text(xmax - 0.15, 1.098, r"$m = 1 + b\,\dfrac{U}{U+u}$", color=S.BLUE, fontsize=S.BASE + 2, ha="right",
            va="center")
    # where the rows sit: each pool's middle 90% of U/u, unioned, mapped through the curve to the margin
    lo = min(np.quantile(p["x"], 0.05) for p in pools)
    hi = max(np.quantile(p["x"], 0.95) for p in pools)
    d_lo, d_hi = 1.0 + blend * lo / (1.0 + lo), 1.0 + blend * hi / (1.0 + hi)
    band = np.linspace(lo, hi, 50)
    ax.fill_between(band, ylo, 1.0 + blend * band / (1.0 + band), color=S.BLUE_LIGHT, alpha=0.45, lw=0, zorder=1)
    ax.fill_between([hi, xmax], d_lo, d_hi, color=S.BLUE_LIGHT, alpha=0.30, lw=0, zorder=1)
    ax.text((lo + hi) / 2, 1.04, "middle 90%\nof rows", color=S.INK, fontsize=S.SMALL, ha="center", va="bottom",
            zorder=4)
    ax.text(xmax - 0.15, (d_lo + d_hi) / 2, f"their doses\n{d_lo:.2f}–{d_hi:.2f}", color=S.INK,
            fontsize=S.SMALL, ha="right", va="center", zorder=4)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"calibrated width in residual units,  $U/u = R\,\eta(s,a)$")
    ax.set_ylabel("BC dose m  (multiplies TD3+BC's BC term)")
    S.panel_label(ax, "a  Width → dose: real rows land near 1.3")
    fact("blend b (BC dose cap is 1 + b)", f"{blend:g} (cap {1 + blend:g})",
         "configs/td3_bc.yaml l.20 (also configs/cql.yaml l.38, configs/rebrac.yaml l.21); ALGORITHMS.md l.337, l.352")
    fact("dose map", "m = 1 + b*U/(U+u), U = R*u*max(eta,1e-6)",
         "calibration/dose.py l.53-60 (level_critic_dose), l.75-76 (frozen_level_dose); ALGORITHMS.md l.337")
    fact("U/u range holding the middle 90% of rows of every pool (union of per-pool 5th-95th percentiles)",
         f"{lo:.2f} to {hi:.2f} (dose {d_lo:.3f} to {d_hi:.3f})",
         "computed from runs/wbcp_signal/frozen/<pool>/heads.npz eta_q1 at the mean iid-bank threshold")

    # ---- a, margin: per-row dose per pool ----
    bins = np.arange(1.0, 1.5 + 1e-9, 0.004)
    for i, p in enumerate(pools):
        lo_p, hi_p = np.quantile(p["dose"], [0.005, 0.995])
        counts, edges = np.histogram(p["dose"], bins=bins)
        mids = (edges[:-1] + edges[1:]) / 2
        keep = (mids >= lo_p) & (mids <= hi_p)
        half = 0.42 * counts[keep] / counts.max()
        axm.fill_betweenx(mids[keep], i - half, i + half, color=S.BLUE, alpha=0.85, lw=0)
        axm.plot([i - 0.42, i + 0.42], [p["dose"].mean()] * 2, color=S.INK, lw=2.0)
        axm.text(i, 1.0 + 0.012, p["name"], rotation=90, ha="center", va="bottom", fontsize=S.SMALL, color=S.SOFT)
        fact(f"{p['name']}: mean deployed WBCP threshold R over 4,000 iid banks (Q1 signal)", f"{p['r_bar']:.4f}",
             f"runs/wbcp_signal/frozen/{p['directory']}/{EVAL} designs[iid].signals.q1.threshold_mean")
        fact(f"{p['name']}: per-row BC dose at that threshold, mean / SD ({p['rows']:,} rows)",
             f"{p['dose'].mean():.4f} / {p['dose'].std():.4f} (step-2 bank average: {p['json_mean']:.4f} / "
             f"{p['json_sd']:.4f})",
             f"computed with the dose.py port from heads.npz; check: runs/wbcp_signal/frozen/{p['directory']}/{EVAL} "
             "doses.q1.banks.iid; experiments/signal/SIGNAL_STUDY.md l.219-223")
    fact("per-pool violins: per-row dose histogram (bin 0.004), drawn between the pool's 0.5th and 99.5th percentiles; "
         "black tick = pool mean", "shape only", "computed from heads.npz with the dose.py port")
    means = [p["dose"].mean() for p in pools]
    sds = [p["dose"].std() for p in pools]
    axm.set_xlim(-0.6, len(pools) - 0.4)
    axm.set_ylim(ylo, yhi)
    axm.set_xticks([])
    axm.tick_params(axis="y", labelleft=False, length=0)
    axm.spines["left"].set_color(S.RULE)
    axm.spines["bottom"].set_visible(False)
    axm.text(0.5, 1.0 + blend - 0.012, f"mean {min(means):.2f}–{max(means):.2f}\nrow SD {min(sds):.2f}–"
             f"{max(sds):.2f}", transform=axm.get_yaxis_transform(), ha="center", va="top", fontsize=S.SMALL,
             color=S.INK)
    axm.set_xlabel("5 pools, Q1 signal")
    fact("pools shown", "5 healthy TD3+BC pools: hopper, walker2d, halfcheetah, maze2d, pen-expert (pen-cloned and "
         "pen-human excluded: critic not converged)", "experiments/signal/SIGNAL_STUDY.md l.32-33")
    fact("dose 1 = the host's own BC weight (no extra BC)", "1", "ALGORITHMS.md l.334-339 (multiplier is 1 when the "
         "reference is not ready or a row is unusable); calibration/dose.py l.60 with ratio 0")
    fact("per-row dose across the five pools: range of means / of row SDs", f"{min(means):.2f}-{max(means):.2f} / "
         f"{min(sds):.2f}-{max(sds):.2f}", "computed (above); SIGNAL_STUDY.md l.219-223 gives 1.262-1.327 and "
         "0.017-0.050")

    # ---- b: same mean, different strength ----
    rows, y = [], 0.0
    for key, label in CASES:
        header_y = y + 0.95
        axb.text(0.82, header_y, label, fontsize=S.BASE, color=S.INK, ha="left", va="center", fontweight="semibold",
                 bbox=dict(facecolor=S.PAPER, edgecolor="none", pad=2.0), zorder=5)
        for behaviour in ("good", "poor"):
            rows.append((key, behaviour, y))
            y -= 1.0
        y -= 1.1
    off = 0.2
    for key, behaviour, yy in rows:
        r = ratios[f"{key}/{behaviour}"]
        o, b = r["oracle_over_constant"], r["bca_over_constant"]
        axb.plot([o["min"], o["max"]], [yy + off] * 2, color=S.ORANGE, lw=7, solid_capstyle="round", zorder=3)
        axb.plot(o["median"], yy + off, "o", ms=11, mfc="white", mec=S.ORANGE, mew=2.5, zorder=4)
        axb.plot([b["min"], b["max"]], [yy - off] * 2, color=S.BLUE, lw=7, solid_capstyle="round", zorder=3)
        axb.plot(b["median"], yy - off, "D", ms=9, mfc="white", mec=S.BLUE, mew=2.5, zorder=4)
        g = gap[(key, behaviour)]
        axb.text(3.95, yy, _signed(g, 2 if abs(g) >= 0.1 else 3), ha="right", va="center", fontsize=S.SMALL,
                 color=S.ORANGE if abs(g) >= 0.1 else S.SOFT, fontweight="semibold" if abs(g) >= 0.1 else "normal")
        fact(f"{key}, {behaviour} data: BC-gradient norm / constant dose's, oracle dose (min / median / max over "
             "every replicate of every kappa/offset cell, Q1 signal)",
             f"{o['min']:.2f} / {o['median']:.2f} / {o['max']:.2f}",
             f"runs/wbcp_signal/strength_reanalysis.json bc_gradient_ratio['{key}/{behaviour}'].oracle_over_constant;"
             " SIGNAL_STUDY.md l.398-404")
        fact(f"{key}, {behaviour} data: BC-gradient norm / constant dose's, BCA dose (min / median / max)",
             f"{b['min']:.3f} / {b['median']:.3f} / {b['max']:.3f}",
             f"runs/wbcp_signal/strength_reanalysis.json bc_gradient_ratio['{key}/{behaviour}'].bca_over_constant")
        fact(f"{key}, {behaviour} data: oracle - constant change in J (Q1 signal, mean over cells)", f"{g:+.4f}",
             "computed from runs/wbcp_signal/lq/results.json (J_change_oracle - J_change_constant); "
             "SIGNAL_STUDY.md l.312-318 (mean over all three signals)")
    top = rows[0][2]
    axb.axvline(1.0, color=S.INK, lw=1.8, ls=(0, (6, 4)), zorder=2)
    axb.text(1.03, top + 1.95, "constant dose,\nsame mean", color=S.INK, fontsize=S.SMALL, ha="left", va="center")
    axb.text(3.95, top + 1.95, "oracle − constant ΔJ\n(true J, cell mean)", color=S.SOFT, fontsize=S.SMALL,
             ha="right", va="center")
    first = ratios[f"{rows[0][0]}/{rows[0][1]}"]
    axb.text(first["oracle_over_constant"]["min"] - 0.08, top + off, "oracle dose", color=S.ORANGE,
             fontsize=S.SMALL, ha="right", va="center", fontweight="semibold")
    axb.text(first["bca_over_constant"]["max"] + 0.08, top - off, "BCA dose", color=S.BLUE, fontsize=S.SMALL,
             ha="left", va="center", fontweight="semibold")
    last = rows[-1][2]
    axb.text(0.82, last - 1.35, "BCA: only 1.00–1.03× as strong. A strength-only model\n"
             f"predicts its small ΔJ vs constant (r = {model['pearson']:.2f}, {model['cells']} cells)",
             color=S.BLUE, fontsize=S.SMALL, ha="left", va="center",
             bbox=dict(facecolor=S.PAPER, edgecolor="none", pad=2.0), zorder=5)
    axb.set_yticks([yy for _, _, yy in rows])
    axb.set_yticklabels([f"{beh} data" for _, beh, _ in rows])
    axb.tick_params(axis="y", length=0)
    axb.spines["left"].set_visible(False)
    axb.set_xlim(0.8, 4.0)
    axb.set_xticks([1, 1.5, 2, 2.5, 3])
    axb.set_xticklabels(["1×", "1.5×", "2×", "2.5×", "3×"])
    axb.set_ylim(last - 2.1, top + 2.65)
    axb.set_xlabel("BC-term gradient norm ÷ constant dose's (LQ harness)")
    S.panel_label(axb, "b  Same mean dose, not the same strength (post hoc)")
    bca_all = [ratios[k]["bca_over_constant"][q] for k in ratios for q in ("min", "max")]
    fact("BCA dose: BC-gradient ratio to the constant dose, all 12 case x behaviour groups (min-max)",
         f"{min(bca_all):.3f}-{max(bca_all):.3f} (shown as 1.00-1.03)",
         "runs/wbcp_signal/strength_reanalysis.json bc_gradient_ratio[*].bca_over_constant")
    fact("strength-only model: Pearson correlation with observed BCA - constant (cell means, all three signals, "
         "all six cases)", f"{model['pearson']:.4f} over {model['cells']} cells", "runs/wbcp_signal/strength_reanalysis.json strength_only_model (recomputed here "
         "from runs/wbcp_signal/lq/results.json); SIGNAL_STUDY.md l.409-411")
    return fig, facts
