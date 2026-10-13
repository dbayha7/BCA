"""Charts for the new studies, in the deck's chart style (3328 x 1280 px, background #F7F6F2)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch

OUT = sys.argv[1]
BG = "#F7F6F2"; BLUE = "#2F6DB5"; ORANGE = "#D9822B"; NAVY = "#14213D"; GRAY = "#5F6672"; LGRAY = "#C9C6BC"
LBLUE = "#A9C4E6"; WATER = "#E3ECF5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": "#B9B6AC",
                     "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY, "axes.linewidth": 1.0})


def fig(ncols=1, **kw):
    f, ax = plt.subplots(1, ncols, figsize=(13.0, 5.0), dpi=256, facecolor=BG, **kw)
    for a in (ax if ncols > 1 else [ax]):
        a.set_facecolor(BG)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    return f, ax


def note(ax, text, x=0.995, y=0.98):
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="top", fontsize=12.5, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDFCF9", ec="#E3E1DA"))


def save(f, name):
    f.savefig(f"{OUT}/{name}.png", facecolor=BG, dpi=256)
    plt.close(f)


# 1. Gapped LQ world: R3 seeds passing per alpha
f, ax = fig()
alphas = ["alpha 2.5", "alpha 5", "alpha 10"]
series = [("First gate, base world (seeds 0-4)", [0, 1, 0], LBLUE),
          ("Amendment A1, base world (fresh seeds 5-9)", [0, 3, 2], BLUE),
          ("Amendment A1, thin-share version (seeds 5-9)", [0, 0, 0], NAVY)]
w = 0.26
for k, (lab, vals, col) in enumerate(series):
    xs = [i + (k - 1) * w for i in range(3)]
    ax.bar(xs, vals, w * 0.92, color=col, label=lab, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.08, str(v), ha="center", va="bottom", fontsize=15, fontweight="bold", color=col if col != LBLUE else BLUE)
ax.axhline(4, color=NAVY, ls="--", lw=1.6, zorder=2)
ax.text(2.45, 4.08, "needed: 4 of 5 seeds", ha="right", va="bottom", fontsize=13.5, color=NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(alphas, fontsize=15)
ax.set_ylim(0, 6.6); ax.set_yticks(range(6)); ax.set_ylabel("Seeds where the loss sits in thin data", fontsize=14)
ax.legend(loc="upper left", frameon=False, fontsize=13)
note(ax, "check R3 · 5 seeds per gate")
f.tight_layout(); save(f, "c1_glq")

# 2. ThrusterChain: the world and the stall
f = plt.figure(figsize=(13.0, 5.0), dpi=256, facecolor=BG)
a1 = f.add_axes([0.01, 0.05, 0.58, 0.9]); a1.set_facecolor(BG); a1.axis("off"); a1.set_xlim(0, 10); a1.set_ylim(0, 6)
a1.add_patch(FancyBboxPatch((0.2, 1.3), 9.6, 3.0, boxstyle="round,pad=0.02,rounding_size=0.25", fc=WATER, ec="none"))
xs = [1.2, 3.0, 4.8, 6.6, 8.4]
for i in range(4):
    x0, x1 = xs[i] + 0.38, xs[i + 1] - 0.38
    n = 9; pts = [(x0 + (x1 - x0) * j / n, 2.8 + (0.12 if j % 2 else -0.12) * (0 < j < n)) for j in range(n + 1)]
    a1.plot([p[0] for p in pts], [p[1] for p in pts], color=GRAY, lw=1.4)
    xm = (xs[i] + xs[i + 1]) / 2
    a1.add_patch(Polygon([(xm - 0.28, 3.55), (xm + 0.28, 3.55), (xm, 3.15)], closed=True, fc=BLUE, ec="none"))
    a1.text(xm, 3.75, f"T{i + 1}", ha="center", va="bottom", fontsize=14, color=BLUE, fontweight="bold")
for i, x in enumerate(xs):
    a1.add_patch(Circle((x, 2.8), 0.36, fc="#FDFCF9", ec=NAVY, lw=2))
    a1.text(x, 2.8, str(i + 1), ha="center", va="center", fontsize=14, color=NAVY, fontweight="bold")
a1.add_patch(FancyArrowPatch((0.6, 1.75), (9.4, 1.75), arrowstyle="-|>", mutation_scale=22, color="#4F7D96", lw=2.2))
a1.text(5.0, 1.45, "current c + wave w₁: the same push on every float", ha="center", va="top", fontsize=13.5, color="#4F7D96")
a1.text(5.0, 5.3, "Five floats on springs, held at rest by four thrusters", ha="center", fontsize=16, color=NAVY, fontweight="bold")
a1.text(5.0, 4.75, "thruster k pushes floats k and k+1 · 13 numbers observed · 4 throttles in [−1, 1]", ha="center", fontsize=12.5, color=GRAY)
a1.text(5.0, 0.6, "reward each step: −(squared distance from rest + 0.3 × squared throttles) · 100 steps", ha="center", fontsize=12.5, color=GRAY)
a2 = f.add_axes([0.66, 0.16, 0.31, 0.72]); a2.set_facecolor(BG)
for s in ("top", "right"):
    a2.spines[s].set_visible(False)
a2.axvspan(0.8, 1.0, color="#F6E2D4", zorder=0)
a2.text(0.9, 0.08, "almost no\ndata here", ha="center", va="bottom", fontsize=12, color=ORANGE)
a2.plot([0, 0.8, 1.0], [0, 0.8, 1.0], color=BLUE, lw=2.2, ls="--", label="no-stall world")
a2.plot([0, 0.8, 1.0], [0, 0.8, 0.5], color=ORANGE, lw=3, label="stalling world")
a2.axvline(0.8, color=GRAY, lw=1, ls=":")
a2.set_xlim(0, 1.0); a2.set_ylim(0, 1.05)
a2.set_xticks([0, 0.4, 0.8, 1.0]); a2.set_xticklabels(["0", "40%", "80%", "100%"])
a2.set_xlabel("throttle command", fontsize=13); a2.set_ylabel("push delivered", fontsize=13)
a2.legend(loc="upper left", frameon=False, fontsize=12)
a2.set_title("Past 80% a thruster stalls", fontsize=14.5, color=NAVY, fontweight="bold", loc="left")
save(f, "c2_env")

# 3. Harm by situation (check 2, alpha 5)
f, ax = fig()
sits = ["Recorded\nconditions", "Strong\ncurrent", "Big\nwaves", "Big\nkicks"]
full = [91.3, 84.6, 86.6, 64.3]; gap = [85.5, -14.3, 70.0, 52.5]; harm = [5.8, 98.9, 16.6, 11.8]
w = 0.36
ax.bar([i - w / 2 for i in range(4)], full, w * 0.94, color=BLUE, label="actor of the full-log critic", zorder=3)
ax.bar([i + w / 2 for i in range(4)], gap, w * 0.94, color=ORANGE, label="actor of the gapped-log critic", zorder=3)
for i in range(4):
    ax.text(i - w / 2, full[i] + 2, f"{full[i]:.1f}", ha="center", va="bottom", fontsize=13, color=BLUE, fontweight="bold")
    yv = gap[i] + 2 if gap[i] >= 0 else gap[i] - 2
    ax.text(i + w / 2, yv, f"{gap[i]:.1f}", ha="center", va="bottom" if gap[i] >= 0 else "top", fontsize=13, color=ORANGE, fontweight="bold")
    ax.text(i, 112, f"harm {harm[i]:.1f}", ha="center", fontsize=14, color=NAVY if i != 1 else ORANGE, fontweight="bold")
ax.axhline(0, color="#B9B6AC", lw=1)
ax.set_xticks(range(4)); ax.set_xticklabels(sits, fontsize=14)
ax.set_ylim(-30, 122); ax.set_ylabel("Points (0 = recording controller,\n100 = stall-aware controller)", fontsize=13)
ax.legend(loc="lower left", frameon=False, fontsize=13, bbox_to_anchor=(0.0, 0.02))
note(ax, "check 2 · alpha 5 · seeds 0-4", y=0.86)
f.tight_layout(); save(f, "c3_harm")

# 4. Room check (check 3): strong current
f, ax = fig()
arms = [("no extra caution", -10.1, LGRAY), ("caution placed by the exact harm", -7.6, BLUE),
        ("even, matched strength", -5.8, LGRAY), ("same total, spread evenly", -7.0, LGRAY),
        ("harm-placed, shuffled", -7.1, LGRAY), ("harm-placed, reversed", -6.4, LGRAY),
        ("harm-placed, factors up to 2", -5.6, LBLUE), ("even ×1.5", -4.5, LGRAY), ("even ×2", -0.7, LGRAY),
        ("even ×3", 7.1, LGRAY)]
ys = list(range(len(arms)))[::-1]
for y, (lab, v, col) in zip(ys, arms):
    ax.barh(y, v, 0.66, color=col, zorder=3)
    ax.text(v + (1.2 if v >= 0 else -1.2), y, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=12.5,
            color=BLUE if col == BLUE else NAVY, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in arms], fontsize=13)
ax.axvline(0, color="#B9B6AC", lw=1)
ax.axvline(84.6, color=BLUE, ls="--", lw=1.6)
ax.text(83.5, 9.4, "full-log critic's actor: 84.6", ha="right", va="center", fontsize=13, color=BLUE)
ax.set_xlim(-22, 92); ax.set_xlabel("Points in strong current", fontsize=13)
note(ax, "check 3 · BCA's channel: extra pull toward the recorded command", y=0.12)
f.tight_layout(); save(f, "c4_room")

# 5. The critic's belief past 80% (anatomy, training states in the full-throttle region)
f, ax = fig()
x = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
truth = [-0.23, -0.14, -0.06, -0.03, 0, -0.10, -0.21, -0.33, -0.46]
gapc = [-0.24, -0.14, -0.06, -0.03, 0, 0.02, 0.04, 0.05, 0.06]
fullc = [-0.28, -0.12, -0.00, 0.02, 0, -0.06, -0.17, -0.30, -0.46]
ax.axvspan(0.8, 1.0, color="#F6E2D4", zorder=0)
ax.text(0.9, -0.57, "past 80%: removed from the data", ha="center", fontsize=13, color=ORANGE)
ax.plot(x, truth, color=NAVY, lw=3, marker="o", label="truth (exact, by simulation)", zorder=4)
ax.plot(x, fullc, color=BLUE, lw=2.2, ls="--", marker="o", ms=5, label="critic trained on the full log", zorder=3)
ax.plot(x, gapc, color=ORANGE, lw=3, marker="o", label="critic trained on the gapped log", zorder=5)
ax.annotate("+0.06", (1.0, 0.06), xytext=(0.965, 0.10), fontsize=15, color=ORANGE, fontweight="bold")
ax.annotate("−0.46", (1.0, -0.46), xytext=(0.955, -0.52), fontsize=15, color=NAVY, fontweight="bold")
ax.axhline(0, color="#B9B6AC", lw=1)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])
ax.set_xlabel("thruster 1 throttle", fontsize=13); ax.set_ylabel("value relative to holding at 80%", fontsize=13)
ax.set_ylim(-0.6, 0.18); ax.set_xlim(0.48, 1.03)
ax.legend(loc="lower left", frameon=False, fontsize=13)
note(ax, "training states where the water pushes hardest · seeds 0-4", x=0.6, y=0.99)
f.tight_layout(); save(f, "c5_belief")

# 6. One belief causes the harm
f, ax = fig()
labs = ["gapped-log critic's actor", "same actor, commands\ncapped at 80% when run", "full-log critic's actor"]
vals = [-14.3, 101.8, 84.6]; cols = [ORANGE, BLUE, LBLUE]
ax.bar(range(3), vals, 0.55, color=cols, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 3 if v >= 0 else v - 3, f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=17,
            fontweight="bold", color=cols[i] if cols[i] != LBLUE else BLUE)
ax.axhline(0, color="#B9B6AC", lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=14)
ax.set_ylim(-32, 120); ax.set_ylabel("Points in strong current", fontsize=13)
note(ax, "past 80% at training states: gapped-critic actor 42% · full-log 10%", x=0.62, y=0.99)
f.tight_layout(); save(f, "c6_onebelief")

# 7. Stage 1: penalising the value the actor climbs (strong current)
f, ax = fig()
arms = [("no extra caution", -14.3, LGRAY), ("exact error penalty at the actor's command", 79.7, BLUE),
        ("  same, half strength", 75.5, LBLUE), ("  exact long-run harm as penalty", 82.2, LBLUE),
        ("  right error, from a different state", -0.3, LGRAY), ("  wrong side of the error", -17.0, LGRAY),
        ("even extra pull, matched strength", -12.7, LGRAY), ("push less everywhere, matched", -9.7, LGRAY),
        ("random penalty surface 1, matched", -19.4, LGRAY), ("random penalty surface 2, matched", -14.5, LGRAY),
        ("clip to the recorded range", 93.1, "#8FA3BF")]
ys = list(range(len(arms)))[::-1]
for y, (lab, v, col) in zip(ys, arms):
    ax.barh(y, v, 0.66, color=col, zorder=3)
    ax.text(v + (1.5 if v >= 0 else -1.5), y, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=12,
            color=BLUE if col in (BLUE, LBLUE) else NAVY, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in arms], fontsize=12.5)
ax.axvline(0, color="#B9B6AC", lw=1)
ax.axvline(-14.3 + 49.5, color=NAVY, ls="--", lw=1.4)
ax.text(-14.3 + 49.5 + 1, 10.45, "mark: recover half the harm", fontsize=12, color=NAVY, va="center")
ax.set_xlim(-30, 108); ax.set_xlabel("Points in strong current (mean of seeds 0-4)", fontsize=13)
f.tight_layout(); save(f, "c7_stage1")

# 8. Rivals: gain where the missing commands hurt vs cost where they help
f, (a, b) = fig(2, sharey=True)
names = ["perfect signal\n(exact error)", "range hinge", "classifier ratio", "distance to data", "behaviour density", "two-head disagreement"]
gain = [93.9, 84.6, 36.8, 9.3, 4.8, 1.2]; cost = [1.3, 20.4, 11.0, 5.5, 7.3, 4.0]
cols = [BLUE, "#8FA3BF", LGRAY, LGRAY, LGRAY, LGRAY]
ys = list(range(6))[::-1]
a.barh(ys, gain, 0.62, color=cols, zorder=3); b.barh(ys, cost, 0.62, color=[ORANGE] + ["#E8B48A"] * 5, zorder=3)
for y, g, c in zip(ys, gain, cost):
    a.text(g + 1.5, y, f"{g:.1f}", va="center", fontsize=13, fontweight="bold", color=NAVY)
    b.text(c + 0.5, y, f"{c:.1f}", va="center", fontsize=13, fontweight="bold", color=NAVY)
a.set_yticks(ys); a.set_yticklabels(names, fontsize=13)
a.set_xlim(0, 110); b.set_xlim(0, 26)
a.set_title("points won back in strong current\n(stalling world: the missing commands hurt)", fontsize=13.5, color=NAVY, loc="left")
b.set_title("points lost in strong current\n(no-stall world: the missing commands help)", fontsize=13.5, color=NAVY, loc="left")
f.tight_layout(); save(f, "c8_rivals")

# 9. Trace test bed
f, (a, b) = fig(2)
share = [0, 0.5, 1, 2, 3, 5, 10]
harm = [99.5, 96.5, 92.6, 85.8, 74.9, 71.5, 23.4]
sel = [0.025, 0.076, 0.065, 0.071, -0.010, 0.050, -0.046]
xs = range(len(share)); lbl = ["0%", "0.5%", "1%", "2%", "3%", "5%", "10%"]
a.bar(xs, harm, 0.6, color=[BLUE if h >= 50 else LGRAY for h in harm], zorder=3)
a.axhline(50, color=NAVY, ls="--", lw=1.4); a.text(6.4, 52, "mark 50", ha="right", fontsize=12, color=NAVY)
for x, h in zip(xs, harm):
    a.text(x, h + 2, f"{h:.0f}", ha="center", fontsize=12.5, fontweight="bold", color=NAVY)
a.set_xticks(list(xs)); a.set_xticklabels(lbl); a.set_ylim(0, 115)
a.set_title("harm in strong current (points)", fontsize=13.5, color=NAVY, loc="left")
a.set_xlabel("share of full-throttle rows kept in the data", fontsize=12.5)
b.bar(xs, sel, 0.6, color=[ORANGE] * 7, zorder=3)
b.axhline(0.2, color=NAVY, ls="--", lw=1.4); b.text(6.4, 0.21, "mark 0.2", ha="right", fontsize=12, color=NAVY)
b.axhline(0, color="#B9B6AC", lw=1)
for x, s in zip(xs, sel):
    b.text(x, s + (0.008 if s >= 0 else -0.008), f"{s:.2f}", ha="center", va="bottom" if s >= 0 else "top", fontsize=12, color=NAVY)
b.set_xticks(list(xs)); b.set_xticklabels(lbl); b.set_ylim(-0.1, 0.26)
b.set_title("can a data-only signal see the trace? (separation)", fontsize=13.5, color=NAVY, loc="left")
b.set_xlabel("share of full-throttle rows kept in the data", fontsize=12.5)
f.tight_layout(); save(f, "c9_trace")
print("charts written")

# 10. Stage 2 detection, uniform version: ranking of harmful above harmless pairs
f, (a, b) = fig(2, sharey=True)
names = ["BCA's band\n(uniform)", "classifier ratio", "behaviour density", "distance to data", "two-head\ndisagreement",
         "residual evidence"]
t12 = [0.804, 0.990, 0.871, 0.774, 0.636, 0.395]
t3 = [0.617, 0.901, 0.804, 0.683, 0.606, 0.467]
cols = [ORANGE] + [LGRAY] * 5
for ax, vals, mark, ttl in ((a, t12, 0.75, "training states (sets T1 + T2)"), (b, t3, 0.70, "strong-current states (set T3)")):
    ys = list(range(6))[::-1]
    ax.barh(ys, vals, 0.62, color=cols, zorder=3)
    for y, v in zip(ys, vals):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.axvline(mark, color=NAVY, ls="--", lw=1.4); ax.text(mark + 0.005, 5.45, f"mark {mark:.2f}", fontsize=12, color=NAVY)
    ax.axvline(0.5, color=GRAY, ls=":", lw=1.2); ax.text(0.505, -0.75, "chance", fontsize=11.5, color=GRAY)
    ax.set_xlim(0.3, 1.08); ax.set_title(ttl, fontsize=13.5, color=NAVY, loc="left")
    ax.set_xlabel("chance a harmful command scores above a harmless one", fontsize=12)
a.set_yticks(list(range(6))[::-1]); a.set_yticklabels(names, fontsize=13)
f.tight_layout(); save(f, "c10_detect")

# 11. Stage 2 detection, both versions against the data-only rivals
f, (a, b) = fig(2, sharey=True)
names = ["BCA's band,\nweighted", "BCA's band,\nuniform", "classifier ratio", "behaviour density", "distance to data",
         "two-head\ndisagreement"]
t12 = [0.989, 0.804, 0.990, 0.871, 0.774, 0.636]
t3 = [0.860, 0.617, 0.901, 0.804, 0.683, 0.606]
cols = [BLUE, ORANGE, "#8FA3BF", LGRAY, LGRAY, LGRAY]
for ax, vals, mark, ttl in ((a, t12, 0.75, "training states"), (b, t3, 0.70, "strong-current states")):
    ys = list(range(6))[::-1]
    ax.barh(ys, vals, 0.62, color=cols, zorder=3)
    for y, v in zip(ys, vals):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.axvline(mark, color=NAVY, ls="--", lw=1.4); ax.text(mark + 0.005, 5.45, f"mark {mark:.2f}", fontsize=12, color=NAVY)
    ax.set_xlim(0.5, 1.08); ax.set_title(ttl, fontsize=13.5, color=NAVY, loc="left")
    ax.set_xlabel("chance a harmful command scores above a harmless one", fontsize=12)
a.set_yticks(list(range(6))[::-1]); a.set_yticklabels(names, fontsize=13)
f.tight_layout(); save(f, "c11_detect_both")
