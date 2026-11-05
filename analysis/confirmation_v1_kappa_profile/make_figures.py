"""Figures for the confirmation v1 attempt 3 readout (knee_s10_candidate_v2), drawn from the saved decision.json only.

Readout: experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1/CF_READOUT_attempt3_20261102.md
Usage:   python analysis/confirmation_v1_kappa_profile/make_figures.py [attempt_3 dir] [out dir]
Made with matplotlib 3.9.2 and DejaVu Sans. Reads one file; trains, evaluates and simulates nothing.
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, 'experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1/cf_run/attempt_3')
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
d = json.load(open(os.path.join(SRC, 'decision.json')))
P, W = d['paired'], d['worlds']
DOSES, SEEDS = ['21', '63', '189'], ['0', '1', '2']
# design-system colors (tokens.json, light theme)
INK, INK2, MUTED, RULE = '#1a201e', '#2b3330', '#4f5853', '#d6dad2'
COL = {'stall3': '#9f2818', 'nostall': '#7e53cc'}           # k-stall, k-rise
NAME = {'stall3': 'Stalling world (stall3): the withheld region is a trap',
        'nostall': 'No-stall world: the withheld region is harmless'}
MARK = {'0': 'o', '1': 's', '2': '^'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def fmt(v):
    return ('−' if v < 0 else '+') + f'{abs(v):.1f}'


# ---------------------------------------------------------------- Figure 1: dJ by dose, seed and world
def spread(ys, gap):
    """label positions: the values themselves, pushed apart to at least `gap` (data units), order kept"""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for j in range(1, len(pos)):
        if pos[j] - pos[j - 1] < gap:
            pos[j] = pos[j - 1] + gap
    out = [0] * len(ys)
    for j, i in enumerate(order):
        out[i] = pos[j]
    return out


fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), constrained_layout=True)
for ax, world in zip(axes, ['stall3', 'nostall']):
    nat = [W[world]['native'][s]['J'] for s in SEEDS]
    ax.axhline(0, color=INK, lw=1)
    vals = {s: [P[world][k][s]['dJ'] for k in DOSES] for s in SEEDS}
    lo = min(min(v) for v in vals.values()); hi = max(max(v) for v in vals.values())
    gap = (hi - lo) * 0.055
    for s in SEEDS:
        ax.plot(range(3), vals[s], color=COL[world], lw=1.6, alpha=.55)
        ax.scatter(range(3), vals[s], marker=MARK[s], s=70, color=COL[world], edgecolor='white', linewidth=1.2, zorder=3)
    for x in range(3):
        ys = [vals[s][x] for s in SEEDS]
        for s, y, ly in zip(SEEDS, ys, spread(ys, gap)):
            txt = fmt(y) + (f'   seed {s}' if x == 2 else '')
            ax.annotate(txt, (x, y), xytext=(x + 0.09, ly), textcoords='data', va='center', fontsize=11,
                        color=COL[world] if x == 2 else INK2)
    bar = 2.0 if world == 'stall3' else -2.0
    ax.plot([0.8, 1.2], [bar, bar], color=INK2, lw=2, solid_capstyle='butt', zorder=4)
    ax.set_xticks(range(3), ['\u03ba = 21', '\u03ba = 63\nconfirmed dose\n(bar ' + fmt(bar) + ' in every seed)', '\u03ba = 189'])
    ax.set_xlim(-0.35, 2.95)
    ax.set_ylabel('dJ: BCA return minus native return\n(same fork, paired; higher is better for BCA)')
    ax.set_title(NAME[world] + '\nnative J: ' + ', '.join(fmt(v).lstrip('+') for v in nat) + ' (seeds 0, 1, 2)',
                 fontsize=12.5, color=INK, loc='left')
    ax.grid(axis='y', color=RULE, lw=.8)
fig.suptitle('Return change from BCA, every dose and fresh seed (confirmation v1, attempt 3)', fontsize=15, color=INK, x=0.01, ha='left')
fig.savefig(os.path.join(OUT, 'fig1_dJ_by_dose.png'), dpi=150)

# ---------------------------------------------------------------- Figure 2: how far the actor leaves W vs dJ
fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), constrained_layout=True)
for ax, world in zip(axes, ['stall3', 'nostall']):
    ax.axhline(0, color=INK, lw=1)
    for k in DOSES:
        for s in SEEDS:
            x, y = P[world][k][s]['dEXPO'], P[world][k][s]['dJ']
            ax.scatter(x, y, marker=MARK[s], s=40 + int(k) * 0.9, color=COL[world], alpha=.85, edgecolor='white', linewidth=1.2, zorder=3)
            ax.annotate(f'κ{k} s{s}', (x, y), xytext=(7, 4), textcoords='offset points', fontsize=10.5, color=INK2)
    ax.set_xlabel('dEXPO: change in the share of the 12,800 steps\nwith a command in W (BCA minus native;\nfurther left = BCA leaves W more)')
    ax.set_ylabel('dJ (BCA minus native return)')
    ax.set_title(NAME[world], fontsize=12.5, color=INK, loc='left')
    ax.grid(color=RULE, lw=.8)
    ax.set_xlim(-0.8, 0.02)
fig.suptitle('How far BCA moved the actor out of W, against its return change (one point per dose and seed; size grows with \u03ba)',
             fontsize=14, color=INK, x=0.01, ha='left')
fig.savefig(os.path.join(OUT, 'fig2_exposure_vs_dJ.png'), dpi=150)
print('ok')
