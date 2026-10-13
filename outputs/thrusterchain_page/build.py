"""Build thrusterchain.html from template.html and page_data.json.

Every number on the page comes from page_data.json (written by make_data.py from the environment's own code),
worldcard_data.json (add_worldcard.py, from the world card's run before the pre-registration) or sources_data.json
(sources.py: literal numbers read from their source files):
  {{path/to/value#fmt}}  in the template is replaced by the value at that path (keys separated by '/'), formatted:
      f<d>  fixed decimals, Unicode minus      t<d>  fixed decimals, ASCII minus (inside TeX)
      p<d>  percent with d decimals            i     integer with thousands separators
      s     plain string                       d<d>  fixed decimals with a sign (+ for positive values)
  <!--@NAME@-->  is replaced by a fragment (table or SVG chart) built below from the same data.
  /*__DATA__*/null  is replaced by the live panel's data.
The build refuses to write if any placeholder is left or a path is missing.
"""

import html
import json
import math
import re
import sys

D = json.load(open("page_data.json"))
# The world card that ran on design seeds 900-904 before the pre-registration (add_worldcard.py reads it, read only).
D["world_card"] = json.load(open("worldcard_data.json"))
# Literal numbers read from their source files (sources.py reads them, read only): the coupling seed, the far rows,
# the 3-state toy world, the design's own measurement of the critic's pull and the design's odds for the gate.
D["sources"] = json.load(open("sources_data.json"))
D["sources"]["design_odds"] = {k: v.replace("-", "\u2013") if k != "source" else v
                               for k, v in D["sources"]["design_odds"].items()}  # 25-30% shown with an en dash
# The task overview's numbers (make_overview_data.py computes them from the environment's own code, read only).
D["overview"] = json.load(open("overview_data.json"))
# After the stop (EXPLORATORY, outside the gate, no BCA run): Stage 1's room test, the deployment-time diagnostic and
# the anatomy of the harm (make_beyond_data.py reads the committed result files, read only).
D["beyond"] = json.load(open("beyond_data.json"))
# The learned actors' and the room check's arms' points, for the controller guide and the controller tooltips
# (make_controllers_data.py reads the gate's committed result files, read only).
D["controllers"] = json.load(open("controllers_data.json"))
# the room check's margins: caution placed at the harm minus each blind control, per situation (the mean over seeds of
# the paired differences, which equals the difference of the means)
_rp = D["controllers"]["room"]["points"]
D["controllers"]["room"]["over"] = {a: {s: _rp["harm_placed"][s]["mean"] - _rp[a][s]["mean"] for s in _rp[a]}
                                    for a in ("strength_matched_even", "amount_matched", "shuffled")}
MINUS = "\u2212"
# numbers that column descriptions (tooltips) quote, as placeholders filled from the data like the rest of the page
TRIG = "{{recording_controller/trigger#f3}}"
STALL90 = "{{environment/thrust_curve/key_points/8/thrust_stall#f2}}"
STALL100 = "{{environment/thrust_curve/key_points/10/thrust_stall#f1}}"


# ------------------------------------------------------------------------------------------------ formatting

def fx(x, d=2, tex=False):
    s = f"{float(x):.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s if tex else s.replace("-", MINUS)


def pc(x, d=1):
    return fx(100 * float(x), d) + "%"


def sg(x, d=1):
    """fixed decimals with a sign: + for a positive value, the Unicode minus for a negative one"""
    s = fx(x, d)
    return s if s.startswith(MINUS) or float(s.replace(MINUS, "-")) == 0 else "+" + s


def ig(x):
    return f"{int(round(float(x))):,}"


def get(path):
    cur = D
    for k in path.split("/"):
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            if k not in cur:
                raise KeyError(f"missing key {k!r} in {path}")
            cur = cur[k]
    return cur


def fmt_value(v, f):
    if f.startswith("f"):
        return fx(v, int(f[1:]))
    if f.startswith("t"):
        return fx(v, int(f[1:]), tex=True)
    if f.startswith("p"):
        return pc(v, int(f[1:]))
    if f.startswith("d"):
        return sg(v, int(f[1:]))
    if f == "i":
        return ig(v)
    if f == "s":
        return str(v)
    if f == "m":
        return str(v).replace("-", MINUS)
    raise ValueError(f"unknown format {f}")


def fill_numbers(text):
    def rep(m):
        path, f = m.group(1).strip(), m.group(2).strip()
        return fmt_value(get(path), f)
    return re.sub(r"\{\{([^#{}]+)#([a-z0-9]+)\}\}", rep, text)


def esc(s):
    return html.escape(str(s), quote=True)


def plain(h):
    """the text of an HTML snippet"""
    return re.sub(r"<[^>]+>", "", str(h))


def table(head, rows, cls="", numeric=None, caption=None, wrap_cls="tbl", minw=0, stack=False, keep1=False, tips=None):
    """head: list of header cells (HTML); rows: list of lists of cells (HTML) or ('class', cells) tuples;
    numeric: set of column indexes that are right-aligned numbers. stack: one card per row at phone width (the page's
    script copies each column name onto its cells); keep1: the first column keeps its name in that layout too.
    tips: one plain-text description per column, in the order of head (required: the build refuses a table without
    them). A description becomes the header cell's tooltip and the footnote of every cell's row tuple. An entry can be
    a pair (full name, description): the full name then names the column wherever the header is shortened or empty
    (the row tuple, the phone-width card). Numbers come from data through {{path#fmt}}; no TeX (write c + w₁)."""
    if tips is None or len(tips) != len(head):
        raise SystemExit(f"table {[plain(h) for h in head]}: one column description per column is needed, got "
                         f"{0 if tips is None else len(tips)}")
    numeric = numeric or set()
    cls = " ".join(c for c in (cls, "stack" if stack else "", "keep1" if keep1 else "") if c)
    style = f' style="min-width:{minw}px"' if minw else ""
    out = [f'<div class="{wrap_cls}"><table{f" class={chr(34)}{cls}{chr(34)}" if cls else ""}{style}>']
    if caption:
        # the span lets a sideways-scrolling table (rows8) wrap its caption inside the visible width
        out.append(f'<caption><span class="cap">{caption}</span></caption>' if "rows8" in cls else f'<caption>{caption}</caption>')
    ths = []
    for i, h in enumerate(head):
        full, desc = tips[i] if isinstance(tips[i], tuple) else (None, tips[i])
        assert desc and desc.strip(), f"empty description for column {plain(h)!r}"
        ths.append(f'<th{" class=" + chr(34) + "n" + chr(34) if i in numeric else ""} scope="col"'
                   f'{f" data-label={chr(34)}{esc(full)}{chr(34)}" if full else ""} data-tip="{esc(desc)}">{h}</th>')
    out.append("<thead><tr>" + "".join(ths) + "</tr></thead><tbody>")
    for r in rows:
        rc = ""
        if isinstance(r, tuple):
            rc, r = r
        if len(r) == 1 and len(head) > 1:  # a group heading across the whole table
            cells = f'<td colspan="{len(head)}">{r[0]}</td>'
        else:
            cells = "".join(f'<td{" class=" + chr(34) + "n" + chr(34) if i in numeric else ""}>{c}</td>' for i, c in enumerate(r))
        out.append(f'<tr{f" class={chr(34)}{rc}{chr(34)}" if rc else ""}>{cells}</tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


SIT = ["recorded", "strong_current", "big_waves", "big_kicks"]
SIT_LABEL = {"recorded": "Recorded conditions", "strong_current": "Strong current", "big_waves": "Big waves",
             "big_kicks": "Big starting pushes"}
CTRL = ["recording", "stall_aware", "rising_thrust", "normal_driving_only", "no_throttle"]
CTRL_LABEL = {"recording": "Recording controller", "stall_aware": "Stall-aware controller",
              "rising_thrust": "Rising-thrust controller", "normal_driving_only": "Normal driving only",
              "no_throttle": "No throttle"}
FRAG = {}

E = D["environment"]
RC = D["recording_controller"]
RR = D["reference_returns"]
DS = D["dataset"]
EP = D["recorded_episode"]
# the recorded episode's counts, for its chart's legend and the text under it: the removal takes every full-throttle
# command of the episode, plus a few normal-driving rows whose commands lie close to the full-throttle pattern
_extra = [t for t in EP["removed_rows_steps"] if t not in EP["full_throttle_steps"]]
assert all(t in EP["removed_rows_steps"] for t in EP["full_throttle_steps"]), "the removal kept a full-throttle command"
EP["counts"] = dict(full_throttle=len(EP["full_throttle_steps"]), removed=len(EP["removed_rows_steps"]),
                    region=len(EP["full_throttle_region_steps"]), extra=len(_extra),
                    extra_steps=", ".join(str(t) for t in _extra[:-1]) + (" and " if len(_extra) > 1 else "")
                    + str(_extra[-1]))
SC = D["shift_cards"]
ME = D["methods"]


# ------------------------------------------------------------------------------------------------ spec table

obs_lo, obs_hi = E["observation_space"]["low"], E["observation_space"]["high"]
assert obs_lo[:10] == [-300.0] * 10 and obs_hi[:10] == [300.0] * 10 and obs_lo[10:] == [0.0, -1.0, -1.0] \
    and obs_hi[10:] == [1.5, 1.0, 1.0]
FRAG["SPEC"] = f"""<div class="tbl"><table class="spec" data-cols="Item|Value" data-coltips="What this row of the specification describes|Its value for ThrusterChain"><tbody>
<tr><th scope="row" data-tip="The set of allowed commands: four throttles, one per thruster, each from −1 to 1.">Action Space</th><td><code>{esc(E['action_space']['repr'])}</code><br><span class="note">4 numbers,
each from −1 to 1. <code>Box(low, high, shape, type)</code> is Gymnasium's description of a list of numbers with these
bounds; <code>float32</code> is how each number is stored (see <a href="#action-space">Action Space</a>).</span></td></tr>
<tr><th scope="row" data-tip="The set of possible observations: the 13 numbers a controller reads at every step.">Observation Space</th><td><code>Box(low, high, (13,), float32)</code><br><span class="note">13
numbers. low: −300 for each of the 10 float numbers, then 0, −1, −1. high: 300 for each of the 10 float numbers, then
1.5, 1, 1.</span></td></tr>
<tr><th scope="row" data-tip="How to load the environment in Python.">import</th><td><pre class="code">import experiments.shift.thruster_chain  # registers the 8 ids
import gymnasium as gym
env = gym.make("BCAShift/ThrusterChain-v0")</pre><span class="note">Run from the repository root. Gymnasium {esc(E['gymnasium_version'])} lives in its own directory; importing <code>experiments.shift</code> puts it on the path, so import it before <code>gymnasium</code>.</span></td></tr>
<tr><th scope="row" data-tip="How long one run lasts, what ends it, and how much later rewards count.">Episode</th><td>{E['episode']['length']} steps, ended by truncation (a cut-off) only; discount
{E['episode']['discount']} (a reward t steps ahead counts {E['episode']['discount']}<sup>t</sup> as much)</td></tr>
<tr><th scope="row" data-tip="The names that gym.make knows for this environment.">Registered ids</th><td>8: four situations × two worlds (<a href="#arguments">Arguments</a>)</td></tr>
<tr><th scope="row" data-tip="Where the environment's code lives in the repository.">Code</th><td><code>experiments/<wbr>shift/<wbr>thruster_chain.py</code> (numpy {esc(E['numpy_version'])})</td></tr>
</tbody></table></div>"""


# ------------------------------------------------------------------------------------------------ action space

rows = []
for e in E["action_space"]["entries"]:
    k = e["index"] + 1
    rows.append([str(e["index"]), f"Throttle of thruster {k}", fx(e["low"], 0), fx(e["high"], 0),
                 f"floats {k} and {k + 1} (observations {2 * k - 1}, {2 * k + 1})",
                 "past 80% (stalling world only)"])
FRAG["ACTION_TABLE"] = table(["Num", "Action", "Control Min", "Control Max", "Pushes the speeds of", "Stalls"], rows,
                             numeric={2, 3}, minw=640, stack=True, keep1=True, tips=[
    "Where this throttle sits in the action, a list of 4 numbers counted from 0 as in Python.",
    "What this number of the action sets: the throttle of one thruster, from −1 (full throttle upstream, against the "
    "water) through 0 (off) to 1 (full throttle downstream).",
    "The lowest throttle the environment applies. A smaller command is clipped to it.",
    "The highest throttle the environment applies. A larger command is clipped to it.",
    "The two floats this thruster sits between. Its push is added to the speed numbers of both (the observation "
    "numbers in brackets), scaled by the push factor 0.3.",
    "When the propeller stalls: past 80% throttle in either direction it pushes less while it still costs more. Only "
    "in the stalling world; the no-stall comparison world has no stall."])

kp = E["thrust_curve"]["key_points"]


def f8(v):
    s = f"{float(v):.8f}".rstrip("0").rstrip(".")
    return s if s else "0"


rows = []
for p in kp:
    t = p["throttle"]
    label = {0.800000011920929: "float32 of 0.8 (the 80% point)", 0.8000000715255737: "next float32 above it"}.get(t)
    if label is None and abs(t - 0.8) < 1e-12:
        label = "80% (float64 0.8)"
    tt = label or f"{100 * t:.0f}%"
    rows.append(("hl" if p["stalled"] else "",
                 [tt, f8(p["thrust_stall"]), f8(p["thrust_no_stall"]), "yes" if p["stalled"] else "no", f8(p["energy_cost"])]))
FRAG["THRUST_POINTS"] = table(["Throttle", "Push, stalling world", "Push, no-stall comparison world", "Stalls",
                               "Throttle cost 0.3 a²"], rows, numeric={1, 2, 4}, minw=540, stack=True, tips=[
    "The size of one thruster's throttle, from off (0%) to full throttle (100%). The rows near 80% give the exact "
    "float32 numbers where the stall starts.",
    f"How hard the thruster really pushes at this throttle in the stalling world: equal to the throttle up to 80%, "
    f"then falling, to {STALL90} at 90% and {STALL100} at full throttle.",
    "How hard the thruster pushes at this throttle in the no-stall comparison world: always equal to the throttle.",
    "Whether the propeller is stalled at this throttle in the stalling world.",
    "What this throttle adds to the step's cost: 0.3 times the throttle squared, charged even while the propeller "
    "stalls."])


def svg_line_chart(series, xr, yr, xticks, yticks, xlab, ylab, w=440, h=270, vlines=(), hlines=(), shade=(),
                   points=(), labels=(), aria="", xfmt=None, yfmt=None, pad=(58, 14, 18, 40), tip=None, tips=None):
    """A small static SVG chart. series: (css class, [(x, y), ...]). Coordinates in data units.
    Tooltips: the whole chart says what its axes and marks are (tip, by default the aria-label); tips maps a curve's,
    line's or shaded band's css class to its own tooltip, shown along a wide invisible stroke over the mark (a shaded
    band shows it over its area); a point can carry its own as a fifth element."""
    tips = tips or {}
    ht = lambda cls: f' data-tip="{esc(tips[cls])}"' if cls in tips else ""
    L, R, T, B = pad
    X = lambda x: L + (w - L - R) * (x - xr[0]) / (xr[1] - xr[0])
    Y = lambda y: T + (h - T - B) * (1 - (y - yr[0]) / (yr[1] - yr[0]))
    xfmt = xfmt or (lambda v: fx(v, 1))
    yfmt = yfmt or (lambda v: fx(v, 1))
    o = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(aria)}" data-tip="{esc(tip or aria)}">']
    for (x0, x1, cls) in shade:
        o.append(f'<rect class="{cls}" x="{X(x0):.1f}" y="{T}" width="{X(x1) - X(x0):.1f}" height="{h - T - B}"{ht(cls)}/>')
    for v in yticks:
        o.append(f'<line class="grid" x1="{L}" x2="{w - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        o.append(f'<text class="tick" x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{yfmt(v)}</text>')
    for v in xticks:
        o.append(f'<text class="tick" x="{X(v):.1f}" y="{h - B + 16}" text-anchor="middle">{xfmt(v)}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}"/>')
    o.append(f'<line class="axis" x1="{L}" x2="{L}" y1="{T}" y2="{h - B}"/>')
    for (x, cls, txt) in vlines:
        o.append(f'<line class="{cls}" x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{T}" y2="{h - B}"/>')
        if cls in tips:
            o.append(f'<line class="hit" x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{T}" y2="{h - B}"{ht(cls)}/>')
        if txt:
            o.append(f'<text class="ann" x="{X(x) - 4:.1f}" y="{h - B - 6}" text-anchor="end">{txt}</text>')
    for (y, cls, txt, anchor) in hlines:
        o.append(f'<line class="{cls}" x1="{L}" x2="{w - R}" y1="{Y(y):.1f}" y2="{Y(y):.1f}"/>')
        key = f"{cls}@{y:g}" if f"{cls}@{y:g}" in tips else cls
        if key in tips:
            o.append(f'<line class="hit" x1="{L}" x2="{w - R}" y1="{Y(y):.1f}" y2="{Y(y):.1f}"{ht(key)}/>')
        if txt:
            o.append(f'<text class="ann" x="{(w - R - 4) if anchor == "end" else L + 6}" y="{Y(y) - 5:.1f}" text-anchor="{anchor}">{txt}</text>')
    for cls, pts in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        o.append(f'<path class="{cls}" d="{d}"/>')
        if cls in tips:
            o.append(f'<path class="hit" d="{d}"{ht(cls)}/>')
    for p in points:  # (x, y, class, radius[, hover title])
        x, y, cls, r = p[:4]
        tip = f' data-tip="{esc(p[4])}"' if len(p) > 4 else ""
        o.append(f'<circle class="{cls}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{r}"{tip}/>')
    for (x, y, cls, txt, anchor, dx, dy) in labels:
        o.append(f'<text class="{cls}" x="{X(x) + dx:.1f}" y="{Y(y) + dy:.1f}" text-anchor="{anchor}">{txt}</text>')
    o.append(f'<text class="axlab" x="{(L + w - R) / 2:.1f}" y="{h - 4}" text-anchor="middle">{xlab}</text>')
    o.append(f'<text class="axlab" transform="translate(14 {(T + h - B) / 2:.1f}) rotate(-90)" text-anchor="middle">{ylab}</text>')
    o.append("</svg>")
    return "".join(o)


grid = E["thrust_curve"]["grid"]
FRAG["THRUST_CHART"] = svg_line_chart(
    [("ln-nostall", [(g["throttle"], g["thrust_no_stall"]) for g in grid]),
     ("ln-cost", [(g["throttle"], g["energy_cost"]) for g in grid]),
     ("ln-stall", [(g["throttle"], g["thrust_stall"]) for g in grid])],
    (0, 1), (0, 1.0), [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "throttle |a|", "push or cost", vlines=[(0.8, "knee", "80%")],
    points=[(0.8, 0.8, "pt-stall", 4, "At 80% throttle the push is 0.8 in both worlds: the most a stalling propeller gives."),
            (0.9, 0.65, "pt-stall", 4, f"Stalling world, 90% throttle: push {STALL90}."),
            (1.0, 0.5, "pt-stall", 4, f"Stalling world, full throttle: push {STALL100}, less than at 80%."),
            (1.0, 1.0, "pt-nostall", 4, "No-stall comparison world, full throttle: push 1."),
            (1.0, 0.3, "pt-cost", 3.5, "Throttle cost at full throttle: 0.3, in both worlds.")],
    labels=[(1.0, 1.0, "lab", "1.0", "start", 9, 5), (1.0, 0.5, "lab", "0.5", "start", 9, 5),
            (1.0, 0.3, "lab", "0.3", "start", 9, 5), (0.8, 0.8, "lab", "0.8", "end", -7, -8)],
    xfmt=lambda v: f"{v:.1f}", yfmt=lambda v: "" if v == 0 else f"{v:.1f}", pad=(58, 40, 18, 40),
    tips={"ln-stall": f"Push in the stalling world: equal to the throttle up to 80%, then falling to {STALL90} at 90% "
                      f"and {STALL100} at full throttle.",
          "ln-nostall": "Push in the no-stall comparison world: it keeps rising with the throttle, to 1 at full "
                        "throttle.",
          "ln-cost": "The throttle cost, 0.3 × a²: it keeps rising, to 0.3 at full throttle, whether or not the "
                     "propeller stalls.",
          "knee": "80% throttle: past this line a propeller stalls in the stalling world."},
    aria="Push against throttle: equal to the throttle up to 80% in both worlds; past 80% it falls to 0.65 at 90% and 0.5 at full throttle in the stalling world, and keeps rising to 1 in the no-stall comparison world. The throttle cost 0.3 a squared keeps rising to 0.3 in both.")


# ------------------------------------------------------------------------------------------------ observation space

desc = {0: "how far float {j} is from rest", 1: "how fast float {j} is moving"}
rows = []
for e in E["observation_space"]["entries"]:
    i = e["index"]
    if i < 10:
        j = i // 2 + 1
        name = f"{'Position' if i % 2 == 0 else 'Speed'} of float {j}"
        what = "changes every step"
    elif i == 10:
        name, what = "Strength of the current, c", "fixed for the whole episode"
    elif i == 11:
        name, what = "Wave, first number w₁ = amplitude × cos(phase): the push the wave adds now", "turns 0.27 radians per step"
    else:
        # the wave turns by R(+0.27), so w2 = amplitude x sin(phase) is the push w1 gave a quarter cycle earlier
        name, what = ("Wave, second number w₂ = amplitude × sin(phase): the push the wave added a quarter cycle ago; "
                      "positive while the wave's push is falling", "turns 0.27 radians per step")
    rows.append([str(i), name, fx(e["low"], 1 if i == 10 else 0), fx(e["high"], 1 if i == 10 else 0), what])
FRAG["OBS_TABLE"] = table(["Num", "Observation", "Min", "Max", "Over an episode"], rows, numeric={2, 3}, minw=560,
                          stack=True, keep1=True, tips=[
    "Where this number sits in the 13-number observation, counted from 0.",
    "What the number measures. Each float has two numbers, a position number and a speed number: the two coordinates "
    "of its swing around rest, both 0 at rest.",
    "The lowest value the observation space declares for this number. The floats never come near −300: from any "
    "allowed start they stay within {{environment/observation_bound/bound#f1}} of rest.",
    "The highest value the observation space declares for this number.",
    "How the number behaves during one episode: the float numbers change at every step, the current never changes, "
    "and the two wave numbers turn by 0.27 radians per step."])

rows = []
for j, m in enumerate(E["dynamics"]["oscillator_modes"]):
    rows.append([f"Float {j + 1}", fx(m["radius"], 4), fx(m["angle_rad"], 3), fx(2 * math.pi / m["angle_rad"], 1),
                 fx(math.log(0.5) / math.log(m["radius"]), 0)])
# each swing of A lies almost entirely on one float (swing j on float j), so the rows are named by their float
assert all(s["float_with_most_weight"] == s["swing"] for s in D["overview"]["swings"]["list"])
assert all(abs(s["fade_per_step"] - m["radius"]) < 1e-6 and abs(s["turn_per_step"] - m["angle_rad"]) < 1e-6
           for s, m in zip(D["overview"]["swings"]["list"], E["dynamics"]["oscillator_modes"]))
FRAG["MODES_TABLE"] = table(["Float", "Shrink per step ρ", "Turn per step θ (radians)", "Period 2π/θ (steps)", "Steps to halve"],
                            rows, numeric={1, 2, 3, 4}, minw=470, stack=True, tips=[
    "The float whose swing this row describes: each of the five swings lies almost entirely on one float.",
    "The factor ρ by which the float's swing shrinks at each step when nothing pushes it. Closer to 1 fades more slowly.",
    "How far the float's swing turns at each step, θ, in radians: a whole swing is 2π, about 6.28 radians.",
    "How many steps one whole swing takes: 2π divided by the turn per step.",
    "How many steps the swing takes to fade to half its size on its own, rounded to whole steps."])


def mat_table(M, rows, cols, d, tips):
    """a matrix: tips is the (full name, description) of the row-name column, then one description per column"""
    head = [""] + cols
    body = []
    for i, r in enumerate(M):
        body.append([f"<b>{rows[i]}</b>"] + [("0" if abs(v) < 0.5 * 10 ** -d else fx(v, d)) for v in r])
    return table(head, body, cls="mat", numeric=set(range(1, len(cols) + 1)), tips=tips)


def lab_long(j):
    """the plain name of float number j (0-9) or water number j (10-12)"""
    if j < 10:
        return f"float {j // 2 + 1}'s {'position' if j % 2 == 0 else 'speed'} number"
    return ["the current's strength c", "w₁, the push the wave adds now", "w₂, the push the wave added a quarter cycle "
            "earlier"][j - 10]


lab = []
for j in range(1, 6):
    lab += [f"pos {j}", f"speed {j}"]
FRAG["A_TABLE"] = mat_table(E["dynamics"]["A_floats_10x10"], lab, lab, 4, [
    ("Next-state number", "The float number of the next state that this row computes, before the thrusters and the "
                          "water add their push.")] + [
    f"How much {lab_long(j)} in this state adds to the row's number in the next state." for j in range(10)])
FRAG["B_TABLE"] = mat_table([r for r in E["dynamics"]["B_13x4"][:10]], lab, [f"thruster {k}" for k in range(1, 5)], 1, [
    ("Float number", "The float number that this row's share of each thruster's push is added to.")] + [
    f"How much of thruster {k}'s push one step adds to this float number: 0.3, the push factor, on the speed numbers of "
    f"floats {k} and {k + 1}, and 0 elsewhere." for k in range(1, 5)])
K = RC["textbook_gain_K_4x13"]
FRAG["K_TABLE"] = mat_table(K, [f"thruster {k}" for k in range(1, 5)], lab + ["current c", "wave w₁", "wave w₂"], 4, [
    ("Thruster", "The thruster whose textbook command this row computes.")] + [
    f"The weight of {lab_long(j)} in the textbook command: each throttle is minus the weighted sum of the 13 observed "
    f"numbers." for j in range(13)])


# ------------------------------------------------------------------------------------------------ starting state

rows = []
outside = {"recorded": "Nothing, by design: these are the conditions the data were recorded in. About 1% of the "
                        "states met still count as outside on the float test of the <a href=\"#shift-card\">shift "
                        "card</a> (The distribution shift), because that test's mark, the 99th percentile, leaves out the farthest 1% of the "
                        "recorded training states themselves.",
           "strong_current": "The current, at every start: it is above the recorded maximum, "
                             "{{environment/start_laws/recorded/current/1#f1}}.",
           "big_waves": "The wave, at every start: its amplitude is above the recorded maximum, "
                        "{{environment/start_laws/recorded/wave_amplitude/1#f1}}.",
           "big_kicks": "The floats, on part of the steps: they lie farther from rest than the 99th percentile of the "
                        "recorded training states."}
tests = {"recorded": "No shift. Here BCA's actor must not lose to plain TD3+BC: in the no-loss check, the lower "
                     "bound of BCA minus plain TD3+BC must be above {{methods/gate/pass_marks/B2/p2_lower_gt#f0}} "
                     "points.",
         "strong_current": "The main test. Holding against this current often asks thruster 1 for 80% or more, where "
                           "its propeller stalls: with the floats at rest, the textbook hold against the current alone "
                           "does so once the current exceeds "
                           "{{recording_controller/current_where_textbook_hold_reaches_80pct_thruster1#f3}}.",
         "big_waves": "A shift in the waves: the water's push swings further into the full-throttle region.",
         "big_kicks": "A shift in the floats: the episode starts far from any recorded state."}
changed = {"recorded": None, "strong_current": 1, "big_waves": 2, "big_kicks": 3}
for s in SIT:
    L_ = E["start_laws"][s]
    cells = [f"<b>{SIT_LABEL[s]}</b>", f"{fx(L_['current'][0], 1)} to {fx(L_['current'][1], 1)}",
             f"{fx(L_['wave_amplitude'][0], 1)} to {fx(L_['wave_amplitude'][1], 1)}", fx(L_['float_start_sd'], 1)]
    if changed[s]:
        cells[changed[s]] = f"<b>{cells[changed[s]]}</b>"
    rows.append(cells + [outside[s], tests[s]])
# the part each shifted situation changes is the only one that differs from the recorded start law
_rec = E["start_laws"]["recorded"]
for s, k in (("strong_current", "current"), ("big_waves", "wave_amplitude"), ("big_kicks", "float_start_sd")):
    assert all((E["start_laws"][s][q] == _rec[q]) == (q != k) for q in ("current", "wave_amplitude", "float_start_sd"))
FRAG["START_TABLE"] = table(["Situation", "Current c", "Wave amplitude", "Float start spread",
                             "What lies beyond the recorded data", "What it tests"], rows, minw=820, stack=True, tips=[
    "One of the four ways a start is drawn: the recorded conditions of the data, or one of the three shifted "
    "situations beyond them.",
    ("Current's strength c", "The range the current's strength is drawn from: every value in it is equally likely. "
     "The current then stays the same for the whole episode."),
    ("Wave amplitude", "The range the wave's amplitude, the largest push the wave adds or takes away, is drawn from: "
     "every value in it is equally likely."),
    ("Float start spread (standard deviation)", "How far from rest the floats typically start: each of the ten float "
     "numbers is drawn from a bell curve around rest (0) with this standard deviation."),
    "Which part of this situation lies beyond the conditions of the recorded data, and how often.",
    "What this situation checks about a learned controller."])


# ------------------------------------------------------------------------------------------------ recording controller

pat = RC["full_throttle_pattern_exact"]
ts = RC["thrust_of_pattern"]
rows = []
for k in range(4):
    a = float(pat[k])
    rows.append([f"Thruster {k + 1}", fx(a, 3), fx(ts["stall"][k], 3), fx(ts["no_stall"][k], 3),
                 fx(RC["full_throttle_pattern_capped_at_80"][k], 3)])
rows.append(("sub", ["Throttle cost 0.3 Σa²", fx(RC["energy_cost"]["pattern"], 3), "", "",
                     fx(RC["energy_cost"]["capped_pattern"], 3)]))
FRAG["PATTERN_TABLE"] = table(["", "Full-throttle command", "Push, stalling world", "Push, no-stall comparison world",
                               "Capped at 80% (command = push)"], rows, numeric={1, 2, 3, 4}, minw=580, stack=True, tips=[
    ("Thruster", "The thruster this row describes. The last row adds up the throttle cost of all four."),
    "The recording controller's full-throttle command, the fixed pattern P: the textbook hold against the current, "
    "scaled up until thruster 1 is at full throttle upstream (−1).",
    "What each thruster really pushes under that command in the stalling world: thruster 1 stalls and pushes only "
    f"{STALL100}.",
    "What each thruster pushes under that command in the no-stall comparison world: the command itself.",
    "The same pattern with thruster 1 held at 80% (the capped pattern): nothing stalls, so each thruster pushes "
    "exactly its command."])


# ------------------------------------------------------------------------------------------------ recorded data

arr = DS["arrays"]
# (plain name, D4RL array name, what it holds); the D4RL names appear only in the code-names table
FIELDS = [("Observation before the step", "observations", "the 13 observed numbers"),
          ("Throttles", "actions", "the 4 throttles the recording controller applied"),
          ("Reward", "rewards", "the step's reward"),
          ("Observation after the step", "next_observations", "the 13 numbers after the step (always read this, never the next row)"),
          ("Ended here?", "terminals", "always no: the task never ends"),
          ("Cut off here?", "timeouts", "yes on the last step of each episode (t = 99), no elsewhere"),
          ("Exact state before the step", "infos/state", "the 13 numbers in float64"),
          ("Exact state after the step", "infos/next_state", "the 13 numbers in float64"),
          ("Episode", "infos/episode", "episode number"), ("Step", "infos/t", "step within the episode, t = 0 to 99"),
          ("Split", "infos/split", "training, calibration or held out")]
rows = []
for plain_name, k, what in FIELDS:
    a = arr[k]
    rows.append([plain_name, a["dtype"], " × ".join(f"{n:,}" for n in a["shape"]), what])
FRAG["FIELDS_TABLE"] = table(["Field", "Type", "Shape (one data seed)", "What it holds"], rows, minw=640, stack=True, tips=[
    "The plain name of one field stored for every recorded row. The file's own names are listed under Names used in "
    "the code.",
    "How the field is stored: float32 and float64 are decimal numbers with about 7 and 16 significant digits, int32 "
    "and int8 are whole numbers, bool is yes or no.",
    "How many numbers the field holds for one data seed: rows × numbers per row.",
    "What the field records for each row."])

FRAG["SPLITS_TABLE"] = table(
    ["Split", "Episodes", "Rows", "Used for"],
    [["Training", ig(DS["episodes"]["train"]), ig(DS["rows"]["train"]), "learning (after the removal described below)"],
     ["Calibration", ig(DS["episodes"]["calibration"]), ig(DS["rows"]["calibration"]),
      f"the calibration set: {ig(DS['rows']['calibration_bank'])} rows (5 per episode) that BCA uses to calibrate its band"],
     ["Held out", ig(DS["episodes"]["evaluation"]), ig(DS["rows"]["evaluation"]),
      "never trained on; used only to measure distances and strengths"]],
    numeric={1, 2}, minw=500, stack=True, tips=[
        "One of the three parts the recorded episodes are divided into. Every episode belongs to exactly one split.",
        "How many recorded episodes of 100 steps the split holds.",
        "How many recorded rows (steps) the split holds: 100 per episode.",
        "What the split is used for."])

# the recorded episode: thruster 1's command, step by step
acts = EP["whole_episode"]["actions"]
ft = set(EP["full_throttle_steps"])
rm = set(EP["removed_rows_steps"])
reg = set(EP["full_throttle_region_steps"])
shade = []
t = 0
while t < 100:
    if t in reg:
        t0 = t
        while t < 100 and t in reg:
            t += 1
        shade.append((t0 - 0.5, t - 0.5, "shade-region"))
    else:
        t += 1
pts = []
for t in range(100):
    a1 = acts[t][0]
    ptip = (f"Step {t}: thruster 1 throttle {fx(a1, 2)}, "
            + ("the full-throttle command" if t in ft else "normal driving")
            + (", in the full-throttle region" if t in reg else ", outside the full-throttle region")
            + ("; deleted by the densest-15% removal." if t in rm else "; kept in the training data."))
    if t in ft:
        pts.append((t, a1, "pt-ft", 3.6, ptip))
    else:
        pts.append((t, a1, "pt-normal", 2.6, ptip))
    if t in rm:
        pts.append((t, a1, "pt-removed", 5.6, ptip))
FRAG["EPISODE_CHART"] = svg_line_chart(
    [("ln-ep", [(t, acts[t][0]) for t in range(100)])], (-1, 100), (-1.1, 0.4), [0, 20, 40, 60, 80, 99],
    [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4], "step t", "thruster 1 throttle", w=560, h=290,
    hlines=[(-0.8, "knee", "", "end")], shade=shade, points=pts,
    xfmt=lambda v: str(int(v)), yfmt=lambda v: fx(v, 1), pad=(62, 12, 14, 40),
    tips={"ln-ep": "Thruster 1's recorded throttle, step by step, over the episode's 100 steps.",
          "knee": "The −80% line: below it thruster 1 pushes upstream past 80%, and its propeller stalls.",
          "shade-region": f"Shaded: steps in the full-throttle region, where the water's push is above {TRIG}. There "
                          "the recording controller flips a coin in each state and on heads gives the full-throttle "
                          "command."},
    aria="Thruster 1's recorded throttle at every step of the episode. Shaded steps are in the full-throttle region. The full-throttle commands sit near minus 1, past the minus 80% line, and every one of them is circled as deleted by the densest-15% removal.")

def rows_block(rs):
    head = ["Field"] + [f"t = {r['t']}" for r in rs]
    body = []
    for i in range(13):
        nm = (f"observation {i}: {'position' if i % 2 == 0 else 'speed'} of float {i // 2 + 1}" if i < 10 else
              ["observation 10: current c", "observation 11: wave w₁", "observation 12: wave w₂"][i - 10])
        body.append([nm] + [r["observations"][i].replace("-", MINUS) for r in rs])
    for k in range(4):
        body.append(([f"throttle of thruster {k + 1} (action {k})"] + [
            f'<span class="{"stallv" if abs(float(r["actions"][k])) > 0.800000011920929 else ""}">{r["actions"][k].replace("-", MINUS)}</span>'
            for r in rs]))
    body.append(("sub", ["reward"] + [r["rewards"].replace("-", MINUS) for r in rs]))
    body.append(["ended here?"] + ["no" for r in rs])
    body.append(["cut off here?"] + [("yes" if r["timeouts"] else "no") for r in rs])
    body.append(["water's push c + w₁"] + [fx(r["water_push"], 4) for r in rs])
    body.append(["in the full-throttle region"] + [("yes" if r["in_full_throttle_region"] else "no") for r in rs])
    body.append(["full-throttle command (diagnostic only)"] + [('<b class="warnt">yes</b>' if r["full_throttle_command"] else "no") for r in rs])
    body.append(["deleted by the densest-15% removal"] + [('<b class="warnt">deleted</b>' if r["removed_by_densest_15pct"] else "kept") for r in rs])
    return table(head, body, cls="steps rows8", numeric=set(range(1, len(rs) + 1)), tips=[
        "What the row shows: one of the 13 observed numbers, one of the 4 throttles, the reward, the two end markers, or "
        "one of three diagnostics shown only on this page."] + [
        f"The row of the file at step {r['t']} of the episode: the state in which the command was given, the command "
        f"itself and its reward." for r in rs])


rows8 = EP["first_8_rows"]
FRAG["EPISODE_ROWS"] = rows_block(rows8[:4]) + "\n" + rows_block(rows8[4:])

w0 = EP["worked_reward_step0"]
st0 = [float(v) for v in w0["state_float64"]]
rows = []
for i in range(10):
    rows.append([f"float {i // 2 + 1}, {'position' if i % 2 == 0 else 'speed'}", f"{st0[i]:.8f}".replace("-", MINUS),
                 f"{w0['squares_of_floats'][i]:.8f}"])
rows.append(("sub", ["sum", "", f"{w0['sum_squares_floats']:.8f}"]))
FRAG["WORKED_FLOATS"] = table(["State number", "Exact value (float64)", "Square"], rows, cls="steps", numeric={1, 2}, tips=[
    "One of the ten float numbers of the episode's first state (step 0).",
    "The number as the environment holds it, in float64: about 16 significant digits.",
    "The number squared. The ten squares add up (last row) to the float cost in the reward of step 0."])
sq = w0["squares_of_floats"]
sqc = w0["squares_of_command"]
cmd = [float(c) for c in w0["command"]]
FRAG["WORKED_TEX"] = (
    "\\[ \\begin{aligned} \\textstyle\\sum_{i=1}^{10} s_i^2 &= " + " + ".join(f"{v:.5f}" for v in sq[:3]) + " \\\\ &\\quad + "
    + " + ".join(f"{v:.5f}" for v in sq[3:6]) + " \\\\ &\\quad + " + " + ".join(f"{v:.5f}" for v in sq[6:9])
    + f" \\\\ &\\quad + {sq[9]:.5f} = {w0['sum_squares_floats']:.5f} \\end{{aligned}} \\]\n"
    + "\\[ \\begin{aligned} 0.3 \\textstyle\\sum_{k=1}^{4} a_k^2 &= 0.3 \\times \\big(" + " + ".join(f"({c:.4f})^2" for c in cmd[:2])
    + " \\\\ &\\qquad + " + " + ".join(f"({c:.4f})^2" for c in cmd[2:]) + "\\big) \\\\ &= 0.3 \\times "
    + f"{w0['sum_squares_command']:.5f} = {w0['throttle_cost_term']:.5f} \\end{{aligned}} \\]\n"
    + f"\\[ r_0 = -({w0['sum_squares_floats']:.5f} + {w0['throttle_cost_term']:.5f}) = {float(w0['reward_float64']):.5f} \\]")


# ------------------------------------------------------------------------------------------------ the shift

g = DS["gap"]
gr = DS["gap_removal"]
rr = DS["random_removal"]
FRAG["GAP_FACTS"] = table(
    ["Training rows", "Full log (zero shift)", "Densest-15% removal", "Same number removed at random"],
    [["rows kept", ig(DS["rows"]["train"]), ig(gr["kept_training_rows"]), ig(rr["kept_training_rows"])],
     ["calibration-set rows kept", ig(DS["rows"]["calibration_bank"]), ig(gr["bank_rows_kept"]),
      ig(DS["rows"]["calibration_bank"] - rr["bank_rows_removed"])],
     ["full-throttle commands removed", "0%", pc(gr["full_throttle_commands_removed_share"], 2), pc(rr["full_throttle_commands_removed_share"], 2)],
     ["kept rows past 80% on any thruster", pc(DS["past_80_share_train"], 2), pc(gr["kept_rows_past_80_share"], 3), pc(rr["kept_rows_past_80_share"], 2)],
     ["past-80% rows that survive", "100%", fx(100 * gr["kept_rows_past_80"] / (DS["past_80_share_train"] * DS["rows"]["train"]), 2) + "%",
      pc(rr["past_80_rows_kept_share_of_all_past_80"], 2)],
     ["full-throttle-region rows kept", "100%", pc(gr["full_throttle_region_rows_kept_share"], 2), "—"],
     ["kept rows that follow the identical rule in both worlds", "", pc(gr["identical_rule_share_kept"], 3), ""]],
    numeric={1, 2, 3}, minw=620, stack=True, tips=[
        "What is measured on the training data.",
        "The training data with nothing removed: the zero-shift control.",
        "The training data after the densest-15% removal, which deletes every training row whose command lies inside "
        "the removed ball.",
        "The training data after the random-removal control: the same number of rows deleted, chosen uniformly at "
        "random."])

# shift card table
CARD_CTRL = ["recording", "stall_aware", "rising_thrust", "normal_driving_only"]
rows = []
for s in SIT:
    rows.append(("sub", [f"{SIT_LABEL[s]}"] + [""] * 9))
    for c in CARD_CTRL:
        k = SC["cards"][s][c]
        ib = k["inside_removed_ball"]
        cov = "—" if k["covered_share"] is None else pc(k["covered_share"], 0)
        rows.append([CTRL_LABEL[c], fx(k["removed_share"]["mean"], 3), pc(k["unrecorded_throttle_share"], 1),
                     pc(k["full_throttle_region_share"], 1), pc(k["outside_recorded_conditions"]["share"], 1),
                     f"{pc(ib['past_80'], 1)} / {pc(ib['at_or_below_80'], 1)}",
                     fx(k["distance_to_data"]["visited"]["p50"], 2), cov,
                     fx(k["points"]["loss_from_unrecorded_throttle"], 1), fx(k["points"]["policy"], 1)])
FRAG["CARD_TABLE"] = table(
    ["Controller", "1. Removed share, mean", "2. Steps past 80%", "3. In the full-throttle region",
     "4. Outside the recorded conditions", "5. Command in the removed ball: past 80% / at or below",
     "6. Distance to the data (median)", "7. Covered share", "8. Loss from pushing past 80% (points)", "Points"],
    rows, cls="card", numeric=set(range(1, 10)), stack=True, tips=[
        "The fixed controller scored in this row, in the situation named by the heading row above it.",
        "Over the states this controller visits, the average removed share: the probability that the recording "
        "controller's command at that state would fall inside the removed ball.",
        "The share of this controller's steps that command more than 80% on any thruster: throttles the training data "
        "no longer show.",
        f"The share of visited states in the full-throttle region, where the water's push c + w₁ is above {TRIG}.",
        "The share of visited states outside the recorded conditions: current above 0.7, wave amplitude above 0.4, or "
        "floats farther from rest than 99% of the recorded training states.",
        "The share of steps whose command lies inside the removed ball, split into commands past 80% (which the data "
        "cannot teach) and commands at or below 80%.",
        "The median, over visited steps, of the distance to the 10th-nearest kept training row, with state and command "
        "measured together and every number divided by its spread in the data.",
        "Of this controller's steps past 80%, the share that lie inside the recorded conditions. A dash: it never "
        "pushes past 80%.",
        "The points this controller loses by pushing past 80%: the points of the same controller with every throttle "
        "capped at 80%, minus its own points.",
        "The controller's score in this situation: 0 is the recording controller, 100 the stall-aware controller."])

# Setting 1: where the capped pattern sits relative to the removed ball (the full-throttle pattern's distance is
# recorded in the page data; the capped pattern's is computed here from the same centre and the exact pattern)
_centre = [float(v) for v in DS["gap"]["centre"]]
_cap = [float(v) for v in RC["full_throttle_pattern_exact"]]
_cap[0] = -0.8
assert all(abs(a - b) < 6e-5 for a, b in zip(_cap, RC["full_throttle_pattern_capped_at_80"]))
D["derived"] = dict(capped_pattern_distance=math.dist(_cap, _centre))
assert D["derived"]["capped_pattern_distance"] < DS["gap"]["radius"] - 0.1
assert abs(math.dist([float(v) for v in RC["full_throttle_pattern_exact"]], _centre)
           - DS["gap"]["distance_centre_to_full_throttle_pattern"]) < 2e-4

tp = SC["region_pool"]["true_price"]
rows = []
for key, lab_ in [("stall_aware|stall", "Stall-aware controller continues, stalling world"),
                  ("stall_aware|no_stall", "Stall-aware controller continues, no-stall comparison world"),
                  ("rising_thrust|stall", "Rising-thrust controller continues, stalling world"),
                  ("rising_thrust|no_stall", "Rising-thrust controller continues, no-stall comparison world")]:
    v = tp[key]
    rows.append([lab_, fx(v["median_true_price"], 4), fx(v["p10"], 3), fx(v["p90"], 3), pc(v["share_positive"], 1)])
FRAG["PRICE_TABLE"] = table(["Who drives after the first step", "Median price", "10th percentile", "90th percentile",
                             "Rows where pushing past 80% is worse"], rows, numeric={1, 2, 3, 4}, minw=620, stack=True, tips=[
    "The fixed controller that drives after the first step, and the world. The first step is either the full-throttle "
    "pattern or the same pattern with thruster 1 capped at 80%.",
    "The middle value of the true price over the rows: the exact value of the capped pattern minus the exact value of "
    "the full-throttle pattern. Positive means full throttle is worse.",
    "The price that 10% of the rows fall below.",
    "The price that 90% of the rows fall below.",
    "The share of rows with a positive price, where full throttle is worth less than holding at 80%."])


# ------------------------------------------------------------------------------------------------ methods

td = ME["td3_bc"]["settings"]
FRAG["TD3_SETTINGS"] = table(
    ["Setting", "Value", "What it means"],
    [["Critic updates", ig(td["updates"]), "each update adjusts the critic once, on one batch"],
     ["Actor updates", f"every {td['policy_freq']} critic updates", "the delay in the name TD3"],
     ["Batch", f"{td['batch_size']} rows drawn at random", "the training rows one update looks at"],
     ["Discount γ", str(td["gamma"]), f"a reward t steps ahead counts {td['gamma']}<sup>t</sup> as much"],
     ["Learning rate (Adam)", f"{td['lr']}", "the step size of each adjustment; Adam is the standard rule for making it"],
     ["Target-network averaging τ", str(td["tau"]),
      f"each update moves the slowly averaged copies {100 * td['tau']:g}% of the way to the current networks"],
     ["Target noise", f"standard deviation {td['policy_noise']}, clipped at ±{td['noise_clip']}",
      "the random change added to the next command inside the critic's target"],
     ["α (critic weight in the actor loss)", f"{td['alpha']} (the published value); 5 and 10 on the "
      '<a href="#alpha-ladder">alpha ladder</a>', "how much the critic counts against the cloning pull"],
     ["Networks", "actor 13–256–256–4 with tanh output; two critics with 256–256 hidden units",
      "a network is an adjustable function. The actor maps the 13 observed numbers through two internal layers of 256 "
      "units (its hidden units) to 4 throttles, squashed into [−1, 1] by the tanh function. Each critic maps the 13 "
      "numbers and 4 throttles through two layers of 256 units to one value"]],
    minw=640, stack=True, tips=[
        "One setting of TD3+BC, the learner BCA is added to. Every run uses the repository's own settings, unchanged.",
        "The value used in every run on this page.",
        "What the setting does, in plain words."])

fp = ME["actor_fixed_point"]["share_of_way"]
MU = ME["actor_fixed_point"]["critic_weight_k_reference"]
EDGE = {k: min(MU / m, 1.0) for k, m in (("dose_1", 1.0), ("dose_1.25", 1.25), ("dose_1.5", 1.5), ("dose_2", 2.0),
                                          ("dose_3", 3.0))}
lev = ME["actor_fixed_point"]["dose_cap_leverage"]
assert abs((EDGE["dose_1"] - EDGE["dose_1.5"]) - lev["leverage_edge"]) < 1e-6
assert abs((fp["dose_1"] - fp["dose_1.5"]) - lev["leverage_bend"]) < 1e-5
D["fixedpoint"] = dict(edge=EDGE, leverage_fraction=dict(bend=lev["leverage_bend"] / fp["dose_1"],
                                                         edge=lev["leverage_edge"] / EDGE["dose_1"]))
FRAG["FIXEDPOINT_TABLE"] = table(
    ["Caution multiplier m", "Bending critic: share μ / (m + μ)", "Edge-slope critic: share min(μ / m, 1)"],
    [[lab_, fx(fp[k], 3), fx(EDGE[k], 3)] for k, lab_ in
     (("dose_1", "1 (plain TD3+BC)"), ("dose_1.25", "1.25"), ("dose_1.5", "1.5 (BCA's cap)"), ("dose_2", "2"),
      ("dose_3", "3"))],
    numeric={1, 2}, caption=f"Share of the way from the recorded command toward the critic's favourite command, at "
                            f"the critic's pull μ = {fx(MU, 2)}", tips=[
        "The caution multiplier m: how hard the actor is pulled toward the recorded command at a row. 1 in plain "
        "TD3+BC; BCA's doses lie between 1 and 1.5.",
        f"The share of the way from the recorded command toward the critic's favourite command that the actor goes when "
        f"the critic bends like the truth past 80%: μ / (m + μ), with the critic's pull μ = {fx(MU, 2)}.",
        "The same share when the critic keeps its slope past the edge of the data: μ / m, but never past the favourite "
        "command, so at most 1."])

WC = D["world_card"]
ref_h = WC["reference"]["harm"]
FRAG["FORECAST_TABLE"] = table(
    ["Situation", "Forecast harm, bending critic (points)", "Forecast harm, edge-slope critic (points)"],
    [[SIT_LABEL[s], fx(ref_h["bend"][s], 1), fx(ref_h["edge"][s], 1)] for s in SIT], numeric={1, 2}, tips=[
        "The situation the forecast is for.",
        "Points lost to pushing past 80%, forecast before any run: a fixed stand-in for the learned actor capped at 80% "
        "minus the same stand-in uncapped, for a critic that bends like the truth past 80%.",
        "The same forecast for a critic that keeps its slope past the edge of the data, which draws the actor further "
        "past 80%."])

cp = ME["caution_placements"]
rows = []
for r in cp["dose_table"]:
    if r["D_over_mean_D"] in (0.0, 0.1, 0.333333, 1.0, 2.0, 5.0, 10.0, 100.0):
        lab_ = {0.333333: "1/3"}.get(r["D_over_mean_D"], f"{r['D_over_mean_D']:g}")
        rows.append([lab_, f"{r['x']:g}", fx(r["harm_placed_dose"], 4), fx(r["cap2_dose"], 4)])
FRAG["DOSE_TABLE"] = table(["Over-preference Δ ÷ its mean", "z = 3Δ ÷ mean(Δ)",
                            "Over-preference-placed dose (BCA's form)", "Dose capped at 2 (diagnostic)"], rows,
                           numeric={0, 1, 2, 3}, tips=[
    "A row's over-preference Δ (how much more the critic prefers the actor's command over the recorded one than the "
    "truth does) divided by its average over the rows.",
    "The scaled over-preference z that sets the dose: three times Δ divided by the mean of Δ.",
    "The caution multiplier the row gets in BCA's own form, 1 + 0.5 z / (1 + z): 1 where the critic does not "
    "over-prefer, rising toward 1.5.",
    "The diagnostic form with doses up to 2, 1 + z / (1 + z): it shows whether BCA's cap of 1.5 is what limits it."])
xs = [i / 20 for i in range(0, 161)]
FRAG["DOSE_CHART"] = svg_line_chart(
    [("ln-cap2", [(r, 1 + 3 * r / (1 + 3 * r)) for r in xs]), ("ln-dose", [(r, 1 + 0.5 * 3 * r / (1 + 3 * r)) for r in xs])],
    (0, 8), (1, 2.05), [0, 1, 2, 4, 6, 8], [1.0, 1.25, 1.5, 1.75, 2.0], "over-preference Δ at the row ÷ its mean",
    "caution multiplier", w=440, h=250, hlines=[(1.5, "knee", "BCA's cap 1.5", "end"), (2.0, "knee", "cap 2", "end")],
    points=[(1.0, 1.375, "pt-dose", 4, "At a row with the mean over-preference, BCA's form gives the dose 1.375."),
            (1 / 3, 1.25, "pt-dose", 4, "At a row with a third of the mean over-preference, the dose is 1.25.")],
    labels=[(1.0, 1.375, "lab", "1.375 at the mean", "start", 7, 16), (1 / 3, 1.25, "lab", "1.25", "start", 6, 14)],
    xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:.2f}",
    tips={"ln-dose": "Over-preference-placed caution in BCA's form: 1 + 0.5 × z / (1 + z), with z three times the row's "
                     "over-preference divided by its mean. It rises from 1 toward BCA's cap of 1.5.",
          "ln-cap2": "The diagnostic form with doses up to 2, 1 + z / (1 + z): the same placement with a range twice as "
                     "wide, to show whether BCA's cap is what limits it.",
          "knee@1.5": "BCA's cap: a caution multiplier never goes above 1.5.",
          "knee@2": "The cap of the diagnostic form: 2."},
    aria="Caution multiplier against the row's over-preference divided by its mean: the over-preference-placed dose in BCA's form rises from 1 toward 1.5, the diagnostic capped at 2 from 1 toward 2.")

pw = ME["power"]["table"]
FRAG["POWER_TABLE"] = table(["Spread σ (standard deviation, points)", "Confirmation seeds N"],
                            [[f"{r['sd_points']:g}", str(r["N"])] for r in pw], numeric={0, 1}, tips=[
    "An example spread σ, in points: how much a paired difference moves from one seed to the next. The power check "
    "would measure it on the gate seeds; it did not run, so these rows are examples, not measurements.",
    "How many confirmation seeds a fair test of BCA would need at that spread: the formula above, rounded up, and at "
    "least 10. Above 20 the test stops as underpowered."])


# ------------------------------------------------------------------------------------------------ reference scores

def ref_table(world):
    R = RR["returns"][world]
    ctrls = CTRL if world == "stall" else ["recording", "rising_thrust", "stall_aware", "normal_driving_only", "no_throttle"]
    rows = []
    for c in ctrls:
        cells = [CTRL_LABEL[c]]
        for s in SIT:
            v = R[s][c]
            cells.append(f"{fx(v['J'], 2)} <span class=\"pts\">({fx(v['points'], 1)})</span>")
        anchor = (c == "recording") or (c == ("stall_aware" if world == "stall" else "rising_thrust"))
        rows.append(("anchor" if anchor else "", cells))
    se = [f"±{fx(R[s]['stall_aware' if world == 'stall' else 'rising_thrust']['J_se'], 2)}" for s in SIT]
    rows.append(("sub", [f"standard error, {'stall-aware' if world == 'stall' else 'rising-thrust'}"] + se))
    top = "stall-aware" if world == "stall" else "rising-thrust"
    wname = "stalling world" if world == "stall" else "no-stall comparison world"
    return table(["Controller"] + [SIT_LABEL[s] for s in SIT], rows, cls="steps", numeric={1, 2, 3, 4}, minw=680,
                 stack=True, tips=[
        f"The fixed controller scored in this row. The last row gives the standard error of the {top} controller's "
        f"average: how much it would typically move with another set of 1,000 starts."] + [
        f"In the {wname}: the controller's exact average 100-step return over the 1,000 starts of the "
        f"{SIT_LABEL[s].lower()} evaluation bank, with its points in brackets (0 is the recording controller, 100 the "
        f"{top} controller)." for s in SIT])


FRAG["REF_STALL"] = ref_table("stall")
FRAG["REF_NOSTALL"] = ref_table("no_stall")
rows = []
for s in SIT:
    R = RR["returns"]["stall"][s]
    rows.append([SIT_LABEL[s], fx(R["recording"]["J_discounted"], 2), fx(R["stall_aware"]["J_discounted"], 2),
                 fx(R["rising_thrust"]["J_discounted"], 2), fx(R["normal_driving_only"]["J_discounted"], 2),
                 fx(R["stall_aware"]["J"] - R["recording"]["J"], 1)])
FRAG["REF_DISC"] = table(["Situation (stalling world)", "Recording, discounted", "Stall-aware, discounted",
                          "Rising-thrust, discounted", "Normal driving only, discounted", "Stall-aware minus recording (100-step)"],
                         rows, numeric={1, 2, 3, 4, 5}, minw=720, stack=True, tips=[
    "The situation of the evaluation bank, in the stalling world."] + [
    f"The {c} controller's average discounted return: each step's reward counts 0.98 to the power of its step number, "
    f"so later steps count less." for c in ("recording", "stall-aware", "rising-thrust", "normal-driving-only")] + [
    "How much higher the stall-aware controller's plain 100-step average return is than the recording controller's."])


# ------------------------------------------------------------------------------------------------ world card (nine checks)

PS0 = WC["per_seed"][0]
assert PS0["seed"] == 900 and WC["passed"] and all(p["passed"] for p in WC["per_seed"])
RG = WC["range"]


def span(key, f):
    lo, hi = RG[key]
    return f"{f(lo)}–{f(hi)}"


p2, p3 = (lambda v: pc(v, 2)), (lambda v: pc(v, 3))
t7 = " / ".join(fx(PS0["T7"][s], 1) for s in SIT)
FRAG["WORLDCARD_TABLE"] = table(
    ["Check", "Rule (from the pre-registration)", "Seed 900", "Seeds 900–904, lowest–highest"],
    [["The removal takes the full-throttle commands", "the densest-15% removal takes at least 99% of the full-throttle commands, and its ball is centred at "
            "thruster 1 ≤ −0.6", f"{p2(PS0['T1_removed'])}, {fx(PS0['T1_centre'], 4)}",
      f"{span('T1_removed', p2)}; {fx(RG['T1_centre'][0], 4)} to {fx(RG['T1_centre'][1], 4)}"],
     ["Almost no data past 80% are left", "kept rows past 80% at most 0.2%; random removal keeps at least 80% of the past-80% rows",
      f"{p3(PS0['T2_kept_past80'])}; {p2(PS0['T2_random_keeps'])}", f"{span('T2_kept_past80', p3)}; {span('T2_random_keeps', p2)}"],
     ["The removed ball is the only dense patch", "the best competing 15% ball needs at least 1.2 times the radius", fx(PS0["T3"], 3), span("T3", lambda v: fx(v, 3))],
     ["The kept data cannot tell the two worlds apart", "at least 99.8% of kept rows follow the identical rule in both worlds", p3(PS0["T4"]), span("T4", p3)],
     ["Enough of the full-throttle region is kept", "at least 40% of full-throttle-region rows kept", p2(PS0["T5"]), span("T5", p2)],
     ["Normal driving alone leaves room to improve", "normal driving only scores at most 85 points in the recorded conditions", fx(PS0["T6"], 1),
      f"{fx(RG['T6'][0], 1)} on every seed (no log needed)"],
     ["Stopping at 80% beats the recording controller", "the stall-aware controller's 100-step return beats the recording controller's by at least 20 in every "
            "situation (recorded / strong current / big waves / big starting pushes)", t7, "the same on every seed"],
     ["The critic's own noise hides no real differences", "the spread of the exact value under TD3's target noise is at most its spread under the recording "
            "controller's wobble: the noise TD3 adds to its own target is smaller than the real differences in value "
            "between recorded commands, so the critic can still learn them",
      f"{fx(PS0['T8_smoothing'], 3)} against {fx(PS0['T8_wobble'], 3)}",
      f"{span('T8_smoothing', lambda v: fx(v, 3))} against {span('T8_wobble', lambda v: fx(v, 3))}"],
     ["The forecast harm is large enough", "forecast harm in strong current at least 5 points with the bending critic and 20 with the edge-slope "
            "critic, at the seed's own critic's pull μ (Methods)",
      f"{fx(PS0['T9_bend'], 1)} / {fx(PS0['T9_edge'], 1)} (μ = {fx(PS0['kappa'], 3)})",
      f"{span('T9_bend', lambda v: fx(v, 1))} / {span('T9_edge', lambda v: fx(v, 1))} (μ {span('kappa', lambda v: fx(v, 3))})"]],
    numeric={2, 3}, minw=720, stack=True, tips=[
        "The world card's check, named by what it asks. The world card is the nine checks on the world itself that "
        "need no learning, run before the pre-registration.",
        "The pass rule, as fixed in the pre-registration before any result existed.",
        "The value on design seed 900, whose log this page uses.",
        "The lowest and highest value over the five design seeds 900 to 904. Every seed passed every check."])
# the page's own seed-900 numbers and the world card's seed 900 must be the same log
assert abs(PS0["T4"] - DS["gap_removal"]["identical_rule_share_kept"]) < 1e-6
assert abs(PS0["T1_removed"] - DS["gap_removal"]["full_throttle_commands_removed_share"]) < 1e-6
assert PS0["T2_kept_past80_rows"] == DS["gap_removal"]["kept_rows_past_80"]
assert all(PS0["T4"] <= p["T4"] and PS0["T2_kept_past80"] >= p["T2_kept_past80"] and PS0["T1_removed"] <= p["T1_removed"]
           for p in WC["per_seed"])


# ------------------------------------------------------------------------------------------------ arguments and names

def wbr_id(s):
    """Allow line breaks only after '/' and before '-' in a long id."""
    return esc(s).replace("/", "/<wbr>").replace("-", "<wbr>-")


rows = []
for e in E["registered_ids"]:
    rows.append([f"<code>{wbr_id(e['id'])}</code>", e["world_plain"], e["situation_plain"],
                 f"<code>stall={e['kwargs']['stall']},</code> <code>start=\"{e['kwargs']['start']}\"</code>"])
FRAG["IDS_TABLE"] = table(["Id (max_episode_steps 100)", "World", "Situation (start law: Starting State)", "Arguments"],
                          rows, stack=True, tips=[
    "The name that gym.make loads. Every id cuts an episode off after 100 steps.",
    "The world the id builds: the stalling world, or the no-stall comparison world, in which push keeps rising up to "
    "full throttle.",
    "The start law the id's default reset draws from.",
    "The arguments of ThrusterChainEnv that the id fixes."])

se_ = E["step_example"]
_r = E["render_example"]["after_one_step"]
_cut = _r.index(" floats=")
_make_id = re.search(r"make\('([^']+)'\)", E["make_line"]).group(1)
FRAG["STEP_EXAMPLE"] = (
    "<figure class=\"example\" id=\"step-example\" aria-label=\"Worked example: one step from reset(seed=0)\">"
    "<p><span class=\"tag acc\">worked example</span> These are not made-up numbers. The block is the real output "
    f"of the environment stepped once from <code>reset(seed=0)</code> in the recorded conditions (<code>{_make_id}"
    "</code>, with <code>render_mode=\"ansi\"</code>). The command given is the stall-aware controller's command for "
    "that starting state. The page's data script, <code>make_data.py</code>, ran these same steps on the "
    "environment's own code and recorded what came back. Another seed or another command gives other numbers.</p>"
    f"<pre class=\"code\">import numpy as np\n"
    f"from experiments.shift import recorder, thruster_chain   # import first: it puts gymnasium on the path and registers the ids\n"
    f"import gymnasium as gym\n\n"
    f"env = gym.make(\"{_make_id}\", render_mode=\"ansi\")\n"
    f"obs, info = env.reset(seed=0)\n"
    f"action = recorder.stall_aware()(info[\"state\"]).astype(np.float32)   # the stall-aware command\n"
    f"# action  = ({', '.join(se_['command'])})\n"
    f"obs, reward, terminated, truncated, info = env.step(action)\n"
                        f"# reward  = {se_['reward']}\n"
                        f"# info    = {{'t': {se_['info']['t']}, 'push': {se_['info']['push']},\n"
                        f"#            'thrust': {se_['info']['thrust']},\n"
                        f"#            'stalled': {[bool(x) for x in se_['info']['stalled']]},\n"
                        f"#            'command': {se_['info']['command']},\n"
                        f"#            'state': (13 float64)}}\n"
                        f"env.render()   # render_mode=\"ansi\"; one line, shown here in two\n"
                        f"# {esc(_r[:_cut])}\n"
                        f"#    {esc(_r[_cut + 1:])}</pre></figure>")

RENAME = {"removal of the densest 15% (the gap)": "the densest-15% removal and its removed ball",
          "the textbook gain": "the textbook rule K",
          "removed share m(s)": "the removed share", "the evaluation banks": "the evaluation banks (1,000 starts each)"}
def wbr_code(h):
    """Allow line breaks after '/' and after a '.' that starts a name, outside tags, so long code names wrap on phones."""
    out, intag = [], False
    for i, ch in enumerate(h):
        out.append(ch)
        if ch == "<":
            intag = True
        elif ch == ">":
            intag = False
        elif not intag and (ch == "/" or (ch == "." and i + 1 < len(h) and h[i + 1].isalpha())):
            out.append("<wbr>")
    return "".join(out)


rows = [[esc(RENAME.get(c["plain"], c["plain"])), f"<code>{esc(c['code'])}</code>"] for c in D["code_names"]]
rows += [["the four situations' registered ids", "<code>BCAShift/ThrusterChain{,-StrongCurrent,-BigWaves,-BigKicks}-v0</code>"],
         ["the no-stall comparison world's ids", "<code>BCAShift/ThrusterChainNoStall{,-StrongCurrent,-BigWaves,-BigKicks}-v0</code>"],
         ["the recorded fields, in the order of The recorded data",
          "<code>" + ", ".join(k for _, k, _ in FIELDS) + "</code> (D4RL's names)"],
         ["the three splits: training, calibration, held out", "<code>infos/split</code> = 0, 1, 2 (2 is named "
          "<code>evaluation</code> in the code)"],
         ["the design and the environment's documentation", "<code>experiments/shift/SHIFT_ENV.md</code>"],
         ["design seeds 900–904 (world card), 900–902 (trigger), 905–909 (retune)",
          "<code>stage0.CALIBRATION_SEEDS, stage0.BISECTION_SEEDS, stage0.RETUNE_SEEDS</code>"],
         ["the world card (its nine checks) and the forecast", "<code>stage0</code>, its nine checks keyed <code>T1</code>"
          " to <code>T9</code> in <code>stage0.COMPUTED</code>; its run before the pre-registration: "
          "<code>runs/shift/pre/stage0_base.json</code>"],
         ["the critic's pull μ (0.58), bending and edge-slope critics", "<code>stage0.CRITIC_K_REFERENCE</code> "
          "(called the critic weight k there); stand-ins <code>'bend'</code> and <code>'edge'</code>"],
         ["the calibration set (1,000 rows, 5 per calibration episode)", "the bank rows: <code>meta['bank_rows']</code>, "
          "<code>dataset.Log.bank_rows()</code> (drawn by <code>calibration/bank.stratified_bank</code>)"],
         ["over-preference-placed caution (the room check) and caution capped at 2", "room arms <code>harm_placed</code> and "
          "<code>cap2</code> (<code>learners.harm_doses</code>)"],
         ["full-throttle pattern P and its 80% cap P<sub>cap</sub>", "<code>c_hi</code>, <code>c_cap</code> in the frozen pools"],
         ["over-preference Δ", "<code>D</code> in <code>truth.over_preference</code>"],
         ["reversed caution", "room arm <code>reversed</code> (<code>learners.reverse_ranks</code>)"],
         ["BCA's band multiplier β and running unit η", "the threshold of <code>calibration/wbcp.py</code> and the "
          "residual-scale unit of <code>calibration/dose.py</code>"],
         ["the three detection checks (ranking, shift-specificity, beyond-noise); band on the frozen plain critic",
          "<code>acceptance.detect_seed</code> and <code>acceptance.PASS_MARKS[\"B1\"]</code>; the checks are keyed "
          "<code>D1</code> (ranking), <code>D2</code> (shift-specificity) and <code>D3</code> (beyond-noise) in "
          "<code>detection</code>; <code>learners.bca_band</code>"],
         ["the three actor checks (no loss, beats copying, no harm without the shift)",
          "<code>P2</code> (no loss), <code>P3</code> (beats copying) and <code>P4</code> (no harm without the shift) "
          "in <code>acceptance</code>, with their marks in <code>acceptance.PASS_MARKS[\"B2\"]</code>"],
         ["the BCA test's beats-plain and placement checks", "<code>level1_min</code> in "
          "<code>acceptance.PASS_MARKS[\"B2\"]</code>; Holm families <code>policy_level_1</code> and "
          "<code>policy_level_2</code>"],
         ["the four harm checks, stop rule, BCA test, verdict",
          "<code>acceptance.PASS_MARKS</code>, <code>MARK_TEXT</code>, <code>VERDICTS</code>, <code>VERDICT_ORDER</code>; "
          "the harm checks are keyed <code>H1</code> (the critic is wrong where data are missing), <code>H2</code> "
          "(acting on the error costs return), <code>H3</code> (the room check) and <code>H4</code> (the power check)"]]
FRAG["CODE_NAMES"] = table(["Plain name on this page", "Name in the code"], [[a, wbr_code(b)] for a, b in rows], stack=True,
                           tips=["The name this page uses, in plain words.",
                                 "The matching name in the repository's code or files, for readers who look it up."])


# ------------------------------------------------------------------------------------------------ the task overview

O = D["overview"]
_offsets = [f["position"] for f in O["water_alone"]["steady_offset_per_unit_current"]]  # all negative: downstream
_periods = [s["period_steps"] for s in O["swings"]["list"]]
_kp = {p["throttle"]: p["thrust_stall"] for p in E["thrust_curve"]["key_points"] if p["throttle"] in (0.8, 0.9, 1.0)}
assert len(_kp) == 3
assert all(p < 0 for p in _offsets) and _offsets == sorted(_offsets)  # float 1 is carried furthest


def _t(x, y, s, cls, size, anchor="start", mono=False):
    fam = ' font-family="var(--mono)"' if mono else ""
    return (f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{fam}>'
            f'{esc(s)}</text>')


# The task diagram is also the stage of the demo episode: one real episode of the stall-aware controller, the page's
# 100-point reference, from the live panel's own rollouts (live_panel/rollouts, world|situation|controller). The build
# draws step 0 of the recorded-conditions episode; the page's script moves the floats, the thrusters' arrows, the
# water's arrows and the throttle marker step by step. Geometry the script needs travels as data-* attributes.
DEMO_SITS = SIT
DEMO_KEY = "stall|{}|stall_aware"
_demo = {s: D["live_panel"]["rollouts"][DEMO_KEY.format(s)] for s in DEMO_SITS}
_dw = D["live_panel"]["water"]
_d0 = _demo["recorded"]
D["demo0"] = dict(push=_dw["recorded"]["push"][0], c=_dw["recorded"]["current"],
                  w1=_d0["states"][0][11], a=_d0["commands"][0], f=_d0["thrust"][0], r=_d0["rewards"][0],
                  g=_d0["running_return"][0], ret=_d0["summary"]["return_env"],
                  bank=RR["returns"]["stall"]["recorded"]["stall_aware"]["J"])
assert abs(_dw["recorded"]["push"][0] - (_d0["states"][0][10] + _d0["states"][0][11])) < 1e-3
for _s in DEMO_SITS:  # the demo shows the stall-aware controller, which never passes 80%
    assert _demo[_s]["summary"]["steps_past_80"] == 0 and len(_demo[_s]["commands"]) == 100


_TIP_CUR = esc("Solid part: the current c, a steady push downstream on every float, drawn at the start of the "
                "episode and fixed for all its 100 steps.")
_TIP_WAVE = esc("Dashed part: the wave's push w₁, which rises and falls every "
                f"{fx(E['dynamics']['wave_period_steps'], 2)} steps and is the same on every float. Added to the "
                "current it gives the water's push c + w₁.")
_TIP_PUSH = esc("The water's push c + w₁ at the step shown, the same on every float: the current plus the wave. Above "
                f"{TRIG} the state is in the full-throttle region.")
_TIP_TRI = esc("Thruster 1's throttle at the step shown: the triangle and the filled part of the bar. The "
               "stall-aware controller never takes it past 80%.")


def _tip_float(j):
    who = (f"thruster {j + 1}" if j == 0 else f"thruster {j}" if j == 4 else f"thrusters {j} and {j + 1}")
    return (f"Float {j + 1} at the step shown. Its height is its position number: below the dashed rest line means "
            f"carried downstream. Left alone it swings with a period of about {fx(_periods[j], 0)} steps; {who} "
            "can push it.")


def _tip_ghost(j):
    return (f"Where the water alone would leave float {j + 1} in this episode's current, with every thruster off. "
            "Float 1 is carried furthest and float 5 least.")


def _tip_thr(k):
    return (f"Thruster {k + 1}, between floats {k + 1} and {k + 2}: it pushes both. Its arrow is the push it really "
            "gives at the step shown, up for upstream; past 80% throttle the propeller stalls and the push falls.")


def task_diagram(narrow):
    """The channel as the live panel draws it: water flows down the page; a float below the dashed rest line has a
    negative position number (downstream); a thruster's arrow points up for a push upstream. Drawn at step 0 of the
    demo episode; the dashed circles are where this episode's current would leave each float with no throttle
    (overview water_alone/steady_offset_per_unit_current times the current), to scale."""
    T = E["dynamics"]["wave_period_steps"]
    if narrow:
        W, xs, fs, ft = 360, [36, 108, 180, 252, 324], 12.5, 13.5
        a0, spush = 62, 38                          # water arrows: start, px per unit of push
        cx0, cy0, cw, ch = 4, 114, 352, 214          # channel
        rest, sp, rad, bw, bh, alen = 172, 40, 14, 18, 12, 26
        bar = dict(title=394, y=404, x0=16, x1=344, tick=438, push=454, note=470)
        H = 478
    else:
        W, xs, fs, ft = 720, [110, 230, 350, 470, 590], 12.5, 14
        a0, spush = 46, 40
        cx0, cy0, cw, ch = 20, 100, 680, 232
        rest, sp, rad, bw, bh, alen = 162, 40, 17, 22, 14, 34
        bar = dict(title=398, y=408, x0=20, x1=700, tick=442, push=458, note=458)
        H = 470
    cbot = cy0 + ch
    s0, a, f = _d0["states"][0], _d0["commands"][0], _d0["thrust"][0]
    c, w1 = s0[10], s0[11]
    ylo, yhi = cy0 + rad + 2, cbot - rad - 24        # floats stay inside the channel, above the thruster labels
    fy = [min(yhi, max(ylo, rest - sp * s0[2 * j])) for j in range(5)]
    gy = [min(yhi, max(ylo, rest - sp * o * c)) for o in _offsets]
    tx = [(xs[k] + xs[k + 1]) / 2 for k in range(4)]
    ty = [(fy[k] + fy[k + 1]) / 2 for k in range(4)]
    geo = dict(rest=rest, sp=sp, a0=a0, spush=spush, alen=alen, bh=bh, ylo=ylo, yhi=yhi, bx0=bar["x0"],
               bx1=bar["x1"], by=bar["y"])
    o = [f'<svg class="dg {"dg-narrow" if narrow else "dg-wide"}" viewBox="0 0 {W} {H}" role="img" '
         + " ".join(f'data-{k}="{v}"' for k, v in geo.items())
         + ' data-tip="' + esc("The channel, playing one real episode of the stall-aware controller step by step. "
                               "Point at, or tap, a float, a thruster, a dashed circle, the water's arrows or the "
                               "throttle bar for what each one shows.") + '"'
         + ' aria-label="'
         + esc("Diagram of the channel, which also plays the demo episode. Water flows down the page and pushes all "
               "five floats with the same push, a steady current plus a wave that rises and falls; it pushes "
               "downstream except when the wave takes away more than the current gives. Dashed circles show where "
               "the episode's current would leave each float with the thrusters off, float 1 furthest and float 5 "
               "nearest; the solid circles are the floats at the current step of the episode. Four thrusters sit "
               "between neighbouring floats; each arrow shows the push its thruster gives, up for upstream. A throttle "
               "bar shows that up to 80% the push equals the throttle and past 80% the propeller stalls: push "
               f"{fx(_kp[0.9], 2)} at 90% and {fx(_kp[1.0], 1)} at full throttle; a triangle marks thruster 1's "
               "throttle at the current step.") + '">']
    # the water's push, the same on every float: a solid shaft for the current and a dashed part for the wave
    if narrow:
        o.append(_t(10, 18, "The water's push c + w₁", "dg-t", ft))
        o.append(_t(10, 36, "solid: current c, steady all episode", "dg-m", fs))
        o.append(_t(10, 53, f"dashed: wave w₁, rises and falls every {fx(T, 2)} steps", "dg-m", fs))
    else:
        o.append(_t(20, 20, "The water's push c + w₁, the same on every float", "dg-t", ft))
        o.append(_t(700, 20, "solid: current c, steady all episode", "dg-m", fs, "end"))
        o.append(_t(700, 37, f"dashed: wave w₁, rises and falls every {fx(T, 2)} steps", "dg-m", fs, "end"))
    yc, yp = a0 + spush * c, a0 + spush * (c + w1)
    for x in xs:
        o.append(f'<line class="dg-push" data-d="cur" x1="{x}" x2="{x}" y1="{a0}" y2="{min(yc, yp):.1f}" '
                 f'stroke-width="4" data-tip="{_TIP_CUR}"/>')
        o.append(f'<line class="{"dg-wavepart" if w1 >= 0 else "dg-waveback"}" data-d="wave" x1="{x}" x2="{x}" '
                 f'y1="{yc:.1f}" y2="{yp:.1f}" stroke-width="4" data-tip="{_TIP_WAVE}"/>')
        d = 1 if c + w1 >= 0 else -1
        o.append(f'<polygon class="dg-push" data-d="head" points="{x - 6},{yp - d * 9:.1f} {x + 6},{yp - d * 9:.1f} '
                 f'{x},{yp:.1f}" data-tip="{_TIP_PUSH}"/>')
    # the channel and the rest line; labels drawn on the channel use the darker ink (dg-t), which keeps 4.5:1 there
    o.append(f'<rect class="dg-chan" x="{cx0}" y="{cy0}" width="{cw}" height="{ch}" rx="10"/>')
    o.append(f'<line class="dg-rest" x1="{cx0 + 6}" x2="{cx0 + cw - 6}" y1="{rest}" y2="{rest}"/>')
    o.append(_t(cx0 + 6, rest - 5, "rest", "dg-t", fs))
    if narrow:
        o.append(_t(cx0 + cw - 6, cy0 + 16, "downstream ↓", "dg-t", fs, "end"))
    else:
        o.append(_t(cx0 + cw - 8, cy0 + 18, "water flows downstream ↓", "dg-t", fs, "end"))
    # where the water alone would leave each float in this episode's current (dashed), to scale
    for j, (x, y) in enumerate(zip(xs, gy)):
        o.append(f'<circle class="dg-ghost" data-g="{j}" cx="{x}" cy="{y:.1f}" r="{rad}" data-tip="{esc(_tip_ghost(j))}"/>')
    # each thruster sits between two floats and pushes both: a line joins the floats through the thruster
    for k in range(4):
        o.append(f'<line class="dg-thrline" data-l="{k}" x1="{xs[k]}" y1="{fy[k]:.1f}" x2="{xs[k + 1]}" y2="{fy[k + 1]:.1f}"/>')
    for j, (x, y) in enumerate(zip(xs, fy)):
        o.append(f'<g class="dg-fl" data-f="{j}" transform="translate(0 {y - rest:.1f})" data-tip="{esc(_tip_float(j))}">'
                 f'<circle class="dg-float" cx="{x}" cy="{rest}" r="{rad}"/>'
                 + _t(x, rest + 5, str(j + 1), "dg-n", 14 if not narrow else 13, "middle") + "</g>")
    for k in range(4):
        x, y = tx[k], ty[k]
        ln = abs(f[k]) / 0.8 * alen
        d = -1 if f[k] < 0 else 1                    # up for a push upstream
        y0_ = y - bh / 2 if d < 0 else y + bh / 2
        tip = y0_ + d * ln
        o.append(f'<g class="dg-th" data-t="{k}" data-tip="{esc(_tip_thr(k))}">'
                 f'<rect class="dg-thr" x="{x - bw / 2}" y="{y - bh / 2:.1f}" width="{bw}" height="{bh}" rx="3"/>'
                 f'<line class="dg-thrarrow" x1="{x}" x2="{x}" y1="{y0_:.1f}" y2="{tip - d * 7:.1f}" stroke-width="2.4"/>'
                 f'<polygon class="dg-thrarrow" points="{x - 5},{tip - d * 8:.1f} {x + 5},{tip - d * 8:.1f} {x},{tip:.1f}"/>'
                 "</g>")
        o.append(_t(x, cbot - 9, f"thruster {k + 1}", "dg-t", fs, "middle"))
    if narrow:
        o.append(_t(10, cbot + 20, "each float's own swing, in steps:", "dg-t", fs))
        for x, p in zip(xs, _periods):
            o.append(_t(x, cbot + 38, fx(p, 0), "dg-t", fs + 0.5, "middle", mono=True))
    else:
        reach = ["only thruster 1", "two thrusters", "two thrusters", "two thrusters", "only thruster 4"]
        for x, txt_, p in zip(xs, reach, _periods):
            o.append(_t(x, cbot + 18, txt_, "dg-t", fs, "middle"))
            o.append(_t(x, cbot + 36, f"swing: {fx(p, 0)} steps", "dg-t", fs, "middle"))
    # one thruster's throttle, from off to full throttle: push equals throttle up to 80%, then the stall; the triangle
    # marks thruster 1's throttle at the current step
    b = bar
    xk = lambda u: b["x0"] + (b["x1"] - b["x0"]) * u
    o.append(_t(b["x0"], b["title"], "Throttle of one thruster, off to full throttle; the triangle is thruster 1 now"
                if not narrow else "Throttle: off to full; ▼ thruster 1 now", "dg-t", ft - 0.5))
    o.append(f'<rect class="dg-ok" x="{b["x0"]}" y="{b["y"]}" width="{xk(0.8) - b["x0"]:.1f}" height="16" '
             f'data-tip="Up to 80% throttle, a thruster pushes exactly as hard as its throttle."/>')
    o.append(f'<rect class="dg-stall" x="{xk(0.8):.1f}" y="{b["y"]}" width="{b["x1"] - xk(0.8):.1f}" height="16" '
             f'data-tip="Past 80% throttle the propeller stalls: push {fx(_kp[0.9], 2)} at 90% and {fx(_kp[1.0], 1)} at '
             f'full throttle, while the throttle cost keeps rising."/>')
    u1 = min(1.0, abs(a[0]))
    o.append(f'<rect class="dg-fill" data-m="fill" x="{b["x0"]}" y="{b["y"]}" width="{xk(u1) - b["x0"]:.1f}" height="16" '
             f'data-tip="{_TIP_TRI}"/>')
    o.append(_t((b["x0"] + xk(0.8)) / 2, b["y"] + 12.5, "push = throttle", "dg-t", fs, "middle"))
    o.append(_t((xk(0.8) + b["x1"]) / 2, b["y"] + 12.5, "stall", "dg-w", fs, "middle"))
    o.append(f'<line class="dg-limit" x1="{xk(0.8):.1f}" x2="{xk(0.8):.1f}" y1="{b["y"] - 6}" y2="{b["y"] + 22}"/>')
    o.append(f'<polygon class="dg-mark" data-m="tri" points="{xk(u1) - 6:.1f},{b["y"] - 9} {xk(u1) + 6:.1f},{b["y"] - 9} '
             f'{xk(u1):.1f},{b["y"] - 1}" data-tip="{_TIP_TRI}"/>')
    marks = [(0.0, "0", "", "start"), (0.8, "80%", f"push {fx(_kp[0.8], 1)}", "middle"),
             (1.0, "100%", fx(_kp[1.0], 1), "end")]
    if not narrow:
        marks.insert(2, (0.9, "90%", fx(_kp[0.9], 2), "middle"))
    for u, top, bottom, anc in marks:
        o.append(f'<line class="dg-tick" x1="{xk(u):.1f}" x2="{xk(u):.1f}" y1="{b["y"] + 16}" y2="{b["y"] + 21}"/>')
        o.append(_t(xk(u), b["tick"], top, "dg-w" if u == 0.8 else "dg-m", fs, anc))
        if bottom:
            o.append(_t(xk(u), b["push"], bottom, "dg-t", fs, anc, mono=True))
    o.append(_t(b["x0"], b["note"], "past 80%: costs more, pushes less" if narrow else
                "past 80% the throttle costs more and pushes less", "dg-m", fs))
    o.append("</svg>")
    return "".join(o)


FRAG["TASK_DIAGRAM"] = task_diagram(False) + "\n" + task_diagram(True)

# thruster 1's throttle by the water's push, state by state, over every state of the four evaluation banks
TP = O["controllers"]["thruster1_by_push"]
_bin = TP[0]["push_to"] - TP[0]["push_from"]
assert all(abs((r["push_to"] - r["push_from"]) - _bin) < 1e-9 for r in TP)
_mid = [(r["push_from"] + r["push_to"]) / 2 for r in TP]


def _tip(r, who):
    rng = f"water's push {fx(r['push_from'], 1)} to {fx(r['push_to'], 1)} ({ig(r['states'])} states): "
    if who == "sa":
        return rng + (f"stall-aware thruster 1 {fx(r['stall_aware_thruster1'], 2)}, at its 80% limit on "
                      f"{pc(r['stall_aware_thruster1_at_cap'], 0)}")
    if who == "rt":
        return rng + (f"rising-thrust thruster 1 throttle {fx(r['rising_thrust_thruster1'], 2)}, past 80% on "
                      f"{pc(r['rising_thrust_thruster1_past_80'], 0)}")
    return rng + f"rising-thrust thruster 1 push {fx(r['rising_thrust_thruster1_push'], 2)}"


_pts = []
for x, r in zip(_mid, TP):
    _pts += [(x, r["rising_thrust_thruster1_push"], "pt-rtpush", 3.2, _tip(r, "rtp")),
             (x, r["rising_thrust_thruster1"], "pt-rt", 3.2, _tip(r, "rt")),
             (x, r["stall_aware_thruster1"], "pt-sa", 3.2, _tip(r, "sa"))]
_last = _mid[-1]
FRAG["TASK_CHART"] = svg_line_chart(
    [("ln-rtpush", [(x, r["rising_thrust_thruster1_push"]) for x, r in zip(_mid, TP)]),
     ("ln-rt", [(x, r["rising_thrust_thruster1"]) for x, r in zip(_mid, TP)]),
     ("ln-sa", [(x, r["stall_aware_thruster1"]) for x, r in zip(_mid, TP)])],
    (-0.6, 1.6), (0.4, -1.05), [-0.5, 0, 0.5, 1.0, 1.5], [0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, -1.0],
    "water's push c + w₁ at the state", "thruster 1 (up: harder upstream)", w=440, h=290,
    hlines=[(-0.8, "knee", "80%: past this the propeller stalls", "start")], points=_pts,
    labels=[(_last, TP[-1]["rising_thrust_thruster1"], "lab", "rising-thrust throttle", "end", 0, -9),
            (_last, TP[-1]["stall_aware_thruster1"], "lab", "stall-aware", "end", 0, 17),
            (_last, TP[-1]["rising_thrust_thruster1_push"], "lab", "rising-thrust push", "end", 0, 19)],
    xfmt=lambda v: fx(v, 1), yfmt=lambda v: fx(v, 1), pad=(58, 14, 18, 40),
    tips={"ln-sa": "The stall-aware controller's average thruster 1 throttle in each band of the water's push: it pushes "
                   "harder upstream as the water pushes harder, and stops at the 80% line.",
          "ln-rt": "The rising-thrust controller's average thruster 1 throttle: it keeps asking for more, up to full "
                   "throttle.",
          "ln-rtpush": "The push the rising-thrust controller's thruster 1 really gives: once it is past 80% the "
                       "propeller stalls and the push falls back.",
          "knee": "The 80% line: past it a propeller stalls in the stalling world."},
    aria="Thruster 1's average throttle against the water's push at the state, over every state of the four "
         "evaluation banks. The stall-aware controller pushes harder upstream as the water pushes harder and stops at "
         "the 80% line. The rising-thrust controller keeps asking for more, up to full throttle, but the push it really "
         "gives falls back toward 0.5 once it stalls.")
FRAG["TASK_PUSH_TABLE"] = table(
    ["Water's push c + w₁", "States", "Stall-aware: thruster 1", "Stall-aware: at its 80% limit",
     "Rising-thrust: thruster 1 throttle", "Rising-thrust: past 80%", "Rising-thrust: thruster 1's push"],
    [[f"{fx(r['push_from'], 1)} to {fx(r['push_to'], 1)}", ig(r["states"]), fx(r["stall_aware_thruster1"], 2),
      pc(r["stall_aware_thruster1_at_cap"], 0), fx(r["rising_thrust_thruster1"], 2),
      pc(r["rising_thrust_thruster1_past_80"], 0), fx(r["rising_thrust_thruster1_push"], 2)] for r in TP],
    numeric={1, 2, 3, 4, 5, 6}, minw=700, stack=True, keep1=True, tips=[
        "A band of the water's push c + w₁ at the state, {{derived/push_bin_width#f1}} wide. Each row averages every "
        "state of the four evaluation banks (stalling world) whose push falls in the band.",
        "How many states fall in the band: one per start and step of the four banks. The water's push at a given start "
        "and step is the same for every controller, because no thruster changes the water.",
        "The stall-aware controller's average thruster 1 throttle in the band. Negative is upstream; it never goes past "
        "−0.8.",
        "The share of the band's states at which the stall-aware controller holds thruster 1 at its 80% limit.",
        "The rising-thrust controller's average thruster 1 throttle in the band: the same textbook command, allowed up "
        "to full throttle.",
        "The share of the band's states at which the rising-thrust controller's thruster 1 is past 80%, where it stalls.",
        "The push the rising-thrust controller's thruster 1 really gives, on average: less than its throttle once it "
        "stalls."])

# holding against a steady current (no wave), current by current, once the floats have settled
SCT = O["controllers"]["steady_current"]
_sct = SCT["table"]
assert abs(SCT["currents"][4] - 0.9) < 1e-9 and all(r["settled"] for r in _sct["stall_aware"])


def _fc(r, d=3):
    return fx(r["float_cost"], d) + ("" if r["settled"] else '<sup aria-label="never settles">†</sup>')


# two header rows: the controller (and world) over its columns, then short column names. Each short name carries the
# full name in data-label, which the page's script copies onto the cells for the phone-width card layout.
_groups = [("", 1, ""), ("No throttle", 1, "All thrusters off: the water acting alone."),
           ("Stall-aware", 2, "The stall-aware controller: the textbook command capped at 80%, so it never stalls."),
           ("Rising-thrust, stalling world", 3, "The rising-thrust controller, the same textbook command allowed up to "
            "full throttle, in the stalling world, where the propeller stalls past 80%."),
           ("Rising-thrust, no-stall world", 2, "The rising-thrust controller in the no-stall comparison world, where "
            "push keeps rising up to full throttle.")]
_settled = "once the floats have settled: the sum of their squared distances from rest"
_cols = [("Current c", "Current c", "The strength of the steady current, with no wave. Each row starts from rest and "
                                    "holds against this current until the floats settle."),
         ("float cost", "No throttle: float cost", f"The floats' cost in each state with every thruster off, {_settled}."),
         ("thruster 1", "Stall-aware: thruster 1", "The stall-aware controller's settled thruster 1 throttle. Negative "
                                                   "is upstream; it stops at −0.8."),
         ("float cost", "Stall-aware: float cost", f"The floats' cost in each state under the stall-aware controller, "
                                                   f"{_settled}."),
         ("thruster 1 throttle", "Rising-thrust, stalling world: thruster 1 throttle",
          "The rising-thrust controller's settled thruster 1 throttle in the stalling world. It goes to full throttle "
          "(−1) once the current needs more than 80%."),
         ("thruster 1's push", "Rising-thrust, stalling world: thruster 1's push",
          f"The push that throttle really gives in the stalling world: only {STALL100} at full throttle, because the "
          f"propeller stalls."),
         ("float cost", "Rising-thrust, stalling world: float cost",
          f"The floats' cost in each state under the rising-thrust controller in the stalling world, {_settled}. † marks "
          f"a current at which it never settles; the row then averages the end of a long run."),
         ("thruster 1", "Rising-thrust, no-stall world: thruster 1",
          "The rising-thrust controller's settled thruster 1 throttle in the no-stall comparison world, where push "
          "equals throttle."),
         ("float cost", "Rising-thrust, no-stall world: float cost",
          f"The floats' cost in each state under the rising-thrust controller in the no-stall comparison world, "
          f"{_settled}.")]
assert sum(n for _, n, _ in _groups) == len(_cols)
_st = table([c for c, _, _ in _cols],
            [[fx(c, 1), fx(n["float_cost"], 2), fx(sa["thruster1"], 2), _fc(sa), fx(rt["thruster1"], 2),
              fx(rt["thruster1_push"], 2), _fc(rt), fx(ns["thruster1"], 2), _fc(ns)]
             for c, n, sa, rt, ns in zip(SCT["currents"], _sct["no_throttle"], _sct["stall_aware"],
                                         _sct["rising_thrust"], _sct["rising_thrust_no_stall_world"])],
            numeric=set(range(1, 9)), minw=760, stack=True, keep1=True, cls="grouped",
            tips=[(full, desc) for _, full, desc in _cols])
_grow = "<tr class=\"grp\">" + "".join(
    f'<th scope="colgroup" colspan="{n}"{" class=" + chr(34) + "g0" + chr(34) if not g else ""}'
    f'{f" data-tip={chr(34)}{esc(d)}{chr(34)}" if d else ""}>{g}</th>'
    for g, n, d in _groups) + "</tr>"
# each short column head carries its full name in data-label (the phone-width card and the row tuple use it)
assert _st.count("data-label=") == len(_cols)
FRAG["TASK_STEADY_TABLE"] = _st.replace("<thead><tr>", "<thead>" + _grow + "<tr>", 1)
D["derived"]["push_bin_width"] = _bin
_lead1 = O["controllers"]["wave_response"]["textbook_rule"]["throttle_lead_steps_against_push"][0]
assert -0.5 < _lead1 < 0  # thruster 1's settled throttle is in step with the wave's push, a hair behind it
D["derived"]["thr1_wave_lag_steps"] = -_lead1
D["derived"]["sa_recorded_throttle_share"] = 1 - O["water_alone"]["cost_split"]["recorded"]["stall_aware"][
    "float_share_of_cost"]
D["derived"]["steady_ratio_at_09"] = _sct["rising_thrust"][4]["float_cost"] / _sct["stall_aware"][4]["float_cost"]
D["derived"]["rt_thruster1_push_size_at_09"] = abs(_sct["rising_thrust"][4]["thruster1_push"])
D["derived"]["unsettled_currents"] = ", ".join(fx(r["current"], 1) for r in _sct["rising_thrust"] if not r["settled"])


# ------------------------------------------------------------------------------------------------ after the stop (EXPLORATORY)
# Stage 1's room test of the value-penalty route, the deployment-time diagnostic: tables and one chart, all from
# beyond_data.json. Every cell is a mean over seeds 0-4; the range under it runs from the lowest to the highest seed.

BY = D["beyond"]
BY_SEEDS = f'{BY["design"]["seeds_lo"]}\u2013{BY["design"]["seeds_hi"]}'
BY_FUT = [int(x) for x in BY["design"]["futility_seeds"].split(" and ")]


def mr(st, d=1, signed=False, pct=False):
    """a mean over seeds, with the lowest and highest seed under it"""
    f = (lambda v: pc(v, d)) if pct else ((lambda v: sg(v, d)) if signed else (lambda v: fx(v, d)))
    return f'{f(st["mean"])}<span class="rng">{f(st["lo"])} to {f(st["hi"])}</span>'


# (key, plain name, chart group): ref = no signal, hi = the critic's exact error at the actor's own command,
# lo = untargeted or misplaced
BY_ARMS = [
    ("clone", "Cloned actor (the start of every arm)", "ref"),
    ("none", "No extra caution (the gate's actor on the gapped critic)", "ref"),
    ("perfect_band", "<b>Oracle penalty at the actor's command</b> (main arm)", "hi"),
    ("perfect_band_half", "Oracle, half strength", "hi"),
    ("perfect_band_centred", "Oracle, two-sided form", "hi"),
    ("perfect_band_literal", "Oracle, the plan's literal form (size of the error)", "hi"),
    ("overrating_half_literal", "Oracle, the plan's literal overrating side", "hi"),
    ("long_run_harm", "Exact long-run harm as the penalty", "hi"),
    ("state_shuffled_band", "Oracle taken from another training state", "lo"),
    ("underrating_half", "The underrating side of the error (wrong side)", "lo"),
    ("even_pull", "Even extra pull, matched strength", "lo"),
    ("push_less", "Push less everywhere, matched", "lo"),
    ("random_1", "Random penalty surface 1, matched", "lo"),
    ("random_2", "Random penalty surface 2, matched", "lo"),
    ("range_hinge", "Clip to the recorded range (no signal)", "ref"),
]
BY_GROUPS = {"clone": "Starting points", "perfect_band": "Penalty from the critic's exact error at the actor's own command",
             "state_shuffled_band": "The same information, misplaced or on the wrong side",
             "even_pull": "Untargeted penalties of matched strength", "range_hinge": "No signal"}
assert {k for k, _, _ in BY_ARMS} == set(BY["stall"]), "every stalling-world arm is shown once"


# each row names its controller for the row's tooltip (data-ctrl, the same explanation as the controller guide's)
_BY_CTRL = {"clone": "clone", "none": "gapped", "random_2": "s1_random_1"}
rows = []
for key, name, _ in BY_ARMS:
    if key in BY_GROUPS:
        rows.append(("sub", [f'<span class="grp">{BY_GROUPS[key]}</span>']))
    rows.append(("main" if key == "perfect_band" else "",
                 [f'<span data-ctrl="ctrl-{_BY_CTRL.get(key, "s1_" + key)}">{name}</span>'] + [mr(BY["stall"][key][s]) for s in SIT] + [mr(BY["stall_past80"][key]["strong_current"], 0, pct=True)]))
FRAG["BEYOND_STALL_TABLE"] = table(
    ["Arm", "Recorded conditions", "Strong current", "Big waves", "Big starting pushes",
     "Strong current: share of states past 80%"], rows, numeric={1, 2, 3, 4, 5}, minw=760, cls="rows8", tips=[
        "One actor-only run against the gate's frozen gapped critic: the cloned start, no extra caution, or one kind of "
        "penalty or caution. The rows are grouped by kind."] + [
        f"Points in {SIT_LABEL[s].lower()}, stalling world: the mean over seeds {BY_SEEDS}, with the lowest to highest "
        f"seed below it. 0 is the recording controller, 100 the stall-aware controller." for s in SIT] + [
        f"In strong current, the share of visited states at which some thruster is past 80%: the mean over seeds "
        f"{BY_SEEDS}, with the lowest to highest seed below it."],
    caption=f"Stalling world, against the gate's frozen gapped critic. Points: mean over seeds {BY_SEEDS}, lowest to highest seed below.")

NS_ROWS = [("perfect_band", "<b>Oracle penalty</b>, from this world's own critic"),
           ("perfect_band_rising", "Oracle, scored with the next command of the rising-thrust controller"),
           ("even_pull_locked", "Even extra pull, at its stalling-world strength"),
           ("range_hinge", "Clip to the recorded range (no signal)")]
rows = [("main" if k == "perfect_band" else "", [n] + [mr(BY["no_stall_cost"][k][s]) for s in SIT]) for k, n in NS_ROWS]
FRAG["BEYOND_NOSTALL_TABLE"] = table(
    ["Arm", "Recorded conditions", "Strong current", "Big waves", "Big starting pushes"], rows, numeric={1, 2, 3, 4},
    minw=640, cls="rows8", tips=[
        "One arm, run in the no-stall comparison world, where pushing past 80% does no harm, so any caution only costs."] + [
        f"What this arm's caution costs in {SIT_LABEL[s].lower()}: the points of no extra caution minus the points of the "
        f"arm, so higher costs more. The mean over seeds {BY_SEEDS}, with the lowest to highest seed below it." for s in SIT],
    caption="No-stall comparison world: the cost of caution, points of no extra caution minus points of the arm "
            f"(higher costs more). Mean over seeds {BY_SEEDS}, lowest to highest seed below.")

REP_ROWS = [("none", "No extra caution (the room check's own actor)", ""),
            ("perfect_band", "<b>Oracle penalty at the actor's command</b>", "main"),
            ("even_pull", "Even extra pull, matched to the oracle's strength", ""),
            ("harm_placed", "Room check: caution placed at the harm (reused)", ""),
            ("strength_matched_even", "Room check: even caution matched in strength (reused)", ""),
            ("cap2", "Room check: caution placed at the harm with doses up to 2 (reused)", "")]
rows = [(c, [n] + [mr(BY["replication"][k][s]) for s in SIT]) for k, n, c in REP_ROWS]
FRAG["BEYOND_REP_TABLE"] = table(
    ["Arm", "Recorded conditions", "Strong current", "Big waves", "Big starting pushes"], rows, numeric={1, 2, 3, 4},
    minw=640, cls="rows8", tips=[
        "One arm, repeated on the room check's own critic, the frozen critic of a plain TD3+BC run on the log after the "
        "removal. Rows marked reused are the room check's own runs."] + [
        f"Points in {SIT_LABEL[s].lower()}, stalling world: the mean over seeds {BY_SEEDS}, with the lowest to highest "
        f"seed below it." for s in SIT],
    caption="Replication on the room check's critic (the frozen critic of a plain TD3+BC run on the log after the removal). Points: "
            f"mean over seeds {BY_SEEDS}, lowest to highest seed below.")

PM, MK = BY["pooled"], BY["marks"]
RC_ = PM["room"]["controls"]
ok = lambda b, t="pass": f'<span class="tag {"ok" if b else "warn"}">{t if b else "fail"}</span>'
rows = [
    [f"Futility look (seeds {BY['design']['futility_seeds']} first)",
     f"stop if the oracle beats the matched even pull by less than {fx(MK['futility_margin'], 0)} points in strong "
     f"current on both seeds",
     " and ".join(f"by {fx(PM['futility'][f'seed{k}'], 1)} (seed {k})" for k in BY_FUT),
     ok(not PM["futility"]["stop"], "continue")],
    ["Room",
     f"in strong current the oracle beats each untargeted control by at least {fx(MK['room_margin'], 0)} points, and "
     f"is ahead in at least {MK['room_positive']} of {MK['room_of']} seeds",
     f"by {fx(RC_['even_pull']['mean'], 1)} over even extra pull, {fx(RC_['push_less']['mean'], 1)} over push less "
     f"everywhere, {fx(RC_['random_1']['mean'], 1)} and {fx(RC_['random_2']['mean'], 1)} over the two random surfaces; "
     f"ahead in {PM['room']['positive_min']} of {MK['room_of']} seeds against each",
     ok(PM["room"]["passed"])],
    ["Recovery",
     f"in strong current at least {fx(MK['recovery_threshold'], 1)} points above no extra caution (half of the "
     f"{fx(MK['recovery_harm'], 1)}-point harm the gate measured)",
     f"{sg(PM['recovery']['mean'], 1)} (lowest to highest seed: {fx(PM['recovery']['lo'], 1)} to {fx(PM['recovery']['hi'], 1)})",
     ok(PM["recovery"]["passed"])],
    ["Keeps the actor's improvement",
     f"recorded conditions: at least no extra caution {MINUS} {fx(MK['keep_recorded_vs_none'], 0)}, and at least the "
     f"cloned actor + {fx(MK['keep_recorded_vs_clone'], 0)}; big waves and big starting pushes: at least no extra "
     f"caution {MINUS} {fx(MK['keep_shifted_vs_none'], 0)}",
     f"recorded conditions {sg(PM['keeps']['recorded_vs_none']['mean'], 1)} over no extra caution and "
     f"{sg(PM['keeps']['recorded_vs_clone']['mean'], 1)} over the cloned actor; big waves "
     f"{sg(PM['keeps']['big_waves_vs_none']['mean'], 1)}; big starting pushes "
     f"{sg(PM['keeps']['big_kicks_vs_none']['mean'], 1)}",
     ok(PM["keeps"]["passed"])],
]
FRAG["BEYOND_MARKS_TABLE"] = table(["Mark", f"Rule, fixed before any return was computed (pooled over seeds {BY_SEEDS} "
                                    "unless the rule names seeds)", "Result", "Verdict"],
                                   rows, minw=720, stack=True, tips=[
    "One pass mark of Stage 1, fixed before any return was computed.",
    "What the mark requires, as fixed in advance.",
    "What the runs gave, pooled over the seeds unless the rule names seeds.",
    "Whether the mark passed (for the early futility look: whether the runs continued)."])

DP = BY["deploy"]
DEP_ROWS = [("check_gain", "Distance check → clip", "main"),
            ("random_gain", "Random clipping at the check's rate", ""),
            ("clip_all_gain", "Clip at every state, no check", ""),
            ("perfect_gain", "Perfect detector of past-80% proposals → clip (reference)", ""),
            ("clone_fallback_gain", "Distance check → cloned actor", "")]
rows = [(c, [f'<span data-ctrl="ctrl-dep_{k}">{n}</span>'] + [sg(DP[k][s]["mean"], 1) for s in SIT]) for k, n, c in DEP_ROWS]
FRAG["BEYOND_DEPLOY_TABLE"] = table(
    ["Rule", "Recorded conditions", "Strong current", "Big waves", "Big starting pushes"], rows, numeric={1, 2, 3, 4},
    cls="rows8", tips=[
        "One rule applied at deployment, with no training: when to replace the actor's command, and with what."] + [
        f"The change in points from the gapped critic's actor alone in {SIT_LABEL[s].lower()}, stalling world: the mean "
        f"over seeds {BY_SEEDS}, on the first {DP['n_starts']} starts of the bank." for s in SIT],
    caption=f"Change in points from the gapped critic's actor alone, stalling world; mean over seeds {BY_SEEDS}, on the "
            f"first {DP['n_starts']} starts of each bank.")


def dot_range_chart():
    """One row per arm: the dot is the mean over seeds, the line runs from the lowest to the highest seed. Rows are
    HTML so the labels reflow at phone width; positions are percentages of the track."""
    st = BY["stall"]
    lo = min(st[k]["strong_current"]["lo"] for k, _, _ in BY_ARMS)
    hi = max(st[k]["strong_current"]["hi"] for k, _, _ in BY_ARMS)
    x0, x1 = 20 * math.floor(lo / 20), max(100, 20 * math.ceil(hi / 20))
    pos = lambda v: 100 * (v - x0) / (x1 - x0)
    ticks = list(range(x0, x1 + 1, 20))
    grid = ", ".join(f"linear-gradient(var(--{'rule-strong' if t in (0, 100) else 'rule'}), "
                     f"var(--{'rule-strong' if t in (0, 100) else 'rule'}))" for t in ticks)
    gpos = ", ".join(f"{pos(t):.2f}% 0" for t in ticks)
    track_style = f'background-image:{grid};background-position:{gpos}'
    o = ['<div class="dr" role="group" aria-label="Points in strong current by arm, stalling world: mean over seeds '
         f'{BY["design"]["seeds_lo"]} to {BY["design"]["seeds_hi"]} with the lowest and highest seed">']
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{fx(t, 0)}</span>' for t in ticks)
    o.append(f'<div class="dr-axis" aria-hidden="true"><span class="dr-name"></span><span class="dr-val"></span>'
             f'<span class="dr-ticks">{axis}</span></div>')
    for key, name, grp in BY_ARMS:
        if key in BY_GROUPS:
            o.append(f'<div class="dr-grp">{BY_GROUPS[key]}</div>')
        v = st[key]["strong_current"]
        m, a, b = pos(v["mean"]), pos(v["lo"]), pos(v["hi"])
        tip = f"{plain(name)}: {fx(v['mean'], 1)}; lowest to highest seed: {fx(v['lo'], 1)} to {fx(v['hi'], 1)}"
        o.append(f'<div class="dr-row {grp}" tabindex="0" aria-label="{esc(tip)}" data-tip="{esc(tip)}">'
                 f'<span class="dr-name">{name}</span><span class="dr-val">{fx(v["mean"], 1)}</span>'
                 f'<span class="dr-track" style="{track_style}">'
                 f'<span class="dr-rng" style="left:{a:.2f}%;width:{b - a:.2f}%"></span>'
                 f'<span class="dr-dot" style="left:{m:.2f}%"></span></span></div>')
    o.append('<div class="dr-axis dr-axlab" aria-hidden="true"><span class="dr-name"></span><span class="dr-val"></span>'
             '<span class="dr-ticks"><span class="lab">points in strong current: 0 is the recording controller, '
             '100 the stall-aware controller</span></span></div>')
    o.append("</div>")
    return "".join(o)


FRAG["BEYOND_CHART"] = dot_range_chart()


# ------------------------------------------------------------------------------------------------ live panel data

LP = D["live_panel"]


def r3(v, d):
    return [round(float(x), d) for x in v]


roll = {}
for key, r in LP["rollouts"].items():
    o = dict(s=[r3(row, 3) for row in r["states"]], a=[r3(row, 4) for row in r["commands"]],
             f=[r3(row, 4) for row in r["thrust"]], st=["".join(str(int(b)) for b in row) for row in r["stalled"]],
             r=r3(r["rewards"], 4), g=r3(r["running_return"], 3), sum=r["summary"])
    if "full_throttle_command" in r:
        o["ft"] = [int(b) for b in r["full_throttle_command"]]
    roll[key] = o
ref = {w: {s: {c: dict(J=round(v["J"], 2), pts=round(v["points"], 1), se=round(v["J_se"], 2))
               for c, v in RR["returns"][w][s].items()} for s in SIT} for w in ("stall", "no_stall")}
water = {s: dict(c=v["current"], amp=v["wave_amplitude"], dist=v["float_start_distance"], push=v["push"],
                 reg=v["in_full_throttle_region"]) for s, v in LP["water"].items()}
payload = dict(roll=roll, ref=ref, water=water, trigger=RC["trigger"], start=LP["start_index"])
# the panel opens at the first full-throttle command of its default episode (the script picks the same step)
D["panel_open_t"] = roll["stall|recorded|recording"]["ft"].index(1)


# ------------------------------------------------------------------------------------------------ controllers

# Every controller and learned actor the page uses, once: how it sets the four throttles, whether it keeps to the 80%
# limit, and its points. The same words become the controller guide in the episode panel (CTRL_GUIDE) and the tooltip
# of every place that names the controller (data-term="ctrl-..."), so a controller reads the same wherever it is named.
CD = D["controllers"]
_GSEEDS = f"{CD['seeds'][0]}–{CD['seeds'][-1]}"


def _ref_pts(world, ctrl):
    return [RR["returns"][world][s][ctrl]["points"] for s in SIT]


def _past(world, ctrl, sit="strong_current"):
    return RR["returns"][world][sit][ctrl]["steps_past_80"]


def _mean_pts(block, key):
    return [block[key][s]["mean"] for s in SIT]


_GP, _G80 = CD["gate"]["points"], CD["gate"]["past80"]
_RP, _R80 = CD["room"]["points"], CD["room"]["past80"]
_SP, _S80 = BY["stall"], BY["stall_past80"]
_PASTW = "Not built in; it goes past 80% on {} of the steps it takes in strong current."
_SIT_SHORT = ["recorded conditions", "strong current", "big waves", "big starting pushes"]


def _named(vals):
    """points per situation, each with its situation's name"""
    return ", ".join(f"{n} {fx(v, 1)}" for n, v in zip(_SIT_SHORT, vals))


def _pastw(share):
    return _PASTW.format(pc(share, 0))


# (key, name, group, how it sets the throttles, the 80% limit, points per situation or None, how the points read)
CTRL_GROUPS = [
    ("fixed", "Fixed controllers: no learning, the same command in the same state"),
    ("gate", f"Learned actors of the gate at α = {fx(CD['alpha'], 0)} (the cloned actor, plain TD3+BC, and "
             f"actor-only runs from the cloned actor), means over gate seeds {_GSEEDS}"),
    ("room", "The room check's arms: extra pull toward the recorded command, on the critic of a plain TD3+BC run "
             f"(means over seeds {_GSEEDS})"),
    ("stage1", "After the stop, Stage 1: a penalty on the value of the actor's own command, on the gate's gapped "
               f"critic (means over seeds {_GSEEDS})"),
    ("deploy", f"After the stop, at deployment: rules applied to the gapped critic's actor, no training (change in "
               f"points against that actor alone, on the first {DP['n_starts']} starts)"),
    ("other", "Not run, or not scored on their own"),
]
CTRLS = [
    ("recording", "Recording controller", "fixed",
     "It made the recorded data. Most of the time it drives normally: it holds fully against the current and the "
     "wave, damps the floats' swinging only a quarter as hard as the textbook rule, and adds a random wobble, all "
     f"squashed inside ±80%. Where the water's push c + w₁ is above {TRIG} (the full-throttle region) it flips a coin in "
     "each state and on heads gives the full-throttle command: thruster 1 at full throttle, the others in the same "
     "pattern.",
     f"Normal driving never passes 80%; the full-throttle command puts thruster 1 at 100%, where it stalls. Past 80% on "
     f"{pc(_past('stall', 'recording', 'recorded'), 0)} of steps in the recorded conditions and "
     f"{pc(_past('stall', 'recording'), 0)} in strong current.",
     _ref_pts("stall", "recording"), "0 by definition, in every situation and in both worlds"),
    ("stall_aware", "Stall-aware controller", "fixed",
     "The textbook command: each throttle is a fixed weighted sum of the 13 observed numbers (the textbook rule K, "
     "the best linear rule if push always equalled throttle), then cut to ±80%.",
     f"Kept: it never goes past 80%, so it never stalls. In strong current it sits at the limit on "
     f"{pc(O['controllers']['stall_aware']['strong_current']['at_cap_thruster1'], 0)} of states (thruster 1).",
     _ref_pts("stall", "stall_aware"),
     "100 by definition in every situation of the stalling world: the page's top reference, good but not "
     "proven best. In the no-stall comparison world it scores " + _named(_ref_pts("no_stall", "stall_aware"))),
    ("rising_thrust", "Rising-thrust controller", "fixed",
     "The same textbook command, cut only at ±100%: it acts as if push kept rising with throttle, as a learned actor "
     "would if it trusted a critic that never saw the stall.",
     f"Ignored: it goes past 80% whenever the textbook command asks, on {pc(_past('stall', 'rising_thrust'), 0)} of steps "
     "in strong current, where a stalled propeller pushes less.",
     _ref_pts("stall", "rising_thrust"), "100 by definition in every situation of the no-stall comparison world"),
    ("normal_driving_only", "Normal driving only", "fixed",
     "The recording controller's normal driving without the wobble and never the full-throttle command. The design "
     "expected copying the log after the removal to score about this; the cloned actor, which does that copying, in "
     "fact scores far lower: {{controllers/gate/points/clone/recorded/mean#f1}} points in the recorded conditions and "
     "{{controllers/gate/points/clone/big_kicks/mean#f1}} in big starting pushes.",
     "Kept: the squashing keeps every throttle inside ±80%.",
     _ref_pts("stall", "normal_driving_only"), ""),
    ("no_throttle", "No throttle", "fixed",
     "All four throttles stay at 0: the water acting alone.", "Never near it.",
     _ref_pts("stall", "no_throttle"), ""),
    ("clone", "Cloned actor", "gate",
     "A network fitted to copy the recorded commands of the log after the removal (behaviour cloning, "
     "{{methods/td3_bc/settings/clone_steps#i}} steps), with no critic. It reads the 13 observed numbers and gives "
     "four throttles. Every actor-only run starts from it.",
     _pastw(_G80["clone"]["strong_current"]["mean"]) + " The log after the removal holds almost no command past 80%.",
     _mean_pts(_GP, "clone"), ""),
    ("td3bc_gapped", "Plain TD3+BC actor", "gate",
     "TD3+BC trained the usual way on the log after the removal: critic and actor learn together, and the actor picks "
     "commands its critic rates highly while staying close to the recorded ones (every caution multiplier 1).",
     _pastw(_G80["td3bc_gapped"]["strong_current"]["mean"]) + " Its critic almost never saw the stall: only "
     "{{dataset/gap_removal/kept_rows_past_80#i}} of the {{dataset/gap_removal/kept_training_rows#i}} kept rows go "
     "past 80%.",
     _mean_pts(_GP, "td3bc_gapped"), ""),
    ("gapped", "Actor of the gapped critic", "gate",
     "Trained from the cloned actor for {{methods/td3_bc/settings/actor_only_steps#i}} steps against a frozen critic of "
     "the stall-aware controller learned from the log after the removal (the gapped critic), which almost never saw "
     "the stall ({{dataset/gap_removal/kept_rows_past_80#i}} of {{dataset/gap_removal/kept_training_rows#i}} kept rows "
     "go past 80%).",
     _pastw(_G80["gapped"]["strong_current"]["mean"]), _mean_pts(_GP, "gapped"), ""),
    ("full", "Actor of the full-log critic", "gate",
     "The same run against a frozen critic learned from the full log, which does show the stall.",
     _pastw(_G80["full"]["strong_current"]["mean"]), _mean_pts(_GP, "full"), ""),
    ("random", "Actor of the random-removal critic", "gate",
     "The same run against a critic learned after removing the same number of rows at random.",
     _pastw(_G80["random"]["strong_current"]["mean"]), _mean_pts(_GP, "random"), ""),
    ("host_swap_gapped", "Actor of the plain TD3+BC run's gapped critic", "gate",
     "The same run against the frozen critic of a plain TD3+BC run on the log after the removal. The room check's "
     "arms all start from this setting; with no extra caution it is the room check's own actor.",
     _pastw(_G80["host_swap_gapped"]["strong_current"]["mean"]), _mean_pts(_GP, "host_swap_gapped"), ""),
    ("host_swap_full", "Actor of the plain TD3+BC run's full-log critic", "gate",
     "The same run against the frozen critic of a plain TD3+BC run on the full log.",
     _pastw(_G80["host_swap_full"]["strong_current"]["mean"]), _mean_pts(_GP, "host_swap_full"), ""),
    ("no_stall_actors", "The same actors in the no-stall comparison world", "gate",
     "The cloned actor and the actors of the gapped and full-log critics, trained on that world's own logs.",
     "Not built in; past 80% does no harm there. Gapped critic's actor in strong current: past 80% on "
     f"{pc(_G80['no_stall_gapped']['strong_current']['mean'], 0)} of steps.",
     _mean_pts(_GP, "no_stall_gapped"),
     "nostall: the full-log critic's actor scores " + _named(_mean_pts(_GP, "no_stall_full"))
     + "; the cloned actor " + _named(_mean_pts(_GP, "no_stall_clone"))),
]
_ROOM = [
    ("none", "No extra caution", "Every caution multiplier is 1: the room check's own actor."),
    ("harm_placed", "Harm-placed caution",
     "Caution multipliers between 1 and 1.5 (BCA's range), placed by the exact over-preference: largest on the rows "
     "where the critic most over-prefers the actor's command. Methods calls it over-preference-placed caution."),
    ("strength_matched_even", "Strength-matched even caution",
     "One multiplier on every row ({{controllers/room/strength_match/multiplier_lo#f2}} to "
     "{{controllers/room/strength_match/multiplier_hi#f2}} over the seeds), matched blind, before any return was "
     "computed, to change the actor as much as harm-placed caution does (within "
     "{{controllers/room/strength_match/max_relative_error#p1}} on every seed)."),
    ("amount_matched", "Amount-matched caution", "The same total caution as harm-placed, spread evenly over the rows."),
    ("shuffled", "Shuffled caution",
     "The harm-placed multipliers in a random row order (the mean of two shuffles)."),
    ("reversed", "Reversed caution", "The harm-placed multipliers with their ranks flipped: the largest on the row with "
     "the least over-preference."),
    ("cap2", "Caution capped at 2", "Harm-placed caution with a range twice as wide as BCA's: factors up to 2."),
] + [(f"even_x{m}", f"Even caution ×{m}", f"The same multiplier, {m}, on every row.") for m in ("1.1", "1.25", "1.5", "2", "3")]
for k, n, how in _ROOM:
    CTRLS.append((f"room_{k}", n, "room", how + " Each room-check arm is an actor-only run from the cloned actor "
                  "against the frozen critic of a plain TD3+BC run on the log after the removal; only its pull toward "
                  "the recorded commands changes.", _pastw(_R80[k]["strong_current"]["mean"]),
                  _mean_pts(_RP, k), ""))
_S1 = {
    "perfect_band": ("Oracle penalty (main arm)",
                     "Lowers the critic's value of the actor's own command by how much more the critic overrates it than "
                     "the row's recorded command, from the critic's exact one-step error. An oracle: exact information "
                     "that no method learning from data has."),
    "perfect_band_half": ("Oracle, half strength", "The oracle penalty at half its weight."),
    "perfect_band_centred": ("Oracle, two-sided form", "The size of the gap between the errors at the actor's and at "
                             "the recorded command, in either direction."),
    "perfect_band_literal": ("Oracle, the plan's literal form", "The plain size of the error at the actor's command."),
    "overrating_half_literal": ("Oracle, the plan's literal overrating side",
                                "The error at the actor's command, counted only where the critic overrates it."),
    "long_run_harm": ("Exact long-run harm as the penalty",
                      "The critic's error over {{beyond/design/harm_rollout_steps#i}} steps instead of one."),
    "state_shuffled_band": ("Oracle taken from another training state", "The oracle's penalty read at the wrong state."),
    "underrating_half": ("The underrating side of the error", "Penalises the commands the critic underrates: the wrong "
                         "side."),
    "even_pull": ("Even extra pull", "No penalty: the pull toward the recorded command made stronger on every row, "
                  "matched in strength to the oracle."),
    "push_less": ("Push less everywhere", "A penalty on the size of every command, matched in strength."),
    "random_1": ("Random penalty surfaces", "A penalty from a frozen network with random settings, of the same shape as "
                 "BCA's error-size network, matched in strength (the first of two; the second scores "
                 + _named(_mean_pts(_SP, "random_2")) + ")."),
    "range_hinge": ("Clip to the recorded range", "A steep penalty on any thruster command outside the range that holds "
                    "the middle 99% of the training commands. It uses no signal at all."),
}
for k, (n, how) in _S1.items():
    lim = _pastw(_S80[k]["strong_current"]["mean"])
    if k == "range_hinge":
        lim = ("A steep but soft penalty outside the range that holds the middle 99% of the training commands, which "
               "ends just below 80%; the actor still goes past 80% on "
               f"{pc(_S80[k]['strong_current']['mean'], 0)} of strong-current steps.")
    CTRLS.append((f"s1_{k}", n, "stage1", how + " Trained from the cloned actor on the gate's frozen gapped critic.",
                  lim, _mean_pts(_SP, k), ""))
_DEP = [("check_gain", "Distance check, falling back to a clip",
         "At each state the actor visits, the check measures how far the (state, command) pair lies from the training "
         "data (distance to its {{beyond/deploy/neighbours#i}}th-nearest training pair, in standard units). Where it "
         "is further than {{beyond/deploy/level#i}}% of held-out recorded rows, the command is clipped into the recorded "
         "range; otherwise the actor's command is used."),
        ("random_gain", "Random clipping at the check's rate", "Clips at randomly chosen states, as often as the check."),
        ("clip_all_gain", "Clip at every state", "Clips every command into the recorded range."),
        ("perfect_gain", "Perfect detector, falling back to a clip", "Clips exactly the commands past 80% (a reference "
         "no real check could match)."),
        ("clone_fallback_gain", "Distance check, falling back to the cloned actor",
         "Where the check fires, the cloned actor's command replaces the actor's.")]
for k, n, how in _DEP:
    CTRLS.append((f"dep_{k}", n, "deploy", how, "Kept only where the rule acts. There every throttle is clipped into "
                  "the range that holds the middle 99% of the recorded commands, which ends just below 80%; elsewhere "
                  "the actor's own command is used." if k != "clone_fallback_gain" else
                  "Kept only where the rule acts; the cloned actor seldom passes 80%.",
                  [DP[k][s]["mean"] for s in SIT], "change"))
CTRLS += [
    ("bca", "BCA's actor", "other",
     "TD3+BC whose caution multiplier at each training row comes from BCA's calibrated band (doses from 1 up to 1.5).",
     "Not built in.", None, "none, it never ran on this environment (the gate stopped before BCA)"),
    ("forecast", "The design's forecast stand-ins", "other",
     "Fixed stand-ins for the learned actor: normal driving moved {{methods/actor_fixed_point/share_of_way/dose_1#f3}} "
     "(a critic that bends like the truth) or {{fixedpoint/edge/dose_1#f3}} (a critic that keeps its slope) of the way "
     "toward the rising-thrust controller's command; the capped stand-in is the same, cut at 80%.",
     "The stand-in follows the rising-thrust controller past 80%; the capped stand-in keeps to it.", None,
     "none of their own; the forecast reports the capped stand-in's points minus the stand-in's"),
    ("standins", "Data-only stand-ins", "other",
     "Eight penalties computed from the recorded data alone, put in the oracle's place in Stage 1's setting.",
     "Not built in.", None, "none yet; they are running"),
    ("iw_recording", "The weighting test's recording controller", "other",
     "Drives normally like the recording controller; in hard water it flips a coin each step and on heads holds "
     "thruster 1 at exactly 80%.", "Kept: its hold is exactly 80%, never past.", None,
     "0 of the weighting test, by definition; that test has not run yet"),
]


def _ctrl_points_text(group, pts, note):
    if pts is None:
        return f"Points: {note}."
    if group == "fixed" and note and len(set(round(v, 6) for v in pts)) == 1:  # the level fixed by definition
        return f"Points: {note}."
    vals = ", ".join(f"{n} {sg(v, 1) if group == 'deploy' else fx(v, 1)}" for n, v in zip(_SIT_SHORT, pts))
    if group == "deploy":
        return f"Change in points against the gapped critic's actor alone: {vals}."
    if note.startswith("nostall: "):
        return (f"Points in the no-stall comparison world (100 is the rising-thrust controller), mean over seeds "
                f"{_GSEEDS}, gapped critic's actor: {vals}. {note[9].upper() + note[10:]}.")
    head = "Points, stalling world" + ("" if group == "fixed" else f", mean over seeds {_GSEEDS}")
    return f"{head}: {vals}." + (f" ({note[0].upper() + note[1:]}.)" if note else "")


CTRL_TERMS = {}
for key, name, group, how, lim, pts, note in CTRLS:
    CTRL_TERMS["ctrl-" + key] = f"{how} The 80% limit: {lim[0].lower() + lim[1:]} {_ctrl_points_text(group, pts, note)}"

rows = []
for g, gname in CTRL_GROUPS:
    rows.append(("sub", [f'<span class="grp">{esc(gname)}</span>']))
    for key, name, group, how, lim, pts, note in CTRLS:
        if group != g:
            continue
        # no points: an en dash in each points column, and the reason under the description
        cells = ["\u2013"] * 4 if pts is None else [sg(v, 1) if g == "deploy" else fx(v, 1) for v in pts]
        nt = note[9:] if note.startswith("nostall: ") else note
        if note.startswith("nostall: "):
            nt = "Points in the no-stall comparison world (100 is the rising-thrust controller); shown: the gapped " \
                 "critic's actor; " + nt
        extra = (f'<span class="rng">{"Points: " if pts is None else ""}'
                 f'{esc(nt[0].upper() + nt[1:] if pts is not None else nt)}.</span>') if note and note != "change" else ""
        rows.append([f'<b>{esc(name)}</b>', esc(how) + f'<span class="lim"><b>The 80% limit.</b> {esc(lim)}</span>'
                     + extra] + cells)
FRAG["CTRL_GUIDE"] = table(
    ["Controller or actor", "How it sets the four throttles, and the 80% limit"] + [SIT_LABEL[s] for s in SIT], rows,
    numeric={2, 3, 4, 5}, minw=700, stack=True, cls="ctrlguide", tips=[
        "A fixed controller or a learned actor used on this page, grouped by where it is used.",
        "How it turns the 13 observed numbers into the four throttles, and whether it keeps every throttle at or below "
        "80%, where a propeller stalls in the stalling world.",
    ] + [(f"Points in {SIT_LABEL[s].lower()}",
          f"Points in {SIT_LABEL[s].lower()}: 0 is the recording controller, 100 the stall-aware controller (stalling "
          f"world). Learned actors: the mean over gate seeds {_GSEEDS}. Rules at deployment: the change in points they "
          "make to the gapped critic's actor.") for s in SIT])
# the episode panel's own explanation of its four controllers uses the same words
payload["ctrlText"] = {k: fill_numbers(CTRL_TERMS["ctrl-" + k]) for k in ("recording", "stall_aware", "rising_thrust",
                                                                         "no_throttle")}
payload["demo"] = dict(offs=_offsets, key=DEMO_KEY)


# ------------------------------------------------------------------------------------------------ tooltips: glossary

# One explanation per term, shown as the tooltip of every element the template marks data-term="key", so a term reads
# the same wherever it is used. Plain text: numbers through {{path#fmt}} like the rest of the page, and no TeX (write
# c + w₁), because these strings go into the page's script, where nothing is typeset.
TERMS = {
    "actor": "The learned controller: the network that TD3+BC trains to choose the four throttles from the 13 observed "
             "numbers. This page says actor for the learned controller and controller for the fixed ones.",
    "critic": "A learned estimate of how good each command is in each state: the discounted return of giving that "
              "command and letting the actor drive afterwards. The actor learns to choose commands its critic rates "
              "highly.",
    "caution": "A harder pull toward the recorded command at chosen training rows. Its size at a row is the dose, or "
               "caution multiplier: 1 means no extra caution.",
    "return": "The total reward over one episode: the sum of its 100 rewards. Every reward is a cost with a minus sign, "
              "so the best possible return is 0 and higher is better.",
    "episode": "One run of 100 steps from one starting state.",
    "points": "A score per situation, on the same 1,000 starts for every controller: 0 is the recording controller's "
              "average return and 100 the stall-aware controller's (in the no-stall comparison world, the "
              "rising-thrust controller's).",
    "row": "One recorded step of the data: the observation, the command the recording controller gave, its reward and "
           "the next observation.",
    "distribution-shift": "A difference between the training data and the situations met at test time. Here it is "
                          "built on purpose and controlled.",
    "full-throttle-region": f"Every state where the water's push c + w₁ is above {TRIG}. There the recording "
                            "controller flips a coin in each state and on heads puts thruster 1 at full throttle.",
    "stall": "Past 80% throttle a propeller stalls in the stalling world: its push falls, to "
             f"{STALL90} at 90% and {STALL100} at full throttle, while its throttle cost keeps rising. The no-stall "
             "comparison world has no stall.",
}
_R = "{{environment/start_laws/recorded/"
_SC = "{{environment/start_laws/strong_current/"
TERMS.update({
    "crest": "The top of the wave's cycle, where the wave adds the most push to the current's: the water's push is at "
             "its highest there. A crest comes once every {{environment/dynamics/wave_period_steps#f2}} steps.",
    "trough": "The bottom of the wave's cycle, where the wave takes the most push away: the water's push is at its "
              "lowest there, and turns upstream when the wave is bigger than the current.",
    "water-push": f"The push of the water on every float, the same on all five: the current's strength plus the "
                  f"wave's push now, c + w₁. Above {TRIG} the state is in the full-throttle region.",
    "recorded-conditions": f"The start law the data were recorded under: current {_R}current/0#f1}}}} to "
                           f"{_R}current/1#f1}}}}, wave amplitude {_R}wave_amplitude/0#f1}}}} to "
                           f"{_R}wave_amplitude/1#f1}}}}, floats' starting spread {_R}float_start_sd#f1}}}}. No shift: here "
                           "BCA's actor must not lose to plain TD3+BC (the no-loss check: the lower bound of BCA minus "
                           "plain TD3+BC above {{methods/gate/pass_marks/B2/p2_lower_gt#f0}} points).",
    "shifted-situation": "One of the three situations beyond the recorded data: strong current, big waves or big "
                         "starting pushes. Each changes one part of the recorded start law and keeps the other two.",
    "strong-current": f"The shifted situation whose current is {_SC}current/0#f1}}}} to {_SC}current/1#f1}}}}, above "
                      "anything recorded. Holding against it often asks thruster 1 for 80% or more, where the "
                      "propeller stalls: the main test.",
    "big-waves": "The shifted situation whose wave amplitude is {{environment/start_laws/big_waves/wave_amplitude/0#f1}} "
                 "to {{environment/start_laws/big_waves/wave_amplitude/1#f1}}, above the recorded maximum "
                 f"{_R}wave_amplitude/1#f1}}}}: the water's push swings further into the full-throttle region.",
    "big-starting-pushes": "The shifted situation whose floats start three times as far from rest (starting spread "
                           "{{environment/start_laws/big_kicks/float_start_sd#f1}} instead of "
                           f"{_R}float_start_sd#f1}}}}), so the episode starts far from any recorded state.",
    "start-law": "The rules for drawing one situation's starting state: the ranges of the current and the wave's "
                 "amplitude, and the floats' starting spread (Starting State).",
    "stalling-world": f"The world of the data: past 80% throttle a propeller stalls, so its push falls ({STALL90} at "
                      f"90%, {STALL100} at full throttle) while its throttle cost keeps rising.",
    "no-stall-world": "A comparison world identical to the stalling world except that push keeps rising with the "
                      "throttle all the way to full throttle. It shows which part of a result the stall causes.",
    "full-throttle-command": "The recording controller's most common single command: thruster 1 at full throttle "
                             "upstream, with the other thrusters set to match. A bad command, because the propeller "
                             "stalls past 80%.",
    "textbook-rule": "The best linear rule if there were no stall, K: each of the four throttles is a fixed weighted "
                     "sum of the 13 observed numbers. The stall-aware controller is this rule capped at 80%.",
    "reference-controllers": "The fixed controllers every score is measured against: the recording controller "
                             "(0 points), the stall-aware controller (100 points), the rising-thrust controller, "
                             "normal driving only and no throttle.",
    "reward": "The score of one step: minus the float cost (the floats' squared distance from rest) and minus the "
              "throttle cost ({{environment/reward/throttle_cost#f1}} times the squared throttles). The best possible step "
              "scores 0.",
    "discounted": "Later rewards count less: a reward t steps ahead counts {{environment/episode/discount#f2}} to the "
                  "power t as much. Every "
                  "learner on this page uses this discount.",
    "true-value": "The exact discounted sum of the rewards that follow a command in a state while a given controller "
                  "drives afterwards. The physics has no randomness, so it is one simulation away.",
    "seed": "A number that fixes every random choice of a run: the data, the network's starting values and the order "
            "of its training batches. Repeating a run on several seeds shows how much a result owes to luck.",
    "pre-registration": "Every rule and pass mark written down, and its fingerprint (hash) published, before any "
                        "result existed, so that no rule could be bent after seeing an outcome.",
    "hash": "A short fingerprint computed from a file. Changing any character of the file changes it, so a published "
            "hash proves the file was not edited later.",
    "gate": "The four pre-registered harm checks that must pass before BCA is run at all. The gate stopped at the "
            "room check, so BCA was never run.",
    "room-check": "The third harm check: could caution placed at the exact harm help? It certified no shifted "
                  "situation, so the gate stopped.",
    "power-check": "The fourth harm check: how many seeds a fair test of BCA needs, from how much results vary from "
                   "seed to seed. It did not run, because the room check certified no situation.",
    "room-certified": "A shifted situation where caution placed at the exact harm beats even caution of the same "
                      "strength, the same amount of caution spread evenly and the same doses shuffled, each by at "
                      "least 3 points. BCA would be judged only there. No situation was.",
    "dose": "The caution multiplier a training row gets: 1 means no extra pull toward the recorded command. BCA's "
            "doses run from 1 up to 1.5.",
    "over-preference": "How much more the critic prefers the actor's command over the recorded one than the exact "
                       "truth does, counted only when positive: the critic error that draws the actor toward worse "
                       "commands.",
    "band": "BCA's range around the critic's estimate that should contain the critic's one-step error. BCA sets its "
            "caution from it.",
    "noise-map": "The spread of TD3's randomised training target at each row under a perfect critic: how much the "
                 "target jitters there by design. Placing doses by it asks whether BCA follows real critic error or "
                 "only this built-in noise.",
    "lower-bound": "A value the true mean over seeds exceeds with 95% confidence, computed from the spread of the "
                   "seed values. A rule that asks for a lower bound above 0 asks for a gain unlikely to be luck.",
    "pooled": "Taken over all five seeds together.",
    "holm": "Holm's correction: a standard way to test several things at once while keeping the chance of any false "
            "alarm at 5%.",
    "evaluation-bank": "One of four fixed sets of 1,000 starts, one per situation. Every reported return uses them, "
                       "the same for every controller.",
    "behaviour-cloning": "Fitting the actor to copy the recorded commands, with no critic. It is a comparison in the "
                         "BCA test and the common starting point of the actor-only runs.",
    "td3bc": "TD3+BC (twin delayed deep deterministic policy gradient plus behaviour cloning): a standard method that "
             "learns a controller only from recorded data. Its actor climbs its critic, while a pull toward the "
             "recorded commands keeps it near the data.",
    "bca": "Bayesian Conformal Aggregation: an add-on to TD3+BC that estimates where the critic may be wrong and adds "
           "caution there, a harder pull toward the recorded command. It was never run on this environment: the gate "
           "stopped first.",
    "d4rl": "Datasets for Deep Data-Driven Reinforcement Learning: the standard benchmark of recorded datasets. This "
            "page's data use its file format, and the shift uses the removal rule the project used on it.",
    "densest-removal": "The rule that makes the shift: it deletes the 15% of recorded commands that sit closest "
                       "together, and with them almost every full-throttle command.",
    "removed-ball": "The region of commands the densest-15% removal deletes: a ball of radius "
                    "{{dataset/gap/radius#f3}} whose centre has thruster 1 at {{dataset/gap/centre/0#f2}}, close to "
                    "the 80% hold. It holds {{dataset/gap_removal/full_throttle_commands_removed_share#p1}} of the "
                    "full-throttle commands, which make up "
                    "{{dataset/gap_removal/removed_rows_that_are_full_throttle_commands#p1}} of the removed rows.",
    "gapped-critic": "The critic trained on the data after the densest-15% removal, which have almost no record of "
                     "pushing past 80%.",
    "full-data-critic": "The critic trained on the full data, which still show what happens past 80%.",
    "actor-only-run": "A run in which the critic is frozen and only the actor learns, so any difference between two "
                      "runs comes from the critic each trusted.",
    "world-card": "The nine checks on the world itself that need no learning, run on design seeds 900–904 before the "
                  "pre-registration. Every check passed on every seed.",
    "shift-card": "The page's measure of how far the states a controller visits lie from the recorded data, "
                  "controller by controller and situation by situation (The distribution shift).",
    "beats-plain-check": "The BCA test's first actor check: BCA beats plain TD3+BC by at least 2 points on a "
                         "room-certified shifted situation.",
    "placement-check": "The BCA test's second actor check: on a situation where BCA beat plain TD3+BC, BCA also beats "
                       "its own doses spread evenly, its doses shuffled, and even caution of the same strength, so the "
                       "placement of its caution, not its amount, made the difference.",
    "standard-deviation": "The typical size of a random number's spread around its mean.",
    "percentile": "The 99th percentile is the value that 99% of the numbers stay under.",
})
# every controller and learned actor (the controller guide's words), wherever the page names one
TERMS.update(CTRL_TERMS)


# ------------------------------------------------------------------------------------------------ symbols and names

# Symbols: a legend under every displayed formula (or run of formulas), listing every symbol in it, and a tooltip on
# every formula in the text that holds a symbol; both from symbols.SYMS, so a symbol reads the same everywhere.
# Controllers: every place in the text that names one carries data-term="ctrl-...", the same words as the guide.
# A name or formula is reachable by the keyboard at its first appearance in each section (data-tipnf marks the
# later ones, which still answer the pointer and a tap), so the Tab key does not stop hundreds of times.
import symbols as SYM

_SYM_PLAIN = {"arithmetic", "mathematics", "probability", ""}


def _sym_entry(k, tag="div"):
    tex, what, rng, frm = SYM.SYMS[k]
    sym = tex if k.startswith("num_") and not tex.startswith("\\") else f"\\({tex}\\)"
    extra = ""
    if rng:
        extra += f' <span class="sl-x"><b>Range</b> {rng}</span>'
    if frm not in _SYM_PLAIN:
        extra += f' <span class="sl-x"><b>From</b> {frm}</span>'
    return sym, what, extra


def legend_html(keys, title):
    items = []
    for k in keys:
        sym, what, extra = _sym_entry(k)
        items.append(f'<div class="sl"><dt>{sym}</dt><dd>{what}{extra}</dd></div>')
    return (f'<div class="symleg" data-nowrap><p class="symleg-h">{title}</p><dl>' + "".join(items) + "</dl></div>")


def tip_template(tid, keys):
    parts = []
    for k in keys:
        tex, what, rng, frm = SYM.SYMS[k]
        sym = tex if k.startswith("num_") and not tex.startswith("\\") else f"\\({tex}\\)"
        x = []
        if rng:
            x.append(f"Range: {rng}.")
        if frm not in _SYM_PLAIN:
            x.append(f"From: {frm}.")
        parts.append(f'<div class="tt-sym"><span class="tt-h">{sym}</span>{what}'
                     + (f'<span class="rng">{" ".join(x)}</span>' if x else "") + "</div>")
    return f'<template id="{tid}">' + "".join(parts) + "</template>"


def _contexts(html):
    """(start, ctx) boundaries: the section a position lies in; the episode panel is 'sim'; the After-the-stop
    subsection of Results is 'beyond'"""
    marks = [(0, "header")]
    for m in re.finditer(r'<section id="([^"]+)"', html):
        marks.append((m.start(), m.group(1)))
    sim = re.search(r'<figure class="panel" id="sim"', html)
    if sim:
        end = html.index("</figure>", sim.start())
        prev = [c for p, c in marks if p <= sim.start()][-1]
        marks += [(sim.start(), "sim"), (end, prev)]
    by = html.find('id="beyond-the-data"')
    if by > 0:  # until the next section starts
        marks.append((by, "beyond"))
        # Stage 2's new version of BCA, where k is the band penalty's weight, until the next heading
        s2 = html.find("<h4>What must change in BCA</h4>", by)
        if s2 > 0:
            marks.append((s2, "beyond-stage2"))
            nxt = re.compile(r"<h[1-4]\b").search(html, s2 + 5)
            if nxt:
                marks.append((nxt.start(), "beyond"))
    marks.sort()
    return marks


def _ctx_at(marks, pos):
    c = "header"
    for p, n in marks:
        if p > pos:
            break
        c = n
    return c


def add_legends(html):
    """a legend after every run of displayed formulas; <!--legend: below--> right after a run hands its symbols to the
    next run's legend; <!--legend: k1 k2--> adds entries (the numbers in a formula); <!--legend-keys: k1 k2--> is a
    legend of its own (under a diagram)"""
    cut = html.index("<script>")
    head = html[:cut]
    marks = _contexts(head)
    runs = []
    for m in re.finditer(r"(?:\\\[.*?\\\]\s*)+", head, flags=re.S):
        texs = re.findall(r"\\\[(.*?)\\\]", m.group(0), flags=re.S)
        end = m.start() + len(m.group(0).rstrip())
        mk = re.match(r"\s*<!--legend:([^>]*)-->", head[end:])
        extra = mk.group(1).split() if mk else []
        if mk:
            end += mk.end()
        runs.append(dict(start=m.start(), end=end, texs=texs, extra=extra, ctx=_ctx_at(marks, m.start())))
    carry, inserts, carried = [], [], 0
    for r in runs:
        keys = list(carry)
        for t in r["texs"]:
            for k in SYM.symbol_keys(t, r["ctx"]):
                if k not in keys:
                    keys.append(k)
        if r["extra"] == ["below"]:
            carry = keys
            carried += len(r["texs"])
            continue
        carry = []
        nform, carried = len(r["texs"]) + carried, 0
        keys += [k for k in r["extra"] if k not in keys]
        for k in keys:
            if k not in SYM.SYMS:
                sys.exit(f"legend: no symbol entry {k!r}")
        title = "Symbols in this formula" if nform == 1 else "Symbols in these formulas"
        # a formula inside a sentence (the text goes on after it): the legend goes at the end of that list item
        rest = head[r["end"]:]
        at = r["end"]
        if rest.lstrip() and not rest.lstrip().startswith("<"):
            at = r["end"] + min(i for i in (rest.find("</li>"), rest.find("</p>")) if i >= 0)
        inserts.append((at, legend_html(keys, title)))
    if carry:
        sys.exit("legend: below, but no formula follows")
    for at, h in sorted(inserts, reverse=True):
        head = head[:at] + "\n" + h + "\n" + head[at:]
    head = re.sub(r"<!--legend:[^>]*-->", "", head)

    def own(m):
        keys = m.group(1).split()
        for k in keys:
            if k not in SYM.SYMS:
                sys.exit(f"legend-keys: no symbol entry {k!r}")
        return legend_html(keys, m.group(2).strip() if m.group(2) else "Symbols in the diagram")
    head = re.sub(r"<!--legend-keys:([^|>]*)(?:\|([^>]*))?-->", own, head)
    return head + html[cut:], len(inserts)


VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
NOWRAP = {"a", "h1", "h2", "h3", "h4", "h5", "h6", "code", "pre", "button", "summary", "label", "svg", "template",
          "script", "style", "select", "option", "textarea", "caption", "title", "table", "nav", "dt"}
CTRL_PHRASES = [
    (r"new recording controller|this test's recording controller", "iw_recording"),
    (r"[Rr]ecording controller", "recording"),
    (r"[Ss]tall-aware(?: controllers?)?", "stall_aware"),
    (r"[Rr]ising-thrust(?: controllers?)?", "rising_thrust"),
    (r"[Nn]ormal driving only", "normal_driving_only"),
    (r"[Nn]o throttle(?!\s+past)", "no_throttle"),
    (r"[Cc]loned actor|cloned start", "clone"),
    (r"actor of the (?:full-data|full-log) critic|(?:full-data|full-log) critic's actor", "full"),
    (r"actor of the gapped critic|gapped critic's actor", "gapped"),
    (r"[Nn]o extra caution", "room_none"),
    (r"(?:[Hh]arm|[Oo]ver-preference)-placed(?: caution)?", "room_harm_placed"),
    (r"[Ss]trength-matched even(?: caution)?", "room_strength_matched_even"),
    (r"[Aa]mount-matched(?: caution)?", "room_amount_matched"),
    (r"[Ss]huffled caution", "room_shuffled"),
    (r"[Rr]eversed (?:caution|placement)", "room_reversed"),
    (r"[Cc]aution capped at 2|[Ff]actors up to 2", "room_cap2"),
    (r"[Oo]racle penalty|[Tt]he oracle(?! taken| penalty)", "s1_perfect_band"),
    (r"[Ee]ven(?: |-)extra(?: |-)pull", "s1_even_pull"),
    (r"[Pp]ush less everywhere", "s1_push_less"),
    (r"random penalty surfaces?", "s1_random_1"),
    (r"[Cc]lip to the recorded range", "s1_range_hinge"),
    (r"[Dd]istance check", "dep_check_gain"),
    (r"[Dd]ata-only stand-ins?", "standins"),
    (r"stand-ins? for the learned actor|capped stand-in", "forecast"),
]
_WS = r"\s+"
# The same words can name different controllers in different places. In After the stop, "no extra caution" is the
# gate's actor on its gapped critic (Stage 1's arm "none"), not the room check's arm; an element can remap names inside
# it with data-ctrlmap="from:to ..." (to "-" for no tooltip), or data-ctrlmap="none" to name no controller at all
# (the design's forecast, whose stand-in arms are not the gate's).
CTRL_CTX_MAP = {"beyond": {"room_none": "gapped"}, "beyond-stage2": {"room_none": "gapped"}}
# Key terms: the first mention of each in a subsection (between headings) gets the term's tooltip, so a term is
# explained where it is used without underlining every repeat. Spaces in a pattern match any run of white space.
# Controller names come first: at the same place a controller's name wins over a term inside it.
TERM_PHRASES = [
    (r"full-throttle region", "full-throttle-region"),
    (r"full-throttle command", "full-throttle-command"),
    (r"no-stall comparison world", "no-stall-world"),
    (r"stalling world", "stalling-world"),
    (r"stall(?:s|ed|ing)?\b(?!-aware)", "stall"),
    (r"crests?\b", "crest"),
    (r"troughs?\b", "trough"),
    (r"water's push", "water-push"),
    (r"recorded conditions", "recorded-conditions"),
    (r"shifted situations?", "shifted-situation"),
    (r"strong current", "strong-current"),
    (r"big waves", "big-waves"),
    (r"big starting pushes", "big-starting-pushes"),
    (r"start laws?\b", "start-law"),
    (r"densest-15% removal", "densest-removal"),
    (r"removed ball", "removed-ball"),
    (r"gapped critic(?!'s actor)", "gapped-critic"),
    (r"(?:full-data|full-log) critic(?!'s actor)", "full-data-critic"),
    (r"textbook (?:rule|command)", "textbook-rule"),
    (r"reference controllers?", "reference-controllers"),
    (r"actor-only runs?", "actor-only-run"),
    (r"actors?\b(?!-only)", "actor"),
    (r"critics?\b", "critic"),
    (r"over-preference(?!-placed)", "over-preference"),
    (r"caution\b", "caution"),
    (r"doses?\b", "dose"),
    (r"(?:(?<=the\s)|(?<=its\s)|(?<=a\s)|(?<=average\s)|(?<=discounted\s)|(?<=exact\s)|(?<=costs\s)|(?<=cost\s)"
     r"|(?<=episode's\s)|(?<=lost\s)|(?<=more\s))returns?\b", "return"),
    (r"(?<!starting\s)(?<!push\s)points\b(?!\s+(?:upstream|downstream))", "points"),
    (r"episodes?\b", "episode"),
    (r"rewards?\b", "reward"),
    (r"discounted", "discounted"),
    (r"(?:true|exact) values?\b", "true-value"),
    (r"room-certified", "room-certified"),
    (r"room check", "room-check"),
    (r"power check", "power-check"),
    (r"pre-regist(?:ration|ered)", "pre-registration"),
    (r"(?<!-)\bgate\b", "gate"),
    (r"seeds?\b", "seed"),
    (r"hash(?:es)?\b", "hash"),
    (r"lower bound", "lower-bound"),
    (r"pooled", "pooled"),
    (r"Holm(?:'s correction| family)", "holm"),
    (r"evaluation banks?", "evaluation-bank"),
    (r"behaviour cloning", "behaviour-cloning"),
    (r"TD3\+BC", "td3bc"),
    (r"BCA\b(?!'s band)", "bca"),
    (r"BCA's band|(?<=a\s)band\b", "band"),
    (r"noise map", "noise-map"),
    (r"D4RL", "d4rl"),
    (r"world card", "world-card"),
    (r"shift card", "shift-card"),
    (r"beats-plain check", "beats-plain-check"),
    (r"placement check", "placement-check"),
    (r"standard deviation", "standard-deviation"),
    (r"99th percentile", "percentile"),
    (r"distribution shift", "distribution-shift"),
]
_CTRL_RE = re.compile("|".join([f"(?P<g{i}>\\b(?:{p.replace(' ', _WS)}))" for i, (p, _) in enumerate(CTRL_PHRASES)]
                               + [f"(?P<t{i}>\\b(?:{p.replace(' ', _WS)}))" for i, (p, _) in enumerate(TERM_PHRASES)]))
assert not {k for _, k in TERM_PHRASES} - set(TERMS), "a term phrase without a glossary entry"
# The weighting test has its own start law and its own reference levels for points: there these terms would mislead.
TERM_SKIP = {"weighting-test": {"points", "recorded-conditions", "shifted-situation", "strong-current", "big-waves",
                                "big-starting-pushes", "start-law", "evaluation-bank"}}


def _term_contexts(html):
    """the start of every subsection: a heading, a section, the episode panel and its end, a starting-number card"""
    pts = [m.start() for m in re.finditer(r'<(?:h[1-6]|section)\b|<figure class="panel" id="sim"|<div class="sp"|'
                                          r'<div class="check"|<details\b', html)]
    sim = html.find('<figure class="panel" id="sim"')
    if sim >= 0:
        pts.append(html.index("</figure>", sim))
    return sorted(set(pts))


def wrap_names_and_symbols(html):
    import bisect
    cut = html.index("<script>")
    head = html[:cut]
    marks = _contexts(head)
    tmarks = _term_contexts(head)
    tctx = lambda pos: bisect.bisect_right(tmarks, pos)
    seen, tips, out, stack = set(), {}, [], []
    tseen = set()
    excl = 0
    ctrlmaps = []  # the data-ctrlmap remappings of the open elements, outermost first
    stats = dict(names=0, syms=0, terms=0)

    def focus(ctx, key):
        if (ctx, key) in seen:
            return " data-tipnf"
        seen.add((ctx, key))
        return ""

    def do_text(text, pos):
        ctx = _ctx_at(marks, pos)
        parts = re.split(r"(\\\(.*?\\\)|\\\[.*?\\\])", text, flags=re.S)
        res = []
        for i, part in enumerate(parts):
            if i % 2 and part.startswith("\\["):  # a displayed formula: its legend explains it; no names or terms inside
                res.append(part)
                continue
            if i % 2:  # a formula
                keys = SYM.symbol_keys(part[2:-2], ctx)
                if not any(not k.startswith("op_") or k in ("op_norm", "op_normal", "op_unif") for k in keys):
                    res.append(part)
                    continue
                tid = ("sym-" if len(keys) == 1 else "syms-") + "-".join(k.replace(".", "_") for k in keys)
                tips[tid] = keys
                stats["syms"] += 1
                res.append(f'<span class="sym" data-tip-ref="{tid}"{focus(ctx, tid)}>{part}</span>')
            else:
                def name(m):
                    if m.lastgroup[0] == "t":  # a key term: its first mention in this subsection only
                        key = TERM_PHRASES[int(m.lastgroup[1:])][1]
                        if (tctx(pos), key) in tseen or key in TERM_SKIP.get(ctx, ()):
                            return m.group(0)
                        tseen.add((tctx(pos), key))
                        stats["terms"] += 1
                        return f'<span class="term" data-term="{key}"{focus(ctx, "term-" + key)}>{m.group(0)}</span>'
                    key = CTRL_PHRASES[int(m.lastgroup[1:])][1]
                    cmap = dict(CTRL_CTX_MAP.get(ctx, {}))
                    for mp in ctrlmaps:
                        cmap.update(mp)
                    if cmap.get("*") == "-":
                        return m.group(0)
                    key = cmap.get(key, key)
                    if key == "-":
                        return m.group(0)
                    stats["names"] += 1
                    return f'<span class="cname" data-term="ctrl-{key}"{focus(ctx, key)}>{m.group(0)}</span>'
                res.append(_CTRL_RE.sub(name, part))
        return "".join(res)

    for m in re.finditer(r"<!--.*?-->|<[^>]+>|[^<]+", head, flags=re.S):
        tok = m.group(0)
        if tok.startswith("<!--"):
            out.append(tok)
        elif tok.startswith("</"):
            name = tok[2:].strip(" >\n").lower()
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    while len(stack) > i:
                        _n, _e, _m = stack.pop()
                        excl -= _e
                        if _m:
                            ctrlmaps.pop()
                    break
            out.append(tok)
        elif tok.startswith("<"):
            mt = re.search(r'\sdata-term="([^"]+)"', tok)
            if mt:
                tseen.add((tctx(m.start()), mt.group(1)))
            mm = re.match(r"<([A-Za-z][\w:-]*)", tok)
            if mm:
                name = mm.group(1).lower()
                e = int(name in NOWRAP or bool(re.search(r"\sdata-(?:term|tip|tip-ref|nowrap)\b|aria-hidden=\"true\"", tok))
                        or bool(re.search(r'class="[^"]*\b(?:crumb|panel-title|symleg|legend)\b', tok)))
                mc = re.search(r'\sdata-ctrlmap="([^"]*)"', tok)
                cm = None
                if mc:
                    v = mc.group(1).strip()
                    cm = {"*": "-"} if v == "none" else dict(x.split(":", 1) for x in v.split())
                    assert all(t == "-" or "ctrl-" + t in CTRL_TERMS for t in cm.values()), v
                if not (tok.endswith("/>") or name in VOID):
                    stack.append((name, e, cm is not None))
                    excl += e
                    if cm is not None:
                        ctrlmaps.append(cm)
            out.append(tok)
        else:
            out.append(tok if excl else do_text(tok, m.start()))
    for k in {k for v in tips.values() for k in v}:
        if k not in SYM.SYMS:
            sys.exit(f"no symbol entry {k!r}")
    templates = "\n".join(tip_template(tid, keys) for tid, keys in sorted(tips.items()))
    return "".join(out) + "\n<!-- symbol tooltips: one template per formula's set of symbols -->\n" + templates + "\n" \
        + html[cut:], stats, len(tips)


# ------------------------------------------------------------------------------------------------ assemble

src = open("template.html", encoding="utf8").read()
assert src.count("/*__DATA__*/null") == 1
for name, frag in FRAG.items():
    tag = f"<!--@{name}@-->"
    if src.count(tag) != 1:
        sys.exit(f"fragment {name}: {src.count(tag)} placeholders")
    src = src.replace(tag, frag)
left = re.findall(r"<!--@[A-Z_0-9]+@-->", src)
if left:
    sys.exit(f"unfilled fragments: {left}")
src = fill_numbers(src)
# keep each inline formula together with the punctuation right after it, so no line starts with ':', ',' or ').'
_i = src.index("<script>")
src = re.sub(r'(\\\((?:(?!\\\))[^<])*?\\\))([.,;:)]+)', r'<span class="nw">\1\2</span>', src[:_i]) + src[_i:]
src, _nleg = add_legends(src)
src, _wstats, _ntips = wrap_names_and_symbols(src)
src = fill_numbers(src)  # the legends' and the symbol tooltips' numbers
if "{{" in src.split("<script>")[0]:
    i = src.index("{{")
    sys.exit(f"unfilled number near: {src[i - 80:i + 80]}")
src = src.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

# tooltips: every data-term names a glossary entry, no tooltip text holds TeX (the math renderer would rewrite the
# attribute), and every table names and describes its columns (table(..., tips=...), or data-tip on a hand-written th;
# a table with no header row carries data-cols and data-coltips)
_page = src.split("<script>")[0]
_used = set(re.findall(r'data-term="([^"]+)"', _page))
if _used - set(TERMS):
    sys.exit(f"data-term without a glossary entry: {sorted(_used - set(TERMS))}")
for _m in re.finditer(r'data-(?:tip|cols|coltips)="([^"]*)"', _page):
    if "\\(" in _m.group(1) or "\\[" in _m.group(1) or "{{" in _m.group(1):
        sys.exit(f"tooltip text with TeX or an unfilled number: {_m.group(1)[:120]}")
_ntab = 0
for _tm in re.finditer(r"<table\b[^>]*>.*?</table>", _page, flags=re.S):
    _tb, _ntab = _tm.group(0), _ntab + 1
    _open = _tb[:_tb.index(">")]
    if "data-notip" in _open:
        continue
    _thead = re.search(r"<thead>(.*?)</thead>", _tb, flags=re.S)
    if not _thead:
        if "data-cols=" not in _open or "data-coltips=" not in _open:
            sys.exit(f"a table without a header row needs data-cols and data-coltips: {plain(_tb)[:100]}")
        continue
    for _th in re.finditer(r"<th\b([^>]*)>(.*?)</th>", _thead.group(1), flags=re.S):
        if not plain(_th.group(2)).strip() and "colgroup" in _th.group(1):
            continue  # an empty group heading over a column that has its own
        if 'data-tip="' not in _th.group(1):
            sys.exit(f"a column without a description (data-tip): {plain(_th.group(2)).strip()!r} in "
                     f"{plain(_tb)[:80]!r}")
_terms = {k: fill_numbers(v) for k, v in TERMS.items()}
assert all("{{" not in v and "</" not in v and "\\(" not in v for v in _terms.values()), "a glossary entry is not plain text"
assert src.count("/*__TERMS__*/null") == 1
src = src.replace("/*__TERMS__*/null", json.dumps(_terms, ensure_ascii=False))
# a table row whose first cell names a fixed controller (or carries data-ctrl) shows that controller's explanation
# under its row tuple, the same words as everywhere else the controller is named
_bad_ctrl = set(re.findall(r'data-ctrl="([^"]+)"', _page)) - set(TERMS)
if _bad_ctrl:
    sys.exit(f"data-ctrl without a controller entry: {sorted(_bad_ctrl)}")
assert src.count("/*__CTRLNAMES__*/null") == 1
src = src.replace("/*__CTRLNAMES__*/null", json.dumps({v.lower(): "ctrl-" + k for k, v in CTRL_LABEL.items()},
                                                      ensure_ascii=False))
print(f"tooltips: {_ntab} tables with column descriptions, {len(_used)} of {len(TERMS)} glossary terms used")
print(f"symbols: {_nleg} formula legends, {_wstats['syms']} formulas in the text with a symbol tooltip "
      f"({_ntips} distinct); controller names marked: {_wstats['names']}; key terms marked: {_wstats['terms']}")
open("thrusterchain.html", "w", encoding="utf8").write(src)
print(f"wrote thrusterchain.html: {len(src):,} chars, {len(FRAG)} fragments")
