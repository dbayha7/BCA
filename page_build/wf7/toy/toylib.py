"""Helpers for the toy-worlds section: worked-arithmetic blocks, tables and static SVG line charts in the page's
chart style (svg.chart, grid/tick/axis/axlab classes, a legend under the chart), with one hover target per x column
whose tooltip (the page's shared tooltip, data-tip-ref to a <template>) lists every series with its color key."""
import html
import math
import re

MINUS = "−"


def num(x, d=3, sign=False):
    """A shown number: fixed decimals, U+2212 for negatives."""
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", MINUS)


def esc(s):
    return html.escape(s, quote=True)


def fr(a, b):
    """A stacked fraction; screen readers hear 'a divided by b'."""
    return (f'<span class="toy-fr"><span class="toy-nu">{a}</span><span class="toy-sr"> divided by </span>'
            f'<span class="toy-de">{b}</span></span>')


def calc(label, rows, so=None, cid=None):
    """A worked-arithmetic block: rows of (left side, right side); a left side of '' continues the previous line,
    so the equals signs line up. label names the block for screen readers."""
    idattr = f' id="{cid}"' if cid else ""
    out = [f'<div class="toy-calc"{idattr} role="group" aria-label="{esc(label)}">',
           f'<span class="toy-ct" aria-hidden="true">{label}</span>']
    for lhs, rhs in rows:
        if lhs is None:          # a full-width line of words between steps
            out.append(f'<span class="toy-cw">{rhs}</span>')
        elif lhs == "so":        # a conclusion: an arrow instead of an equals sign
            out.append(f'<span class="toy-cl"></span><span class="toy-ce">⇒</span><span class="toy-cr">{rhs}</span>')
        else:
            out.append(f'<span class="toy-cl">{lhs}</span><span class="toy-ce">=</span><span class="toy-cr">{rhs}</span>')
    out.append('</div>')
    if so:
        out.append(f'<p class="toy-so"><b>So:</b> {so}</p>')
    return "\n".join(out)


def table(tid, caption, heads, rows, num_cols=(), cls="stack", note=None):
    """A data table in the page's style; heads: column names; rows: lists of cell HTML; num_cols: indexes of
    numeric columns (right-aligned)."""
    idattr = f' id="{tid}"' if tid else ""
    h = "".join(f'<th{" class=" + chr(34) + "n" + chr(34) if i in num_cols else ""} scope="col">{c}</th>'
                for i, c in enumerate(heads))
    body = []
    for r in rows:
        cells = "".join(f'<td{" class=" + chr(34) + "n" + chr(34) if i in num_cols else ""}>{c}</td>'
                        for i, c in enumerate(r))
        body.append(f"<tr>{cells}</tr>")
    s = (f'<div class="tbl"><table{idattr} class="{cls}">\n      <caption><span class="cap">{caption}</span></caption>\n'
         f'      <thead><tr>{h}</tr></thead>\n      <tbody>\n        ' + "\n        ".join(body) +
         '\n      </tbody>\n    </table></div>')
    if note:
        s += f'\n    <p class="note">{note}</p>'
    return s


# ------------------------------------------------------------------ charts
LABEL_W = 7.3     # a generous width per character of a direct label (12.5 px, 13.5 px on phones), in chart units


def _seg_hits_box(seg, box):
    """Does the segment (x1, y1, x2, y2) touch the rectangle (x0, y0, x1, y1)? (Liang-Barsky clipping.)"""
    x1, y1, x2, y2 = seg
    bx0, by0, bx1, by1 = box
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - bx0), (dx, bx1 - x1), (-dy, y1 - by0), (dy, by1 - y1)):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return True


def _boxes_meet(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _pt_seg(px, py, seg):
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _box_seg_dist(box, seg):
    """The distance between a rectangle and a segment (0 when they meet)."""
    if _seg_hits_box(seg, box):
        return 0.0
    bx0, by0, bx1, by1 = box
    x1, y1, x2, y2 = seg
    d = min(_pt_seg(cx, cy, seg) for cx, cy in ((bx0, by0), (bx0, by1), (bx1, by0), (bx1, by1)))
    for px, py in ((x1, y1), (x2, y2)):
        ddx = max(bx0 - px, 0, px - bx1)
        ddy = max(by0 - py, 0, py - by1)
        d = min(d, math.hypot(ddx, ddy))
    return d


def _place(lab, series, pos, Y, obstacles, boxes, plot):
    """The first free spot for a direct label: lab = {series, text, cols (preferred columns, in order)}.
    Tries, at each preferred column, eight spots around the series' point; returns (x, y, anchor) or None."""
    si = lab["series"]
    text = re.sub(r"<[^>]+>|&[a-z]+;", "x", lab["text"])
    tw = LABEL_W * len(text)
    spots = [(7, -7, "start"), (7, 15, "start"), (-7, -7, "end"), (-7, 15, "end"), (0, -11, "middle"),
             (0, 19, "middle")]
    for dy in (-17, 25, -27, 35, -37, 45):
        spots += [(7, dy, "start"), (-7, dy, "end"), (0, dy - 4 if dy < 0 else dy + 4, "middle")]
    pref = list(lab.get("cols", []))
    order = pref + [c for c in range(len(pos) - 1, -1, -1) if c not in pref]
    vals = series[si]["values"]
    points = [(pos[ci], Y(vals[ci])) for ci in order if vals[ci] is not None]
    points += [((pos[ci] + pos[ci + 1]) / 2, (Y(vals[ci]) + Y(vals[ci + 1])) / 2) for ci in order
               if ci + 1 < len(pos) and vals[ci] is not None and vals[ci + 1] is not None]
    # first beside the line; failing that, on the label's own line (its halo cuts the line), clear of the others
    own = set(range(*lab["seg"])) if "seg" in lab else set()
    own_box = set(range(*lab["box"])) if "box" in lab else set()
    for inline in (False, True):
        for px, py in (points[len([v for v in vals if v is not None]):] if inline else points):
            for dx, dy, anchor in ([(0, 4, "middle")] if inline else spots):
                x, y = px + dx, py + dy
                x0 = x if anchor == "start" else x - tw if anchor == "end" else x - tw / 2
                box = (x0 - 2, y - 12, x0 + tw + 2, y + 4)
                if box[0] < plot[0] + 2 or box[2] > plot[2] or box[1] < plot[1] or box[3] > plot[3] - 1:
                    continue
                if any(_seg_hits_box(sg, box) for k, sg in enumerate(obstacles) if not (inline and k in own)):
                    continue
                if any(_boxes_meet(box, b) for k, b in enumerate(boxes) if not (inline and k in own_box)):
                    continue
                # the label must hug its own line and sit clearly nearer to it than to any other line
                d_own = min((_box_seg_dist(box, obstacles[k]) for k in own), default=0.0)
                d_other = min((_box_seg_dist(box, sg) for k, sg in enumerate(obstacles) if k not in own),
                              default=1e9)
                if d_own > 10 or d_other < d_own + 6:
                    continue
                return x, y, anchor
    return None


MARKS = {  # marker shapes: a second cue besides color
    "circle": lambda x, y, r: f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>',
    "square": lambda x, y, r: f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1"/>',
    "diamond": lambda x, y, r: (f'<path d="M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y:.1f} L{x:.1f},{y + r + 1:.1f} '
                                f'L{x - r - 1:.1f},{y:.1f} Z"/>'),
    "tri": lambda x, y, r: (f'<path d="M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y + r:.1f} '
                            f'L{x - r - 1:.1f},{y + r:.1f} Z"/>'),
    "trid": lambda x, y, r: (f'<path d="M{x:.1f},{y + r + 1:.1f} L{x + r + 1:.1f},{y - r:.1f} '
                             f'L{x - r - 1:.1f},{y - r:.1f} Z"/>'),
}


def swatch(color, dash=False, mark="circle", hollow=False):
    """A legend key: a short line in the series color with its marker, as a tiny inline SVG (aria-hidden)."""
    cls = f'toy-s {"toy-dash" if dash else ""}'.strip()
    m = MARKS[mark](11, 5, 3).replace("/>", f' class="toy-m{" toy-hollow" if hollow else ""}"/>', 1)
    return (f'<svg class="toy-sw" viewBox="0 0 22 10" aria-hidden="true" style="--c:var({color})">'
            f'<line class="{cls}" x1="1" x2="21" y1="5" y2="5"/>{m}</svg>')


def line_chart(cid, aria, xs, xlabels, xpos, series, ylo, yhi, yticks, ytick_fmt, xtitle, ytitle, tip_title, tip_val,
               guides=(), hguides=(), w=440, h=280, left=66, right=14, top=16, bottom=58, legend_note=None,
               labels=()):
    """A static SVG line chart.
    xs: x keys; xlabels: tick labels; xpos(i) -> 0..1 position of column i;
    series: list of dicts {name, color (CSS var name), values (None = no point), dash, mark, hollow};
    guides: vertical guide lines [(pos 0..1, label, anchor)]; hguides: horizontal [(y, label)];
    labels: direct labels [(series index, column index, dx, dy, anchor, text)], in ink (text tokens, never the
    series color), beside the named point;
    tip_title(i) -> tooltip heading of column i; tip_val(v) -> shown value."""
    pw, ph = w - left - right, h - top - bottom
    X = lambda p: left + p * pw
    Y = lambda v: top + (1 - (v - ylo) / (yhi - ylo)) * ph
    o = [f'<svg class="chart toy-chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(aria)}">']
    for t in yticks:
        o.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{ytick_fmt(t)}</text>')
    for i, lab in enumerate(xlabels):
        o.append(f'<text class="tick" x="{X(xpos(i)):.1f}" y="{top + ph + 16}" text-anchor="middle">{lab}</text>')
    o.append(f'<line class="axis" x1="{left}" x2="{w - right}" y1="{top + ph}" y2="{top + ph}"/>'
             f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + ph}"/>')
    for p, lab, anchor in guides:
        x = X(p)
        o.append(f'<line class="knee" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + ph}"/>')
        if lab:
            dx = 4 if anchor == "start" else -4
            o.append(f'<text class="ann toy-halo" x="{x + dx:.1f}" y="{top + 10}" text-anchor="{anchor}">{lab}</text>')
    for v, lab in hguides:
        o.append(f'<line class="knee" x1="{left}" x2="{w - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        if lab:
            o.append(f'<text class="ann toy-halo" x="{w - right - 4}" y="{Y(v) - 5:.1f}" text-anchor="end">{lab}</text>')
    # hover targets first, so the lines and markers above let the pointer through to them (page CSS: .pt-hits ~ *)
    pos = [X(xpos(i)) for i in range(len(xs))]
    o.append('<g class="pt-hits">')
    for i in range(len(xs)):
        lo = (pos[i - 1] + pos[i]) / 2 if i > 0 else left
        hi = (pos[i] + pos[i + 1]) / 2 if i + 1 < len(xs) else w - right
        o.append(f'<rect x="{lo:.1f}" y="{top}" width="{hi - lo:.1f}" height="{ph}" data-tip-focus="no" '
                 f'data-tip-title="{esc(tip_title(i))}" data-tip-ref="{cid}-t{i}"/>')
    o.append('</g>')
    for s in series:
        pts = [(pos[i], Y(v)) for i, v in enumerate(s["values"]) if v is not None]
        # break the line where a value is missing
        segs, cur = [], []
        for i, v in enumerate(s["values"]):
            if v is None:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append((pos[i], Y(v)))
        if cur:
            segs.append(cur)
        cls = "toy-ln" + (" toy-dash" if s.get("dash") else "")
        for sg in segs:
            if len(sg) > 1:
                d = " ".join(("M" if k == 0 else "L") + f"{x:.1f},{y:.1f}" for k, (x, y) in enumerate(sg))
                o.append(f'<path class="{cls}" style="--c:var({s["color"]})" d="{d}"/>')
        mk = MARKS[s.get("mark", "circle")]
        g = [f'<g class="toy-pts" style="--c:var({s["color"]})">']
        for x, y in pts:
            g.append(mk(x, y, 3.2).replace("/>", f' class="toy-m{" toy-hollow" if s.get("hollow") else ""}"/>', 1))
        g.append('</g>')
        o.append("".join(g))
    # direct labels, placed where they cross no line, marker, guide or other label
    obstacles = []                                   # segments (x1, y1, x2, y2)
    seg_range, box_range = {}, {}
    for si, s in enumerate(series):
        k0 = len(obstacles)
        prev = None
        for i, v in enumerate(s["values"]):
            cur = None if v is None else (pos[i], Y(v))
            if prev and cur:
                obstacles.append((prev[0], prev[1], cur[0], cur[1]))
            prev = cur
        seg_range[si] = (k0, len(obstacles))
    # (dashed guide lines are not obstacles: a label's halo covers them)
    boxes = []                                       # rectangles (x0, y0, x1, y1) of texts and markers
    for si, s in enumerate(series):
        k0 = len(boxes)
        for i, v in enumerate(s["values"]):
            if v is not None:
                boxes.append((pos[i] - 4.5, Y(v) - 4.5, pos[i] + 4.5, Y(v) + 4.5))
        box_range[si] = (k0, len(boxes))
    for p, lab, anchor in guides:
        if lab:
            tw = LABEL_W * len(lab)
            x0 = X(p) + 4 if anchor == "start" else X(p) - 4 - tw
            boxes.append((x0, top, x0 + tw, top + 13))
    for v, lab in hguides:
        if lab:
            tw = LABEL_W * len(lab)
            boxes.append((w - right - 4 - tw, Y(v) - 16, w - right - 4, Y(v) - 3))
    for lab in labels:
        if isinstance(lab, tuple):
            si, ci, dx, dy, anchor, text = lab
            v = series[si]["values"][ci]
            x, y = pos[ci] + dx, Y(v) + dy
        else:
            lab = dict(lab, seg=seg_range[lab["series"]], box=box_range[lab["series"]])
            placed = _place(lab, series, pos, Y, obstacles, boxes, (left, top, w - right, top + ph))
            if not placed and lab.get("optional"):
                continue
            if not placed:
                raise ValueError(f"{cid}: no free place for the label {lab['text']!r}")
            x, y, anchor = placed
        tw = LABEL_W * len(re.sub(r"<[^>]+>|&[a-z]+;", "x", lab[5] if isinstance(lab, tuple) else lab["text"]))
        x0 = x if anchor == "start" else x - tw if anchor == "end" else x - tw / 2
        boxes.append((x0, y - 11, x0 + tw, y + 3))
        text = lab[5] if isinstance(lab, tuple) else lab["text"]
        o.append(f'<text class="toy-dl toy-halo" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
                 f'aria-hidden="true">{text}</text>')
    o.append(f'<text class="axlab" x="{left + pw / 2:.1f}" y="{h - 10}" text-anchor="middle">{xtitle}</text>')
    o.append(f'<text class="axlab" transform="translate(16 {top + ph / 2:.1f}) rotate(-90)" text-anchor="middle">'
             f'{ytitle}</text>')
    o.append('</svg>')
    # the tooltip of each column: every series, with its key, in the legend's order
    tpl = []
    for i in range(len(xs)):
        rows = []
        for s in series:
            v = s["values"][i]
            vs = "no value" if v is None else tip_val(v)
            rows.append(f'<div class="toy-key">{swatch(s["color"], s.get("dash"), s.get("mark", "circle"), s.get("hollow"))}'
                        f'<span>{s["name"]}: <b>{vs}</b></span></div>')
        tpl.append(f'<template id="{cid}-t{i}">{"".join(rows)}</template>')
    leg = ['<div class="legend toy-legend">']
    for s in series:
        leg.append(f'<span>{swatch(s["color"], s.get("dash"), s.get("mark", "circle"), s.get("hollow"))}{s["name"]}</span>')
    leg.append('</div>')
    if legend_note:
        leg.append(f'<p class="note toy-cnote">{legend_note}</p>')
    return "\n".join(o), "\n".join(leg), "\n".join(tpl)
