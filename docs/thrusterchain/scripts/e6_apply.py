"""Edit 2 of the tooltip pass (comments 6, 8, 9 and 10): content tooltips, legends, the controller guide, the lead-in
sentence about no throttle and the worked-example line. All texts are in e6_content.py.

Exact, asserted edits on thrusterchain.html: every fixed anchor is matched exactly once, and every term or symbol
wrapped in running text is replaced through a context string that is itself matched exactly once.
Never touches thrusterchain.v9.html. Run:
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/e6_apply.py'
Add --dry to write tmp/e6/dry.html instead of the page.
"""
import hashlib
import html as H
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e6_content as C  # noqa: E402

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
BACKUP = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v9.html"
DRY = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/e6/dry.html"
LOG = []


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def once(s, old, new, what):
    n = s.count(old)
    assert n == 1, f"{what}: expected exactly 1 match, found {n}: {old[:90]!r}"
    LOG.append(what)
    return s.replace(old, new, 1)


def esc(x):
    return H.escape(x, quote=True)


def attrs(tip, focus=True, ref=None):
    title, body = tip
    a = f' data-tip-title="{esc(title)}"'
    a += f' data-tip-ref="{ref}"' if ref else f' data-tip="{esc(body)}"'
    if not focus:
        a += ' data-tip-focus="no"'
    return a


# ---------------------------------------------------------------- a scan of the page body: text and inline math
TAG = re.compile(r'<!--.*?-->|<(/?)([a-zA-Z][\w:-]*)((?:[^>"\']|"[^"]*"|\'[^\']*\')*)>', re.S)
VOID = {"br", "img", "input", "meta", "link", "hr", "wbr", "source", "col", "area", "base", "param", "track", "embed"}
SKIP_TAGS = {"table", "button", "a", "summary", "svg", "nav", "template", "dt", "select", "textarea", "label",
             "canvas", "h1", "h2", "h3", "h4", "code", "pre", "title", "figcaption-none"}
SKIP_CLASSES = {"ro", "legend", "symkey", "ctrl-guide", "tt-sr", "optlbl", "tlabel", "panel-title", "g-k", "tag",
                "crumb", "readouts", "td-k", "transport", "sim-k", "f-title", "dr", "iw-controls", "iw-formula", "status-tag",
                "opt", "wt-bar", "bar-notes"}


def scan(s, end):
    """Yield ('text', start, stop, stack) and ('mjx', start, stop, stack, label, display) over s[:end].
    The stack holds (tag, attrs, start, start-tag end) of the open elements."""
    pos, stack = 0, []
    while pos < end:
        m = TAG.search(s, pos, end)
        if not m:
            yield ("text", pos, end, tuple(stack))
            break
        if m.start() > pos:
            yield ("text", pos, m.start(), tuple(stack))
        if m.group(0).startswith("<!--"):
            pos = m.end()
            continue
        closing, name, at = m.group(1), m.group(2).lower(), m.group(3)
        if not closing and name == "mjx-container":
            stop = s.index("</mjx-container>", m.end()) + len("</mjx-container>")
            lab = re.search(r'aria-label="([^"]*)"', at)
            yield ("mjx", m.start(), stop, tuple(stack), H.unescape(lab.group(1)) if lab else "", 'display="true"' in at)
            pos = stop
            continue
        if not closing and name in ("script", "style", "template"):
            pos = s.index(f"</{name}>", m.end()) + len(name) + 3
            continue
        if closing:
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    del stack[i:]
                    break
        elif at.rstrip().endswith("/") or name in VOID:
            pass
        else:
            stack.append((name, at, m.start(), m.end()))
        pos = m.end()


def excluded(stack):
    for name, at, _, _ in stack:
        if name in SKIP_TAGS or "data-tip" in at:
            return True
        cm = re.search(r'class="([^"]*)"', at)
        if cm and SKIP_CLASSES & set(cm.group(1).split()):
            return True
    return False


def section_of(stack):
    for name, at, _, _ in stack:
        if name == "section":
            m = re.search(r'id="([^"]+)"', at)
            if m:
                return m.group(1)
    return "-"


def norm(label):
    return re.sub(r"\s+", " ", label).strip()


def plan_inline(s):
    """The term and symbol edits: (start, stop, replacement)."""
    end = s.index("<script")
    events = list(scan(s, end))
    edits, seen_sym, used = [], set(), set()
    # symbols
    for ev in events:
        if ev[0] != "mjx" or ev[5]:
            continue
        _, a, b, stack, label, _ = ev
        lab = norm(label)
        sec = section_of(stack)
        tip = C.SYM_SECTION.get((sec, lab)) or C.SYM.get(lab)
        if not tip or excluded(stack):
            continue
        first = (sec, lab) not in seen_sym
        seen_sym.add((sec, lab))
        edits.append((a, b, f'<span class="sym"{attrs(tip, focus=first)}>' + s[a:b] + "</span>"))
    nsym = len(edits)
    # terms: the first match after each marker
    for marker, rx, k in C.TERMS:
        mpos = s.index(marker)
        assert s.count(marker) == 1, f"term marker not unique: {marker}"
        hit = None
        for ev in events:
            if ev[0] != "text" or ev[2] <= mpos:
                continue
            _, a, b, stack = ev
            a = max(a, mpos)
            m = re.search(rx, s[a:b + 12])
            if m and m.end() > b - a:
                m = None
            if not m:
                continue
            if excluded(stack) or (a + m.start()) in used:
                continue
            hit = (a + m.start(), a + m.end(), stack)
            break
        assert hit, f"term not found after marker: {k} {marker}"
        p, q, stack = hit
        used.add(p)
        tip = C.T[k]
        parent = stack[-1] if stack else None
        whole = parent and parent[0] in ("i", "b") and s[parent[3]:p].strip() == "" and \
            s[q:q + 3 + len(parent[0])] == f"</{parent[0]}>"
        if whole and "data-tip" not in parent[1]:
            ps, pe = parent[2], parent[3]
            tagtext = s[ps:pe]
            edits.append((ps, pe, tagtext[:-1] + attrs(tip) + ">"))
        else:
            edits.append((p, q, f'<span class="term"{attrs(tip)}>' + s[p:q] + "</span>"))
    return edits, nsym


def apply_inline(s, edits):
    edits = sorted(edits, key=lambda e: e[0], reverse=True)
    for i in range(1, len(edits)):
        assert edits[i][1] <= edits[i - 1][0], "overlapping inline edits"
    for a, b, new in edits:
        k = 40
        while True:
            lo, hi = max(0, a - k), min(len(s), b + k)
            old = s[lo:hi]
            if s.count(old) == 1:
                break
            k *= 2
            assert k < 20000, "no unique context"
        s = s[:lo] + s[lo:a] + new + s[b:hi] + s[hi:]
    return s


# ---------------------------------------------------------------- blocks of new content
def legend_html(heading, items):
    rows = "".join(f"<div><dt>{sym}</dt><dd>{text}</dd></div>" for sym, text in items)
    return f'<div class="symkey"><p class="symkey-h">{heading}</p><dl>{rows}</dl></div>'


def facets(thr, lim, score):
    # the spaces between paragraphs keep the sentences apart in the accessible description (the template's text)
    return (f"<p><b>Sets the throttles:</b> {esc(thr)}</p> <p><b>The 80% limit:</b> {esc(lim)}</p> "
            f"<p><b>Its score:</b> {esc(score)}</p>")


SWATCH = {"recording": '<i class="sw" style="background:var(--c-rec)"></i>',
          "stall_aware": '<i class="sw" style="background:var(--c-sa)"></i>',
          "rising_thrust": '<i class="sw" style="background:var(--c-rt)"></i>',
          "no_throttle": '<i class="sw dash"></i>'}


def guide_html():
    tpl = "".join(f'<template id="tip-ctrl-{k}">{facets(t, l, sc)}</template>' for k, n, t, l, sc in C.CTRL_PANEL)
    panel = "".join(f"<div><dt>{SWATCH[k]}{esc(n)}</dt><dd>{facets(t, l, sc)}</dd></div>"
                    for k, n, t, l, sc in C.CTRL_PANEL)
    other = "".join(f"<div><dt>{esc(n)}</dt><dd>{facets(t, l, sc)}</dd></div>" for n, _, t, l, sc in C.CTRL_OTHER)
    learned = "".join(f"<div><dt>{esc(n)}</dt><dd>{facets(t, l, sc)}</dd></div>" for n, t, l, sc in C.CTRL_LEARNED)
    return (
        tpl + '\n      <details class="ctrl-guide" open>\n'
        '        <summary>All the controllers on this page: how each sets the four throttles, whether it stays within '
        '80%, and where its score comes from</summary>\n'
        '        <div class="inner">\n'
        '          <p class="note">A score is in points: a controller\'s exact average return over a situation\'s 1,000 '
        'starts, placed on a scale where 0 is the recording controller and 100 the stall-aware controller (in the '
        'no-stall comparison world, the rising-thrust controller). Numbers are for the stalling world unless marked.</p>\n'
        '          <p class="cg-h">In this panel</p>\n'
        f'          <dl class="cg">{panel}</dl>\n'
        '          <p class="cg-h">Another fixed controller</p>\n'
        f'          <dl class="cg">{other}</dl>\n'
        '          <p class="cg-h">Learned actors</p>\n'
        '          <p class="note">An actor learns only from the recorded data and never acts in the water while it '
        'learns. Its points are exact averages of the trained actor over each bank; "means over seeds 0–4" averages '
        'five repeat runs.</p>\n'
        f'          <dl class="cg">{learned}</dl>\n'
        '        </div>\n'
        '      </details>\n      ')


def main():
    dry = "--dry" in sys.argv
    backup_md5 = md5(BACKUP)
    s = open(PAGE, encoding="utf-8").read()
    assert 'class="symkey"' not in s and "e6_tooltips" not in s and 'class="sym"' not in s, "edit 2 already applied"

    # 1. terms and symbols in running text (before anything else adds text)
    edits, nsym = plan_inline(s)
    s = apply_inline(s, edits)
    LOG.append(f"inline: {nsym} symbols, {len(edits) - nsym} terms")

    # 2. CSS before the only </style>
    css = open(os.path.join(HERE, "e6_tooltips.css"), encoding="utf-8").read()
    s = once(s, "</style>", css + "</style>", "css")

    # 3. the live panel: options, transport, readouts, drawings, the drawing's legend items
    for k, tip in C.SIT_TIPS.items():
        s = once(s, f'id="opt-sit-{k}"', f'id="opt-sit-{k}"' + attrs(tip), f"opt sit {k}")
    for k, tip in C.WORLD_TIPS.items():
        s = once(s, f'id="opt-world-{k}"', f'id="opt-world-{k}"' + attrs(tip), f"opt world {k}")
    for k, name, *_ in C.CTRL_PANEL:
        s = once(s, f'id="opt-ctrl-{k}"', f'id="opt-ctrl-{k}"' + attrs((name, ""), ref=f"tip-ctrl-{k}"), f"opt ctrl {k}")
    for k, i in (("play", "play"), ("back", "back"), ("fwd", "fwd"), ("slider", "tSlider")):
        s = once(s, f'id="{i}"', f'id="{i}"' + attrs(C.TRANSPORT_TIPS[k]), f"transport {k}")
    for i, tip in C.TD_TRANSPORT_TIPS.items():
        s = once(s, f'id="{i}"', f'id="{i}"' + attrs(tip), f"demo transport {i}")
    ro = {
        "state": '<div class="ro"><span class="k">state <mjx-container',
        "push": "<div class=\"ro\"><span class=\"k\">water's push on every float (full-throttle region above 0.565)</span>",
        "float": '<div class="ro"><span class="k">float cost</span>',
        "throttle": '<div class="ro"><span class="k">throttle cost</span>',
        "reward": '<div class="ro"><span class="k">reward (minus the two costs)</span>',
        "return": '<div class="ro"><span class="k">return so far</span>',
        "J": '<div class="ro"><span class="k">exact average return, 1,000 starts</span>',
        "points": '<div class="ro"><span class="k">points, 1,000 starts</span>',
    }
    for k, old in ro.items():
        s = once(s, old, old.replace('<div class="ro"', '<div class="ro"' + attrs(C.READOUT_TIPS[k]), 1), f"readout {k}")
    td = {
        "push": "<div class=\"ro\"><span class=\"k\">water's push on every float</span><span class=\"v\" id=\"td-push\">",
        "thr": "<div class=\"ro\"><span class=\"k\">thruster 1's throttle (the bar's fill)</span>",
        "thrpush": "<div class=\"ro\"><span class=\"k\">thruster 1's push (the dark tick)</span>",
        "ret": '<div class="ro"><span class="k">return so far (closer to 0 is better)</span>',
    }
    for k, old in td.items():
        s = once(s, old, old.replace('<div class="ro"', '<div class="ro"' + attrs(C.TD_READOUT_TIPS[k]), 1),
                 f"demo readout {k}")
    s = once(s, '<canvas id="channel"', '<canvas id="channel"' + attrs(C.CANVAS_CHANNEL_TIP), "canvas channel")
    s = once(s, '<canvas id="cost"', '<canvas id="cost"' + attrs(C.CANVAS_COST_TIP), "canvas cost")
    sw = {"Recording controller": "var(--c-rec)", "Stall-aware controller": "var(--c-sa)",
          "Rising-thrust controller": "var(--c-rt)"}
    for name, body in C.COST_LEGEND_TIPS.items():
        inner = f'<i class="sw" style="background:{sw[name]}"></i>' if name in sw else '<i class="sw dash"></i>'
        old = f"<span>{inner}{name}</span>"
        s = once(s, old, f"<span{attrs((name, body), focus=False)}>{inner}{name}</span>", f"cost legend {name}")

    # 4. comment 9: the lead-in sentence on no throttle, and the controller guide
    old = "gets shorter even though its throttle is higher.</li>"
    s = once(s, old, old + "\n              " + C.LEAD_BULLET, "lead-in bullet")
    old = "<figcaption>Every episode here was run"
    s = once(s, old, guide_html() + old, "controller guide")

    # 5. comment 10: legends beside every formula and drawing
    for anchor, heading, items in C.LEGENDS:
        s = once(s, anchor, "\n    " + legend_html(heading, items) + anchor, f"legend: {heading} @ {anchor.strip()[:40]}")

    # 6. parts of drawings
    for suffix, tip in C.SVG_TIPS:
        assert s.count(suffix) == 1, f"svg text not unique: {suffix}"
        p = s.index(suffix)
        t0 = s.rindex("<text", 0, p)
        old = s[t0:p + len(suffix)]
        s = once(s, old, "<text" + attrs(tip, focus=False) + old[5:], f"svg tip {suffix[:40]}")

    # 7. the weighting test's controls
    for anchor, tip in C.IW_TIPS.items():
        s = once(s, anchor, anchor + attrs(tip), f"iw tip {anchor}")

    # 8. comment 6: the worked example in Arguments
    s = once(s, C.WORKED_OLD, C.WORKED_NEW, "worked example")

    # 9. tooltip component (edit 1): a big target's tooltip follows the pointer, and a tap on another part of it moves
    old = ('    if (cur === a && how === "hover") tHide = setTimeout(hide, 120);\n  });\n')
    new = old + (
        '  document.addEventListener("pointermove", e => {   // a big target (a drawing): the tooltip follows the pointer\n'
        '    if (!cur || how !== "hover" || e.pointerType === "touch" || !cur.contains(e.target)) return;\n'
        '    const r = cur.getBoundingClientRect();\n'
        '    if (r.height > 140 || r.width > 460) { pt = { x: e.clientX, y: e.clientY }; place(); }\n'
        '  }, { passive: true });\n')
    s = once(s, old, new, "component: follow the pointer")
    old = '    if (cur === a && how === "tap") { hide(); return; }\n'
    new = ('    if (cur === a && how === "tap") {                // a second tap closes it; on a drawing, only near the first\n'
           '      const r = a.getBoundingClientRect(), big = r.height > 140 || r.width > 460;\n'
           '      if (!big || !pt || Math.hypot(e.clientX - pt.x, e.clientY - pt.y) < 24) { hide(); return; }\n'
           '    }\n')
    s = once(s, old, new, "component: taps on a drawing")

    # 10. the script for the drawings and chart points, after the last script
    js = open(os.path.join(HERE, "e6_tooltips.js"), encoding="utf-8").read()
    assert s.endswith("})();\n</script>\n")
    s = s + "\n" + js.replace("<script>", "<script>\n/* e6_tooltips */", 1)

    out = DRY if dry else PAGE
    open(out, "w", encoding="utf-8").write(s)
    assert md5(BACKUP) == backup_md5, "the backup changed"
    print(f"{'dry run' if dry else 'applied'}: {len(s):,} bytes -> {out}; md5 {md5(out)}; backup untouched ({backup_md5})")
    print(f"{len(LOG)} edits:", "; ".join(LOG[:3]), "...")
    for x in LOG:
        if x.startswith("inline") or x.startswith("legend"):
            print("  ", x)


if __name__ == "__main__":
    main()
