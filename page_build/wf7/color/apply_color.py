"""Page version 27 (first built as page version 20 on version 19, rebased on version 23 as page version 24, on
version 25 as page version 26, and on the live page after version 25, whose status box was removed, as page version
27), component "color": the remaining chart colors, a color key on every tooltip line that names a series, and small
text fixes (the status statements outside the removed status box, TD3+BC's target averaging).

usage: python -X utf8 apply_color.py IN.html OUT.html [--no-optional] [--only ID,ID,...] [--skip ID,ID,...]
                                     [--status-time HH:MM] [--status-date "D Mon YYYY"] [--version N] [--list]

IN is the page's core HTML (no publish wrapper): the live page after version 25 (service version 1792156350-92ba,
version 25 without its status box) or a newer working copy built on it. (The version-19 build of this patch is kept
in build/component_changes/v19_based/color/, the version-23 build in build/component_changes/v23_based/color/, the
version-25 build in build/component_changes/v25_based/color/.) Every edit is anchor-based: its anchor must occur exactly as often as stated (once, unless "count" says
otherwise), or nothing is written and the script names the edit, the count found and, when the replacement is already
there, says the edit looks applied already. The output never gains a CR byte or a duplicate id. IN is never modified.

Not touched, by design: the learning curves of finding G (#lc-svg1, #lc-svg2, the overview #lc-grid, their legends and
hover readouts), which version 19 redid; their own tokens (--lc-*) do not depend on any token changed here.

Sources of the design: wf4/understand/palette/palette_plan.json and tools/edits_palette.json (made on version 18). Edit
ids keep that plan's names where an edit is reused; T7, N9, K1, K2 and X1 (learning curves) are left out; K0 and K3
follow version 19's readout layout instead of the plan's; B1-B15 are the text fixes.
"""
import re
import sys

# The time the repository's records were read for the status statements (UTC, 15 Oct 2026). --status-time overrides.
# (Integration of page version 20: 15:53, after the approval of Stage 2 version 7 at 15:45, which B2, B5-B7 and B9
# state; it was 11:11 before that approval.) Page version 24: 22:13, the time the repository's remote-tracking refs
# were last fetched from GitHub (.git/FETCH_HEAD); read at 23:16 UTC, the local branches hold nothing newer. Page
# version 26: 16 Oct 2026, 12:39 UTC, the time the remote-tracking refs were last fetched (.git/FETCH_HEAD, 08:39:13
# -04:00, by another session; this build fetched nothing), read after 13:23 UTC; the local branch equals the shared
# branch (0377dd7). A first reading (refs fetched at 11:50 UTC) was overtaken by the near-80 approval at 11:50:43 UTC.
# Page version 27: 16 Oct 2026, 13:44 UTC, when GitHub's branch heads were read (git ls-remote, read only; they equal
# the fetch of 12:39 UTC), with the two facts of the B section's note that are in no ref (Stage 2 version 7 ended
# without running; the near-80 test runs on the owner's desktop computer).
# --status-date and --status-time set the status time ({D}, {T}). --version is accepted and checked; since page
# version 27 no edit names the page version ({V} was the status box's heading, B1, which is gone with the box).
STATUS_TIME = "13:44"
STATUS_DATE = "16 Oct 2026"
PAGE_VERSION = "27"

EDITS = []


def E(id, group, purpose, old, new, count=1, optional=False):
    EDITS.append(dict(id=id, group=group, purpose=purpose, old=old, new=new, count=count, optional=optional))


# ---------------------------------------------------------------- tokens (both themes; the two dark blocks agree)
E("T1", "tokens", "light --k-none #8a928e -> #7f8582: 3.19:1 on --surface and 2.91:1 on --bg -> 3.76:1 and 3.44:1",
  "  --k-none: #8a928e;           /* no throttle, dashed */",
  "  --k-none: #7f8582;           /* no throttle, dashed; in charts also \"nothing added\" (a diagnostic, a target line) */")
E("T2", "tokens", "dark (prefers-color-scheme block): --k-learn #5957fe -> #6797ff (3.42:1 -> 6.04:1); --k-ref "
  "#d5dbd7 -> #e7ebe8 (the ink); --k-none #8e9793 -> #a3aca8 (apart from the recorded-data green: protanopia 3.6 -> 8.2)",
  "\n    --k-learn: #5957fe; --k-learn-soft: #1e2640; --k-ref: #d5dbd7; --k-none: #8e9793;",
  "\n    --k-learn: #6797ff; --k-learn-soft: #1e2640; --k-ref: #e7ebe8; --k-none: #a3aca8;")
E("T3", "tokens", "dark ([data-theme=dark] block): the same three values as T2",
  "\n  --k-learn: #5957fe; --k-learn-soft: #1e2640; --k-ref: #d5dbd7; --k-none: #8e9793;",
  "\n  --k-learn: #6797ff; --k-learn-soft: #1e2640; --k-ref: #e7ebe8; --k-none: #a3aca8;")
E("T8", "tokens", "light --k-learn #115096 -> #1e4592: green and blue stay apart for tritanopia too (5.5 -> 8.2)",
  "  --k-learn: #115096;          /* the learned parts: critic, actor, BCA's band (the page's accent) */",
  "  --k-learn: #1e4592;          /* the learned parts: critic, actor, BCA's band and dose (the page's accent family) */")
E("T4", "tokens", "--dr-hi becomes the learned-parts blue (one blue for the critic, BCA and the emphasized arms)",
  ':root { --dr-hi: #2346b8; --dr-lo: #69726e; }',
  ':root { --dr-hi: var(--k-learn); --dr-lo: #69726e; }')
E("T5", "tokens", "dark --dr-hi follows --k-learn (media block)",
  '@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --dr-hi: #6d88e4; --dr-lo: #6c7570; } }',
  '@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --dr-hi: var(--k-learn); --dr-lo: #6c7570; } }')
E("T6", "tokens", "dark --dr-hi follows --k-learn (data-theme block)",
  ':root[data-theme="dark"] { --dr-hi: #6d88e4; --dr-lo: #6c7570; }',
  ':root[data-theme="dark"] { --dr-hi: var(--k-learn); --dr-lo: #6c7570; }')

# ---------------------------------------------------------------- the stall's marks: --k-stall (its text stays --warn)
E("S1", "stall", "Action Space chart: push line of the stalling world",
  "svg.chart .ln-stall { stroke: var(--warn); stroke-width: 2.6; }",
  "svg.chart .ln-stall { stroke: var(--k-stall); stroke-width: 2.6; }")
E("S2", "stall", "its points",
  "svg.chart .pt-stall { fill: var(--warn); stroke: var(--surface); stroke-width: 2; }",
  "svg.chart .pt-stall { fill: var(--k-stall); stroke: var(--surface); stroke-width: 2; }")
E("S3", "stall", "its legend swatch",
  '<span><i class="sw" style="background:var(--warn)"></i>push, stalling world</span>',
  '<span><i class="sw" style="background:var(--k-stall)"></i>push, stalling world</span>')
E("S4", "stall", "weighting test's push chart: stall zone",
  '    el("rect", { x: X(0.8), y: m.t, width: X(1) - X(0.8), height: H - m.t - m.b }, svg, "fill:var(--warn-soft)");',
  '    el("rect", { x: X(0.8), y: m.t, width: X(1) - X(0.8), height: H - m.t - m.b }, svg, "fill:var(--k-stall-soft)");')
E("S5", "stall", "weighting test's push line",
  'X, Y, "stroke:var(--warn)");',
  'X, Y, "stroke:var(--k-stall)");')
E("S6", "stall", "its legend swatch",
  '<span><i class="sw" style="background:var(--warn)"></i>Push, stalling world</span>',
  '<span><i class="sw" style="background:var(--k-stall)"></i>Push, stalling world</span>')
E("S7", "stall", "critic viewer: flag key",
  ".qv-lk-flag { width: 3px; height: 9px; background: var(--warn); border-radius: 1px; }",
  ".qv-lk-flag { width: 3px; height: 9px; background: var(--k-stall); border-radius: 1px; }")
E("S8", "stall", "critic viewer: stall key",
  ".qv-bx-stall { background: var(--warn-soft); box-shadow: inset 2px 0 0 var(--warn); }",
  ".qv-bx-stall { background: var(--k-stall-soft); box-shadow: inset 2px 0 0 var(--k-stall); }")
E("S9", "stall", "critic viewer: stall zone",
  "svg.chart .qv-stall { fill: var(--warn-soft); }",
  "svg.chart .qv-stall { fill: var(--k-stall-soft); }")
E("S10", "stall", "critic viewer: 80% line",
  "svg.chart .qv-stall-l { stroke: var(--warn); stroke-width: 1.5; stroke-dasharray: 5 4; }",
  "svg.chart .qv-stall-l { stroke: var(--k-stall); stroke-width: 1.5; stroke-dasharray: 5 4; }")
E("S11", "stall", "critic viewer: flag ticks",
  "svg.chart .qv-flag { stroke: var(--warn); stroke-width: 2; }",
  "svg.chart .qv-flag { stroke: var(--k-stall); stroke-width: 2; }")
E("S12", "stall", "live panel drawing: a stalled thruster and its arrow (stallC = --k-stall, already defined there)",
  "    const col = stalled ? warn : ink2, len = f[k] * 34;",
  "    const col = stalled ? stallC : ink2, len = f[k] * 34;")
E("S13", "stall", "live panel drawing: a stalled gauge bar (its numbers and STALLED stay --warn: they are text)",
  "    g.fillStyle = stalled ? warn : muted;",
  "    g.fillStyle = stalled ? stallC : muted;")
E("S14", "stall", "task diagram: stall tile and limit line (a drawing)",
  "svg.dg .dg-stall { fill: var(--warn-soft); stroke: var(--warn); stroke-width: 1; }\n"
  "svg.dg .dg-limit { stroke: var(--warn); stroke-dasharray: 5 4; stroke-width: 1.6; }",
  "svg.dg .dg-stall { fill: var(--k-stall-soft); stroke: var(--k-stall); stroke-width: 1; }\n"
  "svg.dg .dg-limit { stroke: var(--k-stall); stroke-dasharray: 5 4; stroke-width: 1.6; }", optional=True)
E("S15", "stall", "task diagram: stalled fill",
  "svg.dg .dg-fill.w { fill: var(--warn); }",
  "svg.dg .dg-fill.w { fill: var(--k-stall); }", optional=True)
E("S16", "stall", "task diagram: stalled thruster and arrow",
  "svg.dg .dg-thr.w { fill: var(--warn); }\nsvg.dg .dg-thrarrow.w { stroke: var(--warn); fill: var(--warn); }",
  "svg.dg .dg-thr.w { fill: var(--k-stall); }\nsvg.dg .dg-thrarrow.w { stroke: var(--k-stall); fill: var(--k-stall); }",
  optional=True)
E("S17", "stall", "critic viewer strip: stalled thruster",
  "svg.qv-strip .qv-thr.w { fill: var(--warn); }",
  "svg.qv-strip .qv-thr.w { fill: var(--k-stall); }", optional=True)
E("S18", "stall", "critic viewer strip: stalled gauge and 80% mark",
  "svg.qv-strip .qv-gfill.w { fill: var(--warn); }\nsvg.qv-strip .qv-g80 { stroke: var(--warn); stroke-width: 1.5; }",
  "svg.qv-strip .qv-gfill.w { fill: var(--k-stall); }\nsvg.qv-strip .qv-g80 { stroke: var(--k-stall); stroke-width: 1.5; }",
  optional=True)

# ---------------------------------------------------------------- the learned parts: --k-learn
E("L1", "learned", "Methods chart: BCA's dose line, accent -> learned-parts blue",
  "svg.chart .ln-dose { stroke: var(--accent); stroke-width: 2.6; }",
  "svg.chart .ln-dose { stroke: var(--k-learn); stroke-width: 2.6; }")
E("L2", "learned", "its points (.pt-nostall is re-pointed to --k-rise later in the page, unchanged)",
  "svg.chart .pt-nostall, svg.chart .pt-dose { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }",
  "svg.chart .pt-nostall, svg.chart .pt-dose { fill: var(--k-learn); stroke: var(--surface); stroke-width: 2; }")
E("L3", "learned", "its legend swatch",
  "<span><i class=\"sw\" style=\"background:var(--accent)\"></i>harm-placed dose, BCA's form, [1, 1.5)</span>",
  "<span><i class=\"sw\" style=\"background:var(--k-learn)\"></i>harm-placed dose, BCA's form, [1, 1.5)</span>")
E("L4", "learned", "critic viewer: the critic's legend key",
  ".qv-sw-q { background: var(--accent); }",
  ".qv-sw-q { background: var(--k-learn); }")
E("L5", "learned", "critic viewer: the a-hat marker key",
  ".qv-mk-hat { border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 2px var(--surface), 0 0 0 3px var(--accent); }",
  ".qv-mk-hat { border-radius: 50%; background: var(--k-learn); box-shadow: 0 0 0 2px var(--surface), 0 0 0 3px var(--k-learn); }")
E("L6", "learned", "critic viewer: the critic's line",
  "svg.chart .qv-lq { stroke: var(--accent); stroke-width: 2; fill: none; }",
  "svg.chart .qv-lq { stroke: var(--k-learn); stroke-width: 2; fill: none; }")
E("L7", "learned", "critic viewer: a-hat marker",
  "svg.chart .qv-hat { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }",
  "svg.chart .qv-hat { fill: var(--k-learn); stroke: var(--surface); stroke-width: 2; }")
E("L8", "learned", "critic viewer: hover dot",
  "svg.chart .qv-dq { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }",
  "svg.chart .qv-dq { fill: var(--k-learn); stroke: var(--surface); stroke-width: 2; }")
E("L9", "learned", "critic viewer: tooltip key of the critic (state chart)",
  'tipRow(tip, "--accent", fv(q), "critic, Q(<i>s</i>, <i>a</i>)");',
  'tipRow(tip, "--k-learn", fv(q), "critic, Q(<i>s</i>, <i>a</i>)");')
E("L10", "learned", "critic viewer: tooltip key of the critic (episode chart)",
  'tipRow(tip, "--accent", fv(E.qa[t]), "critic, Q(<i>s</i>, π(<i>s</i>))");',
  'tipRow(tip, "--k-learn", fv(E.qa[t]), "critic, Q(<i>s</i>, π(<i>s</i>))");')

# ---------------------------------------------------------------- neutral series: --ink-2 or --k-none, no dimming
E("N1", "neutral", "Action Space chart: throttle cost line, text gray -> --ink-2 as in the stall chart "
  "(dark: stall line vs it normal 13.0, deuteranopia 5.7 -> stall vs no-stall 16.9 / 16.2)",
  "svg.chart .ln-cost { stroke: var(--muted); stroke-width: 1.8; stroke-dasharray: 2 4; }",
  "svg.chart .ln-cost { stroke: var(--ink-2); stroke-width: 1.8; stroke-dasharray: 2 4; }")
E("N2", "neutral", "its point",
  "svg.chart .pt-cost { fill: var(--muted); stroke: var(--surface); stroke-width: 2; }",
  "svg.chart .pt-cost { fill: var(--ink-2); stroke: var(--surface); stroke-width: 2; }")
E("N3", "neutral", "its legend swatch",
  "repeating-linear-gradient(90deg,var(--muted) 0 2px,transparent 2px 5px)\"></i>throttle cost 0.3 a²",
  "repeating-linear-gradient(90deg,var(--ink-2) 0 2px,transparent 2px 5px)\"></i>throttle cost 0.3 a²")
E("N4", "neutral", "Methods chart: wider-range diagnostic, text gray -> 'nothing added' gray (dark normal 14.1 -> 17.0)",
  "svg.chart .ln-cap2 { stroke: var(--muted); stroke-width: 2; stroke-dasharray: 6 4; }",
  "svg.chart .ln-cap2 { stroke: var(--k-none); stroke-width: 2; stroke-dasharray: 6 4; }")
E("N5", "neutral", "its legend swatch",
  "repeating-linear-gradient(90deg,var(--muted) 0 6px,transparent 6px 10px)\"></i>wider-range diagnostic",
  "repeating-linear-gradient(90deg,var(--k-none) 0 6px,transparent 6px 10px)\"></i>wider-range diagnostic")
E("N6", "neutral", "coverage chart: the 90% target drawn in the color of its legend swatch (.sw.dash = --c-none); "
  "light green vs the drawn gray: normal 9.6, deuteranopia 3.9 -> 15.7, 12.7",
  'line(svg, [[-2, 90], [3, 90]], X, Y, "stroke:var(--muted);stroke-width:1.2;stroke-dasharray:4 4");',
  'line(svg, [[-2, 90], [3, 90]], X, Y, "stroke:var(--k-none);stroke-width:1.2;stroke-dasharray:4 4");')
E("N7", "neutral", "cost chart: controllers not picked drawn opaque and thin (75% opacity put 'no throttle' at 2.28:1)",
  "g.lineWidth = on ? 2.8 : 1.5; g.globalAlpha = on ? 1 : 0.75;",
  "g.lineWidth = on ? 2.8 : 1.5; g.globalAlpha = 1;")
E("N8", "neutral", "weighting bars: every bar in full color (35% opacity put them at 1.7-2.6:1); the chosen test outlined in ink",
  "      const cx = X(i + 0.5), y = Y(GAP[k]), o = !sel || sel === k ? 1 : 0.35;\n"
  "      const r = el(\"path\", { d: `M${cx - bw / 2},${Y(0)} V${y + 4} q0,-4 4,-4 H${cx + bw / 2 - 4} q4,0 4,4 V${Y(0)} Z` }, svg,\n"
  "        `fill:var(${SER[k]});stroke:none;opacity:${o}`);",
  "      const cx = X(i + 0.5), y = Y(GAP[k]), on = !!sel && sel === k;\n"
  "      const r = el(\"path\", { d: `M${cx - bw / 2},${Y(0)} V${y + 4} q0,-4 4,-4 H${cx + bw / 2 - 4} q4,0 4,4 V${Y(0)} Z` }, svg,\n"
  "        `fill:var(${SER[k]});stroke:${on ? \"var(--ink)\" : \"none\"};stroke-width:2`);")

# ---------------------------------------------------------------- tooltips: a key on every line that names a series
E("K0", "tooltip-css", "chart readouts wrap inside their chart (no nowrap); keyed readouts take finding G's readout "
  "layout (key, value, name); the shared tooltip's key rows",
  "  box-shadow: 0 4px 14px rgba(0,0,0,0.14); white-space: nowrap; z-index: 3; }\n",
  "  box-shadow: 0 4px 14px rgba(0,0,0,0.14); white-space: normal; max-width: min(360px, calc(100% - 4px)); z-index: 3; }\n"
  "/* a chart readout that names series (the weighting test's line charts) takes the layout of finding G's learning-curve\n"
  "   readout (#iw-curves .iw-tip): per series its own line key in its color and dash, its value in ink, then its name,\n"
  "   which wraps inside the chart on a phone */\n"
  ".iw-tip.tip-keyed { display: grid; grid-template-columns: auto auto minmax(0, 1fr); column-gap: 8px; align-items: center; }\n"
  ".iw-tip[hidden] { display: none; }\n"
  ".iw-tip.tip-keyed .tip-h { grid-column: 1 / -1; font-weight: 600; }\n"
  ".iw-tip.tip-keyed svg { display: block; }\n"
  ".iw-tip.tip-keyed b { font-weight: 700; color: var(--ink); text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }\n"
  ".iw-tip.tip-keyed span { font-family: var(--body); font-size: 13px; color: var(--ink-2); line-height: 1.3; }\n"
  "/* the shared tooltip: a series' key before each line of its text (data-tip-keys) or before its title (data-tip-key) */\n"
  ".tt .tt-rows { display: grid; grid-template-columns: 1.3em minmax(0, 1fr); gap: 3px 6px; align-items: baseline; }\n"
  ".tt .tt-rows > .key { margin: 0; align-self: center; }\n"
  ".tt .tt-h > .key { margin-right: 0.45em; }\n")
E("K3", "tooltip", "weighting test: helpers that build a keyed readout (fixed text of this script, so innerHTML stays safe)",
  '  const sgn = (v, d) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);\n',
  '  const sgn = (v, d) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);\n'
  "  /* a readout that names series, laid out like finding G's learning-curve readout: a header, then per series its own\n"
  "     line key (its color and dash), its value and its name. All text is this script's own, so innerHTML is safe. */\n"
  '  const tipKey = (c, dash) => `<svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line x1="1" x2="21" y1="5" y2="5" '
  'style="stroke:var(${c});stroke-width:2.4;stroke-linecap:round${dash ? ";stroke-dasharray:" + dash : ""}"/></svg>`;\n'
  '  const tipRows = (head, rows) => `<div class="tip-h">${head}</div>` +\n'
  '    rows.map(([c, dash, v, n]) => `${tipKey(c, dash)}<b>${v}</b><span>${n}</span>`).join("");\n')
E("K3b", "tooltip", "weighting test: its readout boxes take the keyed layout",
  'function hover(svg, box, X, xd, W, H, m, build) {\n'
  '    let tip = box.querySelector(".iw-tip");\n'
  '    if (!tip) { tip = document.createElement("div"); tip.className = "iw-tip"; tip.hidden = true; box.appendChild(tip); }',
  'function hover(svg, box, X, xd, W, H, m, build) {\n'
  '    let tip = box.querySelector(".iw-tip");\n'
  '    if (!tip) { tip = document.createElement("div"); tip.className = "iw-tip tip-keyed"; tip.hidden = true; box.appendChild(tip); }')
E("K4", "tooltip", "weighting test: its readouts stay inside their chart box",
  '      tip.style.left = Math.max(0, left) + "px"; tip.style.top = (ev.clientY - br.top - 10) + "px";',
  '      tip.style.left = Math.max(0, Math.min(left, br.width - tip.offsetWidth)) + "px"; tip.style.top = (ev.clientY - br.top - 10) + "px";')
E("K5", "tooltip", "density chart (#iw-svg-dens): keyed rows",
  'c => `Current c: ${c.toFixed(2)}<br>Training density: ${dens(LR, c).toFixed(3)}<br>Test density: ${dens(lam, c).toFixed(3)}`);',
  'c => tipRows(`Current c: ${c.toFixed(2)}`, [["--iw-train", "", dens(LR, c).toFixed(3), "training density"], [serOf(lam), "", dens(lam, c).toFixed(3), "test density"]]));')
E("K6", "tooltip", "weight chart (#iw-svg-w): keyed row (its color follows the chosen test)",
  'c => `Current c: ${c.toFixed(2)}<br>Weight ω(c): ${wgt(lam, c).toFixed(3)}`);',
  'c => tipRows(`Current c: ${c.toFixed(2)}`, [[serOf(lam), "", wgt(lam, c).toFixed(3), "weight ω(c)"]]));')
E("K7", "tooltip", "coverage chart (#iw-svg-cov): keyed rows",
  'l => `Test lean ξ: ${sgn(l, 2)}<br>Coverage without weights: ${(100 * F(l, tTrain)).toFixed(1)}%<br>Coverage with weights: 90.0%`);',
  'l => tipRows(`Test lean ξ: ${sgn(l, 2)}`, [["--iw-train", "", (100 * F(l, tTrain)).toFixed(1) + "%", "coverage without weights"], ["--iw-weighted", "", "90.0%", "coverage with weights"]]));')
E("K8", "tooltip", "weighting bars (#iw-svg-bars): the shared tooltip shows the bar's color before its title",
  '      r.setAttribute("data-tip-title", NAME[k]);',
  '      r.setAttribute("data-tip-title", NAME[k]); r.setAttribute("data-tip-key", SER[k] + " tile");')
E("K9", "tooltip", "shared tooltip (#tt): a key before the title (data-tip-key=\"--token shape\") or before each line of the "
  "text (data-tip-keys = JSON list of [token, shape] or null, one per line); shapes are the page's .key marks",
  '    const ref = a.dataset.tipRef && document.getElementById(a.dataset.tipRef);\n'
  '    if (a.dataset.tipTitle) out.push(el("div", "tt-h", a.dataset.tipTitle));\n'
  '    if (ref) {',
  '    const ref = a.dataset.tipRef && document.getElementById(a.dataset.tipRef);\n'
  "    /* a series' key: before the title (data-tip-key=\"--token shape\") or before each line of the text (data-tip-keys,\n"
  "       a JSON list of [token, shape] or null, one per line). Shapes are the page's .key marks; the text read out to\n"
  "       assistive technology (tipText) is unchanged, and the marks are hidden from it. */\n"
  '    const SHAPES = ["line", "dash", "dots", "dot", "ring", "tile"];\n'
  '    const keyEl = (c, s) => { const m = el("i", "key " + (SHAPES.includes(s) ? s : "line")); m.setAttribute("aria-hidden", "true");\n'
  '      if (/^--[a-z0-9-]+$/.test(c || "")) m.style.setProperty("--c", "var(" + c + ")"); return m; };\n'
  '    if (a.dataset.tipTitle) { const h = el("div", "tt-h", a.dataset.tipTitle);\n'
  '      if (a.dataset.tipKey) { const [c, s] = a.dataset.tipKey.split(" "); h.prepend(keyEl(c, s)); } out.push(h); }\n'
  '    if (a.dataset.tipKeys && a.dataset.tip && !ref) {\n'
  '      let keys = []; try { keys = JSON.parse(a.dataset.tipKeys); } catch (e) { keys = []; }\n'
  '      const box = el("div", "tt-b tt-rows");\n'
  '      a.dataset.tip.split("\\n").forEach((line, i) => { const k = Array.isArray(keys) ? keys[i] : null;\n'
  '        box.append(Array.isArray(k) ? keyEl(k[0], k[1]) : el("span"), el("span", null, line)); });\n'
  '      out.push(box); return out;\n'
  '    }\n'
  '    if (ref) {')
E("K10", "tooltip", "shared tooltip: redraw an open tooltip when its keys change",
  '    attributeFilter: ["data-tip", "data-tip-title", "data-tip-ref"] });',
  '    attributeFilter: ["data-tip", "data-tip-title", "data-tip-ref", "data-tip-keys", "data-tip-key"] });')
E("K11", "tooltip", "task chart columns: remember each point's series",
  '      cols.get(k).items.push({ txt: t.textContent.replace(/\\s+/g, " ").trim(), rank: SERIES.indexOf(c.getAttribute("class")) });',
  '      cols.get(k).items.push({ txt: t.textContent.replace(/\\s+/g, " ").trim(), rank: SERIES.indexOf(c.getAttribute("class")), cls: c.getAttribute("class") });')
E("K12", "tooltip", "task chart columns: one key per line (stall-aware throttle: ink line; rising-thrust throttle: "
  "violet line; the push it really gives: violet dash)",
  '      r.setAttribute("data-tip", items.map(it => cap(it.txt.slice(head))).join("\\n"));',
  '      r.setAttribute("data-tip", items.map(it => cap(it.txt.slice(head))).join("\\n"));\n'
  '      const KEY = { "pt-sa": ["--c-sa", "line"], "pt-rt": ["--c-rt", "line"], "pt-rtpush": ["--c-rt", "dash"] };\n'
  '      r.setAttribute("data-tip-keys", JSON.stringify(items.map(it => KEY[it.cls] || null)));')
E("K13", "tooltip", "cost chart (canvas#cost): one keyed line per controller instead of one sentence",
  '    const parts = CTRL.map(c => {\n'
  '      const nm = NAME.ctrl[c], v = num(-D.roll[key(S.world, S.sit, c)].g[t], 2);\n'
  '      const who = /^no throttle$/i.test(nm) ? "with no throttle" : "for the " + nm.charAt(0).toLowerCase() + nm.slice(1);\n'
  '      return v + " " + who + (c === S.ctrl ? " (the controller picked in the panel)" : "");\n'
  '    });\n'
  '    return ["Cost so far at state " + t, "At this state the cost so far is " + parts.slice(0, -1).join(", ") + " and " +\n'
  '      parts[parts.length - 1] + ". The cost so far is minus the return so far, so lower is better."];',
  '    const lines = CTRL.map(c => NAME.ctrl[c] + ": " + num(-D.roll[key(S.world, S.sit, c)].g[t], 2) + (c === S.ctrl ? " (picked in the panel)" : ""));\n'
  '    const keys = CTRL.map(c => [CVAR[c], c === "no_throttle" ? "dash" : "line"]);\n'
  '    lines.push("The cost so far is minus the return so far, so lower is better."); keys.push(null);\n'
  '    return ["Cost so far at state " + t, lines.join("\\n"), JSON.stringify(keys)];')
E("K14", "tooltip", "canvas tooltips: carry the keys (a third element) or clear them, before the text changes (version 21's "
  "set(), which falls back to the learned actor's own general text, kept)",
  '    const set = tp => {\n'
  '      if (!tp || tp === base) tp = (typeof simBaseTip === "function" && simBaseTip(cv.id)) || base;\n'
  '      if (cv.getAttribute("data-tip-title") !== tp[0]) cv.setAttribute("data-tip-title", tp[0]);\n'
  '      if (cv.getAttribute("data-tip") !== tp[1]) cv.setAttribute("data-tip", tp[1]);\n'
  '    };',
  '    const set = tp => {\n'
  '      if (!tp || tp === base) tp = (typeof simBaseTip === "function" && simBaseTip(cv.id)) || base;\n'
  '      if (cv.getAttribute("data-tip-title") !== tp[0]) cv.setAttribute("data-tip-title", tp[0]);\n'
  '      if (tp[2]) { if (cv.getAttribute("data-tip-keys") !== tp[2]) cv.setAttribute("data-tip-keys", tp[2]); } else cv.removeAttribute("data-tip-keys");\n'
  '      if (cv.getAttribute("data-tip") !== tp[1]) cv.setAttribute("data-tip", tp[1]);\n'
  '    };')
E("K13b", "tooltip", "cost chart while the learned actor drives (versions 21 and 25): one keyed line per data quality, in "
  "its line's color (--lq-1 .. --lq-5), instead of one sentence; version 25's learner name and cloud rerun in the last line",
  '      const nm = typeof LRNAME !== "undefined" && S.lr && S.lr !== "td3bc" ? LRNAME[S.lr] + " " : "";\n'
  '      const rr = !!learnedEp().meta.learner;\n'
  '      const lp = LQ.map(q => num(-learnedEp(q).g[t], 2) + " for the " + nm + "actor learned from " + q + " data" + (q === S.lq ? " (the one on screen)" : ""));\n'
  '      return ["Cost so far at state " + t, "At this state the cost so far is " + lp.slice(0, -1).join(", ") + " and " +\n'
  '        lp[lp.length - 1] + ". All five learned from " + (S.ld === "gap" ? "data with the 80% hold removed" : "all the data") +\n'
  '        (rr ? ", in the cloud rerun of seed 0," : "") +\n'
  '        " and start from the same start of the " + (S.lm === "strong" ? "strong test mix" : "training mix") +\n'
  '        ". The cost so far is minus the return so far, so lower is better."];',
  '      const nm = typeof LRNAME !== "undefined" && S.lr && S.lr !== "td3bc" ? LRNAME[S.lr] + " " : "";\n'
  '      const rr = !!learnedEp().meta.learner;\n'
  '      const QN = { random: "Random", "medium-replay": "Medium-replay", medium: "Medium", "medium-expert": "Medium-expert", expert: "Expert" };\n'
  '      const lines = LQ.map(q => (QN[q] || q) + " data: " + num(-learnedEp(q).g[t], 2) + (q === S.lq ? " (the one on screen)" : ""));\n'
  '      const keys = LQ.map(q => [typeof LQVAR === "object" && LQVAR[q] ? LQVAR[q] : "--muted", "line"]);\n'
  '      lines.push("All five " + (nm ? nm + "actors " : "") + "learned from " + (S.ld === "gap" ? "data with the 80% hold removed" : "all the data") +\n'
  '        (rr ? ", in the cloud rerun of seed 0," : "") +\n'
  '        " and start from the same start of the " + (S.lm === "strong" ? "strong test mix" : "training mix") +\n'
  '        ". The cost so far is minus the return so far, so lower is better."); keys.push(null);\n'
  '      return ["Cost so far at state " + t, lines.join("\\n"), JSON.stringify(keys)];')
E("N7b", "neutral", "cost chart while the learned actor drives (version 21): the actors not on screen drawn opaque and thin, "
  "as N7 does for the fixed controllers (at 80% opacity the random-data line fell to 2.45:1 light, 2.78:1 dark)",
  "    g.strokeStyle = tok(LQVAR[s.c]); g.lineWidth = on ? 2.8 : 1.5; g.globalAlpha = on ? 1 : 0.8;",
  "    g.strokeStyle = tok(LQVAR[s.c]); g.lineWidth = on ? 2.8 : 1.5; g.globalAlpha = 1;")
for _k, (_q, _n) in enumerate([("Random data", "random"), ("Medium-replay data", "medium-replay"), ("Medium data", "medium"),
                               ("Medium-expert data", "medium-expert"), ("Expert data", "expert")]):
    E("K22" + "abcde"[_k], "tooltip", "learned actors' cost legend (version 21): the line's key before the title, " + _n,
      '<span data-tip-title="%s" data-tip="This line is the cost so far of the actor that TD3+BC learned from %s data' % (_q, _n),
      '<span data-tip-key="--lq-%d line" data-tip-title="%s" data-tip="This line is the cost so far of the actor that TD3+BC '
      'learned from %s data' % (_k + 1, _q, _n))
E("K15", "tooltip", "dot-and-range chart (#beyond-chart): the row's tooltip in the page's tooltip look, led by the row's own mark",
  ".dr-tip { display: none; position: absolute; bottom: calc(100% + 1px); background: var(--ink); color: var(--bg);\n"
  "  font-size: 12.5px; line-height: 1.35; padding: 3px 8px; border-radius: 5px; white-space: nowrap; z-index: 3;\n"
  "  pointer-events: none; }\n"
  ".dr-row:hover .dr-tip, .dr-row:focus .dr-tip { display: block; }",
  "/* the dot-and-range tooltip's shadow: a token, defined for the light and both dark themes */\n"
  ":root { --tip-shadow: rgba(0, 0, 0, .14); }\n"
  "@media (prefers-color-scheme: dark) { :root:not([data-theme=\"light\"]) { --tip-shadow: rgba(0, 0, 0, .45); } }\n"
  ":root[data-theme=\"dark\"] { --tip-shadow: rgba(0, 0, 0, .45); }\n"
  ".dr-tip { display: none; position: absolute; bottom: calc(100% + 1px); background: var(--surface); color: var(--ink);\n"
  "  border: 1px solid var(--rule-strong); box-shadow: 0 4px 14px var(--tip-shadow);\n"
  "  font-size: 12.5px; line-height: 1.35; padding: 3px 8px; border-radius: 5px; white-space: nowrap; z-index: 3;\n"
  "  pointer-events: none; }\n"
  ".dr-tip::before { content: \"\"; display: inline-block; width: 9px; height: 9px; margin-right: 6px; border-radius: 50%;\n"
  "  vertical-align: 0; background: currentColor; }\n"
  ".dr-row.hi .dr-tip::before { background: var(--dr-hi); } .dr-row.lo .dr-tip::before { background: var(--dr-lo); }\n"
  ".dr-row.ref .dr-tip::before { background: var(--surface); box-shadow: inset 0 0 0 2px var(--ink-2); }\n"
  ".dr-row:hover .dr-tip, .dr-row:focus .dr-tip { display: block; }")
E("K16", "tooltip", "live panel's cost legend: the series key before the item's tooltip title",
  '<span data-tip-title="Recording controller" data-tip="',
  '<span data-tip-key="--c-rec line" data-tip-title="Recording controller" data-tip="', optional=True)
E("K17", "tooltip", "live panel's cost legend: key before the title",
  '<span data-tip-title="Stall-aware controller" data-tip="',
  '<span data-tip-key="--c-sa line" data-tip-title="Stall-aware controller" data-tip="', optional=True)
E("K18", "tooltip", "live panel's cost legend: key before the title",
  '<span data-tip-title="Rising-thrust controller" data-tip="',
  '<span data-tip-key="--c-rt line" data-tip-title="Rising-thrust controller" data-tip="', optional=True)
E("K19", "tooltip", "live panel's cost legend: key before the title",
  '<span data-tip-title="No throttle" data-tip="',
  '<span data-tip-key="--c-none dash" data-tip-title="No throttle" data-tip="', optional=True)
E("K20", "tooltip", "critic viewer legends (both charts and their copies): the critic's key before the title",
  "<span data-tip-title=\"The critic's value\" data-tip=\"",
  "<span data-tip-key=\"--k-learn line\" data-tip-title=\"The critic's value\" data-tip=\"", count=4, optional=True)
E("K21", "tooltip", "critic viewer legends: the true value's key before the title",
  '<span data-tip-title="The true value" data-tip="',
  '<span data-tip-key="--ink line" data-tip-title="The true value" data-tip="', count=4, optional=True)
E("X2", "text", "dot-and-range legend: the ring also marks the two starting rows",
  '<span><i class="mk ring"></i>no signal</span>',
  '<span><i class="mk ring"></i>no signal, or a starting point</span>', optional=True)

# ---------------------------------------------------------------- B: small text fixes (facts read from the repository)
# Stage 2 version 7's handoff: commit 3a8ce5b, 15 Oct 2026 07:47:31 UTC (README step 9). Its approval (README step 10):
# stage2_v7/approval_v7_3dfcb6140e2b8e2c.json, "approved_at_utc": "2026-10-15T15:45:42Z", typed at the owner's own
# terminal (approve_local_v7.py), committed on the branch forked in de4851c ("Stage 2: approve v7 3dfcb6140e2b8e2c",
# 15 Oct 15:49:21 UTC). Its own record says it takes effect once the cloud session has imported it (README step 11,
# "approve_v7.py import"); no import record and no run on seeds 5-9 yet (handoff/HANDOFF.json: every seed 5-9 phase
# "not started"). (Integration of page version 20: B2, B5, B6, B7 and B9 were reworded for the approval.)
# Page version 26: the approval was imported (README step 11) on 16 Oct 2026 at 02:27:10 UTC: import_v7_3dfcb6140e2b8e2c.json,
# commit 02e591b on the shared branch (origin/claude/main-bca-connection-x4o46s, "Version 7 can now run on the Framework
# Desktop (step 12)"). No result of the run on seeds 5-9 in any ref as fetched at 12:39 UTC (no result files; the run
# pushes its files with laptop/push_runfiles_v7.sh); B2, B5, B7, B9 say "the run comes next" or "no result yet".
# Page version 27: the header's status box (div#head-status) is gone. The project owner asked for it to be removed
# ("I dont want this section included", page comment of 16 Oct 2026, 12:06 UTC) and the live page removed it (the other
# session's version 26, service version 1792156350-92ba). Its edits B1 (heading), B2 (Stage 2), B4 (four learners and
# the near-80 test), B20 (the weighting test's place in the repository) and B23 (version 25's rerun) are dropped, and
# nothing of the box is added or restored. The status statements outside it stay (B5-B11, B13, B16-B19, B21, B22, B24,
# B25; B14 and B15 were in the old Toy worlds section, which the toy patch now replaces: see below).
# Page version 27's status time, 16 Oct 2026, 13:44 UTC: GitHub's branch heads read then (git ls-remote, read only)
# equal the fetch of 12:39 UTC (the shared branch 0377dd7, forked cab5b3e, main 1ec267e; no near-80 run-files branch).
# Two facts are in no ref and are the ones the workflow's task of 16 Oct (about 13:38 UTC) states as current: Stage 2
# version 7 was ended without running, so seeds 5-9 will not run under it (the near-80 runner starts only once version
# 7 has finished or its ending is recorded on the desktop, "run_near80.sh version7-ended": near80/PREREGISTRATION.md,
# choice 12), and the near-80 test (plan "version 8, near-80"; approved at 11:50:43 UTC, committed at 11:51, f79b89c;
# the approval recorded at 11:52:58 UTC, 287380b) now runs on the owner's Framework Desktop computer. B5, B7, B9, B21,
# B22, B24 and B25 say so.
# The toy notebooks: bca_discrete_toy.ipynb committed in 2c8a0e3 (15 Oct 07:11 UTC; SHA-256 bd6b8c927de23faa..., the
# version the page checked) and bca_channel_toy.ipynb in 6320224 (15 Oct 09:28 UTC; SHA-256 cf1ba0c5f1e09a11..., not
# the checked c8a2c8c312a438cc...), both on the branch forked (pushed to origin/forked), neither on the shared branch.
# Fix round 1: origin/main 1ec267e (pull request 5, 21:36 UTC, after the status time) merged the branch forked, so both
# notebooks (the same versions, checked by SHA-256) are on the main branch too; B3 and B13 say so, dated, as B16 does.
# B3 (the toy notebooks in the status box) is not used from page version 24 on: version 23's shorter box no longer
# speaks of the notebooks and its toy-worlds line is still true; B13 keeps After the stop's sentence on the notebooks
# up to date. B14 and B15 (two sentences of the old Toy worlds section: the second notebook's kill rules, the re-check)
# are dropped in page version 27's toy integration: the build's first stage, the toy patch (wf7/toy/apply_toy.py),
# replaces that section as a whole, and its new text says itself that the notebook "was first committed on 15 Oct, after
# its runs" (build/rebase/edits/e_color_v27t.py).
E("B13", "text", "After the stop: the toy notebooks are now committed (branch forked, merged into main at 21:36 UTC)",
  "(exploratory notebooks, not yet committed\n      to the repository)",
  "(exploratory notebooks, committed on 15 Oct\n      to the repository's branch <code>forked</code> and merged into its main branch at 21:36 UTC)")
E("B5", "text", "Results, opening: version 7 was handed off and approved, and the approval checked and recorded; then "
  "it was ended without running (page version 27)",
  "As of 15 Oct 2026, 05:41 UTC, four\n"
  "      lines of work continue on this environment: Stage 2's confirmatory seeds 5–9 (its pre-registration is approved; "
  "version 7, which changes only where and how\n"
  "      the run happens, still awaits its handoff and its own approval);",
  "As of {D}, {T} UTC, this is where four\n"
  "      lines of work on this environment stand: Stage 2's confirmatory seeds 5–9 (its pre-registration is approved; "
  "version 7, which changes only where and how\n"
  "      the run happens, was handed off on 15 Oct at 07:47 UTC and approved by the project owner at 15:45 UTC, and the\n"
  "      approval was checked and recorded on 16 Oct at 02:27 UTC; later that day version 7 was ended without running,\n"
  "      so those seeds will not run under it);")
E("B6", "text", "After the stop, status box: its date and the commits it was read at",
  "<h4>Where each line of work stands, as of 15 Oct 2026, 05:41 UTC</h4>\n"
  "      <p class=\"note\">Read from the repository's shared branch <code>claude/main-bca-connection-x4o46s</code> at\n"
  "        commit <code>e91452d</code> (15 Oct, 03:57 UTC)",
  "<h4>Where each line of work stands, as of {D}, {T} UTC</h4>\n"
  "      <p class=\"note\">Read from the repository as last fetched from GitHub: its shared branch\n"
  "        <code>claude/main-bca-connection-x4o46s</code> at commit <code>0377dd7</code> (16 Oct, 12:38 UTC), its branch\n"
  "        <code>forked</code> at commit <code>cab5b3e</code> (15 Oct, 21:45 UTC; the approval of version 7 is commit\n"
  "        <code>de4851c</code>, 15:49 UTC), its main branch at commit <code>1ec267e</code> (15 Oct, 21:36 UTC)")
E("B7", "text", "After the stop, status box, Stage 2 seeds 5-9: the handoff, the approval and its check and record were "
  "done; then version 7 was ended without running (page version 27)",
  "(the only seeds that can carry Stage 2's claim): not run. They run under\n"
  "          version 7, which changes only where and how the rest of Stage 2 runs: now on the project owner's Framework\n"
  "          Desktop computer. Its full self-test reported ready on 15 Oct at 00:42 (<code>e8fe0fb</code>). Still to come:\n"
  "          the handoff, the project owner's approval typed at their own terminal, then the run.</li>",
  "(the only seeds that can carry Stage 2's claim): not run, and they will\n"
  "          not run under version 7, which changed only where and how the rest of Stage 2 runs: on the project owner's\n"
  "          Framework Desktop computer. Its full self-test reported ready on 15 Oct at 00:42 (<code>e8fe0fb</code>), and the\n"
  "          handoff followed at 07:47 (<code>3a8ce5b</code>). The project owner typed their approval at their own terminal\n"
  "          at 15:45 (committed at 15:49 on the branch <code>forked</code>, <code>de4851c</code>), and on 16 Oct at 02:27 it\n"
  "          was checked against the repository's history and recorded (<code>02e591b</code>). Later on 16 Oct version 7\n"
  "          was ended without running, and that computer now runs the near-80 test (below); no record of the ending is\n"
  "          in the repository.</li>")
E("B8", "text", "first claim's status: its date",
  "<p><b>Status, as of 15 Oct 2026, 05:41 UTC: not shown on the exploratory seeds; the deciding test has not\n          run.</b>",
  "<p><b>Status, as of {D}, {T} UTC: not shown on the exploratory seeds; the deciding test has not\n          run.</b>")
E("B9", "text", "first claim's status: the handoff, the approval and its check and record were done; then version 7 "
  "was ended without running, so the deciding test will not run under it (page version 27)",
  "with every setting frozen, on fresh <span class=\"nw\">seeds 5–9</span>, runs under version 7 on the project\n"
  "          owner's Framework Desktop computer: its self-test was ready on 15 Oct at 00:42 UTC, and the handoff, the\n"
  "          project owner's approval at their own terminal and the run itself are still to come.",
  "with every setting frozen, on fresh <span class=\"nw\">seeds 5–9</span>, was to run under version 7 on the project\n"
  "          owner's Framework Desktop computer: its self-test was ready on 15 Oct at 00:42 UTC, the handoff was made at\n"
  "          07:47 UTC, the project owner typed their approval at their own terminal at 15:45 UTC, and the approval was\n"
  "          checked and recorded on 16 Oct at 02:27 UTC; later that day version 7 was ended without running, so the\n"
  "          deciding test will not run under it.")
E("B10", "text", "After the stop, sources: the shared branch's head as last fetched (0377dd7, 16 Oct 12:38 UTC)",
  "(draft pull request 1; head <code>e91452d</code>, 15 Oct 2026, 03:57 UTC)",
  "(draft pull request 1; head <code>0377dd7</code>, 16 Oct 2026, 12:38 UTC)")
E("B11", "text", "After the stop, sources: where the handoff record and the approval's import record are",
  "Version 7 is in <code>stage2_v7/</code> and its self-test reports in <code>stage2_v7_transfer/</code>\n"
  "      (<code>7173a31</code>, <code>66a8cd3</code>, <code>e8fe0fb</code>).",
  "Version 7 is in <code>stage2_v7/</code> and its self-test reports in <code>stage2_v7_transfer/</code>\n"
  "      (<code>7173a31</code>, <code>66a8cd3</code>, <code>e8fe0fb</code>), its handoff record in "
  "<code>stage2_v7/handoff/</code> (<code>3a8ce5b</code>) and the approval's import record in\n"
  "      <code>stage2_v7/import_v7_3dfcb6140e2b8e2c.json</code> (<code>02e591b</code>).")
# TD3+BC moves its target copies only inside the actor update (algorithms/td3_bc.py: _do_actor_update applies
# optax.incremental_update with tau to actor_target and critic_target; it runs when it % policy_freq == 0, and
# policy_freq = 2 in td3_bc.Args and in experiments/shift/learners.py). 1 - 0.995**200 = 0.633.
E("B12", "text", "TD3+BC settings: the copies move only every second critic update, so 63% takes 200 moves = 400 critic updates",
  "          <td>Each update moves the slowly averaged copies 0.5% of the way to the current networks.</td>\n"
  "          <td>The target changes slowly: after 200 updates about 63% of a change has reached the copies. This keeps\n"
  "            the critic from chasing its own moving target.</td></tr>",
  "          <td>Each time the actor updates, that is every second critic update, the slowly averaged copies move 0.5% of\n"
  "            the way to the current networks.</td>\n"
  "          <td>The target changes slowly: after 200 such moves, that is 400 critic updates, about 63% of a change has\n"
  "            reached the copies (1 − 0.995<sup>200</sup> ≈ 0.63). This keeps the critic from chasing its own moving\n"
  "            target.</td></tr>")

# ---------------------------------------------------------------- fix round 1 of page version 20
# The Methods chart (#m-placements-chart): its guide lines (the cap 1.5 and the limit 2, class knee) lighter than the
# wider-range diagnostic (--k-none), which is the same gray as --muted in the dark theme (finding C8).
E("N10", "neutral", "Methods: the chart's guide lines lighter than its data (dark: diagnostic vs guides 1.9 -> 30.6)",
  "svg.chart .knee { stroke: var(--muted); stroke-width: 1.2; stroke-dasharray: 4 4; }\n",
  "svg.chart .knee { stroke: var(--muted); stroke-width: 1.2; stroke-dasharray: 4 4; }\n"
  "#methods svg.chart .knee { stroke: var(--rule-strong); }   /* guide lines lighter than the data they frame */\n")
# The weighting test's files: merged into the shared repository's main branch on 15 Oct 2026 at 17:18 UTC (commit
# 03726de, "Merge pull request #3 from dbayha7/claude/shift-hosts"; the branch claude/shift-hosts, 6488de2, was pushed
# at 16:34 UTC): experiments/shift_iw (the code; world.py byte-identical to the workspace's), experiments/iw (SPEC.md,
# spec.json and its .sha256), outputs/shift_iw/explore (433 files), EXPERIMENTS.md entry 17, and the critic viewer's
# scripts and data in docs/thrusterchain/qviewer. No runs/ folder is on main (runs/shift_iw_v5 is not). Read from the
# repository's remote-tracking refs (fetched 15 Oct 2026, 19:40 UTC), read only (finding A2).
E("B16", "text", "weighting test: its status line (the files are in the shared repository since 17:18 UTC)",
  "<p class=\"status\"><span class=\"tag warn\">not yet in the repository</span> This test's specification, code, run\n"
  "      records and experiment-log entry are not yet in the shared repository. They are in the project's local\n"
  "      workspace (<code>bca-shift-work</code>) and in an unpushed clone of the repository on the machine that ran the\n"
  "      test (<a href=\"#iw-srcs\">Sources</a>, at the end of this section).</p>",
  "<p class=\"status\"><span class=\"tag\">in the repository since 15 Oct 2026</span> This test's specification\n"
  "      (<code>experiments/iw</code>), code (<code>experiments/shift_iw</code>), exploratory results\n"
  "      (<code>outputs/shift_iw</code>) and experiment-log entry were merged into the shared repository's main branch on\n"
  "      15 Oct 2026 at 17:18 UTC (commit <code>03726de</code>). Its run records\n"
  "      (<code>runs/shift_iw_v5</code>) are still only in an unpushed clone of the repository on the machine that ran\n"
  "      the test (<a href=\"#iw-srcs\">Sources</a>, at the end of this section).</p>")
E("B17", "text", "weighting test, sources: where the files are now",
  "<p class=\"note\">Not yet in the repository. No branch of the shared repository holds this test's files:\n"
  "      <code>experiments/shift_iw</code> is not there, and no branch's <code>EXPERIMENTS.md</code> has an entry for\n"
  "      this test. The files are in the local workspace folder <code>bca-shift-work</code>, beside the\n"
  "      repository:",
  "<p class=\"note\">In the shared repository since 15 Oct 2026, 17:18 UTC: its main branch (commit <code>03726de</code>)\n"
  "      holds this test's code (<code>experiments/shift_iw</code>), specification (<code>experiments/iw</code>),\n"
  "      exploratory results (<code>outputs/shift_iw/explore</code>) and experiment-log entry (<code>EXPERIMENTS.md</code>,\n"
  "      entry 17). When this section was written they were only in the local workspace folder <code>bca-shift-work</code>,\n"
  "      beside the repository:")
E("B18", "text", "Names used in the code: the weighting test's code is in the shared repository",
  "<tr><td>the weighting test</td><td>not yet in the shared repository. The code is in the local workspace, "
  "<code>bca-shift-work/<wbr>shift_iw</code>, with its specification in <code>bca-shift-work/<wbr>iw</code>; in the "
  "unpushed clone that ran the test it sits at <code>experiments/<wbr>shift_iw</code> (<a href=\"#iw-srcs\">Sources</a>, "
  "at the end of the weighting test)</td></tr>",
  "<tr><td>the weighting test</td><td><code>experiments/<wbr>shift_iw</code>, with its specification in "
  "<code>experiments/<wbr>iw</code>: in the shared repository's main branch since 15 Oct 2026, 17:18 UTC (before that "
  "only in the local workspace, <code>bca-shift-work/<wbr>shift_iw</code> and <code>bca-shift-work/<wbr>iw</code>; "
  "<a href=\"#iw-srcs\">Sources</a>, at the end of the weighting test)</td></tr>")
E("B19", "text", "critic viewer, sources: its scripts and the test's code are in the repository now; the runs are not",
  "<p class=\"note\">Not yet in the shared repository. This section's data come from the weighting test's\n",
  "<p class=\"note\">Partly in the shared repository since 15 Oct 2026, 17:18 UTC: its main branch now holds the\n"
  "      weighting test's code (<code>experiments/shift_iw</code>) and the scripts named below with their data file\n"
  "      (<code>docs/thrusterchain/qviewer</code>), but not the runs they read (<code>runs/shift_iw_v5</code>), so this\n"
  "      section still cannot be rebuilt from the repository alone. The rest of this note was written before that merge.\n"
  "      This section's data come from the weighting test's\n")
E("B21", "text", "After the stop, status box: the near-80 test (its final pre-registration approved by the owner and "
  "the approval recorded; it now runs on the owner's desktop computer; no result yet)",
  "        <li><b>The weighting test</b> (its own data and its own points scale): <a href=\"#weighting-test\">the next\n"
  "          section</a>.</li>",
  "        <li><b>Near-80 test</b> (approved and running; no result yet): a test that also leaves the commands near 80%\n"
  "          throttle (any thruster from 70% to 90%) out of the training data, which the project owner asked about in a\n"
  "          page comment. Its final pre-registration, frozen on 16 Oct at 01:33 (<code>e3e87c3</code>), was approved by\n"
  "          the owner at their own terminal at 11:50 (committed at 11:51, <code>f79b89c</code>) and the approval\n"
  "          recorded at 11:53 (<code>287380b</code>); it now runs on the project owner's Framework Desktop computer.</li>\n"
  "        <li><b>The weighting test</b> (its own data and its own points scale): <a href=\"#weighting-test\">the next\n"
  "          section</a>.</li>")
# Page version 27: the sentence pointed to the header's status box ("both in the status box at the top"), which the
# live page removed; it now points to After the stop's box (B21 states the near-80 test there).
E("B22", "text", "Results, opening: the two newer lines of work (no results yet; the near-80 test runs now)",
  "all three in <a href=\"#ats-status\">After the stop</a>); and <a href=\"#weighting-test\">the weighting test</a>,\n"
  "      on its own data, in the next section.</p>",
  "all three in <a href=\"#ats-status\">After the stop</a>); and <a href=\"#weighting-test\">the weighting test</a>,\n"
  "      on its own data, in the next section. Two more have no results yet: runs of four\n"
  "      learners, whose plan the project owner approved in chat and must still approve at their own terminal, and a test\n"
  "      that leaves the commands near 80% throttle out of the training data, whose final plan the owner approved at\n"
  "      their own terminal on 16 Oct and which now runs on the owner's desktop computer\n"
  "      (<a href=\"#ats-status\">After the stop</a>).</p>")
# Page version 27: Stage 2's outcome looked ahead to seeds 5-9, which will not run under version 7 (B7, B9).
E("B24", "text", "Stage 2's outcome: seeds 5-9 were to decide whether it holds on fresh seeds; version 7 was ended "
  "without running",
  "about 88% (seeds 0 and 1 only). Whether this holds on fresh seeds is what seeds 5–9 will decide.</dd>",
  "about 88% (seeds 0 and 1 only). Seeds 5–9 were to decide whether this holds on fresh seeds; version 7, which\n"
  "          was to run them, was ended on 16 Oct 2026 without running.</dd>")
# Page version 27: Stage 1's plan, "What happened to this plan", said its deciding test "still runs under version 7".
E("B25", "text", "After the stop, what happened to the plan: its deciding test will not run under version 7, which was "
  "ended without running",
  "The plan is not withdrawn: its deciding test, on fresh seeds\n"
  "      5–9, still runs under version 7.",
  "The plan is not withdrawn, but its deciding test, on fresh seeds\n"
  "      5–9, will not run under version 7, which was ended on 16 Oct 2026 without running.")


def ids_of(text):
    return re.findall(r'\sid="([^"]+)"', text)


def main(argv):
    if "--list" in argv:
        for e in EDITS:
            print(f'{e["id"]:5} {"optional" if e["optional"] else "required":8} {e["group"]:11} {e["purpose"]}')
        return 0
    pos = [a for a in argv if not a.startswith("--")]
    opts = {}
    for k in ("--only", "--skip", "--status-time", "--status-date", "--version"):
        if k in argv:
            i = argv.index(k)
            if i + 1 >= len(argv):
                raise SystemExit(f"{k} needs a value")
            opts[k] = argv[i + 1]
            if argv[i + 1] in pos:
                pos.remove(argv[i + 1])
    if len(pos) != 2:
        raise SystemExit(__doc__)
    src, dst = pos
    import os
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("refusing to overwrite the input; write to a new file")
    t = opts.get("--status-time", STATUS_TIME)
    if not re.fullmatch(r"\d\d:\d\d", t):
        raise SystemExit("--status-time must be HH:MM (UTC)")
    d = opts.get("--status-date", STATUS_DATE)
    if not re.fullmatch(r"[1-9]\d? (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}", d):
        raise SystemExit("--status-date must read like 16 Oct 2026")
    v = opts.get("--version", PAGE_VERSION)
    if not re.fullmatch(r"[1-9]\d{0,3}", v):
        raise SystemExit("--version must be a page version number")
    edits = [dict(e) for e in EDITS]
    if "--no-optional" in argv:
        edits = [e for e in edits if not e["optional"]]
    if "--only" in opts:
        keep = set(opts["--only"].split(","))
        unknown = keep - {e["id"] for e in EDITS}
        if unknown:
            raise SystemExit("unknown edit ids: " + ", ".join(sorted(unknown)))
        edits = [e for e in edits if e["id"] in keep]
    if "--skip" in opts:
        drop = set(opts["--skip"].split(","))
        edits = [e for e in edits if e["id"] not in drop]
    for e in edits:
        e["new"] = e["new"].replace("{T}", t).replace("{V}", v).replace("{D}", d)
        if "\r" in e["old"] or "\r" in e["new"]:
            raise SystemExit(f'{e["id"]}: an edit string holds a CR byte')
    raw = open(src, "rb").read()
    text = raw.decode("utf-8")
    cr_in = text.count("\r")
    ids_in = ids_of(text)
    # every anchor must be found exactly as often as stated, before anything is changed
    bad = []
    for e in edits:
        n = text.count(e["old"])
        if n != e["count"]:
            hint = ""
            if n == 0 and e["new"] in text:
                hint = " (its replacement is already in the page: applied before?)"
            bad.append(f'  {e["id"]}: anchor found {n} times, expected {e["count"]}{hint}\n'
                       f'      anchor starts: {e["old"][:110]!r}')
    if bad:
        raise SystemExit("apply_color.py: nothing written; these anchors do not match:\n" + "\n".join(bad))
    for e in edits:
        n = text.count(e["old"])
        if n != e["count"]:
            raise SystemExit(f'{e["id"]}: anchor occurs {n} times after the earlier edits (expected {e["count"]}); nothing written')
        text = text.replace(e["old"], e["new"])
    if text.count("\r") != cr_in:
        raise SystemExit("the edits would add CR bytes; nothing written")
    ids_out = ids_of(text)
    dup_in = {i for i in ids_in if ids_in.count(i) > 1}
    dup_out = {i for i in ids_out if ids_out.count(i) > 1} - dup_in
    if dup_out:
        raise SystemExit("the edits would duplicate ids: " + ", ".join(sorted(dup_out)) + "; nothing written")
    with open(dst, "wb") as f:
        f.write(text.encode("utf-8"))
    print(f"apply_color.py: applied {len(edits)} edits ({sum(e['optional'] for e in edits)} optional), "
          f"status time {d}, {t} UTC, page version {v} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
