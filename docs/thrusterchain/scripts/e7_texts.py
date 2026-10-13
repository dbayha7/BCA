"""Review fixes of the tooltip pass: the code snippets and texts e7_apply.py replaces. Every number in a new text is on
the page already (checked by e7_check.py)."""

# --------------------------------------------------------------------------------------- the tooltip script (edit 1)
JS1 = [
    ("tuples: the whole value, never cut",
     '        if (isCur) p.append(el("b", null, cut(v, 240))); else p.append(cut(v, 64));\n',
     '        if (isCur) p.append(el("b", null, v)); else p.append(v);          // the whole value, never cut\n'),
    ("tuples: no cut helper",
     '  const cut = (s, n) => s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s;\n', ''),
    ("tab stops: first of its kind per section",
     '  function prepTip(el) {\n'
     '    if (!el.dataset || (el.dataset.tip == null && !el.dataset.tipRef) || tip.contains(el) || descBox.contains(el)) return;\n'
     '    let d = descOf.get(el);\n'
     '    if (!d) {\n'
     '      if (!el.matches(FOCUSABLE) && el.dataset.tipFocus !== "no") { el.setAttribute("tabindex", isOn() ? "0" : "-1"); autoTab.add(el); }\n',
     '  // only the first element with a given tooltip in a section (or in one <details>) takes a Tab stop; its repeats show\n'
     '  // the same tooltip on hover and tap and keep the same description for assistive technology\n'
     '  const tabSeen = new WeakMap();\n'
     '  function firstOfItsKind(el) {\n'
     '    const scope = el.closest("details, section[id], figure[id]") || document.body;\n'
     '    let seen = tabSeen.get(scope);\n'
     '    if (!seen) tabSeen.set(scope, seen = new Set());\n'
     '    const k = tipText(el);\n'
     '    if (seen.has(k)) return false;\n'
     '    seen.add(k);\n'
     '    return true;\n'
     '  }\n'
     '  function prepTip(el) {\n'
     '    if (!el.dataset || (el.dataset.tip == null && !el.dataset.tipRef) || tip.contains(el) || descBox.contains(el)) return;\n'
     '    if (el.dataset.tipDesc === "no" && el.dataset.tipFocus === "no") return;   // a formula: its legend sits beside it\n'
     '    let d = descOf.get(el);\n'
     '    if (!d) {\n'
     '      if (!el.matches(FOCUSABLE) && el.dataset.tipFocus !== "no" && firstOfItsKind(el)) { el.setAttribute("tabindex", isOn() ? "0" : "-1"); autoTab.add(el); }\n'),
    ("controls: the ACTION selector",
     '  const FOCUSABLE = "a[href], button, input, select, textarea, summary, iframe, [tabindex], [contenteditable]";\n',
     '  const FOCUSABLE = "a[href], button, input, select, textarea, summary, iframe, [tabindex], [contenteditable]";\n'
     '  const ACTION = "button, a[href], input, select, textarea, summary, [role=\'button\'], [role=\'switch\']";   // a tap works these\n'),
    ("state: input modality, keyboard cell, press and hold",
     '  let cur = null, how = "", pt = null, tShow = 0, tHide = 0, lastType = "mouse", escHid = false;\n',
     '  let cur = null, how = "", pt = null, tShow = 0, tHide = 0, lastType = "mouse", escHid = false;\n'
     '  let ptrAt = -1e9, ptrTarget = null, keyAt = -1e9, kbCell = null, lp = { a: null }, held = null, heldUntil = 0;\n'),
    ("long rows and legends: a wider box",
     '    tip.replaceChildren(...nodes);\n    tip.hidden = false;\n',
     '    tip.replaceChildren(...nodes);\n'
     '    tip.classList.toggle("tt-wide", tip.textContent.length > 360);   // a long row or legend: a wider box, fewer lines\n'
     '    tip.hidden = false;\n'),
    ("touch: press and hold on a control, a slider at once",
     '  document.addEventListener("pointerdown", e => {\n'
     '    lastType = e.pointerType || "mouse";\n'
     '    if (cur && !cur.contains(e.target)) hide();      // a tap or click elsewhere closes it\n'
     '  }, true);\n',
     '  document.addEventListener("pointerdown", e => {\n'
     '    lastType = e.pointerType || "mouse"; ptrAt = performance.now(); ptrTarget = e.target;\n'
     '    if (cur && !cur.contains(e.target)) hide();      // a tap or click elsewhere closes it\n'
     '    clearTimeout(lp.timer); lp = { a: null };\n'
     '    if (lastType !== "touch" || !isOn()) return;\n'
     '    // on a touch screen a tap works a control and shows no tooltip; a press and hold shows it (a slider\'s shows at once)\n'
     '    const a = anchorOf(e.target);\n'
     '    if (!a || !a.matches(ACTION)) return;\n'
     '    if (a.matches(\'input[type="range"]\')) { show(a, "slide"); tHide = setTimeout(hide, 2500); return; }\n'
     '    const p = { x: e.clientX, y: e.clientY };\n'
     '    lp = { a, p, timer: setTimeout(() => { held = a; heldUntil = Infinity; show(a, "tap", p); }, 500) };\n'
     '  }, true);\n'
     '  document.addEventListener("pointermove", e => {      // a finger that moves is scrolling or dragging, not holding\n'
     '    if (lp.a && e.pointerType === "touch" && Math.hypot(e.clientX - lp.p.x, e.clientY - lp.p.y) > 10) { clearTimeout(lp.timer); lp = { a: null }; }\n'
     '  }, { capture: true, passive: true });\n'
     '  const release = () => { clearTimeout(lp.timer); lp = { a: null }; if (held && heldUntil === Infinity) heldUntil = performance.now() + 800; };\n'
     '  document.addEventListener("pointerup", release, true);\n'
     '  document.addEventListener("pointercancel", release, true);\n'
     '  document.addEventListener("contextmenu", e => { if (lp.a || (held && performance.now() < heldUntil)) e.preventDefault(); }, true);\n'
     '  window.addEventListener("click", e => {            // a press and hold showed the tooltip: it does not also work the control\n'
     '    if (!held) return;\n'
     '    const h = held, live = performance.now() < heldUntil;\n'
     '    held = null;\n'
     '    if (live && h.contains(e.target)) { e.preventDefault(); e.stopImmediatePropagation(); }\n'
     '  }, true);\n'
     '  // a slider\'s tooltip stays while it is dragged and goes 2.5 s after the last change\n'
     '  document.addEventListener("input", e => { if (cur === e.target && how === "slide") { clearTimeout(tHide); tHide = setTimeout(hide, 2500); } }, true);\n'),
    ("touch: a tap on a control shows no tooltip",
     '    const a = anchorOf(e.target);\n'
     '    if (!a || !isOn()) return;\n'
     '    if (cur === a && how === "tap") {',
     '    const a = anchorOf(e.target);\n'
     '    if (!a || !isOn() || a.matches(ACTION)) return;   // a control: a tap works it (press and hold shows its tooltip)\n'
     '    if (cur === a && how === "tap") {'),
    ("keyboard cell: remembered unless a click or tap on the cell focused it",
     '    if (t.dataset && t.dataset.tt) setStop(t);\n',
     '    if (t.dataset && t.dataset.tt) {\n'
     '      setStop(t);\n'
     '      const clicked = ptrTarget && t.contains(ptrTarget) && ptrAt > keyAt && performance.now() - ptrAt < 1000;\n'
     '      kbCell = clicked ? null : t;                   // a cell clicked or tapped leaves the arrow keys to the page\n'
     '    }\n'),
    ("keyboard cell: the last key",
     '  // Esc closes the tooltip first (and only the tooltip, inside the walkthrough pop-out too)\n',
     '  window.addEventListener("keydown", () => { keyAt = performance.now(); }, true);   // focus that follows a key is keyboard focus\n'
     '  // Esc closes the tooltip first (and only the tooltip, inside the walkthrough pop-out too)\n'),
    ("keyboard cell: arrows only for a cell reached by keyboard",
     '  // inside a table: the arrow keys, Home and End move between cells (before the walkthrough reads the arrows)\n'
     '  document.addEventListener("keydown", e => {\n'
     '    const t = e.target;\n'
     '    if (!t || !t.dataset || !t.dataset.tt || e.altKey || e.metaKey || e.shiftKey) return;\n',
     '  // inside a table reached by keyboard: the arrow keys, Home and End move between cells (before the walkthrough reads\n'
     '  // the arrows). A cell focused by a mouse click leaves the arrows to the page: scrolling, the walkthrough\'s sections\n'
     '  document.addEventListener("keydown", e => {\n'
     '    const t = e.target;\n'
     '    if (!t || !t.dataset || !t.dataset.tt || t !== kbCell || e.altKey || e.metaKey || e.shiftKey) return;\n'),
]

# --------------------------------------------------------------------------------------- the drawings script (edit 2)
CHART_OLD = (
    '  document.querySelectorAll("svg.chart title").forEach(t => {\n'
    '    const p = t.parentElement;\n'
    '    if (!p || p.localName === "svg" || p.hasAttribute("data-tip")) return;\n'
    '    p.setAttribute("data-tip-focus", "no");\n'
    '    p.setAttribute("data-tip", t.textContent.replace(/\\s+/g, " ").trim());\n'
    '    t.remove();\n'
    '  });\n')
CHART_NEW = (
    '  // chart points: one wide target per column of points (every series at one band of the water\'s push), easy to hit\n'
    '  // with a finger; the points and lines let the pointer through to it\n'
    '  const NS = "http://www.w3.org/2000/svg", SERIES = ["pt-sa", "pt-rt", "pt-rtpush"];\n'
    '  const cap = x => x.charAt(0).toUpperCase() + x.slice(1);\n'
    '  document.querySelectorAll("svg.chart").forEach(svg => {\n'
    '    const cols = new Map();\n'
    '    svg.querySelectorAll("circle").forEach(c => {\n'
    '      const t = c.querySelector("title");\n'
    '      if (!t) return;\n'
    '      const x = Number(c.getAttribute("cx")), k = x.toFixed(1);\n'
    '      if (!cols.has(k)) cols.set(k, { x, items: [] });\n'
    '      cols.get(k).items.push({ txt: t.textContent.replace(/\\s+/g, " ").trim(), rank: SERIES.indexOf(c.getAttribute("class")) });\n'
    '      t.remove();\n'
    '    });\n'
    '    svg.querySelectorAll("title").forEach(t => {          // any other titled part keeps its own tooltip\n'
    '      const p = t.parentElement;\n'
    '      if (!p || p.localName === "svg" || p.hasAttribute("data-tip")) return;\n'
    '      p.setAttribute("data-tip-focus", "no");\n'
    '      p.setAttribute("data-tip", t.textContent.replace(/\\s+/g, " ").trim());\n'
    '      t.remove();\n'
    '    });\n'
    '    if (!cols.size) return;\n'
    '    const xs = [...cols.values()].sort((a, b) => a.x - b.x);\n'
    '    const gap = xs.length > 1 ? Math.min(...xs.slice(1).map((c, i) => c.x - xs[i].x)) : 24;\n'
    '    const ax = [...svg.querySelectorAll("line.axis")].find(l => l.getAttribute("x1") === l.getAttribute("x2"));\n'
    '    const y1 = ax ? Math.min(+ax.getAttribute("y1"), +ax.getAttribute("y2")) : 0;\n'
    '    const y2 = ax ? Math.max(+ax.getAttribute("y1"), +ax.getAttribute("y2")) : svg.viewBox.baseVal.height;\n'
    '    const g = document.createElementNS(NS, "g");\n'
    '    g.setAttribute("class", "pt-hits");\n'
    '    xs.forEach(col => {\n'
    '      const items = col.items.sort((a, b) => (a.rank < 0 ? 99 : a.rank) - (b.rank < 0 ? 99 : b.rank));\n'
    '      const i = items[0].txt.indexOf("): ");\n'
    '      const head = i > 0 && items.every(it => it.txt.slice(0, i + 3) === items[0].txt.slice(0, i + 3)) ? i + 3 : 0;\n'
    '      const r = document.createElementNS(NS, "rect");\n'
    '      r.setAttribute("x", (col.x - gap / 2).toFixed(1)); r.setAttribute("width", gap.toFixed(1));\n'
    '      r.setAttribute("y", String(y1)); r.setAttribute("height", String(y2 - y1));\n'
    '      r.setAttribute("data-tip-focus", "no");\n'
    '      if (head) r.setAttribute("data-tip-title", cap(items[0].txt.slice(0, head - 2)));\n'
    '      r.setAttribute("data-tip", items.map(it => cap(it.txt.slice(head))).join("\\n"));\n'
    '      g.appendChild(r);\n'
    '    });\n'
    '    svg.insertBefore(g, svg.querySelector("text.ann, path[class^=\'ln-\'], circle"));\n'
    '  });\n'
    '  // a formula shown on its own line: its tooltip is the legend beside it, which defines each of its symbols\n'
    '  const keys = [...document.querySelectorAll(".symkey")];\n'
    '  document.querySelectorAll("div.math-display").forEach(d => {\n'
    '    const sec = d.closest("section[id]");\n'
    '    const k = keys.find(x => (d.compareDocumentPosition(x) & Node.DOCUMENT_POSITION_FOLLOWING) && x.closest("section[id]") === sec);\n'
    '    const dl = k && k.querySelector("dl"), h = k && k.querySelector(".symkey-h");\n'
    '    if (!dl) return;\n'
    '    if (!dl.id) dl.id = "symkey-" + (keys.indexOf(k) + 1);\n'
    '    d.querySelectorAll("mjx-container > svg").forEach(f => {\n'
    '      f.setAttribute("data-tip-focus", "no");\n'
    '      f.setAttribute("data-tip-desc", "no");\n'
    '      f.setAttribute("data-tip-title", h ? h.textContent.trim() : "What the symbols mean");\n'
    '      f.setAttribute("data-tip-ref", dl.id);\n'
    '    });\n'
    '  });\n'
    '  // the Play buttons\' tooltips follow their label, Play or Pause\n'
    '  [["play", "the episode"], ["td-play", "the demo episode"]].forEach(([id, what]) => {\n'
    '    const b = document.getElementById(id);\n'
    '    if (!b) return;\n'
    '    const play = [b.getAttribute("data-tip-title") || "", b.getAttribute("data-tip") || ""];\n'
    '    const sync = () => {\n'
    '      const tp = b.textContent.trim() === "Pause" ? ["Pause", "Pause " + what + " on the state on screen."] : play;\n'
    '      if (b.getAttribute("data-tip-title") !== tp[0]) b.setAttribute("data-tip-title", tp[0]);\n'
    '      if (b.getAttribute("data-tip") !== tp[1]) b.setAttribute("data-tip", tp[1]);\n'
    '    };\n'
    '    new MutationObserver(sync).observe(b, { childList: true, characterData: true, subtree: true });\n'
    '    sync();\n'
    '  });\n')
JS2 = [
    ("chart columns, formula legends, Play labels", CHART_OLD, CHART_NEW),
    ("channel: a wave against the current reads minus",
     '        "c + w₁ = " + num(s[10], 3) + " + " + num(s[11], 3) + " = " + num(wat.push[t], 3) +\n',
     '        "c + w₁ = " + num(s[10], 3) + (num(s[11], 3).startsWith(MINUS) ? " − " + num(s[11], 3).slice(1) : " + " + num(s[11], 3)) +\n'
     '        " = " + num(wat.push[t], 3) +\n'),
    ("drawings script: header comment",
     '   - It only reads the live panel\'s data and state',
     '   - Chart points get one wide target per column; a formula on its own line shows its legend; the Play buttons\'\n'
     '     tooltips follow their label.\n'
     '   - It only reads the live panel\'s data and state'),
]

CSS = """/* review fixes: a wider box for long rows and legends; a formula's legend in its tooltip; chart columns; controls */
.tt { max-height: calc(100vh - 16px); overflow: hidden; }
.tt.tt-wide { max-width: min(520px, calc(100vw - 16px)); }
.tt-rich > div { margin: 0 0 3px; }
.tt-rich > div:last-child { margin-bottom: 0; }
.tt-rich > div > dt { display: inline; font-weight: 700; color: var(--ink); }
.tt-rich > div > dt::after { content: "\\2002"; }
.tt-rich > div > dd { display: inline; margin: 0; }
.tt-rich sub, .tt-rich sup { line-height: 0; }
html:not(.tt-off) mjx-container > svg[data-tip-ref] { cursor: help; }
svg.chart .pt-hits rect { fill: transparent; }
svg.chart .pt-hits ~ path, svg.chart .pt-hits ~ circle { pointer-events: none; }
button[data-tip], button[data-tip-ref] { -webkit-touch-callout: none; -webkit-user-select: none; user-select: none; }
"""

# --------------------------------------------------------------------------------------- texts: (what, old, new, mode, n)
# mode "esc": the text sits in an attribute or a template (html-escaped, quote=True); "raw": as written in the page.
# n None: a symbol's tooltip, replaced at every place the symbol has it.
TEXTS = [
    ("no-stall world option",
     "Push keeps rising to 100% throttle. At or below 80% it is identical to the stalling world, so the data cannot "
     "tell the two apart. Points here: 100 is the rising-thrust controller.",
     "Push keeps rising to 100% throttle. At or below 80% it is identical to the stalling world, so the data left "
     "after the densest-15% removal, where only 60 of the 42,500 kept training rows go past 80%, cannot tell the two "
     "apart. Points here: 100 is the rising-thrust controller.", "esc", 1),
    ("guide: normal driving only, its score",
     "78.4 points in the recorded conditions and 71.3 in strong current; about what copying the data after the "
     "removal gives.",
     "78.4 points in the recorded conditions and 71.3 in strong current: about what an exact copy of the data after "
     "the removal would do. The cloned actor, a network fitted to copy that data, scores less: 56.3 in the recorded "
     "conditions and 54.4 in strong current.", "esc", 1),
    ("lead-in: no throttle settles, but not at rest",
     "With No throttle the floats stop drifting and level out, but away from rest: a steady current leaves float 1 "
     "settled 2.92 times the current from rest (about 1.5 at a current of 0.5), and the wave keeps floats 2 and 3 "
     "swinging (float 2 out to 1.04 from rest at amplitude 0.4). The cost is paid on every float's distance from rest "
     "at every step, settled or not, so no throttle",
     "With no throttle the floats stop drifting further, but never come back to rest: a steady current holds float 1 "
     "2.92 times the current from rest (about 1.5 at a current of 0.5), and the wave keeps floats 2 and 3 swinging "
     "(float 2 out to 1.04 from rest at amplitude 0.4). The cost is paid on every float's squared distance from rest "
     "in every state, settled or not, so no throttle", "raw", 1),
    ("no throttle: its score (option and guide)",
     "Its cost is all float cost, paid at every step on the floats' distance from rest:",
     "Its cost is all float cost, paid in every state on the floats' squared distance from rest:", "esc", 2),
    ("diagram legend: swing, in steps",
     "A float's period, the steps one swing takes, rounded under each float: 62.8 for float 1 down to 7.8 for float 5.",
     "A float's period, the steps one swing takes, shown rounded to whole steps under each float: 63 for float 1 "
     "(62.8) down to 8 for float 5 (7.8).", "raw", 1),
    ("Arguments: the worked-example line",
     "In your own code <code>action</code> is whatever four throttles your controller sets. A different seed draws a "
     "different start, and a different command gives a different reward, step info and next state, so every number "
     "after <code>#</code> changes.</p>",
     "In your own code, <code>action</code> is whatever four throttles your controller sets. A different seed draws a "
     "different start, with its own current, wave and floats, and a different command gives a different reward, thrust "
     "and next state, so the printed numbers change, all but the state number t = 1, which is the same after one step "
     "whatever the seed or command.</p>", "raw", 1),
    ("term: caution",
     "A harder pull toward the recorded command at chosen training rows: the actor's cloning term is multiplied there "
     "by a factor above 1, the caution multiplier or dose. BCA's doses lie from 1 to 1.5.",
     "A harder pull toward the recorded command at chosen training rows: the pull is multiplied there by a factor of "
     "1 or more, the caution multiplier or dose. BCA's doses run from 1 up to, but not reaching, 1.5.", "esc", 1),
    ("term: textbook rule K",
     "if push always equalled throttle. The textbook command is −Kx.",
     "if push always equalled throttle. The textbook command is minus K times the 13 observed numbers x.", "esc", 1),
    ("symbol: K",
     "that turns the 13 observed numbers into a command: the textbook command is −Kx. It is the best linear rule",
     "that turns the 13 observed numbers into a command: the textbook command is minus K times the 13 observed "
     "numbers x. It is the best linear rule", "esc", 19),
    ("term: stall-aware controller",
     "The textbook command from the rule K with every throttle capped at 80%, so it never stalls.",
     "The textbook command (each throttle a fixed weighted sum of the 13 observed numbers, the best such rule if push "
     "always equalled throttle), with every throttle capped at 80%, so it never stalls.", "esc", 2),
    ("term: episode",
     "Nothing ends it early; it is cut off after step 100.",
     "Nothing ends it early; it is cut off after 100 steps (states 0 to 99).", "esc", 1),
    ("recording controller: how it sets the throttles (option and guide)",
     "Normal driving most of the time: it holds against the current and waves, damps the floats' swinging a quarter "
     "as hard as the textbook rule K, and wobbles.",
     "Normal driving in most states (every state outside the full-throttle region and about half of those inside "
     "it): it holds against the current and waves, damps the floats' swinging a quarter as hard as the textbook rule "
     "K, and adds a random wobble.", "esc", 2),
    ("symbol: k = 1", "1 in the main arm.", "1 in the main run (the tables call it the main arm).", "esc", 2),
    ("symbol: k = 0.5", "0.5 in the half-strength arm.", "0.5 in the half-strength run.", "esc", 1),
    ("symbol: W", "In the main arm: how much more", "In the main run: how much more", "esc", 3),
    ("legend: k in After the stop",
     "The penalty's strength: 1 in the main arm, 0.5 at half strength.",
     "The penalty's strength: 1 in the main run (the tables call it the main arm), 0.5 in the half-strength run.",
     "raw", 1),
    ("symbol: Q1",
     "The critic has two estimates (two heads); the actor climbs the first.",
     "The critic gives two estimates, one from each of its two networks (the two critics in TD3's name); the actor "
     "climbs the first.", "esc", 2),
    ("symbol: Q1, Q2", "The frozen critic's two heads.",
     "The frozen critic's two estimates, one from each of its two networks (the two critics in TD3's name).",
     "esc", 1),
    ("symbol: beta",
     "the larger of a conformal value (90% of rows) and a Bayesian one (95% credible).",
     "the smallest multiplier that covers the error on 90% of the calibration rows, or a more cautious one found by "
     "re-weighting the rows at random 1,000 times (at least 95% of the re-weightings accept it), whichever is "
     "larger.", "esc", 5),
    ("symbol: S", "the root-mean-square change of its command,",
     "the square root of the average squared change of its command,", "esc", 3),
    ("symbol: n", "Four fresh standard normal numbers each step: the recording controller's wobble",
     "Four fresh random numbers each step, with mean 0 and standard deviation 1: the recording controller's wobble",
     "esc", 1),
    ("legend: n, normal driving",
     "Four fresh standard normal numbers each step, independent of each other; times 0.3 they are the wobble.",
     "Four fresh random numbers each step, each with mean 0 and standard deviation 1 and independent of the others; "
     "times 0.3 they are the wobble.", "raw", 1),
    ("legend: n, the jitter", "A little jitter: four standard normal numbers times 0.03.",
     "A little jitter: four fresh random numbers (mean 0, standard deviation 1) times 0.03.", "raw", 1),
    ("term: harm checks", "Four pre-registered checks run before BCA:",
     "Four pre-registered checks (their rules written down, and fingerprinted, before any learner ran) run before "
     "BCA:", "esc", 1),
    ("term: importance weighting",
     "the weight ω(c) of the weighting test, the test density of the current divided by the training density.",
     "the weight ω(c) of the weighting test: how likely the test is to draw the row's current c, divided by how "
     "likely training was to draw it.", "esc", 1),
    ("term: weight",
     "ω(c), the test density of its current c divided by the training density, above 1",
     "ω(c), how likely the test is to draw its current c divided by how likely training was to draw it, above 1",
     "esc", 1),
    ("weighting test: the weight formula",
     "The test's density of the current c divided by the training density: above 1",
     "How likely the test is to draw the current c, divided by how likely training was to draw it: above 1",
     "esc", 1),
    ("guide: actor of the critic trained after the removal (label)",
     "<dt>Actor of the gapped critic</dt>", "<dt>Actor of the critic trained after the removal</dt>", "raw", 1),
    ("guide: actor of the critic trained after the removal (throttles)",
     "against a frozen critic that learned from the data after the removal (the gapped critic), which rates",
     "against a frozen critic that learned from the data after the removal (the page also calls it the gapped "
     "critic), which rates", "esc", 1),
    ("guide: actor of the critic trained after the removal (score)",
     "−14.3 points in strong current; the same actor with its commands capped at 80% scores 101.8.",
     "−14.3 points in strong current (means over seeds 0–4); the same actor with its commands capped at 80% scores "
     "101.8.", "esc", 1),
    ("guide: exact-error penalty actor (label)",
     "<dt>Oracle-penalty actor (After the stop)</dt>",
     "<dt>Actor trained with an exact-error penalty (After the stop)</dt>", "raw", 1),
    ("guide: exact-error penalty actor (throttles)",
     "The gapped critic's actor trained with a penalty on the value of its own command,",
     "Trained like the actor of the critic trained after the removal, against the same frozen critic, plus a penalty "
     "on the value of its own command,", "esc", 1),
    ("stall-aware controller: how it sets the throttles (option and guide)",
     "The textbook command −Kx: each throttle is a fixed weighted sum of the 13 observed numbers,",
     "The textbook command (minus K times the 13 observed numbers x): each throttle is a fixed weighted sum of the 13 "
     "observed numbers,", "esc", 2),
    ("guide: exact-error penalty actor (80% limit)",
     "Partly: past 80% at 54% of the strong-current states it visits",
     "Mostly not: past 80% at 54% of the strong-current states it visits", "esc", 1),
    ("drawing legend: the off-edge arrow",
     "Its position number <i>p</i>, one of its two numbers; both are 0 at rest.</dd>",
     "Its position number <i>p</i>, one of its two numbers; both are 0 at rest. An arrow after it (↓ or ↑) means the "
     "float is past the edge of the drawing in that direction.</dd>", "raw", 1),
    ("dynamics legend: theta and rho",
     "each swing turns by θ and shrinks by ρ every step, plus a weak coupling.</dd></div>",
     "each swing turns by θ and shrinks by ρ every step, plus a weak coupling.</dd></div><div><dt>θ, ρ</dt><dd>Each "
     "float's turn per step, in radians (0.100 for float 1 to 0.801 for float 5), and its shrink per step (0.9817 to "
     "0.9549), listed in the table below.</dd></div>", "raw", 1),
    ("dynamics legend: R is a turn by any angle",
     "<dd>A turn by the angle θ, in radians (2π is one full turn). <i>R</i>(0.27) turns the wave 0.27 radians per "
     "step: a cycle of 23.27 steps.</dd>",
     "<dd>A turn by any angle, in radians (2π is one full turn); not a float's θ. The wave uses <i>R</i>(0.27), which "
     "turns it 0.27 radians per step: a cycle of 23.27 steps.</dd>", "raw", 1),
    ("symbol: kappa",
     "How steeply the critic's value falls away from its favourite command, along each throttle: 2.36 measured.",
     "How steeply the critic's value falls away from its favourite command, along each throttle: 2.36, the median "
     "measured on the exact value of normal driving (standing in for the critic), at normal driving's own command.",
     "esc", 2),
    ("legend: kappa", "The curvature of the critic's hill: 2.36 measured.",
     "The curvature of the critic's hill: 2.36, the median measured on the exact value of normal driving, at its "
     "own command.", "raw", 1),
    ("symbol: d", 'data-tip-title="d, the number of throttles" data-tip="4."',
     'data-tip-title="d, the number of throttles" data-tip="The number of throttles in a command: 4."', "raw", 1),
    ("symbol: z", 'data-tip-title="z" data-tip="Any number in [z]₊."',
     'data-tip-title="z" data-tip="Stands for whatever number is inside the brackets."', "raw", 1),
    ("term: recorded conditions",
     "current 0.1 to 0.7, wave amplitude 0.1 to 0.4, float spread 0.5. Out of distribution",
     "current 0.1 to 0.7, wave amplitude 0.1 to 0.4, and float numbers starting with a spread (standard deviation) "
     "of 0.5 around rest. Out of distribution", "esc", 2),
    ("chart legend: dots",
     "<dd>One band's average; hover over or tap a dot for its numbers. The table below lists them all.</dd>",
     "<dd>One band's average. Hover over or tap anywhere in a band's column for its three numbers; the table below "
     "lists them all, for keyboard and screen-reader use and with tooltips off.</dd>", "raw", 1),
    ("switch note: press and hold on a touch screen",
     "Tooltips explain terms, controls and table columns when you hover over, focus or tap them; a table cell's "
     "tooltip shows its whole row.</span>",
     "Tooltips explain terms, symbols, controls and table columns when you hover over, focus or tap them; a table "
     "cell's tooltip shows its whole row. On a touch screen, press and hold a button to see its tooltip.</span>",
     "raw", 1),
    ("controller guide starts closed", '<details class="ctrl-guide" open>', '<details class="ctrl-guide">', "raw", 1),
]

# column descriptions (inside the tooltip script's one-line COLS object, json with the text as written; e5_cols.py
# holds the same texts): (what, old, new, n)
COLS_TEXTS = [
    ("column: Spread σ",
     "σ (sigma): the larger standard deviation, over the harm-check seeds, of the seed-count check's two paired "
     "differences, in points.",
     "A possible value of σ (sigma): the larger standard deviation, over the harm-check seeds, of the seed-count "
     "check's two paired differences, in points. The seed-count check never ran, so these rows only show what N each "
     "σ would require.", 1),
    ("column: current c", "This column, scaled until thruster 1 is at full throttle, is the full-throttle pattern P.",
     "This column, with its sign flipped and scaled until thruster 1 is at full throttle, is the full-throttle "
     "pattern P.", 1),
    ("columns: t = 0 to 7", "of design seed 900's training episode 2, counted from 0",
     "of training episode 2 of the recorded log this page uses (design seed 900), counted from 0", 8),
]
