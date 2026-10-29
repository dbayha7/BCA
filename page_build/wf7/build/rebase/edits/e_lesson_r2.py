"""Review round 2 (cloud, 16 Oct 2026): the lesson. A1 (every chart names both axes, on the chart and to screen
readers; the axis titles are the A1 verifier's, tested), C2/D1/F2 (step 7.2: version 7 was ended without running), D2
(1.2's distance line sums the squares as shown), D3 (6.3's exponent from the current as shown; the ratio's inputs to 6
decimals), D4 (4.4's mean Q to 4 decimals; the intro's "≈" rule), D5/E1 (typed fields 44 px on touch screens), D6
(5.5's formula fits at 390 px), D7 (2.3's Try it shows and sets the world), D8/F13 (design seed defined), D9 (6.3's
constant named), D10 (4.6's digits explained), D12 (no nested parentheses), E3 (the state map's trail at full
contrast), E5 (the evidence tags 44 px on touch screens), E6 (7.2's table links 44 px on touch screens)."""
import rep

L, C, F = "lesson/lesson.js", "lesson/lesson.css", "lesson/lesson_fragment.html"
E = [
    # A1
    (L, "role: 'img', 'aria-label': C.title || '', tabindex: C.hover",
        "role: 'img', 'aria-label': (C.title || '') + (C.x.label ? ' Horizontal axis: ' + C.x.label + '.' : '') + (C.y.label ? ' Vertical axis: ' + C.y.label + '.' : ''), tabindex: C.hover", 1),
    (L, "x: { min: -1, max: 1, label: 'throttle a', ticks: [-1, -0.8, -0.5, 0, 0.5, 0.8, 1] }, y: { min: -1, max: 1, ticks: [-1, -0.5, 0, 0.5, 1] },",
        "x: { min: -1, max: 1, label: 'throttle a', ticks: [-1, -0.8, -0.5, 0, 0.5, 0.8, 1] }, y: { min: -1, max: 1, label: 'push', ticks: [-1, -0.5, 0, 0.5, 1] },", 2),
    (L, "x: { min: -1, max: 100, label: 'step t' }, y: { min: lo, max: 0 },", "x: { min: -1, max: 100, label: 'step t' }, y: { min: lo, max: 0, label: 'reward rₜ' },", 1),
    (L, "label: 'steps ahead t' }, y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1] },", "label: 'steps ahead t' }, y: { min: 0, max: 1, label: 'weight γᵗ', ticks: [0, 0.25, 0.5, 0.75, 1] },", 1),
    (L, "y: { min: -160, max: 0, ticks: [-150, -100, -50, 0] },", "y: { min: -160, max: 0, label: 'discounted sum so far', ticks: [-150, -100, -50, 0] },", 1),
    (L, "y: { min: -13.8, max: -13.2, ticks: [-13.8, -13.6, -13.4, -13.2] },", "y: { min: -13.8, max: -13.2, label: 'value (discounted return)', ticks: [-13.8, -13.6, -13.4, -13.2] },", 1),
    (L, "y: { min: -23.5, max: -23.1, ticks: [-23.5,", "y: { min: -23.5, max: -23.1, label: 'one-step target y', ticks: [-23.5,", 1),
    (L, "y: { min: 0, max: Math.ceil(lossMax * 1.15 * 100) / 100 },", "y: { min: 0, max: Math.ceil(lossMax * 1.15 * 100) / 100, label: 'critic loss' },", 1),
    (L, "y: { min: Math.floor(qmin * 1.1), max: 0 },", "y: { min: Math.floor(qmin * 1.1), max: 0, label: 'mean Q₁ at the actor\\'s commands' },", 1),
    (L, "y: { min: 0, max: 4, ticks: [0, 1, 2, 3, 4] },", "y: { min: 0, max: 4, label: 'score', ticks: [0, 1, 2, 3, 4] },", 1),
    (L, "height: 170, x: { min: 0, max: n + 2, ticks: tks, fmt: function (v) { return FI(v); } }, y: { min: 0, max: mx * 1.1, fmt:",
        "height: 170, x: { min: 0, max: n + 2, label: 'sorted score', ticks: tks, fmt: function (v) { return FI(v); } }, y: { min: 0, max: mx * 1.1, label: 'mass', fmt:", 1),
    (L, "label: 'sorted score', ticks: tks, fmt: function (v) { return FI(v); } }, y: { min: 0, max: 1, ticks: [0, 0.5, 0.9] },",
        "label: 'sorted score', ticks: tks, fmt: function (v) { return FI(v); } }, y: { min: 0, max: 1, label: 'running total of the masses', ticks: [0, 0.5, 0.9] },", 1),
    (L, "}) }, y: { min: 0, max: Math.ceil(mx * 1.15) },", "}) }, y: { min: 0, max: Math.ceil(mx * 1.15), label: 'draws crossing at that rank' },", 1),
    (L, "label: 'rank where a draw crosses' }, y: { min: 0, max: Math.ceil(mx * 1.15) },", "label: 'rank where a draw crosses' }, y: { min: 0, max: Math.ceil(mx * 1.15), label: 'draws crossing at that rank' },", 1),
    (L, "y: { min: 0.95, max: 1.55, ticks: [1, 1.1, 1.2, 1.3, 1.4, 1.5] },", "y: { min: 0.95, max: 1.55, label: 'dose m', ticks: [1, 1.1, 1.2, 1.3, 1.4, 1.5] },", 1),
    (L, "y: { min: 0, max: 2.6, ticks: [0, 0.5, 1, 1.5, 2, 2.5] },", "y: { min: 0, max: 2.6, label: 'how often (density)', ticks: [0, 0.5, 1, 1.5, 2, 2.5] },", 1),
    # C2/D1/F2
    (L, "n: 'not run yet', tag: 'pre', tt: 'pre-registered plan, not run yet'",
        "n: 'not run: version 7 was ended without running', tag: 'pre', tt: 'pre-registered plan; version 7, which was to run it, was ended without running'", 1),
    (F, "its confirmatory run on fresh seeds 5 to 9 has not run yet.",
        "its confirmatory run on fresh seeds 5 to 9 did not take place: its plan, version 7, was ended without running.", 1),
    # D2
    (L, "L.set(fl >= 0 ? 'd' + sub(fl + 1) : 'd', ['√(' + par(sp) + '² + ' + par(svv) + '²)', '√(' + F(p2) + ' + ' + F(v2) + ')', '√' + F(p2 + v2), F(d)], d);",
        "var s2 = parseShown(F(p2)) + parseShown(F(v2)); L.set(fl >= 0 ? 'd' + sub(fl + 1) : 'd', ['√(' + par(sp) + '² + ' + par(svv) + '²)', '√(' + F(p2) + ' + ' + F(v2) + ')', '√' + F(s2), F(d)], d);", 1),
    # D3
    (L, "'exp(' + F(e, 4).replace(/0+$/, '').replace(/\\.$/, '') + ')'",
        "'exp(' + F(4 * (parseShown(F(c, 4)) - 0.6), 4).replace(/0+$/, '').replace(/\\.$/, '') + ')'", 1),
    (L, "[F(hte, 4) + ' / ' + F(htr, 4), F(hte / htr, 4)]", "[F(hte, 6) + ' / ' + F(htr, 6), F(hte / htr, 4)]", 1),
    # D4
    (L, "' / ' + F(Math.abs(q), 3), F(lam)], lam, 'mean Q₁ at the actor\\'s commands was ' + F(q, 3)",
        "' / ' + F(Math.abs(q), Math.abs(q) < 10 ? 4 : 3), F(lam)], lam, 'mean Q₁ at the actor\\'s commands was ' + F(q, Math.abs(q) < 10 ? 4 : 3)", 1),
    (F, "when rounding makes them differ in the last digit;", "when rounding makes them differ slightly in the last digits;", 1),
    # D7
    (L, "    add(box, cSeg.el, sT.el, sG.el);",
        "    var wSeg = Seg({ label: 'World', value: EP.world, options: WORLDS, onChange: function (k) { EP.set('world', k, me); } });\n"
        "    add(box, cSeg.el, wSeg.el, sT.el, sG.el);", 1),
    (L, "    EP.on(function () { cSeg.set(EP.ctrl); draw(false); });",
        "    EP.on(function () { cSeg.set(EP.ctrl); wSeg.set(EP.world); draw(false); });", 1),
    # D8/F13
    (F, "(the page&#x27;s 7.527 is design seed 900&#x27;s)",
        "(the page&#x27;s 7.527 is design seed 900&#x27;s; a design seed is a recording made only to design and check the world)", 1),
    # D9
    (F, "M(λ) at 84–90", "the constant that makes each curve&#x27;s total 1 (the page&#x27;s M(ξ)) at 84–90", 1),
    # D10
    (F, '<span class="ls-r">=</span> 148.5</span></span></div>',
        '<span class="ls-r">=</span> 148.5</span> <span class="ls-wn">(the 98.9 points and step 3.5&#x27;s 1.50 per point, with more digits)</span></span></div>', 1),
    # D12
    (F, '<span class="ls-wn">(had the coin been 0.5 or more (4 decimals))</span>', '<span class="ls-wn">(had the coin been 0.5 or more; 4 decimals)</span>', 1),
    (F, '<span class="ls-wn">(a turn keeps the distance, the shrink multiplies it by 0.98 (check: √(0.320267² + 0.035468²) gives the same))</span>',
        '<span class="ls-wn">(a turn keeps the distance and the shrink multiplies it by 0.98; check: √(0.320267² + 0.035468²) gives the same)</span>', 1),
    # D5/E1
    (C, "  #learn .ls-num { min-height: 44px; }", "  #learn input.ls-num { min-height: 44px; }", 1),
    # E5, E6 (in the touch block)
    (C, "  #learn .ls-wl { min-height: 44px; }\n}",
        "  #learn .ls-wl { min-height: 44px; }\n"
        "  #learn .ls-tag { display: inline-flex; align-items: center; min-height: 44px; position: relative; border-color: transparent; }\n"
        "  #learn .ls-tag::before { content: \"\"; position: absolute; inset: 50% 0 auto; height: 23px; transform: translateY(-50%); box-sizing: border-box; border: 1px solid currentColor; border-radius: 999px; pointer-events: none; }\n"
        "  #learn table.ls-t td > a { display: inline-flex; align-items: center; min-height: 44px; }\n}", 1),
    # D6
    (C, "#learn .ls-rule .math-display { margin: 2px 0 -4px; }",
        "#learn .ls-rule .math-display { margin: 2px 0 -4px; }\n#learn .ls-rule .math-display mjx-container > svg { max-width: 100%; height: auto; }", 1),
    # E3
    (C, "#learn .ls-map-trail { fill: none; stroke: var(--accent); stroke-width: 1.2; opacity: .5; }",
        "#learn .ls-map-trail { fill: none; stroke: var(--muted); stroke-width: 1.2; }", 1),
]
rep.apply(E)
