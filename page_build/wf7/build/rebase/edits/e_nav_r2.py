"""Review round 2 (cloud, 16 Oct 2026): the navigation. B1 (nv-boot is removed before the load's go(), so a deep link
to a part that has no box under nv-boot lands; #lc-table5 also reveals the "All five side by side" table), B2 (peek
mode compares with the sheet's own peek height, so a press no longer raises the sheet again), B3 (a table arrives in
its scroll box: 16 px down and with its ring), B5 (the skip link brings the section's top back when it is off
screen), F9 (the note names the bar's button), F10 (the critic viewer's quality names), F20 (the bar's landmark
name)."""
import json
import rep

rep.apply([
    ("nav/src/nav.js", '    applyLayout(false);\n    if (target) {',
     '    applyLayout(false);\n    html.classList.remove("nv-boot");                      // before go(): under nv-boot no section can be scrolled to\n    if (target) {', 1),
    ("nav/src/nav.js", '    html.classList.remove("nv-boot");\n    startBar();', '    startBar();', 1),
    ("nav/src/nav.js",
     '        if (want < side.getBoundingClientRect().height) side.style.setProperty("--nv-peek-h", Math.round(Math.max(W.innerHeight * 0.3, want)) + "px");\n'
     '        else side.style.removeProperty("--nv-peek-h");',
     '        side.style.removeProperty("--nv-peek-h");           // compared with the sheet\'s own peek height, not the one lowered last\n'
     '        if (want < side.getBoundingClientRect().height) side.style.setProperty("--nv-peek-h", Math.round(Math.max(W.innerHeight * 0.3, want)) + "px");', 1),
    ("nav/src/nav.js", '  function scrollToEl(el, near) {\n    const r = el.getBoundingClientRect();',
     '  // a table scrolls inside its box (div.tbl, overflow: auto), which would clip its scroll margin and its arrival ring\n'
     '  const boxOf = el => (el.matches("table") && el.parentElement && el.parentElement.matches(".tbl") ? el.parentElement : el);\n'
     '  function scrollToEl(el, near) {\n    el = boxOf(el);\n    const r = el.getBoundingClientRect();', 1),
    ("nav/src/nav.js", '    if (!el || units.includes(el) || pages.includes(el)) return;\n    el.classList.remove("nv-flash");',
     '    if (!el || units.includes(el) || pages.includes(el)) return;\n    el = boxOf(el);\n    el.classList.remove("nv-flash");', 1),
    ("nav/src/nav.js",
     '    if (a.id === "nv-skip") { e.preventDefault(); const p = cur === learn && one && curPage ? (curPage.querySelector("h3, h4") || headOf(cur)) : headOf(cur); focusOn(p); return; }',
     '    if (a.id === "nv-skip") {\n'
     '      e.preventDefault(); const p = cur === learn && one && curPage ? (curPage.querySelector("h3, h4") || headOf(cur)) : headOf(cur);\n'
     '      const r = p.getBoundingClientRect();                 // off screen (Tab came round from the end): the section\'s top comes back\n'
     '      if (r.bottom < 0 || r.top > W.innerHeight - barSpace()) scrollToEl(cur === learn && one && curPage ? curPage : cur, false);\n'
     '      focusOn(p); return;\n    }', 1),
    ("nav/src/nav.js", "on a phone, the middle button of the bar at the bottom",
     "otherwise the button with the section's name in the bar at the bottom", 2),
    ("nav/src/start_bar.html", "on a phone, the middle button of the bar at the bottom",
     "otherwise the button with the section&#8217;s name in the bar at the bottom", 1),
    ("nav/src/nav.js", 'row("Dataset quality", all("#qv-panel [data-qv-q]"), byAttr("data-qv-q")),',
     'row("Dataset quality", all("#qv-panel [data-qv-q]"), byAttr("data-qv-q"), named(QUALITIES, "data-qv-q")),', 1),
    ("nav/src/nav_markup.html", 'aria-label="Previous and next"', 'aria-label="Navigation bar: contents, this section, previous and next"', 1),
    ("nav/src/nav.css", "html.nv-on main [id], html.nv-on .ls-step, html.nv-on .ls-intro { scroll-margin-top: 16px; }",
     "html.nv-on main [id], html.nv-on main .tbl, html.nv-on .ls-step, html.nv-on .ls-intro { scroll-margin-top: 16px; }", 1),
])
p = rep.W7 / "nav" / "src" / "static_edits.json"
d = json.loads(p.read_text(encoding="utf-8"))
OLD = '<table class="lc-t5" id="lc-table5">'
if not any(e.get("old") == OLD for e in d["edits"]):
    d["edits"].append({"kind": "label", "unit": "weighting-test", "entry": "table", "label": "lc-table5 reveal (review B1)",
                       "line": None, "old": OLD, "new": '<table class="lc-t5" id="lc-table5" data-nav-reveal="#lc-tq [data-lc-tq=all]">',
                       "count": 1})
    d["n_edits"] = len(d["edits"])
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("wrote nav/src/static_edits.json (lc-table5 reveal); %d edits" % len(d["edits"]))
