<script>
/* Navigation (one IIFE; it declares no global name, and reads the live panel's window.render and window.setPlaying as
   properties). The page's sections ("units": the header with At a glance, the lesson, each section of <main> and the
   live panel "Watch one episode") are shown one at a time: the others carry hidden="until-found", so the browser's Find
   still searches them and shows the one that holds a match (the "beforematch" event); a browser without that gets the
   class nv-off (display: none on screen only). Contents (#nv) lists the units by part; In this section (#nv-side) lists
   the parts of the unit on screen, found by rules (nav_spec section 6), with each live panel's choices under its entry:
   every choice button presses the panel's own control. Previous and Next walk through those parts in page order and on
   into the next or previous unit. Every in-page link, the address fragment, Back and Forward come through go().
   A unit or lesson page that is hidden gets "walkthrough:leave"; one that is shown gets "walkthrough:enter" and then a
   window "resize": the page's panels redraw on these (the names are historical: they come from the old full
   walkthrough mode, which this navigation replaces). A change of the page column's width that is not a window resize
   (a column docked or hidden) is followed by a window "resize" too. */
(() => {
  "use strict";
  const doc = document, html = doc.documentElement, W = window;
  const $ = id => doc.getElementById(id);
  const main = doc.querySelector("main"), nav = $("nv"), side = $("nv-side"), stopsEl = $("nv-stops");
  const stopsWrap = $("nv-stops-wrap"), bar = $("nv-bar"), pager = $("nv-pager"), live = $("nv-live"), pill = $("nv-pill");
  const STATE_CLASSES = ["nv-boot", "nv-js", "nv-on", "nv-one", "nv-in-lesson", "nv-nav-open", "nv-nav-closed", "nv-side-open",
    "nv-side-closed", "nv-pill-on", "nv-pill-away", "nv-peek"];
  if (!main || !nav || !side || !stopsEl || !stopsWrap || !bar) { html.classList.remove(...STATE_CLASSES); return; }
  const T0 = performance.now();
  const store = {
    get(k) { try { return W.localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { W.localStorage.setItem(k, v); } catch (e) { /* storage unavailable: the choice lasts this visit */ } },
  };
  const mm = q => W.matchMedia(q);
  const calm = mm("(prefers-reduced-motion: reduce)");
  const mqWide = mm("(min-width: 1240px)"), mqRail = mm("(min-width: 1000px)"), mqShort = mm("(max-height: 599px)");
  const UF = "onbeforematch" in doc.body;                  // hidden="until-found" works in this browser
  const BASE_TITLE = doc.title || "ThrusterChain";
  let running = false, printing = false;

  // ---------- words as shown on screen ----------
  function seenText(n) {                                   // no hidden descriptions, formulas, drawings or templates
    let out = "";
    for (const c of n.childNodes) {
      if (c.nodeType === 3) out += c.nodeValue;
      else if (c.nodeType === 1 && !c.hidden && !c.matches(".tt-sr, .nv-sr, mjx-container, svg, template")) out += seenText(c);
    }
    return out.replace(/\s+/g, " ").trim();
  }
  const txt = el => (el ? seenText(el) : "");
  const trunc = (s, n) => { if (s.length <= n) return s; const c = s.lastIndexOf(" ", n - 1); return s.slice(0, c > n * 0.6 ? c : n - 1).replace(/[,;:.]$/, "") + "…"; };
  const cutAt = (s, seps, n) => { let k = -1; for (const p of seps) { const j = s.indexOf(p); if (j >= 16 && (k < 0 || j < k)) k = j; } return trunc((k > 0 ? s.slice(0, k) : s).replace(/\.$/, ""), n); };
  const slug = s => s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "").split("-").filter(Boolean).slice(0, 6).join("-") || "part";

  // ---------- units, parts and the lesson ----------
  const UNIT_SEL = "main > header#top, main > section[id], main > figure.panel[id]";
  const units = [...main.children].filter(n => n.matches("header#top, section[id], figure.panel[id]"));
  if (!units.length) { html.classList.remove(...STATE_CLASSES); return; }
  const keyOf = u => (u.id === "top" ? "glance" : u.id);
  const unitOf = el => (el && el.closest ? el.closest(UNIT_SEL) : null);
  const learn = $("learn") && units.includes($("learn")) ? $("learn") : null;
  const intro = learn ? learn.querySelector(":scope > .ls-intro") : null;
  const steps = learn ? [...learn.querySelectorAll(".ls-steps > .ls-step[id]")] : [];
  const pages = (intro ? [intro] : []).concat(steps);   // the lesson's pages: the intro (page 0) and the steps
  const secLinks = new Map();
  nav.querySelectorAll("a[data-unit]").forEach(a => {      // a Contents entry whose unit is not on the page is removed
    const k = a.dataset.unit;
    if (k === "glance" ? $("top") : ($(k) && units.includes($(k)))) secLinks.set(k, a); else a.closest("li").remove();
  });
  units.forEach((u, i) => {                                // a unit added later without a Contents entry gets one
    const k = keyOf(u); if (secLinks.has(k)) return;
    const prev = i > 0 && secLinks.get(keyOf(units[i - 1]));
    const li = doc.createElement("li"); li.className = "nv-sec";
    const a = doc.createElement("a"); a.href = "#" + k; a.dataset.unit = k;
    a.textContent = txt(u.querySelector(":scope > h2, :scope > .panel-title")) || k;
    li.appendChild(a);
    if (prev) prev.closest("li").after(li); else { const ul = nav.querySelector(".nv-secs"); if (ul) ul.appendChild(li); }
    secLinks.set(k, a);
  });
  const partOf = u => { const a = secLinks.get(keyOf(u)); const ul = a && a.closest("ul"); return ul ? txt($(ul.getAttribute("aria-labelledby"))) : ""; };
  const nameOf = u => txt(secLinks.get(keyOf(u))) || keyOf(u);
  const headOf = u => (u.id === "top" ? u.querySelector("h1") : u.querySelector(":scope > h2, :scope > .panel-title")) || u;
  const titleOf = p => (p === intro ? "About this lesson" : txt(p.querySelector("h3, h4")) || p.id);
  const pageOf = el => { const st = el.closest && el.closest(".ls-step"); return st && steps.includes(st) ? st : (intro || steps[0] || null); };

  // ---------- state ----------
  let one = store.get("tc-whole") !== "1";                 // one unit at a time (false: the whole page)
  let cur = null, stops = [], si = 0, curPage = null, hold = null, spyRaf = 0, proxies = [], chSig = "", idleRows = [];
  const seen = new Set((store.get("tc-learn-seen") || "").split(" ").filter(Boolean));

  // ---------- hiding and showing, and the events the page's panels listen for ----------
  const hideEl = el => { if (UF) el.setAttribute("hidden", "until-found"); else el.classList.add("nv-off"); };
  const showEl = el => { if (el.getAttribute("hidden") === "until-found") el.removeAttribute("hidden"); el.classList.remove("nv-off"); };
  const isOff = el => el.getAttribute("hidden") === "until-found" || el.classList.contains("nv-off");
  const hasSim = el => el.id === "sim" || !!el.querySelector("#sim");
  function leave(el) {                                     // before hiding: nothing keeps playing unseen
    if (hasSim(el) && typeof W.setPlaying === "function") { try { W.setPlaying(false); } catch (e) { /* the panel's own */ } }
    el.dispatchEvent(new CustomEvent("walkthrough:leave", { bubbles: true }));
  }
  const enterQ = new Set(); let enterRaf = 0;
  function enter(el) { enterQ.add(el); if (!enterRaf) enterRaf = requestAnimationFrame(flushEnter); }
  function flushEnter() {                                  // after showing: redraw at the width now on screen
    enterRaf = 0; const list = [...enterQ]; enterQ.clear();
    list.forEach(el => el.dispatchEvent(new CustomEvent("walkthrough:enter", { bubbles: true })));
    setMainW(); W.dispatchEvent(new Event("resize"));
    if (list.some(hasSim) && typeof W.render === "function") { try { W.render(); } catch (e) { /* the panel's own */ } }
  }
  function hideUnit(u) {
    if (isOff(u)) return;
    if (u === learn && curPage && !isOff(curPage)) curPage.dispatchEvent(new CustomEvent("walkthrough:leave", { bubbles: true }));
    leave(u); hideEl(u);
  }
  function showUnit(u) {
    if (!isOff(u)) return;
    showEl(u); enter(u);
    if (u === learn && curPage && !isOff(curPage) && curPage !== u) enter(curPage);
  }

  // ---------- the parts of a unit: the rules of nav_spec section 6 (the same code as rules.js) ----------
  const PANEL = "figure.ix, figure.panel, div.panel, figure.ovfig";
  const BLOCK = ".finding, .check, .box";
  const CAND = ["h3", "h4", ".subsec", ".finding", ".check", ".box", "details", "table", ".chartbox", PANEL,
    "nav.pipe:not(.pipe-sm)", "[data-nav-label]"].join(", ");
  const TAG = { table: "Table", chart: "Chart", panel: "Live panel", finding: "Finding", check: "Check", details: "Detail" };
  const within = (el, sel, u) => { for (let p = el.parentElement; p && p !== u; p = p.parentElement) if (p.matches(sel)) return p; return null; };
  function blockLabel(b) {
    const h = b.querySelector(":scope > h2, :scope > h3, :scope > h4, :scope > .f-title");
    if (h) return { label: txt(h), head: h.matches("h2, h3, h4") ? h : null };
    const s = b.querySelector(":scope > p:first-child > strong:first-child, :scope > p:first-child > b:first-child");
    return { label: s ? txt(s).replace(/\.$/, "") : "", head: null };
  }
  function chartLabel(c) {
    const g = c.querySelector("[role=img][aria-label], [role=group][aria-label]");
    const a = g && g.getAttribute("aria-label");
    return a ? cutAt(a, [": ", ". "], 90) : cutAt(txt(c.querySelector("figcaption")), [". "], 90);
  }
  const panelLabel = p => txt(p.querySelector(":scope > .panel-title, :scope > figcaption")) || p.getAttribute("aria-label") || "";
  function detect(u) {
    const out = [];
    const add = (el, kind, level, label, extra) => { if (label) out.push(Object.assign({ el, kind, level, label: trunc(label, 140) }, extra || {})); };
    add(u, "top", 0, u === learn ? "About this lesson" : "Start of the section");
    if (u === learn) {                                     // the lesson: its steps, grouped by chapter
      steps.forEach((s, i) => add(s, "step", 1, txt(s.querySelector("h3")) || "Step " + (i + 1), { chapter: s.dataset.chapter || "" }));
      return out;
    }
    const partKind = el => (el.matches("details") ? "details" : el.matches("table") ? "table" : "part");
    const levelUnder = (el, pan, dflt) => {                // one level below the nearest listed element holding it
      for (let p = el.parentElement; p && p !== u.parentElement; p = p.parentElement) {
        const o = out.find(x => x.el === p || (x.panel && x.panel === p)); if (o) return Math.min(3, o.level + 1);
        if (p === pan) break;
      }
      const o = out.find(x => x.el === pan || x.panel === pan);
      return Math.min(3, (o ? o.level : dflt) + 1);
    };
    if (u.matches("figure.panel")) {                       // a unit that is itself a live panel: its labeled parts only
      u.querySelectorAll("[data-nav-label]").forEach(el => { if (!within(el, "[data-nav=skip]", u)) add(el, partKind(el), levelUnder(el, u, 0), el.dataset.navLabel); });
      return out;
    }
    let hl = 0;                                            // the level of the last listed heading
    const blocks = [];                                     // listed findings, checks and boxes (a panel may merge in)
    for (const el of u.querySelectorAll(CAND)) {
      if (el.closest("[data-nav=skip]")) continue;
      if (el.matches("details.srcs")) continue;            // "Sources, for the project team"
      if (within(el, "details", u)) continue;              // inside a collapsed block
      const pan = within(el, PANEL, u);
      if (pan) { if (el.dataset.navLabel) add(el, partKind(el), levelUnder(el, pan, hl), el.dataset.navLabel); continue; }
      const blk = within(el, BLOCK, u);
      if (blk) {
        if (el.matches(PANEL)) { const host = blocks.find(b => b.el === blk); if (host) { host.panel = el; continue; } }
        else {                                             // a block is one entry; a part of it that carries a label
          const host = el.dataset.navLabel && blocks.find(b => b.el === blk);   // is listed one level below it
          if (host) add(el, partKind(el), Math.min(3, host.level + 1), el.dataset.navLabel);
          continue;
        }
      }
      const lab = el.dataset.navLabel;
      if (el.matches("h3, h4")) {
        const up = el.parentElement;
        if (up.matches(".subsec, .box, .check") && el === up.firstElementChild) continue;   // the container's own title
        const lv = el.tagName === "H3" ? 1 : 2; add(el, "heading", lv, lab || txt(el)); hl = lv; continue;
      }
      if (el.matches(".subsec")) { add(el, "heading", 1, lab || txt(el.querySelector(":scope > h3, :scope > h4"))); hl = 1; continue; }
      const lv = Math.min(3, hl + 1);
      if (el.matches(PANEL)) { add(el, "panel", lv, lab || panelLabel(el)); continue; }
      if (el.matches(BLOCK)) {
        const { label, head } = blockLabel(el);
        if (el.matches(".finding") && !el.querySelector(":scope > .f-title")) continue;
        const kind = el.matches(".finding") ? "finding" : el.matches(".check") ? "check" : "box";
        const n0 = out.length; add(el, kind, lv, lab || label, { head });
        if (out.length > n0) blocks.push(out[out.length - 1]);
        continue;
      }
      if (el.matches("details")) { add(el, "details", lv, lab || cutAt(txt(el.querySelector(":scope > summary")), [": "], 90)); continue; }
      if (el.matches("table")) { const c = el.querySelector(":scope > caption"); if (c || lab) add(el, "table", lv, lab || cutAt(txt(c), [". "], 90)); continue; }
      if (el.matches(".chartbox")) { add(el, "chart", lv, lab || chartLabel(el)); continue; }
      if (el.matches("nav.pipe")) { const l = $(el.getAttribute("aria-labelledby")); add(el, "part", lv, lab || cutAt(txt(l), [", as ", ". "], 60)); continue; }
      if (lab) add(el, "part", lv, lab);
    }
    return out;
  }
  // a plain #anchor for every entry: its own id, its title heading's id, the label before it, or (only for content
  // added later without one) an id made from the section and the label's first words; never on a section or figure
  function anchorOf(s, u) {
    if (s.kind === "top") return keyOf(u);
    if (s.el.id) return s.el.id;
    if (s.head && s.head.id) return s.head.id;
    const lb = s.el.getAttribute("aria-labelledby"), le = lb && $(lb);
    if (le && s.el.previousElementSibling === le) return lb;
    const t = s.el.matches("section, figure") ? (s.head || null) : s.el;
    const base = keyOf(u) + "-" + slug(s.label); let id = base, n = 2;
    while ($(id)) id = base + "-" + n++;
    if (t) { t.id = id; return id; }
    return keyOf(u);
  }
  const isHome = (el, u) => el === u || (u.id === "top" && el.id === "glance") || (u === learn && (el === headOf(u) || el === intro));
  function stopIndexOf(el) {                               // the entry that is the target, or the last one before it
    if (cur === learn && one) { const p = el === learn || el === intro ? null : pageOf(el); const j = stops.findIndex(s => s.el === p); return j > 0 ? j : 0; }
    const x = el.id ? stops.findIndex(s => s.id === el.id) : -1;   // an entry's own anchor (it may sit before the entry)
    if (x >= 0) return x;
    let k = 0;
    for (let i = 0; i < stops.length; i++) {
      const s = stops[i].el;
      if (s === el || (s.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING)) k = i; else break;
    }
    return k;
  }

  // ---------- the live panels' choices (the version-18 Choices sidebar, now under each panel's entry) ----------
  // A row may apply to one of a panel's two modes only (the live panel's fixed controllers or its learned actor,
  // versions 21, 22 and 25): its idle() says when the other mode plays, and the row is then marked "not in use now".
  const all = sel => () => [...doc.querySelectorAll(sel)];
  const row = (label, find, key, name, opt) => Object.assign({ label, find, key, name: name || (el => txt(el.querySelector("b") || el)) }, opt || {});
  const byAttr = a => el => el.getAttribute(a) || txt(el);
  const named = (map, attr) => el => map[el.getAttribute(attr)] || txt(el.querySelector("b") || el);
  const LEVELS = { recorded: "Recorded conditions (no shift)", big_kicks: "Big starting pushes (small shift)",
    big_waves: "Big waves (medium shift)", strong_current: "Strong current (large shift)" };
  const lineSwatch = el => el.querySelector(":scope > svg");
  const simLearned = () => { const sm = $("sim"); return !!sm && sm.classList.contains("sim-learned"); };
  const simFixed = () => !simLearned();
  const QUALITIES = { random: "Random", "medium-replay": "Medium-replay", medium: "Medium", "medium-expert": "Medium-expert", expert: "Expert" };
  const GROUPS = [
    { panel: () => $("as-widget"), view: ".aw-thrs", rows: [
      row("Set a command", all("#as-widget [data-preset]"), byAttr("data-preset"), null, { view: "#aw-sum" }),
      row("The bars show", all('#as-widget input[name="aw-data"]'), el => el.value, named({ main: "Main test", iw: "Weighting test" }, "value")) ] },
    { panel: () => $("sim"), view: "#channel", rows: [
      row("Who drives", all('#simtog [data-tog-group="who"]'), byAttr("data-tog-value"),
        named({ fixed: "Fixed controllers", learned: "Learned actor" }, "data-tog-value")),
      row("Situation", all('#sim .opt[data-group="sit"]'), byAttr("data-value"), named(LEVELS, "data-value"), { idle: simLearned }),
      row("World", all('#sim .opt[data-group="world"]'), byAttr("data-value"), named({ stall: "Stalling", no_stall: "No-stall" }, "data-value"), { idle: simLearned }),
      row("Controller", all('#sim .opt[data-group="ctrl"]'), byAttr("data-value"),
        named({ recording: "Recording", stall_aware: "Stall-aware", rising_thrust: "Rising-thrust", no_throttle: "No throttle" }, "data-value"), { idle: simLearned }),
      // version 25: the learner (BCA and BCA IW only with the 80% hold removed; their buttons are disabled otherwise)
      row("Learner", all('#simtog [data-tog-group="lr"]'), byAttr("data-tog-value"),
        named({ td3bc: "TD3+BC", bca: "BCA", bcaiw: "BCA IW", le_none: "No extra caution", rf_exact: "Result F, exact weights",
          rf_est: "Result F, estimated weights" }, "data-tog-value"), { idle: simFixed }),
      row("Data quality", all('#simtog [data-tog-group="lq"]'), byAttr("data-tog-value"), named(QUALITIES, "data-tog-value"),
        { idle: simFixed, swatch: el => el.querySelector(".simtog-dot") }),
      row("The learner's data", all('#simtog [data-tog-group="ld"]'), byAttr("data-tog-value"),
        named({ gap: "80% hold removed", full: "All the data" }, "data-tog-value"), { idle: simFixed }),
      row("Tested on", all('#simtog [data-tog-group="lm"]'), byAttr("data-tog-value"),
        named({ training: "Training mix", strong: "Strong test mix", recorded: "Main test: recorded conditions (no shift)",
          big_kicks: "Main test: big starting pushes (small shift)", big_waves: "Main test: big waves (medium shift)",
          strong_current: "Main test: strong current (large shift)" }, "data-tog-value"), { idle: simFixed }) ] },
    { panel: () => $("task-diagram"), view: "#task-diagram > svg.dg", rows: [
      row("Situation", all("#td-modes:not([hidden]) .td-mode"), byAttr("data-sit"), named(LEVELS, "data-sit")) ] },
    { panel: () => $("stall-chart"), view: "#sc-svg", rows: [
      row("Training data shown", all("#sc-full, #sc-kept"), el => el.id, named({ "sc-full": "Before the removal", "sc-kept": "After the removal" }, "id")) ] },
    { panel: () => $("qv-panel"), view: "#qv-strip", rows: [
      row("Dataset quality", all("#qv-panel [data-qv-q]"), byAttr("data-qv-q"), named(QUALITIES, "data-qv-q")),
      row("The critic's data", all("#qv-panel [data-qv-c]"), byAttr("data-qv-c"), named({ full: "All the data", gap: "80% hold removed" }, "data-qv-c")),
      row("Situation", all("#qv-panel [data-qv-s]"), byAttr("data-qv-s"), named({ training: "Training mix", strong: "Strong test mix" }, "data-qv-s")) ] },
    { panel: () => $("iw-panel-shift") || ($("iw-p-none") && $("iw-p-none").closest(".panel")), view: "#iw-svg-dens", rows: [
      row("Test mix", all("#iw-p-none, #iw-p-mild, #iw-p-strong"), el => el.id,
        named({ "iw-p-none": "No shift", "iw-p-mild": "Mild", "iw-p-strong": "Strong" }, "id")) ] },
    { panel: () => $("lc-panel"), view: "#lc-svg1", rows: [
      row("Data quality", all("#lc-q button"), byAttr("data-lc")),
      row("Test", all("#lc-t button"), byAttr("data-lc")),
      row("Table", all("#lc-tq button"), byAttr("data-lc-tq"), named(Object.assign({ all: "All five side by side" }, QUALITIES), "data-lc-tq")),
      row("Measure", all("#lc-m button"), byAttr("data-lc")),
      row("Show", all("#lc-spread"), el => el.id, () => "Spread across seeds"),
      row("Lines, actor only", all("#lc-leg1 button"), byAttr("data-lc"), el => txt(el), { swatch: lineSwatch, view: "#lc-svg1" }),
      row("Lines, full TD3+BC", all("#lc-leg2 button"), byAttr("data-lc"), el => txt(el), { swatch: lineSwatch, view: "#lc-svg2" }) ] },
  ];
  // a panel added later declares its choices (nav_spec 9.3): rows [data-nav-choices="Row label"] holding buttons with
  // aria-pressed, or radio or checkbox inputs; short name data-nav-name or the text; key data-nav-key, id, value or text
  function groupOf(p) {
    const g = GROUPS.find(x => x.panel() === p); if (g) return g;
    const rows = [...p.querySelectorAll("[data-nav-choices]")].map(r => row(r.dataset.navChoices,
      () => [...r.querySelectorAll("button[aria-pressed], input[type=radio], input[type=checkbox]")],
      el => el.dataset.navKey || el.id || el.value || txt(el), el => el.dataset.navName || trunc(txt(el.querySelector("b") || el), 32)));
    return rows.length ? { panel: () => p, view: p.dataset.navView || null, rows } : null;
  }
  const panelOf = s => (s.kind === "panel" ? s.el : s.panel ? s.panel : s.kind === "top" && cur && cur.matches(PANEL) ? cur : null);
  const isInput = el => el.tagName === "INPUT";
  const stateOf = el => (isInput(el) ? String(el.checked) : el.getAttribute("aria-pressed"));
  function choicesSig() {
    return stops.map(s => { const p = panelOf(s), g = p && groupOf(p); return g ? g.rows.map(r => r.find().map(r.key).join(",")).join(";") : ""; }).join("|");
  }
  function renderChoices(s) {
    const p = panelOf(s), g = p && groupOf(p); if (!g) return;
    const box = doc.createElement("div"); box.className = "nv-ch";
    g.rows.forEach((r, ri) => {
      const items = r.find(); if (!items.length) return;
      const rw = doc.createElement("div"); rw.className = "nv-row"; rw.setAttribute("role", "group");
      const lab = doc.createElement("span"); lab.className = "nv-rl"; lab.textContent = r.label;
      lab.id = "nv-rl-" + stops.indexOf(s) + "-" + ri; rw.setAttribute("aria-labelledby", lab.id);
      const btns = doc.createElement("div"); btns.className = "nv-btns";
      items.forEach(src => {
        const b = doc.createElement("button"); b.type = "button"; b.className = "nv-b";
        const sw = r.swatch ? r.swatch(src) : src.querySelector && src.querySelector("b .sw");
        if (sw) { const c = sw.cloneNode(true); c.removeAttribute("id"); c.setAttribute("aria-hidden", "true"); b.appendChild(c); }
        b.appendChild(doc.createTextNode(r.name(src)));
        const k = r.key(src);
        b.addEventListener("click", () => {               // press the panel's own control, found afresh by its key
          const c = r.find().find(el => r.key(el) === k);
          if (c && !c.disabled) c.click();
          requestAnimationFrame(() => { refreshChoices(); afterChoice(r.view || g.view); });
        });
        proxies.push({ b, r, k }); btns.appendChild(b);
      });
      rw.append(lab, btns); box.appendChild(rw);
      if (r.idle) idleRows.push([rw, r.idle]);
    });
    if (box.childNodes.length) { s.li.insertBefore(box, s.a.nextSibling); s.ch = box; }
  }
  function sync() {
    idleRows.forEach(([rw, f]) => { const v = !!f(); if (rw.classList.contains("nv-idle") !== v) rw.classList.toggle("nv-idle", v); });
    proxies.forEach(({ b, r, k }) => {
      const c = r.find().find(el => r.key(el) === k);
      const st = c ? stateOf(c) : null;
      if (st === null) { if (b.hasAttribute("aria-pressed")) b.removeAttribute("aria-pressed"); }
      else if (b.getAttribute("aria-pressed") !== st) b.setAttribute("aria-pressed", st);
      const dis = !c || !!c.disabled; if (b.disabled !== dis) b.disabled = dis;
    });
  }
  function refreshChoices() {                              // some panels make their buttons anew on every draw
    if (!running) return;
    const sig = choicesSig(); if (sig === chSig) { sync(); return; }
    const f = doc.activeElement, fi = f && f.classList && f.classList.contains("nv-b") ? proxies.findIndex(p => p.b === f) : -1;
    const fk = fi >= 0 ? proxies[fi].k : null, fr = fi >= 0 ? proxies[fi].r : null;
    chSig = sig; proxies = []; idleRows = [];
    stops.forEach(s => { if (s.ch) { s.ch.remove(); s.ch = null; } renderChoices(s); });
    sync(); observeHere();
    if (fr) { const p = proxies.find(x => x.r === fr && x.k === fk) || proxies.find(x => x.r === fr); if (p) p.b.focus({ preventScroll: true }); }
  }
  const firstShown = sel => { if (!sel) return null; for (const e of doc.querySelectorAll(sel)) if (e.getClientRects().length) return e; return null; };
  function afterChoice(view) {                             // keep the drawing that changed in view
    if (mqRail.matches) {                                  // the rail: a panel entirely off screen comes to the top
      requestAnimationFrame(() => {
        const v = firstShown(view); if (!v) return;
        const r = v.getBoundingClientRect();
        if (r.bottom > 0 && r.top < W.innerHeight) return;  // some of it is on screen: the page stays where it is
        const smooth = !calm.matches && Math.abs(r.top) < 1.5 * W.innerHeight;
        W.scrollBy({ top: r.top - 16, behavior: smooth ? "smooth" : "auto" });
        holdAt(smooth ? 900 : 0);
      });
      return;
    }
    if (!html.classList.contains("nv-side-open")) return;
    const isSheet = !mqShort.matches;
    if (isSheet && !html.classList.contains("nv-peek")) { html.classList.add("nv-peek"); }
    requestAnimationFrame(() => {
      const v = firstShown(view); if (!v) return;
      if (isSheet) {                                       // a drawing taller than the room above the sheet: the
        const h = v.getBoundingClientRect().height;       // sheet is lowered for this press (never below 30% of the
        const want = W.innerHeight - barSpace() - h - 24;  // screen's height)
        side.style.removeProperty("--nv-peek-h");           // compared with the sheet's own peek height, not the one lowered last
        if (want < side.getBoundingClientRect().height) side.style.setProperty("--nv-peek-h", Math.round(Math.max(W.innerHeight * 0.3, want)) + "px");
      }
      requestAnimationFrame(() => {
        const r = v.getBoundingClientRect(), limit = isSheet ? side.getBoundingClientRect().top - 8 : W.innerHeight - barSpace() - 8;
        const right = isSheet ? W.innerWidth : side.getBoundingClientRect().left;
        if (r.top >= 8 && r.bottom <= limit && r.left < right) return;
        // taller than the room left: its bottom meets the sheet's top, so the part that changed last stays in view
        W.scrollBy(0, r.height > limit - 16 ? r.bottom - limit : r.top - 16);
        holdAt(0);
      });
    });
  }

  // ---------- In this section ----------
  const caret = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 3 4 4-4 4"/></svg>';
  function renderSide() {
    stopsEl.textContent = ""; proxies = []; idleRows = [];
    $("nv-side-part").textContent = partOf(cur);
    $("nv-side-t").textContent = nameOf(cur);
    let l1 = null, ol1 = null, l2 = null, ol2 = null, chap = null, chapOl = null, chapN = 0;
    stops.forEach((s, i) => {
      s.id = anchorOf(s, cur);
      const li = doc.createElement("li"), a = doc.createElement("a");
      li.className = "nv-s nv-l" + Math.max(1, s.level);
      a.className = "nv-a"; a.href = "#" + s.id; a.dataset.i = i; a.dataset.kind = s.kind; a.title = s.label;
      const tag = (t, cls) => { const e = doc.createElement("span"); e.className = "nv-tag" + (cls ? " " + cls : ""); e.textContent = t; a.appendChild(e); };
      if (TAG[s.kind]) tag(TAG[s.kind]);
      if (s.panel) { tag(TAG.panel, "nv-tag-p"); a.classList.add("nv-has-panel"); }
      const t = doc.createElement("span"); t.className = "nv-t"; t.textContent = s.label; a.appendChild(t);
      if (panelOf(s)) { const o = doc.createElement("span"); o.className = "nv-onscr"; o.textContent = "on screen"; a.appendChild(o); }
      if (s.kind === "step" && seen.has(s.el.id)) t.classList.add("nv-seen");
      li.appendChild(a); s.li = li; s.a = a; s.ch = null;
      if (s.kind === "step") {                            // the lesson: steps under their chapter, a fold row each
        const c = s.chapter || "Steps";
        if (c !== chap) {
          chap = c; chapN++;
          const cli = doc.createElement("li"); cli.className = "nv-chap";
          const b = doc.createElement("button"); b.type = "button"; b.className = "nv-chap-b";
          chapOl = doc.createElement("ol"); chapOl.className = "nv-fold"; chapOl.id = "nv-chap-" + chapN;
          b.setAttribute("aria-controls", chapOl.id); b.setAttribute("aria-expanded", "false");
          const n = stops.filter(x => x.kind === "step" && (x.chapter || "Steps") === c).length;
          b.innerHTML = caret; const bt = doc.createElement("span"); bt.textContent = c;
          const bn = doc.createElement("span"); bn.className = "nv-chap-n"; bn.textContent = n + (n === 1 ? " step" : " steps");
          b.append(bt, bn);
          b.addEventListener("click", () => { const open = !chapOl.classList.contains("nv-open"); cli.dataset.pinned = open ? "1" : ""; setFold(cli, open); });
          cli.append(b, chapOl); stopsEl.appendChild(cli);
        }
        chapOl.appendChild(li); return;
      }
      if (s.level <= 1) { stopsEl.appendChild(li); l1 = li; ol1 = null; l2 = null; ol2 = null; }
      else if (s.level === 2 || !l2) {
        if (!l1) stopsEl.appendChild(li); else { if (!ol1) { ol1 = doc.createElement("ol"); ol1.className = "nv-fold"; l1.appendChild(ol1); } ol1.appendChild(li); }
        l2 = li; ol2 = null;
      } else { if (!ol2) { ol2 = doc.createElement("ol"); ol2.className = "nv-fold"; l2.appendChild(ol2); } ol2.appendChild(li); }
    });
    // a group's toggle: the groups that hold the current entry open by themselves; the others open by hand
    let gi = 0;
    stopsEl.querySelectorAll("li.nv-s").forEach(li => {
      const ol = li.querySelector(":scope > ol"); if (!ol) return;
      ol.id = "nv-g-" + gi++;
      const b = doc.createElement("button"); b.type = "button"; b.className = "nv-tog"; b.innerHTML = caret;
      b.setAttribute("aria-controls", ol.id); b.setAttribute("aria-expanded", "false");
      const n = ol.querySelectorAll("li").length;
      b.setAttribute("aria-label", "Show the " + n + (n === 1 ? " part" : " parts") + " under " + txt(li.querySelector(":scope > .nv-a .nv-t")));
      b.addEventListener("click", () => { const open = !ol.classList.contains("nv-open"); li.dataset.pinned = open ? "1" : ""; setFold(li, open); });
      li.insertBefore(b, li.querySelector(":scope > ol"));
    });
    chSig = choicesSig();
    stops.forEach(renderChoices);
    sync(); observeHere();
  }
  function setFold(li, open) {
    const ol = li.querySelector(":scope > ol"), b = li.querySelector(":scope > .nv-tog, :scope > .nv-chap-b");
    if (!ol || !b) return; ol.classList.toggle("nv-open", open); b.setAttribute("aria-expanded", String(open));
  }
  function openFoldsOf(i) {
    const at = stops[i] && stops[i].li;
    stopsEl.querySelectorAll("li.nv-s, li.nv-chap").forEach(li => {
      if (li.querySelector(":scope > ol")) setFold(li, !!(at && li.contains(at)) || li.dataset.pinned === "1");
    });
  }
  function markStop(i, { scrollList = true } = {}) {
    if (!stops.length) return;
    si = Math.max(0, Math.min(stops.length - 1, i));
    stops.forEach((s, k) => { if (k === si) s.a.setAttribute("aria-current", "location"); else s.a.removeAttribute("aria-current"); });
    openFoldsOf(si);
    if (scrollList) keepInList(stops[si].a);
    describe();
    clearTimeout(markStop.t);
    markStop.t = setTimeout(() => {                        // where the reader is, for "Continue reading" (not At a
      const s = stops[si];                                 // glance, where the button is, nor the lesson, which has its own)
      if (s && cur !== units[0] && cur !== learn) store.set("tc-last", JSON.stringify({ id: s.id, u: nameOf(cur), l: s.label }));
    }, 600);
  }
  function keepInList(a) {                                // the list keeps the current entry in view, unless in use
    if (!a || side.matches(":hover") || side.contains(doc.activeElement) || !stopsWrap.getClientRects().length) return;
    const wr = stopsWrap.getBoundingClientRect(), r = a.getBoundingClientRect();
    if (r.top < wr.top + 8 || r.bottom > wr.bottom - 8) stopsWrap.scrollTop += r.top - wr.top - wr.height * 0.3;
  }
  // the place: in the bar, in the head of In this section, and in Previous and Next's descriptions
  const inLessonPages = () => cur === learn && one;
  function placeOf() {
    const s = stops[si] || {}, n = stops.length;
    if (cur === learn && s.kind === "step") return { a: "Step " + (steps.indexOf(s.el) + 1) + " of " + steps.length + " · " + nameOf(cur), b: s.label, pos: "Step " + (steps.indexOf(s.el) + 1) + " of " + steps.length };
    if (cur === learn) return { a: nameOf(cur), b: "About this lesson", pos: "About this lesson" };
    const b = s.kind !== "top" ? s.label : "Start of the section";
    return { a: (si + 1) + " of " + n + " · " + nameOf(cur), b, pos: (si + 1) + " of " + n };
  }
  function neighbors() {
    const k = units.indexOf(cur), n = stops.length;
    const prev = si > 0 ? { label: stops[si - 1].kind === "top" ? "the start of " + nameOf(cur) : stops[si - 1].label } : k > 0 ? { label: "the section " + nameOf(units[k - 1]) } : null;
    const next = si < n - 1 ? { label: stops[si + 1].label } : k < units.length - 1 ? { label: "the section " + nameOf(units[k + 1]) } : null;
    return { prev, next };
  }
  function describe() {
    const p = placeOf(), s = stops[si] || {};
    $("nv-where-a").textContent = p.a; $("nv-where-b").textContent = p.b;
    $("nv-side-pos").textContent = p.pos + (s.kind === "top" ? "" : " · " + p.b);
    const { prev, next } = neighbors();
    $("nv-prev-d").textContent = prev ? "Goes to " + prev.label : "";
    $("nv-next-d").textContent = next ? "Goes to " + next.label : "";
    const off = { "nv-prev": !prev, "nv-bar-prev": !prev, "nv-next": !next, "nv-bar-next": !next };
    const PART = { "nv-prev": "nv-next", "nv-next": "nv-prev", "nv-bar-prev": "nv-bar-next", "nv-bar-next": "nv-bar-prev" };
    Object.keys(off).forEach(id => { const b = $(id); if (b && !off[id]) b.disabled = false; });
    Object.keys(off).forEach(id => {                      // a button about to be disabled hands its focus to its partner
      const b = $(id); if (!b || !off[id]) return;
      if (doc.activeElement === b) { const p = $(PART[id]); if (p && !p.disabled) p.focus({ preventScroll: true }); }
      b.disabled = true;
    });
  }
  function markContents() {
    secLinks.forEach((a, k) => { if (cur && k === keyOf(cur)) a.setAttribute("aria-current", one ? "page" : "location"); else a.removeAttribute("aria-current"); });
    // docked Contents (1240 px and up): the shown section's entry is scrolled into the list's view, unless in use
    const a = cur && secLinks.get(keyOf(cur)), l = $("nv-list");
    if (!a || !l || !mqWide.matches || html.classList.contains("nv-nav-closed") || nav.matches(":hover") || nav.contains(doc.activeElement) || !l.getClientRects().length) return;
    const r = a.getBoundingClientRect(), lr = l.getBoundingClientRect();
    if (r.top < lr.top || r.bottom > lr.bottom) l.scrollTop += r.top - lr.top - lr.height / 3;
  }
  function renderPager() {
    if (!pager) return;
    pager.textContent = "";
    const k = units.indexOf(cur);
    const mk = (u, cls, label) => {
      const a = doc.createElement("a"); a.className = "nv-pg " + cls; a.href = "#" + keyOf(u);
      const s1 = doc.createElement("span"); s1.className = "nv-pg-k"; s1.textContent = label;
      const s2 = doc.createElement("span"); s2.className = "nv-pg-t"; s2.textContent = nameOf(u);
      const s3 = doc.createElement("span"); s3.className = "nv-pg-p"; s3.textContent = partOf(u);
      a.append(s1, s2, s3); pager.appendChild(a);
    };
    if (k > 0) mk(units[k - 1], "nv-pg-prev", "← Previous section");
    if (k < units.length - 1) mk(units[k + 1], "nv-pg-next", "Next section →");
    const w = doc.createElement("button"); w.type = "button"; w.className = "nv-pg-whole"; w.textContent = "Show the whole page instead";
    w.addEventListener("click", () => setWhole(true, true));
    pager.appendChild(w);
  }
  const say = msg => { if (!live) return; live.textContent = ""; requestAnimationFrame(() => { live.textContent = msg; }); };

  // "on screen": which live panels of the unit are in view (a mark in the list; below 1000 px, the Choices button)
  let io = null; const here = new Set();
  function observeHere() {
    if (!("IntersectionObserver" in W)) return;
    if (io) io.disconnect(); here.clear(); html.classList.remove("nv-pill-on");
    const watched = stops.filter(panelOf);
    if (!watched.length) return;
    io = new IntersectionObserver(es => {
      es.forEach(e => { if (e.isIntersecting) here.add(e.target); else here.delete(e.target); });
      let on = false;
      watched.forEach(s => { const v = here.has(panelOf(s)); if (s.li) s.li.classList.toggle("nv-here", v); if (v && s.ch) on = true; });
      html.classList.toggle("nv-pill-on", on);
      queueAway();
    }, { rootMargin: "-15% 0px -15% 0px" });
    watched.forEach(s => io.observe(panelOf(s)));
  }
  // the Choices button floats at the bottom left below 1000 px; as version 21 did for the old Choices tab, it steps
  // aside rather than cover a panel's own switches (the live panel's switch bar, the learning curves' table selectors):
  // here, while it would overlap one of them on screen
  const OWN_SWITCHES = ["simtog", "lc-tctl"];
  let awayRaf = 0;
  function pillAway() {
    awayRaf = 0;
    let away = false;
    if (pill && html.classList.contains("nv-pill-on") && !mqRail.matches) {
      const p = pill.getBoundingClientRect();
      if (p.width) away = OWN_SWITCHES.some(id => {
        const n = $(id); if (!n || !n.getClientRects().length) return false;
        const r = n.getBoundingClientRect();
        return r.left < p.right + 4 && r.right > p.left - 4 && r.top < p.bottom + 4 && r.bottom > p.top - 4;
      });
    }
    if (html.classList.contains("nv-pill-away") !== away) html.classList.toggle("nv-pill-away", away);
  }
  const queueAway = () => { if (!awayRaf) awayRaf = requestAnimationFrame(pillAway); };

  // ---------- moving between units, entries and lesson pages ----------
  function setUnit(u, why) {
    if (!u || u === cur) return false;
    const old = cur;
    if (one) {
      if (old) hideUnit(old);
      if (u === learn) { const p = curPage && pages.includes(curPage) ? curPage : pages[0]; if (p) showPage(p, { quiet: true, events: false }); }
      showUnit(u);
    }
    cur = u; stops = detect(u); si = 0;
    html.classList.toggle("nv-in-lesson", u === learn);
    renderSide(); markContents(); renderPager();
    const j = u === learn && one && curPage ? stops.findIndex(s => s.el === curPage) : 0;
    markStop(Math.max(0, j), { scrollList: false });
    if (why !== "load") stopsWrap.scrollTop = 0;
    doc.title = u.id === "top" ? BASE_TITLE : nameOf(u) + " · " + BASE_TITLE;
    if (u.id === "top") startBar();
    if (why !== "spy" && why !== "load") say(nameOf(u) + ". Section " + (units.indexOf(u) + 1) + " of " + units.length + (partOf(u) ? ", in " + partOf(u) : "") + ".");
    return true;
  }
  function showPage(p, { quiet = false, events = true } = {}) {   // the lesson: one page (the intro or a step) at a time
    if (!p || !pages.includes(p)) p = pages[0];
    if (!p) return;
    if (one) pages.forEach(x => {
      if (x === p) { if (isOff(x)) { showEl(x); if (events) enter(x); } }
      else if (!isOff(x)) { if (events) x.dispatchEvent(new CustomEvent("walkthrough:leave", { bubbles: true })); hideEl(x); }
    });
    curPage = p;
    if (p !== intro) { seen.add(p.id); store.set("tc-learn-last", p.id); store.set("tc-learn-seen", [...seen].join(" ")); }
    if (p === intro) introContinue();
    if (cur === learn) {
      const j = stops.findIndex(s => s.el === p);
      if (j > 0) { const t = stops[j].a.querySelector(".nv-t"); if (t) t.classList.add("nv-seen"); }
      if (!quiet) say(p === intro ? "About this lesson." : "Step " + (steps.indexOf(p) + 1) + " of " + steps.length + ": " + titleOf(p) + ".");
    }
  }
  function footLink(cls, href, k, name) {                // k: "Previous", "Next", "Start", "Continue"
    const a = doc.createElement("a"); a.className = cls; a.href = href;
    const b = doc.createElement("b"); b.className = "ls-k";
    const ar = doc.createElement("span"); ar.className = "ls-ar"; ar.setAttribute("aria-hidden", "true");
    ar.textContent = cls === "ls-prev" ? "← " : " →";
    const w = doc.createElement("span"); w.className = "ls-w"; w.textContent = k;
    if (cls === "ls-prev") b.append(ar, w); else b.append(w, ar);
    const s = doc.createElement("span"); s.className = "ls-n"; s.textContent = name;
    a.append(b, s); return a;
  }
  function addFeet() {                                     // Back and Next at the end of the intro and of every step
    if (!learn) return;
    const k = units.indexOf(learn), prevU = units[k - 1], nextU = units[k + 1];
    pages.forEach((p, i) => {
      if (p.querySelector(":scope > .ls-foot")) return;
      const f = doc.createElement("nav"); f.className = "ls-foot"; f.setAttribute("aria-label", "Lesson steps");
      if (p === intro) {
        if (prevU) f.appendChild(footLink("ls-prev", "#" + keyOf(prevU), "Previous", "Previous section: " + nameOf(prevU)));
        if (steps[0]) f.appendChild(footLink("ls-next", "#" + steps[0].id, "Start", titleOf(steps[0])));
      } else {
        const pp = pages[i - 1], np = pages[i + 1];
        f.appendChild(pp ? footLink("ls-prev", pp === intro ? "#" + learn.id : "#" + pp.id, "Previous", titleOf(pp))
          : prevU ? footLink("ls-prev", "#" + keyOf(prevU), "Previous", "Previous section: " + nameOf(prevU)) : doc.createElement("span"));
        if (np) f.appendChild(footLink("ls-next", "#" + np.id, "Next", titleOf(np)));
        else if (nextU) f.appendChild(footLink("ls-next", "#" + keyOf(nextU), "Next", "Next section: " + nameOf(nextU)));
      }
      p.appendChild(f);
    });
  }
  function introContinue() {                               // the intro offers the step read last
    if (!intro) return;
    const f = intro.querySelector(":scope > .ls-foot"); if (!f) return;
    let c = f.querySelector(".ls-cont");
    const last = $(store.get("tc-learn-last") || "");
    if (last && steps.includes(last) && last !== steps[0]) {
      if (!c) { c = footLink("ls-cont", "#", "Continue", ""); f.insertBefore(c, f.firstChild); }
      c.href = "#" + last.id; c.querySelector(".ls-n").textContent = titleOf(last);
      const ns = f.querySelector(".ls-next .ls-w"); if (ns) ns.textContent = "Start from step 1";
    } else {
      if (c) c.remove();
      const ns = f.querySelector(".ls-next .ls-w"); if (ns) ns.textContent = "Start";
    }
  }
  function startBar() {                                    // the header's buttons: the lesson, and where the reader was
    const sl = $("start-learn"), sr = $("start-resume"), note = $("start-note");
    if (sl) {
      if (!learn) sl.remove();
      else {
        const last = $(store.get("tc-learn-last") || "");
        if (last && steps.includes(last)) { sl.href = "#" + last.id; sl.textContent = "Continue the lesson: " + titleOf(last); }
        else { sl.href = "#" + learn.id; sl.textContent = "Learn step by step"; }
      }
    }
    if (sr) {
      let last = null; try { last = JSON.parse(store.get("tc-last") || "null"); } catch (e) { last = null; }
      const el = last && last.id ? $(last.id) : null, u = el ? unitOf(el) : null;
      if (el && u && u !== units[0] && u !== learn) {
        sr.href = "#" + last.id;
        sr.textContent = "Continue reading: " + (last.u || nameOf(u)) + (last.l && last.l !== "Start of the section" ? " › " + trunc(last.l, 48) : "");
        sr.hidden = false;
      } else sr.hidden = true;
    }
    if (note) note.firstChild.nodeValue = one ? "The page shows one section at a time. Contents lists every section: it\n        sits at the left of a wide screen, and opens from the Contents button otherwise. \u201cIn this section\u201d (at the\n        right of a wide screen; otherwise the button with the section's name in the bar at the bottom) lists the parts of the section\n        on screen, and Previous and Next step through them in order."
      : "Whole page is on: every section is shown, one after another. Contents lists every section, and \u201cIn this section\u201d (at the right of a wide screen; otherwise the button with the section's name in the bar at the bottom) lists the parts of the section in view.";
  }
  const barSpace = () => (getComputedStyle(bar).display === "none" ? 0 : Math.max(0, W.innerHeight - bar.getBoundingClientRect().top));
  const readLine = () => Math.min(W.innerHeight * 0.3, 220);
  function holdAt(ms) {                                    // a jumped-to entry stays current for a while
    hold = { y: null, t: performance.now() + 400 + ms };
    clearTimeout(holdAt.t);
    holdAt.t = setTimeout(() => { if (hold) hold.y = W.scrollY; }, ms);
  }
  // a table scrolls inside its box (div.tbl, overflow: auto), which would clip its scroll margin and its arrival ring
  const boxOf = el => (el.matches("table") && el.parentElement && el.parentElement.matches(".tbl") ? el.parentElement : el);
  function scrollToEl(el, near) {
    el = boxOf(el);
    const r = el.getBoundingClientRect();
    const smooth = !calm.matches && near && Math.abs(r.top) < 1.5 * W.innerHeight;
    el.scrollIntoView({ block: "start", behavior: smooth ? "smooth" : "auto" });
    holdAt(smooth ? 900 : 0);
  }
  function flash(el) {
    if (!el || units.includes(el) || pages.includes(el)) return;
    el = boxOf(el);
    el.classList.remove("nv-flash"); void el.offsetWidth; el.classList.add("nv-flash");
    clearTimeout(flash.t); const t = el;
    flash.t = setTimeout(() => t.classList.remove("nv-flash"), calm.matches ? 2500 : 1900);
  }
  function focusOn(el) {                                   // keyboard and screen-reader users continue from the target
    if (!el) return;
    if (el.matches("canvas, svg, [data-tip], [data-tip-ref]") || el.closest("svg")) {
      el = el.parentElement && el.parentElement.closest(PANEL + ", .qv-chart, " + UNIT_SEL); if (!el) return;
      if (units.includes(el)) el = headOf(el);
    }
    if (el.matches("details")) el = el.querySelector(":scope > summary") || el;
    if (!el.matches("a[href], button, input, select, textarea, summary, [tabindex]")) { el.setAttribute("tabindex", "-1"); el.setAttribute("data-nv-focus", ""); }
    try { el.focus({ preventScroll: true }); } catch (e) { /* not focusable */ }
  }
  function openAround(el, self) {
    for (let d = el.parentElement && el.parentElement.closest("details"); d; d = d.parentElement && d.parentElement.closest("details")) d.open = true;
    if (self && el.matches("details") && !el.open) el.open = true;
  }
  function reveal(el) {                                    // an entry its panel shows only on a choice (a level's card)
    const sel = el.getAttribute && el.getAttribute("data-nav-reveal");
    if (!sel || el.getClientRects().length) return;
    const b = doc.querySelector(sel);
    if (b && !b.disabled && b.getAttribute("aria-pressed") !== "true") b.click();
  }
  let handled = { hash: null, t: 0, y: null }, selfHash = false;          // a fragment this script set or already followed
  const markHandled = y => { handled = { hash: location.hash, t: performance.now(), y: typeof y === "number" ? y : null }; };
  function setHash(id, push, oldY) {
    try {
      if (push) { history.replaceState(Object.assign({}, history.state, { nvY: oldY }), ""); history.pushState({ nv: 1 }, "", "#" + id); }
      else history.replaceState(Object.assign({}, history.state, { nv: 1 }), "", "#" + id);
    } catch (e) {                                          // a frame that refuses the history: the plain fragment
      try {                                                // (its own popstate is ignored; the browser's jump to it lands
        if (location.hash !== "#" + id) {                  // where go() did, except for the page's top, put back after it)
          selfHash = true; location.hash = id; selfHash = false; markHandled(one && id === keyOf(units[0]) ? 0 : null);
        }
      } catch (x) { selfHash = false; }
    }
  }
  // the one way to go anywhere: a link, Contents, an entry, Previous and Next, the address, Back and Forward, Find
  function go(el, { push = true, focus = true, from = "link", at = null } = {}) {
    const u = unitOf(el); if (!u) return false;
    const oldY = W.scrollY, oldU = cur;
    setUnit(u, from);
    if (u === learn && one) showPage(isHome(el, u) ? (intro || steps[0]) : pageOf(el), { quiet: from === "load" || u !== oldU });
    reveal(el);
    openAround(el, from === "link" || from === "side" || from === "nav" || from === "load" || from === "history");
    const home = isHome(el, u) && !(u === learn && !intro && el === steps[0]);
    let target = el;
    if (!el.getClientRects().length) { target = el.parentElement; while (target && target !== u && !target.getClientRects().length) target = target.parentElement; target = target || u; }
    if (home) { if (one && u.id === "top") { W.scrollTo(0, 0); holdAt(0); } else scrollToEl(u, false); }
    else if (u === learn && one && pages.includes(target)) scrollToEl(target, false);
    else scrollToEl(target, from === "side" || from === "step");
    let i = home ? 0 : stopIndexOf(el);
    if (at === "end") i = stops.length - 1;
    markStop(i);
    if (push !== null) setHash(home ? keyOf(u) : el.id || stops[i].id, push, oldY);
    if (focus) focusOn(home ? headOf(u) : (u === learn && one && pages.includes(el) ? el.querySelector("h3, h4") || el : el));
    if (!home && from !== "load" && !pages.includes(el)) flash(target);
    return true;
  }
  function move(d) {                                       // Previous and Next: the entries in order, then the next unit
    if (!cur || !running) return;
    const j = si + d;
    if (j >= 0 && j < stops.length) {
      const t = stops[j].kind === "top" || stops[j].kind === "step" ? stops[j].el : $(stops[j].id) || stops[j].el;   // its anchor
      go(t, { push: false, focus: false, from: "step" });
      say(stops[si].kind === "top" ? (cur === learn ? "About this lesson" : "Start of " + nameOf(cur)) : (cur === learn && stops[si].kind === "step" ? "Step " + (steps.indexOf(stops[si].el) + 1) + " of " + steps.length + ": " : "") + stops[si].label);
      return;
    }
    const k = units.indexOf(cur) + d;
    if (k < 0 || k >= units.length) return;
    const u = units[k];
    if (d > 0) go(u, { push: true, focus: false, from: "step" });
    else {
      setUnit(u, "step");
      const ls = stops[stops.length - 1];
      const last = u === learn && one ? pages[pages.length - 1] : ls.kind === "top" ? ls.el : $(ls.id) || ls.el;   // its anchor
      go(last || u, { push: true, focus: false, from: "step", at: u === learn ? null : "end" });
    }
  }

  // ---------- which entry is on screen ----------
  function spy() {
    spyRaf = 0;
    if (!cur || !running || printing) return;
    if (hold) {                                            // after a jump, the entry jumped to stays current until the
      if (hold.y === null) return;                         // reader has scrolled 48 px away and 400 ms have passed
      if (Math.abs(W.scrollY - hold.y) < 48 || performance.now() < hold.t) return;
      hold = null;
    }
    const line = readLine();
    if (!one) {                                            // the whole page: the unit in view leads
      const u = units.find(x => x.getBoundingClientRect().bottom > line) || units[units.length - 1];
      if (u !== cur) setUnit(u, "spy");
    }
    if (inLessonPages()) return;                           // the lesson's place is its page, not the scroll
    const top = k => { const e = stops[k].el; return e.getClientRects().length ? e.getBoundingClientRect().top : null; };
    let i = 0;
    for (let k = 1; k < stops.length; k++) { const t = top(k); if (t !== null && t <= line) i = k; }
    const de = doc.documentElement;
    if (W.innerHeight + W.scrollY >= de.scrollHeight - 4) {   // at the very end, the last entry on screen counts
      for (let k = stops.length - 1; k > i; k--) { const t = top(k); if (t !== null && t < W.innerHeight - barSpace() - 24) { i = k; break; } }
    }
    if (i !== si) markStop(i);
  }

  // ---------- drawers, the sheet and the remembered columns ----------
  const pageBehind = () => [doc.querySelector(".wrap"), bar, pill, $("nv-skip")].filter(Boolean);
  let navOpener = null, sideOpener = null;
  function setNavOpen(open, opener) {                      // below 1240 px: Contents as a drawer, the page behind inert
    if (mqWide.matches) open = false;
    html.classList.toggle("nv-nav-open", open);
    ["nv-open-nav", "nv-side-contents"].forEach(id => { const b = $(id); if (b) b.setAttribute("aria-expanded", String(open)); });
    pageBehind().forEach(el => { el.inert = open; });
    if (open) {
      if (html.classList.contains("nv-side-open")) setSideOpen(false);
      navOpener = opener || doc.activeElement;
      const a = cur && secLinks.get(keyOf(cur));
      (a || $("nv-close")).focus({ preventScroll: true });
      if (a) { const l = $("nv-list"), r = a.getBoundingClientRect(), lr = l.getBoundingClientRect(); l.scrollTop += r.top - lr.top - lr.height / 3; }
    } else if (navOpener && navOpener.isConnected && (nav.contains(doc.activeElement) || doc.activeElement === doc.body)) {
      navOpener.focus({ preventScroll: true });
    }
  }
  function setSideOpen(open, { toChoices = null, opener = null, peek = false } = {}) {   // below 1000 px
    if (mqRail.matches) open = false;
    if (!open || !peek) side.style.removeProperty("--nv-peek-h");
    html.classList.toggle("nv-side-open", open);
    html.classList.toggle("nv-peek", open && peek && !mqShort.matches);
    ["nv-where", "nv-pill"].forEach(id => { const b = $(id); if (b) b.setAttribute("aria-expanded", String(open)); });
    if (open) {
      sideOpener = opener || doc.activeElement;
      const s = toChoices || stops[si];
      requestAnimationFrame(() => {
        const t = s && s.a;                                // the entry, with its choices just below it
        if (t) stopsWrap.scrollTop += t.getBoundingClientRect().top - stopsWrap.getBoundingClientRect().top - 8;
        const f = toChoices && toChoices.ch ? (toChoices.ch.querySelector('.nv-b[aria-pressed="true"]') || toChoices.ch.querySelector(".nv-b")) : s && s.a;
        if (f) f.focus({ preventScroll: true });
        if (toChoices) afterChoice(groupOf(panelOf(toChoices)) ? groupOf(panelOf(toChoices)).view : null);
      });
    } else if (sideOpener && sideOpener.isConnected && (side.contains(doc.activeElement) || doc.activeElement === doc.body)) {
      sideOpener.focus({ preventScroll: true });
    }
  }
  // what the reader is looking at: the current entry if its top is on screen, else the block at the reading line;
  // after a column is hidden or docked the page column reflows, and the page is moved back to that block
  const BLOCKISH = "p, li, tr, dt, dd, h2, h3, h4, h5, figure, table, pre, .box, .chartbox, .ls-wl, .ls-blk, section, header";
  function placeAnchor() {
    const s = stops[si], e = s && s.el;
    if (e && e.getClientRects().length) { const t = e.getBoundingClientRect().top; if (t >= 0 && t < W.innerHeight) return { el: e, top: t }; }
    const mr = main.getBoundingClientRect(), x = Math.max(mr.left + 8, Math.min(mr.left + mr.width / 2, W.innerWidth - 8));
    const hit = doc.elementFromPoint(x, 24), el = hit && main.contains(hit) ? (hit.closest(BLOCKISH) || hit) : null;
    return el ? { el, top: el.getBoundingClientRect().top } : null;
  }
  function keepPlace(a) {
    if (!a) return;
    const fix = () => { if (a.el.isConnected && a.el.getClientRects().length) W.scrollBy(0, a.el.getBoundingClientRect().top - a.top); };
    requestAnimationFrame(() => { fix(); holdAt(0); setTimeout(() => { fix(); holdAt(0); }, 260); });   // again after the charts' redraw
  }
  function setWideClosed(which, closed, remember) {       // wide screens: hide or dock a column again; remembered
    const a = placeAnchor();
    html.classList.toggle(which === "nav" ? "nv-nav-closed" : "nv-side-closed", closed);
    if (remember) store.set(which === "nav" ? "tc-contents-wide" : "tc-side-wide", closed ? "0" : "1");
    syncButtons(); setMainW();
    keepPlace(a);
  }
  function syncButtons() {
    const navShown = mqWide.matches ? !html.classList.contains("nv-nav-closed") : html.classList.contains("nv-nav-open");
    const sideShown = mqRail.matches ? !html.classList.contains("nv-side-closed") : html.classList.contains("nv-side-open");
    ["nv-open-nav", "nv-side-contents"].forEach(id => { const b = $(id); if (b) b.setAttribute("aria-expanded", String(navShown)); });
    $("nv-where").setAttribute("aria-expanded", String(sideShown));
    $("nv-where-sr").textContent = mqRail.matches ? ": show In this section again" : ": list the parts of this section";
  }
  function applyLayout(measure) {
    const a = running ? placeAnchor() : null;              // a breakpoint that docks or undocks a column: the same
    if (mqWide.matches) { html.classList.remove("nv-nav-open"); pageBehind().forEach(el => { el.inert = false; }); }
    if (mqRail.matches) html.classList.remove("nv-side-open", "nv-peek");
    html.classList.toggle("nv-nav-closed", mqWide.matches && store.get("tc-contents-wide") === "0");
    html.classList.toggle("nv-side-closed", mqRail.matches && store.get("tc-side-wide") === "0");
    syncButtons(); if (measure !== false) setMainW();
    if (a) keepPlace(a);
  }
  let lastMainW = 0, lastIW = 0, roPrimed = false;       // measured after the first layout (the start reads no layout)
  function setMainW() {
    const w = Math.round(main.getBoundingClientRect().width);
    if (w > 0) html.style.setProperty("--nv-main-w", w + "px");
    return w;
  }
  function setWhole(on, remember) {                        // the whole page, or one unit at a time
    if (on === !one) return;
    const anchor = stops[si] ? stops[si].el : cur;
    const dy = anchor && anchor.getClientRects().length ? anchor.getBoundingClientRect().top : 0;
    one = !on; html.classList.toggle("nv-one", one);
    if (on) {
      units.forEach(u => { if (isOff(u)) { showEl(u); enter(u); } });
      pages.forEach(p => { if (isOff(p)) { showEl(p); enter(p); } });
    } else {
      units.forEach(u => { if (u !== cur) hideUnit(u); });
      if (cur === learn) showPage(curPage || pages[0], { quiet: true });
      else pages.forEach(p => { if (p !== (curPage || pages[0]) && !isOff(p)) { hideEl(p); } });
    }
    if (anchor && anchor.getClientRects().length) W.scrollBy(0, anchor.getBoundingClientRect().top - dy);
    holdAt(0);
    const sw = $("nv-whole"); sw.setAttribute("aria-checked", String(on)); sw.querySelector(".tt-state").textContent = on ? "on" : "off";
    $("nv-find-note").classList.toggle("nv-show", !UF && one);
    if (remember) store.set("tc-whole", on ? "1" : "0");
    const keep = si; stops = detect(cur); renderSide(); markContents(); markStop(Math.min(keep, stops.length - 1));
    if (cur && cur.id === "top") startBar();
    say(on ? "Whole page: every section in one page." : "One section at a time: " + nameOf(cur) + ".");
  }

  // ---------- start ----------
  function start() {
    html.classList.add("nv-js", "nv-on");
    html.classList.toggle("nv-one", one);
    try { history.scrollRestoration = "manual"; } catch (e) { /* not available */ }
    $("nv-find-note").classList.toggle("nv-show", !UF && one);
    const sw = $("nv-whole"); sw.setAttribute("aria-checked", String(!one)); sw.querySelector(".tt-state").textContent = one ? "off" : "on";
    addFeet();
    let h = location.hash.slice(1); try { h = decodeURIComponent(h); } catch (e) { /* keep it */ }
    const t0 = h ? $(h) : null, target = t0 && unitOf(t0) ? t0 : null;
    const first = target ? unitOf(target) : units[0];
    if (one) {
      units.forEach(u => { if (u !== first) { leave(u); hideEl(u); } });
      pages.forEach(p => hideEl(p));
    }
    applyLayout(false);
    html.classList.remove("nv-boot");                      // before go(): under nv-boot no section can be scrolled to
    if (target) {
      go(target, { push: null, focus: false, from: "load" });
      if (one && isHome(target, units[0])) {               // the browser's own jump to the fragment comes at load:
        const top = () => { if (cur === units[0] && Math.abs(target.getBoundingClientRect().top - 16) < 3) W.scrollTo(0, 0); };   // the page's top again
        if (doc.readyState === "complete") requestAnimationFrame(top); else W.addEventListener("load", () => requestAnimationFrame(top), { once: true });
      } else {                                             // that jump can move the page more than the hold allows (the
        const u0 = unitOf(target);                         // layout above the target changed since the start), and the
        const keep = () => {                               // scroll spy would then take an entry that sits just below it:
          if (cur === u0 && Math.abs(target.getBoundingClientRect().top - 16) < 3) { markStop(isHome(target, u0) ? 0 : stopIndexOf(target)); holdAt(0); }
        };                                                 // the entry of the address stays current
        if (doc.readyState === "complete") requestAnimationFrame(keep); else W.addEventListener("load", () => requestAnimationFrame(keep), { once: true });
      }
    }
    else setUnit(units[0], "load");                       // a fresh page is at its top already
    enter(cur);                                            // its drawings redraw at the width the navigation leaves
    if (learn && one && cur !== learn) { const p = $(store.get("tc-learn-last") || ""); curPage = p && steps.includes(p) ? p : null; }
    startBar();
    running = true;
    W.__nvInitMs = Math.round(performance.now() - T0);
  }
  function fail(err) {                                     // anything went wrong: the whole page, as without the script
    running = false;
    html.classList.remove(...STATE_CLASSES);
    units.concat(pages).forEach(el => { try { showEl(el); } catch (e) { /* keep going */ } });
    pageBehind().forEach(el => { el.inert = false; });
    requestAnimationFrame(() => W.dispatchEvent(new Event("resize")));   // the charts redraw at the plain layout's width
    if (W.console) console.warn("The navigation could not start; the whole page is shown.", err);
  }

  // ---------- wiring ----------
  doc.addEventListener("click", e => {                     // every in-page link goes through go()
    if (!running || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest && e.target.closest('a[href^="#"]'); if (!a) return;
    if (a.id === "nv-skip") {
      e.preventDefault(); const p = cur === learn && one && curPage ? (curPage.querySelector("h3, h4") || headOf(cur)) : headOf(cur);
      const r = p.getBoundingClientRect();                 // off screen (Tab came round from the end): the section's top comes back
      if (r.bottom < 0 || r.top > W.innerHeight - barSpace()) scrollToEl(cur === learn && one && curPage ? curPage : cur, false);
      focusOn(p); return;
    }
    let id = a.getAttribute("href").slice(1); try { id = decodeURIComponent(id); } catch (x) { /* keep it */ }
    const el = id && $(id); if (!el || !unitOf(el)) return;
    e.preventDefault();
    const fromSide = side.contains(a), fromNav = nav.contains(a), fromFoot = !!a.closest(".ls-foot");
    if (html.classList.contains("nv-nav-open")) setNavOpen(false);
    if (fromSide && html.classList.contains("nv-side-open")) setSideOpen(false);
    const inLesson = fromFoot && cur === learn && unitOf(el) === learn;
    go(el, { push: inLesson ? false : true, focus: true, from: fromSide ? "side" : fromNav ? "nav" : fromFoot ? "foot" : "link" });
  });
  doc.addEventListener("keydown", e => {                   // Left and Right on a lesson footer: back and forward
    if (!running || (e.key !== "ArrowLeft" && e.key !== "ArrowRight") || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    const a = e.target.closest && e.target.closest(".ls-foot a"); if (!a) return;
    const f = a.closest(".ls-foot"), want = e.key === "ArrowLeft" ? ".ls-prev" : ".ls-next", l = f.querySelector(want);
    if (!l || !l.getAttribute("href")) return;
    e.preventDefault();
    let id = l.getAttribute("href").slice(1); try { id = decodeURIComponent(id); } catch (x) { /* keep it */ }
    const el = $(id); if (!el) return;
    go(el, { push: unitOf(el) !== learn, focus: false, from: "foot" });
    const nf = cur === learn && one && curPage ? curPage.querySelector(":scope > .ls-foot " + want) : null;
    if (nf) nf.focus({ preventScroll: true }); else focusOn(headOf(cur));
  });
  W.addEventListener("popstate", e => {
    if (!running || selfHash) return;
    let h = location.hash.slice(1); try { h = decodeURIComponent(h); } catch (x) { /* keep it */ }
    const el = h ? $(h) : units[0];
    if (!el || !unitOf(el)) return;
    const y = e.state && typeof e.state.nvY === "number" ? e.state.nvY : one && isHome(el, units[0]) ? 0 : null;
    markHandled(y);                                        // the hashchange that may follow is the same move: after the
    go(el, { push: null, focus: false, from: "history" }); // browser's own jump to the fragment, it puts the place back
    if (y !== null) { W.scrollTo(0, y); hold = null; requestAnimationFrame(spy); }   // the entry on screen is marked
  });
  W.addEventListener("hashchange", () => {                // a fragment typed or set some other way
    if (!running) return;
    if (location.hash === handled.hash && performance.now() - handled.t < 1500) {
      if (handled.y !== null) {                            // the browser's own jump may come now or in the next frame
        const y = handled.y; W.scrollTo(0, y); holdAt(0);
        requestAnimationFrame(() => requestAnimationFrame(() => { if (Math.abs(W.scrollY - y) > 2 && location.hash === handled.hash) W.scrollTo(0, y); }));
      }
      handled = { hash: handled.hash, t: 0, y: null }; return;
    }
    let h = location.hash.slice(1); try { h = decodeURIComponent(h); } catch (x) { /* keep it */ }
    const el = h ? $(h) : null; if (!el || !unitOf(el)) return;
    go(el, { push: null, focus: false, from: "history" });
  });
  doc.addEventListener("beforematch", e => {              // Find (or a fragment) reached a hidden unit or lesson page
    if (!running || !one) return;
    const t = e.target;
    if (units.includes(t) && t !== cur) setUnit(t, "find");
    else if (pages.includes(t)) { if (cur !== learn) setUnit(learn, "find"); showPage(t, { quiet: true }); markStop(stopIndexOf(t)); }
    hold = null; requestAnimationFrame(() => requestAnimationFrame(spy));
  }, true);
  W.addEventListener("scroll", () => { if (running && !spyRaf) spyRaf = requestAnimationFrame(spy); if (running) queueAway(); }, { passive: true });
  W.addEventListener("resize", () => { if (running) queueAway(); });
  ["nv-prev", "nv-bar-prev"].forEach(id => $(id).addEventListener("click", () => move(-1)));
  ["nv-next", "nv-bar-next"].forEach(id => $(id).addEventListener("click", () => move(1)));
  $("nv-open-nav").addEventListener("click", e => {
    if (!running) return;
    if (mqWide.matches) { setWideClosed("nav", false, true); $("nv-hide").focus({ preventScroll: true }); }
    else setNavOpen(!html.classList.contains("nv-nav-open"), e.currentTarget);
  });
  $("nv-side-contents").addEventListener("click", e => {
    if (!running) return;
    if (mqWide.matches) { setWideClosed("nav", false, true); $("nv-hide").focus({ preventScroll: true }); }
    else setNavOpen(true, e.currentTarget);
  });
  $("nv-close").addEventListener("click", () => setNavOpen(false));
  $("nv-scrim").addEventListener("click", () => setNavOpen(false));
  $("nv-hide").addEventListener("click", () => {
    setWideClosed("nav", true, true);
    const b = mqRail.matches && !html.classList.contains("nv-side-closed") ? $("nv-side-contents") : $("nv-open-nav");
    requestAnimationFrame(() => b.focus({ preventScroll: true }));
  });
  $("nv-where").addEventListener("click", e => {
    if (!running) return;
    if (mqRail.matches) { setWideClosed("side", false, true); $("nv-side-hide").focus({ preventScroll: true }); }
    else setSideOpen(!html.classList.contains("nv-side-open"), { opener: e.currentTarget });
  });
  $("nv-side-close").addEventListener("click", () => setSideOpen(false));
  $("nv-side-all").addEventListener("click", () => { html.classList.remove("nv-peek"); side.style.removeProperty("--nv-peek-h"); });
  $("nv-side-hide").addEventListener("click", () => { setWideClosed("side", true, true); requestAnimationFrame(() => $("nv-where").focus({ preventScroll: true })); });
  if (pill) pill.addEventListener("click", e => {
    if (!running) return;
    const s = stops.find(x => x.ch && here.has(panelOf(x)));
    setSideOpen(true, { toChoices: s || null, opener: e.currentTarget, peek: true });
  });
  $("nv-whole").addEventListener("click", () => { if (running) setWhole(one, true); });
  doc.addEventListener("keydown", e => {
    if (!running || e.key !== "Escape") return;
    if (html.classList.contains("nv-nav-open")) { e.preventDefault(); setNavOpen(false); }
    else if (html.classList.contains("nv-side-open")) { e.preventDefault(); setSideOpen(false); }
  });
  [mqWide, mqRail, mqShort].forEach(q => { const f = () => { if (running) applyLayout(); }; if (q.addEventListener) q.addEventListener("change", f); else if (q.addListener) q.addListener(f); });
  // a change of the page column's width that is not a window resize (a column docked or hidden): send one
  if ("ResizeObserver" in W) new ResizeObserver(() => {
    if (!running) return;
    const w = setMainW();
    if (!roPrimed) { roPrimed = true; lastMainW = w; lastIW = W.innerWidth; return; }   // the first layout
    if (!w || w === lastMainW) return;
    lastMainW = w;
    if (W.innerWidth !== lastIW) { lastIW = W.innerWidth; return; }   // the browser's own resize already went out
    requestAnimationFrame(() => W.dispatchEvent(new Event("resize")));
  }).observe(main);
  W.addEventListener("resize", () => { if (running && W.innerWidth !== lastIW) { lastIW = W.innerWidth; lastMainW = setMainW(); } });
  // the choices stay in step with the panels, whichever side a choice was made on
  new MutationObserver(ms => { if (running && proxies.length && ms.some(m => !side.contains(m.target))) sync(); })
    .observe(main, { subtree: true, attributes: true, attributeFilter: ["aria-pressed", "disabled", "hidden"] });
  doc.addEventListener("change", e => { if (running && proxies.length && !side.contains(e.target)) requestAnimationFrame(sync); });
  doc.addEventListener("click", e => { if (running && !side.contains(e.target) && !nav.contains(e.target)) requestAnimationFrame(refreshChoices); });
  // printing: every unit and lesson page, every collapsed block open; put back afterwards
  let printed = null;
  W.addEventListener("beforeprint", () => {
    if (!running || printed) return;
    printed = { one, det: [], si, y: W.scrollY };
    printing = true;                                       // the print layout's scrolling changes nothing
    if (one) setWhole(true, false);
    // the units just shown get "walkthrough:enter" now, not at the next frame, which may come only after the print:
    // a lesson step builds its control when it is first shown, and the print should hold it. The collapsed blocks are
    // opened after that, so a control's own collapsed blocks are opened too.
    if (enterRaf) { cancelAnimationFrame(enterRaf); flushEnter(); }
    printed.det = [...main.querySelectorAll("details:not([open])")];
    printed.det.forEach(d => { d.open = true; });
  });
  W.addEventListener("afterprint", () => {
    if (!printed) return; const p = printed; printed = null;
    p.det.forEach(d => { d.open = false; });
    if (p.one) setWhole(false, false);
    W.scrollTo(0, p.y); markStop(p.si); holdAt(0);
    printing = false;
  });
  // two read-only hooks, so the same checks can run on the published page
  W.nvGo = id => { const el = $(id); return !!(running && el && go(el, { push: true, focus: false, from: "link" })); };
  W.nvState = () => ({ unit: cur && keyOf(cur), stop: si, stops: stops.length, label: stops[si] && stops[si].label, id: stops[si] && stops[si].id,
    one, page: curPage && (curPage === intro ? "intro" : curPage.id), uf: UF, running });
  try { start(); } catch (err) { fail(err); }
})();
</script>
