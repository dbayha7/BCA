<script>
/* Tooltips: one shared tooltip for the whole page. It shows on hover, on keyboard focus and on a tap, is placed to stay
   on screen, and closes with Esc, a tap elsewhere or the "Tooltips" switch at the top (on by default; the choice is
   remembered in this browser when storage allows). It carries no controls and lets every pointer event through.
   Two sources of text:
   - any element with data-tip="plain text" (optional data-tip-title="a heading"; data-tip-ref="id of a <template>" for
     formatted content; data-tip-focus="no" keeps a non-focusable element out of the Tab order). Such an element is
     made focusable and gets the text as its accessible description, whether or not tooltips show;
   - every data table, built here from the table itself: each column name gets its description (COLS: by section, then
     by column name), and each cell shows its whole row as a tuple (name: value, ...) with its own value in bold. A table
     is one Tab stop; the arrow keys, Home and End move between its cells. The descriptions also sit in the column
     headers as visually hidden text for screen readers. */
(function () {
  "use strict";
  const COLS = __COLS__;
  const KEY = "thrusterchain-tooltips";
  const root = document.documentElement;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* ---------- plain text of a cell or header: math as symbols, hidden helper text left out ---------- */
  const GREEK = { alpha: "α", beta: "β", gamma: "γ", lambda: "λ", mu: "μ", kappa: "κ", omega: "ω", sigma: "σ",
    Delta: "Δ", delta: "δ", pi: "π", phi: "φ", theta: "θ", rho: "ρ", eta: "η", nu: "ν", epsilon: "ε" };
  const SUP = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹", "+": "⁺", "−": "⁻",
    "-": "⁻", t: "ᵗ", n: "ⁿ", i: "ⁱ" };
  const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };
  const mapAll = (s, m) => [...s].every(ch => m[ch]) ? [...s].map(ch => m[ch]).join("") : null;
  const pow = s => mapAll(s, SUP) || (s.length > 1 ? "^(" + s + ")" : "^" + s);
  const tex = s => s.replace(/\\text\{([^}]*)\}/g, "$1")
    .replace(/\\(alpha|beta|gamma|lambda|mu|kappa|omega|sigma|Delta|delta|pi|phi|theta|rho|eta|nu|epsilon)/g, (m, g) => GREEK[g])
    .replace(/\\times/g, "×").replace(/\\cdot/g, "·").replace(/\\[,;: ]/g, " ")
    .replace(/\^\{?([\w+−-]+)\}?/g, (m, g) => pow(g))
    .replace(/_\{?(\w+)\}?/g, (m, g) => mapAll(g, SUB) || "_" + g)
    .replace(/[{}]/g, "").replace(/\s+/g, " ").trim();
  function plain(n) {
    let s = "";
    for (const c of n.childNodes) {
      if (c.nodeType === 3) { s += c.nodeValue; continue; }
      if (c.nodeType !== 1 || c.hidden || (c.classList && c.classList.contains("tt-sr"))) continue;
      const tag = c.localName;
      if (tag === "mjx-container") s += " " + tex(c.getAttribute("aria-label") || "") + " ";
      else if (tag === "br") s += " ";
      else if (tag === "sup") s += c.getAttribute("aria-label") ? " (" + c.getAttribute("aria-label") + ")" : pow(plain(c).trim());
      else if (tag === "sub") s += mapAll(plain(c).trim(), SUB) || "_" + plain(c).trim();
      else if (c.classList.contains("rng")) s += " (" + plain(c).trim() + ")";
      else if (/^(div|p|pre|ul|ol|li|dl|dt|dd)$/.test(tag) || c.classList.contains("note")) s += " " + plain(c) + " ";
      else s += plain(c);
    }
    return s;
  }
  const clean = s => s.replace(/\s+/g, " ").trim();
  const cut = (s, n) => s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s;

  /* ---------- tables ---------- */
  const TBL = new WeakMap();          // table -> its column names, descriptions and cell grid
  const used = new Set();             // COLS entries already given to a table
  const tables = [];
  function grid(rows) {               // cells by row and column, spans resolved
    const g = rows.map(() => []);
    rows.forEach((tr, r) => {
      let c = 0;
      for (const cell of tr.cells) {
        while (g[r][c]) c++;
        const cs = Math.max(1, cell.colSpan || 1), rs = Math.max(1, cell.rowSpan || 1);
        for (let i = 0; i < rs && r + i < rows.length; i++) for (let j = 0; j < cs; j++) g[r + i][c + j] = cell;
        c += cs;
      }
    });
    return g;
  }
  function entryFor(t, names) {
    const sec = t.closest("section[id]"), list = sec && COLS[sec.id];
    if (!list) return null;
    for (const e of list) {
      if (used.has(e)) continue;
      if (names.every(n => Object.prototype.hasOwnProperty.call(e.cols, n))) { used.add(e); return e; }
    }
    return null;
  }
  const descOfEntry = (e, name) => {
    const v = e && Object.prototype.hasOwnProperty.call(e.cols, name) ? e.cols[name] : null;
    return v == null ? null : (Array.isArray(v) ? { name: v[0], desc: v[1] } : { name, desc: v });
  };
  function addSr(cell, desc) {
    const sr = document.createElement("span");
    sr.className = "tt-sr"; sr.textContent = ". " + desc;
    cell.appendChild(sr);
  }
  function prepTable(t) {
    if (TBL.has(t) || !t.tBodies.length) return;
    const head = t.tHead ? [...t.tHead.rows] : [];
    const body = [...t.tBodies].flatMap(b => [...b.rows]);
    if (!body.length) return;
    const info = { t, kv: !head.length, names: [], descs: [], heads: new Map(), group: "Group", groupOf: new Map(),
      bodyGrid: grid(body), rowIdx: new Map(body.map((tr, i) => [tr, i])), stop: null, cells: [] };
    let entry = null;
    if (info.kv) {                                    // a key-value table: its row headers name the rows
      const names = body.map(tr => tr.querySelector("th") ? clean(plain(tr.querySelector("th"))) : "");
      entry = entryFor(t, names.filter(Boolean));
      body.forEach((tr, i) => {
        const th = tr.querySelector("th");
        if (!th) return;
        const d = descOfEntry(entry, names[i]);
        info.heads.set(th, { name: names[i], desc: d ? d.desc : "" });
        th.dataset.tt = "h";
        if (d) { th.dataset.ttDesc = ""; addSr(th, d.desc); }
      });
    } else {
      const hg = grid(head), ncol = Math.max(...hg.map(r => r.length), ...info.bodyGrid.map(r => r.length));
      for (let c = 0; c < ncol; c++) {
        const last = hg[hg.length - 1][c];
        if (last && last.dataset.label) { info.names.push(clean(last.dataset.label)); continue; }
        const parts = [];
        for (const row of hg) { const s = row[c] ? clean(plain(row[c])) : ""; if (s && !parts.includes(s)) parts.push(s); }
        info.names.push(parts.join(": "));
      }
      entry = entryFor(t, info.names);
      info.names = info.names.map((n, c) => {
        const d = descOfEntry(entry, n);
        info.descs[c] = d ? d.desc : "";
        return d ? d.name : (n || "Column " + (c + 1));
      });
      if (entry && entry.group) info.group = entry.group;
      // header cells: a column's own header, or a group header spanning several columns
      const seen = new Set();
      hg.forEach(row => row.forEach((th, c) => {
        if (!th || seen.has(th)) return;
        seen.add(th);
        const lastRow = hg[hg.length - 1][c] === th;
        const own = clean(plain(th));
        let name = lastRow ? info.names[c] : own, desc = lastRow ? info.descs[c] : "";
        if (!lastRow) { const d = descOfEntry(entry, own); desc = d ? d.desc : ""; }
        if (!own && !desc) return;
        info.heads.set(th, { name, desc });
        th.dataset.tt = "h";
        if (desc) { th.dataset.ttDesc = ""; addSr(th, desc); }
      }));
    }
    // heading rows inside the body (a single filled first cell) group the rows under them
    const ncolB = Math.max(...info.bodyGrid.map(r => r.length));
    let grp = "";
    body.forEach(tr => {
      const filled = [...tr.cells].filter(c => clean(plain(c)));
      const isGroup = !info.kv && filled.length === 1 && filled[0] === tr.cells[0] &&
        (tr.classList.contains("sub") || (tr.cells.length === 1 && tr.cells[0].colSpan >= ncolB));
      if (isGroup) { grp = clean(plain(tr.cells[0])); tr.dataset.ttGroup = ""; return; }
      if (grp) info.groupOf.set(tr, grp);
      for (const cell of tr.cells) {
        if (cell.dataset.tt === "h" || !clean(plain(cell))) continue;
        cell.dataset.tt = "c";
      }
    });
    info.cells = [...t.querySelectorAll("[data-tt]")].filter(c => c.closest("table") === t);
    if (!info.cells.length) return;
    info.cells.forEach(c => c.setAttribute("tabindex", "-1"));
    info.stop = info.cells[0];
    info.full = grid([...t.rows]);
    TBL.set(t, info);
    tables.push(info);
  }
  function tableOf(cell) { const t = cell.closest("table"); return t ? TBL.get(t) : null; }
  function setStop(cell) {
    const info = tableOf(cell);
    if (!info || info.stop === cell) return;
    if (info.stop) info.stop.setAttribute("tabindex", "-1");
    info.stop = cell;
    cell.setAttribute("tabindex", isOn() ? "0" : "-1");
  }
  // a table's Tab stop must be a cell on screen: at phone width the card layout hides the header row
  function fixStops() {
    tables.forEach(info => {
      if (info.stop && info.stop.getClientRects().length) return;
      const v = info.cells.find(c => c.getClientRects().length);
      if (v) setStop(v);
    });
  }
  function moveFrom(cell, key, ctrl) {
    const info = tableOf(cell);
    if (!info) return null;
    const G = info.full, ok = c => c && c !== cell && c.dataset.tt && c.getClientRects().length;
    let r = -1, c = -1;
    for (let i = 0; i < G.length && r < 0; i++) { const j = G[i].indexOf(cell); if (j >= 0) { r = i; c = j; } }
    if (r < 0) return null;
    const scan = (dr, dc) => { for (let i = r + dr, j = c + dc; i >= 0 && i < G.length && j >= 0 && j < (G[i] || []).length; i += dr, j += dc) if (ok(G[i][j])) return G[i][j]; return null; };
    if (key === "ArrowRight") return scan(0, 1);
    if (key === "ArrowLeft") return scan(0, -1);
    if (key === "ArrowDown") return scan(1, 0);
    if (key === "ArrowUp") return scan(-1, 0);
    const vis = info.cells.filter(x => x.getClientRects().length);
    if (ctrl) return key === "Home" ? vis[0] : vis[vis.length - 1];
    const row = G[r].filter((x, k, a) => x && x.dataset.tt && a.indexOf(x) === k && x.getClientRects().length);
    return key === "Home" ? row[0] : row[row.length - 1];
  }

  /* ---------- elements with data-tip ---------- */
  const descBox = document.createElement("div");      // their descriptions, for assistive technology
  descBox.id = "tt-descs"; descBox.hidden = true;
  const descOf = new WeakMap(), autoTab = new Set();
  let seq = 0;
  const FOCUSABLE = "a[href], button, input, select, textarea, summary, iframe, [tabindex], [contenteditable]";
  function tipText(el) {
    const ref = el.dataset.tipRef && document.getElementById(el.dataset.tipRef);
    const body = ref ? clean(ref.content ? ref.content.textContent : ref.textContent) : (el.dataset.tip || "");
    return [el.dataset.tipTitle || "", body].filter(Boolean).join(". ");
  }
  function prepTip(el) {
    if (!el.dataset || (el.dataset.tip == null && !el.dataset.tipRef) || tip.contains(el) || descBox.contains(el)) return;
    let d = descOf.get(el);
    if (!d) {
      if (!el.matches(FOCUSABLE) && el.dataset.tipFocus !== "no") { el.setAttribute("tabindex", isOn() ? "0" : "-1"); autoTab.add(el); }
      d = document.createElement("span"); d.id = "tt-d" + (++seq);
      descBox.appendChild(d); descOf.set(el, d);
      const by = (el.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean);
      by.push(d.id); el.setAttribute("aria-describedby", by.join(" "));
    }
    d.textContent = tipText(el);
  }

  /* ---------- the tooltip ---------- */
  const tip = document.createElement("div");
  tip.className = "tt"; tip.id = "tt"; tip.hidden = true;
  tip.setAttribute("aria-hidden", "true");            // its text is already reachable: hidden header text, descriptions
  let cur = null, how = "", pt = null, tShow = 0, tHide = 0, lastType = "mouse", escHid = false;
  const isOn = () => !root.classList.contains("tt-off");
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const anchorOf = n => (n && n.nodeType === 1 && n.closest) ? n.closest("[data-tip], [data-tip-ref], [data-tt]") : null;

  function build(a) {
    const out = [];
    if (a.dataset.tt === "h") {
      const info = tableOf(a), h = info && info.heads.get(a);
      if (!h || !h.desc) return null;
      out.push(el("div", "tt-h", h.name), el("div", "tt-b", h.desc));
      return out;
    }
    if (a.dataset.tt === "c") {
      const info = tableOf(a);
      if (!info) return null;
      const tr = a.parentElement, r = info.rowIdx.get(tr);
      const box = el("div", "tt-tuple");
      const pairs = [];
      let colName = "", colDesc = "";
      if (info.kv) {
        const th = tr.querySelector("th"), h = th && info.heads.get(th);
        pairs.push([h ? h.name : "", clean(plain(a)), true]);
        if (h) { colName = h.name; colDesc = h.desc; }
      } else {
        if (info.groupOf.get(tr)) pairs.push([info.group, info.groupOf.get(tr), false]);
        const seen = new Set();
        (info.bodyGrid[r] || []).forEach((cell, c) => {
          if (!cell || seen.has(cell)) return;
          seen.add(cell);
          const v = clean(plain(cell));
          if (!v) return;
          pairs.push([info.names[c], v, cell === a]);
          if (cell === a) { colName = info.names[c]; colDesc = info.descs[c]; }
        });
      }
      box.append("(");
      pairs.forEach(([k, v, isCur], i) => {
        if (i) box.append(", ");
        const p = el("span", isCur ? "p cur" : "p");
        if (k) p.append(el("span", "k", k + ": "));
        if (isCur) p.append(el("b", null, cut(v, 240))); else p.append(cut(v, 64));
        box.append(p);
      });
      box.append(")");
      out.push(box);
      if (colDesc) { const d = el("div", "tt-col"); d.append(el("b", null, colName + ": "), colDesc); out.push(d); }
      return out;
    }
    const ref = a.dataset.tipRef && document.getElementById(a.dataset.tipRef);
    if (a.dataset.tipTitle) out.push(el("div", "tt-h", a.dataset.tipTitle));
    if (ref) {
      const frag = ref.content ? ref.content.cloneNode(true) : document.createRange().createContextualFragment(ref.innerHTML);
      frag.querySelectorAll("[id]").forEach(x => x.removeAttribute("id"));
      const b = el("div", "tt-b tt-rich"); b.appendChild(frag); out.push(b);
    } else if (a.dataset.tip) out.push(el("div", "tt-b", a.dataset.tip));
    return out.length ? out : null;
  }
  function host() {
    const d = document.querySelector("dialog[open]");          // a modal pop-out sits in the top layer: go inside it
    return d || document.body;
  }
  function show(a, mode, p) {
    clearTimeout(tShow); clearTimeout(tHide);
    if (!isOn() || !a || !a.isConnected) return;
    const nodes = build(a);
    if (!nodes) { if (cur) hide(); return; }
    const h = host();
    if (tip.parentNode !== h) h.appendChild(tip);
    if (cur && cur !== a) cur.classList.remove("tt-act");
    cur = a; how = mode; pt = p || null;
    tip.replaceChildren(...nodes);
    tip.hidden = false;
    if (a.dataset.tt) a.classList.add("tt-act");
    place();
    if (!cur) return;
    if (reduce.matches) tip.classList.add("on");
    else { tip.classList.remove("on"); void tip.offsetWidth; tip.classList.add("on"); }
  }
  function hide() {
    clearTimeout(tShow); clearTimeout(tHide);
    if (cur) cur.classList.remove("tt-act");
    cur = null; how = ""; pt = null;
    tip.hidden = true; tip.classList.remove("on");
  }
  function place() {
    if (!cur) return;
    const vw = document.documentElement.clientWidth, vh = window.innerHeight, M = 8, G = 8;
    const r = cur.getBoundingClientRect();
    if (!cur.isConnected || (!r.width && !r.height) || r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) { hide(); return; }
    let a = { l: r.left, r: r.right, t: r.top, b: r.bottom };
    if (pt && (r.height > 140 || r.width > 460)) a = { l: pt.x, r: pt.x, t: pt.y - 12, b: pt.y + 18 };   // a big target: follow the pointer
    a.t = Math.max(a.t, 0); a.b = Math.min(a.b, vh);
    tip.style.left = "0px"; tip.style.top = "0px";
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = (Math.max(a.l, 0) + Math.min(a.r, vw)) / 2 - w / 2;
    x = Math.max(M, Math.min(x, vw - w - M));
    let y = a.t - G - h;                                         // above the target when it fits, else below
    if (y < M && (a.b + G + h <= vh - M || vh - a.b > a.t)) y = a.b + G;
    y = Math.max(M, Math.min(y, vh - h - M));
    tip.style.left = Math.round(x) + "px"; tip.style.top = Math.round(y) + "px";
  }

  /* ---------- the switch ---------- */
  const sw = document.getElementById("tt-switch");
  function setOn(on, save) {
    root.classList.toggle("tt-off", !on);
    if (sw) {
      sw.setAttribute("aria-checked", String(on));
      const s = sw.querySelector(".tt-state"); if (s) s.textContent = on ? "on" : "off";
    }
    if (!on) hide();
    tables.forEach(info => info.cells.forEach(c => c.setAttribute("tabindex", on && c === info.stop ? "0" : "-1")));
    autoTab.forEach(e => e.setAttribute("tabindex", on ? "0" : "-1"));
    if (save) { try { window.localStorage.setItem(KEY, on ? "on" : "off"); } catch (err) { /* storage unavailable: the choice lasts this visit */ } }
  }

  /* ---------- start ---------- */
  document.body.append(descBox, tip);
  document.querySelectorAll("table").forEach(t => { try { prepTable(t); } catch (err) { /* a table it cannot read keeps working without tooltips */ } });
  document.querySelectorAll("[data-tip], [data-tip-ref]").forEach(prepTip);
  let stored = null;
  try { stored = window.localStorage.getItem(KEY); } catch (err) { stored = null; }
  setOn(stored !== "off", false);
  fixStops();
  if (sw) {
    const bar = sw.closest(".wt-bar");
    if (bar && bar.hidden) {                         // no pop-out in this browser: show the switch without its neighbour
      bar.hidden = false;
      bar.querySelectorAll(".wt-open, .wt-note").forEach(n => { n.style.display = "none"; });
    }
    sw.addEventListener("click", () => setOn(!isOn(), true));
  }

  /* ---------- events ---------- */
  document.addEventListener("pointerdown", e => {
    lastType = e.pointerType || "mouse";
    if (cur && !cur.contains(e.target)) hide();      // a tap or click elsewhere closes it
  }, true);
  document.addEventListener("pointerover", e => {
    if (e.pointerType === "touch" || !isOn()) return;
    const a = anchorOf(e.target);
    if (!a) return;
    if (a === cur) { clearTimeout(tHide); return; }
    clearTimeout(tShow);
    const p = { x: e.clientX, y: e.clientY };
    if (cur && how !== "tap") show(a, "hover", p);   // moving from one target to the next: no delay
    else tShow = setTimeout(() => show(a, "hover", p), 220);
  });
  document.addEventListener("pointerout", e => {
    if (e.pointerType === "touch") return;
    const a = anchorOf(e.target);
    if (!a || (e.relatedTarget && a.contains(e.relatedTarget))) return;
    clearTimeout(tShow);
    if (cur === a && how === "hover") tHide = setTimeout(hide, 120);
  });
  document.addEventListener("click", e => {          // a tap shows it; a second tap on the same target closes it
    if (lastType !== "touch" && lastType !== "pen") return;
    const a = anchorOf(e.target);
    if (!a || !isOn()) return;
    if (cur === a && how === "tap") { hide(); return; }
    show(a, "tap", { x: e.clientX, y: e.clientY });
  });
  document.addEventListener("focusin", e => {
    const t = e.target;
    if (t.dataset && t.dataset.tt) setStop(t);
    if (anchorOf(t) !== t) return;
    let vis = true;
    try { vis = t.matches(":focus-visible"); } catch (err) { vis = true; }
    if (vis) show(t, "focus");
  });
  document.addEventListener("focusout", e => { if (cur === e.target && how === "focus") hide(); });
  // Esc closes the tooltip first (and only the tooltip, inside the walkthrough pop-out too)
  window.addEventListener("keydown", e => {
    if (e.key !== "Escape" || !cur) return;
    hide(); escHid = true; setTimeout(() => { escHid = false; }, 0);
    e.preventDefault(); e.stopImmediatePropagation();
  }, true);
  window.addEventListener("cancel", e => { if (escHid) { e.preventDefault(); e.stopImmediatePropagation(); } }, true);
  // inside a table: the arrow keys, Home and End move between cells (before the walkthrough reads the arrows)
  document.addEventListener("keydown", e => {
    const t = e.target;
    if (!t || !t.dataset || !t.dataset.tt || e.altKey || e.metaKey || e.shiftKey) return;
    if (!/^(ArrowRight|ArrowLeft|ArrowDown|ArrowUp|Home|End)$/.test(e.key)) return;
    e.preventDefault();
    const n = moveFrom(t, e.key, e.ctrlKey);
    if (n) n.focus();
  }, true);
  document.addEventListener("scroll", () => { if (cur) requestAnimationFrame(place); }, { capture: true, passive: true });
  let rz = 0;
  window.addEventListener("resize", () => { hide(); cancelAnimationFrame(rz); rz = requestAnimationFrame(fixStops); });
  document.addEventListener("toggle", () => requestAnimationFrame(fixStops), true);   // a <details> opened or closed
  document.addEventListener("walkthrough:enter", hide);
  document.addEventListener("walkthrough:leave", hide);
  const dlg = document.getElementById("wt");
  if (dlg) dlg.addEventListener("close", hide);
  // elements given data-tip later (or whose text changes) are prepared as they appear
  new MutationObserver(ms => {
    for (const m of ms) {
      if (m.type === "attributes") { prepTip(m.target); if (m.target === cur) show(cur, how, pt); continue; }
      for (const n of m.addedNodes) {
        if (n.nodeType !== 1 || n === tip || n === descBox || tip.contains(n)) continue;
        if (n.matches("[data-tip], [data-tip-ref]")) prepTip(n);
        n.querySelectorAll("[data-tip], [data-tip-ref]").forEach(prepTip);
      }
    }
    if (cur && !cur.isConnected) hide();
  }).observe(document.body, { subtree: true, childList: true, attributes: true,
    attributeFilter: ["data-tip", "data-tip-title", "data-tip-ref"] });
  window.ttRefresh = () => document.querySelectorAll("[data-tip], [data-tip-ref]").forEach(prepTip);
})();
</script>
