/* Full walkthrough mode: the page's sections one at a time in a pop-out, in reading order. The real nodes move into
   the pop-out and back (never copied), so ids, links, listeners, canvases and charts keep working. Each step's node
   gets a bubbling "walkthrough:enter" event when it is shown and "walkthrough:leave" when it goes back. */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const dlg = $("wt"), openBtn = $("wt-open");
  if (!dlg || !openBtn) return;
  if (typeof dlg.showModal !== "function") return;     // the button stays hidden where a pop-out cannot open
  openBtn.closest(".wt-bar").hidden = false;
  const bodyEl = $("wt-body"), stage = $("wt-stage"), titleEl = $("wt-title"), countEl = $("wt-count"),
    liveEl = $("wt-live"), barEl = $("wt-bar"), prevBtn = $("wt-prev"), nextBtn = $("wt-next"),
    nextName = $("wt-next-name"), closeBtn = $("wt-close");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const OUT_MS = 170;
  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), ' +
    'textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';
  let steps = [], cur = -1, resumeAt = 0, busy = false, savedY = 0;
  const home = new Map();                      // node -> the comment that marks its place in the page

  // the steps follow the page's own order: the glance, then every top-level section and the live panel
  function collect() {
    const list = [$("glance")];
    for (const n of document.querySelector("main").children)
      if (n.matches("section[id], figure.panel[id]")) list.push(n);
    return list.filter(Boolean).map(node => {
      const h = node.querySelector(":scope > h2, :scope > .panel-title");
      return { node, title: (node.dataset.wtTitle || (h ? h.textContent : node.id)).replace(/\s+/g, " ").trim() };
    });
  }
  function mount(i) {
    const n = steps[i].node, ph = document.createComment(" walkthrough: " + n.id + " ");
    n.before(ph); home.set(n, ph); n.classList.add("wt-in"); stage.appendChild(n);
    n.dispatchEvent(new CustomEvent("walkthrough:enter", { bubbles: true }));
  }
  function unmount(i) {
    const n = steps[i].node, ph = home.get(n);
    if (!ph) return;
    if (n.contains(document.activeElement)) titleEl.focus({ preventScroll: true });
    n.classList.remove("wt-in"); ph.replaceWith(n); home.delete(n);
    if (n.id === "sim" && typeof setPlaying === "function") setPlaying(false);   // the live panel never plays unseen
    n.dispatchEvent(new CustomEvent("walkthrough:leave", { bubbles: true }));
  }
  // table scroll cues and the weighting-test charts redraw on a window resize; the live panel redraws through render()
  function relayout(n) {
    window.dispatchEvent(new Event("resize"));
    if (n && (n.id === "sim" || n.querySelector("#sim")) && typeof render === "function") render();
  }
  function label(i) {
    const total = steps.length, end = i === total - 1;
    titleEl.textContent = steps[i].title;
    countEl.textContent = "Section " + (i + 1) + " of " + total;
    barEl.style.width = (100 * (i + 1) / total).toFixed(2) + "%";
    prevBtn.disabled = i === 0;
    nextBtn.textContent = end ? "Finish" : "Next →";
    nextName.textContent = end ? "End of the page" : "Next: " + steps[i + 1].title;
    liveEl.textContent = "Section " + (i + 1) + " of " + total + ": " + steps[i].title;
  }
  function show(i, dir, after) {
    if (i === cur) { if (after) after(); return; }
    if (busy || i < 0 || i >= steps.length) return;
    busy = true;
    const swap = () => {
      if (cur >= 0) unmount(cur);
      cur = resumeAt = i; mount(i); label(i);
      bodyEl.scrollTop = 0;
      if (dir && !reduce.matches) { stage.className = "wt-stage wt-enter-" + dir; void stage.offsetWidth; }
      stage.className = "wt-stage";
      const a = document.activeElement;
      if (!a || a === document.body || !dlg.contains(a) || a.disabled) (a && a.disabled ? nextBtn : titleEl).focus({ preventScroll: true });
      relayout(steps[i].node);
      busy = false;
      if (after) after();
    };
    if (cur < 0 || !dir || reduce.matches) swap();
    else { stage.className = "wt-stage wt-leave-" + dir; setTimeout(swap, OUT_MS); }
  }
  function open() {
    if (dlg.open) return;
    steps = collect();
    if (!steps.length) return;
    savedY = window.scrollY;
    document.documentElement.classList.add("wt-lock");
    dlg.showModal();
    cur = -1;
    show(Math.min(resumeAt, steps.length - 1), null);
    titleEl.focus({ preventScroll: true });
  }
  function close() {
    if (cur < 0 && !dlg.open) return;
    if (cur >= 0) { unmount(cur); cur = -1; }
    if (dlg.open) dlg.close();
    document.documentElement.classList.remove("wt-lock");
    window.scrollTo(0, savedY);
    relayout(null);
    openBtn.focus({ preventScroll: true });
  }
  function trap(e) {
    const f = [...dlg.querySelectorAll(FOCUSABLE)].filter(el => el.getClientRects().length);
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1], a = document.activeElement;
    if (e.shiftKey && (a === first || a === titleEl || !dlg.contains(a))) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && (a === last || !dlg.contains(a))) { e.preventDefault(); first.focus(); }
  }
  const SCROLL_KEYS = { ArrowDown: 1, ArrowUp: -1, PageDown: 2, PageUp: -2, Home: -9, End: 9 };

  openBtn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  prevBtn.addEventListener("click", () => show(cur - 1, "back"));
  nextBtn.addEventListener("click", () => {
    if (cur === steps.length - 1) { resumeAt = 0; close(); } else show(cur + 1, "next");
  });
  dlg.addEventListener("cancel", e => { e.preventDefault(); close(); });   // Esc, or a phone's back gesture
  dlg.addEventListener("close", () => { if (cur >= 0) close(); });         // any other way the dialog closed
  document.addEventListener("keydown", e => {
    if (!dlg.open) return;
    if (e.key === "Tab") { trap(e); return; }
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.defaultPrevented) return;
    const t = e.target;
    if (t && t.closest && t.closest("input, select, textarea, [contenteditable]:not([contenteditable='false'])")) return;
    if (e.key === "ArrowRight" && cur < steps.length - 1) { e.preventDefault(); show(cur + 1, "next"); }
    else if (e.key === "ArrowLeft" && cur > 0) { e.preventDefault(); show(cur - 1, "back"); }
    else if (SCROLL_KEYS[e.key] && (t === titleEl || t === dlg || t === document.body)) {
      // the title holds focus after each move: let the usual scrolling keys scroll the section's text
      e.preventDefault();
      const k = SCROLL_KEYS[e.key], h = bodyEl.clientHeight;
      if (Math.abs(k) === 9) bodyEl.scrollTop = k < 0 ? 0 : bodyEl.scrollHeight;
      else bodyEl.scrollBy({ top: Math.abs(k) === 2 ? Math.sign(k) * h * 0.9 : k * 48 });
    }
  });
  // in-page links inside the pop-out open the step that holds their target instead of scrolling the page behind
  stage.addEventListener("click", e => {
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    const id = decodeURIComponent(a.getAttribute("href").slice(1));
    const target = id ? document.getElementById(id) : null;
    if (!target) return;
    e.preventDefault();
    const i = steps.findIndex(s => s.node === target || s.node.contains(target));
    if (i < 0) { close(); target.scrollIntoView(); return; }
    const reveal = () => {
      const d = target.closest("details");
      if (d && !d.open) d.open = true;
      if (target !== steps[i].node) target.scrollIntoView({ block: "start" });
      titleEl.focus({ preventScroll: true });
    };
    show(i, i > cur ? "next" : "back", reveal);
  });
  window.addEventListener("hashchange", () => { if (dlg.open) close(); });
})();
