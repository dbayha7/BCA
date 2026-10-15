/* Page pass 4, edit 2: two interactive figures.
   - Action Space, "Try a command" (#as-widget): four throttle sliders. Each shows the push its thruster really gives in
     both worlds and its throttle cost, and a strip under it shows where that thruster's recorded commands lie in the
     main test's or the weighting test's training data, before and after the removal. A box says what each part of the
     learning process does at the command set.
   - Summary of the environment, "Why the stall makes this a test of distribution shift" (#stall-chart): push and
     throttle cost against the size of the throttle in both worlds, one axis; under it, on its own axis, how often
     thruster 1's recorded throttle has each size, in the full data and after the main test's removal.
   Both draw with the theme tokens (CSS classes on the SVG parts), redraw to their own width, and work with the keyboard
   (native sliders, buttons, checkboxes and radio buttons; on a slider the arrow keys move 1%, Page Up and Page Down
   10%, Home and End to the ends). Hover or tap on a drawing shows the numbers under the pointer, and a tap sets the
   slider there. A table under each figure holds the same numbers. Nothing moves on its own. The data are the facts of
   page pass 4 (notes/p4_facts.json and tmp/p4/edit2/netvalue.json), inlined below. */
(function () {
  "use strict";
  const DATA = __P4E2_DATA__;
  const NS = "http://www.w3.org/2000/svg", MINUS = "−", KNEE = DATA.knee;
  const $ = id => document.getElementById(id);
  const fmt = (x, d) => { const s = Math.abs(x).toFixed(d); return (x < 0 && Number(s) !== 0 ? MINUS : "") + s; };
  const int = n => Math.round(n).toLocaleString("en-US");
  const pushOf = (a, stall) => { const m = Math.abs(a); return (!stall || m <= KNEE) ? a : Math.sign(a) * (0.8 - 1.5 * (m - 0.8)); };
  const costOf = a => 0.3 * a * a;
  const pc = m => Math.round(m * 100) + "%";
  function mk(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, typeof attrs[k] === "number" ? String(Math.round(attrs[k] * 10) / 10) : attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function hatch(defs, id, base) {      // data removed: a tile hatched at 45 degrees over its own fill
    const p = mk("pattern", { id, patternUnits: "userSpaceOnUse", width: 5, height: 5, patternTransform: "rotate(45)" }, defs);
    mk("rect", { width: 5, height: 5, class: base }, p);
    mk("line", { x1: 1, y1: 0, x2: 1, y2: 5, class: "hx-l" }, p);
  }
  function span(parent, cls, text) {
    const e = document.createElement("span");
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    parent.appendChild(e);
    return e;
  }
  function keys(inp, apply) {          // the slider's keys, the same in every browser
    inp.addEventListener("keydown", e => {
      const step = { ArrowRight: 0.01, ArrowUp: 0.01, ArrowLeft: -0.01, ArrowDown: -0.01, PageUp: 0.1, PageDown: -0.1 }[e.key];
      let v = Number(inp.value);
      if (step != null) v = Math.round((v + step) * 100) / 100;
      else if (e.key === "Home") v = Number(inp.min);
      else if (e.key === "End") v = Number(inp.max);
      else return;
      e.preventDefault();
      v = Math.min(Number(inp.max), Math.max(Number(inp.min), v));
      inp.value = String(v);
      apply(v);
    });
  }
  function watch(el, fn) {             // redraw to the element's own width (the walkthrough moves sections around)
    let raf = 0, w = -1;
    new ResizeObserver(() => {
      const nw = Math.round(el.getBoundingClientRect().width);
      if (nw === w) return;
      w = nw; cancelAnimationFrame(raf); raf = requestAnimationFrame(fn);
    }).observe(el);
  }

  /* ================================ the four-throttle widget ================================ */
  (function () {
    const fig = $("as-widget");
    if (!fig || !window.ResizeObserver) return;
    const T = [1, 2, 3, 4].map(k => ({ k, inp: $("aw-a" + k), out: $("aw-o" + k), svg: $("aw-s" + k), bin: $("aw-b" + k),
      box: fig.querySelector('.aw-thr[data-k="' + k + '"]') }));
    const ui = { cost: $("aw-cost"), ball: $("aw-ball"), msg: $("aw-msg"), live: $("aw-live"), keyRm: $("aw-key-rm"),
      keyRmT: $("aw-key-rm-t"), tnote: $("aw-tnote") };
    if (T.some(t => !t.inp || !t.out || !t.svg || !t.bin || !t.box) || Object.values(ui).some(x => !x)) return;
    const E = DATA.signedEdges, A = [0, 0, 0, 0], hover = [-1, -1, -1, -1];
    let mode = "main", lastCat = "", liveTimer = 0;
    const tested = () => mode === "main" ? "the main test's training data (design seed 900)" : "the weighting test's expert data (data seed 0)";
    function binOf(v) {                // the page's bins: negative bins [lo, hi), positive (lo, hi], 0 in [0, 0.04]
      if (v < 0) { for (let i = 0; i < 25; i++) if (v >= E[i] && v < E[i + 1]) return i; return 0; }
      if (v === 0) return 25;
      for (let i = 25; i < 50; i++) if (v > E[i] && v <= E[i + 1]) return i;
      return 49;
    }
    function binWords(i) {
      const lo = E[i], hi = E[i + 1], up = hi <= 0;
      const a = Math.round(Math.abs(up ? hi : lo) * 100), b = Math.round(Math.abs(up ? lo : hi) * 100);
      let s = "from " + a + "% to " + b + "% " + (up ? "upstream" : "downstream");
      if (i === 5 || i === 44) s += ", 80% itself included";
      if (i < 5 || i > 44) s += ", past 80%";
      return s;
    }
    function binText(t, i) {
      const D = DATA[mode], f = D.full[t.k - 1][i], kp = D.kept[t.k - 1][i];
      let s = "Recorded throttles of thruster " + t.k + " " + binWords(i) + ": ";
      if (mode === "main") s += int(f) + " of the 50,000 training rows in the full data, and " + int(kp) +
        " of the 42,500 rows kept after the densest-15% removal.";
      else {
        s += int(f) + " of the 50,000 training rows" +
          (t.k === 1 && i === 5 ? " (" + int(DATA.iw.holds) + " of them the 80% hold, thruster 1 at exactly 80%)" : "") +
          ", and " + int(kp) + " of the 42,500 rows kept after the removal.";
      }
      t.bin.textContent = s;
    }
    function stripLabel(t) {
      const D = DATA[mode], q = D.pct[t.k - 1], p80 = D.past80[t.k - 1];
      let s = "Where thruster " + t.k + "'s recorded throttles lie in " + tested() + ". ";
      if (mode === "main" && t.k === 1) s += "Before the removal " + int(p80[0]) + " of the 50,000 rows are past 80% upstream, nearly all near full throttle; after it, " + int(p80[1]) + " of the 42,500 kept rows are. ";
      else if (mode === "iw" && t.k === 1) s += "No row is past 80%; " + int(DATA.iw.holds) + " rows are the 80% hold, and the removal deletes almost all of them. ";
      else s += "No row is past 80%. ";
      s += "The middle 90% of the kept rows lies between " + fmt(q[0], 2) + " and " + fmt(q[1], 2) + ". The table below lists every band.";
      return s;
    }
    function drawStrip(t) {
      const svg = t.svg, W = Math.round(svg.getBoundingClientRect().width);
      if (W < 80) return;
      const H = 96, top = 18, base = 74, pad = 10, D = DATA[mode];
      const full = D.full[t.k - 1], kept = D.kept[t.k - 1], fmax = Math.max(...full), ymax = fmax * 1.02;
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      svg.replaceChildren();
      const X = v => pad + (v + 1) / 2 * (W - 2 * pad), Y = n => base - n / ymax * (base - top - 2);
      const hid = "aw-hx-" + t.k;
      hatch(mk("defs", {}, svg), hid, mode === "main" ? "hx-reg" : "hx-hold");
      mk("rect", { x: X(-1), y: top, width: X(-KNEE) - X(-1), height: base - top, class: "ix-zone" }, svg);
      mk("rect", { x: X(KNEE), y: top, width: X(1) - X(KNEE), height: base - top, class: "ix-zone" }, svg);
      // the strip's own scale: its tallest bar, in rows (each strip has its own; the table gives every count)
      mk("line", { x1: pad, x2: W - pad, y1: Y(fmax), y2: Y(fmax), class: "grid" }, svg);
      mk("text", { x: pad, y: top - 5, class: "ann" }, svg, "up to " + int(fmax) + " rows in one band");
      if (X(-KNEE) - X(-1) >= 44) mk("text", { x: (X(KNEE) + X(1)) / 2, y: top - 5, "text-anchor": "middle", class: "ann" }, svg, "stalls");
      for (let i = 0; i < 50; i++) {
        const x0 = X(E[i]) + 1, w = Math.max(1, X(E[i + 1]) - X(E[i]) - 2);
        const hk = kept[i] > 0 ? Math.max(1.5, base - Y(kept[i])) : 0, rm = full[i] - kept[i];
        if (hk) mk("rect", { x: x0, y: base - hk, width: w, height: hk, class: "ix-kept" }, svg);
        if (rm > 0) {
          const hf = Math.max(hk + 1.5, base - Y(full[i]));
          mk("rect", { x: x0, y: base - hf, width: w, height: hf - hk, fill: "url(#" + hid + ")" }, svg);
        }
      }
      [-KNEE, KNEE].forEach(v => mk("line", { x1: X(v), x2: X(v), y1: top, y2: base, class: "knee k80" }, svg));
      mk("line", { x1: pad, x2: W - pad, y1: base, y2: base, class: "axis" }, svg);
      [-0.8, -0.4, 0, 0.4, 0.8].forEach(v => mk("text", { x: X(v), y: base + 16, "text-anchor": "middle", class: "tick" }, svg,
        v === 0 ? "0" : (v < 0 ? MINUS : "") + Math.round(Math.abs(v) * 100) + "%"));
      const h = hover[t.k - 1];
      if (h >= 0) mk("rect", { x: X(E[h]) + 0.5, y: top, width: X(E[h + 1]) - X(E[h]) - 1, height: base - top, class: "ix-hl" }, svg);
      const xm = X(A[t.k - 1]);
      mk("line", { x1: xm, x2: xm, y1: top + 2, y2: base, class: "ix-mark" }, svg);
      mk("path", { d: "M" + (xm - 5).toFixed(1) + "," + top + " L" + (xm + 5).toFixed(1) + "," + top + " L" + xm.toFixed(1) + "," + (top + 6), class: "ix-mark-h" }, svg);
      mk("rect", { x: 0, y: 0, width: W, height: H, class: "ix-hit" }, svg);
      svg.setAttribute("aria-label", stripLabel(t));
      t.geo = { W, pad };
    }
    function readout(t) {
      const a = A[t.k - 1], m = Math.abs(a), st = m > KNEE;
      const where = m < 0.005 ? "off" : pc(m) + (a < 0 ? " upstream" : " downstream");
      const ps = Math.abs(pushOf(a, true)), pn = Math.abs(pushOf(a, false));
      t.out.replaceChildren();
      span(t.out, "aw-t", where);
      const s1 = span(t.out, ""), b1 = document.createElement("b"); b1.textContent = ps.toFixed(2);
      s1.append("push ", b1, " in the stalling world");
      if (st) { s1.append(" "); const k = span(s1, "km", "stalls"); k.dataset.k = "stall"; }
      const s2 = span(t.out, ""), b2 = document.createElement("b"); b2.textContent = pn.toFixed(2);
      s2.append("push ", b2, " in the no-stall world");
      const s3 = span(t.out, ""), b3 = document.createElement("b"); b3.textContent = costOf(a).toFixed(3);
      s3.append("throttle cost ", b3);
      t.box.classList.toggle("is-stall", st);
      t.inp.setAttribute("aria-valuetext", "thruster " + t.k + " at " + where + (st ? ", which stalls" : "") + ": push " +
        ps.toFixed(2) + " in the stalling world and " + pn.toFixed(2) + " in the no-stall world; throttle cost " + costOf(a).toFixed(3));
    }
    function category() {
      const a1 = A[0], others = [2, 3, 4].filter(k => Math.abs(A[k - 1]) > KNEE);
      let c;
      if (a1 < 0 && Math.abs(Math.abs(a1) - 0.8) < 0.0005) c = "hold";
      else if (a1 < -KNEE) c = "past";
      else if (a1 > KNEE) c = "down";
      else c = "inside";
      return { c, others };
    }
    function message(cat, inside) {
      const main = mode === "main", out = [];
      if (cat.c === "inside") {
        out.push("Thruster 1 is within 80% here, and the data hold recorded commands like this one; the bars show how many. " +
          "Where they do, the critic's value rests on real rewards, TD3+BC's pull has recorded commands nearby to pull toward, " +
          "and BCA's band is calibrated on recorded rows like these.");
        if (inside) out.push("But this whole command lies inside the removed region, so the commands like it were deleted " +
          "from the training data: here the critic has to rely on the commands around it.");
      } else if (cat.c === "hold") {
        out.push("Thruster 1 at exactly 80% upstream is the hardest push it can give without a stall.");
        if (main) out.push("In the weighting test this command, with the other three thrusters at the hold's pattern, is the 80% hold, " +
          "which that test's removal deletes. In the main test the removal took the cluster around the full-throttle command, and " +
          "the same pattern with thruster 1 at 80% lies inside that removed region too, " + DATA.main.cappedDist.toFixed(3) +
          " from its centre; the bars show how many recorded commands with thruster 1 near 80% were kept.");
        else out.push("In the weighting test's data it is the 80% hold: in " + int(DATA.iw.holds) + " of the expert's 50,000 " +
          "training rows (data seed 0) thruster 1 is held at exactly 80%, with the other three thrusters at (" +
          DATA.presets.hold.slice(1).map(x => fmt(x, 3)).join(", ") + "), and the removal deletes " +
          (100 * DATA.iw.holdsRemoved).toFixed(1) + "% of those rows. So after the removal the critic barely sees thruster 1 at " +
          "80% either, and it never sees anything past it.");
      } else if (cat.c === "past") {
        if (main) out.push("Thruster 1 stalls here. After the main test's removal only 60 of the 42,500 kept training rows show any " +
          "throttle past 80%, so the critic is almost never corrected here: its value is a guess, carried over from below 80%, " +
          "where more throttle always meant more push. TD3+BC's pull has almost no recorded command here to pull toward. BCA's " +
          "capped pull reads its band at recorded commands only, so it never looks here. Only BCA's lower edge reads the band at " +
          "the actor's own command, and the band was calibrated on recorded rows, which hold almost no command like this one.");
        else out.push("Thruster 1 stalls here. The expert data hold no command past 80% at all, and the other four qualities hold " +
          "only uniform random commands here, about 3% to 10% of their training rows, so the critic guesses past 80% with or " +
          "without the hold removed. In the weighting test's exploratory runs BCA's lower edge, which reads the band at the " +
          "actor's own command, was far too small here: its penalty was about 1/30 to 1/160 of one built from the critic's " +
          "exact one-step error.");
      } else {
        out.push(main ? "Thruster 1 stalls here too, pushing downstream. In the main test's data no recorded command puts thruster 1 " +
          "further downstream than 73%, so the critic has no evidence here either."
          : "Thruster 1 stalls here too, pushing downstream. The expert data hold no command past 80% in either direction.");
      }
      if (cat.others.length) {
        const names = cat.others.map(k => "thruster " + k).join(cat.others.length === 2 ? " and " : ", ");
        out.push((cat.others.length === 1 ? "Also, " + names + " is" : "Also, " + names + " are") + " past 80% and stall" +
          (cat.others.length === 1 ? "s" : "") + ". " + (main ? "No recorded command in the main test's data puts thrusters 2, 3 or 4 past 80%, "
            : "No command in the expert data puts any thruster past 80%, ") + "so the critic has never seen what that does.");
      }
      return out.join(" ");
    }
    function summary(user) {
      const c = A.reduce((s, a) => s + costOf(a), 0);
      ui.cost.textContent = "its four throttles cost " + c.toFixed(3) + " this step. For comparison, the full-throttle " +
        "command costs 0.458, and the same pattern with thruster 1 held at 80% costs 0.350.";
      const B = DATA[mode].ball, d = Math.hypot(...A.map((a, i) => a - B.centre[i])), inside = d <= B.radius;
      ui.ball.textContent = (mode === "main" ? "the main test's densest-15% removal" : "the weighting test's removal (expert data, data seed 0)") +
        " deleted every recorded command within " + B.radius.toFixed(3) + " of the command (" + B.centre.map(x => fmt(x, 2)).join(", ") +
        "), measured over all four throttles together. This command lies " + d.toFixed(3) + " from it, so it is " +
        (inside ? "inside the removed region: commands like it were deleted." : "outside the removed region: commands like it were kept.");
      const cat = category(), txt = message(cat, inside);
      ui.msg.replaceChildren();
      if (cat.c === "past" || cat.c === "down" || cat.others.length) { const k = span(ui.msg, "km", "Stall."); k.dataset.k = "stall"; ui.msg.append(" "); }
      ui.msg.append(txt);
      const key = mode + cat.c + cat.others.join("") + inside;
      if (user && key !== lastCat) {
        clearTimeout(liveTimer);
        liveTimer = setTimeout(() => { ui.live.textContent = txt; }, 700);
      }
      lastCat = key;
    }
    function setA(i, v, user) {
      A[i] = Math.max(-1, Math.min(1, v));
      T[i].inp.value = String(A[i]);
      readout(T[i]); drawStrip(T[i]);
      if (hover[i] < 0) binText(T[i], binOf(A[i]));
      summary(user);
    }
    function table() {
      const D = DATA[mode];
      for (let k = 1; k <= 4; k++) for (let j = 0; j < 10; j++) {
        let f = 0, kp = 0;
        for (let i = 5 * j; i < 5 * j + 5; i++) { f += D.full[k - 1][i]; kp += D.kept[k - 1][i]; }
        const cf = $("aw-t" + k + "-" + j + "f"), ck = $("aw-t" + k + "-" + j + "k");
        if (cf) cf.textContent = int(f);
        if (ck) ck.textContent = int(kp);
      }
      ui.tnote.textContent = mode === "main"
        ? "Rows of the main test's training data (design seed 900) with each thruster's throttle in each band: all 50,000 rows, and the 42,500 rows kept after the densest-15% removal. A negative throttle pushes upstream."
        : "Rows of the weighting test's expert training data (data seed 0) with each thruster's throttle in each band: all 50,000 rows, and the 42,500 rows kept after the removal, which deletes the 80% hold. A negative throttle pushes upstream.";
    }
    function setMode(m) {
      mode = m;
      ui.keyRm.dataset.k = m === "main" ? "region" : "hold";
      ui.keyRmT.textContent = m === "main" ? "recorded commands that the main test's densest-15% removal deleted"
        : "recorded commands that the weighting test's removal deleted (mostly the 80% hold)";
      T.forEach(t => { drawStrip(t); binText(t, hover[t.k - 1] >= 0 ? hover[t.k - 1] : binOf(A[t.k - 1])); });
      table(); summary(false);
    }
    T.forEach((t, i) => {
      t.inp.addEventListener("input", () => setA(i, Math.round(Number(t.inp.value) * 100) / 100, true));
      keys(t.inp, v => setA(i, v, true));
      const valAt = e => {
        const r = t.svg.getBoundingClientRect(), g = t.geo || { W: r.width, pad: 10 };
        return Math.max(-1, Math.min(1, ((e.clientX - r.left) - g.pad) / (g.W - 2 * g.pad) * 2 - 1));
      };
      t.svg.addEventListener("pointermove", e => { hover[i] = binOf(valAt(e)); drawStrip(t); binText(t, hover[i]); });
      t.svg.addEventListener("pointerleave", () => { hover[i] = -1; drawStrip(t); binText(t, binOf(A[i])); });
      t.svg.addEventListener("click", e => { hover[i] = -1; setA(i, Math.round(valAt(e) * 100) / 100, true); });
      watch(t.svg, () => drawStrip(t));
    });
    fig.querySelectorAll("button[data-preset]").forEach(b => b.addEventListener("click", () => {
      const p = DATA.presets[b.dataset.preset === "capped" && mode === "iw" ? "hold" : b.dataset.preset];
      if (!p) return;
      p.forEach((v, i) => { A[i] = v; T[i].inp.value = String(v); readout(T[i]); drawStrip(T[i]); binText(T[i], binOf(v)); });
      summary(true);
    }));
    fig.querySelectorAll('input[name="aw-data"]').forEach(r => r.addEventListener("change", () => { if (r.checked) setMode(r.value); }));
    const chosen = fig.querySelector('input[name="aw-data"]:checked');
    T.forEach((t, i) => { A[i] = Math.round(Number(t.inp.value) * 100) / 100; readout(t); });
    setMode(chosen ? chosen.value : "main");
  })();

  /* ================================ the stall chart ================================ */
  (function () {
    const fig = $("stall-chart");
    if (!fig || !window.ResizeObserver) return;
    const svg = $("sc-svg"), inp = $("sc-a"), cbF = $("sc-full"), cbK = $("sc-kept");
    const ro = { a: $("sc-va"), ps: $("sc-vs"), pn: $("sc-vn"), c: $("sc-vc"), cmp: $("sc-cmp"), val: $("sc-val"), bin: $("sc-bin") };
    if (!svg || !inp || !cbF || !cbK || Object.values(ro).some(x => !x)) return;
    const M = DATA.mag, ME = DATA.magEdges, NET = DATA.net;
    let hov = null, geo = null;
    const binOf = m => { if (m <= 0) return 0; for (let i = 0; i < 25; i++) if (m > ME[i] && m <= ME[i + 1]) return i; return 24; };
    function readouts(a, pointer) {
      const st = a > KNEE, ps = pushOf(a, true), pn = pushOf(a, false), c = costOf(a);
      ro.a.textContent = pc(a) + (pointer ? " (pointer)" : "");
      ro.ps.replaceChildren(ps.toFixed(2));
      if (st) span(ro.ps, "st", "stalls");
      ro.pn.textContent = pn.toFixed(2);
      ro.c.textContent = c.toFixed(3);
      if (a > 0.8 + 1e-9) ro.cmp.textContent = "Against 80%: in the stalling world the push is " + (0.8 - ps).toFixed(2) +
        " lower, while in the no-stall world it is " + (pn - 0.8).toFixed(2) + " higher; the throttle cost is " + (c - 0.192).toFixed(3) +
        " higher in both.";
      else ro.cmp.textContent = "Up to 80% both worlds give the same push, equal to the throttle, so data recorded here " +
        "cannot tell the two worlds apart.";
      const n = NET.reduce((b, r) => Math.abs(r.t - a) < Math.abs(b.t - a) ? r : b, NET[0]);
      const at = Math.round(n.t * 100) + "%", near = Math.abs(a - n.t) > 0.004 ? " (the nearest step)" : "";
      const head = "True value at one example state (floats at rest, a current of 0.6, the wave adding 0.3), computed every 5% of throttle: ";
      if (Math.abs(n.t - 0.8) < 1e-9) ro.val.textContent = head + "80%" + near + " is the best throttle there, in both worlds.";
      else ro.val.textContent = head + "at " + at + near + " thruster 1 is worth " + Math.abs(n.s).toFixed(3) +
        " less than at 80% in the stalling world and " + Math.abs(n.n).toFixed(3) + " less in the no-stall world" +
        (n.t > 0.8 ? "." : ", because too little push lets the floats drift.");
      const i = binOf(a), lo = Math.round(ME[i] * 100), hi = Math.round(ME[i + 1] * 100);
      ro.bin.textContent = "Thruster 1's recorded throttles from " + lo + "% to " + hi + "% in size" + (i === 19 ? " (80% itself included)" : "") +
        ": " + int(M.full[i]) + " of the 50,000 rows in the full data, and " + int(M.kept[i]) + " of the 42,500 rows kept after the main test's removal.";
      if (!pointer) inp.setAttribute("aria-valuetext", pc(a) + (st ? ", which stalls" : "") + ": push " + ps.toFixed(2) +
        " in the stalling world and " + pn.toFixed(2) + " in the no-stall world; throttle cost " + c.toFixed(3));
    }
    function draw() {
      const W = Math.round(svg.getBoundingClientRect().width);
      if (W < 160) return;
      const narrow = W < 560, L = narrow ? 40 : 48, R = 12;
      const hA = Math.round(Math.min(230, Math.max(150, W * 0.3))), hB = Math.round(Math.min(130, Math.max(92, W * 0.15)));
      const tA = 26, bA = tA + hA, tB = bA + 50, bB = tB + hB, H = bB + 46;
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      svg.setAttribute("height", String(H));
      svg.replaceChildren();
      const X = m => L + m * (W - L - R), YA = v => bA - v * hA, YMAX = 7000, YB = n => bB - n / YMAX * hB;
      geo = { L, R, W };
      hatch(mk("defs", {}, svg), "sc-hx", "hx-reg");
      // the stall region, in both panels
      mk("rect", { x: X(0.8), y: tA, width: X(1) - X(0.8), height: hA, class: "ix-zone" }, svg);
      mk("rect", { x: X(0.8), y: tB, width: X(1) - X(0.8), height: hB, class: "ix-zone" }, svg);
      mk("text", { x: (X(0.8) + X(1)) / 2, y: bA - 8, "text-anchor": "middle", class: "ann" }, svg, X(1) - X(0.8) >= 80 ? "stall region" : "stall");
      // panel A: push and throttle cost, one axis from 0 to 1
      mk("text", { x: L, y: tA - 10, class: "ix-ttl" }, svg, narrow ? "Push or throttle cost" : "Push, or throttle cost, of one thruster");
      [0, 0.2, 0.4, 0.6, 0.8, 1].forEach(v => {
        mk("line", { x1: L, x2: W - R, y1: YA(v), y2: YA(v), class: v === 0 ? "axis" : "grid" }, svg);
        mk("text", { x: L - 6, y: YA(v) + 4, "text-anchor": "end", class: "tick" }, svg, v.toFixed(1));
      });
      let d = "";
      for (let k = 0; k <= 50; k++) { const m = k / 50; d += (k ? " L" : "M") + X(m).toFixed(1) + "," + YA(costOf(m)).toFixed(1); }
      mk("path", { d, class: "sc-c" }, svg);
      mk("path", { d: "M" + X(0) + "," + YA(0) + " L" + X(0.8) + "," + YA(0.8) + " L" + X(1) + "," + YA(0.5), class: "sc-ps" }, svg);
      mk("path", { d: "M" + X(0) + "," + YA(0) + " L" + X(1) + "," + YA(1), class: "sc-pn" }, svg);
      mk("text", { x: X(1), y: tA - 10, "text-anchor": "end", class: "lab" }, svg, narrow ? "no stall ↗" : "push, no-stall world ↗");
      mk("text", { x: X(1) - 4, y: YA(0.5) + 18, "text-anchor": "end", class: "lab" }, svg, narrow ? "push, stalling" : "push, stalling world");
      mk("text", { x: X(0.3), y: YA(0.12), "text-anchor": "start", class: "lab" }, svg, "throttle cost");
      // panel B: thruster 1's recorded throttles by size, on their own axis (rows)
      mk("text", { x: L, y: tB - 10, class: "ix-ttl" }, svg, narrow ? "Thruster 1's recorded throttles" : "Thruster 1's recorded throttles in the main test's data, rows");
      [0, 2000, 4000, 6000].forEach(n => {
        mk("line", { x1: L, x2: W - R, y1: YB(n), y2: YB(n), class: n === 0 ? "axis" : "grid" }, svg);
        mk("text", { x: L - 6, y: YB(n) + 4, "text-anchor": "end", class: "tick" }, svg, n ? n / 1000 + "K" : "0");
      });
      const showF = cbF.checked, showK = cbK.checked;
      for (let i = 0; i < 25; i++) {
        const slot = X(ME[i + 1]) - X(ME[i]), w = Math.min(24, slot - 2), x0 = X(ME[i]) + (slot - w) / 2;
        const f = M.full[i], kp = M.kept[i];
        const hk = kp > 0 ? Math.max(1.5, bB - YB(kp)) : 0, hf = f > 0 ? Math.max(1.5, bB - YB(f)) : 0;
        if (showF && showK) {
          if (hk) mk("rect", { x: x0, y: bB - hk, width: w, height: hk, rx: 1, class: "ix-kept" }, svg);
          if (f > kp) { const top = Math.max(hk + 1.5, hf); mk("rect", { x: x0, y: bB - top, width: w, height: top - hk, fill: "url(#sc-hx)" }, svg); }
        } else if (showF) { if (hf) mk("rect", { x: x0, y: bB - hf, width: w, height: hf, rx: 1, class: "ix-kept" }, svg); }
        else if (showK) { if (hk) mk("rect", { x: x0, y: bB - hk, width: w, height: hk, rx: 1, class: "ix-kept" }, svg); }
      }
      if (!showF && !showK) mk("text", { x: L + (W - L - R) / 2, y: tB + hB / 2, "text-anchor": "middle", class: "ann" }, svg,
        "Tick a box above to show the recorded throttles.");
      // the 80% line through both panels, and the axis
      mk("line", { x1: X(0.8), x2: X(0.8), y1: tA, y2: bB, class: "knee k80" }, svg);
      [0, 0.2, 0.4, 0.6, 0.8, 1].forEach(m => mk("text", { x: X(m), y: bB + 16, "text-anchor": "middle", class: "tick" }, svg, pc(m)));
      mk("text", { x: L + (W - L - R) / 2, y: bB + 38, "text-anchor": "middle", class: "axlab" }, svg,
        narrow ? "Size of the throttle (0% off, 100% full)" : "Size of the throttle: 0% is off, 100% is full throttle");
      // the crosshair at the slider's throttle, or at the pointer
      const a = hov != null ? hov : Number(inp.value), xa = X(a), bi = binOf(a);
      mk("rect", { x: X(ME[bi]) + 0.5, y: tB, width: X(ME[bi + 1]) - X(ME[bi]) - 1, height: hB, class: "ix-hl" }, svg);
      mk("line", { x1: xa, x2: xa, y1: tA, y2: bB, class: "ix-cross" }, svg);
      mk("circle", { cx: xa, cy: YA(costOf(a)), r: 4.5, class: "sc-dc" }, svg);
      mk("circle", { cx: xa, cy: YA(pushOf(a, false)), r: 4.5, class: "sc-dpn" }, svg);
      mk("circle", { cx: xa, cy: YA(pushOf(a, true)), r: 4.5, class: "sc-dps" }, svg);
      mk("rect", { x: 0, y: 0, width: W, height: H, class: "ix-hit" }, svg);
    }
    const valAt = e => {
      const r = svg.getBoundingClientRect(), g = geo || { L: 48, R: 12, W: r.width };
      return Math.max(0, Math.min(1, Math.round(((e.clientX - r.left) - g.L) / (g.W - g.L - g.R) * 100) / 100));
    };
    function set(v) { inp.value = String(v); hov = null; readouts(v, false); draw(); }
    inp.addEventListener("input", () => set(Math.round(Number(inp.value) * 100) / 100));
    keys(inp, set);
    [cbF, cbK].forEach(c => c.addEventListener("change", draw));
    svg.addEventListener("pointermove", e => { hov = valAt(e); readouts(hov, true); draw(); });
    svg.addEventListener("pointerleave", () => { hov = null; readouts(Number(inp.value), false); draw(); });
    svg.addEventListener("click", e => set(valAt(e)));
    readouts(Number(inp.value), false);
    watch(svg, draw);
  })();
})();
