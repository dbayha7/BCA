/* The critic in action: one real episode of a trained TD3+BC actor, state by state, with the critic's value Q(s, a)
   of every throttle of thruster 1 next to the exact true value Q^pi(s, a). It reads the JSON data tag with the id
   qv-data and only touches #critic-in-action. Nothing plays until the reader presses Play. */
(function () {
  "use strict";
  const root = document.getElementById("critic-in-action");
  const src = document.getElementById("qv-data");
  if (!root || !src) return;
  let D;
  try { D = JSON.parse(src.textContent); } catch (err) { return; }
  const NS = "http://www.w3.org/2000/svg";
  const $ = id => document.getElementById(id);
  const FI = {};
  D.fields.forEach((f, i) => { FI[f] = i; });
  const U = D.u, N = U.length, I80 = D.i80, I100 = N - 1;
  const MINUS = "−";
  const QL = { random: "random", "medium-replay": "medium-replay", medium: "medium", "medium-expert": "medium-expert",
    expert: "expert" };
  const CL = { gap: "the 80% hold removed", full: "all the data" };
  const SL = { training: "training mix of currents", strong: "strong test mix" };

  /* ---------- numbers as text ---------- */
  function grp(v, d) {
    const a = Math.abs(v);
    const s = a.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
    return (v < 0 && Number(a.toFixed(d)) !== 0 ? MINUS : "") + s;
  }
  const fx = (v, d) => grp(v, d);
  const sg = (v, d) => { const s = grp(v, d); return Number(Math.abs(v).toFixed(d)) === 0 ? s : (v > 0 ? "+" + s : s); };
  const fv = v => fx(v, Math.abs(v) >= 100 ? 1 : 2);                 // a value of the return
  const s4 = v => v === 0 ? "0" : fx(v, Math.max(0, 3 - Math.floor(Math.log10(Math.abs(v)))));
  const s3 = v => v === 0 ? "0" : fx(v, Math.max(0, 2 - Math.floor(Math.log10(Math.abs(v)))));
  const fu = u => (Number.isInteger(u) ? String(u) : u.toFixed(1)) + "%";   // a swept throttle (upstream %)
  const pc1 = u => fx(u, 1) + "%";
  const thr = a => a <= 0 ? pc1(-a * 100) + " upstream" : pc1(a * 100) + " downstream";
  const share = v => { const p = v * 100; return p === 0 ? "0%" : v === 1 ? "100%" : p < 0.01 ? "under 0.01%" : (p < 0.1 ? p.toFixed(2) : p.toFixed(1)) + "%"; };
  const SUP = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻" };
  const FMT = {
    f3: v => fx(v, 3), s4, i: v => grp(v, 0), pct1: v => pc1(v),
    ppm: v => fx(v * 1e6, 1) + " parts in a million",
    sci: v => { const e = Math.floor(Math.log10(Math.abs(v))), mnt = v / Math.pow(10, e);
      return mnt.toFixed(1) + " × 10" + String(e).split("").map(c => SUP[c] || c).join(""); },
    ord: v => (Math.round(v * 1000) / 10).toFixed(1) + "th"
  };
  function pathVal(p) {
    if (p === "ball.u0") return -D.ball.centre[0] * 100;
    return p.split(".").reduce((o, k) => (o == null ? o : o[k]), D);
  }
  root.querySelectorAll("[data-qv-num]").forEach(el => {
    const v = pathVal(el.dataset.qvNum), f = FMT[el.dataset.qvFmt] || s4;
    if (typeof v === "number" && isFinite(v)) el.textContent = f(v);
  });
  // the removal button's tooltip: how many of the holds the ball takes, from the data
  (function () {
    const r = Object.values(D.train).map(t => t.holds_removed_training).filter(v => typeof v === "number");
    const b = root.querySelector('[data-qv-c="gap"]');
    if (!b || !r.length) return;
    const lo = Math.min(...r) * 100, hi = Math.max(...r) * 100;
    b.setAttribute("data-tip", "The critic and actor trained after the densest-15% removal, which deletes every recorded " +
      "training step whose command lies inside this section's removed ball: the ball that holds the densest 15% of the " +
      "expert data's commands. It takes " + lo.toFixed(1) + "% to " + hi.toFixed(1) + "% of the 80% holds, so this critic " +
      "has almost never seen thruster 1 held at 80%.");
  })();

  /* ---------- episodes ---------- */
  const key = (q, c, s) => q + "|" + c + "|" + s;
  const IDX = new Map();
  D.eps.forEach((e, i) => IDX.set(key(e[FI.quality], e[FI.critic], e[FI.situation]), i));
  const CACHE = new Map();
  // a sweep: integers in millionths relative to the value at the actor's command, stored as r0, r1 - r0, then second
  // differences; summed twice and divided by a million they give back exactly the computed 7-digit values
  function decode(rows, refs) {
    return rows.map((row, t) => {
      const out = new Float64Array(row.length), ra = Math.round(refs[t] * 1e6);
      let r = 0, d = 0;
      for (let j = 0; j < row.length; j++) {
        if (j === 0) r = row[0];
        else { d = j === 1 ? row[1] : d + row[j]; r += d; }
        out[j] = (ra + r) / 1e6;
      }
      return out;
    });
  }
  function episode(q, c, s) {
    const k = key(q, c, s);
    if (CACHE.has(k)) return CACHE.get(k);
    const i = IDX.get(k);
    if (i == null) return null;
    const e = D.eps[i];
    const E = { i, q, c, s, ret: e[FI.return], pts: e[FI.points], push: e[FI.push], pos: e[FI.pos], cmd: e[FI.cmd],
      rew: e[FI.reward], qa: e[FI.q_actor], pa: e[FI.qpi_actor],
      qS: decode(e[FI.q_sweep], e[FI.q_actor]), pS: decode(e[FI.qpi_sweep], e[FI.qpi_actor]),
      hat: e[FI.a_hat], best: e[FI.a_best_true], slice: e[FI.slice], inball: e[FI.actor_in_ball], summ: D.summ[i],
      sit: D.sit[s] };
    E.T = E.qa.length;
    E.flag = new Set(E.summ.critic_rises_while_truth_falls_steps || []);
    CACHE.set(k, E);
    return E;
  }

  // the true value lost by taking â instead of the true best of the sweep, from the exact swept values
  function lossText(E, t) {
    const hat = E.hat[t], best = E.best[t];
    if (hat === best) return fx(0, 3);
    const s = fx(E.pS[t][best] - E.pS[t][hat], 3);
    return s === "0.000" ? "under 0.001" : s;
  }

  /* ---------- state ---------- */
  const pressed = attr => { const b = root.querySelector("[" + attr + '][aria-pressed="true"]'); return b ? b.getAttribute(attr) : null; };
  const S = { q: pressed("data-qv-q") || "expert", c: pressed("data-qv-c") || "gap", s: pressed("data-qv-s") || "strong",
    view: pressed("data-qv-v") || "rel", range: pressed("data-qv-r") || "hi", t: 0, timer: 0, h1: null, h2: null };
  const cur = () => episode(S.q, S.c, S.s);

  /* ---------- svg helpers ---------- */
  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const clear = n => { while (n.firstChild) n.removeChild(n.firstChild); };
  const widthOf = n => Math.max(280, Math.round(n.clientWidth || 600));
  function niceTicks(lo, hi, n) {
    if (!(hi > lo)) { const p = Math.max(Math.abs(lo) * 0.01, 0.01); lo -= p; hi += p; }
    const raw = (hi - lo) / n, p10 = Math.pow(10, Math.floor(Math.log10(raw)));
    let step = 10 * p10;
    for (const f of [1, 2, 2.5, 5, 10]) if (f * p10 >= raw - 1e-12) { step = f * p10; break; }
    const t0 = Math.floor(lo / step + 1e-9) * step, t1 = Math.ceil(hi / step - 1e-9) * step;
    const v = [];
    for (let x = t0, i = 0; x <= t1 + step * 1e-6 && i < 40; x += step, i++) v.push(Math.round(x / step) * step);
    let d = 0;
    while (d < 6 && Math.abs(Math.round(step * Math.pow(10, d)) - step * Math.pow(10, d)) > 1e-6) d++;
    return { v, d, lo: v[0], hi: v[v.length - 1] };
  }
  function frame(svg, W, H, m, xd, yd, xt, yt, xf, yf) {
    const X = v => m.l + (v - xd[0]) / (xd[1] - xd[0]) * (W - m.l - m.r);
    const Y = v => H - m.b - (v - yd[0]) / (yd[1] - yd[0]) * (H - m.t - m.b);
    const g = el("g", {}, svg);
    yt.forEach(t => {
      el("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }, g);
      el("text", { class: "tick", x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, g, yf(t));
    });
    el("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, g);
    xt.forEach(t => el("text", { class: "tick", x: X(t), y: H - m.b + 17, "text-anchor": "middle" }, g, xf(t)));
    return { X, Y };
  }
  const pathOf = (pts, X, Y) => pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("");

  /* ---------- tooltip (one per chart, inside its box) ---------- */
  function tipRow(tip, color, value, nameHtml) {
    const r = document.createElement("div"); r.className = "qv-tr";
    const k = document.createElement("span"); k.className = "qv-tk"; if (color) k.style.background = "var(" + color + ")";
    const v = document.createElement("span"); v.className = "qv-tv"; v.textContent = value;
    const n = document.createElement("span"); n.className = "qv-tn"; n.innerHTML = nameHtml;   // fixed text of this script
    r.append(k, v, n); tip.appendChild(r);
  }
  function tipText(tip, cls, text) { const d = document.createElement("div"); d.className = cls; d.textContent = text; tip.appendChild(d); return d; }
  function placeTip(tip, box, svg, sx, sy) {
    tip.hidden = false;
    const br = box.getBoundingClientRect(), sr = svg.getBoundingClientRect();
    const scale = sr.width / (svg.viewBox.baseVal.width || sr.width);
    const x = sr.left - br.left + sx * scale, y = sr.top - br.top + sy * scale;
    const bw = box.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
    if (bw < 520) {
      // a narrow chart: the readout never covers the plot. It goes under the chart, or, when the window has no room
      // there, over the heading and legend above it
      const below = sr.bottom - br.top + 6, above = sr.top - br.top - th - 6;
      const roomBelow = sr.bottom + 6 + th <= window.innerHeight - 8, roomAbove = sr.top - th - 6 >= 8;
      tip.style.left = Math.round(Math.max(0, Math.min(x - tw / 2, bw - tw))) + "px";
      tip.style.top = Math.round(!roomBelow && roomAbove ? above : below) + "px";
      return;
    }
    let left = x + 14;
    if (left + tw > bw) left = x - 14 - tw;
    left = Math.max(0, Math.min(left, bw - tw));
    let top = y - th - 10;
    if (top < 0) top = y + 18;
    tip.style.left = Math.round(left) + "px"; tip.style.top = Math.round(top) + "px";
  }

  /* ---------- the channel strip ---------- */
  function drawStrip() {
    const E = cur(), t = S.t, svg = $("qv-strip");
    const W = widthOf(svg.parentNode), narrow = W < 520;
    const bandT = 0, bandH = narrow ? 70 : 74, H = bandT + bandH + (narrow ? 54 : 40);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    el("rect", { class: "qv-water", x: 0, y: bandT, width: W, height: bandH }, svg);
    const y0 = bandT + bandH / 2 + 4;
    el("line", { class: "qv-rest", x1: 0, x2: W, y1: y0, y2: y0 }, svg);
    el("text", { class: "qv-flow", x: 8, y: bandT + 14 }, svg, narrow ? "water flows down · dashed line: rest" :
      "Water flows down the screen; the dashed line is rest. A float below it has been carried downstream.");
    const col = W / 5, sc = (bandH / 2 - 14) / 2.5, pos = E.pos[t], a = E.cmd[t];
    const fy = j => Math.max(bandT + 22, Math.min(bandT + bandH - 10, y0 - pos[j] * sc));
    const fxp = j => col * (j + 0.5);
    for (let k = 0; k < 4; k++) el("line", { class: "qv-tie", x1: fxp(k), y1: fy(k), x2: fxp(k + 1), y2: fy(k + 1) }, svg);
    for (let k = 0; k < 4; k++) {
      const x = col * (k + 1), y = (fy(k) + fy(k + 1)) / 2, past = Math.abs(a[k]) > 0.8;
      el("rect", { class: "qv-thr" + (past ? " w" : k === 0 ? " on" : ""), x: x - 5, y: y - 5, width: 10, height: 10, rx: 2 }, svg);
      el("text", { class: "qv-tname", x, y: bandT + bandH + 15, "text-anchor": "middle" }, svg, "thruster " + (k + 1));
      // the same one-decimal throttle as the readouts; on a narrow screen the direction goes on a second line
      const pct = pc1(Math.abs(a[k]) * 100), dir = a[k] <= 0 ? "up" : "down", cls = "qv-tval" + (past ? " w" : "");
      if (narrow) {
        el("text", { class: cls, x, y: bandT + bandH + 32, "text-anchor": "middle" }, svg, pct);
        el("text", { class: cls, x, y: bandT + bandH + 47, "text-anchor": "middle" }, svg, dir);
      } else el("text", { class: cls, x, y: bandT + bandH + 32, "text-anchor": "middle" }, svg, pct + " " + dir);
    }
    for (let j = 0; j < 5; j++) el("circle", { class: "qv-float", cx: fxp(j), cy: fy(j), r: narrow ? 8 : 9 }, svg);
    svg.setAttribute("aria-label", "The channel in state " + t + ". Floats 1 to 5 sit " +
      pos.map(p => fx(p, 2)).join(", ") + " from rest (negative is downstream). Thrusters 1 to 4: " +
      a.map(v => thr(v)).join(", ") + ".");
  }

  /* ---------- the value of each throttle of thruster 1, at the state on screen ---------- */
  const M1 = {};
  function inBall(E, t, u) { const s = E.c === "gap" ? E.slice[t] : null; return !!s && u >= s[0] * 100 - 1e-9 && u <= s[1] * 100 + 1e-9; }
  function drawMain() {
    const E = cur(), t = S.t, svg = $("qv-svg1"), box = $("qv-c1");
    const W = widthOf(box), narrow = W < 520, H = narrow ? 268 : 300;
    const m = { l: narrow ? 56 : 60, r: 20, t: 30, b: 42 };
    const rel = S.view === "rel", qa = E.qa[t], pa = E.pa[t];
    const yq = Array.from(E.qS[t], v => rel ? v - qa : v), yp = Array.from(E.pS[t], v => rel ? v - pa : v);
    const u0 = S.range === "hi" ? 60 : 0, k0 = Math.round(u0 / 2.5), inR = u => u >= u0 - 1e-9 && u <= 100 + 1e-9;
    const a1 = E.cmd[t][0], uAct = -a1 * 100, actIn = inR(uAct);
    const ptsQ = [], ptsP = [];
    U.forEach((u, k) => { if (k >= k0) { ptsQ.push([u, yq[k]]); ptsP.push([u, yp[k]]); } });
    if (actIn && !U.some(u => Math.abs(u - uAct) < 1e-9)) {           // the actor's own command, exactly computed
      const at = ptsQ.findIndex(p => p[0] > uAct);
      ptsQ.splice(at, 0, [uAct, rel ? 0 : qa]); ptsP.splice(at, 0, [uAct, rel ? 0 : pa]);
    }
    const ys = ptsQ.concat(ptsP).map(p => p[1]);
    const tk = niceTicks(Math.min(...ys), Math.max(...ys), narrow ? 4 : 5);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    const X0 = v => m.l + (v - u0) / (100 - u0) * (W - m.l - m.r);
    // regions first: the part of the sweep inside the removed ball, then the stall
    const sl = E.c === "gap" ? E.slice[t] : null, regionLabs = [];
    if (sl) {
      const x1 = X0(Math.max(u0, sl[0] * 100)), x2 = X0(Math.min(100, sl[1] * 100));
      if (x2 > x1) el("rect", { class: "qv-ball", x: x1, y: m.t, width: Math.max(0, x2 - x1), height: H - m.t - m.b }, svg);
      if (x2 > x1) [x1, x2].forEach(x => el("line", { class: "qv-ball-l", x1: x, x2: x, y1: m.t, y2: H - m.b }, svg));
      if (x2 - x1 > 54) { el("text", { class: "ann qv-halo qv-rlab", x: (x1 + x2) / 2, y: H - m.b - 6, "text-anchor": "middle" }, svg, "removed");
        regionLabs.push([(x1 + x2) / 2, H - m.b - 6, "middle", "removed"]); }
    }
    el("rect", { class: "qv-stall", x: X0(80), y: m.t, width: X0(100) - X0(80), height: H - m.t - m.b }, svg);
    const yf = v => (rel ? sg(v, tk.d) : fx(v, tk.d));
    const xt = u0 ? [60, 70, 80, 90, 100] : [0, 20, 40, 60, 80, 100];
    const { X, Y } = frame(svg, W, H, m, [u0, 100], [tk.lo, tk.hi], xt, tk.v, v => v + "%", yf);
    el("text", { class: "ann qv-halo qv-rlab", x: (X(80) + X(100)) / 2, y: H - m.b - 6, "text-anchor": "middle" }, svg, "stall");
    regionLabs.push([(X(80) + X(100)) / 2, H - m.b - 6, "middle", "stall"]);
    el("line", { class: "qv-stall-l", x1: X(80), x2: X(80), y1: m.t, y2: H - m.b }, svg);
    if (rel && tk.lo < 0 && tk.hi > 0) el("line", { class: "qv-zero", x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0) }, svg);
    el("text", { class: "axlab", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, svg,
      narrow ? "thruster 1's throttle, upstream" : "thruster 1's throttle, upstream (thrusters 2 to 4 as the actor set them)");
    // the actor's throttle
    if (actIn) {
      el("line", { class: "qv-act", x1: X(uAct), x2: X(uAct), y1: m.t - 4, y2: H - m.b }, svg);
      const lx = Math.max(m.l + 20, Math.min(W - m.r - 20, X(uAct)));
      el("text", { class: "lab qv-halo", x: lx, y: m.t - 9, "text-anchor": "middle" }, svg, "actor " + fu(Math.round(uAct * 10) / 10));
    } else {
      el("text", { class: "lab", x: m.l, y: m.t - 9, "text-anchor": "start" }, svg,
        "← actor: " + thr(a1) + ", off this range");
    }
    // the two curves: the true value under, the critic on top
    el("path", { class: "qv-lp", d: pathOf(ptsP, X, Y) }, svg);
    el("path", { class: "qv-lq", d: pathOf(ptsQ, X, Y) }, svg);
    // markers: a-hat on the critic's curve, the true best on the true curve
    const hIn = E.hat[t] >= k0, bIn = E.best[t] >= k0;
    const hx = X(U[E.hat[t]]), hy = Y(yq[E.hat[t]]), bx = X(U[E.best[t]]), by = Y(yp[E.best[t]]);
    if (bIn) el("path", { class: "qv-best", d: "M" + bx + "," + (by - 6.5) + "L" + (bx + 6.5) + "," + by + "L" + bx + "," + (by + 6.5) + "L" + (bx - 6.5) + "," + by + "Z" }, svg);
    if (hIn) el("circle", { class: "qv-hat", cx: hx, cy: hy, r: 6 }, svg);
    const off = [];
    if (!hIn) off.push("â at " + fu(U[E.hat[t]]));
    if (!bIn) off.push("true best at " + fu(U[E.best[t]]));
    const offNote = off.length ? " Off this range, to the left: " + off.join(" and ") + "." : "";
    // labels: the markers' first, then the two curves' names; each takes the first free spot from its list
    const cw = narrow ? 7.3 : 6.8, bounds = { x1: m.l + 1, x2: W - m.r - 1, y1: m.t + 1, y2: H - m.b - 1 };
    const lbox = (x, y, anchor, text) => { const w = text.length * cw + 4, x1 = anchor === "middle" ? x - w / 2 : anchor === "end" ? x - w : x;
      return { x1, x2: x1 + w, y1: y - 11, y2: y + 3 }; };
    const taken = regionLabs.map(r => lbox(r[0], r[1], r[2], r[3]));
    if (hIn) taken.push({ x1: hx - 7, x2: hx + 7, y1: hy - 7, y2: hy + 7 });
    if (bIn) taken.push({ x1: bx - 7, x2: bx + 7, y1: by - 7, y2: by + 7 });
    if (actIn) taken.push({ x1: X(uAct) - 2, x2: X(uAct) + 2, y1: m.t - 4, y2: H - m.b, line: true });   // the actor's line
    taken.push({ x1: X(80) - 2, x2: X(80) + 2, y1: m.t, y2: H - m.b, line: true });                       // the 80% line
    if (sl) [Math.max(u0, sl[0] * 100), Math.min(100, sl[1] * 100)].forEach(u => {                         // the region's edges
      if (u > u0 && u < 100) taken.push({ x1: X(u) - 1.5, x2: X(u) + 1.5, y1: m.t, y2: H - m.b, line: true }); });
    const clash = (a, b) => a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2;
    const inside = b => b.x1 >= bounds.x1 && b.x2 <= bounds.x2 && b.y1 >= bounds.y1 && b.y2 <= bounds.y2;
    // does either curve pass through a label's box (with 2 px to spare)?
    const curvesPx = [ptsQ, ptsP].map(pts => pts.map(p => [X(p[0]), Y(p[1])]));
    function onCurve(b) {
      const x1 = b.x1 - 2, x2 = b.x2 + 2, y1 = b.y1 - 2, y2 = b.y2 + 2;
      for (const c of curvesPx) for (let i = 1; i < c.length; i++) {
        const [ax, ay] = c[i - 1], [bx2, by2] = c[i];
        if (bx2 < x1 || ax > x2) continue;
        const at = x => bx2 === ax ? ay : ay + (by2 - ay) * (x - ax) / (bx2 - ax);
        const ya = at(Math.max(ax, x1)), yb = at(Math.min(bx2, x2));
        if (Math.max(ya, yb) >= y1 && Math.min(ya, yb) <= y2) return true;
      }
      return false;
    }
    function label(text, cls, cands, must) {
      for (const [x, y, anchor] of cands) {
        const b = lbox(x, y, anchor, text);
        if (!inside(b) || taken.some(o => clash(o, b)) || onCurve(b)) continue;
        taken.push(b);
        el("text", { class: cls + " qv-halo", x, y, "text-anchor": anchor }, svg, text);
        return true;
      }
      if (must && cands.length) {
        // no free spot: take the spot that covers no other label or marker and crosses the fewest lines (the actor's
        // line, the curves), and set the label on a small plate of the chart's own colour so no line runs through it
        let pick = null, best = Infinity;
        for (const [x, y, anchor] of cands) {
          const b = lbox(x, y, anchor, text);
          const cx = Math.max(bounds.x1 - b.x1, Math.min(0, bounds.x2 - b.x2)), cy = Math.max(bounds.y1 - b.y1, Math.min(0, bounds.y2 - b.y2));
          const bb = { x1: b.x1 + cx, x2: b.x2 + cx, y1: b.y1 + cy, y2: b.y2 + cy };
          const marks = taken.filter(o => !o.line && clash(o, bb)).length, lines = taken.filter(o => o.line && clash(o, bb)).length;
          const score = 10 * marks + lines + (onCurve(bb) ? 1 : 0);
          if (score < best) { best = score; pick = [x + cx, y + cy, anchor, bb, marks]; }
        }
        taken.push(pick[3]);
        const tx = el("text", { class: cls + " qv-halo", x: pick[0], y: pick[1], "text-anchor": pick[2] }, svg, text);
        if (best > 0 && pick[4] === 0) {
          let r = null;
          try { r = tx.getBBox(); } catch (err) { r = null; }
          if (r && r.width > 0) svg.insertBefore(el("rect", { class: "qv-lbg", x: r.x - 2, y: r.y - 1, width: r.width + 4, height: r.height + 2, rx: 2 }), tx);
        }
      }
      return false;
    }
    const around = (x, y) => [[x, y - 11, "middle"], [x, y + 20, "middle"], [x + 10, y + 4, "start"], [x - 10, y + 4, "end"],
      [x, y - 25, "middle"], [x, y + 34, "middle"], [x + 10, y - 10, "start"], [x - 10, y - 10, "end"],
      [x + 10, y + 18, "start"], [x - 10, y + 18, "end"], [x, y - 39, "middle"], [x, y + 48, "middle"],
      [x + 10, y - 24, "start"], [x - 10, y - 24, "end"], [x + 10, y + 32, "start"], [x - 10, y + 32, "end"],
      [x + 18, y + 4, "start"], [x - 18, y + 4, "end"], [x + 10, y - 38, "start"], [x - 10, y - 38, "end"]];
    if (hIn) label("â", "qv-mlab", around(hx, hy), true);
    if (bIn) label("true best", "lab", around(bx, by), true);
    const ends = ys => [[X(u0) + 5, Y(ys[k0]) - 7, "start"], [X(u0) + 5, Y(ys[k0]) + 16, "start"],
      [X(100) - 5, Y(ys[I100]) - 7, "end"], [X(100) - 5, Y(ys[I100]) + 16, "end"],
      [X(u0) + 5, Y(ys[k0]) - 21, "start"], [X(u0) + 5, Y(ys[k0]) + 30, "start"],
      [X(100) - 5, Y(ys[I100]) - 21, "end"], [X(100) - 5, Y(ys[I100]) + 30, "end"],
      [X(u0) + 5, Y(ys[k0]) - 35, "start"], [X(u0) + 5, Y(ys[k0]) + 44, "start"]];
    label("critic", "lab", ends(yq), false);
    label("true value", "lab", ends(yp), false);
    // hover and keyboard layer
    const cross = el("g", { visibility: "hidden" }, svg);
    const cl = el("line", { class: "qv-cross", y1: m.t, y2: H - m.b }, cross);
    const dp = el("circle", { class: "qv-dp", r: 4.5 }, cross), dq = el("circle", { class: "qv-dq", r: 4.5 }, cross);
    const hit = el("rect", { class: "qv-hit", x: m.l, y: 0, width: W - m.l - m.r, height: H - m.b }, svg);
    Object.assign(M1, { E, t, X, Y, yq, yp, cross, cl, dp, dq, hit, W, H, m, k0 });
    const kAt = ev => {
      const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) / r.width * W;
      const u = u0 + (sx - m.l) / (W - m.l - m.r) * (100 - u0);
      return Math.max(k0, Math.min(N - 1, Math.round(u / 2.5)));
    };
    hit.addEventListener("pointermove", ev => showMain(kAt(ev), "ptr", ev));
    hit.addEventListener("pointerdown", ev => showMain(kAt(ev), "ptr", ev));
    hit.addEventListener("pointerleave", ev => { if (ev.pointerType !== "touch") hideMain(); });
    $("qv-y1").textContent = (rel
      ? "Vertical axis: each curve's value minus its own value at the actor's command, so both are 0 at the actor's throttle. Higher is better."
      : "Vertical axis: the value itself, the discounted return expected from this state. Higher is better; the gap between the curves is the critic's error.") + offNote +
      (u0 ? " To see every throttle from 0% upstream, choose 0% to 100%." : " To see the values around the 80% limit up close, choose 60% to 100%.");
    svg.setAttribute("aria-label", "State " + t + ": the critic rates " + fu(U[E.hat[t]]) + " highest; the true best of the sweep is " +
      fu(U[E.best[t]]) + "; the actor gives thruster 1 " + thr(a1) + ". Focus the chart and use the left and right arrow keys to read the values throttle by throttle.");
    const lg = $("qv-lg-ball");
    if (lg) { lg.hidden = E.c !== "gap"; lg.classList.toggle("qv-off", E.c === "gap" && !sl); }
    if (S.h1) showMain(Math.max(k0, S.h1.k), S.h1.mode, null);
  }
  function showMain(k, mode, ev) {
    const { E, t, X, Y, yq, yp, cross, cl, dp, dq } = M1;
    S.h1 = { k, mode, ev: ev ? { x: ev.clientX, y: ev.clientY } : (S.h1 && S.h1.ev) };
    const x = X(U[k]);
    cl.setAttribute("x1", x); cl.setAttribute("x2", x);
    dq.setAttribute("cx", x); dq.setAttribute("cy", Y(yq[k]));
    dp.setAttribute("cx", x); dp.setAttribute("cy", Y(yp[k]));
    cross.setAttribute("visibility", "visible");
    const tip = $("qv-tip1"), qa = E.qa[t], pa = E.pa[t], q = E.qS[t][k], p = E.pS[t][k];
    tip.replaceChildren();
    tipText(tip, "qv-th", "Thruster 1 at " + fu(U[k]) + " upstream" + (k > I80 ? ", past the stall limit" : k === I80 ? ", the stall limit" : ""));
    tipRow(tip, "--accent", fv(q), "critic, Q(<i>s</i>, <i>a</i>)");
    tipRow(tip, "--ink", fv(p), "true value, Q<sup>π</sup>(<i>s</i>, <i>a</i>)");
    tipRow(tip, null, sg(q - p, 2), "critic minus true value");
    const notes = [];
    if (S.view === "rel") notes.push("Change from the actor's command: critic " + sg(q - qa, 3) + ", true value " + sg(p - pa, 3) + ".");
    if (k === E.hat[t]) notes.push("â: the critic rates this throttle highest.");
    if (k === E.best[t]) notes.push("The true best of the sweep.");
    if (inBall(E, t, U[k])) notes.push("Inside the removed region: this critic never saw a command there.");
    if (notes.length) tipText(tip, "qv-tx", notes.join(" "));
    const svg = $("qv-svg1");
    if (mode === "ptr" && S.h1.ev) {
      const r = svg.getBoundingClientRect();
      placeTip(tip, $("qv-c1"), svg, x, (S.h1.ev.y - r.top) * (M1.W / r.width));
    } else placeTip(tip, $("qv-c1"), svg, x, Math.min(Y(yq[k]), Y(yp[k])));
  }
  function hideMain() {
    S.h1 = null;
    if (M1.cross) M1.cross.setAttribute("visibility", "hidden");
    $("qv-tip1").hidden = true;
  }

  /* ---------- the whole episode ---------- */
  const M2 = {};
  function drawEp() {
    const E = cur(), svg = $("qv-svg2"), box = $("qv-c2");
    const W = widthOf(box), narrow = W < 520, H = narrow ? 228 : 250;
    const m = { l: narrow ? 56 : 60, r: 12, t: 12, b: 42 };
    const T = E.T, ys = E.qa.concat(E.pa);
    const tk = niceTicks(Math.min(...ys), Math.max(...ys), narrow ? 4 : 5);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    const { X, Y } = frame(svg, W, H, m, [0, T - 1], [tk.lo, tk.hi], [0, 20, 40, 60, 80, T - 1], tk.v, v => String(v), v => fx(v, tk.d));
    el("text", { class: "axlab", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, svg, "state t");
    E.flag.forEach(t => el("line", { class: "qv-flag", x1: X(t), x2: X(t), y1: H - m.b - 1, y2: H - m.b - 10 }, svg));
    el("path", { class: "qv-lp", d: pathOf(E.pa.map((v, t) => [t, v]), X, Y) }, svg);
    el("path", { class: "qv-lq", d: pathOf(E.qa.map((v, t) => [t, v]), X, Y) }, svg);
    const cg = el("g", {}, svg);
    const cline = el("line", { class: "qv-cur", y1: m.t, y2: H - m.b }, cg);
    const cp = el("circle", { class: "qv-dp", r: 4.5 }, cg), cq = el("circle", { class: "qv-dq", r: 4.5 }, cg);
    const cross = el("g", { visibility: "hidden" }, svg);
    const xl = el("line", { class: "qv-cross", y1: m.t, y2: H - m.b }, cross);
    const xp = el("circle", { class: "qv-dp", r: 4 }, cross), xq = el("circle", { class: "qv-dq", r: 4 }, cross);
    const hit = el("rect", { class: "qv-hit", x: m.l - 6, y: 0, width: W - m.l - m.r + 12, height: H - m.b }, svg);
    Object.assign(M2, { E, X, Y, W, H, m, cline, cp, cq, cross, xl, xp, xq });
    const tAt = ev => {
      const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) / r.width * W;
      return Math.max(0, Math.min(T - 1, Math.round((sx - m.l) / (W - m.l - m.r) * (T - 1))));
    };
    hit.addEventListener("pointermove", ev => showEp(tAt(ev), "ptr", ev));
    hit.addEventListener("pointerdown", ev => showEp(tAt(ev), "ptr", ev));
    hit.addEventListener("click", ev => { setPlaying(false); setT(tAt(ev), true); showEp(tAt(ev), "ptr", ev); });
    hit.addEventListener("pointerleave", ev => { if (ev.pointerType !== "touch") hideEp(); });
    moveCursor();
    if (S.h2) showEp(S.h2.t, S.h2.mode, null);
  }
  function moveCursor() {
    const { E, X, Y, cline, cp, cq } = M2;
    if (!E) return;
    const x = X(S.t);
    cline.setAttribute("x1", x); cline.setAttribute("x2", x);
    cq.setAttribute("cx", x); cq.setAttribute("cy", Y(E.qa[S.t]));
    cp.setAttribute("cx", x); cp.setAttribute("cy", Y(E.pa[S.t]));
  }
  function showEp(t, mode, ev) {
    const { E, X, Y, cross, xl, xp, xq } = M2;
    S.h2 = { t, mode, ev: ev ? { x: ev.clientX, y: ev.clientY } : (S.h2 && S.h2.ev) };
    const x = X(t);
    xl.setAttribute("x1", x); xl.setAttribute("x2", x);
    xq.setAttribute("cx", x); xq.setAttribute("cy", Y(E.qa[t]));
    xp.setAttribute("cx", x); xp.setAttribute("cy", Y(E.pa[t]));
    cross.setAttribute("visibility", "visible");
    const tip = $("qv-tip2");
    tip.replaceChildren();
    tipText(tip, "qv-th", "State " + t + (t === S.t ? ", on screen above" : ""));
    tipRow(tip, "--accent", fv(E.qa[t]), "critic, Q(<i>s</i>, π(<i>s</i>))");
    tipRow(tip, "--ink", fv(E.pa[t]), "true value, Q<sup>π</sup>(<i>s</i>, π(<i>s</i>))");
    tipRow(tip, null, sg(E.qa[t] - E.pa[t], 2), "critic minus true value");
    let note = "Water's push " + fx(E.push[t], 3) + "; thruster 1 " + thr(E.cmd[t][0]) + ".";
    if (E.flag.has(t)) note += " Flagged: the critic rates 100% above 80% here, and the true value rates it below.";
    if (mode === "ptr" && t !== S.t) note += " Click or tap to show this state above.";
    tipText(tip, "qv-tx", note);
    const svg = $("qv-svg2");
    if (mode === "ptr" && S.h2.ev) {
      const r = svg.getBoundingClientRect();
      placeTip(tip, $("qv-c2"), svg, x, (S.h2.ev.y - r.top) * (M2.W / r.width));
    } else placeTip(tip, $("qv-c2"), svg, x, Math.min(Y(E.qa[t]), Y(E.pa[t])));
  }
  function hideEp() {
    S.h2 = null;
    if (M2.cross) M2.cross.setAttribute("visibility", "hidden");
    $("qv-tip2").hidden = true;
  }

  /* ---------- readouts, the sentence, the episode's numbers ---------- */
  const set = (id, text) => { const n = $(id); if (n && n.textContent !== text) n.textContent = text; };
  function readouts() {
    const E = cur(), t = S.t, a1 = E.cmd[t][0], hat = E.hat[t], best = E.best[t];
    set("qv-tl", "state " + t);
    set("qv-ro-c", fx(E.sit.current, 3));
    set("qv-ro-push", fx(E.push[t], 3));
    set("qv-ro-a1", thr(a1) + (Math.abs(a1) > 0.8 ? ", stalls" : ""));
    set("qv-ro-r", fx(E.rew[t], 3));
    set("qv-ro-hat", fu(U[hat]) + (hat > I80 ? ", stalls" : ""));
    set("qv-ro-best", fu(U[best]) + (best > I80 ? ", stalls" : ""));
    set("qv-ro-err", sg(E.qa[t] - E.pa[t], 2));
    set("qv-ro-cost", lossText(E, t));
    set("qv-ro-ret", s4(E.ret));
    set("qv-ro-pts", s4(E.pts));
    set("qv-ro-sa", s4(E.sit.stall_aware_from_start));
    set("qv-ro-rnd", s4(E.sit.random_from_start));
  }
  function sentence() {
    const E = cur(), t = S.t, a1 = E.cmd[t][0], hat = E.hat[t], best = E.best[t];
    const d = E.qa[t] - E.pa[t];
    let h = "<b>State " + t + ".</b> The actor gives thruster 1 " + thr(a1) + (Math.abs(a1) > 0.8 ? ", past the stall limit" : "") + ". ";
    if (hat === best) h += "The critic and the true value agree on the best of the swept throttles: " + fu(U[hat]) + ". ";
    else h += "The critic rates " + fu(U[hat]) + " highest (â), but the true best of the sweep is " + fu(U[best]) +
      "; taking â instead would lose " + lossText(E, t) + " in true value. ";
    h += "At the actor's own command the critic's value is " + fv(E.qa[t]) + " and the true value " + fv(E.pa[t]) +
      ": the critic is " + fx(Math.abs(d), 2) + (d >= 0 ? " too high." : " too low.");
    if (E.c === "gap" && E.inball[t]) h += " This command lies inside the removed region: no training step of this critic had a command there.";
    const n = $("qv-now");
    if (n.innerHTML !== h) n.innerHTML = h;                      // numbers and fixed text only
  }
  function epNumbers() {
    const E = cur(), s = E.summ, T = E.T;
    set("qv-ep-h", "This episode in numbers: " + QL[E.q] + " data, the critic trained on " + CL[E.c] + ", " + SL[E.s]);
    const li = [];
    // a count of states or steps out of the episode's T, in words that cannot be misread: "none of", "all", "n of the"
    const of = (n, noun, bold) => { const b = v => bold ? "<b>" + v + "</b>" : String(v);
      return n === 0 ? "none of the " + T + " " + noun + "s" : n === T ? b("all " + T) + " " + noun + "s" :
        b(n) + " of the " + T + " " + noun + "s"; };
    li.push("The critic's top-rated throttle â is past 80% at " + of(s.a_hat_past80_states, "state", true) +
      ", and exactly 100% at " + of(s.a_hat_at_100_states, "state") + ".");
    li.push("The true best of the sweep is past 80% at " + of(s.best_true_past80_states, "state", true) + ", and exactly 80% at " +
      of(s.best_true_at_80_states, "state") + ".");
    li.push("The critic rates 100% above 80% at " + of(s.critic_higher_at_100_than_80_states, "state", true) +
      "; the true value is lower at 100% than at 80% at " + of(s.truth_lower_at_100_than_80_states, "state") + ".");
    const g = s.critic_rises_while_truth_falls_states;
    if (g > 0) li.push("At the " + g + " state" + (g === 1 ? "" : "s") + " where both happen (flagged in the lower chart), the critic's value rises by " +
      s3(s.critic_change_80_to_100_where_truth_falls_mean) + " from 80% to 100% on average, while the true value falls by " +
      s3(-s.truth_change_80_to_100_where_critic_rises_mean) + ".");
    else li.push("At no state does the critic rate 100% above 80% while the true value rates it below.");
    li.push("Taking â instead of the true best of the sweep would cost " + s3(s.true_cost_of_a_hat_mean) +
      " in true value on average, and at most " + s3(s.true_cost_of_a_hat_max) + ".");
    li.push("At the actor's own command, the critic's value minus the true value is " +
      (s.critic_minus_truth_at_actor_command_mean > 0 ? "+" : "") + s3(s.critic_minus_truth_at_actor_command_mean) + " on average.");
    li.push("The actor itself gives thruster 1 more than 80% at " + of(s.actor_thruster1_past80_steps, "step") + ".");
    $("qv-ep-list").innerHTML = li.map(x => "<li>" + x + "</li>").join("");
  }

  /* ---------- tables ----------
     section.html gives every table its rows already, with a dash in each cell, so that the page's own table tooltips
     (each cell shows its whole row; one Tab stop per table; the arrow keys move between cells) prepare these tables when
     the page loads. The functions here change only the cells' text and the highlighted row, never the rows or cells
     themselves, so those tooltips keep working. */
  function fill(table, rows) {                  // rows: [{ cells: [text, ...], sel: true or false }]
    const tb = table.tBodies[0], n = rows[0].cells.length;
    if (tb.rows.length !== rows.length || [...tb.rows].some(tr => tr.cells.length !== n)) {
      // not the rows section.html gave (an edited copy of it): build them; they then work without the page's tooltips
      const head = table.tHead.rows[table.tHead.rows.length - 1].cells;
      tb.replaceChildren(...rows.map(() => {
        const tr = document.createElement("tr");
        for (let j = 0; j < n; j++) { const c = document.createElement("td"); if (head[j] && head[j].classList.contains("n")) c.className = "n"; tr.appendChild(c); }
        return tr;
      }));
    }
    rows.forEach((r, i) => {
      const tr = tb.rows[i];
      tr.classList.toggle("qv-sel", !!r.sel);
      // the page's card layout (tables with class stack) wraps a cell's text in one span.cv: write inside it
      r.cells.forEach((text, j) => { const c = tr.cells[j], v = c.querySelector(":scope > .cv") || c; if (v.textContent !== text) v.textContent = text; });
    });
  }
  function sweepTable() {
    const det = $("qv-tables");
    if (!det || !det.open) return;
    const E = cur(), t = S.t, a1 = E.cmd[t][0], uAct = -a1 * 100;
    set("qv-cap-sweep", "Thruster 1's swept throttles in state " + t + " (" + QL[E.q] + " data, the critic trained on " + CL[E.c] + ", " + SL[E.s] + ")");
    const rows = [];
    U.forEach((u, k) => {
      const q = E.qS[t][k], p = E.pS[t][k], notes = [];
      if (k === E.hat[t]) notes.push("â");
      if (k === E.best[t]) notes.push("true best");
      if (k === I80) notes.push("stall limit");
      if (inBall(E, t, u)) notes.push("removed region");
      rows.push({ u, sel: false, cells: [fu(u), fv(q), fv(p), sg(q - p, 2), notes.join(", ")] });
    });
    rows.push({ u: uAct, sel: true, cells: [thr(a1), fv(E.qa[t]), fv(E.pa[t]), sg(E.qa[t] - E.pa[t], 2), "the actor's command"] });
    rows.sort((x, y) => x.u - y.u || (x.sel ? 1 : -1));
    fill($("qv-tbl-sweep"), rows);
  }
  function epTable() {
    const E = cur(), rows = [];
    set("qv-cap-ep", "Every state of the episode, at the actor's command (" + QL[E.q] + " data, the critic trained on " + CL[E.c] + ", " + SL[E.s] + ")");
    for (let t = 0; t < E.T; t++) {
      const a1 = E.cmd[t][0];
      rows.push({ sel: t === S.t, cells: [String(t), fx(E.push[t], 3), a1 <= 0 ? pc1(-a1 * 100) : pc1(a1 * 100) + " down",
        fx(E.rew[t], 3), fv(E.qa[t]), fv(E.pa[t]), sg(E.qa[t] - E.pa[t], 2), fu(U[E.hat[t]]), fu(U[E.best[t]])] });
    }
    fill($("qv-tbl-ep"), rows);
  }
  function epTableMark() {
    [...$("qv-tbl-ep").tBodies[0].rows].forEach((tr, i) => tr.classList.toggle("qv-sel", i === S.t));
  }
  function qualTable() {
    const how = {
      random: "Every command uniformly random.",
      "medium-replay": "Each recorded episode replaces part of the recording controller's commands with random ones; the share differs from episode to episode, spread evenly from 65% to 100%.",
      medium: "The recording controller, with 65% of its commands replaced by random ones.",
      "medium-expert": "Each recorded episode is medium or expert, with equal chance.",
      expert: "The recording controller alone."
    };
    fill($("qv-tbl-qual"), D.qualities.map(q => {
      const r = D.train[q];
      return { sel: false, cells: [QL[q], how[q] + " Random commands in the training steps: " + share(r.random_share_training) + ".",
        grp(r.full.rows, 0) + " / " + grp(r.gap.rows, 0), share(r.full.at80) + " / " + share(r.gap.at80),
        share(r.full.past80) + " / " + share(r.gap.past80)] };
    }));
  }
  function ctxTable() {
    set("qv-cap-ctx", "In the " + SL[S.s] + " (the situation chosen above): the share of the 1,000 states visited by each critic's own actor (first ten starts, 100 states each).");
    const get = (q, c) => D.ctx.find(x => x.quality === q && x.critic === c && x.situation === S.s);
    fill($("qv-tbl-ctx"), D.qualities.map(q => {
      const f = get(q, "full"), g = get(q, "gap");
      return { sel: q === S.q, cells: [QL[q], share(f.critic_higher_at_100_than_80_share), share(g.critic_higher_at_100_than_80_share),
        share(f.critic_rises_while_truth_falls_share), share(g.critic_rises_while_truth_falls_share),
        share(f.actor_thruster1_past80_share), share(g.actor_thruster1_past80_share)] };
    }));
  }
  function findings() {
    const by = s => D.ctx.filter(c => c.situation === s), st = by("strong"), tr = by("training");
    const rng = (rows, k) => [Math.min(...rows.map(r => r[k])), Math.max(...rows.map(r => r[k]))];
    const pr = (rows, k) => { const r = rng(rows, k); return share(r[0]) + " to " + share(r[1]); };
    const tc = rng(st, "truth_change_80_to_100_mean");
    const lower = D.qualities.filter(q => q !== "expert");
    const p80 = lower.flatMap(q => [D.train[q].full.past80, D.train[q].gap.past80]);
    const expertNone = D.train.expert && D.train.expert.full.past80 === 0 && D.train.expert.gap.past80 === 0;
    const lv = rng(D.summ, "critic_minus_truth_at_actor_command_mean");
    // the lead finding, worded from the data: how the critic's value changes from 80% to 100% against the true value,
    // in each row of the strong mix (a dataset quality and a critic)
    const CK = "critic_change_80_to_100_mean", TK = "truth_change_80_to_100_mean";
    const lessThanHalf = st.every(r => r[TK] < 0 && r[CK] > r[TK] / 2), less = st.every(r => r[CK] > r[TK]);
    const chg = q => st.filter(r => r.quality === q).map(r => r[CK]);
    const rising = D.qualities.filter(q => chg(q).every(v => v > 0)), falling = D.qualities.filter(q => chg(q).every(v => v < 0));
    const mixed = D.qualities.filter(q => !rising.includes(q) && !falling.includes(q));
    const better = rising.length > 0 && rising.length < D.qualities.length && rising.every((q, i) => q === D.qualities[D.qualities.length - rising.length + i]);
    const names = qs => "the " + (qs.length < 2 ? qs.join("") : qs.slice(0, -1).join(", ") + " and " + qs[qs.length - 1]) + " data";
    const span = (qs, f) => { const v = qs.flatMap(chg).map(f); return s3(Math.min(...v)) + " to " + s3(Math.max(...v)); };
    const parts = [];
    if (falling.length) parts.push("falls by " + span(falling, v => -v) + " with " + names(falling));
    if (mixed.length) parts.push("changes by " + span(mixed, v => v) + " with " + names(mixed));
    if (rising.length) parts.push("rises by " + span(rising, v => v) + " with " + names(rising));
    const head = less ? "Past 80%, the critic's value falls " + (lessThanHalf ? "much " : "") + "less than the true value" +
      (better ? ", and with the better datasets it even rises." : ".") : "Past 80%, the critic's value and the true value change differently.";
    const li = [
      "<b>" + head + "</b> Over the first ten starts of the strong test mix (1,000 states for each critic, visited by its own actor), from 80% to 100% the true value falls by " +
        s3(-tc[1]) + " to " + s3(-tc[0]) + " on average, depending on the dataset quality and the critic" +
        (lessThanHalf ? "; the critic's value never falls by even half as much. " : ". ") +
        "The critic's value " + (parts.length > 1 ? parts.slice(0, -1).join(", ") + " and " + parts[parts.length - 1] : parts.join("")) + ". " +
        "The critic rates 100% above 80% at " + pr(st, "critic_higher_at_100_than_80_share") + " of the states, while the true value is lower at 100% than at 80% at " +
        pr(st, "truth_lower_at_100_than_80_share") + " of them.",
      "<b>Removing the 80% hold is not what causes this.</b> The critic trained on all the data misjudges 100% as well; compare the two columns of each pair in the table below. Past 80%, both critics saw only random commands: " +
        share(Math.min(...p80)) + " to " + share(Math.max(...p80)) + " of their training steps for the four lower qualities" +
        (expertNone ? ", and none for expert." : "."),
      "<b>In the training mix it is rarer.</b> There the critic rates 100% above 80% at " + pr(tr, "critic_higher_at_100_than_80_share") +
        " of the states, and the actor gives thruster 1 more than 80% at " + pr(tr, "actor_thruster1_past80_share") + " of the steps, against " +
        pr(st, "actor_thruster1_past80_share") + " in the strong test mix.",
      "<b>The critic's level is off as well.</b> At the actor's own command, the critic's value minus the true value averages " +
        sg(lv[0], 2) + " to " + sg(lv[1], 2) + " over an episode, across the " + D.summ.length + " episodes here. That is why the upper chart shows, by default, each curve's change from its own value at the actor's command: the two shapes, and where each one peaks, can then be compared."
    ];
    $("qv-find").innerHTML = li.map(x => "<li>" + x + "</li>").join("");
  }

  /* ---------- controls ---------- */
  const playBtn = $("qv-play"), slider = $("qv-t");
  function setT(t, user) {
    const E = cur();
    S.t = Math.max(0, Math.min(E.T - 1, t | 0));
    if (Number(slider.value) !== S.t) slider.value = String(S.t);
    $("qv-now").setAttribute("aria-live", user && !S.timer ? "polite" : "off");
    drawStrip(); readouts(); drawMain(); moveCursor(); sentence(); sweepTable(); epTableMark();
    if (S.h2 && S.h2.mode === "key") showEp(S.t, "key", null);
  }
  function setPlaying(on) {
    if (on === !!S.timer) return;
    if (on) {
      if (S.t >= cur().T - 1) setT(0, false);
      S.timer = setInterval(() => {
        if (S.t >= cur().T - 1) { setPlaying(false); return; }
        setT(S.t + 1, false);
      }, 450);
    } else { clearInterval(S.timer); S.timer = 0; }
    playBtn.textContent = on ? "Pause" : "Play";
    playBtn.setAttribute("data-tip-title", on ? "Pause" : "Play");
    playBtn.setAttribute("data-tip", on ? "Pause the episode on the state on screen." :
      "Play the episode state by state from the state on screen, to state 99. Press again to pause.");
  }
  function onEpisode() {
    const E = cur();
    if (!E) return;
    S.t = Math.min(S.t, E.T - 1);
    slider.max = String(E.T - 1);
    drawEp(); epNumbers(); epTable(); ctxTable();
    setT(S.t, false);
  }
  root.querySelectorAll(".qv-seg button").forEach(b => b.addEventListener("click", () => {
    const attr = ["data-qv-q", "data-qv-c", "data-qv-s", "data-qv-v", "data-qv-r"].find(a => b.hasAttribute(a));
    const k = { "data-qv-q": "q", "data-qv-c": "c", "data-qv-s": "s", "data-qv-v": "view", "data-qv-r": "range" }[attr];
    const v = b.getAttribute(attr);
    root.querySelectorAll("[" + attr + "]").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    if (S[k] === v) return;
    S[k] = v;
    if (k === "view" || k === "range") drawMain(); else onEpisode();
  }));
  playBtn.addEventListener("click", () => setPlaying(!S.timer));
  $("qv-back").addEventListener("click", () => { setPlaying(false); setT(S.t - 1, true); });
  $("qv-fwd").addEventListener("click", () => { setPlaying(false); setT(S.t + 1, true); });
  slider.addEventListener("input", () => { setPlaying(false); setT(Number(slider.value), true); });
  $("qv-tables").addEventListener("toggle", sweepTable);

  // keyboard: the upper chart reads throttle by throttle, the lower chart moves through the states.
  // Esc closes a chart's readout first, and only the readout (inside the walkthrough pop-out too, as the page's own
  // tooltips do); with no readout showing, Esc goes on to the page and closes the pop-out.
  const svg1 = $("qv-svg1"), svg2 = $("qv-svg2");
  let escHid = false;
  function escape(ev, tipId, hideFn) {
    if ($(tipId).hidden) return;
    hideFn();
    ev.preventDefault(); ev.stopPropagation();
    escHid = true; setTimeout(() => { escHid = false; }, 0);
  }
  window.addEventListener("cancel", ev => { if (escHid) { ev.preventDefault(); ev.stopImmediatePropagation(); } }, true);
  svg1.addEventListener("keydown", ev => {
    const lo = M1.k0 || 0, kc = S.h1 ? S.h1.k : Math.max(lo, cur().hat[S.t]);
    let k = null;
    if (ev.key === "ArrowRight") k = kc + 1; else if (ev.key === "ArrowLeft") k = kc - 1;
    else if (ev.key === "Home") k = lo; else if (ev.key === "End") k = N - 1;
    else if (ev.key === "Escape") { escape(ev, "qv-tip1", hideMain); return; }
    if (k == null) return;
    ev.preventDefault();
    showMain(Math.max(lo, Math.min(N - 1, k)), "key", null);
  });
  svg1.addEventListener("focus", () => { if (svg1.matches(":focus-visible")) showMain(S.h1 ? S.h1.k : Math.max(M1.k0 || 0, cur().hat[S.t]), "key", null); });
  svg1.addEventListener("blur", hideMain);
  svg2.addEventListener("keydown", ev => {
    let t = null;
    if (ev.key === "ArrowRight") t = S.t + 1; else if (ev.key === "ArrowLeft") t = S.t - 1;
    else if (ev.key === "PageUp") t = S.t + 10; else if (ev.key === "PageDown") t = S.t - 10;
    else if (ev.key === "Home") t = 0; else if (ev.key === "End") t = cur().T - 1;
    else if (ev.key === "Escape") { escape(ev, "qv-tip2", hideEp); return; }
    if (t == null) return;
    ev.preventDefault();
    setPlaying(false);
    setT(t, true);
    showEp(S.t, "key", null);
  });
  svg2.addEventListener("focus", () => { if (svg2.matches(":focus-visible")) showEp(S.t, "key", null); });
  svg2.addEventListener("blur", hideEp);
  // a tap elsewhere closes a chart's readout on a touch screen
  document.addEventListener("pointerdown", ev => {
    if (ev.pointerType !== "touch") return;
    if (!$("qv-c1").contains(ev.target)) hideMain();
    if (!$("qv-c2").contains(ev.target)) hideEp();
  }, true);

  // redraw at a new width (the window, or the walkthrough pop-out); pause when the panel leaves the screen
  let lastW = 0, rz = 0;
  const redraw = () => {
    const w = $("qv-panel").clientWidth;
    if (!w || w === lastW) return;
    lastW = w;
    drawStrip(); drawMain(); drawEp();
  };
  if (window.ResizeObserver) new ResizeObserver(() => { cancelAnimationFrame(rz); rz = requestAnimationFrame(redraw); }).observe($("qv-panel"));
  else window.addEventListener("resize", () => { cancelAnimationFrame(rz); rz = requestAnimationFrame(redraw); });
  if (window.IntersectionObserver) new IntersectionObserver(es => { es.forEach(e => { if (!e.isIntersecting) setPlaying(false); }); }).observe($("qv-panel"));
  root.addEventListener("walkthrough:leave", () => { setPlaying(false); hideMain(); hideEp(); });
  root.addEventListener("walkthrough:enter", () => { lastW = 0; requestAnimationFrame(redraw); });

  qualTable(); findings();
  onEpisode();
  lastW = $("qv-panel").clientWidth;
  window.qvState = () => ({ q: S.q, c: S.c, s: S.s, t: S.t, view: S.view, range: S.range, playing: !!S.timer });
  window.qvSet = o => {                                    // for tests and screenshots: pick an episode and a state
    ["q", "c", "s", "view", "range"].forEach(k => {
      if (o[k] == null) return;
      const attr = { q: "data-qv-q", c: "data-qv-c", s: "data-qv-s", view: "data-qv-v", range: "data-qv-r" }[k];
      const b = root.querySelector("[" + attr + '="' + o[k] + '"]');
      if (b) b.click();
    });
    if (o.t != null) setT(o.t, false);
    return window.qvState();
  };
})();
