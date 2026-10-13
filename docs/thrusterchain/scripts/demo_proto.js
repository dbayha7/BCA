/* ThrusterChain: the task diagram (#task-diagram) plays one demo episode.
   The episode is real: the live panel's record "stall|recorded|stall_aware" in the data object D (the stall-aware
   controller, the stalling world, the recorded conditions, start 11 of the evaluation bank). Both static drawings,
   the wide and the narrow one, are moved in place, so their labels and classes stay the single source of the picture.
   If anything this needs is missing, the script stops and the static drawing stays as it is.
   Paste as its own <script> block after the script that defines D (top-level const D is shared by classic scripts). */
(function () {
  "use strict";
  const KEY = "stall|recorded|stall_aware", SIT = "recorded";
  const STEP_MS = 160;        // one step of the episode every 0.16 s: 100 steps in 16 s
  const END_HOLD_MS = 1800;   // autoplay rests on the last step before starting again
  const KNEE = 0.8;           // the 80% throttle limit
  const PUSH_FLOOR = 0.5;     // the top arrows are scaled to the episode's largest push, but never past 0.5 per strip
  /* Per drawing (keyed by its class): the band the float centres may use, clear of the labels above and below the
     floats; pixels of thruster arrow per unit of push; and the most pixels per unit of float position. */
  const LAYOUT = {
    "dg-wide": { zone: [121, 215], thrK: 48, unitCap: 70 },
    "dg-narrow": { zone: [126, 216], thrK: 40, unitCap: 70 },
  };
  const NS = "http://www.w3.org/2000/svg";
  const MINUS = "−";

  const num = (e, a) => parseFloat(e.getAttribute(a));
  const set = (e, o) => { for (const k in o) e.setAttribute(k, typeof o[k] === "number" ? o[k].toFixed(1) : o[k]); };
  const show = (e, on) => { e.style.display = on ? "" : "none"; };
  const cls = (e, name, on) => {
    const c = (e.getAttribute("class") || "").split(/\s+/).filter(x => x && x !== name);
    if (on) c.push(name);
    e.setAttribute("class", c.join(" "));
  };
  const hasCls = (e, name) => (e.getAttribute("class") || "").split(/\s+/).indexOf(name) >= 0;
  const pts = e => e.getAttribute("points").trim().split(/\s+/).map(p => p.split(",").map(Number));
  const fmt = (x, d) => { const s = Math.abs(x).toFixed(d); return (x < 0 && Number(s) !== 0 ? MINUS : "") + s; };

  /* the float scale: the whole episode's positions fill the band, rest placed so both extremes fit */
  function floatScale(R, zone, cap) {
    let lo = 0, hi = 0;
    R.s.forEach(x => { for (let j = 0; j < 5; j++) { lo = Math.min(lo, x[2 * j]); hi = Math.max(hi, x[2 * j]); } });
    const room = zone[1] - zone[0], unit = Math.min(cap, room / Math.max(1e-6, hi - lo));
    return { unit, restY: zone[0] + hi * unit + (room - (hi - lo) * unit) / 2 };
  }

  /* read one static drawing: its moving parts, their static geometry, and the labels that ride on the thrusters */
  function bind(svg, R, WAT) {
    const name = (svg.getAttribute("class") || "").split(/\s+/).find(c => LAYOUT[c]);
    if (!name) return null;
    const q = s => Array.from(svg.querySelectorAll(s));
    const v = {
      svg, lay: LAYOUT[name],
      floats: q("circle.dg-float"), nums: q("text.dg-n"), springs: q("line.dg-thrline"), boxes: q("rect.dg-thr"),
      shafts: q("line.dg-thrarrow"), heads: q("polygon.dg-thrarrow"), leaders: q("line.dg-leader"),
      wSolid: q("line.dg-push"), wWave: q("line.dg-wavepart"), wHead: q("polygon.dg-push"),
      chan: svg.querySelector("rect.dg-chan"), rest: svg.querySelector("line.dg-rest"),
      ok: svg.querySelector("rect.dg-ok"), stall: svg.querySelector("rect.dg-stall"),
    };
    if (v.floats.length !== 5 || v.nums.length !== 5 || v.springs.length !== 4 || v.boxes.length !== 4 ||
      v.shafts.length !== 4 || v.heads.length !== 4 || v.wSolid.length !== 5 || v.wWave.length !== 5 ||
      v.wHead.length !== 5 || !v.chan || !v.rest || !v.ok || !v.stall ||
      (v.leaders.length !== 0 && v.leaders.length !== 4)) return null;
    v.numDy = num(v.nums[0], "y") - num(v.floats[0], "cy");
    v.bx = v.boxes.map(b => num(b, "x") + num(b, "width") / 2);
    v.bh = num(v.boxes[0], "height");
    const boxCy0 = v.boxes.map(b => num(b, "y") + v.bh / 2);
    const hp = pts(v.heads[0]);
    v.headH = Math.max(...hp.map(p => p[1])) - Math.min(...hp.map(p => p[1]));
    v.headW = (Math.max(...hp.map(p => p[0])) - Math.min(...hp.map(p => p[0]))) / 2;
    v.leadY2 = v.leaders.map(l => num(l, "y2"));
    // the water's arrows: anchor, longest length (to the static tip) and head size
    v.wx = v.wSolid.map(l => num(l, "x1"));
    v.wy0 = num(v.wSolid[0], "y1");
    const wp = pts(v.wHead[0]);
    v.wTip = Math.max(...wp.map(p => p[1]));
    v.wHeadH = v.wTip - Math.min(...wp.map(p => p[1]));
    v.wHeadW = (Math.max(...wp.map(p => p[0])) - Math.min(...wp.map(p => p[0]))) / 2;
    v.wMax = v.wTip - v.wy0;
    v.wK = (v.wMax - v.wHeadH) / Math.max(PUSH_FLOOR, ...WAT.push.map(Math.abs));   // this episode's largest push fills the strip
    // texts: the sign beside each water arrow, the rest label, and the labels next to each thruster
    const texts = q("text").filter(t => !hasCls(t, "dg-n"));
    v.signs = v.wx.map(x => texts.find(t => Math.abs(num(t, "x") - x - 8) <= 4 && num(t, "y") > v.wy0 && num(t, "y") < v.wTip) || null);
    const restY0 = num(v.rest, "y1"), restX0 = num(v.rest, "x1");
    v.restLab = texts.find(t => restY0 - num(t, "y") > 0 && restY0 - num(t, "y") <= 10 && num(t, "x") <= restX0 + 20) || null;
    v.restLabDy = v.restLab ? num(v.restLab, "y") - restY0 : 0;
    // a thruster's own name, centred under its box, follows the box; any other text placed next to a box in the
    // static drawing was placed for floats that no longer stand there, so it is hidden while the drawing plays
    v.follow = []; v.hidden = [];
    texts.forEach(t => {
      const x = num(t, "x"), y = num(t, "y");
      for (let k = 0; k < 4; k++) {
        if (Math.abs(x - v.bx[k]) <= 40 && Math.abs(y - boxCy0[k]) <= 30) {
          if (Math.abs(x - v.bx[k]) < 1 && t.getAttribute("text-anchor") === "middle") v.follow.push({ t, k, dy: y - boxCy0[k] });
          else v.hidden.push(t);
          break;
        }
      }
    });
    // the throttle bar: thruster 1's throttle fills it, a dark tick marks the push it really gives
    v.x0 = num(v.ok, "x"); v.x1 = num(v.stall, "x") + num(v.stall, "width");
    v.barY = num(v.ok, "y"); v.barH = num(v.ok, "height");
    const sc = floatScale(R, v.lay.zone, v.lay.unitCap);
    v.unit = sc.unit; v.restY = sc.restY;
    v.xs = v.floats.map(c => num(c, "cx")); v.r = num(v.floats[0], "r");
    if (v.restLab) v.restLabDy = restLabelDy(v, R);
    return v;
  }

  /* keep the rest label clear of any float that passes under it: floats move only up and down, so the floats it
     shares columns with, and how far above and below rest they go in the episode, fix a clear slot near the line */
  function restLabelDy(v, R) {
    const t = v.restLab, fs = num(t, "font-size") || 12.5, w = 0.5 * fs * t.textContent.length, x0 = num(t, "x");
    const near = v.xs.map((x, j) => (x + v.r > x0 && x - v.r < x0 + w ? j : -1)).filter(j => j >= 0);
    if (!near.length) return v.restLabDy;
    let top = Infinity, bot = -Infinity;   // the shared floats' centres, in pixels below the line (negative: above)
    R.s.forEach(s => near.forEach(j => { const d = -s[2 * j] * v.unit; top = Math.min(top, d); bot = Math.max(bot, d); }));
    const above = Math.min(v.restLabDy, top - v.r - 1 - 0.2 * fs);    // baseline above every circle's top
    if (above >= -24) return above;
    const below = Math.max(4 + 0.75 * fs, bot + v.r + 1 + 0.75 * fs); // or below every circle's bottom
    if (below <= 24) return below;
    return v.restLabDy;                                                // no clear slot: the halo keeps it readable
  }

  /* change the drawing once, from static to playing: new parts, rest line moved, misplaced labels hidden */
  function arm(v) {
    const doc = v.svg.ownerDocument;
    v.fill = doc.createElementNS(NS, "rect"); v.fill.setAttribute("class", "dg-fill");
    v.tick = doc.createElementNS(NS, "line"); v.tick.setAttribute("class", "dg-ptick");
    set(v.fill, { x: v.x0, y: v.barY, height: v.barH, width: 0 });
    set(v.tick, { y1: v.barY - 3, y2: v.barY + v.barH + 3, x1: v.x0, x2: v.x0 });
    // after the bar's two parts, before its text, so "push = throttle" stays readable on top of the fill
    v.stall.parentNode.insertBefore(v.tick, v.stall.nextSibling);
    v.stall.parentNode.insertBefore(v.fill, v.stall.nextSibling);
    set(v.rest, { y1: v.restY, y2: v.restY });
    if (v.restLab) {
      set(v.restLab, { y: v.restY + v.restLabDy });
      cls(v.restLab, "dg-halo", true);
      v.svg.appendChild(v.restLab);   // drawn last, over a float that passes under it
    }
    v.hidden.forEach(t => show(t, false));
    v.follow.forEach(o => cls(o.t, "dg-halo", true));   // a thruster's name now crosses the rest line and the springs
    v.svg.setAttribute("data-playing", "");
  }

  /* one frame: state t, and u in [0, 1) of the way to state t + 1 (the floats glide; throttles and water hold) */
  function draw(v, R, WAT, t, u) {
    const s0 = R.s[t], s1 = R.s[Math.min(t + 1, R.s.length - 1)];
    const ys = [0, 1, 2, 3, 4].map(j => v.restY - (s0[2 * j] + (s1[2 * j] - s0[2 * j]) * u) * v.unit);
    ys.forEach((y, j) => { set(v.floats[j], { cy: y }); set(v.nums[j], { y: y + v.numDy }); });
    for (let k = 0; k < 4; k++) {
      set(v.springs[k], { y1: ys[k], y2: ys[k + 1] });
      const cy = (ys[k] + ys[k + 1]) / 2, top = cy - v.bh / 2, bot = cy + v.bh / 2, x = v.bx[k];
      const f = R.f[t][k], stalled = R.st[t][k] === "1", L = Math.abs(f) * v.lay.thrK, up = f < 0;
      set(v.boxes[k], { y: top });
      [v.boxes[k], v.shafts[k], v.heads[k]].forEach(e => cls(e, "w", stalled));
      const dir = up ? -1 : 1, base = up ? top : bot, tip = base + dir * L, hh = Math.min(v.headH, L);
      show(v.heads[k], L >= 1.5); show(v.shafts[k], L > hh + 0.5);
      set(v.shafts[k], { x1: x, x2: x, y1: base, y2: tip - dir * (hh - 1) });
      v.heads[k].setAttribute("points", [[x - v.headW, tip - dir * hh], [x + v.headW, tip - dir * hh], [x, tip]]
        .map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" "));
      v.follow.forEach(o => { if (o.k === k) set(o.t, { y: cy + o.dy }); });
      if (v.leaders.length) { set(v.leaders[k], { y1: bot }); show(v.leaders[k], bot < v.leadY2[k] - 2); }
    }
    /* the water's push, the same on every float. The shaft's length is the push: solid for the current, dashed for
       the wave when it adds; the head sits beyond. When the wave takes push away, the shaft is shorter than the
       current alone and a faint dashed piece past the head shows the part taken away. */
    const c = WAT.c, push = WAT.push[t], w1 = push - c, K = v.wK, H = v.wHeadH;
    for (let i = 0; i < 5; i++) {
      const x = v.wx[i], solid = v.wSolid[i], wave = v.wWave[i], head = v.wHead[i];
      let hy, tip, wy1, wy2, faint = false;
      if (push >= 0) {
        hy = v.wy0 + push * K; tip = hy + H;
        const s2 = w1 >= 0 ? v.wy0 + c * K : hy + 1;
        set(solid, { y1: v.wy0, y2: s2 }); show(solid, s2 > v.wy0 + 0.5);
        if (w1 >= 0) { wy1 = v.wy0 + c * K; wy2 = hy + 1; }
        else { wy1 = tip; wy2 = v.wy0 + c * K + H; faint = true; }
      } else {  // the wave takes away more than the current gives: a net push upstream, drawn from the bottom up
        const yb = v.wy0 + v.wMax; hy = yb + push * K; tip = hy - H;
        show(solid, false); wy1 = yb; wy2 = hy - 1;
      }
      set(wave, { y1: wy1, y2: wy2 }); show(wave, Math.abs(wy2 - wy1) > 1.5); cls(wave, "faint", faint);
      head.setAttribute("points", [[x - v.wHeadW, hy], [x + v.wHeadW, hy], [x, tip]].map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" "));
      if (v.signs[i]) v.signs[i].textContent = w1 >= 0 ? "+" : MINUS;
    }
    // thruster 1 on the throttle bar
    const a1 = R.a[t][0], f1 = R.f[t][0], span = v.x1 - v.x0;
    set(v.fill, { width: Math.min(1, Math.abs(a1)) * span }); cls(v.fill, "w", Math.abs(a1) > KNEE + 1e-9);
    const px = v.x0 + Math.min(1, Math.abs(f1)) * span; set(v.tick, { x1: px, x2: px });
  }

  function init(doc, win) {
    const fig = doc.getElementById("task-diagram");
    let data;
    try { data = D; } catch (e) { return null; }   // D is the live panel's data object, defined by the earlier script
    if (!fig || !data || !data.roll || !data.roll[KEY] || !data.water || !data.water[SIT]) return null;
    const R = data.roll[KEY], WAT = data.water[SIT], N = R.a.length;
    const ui = id => doc.getElementById(id);
    const box = ui("td-ctrls"), play = ui("td-play"), back = ui("td-back"), fwd = ui("td-fwd"), slider = ui("td-slider");
    if (!box || !play || !back || !fwd || !slider) return null;
    const views = Array.from(fig.querySelectorAll("svg.dg")).map(s => bind(s, R, WAT)).filter(Boolean);
    if (!views.length) return null;
    views.forEach(arm);
    slider.max = String(N - 1);
    box.hidden = false;

    const RM = win.matchMedia ? win.matchMedia("(prefers-reduced-motion: reduce)") : { matches: false };
    const S = { t: 0, u: 0, playing: false, auto: false, user: false, raf: 0, last: 0, hold: 0 };
    const out = (id, s) => { const e = ui(id); if (e) e.textContent = s; };
    function readouts() {
      const t = S.t, push = WAT.push[t], a1 = R.a[t][0], f1 = R.f[t][0], stalled = R.st[t][0] === "1";
      out("td-step", "step " + (t + 1) + " of " + N);
      out("td-push", push.toFixed(3));
      out("td-pushparts", "current " + WAT.c.toFixed(3) + (push - WAT.c >= 0 ? " + " : " " + MINUS + " ") +
        "wave " + Math.abs(push - WAT.c).toFixed(3));
      out("td-thr", Math.round(Math.abs(a1) * 100) + "%" + (a1 < 0 ? " upstream" : a1 > 0 ? " downstream" : ""));
      out("td-thrpush", Math.abs(f1).toFixed(2) + (stalled ? " stalled" : ""));
      out("td-ret", fmt(R.g[t], 2));
      slider.value = String(t);
      slider.setAttribute("aria-valuetext", "step " + (t + 1) + " of " + N);
    }
    function render() {
      const u = RM.matches ? 0 : S.u;
      views.forEach(v => draw(v, R, WAT, S.t, u));
      readouts();
    }
    function setPlaying(on, auto) {
      S.playing = on; S.auto = !!(on && auto); S.last = 0; S.hold = 0;
      play.textContent = on ? "Pause" : "Play";
      win.cancelAnimationFrame(S.raf);
      if (on) { if (S.t >= N - 1) { S.t = 0; S.u = 0; } S.raf = win.requestAnimationFrame(tick); }
      else { S.u = 0; render(); }
    }
    function tick(now) {
      if (!S.playing) return;
      const dt = S.last ? Math.min(100, now - S.last) : 0; S.last = now;
      if (S.hold > 0) {
        S.hold -= dt;
        if (S.hold <= 0) { S.hold = 0; S.t = 0; S.u = 0; render(); }
      } else {
        S.u += dt / STEP_MS;
        while (S.u >= 1) { S.u -= 1; S.t += 1; }
        if (S.t >= N - 1) {
          S.t = N - 1; S.u = 0;
          if (S.auto) { S.hold = END_HOLD_MS; render(); }
          else { setPlaying(false); return; }
        } else render();
      }
      S.raf = win.requestAnimationFrame(tick);
    }
    const take = () => { S.user = true; setPlaying(false); };
    play.addEventListener("click", () => { S.user = true; setPlaying(!S.playing, false); });
    back.addEventListener("click", () => { take(); S.t = Math.max(0, S.t - 1); render(); });
    fwd.addEventListener("click", () => { take(); S.t = Math.min(N - 1, S.t + 1); render(); });
    slider.addEventListener("input", () => {   // read the value before pausing: pausing redraws and rewrites the slider
      const t = Math.max(0, Math.min(N - 1, Number(slider.value) || 0));
      take(); S.t = t; render();
    });
    // autoplay only while the drawing is on screen, never after the reader takes control, never with reduced motion
    if ("IntersectionObserver" in win) {
      new win.IntersectionObserver(es => {
        const vis = es[es.length - 1].isIntersecting;
        if (vis && !S.user && !RM.matches && !S.playing) setPlaying(true, true);
        else if (!vis && S.playing && S.auto) setPlaying(false);
      }, { threshold: 0.35 }).observe(fig);
    }
    const onRM = () => { if (RM.matches && S.playing && S.auto) setPlaying(false); };
    if (RM.addEventListener) RM.addEventListener("change", onRM); else if (RM.addListener) RM.addListener(onRM);
    render();
    return { S, views, render, setPlaying, N };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { init, bind, arm, draw, floatScale, LAYOUT, KEY, SIT };
  } else if (typeof document !== "undefined") {
    window.taskDemo = init(document, window);   // null when the static drawing is kept
  }
})();
