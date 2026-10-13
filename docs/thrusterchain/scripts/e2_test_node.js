/* Node test of e2_demo.js (copy of demo_test_node.js) on the real drawings and the real episode, with a tiny DOM stand-in.
   Run: node demo_test_node.js [path to thrusterchain.html]   (reads only; writes nothing)
   Checks, for both drawings, every step and half-steps: no NaN; floats inside the channel and their band; thruster
   arrows inside the channel; water arrows inside their strip; the bar fill inside the bar; float circles clear of
   every fixed label; the thruster names clear of the floats. Prints the facts the caption and plan quote. */
"use strict";
const fs = require("fs"), path = require("path");
const PAGE = process.argv[2] || path.join(__dirname, "..", "thrusterchain.html");
const html = fs.readFileSync(PAGE, "utf8");

/* ---------- a tiny DOM: enough for bind/arm/draw ---------- */
class El {
  constructor(tag, attrs, text) {
    this.tagName = tag; this.attrs = new Map(Object.entries(attrs || {})); this.children = []; this.parentNode = null;
    this._text = text || ""; this.style = { display: "" }; this.ownerDocument = DOC;
  }
  getAttribute(k) { return this.attrs.has(k) ? this.attrs.get(k) : null; }
  setAttribute(k, v) {
    const s = String(v);
    if (/NaN|undefined|Infinity/.test(s)) throw new Error(`bad value ${k}=${s} on <${this.tagName} class=${this.getAttribute("class")}>`);
    this.attrs.set(k, s);
  }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); }
  appendChild(c) { if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.children.push(c); return c; }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentNode = null; return c; }
  insertBefore(c, ref) {
    if (c.parentNode) c.parentNode.removeChild(c);
    c.parentNode = this; const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(c); else this.children.splice(i, 0, c); return c;
  }
  get nextSibling() { const p = this.parentNode; if (!p) return null; return p.children[p.children.indexOf(this) + 1] || null; }
  querySelectorAll(sel) {
    const [tag, cl] = sel.split("."), out = [];
    const walk = e => e.children.forEach(c => {
      if ((!tag || c.tagName === tag) && (!cl || (c.getAttribute("class") || "").split(/\s+/).includes(cl))) out.push(c);
      walk(c);
    });
    walk(this); return out;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}
const DOC = { createElementNS: (ns, tag) => new El(tag) };
const ent = s => s.replace(/&#x27;/g, "'").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
function parseSvg(markup) {
  const open = markup.match(/^<svg([^>]*)>/);
  const attrsOf = a => { const o = {}; a.replace(/([\w:-]+)="([^"]*)"/g, (_, k, v) => { o[k] = ent(v); }); return o; };
  const root = new El("svg", attrsOf(open[1]));
  const re = /<(\w+)([^>]*?)(\/?)>(?:([^<]*)<\/\1>)?/g;
  re.lastIndex = open[0].length;
  let m;
  while ((m = re.exec(markup))) { if (m[1] === "svg") break; root.appendChild(new El(m[1], attrsOf(m[2]), ent(m[4] || ""))); }
  return root;
}

/* ---------- inputs: the two drawings and the data object, as they are in the page ---------- */
const svgs = ["dg dg-wide", "dg dg-narrow"].map(c => {
  const a = html.indexOf(`<svg class="${c}"`), b = html.indexOf("</svg>", a) + 6;
  if (a < 0) throw new Error("missing drawing " + c);
  return parseSvg(html.slice(a, b));
});
const dl = html.slice(html.indexOf("const D = ") + 10, html.indexOf("\n", html.indexOf("const D = ")));
const D = JSON.parse(dl.replace(/;\s*$/, ""));
const P = require("./e2_demo.js");
const R = D.roll[P.KEY], WAT = D.water[P.SIT], N = R.a.length;

/* ---------- checks ---------- */
let fails = 0;
const bad = msg => { fails++; if (fails <= 40) console.log("FAIL", msg); };
const n = (e, a) => parseFloat(e.getAttribute(a));
const textBox = t => {   // approximate: Source Sans 3 averages about 0.5 em per character
  const fs_ = n(t, "font-size") || 12.5, w = 0.5 * fs_ * t.textContent.length, x = n(t, "x"), y = n(t, "y");
  const anc = t.getAttribute("text-anchor") || "start";
  const x0 = anc === "middle" ? x - w / 2 : anc === "end" ? x - w : x;
  return { x0, x1: x0 + w, y0: y - 0.75 * fs_, y1: y + 0.2 * fs_, s: t.textContent };
};
const hit = (b, cx, cy, r) => cx + r > b.x0 && cx - r < b.x1 && cy + r > b.y0 && cy - r < b.y1;
const restLabelTouches = [0, 0];

svgs.forEach((svg, si) => {
  const v = P.bind(svg, R, WAT);
  if (!v) { bad("bind failed for drawing " + si); return; }
  P.arm(v);
  const name = si ? "narrow" : "wide";
  const ch = { y0: n(v.chan, "y"), y1: n(v.chan, "y") + n(v.chan, "height") };
  const r = n(v.floats[0], "r");
  console.log(`${name}: unit ${v.unit.toFixed(1)} px per position unit, rest at y ${v.restY.toFixed(1)} (static ${svg.querySelector("line.dg-rest") ? "moved" : "?"}),`,
    `water ${v.wK.toFixed(1)} px per push unit, follow labels [${v.follow.map(o => o.t.textContent).join(", ")}],`,
    `hidden [${v.hidden.map(t => t.textContent).join(", ")}], rest label ${v.restLab ? JSON.stringify(v.restLab.textContent) + " baseline " + v.restLabDy.toFixed(1) + " px from the line" : "none"},`,
    `signs ${v.signs.filter(Boolean).length}/5`);
  const followSet = new Set(v.follow.map(o => o.t)), hidSet = new Set(v.hidden);
  const fixed = svg.querySelectorAll("text").filter(t => !followSet.has(t) && !hidSet.has(t) && t !== v.restLab &&
    !(t.getAttribute("class") || "").includes("dg-n") && n(t, "y") > ch.y0 && n(t, "y") < ch.y1).map(textBox);
  for (let t = 0; t < N; t++) for (const u of [0, 0.5]) {
    if (t === N - 1 && u) continue;
    try { P.draw(v, R, WAT, t, u); } catch (e) { bad(`${name} t=${t} u=${u}: ${e.message}`); continue; }
    v.floats.forEach((c, j) => {
      const cy = n(c, "cy"), cx = n(c, "cx");
      if (cy - r < ch.y0 || cy + r > ch.y1) bad(`${name} t=${t} float ${j + 1} outside the channel (cy ${cy})`);
      if (cy < v.lay.zone[0] - 0.05 || cy > v.lay.zone[1] + 0.05) bad(`${name} t=${t} float ${j + 1} outside its band (cy ${cy})`);
      fixed.forEach(b => { if (hit(b, cx, cy, r)) bad(`${name} t=${t} float ${j + 1} overlaps label "${b.s}"`); });
      v.follow.forEach(o => { if (hit(textBox(o.t), cx, cy, r)) bad(`${name} t=${t} float ${j + 1} overlaps "${o.t.textContent}"`); });
      if (v.restLab && hit(textBox(v.restLab), cx, cy, r)) restLabelTouches[si]++;
    });
    v.heads.forEach((h, k) => {
      if (h.style.display === "none") return;
      const ps = h.getAttribute("points").split(" ").map(p => p.split(",").map(Number));
      ps.forEach(p => { if (p[1] < ch.y0 || p[1] > ch.y1) bad(`${name} t=${t} thruster ${k + 1} arrow leaves the channel (y ${p[1]})`); });
      const hb = { x0: Math.min(...ps.map(p => p[0])), x1: Math.max(...ps.map(p => p[0])), y0: Math.min(...ps.map(p => p[1])), y1: Math.max(...ps.map(p => p[1])) };
      fixed.concat(v.restLab ? [textBox(v.restLab)] : []).forEach(b => {
        if (hb.x1 > b.x0 && hb.x0 < b.x1 && hb.y1 > b.y0 && hb.y0 < b.y1) bad(`${name} t=${t} thruster ${k + 1} arrow overlaps label "${b.s}"`); });
    });
    v.wHead.forEach((h, i) => h.getAttribute("points").split(" ").forEach(p => { const y = +p.split(",")[1];
      if (y < v.wy0 - 0.05 || y > v.wTip + 0.05) bad(`${name} t=${t} water arrow ${i + 1} leaves its strip (y ${y})`); }));
    v.wSolid.concat(v.wWave).forEach((l, i) => ["y1", "y2"].forEach(a => { const y = n(l, a);
      if (l.style.display !== "none" && (y < v.wy0 - 0.05 || y > v.wTip + 0.05)) bad(`${name} t=${t} water line ${i} leaves its strip (${a} ${y})`); }));
    if (t === 0 && u === 0) { const w = v.wWave[0]; console.log(`  ${name} step 1 water arrow: solid ${v.wSolid[0].getAttribute("y1")}-${v.wSolid[0].getAttribute("y2")}, wave ${w.getAttribute("y1")}-${w.getAttribute("y2")} (${w.style.display || "shown"}), head ${v.wHead[0].getAttribute("points")}`); }
    if (t === 20 && u === 0) { const w = v.wWave[0]; console.log(`  ${name} step 21 (push ${WAT.push[20]}) water arrow: solid ${v.wSolid[0].getAttribute("y1")}-${v.wSolid[0].getAttribute("y2")}, wave ${w.getAttribute("y1")}-${w.getAttribute("y2")} class ${w.getAttribute("class")} (${w.style.display || "shown"}), sign ${v.signs[0] && v.signs[0].textContent}`); }
    const fw = n(v.fill, "width");
    if (fw < 0 || fw > v.x1 - v.x0 + 0.05) bad(`${name} t=${t} fill width ${fw}`);
    if (u === 0 && t === 0) {
      console.log(`  ${name} step 1: floats y [${v.floats.map(c => c.getAttribute("cy")).join(", ")}], thruster boxes y [${v.boxes.map(b => b.getAttribute("y")).join(", ")}],`,
        `water arrow 1 solid ${v.wSolid[0].getAttribute("y1")}-${v.wSolid[0].getAttribute("y2")}, wave ${v.wWave[0].getAttribute("y1")}-${v.wWave[0].getAttribute("y2")}, head ${v.wHead[0].getAttribute("points")},`,
        `bar fill ${v.fill.getAttribute("width")} of ${(v.x1 - v.x0).toFixed(1)}, tick x ${v.tick.getAttribute("x1")}`);
    }
  }
});
console.log(`rest label touched by a float on ${restLabelTouches[0]} wide and ${restLabelTouches[1]} narrow frames of ${2 * N - 1} (drawn with a halo, on top)`);

/* ---------- facts for the caption and plan ---------- */
const dist = (x, j) => Math.hypot(x[2 * j], x[2 * j + 1]);
const a1 = R.a.map(x => x[0]);
const firstAllNear = lim => R.s.findIndex((x, t) => R.s.slice(t, 100).every(y => [0, 1, 2, 3, 4].every(j => dist(y, j) <= lim)));
const maxDistFrom = t0 => Math.max(...R.s.slice(t0, 100).map(x => Math.max(...[0, 1, 2, 3, 4].map(j => dist(x, j)))));
const corr = (xs, ys) => { const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
  let sxy = 0, sxx = 0, syy = 0; xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; syy += (ys[i] - my) ** 2; });
  return sxy / Math.sqrt(sxx * syy); };
console.log("facts:");
console.log("  start: current", WAT.c, "amplitude", WAT.amp, "length of the ten float numbers", WAT.dist,
  "float cost", R.s[0].slice(0, 10).reduce((s, x) => s + x * x, 0).toFixed(3), "evaluation-bank start", D.start);
console.log("  steps (1-based) with thruster 1 at the 80% cap:", a1.map((x, t) => Math.abs(x) >= 0.7999 ? t + 1 : 0).filter(Boolean).join(", "));
console.log("  steps (1-based) with thruster 4 at the 80% cap:", R.a.map((x, t) => Math.abs(x[3]) >= 0.7999 ? t + 1 : 0).filter(Boolean).join(", "));
console.log("  any throttle past 80%:", R.a.some(x => x.some(v => Math.abs(v) > 0.8 + 1e-9)), "stalled thruster-steps:", R.sum.stalled_thruster_steps);
console.log("  largest float distance from rest from step 11 on:", maxDistFrom(10).toFixed(3), "| from step 21 on:", maxDistFrom(20).toFixed(3));
console.log("  first step from which every float stays within 0.3 of rest:", firstAllNear(0.3) + 1, "| within 0.2:", firstAllNear(0.2) + 1);
console.log("  thruster 1 throttle from step 11 on: from", Math.min(...a1.slice(10).map(Math.abs)).toFixed(3), "to", Math.max(...a1.slice(10).map(Math.abs)).toFixed(3));
console.log("  correlation, step 11 on, of thruster 1's upstream throttle with the water's push:", corr(a1.slice(10).map(x => -x), WAT.push.slice(10)).toFixed(3));
console.log("  water's push from", Math.min(...WAT.push), "to", Math.max(...WAT.push), "| steps above 0.565:", WAT.reg.reduce((a, b) => a + b, 0));
console.log("  return: this episode", R.sum.return_env, "| bank average", D.ref.stall.recorded.stall_aware.J, "| reward step 1", R.r[0], "| final float distance", R.sum.final_float_distance);
console.log(fails ? `${fails} FAILURES` : "ALL CHECKS PASSED");
process.exit(fails ? 1 : 0);
