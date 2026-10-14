"""Finding G of the weighting test: learning curves for every setup (5 data qualities x 3 tests), from the reruns with
checkpoints (bca-shift-work/curves). Re-runnable: everything sits between <!-- lc:begin --> and <!-- lc:end -->, which
is replaced on a second run (the data and the text are refreshed). Also adds the table's column descriptions (COLS)
and a navigation link, once.

    python scripts/p5_curves_apply.py PAGE DATA.json [OBSERVATIONS.html]
OBSERVATIONS.html: the "What the curves show" text (written after the data are complete); a placeholder otherwise.
"""
import json
import re
import sys
from pathlib import Path

page, data_path = Path(sys.argv[1]), Path(sys.argv[2])
obs = Path(sys.argv[3]).read_text(encoding="utf-8") if len(sys.argv) > 3 else None
s = page.read_text(encoding="utf-8")
data = data_path.read_text(encoding="utf-8")
json.loads(data)
assert "</script" not in data

N_DONE = None
d = json.loads(data)
missing = d.get("missing", [])

STATUS = ("Asked by the project owner on 14 October 2026: learning curves for every setup. Every curve reruns a training "
          "run of this test with checkpoints added, and every checkpoint is evaluated exactly.")
if missing:
    STATUS += f" <b>Partial: {len(missing)} of the runs are still computing.</b>"

OBS = obs or """<p class="note">What the curves show will be written once every run has finished.</p>"""

SECTION = f"""<!-- lc:begin -->
    <div class="finding" id="iw-curves">
      <p class="f-title">G. Learning curves for every setup</p>
      <p class="stepnote"><span class="stepchip" data-step="train">Step 3</span><span class="stepchip" data-step="test">Step 5</span>
        The actor's return while it trains, for each data quality and each test.</p>
      <p class="status"><span class="tag acc">exploratory</span> {STATUS}</p>
      <dl class="kv kv1 pro">
        <dt>How the curves were made</dt><dd>Each run of this test was trained again exactly as before, with the actor
          saved along the way, and every saved actor was scored like the final ones: the exact 100-step return from the
          same 1,000 fixed starts of each test. A rerun that has an earlier final actor must end at that actor bit for
          bit, and the table says whether it does. Two kinds of training are shown. <b>Actor only</b>: 20,000 steps
          from the actor fitted to the recorded commands, against the frozen critic, saved every 1,000 steps; this is
          how every lower edge and the perfect lower edge train. <b>Full TD3+BC</b>: 100,000 updates of critic and actor
          together, saved every 5,000; this is how plain TD3+BC and BCA's capped pull train. The arms weighted for the
          actor's commands ran on the three lower qualities for the first time here, with expectations written before
          and no pass marks.</dd>
        <dt>How to read them</dt><dd>Pick a quality and a test, or click a cell of the overview. Each line is the mean
          over seeds 0–4; "Spread across seeds" shades the lowest to highest seed. Lines of one colour belong together:
          grey is no extra caution and the untargeted controls, green is BCA's band as version 5 used it, blue is the
          band weighted for the actor's commands, black is the perfect lower edge, built from the critic's exact error.
          The buttons above each chart turn lines on and off.</dd>
      </dl>
      <div class="panel lc" id="lc-panel">
        <h4>Every setup at once: actor-only training</h4>
        <p class="note">Rows are data qualities, columns are tests; the y-axis is points and differs between cells.
          Click a cell to open it below.</p>
        <div class="lc-gridwrap"><div class="lc-grid" id="lc-grid" role="group" aria-label="Every setup"></div></div>
        <div class="iw-controls lc-controls">
          <div class="iw-seg" role="group" aria-label="Data quality" id="lc-q"></div>
          <div class="iw-seg" role="group" aria-label="Test" id="lc-t"></div>
          <div class="iw-seg" role="group" aria-label="Measure" id="lc-m"></div>
          <label class="lc-chk" for="lc-spread"><input type="checkbox" id="lc-spread"> Spread across seeds</label>
        </div>
        <div class="iw-chart" id="lc-c1">
          <h4 id="lc-h1">Actor only</h4>
          <div class="lc-legend" id="lc-leg1" role="group" aria-label="Lines of the actor-only chart"></div>
          <svg class="chart" id="lc-svg1" role="img" aria-label="Learning curves, actor-only training"></svg>
        </div>
        <div class="iw-chart" id="lc-c2">
          <h4 id="lc-h2">Full TD3+BC</h4>
          <div class="lc-legend" id="lc-leg2" role="group" aria-label="Lines of the full-training chart"></div>
          <svg class="chart" id="lc-svg2" role="img" aria-label="Learning curves, full TD3+BC training"></svg>
        </div>
        <div class="tbl"><table class="grouped stack p5-lc" id="lc-table">
          <caption id="lc-cap">Each run on the chosen quality and test, mean over seeds 0–4.</caption>
          <thead><tr><th>Run</th>
            <th class="n" data-label="First checkpoint">First checkpoint</th>
            <th class="n" data-label="Last checkpoint">Last checkpoint</th>
            <th class="n" data-label="Last checkpoint, lowest to highest seed">Lowest to highest seed</th>
            <th class="n" data-label="Change during training">Change</th>
            <th class="n" data-label="Ends at the saved actor">Ends at the saved actor</th></tr></thead>
          <tbody id="lc-tbody"></tbody>
        </table></div>
      </div>
      {OBS}
    </div>
<script type="application/json" id="lc-data">{data}</script>
<style>
#iw-curves .lc {{ --lc-none: var(--muted); --lc-ref: var(--ink); --lc-v5: var(--sit-rec); --lc-cmd: var(--k-learn); }}
#iw-curves .lc-gridwrap {{ overflow-x: auto; margin: 6px 0 14px; }}
#iw-curves .lc-grid {{ display: grid; grid-template-columns: 7.5em repeat(3, minmax(170px, 1fr)); gap: 6px; min-width: 620px; }}
#iw-curves .lc-grid .lc-rowh, #iw-curves .lc-grid .lc-colh {{ font-size: 13px; color: var(--ink); font-weight: 600; align-self: center; }}
#iw-curves .lc-grid .lc-colh {{ text-align: center; }}
#iw-curves .lc-cell {{ border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); padding: 2px; cursor: pointer;
  display: block; width: 100%; font: inherit; color: inherit; }}
#iw-curves .lc-cell[aria-pressed="true"] {{ outline: 2px solid var(--k-learn); outline-offset: 1px; }}
#iw-curves .lc-cell:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
#iw-curves .lc-cell svg {{ display: block; width: 100%; height: auto; }}
#iw-curves .lc-controls {{ margin-top: 4px; }}
#iw-curves .lc-chk {{ display: inline-flex; align-items: center; gap: 6px; font-size: 14px; color: var(--ink); }}
#iw-curves .lc-legend {{ display: flex; flex-wrap: wrap; gap: 6px 8px; margin: 4px 0 6px; }}
#iw-curves .lc-legend button {{ display: inline-flex; align-items: center; gap: 6px; font: inherit; font-size: 13px; color: var(--ink);
  background: var(--surface); border: 1px solid var(--rule); border-radius: 999px; padding: 3px 10px 3px 6px; cursor: pointer; }}
#iw-curves .lc-legend button[aria-pressed="false"] {{ color: var(--muted); border-style: dashed; }}
#iw-curves .lc-legend button[aria-pressed="false"] svg {{ opacity: 0.35; }}
#iw-curves .lc-legend button.lc-more {{ color: var(--k-learn); border-color: var(--k-learn); padding: 3px 12px; }}
#iw-curves .lc-legend button:disabled {{ opacity: 0.5; cursor: default; }}
#iw-curves .lc-legend button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
#iw-curves .iw-chart {{ position: relative; }}
#iw-curves .lc-miss {{ font-size: 13px; color: var(--muted); }}
</style>
<script>
/* finding G: learning curves (data: #lc-data, built from the reruns with checkpoints; nothing here is computed but means,
   lowest and highest over seeds) */
(function () {{
  "use strict";
  const D = JSON.parse(document.getElementById("lc-data").textContent);
  const $ = id => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const QN = {{ "random": "random", "medium-replay": "medium-replay", "medium": "medium", "medium-expert": "medium-expert", "expert": "expert" }};
  const TN = {{ none: "No shift", mild: "Mild test", strong: "Strong test" }};
  const TL = {{ none: "with no shift", mild: "on the mild test", strong: "on the strong test" }};
  const A1 = [
    ["le_none", "No extra caution", "--lc-none", "", 2, true],
    ["C2_untargeted", "Untargeted control", "--lc-none", "2 4", 2, true],
    ["v5_unweighted", "BCA's lower edge, unweighted", "--lc-v5", "", 2, true],
    ["v5_exact_strong", "BCA's lower edge, weighted for the currents", "--lc-v5", "6 4", 2, false],
    ["A2_cmd_strong", "Weighted for the actor's commands, exact", "--lc-cmd", "", 3, true],
    ["A3_estimated", "Weighted for the actor's commands, estimated", "--lc-cmd", "8 4", 2.2, true],
    ["le_oracle", "Perfect lower edge", "--lc-ref", "7 4", 2, true],
    ["B2_cmd_train", "Exact, training mix of currents", "--lc-cmd", "2 3", 1.6, false],
    ["C3_untargeted", "Untargeted control of the estimated", "--lc-none", "6 3", 1.6, false],
    ["cmd_strong", "First test: no value credit for every refusal", "--lc-cmd", "1 3", 1.8, false],
    ["cmd_untargeted", "First test's untargeted control", "--lc-none", "1 3", 1.6, false],
    ["v5_exact_mild", "BCA's lower edge, weighted for the mild test", "--lc-v5", "2 3", 1.6, false],
    ["le_oracle_x1.5", "Perfect lower edge ×1.5", "--lc-ref", "2 3", 1.6, false]
  ].map(a => ({{ id: a[0], name: a[1], c: a[2], dash: a[3], w: a[4], on: a[5], core: a[5] }}));
  const A2 = [
    ["host_gap", "Plain TD3+BC, 80% hold removed", "--lc-none", "", 2, true],
    ["host_full", "Plain TD3+BC, full data", "--lc-ref", "7 4", 2, true],
    ["capped_unweighted", "BCA's capped pull, unweighted", "--lc-v5", "", 2, true],
    ["capped_exact_strong", "BCA's capped pull, weighted per step", "--lc-v5", "6 4", 2, true]
  ].map(a => ({{ id: a[0], name: a[1], c: a[2], dash: a[3], w: a[4], on: a[5], core: true }}));
  const st = {{ q: "expert", t: "strong", m: "pts", spread: false, more: false }};
  function el(tag, attrs, parent, style) {{
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (style) e.setAttribute("style", style);
    if (parent) parent.appendChild(e);
    return e;
  }}
  const fmt = (v, m) => m === "pts" ? v.toFixed(1) : v.toFixed(0);
  const pending = q => (D.missing || []).some(x => x.startsWith(q + "/"));
  const absent = q => pending(q) ? "still computing" : "not run on this quality";
  function series(group, arm, q, t, m) {{
    const a = D[group].arms[arm] && D[group].arms[arm][q];
    if (!a) return null;
    const runs = a[m][t].filter(Boolean);
    if (!runs.length) return null;
    const n = runs[0].length, mean = [], lo = [], hi = [];
    for (let i = 0; i < n; i++) {{
      const v = runs.map(r => r[i]);
      mean.push(v.reduce((x, y) => x + y, 0) / v.length); lo.push(Math.min(...v)); hi.push(Math.max(...v));
    }}
    return {{ mean, lo, hi, seeds: runs.length, match: a.match }};
  }}
  function nice(lo, hi, n) {{
    const span = Math.max(hi - lo, 1e-6), raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
    const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step, t = [];
    for (let v = a; v <= b + step / 2; v += step) t.push(+v.toFixed(6));
    return {{ a, b, t, dec: step >= 1 ? 0 : step >= 0.1 ? 1 : 2 }};
  }}
  function chart(svg, box, group, arms, W, H, opts) {{
    svg.innerHTML = ""; svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);
    const steps = D[group].steps, m = opts.m;
    const S = arms.filter(a => a.on).map(a => [a, series(group, a.id, st.q, st.t, st.m)]).filter(x => x[1]);
    if (!S.length) {{ el("text", {{ class: "lab", x: W / 2, y: H / 2, "text-anchor": "middle" }}, svg).textContent = pending(st.q) ? "Still computing" : "No runs of this kind here"; return; }}
    let lo = Infinity, hi = -Infinity;
    S.forEach(([a, s]) => {{ const L = st.spread ? s.lo : s.mean, U = st.spread ? s.hi : s.mean;
      lo = Math.min(lo, ...L); hi = Math.max(hi, ...U); }});
    const yt = nice(lo, hi, 5), x0 = steps[0] === 0 ? 0 : 0, x1 = steps[steps.length - 1];
    const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    const Y = v => H - m.b - (v - yt.a) / (yt.b - yt.a) * (H - m.t - m.b);
    yt.t.forEach(t => {{ el("line", {{ class: "grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }}, svg);
      el("text", {{ class: "tick", x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }}, svg).textContent = t.toFixed(yt.dec); }});
    el("line", {{ class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }}, svg);
    const xt = group === "actor" ? [0, 5000, 10000, 15000, 20000] : [0, 25000, 50000, 75000, 100000];
    xt.forEach(t => {{ el("text", {{ class: "tick", x: X(t), y: H - m.b + 17, "text-anchor": "middle" }}, svg).textContent = t === 0 ? "0" : (t / 1000) + "k"; }});
    el("text", {{ class: "axlab", x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle" }}, svg).textContent =
      group === "actor" ? (W < 520 ? "training steps" : "actor-only training steps (0 = the actor fitted to the recorded commands)")
        : "TD3+BC updates";
    el("text", {{ class: "axlab", x: 2, y: 11 }}, svg).textContent = st.m === "pts" ? "points" : "return units";
    if (st.spread) S.forEach(([a, s]) => {{
      const up = steps.map((x, i) => X(x).toFixed(1) + "," + Y(s.hi[i]).toFixed(1));
      const dn = steps.map((x, i) => X(x).toFixed(1) + "," + Y(s.lo[i]).toFixed(1)).reverse();
      el("path", {{ d: "M" + up.join("L") + "L" + dn.join("L") + "Z" }}, svg, `fill:var(${{a.c}});opacity:0.13;stroke:none`); }});
    S.forEach(([a, s]) => {{
      const d = steps.map((x, i) => (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(s.mean[i]).toFixed(1)).join("");
      el("path", {{ d }}, svg, `fill:none;stroke:var(${{a.c}});stroke-width:${{a.w}};stroke-linecap:round;stroke-linejoin:round${{a.dash ? ";stroke-dasharray:" + a.dash : ""}}`);
    }});
    // hover: the nearest checkpoint, every line's mean (and range)
    let tip = box.querySelector(".iw-tip");
    if (!tip) {{ tip = document.createElement("div"); tip.className = "iw-tip"; tip.hidden = true; box.appendChild(tip); }}
    const ln = el("line", {{ class: "axis", y1: m.t, y2: H - m.b, visibility: "hidden" }}, svg);
    const dots = S.map(([a]) => el("circle", {{ r: 4, visibility: "hidden" }}, svg, `fill:var(${{a.c}});stroke:var(--surface);stroke-width:1.5`));
    const hit = el("rect", {{ x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }}, svg, "fill:transparent");
    function move(ev) {{
      const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) / r.width * W;
      const v = x0 + (sx - m.l) / (W - m.l - m.r) * (x1 - x0);
      let i = 0; steps.forEach((x, k) => {{ if (Math.abs(x - v) < Math.abs(steps[i] - v)) i = k; }});
      ln.setAttribute("x1", X(steps[i])); ln.setAttribute("x2", X(steps[i])); ln.setAttribute("visibility", "visible");
      const rows = S.map(([a, s], k) => {{ dots[k].setAttribute("cx", X(steps[i])); dots[k].setAttribute("cy", Y(s.mean[i]));
        dots[k].setAttribute("visibility", "visible"); return [a, s.mean[i], s.lo[i], s.hi[i]]; }}).sort((p, q) => q[1] - p[1]);
      tip.textContent = "";
      const h = document.createElement("div"); h.style.fontWeight = "600";
      h.textContent = (group === "actor" ? "Step " : "Update ") + steps[i].toLocaleString("en-US"); tip.appendChild(h);
      rows.forEach(([a, mv, l, u]) => {{ const p = document.createElement("div");
        p.textContent = `${{a.name}}: ${{fmt(mv, st.m)}}` + (st.spread ? ` (${{fmt(l, st.m)}} to ${{fmt(u, st.m)}})` : ""); tip.appendChild(p); }});
      tip.hidden = false;
      const br = box.getBoundingClientRect(); let left = ev.clientX - br.left + 12;
      if (left + tip.offsetWidth > br.width) left = ev.clientX - br.left - tip.offsetWidth - 12;
      tip.style.left = Math.max(0, left) + "px"; tip.style.top = (ev.clientY - br.top + 14) + "px";
    }}
    hit.addEventListener("pointermove", move); hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", () => {{ ln.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); tip.hidden = true; }});
    svg.setAttribute("aria-label", `Learning curves, ${{group === "actor" ? "actor-only training" : "full TD3+BC training"}}, ${{st.q}} data ${{TL[st.t]}}: ` +
      S.map(([a, s]) => `${{a.name}} from ${{fmt(s.mean[0], st.m)}} to ${{fmt(s.mean[s.mean.length - 1], st.m)}}`).join("; "));
  }}
  function swatch(a) {{
    const s = el("svg", {{ width: 26, height: 10, viewBox: "0 0 26 10", "aria-hidden": "true" }});
    el("line", {{ x1: 1, x2: 25, y1: 5, y2: 5 }}, s, `stroke:var(${{a.c}});stroke-width:${{Math.max(2, a.w)}};stroke-linecap:round${{a.dash ? ";stroke-dasharray:" + a.dash : ""}}`);
    return s;
  }}
  function legend(id, group, arms) {{
    const box = $(id); box.textContent = "";
    const extra = arms.filter(a => !a.core && !a.on).length;
    arms.filter(a => a.core || st.more || a.on).forEach(a => {{
      const has = !!series(group, a.id, st.q, st.t, st.m);
      const b = document.createElement("button"); b.type = "button";
      b.setAttribute("aria-pressed", String(a.on && has)); b.disabled = !has;
      b.appendChild(swatch(a)); b.appendChild(document.createTextNode(a.name + (has ? "" : ` (${{absent(st.q)}})`)));
      b.addEventListener("click", () => {{ a.on = !a.on; draw(); }});
      box.appendChild(b);
    }});
    if (extra) {{
      const b = document.createElement("button"); b.type = "button"; b.className = "lc-more";
      b.setAttribute("aria-expanded", String(st.more));
      b.textContent = st.more ? "Fewer lines" : `More lines (${{extra}})`;
      b.addEventListener("click", () => {{ st.more = !st.more; draw(); }});
      box.appendChild(b);
    }}
  }}
  function seg(id, items, key) {{
    const box = $(id); box.textContent = "";
    items.forEach(([v, label]) => {{ const b = document.createElement("button"); b.type = "button"; b.textContent = label;
      b.setAttribute("aria-pressed", String(st[key] === v)); b.addEventListener("click", () => {{ st[key] = v; draw(); }}); box.appendChild(b); }});
  }}
  function table() {{
    const tb = $("lc-tbody"), rows = [];
    [["actor", A1], ["cotrain", A2]].forEach(([g, arms]) => arms.forEach(a => rows.push([g, a])));
    if (!tb.rows.length) rows.forEach(([g, a]) => {{ const tr = tb.insertRow(); for (let k = 0; k < 6; k++) tr.insertCell(); tr.cells[0].textContent = a.name + (g === "cotrain" ? " (full TD3+BC)" : ""); for (let k = 1; k < 6; k++) tr.cells[k].className = "n"; }});
    rows.forEach(([g, a], r) => {{
      const s = series(g, a.id, st.q, st.t, st.m), c = tb.rows[r].cells;
      if (!s) {{ for (let k = 1; k < 6; k++) c[k].textContent = k === 1 ? absent(st.q) : "—"; return; }}
      const n = s.mean.length - 1, mt = s.match.filter(x => x !== null && x !== undefined);
      c[1].textContent = fmt(s.mean[0], st.m); c[2].textContent = fmt(s.mean[n], st.m);
      c[3].textContent = `${{fmt(s.lo[n], st.m)}} to ${{fmt(s.hi[n], st.m)}}`;
      const ch = s.mean[n] - s.mean[0]; c[4].textContent = (ch >= 0 ? "+" : "−") + fmt(Math.abs(ch), st.m);
      c[5].textContent = mt.length ? (mt.every(Boolean) ? `yes, ${{mt.length}} of ${{mt.length}}` : `${{mt.filter(Boolean).length}} of ${{mt.length}}`) : "new run";
    }});
    $("lc-cap").textContent = `Each run on ${{st.q}} data ${{TL[st.t]}}, mean over seeds 0–4, in ${{st.m === "pts" ? "points (0 = uniform random commands, 100 = the stall-aware controller)" : "units of return"}}. Full TD3+BC runs start at their first checkpoint, 5,000 updates.`;
  }}
  function grid() {{
    const g = $("lc-grid"); g.textContent = "";
    const corner = document.createElement("div"); g.appendChild(corner);
    D.laws.forEach(t => {{ const h = document.createElement("div"); h.className = "lc-colh"; h.textContent = TN[t]; g.appendChild(h); }});
    D.qualities.forEach(q => {{
      const h = document.createElement("div"); h.className = "lc-rowh"; h.textContent = QN[q]; g.appendChild(h);
      D.laws.forEach(t => {{
        const b = document.createElement("button"); b.type = "button"; b.className = "lc-cell";
        b.setAttribute("aria-pressed", String(st.q === q && st.t === t));
        b.setAttribute("aria-label", `${{q}} data, ${{TN[t].toLowerCase()}}: open below`);
        const W = 260, H = 120, m = {{ l: 30, r: 6, t: 8, b: 14 }};
        const svg = el("svg", {{ viewBox: `0 0 ${{W}} ${{H}}`, role: "img" }}, b);
        const S = A1.filter(a => ["le_none", "C2_untargeted", "v5_unweighted", "A2_cmd_strong", "A3_estimated", "le_oracle"].includes(a.id))
          .map(a => [a, series("actor", a.id, q, t, "pts")]).filter(x => x[1]);
        if (S.length) {{
          let lo = Infinity, hi = -Infinity; S.forEach(([a, s]) => {{ lo = Math.min(lo, ...s.mean); hi = Math.max(hi, ...s.mean); }});
          const yt = nice(lo, hi, 2), steps = D.actor.steps;
          const X = v => m.l + v / 20000 * (W - m.l - m.r), Y = v => H - m.b - (v - yt.a) / (yt.b - yt.a) * (H - m.t - m.b);
          yt.t.forEach(v => {{ el("line", {{ class: "grid", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }}, svg);
            el("text", {{ class: "tick", x: m.l - 4, y: Y(v) + 4, "text-anchor": "end" }}, svg, "font-size:10px").textContent = v.toFixed(0); }});
          S.forEach(([a, s]) => el("path", {{ d: steps.map((x, i) => (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(s.mean[i]).toFixed(1)).join("") }}, svg,
            `fill:none;stroke:var(${{a.c}});stroke-width:${{Math.min(a.w, 2.2)}};stroke-linecap:round${{a.dash ? ";stroke-dasharray:" + a.dash : ""}}`));
        }} else el("text", {{ class: "lc-miss", x: W / 2, y: H / 2, "text-anchor": "middle" }}, svg, "fill:var(--muted);font-size:12px").textContent = "computing";
        b.addEventListener("click", () => {{ st.q = q; st.t = t; draw(); $("lc-c1").scrollIntoView({{ block: "nearest" }}); }});
        g.appendChild(b);
      }});
    }});
  }}
  const widthOf = svg => Math.max(300, Math.min(900, (svg.parentNode.clientWidth || 700)));
  function draw() {{
    seg("lc-q", D.qualities.map(q => [q, QN[q]]), "q");
    seg("lc-t", D.laws.map(t => [t, TN[t]]), "t");
    seg("lc-m", [["pts", "Points"], ["ret", "Return units"]], "m");
    $("lc-spread").checked = st.spread;
    legend("lc-leg1", "actor", A1); legend("lc-leg2", "cotrain", A2);
    $("lc-h1").textContent = `Actor only, ${{st.q}} data, ${{TL[st.t]}}`;
    $("lc-h2").textContent = `Full TD3+BC, ${{st.q}} data, ${{TL[st.t]}}`;
    const s1 = $("lc-svg1"), s2 = $("lc-svg2"), W1 = widthOf(s1), W2 = widthOf(s2), narrow = W1 < 520;
    const m = {{ l: narrow ? 40 : 48, r: 12, t: 26, b: 36 }};
    chart(s1, $("lc-c1"), "actor", A1, W1, narrow ? 250 : 300, {{ m }});
    chart(s2, $("lc-c2"), "cotrain", A2, W2, narrow ? 230 : 260, {{ m }});
    table(); grid();
  }}
  $("lc-spread").addEventListener("change", e => {{ st.spread = e.target.checked; draw(); }});
  let tm = null; window.addEventListener("resize", () => {{ clearTimeout(tm); tm = setTimeout(draw, 120); }});
  draw();
}})();
</script>
<!-- lc:end -->"""

if "<!-- lc:begin -->" in s:
    a, b = s.index("<!-- lc:begin -->"), s.index("<!-- lc:end -->") + len("<!-- lc:end -->")
    s = s[:a] + SECTION + s[b:]
else:
    anchor = '\n\n    <h3 id="iw-results">What the results mean</h3>'
    assert s.count(anchor) == 1
    s = s.replace(anchor, "\n" + SECTION + anchor)
    nav_old = '  <a href="#iw-return" class="sub">Return from weighting</a>\n'
    assert s.count(nav_old) == 1
    s = s.replace(nav_old, nav_old + '  <a href="#iw-curves" class="sub">Learning curves</a>\n')
    m = re.search(r"const COLS = (\{.*?\});\n", s, flags=re.S)
    src = m.group(1)
    cols = json.loads(src)
    assert json.dumps(cols, ensure_ascii=False, separators=(",", ":")) == src
    cols["weighting-test"].append({"cols": {
        "Run": "Each row is one way of training the actor, as its line in the charts above names it.",
        "First checkpoint": "The mean over seeds at the first saved actor: the actor fitted to the recorded commands for actor-only runs, 5,000 updates for full TD3+BC.",
        "Last checkpoint": "The mean over seeds at the last saved actor: 20,000 steps for actor-only runs, 100,000 updates for full TD3+BC.",
        "Last checkpoint, lowest to highest seed": "The lowest and highest of the five seeds at the last saved actor.",
        "Change during training": "The last checkpoint minus the first.",
        "Ends at the saved actor": "Whether the rerun with checkpoints ends at the very actor the original run saved, bit for bit, on how many seeds. New run: there was no earlier run."}})
    s = s.replace(src, json.dumps(cols, ensure_ascii=False, separators=(",", ":")), 1)
page.write_text(s, encoding="utf-8")
print(f"finding G written ({len(missing)} runs missing); page {len(s)} characters")
