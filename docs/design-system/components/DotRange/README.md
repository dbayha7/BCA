# DotRange

A chart with one row per arm: the dot is the mean over seeds, the line runs from the lowest to the highest seed, and color separates the arms the chart is about from the rest.

**Use for** comparing many arms of one experiment on one scale. It is HTML, not SVG, so long arm names wrap and every row is focusable.

**Markup** (CSS only; the React component renders the same)

```html
<figure class="chartbox drbox"><div class="dr">
  <div class="dr-axis dr-head"><span class="dr-name">Arm</span><span class="dr-val">Mean</span>
    <span class="dr-ticks"><span style="left:0%">−40</span>…</span></div>
  <div class="dr-grp">Starting points</div>
  <div class="dr-row hi" tabindex="0"><span class="dr-name">…</span><span class="dr-val">79.7</span>
    <span class="dr-track"><span class="dr-rng" style="left:80.53%;width:7.67%"></span>
    <span class="dr-dot" style="left:85.47%"></span></span></div>
</div><div class="legend">…</div><figcaption>…</figcaption></figure>
```

- Rows: `.hi` in `dr-hi` (the learned blue, for the arms the chart is about), `.lo` in `dr-lo` (the de-emphasis gray), `.ref` in `ink-2` with a hollow dot (a starting point, or no signal).
- `.dr-grp` labels a group of rows (12px uppercase, `muted`). The axis title goes in a last `.dr-axis.dr-axlab` row and names the scale's ends.
- Gridlines are background images on `.dr-track` in `rule`, the zero line in `rule-strong`.
- The dot keeps a 2px `surface` ring; hover and focus tint the row `surface-2`, focus adds a 2px `accent` inset.
- Always a legend and a caption; a table gives every number.

**React** (`window.BCA`, React 18; hand-written from the page's markup in `page_build/wf7/build/core_v27.html`)

```js
const { DotRange } = window.BCA;
<DotRange min={-40} max={100} ticks={[-40, -20, 0, 20, 40, 60, 80, 100]}
  axisLabel="Points in strong current, stalling world (0 = recording controller, 100 = stall-aware controller)"
  rows={[{ group: 'Starting points' },
         { name: 'Cloned actor (the start of every arm)', mean: 54.4, lo: 47.6, hi: 61.8, kind: 'ref' },
         { name: "Oracle penalty at the actor's command", note: '(main arm)', strong: true, mean: 79.7, lo: 72.7, hi: 83.5, kind: 'hi' }]}
  legend={[{ swatch: 'dr-hi', label: '…' }, { swatch: 'dr-lo', label: '…' }, { ring: true, label: 'no signal' }]}
  caption="Dot: mean over seeds 0–4." />
```

Rows are focusable; hovering or focusing one shows "name: mean; lowest to highest seed: lo to hi", as on the page. Gridlines fall on `ticks`, with zero in `rule-strong`. Values print with a true minus and one decimal unless `format` says otherwise.
