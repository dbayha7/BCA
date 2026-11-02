# DotRange

A chart with one row per arm: the dot is the mean over seeds, the line runs from the lowest to the highest seed, and color separates the arms the chart is about from the rest.

**Use for** comparing many arms of one experiment on one scale. It is HTML, not SVG, so long arm names wrap and every row is focusable.

**Markup**

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
