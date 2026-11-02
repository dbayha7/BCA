# Chart

An SVG chart drawn to its box and styled only by classes: thin neutral axes, mono ticks, thick data lines in the colors of what they show, and labels written on the lines.

**Use for** any line or point chart on a page. Every chart sits in a `figure.chartbox` with a legend (or direct labels) and a caption, and where the numbers matter a `Table` gives them.

**Markup**

```html
<figure class="chartbox">
  <svg class="chart" viewBox="0 0 560 304">
    <line class="grid" …/> <line class="axis" …/> <text class="tick" …>0.2</text>
    <line class="knee k80" …/>                <!-- the 80% stall line -->
    <path class="ln-stall" d="…"/> <circle class="pt-stall" …/>
    <text class="lab" …>Stalling world</text>
  </svg>
  <div class="legend">…</div>
  <figcaption>…</figcaption>
</figure>
```

- Frame: `surface`, `rule` border, `radius-8`, `space-10` padding, at most 600px wide (`.wide`: 720px).
- `.grid` in `rule`, `.axis` in `rule-strong` (`stroke-axis`); `.tick` mono 12.5px `muted`; `.axlab` 13px `ink-2`; `.ann` 12px `muted`.
- Data lines `stroke-data` (2.6): `.ln-stall` `k-stall`, `.ln-nostall` `k-rise`, `.ln-dose` `k-learn`. Controllers `stroke-ref` (2.4): `.ln-sa` `k-ref`, `.ln-rt` `k-rise`. `.ln-cost` dotted `ink-2` (`dash-cost`), `.ln-cap2` dashed `k-none` (`dash-cap`), `.ln-ep` a faint `rule-strong` trace.
- Guides: `.knee` dashed `muted` (`dash-guide`); `.knee.k80` the 80% stall line in `k-stall`; `.k565` the full-throttle threshold in `k-region-ink` (`dash-threshold`); `.shade-region` in `region`.
- Points keep a `surface` ring. Full-throttle points (`.pt-ft`) are gold outlined in `k-region-ink`; removed points (`.pt-removed`) are hollow ink rings.
- `.lab`: direct labels in `ink-2` on a 4px halo of `surface`, so a line never runs through them.
- Research-interface method series: `.ln-ref`, `.pt-rawq`, `.pt-bca`, `.pt-dist`, `.pt-native` in the `m-` colors.
