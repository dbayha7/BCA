# SymbolLegend

The key under a formula or a drawing: every symbol in the left column, every meaning starting from one left edge, across all groups.

**Use for** any formula, drawing or chart whose symbols a reader must look up. Put it right after the thing it explains, often inside a `details` that is open on desktop.

**Markup**

```html
<div class="symkey">
  <p class="symkey-h">What the symbols mean</p>
  <dl>
    <div><dt><i>d</i><sub><i>j</i></sub></dt><dd>float j's distance from rest …</dd></div>
  </dl>
  <p class="symkey-g">Also new in this step</p>
  <dl>…</dl>
  <p class="symkey-ex"><b>Worked example.</b> …</p>
</div>
```

- The box is a two-column grid; each `dl` and each row are subgrids, so every definition aligns across groups and rows keep their meaning for assistive technology.
- `.symkey-h` 12.5px `ink-2`, 0.07em tracking; `.symkey-g` group headings in the display face 15px/700 with a `rule-strong` line above; rows split by `rule`.
- `.symkey-ex`: a worked example on `surface-2`, `radius-6`. A `StepChip` inside it gets the surface and a rule.
- At 560px and below it becomes one column, each definition indented 12px under its symbol.
- A symbol may carry a `Key` before it.
