# SymbolLegend

The key under a formula or a drawing: every symbol in the left column, every meaning starting from one left edge, across all groups.

**Use for** any formula, drawing or chart whose symbols a reader must look up. Put it right after the thing it explains, often inside a `details` that is open on desktop.

**Markup** (CSS only; the React component renders the same)

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

**React** (`window.BCA`, React 18; hand-written from the page's markup in `page_build/wf7/build/core_v27.html`)

```js
const { SymbolLegend } = window.BCA;
<SymbolLegend collapsible
  groups={[{ items: [{ symbol: <i>j</i>, meaning: 'a float, numbered 1 to 5' }] },
           { title: 'Also new in this step', items: [{ symbol: 'start 0', meaning: '…' }] }]}
  example="At start 0, float 1 has p₁ = −0.3216." />
```

`collapsible` wraps it in a `details` (open unless `defaultOpen={false}`) with the heading as its summary. An item may carry a `key` (a `Key`'s props) before its symbol.
