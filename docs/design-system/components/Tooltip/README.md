# Tooltip

One shared box that explains whatever the pointer or focus is on: a term, a column name, a table cell or a control.

**Use for** every term a reader might not know and every column name and cell whose meaning is not obvious. What has a tooltip shows a dotted underline in `muted`; a "Tooltips" switch at the top of the page turns them all off.

**Markup** (the page's script fills and places one `.tt`; terms carry their text)

```html
<i data-tip-title="Episode" data-tip="An episode is one run of 100 steps …">episodes</i>

<div class="tt on" role="tooltip">
  <div class="tt-h">Recording controller</div>
  <div class="tt-b">The fixed controller that made the training data …</div>
  <div class="tt-col"><b>Column:</b> …</div>
</div>
```

- `surface`, a `rule-strong` border, `radius-8`, `shadow-tooltip`; 14px/1.45, at most 380px wide.
- `.tt-h` the title in ink 700; `.tt-b` the text in `ink-2`; `.tt-col` a second part under a `rule`, 13.5px.
- `.tt-tuple` lists a row's values; `.cur` marks the one being read on `accent-soft`.
- The cell being read gets `accent-soft` (`td.tt-act`). Focus shows a 2px `accent` outline.
- It fades in over 120ms, and not at all under reduced motion.

**Do** write the tooltip as a definition in one or two sentences, starting with the term. **Don't** hide anything in a tooltip that the page needs to make its point.

**React** (`window.BCA`, React 18; hand-written from the page's markup in `page_build/wf7/build/core_v27.html`)

```js
const { Tooltip, TooltipBox } = window.BCA;
The <Tooltip title="Episode" text="An episode is one run of 100 steps from one starting state.">episodes</Tooltip> …
<TooltipBox title="Recording controller" text="…" more="Used in: pipeline step 1." />   // a static box, for docs
```

The term gets a dotted underline and a tab stop; on hover or focus the shared box opens below it (above it near the bottom of the screen), is linked with `aria-describedby`, and closes on Escape, blur or pointer leave. `more` adds a second part under a rule.
