# Readout

A row of named values under a live figure: the name in small text, the value in mono.

**Use for** the numbers a live figure is showing right now (the state, the water's push, a throttle, a reward). Four to a row, two at 640px and below.

**Markup**

```html
<div class="readouts">
  <div class="ro"><span class="k">state t (counted from 0)</span><span class="v">37</span></div>
  …
</div>
```

- `.ro`: `bg` ground, `radius-8`, 7px by 10px. Its name `.k` is 12.5px in `ink-2` (names are often sentences); its value `.v` is mono 16px with tabular figures in `ink`.
- A value that enters a meaningful range may take that thing's ink, never a mark color (the full-throttle region's value turns `k-region-ink`, never `k-region`).

**Don't** use a readout for a result: results go in a `Table` or a `Finding`.
