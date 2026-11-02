# Readout

A row of named values under a live figure: the name in small text, the value in mono.

**Use for** the numbers a live figure is showing right now (the state, the water's push, a throttle, a reward). Four to a row, two at 640px and below.

**Markup** (CSS only; the React component renders the same)

```html
<div class="readouts">
  <div class="ro"><span class="k">state t (counted from 0)</span><span class="v">37</span></div>
  …
</div>
```

- `.ro`: `bg` ground, `radius-8`, 7px by 10px. Its name `.k` is 12.5px in `ink-2` (names are often sentences); its value `.v` is mono 16px with tabular figures in `ink`.
- A value that enters a meaningful range may take that thing's ink, never a mark color (the full-throttle region's value turns `k-region-ink`, never `k-region`).

**Don't** use a readout for a result: results go in a `Table` or a `Finding`.

**React** (`window.BCA`, React 18; hand-written from the page's markup in `page_build/wf7/build/core_v27.html`)

```js
const { Readout } = window.BCA;
<Readout items={[{ name: 'state t (counted from 0)', value: t },
                 { name: "water's push on every float (full-throttle region above 0.565)", value: '0.612', tone: 'region' },
                 { name: 'thruster 1 push', value: '0.50 STALLED', tone: 'stall' }]} />
```

`tone` is `stall` (warn, bold) or `region` (`k-region-ink`). Pass `live` only for values that change on a reader's action, not every step. The preview computes the push from the environment's own rule (`canonical/thrusterchain.py`): equal to the throttle up to 80%, then 0.8 − 1.5 × (throttle − 0.8) in the stalling world.
