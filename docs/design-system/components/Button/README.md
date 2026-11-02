# Button

One filled button in the accent for the main action of a control row, and a ghost button beside it for the rest.

**Use for** playing, pausing and stepping a live figure, or applying a setting. A choice between things on the page (a situation, a world, a controller) is an `Option`, not a button.

**Markup**

```html
<div class="transport">
  <button class="btn">Play</button>
  <button class="btn ghost">Step</button>
  <input type="range" min="0" max="99" value="37" aria-label="State">
  <span class="tlabel">t = 37</span>
</div>
```

- `.btn`: `accent` fill, text in `surface` (white in light, near-black in dark: 8.0:1 and 6.3:1), 14px/600, `radius-8`, at least 84px wide.
- `.btn.ghost`: no fill, `accent` border, `accent-ink` text.
- `.transport` lays a row of buttons, a range input (in `accent`) and a mono time label (`.tlabel`).
- Focus: 2px solid `accent`, 2px offset.

**Do** write the verb ("Play", "Step", "Reset"). **Don't** put more than one filled button in a row, or use a category color on a button.
