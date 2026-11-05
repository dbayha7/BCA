# Option

A toggle card that names one thing (a situation, a world or a controller), describes it in a line, and wears that thing's color.

**Use for** the choices of a live panel: one group per question, numbered, each with its buttons in a grid underneath. One option per group is pressed.

**Markup** (CSS only; the React component renders the same)

```html
<div class="optgroup">
  <p class="optlbl"><span class="optn">2</span>Which world? <span class="q">The data were recorded in the stalling world.</span></p>
  <div class="optbtns">
    <button class="opt" data-group="world" data-value="stall" aria-pressed="true">
      <b>Stalling world</b><span>Past 80% throttle the propeller stalls …</span></button>
    <button class="opt" data-group="world" data-value="no_stall" aria-pressed="false">…</button>
  </div>
</div>
```

- `data-group` + `data-value` pick the color: `sit` (`recorded`, `strong_current`, `big_waves`, `big_kicks`) draws a dot before the name; `world` (`stall`, `no_stall`) a line; `ctrl` (`recording`, `stall_aware`, `rising_thrust`, `no_throttle`) carries `<span class="sw"></span>` (or `.sw.dash` for no throttle) inside its `<b>`.
- The card's left edge is 4px of the thing's color; hover turns the border that color.
- Pressed (`aria-pressed="true"`): the thing's tint, its color as border plus a 1px inset, and a "SELECTED" label in ink. Text stays in `ink` on every tint.
- Wrap the groups in `.opts` inside a `.panel` (`radius-12`, `surface`).
- An optional `<span class="opt-step">` line says which pipeline step the option stands for, with a `StepChip`.

**Do** keep the name to a few words and the line to one sentence. **Don't** use an option for an action, or color an option that names no thing (it falls back to `rule`/`accent`).

**React** (`window.BCA`, React 18; hand-written from the page's markup in `page_build/wf7/build/core_v27.html`)

```js
const { Panel, OptionGroup, Option } = window.BCA;
<Panel>
  <div className="opts">
    <OptionGroup n={2} question="Which world?" group="world" value={world} onChange={setWorld}
      options={[{ value: 'stall', name: 'Stalling world', description: 'Past 80% throttle the propeller stalls…' },
                { value: 'no_stall', name: 'No-stall comparison world', description: 'Push keeps rising…' }]} />
  </div>
</Panel>
```

`OptionGroup` owns one pressed option (controlled with `value`/`onChange`, or `defaultValue`); `value` must be one of the page's keys (situations, worlds or controllers) so the option wears that thing's color. A single `Option` takes `group`, `value`, `name`, `description`, `selected`, `onSelect`, `dash` and `step`. `KeyValue` is the two-column list the live panel uses to say what is chosen.
