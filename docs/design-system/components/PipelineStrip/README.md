# PipelineStrip

The experiment's five steps as linked cards, each with a bar in the color of what that step acts on; a compact one-row form shows where a block of the page sits.

**Use for** the top of an experiment page (the full strip, which replaces an at-a-glance summary) and inside a live panel or a section (the compact strip, with the steps that block stands for outlined). `StepChip`s name a step inline.

| step | bar |
|---|---|
| 1 Collect the data | `step-collect` (the recorded data's green) |
| 2 Remove commands | the hatch over half `k-region` (main test) and half `surface-2` (weighting test's 80% hold) |
| 3 Train TD3+BC | `step-train` (the learned parts' blue) |
| 4 Add BCA's band | a band: two `step-band` lines with `k-learn-soft` between |
| 5 Test the actor | the four situation colors in order |

**Markup** (CSS only; the React component renders the same)

```html
<nav class="pipe" aria-label="…"><ol>
  <li class="pipe-step" data-step="collect"><div class="pipe-card">
    <p class="pipe-head"><span class="pipe-n">1</span><span class="pipe-t">Collect the data</span></p>
    <p class="pipe-d">…</p><a class="pipe-go" href="#dataset">The recorded data</a></div></li>
  …</ol></nav>

<nav class="pipe pipe-sm"><ol><li class="pipe-step is-here" data-step="train"><a href="#methods"><span class="pipe-n">3</span>Train</a></li>…</ol></nav>

<span class="stepchip" data-step="band">Step 4</span>
```

- The full strip is a container: from 860px of its own width it lays five columns with the bar on top and arrows across; narrower, one card per row with the bar on the left and arrows down.
- Cards: `surface`, `rule` border, `radius-8`; the title in the display face 16.5px/700; the text 14.5px in ink; an optional `.pipe-why` line under a rule; `.pipe-go` links in `accent-ink` with a →.
- Compact strip: pills (`radius-pill`) with a 3px bar along the bottom; `.is-here` outlines a step in `ink-2`.
- `.stepchip` (12.5px/700 on `surface-2`, with the step's color as a dot); `.stepchip.in` inside a sentence; `data-step="none"` for a setting that stands for no step.

**React** (`window.BCA`, React 18; hand-written from the page's markup in `page_build/wf7/build/core_v27.html`)

```js
const { PipelineStrip, StepChip } = window.BCA;
<PipelineStrip steps={steps} />                                      // five cards
<PipelineStrip steps={steps} compact here={['train', 'band']} />     // pills; the block's steps outlined
<StepChip step="band" />  <StepChip step="band" inline>Step 4</StepChip>
```

A step is `{ step, title, short, text, why, href, go }`; `step` is `collect`, `remove`, `train`, `band` or `test` and sets its bar. The compact strip follows its links unless `selectable`, which makes a click select the step (`here`, `onHereChange`, `onSelect`).
