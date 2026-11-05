BCA is the visual system of the BCA project: Bayesian Conformal Aggregation, a caution signal for offline reinforcement learning, and the test environments built to check it. It was drawn for long reference pages that explain an experiment to a careful reader (the ThrusterChain environment page is its fullest use), and it carries over to the project's research interface, its slide decks and its figures. It reads like a lab notebook: warm gray-green paper, near-black ink, one blue for the interface, and a small set of colors that each mean exactly one thing.

## Content fundamentals

- **Plain words first, the term second.** "Big starting pushes (initial-state shift)", "the critic, which scores every command in every state". Define every term at its first use, in the sentence or in a `Tooltip` that starts with the term: "An episode is one run of 100 steps from one starting state."
- **Sentence case** for every heading. A heading names the thing ("Transition Dynamics", "The recorded data") or asks the question the section answers ("Is our WBCP code right?"). A finding's title is the finding: "Placement is almost the whole effect".
- **Full sentences** in captions, notes and legends. A caption says how to read the figure: "Dot: mean over seeds 0–4."
- **Every number with its scale and its sample.** "Points in strong current, stalling world (0 = recording controller, 100 = stall-aware controller)"; "1,000 fixed starts"; "seeds 0–4"; the date a test stopped.
- **Keep three things apart in every result, and never let one stand in for another** (the `Finding`):
  - *The verdict*: the predefined test's outcome under its own rule, in its own words, stated once, never softened and never re-scored with a rule written afterwards ("Stopped under the pre-registered rule (12 Oct), before any BCA run"). It is a statement about that test, not about the idea: an inconclusive or failed verdict is not a verdict on the research, and it does not end it.
  - *What the data show*: every tested value and arm, read on its own terms, including patterns the rule did not score ("Perfectly placed, the pull beat the three controls by −1.9 to +1.1 points"). A setting that helps in every seed of one world and hurts in every seed of another is a finding about when the method helps, and it is stated even when the verdict is inconclusive.
  - *Open questions and next steps*: hypotheses named as hypotheses, each with the run or analysis that would test it, and what is not shown yet ("Not yet shown: any effect on return").
  - The boundaries keep the goalposts still, not the questions closed: a new claim gets its own rule, written before its runs, and the old verdict stays exactly as it was. Exploratory work carries the `exploratory` tag, which marks its standing, not its worth.
- **Voice.** Mostly third person about the system ("The recording controller made the data"); "we" for the authors' choices ("What we tested"). American spelling. No exclamation marks, no emoji, no marketing adjectives.
- **Characters that carry meaning:** a true minus for negatives (−14.3), × for products and sizes, → after a link that goes somewhere ("The recorded data →"), ✓ for a step already seen, · between the parts of a slide kicker. STALLED is the one word set in capitals.

## Color

**One meaning per color, everywhere on a page.** Text stays in the ink tokens; a color marks *which thing*, and never appears without its label (a `Key`, a legend or a direct label). The shape of a key says the kind of thing, the color says which one: dot = a test situation, line = a world, a controller or a learned part, tile = a region, hatched tile = data removed, ring = a float or no signal, dashed = no throttle.

| meaning | tokens | key |
|---|---|---|
| Text | `ink` (body, notes, captions), `ink-2` (secondary), `muted` (short labels only) | — |
| The interface: buttons, focus, selection | `accent`, `accent-soft`; links `accent-ink` | — |
| The recorded data and recorded conditions (in distribution) | `sit-rec`, `sit-rec-soft` | dot |
| Strong current · big waves · big starting pushes (the three shifts) | `sit-strong`, `sit-waves`, `sit-kicks` and their `-soft` tints | dot |
| The stall, and nothing else | `k-stall` (marks), `warn` (its text), `k-stall-soft` / `warn-soft` | line, tile |
| Push keeps rising (no-stall world, rising-thrust controller) | `k-rise`, `k-rise-soft` | line |
| Full throttle (an area) | `k-region` outlined and labeled in `k-region-ink`, `k-region-soft` / `region` | tile |
| The learned parts: critic, actor, BCA's band | `k-learn`, `k-learn-soft` | line; the band is two lines with the tint between |
| The reference (stall-aware controller, the 100 of the points scale) | `k-ref` (ink) | line |
| No throttle | `k-none` | dashed line |
| Pass | `ok`, `ok-soft` | — |

- Set text in `ink` on every surface and tint. `ink-2` is for secondary text; `muted` only for axis ticks, readout names and chip text, never a sentence. Measured floors: `ink` 13.3:1 (light) and 9.8:1 (dark) on its worst ground, `ink-2` 10.4 and 8.4, `muted` 5.9 and 5.4.
- `accent` fills controls and draws focus; it is never running text. Links are `accent-ink`, and in dark they fall to 6.3:1 on a tint, so keep links off tinted rows.
- Red means the stall and nothing else: a stop rule, a failed check or a kill rule is a decision and stays neutral (`Callout`'s `.box.stop`, a ring, a plain tag) with its words saying why.
- `k-region` is an area color (2.2:1 on white by design): draw it as a tint or a tile, always with a 1px `k-region-ink` edge and a label, never as a thin line.
- The situation colors are marks (3.0 to 4.2:1), never text, and always sit beside their label. They are categorical: when a section orders the situations by difficulty, the order goes in the text and the table, not in a ramp.
- A pipeline step borrows the color of what it acts on (`step-collect`, `step-remove`, `step-train`, `step-band`).
- Charts of runs use their own sets: `lq-1` to `lq-5`, a sequential blue for data quality from random to expert; `lc-*` for learning curves, one hue family per kind of run; `toy-c1` to `toy-c4` plus `toy-ref` for the toy notebooks; `m-ref`, `m-rawq`, `m-bca`, `m-dist`, `m-native` for the research interface's method comparisons. Use at most three hues plus a gray reference in one chart, and keep an entity's hue and marker the same in every chart. `dr-hi` and `dr-lo` split a dot-and-range chart into the arms it is about and the rest.
- `m-bca` is violet like `k-rise`. On a page that also draws the two worlds, draw BCA in `k-learn`.
- Both themes are first-class: every token has a dark value, chosen for the dark surface, not inverted. The `deck-` colors are the exception: slides and figures have one look.

## Type

- **Bricolage Grotesque** (`--font-display`, 600 and 700, tracked −0.01em) for headings and titles only: `ds-title` once per page, `ds-section` with a `rule` hairline 8px below, `ds-subsection`, and the small display styles for card, strip and legend-group titles.
- **Source Sans 3** (`--font-body`) for everything read: `tx-body` 17px/1.6 for running text, `tx-note` 15px for notes and table cells, `tx-caption` and `tx-small` for captions, tooltips and legends. Inside the reading column (`.wrap`) running text is justified and hyphenated at desktop widths and left-aligned at 600px and below; cells, code, legends, buttons and tooltips are always left-aligned.
- **JetBrains Mono** (`--font-mono`) for every number a reader compares (`td.n`, readout values, chart ticks), right-aligned with tabular figures, and for code.
- Headings balance their lines. A word in a formula, a code word, a number or a colored mention never hyphenates.
- The fonts are Google-hosted; `components/bundle.css` imports them.

## Layout and spacing

- A reading column and a 200px contents rail (`rail`) on the right, at most 1200px (`content-max`) with a 20px gutter (`space-20`) and 48px between them (`space-48`). Below 1000px (`bp-rail`) the contents become a row of pills above the column. From 1240px (`bp-dock`) the navigation can dock left (`nv-w`) and "In this section" right (`nv-side-w`).
- `space-56` between sections, `space-14` between blocks inside a section, `space-8` to `space-12` inside boxes. Boxes pad `space-14` by `space-16`; panels `space-16`.
- Tables and wide charts scroll sideways inside their own box, never the page; the hint "Wider than the screen: scroll the table sideways →" appears when they do, and the first column stays in view.
- At 560px (`bp-phone`) and below: options one per row, legends one column, boxed text left-aligned. Readouts go two per row at 640px (`bp-readouts`).

## Surfaces, borders, radii and shadows

- Flat paper: `bg` behind everything; `surface` for boxes, tables, panels, charts and tooltips; `surface-2` for table headers, group rows and code. Controls inside a panel (options, readouts) sit on `bg`.
- Borders, not shadows: 1px `rule` around and between things, `rule-strong` for axes and stronger edges. Only what floats casts a shadow (`shadow-tooltip`, `shadow-tip`). The research interface lifts its section cards with `shadow-raised`; the reference page does not.
- `radius-8` is the default for every box, table, card, button and chart frame; `radius-12` for the live panel, `radius-6` for code blocks, `radius-4` for inline code, `radius-pill` for tags and chips.
- No gradients, no glows, no colored left-border cards except where a thing's color is its edge (an option, a situation's group row, a pipeline step's bar).

## States and interaction

- **Hover:** the border turns `accent`, or the thing's own color on an `Option`.
- **Selected:** the thing's tint, its color as the border plus a 1px inset, and a small uppercase "selected" label. Text stays `ink`.
- **Focus:** a 2px solid `accent` outline, 2px offset, on every focusable thing (rows of a dot-and-range chart get a 2px inset instead).
- **Tooltips:** a dotted `muted` underline marks every term, column name and cell that has one; one shared tooltip explains it; a "Tooltips" switch turns them off. The cell being read takes `accent-soft`.
- **Touch:** lesson controls are at least 44px tall on coarse pointers.
- **Motion:** none, apart from the tooltip's 120ms fade; under `prefers-reduced-motion` that goes too. In forced-colors mode keys and step bars keep their colors.

## Charts

- Every chart is a `figure.chartbox` with a legend of `Key`s or direct labels, and a caption that says how to read it; where the numbers matter a `Table` gives them.
- Axes `rule-strong`, gridlines `rule`, ticks mono 12.5px in `muted`, axis titles in `ink-2` naming the scale's ends.
- Data lines `stroke-data` (2.6) in the color of what they show; controllers `stroke-ref` (2.4); guides `stroke-guide` (1.2), dashed `dash-guide`. The 80% stall line is `k-stall`; the 0.565 full-throttle threshold is `k-region-ink`, dashed `dash-threshold`. Points keep a 2px `surface` ring.
- Labels sit on the lines in `ink-2` with a 4px halo of `surface`, so a line never runs through text.
- A variant of the same thing is dashed with hollow markers; a de-emphasized series is `dr-lo` gray, never a lighter tint of a category.

## Iconography

There is no icon set and no logo. The project's name is set in Bricolage Grotesque. Every mark is drawn in CSS by the `Key` component (dot, line, dash, dotted line, tile, hatched tile, ring, four-quarter dot), so marks inherit the theme. Arrows and checks are text characters (→, ✓). Drawings of the environment (floats, thrusters, water) are inline SVG styled by the same tokens. No emoji, no pictograms.

## Components

- The components are React 18, on `window.BCA` from `components/bundle.js`, loaded after React, `tokens.css` and `components/bundle.css`. Each renders the reference page's own markup and classes, so the same HTML with `bundle.css` alone gives the same look without the behavior.
- Their behavior is the page's: options select one thing per question; a term, a column name or a dot-and-range row explains itself on hover and on keyboard focus; a chart reads every line at one point under the pointer or the arrow keys; a table that outgrows its box says so; a legend folds away. Everything that responds to the pointer also responds to the keyboard, and Escape closes what opened.
- Name colored things only through `sit`, `world` and `k` (or an option's `group` and `value`), never with a color: the component picks the color, its tint and its key shape.
- Props and examples are in each component's guide and in `components/index.d.ts`.

## Slides and figures

- Slides are 1920 by 1080 with 128px margins; nothing under 24px. Navy slides (`deck-ink`, text `deck-on-ink`) open the deck, each part and each question; paper slides (`deck-paper`, text `deck-ink` and `deck-soft`) carry the content. See `DeckSlide`.
- Slide type: IBM Plex Sans (`--font-deck-sans`) for text, Source Serif 4 (`--font-deck-serif`) for titles (`dk-cover`, `dk-title`, `dk-heading`), JetBrains Mono for code (`dk-code` on the `deck-code` panel).
- `deck-blue` is the method working as intended (WBCP, Q1); `deck-orange` is the failure being explained (uniform BQ-CP, min(Q1, Q2)); `deck-muted` is de-emphasized data. Orange is for lines and fills only; orange-toned text is `deck-caution`.
- Figures (matplotlib, `experiments/wbcp/viz/style.py`): 16.64 by 7.0 inches (16.64 by 6.4 under a one-line slide headline) saved at 200 dpi on `deck-paper`; IBM Plex Sans 19pt with 17pt ticks (22 and 20pt for single-message slide charts); lines 3pt (4pt on slides); top and right spines off, no grid, legends without frames; axes and ticks `deck-soft`, labels and the target line `deck-ink`. One idea per chart, direct labels, no title (the slide carries it).

## Not synced

- **Variables left in the stylesheet, not tokens:** `--hatch` (a gradient) and the per-element `--c`, `--c-soft`, `--bar`, `--sit` and `--world`, all in `components/bundle.css`.
- **Legacy aliases not carried:** `--c-rec`, `--c-sa`, `--c-rt`, `--c-none`, the weighting test's `--iw-*`, and the navigation's layout offsets (`--nv-bar-space`, `--nv-dock-l`).
- **The research interface (`docs/thrusterchain/site/site.css`):** its neutrals, `accent`, `warn` (amber there), `water`, `stall`, `good`, `bad` and `focus` share names with the reference page's tokens but have different values; the reference page's values are used. Only its method series came across, renamed `m-*`.
- **Older surfaces:** the mechanism explainer (`docs/bca_mechanics`, Georgia serif with works/caveat/fails verdict badges) and the OOD demo (`docs/ood-demo`) predate the system and are not part of it, nor are the per-analysis matplotlib palettes in `analysis/`.
- **Fonts:** Google-hosted, no files in the repository; none fetched.
- **Components:** the repository has no component library. The React components in `components/bundle.js` are hand-written from the reference page's markup (`page_build/wf7/build/core_v27.html`) over its own CSS classes (`components/bundle.css`); React 18.3.1 is carried in `components/lib/`. Two pieces are new, since the page draws them differently: the chart's crosshair reading (`.xhair`, `.xdot`, `.chart-read`) and the Transport's Step and Reset buttons. Not included: the live simulation panel, the critic viewer (`qviewer`), the step-by-step lesson (`ls-*`), the page navigation (`nv-*`), the walkthrough dialog and the stall-chart widget.
