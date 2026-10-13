# ThrusterChain pass 4: colour system, ink, running text, legends, pipeline strip

Design only; the page is untouched (md5 `f3736a56dc7698e0b57633acbb91de3c` before and after the mock build).
CSS: `notes/p4_design.css`. Mock: `tmp/p4/design/mock.html` (built by `scripts/p4_design_mock.py`, read-only on the
page). Screenshots: `tmp/p4/design/shots/` (390 px light, 1200 px dark). Measurements and validator runs:
`tmp/p4/design/measure/`.

## Takeaway

- **One meaning per colour, page-wide.** Four situation colours, a stall red, a "push keeps rising" violet, a
  full-throttle gold, and the page's blue for the learned parts (critic, actor, BCA's band). Each pipeline step borrows
  the colour of what it acts on, so the strip, the live panel, the tables and the charts all speak the same code.
- **Shape tells the kind, colour tells which one:** dot = situation, line = world or controller, tile = region,
  hatched tile = data removed. Text always stays in ink, with the mark in front of the word, so colour is never alone.
- **Every categorical set passes the dataviz validator in both modes** (no FAIL). Two documented WARNs: the gold is
  under 3:1 on white (it is an area colour, always labelled and outlined in its own ink), and recorded-conditions green
  against the stall red sits in the colour-blind floor band (6.2 to 7.1; never in the same chart as two series, and
  the stall always reads STALLED).
- **Ink:** body text is at least 9.8:1 and secondary text at least 8.4:1 on every surface in both themes; the muted
  token (short labels only) is at least 5.4:1.
- **Running text** fills the column (all 52ch to 86ch caps lifted), justified with hyphenation; boxed text goes
  left-aligned at phone width, where justification opened wide gaps in the mock.
- **Legends** become one aligned two-column grid per legend (subgrid), with optional group headings, a worked-example
  line and colour keys; stacked at phone width.
- The mock passes `p3f_check.py` except the critic section's missing column descriptions (already queued for that
  section); no console errors and no horizontal overflow at 390 or 1200 px.

## 1. The colour system

| Colour | Light | Dark | Means, everywhere | Key shape | Where it appears |
|---|---|---|---|---|---|
| Green | `#00714f` | `#10a07b` | **The recorded data**: the recorded conditions (in distribution), the recording controller that made the data, the weighting test's training data | dot (situation), line (controller) | situation button, drawing frame, table group row, strip step 1 and the first segment of step 5, cost-chart line, "normal driving" points, weighting-test training curve |
| Orange | `#d56c00` | `#a95507` | **Strong current** (dynamics extrapolation); the weighting test's test that leans to strong currents | dot | button, frame, table, strip step 5, any per-situation chart, weighting-test strong test |
| Blue | `#0092c1` | `#1176a2` | **Big waves** (disturbance shift) | dot | as above |
| Pink | `#ce4c8e` | `#b1397c` | **Big starting pushes** (initial-state shift) | dot | as above |
| Red | `#9f2818` | `#dc6e76` (marks), `#ec8b84` (text) | **The stall**: the stalling world, a throttle past 80% in it, STALLED, the 80% line | line (world), tile (stall region) | world button, push curve, gauges, 80% lines, stall shading, warning text |
| Violet | `#7e53cc` | `#b06bd8` | **Push keeps rising**: the no-stall comparison world and the rising-thrust controller, which acts as if that world were true | line | world button, no-stall push curve, rising-thrust lines |
| Gold | `#c6b042` (area), ink `#735d10` | `#ac9202`, ink `#c9b23a` | **Full throttle**: the full-throttle region (water's push above 0.565) and the full-throttle commands in it, which the main test's densest-15% removal takes out | tile; hatched tile when removed | drawing's top bar, region shading, full-throttle points, 0.565 lines, strip step 2 |
| Page blue | `#2346b8` | `#758ae0` (marks; the accent itself stays `#8199ec`) | **The learned parts**: critic, actor, BCA's band; also the interface (links, buttons, focus) | line | strip steps 3 and 4, critic lines, dose lines, the weighting test's exactly weighted band |
| Ink | `#2b3330` | `#d5dbd7` | **The reference**: the stall-aware controller (the 100 of the points scale) | line | cost chart, task chart |
| Grey, dashed | `#8a928e` | `#8e9793` | No throttle | dashed line | cost chart |
| Hatch | ink-2 lines at 135° | same | **Removed from the data**, over the region it took (gold: main test; neutral: the weighting test's 80% hold) | hatched tile | strip step 2, removed rows and points |

Why these assignments:

- **Recorded conditions are the data's colour.** The pipeline starts with "the recording controller collects data in
  the recorded conditions", so the controller, the conditions and the training distribution are one thing to the
  reader: where the data come from. Green also reads as the safe, in-distribution case against three warmer or bluer
  shifts. This is why the recording controller moves from amber to green, and why the stall-aware controller moves
  from green to ink: it is the reference line, like the true value in the critic section.
- **Orange for strong current** matches the weighting test's existing "strong" orange (the test that leans to strong
  currents), so the two strong-current tests share a colour. Blue for big waves follows the water; pink for big
  starting pushes is the remaining well-separated hue.
- **Red stays the stall and nothing else.** Today the full-throttle points and the in-region top bar are drawn in the
  warning red, which mixes "full throttle" with "stall". Full throttle gets its own gold; red is kept for the stall.
- **Violet ties the no-stall world to the rising-thrust controller**: both mean "push keeps rising past 80%". The
  page already draws the rising-thrust controller's assumed push in violet.
- The four situations are **categorical, not a ramp**. When a section orders them as difficulty levels (by the harm
  the missing data cause), the order goes in the text and the table order, not in a light-to-dark colour ramp.

Pipeline step colours: step 1 green; step 2 hatched, half gold (main test: the densest 15%) and half neutral
(weighting test: the 80% hold); steps 3 and 4 page blue (step 4 drawn as a band: two lines with a tint between);
step 5 four segments, green then orange, blue, pink.

## 2. Ink and contrast (measured, WCAG)

Text token changes: `--ink-2` `#434c48` → `#2b3330` (dark `#c2c9c5` → `#d5dbd7`); `--muted` `#69726e` → `#4f5853`
(dark `#949d98` → `#a9b2ad`); `--warn` `#b33f29` → `#9f2818` (dark `#ec8b74` → `#ec8b84`). `--ink` unchanged.
Explanatory notes, captions, lead-ins, definitions and the live panel's four intro cards move from `--ink-2`/`--muted`
to `--ink`; card titles move from blue to ink. `--muted` is left for very short labels only (axis ticks, readout
names, the column name inside a phone card, chip text). The old muted grey measured 3.99:1 on the stall tint and
4.21:1 on surface-2, below the 4.5:1 floor.

Light theme, every surface the text can sit on (minimum in the last column):

| text token | surface | bg | surface-2 | water | ok-soft | accent-soft | rec tint | strong tint | waves tint | pushes tint | stall tint | rise tint | region tint | lowest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ink #1a201e | 16.5 | 15.1 | 14.2 | 14.1 | 13.8 | 13.5 | 13.9 | 14.4 | 14.4 | 14.2 | 13.3 | 14.1 | 13.7 | **13.30** |
| ink-2 #2b3330 | 13.0 | 11.8 | 11.1 | 11.0 | 10.8 | 10.6 | 10.9 | 11.3 | 11.3 | 11.2 | 10.4 | 11.0 | 10.7 | **10.43** |
| muted #4f5853 | 7.4 | 6.7 | 6.3 | 6.3 | 6.1 | 6.0 | 6.2 | 6.4 | 6.4 | 6.3 | 5.9 | 6.3 | 6.1 | **5.92** |
| warn #9f2818 | 7.5 | 6.9 | 6.4 | 6.4 | 6.3 | 6.1 | 6.3 | 6.5 | 6.5 | 6.5 | 6.1 | 6.4 | 6.2 | **6.05** |
| accent-ink #1b3a9c | 9.9 | 9.0 | 8.4 | 8.4 | 8.2 | 8.0 | 8.2 | 8.6 | 8.5 | 8.5 | 7.9 | 8.4 | 8.1 | **7.92** |
| region ink #735d10 | 6.4 | 5.8 | 5.4 | 5.4 | 5.3 | 5.2 | 5.3 | 5.5 | 5.5 | 5.5 | 5.1 | 5.4 | 5.3 | **5.11** |

Surfaces (light): surface #ffffff, bg #f4f5f1, surface-2 #eceee8, water #e6eef3, ok-soft #dcefe3, accent-soft #e3e8f7,
tints #e0eeea, #faede0, #e0f2f8, #f9eaf1, stall #f3e3e1, rise #f0eaf9, region #f0eace.

Dark theme:

| text token | surface | bg | surface-2 | water | ok-soft | accent-soft | rec tint | strong tint | waves tint | pushes tint | stall tint | rise tint | region tint | lowest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ink #e7ebe8 | 14.2 | 15.4 | 12.6 | 13.3 | 11.8 | 12.4 | 10.8 | 11.8 | 11.7 | 12.1 | 10.3 | 10.8 | 9.8 | **9.76** |
| ink-2 #d5dbd7 | 12.1 | 13.2 | 10.8 | 11.4 | 10.1 | 10.6 | 9.2 | 10.1 | 10.0 | 10.4 | 8.8 | 9.3 | 8.4 | **8.36** |
| muted #a9b2ad | 7.8 | 8.5 | 7.0 | 7.4 | 6.5 | 6.9 | 6.0 | 6.6 | 6.5 | 6.7 | 5.7 | 6.0 | 5.4 | **5.40** |
| warn #ec8b84 | 7.0 | 7.6 | 6.2 | 6.6 | 5.8 | 6.1 | 5.3 | 5.8 | 5.7 | 6.0 | 5.1 | 5.3 | 4.8 | **4.81** |
| accent-ink #aabcf3 | 9.1 | 9.9 | 8.1 | 8.6 | 7.6 | 8.0 | 6.9 | 7.6 | 7.5 | 7.8 | 6.6 | 6.9 | 6.3 | **6.26** |
| region ink #c9b23a | 8.0 | 8.8 | 7.2 | 7.6 | 6.7 | 7.0 | 6.1 | 6.7 | 6.6 | 6.9 | 5.9 | 6.1 | 5.5 | **5.54** |

Surfaces (dark): surface #191d1b, bg #111412, surface-2 #222724, water #16232a, ok-soft #163024, accent-soft #1e2640,
tints #17372e, #362817, #172f36, #37232e, stall #442f2f, rise #372d41, region #3c3915.

Against the owner's floors (7:1 body, 4.5:1 labels): ink and ink-2 pass on every surface in both themes. The warning
red is used only for short labels and terms (4.81 minimum in dark). Links (accent-ink) are 8.1:1 or more on the three
main surfaces in both themes; on a tinted table row in dark they fall to 6.3, so a link should not sit on a tint.

Marks (lines, dots, tiles) against the main surfaces, for the 3:1 mark rule:

| mark | light | L / C / H | surface, bg, surface-2 | dark | L / C / H | surface, bg, surface-2 |
|---|---|---|---|---|---|---|
| recorded | #00714f | .486 / .104 / 164 | 6.04, 5.52, 5.17 | #10a07b | .628 / .124 / 169 | 5.14, 5.59, 4.58 |
| strong current | #d56c00 | .642 / .161 / 54 | 3.52, 3.21, 3.01 | #a95507 | .542 / .134 / 54 | 3.24, 3.52, 2.88 |
| big waves | #0092c1 | .618 / .123 / 230 | 3.57, 3.26, 3.05 | #1176a2 | .534 / .108 / 235 | 3.36, 3.65, 2.99 |
| big starting pushes | #ce4c8e | .610 / .177 / 352 | 4.16, 3.80, 3.56 | #b1397c | .538 / .169 / 350 | 3.04, 3.31, 2.71 |
| stall | #9f2818 | .466 / .158 / 31 | 7.52, 6.87, 6.44 | #dc6e76 | .669 / .138 / 17 | 5.30, 5.77, 4.72 |
| push keeps rising | #7e53cc | .550 / .180 / 296 | 5.25, 4.80, 4.49 | #b06bd8 | .647 / .170 / 312 | 4.82, 5.25, 4.29 |
| full throttle | #c6b042 | .756 / .130 / 98 | 2.17, 1.98, 1.85 | #ac9202 | .664 / .136 / 96 | 5.57, 6.06, 4.96 |
| learned | #2346b8 | .446 / .184 / 266 | 7.96, 7.27, 6.80 | #758ae0 | .655 / .131 / 272 | 5.25, 5.71, 4.67 |

In dark, three situation marks fall just under 3:1 on surface-2 (table headers, gauge wells). Situation marks there
are always beside their label. The light gold is an area colour by design; every gold tile carries a 1 px outline in
its ink (5.1:1 or more).

## 3. Validator runs (dataviz `validate_palette.js`, pasted)

Sets are the colours that appear together: the four situations (page order and all pairs), the push chart (stall,
no-stall, full throttle), the live drawing (situation frame beside the stall), the cost chart's coloured controllers,
the strip in step order, the whole system (adjacent and all pairs), and the weighting test's pair. Also run on the
panel backgrounds (`#f4f5f1`, `#111412`) with the same verdicts. Scripts: `tmp/p4/design/measure/valsets.sh`.

```
### light 1. situations in page order (recorded conditions, strong current, big waves, big starting pushes), adjacent
  [PASS] Lightness band         all 4 inside L 0.43–0.77
  [PASS] Chroma floor           all 4 >= 0.1
  [PASS] CVD separation         worst adjacent #ce4c8e↔#0092c1 ΔE 9.4 (deutan) · tritan 29.4
  [PASS] Normal-vision floor    worst adjacent #ce4c8e↔#0092c1 ΔE 26.4 (normal)
  [PASS] Contrast vs surface    all 4 >= 3:1
### light 1b. situations, all pairs
  [PASS] CVD separation         worst all-pairs #ce4c8e↔#0092c1 ΔE 9.4 (deutan) · tritan 4.6
  [PASS] Normal-vision floor    worst all-pairs #ce4c8e↔#d56c00 ΔE 17.7 (normal)
  [PASS] Lightness band, Chroma floor, Contrast vs surface (all 4 >= 3:1)
### light 2. worlds and regions (stall / stalling world, no-stall world, full-throttle region), all pairs
  [PASS] CVD separation         worst all-pairs #7e53cc↔#9f2818 ΔE 25.8 (deutan) · tritan 21.4
  [PASS] Normal-vision floor    worst all-pairs #7e53cc↔#9f2818 ΔE 26.3 (normal)
  [WARN] Contrast vs surface    below 3:1 — relief required (visible labels or table view): [["#c6b042",2.17]]
### light 3. live drawing: each situation's frame colour beside the stall red, all pairs
  [WARN] CVD separation         worst all-pairs #9f2818↔#00714f ΔE 7.1 (deutan) · tritan 4.6
  [PASS] Normal-vision floor    worst all-pairs #ce4c8e↔#d56c00 ΔE 17.7 (normal)
  [PASS] Contrast vs surface    all 5 >= 3:1
### light 4. coloured controllers in the cost chart (recording = recorded-data colour, rising-thrust = violet)
  [PASS] CVD separation         worst adjacent #7e53cc↔#00714f ΔE 19.2 (deutan) · tritan 11.2
  [PASS] Normal-vision floor    worst adjacent #7e53cc↔#00714f ΔE 26.9 (normal)
### light 5. pipeline strip in step order (collect, remove, train and band, test: recorded, strong, waves, starting pushes)
  [PASS] CVD separation         worst adjacent #ce4c8e↔#0092c1 ΔE 9.4 (deutan) · tritan 5.7
  [PASS] Normal-vision floor    worst adjacent #00714f↔#2346b8 ΔE 23.3 (normal)
  [WARN] Contrast vs surface    below 3:1 — relief required (visible labels or table view): [["#c6b042",2.17]]
### light 6. the whole system in one order, adjacent
  [PASS] CVD separation         worst adjacent #d56c00↔#00714f ΔE 10.1 (protan) · tritan 19.7
  [PASS] Normal-vision floor    worst adjacent #7e53cc↔#ce4c8e ΔE 17.8 (normal)
  [WARN] Contrast vs surface    below 3:1 — relief required: [["#c6b042",2.17]]
### light 6b. the whole system, all pairs
  [WARN] CVD separation         worst all-pairs #9f2818↔#00714f ΔE 7.1 (deutan) · tritan 4.6
  [PASS] Normal-vision floor    worst all-pairs #c6b042↔#d56c00 ΔE 16.0 (normal)
  [WARN] Contrast vs surface    below 3:1 — relief required: [["#c6b042",2.17]]
### light 7. weighting test: training data and the test leaning to strong currents
  [PASS] CVD separation         worst adjacent #d56c00↔#00714f ΔE 10.1 (protan) · tritan 31.2
  [PASS] Normal-vision floor    worst adjacent #d56c00↔#00714f ΔE 26.9 (normal)
  (every light run: Lightness band PASS, Chroma floor PASS, → ALL CHECKS PASS)

### dark 1. situations in page order, adjacent
  [PASS] CVD separation         worst adjacent #b1397c↔#1176a2 ΔE 8.2 (deutan) · tritan 24.9
  [PASS] Normal-vision floor    worst adjacent #a95507↔#10a07b ΔE 23.4 (normal)
  [PASS] Contrast vs surface    all 4 >= 3:1
### dark 1b. situations, all pairs
  [PASS] CVD separation         worst all-pairs #b1397c↔#1176a2 ΔE 8.2 (deutan) · tritan 3.5
  [PASS] Normal-vision floor    worst all-pairs #1176a2↔#10a07b ΔE 15.8 (normal)
### dark 2. worlds and regions, all pairs
  [PASS] CVD separation         worst all-pairs #ac9202↔#dc6e76 ΔE 8.3 (deutan) · tritan 5.9
  [PASS] Normal-vision floor    worst all-pairs #b06bd8↔#dc6e76 ΔE 16.9 (normal)
  [PASS] Contrast vs surface    all 3 >= 3:1
### dark 3. live drawing: each situation's frame colour beside the stall red, all pairs
  [WARN] CVD separation         worst all-pairs #dc6e76↔#10a07b ΔE 6.2 (deutan) · tritan 3.5
  [PASS] Normal-vision floor    worst all-pairs #dc6e76↔#b1397c ΔE 15.2 (normal)
### dark 4. coloured controllers in the cost chart
  [PASS] CVD separation         worst adjacent #b06bd8↔#10a07b ΔE 13.8 (deutan) · tritan 17.7
  [PASS] Normal-vision floor    worst adjacent #b06bd8↔#10a07b ΔE 27.9 (normal)
### dark 5. pipeline strip in step order
  [PASS] CVD separation         worst adjacent #b1397c↔#1176a2 ΔE 8.2 (deutan) · tritan 5.2
  [PASS] Normal-vision floor    worst adjacent #ac9202↔#10a07b ΔE 15.8 (normal)
  [PASS] Contrast vs surface    all 7 >= 3:1
### dark 6. the whole system in one order, adjacent
  [PASS] CVD separation         worst adjacent #a95507↔#10a07b ΔE 12.3 (deutan) · tritan 12.6
  [PASS] Normal-vision floor    worst adjacent #b06bd8↔#b1397c ΔE 15.5 (normal)
### dark 6b. the whole system, all pairs
  [WARN] CVD separation         worst all-pairs #dc6e76↔#10a07b ΔE 6.2 (deutan) · tritan 3.5
  [PASS] Normal-vision floor    worst all-pairs #dc6e76↔#b1397c ΔE 15.2 (normal)
### dark 7. weighting test pair
  [PASS] CVD separation         worst adjacent #a95507↔#10a07b ΔE 12.3 (deutan) · tritan 28.0
  [PASS] Normal-vision floor    worst adjacent #a95507↔#10a07b ΔE 23.4 (normal)
  (every dark run: Lightness band PASS, Chroma floor PASS, Contrast PASS, → ALL CHECKS PASS)
```

Full, unabridged outputs: `tmp/p4/design/measure/val_light.txt`, `val_dark.txt`.

What was fixed on the way (first candidates failed): the strong-current orange against the stall red (normal-vision
ΔE 9.8 to 11.3) was fixed by darkening the red to `#9f2818`; green against orange collapsed for colour-blind readers
(4.1) until the green went darker and toward teal; the first gold sat 14.3 from the orange (normal-vision FAIL) and
moved yellower; the dark pink first measured 2.997:1 and the dark accent was outside the dark lightness band, so the
strip uses `--k-learn` `#758ae0` for the learned steps in dark.

The two WARNs and their relief: (1) the light gold is an area colour, always a tint or an outlined tile, always named
("full-throttle region"); (2) green against the stall red is in the 6 to 8 floor band for deuteranopia, legal only
with secondary encoding, which every use has: the stall is a red gauge plus the word STALLED, a situation is a dot
plus its name, and they are never two series in one chart.

Not validated as series (by design): the reference ink and the dashed grey (neutral identities), and the status
green of pass chips (`--ok`, unchanged). Status green sits close to the recorded-data green (ΔE 3.2); pass and fail
chips already carry the words "pass"/"fail" and never appear in a situation chart.

## 4. Running text: width, justification, hyphenation

- **Lifted caps** (to `max-width: none`): `p`, `ul`, `ol`, `.callout`, `.box`, `.sim-intro`, `.ro-note`,
  `figcaption`, `.status`, `.td-lead p/ul`, `.kv dd`, `p.worked`, `.iw-claims`, `.claims`, `.finding`, `.g-row p`,
  `.symkey`, and the critic section's `.qv-now`, `.qv-cap`, `.qv-eplist`, `.qv-findings` (outside its style block;
  drop those four if that section's owner objects). Charts and figures keep their own widths. At 1200 px a line is
  now about 105 to 115 characters.
- **Justified with hyphenation**: `:where(.wrap, dialog.wt) :where(p, li, dd, figcaption, blockquote)` gets
  `text-align: justify; hyphens: auto; hyphenate-limit-chars: 7 3 3; text-wrap: pretty` at zero specificity.
- **Left-aligned, unhyphenated** (they win over the zero-specificity rule): anything inside `td`, `th`, `caption`,
  `pre`, `code`, `.math-display`, buttons, `.opt`, `.optlbl`, readouts, `.legend`, `.symkey` (legends), `.tt` and the
  chart tooltips, `nav`, the pipeline strip, tags and chips, the dot-range chart, the walkthrough head, foot and bar,
  `.kv dt`, `.cg dt`, chart sub-captions; and the label-like paragraphs `.crumb`, `.symkey-h`, `.symkey-g`, `.sim-k`,
  `.td-k`, `.qv-k`, `.cg-h`, `.find-t`, `.f-title`, `.g-k`, `.ex-head`, `.panel-title`. Headings were never justified.
  Code words, formulas, numbers (`.nw`) and colour mentions never hyphenate.
- **Phone (≤ 480 px)**: text inside boxes (`.panel`, `.box`, `.callout`, `.check`, `details`, `.finding`, `.explain`,
  `.sim-key`, `.qv-key`, `.claims`) is left-aligned. In the mock, justified text in the 300 px explain box opened
  gaps several spaces wide around "0 = recording controller, 100 = stall-aware controller". The main column
  (358 px) stays justified; the screenshot shows gaps of at most about twice a normal space without hyphenation,
  and phone browsers that hyphenate English do better.
- **`lang="en"`** on `<div class="wrap">` and on `<dialog class="wt">` (sections move into the dialog in walkthrough
  mode; the tooltip lives on `body` and is left-aligned anyway). The publish skeleton may not set `lang`.

## 5. Symbol legends

- The legend box is the grid (`fit-content(15em) minmax(0, 1fr)`); its `dl` and every row are subgrids, so every
  definition in a legend starts at the same left edge, across groups, and the rows keep their boxes (no
  `display: contents`, so no accessibility-tree risk). Thin rules between rows. Symbols may wrap past 15em; formulas
  in a symbol do not. Browsers without subgrid fall back to a fixed 13em first column per row.
- **Phone (≤ 560 px)**: one column, the symbol on its own line, its definition under it indented 12 px.
- **Groups**: a `<p class="symkey-g">` before each `<dl>`; a worked example as a last `<p class="symkey-ex">`. For
  example, Normal driving (251ff956): "What the controller reads" / "Its fixed gains" / "Its wobble and noise" / "What
  it outputs"; Starting State (6994e47f): "The start state" / "The start laws and their spreads" / "The four test
  situations" (with situation dots); the live panel's "Reading the drawing" (511d4dd7): "The water bar at the top" /
  "The channel and the floats" / "The thrusters and their gauges" / "The chart under the drawing", built in the mock.
- **Keys in a legend**: `<dt><i class="key dot" data-sit="big_waves"></i>Big waves</dt>`; `key line` +
  `data-world` or `data-k="recording|stall_aware|rising_thrust|learn"`; `key dash` + `data-k="no_throttle"`;
  `key tile` + `data-k="region|stall"`; `key removed` + `data-k="region|hold"`; `key sitall` for "the situation you
  picked".
- Only the existing markup is needed for the plain grid: all 28 legends are `div.symkey > p.symkey-h + dl > div > dt + dd`,
  one symbol per row (checked).

## 6. Components

**Situation colours in text.** `<span class="km" data-sit="strong_current">strong current</span>`; keys follow the
page's option values `recorded`, `strong_current`, `big_waves`, `big_kicks`; worlds use `data-world="stall|no_stall"`;
the region `data-k="region"`. The mark is a background in the span's own padding, so a line break can never separate
it from its word, and the word keeps the body ink. Use it at the first mention in a section, in lead-ins, captions,
table group rows and headings of situation blocks, not on every repetition ("strong current" occurs 166 times in the
page source).

**Live panel.** No markup change: `.opt[data-group][data-value]` colour every button (a 4 px edge, a dot or line
before the name, the tint when pressed), and `#sim:has(... [aria-pressed="true"])` sets `--sit` and `--world`, which
frame the drawing (outline, so the canvas size is untouched) and mark the "Situation" and "World" lines of the
explanation box. The canvas reads the re-pointed tokens, so the cost chart already shows green, ink and violet.
Recommended script edits in `drawChannel` (the in-region cues still use the warning red):
add `regInk = tok("--k-region-ink")`; the push number `inReg ? warn : ink2` → `inReg ? regInk : ink2`; the bar fill
`inReg ? warn : waterInk` → `inReg ? regInk : waterInk`; the narrow "full-throttle region" label and the
"full-throttle command" label `warn` → `regInk`; and in the readouts
`$("roW").style.color = ... "var(--warn)"` → `"var(--k-region-ink)"`. Optional: un-stalled thrusters `acc` → `ink2`,
so blue keeps meaning "learned".

**Pipeline strip** (replaces `.glance`; markup in `scripts/p4_design_mock.py`, `STRIP`):
`nav.pipe > ol > li.pipe-step[data-step=collect|remove|train|band|test] > a[href] > .pipe-head(.pipe-n + .pipe-t) +
.pipe-d + .pipe-go`. Five equal columns with arrows from 761 px; below that, one step per row with the colour bar on
the left and downward arrows. Links: step 1 `#dataset`, 2 `#shift` ("Setting 1: which training rows are removed"), 3 `#methods`, 4 `#bca-test`,
5 `#results` (or "Setting 3: where the learned actor is tested" in `#shift`, if it gets an id).
Compact copy for the live panel: `nav.pipe.pipe-sm` with five chips, the steps the block stands for marked
`.is-here` (the panel: 1 and 5). One-line step notes under a setting: `<p class="stepnote"><span class="stepchip"
data-step="test">Step 5</span> …</p>`.

**Tables.** `<tr class="sub" data-sit="strong_current"><td><span class="km" data-sit="strong_current">Strong
current</span>` tints the group row and draws a 4 px situation edge; in phone cards the heading turns ink. A header
cell for one situation: `th[data-sit]` gets a 3 px top rule.

**Charts.** Re-pointed in CSS: `.ln-nostall`, `.pt-nostall` → violet; `.pt-ft` → gold with its ink outline;
`.pt-normal` → recorded green; `.ln-sa`, `.pt-sa` → reference ink; `.knee.k80` → stall red; `.k565` → dashed region ink.
Markup edits that must follow (inline legend swatches):

| line (today) | legend item | now | change to |
|---|---|---|---|
| 990 | push, no-stall comparison world | `var(--accent)` | `var(--k-rise)` |
| 2135 | full-throttle command (36 steps) | `var(--warn)` | `var(--k-region)` with a 1 px `var(--k-region-ink)` ring |
| 2136 | normal driving | `var(--accent)` | `var(--sit-rec)` |
| 2139, 1760 | −80% line / 80% line | dashed `var(--muted)` | dashed `var(--k-stall)`; add class `k80` to the knee lines of the charts at lines 987, 1755, 2133 (not 2945: those mark BCA's dose cap) |
| 4507 | push, stalling world | `var(--warn)` | unchanged (the stall) |

**Weighting test.** `--iw-train` → recorded green (training and calibration data are recorded data); `--iw-strong` →
strong-current orange; `--iw-mild` → the learned blue, because today that one token draws both the even test
(ξ = 0) in the density chart and the exactly weighted band in the coverage chart. Better, for the worker on that
section: split it into `--iw-weighted: var(--k-learn)` (the band) and keep the even test neutral (`var(--ink-2)`).
For the new exploratory results: unweighted band green, weighted band learned blue, plain TD3+BC's two-critic band
reference ink, each with its name beside it.

**Critic section (not editable now).** Its "removed region" (the 80% hold's ball) is drawn with `--region`, which
becomes the gold tint, while this system draws a removal as a hatch. Its legend calls it "Removed region" and says it
takes in the 80% hold, so it stays readable; a later pass on that section could draw the ball hatched. The CSS also
sets that section's card titles (`.qv-k`) and lead text to ink.

## 7. Mock-up and what the screenshots show

Built from the current page plus the CSS and the mock markup listed above. Checks on the mock: 0 console errors, 0 px
overflow at 390 and 1200; `p3f_check.py --page tmp/p4/design/mock.html` passes everything except the critic section's
19 missing column descriptions (pre-existing, queued).

- `strip-390-light.png`, `strip-1200-dark.png`: five steps, coloured bars (step 2 half gold, half neutral, both
  hatched; step 5 four segments), arrows; one row at 1200, a column at 390.
- `legend-390-light.png`, `legend-1200-dark.png`: the Transition Dynamics legend, one aligned grid, rules between rows;
  stacked at 390.
- `drawkey-*.png`: the regrouped "Reading the drawing" legend: four group headings, keys for the region, the stall,
  the frame (four-colour dot) and the four cost-chart lines; definitions align across groups.
- `para-*.png`: one justified paragraph with situation and region marks; at 390, gaps up to about twice a space.
- `sim-390-light-*.png`, `sim-1200-dark-*.png`: compact strip; step notes under each option row; coloured option
  buttons (pressed: tint and edge); the drawing framed in the situation colour; cost chart and legend in green, ink and
  violet; explanation box with situation dot and world line. The canvas still draws the in-region bar and labels red
  until the script edits above are made.
- `table-390-light-*.png`, `table-1200-dark.png`: the shift table's four group rows tinted and edged by situation.

## 8. To apply

1. Copy the page to `tmp/p4/<step>.before.html`.
2. Either append `notes/p4_design.css` as a new `<style>` after the main one (it must come after the page's own token
   blocks), or merge section 1 into the three `:root` blocks and the rest at the end of the main style. Keep the
   legacy re-points in the two dark blocks: the page's own dark blocks set them with a stronger selector.
3. `lang="en"` on `.wrap` and `dialog.wt`.
4. Markup: the strip, `data-sit` on situation rows and blocks, `.km` mentions, legend groups, the legend-swatch table
   above, `k80` classes, and the `drawChannel` edits.
5. Re-run `p3f_check.py`, the walkthrough tests, the critic section's tests, and screenshots of all 28 legends at 390
   and 1200, light and dark.
