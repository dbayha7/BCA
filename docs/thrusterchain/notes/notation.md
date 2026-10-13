# Notation inventory for comment B (starred symbols and "favourite")

Page read: `thrusterchain.html` as published in version 11 (identical to `thrusterchain.v11.html`; neither was
modified). All counts below were checked by `scripts/nb_edits.py` against the current page.

## Outcome

- **Only two starred symbols are on the page, and neither means "optimal".**
  - `a*` (3 formulas, 1 tooltip, 2 legend entries, all in Methods, "How far the actor moves") is the command the
    critic rates highest. It becomes `\hat{a}`, read "a-hat", named "the command the critic rates highest".
  - `δ*` (10 formulas, 3 tooltip titles, 1 legend entry, all in Results > "After the stop" (#beyond-the-data), Stage 1)
    is "the critic's exact one-step error", measured with one exact simulator step. It is an exact (oracle) quantity,
    not an optimal one, so it loses its star too. Proposal: `\delta^{\text{exact}}` (legend `δ<sup>exact</sup>`,
    tooltip title `δᵉˣᵃᶜᵗ`), written like the page's existing `Q^{\text{true}}` / `Qᵗʳᵘᵉ`. Not `δ^true`: the page
    already uses plain `δ(x, a) = Q(x, a) − Q^true(x, a)` for the critic's error against the exact value, so
    "true" would blur the two; "exact" is the word the page already uses for this quantity's name.
  - After the edits no star is left anywhere on the page. No star is kept, because no quantity on the page is an
    optimal one.
- **"favourite": 28 occurrences on the page** (paragraphs, a table caption, legends, 16 tooltip copies, one column
  description in a script) and 8 in the tooltip source `ta_texts.py` (plus 5 uses of "hill" there). All become "the command the critic rates
  highest" (or "â, the command the critic rates highest" where the symbol helps).
- **The "hill" picture goes with it** (11 occurrences, all in the same Methods passage): it is the metaphor the owner
  quoted ("the top of its hill"). It is replaced by the plain statement: the critic gives every command a value, and
  the curvature says how fast that value falls as a command moves away from â.
- Every edit is ready as an exact, counted replacement in `scripts/nb_edits.py` (25 text edits, 11 re-typeset
  formulas, 9 source edits). A full trial on a copy (`tmp/nb/b_applied.html`) passes: math count unchanged, all 1,633
  glyph ids unique, no "favourite", `a*`, `δ*` or "hill" left, no console errors, no horizontal overflow at 390 px.

## The renderer

Installed: `mathjax-full@3.2.2` in `C:\Users\David\dev\bca-shift-work\mathbuild` with the Windows node
(`C:\Users\David\AppData\Local\hermes\node\node.exe`, v22.22.3).

Settings matched from the page's containers: TeX input (base, ams, newcommand, noundefined), SVG output with
`fontCache: 'local'` (each `<svg>` carries its own `<defs>`), lite DOM adaptor, and the page's wrapper
`<mjx-container role="img" aria-label="TEX" class="MathJax" jax="SVG"[ display="true"]>`. The aria-label holds the TeX
with `&`, `"`, `<` escaped (`>` left as is, as on the page). Glyph ids are `<PREFIX>-<N>-TEX-<variant>-<codepoint>`;
the page has `MJX-1..289` and `MJXB-29..325`, one number per container, all unique.

**Proof of match.** `mathbuild/repro_test.js` re-renders all 325 containers on the page from their aria-label,
display flag, prefix and number: **322 are byte-identical from the label alone; the other 3 are byte-identical when
their bars `|...|` are typeset as `\vert ... \vert`** (the label keeps `|`; that is how they were made). 325 of 325.
Three examples (sha256, first 16 hex digits):

| Container | Page | Re-rendered |
|---|---|---|
| `t` (inline, MJX-29), 1,033 bytes | 6b3aee2a86c862a4 | 6b3aee2a86c862a4 |
| `\pi = \frac{m\,a + \mu\,a^*}{m + \mu}, \qquad \mu = \frac{d\,\lambda\,\kappa}{2}` (display, MJX-22), 9,510 bytes | fbe0acfd0aa1fa7c | fbe0acfd0aa1fa7c |
| `\begin{aligned} \delta^*(x, a) = {} & r(x, a) + 0.98\,\min_{j=1,2} Q_j\big(x', c(x')\big) \\ & - \min_{j=1,2} Q_j(x, a) \end{aligned}` (display, MJXB-30), 19,041 bytes | 895c1c4d7402c594 | 895c1c4d7402c594 |

The new formulas render correctly in light and dark (they draw in `currentColor`); screenshots in `page/tmp/nb/`
(`nbA*.png`, `nbD*.png`, `c-nbP*.png`, `c-nbK1*.png`).

### Usage

WSL cannot execute `node.exe` on this machine (interop is off), so rendering runs from Git Bash or PowerShell and the
apply step runs in WSL Python.

```
# 1. Git Bash: render (inline by default)
NODE=/c/Users/David/AppData/Local/hermes/node/node.exe
cd /c/Users/David/dev/bca-shift-work/mathbuild
$NODE render.js --tex "\hat{a}"                                   # prints one container
$NODE render.js --tex "x = 1" --display [--label "aria text"]
$NODE render.js jobs/<name>.json out/<name>.json                  # batch
#    job file: [{"key": "...", "tex": "...", "display": false}, ...]   (optional: "label", "prefix", "id")
#    output:   {"key": "<mjx-container ...>...</mjx-container>", ...}

# 2. WSL Python apply script (scripts/mjx.py)
import mjx
R = mjx.load('<name>')                                  # reads mathbuild/out/<name>.json
s = mjx.swap(s, old_tex, R['key'], count=1)             # replaces whole containers by aria-label (TeX), asserted count,
                                                        # display flag must match; each copy gets fresh ids MJXC-<n>
s2 = mjx.retag(R['key'], s)                             # for a new container placed by hand: fresh MJXC-<n> ids
mjx.check(s)                                            # all glyph ids unique; every <use> resolves inside its own svg
```

`mjx.find(page, tex)` lists the spans of containers with that label; `mjx.label(container)` reads one back.
Re-run `mathbuild/repro_test.js` (after `scripts/nb_labels.py` refreshes `mathbuild/page_math.json`) to recheck the
match at any time.

## A. Starred symbols in formulas (13 containers)

Rendered in `mathbuild/out/notation_b.json` from `mathbuild/jobs/notation_b.json`. Widths in ex (1 ex is about 8 px).

| Key | Where | Old TeX (aria-label) | New TeX | Copies | Width old → new |
|---|---|---|---|---|---|
| `ahat` | Methods, How far the actor moves (paragraph) | `a^*` | `\hat{a}` | 1 | 2.19 → 1.20 |
| `pi_settle` | same, display | `\pi = \frac{m\,a + \mu\,a^*}{m + \mu}, \qquad \mu = \frac{d\,\lambda\,\kappa}{2}` | `... \mu\,\hat{a} ...` | 1 | 31.02 → 30.03 |
| `critic_slope` | same, "Two shapes of critic" list | `\lambda\,\kappa\,(a^* - a)` | `\lambda\,\kappa\,(\hat{a} - a)` | 1 | 11.29 → 10.30 |
| `dexact_def` | After the stop, Stage 1, oracle definition (display) | `\begin{aligned} \delta^*(x, a) = {} & ... \end{aligned}` | `\delta^{\text{exact}}(x, a)` in place of `\delta^*(x, a)` | 1 | 40.71 → 43.60 |
| `dexact` | same, three sentences | `\delta^*` | `\delta^{\text{exact}}` | 3 | 1.99 → 4.88 |
| `W_def` | same, penalty (display) | `W(x_i, a) = \big[\delta^*(x_i, a_i) - \delta^*(x_i, a)\big]_+` | both `\delta^*` → `\delta^{\text{exact}}` | 1 | 34.19 → 39.96 |
| `abs_dexact_xa` | same, "Why the error is measured from the recorded command" | `\|\delta^*(x, a)\|` | `\|\delta^{\text{exact}}(x, a)\|` | 1 | 8.51 → 11.40 |
| `abs_diff` | same, list of reported penalties | `\|\delta^*(x_i, a) - \delta^*(x_i, a_i)\|` | both stars → exact | 1 | 20.74 → 26.52 |
| `abs_dexact_xia` | same | `\|\delta^*(x_i, a)\|` | `\|\delta^{\text{exact}}(x_i, a)\|` | 1 | 9.25 → 12.14 |
| `over_side` | same | `[-\delta^*(x_i, a)]_+` | `[-\delta^{\text{exact}}(x_i, a)]_+` | 1 | 12.44 → 15.33 |
| `under_side` | same | `[\delta^*(x_i, a) - \delta^*(x_i, a_i)]_+` | both stars → exact | 1 | 22.17 → 27.95 |

The widest display, `dexact_def`, is 43.6 ex, about 350 px: it fits the 358 px column at 390 px width (checked), and
`.math-display` scrolls sideways inside itself if a font makes it wider. The page's other bars (`|...|`) that were made
with `\vert` are not touched.

## B. Starred symbols in tooltips and legends (7 places)

| Edit | Where | Now | Proposed |
|---|---|---|---|
| B02 | tooltip on `a*` (Methods) | title "a*, the favourite command"; tip "This is the command the critic rates best at this state, the top of its hill (the critic's value drawn as a hill over the possible commands)." | title "â, the command the critic rates highest"; tip "This is the command the critic rates highest at this state, written â and read "a-hat". It is the critic's own choice, which need not be the truly best command." |
| B08 | legend under the π formula | `a*`: "It is the critic's favourite command, the top of its hill (...)" | `â`: "It is the command the critic rates highest at this state, read "a-hat". It is the critic's own choice, which need not be the truly best command." |
| B10 | same legend, share entry | "It is the share of the way toward a* for a critic that curves down past 80%, ..." | "... toward â for a critic ..." |
| B24 | three tooltip titles on `δ*` (Stage 1) | "δ*, the critic's exact one-step error" | "δᵉˣᵃᶜᵗ, the critic's exact one-step error" (tip text unchanged) |
| B25 | legend under the oracle formula | `δ*(x, a)` | `δ<sup>exact</sup>(x, a)` (dd text unchanged) |

## C. "favourite" and the hill picture

Methods, paragraph "How far the actor moves" and what follows (edits B01-B16, section-limited to #methods):

| Edit | Now | Proposed |
|---|---|---|
| B01, B03-B05 | "At one state, picture the critic as a hill over the possible commands. The top of the hill is the command the critic rates best, its *favourite command* a*, and the curvature κ says how steeply the hill falls away from the top. The cloning pull is a second hill, whose top is the recorded command a and whose steepness is set by the multiplier m. The actor settles where the two slopes balance:" | "At one state, the critic gives every possible command a value. Write the command it rates highest as â, read "a-hat". The curvature κ says how fast the critic's value falls as a command moves away from it. The cloning pull works the same way around the recorded command a: it is zero at that command and grows with the squared distance from it, at a rate set by the multiplier m. The actor settles where the two slopes balance:" |
| B06 | "Call μ the critic's pull: how hard the critic's hill draws the actor toward the favourite command, ..." | "... how hard the critic draws the actor toward the command it rates highest, ..." |
| B07 | "... toward the critic's favourite command. ... 1 means it reaches the favourite." | "... toward the command the critic rates highest. ... 1 means it reaches the command the critic rates highest." |
| B09 | legend κ: "It is the curvature of the critic's hill: 2.36, ..." | "It is the curvature of the critic's value around â: 2.36, ..." |
| B11 | "A critic that curves down past 80%, as the exact value does, keeps its hill: ..." | "... keeps its curvature there: ..." |
| B12 | "... of the way, but never past the favourite:" | "... but never past the command the critic rates highest:" |
| B13 | table caption "Share of the way from the recorded command toward the critic's favourite command, at the critic's pull μ = 0.58" | "... toward the command the critic rates highest, ..." |
| B14 | "... of the actor's move toward the critic's favourite command." | "... toward the command the critic rates highest." |
| B15 | "... the rising-thrust controller's command, which is the favourite command of a critic that believes push keeps rising." | "... which is the command rated highest by a critic whose values treat push as rising all the way to full throttle." |
| B16 | "the actor whose critic knows the stall" | "the actor whose critic values throttle past 80% correctly" |

Tooltips (data-tip text; copies counted on the page) and the column description:

| Edit | Symbol | Copies | Now | Proposed |
|---|---|---|---|---|
| B17 | κ, curvature | 2 | "...falls away from its favourite command (the command it rates best) along each throttle." | "...falls as a command moves away from â, the command the critic rates highest, along each throttle." |
| B18 | μ, the critic's pull | 9 | "...toward its favourite command (the command it rates best), on the same scale..." | "...toward â, the command the critic rates highest, on the same scale..." |
| B19 | The share of the way | 2 | "...toward the critic's favourite command, for a critic that curves down past 80%." | "...toward â, the command the critic rates highest, for a critic..." |
| B20 | μ / m, the straight critic's share | 1 | "...toward the favourite command, but never more than the whole way (a share of 1)." | "...toward â, the command the critic rates highest, but never more..." |
| B21 | min(μ / m, 1), the capped share | 1 | "...does not go past the critic's favourite command." | "...does not go past â, the command the critic rates highest." |
| B22 | column "Curving critic (bends past 80%): share μ / (m + μ)" (script dictionary) | 1 | "...toward the critic's favourite command (0 keeps the recorded command, 1 reaches the favourite)..." | "...toward â, the command the critic rates highest (0 keeps the recorded command, 1 reaches â)..." |

Results (section-limited to #results):

| Edit | Now | Proposed |
|---|---|---|
| B23 | "Plain TD3+BC moves only 0.367 to 0.580 of the way toward its critic's favourite command." | "... toward the command its critic rates highest." |

Tooltip text sources: `ta_texts.py` holds the same wording (tip-176, tip-177, tip-181, tip-182, tip-184, tip-185,
the tip-179 legend lines for â and κ, and the "How far the actor goes" column text). Edits S01-S09 in
`nb_edits.py` change them to the proposed wording, so a later `ta_apply.py` run agrees with the page.

## D. Other informal or figurative words found (not in nb_edits.py)

Found by `scripts/nb_words.py`; for the agents rewriting those sections.

| Word | Count and where | Proposal |
|---|---|---|
| climbs | 9: live panel controller descriptions (2), Results, Stage 1 (heading, text and tooltips; 7) | "is trained to raise" or "the value the actor is trained to maximise" (Stage 1 heading: "can caution reach the harm through the critic's value, which the actor is trained to maximise?") |
| nudge | 2: Transition Dynamics paragraph and its tooltip ("the floats nudge each other weakly") | part of comment C's rewrite: say what the coupling matrix does, with its measured size |
| wobble / wobbles | 14: live panel (3), recording controller (7), distribution shift (1), Results (1), script strings (2) | "random variation" (the recording controller's exploration noise, standard deviation 0.3) |
| trusts | 5: header, Methods (TD3's "trusts the lower of their two estimates"), Results (2), one script string | "relies on" or "is trained against"; for TD3, "uses the lower of their two estimates" |
| knows | 3 besides B16: The task ("it knows how hard the wave pushes"), Rewards ("a controller that knows to stop at 80%"), recording controller ("Everything it knows comes from a log") | "reads", "a controller that stops at 80%", "everything it learns comes from a log" |
| toy | 1: Version History | removed with that section (comment F) |

"The stalling world" (105 uses) is the page's defined name for the world with the stall; it was left alone. "Kind of"
hits mean "type of" and were left alone.

## How to apply

```
wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb_edits.py'                 # check only
wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb_edits.py --copy'          # tmp/nb/b_applied.html
wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb_edits.py --write'         # the working page
wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb_edits.py --write-sources' # ta_texts.py
```

Nothing is written unless every old string occurs exactly as counted. B23-B25 and the ten δ formulas sit in Results,
which comment D rewrites: either run `nb_edits.py --write` before that rewrite, or have the rewrite use `â` and
`δ^exact` directly (render new formulas with `render.js`); a stale edit then fails its count instead of misapplying.
