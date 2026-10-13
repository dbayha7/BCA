# ThrusterChain page: structure plan for comments A, E and F (pass 3)

Surveyed read-only on the working copy, which is still byte-identical to the backup
(`thrusterchain.v11.html`, md5 `79f00557ba8803625b9f36bc497f7cc6`; every edit script must assert that the backup's md5
is unchanged). Survey scripts, all read-only: `scripts/st3_survey.py` (blocks, positional words, text, find),
`st3_facts.py` (where a Version History fact also appears), `st3_firstuse.py` (first use of a term, old vs new order),
`st3_ctx.py`, `st3_linkctx.py`, `st3_marks.py` (boundary markers and their counts). Line numbers below are those of v11
and will move; every cut is anchored on a marker whose count was checked (all count 1, LF line endings).
`st3_dryrun.py` applies every structural edit below in memory, asserting each marker, and writes nothing. On v11 it
passes. It prints the new `<main>` order and the rail, and both match section A (the page's md5 was unchanged afterwards).

## Takeaway

- **A is a pure block move.** The six sections already sit in the owner's order (Action Space, Observation Space,
  Starting State, Transition Dynamics, Rewards, Episode End). Swap the block "live panel + The task" with the block of
  the six. Nothing else moves. The walkthrough has **no step list to edit**: it reads the DOM order when it opens.
- **The move creates eight first-use gaps.** Some terms used in the six were defined only in the live panel or The task
  ("stalling world", "full-throttle region", "evaluation bank", "textbook" command, "rising-thrust controller" and
  others). Each needs a short clause where it now first appears. The wording comes from definitions already on the page.
- **F:** there are two links to `#version-history` (the Transition Dynamics prose and the contents rail). Three facts
  live only in Version History and are needed: why this world replaced the earlier one (two design reasons), where its
  float dynamics come from, and the weighting test's package name. The rest is covered elsewhere or not needed.
- **E:** one line of markup to delete. The caption keeps the word "Assumed".

---

## A. New order of `<main>`

Current top-level children (v11): `header#top`, `figure.panel#sim`, `section#task`, the six, `#description`, ...

| # | Block | Heading (= walkthrough title) | v11 char span |
|---|---|---|---|
| 1 | `header#top` (walkthrough step = `#glance`) | At a glance | 48617-65700 |
| 2 | `section#action-space` | Action Space | 250511-290261 |
| 3 | `section#observation-space` | Observation Space | 290261-339357 |
| 4 | `section#starting-state` | Starting State | 339357-408900 |
| 5 | `section#dynamics` | Transition Dynamics | 408900-512145 |
| 6 | `section#rewards` | Rewards | 512145-580866 |
| 7 | `section#episode-end` | Episode End | 580866-611405 |
| 8 | `figure.panel#sim` | Watch one episode | 65700-102518 |
| 9 | `section#task` | The task | 102518-250511 |
| 10 | `section#description` | Summary of the environment | unchanged from here on |
| 11 | `section#recording-controller` | The recording controller | |
| 12 | `section#reference-scores` | Reference Scores | |
| 13 | `section#dataset` | The recorded data | |
| 14 | `section#shift` | The distribution shift | |
| (14a) | `section#critic-in-action`, to be inserted later by the other session, before `#methods` | (not created here) | |
| 15 | `section#methods` | Methods | |
| 16 | `section#harm-checks` | The harm checks | |
| 17 | `section#bca-test` | The BCA test | |
| 18 | `section#results` (holds `div#beyond-the-data`) | Results | |
| 19 | `section#weighting-test` | The weighting test | |
| 20 | `section#arguments` | Arguments | |
| ~~21~~ | ~~`section#version-history`~~ | deleted (F) | 1643791-1647328 |
| 21 | `section#code-names` | Names used in the code | |

**Where "Summary of the environment" goes.** It stays where it is relative to everything except the six, so it now
follows The task. The owner's comment names only the six, and the instruction puts them directly before the live panel
and The task. I considered putting the Summary right after Episode End, so that the specification block ends with its
recap. I rejected that because it would put a section between the six and "Watch one episode". If the owner prefers
the recap there, it is the same move with the cut at `#sim` instead of `#description`.

### The move (exact markers, all count 1 in v11)

```python
s = open(PAGE, encoding="utf-8").read()
a = s.index('\n    <figure class="panel" id="sim"') + 1      # start of the live panel (just after "  </header>\n\n")
b = s.index('\n  <section id="action-space">') + 1           # start of the six
c = s.index('\n  <section id="description">') + 1            # start of the Summary
assert s.count('\n    <figure class="panel" id="sim"') == s.count('\n  <section id="action-space">') == \
       s.count('\n  <section id="description">') == 1 and a < b < c
assert s[:a].endswith("  </header>\n\n")
simtask, six = s[a:b], s[b:c]
assert simtask.endswith("  </section>\n\n") and six.endswith("  </section>\n\n")
assert six.count('<section id="') == 6 and simtask.count('<section id="') == 1
s = s[:a] + six + simtask + s[c:]
```

Checks after the move: the order of `main > *` ids is exactly the table above (without 14a). Each moved block's length
and md5 are unchanged. The page's total length is unchanged by this step.

### Contents rail (`nav.toc`, v11 lines 3776-3801)

Use two targeted replacements so the rail survives other sessions' edits (for example a later critic-in-action entry):

1. Remove `  <a href="#sim">Watch one episode</a>\n  <a href="#task">The task</a>\n` (count 1).
2. Replace `  <a href="#episode-end">Episode End</a>\n` (count 1) with
   `  <a href="#episode-end">Episode End</a>\n  <a href="#sim">Watch one episode</a>\n  <a href="#task">The task</a>\n`.
3. (F) Remove `  <a href="#version-history">Version History</a>\n` (count 1).

The resulting rail:

```
At a glance
Action Space
Observation Space
Starting State
Transition Dynamics
Rewards
Episode End
Watch one episode
The task
Summary
Recording controller
Reference Scores
Recorded data
Distribution shift
   (slot for the critic-in-action entry, added later by the other session)
Methods
Harm checks
BCA test
Results
  After the stop   (class="sub", unchanged)
Weighting test
Arguments
Names in the code
```

### The at-a-glance sentence that announces the order (v11 line 780, count 1)

Old (inside the "Task" row of `div.glance`, right after the stall term):

```
pushes less but still costs more. The live panel "<a href="#sim">Watch one episode</a>" comes next, and right after it the section "<a href="#task">The task</a>" explains the whole task.
```

New:

```
pushes less but still costs more. The next six sections specify the environment: what a controller sets (<a href="#action-space">Action Space</a>), what it reads (<a href="#observation-space">Observation Space</a>), how an episode starts (<a href="#starting-state">Starting State</a>), how one step moves the floats (<a href="#dynamics">Transition Dynamics</a>), what each step costs (<a href="#rewards">Rewards</a>) and when an episode ends (<a href="#episode-end">Episode End</a>). After them, the live panel "<a href="#sim">Watch one episode</a>" replays real episodes, and the section "<a href="#task">The task</a>" explains what makes the task hard.
```

Do not say that this is the order of a Gymnasium page. Gymnasium's own pages put Rewards before Starting State.

Also update the CSS comment at v11 lines 6-8 (invisible, but it describes the layout):
`Reading order: the summary and the live panel, the task, then the sections of a Gymnasium environment page (Action Space first), the methods and the results.`
becomes
`Reading order: the at-a-glance summary, the six specification sections (Action Space first), the live panel, the task, the summary of the environment, the methods and the results.`

### First-use gaps the move creates, and the clause for each

`st3_firstuse.py` compares where each tooltip title and key phrase first appears in the old and new orders. All other
cross-links between the moved blocks are worded without direction words (checked with `st3_linkctx.py`). The only
"above" between them is "counted as in the live panel above" in The task, and it stays true. Each clause below reuses
a definition already on the page; the source is given. No new numbers.

| # | Where (v11 line) | Term now used before it is defined | Proposed clause | Source on the page |
|---|---|---|---|---|
| 1 | Action Space, after "past 80% the propeller stalls (below)." (1324) | stalling world, no-stall comparison world (the table's "Stalls" column and the push formula use them) | "The environment has two worlds that differ only here: in the <i>stalling world</i>, the default and the one the data were recorded in, the propeller stalls; in the <i>no-stall comparison world</i>, push keeps rising up to 100% throttle, which shows what the same learning costs when the throttles missing from the data hide nothing." | Summary, "Worlds (2)" (1687-1689) |
| 2 | Observation Space, end of the "Wave (2 numbers)" item (1384-1388) | amplitude, phase (table rows 11-12) | "The wave's amplitude, the largest push it adds or takes away, and its phase, where it is in its cycle, are drawn at the start (Starting State)." | The task's opening; φ tooltip |
| 3 | Starting State, "by the rule of the chosen situation." (1437) | situation and the names strong current, big waves, big starting pushes | "...by the rule of the chosen <i>situation</i>: the recorded conditions, the rule the training data come from, or one of three shifted situations (strong current, big waves and big starting pushes), each of which changes one setting beyond anything recorded." | At a glance; Summary, "Situations (4)" |
| 4 | Starting State, "a starting state's float cost averages 2.5" (1444) | float cost | "...float cost (the floats' squared distance from rest, the main part of each step's cost; see Rewards) averages 2.5..." | Rewards |
| 5 | Starting State, "The textbook hold against the current alone" (1450) | textbook command | "...The textbook hold against the current alone (the textbook command makes each throttle a fixed weighted sum of the 13 observed numbers, the best such rule if push always equalled throttle)..." | Stall-aware tooltip in At a glance |
| 6 | Starting State, "each start of the evaluation bank" (1455) | evaluation bank (defined only at the section's end) | "...each start of the evaluation bank (the fixed 1,000 starts that every reported return uses, described at the end of this section)..." | Live panel; the end of Starting State |
| 7 | Starting State, "swings the water's push further every cycle: further into the full-throttle region near each crest" (1456-1457) | water's push, full-throttle region | "...swings the water's push (the current plus the push the wave adds now, the same on every float) further every cycle: further into the full-throttle region (the states where the water's push is above 0.565, where the recording controller puts thruster 1 at full throttle in half of the states, by a coin flip) near each crest..." | Live panel readout; full-throttle-region tooltip |
| 8 | Rewards, "Points." paragraph: "(in the no-stall comparison world, the rising-thrust controller's)", first occurrence in running text, not the legend | rising-thrust controller | "...(in the no-stall comparison world, the rising-thrust controller's: it gives the same textbook command but allows up to 100% throttle, as if push kept rising)..." | Rising-thrust tooltip in the live panel |

Optional: the live panel's first sentence could start with "The six sections above specify the environment; here it
runs." Its own "What you are looking at" text now repeats definitions the reader has already seen, which does no harm.

### Walkthrough (Full walkthrough mode)

- **Where it is defined.** It is the last `<script>` block (v11 line 4643 on, the IIFE that starts "Full walkthrough
  mode: the page's sections one at a time"). `collect()` (v11 lines 4665-4673) builds the step list each time the
  pop-out opens. The list is `#glance` (title from `data-wt-title="At a glance"`, v11 line 768), then every child of
  `<main>` matching `section[id], figure.panel[id]` in DOM order. Each title is the block's `h2`, or `.panel-title` for
  the live panel. **No list needs editing:** the block move reorders the steps and deleting Version History removes
  its step.
- **New step list (21 steps; 22 once the critic-in-action section exists):** 1 At a glance, 2 Action Space,
  3 Observation Space, 4 Starting State, 5 Transition Dynamics, 6 Rewards, 7 Episode End, 8 Watch one episode,
  9 The task, 10 Summary of the environment, 11 The recording controller, 12 Reference Scores, 13 The recorded data,
  14 The distribution shift, 15 Methods, 16 The harm checks, 17 The BCA test, 18 Results, 19 The weighting test,
  20 Arguments, 21 Names used in the code.
- **Static text:** the dialog's `<p class="wt-count" id="wt-count">Section 1 of 22</p>` (v11 line 3806, count 1) is
  overwritten by `label()` when the pop-out opens. Change it to `Section 1 of 21` for consistency, or leave it.
- **Nothing else depends on the order.** The live panel's autoplay uses an IntersectionObserver, so it starts when the
  reader scrolls to the panel, not on load. Pausing on leaving the panel is keyed on `n.id === "sim"`. Column
  tooltips (`COLS[sec.id]`) and Tab stops are scoped per section. Legend ids `symkey-N` are numbered at run time and
  nothing static refers to them. The CSS has no sibling or order selectors (`main` spaces its children with
  `gap: 56px`).
- **The browser test `scripts/e1_wt_test.py` hardcodes the old order and must be updated (or copied, for example as
  `st3_wt_test.py`):**
  - Compute `N = pg.evaluate("1 + document.querySelectorAll('main > section[id], main > figure.panel[id]').length")`
    (21 now) and use it everywhere `22` appears: `"Section 1 of {N}"`, `range(N)`, `if i < N - 1`,
    `"Section {N-1} of {N}"` (after Back) and `"Section {N-2} of {N}"` (after ArrowLeft and on reopening).
  - `expect` becomes `["At a glance", "Action Space", "Observation Space", "Starting State", "Transition Dynamics",
    "Rewards", "Episode End", "Watch one episode", "The task", "Summary of the environment",
    "The recording controller", "Reference Scores"]`.
  - The check "leaving the panel pauses it" is tied to `i == 1`. Tie it to the step after the one whose
    `info["id"] == "sim"` (now `i == 7`).
  - The `for _ in range(17)` ArrowLeft loop that walks back to "The task" becomes
    `range((N - 2) - (titles.index("The task") + 1))`, which is 10 now.
  - The two link checks (`#starting-state` clicked from inside The task, then `#task-situations` from inside Starting
    State) still hold.

---

## F. Delete Version History

### Section deletion (markers count 1)

```python
v0 = s.index('\n  <section id="version-history">') + 1
v1 = s.index('\n  <section id="code-names">') + 1
blk = s[v0:v1]
assert blk.count('<section id="') == 1 and blk.endswith("  </section>\n\n") and "<h2>Version History</h2>" in blk
s = s[:v0] + s[v1:]
```

Afterwards, `#version-history` and the text "Version History" must occur 0 times in the page. The COLS dictionaries,
the tooltip texts (`ta_texts.py`, `ta_code.py`) and the scripts never mention it (checked).

### Every link to `#version-history` (two in total) and its rewording

| # | Where | Old | New |
|---|---|---|---|
| 1 | Transition Dynamics, the `A` item of the "where:" list (v11 lines 1538-1540, count 1) | `These are the same\n        float dynamics as in the earlier world this one replaces (<a href="#version-history">Version History</a>).</li>` | `These float dynamics, coupling included,\n        are reused unchanged from the project's earlier test world, which had no stall (the box at the end of the section "<a href="#description">Summary of the environment</a>" says why that world was replaced).</li>` |
| 2 | Contents rail (v11 line 3799) | `  <a href="#version-history">Version History</a>\n` | removed (rail step 3 above) |

The code confirms the provenance in row 1: `experiments/shift/thruster_chain.py` imports
`experiments.signal.gapped_lq` and takes `A = gapped_lq.transition_matrix(STAGE0_SPEC)` (its docstring, lines 8-9:
"radii 0.98-0.96, angles 0.10/0.20/0.35/0.55/0.80 rad, coupling seed 20261011"). SHIFT_ENV.md line 87 says the same.
**Coordinate with comment C:** C's rewrite of this paragraph may replace the sentence. If it does, it must keep the
provenance and the link to `#description`, and never link to `#version-history`.

### Facts in Version History, checked against the rest of the page (`st3_facts.py`)

| Fact (Version History item) | Elsewhere on the page? | Action |
|---|---|---|
| Weighting test: same environment and physics; start rule with the current from 0.1 to 1.1, leaning by a known amount; new recording controller whose common command in hard water is the 80% hold | Yes, in full in `#weighting-test` (λ, 0.1 to 1.1, hard water, 80% hold) | None |
| Weighting test: package `experiments/shift_iw` | **No** (0 hits) | **Move** to "Names used in the code" (row below) |
| ThrusterChain v0: eight ids, pre-registered, "stopped before BCA", harm-check outcomes | Yes: Arguments (eight ids), header status, Harm checks, Results | None. Its "caution of BCA's kind cannot reach it" is deleted with the section, which matches D's narrowing. |
| **Why ThrusterChain replaced the earlier world, reason 1:** with linear dynamics and a quadratic cost, every true value is a quadratic formula of state and command, so any patch of data fixes the critic everywhere and missing data hide nothing; here the stall bends the value at 80%, and the removed command is the log's only record past 80% | **No** (0 hits for "quadratic formula", "fixes the critic") | **Move** to the Summary's callout (text below) |
| **Reason 2:** the removed command must sit at an edge of what is possible, not in the middle, so caution toward the data cannot pull the controller into the removed commands | **No** (0 hits) | **Move** to the same callout. Results (D) may also cite it when explaining why the pull toward the recorded commands could not win back the loss. |
| Earlier world ("Gapped LQ"): same five floats and float dynamics; rejected twice (seeds 0-4; then fresh seeds 5-9 under a pre-registered amendment with marks loosened after seeing 0-4), because the shortfall (and then the critics' error too) did not sit mainly in the thinly recorded contexts; BCA never run in it | **No** (the page's "gapped critic" means something else) | **Move, compressed**, into the same callout. Its name goes to "Names used in the code". Drop the "five observed contexts" detail: no other part of the page uses contexts, and it would add a term. |
| The three-number toy world, retired (3 state numbers, 2 throttles, discount 0.95, 50 steps; errors written by hand; no thinly recorded data; perfect caution added only −1.6 to +0.04 points) | No | **Drop.** No other part of the page needs it. If the owner wants it kept, it is one clause at the end of the callout (text below). |
| The text spliced into the v0 item: "The user did not accept the stop ... a test of a new version of BCA ... pre-registered, committed ... approved by the user, and no BCA run has started yet (After the stop). The environment itself is unchanged." | Yes in `#beyond-the-data` (results line 2912: "The user did not accept the stop"; line 3297: approved, committed 0af5e0a, not run yet). **"The environment itself is unchanged" is not.** | Results (D) should state it near the top of "After the stop" ("the environment itself did not change"). Note that the v0 item is corrupted: this paragraph sits in the middle of the sentence "...the removed command is the [spliced text] log's only record past 80%." The text proposed below rejoins it. |

### Destination text

**(1) Summary of the environment.** Add a second paragraph to the existing `div.callout` (v11 lines 1715-1722),
after the paragraph that starts "Why the stall makes this a test of distribution shift". Insert it before the
callout's closing `</div>`. Anchor: `(<a href="#task-strong">Why strong current is the hard case</a>).</p>\n    </div>`
(count 1).

> <strong>Why this environment, and not the earlier one.</strong> ThrusterChain replaces the project's earlier test
> world, which had the same five floats and the same float dynamics but no stall: its dynamics were linear and its cost
> quadratic, meaning built from squares and products of the numbers, like the float cost. Its harm checks rejected it
> twice, on seeds 0–4 and again on fresh seeds 5–9 under a pre-registered amendment whose pass marks were loosened
> after seeing seeds 0–4. Both times the learned actors' shortfall did not sit mainly where the data were thin, so
> there was no harm caused by the critic for BCA to find, and BCA was never run in it. The design work found two
> reasons. First, in a world with linear dynamics and a quadratic cost every true value is itself a quadratic formula
> of the state and the command, so any patch of data fixes the critic everywhere and missing data hide nothing. Here
> the stall bends the true value at 80%, and the removed command is the log's only record past 80%. Second, the
> removed command must sit at an edge of what is possible, not in the middle, so that caution toward the recorded
> commands cannot pull the controller into the commands that were removed. Full throttle, the end of the throttle
> range, is such an edge.

The last sentence is the only one not copied from Version History. It follows from the throttle range of −1 to 1
stated in Action Space; drop it if a strictly verbatim move is wanted. Optional toy-world clause, if kept: "An even
earlier three-number world was retired: its critic errors were written by hand instead of learned, it had no thinly
recorded data, and even caution placed perfectly added only −1.6 to +0.04 points."

**(2) Names used in the code.** Add two rows at the end of the table. The anchor
`</tbody></table></div>\n  </section>\n</main>` has count 1. Insert before it:

```html
<tr><td>the floats' own motion A, coupling included, reused from the project's earlier linear-quadratic world</td><td><code>experiments/<wbr>signal/<wbr>gapped_lq.<wbr>transition_matrix(STAGE0_SPEC)</code>; that world is called "Gapped LQ" in the project's records (also the <code>glq</code> in <code>glq_match.py</code>)</td></tr>
<tr><td>the weighting test</td><td><code>experiments/<wbr>shift_iw</code></td></tr>
```

The section's intro ("This table gives the name of each in `experiments/shift`, the repository and the recorded
files.") still fits, since the repository covers both packages. The column tooltips for this table (`COLS["code-names"]`)
are per column, so the new rows need no tooltip text.

---

## E. Remove the "illustration, not a result" tag

Location: `#weighting-test`, subsection "Why weighting should matter" (`h3`, v11 line 3532), first child of the
`div.panel` that holds the coverage chart `#iw-c-cov`.

Exact markup (v11 line 3539, count 1, six-space indent):

```html
      <p><span class="tag warn">illustration, not a result</span></p>
```

Edit (count 1):
`    <div class="panel">\n      <p><span class="tag warn">illustration, not a result</span></p>\n      <div class="iw-chart" id="iw-c-cov">`
becomes
`    <div class="panel">\n      <div class="iw-chart" id="iw-c-cov">`

The chart's caption stays and still states that the error model is assumed (v11 line 3542):
`<p class="note">Assumed error size 0.05 + 0.25·<i>c</i>, normally distributed; exact weights and an unlimited calibration set</p>`.
The style rules want complete sentences, and the reader needs to be told the chart is computed rather than measured.
Suggested wording: `<p class="note">This chart is computed from an assumed error, not measured: the critic's error is
taken to be normally distributed with typical size 0.05 + 0.25·<i>c</i>, with exact weights and an unlimited calibration
set.</p>`

The word "illustration" also appears in two places that are not the tag. Neither is required by E; both are optional
wording fixes:
- the legend entry (v11 line 3550) `An assumption made for this illustration only:` → `An assumption made for this chart only:`;
- the chart's `aria-label` (v11 line 3544, count 1) `Illustrative coverage of unweighted and weighted bands against the test lean λ`
  → `Coverage of unweighted and weighted bands against the test lean λ, computed from an assumed error size`.

A JavaScript comment at v11 line 4587 ("illustration only: ...") is not shown to readers; leave it.

Now that version 5 has run, the note under the chart says "The real test uses a finite calibration set (so weighted
coverage will scatter around 90%)". The future tense is stale. The weighting-test agent should turn it into the
measured result, or point to it.

---

## Notes for the other agents (found during the survey, not acted on)

- **D (Results).** The broad "caution cannot help" claim also appears outside Results: the header's status line
  ("The outcome: ... caution of the kind BCA applies cannot win back that lost return, in any situation", v11 line 801),
  Harm checks (line 2530), the weighting test (lines 3569, 3594) and Results (lines 2800-2805, 2908, 3138). The
  Version History copies (lines 3693, 3700) go away with F. Search `caution[^.<]{0,120}(cannot|can.t|could not|never)`.
- **B.** "favourite" is also in the μ tooltip in Methods ("draws the actor toward its favourite command", v11 line
  2239).
- **Weighting test.** The section still opens with "being built / Specification, first draft, under review" and a
  progress strip with "Specification review: in progress". All of it is now stale after version 5.
- **Order of application.** Every edit above anchors on markers that other passes do not touch, so the edits commute
  with B, C and D. Apply the dynamics-link rewording before or together with the Version History deletion, so the page
  is never left with a link to a missing section. After all passes, check: 0 occurrences of `version-history`; the
  `main > *` id order matches the table in A; the rail lists the same ids in the same order (plus `beyond-the-data`);
  the walkthrough test passes with N computed from the page; and screenshots at phone width (390) and desktop, in light
  and dark, show the moved top of the page.
